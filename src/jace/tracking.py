"""Keep each training run's record as plain files, and be the only code that writes them.

There is no tracking service. A run is a folder that any program can read,
and the layout below is a public contract: the frontend reads these files
directly, so a change to a name or a field here is a breaking change.

    <root>/<run id>/
      identity.json        what the run is, written once
      state.jsonl          one line per state change
      metrics.jsonl        one line per update per model
      evaluations.jsonl    one line per evaluation
      checkpoints/
        <model>-<update>.pt  a state dict, update zero padded to 8 digits
        index.jsonl          one line per checkpoint
      lock                 held with flock by the process that is writing

The run id is the creation time in UTC to the second, a dash, and the first
8 hex digits of the SHA-256 of the identity, for example
`20261111T153045Z-1a2b3c4d`. Ids therefore sort by creation time as text.
Creating two runs with the same identity in the same second raises
`RunExistsError` instead of sharing a folder.

Every `.jsonl` file holds one JSON object per line, each written whole with
a single append, and each file exists from creation. No value is ever NaN
or infinite: logging one raises `ValueError`. The fields are:

    identity.json      format (1), run_id, created, config, seed, jace_commit,
                       manabot_commit, inputs (a list of checkpoint objects)
    state.jsonl        state, time, and error when the state is "failed"
    metrics.jsonl      model, update, metrics (an object of name to number)
    evaluations.jsonl  checkpoint (an object), opponent, deals, rate, low, high
    index.jsonl        model, update, sha256
    checkpoint object  run_id, model, update, sha256

Times are ISO 8601 in UTC. `sha256` is the hex digest of the `.pt` file. A
`.pt` file with no line in `index.jsonl` was cut off while being saved.

A run moves created -> running -> completed | failed | interrupted, and
interrupted -> running. The last line of `state.jsonl` is the state, with
one exception: a killed process cannot record its own end, so "running"
with nobody holding the lock means interrupted. `read_state` and `writing`
resolve that and append the "interrupted" line. A reader that does not call
them can test the lock itself.

The lock is an OS lock that dies with its process, so there is never a
stale lock to clean up. A child forked inside a `writing` block inherits
it, and the run reads as running until that child exits too. Nothing is
fsynced: the record survives a killed process but not a power loss.

Every function that writes a time takes a `clock`, a callable that returns
the current time as a timezone-aware `datetime`. This module never reads
the system clock or runs git: the caller supplies both commits.
"""

import enum
import fcntl
import hashlib
import json
import os
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch
from torch import nn

type ModelName = Literal["embedding", "drafter", "deck_builder", "deck_score", "guess", "player"]
type Clock = Callable[[], datetime]

# The version of the file layout, written to `identity.json` so a reader can tell layouts apart.
_FORMAT = 1
# Checkpoint file names pad the update to this many digits, so they sort by update as text.
_UPDATE_DIGITS = 8
# Hex digits of the identity's SHA-256 kept in a run id: enough to tell apart runs created in the same second.
_ID_DIGEST_DIGITS = 8
# Permissions of a record file this module creates: its owner reads and writes, everyone else reads.
_RECORD_FILE_MODE = 0o644


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    run_id: str
    model: ModelName
    update: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Identity:
    config: Mapping[str, object]
    seed: int
    jace_commit: str
    manabot_commit: str
    inputs: tuple[CheckpointRef, ...]  # checkpoints from other runs that this run starts from


class RunState(enum.Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class Running:
    """Proof that this process holds the run's writer lock.

    Only `writing` makes one, and it is valid until that block ends. Every
    function that writes to a run takes it, so a write outside a `writing`
    block does not type-check. `lock` is the file descriptor of the held
    lock file.
    """

    folder: Path
    lock: int


class RunExistsError(FileExistsError):
    def __init__(self, folder: Path) -> None:
        super().__init__(f"run folder {folder} already exists: same identity created in the same second")


class RunLockedError(RuntimeError):
    def __init__(self, folder: Path) -> None:
        super().__init__(f"another live process is writing run {folder.name}")


class RunNotResumableError(RuntimeError):
    def __init__(self, folder: Path, *, state: RunState) -> None:
        super().__init__(f"run {folder.name} is {state.value}; only a created or interrupted run can be written")


class CheckpointExistsError(FileExistsError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"checkpoint {path} already exists; a checkpoint is never overwritten")


class CheckpointMismatchError(ValueError):
    def __init__(self, path: Path, *, expected: str, actual: str) -> None:
        super().__init__(f"checkpoint {path} has sha256 {actual}, but the reference says {expected}")


def create_run(root: Path, identity: Identity, *, clock: Clock) -> Path:
    """Make a new run folder under `root` in the state CREATED and return it.

    The folder's name is the run id, described in the module docstring.
    `root` is created when it is missing. Raises `TypeError` before anything
    is written when the config holds a value JSON cannot represent, and
    `RunExistsError` when the folder already exists.
    """
    created = clock().astimezone(UTC)
    described = _described(identity)
    digest = hashlib.sha256(json.dumps(described, sort_keys=True, allow_nan=False).encode()).hexdigest()
    run_id = f"{created:%Y%m%dT%H%M%SZ}-{digest[:_ID_DIGEST_DIGITS]}"
    record = {"format": _FORMAT, "run_id": run_id, "created": created.isoformat(), **described}
    identity_text = json.dumps(record, indent=2, allow_nan=False) + "\n"

    folder = root / run_id
    root.mkdir(parents=True, exist_ok=True)
    try:
        folder.mkdir()
    except FileExistsError:
        raise RunExistsError(folder) from None
    (folder / "checkpoints").mkdir()
    (folder / "identity.json").write_text(identity_text)
    for name in ("lock", "metrics.jsonl", "evaluations.jsonl", "checkpoints/index.jsonl"):
        (folder / name).touch()
    _record_state(folder, RunState.CREATED, clock=clock)
    return folder


def read_state(folder: Path, *, clock: Clock) -> RunState:
    """Return the run's state, recording INTERRUPTED first when its writer died.

    A run whose last recorded state is RUNNING is still running only while a
    live process holds its lock. When nobody does, this appends INTERRUPTED
    to `state.jsonl` and returns that. It is safe to call while another
    process, or this one, is writing the run.
    """
    state = _last_recorded_state(folder)
    if state is not RunState.RUNNING:
        return state
    lock = _try_lock(folder)
    if lock is None:
        return RunState.RUNNING
    try:
        return _settled_state(folder, clock=clock)
    finally:
        os.close(lock)


@contextmanager
def writing(folder: Path, *, clock: Clock) -> Generator[Running]:
    """Take the writer lock and enter RUNNING, from CREATED or INTERRUPTED only.

    Leaving the block records how it ended: COMPLETED on a normal exit,
    INTERRUPTED on `KeyboardInterrupt`, and FAILED with the error's type and
    message on any other exception. The exception is re-raised and the lock
    is released in every case.

    Raises `RunLockedError` when another live process (or another `writing`
    block in this one) holds the run, and `RunNotResumableError` when the
    run is completed or failed. A run left RUNNING by a dead writer counts
    as interrupted, so it can be resumed without calling `read_state` first.
    """
    lock = _try_lock(folder)
    if lock is None:
        raise RunLockedError(folder)
    try:
        _enter_running(folder, clock=clock)
        try:
            yield Running(folder=folder, lock=lock)
        except KeyboardInterrupt:
            _record_state(folder, RunState.INTERRUPTED, clock=clock)
            raise
        except BaseException as error:
            _record_state(folder, RunState.FAILED, clock=clock, error=f"{type(error).__name__}: {error}")
            raise
        _record_state(folder, RunState.COMPLETED, clock=clock)
    finally:
        os.close(lock)


def log_update(run: Running, *, model: ModelName, update: int, metrics: Mapping[str, float]) -> None:
    record = {"model": model, "update": update, "metrics": dict(metrics)}
    _append(run.folder / "metrics.jsonl", record)


def save_checkpoint(run: Running, *, model: ModelName, update: int, module: nn.Module) -> CheckpointRef:
    """Save the module's state dict as this run's checkpoint for `model` at `update`.

    The file is written under a temporary name and renamed once complete, so
    a file with the final name is never half written. It is then added to
    `checkpoints/index.jsonl`. Raises `CheckpointExistsError` when the run
    already has a checkpoint for this model and update.
    """
    path = _checkpoint_path(run.folder, model=model, update=update)
    if path.exists():
        raise CheckpointExistsError(path)
    unfinished = path.with_suffix(".partial")
    torch.save(module.state_dict(), unfinished)
    sha256 = _sha256(unfinished)
    unfinished.rename(path)
    _append(path.parent / "index.jsonl", {"model": model, "update": update, "sha256": sha256})
    return CheckpointRef(run_id=run.folder.name, model=model, update=update, sha256=sha256)


def record_evaluation(
    run: Running, *, checkpoint: CheckpointRef, opponent: str, deals: int, rate: float, low: float, high: float
) -> None:
    record = {
        "checkpoint": asdict(checkpoint),
        "opponent": opponent,
        "deals": deals,
        "rate": rate,
        "low": low,
        "high": high,
    }
    _append(run.folder / "evaluations.jsonl", record)


def load_checkpoint(root: Path, ref: CheckpointRef, *, into: nn.Module) -> None:
    """Load the referenced checkpoint's weights into `into`.

    `root` is the folder that holds the run folders, and the reference names
    the run. The file is hashed first and `CheckpointMismatchError` is raised
    when the digest differs from the reference, before anything is loaded.
    A missing file raises `FileNotFoundError`.
    """
    path = _checkpoint_path(root / ref.run_id, model=ref.model, update=ref.update)
    actual = _sha256(path)
    if actual != ref.sha256:
        raise CheckpointMismatchError(path, expected=ref.sha256, actual=actual)
    into.load_state_dict(torch.load(path, weights_only=True))


def _described(identity: Identity) -> dict[str, object]:
    # The identity as JSON values. The run id hashes exactly this, and `identity.json` holds it.
    return {
        "config": dict(identity.config),
        "seed": identity.seed,
        "jace_commit": identity.jace_commit,
        "manabot_commit": identity.manabot_commit,
        "inputs": [asdict(checkpoint) for checkpoint in identity.inputs],
    }


def _enter_running(folder: Path, *, clock: Clock) -> None:
    # The caller holds the lock.
    state = _settled_state(folder, clock=clock)
    if state not in {RunState.CREATED, RunState.INTERRUPTED}:
        raise RunNotResumableError(folder, state=state)
    _record_state(folder, RunState.RUNNING, clock=clock)


def _checkpoint_path(folder: Path, *, model: ModelName, update: int) -> Path:
    return folder / "checkpoints" / f"{model}-{update:0{_UPDATE_DIGITS}d}.pt"


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _append(path: Path, record: Mapping[str, object]) -> None:
    # One write call to a file opened for append lands as one whole line, even when the process is killed.
    line = (json.dumps(record, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _RECORD_FILE_MODE)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def _record_state(folder: Path, state: RunState, *, clock: Clock, error: str | None = None) -> None:
    record: dict[str, object] = {"state": state.value, "time": clock().astimezone(UTC).isoformat()}
    if error is not None:
        record["error"] = error
    _append(folder / "state.jsonl", record)


def _last_recorded_state(folder: Path) -> RunState:
    last_line = (folder / "state.jsonl").read_text().splitlines()[-1]
    return RunState(json.loads(last_line)["state"])


def _try_lock(folder: Path) -> int | None:
    descriptor = os.open(folder / "lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def _settled_state(folder: Path, *, clock: Clock) -> RunState:
    # The caller holds the lock, so a recorded RUNNING was left by a writer that died.
    state = _last_recorded_state(folder)
    if state is not RunState.RUNNING:
        return state
    _record_state(folder, RunState.INTERRUPTED, clock=clock)
    return RunState.INTERRUPTED


__all__ = (
    "CheckpointExistsError",
    "CheckpointMismatchError",
    "CheckpointRef",
    "Clock",
    "Identity",
    "ModelName",
    "RunExistsError",
    "RunLockedError",
    "RunNotResumableError",
    "RunState",
    "Running",
    "create_run",
    "load_checkpoint",
    "log_update",
    "read_state",
    "record_evaluation",
    "save_checkpoint",
    "writing",
)

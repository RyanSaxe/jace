"""Remote bash script builders and chunked transfer helpers for `scripts/colab_run.py`.

Defines the Colab launcher configuration and command protocols (`ColabRunConfig`,
`CommandResult`, `CommandRunner`), remote bootstrap/restart/poll script generation,
SHA-256-verified chunked uploads, incremental snapshot downloads, and dry-run plan
formatting.
"""

from __future__ import annotations

import hashlib
import logging
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    "CHECKPOINT_SUFFIXES",
    "COLAB_CMD",
    "DEFAULT_EVAL_STALL_TIMEOUT_S",
    "DEFAULT_EXPERIMENT",
    "DEFAULT_MAX_STALL_RESTARTS",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_REPO_URL",
    "DEFAULT_STALL_TIMEOUT_S",
    "DEFAULT_SYNC_INTERVAL_S",
    "DEFAULT_UPLOAD_CHUNK_BYTES",
    "GPU_CHOICES",
    "REMOTE_DECKS_PATH",
    "REMOTE_EXEC_TIMEOUT_S",
    "REMOTE_EXIT_FILE",
    "REMOTE_LOG_FILE",
    "REMOTE_PID_FILE",
    "REMOTE_REPO_DIR",
    "REMOTE_RUNS_DIR",
    "RESUME_FILENAME",
    "TRANSFER_TIMEOUT_S",
    "ColabRunConfig",
    "CommandResult",
    "CommandRunner",
    "GpuKind",
    "build_preflight_args",
    "build_remote_script",
    "build_restart_script",
    "build_train_args",
    "extract_int_flag",
    "format_dry_run_plan",
    "incremental_snapshot_targets",
    "is_checkpoint_artifact",
    "is_checkpoint_artifact_name",
    "planned_commands",
    "prepare_resume_dirs_command",
    "provision_command",
    "remote_poll_code",
    "remote_run_dir",
    "resume_uploads",
    "shift_seed_args",
    "sync_metrics_only",
    "sync_once",
    "sync_targets",
    "upload_file",
    "upload_resume_files",
)

GpuKind = Literal["T4", "L4", "G4", "A100", "H100"]

GPU_CHOICES: tuple[GpuKind, ...] = ("T4", "L4", "G4", "A100", "H100")
DEFAULT_REPO_URL = "https://github.com/RyanSaxe/jace.git"
DEFAULT_EXPERIMENT = "colab"
DEFAULT_POLL_INTERVAL_S = 300
DEFAULT_SYNC_INTERVAL_S = 900
DEFAULT_STALL_TIMEOUT_S = 180
DEFAULT_EVAL_STALL_TIMEOUT_S = 1800
DEFAULT_MAX_STALL_RESTARTS = 20
DEFAULT_UPLOAD_CHUNK_BYTES = 48 * 1024 * 1024

COLAB_CMD: tuple[str, ...] = ("uvx", "--from", "google-colab-cli", "colab")
REMOTE_DECKS_PATH = "/content/decks.pt"
REMOTE_REPO_DIR = "/content/jace"
REMOTE_RUNS_DIR = "/content/runs"
REMOTE_PID_FILE = "/content/train.pid"
REMOTE_EXIT_FILE = "/content/train.exit"
REMOTE_LOG_FILE = "/content/train.log"
RESUME_FILENAME = "resume-state.pt"
CHECKPOINT_SUFFIXES: tuple[str, ...] = (".safetensors", ".policy.json")
REMOTE_EXEC_TIMEOUT_S = 45
TRANSFER_TIMEOUT_S = 600.0

_LOGGER = logging.getLogger(__name__)
_KILL_REMOTE_TRAIN = (
    "if [ -f /content/train.pid ]; then "
    'OLD=$(cat /content/train.pid); pkill -9 -P "$OLD" 2>/dev/null || true; kill -9 "$OLD" 2>/dev/null || true; '
    "fi; pkill -9 -f '/content/jace/.venv/bin/python -m jace\\.train\\.full_data' || true"
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str = ""


class CommandRunner(Protocol):
    def __call__(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        input_text: str | None = None,
        timeout: float | None = None,
    ) -> CommandResult: ...


@dataclass(frozen=True, slots=True)
class ColabRunConfig:
    session: str
    decks_path: Path
    backup_dir: Path
    run_id: str
    gpu: GpuKind = "L4"
    high_mem: bool = False
    repo_url: str = DEFAULT_REPO_URL
    branch: str = "main"
    resume_from: Path | None = None
    experiment: str = DEFAULT_EXPERIMENT
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S
    sync_interval_s: int = DEFAULT_SYNC_INTERVAL_S
    stall_timeout_s: int = DEFAULT_STALL_TIMEOUT_S
    eval_stall_timeout_s: int = DEFAULT_EVAL_STALL_TIMEOUT_S
    max_stall_restarts: int = DEFAULT_MAX_STALL_RESTARTS
    upload_chunk_bytes: int = DEFAULT_UPLOAD_CHUNK_BYTES
    reuse_session: bool = False
    keep_vm: bool = False
    dry_run: bool = False
    preflight_first: bool = False
    extra_args: tuple[str, ...] = ()


def remote_run_dir(cfg: ColabRunConfig) -> str:
    return f"{REMOTE_RUNS_DIR}/{cfg.experiment}/{cfg.run_id}"


def provision_command(cfg: ColabRunConfig) -> tuple[str, ...]:
    extra = ("--high-mem",) if cfg.high_mem else ()
    return (*COLAB_CMD, "new", "-s", cfg.session, "--gpu", cfg.gpu, *extra)


def is_checkpoint_artifact_name(name: str) -> bool:
    if name == RESUME_FILENAME:
        return False
    is_weight_snapshot = name.startswith("weights-u") and name.endswith(".pt")
    return name.endswith(CHECKPOINT_SUFFIXES) or is_weight_snapshot


def is_checkpoint_artifact(path: Path) -> bool:
    return path.is_file() and is_checkpoint_artifact_name(path.name)


def resume_uploads(resume_from: Path, *, remote_run_dir_path: str) -> list[tuple[Path, str]]:
    uploads: list[tuple[Path, str]] = [
        (candidate, f"{remote_run_dir_path}/{filename}")
        for filename in ("config.json", "metrics.jsonl")
        if (candidate := resume_from / filename).is_file()
    ]
    checkpoints_dir = resume_from / "checkpoints"
    uploads.append((checkpoints_dir / RESUME_FILENAME, f"{remote_run_dir_path}/checkpoints/{RESUME_FILENAME}"))
    uploads.extend(
        (path, f"{remote_run_dir_path}/checkpoints/{path.name}")
        for path in sorted(checkpoints_dir.iterdir())
        if is_checkpoint_artifact(path)
    )
    return uploads


def prepare_resume_dirs_command(session: str, *, remote_run_dir_path: str) -> tuple[str, ...]:
    dirs = (
        f"{remote_run_dir_path}/checkpoints",
        f"{remote_run_dir_path}/games",
        f"{remote_run_dir_path}/forensics",
    )
    return (*COLAB_CMD, "ssh", "-s", session, "--", "mkdir", "-p", *dirs)


def sync_targets(backup_dir: Path, *, remote_run_dir_path: str) -> tuple[tuple[str, Path], ...]:
    return (
        (f"{remote_run_dir_path}/metrics.jsonl", backup_dir / "metrics.jsonl"),
        (f"{remote_run_dir_path}/config.json", backup_dir / "config.json"),
        (f"{remote_run_dir_path}/checkpoints/{RESUME_FILENAME}", backup_dir / "checkpoints" / RESUME_FILENAME),
    )


def extract_int_flag(extra_args: Sequence[str], *, flag: str, default: int) -> int:
    prefix = f"{flag}="
    for idx, arg in enumerate(extra_args):
        if arg == flag and idx + 1 < len(extra_args) and extra_args[idx + 1].lstrip("-").isdigit():
            return int(extra_args[idx + 1])
        if arg.startswith(prefix):
            raw = arg.removeprefix(prefix)
            if raw.lstrip("-").isdigit():
                return int(raw)
    return default


def shift_seed_args(extra_args: Sequence[str], *, seed_offset: int) -> list[str]:
    base_seed = extract_int_flag(extra_args, flag="--seed", default=1)
    filtered: list[str] = []
    skip_next = False
    for idx, arg in enumerate(extra_args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--seed" and idx + 1 < len(extra_args):
            skip_next = True
        elif not arg.startswith("--seed="):
            filtered.append(arg)
    return [*filtered, "--seed", str(base_seed + seed_offset)]


def build_train_args(cfg: ColabRunConfig, *, force_resume: bool = False, seed_offset: int = 0) -> list[str]:
    train_args = [
        *("uv", "run", "python", "-m", "jace.train.full_data"),
        *("--runs-dir", REMOTE_RUNS_DIR, "--experiment", cfg.experiment),
        *("--run-id", cfg.run_id, "--decks-path", REMOTE_DECKS_PATH),
    ]
    if (cfg.resume_from is not None or force_resume) and "--resume" not in cfg.extra_args:
        train_args.append("--resume")
    if seed_offset == 0:
        train_args.extend(cfg.extra_args)
    else:
        train_args.extend(shift_seed_args(cfg.extra_args, seed_offset=seed_offset))
    return train_args


def build_preflight_args(cfg: ColabRunConfig) -> list[str]:
    args = [arg for arg in build_train_args(cfg, force_resume=False) if arg != "--resume"]
    if "--preflight" not in args:
        args.append("--preflight")
    return args


def build_remote_script(cfg: ColabRunConfig) -> str:
    train_args = build_train_args(cfg, force_resume=cfg.reuse_session)
    branch_q = shlex.quote(cfg.branch)
    clone_cmd = (
        f"  git clone --branch {branch_q} --single-branch {shlex.quote(cfg.repo_url)} {shlex.quote(REMOTE_REPO_DIR)}"
    )
    inner_cmd = f"{shlex.join(train_args)}; rc=$?; echo $rc > {REMOTE_EXIT_FILE}; exit $rc"
    repo_sync_cmd = (
        f"if [ ! -d {shlex.quote(REMOTE_REPO_DIR)}/.git ]; then\n{clone_cmd}\nelse\n"
        f"  git -C {shlex.quote(REMOTE_REPO_DIR)} fetch origin {branch_q} && "
        f"git -C {shlex.quote(REMOTE_REPO_DIR)} checkout -B {branch_q} FETCH_HEAD\nfi"
    )
    preflight_lines = [shlex.join(build_preflight_args(cfg))] if cfg.preflight_first else []
    lines = [
        "set -euo pipefail",
        'export PATH="$HOME/.local/bin:$PATH"',
        'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"',
        "if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi",
        repo_sync_cmd,
        *([_KILL_REMOTE_TRAIN, "sleep 2"] if cfg.reuse_session else []),
        f"cd {shlex.quote(REMOTE_REPO_DIR)}",
        "uv sync --frozen",
        *preflight_lines,
        f"rm -f {REMOTE_EXIT_FILE}",
        f"nohup bash -lc {shlex.quote(inner_cmd)} >{REMOTE_LOG_FILE} 2>&1 </dev/null &",
        f"PID=$!; echo $PID > {REMOTE_PID_FILE}; sleep 2; kill -0 $PID; echo LAUNCHED pid=$PID",
    ]
    return "\n".join(lines) + "\n"


def build_restart_script(cfg: ColabRunConfig, *, seed_offset: int) -> str:
    train_args = build_train_args(cfg, force_resume=True, seed_offset=seed_offset)
    inner_cmd = f"{shlex.join(train_args)}; rc=$?; echo $rc > {REMOTE_EXIT_FILE}; exit $rc"
    lines = [
        "set -euo pipefail",
        'export PATH="$HOME/.local/bin:$PATH"',
        'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"',
        _KILL_REMOTE_TRAIN,
        "sleep 2",
        f"cd {shlex.quote(REMOTE_REPO_DIR)}",
        f"rm -f {REMOTE_EXIT_FILE}",
        f"nohup bash -lc {shlex.quote(inner_cmd)} >>{REMOTE_LOG_FILE} 2>&1 </dev/null &",
        f"PID=$!; echo $PID > {REMOTE_PID_FILE}; sleep 2; kill -0 $PID; echo RESTARTED pid=$PID",
    ]
    return "\n".join(lines) + "\n"


def remote_poll_code(remote_run_dir_path: str) -> str:
    metrics_path = f"{remote_run_dir_path}/metrics.jsonl"
    checkpoints_path = f"{remote_run_dir_path}/checkpoints"
    return (
        "import json, os, pathlib, subprocess\n"
        f"pid_file, exit_file = pathlib.Path({REMOTE_PID_FILE!r}), pathlib.Path({REMOTE_EXIT_FILE!r})\n"
        "pid = int(pid_file.read_text().strip()) if pid_file.exists() else -1\n"
        "alive = False\n"
        "if pid > 0:\n"
        "    try:\n"
        "        os.kill(pid, 0)\n"
        '        stat = subprocess.check_output(["ps", "-p", str(pid), "-o", "stat="]).decode().strip()\n'
        '        alive = bool(stat) and "Z" not in stat\n'
        "    except Exception:\n"
        "        alive = False\n"
        "exit_code = exit_file.read_text().strip() if exit_file.exists() else ''\n"
        'gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",'
        ' "--format=csv,noheader"]).decode().strip()\n'
        f"mp = pathlib.Path({metrics_path!r})\n"
        "lines = mp.read_text().splitlines() if mp.exists() and mp.stat().st_size else []\n"
        'last = lines[-1][:140] if lines else ""\n'
        'lu, lk = "none", "none"\n'
        "if lines:\n"
        "    try:\n"
        "        rec = json.loads(lines[-1])\n"
        '        lu, lk = str(rec.get("update", "none")), str(rec.get("kind", "none"))\n'
        "    except Exception:\n"
        "        pass\n"
        f"cp = pathlib.Path({checkpoints_path!r})\n"
        'snaps = ",".join(sorted(p.name for p in cp.glob("weights-u*.pt") if p.is_file())) if cp.is_dir() else ""\n'
        'print(f"ALIVE={alive} EXIT={exit_code} PID={pid} LINES={len(lines)} '
        'LAST_UPDATE={lu} LAST_KIND={lk} SNAPSHOTS={snaps} GPU=[{gpu}] LAST={last}")\n'
        f"lp = pathlib.Path({REMOTE_LOG_FILE!r})\n"
        'if not alive and lp.exists(): print("LOG_TAIL:\\n" + lp.read_text(errors="ignore")[-1500:])\n'
    )


def upload_file(
    cfg: ColabRunConfig,
    *,
    local_path: Path,
    remote_path: str,
    runner: CommandRunner,
) -> None:
    if local_path.stat().st_size <= cfg.upload_chunk_bytes:
        runner(
            (*COLAB_CMD, "upload", "-s", cfg.session, str(local_path), remote_path),
            check=True,
            timeout=TRANSFER_TIMEOUT_S,
        )
        return

    payload = local_path.read_bytes()
    expected_sha = hashlib.sha256(payload).hexdigest()
    remote_parts_dir = f"{remote_path}.parts"
    runner((*COLAB_CMD, "ssh", "-s", cfg.session, "--", "mkdir", "-p", remote_parts_dir), check=True, timeout=60.0)
    with tempfile.TemporaryDirectory(prefix="jace-colab-chunk-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        chunk_size = cfg.upload_chunk_bytes
        for idx, offset in enumerate(range(0, len(payload), chunk_size)):
            part_path = tmp_root / f"part-{idx:04d}"
            part_path.write_bytes(payload[offset : offset + chunk_size])
            runner(
                (*COLAB_CMD, "upload", "-s", cfg.session, str(part_path), f"{remote_parts_dir}/{part_path.name}"),
                check=True,
                timeout=TRANSFER_TIMEOUT_S,
            )
            part_path.unlink()

    reassemble = (
        f"set -euo pipefail; cat {shlex.quote(remote_parts_dir)}/part-* > {shlex.quote(remote_path)}; "
        f"rm -rf {shlex.quote(remote_parts_dir)}; "
        f"actual=$(sha256sum {shlex.quote(remote_path)} | awk '{{print $1}}'); "
        f'test "$actual" = {shlex.quote(expected_sha)}'
    )
    runner(
        (*COLAB_CMD, "ssh", "-s", cfg.session, "--", "bash", "-lc", reassemble),
        check=True,
        timeout=TRANSFER_TIMEOUT_S,
    )


def incremental_snapshot_targets(
    backup_dir: Path,
    *,
    remote_run_dir_path: str,
    snapshot_names: Sequence[str] | set[str],
) -> list[tuple[str, Path]]:
    targets: list[tuple[str, Path]] = []
    for name in sorted(set(snapshot_names)):
        if not is_checkpoint_artifact_name(name):
            continue
        local_path = backup_dir / "checkpoints" / name
        if local_path.is_file():
            continue
        targets.append((f"{remote_run_dir_path}/checkpoints/{name}", local_path))
    return targets


def sync_once(
    cfg: ColabRunConfig,
    *,
    remote_run_dir: str,
    runner: CommandRunner,
    snapshot_names: Sequence[str] | set[str] = (),
) -> None:
    (cfg.backup_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    targets = [
        *sync_targets(cfg.backup_dir, remote_run_dir_path=remote_run_dir),
        *incremental_snapshot_targets(
            cfg.backup_dir,
            remote_run_dir_path=remote_run_dir,
            snapshot_names=snapshot_names,
        ),
    ]
    for remote_path, local_path in targets:
        cmd = (*COLAB_CMD, "download", "-s", cfg.session, remote_path, str(local_path))
        try:
            result = runner(cmd, check=False, timeout=TRANSFER_TIMEOUT_S)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            _LOGGER.warning("sync failed for %s: %s", remote_path, exc)
            continue
        if result.returncode != 0:
            _LOGGER.debug("sync skipped %s (exit %d)", remote_path, result.returncode)


def upload_resume_files(cfg: ColabRunConfig, *, remote_run_dir_path: str, runner: CommandRunner) -> None:
    if cfg.resume_from is None:
        return
    # Jupyter's Contents API requires parent folders to exist before PUT uploads.
    runner(prepare_resume_dirs_command(cfg.session, remote_run_dir_path=remote_run_dir_path), check=True, timeout=60.0)
    for local_path, remote_path in resume_uploads(cfg.resume_from, remote_run_dir_path=remote_run_dir_path):
        upload_file(cfg, local_path=local_path, remote_path=remote_path, runner=runner)


def sync_metrics_only(cfg: ColabRunConfig, *, remote_run_dir_path: str, runner: CommandRunner) -> None:
    cfg.backup_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        *COLAB_CMD,
        "download",
        "-s",
        cfg.session,
        f"{remote_run_dir_path}/metrics.jsonl",
        str(cfg.backup_dir / "metrics.jsonl"),
    )
    try:
        runner(cmd, check=False, timeout=TRANSFER_TIMEOUT_S)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        _LOGGER.warning("metrics sync before restart failed: %s", exc)


def planned_commands(cfg: ColabRunConfig, *, remote_run_dir_path: str, remote_script: str) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = [
        provision_command(cfg),
        (*COLAB_CMD, "upload", "-s", cfg.session, str(cfg.decks_path), REMOTE_DECKS_PATH),
    ]
    if cfg.resume_from is not None:
        commands.append(prepare_resume_dirs_command(cfg.session, remote_run_dir_path=remote_run_dir_path))
        commands.extend(
            (*COLAB_CMD, "upload", "-s", cfg.session, str(local_path), remote_path)
            for local_path, remote_path in resume_uploads(cfg.resume_from, remote_run_dir_path=remote_run_dir_path)
        )
    commands.append((*COLAB_CMD, "ssh", "-s", cfg.session, "--", "bash", "-lc", remote_script))
    commands.append((*COLAB_CMD, "exec", "-s", cfg.session, "--timeout", str(REMOTE_EXEC_TIMEOUT_S)))
    commands.extend(
        (*COLAB_CMD, "download", "-s", cfg.session, remote_path, str(local_path))
        for remote_path, local_path in sync_targets(cfg.backup_dir, remote_run_dir_path=remote_run_dir_path)
    )
    if not cfg.keep_vm:
        commands.append((*COLAB_CMD, "stop", "-s", cfg.session))
        commands.append((*COLAB_CMD, "usage"))
    return commands


def format_dry_run_plan(cfg: ColabRunConfig) -> str:
    run_dir = remote_run_dir(cfg)
    remote_script = build_remote_script(cfg)
    commands = planned_commands(cfg, remote_run_dir_path=run_dir, remote_script=remote_script)
    sections = [
        "Local commands:",
        *(f"  {shlex.join(cmd)}" for cmd in commands),
        "",
        "Remote script:",
        remote_script.rstrip("\n"),
    ]
    return "\n".join(sections) + "\n"

"""Check a run's life on disk through the tracking module's public functions.

Every test makes a real run folder under pytest's temporary directory and
reads it back the way a caller would. The dead writer is a real child
process that enters the writing block and exits without leaving it, so the
operating system, not a stand-in, releases its lock.
"""

import functools
import multiprocessing
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from torch import nn

from jace.tracking import (
    CheckpointMismatchError,
    Identity,
    RunLockedError,
    RunNotResumableError,
    RunState,
    create_run,
    load_checkpoint,
    read_state,
    save_checkpoint,
    writing,
)

__all__ = ()


_now = functools.partial(datetime.now, UTC)


def _new_run(root: Path) -> Path:
    identity = Identity(config={"lr": 0.001}, seed=7, jace_commit="abc", manabot_commit="def", inputs=())
    return create_run(root, identity, clock=_now)


def _die_while_writing(folder: Path) -> None:
    with writing(folder, clock=_now):
        os._exit(0)


def test_a_block_that_exits_normally_completes_the_run_for_good(tmp_path: Path) -> None:
    folder = _new_run(tmp_path)
    assert read_state(folder, clock=_now) is RunState.CREATED

    with writing(folder, clock=_now):
        pass

    assert read_state(folder, clock=_now) is RunState.COMPLETED
    with pytest.raises(RunNotResumableError, match="completed"), writing(folder, clock=_now):
        pass


def test_an_error_in_the_block_is_re_raised_and_fails_the_run_for_good(tmp_path: Path) -> None:
    folder = _new_run(tmp_path)

    message = "the trainer broke"
    with pytest.raises(RuntimeError, match=message), writing(folder, clock=_now):
        raise RuntimeError(message)

    assert read_state(folder, clock=_now) is RunState.FAILED
    with pytest.raises(RunNotResumableError, match="failed"), writing(folder, clock=_now):
        pass


def test_a_keyboard_interrupt_is_re_raised_and_leaves_the_run_resumable(tmp_path: Path) -> None:
    folder = _new_run(tmp_path)

    with pytest.raises(KeyboardInterrupt), writing(folder, clock=_now):
        raise KeyboardInterrupt

    assert read_state(folder, clock=_now) is RunState.INTERRUPTED
    with writing(folder, clock=_now):
        pass
    assert read_state(folder, clock=_now) is RunState.COMPLETED


def test_a_run_whose_writer_died_reads_interrupted_and_can_be_resumed(tmp_path: Path) -> None:
    folder = _new_run(tmp_path)
    writer = multiprocessing.get_context("spawn").Process(target=_die_while_writing, args=(folder,))
    writer.start()
    writer.join()
    assert writer.exitcode == 0

    assert read_state(folder, clock=_now) is RunState.INTERRUPTED
    with writing(folder, clock=_now):
        assert read_state(folder, clock=_now) is RunState.RUNNING


def test_a_second_writer_is_refused_while_the_first_holds_the_run(tmp_path: Path) -> None:
    folder = _new_run(tmp_path)

    with writing(folder, clock=_now):
        with pytest.raises(RunLockedError), writing(folder, clock=_now):
            pass
        assert read_state(folder, clock=_now) is RunState.RUNNING

    assert read_state(folder, clock=_now) is RunState.COMPLETED


def test_a_checkpoint_round_trips_and_a_tampered_file_is_rejected(tmp_path: Path) -> None:
    saved = nn.Linear(3, 2)
    with writing(_new_run(tmp_path), clock=_now) as run:
        ref = save_checkpoint(run, model="player", update=3, module=saved)

    loaded = nn.Linear(3, 2)
    load_checkpoint(tmp_path, ref, into=loaded)
    assert torch.equal(loaded.weight, saved.weight)
    assert torch.equal(loaded.bias, saved.bias)

    (checkpoint_file,) = tmp_path.rglob("*.pt")
    with checkpoint_file.open("ab") as file:
        file.write(b"\0")
    with pytest.raises(CheckpointMismatchError):
        load_checkpoint(tmp_path, ref, into=nn.Linear(3, 2))

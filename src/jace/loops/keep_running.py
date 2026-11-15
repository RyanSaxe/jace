"""Run limited training and restart it from its last checkpoint whenever it stalls.

The game engine can stay inside one step for many minutes, and Python
cannot interrupt a call into it. This script therefore starts a long run
as a child process and watches it. The child rewrites its curve file after
every iteration, so when the curve file has not changed for
`--stall-seconds`, the script treats the child as stuck. It kills the
child and starts it again with `--resume`, which continues from the newest
checkpoints. Pass a `--checkpoint-every` small enough that a restart loses
little.

Run it with
`uv run python -m jace.loops.keep_running --stall-seconds 900 -- <train_limited flags>`.
"""

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

__all__ = ("main",)

_POLL_SECONDS = 15.0
_TRAIN = (sys.executable, "-m", "jace.loops.train_limited")


def _newest_run(run_dir: Path, *, known: frozenset[Path]) -> Path | None:
    # The run folder made since `known` was listed, or None until the child makes one. Run ids begin with their
    # creation time, so the newest folder sorts last by name.
    created = [folder for folder in run_dir.iterdir() if folder.is_dir() and folder not in known]
    return max(created, key=lambda folder: folder.name) if created else None


def _stalled(child: subprocess.Popen[bytes], progress: Path | None, *, since: float, stall_seconds: float) -> bool:
    # Waits until the child exits or has made no progress for `stall_seconds`, and returns whether it stalled.
    # The last progress is the later of `since` and the curve file's last change. After a restart, the curve file
    # was last changed before the kill, so counting from `since` gives the new child the full `stall_seconds`.
    while child.poll() is None:
        changed = progress.stat().st_mtime if progress is not None and progress.exists() else since
        if time.time() - max(changed, since) > stall_seconds:
            return True
        time.sleep(_POLL_SECONDS)
    return False


def main(argv: Sequence[str]) -> None:
    """Run `train_limited` with the flags after `--`, restarting it when it stalls.

    When the child exits on its own, `main` exits with the child's exit
    code. For each restart, it writes a line to standard output that names
    the run folder being resumed.

    A child that stalls before its first checkpoint cannot be resumed. Its
    restart then fails with an error that names the missing checkpoint, and
    `main` exits with that failure's exit code.
    """
    parser = argparse.ArgumentParser(description="Run limited training and restart it whenever it stalls.")
    parser.add_argument("--stall-seconds", type=float, required=True, help="restart after this long with no iteration")
    parser.add_argument("flags", nargs=argparse.REMAINDER, help="-- followed by the flags of jace.loops.train_limited")
    arguments = parser.parse_args(argv)
    flags = [flag for flag in arguments.flags if flag != "--"]
    run_dir = Path(flags[flags.index("--run-dir") + 1])
    run_dir.mkdir(parents=True, exist_ok=True)
    known = frozenset(run_dir.iterdir())
    folder: Path | None = None
    restarts = 0
    while True:
        resume = [] if folder is None else ["--resume", str(folder)]
        started = time.time()
        child = subprocess.Popen([*_TRAIN, *flags, *resume])
        curve = None
        while folder is None and child.poll() is None:
            folder = _newest_run(run_dir, known=known)
            time.sleep(1.0)
        if folder is not None:
            curve = folder.with_name(f"{folder.name}.curve.json")
        if not _stalled(child, curve, since=started, stall_seconds=arguments.stall_seconds):
            sys.exit(child.returncode)
        child.kill()
        child.wait()
        restarts += 1
        sys.stdout.write(f"no iteration for {arguments.stall_seconds:.0f} seconds; restart {restarts} from {folder}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:])

"""Run limited training and restart it from its last checkpoint whenever it stalls.

The game engine can stay inside one step for many minutes, and a call into
it cannot be interrupted from Python. A long run is therefore started as a
child process and watched from here: the child rewrites its curve file
after every iteration, so a curve file that has not changed for
`--stall-seconds` means the child is stuck. It is killed and started again
with `--resume`, which continues from the newest checkpoints, so pass a
`--checkpoint-every` small enough that a restart loses little. Run it with
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
    # The run folder created since `known` was listed, once the child has made it.
    created = [folder for folder in run_dir.iterdir() if folder.is_dir() and folder not in known]
    return max(created, key=lambda folder: folder.name) if created else None


def _stalled(child: subprocess.Popen[bytes], progress: Path | None, *, since: float, stall_seconds: float) -> bool:
    # Waits until the child exits or has shown no progress for `stall_seconds`. Returns whether it stalled.
    # Progress is the curve file's modification time, or `since` while the file does not exist yet.
    while child.poll() is None:
        changed = progress.stat().st_mtime if progress is not None and progress.exists() else since
        if time.time() - max(changed, since) > stall_seconds:
            return True
        time.sleep(_POLL_SECONDS)
    return False


def main(argv: Sequence[str]) -> None:
    """Run `train_limited` with the flags after `--`, restarting it when it stalls.

    Exits with the child's exit code once it finishes by itself. Each
    restart is reported on standard output with the run folder it resumes.
    A child that stalls before its first checkpoint cannot be resumed, and
    its restart fails with the error that names the missing checkpoint.
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

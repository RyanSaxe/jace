#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Reject Python files longer than the repository's line limit.

A file past the limit usually holds more than one idea, and splitting it is
cheaper early than late. The check takes file paths as arguments, or scans
every tracked and untracked-but-not-ignored Python file when given none,
which is how the CI gate runs it. BYOR's own vendored scripts under `.byor/`
are skipped, because this repository does not maintain them.
"""

import shutil
import subprocess
import sys
from pathlib import Path

MAX_LINES = 500
SKIPPED_FOLDER = ".byor"


def main(arguments: list[str]) -> int:
    too_long = [(path, lines) for path in _python_files(arguments) if (lines := _line_count(path)) > MAX_LINES]
    for path, lines in too_long:
        sys.stdout.write(f"{path}: {lines} lines, the limit is {MAX_LINES}; split the file by responsibility\n")
    return 1 if too_long else 0


def _python_files(arguments: list[str]) -> list[Path]:
    named = arguments or _repository_files()
    paths = [Path(name) for name in named if name.endswith((".py", ".pyi"))]
    return [path for path in paths if SKIPPED_FOLDER not in path.parts and path.is_file()]


def _repository_files() -> list[str]:
    git = shutil.which("git")
    if git is None:
        message = "git is needed to list the repository's files and was not found"
        raise FileNotFoundError(message)
    listed = subprocess.run(
        (git, "ls-files", "-co", "--exclude-standard", "-z"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [name for name in listed.stdout.split("\0") if name]


def _line_count(path: Path) -> int:
    with path.open("rb") as file:
        return sum(1 for _ in file)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

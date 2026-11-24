#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Google Colab GPU training launcher with persistent SSH and checkpoint sync.

Provisions a Colab runtime through `uvx --from google-colab-cli colab`, uploads
cached 17Lands decks and optional resume artifacts, and runs
`jace.train.full_data` over a single foreground SSH session so Colab's
idle-client reaper does not terminate long runs. A background thread syncs
`metrics.jsonl`, `config.json`, and `checkpoints/resume-state.pt` over the
Jupyter HTTP API while training runs, and cleanup always performs a final sync
before stopping the VM unless `--keep-vm` is set.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    "DEFAULT_EXPERIMENT",
    "DEFAULT_REPO_URL",
    "DEFAULT_SYNC_INTERVAL_S",
    "GPU_CHOICES",
    "ColabRunConfig",
    "CommandRunner",
    "GpuKind",
    "build_remote_script",
    "format_dry_run_plan",
    "main",
    "run_colab_training",
)

GpuKind = Literal["T4", "L4", "A100", "H100"]

GPU_CHOICES: tuple[GpuKind, ...] = ("T4", "L4", "A100", "H100")
DEFAULT_REPO_URL = "https://github.com/RyanSaxe/jace.git"
DEFAULT_EXPERIMENT = "v2-colab"
DEFAULT_SYNC_INTERVAL_S = 900

_COLAB_CMD: tuple[str, ...] = ("uvx", "--from", "google-colab-cli", "colab")
_REMOTE_DECKS_PATH = "/content/decks.pt"
_REMOTE_REPO_DIR = "/content/jace"
_REMOTE_RUNS_DIR = "/content/runs"
_RESUME_FILENAME = "resume-state.pt"
_CHECKPOINT_SUFFIXES: tuple[str, ...] = (".safetensors", ".policy.json")
_GPU_MAP: dict[str, GpuKind] = {"T4": "T4", "L4": "L4", "A100": "A100", "H100": "H100"}
_LOGGER = logging.getLogger(__name__)


class CommandRunner(Protocol):
    def __call__(self, args: Sequence[str], *, check: bool = True) -> int: ...


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
    sync_interval_s: int = DEFAULT_SYNC_INTERVAL_S
    keep_vm: bool = False
    dry_run: bool = False
    extra_args: tuple[str, ...] = ()


def _remote_run_dir(cfg: ColabRunConfig) -> str:
    return f"{_REMOTE_RUNS_DIR}/{cfg.experiment}/{cfg.run_id}"


def _run_subprocess(args: Sequence[str], *, check: bool = True) -> int:
    completed = subprocess.run(tuple(args), check=check)
    return completed.returncode


def _validate_inputs(cfg: ColabRunConfig) -> None:
    if not cfg.decks_path.is_file():
        msg = f"decks file not found: {cfg.decks_path}"
        raise FileNotFoundError(msg)
    if cfg.sync_interval_s < 0:
        msg = f"sync_interval_s must be >= 0, got {cfg.sync_interval_s}"
        raise ValueError(msg)
    if cfg.resume_from is not None:
        resume_state = cfg.resume_from / "checkpoints" / _RESUME_FILENAME
        if not resume_state.is_file():
            msg = f"resume state not found: {resume_state}"
            raise FileNotFoundError(msg)


def _provision_command(cfg: ColabRunConfig) -> tuple[str, ...]:
    extra = ("--high-mem",) if cfg.high_mem else ()
    return (*_COLAB_CMD, "new", "-s", cfg.session, "--gpu", cfg.gpu, *extra)


def _is_checkpoint_artifact(path: Path) -> bool:
    return path.is_file() and path.name != _RESUME_FILENAME and path.name.endswith(_CHECKPOINT_SUFFIXES)


def _resume_uploads(resume_from: Path, *, remote_run_dir: str) -> list[tuple[Path, str]]:
    uploads: list[tuple[Path, str]] = [
        (candidate, f"{remote_run_dir}/{filename}")
        for filename in ("config.json", "metrics.jsonl")
        if (candidate := resume_from / filename).is_file()
    ]
    checkpoints_dir = resume_from / "checkpoints"
    uploads.append((checkpoints_dir / _RESUME_FILENAME, f"{remote_run_dir}/checkpoints/{_RESUME_FILENAME}"))
    uploads.extend(
        (path, f"{remote_run_dir}/checkpoints/{path.name}")
        for path in sorted(checkpoints_dir.iterdir())
        if _is_checkpoint_artifact(path)
    )
    return uploads


def _prepare_resume_dirs_command(session: str, *, remote_run_dir: str) -> tuple[str, ...]:
    dirs = (f"{remote_run_dir}/checkpoints", f"{remote_run_dir}/games", f"{remote_run_dir}/forensics")
    return (*_COLAB_CMD, "ssh", "-s", session, "--", "mkdir", "-p", *dirs)


def _sync_targets(backup_dir: Path, *, remote_run_dir: str) -> tuple[tuple[str, Path], ...]:
    return (
        (f"{remote_run_dir}/metrics.jsonl", backup_dir / "metrics.jsonl"),
        (f"{remote_run_dir}/config.json", backup_dir / "config.json"),
        (f"{remote_run_dir}/checkpoints/{_RESUME_FILENAME}", backup_dir / "checkpoints" / _RESUME_FILENAME),
    )


def build_remote_script(cfg: ColabRunConfig) -> str:
    train_args = [
        *("uv", "run", "python", "-m", "jace.train.full_data"),
        *("--runs-dir", _REMOTE_RUNS_DIR, "--experiment", cfg.experiment),
        *("--run-id", cfg.run_id, "--decks-path", _REMOTE_DECKS_PATH),
    ]
    if cfg.resume_from is not None and "--resume" not in cfg.extra_args:
        train_args.append("--resume")
    train_args.extend(cfg.extra_args)

    clone_cmd = (
        f"  git clone --branch {shlex.quote(cfg.branch)} --single-branch "
        f"{shlex.quote(cfg.repo_url)} {shlex.quote(_REMOTE_REPO_DIR)}"
    )
    lines = [
        "set -euo pipefail",
        'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"',
        "if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi",
        (
            "if ! command -v cargo >/dev/null 2>&1; then "
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; fi"
        ),
        f"if [ ! -d {shlex.quote(_REMOTE_REPO_DIR)}/.git ]; then\n{clone_cmd}\nfi",
        f"cd {shlex.quote(_REMOTE_REPO_DIR)}",
        "uv sync --frozen",
        f"exec {shlex.join(train_args)}",
    ]
    return "\n".join(lines) + "\n"


def _sync_once(
    cfg: ColabRunConfig,
    *,
    remote_run_dir: str,
    runner: CommandRunner,
    sync_lock: threading.Lock,
) -> None:
    with sync_lock:
        (cfg.backup_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        for remote_path, local_path in _sync_targets(cfg.backup_dir, remote_run_dir=remote_run_dir):
            cmd = (*_COLAB_CMD, "download", "-s", cfg.session, remote_path, str(local_path))
            try:
                code = runner(cmd, check=False)
            except (subprocess.CalledProcessError, OSError) as exc:
                _LOGGER.warning("sync failed for %s: %s", remote_path, exc)
                continue
            if code != 0:
                _LOGGER.debug("sync skipped %s (exit %d)", remote_path, code)


def _start_sync_thread(
    cfg: ColabRunConfig,
    *,
    remote_run_dir: str,
    runner: CommandRunner,
    stop_event: threading.Event,
    sync_lock: threading.Lock,
) -> threading.Thread | None:
    if cfg.sync_interval_s <= 0:
        return None

    def _loop() -> None:
        while not stop_event.wait(timeout=cfg.sync_interval_s):
            _sync_once(cfg, remote_run_dir=remote_run_dir, runner=runner, sync_lock=sync_lock)

    thread = threading.Thread(target=_loop, name=f"colab-sync-{cfg.session}", daemon=True)
    thread.start()
    return thread


def _upload_resume_files(cfg: ColabRunConfig, *, remote_run_dir: str, runner: CommandRunner) -> None:
    if cfg.resume_from is None:
        return
    # Jupyter's Contents API requires parent folders to exist before PUT uploads.
    runner(_prepare_resume_dirs_command(cfg.session, remote_run_dir=remote_run_dir), check=True)
    for local_path, remote_path in _resume_uploads(cfg.resume_from, remote_run_dir=remote_run_dir):
        runner((*_COLAB_CMD, "upload", "-s", cfg.session, str(local_path), remote_path), check=True)


def _stop_session(session: str, *, runner: CommandRunner) -> None:
    stop_code = runner((*_COLAB_CMD, "stop", "-s", session), check=False)
    if stop_code != 0:
        _LOGGER.warning("colab stop -s %s exited with %d", session, stop_code)
    runner((*_COLAB_CMD, "usage"), check=False)


def _planned_commands(cfg: ColabRunConfig, *, remote_run_dir: str, remote_script: str) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = [
        _provision_command(cfg),
        (*_COLAB_CMD, "upload", "-s", cfg.session, str(cfg.decks_path), _REMOTE_DECKS_PATH),
    ]
    if cfg.resume_from is not None:
        commands.append(_prepare_resume_dirs_command(cfg.session, remote_run_dir=remote_run_dir))
        commands.extend(
            (*_COLAB_CMD, "upload", "-s", cfg.session, str(local_path), remote_path)
            for local_path, remote_path in _resume_uploads(cfg.resume_from, remote_run_dir=remote_run_dir)
        )
    commands.append((*_COLAB_CMD, "ssh", "-s", cfg.session, "--", "bash", "-lc", remote_script))
    commands.extend(
        (*_COLAB_CMD, "download", "-s", cfg.session, remote_path, str(local_path))
        for remote_path, local_path in _sync_targets(cfg.backup_dir, remote_run_dir=remote_run_dir)
    )
    if not cfg.keep_vm:
        commands.append((*_COLAB_CMD, "stop", "-s", cfg.session))
        commands.append((*_COLAB_CMD, "usage"))
    return commands


def format_dry_run_plan(cfg: ColabRunConfig) -> str:
    remote_run_dir = _remote_run_dir(cfg)
    remote_script = build_remote_script(cfg)
    commands = _planned_commands(cfg, remote_run_dir=remote_run_dir, remote_script=remote_script)
    sections = [
        "Local commands:",
        *(f"  {shlex.join(cmd)}" for cmd in commands),
        "",
        "Remote script:",
        remote_script.rstrip("\n"),
    ]
    return "\n".join(sections) + "\n"


def run_colab_training(cfg: ColabRunConfig, *, runner: CommandRunner | None = None) -> int:
    _validate_inputs(cfg)
    if cfg.dry_run:
        sys.stdout.write(format_dry_run_plan(cfg))
        return 0

    active_runner: CommandRunner = _run_subprocess if runner is None else runner
    remote_run_dir = _remote_run_dir(cfg)
    remote_script = build_remote_script(cfg)
    (cfg.backup_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    active_runner(_provision_command(cfg), check=True)

    stop_event = threading.Event()
    sync_lock = threading.Lock()
    sync_thread: threading.Thread | None = None
    try:
        active_runner(
            (*_COLAB_CMD, "upload", "-s", cfg.session, str(cfg.decks_path), _REMOTE_DECKS_PATH),
            check=True,
        )
        _upload_resume_files(cfg, remote_run_dir=remote_run_dir, runner=active_runner)
        sync_thread = _start_sync_thread(
            cfg,
            remote_run_dir=remote_run_dir,
            runner=active_runner,
            stop_event=stop_event,
            sync_lock=sync_lock,
        )
        ssh_cmd = (*_COLAB_CMD, "ssh", "-s", cfg.session, "--", "bash", "-lc", remote_script)
        return active_runner(ssh_cmd, check=True)
    finally:
        stop_event.set()
        if sync_thread is not None:
            sync_thread.join(timeout=30.0)
        _sync_once(cfg, remote_run_dir=remote_run_dir, runner=active_runner, sync_lock=sync_lock)
        if not cfg.keep_vm:
            _stop_session(cfg.session, runner=active_runner)


def _parse_gpu(raw: str) -> GpuKind:
    if raw not in _GPU_MAP:
        msg = f"unsupported GPU {raw!r}; expected one of {', '.join(GPU_CHOICES)}"
        raise ValueError(msg)
    return _GPU_MAP[raw]


def _parse_args(argv: Sequence[str] | None) -> ColabRunConfig:
    parser = argparse.ArgumentParser(
        description="Run jace.train.full_data on a Google Colab GPU with persistent SSH and local sync.",
        epilog="Additional arguments (optionally after '--') are forwarded to jace.train.full_data.",
    )
    parser.add_argument("--session", required=True, help="Colab session name")
    parser.add_argument("--gpu", choices=GPU_CHOICES, default="L4", help="Colab GPU accelerator")
    parser.add_argument("--high-mem", action="store_true", default=False, help="request high-RAM machine shape")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Git repository URL cloned on the VM")
    parser.add_argument("--branch", default="main", help="Git branch checked out on the VM")
    parser.add_argument("--decks-path", type=Path, required=True, help="local cached decks.pt to upload")
    parser.add_argument("--backup-dir", type=Path, required=True, help="local folder receiving synced metrics/state")
    parser.add_argument(
        "--resume-from", type=Path, default=None, help="local run folder with checkpoints/resume-state.pt"
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="experiment folder under /content/runs")
    parser.add_argument("--run-id", required=True, help="run identifier under /content/runs/<experiment>")
    parser.add_argument(
        "--sync-interval-s",
        type=int,
        default=DEFAULT_SYNC_INTERVAL_S,
        help="seconds between periodic HTTP syncs (0 disables periodic sync)",
    )
    parser.add_argument("--keep-vm", action="store_true", default=False, help="keep Colab VM running on exit")
    parser.add_argument("--dry-run", action="store_true", default=False, help="print plan without calling colab")
    ns, extra = parser.parse_known_args(argv)
    forwarded = tuple(extra[1:] if extra and extra[0] == "--" else extra)
    return ColabRunConfig(
        session=ns.session,
        decks_path=ns.decks_path,
        backup_dir=ns.backup_dir,
        run_id=ns.run_id,
        gpu=_parse_gpu(ns.gpu),
        high_mem=ns.high_mem,
        repo_url=ns.repo_url,
        branch=ns.branch,
        resume_from=ns.resume_from,
        experiment=ns.experiment,
        sync_interval_s=ns.sync_interval_s,
        keep_vm=ns.keep_vm,
        dry_run=ns.dry_run,
        extra_args=forwarded,
    )


def main(argv: Sequence[str] | None = None) -> int:
    cfg = _parse_args(argv)
    return run_colab_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

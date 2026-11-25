#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Google Colab GPU training launcher with chunked resume, kernel keepalive, and sync.

Provisions a Colab runtime through `uvx --from google-colab-cli colab`, uploads cached
17Lands decks and optional resume artifacts (automatically chunking and SHA-256 verifying
files above 48 MiB so large `resume-state.pt` snapshots stay under Jupyter's HTTP limit),
bootstraps the repository over short-lived SSH, and launches `jace.train.full_data`
detached under `nohup`. A tunnel-free `colab exec` poll loop keeps the Colab kernel
active so the 3-hour idle-runtime reaper never fires, syncs `metrics.jsonl`,
`config.json`, and `checkpoints/resume-state.pt` on each sync cadence, and stops the VM
in cleanup unless `--keep-vm` is set.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    "DEFAULT_EXPERIMENT",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_REPO_URL",
    "DEFAULT_SYNC_INTERVAL_S",
    "DEFAULT_UPLOAD_CHUNK_BYTES",
    "GPU_CHOICES",
    "ColabRunConfig",
    "CommandResult",
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
DEFAULT_POLL_INTERVAL_S = 300
DEFAULT_SYNC_INTERVAL_S = 900
DEFAULT_UPLOAD_CHUNK_BYTES = 48 * 1024 * 1024

_COLAB_CMD: tuple[str, ...] = ("uvx", "--from", "google-colab-cli", "colab")
_REMOTE_DECKS_PATH = "/content/decks.pt"
_REMOTE_REPO_DIR = "/content/jace"
_REMOTE_RUNS_DIR = "/content/runs"
_REMOTE_PID_FILE = "/content/train.pid"
_REMOTE_EXIT_FILE = "/content/train.exit"
_REMOTE_LOG_FILE = "/content/train.log"
_RESUME_FILENAME = "resume-state.pt"
_CHECKPOINT_SUFFIXES: tuple[str, ...] = (".safetensors", ".policy.json")
_GPU_MAP: dict[str, GpuKind] = {"T4": "T4", "L4": "L4", "A100": "A100", "H100": "H100"}
_LOGGER = logging.getLogger(__name__)

_EXEC_TIMEOUT_S = 90.0
_REMOTE_EXEC_TIMEOUT_S = 45
_TRANSFER_TIMEOUT_S = 180.0
_SETUP_TIMEOUT_S = 900.0
_STOP_TIMEOUT_S = 60.0


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
    upload_chunk_bytes: int = DEFAULT_UPLOAD_CHUNK_BYTES
    keep_vm: bool = False
    dry_run: bool = False
    extra_args: tuple[str, ...] = ()


def _remote_run_dir(cfg: ColabRunConfig) -> str:
    return f"{_REMOTE_RUNS_DIR}/{cfg.experiment}/{cfg.run_id}"


def _run_subprocess(
    args: Sequence[str],
    *,
    check: bool = True,
    input_text: str | None = None,
    timeout: float | None = None,
) -> CommandResult:
    capture = input_text is not None
    completed = subprocess.run(
        tuple(args),
        check=check,
        input=input_text,
        text=capture,
        capture_output=capture,
        timeout=timeout,
    )
    return CommandResult(returncode=completed.returncode, stdout=completed.stdout or "")


def _validate_inputs(cfg: ColabRunConfig) -> None:
    if not cfg.decks_path.is_file():
        msg = f"decks file not found: {cfg.decks_path}"
        raise FileNotFoundError(msg)
    if cfg.poll_interval_s < 0 or cfg.sync_interval_s < 0:
        msg = f"intervals must be >= 0, got poll={cfg.poll_interval_s}, sync={cfg.sync_interval_s}"
        raise ValueError(msg)
    if cfg.upload_chunk_bytes <= 0:
        msg = f"upload_chunk_bytes must be > 0, got {cfg.upload_chunk_bytes}"
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
    inner_cmd = f"{shlex.join(train_args)}; rc=$?; echo $rc > {_REMOTE_EXIT_FILE}; exit $rc"
    lines = [
        "set -euo pipefail",
        'export PATH="$HOME/.local/bin:$PATH"',
        'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"',
        "if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi",
        f"if [ ! -d {shlex.quote(_REMOTE_REPO_DIR)}/.git ]; then\n{clone_cmd}\nfi",
        f"cd {shlex.quote(_REMOTE_REPO_DIR)}",
        "uv sync --frozen",
        f"rm -f {_REMOTE_EXIT_FILE}",
        f"nohup bash -lc {shlex.quote(inner_cmd)} >{_REMOTE_LOG_FILE} 2>&1 </dev/null &",
        f"PID=$!; echo $PID > {_REMOTE_PID_FILE}; sleep 2; kill -0 $PID; echo LAUNCHED pid=$PID",
    ]
    return "\n".join(lines) + "\n"


def _remote_poll_code(remote_run_dir: str) -> str:
    metrics_path = f"{remote_run_dir}/metrics.jsonl"
    return (
        "import os, pathlib, subprocess\n"
        f"pid_file, exit_file = pathlib.Path({_REMOTE_PID_FILE!r}), pathlib.Path({_REMOTE_EXIT_FILE!r})\n"
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
        'last = mp.read_text().splitlines()[-1][:140] if mp.exists() and mp.stat().st_size else ""\n'
        'print(f"ALIVE={alive} EXIT={exit_code} PID={pid} GPU=[{gpu}] LAST={last}")\n'
        f"lp = pathlib.Path({_REMOTE_LOG_FILE!r})\n"
        'if not alive and lp.exists(): print("LOG_TAIL:\\n" + lp.read_text(errors="ignore")[-1500:])\n'
    )


def _upload_file(
    cfg: ColabRunConfig,
    *,
    local_path: Path,
    remote_path: str,
    runner: CommandRunner,
) -> None:
    if local_path.stat().st_size <= cfg.upload_chunk_bytes:
        runner(
            (*_COLAB_CMD, "upload", "-s", cfg.session, str(local_path), remote_path),
            check=True,
            timeout=_TRANSFER_TIMEOUT_S,
        )
        return

    payload = local_path.read_bytes()
    expected_sha = hashlib.sha256(payload).hexdigest()
    remote_parts_dir = f"{remote_path}.parts"
    runner((*_COLAB_CMD, "ssh", "-s", cfg.session, "--", "mkdir", "-p", remote_parts_dir), check=True, timeout=60.0)
    with tempfile.TemporaryDirectory(prefix="jace-colab-chunk-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        chunk_size = cfg.upload_chunk_bytes
        for idx, offset in enumerate(range(0, len(payload), chunk_size)):
            part_path = tmp_root / f"part-{idx:04d}"
            part_path.write_bytes(payload[offset : offset + chunk_size])
            runner(
                (*_COLAB_CMD, "upload", "-s", cfg.session, str(part_path), f"{remote_parts_dir}/{part_path.name}"),
                check=True,
                timeout=_TRANSFER_TIMEOUT_S,
            )
            part_path.unlink()

    reassemble = (
        f"set -euo pipefail; cat {shlex.quote(remote_parts_dir)}/part-* > {shlex.quote(remote_path)}; "
        f"rm -rf {shlex.quote(remote_parts_dir)}; "
        f"actual=$(sha256sum {shlex.quote(remote_path)} | awk '{{print $1}}'); "
        f'test "$actual" = {shlex.quote(expected_sha)}'
    )
    runner(
        (*_COLAB_CMD, "ssh", "-s", cfg.session, "--", "bash", "-lc", reassemble),
        check=True,
        timeout=_TRANSFER_TIMEOUT_S,
    )


def _sync_once(cfg: ColabRunConfig, *, remote_run_dir: str, runner: CommandRunner) -> None:
    (cfg.backup_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    for remote_path, local_path in _sync_targets(cfg.backup_dir, remote_run_dir=remote_run_dir):
        cmd = (*_COLAB_CMD, "download", "-s", cfg.session, remote_path, str(local_path))
        try:
            result = runner(cmd, check=False, timeout=_TRANSFER_TIMEOUT_S)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            _LOGGER.warning("sync failed for %s: %s", remote_path, exc)
            continue
        if result.returncode != 0:
            _LOGGER.debug("sync skipped %s (exit %d)", remote_path, result.returncode)


def _upload_resume_files(cfg: ColabRunConfig, *, remote_run_dir: str, runner: CommandRunner) -> None:
    if cfg.resume_from is None:
        return
    # Jupyter's Contents API requires parent folders to exist before PUT uploads.
    runner(_prepare_resume_dirs_command(cfg.session, remote_run_dir=remote_run_dir), check=True, timeout=60.0)
    for local_path, remote_path in _resume_uploads(cfg.resume_from, remote_run_dir=remote_run_dir):
        _upload_file(cfg, local_path=local_path, remote_path=remote_path, runner=runner)


def _parse_remote_exit_code(stdout: str) -> int:
    for token in stdout.split():
        if token.startswith("EXIT="):
            raw = token.removeprefix("EXIT=")
            return int(raw) if raw.isdigit() else 1
    return 1


def _poll_once(
    exec_cmd: tuple[str, ...],
    *,
    poll_code: str,
    ticks: int,
    runner: CommandRunner,
) -> str:
    try:
        stdout = runner(exec_cmd, check=False, input_text=poll_code, timeout=_EXEC_TIMEOUT_S).stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        _LOGGER.warning("poll keepalive failed on tick %d: %s", ticks, exc)
        return ""
    if stdout:
        sys.stdout.write(stdout if stdout.endswith("\n") else f"{stdout}\n")
        sys.stdout.flush()
    return stdout


def _watch_remote_training(cfg: ColabRunConfig, *, remote_run_dir: str, runner: CommandRunner) -> int:
    poll_code = _remote_poll_code(remote_run_dir)
    exec_cmd = (*_COLAB_CMD, "exec", "-s", cfg.session, "--timeout", str(_REMOTE_EXEC_TIMEOUT_S))
    sync_every = max(1, cfg.sync_interval_s // max(1, cfg.poll_interval_s)) if cfg.sync_interval_s > 0 else 0
    ticks = 0
    while True:
        if cfg.poll_interval_s > 0:
            time.sleep(cfg.poll_interval_s)
        ticks += 1
        stdout = _poll_once(exec_cmd, poll_code=poll_code, ticks=ticks, runner=runner)
        if sync_every > 0 and ticks % sync_every == 0:
            _sync_once(cfg, remote_run_dir=remote_run_dir, runner=runner)
        if "ALIVE=False" in stdout:
            return _parse_remote_exit_code(stdout)


def _stop_session(session: str, *, runner: CommandRunner) -> None:
    stop_result = runner((*_COLAB_CMD, "stop", "-s", session), check=False, timeout=_STOP_TIMEOUT_S)
    if stop_result.returncode != 0:
        _LOGGER.warning("colab stop -s %s exited with %d", session, stop_result.returncode)
    runner((*_COLAB_CMD, "usage"), check=False, timeout=_STOP_TIMEOUT_S)


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
    commands.append((*_COLAB_CMD, "exec", "-s", cfg.session, "--timeout", str(_REMOTE_EXEC_TIMEOUT_S)))
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
    active_runner(_provision_command(cfg), check=True, timeout=_TRANSFER_TIMEOUT_S)

    try:
        _upload_file(cfg, local_path=cfg.decks_path, remote_path=_REMOTE_DECKS_PATH, runner=active_runner)
        _upload_resume_files(cfg, remote_run_dir=remote_run_dir, runner=active_runner)
        ssh_cmd = (*_COLAB_CMD, "ssh", "-s", cfg.session, "--", "bash", "-lc", remote_script)
        active_runner(ssh_cmd, check=True, timeout=_SETUP_TIMEOUT_S)
        try:
            return _watch_remote_training(cfg, remote_run_dir=remote_run_dir, runner=active_runner)
        finally:
            _sync_once(cfg, remote_run_dir=remote_run_dir, runner=active_runner)
    finally:
        if not cfg.keep_vm:
            _stop_session(cfg.session, runner=active_runner)


def _parse_gpu(raw: str) -> GpuKind:
    if raw not in _GPU_MAP:
        msg = f"unsupported GPU {raw!r}; expected one of {', '.join(GPU_CHOICES)}"
        raise ValueError(msg)
    return _GPU_MAP[raw]


def _parse_args(argv: Sequence[str] | None) -> ColabRunConfig:
    parser = argparse.ArgumentParser(
        description="Run jace.train.full_data on a Google Colab GPU with chunked resume, keepalive, and local sync.",
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
        "--poll-interval-s",
        type=int,
        default=DEFAULT_POLL_INTERVAL_S,
        help="seconds between colab exec keepalive/liveness polls",
    )
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
        poll_interval_s=ns.poll_interval_s,
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

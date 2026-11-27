#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Google Colab GPU training launcher with chunked resume, kernel keepalive, and sync.

Provisions a Colab runtime through `uvx --from google-colab-cli colab`, uploads cached
17Lands decks and optional resume artifacts via `scripts/_colab_remote.py`, bootstraps
the repository over short-lived SSH, and launches `jace.train.full_data` detached under
`nohup`. A tunnel-free `colab exec` poll loop keeps the Colab kernel active so the 3-hour
idle-runtime reaper never fires, syncs `metrics.jsonl`, `config.json`, and
`checkpoints/resume-state.pt` on each sync cadence, and stops the VM in cleanup unless
`--keep-vm` is set.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from _colab_remote import (
    COLAB_CMD,
    DEFAULT_EVAL_STALL_TIMEOUT_S,
    DEFAULT_EXPERIMENT,
    DEFAULT_MAX_STALL_RESTARTS,
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_REPO_URL,
    DEFAULT_STALL_TIMEOUT_S,
    DEFAULT_SYNC_INTERVAL_S,
    DEFAULT_UPLOAD_CHUNK_BYTES,
    GPU_CHOICES,
    REMOTE_DECKS_PATH,
    REMOTE_EXEC_TIMEOUT_S,
    RESUME_FILENAME,
    TRANSFER_TIMEOUT_S,
    ColabRunConfig,
    CommandResult,
    CommandRunner,
    GpuKind,
    build_remote_script,
    build_restart_script,
    extract_int_flag,
    format_dry_run_plan,
    is_checkpoint_artifact,
    is_checkpoint_artifact_name,
    provision_command,
    remote_poll_code,
    remote_run_dir,
    sync_metrics_only,
    sync_once,
    upload_file,
    upload_resume_files,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = (
    "DEFAULT_EXPERIMENT",
    "DEFAULT_MAX_STALL_RESTARTS",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_REPO_URL",
    "DEFAULT_STALL_TIMEOUT_S",
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

_GPU_MAP: dict[str, GpuKind] = {"T4": "T4", "L4": "L4", "G4": "G4", "A100": "A100", "H100": "H100"}
_LOGGER = logging.getLogger(__name__)

_EXEC_TIMEOUT_S = 90.0
_PROVISION_TIMEOUT_S = 420.0
_TRANSFER_TIMEOUT_S = TRANSFER_TIMEOUT_S
_SETUP_TIMEOUT_S = 900.0
_STOP_TIMEOUT_S = 60.0
_is_checkpoint_artifact = is_checkpoint_artifact
_sync_once = sync_once


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
        resume_state = cfg.resume_from / "checkpoints" / RESUME_FILENAME
        if not resume_state.is_file():
            msg = f"resume state not found: {resume_state}"
            raise FileNotFoundError(msg)


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


def _parse_poll_token(stdout: str, prefix: str) -> str | None:
    for item in stdout.split():
        if item.startswith(prefix):
            return item.removeprefix(prefix)
    return None


def _parse_remote_lines(stdout: str) -> int | None:
    raw = _parse_poll_token(stdout, "LINES=")
    return int(raw) if raw is not None and raw.isdigit() else None


def _parse_snapshot_names(stdout: str) -> tuple[str, ...]:
    raw = _parse_poll_token(stdout, "SNAPSHOTS=")
    if not raw:
        return ()
    return tuple(name for name in raw.split(",") if is_checkpoint_artifact_name(name))


def _is_in_eval_window(stdout: str, *, eval_every: int) -> bool:
    if eval_every <= 0:
        return False
    if _parse_poll_token(stdout, "LAST_KIND=") != "train":
        return False
    raw_u = _parse_poll_token(stdout, "LAST_UPDATE=")
    if raw_u is None or not raw_u.isdigit():
        return False
    update = int(raw_u)
    return update > 0 and update % eval_every == 0


def _restart_stalled_training(
    cfg: ColabRunConfig,
    *,
    remote_run_dir_path: str,
    restart_idx: int,
    runner: CommandRunner,
) -> None:
    _LOGGER.warning(
        "stall watchdog triggered (restart %d/%d); syncing metrics and restarting with shifted seed",
        restart_idx,
        cfg.max_stall_restarts,
    )
    sync_metrics_only(cfg, remote_run_dir_path=remote_run_dir_path, runner=runner)
    restart_script = build_restart_script(cfg, seed_offset=restart_idx * 1000)
    ssh_cmd = (*COLAB_CMD, "ssh", "-s", cfg.session, "--", "bash", "-lc", restart_script)
    runner(ssh_cmd, check=True, timeout=_SETUP_TIMEOUT_S)


@dataclass(slots=True)
class _StallTracker:
    last_lines: int | None
    last_progress: float
    eval_every: int
    restarts: int = 0

    def observe(
        self,
        stdout: str,
        *,
        cfg: ColabRunConfig,
        remote_run_dir_path: str,
        runner: CommandRunner,
        clock: Callable[[], float],
    ) -> int | None:
        lines = _parse_remote_lines(stdout)
        if lines is not None and (self.last_lines is None or lines > self.last_lines):
            self.last_lines = lines
            self.last_progress = clock()
            return None
        threshold = (
            cfg.eval_stall_timeout_s if _is_in_eval_window(stdout, eval_every=self.eval_every) else cfg.stall_timeout_s
        )
        if threshold <= 0 or (clock() - self.last_progress) < threshold:
            return None
        if self.restarts >= cfg.max_stall_restarts:
            _LOGGER.error("stall watchdog exhausted %d restarts; stopping VM", cfg.max_stall_restarts)
            return 124
        self.restarts += 1
        _restart_stalled_training(
            cfg,
            remote_run_dir_path=remote_run_dir_path,
            restart_idx=self.restarts,
            runner=runner,
        )
        self.last_progress = clock()
        return None


def _watch_remote_training(
    cfg: ColabRunConfig,
    *,
    remote_run_dir_path: str,
    runner: CommandRunner,
    clock: Callable[[], float] = time.monotonic,
    discovered_snapshots: set[str] | None = None,
) -> int:
    poll_code = remote_poll_code(remote_run_dir_path)
    exec_cmd = (*COLAB_CMD, "exec", "-s", cfg.session, "--timeout", str(REMOTE_EXEC_TIMEOUT_S))
    sync_every = max(1, cfg.sync_interval_s // max(1, cfg.poll_interval_s)) if cfg.sync_interval_s > 0 else 0
    ticks = 0
    snapshots = set() if discovered_snapshots is None else discovered_snapshots
    eval_every = extract_int_flag(cfg.extra_args, flag="--eval-every", default=40)
    tracker = _StallTracker(last_lines=None, last_progress=clock(), eval_every=eval_every)
    while True:
        if cfg.poll_interval_s > 0:
            time.sleep(cfg.poll_interval_s)
        ticks += 1
        stdout = _poll_once(exec_cmd, poll_code=poll_code, ticks=ticks, runner=runner)
        snapshots.update(_parse_snapshot_names(stdout))
        if sync_every > 0 and ticks % sync_every == 0:
            sync_once(cfg, remote_run_dir=remote_run_dir_path, runner=runner, snapshot_names=snapshots)
        if "ALIVE=False" in stdout:
            return _parse_remote_exit_code(stdout)
        stall_exit = tracker.observe(
            stdout,
            cfg=cfg,
            remote_run_dir_path=remote_run_dir_path,
            runner=runner,
            clock=clock,
        )
        if stall_exit is not None:
            return stall_exit


def _stop_session(session: str, *, runner: CommandRunner) -> None:
    stop_result = runner((*COLAB_CMD, "stop", "-s", session), check=False, timeout=_STOP_TIMEOUT_S)
    if stop_result.returncode != 0:
        _LOGGER.warning("colab stop -s %s exited with %d", session, stop_result.returncode)
    runner((*COLAB_CMD, "usage"), check=False, timeout=_STOP_TIMEOUT_S)


def run_colab_training(
    cfg: ColabRunConfig,
    *,
    runner: CommandRunner | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    _validate_inputs(cfg)
    if cfg.dry_run:
        sys.stdout.write(format_dry_run_plan(cfg))
        return 0

    active_runner: CommandRunner = _run_subprocess if runner is None else runner
    run_dir = remote_run_dir(cfg)
    remote_script = build_remote_script(cfg)
    (cfg.backup_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if not cfg.reuse_session:
        active_runner(provision_command(cfg), check=True, timeout=_PROVISION_TIMEOUT_S)

    try:
        if not cfg.reuse_session:
            upload_file(cfg, local_path=cfg.decks_path, remote_path=REMOTE_DECKS_PATH, runner=active_runner)
            upload_resume_files(cfg, remote_run_dir_path=run_dir, runner=active_runner)
        ssh_cmd = (*COLAB_CMD, "ssh", "-s", cfg.session, "--", "bash", "-lc", remote_script)
        active_runner(ssh_cmd, check=True, timeout=_SETUP_TIMEOUT_S)
        discovered_snapshots: set[str] = set()
        try:
            return _watch_remote_training(
                cfg,
                remote_run_dir_path=run_dir,
                runner=active_runner,
                clock=clock,
                discovered_snapshots=discovered_snapshots,
            )
        finally:
            sync_once(cfg, remote_run_dir=run_dir, runner=active_runner, snapshot_names=discovered_snapshots)
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
    parser.add_argument(
        "--stall-timeout-s",
        type=int,
        default=DEFAULT_STALL_TIMEOUT_S,
        help="seconds with unchanged metrics.jsonl during training before auto-restarting (0 disables)",
    )
    parser.add_argument(
        "--eval-stall-timeout-s",
        type=int,
        default=DEFAULT_EVAL_STALL_TIMEOUT_S,
        help="seconds allowed at eval_every boundaries before auto-restarting (0 disables)",
    )
    parser.add_argument(
        "--max-stall-restarts",
        type=int,
        default=DEFAULT_MAX_STALL_RESTARTS,
        help="max automatic in-place stall restarts before stopping the VM",
    )
    parser.add_argument(
        "--reuse-session",
        action="store_true",
        default=False,
        help="attach to an already-provisioned Colab session without re-provisioning or re-uploading",
    )
    parser.add_argument("--keep-vm", action="store_true", default=False, help="keep Colab VM running on exit")
    parser.add_argument("--dry-run", action="store_true", default=False, help="print plan without calling colab")
    parser.add_argument(
        "--preflight-first",
        action="store_true",
        default=False,
        help="run synchronous --preflight smoke gate over SSH before detaching training",
    )
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
        stall_timeout_s=ns.stall_timeout_s,
        eval_stall_timeout_s=ns.eval_stall_timeout_s,
        max_stall_restarts=ns.max_stall_restarts,
        reuse_session=ns.reuse_session,
        keep_vm=ns.keep_vm,
        dry_run=ns.dry_run,
        preflight_first=ns.preflight_first,
        extra_args=forwarded,
    )


def main(argv: Sequence[str] | None = None) -> int:
    cfg = _parse_args(argv)
    return run_colab_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

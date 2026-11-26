"""Failure-mode invariant tests for the standalone `scripts/colab_run.py` launcher.

Uses an injected `_RecordingRunner` strictly to guard local lifecycle and billing
invariants: fail-fast input checks before `colab new`, guaranteed VM teardown without
clobbering local backups when setup fails, chunked SHA-256 uploads for large resume
states, tunnel-free `colab exec` keepalive polling with timeout recovery and exit-code
propagation, and side-effect-free `--dry-run` / `--` flag forwarding. Live
`google-colab-cli` flag compatibility is verified against real Colab runtimes rather
than mocked subprocesses.
"""

from __future__ import annotations

import hashlib
import subprocess
from typing import TYPE_CHECKING

import pytest

from colab_run import ColabRunConfig, CommandResult, main, run_colab_training

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class _RecordingRunner:
    def __init__(
        self,
        *,
        fail_on_subcommand: str | None = None,
        exec_responses: Sequence[CommandResult | Exception] = (CommandResult(0, "ALIVE=False EXIT=0\n"),),
    ) -> None:
        self.calls: list[tuple[tuple[str, ...], bool, str | None, float | None]] = []
        self._fail_on_subcommand = fail_on_subcommand
        self._exec_responses = list(exec_responses)

    def __call__(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        input_text: str | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        cmd = tuple(args)
        self.calls.append((cmd, check, input_text, timeout))
        subcommand = cmd[4] if len(cmd) > 4 else ""
        if self._fail_on_subcommand == subcommand:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        if subcommand == "exec":
            next_resp = (
                self._exec_responses.pop(0) if self._exec_responses else CommandResult(0, "ALIVE=False EXIT=0\n")
            )
            if isinstance(next_resp, Exception):
                raise next_resp
            return next_resp
        return CommandResult(returncode=0)

    def subcommands(self) -> list[str]:
        return [cmd[4] for cmd, _, _, _ in self.calls]


def _make_decks(tmp_path: Path) -> Path:
    decks = tmp_path / "decks.pt"
    decks.write_bytes(b"decks")
    return decks


def test_missing_decks_or_resume_state_fails_before_provisioning(tmp_path: Path) -> None:
    runner = _RecordingRunner()
    missing_decks_cfg = ColabRunConfig(
        session="exp-a",
        decks_path=tmp_path / "missing.pt",
        backup_dir=tmp_path / "backup",
        run_id="r1",
    )
    with pytest.raises(FileNotFoundError):
        run_colab_training(missing_decks_cfg, runner=runner)
    assert runner.calls == []

    decks = _make_decks(tmp_path)
    empty_resume_dir = tmp_path / "incomplete-run"
    empty_resume_dir.mkdir()
    bad_resume_cfg = ColabRunConfig(
        session="exp-a",
        decks_path=decks,
        backup_dir=tmp_path / "backup",
        run_id="r1",
        resume_from=empty_resume_dir,
    )
    with pytest.raises(FileNotFoundError):
        run_colab_training(bad_resume_cfg, runner=runner)
    assert runner.calls == []


def test_setup_failure_stops_vm_without_syncing_and_download_errors_still_stop(tmp_path: Path) -> None:
    decks = _make_decks(tmp_path)
    stop_runner = _RecordingRunner(fail_on_subcommand="ssh")
    stop_cfg = ColabRunConfig(
        session="fail-stop",
        decks_path=decks,
        backup_dir=tmp_path / "backup-stop",
        run_id="r-fail",
        poll_interval_s=0,
        sync_interval_s=0,
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_colab_training(stop_cfg, runner=stop_runner)
    assert stop_runner.subcommands() == ["new", "upload", "ssh", "stop", "usage"]

    keep_runner = _RecordingRunner(fail_on_subcommand="ssh")
    keep_cfg = ColabRunConfig(
        session="fail-keep",
        decks_path=decks,
        backup_dir=tmp_path / "backup-keep",
        run_id="r-keep",
        poll_interval_s=0,
        sync_interval_s=0,
        keep_vm=True,
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_colab_training(keep_cfg, runner=keep_runner)
    assert keep_runner.subcommands() == ["new", "upload", "ssh"]

    sync_runner = _RecordingRunner(fail_on_subcommand="download")
    sync_cfg = ColabRunConfig(
        session="sync-err",
        decks_path=decks,
        backup_dir=tmp_path / "backup-sync",
        run_id="r-sync",
        poll_interval_s=0,
        sync_interval_s=0,
    )
    assert run_colab_training(sync_cfg, runner=sync_runner) == 0
    assert sync_runner.subcommands() == [
        "new",
        "upload",
        "ssh",
        "exec",
        "download",
        "download",
        "download",
        "stop",
        "usage",
    ]


def test_resume_from_prepares_dirs_chunks_large_state_and_verifies_sha256(tmp_path: Path) -> None:
    decks = _make_decks(tmp_path)
    resume_dir = tmp_path / "prior-run"
    ckpts_dir = resume_dir / "checkpoints"
    ckpts_dir.mkdir(parents=True)
    (resume_dir / "config.json").write_text("{}\n")
    (resume_dir / "metrics.jsonl").write_text('{"update": 40}\n')
    large_state = b"0123456789" * 5  # 50 bytes -> 3 chunks at chunk_bytes=20
    (ckpts_dir / "resume-state.pt").write_bytes(large_state)
    (ckpts_dir / "iter-0040.safetensors").write_bytes(b"wt")
    (ckpts_dir / "iter-0040.policy.json").write_text("{}")

    runner = _RecordingRunner()
    cfg = ColabRunConfig(
        session="tla-resume",
        decks_path=decks,
        backup_dir=tmp_path / "backup",
        run_id="run-02",
        resume_from=resume_dir,
        poll_interval_s=0,
        sync_interval_s=0,
        upload_chunk_bytes=20,
    )
    assert run_colab_training(cfg, runner=runner) == 0

    expected_sha = hashlib.sha256(large_state).hexdigest()
    part_uploads = [
        cmd for cmd, _, _, _ in runner.calls if cmd[4] == "upload" and "resume-state.pt.parts/part-" in cmd[8]
    ]
    assert len(part_uploads) == 3

    ssh_cmds = [cmd for cmd, _, _, _ in runner.calls if cmd[4] == "ssh"]
    assert any("forensics" in " ".join(cmd) and "games" in " ".join(cmd) for cmd in ssh_cmds)
    assert any(expected_sha in cmd[-1] and "sha256sum" in cmd[-1] for cmd in ssh_cmds)
    assert any("--decks-path /content/decks.pt --resume" in cmd[-1] for cmd in ssh_cmds)


def test_poll_loop_recovers_timeout_syncs_periodically_and_returns_remote_exit(tmp_path: Path) -> None:
    decks = _make_decks(tmp_path)
    runner = _RecordingRunner(
        exec_responses=[
            subprocess.TimeoutExpired(cmd=("colab", "exec"), timeout=90.0),
            CommandResult(0, "ALIVE=True EXIT= PID=42 GPU=[7000 MiB, 23034 MiB, 85 %] LAST={}\n"),
            CommandResult(0, "ALIVE=False EXIT=7 PID=42 GPU=[0 MiB, 23034 MiB, 0 %] LAST={}\n"),
        ]
    )
    cfg = ColabRunConfig(
        session="poll-test",
        decks_path=decks,
        backup_dir=tmp_path / "backup-poll",
        run_id="r-poll",
        poll_interval_s=0,
        sync_interval_s=1,
    )
    assert run_colab_training(cfg, runner=runner) == 7
    exec_calls = [call for call in runner.calls if call[0][4] == "exec"]
    assert len(exec_calls) == 3
    assert all(timeout == 90.0 for _, _, _, timeout in exec_calls)
    download_calls = [call for call in runner.calls if call[0][4] == "download"]
    assert len(download_calls) == 12  # 3 periodic ticks * 3 files + final sync * 3 files
    assert runner.subcommands()[-2:] == ["stop", "usage"]


def test_cli_dry_run_forwards_after_double_dash_and_executes_zero_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    decks = _make_decks(tmp_path)
    backup_dir = tmp_path / "backup-dry"
    runner = _RecordingRunner()
    dry_cfg = ColabRunConfig(
        session="dry-sess",
        decks_path=decks,
        backup_dir=backup_dir,
        run_id="dry-01",
        dry_run=True,
        extra_args=("--updates", "10"),
    )
    assert run_colab_training(dry_cfg, runner=runner) == 0
    assert runner.calls == []
    assert not backup_dir.exists()
    capsys.readouterr()

    code = main(
        [
            "--session",
            "cli-dry",
            "--decks-path",
            str(decks),
            "--backup-dir",
            str(backup_dir),
            "--run-id",
            "cli-01",
            "--dry-run",
            "--",
            "--updates",
            "25",
            "--eval-every",
            "5",
        ]
    )
    assert code == 0
    assert not backup_dir.exists()
    output = capsys.readouterr().out
    assert 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' in output
    assert "nohup bash -lc" in output
    assert "uv run python -m jace.train.full_data" in output
    assert "--updates 25 --eval-every 5" in output
    assert " -- --updates" not in output


def test_stall_watchdog_restarts_with_shifted_seed_and_stops_on_exhaustion(tmp_path: Path) -> None:
    decks = _make_decks(tmp_path)
    clock_val = [0.0]

    def fake_clock() -> float:
        clock_val[0] += 100.0
        return clock_val[0]

    # First tick establishes LINES=10; second tick stays at LINES=10 past stall_timeout_s=50 -> restarts with seed 1005;
    # third tick finishes cleanly.
    recover_runner = _RecordingRunner(
        exec_responses=[
            CommandResult(0, "ALIVE=True EXIT= PID=42 LINES=10 GPU=[1000 MiB, 97887 MiB, 0 %] LAST={}\n"),
            CommandResult(0, "ALIVE=True EXIT= PID=42 LINES=10 GPU=[1000 MiB, 97887 MiB, 0 %] LAST={}\n"),
            CommandResult(0, "ALIVE=False EXIT=0 PID=99 LINES=12 GPU=[0 MiB, 97887 MiB, 0 %] LAST={}\n"),
        ]
    )
    recover_cfg = ColabRunConfig(
        session="stall-recover",
        decks_path=decks,
        backup_dir=tmp_path / "backup-recover",
        run_id="r-rec",
        poll_interval_s=0,
        sync_interval_s=0,
        stall_timeout_s=50,
        max_stall_restarts=2,
        extra_args=("--seed", "5"),
    )
    assert run_colab_training(recover_cfg, runner=recover_runner, clock=fake_clock) == 0
    ssh_cmds = [cmd[-1] for cmd, _, _, _ in recover_runner.calls if cmd[4] == "ssh"]
    assert len(ssh_cmds) == 2
    assert "pkill -9 -f jace.train.full_data" in ssh_cmds[1]
    assert "--resume" in ssh_cmds[1]
    assert "--seed 1005" in ssh_cmds[1]

    # Exhausting max_stall_restarts=0 stops the VM and returns exit code 124
    exhaust_runner = _RecordingRunner(
        exec_responses=[
            CommandResult(0, "ALIVE=True EXIT= PID=42 LINES=10 GPU=[1000 MiB, 97887 MiB, 0 %] LAST={}\n"),
            CommandResult(0, "ALIVE=True EXIT= PID=42 LINES=10 GPU=[1000 MiB, 97887 MiB, 0 %] LAST={}\n"),
        ]
    )
    exhaust_cfg = ColabRunConfig(
        session="stall-exhaust",
        decks_path=decks,
        backup_dir=tmp_path / "backup-exhaust",
        run_id="r-exh",
        poll_interval_s=0,
        sync_interval_s=0,
        stall_timeout_s=50,
        max_stall_restarts=0,
    )
    assert run_colab_training(exhaust_cfg, runner=exhaust_runner, clock=fake_clock) == 124
    assert exhaust_runner.subcommands()[-2:] == ["stop", "usage"]

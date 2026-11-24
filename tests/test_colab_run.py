"""Failure-mode invariant tests for the standalone `scripts/colab_run.py` launcher.

Uses an injected `_RecordingRunner` strictly to guard local lifecycle and billing
invariants (fail-fast input checks before `colab new`, guaranteed final sync and VM
teardown on SSH or transient download failures unless `--keep-vm`, remote directory
preparation before resume uploads, and side-effect-free `--dry-run` / `--` flag
forwarding). Live `google-colab-cli` flag compatibility is verified against real
sessions rather than mocked subprocesses.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from colab_run import ColabRunConfig, main, run_colab_training

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class _RecordingRunner:
    def __init__(self, *, fail_on_subcommand: str | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], bool]] = []
        self._fail_on_subcommand = fail_on_subcommand

    def __call__(self, args: Sequence[str], *, check: bool = True) -> int:
        cmd = tuple(args)
        self.calls.append((cmd, check))
        subcommand = cmd[4] if len(cmd) > 4 else ""
        if self._fail_on_subcommand == subcommand:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return 0

    def subcommands(self) -> list[str]:
        return [cmd[4] for cmd, _ in self.calls]


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


def test_ssh_or_download_failure_still_syncs_and_stops_unless_keep_vm(tmp_path: Path) -> None:
    decks = _make_decks(tmp_path)
    stop_runner = _RecordingRunner(fail_on_subcommand="ssh")
    stop_cfg = ColabRunConfig(
        session="fail-stop",
        decks_path=decks,
        backup_dir=tmp_path / "backup-stop",
        run_id="r-fail",
        sync_interval_s=0,
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_colab_training(stop_cfg, runner=stop_runner)
    assert stop_runner.subcommands() == ["new", "upload", "ssh", "download", "download", "download", "stop", "usage"]

    keep_runner = _RecordingRunner(fail_on_subcommand="ssh")
    keep_cfg = ColabRunConfig(
        session="fail-keep",
        decks_path=decks,
        backup_dir=tmp_path / "backup-keep",
        run_id="r-keep",
        sync_interval_s=0,
        keep_vm=True,
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_colab_training(keep_cfg, runner=keep_runner)
    assert keep_runner.subcommands() == ["new", "upload", "ssh", "download", "download", "download"]

    sync_runner = _RecordingRunner(fail_on_subcommand="download")
    sync_cfg = ColabRunConfig(
        session="sync-err",
        decks_path=decks,
        backup_dir=tmp_path / "backup-sync",
        run_id="r-sync",
        sync_interval_s=0,
    )
    assert run_colab_training(sync_cfg, runner=sync_runner) == 0
    assert sync_runner.subcommands() == ["new", "upload", "ssh", "download", "download", "download", "stop", "usage"]


def test_resume_from_prepares_remote_dirs_before_uploads_and_appends_resume(tmp_path: Path) -> None:
    decks = _make_decks(tmp_path)
    resume_dir = tmp_path / "prior-run"
    ckpts_dir = resume_dir / "checkpoints"
    ckpts_dir.mkdir(parents=True)
    (resume_dir / "config.json").write_text("{}\n")
    (resume_dir / "metrics.jsonl").write_text('{"update": 40}\n')
    (ckpts_dir / "resume-state.pt").write_bytes(b"state")
    (ckpts_dir / "iter-0040.safetensors").write_bytes(b"weights")
    (ckpts_dir / "iter-0040.policy.json").write_text("{}\n")

    runner = _RecordingRunner()
    cfg = ColabRunConfig(
        session="tla-resume",
        decks_path=decks,
        backup_dir=tmp_path / "backup",
        run_id="run-02",
        resume_from=resume_dir,
        sync_interval_s=0,
    )
    run_colab_training(cfg, runner=runner)

    mkdir_idx = next(
        idx for idx, (cmd, _) in enumerate(runner.calls) if cmd[4] == "ssh" and cmd[8:10] == ("mkdir", "-p")
    )
    resume_upload_indices = [
        idx for idx, (cmd, _) in enumerate(runner.calls) if cmd[4] == "upload" and cmd[8].startswith("/content/runs/")
    ]
    assert len(resume_upload_indices) == 5
    assert all(idx > mkdir_idx for idx in resume_upload_indices)

    ssh_calls = [cmd for cmd, _ in runner.calls if cmd[4] == "ssh"]
    assert len(ssh_calls) == 2
    assert "--decks-path /content/decks.pt --resume" in ssh_calls[1][10]


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
    assert "exec uv run python -m jace.train.full_data" in output
    assert "--updates 25 --eval-every 5" in output
    assert " -- --updates" not in output

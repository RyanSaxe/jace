from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
import torch

from colab_run import (
    _TRANSFER_TIMEOUT_S,
    ColabRunConfig,
    CommandResult,
    _is_checkpoint_artifact,
    _sync_once,
    build_remote_script,
    run_colab_training,
)
from jace.collect.agent import build_agent
from jace.eval.puzzles import (
    PUZZLE_MIN_MAX_PLIES,
    PuzzleRecord,
    PuzzleSource,
    append_puzzle,
    load_puzzles,
    passes_value_prefilter,
    puzzles_path,
    score_puzzles,
    summarize_paired_outcomes,
    summarize_puzzle_scores,
    verify_puzzle_candidate,
)
from jace.rl.ema import EmaWeights
from jace.storage.gameplay import (
    DENSE_EARLY_UPDATES,
    create_run,
    load_weight_snapshot,
    save_weight_snapshot,
    should_save_weight_snapshot,
)
from jace.train.full_data import (
    FullDataConfig,
    _collect_preflight_failures,
    _parse_args,
    run_preflight,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class _FakeRunner:
    def __init__(self, exec_responses: Sequence[CommandResult] = ()) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._exec_responses = list(exec_responses)

    def __call__(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        input_text: str | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del check, input_text, timeout
        cmd = tuple(args)
        self.calls.append(cmd)
        if len(cmd) > 4 and cmd[4] == "exec":
            return self._exec_responses.pop(0) if self._exec_responses else CommandResult(0, "ALIVE=False EXIT=0\n")
        return CommandResult(0)


@dataclass
class _StubActionSpace:
    actions: tuple[int, ...]


@dataclass
class _StubAgent:
    player_index: int


@dataclass
class _StubObs:
    agent: _StubAgent
    action_space: _StubActionSpace


class _StubPairedEnv:
    def __init__(self, *, win_probs_by_action: dict[int, float]) -> None:
        self._win_probs = win_probs_by_action
        self._seed = 0
        self._winner: int | None = None
        self._done = False
        self.clones_created = 0

    def clone_env(self) -> _StubPairedEnv:
        self.clones_created += 1
        return _StubPairedEnv(win_probs_by_action=self._win_probs)

    def determinize(self, seed: int, _seat: int, /) -> None:
        self._seed = seed

    def step(self, action: int, /) -> tuple[_StubObs, float, bool, bool, dict[str, object]]:
        rng = random.Random(self._seed)
        draw = rng.random()
        self._winner = 0 if draw < self._win_probs[action] else 1
        self._done = True
        return _StubObs(_StubAgent(0), _StubActionSpace((0, 1))), 0.0, True, False, {}

    def is_game_over(self) -> bool:
        return self._done

    def winner_index(self) -> int | None:
        return self._winner


def _sample_puzzle(puzzle_id: str = "puz-001", *, source: PuzzleSource = "policy_disagreement") -> PuzzleRecord:
    return PuzzleRecord(
        puzzle_id=puzzle_id,
        format_id="tla-limited-0.2.6",
        source=source,
        run_id="flagship",
        iteration=2000,
        game_idx=3,
        step=41,
        turn=6,
        phase="Combat",
        step_name="DeclareAttackers",
        decision_kind="DECLARE_ATTACKER",
        hero_seat=0,
        seed=9901,
        hero_summary="life=14 hand=2 lands=5 creatures=3",
        opp_summary="life=11 hand=1 lands=5 creatures=2",
        key_cards=("Bear", "Dragon"),
        root_value_v=0.58,
        best_action_index=1,
        best_action_label="Attack with Dragon",
        alt_action_index=0,
        alt_action_label="Don't attack (Dragon)",
        best_playout_wr=0.78,
        alt_playout_wr=0.50,
        playouts_per_action=64,
        max_plies=240,
        margin=0.28,
        paired_ci_low=0.14,
        paired_ci_high=0.42,
    )


def _write_tiny_decks_pt(path: Path) -> Path:
    names = ["Plains", "Mountain", "Forest", "Island", "Swamp"]
    decks = torch.tensor(
        [
            [20, 20, 0, 0, 0],
            [0, 20, 20, 0, 0],
            [20, 0, 20, 0, 0],
            [0, 0, 20, 20, 0],
            [20, 0, 0, 20, 0],
            [0, 20, 0, 0, 20],
            [0, 0, 20, 0, 20],
            [20, 0, 0, 0, 20],
            [0, 20, 0, 20, 0],
            [0, 0, 0, 20, 20],
        ],
        dtype=torch.uint8,
    )
    split_key = torch.tensor([10, 50, 150, 200, 300, 400, 500, 600, 700, 800], dtype=torch.int16)
    torch.save(
        {
            "names": names,
            "tensors": {
                "decks": decks,
                "pools": decks.clone(),
                "wins": torch.zeros(10, dtype=torch.int16),
                "losses": torch.zeros(10, dtype=torch.int16),
                "split_key": split_key,
            },
        },
        path,
    )
    return path


def test_should_save_weight_snapshot_schedule_and_path(tmp_path: Path) -> None:
    expected_early = frozenset({25, 50, 100, 150, 200, 300, 400})
    assert expected_early == DENSE_EARLY_UPDATES
    for update in (25, 50, 100, 150, 200, 300, 400, 500, 1000):
        assert should_save_weight_snapshot(update, dense_early=True)
    for update in (0, 1, 24, 26, 75, 125, 250, 350, 450):
        assert not should_save_weight_snapshot(update, dense_early=True)
    for update in (0, 25, 50, 150):
        assert not should_save_weight_snapshot(update, dense_early=False)
    for update in (100, 200, 300, 400, 500):
        assert should_save_weight_snapshot(update, dense_early=False)

    layout = create_run(runs_dir=tmp_path, experiment="exp", run_id="r-path", config={"ok": True})
    assert layout.weight_snapshot_path(25) == layout.checkpoints_dir / "weights-u00025.pt"


def test_save_and_load_weight_snapshot_bf16_and_buffer_dtypes(tmp_path: Path) -> None:
    agent, _ = build_agent(hidden_dim=16, attention_layers=1)
    agent.register_buffer("test_int_buf", torch.tensor([1, 2, 3], dtype=torch.int64))
    agent.register_buffer("test_bool_buf", torch.tensor([True, False], dtype=torch.bool))
    ema = EmaWeights(agent, decay=0.99)
    with torch.no_grad():
        for param in agent.parameters():
            param.add_(0.05)
    ema.update(agent)

    layout = create_run(runs_dir=tmp_path, experiment="exp", run_id="r-snap", config={"ok": True})
    snap_path = save_weight_snapshot(layout, iteration=25, agent=agent, ema=ema)
    assert snap_path == layout.weight_snapshot_path(25)
    assert snap_path.is_file()
    assert not snap_path.with_suffix(".pt.partial").exists()

    loaded = load_weight_snapshot(snap_path)
    assert set(loaded) == {"iteration", "live_state_dict", "ema_state_dict"}
    assert "optimizer" not in loaded
    assert loaded["iteration"] == 25

    live_sd = loaded["live_state_dict"]
    ema_sd = loaded["ema_state_dict"]
    assert isinstance(live_sd, dict)
    assert isinstance(ema_sd, dict)
    assert live_sd["test_int_buf"].dtype == torch.int64
    assert live_sd["test_bool_buf"].dtype == torch.bool
    assert ema_sd["test_int_buf"].dtype == torch.int64
    assert ema_sd["test_bool_buf"].dtype == torch.bool
    for key, tensor in live_sd.items():
        orig = agent.state_dict()[key]
        assert tensor.device.type == "cpu"
        if orig.is_floating_point():
            assert tensor.dtype == torch.bfloat16
        else:
            assert tensor.dtype == orig.dtype

    fresh, _ = build_agent(hidden_dim=16, attention_layers=1)
    fresh.register_buffer("test_int_buf", torch.zeros(3, dtype=torch.int64))
    fresh.register_buffer("test_bool_buf", torch.zeros(2, dtype=torch.bool))
    fresh.load_state_dict(live_sd)
    assert torch.equal(fresh.get_buffer("test_int_buf"), torch.tensor([1, 2, 3], dtype=torch.int64))
    assert torch.equal(fresh.get_buffer("test_bool_buf"), torch.tensor([True, False], dtype=torch.bool))

    no_ema_path = save_weight_snapshot(layout, iteration=50, agent=agent, ema=None)
    assert load_weight_snapshot(no_ema_path)["ema_state_dict"] is None


def test_full_data_cli_flags_preflight_validation_and_smoke_gate(tmp_path: Path) -> None:
    parsed = _parse_args(["--run-id", "smoke", "--no-dense-early-checkpoints", "--preflight"])
    assert parsed.dense_early_checkpoints is False
    assert parsed.preflight is True
    assert _parse_args(["--run-id", "default-dense"]).dense_early_checkpoints is True

    assert (
        _collect_preflight_failures(
            updates_run=2,
            weights_finite=True,
            losses_finite=True,
            max_grad_norm_seen=0.5,
            max_grad_norm_limit=2.67,
            max_collection_kl=0.05,
            peak_vram_gb=8.0,
            vram_budget_gb=24.0,
            tps=1200.0,
            checkpoint_reload_ok=True,
        )
        == ()
    )
    bad_failures = _collect_preflight_failures(
        updates_run=1,
        weights_finite=False,
        losses_finite=False,
        max_grad_norm_seen=5.0,
        max_grad_norm_limit=2.67,
        max_collection_kl=0.75,
        peak_vram_gb=22.5,
        vram_budget_gb=24.0,
        tps=0.0,
        checkpoint_reload_ok=False,
    )
    assert len(bad_failures) == 8

    decks_path = _write_tiny_decks_pt(tmp_path / "decks.pt")
    runs_dir = tmp_path / "runs"
    cfg = FullDataConfig(
        runs_dir=runs_dir,
        experiment="preflight-exp",
        run_id="smoke-01",
        decks_path=decks_path,
        num_envs=2,
        steps_per_env=8,
        minibatch_size=16,
        hidden_dim=16,
        attention_layers=1,
        max_grad_norm=50.0,
        ema=0.99,
        collect_device="cpu",
        learn_device="cpu",
        precision="float32",
        live_tokens=False,
        preflight=True,
        seed=7,
    )
    report = run_preflight(cfg)
    assert report.passed, f"preflight failed: {report.failures}"
    assert report.updates_run == 2
    assert report.checkpoint_reload_ok is True
    assert report.tps > 0.0
    assert not runs_dir.exists()


def test_colab_run_preflight_first_timeout_and_incremental_snapshot_sync(tmp_path: Path) -> None:
    assert _TRANSFER_TIMEOUT_S == 600.0

    decks = tmp_path / "decks.pt"
    decks.write_bytes(b"decks")
    backup_dir = tmp_path / "backup"
    ckpts = backup_dir / "checkpoints"
    ckpts.mkdir(parents=True)
    snap25 = ckpts / "weights-u00025.pt"
    snap25.write_bytes(b"already-local")
    resume_file = ckpts / "resume-state.pt"
    resume_file.write_bytes(b"resume")
    tmp_file = ckpts / "weights-u00050.pt.partial"
    tmp_file.write_bytes(b"partial")

    assert _is_checkpoint_artifact(snap25)
    assert not _is_checkpoint_artifact(resume_file)
    assert not _is_checkpoint_artifact(tmp_file)

    cfg = ColabRunConfig(
        session="s-pre",
        decks_path=decks,
        backup_dir=backup_dir,
        run_id="r-pre",
        preflight_first=True,
        poll_interval_s=0,
        sync_interval_s=1,
        extra_args=("--dense-early-checkpoints", "--updates", "100"),
    )
    script = build_remote_script(cfg)
    preflight_idx = script.index("--preflight")
    nohup_idx = script.index("nohup bash -lc")
    assert preflight_idx < nohup_idx
    assert "uv run python -m jace.train.full_data" in script[:nohup_idx]

    no_pre_cfg = ColabRunConfig(session="s-no", decks_path=decks, backup_dir=backup_dir, run_id="r")
    assert "--preflight" not in build_remote_script(no_pre_cfg)

    runner = _FakeRunner()
    _sync_once(
        cfg,
        remote_run_dir="/content/runs/v2-colab/r-pre",
        runner=runner,
        snapshot_names=("weights-u00025.pt", "weights-u00050.pt", "resume-state.pt"),
    )
    downloaded_remotes = [cmd[7] for cmd in runner.calls if len(cmd) > 7 and cmd[4] == "download"]
    assert "/content/runs/v2-colab/r-pre/checkpoints/weights-u00050.pt" in downloaded_remotes
    assert "/content/runs/v2-colab/r-pre/checkpoints/weights-u00025.pt" not in downloaded_remotes

    poll_runner = _FakeRunner(
        exec_responses=[
            CommandResult(0, "ALIVE=False EXIT=0 PID=10 LINES=5 SNAPSHOTS=weights-u00025.pt,weights-u00100.pt\n")
        ]
    )
    assert run_colab_training(cfg, runner=poll_runner) == 0
    poll_downloads = [cmd[7] for cmd in poll_runner.calls if len(cmd) > 7 and cmd[4] == "download"]
    assert any(path.endswith("weights-u00100.pt") for path in poll_downloads)
    assert not any(path.endswith("weights-u00025.pt") for path in poll_downloads)


def test_puzzle_locked_jsonl_persistence_and_deduplication(tmp_path: Path) -> None:
    assert puzzles_path(tmp_path, "tla-limited-0.2.6") == tmp_path / "puzzles" / "tla-limited-0.2.6.jsonl"
    with pytest.raises(ValueError, match="invalid format_id"):
        puzzles_path(tmp_path, "../escape")
    assert load_puzzles(tmp_path, "tla-limited-0.2.6") == []

    rec1 = _sample_puzzle("puz-aaa", source="policy_disagreement")
    rec2 = _sample_puzzle("puz-bbb", source="high_swing_loss")
    out_path = append_puzzle(tmp_path, rec1)
    append_puzzle(tmp_path, rec1)  # duplicate puzzle_id must be ignored
    append_puzzle(tmp_path, rec2)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["puzzle_id"] == "puz-aaa"

    loaded = load_puzzles(tmp_path, "tla-limited-0.2.6")
    assert loaded == [rec1, rec2]


def test_puzzle_admission_gate_paired_crn_and_scoring() -> None:
    assert passes_value_prefilter(0.20)
    assert passes_value_prefilter(0.80)
    assert not passes_value_prefilter(0.19)
    assert not passes_value_prefilter(0.81)

    # Passing case: +25% paired margin with tight positive CI over 64 CRN pairs
    passing_pairs = [(1.0, 0.0)] * 16 + [(1.0, 1.0)] * 24 + [(0.0, 0.0)] * 24
    passed = summarize_paired_outcomes(
        passing_pairs, best_action=1, alt_action=0, playouts_per_action=64, max_plies=240
    )
    assert passed.admitted is True
    assert passed.margin == pytest.approx(0.25)
    assert passed.paired_ci_low > 0.0

    # Failing case 1: positive CI but margin (+12.5%) is below the +15% gate
    small_margin_pairs = [(1.0, 0.0)] * 8 + [(1.0, 1.0)] * 28 + [(0.0, 0.0)] * 28
    rejected_margin = summarize_paired_outcomes(
        small_margin_pairs, best_action=1, alt_action=0, playouts_per_action=64, max_plies=240
    )
    assert rejected_margin.admitted is False
    assert rejected_margin.margin == pytest.approx(0.125)
    assert rejected_margin.paired_ci_low > 0.0

    # Failing case 2: margin >= 0.15 (+20%) but high variance / small sample causes CI to cross 0
    noisy_pairs = [(1.0, 0.0)] * 3 + [(0.0, 1.0)] * 2
    rejected_ci = summarize_paired_outcomes(
        noisy_pairs, best_action=1, alt_action=0, playouts_per_action=5, max_plies=240
    )
    assert rejected_ci.admitted is False
    assert rejected_ci.margin == pytest.approx(0.20)
    assert rejected_ci.paired_ci_low <= 0.0

    # End-to-end CRN verify_puzzle_candidate with stub env
    stub_obs = _StubObs(_StubAgent(0), _StubActionSpace((0, 1)))
    strong_env = _StubPairedEnv(win_probs_by_action={0: 0.30, 1: 0.85})
    ver_pass = verify_puzzle_candidate(
        strong_env,
        stub_obs,
        best_action=1,
        alt_action=0,
        playouts=64,
        max_plies=PUZZLE_MIN_MAX_PLIES,
        seed=42,
    )
    assert ver_pass.admitted is True
    assert ver_pass.margin >= 0.15
    assert ver_pass.paired_ci_low > 0.0
    assert strong_env.clones_created == 128

    weak_env = _StubPairedEnv(win_probs_by_action={0: 0.45, 1: 0.50})
    ver_fail = verify_puzzle_candidate(
        weak_env,
        stub_obs,
        best_action=1,
        alt_action=0,
        playouts=64,
        max_plies=PUZZLE_MIN_MAX_PLIES,
        seed=42,
    )
    assert ver_fail.admitted is False

    with pytest.raises(ValueError, match="max_plies"):
        verify_puzzle_candidate(
            strong_env,
            stub_obs,
            best_action=1,
            alt_action=0,
            playouts=8,
            max_plies=160,
        )

    puzzles = [_sample_puzzle("puz-1", source="policy_disagreement"), _sample_puzzle("puz-2", source="high_swing_loss")]
    rows = score_puzzles(
        puzzles,
        eval_fn=lambda p: ((0.2, 0.8), 0.62) if p.puzzle_id == "puz-1" else ((0.7, 0.3), 0.41),
    )
    assert [r.solved for r in rows] == [True, False]
    summary = summarize_puzzle_scores(rows)
    assert summary["total"] == 2
    assert summary["solved"] == 1
    assert summary["solve_rate"] == pytest.approx(0.5)

"""Behavior tests for Jace v2's fast experiment harness and adaptive mirrored evaluator.

Covers paired variance reduction on mirrored `1-1` draw-luck splits, adaptive sequential
stopping and extension across the 200 -> 400 -> 1,000 mirrored game stages, promotion
and land-drop regression gating, multi-process CPU deal evaluation, and warm-start
checkpoint loading in `jace.train.ten_decks`. All tests run strictly on CPU in seconds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from manabot.backends.torch.checkpoint import load_checkpoint

from jace.eval import (
    AdaptiveEvalConfig,
    PairedDealResult,
    build_deals,
    elemental_decks,
    evaluate_adaptive,
    play_deals_parallel,
    summarize_paired_deals,
)
from jace.storage.gameplay import create_run, save_policy_checkpoint
from jace.train.harness import decide_promotion
from jace.train.ten_decks import LabConfig, build_agent, run_ten_deck_lab

if TYPE_CHECKING:
    from pathlib import Path


def _make_deals(wins_per_deal: list[int], *, offset: int = 0) -> list[PairedDealResult]:
    return [
        PairedDealResult(
            deal_index=offset + idx,
            wins=wins,
            losses=2 - wins,
            draws=0,
            decisions=200,
            turns=14,
        )
        for idx, wins in enumerate(wins_per_deal)
    ]


def test_paired_stats_cancels_draw_luck_splits() -> None:
    splits = _make_deals([1] * 100)
    score = summarize_paired_deals(splits)
    assert score.deals == 100
    assert score.games == 200
    assert score.win_rate == pytest.approx(0.5)
    assert score.deal_splits == 100
    assert score.paired_se == pytest.approx(0.0)
    assert score.ci_low == pytest.approx(0.5)
    assert score.ci_high == pytest.approx(0.5)

    # 60% overall via 20 sweeps won + 80 pure splits yields a much tighter paired SE than unpaired 3.5%.
    edge = _make_deals([2] * 20 + [1] * 80)
    edge_score = summarize_paired_deals(edge)
    assert edge_score.win_rate == pytest.approx(0.6)
    assert edge_score.paired_se < 0.021
    assert edge_score.ci_low > 0.55


def test_adaptive_eval_stops_early_on_clear_win_and_loss() -> None:
    calls: list[tuple[int, int]] = []

    def win_chunk(start: int, end: int) -> list[PairedDealResult]:
        calls.append((start, end))
        # 65% win rate via 30 sweeps won + 70 splits -> clear win at stage 1.
        return _make_deals([2] * 30 + [1] * (end - start - 30), offset=start)

    win_res = evaluate_adaptive(win_chunk, config=AdaptiveEvalConfig(stages=(100, 200, 500)))
    assert win_res.verdict == "WIN"
    assert win_res.stage_reached == 100
    assert calls == [(0, 100)]

    loss_calls: list[tuple[int, int]] = []

    def loss_chunk(start: int, end: int) -> list[PairedDealResult]:
        loss_calls.append((start, end))
        return _make_deals([0] * 30 + [1] * (end - start - 30), offset=start)

    loss_res = evaluate_adaptive(loss_chunk, config=AdaptiveEvalConfig(stages=(100, 200, 500)))
    assert loss_res.verdict == "LOSS"
    assert loss_res.stage_reached == 100
    assert loss_calls == [(0, 100)]


def test_adaptive_eval_extends_to_1000_games_in_close_magic_zone() -> None:
    calls: list[tuple[int, int]] = []

    def close_edge_chunk(start: int, end: int) -> list[PairedDealResult]:
        calls.append((start, end))
        # Realistic Magic variance: 34% 2-0, 27% 0-2, 39% 1-1 splits (+3.5 pp edge).
        # CI still crosses 50% at 200 and 400 games, then confirms WIN at 1,000 games.
        pattern = [2 if idx % 100 < 34 else (0 if idx % 100 < 61 else 1) for idx in range(start, end)]
        return _make_deals(pattern, offset=start)

    res = evaluate_adaptive(close_edge_chunk, config=AdaptiveEvalConfig(stages=(100, 200, 500)))
    assert res.verdict == "WIN"
    assert res.stage_reached == 500
    assert calls == [(0, 100), (100, 200), (200, 500)]
    decision, _ = decide_promotion(res, missed_land_delta=0.01)
    assert decision == "PROMOTE"
    killed, reason = decide_promotion(res, missed_land_delta=0.05)
    assert killed == "KILL"
    assert "missed land" in reason


def test_play_deals_parallel_two_workers_smoke() -> None:
    deals = build_deals(elemental_decks(), count=2, seed=77, decklists="closed")
    results = play_deals_parallel("heuristic", "random", deals=deals, num_workers=2, seed=77, deterministic=True)
    assert len(results) == 2
    assert [r.deal_index for r in results] == [0, 1]
    score = summarize_paired_deals(results)
    assert score.games == 4


def test_init_checkpoint_warm_starts_train_ten_decks(tmp_path: Path) -> None:
    torch.manual_seed(11)
    src_agent, obs_space = build_agent(hidden_dim=32, attention_layers=1)
    first_name, first_param = next(iter(src_agent.named_parameters()))
    # Shift one parameter tensor by a distinctive constant so we can verify warm-start loaded it.
    with torch.no_grad():
        first_param.fill_(3.25)
    seed_layout = create_run(runs_dir=tmp_path, experiment="seed", run_id="parent", config={"seed": 11})
    ckpt_path, _ = save_policy_checkpoint(src_agent, obs_space=obs_space, layout=seed_layout, iteration=0)

    layout = run_ten_deck_lab(
        LabConfig(
            runs_dir=tmp_path,
            experiment="warm",
            run_id="child",
            updates=0,
            schedule_kind="none",
            num_envs=2,
            steps_per_env=4,
            hidden_dim=32,
            attention_layers=1,
            eval_every=1,
            eval_deals=1,
            probe_count=4,
            record_train_fraction=0.0,
            collect_device="cpu",
            learn_device="cpu",
            init_checkpoint=ckpt_path,
            seed=11,
        )
    )
    child_ckpt = layout.checkpoint_path(0)
    assert child_ckpt.exists()
    loaded = load_checkpoint(str(child_ckpt))[0]
    loaded_param = loaded.get_parameter(first_name)
    assert float(loaded_param.mean().item()) == pytest.approx(3.25, abs=1e-4)

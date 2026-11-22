"""Parallel mirrored-deal evaluation and adaptive sequential judge for Jace v2.

Evaluates two Magic policies across mirrored paired deals (identical shuffles, swapped
seats) in parallel across spawned CPU worker processes (`torch.set_num_threads(1)` per
worker) so 200 mirrored games run in ~35 seconds and 400 in ~70 seconds. Because paired
deals cancel out pure mana-screw and mana-flood splits (`1-1` deals contribute zero
paired variance), `summarize_paired_deals` computes paired standard errors and 95%
confidence intervals that detect +3 to +5 percentage point Limited Magic gains.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from manabot.backends.torch.checkpoint import load_checkpoint

from jace.eval.deals import (
    Deal,
    PlayerFn,
    PolicyPlayer,
    build_deals,
    elemental_decks,
    heuristic_player,
    play_match,
    random_player,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = (
    "AdaptiveEvalConfig",
    "AdaptiveEvalResult",
    "PairedDealResult",
    "PairedMatchScore",
    "PlayerSpec",
    "evaluate_adaptive",
    "play_adaptive_parallel",
    "play_deals_parallel",
    "summarize_paired_deals",
)

type PlayerSpec = Path | Literal["heuristic", "random"]

_Z_95 = 1.959963984540054
_GAMES_PER_DEAL = 2
_EVEN_WIN_RATE = 0.5


@dataclass(frozen=True, slots=True)
class PairedDealResult:
    deal_index: int
    wins: int
    losses: int
    draws: int
    decisions: int
    turns: int

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return (self.wins + 0.5 * self.draws) / total if total else 0.5


@dataclass(frozen=True, slots=True)
class PairedMatchScore:
    deals: int
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    paired_diff: float
    paired_se: float
    ci_low: float
    ci_high: float
    deal_sweeps_won: int
    deal_splits: int
    deal_sweeps_lost: int
    mean_decisions: float
    mean_turns: float


@dataclass(frozen=True, slots=True)
class AdaptiveEvalConfig:
    stages: tuple[int, ...] = (100, 200, 500)
    clear_win_rate: float = 0.58
    clear_loss_rate: float = 0.44
    close_low: float = 0.48
    close_high: float = 0.56


@dataclass(frozen=True, slots=True)
class AdaptiveEvalResult:
    score: PairedMatchScore
    verdict: Literal["WIN", "LOSS", "TIE"]
    stage_reached: int
    reason: str


def summarize_paired_deals(results: Sequence[PairedDealResult]) -> PairedMatchScore:
    """Compute paired win-rate statistics across mirrored two-leg deals.

    Each `PairedDealResult` summarizes both seats of one shuffle (`2-0`, `1-1`, or `0-2`).
    Computes the sample standard error over per-deal win rates so `1-1` draw-luck splits
    reduce variance instead of inflating the confidence interval. Returns zeroed fields
    when `results` is empty.
    """
    n_deals = len(results)
    if n_deals == 0:
        return PairedMatchScore(
            deals=0,
            games=0,
            wins=0,
            losses=0,
            draws=0,
            win_rate=0.0,
            paired_diff=0.0,
            paired_se=0.0,
            ci_low=0.0,
            ci_high=1.0,
            deal_sweeps_won=0,
            deal_splits=0,
            deal_sweeps_lost=0,
            mean_decisions=0.0,
            mean_turns=0.0,
        )
    wins = sum(r.wins for r in results)
    losses = sum(r.losses for r in results)
    draws = sum(r.draws for r in results)
    games = wins + losses + draws
    rates = [r.win_rate for r in results]
    mean_wr = sum(rates) / n_deals
    if n_deals > 1:
        var = sum((rate - mean_wr) ** 2 for rate in rates) / (n_deals - 1)
        paired_se = math.sqrt(var / n_deals)
    else:
        paired_se = 0.5
    half = _Z_95 * paired_se
    sweeps_won = sum(1 for r in results if r.wins == _GAMES_PER_DEAL and r.losses == 0)
    sweeps_lost = sum(1 for r in results if r.losses == _GAMES_PER_DEAL and r.wins == 0)
    splits = n_deals - sweeps_won - sweeps_lost
    return PairedMatchScore(
        deals=n_deals,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=mean_wr,
        paired_diff=2.0 * mean_wr - 1.0,
        paired_se=paired_se,
        ci_low=max(0.0, mean_wr - half),
        ci_high=min(1.0, mean_wr + half),
        deal_sweeps_won=sweeps_won,
        deal_splits=splits,
        deal_sweeps_lost=sweeps_lost,
        mean_decisions=sum(r.decisions for r in results) / games if games else 0.0,
        mean_turns=sum(r.turns for r in results) / games if games else 0.0,
    )


def _open_spec_player(
    spec: PlayerSpec,
    *,
    seed: int,
    deterministic: bool,
) -> tuple[PlayerFn, PolicyPlayer | None]:
    if spec == "heuristic":
        return heuristic_player(), None
    if spec == "random":
        return random_player(seed=seed), None
    policy = load_checkpoint(str(spec))[0].eval()
    player = PolicyPlayer(policy, policy.observation_space, seed=seed, deterministic=deterministic)
    return player, player


def _eval_worker(
    args: tuple[PlayerSpec, PlayerSpec, list[tuple[int, Deal]], int, bool, int],
) -> list[PairedDealResult]:
    spec_a, spec_b, indexed_deals, seed, deterministic, max_actions = args
    logging.getLogger("manabot").setLevel(logging.WARNING)
    torch.set_num_threads(1)
    fn_a, ctx_a = _open_spec_player(spec_a, seed=seed, deterministic=deterministic)
    fn_b, ctx_b = _open_spec_player(spec_b, seed=seed + 10_000, deterministic=deterministic)
    results: list[PairedDealResult] = []
    if ctx_a is not None:
        ctx_a.__enter__()
    if ctx_b is not None:
        ctx_b.__enter__()
    try:
        for deal_idx, deal in indexed_deals:
            score = play_match(fn_a, fn_b, deals=(deal,), max_actions=max_actions)
            results.append(
                PairedDealResult(
                    deal_index=deal_idx,
                    wins=score.wins,
                    losses=score.losses,
                    draws=score.draws,
                    decisions=round(score.mean_decisions * score.games),
                    turns=round(score.mean_turns * score.games),
                )
            )
    finally:
        if ctx_b is not None:
            ctx_b.__exit__(None, None, None)
        if ctx_a is not None:
            ctx_a.__exit__(None, None, None)
    return results


def play_deals_parallel(
    player_a: PlayerSpec,
    player_b: PlayerSpec,
    *,
    deals: Sequence[Deal],
    num_workers: int = 8,
    seed: int = 0,
    deterministic: bool = True,
    max_actions: int = 1200,
) -> list[PairedDealResult]:
    """Evaluate `player_a` vs. `player_b` across mirrored `deals` in parallel CPU processes.

    Distributes deals round-robin across `min(num_workers, len(deals))` spawned CPU workers
    so each worker loads checkpoints once and runs `torch.set_num_threads(1)`. Returns
    results sorted by `deal_index`.
    """
    if not deals:
        return []
    workers = max(1, min(num_workers, len(deals)))
    shards: list[list[tuple[int, Deal]]] = [[] for _ in range(workers)]
    for idx, deal in enumerate(deals):
        shards[idx % workers].append((idx, deal))
    tasks = [
        (player_a, player_b, shard, seed + rank * 1000, deterministic, max_actions)
        for rank, shard in enumerate(shards)
        if shard
    ]
    if len(tasks) == 1:
        combined = _eval_worker(tasks[0])
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(tasks)) as pool:
            parts = pool.map(_eval_worker, tasks)
        combined = [item for part in parts for item in part]
    combined.sort(key=lambda item: item.deal_index)
    return combined


_DEFAULT_ADAPTIVE_CONFIG = AdaptiveEvalConfig()


def _first_stage_decision(
    score: PairedMatchScore,
    config: AdaptiveEvalConfig,
) -> tuple[Literal["WIN", "LOSS", "TIE"], str] | None:
    if score.ci_low > _EVEN_WIN_RATE and score.win_rate >= config.clear_win_rate:
        return "WIN", f"clear win at {score.games} games ({score.win_rate:.1%})"
    if score.ci_high < _EVEN_WIN_RATE and score.win_rate <= config.clear_loss_rate:
        return "LOSS", f"clear loss at {score.games} games ({score.win_rate:.1%})"
    return None


def _mid_stage_decision(
    score: PairedMatchScore,
    config: AdaptiveEvalConfig,
) -> tuple[Literal["WIN", "LOSS", "TIE"], str] | None:
    if score.ci_low > _EVEN_WIN_RATE:
        return "WIN", f"paired 95% CI above 50% at {score.games} games ({score.win_rate:.1%})"
    if score.ci_high < _EVEN_WIN_RATE:
        return "LOSS", f"paired 95% CI below 50% at {score.games} games ({score.win_rate:.1%})"
    if not (config.close_low <= score.win_rate <= config.close_high):
        return "TIE", f"outside close zone without significance at {score.games} games"
    return None


def _final_stage_decision(score: PairedMatchScore) -> tuple[Literal["WIN", "LOSS", "TIE"], str]:
    if score.ci_low > _EVEN_WIN_RATE:
        return "WIN", f"confirmed win at {score.games} games ({score.win_rate:.1%})"
    if score.ci_high < _EVEN_WIN_RATE:
        return "LOSS", f"confirmed loss at {score.games} games ({score.win_rate:.1%})"
    return "TIE", f"inconclusive after {score.games} games ({score.win_rate:.1%})"


def _stage_decision(
    score: PairedMatchScore,
    *,
    stage_idx: int,
    total_stages: int,
    config: AdaptiveEvalConfig,
) -> tuple[Literal["WIN", "LOSS", "TIE"], str] | None:
    if stage_idx == 0 and total_stages > 1:
        return _first_stage_decision(score, config)
    if stage_idx < total_stages - 1:
        return _mid_stage_decision(score, config)
    return _final_stage_decision(score)


def evaluate_adaptive(
    run_chunk: Callable[[int, int], Sequence[PairedDealResult]],
    *,
    config: AdaptiveEvalConfig = _DEFAULT_ADAPTIVE_CONFIG,
) -> AdaptiveEvalResult:
    """Run staged adaptive mirrored evaluation, extending only on close Magic outcomes.

    Calls `run_chunk(start_deal, end_deal)` for each deal stage in `config.stages`
    (default 100 -> 200 -> 500 deals = 200 -> 400 -> 1,000 mirrored games). Stops early on
    clear blowouts at Stage 1, stops at Stage 2 when the paired 95% CI excludes 50%, and
    extends to the final stage only when the win rate sits inside the close +2 to +6 pp
    zone. Raises `ValueError` if `config.stages` is empty or non-increasing.
    """
    if not config.stages or any(a >= b for a, b in zip(config.stages, config.stages[1:], strict=False)):
        msg = f"stages must be a non-empty strictly increasing tuple, got {config.stages}"
        raise ValueError(msg)
    collected: list[PairedDealResult] = []
    prev_end = 0
    for stage_idx, target_deals in enumerate(config.stages):
        collected.extend(run_chunk(prev_end, target_deals))
        prev_end = target_deals
        score = summarize_paired_deals(collected)
        decision = _stage_decision(score, stage_idx=stage_idx, total_stages=len(config.stages), config=config)
        if decision is not None:
            verdict, reason = decision
            return AdaptiveEvalResult(score=score, verdict=verdict, stage_reached=target_deals, reason=reason)
    final_score = summarize_paired_deals(collected)
    return AdaptiveEvalResult(score=final_score, verdict="TIE", stage_reached=prev_end, reason="exhausted stages")


def play_adaptive_parallel(
    player_a: PlayerSpec,
    player_b: PlayerSpec,
    *,
    seed: int = 20_000,
    num_workers: int = 8,
    deterministic: bool = True,
    config: AdaptiveEvalConfig = _DEFAULT_ADAPTIVE_CONFIG,
) -> AdaptiveEvalResult:
    """Run staged adaptive parallel mirrored evaluation over the 10 closed TLA decks.

    Pre-generates `max(config.stages)` deterministic deals from `elemental_decks()` so
    every stage shares an identical, reproducible deal stream across experiments. Each
    stage dispatches only the newly added deal slice across `num_workers` spawned CPU
    processes, shifts each deal's index back to its global position in the shared deal
    stream, and stops as soon as `evaluate_adaptive` reaches a `WIN`, `LOSS`, or `TIE`.
    Returns an `AdaptiveEvalResult` with the combined paired score and stopping reason.
    """
    all_deals = build_deals(elemental_decks(), count=max(config.stages), seed=seed, decklists="closed")

    def _run_chunk(start: int, end: int) -> list[PairedDealResult]:
        if end <= start:
            return []
        chunk = play_deals_parallel(
            player_a,
            player_b,
            deals=all_deals[start:end],
            num_workers=num_workers,
            seed=seed + start * 10,
            deterministic=deterministic,
        )
        return [
            PairedDealResult(
                deal_index=start + item.deal_index,
                wins=item.wins,
                losses=item.losses,
                draws=item.draws,
                decisions=item.decisions,
                turns=item.turns,
            )
            for item in chunk
        ]

    return evaluate_adaptive(_run_chunk, config=config)

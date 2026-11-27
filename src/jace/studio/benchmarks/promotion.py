"""Paired CRN playout verification and Watchlist-to-puzzle promotion helpers."""

from __future__ import annotations

import math
import random
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import managym
from managym.enums import ActionSpaceEnum, ZoneEnum

from jace.eval.puzzles import PuzzleRecord, append_puzzle, summarize_paired_outcomes
from jace.studio.replay import engine_step_errors, seat_aligned_obs, serialize_legal_actions

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from jace.studio.inference import InferenceEngine
    from jace.studio.replay import ReplayService

__all__ = (
    "PuzzlePromotionResult",
    "as_float",
    "as_int",
    "promote_watchlist_position",
    "summarize_player_counts",
)

_MAX_PROMOTE_ACTIONS = 4
_MIN_LEGAL_ACTIONS = 2
_PROMOTE_MAX_PLIES = 240
_SAFE_FORMAT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})


@dataclass(frozen=True, slots=True)
class PuzzlePromotionResult:
    admitted: bool
    margin: float
    ci_low: float
    ci_high: float
    detail: str | None = None
    puzzle: dict[str, object] | None = None


def as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return default


def as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    return default


def summarize_player_counts(obs: managym.Observation, *, hero_seat: int) -> tuple[str, str, list[str]]:
    hero_pack, opp_pack = seat_aligned_obs(obs, hero_seat=hero_seat)
    hero_p, hero_perms, _ = hero_pack
    opp_p, opp_perms, _ = opp_pack
    hero_lands = sum(1 for p in hero_perms if p.characteristics.card_types.is_land)
    opp_lands = sum(1 for p in opp_perms if p.characteristics.card_types.is_land)
    hero_hand = int(hero_p.zone_counts[int(ZoneEnum.HAND)])
    opp_hand = int(opp_p.zone_counts[int(ZoneEnum.HAND)])
    hero_summary = f"Hero: {int(hero_p.life)} life · {hero_lands} lands · {hero_hand} cards"
    opp_summary = f"Opp: {int(opp_p.life)} life · {opp_lands} lands · {opp_hand} cards"
    key_cards: list[str] = []
    for perm in (*hero_perms, *opp_perms):
        name = str(perm.characteristics.name)
        if name not in _BASIC_LANDS and name not in key_cards:
            key_cards.append(name)
        if len(key_cards) >= _MIN_LEGAL_ACTIONS:
            break
    return hero_summary, opp_summary, key_cards


def _validate_format_id(format_id: str) -> str:
    cleaned = format_id.strip()
    if not _SAFE_FORMAT_RE.match(cleaned):
        msg = f"invalid format name: {format_id!r}"
        raise ValueError(msg)
    return cleaned


def _determinize_or_seed(
    cloned: managym.Env,
    *,
    deal_seed: int,
    acting_seat: int,
    step_errors: tuple[type[BaseException], ...],
) -> None:
    try:
        cloned.determinize(deal_seed, acting_seat)
    except step_errors:
        cloned.set_seed(deal_seed)


def _rollout_heuristic(
    cloned: managym.Env,
    obs: managym.Observation,
    *,
    heuristic: managym.Heuristic,
    rng: random.Random,
    max_plies: int,
    step_errors: tuple[type[BaseException], ...],
) -> None:
    for _ in range(max_plies):
        if cloned.is_game_over():
            break
        try:
            chosen = int(heuristic.select_action(obs))
        except step_errors:
            chosen = rng.randrange(max(1, len(obs.action_space.actions)))
        try:
            obs, _, done, truncated, _ = cloned.step(chosen)
        except step_errors:
            break
        if done or truncated:
            break


def _play_single_crn_branch(
    env: managym.Env,
    *,
    action_idx: int,
    deal_seed: int,
    acting_seat: int,
    hero_seat: int,
    heuristic: managym.Heuristic,
    max_plies: int,
) -> float:
    step_errors = engine_step_errors()
    cloned = env.clone_env()
    _determinize_or_seed(cloned, deal_seed=deal_seed, acting_seat=acting_seat, step_errors=step_errors)
    try:
        obs, _, done, truncated, _ = cloned.step(action_idx)
    except step_errors:
        return 0.5
    if not done and not truncated:
        _rollout_heuristic(
            cloned,
            obs,
            heuristic=heuristic,
            rng=random.Random(deal_seed ^ 0x5A5A),
            max_plies=max_plies,
            step_errors=step_errors,
        )
    winner = cloned.winner_index() if cloned.is_game_over() else None
    if winner is None:
        return 0.5
    return 1.0 if int(winner) == hero_seat else 0.0


def _run_paired_crn_outcomes(
    env: managym.Env,
    *,
    action_indices: list[int],
    playouts: int,
    acting_seat: int,
    hero_seat: int,
    seed: int,
    max_plies: int = _PROMOTE_MAX_PLIES,
) -> dict[int, list[float]]:
    heuristic = managym.Heuristic()
    outcomes: dict[int, list[float]] = {idx: [] for idx in action_indices}
    for deal_idx in range(playouts):
        deal_seed = seed * 1_000_003 + deal_idx * 9973
        for action_idx in action_indices:
            outcomes[action_idx].append(
                _play_single_crn_branch(
                    env,
                    action_idx=action_idx,
                    deal_seed=deal_seed,
                    acting_seat=acting_seat,
                    hero_seat=hero_seat,
                    heuristic=heuristic,
                    max_plies=max_plies,
                )
            )
    return outcomes


def _pick_top_action_indices(probs: list[float], num_actions: int) -> list[int]:
    if not probs or len(probs) < num_actions:
        return list(range(min(num_actions, _MAX_PROMOTE_ACTIONS)))
    return sorted(range(num_actions), key=lambda idx: probs[idx], reverse=True)[:_MAX_PROMOTE_ACTIONS]


def _build_promoted_record(
    *,
    item: dict[str, object],
    obs: managym.Observation,
    format_id: str,
    hero_seat: int,
    actual_step: int,
    seed: int,
    root_v: float,
    legal_actions: list[dict[str, object]],
    best_idx: int,
    alt_idx: int,
    best_wr: float,
    alt_wr: float,
    playouts: int,
    margin: float,
    ci_low: float,
    ci_high: float,
) -> PuzzleRecord:
    item_id = str(item.get("id") or uuid.uuid4().hex[:6])
    hero_summary, opp_summary, obs_cards = summarize_player_counts(obs, hero_seat=hero_seat)
    raw_cards = item.get("key_cards")
    item_cards = [str(c) for c in raw_cards if isinstance(c, str)][:2] if isinstance(raw_cards, list) else []
    decision_kind = ActionSpaceEnum(int(obs.action_space.action_space_type)).name
    return PuzzleRecord(
        puzzle_id=f"wl-{item_id}",
        format_id=format_id,
        source="watchlist_promotion",
        run_id=str(item.get("run_id") or ""),
        iteration=as_int(item.get("iteration")),
        game_idx=as_int(item.get("game_idx")),
        step=actual_step,
        turn=int(obs.turn.turn_number),
        phase=str(obs.turn.phase).split(".")[-1],
        step_name=str(obs.turn.step).split(".")[-1],
        decision_kind=decision_kind,
        hero_seat=hero_seat,
        seed=seed,
        hero_summary=hero_summary,
        opp_summary=opp_summary,
        key_cards=tuple(item_cards or obs_cards),
        root_value_v=round(root_v, 4),
        best_action_index=best_idx,
        best_action_label=str(legal_actions[best_idx]["label"]),
        alt_action_index=alt_idx,
        alt_action_label=str(legal_actions[alt_idx]["label"]),
        best_playout_wr=round(best_wr, 4),
        alt_playout_wr=round(alt_wr, 4),
        playouts_per_action=playouts,
        max_plies=_PROMOTE_MAX_PLIES,
        margin=round(margin, 4),
        paired_ci_low=round(ci_low, 4),
        paired_ci_high=round(ci_high, 4),
    )


def promote_watchlist_position(
    runs_dir: Path,
    item: dict[str, object],
    *,
    replays: ReplayService,
    inference: InferenceEngine,
    card_builder: Callable[[dict[str, object], str], dict[str, object]],
    format_id: str,
    playouts: int = 64,
    min_margin: float = 0.15,
    model_ref: str | None = None,
) -> PuzzlePromotionResult:
    safe_format = _validate_format_id(format_id)
    run_id = str(item.get("run_id") or "")
    iteration = as_int(item.get("iteration"))
    game_idx = as_int(item.get("game_idx"))
    game, env, obs, actual_step, _, _, _, _ = replays.replay_to_step(
        run_id=run_id, iteration=iteration, game_idx=game_idx, step=as_int(item.get("step"))
    )
    legal_actions = serialize_legal_actions(obs, hero_seat=game.hero_seat)
    if len(legal_actions) < _MIN_LEGAL_ACTIONS:
        return PuzzlePromotionResult(
            admitted=False,
            margin=0.0,
            ci_low=0.0,
            ci_high=0.0,
            detail="Position has fewer than 2 legal actions and cannot form a tactical puzzle.",
        )
    acting_seat = int(obs.agent.player_index)
    eval_ref = model_ref or inference.find_checkpoint_for_iteration(run_id, iteration) or "heuristic"
    probs_list, wdls_list, _, _ = inference.evaluate_batch_full(
        [obs], run_id=run_id, checkpoint_name=eval_ref, hero_seats=[acting_seat]
    )
    root_v = as_float(wdls_list[0].get("win"), 0.5) if wdls_list else 0.5
    action_indices = _pick_top_action_indices(probs_list[0] if probs_list else [], len(legal_actions))
    outcomes = _run_paired_crn_outcomes(
        env,
        action_indices=action_indices,
        playouts=playouts,
        acting_seat=acting_seat,
        hero_seat=acting_seat,
        seed=game.seed or 42,
        max_plies=_PROMOTE_MAX_PLIES,
    )
    ranked = sorted(action_indices, key=lambda idx: sum(outcomes[idx]) / max(1, len(outcomes[idx])), reverse=True)
    best_idx, alt_idx = ranked[0], ranked[1]
    pairs = list(zip(outcomes[best_idx], outcomes[alt_idx], strict=True))
    ver = summarize_paired_outcomes(
        pairs,
        best_action=best_idx,
        alt_action=alt_idx,
        playouts_per_action=playouts,
        max_plies=_PROMOTE_MAX_PLIES,
        min_margin=min_margin,
    )
    if not ver.admitted:
        detail = (
            f"Margin {ver.margin * 100:+.1f}% "
            f"(95% CI [{ver.paired_ci_low * 100:+.1f}%, {ver.paired_ci_high * 100:+.1f}%]) "
            f"did not pass +{min_margin * 100:.1f}% & CI > 0 gate"
        )
        return PuzzlePromotionResult(
            admitted=False,
            margin=round(ver.margin, 4),
            ci_low=round(ver.paired_ci_low, 4),
            ci_high=round(ver.paired_ci_high, 4),
            detail=detail,
        )
    record = _build_promoted_record(
        item=item,
        obs=obs,
        format_id=safe_format,
        hero_seat=acting_seat,
        actual_step=actual_step,
        seed=game.seed,
        root_v=root_v,
        legal_actions=legal_actions,
        best_idx=best_idx,
        alt_idx=alt_idx,
        best_wr=ver.best_playout_wr,
        alt_wr=ver.alt_playout_wr,
        playouts=playouts,
        margin=ver.margin,
        ci_low=ver.paired_ci_low,
        ci_high=ver.paired_ci_high,
    )
    append_puzzle(runs_dir, record)
    card = card_builder(record.to_dict(), safe_format)
    return PuzzlePromotionResult(
        admitted=True,
        margin=round(ver.margin, 4),
        ci_low=round(ver.paired_ci_low, 4),
        ci_high=round(ver.paired_ci_high, 4),
        puzzle=card,
    )

"""Batched counterfactual action playouts and cross-checkpoint comparison for Jace Studio.

Runs paired-seed determinized playouts across legal actions in a single batched
forward pass per ply (or via native `managym.Heuristic`) with Wilson 95%
confidence intervals, and compares any two checkpoints across stratified
decision states sampled from playable indexed games.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import managym
from managym.enums import ActionSpaceEnum

from jace.studio.index import GameQuery
from jace.studio.replay import (
    engine_step_errors,
    load_recorded_game,
    sample_game_observations,
    seat_aligned_life,
    serialize_legal_actions,
)

if TYPE_CHECKING:
    from pathlib import Path

    from jace.studio.index import GameRecordSummary, StudioIndex
    from jace.studio.inference import InferenceEngine

__all__ = (
    "compare_checkpoints",
    "run_action_playouts",
)

_WILSON_Z_95 = 1.959963984540054
_DEFAULT_MAX_PLIES = 160
_MIN_DECISION_ACTIONS = 2
_COMPARE_STRATA_TARGET = 20


@dataclass(slots=True)
class _PlayoutBranch:
    action_idx: int
    deal_idx: int
    env: managym.Env
    obs: managym.Observation | None
    rng: random.Random
    done: bool = False
    errored: bool = False


@dataclass(frozen=True, slots=True)
class _SampledPosition:
    game: GameRecordSummary
    hero_seat: int
    step: int
    decision_kind: str
    obs: managym.Observation


def _wilson_ci(successes: float, total: int, *, z: float = _WILSON_Z_95) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = max(0.0, min(1.0, successes / total))
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def _select_action_indices(num_actions: int, requested: list[int] | None) -> list[int]:
    if requested is None:
        return list(range(num_actions))
    seen: set[int] = set()
    selected: list[int] = []
    for idx in requested:
        if 0 <= idx < num_actions and idx not in seen:
            seen.add(idx)
            selected.append(idx)
    return selected


def _determinize_branch(
    branch_env: managym.Env,
    *,
    deal_seed: int,
    acting_seat: int,
    step_errors: tuple[type[BaseException], ...],
) -> None:
    try:
        branch_env.determinize(deal_seed, acting_seat)
    except step_errors:
        try:
            branch_env.set_seed(deal_seed)
        except step_errors:
            return


def _spawn_playout_branches(
    env: managym.Env,
    *,
    action_indices: list[int],
    deals: int,
    batch_offset: int,
    acting_seat: int,
    seed: int,
) -> list[_PlayoutBranch]:
    step_errors = engine_step_errors()
    branches: list[_PlayoutBranch] = []
    for deal_idx in range(batch_offset, batch_offset + deals):
        deal_seed = seed * 1_000_003 + deal_idx * 9973
        for action_idx in action_indices:
            cloned = env.clone_env()
            _determinize_branch(cloned, deal_seed=deal_seed, acting_seat=acting_seat, step_errors=step_errors)
            errored = False
            try:
                next_obs, _, done, truncated, _ = cloned.step(action_idx)
                is_done = bool(done or truncated or cloned.is_game_over())
            except step_errors:
                next_obs, is_done, errored = None, True, True
            branches.append(
                _PlayoutBranch(
                    action_idx=action_idx,
                    deal_idx=deal_idx,
                    env=cloned,
                    obs=next_obs,
                    rng=random.Random(deal_seed ^ (action_idx * 31337)),
                    done=is_done,
                    errored=errored,
                )
            )
    return branches


def _step_playout_branch(
    branch: _PlayoutBranch,
    action: int,
    *,
    step_errors: tuple[type[BaseException], ...],
) -> None:
    try:
        next_obs, _, done, truncated, _ = branch.env.step(action)
    except step_errors:
        branch.obs, branch.done, branch.errored = None, True, True
        return
    branch.obs = next_obs
    branch.done = bool(done or truncated or branch.env.is_game_over())


def _step_branches_heuristic(
    branches: list[_PlayoutBranch],
    *,
    max_plies: int,
    step_errors: tuple[type[BaseException], ...],
) -> None:
    heuristic = managym.Heuristic()
    for _ in range(max_plies):
        active = [b for b in branches if not b.done and b.obs is not None]
        if not active:
            break
        for branch in active:
            obs = branch.obs
            if obs is None:
                continue
            try:
                chosen = int(heuristic.select_action(obs))
            except step_errors:
                chosen = 0
            _step_playout_branch(branch, chosen, step_errors=step_errors)


def _step_branches_model(
    branches: list[_PlayoutBranch],
    *,
    engine: InferenceEngine,
    run_id: str,
    model_ref: str,
    max_plies: int,
    step_errors: tuple[type[BaseException], ...],
) -> tuple[str, str | None]:
    device = "cpu"
    warning: str | None = None
    for _ in range(max_plies):
        active = [b for b in branches if not b.done and b.obs is not None]
        if not active:
            break
        obs_list = [b.obs for b in active if b.obs is not None]
        probs_batch, device, warn = engine.evaluate_batch_action_probs(
            obs_list, run_id=run_id, checkpoint_name=model_ref
        )
        if warn and warning is None:
            warning = warn
        for branch, probs in zip(active, probs_batch, strict=True):
            if not probs:
                branch.done = True
                continue
            action = branch.rng.choices(range(len(probs)), weights=probs, k=1)[0]
            _step_playout_branch(branch, action, step_errors=step_errors)
    return device, warning


def _summarize_action_branches(
    action_idx: int,
    label: str,
    *,
    branches: list[_PlayoutBranch],
    hero_seat: int,
) -> dict[str, object]:
    wins = 0
    draws = 0
    losses = 0
    for branch in branches:
        if branch.errored:
            continue
        winner = branch.env.winner_index() if branch.env.is_game_over() else None
        if winner is None:
            draws += 1
        elif int(winner) == hero_seat:
            wins += 1
        else:
            losses += 1
    total = wins + draws + losses
    score = wins + 0.5 * draws
    win_rate = round(score / total, 4) if total > 0 else 0.5
    ci_low, ci_high = _wilson_ci(score, total)
    return {
        "action_index": action_idx,
        "label": label,
        "deals_done": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": win_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _execute_playout_branches(
    engine: InferenceEngine,
    branches: list[_PlayoutBranch],
    *,
    run_id: str,
    model_ref: str,
    max_plies: int,
    step_errors: tuple[type[BaseException], ...],
) -> tuple[str, str | None]:
    use_model = bool(model_ref) and model_ref != "heuristic" and engine.can_load_checkpoint(run_id, model_ref)
    if use_model:
        return _step_branches_model(
            branches, engine=engine, run_id=run_id, model_ref=model_ref, max_plies=max_plies, step_errors=step_errors
        )
    _step_branches_heuristic(branches, max_plies=max_plies, step_errors=step_errors)
    warning = (
        f"Checkpoint '{model_ref}' not found; fell back to Heuristic."
        if model_ref and model_ref != "heuristic"
        else None
    )
    return "cpu", warning


def run_action_playouts(
    engine: InferenceEngine,
    env: managym.Env,
    *,
    obs: managym.Observation,
    run_id: str,
    hero_seat: int,
    model_ref: str,
    deals: int,
    batch_offset: int = 0,
    action_indices: list[int] | None = None,
    seed: int = 42,
    max_plies: int = _DEFAULT_MAX_PLIES,
    hide_actions: bool = False,
) -> dict[str, object]:
    start = time.perf_counter()
    legal_actions = [] if hide_actions else serialize_legal_actions(obs, hero_seat=hero_seat)
    chosen_indices = _select_action_indices(len(legal_actions), action_indices)
    if not chosen_indices or deals <= 0:
        return {
            "model_ref": model_ref,
            "device": "cpu",
            "deals_done": 0,
            "batch_offset": batch_offset,
            "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 1),
            "results": [],
        }
    branches = _spawn_playout_branches(
        env,
        action_indices=chosen_indices,
        deals=deals,
        batch_offset=batch_offset,
        acting_seat=int(obs.agent.player_index),
        seed=seed,
    )
    device, warning = _execute_playout_branches(
        engine, branches, run_id=run_id, model_ref=model_ref, max_plies=max_plies, step_errors=engine_step_errors()
    )
    grouped: dict[int, list[_PlayoutBranch]] = {idx: [] for idx in chosen_indices}
    for branch in branches:
        grouped[branch.action_idx].append(branch)
    results = [
        _summarize_action_branches(
            idx,
            str(legal_actions[idx]["label"]),
            branches=grouped[idx],
            hero_seat=hero_seat,
        )
        for idx in chosen_indices
    ]
    payload: dict[str, object] = {
        "model_ref": model_ref,
        "device": device,
        "deals_done": deals,
        "batch_offset": batch_offset,
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 1),
        "results": results,
    }
    if warning:
        payload["warning"] = warning
    return payload


def _extract_game_positions(
    runs_dir: Path,
    game_summary: GameRecordSummary,
    *,
    kind_filter: str | None,
) -> list[_SampledPosition]:
    try:
        recorded = load_recorded_game(
            runs_dir,
            run_id=game_summary.run_id,
            iteration=game_summary.iteration,
            game_idx=game_summary.game_idx,
        )
    except (FileNotFoundError, IndexError, ValueError):
        return []
    playable, sampled = sample_game_observations(recorded, stride=1)
    if not playable:
        return []
    found: list[_SampledPosition] = []
    for step_idx, obs in sampled:
        if len(obs.action_space.actions) < _MIN_DECISION_ACTIONS:
            continue
        kind = ActionSpaceEnum(int(obs.action_space.action_space_type)).name
        if kind_filter and kind.lower() != kind_filter:
            continue
        found.append(_SampledPosition(game_summary, recorded.hero_seat, step_idx, kind, obs))
    return found


def _collect_stratified_positions(
    runs_dir: Path,
    games: list[GameRecordSummary],
    *,
    sample_count: int,
    decision_kind: str | None,
    seed: int,
) -> list[_SampledPosition]:
    if not games or sample_count <= 0:
        return []
    rng = random.Random(seed)
    shuffled = list(games)
    rng.shuffle(shuffled)
    kind_filter = decision_kind.strip().lower() if decision_kind else None
    per_game_cap = max(2, math.ceil(sample_count / max(1, min(len(shuffled), _COMPARE_STRATA_TARGET))))
    chosen: list[_SampledPosition] = []
    overflow: list[_SampledPosition] = []
    for game_summary in shuffled:
        if len(chosen) >= sample_count:
            break
        candidates = _extract_game_positions(runs_dir, game_summary, kind_filter=kind_filter)
        if not candidates:
            continue
        rng.shuffle(candidates)
        take = min(per_game_cap, len(candidates), sample_count - len(chosen))
        chosen.extend(candidates[:take])
        overflow.extend(candidates[take:])
    if len(chosen) < sample_count and overflow:
        rng.shuffle(overflow)
        chosen.extend(overflow[: sample_count - len(chosen)])
    return chosen


def _top_action_summary(
    legal_actions: list[dict[str, object]],
    probs: list[float],
) -> tuple[str, float]:
    if not probs or not legal_actions:
        return "", 0.0
    top_idx = max(range(len(probs)), key=probs.__getitem__)
    label = str(legal_actions[top_idx]["label"]) if top_idx < len(legal_actions) else f"Action #{top_idx}"
    return label, probs[top_idx]


def _build_compare_row(
    pos: _SampledPosition,
    *,
    probs_a: list[float],
    wdl_a: dict[str, float],
    probs_b: list[float],
    wdl_b: dict[str, float],
    diff: dict[str, object],
) -> dict[str, object]:
    legal_actions = serialize_legal_actions(pos.obs, hero_seat=pos.hero_seat)
    top_action_a, top_prob_a = _top_action_summary(legal_actions, probs_a)
    top_action_b, top_prob_b = _top_action_summary(legal_actions, probs_b)
    hero_life, opp_life = seat_aligned_life(pos.obs, hero_seat=pos.hero_seat)
    win_a = float(wdl_a["win"])
    win_b = float(wdl_b["win"])
    actor_side = "hero" if int(pos.obs.agent.player_index) == pos.hero_seat else "opp"
    return {
        "run_id": pos.game.run_id,
        "iteration": pos.game.iteration,
        "game_idx": pos.game.game_idx,
        "seed": pos.game.seed,
        "step": pos.step,
        "turn": int(pos.obs.turn.turn_number),
        "decision_kind": pos.decision_kind,
        "actor_side": actor_side,
        "move_changed": bool(diff["argmax_changed"]),
        "tv": float(diff["tv"]) if isinstance(diff["tv"], (int, float)) else 0.0,
        "kl": float(diff["kl"]) if isinstance(diff["kl"], (int, float)) else 0.0,
        "top_action_a": top_action_a,
        "top_prob_a": top_prob_a,
        "top_action_b": top_action_b,
        "top_prob_b": top_prob_b,
        "win_a": win_a,
        "win_b": win_b,
        "win_delta": round(win_b - win_a, 4),
        "hero_life": hero_life,
        "opp_life": opp_life,
        "key_cards": list(pos.game.key_cards),
        "legal_actions": legal_actions,
        "probs_a": probs_a,
        "probs_b": probs_b,
    }


def _kind_stat_sort_key(item: dict[str, object]) -> tuple[int, str]:
    count = item["count"]
    return (-(count if isinstance(count, int) else 0), str(item["kind"]))


def _aggregate_by_kind(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault(str(row["decision_kind"]), []).append(row)
    stats: list[dict[str, object]] = []
    for kind, items in buckets.items():
        count = len(items)
        changed = sum(1 for item in items if item["move_changed"])
        tv_sum = sum(float(item["tv"]) for item in items if isinstance(item["tv"], (int, float)))
        stats.append(
            {
                "kind": kind,
                "count": count,
                "changed": changed,
                "change_rate": round(changed / count, 4) if count else 0.0,
                "mean_tv": round(tv_sum / count, 4) if count else 0.0,
            }
        )
    return sorted(stats, key=_kind_stat_sort_key)


def _row_sort_key(row: dict[str, object]) -> tuple[bool, float, float]:
    tv = float(row["tv"]) if isinstance(row["tv"], (int, float)) else 0.0
    delta = float(row["win_delta"]) if isinstance(row["win_delta"], (int, float)) else 0.0
    return (not bool(row["move_changed"]), -tv, -abs(delta))


def compare_checkpoints(
    engine: InferenceEngine,
    *,
    runs_dir: Path,
    index: StudioIndex,
    ckpt_a: str,
    ckpt_b: str,
    sample_count: int = 64,
    run_id: str | None = None,
    decision_kind: str | None = None,
    seed: int = 42,
) -> dict[str, object]:
    games, _ = index.query_games(GameQuery(run_id=run_id, sort="iter_desc", limit=0))
    positions = _collect_stratified_positions(
        runs_dir,
        games,
        sample_count=sample_count,
        decision_kind=decision_kind,
        seed=seed,
    )
    if not positions:
        return {
            "ckpt_a": ckpt_a,
            "ckpt_b": ckpt_b,
            "sampled_positions": 0,
            "moves_changed": 0,
            "change_rate": 0.0,
            "mean_tv": 0.0,
            "mean_kl": 0.0,
            "mean_win_delta": 0.0,
            "device": engine.preferred_device,
            "warnings": ["No playable decision positions matched the selected filter."],
            "by_decision_kind": [],
            "positions": [],
        }
    fallback_run_id = run_id or positions[0].game.run_id
    obs_list = [pos.obs for pos in positions]
    hero_seats = [pos.hero_seat for pos in positions]
    probs_a, wdls_a, dev_a, warn_a = engine.evaluate_batch_full(
        obs_list, run_id=fallback_run_id, checkpoint_name=ckpt_a, hero_seats=hero_seats
    )
    probs_b, wdls_b, dev_b, warn_b = engine.evaluate_batch_full(
        obs_list, run_id=fallback_run_id, checkpoint_name=ckpt_b, hero_seats=hero_seats
    )
    rows = [
        _build_compare_row(
            pos,
            probs_a=pa,
            wdl_a=wa,
            probs_b=pb,
            wdl_b=wb,
            diff=engine.diff_policies(pa, pb),
        )
        for pos, pa, wa, pb, wb in zip(positions, probs_a, wdls_a, probs_b, wdls_b, strict=True)
    ]
    rows.sort(key=_row_sort_key)
    total = len(rows)
    changed = sum(1 for row in rows if row["move_changed"])
    mean_tv = round(sum(float(r["tv"]) for r in rows if isinstance(r["tv"], (int, float))) / total, 4)
    mean_kl = round(sum(float(r["kl"]) for r in rows if isinstance(r["kl"], (int, float))) / total, 4)
    mean_delta = round(
        sum(float(r["win_delta"]) for r in rows if isinstance(r["win_delta"], (int, float))) / total,
        4,
    )
    warnings = [w for w in (warn_a, warn_b) if w]
    device = dev_a if dev_a != "cpu" else dev_b
    return {
        "ckpt_a": ckpt_a,
        "ckpt_b": ckpt_b,
        "sampled_positions": total,
        "moves_changed": changed,
        "change_rate": round(changed / total, 4),
        "mean_tv": mean_tv,
        "mean_kl": mean_kl,
        "mean_win_delta": mean_delta,
        "device": device,
        "warnings": warnings,
        "by_decision_kind": _aggregate_by_kind(rows),
        "positions": rows,
    }

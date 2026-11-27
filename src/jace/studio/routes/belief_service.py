"""Service helpers for Studio opponent belief inference, overrides, and local search.

Loads saved `FormatPrior` and optional `BeliefNet` artifacts from
`<runs-dir>/belief/{format}/`, falls back to a dynamic prior built from
recorded run decks and indexed cards, computes Turn-0 `V_0` keep/mulligan
deltas, and executes checkpoint-backed or heuristic multi-world local search.
"""

from __future__ import annotations

import math
import re
import time
from typing import TYPE_CHECKING

import managym
import torch
from fastapi import HTTPException
from managym.enums import ActionSpaceEnum
from pydantic import BaseModel, Field

from jace.belief import (
    BeliefNet,
    BeliefOverride,
    BeliefState,
    FormatPrior,
    evaluate_opening_hand_vs_v0,
    fit_format_prior,
    load_belief_net,
    load_format_prior,
    project_belief_overrides,
)
from jace.forks.search import (
    SearchConfig,
    SearchResult,
    build_belief_from_obs,
    format_pin_materialization_warnings,
    run_local_search,
    sample_opponent_worlds,
)
from jace.studio.board import heuristic_win_estimate
from jace.studio.replay import (
    RecordedGame,
    ReplayService,
    engine_step_errors,
    load_recorded_games,
    serialize_legal_actions,
)
from jace.studio.runs import resolve_run_dir

if TYPE_CHECKING:
    from pathlib import Path

    from jace.studio.index import StudioIndex
    from jace.studio.inference import InferenceEngine

__all__ = (
    "DEFAULT_V0_EQUITY",
    "BeliefInferRequest",
    "BeliefOverrideModel",
    "BeliefProjectRequest",
    "SearchEvaluateRequest",
    "compute_turn0_v0",
    "execute_search",
    "format_search_actions",
    "load_step_belief",
    "project_or_422",
    "serialize_belief_cards",
    "to_belief_override",
)

DEFAULT_V0_EQUITY = 0.5
_FORMAT_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
_BASIC_LAND_LIMIT = 20.0
_DEFAULT_COPY_LIMIT = 4.0
_MAX_RUN_DECKS = 128
_Z_95 = 1.959963984540054


class BeliefOverrideModel(BaseModel):
    pinned_in_hand: dict[str, int] = Field(default_factory=dict)
    excluded_cards: list[str] = Field(default_factory=list)
    copy_multipliers: dict[str, float] = Field(default_factory=dict)


class BeliefInferRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=256)
    iteration: int = Field(ge=0)
    game_idx: int = Field(ge=0)
    step: int = Field(ge=0)
    format_id: str | None = Field(default=None, max_length=128)
    prior_only: bool = False


class BeliefProjectRequest(BeliefInferRequest):
    override: BeliefOverrideModel = Field(default_factory=BeliefOverrideModel)


class SearchEvaluateRequest(BeliefInferRequest):
    ckpt_a: str = Field(default="heuristic", min_length=1, max_length=300)
    worlds: int = Field(default=16, ge=1, le=256)
    horizon_plies: int = Field(default=24, ge=1, le=240)
    seed: int = 42
    override: BeliefOverrideModel | None = None


def to_belief_override(model: BeliefOverrideModel | None) -> BeliefOverride | None:
    if model is None:
        return None
    return BeliefOverride(
        pinned_in_hand={str(k): int(v) for k, v in model.pinned_in_hand.items()},
        excluded_cards=frozenset(str(c) for c in model.excluded_cards),
        copy_multipliers={str(k): float(v) for k, v in model.copy_multipliers.items()},
    )


def _validate_format_id(format_id: str) -> str:
    cleaned = format_id.strip()
    if not cleaned or not _FORMAT_ID_RE.match(cleaned) or ".." in cleaned:
        raise HTTPException(status_code=400, detail=f"invalid format_id: {format_id!r}")
    return cleaned


def _collect_run_decks(runs_dir: Path, game: RecordedGame) -> list[dict[str, int]]:
    decks = [dict(spec.decklist) for spec in game.players if spec.decklist]
    try:
        run_dir = resolve_run_dir(runs_dir, game.run_id)
    except (FileNotFoundError, ValueError):
        return decks
    jsonl_path = run_dir / "games" / f"eval-iter-{game.iteration:04d}.jsonl"
    if not jsonl_path.is_file():
        return decks
    for rec in load_recorded_games(jsonl_path, run_id=game.run_id, iteration=game.iteration):
        decks.extend(dict(spec.decklist) for spec in rec.players if spec.decklist)
        if len(decks) >= _MAX_RUN_DECKS:
            break
    return decks[:_MAX_RUN_DECKS]


def _copy_limit_for_card(name: str, decks: list[dict[str, int]]) -> float:
    base = _BASIC_LAND_LIMIT if name in _BASIC_LANDS else _DEFAULT_COPY_LIMIT
    return max(base, max((float(deck.get(name, 0)) for deck in decks), default=0.0))


def _build_dynamic_prior(
    *, runs_dir: Path, index: StudioIndex, game: RecordedGame, format_id: str | None
) -> FormatPrior:
    decks = _collect_run_decks(runs_dir, game)
    card_set = {name for deck in decks for name in deck} | set(index.available_cards(run_id=game.run_id))
    card_names = tuple(sorted(card_set))
    if not card_names:
        raise HTTPException(status_code=422, detail="no cards available to build dynamic format prior")
    if not decks:
        decks = [{card_names[0]: 40}]
    name_to_idx = {name: idx for idx, name in enumerate(card_names)}
    matrix = torch.zeros((len(decks), len(card_names)), dtype=torch.float32)
    for row_idx, deck in enumerate(decks):
        for card_name, qty in deck.items():
            if (col_idx := name_to_idx.get(card_name)) is not None and qty > 0:
                matrix[row_idx, col_idx] = float(qty)
    limits = torch.tensor([_copy_limit_for_card(name, decks) for name in card_names], dtype=torch.float32)
    return fit_format_prior(format_id or "dynamic", card_names, matrix, copy_limits=limits)


def _load_or_build_belief_artifacts(
    *, runs_dir: Path, index: StudioIndex, game: RecordedGame, format_id: str | None
) -> tuple[FormatPrior, BeliefNet | None]:
    safe_fmt = _validate_format_id(format_id) if format_id is not None else None
    belief_net: BeliefNet | None = None
    prior: FormatPrior | None = None
    if safe_fmt is not None:
        fmt_dir = (runs_dir / "belief" / safe_fmt).resolve()
        if (prior_path := fmt_dir / "prior.pt").is_file():
            prior = load_format_prior(prior_path)
        if (net_path := fmt_dir / "belief_net.pt").is_file():
            belief_net = load_belief_net(net_path)
    if prior is None:
        prior = _build_dynamic_prior(runs_dir=runs_dir, index=index, game=game, format_id=safe_fmt)
    return prior, belief_net


def _replay_step_or_404(
    replays: ReplayService, req: BeliefInferRequest
) -> tuple[RecordedGame, managym.Env, managym.Observation]:
    try:
        game, env, obs, *_ = replays.replay_to_step(
            run_id=req.run_id, iteration=req.iteration, game_idx=req.game_idx, step=req.step
        )
    except (FileNotFoundError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return game, env, obs


def compute_turn0_v0(
    obs: managym.Observation,
    *,
    inference: InferenceEngine,
    game: RecordedGame,
    ckpt_a: str | None,
    v0_equity: float,
) -> dict[str, object] | None:
    is_mull = ActionSpaceEnum(int(obs.action_space.action_space_type)).name == "MULLIGAN"
    if int(obs.turn.turn_number) != 0 and not is_mull:
        return None
    resolved_ckpt = ckpt_a or inference.find_checkpoint_for_iteration(game.run_id, game.iteration)
    eval_pair = inference.evaluate_pair(
        obs, run_id=game.run_id, hero_seat=game.hero_seat, ckpt_a=resolved_ckpt, hide_actions=False
    )
    primary = eval_pair.get("primary")
    wdl = primary.get("wdl") if isinstance(primary, dict) else None
    keep_v = float(wdl["win"]) if isinstance(wdl, dict) and isinstance(wdl.get("win"), (int, float)) else 0.5
    compared = evaluate_opening_hand_vs_v0(keep_v, v0_equity=v0_equity)
    return {
        "is_turn0": True,
        "v0_equity": round(v0_equity, 4),
        "keep_v": round(float(compared["keep_v"]), 4),
        "delta": round(float(compared["delta"]), 4),
        "recommendation": "Keep" if compared["recommendation"] == "keep" else "Mulligan",
    }


def _card_entry_sort_key(row: dict[str, object]) -> tuple[bool, float, float, str]:
    pinned, hand_prob, deck_copies = row["pinned"], row["hand_prob"], row["deck_copies"]
    return (
        not (isinstance(pinned, int) and pinned > 0),
        -(float(hand_prob) if isinstance(hand_prob, (int, float)) else 0.0),
        -(float(deck_copies) if isinstance(deck_copies, (int, float)) else 0.0),
        str(row["name"]),
    )


def serialize_belief_cards(
    belief: BeliefState, *, revealed: torch.Tensor, override: BeliefOverride | None
) -> tuple[list[dict[str, object]], int, int, float]:
    pinned_map = override.pinned_in_hand if override is not None else {}
    excluded_set = override.excluded_cards if override is not None else frozenset()
    mult_map = override.copy_multipliers if override is not None else {}
    hand_vals = belief.hand_probs.detach().cpu().tolist()
    deck_vals = belief.deck_copies.detach().cpu().tolist()
    rev_vals = revealed.detach().cpu().tolist()

    cards: list[dict[str, object]] = []
    unedited_count, unedited_hand_sum = 0, 0.0
    for name, hand_p, deck_c, rev_c in zip(belief.card_names, hand_vals, deck_vals, rev_vals, strict=True):
        pin, excl, mult = int(pinned_map.get(name, 0)), name in excluded_set, float(mult_map.get(name, 1.0))
        if not (pin > 0 or excl or mult != 1.0):
            unedited_count += 1
            unedited_hand_sum += float(hand_p)
        cards.append(
            {
                "name": name,
                "hand_prob": round(float(hand_p), 4),
                "deck_copies": round(float(deck_c), 4),
                "revealed_count": round(float(rev_c)),
                "pinned": pin,
                "excluded": excl,
                "multiplier": mult,
            }
        )
    cards.sort(key=_card_entry_sort_key)
    pinned_total = sum(v for v in pinned_map.values() if v > 0)
    unedited_mass = (
        min(1.0, max(0.0, unedited_hand_sum / float(belief.opp_hand_size)))
        if belief.opp_hand_size > 0
        else (1.0 if unedited_count > 0 else 0.0)
    )
    return cards, pinned_total, unedited_count, round(unedited_mass, 4)


def project_or_422(
    belief: BeliefState, override: BeliefOverride, *, revealed: torch.Tensor, copy_limits: torch.Tensor | None
) -> BeliefState:
    try:
        return project_belief_overrides(belief, (override,), revealed_counts=revealed, copy_limits=copy_limits)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _heuristic_leaf_score(cloned: managym.Env, obs: managym.Observation, *, hero_seat: int) -> float:
    if not cloned.is_game_over():
        return heuristic_win_estimate(obs, hero_seat=hero_seat)
    winner = cloned.winner_index()
    return 0.5 if winner is None else (1.0 if int(winner) == hero_seat else 0.0)


def _score_heuristic_rollout(
    world_env: managym.Env, *, action_idx: int, hero_seat: int, horizon_plies: int, heuristic: managym.Heuristic
) -> float | None:
    cloned = world_env.clone_env()
    try:
        obs, _, done, truncated, _ = cloned.step(action_idx)
        for _ in range(horizon_plies):
            if done or truncated or cloned.is_game_over():
                break
            obs, _, done, truncated, _ = cloned.step(int(heuristic.select_action(obs)))
    except engine_step_errors():
        return None
    return _heuristic_leaf_score(cloned, obs, hero_seat=hero_seat)


def _mean_and_ci95(scores: list[float]) -> tuple[float, tuple[float, float]]:
    if not scores:
        return 0.5, (0.0, 1.0)
    mean_val = sum(scores) / len(scores)
    if len(scores) == 1:
        return mean_val, (max(0.0, mean_val - 0.5), min(1.0, mean_val + 0.5))
    var = sum((s - mean_val) ** 2 for s in scores) / (len(scores) - 1)
    half = _Z_95 * math.sqrt(max(0.0, var) / len(scores))
    return mean_val, (max(0.0, mean_val - half), min(1.0, mean_val + half))


def _run_heuristic_search(
    env: managym.Env,
    obs: managym.Observation,
    *,
    belief: BeliefState,
    config: SearchConfig,
    seed: int,
    revealed: torch.Tensor,
    copy_limits: torch.Tensor | None,
    override: BeliefOverride | None,
) -> SearchResult:
    t0 = time.perf_counter()
    num_actions = len(obs.action_space.actions)
    heuristic, horizon = managym.Heuristic(), config.horizon_plies or 24
    h_choice = int(heuristic.select_action(obs)) if num_actions > 0 else 0
    priors = tuple(1.0 if idx == h_choice else 0.0 for idx in range(num_actions))
    hero_seat, opp_seat = int(obs.agent.player_index), int(obs.opponent.player_index)
    unmaterialized_pins: set[str] = set()
    worlds = sample_opponent_worlds(
        env,
        belief,
        opp_seat=opp_seat,
        k=config.worlds,
        seed=seed,
        revealed_counts=revealed,
        copy_limits=copy_limits,
        overrides=override,
        already_projected=True,
        unmaterialized_pins=unmaterialized_pins,
    )
    per_action: list[list[float]] = [[] for _ in range(num_actions)]
    for world_env in worlds:
        for act_idx in range(num_actions):
            score = _score_heuristic_rollout(
                world_env, action_idx=act_idx, hero_seat=hero_seat, horizon_plies=horizon, heuristic=heuristic
            )
            if score is not None:
                per_action[act_idx].append(score)
    stats = [_mean_and_ci95(scores) for scores in per_action]
    values, cis = tuple(item[0] for item in stats), tuple(item[1] for item in stats)
    chosen = max(range(num_actions), key=lambda i: (values[i], priors[i])) if num_actions else 0
    warnings = format_pin_materialization_warnings(unmaterialized_pins)
    return SearchResult(
        chosen, values, cis, priors, len(worlds), horizon, (time.perf_counter() - t0) * 1000.0, warnings
    )


def format_search_actions(obs: managym.Observation, result: SearchResult, *, hero_seat: int) -> list[dict[str, object]]:
    actions = serialize_legal_actions(obs, hero_seat=hero_seat)
    priors = [float(result.policy_priors[i]) if i < len(result.policy_priors) else 0.0 for i in range(len(actions))]
    values = [float(result.action_values[i]) if i < len(result.action_values) else 0.5 for i in range(len(actions))]
    policy_top_idx = max(range(len(priors)), key=priors.__getitem__) if priors else 0
    policy_top_wr = values[policy_top_idx] if values else 0.5
    rows: list[dict[str, object]] = []
    for idx, act in enumerate(actions):
        wr = values[idx]
        ci_lo, ci_hi = result.action_ci95[idx] if idx < len(result.action_ci95) else (0.0, 1.0)
        rows.append(
            {
                "index": idx,
                "label": str(act["label"]),
                "search_wr": round(wr, 4),
                "ci_low": round(float(ci_lo), 4),
                "ci_high": round(float(ci_hi), 4),
                "policy_prob": round(priors[idx], 4),
                # Win-rate gain over the raw policy's top-ranked action (same win-probability units).
                "delta_vs_policy": round(wr - policy_top_wr, 4),
            }
        )
    return rows


def load_step_belief(
    req: BeliefInferRequest, *, runs_dir: Path, replays: ReplayService, index: StudioIndex
) -> tuple[RecordedGame, managym.Env, managym.Observation, FormatPrior, BeliefState, torch.Tensor, bool]:
    game, env, obs = _replay_step_or_404(replays, req)
    prior, belief_net = _load_or_build_belief_artifacts(
        runs_dir=runs_dir, index=index, game=game, format_id=req.format_id
    )
    acting_seat = int(obs.agent.player_index)
    belief, revealed = build_belief_from_obs(
        obs, prior=prior, hero_seat=acting_seat, belief_net=belief_net, prior_only=req.prior_only
    )
    return game, env, obs, prior, belief, revealed, belief_net is not None


def execute_search(
    req: SearchEvaluateRequest,
    *,
    inference: InferenceEngine,
    env: managym.Env,
    obs: managym.Observation,
    belief: BeliefState,
    revealed: torch.Tensor,
    copy_limits: torch.Tensor | None,
    override: BeliefOverride | None,
) -> SearchResult:
    config = SearchConfig(worlds=req.worlds, horizon_plies=req.horizon_plies, prior_only=req.prior_only)
    if req.ckpt_a == "heuristic":
        return _run_heuristic_search(
            env,
            obs,
            belief=belief,
            config=config,
            seed=req.seed,
            revealed=revealed,
            copy_limits=copy_limits,
            override=override,
        )
    try:
        return inference.run_with_checkpoint(
            run_id=req.run_id,
            checkpoint_name=req.ckpt_a,
            fn=lambda scope, obs_space: run_local_search(
                env,
                obs,
                scope=scope,
                obs_space=obs_space,
                belief=belief,
                config=config,
                seed=req.seed,
                revealed_counts=revealed,
                copy_limits=copy_limits,
                overrides=override,
                already_projected=True,
            ),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"checkpoint not found: {req.ckpt_a} ({exc})") from exc

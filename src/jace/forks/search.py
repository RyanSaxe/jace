"""Belief-conditioned local rollout search and opponent-world sampling.

Samples opponent hidden hands from public-observation `BeliefState`
distributions and evaluates legal root actions across paired CRN worlds in one
wide batched forward pass per ply.

Engine boundary (`managym 0.2.6`): `env.determinize()` rejects closed-decklist
environments and `scenario_force_card_in_hand()` moves existing copies from the
opponent's instantiated library (or graveyard) without creating absent card
definitions. `_materialize_world` strictly guards the opponent's public
graveyard (never pulling graveyard cards into hand), fills hidden hand slots
using belief-weighted sampling over available library copies, and surfaces
explicit warnings whenever a pinned counterfactual card cannot be materialized
in the current instantiated decklist.
"""

from __future__ import annotations

import contextlib
import math
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from managym.enums import ZoneEnum

from jace.belief import (
    GAMEPLAY_SIGNAL_DIM,
    BayesianDeckPoolPrior,
    BeliefNet,
    BeliefOverride,
    BeliefState,
    FormatPrior,
    PriorBackendName,
    SetTransformerPrior,
    infer_belief_state,
    project_belief_overrides,
)
from jace.eval.deals import wilson_interval
from jace.forks.batch import (
    _batched_eval,
    _engine_error_types,
    _hero_win_prob,
    _safe_clone,
    _safe_step,
    _terminal_score,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import managym
    from manabot.backends.torch.policy import PolicyInference
    from manabot.env.observation import ObservationSpace
    from torch import Tensor

__all__ = (
    "SearchConfig",
    "SearchResult",
    "build_belief_from_obs",
    "extract_public_belief_inputs",
    "run_local_search",
    "sample_opponent_worlds",
)

_Z_95 = 1.959963984540054
_WORLD_STRIDE = 9973
_BRANCH_MULT = 1_000_003
_TIE_XOR = 0x5A5A
_REFILL_FACTOR = 8
_COLORS = 5
_TURN_SCALE = 20.0
_PHASE_SCALE = 10.0
_STEP_SCALE = 15.0
_LIFE_SCALE = 20.0
_BASIC_SLOTS = {"Plains": 0, "Island": 1, "Swamp": 2, "Mountain": 3, "Forest": 4}
_PUBLIC_ZONES = frozenset(
    int(z) for z in (ZoneEnum.BATTLEFIELD, ZoneEnum.GRAVEYARD, ZoneEnum.STACK, ZoneEnum.EXILE, ZoneEnum.COMMAND)
)


@dataclass(frozen=True, slots=True)
class SearchConfig:
    worlds: int = 16
    horizon_plies: int | None = 24
    max_plies: int = 240
    temperature: float = 0.0
    prior_only: bool = False


@dataclass(frozen=True, slots=True)
class SearchResult:
    chosen_action: int
    action_values: tuple[float, ...]
    action_ci95: tuple[tuple[float, float], ...]
    policy_priors: tuple[float, ...]
    worlds_sampled: int
    rollout_horizon: int | None
    elapsed_ms: float
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class _SearchBranch:
    world_idx: int
    action_idx: int
    hero_idx: int
    env: managym.Env
    obs: managym.Observation | None
    rng: random.Random
    done: bool = False
    failed: bool = False
    steps: int = 0
    leaf_value: float | None = None


def _add_public_counts(
    counts: Tensor, perms: Sequence[managym.Permanent], *, cards: Sequence[managym.Card], idx_map: dict[str, int]
) -> None:
    for perm in perms:
        if (idx := None if bool(perm.is_token) else idx_map.get(str(perm.characteristics.name))) is not None:
            counts[idx] += 1.0
    for card in cards:
        ok = not bool(card.is_token) and int(card.zone) in _PUBLIC_ZONES
        if (idx := idx_map.get(str(card.characteristics.name)) if ok else None) is not None:
            counts[idx] += 1.0


def _opp_board_tells(
    opp: managym.Player, perms: Sequence[managym.Permanent]
) -> tuple[list[float], float, float, float, float]:
    pool = [float(v) for v in opp.mana]
    colors = [pool[i] if i < len(pool) else 0.0 for i in range(_COLORS)]
    untapped = attacked = blocked = recent_lands = 0.0
    for p in perms:
        if p.characteristics.card_types.is_land and not bool(p.tapped):
            untapped += 1.0
            if (slot := _BASIC_SLOTS.get(str(p.characteristics.name))) is not None:
                colors[slot] += 1.0
        attacked = 1.0 if (attacked or bool(p.attacking) or bool(p.attacked_this_turn)) else 0.0
        blocked = 1.0 if (blocked or bool(p.blocking)) else 0.0
        recent_lands += 1.0 if (p.characteristics.card_types.is_land and bool(p.entered_this_turn)) else 0.0
    return colors, untapped + sum(pool), attacked, blocked, recent_lands


def extract_public_belief_inputs(
    obs: managym.Observation, *, card_names: Sequence[str], hero_seat: int
) -> tuple[Tensor, Tensor, int, int]:
    if int(obs.agent.player_index) == hero_seat:
        hero, opp, perms, cards = obs.agent, obs.opponent, obs.opponent_permanents, obs.opponent_cards
    else:
        hero, opp, perms, cards = obs.opponent, obs.agent, obs.agent_permanents, obs.agent_cards
    revealed = torch.zeros(len(card_names), dtype=torch.float32)
    _add_public_counts(revealed, perms, cards=cards, idx_map={n: i for i, n in enumerate(card_names)})
    colors, total_open, attacked, blocked, recent_lands = _opp_board_tells(opp, perms)
    turn_sig = (
        obs.turn.turn_number / _TURN_SCALE,
        int(obs.turn.phase) / _PHASE_SCALE,
        int(obs.turn.step) / _STEP_SCALE,
    )
    opp_sig = (0.0 if bool(opp.is_active) else 1.0, attacked, blocked, float(opp.spells_cast), recent_lands)
    raw_sig = [*colors, total_open, *turn_sig, *opp_sig, hero.life / _LIFE_SCALE, opp.life / _LIFE_SCALE]
    sig = torch.tensor(raw_sig[:GAMEPLAY_SIGNAL_DIM], dtype=torch.float32)
    return revealed, sig, int(opp.zone_counts[int(ZoneEnum.HAND)]), int(opp.zone_counts[int(ZoneEnum.LIBRARY)])


def _prior_only_belief(
    raw_deck: Tensor, *, card_names: tuple[str, ...], hand_sz: int, lib_sz: int, backend_name: PriorBackendName
) -> BeliefState:
    unseen = hand_sz + lib_sz
    if unseen <= 0:
        z = torch.zeros(len(card_names), dtype=torch.float32)
        return BeliefState(card_names, z, z.clone(), hand_sz, lib_sz, backend_name)
    w = raw_deck.to(dtype=torch.float32).clamp_min(0.0)
    w = w if float(w.sum().item()) > 0.0 else torch.ones_like(w)
    deck = w * (float(unseen) / float(w.sum().item()))
    return BeliefState(card_names, deck, deck * (float(hand_sz) / float(unseen)), hand_sz, lib_sz, backend_name)


def build_belief_from_obs(
    obs: managym.Observation,
    *,
    prior: FormatPrior,
    hero_seat: int,
    belief_net: BeliefNet | None = None,
    prior_only: bool = False,
    prior_backend: BayesianDeckPoolPrior | None = None,
) -> tuple[BeliefState, Tensor]:
    revealed, signals, hand_sz, lib_sz = extract_public_belief_inputs(
        obs, card_names=prior.card_names, hero_seat=hero_seat
    )
    backend = prior_backend if prior_backend is not None else BayesianDeckPoolPrior()
    name: PriorBackendName = "set_transformer" if isinstance(backend, SetTransformerPrior) else "bayesian_pool"
    if not prior_only and belief_net is not None:
        state = infer_belief_state(
            belief_net,
            prior=prior,
            revealed_counts=revealed,
            gameplay_signals=signals,
            opp_hand_size=hand_sz,
            opp_library_size=lib_sz,
            prior_backend=backend,
        )
        return state, revealed
    state = _prior_only_belief(
        backend.posterior_deck(revealed, prior),
        card_names=prior.card_names,
        hand_sz=hand_sz,
        lib_sz=lib_sz,
        backend_name=name,
    )
    return state, revealed


def _opp_graveyard_counts(env: managym.Env, *, opp_seat: int) -> dict[str, int]:
    viewer = 1 - opp_seat
    obs = env.observation_for_player(viewer)
    gy_zone = int(ZoneEnum.GRAVEYARD)
    counts: dict[str, int] = {}
    for card in obs.opponent_cards:
        if not bool(card.is_token) and int(card.zone) == gy_zone:
            name = str(card.characteristics.name)
            counts[name] = counts.get(name, 0) + 1
    return counts


def _sample_hand_cards(
    belief: BeliefState, *, overrides: BeliefOverride | None, rng: random.Random
) -> tuple[list[str], list[str], list[tuple[str, float]]]:
    excluded = overrides.excluded_cards if overrides is not None else frozenset()
    pinned_map = overrides.pinned_in_hand if overrides is not None else {}
    pinned = [n for n, c in sorted(pinned_map.items()) for _ in range(c)]
    probs = belief.hand_probs.detach().cpu().tolist()
    copies = belief.deck_copies.detach().cpu().tolist()
    entries = [
        (n, float(p), max(0, max(1, math.ceil(float(q))) - pinned_map.get(n, 0)))
        for n, p, q in zip(belief.card_names, probs, copies, strict=True)
        if n not in excluded and float(p) > 0.0
    ]
    names, weights, avail = [e[0] for e in entries], [e[1] for e in entries], [e[2] for e in entries]
    drawn: list[str] = []
    for _ in range(max(0, int(belief.opp_hand_size) - len(pinned))):
        if not names:
            break
        active = [w if c > 0 else 0.0 for w, c in zip(weights, avail, strict=True)]
        idx = rng.choices(range(len(names)), weights=active if sum(active) > 0.0 else weights, k=1)[0]
        avail[idx] = max(0, avail[idx] - 1)
        drawn.append(names[idx])
    return pinned, drawn, list(zip(names, weights, strict=True))


def _try_force_from_library(
    clone: managym.Env,
    name: str,
    *,
    opp_seat: int,
    gy_counts: dict[str, int],
    errors: tuple[type[BaseException], ...],
) -> tuple[managym.Env, bool]:
    min_gy = gy_counts.get(name, 0)
    backup = clone.clone_env() if min_gy > 0 else None
    try:
        clone.scenario_force_card_in_hand(opp_seat, name)
    except errors:
        return clone, False
    if backup is not None:
        after_gy = _opp_graveyard_counts(clone, opp_seat=opp_seat).get(name, 0)
        if after_gy < min_gy:
            return backup, False
    return clone, True


def _refill_slots(
    clone: managym.Env,
    pool: Sequence[tuple[str, float]],
    *,
    opp_seat: int,
    gy_counts: dict[str, int],
    needed: int,
    failed: set[str],
    rng: random.Random,
    errors: tuple[type[BaseException], ...],
) -> managym.Env:
    rem = [(n, w) for n, w in pool if n not in failed and w > 0.0]
    placed = 0
    for _ in range(max(len(rem), needed * _REFILL_FACTOR)):
        if placed >= needed or not rem:
            break
        name = rem[rng.choices(range(len(rem)), weights=[w for _, w in rem], k=1)[0]][0]
        clone, ok = _try_force_from_library(clone, name, opp_seat=opp_seat, gy_counts=gy_counts, errors=errors)
        if ok:
            placed += 1
        else:
            failed.add(name)
            rem = [item for item in rem if item[0] != name]
    return clone


def _place_cards(
    clone: managym.Env,
    cards: Sequence[str],
    *,
    opp_seat: int,
    gy_counts: dict[str, int],
    failed: set[str],
    errors: tuple[type[BaseException], ...],
    missing_out: set[str] | None = None,
) -> tuple[managym.Env, int]:
    placed = 0
    for name in cards:
        if name not in failed:
            clone, ok = _try_force_from_library(clone, name, opp_seat=opp_seat, gy_counts=gy_counts, errors=errors)
            if ok:
                placed += 1
                continue
            failed.add(name)
        if missing_out is not None:
            missing_out.add(name)
    return clone, placed


def _materialize_world(
    env: managym.Env,
    belief: BeliefState,
    *,
    opp_seat: int,
    gy_counts: dict[str, int],
    overrides: BeliefOverride | None,
    world_seed: int,
    unmaterialized_pins: set[str],
) -> managym.Env:
    rng = random.Random(world_seed)
    pinned, drawn, pool = _sample_hand_cards(belief, overrides=overrides, rng=rng)
    clone, errors = env.clone_env(), _engine_error_types()
    clone.scenario_clear_hand(opp_seat)
    failed: set[str] = set()
    clone, placed_pins = _place_cards(
        clone,
        pinned,
        opp_seat=opp_seat,
        gy_counts=gy_counts,
        failed=failed,
        errors=errors,
        missing_out=unmaterialized_pins,
    )
    clone, placed_drawn = _place_cards(
        clone, drawn, opp_seat=opp_seat, gy_counts=gy_counts, failed=failed, errors=errors
    )
    placed = placed_pins + placed_drawn
    if placed < belief.opp_hand_size:
        clone = _refill_slots(
            clone,
            pool,
            opp_seat=opp_seat,
            gy_counts=gy_counts,
            needed=belief.opp_hand_size - placed,
            failed=failed,
            rng=rng,
            errors=errors,
        )
    with contextlib.suppress(*errors):
        clone.scenario_refresh()
    clone.set_seed(world_seed)
    return clone


def sample_opponent_worlds(
    env: managym.Env,
    belief: BeliefState,
    *,
    opp_seat: int,
    k: int,
    seed: int,
    revealed_counts: Tensor | None = None,
    copy_limits: Tensor | None = None,
    overrides: BeliefOverride | None = None,
    already_projected: bool = False,
    unmaterialized_pins: set[str] | None = None,
) -> list[managym.Env]:
    eff = belief
    if overrides is not None and not already_projected:
        rev = (
            revealed_counts if revealed_counts is not None else torch.zeros(len(belief.card_names), dtype=torch.float32)
        )
        eff = project_belief_overrides(belief, (overrides,), revealed_counts=rev, copy_limits=copy_limits)
    gy_counts = _opp_graveyard_counts(env, opp_seat=opp_seat)
    missing_pins = unmaterialized_pins if unmaterialized_pins is not None else set()
    return [
        _materialize_world(
            env,
            eff,
            opp_seat=opp_seat,
            gy_counts=gy_counts,
            overrides=overrides,
            world_seed=seed + w * _WORLD_STRIDE,
            unmaterialized_pins=missing_pins,
        )
        for w in range(max(0, k))
    ]


def _spawn_branches(
    worlds: Sequence[managym.Env], *, num_actions: int, hero_idx: int, seed: int
) -> list[_SearchBranch]:
    branches: list[_SearchBranch] = []
    for w_idx, world_env in enumerate(worlds):
        pseed = seed * _BRANCH_MULT + w_idx * _WORLD_STRIDE
        for a_idx in range(num_actions):
            b_env = _safe_clone(world_env)
            if b_env is not None:
                b_env.set_seed(pseed)
                nobs, done, failed = _safe_step(b_env, a_idx)
                br = _SearchBranch(w_idx, a_idx, hero_idx, b_env, nobs, random.Random(pseed), done, failed, 1)
            else:
                br = _SearchBranch(
                    w_idx, a_idx, hero_idx, world_env, None, random.Random(pseed), done=True, failed=True
                )
            branches.append(br)
    return branches


def _step_branches(
    branches: Sequence[_SearchBranch], *, scope: PolicyInference, obs_space: ObservationSpace, config: SearchConfig
) -> None:
    limit = config.max_plies if config.horizon_plies is None else min(config.max_plies, config.horizon_plies)
    for _ in range(limit + 1):
        active = [b for b in branches if not b.done and b.obs is not None]
        if not active:
            break
        probs_b, vals_b, _ = _batched_eval(scope, obs_space=obs_space, observations=[b.obs for b in active if b.obs])
        for b, probs, val in zip(active, probs_b, vals_b, strict=True):
            if not probs:
                b.done, b.failed = True, True
            elif b.steps >= limit:
                b.leaf_value = _hero_win_prob(b.env, hero_idx=b.hero_idx, active_value=val)
                b.done = True
            else:
                b.obs, b.done, b.failed = _safe_step(b.env, b.rng.choices(range(len(probs)), weights=probs, k=1)[0])
                b.steps += 1


def _action_stats(scores: Sequence[float]) -> tuple[float, tuple[float, float]]:
    if not scores:
        return 0.5, (0.0, 1.0)
    mean_v = sum(scores) / len(scores)
    if len(scores) == 1:
        return mean_v, wilson_interval(mean_v, 1)
    var = sum((s - mean_v) ** 2 for s in scores) / (len(scores) - 1)
    half = _Z_95 * math.sqrt(max(0.0, var) / len(scores))
    return mean_v, (max(0.0, mean_v - half), min(1.0, mean_v + half))


def _select_action(values: Sequence[float], *, priors: Sequence[float], temperature: float, seed: int) -> int:
    if temperature <= 0.0:
        return max(range(len(values)), key=lambda i: (values[i], priors[i] if i < len(priors) else 0.0))
    hi = max(values)
    weights = [math.exp((v - hi) / temperature) for v in values]
    return random.Random(seed ^ _TIE_XOR).choices(range(len(values)), weights=weights, k=1)[0]


def format_pin_materialization_warnings(unmaterialized_pins: set[str]) -> tuple[str, ...]:
    if not unmaterialized_pins:
        return ()
    names = ", ".join(sorted(unmaterialized_pins))
    msg = (
        f"Could not materialize pinned card(s) absent from the instantiated opponent library in managym 0.2.6: {names}."
    )
    return (msg,)


def run_local_search(
    env: managym.Env,
    obs: managym.Observation,
    *,
    scope: PolicyInference,
    obs_space: ObservationSpace,
    belief: BeliefState,
    config: SearchConfig,
    seed: int,
    revealed_counts: Tensor | None = None,
    copy_limits: Tensor | None = None,
    overrides: BeliefOverride | None = None,
    already_projected: bool = False,
) -> SearchResult:
    t0 = time.perf_counter()
    n_act = len(obs.action_space.actions)
    probs_list, _, _ = _batched_eval(scope, obs_space=obs_space, observations=[obs])
    priors = probs_list[0] if probs_list and probs_list[0] else tuple([1.0 / max(1, n_act)] * n_act)
    if n_act <= 1 or config.worlds <= 0:
        best_prior = max(range(n_act), key=lambda i: priors[i]) if n_act > 0 else 0
        ms = (time.perf_counter() - t0) * 1000.0
        return SearchResult(
            best_prior, (0.5,) * n_act, ((0.0, 1.0),) * n_act, tuple(priors), 0, config.horizon_plies, ms
        )
    hero, opp = int(obs.agent.player_index), int(obs.opponent.player_index)
    unmaterialized_pins: set[str] = set()
    worlds = sample_opponent_worlds(
        env,
        belief,
        opp_seat=opp,
        k=config.worlds,
        seed=seed,
        revealed_counts=revealed_counts,
        copy_limits=copy_limits,
        overrides=overrides,
        already_projected=already_projected,
        unmaterialized_pins=unmaterialized_pins,
    )
    branches = _spawn_branches(worlds, num_actions=n_act, hero_idx=hero, seed=seed)
    _step_branches(branches, scope=scope, obs_space=obs_space, config=config)
    per_act: list[list[float]] = [[] for _ in range(n_act)]
    for b in branches:
        if not b.failed:
            per_act[b.action_idx].append(_terminal_score(b.env, hero_idx=hero, leaf_value=b.leaf_value))
    stats = [_action_stats(s) for s in per_act]
    vals, ci95 = tuple(s[0] for s in stats), tuple(s[1] for s in stats)
    chosen = _select_action(vals, priors=priors, temperature=config.temperature, seed=seed)
    ms = (time.perf_counter() - t0) * 1000.0
    warnings = format_pin_materialization_warnings(unmaterialized_pins)
    return SearchResult(chosen, vals, ci95, tuple(priors), len(worlds), config.horizon_plies, ms, warnings)

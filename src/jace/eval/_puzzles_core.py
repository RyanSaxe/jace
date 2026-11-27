"""Format-agnostic tactical puzzle CRN playout verification and scoring.

Mines decision states without hardcoding card names or set codes so the same pipeline
applies across Limited and Constructed formats. Every candidate passes a pre-filter
(`0.20 <= V(s) <= 0.80`) and a strict paired Common Random Numbers (CRN) determinized
playout gate (`margin >= 0.15` and `paired_ci_low > 0.0` over `max_plies >= 240`).
Board summaries, puzzle record formatting, and locked JSONL persistence live in
`jace.eval._puzzles_render` and are re-exported here.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import managym
import torch
from manabot.backends.torch.bot import evaluate_observation

from jace.eval._puzzles_render import (
    PuzzleRecord,
    PuzzleScoreRow,
    PuzzleSource,
    PuzzleVerificationResult,
    append_puzzle,
    build_puzzle_record,
    load_puzzles,
    puzzles_path,
)

if TYPE_CHECKING:
    from manabot.backends.torch.policy import ActorCritic, PolicyInference
    from manabot.env.observation import ObservationSpace

__all__ = (
    "PUZZLE_DEFAULT_PLAYOUTS",
    "PUZZLE_MIN_MARGIN",
    "PUZZLE_MIN_MAX_PLIES",
    "PUZZLE_VALUE_MAX",
    "PUZZLE_VALUE_MIN",
    "ActionChooser",
    "PuzzleEnv",
    "PuzzleEvalFn",
    "PuzzleObservation",
    "PuzzleRecord",
    "PuzzleScoreRow",
    "PuzzleSource",
    "PuzzleVerificationResult",
    "append_puzzle",
    "build_puzzle_record",
    "engine_step_errors",
    "load_puzzles",
    "passes_value_prefilter",
    "puzzles_path",
    "score_puzzles",
    "summarize_paired_outcomes",
    "summarize_puzzle_scores",
    "verify_puzzle_candidate",
)

type PuzzleEvalFn = Callable[[PuzzleRecord], tuple[Sequence[float], float]]
type ActionChooser = Callable[[Sequence[PuzzleObservation], Sequence[random.Random]], Sequence[int]]

PUZZLE_DEFAULT_PLAYOUTS = 64
PUZZLE_MIN_MARGIN = 0.15
PUZZLE_MIN_MAX_PLIES = 240
PUZZLE_VALUE_MIN = 0.20
PUZZLE_VALUE_MAX = 0.80

_Z_95 = 1.959963984540054
_MIN_PAIRED_SAMPLES = 2


class _ObsAgent(Protocol):
    @property
    def player_index(self) -> int: ...


class _ObsActionSpace(Protocol):
    @property
    def actions(self) -> Sequence[object]: ...


class PuzzleObservation(Protocol):
    @property
    def agent(self) -> _ObsAgent: ...
    @property
    def action_space(self) -> _ObsActionSpace: ...


class PuzzleEnv(Protocol):
    def clone_env(self) -> PuzzleEnv: ...
    def determinize(self, seed: int, perspective: int, /) -> None: ...
    def step(self, action: int, /) -> tuple[PuzzleObservation, float, bool, bool, object]: ...
    def is_game_over(self) -> bool: ...
    def winner_index(self) -> int | None: ...


@dataclass(slots=True)
class _PlayoutBranch:
    pair_idx: int
    arm_idx: int
    env: PuzzleEnv
    obs: PuzzleObservation | None
    rng: random.Random
    done: bool = False
    errored: bool = False


def passes_value_prefilter(root_value_v: float) -> bool:
    return PUZZLE_VALUE_MIN <= root_value_v <= PUZZLE_VALUE_MAX


def _admission_reason(*, valid_pairs: int, margin: float, min_margin: float, ci_low: float) -> str | None:
    if valid_pairs < _MIN_PAIRED_SAMPLES:
        return f"insufficient valid paired playouts ({valid_pairs} < {_MIN_PAIRED_SAMPLES})"
    if margin < min_margin:
        return f"margin {margin:.4f} below threshold {min_margin:.4f}"
    if ci_low <= 0.0:
        return f"95% paired CI lower bound {ci_low:.4f} <= 0.0"
    return None


def summarize_paired_outcomes(
    pairs: Sequence[tuple[float, float]],
    *,
    best_action: int,
    alt_action: int,
    playouts_per_action: int,
    max_plies: int,
    min_margin: float = PUZZLE_MIN_MARGIN,
) -> PuzzleVerificationResult:
    n = len(pairs)
    if n == 0:
        best_wr, alt_wr, margin, paired_se, ci_low, ci_high = 0.5, 0.5, 0.0, 1.0, -1.0, 1.0
    else:
        best_wr = sum(pair[0] for pair in pairs) / n
        alt_wr = sum(pair[1] for pair in pairs) / n
        diffs = [pair[0] - pair[1] for pair in pairs]
        margin = sum(diffs) / n
        var = sum((d - margin) ** 2 for d in diffs) / (n - 1) if n >= _MIN_PAIRED_SAMPLES else 0.0
        paired_se = 1.0 if n < _MIN_PAIRED_SAMPLES else math.sqrt(max(0.0, var) / n)
        ci_low, ci_high = max(-1.0, margin - _Z_95 * paired_se), min(1.0, margin + _Z_95 * paired_se)
    admitted = n >= _MIN_PAIRED_SAMPLES and margin >= min_margin and ci_low > 0.0
    reason = None if admitted else _admission_reason(valid_pairs=n, margin=margin, min_margin=min_margin, ci_low=ci_low)
    return PuzzleVerificationResult(
        admitted,
        best_action,
        alt_action,
        round(best_wr, 4),
        round(alt_wr, 4),
        playouts_per_action,
        n,
        max_plies,
        round(margin, 4),
        round(paired_se, 4),
        round(ci_low, 4),
        round(ci_high, 4),
        reason,
    )


def engine_step_errors() -> tuple[type[BaseException], ...]:
    candidates = (*BaseException.__subclasses__(), *Exception.__subclasses__())
    engine_errs = tuple(cls for cls in candidates if cls.__name__ in ("PanicException", "PyAgentError"))
    return (*engine_errs, managym.ObservationCapacity, RuntimeError, ValueError, IndexError)


def _determinize_env(env: PuzzleEnv, *, deal_seed: int, acting_seat: int) -> None:
    step_errors = engine_step_errors()
    try:
        env.determinize(deal_seed, acting_seat)
    except step_errors:
        set_seed = getattr(env, "set_seed", None)
        if callable(set_seed):
            try:
                set_seed(deal_seed)
            except step_errors:
                return


def _spawn_paired_branches(
    env: PuzzleEnv, *, best_action: int, alt_action: int, playouts: int, acting_seat: int, seed: int
) -> list[_PlayoutBranch]:
    step_errors = engine_step_errors()
    branches: list[_PlayoutBranch] = []
    for pair_idx in range(playouts):
        deal_seed = seed * 1_000_003 + pair_idx * 9973
        for arm_idx, root_action in enumerate((best_action, alt_action)):
            cloned = env.clone_env()
            _determinize_env(cloned, deal_seed=deal_seed, acting_seat=acting_seat)
            errored = False
            try:
                next_obs, _, done, truncated, _ = cloned.step(root_action)
                is_done = bool(done or truncated or cloned.is_game_over())
            except step_errors:
                next_obs, is_done, errored = None, True, True
            branches.append(
                _PlayoutBranch(pair_idx, arm_idx, cloned, next_obs, random.Random(deal_seed), is_done, errored)
            )
    return branches


def _batch_policy_probs(
    scope: PolicyInference, obs_list: Sequence[managym.Observation], *, obs_space: ObservationSpace
) -> list[tuple[float, ...]]:
    step_errors = engine_step_errors()
    max_actions = obs_space.config.max_actions
    batch_rows: list[tuple[int, managym.Observation, int]] = []
    probs_out: list[tuple[float, ...]] = [()] * len(obs_list)
    for idx, obs in enumerate(obs_list):
        count = len(obs.action_space.actions)
        if count <= max_actions:
            batch_rows.append((idx, obs, count))
        else:
            try:
                probs_out[idx], _ = evaluate_observation(scope, obs_space, obs)
            except step_errors:
                probs_out[idx] = ()
    if batch_rows:
        encoded = managym.encode_observation_batch(
            [row[1] for row in batch_rows], obs_space.layout, obs_space.config.action_overflow
        )
        with torch.no_grad():
            logits, _ = scope.agent(scope.prepare(encoded))
            probs_t = torch.softmax(logits, dim=-1).cpu()
        for row_idx, (out_idx, _, count) in enumerate(batch_rows):
            probs_out[out_idx] = tuple(probs_t[row_idx, :count].tolist())
    return probs_out


def _make_agent_chooser(scope: PolicyInference, *, obs_space: ObservationSpace) -> ActionChooser:
    def _choose(obs_list: Sequence[PuzzleObservation], rngs: Sequence[random.Random]) -> list[int]:
        native = [obs for obs in obs_list if isinstance(obs, managym.Observation)]
        if len(native) != len(obs_list):
            msg = "playout_agent requires native managym.Observation instances"
            raise TypeError(msg)
        probs_batch = _batch_policy_probs(scope, native, obs_space=obs_space)
        return [
            rng.choices(range(len(probs)), weights=probs, k=1)[0] if probs else -1
            for probs, rng in zip(probs_batch, rngs, strict=True)
        ]

    return _choose


def _make_heuristic_chooser() -> ActionChooser:
    heuristic = managym.Heuristic()
    step_errors = engine_step_errors()

    def _choose(obs_list: Sequence[PuzzleObservation], _rngs: Sequence[random.Random]) -> list[int]:
        chosen: list[int] = []
        for obs in obs_list:
            if not isinstance(obs, managym.Observation):
                chosen.append(0)
                continue
            try:
                chosen.append(int(heuristic.select_action(obs)))
            except step_errors:
                chosen.append(0)
        return chosen

    return _choose


def _run_branches_to_limit(branches: Sequence[_PlayoutBranch], *, chooser: ActionChooser, max_plies: int) -> None:
    step_errors = engine_step_errors()
    for _ in range(max_plies):
        active = [b for b in branches if not b.done and b.obs is not None]
        if not active:
            break
        obs_list = [b.obs for b in active if b.obs is not None]
        for branch, action in zip(active, chooser(obs_list, [b.rng for b in active]), strict=True):
            if action < 0:
                branch.done, branch.errored = True, True
                continue
            try:
                next_obs, _, done, truncated, _ = branch.env.step(action)
                branch.obs, branch.done = next_obs, bool(done or truncated or branch.env.is_game_over())
            except step_errors:
                branch.obs, branch.done, branch.errored = None, True, True


def _validate_verify_args(
    obs: PuzzleObservation,
    *,
    best_action: int,
    alt_action: int,
    playout_agent: ActorCritic | None,
    obs_space: ObservationSpace | None,
    playouts: int,
    max_plies: int,
) -> None:
    num_actions = len(obs.action_space.actions)
    if best_action == alt_action or not (0 <= best_action < num_actions) or not (0 <= alt_action < num_actions):
        msg = f"best_action={best_action} and alt_action={alt_action} must be distinct indices in 0..{num_actions - 1}"
        raise ValueError(msg)
    if playouts <= 0:
        msg = f"playouts must be > 0, got {playouts}"
        raise ValueError(msg)
    if max_plies < PUZZLE_MIN_MAX_PLIES:
        msg = f"max_plies must be >= {PUZZLE_MIN_MAX_PLIES} to avoid draw truncation, got {max_plies}"
        raise ValueError(msg)
    if playout_agent is not None and obs_space is None:
        msg = "obs_space is required when playout_agent is provided"
        raise ValueError(msg)


def _branch_score(branch: _PlayoutBranch, *, hero_seat: int) -> float:
    winner = branch.env.winner_index() if branch.env.is_game_over() else None
    return 0.5 if winner is None else (1.0 if int(winner) == hero_seat else 0.0)


def verify_puzzle_candidate(
    env: PuzzleEnv,
    obs: PuzzleObservation,
    *,
    best_action: int,
    alt_action: int,
    playout_agent: ActorCritic | None = None,
    obs_space: ObservationSpace | None = None,
    playouts: int = PUZZLE_DEFAULT_PLAYOUTS,
    max_plies: int = PUZZLE_MIN_MAX_PLIES,
    min_margin: float = PUZZLE_MIN_MARGIN,
    seed: int = 42,
    hero_seat: int | None = None,
    action_chooser: ActionChooser | None = None,
) -> PuzzleVerificationResult:
    """Verify a candidate puzzle with paired CRN determinized rollouts.

    Spawns `playouts` paired clones sharing each pair's determinization and continuation
    RNG seed, steps `best_action` vs `alt_action` up to `max_plies` (`>= 240`), and admits
    the puzzle iff `margin >= min_margin` and `paired_ci_low > 0.0`.
    """
    _validate_verify_args(
        obs,
        best_action=best_action,
        alt_action=alt_action,
        playout_agent=playout_agent,
        obs_space=obs_space,
        playouts=playouts,
        max_plies=max_plies,
    )
    acting_seat = int(obs.agent.player_index)
    target_hero = acting_seat if hero_seat is None else hero_seat
    branches = _spawn_paired_branches(
        env, best_action=best_action, alt_action=alt_action, playouts=playouts, acting_seat=acting_seat, seed=seed
    )
    if action_chooser is not None:
        _run_branches_to_limit(branches, chooser=action_chooser, max_plies=max_plies)
    elif playout_agent is not None and obs_space is not None:
        with playout_agent.inference_scope() as scope:
            chooser = _make_agent_chooser(scope, obs_space=obs_space)
            _run_branches_to_limit(branches, chooser=chooser, max_plies=max_plies)
    else:
        _run_branches_to_limit(branches, chooser=_make_heuristic_chooser(), max_plies=max_plies)
    grouped = {(b.pair_idx, b.arm_idx): b for b in branches}
    valid_pairs = [
        (_branch_score(grouped[(i, 0)], hero_seat=target_hero), _branch_score(grouped[(i, 1)], hero_seat=target_hero))
        for i in range(playouts)
        if not grouped[(i, 0)].errored and not grouped[(i, 1)].errored
    ]
    return summarize_paired_outcomes(
        valid_pairs,
        best_action=best_action,
        alt_action=alt_action,
        playouts_per_action=playouts,
        max_plies=max_plies,
        min_margin=min_margin,
    )


def score_puzzles(puzzles: Sequence[PuzzleRecord], *, eval_fn: PuzzleEvalFn) -> list[PuzzleScoreRow]:
    rows: list[PuzzleScoreRow] = []
    for puzzle in puzzles:
        probs, value_v = eval_fn(puzzle)
        chosen_idx = max(range(len(probs)), key=probs.__getitem__) if probs else -1
        best_prob = float(probs[puzzle.best_action_index]) if 0 <= puzzle.best_action_index < len(probs) else 0.0
        alt_prob = float(probs[puzzle.alt_action_index]) if 0 <= puzzle.alt_action_index < len(probs) else 0.0
        rows.append(
            PuzzleScoreRow(
                puzzle.puzzle_id,
                puzzle.format_id,
                puzzle.source,
                chosen_idx == puzzle.best_action_index,
                chosen_idx,
                puzzle.best_action_index,
                round(best_prob, 4),
                round(alt_prob, 4),
                round(float(value_v), 4),
                puzzle.margin,
            )
        )
    return rows


def summarize_puzzle_scores(rows: Sequence[PuzzleScoreRow]) -> dict[str, object]:
    total = len(rows)
    if total == 0:
        return {"total": 0, "solved": 0, "solve_rate": 0.0, "mean_best_prob": 0.0, "by_source": {}}
    solved = sum(1 for row in rows if row.solved)
    by_source = {
        src: {"total": len(sub), "solved": (s := sum(1 for r in sub if r.solved)), "solve_rate": round(s / len(sub), 4)}
        for src in sorted({row.source for row in rows})
        for sub in ([row for row in rows if row.source == src],)
    }
    return {
        "total": total,
        "solved": solved,
        "solve_rate": round(solved / total, 4),
        "mean_best_prob": round(sum(row.best_action_prob for row in rows) / total, 4),
        "by_source": by_source,
    }

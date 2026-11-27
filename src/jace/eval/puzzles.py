"""Format-agnostic tactical puzzle mining, CRN playout verification, and scoring.

Mines decision states without hardcoding card names or set codes so the same pipeline
applies across Limited and Constructed formats. Core records, locked JSONL storage,
and paired CRN verification live in `jace.eval._puzzles_core` and are re-exported here
alongside replay mining and the `mine` / `score` CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import managym
from manabot.backends.torch.bot import evaluate_observation
from manabot.backends.torch.checkpoint import load_checkpoint

from jace.eval._puzzles_core import (
    PUZZLE_DEFAULT_PLAYOUTS,
    PUZZLE_MIN_MARGIN,
    PUZZLE_MIN_MAX_PLIES,
    PUZZLE_VALUE_MAX,
    PUZZLE_VALUE_MIN,
    ActionChooser,
    PuzzleEnv,
    PuzzleEvalFn,
    PuzzleObservation,
    PuzzleRecord,
    PuzzleScoreRow,
    PuzzleSource,
    PuzzleVerificationResult,
    append_puzzle,
    build_puzzle_record,
    engine_step_errors,
    load_puzzles,
    passes_value_prefilter,
    puzzles_path,
    score_puzzles,
    summarize_paired_outcomes,
    summarize_puzzle_scores,
    verify_puzzle_candidate,
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
    "load_puzzles",
    "main",
    "passes_value_prefilter",
    "puzzles_path",
    "score_puzzles",
    "summarize_paired_outcomes",
    "summarize_puzzle_scores",
    "verify_puzzle_candidate",
)

_POLICY_DISAGREEMENT_TV = 0.75
_SWING_LOSS_DELTA_V = 0.30
_MIN_DECISION_ACTIONS = 2


@dataclass(frozen=True, slots=True)
class _RecordedReplay:
    run_id: str
    iteration: int
    game_idx: int
    seed: int
    hero_seat: int
    players: tuple[managym.PlayerConfig, ...]
    actions: tuple[int, ...]
    winner: int | None
    skip_trivial: bool
    decision_memory: bool


def _resolve_eval_jsonl(runs_dir: Path, *, run_id: str, iteration: int) -> Path:
    filename = f"eval-iter-{iteration:04d}.jsonl"
    for candidate in (runs_dir / run_id / "games" / filename, *sorted(runs_dir.glob(f"*/{run_id}/games/{filename}"))):
        if candidate.is_file():
            return candidate
    msg = f"eval games file not found for run_id={run_id!r} iteration={iteration} under {runs_dir}"
    raise FileNotFoundError(msg)


def _hero_seat_from_context(context: object) -> int:
    if isinstance(context, Mapping) and isinstance(players := context.get("players"), list):
        for idx, role in enumerate(players[:2]):
            if str(role).lower() == "agent":
                return idx
    return 0


def _player_configs_from_raw(raw_players: object) -> tuple[managym.PlayerConfig, ...]:
    if not isinstance(raw_players, list):
        return ()
    configs: list[managym.PlayerConfig] = []
    for entry in raw_players:
        if not isinstance(entry, Mapping):
            continue
        deck = {str(k): int(v) for k, v in d.items()} if isinstance(d := entry.get("decklist"), Mapping) else {}
        sb = {str(k): int(v) for k, v in s.items()} if isinstance(s := entry.get("sideboard"), Mapping) else None
        world, disc = entry.get("world"), entry.get("disclosure")
        configs.append(
            managym.PlayerConfig(
                name=str(entry.get("name") or f"player_{len(configs)}"),
                decklist=deck,
                sideboard=sb or None,
                world=world if isinstance(world, str) else None,
                disclosure=disc if disc in ("open", "closed") else None,
            )
        )
    return tuple(configs)


def _load_recorded_replays(jsonl_path: Path, *, run_id: str, iteration: int) -> list[_RecordedReplay]:
    replays: list[_RecordedReplay] = []
    for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not (line := raw_line.strip()) or not isinstance(payload := json.loads(line), Mapping):
            continue
        items = games if isinstance(games := payload.get("games"), list) else [payload]
        top_seat = _hero_seat_from_context(payload.get("context"))
        skip_triv, dec_mem = bool(payload.get("skip_trivial", True)), bool(payload.get("decision_memory", False))
        for item in items:
            if not isinstance(item, Mapping) or not isinstance(acts := item.get("actions"), list):
                continue
            end = item.get("end")
            winner = end.get("winner") if isinstance(end, Mapping) and isinstance(end.get("winner"), int) else None
            raw_seed = item.get("seed", 0)
            seat = _hero_seat_from_context(item.get("context")) if "context" in item else top_seat
            actions = tuple(a for a in acts if isinstance(a, int) and not isinstance(a, bool))
            replays.append(
                _RecordedReplay(
                    run_id,
                    iteration,
                    len(replays),
                    int(raw_seed) if isinstance(raw_seed, (int, float)) else 0,
                    seat,
                    _player_configs_from_raw(item.get("players")),
                    actions,
                    winner,
                    skip_triv,
                    dec_mem,
                )
            )
    return replays


def _hero_win_prob(obs: managym.Observation, *, raw_val: float, hero_seat: int) -> float:
    signed = raw_val if int(obs.agent.player_index) == hero_seat else -raw_val
    return float(min(1.0, max(0.0, 0.5 * (signed + 1.0))))


@dataclass(frozen=True, slots=True)
class _MineContext:
    agent_a: ActorCritic
    scope_a: PolicyInference
    obs_space_a: ObservationSpace
    scope_b: PolicyInference | None
    obs_space_b: ObservationSpace | None
    sources: frozenset[str]
    ns: argparse.Namespace


@dataclass(slots=True)
class _PrevHeroStep:
    env: managym.Env
    obs: managym.Observation
    step_idx: int
    probs_a: tuple[float, ...]
    root_v: float
    taken: int


def _detect_policy_candidate(
    obs: managym.Observation,
    *,
    ctx: _MineContext,
    probs_a: Sequence[float],
    root_v: float,
) -> tuple[PuzzleSource, int, int, float] | None:
    if not passes_value_prefilter(root_v) or len(probs_a) < _MIN_DECISION_ACTIONS:
        return None
    if "policy" not in ctx.sources or ctx.scope_b is None or ctx.obs_space_b is None:
        return None
    ranked = sorted(range(len(probs_a)), key=probs_a.__getitem__, reverse=True)
    probs_b, _ = evaluate_observation(ctx.scope_b, ctx.obs_space_b, obs)
    tv = 0.5 * sum(abs(p - q) for p, q in zip(probs_a, probs_b, strict=False))
    top_b = max(range(len(probs_b)), key=probs_b.__getitem__)
    if tv > _POLICY_DISAGREEMENT_TV and top_b != ranked[0]:
        return ("policy_disagreement", ranked[0], top_b, root_v)
    return None


def _detect_swing_candidate(
    prev: _PrevHeroStep | None, *, ctx: _MineContext, root_v: float, lost_game: bool
) -> tuple[PuzzleSource, int, int, float] | None:
    if prev is None or "swing" not in ctx.sources or not lost_game:
        return None
    if not passes_value_prefilter(prev.root_v) or len(prev.probs_a) < _MIN_DECISION_ACTIONS:
        return None
    if abs(root_v - prev.root_v) <= _SWING_LOSS_DELTA_V:
        return None
    ranked = sorted(range(len(prev.probs_a)), key=prev.probs_a.__getitem__, reverse=True)
    alt = prev.taken if prev.taken != ranked[0] and 0 <= prev.taken < len(prev.probs_a) else ranked[1]
    return ("high_swing_loss", ranked[0], alt, prev.root_v)


def _step_replay_safely(env: managym.Env, action: int) -> managym.Observation | None:
    try:
        next_obs, _, done, truncated, _ = env.step(action)
    except engine_step_errors():
        return None
    return None if (done or truncated) else next_obs


def _try_admit_hit(
    env: managym.Env,
    obs: managym.Observation,
    *,
    ctx: _MineContext,
    replay: _RecordedReplay,
    step_idx: int,
    hit: tuple[PuzzleSource, int, int, float],
    seed: int,
) -> int:
    source, best_act, alt_act, root_v = hit
    ns, acting_seat = ctx.ns, int(obs.agent.player_index)
    ver = verify_puzzle_candidate(
        env,
        obs,
        best_action=best_act,
        alt_action=alt_act,
        playout_agent=ctx.agent_a,
        obs_space=ctx.obs_space_a,
        playouts=ns.playouts,
        max_plies=ns.max_plies,
        min_margin=ns.min_margin,
        seed=seed,
        hero_seat=acting_seat,
    )
    if not ver.admitted:
        return 0
    key = f"{ns.format_id}:{replay.run_id}:{replay.iteration}:{replay.game_idx}:{step_idx}:{source}"
    rec = build_puzzle_record(
        obs,
        ver,
        puzzle_id=f"puz-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}",
        format_id=ns.format_id,
        source=source,
        run_id=replay.run_id,
        iteration=replay.iteration,
        game_idx=replay.game_idx,
        step=step_idx,
        hero_seat=acting_seat,
        seed=replay.seed,
        root_value_v=root_v,
    )
    append_puzzle(ns.runs_dir, rec)
    return 1


def _process_hero_step(
    env: managym.Env,
    obs: managym.Observation,
    *,
    step_idx: int,
    taken: int,
    prev_hero: _PrevHeroStep | None,
    ctx: _MineContext,
    replay: _RecordedReplay,
    checked_offset: int,
    budget_left: int,
    lost: bool,
) -> tuple[int, int, _PrevHeroStep | None]:
    probs_a, raw_a = evaluate_observation(ctx.scope_a, ctx.obs_space_a, obs)
    root_v = _hero_win_prob(obs, raw_val=float(raw_a), hero_seat=replay.hero_seat)
    checked, admitted = 0, 0
    swing_hit = _detect_swing_candidate(prev_hero, ctx=ctx, root_v=root_v, lost_game=lost)
    if swing_hit is not None and prev_hero is not None:
        checked += 1
        admitted += _try_admit_hit(
            prev_hero.env,
            prev_hero.obs,
            ctx=ctx,
            replay=replay,
            step_idx=prev_hero.step_idx,
            hit=swing_hit,
            seed=ctx.ns.seed + checked_offset + checked,
        )
    if admitted < budget_left:
        policy_hit = _detect_policy_candidate(obs, ctx=ctx, probs_a=probs_a, root_v=root_v)
        if policy_hit is not None:
            checked += 1
            admitted += _try_admit_hit(
                env,
                obs,
                ctx=ctx,
                replay=replay,
                step_idx=step_idx,
                hit=policy_hit,
                seed=ctx.ns.seed + checked_offset + checked,
            )
    next_prev = (
        _PrevHeroStep(env.clone_env(), obs, step_idx, tuple(float(p) for p in probs_a), root_v, taken)
        if ("swing" in ctx.sources and lost)
        else None
    )
    return checked, admitted, next_prev


def _mine_and_store_game(
    replay: _RecordedReplay, *, ctx: _MineContext, checked_start: int, remaining_budget: int
) -> tuple[int, int]:
    try:
        env = managym.Env(seed=replay.seed, skip_trivial=replay.skip_trivial, decision_memory=replay.decision_memory)
        obs: managym.Observation | None
        obs, _ = env.reset(list(replay.players))
    except engine_step_errors():
        return 0, 0
    checked, admitted, prev_hero = 0, 0, None
    lost = replay.winner is not None and replay.winner != replay.hero_seat
    for step_idx, taken in enumerate(replay.actions):
        if obs is None or admitted >= remaining_budget:
            break
        if int(obs.agent.player_index) == replay.hero_seat and len(obs.action_space.actions) >= _MIN_DECISION_ACTIONS:
            c_inc, a_inc, prev_hero = _process_hero_step(
                env,
                obs,
                step_idx=step_idx,
                taken=taken,
                prev_hero=prev_hero,
                ctx=ctx,
                replay=replay,
                checked_offset=checked_start + checked,
                budget_left=remaining_budget - admitted,
                lost=lost,
            )
            checked, admitted = checked + c_inc, admitted + a_inc
        obs = _step_replay_safely(env, taken)
    return checked, admitted


def _run_mine_cli(ns: argparse.Namespace) -> int:
    sources = frozenset(part.strip().lower() for part in str(ns.sources).split(",") if part.strip())
    agent_a, obs_space_a = load_checkpoint(str(ns.checkpoint))
    agent_b, obs_space_b = load_checkpoint(str(ns.alt_checkpoint)) if ns.alt_checkpoint else (None, None)
    files = sorted(ns.runs_dir.glob("*/*/games/eval-iter-*.jsonl"))
    if ns.run_id:
        files = [
            p
            for p in files
            if p.parents[1].name == ns.run_id or p.parents[1].relative_to(ns.runs_dir).as_posix() == ns.run_id
        ]
    checked_total, admitted_total = 0, 0
    with ExitStack() as stack:
        scope_a = stack.enter_context(agent_a.inference_scope())
        scope_b = stack.enter_context(agent_b.inference_scope()) if agent_b is not None else None
        ctx = _MineContext(agent_a, scope_a, obs_space_a, scope_b, obs_space_b, sources, ns)
        for jsonl_path in files:
            if admitted_total >= ns.max_puzzles:
                break
            run_id = jsonl_path.parents[1].relative_to(ns.runs_dir).as_posix()
            iteration = int(jsonl_path.stem.removeprefix("eval-iter-"))
            for replay in _load_recorded_replays(jsonl_path, run_id=run_id, iteration=iteration)[: ns.max_games]:
                checked, admitted = _mine_and_store_game(
                    replay, ctx=ctx, checked_start=checked_total, remaining_budget=ns.max_puzzles - admitted_total
                )
                checked_total += checked
                admitted_total += admitted
    out_path = str(puzzles_path(ns.runs_dir, ns.format_id))
    sys.stdout.write(
        json.dumps({"checked": checked_total, "admitted": admitted_total, "path": out_path}, sort_keys=True) + "\n"
    )
    return 0


def _run_score_cli(ns: argparse.Namespace) -> int:
    puzzles = load_puzzles(ns.runs_dir, ns.format_id)
    agent, obs_space = load_checkpoint(str(ns.checkpoint))
    cache: dict[tuple[str, int], list[_RecordedReplay]] = {}
    with agent.inference_scope() as scope:

        def _eval_puzzle(puzzle: PuzzleRecord) -> tuple[Sequence[float], float]:
            key = (puzzle.run_id, puzzle.iteration)
            if key not in cache:
                jsonl = _resolve_eval_jsonl(ns.runs_dir, run_id=puzzle.run_id, iteration=puzzle.iteration)
                cache[key] = _load_recorded_replays(jsonl, run_id=puzzle.run_id, iteration=puzzle.iteration)
            replay = cache[key][puzzle.game_idx]
            env = managym.Env(
                seed=replay.seed, skip_trivial=replay.skip_trivial, decision_memory=replay.decision_memory
            )
            obs, _ = env.reset(list(replay.players))
            for action in replay.actions[: puzzle.step]:
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            probs, raw_val = evaluate_observation(scope, obs_space, obs)
            return probs, _hero_win_prob(obs, raw_val=float(raw_val), hero_seat=puzzle.hero_seat)

        rows = score_puzzles(puzzles, eval_fn=_eval_puzzle)

    payload = {
        "format_id": ns.format_id,
        "checkpoint": str(ns.checkpoint),
        "summary": summarize_puzzle_scores(rows),
        "rows": [asdict(row) for row in rows],
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def _add_common_cli_args(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("--runs-dir", type=Path, required=True)
    cmd.add_argument("--format", dest="format_id", required=True)
    cmd.add_argument("--checkpoint", type=Path, required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine, verify, and score format-agnostic tactical puzzles.")
    sub = parser.add_subparsers(dest="command", required=True)
    mine_cmd = sub.add_parser("mine", help="Mine and verify puzzles from recorded eval games.")
    _add_common_cli_args(mine_cmd)
    mine_cmd.add_argument("--alt-checkpoint", type=Path, default=None)
    mine_cmd.add_argument("--run-id", default=None)
    mine_cmd.add_argument("--sources", default="policy,swing")
    mine_cmd.add_argument("--playouts", type=int, default=PUZZLE_DEFAULT_PLAYOUTS)
    mine_cmd.add_argument("--max-plies", type=int, default=PUZZLE_MIN_MAX_PLIES)
    mine_cmd.add_argument("--min-margin", type=float, default=PUZZLE_MIN_MARGIN)
    mine_cmd.add_argument("--max-games", type=int, default=16)
    mine_cmd.add_argument("--max-puzzles", type=int, default=20)
    mine_cmd.add_argument("--seed", type=int, default=42)
    _add_common_cli_args(sub.add_parser("score", help="Score verified puzzles against a checkpoint."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    return _run_mine_cli(ns) if ns.command == "mine" else _run_score_cli(ns)


if __name__ == "__main__":
    raise SystemExit(main())

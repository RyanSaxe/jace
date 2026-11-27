"""Native managym game replay and observation sampling for Jace Studio.

Reads recorded games from `games/eval-iter-*.jsonl`, preserves full `PlayerConfig`
setup (`disclosure`, `sideboard`, `world`, `decision_memory`) and mirrored
paired-deal `hero_seat`, replays deterministically through `managym.Env`, samples
game trajectories for mistake indexing and checkpoint comparison, and re-exports
board serialization helpers from `jace.studio.board`.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import managym
from managym.enums import ActionEnum, ActionSpaceEnum

from jace.studio.board import (
    PovMode,
    format_action_for_event,
    heuristic_win_estimate,
    non_permanent_card_names,
    seat_aligned_life,
    seat_aligned_obs,
    serialize_legal_actions,
    serialize_observation,
)
from jace.studio.runs import resolve_run_dir

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

type WinScorer = Callable[[list[managym.Observation], RecordedGame], tuple[list[float], str]]

__all__ = (
    "PlayerSpec",
    "PovMode",
    "RecordedGame",
    "ReplayService",
    "engine_step_errors",
    "load_recorded_game",
    "load_recorded_games",
    "non_permanent_card_names",
    "sample_game_observations",
    "seat_aligned_life",
    "serialize_legal_actions",
    "serialize_observation",
)

type DisclosureMode = Literal["open", "closed"]

_MAX_CACHED_GAMES = 16
_MAX_RECENT_EVENTS = 6


@dataclass(frozen=True, slots=True)
class PlayerSpec:
    name: str
    decklist: dict[str, int]
    sideboard: dict[str, int]
    disclosure: DisclosureMode | None = None
    world: str | None = None


@dataclass(frozen=True, slots=True)
class RecordedGame:
    run_id: str
    iteration: int
    game_idx: int
    seed: int
    hero_seat: int
    players: tuple[PlayerSpec, ...]
    actions: tuple[int, ...]
    winner: int | None
    end_turn: int
    skip_trivial: bool
    decision_memory: bool = False
    decision_memory_include_own: bool = False


@dataclass(frozen=True, slots=True)
class _RecordedPayload:
    game: dict[str, object]
    hero_seat: int
    skip_trivial: bool
    decision_memory: bool
    decision_memory_include_own: bool


def _resolve_hero_seat(context: object) -> int:
    if not isinstance(context, dict):
        return 0
    players = context.get("players")
    if isinstance(players, list):
        for idx, role in enumerate(players[:2]):
            if str(role).lower() == "agent":
                return idx
    return 0


def _parse_counts_map(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _parse_disclosure(raw: object) -> DisclosureMode | None:
    if raw == "open":
        return "open"
    if raw == "closed":
        return "closed"
    return None


def _extract_player_specs(raw_players: object) -> tuple[PlayerSpec, ...]:
    if not isinstance(raw_players, list):
        return ()
    specs: list[PlayerSpec] = []
    for entry in raw_players:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or f"player_{len(specs)}")
        world_raw = entry.get("world")
        specs.append(
            PlayerSpec(
                name=name,
                decklist=_parse_counts_map(entry.get("decklist")),
                sideboard=_parse_counts_map(entry.get("sideboard")),
                disclosure=_parse_disclosure(entry.get("disclosure")),
                world=world_raw if isinstance(world_raw, str) else None,
            )
        )
    return tuple(specs)


def _payloads_from_line(raw_line: str) -> list[_RecordedPayload]:
    line = raw_line.strip()
    if not line:
        return []
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    top_seat = _resolve_hero_seat(payload.get("context"))
    skip_trivial = bool(payload.get("skip_trivial", True))
    dec_mem = bool(payload.get("decision_memory", False))
    dec_own = bool(payload.get("decision_memory_include_own", False))
    games = payload.get("games")
    items = games if isinstance(games, list) else [payload]
    return [
        _RecordedPayload(
            item,
            _resolve_hero_seat(item.get("context")) if "context" in item else top_seat,
            skip_trivial,
            dec_mem,
            dec_own,
        )
        for item in items
        if isinstance(item, dict) and "actions" in item
    ]


def _iter_game_payloads(jsonl_path: Path) -> list[_RecordedPayload]:
    found: list[_RecordedPayload] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            found.extend(_payloads_from_line(raw_line))
    return found


def _parse_actions(raw_actions: object) -> tuple[int, ...]:
    if not isinstance(raw_actions, list):
        return ()
    return tuple(item for item in raw_actions if isinstance(item, int) and not isinstance(item, bool))


def _recorded_game_from_payload(
    rec: _RecordedPayload,
    *,
    run_id: str,
    iteration: int,
    game_idx: int,
) -> RecordedGame:
    raw_game = rec.game
    end = raw_game.get("end")
    winner = end.get("winner") if isinstance(end, dict) and isinstance(end.get("winner"), int) else None
    end_turn = int(end.get("turn") or 0) if isinstance(end, dict) and isinstance(end.get("turn"), (int, float)) else 0
    raw_seed = raw_game.get("seed")
    seed = int(raw_seed) if isinstance(raw_seed, (int, float)) else 0
    return RecordedGame(
        run_id=run_id,
        iteration=iteration,
        game_idx=game_idx,
        seed=seed,
        hero_seat=rec.hero_seat,
        players=_extract_player_specs(raw_game.get("players")),
        actions=_parse_actions(raw_game.get("actions")),
        winner=winner,
        end_turn=end_turn,
        skip_trivial=rec.skip_trivial,
        decision_memory=rec.decision_memory,
        decision_memory_include_own=rec.decision_memory_include_own,
    )


def load_recorded_games(jsonl_path: Path, *, run_id: str, iteration: int) -> list[RecordedGame]:
    entries = _iter_game_payloads(jsonl_path)
    return [
        _recorded_game_from_payload(rec, run_id=run_id, iteration=iteration, game_idx=idx)
        for idx, rec in enumerate(entries)
    ]


def load_recorded_game(runs_dir: Path, *, run_id: str, iteration: int, game_idx: int) -> RecordedGame:
    run_dir = resolve_run_dir(runs_dir, run_id)
    jsonl_path = run_dir / "games" / f"eval-iter-{iteration:04d}.jsonl"
    if not jsonl_path.is_file():
        msg = f"eval game file not found for {run_id} iter {iteration}"
        raise FileNotFoundError(msg)
    entries = _iter_game_payloads(jsonl_path)
    if game_idx < 0 or game_idx >= len(entries):
        msg = f"game_idx {game_idx} out of range (0..{max(0, len(entries) - 1)})"
        raise IndexError(msg)
    return _recorded_game_from_payload(entries[game_idx], run_id=run_id, iteration=iteration, game_idx=game_idx)


def engine_step_errors() -> tuple[type[BaseException], ...]:
    candidates = (*BaseException.__subclasses__(), *Exception.__subclasses__())
    engine_errs = tuple(cls for cls in candidates if cls.__name__ in ("PanicException", "PyAgentError"))
    return (*engine_errs, RuntimeError, ValueError, IndexError)


def _new_env_for_game(game: RecordedGame) -> tuple[managym.Env, managym.Observation]:
    configs = [
        managym.PlayerConfig(
            name=spec.name,
            decklist=dict(spec.decklist),
            sideboard=dict(spec.sideboard) if spec.sideboard else None,
            world=spec.world,
            disclosure=spec.disclosure,
        )
        for spec in game.players
    ]
    env = managym.Env(
        seed=game.seed,
        skip_trivial=game.skip_trivial,
        decision_memory=game.decision_memory,
        decision_memory_include_own=game.decision_memory_include_own,
    )
    obs, _ = env.reset(configs)
    return env, obs


def _collect_sampled_observations(
    env: managym.Env,
    obs: managym.Observation,
    *,
    actions: tuple[int, ...],
    stride: int,
    step_errors: tuple[type[BaseException], ...],
) -> tuple[bool, list[tuple[int, managym.Observation]]]:
    sampled: list[tuple[int, managym.Observation]] = []
    last_idx = len(actions) - 1
    for step_idx, action in enumerate(actions):
        if step_idx % stride == 0 or step_idx == last_idx:
            sampled.append((step_idx, obs))
        try:
            obs, _, done, truncated, _ = env.step(action)
        except step_errors:
            return False, sampled
        if done or truncated:
            return step_idx >= last_idx, sampled
    return True, sampled


def sample_game_observations(
    game: RecordedGame,
    *,
    stride: int = 4,
) -> tuple[bool, list[tuple[int, managym.Observation]]]:
    if not game.actions:
        return False, []
    step_errors = engine_step_errors()
    try:
        env, obs = _new_env_for_game(game)
    except step_errors:
        return False, []
    return _collect_sampled_observations(env, obs, actions=game.actions, stride=max(1, stride), step_errors=step_errors)


def _event_for_step(
    obs: managym.Observation,
    *,
    step_idx: int,
    action_idx: int,
    hero_seat: int,
) -> dict[str, object] | None:
    if action_idx < 0 or action_idx >= len(obs.action_space.actions):
        return None
    action = obs.action_space.actions[action_idx]
    act_type = int(action.action_type)
    if act_type == int(ActionEnum.PRIORITY_PASS_PRIORITY):
        return None
    label = format_action_for_event(obs, action)
    actor_side = "hero" if int(obs.agent.player_index) == hero_seat else "opp"
    act_name = ActionEnum(act_type).name
    private_hand = "DISCARD" in act_name or "MULLIGAN" in act_name
    return {
        "step": step_idx,
        "turn": int(obs.turn.turn_number),
        "actor_side": actor_side,
        "text": label,
        "private_hand": private_hand,
    }


def _apply_win_scorer(
    turns_map: dict[int, dict[str, object]],
    *,
    turn_obs: dict[int, managym.Observation],
    sorted_turns: list[int],
    game: RecordedGame,
    win_scorer: WinScorer | None,
) -> str:
    if win_scorer is None or not sorted_turns:
        return "heuristic"
    probs, src = win_scorer([turn_obs[t] for t in sorted_turns], game)
    if len(probs) != len(sorted_turns):
        return "heuristic"
    for turn_num, val in zip(sorted_turns, probs, strict=True):
        turns_map[turn_num]["eval_win"] = float(val)
    return src


def _row_float(row: dict[str, object], key: str) -> float:
    raw = row.get(key)
    return float(raw) if isinstance(raw, (int, float)) else 0.5


def _row_int(row: dict[str, object], key: str) -> int:
    raw = row.get(key)
    return int(raw) if isinstance(raw, (int, float)) else 0


def _resolve_turning_step(timeline: list[dict[str, object]], *, game: RecordedGame, fallback: int) -> int:
    if not timeline:
        return fallback
    if game.winner is not None and game.winner != game.hero_seat:
        best = max(timeline, key=lambda row: _row_float(row, "eval_win"))
        return _row_int(best, "step")
    if game.winner == game.hero_seat:
        worst = min(timeline, key=lambda row: _row_float(row, "eval_win"))
        return _row_int(worst, "step")
    return fallback


def _append_recent_event(events: list[dict[str, object]], event: dict[str, object] | None) -> None:
    if event is None:
        return
    events.append(event)
    if len(events) > _MAX_RECENT_EVENTS:
        del events[0]


def _turn_snapshot(
    obs: managym.Observation,
    *,
    step_idx: int,
    turn_num: int,
    hero_seat: int,
) -> tuple[dict[str, object], int]:
    hero_pack, opp_pack = seat_aligned_obs(obs, hero_seat=hero_seat)
    crowd = len(hero_pack[1]) + len(opp_pack[1]) + 2 * len(obs.stack_objects)
    row: dict[str, object] = {
        "turn": turn_num,
        "step": step_idx,
        "hero_life": int(hero_pack[0].life),
        "opp_life": int(opp_pack[0].life),
        "hero_perms": len(hero_pack[1]),
        "opp_perms": len(opp_pack[1]),
        "stack_len": len(obs.stack_objects),
        "eval_win": heuristic_win_estimate(obs, hero_seat=hero_seat),
    }
    return row, crowd


def _find_next_priority_step(
    env: managym.Env,
    game: RecordedGame,
    *,
    start_step: int,
    priority_code: int,
    step_errors: tuple[type[BaseException], ...],
) -> int | None:
    if start_step >= len(game.actions):
        return None
    probe = env.clone_env()
    for idx in range(start_step, len(game.actions)):
        try:
            next_obs, _, done, truncated, _ = probe.step(game.actions[idx])
        except step_errors:
            return None
        if done or truncated:
            return None
        if int(next_obs.action_space.action_space_type) == priority_code:
            return idx + 1
    return None


def _pick_nearest_priority(*, target: int, prev_step: int | None, next_step: int | None) -> int | None:
    if prev_step is None:
        return next_step
    if next_step is None:
        return prev_step
    return next_step if (next_step - target) < (target - prev_step) else prev_step


def _resolve_nearest_priority(
    env: managym.Env,
    obs: managym.Observation,
    *,
    game: RecordedGame,
    target: int,
    last_priority_step: int | None,
    step_errors: tuple[type[BaseException], ...],
) -> int | None:
    priority_code = int(ActionSpaceEnum.PRIORITY)
    if int(obs.action_space.action_space_type) == priority_code:
        return target
    next_priority = _find_next_priority_step(
        env, game, start_step=target, priority_code=priority_code, step_errors=step_errors
    )
    return _pick_nearest_priority(target=target, prev_step=last_priority_step, next_step=next_priority)


class ReplayService:
    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir.resolve()
        self._timeline_cache: OrderedDict[tuple[str, int, int], dict[str, object]] = OrderedDict()
        self._lock = threading.Lock()

    def clear_cache(self) -> None:
        with self._lock:
            self._timeline_cache.clear()

    def _put_cache(self, key: tuple[str, int, int], value: dict[str, object]) -> None:
        with self._lock:
            self._timeline_cache[key] = value
            self._timeline_cache.move_to_end(key)
            while len(self._timeline_cache) > _MAX_CACHED_GAMES:
                self._timeline_cache.popitem(last=False)

    def get_game_overview(
        self,
        *,
        run_id: str,
        iteration: int,
        game_idx: int,
        win_scorer: WinScorer | None = None,
    ) -> dict[str, object]:
        key = (run_id, iteration, game_idx)
        with self._lock:
            cached = self._timeline_cache.get(key)
            if cached is not None:
                self._timeline_cache.move_to_end(key)
                return cached
        game = load_recorded_game(self._runs_dir, run_id=run_id, iteration=iteration, game_idx=game_idx)
        overview = self._build_game_overview(game, win_scorer=win_scorer)
        self._put_cache(key, overview)
        return overview

    def _build_game_overview(self, game: RecordedGame, *, win_scorer: WinScorer | None = None) -> dict[str, object]:
        step_errors = engine_step_errors()
        try:
            env, obs = _new_env_for_game(game)
        except step_errors as exc:
            return {
                "run_id": game.run_id,
                "iteration": game.iteration,
                "game_idx": game.game_idx,
                "seed": game.seed,
                "hero_seat": game.hero_seat,
                "winner": game.winner,
                "total_steps": len(game.actions),
                "max_playable_step": 0,
                "peak_step": 0,
                "turning_point_step": 0,
                "eval_source": "none",
                "replay_warning": f"Replay reset failed (older engine version): {exc}",
                "players": [{"name": spec.name, "deck_size": sum(spec.decklist.values())} for spec in game.players],
                "timeline": [],
            }
        turns_map: dict[int, dict[str, object]] = {}
        turn_obs: dict[int, managym.Observation] = {}
        peak_step, peak_score, last_valid_step = 0, -1, 0
        warning: str | None = None
        for step_idx, action in enumerate(game.actions):
            last_valid_step = step_idx
            turn_num = int(obs.turn.turn_number)
            row, crowd = _turn_snapshot(obs, step_idx=step_idx, turn_num=turn_num, hero_seat=game.hero_seat)
            if crowd >= peak_score:
                peak_score, peak_step = crowd, step_idx
            turns_map[turn_num] = row
            turn_obs[turn_num] = obs
            try:
                obs, _, done, truncated, _ = env.step(action)
            except step_errors as exc:
                warning = f"Replay stopped at step {step_idx} (recorded on an older engine version): {exc}"
                break
            if done or truncated:
                break
        sorted_turns = sorted(turns_map)
        eval_source = _apply_win_scorer(
            turns_map, turn_obs=turn_obs, sorted_turns=sorted_turns, game=game, win_scorer=win_scorer
        )
        timeline = [turns_map[t] for t in sorted_turns]
        return {
            "run_id": game.run_id,
            "iteration": game.iteration,
            "game_idx": game.game_idx,
            "seed": game.seed,
            "hero_seat": game.hero_seat,
            "winner": game.winner,
            "total_steps": len(game.actions),
            "max_playable_step": last_valid_step,
            "peak_step": peak_step,
            "turning_point_step": _resolve_turning_step(timeline, game=game, fallback=peak_step),
            "eval_source": eval_source,
            "replay_warning": warning,
            "players": [{"name": spec.name, "deck_size": sum(spec.decklist.values())} for spec in game.players],
            "timeline": timeline,
        }

    def replay_to_step(
        self,
        *,
        run_id: str,
        iteration: int,
        game_idx: int,
        step: int,
    ) -> tuple[
        RecordedGame,
        managym.Env,
        managym.Observation,
        int,
        int | None,
        str | None,
        list[dict[str, object]],
        int | None,
    ]:
        game = load_recorded_game(self._runs_dir, run_id=run_id, iteration=iteration, game_idx=game_idx)
        step_errors = engine_step_errors()
        try:
            env, obs = _new_env_for_game(game)
        except step_errors as exc:
            msg = f"Replay reset failed for {run_id} iter {iteration} game {game_idx}: {exc}"
            raise ValueError(msg) from exc
        target = max(0, min(step, max(0, len(game.actions) - 1)))
        events: list[dict[str, object]] = []
        last_priority_step: int | None = None
        priority_code = int(ActionSpaceEnum.PRIORITY)
        for idx in range(target):
            if int(obs.action_space.action_space_type) == priority_code:
                last_priority_step = idx
            ev = _event_for_step(obs, step_idx=idx, action_idx=game.actions[idx], hero_seat=game.hero_seat)
            _append_recent_event(events, ev)
            try:
                obs, _, done, truncated, _ = env.step(game.actions[idx])
            except step_errors as exc:
                warn = f"Replay stopped at step {idx} (older engine version): {exc}"
                return game, env, obs, idx, None, warn, events, last_priority_step
            if done or truncated:
                return game, env, obs, idx, None, None, events, last_priority_step
        nearest_priority = _resolve_nearest_priority(
            env, obs, game=game, target=target, last_priority_step=last_priority_step, step_errors=step_errors
        )
        chosen = game.actions[target] if target < len(game.actions) else None
        return game, env, obs, target, chosen, None, events, nearest_priority

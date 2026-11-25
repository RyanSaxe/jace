"""Native managym game replay and board serializer for Jace Studio.

Reads recorded games from `games/eval-iter-*.jsonl`, replays deterministically
through `managym.Env`, and serializes `Observation` fields directly (avoiding
`semantic_observation_json()`, which raises on certain reveal steps). Enforces
strict `Hero POV` vs. `Omniscient` hidden-information boundaries.
"""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import managym
from managym.enums import ActionEnum, ActionSpaceEnum, ZoneEnum

from jace.studio.runs import resolve_run_dir

if TYPE_CHECKING:
    from pathlib import Path

__all__ = (
    "PovMode",
    "RecordedGame",
    "ReplayService",
    "load_recorded_game",
    "serialize_observation",
)

type PovMode = Literal["hero", "omniscient"]

_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
_MAX_CACHED_GAMES = 16


@dataclass(frozen=True, slots=True)
class RecordedGame:
    run_id: str
    iteration: int
    game_idx: int
    seed: int
    players: tuple[tuple[str, dict[str, int]], ...]
    actions: tuple[int, ...]
    winner: int | None
    end_turn: int
    skip_trivial: bool


def _extract_player_specs(raw_players: object) -> tuple[tuple[str, dict[str, int]], ...]:
    if not isinstance(raw_players, list):
        return ()
    specs: list[tuple[str, dict[str, int]]] = []
    for entry in raw_players:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or f"player_{len(specs)}")
        raw_deck = entry.get("decklist")
        deck = {str(k): int(v) for k, v in raw_deck.items()} if isinstance(raw_deck, dict) else {}
        specs.append((name, deck))
    return tuple(specs)


def _iter_game_dicts(jsonl_path: Path) -> list[tuple[dict[str, object], bool]]:
    found: list[tuple[dict[str, object], bool]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            skip_trivial = bool(payload.get("skip_trivial", True))
            games = payload.get("games")
            items = games if isinstance(games, list) else [payload]
            found.extend((item, skip_trivial) for item in items if isinstance(item, dict) and "actions" in item)
    return found


def load_recorded_game(runs_dir: Path, *, run_id: str, iteration: int, game_idx: int) -> RecordedGame:
    run_dir = resolve_run_dir(runs_dir, run_id)
    jsonl_path = run_dir / "games" / f"eval-iter-{iteration:04d}.jsonl"
    if not jsonl_path.is_file():
        msg = f"eval game file not found: {jsonl_path}"
        raise FileNotFoundError(msg)
    entries = _iter_game_dicts(jsonl_path)
    if game_idx < 0 or game_idx >= len(entries):
        msg = f"game_idx {game_idx} out of range (0..{len(entries) - 1})"
        raise IndexError(msg)
    raw_game, skip_trivial = entries[game_idx]
    end = raw_game.get("end")
    winner = end.get("winner") if isinstance(end, dict) and isinstance(end.get("winner"), int) else None
    end_turn = int(end.get("turn") or 0) if isinstance(end, dict) and isinstance(end.get("turn"), (int, float)) else 0
    raw_actions = raw_game.get("actions")
    actions = tuple(int(a) for a in raw_actions) if isinstance(raw_actions, list) else ()
    raw_seed = raw_game.get("seed")
    seed = int(raw_seed) if isinstance(raw_seed, (int, float)) else 0
    return RecordedGame(
        run_id=run_id,
        iteration=iteration,
        game_idx=game_idx,
        seed=seed,
        players=_extract_player_specs(raw_game.get("players")),
        actions=actions,
        winner=winner,
        end_turn=end_turn,
        skip_trivial=skip_trivial,
    )


def _permanent_dict(perm: managym.Permanent) -> dict[str, object]:
    chars = perm.characteristics
    types = chars.card_types
    return {
        "id": int(perm.id),
        "name": str(chars.name),
        "is_land": bool(types.is_land),
        "is_creature": bool(types.is_creature),
        "is_artifact": bool(types.is_artifact),
        "is_enchantment": bool(types.is_enchantment),
        "is_token": bool(perm.is_token),
        "is_legendary": bool(chars.is_legendary),
        "power": int(chars.power or 0),
        "toughness": int(chars.toughness or 0),
        "damage": int(perm.damage),
        "plus1_counters": int(perm.plus1_counters),
        "tapped": bool(perm.tapped),
        "sick": bool(perm.is_summoning_sick),
        "mana_value": int(perm.mana_value),
    }


def _partition_permanents(
    perms: list[managym.Permanent],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    basic_counts: dict[tuple[str, bool], int] = {}
    nonbasic_lands: list[dict[str, object]] = []
    creatures: list[dict[str, object]] = []
    others: list[dict[str, object]] = []
    for perm in perms:
        data = _permanent_dict(perm)
        name = str(data["name"])
        if data["is_land"] and name in _BASIC_LANDS and not data["is_creature"]:
            key = (name, bool(data["tapped"]))
            basic_counts[key] = basic_counts.get(key, 0) + 1
        elif data["is_land"] and not data["is_creature"]:
            nonbasic_lands.append({"name": name, "tapped": bool(data["tapped"]), "count": 1, "id": data["id"]})
        elif data["is_creature"]:
            creatures.append(data)
        else:
            others.append(data)
    land_piles: list[dict[str, object]] = [
        {"name": name, "tapped": tapped, "count": count, "id": None}
        for (name, tapped), count in sorted(basic_counts.items(), key=lambda item: (item[0][1], item[0][0]))
    ]
    land_piles.extend(nonbasic_lands)
    return land_piles, creatures, others


def _hand_list(env: managym.Env, seat: int) -> list[str]:
    snapshot = env.hand_snapshot(seat)
    result: list[str] = []
    for name, count in sorted(snapshot.cards.items()):
        result.extend([str(name)] * int(count))
    return result


def _remaining_deck_summary(
    decklist: dict[str, int],
    *,
    known_outside_library: list[str],
    library_count: int,
) -> list[dict[str, object]]:
    remaining = Counter(decklist)
    for name in known_outside_library:
        if remaining[name] > 0:
            remaining[name] -= 1
    rows: list[dict[str, object]] = [
        {"name": name, "count": count} for name, count in sorted(remaining.items()) if count > 0
    ]
    if not rows and library_count > 0:
        return [{"name": "Unknown", "count": library_count}]
    return rows


def _serialize_player(
    *,
    env: managym.Env,
    seat: int,
    player_obs: managym.Player,
    perms: list[managym.Permanent],
    cards: list[managym.Card],
    decklist: dict[str, int],
    pov: PovMode,
    is_hero: bool,
) -> dict[str, object]:
    lands, creatures, others = _partition_permanents(perms)
    graveyard = [str(card.characteristics.name) for card in cards if card.zone == ZoneEnum.GRAVEYARD]
    exile = [str(card.characteristics.name) for card in cards if card.zone == ZoneEnum.EXILE]
    full_hand = _hand_list(env, seat)
    reveal_hand = is_hero or pov == "omniscient"
    zone_counts = player_obs.zone_counts
    lib_count = int(zone_counts[0]) if len(zone_counts) > 0 else 0
    seen_cards = [*full_hand, *graveyard, *exile, *(str(p.characteristics.name) for p in perms if not p.is_token)]
    lib_summary = (
        _remaining_deck_summary(decklist, known_outside_library=seen_cards, library_count=lib_count)
        if reveal_hand
        else []
    )
    return {
        "seat": seat,
        "life": int(player_obs.life),
        "zone_counts": {
            "library": lib_count,
            "hand": len(full_hand),
            "battlefield": len(perms),
            "graveyard": len(graveyard),
            "exile": len(exile),
        },
        "lands": lands,
        "creatures": creatures,
        "other_permanents": others,
        "hand": full_hand if reveal_hand else [],
        "hand_hidden": not reveal_hand,
        "graveyard": graveyard,
        "exile": exile,
        "library_summary": lib_summary,
        "library_hidden": not reveal_hand,
    }


def _build_id_name_map(obs: managym.Observation) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for perm in (*obs.agent_permanents, *obs.opponent_permanents):
        mapping[int(perm.id)] = str(perm.characteristics.name)
    for card in (*obs.agent_cards, *obs.opponent_cards):
        mapping[int(card.id)] = str(card.characteristics.name)
    return mapping


def _serialize_stack(obs: managym.Observation, id_map: dict[int, str]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for idx, obj in enumerate(obs.stack_objects):
        src_id = getattr(obj, "source_permanent_id", None)
        chars = getattr(obj, "characteristics", None)
        card_name = (
            str(chars.name)
            if chars is not None and getattr(chars, "name", None)
            else (id_map.get(int(src_id), "Spell/Ability") if isinstance(src_id, int) else "Spell/Ability")
        )
        desc = str(getattr(obj, "description", "") or card_name)
        items.append(
            {
                "index": idx + 1,
                "card_name": card_name,
                "description": desc,
                "kind": str(getattr(obj, "kind", "StackObject")).split(".")[-1],
                "source_permanent_id": int(src_id) if isinstance(src_id, int) else None,
                "controller_id": int(getattr(obj, "controller_id", 0)),
            }
        )
    return items


def _format_action_label(action: managym.Action, id_map: dict[int, str]) -> str:
    if action.description:
        return str(action.description)
    kind = ActionEnum(int(action.action_type)).name.replace("_", " ").title()
    focus = [int(x) for x in (getattr(action, "focus", None) or ())]
    if focus:
        names = ", ".join(id_map.get(fid, f"#{fid}") for fid in focus[:2])
        return f"{kind}: {names}"
    if int(action.action_type) in (int(ActionEnum.DECLARE_ATTACKER), int(ActionEnum.DECLARE_BLOCKER)):
        return f"{kind}: {'Yes' if action.declared is True else 'No'}"
    return kind


def _serialize_actions(obs: managym.Observation, id_map: dict[int, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, action in enumerate(obs.action_space.actions):
        focus = [int(x) for x in (getattr(action, "focus", None) or ())]
        rows.append(
            {
                "index": idx,
                "label": _format_action_label(action, id_map),
                "action_type": ActionEnum(int(action.action_type)).name,
                "focus_ids": focus,
            }
        )
    return rows


def _seat_aligned_obs(
    obs: managym.Observation,
) -> tuple[
    tuple[managym.Player, list[managym.Permanent], list[managym.Card]],
    tuple[managym.Player, list[managym.Permanent], list[managym.Card]],
]:
    if int(obs.agent.player_index) == 0:
        return (
            (obs.agent, list(obs.agent_permanents), list(obs.agent_cards)),
            (obs.opponent, list(obs.opponent_permanents), list(obs.opponent_cards)),
        )
    return (
        (obs.opponent, list(obs.opponent_permanents), list(obs.opponent_cards)),
        (obs.agent, list(obs.agent_permanents), list(obs.agent_cards)),
    )


def serialize_observation(
    env: managym.Env,
    obs: managym.Observation,
    *,
    decklists: tuple[dict[str, int], dict[str, int]],
    pov: PovMode = "hero",
    step_index: int = 0,
    total_steps: int = 0,
    chosen_action: int | None = None,
) -> dict[str, object]:
    hero_pack, opp_pack = _seat_aligned_obs(obs)
    id_map = _build_id_name_map(obs)
    space_kind = ActionSpaceEnum(int(obs.action_space.action_space_type)).name
    return {
        "step": step_index,
        "total_steps": total_steps,
        "turn": int(obs.turn.turn_number),
        "phase": str(obs.turn.phase).split(".")[-1],
        "step_name": str(obs.turn.step).split(".")[-1],
        "action_space_kind": space_kind,
        "acting_seat": int(obs.agent.player_index),
        "chosen_action": chosen_action,
        "pov": pov,
        "hero": _serialize_player(
            env=env,
            seat=0,
            player_obs=hero_pack[0],
            perms=hero_pack[1],
            cards=hero_pack[2],
            decklist=decklists[0],
            pov=pov,
            is_hero=True,
        ),
        "opp": _serialize_player(
            env=env,
            seat=1,
            player_obs=opp_pack[0],
            perms=opp_pack[1],
            cards=opp_pack[2],
            decklist=decklists[1],
            pov=pov,
            is_hero=False,
        ),
        "stack": _serialize_stack(obs, id_map),
        "legal_actions": _serialize_actions(obs, id_map),
    }


def _heuristic_win_estimate(obs: managym.Observation) -> float:
    hero_pack, opp_pack = _seat_aligned_obs(obs)
    life_diff = int(hero_pack[0].life) - int(opp_pack[0].life)
    hero_pow = sum(
        int(p.characteristics.power or 0) + int(p.characteristics.toughness or 0)
        for p in hero_pack[1]
        if p.characteristics.card_types.is_creature
    )
    opp_pow = sum(
        int(p.characteristics.power or 0) + int(p.characteristics.toughness or 0)
        for p in opp_pack[1]
        if p.characteristics.card_types.is_creature
    )
    score = 0.5 + 0.015 * life_diff + 0.012 * (hero_pow - opp_pow)
    return round(max(0.05, min(0.95, score)), 4)


def _new_env_for_game(game: RecordedGame) -> tuple[managym.Env, managym.Observation]:
    configs = [managym.PlayerConfig(name=name, decklist=dict(deck)) for name, deck in game.players]
    env = managym.Env(seed=game.seed, skip_trivial=game.skip_trivial)
    obs, _ = env.reset(configs)
    return env, obs


class ReplayService:
    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir.resolve()
        self._timeline_cache: OrderedDict[tuple[str, int, int], dict[str, object]] = OrderedDict()

    def _put_cache(self, key: tuple[str, int, int], value: dict[str, object]) -> None:
        self._timeline_cache[key] = value
        self._timeline_cache.move_to_end(key)
        while len(self._timeline_cache) > _MAX_CACHED_GAMES:
            self._timeline_cache.popitem(last=False)

    def get_game_overview(self, *, run_id: str, iteration: int, game_idx: int) -> dict[str, object]:
        key = (run_id, iteration, game_idx)
        cached = self._timeline_cache.get(key)
        if cached is not None:
            self._timeline_cache.move_to_end(key)
            return cached
        game = load_recorded_game(self._runs_dir, run_id=run_id, iteration=iteration, game_idx=game_idx)
        overview = self._build_game_overview(game)
        self._put_cache(key, overview)
        return overview

    def _build_game_overview(self, game: RecordedGame) -> dict[str, object]:
        env, obs = _new_env_for_game(game)
        turns_map: dict[int, dict[str, object]] = {}
        peak_step = 0
        peak_score = -1
        for step_idx, action in enumerate(game.actions):
            turn_num = int(obs.turn.turn_number)
            hero_pack, opp_pack = _seat_aligned_obs(obs)
            crowd = len(hero_pack[1]) + len(opp_pack[1]) + 2 * len(obs.stack_objects)
            if crowd >= peak_score:
                peak_score = crowd
                peak_step = step_idx
            turns_map[turn_num] = {
                "turn": turn_num,
                "step": step_idx,
                "hero_life": int(hero_pack[0].life),
                "opp_life": int(opp_pack[0].life),
                "hero_perms": len(hero_pack[1]),
                "opp_perms": len(opp_pack[1]),
                "stack_len": len(obs.stack_objects),
                "eval_win": _heuristic_win_estimate(obs),
            }
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        timeline = [turns_map[t] for t in sorted(turns_map)]
        return {
            "run_id": game.run_id,
            "iteration": game.iteration,
            "game_idx": game.game_idx,
            "seed": game.seed,
            "winner": game.winner,
            "total_steps": len(game.actions),
            "peak_step": peak_step,
            "players": [{"name": name, "deck_size": sum(deck.values())} for name, deck in game.players],
            "timeline": timeline,
        }

    def replay_to_step(
        self,
        *,
        run_id: str,
        iteration: int,
        game_idx: int,
        step: int,
    ) -> tuple[RecordedGame, managym.Env, managym.Observation, int, int | None]:
        game = load_recorded_game(self._runs_dir, run_id=run_id, iteration=iteration, game_idx=game_idx)
        env, obs = _new_env_for_game(game)
        target = max(0, min(step, max(0, len(game.actions) - 1)))
        for idx in range(target):
            obs, _, done, truncated, _ = env.step(game.actions[idx])
            if done or truncated:
                return game, env, obs, idx, None
        chosen = game.actions[target] if target < len(game.actions) else None
        return game, env, obs, target, chosen

"""Native managym game replay and board serializer for Jace Studio.

Reads recorded games from `games/eval-iter-*.jsonl`, preserves full `PlayerConfig`
setup (`disclosure`, `sideboard`, `world`, `decision_memory`) and mirrored
paired-deal `hero_seat`, replays deterministically through `managym.Env`, and
enforces strict `Hero POV` vs. `Omniscient` hidden-information boundaries.
"""

from __future__ import annotations

import json
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Literal

import managym
from managym.enums import ActionEnum, ActionSpaceEnum, ZoneEnum

from jace.studio.runs import resolve_run_dir

if TYPE_CHECKING:
    from pathlib import Path

__all__ = (
    "PlayerSpec",
    "PovMode",
    "RecordedGame",
    "ReplayService",
    "load_recorded_game",
    "non_permanent_card_names",
    "serialize_observation",
)

type PovMode = Literal["hero", "omniscient"]
type DisclosureMode = Literal["open", "closed"]

_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
_MANA_SYMBOLS = ("W", "U", "B", "R", "G", "C")
_MAX_CACHED_GAMES = 16
_MAX_RECENT_EVENTS = 6
_TWO_TARGETS = 2


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


def _is_spell_only_def(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return None
    types_obj = entry.get("types")
    types_list = types_obj.get("types") if isinstance(types_obj, dict) else None
    if isinstance(types_list, list) and types_list and set(types_list).issubset({"Instant", "Sorcery"}):
        name = entry.get("name")
        return name if isinstance(name, str) else None
    return None


@cache
def non_permanent_card_names() -> frozenset[str]:
    try:
        raw_defs = json.loads(managym.card_definitions_json())
    except (RuntimeError, ValueError, json.JSONDecodeError):
        return frozenset()
    if not isinstance(raw_defs, list):
        return frozenset()
    return frozenset(name for entry in raw_defs if (name := _is_spell_only_def(entry)) is not None)


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
    rec = entries[game_idx]
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


def _permanent_dict(perm: managym.Permanent) -> dict[str, object]:
    chars = perm.characteristics
    types = chars.card_types
    blocking = perm.blocking
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
        "attacking": bool(perm.attacking),
        "blocking": blocking is not None,
        "blocking_id": int(blocking) if blocking is not None else None,
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


def _format_mana_pool(player_obs: managym.Player) -> str:
    raw = getattr(player_obs, "mana", None)
    symbols: list[str] = []
    if isinstance(raw, (list, tuple)):
        for sym, count in zip(_MANA_SYMBOLS, raw, strict=False):
            n = int(count)
            if n > 0:
                symbols.extend([f"{{{sym}}}"] * min(n, 12))
    combat = int(getattr(player_obs, "combat_mana", 0) or 0)
    if combat > 0 and not symbols:
        symbols.extend(["{R}"] * min(combat, 12))
    return "".join(symbols)


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


def _stack_card_names_for_seat(obs: managym.Observation, seat: int) -> list[str]:
    names: list[str] = []
    for obj in obs.stack_objects:
        owner = getattr(obj, "owner_id", None)
        if not isinstance(owner, int) or owner != seat:
            continue
        chars = getattr(obj, "characteristics", None)
        if chars is not None and getattr(chars, "name", None):
            names.append(str(chars.name))
    return names


def _serialize_player(
    *,
    env: managym.Env,
    obs: managym.Observation,
    seat: int,
    player_obs: managym.Player,
    perms: list[managym.Permanent],
    cards: list[managym.Card],
    decklist: dict[str, int],
    pov: PovMode,
    is_hero: bool,
) -> dict[str, object]:
    lands, creatures, others = _partition_permanents(perms)
    # CR 404.2: reverse so the most recently put card is #1 (top of graveyard).
    graveyard = list(reversed([str(card.characteristics.name) for card in cards if card.zone == ZoneEnum.GRAVEYARD]))
    exile = list(reversed([str(card.characteristics.name) for card in cards if card.zone == ZoneEnum.EXILE]))
    full_hand = _hand_list(env, seat)
    reveal_hand = is_hero or pov == "omniscient"
    zone_counts = player_obs.zone_counts
    lib_count = int(zone_counts[0]) if len(zone_counts) > 0 else 0
    stack_cards = _stack_card_names_for_seat(obs, seat)
    seen_cards = [
        *full_hand,
        *graveyard,
        *exile,
        *stack_cards,
        *(str(p.characteristics.name) for p in perms if not p.is_token),
    ]
    lib_summary = (
        _remaining_deck_summary(decklist, known_outside_library=seen_cards, library_count=lib_count)
        if reveal_hand
        else []
    )
    open_lands = sum(1 for p in perms if p.characteristics.card_types.is_land and not p.tapped)
    return {
        "seat": seat,
        "life": int(player_obs.life),
        "is_active": bool(player_obs.is_active),
        "mana_pool": _format_mana_pool(player_obs),
        "open_lands": open_lands,
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
    for obj in obs.stack_objects:
        oid = getattr(obj, "stack_object_id", None) or getattr(obj, "object_id", None)
        chars = getattr(obj, "characteristics", None)
        if isinstance(oid, int) and chars is not None and getattr(chars, "name", None):
            mapping[oid] = str(chars.name)
    return mapping


def _resolve_stack_name(obj: managym.StackObject, id_map: dict[int, str]) -> str:
    chars = getattr(obj, "characteristics", None)
    if chars is not None and getattr(chars, "name", None):
        return str(chars.name)
    src_id = getattr(obj, "source_permanent_id", None)
    if isinstance(src_id, int) and src_id in id_map:
        return id_map[src_id]
    desc = str(getattr(obj, "description", "") or "")
    if " — " in desc:
        return desc.split(" — ", 1)[0].strip()
    return "Spell/Ability"


def _serialize_stack(
    obs: managym.Observation,
    *,
    id_map: dict[int, str],
    hero_seat: int,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for idx, obj in enumerate(obs.stack_objects):
        src_id = getattr(obj, "source_permanent_id", None)
        card_name = _resolve_stack_name(obj, id_map)
        desc = str(getattr(obj, "description", "") or card_name)
        raw_ctrl = getattr(obj, "controller_id", None)
        ctrl = raw_ctrl if isinstance(raw_ctrl, int) else 0
        items.append(
            {
                "index": idx + 1,
                "card_name": card_name,
                "description": desc,
                "kind": str(getattr(obj, "kind", "StackObject")).split(".")[-1],
                "source_permanent_id": int(src_id) if isinstance(src_id, int) else None,
                "controller_id": ctrl,
                "controller_side": "hero" if ctrl == hero_seat else "opp",
            }
        )
    return items


def _format_combat_label(action: managym.Action, id_map: dict[int, str]) -> str:
    focus = [int(x) for x in (getattr(action, "focus", None) or ())]
    names = [id_map.get(fid, f"#{fid}") for fid in focus[:_TWO_TARGETS]]
    act_type = int(action.action_type)
    if act_type == int(ActionEnum.DECLARE_ATTACKER):
        target = names[0] if names else "creature"
        return f"Attack with {target}" if action.declared is True else f"Don't attack ({target})"
    if len(names) >= _TWO_TARGETS:
        return f"Block {names[1]} with {names[0]}"
    blocker = names[0] if names else "creature"
    return f"Block with {blocker}" if action.declared is True else f"Don't block ({blocker})"


def _format_action_label(action: managym.Action, id_map: dict[int, str], *, stack_top: str | None = None) -> str:
    act_type = int(action.action_type)
    if act_type in (int(ActionEnum.DECLARE_ATTACKER), int(ActionEnum.DECLARE_BLOCKER)):
        return _format_combat_label(action, id_map)
    if action.description:
        return str(action.description)
    kind = ActionEnum(act_type).name.replace("_", " ").title()
    focus = [int(x) for x in (getattr(action, "focus", None) or ())]
    if focus:
        resolved = [id_map.get(fid, stack_top or f"#{fid}") for fid in focus[:_TWO_TARGETS]]
        return f"{kind}: {', '.join(resolved)}"
    return kind


def _serialize_actions(
    obs: managym.Observation,
    *,
    id_map: dict[int, str],
    stack_top: str | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, action in enumerate(obs.action_space.actions):
        focus = [int(x) for x in (getattr(action, "focus", None) or ())]
        rows.append(
            {
                "index": idx,
                "label": _format_action_label(action, id_map, stack_top=stack_top),
                "action_type": ActionEnum(int(action.action_type)).name,
                "focus_ids": focus,
            }
        )
    return rows


def _seat_aligned_obs(
    obs: managym.Observation,
    *,
    hero_seat: int = 0,
) -> tuple[
    tuple[managym.Player, list[managym.Permanent], list[managym.Card]],
    tuple[managym.Player, list[managym.Permanent], list[managym.Card]],
]:
    if int(obs.agent.player_index) == hero_seat:
        return (
            (obs.agent, list(obs.agent_permanents), list(obs.agent_cards)),
            (obs.opponent, list(obs.opponent_permanents), list(obs.opponent_cards)),
        )
    return (
        (obs.opponent, list(obs.opponent_permanents), list(obs.opponent_cards)),
        (obs.agent, list(obs.agent_permanents), list(obs.agent_cards)),
    )


def _format_phase_step(obs: managym.Observation, space_kind: str) -> tuple[str, str]:
    turn = int(obs.turn.turn_number)
    if turn == 0 or space_kind == "MULLIGAN":
        return "Pregame", "Mulligan"
    return str(obs.turn.phase).split(".")[-1], str(obs.turn.step).split(".")[-1]


def _filter_recent_events(
    events: list[dict[str, object]] | None,
    *,
    pov: PovMode,
) -> list[dict[str, object]]:
    if not events:
        return []
    out: list[dict[str, object]] = []
    for ev in events[-_MAX_RECENT_EVENTS:]:
        if pov == "hero" and ev.get("actor_side") == "opp" and ev.get("private_hand"):
            out.append({**ev, "text": "Opponent chose a hidden hand action"})
        else:
            out.append(ev)
    return out


def serialize_observation(
    env: managym.Env,
    obs: managym.Observation,
    *,
    decklists: tuple[dict[str, int], dict[str, int]],
    hero_seat: int = 0,
    pov: PovMode = "hero",
    step_index: int = 0,
    total_steps: int = 0,
    chosen_action: int | None = None,
    recent_events: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    opp_seat = 1 - hero_seat
    hero_pack, opp_pack = _seat_aligned_obs(obs, hero_seat=hero_seat)
    id_map = _build_id_name_map(obs)
    space_kind = ActionSpaceEnum(int(obs.action_space.action_space_type)).name
    acting_seat = int(obs.agent.player_index)
    actor_is_hero = acting_seat == hero_seat
    active_is_hero = bool(hero_pack[0].is_active)
    phase, step_name = _format_phase_step(obs, space_kind)
    stack = _serialize_stack(obs, id_map=id_map, hero_seat=hero_seat)
    stack_top = str(stack[0]["card_name"]) if stack else None
    actions_hidden = pov == "hero" and not actor_is_hero
    legal_actions = [] if actions_hidden else _serialize_actions(obs, id_map=id_map, stack_top=stack_top)
    non_perm_set = non_permanent_card_names()
    non_perms = sorted({name for deck in decklists for name in deck if name in non_perm_set})
    return {
        "step": step_index,
        "total_steps": total_steps,
        "turn": int(obs.turn.turn_number),
        "phase": phase,
        "step_name": step_name,
        "action_space_kind": space_kind,
        "hero_seat": hero_seat,
        "acting_seat": acting_seat,
        "actor_is_hero": actor_is_hero,
        "active_is_hero": active_is_hero,
        "actions_hidden": actions_hidden,
        "chosen_action": None if actions_hidden else chosen_action,
        "pov": pov,
        "hero": _serialize_player(
            env=env,
            obs=obs,
            seat=hero_seat,
            player_obs=hero_pack[0],
            perms=hero_pack[1],
            cards=hero_pack[2],
            decklist=decklists[hero_seat],
            pov=pov,
            is_hero=True,
        ),
        "opp": _serialize_player(
            env=env,
            obs=obs,
            seat=opp_seat,
            player_obs=opp_pack[0],
            perms=opp_pack[1],
            cards=opp_pack[2],
            decklist=decklists[opp_seat],
            pov=pov,
            is_hero=False,
        ),
        "stack": stack,
        "legal_actions": legal_actions,
        "recent_events": _filter_recent_events(recent_events, pov=pov),
        "non_permanent_names": non_perms,
    }


def _heuristic_win_estimate(obs: managym.Observation, *, hero_seat: int = 0) -> float:
    hero_pack, opp_pack = _seat_aligned_obs(obs, hero_seat=hero_seat)
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


def _engine_step_errors() -> tuple[type[BaseException], ...]:
    candidates = (*BaseException.__subclasses__(), *Exception.__subclasses__())
    engine_errs = tuple(cls for cls in candidates if cls.__name__ in ("PanicException", "PyAgentError"))
    return (*engine_errs, RuntimeError, ValueError)


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
    id_map = _build_id_name_map(obs)
    stack_top = _resolve_stack_name(obs.stack_objects[0], id_map) if obs.stack_objects else None
    label = _format_action_label(action, id_map, stack_top=stack_top)
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


class ReplayService:
    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir.resolve()
        self._timeline_cache: OrderedDict[tuple[str, int, int], dict[str, object]] = OrderedDict()
        self._lock = threading.Lock()

    def _put_cache(self, key: tuple[str, int, int], value: dict[str, object]) -> None:
        with self._lock:
            self._timeline_cache[key] = value
            self._timeline_cache.move_to_end(key)
            while len(self._timeline_cache) > _MAX_CACHED_GAMES:
                self._timeline_cache.popitem(last=False)

    def get_game_overview(self, *, run_id: str, iteration: int, game_idx: int) -> dict[str, object]:
        key = (run_id, iteration, game_idx)
        with self._lock:
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
        step_errors = _engine_step_errors()
        turns_map: dict[int, dict[str, object]] = {}
        peak_step = 0
        peak_score = -1
        last_valid_step = 0
        warning: str | None = None
        for step_idx, action in enumerate(game.actions):
            last_valid_step = step_idx
            turn_num = int(obs.turn.turn_number)
            hero_pack, opp_pack = _seat_aligned_obs(obs, hero_seat=game.hero_seat)
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
                "eval_win": _heuristic_win_estimate(obs, hero_seat=game.hero_seat),
            }
            try:
                obs, _, done, truncated, _ = env.step(action)
            except step_errors as exc:
                warning = f"Replay stopped at step {step_idx} (recorded on an older engine version): {exc}"
                break
            if done or truncated:
                break
        timeline = [turns_map[t] for t in sorted(turns_map)]
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
    ) -> tuple[RecordedGame, managym.Env, managym.Observation, int, int | None, str | None, list[dict[str, object]]]:
        game = load_recorded_game(self._runs_dir, run_id=run_id, iteration=iteration, game_idx=game_idx)
        env, obs = _new_env_for_game(game)
        step_errors = _engine_step_errors()
        target = max(0, min(step, max(0, len(game.actions) - 1)))
        events: list[dict[str, object]] = []
        for idx in range(target):
            ev = _event_for_step(obs, step_idx=idx, action_idx=game.actions[idx], hero_seat=game.hero_seat)
            if ev is not None:
                events.append(ev)
                if len(events) > _MAX_RECENT_EVENTS:
                    del events[0]
            try:
                obs, _, done, truncated, _ = env.step(game.actions[idx])
            except step_errors as exc:
                warn = f"Replay stopped at step {idx} (older engine version): {exc}"
                return game, env, obs, idx, None, warn, events
            if done or truncated:
                return game, env, obs, idx, None, None, events
        raw_chosen = game.actions[target] if target < len(game.actions) else None
        chosen = raw_chosen if isinstance(raw_chosen, int) else None
        return game, env, obs, target, chosen, None, events

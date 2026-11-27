"""Board, player, stack, and disambiguated action serialization for Jace Studio.

Translates `managym.Observation` snapshots into JSON-ready dictionaries with
strict `Hero POV` vs. `Omniscient` hidden-information boundaries and stable
per-player `copy_index` disambiguation (`#1`, `#2`, ...) for duplicate
permanents sharing a card name.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import cache
from typing import Literal

import managym
from managym.enums import ActionEnum, ActionSpaceEnum, ZoneEnum

__all__ = (
    "PovMode",
    "format_action_for_event",
    "heuristic_win_estimate",
    "non_permanent_card_names",
    "seat_aligned_life",
    "seat_aligned_obs",
    "serialize_legal_actions",
    "serialize_observation",
)

type PovMode = Literal["hero", "omniscient"]

_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
_MANA_SYMBOLS = ("W", "U", "B", "R", "G", "C")
_MAX_RECENT_EVENTS = 6
_TWO_TARGETS = 2
_MIN_DUPLICATE_COUNT = 2


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


def _compute_copy_indices(perms: list[managym.Permanent]) -> dict[int, int]:
    by_name: dict[str, list[int]] = {}
    for perm in perms:
        by_name.setdefault(str(perm.characteristics.name), []).append(int(perm.id))
    return {
        perm_id: copy_num
        for perm_ids in by_name.values()
        if len(perm_ids) >= _MIN_DUPLICATE_COUNT
        for copy_num, perm_id in enumerate(sorted(perm_ids), start=1)
    }


def _all_copy_indices(obs: managym.Observation) -> dict[int, int]:
    agent_copies = _compute_copy_indices(list(obs.agent_permanents))
    opp_copies = _compute_copy_indices(list(obs.opponent_permanents))
    return {**agent_copies, **opp_copies}


def _permanent_dict(perm: managym.Permanent, *, copy_index: int | None = None) -> dict[str, object]:
    chars = perm.characteristics
    types = chars.card_types
    blocking = perm.blocking
    return {
        "id": int(perm.id),
        "name": str(chars.name),
        "copy_index": copy_index,
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
    *,
    copy_indices: dict[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    basic_counts: dict[tuple[str, bool], int] = {}
    nonbasic_lands: list[dict[str, object]] = []
    creatures: list[dict[str, object]] = []
    others: list[dict[str, object]] = []
    for perm in perms:
        data = _permanent_dict(perm, copy_index=copy_indices.get(int(perm.id)))
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
    copy_indices: dict[int, int],
    cards: list[managym.Card],
    decklist: dict[str, int],
    pov: PovMode,
    is_hero: bool,
) -> dict[str, object]:
    lands, creatures, others = _partition_permanents(perms, copy_indices=copy_indices)
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


def _tag_single_copy_in_label(label: str, *, base_name: str, copy_num: int) -> str:
    tagged = f"{base_name} #{copy_num}"
    if tagged in label:
        return label
    pattern = rf"{re.escape(base_name)}(?!\s*#\d+)"
    updated, count = re.subn(pattern, tagged, label, count=1)
    if count > 0:
        return updated
    return label if f"#{copy_num}" in label else f"{label} ({tagged})"


def _disambiguate_description(
    label: str,
    *,
    focus: list[int],
    id_map: dict[int, str],
    copy_indices: dict[int, int],
) -> str:
    result = label
    for fid in focus:
        copy_num = copy_indices.get(fid)
        base_name = id_map.get(fid)
        if copy_num is not None and base_name:
            result = _tag_single_copy_in_label(result, base_name=base_name, copy_num=copy_num)
    return result


def _format_action_label(
    action: managym.Action,
    id_map: dict[int, str],
    *,
    action_id_map: dict[int, str],
    copy_indices: dict[int, int],
    stack_top: str | None = None,
) -> str:
    act_type = int(action.action_type)
    if act_type in (int(ActionEnum.DECLARE_ATTACKER), int(ActionEnum.DECLARE_BLOCKER)):
        return _format_combat_label(action, action_id_map)
    focus = [int(x) for x in (getattr(action, "focus", None) or ())]
    if action.description:
        return _disambiguate_description(str(action.description), focus=focus, id_map=id_map, copy_indices=copy_indices)
    kind = ActionEnum(act_type).name.replace("_", " ").title()
    if focus:
        resolved = [action_id_map.get(fid, stack_top or f"#{fid}") for fid in focus[:_TWO_TARGETS]]
        return f"{kind}: {', '.join(resolved)}"
    return kind


def _serialize_actions(
    obs: managym.Observation,
    *,
    id_map: dict[int, str],
    copy_indices: dict[int, int],
    stack_top: str | None,
) -> list[dict[str, object]]:
    action_id_map = {
        oid: f"{name} #{copy_indices[oid]}" if oid in copy_indices else name for oid, name in id_map.items()
    }
    rows: list[dict[str, object]] = []
    for idx, action in enumerate(obs.action_space.actions):
        focus = [int(x) for x in (getattr(action, "focus", None) or ())]
        rows.append(
            {
                "index": idx,
                "label": _format_action_label(
                    action,
                    id_map,
                    action_id_map=action_id_map,
                    copy_indices=copy_indices,
                    stack_top=stack_top,
                ),
                "action_type": ActionEnum(int(action.action_type)).name,
                "focus_ids": focus,
            }
        )
    return rows


def format_action_for_event(obs: managym.Observation, action: managym.Action) -> str:
    id_map = _build_id_name_map(obs)
    copy_indices = _all_copy_indices(obs)
    action_id_map = {
        oid: f"{name} #{copy_indices[oid]}" if oid in copy_indices else name for oid, name in id_map.items()
    }
    stack_top = _resolve_stack_name(obs.stack_objects[0], id_map) if obs.stack_objects else None
    return _format_action_label(
        action,
        id_map,
        action_id_map=action_id_map,
        copy_indices=copy_indices,
        stack_top=stack_top,
    )


def serialize_legal_actions(obs: managym.Observation, *, hero_seat: int = 0) -> list[dict[str, object]]:
    id_map = _build_id_name_map(obs)
    copy_indices = _all_copy_indices(obs)
    stack = _serialize_stack(obs, id_map=id_map, hero_seat=hero_seat)
    stack_top = str(stack[0]["card_name"]) if stack else None
    return _serialize_actions(obs, id_map=id_map, copy_indices=copy_indices, stack_top=stack_top)


def seat_aligned_obs(
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


def seat_aligned_life(obs: managym.Observation, *, hero_seat: int = 0) -> tuple[int, int]:
    hero_pack, opp_pack = seat_aligned_obs(obs, hero_seat=hero_seat)
    return int(hero_pack[0].life), int(opp_pack[0].life)


def heuristic_win_estimate(obs: managym.Observation, *, hero_seat: int = 0) -> float:
    hero_pack, opp_pack = seat_aligned_obs(obs, hero_seat=hero_seat)
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
    hero_pack, opp_pack = seat_aligned_obs(obs, hero_seat=hero_seat)
    hero_copies = _compute_copy_indices(hero_pack[1])
    opp_copies = _compute_copy_indices(opp_pack[1])
    all_copies = {**hero_copies, **opp_copies}
    id_map = _build_id_name_map(obs)
    space_kind = ActionSpaceEnum(int(obs.action_space.action_space_type)).name
    acting_seat = int(obs.agent.player_index)
    actor_is_hero = acting_seat == hero_seat
    active_is_hero = bool(hero_pack[0].is_active)
    phase, step_name = _format_phase_step(obs, space_kind)
    stack = _serialize_stack(obs, id_map=id_map, hero_seat=hero_seat)
    stack_top = str(stack[0]["card_name"]) if stack else None
    actions_hidden = pov == "hero" and not actor_is_hero
    legal_actions = (
        [] if actions_hidden else _serialize_actions(obs, id_map=id_map, copy_indices=all_copies, stack_top=stack_top)
    )
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
            copy_indices=hero_copies,
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
            copy_indices=opp_copies,
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

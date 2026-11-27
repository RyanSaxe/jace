"""Stateless counterfactual board/hand edits and any-step action branching.

Clones a `managym.Env` at any replayed step, applies stage-indexed board and
hand edits (with CR 304.4 validation against putting instants/sorceries onto
the battlefield), and steps through counterfactual action branches while
recording `BranchStepEvent` metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from managym.enums import ActionEnum

from jace.studio.replay import engine_step_errors, non_permanent_card_names

if TYPE_CHECKING:
    import managym

__all__ = (
    "BranchStepEvent",
    "ForkEdit",
    "apply_fork_and_branch",
    "apply_fork_edits",
)


@dataclass(frozen=True, slots=True)
class ForkEdit:
    op: str
    seat: int
    value: int | None = None
    card: str | None = None
    replacement_card: str | None = None
    after_branch: int = 0


@dataclass(frozen=True, slots=True)
class BranchStepEvent:
    branch_index: int
    turn: int
    actor_side: str
    action_index: int
    label: str


def _hand_cards_list(env: managym.Env, seat: int) -> list[str]:
    snap = env.hand_snapshot(seat)
    return [str(name) for name, count in sorted(snap.cards.items()) for _ in range(int(count))]


def _remove_hand_card(cloned: managym.Env, *, seat: int, card: str) -> str | None:
    cards = _hand_cards_list(cloned, seat)
    if card not in cards:
        return f"Card '{card}' is not in seat {seat}'s hand."
    cards.remove(card)
    cloned.scenario_clear_hand(seat)
    for name in cards:
        cloned.scenario_force_card_in_hand(seat, name)
    return None


def _swap_hand_card(
    cloned: managym.Env,
    *,
    seat: int,
    card: str,
    replacement_card: str,
) -> str | None:
    cards = _hand_cards_list(cloned, seat)
    if card not in cards:
        return f"Card '{card}' is not in seat {seat}'s hand."
    cards.remove(card)
    cloned.scenario_clear_hand(seat)
    step_errors = engine_step_errors()
    try:
        cloned.scenario_force_card_in_hand(seat, replacement_card)
    except step_errors as exc:
        for name in (*cards, card):
            cloned.scenario_force_card_in_hand(seat, name)
        return f"Cannot swap '{card}' for '{replacement_card}': {exc}"
    for name in cards:
        cloned.scenario_force_card_in_hand(seat, name)
    return None


def _apply_hand_edit(cloned: managym.Env, edit: ForkEdit) -> str | None:
    if edit.op == "clear_hand":
        cloned.scenario_clear_hand(edit.seat)
        return None
    if edit.op == "add_hand_card" and edit.card:
        cloned.scenario_force_card_in_hand(edit.seat, edit.card)
        return None
    if edit.op == "remove_hand_card" and edit.card:
        return _remove_hand_card(cloned, seat=edit.seat, card=edit.card)
    if edit.op == "swap_hand_card" and edit.card and edit.replacement_card:
        return _swap_hand_card(cloned, seat=edit.seat, card=edit.card, replacement_card=edit.replacement_card)
    return f"unsupported fork edit: {edit.op}"


def _apply_single_edit(cloned: managym.Env, edit: ForkEdit) -> str | None:
    if edit.op == "set_life" and edit.value is not None:
        cloned.scenario_set_life(edit.seat, max(1, min(99, int(edit.value))))
        return None
    if edit.op == "add_battlefield" and edit.card:
        if edit.card in non_permanent_card_names():
            return f"Cannot put Instant/Sorcery '{edit.card}' onto the battlefield (CR 304.4) - add to Hand instead."
        cloned.scenario_force_battlefield(edit.seat, edit.card, ready=True)
        return None
    return _apply_hand_edit(cloned, edit)


def _describe_branch_action(obs: managym.Observation, action_idx: int) -> str:
    actions = obs.action_space.actions
    if action_idx < 0 or action_idx >= len(actions):
        return f"Action #{action_idx}"
    action = actions[action_idx]
    if action.description:
        return str(action.description)
    return ActionEnum(int(action.action_type)).name.replace("_", " ").title()


def apply_fork_edits(
    env: managym.Env,
    base_obs: managym.Observation,
    *,
    edits: list[ForkEdit],
) -> tuple[managym.Env, managym.Observation, list[str]]:
    if not edits:
        return env.clone_env(), base_obs, []
    step_errors = engine_step_errors()
    cloned = env.clone_env()
    warnings: list[str] = []
    for edit in edits:
        try:
            warn = _apply_single_edit(cloned, edit)
            if warn is not None:
                warnings.append(warn)
        except step_errors as exc:
            warnings.append(f"{edit.op} ({edit.card or edit.value}): {exc}")
    try:
        refreshed = cloned.scenario_refresh()
    except step_errors as exc:
        warnings.append(f"Board edits require a Priority step ({exc})")
        return env.clone_env(), base_obs, warnings
    return cloned, refreshed, warnings


def _group_edits_by_stage(edits: list[ForkEdit], max_stage: int) -> dict[int, list[ForkEdit]]:
    grouped: dict[int, list[ForkEdit]] = {}
    for edit in edits:
        stage = max(0, min(int(edit.after_branch), max_stage))
        grouped.setdefault(stage, []).append(edit)
    return grouped


def _step_branch_once(
    cloned: managym.Env,
    current_obs: managym.Observation,
    *,
    idx: int,
    action_idx: int,
    hero_seat: int,
    step_errors: tuple[type[BaseException], ...],
) -> tuple[managym.Observation, BranchStepEvent | None, bool, str | None]:
    actions = current_obs.action_space.actions
    if action_idx < 0 or action_idx >= len(actions):
        err = f"Branch step {idx + 1}: action index {action_idx} out of bounds ({len(actions)} legal)"
        return current_obs, None, False, err
    label = _describe_branch_action(current_obs, action_idx)
    actor_side = "hero" if int(current_obs.agent.player_index) == hero_seat else "opp"
    turn_num = int(current_obs.turn.turn_number)
    try:
        next_obs, _, done, truncated, _ = cloned.step(action_idx)
    except step_errors as exc:
        return current_obs, None, False, f"Branch step {idx + 1} failed: {exc}"
    event = BranchStepEvent(
        branch_index=idx, turn=turn_num, actor_side=actor_side, action_index=action_idx, label=label
    )
    return next_obs, event, bool(done or truncated), None


def apply_fork_and_branch(
    env: managym.Env,
    base_obs: managym.Observation,
    *,
    edits: list[ForkEdit],
    branch_actions: list[int],
    hero_seat: int = 0,
) -> tuple[managym.Env, managym.Observation, list[str], list[BranchStepEvent], bool]:
    step_errors = engine_step_errors()
    edits_by_stage = _group_edits_by_stage(edits, len(branch_actions))
    cloned, current_obs, warnings = apply_fork_edits(env, base_obs, edits=edits_by_stage.get(0, []))
    branch_events: list[BranchStepEvent] = []
    branch_done = False
    for idx, action_idx in enumerate(branch_actions):
        next_obs, event, is_done, err = _step_branch_once(
            cloned, current_obs, idx=idx, action_idx=action_idx, hero_seat=hero_seat, step_errors=step_errors
        )
        if err is not None or event is None:
            if err is not None:
                warnings.append(err)
            break
        branch_events.append(event)
        current_obs = next_obs
        if is_done:
            branch_done = True
            break
        stage_edits = edits_by_stage.get(idx + 1, [])
        if stage_edits:
            cloned, current_obs, stage_warns = apply_fork_edits(cloned, current_obs, edits=stage_edits)
            warnings.extend(stage_warns)
    return cloned, current_obs, warnings, branch_events, branch_done

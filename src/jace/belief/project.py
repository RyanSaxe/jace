"""Counterfactual opponent belief override projection enforcing all four legality invariants.

Projects user-specified pinned hand copies, excluded cards, and copy-count
multipliers onto a legal `BeliefState` where:
1. Pinned counts are fixed exactly and excluded cards receive zero mass.
2. `sum(hand_probs) == opp_hand_size` with proportional waterfill over unedited cards.
3. `0 <= hand_probs <= deck_copies <= max(0, copy_limits - revealed_counts)`.
4. `sum(deck_copies) == opp_hand_size + opp_library_size`.
"""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from jace.belief.types import BeliefOverride, BeliefState

__all__ = ("project_belief_overrides",)

_ZERO_EPS = 1e-7
_TOLERANCE = 1e-5


def project_belief_overrides(
    belief: BeliefState,
    overrides: Sequence[BeliefOverride],
    *,
    revealed_counts: Tensor,
    copy_limits: Tensor | None = None,
) -> BeliefState:
    """Apply counterfactual overrides while enforcing all four conservation and bound invariants.

    Raises `ValueError` when pinned counts exceed `opp_hand_size`, conflict with
    `excluded_cards`, violate `copy_limits - revealed_counts`, or leave too
    little legal copy capacity to fill `opp_hand_size + opp_library_size`.
    """
    num_cards = len(belief.card_names)
    _validate_projection_inputs(revealed_counts, copy_limits=copy_limits, num_cards=num_cards)
    name_to_idx = {name: idx for idx, name in enumerate(belief.card_names)}
    pinned, excluded, multipliers = _merge_overrides(overrides, name_to_idx=name_to_idx)
    max_unseen = _compute_max_unseen(
        revealed_counts, copy_limits=copy_limits, excluded_indices=excluded, device=belief.deck_copies.device
    )
    _validate_override_feasibility(
        pinned=pinned,
        excluded=excluded,
        max_unseen=max_unseen,
        opp_hand_size=belief.opp_hand_size,
        opp_library_size=belief.opp_library_size,
    )
    projected_hand = _project_hand_probs(
        belief.hand_probs,
        pinned=pinned,
        multipliers=multipliers,
        max_unseen=max_unseen,
        opp_hand_size=belief.opp_hand_size,
    )
    projected_deck = _project_deck_copies(
        belief.deck_copies,
        hand_probs=projected_hand,
        multipliers=multipliers,
        max_unseen=max_unseen,
        unseen_total=belief.opp_hand_size + belief.opp_library_size,
    )
    return BeliefState(
        card_names=belief.card_names,
        deck_copies=projected_deck,
        hand_probs=projected_hand,
        opp_hand_size=belief.opp_hand_size,
        opp_library_size=belief.opp_library_size,
        prior_backend=belief.prior_backend,
        matchup_equity=belief.matchup_equity,
    )


def _validate_projection_inputs(revealed_counts: Tensor, *, copy_limits: Tensor | None, num_cards: int) -> None:
    if revealed_counts.shape != (num_cards,) or bool((revealed_counts < 0).any()):
        msg = f"revealed_counts must be non-negative with shape ({num_cards},)"
        raise ValueError(msg)
    if copy_limits is not None and (copy_limits.shape != (num_cards,) or bool((copy_limits < 0).any())):
        msg = f"copy_limits must be non-negative with shape ({num_cards},)"
        raise ValueError(msg)


def _merge_overrides(
    overrides: Sequence[BeliefOverride], *, name_to_idx: dict[str, int]
) -> tuple[dict[int, int], set[int], dict[int, float]]:
    pinned: dict[int, int] = {}
    excluded: set[int] = set()
    multipliers: dict[int, float] = {}
    for override_item in overrides:
        excluded.update(_card_index(name, name_to_idx) for name in override_item.excluded_cards)
        _collect_pinned(override_item.pinned_in_hand, name_to_idx=name_to_idx, out=pinned)
        _collect_multipliers(override_item.copy_multipliers, name_to_idx=name_to_idx, out=multipliers)
    return pinned, excluded, multipliers


def _collect_pinned(entries: dict[str, int], *, name_to_idx: dict[str, int], out: dict[int, int]) -> None:
    for name, count in entries.items():
        if count < 0:
            msg = f"pinned_in_hand count for {name!r} must be non-negative, got {count}"
            raise ValueError(msg)
        out[_card_index(name, name_to_idx)] = count


def _collect_multipliers(entries: dict[str, float], *, name_to_idx: dict[str, int], out: dict[int, float]) -> None:
    for name, mult in entries.items():
        if not math.isfinite(mult) or mult < 0.0:
            msg = f"copy_multipliers value for {name!r} must be finite and non-negative, got {mult}"
            raise ValueError(msg)
        out[_card_index(name, name_to_idx)] = mult


def _card_index(name: str, name_to_idx: dict[str, int]) -> int:
    if name not in name_to_idx:
        msg = f"unknown card name in override: {name!r}"
        raise ValueError(msg)
    return name_to_idx[name]


def _compute_max_unseen(
    revealed_counts: Tensor, *, copy_limits: Tensor | None, excluded_indices: set[int], device: torch.device
) -> Tensor:
    revealed_f = revealed_counts.to(device=device, dtype=torch.float32)
    max_unseen = (
        torch.full_like(revealed_f, torch.inf)
        if copy_limits is None
        else (copy_limits.to(device=device, dtype=torch.float32) - revealed_f).clamp_min(0.0)
    )
    for idx in excluded_indices:
        max_unseen[idx] = 0.0
    return max_unseen


def _validate_override_feasibility(
    *, pinned: dict[int, int], excluded: set[int], max_unseen: Tensor, opp_hand_size: int, opp_library_size: int
) -> None:
    pinned_total = sum(pinned.values())
    if pinned_total > opp_hand_size:
        msg = f"total pinned_in_hand ({pinned_total}) exceeds opp_hand_size ({opp_hand_size})"
        raise ValueError(msg)
    for idx, count in pinned.items():
        if idx in excluded and count > 0:
            msg = "cannot pin a card that is also in excluded_cards"
            raise ValueError(msg)
        cap = float(max_unseen[idx].item())
        if count > cap + _TOLERANCE:
            msg = f"pinned count {count} exceeds remaining legal copy limit {cap}"
            raise ValueError(msg)
    total_capacity = float(max_unseen.sum().item())
    if total_capacity < float(opp_hand_size + opp_library_size) - _TOLERANCE:
        msg = (
            f"remaining copy capacity ({total_capacity}) cannot satisfy "
            f"opp_hand_size + opp_library_size ({opp_hand_size + opp_library_size})"
        )
        raise ValueError(msg)


def _project_hand_probs(
    hand_probs: Tensor,
    *,
    pinned: dict[int, int],
    multipliers: dict[int, float],
    max_unseen: Tensor,
    opp_hand_size: int,
) -> Tensor:
    weights = hand_probs.to(dtype=torch.float32).clone().clamp_min(0.0)
    for idx, mult in multipliers.items():
        weights[idx] *= mult
    lower = torch.zeros_like(weights)
    upper = max_unseen.clone()
    for idx, count in pinned.items():
        lower[idx] = float(count)
        upper[idx] = float(count)
        weights[idx] = 0.0
    return _waterfill_scale(weights, target=float(opp_hand_size), lower=lower, upper=upper)


def _project_deck_copies(
    deck_copies: Tensor,
    *,
    hand_probs: Tensor,
    multipliers: dict[int, float],
    max_unseen: Tensor,
    unseen_total: int,
) -> Tensor:
    base = deck_copies.to(dtype=torch.float32).clone().clamp_min(0.0)
    for idx, mult in multipliers.items():
        base[idx] *= mult
    base = torch.where(max_unseen > 0.0, base, torch.zeros_like(base))
    lib_weights = (base - hand_probs).clamp_min(0.0)
    weights = lib_weights if float(lib_weights.sum().item()) > 0.0 else base
    return _waterfill_scale(weights, target=float(unseen_total), lower=hand_probs, upper=max_unseen)


def _waterfill_scale(weights: Tensor, *, target: float, lower: Tensor, upper: Tensor) -> Tensor:
    lower_sum = float(lower.sum().item())
    remaining = target - lower_sum
    if remaining < -_TOLERANCE:
        msg = f"target total {target} is below lower bound sum {lower_sum}"
        raise ValueError(msg)
    if remaining <= _ZERO_EPS:
        return lower.clone()
    room = (upper - lower).clamp_min(0.0)
    if float(room.sum().item()) < remaining - _TOLERANCE:
        msg = f"cannot distribute {remaining} across remaining capacity {float(room.sum().item())}"
        raise ValueError(msg)
    extra = torch.zeros_like(lower)
    active_weights = torch.where(room > 0.0, weights.clamp_min(0.0), torch.zeros_like(weights))
    return lower + _fill_headroom(extra, room=room, weights=active_weights, remaining=remaining)


def _fill_headroom(extra: Tensor, *, room: Tensor, weights: Tensor, remaining: float) -> Tensor:
    active_weights = weights.clone()
    for _ in range(extra.numel() + 1):
        if remaining <= _ZERO_EPS:
            return extra
        avail = (room - extra).clamp_min(0.0)
        active = (avail > _ZERO_EPS) & (active_weights > 0.0)
        if not bool(active.any()):
            active = avail > _ZERO_EPS
            active_weights = active.to(dtype=torch.float32)
        extra, remaining, done = _waterfill_step(
            extra, avail=avail, weights=active_weights, active=active, remaining=remaining
        )
        if done:
            return extra
    msg = "failed to converge while projecting belief distribution"
    raise ValueError(msg)


def _waterfill_step(
    extra: Tensor, *, avail: Tensor, weights: Tensor, active: Tensor, remaining: float
) -> tuple[Tensor, float, bool]:
    total_w = float(weights[active].sum().item())
    if total_w <= 0.0:
        msg = "cannot distribute remaining probability mass across zero-capacity cards"
        raise ValueError(msg)
    proposal = remaining * (weights[active] / total_w)
    active_avail = avail[active]
    saturated = proposal >= active_avail - 1e-9
    if not bool(saturated.any()):
        extra[active] = extra[active] + proposal
        return extra, 0.0, True
    active_indices = active.nonzero(as_tuple=False).squeeze(-1)
    sat_indices = active_indices[saturated]
    used = float(avail[sat_indices].sum().item())
    extra[sat_indices] = extra[sat_indices] + avail[sat_indices]
    return extra, max(0.0, remaining - used), False

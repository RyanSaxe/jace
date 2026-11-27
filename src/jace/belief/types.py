"""Pure-tensor data types for opponent belief states, format priors, and user overrides.

`FormatPrior` holds a format's Turn 0 deck prior, pairwise conditional copy
expectations, day-0 fallback odds, and optional live deck pool and per-card
copy limits. `BeliefState` stores an inferred unseen deck and hand distribution
across all legal format cards with unconstrained opponent hand and library
sizes. `BeliefOverride` records user-pinned hand copies, excluded cards, and
copy-count multipliers for counterfactual projection.
"""

from dataclasses import dataclass, field
from typing import Literal

from torch import Tensor

__all__ = (
    "BeliefOverride",
    "BeliefState",
    "FormatPrior",
    "PriorBackendName",
)

type PriorBackendName = Literal["bayesian_pool", "set_transformer"]

_MATRIX_NDIM = 2


@dataclass(frozen=True, slots=True)
class FormatPrior:
    format_id: str
    card_names: tuple[str, ...]
    mean_copies: Tensor
    cond_copies: Tensor
    booster_fallback: Tensor
    deck_pool: Tensor | None = None
    copy_limits: Tensor | None = None

    def __post_init__(self) -> None:
        num_cards = len(self.card_names)
        if num_cards == 0:
            msg = "FormatPrior requires at least one card name"
            raise ValueError(msg)
        _check_prior_shapes(self, num_cards)


@dataclass(frozen=True, slots=True)
class BeliefOverride:
    pinned_in_hand: dict[str, int] = field(default_factory=dict)
    excluded_cards: frozenset[str] = field(default_factory=frozenset)
    copy_multipliers: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BeliefState:
    card_names: tuple[str, ...]
    deck_copies: Tensor
    hand_probs: Tensor
    opp_hand_size: int
    opp_library_size: int
    prior_backend: PriorBackendName
    matchup_equity: float | None = None

    def __post_init__(self) -> None:
        num_cards = len(self.card_names)
        if num_cards == 0:
            msg = "BeliefState requires at least one card name"
            raise ValueError(msg)
        if self.deck_copies.shape != (num_cards,) or self.hand_probs.shape != (num_cards,):
            msg = f"deck_copies and hand_probs must have shape ({num_cards},)"
            raise ValueError(msg)
        if self.opp_hand_size < 0 or self.opp_library_size < 0:
            msg = "opp_hand_size and opp_library_size must be non-negative"
            raise ValueError(msg)


def _check_prior_shapes(prior: FormatPrior, num_cards: int) -> None:
    if prior.mean_copies.shape != (num_cards,):
        msg = f"mean_copies must have shape ({num_cards},), got {tuple(prior.mean_copies.shape)}"
        raise ValueError(msg)
    if prior.cond_copies.shape != (num_cards, num_cards):
        msg = f"cond_copies must have shape ({num_cards}, {num_cards}), got {tuple(prior.cond_copies.shape)}"
        raise ValueError(msg)
    if prior.booster_fallback.shape != (num_cards,):
        msg = f"booster_fallback must have shape ({num_cards},), got {tuple(prior.booster_fallback.shape)}"
        raise ValueError(msg)
    if prior.deck_pool is not None and (prior.deck_pool.ndim != _MATRIX_NDIM or prior.deck_pool.shape[1] != num_cards):
        msg = f"deck_pool must have shape (M, {num_cards}), got {tuple(prior.deck_pool.shape)}"
        raise ValueError(msg)
    if prior.copy_limits is not None and prior.copy_limits.shape != (num_cards,):
        msg = f"copy_limits must have shape ({num_cards},), got {tuple(prior.copy_limits.shape)}"
        raise ValueError(msg)

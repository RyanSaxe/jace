"""Open boosters from a spec that gives each card slot its rarity odds.

Booster contents differ by card set, so the slots are data the caller passes
and `open_booster` assumes no set. A booster never holds the same card twice.
Each slot draws a rarity from its odds and then a card of that rarity that
the booster does not hold yet, with every such card equally likely. Real
boosters can repeat a card between a foil slot and another slot, but the
drafter model reads a pack as one 0/1 entry per card and could not see the
second copy.
"""

import math
import random
from collections.abc import Mapping, Sequence

from jace.cards import CardId, CardPool, Rarity
from jace.draft.pod import Pack

__all__ = ("TLA_PLAY_BOOSTER", "BoosterSpec", "BoosterSpecError", "SlotOdds", "open_booster")

# One card slot of a booster: the chance of each rarity. The chances sum to 1.
type SlotOdds = Mapping[Rarity, float]
type BoosterSpec = Sequence[SlotOdds]

_COMMON: SlotOdds = {Rarity.COMMON: 1.0}
_UNCOMMON: SlotOdds = {Rarity.UNCOMMON: 1.0}

# One rare slot in eight holds a mythic instead.
_MYTHIC_CHANCE = 1 / 8
_RARE_OR_MYTHIC: SlotOdds = {Rarity.RARE: 1 - _MYTHIC_CHANCE, Rarity.MYTHIC: _MYTHIC_CHANCE}

# Placeholder odds for the wildcard and foil slots, to be replaced with sourced odds.
_PLACEHOLDER_COMMON_CHANCE = 0.6
_PLACEHOLDER_UNCOMMON_CHANCE = 0.3
_PLACEHOLDER_RARE_CHANCE = 0.0875
_PLACEHOLDER_MYTHIC_CHANCE = 0.0125
_ANY_RARITY: SlotOdds = {
    Rarity.COMMON: _PLACEHOLDER_COMMON_CHANCE,
    Rarity.UNCOMMON: _PLACEHOLDER_UNCOMMON_CHANCE,
    Rarity.RARE: _PLACEHOLDER_RARE_CHANCE,
    Rarity.MYTHIC: _PLACEHOLDER_MYTHIC_CHANCE,
}

# The land slot holds a basic land or a common land, each half the time.
_BASIC_LAND_CHANCE = 0.5
_LAND: SlotOdds = {Rarity.BASIC: _BASIC_LAND_CHANCE, Rarity.COMMON: 1 - _BASIC_LAND_CHANCE}

# A first, simplified Avatar: The Last Airbender play booster: 7 commons, 3 uncommons,
# 1 rare or mythic, 1 wildcard, 1 foil and 1 land.
_COMMON_SLOTS = 7
_UNCOMMON_SLOTS = 3
TLA_PLAY_BOOSTER: BoosterSpec = (
    *[_COMMON] * _COMMON_SLOTS,
    *[_UNCOMMON] * _UNCOMMON_SLOTS,
    _RARE_OR_MYTHIC,
    _ANY_RARITY,
    _ANY_RARITY,
    _LAND,
)


class BoosterSpecError(ValueError):
    """A booster spec cannot be opened from the pool it was given.

    Either one slot's odds do not sum to 1, or the pool holds fewer cards of
    a rarity than the spec has slots that can draw it. The second check does
    not depend on the random draws, so a spec that could run out of cards
    fails on the first booster and not on a rare one.
    """


def open_booster(spec: BoosterSpec, pool: CardPool, *, rng: random.Random) -> Pack:
    """Draw one card for each slot of `spec`, in slot order.

    No card appears twice in the booster. Raises `BoosterSpecError` when a
    slot's odds do not sum to 1, or when the pool holds fewer cards of a
    rarity than the number of slots that can draw that rarity. A pool with no
    cards of a slot's rarity is the plainest case of the second.
    """
    available = {rarity: list(pool.of_rarity(rarity)) for rarity in Rarity}
    _check_spec(spec, available)
    booster: list[CardId] = []
    for odds in spec:
        (rarity,) = rng.choices(tuple(odds), weights=tuple(odds.values()))
        cards = available[rarity]
        booster.append(cards.pop(rng.randrange(len(cards))))
    return tuple(booster)


def _check_spec(spec: BoosterSpec, available: Mapping[Rarity, Sequence[CardId]]) -> None:
    for slot, odds in enumerate(spec):
        if not math.isclose(sum(odds.values()), 1.0):
            message = f"the odds of slot {slot} sum to {sum(odds.values())}, not 1: {dict(odds)}"
            raise BoosterSpecError(message)
    for rarity, cards in available.items():
        slots = sum(1 for odds in spec if odds.get(rarity, 0.0) > 0.0)
        if slots > len(cards):
            message = f"{slots} slots can draw a {rarity.value} card, but the pool holds {len(cards)}"
            raise BoosterSpecError(message)

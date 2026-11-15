"""Open boosters from a spec that gives each card slot its rarity odds.

Booster contents differ by card set, so the caller passes the slots as
data, and `open_booster` assumes no set. For each slot, `open_booster` draws
a rarity from the slot's odds. It then draws a card of that rarity that the
booster does not contain yet, with every such card equally likely.

A booster opened this way never contains the same card twice. Real boosters
can repeat a card between a foil slot and another slot, but the drafter
model reads a pack as one 0/1 entry per card and could not see the second
copy.

`copies_per_booster` gives how often each card appears in a booster opened
this way. Before anyone has seen a deck of the format, these rates are all
that is known about which cards its decks contain.
"""

import math
import random
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from jace.cards import CardId, Rarity
from jace.draft.pod import Pack

__all__ = ("TLA_PLAY_BOOSTER", "BoosterSpec", "BoosterSpecError", "SlotOdds", "copies_per_booster", "open_booster")

# One card slot of a booster, given as the chance of each rarity. The chances sum to 1.
type SlotOdds = Mapping[Rarity, float]
type BoosterSpec = Sequence[SlotOdds]

_COMMON: SlotOdds = {Rarity.COMMON: 1.0}
_UNCOMMON: SlotOdds = {Rarity.UNCOMMON: 1.0}

# One rare slot in eight gets a mythic instead.
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

# The land slot gets a basic land half the time and any common card the other half. In the printed booster, the
# other half is a common land. These odds cannot express that, because `open_booster` fills a slot by rarity alone.
_BASIC_LAND_CHANCE = 0.5
_LAND: SlotOdds = {Rarity.BASIC: _BASIC_LAND_CHANCE, Rarity.COMMON: 1 - _BASIC_LAND_CHANCE}

# A first, simplified version of the Avatar: The Last Airbender play booster. It has 7 commons, 3 uncommons,
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
    """A booster spec cannot be opened from the cards it was given.

    Either one slot's odds do not sum to 1, or the cards given include fewer
    cards of a rarity than the spec has slots that can draw that rarity. The
    second check does not depend on the random draws, so a spec that could
    run out of cards fails on the first booster, not only on an unlucky one.
    """


def open_booster(spec: BoosterSpec, cards: Mapping[Rarity, Sequence[CardId]], *, rng: random.Random) -> Pack:
    """Draw one card for each slot of `spec`, in slot order.

    `cards` lists, by rarity, the cards a booster may contain. It can leave
    out cards of the pool, such as ones the game engine cannot play. No card
    appears twice in the booster.

    Raises `BoosterSpecError` when a slot's odds do not sum to 1, or when
    `cards` lists fewer cards of a rarity than the number of slots that can
    draw that rarity.
    """
    available = {rarity: list(cards.get(rarity, ())) for rarity in Rarity}
    _check_spec(spec, available)
    booster: list[CardId] = []
    for odds in spec:
        (rarity,) = rng.choices(tuple(odds), weights=tuple(odds.values()))
        left = available[rarity]
        booster.append(left.pop(rng.randrange(len(left))))
    return tuple(booster)


def _check_spec(spec: BoosterSpec, available: Mapping[Rarity, Sequence[CardId]]) -> None:
    for slot, odds in enumerate(spec):
        if not math.isclose(sum(odds.values()), 1.0):
            message = f"the odds of slot {slot} sum to {sum(odds.values())}, not 1: {dict(odds)}"
            raise BoosterSpecError(message)
    for rarity, cards in available.items():
        slots = sum(1 for odds in spec if odds.get(rarity, 0.0) > 0.0)
        if slots > len(cards):
            message = f"{slots} slots can draw a {rarity.value} card, but there are {len(cards)}"
            raise BoosterSpecError(message)


def copies_per_booster(spec: BoosterSpec, cards: Mapping[Rarity, Sequence[CardId]], *, pool_size: int) -> Tensor:
    """Give each card's expected copies in one booster that `open_booster` opens, as float `[pool_size]`.

    `spec` and `cards` are as `open_booster` takes them. Every listed card
    of one rarity gets the same expected copies. A card that `cards` does
    not list gets zero, and so does every card of a rarity that no slot can
    draw.
    """
    copies = torch.zeros(pool_size)
    for rarity, listed in cards.items():
        if listed:
            # A booster never repeats a card, but every listed card of a rarity is still as likely as any other to
            # fill a slot that `open_booster` gave that rarity. Each card therefore gets an equal share of the
            # rarity's chance, summed over the slots.
            copies[list(listed)] = sum(odds.get(rarity, 0.0) for odds in spec) / len(listed)
    return copies

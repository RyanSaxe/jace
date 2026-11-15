"""Hold the sizes of each limited format in one record, and turn them into deck rules.

A limited format fixes how a draft runs and what makes a deck legal. For
the draft, its record gives the seats in a pod, the boosters each seat
opens and the picks a seat takes from each. For a deck, it gives the
number of cards and how many of them may be lands. These numbers are rules
of the format, not anything a model learns, so every training entry point
looks its format up here by set code. A new format needs one more record
in `_FORMATS`.

The record is in `jace.loops` because it contains the draft's format
and booster spec, and only the loops may import `jace.draft`.

`deck_rules` turns a format's deck sizes into the `BuildRules` of one card
set, with that set's card ids for the basic and nonbasic lands.
`booster_meta` builds the format's day-0 meta. A meta contains each card's
mean copies in the format's decks and how often cards appear together in
them, and the guess model reads it. The day-0 meta is the one used before
any deck of the format has been seen, so it is computed from how often the
format's boosters contain each card.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from jace.cards import CardId, CardPool, Rarity
from jace.data import SetCards
from jace.decks import BuildRules
from jace.draft import TLA_PLAY_BOOSTER, BoosterSpec, DraftFormat, copies_per_booster
from jace.play import Meta, day_zero_meta

__all__ = (
    "TLA_LIMITED",
    "LimitedFormat",
    "UnknownFormatError",
    "booster_cards",
    "booster_meta",
    "deck_rules",
    "limited_format",
)


@dataclass(frozen=True, slots=True)
class LimitedFormat:
    """Every number that sizes one limited format.

    `draft` gives the number of seats in a pod, the number of rounds, and
    the picks a seat takes in each round. In every round, each seat opens
    one booster. `booster` gives the odds of every booster slot.

    A deck has exactly `deck_size` cards, and from `min_lands` to
    `max_lands` of them are lands. A seat's pool is every card it picked,
    so a pool has `picks` cards.
    """

    draft: DraftFormat
    booster: BoosterSpec
    deck_size: int
    min_lands: int
    max_lands: int

    @property
    def picks(self) -> int:
        return self.draft.rounds * self.draft.picks_per_round


# Eight seats each open three boosters, and the seats pick every card of them. A deck is 40 cards,
# and 99.8% of the human Avatar decks have 15 to 18 lands.
TLA_LIMITED = LimitedFormat(
    draft=DraftFormat(seats=8, rounds=3, picks_per_round=len(TLA_PLAY_BOOSTER)),
    booster=TLA_PLAY_BOOSTER,
    deck_size=40,
    min_lands=15,
    max_lands=18,
)

_FORMATS: Mapping[str, LimitedFormat] = {"TLA": TLA_LIMITED}


class UnknownFormatError(ValueError):
    def __init__(self, set_code: str) -> None:
        super().__init__(f"no limited format is defined for set {set_code}; add one to jace.loops.formats")


def limited_format(set_code: str) -> LimitedFormat:
    # The format of the set's limited play. Raises `UnknownFormatError` for a set with none.
    if set_code not in _FORMATS:
        raise UnknownFormatError(set_code)
    return _FORMATS[set_code]


def deck_rules(limited: LimitedFormat, cards: SetCards) -> BuildRules:
    # The format's deck rules for this card set. The basics are the set's cards of basic rarity, and the nonbasic
    # lands are every other card whose type line is a land.
    is_land = cards.printed[:, cards.printed_names.index("type_land")] > 0  # bool [cards]
    basics = cards.pool.of_rarity(Rarity.BASIC)
    return BuildRules(
        size=limited.deck_size,
        basics=basics,
        nonbasic_lands=frozenset(is_land.nonzero().squeeze(1).tolist()) - frozenset(basics),
        min_lands=limited.min_lands,
        max_lands=limited.max_lands,
    )


def booster_cards(pool: CardPool) -> dict[Rarity, list[CardId]]:
    # The cards a booster of the set may contain, by rarity. Only cards the engine can play are included, so that
    # every drafted deck can be played.
    return {rarity: [card for card in pool.of_rarity(rarity) if pool[card].program] for rarity in Rarity}


def booster_meta(limited: LimitedFormat, pool: CardPool) -> Meta:
    # The day-0 meta of the format on this card pool, from how often its boosters contain each card.
    copies = copies_per_booster(limited.booster, booster_cards(pool), pool_size=len(pool))
    return day_zero_meta(copies, deck_size=limited.deck_size)

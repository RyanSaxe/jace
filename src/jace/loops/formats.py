"""Hold the sizes of each limited format in one record, and turn them into deck rules.

A limited format fixes how a draft runs and what a legal deck is: how many
seats draft, how many boosters of which spec each opens, how many picks a
booster gives, how many cards a deck holds and how many of them are lands.
No model fixes any of these numbers, so every entry point looks its format
up here by set code and passes the numbers in. A new format is one more
record in `_FORMATS`. The record lives in `jace.loops` because it holds the
draft's format and booster spec, and only the loops may import `jace.draft`.
`deck_rules` turns a format's deck numbers into the `BuildRules` of one card
set, whose basics and nonbasic lands are that set's card ids.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from jace.cards import Rarity
from jace.data import SetCards
from jace.decks import BuildRules
from jace.draft import TLA_PLAY_BOOSTER, BoosterSpec, DraftFormat

__all__ = ("TLA_LIMITED", "LimitedFormat", "UnknownFormatError", "deck_rules", "limited_format")


@dataclass(frozen=True, slots=True)
class LimitedFormat:
    """Every number that sizes one limited format.

    `draft` gives the seats of a pod, the boosters each seat opens and the
    picks it takes from each, and `booster` the odds of every booster slot.
    A deck holds exactly `deck_size` cards, and from `min_lands` to
    `max_lands` of them are lands. A seat's pool is every card it picked,
    so it holds `picks` cards.
    """

    draft: DraftFormat
    booster: BoosterSpec
    deck_size: int
    min_lands: int
    max_lands: int

    @property
    def picks(self) -> int:
        return self.draft.rounds * self.draft.picks_per_round


# Eight seats each open three boosters and pick every card of them. A deck is 40 cards,
# and 99.8% of the human Avatar decks hold 15 to 18 lands.
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
    # The format's deck rules for this card set: its basic lands, and every other card whose type line is a land.
    is_land = cards.printed[:, cards.printed_names.index("type_land")] > 0  # bool [cards]
    basics = cards.pool.of_rarity(Rarity.BASIC)
    return BuildRules(
        size=limited.deck_size,
        basics=basics,
        nonbasic_lands=frozenset(is_land.nonzero().squeeze(1).tolist()) - frozenset(basics),
        min_lands=limited.min_lands,
        max_lands=limited.max_lands,
    )

"""Hold the sizes of each limited format in one record, and turn them into deck rules.

A limited format fixes how a draft runs and what makes a deck legal: pod seat
count, booster rounds and picks, deck size, and allowed land counts. Every
entry point looks up its format by set code in `_FORMATS`, and `deck_rules`
turns a format into `BuildRules` for one set's card pool.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from jace.cards import CardId, CardPool, Rarity
from jace.data import SetCards
from jace.decks import BuildRules
from jace.draft import TLA_PLAY_BOOSTER, BoosterSpec, DraftFormat

__all__ = (
    "TLA_LIMITED",
    "LimitedFormat",
    "UnknownFormatError",
    "booster_cards",
    "deck_rules",
    "limited_format",
)


@dataclass(frozen=True, slots=True)
class LimitedFormat:
    """Every number that sizes one limited format.

    `draft` gives the seats, rounds, and picks per round; `booster` gives the
    booster slot distribution; `deck_size`, `min_lands`, and `max_lands` give
    the deck construction bounds.
    """

    draft: DraftFormat
    booster: BoosterSpec
    deck_size: int
    min_lands: int
    max_lands: int

    @property
    def picks(self) -> int:
        return self.draft.rounds * self.draft.picks_per_round


# Eight seats each open three boosters; 99.8% of human TLA decks play 15 to 18 lands.
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
    if set_code not in _FORMATS:
        raise UnknownFormatError(set_code)
    return _FORMATS[set_code]


def deck_rules(limited: LimitedFormat, cards: SetCards) -> BuildRules:
    basics = cards.pool.of_rarity(Rarity.BASIC)
    return BuildRules(
        size=limited.deck_size,
        basics=basics,
        nonbasic_lands=frozenset(cards.is_land.nonzero().squeeze(1).tolist()) - frozenset(basics),
        min_lands=limited.min_lands,
        max_lands=limited.max_lands,
    )


def booster_cards(pool: CardPool) -> dict[Rarity, list[CardId]]:
    return {rarity: [card for card in pool.of_rarity(rarity) if pool[card].program] for rarity in Rarity}

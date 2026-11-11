"""Card records and the pool that gives every card its id.

A `CardId` is a card's position in a `CardPool`, and every tensor indexed by
card uses that order. The pool is fixed for the life of a run, so ids saved
in a checkpoint or a run record keep their meaning. A pool rejects two cards
with one name, because the engine identifies cards by printed name.
"""

import enum
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ("CardId", "CardPool", "CardRecord", "Deck", "DuplicateCardError", "Pool", "Rarity")

type CardId = int
type Pool = tuple[CardId, ...]
type Deck = tuple[CardId, ...]


class Rarity(enum.Enum):
    BASIC = "basic"
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    MYTHIC = "mythic"


@dataclass(frozen=True, slots=True)
class CardRecord:
    name: str
    set_code: str
    rarity: Rarity
    text: str
    # The engine's typed tokens for this card, empty when the engine cannot play it.
    program: tuple[int, ...]


class DuplicateCardError(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(f"two cards in one pool are named {name!r}")


class CardPool:
    """Every card a run may use, in the fixed order that defines `CardId`.

    Look cards up by id with `pool[card]` and by printed name with
    `card_id`. `of_rarity` lists the ids of one rarity in pool order, which
    is what booster generation samples from.
    """

    def __init__(self, records: Sequence[CardRecord]) -> None:
        self._records: tuple[CardRecord, ...] = tuple(records)
        self._ids: dict[str, CardId] = {}
        for card, record in enumerate(self._records):
            if record.name in self._ids:
                raise DuplicateCardError(record.name)
            self._ids[record.name] = card

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, card: CardId) -> CardRecord:
        return self._records[card]

    def card_id(self, name: str) -> CardId:
        return self._ids[name]

    def of_rarity(self, rarity: Rarity) -> tuple[CardId, ...]:
        return tuple(card for card, record in enumerate(self._records) if record.rarity is rarity)

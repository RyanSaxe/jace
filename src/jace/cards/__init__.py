"""Card data and the card vectors every model reads.

Every other module names cards by `CardId` and reads them through one
`CardVectors`: the `CardEncoder` while the drafter's pretraining trains it,
and the `CardTable` frozen from it everywhere after. Nothing here knows
about drafting, deck building or play, so those modules can change without
touching how a card is represented.
"""

from jace.cards.encoder import (
    FIRST_PROGRAM_TOKEN,
    MAX_PROGRAM_TOKENS,
    NO_PROGRAM_TOKEN,
    PADDING_TOKEN,
    UNKNOWN_TOKEN,
    CardEncoder,
    CardInputs,
    CardTable,
    CardVectors,
    frozen_table,
    own_parameters,
)
from jace.cards.pool import CardId, CardPool, CardRecord, Deck, DuplicateCardError, Pool, Rarity

__all__ = (
    "FIRST_PROGRAM_TOKEN",
    "MAX_PROGRAM_TOKENS",
    "NO_PROGRAM_TOKEN",
    "PADDING_TOKEN",
    "UNKNOWN_TOKEN",
    "CardEncoder",
    "CardId",
    "CardInputs",
    "CardPool",
    "CardRecord",
    "CardTable",
    "CardVectors",
    "Deck",
    "DuplicateCardError",
    "Pool",
    "Rarity",
    "frozen_table",
    "own_parameters",
)

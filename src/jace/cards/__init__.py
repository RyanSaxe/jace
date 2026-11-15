"""Card data and the card vectors every model reads.

Every other module refers to a card by its `CardId`, and every model gets
a card's vector from a `CardVectors`. The drafter's pretraining trains a
`CardEncoder` and reads the vectors from it. Every later model reads a
`CardTable`, which contains the vectors frozen from that encoder. This
package does not import the drafting, deck-building or play code, so that
code can change without changing how a card is represented.
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

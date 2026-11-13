"""Card data and the shared card embedding.

Every other module names cards by `CardId` and reads them through the one
`CardEmbedding`. Nothing here knows about drafting, deck building or play,
so those modules can change without touching how a card is represented.
"""

from jace.cards.embedding import CardEmbedding, own_parameters
from jace.cards.encoder import (
    FIRST_PROGRAM_TOKEN,
    MAX_PROGRAM_TOKENS,
    NO_PROGRAM_TOKEN,
    PADDING_TOKEN,
    UNKNOWN_TOKEN,
)
from jace.cards.pool import CardId, CardPool, CardRecord, Deck, DuplicateCardError, Pool, Rarity

__all__ = (
    "FIRST_PROGRAM_TOKEN",
    "MAX_PROGRAM_TOKENS",
    "NO_PROGRAM_TOKEN",
    "PADDING_TOKEN",
    "UNKNOWN_TOKEN",
    "CardEmbedding",
    "CardId",
    "CardPool",
    "CardRecord",
    "Deck",
    "DuplicateCardError",
    "Pool",
    "Rarity",
    "own_parameters",
)

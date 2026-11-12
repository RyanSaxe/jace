"""Human data: the real card pool, human drafts and human decks, as tensors.

Everything comes from the free 17Lands public datasets, read through the
`spells` package by `read_set`. The other functions take the tables it
returns, so they also work on a small hand-made table. Nothing here knows
about the drafter or the game engine: a loop in `jace.loops` passes these
tensors to a model, and passes in the engine's card programs by name.
"""

from jace.data.cards import SetCards, card_table
from jace.data.decks import HumanDecks, load_decks
from jace.data.drafts import HumanDrafts, PoolAtSecondPickError, load_drafts
from jace.data.seventeen_lands import SetData, SetNotDownloadedError, read_set
from jace.data.store import CardColumnsError, StaleCacheError

__all__ = (
    "CardColumnsError",
    "HumanDecks",
    "HumanDrafts",
    "PoolAtSecondPickError",
    "SetCards",
    "SetData",
    "SetNotDownloadedError",
    "StaleCacheError",
    "card_table",
    "load_decks",
    "load_drafts",
    "read_set",
)

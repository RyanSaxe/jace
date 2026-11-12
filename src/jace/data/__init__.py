"""Human data: the real card pool, human drafts and human decks, as tensors.

Everything comes from the free 17Lands public datasets, read through the
`spells` package by `read_set`, except the card vectors `text_vectors`
makes from printed text with a language model and the ones
`program_vectors` makes from the engine programs stored in the pool. The other functions take the tables it
returns, so they also work on a small hand-made table. Nothing here knows
about the drafter or the game engine: a loop in `jace.loops` passes these
tensors to a model, and passes in the engine's card programs by name.
"""

from jace.data.cards import SetCards, card_table
from jace.data.decks import HumanDecks, load_decks
from jace.data.drafts import HumanDrafts, PoolAtSecondPickError, load_drafts
from jace.data.programs import NoProgramsError, program_vectors
from jace.data.seventeen_lands import SetData, SetNotDownloadedError, read_set
from jace.data.store import CardColumnsError, StaleCacheError
from jace.data.text import TEXT_MODEL, TextEncoder, card_texts, hugging_face_encoder, text_vectors

__all__ = (
    "TEXT_MODEL",
    "CardColumnsError",
    "HumanDecks",
    "HumanDrafts",
    "NoProgramsError",
    "PoolAtSecondPickError",
    "SetCards",
    "SetData",
    "SetNotDownloadedError",
    "StaleCacheError",
    "TextEncoder",
    "card_table",
    "card_texts",
    "hugging_face_encoder",
    "load_decks",
    "load_drafts",
    "program_vectors",
    "read_set",
    "text_vectors",
)

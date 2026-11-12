"""Human data: the real card pool, human drafts and human decks, as tensors.

Everything comes from the free 17Lands public datasets, read through the
`spells` package by `read_set`, except the card vectors `text_vectors`
makes from printed text with a language model and the ones
`program_vectors` makes from the engine programs stored in the pool. The
other functions take the tables `read_set` returns, so they also work on a
small hand-made table. Nothing here knows about the drafter or the game
engine: a loop in `jace.loops` passes these tensors to a model, and passes
in the engine's card programs by name. The encoder that runs a Hugging Face
model lives in `jace.data.hugging_face` and is not imported here, so
importing this package does not load `transformers`.
"""

from jace.data.cards import SetCards, card_table
from jace.data.decks import HumanDecks, load_decks
from jace.data.drafts import RESULT_FLOOR, HumanDrafts, PoolAtSecondPickError, load_drafts
from jace.data.programs import NoProgramsError, program_vectors
from jace.data.seventeen_lands import SetData, SetNotDownloadedError, read_set
from jace.data.store import CardColumnsError, StaleCacheError
from jace.data.text import TEXT_MODEL, TextEncoder, card_texts, text_vectors

__all__ = (
    "RESULT_FLOOR",
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
    "load_decks",
    "load_drafts",
    "program_vectors",
    "read_set",
    "text_vectors",
)

"""Human data: the real card pool, human drafts and human decks, as tensors.

`read_set` reads one card set through the `spells` package. The card
attributes come from MTGJSON, and the card statistics, human picks and
games come from the free 17Lands public datasets. The other functions take
the tables `read_set` returns rather than a set code, so they also work on
a small table made by hand.

This package also makes the card encoder's inputs. `card_table` reads each
card's printed facts, `text_vectors` embeds each card's printed text with a
language model, and `program_tokens` turns each card's engine program into
token ids. This package does not import the drafter or the game engine.
Instead, a loop in `jace.loops` passes in the engine's card programs, keyed
by card name, and the engine's vocabulary. The loop then gives the
resulting tensors to a model.

The encoder that runs a Hugging Face model is in `jace.data.hugging_face`.
This package does not import that module, so importing `jace.data` does not
load `transformers`.
"""

from jace.data.cards import SetCards, card_table
from jace.data.decks import HumanDecks, load_decks
from jace.data.drafts import RESULT_FLOOR, HumanDrafts, PoolAtSecondPickError, load_drafts
from jace.data.programs import ProgramTooLongError, program_tokens
from jace.data.seventeen_lands import SetData, SetNotDownloadedError, read_set
from jace.data.store import CardColumnsError, StaleCacheError
from jace.data.text import TEXT_MODEL, TextEncoder, card_texts, text_vectors

__all__ = (
    "RESULT_FLOOR",
    "TEXT_MODEL",
    "CardColumnsError",
    "HumanDecks",
    "HumanDrafts",
    "PoolAtSecondPickError",
    "ProgramTooLongError",
    "SetCards",
    "SetData",
    "SetNotDownloadedError",
    "StaleCacheError",
    "TextEncoder",
    "card_table",
    "card_texts",
    "load_decks",
    "load_drafts",
    "program_tokens",
    "read_set",
    "text_vectors",
)

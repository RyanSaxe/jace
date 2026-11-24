"""Human data: the real card pool, human drafts and human decks, as tensors.

`read_set` reads one card set through the `spells` package. The card
attributes come from MTGJSON, and the card statistics, human picks and
games come from the free 17Lands public datasets. The other functions take
the tables `read_set` returns rather than a set code, so they also work on
a small table made by hand.

`program_tokens` turns each card's engine program into token ids for
`CardEncoder`. This package does not import the drafter or the game engine
directly; `jace.data.catalog` reads the engine's card catalog and vocabulary,
and an entry point in `jace.pretrain` passes the resulting tensors to a model.
"""

from jace.data.cards import SetCards, card_table
from jace.data.decks import HumanDecks, load_decks
from jace.data.drafts import RESULT_FLOOR, HumanDrafts, PoolAtSecondPickError, load_drafts
from jace.data.programs import ProgramTooLongError, program_tokens
from jace.data.seventeen_lands import SetData, SetNotDownloadedError, read_set
from jace.data.store import CardColumnsError, StaleCacheError

__all__ = (
    "RESULT_FLOOR",
    "CardColumnsError",
    "HumanDecks",
    "HumanDrafts",
    "PoolAtSecondPickError",
    "ProgramTooLongError",
    "SetCards",
    "SetData",
    "SetNotDownloadedError",
    "StaleCacheError",
    "card_table",
    "load_decks",
    "load_drafts",
    "program_tokens",
    "read_set",
)

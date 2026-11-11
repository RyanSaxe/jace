"""Games, behind an interface that hides the engine.

`GameBatch` and its records are all that the rest of the package uses. The
adapter that implements them on manabot lives in `jace.engine.manabot`, and
it is the only file in the package allowed to import manabot.
"""

from jace.engine.batch import SEATS, Board, Decisions, Finished, GameBatch, HiddenCards, Outcome

__all__ = ("SEATS", "Board", "Decisions", "Finished", "GameBatch", "HiddenCards", "Outcome")

"""Games, behind an interface that hides the engine.

`GameBatch` and its records are all that training uses. The adapter that
implements them on manabot lives in `jace.engine.manabot`. Scoring plays one
game at a time between any two players, in `jace.engine.scoring`. Those two
files are the only ones in the package allowed to import manabot.
"""

from jace.engine.batch import SEATS, Board, Decisions, Finished, GameBatch, HiddenCards, Outcome

__all__ = ("SEATS", "Board", "Decisions", "Finished", "GameBatch", "HiddenCards", "Outcome")

"""Games, behind an interface that hides the engine.

`GameBatch` and its records are all that training uses. The adapter in
`jace.engine.manabot` implements `GameBatch` on manabot's engine. For
scoring, `jace.engine.scoring` plays one game at a time between any two
players. Those two modules are the only ones in jace allowed to import
manabot or its engine, managym.
"""

from jace.engine.batch import (
    SEATS,
    Board,
    BoardLayout,
    Decisions,
    Finished,
    GameBatch,
    HiddenCards,
    Matchup,
    Outcome,
)

__all__ = (
    "SEATS",
    "Board",
    "BoardLayout",
    "Decisions",
    "Finished",
    "GameBatch",
    "HiddenCards",
    "Matchup",
    "Outcome",
)

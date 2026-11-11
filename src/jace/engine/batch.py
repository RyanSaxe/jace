"""The interface between a game engine and everything that plays games.

Training and evaluation step many games at once through `GameBatch` and never
see the engine behind it, so tests can hand them a small fake. A single game
is a batch of one. Each slot of a batch always holds a game that is waiting
on a decision: when a game ends, `act` reports it and the slot starts the
next matchup before `act` returns.
"""

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from jace.cards import CardId, Deck

__all__ = ("Board", "Decisions", "Finished", "GameBatch", "HiddenCards", "Outcome")

# One named array per kind of thing a seat can see, each with a leading batch axis.
type Board = Mapping[str, npt.NDArray[np.generic]]


class Outcome(enum.Enum):
    SEAT_0_WON = 0
    SEAT_1_WON = 1
    DRAW = 2


@dataclass(frozen=True, slots=True)
class Decisions:
    # seats: [n], the seat that must act in each game.
    seats: npt.NDArray[np.int64]
    # boards: each array [n, ...], only what the acting seat may see.
    boards: Board
    # action_counts: [n]; game i's legal actions are 0 .. action_counts[i] - 1.
    action_counts: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Finished:
    slot: int
    outcome: Outcome
    decks: tuple[Deck, Deck]


@dataclass(frozen=True, slots=True)
class HiddenCards:
    hand: tuple[CardId, ...]
    # The library's contents in no particular order.
    library: tuple[CardId, ...]


class GameBatch(Protocol):
    """A fixed number of games stepped together.

    `decisions` returns the pending decision in every slot. `act` applies
    one action per slot and returns the games that ended on that step, in
    slot order. `hidden_cards` returns both seats' true hidden cards for
    every slot, as `[slot][seat]`; it exists to label training data for the
    guess model and must never reach a player.
    """

    def decisions(self) -> Decisions: ...

    def act(self, actions: npt.NDArray[np.int64]) -> tuple[Finished, ...]: ...

    def hidden_cards(self) -> tuple[tuple[HiddenCards, HiddenCards], ...]: ...

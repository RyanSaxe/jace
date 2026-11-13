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

from jace.cards import Deck

__all__ = ("SEATS", "Board", "BoardLayout", "Decisions", "Finished", "GameBatch", "HiddenCards", "Outcome")

# Every game has two players, seat 0 and seat 1.
SEATS = 2

# One named array per kind of thing a seat can see, each with a leading batch axis.
type Board = Mapping[str, npt.NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class BoardLayout:
    """What a model must know about an engine's boards before it sees one.

    The first five fields are how many columns one row of each kind holds,
    which size the models' input layers. A player's and a card's count
    include the columns of their turn state when the engine reports one.
    `permanent_definition` is the column of a permanent's row that holds
    its card definition id plus one, with zero for none. An engine adapter
    builds this from what its engine reports, and `jace.play` takes it as
    plain data, so the models never import an engine.
    """

    player_features: int
    card_features: int
    permanent_features: int
    action_features: int
    event_features: int
    permanent_definition: int


class Outcome(enum.Enum):
    SEAT_0_WON = 0
    SEAT_1_WON = 1
    DRAW = 2
    # The engine could not continue the game, so it has no result and its decisions are discarded.
    ABANDONED = 3
    # The game ran past a time limit and its process was killed, so it has no result. Only code that
    # runs games under a time limit reports it; a `GameBatch` never does.
    HUNG = 4


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
    # hands: [n, 2, cards], copies of each card in each seat's hand.
    hands: npt.NDArray[np.int64]
    # libraries: [n, 2, cards], copies of each card in each seat's library, order unknown.
    libraries: npt.NDArray[np.int64]


class GameBatch(Protocol):
    """A fixed number of games stepped together.

    `decisions` returns the pending decision in every slot. `act` applies
    one action per slot and returns the games that ended on that step, in
    slot order. `hidden_cards` returns both seats' true hands and libraries
    in every slot as card counts; it exists to label training data for the
    guess model and must never reach a player.
    """

    def decisions(self) -> Decisions: ...

    def act(self, actions: npt.NDArray[np.int64]) -> tuple[Finished, ...]: ...

    def hidden_cards(self) -> HiddenCards: ...

"""The interface between a game engine and everything that plays games.

Training steps many games at once through `GameBatch` and never sees the
engine behind it, so tests can hand it a small fake. A batch is made for a
fixed list of matchups and plays exactly those games, a few at a time: when
a game ends, its place is taken by the next matchup of the list that has not
started, and when none is left the place stays empty. A game is known by its
position in the list, so nothing about a batch depends on a source of
matchups that it would have to ask.
"""

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from jace.cards import Deck

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

# Every game has two players, seat 0 and seat 1.
SEATS = 2

# The decks of one game: seat 0's, then seat 1's.
type Matchup = tuple[Deck, Deck]

# One named array per kind of thing a seat can see, each with a leading batch axis.
type Board = Mapping[str, npt.NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class BoardLayout:
    """What a model must know about an engine's boards before it sees one.

    Every column of a player's and a card's row is a number, so
    `player_features` and `card_features` are column counts, which include
    the columns of their turn state when the engine reports one. A
    permanent's, an action's and an event's row also hold columns that name
    something, and each of those is listed by what it names. A column in no
    list is not read. An engine adapter builds this from what its engine
    reports, and `jace.play` takes it as plain data, so the models never
    import an engine.
    """

    player_features: int
    card_features: int
    # The columns read as numbers: flags, counts, amounts, ranks and kinds.
    permanent_features: tuple[int, ...]
    action_features: tuple[int, ...]
    event_features: tuple[int, ...]
    # The column of a permanent's row that holds its card definition id plus one, zero for none.
    permanent_definition: int
    # The columns of a permanent's row that name another board object by its index in the order
    # that `action_focus` uses, plus one, zero for none: the object it is attached to and the
    # attacker it blocks.
    permanent_links: tuple[int, ...]
    # The columns that hold a small whole number, with -1 or 0 for none: an action's mode and
    # ability, and the seat that controlled an event.
    action_categories: tuple[int, ...]
    event_categories: tuple[int, ...]
    # The columns of a decklist row: the card definition id plus one, zero for none, then the
    # copies of that card not yet seen and the copies in the deck.
    decklist_definition: int
    decklist_copies: tuple[int, int]


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
    # One row for each game in play. n is zero once every game of the list is over.
    # games: [n], each game's position in the batch's list of matchups.
    games: npt.NDArray[np.int64]
    # seats: [n], the seat that must act in each game.
    seats: npt.NDArray[np.int64]
    # boards: each array [n, ...], only what the acting seat may see.
    boards: Board
    # action_counts: [n]; game i's legal actions are 0 .. action_counts[i] - 1.
    action_counts: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Finished:
    # game: the game's position in the batch's list of matchups.
    game: int
    outcome: Outcome
    decks: tuple[Deck, Deck]


@dataclass(frozen=True, slots=True)
class HiddenCards:
    # One row for each game in play, in the order of `Decisions`.
    # hands: [n, 2, cards], copies of each card in each seat's hand.
    hands: npt.NDArray[np.int64]
    # libraries: [n, 2, cards], copies of each card in each seat's library, order unknown.
    libraries: npt.NDArray[np.int64]


class GameBatch(Protocol):
    """A fixed list of games, stepped a few at a time until all are over.

    `decisions` returns the pending decision of every game in play. `act`
    applies one action to each of those games, in the same order, and
    returns the games that ended on that step. A game that ends makes room
    for the next one of the list, so the rows of the next `decisions` may
    belong to other games. `hidden_cards` returns both seats' true hands
    and libraries in every game in play as card counts; it exists to label
    training data for the guess model and must never reach a player.
    """

    def decisions(self) -> Decisions: ...

    def act(self, actions: npt.NDArray[np.int64]) -> tuple[Finished, ...]: ...

    def hidden_cards(self) -> HiddenCards: ...

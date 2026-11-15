"""The interface between a game engine and everything that plays games.

Training steps many games at once through `GameBatch` and never imports the
engine behind it. A test can therefore give training a small fake batch,
such as `jace.engine.toy.ToyBatch`, in place of the engine.

A batch is made for a fixed list of matchups and plays exactly those games,
a few at a time. When a game ends, the next matchup that has not started
takes its place. Once every matchup has started, the place of a game that
ends stays empty. `Decisions` and `Finished` identify a game by its position
in the list. The batch is given the whole list when it is made, and no matchup
is added later.
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
    """What a model is told about an engine's boards before it reads one.

    Every column of a player's row and of a card's row is a number, so
    `player_features` and `card_features` are simply column counts. Each
    count includes the row's turn-state columns when the engine reports
    them.

    The rows of permanents, actions and events also have columns that name
    something, such as a card definition or another object on the board. The
    layout lists those columns by what they name. A column that is in no
    list is not read.

    The guess model also counts the opponent's cards. For that count, the
    layout names a card's zone flags, a player's hand and library sizes, and
    the columns of the known-hand rows.

    An engine adapter builds the layout from its engine's description of
    the boards. `jace.play` takes the layout as plain data, so the models
    never import an engine.
    """

    player_features: int
    card_features: int
    # The columns of a permanent's, an action's and an event's row that are read as numbers: flags,
    # counts, amounts, ranks and kinds.
    permanent_features: tuple[int, ...]
    action_features: tuple[int, ...]
    event_features: tuple[int, ...]
    # The column of a permanent's row that contains its card definition id plus one, or zero for none.
    permanent_definition: int
    # The columns of a permanent's row that point to another object on the board: the object the
    # permanent is attached to, and the attacker it blocks. Each contains that object's index plus
    # one, or zero for none. The index counts objects in the order that `action_focus` uses.
    permanent_links: tuple[int, ...]
    # The columns that contain a small whole number standing for a category. In an action's row these
    # are its mode and its ability, with 0 for none. In an event's row it is the seat that controlled
    # the event, with -1 for none.
    action_categories: tuple[int, ...]
    event_categories: tuple[int, ...]
    # The columns of a decklist row. `decklist_definition` contains the card definition id plus one,
    # or zero for none. `decklist_copies` is a pair of columns: how many copies of that card are not
    # yet seen, then how many are in the deck.
    decklist_definition: int
    decklist_copies: tuple[int, int]
    # The flag columns of a card's row that mark the card as in its owner's library, in its owner's
    # hand, or as a token. A card with none of them set is a card of its owner's deck that both seats
    # can see, on the battlefield, in the graveyard, in exile or on the stack.
    card_unshown_flags: tuple[int, ...]
    # The columns of a player's row that contain the number of cards in the player's hand and in their
    # library. Each number is divided by `zone_count_scale`.
    hand_count: int
    library_count: int
    zone_count_scale: float
    # The columns of a known-hand row. `known_hand_definition` contains the card definition id plus
    # one, or zero for none. `known_hand_copies` contains how many copies of that card the player is
    # known to hold because they were revealed.
    known_hand_definition: int
    known_hand_copies: int


class Outcome(enum.Enum):
    SEAT_0_WON = 0
    SEAT_1_WON = 1
    DRAW = 2
    # The game has no result, because the engine either could not continue it or cut it short. Training
    # discards its decisions.
    ABANDONED = 3
    # The game has no result, because it ran past a time limit and its process was killed. Only
    # `jace.engine.scoring.play_games`, which runs games under a time limit, reports it. A
    # `GameBatch` never does.
    HUNG = 4


@dataclass(frozen=True, slots=True)
class Decisions:
    # Each array has one row for each game in play, n rows in all. n is zero once every game of the
    # list is over.
    # Each game's position in the batch's list of matchups, [n].
    games: npt.NDArray[np.int64]
    # The seat that must act in each game, [n].
    seats: npt.NDArray[np.int64]
    # Each game's boards as the acting seat sees them, each array [n, ...]. Nothing hidden from that
    # seat is included.
    boards: Board
    # How many legal actions each game has, [n]. Game i's legal actions are 0 to action_counts[i] - 1.
    action_counts: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Finished:
    # The game's position in the batch's list of matchups.
    game: int
    outcome: Outcome
    decks: tuple[Deck, Deck]


@dataclass(frozen=True, slots=True)
class HiddenCards:
    # Each array has one row for each game in play, in the same order as the rows of `Decisions`.
    # The copies of each card in each seat's hand, [n, 2, cards].
    hands: npt.NDArray[np.int64]
    # The copies of each card in each seat's library, [n, 2, cards]. The library's order is not given.
    libraries: npt.NDArray[np.int64]


class GameBatch(Protocol):
    """A fixed list of games, stepped a few at a time until all are over.

    `decisions` returns the pending decision of every game in play. Each
    call returns new arrays, and no later step changes them, so a caller may
    keep them without copying.

    `act` applies one action to each of those games, in the same order, and
    returns the games that ended on that step. When a game ends, the next
    game of the list takes its place, so the rows of the next `decisions`
    may belong to different games.

    `hidden_cards` returns both seats' true hands and libraries in every
    game in play, as card counts. Training uses them as the guess model's
    targets and as an input to the win-chance model, which only training
    calls. They must never reach a player.
    """

    def decisions(self) -> Decisions: ...

    def act(self, actions: npt.NDArray[np.int64]) -> tuple[Finished, ...]: ...

    def hidden_cards(self) -> HiddenCards: ...

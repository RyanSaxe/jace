"""A tiny game that implements `GameBatch` without any engine.

In High Card each seat holds three cards with distinct values, picks one, and
the higher pick wins. Seat 0 picks first and seat 1 sees that pick. Equal
picks deal both seats again and the game goes on, so games differ in length
and the slots of a batch drift out of step, as real games do. It exists
so that collection, training and evaluation can be exercised in tests and
examples with a game whose best play is obvious: always pick the highest
card. It is not a model of Magic.
"""

import random
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from jace.cards import Deck
from jace.engine.batch import Decisions, Finished, HiddenCards, Outcome

__all__ = ("HAND_SIZE", "TOY_CARDS", "ToyBatch")

HAND_SIZE = 3
TOY_CARDS = 10


class ToyBatch:
    """`games` slots of High Card, each replaced by a fresh deal when it ends.

    A board has two arrays: `hand`, float32 `[n, 3]`, the acting seat's card
    values scaled to 0..1, and `seen`, float32 `[n, 1]`, the value seat 0
    picked, or -1 while seat 0 has not picked. A card's id is its value,
    0 to 9. `matchups` supplies the decks reported with each finished game;
    the toy does not use them to play.
    """

    def __init__(self, matchups: Callable[[], tuple[Deck, Deck]], *, games: int, seed: int) -> None:
        self._matchups: Callable[[], tuple[Deck, Deck]] = matchups
        self._rng: random.Random = random.Random(seed)
        self._hands: npt.NDArray[np.int64] = np.zeros((games, 2, HAND_SIZE), dtype=np.int64)
        self._first_pick: npt.NDArray[np.int64] = np.full(games, -1, dtype=np.int64)
        self._decks: list[tuple[Deck, Deck]] = []
        for slot in range(games):
            self._decks.append(matchups())
            self._deal(slot)

    def decisions(self) -> Decisions:
        seats = (self._first_pick >= 0).astype(np.int64)
        slots = np.arange(len(seats))
        hands = self._hands[slots, seats].astype(np.float32) / (TOY_CARDS - 1)
        seen = np.where(self._first_pick >= 0, self._first_pick / (TOY_CARDS - 1), -1.0).astype(np.float32)
        return Decisions(
            seats=seats,
            boards={"hand": hands, "seen": seen[:, None]},
            action_counts=np.full(len(seats), HAND_SIZE, dtype=np.int64),
        )

    def act(self, actions: npt.NDArray[np.int64]) -> tuple[Finished, ...]:
        finished: list[Finished] = []
        for slot, action in enumerate(actions.tolist()):
            if self._first_pick[slot] < 0:
                self._first_pick[slot] = self._hands[slot, 0, action]
                continue
            first, second = int(self._first_pick[slot]), int(self._hands[slot, 1, action])
            if first == second:
                self._deal(slot)
                continue
            outcome = Outcome.SEAT_0_WON if first > second else Outcome.SEAT_1_WON
            finished.append(Finished(slot=slot, outcome=outcome, decks=self._decks[slot]))
            self._decks[slot] = self._matchups()
            self._deal(slot)
        return tuple(finished)

    def hidden_cards(self) -> HiddenCards:
        hands = np.zeros((len(self._hands), 2, TOY_CARDS), dtype=np.int64)
        np.put_along_axis(hands, self._hands, 1, axis=2)
        return HiddenCards(hands=hands, libraries=np.zeros_like(hands))

    def _deal(self, slot: int) -> None:
        self._hands[slot, 0] = self._rng.sample(range(TOY_CARDS), HAND_SIZE)
        self._hands[slot, 1] = self._rng.sample(range(TOY_CARDS), HAND_SIZE)
        self._first_pick[slot] = -1

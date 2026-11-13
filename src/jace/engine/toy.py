"""A tiny game that implements `GameBatch` without any engine.

In High Card each seat holds three cards with distinct values, picks one, and
the higher pick wins. Seat 0 picks first and seat 1 sees that pick. Equal
picks deal both seats again and the game goes on, so games differ in length
and the games of a batch drift out of step, as real games do. It exists
so that collection, training and evaluation can be exercised in tests and
examples with a game whose best play is obvious: always pick the highest
card. It is not a model of Magic.
"""

import random
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from jace.engine.batch import SEATS, Decisions, Finished, HiddenCards, Matchup, Outcome

__all__ = ("HAND_SIZE", "TOY_CARDS", "ToyBatch")

HAND_SIZE = 3
TOY_CARDS = 10


class ToyBatch:
    """The games of `matchups` at High Card, at most `games` of them at once.

    A board has two arrays: `hand`, float32 `[n, 3]`, the acting seat's card
    values scaled to 0..1, and `seen`, float32 `[n, 1]`, the value seat 0
    picked, or -1 while seat 0 has not picked. A card's id is its value,
    0 to 9. The decks of a matchup are reported with its finished game; the
    toy does not use them to play.
    """

    def __init__(self, matchups: Sequence[Matchup], *, games: int, seed: int) -> None:
        self._matchups: Sequence[Matchup] = matchups
        self._rng: random.Random = random.Random(seed)
        slots = min(games, len(matchups))
        # game: int64 [slots], the position in `matchups` of the game each slot plays, or -1 for none.
        self._game: npt.NDArray[np.int64] = np.arange(slots, dtype=np.int64)
        self._started: int = slots
        self._hands: npt.NDArray[np.int64] = np.zeros((slots, SEATS, HAND_SIZE), dtype=np.int64)
        self._first_pick: npt.NDArray[np.int64] = np.full(slots, -1, dtype=np.int64)
        for slot in range(slots):
            self._deal(slot)

    def decisions(self) -> Decisions:
        playing = np.flatnonzero(self._game >= 0)
        first_pick = self._first_pick[playing]
        seats = (first_pick >= 0).astype(np.int64)
        hands = self._hands[playing, seats].astype(np.float32) / (TOY_CARDS - 1)
        seen = np.where(first_pick >= 0, first_pick / (TOY_CARDS - 1), -1.0).astype(np.float32)
        return Decisions(
            games=self._game[playing],
            seats=seats,
            boards={"hand": hands, "seen": seen[:, None]},
            action_counts=np.full(len(seats), HAND_SIZE, dtype=np.int64),
        )

    def act(self, actions: npt.NDArray[np.int64]) -> tuple[Finished, ...]:
        finished: list[Finished] = []
        for slot, action in zip(np.flatnonzero(self._game >= 0).tolist(), actions.tolist(), strict=True):
            if self._first_pick[slot] < 0:
                self._first_pick[slot] = self._hands[slot, 0, action]
                continue
            first, second = int(self._first_pick[slot]), int(self._hands[slot, 1, action])
            if first == second:
                self._deal(slot)
                continue
            outcome = Outcome.SEAT_0_WON if first > second else Outcome.SEAT_1_WON
            game = int(self._game[slot])
            finished.append(Finished(game=game, outcome=outcome, decks=self._matchups[game]))
            self._game[slot] = self._started if self._started < len(self._matchups) else -1
            self._started += self._game[slot] >= 0
            self._deal(slot)
        return tuple(finished)

    def hidden_cards(self) -> HiddenCards:
        held = self._hands[self._game >= 0]
        hands = np.zeros((len(held), SEATS, TOY_CARDS), dtype=np.int64)
        np.put_along_axis(hands, held, 1, axis=2)
        return HiddenCards(hands=hands, libraries=np.zeros_like(hands))

    def _deal(self, slot: int) -> None:
        self._hands[slot, 0] = self._rng.sample(range(TOY_CARDS), HAND_SIZE)
        self._hands[slot, 1] = self._rng.sample(range(TOY_CARDS), HAND_SIZE)
        self._first_pick[slot] = -1

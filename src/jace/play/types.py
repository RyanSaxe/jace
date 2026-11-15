"""The types that connect models to the code that plays and trains.

The code that collects games, trains and evaluates takes a policy and a
guesser as plain functions over batched tensors, and training also takes a
win chance that way. That code therefore works with any model, and with
the toy engine in tests. The real models in this package satisfy these
types, and so does a three-line function written in a test. Every tensor
here has a leading batch axis, with one row for each pending decision.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from torch import Tensor

__all__ = ("Boards", "Earlier", "Guess", "Guesser", "Policy", "WinChance")

# The engine's named arrays as tensors, each [n, ...].
type Boards = Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class Guess:
    # deck: [..., cards], each card's share of the opponent's unseen cards. Rows sum to 1, or are all
    # zero when the opponent has no unseen card.
    deck: Tensor
    # hand: [..., cards], the chance the opponent holds at least one copy of each card.
    hand: Tensor


@dataclass(frozen=True, slots=True)
class Earlier:
    # guesses: the same seat's earlier guesses in this game, newest first, each [n, t, cards].
    guesses: Guess
    # mask: bool [n, t], False where the game had fewer earlier decisions. t may be 0.
    mask: Tensor


# Given the boards and a guess, a policy scores every action slot of each board. It returns float
# [n, max_actions] logits, with -inf for each illegal action.
type Policy = Callable[[Boards, Guess], Tensor]

# A guesser makes a new guess from the boards and the same seat's earlier guesses in this game. Each
# part of the new guess is float [n, cards].
type Guesser = Callable[[Boards, Earlier], Guess]

# A win chance estimates the acting seat's chance to win, float [n], from the boards and the
# opponent's true hidden cards. The hidden cards come as a `Guess`: its `deck` is each card's share
# of the opponent's hand and library, and its `hand` is 1 where the opponent holds the card. Only
# training calls a win chance, so the hidden cards never reach a player.
type WinChance = Callable[[Boards, Guess], Tensor]

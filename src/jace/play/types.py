"""The types that connect models to the code that plays and trains.

Collection, training and evaluation take a policy and a guesser as plain
functions over batched tensors, so they work with any model and with the toy
engine in tests. The real models in this package satisfy these types, and so
does a three-line function written in a test. Every tensor here has a leading
batch axis, one row per pending decision.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from torch import Tensor

__all__ = ("Boards", "Guess", "Guesser", "Policy")

# The engine's named arrays as tensors, each [n, ...].
type Boards = Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class Guess:
    # deck: [..., cards], each card's share of the opponent's unseen cards. Rows sum to 1.
    deck: Tensor
    # hand: [..., cards], the chance the opponent holds at least one copy of each card.
    hand: Tensor


# (boards, guess) -> (action logits [n, max_actions] with illegal actions at -inf, win chance [n]).
type Policy = Callable[[Boards, Guess], tuple[Tensor, Tensor]]

# (boards, earlier guesses [n, t, cards], earlier mask [n, t] bool) -> the new guess [n, cards].
type Guesser = Callable[[Boards, Guess, Tensor], Guess]

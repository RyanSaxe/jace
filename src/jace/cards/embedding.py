"""The one card embedding that every model reads.

Each card starts from a fixed vector. The loops build it from 17Lands play
statistics and printed facts, and the drafter's pretraining can also use the
card's rules text and engine program. Training never changes that vector. It
changes a small head shared by all cards, and optionally a bounded offset per
card, so a card from a new set gets an embedding from its starting vector
alone, as long as that vector holds nothing a new card lacks. The head's last layer
starts at zero, which makes the embedding equal the starting vector before
any training.
"""

from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional

__all__ = ("CardEmbedding", "own_parameters")


class CardEmbedding(nn.Module):
    """Compute `e = s + g(s) [+ o]` for card ids.

    `s` is the frozen starting vector, `g` is one head shared by every card,
    and `o` is a per-card offset that exists only when `offset_bound` is set.
    The offset starts at zero and its length is clipped to `offset_bound`, so
    it can refine a known card without replacing what the head learned.
    """

    def __init__(self, starting: Tensor, *, hidden: int, offset_bound: float | None) -> None:
        super().__init__()
        cards, width = starting.shape
        self.starting: Tensor
        self.register_buffer("starting", starting.detach().clone())
        self.head: nn.Sequential = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))
        self.offset_bound: float | None = offset_bound
        self.offset: nn.Parameter | None = None if offset_bound is None else nn.Parameter(torch.zeros(cards, width))
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @override
    def forward(self, cards: Tensor) -> Tensor:
        # cards: int64 card ids of any shape [...]. Returns [..., width].
        starting = self.starting[cards]
        embedded = starting + self.head(starting)
        if self.offset is None or self.offset_bound is None:
            return embedded
        offset = functional.embedding(cards, self.offset)
        length = offset.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(self.offset_bound)
        # `length` is max(|offset|, bound), so an offset shorter than the bound is unchanged and a
        # longer one is scaled down to the bound.
        return embedded + offset * length.reciprocal() * self.offset_bound

    def pull(self) -> Tensor:
        # Scalar: the mean squared length of the head's change over all cards.
        # Added to a loss, it is the penalty that keeps the embedding near its start.
        change = self.head(self.starting)
        return change.pow(2).sum(dim=-1).mean()


def own_parameters(model: nn.Module, embedding: CardEmbedding) -> tuple[nn.Parameter, ...]:
    """Return the parameters of `model` that are not the shared embedding's.

    A model that reads cards holds the embedding as a submodule, so
    `model.parameters()` includes the embedding's parameters. The embedding
    is trained by its own optimizer, and a parameter held by two optimizers
    is stepped twice for one loss. Build a model's optimizer from this. The
    embedding is recognised by identity, so it may sit at any depth and
    under any attribute name.
    """
    shared = {id(parameter) for parameter in embedding.parameters()}
    return tuple(parameter for parameter in model.parameters() if id(parameter) not in shared)

"""Search a pool's decks by sampling many from the builder and scoring each once.

A search that starts from one deck and swaps one card at a time cannot leave
the colours it started in, and scoring every swap costs millions of deck
scores. So the search samples many whole decks for a pool in one batch of
the builder, at its own temperature, and lets the deck score rank them in
one call. A deck sampled twice is scored once. The builder only ever writes
decks that `BuildRules` allows, so every candidate is legal. Choosing which
of the ranked decks to play is the caller's: the training loop plays several
of them, one in each block of games.
"""

import torch
from torch import Tensor

from jace.cards import Pool
from jace.decks.builder import BuildRules, Builds, DeckBuilder, sample_builds
from jace.decks.score import Scorer

__all__ = ("build_decks",)


def build_decks(
    pool: Pool,
    *,
    builder: DeckBuilder,
    score: Scorer,
    rules: BuildRules,
    cards: int,
    samples: int,
    keep: int,
    spells_per_round: int,
    temperature: float,
    generator: torch.Generator,
) -> tuple[Builds, Tensor]:
    """Sample `samples` decks for `pool`, score the distinct ones once and return the best, best first.

    `cards` is the number of cards in the card pool, the width of every
    count vector, and the builder's device is where both models compute.
    `spells_per_round` and `temperature` are as `sample_builds` takes them.
    Returns the builds of at most `keep` distinct decks, as one `Builds`
    with a row per deck, and their scores, float `[k]`. A deck sampled more
    than once comes with the first build that wrote it.
    """
    # TODO: The builder reads only its own pool. When decks play opponents from their own pod, the cards
    # passed to them could change a build, such as a main-deck artifact answer after passing three artifact
    # bombs. That changes a build rarely, so the builder ignores draft history until in-pod play matters.
    device = builder.cards.embedding.device
    copies = torch.ones(len(pool))
    pool_counts = torch.zeros(cards).index_add_(0, torch.tensor(pool, dtype=torch.int64), copies).to(device)
    builds, _ = sample_builds(
        builder,
        pool_counts.expand(samples, -1),
        rules=rules,
        spells_per_round=spells_per_round,
        temperature=temperature,
        generator=generator,
    )
    # unique works on the CPU for every device, and returns which distinct deck each sample is.
    distinct, which = builds.decks.cpu().unique(dim=0, return_inverse=True)
    first = torch.full((len(distinct),), samples).scatter_reduce(0, which, torch.arange(samples), reduce="amin")
    with torch.no_grad():
        scores = score(distinct.to(device))
    best = scores.argsort(descending=True)[:keep]
    return builds.rows(first.to(device)[best]), scores[best]

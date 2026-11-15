"""Search a pool's decks by sampling many from the builder and scoring each once.

A search that starts from one deck and swaps one card at a time cannot leave
the colours it started in, and scoring every swap would cost millions of
deck scores. Instead, `build_decks` samples many whole decks for a pool in
one batch of the builder, at its own temperature, and the deck score ranks
them all in one call. A deck sampled twice is scored once. The builder only
ever writes decks that `BuildRules` allows, so every candidate is legal.

The caller chooses which of the ranked decks to play. The training loop
plays several of them, one in each block of games.
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

    `cards` is the number of cards in the card pool, which is the width of
    every count vector. The decks are sampled and scored on the builder's
    device. `spells_per_round` and `temperature` are as `sample_builds`
    takes them.

    Returns the builds of at most `keep` distinct decks, as one `Builds`
    with a row per deck, and their scores as float `[k]`. A deck sampled
    more than once comes with the build of its first sample.
    """
    # TODO: The builder reads only its own pool. Once a deck plays opponents from its own pod, the cards its
    # drafter passed them could change the best build, such as a main-deck artifact answer after passing three
    # artifact bombs. That rarely changes a build, so the builder reads no draft history until play within a pod
    # matters.
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
    # The decks are moved to the CPU for `unique`, which runs there whatever the builder's device. `which` gives
    # the distinct deck of each sample.
    distinct, which = builds.decks.cpu().unique(dim=0, return_inverse=True)
    first = torch.full((len(distinct),), samples).scatter_reduce(0, which, torch.arange(samples), reduce="amin")
    with torch.no_grad():
        scores = score(distinct.to(device))
    best = scores.argsort(descending=True)[:keep]
    return builds.rows(first.to(device)[best]), scores[best]

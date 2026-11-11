"""Propose whole decks for a pool, improve them by swaps and rank them.

A search that starts from one deck and swaps one card at a time cannot leave
the colours it started in, so the builder starts from many whole decks
instead. A proposer writes them with no colour rule, `improve` climbs from
each one against the deck score, and `build_decks` returns every distinct
result with its score, best first. Deck shape is rules, not learning: the
proposers and the swap search only ever make decks that `BuildRules` allows.
Decks travel as float count vectors `[cards]` until `build_decks` turns the
survivors into tuples of card ids.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from jace.cards import CardEmbedding, CardId, Deck, Pool
from jace.decks.score import CardSetEncoder, Scorer
from jace.modules import run

__all__ = (
    "BuildRules",
    "Candidate",
    "DeckBuilderModel",
    "InvalidBuildRulesError",
    "NoLegalDeckError",
    "Proposer",
    "build_decks",
    "improve",
    "model_proposer",
    "random_proposer",
)

# Takes the pool's float [cards] copy counts and a number of decks n.
# Returns float [n, cards] copy counts, one legal deck per row.
type Proposer = Callable[[Tensor, int], Tensor]

# Takes (available, partial), both float [n, cards], and returns float [n, cards] logits for the next card.
type _NextCard = Callable[[Tensor, Tensor], Tensor]


class InvalidBuildRulesError(ValueError):
    def __init__(self, *, size: int, min_lands: int, max_lands: int) -> None:
        super().__init__(f"need 0 <= min_lands <= max_lands <= size, got {min_lands}, {max_lands} and size {size}")


class NoLegalDeckError(ValueError):
    def __init__(self, *, fewest: int, most: int) -> None:
        super().__init__(f"the pool allows no legal deck: it needs at least {fewest} lands and can hold at most {most}")


@dataclass(frozen=True, slots=True)
class BuildRules:
    """What makes a deck legal: its size and how many lands it holds.

    A deck is exactly `size` cards. It may hold any number of copies of a
    card in `basics` and no more copies of any other card than the pool
    does. Its lands are its basics plus its copies of `nonbasic_lands`, and
    their total must lie in `[min_lands, max_lands]`. Basics always count as
    lands, so they are not repeated in `nonbasic_lands`.
    """

    size: int
    basics: tuple[CardId, ...]
    nonbasic_lands: frozenset[CardId]
    min_lands: int
    max_lands: int

    def __post_init__(self) -> None:
        if not 0 <= self.min_lands <= self.max_lands <= self.size:
            raise InvalidBuildRulesError(size=self.size, min_lands=self.min_lands, max_lands=self.max_lands)


@dataclass(frozen=True, slots=True)
class Candidate:
    deck: Deck
    score: float


class DeckBuilderModel(nn.Module):
    """Choose the next card of a deck that is written one card at a time.

    The model reads every card that is in the partial deck or still available
    as one token, and turns the set into a query vector. A card's logit is
    the dot product of the query with a projection of that card's embedding,
    so no parameter has a slot per card. It knows nothing about deck rules:
    the proposer masks the moves that would break them.
    """

    def __init__(self, embedding: CardEmbedding, *, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.cards: CardSetEncoder = CardSetEncoder(embedding, amounts=2, width=width, layers=layers, heads=heads)
        self.query: nn.Linear = nn.Linear(width, width)
        self.key: nn.Linear = nn.Linear(embedding.starting.shape[1], width)

    @override
    def forward(self, available: Tensor, partial: Tensor) -> Tensor:
        # available: float [b, cards], copies that can still be added.
        # partial: float [b, cards], copies already in the deck.
        # Returns float [b, cards] logits, -inf where no copy is available.
        query = run(self.query, run(self.cards, torch.stack((available, partial), dim=-1)))  # [b, width]
        every_card = torch.arange(available.shape[1], device=available.device)
        keys = run(self.key, run(self.cards.embedding, every_card))  # [cards, width]
        return (query @ keys.T).masked_fill(available <= 0, float("-inf"))


def model_proposer(model: DeckBuilderModel, *, rules: BuildRules, generator: torch.Generator) -> Proposer:
    def propose(pool_counts: Tensor, decks: int) -> Tensor:
        fewest, most = _land_range(pool_counts, rules=rules)
        lands = torch.full((decks,), fewest, device=pool_counts.device)
        with torch.no_grad():
            return _write_decks(
                functools.partial(run, model),
                pool_counts=pool_counts,
                min_lands=lands,
                max_lands=lands + (most - fewest),
                rules=rules,
                generator=generator,
            )

    return propose


def random_proposer(*, rules: BuildRules, generator: torch.Generator) -> Proposer:
    """Propose legal decks at random, with no model.

    Each deck draws its land count uniformly from the counts the pool allows
    and its spells as a uniform sample of the pool's spell copies. Each deck
    also draws its own preference for every land, uniform over all mixes, so
    the proposals range from one basic to an even split of all of them.
    """

    def propose(pool_counts: Tensor, decks: int) -> Tensor:
        fewest, most = _land_range(pool_counts, rules=rules)
        device = pool_counts.device
        lands = torch.randint(fewest, most + 1, (decks,), generator=generator, device=device)
        uniform = torch.rand((decks, pool_counts.shape[0]), generator=generator, device=device)
        # The log of an exponential draw: sampling by these logits splits the lands by a uniform Dirichlet draw.
        preferences = uniform.clamp_min(1e-20).log().neg().log()
        return _write_decks(
            lambda _available, _partial: preferences,
            pool_counts=pool_counts,
            min_lands=lands,
            max_lands=lands,
            rules=rules,
            generator=generator,
        )

    return propose


def improve(decks: Tensor, *, pool_counts: Tensor, score: Scorer, rules: BuildRules, rounds: int) -> Tensor:
    """Climb from each deck by the best one-card swap, for at most `rounds` rounds.

    A swap takes one copy out and puts one copy of another card in, keeping
    the deck legal. It may trade a spell for a basic or one basic for
    another, so land counts and colours can move. Each round scores every
    swap of every deck that is still improving in one call to `score`, and
    applies the best swap to each deck it beats. A deck with no better swap
    is finished, and the search stops when every deck is.

    `decks` is float `[n, cards]` of legal decks and is not modified. Returns
    float `[n, cards]`, row for row, each scoring at least as well as it did.
    """
    is_land, is_basic = _land_kinds(rules, cards=pool_counts.shape[0], device=pool_counts.device)
    # Only pool cards and basics can enter or leave a deck, so swaps are laid out over those m cards.
    movable = ((pool_counts > 0) | is_basic).nonzero().squeeze(1)  # int64 [m]
    land = is_land[movable].int()
    land_change = land.unsqueeze(0) - land.unsqueeze(1)  # int [m out, m in]
    different = ~torch.eye(movable.shape[0], dtype=torch.bool, device=movable.device)
    decks = decks.clone()
    with torch.no_grad():
        scores = score(decks)  # float [n]
        climbing = torch.arange(decks.shape[0], device=decks.device)  # rows that improved last round
        for _ in range(rounds):
            held = decks[climbing][:, movable]  # float [a, m]
            addable = (held < pool_counts[movable]) | is_basic[movable]
            lands = (held * land).sum(dim=1).long().view(-1, 1, 1) + land_change  # int64 [a, m, m]
            legal = (
                held.gt(0).unsqueeze(2)
                & addable.unsqueeze(1)
                & different
                & (lands >= rules.min_lands)
                & (lands <= rules.max_lands)
            )
            row, out, into = legal.nonzero().unbind(dim=1)  # int64 [k] each
            if row.shape[0] == 0:
                break
            swapped = decks[climbing[row]]  # float [k, cards]
            each = torch.arange(row.shape[0], device=decks.device)
            swapped[each, movable[out]] -= 1
            swapped[each, movable[into]] += 1
            swap_scores = torch.full(legal.shape, float("-inf"), device=decks.device)
            swap_scores[legal] = score(swapped).to(swap_scores.dtype)
            best, flat = swap_scores.flatten(1).max(dim=1)  # [a]
            better = best > scores[climbing]
            climbing = climbing[better]
            if climbing.shape[0] == 0:
                break
            chosen = flat[better]
            decks[climbing, movable[chosen // movable.shape[0]]] -= 1
            decks[climbing, movable[chosen % movable.shape[0]]] += 1
            scores[climbing] = best[better].to(scores.dtype)
    return decks


def build_decks(
    pool: Pool,
    *,
    cards: int,
    propose: Proposer,
    score: Scorer,
    rules: BuildRules,
    proposals: int,
    rounds: int,
) -> tuple[Candidate, ...]:
    """Propose decks for a pool, improve each and return the distinct results, best first.

    `cards` is the number of cards in the card pool, which is the width of
    every count vector. `propose` writes `proposals` decks, `improve` climbs
    from each for at most `rounds` rounds, and proposals that end at the same
    deck are returned once. Choosing which candidate to play is left to the
    caller. Raises `NoLegalDeckError` when the pool cannot make a legal deck.
    """
    pool_counts = torch.zeros(cards).index_add_(0, torch.tensor(pool, dtype=torch.int64), torch.ones(len(pool)))
    proposed = propose(pool_counts, proposals)
    improved = improve(proposed, pool_counts=pool_counts, score=score, rules=rules, rounds=rounds)
    distinct = improved.unique(dim=0)  # float [d, cards]
    with torch.no_grad():
        scores = score(distinct)
    ids = torch.arange(cards)
    return tuple(
        Candidate(deck=tuple(ids.repeat_interleave(distinct[index].long()).tolist()), score=float(scores[index]))
        for index in scores.argsort(descending=True).tolist()
    )


def _land_kinds(rules: BuildRules, *, cards: int, device: torch.device) -> tuple[Tensor, Tensor]:
    # Returns (is_land, is_basic), both bool [cards].
    is_basic = torch.zeros(cards, dtype=torch.bool, device=device)
    is_basic[list(rules.basics)] = True
    is_land = is_basic.clone()
    is_land[sorted(rules.nonbasic_lands)] = True
    return is_land, is_basic


def _land_range(pool_counts: Tensor, *, rules: BuildRules) -> tuple[int, int]:
    is_land, is_basic = _land_kinds(rules, cards=pool_counts.shape[0], device=pool_counts.device)
    spells = int(pool_counts[~is_land].sum())
    lands = rules.size if rules.basics else int(pool_counts[is_land & ~is_basic].sum())
    fewest = max(rules.min_lands, rules.size - spells)
    most = min(rules.max_lands, lands)
    if fewest > most:
        raise NoLegalDeckError(fewest=fewest, most=most)
    return fewest, most


def _write_decks(
    next_card: _NextCard,
    *,
    pool_counts: Tensor,
    min_lands: Tensor,
    max_lands: Tensor,
    rules: BuildRules,
    generator: torch.Generator,
) -> Tensor:
    # min_lands, max_lands: int64 [n], each deck's own land bounds, inside the range the pool allows.
    # Returns float [n, cards]. Every row is a legal deck because a step only offers moves that leave one reachable.
    is_land, is_basic = _land_kinds(rules, cards=pool_counts.shape[0], device=pool_counts.device)
    decks = torch.zeros((min_lands.shape[0], pool_counts.shape[0]), device=pool_counts.device)
    one = torch.ones((min_lands.shape[0], 1), device=pool_counts.device)
    for written in range(rules.size):
        available = (pool_counts - decks).clamp_min(0)
        available[:, is_basic] = rules.size - written
        lands = decks[:, is_land].sum(dim=1)
        may_add_land = lands < max_lands
        may_add_spell = written - lands < rules.size - min_lands
        allowed = torch.where(is_land, may_add_land.unsqueeze(1), may_add_spell.unsqueeze(1)) & (available > 0)
        logits = next_card(available, decks).masked_fill(~allowed, float("-inf"))
        choice = torch.multinomial(logits.softmax(dim=1), 1, generator=generator)  # int64 [n, 1]
        decks.scatter_add_(1, choice, one)
    return decks

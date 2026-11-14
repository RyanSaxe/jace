"""Sample whole decks from a pool in batches, and give each its log-probability.

The builder writes a deck in three kinds of choice. A land head picks the
land total from the range the rules allow. The spells and nonbasic lands are
then drawn in rounds: one call of the model scores every card of the pool
given what is already in the deck, a few copies are drawn from those scores
without replacement, and the next round scores the pool again, so a sixth
four-drop scores lower than the first. Rounds go on until the spell slots
are full. Last, a basics head splits the land slots the nonbasic lands left.

Every choice is masked by `BuildRules`, so every sampled deck is legal, and
every choice's probability is kept, so a deck's log-probability is the sum of
its land choice, the Plackett-Luce terms of its draws and its basics split.
`sample_builds` draws decks this way; `build_log_probs` scores recorded
builds with gradients, which is how both teacher forcing on human decks and
a policy-gradient step train the model. Each copy of a card carries its
card's score within a round, so drawing a card is weighed by its copies left.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional

from jace.cards import CardId, CardVectors
from jace.decks.score import CardSetEncoder

__all__ = (
    "BuildRules",
    "BuildScores",
    "Builds",
    "DeckBuilder",
    "InvalidBuildRulesError",
    "NoBasicLandsError",
    "build_log_probs",
    "builds_from_decks",
    "join_builds",
    "sample_builds",
)

# A card's token reads two amounts: its copies in the pool and its copies already in the deck.
_AMOUNTS_PER_CARD = 2
# The floor for a uniform draw before its log is taken, so a draw of exactly 0 does not become infinite.
_SMALLEST_UNIFORM = 1e-20
# The land head reads a land total's share of the deck with a sine and a cosine at each of these frequencies, in cycles
# over the whole deck. Totals one apart then differ in more than one small input, which lets the head prefer a total in
# the middle of the legal range; read through the share alone it learned odds that only rise with the total.
_LAND_FREQUENCIES = (1.0, 2.0, 4.0, 8.0, 16.0)


class InvalidBuildRulesError(ValueError):
    def __init__(self, *, size: int, min_lands: int, max_lands: int) -> None:
        super().__init__(f"need 0 <= min_lands <= max_lands <= size, got {min_lands}, {max_lands} and size {size}")


class NoBasicLandsError(ValueError):
    def __init__(self) -> None:
        super().__init__("deck rules need at least one basic land to fill the land slots other lands leave")


@dataclass(frozen=True, slots=True)
class BuildRules:
    """What makes a deck legal: its size and how many lands it holds.

    A deck is exactly `size` cards. It may hold any number of copies of a
    card in `basics`, which names at least one card, and no more copies of
    any other card than its pool does. Its lands are its basics plus its
    copies of `nonbasic_lands`, and their total lies in
    `[min_lands, max_lands]`. A pool with too few other cards, its spells,
    to fill a deck within that range plays every spell it has and lands in
    the slots left. Basics always count as lands, so they are not repeated
    in `nonbasic_lands`.
    """

    size: int
    basics: tuple[CardId, ...]
    nonbasic_lands: frozenset[CardId]
    min_lands: int
    max_lands: int

    def __post_init__(self) -> None:
        if not 0 <= self.min_lands <= self.max_lands <= self.size:
            raise InvalidBuildRulesError(size=self.size, min_lands=self.min_lands, max_lands=self.max_lands)
        if not self.basics:
            raise NoBasicLandsError


@dataclass(frozen=True, slots=True)
class BuildScores:
    # The builder's scores for each row's deck so far. cards: float [n, cards], a score for taking one more copy of
    # each card, -inf for a card not in the pool and for every basic. lands: float [n, size + 1], a score for each
    # land total from 0 to the deck size. basics: float [n, basics], a score for each card of `rules.basics`.
    cards: Tensor
    lands: Tensor
    basics: Tensor


@dataclass(frozen=True, slots=True)
class Builds:
    """Decks with every choice that wrote them, one deck per row.

    `pools` is float `[n, cards]`, each deck's pool, and `decks` is float
    `[n, cards]`, the deck. `lands` is int64 `[n]`, the land total chosen
    first. `order` is int64 `[n, size]`: the spells and nonbasic lands in the
    order they were taken, then -1. They are taken in rounds of
    `spells_per_round`, so the rounds are the consecutive chunks of `order`.
    `basics` is float `[n, basics]`, the copies of each card of
    `rules.basics`.
    """

    pools: Tensor
    decks: Tensor
    lands: Tensor
    order: Tensor
    basics: Tensor

    def rows(self, index: Tensor) -> "Builds":
        # The builds `index` selects, in its order.
        selected = {field.name: getattr(self, field.name)[index] for field in fields(Builds)}
        return Builds(**selected)


def join_builds(parts: Sequence[Builds]) -> Builds:
    joined = {field.name: torch.cat([getattr(part, field.name) for part in parts]) for field in fields(Builds)}
    return Builds(**joined)


class DeckBuilder(nn.Module):
    """Score the choices that write a deck from a pool.

    A `CardSetEncoder` reads every card of the pool that is not a basic as
    one token with two amounts, its copies in the pool and its copies
    already in the deck, and a card's score comes from its own token's
    encoding. The mean of the encodings summarises the pool and the deck so
    far. The land head scores each land total from that summary and the
    total's share of the deck, read with sines and cosines of that share.
    The basics head scores each basic by the dot product of a query from
    the summary with a projection of the basic's embedding. No parameter
    has a slot per card, and the deck size and the basics arrive with
    `rules` at every call, so nothing here fixes the size of a pool or a
    deck.
    """

    def __init__(self, embedding: CardVectors, *, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.cards: CardSetEncoder = CardSetEncoder(
            embedding, amounts=_AMOUNTS_PER_CARD, width=width, layers=layers, heads=heads
        )
        self.card_head: nn.Linear = nn.Linear(width, 1)
        land_inputs = width + 1 + 2 * len(_LAND_FREQUENCIES)
        self.land_head: nn.Sequential = nn.Sequential(nn.Linear(land_inputs, width), nn.GELU(), nn.Linear(width, 1))
        self.basic_query: nn.Linear = nn.Linear(width, width)
        self.basic_key: nn.Linear = nn.Linear(embedding.width, width)

    @override
    def forward(self, pools: Tensor, taken: Tensor, *, rules: BuildRules) -> BuildScores:
        # pools: float [n, cards], each pool's copies of every card. taken: float [n, cards], the copies already in
        # each deck, never a basic.
        basics = torch.tensor(rules.basics, device=pools.device)
        not_basics = pools.index_fill(1, basics, 0.0)
        encoded, cards, real = self.cards.encode(torch.stack((not_basics, taken), dim=-1))
        weights = real.unsqueeze(-1).to(encoded.dtype)
        summary = (encoded * weights).sum(dim=1) / weights.sum(dim=1)  # [n, width]
        token_scores = self.card_head(encoded).squeeze(-1).masked_fill(~real, float("-inf"))  # [n, longest]
        card_scores = torch.full_like(pools, float("-inf")).scatter(1, cards, token_scores)
        shares = torch.arange(rules.size + 1, device=pools.device).to(summary.dtype) / rules.size  # [size + 1]
        angles = 2 * math.pi * shares[:, None] * torch.tensor(_LAND_FREQUENCIES, device=pools.device)
        share_features = torch.cat((shares[:, None], angles.sin(), angles.cos()), dim=-1)  # [size + 1, 1 + 2 * freqs]
        land_inputs = torch.cat(
            (
                summary.unsqueeze(1).expand(-1, len(shares), -1),
                share_features.expand(len(summary), -1, -1),
            ),
            dim=-1,
        )
        basic_scores = self.basic_query(summary) @ self.basic_key(self.cards.embedding(basics)).T
        return BuildScores(cards=card_scores, lands=self.land_head(land_inputs).squeeze(-1), basics=basic_scores)


@dataclass(frozen=True, slots=True)
class _Kinds:
    # bool [cards] each: the basics, the nonbasic lands and every other card, the spells.
    basic: Tensor
    nonbasic: Tensor
    spell: Tensor


def _kinds(rules: BuildRules, *, cards: int, device: torch.device) -> _Kinds:
    basic = torch.zeros(cards, dtype=torch.bool, device=device)
    basic[list(rules.basics)] = True
    nonbasic = torch.zeros(cards, dtype=torch.bool, device=device)
    nonbasic[sorted(rules.nonbasic_lands)] = True
    return _Kinds(basic=basic, nonbasic=nonbasic, spell=~(basic | nonbasic))


def _land_log_probs(scores: Tensor, *, pools: Tensor, rules: BuildRules, kinds: _Kinds, temperature: float) -> Tensor:
    # scores: float [n, size + 1]. Returns the land head's log-probabilities over land totals, -inf outside each row's
    # range: the rules' range, raised to the slots its pool's spells cannot fill.
    spells = pools[:, kinds.spell].sum(dim=1)
    fewest = (rules.size - spells).clamp_min(rules.min_lands)
    most = (rules.size - spells).clamp_min(rules.max_lands)
    totals = torch.arange(rules.size + 1, device=pools.device)
    allowed = (totals >= fewest[:, None]) & (totals <= most[:, None])
    return (scores / temperature).masked_fill(~allowed, float("-inf")).log_softmax(dim=-1)


def _card_log_probs(
    scores: Tensor, *, pools: Tensor, taken: Tensor, lands: Tensor, rules: BuildRules, kinds: _Kinds, temperature: float
) -> Tensor:
    """Return the log-probability of taking one more copy of each card, given each deck so far.

    `scores`, `pools` and `taken` are float `[..., cards]` and `lands` is
    int64 `[...]`, the land total each deck chose. A spell may be taken
    while the deck has spell slots left, and a nonbasic land while its
    lands are fewer than its land total. Once the spell slots are full the
    deck is finished: its row is -inf everywhere. Each copy left carries its
    card's score, so a card's odds also weigh its copies left, which is
    Plackett-Luce over the copies.
    """
    copies_left = pools - taken
    spells_left = (rules.size - lands) - taken[..., kinds.spell].sum(dim=-1)
    land_room = lands - taken[..., kinds.nonbasic].sum(dim=-1)
    may_take = (copies_left > 0) & (spells_left > 0).unsqueeze(-1)
    may_take &= kinds.spell | (kinds.nonbasic & (land_room > 0).unsqueeze(-1))
    logits = (scores / temperature + copies_left.clamp_min(1.0).log()).masked_fill(~may_take, float("-inf"))
    # A finished row has nothing to take, so it is given finite logits before the softmax and masked after it.
    unfinished = may_take.any(dim=-1, keepdim=True)
    return logits.masked_fill(~unfinished, 0.0).log_softmax(dim=-1).masked_fill(~may_take, float("-inf"))


def _basics_log_prob(log_shares: Tensor, counts: Tensor) -> Tensor:
    # log_shares: float [n, basics], the head's log-probability of each basic. counts: float [n, basics].
    # Returns float [n]: the multinomial log-probability of the split, the slots drawn one basic at a time.
    slots = counts.sum(dim=-1)
    arrangements = torch.lgamma(slots + 1) - torch.lgamma(counts + 1).sum(dim=-1)
    return arrangements + (counts * log_shares).sum(dim=-1)


def _entropy(log_probs: Tensor) -> Tensor:
    # log_probs: float [..., options], -inf where an option is not allowed. Returns float [...], 0 for a row with none.
    return -(log_probs.exp() * log_probs.nan_to_num(neginf=0.0)).sum(dim=-1)


def _draw(log_probs: Tensor, *, generator: torch.Generator, greedy: bool) -> Tensor:
    # log_probs: float [..., options]. Returns int64 [...]: the likeliest option when greedy, and otherwise one drawn
    # with the Gumbel-max trick. The generator lives on the CPU, so the noise is drawn there and moved.
    if greedy:
        return log_probs.argmax(dim=-1)
    uniform = torch.rand(log_probs.shape, generator=generator).to(log_probs.device)
    return (log_probs - uniform.clamp_min(_SMALLEST_UNIFORM).log().neg().log()).argmax(dim=-1)


def _draw_basics(
    log_shares: Tensor, *, slots: Tensor, rules: BuildRules, generator: torch.Generator, greedy: bool
) -> Tensor:
    # log_shares: float [n, basics]. slots: int64 [n], the land slots left for basics. Returns float [n, basics]:
    # each slot drawn from the shares, or when greedy the split nearest slots * shares, by largest remainders.
    if greedy:
        exact = slots[:, None] * log_shares.exp()
        floor = exact.floor()
        left = slots - floor.sum(dim=-1).long()
        rank = (exact - floor).argsort(dim=-1, descending=True).argsort(dim=-1)
        return floor + (rank < left[:, None]).float()
    draws = _draw(log_shares.unsqueeze(1).expand(-1, rules.size, -1), generator=generator, greedy=False)  # [n, size]
    counted = torch.arange(rules.size, device=slots.device) < slots[:, None]
    return (functional.one_hot(draws, log_shares.shape[-1]) * counted.unsqueeze(-1)).sum(dim=1).float()


def _draw_cards(
    builder: DeckBuilder,
    pools: Tensor,
    *,
    lands: Tensor,
    first: Tensor,
    rules: BuildRules,
    kinds: _Kinds,
    spells_per_round: int,
    temperature: float,
    generator: torch.Generator,
    greedy: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    # first: float [n, cards], the card scores of the empty deck. Returns (taken float [n, cards], order int64
    # [n, size], log_probs float [n]): the spells and nonbasic lands drawn, their order and their log-probability.
    taken = torch.zeros_like(pools)
    order = torch.full((len(pools), rules.size), -1, dtype=torch.int64, device=pools.device)
    log_probs = torch.zeros(len(pools), device=pools.device)
    scores = first
    for step in range(rules.size):
        unfinished = (rules.size - lands) - taken[:, kinds.spell].sum(dim=1) > 0
        if not bool(unfinished.any()):
            break
        if step > 0 and step % spells_per_round == 0:
            scores = builder(pools, taken, rules=rules).cards
        step_log_probs = _card_log_probs(
            scores, pools=pools, taken=taken, lands=lands, rules=rules, kinds=kinds, temperature=temperature
        )
        choice = _draw(step_log_probs, generator=generator, greedy=greedy)
        chosen = step_log_probs.gather(1, choice[:, None]).squeeze(1)
        log_probs += torch.where(unfinished, chosen, 0.0)
        taken.scatter_add_(1, choice[:, None], unfinished[:, None].to(taken.dtype))
        order[:, step] = torch.where(unfinished, choice, -1)
    return taken, order, log_probs


def sample_builds(
    builder: DeckBuilder,
    pools: Tensor,
    *,
    rules: BuildRules,
    spells_per_round: int,
    temperature: float,
    generator: torch.Generator,
    greedy: bool = False,
) -> tuple[Builds, Tensor]:
    """Sample one deck from each row of `pools` and return the builds with their log-probabilities.

    `pools` is float `[n, cards]`; repeat one pool's row to sample many
    decks for it in one batch. Each round draws up to `spells_per_round`
    spells or nonbasic lands, and every logit is divided by `temperature`
    first. With `greedy` every choice is the likeliest one and the basics
    split is the one nearest the head's shares, so `generator` is not used.
    The log-probabilities, float `[n]`, are those of the tempered sampling,
    computed without gradients; `build_log_probs` gives the same numbers
    with them. Every deck obeys `rules`.
    """
    kinds = _kinds(rules, cards=pools.shape[1], device=pools.device)
    with torch.no_grad():
        first = builder(pools, torch.zeros_like(pools), rules=rules)
        land_log_probs = _land_log_probs(first.lands, pools=pools, rules=rules, kinds=kinds, temperature=temperature)
        lands = _draw(land_log_probs, generator=generator, greedy=greedy)
        taken, order, card_log_probs = _draw_cards(
            builder,
            pools,
            lands=lands,
            first=first.cards,
            rules=rules,
            kinds=kinds,
            spells_per_round=spells_per_round,
            temperature=temperature,
            generator=generator,
            greedy=greedy,
        )
        log_shares = (builder(pools, taken, rules=rules).basics / temperature).log_softmax(dim=-1)
        slots = lands - taken[:, kinds.nonbasic].sum(dim=1).long()
        basics = _draw_basics(log_shares, slots=slots, rules=rules, generator=generator, greedy=greedy)
        log_probs = land_log_probs.gather(1, lands[:, None]).squeeze(1) + card_log_probs
        log_probs += _basics_log_prob(log_shares, basics)
    decks = taken.index_add(1, torch.tensor(rules.basics, device=pools.device), basics)
    return Builds(pools=pools, decks=decks, lands=lands, order=order, basics=basics), log_probs


def build_log_probs(
    builder: DeckBuilder, builds: Builds, *, rules: BuildRules, spells_per_round: int, temperature: float
) -> tuple[Tensor, Tensor]:
    """Score recorded builds as `sample_builds` would have, with gradients.

    Returns `(log_probs, entropies)`, both float `[n]`. A build's
    log-probability is that of its land total, of each card in `order`
    given the cards before it, and of its basics split, under the sampling
    at `temperature` with rounds of `spells_per_round`. Its entropy is the
    sum of the entropies of the choices it made, with the basics split
    counted as one draw per slot. Every round of every build is encoded in
    one call of `builder`, because the deck at the start of each round is
    known from `order`.
    """
    count, cards = builds.pools.shape
    kinds = _kinds(rules, cards=cards, device=builds.pools.device)
    took = builds.order >= 0  # bool [n, size]
    steps = int(took.sum(dim=1).max())
    rounds = -(-steps // spells_per_round)
    added = functional.one_hot(builds.order.clamp_min(0), cards) * took.unsqueeze(-1)  # [n, size, cards]
    after = added.cumsum(dim=1).to(builds.pools.dtype)
    before = torch.cat((torch.zeros_like(after[:, :1]), after[:, :-1]), dim=1)  # [n, size, cards]
    # The deck at the start of every round, then the finished deck, which the basics head reads.
    states = torch.cat((before[:, : rounds * spells_per_round : spells_per_round], after[:, -1:]), dim=1)
    scores = builder(builds.pools.repeat_interleave(rounds + 1, dim=0), states.flatten(0, 1), rules=rules)
    round_of_step = torch.arange(steps, device=builds.pools.device) // spells_per_round
    step_log_probs = _card_log_probs(
        scores.cards.view(count, rounds + 1, cards)[:, round_of_step],
        pools=builds.pools.unsqueeze(1),
        taken=before[:, :steps],
        lands=builds.lands.unsqueeze(1),
        rules=rules,
        kinds=kinds,
        temperature=temperature,
    )  # [n, steps, cards]
    chosen = step_log_probs.gather(2, builds.order[:, :steps, None].clamp_min(0)).squeeze(2)
    land_log_probs = _land_log_probs(
        scores.lands.view(count, rounds + 1, -1)[:, 0],
        pools=builds.pools,
        rules=rules,
        kinds=kinds,
        temperature=temperature,
    )
    log_shares = (scores.basics.view(count, rounds + 1, -1)[:, -1] / temperature).log_softmax(dim=-1)
    log_probs = land_log_probs.gather(1, builds.lands[:, None]).squeeze(1) + _basics_log_prob(log_shares, builds.basics)
    log_probs = log_probs + torch.where(took[:, :steps], chosen, 0.0).sum(dim=1)
    entropies = _entropy(land_log_probs) + _entropy(step_log_probs).sum(dim=1)
    entropies = entropies + builds.basics.sum(dim=1) * _entropy(log_shares)
    return log_probs, entropies


def builds_from_decks(decks: Tensor, *, pools: Tensor, rules: BuildRules, generator: torch.Generator) -> Builds:
    """Lay finished decks out as the builds that would have written them, for teacher forcing.

    `decks` and `pools` are float `[n, cards]`, each deck with the pool it
    was built from, and the result is on their device. Each deck's spells
    and nonbasic lands are put in a random order with a spell last, since
    sampling stops once the spell slots are full. Its land total and its
    basics split are its own. `generator` draws the orders on the CPU.
    """
    kinds = _kinds(rules, cards=decks.shape[1], device=torch.device("cpu"))
    every_card = torch.arange(decks.shape[1])
    orders: list[Tensor] = []
    for deck in decks.cpu().long():
        drawn = every_card.repeat_interleave(deck * ~kinds.basic)
        keys = torch.rand(len(drawn), generator=generator)
        spell = kinds.spell[drawn]
        # Scaling every nonbasic land's key below the highest spell's makes that spell last and leaves the order
        # of the rest uniform: given their maximum, the other spells' keys are uniform below it too.
        keys = torch.where(spell, keys, keys * keys[spell].max())
        orders.append(functional.pad(drawn[keys.argsort()], (0, rules.size - len(drawn)), value=-1))
    lands = decks[:, ~kinds.spell.to(decks.device)].sum(dim=1).long()
    return Builds(
        pools=pools,
        decks=decks,
        lands=lands,
        order=torch.stack(orders).to(decks.device),
        basics=decks[:, list(rules.basics)],
    )

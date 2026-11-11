"""One iteration of limited training, written from the package's pieces.

This is an example loop, not a framework: it drafts pods, builds a deck from
every pool, plays the decks against each other and updates every model, in
that order, as plain calls. Constructed play would be another short function
that skips the first two steps. A user who wants a different order, another
reward or another explore rule edits this function or writes their own; the
pieces it calls do not change.
"""

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor
from torch.optim import Optimizer

from jace.cards import CardPool, Deck
from jace.decks import (
    BuildRules,
    Candidate,
    DeckBuilderModel,
    DeckScore,
    build_decks,
    model_proposer,
    writing_steps,
)
from jace.draft import (
    BoosterSpec,
    Drafter,
    DrafterModel,
    DraftFormat,
    DraftRecord,
    SeatView,
    model_drafter,
    open_booster,
    run_drafts,
    view_tensors,
)
from jace.engine import SEATS, GameBatch
from jace.loops.schedules import Dials
from jace.modules import run
from jace.play import Guesser, Policy
from jace.train import (
    GameResult,
    Metrics,
    PlayerRule,
    Rollout,
    apply,
    choice_loss,
    collect,
    entropy,
    kl_to,
    score_loss,
    update_guess,
    update_player,
)

__all__ = ("Learners", "LimitedSetup", "StartBatch", "limited_iteration")

# PyTorch seeds a generator with a signed 64-bit integer, so a seed drawn from `random` uses 63 bits.
_SEED_BITS = 63

# Starts a batch of games whose matchups come from the function it is given.
type StartBatch = Callable[[Callable[[], tuple[Deck, Deck]]], GameBatch]


@dataclass(frozen=True, slots=True)
class LimitedSetup:
    """Everything about a run that stays fixed from one iteration to the next.

    `pods` drafts are run per iteration, and each pool gets `proposals`
    candidate decks improved for `search_rounds` rounds. Games are played
    until finished games hold `decisions` learner decisions. `history` is
    how many earlier guesses the guess model is shown. `player_rule` is the
    player's update rule; its `spread` is replaced each iteration by the
    schedule's. `score_chunk` bounds how many decks the deck score rates in
    one call during the search.
    """

    pool: CardPool
    draft_format: DraftFormat
    booster: BoosterSpec
    rules: BuildRules
    pods: int
    proposals: int
    search_rounds: int
    score_chunk: int
    decisions: int
    history: int
    slice_rows: int
    max_grad_norm: float
    player_rule: PlayerRule
    lam_advantage: float
    lam_value: float


@dataclass(frozen=True, slots=True)
class Learners:
    """The models being trained and the optimizer that moves each.

    `policy` and `guesser` are the player's and the guess model's functions,
    which depend on the engine's board layout. `human_drafter` is the frozen
    drafter trained on human picks that `Dials.human_pull` pulls toward, or
    `None` for no pull. `optimizers` holds one optimizer per trained thing,
    keyed "player", "guess", "drafter", "deck_builder", "deck_score" and
    "embedding"; the embedding's steps together with whichever model is
    being updated.
    """

    policy: Policy
    guesser: Guesser
    drafter: DrafterModel
    human_drafter: DrafterModel | None
    builder: DeckBuilderModel
    score: DeckScore
    optimizers: Mapping[str, Optimizer]


def _set_rate(optimizer: Optimizer, rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = rate


def _draft(
    setup: LimitedSetup, *, drafter: Drafter, other_seats: Sequence[Drafter], rng: random.Random
) -> tuple[DraftRecord, ...]:
    # The learner's drafter fills every seat that `other_seats` does not.
    seats = setup.draft_format.seats
    drafters = [*([drafter] * (seats - len(other_seats))), *other_seats]
    boosters = [
        [
            [open_booster(setup.booster, setup.pool, rng=rng) for _ in range(seats)]
            for _ in range(setup.draft_format.rounds)
        ]
        for _ in range(setup.pods)
    ]
    return run_drafts(setup.draft_format, boosters, drafters=drafters)


def _build(
    setup: LimitedSetup, learners: Learners, *, records: Sequence[DraftRecord], dials: Dials, rng: random.Random
) -> tuple[list[Candidate], list[Candidate]]:
    # Returns (the deck played for each pool, the top-scored deck for each pool), pools in pod then seat order.
    generator = torch.Generator().manual_seed(rng.getrandbits(_SEED_BITS))
    propose = model_proposer(learners.builder, rules=setup.rules, generator=generator)

    def score(decks: Tensor) -> Tensor:
        parts = [run(learners.score, part) for part in decks.split(setup.score_chunk)]
        return torch.cat(parts)

    played: list[Candidate] = []
    best: list[Candidate] = []
    for record in records:
        for pool in record.pools:
            candidates = build_decks(
                pool,
                cards=len(setup.pool),
                propose=propose,
                score=score,
                rules=setup.rules,
                proposals=setup.proposals,
                rounds=setup.search_rounds,
            )
            explore = len(candidates) > 1 and rng.random() < dials.explore_share
            played.append(rng.choice(candidates[1:]) if explore else candidates[0])
            best.append(candidates[0])
    return played, best


def _counts(decks: Sequence[Deck], *, cards: int) -> Tensor:
    # Returns float [decks, cards], the copies of each card in each deck.
    counts = torch.zeros(len(decks), cards)
    for row, deck in enumerate(decks):
        counts[row].index_add_(0, torch.tensor(deck), torch.ones(len(deck)))
    return counts


def _deck_records(decks: Sequence[Deck], games: Sequence[GameResult]) -> tuple[Tensor, Tensor]:
    # Returns (wins [decks], games played [decks]); a draw counts as half a win for each deck.
    index = {deck: row for row, deck in enumerate(decks)}
    wins = torch.zeros(len(decks))
    played = torch.zeros(len(decks))
    for game in games:
        learner_deck, other_deck = game.decks[game.learner_seat], game.decks[1 - game.learner_seat]
        for deck, result in ((learner_deck, game.learner_result), (other_deck, 1.0 - game.learner_result)):
            if deck in index:
                wins[index[deck]] += result
                played[index[deck]] += 1
    return wins, played


def _update_drafter(
    setup: LimitedSetup,
    learners: Learners,
    *,
    records: Sequence[DraftRecord],
    learner_seats: int,
    win_rates: Tensor,
    played: Tensor,
    dials: Dials,
) -> Metrics:
    # win_rates, played: [pods * seats] in pod then seat order. Seats whose deck played no game teach nothing.
    seats = setup.draft_format.seats
    views: list[SeatView] = []
    chosen: list[int] = []
    rewards: list[float] = []
    for pod, record in enumerate(records):
        for seat in range(learner_seats):
            deck = pod * seats + seat
            if played[deck] == 0:
                continue
            views.extend(record.views[seat])
            chosen.extend(record.pools[seat])
            rewards.extend([float(win_rates[deck])] * len(record.views[seat]))
    if not views:
        nothing: dict[str, float] = {}
        return nothing
    reward = torch.tensor(rewards)
    seen, picks, pack = view_tensors(views, cards=len(setup.pool), device=reward.device)
    logits = learners.drafter.forward(seen, picks, pack=pack)
    loss = choice_loss(logits, torch.tensor(chosen), weights=reward - reward.mean())
    loss = loss - entropy(logits).mean() * dials.drafter_entropy
    if learners.human_drafter is not None and dials.human_pull > 0:
        with torch.no_grad():
            human = learners.human_drafter.forward(seen, picks, pack=pack)
        loss = loss + kl_to(logits, human).mean() * dials.human_pull
    apply(
        loss,
        [learners.optimizers["drafter"], learners.optimizers["embedding"]],
        max_grad_norm=setup.max_grad_norm,
    )
    return {"drafter/loss": float(loss.detach()), "drafter/mean_reward": float(reward.mean())}


def _update_play_models(setup: LimitedSetup, learners: Learners, *, rollout: Rollout, dials: Dials) -> Metrics:
    embedding = learners.optimizers["embedding"]
    player = update_player(
        learners.policy,
        rollout,
        rule=replace(setup.player_rule, spread=dials.spread),
        lam_advantage=setup.lam_advantage,
        lam_value=setup.lam_value,
        slice_rows=setup.slice_rows,
        max_grad_norm=setup.max_grad_norm,
        optimizers=[learners.optimizers["player"], embedding],
    )
    guess = update_guess(
        learners.guesser,
        rollout,
        history=setup.history,
        slice_rows=setup.slice_rows,
        max_grad_norm=setup.max_grad_norm,
        optimizers=[learners.optimizers["guess"], embedding],
    )
    named = {f"player/{name}": value for name, value in player.items()}
    return named | {f"guess/{name}": value for name, value in guess.items()}


def _update_deck_score(
    setup: LimitedSetup, learners: Learners, *, decks: Sequence[Deck], wins: Tensor, played: Tensor
) -> Metrics:
    # wins, played: [decks]. Decks that played no game teach nothing.
    seen_play = played > 0
    if not bool(seen_play.any()):
        nothing: dict[str, float] = {}
        return nothing
    logits = run(learners.score, _counts(decks, cards=len(setup.pool))[seen_play])
    loss = score_loss(logits, wins=wins[seen_play], games=played[seen_play])
    optimizers = [learners.optimizers["deck_score"], learners.optimizers["embedding"]]
    apply(loss, optimizers, max_grad_norm=setup.max_grad_norm)
    return {"deck_score/loss": float(loss.detach())}


def _update_builder(
    setup: LimitedSetup,
    learners: Learners,
    *,
    records: Sequence[DraftRecord],
    best_decks: Sequence[Candidate],
    generator: torch.Generator,
) -> Metrics:
    # Teaches the builder to write the top-scored deck the search found for each pool.
    cards = len(setup.pool)
    pools = _counts([pool for record in records for pool in record.pools], cards=cards)
    best = _counts([candidate.deck for candidate in best_decks], cards=cards)
    available, partial, chosen = writing_steps(best, pool_counts=pools, rules=setup.rules, generator=generator)
    loss = choice_loss(run(learners.builder, available, partial), chosen, weights=torch.ones(len(chosen)))
    optimizers = [learners.optimizers["deck_builder"], learners.optimizers["embedding"]]
    apply(loss, optimizers, max_grad_norm=setup.max_grad_norm)
    return {"deck_builder/loss": float(loss.detach())}


def limited_iteration(
    setup: LimitedSetup,
    learners: Learners,
    *,
    dials: Dials,
    opponent: Policy,
    other_seats: Sequence[Drafter],
    start: StartBatch,
    rng: random.Random,
) -> Metrics:
    """Draft, build, play and learn once, and return the metrics of every update.

    `dials` are this iteration's values from the schedule. `opponent` plays
    the seat the learner does not, and `other_seats` are the drafters for
    the last seats of every pod, such as past versions; the learner drafts
    the rest. `start` begins a batch of games on whatever engine the caller
    uses. Games still running when enough decisions are collected are
    discarded, which matters little while `setup.decisions` is many games'
    worth.
    """
    generator = torch.Generator().manual_seed(rng.getrandbits(_SEED_BITS))
    _set_rate(learners.optimizers["player"], dials.player_rate)
    _set_rate(learners.optimizers["drafter"], dials.drafter_rate)

    drafter = model_drafter(learners.drafter, generator=generator)
    records = _draft(setup, drafter=drafter, other_seats=other_seats, rng=rng)
    played_decks, best_decks = _build(setup, learners, records=records, dials=dials, rng=rng)
    decks = [candidate.deck for candidate in played_decks]

    def matchup() -> tuple[Deck, Deck]:
        first, second = rng.sample(decks, SEATS)
        return first, second

    collected = collect(
        start(matchup),
        None,
        learner=learners.policy,
        opponent=opponent,
        guesser=learners.guesser,
        cards=len(setup.pool),
        history=setup.history,
        decisions=setup.decisions,
        generator=generator,
    )
    wins, played = _deck_records(decks, collected.games)

    metrics: dict[str, float] = {"games": float(len(collected.games))}
    metrics |= _update_play_models(setup, learners, rollout=collected.rollout, dials=dials)
    metrics |= _update_deck_score(setup, learners, decks=decks, wins=wins, played=played)
    metrics |= _update_builder(setup, learners, records=records, best_decks=best_decks, generator=generator)
    if dials.drafter_rate > 0:
        metrics |= _update_drafter(
            setup,
            learners,
            records=records,
            learner_seats=setup.draft_format.seats - len(other_seats),
            win_rates=wins / played.clamp_min(1.0),
            played=played,
            dials=dials,
        )
    return metrics

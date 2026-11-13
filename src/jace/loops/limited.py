"""One iteration of limited training, written from the package's pieces.

This is an example loop, not a framework: it drafts pods, builds a deck from
every pool, plays the decks against each other and updates every model, in
that order, as plain calls. Constructed play would be another short function
that skips the first two steps. A user who wants a different order, another
reward or another explore rule edits this function or writes their own; the
pieces it calls do not change.

No update here steps the shared card embedding. Every model reads cards
through it, so a step taken for one model's loss would change what every
other model sees, including a model whose own rate is zero. Each model
adapts through its own projection of the card vectors instead.
"""

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor
from torch.optim import Optimizer

from jace.cards import CardId, CardPool, Deck, Rarity
from jace.decks import (
    BuildRules,
    Candidate,
    DeckBuilderModel,
    DeckScore,
    build_decks,
    model_proposer,
    rules_for_pool,
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
from jace.engine import SEATS, Matchup
from jace.loops.schedules import Dials
from jace.modules import run
from jace.play import Guesser, Policy
from jace.tracking import ModelName
from jace.train import (
    GameResult,
    Metrics,
    PlayerRule,
    PlayGames,
    Rollout,
    apply,
    choice_loss,
    entropy,
    kl_to,
    score_loss,
    update_guess,
    update_player,
)

__all__ = ("Learners", "LimitedSetup", "UnevenDecksError", "balanced_matchups", "limited_iteration")

# PyTorch seeds a generator with a signed 64-bit integer, so a seed drawn from `random` uses 63 bits.
_SEED_BITS = 63
# Either deck of a pair is as likely to take the first seat.
_EVEN_ODDS = 0.5


@dataclass(frozen=True, slots=True)
class LimitedSetup:
    """Everything about a run that stays fixed from one iteration to the next.

    `booster_cards` lists the cards a booster may hold, by rarity. `pods`
    drafts are run per iteration, and each pool gets `proposals`
    candidate decks improved for `search_rounds` rounds. Every deck then
    plays `games_per_deck` games. `history` is how many earlier guesses the
    guess model is shown. `player_rule` is the player's update rule; its
    `spread` is replaced each iteration by the schedule's, and `passes` is
    how many times the player's update goes over the games. `score_chunk`
    bounds how many decks the deck score rates in one call during the
    search.
    """

    pool: CardPool
    draft_format: DraftFormat
    booster: BoosterSpec
    booster_cards: Mapping[Rarity, Sequence[CardId]]
    rules: BuildRules
    pods: int
    proposals: int
    search_rounds: int
    score_chunk: int
    games_per_deck: int
    history: int
    slice_rows: int
    max_grad_norm: float
    player_rule: PlayerRule
    passes: int
    lam_advantage: float
    lam_value: float


class UnevenDecksError(ValueError):
    def __init__(self, decks: int) -> None:
        super().__init__(f"{decks} decks cannot each play the same number of games: the count must be even")


@dataclass(frozen=True, slots=True)
class Learners:
    """The models being trained and the optimizer that moves each.

    `policy` and `guesser` are the player's and the guess model's functions,
    which depend on the engine's board layout. `human_drafter` is the frozen
    drafter trained on human picks that `Dials.human_pull` pulls toward, or
    `None` for no pull. `optimizers` holds one optimizer per model, keyed
    "player", "guess", "drafter", "deck_builder" and "deck_score", over
    that model's own parameters. None of them may hold the shared card
    embedding's parameters: build each from `own_parameters`.
    """

    policy: Policy
    guesser: Guesser
    drafter: DrafterModel
    human_drafter: DrafterModel | None
    builder: DeckBuilderModel
    score: DeckScore
    optimizers: Mapping[ModelName, Optimizer]


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
            [open_booster(setup.booster, setup.booster_cards, rng=rng) for _ in range(seats)]
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

    def score(decks: Tensor) -> Tensor:
        parts = [run(learners.score, part) for part in decks.split(setup.score_chunk)]
        return torch.cat(parts)

    played: list[Candidate] = []
    best: list[Candidate] = []
    for record in records:
        for pool in record.pools:
            # A drafter that took too many lands still gets a deck, with as many lands as it must play.
            rules = rules_for_pool(setup.rules, pool)
            candidates = build_decks(
                pool,
                cards=len(setup.pool),
                device=learners.drafter.embedding.starting.device,
                propose=model_proposer(learners.builder, rules=rules, generator=generator),
                score=score,
                rules=rules,
                proposals=setup.proposals,
                rounds=setup.search_rounds,
            )
            explore = len(candidates) > 1 and rng.random() < dials.explore_share
            played.append(rng.choice(candidates[1:]) if explore else candidates[0])
            best.append(candidates[0])
    return played, best


def _counts(decks: Sequence[Deck], *, cards: int, device: torch.device) -> Tensor:
    # Returns float [decks, cards] on `device`, the copies of each card in each deck.
    counts = torch.zeros(len(decks), cards)
    for row, deck in enumerate(decks):
        counts[row].index_add_(0, torch.tensor(deck), torch.ones(len(deck)))
    return counts.to(device)


def balanced_matchups(pods: int, *, seats: int, games_per_deck: int, rng: random.Random) -> list[tuple[int, int]]:
    """Schedule the games of an iteration's decks, as pairs of deck numbers.

    Deck `pod * seats + seat` is the deck of that seat of that pod. Every
    deck plays exactly `games_per_deck` games and never itself. With more
    than one pod, every game is between decks of different pods, so a deck
    is measured against decks that did not draft at its table. The first
    deck of a pair takes seat 0, and which one that is is drawn at random.
    Raises `UnevenDecksError` when the number of decks is odd.
    """
    decks = pods * seats
    if decks % SEATS:
        raise UnevenDecksError(decks)
    half = decks // SEATS
    games: list[tuple[int, int]] = []
    for _ in range(games_per_deck):
        # Pods in a random order, each a block of its decks in a random order. A deck plays the
        # deck half the line away, which is in another pod whenever there are two pods or more.
        line = [pod * seats + seat for pod in rng.sample(range(pods), pods) for seat in rng.sample(range(seats), seats)]
        pairs = zip(line[:half], line[half:], strict=True)
        games += [(first, second) if rng.random() < _EVEN_ODDS else (second, first) for first, second in pairs]
    return games


def _deck_records(
    schedule: Sequence[tuple[int, int]], games: Sequence[GameResult], *, decks: int
) -> tuple[Tensor, Tensor]:
    # schedule: the deck numbers of each game that was to be played, seat 0's first. Returns
    # (wins [decks], games played [decks]) over the games that finished; a draw counts as half
    # a win for each deck. Decks are told apart by number, so two equal decks stay two decks.
    wins = torch.zeros(decks)
    played = torch.zeros(decks)
    for game in games:
        learner_deck, other_deck = (schedule[game.game][seat] for seat in (game.learner_seat, 1 - game.learner_seat))
        wins[learner_deck] += game.learner_result
        wins[other_deck] += 1.0 - game.learner_result
        played[[learner_deck, other_deck]] += 1
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
    device = learners.drafter.embedding.starting.device
    reward = torch.tensor(rewards, device=device)
    seen, picks, pack = view_tensors(views, cards=len(setup.pool), device=device)
    logits = learners.drafter.forward(seen, picks, pack=pack)
    loss = choice_loss(logits, torch.tensor(chosen, device=device), weights=reward - reward.mean())
    loss = loss - entropy(logits).mean() * dials.drafter_entropy
    if learners.human_drafter is not None and dials.human_pull > 0:
        with torch.no_grad():
            human = learners.human_drafter.forward(seen, picks, pack=pack)
        loss = loss + kl_to(logits, human).mean() * dials.human_pull
    apply(loss, [learners.optimizers["drafter"]], max_grad_norm=setup.max_grad_norm)
    return {"drafter/loss": float(loss.detach()), "drafter/mean_reward": float(reward.mean())}


def _update_play_models(
    setup: LimitedSetup, learners: Learners, *, rollout: Rollout, dials: Dials, generator: torch.Generator
) -> Metrics:
    player = update_player(
        learners.policy,
        rollout,
        rule=replace(setup.player_rule, spread=dials.spread),
        lam_advantage=setup.lam_advantage,
        lam_value=setup.lam_value,
        slice_rows=setup.slice_rows,
        max_grad_norm=setup.max_grad_norm,
        optimizers=[learners.optimizers["player"]],
        passes=setup.passes,
        generator=generator,
    )
    guess = update_guess(
        learners.guesser,
        rollout,
        history=setup.history,
        slice_rows=setup.slice_rows,
        max_grad_norm=setup.max_grad_norm,
        optimizers=[learners.optimizers["guess"]],
        generator=generator,
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
    device = learners.drafter.embedding.starting.device
    logits = run(learners.score, _counts(decks, cards=len(setup.pool), device=device)[seen_play.to(device)])
    loss = score_loss(logits, wins=wins[seen_play].to(device), games=played[seen_play].to(device))
    apply(loss, [learners.optimizers["deck_score"]], max_grad_norm=setup.max_grad_norm)
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
    device = learners.drafter.embedding.starting.device
    pools = _counts([pool for record in records for pool in record.pools], cards=cards, device=device)
    best = _counts([candidate.deck for candidate in best_decks], cards=cards, device=device)
    available, partial, chosen = writing_steps(best, pool_counts=pools, rules=setup.rules, generator=generator)
    loss = choice_loss(
        run(learners.builder, available, partial), chosen, weights=torch.ones(len(chosen), device=device)
    )
    apply(loss, [learners.optimizers["deck_builder"]], max_grad_norm=setup.max_grad_norm)
    return {"deck_builder/loss": float(loss.detach())}


def limited_iteration(
    setup: LimitedSetup,
    learners: Learners,
    *,
    dials: Dials,
    other_seats: Sequence[Drafter],
    play: PlayGames,
    rng: random.Random,
) -> Metrics:
    """Draft, build, play and learn once, and return the metrics of every update.

    `dials` are this iteration's values from the schedule. `other_seats`
    are the drafters for the last seats of every pod, such as past
    versions; the learner drafts the rest. `play` plays a list of matchups
    to the end with the learners' player in both seats, on whatever engine
    and in whatever processes the caller chose. The games are the
    `balanced_matchups` of the decks just built, so every deck plays the
    same number of games and none is cut off. The metrics include the
    seconds each of the four steps took.
    """
    generator = torch.Generator().manual_seed(rng.getrandbits(_SEED_BITS))
    _set_rate(learners.optimizers["player"], dials.player_rate)
    _set_rate(learners.optimizers["drafter"], dials.drafter_rate)
    _set_rate(learners.optimizers["deck_score"], dials.deck_rate)
    _set_rate(learners.optimizers["deck_builder"], dials.deck_rate)

    began = time.perf_counter()
    drafter = model_drafter(learners.drafter, generator=generator)
    records = _draft(setup, drafter=drafter, other_seats=other_seats, rng=rng)
    drafted = time.perf_counter()
    played_decks, best_decks = _build(setup, learners, records=records, dials=dials, rng=rng)
    decks = [candidate.deck for candidate in played_decks]
    built = time.perf_counter()
    schedule = balanced_matchups(
        setup.pods, seats=setup.draft_format.seats, games_per_deck=setup.games_per_deck, rng=rng
    )
    matchups: list[Matchup] = [(decks[first], decks[second]) for first, second in schedule]
    collected = play(matchups)
    wins, played = _deck_records(schedule, collected.games, decks=len(decks))
    finished = time.perf_counter()

    metrics: dict[str, float] = {
        "games": float(len(collected.games)),
        "abandoned": float(collected.abandoned),
        "lost": float(collected.lost),
        "decisions": float(len(collected.rollout.games)),
    }
    metrics |= _update_play_models(setup, learners, rollout=collected.rollout, dials=dials, generator=generator)
    if dials.deck_rate > 0:
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
    return metrics | {
        "draft_seconds": drafted - began,
        "build_seconds": built - drafted,
        "play_seconds": finished - built,
        "update_seconds": time.perf_counter() - finished,
    }

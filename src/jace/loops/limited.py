"""One iteration of limited training, written from the package's pieces.

This is an example loop, not a framework: it drafts pods, searches decks
for every pool, plays them against each other in blocks and updates every
model, in that order, as plain calls. The search keeps each pool's best
few decks by the deck score, and each block plays one build of every
pool, chosen by `block_builds` so every block mixes builds of every rank.
A pool's builds then meet the same mix of opponents, and their win rates
can be compared. The player and
the guess model update after every block, because a policy-gradient step
learns only from games its current weights played. The deck models and the
drafter update once per iteration, from every block's games. The builder
learns from each build's win rate against its pool's other builds, and only
once the search reads the deck score that learns from self-play, which
`jace.loops.checked_score` decides. Constructed play would be another short
function that skips the first two steps. A user who wants a different
order, another reward or another explore rule edits this function or writes
their own; the pieces it calls do not change.

No update here steps the shared card embedding. Every model reads cards
through it, so a step taken for one model's loss would change what every
other model sees, including a model whose own rate is zero. Each model
adapts through its own projection of the card vectors instead.
"""

import collections
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor
from torch.optim import Optimizer

from jace.cards import CardId, CardPool, Deck, Rarity
from jace.decks import BuildRules, Builds, DeckBuilder, DeckScore, build_decks, build_log_probs, join_builds
from jace.draft import (
    Drafter,
    DrafterModel,
    DraftRecord,
    SeatView,
    model_drafter,
    open_booster,
    run_drafts,
    view_tensors,
)
from jace.engine import SEATS
from jace.loops.checked_score import PlayedDecks, ScoreCheck, ScoreRule, check_scores, train_learning_score
from jace.loops.formats import LimitedFormat
from jace.loops.schedules import Dials
from jace.play import Guesser, Policy, WinChance
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
    update_guess,
    update_player,
)

__all__ = ("Learners", "LimitedSetup", "UnevenDecksError", "balanced_matchups", "block_builds", "limited_iteration")

# PyTorch seeds a generator with a signed 64-bit integer, so a seed drawn from `random` uses 63 bits.
_SEED_BITS = 63
# Either deck of a pair is as likely to take the first seat.
_EVEN_ODDS = 0.5


@dataclass(frozen=True, slots=True)
class LimitedSetup:
    """Everything about a run that stays fixed from one iteration to the next.

    `limited_format` sizes the drafts, and `rules` are the deck rules
    `deck_rules` derives from it for the card pool. `booster_cards` lists
    the cards a booster may hold, by rarity. `pods`
    drafts are run per iteration, and the search samples `deck_samples`
    decks for each pool, `spells_per_round` cards in each round of the
    builder at `deck_temperature`, and keeps the `blocks` distinct decks
    the deck score ranks best. Each block plays the build of every pool
    that `block_builds` names, `games_per_block` games each, and the player and the guess model update
    after each block. `score_rule` says how the learning deck score trains
    and when the search switches to it. `history` is how many earlier guesses the
    guess model is shown. `player_rule` is the player's update rule; its
    `spread` is replaced each iteration by the schedule's, and `passes` is
    how many times the player's and the guess model's updates go over a
    block's games.
    """

    pool: CardPool
    limited_format: LimitedFormat
    booster_cards: Mapping[Rarity, Sequence[CardId]]
    rules: BuildRules
    pods: int
    deck_samples: int
    spells_per_round: int
    deck_temperature: float
    score_rule: ScoreRule
    blocks: int
    games_per_block: int
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

    `policy`, `win_chance` and `guesser` are the player's, the win-chance
    model's and the guess model's functions, which depend on the engine's
    board layout. `human_drafter` is the frozen drafter trained on human
    picks that `Dials.human_pull` pulls toward, or `None` for no pull.
    `score` is the deck score that learns from self-play, and `human_score`
    the frozen one trained on human decks, which the search reads until the
    learning copy has earned its place. `optimizers` holds one optimizer per
    model, keyed "player", "win_chance", "guess", "drafter", "deck_builder"
    and "deck_score", over that model's own parameters. The win chance
    learns at the player's rate. None of them may hold the shared card
    embedding's parameters: build each from `own_parameters`.
    """

    policy: Policy
    win_chance: WinChance
    guesser: Guesser
    drafter: DrafterModel
    human_drafter: DrafterModel | None
    builder: DeckBuilder
    score: DeckScore
    human_score: DeckScore
    optimizers: Mapping[ModelName, Optimizer]


def _set_rate(optimizer: Optimizer, rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = rate


def _draft(
    setup: LimitedSetup, *, drafter: Drafter, other_seats: Sequence[Drafter], rng: random.Random
) -> tuple[DraftRecord, ...]:
    # The learner's drafter fills every seat that `other_seats` does not.
    seats = setup.limited_format.draft.seats
    drafters = [*([drafter] * (seats - len(other_seats))), *other_seats]
    boosters = [
        [
            [open_booster(setup.limited_format.booster, setup.booster_cards, rng=rng) for _ in range(seats)]
            for _ in range(setup.limited_format.draft.rounds)
        ]
        for _ in range(setup.pods)
    ]
    return run_drafts(setup.limited_format.draft, boosters, drafters=drafters)


def _build(
    setup: LimitedSetup,
    learners: Learners,
    *,
    records: Sequence[DraftRecord],
    check: ScoreCheck,
    generator: torch.Generator,
) -> Builds:
    # Returns one row per pool and block, pools in pod then seat order and each pool's blocks in a row: block b
    # plays the build the search ranked b-th. A pool with fewer distinct decks than blocks plays its best again.
    score = learners.score if check.reads_learning else learners.human_score
    ranked = [
        build_decks(
            pool,
            builder=learners.builder,
            score=score,
            rules=setup.rules,
            cards=len(setup.pool),
            samples=setup.deck_samples,
            keep=setup.blocks,
            spells_per_round=setup.spells_per_round,
            temperature=setup.deck_temperature,
            generator=generator,
        )[0]
        for record in records
        for pool in record.pools
    ]
    in_each_block = torch.arange(setup.blocks, device=learners.builder.cards.embedding.device)
    return join_builds([builds.rows(in_each_block % len(builds.decks)) for builds in ranked])


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
    games_played = torch.zeros(decks)
    for game in games:
        learner_deck, other_deck = (schedule[game.game][seat] for seat in (game.learner_seat, 1 - game.learner_seat))
        wins[learner_deck] += game.learner_result
        wins[other_deck] += 1.0 - game.learner_result
        games_played[[learner_deck, other_deck]] += 1
    return wins, games_played


def _update_drafter(
    setup: LimitedSetup,
    learners: Learners,
    *,
    records: Sequence[DraftRecord],
    learner_seats: int,
    win_rates: Tensor,
    games_played: Tensor,
    dials: Dials,
) -> Metrics:
    # win_rates, games_played: [pods * seats] in pod then seat order. Seats whose deck played no game teach nothing.
    seats = setup.limited_format.draft.seats
    views: list[SeatView] = []
    chosen: list[int] = []
    rewards: list[float] = []
    for pod, record in enumerate(records):
        for seat in range(learner_seats):
            deck = pod * seats + seat
            if games_played[deck] == 0:
                continue
            views.extend(record.views[seat])
            chosen.extend(record.pools[seat])
            rewards.extend([float(win_rates[deck])] * len(record.views[seat]))
    if not views:
        nothing: dict[str, float] = {}
        return nothing
    device = learners.drafter.embedding.device
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
        win_chance=learners.win_chance,
        rule=replace(setup.player_rule, spread=dials.spread),
        lam_advantage=setup.lam_advantage,
        lam_value=setup.lam_value,
        slice_rows=setup.slice_rows,
        max_grad_norm=setup.max_grad_norm,
        policy_optimizers=[learners.optimizers["player"]],
        win_chance_optimizers=[learners.optimizers["win_chance"]],
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
        passes=setup.passes,
        generator=generator,
    )
    named = {f"player/{name}": value for name, value in player.items()}
    return named | {f"guess/{name}": value for name, value in guess.items()}


@dataclass(frozen=True, slots=True)
class _Blocks:
    # wins, games_played: float [blocks, decks], each block's record of each deck. metrics: the counts summed over
    # the blocks and each update's numbers averaged over them. seconds: (playing, updating), summed over the blocks.
    wins: Tensor
    games_played: Tensor
    metrics: Metrics
    seconds: tuple[float, float]


def _play_blocks(
    setup: LimitedSetup,
    learners: Learners,
    *,
    decks: Sequence[Sequence[Deck]],
    play: PlayGames,
    dials: Dials,
    rng: random.Random,
    generator: torch.Generator,
) -> _Blocks:
    # decks: one list of every pool's deck per block. Plays each block's games and updates the player and the guess
    # model after each.
    wins = torch.zeros(len(decks), len(decks[0]))
    games_played = torch.zeros(len(decks), len(decks[0]))
    counts: collections.Counter[str] = collections.Counter()
    updates: collections.defaultdict[str, list[float]] = collections.defaultdict(list)
    playing = updating = 0.0
    for block, block_decks in enumerate(decks):
        began = time.perf_counter()
        schedule = balanced_matchups(
            setup.pods, seats=setup.limited_format.draft.seats, games_per_deck=setup.games_per_block, rng=rng
        )
        collected = play([(block_decks[first], block_decks[second]) for first, second in schedule])
        wins[block], games_played[block] = _deck_records(schedule, collected.games, decks=len(block_decks))
        counts.update(
            games=len(collected.games),
            abandoned=collected.abandoned,
            lost=collected.lost,
            decisions=len(collected.rollout.games),
        )
        played = time.perf_counter()
        for name, value in _update_play_models(
            setup, learners, rollout=collected.rollout, dials=dials, generator=generator
        ).items():
            updates[name].append(value)
        playing += played - began
        updating += time.perf_counter() - played
    metrics = {name: float(count) for name, count in counts.items()}
    metrics |= {name: sum(values) / len(values) for name, values in updates.items()}
    return _Blocks(wins=wins, games_played=games_played, metrics=metrics, seconds=(playing, updating))


def _update_builder(
    setup: LimitedSetup, learners: Learners, *, played: Builds, wins: Tensor, games: Tensor, dials: Dials
) -> Metrics:
    # played: one row per pool and block, as `_build` returns them. wins, games: float [pools * blocks], each build's
    # record. A build's reward is its win rate minus the mean of its pool's builds', the shared baseline of POMO.
    games_by_pool = games.view(-1, setup.blocks)
    rates = wins.view(-1, setup.blocks) / games_by_pool.clamp_min(1.0)
    has_games = games_by_pool > 0
    baseline = (rates * has_games).sum(dim=1, keepdim=True) / has_games.sum(dim=1, keepdim=True).clamp_min(1)
    advantages = (rates - baseline)[has_games]
    log_probs, entropies = build_log_probs(
        learners.builder,
        played.rows(has_games.flatten().nonzero().squeeze(1)),
        rules=setup.rules,
        spells_per_round=setup.spells_per_round,
        temperature=setup.deck_temperature,
    )
    loss = -(advantages * log_probs).mean() - dials.builder_entropy * entropies.mean()
    apply(loss, [learners.optimizers["deck_builder"]], max_grad_norm=setup.max_grad_norm)
    return {"deck_builder/loss": float(loss.detach()), "deck_builder/entropy": float(entropies.mean().detach())}


def block_builds(pools: int, *, blocks: int) -> Tensor:
    """Say which build of each pool plays in each block, as int64 `[blocks, pools]`.

    A pool's builds are numbered best first. Pool `p` plays its build
    `(p + b) % blocks` in block `b`, so each pool plays each build once.
    When the pools outnumber the blocks, every block holds builds of every
    rank, so a pool's best build does not meet only other pools' best
    builds, and a pool's builds meet the same mix of opponents. Were block
    `b` to play every pool's build `b`, each block's win rates would
    average one half whatever the builds' ranks, and no rank could be told
    better than another.
    """
    return (torch.arange(pools)[None] + torch.arange(blocks)[:, None]) % blocks


def _block_decks(decks: Tensor, *, builds: Tensor) -> list[list[Deck]]:
    # decks: float [pools * blocks, cards], as `_build` lays them out. builds: `block_builds`'s [blocks, pools].
    # Returns each block's list of every pool's deck.
    blocks = builds.shape[0]
    every_card = torch.arange(decks.shape[1])
    as_ids = [tuple(every_card.repeat_interleave(row).tolist()) for row in decks.long().cpu()]
    return [[as_ids[pool * blocks + int(build)] for pool, build in enumerate(row)] for row in builds]


def limited_iteration(
    setup: LimitedSetup,
    learners: Learners,
    *,
    check: ScoreCheck,
    dials: Dials,
    other_seats: Sequence[Drafter],
    play: PlayGames,
    rng: random.Random,
) -> tuple[Metrics, ScoreCheck]:
    """Draft, build, play and learn once, and return the metrics of every update with the deck-score check after it.

    `check` is what the deck-score check carried out of the iteration
    before, and `FIRST_CHECK` for the first. `dials` are this iteration's
    values from the schedule. `other_seats` are the drafters for the last
    seats of every pod, such as past versions; the learner drafts the rest.
    `play` plays a list of matchups to the end with the learners' player in
    both seats, on whatever engine and in whatever processes the caller
    chose. Each block's games are the `balanced_matchups` of that block's
    builds, so every deck plays the same number of games and none is cut
    off. The builder learns only in an iteration whose search read the
    learning deck score. The metrics include the seconds each of the four
    steps took, playing and updating summed over the blocks.
    """
    generator = torch.Generator().manual_seed(rng.getrandbits(_SEED_BITS))
    _set_rate(learners.optimizers["player"], dials.player_rate)
    _set_rate(learners.optimizers["win_chance"], dials.player_rate)
    _set_rate(learners.optimizers["drafter"], dials.drafter_rate)
    _set_rate(learners.optimizers["deck_score"], dials.score_rate)
    _set_rate(learners.optimizers["deck_builder"], dials.builder_rate)

    began = time.perf_counter()
    drafter = model_drafter(learners.drafter, generator=generator)
    records = _draft(setup, drafter=drafter, other_seats=other_seats, rng=rng)
    drafted = time.perf_counter()
    played = _build(setup, learners, records=records, check=check, generator=generator)
    built = time.perf_counter()
    builds = block_builds(len(played.decks) // setup.blocks, blocks=setup.blocks)
    blocks = _play_blocks(
        setup,
        learners,
        decks=_block_decks(played.decks, builds=builds),
        play=play,
        dials=dials,
        rng=rng,
        generator=generator,
    )
    deck_updates_began = time.perf_counter()

    device = played.decks.device
    # Each build's record, laid out as `_build` lays out the decks: each pool's builds, best first.
    by_build = torch.zeros(builds.shape[1], setup.blocks)
    wins = by_build.scatter(1, builds.T, blocks.wins.T).flatten().to(device)
    games = by_build.scatter(1, builds.T, blocks.games_played.T).flatten().to(device)
    has_games = games > 0
    tested = PlayedDecks(decks=played.decks[has_games], wins=wins[has_games], games=games[has_games])
    after, checked = check_scores(
        check, played=tested, human=learners.human_score, learning=learners.score, rule=setup.score_rule
    )
    metrics = dict(blocks.metrics) | dict(checked)
    if dials.score_rate > 0:
        metrics |= train_learning_score(
            learners.score,
            learners.human_score,
            recent=after.recent,
            rule=setup.score_rule,
            optimizer=learners.optimizers["deck_score"],
            max_grad_norm=setup.max_grad_norm,
        )
    if check.reads_learning and dials.builder_rate > 0:
        metrics |= _update_builder(setup, learners, played=played, wins=wins, games=games, dials=dials)
    if dials.drafter_rate > 0:
        # A pool's deck is its best build, the one the deck score ranked first.
        best_wins, best_games = wins.view(-1, setup.blocks)[:, 0].cpu(), games.view(-1, setup.blocks)[:, 0].cpu()
        metrics |= _update_drafter(
            setup,
            learners,
            records=records,
            learner_seats=setup.limited_format.draft.seats - len(other_seats),
            win_rates=best_wins / best_games.clamp_min(1.0),
            games_played=best_games,
            dials=dials,
        )
    playing, updating = blocks.seconds
    seconds = {
        "draft_seconds": drafted - began,
        "build_seconds": built - drafted,
        "play_seconds": playing,
        "update_seconds": updating + time.perf_counter() - deck_updates_began,
    }
    return metrics | seconds, after

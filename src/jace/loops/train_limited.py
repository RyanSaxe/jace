"""Train every model on real games of one card set, starting from human data.

This is the limited loop run for real: packs of the set's playable cards are
drafted, a deck is built from every pool, and the decks play each other on
the manabot engine. The drafter, the deck models and the card embedding can
start from checkpoints trained on human data; the player and the guess model
always start untrained. A model started from a checkpoint reads cards
through the embedding of `--embedding-run`, which must then be given,
because a model's weights fit only the embedding they were trained with.
Every few iterations the player is scored against
uniformly random play on human decks it never trains on, which is the one
number that shows whether play is improving. Run it with
`uv run python -m jace.loops.train_limited --help`.
"""

import argparse
import copy
import functools
import json
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from jace.cards import CardEmbedding, Deck, Rarity, own_parameters
from jace.data import SetCards, card_table, load_decks, read_set
from jace.decks import DeckBuilderModel, DeckScore
from jace.draft import TLA_PLAY_BOOSTER, DrafterModel, DraftFormat
from jace.engine import SEATS, BoardLayout, GameBatch
from jace.engine.manabot import board_layout, engine_card_ids, engine_cards, start_batch
from jace.eval import WinRate, win_rate
from jace.loops.entry_points import RunArguments, add_run_flags, limited_rules, playable_decks, start_run
from jace.loops.limited import Learners, LimitedSetup, StartBatch, limited_iteration
from jace.loops.schedules import phased_schedule
from jace.play import Guesser, GuessModel, PlayerModel, Policy, random_policy
from jace.tracking import (
    CheckpointRef,
    ModelName,
    Running,
    latest_checkpoint,
    load_checkpoint,
    log_update,
    save_checkpoint,
    writing,
)
from jace.train import PlayerRule, collect

__all__ = ("main",)

_SEATS_IN_A_POD = 8
_BOOSTERS_EACH = 3
# Ataraxos's player rule. Its `keep_quantile` and `spread` are replaced by the arguments and the schedule.
_PLAYER_RULE = PlayerRule(
    clip=0.2, stay=0.1, spread=0.0, keep_quantile=0.75, keep_floor=0.0, value_weight=0.5, normalize=False
)
_RANGE_RESAMPLES = 1000
_SCORE_CHUNK = 1024
_SLICE_ROWS = 512
# PyTorch seeds a generator with a signed 64-bit integer, and the engine with an unsigned 32-bit one.
_SEED_BITS = 63
_ENGINE_SEED_BITS = 31


@dataclass(frozen=True, slots=True)
class _Arguments(RunArguments):
    iterations: int
    drafter_run: Path | None
    decks_run: Path | None
    embedding_run: Path | None
    checkpoint_every: int
    resume: Path | None
    games: int
    decisions: int
    pods: int
    proposals: int
    search_rounds: int
    evaluate_every: int
    evaluate_decisions: int
    evaluate_decks: int
    drafter_starts: int
    pull_ends: int
    history: int
    play_width: int
    play_layers: int
    play_heads: int
    guess_cards: int
    drafter_width: int
    drafter_layers: int
    drafter_heads: int
    deck_width: int
    deck_layers: int
    deck_heads: int
    embedding_hidden: int
    rate: float
    player_rate: float | None
    lam_advantage: float
    lam_value: float
    keep_quantile: float


def _arguments(argv: Sequence[str]) -> _Arguments:
    parser = argparse.ArgumentParser(description="Train every model on real games of one card set.")
    add_run_flags(parser)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--drafter-run", type=Path, help="a run folder whose newest drafter to start from")
    parser.add_argument("--decks-run", type=Path, help="a run folder whose newest deck models to start from")
    parser.add_argument(
        "--embedding-run",
        type=Path,
        help="a run folder whose newest card embedding to start from; required with either flag above",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=0, help="also save every checkpoint each N iterations; 0 for never"
    )
    parser.add_argument("--resume", type=Path, help="a stopped run folder of this command to continue from")
    parser.add_argument("--games", type=int, default=64, help="games played at once")
    parser.add_argument("--decisions", type=int, default=4096, help="learner decisions collected per iteration")
    parser.add_argument("--pods", type=int, default=2, help="eight-seat drafts per iteration")
    parser.add_argument("--proposals", type=int, default=8, help="candidate decks per pool")
    parser.add_argument("--search-rounds", type=int, default=1)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--evaluate-decisions", type=int, default=8192)
    parser.add_argument("--evaluate-decks", type=int, default=64, help="held-out human decks used for scoring")
    parser.add_argument("--drafter-starts", type=int, default=20, help="the iteration the drafter starts learning at")
    parser.add_argument("--pull-ends", type=int, default=200, help="the iteration the human pull reaches zero at")
    parser.add_argument("--history", type=int, default=4, help="earlier guesses the guess model is shown")
    parser.add_argument("--play-width", type=int, default=128)
    parser.add_argument("--play-layers", type=int, default=2)
    parser.add_argument("--play-heads", type=int, default=4)
    parser.add_argument("--guess-cards", type=int, default=8)
    parser.add_argument("--drafter-width", type=int, default=256)
    parser.add_argument("--drafter-layers", type=int, default=6)
    parser.add_argument("--drafter-heads", type=int, default=8)
    parser.add_argument("--deck-width", type=int, default=128)
    parser.add_argument("--deck-layers", type=int, default=2)
    parser.add_argument("--deck-heads", type=int, default=4)
    parser.add_argument("--embedding-hidden", type=int, default=256)
    parser.add_argument("--rate", type=float, default=1e-4, help="the learning rate of the models with no schedule")
    parser.add_argument("--player-rate", type=float, help="a fixed player learning rate in place of the schedule's")
    parser.add_argument("--lam-advantage", type=float, default=0.5, help="how far back a result's credit reaches")
    parser.add_argument("--lam-value", type=float, default=0.8)
    parser.add_argument("--keep-quantile", type=float, default=0.75, help="learn from decisions above this quantile")
    arguments = _Arguments(**vars(parser.parse_args(argv)))
    starts_a_model = arguments.drafter_run is not None or arguments.decks_run is not None
    if starts_a_model and arguments.embedding_run is None:
        # A model's weights mean something only with the card embedding they were trained on.
        parser.error("--drafter-run and --decks-run need --embedding-run, the run whose card embedding the models read")
    return arguments


@dataclass(frozen=True, slots=True)
class _Models:
    # Every trained thing of a run. The five models share `embedding`.
    player: PlayerModel
    guess: GuessModel
    drafter: DrafterModel
    builder: DeckBuilderModel
    score: DeckScore
    embedding: CardEmbedding


def _by_name(models: _Models) -> dict[ModelName, nn.Module]:
    return {
        "player": models.player,
        "guess": models.guess,
        "drafter": models.drafter,
        "deck_builder": models.builder,
        "deck_score": models.score,
        "embedding": models.embedding,
    }


def _build_models(
    arguments: _Arguments, cards: SetCards, *, layout: BoardLayout, engine_to_card: Tensor, picks: int
) -> _Models:
    embedding = CardEmbedding(cards.features, hidden=arguments.embedding_hidden, offset_bound=None)
    deck_shape = {"width": arguments.deck_width, "layers": arguments.deck_layers, "heads": arguments.deck_heads}
    play_shape = {"width": arguments.play_width, "layers": arguments.play_layers, "heads": arguments.play_heads}
    return _Models(
        player=PlayerModel(
            embedding, engine_to_card=engine_to_card, layout=layout, guess_cards=arguments.guess_cards, **play_shape
        ),
        guess=GuessModel(embedding, engine_to_card=engine_to_card, layout=layout, **play_shape),
        drafter=DrafterModel(
            embedding,
            width=arguments.drafter_width,
            layers=arguments.drafter_layers,
            heads=arguments.drafter_heads,
            picks=picks,
        ),
        builder=DeckBuilderModel(embedding, **deck_shape),
        score=DeckScore(embedding, **deck_shape),
        embedding=embedding,
    )


def _load_inputs(arguments: _Arguments, models: _Models) -> tuple[CheckpointRef, ...]:
    # Loads the newest checkpoints of the runs the arguments name and returns their references.
    # Only `--embedding-run` sets the shared embedding: a model's checkpoint never replaces it.
    modules = _by_name(models)
    sources: list[tuple[Path | None, ModelName]] = [
        (arguments.embedding_run, "embedding"),
        (arguments.drafter_run, "drafter"),
        (arguments.decks_run, "deck_builder"),
        (arguments.decks_run, "deck_score"),
    ]
    inputs: list[CheckpointRef] = []
    for run_folder, model in sources:
        if run_folder is not None:
            ref = latest_checkpoint(run_folder, model)
            keeping = None if model == "embedding" else models.embedding
            load_checkpoint(run_folder.parent, ref, into=modules[model], keeping=keeping)
            inputs.append(ref)
    return tuple(inputs)


def _resume(run_folder: Path, models: _Models) -> int:
    # Loads the newest checkpoint of every module of the stopped run at `run_folder` and returns
    # the iteration they were saved after. Optimizer state is not checkpointed, so it starts fresh.
    modules = _by_name(models)
    updates: set[int] = set()
    for model in ("embedding", *(name for name in modules if name != "embedding")):
        ref = latest_checkpoint(run_folder, model)
        keeping = None if model == "embedding" else models.embedding
        load_checkpoint(run_folder.parent, ref, into=modules[model], keeping=keeping)
        updates.add(ref.update)
    if len(updates) != 1:
        message = f"the newest checkpoints in {run_folder} are from different iterations: {sorted(updates)}"
        raise ValueError(message)
    return updates.pop()


def _score_against(
    opponent: Policy,
    *,
    learner: Policy,
    guesser: Guesser,
    start: StartBatch,
    decks: Sequence[Deck],
    cards: int,
    history: int,
    decisions: int,
    rng: random.Random,
) -> WinRate:
    """Play `learner` against `opponent` on `decks` and return the learner's win rate.

    Each game deals two of `decks` at random and gives the learner a random
    seat. Games are played until the finished ones hold `decisions` learner
    decisions, so long games that are still running at that point are left
    out, which favours short games a little. The range treats games as
    independent.
    """
    generator = torch.Generator().manual_seed(rng.getrandbits(_SEED_BITS))

    def matchup() -> tuple[Deck, Deck]:
        first, second = rng.sample(decks, SEATS)
        return first, second

    collected = collect(
        start(matchup),
        None,
        learner=learner,
        opponent=opponent,
        guesser=guesser,
        cards=cards,
        history=history,
        decisions=decisions,
        generator=generator,
    )
    results = torch.tensor([game.learner_result for game in collected.games])
    return win_rate(results, resamples=_RANGE_RESAMPLES, generator=generator)


def _train(
    run: Running,
    arguments: _Arguments,
    *,
    iterate: Callable[[int], dict[str, float]],
    score_against_random: Callable[[], WinRate],
    modules: Mapping[ModelName, nn.Module],
    curve_file: Path,
    done: int,
) -> None:
    """Run every iteration, scoring the player and saving checkpoints on the way.

    The player is scored before the first iteration and after every
    `evaluate_every`-th. Every module of `modules` is checkpointed after
    every `checkpoint_every`-th iteration, when that is above zero, and
    after the last. The curve file is rewritten after each iteration, so a
    run that is stopped still has its curve. `done` is how many iterations
    a resumed run has already finished and checkpointed; a new run passes
    zero. A resumed run keeps the curve up to `done` and goes on from there.
    """
    every = arguments.checkpoint_every
    checkpoint_at = {arguments.iterations, *(range(every, arguments.iterations, every) if every > 0 else ())}
    earlier = json.loads(curve_file.read_text()) if done > 0 else {"iterations": [], "evaluations": []}
    iterations: list[dict[str, float]] = [point for point in earlier["iterations"] if point["iteration"] <= done]
    evaluations: list[dict[str, float]] = [point for point in earlier["evaluations"] if point["iteration"] <= done]
    started = time.perf_counter()
    for iteration in range(done, arguments.iterations + 1):
        if iteration > done:
            began = time.perf_counter()
            metrics = {**iterate(iteration), "seconds": time.perf_counter() - began}
            log_update(run, model="player", update=iteration, metrics=metrics)
            iterations.append({"iteration": iteration, **metrics})
        # A resumed run scored its last finished iteration before it stopped.
        if iteration % arguments.evaluate_every == 0 and (iteration > done or done == 0):
            rate = score_against_random()
            evaluations.append({"iteration": iteration, "rate": rate.rate, "low": rate.low, "high": rate.high})
            sys.stdout.write(f"iteration {iteration:4d}  win rate against random {rate.rate:.3f}\n")
            sys.stdout.flush()
        if iteration in checkpoint_at and iteration > done:
            for name, module in modules.items():
                save_checkpoint(run, model=name, update=iteration, module=module)
        summary = {"run_id": run.folder.name, "seconds": time.perf_counter() - started}
        curve_file.write_text(json.dumps({**summary, "iterations": iterations, "evaluations": evaluations}))


def main(argv: Sequence[str]) -> None:
    """Run limited training on one card set and save its record, checkpoints and curve.

    The run folder is created under `--run-dir` by `jace.tracking` and gets
    a checkpoint of every trained thing at the end, and every
    `--checkpoint-every` iterations before that when the flag is given. The
    curve is written beside it as `<run id>.curve.json`: the metrics of
    every iteration and the player's win rate against random play at each
    evaluation.
    """
    arguments = _arguments(argv)
    torch.manual_seed(arguments.seed)
    rng = random.Random(arguments.seed)
    cpu = torch.device("cpu")
    data = read_set(arguments.set_code)
    cards = card_table(data.attributes, data.statistics, programs={card.name: card.program for card in engine_cards()})
    pool = cards.pool
    playable = {rarity: [card for card in pool.of_rarity(rarity) if pool[card].program] for rarity in Rarity}
    draft_format = DraftFormat(seats=_SEATS_IN_A_POD, rounds=_BOOSTERS_EACH, picks_per_round=len(TLA_PLAY_BOOSTER))
    human = load_decks(data.games, pool=pool, cache_dir=arguments.data_dir, held_out_share=arguments.held_out_share)
    scoring_decks = playable_decks(human.decks[human.held_out], pool=pool, count=arguments.evaluate_decks)

    def start(matchups: Callable[[], tuple[Deck, Deck]]) -> GameBatch:
        # Every batch needs its own seed, so this cannot be a partial application of `start_batch`.
        # ast-grep-ignore: no-routing-functions
        return start_batch(matchups, pool=pool, games=arguments.games, seed=rng.getrandbits(_ENGINE_SEED_BITS))

    engine_to_card = engine_card_ids(pool)
    picks = draft_format.rounds * draft_format.picks_per_round
    models = _build_models(arguments, cards, layout=board_layout(), engine_to_card=engine_to_card, picks=picks)
    inputs = _load_inputs(arguments, models)
    done = 0 if arguments.resume is None else _resume(arguments.resume, models)
    # A resumed run must not replay the games it already played.
    rng.seed(arguments.seed + done)
    human_drafter = None
    if arguments.drafter_run is not None:
        human_drafter = copy.deepcopy(models.drafter).eval().requires_grad_(requires_grad=False)

    modules = _by_name(models)
    # The embedding's parameters are in its own optimizer only, so one loss never steps them twice.
    optimizers: dict[ModelName, torch.optim.Optimizer] = {
        name: torch.optim.Adam(own_parameters(module, models.embedding), lr=arguments.rate)
        for name, module in modules.items()
        if module is not models.embedding
    }
    optimizers["embedding"] = torch.optim.Adam(models.embedding.parameters(), lr=arguments.rate)
    setup = LimitedSetup(
        pool=pool,
        draft_format=draft_format,
        booster=TLA_PLAY_BOOSTER,
        booster_cards=playable,
        rules=limited_rules(cards),
        pods=arguments.pods,
        proposals=arguments.proposals,
        search_rounds=arguments.search_rounds,
        score_chunk=_SCORE_CHUNK,
        decisions=arguments.decisions,
        history=arguments.history,
        slice_rows=_SLICE_ROWS,
        max_grad_norm=1.0,
        player_rule=replace(_PLAYER_RULE, keep_quantile=arguments.keep_quantile),
        lam_advantage=arguments.lam_advantage,
        lam_value=arguments.lam_value,
    )
    learners = Learners(
        policy=models.player.policy,
        guesser=models.guess.guess,
        drafter=models.drafter,
        human_drafter=human_drafter,
        builder=models.builder,
        score=models.score,
        optimizers=optimizers,
    )
    schedule = phased_schedule(drafter_starts=arguments.drafter_starts, pull_ends=arguments.pull_ends)

    def iterate(iteration: int) -> dict[str, float]:
        dials = schedule(iteration)
        if arguments.player_rate is not None:
            dials = replace(dials, player_rate=arguments.player_rate)
        metrics = limited_iteration(
            setup,
            learners,
            dials=dials,
            opponent=models.player.policy,
            other_seats=(),
            start=start,
            rng=rng,
        )
        return dict(metrics)

    score_against_random = functools.partial(
        _score_against,
        random_policy,
        learner=models.player.policy,
        guesser=models.guess.guess,
        start=start,
        decks=scoring_decks,
        cards=len(pool),
        history=arguments.history,
        decisions=arguments.evaluate_decisions,
        rng=rng,
    )
    clock = functools.partial(datetime.now, UTC)
    if arguments.resume is None:
        folder, curve_file = start_run(arguments, device=cpu, inputs=inputs, clock=clock)
    else:
        folder, curve_file = arguments.resume, arguments.resume.with_name(f"{arguments.resume.name}.curve.json")
    with writing(folder, clock=clock) as run:
        _train(
            run,
            arguments,
            iterate=iterate,
            score_against_random=score_against_random,
            modules=modules,
            curve_file=curve_file,
            done=done,
        )


if __name__ == "__main__":
    main(sys.argv[1:])

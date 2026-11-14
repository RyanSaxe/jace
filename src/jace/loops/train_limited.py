"""Train every model on real games of one card set, starting from human data.

This is the limited loop run for real: packs of the set's playable cards are
drafted, a deck is built from every pool, and the decks play each other on
the engine in worker processes, every deck the same number of games and
every game to its end. Every model reads cards through the card vectors of
`--embedding-run`, frozen into a `CardTable`. The drafter and the deck
models can start from checkpoints trained on human data; the player and
the guess model always start untrained. A model's weights fit only the card
vectors they were trained with, so `--embedding-run` should be the run its
checkpoint came from or one that read the same vectors. The table is not
trained here, so a model whose rate is zero does not change. Every few
iterations the player is scored with the scoring harness on paired deals of
human decks it never trains on, against random play, scripted greedy and
the heuristic player. Run it with
`uv run python -m jace.loops.train_limited --help`.
"""

import argparse
import copy
import functools
import json
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from jace.cards import CardPool, CardTable, Rarity, own_parameters
from jace.data import read_set
from jace.decks import DeckBuilderModel, DeckScore
from jace.draft import TLA_PLAY_BOOSTER, DrafterModel, DraftFormat
from jace.engine import Matchup
from jace.loops.entry_points import (
    PlayArguments,
    PlayModels,
    add_play_flags,
    add_run_flags,
    build_play_models,
    card_inputs,
    device_named,
    limited_rules,
    load_card_table,
    play_actor,
    play_weights,
    score_player,
    scoring_harness,
    set_cards,
    start_run,
)
from jace.loops.limited import Learners, LimitedSetup, limited_iteration
from jace.loops.matches import Harness
from jace.loops.schedules import phased_schedule
from jace.play import GuessModel, PlayerModel, guesser_on, policy_on
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
from jace.train import Collected, PlayerRule
from jace.train.workers import play_on_workers, start_workers, stop_workers

__all__ = ("main",)

_SEATS_IN_A_POD = 8
_BOOSTERS_EACH = 3
# Ataraxos's player rule. Its `keep_quantile`, `normalize` and `spread` are replaced by the arguments and the schedule.
_PLAYER_RULE = PlayerRule(
    clip=0.2, stay=0.1, spread=0.0, keep_quantile=0.75, keep_floor=0.0, value_weight=0.5, normalize=True
)
_SCORE_CHUNK = 1024
_SLICE_ROWS = 512


@dataclass(frozen=True, slots=True)
class _Arguments(PlayArguments):
    device: str
    iterations: int
    drafter_run: Path | None
    decks_run: Path | None
    embedding_run: Path
    checkpoint_every: int
    resume: Path | None
    pods: int
    blocks: int
    games_per_block: int
    proposals: int
    search_rounds: int
    decks_start: int
    drafter_starts: int
    pull_ends: int
    drafter_width: int
    drafter_layers: int
    drafter_heads: int
    deck_width: int
    deck_layers: int
    deck_heads: int
    rate: float
    player_rate: float | None
    lam_advantage: float
    lam_value: float
    keep_quantile: float


def _arguments(argv: Sequence[str]) -> _Arguments:
    parser = argparse.ArgumentParser(description="Train every model on real games of one card set.")
    add_run_flags(parser)
    add_play_flags(parser)
    parser.add_argument("--device", default="auto", help="where the updates run: cpu, mps or cuda; auto takes mps")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--drafter-run", type=Path, help="a run folder whose newest drafter to start from")
    parser.add_argument("--decks-run", type=Path, help="a run folder whose newest deck models to start from")
    parser.add_argument(
        "--embedding-run",
        type=Path,
        required=True,
        help="a drafter pretraining, deck pretraining or self-play run folder whose newest card vectors to read",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=0, help="also save every checkpoint each N iterations; 0 for never"
    )
    parser.add_argument("--resume", type=Path, help="a stopped run folder of this command to continue from")
    parser.add_argument("--pods", type=int, default=2, help="eight-seat drafts per iteration")
    parser.add_argument(
        "--blocks", type=int, default=4, help="blocks of games per iteration; the player updates after each"
    )
    parser.add_argument("--games-per-block", type=int, default=25, help="games every drafted deck plays in each block")
    parser.add_argument("--proposals", type=int, default=8, help="candidate decks per pool")
    parser.add_argument("--search-rounds", type=int, default=1)
    parser.add_argument("--decks-start", type=int, default=20, help="the iteration the deck models start learning at")
    parser.add_argument("--drafter-starts", type=int, default=20, help="the iteration the drafter starts learning at")
    parser.add_argument("--pull-ends", type=int, default=200, help="the iteration the human pull reaches zero at")
    parser.add_argument("--drafter-width", type=int, default=256)
    parser.add_argument("--drafter-layers", type=int, default=6)
    parser.add_argument("--drafter-heads", type=int, default=8)
    parser.add_argument("--deck-width", type=int, default=128)
    parser.add_argument("--deck-layers", type=int, default=2)
    parser.add_argument("--deck-heads", type=int, default=4)
    parser.add_argument("--rate", type=float, default=1e-4, help="the learning rate of the models with no schedule")
    parser.add_argument("--player-rate", type=float, help="a fixed player learning rate in place of the schedule's")
    parser.add_argument("--lam-advantage", type=float, default=0.5, help="how far back a result's credit reaches")
    parser.add_argument("--lam-value", type=float, default=0.8)
    parser.add_argument("--keep-quantile", type=float, default=0.75, help="learn from decisions above this quantile")
    arguments = _Arguments(**vars(parser.parse_args(argv)))
    if arguments.resume is not None and arguments.drafter_run is None:
        started_with = json.loads((arguments.resume / "identity.json").read_text())["config"]["drafter_run"]
        if started_with is not None:
            # The pull is toward the drafter that run started from, which only its checkpoint holds.
            parser.error(f"the run being resumed pulls its drafter toward {started_with}: pass it as --drafter-run")
    return arguments


@dataclass(frozen=True, slots=True)
class _Models:
    # Every trained thing of a run. The five models share `embedding`, which is not trained.
    player: PlayerModel
    guess: GuessModel
    drafter: DrafterModel
    builder: DeckBuilderModel
    score: DeckScore
    embedding: CardTable


def _by_name(models: _Models) -> dict[ModelName, nn.Module]:
    return {
        "player": models.player,
        "guess": models.guess,
        "drafter": models.drafter,
        "deck_builder": models.builder,
        "deck_score": models.score,
        "embedding": models.embedding,
    }


def _build_models(arguments: _Arguments, embedding: CardTable, *, pool: CardPool, picks: int) -> _Models:
    play = build_play_models(arguments, embedding, pool=pool)
    deck_shape = {"width": arguments.deck_width, "layers": arguments.deck_layers, "heads": arguments.deck_heads}
    return _Models(
        player=play.player,
        guess=play.guess,
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
    # Loads the newest model checkpoints of the runs the arguments name and returns their references.
    # A model's checkpoint never replaces the shared card table.
    modules = _by_name(models)
    sources: list[tuple[Path | None, ModelName]] = [
        (arguments.drafter_run, "drafter"),
        (arguments.decks_run, "deck_builder"),
        (arguments.decks_run, "deck_score"),
    ]
    inputs: list[CheckpointRef] = []
    for run_folder, model in sources:
        if run_folder is not None:
            ref = latest_checkpoint(run_folder, model)
            load_checkpoint(run_folder.parent, ref, into=modules[model], keeping=models.embedding)
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


def _save(run: Running, models: _Models, *, iteration: int) -> CheckpointRef:
    # Saves every module and returns the player's checkpoint.
    saved = {
        name: save_checkpoint(run, model=name, update=iteration, module=module)
        for name, module in _by_name(models).items()
    }
    return saved["player"]


def _train(
    run: Running,
    arguments: _Arguments,
    *,
    iterate: Callable[[int], dict[str, float]],
    models: _Models,
    harness: Harness,
    curve_file: Path,
    done: int,
) -> None:
    """Run every iteration after `done`, scoring the player and saving checkpoints on the way.

    Every module is saved after every `checkpoint_every`-th iteration, when
    that is above zero, after the last, and after every `evaluate_every`-th,
    when the player is also scored on the harness's deals. The curve file
    is rewritten after each iteration, so a run that is stopped still has
    its curve. `done` is how many iterations a resumed run has already
    finished and checkpointed; a new run passes zero. A resumed run keeps
    the curve up to `done` and goes on from there.
    """
    every = arguments.checkpoint_every
    checkpoint_at = {arguments.iterations, *(range(every, arguments.iterations, every) if every > 0 else ())}
    earlier = json.loads(curve_file.read_text()) if done > 0 else {"iterations": [], "evaluations": []}
    iterations: list[dict[str, float]] = [point for point in earlier["iterations"] if point["iteration"] <= done]
    evaluations: list[dict[str, float]] = [point for point in earlier["evaluations"] if point["iteration"] <= done]
    play_models = PlayModels(player=models.player, guess=models.guess, embedding=models.embedding)
    started = time.perf_counter()
    for iteration in range(done + 1, arguments.iterations + 1):
        metrics = iterate(iteration)
        log_update(run, model="player", update=iteration, metrics=metrics)
        iterations.append({"iteration": iteration, **metrics})
        line = f"iteration {iteration:4d}  {metrics['decisions']:.0f} decisions in {metrics['games']:.0f} games  "
        line += "  ".join(f"{step} {metrics[f'{step}_seconds']:.1f}s" for step in ("draft", "build", "play", "update"))
        if iteration % arguments.evaluate_every == 0:
            scoring = time.perf_counter()
            checkpoint = _save(run, models, iteration=iteration)
            rates = score_player(run, checkpoint, models=play_models, harness=harness, history=arguments.history)
            evaluations.append({"iteration": iteration, **rates, "score_seconds": time.perf_counter() - scoring})
            line += "  wins, sampled/top: " + ", ".join(
                f"{name} {rates[name]:.2f}/{rates[f'{name}_top']:.2f}" for name in ("random", "greedy", "heuristic")
            )
        elif iteration in checkpoint_at:
            _save(run, models, iteration=iteration)
        summary = {"run_id": run.folder.name, "seconds": time.perf_counter() - started}
        curve_file.write_text(json.dumps({**summary, "iterations": iterations, "evaluations": evaluations}))
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _move(models: _Models, human_drafter: DrafterModel | None, *, to: torch.device) -> torch.device:
    # Drafting, deck building and every update run on the device. The workers play on the CPU,
    # with weights sent from here.
    for module in (*_by_name(models).values(), *filter(None, [human_drafter])):
        module.to(to)
    return to


def main(argv: Sequence[str]) -> None:
    """Run limited training on one card set and save its record, checkpoints and curve.

    The run folder is created under `--run-dir` by `jace.tracking` and gets
    a checkpoint of every model at the end, at every evaluation, and every
    `--checkpoint-every` iterations. The curve is written beside it as
    `<run id>.curve.json`: the metrics of every iteration, with the seconds
    each step took, and the player's win rates against the three scripted
    players at each evaluation.
    """
    arguments = _arguments(argv)
    torch.manual_seed(arguments.seed)
    data = read_set(arguments.set_code)
    cards = set_cards(data)
    pool = cards.pool
    playable = {rarity: [card for card in pool.of_rarity(rarity) if pool[card].program] for rarity in Rarity}
    draft_format = DraftFormat(seats=_SEATS_IN_A_POD, rounds=_BOOSTERS_EACH, picks_per_round=len(TLA_PLAY_BOOSTER))
    embedding, embedding_ref = load_card_table(
        arguments.embedding_run, inputs=card_inputs(data, cards, data_dir=arguments.data_dir)
    )
    models = _build_models(arguments, embedding, pool=pool, picks=draft_format.rounds * draft_format.picks_per_round)
    inputs = (embedding_ref, *_load_inputs(arguments, models))
    # The copy is taken before a resumed run's own drafter is loaded, so the pull stays toward
    # the drafter trained on human picks. It keeps its own copy of the card table.
    human_drafter = None
    if arguments.drafter_run is not None:
        human_drafter = copy.deepcopy(models.drafter).eval().requires_grad_(requires_grad=False)
    done = 0 if arguments.resume is None else _resume(arguments.resume, models)
    # A resumed run must not replay the games it already played.
    rng = random.Random(arguments.seed + done)

    # The card table has no parameters, so it has no optimizer: self-play does not train it.
    optimizers: dict[ModelName, torch.optim.Optimizer] = {
        name: torch.optim.Adam(own_parameters(module, models.embedding), lr=arguments.rate)
        for name, module in _by_name(models).items()
        if module is not models.embedding
    }
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
        blocks=arguments.blocks,
        games_per_block=arguments.games_per_block,
        history=arguments.history,
        slice_rows=_SLICE_ROWS,
        max_grad_norm=1.0,
        player_rule=replace(_PLAYER_RULE, keep_quantile=arguments.keep_quantile, normalize=arguments.normalize),
        passes=arguments.passes,
        lam_advantage=arguments.lam_advantage,
        lam_value=arguments.lam_value,
    )
    device = _move(models, human_drafter, to=device_named(arguments.device))
    learners = Learners(
        policy=policy_on(models.player.policy, device),
        guesser=guesser_on(models.guess.guess, device),
        drafter=models.drafter,
        human_drafter=human_drafter,
        builder=models.builder,
        score=models.score,
        optimizers=optimizers,
    )
    schedule = phased_schedule(
        decks_start=arguments.decks_start, drafter_starts=arguments.drafter_starts, pull_ends=arguments.pull_ends
    )
    workers = start_workers(
        functools.partial(play_actor, arguments=arguments, card_width=embedding.width),
        count=arguments.workers,
        seed=arguments.seed + done,
    )
    play_models = PlayModels(player=models.player, guess=models.guess, embedding=models.embedding)

    def play(matchups: Sequence[Matchup]) -> Collected:
        # The workers of one round are the ones the round before left, with any stalled one replaced.
        # Every opponent of the round holds one of its decks, so they are the guess model's field
        # while the workers play the games and while it learns from them afterwards.
        nonlocal workers
        play_models.guess.set_field([deck for matchup in matchups for deck in matchup])
        collected, workers = play_on_workers(
            workers, matchups, weights=play_weights(play_models), stall_seconds=arguments.stall_seconds
        )
        return collected

    def iterate(iteration: int) -> dict[str, float]:
        dials = schedule(iteration)
        if arguments.player_rate is not None:
            dials = replace(dials, player_rate=arguments.player_rate)
        return dict(limited_iteration(setup, learners, dials=dials, other_seats=(), play=play, rng=rng))

    clock = functools.partial(datetime.now, UTC)
    if arguments.resume is None:
        folder, curve_file = start_run(arguments, device=device, inputs=inputs, clock=clock)
    else:
        folder, curve_file = arguments.resume, arguments.resume.with_name(f"{arguments.resume.name}.curve.json")
    try:
        with writing(folder, clock=clock) as run:
            _train(
                run,
                arguments,
                iterate=iterate,
                models=models,
                harness=scoring_harness(arguments, cards),
                curve_file=curve_file,
                done=done,
            )
    finally:
        stop_workers(workers)


if __name__ == "__main__":
    main(sys.argv[1:])

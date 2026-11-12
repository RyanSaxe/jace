"""Train only the player and the guess model, by self-play on a fixed list of human decks.

This is the limited loop with the draft and the deck building left out: the
decks are human decks from the training share of the 17Lands data, so what
is measured is how well the player learns to play, and nothing else moves.
The held-out decks that `jace.loops.score` deals from are never played
here. Games are collected in worker processes by `jace.train.workers`, so
an engine step that takes minutes costs one worker its games and not the
run, and no supervisor is needed around this command. Every flag that
shapes the player's update is on the command line, because this loop exists
to compare them. The run's config holds what `jace.loops.score` reads, so
that command scores these checkpoints unchanged. Run it with
`uv run python -m jace.loops.train_player --help`.
"""

import argparse
import functools
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from jace.cards import CardEmbedding, Deck, own_parameters
from jace.data import SetCards, card_table, load_decks, read_set
from jace.engine import SEATS
from jace.engine.manabot import engine_card_ids, engine_cards, land_and_spell_actions, start_batch
from jace.loops.entry_points import RunArguments, add_run_flags, playable_decks, start_run
from jace.play import GuessModel, PlayerModel, as_boards
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
from jace.train import PlayerRule, Rollout, update_guess, update_player
from jace.train.workers import Actor, Worker, gather, start_workers, stop_workers

__all__ = ("main",)

# The dials of the player's rule that no flag sets, as `train_limited` has them.
_CLIP = 0.2
_STAY = 0.1
_VALUE_WEIGHT = 0.5
_SLICE_ROWS = 512
_MAX_GRAD_NORM = 1.0
# PyTorch seeds a generator with a signed 64-bit integer, and the engine with an unsigned 32-bit one.
_SEED_BITS = 63
_ENGINE_SEED_BITS = 31
_TRAINED: tuple[ModelName, ...] = ("embedding", "player", "guess")


@dataclass(frozen=True, slots=True)
class _Arguments(RunArguments):
    iterations: int
    embedding_run: Path | None
    checkpoint_every: int
    resume: Path | None
    workers: int
    games_per_worker: int
    stall_seconds: float
    decisions: int
    decks: int
    evaluate_every: int
    passes: int
    shuffle: bool
    normalize: bool
    rate: float
    spread: float
    lam_advantage: float
    lam_value: float
    keep_quantile: float
    history: int
    play_width: int
    play_layers: int
    play_heads: int
    guess_cards: int
    embedding_hidden: int


@dataclass(frozen=True, slots=True)
class _Models:
    player: PlayerModel
    guess: GuessModel
    embedding: CardEmbedding


def _arguments(argv: Sequence[str]) -> _Arguments:
    parser = argparse.ArgumentParser(description="Train only the player and the guess model on human decks.")
    add_run_flags(parser)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--embedding-run", type=Path, help="a run folder whose newest card embedding to start from")
    parser.add_argument(
        "--checkpoint-every", type=int, default=0, help="also save every checkpoint each N iterations; 0 for never"
    )
    parser.add_argument("--resume", type=Path, help="a stopped run folder of this command to continue from")
    parser.add_argument("--workers", type=int, default=12, help="processes that play games")
    parser.add_argument("--games-per-worker", type=int, default=8, help="games each worker plays at once")
    parser.add_argument(
        "--stall-seconds", type=float, default=60.0, help="a worker that completes no step for this long is replaced"
    )
    parser.add_argument("--decisions", type=int, default=8192, help="decisions collected per iteration")
    parser.add_argument(
        "--decks", type=int, default=256, help="human decks, not held out, that the games are dealt from"
    )
    parser.add_argument(
        "--evaluate-every", type=int, default=5, help="record the land and spell rates each N iterations"
    )
    parser.add_argument("--passes", type=int, default=1, help="passes of the player's update over each rollout")
    parser.add_argument("--shuffle", action="store_true", help="take the rows in a new random order on every pass")
    parser.add_argument("--normalize", action="store_true", help="scale advantages to mean zero and unit spread")
    parser.add_argument("--rate", type=float, default=3e-4, help="the learning rate of every model")
    parser.add_argument("--spread", type=float, default=0.0, help="the pull toward even odds over legal actions")
    parser.add_argument("--lam-advantage", type=float, default=0.95, help="how far back a result's credit reaches")
    parser.add_argument("--lam-value", type=float, default=0.95)
    parser.add_argument("--keep-quantile", type=float, default=0.0, help="learn from decisions above this quantile")
    parser.add_argument("--history", type=int, default=4, help="earlier guesses the guess model is shown")
    parser.add_argument("--play-width", type=int, default=128)
    parser.add_argument("--play-layers", type=int, default=2)
    parser.add_argument("--play-heads", type=int, default=4)
    parser.add_argument("--guess-cards", type=int, default=8)
    parser.add_argument("--embedding-hidden", type=int, default=256)
    return _Arguments(**vars(parser.parse_args(argv)))


def _cards_and_decks(arguments: _Arguments) -> tuple[SetCards, list[Deck]]:
    # The card set, and the human decks of the training share that the engine can play.
    data = read_set(arguments.set_code)
    cards = card_table(data.attributes, data.statistics, programs={card.name: card.program for card in engine_cards()})
    human = load_decks(
        data.games, pool=cards.pool, cache_dir=arguments.data_dir, held_out_share=arguments.held_out_share
    )
    return cards, playable_decks(human.decks[~human.held_out], pool=cards.pool, count=arguments.decks)


def _actor(seed: int, *, arguments: _Arguments) -> Actor:
    """Build one worker's games and models, inside the worker's process.

    The games deal two different decks of the training share at random,
    from a generator made from `seed`, and the engine is seeded from it
    too. The models have the shapes the arguments give and whatever
    weights PyTorch starts them with: the first round replaces them with
    the training process's.
    """
    cards, decks = _cards_and_decks(arguments)
    rng = random.Random(seed)

    def matchup() -> tuple[Deck, Deck]:
        first, second = rng.sample(decks, SEATS)
        return first, second

    batch = start_batch(
        matchup, pool=cards.pool, games=arguments.games_per_worker, seed=rng.getrandbits(_ENGINE_SEED_BITS)
    )
    models = _build_models(arguments, cards, decks=decks)
    return Actor(
        batch=batch,
        policy=models.player.policy,
        guesser=models.guess.guess,
        modules=_by_name(models),
        cards=len(cards.pool),
        history=arguments.history,
    )


def _build_models(arguments: _Arguments, cards: SetCards, *, decks: Sequence[Deck]) -> _Models:
    # One game is started and never stepped: the models read their input widths from its boards.
    sample = as_boards(
        start_batch(lambda: (decks[0], decks[1]), pool=cards.pool, games=1, seed=0).decisions(),
        device=torch.device("cpu"),
    )
    engine_to_card = engine_card_ids(cards.pool)
    embedding = CardEmbedding(cards.features, hidden=arguments.embedding_hidden, offset_bound=None)
    shape = {"width": arguments.play_width, "layers": arguments.play_layers, "heads": arguments.play_heads}
    return _Models(
        player=PlayerModel(
            embedding, engine_to_card=engine_to_card, sample=sample, guess_cards=arguments.guess_cards, **shape
        ),
        guess=GuessModel(embedding, engine_to_card=engine_to_card, sample=sample, **shape),
        embedding=embedding,
    )


def _by_name(models: _Models) -> dict[str, nn.Module]:
    return {"embedding": models.embedding, "player": models.player, "guess": models.guess}


def _resume(run_folder: Path, models: _Models) -> int:
    # Loads the newest checkpoint of every module of the stopped run at `run_folder` and returns
    # the iteration they were saved after. Optimizer state is not checkpointed, so it starts fresh.
    modules = _by_name(models)
    updates: set[int] = set()
    for model in _TRAINED:
        ref = latest_checkpoint(run_folder, model)
        keeping = None if model == "embedding" else models.embedding
        load_checkpoint(run_folder.parent, ref, into=modules[model], keeping=keeping)
        updates.add(ref.update)
    if len(updates) != 1:
        message = f"the newest checkpoints in {run_folder} are from different iterations: {sorted(updates)}"
        raise ValueError(message)
    return updates.pop()


def _land_and_spell_rates(rollout: Rollout) -> dict[str, float]:
    # How likely the player was to play a land where it could, and to cast a spell where it could:
    # the chance it gave those actions, averaged over the decisions that offered one.
    chances = rollout.logits.softmax(dim=-1)  # [n, max_actions]
    rates: dict[str, float] = {}
    for name, offered in zip(("land_rate", "spell_rate"), land_and_spell_actions(rollout.boards), strict=True):
        could = offered.any(dim=-1)
        rates[name] = float((chances * offered).sum(dim=-1)[could].mean()) if bool(could.any()) else 0.0
    return rates


def _iterate(
    arguments: _Arguments,
    models: _Models,
    *,
    workers: Sequence[Worker],
    optimizers: Mapping[ModelName, torch.optim.Optimizer],
    generator: torch.Generator,
) -> tuple[dict[str, float], Rollout, tuple[Worker, ...]]:
    # One round of games and one update of the player and the guess model. Returns the
    # iteration's metrics, its rollout and the workers for the next round.
    began = time.perf_counter()
    weights = {name: module.state_dict() for name, module in _by_name(models).items()}
    gathered = gather(workers, weights=weights, decisions=arguments.decisions, stall_seconds=arguments.stall_seconds)
    collected = time.perf_counter()
    rollout = gathered.rollout
    rule = PlayerRule(
        clip=_CLIP,
        stay=_STAY,
        spread=arguments.spread,
        keep_quantile=arguments.keep_quantile,
        keep_floor=0.0,
        value_weight=_VALUE_WEIGHT,
        normalize=arguments.normalize,
    )
    player = update_player(
        models.player.policy,
        rollout,
        rule=rule,
        lam_advantage=arguments.lam_advantage,
        lam_value=arguments.lam_value,
        slice_rows=_SLICE_ROWS,
        max_grad_norm=_MAX_GRAD_NORM,
        optimizers=[optimizers["player"], optimizers["embedding"]],
        passes=arguments.passes,
        generator=generator if arguments.shuffle else None,
    )
    guess = update_guess(
        models.guess.guess,
        rollout,
        history=arguments.history,
        slice_rows=_SLICE_ROWS,
        max_grad_norm=_MAX_GRAD_NORM,
        optimizers=[optimizers["guess"], optimizers["embedding"]],
    )
    decisions = len(rollout.games)
    metrics = {
        "games": float(len(gathered.games)),
        "abandoned": float(gathered.abandoned),
        "lost_workers": float(gathered.lost_workers),
        "decisions": float(decisions),
        "collect_seconds": collected - began,
        "update_seconds": time.perf_counter() - collected,
        "decisions_per_second": decisions / (collected - began),
        **{f"player/{name}": value for name, value in player.items()},
        **{f"guess/{name}": value for name, value in guess.items()},
    }
    return metrics, rollout, gathered.workers


def _train(
    run: Running, arguments: _Arguments, *, models: _Models, workers: Sequence[Worker], curve_file: Path, done: int
) -> tuple[Worker, ...]:
    """Run every iteration after `done`, saving the curve and the checkpoints on the way.

    Every module is checkpointed after every `checkpoint_every`-th
    iteration, when that is above zero, and after the last. The land and
    spell rates of every `evaluate_every`-th iteration's games go in the
    curve's evaluations. The curve file is rewritten after each iteration,
    so a stopped run still has its curve, and a resumed run keeps the curve
    up to `done`. Returns the workers as the last round left them.
    """
    modules = _by_name(models)
    optimizers: dict[ModelName, torch.optim.Optimizer] = {
        name: torch.optim.Adam(own_parameters(modules[name], models.embedding), lr=arguments.rate)
        for name in ("player", "guess")
    }
    optimizers["embedding"] = torch.optim.Adam(models.embedding.parameters(), lr=arguments.rate)
    generator = torch.Generator().manual_seed(random.Random(arguments.seed + done).getrandbits(_SEED_BITS))
    every = arguments.checkpoint_every
    checkpoint_at = {arguments.iterations, *(range(every, arguments.iterations, every) if every > 0 else ())}
    earlier = json.loads(curve_file.read_text()) if done > 0 else {"iterations": [], "evaluations": []}
    iterations: list[dict[str, float]] = [point for point in earlier["iterations"] if point["iteration"] <= done]
    evaluations: list[dict[str, float]] = [point for point in earlier["evaluations"] if point["iteration"] <= done]
    started = time.perf_counter()
    current = tuple(workers)
    for iteration in range(done + 1, arguments.iterations + 1):
        metrics, rollout, current = _iterate(
            arguments, models, workers=current, optimizers=optimizers, generator=generator
        )
        log_update(run, model="player", update=iteration, metrics=metrics)
        iterations.append({"iteration": iteration, **metrics})
        if iteration % arguments.evaluate_every == 0:
            evaluations.append({"iteration": iteration, **_land_and_spell_rates(rollout)})
        if iteration in checkpoint_at:
            for name in _TRAINED:
                save_checkpoint(run, model=name, update=iteration, module=modules[name])
        summary = {"run_id": run.folder.name, "seconds": time.perf_counter() - started}
        curve_file.write_text(json.dumps({**summary, "iterations": iterations, "evaluations": evaluations}))
        sys.stdout.write(
            f"iteration {iteration:4d}  {metrics['decisions']:.0f} decisions in {metrics['games']:.0f} games  "
            f"collect {metrics['collect_seconds']:.1f}s  update {metrics['update_seconds']:.1f}s  "
            f"lost workers {metrics['lost_workers']:.0f}\n"
        )
        sys.stdout.flush()
    return current


def main(argv: Sequence[str]) -> None:
    """Train the player and the guess model on human decks and save the run's record.

    The run folder is created under `--run-dir` by `jace.tracking` and gets
    a checkpoint of the embedding, the player and the guess model at the
    end, and every `--checkpoint-every` iterations before that. The curve
    is written beside it as `<run id>.curve.json`: the metrics of every
    iteration, and the land and spell rates at each evaluation.
    """
    arguments = _arguments(argv)
    torch.manual_seed(arguments.seed)
    cards, decks = _cards_and_decks(arguments)
    models = _build_models(arguments, cards, decks=decks)
    inputs: tuple[CheckpointRef, ...] = ()
    if arguments.embedding_run is not None:
        ref = latest_checkpoint(arguments.embedding_run, "embedding")
        load_checkpoint(arguments.embedding_run.parent, ref, into=models.embedding, keeping=None)
        inputs = (ref,)
    done = 0 if arguments.resume is None else _resume(arguments.resume, models)
    clock = functools.partial(datetime.now, UTC)
    if arguments.resume is None:
        folder, curve_file = start_run(arguments, device=torch.device("cpu"), inputs=inputs, clock=clock)
    else:
        folder, curve_file = arguments.resume, arguments.resume.with_name(f"{arguments.resume.name}.curve.json")
    # A resumed run must not replay the games it already played.
    workers = start_workers(
        functools.partial(_actor, arguments=arguments), count=arguments.workers, seed=arguments.seed + done
    )
    try:
        with writing(folder, clock=clock) as run:
            workers = _train(run, arguments, models=models, workers=workers, curve_file=curve_file, done=done)
    finally:
        stop_workers(workers)


if __name__ == "__main__":
    main(sys.argv[1:])

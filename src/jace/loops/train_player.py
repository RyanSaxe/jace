"""Train only the player, its win chance and the guess model, by self-play on fixed human decks.

This is the limited loop without the draft and the deck building. The decks are
human decks from the training split of the 17Lands data, and only the player,
its win chance and the guess model train. What the loop measures is therefore
how well the player learns to play. Each iteration plays `--games` games to
their end in worker processes, all with that iteration's weights, and then
updates the three models.

All three read cards through the card vectors of `--embedding-run`, frozen into
a `CardTable`. The table is not trained here, because a step taken for one
model would change the other models' input. The guess model's meta, its
statistics of the format's decks, starts from the human decks of the training
split, and each iteration's decks join it once their games are over.

Every few iterations the player is scored against random play, scripted greedy
and the heuristic player, on paired deals of held-out human decks that training
never plays. A paired deal is two decks and an engine seed, played four times
so that each player plays each deck from each seat. Most settings of the
player's update are flags, because this loop exists to compare them. Run it
with `uv run python -m jace.loops.train_player --help`.
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

from jace.cards import Deck, own_parameters
from jace.data import load_decks, read_set
from jace.engine import SEATS, Matchup
from jace.loops.entry_points import (
    PlayArguments,
    PlayModels,
    add_play_flags,
    add_run_flags,
    build_play_models,
    card_inputs,
    device_named,
    land_and_spell_rates,
    load_card_table,
    play_actor,
    play_weights,
    playable_decks,
    score_player,
    scoring_harness,
    set_cards,
    start_run,
    starting_meta,
)
from jace.loops.matches import Harness
from jace.play import blend_meta
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
from jace.train import Collected, PlayerRule, PlayGames, Rollout, to_device, update_guess, update_player
from jace.train.workers import play_on_workers, start_workers, stop_workers

__all__ = ("main",)

# The update settings that no flag sets, with the values `train_limited` uses.
_CLIP = 0.2
_STAY = 0.1
_SLICE_ROWS = 512
_MAX_GRAD_NORM = 1.0
# A PyTorch generator accepts any seed from -2**63 to 2**64 - 1, so 63 random bits always make a valid seed.
_SEED_BITS = 63
_SAVED: tuple[ModelName, ...] = ("embedding", "player", "win_chance", "guess")


@dataclass(frozen=True, slots=True)
class _Arguments(PlayArguments):
    iterations: int
    embedding_run: Path
    checkpoint_every: int
    resume: Path | None
    games: int
    decks: int
    rate: float
    spread: float
    lam_advantage: float
    lam_value: float
    keep_quantile: float
    device: str


def _arguments(argv: Sequence[str]) -> _Arguments:
    parser = argparse.ArgumentParser(
        description="Train only the player, its win chance and the guess model on human decks."
    )
    add_run_flags(parser)
    add_play_flags(parser)
    parser.add_argument("--iterations", type=int, required=True)
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
    parser.add_argument("--games", type=int, default=48, help="games played per iteration, each to its end")
    parser.add_argument(
        "--decks", type=int, default=256, help="human decks, not held out, that the games are dealt from"
    )
    parser.add_argument("--rate", type=float, default=3e-4, help="the learning rate of every model")
    parser.add_argument("--spread", type=float, default=0.0, help="the pull toward even odds over legal actions")
    parser.add_argument("--lam-advantage", type=float, default=0.95, help="how far back a result's credit reaches")
    parser.add_argument("--lam-value", type=float, default=0.95)
    parser.add_argument("--keep-quantile", type=float, default=0.0, help="learn from decisions above this quantile")
    parser.add_argument("--device", default="auto", help="where the update runs: cpu, mps or cuda; auto takes mps")
    return _Arguments(**vars(parser.parse_args(argv)))


def _resume(run_folder: Path, models: PlayModels) -> int:
    # Returns the iteration the newest checkpoints were saved after. Optimizers are not checkpointed and start fresh.
    modules = {
        "embedding": models.embedding,
        "player": models.player,
        "win_chance": models.win_chance,
        "guess": models.guess,
    }
    updates: set[int] = set()
    for model in _SAVED:
        ref = latest_checkpoint(run_folder, model)
        keeping = None if model == "embedding" else models.embedding
        load_checkpoint(run_folder.parent, ref, into=modules[model], keeping=keeping)
        updates.add(ref.update)
    if len(updates) != 1:
        message = f"the newest checkpoints in {run_folder} are from different iterations: {sorted(updates)}"
        raise ValueError(message)
    return updates.pop()


def _update(
    arguments: _Arguments,
    models: PlayModels,
    *,
    rollout: Rollout,
    optimizers: Mapping[ModelName, torch.optim.Optimizer],
    generator: torch.Generator,
) -> dict[str, float]:
    on_device = to_device(rollout, models.embedding.device)
    rule = PlayerRule(
        clip=_CLIP,
        stay=_STAY,
        spread=arguments.spread,
        keep_quantile=arguments.keep_quantile,
        keep_floor=0.0,
        normalize=arguments.normalize,
    )
    player = update_player(
        models.player.policy,
        on_device,
        win_chance=models.win_chance.win_chance,
        rule=rule,
        lam_advantage=arguments.lam_advantage,
        lam_value=arguments.lam_value,
        slice_rows=_SLICE_ROWS,
        max_grad_norm=_MAX_GRAD_NORM,
        policy_optimizers=[optimizers["player"]],
        win_chance_optimizers=[optimizers["win_chance"]],
        passes=arguments.passes,
        generator=generator,
    )
    guess = update_guess(
        models.guess.guess,
        on_device,
        history=arguments.history,
        slice_rows=_SLICE_ROWS,
        max_grad_norm=_MAX_GRAD_NORM,
        optimizers=[optimizers["guess"]],
        passes=arguments.passes,
        generator=generator,
    )
    named = {f"player/{name}": value for name, value in player.items()}
    return named | {f"guess/{name}": value for name, value in guess.items()}


def _save(run: Running, models: PlayModels, *, iteration: int) -> CheckpointRef:
    # The card table does not change in this loop. It is saved anyway, so that this run's checkpoints can be loaded
    # and scored without the run their card vectors came from.
    save_checkpoint(run, model="embedding", update=iteration, module=models.embedding)
    save_checkpoint(run, model="win_chance", update=iteration, module=models.win_chance)
    save_checkpoint(run, model="guess", update=iteration, module=models.guess)
    return save_checkpoint(run, model="player", update=iteration, module=models.player)


def _train(
    run: Running,
    arguments: _Arguments,
    *,
    models: PlayModels,
    decks: Sequence[Deck],
    play: PlayGames,
    harness: Harness,
    curve_file: Path,
    done: int,
) -> None:
    """Run every iteration after `done`, saving the curve and the checkpoints on the way.

    An iteration plays `--games` games, each between two decks drawn at random
    from `decks`, and then updates the three models on them. Every module is
    saved after the last iteration and after every `evaluate_every`-th, when
    the player is also scored on the harness's deals. With `checkpoint_every`
    above zero, every module is also saved after every `checkpoint_every`-th
    iteration. The curve file is rewritten after each iteration, so a stopped
    run still has its curve.

    `done` is how many iterations a resumed run has already finished and
    checkpointed, and a new run passes zero. A resumed run keeps its curve up
    to `done`, drops any later points, and goes on from there.
    """
    optimizers: dict[ModelName, torch.optim.Optimizer] = {
        name: torch.optim.Adam(own_parameters(module, models.embedding), lr=arguments.rate)
        for name, module in (("player", models.player), ("win_chance", models.win_chance), ("guess", models.guess))
    }
    # A resumed run must not replay the games it already played.
    rng = random.Random(arguments.seed + done)
    generator = torch.Generator().manual_seed(rng.getrandbits(_SEED_BITS))
    every = arguments.checkpoint_every
    checkpoint_at = {arguments.iterations, *(range(every, arguments.iterations, every) if every > 0 else ())}
    earlier = json.loads(curve_file.read_text()) if done > 0 else {"iterations": [], "evaluations": []}
    iterations: list[dict[str, float]] = [point for point in earlier["iterations"] if point["iteration"] <= done]
    evaluations: list[dict[str, float]] = [point for point in earlier["evaluations"] if point["iteration"] <= done]
    started = time.perf_counter()
    for iteration in range(done + 1, arguments.iterations + 1):
        began = time.perf_counter()
        matchups: list[Matchup] = []
        for _ in range(arguments.games):
            first, second = rng.sample(decks, SEATS)
            matchups.append((first, second))
        collected = play(matchups)
        play_ended = time.perf_counter()
        updates = _update(arguments, models, rollout=collected.rollout, optimizers=optimizers, generator=generator)
        # Each deck of the iteration's matchups joins the meta once, however many games it played, and only after
        # the iteration's games are over.
        played = sorted({deck for matchup in matchups for deck in matchup})
        copies = torch.stack([torch.bincount(torch.tensor(deck), minlength=models.embedding.cards) for deck in played])
        meta = blend_meta(models.guess.meta, copies.float().to(models.embedding.device), decay=arguments.meta_decay)
        models.guess.set_meta(meta)
        decisions = len(collected.rollout.games)
        metrics = {
            "games": float(len(collected.games)),
            "abandoned": float(collected.abandoned),
            "lost": float(collected.lost),
            "decisions": float(decisions),
            "play_seconds": play_ended - began,
            "update_seconds": time.perf_counter() - play_ended,
            "decisions_per_second": decisions / (play_ended - began),
            **updates,
        }
        log_update(run, model="player", update=iteration, metrics=metrics)
        iterations.append({"iteration": iteration, **metrics})
        line = f"iteration {iteration:4d}  {decisions} decisions in {metrics['games']:.0f} games  "
        line += f"play {metrics['play_seconds']:.1f}s  update {metrics['update_seconds']:.1f}s"
        if iteration % arguments.evaluate_every == 0:
            scoring = time.perf_counter()
            checkpoint = _save(run, models, iteration=iteration)
            rates = score_player(run, checkpoint, models=models, harness=harness, history=arguments.history)
            rates |= land_and_spell_rates(collected.rollout)
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


def main(argv: Sequence[str]) -> None:
    """Train the player, its win chance and the guess model on human decks and save the run's record.

    `jace.tracking` creates the run folder under `--run-dir`. The run saves a
    checkpoint of the card table, the player, the win chance and the guess
    model at the end, at every evaluation and every `--checkpoint-every`
    iterations. The curve file `<run id>.curve.json` is next to the folder. It
    contains the metrics of every iteration and, at each evaluation, the
    player's win rates against the three scripted players, with its land and
    spell rates in that iteration's games.
    """
    arguments = _arguments(argv)
    torch.manual_seed(arguments.seed)
    data = read_set(arguments.set_code)
    cards = set_cards(data)
    human = load_decks(
        data.games, pool=cards.pool, cache_dir=arguments.data_dir, held_out_share=arguments.held_out_share
    )
    decks = playable_decks(human.decks[~human.held_out], pool=cards.pool, count=arguments.decks)
    embedding, embedding_ref = load_card_table(
        arguments.embedding_run, inputs=card_inputs(data, cards, data_dir=arguments.data_dir)
    )
    models = build_play_models(arguments, embedding, pool=cards.pool)
    done = 0 if arguments.resume is None else _resume(arguments.resume, models)
    device = device_named(arguments.device)
    for module in (models.embedding, models.player, models.win_chance, models.guess):
        module.to(device)
    clock = functools.partial(datetime.now, UTC)
    if arguments.resume is None:
        folder, curve_file = start_run(arguments, device=device, inputs=(embedding_ref,), clock=clock)
        models.guess.set_meta(starting_meta(arguments, cards))
    else:
        # The resumed guess model's checkpoint already contains the meta the run had reached, so it is not set again.
        folder, curve_file = arguments.resume, arguments.resume.with_name(f"{arguments.resume.name}.curve.json")
    workers = start_workers(
        functools.partial(play_actor, arguments=arguments, card_width=embedding.width),
        count=arguments.workers,
        seed=arguments.seed + done,
    )

    def play(matchups: Sequence[Matchup]) -> Collected:
        # Reuses the last call's workers, with any stalled one replaced. The decisions are returned on the CPU, and
        # `_update` moves them to the models' device.
        nonlocal workers
        collected, workers = play_on_workers(
            workers,
            matchups,
            weights=play_weights(models),
            stall_seconds=arguments.stall_seconds,
            device=torch.device("cpu"),
        )
        return collected

    try:
        with writing(folder, clock=clock) as run:
            _train(
                run,
                arguments,
                models=models,
                decks=decks,
                play=play,
                harness=scoring_harness(arguments, cards),
                curve_file=curve_file,
                done=done,
            )
    finally:
        stop_workers(workers)


if __name__ == "__main__":
    main(sys.argv[1:])

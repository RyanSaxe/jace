"""Train the deck score and the deck builder on human decks, before any self-play.

The deck score learns each deck's chance of winning a game from the wins and
losses of the event it played. The deck builder learns the human's deck for a
pool by teacher forcing, which trains it to make each of the human's choices
likely, given the deck built so far. The choices are the human's land total,
the deck's spells and nonbasic lands, and the human's split of basic lands. The
spells and nonbasic lands come in a random order, cut into the rounds the
builder samples in. Each time a deck is drawn it gets a new order, so the
builder learns to finish a deck from any partial one. Both models read the card
vectors of a drafter pretraining run, frozen into a `CardTable`, so neither can
change what the drafter learned about the cards.

Under `BuildRules`, a deck may contain any number of basic lands. The builder
therefore never reads the basics counted in a human pool, and its basics head
chooses how many of each basic fill the land slots that are left. The builder
trains and is tested only on human decks that the rules allow, while the deck
score trains on every deck. Both models are measured with
`jace.loops.deck_measures`, on held-out decks and on as many training decks, so
the two curves show when a model starts to fit its training decks better than
decks it has not seen.

Run it with `uv run python -m jace.loops.pretrain_decks --help`. The card
set must already be downloaded with `uv run spells add <set>`.
"""

import argparse
import functools
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from jace.cards import own_parameters
from jace.data import RESULT_FLOOR, HumanDecks, load_decks, read_set
from jace.decks import BuildRules, Builds, DeckBuilder, DeckScore, build_log_probs, builds_from_decks, sample_builds
from jace.loops.deck_measures import (
    GAMES_FOR_A_RATE,
    allowed_by,
    build_log_loss,
    deck_overlap,
    land_total_match,
    score_fit,
)
from jace.loops.entry_points import (
    ADAM_BETAS,
    RunArguments,
    add_run_flags,
    device_named,
    load_card_table,
    set_cards,
    start_run,
)
from jace.loops.formats import deck_rules, limited_format
from jace.loops.schedules import RateScale, current_rates, scale_rates, warm_up_then_decay
from jace.loops.supervised import apply, score_loss
from jace.tracking import Running, log_update, save_checkpoint
from jace.tracking import writing as writing_run

__all__ = (
    "BuilderPoint",
    "DeckSettings",
    "ScorePoint",
    "main",
    "pretrain_deck_builder",
    "pretrain_deck_score",
)


@dataclass(frozen=True, slots=True)
class DeckSettings:
    """How long to train one deck model and how often to measure it.

    Each of the `steps` steps trains on `decks_per_step` human decks, drawn at
    random from the training decks, and `max_grad_norm` caps the norm of each
    step's gradient. The model is measured every `evaluate_every` steps and
    after the last step, on the first `evaluate_decks` held-out decks and on as
    many of the first training decks. Each measurement is logged with the mean
    training loss since the previous one.
    """

    steps: int
    decks_per_step: int
    evaluate_every: int
    evaluate_decks: int
    max_grad_norm: float


@dataclass(frozen=True, slots=True)
class ScorePoint:
    # `loss` is the mean training loss since the previous point. `log_loss` and `correlation` are the `score_fit` of
    # the held-out decks after `step` steps, and `training_log_loss` is the log loss on as many training decks.
    step: int
    loss: float
    log_loss: float
    correlation: float
    training_log_loss: float


@dataclass(frozen=True, slots=True)
class BuilderPoint:
    """One measurement of a deck builder pretraining run, after `step` steps.

    `loss` is the mean training loss since the previous point, with each deck
    weighted as in training. `log_loss` is the unweighted `build_log_loss` of
    the held-out human builds, and `training_log_loss` is the same on as many
    training builds. `overlap` and `lands` measure the builder's likeliest deck
    for each held-out pool. `overlap` is its `deck_overlap` with the human
    deck, and `lands` is its `land_total_match`.
    """

    step: int
    loss: float
    log_loss: float
    training_log_loss: float
    overlap: float
    lands: float


def pretrain_deck_score(
    model: DeckScore,
    decks: HumanDecks,
    *,
    optimizers: Sequence[Optimizer],
    rate_scale: RateScale,
    settings: DeckSettings,
    run: Running,
    generator: torch.Generator,
) -> tuple[ScorePoint, ...]:
    """Train `model` on the decks that are not held out and return its learning curve.

    A deck's target is its event's record. The loss is the binomial log loss of
    the deck's wins out of its games, so a deck with more games counts for
    more.

    At each step, each optimizer's learning rate is its rate at the call times
    `rate_scale(step)`. `generator` draws the batches and must be a CPU
    generator. Every point of the curve is also logged to `run` under
    "deck_score".
    """
    device = model.cards.embedding.device
    counts = decks.decks.to(device)  # uint8 [drafts, cards]
    wins = decks.wins.to(device).float()
    games = wins + decks.losses.to(device).float()
    training = (~decks.held_out).nonzero().squeeze(1)
    held_out = decks.held_out.nonzero().squeeze(1)[: settings.evaluate_decks]
    measured_training = training[: len(held_out)]
    starting_rates = current_rates(optimizers)
    curve: list[ScorePoint] = []
    loss_sum = 0.0
    for step in range(1, settings.steps + 1):
        chosen = training[torch.randint(len(training), (settings.decks_per_step,), generator=generator)].to(device)
        loss = score_loss(model(counts[chosen]), wins=wins[chosen], games=games[chosen])
        scale_rates(optimizers, starting_rates, factor=rate_scale(step))
        apply(loss, optimizers, max_grad_norm=settings.max_grad_norm)
        loss_sum += float(loss.detach())
        if step % settings.evaluate_every == 0 or step == settings.steps:
            fit = score_fit(model, decks, rows=held_out, device=device)
            point = ScorePoint(
                step=step,
                loss=loss_sum / (step - (curve[-1].step if curve else 0)),
                log_loss=fit.log_loss,
                correlation=fit.correlation,
                training_log_loss=score_fit(model, decks, rows=measured_training, device=device).log_loss,
            )
            metrics = {
                "loss": point.loss,
                "training_log_loss": point.training_log_loss,
                "held_out_log_loss": point.log_loss,
                "held_out_correlation": point.correlation,
            }
            log_update(run, model="deck_score", update=step, metrics=metrics)
            curve.append(point)
            loss_sum = 0.0
    return tuple(curve)


def pretrain_deck_builder(
    model: DeckBuilder,
    decks: HumanDecks,
    *,
    rules: BuildRules,
    spells_per_round: int,
    optimizers: Sequence[Optimizer],
    rate_scale: RateScale,
    settings: DeckSettings,
    run: Running,
    generator: torch.Generator,
) -> tuple[BuilderPoint, ...]:
    """Train `model` to write the human deck for a pool and return its learning curve.

    The builder learns by teacher forcing. Each step lays out `decks_per_step`
    human decks with `builds_from_decks` and lowers their negative
    log-probability under the builder's sampling at temperature 1, in rounds of
    `spells_per_round`. One call of the model scores every round of every deck.
    Each deck's log-probability is weighted by the share of games its event
    won, kept at or above `jace.data.RESULT_FLOOR` as the pick weights are, and
    the weights of a batch average one. Only decks that
    `allowed_by(rules, decks)` accepts are used, for training and for measuring.

    Each measured deck is laid out in one random order before the first step,
    so every point of the curve scores the same builds. Every point is also
    logged to `run` under "deck_builder".

    `optimizers` and `rate_scale` are as in `pretrain_deck_score`. `generator`
    draws the batches and the orders, and must be a CPU generator.
    """
    device = model.cards.embedding.device
    allowed = allowed_by(rules, decks)
    training = (allowed & ~decks.held_out).nonzero().squeeze(1)
    held_out = (allowed & decks.held_out).nonzero().squeeze(1)[: settings.evaluate_decks]
    wins = decks.wins.float()
    result = (wins / (wins + decks.losses.float()).clamp_min(1.0)).clamp_min(RESULT_FLOOR)  # [drafts]

    def lay_out(rows: Tensor) -> Builds:
        human = decks.decks[rows].to(device).float()
        return builds_from_decks(human, pools=decks.pools[rows].to(device).float(), rules=rules, generator=generator)

    def likeliest(pools: Tensor) -> Tensor:
        built, _ = sample_builds(
            model,
            pools,
            rules=rules,
            spells_per_round=spells_per_round,
            temperature=1.0,
            generator=generator,
            greedy=True,
        )
        return built.decks

    held_out_builds, training_builds = lay_out(held_out), lay_out(training[: len(held_out)])
    starting_rates = current_rates(optimizers)
    curve: list[BuilderPoint] = []
    loss_sum = 0.0
    for step in range(1, settings.steps + 1):
        chosen = training[torch.randint(len(training), (settings.decks_per_step,), generator=generator)]
        log_probs, _ = build_log_probs(
            model, lay_out(chosen), rules=rules, spells_per_round=spells_per_round, temperature=1.0
        )
        weights = result[chosen].to(device)
        loss = -(weights / weights.mean() * log_probs).mean()
        scale_rates(optimizers, starting_rates, factor=rate_scale(step))
        apply(loss, optimizers, max_grad_norm=settings.max_grad_norm)
        loss_sum += float(loss.detach())
        if step % settings.evaluate_every == 0 or step == settings.steps:
            built = likeliest(held_out_builds.pools)
            point = BuilderPoint(
                step=step,
                loss=loss_sum / (step - (curve[-1].step if curve else 0)),
                log_loss=build_log_loss(model, held_out_builds, rules=rules, spells_per_round=spells_per_round),
                training_log_loss=build_log_loss(
                    model, training_builds, rules=rules, spells_per_round=spells_per_round
                ),
                overlap=deck_overlap(built, held_out_builds.decks, rules=rules),
                lands=land_total_match(built, held_out_builds.decks, rules=rules),
            )
            metrics = {
                "loss": point.loss,
                "training_log_loss": point.training_log_loss,
                "held_out_log_loss": point.log_loss,
                "held_out_overlap": point.overlap,
                "held_out_lands": point.lands,
            }
            log_update(run, model="deck_builder", update=step, metrics=metrics)
            curve.append(point)
            loss_sum = 0.0
    return tuple(curve)


@dataclass(frozen=True, slots=True)
class _Arguments(RunArguments):
    device: str
    score_steps: int
    builder_steps: int
    embedding_run: Path
    score_decks_per_step: int
    builder_decks_per_step: int
    score_evaluate_every: int
    builder_evaluate_every: int
    builder_evaluate_decks: int
    spells_per_round: int
    width: int
    layers: int
    heads: int
    rate: float


def _arguments(argv: Sequence[str]) -> _Arguments:
    parser = argparse.ArgumentParser(description="Train the deck score and the deck builder on human decks.")
    add_run_flags(parser)
    parser.add_argument("--device", default="auto", help="cpu, mps or cuda; auto prefers cuda, then mps")
    parser.add_argument(
        "--score-steps",
        type=int,
        default=2000,
        help="training steps of the deck score; its held-out log loss is lowest near 2,000 and rises after",
    )
    parser.add_argument(
        "--builder-steps",
        type=int,
        default=9000,
        help="training steps of the deck builder; every run of 3,000 steps still lowered its held-out loss at the end",
    )
    parser.add_argument(
        "--embedding-run",
        type=Path,
        required=True,
        help="a drafter pretraining run folder whose newest card encoder the models read, frozen",
    )
    parser.add_argument("--score-decks-per-step", type=int, default=256)
    parser.add_argument("--builder-decks-per-step", type=int, default=128)
    parser.add_argument("--score-evaluate-every", type=int, default=250)
    parser.add_argument("--builder-evaluate-every", type=int, default=1000)
    parser.add_argument(
        "--builder-evaluate-decks",
        type=int,
        default=2000,
        help="held-out pools built per measurement; with 300 the overlap's noise hides its trend",
    )
    parser.add_argument(
        "--spells-per-round", type=int, default=6, help="spells or nonbasic lands the builder draws between two calls"
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--rate", type=float, default=5e-4, help="the peak learning rate of every optimizer")
    return _Arguments(**vars(parser.parse_args(argv)))


def main(argv: Sequence[str]) -> None:
    """Pretrain a deck score and a deck builder on one card set and save the run.

    The score trains first and the builder second, both on one card table.
    `jace.tracking` creates the run folder under `--run-dir`, and the run saves
    a "deck_score", a "deck_builder" and an "embedding" checkpoint in it. The
    curve file `<run id>.curve.json` is written beside the folder, with a
    baseline for each model.

    In the curve file, the score's baseline predicts the training decks'
    overall win rate for every held-out deck, with its log loss. The builder's
    curve is measured on its likeliest deck for each held-out pool, and the
    file adds the overlap of one deck sampled at temperature 1 from each of the
    same pools. The land head's baseline is the most common land total among
    the training decks the rules allow, with the share of the human decks of
    those held-out pools that have it.
    """
    arguments = _arguments(argv)
    device = device_named(arguments.device)
    torch.manual_seed(arguments.seed)
    data = read_set(arguments.set_code)
    cards = set_cards(data)
    decks = load_decks(
        data.games, pool=cards.pool, cache_dir=arguments.data_dir, held_out_share=arguments.held_out_share
    )
    rules = deck_rules(limited_format(arguments.set_code), cards)
    embedding, embedding_ref = load_card_table(arguments.embedding_run, pool=cards.pool)
    shape = {"width": arguments.width, "layers": arguments.layers, "heads": arguments.heads}
    score = DeckScore(embedding, **shape).to(device)
    builder = DeckBuilder(embedding, **shape).to(device)

    def optimizers(model: nn.Module) -> list[Optimizer]:
        return [torch.optim.Adam(own_parameters(model, embedding), lr=arguments.rate, betas=ADAM_BETAS)]

    held_out = decks.held_out.nonzero().squeeze(1)
    score_settings = DeckSettings(
        steps=arguments.score_steps,
        decks_per_step=arguments.score_decks_per_step,
        evaluate_every=arguments.score_evaluate_every,
        evaluate_decks=len(held_out),
        max_grad_norm=1.0,
    )
    builder_settings = DeckSettings(
        steps=arguments.builder_steps,
        decks_per_step=arguments.builder_decks_per_step,
        evaluate_every=arguments.builder_evaluate_every,
        evaluate_decks=arguments.builder_evaluate_decks,
        max_grad_norm=1.0,
    )
    clock = functools.partial(datetime.now, UTC)
    folder, curve_file = start_run(arguments, device=device, inputs=(embedding_ref,), clock=clock)
    generator = torch.Generator().manual_seed(arguments.seed)
    started = time.perf_counter()
    with writing_run(folder, clock=clock) as running:
        score_curve = pretrain_deck_score(
            score,
            decks,
            optimizers=optimizers(score),
            rate_scale=warm_up_then_decay(arguments.score_steps),
            settings=score_settings,
            run=running,
            generator=generator,
        )
        score_seconds = time.perf_counter() - started
        builder_curve = pretrain_deck_builder(
            builder,
            decks,
            rules=rules,
            spells_per_round=arguments.spells_per_round,
            optimizers=optimizers(builder),
            rate_scale=warm_up_then_decay(arguments.builder_steps),
            settings=builder_settings,
            run=running,
            generator=generator,
        )
        builder_seconds = time.perf_counter() - started - score_seconds
        save_checkpoint(running, model="deck_score", update=arguments.score_steps, module=score)
        save_checkpoint(running, model="deck_builder", update=arguments.builder_steps, module=builder)
        steps = arguments.score_steps + arguments.builder_steps
        save_checkpoint(running, model="embedding", update=steps, module=embedding)

    training = ~decks.held_out
    games = decks.wins.double() + decks.losses.double()
    mean_rate = decks.wins[training].double().sum() / games[training].sum()
    cpu = torch.device("cpu")
    overall = score_fit(lambda counts: mean_rate.logit().expand(len(counts)), decks, rows=held_out, device=cpu)
    allowed = allowed_by(rules, decks)
    built = (allowed & decks.held_out).nonzero().squeeze(1)[: arguments.builder_evaluate_decks]
    human = decks.decks[built].to(device).float()
    sampled, _ = sample_builds(
        builder,
        decks.pools[built].to(device).float(),
        rules=rules,
        spells_per_round=arguments.spells_per_round,
        temperature=1.0,
        generator=generator,
    )
    land_columns = [*rules.basics, *sorted(rules.nonbasic_lands)]
    commonest = int(decks.decks[allowed & training][:, land_columns].sum(dim=1).mode().values)
    commonest_match = float((human[:, land_columns].sum(dim=1) == commonest).float().mean())
    summary = {
        "run_id": folder.name,
        "device": device.type,
        "score": {
            "steps": arguments.score_steps,
            "seconds": score_seconds,
            "training_decks": int(training.sum()),
            "held_out_decks": len(held_out),
            "held_out_decks_with_a_rate": int((games[held_out] >= GAMES_FOR_A_RATE).sum()),
            "baselines": {"overall_win_rate": {"rate": float(mean_rate), "log_loss": overall.log_loss}},
            "curve": [asdict(point) for point in score_curve],
        },
        "builder": {
            "steps": arguments.builder_steps,
            "seconds": builder_seconds,
            "training_decks": int((allowed & training).sum()),
            "held_out_pools_built": len(built),
            "sampled_overlap": deck_overlap(sampled.decks, human, rules=rules),
            "baselines": {"most_common_land_total": {"lands": commonest, "lands_match": commonest_match}},
            "curve": [asdict(point) for point in builder_curve],
        },
    }
    curve_file.write_text(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(json.dumps({**summary, "curve_file": str(curve_file)}, indent=2))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])

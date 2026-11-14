"""Train the deck score and the deck builder on human decks, before any self-play.

The deck score learns each deck's chance of winning a game from the wins
and losses of the event it played. The deck builder learns the human's deck
for a pool by teacher forcing: the deck's spells and nonbasic lands in a
random order, cut into the rounds the builder samples in, the human's land
total and the human's basics split. A new random order each time the deck
is drawn teaches it to finish a deck from any partial one. Both read the
card vectors of a drafter pretraining run, frozen into a `CardTable`, so
neither can move what the drafter learned about the cards.

Basic lands follow `BuildRules`: a deck may hold any number of them, so the
builder never reads the basics counted in a human pool and splits the land
slots left with its basics head. Only human decks that the rules allow
teach or test the builder. The deck score reads every deck.

Run it with `uv run python -m jace.loops.pretrain_decks --help`. The card
set must already be downloaded with `uv run spells add <set>`.
"""

import argparse
import functools
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from jace.cards import own_parameters
from jace.data import RESULT_FLOOR, HumanDecks, load_decks, read_set
from jace.decks import BuildRules, DeckBuilder, DeckScore, Scorer, build_log_probs, builds_from_decks, sample_builds
from jace.loops.entry_points import (
    ADAM_BETAS,
    RunArguments,
    add_run_flags,
    card_inputs,
    device_named,
    load_card_table,
    set_cards,
    start_run,
)
from jace.loops.formats import deck_rules, limited_format
from jace.loops.schedules import RateScale, current_rates, scale_rates, warm_up_then_decay
from jace.tracking import Running, log_update, save_checkpoint
from jace.tracking import writing as writing_run
from jace.train import apply, score_loss

__all__ = (
    "BuilderPoint",
    "DeckSettings",
    "ScoreFit",
    "ScorePoint",
    "allowed_by",
    "deck_overlap",
    "main",
    "pretrain_deck_builder",
    "pretrain_deck_score",
    "score_fit",
)

# A deck's actual win rate is compared with its predicted one only when it played this many games.
_GAMES_FOR_A_RATE = 5
# Held-out decks scored in one call.
_SCORE_CHUNK = 1024
# Keeps the correlation finite when every prediction is the same.
_SMALLEST_CORRELATION_SCALE = 1e-12

# Takes float [n, cards] pools and returns float [n, cards], one deck built from each.
type WriteDecks = Callable[[Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class DeckSettings:
    """How long to train one deck model and how often to measure it.

    Each of `steps` steps trains on `decks_per_step` human decks drawn at
    random from the training decks. Every `evaluate_every` steps, and after
    the last step, the model is measured on the first `evaluate_decks`
    held-out decks, and the result is logged with the mean training loss
    since the previous measurement.
    """

    steps: int
    decks_per_step: int
    evaluate_every: int
    evaluate_decks: int
    max_grad_norm: float


@dataclass(frozen=True, slots=True)
class ScoreFit:
    """How well a scorer predicts the results of `decks` decks.

    `log_loss` is the binomial log loss for each game the decks played, in
    nats, so a scorer that says 0.5 for every deck gets 0.693. `correlation`
    is Pearson's, between the predicted and the actual win rate, over the
    decks that played at least five games. It is zero when every prediction
    is the same.
    """

    log_loss: float
    correlation: float
    decks: int


@dataclass(frozen=True, slots=True)
class ScorePoint:
    # `loss` is the mean training loss since the previous point; the rest is the held-out fit at `step`.
    step: int
    loss: float
    log_loss: float
    correlation: float


@dataclass(frozen=True, slots=True)
class BuilderPoint:
    # `loss` is the mean training loss since the previous point. `overlap` is `deck_overlap` on held-out pools.
    step: int
    loss: float
    overlap: float


def allowed_by(rules: BuildRules, decks: HumanDecks) -> Tensor:
    # Returns bool [drafts]: the human decks with exactly `rules.size` cards and a land count the rules allow.
    counts = decks.decks.long()
    lands = counts[:, [*rules.basics, *sorted(rules.nonbasic_lands)]].sum(dim=1)
    return (counts.sum(dim=1) == rules.size) & (lands >= rules.min_lands) & (lands <= rules.max_lands)


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

    A deck's target is its event's record: the loss is the binomial log
    loss of the wins out of the games played, so a deck with more games
    counts for more. Each of `optimizers` has its learning rate set to its
    rate at the call times `rate_scale(step)`. `generator` draws the batches
    and lives on the CPU. Every point is also logged to `run` under
    "deck_score".
    """
    device = model.cards.embedding.device
    counts = decks.decks.to(device)  # uint8 [drafts, cards]
    wins = decks.wins.to(device).float()
    games = wins + decks.losses.to(device).float()
    training = (~decks.held_out).nonzero().squeeze(1)
    held_out = decks.held_out.nonzero().squeeze(1)[: settings.evaluate_decks]
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
            steps_since_point = step - (curve[-1].step if curve else 0)
            point = ScorePoint(step, loss_sum / steps_since_point, fit.log_loss, fit.correlation)
            metrics = {"loss": point.loss, "held_out_log_loss": point.log_loss}
            log_update(
                run, model="deck_score", update=step, metrics={**metrics, "held_out_correlation": fit.correlation}
            )
            curve.append(point)
            loss_sum = 0.0
    return tuple(curve)


def score_fit(score: Scorer, decks: HumanDecks, *, rows: Tensor, device: torch.device) -> ScoreFit:
    # rows: int64 [n], the decks to score. `score` returns the logit of winning a game. No gradients are kept.
    logits: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(rows), _SCORE_CHUNK):
            chosen = rows[start : start + _SCORE_CHUNK]
            logits.append(score(decks.decks[chosen].to(device).float()).cpu())
    predicted = torch.cat(logits).double()  # [n]
    wins = decks.wins[rows].double()
    games = wins + decks.losses[rows].double()
    log_loss = score_loss(predicted, wins=wins, games=games)
    enough = games >= _GAMES_FOR_A_RATE
    predicted_centred = predicted.sigmoid()[enough] - predicted.sigmoid()[enough].mean()
    actual_centred = (wins / games.clamp_min(1.0))[enough] - (wins / games.clamp_min(1.0))[enough].mean()
    # Pearson's correlation: the sum of products of the centred values over this scale.
    scale = (predicted_centred.pow(2).sum() * actual_centred.pow(2).sum()).sqrt().clamp_min(_SMALLEST_CORRELATION_SCALE)
    return ScoreFit(
        log_loss=float(log_loss), correlation=float((predicted_centred * actual_centred).sum() / scale), decks=len(rows)
    )


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

    Each step lays `decks_per_step` human decks out with `builds_from_decks`
    and lowers their negative log-probability under the builder's sampling
    at temperature 1, in rounds of `spells_per_round`, so one call of the
    model learns every round of every deck. A deck's log-probability weighs
    the share of games its event won, kept at or above
    `jace.data.RESULT_FLOOR` as the pick weights are, and the weights of a
    batch average one. Only decks that `allowed_by(rules, decks)` accepts
    are used, for training and measuring. The measure is `deck_overlap` of
    the builder's likeliest deck for each held-out pool.

    `optimizers` and `rate_scale` are as in `pretrain_deck_score`.
    `generator` draws the batches and the orders and lives on the CPU. Every
    point is also logged to `run` under "deck_builder".
    """
    device = model.cards.embedding.device
    allowed = allowed_by(rules, decks)
    training = (allowed & ~decks.held_out).nonzero().squeeze(1)
    held_out = (allowed & decks.held_out).nonzero().squeeze(1)[: settings.evaluate_decks]
    wins = decks.wins.float()
    result = (wins / (wins + decks.losses.float()).clamp_min(1.0)).clamp_min(RESULT_FLOOR)  # [drafts]

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

    starting_rates = current_rates(optimizers)
    curve: list[BuilderPoint] = []
    loss_sum = 0.0
    for step in range(1, settings.steps + 1):
        chosen = training[torch.randint(len(training), (settings.decks_per_step,), generator=generator)]
        builds = builds_from_decks(
            decks.decks[chosen].to(device).float(),
            pools=decks.pools[chosen].to(device).float(),
            rules=rules,
            generator=generator,
        )
        log_probs, _ = build_log_probs(model, builds, rules=rules, spells_per_round=spells_per_round, temperature=1.0)
        weights = result[chosen].to(device)
        loss = -(weights / weights.mean() * log_probs).mean()
        scale_rates(optimizers, starting_rates, factor=rate_scale(step))
        apply(loss, optimizers, max_grad_norm=settings.max_grad_norm)
        loss_sum += float(loss.detach())
        if step % settings.evaluate_every == 0 or step == settings.steps:
            overlap = deck_overlap(likeliest, decks, rows=held_out, rules=rules, device=device)
            steps_since_point = step - (curve[-1].step if curve else 0)
            point = BuilderPoint(step, loss_sum / steps_since_point, overlap)
            log_update(
                run, model="deck_builder", update=step, metrics={"loss": point.loss, "held_out_overlap": overlap}
            )
            curve.append(point)
            loss_sum = 0.0
    return tuple(curve)


def deck_overlap(
    write: WriteDecks, decks: HumanDecks, *, rows: Tensor, rules: BuildRules, device: torch.device
) -> float:
    """Measure how much of each human deck `write` rebuilds from the same pool.

    `rows` is int64 `[n]`, the decks to rebuild. `write` builds one deck
    from each of their human pools, all in one call. A deck's share is the
    number of the human deck's non-basic cards that the written deck also
    holds, counting copies, over the human deck's non-basic cards. Returns
    the mean share. Basics are left out because they are free to add, so
    matching them says little.
    """
    not_basic = torch.ones(decks.decks.shape[1], dtype=torch.bool, device=device)
    not_basic[list(rules.basics)] = False
    with torch.no_grad():
        built = write(decks.pools[rows].to(device).float())[:, not_basic]
    human = decks.decks[rows].to(device).float()[:, not_basic]
    return float((torch.minimum(built, human).sum(dim=1) / human.sum(dim=1)).mean())


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
    parser.add_argument("--device", default="auto", help="cpu, mps or cuda; auto takes mps when it is available")
    parser.add_argument("--score-steps", type=int, required=True, help="training steps of the deck score")
    parser.add_argument("--builder-steps", type=int, required=True, help="training steps of the deck builder")
    parser.add_argument(
        "--embedding-run",
        type=Path,
        required=True,
        help="a drafter pretraining run folder whose newest card encoder the models read, frozen",
    )
    parser.add_argument("--score-decks-per-step", type=int, default=256)
    parser.add_argument("--builder-decks-per-step", type=int, default=128)
    parser.add_argument("--score-evaluate-every", type=int, default=250)
    parser.add_argument("--builder-evaluate-every", type=int, default=500)
    parser.add_argument("--builder-evaluate-decks", type=int, default=300, help="held-out pools built per measurement")
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

    The score trains first and the builder second, on one card table.
    The run folder is created under `--run-dir` by `jace.tracking` and gets
    a "deck_score", a "deck_builder" and an "embedding" checkpoint. The
    curves are written beside it as `<run id>.curve.json`, with the log
    loss of predicting the training decks' overall win rate for every
    held-out deck as the score's baseline. The builder's curve measures its
    likeliest deck, and the summary adds the overlap of one deck sampled at
    temperature 1 from each of the same held-out pools.
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
    embedding, embedding_ref = load_card_table(
        arguments.embedding_run, inputs=card_inputs(data, cards, data_dir=arguments.data_dir)
    )
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

    def sampled(pools: Tensor) -> Tensor:
        builds, _ = sample_builds(
            builder,
            pools,
            rules=rules,
            spells_per_round=arguments.spells_per_round,
            temperature=1.0,
            generator=generator,
        )
        return builds.decks

    summary = {
        "run_id": folder.name,
        "device": device.type,
        "score": {
            "steps": arguments.score_steps,
            "seconds": score_seconds,
            "training_decks": int(training.sum()),
            "held_out_decks": len(held_out),
            "held_out_decks_with_a_rate": int((games[held_out] >= _GAMES_FOR_A_RATE).sum()),
            "baselines": {"overall_win_rate": {"rate": float(mean_rate), "log_loss": overall.log_loss}},
            "curve": [asdict(point) for point in score_curve],
        },
        "builder": {
            "steps": arguments.builder_steps,
            "seconds": builder_seconds,
            "training_decks": int((allowed & training).sum()),
            "held_out_pools_built": len(built),
            "sampled_overlap": deck_overlap(sampled, decks, rows=built, rules=rules, device=device),
            "curve": [asdict(point) for point in builder_curve],
        },
    }
    curve_file.write_text(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(json.dumps({**summary, "curve_file": str(curve_file)}, indent=2))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])

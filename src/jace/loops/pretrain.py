"""Train the drafter and the card encoder to copy human picks, before any self-play.

Each step scores every pick of a batch of whole human drafts in one pass
and lowers the weighted cross-entropy of the card the human took, over the
cards in the pack. How well it copies is measured on held-out drafts as the
share of picks where the human's card is the model's first choice, or among
its first two or three. That measure counts every recorded pick equally and
ignores the sample weights. The two penalties of an earlier human-trained
drafter, against taking rares and against costly cards, are hand-added
priors: they are off unless `PickPriors` is passed.

The card encoder reads each card's text, engine program and printed facts,
and the pick loss trains it with the drafter, each with its own optimizer.
It is saved as the run's "embedding" checkpoint, which `pretrain_decks` and
self-play read frozen. Run it with
`uv run python -m jace.loops.pretrain --help`. The card set must already be
downloaded with `uv run spells add <set>`, and the first run on a data
folder without cached text vectors downloads the text model from Hugging
Face.
"""

import argparse
import functools
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import torch
from torch import Tensor
from torch.optim import Optimizer

from jace.cards import FIRST_PROGRAM_TOKEN, CardEncoder, Rarity, own_parameters
from jace.data import HumanDrafts, SetCards, card_table, load_drafts, read_set
from jace.draft import DrafterModel
from jace.engine.manabot import engine_cards, program_vocabulary
from jace.loops.entry_points import (
    ADAM_BETAS,
    RunArguments,
    add_card_flags,
    add_run_flags,
    card_inputs,
    device_named,
    start_run,
)
from jace.loops.schedules import RateScale, current_rates, scale_rates, warm_up_then_decay
from jace.tracking import Running, log_update, save_checkpoint, writing
from jace.train import apply, choice_loss

__all__ = (
    "CurvePoint",
    "NonFiniteScoreError",
    "PickAccuracy",
    "PickPriors",
    "PickScorer",
    "PretrainSettings",
    "main",
    "pick_accuracy",
    "pretrain_drafter",
)

# (seen [b, t, cards] 0/1, picks int64 [b, t]) -> float [b, t, cards]: a score for every card at
# every step, finite for a card in the pack and -inf for every other. `DrafterModel.whole_draft` is one.
type PickScorer = Callable[[Tensor, Tensor], Tensor]

_TOP_RANKS = (1, 2, 3)
_EVALUATE_DRAFTS = 512


@dataclass(frozen=True, slots=True)
class PretrainSettings:
    """How long to train and how often to measure.

    Each of `steps` steps trains on `drafts_per_step` whole drafts drawn at
    random from the training drafts. Every `evaluate_every` steps, and
    after the last step, the held-out drafts are scored `evaluate_drafts`
    at a time, and the result is logged with the mean training loss since
    the previous measurement.
    """

    steps: int
    drafts_per_step: int
    evaluate_every: int
    evaluate_drafts: int
    max_grad_norm: float


@dataclass(frozen=True, slots=True)
class PickPriors:
    """Two hand-added penalties from an earlier human-trained drafter, off unless passed.

    `rare` is float `[cards]`, one for rare and mythic cards. When the human
    took a card that is not rare, the model pays `rare_weight` times the
    chance it gives to rare cards. `mana_value` is float `[cards]`. The
    model pays `mana_weight` times how far the expected mana value of its
    pick exceeds the human's pick, less `mana_margin`. Neither has been
    shown to help this model, so compare a run with and without them.
    """

    rare: Tensor
    rare_weight: float
    mana_value: Tensor
    mana_weight: float
    mana_margin: float


@dataclass(frozen=True, slots=True)
class PickAccuracy:
    # The share of `picks` recorded picks where the human's card was among the scorer's best 1, 2 and 3.
    top1: float
    top2: float
    top3: float
    picks: int


@dataclass(frozen=True, slots=True)
class CurvePoint:
    # `loss` is the mean training loss since the previous point; the rest is held-out accuracy at `step`.
    step: int
    loss: float
    top1: float
    top2: float
    top3: float


class NonFiniteScoreError(ValueError):
    def __init__(self) -> None:
        super().__init__("the scorer gave a card in the pack a score that is NaN or infinite, so it ranks nothing")


def pretrain_drafter(
    model: DrafterModel,
    drafts: HumanDrafts,
    *,
    optimizers: Sequence[Optimizer],
    rate_scale: RateScale,
    settings: PretrainSettings,
    run: Running,
    generator: torch.Generator,
    priors: PickPriors | None = None,
) -> tuple[CurvePoint, ...]:
    """Train `model` on the drafts that are not held out and return its learning curve.

    `optimizers` holds the drafter's optimizer and the card encoder's, and
    every step moves both. Each one's learning rate is its rate at the call
    times `rate_scale(step)`. `generator` draws the training batches and
    lives on the CPU. Every point of the curve is also logged to `run` under
    "drafter". The last point holds the final held-out accuracy.
    """
    device = model.embedding.device
    cards = model.embedding.cards
    packs, picks, weights = drafts.packs.to(device), drafts.picks.to(device).long(), drafts.weights.to(device)
    training = (~drafts.held_out).nonzero().squeeze(1)
    held_out = drafts.held_out.nonzero().squeeze(1)
    starting_rates = current_rates(optimizers)
    curve: list[CurvePoint] = []
    loss_sum = 0.0
    for step in range(1, settings.steps + 1):
        chosen = training[torch.randint(len(training), (settings.drafts_per_step,), generator=generator)].to(device)
        real = weights[chosen] > 0  # [b, t]
        logits = model.whole_draft(_seen(packs[chosen], cards=cards), picks[chosen])[real]  # [m, cards]
        picked, pick_weights = picks[chosen][real], weights[chosen][real]
        pick_weights = pick_weights / pick_weights.mean()
        loss = choice_loss(logits, picked, weights=pick_weights)
        if priors is not None:
            loss = loss + _prior_loss(logits, picked, weights=pick_weights, priors=priors)
        scale_rates(optimizers, starting_rates, factor=rate_scale(step))
        apply(loss, optimizers, max_grad_norm=settings.max_grad_norm)
        loss_sum += float(loss.detach())
        if step % settings.evaluate_every == 0 or step == settings.steps:
            accuracy = pick_accuracy(
                model.whole_draft, drafts, rows=held_out, cards=cards, device=device, settings=settings
            )
            steps_since_point = step - (curve[-1].step if curve else 0)
            point = CurvePoint(step, loss_sum / steps_since_point, accuracy.top1, accuracy.top2, accuracy.top3)
            metrics = {"loss": point.loss, "held_out_top1": point.top1, "held_out_top2": point.top2}
            log_update(run, model="drafter", update=step, metrics={**metrics, "held_out_top3": point.top3})
            curve.append(point)
            loss_sum = 0.0
    return tuple(curve)


def pick_accuracy(
    score: PickScorer,
    drafts: HumanDrafts,
    *,
    rows: Tensor,
    cards: int,
    device: torch.device,
    settings: PretrainSettings,
) -> PickAccuracy:
    """Measure how often `score` ranks the human's card first, in the first two and in the first three.

    `rows` is int64 `[n]`, the drafts to score, and `cards` is the number of
    cards in the pool. Every recorded pick counts once, whatever its sample
    weight; a first pick rebuilt from the pool is not counted. A pack with
    fewer cards than the rank asked for always counts as a hit, so the last
    pick of every round does. No gradients are kept.

    A card that scores the same as the human's card counts as ranked above
    it, so a scorer that cannot tell cards apart gets no credit for them.
    Raises `NonFiniteScoreError` when a card in a pack gets a score that is
    NaN or infinite, which is what a model that diverged returns.
    """
    hits = torch.zeros(len(_TOP_RANKS), dtype=torch.int64)
    counted = 0
    with torch.no_grad():
        for start in range(0, len(rows), settings.evaluate_drafts):
            chosen = rows[start : start + settings.evaluate_drafts]
            picks = drafts.picks[chosen].to(device).long()
            real = drafts.weights[chosen].to(device) > 0  # [b, t]
            seen = _seen(drafts.packs[chosen].to(device), cards=cards)
            scores = score(seen, picks)  # [b, t, cards]
            if not bool(scores[seen > 0].isfinite().all()):
                raise NonFiniteScoreError
            # [m], the other cards ranked level with the pick or above it. The pick is level with itself.
            ahead = (scores >= scores.gather(2, picks[:, :, None])).sum(dim=-1)[real] - 1
            hits += torch.stack([(ahead < top).sum() for top in _TOP_RANKS]).cpu()
            counted += int(real.sum())
    top1, top2, top3 = (hits / max(counted, 1)).tolist()
    return PickAccuracy(top1=top1, top2=top2, top3=top3, picks=counted)


def _seen(packs: Tensor, *, cards: int) -> Tensor:
    # packs: int16 [b, t, pack_size] card ids, -1 past a pack's end. Returns float [b, t, cards] 0/1.
    # Ids shift up by one so the -1 padding lands in column 0, which is then dropped.
    with_padding = torch.zeros(*packs.shape[:2], cards + 1, device=packs.device)
    with_padding.scatter_(2, packs.long() + 1, 1.0)
    return with_padding[:, :, 1:]


def _prior_loss(logits: Tensor, picked: Tensor, *, weights: Tensor, priors: PickPriors) -> Tensor:
    # logits: [m, cards] with cards outside the pack at -inf. picked: int64 [m]. weights: [m].
    chances = logits.softmax(dim=-1)
    rare_penalty = (1.0 - priors.rare[picked]) * (chances * priors.rare).sum(dim=-1)
    mana_excess = (chances * priors.mana_value).sum(dim=-1) - priors.mana_value[picked] + priors.mana_margin
    return (weights * (priors.rare_weight * rare_penalty + priors.mana_weight * mana_excess.clamp_min(0.0))).mean()


@dataclass(frozen=True, slots=True)
class _Arguments(RunArguments):
    device: str
    steps: int
    drafts_per_step: int
    evaluate_every: int
    width: int
    layers: int
    heads: int
    card_width: int
    card_layers: int
    card_heads: int
    rate: float
    rare_penalty: float
    mana_penalty: float
    mana_margin: float


def _arguments(argv: Sequence[str]) -> _Arguments:
    parser = argparse.ArgumentParser(description="Train the drafter on human picks from 17Lands.")
    add_run_flags(parser)
    add_card_flags(parser)
    parser.add_argument("--device", default="auto", help="cpu, mps or cuda; auto takes mps when it is available")
    parser.add_argument("--steps", type=int, required=True, help="training steps")
    parser.add_argument("--drafts-per-step", type=int, default=128)
    parser.add_argument("--evaluate-every", type=int, default=500)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--rate", type=float, default=5e-4, help="the peak learning rate of both optimizers")
    parser.add_argument("--rare-penalty", type=float, default=0.0, help="weight of the rare-drafting prior, 0 for off")
    parser.add_argument("--mana-penalty", type=float, default=0.0, help="weight of the mana-cost prior, 0 for off")
    parser.add_argument("--mana-margin", type=float, default=1.0)
    return _Arguments(**vars(parser.parse_args(argv)))


def _priors(arguments: _Arguments, cards: SetCards, *, device: torch.device) -> PickPriors | None:
    if arguments.rare_penalty <= 0 and arguments.mana_penalty <= 0:
        return None
    rare = torch.zeros(len(cards.pool))
    rare[[*cards.pool.of_rarity(Rarity.RARE), *cards.pool.of_rarity(Rarity.MYTHIC)]] = 1.0
    return PickPriors(
        rare=rare.to(device),
        rare_weight=arguments.rare_penalty,
        mana_value=cards.mana_value.to(device),
        mana_weight=arguments.mana_penalty,
        mana_margin=arguments.mana_margin,
    )


def _baselines(
    drafts: HumanDrafts, cards: SetCards, *, device: torch.device, settings: PretrainSettings
) -> tuple[dict[str, float], PickAccuracy]:
    """Score two pickers that learn nothing on the held-out picks.

    The first is a uniformly random pick, computed exactly from the pack
    sizes. The second always takes the pack's card with the lowest average
    pick position. That position is averaged over every draft of the set,
    the held-out ones included, so the second baseline has seen a summary
    of the picks it is scored on.
    """
    held_out = drafts.held_out.nonzero().squeeze(1)
    # A card nobody took has no average pick position and ranks after every card that has one.
    never_taken = float(drafts.picks.shape[1])
    eagerness = -cards.ata.nan_to_num(nan=never_taken).to(device)  # [cards]

    def earliest_taken(seen: Tensor, _picks: Tensor) -> Tensor:
        outside = seen == 0
        return eagerness.expand_as(seen).masked_fill(outside, float("-inf"))

    by_ata = pick_accuracy(
        earliest_taken, drafts, rows=held_out, cards=len(cards.pool), device=device, settings=settings
    )
    real = drafts.weights[held_out] > 0
    sizes = (drafts.packs[held_out] >= 0).sum(dim=-1)[real].float()  # [m] cards in each scored pack
    uniform = {f"top{top}": float((sizes.clamp_max(top) / sizes).mean()) for top in _TOP_RANKS}
    return uniform, by_ata


def main(argv: Sequence[str]) -> None:
    """Pretrain a drafter on one card set and save its record, checkpoints and learning curve.

    The run folder is created under `--run-dir` by `jace.tracking` and gets
    a "drafter" and an "embedding" checkpoint. The curve is written beside
    it as `<run id>.curve.json`, with the device, the wall time and two
    baselines scored on the same held-out picks: a uniformly random pick,
    computed exactly, and always taking the pack's card with the lowest
    average pick position.
    """
    arguments = _arguments(argv)
    device = device_named(arguments.device)
    torch.manual_seed(arguments.seed)
    data = read_set(arguments.set_code)
    programs = {card.name: card.program for card in engine_cards()}
    cards = card_table(data.attributes, data.statistics, programs=programs)
    drafts = load_drafts(
        data.picks, pool=cards.pool, cache_dir=arguments.data_dir, held_out_share=arguments.held_out_share
    )
    encoder = CardEncoder(
        card_inputs(data, cards, data_dir=arguments.data_dir),
        width=arguments.card_width,
        program_vocabulary=FIRST_PROGRAM_TOKEN + len(program_vocabulary()),
        program_layers=arguments.card_layers,
        heads=arguments.card_heads,
    )
    model = DrafterModel(
        encoder, width=arguments.width, layers=arguments.layers, heads=arguments.heads, picks=drafts.picks.shape[1]
    ).to(device)
    optimizers = [
        torch.optim.Adam(own_parameters(model, encoder), lr=arguments.rate, betas=ADAM_BETAS),
        torch.optim.Adam(encoder.parameters(), lr=arguments.rate, betas=ADAM_BETAS),
    ]
    settings = PretrainSettings(
        steps=arguments.steps,
        drafts_per_step=arguments.drafts_per_step,
        evaluate_every=arguments.evaluate_every,
        evaluate_drafts=_EVALUATE_DRAFTS,
        max_grad_norm=1.0,
    )
    clock = functools.partial(datetime.now, UTC)
    folder, curve_file = start_run(arguments, device=device, inputs=(), clock=clock)
    started = time.perf_counter()
    with writing(folder, clock=clock) as run:
        curve = pretrain_drafter(
            model,
            drafts,
            optimizers=optimizers,
            rate_scale=warm_up_then_decay(arguments.steps),
            settings=settings,
            run=run,
            generator=torch.Generator().manual_seed(arguments.seed),
            priors=_priors(arguments, cards, device=device),
        )
        seconds = time.perf_counter() - started
        save_checkpoint(run, model="drafter", update=arguments.steps, module=model)
        save_checkpoint(run, model="embedding", update=arguments.steps, module=encoder)

    uniform, by_ata = _baselines(drafts, cards, device=device, settings=settings)
    # A step trains on the picks with a weight above zero, which leaves out a first pick rebuilt from the pool.
    picks_per_draft = float((drafts.weights[~drafts.held_out] > 0).sum(dim=1).float().mean())
    summary = {
        "run_id": folder.name,
        "device": device.type,
        "steps": arguments.steps,
        "seconds": seconds,
        "picks_per_second": arguments.steps * arguments.drafts_per_step * picks_per_draft / seconds,
        "training_drafts": int((~drafts.held_out).sum()),
        "held_out_drafts": int(drafts.held_out.sum()),
        "held_out_picks": by_ata.picks,
        "baselines": {
            "uniform": uniform,
            "best_ata": {f"top{top}": getattr(by_ata, f"top{top}") for top in _TOP_RANKS},
        },
        "curve": [asdict(point) for point in curve],
    }
    curve_file.write_text(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(json.dumps({**summary, "curve": summary["curve"][-1:], "curve_file": str(curve_file)}, indent=2))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])

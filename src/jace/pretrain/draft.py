"""Train the drafter and the card encoder to copy human picks, before any self-play.

Each step scores every pick of a batch of whole human drafts in one pass. It
lowers the cross-entropy of the card the human took, over the cards in the
pack, with each pick weighted by its sample weight.

The model is measured on held-out drafts by how often the human's card is its
first choice, or among its first two or three, and by the log loss of the
human's card. These measures count every recorded pick equally, whatever its
sample weight. Taking the same measures on as many training drafts shows when
the model starts to fit its training drafts better than drafts it has not seen.

RyanSaxe/mtg, an earlier drafter trained on human picks, added two penalties to
its loss. One penalizes taking rare cards and the other taking costly cards.
Here they are hand-added priors, off unless `pretrain_drafter` is given a
`PickPriors`.

The card encoder reads each card's engine program. The pick loss trains it
together with the drafter, each with its own optimizer. The encoder is saved as
the run's "embedding" checkpoint, which `jace.pretrain.decks` and the self-play
loops read frozen.

Run it with `uv run python -m jace.pretrain.draft --help`. The card set must
already be downloaded with `uv run spells add <set>`.
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
from jace.data import HumanDrafts, SetCards, load_drafts, program_tokens, read_set
from jace.data.catalog import program_vocabulary
from jace.draft import DrafterModel
from jace.modules import device_named
from jace.pretrain.common import (
    ADAM_BETAS,
    RunArguments,
    add_card_flags,
    add_run_flags,
    set_cards,
    start_run,
)
from jace.pretrain.formats import limited_format
from jace.pretrain.schedules import RateScale, current_rates, scale_rates, warm_up_then_decay
from jace.pretrain.supervised import apply, choice_loss
from jace.storage.pretrain import Running, log_update, save_checkpoint, writing

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

# A pick scorer takes `seen`, float [b, t, cards] with a one for each card in the pack at each step, and `picks`,
# int64 [b, t], the card taken at each step. It returns float [b, t, cards], a score for every card at every step,
# which must be finite for a card in the pack and -inf for every other card. `DrafterModel.whole_draft` is one.
type PickScorer = Callable[[Tensor, Tensor], Tensor]

_TOP_RANKS = (1, 2, 3)
_EVALUATE_DRAFTS = 512


@dataclass(frozen=True, slots=True)
class PretrainSettings:
    """How long to train and how often to measure.

    Each of the `steps` steps trains on `drafts_per_step` whole drafts, drawn
    at random from the training drafts, and `max_grad_norm` caps the norm of
    each step's gradient. The model is measured every `evaluate_every` steps
    and after the last step. A measurement scores the held-out drafts
    `evaluate_drafts` at a time, and the result is logged with the mean
    training loss since the previous measurement.
    """

    steps: int
    drafts_per_step: int
    evaluate_every: int
    evaluate_drafts: int
    max_grad_norm: float


@dataclass(frozen=True, slots=True)
class PickPriors:
    """Two hand-added penalties from an earlier human-trained drafter, off unless passed.

    `rare` is float `[cards]`, one for each rare or mythic card and zero for
    every other card. When the human took a card that is not rare, the loss
    adds `rare_weight` times the chance the model gives to rare cards.

    `mana_value` is float `[cards]`, each card's mana value. The mana penalty
    compares the expected mana value of the model's pick with the mana value
    of the human's card. It is zero while the expected value is at least
    `mana_margin` lower, and it grows by `mana_weight` for each point of mana
    value above that.

    Both penalties are weighted like the pick loss. Neither has been shown to
    help this model, so compare a run with and without them.
    """

    rare: Tensor
    rare_weight: float
    mana_value: Tensor
    mana_weight: float
    mana_margin: float


@dataclass(frozen=True, slots=True)
class PickAccuracy:
    # Measured over `picks` recorded picks. `top1`, `top2` and `top3` are the shares of picks where the human's card
    # was among the scorer's best 1, 2 and 3 cards. `log_loss` is the mean negative log-probability of the human's
    # card under the softmax of the scores over the pack, in nats.
    top1: float
    top2: float
    top3: float
    log_loss: float
    picks: int


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """One measurement of a drafter pretraining run, after `step` steps.

    `loss` is the mean training loss since the previous point, with each pick
    weighted as in training. `top1`, `top2`, `top3` and `log_loss` are the
    `PickAccuracy` measures on the held-out drafts. `training_top1` and
    `training_log_loss` are the same measures on as many training drafts, so
    the two log losses can be compared directly.
    """

    step: int
    loss: float
    top1: float
    top2: float
    top3: float
    log_loss: float
    training_top1: float
    training_log_loss: float


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

    `optimizers` contains the drafter's optimizer and the card encoder's, and
    every step updates both. At each step, each optimizer's learning rate is
    its rate at the call times `rate_scale(step)`. `generator` draws the
    training batches and must be a CPU generator. When `priors` is given, the
    loss also adds its two penalties.

    The model is measured every `settings.evaluate_every` steps and after the
    last step, on the held-out drafts and on the first training drafts, as
    many as are held out. It is measured in eval mode, so without its dropout,
    and is left in training mode. Every point of the curve is also logged to
    `run` under "drafter", and the last point contains the final held-out
    accuracy.
    """
    device = model.embedding.device
    cards = model.embedding.cards
    packs, picks, weights = drafts.packs.to(device), drafts.picks.to(device).long(), drafts.weights.to(device)
    training = (~drafts.held_out).nonzero().squeeze(1)
    held_out = drafts.held_out.nonzero().squeeze(1)
    measured_training = training[: len(held_out)]
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
            model.eval()  # turns dropout off while measuring
            accuracy, on_training = (
                pick_accuracy(model.whole_draft, drafts, rows=rows, cards=cards, device=device, settings=settings)
                for rows in (held_out, measured_training)
            )
            model.train()
            point = CurvePoint(
                step=step,
                loss=loss_sum / (step - (curve[-1].step if curve else 0)),
                top1=accuracy.top1,
                top2=accuracy.top2,
                top3=accuracy.top3,
                log_loss=accuracy.log_loss,
                training_top1=on_training.top1,
                training_log_loss=on_training.log_loss,
            )
            metrics = {
                "loss": point.loss,
                "held_out_top1": point.top1,
                "held_out_top2": point.top2,
                "held_out_top3": point.top3,
                "held_out_log_loss": point.log_loss,
                "training_top1": point.training_top1,
                "training_log_loss": point.training_log_loss,
            }
            log_update(run, model="drafter", update=step, metrics=metrics)
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
    weight. A first pick rebuilt from the pool is not counted. The log loss
    treats the scores as logits over the pack. No gradients are kept.

    A pick from a pack of k cards or fewer always counts as a hit at rank k, so
    the last pick of every round is a hit at every rank. A card that scores the
    same as the human's card counts as ranked above it, so a scorer that gives
    every card the same score gets no credit for them.

    Raises `NonFiniteScoreError` when a card in a pack gets a score that is
    NaN or infinite, which is what a model that diverged returns.
    """
    hits = torch.zeros(len(_TOP_RANKS), dtype=torch.int64)
    counted = 0
    surprise = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), settings.evaluate_drafts):
            chosen = rows[start : start + settings.evaluate_drafts]
            picks = drafts.picks[chosen].to(device).long()
            real = drafts.weights[chosen].to(device) > 0  # [b, t]
            seen = _seen(drafts.packs[chosen].to(device), cards=cards)
            scores = score(seen, picks)  # [b, t, cards]
            if not bool(scores[seen > 0].isfinite().all()):
                raise NonFiniteScoreError
            # `ahead` is int64 [m], how many other cards score at least as high as the human's card at each real pick.
            # The `- 1` leaves out the human's card itself.
            ahead = (scores >= scores.gather(2, picks[:, :, None])).sum(dim=-1)[real] - 1
            hits += torch.stack([(ahead < top).sum() for top in _TOP_RANKS]).cpu()
            counted += int(real.sum())
            # A padded step's row is -inf everywhere, so its log-softmax is NaN. Indexing with `real` drops that row
            # before the sum.
            surprise -= float(scores.log_softmax(dim=-1).gather(2, picks[:, :, None]).squeeze(2)[real].sum())
    top1, top2, top3 = (hits / max(counted, 1)).tolist()
    return PickAccuracy(top1=top1, top2=top2, top3=top3, log_loss=surprise / max(counted, 1), picks=counted)


def _seen(packs: Tensor, *, cards: int) -> Tensor:
    # `packs` is int16 [b, t, pack_size], the card ids in each pack, padded with -1. Returns float [b, t, cards], one
    # for each card in the pack and zero elsewhere. The ids shift up by one so that the -1 padding lands in column 0,
    # which is then dropped.
    with_padding = torch.zeros(*packs.shape[:2], cards + 1, device=packs.device)
    with_padding.scatter_(2, packs.long() + 1, 1.0)
    return with_padding[:, :, 1:]


def _prior_loss(logits: Tensor, picked: Tensor, *, weights: Tensor, priors: PickPriors) -> Tensor:
    # `logits` is [m, cards], with cards outside the pack at -inf. `picked` is int64 [m] and `weights` is [m], one
    # entry per pick.
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
    encoder_rate: float
    dropout: float
    rare_penalty: float
    mana_penalty: float
    mana_margin: float


def _arguments(argv: Sequence[str]) -> _Arguments:
    parser = argparse.ArgumentParser(description="Train the drafter on human picks from 17Lands.")
    add_run_flags(parser)
    add_card_flags(parser)
    parser.add_argument("--device", default="auto", help="cpu, mps or cuda; auto prefers cuda, then mps")
    parser.add_argument(
        "--steps",
        type=int,
        default=4000,
        help="training steps; with dropout 0.2 the held-out log loss still falls slowly at 4,000",
    )
    parser.add_argument("--drafts-per-step", type=int, default=256)
    parser.add_argument("--evaluate-every", type=int, default=500)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6, help="the drafter's layers, as train_limited builds it")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--rate", type=float, default=5e-4, help="the drafter's peak learning rate")
    parser.add_argument(
        "--encoder-rate",
        type=float,
        default=1.5e-3,
        help="the card encoder's peak learning rate; at a third of the drafter's it copies picks worse, at three times "
        "a little better",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout in the drafter's transformer; without it the drafter overfits its training drafts",
    )
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
    """Score two untrained pickers on the held-out picks.

    The first is a uniformly random pick, computed exactly from the pack
    sizes. The second always takes the pack's card with the lowest average
    pick position. That position is averaged over every draft of the set,
    the held-out ones included, so the second picker's choices already
    depend on the picks it is scored on.

    Returns the random pick's `top1` to `top3` shares and `log_loss` as a
    dictionary, and the second picker's `PickAccuracy`.
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
    sizes = (drafts.packs[held_out] >= 0).sum(dim=-1)[real].float()  # [m], the size of each scored pack
    uniform = {f"top{top}": float((sizes.clamp_max(top) / sizes).mean()) for top in _TOP_RANKS}
    return {**uniform, "log_loss": float(sizes.log().mean())}, by_ata


def main(argv: Sequence[str]) -> None:
    """Pretrain a drafter on one card set and save its record, checkpoints and learning curve.

    `jace.storage.pretrain` creates the run folder under `--run-dir`, and the run saves
    a "drafter" and an "embedding" checkpoint in it. The curve file
    `<run id>.curve.json` is written beside the folder. Along with the curve,
    it contains the device, the wall time and two baselines scored on the same
    held-out picks. The first is a uniformly random pick, computed exactly. The
    second always takes the pack's card with the lowest average pick position.
    """
    arguments = _arguments(argv)
    device = device_named(arguments.device)
    torch.manual_seed(arguments.seed)
    data = read_set(arguments.set_code)
    cards = set_cards(data)
    drafts = load_drafts(
        data.picks, pool=cards.pool, cache_dir=arguments.data_dir, held_out_share=arguments.held_out_share
    )
    vocabulary = program_vocabulary()
    encoder = CardEncoder(
        program_tokens(cards.pool, vocabulary=vocabulary),
        width=arguments.card_width,
        program_vocabulary=FIRST_PROGRAM_TOKEN + len(vocabulary),
        program_layers=arguments.card_layers,
        heads=arguments.card_heads,
    )
    model = DrafterModel(
        encoder,
        width=arguments.width,
        layers=arguments.layers,
        heads=arguments.heads,
        picks=limited_format(arguments.set_code).picks,
        dropout=arguments.dropout,
    ).to(device)
    optimizers = [
        torch.optim.Adam(own_parameters(model, encoder), lr=arguments.rate, betas=ADAM_BETAS),
        torch.optim.Adam(encoder.parameters(), lr=arguments.encoder_rate, betas=ADAM_BETAS),
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

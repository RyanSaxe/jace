"""The losses that train each model, as plain functions of tensors.

A training loop is written by calling a model, passing its output to one of
these functions and handing the result to `apply`. None of them knows which
model produced the tensors, so the same `choice_loss` trains a drafter from
human picks, a drafter from game results and a deck builder from the decks a
search preferred. Every function returns a scalar loss unless it says
otherwise.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from jace.play import Guess

__all__ = ("PlayerRule", "choice_loss", "entropy", "guess_loss", "kl_to", "player_loss", "score_loss")


@dataclass(frozen=True, slots=True)
class PlayerRule:
    """The dials of the player's update.

    `clip` bounds how far one update may move an action's probability.
    `stay` weighs the pull toward the policy that played the games, and
    `spread` the pull toward even odds over legal actions; a loop lowers
    `spread` as training goes on. Only decisions whose advantage is at least
    the `keep_quantile` of advantage sizes, and at least `keep_floor`, teach
    the policy. `value_weight` weighs the win-chance error. With `stay`,
    `spread` and `keep_quantile` at zero this is plain PPO.
    """

    clip: float
    stay: float
    spread: float
    keep_quantile: float
    keep_floor: float
    value_weight: float


def _masked_log_probs(logits: Tensor) -> tuple[Tensor, Tensor]:
    # logits: [m, k] with unavailable choices at -inf. Returns (log-probabilities with 0 there, bool mask).
    available = logits.isfinite()
    return torch.where(available, logits.log_softmax(dim=-1), 0.0), available


def entropy(logits: Tensor) -> Tensor:
    # logits: [m, k] with unavailable choices at -inf. Returns [m], higher when the odds are more even.
    log_probs, available = _masked_log_probs(logits)
    return -(log_probs.exp() * log_probs * available).sum(dim=-1)


def kl_to(logits: Tensor, reference: Tensor) -> Tensor:
    """Measure how far the odds in `logits` are from those in `reference`, as `[m]`.

    Both are `[m, k]` with unavailable choices at `-inf` in the same places.
    It is the KL divergence from `logits` to `reference`, zero when the two
    agree. With a human-trained model's output as `reference` it is the pull
    toward human play; with the acting policy's it limits one update's step.
    """
    log_probs, available = _masked_log_probs(logits)
    reference_log_probs, _ = _masked_log_probs(reference)
    return (log_probs.exp() * (log_probs - reference_log_probs) * available).sum(dim=-1)


def choice_loss(logits: Tensor, chosen: Tensor, *, weights: Tensor) -> Tensor:
    """Make the `chosen` options more likely in proportion to `weights`.

    `logits` is `[m, k]`, `chosen` is int64 `[m]` and `weights` is `[m]`.
    Weights of one give ordinary cross-entropy, which copies the choices.
    Weights equal to "result minus the average result" give the policy
    gradient, which makes choices that led to better-than-average results
    more likely and the others less.
    """
    log_probs = logits.log_softmax(dim=-1).gather(1, chosen[:, None]).squeeze(1)
    return -(weights * log_probs).mean()


def player_loss(
    logits: Tensor,
    values: Tensor,
    *,
    old_logits: Tensor,
    actions: Tensor,
    advantages: Tensor,
    value_targets: Tensor,
    keep: Tensor,
    rule: PlayerRule,
) -> Tensor:
    """Compute the clipped policy loss with its two pulls and the value error.

    `logits` `[m, k]` and `values` `[m]` are the policy's fresh output on the
    rows; `old_logits` are what it output when it played them. `actions` is
    int64 `[m]`, `advantages` and `value_targets` are `[m]`, and `keep` is
    bool `[m]`, True for the rows that teach the policy. Rows outside `keep`
    still train the win chance.
    """
    log_probs, available = _masked_log_probs(logits)
    old_log_probs, _ = _masked_log_probs(old_logits)
    chosen = actions[:, None]
    ratio = (log_probs.gather(1, chosen) - old_log_probs.gather(1, chosen)).squeeze(1).exp()
    gain = torch.minimum(ratio * advantages, ratio.clamp(1.0 - rule.clip, 1.0 + rule.clip) * advantages)
    kept = keep.float()
    weight = kept.sum().clamp_min(1.0)
    stay = (log_probs.exp() * (log_probs - old_log_probs) * available).sum(dim=-1)
    even = available.float().sum(dim=-1).log() - entropy(logits)
    policy = ((-gain + rule.stay * stay + rule.spread * even) * kept).sum() / weight
    return policy + rule.value_weight * functional.mse_loss(values, value_targets)


def guess_loss(guess: Guess, *, deck_truth: Tensor, hand_truth: Tensor) -> Tensor:
    # guess.deck, guess.hand, deck_truth, hand_truth: [m, cards]. A confident wrong guess costs most.
    deck = -(deck_truth * guess.deck.clamp_min(1e-9).log()).sum(dim=-1).mean()
    return deck + functional.binary_cross_entropy(guess.hand.clamp(1e-6, 1.0 - 1e-6), hand_truth)


def score_loss(logits: Tensor, *, wins: Tensor, games: Tensor) -> Tensor:
    # logits: [d] deck scores; wins, games: [d] games won and played by each deck.
    lost = games - wins
    per_deck = wins * functional.logsigmoid(logits) + lost * functional.logsigmoid(-logits)
    return -per_deck.sum() / games.sum().clamp_min(1.0)

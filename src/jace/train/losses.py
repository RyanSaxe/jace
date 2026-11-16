"""The losses that train each model, as plain functions of tensors.

A training loop calls a model, passes its output to one of these functions
and hands the loss to `apply`. The functions take plain tensors and do not
depend on which model produced them. That is why the loops can train the
drafter with one `choice_loss`, both from human picks and from the win rates
of the decks it drafted. Every function returns a scalar loss except
`entropy` and `kl_to`, which return one value per row.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from jace.play import Guess

__all__ = (
    "PlayerRule",
    "choice_loss",
    "entropy",
    "guess_loss",
    "kl_to",
    "player_loss",
    "score_loss",
    "win_chance_loss",
)

# Before taking a logarithm, `guess_loss` raises every share to at least `_SMALLEST_SHARE` and keeps
# every chance at least `_CHANCE_MARGIN` from 0 and from 1. A guess of exactly 0 or 1 then costs a
# large but finite amount.
_SMALLEST_SHARE = 1e-9
_CHANCE_MARGIN = 1e-6


@dataclass(frozen=True, slots=True)
class PlayerRule:
    """The settings of the player's update.

    `clip` limits how far one update moves the probability of an action the
    policy took. The objective compares that probability with its
    probability during play. A ratio above 1 + `clip` for a positive
    advantage, or below 1 - `clip` for a negative one, adds nothing more to
    the objective.

    `stay` is the weight of a pull toward the policy that played the games,
    and `spread` the weight of a pull toward even odds over the legal
    actions. A pull is a KL divergence to another policy, added to the loss.
    The limited loop lowers `spread` as training goes on.

    The policy is trained only on the learner's decisions with a large
    enough advantage. A decision is kept when the absolute value of its
    advantage is at least `keep_floor`, and at least the `keep_quantile`
    quantile of the learner's absolute advantages.

    With `normalize`, every advantage is shifted by the kept decisions' mean
    and divided by their standard deviation before the loss. The size of a
    step then does not depend on how large the advantages happen to be. The
    advantages are left as they are when fewer than two decisions are kept.

    With `stay`, `spread`, `keep_quantile` and `keep_floor` all zero, the
    loss is plain PPO's clipped policy loss.
    """

    clip: float
    stay: float
    spread: float
    keep_quantile: float
    keep_floor: float
    normalize: bool


def _masked_log_probs(logits: Tensor) -> tuple[Tensor, Tensor]:
    # `logits` is [m, k] with unavailable choices at -inf. Returns the log-probabilities and a bool
    # mask of the available choices. An unavailable choice gets a log-probability of 0, not -inf,
    # because -inf times the mask's 0 would be NaN. Callers multiply by the mask to drop it.
    available = logits.isfinite()
    return torch.where(available, logits.log_softmax(dim=-1), 0.0), available


def entropy(logits: Tensor) -> Tensor:
    # `logits` is [m, k] with unavailable choices at -inf. Returns the entropy of each row's odds over
    # its available choices, as [m]: 0 when one choice is certain, and highest when the odds are even.
    log_probs, available = _masked_log_probs(logits)
    return -(log_probs.exp() * log_probs * available).sum(dim=-1)


def kl_to(logits: Tensor, reference: Tensor) -> Tensor:
    """Measure how far the odds in `logits` are from those in `reference`, as `[m]`.

    Both are `[m, k]`, with unavailable choices at `-inf` in the same
    places. Each row's value is KL(p || q), where p is the softmax of
    `logits` and q the softmax of `reference`. It is zero when the two
    are equal.

    Added to a loss with a human-trained model's output as `reference`, it
    is a pull toward human play. The `stay` term of `player_loss` is the
    same divergence, measured to the policy that played the games, and it
    limits how far one update moves the policy.
    """
    log_probs, available = _masked_log_probs(logits)
    reference_log_probs, _ = _masked_log_probs(reference)
    return (log_probs.exp() * (log_probs - reference_log_probs) * available).sum(dim=-1)


def choice_loss(logits: Tensor, chosen: Tensor, *, weights: Tensor) -> Tensor:
    """Make the `chosen` options more likely in proportion to `weights`.

    `logits` is `[m, k]`, `chosen` is int64 `[m]` and `weights` is `[m]`. A
    negative weight makes its choice less likely.

    With every weight at one, the loss is ordinary cross-entropy, which
    trains the model to imitate the choices. With each weight set to its
    choice's result minus the mean result, the loss gives a policy gradient:
    choices that led to better-than-average results become more likely, and
    the others less likely.
    """
    log_probs = logits.log_softmax(dim=-1).gather(1, chosen[:, None]).squeeze(1)
    return -(weights * log_probs).mean()


def player_loss(
    logits: Tensor,
    *,
    old_logits: Tensor,
    actions: Tensor,
    advantages: Tensor,
    keep: Tensor,
    rule: PlayerRule,
) -> Tensor:
    """Compute PPO's clipped policy loss, with the `stay` and `spread` pulls that `rule` weights.

    `logits` `[m, k]` are the policy's output now, and `old_logits` are its
    output when it made those decisions. `actions` is int64 `[m]`, the
    actions taken, and `advantages` is `[m]`. `keep` is bool `[m]`, True
    for the rows the policy is trained on.

    The loss is the mean over the kept rows, and zero when no row is kept.
    """
    log_probs, available = _masked_log_probs(logits)
    old_log_probs, _ = _masked_log_probs(old_logits)
    chosen = actions[:, None]
    # How many times more likely the policy now makes each taken action than when it took it.
    ratio = (log_probs.gather(1, chosen) - old_log_probs.gather(1, chosen)).squeeze(1).exp()
    # PPO's clipped objective takes the lesser of the plain and the clipped gain. So a ratio above
    # 1 + clip for a positive advantage, or below 1 - clip for a negative one, adds nothing more to
    # the objective in this update.
    gain = torch.minimum(ratio * advantages, ratio.clamp(1.0 - rule.clip, 1.0 + rule.clip) * advantages)
    kept = keep.float()
    kept_count = kept.sum().clamp_min(1.0)
    kl_to_old = (log_probs.exp() * (log_probs - old_log_probs) * available).sum(dim=-1)
    # The log of the number of legal actions minus the entropy is the KL divergence to even odds.
    kl_to_even = available.float().sum(dim=-1).log() - entropy(logits)
    return ((-gain + rule.stay * kl_to_old + rule.spread * kl_to_even) * kept).sum() / kept_count


def win_chance_loss(win_chance: Tensor, *, targets: Tensor, learner: Tensor) -> Tensor:
    """Compute the squared error of a win chance to its targets, over the learner's rows.

    `win_chance` and `targets` are `[m]`, and `learner` is bool `[m]`, True
    where the learner's policy acted. The win chance estimates the acting
    seat's chance to win when the learner's policy plays that seat. At an
    opponent's row another policy plays the seat, and the game's result
    there depends on that other policy, so the loss leaves those rows out.

    The loss is the mean over the learner's rows, and zero when no row is
    the learner's.
    """
    rows = learner.float()
    return ((win_chance - targets).pow(2) * rows).sum() / rows.sum().clamp_min(1.0)


def guess_loss(guess: Guess, *, deck_truth: Tensor, hand_truth: Tensor) -> Tensor:
    # Every tensor here is [m, cards]. Both terms are log losses, so a confident wrong guess costs the
    # most. Each term is summed over cards before the mean over decisions, so the hand guess and the
    # deck guess count with equal weight in every decision's loss.
    deck = -(deck_truth * guess.deck.clamp_min(_SMALLEST_SHARE).log()).sum(dim=-1)
    hand_chance = guess.hand.clamp(_CHANCE_MARGIN, 1.0 - _CHANCE_MARGIN)
    hand = functional.binary_cross_entropy(hand_chance, hand_truth, reduction="none").sum(dim=-1)
    return (deck + hand).mean()


def score_loss(logits: Tensor, *, wins: Tensor, games: Tensor) -> Tensor:
    # `logits` is [d], one score per deck, and `wins` and `games` are [d], the games each deck won
    # and played. The loss is the mean over all games played, so a deck counts in proportion to its
    # games.
    lost = games - wins
    # The binomial log-likelihood of each deck's record at the win chance sigmoid(logit).
    per_deck = wins * functional.logsigmoid(logits) + lost * functional.logsigmoid(-logits)
    return -per_deck.sum() / games.sum().clamp_min(1.0)

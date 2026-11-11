"""Turn a loss into a training step, and the two updates that need a rollout.

`apply` is the one place a gradient step happens: it steps every optimizer
it is given, which is how the shared card embedding's own optimizer moves
together with a model's. `update_player` and `update_guess` walk a rollout
in order, in slices, because the player's rule filters decisions over the
whole rollout and the guess model needs each decision's earlier guesses.
"""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer

from jace.play import Guesser, Policy
from jace.train.collect import Rollout, select
from jace.train.losses import PlayerRule, guess_loss, player_loss
from jace.train.sequences import advantages, earlier_guesses, sequences

__all__ = ("Metrics", "apply", "update_guess", "update_player")

type Metrics = Mapping[str, float]


def apply(loss: Tensor, optimizers: Sequence[Optimizer], *, max_grad_norm: float) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad()
    loss.backward()
    parameters = [
        parameter for optimizer in optimizers for group in optimizer.param_groups for parameter in group["params"]
    ]
    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    for optimizer in optimizers:
        optimizer.step()


def update_player(
    policy: Policy,
    rollout: Rollout,
    *,
    rule: PlayerRule,
    lam_advantage: float,
    lam_value: float,
    slice_rows: int,
    max_grad_norm: float,
    optimizers: Sequence[Optimizer],
) -> Metrics:
    """Run one pass of the player's update over `rollout`, in order.

    Advantages use `lam_advantage` and the win-chance targets use
    `lam_value`. The decisions that teach the policy are chosen once over
    the whole rollout: the learner's own, with an advantage at least as
    large as `rule` asks. Returns the mean loss, the share of decisions
    kept, and the learner's mean advantage size.
    """
    layout = sequences(rollout.games, rollout.seats)
    advantage = advantages(rollout.values, rollout.results, layout=layout, lam=lam_advantage)
    value_targets = rollout.values + advantages(rollout.values, rollout.results, layout=layout, lam=lam_value)
    size = advantage.abs()
    learner_sizes = size[rollout.learner]
    threshold = float(learner_sizes.quantile(rule.keep_quantile)) if len(learner_sizes) else 0.0
    keep = rollout.learner & (size >= max(threshold, rule.keep_floor))
    total = 0.0
    slices = 0
    for start in range(0, len(advantage), slice_rows):
        rows = torch.arange(start, min(start + slice_rows, len(advantage)))
        part = select(rollout, rows)
        logits, values = policy(part.boards, part.guesses)
        loss = player_loss(
            logits,
            values,
            old_logits=part.logits,
            actions=part.actions,
            advantages=advantage[rows],
            value_targets=value_targets[rows],
            keep=keep[rows],
            rule=rule,
        )
        apply(loss, optimizers, max_grad_norm=max_grad_norm)
        total += float(loss.detach())
        slices += 1
    return {
        "loss": total / max(slices, 1),
        "kept_share": float(keep.float().mean()),
        "advantage_size": float(learner_sizes.mean()) if len(learner_sizes) else 0.0,
    }


def update_guess(
    guesser: Guesser,
    rollout: Rollout,
    *,
    history: int,
    slice_rows: int,
    max_grad_norm: float,
    optimizers: Sequence[Optimizer],
) -> Metrics:
    """Run one pass of the guess model's update over `rollout`, in order.

    Each decision is scored against the opponent's true deck and hand. Its
    input includes the same seat's last `history` guesses in that game, as
    they were made during play, so the model learns from its own earlier
    guesses and not from the truth. Returns the mean loss.
    """
    layout = sequences(rollout.games, rollout.seats)
    total = 0.0
    slices = 0
    for start in range(0, len(rollout.games), slice_rows):
        rows = torch.arange(start, min(start + slice_rows, len(rollout.games)))
        part = select(rollout, rows)
        guess = guesser(part.boards, earlier_guesses(rollout.guesses, layout, rows=rows, history=history))
        loss = guess_loss(guess, deck_truth=part.deck_truth, hand_truth=part.hand_truth)
        apply(loss, optimizers, max_grad_norm=max_grad_norm)
        total += float(loss.detach())
        slices += 1
    return {"loss": total / max(slices, 1)}

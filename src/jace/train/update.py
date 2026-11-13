"""Turn a loss into a training step, and the two updates that need a rollout.

`apply` is the one place a gradient step happens: it steps every optimizer
it is given. `update_player` and `update_guess` walk a rollout in slices,
because the player's rule filters decisions over the whole rollout and the
guess model needs each decision's earlier guesses. Both take the rows in a
new random order on every pass. A rollout lists its games one worker after
another, so a slice cut in play order would hold a handful of games, and
its gradient would follow those games and not the round.
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

# Advantages are divided by at least this when they are normalized, so equal advantages stay finite.
_SMALLEST_SPREAD = 1e-8


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
    passes: int,
    generator: torch.Generator,
) -> Metrics:
    """Run `passes` passes of the player's update over `rollout`.

    Advantages use `lam_advantage` and the win-chance targets use
    `lam_value`. Both are computed once, from the win chances the player
    predicted when it played. The decisions that teach the policy are also
    chosen once over the whole rollout: the learner's own, with an
    advantage at least as large as `rule` asks. Every pass takes the rows
    in a new random order drawn from `generator`. Returns the mean loss,
    the share of decisions kept, and the learner's mean advantage size
    before any normalizing.
    """
    layout = sequences(rollout.games, rollout.seats)
    advantage = advantages(rollout.values, rollout.results, layout=layout, lam=lam_advantage)
    value_targets = rollout.values + advantages(rollout.values, rollout.results, layout=layout, lam=lam_value)
    size = advantage.abs()
    learner_sizes = size[rollout.learner]
    threshold = float(learner_sizes.quantile(rule.keep_quantile)) if len(learner_sizes) else 0.0
    keep = rollout.learner & (size >= max(threshold, rule.keep_floor))
    if rule.normalize and int(keep.sum()) > 1:
        advantage = (advantage - advantage[keep].mean()) / advantage[keep].std().clamp_min(_SMALLEST_SPREAD)
    total = 0.0
    slices = 0
    for _ in range(passes):
        for rows in torch.randperm(len(advantage), generator=generator).split(slice_rows):
            total += _player_step(
                policy,
                select(rollout, rows),
                advantages=advantage[rows],
                value_targets=value_targets[rows],
                keep=keep[rows],
                rule=rule,
                max_grad_norm=max_grad_norm,
                optimizers=optimizers,
            )
            slices += 1
    return {
        "loss": total / max(slices, 1),
        "kept_share": float(keep.float().mean()),
        "advantage_size": float(learner_sizes.mean()) if len(learner_sizes) else 0.0,
    }


def _player_step(
    policy: Policy,
    part: Rollout,
    *,
    advantages: Tensor,
    value_targets: Tensor,
    keep: Tensor,
    rule: PlayerRule,
    max_grad_norm: float,
    optimizers: Sequence[Optimizer],
) -> float:
    # One gradient step on the rows of `part`. advantages, value_targets, keep: [m], for those rows.
    logits, values = policy(part.boards, part.guesses)
    loss = player_loss(
        logits,
        values,
        old_logits=part.logits,
        actions=part.actions,
        advantages=advantages,
        value_targets=value_targets,
        keep=keep,
        learner=part.learner,
        rule=rule,
    )
    apply(loss, optimizers, max_grad_norm=max_grad_norm)
    return float(loss.detach())


def update_guess(
    guesser: Guesser,
    rollout: Rollout,
    *,
    history: int,
    slice_rows: int,
    max_grad_norm: float,
    optimizers: Sequence[Optimizer],
    generator: torch.Generator,
) -> Metrics:
    """Run one pass of the guess model's update over `rollout`, in a random order.

    Each decision is scored against the opponent's true deck and hand. Its
    input includes the same seat's last `history` guesses in that game, as
    they were made during play, so the model learns from its own earlier
    guesses and not from the truth. Those are looked up by row, so the
    order of the rows does not matter to them. Returns the mean loss.
    """
    layout = sequences(rollout.games, rollout.seats)
    total = 0.0
    slices = 0
    for rows in torch.randperm(len(rollout.games), generator=generator).split(slice_rows):
        part = select(rollout, rows)
        guess = guesser(part.boards, earlier_guesses(rollout.guesses, layout, rows=rows, history=history))
        loss = guess_loss(guess, deck_truth=part.deck_truth, hand_truth=part.hand_truth)
        apply(loss, optimizers, max_grad_norm=max_grad_norm)
        total += float(loss.detach())
        slices += 1
    return {"loss": total / max(slices, 1)}

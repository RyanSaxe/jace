"""Turn a loss into a training step, and the two updates that need a rollout.

`apply` is the one place a gradient step happens: it steps every optimizer
it is given. `update_player` and `update_guess` walk a rollout in slices,
because the player's rule filters decisions over the whole rollout and the
guess model needs each decision's earlier guesses. Both take the rows in a
new random order on every pass. A rollout lists its games one worker after
another, so a slice cut in play order would hold a handful of games, and
its gradient would follow those games and not the round. `update_player`
also trains the win chance its advantages come from, on the same slices.
"""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer

from jace.play import Guess, Guesser, Policy, WinChance
from jace.train.collect import Rollout, select
from jace.train.losses import PlayerRule, guess_loss, player_loss, win_chance_loss
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
    win_chance: WinChance,
    rule: PlayerRule,
    lam_advantage: float,
    lam_value: float,
    slice_rows: int,
    max_grad_norm: float,
    policy_optimizers: Sequence[Optimizer],
    win_chance_optimizers: Sequence[Optimizer],
    passes: int,
    generator: torch.Generator,
) -> Metrics:
    """Run `passes` passes of the player's update and its win chance's over `rollout`.

    First the win chance of every row is computed, in slices of
    `slice_rows` and without gradients, from the acting seat's boards and
    the opponent's true hidden cards. The advantages, with
    `lam_advantage`, and the win chance's targets, the lambda-returns with
    `lam_value`, are computed once from those. So are the decisions that
    teach the policy: the learner's own, with an advantage at least as
    large as `rule` asks. Every pass takes the rows in a new random order
    drawn from `generator`, and each slice steps the policy on its loss
    with `policy_optimizers` and then the win chance on its error to the
    targets with `win_chance_optimizers`. Returns the policy's mean loss as
    `loss` and the win chance's as `win_chance_loss`, the share of decisions
    kept, and the learner's mean advantage size before any normalizing.
    """
    layout = sequences(rollout.games, rollout.seats)
    chances = torch.empty(len(rollout.games))
    with torch.no_grad():
        for rows in torch.arange(len(rollout.games)).split(slice_rows):
            part = select(rollout, rows)
            chances[rows] = win_chance(part.boards, Guess(deck=part.deck_truth, hand=part.hand_truth))
    advantage = advantages(chances, rollout.results, layout=layout, lam=lam_advantage)
    # A prediction plus its advantage is the lambda-return, the target the win chance learns.
    chance_targets = chances + advantages(chances, rollout.results, layout=layout, lam=lam_value)
    advantage_size = advantage.abs()
    learner_sizes = advantage_size[rollout.learner]
    threshold = float(learner_sizes.quantile(rule.keep_quantile)) if len(learner_sizes) else 0.0
    keep = rollout.learner & (advantage_size >= max(threshold, rule.keep_floor))
    if rule.normalize and int(keep.sum()) > 1:
        advantage = (advantage - advantage[keep].mean()) / advantage[keep].std().clamp_min(_SMALLEST_SPREAD)
    policy_losses: list[float] = []
    chance_losses: list[float] = []
    for _ in range(passes):
        for rows in torch.randperm(len(advantage), generator=generator).split(slice_rows):
            part = select(rollout, rows)
            policy_loss = player_loss(
                policy(part.boards, part.guesses),
                old_logits=part.logits,
                actions=part.actions,
                advantages=advantage[rows],
                keep=keep[rows],
                rule=rule,
            )
            apply(policy_loss, policy_optimizers, max_grad_norm=max_grad_norm)
            chance_loss = win_chance_loss(
                win_chance(part.boards, Guess(deck=part.deck_truth, hand=part.hand_truth)),
                targets=chance_targets[rows],
                learner=part.learner,
            )
            apply(chance_loss, win_chance_optimizers, max_grad_norm=max_grad_norm)
            policy_losses.append(float(policy_loss.detach()))
            chance_losses.append(float(chance_loss.detach()))
    return {
        "loss": sum(policy_losses) / max(len(policy_losses), 1),
        "win_chance_loss": sum(chance_losses) / max(len(chance_losses), 1),
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
    passes: int,
    generator: torch.Generator,
) -> Metrics:
    """Run `passes` passes of the guess model's update over `rollout`.

    Each decision is scored against the opponent's true deck and hand. Its
    input includes the same seat's last `history` guesses in that game, as
    they were made during play, so the model learns from its own earlier
    guesses and not from the truth, on every pass. Those are looked up by
    row, so the order of the rows does not matter to them. Every pass takes
    the rows in a new random order drawn from `generator`. Returns the mean
    loss.
    """
    layout = sequences(rollout.games, rollout.seats)
    loss_total = 0.0
    slice_count = 0
    for _ in range(passes):
        for rows in torch.randperm(len(rollout.games), generator=generator).split(slice_rows):
            part = select(rollout, rows)
            guess = guesser(part.boards, earlier_guesses(rollout.guesses, layout, rows=rows, history=history))
            loss = guess_loss(guess, deck_truth=part.deck_truth, hand_truth=part.hand_truth)
            apply(loss, optimizers, max_grad_norm=max_grad_norm)
            loss_total += float(loss.detach())
            slice_count += 1
    return {"loss": loss_total / max(slice_count, 1)}

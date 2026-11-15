"""Turn a loss into a training step, and the two updates that need a rollout.

`apply` is the one place a gradient step happens, and it steps every
optimizer it is given. `update_player` and `update_guess` take a whole
rollout and go through it in slices. They need the whole rollout because the
decisions the player is trained on are picked by a quantile over all of
them, and because the guess model reads each decision's earlier guesses.
`update_player` also trains the win chance that its advantages come from,
on the same slices.

Both take the rows in a new random order on every pass. A rollout lists its
games one worker after another, so a slice cut in play order would contain
only a handful of games. That slice's gradient would follow those few games
rather than the whole round.

Both run on the rollout's device, which must also be the models' device.
Every slice is cut and every loss computed there, and the losses are read
back once, after the last pass.
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

    The win chance reads the acting seat's boards and the opponent's true
    hidden cards. Before the first pass, its predictions give the
    advantages, the win chance's own targets and the decisions the policy is
    trained on. These stay the same through every pass. `lam_advantage` is
    the `lam` of the advantages, and `lam_value` is the `lam` of the
    targets, which are lambda-returns. Both models learn only from the
    learner's decisions, and the policy only from those that `rule` keeps.

    For each slice of `slice_rows` rows, the policy takes a step with
    `policy_optimizers` and then the win chance takes one with
    `win_chance_optimizers`. The rows are shuffled with `generator` on every
    pass. No model call runs on more than `slice_rows` rows at once.

    The metrics returned are:

    - `loss`, the policy's mean loss over every slice;
    - `win_chance_loss`, the win chance's mean loss over every slice;
    - `kept_share`, the share of all the rollout's decisions that the policy
      is trained on;
    - `advantage_size`, the mean absolute advantage of the learner's
      decisions before any normalizing, or zero when there are none.
    """
    device = rollout.results.device
    layout = sequences(rollout.games, rollout.seats)
    chances = torch.empty(len(rollout.games), device=device)
    with torch.no_grad():
        for rows in torch.arange(len(rollout.games), device=device).split(slice_rows):
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
    policy_losses: list[Tensor] = []
    chance_losses: list[Tensor] = []
    for _ in range(passes):
        for rows in torch.randperm(len(advantage), generator=generator).to(device).split(slice_rows):
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
            policy_losses.append(policy_loss.detach())
            chance_losses.append(chance_loss.detach())
    return {
        "loss": _mean(policy_losses),
        "win_chance_loss": _mean(chance_losses),
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

    Each decision's guess is scored against the opponent's true unseen cards
    and hand. The model also reads the same seat's last `history` guesses in
    that game, exactly as the guesser made them during play. It is
    therefore trained with the earlier guesses it reads in play, not with
    the true cards, and these inputs stay the same on every pass.

    The rows are shuffled with `generator` on every pass. The earlier
    guesses are looked up by row, so the shuffle does not change them. No
    model call runs on more than `slice_rows` rows at once. Returns the mean
    loss over every slice as `loss`.
    """
    device = rollout.results.device
    layout = sequences(rollout.games, rollout.seats)
    losses: list[Tensor] = []
    for _ in range(passes):
        for rows in torch.randperm(len(rollout.games), generator=generator).to(device).split(slice_rows):
            part = select(rollout, rows)
            guess = guesser(part.boards, earlier_guesses(rollout.guesses, layout, rows=rows, history=history))
            loss = guess_loss(guess, deck_truth=part.deck_truth, hand_truth=part.hand_truth)
            apply(loss, optimizers, max_grad_norm=max_grad_norm)
            losses.append(loss.detach())
    return {"loss": _mean(losses)}


def _mean(losses: list[Tensor]) -> float:
    # The mean of the slices' losses, or zero when there was no slice. Reading a loss back waits for
    # the device to finish, so the losses are read once, together.
    return float(torch.stack(losses).mean()) if losses else 0.0

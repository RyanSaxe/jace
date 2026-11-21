"""Small supervised loss and optimizer-step helpers shared by pretraining loops.

The drafter, deck score, and deck builder pretrain on 17Lands human data using
standard cross-entropy, binomial log-likelihood, and clipped gradient steps.
Keeping these three functions in one small helper avoids coupling supervised
pretraining to the RL modules in `jace.rl`.
"""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional
from torch.optim import Optimizer

__all__ = ("apply", "choice_loss", "score_loss")


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


def choice_loss(logits: Tensor, chosen: Tensor, *, weights: Tensor) -> Tensor:
    log_probs = logits.log_softmax(dim=-1).gather(1, chosen[:, None]).squeeze(1)
    return -(weights * log_probs).mean()


def score_loss(logits: Tensor, *, wins: Tensor, games: Tensor) -> Tensor:
    lost = games - wins
    per_deck = wins * functional.logsigmoid(logits) + lost * functional.logsigmoid(-logits)
    return -per_deck.sum() / games.sum().clamp_min(1.0)

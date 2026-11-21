"""Advantage-based row selection filters for policy updates.

Decides which collected decisions contribute to the policy gradient while leaving the
critic free to learn from the full rollout. Keeping selection as a pure tensor function
lets experiments switch between full-batch PPO and top-fraction leverage filtering
without touching collection or loss code.
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor

__all__ = (
    "SelectFn",
    "all_rows",
    "top_fraction",
)

type SelectFn = Callable[[Tensor], Tensor]


def all_rows(advantages: Tensor) -> Tensor:
    if advantages.ndim == 0:
        msg = "advantages must have at least one dimension"
        raise ValueError(msg)
    return torch.ones(advantages.shape, dtype=torch.bool, device=advantages.device)


def top_fraction(fraction: float, *, minimum: float = 0.0) -> SelectFn:
    """Select the highest-magnitude advantage rows for policy updates.

    Keeps the top `fraction` of flat rows ranked by `|advantage|` (using stable
    argsort so ties stay deterministic) and drops any row whose magnitude falls
    below `minimum`. Raises `ValueError` unless `0 < fraction <= 1` and
    `minimum >= 0`.
    """
    if not (0.0 < fraction <= 1.0):
        msg = f"fraction must be in (0, 1], got {fraction}"
        raise ValueError(msg)
    if minimum < 0.0:
        msg = f"minimum must be non-negative, got {minimum}"
        raise ValueError(msg)

    def select(advantages: Tensor) -> Tensor:
        flat = advantages.detach().reshape(-1).abs()
        mask = torch.zeros_like(flat, dtype=torch.bool)
        if flat.numel() == 0:
            return mask.reshape_as(advantages)
        count = max(1, math.ceil(flat.numel() * fraction))
        order = torch.argsort(flat, descending=True, stable=True)[:count]
        kept = order[flat[order] >= minimum]
        mask[kept] = True
        return mask.reshape_as(advantages)

    return select

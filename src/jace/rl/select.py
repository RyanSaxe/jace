"""Advantage-based row selection filters for policy updates.

Decides which collected decisions contribute to the policy gradient while leaving the
critic free to learn from the full rollout. Keeping selection as a pure tensor function
lets experiments switch between full-batch PPO and top-fraction leverage filtering
without touching collection or loss code.
"""

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from torch import Tensor

__all__ = (
    "SelectFn",
    "all_rows",
    "bucketed_minibatch_order",
    "top_fraction",
    "training_row_indexes",
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


def bucketed_minibatch_order(
    total_rows: int,
    *,
    minibatch_size: int,
    row_sizes: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a row permutation grouped into length-coherent minibatches.

    When `row_sizes` is supplied, shuffles rows first so equal-size ties break
    randomly, sorts into `minibatch_size` buckets by board size to avoid padding
    early-game rows to stalemate lengths on MPS, and shuffles the bucket order.
    """
    perm = rng.permutation(total_rows).astype(np.int64)
    if row_sizes is None or total_rows <= minibatch_size:
        return perm
    sorted_perm = perm[np.argsort(row_sizes[perm], kind="mergesort")]
    buckets = [sorted_perm[start : start + minibatch_size] for start in range(0, total_rows, minibatch_size)]
    rng.shuffle(buckets)
    return np.concatenate(buckets)


def training_row_indexes(
    selected_mask: Tensor,
    *,
    filter_scope: Literal["actor", "actor_critic"],
) -> np.ndarray:
    """Resolve the flat rollout row indices to admit and visit during minibatch updates.

    Under `filter_scope="actor"`, returns every row `[0, N)` so the critic trains on
    the full rollout while the actor masks by `selected_mask`. Under `"actor_critic"`,
    restricts both actor and critic to the selected high-leverage rows, falling back
    to all rows when zero rows pass the filter.
    """
    total_rows = int(selected_mask.numel())
    if filter_scope == "actor":
        return np.arange(total_rows, dtype=np.int64)
    selected = selected_mask.detach().reshape(-1).nonzero(as_tuple=False).flatten().cpu().numpy().astype(np.int64)
    if selected.size == 0:
        return np.arange(total_rows, dtype=np.int64)
    return selected

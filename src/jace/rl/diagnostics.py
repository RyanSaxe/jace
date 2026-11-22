"""Pure-PyTorch and NumPy training diagnostics for self-play RL updates.

Computes batch-to-batch gradient cosine similarity across consecutive PPO updates on
CPU float32 vectors so accelerator memory stays flat while diagnosing whether wide
human-deck diversity drowns update signal. Also extracts empirical learner win rates
from 1-seat historical snapshot rollout batches to drive even-odds opponent matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.nn import Parameter

__all__ = (
    "GradientAgreementTracker",
    "snapshot_win_rate",
)

_MIN_NORM = 1e-12


class GradientAgreementTracker:
    """Track cosine similarity between consecutive updates' mean gradient directions.

    Call `begin_update` before an update's minibatches, `accumulate` after each
    `loss.backward()` (before `optimizer.step()`), and `finish_update` once the
    update completes. Keeps only two flat CPU float32 vectors in memory so MPS
    memory stays unaffected.
    """

    def __init__(self) -> None:
        self._prev_mean: Tensor | None = None
        self._accum: Tensor | None = None
        self._steps: int = 0

    def begin_update(self) -> None:
        self._accum = None
        self._steps = 0

    def accumulate(self, parameters: Iterable[Parameter]) -> None:
        trainable = [param for param in parameters if param.requires_grad]
        if not any(param.grad is not None for param in trainable):
            return
        flat = torch.cat(
            [
                param.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
                if param.grad is not None
                else torch.zeros(param.numel(), dtype=torch.float32)
                for param in trainable
            ]
        )
        if self._accum is None:
            self._accum = flat
        elif self._accum.shape == flat.shape:
            self._accum.add_(flat)
        else:
            # Parameter set changed (for example, lazy module init); reset accumulator.
            self._accum = flat
            self._steps = 0
        self._steps += 1

    def finish_update(self) -> float | None:
        if self._accum is None or self._steps <= 0:
            return None
        current = self._accum / float(self._steps)
        cur_norm = float(torch.linalg.vector_norm(current).item())
        cosine: float | None = None
        if self._prev_mean is not None and self._prev_mean.shape == current.shape:
            prev_norm = float(torch.linalg.vector_norm(self._prev_mean).item())
            if cur_norm > _MIN_NORM and prev_norm > _MIN_NORM:
                cosine = float(torch.dot(self._prev_mean, current).item() / (prev_norm * cur_norm))
        self._prev_mean = current
        self._accum = None
        self._steps = 0
        return cosine


def snapshot_win_rate(rewards: np.ndarray, dones: np.ndarray) -> float | None:
    done_mask = np.asarray(dones, dtype=bool).reshape(-1)
    if not np.any(done_mask):
        return None
    terminal = np.asarray(rewards, dtype=np.float64).reshape(-1)[done_mask]
    wins = float(np.sum(terminal > 0.0))
    draws = float(np.sum(terminal == 0.0))
    return (wins + 0.5 * draws) / float(terminal.size)

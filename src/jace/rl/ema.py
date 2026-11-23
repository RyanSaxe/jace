"""Exponential moving average (EMA) shadow weights for smoother self-play policies.

Maintains detached CPU copies of a module's parameters and buffers, advancing shadow
weights after each training update via `decay * shadow + (1 - decay) * live` so
evaluation checkpoints damp step-to-step stochasticity without consuming extra
accelerator memory. Supports `copy_to`, `sync_from`, `export_state`, and
`restore_state` for portable evaluation snapshots and atomic checkpoint resume.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from jace.rl.pool import TrainConfig

__all__ = (
    "EmaWeights",
    "resolve_ema_weights",
)


class EmaWeights:
    """Detached CPU exponential moving average tracker for PyTorch module weights.

    Copies parameters and buffers onto CPU at construction, updates parameters via
    `lerp_(live, 1 - decay)` on each `update` call while mirroring buffers verbatim,
    and writes the smoothed weights into an evaluation copy with `copy_to`. Raises
    `ValueError` if `decay` is outside `[0, 1)`.
    """

    _decay: float
    _updates: int
    _params: dict[str, Tensor]
    _buffers: dict[str, Tensor]

    def __init__(self, module: nn.Module, *, decay: float = 0.999) -> None:
        if not (0.0 <= decay < 1.0):
            msg = f"decay must be in [0, 1), got {decay}"
            raise ValueError(msg)
        self._decay = float(decay)
        self._updates = 0
        self._params = {name: param.detach().cpu().clone() for name, param in module.named_parameters()}
        self._buffers = {name: buf.detach().cpu().clone() for name, buf in module.named_buffers()}

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def updates(self) -> int:
        return self._updates

    @torch.no_grad()
    def sync_from(self, module: nn.Module) -> None:
        self._params = {name: param.detach().cpu().clone() for name, param in module.named_parameters()}
        self._buffers = {name: buf.detach().cpu().clone() for name, buf in module.named_buffers()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        alpha = 1.0 - self._decay
        for name, param in module.named_parameters():
            src = param.detach().cpu()
            shadow = self._params.get(name)
            if shadow is None:
                self._params[name] = src.clone()
            else:
                shadow.lerp_(src.to(dtype=shadow.dtype), alpha)
        for name, buf in module.named_buffers():
            self._buffers[name] = buf.detach().cpu().clone()
        self._updates += 1

    @torch.no_grad()
    def copy_to(self, target: nn.Module) -> None:
        for name, param in target.named_parameters():
            if name in self._params:
                param.copy_(self._params[name].to(device=param.device, dtype=param.dtype))
        for name, buf in target.named_buffers():
            if name in self._buffers:
                buf.copy_(self._buffers[name].to(device=buf.device, dtype=buf.dtype))

    def export_state(self) -> dict[str, object]:
        return {
            "decay": self._decay,
            "updates": self._updates,
            "params": {name: tensor.detach().cpu().clone() for name, tensor in self._params.items()},
            "buffers": {name: tensor.detach().cpu().clone() for name, tensor in self._buffers.items()},
        }

    def restore_state(self, payload: Mapping[str, object]) -> None:
        raw_decay = payload.get("decay")
        if isinstance(raw_decay, (int, float)) and 0.0 <= float(raw_decay) < 1.0:
            self._decay = float(raw_decay)
        raw_updates = payload.get("updates")
        if isinstance(raw_updates, int):
            self._updates = raw_updates
        self._params = _extract_tensor_map(payload.get("params"))
        self._buffers = _extract_tensor_map(payload.get("buffers"))


def _extract_tensor_map(raw: object) -> dict[str, Tensor]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value.detach().cpu().clone() for key, value in raw.items() if isinstance(value, Tensor)}


def resolve_ema_weights(
    ema_weights: EmaWeights | None,
    module: nn.Module,
    *,
    config: TrainConfig,
) -> EmaWeights | None:
    if ema_weights is not None:
        return ema_weights
    if config.ema is None:
        return None
    return EmaWeights(module, decay=config.ema)

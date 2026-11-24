"""Learning-rate and magnet-KL schedules plus optimizer rate helpers for self-play RL.

Pure PyTorch schedule math with zero `manabot`, `managym`, or higher-layer `jace`
imports so decay curves can be verified in milliseconds. `DampedSchedule` implements
the horizon-mapped power-law curves of damped policy iteration, while `DecaySchedule`
provides inverse-linear decay for dual reverse-KL PPO updates.
"""

from dataclasses import dataclass

from torch.optim import Optimizer

__all__ = (
    "DampedSchedule",
    "DecaySchedule",
    "current_optimizer_lr",
    "set_optimizer_lr",
)

_MIN_REFERENCE_STEPS = 2


@dataclass(frozen=True, slots=True)
class DampedSchedule:
    """Horizon-mapped power-law schedule for learning rate and magnet KL weight `tau`.

    Maps 1-based update indices linearly onto `[1, reference_steps]` so short and long
    runs traverse the same decay curve (`lr = clamp(lr_scale / pos**lr_power, lr_min,
    lr_max)` and `tau = tau_scale / pos**tau_power`). For `total_updates == 1`, `position`
    returns `1.0`. Raises `ValueError` if bounds or update indices are invalid.
    """

    total_updates: int
    reference_steps: int = 10_000
    lr_scale: float = 0.5
    lr_power: float = 1.1
    lr_min: float = 5e-6
    lr_max: float = 1e-4
    tau_scale: float = 0.05
    tau_power: float = 0.3

    def __post_init__(self) -> None:
        if self.total_updates < 1 or self.reference_steps < _MIN_REFERENCE_STEPS:
            msg = "total_updates must be >= 1 and reference_steps must be >= 2"
            raise ValueError(msg)
        if self.lr_min <= 0.0 or self.lr_min > self.lr_max:
            msg = "learning rate bounds must satisfy 0 < lr_min <= lr_max"
            raise ValueError(msg)

    def position(self, update: int) -> float:
        if update < 1:
            msg = f"update must be >= 1, got {update}"
            raise ValueError(msg)
        if self.total_updates == 1:
            return 1.0
        progress = min(1.0, (update - 1) / (self.total_updates - 1))
        return 1.0 + progress * (self.reference_steps - 1)

    def rates(self, update: int) -> tuple[float, float]:
        pos = self.position(update)
        raw_lr = self.lr_scale / (pos**self.lr_power)
        lr = min(self.lr_max, max(self.lr_min, raw_lr))
        tau = self.tau_scale / (pos**self.tau_power)
        return lr, tau


@dataclass(frozen=True, slots=True)
class DecaySchedule:
    """Inverse-linear horizon decay schedule for PPO with dual reverse-KL anchors.

    Computes `initial / (1 + decay * progress)` across 1-based updates `1..total_updates`
    where `progress` runs linearly from `0.0` to `1.0` (returning initial rates when
    `total_updates == 1`). Raises `ValueError` for invalid horizon or negative rates.
    """

    total_updates: int
    lr_initial: float = 2.5e-4
    tau_initial: float = 0.01
    decay: float = 9.0

    def __post_init__(self) -> None:
        if self.total_updates < 1 or self.lr_initial <= 0.0 or self.tau_initial < 0.0 or self.decay < 0.0:
            msg = "total_updates must be >= 1, lr_initial > 0, and tau_initial/decay non-negative"
            raise ValueError(msg)

    def rates(self, update: int) -> tuple[float, float]:
        if update < 1:
            msg = f"update must be >= 1, got {update}"
            raise ValueError(msg)
        progress = 0.0 if self.total_updates == 1 else min(1.0, (update - 1) / (self.total_updates - 1))
        factor = 1.0 + self.decay * progress
        return self.lr_initial / factor, self.tau_initial / factor


def set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def current_optimizer_lr(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0

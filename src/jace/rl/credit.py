"""Credit assignment functions for self-play reinforcement learning.

Turns per-seat rollout trajectories of rewards, critic predictions, and terminal flags
into policy advantages and value targets without importing any engine or policy code.
Trajectories may be dense `[time, stream]` arrays or packed complete-game columns
separated by `dones`, and every recursion cuts cleanly at terminal flags so credit
never leaks across games or seats.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

__all__ = (
    "CreditFn",
    "CreditTargets",
    "Trajectory",
    "gae_wdl",
    "monte_carlo",
    "standardize_advantages",
)


@dataclass(frozen=True, slots=True)
class Trajectory:
    rewards: Tensor
    values: Tensor
    dones: Tensor
    next_value: Tensor
    outcome_probs: Tensor | None = None
    next_outcome_probs: Tensor | None = None


@dataclass(frozen=True, slots=True)
class CreditTargets:
    advantages: Tensor
    raw_advantages: Tensor
    value_targets: Tensor


type CreditFn = Callable[[Trajectory], CreditTargets]


def _reverse_waves(ends: NDArray[np.bool_]) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Group trajectory steps by distance from each episode's end for parallel backward scans.

    Each wave holds `(steps, columns)` pairs whose next step was resolved in an earlier
    wave (or is terminal), allowing packed complete-game columns of thousands of rows to
    be scanned in as many iterations as the longest single game rather than total rows.
    """
    steps, streams = ends.shape
    waves: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    terminal_mask = ends.copy()
    terminal_mask[-1, :] = True
    end_steps, end_cols = np.nonzero(terminal_mask)
    if len(end_steps) == 0:
        return waves
    distances = np.zeros((steps, streams), dtype=np.int64)
    for col in range(streams):
        col_ends = end_steps[end_cols == col]
        prev = -1
        for stop in col_ends:
            length = int(stop) - prev
            distances[prev + 1 : int(stop) + 1, col] = np.arange(length - 1, -1, -1, dtype=np.int64)
            prev = int(stop)
    max_dist = int(distances.max())
    for dist in range(max_dist + 1):
        wave_steps, wave_cols = np.nonzero(distances == dist)
        waves.append((wave_steps.astype(np.int64), wave_cols.astype(np.int64)))
    return waves


def _scalar_gae(
    *,
    rewards: NDArray[np.float32],
    values: NDArray[np.float32],
    ends: NDArray[np.bool_],
    next_value: NDArray[np.float32],
    gamma: float,
    lam: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    steps, streams = rewards.shape
    following = np.concatenate([values[1:], next_value[None, :]], axis=0)
    not_done = (~ends).astype(np.float32)
    deltas = rewards + gamma * following * not_done - values
    advantages = np.zeros((steps + 1, streams), dtype=np.float32)
    for wave_steps, wave_cols in _reverse_waves(ends):
        carry = gamma * lam * not_done[wave_steps, wave_cols] * advantages[wave_steps + 1, wave_cols]
        advantages[wave_steps, wave_cols] = deltas[wave_steps, wave_cols] + carry
    trimmed = advantages[:-1]
    return trimmed, trimmed + values


def _categorical_lambda_returns(
    *,
    outcome_probs: NDArray[np.float32],
    rewards: NDArray[np.float32],
    ends: NDArray[np.bool_],
    next_outcome_probs: NDArray[np.float32],
    lam: float,
) -> NDArray[np.float32]:
    steps, streams, _ = outcome_probs.shape
    following = np.concatenate([outcome_probs[1:], next_outcome_probs[None, :, :]], axis=0)
    outcomes = np.clip(np.rint(rewards).astype(np.int64) + 1, 0, 2)
    terminal = np.eye(3, dtype=np.float32)[outcomes]
    targets = np.zeros((steps + 1, streams, 3), dtype=np.float32)
    targets[-1] = next_outcome_probs
    for wave_steps, wave_cols in _reverse_waves(ends):
        continuation = (1.0 - lam) * following[wave_steps, wave_cols] + lam * targets[wave_steps + 1, wave_cols]
        is_end = ends[wave_steps, wave_cols, None]
        targets[wave_steps, wave_cols] = np.where(is_end, terminal[wave_steps, wave_cols], continuation)
    return targets[:-1]


def standardize_advantages(advantages: Tensor, *, mask: Tensor | None = None) -> Tensor:
    reference = advantages if mask is None else advantages[mask]
    if reference.numel() == 0:
        return torch.zeros_like(advantages)
    std = reference.std(unbiased=False).clamp_min(1e-8)
    return (advantages - reference.mean()) / std


def gae_wdl(*, gamma: float = 1.0, lam_policy: float = 0.5, lam_value: float = 0.8) -> CreditFn:
    """Build a credit function combining GAE policy advantages with WDL or scalar lambda-returns.

    Uses `lam_policy` for policy advantage estimation and `lam_value` for critic targets.
    When `trajectory.outcome_probs` (`[T, E, 3]`) is supplied, computes undiscounted
    categorical loss/draw/win lambda-returns; otherwise computes scalar lambda-returns
    `values + GAE`. Raises `ValueError` if discount or trace parameters fall outside `[0, 1]`.
    """
    if not (0.0 <= gamma <= 1.0 and 0.0 <= lam_policy <= 1.0 and 0.0 <= lam_value <= 1.0):
        msg = f"gamma and lambdas must lie in [0, 1], got gamma={gamma}, lam_policy={lam_policy}, lam_value={lam_value}"
        raise ValueError(msg)

    def compute(trajectory: Trajectory) -> CreditTargets:
        rewards_np = trajectory.rewards.detach().cpu().numpy().astype(np.float32, copy=False)
        values_np = trajectory.values.detach().cpu().numpy().astype(np.float32, copy=False)
        ends_np = trajectory.dones.detach().cpu().numpy().astype(bool, copy=False)
        next_val_np = trajectory.next_value.detach().cpu().numpy().astype(np.float32, copy=False)

        raw_adv_np, scalar_targets_np = _scalar_gae(
            rewards=rewards_np,
            values=values_np,
            ends=ends_np,
            next_value=next_val_np,
            gamma=gamma,
            lam=lam_policy,
        )
        if trajectory.outcome_probs is not None:
            if trajectory.next_outcome_probs is None:
                msg = "next_outcome_probs is required when outcome_probs is provided"
                raise ValueError(msg)
            value_targets_np = _categorical_lambda_returns(
                outcome_probs=trajectory.outcome_probs.detach().cpu().numpy().astype(np.float32, copy=False),
                rewards=rewards_np,
                ends=ends_np,
                next_outcome_probs=trajectory.next_outcome_probs.detach().cpu().numpy().astype(np.float32, copy=False),
                lam=lam_value,
            )
        elif lam_value == lam_policy:
            value_targets_np = scalar_targets_np
        else:
            _, value_targets_np = _scalar_gae(
                rewards=rewards_np,
                values=values_np,
                ends=ends_np,
                next_value=next_val_np,
                gamma=gamma,
                lam=lam_value,
            )

        device = trajectory.rewards.device
        raw_adv = torch.from_numpy(raw_adv_np).to(device)
        return CreditTargets(
            advantages=standardize_advantages(raw_adv),
            raw_advantages=raw_adv,
            value_targets=torch.from_numpy(value_targets_np).to(device),
        )

    return compute


def monte_carlo(trajectory: Trajectory) -> CreditTargets:
    """Compute undiscounted Monte Carlo episode returns and advantages (`gamma=1, lam=1`).

    Uses exact terminal episode outcomes for every decision in a completed game (with
    critic bootstrapping only on truncated non-terminal tails), avoiding value-function
    bias at the cost of higher variance than GAE.
    """
    compute = gae_wdl(gamma=1.0, lam_policy=1.0, lam_value=1.0)
    targets = compute(trajectory)
    if targets.raw_advantages.shape != trajectory.rewards.shape:
        msg = "Monte Carlo advantages shape mismatch"
        raise RuntimeError(msg)
    return targets

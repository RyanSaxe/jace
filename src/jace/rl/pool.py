"""Historical snapshot opponent pool and training configuration records for self-play RL.

Pure PyTorch and NumPy utilities with zero `manabot` or `managym` imports so opponent
checkpoint management and update configuration can be unit-tested in milliseconds.
`SnapshotPool` maintains a bounded FIFO ring buffer of detached CPU parameter snapshots
with optional AlphaStar-style even-odds matching.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

__all__ = (
    "SnapshotEntry",
    "SnapshotPool",
    "StepRecord",
    "TrainConfig",
    "UpdateConfig",
    "UpdateStats",
    "load_snapshot",
    "resolve_snapshot_pool",
    "should_use_snapshot",
    "update_snapshot_pool",
)

_EVEN_ODDS_FLOOR = 0.05


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    update: int
    state: dict[str, Tensor]


class SnapshotPool:
    """Bounded FIFO ring buffer of detached CPU model snapshots for historical self-play.

    Each call to `add` clones every tensor in `module.state_dict()` onto CPU so subsequent
    optimizer steps on the live model never alias stored weights and accelerator memory
    stays constant. Supports uniform sampling or AlphaStar-style even-odds matching
    (`matching="even_odds"`), which weights snapshots by `max(0.05, p * (1 - p))` using
    each snapshot's exponential moving average win rate from prior games.
    """

    def __init__(
        self,
        *,
        max_size: int = 16,
        matching: Literal["uniform", "even_odds"] = "uniform",
        ema_alpha: float = 0.2,
    ) -> None:
        if max_size < 1:
            msg = f"max_size must be >= 1, got {max_size}"
            raise ValueError(msg)
        if not 0.0 < ema_alpha <= 1.0:
            msg = f"ema_alpha must be in (0, 1], got {ema_alpha}"
            raise ValueError(msg)
        self._entries: deque[SnapshotEntry] = deque(maxlen=max_size)
        self._matching: Literal["uniform", "even_odds"] = matching
        self._ema_alpha = ema_alpha
        self._win_rates: dict[int, float] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def updates(self) -> tuple[int, ...]:
        return tuple(entry.update for entry in self._entries)

    @property
    def win_rates(self) -> dict[int, float]:
        return {entry.update: self._win_rates.get(entry.update, 0.5) for entry in self._entries}

    def add(self, module: nn.Module, *, update: int) -> None:
        cloned = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        self._entries.append(SnapshotEntry(update=update, state=cloned))
        self._win_rates[update] = 0.5
        active = {entry.update for entry in self._entries}
        self._win_rates = {key: val for key, val in self._win_rates.items() if key in active}

    def record_outcome(self, update: int, *, win_rate: float) -> None:
        clamped = min(1.0, max(0.0, float(win_rate)))
        prev = self._win_rates.get(update, 0.5)
        self._win_rates[update] = (1.0 - self._ema_alpha) * prev + self._ema_alpha * clamped

    def sampling_weights(self) -> np.ndarray:
        count = len(self._entries)
        if count == 0:
            return np.zeros(0, dtype=np.float64)
        if self._matching == "uniform":
            return np.full(count, 1.0 / count, dtype=np.float64)
        raw = np.array(
            [
                max(_EVEN_ODDS_FLOOR, p * (1.0 - p))
                for p in (self._win_rates.get(entry.update, 0.5) for entry in self._entries)
            ],
            dtype=np.float64,
        )
        return raw / float(raw.sum())

    def sample(self, rng: np.random.Generator) -> SnapshotEntry:
        if not self._entries:
            msg = "cannot sample from an empty snapshot pool"
            raise RuntimeError(msg)
        if self._matching == "uniform":
            idx = int(rng.integers(0, len(self._entries)))
            return self._entries[idx]
        probs = self.sampling_weights()
        idx = int(rng.choice(len(self._entries), p=probs))
        return self._entries[idx]

    def export_state(self) -> dict[str, object]:
        return {
            "matching": self._matching,
            "ema_alpha": self._ema_alpha,
            "entries": [{"update": entry.update, "state": entry.state} for entry in self._entries],
            "win_rates": dict(self._win_rates),
        }

    def restore_state(self, payload: Mapping[str, object]) -> None:
        self._entries.clear()
        self._win_rates.clear()
        for entry in _parse_snapshot_entries(payload.get("entries")):
            self._entries.append(entry)
        active = {entry.update for entry in self._entries}
        raw_rates = payload.get("win_rates")
        if isinstance(raw_rates, Mapping):
            self._win_rates = {
                int(key): float(val)
                for key, val in raw_rates.items()
                if int(key) in active and isinstance(val, (int, float))
            }


def _parse_snapshot_entries(raw: object) -> list[SnapshotEntry]:
    if not isinstance(raw, list):
        return []
    parsed: list[SnapshotEntry] = []
    for item in raw:
        if not isinstance(item, Mapping) or "update" not in item or not isinstance(item.get("state"), Mapping):
            continue
        state = {
            str(key): value.detach().cpu().clone() for key, value in item["state"].items() if isinstance(value, Tensor)
        }
        parsed.append(SnapshotEntry(update=int(item["update"]), state=state))
    return parsed


def load_snapshot(target: nn.Module, entry: SnapshotEntry) -> None:
    with torch.no_grad():
        target.load_state_dict(entry.state)
    target.eval()


@dataclass(frozen=True, slots=True)
class UpdateConfig:
    epochs: int = 1
    minibatch_size: int = 256
    max_grad_norm: float = 0.267
    normalize_advantages: bool = False
    filter_scope: Literal["actor", "actor_critic"] = "actor"
    reference: Literal["uniform", "action_type_uniform"] = "action_type_uniform"
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class UpdateStats:
    rows: int
    selected_rows: int
    completed_games: int
    policy_loss: float
    value_loss: float
    collection_kl: float
    magnet_kl: float
    entropy: float
    clip_fraction: float
    grad_norm: float
    mean_advantage_abs: float


@dataclass(frozen=True, slots=True)
class TrainConfig:
    updates: int
    num_envs: int = 16
    steps_per_env: int = 64
    seed: int = 1
    collect_device: str = "cpu"
    native_workers: Literal[1, 2, 4] = 1
    async_collect: bool = False
    update: UpdateConfig = UpdateConfig()
    games_path: Path | None = None
    record_fraction: float = 0.0
    snapshot_fraction: float = 0.0
    snapshot_every: int = 10
    snapshot_pool_size: int = 16
    snapshot_matching: Literal["uniform", "even_odds"] = "uniform"
    start_update: int = 1
    log_grad_agreement: bool = False
    ema: float | None = None


@dataclass(frozen=True, slots=True)
class StepRecord:
    update: int
    stats: UpdateStats
    collect_s: float
    learn_s: float
    learning_rate: float = 0.0
    magnet_tau: float = 0.0
    opponent: str = "self"
    grad_cosine: float | None = None
    snapshot_win_rate: float | None = None


def should_use_snapshot(update_idx: int, *, fraction: float, pool_size: int) -> bool:
    if pool_size <= 0 or fraction <= 0.0:
        return False
    interval = max(1, round(1.0 / fraction))
    return update_idx % interval == 0


def resolve_snapshot_pool(snapshot_pool: SnapshotPool | None, *, config: TrainConfig) -> SnapshotPool | None:
    if snapshot_pool is not None or config.snapshot_fraction <= 0.0:
        return snapshot_pool
    return SnapshotPool(max_size=config.snapshot_pool_size, matching=config.snapshot_matching)


def update_snapshot_pool(
    pool: SnapshotPool | None,
    module: nn.Module,
    *,
    entry: SnapshotEntry | None,
    snap_wr: float | None,
    update_idx: int,
    snapshot_every: int,
) -> None:
    if pool is None:
        return
    if entry is not None and snap_wr is not None:
        pool.record_outcome(entry.update, win_rate=snap_wr)
    if update_idx % max(snapshot_every, 1) == 0:
        pool.add(module, update=update_idx)

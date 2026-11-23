"""Double-buffered 1-step-lag rollout prefetching for self-play RL.

Overlaps background rollout collection with accelerator learning steps using a
dedicated single-worker thread pool and a detached frozen copy of the policy.
Weights are copied into the frozen policy only at the synchronization barrier
between `take()` and the next `submit()`, so collection and learning never share
mutable tensors. Closing the scheduler drains any in-flight rollout before the
caller closes native environment resources.
"""

from __future__ import annotations

import copy
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Self

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from jace.rl.pool import SnapshotEntry, SnapshotPool

from jace.rl.pool import should_use_snapshot

__all__ = (
    "AsyncPrefetcher",
    "RolloutScheduler",
    "clone_frozen_module",
    "sample_opponent",
    "sync_frozen_module",
)


def sample_opponent(
    pool: SnapshotPool | None, *, rng: np.random.Generator, update_idx: int, fraction: float
) -> SnapshotEntry | None:
    if pool is None or not should_use_snapshot(update_idx, fraction=fraction, pool_size=len(pool)):
        return None
    return pool.sample(rng)


def clone_frozen_module[M: nn.Module](module: M, *, device: torch.device | str) -> M:
    frozen = copy.deepcopy(module).to(device).eval()
    for param in frozen.parameters():
        param.requires_grad = False
    return frozen


def sync_frozen_module(target: nn.Module, source: nn.Module) -> None:
    state = {key: tensor.detach() for key, tensor in source.state_dict().items()}
    with torch.no_grad():
        target.load_state_dict(state)
    target.eval()


class AsyncPrefetcher[T]:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jace-prefetch")
        self._pending: Future[T] | None = None
        self._closed = False

    def submit(self, fn: Callable[[], T]) -> None:
        if self._closed:
            msg = "cannot submit to a closed AsyncPrefetcher"
            raise RuntimeError(msg)
        if self._pending is not None:
            msg = "AsyncPrefetcher already has a pending job"
            raise RuntimeError(msg)
        self._pending = self._executor.submit(fn)

    def take(self) -> tuple[T, float]:
        if self._pending is None:
            msg = "AsyncPrefetcher has no pending job to take"
            raise RuntimeError(msg)
        future = self._pending
        self._pending = None
        t0 = time.perf_counter()
        result = future.result()
        return result, time.perf_counter() - t0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pending is not None:
            wait([self._pending])
            self._pending = None
        self._executor.shutdown(wait=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()


class RolloutScheduler[M: nn.Module, T]:
    """Coordinates synchronous or 1-update overlapped rollout collection.

    In synchronous mode (`async_mode=False`), `next_batch` invokes `collect_fn` on the
    live `agent` directly. In async mode (`async_mode=True`), `prime` starts update 1 in
    the background on a frozen copy of `agent`; each `next_batch` waits for the current
    update's batch, syncs the latest weights into the frozen copy, and immediately
    launches the next update's rollout before returning so collection overlaps with
    `update_step`.
    """

    def __init__(
        self,
        collect_fn: Callable[..., T],
        agent: M,
        *,
        collect_device: str,
        async_mode: bool,
    ) -> None:
        self._collect_fn = collect_fn
        self._async_mode = async_mode
        self._frozen = clone_frozen_module(agent, device=collect_device) if async_mode else None
        self._prefetcher = AsyncPrefetcher[T]() if async_mode else None
        self._in_flight_entry: SnapshotEntry | None = None

    def _submit(self, agent: M, *, opponent_entry: SnapshotEntry | None, update_idx: int) -> None:
        if self._frozen is None or self._prefetcher is None:
            msg = "async prefetching is not enabled on this RolloutScheduler"
            raise RuntimeError(msg)
        sync_frozen_module(self._frozen, agent)
        self._in_flight_entry = opponent_entry
        frozen = self._frozen
        self._prefetcher.submit(lambda: self._collect_fn(frozen, opponent_entry=opponent_entry, update_idx=update_idx))

    def prime(
        self,
        agent: M,
        *,
        pool: SnapshotPool | None,
        rng: np.random.Generator,
        start_update: int,
        total_updates: int,
        snapshot_fraction: float,
    ) -> None:
        if self._async_mode and start_update <= total_updates:
            first_entry = sample_opponent(pool, rng=rng, update_idx=start_update, fraction=snapshot_fraction)
            self._submit(agent, opponent_entry=first_entry, update_idx=start_update)

    def next_batch(
        self,
        agent: M,
        *,
        pool: SnapshotPool | None,
        rng: np.random.Generator,
        update_idx: int,
        total_updates: int,
        snapshot_fraction: float,
    ) -> tuple[T, SnapshotEntry | None, float]:
        if not self._async_mode:
            entry = sample_opponent(pool, rng=rng, update_idx=update_idx, fraction=snapshot_fraction)
            t0 = time.perf_counter()
            batch = self._collect_fn(agent, opponent_entry=entry, update_idx=update_idx)
            return batch, entry, time.perf_counter() - t0
        if self._prefetcher is None:
            msg = "async prefetcher missing in async mode"
            raise RuntimeError(msg)
        entry = self._in_flight_entry
        batch, wait_s = self._prefetcher.take()
        if update_idx < total_updates:
            next_idx = update_idx + 1
            next_entry = sample_opponent(pool, rng=rng, update_idx=next_idx, fraction=snapshot_fraction)
            self._submit(agent, opponent_entry=next_entry, update_idx=next_idx)
        return batch, entry, wait_s

    def close(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()

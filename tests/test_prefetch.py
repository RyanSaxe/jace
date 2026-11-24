"""Unit tests for double-buffered async rollout prefetching and ShardedVectorEnv wiring.

Verifies `RolloutScheduler`'s strict 1-update-lag contract, frozen collector weight
isolation against concurrent learner mutations, and background thread draining on
worker exceptions. Also runs an end-to-end `train_self_play` smoke test with
`native_workers=4` (`ShardedVectorEnv`) and `async_collect=True` on real closed-deck
games.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch
from manabot.training.decks import UniformDecks
from torch import nn

from jace.collect.agent import build_agent
from jace.collect.decks import elemental_decks
from jace.rl import (
    SnapshotEntry,
    SnapshotPool,
    TrainConfig,
    UpdateConfig,
    damped_policy_loss,
    gae_wdl,
    top_fraction,
    train_self_play,
)
from jace.rl.prefetch import AsyncPrefetcher, RolloutScheduler


def test_async_prefetcher_overlaps_and_drains_on_close() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_job() -> int:
        started.set()
        release.wait(timeout=2.0)
        finished.set()
        return 42

    with AsyncPrefetcher[int]() as prefetcher:
        prefetcher.submit(slow_job)
        assert started.wait(timeout=1.0)
        release.set()
        value, wait_s = prefetcher.take()
        assert value == 42
        assert wait_s >= 0.0
    assert finished.is_set()


def test_rollout_scheduler_async_keeps_strict_one_update_lag_and_frozen_weights() -> None:
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    seen: list[tuple[int, float]] = []

    def fake_collect(
        cur_model: nn.Module, *, opponent_entry: SnapshotEntry | None, update_idx: int
    ) -> tuple[int, float]:
        del opponent_entry
        time.sleep(0.02)
        val = float(next(cur_model.parameters())[0, 0].item())
        seen.append((update_idx, val))
        return update_idx, val

    rng = np.random.default_rng(0)
    pool = SnapshotPool(max_size=4)
    with RolloutScheduler(fake_collect, model, collect_device="cpu", async_mode=True) as scheduler:
        scheduler.prime(model, pool=pool, rng=rng, start_update=1, total_updates=4, snapshot_fraction=0.0)
        for update_idx in range(1, 5):
            batch, entry, _ = scheduler.next_batch(
                model, pool=pool, rng=rng, update_idx=update_idx, total_updates=4, snapshot_fraction=0.0
            )
            assert entry is None
            assert batch[0] == update_idx
            # Mutate live model during "learn" while next update collects concurrently on frozen copy.
            with torch.no_grad():
                model.weight.fill_(float(update_idx + 1))

    # Update 1 collected w=1.0; Update 2 primed before update 1 mutated -> w=1.0; Update 3 -> w=2.0; Update 4 -> w=3.0.
    assert seen == [(1, 1.0), (2, 1.0), (3, 2.0), (4, 3.0)]


def test_rollout_scheduler_propagates_worker_exception_and_cleans_up() -> None:
    model = nn.Linear(2, 2)
    rng = np.random.default_rng(0)

    def failing_collect(cur_model: nn.Module, *, opponent_entry: SnapshotEntry | None, update_idx: int) -> int:
        del cur_model, opponent_entry, update_idx
        msg = "boom in collector"
        raise ValueError(msg)

    with RolloutScheduler(failing_collect, model, collect_device="cpu", async_mode=True) as scheduler:
        scheduler.prime(model, pool=None, rng=rng, start_update=1, total_updates=2, snapshot_fraction=0.0)
        with pytest.raises(ValueError, match="boom in collector"):
            scheduler.next_batch(model, pool=None, rng=rng, update_idx=1, total_updates=2, snapshot_fraction=0.0)


def test_train_self_play_with_sharded_native_workers_and_async_collect() -> None:
    agent, obs_space = build_agent(hidden_dim=16, attention_layers=1)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    named = elemental_decks()
    deck_source = UniformDecks(
        decks=tuple(deck for _, deck in named[:2]),
        decklists="closed",
        sampling="uniform-independent-v1",
    )
    cfg = TrainConfig(
        updates=3,
        num_envs=4,
        steps_per_env=4,
        seed=7,
        collect_device="cpu",
        native_workers=4,
        async_collect=True,
        update=UpdateConfig(epochs=1, minibatch_size=64, device="cpu"),
    )
    history = train_self_play(
        agent,
        optimizer,
        obs_space=obs_space,
        deck_source=deck_source,
        credit=gae_wdl(gamma=1.0, lam_policy=0.5, lam_value=0.8),
        select=top_fraction(0.5, minimum=0.01),
        loss=damped_policy_loss(),
        config=cfg,
    )
    assert len(history) == 3
    for stats in history:
        assert stats.rows > 0
        assert np.isfinite(stats.value_loss)
        assert np.isfinite(stats.policy_loss)

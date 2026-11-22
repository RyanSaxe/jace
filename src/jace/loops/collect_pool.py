"""Persistent multi-process CPU rollout collector for Jace v2 self-play.

Upstream `manabot`'s single-process collector caps thread concurrency at four and hits
Python GIL contention well below the sixteen cores of an Apple M3 Max. Spawning one
persistent CPU worker process per core (with `torch.set_num_threads(1)` inside each
process) collects complete-game rollouts in parallel for both current-self play and
frozen-snapshot opponents, then merges every worker's `RolloutBatch` into one flat
batch for MPS learning. Only rank 0 writes Academy training replays so concurrent
processes never interleave JSONL records.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
import torch
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig, RewardConfig
from manabot.training.decks import UniformDecks
from manabot.training.rollout.collection import (
    Collector,
    PackedActions,
    RecordedObservations,
    RolloutBatch,
    close_collector,
    collect,
    collect_pair,
    open_collector,
)

from jace.eval import elemental_decks

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from multiprocessing.process import BaseProcess
    from pathlib import Path

    from jace.rl.pool import SnapshotEntry

__all__ = (
    "MultiProcessCollector",
    "WorkerPoolConfig",
    "merge_rollout_batches",
)

_MIN_CROSS_BALANCED_ENVS = 4
_collect_rollout: Callable[..., RolloutBatch] = collect
_collect_pair_rollout: Callable[..., tuple[RolloutBatch, RolloutBatch]] = collect_pair
_open_collector: Callable[..., Collector] = open_collector

type _WorkerTask = tuple[int, dict[str, torch.Tensor], dict[str, torch.Tensor] | None] | None
type _WorkerResult = tuple[int, RolloutBatch] | tuple[str, int]


@dataclass(frozen=True, slots=True)
class WorkerPoolConfig:
    num_workers: int
    envs_per_worker: int
    steps_per_env: int
    hidden_dim: int
    attention_layers: int
    seed: int = 1
    games_path: Path | None = None
    record_fraction: float = 0.0
    enable_snapshots: bool = False


def _merge_recorded(batches: Sequence[RolloutBatch]) -> RecordedObservations:
    first = batches[0].recorded
    merged_rows = {key: np.concatenate([batch.recorded.rows[key] for batch in batches], axis=0) for key in first.rows}
    offsets: list[np.ndarray] = []
    row_base = 0
    for batch in batches:
        offsets.append(batch.recorded.index + row_base)
        first_array = next(iter(batch.recorded.rows.values()))
        row_base += int(first_array.shape[0])
    merged_needs: np.ndarray | None = None
    if all(batch.recorded.needs is not None for batch in batches):
        merged_needs = np.concatenate([b.recorded.needs for b in batches if b.recorded.needs is not None])
    return RecordedObservations(
        rows=merged_rows,
        index=np.concatenate(offsets, axis=0),
        needs=merged_needs,
    )


def merge_rollout_batches(batches: Sequence[RolloutBatch]) -> RolloutBatch:
    """Concatenate complete-game `RolloutBatch` payloads from parallel workers.

    Joins every worker's time-major rollout arrays (`actions`, `logprobs`, `rewards`,
    `dones`, `values`, and optional `outcome_probabilities`) along axis 0, shifts
    `RecordedObservations.index` by each worker's recorded row count, and concatenates
    `PackedActions` via `PackedActions.concatenate`. Because complete-game rollouts
    terminate every trajectory before closing, `next_obs` and `next_done` carry forward
    from the final batch. Raises `ValueError` when `batches` is empty.
    """
    if not batches:
        msg = "batches must contain at least one RolloutBatch"
        raise ValueError(msg)
    if len(batches) == 1:
        return batches[0]
    outcomes = (
        np.concatenate([b.outcome_probabilities for b in batches if b.outcome_probabilities is not None], axis=0)
        if all(b.outcome_probabilities is not None for b in batches)
        else None
    )
    return RolloutBatch(
        recorded=_merge_recorded(batches),
        actions=np.concatenate([b.actions for b in batches], axis=0),
        logprobs=np.concatenate([b.logprobs for b in batches], axis=0),
        rewards=np.concatenate([b.rewards for b in batches], axis=0),
        dones=np.concatenate([b.dones for b in batches], axis=0),
        values=np.concatenate([b.values for b in batches], axis=0),
        next_obs=batches[-1].next_obs,
        next_done=batches[-1].next_done,
        action_data=PackedActions.concatenate([b.action_data for b in batches]),
        outcome_probabilities=outcomes,
    )


def _sample_pair_match(deck_source: UniformDecks, rng: np.random.Generator) -> Match:
    players = deck_source.players()
    hero = players[int(rng.integers(0, len(players)))]
    villain = players[int(rng.integers(0, len(players)))]
    disclosure = "open" if deck_source.decklists == "open" else "closed"
    return Match(
        MatchConfig(
            world="world",
            decklists=disclosure,
            hero_deck=dict(hero.decklist),
            hero_sideboard=dict(hero.sideboard),
            villain_deck=dict(villain.decklist),
            villain_sideboard=dict(villain.sideboard),
        )
    )


def _build_worker_agent(cfg: WorkerPoolConfig) -> tuple[ActorCritic, ObservationSpace]:
    obs_space = ObservationSpace(
        ObservationConfig(
            policy_input_version=11,
            max_cards_per_player=80,
            max_permanents_per_player=64,
        )
    )
    model_cfg = ModelConfig(
        world="world",
        hidden_dim=cfg.hidden_dim,
        attention_layers=cfg.attention_layers,
        num_attention_heads=4,
        policy_film=True,
        value_kind="categorical_wdl",
        value_aggregation="masked_mean",
    )
    agent = ActorCritic(obs_space, model_cfg, bucketed_shapes=True, live_tokens=False)
    return agent.eval(), obs_space


def _collect_snapshot_on_worker(
    agent: ActorCritic,
    opponent: ActorCritic,
    *,
    obs_space: ObservationSpace,
    deck_source: UniformDecks,
    reward: Reward,
    cfg: WorkerPoolConfig,
    seed: int,
    rng: np.random.Generator,
) -> RolloutBatch:
    pair_match = _sample_pair_match(deck_source, rng)
    curriculum = "cross-balanced" if cfg.envs_per_worker % _MIN_CROSS_BALANCED_ENVS == 0 else "fixed"
    session = CollectionSession(device="cpu", seed=seed)
    collector = _open_collector(
        obs_space,
        pair_match,
        reward,
        num_envs=cfg.envs_per_worker,
        seed=seed,
        opponent_mode="learning",
        opponent_agent=opponent,
        numerics=session,
        learning_seats="both",
        matchup_curriculum=curriculum,
        batch_closure="complete-games",
        workers=1,
    )
    try:
        return _collect_pair_rollout(collector, agent, cfg.steps_per_env)[0]
    finally:
        close_collector(collector)


def _worker_main(
    rank: int,
    cfg: WorkerPoolConfig,
    *,
    req_queue: mp.Queue[_WorkerTask],
    res_queue: mp.Queue[_WorkerResult],
) -> None:
    logging.getLogger("manabot").setLevel(logging.WARNING)
    torch.set_num_threads(1)
    named_decks = elemental_decks()
    deck_source = UniformDecks(
        decks=tuple(deck for _, deck in named_decks),
        decklists="closed",
        sampling="uniform-independent-v1",
    )
    agent, obs_space = _build_worker_agent(cfg)
    opponent = _build_worker_agent(cfg)[0] if cfg.enable_snapshots else None
    worker_seed = cfg.seed + rank * 1000
    rng = np.random.default_rng(worker_seed + 500)
    reward = Reward(RewardConfig())
    # Only rank 0 writes Academy training replays so processes never interleave JSONL lines.
    games_path = cfg.games_path if rank == 0 else None
    record_fraction = min(1.0, cfg.record_fraction * cfg.num_workers) if rank == 0 else 0.0
    session = CollectionSession(device="cpu", seed=worker_seed)
    collector = _open_collector(
        obs_space,
        Match(deck_source.match_config()),
        reward,
        num_envs=cfg.envs_per_worker,
        seed=worker_seed,
        opponent_mode="self",
        numerics=session,
        learning_seats="both",
        batch_closure="complete-games",
        deck_source=deck_source,
        workers=1,
        games_path=games_path,
        record_fraction=record_fraction,
    )
    res_queue.put(("ready", rank))
    try:
        while True:
            task = req_queue.get()
            if task is None:
                break
            update_idx, agent_weights, opp_weights = task
            agent.load_state_dict(agent_weights)
            agent.clear_cache()
            if opp_weights is not None and opponent is not None:
                opponent.load_state_dict(opp_weights)
                opponent.cache_programs()
                batch = _collect_snapshot_on_worker(
                    agent,
                    opponent,
                    obs_space=obs_space,
                    deck_source=deck_source,
                    reward=reward,
                    cfg=cfg,
                    seed=worker_seed + 10_000 + update_idx,
                    rng=rng,
                )
            else:
                batch = _collect_rollout(collector, agent, cfg.steps_per_env)
            res_queue.put((rank, batch))
    finally:
        close_collector(collector)


class MultiProcessCollector:
    """Spawned CPU worker pool that collects complete-game rollouts in parallel.

    Spawns `cfg.num_workers` persistent worker processes once on construction, broadcasts
    CPU weight snapshots on each `collect` call, and merges every worker's complete-game
    `RolloutBatch` in deterministic rank order. Use as a context manager (or call `close`)
    to send shutdown sentinels and join every worker process cleanly.
    """

    _cfg: WorkerPoolConfig
    _req_queues: list[mp.Queue[_WorkerTask]]
    _res_queue: mp.Queue[_WorkerResult]
    _procs: list[BaseProcess]
    _closed: bool

    def __init__(self, cfg: WorkerPoolConfig) -> None:
        if cfg.num_workers < 1:
            msg = f"num_workers must be >= 1, got {cfg.num_workers}"
            raise ValueError(msg)
        self._cfg = cfg
        ctx = mp.get_context("spawn")
        self._req_queues = [ctx.Queue() for _ in range(cfg.num_workers)]
        self._res_queue = ctx.Queue()
        self._procs = [
            ctx.Process(
                target=_worker_main,
                args=(rank, cfg),
                kwargs={"req_queue": self._req_queues[rank], "res_queue": self._res_queue},
            )
            for rank in range(cfg.num_workers)
        ]
        self._closed = False
        for proc in self._procs:
            proc.start()
        for _ in range(cfg.num_workers):
            self._res_queue.get()

    def collect(
        self,
        agent: ActorCritic,
        *,
        opponent_entry: SnapshotEntry | None,
        update_idx: int,
    ) -> RolloutBatch:
        """Broadcast weights to every worker and return one merged `RolloutBatch`.

        Detaches `agent`'s parameters onto CPU before enqueueing so MPS tensors never
        cross process boundaries. When `opponent_entry` is provided, each worker loads
        `opponent_entry.state` into its frozen opponent and collects one-seat games;
        otherwise every worker runs both-seat current-self play on its persistent collector.
        """
        agent_weights = {key: tensor.detach().to("cpu") for key, tensor in agent.state_dict().items()}
        opp_weights = None if opponent_entry is None else opponent_entry.state
        for queue in self._req_queues:
            queue.put((update_idx, agent_weights, opp_weights))
        ordered: dict[int, RolloutBatch] = {}
        for _ in range(self._cfg.num_workers):
            rank_or_tag, payload = self._res_queue.get()
            if isinstance(rank_or_tag, int) and isinstance(payload, RolloutBatch):
                ordered[rank_or_tag] = payload
        return merge_rollout_batches([ordered[rank] for rank in sorted(ordered)])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for queue in self._req_queues:
            queue.put(None)
        for proc in self._procs:
            proc.join(timeout=30.0)
            if proc.is_alive():
                proc.terminate()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()

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
from functools import partial
from queue import Empty
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import torch
from manabot.backends.torch.collection import CollectionSession
from manabot.env.match import Match, Reward
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.training.decks import UniformDecks
from manabot.training.rollout.collection import (
    Collector,
    RolloutBatch,
    close_collector,
    collect,
    collect_pair,
    open_collector,
)
from manabot.training.state.game_errors import GameAttempt, GameErrorPolicy, GameErrorState, settle

from jace.collect.agent import (
    DecisionMemoryKind,
    build_agent,
    merge_rollout_batches,
    pack_batch_for_ipc,
    unpack_batch_from_ipc,
)
from jace.collect.decks import (
    HumanDeckCatalog,
    build_curriculum_deck_source,
    build_human_deck_source,
    elemental_decks,
    load_human_deck_catalog,
    needs_curriculum_refresh,
    top_card_pool,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.process import BaseProcess
    from pathlib import Path

    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.memory_store import MemoryArchive
    from manabot.env.observation import ObservationSpace

    from jace.rl.pool import SnapshotEntry

__all__ = (
    "MultiProcessCollector",
    "WorkerPoolConfig",
    "merge_rollout_batches",
)

_MIN_CROSS_BALANCED_ENVS = 4
_POLL_TIMEOUT_S = 5.0
_GAME_ERROR_POLICY = GameErrorPolicy(threshold=0.05, window=200, warmup=100, recover_capacity=True)
_LOGGER = logging.getLogger(__name__)
_collect_rollout: Callable[..., RolloutBatch] = collect
_collect_pair_rollout: Callable[..., tuple[RolloutBatch, RolloutBatch]] = collect_pair
_open_collector: Callable[..., Collector] = open_collector

type _WorkerTask = tuple[int, dict[str, torch.Tensor], dict[str, torch.Tensor] | None] | None
type _WorkerResult = (
    tuple[int, RolloutBatch, MemoryArchive | None] | tuple[Literal["ready"], int] | tuple[Literal["error"], int, str]
)


def _worker_error_types() -> tuple[type[BaseException], ...]:
    panic_types = tuple(cls for cls in BaseException.__subclasses__() if cls.__name__ == "PanicException")
    return (*panic_types, Exception)


def _record_game_attempt(state: GameErrorState, attempt: GameAttempt, *, rank: int) -> GameErrorState:
    if attempt.outcome != "completed":
        _LOGGER.warning("worker %d isolated %s (seed=%d): %s", rank, attempt.outcome, attempt.seed, attempt.error)
    return settle(state, attempt, _GAME_ERROR_POLICY)


@dataclass(frozen=True, slots=True)
class WorkerPoolConfig:
    num_workers: int
    envs_per_worker: int
    steps_per_env: int
    hidden_dim: int
    attention_layers: int
    decision_memory: DecisionMemoryKind = "none"
    seed: int = 1
    games_path: Path | None = None
    record_fraction: float = 0.0
    enable_snapshots: bool = False
    human_decks_path: Path | None = None
    shard_size: int = 512
    shard_refresh_every: int = 50
    card_pool_size: int | None = None
    curriculum_pools: tuple[tuple[int, int, frozenset[str]], ...] = ()
    curriculum_blend_updates: int = 10
    start_update: int = 1


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
    agent, obs_space = build_agent(
        hidden_dim=cfg.hidden_dim,
        attention_layers=cfg.attention_layers,
        decision_memory=cfg.decision_memory,
        bucketed_shapes=True,
        live_tokens=False,
    )
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
    rank: int,
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
        game_errors=_GAME_ERROR_POLICY,
        record_attempt=partial(_record_game_attempt, rank=rank),
    )
    try:
        return _collect_pair_rollout(collector, agent, cfg.steps_per_env)[0]
    finally:
        close_collector(collector)


def _build_worker_deck_source(
    cfg: WorkerPoolConfig,
    *,
    catalog: HumanDeckCatalog | None,
    allowed_cards: frozenset[str] | None,
    seed: int,
    update_idx: int,
) -> UniformDecks:
    if catalog is not None and cfg.curriculum_pools:
        return build_curriculum_deck_source(
            catalog,
            count=cfg.shard_size,
            seed=seed,
            update_idx=update_idx,
            stage_pools=cfg.curriculum_pools,
            blend_updates=cfg.curriculum_blend_updates,
        )
    if catalog is not None:
        return build_human_deck_source(catalog, count=cfg.shard_size, seed=seed, allowed_cards=allowed_cards)
    named_decks = elemental_decks()
    return UniformDecks(
        decks=tuple(deck for _, deck in named_decks), decklists="closed", sampling="uniform-independent-v1"
    )


def _open_worker_collector(
    obs_space: ObservationSpace,
    *,
    deck_source: UniformDecks,
    reward: Reward,
    cfg: WorkerPoolConfig,
    worker_seed: int,
    rank: int,
    games_path: Path | None,
    record_fraction: float,
) -> Collector:
    match = Match(deck_source.match_config())
    session = CollectionSession(device="cpu", seed=worker_seed)
    return _open_collector(
        obs_space,
        match,
        reward,
        num_envs=cfg.envs_per_worker,
        seed=worker_seed,
        opponent_mode="self",
        numerics=session,
        learning_seats="both",
        batch_closure="complete-games",
        deck_source=deck_source,
        workers=1,
        game_errors=_GAME_ERROR_POLICY,
        record_attempt=partial(_record_game_attempt, rank=rank),
        games_path=games_path,
        record_fraction=record_fraction,
    )


def _should_refresh_shard(cfg: WorkerPoolConfig, *, has_catalog: bool, update_idx: int) -> bool:
    if not has_catalog or update_idx <= 1:
        return False
    if cfg.curriculum_pools:
        stages = [(size, start_u) for size, start_u, _ in cfg.curriculum_pools]
        if needs_curriculum_refresh(update_idx, stages=stages, blend_updates=cfg.curriculum_blend_updates):
            return True
    return cfg.shard_refresh_every > 0 and (update_idx - 1) % cfg.shard_refresh_every == 0


def _run_worker_loop(
    rank: int,
    cfg: WorkerPoolConfig,
    *,
    req_queue: mp.Queue[_WorkerTask],
    res_queue: mp.Queue[_WorkerResult],
) -> None:
    catalog = load_human_deck_catalog(cfg.human_decks_path) if cfg.human_decks_path is not None else None
    allowed_cards = (
        top_card_pool(catalog, card_count=cfg.card_pool_size)
        if catalog is not None and cfg.card_pool_size is not None and not cfg.curriculum_pools
        else None
    )
    worker_seed = cfg.seed + rank * 1000
    deck_source = _build_worker_deck_source(
        cfg, catalog=catalog, allowed_cards=allowed_cards, seed=worker_seed, update_idx=cfg.start_update
    )
    agent, obs_space = _build_worker_agent(cfg)
    opponent = _build_worker_agent(cfg)[0] if cfg.enable_snapshots else None
    rng = np.random.default_rng(worker_seed + 500)
    reward = Reward(RewardConfig())
    # Only rank 0 writes Academy training replays so processes never interleave JSONL lines.
    games_path = cfg.games_path if rank == 0 else None
    record_fraction = min(1.0, cfg.record_fraction * cfg.num_workers) if rank == 0 else 0.0
    collector = _open_worker_collector(
        obs_space,
        deck_source=deck_source,
        reward=reward,
        cfg=cfg,
        worker_seed=worker_seed,
        rank=rank,
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
            if _should_refresh_shard(cfg, has_catalog=catalog is not None, update_idx=update_idx):
                close_collector(collector)
                refresh_seed = worker_seed + update_idx * 37
                deck_source = _build_worker_deck_source(
                    cfg, catalog=catalog, allowed_cards=allowed_cards, seed=refresh_seed, update_idx=update_idx
                )
                collector = _open_worker_collector(
                    obs_space,
                    deck_source=deck_source,
                    reward=reward,
                    cfg=cfg,
                    worker_seed=refresh_seed,
                    rank=rank,
                    games_path=games_path,
                    record_fraction=record_fraction,
                )
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
                    rank=rank,
                    rng=rng,
                )
            else:
                batch = _collect_rollout(collector, agent, cfg.steps_per_env)
            packed, archive = pack_batch_for_ipc(batch)
            res_queue.put((rank, packed, archive))
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
    try:
        _run_worker_loop(rank, cfg, req_queue=req_queue, res_queue=res_queue)
    except _worker_error_types() as exc:
        res_queue.put(("error", rank, f"{type(exc).__name__}: {exc}"))


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
        try:
            for _ in range(cfg.num_workers):
                self._get_result()
        except RuntimeError:
            self.close()
            raise

    def _get_result(self) -> _WorkerResult:
        while True:
            try:
                item = self._res_queue.get(timeout=_POLL_TIMEOUT_S)
            except Empty:
                dead = [
                    (rank, proc.exitcode)
                    for rank, proc in enumerate(self._procs)
                    if not proc.is_alive() and proc.exitcode not in (None, 0)
                ]
                if dead:
                    msg = f"rollout worker exited unexpectedly: {dead}"
                    raise RuntimeError(msg) from None
                continue
            if item[0] == "error":
                msg = f"rollout worker {item[1]} failed: {item[2]}"
                raise RuntimeError(msg)
            return item

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
            item = self._get_result()
            if isinstance(item[0], int):
                ordered[item[0]] = unpack_batch_from_ipc(item[1], item[2])
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

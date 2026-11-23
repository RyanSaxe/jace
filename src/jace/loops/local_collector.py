"""In-process sharded collector for GPU/multi-thread native collection on full 17Lands data.

Wraps `manabot`'s `open_collector` (`ShardedVectorEnv` when `native_workers` is 2 or 4)
with rotating 17Lands deck shards, smooth nested curriculum transitions, snapshot
opponent games, and `GameErrorPolicy` panic isolation. Keeping shard rotation and
curriculum blending in-process lets CUDA and MPS runs use `--collect-workers 0` with the
exact same deck coverage as `MultiProcessCollector` without spawning Python worker
processes.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import torch
from manabot.backends.torch.collection import CollectionSession
from manabot.env.match import Match, Reward
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.training.rollout.collection import (
    Collector,
    RolloutBatch,
    close_collector,
    collect,
    collect_pair,
    open_collector,
)
from manabot.training.state.game_errors import GameAttempt, GameErrorPolicy, GameErrorState, settle

from jace.eval import (
    HumanDeckCatalog,
    build_curriculum_deck_source,
    build_human_deck_source,
    needs_curriculum_refresh,
    top_card_pool,
)
from jace.rl.pool import SnapshotEntry, load_snapshot

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable
    from pathlib import Path

    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace
    from manabot.training.decks import UniformDecks

__all__ = (
    "LocalCollectorConfig",
    "LocalShardedCollector",
    "add_full_data_rl_args",
)

_MIN_CROSS_BALANCED_ENVS = 4
_GAME_ERROR_POLICY = GameErrorPolicy(threshold=0.05, window=200, warmup=100, recover_capacity=True)
_LOGGER = logging.getLogger(__name__)
_collect_rollout: Callable[..., RolloutBatch] = collect
_collect_pair_rollout: Callable[..., tuple[RolloutBatch, RolloutBatch]] = collect_pair
_open_collector: Callable[..., Collector] = open_collector


def _record_attempt(state: GameErrorState, attempt: GameAttempt) -> GameErrorState:
    if attempt.outcome != "completed":
        _LOGGER.warning("local collector isolated %s (seed=%d): %s", attempt.outcome, attempt.seed, attempt.error)
    return settle(state, attempt, _GAME_ERROR_POLICY)


def add_full_data_rl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--decision-memory", choices=("none", "gru", "transformer"), default="none")
    parser.add_argument("--lam-policy", type=float, default=0.5)
    parser.add_argument("--lam-value", type=float, default=0.8)
    parser.add_argument("--select", dest="select_kind", choices=("top25", "top50", "all"), default="top50")
    parser.add_argument("--filter-scope", choices=("actor", "actor_critic"), default="actor")
    parser.add_argument("--min-advantage", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--collection-kl", type=float, default=0.1)
    parser.add_argument("--magnet-kl", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--schedule-kind", choices=("damped", "decay", "none"), default="decay")
    parser.add_argument("--schedule-decay", type=float, default=9.0)
    parser.add_argument("--ema", type=float, default=None)
    parser.add_argument("--snapshot-fraction", type=float, default=0.25)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--snapshot-pool-size", type=int, default=16)
    parser.add_argument("--snapshot-matching", choices=("uniform", "even_odds"), default="even_odds")
    parser.add_argument("--log-grad-agreement", action="store_true", default=True)
    parser.add_argument("--no-log-grad-agreement", dest="log_grad_agreement", action="store_false")
    parser.add_argument("--collect-workers", type=int, default=8)
    parser.add_argument("--native-workers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--async-collect", action="store_true", default=False)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--shard-refresh-every", type=int, default=50)
    parser.add_argument("--card-pool-size", type=int, default=None)
    parser.add_argument("--curriculum-stages", type=str, default=None)
    parser.add_argument("--curriculum-blend-updates", type=int, default=10)


@dataclass(frozen=True, slots=True)
class LocalCollectorConfig:
    num_envs: int
    steps_per_env: int
    collect_device: str
    native_workers: Literal[1, 2, 4] = 1
    seed: int = 1
    games_path: Path | None = None
    record_fraction: float = 0.0
    enable_snapshots: bool = False
    shard_size: int = 512
    shard_refresh_every: int = 50
    card_pool_size: int | None = None
    curriculum_pools: tuple[tuple[int, int, frozenset[str]], ...] = ()
    curriculum_blend_updates: int = 10
    start_update: int = 1


class LocalShardedCollector:
    """In-process `manabot` collector with rotating 17Lands shards and native Rust threads.

    Combines `ShardedVectorEnv` (`native_workers=2` or `4`) with batched `collect_device`
    inference (`cuda`, `mps`, or `cpu`), rotating 17Lands deck shards and advancing nested
    curriculum pools on schedule without spawning separate Python worker processes.
    """

    def __init__(
        self,
        cfg: LocalCollectorConfig,
        *,
        catalog: HumanDeckCatalog,
        obs_space: ObservationSpace,
        template_agent: ActorCritic,
    ) -> None:
        self._cfg = cfg
        self._catalog = catalog
        self._obs_space = obs_space
        self._reward = Reward(RewardConfig())
        self._device = torch.device(cfg.collect_device)
        self._rng = np.random.default_rng(cfg.seed + 202)
        self._allowed = (
            top_card_pool(catalog, card_count=cfg.card_pool_size)
            if not cfg.curriculum_pools and cfg.card_pool_size is not None
            else None
        )
        self._opponent = copy.deepcopy(template_agent).to(self._device).eval() if cfg.enable_snapshots else None
        self._deck_source = self._build_deck_source(cfg.start_update)
        self._collector: Collector | None = self._open_self_collector(cfg.start_update)
        self._closed = False

    def _build_deck_source(self, update_idx: int) -> UniformDecks:
        seed = self._cfg.seed + 7_919 * update_idx
        if self._cfg.curriculum_pools:
            return build_curriculum_deck_source(
                self._catalog,
                count=self._cfg.shard_size,
                seed=seed,
                update_idx=update_idx,
                stage_pools=self._cfg.curriculum_pools,
                blend_updates=self._cfg.curriculum_blend_updates,
            )
        return build_human_deck_source(
            self._catalog, count=self._cfg.shard_size, seed=seed, allowed_cards=self._allowed
        )

    def _needs_refresh(self, update_idx: int) -> bool:
        if update_idx <= self._cfg.start_update:
            return False
        if self._cfg.curriculum_pools:
            stages = [(size, start_u) for size, start_u, _ in self._cfg.curriculum_pools]
            if needs_curriculum_refresh(update_idx, stages=stages, blend_updates=self._cfg.curriculum_blend_updates):
                return True
        every = self._cfg.shard_refresh_every
        return every > 0 and (update_idx - 1) % every == 0

    def _open_self_collector(self, update_idx: int) -> Collector:
        seed = self._cfg.seed + 1_009 * update_idx
        return _open_collector(
            self._obs_space,
            Match(self._deck_source.match_config()),
            self._reward,
            num_envs=self._cfg.num_envs,
            seed=seed,
            opponent_mode="self",
            numerics=CollectionSession(device=self._cfg.collect_device, seed=seed),
            learning_seats="both",
            batch_closure="complete-games",
            deck_source=self._deck_source,
            workers=self._cfg.native_workers,
            game_errors=_GAME_ERROR_POLICY,
            record_attempt=_record_attempt,
            games_path=self._cfg.games_path,
            record_fraction=self._cfg.record_fraction,
        )

    def _collect_snapshot(self, agent: ActorCritic, *, entry: SnapshotEntry, update_idx: int) -> RolloutBatch:
        if self._opponent is None:
            msg = "snapshot opponent requested when enable_snapshots=False"
            raise RuntimeError(msg)
        load_snapshot(self._opponent, entry)
        self._opponent.to(self._device)
        self._opponent.cache_programs()
        players = self._deck_source.players()
        hero = players[int(self._rng.integers(0, len(players)))]
        villain = players[int(self._rng.integers(0, len(players)))]
        disclosure: Literal["open", "closed"] = "open" if self._deck_source.decklists == "open" else "closed"
        match = Match(MatchConfig.from_players(hero, villain, decklists=disclosure))
        curriculum: Literal["cross-balanced", "fixed"] = (
            "cross-balanced" if self._cfg.num_envs % _MIN_CROSS_BALANCED_ENVS == 0 else "fixed"
        )
        seed = self._cfg.seed + 10_000 + update_idx
        collector = _open_collector(
            self._obs_space,
            match,
            self._reward,
            num_envs=self._cfg.num_envs,
            seed=seed,
            opponent_mode="learning",
            opponent_agent=self._opponent,
            numerics=CollectionSession(device=self._cfg.collect_device, seed=seed),
            learning_seats="both",
            matchup_curriculum=curriculum,
            batch_closure="complete-games",
            workers=self._cfg.native_workers,
            game_errors=_GAME_ERROR_POLICY,
            record_attempt=_record_attempt,
        )
        try:
            return _collect_pair_rollout(collector, agent, self._cfg.steps_per_env)[0]
        finally:
            close_collector(collector)

    def collect(
        self,
        agent: ActorCritic,
        *,
        opponent_entry: SnapshotEntry | None,
        update_idx: int,
    ) -> RolloutBatch:
        if self._closed or self._collector is None:
            msg = "LocalShardedCollector is closed"
            raise RuntimeError(msg)
        if self._needs_refresh(update_idx):
            close_collector(self._collector)
            self._deck_source = self._build_deck_source(update_idx)
            self._collector = self._open_self_collector(update_idx)
        agent.to(self._device)
        agent.eval()
        if opponent_entry is not None:
            return self._collect_snapshot(agent, entry=opponent_entry, update_idx=update_idx)
        return _collect_rollout(self._collector, agent, self._cfg.steps_per_env)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._collector is not None:
            close_collector(self._collector)
            self._collector = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()

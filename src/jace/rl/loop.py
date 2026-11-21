"""Self-play reinforcement learning update step and training loop on `manabot` ActorCritic.

Bridges upstream `manabot` collection (`open_collector`, `collect`, `AdmittedRollout`,
`ActorCritic`) to Jace's pure RL math (`credit`, `select`, `losses`, `pool`) without
importing any upstream learner update or recipe module. Separate `collect_device` and
`learn_device` settings let Apple Silicon runs collect on CPU streams while updating on
MPS, and an optional historical `SnapshotPool` mixes frozen past opponents into self-play.
"""

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.policy.prepare import reference_distribution
from manabot.training.decks import UniformDecks
from manabot.training.rollout.collection import (
    Collector,
    RolloutBatch,
    close_collector,
    collect,
    open_collector,
)
from torch.optim import Optimizer

from jace.rl.credit import CreditFn, Trajectory, standardize_advantages
from jace.rl.losses import LossFn, PolicyBatch
from jace.rl.pool import SnapshotEntry, SnapshotPool, load_snapshot
from jace.rl.select import SelectFn

__all__ = (
    "LossScheduleFn",
    "StepRecord",
    "TrainConfig",
    "UpdateCallback",
    "UpdateConfig",
    "UpdateStats",
    "train_self_play",
    "update_step",
)

_SCALAR_TARGET_NDIM = 2
_CROSS_BALANCED_STRIDE = 4
_collect_rollout: Callable[..., RolloutBatch] = collect
_open_collector: Callable[..., Collector] = open_collector


@dataclass(frozen=True, slots=True)
class UpdateConfig:
    epochs: int = 1
    minibatch_size: int = 256
    max_grad_norm: float = 0.267
    normalize_advantages: bool = False
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
    update: UpdateConfig = UpdateConfig()
    games_path: Path | None = None
    record_fraction: float = 0.0
    snapshot_fraction: float = 0.0
    snapshot_every: int = 10
    snapshot_pool_size: int = 16


@dataclass(frozen=True, slots=True)
class StepRecord:
    update: int
    stats: UpdateStats
    collect_s: float
    learn_s: float
    learning_rate: float = 0.0
    magnet_tau: float = 0.0
    opponent: str = "self"


type UpdateCallback = Callable[[StepRecord], None]
type LossScheduleFn = Callable[[int], tuple[float, float, LossFn]]


def _bootstrap_values(
    agent: ActorCritic,
    rollout: AdmittedRollout,
    *,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    values: list[torch.Tensor] = []
    outcomes: list[torch.Tensor] = []
    was_training = agent.training
    agent.eval()
    try:
        with torch.no_grad():
            for obs in rollout.next_observations(device, chunk_size):
                _, raw_value = agent.forward_distribution(obs)
                raw_value = raw_value.float()
                if agent.config.value_kind == "categorical_wdl":
                    probs = raw_value.softmax(dim=-1)
                    values.append((probs[:, 2] - probs[:, 0]).cpu())
                    outcomes.append(probs.cpu())
                else:
                    values.append(raw_value.squeeze(-1).cpu())
    finally:
        if was_training:
            agent.train()
    next_value = torch.cat(values, dim=0)
    next_outcomes = torch.cat(outcomes, dim=0) if outcomes else None
    return next_value, next_outcomes


def _trajectory_from_batch(
    batch: RolloutBatch,
    *,
    next_value: torch.Tensor,
    next_outcomes: torch.Tensor | None,
) -> Trajectory:
    outcome_probs = (
        torch.as_tensor(batch.outcome_probabilities, dtype=torch.float32)
        if batch.outcome_probabilities is not None
        else None
    )
    return Trajectory(
        rewards=torch.as_tensor(batch.rewards, dtype=torch.float32),
        values=torch.as_tensor(batch.values, dtype=torch.float32),
        dones=torch.as_tensor(batch.dones, dtype=torch.bool),
        next_value=next_value,
        outcome_probs=outcome_probs,
        next_outcome_probs=next_outcomes,
    )


def _minibatch_step(
    agent: ActorCritic,
    optimizer: Optimizer,
    *,
    batch: PolicyBatch,
    loss_fn: LossFn,
    max_grad_norm: float,
) -> tuple[dict[str, float], float]:
    breakdown = loss_fn(batch)
    optimizer.zero_grad(set_to_none=True)
    breakdown.total.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm).item())
    optimizer.step()
    metrics = {
        "policy_loss": breakdown.policy,
        "value_loss": breakdown.value,
        "collection_kl": breakdown.collection_kl,
        "magnet_kl": breakdown.magnet_kl,
        "entropy": breakdown.entropy,
        "clip_fraction": breakdown.clip_fraction,
    }
    return metrics, grad_norm


def _set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _current_optimizer_lr(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0


def _should_use_snapshot(update_idx: int, *, fraction: float, pool_size: int) -> bool:
    if pool_size <= 0 or fraction <= 0.0:
        return False
    interval = max(1, round(1.0 / fraction))
    return update_idx % interval == 0


def _sample_pair_match(deck_source: UniformDecks, rng: np.random.Generator) -> Match:
    players = deck_source.players()
    hero = players[int(rng.integers(0, len(players)))]
    villain = players[int(rng.integers(0, len(players)))]
    disclosure: Literal["open", "closed"] = "open" if deck_source.decklists == "open" else "closed"
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


def _collect_against_snapshot(
    agent: ActorCritic,
    opponent: ActorCritic,
    *,
    entry: SnapshotEntry,
    obs_space: ObservationSpace,
    deck_source: UniformDecks,
    reward: Reward,
    config: TrainConfig,
    seed: int,
    rng: np.random.Generator,
) -> RolloutBatch:
    load_snapshot(opponent, entry)
    opponent.to(torch.device(config.collect_device))
    pair_match = _sample_pair_match(deck_source, rng)
    curriculum: Literal["cross-balanced", "fixed"] = (
        "cross-balanced" if config.num_envs % _CROSS_BALANCED_STRIDE == 0 else "fixed"
    )
    session = CollectionSession(device=config.collect_device, seed=seed)
    collector = _open_collector(
        obs_space,
        pair_match,
        reward,
        num_envs=config.num_envs,
        seed=seed,
        opponent_mode="frozen",
        opponent_agent=opponent,
        numerics=session,
        learning_seats="one",
        matchup_curriculum=curriculum,
        batch_closure="complete-games",
    )
    try:
        return _collect_rollout(collector, agent, config.steps_per_env)
    finally:
        close_collector(collector)


def update_step(
    agent: ActorCritic,
    optimizer: Optimizer,
    *,
    batch: RolloutBatch,
    credit: CreditFn,
    select: SelectFn,
    loss: LossFn,
    config: UpdateConfig,
    rng: np.random.Generator,
    learning_seats: Literal["both", "one"] = "both",
) -> UpdateStats:
    """Run one on-policy RL update over a collected `RolloutBatch`.

    Computes advantages and value targets with `credit`, selects policy rows with
    `select` while keeping every row for critic supervision, leaves advantages at their
    raw win-probability scale unless `config.normalize_advantages` is set, admits
    observations once, and runs `config.epochs` shuffled minibatch passes using `loss`.
    """
    device = torch.device(config.device)
    agent.to(device)
    rollout = AdmittedRollout(agent, batch)
    next_value, next_outcomes = _bootstrap_values(
        agent,
        rollout,
        device=device,
        chunk_size=config.minibatch_size,
    )
    targets = credit(_trajectory_from_batch(batch, next_value=next_value, next_outcomes=next_outcomes))
    raw_flat = targets.raw_advantages.reshape(-1)
    selected_mask = select(raw_flat).bool()
    adv_flat = standardize_advantages(raw_flat, mask=selected_mask) if config.normalize_advantages else raw_flat
    value_targets = targets.value_targets.reshape(raw_flat.shape[0], -1)
    if targets.value_targets.ndim == _SCALAR_TARGET_NDIM:
        value_targets = targets.value_targets.reshape(-1)

    total_rows = int(batch.actions.size)
    all_indexes = np.arange(total_rows, dtype=np.int64)
    rollout.admit_rows(all_indexes, chunk_rows=config.minibatch_size)
    flat_actions = batch.actions.reshape(-1)

    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "collection_kl": 0.0,
        "magnet_kl": 0.0,
        "entropy": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
    }
    steps = 0
    agent.train()
    for _ in range(config.epochs):
        order = rng.permutation(total_rows).astype(np.int64)
        with rollout.minibatches(order, config.minibatch_size, device) as minibatches:
            for idx, obs_t, rows in minibatches:
                logits, value_output = agent.forward_distribution(obs_t)
                ref = reference_distribution(rows.observation, config.reference)
                policy_batch = PolicyBatch(
                    logits=logits,
                    value_output=value_output,
                    actions=torch.as_tensor(flat_actions[idx], dtype=torch.long, device=device),
                    advantages=adv_flat[idx].to(device),
                    value_targets=value_targets[idx].to(device),
                    behavior_logs=torch.as_tensor(rows.log_probabilities, dtype=torch.float32, device=device),
                    reference_probs=torch.as_tensor(ref, dtype=torch.float32, device=device),
                    legal_mask=rollout.legal(obs_t).bool(),
                    policy_mask=selected_mask[idx],
                )
                metrics, grad_norm = _minibatch_step(
                    agent,
                    optimizer,
                    batch=policy_batch,
                    loss_fn=loss,
                    max_grad_norm=config.max_grad_norm,
                )
                for key, value in metrics.items():
                    totals[key] += value
                totals["grad_norm"] += grad_norm
                steps += 1

    denom = max(steps, 1)
    seats_per_game = 2 if learning_seats == "both" else 1
    return UpdateStats(
        rows=total_rows,
        selected_rows=int(selected_mask.sum().item()),
        completed_games=int(batch.dones.sum()) // seats_per_game,
        policy_loss=totals["policy_loss"] / denom,
        value_loss=totals["value_loss"] / denom,
        collection_kl=totals["collection_kl"] / denom,
        magnet_kl=totals["magnet_kl"] / denom,
        entropy=totals["entropy"] / denom,
        clip_fraction=totals["clip_fraction"] / denom,
        grad_norm=totals["grad_norm"] / denom,
        mean_advantage_abs=float(raw_flat.abs().mean().item()),
    )


def _resolve_update_loss(
    update_idx: int,
    *,
    optimizer: Optimizer,
    default_loss: LossFn,
    loss_schedule: LossScheduleFn | None,
) -> tuple[float, float, LossFn]:
    if loss_schedule is None:
        return _current_optimizer_lr(optimizer), 0.0, default_loss
    lr, tau, scheduled_loss = loss_schedule(update_idx)
    _set_optimizer_lr(optimizer, lr)
    return lr, tau, scheduled_loss


def train_self_play(
    agent: ActorCritic,
    optimizer: Optimizer,
    *,
    obs_space: ObservationSpace,
    deck_source: UniformDecks,
    credit: CreditFn,
    select: SelectFn,
    loss: LossFn,
    config: TrainConfig,
    loss_schedule: LossScheduleFn | None = None,
    on_update: UpdateCallback | None = None,
) -> list[UpdateStats]:
    """Run complete-game self-play training over `deck_source` using Jace's RL pipeline.

    Collects both seats under the current frozen policy via `manabot`'s native collector,
    optionally mixes 1-seat updates against sampled historical checkpoints when
    `config.snapshot_fraction > 0`, applies `loss_schedule` and `update_step` for
    `config.updates` iterations, and invokes `on_update` after each update.
    """
    deck_source.check()
    match = Match(deck_source.match_config())
    reward = Reward(RewardConfig())
    session = CollectionSession(device=config.collect_device, seed=config.seed)
    collector = open_collector(
        obs_space,
        match,
        reward,
        num_envs=config.num_envs,
        seed=config.seed,
        opponent_mode="self",
        numerics=session,
        learning_seats="both",
        batch_closure="complete-games",
        deck_source=deck_source,
        games_path=config.games_path,
        record_fraction=config.record_fraction,
    )
    rng = np.random.default_rng(config.seed + 101)
    pool = SnapshotPool(max_size=config.snapshot_pool_size) if config.snapshot_fraction > 0.0 else None
    frozen_opponent = copy.deepcopy(agent).to(config.collect_device).eval() if pool is not None else None
    history: list[UpdateStats] = []
    collect_device = torch.device(config.collect_device)
    try:
        for update_idx in range(1, config.updates + 1):
            lr, tau, step_loss = _resolve_update_loss(
                update_idx,
                optimizer=optimizer,
                default_loss=loss,
                loss_schedule=loss_schedule,
            )
            agent.to(collect_device)
            agent.eval()
            t_collect = time.perf_counter()
            if (
                pool is not None
                and frozen_opponent is not None
                and _should_use_snapshot(update_idx, fraction=config.snapshot_fraction, pool_size=len(pool))
            ):
                entry = pool.sample(rng)
                batch = _collect_against_snapshot(
                    agent,
                    frozen_opponent,
                    entry=entry,
                    obs_space=obs_space,
                    deck_source=deck_source,
                    reward=reward,
                    config=config,
                    seed=config.seed + 10_000 + update_idx,
                    rng=rng,
                )
                seats: Literal["both", "one"] = "one"
                opponent_label = f"snapshot@u{entry.update}"
            else:
                batch = _collect_rollout(collector, agent, config.steps_per_env)
                seats = "both"
                opponent_label = "self"
            collect_s = time.perf_counter() - t_collect

            t_learn = time.perf_counter()
            stats = update_step(
                agent,
                optimizer,
                batch=batch,
                credit=credit,
                select=select,
                loss=step_loss,
                config=config.update,
                rng=rng,
                learning_seats=seats,
            )
            learn_s = time.perf_counter() - t_learn
            if pool is not None and update_idx % max(config.snapshot_every, 1) == 0:
                pool.add(agent, update=update_idx)
            history.append(stats)
            if on_update is not None:
                on_update(
                    StepRecord(
                        update=update_idx,
                        stats=stats,
                        collect_s=collect_s,
                        learn_s=learn_s,
                        learning_rate=lr,
                        magnet_tau=tau,
                        opponent=opponent_label,
                    )
                )
    finally:
        close_collector(collector)
    return history

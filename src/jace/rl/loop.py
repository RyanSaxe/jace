"""Self-play reinforcement learning update step and training loop on `manabot` ActorCritic.

Bridges upstream `manabot` collection (`AdmittedRollout`, `ActorCritic`) to Jace's pure
RL math (`credit`, `select`, `losses`, `pool`, `diagnostics`) without importing any
upstream learner update or recipe module. Separate `collect_device` and `learn_device`
settings let Apple Silicon runs collect on CPU streams while updating on MPS with
length-bucketed minibatches and once-per-epoch catalog program encoding.
"""

from __future__ import annotations

import copy
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import torch
from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.env.match import Match, Reward
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.policy.prepare import catalog_programs, reference_distribution
from manabot.training.rollout.collection import (
    Collector,
    RolloutBatch,
    close_collector,
    collect,
    collect_pair,
    open_collector,
)

from jace.rl.credit import CreditFn, Trajectory, standardize_advantages
from jace.rl.diagnostics import GradientAgreementTracker, snapshot_win_rate
from jace.rl.ema import EmaWeights, resolve_ema_weights
from jace.rl.losses import LossFn, PolicyBatch
from jace.rl.pool import (
    SnapshotEntry,
    SnapshotPool,
    StepRecord,
    TrainConfig,
    UpdateConfig,
    UpdateStats,
    current_optimizer_lr,
    load_snapshot,
    resolve_snapshot_pool,
    set_optimizer_lr,
    should_use_snapshot,
    update_snapshot_pool,
)
from jace.rl.select import SelectFn, bucketed_minibatch_order, training_row_indexes

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace
    from manabot.training.decks import UniformDecks
    from torch.optim import Optimizer

__all__ = (
    "AuxLossFn",
    "CollectFn",
    "LossScheduleFn",
    "PrepareAuxFn",
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
_METRIC_KEYS = ("policy_loss", "value_loss", "collection_kl", "magnet_kl", "entropy", "clip_fraction", "grad_norm")
_collect_rollout: Callable[..., RolloutBatch] = collect
_collect_pair_rollout: Callable[..., tuple[RolloutBatch, RolloutBatch]] = collect_pair
_open_collector: Callable[..., Collector] = open_collector

type UpdateCallback = Callable[[StepRecord], None]
type LossScheduleFn = Callable[[int], tuple[float, float, LossFn]]
type AuxLossFn = Callable[[], torch.Tensor | None]
type PrepareAuxFn = Callable[[ActorCritic, int], AuxLossFn | None]


class CollectFn(Protocol):
    def __call__(
        self, agent: ActorCritic, *, opponent_entry: SnapshotEntry | None, update_idx: int
    ) -> RolloutBatch: ...


def _bootstrap_values(
    agent: ActorCritic, rollout: AdmittedRollout, *, device: torch.device, chunk_size: int
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
    return torch.cat(values, dim=0), (torch.cat(outcomes, dim=0) if outcomes else None)


def _trajectory_from_batch(
    batch: RolloutBatch, *, next_value: torch.Tensor, next_outcomes: torch.Tensor | None
) -> Trajectory:
    outcome_probs = (
        None
        if batch.outcome_probabilities is None
        else torch.as_tensor(batch.outcome_probabilities, dtype=torch.float32)
    )
    return Trajectory(
        rewards=torch.as_tensor(batch.rewards, dtype=torch.float32),
        values=torch.as_tensor(batch.values, dtype=torch.float32),
        dones=torch.as_tensor(batch.dones, dtype=torch.bool),
        next_value=next_value,
        outcome_probs=outcome_probs,
        next_outcome_probs=next_outcomes,
    )


@contextmanager
def _epoch_catalog_cache(
    agent: ActorCritic, optimizer: Optimizer, *, device: torch.device, max_grad_norm: float
) -> Generator[tuple[torch.Tensor, torch.Tensor] | None]:
    # Encode the semantic catalog once per epoch and accumulate gradients across minibatches.
    semantic = agent.semantic_cards
    if semantic is None:
        yield None
        return
    tokens_np = semantic.get_buffer("tokens").detach().cpu().numpy()
    lengths_np = semantic.get_buffer("lengths").detach().cpu().numpy()
    full_defs = semantic.definitions(on_device(catalog_programs(tokens_np, lengths_np), device))
    defs_leaf = full_defs.detach().clone()
    defs_leaf.requires_grad = True
    defs_grad = torch.zeros_like(defs_leaf)

    def _cached_definitions(_programs: object) -> torch.Tensor:
        return defs_leaf

    semantic.__dict__["definitions"] = _cached_definitions
    try:
        yield defs_leaf, defs_grad
    finally:
        semantic.__dict__.pop("definitions", None)
    if float(defs_grad.abs().sum().item()) > 0.0:
        optimizer.zero_grad(set_to_none=True)
        full_defs.backward(defs_grad)
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()


def _minibatch_step(
    agent: ActorCritic,
    optimizer: Optimizer,
    *,
    batch: PolicyBatch,
    loss_fn: LossFn,
    max_grad_norm: float,
    catalog_pair: tuple[torch.Tensor, torch.Tensor] | None = None,
    aux_loss: AuxLossFn | None = None,
    grad_tracker: GradientAgreementTracker | None = None,
) -> tuple[dict[str, float], float]:
    breakdown = loss_fn(batch)
    total = breakdown.total
    if aux_loss is not None:
        aux = aux_loss()
        if aux is not None:
            total = total + aux.to(device=total.device).to(dtype=total.dtype)
    if catalog_pair is not None:
        catalog_pair[0].grad = None
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    if catalog_pair is not None and catalog_pair[0].grad is not None:
        catalog_pair[1].add_(catalog_pair[0].grad)
    if grad_tracker is not None:
        grad_tracker.accumulate(agent.parameters())
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
    opponent.cache_programs()
    players = deck_source.players()
    hero, villain = players[int(rng.integers(0, len(players)))], players[int(rng.integers(0, len(players)))]
    disclosure: Literal["open", "closed"] = "open" if deck_source.decklists == "open" else "closed"
    match = Match(MatchConfig.from_players(hero, villain, decklists=disclosure))
    curriculum: Literal["cross-balanced", "fixed"] = (
        "cross-balanced" if config.num_envs % _CROSS_BALANCED_STRIDE == 0 else "fixed"
    )
    collector = _open_collector(
        obs_space,
        match,
        reward,
        num_envs=config.num_envs,
        seed=seed,
        opponent_mode="learning",
        opponent_agent=opponent,
        numerics=CollectionSession(device=config.collect_device, seed=seed),
        learning_seats="both",
        matchup_curriculum=curriculum,
        batch_closure="complete-games",
    )
    try:
        return _collect_pair_rollout(collector, agent, config.steps_per_env)[0]
    finally:
        close_collector(collector)


def _open_local_collector(
    agent: ActorCritic,
    *,
    obs_space: ObservationSpace,
    deck_source: UniformDecks,
    config: TrainConfig,
    enable_snapshots: bool,
    rng: np.random.Generator,
) -> tuple[Collector, CollectFn]:
    reward = Reward(RewardConfig())
    collector = _open_collector(
        obs_space,
        Match(deck_source.match_config()),
        reward,
        num_envs=config.num_envs,
        seed=config.seed,
        opponent_mode="self",
        numerics=CollectionSession(device=config.collect_device, seed=config.seed),
        learning_seats="both",
        batch_closure="complete-games",
        deck_source=deck_source,
        games_path=config.games_path,
        record_fraction=config.record_fraction,
    )
    frozen = copy.deepcopy(agent).to(config.collect_device).eval() if enable_snapshots else None
    collect_device = torch.device(config.collect_device)

    def _collect(agent: ActorCritic, *, opponent_entry: SnapshotEntry | None, update_idx: int) -> RolloutBatch:
        agent.to(collect_device)
        agent.eval()
        if opponent_entry is not None and frozen is not None:
            return _collect_against_snapshot(
                agent,
                frozen,
                entry=opponent_entry,
                obs_space=obs_space,
                deck_source=deck_source,
                reward=reward,
                config=config,
                seed=config.seed + 10_000 + update_idx,
                rng=rng,
            )
        return _collect_rollout(collector, agent, config.steps_per_env)

    return collector, _collect


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
    aux_loss: AuxLossFn | None = None,
    grad_tracker: GradientAgreementTracker | None = None,
) -> UpdateStats:
    """Run one on-policy RL update over a collected `RolloutBatch`.

    Computes advantages and value targets with `credit`, selects policy rows with
    `select` while keeping every row for critic supervision, leaves advantages at their
    raw win-probability scale unless `config.normalize_advantages` is set, admits
    observations once, and runs `config.epochs` length-bucketed minibatch passes with
    once-per-epoch semantic catalog encoding.
    """
    device = torch.device(config.device)
    agent.to(device)
    rollout = AdmittedRollout(agent, batch)
    next_value, next_outcomes = _bootstrap_values(agent, rollout, device=device, chunk_size=config.minibatch_size)
    targets = credit(_trajectory_from_batch(batch, next_value=next_value, next_outcomes=next_outcomes))
    raw_flat = targets.raw_advantages.reshape(-1)
    selected_mask = select(raw_flat).bool()
    adv_flat = standardize_advantages(raw_flat, mask=selected_mask) if config.normalize_advantages else raw_flat
    value_targets = (
        targets.value_targets.reshape(-1)
        if targets.value_targets.ndim == _SCALAR_TARGET_NDIM
        else targets.value_targets.reshape(raw_flat.shape[0], -1)
    )
    total_rows = int(batch.actions.size)
    active_indexes = training_row_indexes(selected_mask, filter_scope=config.filter_scope)
    rollout.admit_rows(active_indexes, chunk_rows=config.minibatch_size)
    raw_needs = batch.recorded.gather_needs(active_indexes)
    row_sizes = None if raw_needs is None else raw_needs.sum(axis=1)
    flat_actions = batch.actions.reshape(-1)
    totals = dict.fromkeys(_METRIC_KEYS, 0.0)
    steps = 0
    if grad_tracker is not None:
        grad_tracker.begin_update()
    agent.train()
    for _ in range(config.epochs):
        rel_order = bucketed_minibatch_order(
            len(active_indexes), minibatch_size=config.minibatch_size, row_sizes=row_sizes, rng=rng
        )
        order = active_indexes[rel_order]
        with (
            _epoch_catalog_cache(agent, optimizer, device=device, max_grad_norm=config.max_grad_norm) as catalog_pair,
            rollout.minibatches(order, config.minibatch_size, device) as minibatches,
        ):
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
                    catalog_pair=catalog_pair,
                    aux_loss=aux_loss,
                    grad_tracker=grad_tracker,
                )
                for key, value in metrics.items():
                    totals[key] += value
                totals["grad_norm"] += grad_norm
                steps += 1
    if device.type == "mps":
        torch.mps.empty_cache()
    denom = max(steps, 1)
    seats_per_game = 2 if learning_seats == "both" else 1
    return UpdateStats(
        total_rows,
        int(selected_mask.sum().item()),
        int(batch.dones.sum()) // seats_per_game,
        *(totals[k] / denom for k in _METRIC_KEYS),
        float(raw_flat.abs().mean().item()),
    )


def _resolve_update_loss(
    update_idx: int, *, optimizer: Optimizer, default_loss: LossFn, loss_schedule: LossScheduleFn | None
) -> tuple[float, float, LossFn]:
    if loss_schedule is None:
        return current_optimizer_lr(optimizer), 0.0, default_loss
    lr, tau, scheduled_loss = loss_schedule(update_idx)
    set_optimizer_lr(optimizer, lr)
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
    collect_fn: CollectFn | None = None,
    prepare_aux: PrepareAuxFn | None = None,
    on_update: UpdateCallback | None = None,
    snapshot_pool: SnapshotPool | None = None,
    ema_weights: EmaWeights | None = None,
) -> list[UpdateStats]:
    """Run complete-game self-play training over `deck_source` using Jace's RL pipeline.

    Collects both seats under the current frozen policy (or via an injected `collect_fn`),
    mixes optional 1-seat snapshot updates (uniform or even-odds), advances optional EMA
    shadow weights after each update, logs optional gradient cosine agreement across
    consecutive updates, and invokes `on_update` per step.
    """
    deck_source.check()
    rng = np.random.default_rng(config.seed + 101)
    pool = resolve_snapshot_pool(snapshot_pool, config=config)
    ema = resolve_ema_weights(ema_weights, agent, config=config)
    grad_tracker = GradientAgreementTracker() if config.log_grad_agreement else None
    collector: Collector | None = None
    if collect_fn is None:
        collector, active_collect = _open_local_collector(
            agent,
            obs_space=obs_space,
            deck_source=deck_source,
            config=config,
            enable_snapshots=pool is not None,
            rng=rng,
        )
    else:
        active_collect = collect_fn
    history: list[UpdateStats] = []
    try:
        for update_idx in range(config.start_update, config.updates + 1):
            lr, tau, step_loss = _resolve_update_loss(
                update_idx, optimizer=optimizer, default_loss=loss, loss_schedule=loss_schedule
            )
            entry = (
                pool.sample(rng)
                if pool is not None
                and should_use_snapshot(update_idx, fraction=config.snapshot_fraction, pool_size=len(pool))
                else None
            )
            t_collect = time.perf_counter()
            batch = active_collect(agent, opponent_entry=entry, update_idx=update_idx)
            collect_s = time.perf_counter() - t_collect
            snap_wr = snapshot_win_rate(batch.rewards, batch.dones) if entry is not None else None
            aux_loss = prepare_aux(agent, update_idx) if prepare_aux is not None else None
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
                learning_seats="one" if entry is not None else "both",
                aux_loss=aux_loss,
                grad_tracker=grad_tracker,
            )
            learn_s = time.perf_counter() - t_learn
            if ema is not None:
                ema.update(agent)
            grad_cosine = grad_tracker.finish_update() if grad_tracker is not None else None
            update_snapshot_pool(
                pool, agent, entry=entry, snap_wr=snap_wr, update_idx=update_idx, snapshot_every=config.snapshot_every
            )
            history.append(stats)
            if on_update is not None:
                opp = f"snapshot@u{entry.update}" if entry is not None else "self"
                on_update(StepRecord(update_idx, stats, collect_s, learn_s, lr, tau, opp, grad_cosine, snap_wr))
    finally:
        if collector is not None:
            close_collector(collector)
    return history

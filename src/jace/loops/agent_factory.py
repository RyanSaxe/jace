"""Shared v11 policy construction and decision-memory-safe rollout IPC helpers.

Centralizes `ActorCritic` + `ObservationSpace` creation across self-play loops,
multi-process rollout workers, and batched counterfactual fork workers so every
process constructs identical v11 observation capacities (including optional
decision-memory buffers). Also packs `RecordedObservations.memory` prefixes into
compact `MemoryArchive` payloads for multiprocessing IPC and merges worker batches
without dropping decision history.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from manabot.backends.torch.policy import ActorCritic
from manabot.env.memory_store import MemoryArchive, archive_memory, restore_memory
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.training.rollout.collection import PackedActions, RecordedObservations, RolloutBatch

from jace.rl import (
    CreditFn,
    DampedSchedule,
    DecaySchedule,
    LossFn,
    LossScheduleFn,
    SelectFn,
    all_rows,
    damped_policy_loss,
    gae_wdl,
    top_fraction,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    "DECISION_MEMORY_CAPACITIES",
    "DecisionMemoryKind",
    "build_agent",
    "build_damped_rl_components",
    "merge_rollout_batches",
    "pack_batch_for_ipc",
    "unpack_batch_from_ipc",
)

type DecisionMemoryKind = Literal["none", "gru", "transformer"]

DECISION_MEMORY_CAPACITIES: tuple[int, int, int, int] = (1024, 8192, 256, 16384)


def build_agent(
    *,
    hidden_dim: int = 64,
    attention_layers: int = 2,
    decision_memory: DecisionMemoryKind = "none",
    bucketed_shapes: bool = True,
    live_tokens: bool = False,
) -> tuple[ActorCritic, ObservationSpace]:
    """Build a Jace v2 `ActorCritic` and matching v11 `ObservationSpace`.

    When `decision_memory` is `"none"`, leaves all four decision-memory observation
    capacities at zero for exact backwards compatibility with stateless checkpoints.
    When set to `"gru"` or `"transformer"`, populates `DECISION_MEMORY_CAPACITIES`
    so native collectors and scalar environments capture opponent decision history.
    """
    if decision_memory == "none":
        obs_cfg = ObservationConfig(policy_input_version=11, max_cards_per_player=80, max_permanents_per_player=64)
    else:
        max_records, max_details, max_refs, max_members = DECISION_MEMORY_CAPACITIES
        obs_cfg = ObservationConfig(
            policy_input_version=11,
            max_cards_per_player=80,
            max_permanents_per_player=64,
            max_decision_records=max_records,
            max_decision_reference_details=max_details,
            max_decision_references=max_refs,
            max_decision_members=max_members,
        )
    obs_space = ObservationSpace(obs_cfg)
    model_cfg = ModelConfig(
        world="world",
        hidden_dim=hidden_dim,
        attention_layers=attention_layers,
        num_attention_heads=4,
        policy_film=True,
        value_kind="categorical_wdl",
        value_aggregation="masked_mean",
        decision_memory=decision_memory,
    )
    return ActorCritic(obs_space, model_cfg, bucketed_shapes=bucketed_shapes, live_tokens=live_tokens), obs_space


def pack_batch_for_ipc(batch: RolloutBatch) -> tuple[RolloutBatch, MemoryArchive | None]:
    recorded = batch.recorded
    if recorded.memory_capacities is None:
        return batch, None
    archive = archive_memory(recorded.memory, recorded.memory_capacities)
    stripped = replace(recorded, memory=(), memory_capacities=None)
    return replace(batch, recorded=stripped), archive


def unpack_batch_from_ipc(batch: RolloutBatch, archive: MemoryArchive | None) -> RolloutBatch:
    if archive is None:
        return batch
    restored = restore_memory(archive)
    return replace(batch, recorded=replace(batch.recorded, memory=restored, memory_capacities=archive.capacities))


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
    merged_memory = tuple(prefix for batch in batches for prefix in batch.recorded.memory)
    return RecordedObservations(
        rows=merged_rows,
        index=np.concatenate(offsets, axis=0),
        needs=merged_needs,
        memory=merged_memory,
        memory_capacities=first.memory_capacities,
    )


def merge_rollout_batches(batches: Sequence[RolloutBatch]) -> RolloutBatch:
    """Concatenate complete-game `RolloutBatch` payloads from parallel workers.

    Joins every worker's time-major rollout arrays (`actions`, `logprobs`, `rewards`,
    `dones`, `values`, and optional `outcome_probabilities`) along axis 0, shifts
    `RecordedObservations.index` by each worker's recorded row count, preserves
    decision-memory prefixes and capacities, and concatenates `PackedActions`.
    Raises `ValueError` when `batches` is empty.
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


def build_damped_rl_components(
    *,
    updates: int,
    learning_rate: float,
    lam_policy: float,
    lam_value: float,
    select_kind: str,
    min_advantage: float,
    clip: float,
    collection_kl: float,
    magnet_kl: float,
    value_coef: float,
    schedule_kind: str,
    schedule_decay: float,
) -> tuple[CreditFn, SelectFn, LossFn, LossScheduleFn | None]:
    credit: CreditFn = gae_wdl(gamma=1.0, lam_policy=lam_policy, lam_value=lam_value)
    select: SelectFn = (
        all_rows
        if select_kind == "all"
        else top_fraction(0.5 if select_kind == "top50" else 0.25, minimum=min_advantage)
    )
    default_loss = damped_policy_loss(
        clip=clip, collection_kl=collection_kl, magnet_kl=magnet_kl, value_coef=value_coef
    )
    if schedule_kind == "none":
        return credit, select, default_loss, None
    schedule: DampedSchedule | DecaySchedule = (
        DecaySchedule(
            total_updates=updates,
            lr_initial=learning_rate,
            tau_initial=magnet_kl,
            decay=schedule_decay,
        )
        if schedule_kind == "decay"
        else DampedSchedule(total_updates=updates, lr_max=learning_rate, tau_scale=magnet_kl)
    )

    def _loss_for_update(update_idx: int) -> tuple[float, float, LossFn]:
        lr, tau = schedule.rates(update_idx)
        return (
            lr,
            tau,
            damped_policy_loss(clip=clip, collection_kl=collection_kl, magnet_kl=tau, value_coef=value_coef),
        )

    return credit, select, default_loss, _loss_for_update

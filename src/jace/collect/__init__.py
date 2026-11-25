"""Rollout collection, v11 policy construction, and deck-source curricula for self-play.

Groups the engine-facing collection primitives (`agent`, `decks`, `local`, `pool`) that
build `ActorCritic` models, shard 17Lands and built-in deck catalogs across curriculum
stages, and gather complete-game rollouts in-process or across spawned CPU workers.
Never imports `jace.eval`, keeping rollout collection independent of evaluation code.
"""

from jace.collect.agent import (
    DECISION_MEMORY_CAPACITIES,
    DecisionMemoryKind,
    build_agent,
    build_damped_rl_components,
    pack_batch_for_ipc,
    unpack_batch_from_ipc,
)
from jace.collect.decks import (
    HumanDeckCatalog,
    build_curriculum_deck_source,
    build_curriculum_stage_pools,
    build_human_deck_source,
    curriculum_blend_spec,
    elemental_decks,
    load_human_deck_catalog,
    needs_curriculum_refresh,
    nested_card_pools,
    parse_curriculum_stages,
    row_to_deck,
    sample_human_decks,
    top_card_pool,
)
from jace.collect.local import LocalCollectorConfig, LocalShardedCollector, add_full_data_rl_args
from jace.collect.pool import MultiProcessCollector, WorkerPoolConfig

__all__ = (
    "DECISION_MEMORY_CAPACITIES",
    "DecisionMemoryKind",
    "HumanDeckCatalog",
    "LocalCollectorConfig",
    "LocalShardedCollector",
    "MultiProcessCollector",
    "WorkerPoolConfig",
    "add_full_data_rl_args",
    "build_agent",
    "build_curriculum_deck_source",
    "build_curriculum_stage_pools",
    "build_damped_rl_components",
    "build_human_deck_source",
    "curriculum_blend_spec",
    "elemental_decks",
    "load_human_deck_catalog",
    "needs_curriculum_refresh",
    "nested_card_pools",
    "pack_batch_for_ipc",
    "parse_curriculum_stages",
    "row_to_deck",
    "sample_human_decks",
    "top_card_pool",
    "unpack_batch_from_ipc",
)

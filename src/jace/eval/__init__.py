"""Evaluation, behavioral decision forensics, and checkpoint diffs for Jace v2.

Provides mirrored paired-deal scoring with optional native Academy game recording
(`deals`), 40-game Magic-principles behavioral auditing against `managym.Heuristic`
(`forensics`), and fixed held-out probe-state policy diffs across checkpoints (`diffs`).
Every evaluation runs directly over `managym` observations without importing any
training loop or upstream recipe.
"""

from jace.eval.deals import (
    Deal,
    MatchScore,
    PlayerFn,
    PolicyPlayer,
    build_deals,
    elemental_decks,
    heuristic_player,
    play_match,
    random_player,
    wilson_interval,
)
from jace.eval.diffs import DecisionShift, DiffSummary, ProbeState, capture_probes, diff_checkpoints
from jace.eval.forensics import ForensicsReport, run_forensics
from jace.eval.human_decks import (
    HumanDeckCatalog,
    build_curriculum_deck_source,
    build_curriculum_stage_pools,
    build_human_deals,
    build_human_deck_source,
    curriculum_blend_spec,
    load_human_deck_catalog,
    needs_curriculum_refresh,
    nested_card_pools,
    parse_curriculum_stages,
    row_to_deck,
    sample_human_decks,
    top_card_pool,
)
from jace.eval.parallel import (
    AdaptiveEvalConfig,
    AdaptiveEvalResult,
    PairedDealResult,
    PairedMatchScore,
    PlayerSpec,
    evaluate_adaptive,
    play_adaptive_parallel,
    play_deals_parallel,
    summarize_paired_deals,
)

__all__ = (
    "AdaptiveEvalConfig",
    "AdaptiveEvalResult",
    "Deal",
    "DecisionShift",
    "DiffSummary",
    "ForensicsReport",
    "HumanDeckCatalog",
    "MatchScore",
    "PairedDealResult",
    "PairedMatchScore",
    "PlayerFn",
    "PlayerSpec",
    "PolicyPlayer",
    "ProbeState",
    "build_curriculum_deck_source",
    "build_curriculum_stage_pools",
    "build_deals",
    "build_human_deals",
    "build_human_deck_source",
    "capture_probes",
    "curriculum_blend_spec",
    "diff_checkpoints",
    "elemental_decks",
    "evaluate_adaptive",
    "heuristic_player",
    "load_human_deck_catalog",
    "needs_curriculum_refresh",
    "nested_card_pools",
    "parse_curriculum_stages",
    "play_adaptive_parallel",
    "play_deals_parallel",
    "play_match",
    "random_player",
    "row_to_deck",
    "run_forensics",
    "sample_human_decks",
    "summarize_paired_deals",
    "top_card_pool",
    "wilson_interval",
)

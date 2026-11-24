"""Evaluation, behavioral decision forensics, and checkpoint diffs for Jace v2.

Provides mirrored paired-deal scoring with optional native Academy game recording
(`deals`), multi-worker parallel and adaptive sequential evaluation (`parallel`),
40-game Magic-principles behavioral auditing against `managym.Heuristic` (`forensics`),
and fixed held-out probe-state policy diffs across checkpoints (`diffs`). Every
evaluation runs directly over `managym` observations without importing any training
loop or upstream recipe.
"""

from jace.eval.deals import (
    Deal,
    MatchScore,
    PlayerFn,
    PolicyPlayer,
    build_deals,
    build_human_deals,
    elemental_decks,
    heuristic_player,
    play_match,
    random_player,
    wilson_interval,
)
from jace.eval.diffs import DecisionShift, DiffSummary, ProbeState, capture_probes, diff_checkpoints
from jace.eval.forensics import ForensicsReport, run_forensics
from jace.eval.parallel import (
    AdaptiveEvalConfig,
    AdaptiveEvalResult,
    PairedDealResult,
    PairedMatchScore,
    PlayerSpec,
    evaluate_adaptive,
    evaluate_agent_parallel,
    paired_score_metrics,
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
    "MatchScore",
    "PairedDealResult",
    "PairedMatchScore",
    "PlayerFn",
    "PlayerSpec",
    "PolicyPlayer",
    "ProbeState",
    "build_deals",
    "build_human_deals",
    "capture_probes",
    "diff_checkpoints",
    "elemental_decks",
    "evaluate_adaptive",
    "evaluate_agent_parallel",
    "heuristic_player",
    "paired_score_metrics",
    "play_adaptive_parallel",
    "play_deals_parallel",
    "play_match",
    "random_player",
    "run_forensics",
    "summarize_paired_deals",
    "wilson_interval",
)

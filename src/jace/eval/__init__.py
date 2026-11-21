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

__all__ = (
    "Deal",
    "DecisionShift",
    "DiffSummary",
    "ForensicsReport",
    "MatchScore",
    "PlayerFn",
    "PolicyPlayer",
    "ProbeState",
    "build_deals",
    "capture_probes",
    "diff_checkpoints",
    "elemental_decks",
    "heuristic_player",
    "play_match",
    "random_player",
    "run_forensics",
    "wilson_interval",
)

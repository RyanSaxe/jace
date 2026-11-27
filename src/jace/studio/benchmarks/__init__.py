"""Studio Benchmarks package combining Arena Elo leaderboards and tactical puzzles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jace.studio.benchmarks.arena import ArenaLeaderboardService, discover_benchmark_formats
from jace.studio.benchmarks.promotion import PuzzlePromotionResult
from jace.studio.benchmarks.puzzles import PuzzleService

if TYPE_CHECKING:
    from pathlib import Path

    from jace.studio.inference import InferenceEngine
    from jace.studio.replay import ReplayService

__all__ = (
    "ArenaLeaderboardService",
    "BenchmarkService",
    "PuzzlePromotionResult",
    "PuzzleService",
    "discover_benchmark_formats",
)


class BenchmarkService(ArenaLeaderboardService, PuzzleService):
    def __init__(
        self,
        runs_dir: Path,
        *,
        replays: ReplayService,
        inference: InferenceEngine,
    ) -> None:
        ArenaLeaderboardService.__init__(self, runs_dir)
        PuzzleService.__init__(self, runs_dir, replays=replays, inference=inference)

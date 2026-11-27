"""FastAPI server factory and CLI entrypoint for Jace Studio.

Wires Studio services (`InferenceEngine`, `StudioIndex`, `ReplayService`,
`ScryfallCache`, `WatchlistStore`, `BenchmarkService`) and registers all API
route modules from `jace.studio.routes`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from jace.studio.benchmarks import BenchmarkService
from jace.studio.index import StudioIndex
from jace.studio.inference import InferenceEngine
from jace.studio.replay import ReplayService
from jace.studio.routes import (
    WatchlistStore,
    register_asset_routes,
    register_belief_and_search_routes,
    register_benchmark_routes,
    register_fork_and_analysis_routes,
    register_index_routes,
    register_replay_routes,
    register_run_routes,
    register_watchlist_routes,
)
from jace.studio.scryfall import ScryfallCache

__all__ = (
    "StudioConfig",
    "create_app",
    "main",
)

_DEFAULT_WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


@dataclass(frozen=True, slots=True)
class StudioConfig:
    runs_dir: Path
    web_dist: Path | None = None
    allow_scryfall_network: bool = True


def create_app(config: StudioConfig) -> FastAPI:
    runs_dir = config.runs_dir.resolve()
    app = FastAPI(title="Jace Studio", version="0.3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    inference = InferenceEngine(runs_dir)
    index = StudioIndex(runs_dir, inference=inference)
    replays = ReplayService(runs_dir)
    scryfall = ScryfallCache(runs_dir / ".studio" / "scryfall", allow_network=config.allow_scryfall_network)
    watchlist = WatchlistStore(runs_dir / ".studio" / "watchlist.json")
    benchmarks = BenchmarkService(runs_dir, replays=replays, inference=inference)
    register_run_routes(app, runs_dir=runs_dir, index=index)
    register_index_routes(app, index=index, replays=replays)
    register_replay_routes(app, replays=replays, inference=inference)
    register_fork_and_analysis_routes(app, runs_dir=runs_dir, replays=replays, inference=inference, index=index)
    register_asset_routes(app, scryfall)
    register_watchlist_routes(app, watchlist)
    register_benchmark_routes(app, benchmarks=benchmarks, watchlist=watchlist)
    register_belief_and_search_routes(app, runs_dir=runs_dir, replays=replays, inference=inference, index=index)
    web_dist = config.web_dist if config.web_dist is not None else _DEFAULT_WEB_DIST
    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="studio-web")
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m jace.studio",
        description="Run Jace Studio local evaluation workbench for MTG RL runs and replays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Root directory containing Jace training run folders (must exist).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for the HTTP server.")
    parser.add_argument("--port", type=int, default=8474, help="Port for the HTTP server.")
    parser.add_argument(
        "--offline-cards",
        action="store_true",
        help="Disable Scryfall network fetches and serve local cache or token SVGs only.",
    )
    args = parser.parse_args(argv)
    if not args.runs_dir.is_dir():
        parser.error(f"--runs-dir does not exist: {args.runs_dir}")
    app = create_app(StudioConfig(runs_dir=args.runs_dir, allow_scryfall_network=not args.offline_cards))
    uvicorn.run(app, host=args.host, port=args.port)

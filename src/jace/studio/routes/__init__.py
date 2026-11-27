"""FastAPI route registration modules for Jace Studio."""

from jace.studio.routes.belief import register_belief_and_search_routes
from jace.studio.routes.benchmarks import WatchlistStore, register_benchmark_routes, register_watchlist_routes
from jace.studio.routes.replay import register_fork_and_analysis_routes, register_replay_routes
from jace.studio.routes.runs import register_asset_routes, register_index_routes, register_run_routes

__all__ = (
    "WatchlistStore",
    "register_asset_routes",
    "register_belief_and_search_routes",
    "register_benchmark_routes",
    "register_fork_and_analysis_routes",
    "register_index_routes",
    "register_replay_routes",
    "register_run_routes",
    "register_watchlist_routes",
)

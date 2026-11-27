"""FastAPI routes for run discovery, DuckDB game search, and Scryfall card assets."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

from jace.studio.index import GameQuery, StudioIndex
from jace.studio.runs import discover_runs, list_all_checkpoints, load_run_detail

if TYPE_CHECKING:
    from pathlib import Path

    from jace.studio.replay import ReplayService
    from jace.studio.scryfall import ScryfallCache

__all__ = (
    "GamesFilterParams",
    "register_asset_routes",
    "register_index_routes",
    "register_run_routes",
)


class GamesFilterParams(BaseModel):
    run_id: str | None = None
    set_code: str | None = None
    card: str | None = None
    result: str | None = None
    colors: str | None = None
    min_steps: int | None = Field(default=None, ge=0)
    min_turns: int | None = Field(default=None, ge=0)
    max_turns: int | None = Field(default=None, ge=0)
    min_surprise: float | None = Field(default=None, ge=0.0, le=1.0)
    min_swing: float | None = Field(default=None, ge=0.0, le=1.0)
    include_unplayable: bool = False
    search: str | None = None
    sort: str = "surprise_desc"
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


def register_run_routes(app: FastAPI, *, runs_dir: Path, index: StudioIndex) -> None:
    @app.get("/api/health", summary="Server health status")
    def health() -> dict[str, object]:
        return {"ok": True, "runs_dir": str(runs_dir)}

    @app.get("/api/runs", summary="Discover training runs, dynamic metrics, and playable game counts")
    def list_runs(*, include_archive: Annotated[bool, Query()] = False) -> dict[str, object]:
        summaries = discover_runs(runs_dir, include_archive=include_archive)
        counts = index.run_game_counts()
        enriched = [
            replace(
                item,
                playable_games=counts.get(item.run_id, (0, 0))[0],
                legacy_games=counts.get(item.run_id, (0, 0))[1],
            )
            for item in summaries
        ]
        experiments = sorted({item.experiment for item in enriched})
        eval_metric_keys = sorted({key for item in enriched for key in item.eval_metric_keys})
        train_metric_keys = sorted({key for item in enriched for key in item.train_metric_keys})
        return {
            "runs": [asdict(item) for item in enriched],
            "experiments": experiments,
            "eval_metric_keys": eval_metric_keys,
            "train_metric_keys": train_metric_keys,
        }

    @app.get("/api/checkpoints", summary="Enumerate all saved checkpoints across all discovered runs")
    def list_checkpoints(*, include_archive: Annotated[bool, Query()] = False) -> dict[str, object]:
        options = list_all_checkpoints(runs_dir, include_archive=include_archive)
        return {"checkpoints": [asdict(item) for item in options]}

    @app.get("/api/runs/{run_id:path}", summary="Load full run curves, config, and checkpoints")
    def get_run(run_id: str) -> dict[str, object]:
        try:
            detail = load_run_detail(runs_dir, run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        playable, legacy = index.run_game_counts().get(detail.summary.run_id, (0, 0))
        enriched_summary = replace(detail.summary, playable_games=playable, legacy_games=legacy)
        return asdict(replace(detail, summary=enriched_summary))


def register_index_routes(app: FastAPI, *, index: StudioIndex, replays: ReplayService) -> None:
    @app.post("/api/index/refresh", summary="Force-sync Parquet game index from disk")
    def refresh_index() -> dict[str, int]:
        replays.clear_cache()
        return index.sync(force=True)

    @app.get("/api/games/facets", summary="Dynamic filter facets discovered from playable games")
    def games_facets(run_id: Annotated[str | None, Query()] = None) -> dict[str, object]:
        return asdict(index.facets(run_id=run_id))

    @app.get("/api/games", summary="Search playable evaluation games via DuckDB")
    def search_games(filters: Annotated[GamesFilterParams, Depends()]) -> dict[str, object]:
        query = GameQuery(
            run_id=filters.run_id,
            set_code=filters.set_code,
            card=filters.card,
            result=filters.result,
            colors=filters.colors,
            min_steps=filters.min_steps,
            min_turns=filters.min_turns,
            max_turns=filters.max_turns,
            min_surprise=filters.min_surprise,
            min_swing=filters.min_swing,
            include_unplayable=filters.include_unplayable,
            search=filters.search,
            sort=filters.sort,
            limit=filters.limit,
            offset=filters.offset,
        )
        items, total = index.query_games(query)
        return {"total": total, "games": [asdict(item) for item in items]}

    @app.get("/api/cards-catalog", summary="List distinct indexed card names")
    def list_cards(run_id: Annotated[str | None, Query()] = None) -> dict[str, object]:
        return {"cards": index.available_cards(run_id=run_id)}


def register_asset_routes(app: FastAPI, scryfall: ScryfallCache) -> None:
    @app.get("/api/cards/{card_name:path}/image", summary="Cached Scryfall card or token JPEG, or SVG fallback")
    def card_image(card_name: str) -> Response:
        content, media_type = scryfall.get_image(card_name)
        return Response(content=content, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/cards/{card_name:path}", summary="Cached card or token metadata and oracle text")
    def card_metadata(card_name: str) -> dict[str, object]:
        return asdict(scryfall.get_metadata(card_name))

"""FastAPI routes and persistence store for Watchlist CRUD, Arena, and Puzzles."""

from __future__ import annotations

import json
import threading
import uuid
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from jace.studio.benchmarks import BenchmarkService

__all__ = (
    "PromotePuzzleRequest",
    "PuzzlesQueryParams",
    "WatchlistItemPayload",
    "WatchlistPatchPayload",
    "WatchlistStore",
    "register_benchmark_routes",
    "register_watchlist_routes",
)


class WatchlistItemPayload(BaseModel):
    run_id: str = Field(min_length=1, max_length=256)
    iteration: int = Field(ge=0)
    game_idx: int = Field(ge=0)
    step: int = Field(ge=0)
    turn: int = Field(ge=0)
    seed: int
    label: str = Field(min_length=1, max_length=160)
    note: str = Field(default="", max_length=500)
    tag: str = Field(default="", max_length=64)
    summary: str = Field(default="", max_length=300)
    key_cards: list[str] = Field(default_factory=list)


class WatchlistPatchPayload(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=160)
    note: str | None = Field(default=None, max_length=500)
    tag: str | None = Field(default=None, max_length=64)
    summary: str | None = Field(default=None, max_length=300)
    key_cards: list[str] | None = None


class PromotePuzzleRequest(BaseModel):
    format: str = Field(min_length=1, max_length=128)
    playouts: int = Field(default=64, ge=4, le=512)
    min_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    model_ref: str | None = None


class PuzzlesQueryParams(BaseModel):
    format: str | None = None
    model_a: str | None = None
    model_b: str | None = None
    phase: str | None = None
    source: str | None = None


class WatchlistStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def _read_locked(self) -> list[dict[str, object]]:
        if not self._path.is_file():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []

    def list_items(self) -> list[dict[str, object]]:
        with self._lock:
            return self._read_locked()

    def get_item(self, item_id: str) -> dict[str, object] | None:
        with self._lock:
            return next((item for item in self._read_locked() if item.get("id") == item_id), None)

    def add_item(self, payload: WatchlistItemPayload) -> dict[str, object]:
        with self._lock:
            items = self._read_locked()
            entry: dict[str, object] = {"id": uuid.uuid4().hex[:10], **payload.model_dump()}
            items.insert(0, entry)
            self._write_locked(items)
            return entry

    def update_item(self, item_id: str, patch: WatchlistPatchPayload) -> dict[str, object] | None:
        updates = patch.model_dump(exclude_none=True)
        with self._lock:
            items = self._read_locked()
            for idx, item in enumerate(items):
                if item.get("id") == item_id:
                    updated = {**item, **updates}
                    items[idx] = updated
                    self._write_locked(items)
                    return updated
            return None

    def delete_item(self, item_id: str) -> bool:
        with self._lock:
            items = self._read_locked()
            kept = [item for item in items if item.get("id") != item_id]
            if len(kept) == len(items):
                return False
            self._write_locked(kept)
            return True

    def _write_locked(self, items: list[dict[str, object]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(f".{self._path.name}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self._path)


def _require_watchlist_found(item_id: str, *, found: dict[str, object] | bool | None) -> dict[str, object]:
    if not found:
        raise HTTPException(status_code=404, detail=f"watchlist item not found: {item_id}")
    return {"item": found} if isinstance(found, dict) else {"deleted": item_id}


def register_watchlist_routes(app: FastAPI, watchlist: WatchlistStore) -> None:
    @app.get("/api/watchlist", summary="List pinned watchlist positions")
    def get_watchlist() -> dict[str, object]:
        return {"items": watchlist.list_items()}

    @app.post("/api/watchlist", summary="Pin a position to the watchlist")
    def add_watchlist(payload: WatchlistItemPayload) -> dict[str, object]:
        return {"item": watchlist.add_item(payload)}

    @app.patch("/api/watchlist/{item_id}", summary="Update a pinned watchlist position's metadata")
    def patch_watchlist(item_id: str, patch: WatchlistPatchPayload) -> dict[str, object]:
        return _require_watchlist_found(item_id, found=watchlist.update_item(item_id, patch))

    @app.delete("/api/watchlist/{item_id}", summary="Remove a watchlist position")
    def delete_watchlist(item_id: str) -> dict[str, object]:
        return _require_watchlist_found(item_id, found=watchlist.delete_item(item_id))


def _run_watchlist_promotion(
    benchmarks: BenchmarkService,
    *,
    item: dict[str, object],
    req: PromotePuzzleRequest,
) -> dict[str, object] | JSONResponse:
    try:
        result = benchmarks.promote_watchlist_item(
            item,
            format_id=req.format,
            playouts=req.playouts,
            min_margin=req.min_margin,
            model_ref=req.model_ref,
        )
    except (FileNotFoundError, IndexError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not result.admitted:
        return JSONResponse(
            status_code=422,
            content={
                "detail": result.detail or "Position did not pass puzzle promotion gate.",
                "margin": result.margin,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
            },
        )
    return {"admitted": True, "puzzle": result.puzzle}


def _call_benchmark[T](fn: Callable[[], T]) -> T:
    try:
        return fn()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def register_benchmark_routes(
    app: FastAPI,
    *,
    benchmarks: BenchmarkService,
    watchlist: WatchlistStore,
) -> None:
    @app.get("/api/arena/leaderboard", summary="Joint Bradley-Terry Elo leaderboard and early-run forecast")
    def arena_leaderboard(
        format_id: Annotated[str | None, Query(alias="format")] = None,
        anchor: Annotated[str | None, Query()] = None,
    ) -> dict[str, object]:
        return _call_benchmark(lambda: benchmarks.get_leaderboard(format_id=format_id, anchor=anchor))

    @app.get("/api/arena/matches", summary="Mirrored match ledger rows for one arena model")
    def arena_matches(
        format_id: Annotated[str, Query(alias="format")],
        model: Annotated[str, Query()],
    ) -> dict[str, object]:
        return _call_benchmark(lambda: benchmarks.get_matches(format_id=format_id, model=model))

    @app.get("/api/puzzles", summary="Verified tactical puzzles with cached checkpoint policy scoring")
    def list_puzzles(params: Annotated[PuzzlesQueryParams, Depends()]) -> dict[str, object]:
        return _call_benchmark(
            lambda: benchmarks.get_puzzles(
                format_id=params.format,
                model_a=params.model_a,
                model_b=params.model_b,
                phase=params.phase,
                source=params.source,
            )
        )

    @app.post(
        "/api/watchlist/{item_id}/promote-puzzle",
        summary="Verify a Watchlist board with paired playouts and promote to verified puzzles",
        response_model=None,
    )
    def promote_watchlist_puzzle(item_id: str, req: PromotePuzzleRequest) -> dict[str, object] | JSONResponse:
        item = watchlist.get_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"watchlist item not found: {item_id}")
        return _run_watchlist_promotion(benchmarks, item=item, req=req)

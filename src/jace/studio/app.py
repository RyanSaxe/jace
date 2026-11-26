"""FastAPI server and CLI entrypoint for Jace Studio.

Exposes live run telemetry, DuckDB/Parquet faceted game search, native `managym`
step replay, local Scryfall card art caching, watchlist persistence, and stateless
counterfactual forking (`POST /api/fork`). Serves the built React app from
`web/dist` when present.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jace.studio.index import GameQuery, StudioIndex
from jace.studio.inference import ForkEdit, InferenceEngine, apply_fork_edits
from jace.studio.replay import PovMode, ReplayService, serialize_observation
from jace.studio.runs import discover_runs, load_run_detail
from jace.studio.scryfall import ScryfallCache

__all__ = (
    "StudioConfig",
    "create_app",
    "main",
)

_DEFAULT_RUNS_DIR = Path.home() / "Projects" / "MTG" / "jace-runs"
_DEFAULT_WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
_TWO_PLAYERS = 2


@dataclass(frozen=True, slots=True)
class StudioConfig:
    runs_dir: Path
    web_dist: Path | None = None
    allow_scryfall_network: bool = True


class ForkEditPayload(BaseModel):
    op: str
    seat: int = Field(ge=0, le=1)
    value: int | None = None
    card: str | None = None


class ForkRequest(BaseModel):
    run_id: str
    iteration: int
    game_idx: int
    step: int
    pov: PovMode = "hero"
    ckpt_a: str | None = None
    ckpt_b: str | None = None
    edits: list[ForkEditPayload] = Field(default_factory=list)


class WatchlistItemPayload(BaseModel):
    run_id: str
    iteration: int
    game_idx: int
    step: int
    turn: int
    seed: int
    label: str
    note: str = ""


class GamesFilterParams(BaseModel):
    run_id: str | None = None
    set_code: str | None = None
    card: str | None = None
    result: str | None = None
    min_steps: int | None = None
    search: str | None = None
    sort: str = "steps_desc"
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class StepQueryParams(BaseModel):
    run_id: str
    iteration: int
    game_idx: int
    step: int = 0
    pov: PovMode = "hero"
    ckpt_a: str | None = None
    ckpt_b: str | None = None


class _WatchlistStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def list_items(self) -> list[dict[str, object]]:
        if not self._path.is_file():
            return []
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []

    def add_item(self, payload: WatchlistItemPayload) -> dict[str, object]:
        items = self.list_items()
        entry: dict[str, object] = {"id": uuid.uuid4().hex[:10], **payload.model_dump()}
        items.insert(0, entry)
        self._write(items)
        return entry

    def delete_item(self, item_id: str) -> bool:
        items = self.list_items()
        kept = [item for item in items if item.get("id") != item_id]
        if len(kept) == len(items):
            return False
        self._write(kept)
        return True

    def _write(self, items: list[dict[str, object]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")


def _extract_decks(players: tuple[tuple[str, dict[str, int]], ...]) -> tuple[dict[str, int], dict[str, int]]:
    if len(players) >= _TWO_PLAYERS:
        return players[0][1], players[1][1]
    return {}, {}


def _register_run_routes(app: FastAPI, runs_dir: Path) -> None:
    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "runs_dir": str(runs_dir)}

    @app.get("/api/runs")
    def list_runs(*, include_archive: Annotated[bool, Query()] = False) -> dict[str, object]:
        summaries = discover_runs(runs_dir, include_archive=include_archive)
        return {"runs": [asdict(item) for item in summaries]}

    @app.get("/api/runs/{run_id:path}")
    def get_run(run_id: str) -> dict[str, object]:
        try:
            detail = load_run_detail(runs_dir, run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return asdict(detail)


def _register_index_routes(app: FastAPI, index: StudioIndex) -> None:
    @app.post("/api/index/refresh")
    def refresh_index() -> dict[str, int]:
        return index.sync()

    @app.get("/api/games")
    def search_games(filters: Annotated[GamesFilterParams, Depends()]) -> dict[str, object]:
        query = GameQuery(
            run_id=filters.run_id,
            set_code=filters.set_code,
            card=filters.card,
            result=filters.result,
            min_steps=filters.min_steps,
            search=filters.search,
            sort=filters.sort,
            limit=filters.limit,
            offset=filters.offset,
        )
        items, total = index.query_games(query)
        return {"total": total, "games": [asdict(item) for item in items]}

    @app.get("/api/cards-catalog")
    def list_cards(run_id: Annotated[str | None, Query()] = None) -> dict[str, object]:
        return {"cards": index.available_cards(run_id=run_id)}


def _register_replay_routes(app: FastAPI, *, replays: ReplayService, inference: InferenceEngine) -> None:
    @app.get("/api/game-detail")
    def game_overview(params: Annotated[StepQueryParams, Depends()]) -> dict[str, object]:
        try:
            return replays.get_game_overview(run_id=params.run_id, iteration=params.iteration, game_idx=params.game_idx)
        except (FileNotFoundError, IndexError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/game-step")
    def game_step(params: Annotated[StepQueryParams, Depends()]) -> dict[str, object]:
        try:
            game, env, obs, actual_step, chosen, replay_warning = replays.replay_to_step(
                run_id=params.run_id, iteration=params.iteration, game_idx=params.game_idx, step=params.step
            )
        except (FileNotFoundError, IndexError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        board = serialize_observation(
            env,
            obs,
            decklists=_extract_decks(game.players),
            pov=params.pov,
            step_index=actual_step,
            total_steps=len(game.actions),
            chosen_action=chosen,
        )
        evaluation = inference.evaluate_pair(obs, run_id=params.run_id, ckpt_a=params.ckpt_a, ckpt_b=params.ckpt_b)
        return {"board": board, "evaluation": evaluation, "replay_warning": replay_warning}


def _register_fork_routes(app: FastAPI, *, replays: ReplayService, inference: InferenceEngine) -> None:
    @app.post("/api/fork")
    def fork_position(req: ForkRequest) -> dict[str, object]:
        try:
            game, env, base_obs, actual_step, _, replay_warning = replays.replay_to_step(
                run_id=req.run_id, iteration=req.iteration, game_idx=req.game_idx, step=req.step
            )
        except (FileNotFoundError, IndexError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        edits = [ForkEdit(op=e.op, seat=e.seat, value=e.value, card=e.card) for e in req.edits]
        cloned, forked_obs, warnings = apply_fork_edits(env, base_obs, edits=edits)
        if replay_warning:
            warnings.insert(0, replay_warning)
        board = serialize_observation(
            cloned,
            forked_obs,
            decklists=_extract_decks(game.players),
            pov=req.pov,
            step_index=actual_step,
            total_steps=len(game.actions),
            chosen_action=None,
        )
        base_eval = inference.evaluate_pair(base_obs, run_id=req.run_id, ckpt_a=req.ckpt_a, ckpt_b=req.ckpt_b)
        fork_eval = inference.evaluate_pair(forked_obs, run_id=req.run_id, ckpt_a=req.ckpt_a, ckpt_b=req.ckpt_b)
        return {"board": board, "base_evaluation": base_eval, "fork_evaluation": fork_eval, "warnings": warnings}


def _register_asset_routes(app: FastAPI, scryfall: ScryfallCache) -> None:
    @app.get("/api/cards/{card_name:path}/image")
    def card_image(card_name: str) -> Response:
        content, media_type = scryfall.get_image(card_name)
        return Response(content=content, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/cards/{card_name:path}")
    def card_metadata(card_name: str) -> dict[str, object]:
        return asdict(scryfall.get_metadata(card_name))


def _register_watchlist_routes(app: FastAPI, watchlist: _WatchlistStore) -> None:
    @app.get("/api/watchlist")
    def get_watchlist() -> dict[str, object]:
        return {"items": watchlist.list_items()}

    @app.post("/api/watchlist")
    def add_watchlist(payload: WatchlistItemPayload) -> dict[str, object]:
        return {"item": watchlist.add_item(payload)}

    @app.delete("/api/watchlist/{item_id}")
    def delete_watchlist(item_id: str) -> dict[str, object]:
        if not watchlist.delete_item(item_id):
            raise HTTPException(status_code=404, detail=f"watchlist item not found: {item_id}")
        return {"deleted": item_id}


def create_app(config: StudioConfig) -> FastAPI:
    runs_dir = config.runs_dir.resolve()
    app = FastAPI(title="Jace Studio", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    index = StudioIndex(runs_dir)
    replays = ReplayService(runs_dir)
    inference = InferenceEngine(runs_dir)
    scryfall = ScryfallCache(runs_dir / ".studio" / "scryfall", allow_network=config.allow_scryfall_network)
    watchlist = _WatchlistStore(runs_dir / ".studio" / "watchlist.json")
    _register_run_routes(app, runs_dir)
    _register_index_routes(app, index)
    _register_replay_routes(app, replays=replays, inference=inference)
    _register_fork_routes(app, replays=replays, inference=inference)
    _register_asset_routes(app, scryfall)
    _register_watchlist_routes(app, watchlist)
    web_dist = config.web_dist if config.web_dist is not None else _DEFAULT_WEB_DIST
    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="studio-web")
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Jace Studio local evaluation workbench.")
    parser.add_argument("--runs-dir", type=Path, default=_DEFAULT_RUNS_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8474)
    parser.add_argument("--offline-cards", action="store_true")
    args = parser.parse_args(argv)
    app = create_app(StudioConfig(runs_dir=args.runs_dir, allow_scryfall_network=not args.offline_cards))
    uvicorn.run(app, host=args.host, port=args.port)

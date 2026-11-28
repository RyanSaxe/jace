"""FastAPI server and CLI entrypoint for Jace Studio.

Exposes dynamic training run telemetry and global checkpoint discovery,
DuckDB/Parquet faceted game search, native `managym` step replay, Scryfall card
and token art caching, watchlist CRUD and puzzle promotion, Arena Elo
leaderboards and match ledgers, verified tactical puzzles, stateless
counterfactual forking + any-step action branching (`POST /api/fork`), N-model
policy matrices (`POST /api/matrix`), batched counterfactual playouts
(`POST /api/playout`), and cross-checkpoint policy comparisons (`POST /api/compare`).
"""

from __future__ import annotations

import argparse
import json
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jace.studio.analysis import compare_checkpoints, run_action_playouts
from jace.studio.belief_api import register_belief_and_search_routes
from jace.studio.benchmarks import BenchmarkService
from jace.studio.index import GameQuery, StudioIndex
from jace.studio.inference import ForkEdit, InferenceEngine, apply_fork_and_branch
from jace.studio.replay import PlayerSpec, PovMode, RecordedGame, ReplayService, serialize_observation
from jace.studio.runs import discover_runs, list_all_checkpoints, load_run_detail
from jace.studio.scryfall import ScryfallCache

if TYPE_CHECKING:
    from collections.abc import Callable

    import managym

__all__ = (
    "StudioConfig",
    "create_app",
    "main",
)

_DEFAULT_RUNS_DIR = Path.home() / "Projects" / "MTG" / "jace-runs"
_DEFAULT_WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
_TWO_PLAYERS = 2

type ForkOp = Literal[
    "set_life",
    "clear_hand",
    "add_hand_card",
    "remove_hand_card",
    "swap_hand_card",
    "add_battlefield",
]


@dataclass(frozen=True, slots=True)
class StudioConfig:
    runs_dir: Path
    web_dist: Path | None = None
    allow_scryfall_network: bool = True


class ForkEditPayload(BaseModel):
    op: ForkOp
    seat: int = Field(ge=0, le=1)
    value: int | None = Field(default=None, ge=1, le=99)
    card: str | None = Field(default=None, max_length=120)
    replacement_card: str | None = Field(default=None, max_length=120)
    after_branch: int = Field(default=0, ge=0, le=64)


class ForkRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=256)
    iteration: int = Field(ge=0)
    game_idx: int = Field(ge=0)
    step: int = Field(ge=0)
    pov: PovMode = "hero"
    ckpt_a: str | None = None
    ckpt_b: str | None = None
    edits: list[ForkEditPayload] = Field(default_factory=list, max_length=32)
    branch_actions: list[int] = Field(default_factory=list, max_length=64)


class MatrixRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=256)
    iteration: int = Field(ge=0)
    game_idx: int = Field(ge=0)
    step: int = Field(ge=0)
    pov: PovMode = "hero"
    edits: list[ForkEditPayload] = Field(default_factory=list, max_length=32)
    branch_actions: list[int] = Field(default_factory=list, max_length=64)
    models: list[str] = Field(default_factory=list)


class PlayoutRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=256)
    iteration: int = Field(ge=0)
    game_idx: int = Field(ge=0)
    step: int = Field(ge=0)
    pov: PovMode = "hero"
    edits: list[ForkEditPayload] = Field(default_factory=list, max_length=32)
    branch_actions: list[int] = Field(default_factory=list, max_length=64)
    model_ref: str = Field(default="heuristic", min_length=1, max_length=300)
    deals: int = Field(default=16, ge=1)
    batch_offset: int = Field(default=0, ge=0)
    action_indices: list[int] | None = None
    seed: int = 42


class CompareRequest(BaseModel):
    ckpt_a: str = Field(min_length=1, max_length=300)
    ckpt_b: str = Field(min_length=1, max_length=300)
    sample_count: int = Field(default=64, ge=1)
    run_id: str | None = None
    decision_kind: str | None = None
    seed: int = 42


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


class GameOverviewParams(BaseModel):
    run_id: str = Field(min_length=1)
    iteration: int = Field(ge=0)
    game_idx: int = Field(ge=0)


class StepQueryParams(BaseModel):
    run_id: str = Field(min_length=1)
    iteration: int = Field(ge=0)
    game_idx: int = Field(ge=0)
    step: int = Field(default=0, ge=0)
    pov: PovMode = "hero"
    ckpt_a: str | None = None
    ckpt_b: str | None = None


class _WatchlistStore:
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


def _extract_decks(players: tuple[PlayerSpec, ...]) -> tuple[dict[str, int], dict[str, int]]:
    if len(players) >= _TWO_PLAYERS:
        return players[0].decklist, players[1].decklist
    return {}, {}


@dataclass(frozen=True, slots=True)
class _ForkContext:
    game: RecordedGame
    env: managym.Env
    base_obs: managym.Observation
    forked_obs: managym.Observation
    actual_step: int
    recent_events: list[dict[str, object]]
    nearest_priority: int | None
    warnings: list[str]
    branch_events: list[dict[str, object]]
    branch_done: bool


def _register_run_routes(app: FastAPI, *, runs_dir: Path, index: StudioIndex) -> None:
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


def _register_index_routes(app: FastAPI, *, index: StudioIndex, replays: ReplayService) -> None:
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


def _register_replay_routes(app: FastAPI, *, replays: ReplayService, inference: InferenceEngine) -> None:
    def _score_timeline(obs_list: list[managym.Observation], game: RecordedGame) -> tuple[list[float], str]:
        ckpt = inference.find_checkpoint_for_iteration(game.run_id, game.iteration)
        return inference.evaluate_batch_win_probs(
            obs_list,
            run_id=game.run_id,
            checkpoint_name=ckpt,
            hero_seat=game.hero_seat,
        )

    @app.get("/api/game-detail", summary="Replay turn-level overview, model win curve, and turning point")
    def game_overview(params: Annotated[GameOverviewParams, Depends()]) -> dict[str, object]:
        try:
            return replays.get_game_overview(
                run_id=params.run_id,
                iteration=params.iteration,
                game_idx=params.game_idx,
                win_scorer=_score_timeline,
            )
        except (FileNotFoundError, IndexError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/game-step", summary="Replay a single step with checkpoint policy/WDL diff")
    def game_step(params: Annotated[StepQueryParams, Depends()]) -> dict[str, object]:
        try:
            game, env, obs, actual_step, chosen, replay_warning, recent_events, nearest_priority = (
                replays.replay_to_step(
                    run_id=params.run_id, iteration=params.iteration, game_idx=params.game_idx, step=params.step
                )
            )
        except (FileNotFoundError, IndexError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        board = serialize_observation(
            env,
            obs,
            decklists=_extract_decks(game.players),
            hero_seat=game.hero_seat,
            pov=params.pov,
            step_index=actual_step,
            total_steps=len(game.actions),
            chosen_action=chosen,
            recent_events=recent_events,
        )
        evaluation = inference.evaluate_pair(
            obs,
            run_id=params.run_id,
            hero_seat=game.hero_seat,
            ckpt_a=params.ckpt_a,
            ckpt_b=params.ckpt_b,
            hide_actions=bool(board["actions_hidden"]),
        )
        return {
            "board": board,
            "evaluation": evaluation,
            "replay_warning": replay_warning,
            "nearest_priority_step": nearest_priority,
        }


def _replay_and_fork(
    replays: ReplayService,
    *,
    run_id: str,
    iteration: int,
    game_idx: int,
    step: int,
    edits: list[ForkEditPayload],
    branch_actions: list[int],
) -> _ForkContext:
    try:
        game, env, base_obs, actual_step, _, replay_warning, recent_events, nearest_priority = replays.replay_to_step(
            run_id=run_id, iteration=iteration, game_idx=game_idx, step=step
        )
    except (FileNotFoundError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    fork_edits = [
        ForkEdit(
            op=e.op,
            seat=e.seat,
            value=e.value,
            card=e.card,
            replacement_card=e.replacement_card,
            after_branch=e.after_branch,
        )
        for e in edits
    ]
    cloned, forked_obs, warnings, branch_events, branch_done = apply_fork_and_branch(
        env, base_obs, edits=fork_edits, branch_actions=branch_actions, hero_seat=game.hero_seat
    )
    if replay_warning:
        warnings.insert(0, replay_warning)
    return _ForkContext(
        game=game,
        env=cloned,
        base_obs=base_obs,
        forked_obs=forked_obs,
        actual_step=actual_step,
        recent_events=recent_events,
        nearest_priority=nearest_priority,
        warnings=warnings,
        branch_events=[asdict(ev) for ev in branch_events],
        branch_done=branch_done,
    )


def _register_fork_and_analysis_routes(
    app: FastAPI,
    *,
    runs_dir: Path,
    replays: ReplayService,
    inference: InferenceEngine,
    index: StudioIndex,
) -> None:
    @app.post("/api/fork", summary="Evaluate stateless board/hand edits and any-step counterfactual action branches")
    def fork_position(req: ForkRequest) -> dict[str, object]:
        ctx = _replay_and_fork(
            replays,
            run_id=req.run_id,
            iteration=req.iteration,
            game_idx=req.game_idx,
            step=req.step,
            edits=req.edits,
            branch_actions=req.branch_actions,
        )
        board = serialize_observation(
            ctx.env,
            ctx.forked_obs,
            decklists=_extract_decks(ctx.game.players),
            hero_seat=ctx.game.hero_seat,
            pov=req.pov,
            step_index=ctx.actual_step + len(ctx.branch_events),
            total_steps=len(ctx.game.actions),
            chosen_action=None,
            recent_events=ctx.recent_events,
        )
        base_hide = req.pov == "hero" and int(ctx.base_obs.agent.player_index) != ctx.game.hero_seat
        base_eval = inference.evaluate_pair(
            ctx.base_obs,
            run_id=req.run_id,
            hero_seat=ctx.game.hero_seat,
            ckpt_a=req.ckpt_a,
            ckpt_b=req.ckpt_b,
            hide_actions=base_hide,
        )
        fork_eval = inference.evaluate_pair(
            ctx.forked_obs,
            run_id=req.run_id,
            hero_seat=ctx.game.hero_seat,
            ckpt_a=req.ckpt_a,
            ckpt_b=req.ckpt_b,
            hide_actions=bool(board["actions_hidden"]),
        )
        return {
            "board": board,
            "base_evaluation": base_eval,
            "fork_evaluation": fork_eval,
            "warnings": ctx.warnings,
            "branch_events": ctx.branch_events,
            "branch_done": ctx.branch_done,
            "nearest_priority_step": ctx.nearest_priority,
        }

    @app.post("/api/matrix", summary="Evaluate N checkpoints side-by-side on a single position")
    def policy_matrix(req: MatrixRequest) -> dict[str, object]:
        ctx = _replay_and_fork(
            replays,
            run_id=req.run_id,
            iteration=req.iteration,
            game_idx=req.game_idx,
            step=req.step,
            edits=req.edits,
            branch_actions=req.branch_actions,
        )
        hide_actions = req.pov == "hero" and int(ctx.forked_obs.agent.player_index) != ctx.game.hero_seat
        return inference.evaluate_matrix(
            ctx.forked_obs,
            run_id=req.run_id,
            hero_seat=ctx.game.hero_seat,
            models=req.models,
            hide_actions=hide_actions,
        )

    @app.post("/api/playout", summary="Run batched counterfactual playouts per legal action")
    def action_playout(req: PlayoutRequest) -> dict[str, object]:
        ctx = _replay_and_fork(
            replays,
            run_id=req.run_id,
            iteration=req.iteration,
            game_idx=req.game_idx,
            step=req.step,
            edits=req.edits,
            branch_actions=req.branch_actions,
        )
        hide_actions = req.pov == "hero" and int(ctx.forked_obs.agent.player_index) != ctx.game.hero_seat
        return run_action_playouts(
            inference,
            ctx.env,
            obs=ctx.forked_obs,
            run_id=req.run_id,
            hero_seat=ctx.game.hero_seat,
            model_ref=req.model_ref,
            deals=req.deals,
            batch_offset=req.batch_offset,
            action_indices=req.action_indices,
            seed=req.seed,
            hide_actions=hide_actions,
        )

    @app.post("/api/compare", summary="Compare two checkpoints across sampled playable decision states")
    def compare_models(req: CompareRequest) -> dict[str, object]:
        return compare_checkpoints(
            inference,
            runs_dir=runs_dir,
            index=index,
            ckpt_a=req.ckpt_a,
            ckpt_b=req.ckpt_b,
            sample_count=req.sample_count,
            run_id=req.run_id,
            decision_kind=req.decision_kind,
            seed=req.seed,
        )


def _register_asset_routes(app: FastAPI, scryfall: ScryfallCache) -> None:
    @app.get("/api/cards/{card_name:path}/image", summary="Cached Scryfall card or token JPEG, or SVG fallback")
    def card_image(card_name: str) -> Response:
        content, media_type = scryfall.get_image(card_name)
        return Response(content=content, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/cards/{card_name:path}", summary="Cached card or token metadata and oracle text")
    def card_metadata(card_name: str) -> dict[str, object]:
        return asdict(scryfall.get_metadata(card_name))


def _require_watchlist_found(item_id: str, *, found: dict[str, object] | bool | None) -> dict[str, object]:
    if not found:
        raise HTTPException(status_code=404, detail=f"watchlist item not found: {item_id}")
    return {"item": found} if isinstance(found, dict) else {"deleted": item_id}


def _register_watchlist_routes(app: FastAPI, watchlist: _WatchlistStore) -> None:
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


def _register_benchmark_routes(
    app: FastAPI,
    *,
    benchmarks: BenchmarkService,
    watchlist: _WatchlistStore,
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
    watchlist = _WatchlistStore(runs_dir / ".studio" / "watchlist.json")
    benchmarks = BenchmarkService(runs_dir, replays=replays, inference=inference)
    _register_run_routes(app, runs_dir=runs_dir, index=index)
    _register_index_routes(app, index=index, replays=replays)
    _register_replay_routes(app, replays=replays, inference=inference)
    _register_fork_and_analysis_routes(app, runs_dir=runs_dir, replays=replays, inference=inference, index=index)
    _register_asset_routes(app, scryfall)
    _register_watchlist_routes(app, watchlist)
    _register_benchmark_routes(app, benchmarks=benchmarks, watchlist=watchlist)
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
        default=_DEFAULT_RUNS_DIR,
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

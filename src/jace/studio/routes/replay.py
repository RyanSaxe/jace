"""FastAPI routes for replay inspection, counterfactual forking, matrix, playouts, and compare."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from jace.studio.analysis import compare_checkpoints, run_action_playouts
from jace.studio.forking import ForkEdit, apply_fork_and_branch
from jace.studio.replay import PlayerSpec, PovMode, RecordedGame, ReplayService, serialize_observation

if TYPE_CHECKING:
    from pathlib import Path

    import managym

    from jace.studio.index import StudioIndex
    from jace.studio.inference import InferenceEngine

__all__ = (
    "CompareRequest",
    "ForkEditPayload",
    "ForkOp",
    "ForkRequest",
    "GameOverviewParams",
    "MatrixRequest",
    "PlayoutRequest",
    "StepQueryParams",
    "register_fork_and_analysis_routes",
    "register_replay_routes",
)

_TWO_PLAYERS = 2

type ForkOp = Literal[
    "set_life",
    "clear_hand",
    "add_hand_card",
    "remove_hand_card",
    "swap_hand_card",
    "add_battlefield",
]


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


def _extract_decks(players: tuple[PlayerSpec, ...]) -> tuple[dict[str, int], dict[str, int]]:
    if len(players) >= _TWO_PLAYERS:
        return players[0].decklist, players[1].decklist
    return {}, {}


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


def register_replay_routes(app: FastAPI, *, replays: ReplayService, inference: InferenceEngine) -> None:
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


def register_fork_and_analysis_routes(
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
        try:
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
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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

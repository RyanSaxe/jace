"""FastAPI routes for opponent belief inference, override projection, and search.

Exposes `POST /api/belief/infer`, `POST /api/belief/project`, and
`POST /api/search/evaluate` on the Jace Studio FastAPI application, delegating
prior/model loading and search execution to `jace.studio.routes.belief_service`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jace.studio.routes.belief_service import (
    DEFAULT_V0_EQUITY,
    BeliefInferRequest,
    BeliefOverrideModel,
    BeliefProjectRequest,
    SearchEvaluateRequest,
    compute_turn0_v0,
    execute_search,
    format_search_actions,
    load_step_belief,
    project_or_422,
    serialize_belief_cards,
    to_belief_override,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI

    from jace.studio.index import StudioIndex
    from jace.studio.inference import InferenceEngine
    from jace.studio.replay import ReplayService

__all__ = (
    "BeliefInferRequest",
    "BeliefOverrideModel",
    "BeliefProjectRequest",
    "SearchEvaluateRequest",
    "register_belief_and_search_routes",
)


def register_belief_and_search_routes(
    app: FastAPI, *, runs_dir: Path, replays: ReplayService, inference: InferenceEngine, index: StudioIndex
) -> None:
    @app.post("/api/belief/infer", summary="Infer full-format opponent belief and Turn-0 V0 equity")
    def belief_infer(req: BeliefInferRequest) -> dict[str, object]:
        game, _, obs, _, belief, revealed, has_net = load_step_belief(
            req, runs_dir=runs_dir, replays=replays, index=index
        )
        acting_seat = int(obs.agent.player_index)
        v0_equity = float(belief.matchup_equity) if belief.matchup_equity is not None else DEFAULT_V0_EQUITY
        v0_info = compute_turn0_v0(obs, inference=inference, game=game, ckpt_a=None, v0_equity=v0_equity)
        cards, pinned_total, unedited_count, unedited_mass = serialize_belief_cards(
            belief, revealed=revealed, override=None
        )
        return {
            "card_names": list(belief.card_names),
            "hand_probs": [round(float(x), 4) for x in belief.hand_probs.detach().cpu().tolist()],
            "deck_copies": [round(float(x), 4) for x in belief.deck_copies.detach().cpu().tolist()],
            "revealed_counts": [round(float(x)) for x in revealed.detach().cpu().tolist()],
            "cards": cards,
            "opp_hand_size": int(belief.opp_hand_size),
            "opp_library_size": int(belief.opp_library_size),
            "pinned_total": pinned_total,
            "unedited_count": unedited_count,
            "unedited_mass": unedited_mass,
            "prior_backend": str(belief.prior_backend),
            "has_belief_net": has_net,
            "acting_seat": acting_seat,
            "perspective": "hero" if acting_seat == game.hero_seat else "opponent",
            "matchup_equity": round(v0_equity, 4),
            "v0_equity": round(v0_equity, 4),
            "v0": v0_info,
        }

    @app.post("/api/belief/project", summary="Project user counterfactual belief overrides onto legal distribution")
    def belief_project(req: BeliefProjectRequest) -> dict[str, object]:
        game, _, obs, prior, belief, revealed, has_net = load_step_belief(
            req, runs_dir=runs_dir, replays=replays, index=index
        )
        acting_seat = int(obs.agent.player_index)
        override = to_belief_override(req.override)
        projected = (
            project_or_422(belief, override, revealed=revealed, copy_limits=prior.copy_limits)
            if override is not None
            else belief
        )
        cards, pinned_total, unedited_count, unedited_mass = serialize_belief_cards(
            projected, revealed=revealed, override=override
        )
        return {
            "cards": cards,
            "opp_hand_size": int(projected.opp_hand_size),
            "opp_library_size": int(projected.opp_library_size),
            "pinned_total": pinned_total,
            "unedited_count": unedited_count,
            "unedited_mass": unedited_mass,
            "has_belief_net": has_net,
            "acting_seat": acting_seat,
            "perspective": "hero" if acting_seat == game.hero_seat else "opponent",
        }

    @app.post("/api/search/evaluate", summary="Run belief-conditioned multi-world local search on a step")
    def search_evaluate(req: SearchEvaluateRequest) -> dict[str, object]:
        game, env, obs, prior, belief, revealed, _ = load_step_belief(
            req, runs_dir=runs_dir, replays=replays, index=index
        )
        acting_seat = int(obs.agent.player_index)
        override = to_belief_override(req.override)
        if override is not None:
            belief = project_or_422(belief, override, revealed=revealed, copy_limits=prior.copy_limits)
        result = execute_search(
            req,
            inference=inference,
            env=env,
            obs=obs,
            belief=belief,
            revealed=revealed,
            copy_limits=prior.copy_limits,
            override=override,
        )
        v0_equity = float(belief.matchup_equity) if belief.matchup_equity is not None else DEFAULT_V0_EQUITY
        v0_info = compute_turn0_v0(obs, inference=inference, game=game, ckpt_a=req.ckpt_a, v0_equity=v0_equity)
        return {
            "chosen_action": int(result.chosen_action),
            "actions": format_search_actions(obs, result, hero_seat=acting_seat),
            "worlds_sampled": int(result.worlds_sampled),
            "rollout_horizon": int(result.rollout_horizon or req.horizon_plies),
            "elapsed_ms": round(float(result.elapsed_ms), 1),
            "opp_hand_size": int(belief.opp_hand_size),
            "acting_seat": acting_seat,
            "perspective": "hero" if acting_seat == game.hero_seat else "opponent",
            "warnings": list(result.warnings),
            "v0_equity": round(v0_equity, 4),
            "v0": v0_info,
        }

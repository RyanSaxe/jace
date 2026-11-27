"""Arena Elo leaderboard, match ledger orientation, and early-run forecast service.

Reads mirrored match records from `<runs-dir>/arena/ledger.jsonl` via
`jace.eval.arena`, fits selectable-anchor Bradley-Terry ratings, orients match
rows per model, and adapts early-run `u200` peak forecasts from
`jace.eval.report.discover_early_forecast_points`.
"""

from __future__ import annotations

import re
from pathlib import Path

from jace.eval.arena import (
    ArenaMatchRecord,
    BradleyTerryRating,
    ledger_path_for,
    load_match_records,
    player_id_for,
    solve_bradley_terry,
)
from jace.eval.report import discover_early_forecast_points
from jace.studio.runs import list_all_checkpoints

__all__ = (
    "ArenaLeaderboardService",
    "compute_early_run_forecast",
    "discover_benchmark_formats",
)

_DEFAULT_ANCHOR_ELO = 1500.0
_STUDIO_BT_BOOTSTRAP = 200
_FORECAST_MIN_POINTS = 2
_UPDATE_TOKEN_RE = re.compile(r"(?:^|[-_:\s])(?:u|iter-?)0*(\d+)(?:$|\b|[-_:\s])", re.IGNORECASE)


def _list_puzzle_formats(runs_dir: Path) -> list[str]:
    puzzles_dir = runs_dir / "puzzles"
    if not puzzles_dir.is_dir():
        return []
    return sorted(p.stem for p in puzzles_dir.iterdir() if p.is_file() and p.suffix == ".jsonl")


def discover_benchmark_formats(runs_dir: Path) -> tuple[list[str], str]:
    records = load_match_records(ledger_path_for(runs_dir))
    arena_formats = [rec.format_id for rec in records]
    puzzle_formats = _list_puzzle_formats(runs_dir)
    all_formats = sorted({*arena_formats, *puzzle_formats})
    if arena_formats:
        return all_formats, arena_formats[-1]
    if puzzle_formats:
        return all_formats, puzzle_formats[0]
    return [], ""


def _infer_updates(model_id: str) -> int | None:
    if model_id.lower() == "heuristic":
        return 0
    match = _UPDATE_TOKEN_RE.search(model_id)
    return int(match.group(1)) if match is not None else None


def _build_known_checkpoint_map(runs_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for opt in list_all_checkpoints(runs_dir):
        mapping[opt.ref] = opt.ref
        mapping[f"{opt.run_name}:{opt.checkpoint}"] = opt.ref
        mapping[f"{opt.run_name}-{opt.checkpoint}".lower()] = opt.ref
    return mapping


def _infer_checkpoint_ref(model_id: str, *, known_refs: dict[str, str]) -> str | None:
    if model_id.lower() == "heuristic":
        return "heuristic"
    if "::" in model_id:
        return model_id
    return known_refs.get(model_id) or known_refs.get(model_id.lower())


def _resolve_effective_anchor(ratings: list[BradleyTerryRating], requested: str | None) -> str:
    players = [r.player_id for r in ratings]
    if requested:
        matched = next((p for p in players if p.lower() == requested.lower()), None)
        if matched is not None:
            return matched
    heuristic = next((p for p in players if p.lower() == "heuristic"), None)
    return heuristic if heuristic is not None else players[0]


def compute_early_run_forecast(runs_dir: Path) -> dict[str, object] | None:
    summary = discover_early_forecast_points(runs_dir)
    if len(summary.points) < _FORECAST_MIN_POINTS:
        return None
    points = [
        {
            "run_id": Path(pt.run_id).name,
            "early_wr": round(pt.early_win_rate, 4),
            "peak_wr": round(pt.peak_win_rate, 4),
        }
        for pt in summary.points
    ]
    best_pt = max(summary.points, key=lambda pt: pt.peak_win_rate)
    power_pred = round(best_pt.forecast.projected_win_rate, 4) if best_pt.forecast is not None else None
    return {
        "n": len(points),
        "pearson_r": round(summary.pearson_r, 4) if summary.pearson_r is not None else 0.0,
        "spearman_rho": round(summary.spearman_rho, 4) if summary.spearman_rho is not None else 0.0,
        "points": points,
        "power_law_pred": power_pred,
    }


def _rating_to_row(rating: BradleyTerryRating, *, known_refs: dict[str, str]) -> dict[str, object]:
    ci_half = round(0.5 * (rating.ci95_high - rating.ci95_low), 1)
    label = "Heuristic" if rating.player_id.lower() == "heuristic" else rating.player_id
    return {
        "model_id": rating.player_id,
        "label": label,
        "elo": round(rating.elo, 1),
        "ci_half": ci_half,
        "matches": rating.matches,
        "games": rating.games,
        "vs_anchor_wr": round(rating.vs_anchor_wr, 4) if rating.vs_anchor_wr is not None else None,
        "updates": _infer_updates(rating.player_id),
        "search_ms_per_dec": (
            round(rating.mean_ms_per_decision, 2) if rating.mean_ms_per_decision is not None else None
        ),
        "checkpoint_ref": _infer_checkpoint_ref(rating.player_id, known_refs=known_refs),
    }


def _orient_match_row(
    rec: ArenaMatchRecord,
    *,
    model: str,
    known_refs: dict[str, str],
) -> dict[str, object]:
    pid_a = player_id_for(rec.model_a, rec.search_config_a)
    pid_b = player_id_for(rec.model_b, rec.search_config_b)
    is_a = pid_a == model
    opp_id = pid_b if is_a else pid_a
    total = rec.wins_a + rec.losses_a + rec.draws
    games = total if total > 0 else max(0, 2 * rec.deals)
    ref_self = _infer_checkpoint_ref(model, known_refs=known_refs)
    ref_opp = _infer_checkpoint_ref(opp_id, known_refs=known_refs)
    compare_target = (
        {"ckptA": ref_self, "ckptB": ref_opp}
        if ref_self and ref_opp and not (ref_self == "heuristic" and ref_opp == "heuristic")
        else None
    )
    opp_label = "Heuristic" if opp_id.lower() == "heuristic" else opp_id
    return {
        "match_id": rec.match_id,
        "opponent_id": opp_id,
        "opponent_label": opp_label,
        "games": games,
        "winrate": round(rec.win_rate_a if is_a else 1.0 - rec.win_rate_a, 4),
        "ci_low": round(rec.ci95_low if is_a else 1.0 - rec.ci95_high, 4),
        "ci_high": round(rec.ci95_high if is_a else 1.0 - rec.ci95_low, 4),
        "sweeps_w": rec.sweeps_a if is_a else rec.sweeps_b,
        "sweeps_l": rec.sweeps_b if is_a else rec.sweeps_a,
        "los": round(rec.los_a if is_a else 1.0 - rec.los_a, 4),
        "compare_target": compare_target,
    }


class ArenaLeaderboardService:
    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir.resolve()

    def get_leaderboard(
        self,
        *,
        format_id: str | None = None,
        anchor: str | None = None,
    ) -> dict[str, object]:
        formats, default_format = discover_benchmark_formats(self._runs_dir)
        active_format = format_id.strip() if format_id and format_id.strip() else default_format
        records = load_match_records(ledger_path_for(self._runs_dir), format_id=active_format) if active_format else []
        if not records:
            return {
                "format": active_format,
                "formats": formats,
                "anchor_model": anchor or "heuristic",
                "anchor_elo": _DEFAULT_ANCHOR_ELO,
                "models": [],
                "forecast": compute_early_run_forecast(self._runs_dir),
            }
        players = sorted(
            {
                pid
                for rec in records
                for pid in (
                    player_id_for(rec.model_a, rec.search_config_a),
                    player_id_for(rec.model_b, rec.search_config_b),
                )
            }
        )
        requested_anchor: str | None = None
        if anchor and anchor.strip():
            needle = anchor.strip().lower()
            matched = next((p for p in players if p.lower() == needle), None)
            if matched is None:
                msg = f"unknown anchor player {anchor!r} for format {active_format!r}"
                raise ValueError(msg)
            requested_anchor = matched
        ratings = solve_bradley_terry(
            records,
            format_id=active_format,
            anchor_player=requested_anchor,
            anchor_elo=_DEFAULT_ANCHOR_ELO,
            bootstrap_samples=_STUDIO_BT_BOOTSTRAP,
        )
        anchor_model = _resolve_effective_anchor(ratings, requested_anchor)
        known_refs = _build_known_checkpoint_map(self._runs_dir)
        models = [_rating_to_row(rating, known_refs=known_refs) for rating in ratings]
        return {
            "format": active_format,
            "formats": formats,
            "anchor_model": anchor_model,
            "anchor_elo": _DEFAULT_ANCHOR_ELO,
            "models": models,
            "forecast": compute_early_run_forecast(self._runs_dir),
        }

    def get_matches(self, *, format_id: str, model: str) -> dict[str, object]:
        records = load_match_records(ledger_path_for(self._runs_dir), format_id=format_id)
        known_refs = _build_known_checkpoint_map(self._runs_dir)
        matches = [
            _orient_match_row(rec, model=model, known_refs=known_refs)
            for rec in records
            if model
            in {
                player_id_for(rec.model_a, rec.search_config_a),
                player_id_for(rec.model_b, rec.search_config_b),
            }
        ]
        return {"matches": matches}

"""Arena Elo leaderboard, tactical puzzles, and watchlist puzzle promotion for Jace Studio.

Reads mirrored match records from `<runs-dir>/arena/ledger.jsonl` via
`jace.eval.arena`, fits selectable-anchor Bradley-Terry ratings, computes
early-run `u200` peak forecasts from discovered training runs, serves verified
tactical puzzles from `<runs-dir>/puzzles/{format}.jsonl` with cached checkpoint
policy scoring, and promotes bookmarked Watchlist positions through paired CRN
playout verification.
"""

from __future__ import annotations

import json
import math
import random
import re
import threading
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import managym
from managym.enums import ActionSpaceEnum, ZoneEnum

from jace.eval.arena import (
    ArenaMatchRecord,
    BradleyTerryRating,
    ledger_path_for,
    load_match_records,
    player_id_for,
    solve_bradley_terry,
)
from jace.eval.puzzles import PuzzleRecord, append_puzzle, puzzles_path, summarize_paired_outcomes
from jace.studio.replay import engine_step_errors, seat_aligned_obs, serialize_legal_actions
from jace.studio.runs import discover_runs, list_all_checkpoints, load_run_detail

if TYPE_CHECKING:
    from pathlib import Path

    from jace.studio.inference import InferenceEngine
    from jace.studio.replay import ReplayService

__all__ = (
    "BenchmarkService",
    "PuzzlePromotionResult",
    "discover_benchmark_formats",
)

type PuzzleSourceKind = Literal["auto_mined", "search_disagreement", "watchlist"]

_DEFAULT_ANCHOR_ELO = 1500.0
_STUDIO_BT_BOOTSTRAP = 200
_FORECAST_EARLY_TARGET = 200
_FORECAST_EARLY_MAX = 400
_FORECAST_MIN_POINTS = 2
_FORECAST_DEFAULT_TARGET = 2000
_MAX_PROMOTE_ACTIONS = 4
_MIN_LEGAL_ACTIONS = 2
_PROMOTE_MAX_PLIES = 240
_SAFE_FORMAT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UPDATE_TOKEN_RE = re.compile(r"(?:^|[-_:\s])(?:u|iter-?)0*(\d+)(?:$|\b|[-_:\s])", re.IGNORECASE)
_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
_WIN_RATE_KEYS = (
    "ensemble_held_out_det_win_rate",
    "held_out_det_win_rate",
    "live_held_out_det_win_rate",
)


@dataclass(frozen=True, slots=True)
class _PuzzleScoreCacheEntry:
    probs: tuple[float, ...]
    solved_best_idx: int | None


@dataclass(frozen=True, slots=True)
class PuzzlePromotionResult:
    admitted: bool
    margin: float
    ci_low: float
    ci_high: float
    detail: str | None = None
    puzzle: dict[str, object] | None = None


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return default


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    return default


def _parse_jsonl_line(raw_line: str) -> dict[str, object] | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [row for line in lines if (row := _parse_jsonl_line(line)) is not None]


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


def _humanize_model_label(model_id: str) -> str:
    return "Heuristic" if model_id.lower() == "heuristic" else model_id


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


def _extract_point_win_rate(point: dict[str, object]) -> float | None:
    for key in _WIN_RATE_KEYS:
        val = point.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
            return float(val)
    return None


def _rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        end = idx
        while end + 1 < len(order) and values[order[end + 1]] == values[order[idx]]:
            end += 1
        avg_rank = 0.5 * (idx + end) + 1.0
        for pos in range(idx, end + 1):
            ranks[order[pos]] = avg_rank
        idx = end + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < _FORECAST_MIN_POINTS:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(num / (dx * dy), 4) if dx > 0.0 and dy > 0.0 else 0.0


def _fit_power_law_pred(points: list[tuple[int, float]], *, target_update: int) -> float | None:
    usable = [(u, wr) for u, wr in points if 0 < u <= _FORECAST_EARLY_MAX and 0.0 < wr < 1.0]
    if len(usable) < _FORECAST_MIN_POINTS:
        return None
    lx = [math.log(float(u)) for u, _ in usable]
    ly = [math.log(1.0 - wr) for _, wr in usable]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    denom = sum((x - mx) ** 2 for x in lx)
    if denom <= 0.0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly, strict=True)) / denom
    intercept = my - slope * mx
    loss_pred = math.exp(intercept + slope * math.log(float(max(target_update, 1))))
    return round(max(0.0, min(1.0, 1.0 - loss_pred)), 4)


def _compute_early_run_forecast(runs_dir: Path) -> dict[str, object] | None:
    points: list[dict[str, object]] = []
    best_series: tuple[float, int, list[tuple[int, float]]] | None = None
    for summary in discover_runs(runs_dir):
        try:
            detail = load_run_detail(runs_dir, summary.run_id)
        except (FileNotFoundError, ValueError):
            continue
        series = [
            (_as_int(pt.get("update")), wr)
            for pt in detail.eval_points
            if (wr := _extract_point_win_rate(pt)) is not None and _as_int(pt.get("update")) > 0
        ]
        early_candidates = [(u, wr) for u, wr in series if u <= _FORECAST_EARLY_MAX]
        if not early_candidates or not series:
            continue
        early_wr = min(early_candidates, key=lambda item: abs(item[0] - _FORECAST_EARLY_TARGET))[1]
        peak_wr = max(wr for _, wr in series)
        points.append({"run_id": summary.name, "early_wr": round(early_wr, 4), "peak_wr": round(peak_wr, 4)})
        if best_series is None or peak_wr > best_series[0]:
            target_u = max(_FORECAST_DEFAULT_TARGET, summary.total_updates, summary.latest_update)
            best_series = (peak_wr, target_u, series)
    if len(points) < _FORECAST_MIN_POINTS:
        return None
    xs = [_as_float(p["early_wr"]) for p in points]
    ys = [_as_float(p["peak_wr"]) for p in points]
    power_pred = _fit_power_law_pred(best_series[2], target_update=best_series[1]) if best_series else None
    return {
        "n": len(points),
        "pearson_r": _pearson(xs, ys),
        "spearman_rho": _pearson(_rank_values(xs), _rank_values(ys)),
        "points": points,
        "power_law_pred": power_pred,
    }


def _canonical_puzzle_source(raw_source: str) -> PuzzleSourceKind:
    lowered = raw_source.strip().lower()
    if lowered == "search_disagreement":
        return "search_disagreement"
    if lowered in {"watchlist", "watchlist_promotion"}:
        return "watchlist"
    return "auto_mined"


def _humanize_phase(raw_phase: str) -> str:
    cleaned = raw_phase.strip()
    if not cleaned:
        return "Priority"
    return cleaned.replace("_", " ").title() if ("_" in cleaned or cleaned.isupper()) else cleaned


def _summarize_player_counts(obs: managym.Observation, *, hero_seat: int) -> tuple[str, str, list[str]]:
    hero_pack, opp_pack = seat_aligned_obs(obs, hero_seat=hero_seat)
    hero_p, hero_perms, _ = hero_pack
    opp_p, opp_perms, _ = opp_pack
    hero_lands = sum(1 for p in hero_perms if p.characteristics.card_types.is_land)
    opp_lands = sum(1 for p in opp_perms if p.characteristics.card_types.is_land)
    hero_hand = int(hero_p.zone_counts[int(ZoneEnum.HAND)])
    opp_hand = int(opp_p.zone_counts[int(ZoneEnum.HAND)])
    hero_summary = f"Hero: {int(hero_p.life)} life · {hero_lands} lands · {hero_hand} cards"
    opp_summary = f"Opp: {int(opp_p.life)} life · {opp_lands} lands · {opp_hand} cards"
    key_cards: list[str] = []
    for perm in (*hero_perms, *opp_perms):
        name = str(perm.characteristics.name)
        if name not in _BASIC_LANDS and name not in key_cards:
            key_cards.append(name)
        if len(key_cards) >= _MIN_LEGAL_ACTIONS:
            break
    return hero_summary, opp_summary, key_cards


def _validate_format_id(format_id: str) -> str:
    cleaned = format_id.strip()
    if not _SAFE_FORMAT_RE.match(cleaned):
        msg = f"invalid format name: {format_id!r}"
        raise ValueError(msg)
    return cleaned


def _determinize_or_seed(
    cloned: managym.Env,
    *,
    deal_seed: int,
    acting_seat: int,
    step_errors: tuple[type[BaseException], ...],
) -> None:
    try:
        cloned.determinize(deal_seed, acting_seat)
    except step_errors:
        cloned.set_seed(deal_seed)


def _rollout_heuristic(
    cloned: managym.Env,
    obs: managym.Observation,
    *,
    heuristic: managym.Heuristic,
    rng: random.Random,
    max_plies: int,
    step_errors: tuple[type[BaseException], ...],
) -> None:
    for _ in range(max_plies):
        if cloned.is_game_over():
            break
        try:
            chosen = int(heuristic.select_action(obs))
        except step_errors:
            chosen = rng.randrange(max(1, len(obs.action_space.actions)))
        try:
            obs, _, done, truncated, _ = cloned.step(chosen)
        except step_errors:
            break
        if done or truncated:
            break


def _play_single_crn_branch(
    env: managym.Env,
    *,
    action_idx: int,
    deal_seed: int,
    acting_seat: int,
    hero_seat: int,
    heuristic: managym.Heuristic,
    max_plies: int,
) -> float:
    step_errors = engine_step_errors()
    cloned = env.clone_env()
    _determinize_or_seed(cloned, deal_seed=deal_seed, acting_seat=acting_seat, step_errors=step_errors)
    try:
        obs, _, done, truncated, _ = cloned.step(action_idx)
    except step_errors:
        return 0.5
    if not done and not truncated:
        _rollout_heuristic(
            cloned,
            obs,
            heuristic=heuristic,
            rng=random.Random(deal_seed ^ 0x5A5A),
            max_plies=max_plies,
            step_errors=step_errors,
        )
    winner = cloned.winner_index() if cloned.is_game_over() else None
    if winner is None:
        return 0.5
    return 1.0 if int(winner) == hero_seat else 0.0


def _run_paired_crn_outcomes(
    env: managym.Env,
    *,
    action_indices: list[int],
    playouts: int,
    acting_seat: int,
    hero_seat: int,
    seed: int,
    max_plies: int = _PROMOTE_MAX_PLIES,
) -> dict[int, list[float]]:
    heuristic = managym.Heuristic()
    outcomes: dict[int, list[float]] = {idx: [] for idx in action_indices}
    for deal_idx in range(playouts):
        deal_seed = seed * 1_000_003 + deal_idx * 9973
        for action_idx in action_indices:
            outcomes[action_idx].append(
                _play_single_crn_branch(
                    env,
                    action_idx=action_idx,
                    deal_seed=deal_seed,
                    acting_seat=acting_seat,
                    hero_seat=hero_seat,
                    heuristic=heuristic,
                    max_plies=max_plies,
                )
            )
    return outcomes


def _pick_top_action_indices(probs: list[float], num_actions: int) -> list[int]:
    if not probs or len(probs) < num_actions:
        return list(range(min(num_actions, _MAX_PROMOTE_ACTIONS)))
    return sorted(range(num_actions), key=lambda idx: probs[idx], reverse=True)[:_MAX_PROMOTE_ACTIONS]


class BenchmarkService:
    def __init__(
        self,
        runs_dir: Path,
        *,
        replays: ReplayService,
        inference: InferenceEngine,
    ) -> None:
        self._runs_dir = runs_dir.resolve()
        self._replays = replays
        self._inference = inference
        self._score_cache: dict[tuple[str, str], _PuzzleScoreCacheEntry] = {}
        self._cache_lock = threading.Lock()

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
                "forecast": _compute_early_run_forecast(self._runs_dir),
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
        models = [self._rating_to_row(rating, known_refs=known_refs) for rating in ratings]
        return {
            "format": active_format,
            "formats": formats,
            "anchor_model": anchor_model,
            "anchor_elo": _DEFAULT_ANCHOR_ELO,
            "models": models,
            "forecast": _compute_early_run_forecast(self._runs_dir),
        }

    @staticmethod
    def _rating_to_row(rating: BradleyTerryRating, *, known_refs: dict[str, str]) -> dict[str, object]:
        ci_half = round(0.5 * (rating.ci95_high - rating.ci95_low), 1)
        return {
            "model_id": rating.player_id,
            "label": _humanize_model_label(rating.player_id),
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

    def get_matches(self, *, format_id: str, model: str) -> dict[str, object]:
        records = load_match_records(ledger_path_for(self._runs_dir), format_id=format_id)
        known_refs = _build_known_checkpoint_map(self._runs_dir)
        matches = [
            self._orient_match_row(rec, model=model, known_refs=known_refs)
            for rec in records
            if model
            in {
                player_id_for(rec.model_a, rec.search_config_a),
                player_id_for(rec.model_b, rec.search_config_b),
            }
        ]
        return {"matches": matches}

    @staticmethod
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
        return {
            "match_id": rec.match_id,
            "opponent_id": opp_id,
            "opponent_label": _humanize_model_label(opp_id),
            "games": games,
            "winrate": round(rec.win_rate_a if is_a else 1.0 - rec.win_rate_a, 4),
            "ci_low": round(rec.ci95_low if is_a else 1.0 - rec.ci95_high, 4),
            "ci_high": round(rec.ci95_high if is_a else 1.0 - rec.ci95_low, 4),
            "sweeps_w": rec.sweeps_a if is_a else rec.sweeps_b,
            "sweeps_l": rec.sweeps_b if is_a else rec.sweeps_a,
            "los": round(rec.los_a if is_a else 1.0 - rec.los_a, 4),
            "compare_target": compare_target,
        }

    def get_puzzles(
        self,
        *,
        format_id: str | None = None,
        model_a: str | None = None,
        model_b: str | None = None,
        phase: str | None = None,
        source: str | None = None,
    ) -> dict[str, object]:
        formats, default_format = discover_benchmark_formats(self._runs_dir)
        active_format = format_id.strip() if format_id and format_id.strip() else default_format
        if not active_format:
            return {"format": "", "formats": formats, "puzzles": [], "solved_a": 0, "solved_b": 0, "total": 0}
        raw_rows = _parse_jsonl(puzzles_path(self._runs_dir, active_format))
        phase_filter = phase.strip().lower() if phase and phase.strip() else None
        source_filter = _canonical_puzzle_source(source) if source and source.strip() else None
        cards: list[dict[str, object]] = []
        solved_a = 0
        solved_b = 0
        for idx, raw in enumerate(raw_rows):
            card, a_ok, b_ok = self._build_puzzle_card(
                raw, active_format=active_format, fallback_idx=idx, model_a=model_a, model_b=model_b
            )
            if phase_filter and str(card["phase"]).lower() != phase_filter:
                continue
            if source_filter and card["source"] != source_filter:
                continue
            cards.append(card)
            solved_a += int(a_ok)
            solved_b += int(b_ok)
        return {
            "format": active_format,
            "formats": formats,
            "puzzles": cards,
            "solved_a": solved_a,
            "solved_b": solved_b,
            "total": len(cards),
        }

    def _score_puzzle_model(
        self,
        raw: dict[str, object],
        *,
        puzzle_id: str,
        model_ref: str | None,
    ) -> _PuzzleScoreCacheEntry | None:
        if not model_ref or not model_ref.strip():
            return None
        key = (puzzle_id, model_ref)
        with self._cache_lock:
            cached = self._score_cache.get(key)
        if cached is not None:
            return cached
        entry = self._evaluate_puzzle_position(raw, model_ref=model_ref)
        if entry is not None:
            with self._cache_lock:
                self._score_cache[key] = entry
        return entry

    def _evaluate_puzzle_position(
        self,
        raw: dict[str, object],
        *,
        model_ref: str,
    ) -> _PuzzleScoreCacheEntry | None:
        run_id = raw.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return None
        try:
            game, _, obs, _, _, _, _, _ = self._replays.replay_to_step(
                run_id=run_id,
                iteration=_as_int(raw.get("iteration")),
                game_idx=_as_int(raw.get("game_idx")),
                step=_as_int(raw.get("step")),
            )
        except (FileNotFoundError, IndexError, ValueError):
            return None
        probs_list, _, _, _ = self._inference.evaluate_batch_full(
            [obs], run_id=run_id, checkpoint_name=model_ref, hero_seats=[game.hero_seat]
        )
        probs = tuple(probs_list[0]) if probs_list else ()
        best_idx = max(range(len(probs)), key=probs.__getitem__) if probs else None
        return _PuzzleScoreCacheEntry(probs=probs, solved_best_idx=best_idx)

    @staticmethod
    def _extract_cached_prob(
        entry: _PuzzleScoreCacheEntry | None,
        *,
        action_idx: int,
        fallback: object,
    ) -> float | None:
        if entry is not None and 0 <= action_idx < len(entry.probs):
            return round(entry.probs[action_idx], 4)
        return _as_float(fallback) if isinstance(fallback, (int, float)) and not isinstance(fallback, bool) else None

    def _enrich_puzzle_board_fields(self, raw: dict[str, object]) -> tuple[str, str, str, list[str]]:
        hero_summary = str(raw.get("hero_summary") or "")
        opp_summary = str(raw.get("opp_summary") or "")
        raw_phase = str(raw.get("decision_kind") or raw.get("phase") or "")
        raw_cards = raw.get("key_cards")
        key_cards = [str(c) for c in raw_cards if isinstance(c, str)][:2] if isinstance(raw_cards, list) else []
        if hero_summary and opp_summary and raw_phase:
            return _humanize_phase(raw_phase), hero_summary, opp_summary, key_cards
        run_id = raw.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return _humanize_phase(raw_phase), hero_summary or "Hero: --", opp_summary or "Opp: --", key_cards
        try:
            game, _, obs, _, _, _, _, _ = self._replays.replay_to_step(
                run_id=run_id,
                iteration=_as_int(raw.get("iteration")),
                game_idx=_as_int(raw.get("game_idx")),
                step=_as_int(raw.get("step")),
            )
        except (FileNotFoundError, IndexError, ValueError):
            return _humanize_phase(raw_phase), hero_summary or "Hero: --", opp_summary or "Opp: --", key_cards
        h_sum, o_sum, cards = _summarize_player_counts(obs, hero_seat=game.hero_seat)
        inferred_phase = ActionSpaceEnum(int(obs.action_space.action_space_type)).name
        return (
            _humanize_phase(raw_phase or inferred_phase),
            hero_summary or h_sum,
            opp_summary or o_sum,
            key_cards or cards,
        )

    def _build_puzzle_card(
        self,
        raw: dict[str, object],
        *,
        active_format: str,
        fallback_idx: int,
        model_a: str | None,
        model_b: str | None,
    ) -> tuple[dict[str, object], bool, bool]:
        puzzle_id = str(raw.get("puzzle_id") or f"puz-{fallback_idx}")
        best_idx = _as_int(raw.get("best_action_index"), 0)
        alt_idx = _as_int(raw.get("alt_action_index"), 1)
        score_a = self._score_puzzle_model(raw, puzzle_id=puzzle_id, model_ref=model_a)
        score_b = self._score_puzzle_model(raw, puzzle_id=puzzle_id, model_ref=model_b)
        prob_a_best = self._extract_cached_prob(score_a, action_idx=best_idx, fallback=raw.get("prob_a_best"))
        prob_a_alt = self._extract_cached_prob(score_a, action_idx=alt_idx, fallback=raw.get("prob_a_alt"))
        prob_b_best = self._extract_cached_prob(score_b, action_idx=best_idx, fallback=raw.get("prob_b_best"))
        prob_b_alt = self._extract_cached_prob(score_b, action_idx=alt_idx, fallback=raw.get("prob_b_alt"))
        phase, hero_summary, opp_summary, key_cards = self._enrich_puzzle_board_fields(raw)
        margin = _as_float(raw.get("margin"))
        best_wr = _as_float(raw.get("best_playout_wr"), min(1.0, 0.5 + 0.5 * margin))
        alt_wr = _as_float(raw.get("alt_playout_wr"), max(0.0, best_wr - margin))
        seed = _as_int(raw.get("seed"))
        source = _canonical_puzzle_source(str(raw.get("source") or "auto_mined"))
        fallback_seed_label = f"Watchlist #{puzzle_id[:6]}" if source == "watchlist" else f"Seed #{seed}"
        seed_label = str(raw.get("seed_label") or fallback_seed_label)
        card: dict[str, object] = {
            "puzzle_id": puzzle_id,
            "format": str(raw.get("format_id") or raw.get("format") or active_format),
            "turn": _as_int(raw.get("turn")),
            "phase": phase,
            "source": source,
            "seed_label": seed_label,
            "hero_summary": hero_summary,
            "opp_summary": opp_summary,
            "key_cards": key_cards,
            "actions": [
                {
                    "label": str(raw.get("best_action_label") or f"Action #{best_idx}"),
                    "playout_wr": round(best_wr, 4),
                    "prob_a": prob_a_best,
                    "prob_b": prob_b_best,
                    "is_best": True,
                },
                {
                    "label": str(raw.get("alt_action_label") or f"Action #{alt_idx}"),
                    "playout_wr": round(alt_wr, 4),
                    "prob_a": prob_a_alt,
                    "prob_b": prob_b_alt,
                    "is_best": False,
                },
            ],
            "margin": round(margin, 4),
            "ci_low": round(_as_float(raw.get("paired_ci_low", raw.get("ci_low"))), 4),
            "ci_high": round(_as_float(raw.get("paired_ci_high", raw.get("ci_high"))), 4),
            "playouts": _as_int(raw.get("playouts_per_action", raw.get("playouts")), 64),
            "tabletop_target": {
                "run_id": str(raw.get("run_id") or ""),
                "iteration": _as_int(raw.get("iteration")),
                "game_idx": _as_int(raw.get("game_idx")),
                "step": _as_int(raw.get("step")),
                "seed": seed,
            },
        }
        a_solved = (
            score_a.solved_best_idx == best_idx
            if score_a is not None
            else (prob_a_best is not None and prob_a_alt is not None and prob_a_best > prob_a_alt)
        )
        b_solved = (
            score_b.solved_best_idx == best_idx
            if score_b is not None
            else (prob_b_best is not None and prob_b_alt is not None and prob_b_best > prob_b_alt)
        )
        return card, a_solved, b_solved

    def promote_watchlist_item(
        self,
        item: dict[str, object],
        *,
        format_id: str,
        playouts: int = 64,
        min_margin: float = 0.15,
        model_ref: str | None = None,
    ) -> PuzzlePromotionResult:
        safe_format = _validate_format_id(format_id)
        run_id = str(item.get("run_id") or "")
        iteration = _as_int(item.get("iteration"))
        game_idx = _as_int(item.get("game_idx"))
        game, env, obs, actual_step, _, _, _, _ = self._replays.replay_to_step(
            run_id=run_id, iteration=iteration, game_idx=game_idx, step=_as_int(item.get("step"))
        )
        legal_actions = serialize_legal_actions(obs, hero_seat=game.hero_seat)
        if len(legal_actions) < _MIN_LEGAL_ACTIONS:
            return PuzzlePromotionResult(
                admitted=False,
                margin=0.0,
                ci_low=0.0,
                ci_high=0.0,
                detail="Position has fewer than 2 legal actions and cannot form a tactical puzzle.",
            )
        acting_seat = int(obs.agent.player_index)
        eval_ref = model_ref or self._inference.find_checkpoint_for_iteration(run_id, iteration) or "heuristic"
        probs_list, wdls_list, _, _ = self._inference.evaluate_batch_full(
            [obs], run_id=run_id, checkpoint_name=eval_ref, hero_seats=[acting_seat]
        )
        root_v = _as_float(wdls_list[0].get("win"), 0.5) if wdls_list else 0.5
        action_indices = _pick_top_action_indices(probs_list[0] if probs_list else [], len(legal_actions))
        outcomes = _run_paired_crn_outcomes(
            env,
            action_indices=action_indices,
            playouts=playouts,
            acting_seat=acting_seat,
            hero_seat=acting_seat,
            seed=game.seed or 42,
            max_plies=_PROMOTE_MAX_PLIES,
        )
        ranked = sorted(action_indices, key=lambda idx: sum(outcomes[idx]) / max(1, len(outcomes[idx])), reverse=True)
        best_idx, alt_idx = ranked[0], ranked[1]
        pairs = list(zip(outcomes[best_idx], outcomes[alt_idx], strict=True))
        ver = summarize_paired_outcomes(
            pairs,
            best_action=best_idx,
            alt_action=alt_idx,
            playouts_per_action=playouts,
            max_plies=_PROMOTE_MAX_PLIES,
            min_margin=min_margin,
        )
        if not ver.admitted:
            detail = (
                f"Margin {ver.margin * 100:+.1f}% "
                f"(95% CI [{ver.paired_ci_low * 100:+.1f}%, {ver.paired_ci_high * 100:+.1f}%]) "
                f"did not pass +{min_margin * 100:.1f}% & CI > 0 gate"
            )
            return PuzzlePromotionResult(
                admitted=False,
                margin=round(ver.margin, 4),
                ci_low=round(ver.paired_ci_low, 4),
                ci_high=round(ver.paired_ci_high, 4),
                detail=detail,
            )
        record = self._build_promoted_record(
            item=item,
            obs=obs,
            format_id=safe_format,
            hero_seat=acting_seat,
            actual_step=actual_step,
            seed=game.seed,
            root_v=root_v,
            legal_actions=legal_actions,
            best_idx=best_idx,
            alt_idx=alt_idx,
            best_wr=ver.best_playout_wr,
            alt_wr=ver.alt_playout_wr,
            playouts=playouts,
            margin=ver.margin,
            ci_low=ver.paired_ci_low,
            ci_high=ver.paired_ci_high,
        )
        append_puzzle(self._runs_dir, record)
        card, _, _ = self._build_puzzle_card(
            record.to_dict(), active_format=safe_format, fallback_idx=0, model_a=None, model_b=None
        )
        return PuzzlePromotionResult(
            admitted=True,
            margin=round(ver.margin, 4),
            ci_low=round(ver.paired_ci_low, 4),
            ci_high=round(ver.paired_ci_high, 4),
            puzzle=card,
        )

    @staticmethod
    def _build_promoted_record(
        *,
        item: dict[str, object],
        obs: managym.Observation,
        format_id: str,
        hero_seat: int,
        actual_step: int,
        seed: int,
        root_v: float,
        legal_actions: list[dict[str, object]],
        best_idx: int,
        alt_idx: int,
        best_wr: float,
        alt_wr: float,
        playouts: int,
        margin: float,
        ci_low: float,
        ci_high: float,
    ) -> PuzzleRecord:
        item_id = str(item.get("id") or uuid.uuid4().hex[:6])
        hero_summary, opp_summary, obs_cards = _summarize_player_counts(obs, hero_seat=hero_seat)
        raw_cards = item.get("key_cards")
        item_cards = [str(c) for c in raw_cards if isinstance(c, str)][:2] if isinstance(raw_cards, list) else []
        decision_kind = ActionSpaceEnum(int(obs.action_space.action_space_type)).name
        return PuzzleRecord(
            puzzle_id=f"wl-{item_id}",
            format_id=format_id,
            source="watchlist_promotion",
            run_id=str(item.get("run_id") or ""),
            iteration=_as_int(item.get("iteration")),
            game_idx=_as_int(item.get("game_idx")),
            step=actual_step,
            turn=int(obs.turn.turn_number),
            phase=str(obs.turn.phase).split(".")[-1],
            step_name=str(obs.turn.step).split(".")[-1],
            decision_kind=decision_kind,
            hero_seat=hero_seat,
            seed=seed,
            hero_summary=hero_summary,
            opp_summary=opp_summary,
            key_cards=tuple(item_cards or obs_cards),
            root_value_v=round(root_v, 4),
            best_action_index=best_idx,
            best_action_label=str(legal_actions[best_idx]["label"]),
            alt_action_index=alt_idx,
            alt_action_label=str(legal_actions[alt_idx]["label"]),
            best_playout_wr=round(best_wr, 4),
            alt_playout_wr=round(alt_wr, 4),
            playouts_per_action=playouts,
            max_plies=_PROMOTE_MAX_PLIES,
            margin=round(margin, 4),
            paired_ci_low=round(ci_low, 4),
            paired_ci_high=round(ci_high, 4),
        )

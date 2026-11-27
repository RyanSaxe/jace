"""Tactical puzzle listing, board enrichment, and cached checkpoint scoring."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from managym.enums import ActionSpaceEnum

from jace.eval.puzzles import puzzles_path
from jace.studio.benchmarks.arena import discover_benchmark_formats
from jace.studio.benchmarks.promotion import (
    PuzzlePromotionResult,
    as_float,
    as_int,
    promote_watchlist_position,
    summarize_player_counts,
)

if TYPE_CHECKING:
    from pathlib import Path

    from jace.studio.inference import InferenceEngine
    from jace.studio.replay import ReplayService

__all__ = (
    "PuzzlePromotionResult",
    "PuzzleService",
    "PuzzleSourceKind",
)

type PuzzleSourceKind = Literal["auto_mined", "search_disagreement", "watchlist"]


@dataclass(frozen=True, slots=True)
class _PuzzleScoreCacheEntry:
    probs: tuple[float, ...]
    solved_best_idx: int | None


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


class PuzzleService:
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

    def promote_watchlist_item(
        self,
        item: dict[str, object],
        *,
        format_id: str,
        playouts: int = 64,
        min_margin: float = 0.15,
        model_ref: str | None = None,
    ) -> PuzzlePromotionResult:
        def _build_card(raw: dict[str, object], fmt: str) -> dict[str, object]:
            card, _, _ = self._build_puzzle_card(raw, active_format=fmt, fallback_idx=0, model_a=None, model_b=None)
            return card

        return promote_watchlist_position(
            self._runs_dir,
            item,
            replays=self._replays,
            inference=self._inference,
            card_builder=_build_card,
            format_id=format_id,
            playouts=playouts,
            min_margin=min_margin,
            model_ref=model_ref,
        )

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
                iteration=as_int(raw.get("iteration")),
                game_idx=as_int(raw.get("game_idx")),
                step=as_int(raw.get("step")),
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
        return as_float(fallback) if isinstance(fallback, (int, float)) and not isinstance(fallback, bool) else None

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
                iteration=as_int(raw.get("iteration")),
                game_idx=as_int(raw.get("game_idx")),
                step=as_int(raw.get("step")),
            )
        except (FileNotFoundError, IndexError, ValueError):
            return _humanize_phase(raw_phase), hero_summary or "Hero: --", opp_summary or "Opp: --", key_cards
        h_sum, o_sum, cards = summarize_player_counts(obs, hero_seat=game.hero_seat)
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
        best_idx = as_int(raw.get("best_action_index"), 0)
        alt_idx = as_int(raw.get("alt_action_index"), 1)
        score_a = self._score_puzzle_model(raw, puzzle_id=puzzle_id, model_ref=model_a)
        score_b = self._score_puzzle_model(raw, puzzle_id=puzzle_id, model_ref=model_b)
        prob_a_best = self._extract_cached_prob(score_a, action_idx=best_idx, fallback=raw.get("prob_a_best"))
        prob_a_alt = self._extract_cached_prob(score_a, action_idx=alt_idx, fallback=raw.get("prob_a_alt"))
        prob_b_best = self._extract_cached_prob(score_b, action_idx=best_idx, fallback=raw.get("prob_b_best"))
        prob_b_alt = self._extract_cached_prob(score_b, action_idx=alt_idx, fallback=raw.get("prob_b_alt"))
        phase, hero_summary, opp_summary, key_cards = self._enrich_puzzle_board_fields(raw)
        margin = as_float(raw.get("margin"))
        best_wr = as_float(raw.get("best_playout_wr"), min(1.0, 0.5 + 0.5 * margin))
        alt_wr = as_float(raw.get("alt_playout_wr"), max(0.0, best_wr - margin))
        seed = as_int(raw.get("seed"))
        source = _canonical_puzzle_source(str(raw.get("source") or "auto_mined"))
        fallback_seed_label = f"Watchlist #{puzzle_id[:6]}" if source == "watchlist" else f"Seed #{seed}"
        seed_label = str(raw.get("seed_label") or fallback_seed_label)
        card: dict[str, object] = {
            "puzzle_id": puzzle_id,
            "format": str(raw.get("format_id") or raw.get("format") or active_format),
            "turn": as_int(raw.get("turn")),
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
            "ci_low": round(as_float(raw.get("paired_ci_low", raw.get("ci_low"))), 4),
            "ci_high": round(as_float(raw.get("paired_ci_high", raw.get("ci_high"))), 4),
            "playouts": as_int(raw.get("playouts_per_action", raw.get("playouts")), 64),
            "tabletop_target": {
                "run_id": str(raw.get("run_id") or ""),
                "iteration": as_int(raw.get("iteration")),
                "game_idx": as_int(raw.get("game_idx")),
                "step": as_int(raw.get("step")),
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

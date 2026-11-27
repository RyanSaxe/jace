"""Incremental Parquet game indexer with replayability and surprise scoring.

Incrementally scans `games/eval-iter-*.jsonl` recordings across discovered runs,
derives metadata solely from recorded payloads (never fabricating set codes),
verifies replayability on the installed `managym` engine, scores playable games
for model overconfidence/underconfidence surprises and win% swings, and delegates
faceted DuckDB queries to `jace.studio.index.queries`.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from jace.studio.index.queries import (
    GameFacets,
    GameQuery,
    GameRecordSummary,
    available_cards_parquet,
    facets_parquet,
    query_games_parquet,
    run_game_counts_parquet,
)
from jace.studio.replay import RecordedGame, engine_step_errors, load_recorded_games, sample_game_observations
from jace.studio.runs import discover_runs

if TYPE_CHECKING:
    from pathlib import Path

    from jace.studio.inference import InferenceEngine

__all__ = ("StudioIndex",)

_EVAL_FILE_RE = re.compile(r"^eval-iter-(\d+)\.jsonl$")
_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
_INDEX_SCHEMA_VERSION = "3"
_SYNC_TTL_SECONDS = 5.0
_FAST_FAIL_SAMPLE = 3

_GAMES_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("iteration", pa.int32()),
        ("game_idx", pa.int32()),
        ("set_code", pa.string()),
        ("seed", pa.int64()),
        ("result", pa.string()),
        ("winner_seat", pa.int32()),
        ("hero_seat", pa.int32()),
        ("steps", pa.int32()),
        ("turns", pa.int32()),
        ("hero_name", pa.string()),
        ("opp_name", pa.string()),
        ("hero_colors", pa.string()),
        ("opp_colors", pa.string()),
        ("cards_seen", pa.list_(pa.string())),
        ("key_cards", pa.list_(pa.string())),
        ("playable", pa.bool_()),
        ("max_pred_win", pa.float32()),
        ("min_pred_win", pa.float32()),
        ("surprise_score", pa.float32()),
        ("max_swing", pa.float32()),
        ("turning_point_step", pa.int32()),
        ("eval_source", pa.string()),
        ("source_file", pa.string()),
    ]
)

_COLOR_BY_LAND = {
    "Plains": "W",
    "Island": "U",
    "Swamp": "B",
    "Mountain": "R",
    "Forest": "G",
}


def _deck_colors(decklist: dict[str, object]) -> str:
    letters = [color for land, color in _COLOR_BY_LAND.items() if land in decklist]
    return "".join(letters) or "C"


def _resolve_hero_seat(context: object) -> int:
    if not isinstance(context, dict):
        return 0
    players = context.get("players")
    if isinstance(players, list):
        for idx, role in enumerate(players[:2]):
            if str(role).lower() == "agent":
                return idx
    return 0


def _resolve_set_code(game: dict[str, object], context: object) -> str:
    raw = game.get("set_code") or game.get("set")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().upper()
    if isinstance(context, dict):
        ctx_set = context.get("set_code") or context.get("set")
        if isinstance(ctx_set, str) and ctx_set.strip():
            return ctx_set.strip().upper()
    return ""


def _extract_result(end_block: object, *, hero_seat: int) -> tuple[str, int, int]:
    if not isinstance(end_block, dict):
        return "draw", -1, 0
    winner = end_block.get("winner")
    turns = end_block.get("turn") or end_block.get("turns") or 0
    turns_int = int(turns) if isinstance(turns, (int, float)) else 0
    if isinstance(winner, int) and winner in (0, 1):
        return ("win" if winner == hero_seat else "loss"), winner, turns_int
    return "draw", -1, turns_int


def _extract_player_deck(players: object, idx: int) -> tuple[str, dict[str, object]]:
    fallback = "Hero" if idx == 0 else "Opponent"
    if not isinstance(players, list) or idx >= len(players):
        return fallback, {}
    entry = players[idx]
    if not isinstance(entry, dict):
        return fallback, {}
    raw_deck = entry.get("decklist")
    deck: dict[str, object] = {str(k): v for k, v in raw_deck.items()} if isinstance(raw_deck, dict) else {}
    return str(entry.get("name") or fallback), deck


def _parse_single_game(
    game: dict[str, object],
    *,
    run_id: str,
    iteration: int,
    game_idx: int,
    hero_seat: int,
    context: object,
    source_file: str,
) -> dict[str, object]:
    opp_seat = 1 - hero_seat
    hero_name, hero_deck = _extract_player_deck(game.get("players"), hero_seat)
    opp_name, opp_deck = _extract_player_deck(game.get("players"), opp_seat)
    all_cards = sorted({*hero_deck.keys(), *opp_deck.keys()})
    non_basics = [name for name in all_cards if name not in _BASIC_LANDS]
    actions = game.get("actions")
    step_count = len(actions) if isinstance(actions, list) else 0
    result, winner_seat, turns = _extract_result(game.get("end"), hero_seat=hero_seat)
    est_turns = turns if turns > 0 else max(1, step_count // 12)
    seed_raw = game.get("seed")
    seed = int(seed_raw) if isinstance(seed_raw, (int, float)) else game_idx
    return {
        "run_id": run_id,
        "iteration": iteration,
        "game_idx": game_idx,
        "set_code": _resolve_set_code(game, context),
        "seed": seed,
        "result": result,
        "winner_seat": winner_seat,
        "hero_seat": hero_seat,
        "steps": step_count,
        "turns": est_turns,
        "hero_name": hero_name,
        "opp_name": opp_name,
        "hero_colors": _deck_colors(hero_deck),
        "opp_colors": _deck_colors(opp_deck),
        "cards_seen": all_cards,
        "key_cards": non_basics[:8],
        "playable": False,
        "max_pred_win": 0.5,
        "min_pred_win": 0.5,
        "surprise_score": 0.0,
        "max_swing": 0.0,
        "turning_point_step": 0,
        "eval_source": "none",
        "source_file": source_file,
    }


def _parse_jsonl_games(raw_line: str) -> list[tuple[dict[str, object], int, object]]:
    line = raw_line.strip()
    if not line:
        return []
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    top_ctx = payload.get("context")
    top_seat = _resolve_hero_seat(top_ctx)
    games = payload.get("games")
    items = games if isinstance(games, list) else [payload]
    found: list[tuple[dict[str, object], int, object]] = []
    for item in items:
        if isinstance(item, dict) and "actions" in item:
            item_ctx = item.get("context", top_ctx)
            seat = _resolve_hero_seat(item_ctx) if "context" in item else top_seat
            found.append((item, seat, item_ctx))
    return found


def _summarize_win_curve(
    step_ids: list[int],
    win_probs: list[float],
    *,
    result: str,
) -> tuple[float, float, float, float, int]:
    if not win_probs:
        return 0.5, 0.5, 0.0, 0.0, 0
    max_idx = max(range(len(win_probs)), key=win_probs.__getitem__)
    min_idx = min(range(len(win_probs)), key=win_probs.__getitem__)
    max_win, min_win = win_probs[max_idx], win_probs[min_idx]
    max_swing = 0.0
    swing_step = step_ids[0]
    for prev_p, cur_p, cur_step in zip(win_probs, win_probs[1:], step_ids[1:], strict=False):
        delta = abs(cur_p - prev_p)
        if delta > max_swing:
            max_swing = delta
            swing_step = cur_step
    if result == "loss":
        return max_win, min_win, round(max_win, 4), round(max_swing, 4), step_ids[max_idx]
    if result == "win":
        return max_win, min_win, round(1.0 - min_win, 4), round(max_swing, 4), step_ids[min_idx]
    return max_win, min_win, round(abs(max_win - 0.5), 4), round(max_swing, 4), swing_step


def _score_row_replayability(
    row: dict[str, object],
    *,
    recorded: RecordedGame,
    inference: InferenceEngine | None,
    checkpoint_name: str | None,
) -> bool:
    run_id = recorded.run_id
    step_errors = engine_step_errors()
    try:
        playable, sampled = sample_game_observations(recorded, stride=4)
    except (*step_errors, FileNotFoundError):
        return False
    if not playable:
        return False
    row["playable"] = True
    step_ids = [idx for idx, _ in sampled]
    obs_list = [obs for _, obs in sampled]
    if inference is not None and obs_list:
        win_probs, src = inference.evaluate_batch_win_probs(
            obs_list, run_id=run_id, checkpoint_name=checkpoint_name, hero_seat=recorded.hero_seat
        )
    else:
        win_probs, src = [], "none"
    max_w, min_w, surprise, swing, turn_step = _summarize_win_curve(step_ids, win_probs, result=str(row["result"]))
    row["max_pred_win"] = max_w
    row["min_pred_win"] = min_w
    row["surprise_score"] = surprise
    row["max_swing"] = swing
    row["turning_point_step"] = turn_step
    row["eval_source"] = src
    return True


def _extract_games_from_jsonl(
    jsonl_path: Path,
    *,
    run_id: str,
    iteration: int,
    inference: InferenceEngine | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            for item, hero_seat, ctx in _parse_jsonl_games(raw_line):
                rows.append(
                    _parse_single_game(
                        item,
                        run_id=run_id,
                        iteration=iteration,
                        game_idx=len(rows),
                        hero_seat=hero_seat,
                        context=ctx,
                        source_file=str(jsonl_path),
                    )
                )
    recorded_games = load_recorded_games(jsonl_path, run_id=run_id, iteration=iteration)
    ckpt = inference.find_checkpoint_for_iteration(run_id, iteration) if inference is not None else None
    consecutive_failures = 0
    saw_playable = False
    for row, recorded in zip(rows, recorded_games, strict=False):
        if not saw_playable and consecutive_failures >= _FAST_FAIL_SAMPLE:
            break
        ok = _score_row_replayability(row, recorded=recorded, inference=inference, checkpoint_name=ckpt)
        if ok:
            saw_playable = True
            consecutive_failures = 0
        else:
            consecutive_failures += 1
    return rows


def _partition_path(games_root: Path, *, run_id: str, iteration: int) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("_")[:48] or "run"
    digest = hashlib.sha1(run_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return games_root / f"run={slug}_{digest}" / f"iter-{iteration:04d}.parquet"


def _write_partition(rows: list[dict[str, object]], *, games_root: Path, run_id: str, iteration: int) -> None:
    target = _partition_path(games_root, run_id=run_id, iteration=iteration)
    if not rows:
        target.unlink(missing_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{threading.get_ident()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=_GAMES_SCHEMA), tmp)
    tmp.replace(target)


def _file_stamp(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


class StudioIndex:
    def __init__(
        self,
        runs_dir: Path,
        *,
        index_dir: Path | None = None,
        inference: InferenceEngine | None = None,
    ) -> None:
        self._runs_dir = runs_dir.resolve()
        self._index_dir = (index_dir or (self._runs_dir / ".studio" / "index")).resolve()
        self._games_root = self._index_dir / "games"
        self._manifest_path = self._index_dir / "manifest.json"
        self._inference = inference
        self._lock = threading.Lock()
        self._last_sync_mono = 0.0

    def _load_manifest(self) -> dict[str, str]:
        if not self._manifest_path.is_file():
            return {}
        try:
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(raw, dict) or raw.get("__schema_version__") != _INDEX_SCHEMA_VERSION:
            if self._games_root.is_dir():
                shutil.rmtree(self._games_root, ignore_errors=True)
            return {}
        return {str(k): str(v) for k, v in raw.items() if k != "__schema_version__"}

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        payload = {"__schema_version__": _INDEX_SCHEMA_VERSION, **manifest}
        tmp = self._manifest_path.with_name(f".manifest.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self._manifest_path)

    def _sync_run_games(
        self,
        run_id: str,
        *,
        manifest: dict[str, str],
        seen_keys: set[str],
        seen_partitions: set[Path],
    ) -> tuple[int, int]:
        games_dir = self._runs_dir / run_id / "games"
        if not games_dir.is_dir():
            return 0, 0
        files_done = 0
        games_done = 0
        for path in sorted(games_dir.iterdir()):
            match = _EVAL_FILE_RE.match(path.name)
            if match is None or not path.is_file():
                continue
            iteration = int(match.group(1))
            seen_partitions.add(_partition_path(self._games_root, run_id=run_id, iteration=iteration).resolve())
            key = str(path.resolve())
            seen_keys.add(key)
            stamp = _file_stamp(path)
            if manifest.get(key) == stamp:
                continue
            rows = _extract_games_from_jsonl(path, run_id=run_id, iteration=iteration, inference=self._inference)
            _write_partition(rows, games_root=self._games_root, run_id=run_id, iteration=iteration)
            manifest[key] = stamp
            files_done += 1
            games_done += len(rows)
        return files_done, games_done

    def _prune_orphan_partitions(self, seen_partitions: set[Path]) -> None:
        if not self._games_root.is_dir():
            return
        for part in self._games_root.rglob("*.parquet"):
            if part.resolve() not in seen_partitions:
                part.unlink(missing_ok=True)

    def sync(self, *, force: bool = True) -> dict[str, int]:
        with self._lock:
            now = time.monotonic()
            if not force and self._last_sync_mono > 0.0 and (now - self._last_sync_mono) < _SYNC_TTL_SECONDS:
                return {"indexed_files": 0, "indexed_games": 0}
            manifest = self._load_manifest()
            seen_keys: set[str] = set()
            seen_partitions: set[Path] = set()
            indexed_files = 0
            indexed_games = 0
            for summary in discover_runs(self._runs_dir):
                f_cnt, g_cnt = self._sync_run_games(
                    summary.run_id, manifest=manifest, seen_keys=seen_keys, seen_partitions=seen_partitions
                )
                indexed_files += f_cnt
                indexed_games += g_cnt
            pruned = [key for key in manifest if key not in seen_keys]
            for key in pruned:
                del manifest[key]
            self._prune_orphan_partitions(seen_partitions)
            if indexed_files > 0 or pruned or not self._manifest_path.exists():
                self._save_manifest(manifest)
            self._last_sync_mono = time.monotonic()
            return {"indexed_files": indexed_files, "indexed_games": indexed_games}

    def query_games(self, query: GameQuery) -> tuple[list[GameRecordSummary], int]:
        self.sync(force=False)
        return query_games_parquet(self._games_root, query)

    def run_game_counts(self) -> dict[str, tuple[int, int]]:
        self.sync(force=False)
        return run_game_counts_parquet(self._games_root)

    def available_cards(self, *, run_id: str | None = None, limit: int | None = None) -> list[str]:
        self.sync(force=False)
        return available_cards_parquet(self._games_root, run_id=run_id, limit=limit)

    def facets(self, *, run_id: str | None = None) -> GameFacets:
        self.sync(force=False)
        return facets_parquet(self._games_root, run_id=run_id)

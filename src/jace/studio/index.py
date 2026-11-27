"""DuckDB and Parquet game indexer with replayability and mistake/surprise scoring.

Incrementally scans `games/eval-iter-*.jsonl` recordings across discovered runs,
derives metadata solely from recorded payloads (never fabricating set codes),
verifies replayability on the installed `managym` engine, scores playable games
for model overconfidence/underconfidence surprises and win% swings, and serves
faceted DuckDB queries in milliseconds.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from jace.studio.replay import RecordedGame, engine_step_errors, load_recorded_games, sample_game_observations
from jace.studio.runs import discover_runs

if TYPE_CHECKING:
    from pathlib import Path

    from jace.studio.inference import InferenceEngine

__all__ = (
    "GameFacets",
    "GameQuery",
    "GameRecordSummary",
    "StudioIndex",
)

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

_SORT_SQL = {
    "surprise_desc": "surprise_score DESC, max_swing DESC, steps DESC",
    "swing_desc": "max_swing DESC, surprise_score DESC, steps DESC",
    "steps_desc": "steps DESC, iteration DESC, seed ASC",
    "steps_asc": "steps ASC, iteration DESC, seed ASC",
    "turns_desc": "turns DESC, steps DESC",
    "iter_desc": "iteration DESC, game_idx ASC",
    "seed_asc": "seed ASC, iteration DESC",
}


@dataclass(frozen=True, slots=True)
class GameQuery:
    run_id: str | None = None
    set_code: str | None = None
    card: str | None = None
    result: str | None = None
    colors: str | None = None
    min_steps: int | None = None
    min_turns: int | None = None
    max_turns: int | None = None
    min_surprise: float | None = None
    min_swing: float | None = None
    include_unplayable: bool = False
    search: str | None = None
    sort: str = "surprise_desc"
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True, slots=True)
class GameRecordSummary:
    run_id: str
    iteration: int
    game_idx: int
    set_code: str
    seed: int
    result: str
    winner_seat: int
    hero_seat: int
    steps: int
    turns: int
    hero_name: str
    opp_name: str
    hero_colors: str
    opp_colors: str
    cards_seen: tuple[str, ...]
    key_cards: tuple[str, ...]
    playable: bool
    max_pred_win: float
    min_pred_win: float
    surprise_score: float
    max_swing: float
    turning_point_step: int
    eval_source: str


@dataclass(frozen=True, slots=True)
class GameFacets:
    runs: tuple[str, ...]
    colors: tuple[str, ...]
    sets: tuple[str, ...]
    results: tuple[str, ...]
    cards: tuple[str, ...]
    total_playable: int
    total_unplayable: int


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


def _append_scalar_filters(query: GameQuery, *, clauses: list[str], params: list[object]) -> None:
    scalar_filters: tuple[tuple[str, object], ...] = (
        ("run_id = ?", query.run_id),
        ("set_code = ?", query.set_code.upper() if query.set_code else None),
        ("result = ?", query.result.lower() if query.result else None),
        ("steps >= ?", query.min_steps),
        ("turns >= ?", query.min_turns),
        ("turns <= ?", query.max_turns),
        ("surprise_score >= ?", query.min_surprise),
        ("max_swing >= ?", query.min_swing),
    )
    for sql_frag, val in scalar_filters:
        if val is not None and val != "":
            clauses.append(sql_frag)
            params.append(val)


def _build_where_clause(query: GameQuery) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if not query.include_unplayable:
        clauses.append("playable = TRUE")
    _append_scalar_filters(query, clauses=clauses, params=params)
    if query.colors:
        raw_colors = query.colors.strip().upper()
        canonical = "".join(c for c in "WUBRG" if c in raw_colors) or raw_colors
        clauses.append("(hero_colors = ? OR opp_colors = ?)")
        params.extend((canonical, canonical))
    if query.card:
        clauses.append("contains(lower(array_to_string(cards_seen, '|')), lower(?))")
        params.append(query.card.strip())
    if query.search:
        term = query.search.strip()
        clauses.append(
            "(contains(lower(run_id), lower(?)) OR CAST(seed AS VARCHAR) = ? "
            "OR contains(lower(hero_colors), lower(?)) OR contains(lower(opp_colors), lower(?)) "
            "OR contains(lower(array_to_string(cards_seen, '|')), lower(?)))"
        )
        params.extend((term, term, term, term, term))
    return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params


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
        if not self._games_root.is_dir() or not any(self._games_root.rglob("*.parquet")):
            return [], 0
        glob_pattern = str(self._games_root / "**" / "*.parquet")
        where_sql, params = _build_where_clause(query)
        order_sql = _SORT_SQL.get(query.sort, _SORT_SQL["surprise_desc"])
        count_parts = ["SELECT COUNT(*) FROM read_parquet(?, hive_partitioning = false)", where_sql]
        select_parts = [
            "SELECT run_id, iteration, game_idx, set_code, seed, result, winner_seat, hero_seat,",
            "steps, turns, hero_name, opp_name, hero_colors, opp_colors, cards_seen, key_cards,",
            "playable, max_pred_win, min_pred_win, surprise_score, max_swing, turning_point_step, eval_source",
            "FROM read_parquet(?, hive_partitioning = false)",
            where_sql,
            "ORDER BY",
            order_sql,
        ]
        select_params: list[object] = [glob_pattern, *params]
        if query.limit > 0:
            select_parts.append("LIMIT ? OFFSET ?")
            select_params.extend([query.limit, query.offset])
        count_sql = " ".join(count_parts)
        select_sql = " ".join(select_parts)
        with duckdb.connect(":memory:") as conn:
            total_row = conn.execute(count_sql, [glob_pattern, *params]).fetchone()
            total = int(total_row[0]) if total_row is not None else 0
            records = conn.execute(select_sql, select_params).fetchall()
        items = [
            GameRecordSummary(
                run_id=str(row[0]),
                iteration=int(row[1]),
                game_idx=int(row[2]),
                set_code=str(row[3] or ""),
                seed=int(row[4]),
                result=str(row[5]),
                winner_seat=int(row[6]),
                hero_seat=int(row[7]),
                steps=int(row[8]),
                turns=int(row[9]),
                hero_name=str(row[10]),
                opp_name=str(row[11]),
                hero_colors=str(row[12]),
                opp_colors=str(row[13]),
                cards_seen=tuple(row[14] or ()),
                key_cards=tuple(row[15] or ()),
                playable=bool(row[16]),
                max_pred_win=round(float(row[17]), 4),
                min_pred_win=round(float(row[18]), 4),
                surprise_score=round(float(row[19]), 4),
                max_swing=round(float(row[20]), 4),
                turning_point_step=int(row[21]),
                eval_source=str(row[22] or "none"),
            )
            for row in records
        ]
        return items, total

    def run_game_counts(self) -> dict[str, tuple[int, int]]:
        self.sync(force=False)
        if not self._games_root.is_dir() or not any(self._games_root.rglob("*.parquet")):
            return {}
        glob_pattern = str(self._games_root / "**" / "*.parquet")
        counts_sql = (
            "SELECT run_id, "
            "SUM(CASE WHEN playable THEN 1 ELSE 0 END) AS playable_cnt, "
            "SUM(CASE WHEN NOT playable THEN 1 ELSE 0 END) AS legacy_cnt "
            "FROM read_parquet(?, hive_partitioning = false) "
            "GROUP BY run_id"
        )
        with duckdb.connect(":memory:") as conn:
            rows = conn.execute(counts_sql, [glob_pattern]).fetchall()
        return {str(row[0]): (int(row[1] or 0), int(row[2] or 0)) for row in rows}

    def available_cards(self, *, run_id: str | None = None, limit: int | None = None) -> list[str]:
        self.sync(force=False)
        if not self._games_root.is_dir() or not any(self._games_root.rglob("*.parquet")):
            return []
        glob_pattern = str(self._games_root / "**" / "*.parquet")
        where_sql = "WHERE playable = TRUE AND run_id = ?" if run_id else "WHERE playable = TRUE"
        params: list[object] = [glob_pattern, run_id] if run_id else [glob_pattern]
        cards_parts = [
            "SELECT DISTINCT unnest(cards_seen) AS card",
            "FROM read_parquet(?, hive_partitioning = false)",
            where_sql,
            "ORDER BY card ASC",
        ]
        if limit is not None and limit > 0:
            cards_parts.append("LIMIT ?")
            params.append(limit)
        cards_sql = " ".join(cards_parts)
        with duckdb.connect(":memory:") as conn:
            rows = conn.execute(cards_sql, params).fetchall()
        return [str(row[0]) for row in rows if row[0] is not None]

    def facets(self, *, run_id: str | None = None) -> GameFacets:
        self.sync(force=False)
        if not self._games_root.is_dir() or not any(self._games_root.rglob("*.parquet")):
            return GameFacets((), (), (), (), (), 0, 0)
        glob_pattern = str(self._games_root / "**" / "*.parquet")
        where_sql = "WHERE playable = TRUE AND run_id = ?" if run_id else "WHERE playable = TRUE"
        params: list[object] = [glob_pattern, run_id] if run_id else [glob_pattern]
        colors_parts = [
            "SELECT DISTINCT col FROM (",
            "SELECT hero_colors AS col FROM read_parquet(?, hive_partitioning = false)",
            where_sql,
            "UNION SELECT opp_colors AS col FROM read_parquet(?, hive_partitioning = false)",
            where_sql,
            ") WHERE col != '' ORDER BY length(col) ASC, col ASC",
        ]
        sets_parts = [
            "SELECT DISTINCT set_code FROM read_parquet(?, hive_partitioning = false)",
            where_sql,
            "AND set_code != '' ORDER BY set_code ASC",
        ]
        results_parts = [
            "SELECT DISTINCT result FROM read_parquet(?, hive_partitioning = false)",
            where_sql,
            "ORDER BY result ASC",
        ]
        cards_parts = [
            "SELECT DISTINCT unnest(cards_seen) AS card FROM read_parquet(?, hive_partitioning = false)",
            where_sql,
            "ORDER BY card ASC",
        ]
        counts_where = "WHERE run_id = ?" if run_id else ""
        counts_parts = [
            "SELECT SUM(CASE WHEN playable THEN 1 ELSE 0 END), SUM(CASE WHEN NOT playable THEN 1 ELSE 0 END)",
            "FROM read_parquet(?, hive_partitioning = false)",
            counts_where,
        ]
        with duckdb.connect(":memory:") as conn:
            counts = conn.execute(" ".join(counts_parts), params).fetchone()
            runs = [
                str(r[0])
                for r in conn.execute(
                    "SELECT DISTINCT run_id FROM read_parquet(?, hive_partitioning = false) "
                    "WHERE playable = TRUE ORDER BY run_id ASC",
                    [glob_pattern],
                ).fetchall()
            ]
            colors = [str(r[0]) for r in conn.execute(" ".join(colors_parts), [*params, *params]).fetchall()]
            sets = [str(r[0]) for r in conn.execute(" ".join(sets_parts), params).fetchall()]
            results = [str(r[0]) for r in conn.execute(" ".join(results_parts), params).fetchall()]
            cards = [str(r[0]) for r in conn.execute(" ".join(cards_parts), params).fetchall() if r[0] is not None]
        playable_cnt = int(counts[0] or 0) if counts else 0
        unplayable_cnt = int(counts[1] or 0) if counts else 0
        return GameFacets(
            runs=tuple(runs),
            colors=tuple(colors),
            sets=tuple(sets),
            results=tuple(results),
            cards=tuple(cards),
            total_playable=playable_cnt,
            total_unplayable=unplayable_cnt,
        )

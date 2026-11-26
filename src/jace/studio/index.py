"""DuckDB and Hive-partitioned Parquet game indexer for Jace Studio.

Incrementally scans `games/eval-iter-*.jsonl` recordings across all discovered
runs, respects mirrored paired-deal seat swaps (`context.players`), writes
atomic Parquet partitions under `<index_dir>/games/set=<SET>/...`, and answers
faceted queries in milliseconds through in-process DuckDB.
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

from jace.studio.runs import discover_runs

if TYPE_CHECKING:
    from pathlib import Path

__all__ = (
    "GameQuery",
    "GameRecordSummary",
    "StudioIndex",
)

_EVAL_FILE_RE = re.compile(r"^eval-iter-(\d+)\.jsonl$")
_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
_INDEX_SCHEMA_VERSION = "2"
_SYNC_TTL_SECONDS = 5.0

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
    "steps_desc": "steps DESC, iteration DESC, seed ASC",
    "steps_asc": "steps ASC, iteration DESC, seed ASC",
    "iter_desc": "iteration DESC, game_idx ASC",
    "turns_desc": "turns DESC, steps DESC",
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
    search: str | None = None
    sort: str = "steps_desc"
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
        "set_code": "TLA",
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
        "source_file": source_file,
    }


def _parse_jsonl_games(raw_line: str) -> list[tuple[dict[str, object], int]]:
    line = raw_line.strip()
    if not line:
        return []
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    top_seat = _resolve_hero_seat(payload.get("context"))
    games = payload.get("games")
    items = games if isinstance(games, list) else [payload]
    found: list[tuple[dict[str, object], int]] = []
    for item in items:
        if isinstance(item, dict) and "actions" in item:
            seat = _resolve_hero_seat(item.get("context")) if "context" in item else top_seat
            found.append((item, seat))
    return found


def _extract_games_from_jsonl(jsonl_path: Path, *, run_id: str, iteration: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            for item, hero_seat in _parse_jsonl_games(raw_line):
                rows.append(
                    _parse_single_game(
                        item,
                        run_id=run_id,
                        iteration=iteration,
                        game_idx=len(rows),
                        hero_seat=hero_seat,
                        source_file=str(jsonl_path),
                    )
                )
    return rows


def _partition_path(games_root: Path, *, set_code: str, run_id: str, iteration: int) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("_")[:48] or "run"
    digest = hashlib.sha1(run_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return games_root / f"set={set_code}" / f"run={slug}_{digest}" / f"iter-{iteration:04d}.parquet"


def _write_partition(rows: list[dict[str, object]], *, games_root: Path, run_id: str, iteration: int) -> None:
    if not rows:
        return
    target = _partition_path(games_root, set_code=str(rows[0]["set_code"]), run_id=run_id, iteration=iteration)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{threading.get_ident()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=_GAMES_SCHEMA), tmp)
    tmp.replace(target)


def _file_stamp(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _build_where_clause(query: GameQuery) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    equality_filters = (
        ("run_id = ?", query.run_id),
        ("set_code = ?", query.set_code.upper() if query.set_code else None),
        ("result = ?", query.result.lower() if query.result else None),
        ("steps >= ?", query.min_steps),
    )
    for sql_frag, val in equality_filters:
        if val is not None and val != "":
            clauses.append(sql_frag)
            params.append(val)
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
    def __init__(self, runs_dir: Path, *, index_dir: Path | None = None) -> None:
        self._runs_dir = runs_dir.resolve()
        self._index_dir = (index_dir or (self._runs_dir / ".studio" / "index")).resolve()
        self._games_root = self._index_dir / "games"
        self._manifest_path = self._index_dir / "manifest.json"
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

    def _sync_run_games(self, run_id: str, *, manifest: dict[str, str], seen_keys: set[str]) -> tuple[int, int]:
        games_dir = self._runs_dir / run_id / "games"
        if not games_dir.is_dir():
            return 0, 0
        files_done = 0
        games_done = 0
        for path in sorted(games_dir.iterdir()):
            match = _EVAL_FILE_RE.match(path.name)
            if match is None or not path.is_file():
                continue
            key = str(path.resolve())
            seen_keys.add(key)
            stamp = _file_stamp(path)
            if manifest.get(key) == stamp:
                continue
            iteration = int(match.group(1))
            rows = _extract_games_from_jsonl(path, run_id=run_id, iteration=iteration)
            _write_partition(rows, games_root=self._games_root, run_id=run_id, iteration=iteration)
            manifest[key] = stamp
            files_done += 1
            games_done += len(rows)
        return files_done, games_done

    def sync(self, *, force: bool = True) -> dict[str, int]:
        with self._lock:
            now = time.monotonic()
            if not force and self._last_sync_mono > 0.0 and (now - self._last_sync_mono) < _SYNC_TTL_SECONDS:
                return {"indexed_files": 0, "indexed_games": 0}
            manifest = self._load_manifest()
            seen_keys: set[str] = set()
            indexed_files = 0
            indexed_games = 0
            for summary in discover_runs(self._runs_dir):
                f_cnt, g_cnt = self._sync_run_games(summary.run_id, manifest=manifest, seen_keys=seen_keys)
                indexed_files += f_cnt
                indexed_games += g_cnt
            pruned = [key for key in manifest if key not in seen_keys]
            for key in pruned:
                del manifest[key]
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
        order_sql = _SORT_SQL.get(query.sort, _SORT_SQL["steps_desc"])
        count_parts = ["SELECT COUNT(*) FROM read_parquet(?, hive_partitioning = false)", where_sql]
        select_parts = [
            "SELECT run_id, iteration, game_idx, set_code, seed, result, winner_seat, hero_seat,",
            "steps, turns, hero_name, opp_name, hero_colors, opp_colors, cards_seen, key_cards",
            "FROM read_parquet(?, hive_partitioning = false)",
            where_sql,
            "ORDER BY",
            order_sql,
            "LIMIT ? OFFSET ?",
        ]
        count_sql = " ".join(count_parts)
        select_sql = " ".join(select_parts)
        with duckdb.connect(":memory:") as conn:
            total_row = conn.execute(count_sql, [glob_pattern, *params]).fetchone()
            total = int(total_row[0]) if total_row is not None else 0
            records = conn.execute(select_sql, [glob_pattern, *params, query.limit, query.offset]).fetchall()
        items = [
            GameRecordSummary(
                run_id=str(row[0]),
                iteration=int(row[1]),
                game_idx=int(row[2]),
                set_code=str(row[3]),
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
            )
            for row in records
        ]
        return items, total

    def available_cards(self, *, run_id: str | None = None, limit: int = 2000) -> list[str]:
        self.sync(force=False)
        if not self._games_root.is_dir() or not any(self._games_root.rglob("*.parquet")):
            return []
        glob_pattern = str(self._games_root / "**" / "*.parquet")
        where_sql = "WHERE run_id = ?" if run_id else ""
        params: list[object] = [glob_pattern, run_id] if run_id else [glob_pattern]
        cards_parts = [
            "SELECT DISTINCT unnest(cards_seen) AS card",
            "FROM read_parquet(?, hive_partitioning = false)",
            where_sql,
            "ORDER BY card ASC LIMIT ?",
        ]
        cards_sql = " ".join(cards_parts)
        with duckdb.connect(":memory:") as conn:
            rows = conn.execute(cards_sql, [*params, limit]).fetchall()
        return [str(row[0]) for row in rows if row[0] is not None]

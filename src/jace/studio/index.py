"""DuckDB and Hive-partitioned Parquet game indexer for Jace Studio.

Incrementally scans `games/eval-iter-*.jsonl` recordings across all discovered
runs, writes compact Parquet partitions under `<index_dir>/games/set=<SET>/...`,
and answers faceted queries (by run, set, card name, outcome, turn count, or
free-text search) in milliseconds through in-process DuckDB.
"""

from __future__ import annotations

import json
import re
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
_GAMES_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("iteration", pa.int32()),
        ("game_idx", pa.int32()),
        ("set_code", pa.string()),
        ("seed", pa.int64()),
        ("result", pa.string()),
        ("winner_seat", pa.int32()),
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


def _infer_set_code(cards: list[str]) -> str:
    joined = " ".join(cards)
    if any(token in joined for token in ("Zuko", "Appa", "Aang", "Sokka", "Kyoshi", "Firebender", "Airbender")):
        return "TLA"
    return "TLA"


def _extract_result(end_block: object) -> tuple[str, int, int]:
    if not isinstance(end_block, dict):
        return "draw", -1, 0
    winner = end_block.get("winner")
    turns = end_block.get("turn") or end_block.get("turns") or 0
    turns_int = int(turns) if isinstance(turns, (int, float)) else 0
    if winner == 0:
        return "win", 0, turns_int
    if winner == 1:
        return "loss", 1, turns_int
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
    source_file: str,
) -> dict[str, object]:
    hero_name, deck0 = _extract_player_deck(game.get("players"), 0)
    opp_name, deck1 = _extract_player_deck(game.get("players"), 1)
    all_cards = sorted({*deck0.keys(), *deck1.keys()})
    non_basics = [name for name in all_cards if name not in _BASIC_LANDS]
    actions = game.get("actions")
    step_count = len(actions) if isinstance(actions, list) else 0
    result, winner_seat, turns = _extract_result(game.get("end"))
    est_turns = turns if turns > 0 else max(1, step_count // 12)
    seed_raw = game.get("seed")
    seed = int(seed_raw) if isinstance(seed_raw, (int, float)) else game_idx
    return {
        "run_id": run_id,
        "iteration": iteration,
        "game_idx": game_idx,
        "set_code": _infer_set_code(all_cards),
        "seed": seed,
        "result": result,
        "winner_seat": winner_seat,
        "steps": step_count,
        "turns": est_turns,
        "hero_name": hero_name,
        "opp_name": opp_name,
        "hero_colors": _deck_colors(deck0),
        "opp_colors": _deck_colors(deck1),
        "cards_seen": all_cards,
        "key_cards": non_basics[:8],
        "source_file": source_file,
    }


def _parse_jsonl_games(raw_line: str) -> list[dict[str, object]]:
    line = raw_line.strip()
    if not line:
        return []
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    games = payload.get("games")
    items = games if isinstance(games, list) else [payload]
    return [item for item in items if isinstance(item, dict) and "actions" in item]


def _extract_games_from_jsonl(jsonl_path: Path, *, run_id: str, iteration: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            for item in _parse_jsonl_games(raw_line):
                rows.append(
                    _parse_single_game(
                        item,
                        run_id=run_id,
                        iteration=iteration,
                        game_idx=len(rows),
                        source_file=str(jsonl_path),
                    )
                )
    return rows


def _write_partition(rows: list[dict[str, object]], *, games_root: Path, run_id: str, iteration: int) -> None:
    if not rows:
        return
    set_code = str(rows[0]["set_code"])
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", run_id).strip("_") or "run"
    out_dir = games_root / f"set={set_code}" / f"run={slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_GAMES_SCHEMA)
    pq.write_table(table, out_dir / f"iter-{iteration:04d}.parquet")


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
    if query.card:
        clauses.append("contains(lower(array_to_string(cards_seen, '|')), lower(?))")
        params.append(query.card.strip())
    if query.search:
        term = query.search.strip()
        clauses.append(
            "(contains(lower(run_id), lower(?)) OR CAST(seed AS VARCHAR) = ? "
            "OR contains(lower(array_to_string(cards_seen, '|')), lower(?)))"
        )
        params.extend((term, term, term))
    return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params


class StudioIndex:
    def __init__(self, runs_dir: Path, *, index_dir: Path | None = None) -> None:
        self._runs_dir = runs_dir.resolve()
        self._index_dir = (index_dir or (self._runs_dir / ".studio" / "index")).resolve()
        self._games_root = self._index_dir / "games"
        self._manifest_path = self._index_dir / "manifest.json"

    def _load_manifest(self) -> dict[str, str]:
        if not self._manifest_path.is_file():
            return {}
        raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _sync_run_games(self, run_id: str, manifest: dict[str, str]) -> tuple[int, int]:
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

    def sync(self) -> dict[str, int]:
        manifest = self._load_manifest()
        indexed_files = 0
        indexed_games = 0
        for summary in discover_runs(self._runs_dir):
            f_cnt, g_cnt = self._sync_run_games(summary.run_id, manifest)
            indexed_files += f_cnt
            indexed_games += g_cnt
        if indexed_files > 0 or not self._manifest_path.exists():
            self._save_manifest(manifest)
        return {"indexed_files": indexed_files, "indexed_games": indexed_games}

    def query_games(self, query: GameQuery) -> tuple[list[GameRecordSummary], int]:
        self.sync()
        if not self._games_root.is_dir() or not any(self._games_root.rglob("*.parquet")):
            return [], 0
        glob_pattern = str(self._games_root / "**" / "*.parquet")
        where_sql, params = _build_where_clause(query)
        order_sql = _SORT_SQL.get(query.sort, _SORT_SQL["steps_desc"])
        count_parts = ["SELECT COUNT(*) FROM read_parquet(?, hive_partitioning = false)", where_sql]
        select_parts = [
            "SELECT run_id, iteration, game_idx, set_code, seed, result, winner_seat,",
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
                steps=int(row[7]),
                turns=int(row[8]),
                hero_name=str(row[9]),
                opp_name=str(row[10]),
                hero_colors=str(row[11]),
                opp_colors=str(row[12]),
                cards_seen=tuple(row[13] or ()),
                key_cards=tuple(row[14] or ()),
            )
            for row in records
        ]
        return items, total

    def available_cards(self, *, run_id: str | None = None, limit: int = 200) -> list[str]:
        self.sync()
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

"""DuckDB SQL query and aggregation builders over Studio Parquet game partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    from pathlib import Path

__all__ = (
    "GameFacets",
    "GameQuery",
    "GameRecordSummary",
    "available_cards_parquet",
    "facets_parquet",
    "query_games_parquet",
    "run_game_counts_parquet",
)

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


def _parquet_glob(games_root: Path) -> str | None:
    if not games_root.is_dir() or not any(games_root.rglob("*.parquet")):
        return None
    return str(games_root / "**" / "*.parquet")


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


def query_games_parquet(games_root: Path, query: GameQuery) -> tuple[list[GameRecordSummary], int]:
    glob_pattern = _parquet_glob(games_root)
    if glob_pattern is None:
        return [], 0
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


def run_game_counts_parquet(games_root: Path) -> dict[str, tuple[int, int]]:
    glob_pattern = _parquet_glob(games_root)
    if glob_pattern is None:
        return {}
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


def available_cards_parquet(
    games_root: Path,
    *,
    run_id: str | None = None,
    limit: int | None = None,
) -> list[str]:
    glob_pattern = _parquet_glob(games_root)
    if glob_pattern is None:
        return []
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


def facets_parquet(games_root: Path, *, run_id: str | None = None) -> GameFacets:
    glob_pattern = _parquet_glob(games_root)
    if glob_pattern is None:
        return GameFacets((), (), (), (), (), 0, 0)
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

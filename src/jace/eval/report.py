"""Local Arena report generator and early-run power-law forecaster.

Reads `<runs-dir>/arena/ledger.jsonl`, training `metrics.jsonl` files, optional
`<runs-dir>/arena/failures.jsonl`, and optional `<runs-dir>/puzzles/{format}.jsonl` to
write `<runs-dir>/report/report.html`, `report.md`, and `figures/*.png` strictly outside
git. Pure-PNG figure rendering lives in `jace.eval._report_charts`.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from jace.eval._report_charts import (
    render_elo_figure,
    render_forecast_figure,
    render_pareto_figure,
    render_puzzle_figure,
)
from jace.eval.arena import BradleyTerryRating, ledger_path_for, load_match_records, solve_bradley_terry
from jace.eval.puzzles import puzzles_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = (
    "EarlyForecast",
    "EarlyForecastPoint",
    "EarlyForecastSummary",
    "PuzzleSourceSummary",
    "discover_early_forecast_points",
    "fit_early_forecast",
    "generate_local_report",
)

_MIN_FORECAST_POINTS = 2
_MIN_RESIDUAL_DOF = 3
_FALLBACK_LOG_ERR_SE = 0.15
_Z_95 = 1.959963984540054
_EVAL_WR_KEYS = ("ensemble_held_out_det_win_rate", "held_out_det_win_rate", "win_rate_vs_heuristic")
_SKIP_DIR_NAMES = frozenset({"report", ".studio", "arena", "puzzles"})


@dataclass(frozen=True, slots=True)
class EarlyForecast:
    run_id: str
    points_used: int
    max_update_used: int
    c_param: float
    alpha_param: float
    target_update: int
    projected_win_rate: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True, slots=True)
class EarlyForecastPoint:
    run_id: str
    early_update: int
    early_win_rate: float
    peak_update: int
    peak_win_rate: float
    forecast: EarlyForecast | None


@dataclass(frozen=True, slots=True)
class EarlyForecastSummary:
    points: tuple[EarlyForecastPoint, ...]
    pearson_r: float | None
    spearman_rho: float | None


@dataclass(frozen=True, slots=True)
class PuzzleSourceSummary:
    source: str
    count: int
    mean_margin: float


def fit_early_forecast(
    points: Sequence[tuple[int, float]],
    *,
    max_update: int = 400,
    target_update: int = 2000,
    run_id: str = "",
) -> EarlyForecast | None:
    if target_update <= 0:
        return None
    valid = sorted(
        (int(update), float(win_rate))
        for update, win_rate in points
        if 0 < update <= max_update and 0.0 < win_rate < 1.0
    )
    if len(valid) < _MIN_FORECAST_POINTS:
        return None
    xs = np.array([math.log(u) for u, _ in valid], dtype=np.float64)
    ys = np.array([math.log(1.0 - wr) for _, wr in valid], dtype=np.float64)
    x_mean = float(np.mean(xs))
    y_mean = float(np.mean(ys))
    sxx = float(np.sum((xs - x_mean) ** 2))
    if sxx <= 0.0:
        return None
    slope = float(np.sum((xs - x_mean) * (ys - y_mean))) / sxx
    intercept = y_mean - slope * x_mean
    log_target = math.log(target_update)
    pred_log_err = intercept + slope * log_target
    if len(valid) >= _MIN_RESIDUAL_DOF:
        resid = ys - (intercept + slope * xs)
        dof = len(valid) - 2
        sigma = math.sqrt(float(np.sum(resid**2)) / dof)
        se_pred = sigma * math.sqrt(1.0 / len(valid) + ((log_target - x_mean) ** 2) / sxx)
        crit = _t_crit_95(dof)
    else:
        se_pred = _FALLBACK_LOG_ERR_SE
        crit = _Z_95
    half = crit * se_pred
    projected = max(0.0, min(1.0, 1.0 - math.exp(pred_log_err)))
    ci_low = max(0.0, min(1.0, 1.0 - math.exp(pred_log_err + half)))
    ci_high = max(0.0, min(1.0, 1.0 - math.exp(pred_log_err - half)))
    return EarlyForecast(
        run_id=run_id,
        points_used=len(valid),
        max_update_used=valid[-1][0],
        c_param=math.exp(intercept),
        alpha_param=-slope,
        target_update=target_update,
        projected_win_rate=projected,
        ci95_low=min(ci_low, projected),
        ci95_high=max(ci_high, projected),
    )


_MAX_EXACT_T_DOF = 30
_STUDENT_T_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}


def _t_crit_95(dof: int) -> float:
    if dof in _STUDENT_T_95:
        return _STUDENT_T_95[dof]
    if dof < _MAX_EXACT_T_DOF:
        nearest = max(k for k in _STUDENT_T_95 if k <= dof)
        return _STUDENT_T_95[nearest]
    return _Z_95 + 2.4 / dof


def _extract_eval_win_rate(row: dict[str, object]) -> float | None:
    for key in _EVAL_WR_KEYS:
        val = row.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _read_eval_series(metrics_path: Path) -> list[tuple[int, float]]:
    by_update: dict[int, float] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("kind") != "eval":
                continue
            update = row.get("update")
            win_rate = _extract_eval_win_rate(row)
            if isinstance(update, int) and win_rate is not None:
                by_update[update] = win_rate
    return sorted(by_update.items())


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float | None:
    if xs.size < _MIN_FORECAST_POINTS:
        return None
    dx = xs - np.mean(xs)
    dy = ys - np.mean(ys)
    denom = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    if denom <= 0.0:
        return None
    return float(np.sum(dx * dy) / denom)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    sorted_vals = values[order]
    n = sorted_vals.size
    idx = 0
    while idx < n:
        end = idx + 1
        while end < n and sorted_vals[end] == sorted_vals[idx]:
            end += 1
        avg_rank = 0.5 * (idx + end - 1) + 1.0
        ranks[order[idx:end]] = avg_rank
        idx = end
    return ranks


def _spearman(xs: np.ndarray, ys: np.ndarray) -> float | None:
    if xs.size < _MIN_FORECAST_POINTS:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def discover_early_forecast_points(
    runs_dir: Path,
    *,
    max_early_update: int = 200,
    min_late_update: int = 400,
) -> EarlyForecastSummary:
    if not runs_dir.exists():
        return EarlyForecastSummary(points=(), pearson_r=None, spearman_rho=None)
    found: list[EarlyForecastPoint] = []
    for metrics_path in sorted(runs_dir.rglob("metrics.jsonl")):
        rel_parts = metrics_path.relative_to(runs_dir).parts
        if any(part in _SKIP_DIR_NAMES for part in rel_parts):
            continue
        series = _read_eval_series(metrics_path)
        early_pts = [(u, wr) for u, wr in series if 0 < u <= max_early_update]
        late_pts = [(u, wr) for u, wr in series if u >= min_late_update]
        if not early_pts or not late_pts:
            continue
        run_id = str(metrics_path.parent.relative_to(runs_dir))
        early_u, early_wr = early_pts[-1]
        peak_u, peak_wr = max(late_pts, key=lambda item: item[1])
        forecast = fit_early_forecast(series, max_update=400, target_update=2000, run_id=run_id)
        found.append(
            EarlyForecastPoint(
                run_id=run_id,
                early_update=early_u,
                early_win_rate=early_wr,
                peak_update=peak_u,
                peak_win_rate=peak_wr,
                forecast=forecast,
            )
        )
    xs = np.array([item.early_win_rate for item in found], dtype=np.float64)
    ys = np.array([item.peak_win_rate for item in found], dtype=np.float64)
    return EarlyForecastSummary(
        points=tuple(found),
        pearson_r=_pearson(xs, ys),
        spearman_rho=_spearman(xs, ys),
    )


def _load_puzzle_summary(runs_dir: Path, *, format_id: str) -> list[PuzzleSourceSummary]:
    puzzle_path = puzzles_path(runs_dir, format_id)
    if not puzzle_path.exists():
        return []
    grouped: dict[str, list[float]] = {}
    with puzzle_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            source = str(row.get("source", "unknown"))
            margin = float(row.get("margin", 0.0))
            grouped.setdefault(source, []).append(margin)
    return [
        PuzzleSourceSummary(source=src, count=len(margins), mean_margin=sum(margins) / len(margins))
        for src, margins in sorted(grouped.items())
    ]


def _load_failures(runs_dir: Path) -> list[dict[str, object]]:
    failures_path = runs_dir / "arena" / "failures.jsonl"
    if not failures_path.exists():
        return []
    entries: list[dict[str, object]] = []
    with failures_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if isinstance(row, dict):
                entries.append(row)
    return entries


def _resolve_report_format(runs_dir: Path, *, format_id: str | None) -> str:
    if format_id is not None:
        return format_id
    all_records = load_match_records(ledger_path_for(runs_dir))
    if not all_records:
        return "default"
    return all_records[-1].format_id


def _md_leaderboard(ratings: Sequence[BradleyTerryRating]) -> list[str]:
    if not ratings:
        return []
    rows = [
        "## Bradley-Terry Leaderboard",
        "",
        "| Rank | Player | Elo | 95% CI | Games | vs Anchor | ms/decision |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(ratings, start=1):
        half = 0.5 * (item.ci95_high - item.ci95_low)
        vs_a = f"{item.vs_anchor_wr:.1%}" if item.vs_anchor_wr is not None else "-"
        ms_s = f"{item.mean_ms_per_decision:.2f}" if item.mean_ms_per_decision is not None else "-"
        rows.append(f"| {rank} | `{item.player_id}` | {item.elo:.1f} | ±{half:.1f} | {item.games} | {vs_a} | {ms_s} |")
    rows.append("")
    return rows


def _md_forecast(summary: EarlyForecastSummary) -> list[str]:
    if not summary.points:
        return []
    r_str = f"{summary.pearson_r:.3f}" if summary.pearson_r is not None else "n/a"
    rho_str = f"{summary.spearman_rho:.3f}" if summary.spearman_rho is not None else "n/a"
    rows = [
        f"## Early-Run Forecast (`u <= 200` vs Peak, r={r_str}, rho={rho_str})",
        "",
        "| Run | Early Update | Early WR | Peak Update | Peak WR | Power-Law u2000 Forecast |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pt in summary.points:
        proj = f"{pt.forecast.projected_win_rate:.1%}" if pt.forecast is not None else "-"
        row = (
            f"| `{pt.run_id}` | {pt.early_update} | {pt.early_win_rate:.1%} | "
            f"{pt.peak_update} | {pt.peak_win_rate:.1%} | {proj} |"
        )
        rows.append(row)
    rows.append("")
    return rows


def _md_puzzles(puzzles: Sequence[PuzzleSourceSummary]) -> list[str]:
    if not puzzles:
        return []
    rows = ["## Verified Puzzles", "", "| Source | Count | Mean Margin |", "| --- | ---: | ---: |"]
    rows.extend(f"| `{item.source}` | {item.count} | {item.mean_margin:.3f} |" for item in puzzles)
    rows.append("")
    return rows


def _md_figures(figures: Sequence[tuple[str, str]]) -> list[str]:
    if not figures:
        return []
    rows = ["## Figures", ""]
    for title, rel_path in figures:
        rows.extend((f"### {title}", f"![{title}]({rel_path})", ""))
    return rows


def _build_markdown(
    *,
    format_id: str,
    ratings: Sequence[BradleyTerryRating],
    forecast_summary: EarlyForecastSummary,
    puzzles: Sequence[PuzzleSourceSummary],
    failures: Sequence[dict[str, object]],
    figures: Sequence[tuple[str, str]],
) -> str:
    lines = [f"# Jace Local Arena Report ({format_id})", ""]
    lines.extend(_md_leaderboard(ratings))
    lines.extend(_md_forecast(forecast_summary))
    lines.extend(_md_puzzles(puzzles))
    if failures:
        lines.extend((f"## Failure Log ({len(failures)} recorded)", ""))
    lines.extend(_md_figures(figures))
    return "\n".join(lines).rstrip() + "\n"


def _html_leaderboard_rows(ratings: Sequence[BradleyTerryRating]) -> str:
    rows: list[str] = []
    for rank, item in enumerate(ratings, start=1):
        half = 0.5 * (item.ci95_high - item.ci95_low)
        vs_a = f"{item.vs_anchor_wr:.1%}" if item.vs_anchor_wr is not None else "-"
        ms_s = f"{item.mean_ms_per_decision:.2f}" if item.mean_ms_per_decision is not None else "-"
        rows.append(
            f"<tr><td>{rank}</td><td><code>{html.escape(item.player_id)}</code></td>"
            f"<td>{item.elo:.1f}</td><td>±{half:.1f}</td><td>{item.games}</td>"
            f"<td>{vs_a}</td><td>{ms_s}</td></tr>"
        )
    return "".join(rows)


def _html_forecast_section(summary: EarlyForecastSummary) -> str:
    if not summary.points:
        return ""
    r_str = f"{summary.pearson_r:.3f}" if summary.pearson_r is not None else "n/a"
    rho_str = f"{summary.spearman_rho:.3f}" if summary.spearman_rho is not None else "n/a"
    rows = "".join(
        f"<tr><td><code>{html.escape(pt.run_id)}</code></td><td>{pt.early_update}</td>"
        f"<td>{pt.early_win_rate:.1%}</td><td>{pt.peak_update}</td><td>{pt.peak_win_rate:.1%}</td>"
        f"<td>{f'{pt.forecast.projected_win_rate:.1%}' if pt.forecast is not None else '-'}</td></tr>"
        for pt in summary.points
    )
    return (
        f"<h2>Early-Run Forecast (r={r_str}, rho={rho_str})</h2>"
        "<table><thead><tr><th>Run</th><th>Early u</th><th>Early WR</th><th>Peak u</th><th>Peak WR</th>"
        f"<th>u2000 Forecast</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _html_puzzle_section(puzzles: Sequence[PuzzleSourceSummary]) -> str:
    if not puzzles:
        return ""
    rows = "".join(
        f"<tr><td><code>{html.escape(item.source)}</code></td><td>{item.count}</td><td>{item.mean_margin:.3f}</td></tr>"
        for item in puzzles
    )
    return (
        "<h2>Verified Puzzles</h2>"
        "<table><thead><tr><th>Source</th><th>Count</th><th>Mean Margin</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _build_html(
    *,
    format_id: str,
    ratings: Sequence[BradleyTerryRating],
    forecast_summary: EarlyForecastSummary,
    puzzles: Sequence[PuzzleSourceSummary],
    failures: Sequence[dict[str, object]],
    figures: Sequence[tuple[str, str]],
) -> str:
    lb_html = ""
    if ratings:
        lb_html = (
            "<h2>Bradley-Terry Leaderboard</h2>"
            "<table><thead><tr><th>Rank</th><th>Player</th><th>Elo</th><th>95% CI</th>"
            "<th>Games</th><th>vs Anchor</th><th>ms/dec</th></tr></thead>"
            f"<tbody>{_html_leaderboard_rows(ratings)}</tbody></table>"
        )
    fig_html = "".join(
        f"<figure><figcaption>{html.escape(title)}</figcaption>"
        f'<img src="{html.escape(rel_path)}" alt="{html.escape(title)}"/></figure>'
        for title, rel_path in figures
    )
    fail_html = f"<h2>Failures</h2><p>{len(failures)} failure records logged.</p>" if failures else ""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'/>"
        f"<title>Jace Local Arena Report ({html.escape(format_id)})</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1040px;margin:2rem auto;padding:0 1rem;"
        "background:#fcfbf8;color:#1c1917}table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{border:1px solid #d4d0c5;padding:0.45rem 0.65rem;text-align:left}th{background:#f3f0e8}"
        "figure{margin:1.5rem 0}img{max-width:100%;border:1px solid #d4d0c5}</style></head><body>"
        f"<h1>Jace Local Arena Report ({html.escape(format_id)})</h1>"
        f"{lb_html}{_html_forecast_section(forecast_summary)}{_html_puzzle_section(puzzles)}{fail_html}"
        f"{fig_html}</body></html>\n"
    )


def generate_local_report(
    runs_dir: Path,
    *,
    format_id: str | None = None,
    anchor_player: str | None = None,
) -> Path:
    resolved_format = _resolve_report_format(runs_dir, format_id=format_id)
    records = load_match_records(ledger_path_for(runs_dir), format_id=resolved_format)
    ratings = solve_bradley_terry(records, format_id=resolved_format, anchor_player=anchor_player)
    forecast_summary = discover_early_forecast_points(runs_dir)
    puzzles = _load_puzzle_summary(runs_dir, format_id=resolved_format)
    failures = _load_failures(runs_dir)

    report_dir = runs_dir / "report"
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, str]] = []
    if render_elo_figure(ratings, figures_dir / "elo_progression.png", format_id=resolved_format):
        figures.append(("Elo Progression", "figures/elo_progression.png"))
    if render_pareto_figure(ratings, figures_dir / "elo_vs_ms.png", format_id=resolved_format):
        figures.append(("Elo vs ms/decision Frontier", "figures/elo_vs_ms.png"))
    if render_forecast_figure(forecast_summary, figures_dir / "early_forecast.png"):
        figures.append(("Early-Run Forecast", "figures/early_forecast.png"))
    if render_puzzle_figure(puzzles, figures_dir / "puzzle_summary.png", format_id=resolved_format):
        figures.append(("Verified Puzzle Summary", "figures/puzzle_summary.png"))

    md_text = _build_markdown(
        format_id=resolved_format,
        ratings=ratings,
        forecast_summary=forecast_summary,
        puzzles=puzzles,
        failures=failures,
        figures=figures,
    )
    html_text = _build_html(
        format_id=resolved_format,
        ratings=ratings,
        forecast_summary=forecast_summary,
        puzzles=puzzles,
        failures=failures,
        figures=figures,
    )
    (report_dir / "report.md").write_text(md_text, encoding="utf-8")
    html_path = report_dir / "report.html"
    html_path.write_text(html_text, encoding="utf-8")
    return html_path

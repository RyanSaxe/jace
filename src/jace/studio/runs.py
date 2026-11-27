"""Training run discovery and live metrics parsing for Jace Studio.

Scans a runs root directory for Jace training runs containing `metrics.jsonl`,
reads run configurations, dynamically discovers numeric train and eval metrics,
and enumerates saved `.safetensors` checkpoints and `games/eval-iter-*.jsonl`
recordings.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = (
    "CheckpointInfo",
    "GlobalCheckpointOption",
    "RunDetail",
    "RunSummary",
    "discover_runs",
    "list_all_checkpoints",
    "load_run_detail",
    "resolve_run_dir",
)

_EVAL_FILE_RE = re.compile(r"^eval-iter-(\d+)\.jsonl$")
_CKPT_ITER_RE = re.compile(r"iter-(\d+)")
_MIN_NESTED_PARTS = 2
_RUNNING_STALE_SECONDS = 1800.0
_MAX_TAG_STR_LEN = 64
_INDEX_METRIC_KEYS = frozenset({"update", "iteration", "kind", "step", "timestamp"})


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    iteration: int
    name: str
    path: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class GlobalCheckpointOption:
    ref: str
    run_id: str
    run_name: str
    experiment: str
    checkpoint: str
    iteration: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    experiment: str
    name: str
    status: str
    latest_update: int
    total_updates: int
    latest_tps: float
    latest_live_win: float | None
    latest_ema_win: float | None
    latest_ensemble_win: float | None
    peak_live_win: float | None
    peak_ema_win: float | None
    hidden_dim: int | None
    attention_layers: int | None
    eval_files: int
    checkpoints: tuple[int, ...]
    updated_at: float
    peak_ensemble_win: float | None = None
    playable_games: int = 0
    legacy_games: int = 0
    latest_metrics: dict[str, float] = field(default_factory=dict)
    best_metrics: dict[str, float] = field(default_factory=dict)
    eval_metric_keys: tuple[str, ...] = ()
    train_metric_keys: tuple[str, ...] = ()
    config_tags: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunDetail:
    summary: RunSummary
    config: dict[str, object]
    eval_points: tuple[dict[str, object], ...]
    train_points: tuple[dict[str, object], ...]
    checkpoints: tuple[CheckpointInfo, ...]
    eval_iterations: tuple[int, ...]
    eval_metric_keys: tuple[str, ...] = ()
    train_metric_keys: tuple[str, ...] = ()


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return None


def _safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return default


def _read_config(run_dir: Path) -> dict[str, object]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _is_compact_config_tag(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped) and len(stripped) <= _MAX_TAG_STR_LEN and not stripped.startswith("/")
    return False


def _parse_metric_line(raw_line: str) -> dict[str, object] | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        return None
    return item if isinstance(item, dict) else None


def _read_metrics(metrics_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    try:
        handle = metrics_path.open("r", encoding="utf-8")
    except OSError:
        return train_rows, eval_rows
    with handle:
        for raw_line in handle:
            item = _parse_metric_line(raw_line)
            if item is None:
                continue
            if item.get("kind") == "eval":
                eval_rows.append(item)
            elif item.get("kind") == "train" or "update" in item:
                train_rows.append(item)
    return train_rows, eval_rows


def _discover_metric_keys(rows: list[dict[str, object]]) -> tuple[str, ...]:
    keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key not in _INDEX_METRIC_KEYS and _safe_float(value) is not None:
                keys.add(str(key))
    return tuple(sorted(keys))


def _list_checkpoints(run_dir: Path) -> tuple[CheckpointInfo, ...]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return ()
    found: list[CheckpointInfo] = []
    for path in sorted(ckpt_dir.iterdir()):
        if not path.is_file() or path.suffix != ".safetensors":
            continue
        match = _CKPT_ITER_RE.search(path.stem)
        iteration = int(match.group(1)) if match is not None else -1
        found.append(
            CheckpointInfo(
                iteration=iteration,
                name=path.stem,
                path=str(path),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(sorted(found, key=lambda item: (item.iteration, item.name)))


def _list_eval_iterations(run_dir: Path) -> tuple[int, ...]:
    games_dir = run_dir / "games"
    if not games_dir.is_dir():
        return ()
    iters: list[int] = []
    for path in games_dir.iterdir():
        match = _EVAL_FILE_RE.match(path.name)
        if match is not None and path.is_file():
            iters.append(int(match.group(1)))
    return tuple(sorted(iters))


def _downsample_train(rows: list[dict[str, object]], *, max_points: int = 240) -> tuple[dict[str, object], ...]:
    if len(rows) <= max_points:
        return tuple(rows)
    step = len(rows) / max_points
    sampled = [rows[min(int(i * step), len(rows) - 1)] for i in range(max_points - 1)]
    sampled.append(rows[-1])
    return tuple(sampled)


def _split_run_id(rel_path: Path) -> tuple[str, str, str]:
    parts = rel_path.parts
    if not parts:
        return "root", "root", "root"
    run_id = rel_path.as_posix()
    if len(parts) >= _MIN_NESTED_PARTS:
        return run_id, parts[0], "/".join(parts[1:])
    return run_id, parts[0], parts[0]


def _peak_metric(rows: list[dict[str, object]], key: str) -> float | None:
    vals = [val for row in rows if (val := _safe_float(row.get(key))) is not None]
    return max(vals) if vals else None


def _latest_metric(rows: list[dict[str, object]], key: str) -> float | None:
    for row in reversed(rows):
        val = _safe_float(row.get(key))
        if val is not None:
            return val
    return None


def _populate_metric_maps(
    rows: list[dict[str, object]],
    *,
    keys: tuple[str, ...],
    latest: dict[str, float],
    best: dict[str, float],
) -> None:
    for key in keys:
        if key not in latest and (latest_val := _latest_metric(rows, key)) is not None:
            latest[key] = latest_val
        if key not in best and (best_val := _peak_metric(rows, key)) is not None:
            best[key] = best_val


def _infer_status(latest_update: int, total_updates: int, *, age_seconds: float) -> str:
    if total_updates > 0 and latest_update >= total_updates:
        return "completed"
    if age_seconds <= _RUNNING_STALE_SECONDS:
        return "running"
    return "stopped"


def _build_run_summary(
    run_dir: Path,
    *,
    runs_dir: Path,
    now: float,
    train_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
    config: dict[str, object],
    checkpoints: tuple[CheckpointInfo, ...],
    eval_iterations: tuple[int, ...],
) -> RunSummary:
    run_id, experiment, name = _split_run_id(run_dir.relative_to(runs_dir))
    latest_train = train_rows[-1] if train_rows else {}
    latest_eval = eval_rows[-1] if eval_rows else {}
    latest_update = max(
        _safe_int(latest_train.get("update")),
        _safe_int(latest_eval.get("update")),
    )
    total_updates = _safe_int(config.get("updates"))
    mtime = (run_dir / "metrics.jsonl").stat().st_mtime
    eval_keys = _discover_metric_keys(eval_rows)
    train_keys = _discover_metric_keys(train_rows)
    latest_metrics: dict[str, float] = {}
    best_metrics: dict[str, float] = {}
    _populate_metric_maps(eval_rows, keys=eval_keys, latest=latest_metrics, best=best_metrics)
    _populate_metric_maps(train_rows, keys=train_keys, latest=latest_metrics, best=best_metrics)
    return RunSummary(
        run_id=run_id,
        experiment=experiment,
        name=name,
        status=_infer_status(latest_update, total_updates, age_seconds=max(0.0, now - mtime)),
        latest_update=latest_update,
        total_updates=total_updates,
        latest_tps=_safe_float(latest_train.get("tps")) or 0.0,
        latest_live_win=_latest_metric(eval_rows, "live_held_out_det_win_rate"),
        latest_ema_win=_latest_metric(eval_rows, "held_out_det_win_rate"),
        latest_ensemble_win=_latest_metric(eval_rows, "ensemble_held_out_det_win_rate"),
        peak_live_win=_peak_metric(eval_rows, "live_held_out_det_win_rate"),
        peak_ema_win=_peak_metric(eval_rows, "held_out_det_win_rate"),
        peak_ensemble_win=_peak_metric(eval_rows, "ensemble_held_out_det_win_rate"),
        hidden_dim=_safe_int(config.get("hidden_dim")) or None,
        attention_layers=_safe_int(config.get("attention_layers")) or None,
        eval_files=len(eval_iterations),
        checkpoints=tuple(ckpt.iteration for ckpt in checkpoints),
        updated_at=mtime,
        latest_metrics=latest_metrics,
        best_metrics=best_metrics,
        eval_metric_keys=eval_keys,
        train_metric_keys=train_keys,
        config_tags={str(key): val for key, val in sorted(config.items()) if _is_compact_config_tag(val)},
    )


def _is_skipped_dir(rel_parts: tuple[str, ...], *, include_archive: bool) -> bool:
    if any(part.startswith(".") for part in rel_parts):
        return True
    return not include_archive and "archive" in rel_parts


def resolve_run_dir(runs_dir: Path, run_id: str) -> Path:
    base = runs_dir.resolve()
    candidate = (base / run_id).resolve()
    if not candidate.is_relative_to(base):
        msg = f"run_id escapes runs directory: {run_id}"
        raise ValueError(msg)
    if not (candidate / "metrics.jsonl").is_file():
        msg = f"run not found: {run_id}"
        raise FileNotFoundError(msg)
    return candidate


def discover_runs(
    runs_dir: Path,
    *,
    include_archive: bool = False,
    now: float | None = None,
) -> list[RunSummary]:
    if not runs_dir.is_dir():
        return []
    resolved = runs_dir.resolve()
    clock = now if now is not None else time.time()
    summaries: list[RunSummary] = []
    for metrics_path in sorted(resolved.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        rel_parts = run_dir.relative_to(resolved).parts
        if _is_skipped_dir(rel_parts, include_archive=include_archive):
            continue
        config = _read_config(run_dir)
        if not config and not include_archive:
            continue
        train_rows, eval_rows = _read_metrics(metrics_path)
        summaries.append(
            _build_run_summary(
                run_dir,
                runs_dir=resolved,
                now=clock,
                train_rows=train_rows,
                eval_rows=eval_rows,
                config=config,
                checkpoints=_list_checkpoints(run_dir),
                eval_iterations=_list_eval_iterations(run_dir),
            )
        )
    return sorted(summaries, key=lambda item: item.updated_at, reverse=True)


def list_all_checkpoints(
    runs_dir: Path,
    *,
    include_archive: bool = False,
) -> list[GlobalCheckpointOption]:
    resolved = runs_dir.resolve()
    options: list[GlobalCheckpointOption] = []
    for summary in discover_runs(resolved, include_archive=include_archive):
        run_dir = resolve_run_dir(resolved, summary.run_id)
        ckpts = sorted(
            _list_checkpoints(run_dir),
            key=lambda item: (item.iteration, item.name),
            reverse=True,
        )
        options.extend(
            GlobalCheckpointOption(
                ref=f"{summary.run_id}::{ckpt.name}",
                run_id=summary.run_id,
                run_name=summary.name,
                experiment=summary.experiment,
                checkpoint=ckpt.name,
                iteration=ckpt.iteration,
                size_bytes=ckpt.size_bytes,
            )
            for ckpt in ckpts
        )
    return options


def load_run_detail(
    runs_dir: Path,
    run_id: str,
    *,
    max_train_points: int = 240,
    now: float | None = None,
) -> RunDetail:
    resolved = runs_dir.resolve()
    run_dir = resolve_run_dir(resolved, run_id)
    config = _read_config(run_dir)
    train_rows, eval_rows = _read_metrics(run_dir / "metrics.jsonl")
    checkpoints = _list_checkpoints(run_dir)
    eval_iterations = _list_eval_iterations(run_dir)
    clock = now if now is not None else time.time()
    summary = _build_run_summary(
        run_dir,
        runs_dir=resolved,
        now=clock,
        train_rows=train_rows,
        eval_rows=eval_rows,
        config=config,
        checkpoints=checkpoints,
        eval_iterations=eval_iterations,
    )
    return RunDetail(
        summary=summary,
        config=config,
        eval_points=tuple(eval_rows),
        train_points=_downsample_train(train_rows, max_points=max_train_points),
        checkpoints=checkpoints,
        eval_iterations=eval_iterations,
        eval_metric_keys=summary.eval_metric_keys,
        train_metric_keys=summary.train_metric_keys,
    )

"""Append-only Arena head-to-head match ledger and Bradley-Terry Elo ratings.

Records mirrored paired-deal matches under `<runs-dir>/arena/ledger.jsonl` with
exclusive POSIX file locking and fsync, fits shift-invariant joint Bradley-Terry MLE
ratings per format with any selectable anchor player, imports external JSONL match
summaries idempotently via `import_match_summaries`, and exposes a CLI for running
matches, printing leaderboards, importing summaries, and building local reports.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from jace.collect.decks import elemental_decks, load_human_deck_catalog
from jace.eval.deals import Deal, build_deals, build_human_deals, wilson_interval
from jace.eval.parallel import PairedDealResult, PlayerSpec, play_deals_parallel, summarize_paired_deals

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

type DealRunner = Callable[[PlayerSpec, PlayerSpec, Sequence[Deal]], list[PairedDealResult]]

__all__ = (
    "ArenaMatchRecord",
    "BradleyTerryRating",
    "SearchConfigSummary",
    "append_match_record",
    "import_match_summaries",
    "ledger_path_for",
    "load_match_records",
    "main",
    "paired_los",
    "player_id_for",
    "resolve_player_spec",
    "run_arena_match",
    "solve_bradley_terry",
)

_ELO_SCALE = 400.0 / math.log(10.0)
_Z_95 = 1.959963984540054
_EVEN_WIN_RATE = 0.5
_BT_RIDGE = 1e-4
_BT_MAX_ITER = 60
_BT_TOL = 1e-9
_BT_MAX_STEP = 2.0
_BT_DIFF_CLAMP = 20.0
_MIN_BOOT_SAMPLES = 2
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class SearchConfigSummary:
    worlds: int
    horizon: int | None
    belief_id: str | None = None


@dataclass(frozen=True, slots=True)
class ArenaMatchRecord:
    match_id: str
    timestamp_utc: str
    format_id: str
    model_a: str
    model_b: str
    search_config_a: SearchConfigSummary | None
    search_config_b: SearchConfigSummary | None
    deals: int
    seed: int
    wins_a: int
    losses_a: int
    draws: int
    sweeps_a: int
    sweeps_b: int
    splits: int
    win_rate_a: float
    ci95_low: float
    ci95_high: float
    los_a: float
    ms_per_decision_a: float
    ms_per_decision_b: float
    manabot_version: str
    git_sha: str


@dataclass(frozen=True, slots=True)
class BradleyTerryRating:
    player_id: str
    elo: float
    ci95_low: float
    ci95_high: float
    games: int
    matches: int
    mean_ms_per_decision: float | None
    vs_anchor_wr: float | None = None


def ledger_path_for(runs_dir: Path) -> Path:
    return runs_dir / "arena" / "ledger.jsonl"


def player_id_for(model: str, search: SearchConfigSummary | None) -> str:
    if search is None:
        return model
    horizon_part = f"h{search.horizon}" if search.horizon is not None else "full"
    belief_part = f",{search.belief_id}" if search.belief_id else ""
    return f"{model}[search:w{search.worlds},{horizon_part}{belief_part}]"


def paired_los(win_rate_a: float, paired_se: float) -> float:
    if paired_se <= 0.0:
        if win_rate_a > _EVEN_WIN_RATE:
            return 1.0
        if win_rate_a < _EVEN_WIN_RATE:
            return 0.0
        return _EVEN_WIN_RATE
    z = (win_rate_a - _EVEN_WIN_RATE) / (paired_se * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _req_int(raw: dict[str, object], key: str) -> int:
    val = raw[key]
    if isinstance(val, bool) or not isinstance(val, int):
        msg = f"expected int for {key!r}, got {type(val).__name__}"
        raise TypeError(msg)
    return val


def _req_float(raw: dict[str, object], key: str) -> float:
    val = raw[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        msg = f"expected float for {key!r}, got {type(val).__name__}"
        raise TypeError(msg)
    return float(val)


def _decode_search_config(raw: object) -> SearchConfigSummary | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        msg = f"expected dict or None for search_config, got {type(raw).__name__}"
        raise TypeError(msg)
    horizon_raw = raw.get("horizon")
    belief_raw = raw.get("belief_id")
    return SearchConfigSummary(
        worlds=_req_int(raw, "worlds"),
        horizon=_req_int(raw, "horizon") if horizon_raw is not None else None,
        belief_id=str(belief_raw) if belief_raw is not None else None,
    )


def _record_from_dict(raw: dict[str, object]) -> ArenaMatchRecord:
    if not isinstance(raw, dict):
        msg = f"expected JSON object for ArenaMatchRecord, got {type(raw).__name__}"
        raise TypeError(msg)
    return ArenaMatchRecord(
        match_id=str(raw["match_id"]),
        timestamp_utc=str(raw["timestamp_utc"]),
        format_id=str(raw["format_id"]),
        model_a=str(raw["model_a"]),
        model_b=str(raw["model_b"]),
        search_config_a=_decode_search_config(raw.get("search_config_a")),
        search_config_b=_decode_search_config(raw.get("search_config_b")),
        deals=_req_int(raw, "deals"),
        seed=_req_int(raw, "seed"),
        wins_a=_req_int(raw, "wins_a"),
        losses_a=_req_int(raw, "losses_a"),
        draws=_req_int(raw, "draws"),
        sweeps_a=_req_int(raw, "sweeps_a"),
        sweeps_b=_req_int(raw, "sweeps_b"),
        splits=_req_int(raw, "splits"),
        win_rate_a=_req_float(raw, "win_rate_a"),
        ci95_low=_req_float(raw, "ci95_low"),
        ci95_high=_req_float(raw, "ci95_high"),
        los_a=_req_float(raw, "los_a"),
        ms_per_decision_a=_req_float(raw, "ms_per_decision_a"),
        ms_per_decision_b=_req_float(raw, "ms_per_decision_b"),
        manabot_version=str(raw["manabot_version"]),
        git_sha=str(raw["git_sha"]),
    )


def append_match_record(
    ledger_path: Path,
    record: ArenaMatchRecord,
    *,
    dedupe_by_match_id: bool = False,
) -> bool:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), sort_keys=True) + "\n"
    with ledger_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            if dedupe_by_match_id:
                handle.seek(0)
                for raw in handle:
                    stripped = raw.strip()
                    if stripped and json.loads(stripped).get("match_id") == record.match_id:
                        return False
                handle.seek(0, os.SEEK_END)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return True


def load_match_records(ledger_path: Path, *, format_id: str | None = None) -> list[ArenaMatchRecord]:
    if not ledger_path.exists():
        return []
    records: list[ArenaMatchRecord] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = _record_from_dict(json.loads(stripped))
            if format_id is None or record.format_id == format_id:
                records.append(record)
    return records


def _resolve_anchor(
    records: Sequence[ArenaMatchRecord],
    players: Sequence[str],
    *,
    anchor_player: str | None,
) -> str:
    player_set = set(players)
    if anchor_player is not None:
        if anchor_player not in player_set:
            msg = f"anchor_player {anchor_player!r} not found in format players: {sorted(player_set)}"
            raise ValueError(msg)
        return anchor_player
    if "heuristic" in player_set:
        return "heuristic"
    for player in players:
        if player.lower() == "heuristic":
            return player
    ordered = sorted(enumerate(records), key=lambda item: (item[1].timestamp_utc, item[0]))
    return player_id_for(ordered[0][1].model_a, ordered[0][1].search_config_a)


def _bt_accumulate(
    pairs: Sequence[tuple[int, int, float, float]],
    theta: np.ndarray,
    *,
    anchor_i: int,
    free_pos: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    n_free = len(free_pos)
    grad = np.zeros(n_free, dtype=np.float64)
    hess = np.eye(n_free, dtype=np.float64) * _BT_RIDGE
    for ia, ib, score_a, total in pairs:
        if total <= 0.0:
            continue
        diff = max(-_BT_DIFF_CLAMP, min(_BT_DIFF_CLAMP, float(theta[ia] - theta[ib])))
        prob = 1.0 / (1.0 + math.exp(-diff))
        g_val = score_a - total * prob
        h_val = total * prob * (1.0 - prob)
        if ia != anchor_i:
            ka = free_pos[ia]
            grad[ka] += g_val
            hess[ka, ka] += h_val
        if ib != anchor_i:
            kb = free_pos[ib]
            grad[kb] -= g_val
            hess[kb, kb] += h_val
        if anchor_i not in {ia, ib}:
            ka, kb = free_pos[ia], free_pos[ib]
            hess[ka, kb] -= h_val
            hess[kb, ka] -= h_val
    return grad, hess


def _fit_bt_theta(
    pairs: Sequence[tuple[int, int, float, float]],
    *,
    n_players: int,
    anchor_i: int,
    init_theta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    free = [idx for idx in range(n_players) if idx != anchor_i]
    free_pos = {player_i: pos for pos, player_i in enumerate(free)}
    theta = np.zeros(n_players, dtype=np.float64) if init_theta is None else init_theta.copy()
    theta[anchor_i] = 0.0
    if not free:
        return theta, np.zeros((0, 0), dtype=np.float64)
    hess = np.eye(len(free), dtype=np.float64) * _BT_RIDGE
    for _ in range(_BT_MAX_ITER):
        grad, hess = _bt_accumulate(pairs, theta, anchor_i=anchor_i, free_pos=free_pos)
        step = np.linalg.solve(hess, grad)
        max_step = float(np.max(np.abs(step)))
        if max_step > _BT_MAX_STEP:
            step *= _BT_MAX_STEP / max_step
        for pos, player_i in enumerate(free):
            theta[player_i] += float(step[pos])
        if max_step < _BT_TOL:
            break
    _, final_hess = _bt_accumulate(pairs, theta, anchor_i=anchor_i, free_pos=free_pos)
    return theta, np.linalg.inv(final_hess)


def _resample_match_score(record: ArenaMatchRecord, rng: np.random.Generator) -> tuple[float, float]:
    paired_total = record.sweeps_a + record.sweeps_b + record.splits
    if record.deals > 0 and paired_total == record.deals:
        probs = np.array(
            [record.sweeps_a / paired_total, record.splits / paired_total, record.sweeps_b / paired_total],
            dtype=np.float64,
        )
        draw = rng.multinomial(record.deals, probs)
        score_a = 2.0 * float(draw[0]) + 1.0 * float(draw[1])
        return score_a, 2.0 * float(record.deals)
    total_games = record.wins_a + record.losses_a + record.draws
    if total_games <= 0:
        return 0.0, 0.0
    probs = np.array(
        [record.wins_a / total_games, record.draws / total_games, record.losses_a / total_games],
        dtype=np.float64,
    )
    draw = rng.multinomial(total_games, probs)
    return float(draw[0]) + 0.5 * float(draw[1]), float(total_games)


def _bootstrap_se_elo(
    records: Sequence[ArenaMatchRecord],
    *,
    idx_map: dict[str, int],
    anchor_i: int,
    theta_hat: np.ndarray,
    bootstrap_samples: int,
) -> np.ndarray | None:
    if bootstrap_samples < _MIN_BOOT_SAMPLES or len(idx_map) <= 1:
        return None
    rng = np.random.default_rng(0)
    n_players = len(idx_map)
    samples = np.zeros((bootstrap_samples, n_players), dtype=np.float64)
    endpoints = [
        (
            idx_map[player_id_for(rec.model_a, rec.search_config_a)],
            idx_map[player_id_for(rec.model_b, rec.search_config_b)],
            rec,
        )
        for rec in records
    ]
    for sample_idx in range(bootstrap_samples):
        boot_pairs = [(ia, ib, *_resample_match_score(rec, rng)) for ia, ib, rec in endpoints]
        theta_b, _ = _fit_bt_theta(boot_pairs, n_players=n_players, anchor_i=anchor_i, init_theta=theta_hat)
        samples[sample_idx] = theta_b * _ELO_SCALE
    return np.std(samples, axis=0, ddof=1)


@dataclass(slots=True)
class _PlayerStats:
    games: int = 0
    matches: int = 0
    ms_weighted: float = 0.0
    ms_games: int = 0
    vs_anchor_score: float = 0.0
    vs_anchor_games: int = 0


def _collect_player_stats(
    records: Sequence[ArenaMatchRecord],
    *,
    anchor: str,
) -> dict[str, _PlayerStats]:
    stats: dict[str, _PlayerStats] = {}
    for rec in records:
        pid_a = player_id_for(rec.model_a, rec.search_config_a)
        pid_b = player_id_for(rec.model_b, rec.search_config_b)
        total = rec.wins_a + rec.losses_a + rec.draws
        score_a = rec.wins_a + 0.5 * rec.draws
        score_b = rec.losses_a + 0.5 * rec.draws
        for pid, opp, score, ms_val in (
            (pid_a, pid_b, score_a, rec.ms_per_decision_a),
            (pid_b, pid_a, score_b, rec.ms_per_decision_b),
        ):
            entry = stats.setdefault(pid, _PlayerStats())
            entry.games += total
            entry.matches += 1
            if ms_val > 0.0 and total > 0:
                entry.ms_weighted += ms_val * total
                entry.ms_games += total
            if opp == anchor and pid != anchor and total > 0:
                entry.vs_anchor_score += score
                entry.vs_anchor_games += total
    return stats


def _combined_elo_se(
    *,
    player_i: int,
    anchor_i: int,
    free_pos: dict[int, int],
    cov: np.ndarray,
    boot_se: np.ndarray | None,
) -> float:
    if player_i == anchor_i:
        return 0.0
    hess_se = math.sqrt(max(0.0, float(cov[free_pos[player_i], free_pos[player_i]]))) * _ELO_SCALE
    if boot_se is None:
        return hess_se
    return 0.5 * (hess_se + float(boot_se[player_i]))


def solve_bradley_terry(
    records: Sequence[ArenaMatchRecord],
    *,
    format_id: str,
    anchor_player: str | None = None,
    anchor_elo: float = 1500.0,
    bootstrap_samples: int = 400,
) -> list[BradleyTerryRating]:
    """Fit shift-invariant joint Bradley-Terry MLE ratings within `format_id`.

    Draws count as 0.5 wins for each side. When `anchor_player` is `None`, anchors on
    `"heuristic"` if present in `format_id`, otherwise the earliest chronologically
    appearing player. Combines inverse-Hessian and deal-level bootstrap standard errors
    relative to the chosen anchor so the anchor sits at `[anchor_elo, anchor_elo]`.
    """
    filtered = [rec for rec in records if rec.format_id == format_id]
    if not filtered:
        return []
    stats = _collect_player_stats(filtered, anchor="")
    players = sorted(stats)
    anchor = _resolve_anchor(filtered, players, anchor_player=anchor_player)
    stats = _collect_player_stats(filtered, anchor=anchor)
    idx_map = {pid: i for i, pid in enumerate(players)}
    anchor_i = idx_map[anchor]
    free_pos = {i: pos for pos, i in enumerate(i for i in range(len(players)) if i != anchor_i)}
    pairs = [
        (
            idx_map[player_id_for(rec.model_a, rec.search_config_a)],
            idx_map[player_id_for(rec.model_b, rec.search_config_b)],
            rec.wins_a + 0.5 * rec.draws,
            float(rec.wins_a + rec.losses_a + rec.draws),
        )
        for rec in filtered
    ]
    theta, cov = _fit_bt_theta(pairs, n_players=len(players), anchor_i=anchor_i)
    boot_se = _bootstrap_se_elo(
        filtered,
        idx_map=idx_map,
        anchor_i=anchor_i,
        theta_hat=theta,
        bootstrap_samples=bootstrap_samples,
    )
    ratings: list[BradleyTerryRating] = []
    for pid in players:
        player_i = idx_map[pid]
        entry = stats[pid]
        elo = anchor_elo + float(theta[player_i]) * _ELO_SCALE
        half = _Z_95 * _combined_elo_se(
            player_i=player_i,
            anchor_i=anchor_i,
            free_pos=free_pos,
            cov=cov,
            boot_se=boot_se,
        )
        mean_ms = entry.ms_weighted / entry.ms_games if entry.ms_games > 0 else None
        vs_anchor = entry.vs_anchor_score / entry.vs_anchor_games if entry.vs_anchor_games > 0 else None
        ratings.append(
            BradleyTerryRating(
                player_id=pid,
                elo=elo,
                ci95_low=elo - half,
                ci95_high=elo + half,
                games=entry.games,
                matches=entry.matches,
                mean_ms_per_decision=mean_ms,
                vs_anchor_wr=vs_anchor,
            )
        )
    ratings.sort(key=lambda item: (-item.elo, item.player_id))
    return ratings


def _resolve_single_path(raw: str, *, runs_dir: Path | None) -> Path:
    candidate = Path(raw)
    if candidate.exists():
        return candidate
    if runs_dir is not None and (runs_dir / raw).exists():
        return runs_dir / raw
    msg = f"checkpoint path not found: {raw}"
    raise FileNotFoundError(msg)


def resolve_player_spec(spec: str, *, runs_dir: Path | None = None) -> PlayerSpec:
    cleaned = spec.strip()
    if cleaned == "heuristic":
        return "heuristic"
    if cleaned == "random":
        return "random"
    if "," in cleaned:
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if not parts:
            msg = f"empty ensemble spec: {spec!r}"
            raise ValueError(msg)
        return tuple(_resolve_single_path(part, runs_dir=runs_dir) for part in parts)
    return _resolve_single_path(cleaned, runs_dir=runs_dir)


def _resolve_arena_decks_path(runs_dir: Path, decks_path: Path | None) -> Path | None:
    if decks_path is not None:
        return decks_path
    default_decks = runs_dir / "data" / "decks.pt"
    return default_decks if default_decks.exists() else None


def _build_arena_deals(runs_dir: Path, *, deals: int, seed: int, decks_path: Path | None) -> list[Deal]:
    resolved_decks = _resolve_arena_decks_path(runs_dir, decks_path)
    if resolved_decks is not None:
        catalog = load_human_deck_catalog(resolved_decks)
        return build_human_deals(catalog, split="held_out", count=deals, seed=seed, decklists="closed")
    return build_deals(elemental_decks(), count=deals, seed=seed, decklists="closed")


def _manabot_version() -> str:
    try:
        return importlib.metadata.version("manabot")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _read_git_head(git_dir: Path) -> str:
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return "unknown"
    head_text = head_path.read_text(encoding="utf-8").strip()
    if head_text.startswith("ref: "):
        ref_path = git_dir / head_text.removeprefix("ref: ").strip()
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()[:12]
        return "unknown"
    return head_text[:12] if head_text else "unknown"


def _git_sha(start: Path | None = None) -> str:
    current = (start or Path(__file__)).resolve()
    for parent in (current, *current.parents):
        dot_git = parent / ".git"
        if dot_git.is_dir():
            return _read_git_head(dot_git)
        if dot_git.is_file():
            content = dot_git.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir: "):
                target = Path(content.removeprefix("gitdir: ").strip())
                git_dir = target if target.is_absolute() else (parent / target)
                return _read_git_head(git_dir)
    return "unknown"


def run_arena_match(
    runs_dir: Path,
    *,
    format_id: str,
    model_a: str,
    model_b: str,
    deals: int = 200,
    seed: int = 9901,
    num_workers: int = 8,
    decks_path: Path | None = None,
    deterministic: bool = True,
    max_actions: int = 1200,
    search_config_a: SearchConfigSummary | None = None,
    search_config_b: SearchConfigSummary | None = None,
    model_a_spec: PlayerSpec | None = None,
    model_b_spec: PlayerSpec | None = None,
    deal_runner: DealRunner | None = None,
) -> ArenaMatchRecord:
    spec_a = model_a_spec if model_a_spec is not None else resolve_player_spec(model_a, runs_dir=runs_dir)
    spec_b = model_b_spec if model_b_spec is not None else resolve_player_spec(model_b, runs_dir=runs_dir)
    deal_list = _build_arena_deals(runs_dir, deals=deals, seed=seed, decks_path=decks_path)
    if deal_runner is not None:
        deal_results = deal_runner(spec_a, spec_b, deal_list)
    elif search_config_a is not None or search_config_b is not None:
        msg = (
            "search-enabled Arena matches require a deal_runner; "
            "use jace.forks.arena.run_search_arena_match or `python -m jace.forks.arena match`"
        )
        raise ValueError(msg)
    else:
        deal_results = play_deals_parallel(
            spec_a,
            spec_b,
            deals=deal_list,
            num_workers=num_workers,
            seed=seed,
            deterministic=deterministic,
            max_actions=max_actions,
        )
    summary = summarize_paired_deals(deal_results)
    total_dec_a = sum(item.decisions_a for item in deal_results)
    total_dec_b = sum(item.decisions_b for item in deal_results)
    ms_a = sum(item.decision_ms_a for item in deal_results) / total_dec_a if total_dec_a > 0 else 0.0
    ms_b = sum(item.decision_ms_b for item in deal_results) / total_dec_b if total_dec_b > 0 else 0.0
    now = datetime.now(UTC)
    timestamp_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    slug_a = _SLUG_RE.sub("-", player_id_for(model_a, search_config_a).lower()).strip("-") or "player"
    slug_b = _SLUG_RE.sub("-", player_id_for(model_b, search_config_b).lower()).strip("-") or "player"
    match_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{slug_a}-vs-{slug_b}-s{seed}-d{deals}"
    record = ArenaMatchRecord(
        match_id=match_id,
        timestamp_utc=timestamp_utc,
        format_id=format_id,
        model_a=model_a,
        model_b=model_b,
        search_config_a=search_config_a,
        search_config_b=search_config_b,
        deals=summary.deals,
        seed=seed,
        wins_a=summary.wins,
        losses_a=summary.losses,
        draws=summary.draws,
        sweeps_a=summary.deal_sweeps_won,
        sweeps_b=summary.deal_sweeps_lost,
        splits=summary.deal_splits,
        win_rate_a=summary.win_rate,
        ci95_low=summary.ci_low,
        ci95_high=summary.ci_high,
        los_a=paired_los(summary.win_rate, summary.paired_se),
        ms_per_decision_a=ms_a,
        ms_per_decision_b=ms_b,
        manabot_version=_manabot_version(),
        git_sha=_git_sha(),
    )
    append_match_record(ledger_path_for(runs_dir), record)
    return record


def _record_from_summary_row(index: int, raw: dict[str, object], *, format_id: str) -> ArenaMatchRecord:
    model_a, model_b = str(raw["model_a"]), str(raw["model_b"])
    wins_a, losses_a = _req_int(raw, "wins_a"), _req_int(raw, "losses_a")
    draws = _req_int(raw, "draws") if "draws" in raw else 0
    total_games = wins_a + losses_a + draws
    if total_games <= 0:
        msg = f"match summary row {index} must have positive total games"
        raise ValueError(msg)
    score_a = wins_a + 0.5 * draws
    win_rate = score_a / total_games
    ci_low, ci_high = wilson_interval(score_a, total_games)
    binom_se = math.sqrt(max(0.0, win_rate * (1.0 - win_rate)) / total_games)
    slug_fmt = _SLUG_RE.sub("-", format_id.lower()).strip("-") or "format"
    slug_a = _SLUG_RE.sub("-", model_a.lower()).strip("-") or "player"
    slug_b = _SLUG_RE.sub("-", model_b.lower()).strip("-") or "player"
    match_id = str(raw.get("match_id") or f"import-{slug_fmt}-{index:03d}-{slug_a}-vs-{slug_b}")
    timestamp = str(raw.get("timestamp_utc") or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    deals = _req_int(raw, "deals") if "deals" in raw else max(1, total_games // 2)
    seed = _req_int(raw, "seed") if "seed" in raw else 0
    return ArenaMatchRecord(
        match_id=match_id,
        timestamp_utc=timestamp,
        format_id=str(raw.get("format_id") or format_id),
        model_a=model_a,
        model_b=model_b,
        search_config_a=_decode_search_config(raw.get("search_config_a")),
        search_config_b=_decode_search_config(raw.get("search_config_b")),
        deals=deals,
        seed=seed,
        wins_a=wins_a,
        losses_a=losses_a,
        draws=draws,
        sweeps_a=_req_int(raw, "sweeps_a") if "sweeps_a" in raw else 0,
        sweeps_b=_req_int(raw, "sweeps_b") if "sweeps_b" in raw else 0,
        splits=_req_int(raw, "splits") if "splits" in raw else 0,
        win_rate_a=win_rate,
        ci95_low=ci_low,
        ci95_high=ci_high,
        los_a=paired_los(win_rate, binom_se),
        ms_per_decision_a=_req_float(raw, "ms_per_decision_a") if "ms_per_decision_a" in raw else 0.0,
        ms_per_decision_b=_req_float(raw, "ms_per_decision_b") if "ms_per_decision_b" in raw else 0.0,
        manabot_version=str(raw.get("manabot_version") or _manabot_version()),
        git_sha=str(raw.get("git_sha") or _git_sha()),
    )


def import_match_summaries(runs_dir: Path, *, input_path: Path, format_id: str) -> int:
    ledger = ledger_path_for(runs_dir)
    added = 0
    for idx, line in enumerate(input_path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        raw = json.loads(stripped)
        if not isinstance(raw, dict):
            msg = f"expected JSON object on line {idx + 1} of {input_path}"
            raise TypeError(msg)
        record = _record_from_summary_row(idx, raw, format_id=format_id)
        if append_match_record(ledger, record, dedupe_by_match_id=True):
            added += 1
    return added


def _format_leaderboard_table(ratings: Sequence[BradleyTerryRating], *, format_id: str) -> str:
    header = f"Arena Leaderboard ({format_id})\n"
    columns = (
        f"{'Rank':>4}  {'Player':<26}  {'Elo':>7}  {'±95% CI':>8}  {'Games':>6}  {'vs Anchor':>9}  {'ms/dec':>7}\n"
    )
    sep = "-" * (len(columns) - 1) + "\n"
    rows = [header, columns, sep]
    for rank, item in enumerate(ratings, start=1):
        half = 0.5 * (item.ci95_high - item.ci95_low)
        vs_str = f"{item.vs_anchor_wr * 100.0:8.1f}%" if item.vs_anchor_wr is not None else "        -"
        ms_str = f"{item.mean_ms_per_decision:7.2f}" if item.mean_ms_per_decision is not None else "      -"
        rows.append(
            f"{rank:>4}  {item.player_id:<26}  {item.elo:7.1f}  ±{half:7.1f}  {item.games:6d}  {vs_str}  {ms_str}\n"
        )
    return "".join(rows)


def _cmd_match(args: argparse.Namespace) -> int:
    record = run_arena_match(
        Path(args.runs_dir),
        format_id=args.format_id,
        model_a=args.model_a,
        model_b=args.model_b,
        deals=args.deals,
        seed=args.seed,
        num_workers=args.num_workers,
        decks_path=Path(args.decks_path) if args.decks_path else None,
    )
    sys.stdout.write(json.dumps(asdict(record), indent=2) + "\n")
    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    records = load_match_records(ledger_path_for(Path(args.runs_dir)), format_id=args.format_id)
    ratings = solve_bradley_terry(
        records,
        format_id=args.format_id,
        anchor_player=args.anchor,
        anchor_elo=args.anchor_elo,
        bootstrap_samples=args.bootstrap_samples,
    )
    if args.as_json:
        sys.stdout.write(json.dumps([asdict(item) for item in ratings], indent=2) + "\n")
    else:
        sys.stdout.write(_format_leaderboard_table(ratings, format_id=args.format_id))
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir)
    added = import_match_summaries(runs_dir, input_path=Path(args.input_path), format_id=args.format_id)
    total = len(load_match_records(ledger_path_for(runs_dir), format_id=args.format_id))
    sys.stdout.write(
        json.dumps({"format_id": args.format_id, "added": added, "total_in_format": total}, indent=2) + "\n"
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    # Lazy import breaks the cycle because report.py imports arena ledger helpers.
    from jace.eval.report import generate_local_report  # noqa: PLC0415 -- break arena<->report CLI import cycle

    out_path = generate_local_report(
        Path(args.runs_dir),
        format_id=args.format_id,
        anchor_player=args.anchor,
    )
    sys.stdout.write(json.dumps({"report_html": str(out_path)}, indent=2) + "\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jace.eval.arena", description="Arena match ledger and Elo ratings")
    sub = parser.add_subparsers(dest="command", required=True)

    p_match = sub.add_parser("match", help="Run mirrored deals and append to ledger.jsonl")
    p_match.add_argument("--runs-dir", required=True)
    p_match.add_argument("--format", dest="format_id", required=True)
    p_match.add_argument("--model-a", required=True)
    p_match.add_argument("--model-b", required=True)
    p_match.add_argument("--deals", type=int, default=200)
    p_match.add_argument("--seed", type=int, default=9901)
    p_match.add_argument("--num-workers", type=int, default=8)
    p_match.add_argument("--decks-path", default=None)

    p_lb = sub.add_parser("leaderboard", help="Compute joint Bradley-Terry ratings for a format")
    p_lb.add_argument("--runs-dir", required=True)
    p_lb.add_argument("--format", dest="format_id", required=True)
    p_lb.add_argument("--anchor", default=None)
    p_lb.add_argument("--anchor-elo", type=float, default=1500.0)
    p_lb.add_argument("--bootstrap-samples", type=int, default=400)
    p_lb.add_argument("--json", dest="as_json", action="store_true")

    p_imp = sub.add_parser("import", help="Idempotently import external JSONL match summaries into ledger.jsonl")
    p_imp.add_argument("--runs-dir", required=True)
    p_imp.add_argument("--format", dest="format_id", required=True)
    p_imp.add_argument("--input", dest="input_path", required=True)

    p_rep = sub.add_parser("report", help="Generate <runs-dir>/report/report.html, report.md, and figures")
    p_rep.add_argument("--runs-dir", required=True)
    p_rep.add_argument("--format", dest="format_id", default=None)
    p_rep.add_argument("--anchor", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    dispatch = {
        "match": _cmd_match,
        "leaderboard": _cmd_leaderboard,
        "import": _cmd_import,
        "report": _cmd_report,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

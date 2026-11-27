"""Bradley-Terry Elo rating solver and Arena match record types.

Defines the immutable Arena match and rating dataclasses (`SearchConfigSummary`,
`ArenaMatchRecord`, `BradleyTerryRating`), canonical player-id formatting
(`player_id_for`), and the shift-invariant joint Bradley-Terry MLE solver
(`solve_bradley_terry`) with combined inverse-Hessian and deal-level bootstrap
confidence intervals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    "ArenaMatchRecord",
    "BradleyTerryRating",
    "SearchConfigSummary",
    "player_id_for",
    "solve_bradley_terry",
)

_ELO_SCALE = 400.0 / math.log(10.0)
_Z_95 = 1.959963984540054
_BT_RIDGE = 1e-4
_BT_MAX_ITER = 60
_BT_TOL = 1e-9
_BT_MAX_STEP = 2.0
_BT_DIFF_CLAMP = 20.0
_MIN_BOOT_SAMPLES = 2


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


def player_id_for(model: str, search: SearchConfigSummary | None) -> str:
    if search is None:
        return model
    horizon_part = f"h{search.horizon}" if search.horizon is not None else "full"
    belief_part = f",{search.belief_id}" if search.belief_id else ""
    return f"{model}[search:w{search.worlds},{horizon_part}{belief_part}]"


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

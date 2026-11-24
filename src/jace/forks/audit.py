"""Batched fork auxiliary loss staging and short-horizon value-leaf correlation audit.

Stages decisive root fork comparisons alongside on-policy continuation WDL value targets
(Cobbe et al., 2020) into a single auxiliary loss closure for `train_self_play`. Also
provides a paired full-game vs. intermediate-horizon correlation audit so truncated
value-leaf rollouts are empirically validated before training with them.
"""

from __future__ import annotations

import functools
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from manabot.backends.torch.arrays import on_device

from jace.forks.batch import (
    BatchedForkCollectorPool,
    BatchedForkConfig,
    BatchedForkResult,
    ContinuationValueSample,
    _ActiveBranch,
    _assemble_results,
    _batched_eval,
    _harvest_roots_from_deal,
    _HarvestedRoot,
    _hero_win_prob,
    _safe_step,
    _spawn_all_branches,
    _terminal_score,
)
from jace.forks.collect import ForkStats
from jace.rl.losses import categorical_value_loss, fork_pairwise_loss, policy_log_probs
from jace.storage.gameplay import RunLayout, append_metric

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import ExitStack
    from pathlib import Path

    from manabot.backends.torch.policy import ActorCritic, PolicyInference
    from manabot.env.observation import ObservationSpace

    from jace.collect.agent import DecisionMemoryKind
    from jace.eval.deals import Deal
    from jace.rl.loop import PrepareAuxFn

__all__ = (
    "BatchedForkSpec",
    "HorizonAuditSummary",
    "add_fork_cli_args",
    "audit_horizon_correlation",
    "build_batched_fork_aux_loss",
    "make_batched_fork_prepare_aux",
    "open_batched_fork_hook",
    "parse_fork_kinds",
)

_DECISIVE_GAP = 0.25
_MIN_CORR_SAMPLES = 2
_MIN_STD = 1e-9
_WDL_CLASSES = 3


@dataclass(frozen=True, slots=True)
class HorizonAuditSummary:
    horizon: int
    total_forks: int
    decisive_full: int
    pearson_r: float
    sign_agreement: float
    mean_abs_value_gap: float
    mean_abs_full_gap: float


def _stage_value_reuse(
    agent: ActorCritic, *, samples: Sequence[ContinuationValueSample], weight: float, device: torch.device
) -> Callable[[], torch.Tensor] | None:
    if not samples or weight <= 0.0:
        return None
    stacked = {key: np.stack([item.encoded_obs[key] for item in samples], axis=0) for key in samples[0].encoded_obs}
    prepared = on_device(agent.prepare(stacked), device)
    targets = torch.zeros((len(samples), _WDL_CLASSES), dtype=torch.float32, device=device)
    for idx, sample in enumerate(samples):
        targets[idx, sample.outcome_class] = 1.0

    def _value_term() -> torch.Tensor:
        _, raw_value = agent.forward_distribution(prepared)
        return weight * categorical_value_loss(raw_value, targets).mean()

    return _value_term


def build_batched_fork_aux_loss(
    agent: ActorCritic,
    *,
    result: BatchedForkResult,
    config: BatchedForkConfig,
    device: torch.device,
    collect_s: float = 0.0,
) -> tuple[Callable[[], torch.Tensor | None] | None, ForkStats]:
    """Build a joint auxiliary closure combining pairwise fork loss and value-head reuse.

    Stages decisive root fork comparisons (anchored against pre-update log-probabilities)
    together with on-policy continuation WDL value targets on `device`, returning a single
    zero-argument closure for `train_self_play`'s `aux_loss` hook plus `ForkStats`.
    """
    t0 = time.perf_counter()
    examples = result.examples
    total = len(examples)
    agent.to(device)
    value_term = _stage_value_reuse(
        agent, samples=result.value_samples, weight=config.value_reuse_weight, device=device
    )
    if total == 0:
        stats = ForkStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, collect_s, time.perf_counter() - t0)
        return value_term, stats

    mean_top_p = sum(item.policy_pair_probs[0] for item in examples) / total
    mean_alt_p = sum(item.policy_pair_probs[1] for item in examples) / total
    decisive = [item for item in examples if abs(item.pair_returns[0] - item.pair_returns[1]) >= config.min_gap]
    alt_better = sum(1 for item in decisive if item.pair_returns[1] > item.pair_returns[0])
    if not decisive:
        stats = ForkStats(total, 0, 0, 0.0, mean_top_p, mean_alt_p, 0.0, 0.0, 0.0, collect_s, time.perf_counter() - t0)
        return value_term, stats

    stacked = {key: np.stack([item.encoded_obs[key] for item in decisive], axis=0) for key in decisive[0].encoded_obs}
    action_pairs = torch.tensor([item.action_pair for item in decisive], dtype=torch.long, device=device)
    pair_returns = torch.tensor([item.pair_returns for item in decisive], dtype=torch.float32, device=device)
    counts = [item.num_actions for item in decisive]

    prepared = on_device(agent.prepare(stacked), device)
    agent.eval()
    with torch.no_grad():
        old_logits, _ = agent.forward_distribution(prepared)
        legal_mask = (
            torch.arange(old_logits.shape[-1], device=device)[None, :]
            < torch.tensor(counts, dtype=torch.long, device=device)[:, None]
        )
        ref_logs = policy_log_probs(old_logits, legal_mask).detach()
        eval_loss = functools.partial(
            fork_pairwise_loss,
            action_pairs=action_pairs,
            pair_returns=pair_returns,
            legal_mask=legal_mask,
            reference_logs=ref_logs,
            temperature=config.temperature,
            min_gap=config.min_gap,
            anchor_kl=config.anchor_kl,
        )
        preview = eval_loss(old_logits)

    def _compute_aux() -> torch.Tensor:
        logits, _ = agent.forward_distribution(prepared)
        loss_val = config.fork_weight * eval_loss(logits).total
        return loss_val if value_term is None else loss_val + value_term()

    stats = ForkStats(
        total,
        preview.active_count,
        alt_better,
        preview.mean_gap,
        mean_top_p,
        mean_alt_p,
        preview.cross_entropy,
        preview.anchor_kl,
        0.0,
        collect_s,
        time.perf_counter() - t0,
    )
    return _compute_aux, stats


def _step_and_record_horizons(
    branches: Sequence[_ActiveBranch],
    *,
    scope: PolicyInference,
    obs_space: ObservationSpace,
    horizon_set: frozenset[int],
    max_steps: int,
) -> dict[tuple[int, int, int, int], float]:
    snapshots: dict[tuple[int, int, int, int], float] = {}
    for _ in range(max_steps):
        active = [branch for branch in branches if not branch.done and branch.obs is not None]
        if not active:
            break
        probs_batch, vals_batch, _ = _batched_eval(
            scope,
            obs_space=obs_space,
            observations=[b.obs for b in active if b.obs is not None],
        )
        for branch, probs, val in zip(active, probs_batch, vals_batch, strict=True):
            if branch.steps in horizon_set:
                snapshots[(branch.fork_idx, branch.pair_idx, branch.arm_idx, branch.steps)] = _hero_win_prob(
                    branch.env, hero_idx=branch.hero_idx, active_value=val
                )
            action = branch.rng.choices(range(len(probs)), weights=probs, k=1)[0]
            next_obs, is_done, is_failed = _safe_step(branch.env, action)
            branch.obs = next_obs
            branch.done = is_done
            branch.failed = is_failed
            branch.steps += 1
    return snapshots


def _summarize_one_horizon(
    roots: Sequence[_HarvestedRoot],
    *,
    grouped: Mapping[tuple[int, int, int], _ActiveBranch],
    snapshots: Mapping[tuple[int, int, int, int], float],
    horizon: int,
    playouts_per_action: int,
    total_valid: int,
) -> HorizonAuditSummary:
    val_gaps: list[float] = []
    full_gaps: list[float] = []
    for fork_idx, root in enumerate(roots):
        v_diffs: list[float] = []
        f_diffs: list[float] = []
        for pair_idx in range(playouts_per_action):
            b0, b1 = grouped[(fork_idx, pair_idx, 0)], grouped[(fork_idx, pair_idx, 1)]
            if b0.failed or b1.failed:
                continue
            f0 = _terminal_score(b0.env, hero_idx=root.hero_idx, leaf_value=None)
            f1 = _terminal_score(b1.env, hero_idx=root.hero_idx, leaf_value=None)
            v0 = snapshots.get((fork_idx, pair_idx, 0, horizon), f0)
            v1 = snapshots.get((fork_idx, pair_idx, 1, horizon), f1)
            v_diffs.append(v1 - v0)
            f_diffs.append(f1 - f0)
        if f_diffs:
            val_gaps.append(sum(v_diffs) / len(v_diffs))
            full_gaps.append(sum(f_diffs) / len(f_diffs))
    vg = np.asarray(val_gaps, dtype=np.float64)
    fg = np.asarray(full_gaps, dtype=np.float64)
    decisive = np.abs(fg) >= _DECISIVE_GAP
    dec_count = int(np.sum(decisive))
    corr = (
        float(np.corrcoef(vg, fg)[0, 1])
        if vg.size >= _MIN_CORR_SAMPLES and float(np.std(vg)) > _MIN_STD and float(np.std(fg)) > _MIN_STD
        else 0.0
    )
    agree = float(np.mean(np.sign(vg[decisive]) == np.sign(fg[decisive]))) if dec_count > 0 else 0.0
    return HorizonAuditSummary(
        horizon=horizon,
        total_forks=total_valid,
        decisive_full=dec_count,
        pearson_r=round(corr, 4),
        sign_agreement=round(agree, 4),
        mean_abs_value_gap=round(float(np.mean(np.abs(vg))) if vg.size else 0.0, 4),
        mean_abs_full_gap=round(float(np.mean(np.abs(fg))) if fg.size else 0.0, 4),
    )


def audit_horizon_correlation(
    agent: ActorCritic,
    obs_space: ObservationSpace,
    *,
    deals: Sequence[Deal],
    horizons: Sequence[int] = (8, 16, 24, 32),
    playouts_per_action: int = 4,
    max_forks: int = 64,
    decision_kinds: tuple[str, ...] | None = ("declare_attacker",),
    seed: int = 1,
) -> list[HorizonAuditSummary]:
    """Measure agreement between truncated WDL value-head gaps and full-game fork outcomes.

    Steps every paired branch to true game termination once while recording the hero's
    calibrated WDL win probability at each step count in `horizons`. Returns Pearson
    correlation and sign agreement on decisive full-game forks for each horizon.
    """
    cfg = BatchedForkConfig(
        forks_per_update=max_forks,
        playouts_per_action=playouts_per_action,
        decision_kinds=decision_kinds,
        value_samples_per_fork=0,
    )
    rng = random.Random(seed)
    allowed = None if decision_kinds is None else frozenset(decision_kinds)
    roots: list[_HarvestedRoot] = []
    ordered_horizons = sorted(set(horizons))

    agent.eval()
    with agent.inference_scope() as scope:
        for deal_idx, deal in enumerate(deals):
            if len(roots) >= max_forks:
                break
            roots.extend(
                _harvest_roots_from_deal(
                    deal,
                    scope=scope,
                    obs_space=obs_space,
                    remaining=max_forks - len(roots),
                    allowed=allowed,
                    config=cfg,
                    rng=rng,
                    base_seed=seed * 100_003 + deal_idx * 1009,
                )
            )
        branches = _spawn_all_branches(roots, config=cfg)
        snapshots = _step_and_record_horizons(
            branches,
            scope=scope,
            obs_space=obs_space,
            horizon_set=frozenset(ordered_horizons),
            max_steps=cfg.max_steps,
        )

    full_result = _assemble_results(roots, branches, playouts_per_action=playouts_per_action)
    grouped = {(b.fork_idx, b.pair_idx, b.arm_idx): b for b in branches}
    return [
        _summarize_one_horizon(
            roots,
            grouped=grouped,
            snapshots=snapshots,
            horizon=horizon,
            playouts_per_action=playouts_per_action,
            total_valid=len(full_result.examples),
        )
        for horizon in ordered_horizons
    ]


def parse_fork_kinds(spec: str | None) -> tuple[str, ...] | None:
    if spec is None:
        return None
    cleaned = spec.strip()
    if not cleaned or cleaned.lower() == "all":
        return None
    return tuple(part.strip() for part in cleaned.split(",") if part.strip())


def add_fork_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--forks-per-update", type=int, default=0)
    parser.add_argument("--fork-every", type=int, default=1)
    parser.add_argument("--fork-playouts", type=int, default=4)
    parser.add_argument("--fork-kinds", type=str, default="declare_attacker")
    parser.add_argument("--fork-weight", type=float, default=0.03)
    parser.add_argument("--fork-value-weight", type=float, default=0.25)
    parser.add_argument("--fork-workers", type=int, default=8)


@dataclass(frozen=True, slots=True)
class BatchedForkSpec:
    forks_per_update: int = 0
    fork_every: int = 1
    fork_playouts: int = 4
    fork_kinds: str = "declare_attacker"
    fork_weight: float = 0.03
    fork_value_weight: float = 0.25
    fork_workers: int = 8

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> BatchedForkSpec:
        forks = getattr(namespace, "forks_per_update", 0)
        every = getattr(namespace, "fork_every", 1)
        playouts = getattr(namespace, "fork_playouts", 4)
        kinds = getattr(namespace, "fork_kinds", "declare_attacker")
        weight = getattr(namespace, "fork_weight", 0.03)
        value_weight = getattr(namespace, "fork_value_weight", 0.25)
        workers = getattr(namespace, "fork_workers", 8)
        if not isinstance(forks, int) or forks < 0:
            msg = f"forks_per_update must be a non-negative int, got {forks!r}"
            raise ValueError(msg)
        if not isinstance(every, int) or every < 1:
            msg = f"fork_every must be >= 1, got {every!r}"
            raise ValueError(msg)
        if not isinstance(playouts, int) or playouts < 1:
            msg = f"fork_playouts must be >= 1, got {playouts!r}"
            raise ValueError(msg)
        if not isinstance(workers, int) or workers < 1:
            msg = f"fork_workers must be >= 1, got {workers!r}"
            raise ValueError(msg)
        if not isinstance(weight, (int, float)) or not isinstance(value_weight, (int, float)):
            msg = "fork weights must be numeric"
            raise TypeError(msg)
        return cls(
            forks_per_update=forks,
            fork_every=every,
            fork_playouts=playouts,
            fork_kinds=str(kinds),
            fork_weight=float(weight),
            fork_value_weight=float(value_weight),
            fork_workers=workers,
        )


def open_batched_fork_hook(
    stack: ExitStack,
    spec: BatchedForkSpec,
    *,
    hidden_dim: int,
    attention_layers: int,
    decks_path: Path | None,
    learn_device: str,
    seed: int,
    decision_memory: DecisionMemoryKind = "none",
    layout: RunLayout | None = None,
) -> PrepareAuxFn | None:
    if spec.forks_per_update <= 0:
        return None
    fork_cfg = BatchedForkConfig(
        forks_per_update=spec.forks_per_update,
        playouts_per_action=spec.fork_playouts,
        decision_kinds=parse_fork_kinds(spec.fork_kinds),
        num_workers=spec.fork_workers,
        fork_weight=spec.fork_weight,
        value_reuse_weight=spec.fork_value_weight,
    )
    fork_pool = stack.enter_context(
        BatchedForkCollectorPool(
            hidden_dim=hidden_dim,
            attention_layers=attention_layers,
            decision_memory=decision_memory,
            num_workers=spec.fork_workers,
            human_decks_path=decks_path,
        )
    )
    return make_batched_fork_prepare_aux(
        fork_pool,
        config=fork_cfg,
        device=torch.device(learn_device),
        seed=seed,
        fork_every=spec.fork_every,
        layout=layout,
    )


def make_batched_fork_prepare_aux(
    pool: BatchedForkCollectorPool,
    *,
    config: BatchedForkConfig,
    device: torch.device,
    seed: int,
    fork_every: int = 1,
    layout: RunLayout | None = None,
) -> PrepareAuxFn:
    """Build a `train_self_play` `prepare_aux` callback backed by `BatchedForkCollectorPool`.

    Collects closed-deck counterfactual forks and continuation value samples every
    `fork_every` updates, logs a `kind="fork"` metric row when `layout` is given, and
    returns the staged auxiliary loss closure.
    """
    interval = max(1, fork_every)

    def _prepare(agent: ActorCritic, update_idx: int) -> Callable[[], torch.Tensor | None] | None:
        if update_idx % interval != 0:
            return None
        t0 = time.perf_counter()
        result = pool.collect(agent, config=config, seed=seed + update_idx * 97)
        collect_s = time.perf_counter() - t0
        aux_fn, stats = build_batched_fork_aux_loss(
            agent, result=result, config=config, device=device, collect_s=collect_s
        )
        if layout is not None:
            append_metric(
                layout,
                {
                    "kind": "fork",
                    "update": update_idx,
                    "total_forks": stats.total_forks,
                    "decisive_forks": stats.decisive_forks,
                    "alt_better_forks": stats.alt_better_forks,
                    "value_samples": len(result.value_samples),
                    "mean_gap": round(stats.mean_gap, 4),
                    "mean_top_prob": round(stats.mean_top_prob, 4),
                    "mean_alt_prob": round(stats.mean_alt_prob, 4),
                    "fork_loss": round(stats.fork_loss, 5),
                    "collect_s": round(stats.collect_s, 3),
                },
            )
        return aux_fn

    return _prepare

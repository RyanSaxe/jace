"""Configuration, CLI parsing, and preflight smoke-gate helpers for `jace.train.full_data`.

Owns `FullDataConfig`, `PreflightReport`, precision/live-token resolution, CLI argument
parsing, and the 2-update preflight validation gate (VRAM headroom, numerical finiteness,
gradient/KL checks, and weight-snapshot + resume-state reload round-trips).
"""

from __future__ import annotations

import argparse
import math
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from jace.collect.agent import DecisionMemoryKind, build_agent
from jace.collect.local import add_full_data_rl_args
from jace.forks.audit import BatchedForkSpec, add_fork_cli_args
from jace.modules import default_device_name
from jace.rl import EmaWeights, PrecisionKind, SnapshotPool, StepRecord
from jace.storage.gameplay import (
    RunLayout,
    load_resume_state,
    load_weight_snapshot,
    save_resume_state,
    save_weight_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from manabot.backends.torch.policy import ActorCritic

__all__ = (
    "FullDataConfig",
    "PrecisionChoice",
    "PreflightReport",
    "TrainingSession",
    "collect_preflight_failures",
    "execute_preflight",
    "parse_full_data_args",
    "resolve_live_tokens",
    "resolve_precision",
)

_PREFLIGHT_UPDATES = 2
_PREFLIGHT_MAX_KL = 0.5
_PREFLIGHT_GRAD_NORM_MULT = 10.0
_PREFLIGHT_MAX_VRAM_FRAC = 0.9
_BYTES_PER_GIB = float(1024**3)
type PrecisionChoice = Literal["auto", "float32", "tf32", "bf16"]


def resolve_precision(precision: PrecisionChoice, *, device: str) -> PrecisionKind:
    if precision == "auto":
        return "bf16" if device.startswith("cuda") else "float32"
    return precision


def resolve_live_tokens(*, live_tokens: bool | None, device: str) -> bool:
    if live_tokens is None:
        return device.startswith("cuda")
    return live_tokens


@dataclass(frozen=True, slots=True)
class FullDataConfig:
    runs_dir: Path
    experiment: str
    run_id: str
    decks_path: Path
    updates: int = 1800
    num_envs: int = 64
    steps_per_env: int = 64
    epochs: int = 1
    minibatch_size: int = 1024
    learning_rate: float = 2.5e-4
    max_grad_norm: float = 0.267
    hidden_dim: int = 160
    attention_layers: int = 2
    decision_memory: DecisionMemoryKind = "none"
    stream_norm: bool = False
    focus_film: bool = False
    live_tokens: bool | None = None
    precision: PrecisionChoice = "auto"
    lam_policy: float = 0.5
    lam_value: float = 0.8
    select_kind: str = "top50"
    filter_scope: Literal["actor", "actor_critic"] = "actor"
    min_advantage: float = 0.01
    clip: float = 0.2
    collection_kl: float = 0.1
    magnet_kl: float = 0.01
    value_coef: float = 1.0
    schedule_kind: str = "decay"
    schedule_decay: float = 9.0
    ema: float | None = None
    snapshot_fraction: float = 0.25
    snapshot_every: int = 10
    snapshot_pool_size: int = 16
    snapshot_matching: Literal["uniform", "even_odds"] = "even_odds"
    log_grad_agreement: bool = True
    collect_workers: int = 8
    native_workers: Literal[1, 2, 4] = 1
    async_collect: bool = False
    eval_workers: int = 8
    shard_size: int = 512
    shard_refresh_every: int = 50
    card_pool_size: int | None = None
    curriculum_stages: str | None = None
    curriculum_blend_updates: int = 10
    collect_device: str = "cpu"
    learn_device: str = field(default_factory=default_device_name)
    eval_every: int = 40
    resume_save_every: int = 25
    eval_deals: int = 100
    forensics_deals: int = 20
    probe_count: int = 64
    record_train_fraction: float = 0.01
    init_checkpoint: Path | None = None
    resume: bool = False
    dense_early_checkpoints: bool = True
    preflight: bool = False
    fork: BatchedForkSpec = field(default_factory=BatchedForkSpec)
    seed: int = 1


@dataclass(frozen=True, slots=True)
class PreflightReport:
    passed: bool
    elapsed_s: float
    updates_run: int
    peak_vram_gb: float
    vram_budget_gb: float
    tps: float
    max_grad_norm_seen: float
    max_collection_kl: float
    checkpoint_reload_ok: bool
    failures: tuple[str, ...]


@dataclass(slots=True)
class TrainingSession:
    layout: RunLayout
    agent: ActorCritic
    optimizer: torch.optim.Optimizer
    pool: SnapshotPool | None
    ema_weights: EmaWeights | None
    records: list[StepRecord]


def _is_cuda_device(device_name: str) -> bool:
    return device_name.startswith("cuda") and torch.cuda.is_available()


def _reset_cuda_peak_memory(device_name: str) -> None:
    if _is_cuda_device(device_name):
        torch.cuda.reset_peak_memory_stats(torch.device(device_name))


def _cuda_vram_stats(device_name: str) -> tuple[float, float]:
    if not _is_cuda_device(device_name):
        return 0.0, 0.0
    device = torch.device(device_name)
    peak_gb = torch.cuda.max_memory_allocated(device) / _BYTES_PER_GIB
    budget_gb = torch.cuda.get_device_properties(device).total_memory / _BYTES_PER_GIB
    return peak_gb, budget_gb


def _tensors_are_finite(state: Mapping[str, object]) -> bool:
    for val in state.values():
        if isinstance(val, torch.Tensor) and val.is_floating_point() and not bool(torch.isfinite(val).all().item()):
            return False
    return True


def _verify_snapshot_reload(
    session: TrainingSession,
    *,
    fresh_agent: ActorCritic,
    iteration: int,
) -> bool:
    snap_path = save_weight_snapshot(session.layout, iteration=iteration, agent=session.agent, ema=session.ema_weights)
    snap = load_weight_snapshot(snap_path)
    live_state = snap.get("live_state_dict")
    if snap.get("iteration") != iteration or not isinstance(live_state, Mapping) or not _tensors_are_finite(live_state):
        return False
    fresh_agent.load_state_dict(live_state)
    ema_state = snap.get("ema_state_dict")
    if session.ema_weights is not None:
        if not isinstance(ema_state, Mapping) or not _tensors_are_finite(ema_state):
            return False
        fresh_agent.load_state_dict(ema_state)
    return True


def _verify_checkpoint_roundtrip(session: TrainingSession, *, cfg: FullDataConfig, iteration: int) -> bool:
    use_live_tokens = resolve_live_tokens(live_tokens=cfg.live_tokens, device=cfg.learn_device)
    fresh_agent, _ = build_agent(
        hidden_dim=cfg.hidden_dim,
        attention_layers=cfg.attention_layers,
        decision_memory=cfg.decision_memory,
        stream_norm=cfg.stream_norm,
        focus_film=cfg.focus_film,
        bucketed_shapes=True,
        live_tokens=use_live_tokens,
    )
    if not _verify_snapshot_reload(session, fresh_agent=fresh_agent, iteration=iteration):
        return False
    save_resume_state(
        session.layout,
        iteration=iteration,
        optimizer=session.optimizer,
        snapshot_pool=session.pool,
        agent=session.agent,
        ema=session.ema_weights,
    )
    fresh_opt = torch.optim.Adam(fresh_agent.parameters(), lr=cfg.learning_rate)
    fresh_pool = (
        SnapshotPool(max_size=cfg.snapshot_pool_size, matching=cfg.snapshot_matching)
        if session.pool is not None
        else None
    )
    fresh_ema = EmaWeights(fresh_agent, decay=cfg.ema) if cfg.ema is not None else None
    loaded_iter = load_resume_state(
        session.layout,
        agent=fresh_agent,
        optimizer=fresh_opt,
        snapshot_pool=fresh_pool,
        ema=fresh_ema,
        device="cpu",
    )
    return loaded_iter == iteration and _tensors_are_finite(fresh_agent.state_dict())


def collect_preflight_failures(
    *,
    updates_run: int,
    weights_finite: bool,
    losses_finite: bool,
    max_grad_norm_seen: float,
    max_grad_norm_limit: float,
    max_collection_kl: float,
    peak_vram_gb: float,
    vram_budget_gb: float,
    tps: float,
    checkpoint_reload_ok: bool,
) -> tuple[str, ...]:
    checks = (
        (updates_run < _PREFLIGHT_UPDATES, f"expected {_PREFLIGHT_UPDATES} updates, ran {updates_run}"),
        (not weights_finite, "non-finite weights detected after preflight updates"),
        (not losses_finite, "non-finite policy_loss or value_loss detected"),
        (
            not math.isfinite(max_grad_norm_seen) or max_grad_norm_seen <= 0.0,
            f"grad_norm collapsed: {max_grad_norm_seen}",
        ),
        (
            math.isfinite(max_grad_norm_seen) and max_grad_norm_seen > max_grad_norm_limit,
            f"grad_norm exploded: {max_grad_norm_seen:.4f} > {max_grad_norm_limit:.4f}",
        ),
        (
            not math.isfinite(max_collection_kl) or max_collection_kl > _PREFLIGHT_MAX_KL,
            f"collection_kl exceeded {_PREFLIGHT_MAX_KL}: {max_collection_kl:.4f}",
        ),
        (
            vram_budget_gb > 0.0 and peak_vram_gb > _PREFLIGHT_MAX_VRAM_FRAC * vram_budget_gb,
            f"peak VRAM {peak_vram_gb:.2f} GB exceeded 90% of {vram_budget_gb:.2f} GB",
        ),
        (not math.isfinite(tps) or tps <= 0.0, f"throughput collapsed: tps={tps}"),
        (not checkpoint_reload_ok, "weight snapshot or resume state reload verification failed"),
    )
    return tuple(message for failed, message in checks if failed)


def execute_preflight(
    cfg: FullDataConfig,
    *,
    run_session: Callable[[FullDataConfig], TrainingSession],
) -> PreflightReport:
    start_s = time.perf_counter()
    _reset_cuda_peak_memory(cfg.learn_device)
    with tempfile.TemporaryDirectory(prefix="jace-preflight-") as tmp_dir:
        smoke_cfg = replace(
            cfg,
            runs_dir=Path(tmp_dir),
            run_id=f"{cfg.run_id}-preflight",
            updates=_PREFLIGHT_UPDATES,
            eval_every=_PREFLIGHT_UPDATES,
            resume_save_every=_PREFLIGHT_UPDATES,
            eval_deals=2,
            forensics_deals=2,
            probe_count=4,
            collect_workers=0,
            eval_workers=1,
            resume=False,
            preflight=False,
        )
        session = run_session(smoke_cfg)
        updates_run = len(session.records)
        weights_finite = _tensors_are_finite(session.agent.state_dict())
        losses_finite = all(
            math.isfinite(rec.stats.policy_loss) and math.isfinite(rec.stats.value_loss) for rec in session.records
        )
        max_grad_norm_seen = max((rec.stats.grad_norm for rec in session.records), default=0.0)
        max_collection_kl = max((rec.stats.collection_kl for rec in session.records), default=0.0)
        total_rows = sum(rec.stats.rows for rec in session.records)
        total_step_s = sum(rec.collect_s + rec.learn_s for rec in session.records)
        tps = total_rows / max(total_step_s, 1e-6) if total_rows > 0 else 0.0
        checkpoint_reload_ok = _verify_checkpoint_roundtrip(session, cfg=smoke_cfg, iteration=updates_run)
    peak_vram_gb, vram_budget_gb = _cuda_vram_stats(cfg.learn_device)
    failures = collect_preflight_failures(
        updates_run=updates_run,
        weights_finite=weights_finite,
        losses_finite=losses_finite,
        max_grad_norm_seen=max_grad_norm_seen,
        max_grad_norm_limit=_PREFLIGHT_GRAD_NORM_MULT * cfg.max_grad_norm,
        max_collection_kl=max_collection_kl,
        peak_vram_gb=peak_vram_gb,
        vram_budget_gb=vram_budget_gb,
        tps=tps,
        checkpoint_reload_ok=checkpoint_reload_ok,
    )
    return PreflightReport(
        passed=not failures,
        elapsed_s=round(time.perf_counter() - start_s, 3),
        updates_run=updates_run,
        peak_vram_gb=round(peak_vram_gb, 3),
        vram_budget_gb=round(vram_budget_gb, 3),
        tps=round(tps, 1),
        max_grad_norm_seen=round(max_grad_norm_seen, 5),
        max_collection_kl=round(max_collection_kl, 5),
        checkpoint_reload_ok=checkpoint_reload_ok,
        failures=failures,
    )


def parse_full_data_args(argv: Sequence[str] | None) -> FullDataConfig:
    parser = argparse.ArgumentParser(description="Train Jace self-play across a human Limited deck catalog.")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--experiment", default="full-data")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--decks-path", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1800)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.267)
    add_full_data_rl_args(parser)
    parser.add_argument("--collect-device", default="cpu")
    parser.add_argument("--learn-device", default=default_device_name())
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--resume-save-every", type=int, default=25)
    parser.add_argument("--eval-deals", type=int, default=100)
    parser.add_argument("--forensics-deals", type=int, default=20)
    parser.add_argument("--probe-count", type=int, default=64)
    parser.add_argument("--record-train-fraction", type=float, default=0.01)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--dense-early-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight", action="store_true", default=False)
    add_fork_cli_args(parser)
    parser.add_argument("--seed", type=int, default=1)
    ns = parser.parse_args(argv)
    fork_spec = BatchedForkSpec.from_namespace(ns)
    raw = {key: val for key, val in vars(ns).items() if not key.startswith("fork")}
    return FullDataConfig(**raw, fork=fork_spec)

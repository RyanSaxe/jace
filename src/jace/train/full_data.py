"""Full-data 17Lands TLA self-play training loop with dense snapshots, preflight, and resume.

Trains Jace v2's `ActorCritic` across all 91,359 human Limited decks in `decks.pt` using
sharded multi-process CPU or in-process GPU/native collection, auto-resolved CUDA
precision (`bf16`) and live-token attention, optional `stream_norm`/`focus_film`,
AlphaStar-style even-odds snapshot matching, batch-to-batch gradient-agreement logging,
compact `bf16` dense early weight snapshots, atomic checkpoint resume, a `< 60s`
`--preflight` smoke gate, and fast 8-worker mirrored evaluations strictly on the 10,137
unseen held-out human decks.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import tempfile
import time
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from manabot.backends.torch.checkpoint import load_checkpoint

from jace.collect.agent import DecisionMemoryKind, build_agent, build_damped_rl_components
from jace.collect.decks import (
    HumanDeckCatalog,
    build_curriculum_stage_pools,
    build_human_deck_source,
    load_human_deck_catalog,
    top_card_pool,
)
from jace.collect.local import LocalCollectorConfig, LocalShardedCollector, add_full_data_rl_args
from jace.collect.pool import MultiProcessCollector, WorkerPoolConfig
from jace.eval import (
    Deal,
    ProbeState,
    build_human_deals,
    capture_probes,
    diff_checkpoints,
    evaluate_agent_parallel,
    evaluate_ensemble_parallel,
    paired_score_metrics,
    play_deals_parallel,
    run_forensics,
    summarize_paired_deals,
)
from jace.forks.audit import BatchedForkSpec, add_fork_cli_args, open_batched_fork_hook
from jace.modules import default_device_name
from jace.rl import (
    EmaWeights,
    PrecisionKind,
    SnapshotPool,
    StepRecord,
    TrainConfig,
    UpdateConfig,
    train_self_play,
)
from jace.storage.gameplay import (
    RunLayout,
    academy_commands,
    append_metric,
    append_train_metric,
    create_run,
    load_resume_state,
    load_weight_snapshot,
    save_forensics,
    save_policy_checkpoint,
    save_resume_state,
    save_weight_snapshot,
    should_save_weight_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace

__all__ = (
    "FullDataConfig",
    "PreflightReport",
    "main",
    "resolve_live_tokens",
    "resolve_precision",
    "run_full_data_training",
    "run_preflight",
)

_RESUME_FILENAME = "resume-state.pt"
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
class _TrainingSession:
    layout: RunLayout
    agent: ActorCritic
    optimizer: torch.optim.Optimizer
    pool: SnapshotPool | None
    ema_weights: EmaWeights | None
    records: list[StepRecord]


def _evaluate_full_data(
    agent: ActorCritic,
    baseline: ActorCritic,
    *,
    obs_space: ObservationSpace,
    layout: RunLayout,
    iteration: int,
    held_out_deals: Sequence[Deal],
    forensics_deals: Sequence[Deal],
    probes: Sequence[ProbeState],
    cfg: FullDataConfig,
    optimizer: torch.optim.Optimizer,
    snapshot_pool: SnapshotPool | None,
    ema_weights: EmaWeights | None,
    record_games: bool,
) -> dict[str, object]:
    eval_agent = copy.deepcopy(agent).to("cpu").eval()
    if ema_weights is not None:
        ema_weights.copy_to(eval_agent)
    ckpt_path, _ = save_policy_checkpoint(
        eval_agent,
        obs_space=obs_space,
        layout=layout,
        iteration=iteration,
        deterministic=False,
        extra={"iteration": iteration, "ema": cfg.ema},
    )
    save_resume_state(
        layout,
        iteration=iteration,
        optimizer=optimizer,
        snapshot_pool=snapshot_pool,
        agent=agent,
        ema=ema_weights,
    )
    eval_seed = cfg.seed + iteration * 13
    paired_det = summarize_paired_deals(
        play_deals_parallel(
            ckpt_path,
            "heuristic",
            deals=held_out_deals,
            num_workers=cfg.eval_workers,
            seed=eval_seed,
            deterministic=True,
        )
    )
    games_path = layout.eval_games_path(iteration) if record_games else None
    report = run_forensics(
        eval_agent,
        obs_space,
        deals=forensics_deals,
        seed=cfg.seed + iteration,
        deterministic=False,
        games_path=games_path,
    )
    diff = diff_checkpoints(baseline, eval_agent, obs_space=obs_space, probes=probes, top_k=8)
    payload: dict[str, object] = {
        "iteration": iteration,
        "held_out_det": asdict(paired_det),
        "forensics": report.to_dict(),
        "diff_from_init": diff.to_dict(),
        "academy": academy_commands(layout, iteration=iteration),
    }
    live_metrics: dict[str, object] = {}
    ensemble_metrics: dict[str, object] = {}
    if ema_weights is not None:
        live_det = evaluate_agent_parallel(
            agent, obs_space=obs_space, deals=held_out_deals, num_workers=cfg.eval_workers, seed=eval_seed
        )
        ensemble_det = evaluate_ensemble_parallel(
            (agent, eval_agent),
            obs_space=obs_space,
            deals=held_out_deals,
            num_workers=cfg.eval_workers,
            seed=eval_seed,
        )
        payload["live_held_out_det"] = asdict(live_det)
        payload["ensemble_held_out_det"] = asdict(ensemble_det)
        live_metrics = paired_score_metrics(live_det, prefix="live_held_out_det")
        ensemble_metrics = paired_score_metrics(ensemble_det, prefix="ensemble_held_out_det")
    save_forensics(layout, iteration=iteration, report=payload)
    append_metric(
        layout,
        {
            "kind": "eval",
            "update": iteration,
            **paired_score_metrics(paired_det, prefix="held_out_det"),
            **live_metrics,
            **ensemble_metrics,
            "win_rate_vs_heuristic": round(report.win_rate, 4),
            "attack_rate": round(report.attack_rate, 4),
            "heuristic_attack_rate": round(report.heuristic_attack_rate, 4),
            "attack_declare_spread": round(report.attack_declare_spread, 4),
            "block_rate": round(report.block_rate, 4),
            "heuristic_block_rate": round(report.heuristic_block_rate, 4),
            "missed_land_drop_rate": round(report.missed_land_drop_rate, 4),
            "wrong_side_target_rate": round(report.wrong_side_target_rate, 4),
            "diff_mean_tv": round(diff.mean_tv, 4),
            "diff_argmax_flip_rate": round(diff.argmax_flip_rate, 4),
        },
    )
    return payload


def _open_or_create_layout(cfg: FullDataConfig, *, catalog: HumanDeckCatalog, params: int) -> RunLayout:
    root = cfg.runs_dir / cfg.experiment / cfg.run_id
    if cfg.resume and root.exists():
        return RunLayout.for_root(root).ensure_dirs()
    run_config: dict[str, object] = {
        **asdict(cfg),
        "runs_dir": str(cfg.runs_dir),
        "decks_path": str(cfg.decks_path),
        "init_checkpoint": None if cfg.init_checkpoint is None else str(cfg.init_checkpoint),
        "train_decks": int(catalog.train_decks.shape[0]),
        "held_out_decks": int(catalog.held_out_decks.shape[0]),
        "unique_cards": len(catalog.card_names),
        "decklists": "closed",
        "policy_input_version": 11,
        "resolved_precision": resolve_precision(cfg.precision, device=cfg.learn_device),
        "resolved_collect_precision": resolve_precision(cfg.precision, device=cfg.collect_device),
        "resolved_live_tokens": resolve_live_tokens(live_tokens=cfg.live_tokens, device=cfg.learn_device),
        "params": params,
    }
    return create_run(runs_dir=cfg.runs_dir, experiment=cfg.experiment, run_id=cfg.run_id, config=run_config)


def _init_training_state(
    cfg: FullDataConfig,
    *,
    layout: RunLayout,
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    pool: SnapshotPool | None,
    ema_weights: EmaWeights | None,
) -> int:
    if cfg.resume and (layout.checkpoints_dir / _RESUME_FILENAME).exists():
        done = load_resume_state(
            layout,
            agent=agent,
            optimizer=optimizer,
            snapshot_pool=pool,
            ema=ema_weights,
            device=cfg.learn_device,
        )
        return done + 1
    if cfg.init_checkpoint is not None:
        agent.load_state_dict(load_checkpoint(str(cfg.init_checkpoint))[0].state_dict())
        if ema_weights is not None:
            ema_weights.sync_from(agent)
    return 1


def _run_training_session(cfg: FullDataConfig) -> _TrainingSession:
    logging.getLogger("manabot").setLevel(logging.WARNING)
    torch.manual_seed(cfg.seed)
    catalog = load_human_deck_catalog(cfg.decks_path)
    curriculum_pools = build_curriculum_stage_pools(catalog, spec=cfg.curriculum_stages)
    allowed_cards = (
        curriculum_pools[0][2]
        if curriculum_pools
        else (top_card_pool(catalog, card_count=cfg.card_pool_size) if cfg.card_pool_size is not None else None)
    )
    reference_source = build_human_deck_source(
        catalog, count=min(cfg.shard_size, 64), seed=cfg.seed, allowed_cards=allowed_cards
    )
    learn_precision = resolve_precision(cfg.precision, device=cfg.learn_device)
    collect_precision = resolve_precision(cfg.precision, device=cfg.collect_device)
    use_live_tokens = resolve_live_tokens(live_tokens=cfg.live_tokens, device=cfg.learn_device)
    agent, obs_space = build_agent(
        hidden_dim=cfg.hidden_dim,
        attention_layers=cfg.attention_layers,
        decision_memory=cfg.decision_memory,
        stream_norm=cfg.stream_norm,
        focus_film=cfg.focus_film,
        bucketed_shapes=True,
        live_tokens=use_live_tokens,
    )
    layout = _open_or_create_layout(cfg, catalog=catalog, params=sum(p.numel() for p in agent.parameters()))
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
    pool = (
        SnapshotPool(max_size=cfg.snapshot_pool_size, matching=cfg.snapshot_matching)
        if cfg.snapshot_fraction > 0.0
        else None
    )
    ema_weights = EmaWeights(agent, decay=cfg.ema) if cfg.ema is not None else None
    start_update = _init_training_state(
        cfg, layout=layout, agent=agent, optimizer=optimizer, pool=pool, ema_weights=ema_weights
    )
    baseline = copy.deepcopy(agent).to("cpu").eval()
    credit, select, loss, loss_schedule = build_damped_rl_components(
        updates=cfg.updates,
        learning_rate=cfg.learning_rate,
        lam_policy=cfg.lam_policy,
        lam_value=cfg.lam_value,
        select_kind=cfg.select_kind,
        min_advantage=cfg.min_advantage,
        clip=cfg.clip,
        collection_kl=cfg.collection_kl,
        magnet_kl=cfg.magnet_kl,
        value_coef=cfg.value_coef,
        schedule_kind=cfg.schedule_kind,
        schedule_decay=cfg.schedule_decay,
    )
    held_out_deals = build_human_deals(catalog, split="held_out", count=cfg.eval_deals, seed=cfg.seed + 20_000)
    forensics_deals = held_out_deals[: cfg.forensics_deals]
    probes = capture_probes(forensics_deals, max_probes=cfg.probe_count)
    records: list[StepRecord] = []

    run_eval = partial(
        _evaluate_full_data,
        agent,
        baseline,
        obs_space=obs_space,
        layout=layout,
        held_out_deals=held_out_deals,
        forensics_deals=forensics_deals,
        probes=probes,
        cfg=cfg,
        optimizer=optimizer,
        snapshot_pool=pool,
        ema_weights=ema_weights,
    )
    if start_update == 1:
        run_eval(iteration=0, record_games=False)

    def on_update(record: StepRecord) -> None:
        append_train_metric(layout, record)
        records.append(record)
        if should_save_weight_snapshot(record.update, dense_early=cfg.dense_early_checkpoints):
            save_weight_snapshot(layout, iteration=record.update, agent=agent, ema=ema_weights)
        if record.update % cfg.eval_every == 0 or record.update == cfg.updates:
            run_eval(iteration=record.update, record_games=True)
        elif cfg.resume_save_every > 0 and record.update % cfg.resume_save_every == 0:
            save_resume_state(
                layout,
                iteration=record.update,
                optimizer=optimizer,
                snapshot_pool=pool,
                agent=agent,
                ema=ema_weights,
            )

    train_cfg = TrainConfig(
        updates=cfg.updates,
        num_envs=cfg.num_envs,
        steps_per_env=cfg.steps_per_env,
        seed=cfg.seed,
        collect_device=cfg.collect_device,
        precision=collect_precision,
        native_workers=cfg.native_workers,
        async_collect=cfg.async_collect,
        update=UpdateConfig(
            epochs=cfg.epochs,
            minibatch_size=cfg.minibatch_size,
            max_grad_norm=cfg.max_grad_norm,
            normalize_advantages=False,
            filter_scope=cfg.filter_scope,
            device=cfg.learn_device,
            precision=learn_precision,
        ),
        games_path=layout.train_games_path(),
        record_fraction=cfg.record_train_fraction,
        snapshot_fraction=cfg.snapshot_fraction,
        snapshot_every=cfg.snapshot_every,
        snapshot_pool_size=cfg.snapshot_pool_size,
        snapshot_matching=cfg.snapshot_matching,
        start_update=start_update,
        log_grad_agreement=cfg.log_grad_agreement,
        ema=cfg.ema,
    )
    with ExitStack() as stack:
        collect_fn = (
            stack.enter_context(
                MultiProcessCollector(
                    WorkerPoolConfig(
                        num_workers=cfg.collect_workers,
                        envs_per_worker=max(1, cfg.num_envs // cfg.collect_workers),
                        steps_per_env=cfg.steps_per_env,
                        hidden_dim=cfg.hidden_dim,
                        attention_layers=cfg.attention_layers,
                        decision_memory=cfg.decision_memory,
                        stream_norm=cfg.stream_norm,
                        focus_film=cfg.focus_film,
                        seed=cfg.seed,
                        games_path=layout.train_games_path(),
                        record_fraction=cfg.record_train_fraction,
                        enable_snapshots=cfg.snapshot_fraction > 0.0,
                        human_decks_path=cfg.decks_path,
                        shard_size=cfg.shard_size,
                        shard_refresh_every=cfg.shard_refresh_every,
                        card_pool_size=cfg.card_pool_size,
                        curriculum_pools=curriculum_pools,
                        curriculum_blend_updates=cfg.curriculum_blend_updates,
                        start_update=start_update,
                    )
                )
            ).collect
            if cfg.collect_workers > 0
            else stack.enter_context(
                LocalShardedCollector(
                    LocalCollectorConfig(
                        num_envs=cfg.num_envs,
                        steps_per_env=cfg.steps_per_env,
                        collect_device=cfg.collect_device,
                        precision=collect_precision,
                        native_workers=cfg.native_workers,
                        seed=cfg.seed,
                        games_path=layout.train_games_path(),
                        record_fraction=cfg.record_train_fraction,
                        enable_snapshots=cfg.snapshot_fraction > 0.0,
                        shard_size=cfg.shard_size,
                        shard_refresh_every=cfg.shard_refresh_every,
                        card_pool_size=cfg.card_pool_size,
                        curriculum_pools=curriculum_pools,
                        curriculum_blend_updates=cfg.curriculum_blend_updates,
                        start_update=start_update,
                    ),
                    catalog=catalog,
                    obs_space=obs_space,
                    template_agent=agent,
                )
            ).collect
        )
        prepare_aux = open_batched_fork_hook(
            stack,
            cfg.fork,
            hidden_dim=cfg.hidden_dim,
            attention_layers=cfg.attention_layers,
            decision_memory=cfg.decision_memory,
            stream_norm=cfg.stream_norm,
            focus_film=cfg.focus_film,
            decks_path=cfg.decks_path,
            learn_device=cfg.learn_device,
            seed=cfg.seed + 50_000,
            layout=layout,
        )
        train_self_play(
            agent,
            optimizer,
            obs_space=obs_space,
            deck_source=reference_source,
            credit=credit,
            select=select,
            loss=loss,
            loss_schedule=loss_schedule,
            config=train_cfg,
            collect_fn=collect_fn,
            prepare_aux=prepare_aux,
            on_update=on_update,
            snapshot_pool=pool,
            ema_weights=ema_weights,
        )
    return _TrainingSession(
        layout=layout,
        agent=agent,
        optimizer=optimizer,
        pool=pool,
        ema_weights=ema_weights,
        records=records,
    )


def run_full_data_training(cfg: FullDataConfig) -> RunLayout:
    """Train Jace v2 across all 17Lands TLA human decks with held-out evaluation and resume.

    Loads `cfg.decks_path` (`decks.pt`), runs sharded 8-worker self-play over the 91,359
    training decks, evaluates every `cfg.eval_every` updates on 200 mirrored games drawn
    strictly from the 10,137 unseen held-out human decks, writes dense `bf16` weight
    snapshots, and saves atomic resume state.
    """
    return _run_training_session(cfg).layout


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
    session: _TrainingSession,
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


def _verify_checkpoint_roundtrip(session: _TrainingSession, *, cfg: FullDataConfig, iteration: int) -> bool:
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


def _collect_preflight_failures(
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


def run_preflight(cfg: FullDataConfig) -> PreflightReport:
    """Run a self-contained 2-update smoke gate in a temporary directory.

    Preserves the caller's model dimensions, precision, device, and RL hyperparameters
    while shrinking evaluation and worker counts so numerical stability, VRAM headroom,
    throughput, and both weight-snapshot and resume-state round-trips are verified before
    committing to a long GPU training run.
    """
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
        session = _run_training_session(smoke_cfg)
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
    failures = _collect_preflight_failures(
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


def _parse_args(argv: Sequence[str] | None) -> FullDataConfig:
    parser = argparse.ArgumentParser(description="Train Jace v2 across all 17Lands TLA human decks.")
    parser.add_argument("--runs-dir", type=Path, default=Path("../jace-runs"))
    parser.add_argument("--experiment", default="v2-full-data")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--decks-path", type=Path, default=Path("../jace-runs/data/decks.pt"))
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


def main(argv: Sequence[str] | None = None) -> int:
    cfg = _parse_args(argv)
    if cfg.preflight:
        report = run_preflight(cfg)
        sys.stdout.write(json.dumps(asdict(report), sort_keys=True) + "\n")
        return 0 if report.passed else 1
    layout = run_full_data_training(cfg)
    sys.stdout.write(json.dumps({"run_root": str(layout.root)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

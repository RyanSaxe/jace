"""Full-data human-deck self-play training loop with dense snapshots, preflight, and resume.

Trains Jace's `ActorCritic` across all human Limited decks in `decks.pt` using
sharded multi-process CPU or in-process GPU/native collection, auto-resolved CUDA
precision (`bf16`) and live-token attention, optional `stream_norm`/`focus_film`,
AlphaStar-style even-odds snapshot matching, batch-to-batch gradient-agreement logging,
compact `bf16` dense early weight snapshots, atomic checkpoint resume, a `< 60s`
`--preflight` smoke gate, and fast multi-worker mirrored evaluations strictly on
unseen held-out human decks. CLI parsing, configuration, and preflight checks live in
`jace.train._full_data_cli` and are re-exported here.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
from typing import TYPE_CHECKING

import torch
from manabot.backends.torch.checkpoint import load_checkpoint

from jace.collect.agent import build_agent, build_damped_rl_components
from jace.collect.decks import (
    HumanDeckCatalog,
    build_curriculum_stage_pools,
    build_human_deck_source,
    load_human_deck_catalog,
    top_card_pool,
)
from jace.collect.local import LocalCollectorConfig, LocalShardedCollector
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
from jace.forks.audit import open_batched_fork_hook
from jace.rl import (
    EmaWeights,
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
    save_forensics,
    save_policy_checkpoint,
    save_resume_state,
    save_weight_snapshot,
    should_save_weight_snapshot,
)
from jace.train._full_data_cli import (
    FullDataConfig,
    PreflightReport,
    TrainingSession,
    collect_preflight_failures,
    execute_preflight,
    parse_full_data_args,
    resolve_live_tokens,
    resolve_precision,
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
_collect_preflight_failures = collect_preflight_failures
_parse_args = parse_full_data_args


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


def _run_training_session(cfg: FullDataConfig) -> TrainingSession:
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
    return TrainingSession(
        layout=layout,
        agent=agent,
        optimizer=optimizer,
        pool=pool,
        ema_weights=ema_weights,
        records=records,
    )


def run_full_data_training(cfg: FullDataConfig) -> RunLayout:
    """Train Jace across human Limited decks in `cfg.decks_path` with held-out evaluation and resume.

    Loads `cfg.decks_path` (`decks.pt`), runs self-play over the training split,
    evaluates every `cfg.eval_every` updates on mirrored games drawn strictly
    from the unseen held-out split, writes dense `bf16` weight snapshots, and
    saves atomic resume state.
    """
    return _run_training_session(cfg).layout


def run_preflight(cfg: FullDataConfig) -> PreflightReport:
    """Run a self-contained 2-update smoke gate in a temporary directory.

    Preserves the caller's model dimensions, precision, device, and RL hyperparameters
    while shrinking evaluation and worker counts so numerical stability, VRAM headroom,
    throughput, and both weight-snapshot and resume-state round-trips are verified before
    committing to a long GPU training run.
    """
    return execute_preflight(cfg, run_session=_run_training_session)


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

"""Full-data 17Lands TLA self-play training loop with held-out parallel eval and resume.

Trains Jace v2's `ActorCritic` across all 91,359 human Limited decks in `decks.pt` using
sharded multi-process CPU or in-process GPU/native collection, auto-resolved CUDA
precision (`bf16`) and live-token attention, optional `stream_norm`/`focus_film`,
AlphaStar-style even-odds snapshot matching, batch-to-batch gradient-agreement logging,
atomic checkpoint resume, and fast 8-worker mirrored evaluations strictly on the 10,137
unseen held-out human decks.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
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
    save_forensics,
    save_policy_checkpoint,
    save_resume_state,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace

__all__ = ("FullDataConfig", "main", "resolve_live_tokens", "resolve_precision", "run_full_data_training")

_RESUME_FILENAME = "resume-state.pt"
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
    fork: BatchedForkSpec = field(default_factory=BatchedForkSpec)
    seed: int = 1


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


def run_full_data_training(cfg: FullDataConfig) -> RunLayout:
    """Train Jace v2 across all 17Lands TLA human decks with held-out evaluation and resume.

    Loads `cfg.decks_path` (`decks.pt`), runs sharded 8-worker self-play over the 91,359
    training decks, evaluates every `cfg.eval_every` updates on 200 mirrored games drawn
    strictly from the 10,137 unseen held-out human decks, and saves atomic resume state.
    """
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
    return layout


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
    add_fork_cli_args(parser)
    parser.add_argument("--seed", type=int, default=1)
    ns = parser.parse_args(argv)
    fork_spec = BatchedForkSpec.from_namespace(ns)
    raw = {key: val for key, val in vars(ns).items() if not key.startswith("fork")}
    return FullDataConfig(**raw, fork=fork_spec)


def main(argv: Sequence[str] | None = None) -> int:
    cfg = _parse_args(argv)
    layout = run_full_data_training(cfg)
    sys.stdout.write(json.dumps({"run_root": str(layout.root)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

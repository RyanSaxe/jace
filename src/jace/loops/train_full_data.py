"""Full-data 17Lands TLA self-play training loop with held-out parallel eval and resume.

Trains Jace v2's width-160 `ActorCritic` across all 91,359 human Limited decks in
`decks.pt` using sharded multi-process CPU collection and MPS updates. Supports
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from manabot.backends.torch.checkpoint import load_checkpoint

from jace.eval import (
    Deal,
    HumanDeckCatalog,
    ProbeState,
    build_human_deals,
    build_human_deck_source,
    capture_probes,
    diff_checkpoints,
    load_human_deck_catalog,
    play_deals_parallel,
    run_forensics,
    summarize_paired_deals,
    top_card_pool,
)
from jace.loops.collect_pool import MultiProcessCollector, WorkerPoolConfig
from jace.loops.train_ten_decks import build_agent
from jace.rl import (
    CreditFn,
    DampedSchedule,
    DecaySchedule,
    LossFn,
    LossScheduleFn,
    SelectFn,
    SnapshotPool,
    StepRecord,
    TrainConfig,
    UpdateConfig,
    all_rows,
    damped_policy_loss,
    gae_wdl,
    top_fraction,
    train_self_play,
)
from jace.runs import (
    RunLayout,
    academy_commands,
    append_metric,
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

__all__ = (
    "FullDataConfig",
    "main",
    "run_full_data_training",
)

_RESUME_FILENAME = "resume-state.pt"


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
    lam_policy: float = 0.5
    lam_value: float = 0.8
    select_kind: str = "top50"
    min_advantage: float = 0.01
    clip: float = 0.2
    collection_kl: float = 0.1
    magnet_kl: float = 0.01
    value_coef: float = 1.0
    schedule_kind: str = "decay"
    schedule_decay: float = 9.0
    snapshot_fraction: float = 0.25
    snapshot_every: int = 10
    snapshot_pool_size: int = 16
    snapshot_matching: Literal["uniform", "even_odds"] = "even_odds"
    log_grad_agreement: bool = True
    collect_workers: int = 8
    eval_workers: int = 8
    shard_size: int = 512
    shard_refresh_every: int = 50
    card_pool_size: int | None = None
    collect_device: str = "cpu"
    learn_device: str = "mps"
    eval_every: int = 40
    eval_deals: int = 100
    forensics_deals: int = 20
    probe_count: int = 64
    record_train_fraction: float = 0.01
    init_checkpoint: Path | None = None
    resume: bool = False
    seed: int = 1


def _build_rl_components(cfg: FullDataConfig) -> tuple[CreditFn, SelectFn, LossFn, LossScheduleFn | None]:
    credit: CreditFn = gae_wdl(gamma=1.0, lam_policy=cfg.lam_policy, lam_value=cfg.lam_value)
    select: SelectFn = (
        all_rows
        if cfg.select_kind == "all"
        else top_fraction(0.5 if cfg.select_kind == "top50" else 0.25, minimum=cfg.min_advantage)
    )
    default_loss = damped_policy_loss(
        clip=cfg.clip, collection_kl=cfg.collection_kl, magnet_kl=cfg.magnet_kl, value_coef=cfg.value_coef
    )
    if cfg.schedule_kind == "none":
        return credit, select, default_loss, None
    schedule: DampedSchedule | DecaySchedule = (
        DecaySchedule(
            total_updates=cfg.updates,
            lr_initial=cfg.learning_rate,
            tau_initial=cfg.magnet_kl,
            decay=cfg.schedule_decay,
        )
        if cfg.schedule_kind == "decay"
        else DampedSchedule(total_updates=cfg.updates, lr_max=cfg.learning_rate, tau_scale=cfg.magnet_kl)
    )

    def _loss_for_update(update_idx: int) -> tuple[float, float, LossFn]:
        lr, tau = schedule.rates(update_idx)
        return (
            lr,
            tau,
            damped_policy_loss(
                clip=cfg.clip, collection_kl=cfg.collection_kl, magnet_kl=tau, value_coef=cfg.value_coef
            ),
        )

    return credit, select, default_loss, _loss_for_update


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
    record_games: bool,
) -> dict[str, object]:
    eval_agent = copy.deepcopy(agent).to("cpu").eval()
    ckpt_path, _ = save_policy_checkpoint(
        eval_agent,
        obs_space=obs_space,
        layout=layout,
        iteration=iteration,
        deterministic=False,
        extra={"iteration": iteration},
    )
    save_resume_state(layout, iteration=iteration, optimizer=optimizer, snapshot_pool=snapshot_pool)
    paired_det = summarize_paired_deals(
        play_deals_parallel(
            ckpt_path,
            "heuristic",
            deals=held_out_deals,
            num_workers=cfg.eval_workers,
            seed=cfg.seed + iteration * 13,
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
    save_forensics(layout, iteration=iteration, report=payload)
    append_metric(
        layout,
        {
            "kind": "eval",
            "update": iteration,
            "held_out_det_win_rate": round(paired_det.win_rate, 4),
            "held_out_det_ci_low": round(paired_det.ci_low, 4),
            "held_out_det_ci_high": round(paired_det.ci_high, 4),
            "held_out_det_sweeps_won": paired_det.deal_sweeps_won,
            "held_out_det_sweeps_lost": paired_det.deal_sweeps_lost,
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
        return RunLayout.for_root(root)
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
        "params": params,
    }
    return create_run(runs_dir=cfg.runs_dir, experiment=cfg.experiment, run_id=cfg.run_id, config=run_config)


def _log_train_step(layout: RunLayout, record: StepRecord) -> None:
    stats = record.stats
    row: dict[str, object] = {
        "kind": "train",
        "update": record.update,
        "opponent": record.opponent,
        "lr": round(record.learning_rate, 7),
        "tau": round(record.magnet_tau, 6),
        "rows": stats.rows,
        "selected_rows": stats.selected_rows,
        "completed_games": stats.completed_games,
        "policy_loss": round(stats.policy_loss, 5),
        "value_loss": round(stats.value_loss, 5),
        "collection_kl": round(stats.collection_kl, 5),
        "magnet_kl": round(stats.magnet_kl, 5),
        "entropy": round(stats.entropy, 5),
        "clip_fraction": round(stats.clip_fraction, 5),
        "grad_norm": round(stats.grad_norm, 4),
        "mean_advantage_abs": round(stats.mean_advantage_abs, 5),
        "collect_s": round(record.collect_s, 3),
        "learn_s": round(record.learn_s, 3),
        "tps": round(stats.rows / max(record.collect_s + record.learn_s, 1e-6), 1),
    }
    if record.grad_cosine is not None:
        row["grad_cosine"] = round(record.grad_cosine, 5)
    if record.snapshot_win_rate is not None:
        row["snapshot_win_rate"] = round(record.snapshot_win_rate, 4)
    append_metric(layout, row)


def _init_training_state(
    cfg: FullDataConfig,
    *,
    layout: RunLayout,
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    pool: SnapshotPool | None,
) -> int:
    if cfg.resume and (layout.checkpoints_dir / _RESUME_FILENAME).exists():
        done = load_resume_state(layout, agent=agent, optimizer=optimizer, snapshot_pool=pool, device=cfg.learn_device)
        return done + 1
    if cfg.init_checkpoint is not None:
        agent.load_state_dict(load_checkpoint(str(cfg.init_checkpoint))[0].state_dict())
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
    allowed_cards = top_card_pool(catalog, card_count=cfg.card_pool_size) if cfg.card_pool_size is not None else None
    reference_source = build_human_deck_source(
        catalog, count=min(cfg.shard_size, 64), seed=cfg.seed, allowed_cards=allowed_cards
    )
    agent, obs_space = build_agent(
        hidden_dim=cfg.hidden_dim, attention_layers=cfg.attention_layers, bucketed_shapes=True, live_tokens=False
    )
    layout = _open_or_create_layout(cfg, catalog=catalog, params=sum(p.numel() for p in agent.parameters()))
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
    pool = (
        SnapshotPool(max_size=cfg.snapshot_pool_size, matching=cfg.snapshot_matching)
        if cfg.snapshot_fraction > 0.0
        else None
    )
    start_update = _init_training_state(cfg, layout=layout, agent=agent, optimizer=optimizer, pool=pool)
    baseline = copy.deepcopy(agent).to("cpu").eval()
    credit, select, loss, loss_schedule = _build_rl_components(cfg)
    held_out_deals = build_human_deals(catalog, split="held_out", count=cfg.eval_deals, seed=cfg.seed + 20_000)
    forensics_deals = held_out_deals[: cfg.forensics_deals]
    probes = capture_probes(forensics_deals, max_probes=cfg.probe_count)
    if start_update == 1:
        _evaluate_full_data(
            agent,
            baseline,
            obs_space=obs_space,
            layout=layout,
            iteration=0,
            held_out_deals=held_out_deals,
            forensics_deals=forensics_deals,
            probes=probes,
            cfg=cfg,
            optimizer=optimizer,
            snapshot_pool=pool,
            record_games=False,
        )

    def on_update(record: StepRecord) -> None:
        _log_train_step(layout, record)
        if record.update % cfg.eval_every == 0 or record.update == cfg.updates:
            _evaluate_full_data(
                agent,
                baseline,
                obs_space=obs_space,
                layout=layout,
                iteration=record.update,
                held_out_deals=held_out_deals,
                forensics_deals=forensics_deals,
                probes=probes,
                cfg=cfg,
                optimizer=optimizer,
                snapshot_pool=pool,
                record_games=True,
            )

    train_cfg = TrainConfig(
        updates=cfg.updates,
        num_envs=cfg.num_envs,
        steps_per_env=cfg.steps_per_env,
        seed=cfg.seed,
        collect_device=cfg.collect_device,
        update=UpdateConfig(
            epochs=cfg.epochs,
            minibatch_size=cfg.minibatch_size,
            max_grad_norm=cfg.max_grad_norm,
            normalize_advantages=False,
            device=cfg.learn_device,
        ),
        games_path=layout.train_games_path(),
        record_fraction=cfg.record_train_fraction,
        snapshot_fraction=cfg.snapshot_fraction,
        snapshot_every=cfg.snapshot_every,
        snapshot_pool_size=cfg.snapshot_pool_size,
        snapshot_matching=cfg.snapshot_matching,
        start_update=start_update,
        log_grad_agreement=cfg.log_grad_agreement,
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
                        seed=cfg.seed,
                        games_path=layout.train_games_path(),
                        record_fraction=cfg.record_train_fraction,
                        enable_snapshots=cfg.snapshot_fraction > 0.0,
                        human_decks_path=cfg.decks_path,
                        shard_size=cfg.shard_size,
                        shard_refresh_every=cfg.shard_refresh_every,
                        card_pool_size=cfg.card_pool_size,
                    )
                )
            ).collect
            if cfg.collect_workers > 0
            else None
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
            on_update=on_update,
            snapshot_pool=pool,
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
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--lam-policy", type=float, default=0.5)
    parser.add_argument("--lam-value", type=float, default=0.8)
    parser.add_argument("--select", dest="select_kind", choices=("top25", "top50", "all"), default="top50")
    parser.add_argument("--min-advantage", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--collection-kl", type=float, default=0.1)
    parser.add_argument("--magnet-kl", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--schedule-kind", choices=("damped", "decay", "none"), default="decay")
    parser.add_argument("--schedule-decay", type=float, default=9.0)
    parser.add_argument("--snapshot-fraction", type=float, default=0.25)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--snapshot-pool-size", type=int, default=16)
    parser.add_argument("--snapshot-matching", choices=("uniform", "even_odds"), default="even_odds")
    parser.add_argument("--log-grad-agreement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collect-workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--shard-refresh-every", type=int, default=50)
    parser.add_argument("--card-pool-size", type=int, default=None)
    parser.add_argument("--collect-device", default="cpu")
    parser.add_argument("--learn-device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--eval-deals", type=int, default=100)
    parser.add_argument("--forensics-deals", type=int, default=20)
    parser.add_argument("--probe-count", type=int, default=64)
    parser.add_argument("--record-train-fraction", type=float, default=0.01)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    return FullDataConfig(**vars(parser.parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> int:
    cfg = _parse_args(argv)
    layout = run_full_data_training(cfg)
    sys.stdout.write(json.dumps({"run_root": str(layout.root)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

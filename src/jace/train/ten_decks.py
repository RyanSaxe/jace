"""Train a Jace v2 self-play agent on the 10-deck TLA lab with Academy-native artifacts.

Wires `jace.rl`, `jace.collect`, `jace.eval`, `jace.forks`, and `jace.storage.gameplay`
over the ten closed-decklist `elemental-world-v2` decks using upstream `manabot` v11
observations and `ActorCritic` (`policy_film=True`, `categorical_wdl`, `masked_mean`).
Supports decaying damped-KL schedules, raw unnormalized win-probability advantages, a
historical snapshot opponent pool, and full Academy replays.
"""

import argparse
import copy
import logging
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.training.decks import UniformDecks

from jace.collect.agent import build_agent
from jace.collect.decks import elemental_decks
from jace.collect.pool import MultiProcessCollector, WorkerPoolConfig
from jace.eval import (
    Deal,
    ProbeState,
    build_deals,
    capture_probes,
    diff_checkpoints,
    run_forensics,
)
from jace.forks.collect import ForkCollectorPool, ForkConfig
from jace.modules import default_device_name
from jace.rl import (
    CollectFn,
    CreditFn,
    DampedSchedule,
    DecaySchedule,
    LossFn,
    LossScheduleFn,
    SelectFn,
    StepRecord,
    TrainConfig,
    UpdateConfig,
    all_rows,
    damped_policy_loss,
    gae_wdl,
    monte_carlo,
    ppo_loss,
    top_fraction,
    train_self_play,
)
from jace.rl.loop import AuxLossFn
from jace.storage.gameplay import (
    RunLayout,
    academy_commands,
    append_metric,
    create_run,
    save_forensics,
    save_policy_checkpoint,
)

__all__ = (
    "LabConfig",
    "build_agent",
    "main",
    "run_ten_deck_lab",
)


@dataclass(frozen=True, slots=True)
class LabConfig:
    runs_dir: Path
    experiment: str
    run_id: str
    updates: int = 150
    num_envs: int = 32
    steps_per_env: int = 64
    epochs: int = 1
    minibatch_size: int = 512
    learning_rate: float = 1e-4
    max_grad_norm: float = 0.267
    hidden_dim: int = 64
    attention_layers: int = 2
    credit_kind: str = "gae"
    lam_policy: float = 0.5
    lam_value: float = 0.8
    select_kind: str = "top25"
    min_advantage: float = 0.01
    loss_kind: str = "damped"
    clip: float = 0.2
    collection_kl: float = 0.1
    magnet_kl: float = 0.05
    value_coef: float = 1.0
    normalize_advantages: bool | None = None
    schedule_kind: str = "damped"
    schedule_decay: float = 9.0
    snapshot_fraction: float = 0.25
    snapshot_every: int = 10
    snapshot_pool_size: int = 16
    collect_workers: int = 0
    native_workers: Literal[1, 2, 4] = 1
    async_collect: bool = False
    collect_device: str = "cpu"
    learn_device: str = field(default_factory=default_device_name)
    eval_every: int = 25
    eval_deals: int = 20
    probe_count: int = 64
    record_train_fraction: float = 0.02
    init_checkpoint: Path | None = None
    fork_config: ForkConfig | None = None
    seed: int = 1


def _build_select(cfg: LabConfig) -> SelectFn:
    if cfg.select_kind == "all":
        return all_rows
    return top_fraction(0.5 if cfg.select_kind == "top50" else 0.25, minimum=cfg.min_advantage)


def _build_schedule(cfg: LabConfig) -> DampedSchedule | DecaySchedule | None:
    if cfg.schedule_kind == "none":
        return None
    if cfg.schedule_kind == "decay":
        return DecaySchedule(
            total_updates=cfg.updates,
            lr_initial=cfg.learning_rate,
            tau_initial=cfg.magnet_kl,
            decay=cfg.schedule_decay,
        )
    return DampedSchedule(total_updates=cfg.updates, lr_max=cfg.learning_rate, tau_scale=cfg.magnet_kl)


def _select_components(cfg: LabConfig) -> tuple[CreditFn, SelectFn, LossFn, LossScheduleFn | None]:
    credit: CreditFn = (
        monte_carlo
        if cfg.credit_kind == "mc"
        else gae_wdl(gamma=1.0, lam_policy=cfg.lam_policy, lam_value=cfg.lam_value)
    )
    select = _build_select(cfg)
    if cfg.loss_kind == "ppo":
        return credit, select, ppo_loss(clip=cfg.clip, entropy_coef=0.01, value_coef=cfg.value_coef), None
    default_loss = damped_policy_loss(
        clip=cfg.clip,
        collection_kl=cfg.collection_kl,
        magnet_kl=cfg.magnet_kl,
        value_coef=cfg.value_coef,
    )
    schedule = _build_schedule(cfg)
    if schedule is None:
        return credit, select, default_loss, None

    def loss_for_update(update_idx: int) -> tuple[float, float, LossFn]:
        lr, tau = schedule.rates(update_idx)
        return (
            lr,
            tau,
            damped_policy_loss(
                clip=cfg.clip,
                collection_kl=cfg.collection_kl,
                magnet_kl=tau,
                value_coef=cfg.value_coef,
            ),
        )

    return credit, select, default_loss, loss_for_update


def _evaluate_and_save(
    agent: ActorCritic,
    baseline: ActorCritic,
    *,
    obs_space: ObservationSpace,
    layout: RunLayout,
    iteration: int,
    deals: Sequence[Deal],
    probes: Sequence[ProbeState],
    seed: int,
    record_games: bool,
) -> dict[str, object]:
    eval_agent = copy.deepcopy(agent).to("cpu").eval()
    save_policy_checkpoint(
        eval_agent,
        obs_space=obs_space,
        layout=layout,
        iteration=iteration,
        deterministic=False,
        extra={"iteration": iteration},
    )
    games_path = layout.eval_games_path(iteration) if record_games else None
    report = run_forensics(
        eval_agent,
        obs_space,
        deals=deals,
        seed=seed + iteration,
        deterministic=False,
        games_path=games_path,
    )
    diff = diff_checkpoints(baseline, eval_agent, obs_space=obs_space, probes=probes, top_k=8)
    payload: dict[str, object] = {
        "iteration": iteration,
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
            "win_rate_vs_heuristic": round(report.win_rate, 4),
            "ci_low": round(report.ci_low, 4),
            "ci_high": round(report.ci_high, 4),
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


def _open_pools(
    stack: ExitStack, *, cfg: LabConfig, layout: RunLayout
) -> tuple[CollectFn | None, ForkCollectorPool | None]:
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
                )
            )
        ).collect
        if cfg.collect_workers > 0
        else None
    )
    fork_pool = (
        stack.enter_context(
            ForkCollectorPool(
                hidden_dim=cfg.hidden_dim,
                attention_layers=cfg.attention_layers,
                num_workers=cfg.fork_config.num_workers,
            )
        )
        if cfg.fork_config is not None
        else None
    )
    return collect_fn, fork_pool


def run_ten_deck_lab(cfg: LabConfig) -> RunLayout:
    """Run the 10-deck TLA self-play lab and persist all Academy-native artifacts.

    Evaluates update 0 before training, runs `cfg.updates` self-play updates on the ten
    closed-decklist decks (optionally mixing frozen historical snapshots), and evaluates
    every `cfg.eval_every` updates plus the final checkpoint with Academy replay logs.
    """
    logging.getLogger("manabot").setLevel(logging.WARNING)
    torch.manual_seed(cfg.seed)
    named_decks = elemental_decks()
    deck_source = UniformDecks(
        decks=tuple(deck for _, deck in named_decks),
        decklists="closed",
        sampling="uniform-independent-v1",
    )
    agent, obs_space = build_agent(
        hidden_dim=cfg.hidden_dim,
        attention_layers=cfg.attention_layers,
        bucketed_shapes=True,
        live_tokens=False,
    )
    if cfg.init_checkpoint is not None:
        loaded_policy = load_checkpoint(str(cfg.init_checkpoint))[0]
        agent.load_state_dict(loaded_policy.state_dict())
    baseline = copy.deepcopy(agent).to("cpu").eval()
    credit, select, loss, loss_schedule = _select_components(cfg)
    normalize_advantages = cfg.loss_kind == "ppo" if cfg.normalize_advantages is None else cfg.normalize_advantages

    run_config: dict[str, object] = {
        **asdict(cfg),
        "runs_dir": str(cfg.runs_dir),
        "init_checkpoint": None if cfg.init_checkpoint is None else str(cfg.init_checkpoint),
        "decks": [name for name, _ in named_decks],
        "decklists": "closed",
        "policy_input_version": 11,
        "normalize_advantages": normalize_advantages,
        "params": sum(p.numel() for p in agent.parameters()),
    }
    layout = create_run(
        runs_dir=cfg.runs_dir,
        experiment=cfg.experiment,
        run_id=cfg.run_id,
        config=run_config,
    )

    deals = build_deals(named_decks, count=cfg.eval_deals, seed=cfg.seed + 10_000, decklists="closed")
    probes = capture_probes(deals, max_probes=cfg.probe_count)
    _evaluate_and_save(
        agent,
        baseline,
        obs_space=obs_space,
        layout=layout,
        iteration=0,
        deals=deals,
        probes=probes,
        seed=cfg.seed,
        record_games=False,
    )

    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
    fork_pool: ForkCollectorPool | None = None

    def on_update(record: StepRecord) -> None:
        stats = record.stats
        append_metric(
            layout,
            {
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
            },
        )
        if record.update % cfg.eval_every == 0 or record.update == cfg.updates:
            _evaluate_and_save(
                agent,
                baseline,
                obs_space=obs_space,
                layout=layout,
                iteration=record.update,
                deals=deals,
                probes=probes,
                seed=cfg.seed,
                record_games=True,
            )

    train_cfg = TrainConfig(
        updates=cfg.updates,
        num_envs=cfg.num_envs,
        steps_per_env=cfg.steps_per_env,
        seed=cfg.seed,
        collect_device=cfg.collect_device,
        native_workers=cfg.native_workers,
        async_collect=cfg.async_collect,
        update=UpdateConfig(
            epochs=cfg.epochs,
            minibatch_size=cfg.minibatch_size,
            max_grad_norm=cfg.max_grad_norm,
            normalize_advantages=normalize_advantages,
            device=cfg.learn_device,
        ),
        games_path=layout.train_games_path(),
        record_fraction=cfg.record_train_fraction,
        snapshot_fraction=cfg.snapshot_fraction,
        snapshot_every=cfg.snapshot_every,
        snapshot_pool_size=cfg.snapshot_pool_size,
    )
    with ExitStack() as stack:
        collect_fn, fork_pool = _open_pools(stack, cfg=cfg, layout=layout)

        def prepare_aux(cur_agent: ActorCritic, update_idx: int) -> AuxLossFn | None:
            if cfg.fork_config is None or fork_pool is None:
                return None
            aux_fn, f_stats = fork_pool.prepare_joint_aux(
                cur_agent,
                config=cfg.fork_config,
                device=torch.device(cfg.learn_device),
                seed=cfg.seed * 10_000 + update_idx * 97,
            )
            append_metric(layout, {"kind": "fork", "update": update_idx, **asdict(f_stats)})
            return aux_fn

        train_self_play(
            agent,
            optimizer,
            obs_space=obs_space,
            deck_source=deck_source,
            credit=credit,
            select=select,
            loss=loss,
            loss_schedule=loss_schedule,
            config=train_cfg,
            collect_fn=collect_fn,
            prepare_aux=prepare_aux if cfg.fork_config is not None else None,
            on_update=on_update,
        )
    return layout


def _parse_args(argv: Sequence[str] | None) -> LabConfig:
    parser = argparse.ArgumentParser(description="Train Jace v2 on the 10-deck TLA self-play lab.")
    parser.add_argument("--runs-dir", type=Path, default=Path("../jace-runs"))
    parser.add_argument("--experiment", default="v2-ten-deck-lab")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.267)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--credit", dest="credit_kind", choices=("gae", "mc"), default="gae")
    parser.add_argument("--lam-policy", type=float, default=0.5)
    parser.add_argument("--lam-value", type=float, default=0.8)
    parser.add_argument("--select", dest="select_kind", choices=("top25", "top50", "all"), default="top25")
    parser.add_argument("--min-advantage", type=float, default=0.01)
    parser.add_argument("--loss", dest="loss_kind", choices=("damped", "ppo"), default="damped")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--collection-kl", type=float, default=0.1)
    parser.add_argument("--magnet-kl", type=float, default=0.05)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--schedule-kind", choices=("damped", "decay", "none"), default="damped")
    parser.add_argument("--schedule-decay", type=float, default=9.0)
    parser.add_argument("--snapshot-fraction", type=float, default=0.25)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--snapshot-pool-size", type=int, default=16)
    parser.add_argument("--collect-workers", type=int, default=0)
    parser.add_argument("--native-workers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--async-collect", action="store_true", default=False)
    parser.add_argument("--collect-device", default="cpu")
    parser.add_argument("--learn-device", default=default_device_name())
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-deals", type=int, default=20)
    parser.add_argument("--probe-count", type=int, default=64)
    parser.add_argument("--record-train-fraction", type=float, default=0.02)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    return LabConfig(**vars(parser.parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> int:
    cfg = _parse_args(argv)
    layout = run_ten_deck_lab(cfg)
    sys.stdout.write(f"Run complete: {layout.root}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

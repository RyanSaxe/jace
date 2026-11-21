"""Train a Jace v2 self-play agent on the 10-deck TLA lab with Academy-native artifacts.

Wires `jace.rl`, `jace.eval`, and `jace.runs` over the ten closed-decklist `elemental-world-v2`
decks using upstream `manabot` v11 observations and `ActorCritic` (`policy_film=True`,
`categorical_wdl`, `masked_mean`). Periodically saves portable `.safetensors` checkpoints
plus Academy `PolicySpec` files, audits 40-game decision forensics against
`managym.Heuristic`, and diffs fixed held-out probe states against the initial policy.
"""

import argparse
import copy
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.training.decks import UniformDecks

from jace.eval import (
    Deal,
    ProbeState,
    build_deals,
    capture_probes,
    diff_checkpoints,
    elemental_decks,
    run_forensics,
)
from jace.rl import (
    CreditFn,
    LossFn,
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
from jace.runs import (
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
    updates: int = 80
    num_envs: int = 32
    steps_per_env: int = 64
    epochs: int = 2
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    hidden_dim: int = 64
    attention_layers: int = 2
    credit_kind: str = "gae"
    lam_policy: float = 0.5
    lam_value: float = 0.8
    select_kind: str = "top25"
    loss_kind: str = "damped"
    collection_kl: float = 0.5
    magnet_kl: float = 0.05
    collect_device: str = "cpu"
    learn_device: str = "mps"
    eval_every: int = 20
    eval_deals: int = 20
    probe_count: int = 64
    record_train_fraction: float = 0.05
    seed: int = 1


def build_agent(
    *,
    hidden_dim: int = 64,
    attention_layers: int = 2,
    bucketed_shapes: bool = True,
    live_tokens: bool = False,
) -> tuple[ActorCritic, ObservationSpace]:
    obs_space = ObservationSpace(
        ObservationConfig(
            policy_input_version=11,
            max_cards_per_player=80,
            max_permanents_per_player=64,
        )
    )
    model_cfg = ModelConfig(
        world="world",
        hidden_dim=hidden_dim,
        attention_layers=attention_layers,
        num_attention_heads=4,
        policy_film=True,
        value_kind="categorical_wdl",
        value_aggregation="masked_mean",
    )
    agent = ActorCritic(
        obs_space,
        model_cfg,
        bucketed_shapes=bucketed_shapes,
        live_tokens=live_tokens,
    )
    return agent, obs_space


def _select_components(cfg: LabConfig) -> tuple[CreditFn, SelectFn, LossFn]:
    credit: CreditFn = (
        monte_carlo
        if cfg.credit_kind == "mc"
        else gae_wdl(gamma=1.0, lam_policy=cfg.lam_policy, lam_value=cfg.lam_value)
    )
    select: SelectFn = all_rows if cfg.select_kind == "all" else top_fraction(0.25)
    loss: LossFn = (
        ppo_loss(clip=0.2, entropy_coef=0.01, value_coef=0.5)
        if cfg.loss_kind == "ppo"
        else damped_policy_loss(
            clip=0.2,
            collection_kl=cfg.collection_kl,
            magnet_kl=cfg.magnet_kl,
            value_coef=0.5,
        )
    )
    return credit, select, loss


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


def run_ten_deck_lab(cfg: LabConfig) -> RunLayout:
    """Run the 10-deck TLA self-play lab and persist all Academy-native artifacts.

    Evaluates update 0 before training, runs `cfg.updates` self-play updates on the ten
    closed-decklist decks, and evaluates every `cfg.eval_every` updates plus the final
    checkpoint, recording native Academy replay files on the final evaluation.
    """
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
    baseline = copy.deepcopy(agent).to("cpu").eval()
    credit, select, loss = _select_components(cfg)

    layout = create_run(
        runs_dir=cfg.runs_dir,
        experiment=cfg.experiment,
        run_id=cfg.run_id,
        config={
            "decks": [name for name, _ in named_decks],
            "decklists": "closed",
            "policy_input_version": 11,
            "updates": cfg.updates,
            "num_envs": cfg.num_envs,
            "steps_per_env": cfg.steps_per_env,
            "epochs": cfg.epochs,
            "minibatch_size": cfg.minibatch_size,
            "learning_rate": cfg.learning_rate,
            "hidden_dim": cfg.hidden_dim,
            "attention_layers": cfg.attention_layers,
            "credit_kind": cfg.credit_kind,
            "lam_policy": cfg.lam_policy,
            "lam_value": cfg.lam_value,
            "select_kind": cfg.select_kind,
            "loss_kind": cfg.loss_kind,
            "collection_kl": cfg.collection_kl,
            "magnet_kl": cfg.magnet_kl,
            "collect_device": cfg.collect_device,
            "learn_device": cfg.learn_device,
            "params": sum(p.numel() for p in agent.parameters()),
            "seed": cfg.seed,
        },
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

    def on_update(record: StepRecord) -> None:
        stats = record.stats
        append_metric(
            layout,
            {
                "kind": "train",
                "update": record.update,
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
        should_eval = record.update % cfg.eval_every == 0 or record.update == cfg.updates
        if should_eval:
            _evaluate_and_save(
                agent,
                baseline,
                obs_space=obs_space,
                layout=layout,
                iteration=record.update,
                deals=deals,
                probes=probes,
                seed=cfg.seed,
                record_games=record.update == cfg.updates,
            )

    train_self_play(
        agent,
        optimizer,
        obs_space=obs_space,
        deck_source=deck_source,
        credit=credit,
        select=select,
        loss=loss,
        config=TrainConfig(
            updates=cfg.updates,
            num_envs=cfg.num_envs,
            steps_per_env=cfg.steps_per_env,
            seed=cfg.seed,
            collect_device=cfg.collect_device,
            update=UpdateConfig(
                epochs=cfg.epochs,
                minibatch_size=cfg.minibatch_size,
                device=cfg.learn_device,
            ),
            games_path=layout.train_games_path(),
            record_fraction=cfg.record_train_fraction,
        ),
        on_update=on_update,
    )
    return layout


def _parse_args(argv: Sequence[str] | None) -> LabConfig:
    parser = argparse.ArgumentParser(description="Train Jace v2 on the 10-deck TLA self-play lab.")
    parser.add_argument("--runs-dir", type=Path, default=Path("../jace-runs"))
    parser.add_argument("--experiment", default="v2-ten-deck-lab")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--credit", choices=("gae", "mc"), default="gae")
    parser.add_argument("--lam-policy", type=float, default=0.5)
    parser.add_argument("--lam-value", type=float, default=0.8)
    parser.add_argument("--select", choices=("top25", "all"), default="top25")
    parser.add_argument("--loss", choices=("damped", "ppo"), default="damped")
    parser.add_argument("--collection-kl", type=float, default=0.5)
    parser.add_argument("--magnet-kl", type=float, default=0.05)
    parser.add_argument("--collect-device", default="cpu")
    parser.add_argument("--learn-device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-deals", type=int, default=20)
    parser.add_argument("--probe-count", type=int, default=64)
    parser.add_argument("--record-train-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    ns = parser.parse_args(argv)
    return LabConfig(
        runs_dir=ns.runs_dir,
        experiment=ns.experiment,
        run_id=ns.run_id,
        updates=ns.updates,
        num_envs=ns.num_envs,
        steps_per_env=ns.steps_per_env,
        epochs=ns.epochs,
        minibatch_size=ns.minibatch_size,
        learning_rate=ns.lr,
        hidden_dim=ns.hidden_dim,
        attention_layers=ns.attention_layers,
        credit_kind=ns.credit,
        lam_policy=ns.lam_policy,
        lam_value=ns.lam_value,
        select_kind=ns.select,
        loss_kind=ns.loss,
        collection_kl=ns.collection_kl,
        magnet_kl=ns.magnet_kl,
        collect_device=ns.collect_device,
        learn_device=ns.learn_device,
        eval_every=ns.eval_every,
        eval_deals=ns.eval_deals,
        probe_count=ns.probe_count,
        record_train_fraction=ns.record_train_fraction,
        seed=ns.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    cfg = _parse_args(argv)
    layout = run_ten_deck_lab(cfg)
    sys.stdout.write(f"Run complete: {layout.root}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

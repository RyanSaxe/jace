"""Fast, directionally accurate experiment harness CLI for Jace v2.

Runs either a warm-start branch from a saved checkpoint (default for strategy, fork,
and credit ideas) or a short from-scratch micro-run (for architecture changes) over
all ten closed TLA decks, then judges the final checkpoint against a baseline using
adaptive parallel mirrored evaluation (200 -> 400 -> 1,000 mirrored games across CPU
workers) plus forensics gating. Also supports standalone `--eval-only` adaptive H2H
scoring between any two checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from jace.eval import AdaptiveEvalConfig, AdaptiveEvalResult, play_adaptive_parallel
from jace.loops.forks import ForkConfig
from jace.loops.train_ten_decks import LabConfig, run_ten_deck_lab

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    "HarnessVerdict",
    "decide_promotion",
    "evaluate_checkpoints",
    "main",
)

_MAX_LAND_REGRESSION = 0.03
_MIN_EVAL_POINTS = 2
_DEFAULT_EVAL_CONFIG = AdaptiveEvalConfig()


@dataclass(frozen=True, slots=True)
class HarnessVerdict:
    decision: Literal["PROMOTE", "KILL", "INCONCLUSIVE"]
    reason: str
    h2h: AdaptiveEvalResult
    vs_heuristic: AdaptiveEvalResult | None
    missed_land_delta: float | None


def decide_promotion(
    h2h: AdaptiveEvalResult,
    *,
    missed_land_delta: float | None = None,
) -> tuple[Literal["PROMOTE", "KILL", "INCONCLUSIVE"], str]:
    """Map adaptive H2H result and land-drop forensics onto a promotion verdict.

    Kills any candidate whose missed land drop rate regresses by more than 3 percentage
    points or whose adaptive H2H verdict is `LOSS`. Promotes only candidates with a
    confirmed `WIN` verdict, leaving close unresolved outcomes as `INCONCLUSIVE`.
    """
    if missed_land_delta is not None and missed_land_delta > _MAX_LAND_REGRESSION:
        return "KILL", f"missed land drop regressed by +{missed_land_delta:.1%}"
    if h2h.verdict == "WIN":
        return "PROMOTE", h2h.reason
    if h2h.verdict == "LOSS":
        return "KILL", h2h.reason
    return "INCONCLUSIVE", h2h.reason


def evaluate_checkpoints(
    candidate: Path,
    baseline: Path,
    *,
    num_workers: int = 8,
    seed: int = 20_000,
    eval_heuristic: bool = True,
    missed_land_delta: float | None = None,
    config: AdaptiveEvalConfig = _DEFAULT_EVAL_CONFIG,
) -> HarnessVerdict:
    """Judge `candidate` against `baseline` with adaptive parallel mirrored games.

    Runs staged mirrored head-to-head evaluation (`config.stages`, default 200 -> 400 ->
    1,000 games) across `num_workers` CPU processes, optionally scores `candidate`
    against `managym.Heuristic` on 200 mirrored games, and returns a `HarnessVerdict`.
    """
    h2h = play_adaptive_parallel(
        candidate,
        baseline,
        seed=seed,
        num_workers=num_workers,
        deterministic=True,
        config=config,
    )
    vs_heur = (
        play_adaptive_parallel(
            candidate,
            "heuristic",
            seed=seed + 50_000,
            num_workers=num_workers,
            deterministic=True,
            config=AdaptiveEvalConfig(stages=(config.stages[0],)),
        )
        if eval_heuristic
        else None
    )
    decision, reason = decide_promotion(h2h, missed_land_delta=missed_land_delta)
    return HarnessVerdict(
        decision=decision,
        reason=reason,
        h2h=h2h,
        vs_heuristic=vs_heur,
        missed_land_delta=missed_land_delta,
    )


def _missed_land_delta(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    evals = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip() and json.loads(line).get("kind") == "eval"
    ]
    if len(evals) < _MIN_EVAL_POINTS:
        return None
    return float(evals[-1]["missed_land_drop_rate"]) - float(evals[0]["missed_land_drop_rate"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fast experiment screening and adaptive mirrored evaluation.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and score two existing checkpoints.")
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("../jace-runs"))
    parser.add_argument("--experiment", default="v2-fast-harness")
    parser.add_argument("--run-id", default="screen")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--collect-workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--lam-policy", type=float, default=0.95)
    parser.add_argument("--lam-value", type=float, default=0.9)
    parser.add_argument("--magnet-kl", type=float, default=0.005)
    parser.add_argument("--snapshot-fraction", type=float, default=0.10)
    parser.add_argument("--stages", default="100,200,500", help="Comma-separated mirrored deal counts.")
    parser.add_argument("--forks-per-update", type=int, default=0, help="Counterfactual forks collected per update.")
    parser.add_argument("--fork-playouts", type=int, default=4, help="Paired playouts per option (R).")
    parser.add_argument("--fork-kinds", default="declare_attacker", help="Comma-separated decision kinds, or 'all'.")
    parser.add_argument("--fork-weight", type=float, default=0.25)
    parser.add_argument("--fork-anchor-kl", type=float, default=0.2)
    parser.add_argument("--skip-heuristic", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser


def _parse_stages(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _build_fork_config(ns: argparse.Namespace) -> ForkConfig | None:
    if ns.forks_per_update <= 0:
        return None
    raw = str(ns.fork_kinds).strip().lower()
    kinds = None if raw in ("", "all") else tuple(part.strip() for part in raw.split(",") if part.strip())
    return ForkConfig(
        forks_per_update=ns.forks_per_update,
        playouts_per_action=ns.fork_playouts,
        decision_kinds=kinds,
        num_workers=ns.collect_workers,
        fork_weight=ns.fork_weight,
        anchor_kl=ns.fork_anchor_kl,
    )


def main(argv: Sequence[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)
    stages = _parse_stages(ns.stages)
    eval_cfg = AdaptiveEvalConfig(stages=stages)
    land_delta: float | None = None
    output_path: Path | None = None

    if ns.eval_only:
        if ns.candidate is None or ns.baseline is None:
            msg = "--eval-only requires both --candidate and --baseline"
            raise ValueError(msg)
        candidate, baseline = ns.candidate, ns.baseline
    else:
        lab_cfg = LabConfig(
            runs_dir=ns.runs_dir,
            experiment=ns.experiment,
            run_id=ns.run_id,
            updates=ns.updates,
            num_envs=ns.num_envs,
            collect_workers=ns.collect_workers,
            steps_per_env=64,
            epochs=ns.epochs,
            minibatch_size=ns.minibatch_size,
            learning_rate=ns.lr,
            hidden_dim=ns.hidden_dim,
            attention_layers=ns.attention_layers,
            lam_policy=ns.lam_policy,
            lam_value=ns.lam_value,
            select_kind="top50",
            loss_kind="damped",
            clip=0.2,
            collection_kl=0.1,
            magnet_kl=ns.magnet_kl,
            normalize_advantages=True,
            schedule_kind="decay",
            schedule_decay=2.0,
            snapshot_fraction=ns.snapshot_fraction,
            snapshot_every=max(5, ns.updates // 5),
            eval_every=ns.updates,
            eval_deals=20,
            probe_count=64,
            record_train_fraction=0.0,
            init_checkpoint=ns.init_checkpoint,
            fork_config=_build_fork_config(ns),
            learn_device="mps" if torch.backends.mps.is_available() else "cpu",
            seed=ns.seed,
        )
        layout = run_ten_deck_lab(lab_cfg)
        candidate = layout.checkpoint_path(ns.updates)
        baseline = ns.baseline if ns.baseline is not None else ns.init_checkpoint
        if baseline is None:
            baseline = layout.checkpoint_path(0)
        land_delta = _missed_land_delta(layout.metrics_path)
        output_path = layout.root / "verdict.json"

    verdict = evaluate_checkpoints(
        candidate,
        baseline,
        num_workers=ns.eval_workers,
        seed=20_000 + ns.seed,
        eval_heuristic=not ns.skip_heuristic,
        missed_land_delta=land_delta,
        config=eval_cfg,
    )
    payload = json.dumps(asdict(verdict), indent=2)
    if output_path is not None:
        output_path.write_text(payload + "\n")
    sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

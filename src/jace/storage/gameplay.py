"""Academy-native experiment run storage for checkpoints, metrics, games, and forensics.

Every training run writes to one self-contained directory under
`<runs_dir>/<experiment>/<run_id>/` using upstream `manabot` and `managym` formats so
every trained checkpoint and recorded game opens directly in Academy without conversion.
Evaluation checkpoints use portable `.safetensors` containers beside companion
`PolicySpec` JSON files, compact `bf16` weight snapshots (`weights-u*.pt`) track dense
early progression without optimizer state, atomic `resume-state.pt` files preserve full
crash-recovery state, game recordings use `managym`'s native JSONL schema, and
structured metrics and forensics stay plain JSON for lightweight inspection.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace

    from jace.rl.ema import EmaWeights
    from jace.rl.pool import SnapshotPool, StepRecord

__all__ = (
    "DENSE_EARLY_UPDATES",
    "RunExistsError",
    "RunLayout",
    "academy_commands",
    "append_metric",
    "append_train_metric",
    "create_run",
    "load_resume_state",
    "load_weight_snapshot",
    "save_forensics",
    "save_policy_checkpoint",
    "save_resume_state",
    "save_weight_snapshot",
    "should_save_weight_snapshot",
)

DENSE_EARLY_UPDATES: frozenset[int] = frozenset({25, 50, 100, 150, 200, 300, 400})
_WEIGHT_SNAPSHOT_INTERVAL = 100
_RESUME_FILENAME = "resume-state.pt"


class RunExistsError(FileExistsError):
    def __init__(self, root: Path) -> None:
        super().__init__(f"run directory already exists: {root}")


@dataclass(frozen=True, slots=True)
class RunLayout:
    root: Path
    config_path: Path
    metrics_path: Path
    checkpoints_dir: Path
    games_dir: Path
    forensics_dir: Path

    @classmethod
    def for_root(cls, root: Path) -> RunLayout:
        resolved = root.resolve()
        return cls(
            root=resolved,
            config_path=resolved / "config.json",
            metrics_path=resolved / "metrics.jsonl",
            checkpoints_dir=resolved / "checkpoints",
            games_dir=resolved / "games",
            forensics_dir=resolved / "forensics",
        )

    def ensure_dirs(self) -> RunLayout:
        for directory in (self.checkpoints_dir, self.games_dir, self.forensics_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def checkpoint_path(self, iteration: int) -> Path:
        return self.checkpoints_dir / f"iter-{iteration:04d}.safetensors"

    def weight_snapshot_path(self, iteration: int) -> Path:
        return self.checkpoints_dir / f"weights-u{iteration:05d}.pt"

    def policy_spec_path(self, iteration: int) -> Path:
        return self.checkpoints_dir / f"iter-{iteration:04d}.policy.json"

    def train_games_path(self) -> Path:
        return self.games_dir / "train.jsonl"

    def eval_games_path(self, iteration: int) -> Path:
        return self.games_dir / f"eval-iter-{iteration:04d}.jsonl"

    def forensics_path(self, iteration: int) -> Path:
        return self.forensics_dir / f"iter-{iteration:04d}.json"


def create_run(
    *,
    runs_dir: Path,
    experiment: str,
    run_id: str,
    config: Mapping[str, object],
) -> RunLayout:
    layout = RunLayout.for_root(runs_dir / experiment / run_id)
    if layout.root.exists():
        raise RunExistsError(layout.root)
    for directory in (layout.checkpoints_dir, layout.games_dir, layout.forensics_dir):
        directory.mkdir(parents=True, exist_ok=False)
    layout.config_path.write_text(json.dumps(dict(config), indent=2, sort_keys=True) + "\n")
    return layout


def save_policy_checkpoint(
    agent: ActorCritic,
    *,
    obs_space: ObservationSpace,
    layout: RunLayout,
    iteration: int,
    deterministic: bool = False,
    extra: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    """Save a portable safetensors checkpoint alongside an Academy `PolicySpec` file.

    Saves with `player_configs=None` so the checkpoint can be played in Academy
    against any admitted deck rather than being locked to one training matchup.
    `deterministic` controls whether Academy samples or takes argmax moves, and
    `extra` attaches arbitrary JSON-serializable metadata inside the container.
    Returns `(checkpoint_path, policy_spec_path)`.
    """
    checkpoint_path = layout.checkpoint_path(iteration)
    save_checkpoint(agent, obs_space, checkpoint_path, player_configs=None, extra=extra)
    spec_path = layout.policy_spec_path(iteration)
    spec = {
        "provider": "manabot",
        "artifact": str(checkpoint_path.resolve()),
        "config": {"deterministic": deterministic},
    }
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return checkpoint_path, spec_path


def append_metric(layout: RunLayout, record: Mapping[str, object]) -> None:
    line = json.dumps(dict(record), sort_keys=True) + "\n"
    with layout.metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def append_train_metric(layout: RunLayout, record: StepRecord) -> None:
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


def save_forensics(layout: RunLayout, *, iteration: int, report: Mapping[str, object]) -> Path:
    path = layout.forensics_path(iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n")
    return path


def academy_commands(layout: RunLayout, *, iteration: int) -> dict[str, str]:
    spec = layout.policy_spec_path(iteration)
    return {
        "play": f"uv run managym academy play --policy {spec} --opponent {spec.stem}",
        "replay_train": f"uv run managym academy replay {layout.train_games_path()}",
        "replay_eval": f"uv run managym academy replay {layout.eval_games_path(iteration)}",
    }


def save_resume_state(
    layout: RunLayout,
    *,
    iteration: int,
    optimizer: torch.optim.Optimizer,
    snapshot_pool: SnapshotPool | None,
    agent: ActorCritic | None = None,
    ema: EmaWeights | None = None,
) -> Path:
    payload: dict[str, object] = {
        "iteration": iteration,
        "checkpoint": str(layout.checkpoint_path(iteration)),
        "optimizer": optimizer.state_dict(),
        "snapshot_pool": None if snapshot_pool is None else snapshot_pool.export_state(),
    }
    if agent is not None:
        payload["live_agent_state"] = {key: val.detach().cpu().clone() for key, val in agent.state_dict().items()}
    if ema is not None:
        payload["ema_state"] = ema.export_state()
    target = layout.checkpoints_dir / _RESUME_FILENAME
    partial = target.with_suffix(".pt.partial")
    torch.save(payload, partial)
    partial.rename(target)
    return target


def load_resume_state(
    layout: RunLayout,
    *,
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    snapshot_pool: SnapshotPool | None,
    ema: EmaWeights | None = None,
    device: str | torch.device | None = None,
) -> int:
    path = layout.checkpoints_dir / _RESUME_FILENAME
    if not path.exists():
        msg = f"resume state not found: {path}"
        raise FileNotFoundError(msg)
    state = torch.load(path, map_location="cpu", weights_only=False)
    iteration = int(state["iteration"])
    live_state = state.get("live_agent_state")
    if isinstance(live_state, Mapping):
        agent.load_state_dict(live_state)
    else:
        ckpt_path = Path(str(state["checkpoint"]))
        agent.load_state_dict(load_checkpoint(str(ckpt_path))[0].state_dict())
    if device is not None:
        agent.to(device)
    optimizer.load_state_dict(state["optimizer"])
    pool_state = state.get("snapshot_pool")
    if snapshot_pool is not None and isinstance(pool_state, Mapping):
        snapshot_pool.restore_state(pool_state)
    _restore_ema_state(ema, state.get("ema_state"), agent=agent)
    return iteration


def _restore_ema_state(ema: EmaWeights | None, raw_state: object, *, agent: ActorCritic) -> None:
    if ema is None:
        return
    if isinstance(raw_state, Mapping):
        ema.restore_state(raw_state)
    else:
        ema.sync_from(agent)


def should_save_weight_snapshot(update: int, *, dense_early: bool = True) -> bool:
    if dense_early and update in DENSE_EARLY_UPDATES:
        return True
    return update > 0 and update % _WEIGHT_SNAPSHOT_INTERVAL == 0


def _to_bf16_cpu_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        cpu_tensor = tensor.detach().cpu()
        converted[key] = cpu_tensor.to(dtype=torch.bfloat16) if cpu_tensor.is_floating_point() else cpu_tensor.clone()
    return converted


def _extract_ema_tensors(ema: EmaWeights, *, live_keys: set[str]) -> dict[str, torch.Tensor]:
    exported = ema.export_state()
    merged: dict[str, torch.Tensor] = {}
    for section in ("params", "buffers"):
        raw = exported.get(section)
        if not isinstance(raw, Mapping):
            continue
        for key, val in raw.items():
            name = str(key)
            if isinstance(val, torch.Tensor) and name in live_keys:
                merged[name] = val
    return _to_bf16_cpu_state(merged)


def save_weight_snapshot(
    layout: RunLayout,
    *,
    iteration: int,
    agent: ActorCritic,
    ema: EmaWeights | None = None,
) -> Path:
    """Atomically write compact bf16 live and optional EMA weights without optimizer state.

    Floating-point tensors are stored in `torch.bfloat16` on CPU while integer and boolean
    buffers keep their exact original dtypes. Writes to a `.partial` sibling file and
    renames into place so interrupted transfers or crashes never leave corrupt snapshots.
    """
    live_raw = agent.state_dict()
    payload: dict[str, object] = {
        "iteration": iteration,
        "live_state_dict": _to_bf16_cpu_state(live_raw),
        "ema_state_dict": None if ema is None else _extract_ema_tensors(ema, live_keys=set(live_raw)),
    }
    target = layout.weight_snapshot_path(iteration)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".pt.partial")
    torch.save(payload, partial)
    partial.rename(target)
    return target


def load_weight_snapshot(path: Path) -> dict[str, object]:
    if not path.exists():
        msg = f"weight snapshot not found: {path}"
        raise FileNotFoundError(msg)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        msg = f"invalid weight snapshot payload in {path}"
        raise TypeError(msg)
    return dict(raw)

"""Academy-native experiment run storage for checkpoints, metrics, games, and forensics.

Every training run writes to one self-contained directory under
`<runs_dir>/<experiment>/<run_id>/` using upstream `manabot` and `managym` formats so
every trained checkpoint and recorded game opens directly in Academy without conversion.
Checkpoints are stored as portable `.safetensors` containers beside companion
`PolicySpec` JSON files, game recordings use `managym`'s native JSONL schema, and
structured metrics and forensics stay plain JSON for lightweight inspection.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace

__all__ = (
    "RunExistsError",
    "RunLayout",
    "academy_commands",
    "append_metric",
    "create_run",
    "save_forensics",
    "save_policy_checkpoint",
)


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
    def for_root(cls, root: Path) -> "RunLayout":
        resolved = root.resolve()
        return cls(
            root=resolved,
            config_path=resolved / "config.json",
            metrics_path=resolved / "metrics.jsonl",
            checkpoints_dir=resolved / "checkpoints",
            games_dir=resolved / "games",
            forensics_dir=resolved / "forensics",
        )

    def checkpoint_path(self, iteration: int) -> Path:
        return self.checkpoints_dir / f"iter-{iteration:04d}.safetensors"

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


def save_forensics(layout: RunLayout, *, iteration: int, report: Mapping[str, object]) -> Path:
    path = layout.forensics_path(iteration)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n")
    return path


def academy_commands(layout: RunLayout, *, iteration: int) -> dict[str, str]:
    spec = layout.policy_spec_path(iteration)
    return {
        "play": f"uv run managym academy play --policy {spec} --opponent {spec.stem}",
        "replay_train": f"uv run managym academy replay {layout.train_games_path()}",
        "replay_eval": f"uv run managym academy replay {layout.eval_games_path(iteration)}",
    }

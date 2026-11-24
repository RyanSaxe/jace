"""Behavior tests for Academy-native experiment run storage.

Verifies run directory creation and collision refusal, JSONL metric and forensics
persistence, portable safetensors round-tripping through `manabot`, and policy
registration through `managym.policy.PolicyRegistry`. Every check runs against a
temporary directory using real `ActorCritic` weights and `ObservationSpace` v11
layouts so Academy admission stays verified end to end.
"""

import json
from pathlib import Path

import pytest
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from managym.policy import PolicyRegistry, PolicySpec

from jace.storage.gameplay import (
    RunExistsError,
    RunLayout,
    academy_commands,
    append_metric,
    create_run,
    save_forensics,
    save_policy_checkpoint,
)


def test_create_run_rejects_existing_directory(tmp_path: Path) -> None:
    layout = create_run(runs_dir=tmp_path, experiment="bench", run_id="r1", config={"seed": 1})
    assert json.loads(layout.config_path.read_text()) == {"seed": 1}
    assert layout.checkpoints_dir.is_dir()
    assert layout.games_dir.is_dir()
    assert layout.forensics_dir.is_dir()
    with pytest.raises(RunExistsError):
        create_run(runs_dir=tmp_path, experiment="bench", run_id="r1", config={"seed": 2})


def test_metrics_and_forensics_persist_cleanly(tmp_path: Path) -> None:
    layout = create_run(runs_dir=tmp_path, experiment="bench", run_id="r2", config={"lr": 3e-4})
    append_metric(layout, {"update": 1, "loss": 0.5})
    append_metric(layout, {"update": 2, "loss": 0.4})
    rows = [json.loads(line) for line in layout.metrics_path.read_text().splitlines()]
    assert rows == [{"loss": 0.5, "update": 1}, {"loss": 0.4, "update": 2}]

    forensics_path = save_forensics(layout, iteration=10, report={"wrong_side_rate": 0.02})
    assert json.loads(forensics_path.read_text()) == {"wrong_side_rate": 0.02}


def test_resume_layout_creates_missing_forensics_and_games_dirs(tmp_path: Path) -> None:
    partial_root = tmp_path / "bench" / "resumed"
    (partial_root / "checkpoints").mkdir(parents=True)
    layout = RunLayout.for_root(partial_root)
    forensics_path = save_forensics(layout, iteration=400, report={"ok": True})
    assert json.loads(forensics_path.read_text()) == {"ok": True}

    layout.ensure_dirs()
    assert layout.games_dir.is_dir()
    assert layout.forensics_dir.is_dir()


def test_saved_policy_checkpoint_reloads_and_registers_in_academy(tmp_path: Path) -> None:
    obs_space = ObservationSpace(ObservationConfig())
    model_cfg = ModelConfig(
        world="world",
        hidden_dim=32,
        attention_layers=1,
        num_attention_heads=4,
        policy_film=True,
        value_kind="categorical_wdl",
        value_aggregation="masked_mean",
    )
    agent = ActorCritic(obs_space, model_cfg)
    layout = create_run(runs_dir=tmp_path, experiment="bench", run_id="r3", config={"model": "tiny"})

    checkpoint_path, spec_path = save_policy_checkpoint(
        agent,
        obs_space=obs_space,
        layout=layout,
        iteration=5,
        deterministic=False,
    )
    assert checkpoint_path.is_file()
    assert spec_path.is_file()

    reloaded, reloaded_space = load_checkpoint(str(checkpoint_path))
    assert reloaded.config.value_kind == "categorical_wdl"
    assert reloaded_space.encoder.config.policy_input_version == 11

    spec = PolicySpec.read(spec_path)
    registry = PolicyRegistry()
    policy_id = registry.register(spec, name=spec_path.stem)
    assert registry.aliases[spec_path.stem] == policy_id

    commands = academy_commands(layout, iteration=5)
    assert str(spec_path) in commands["play"]
    assert str(layout.train_games_path()) in commands["replay_train"]

"""Smoke tests for real-game `BeliefNet` and `MatchupPredictor` training CLI (`jace.train.belief`)."""

from pathlib import Path

import torch

from jace.collect.decks import elemental_decks
from jace.train.belief import BeliefExperimentConfig, run_belief_experiment
from jace.train.belief_stream import BeliefStreamSpec
from jace.train.full_data import FullDataConfig, run_full_data_training


def _write_tiny_deck_catalog(path: Path) -> None:
    decks = [deck.main for _, deck in elemental_decks()[:4]]
    names = tuple(sorted({name for d in decks for name in d}))
    idx = {name: i for i, name in enumerate(names)}
    mat = torch.zeros((len(decks), len(names)), dtype=torch.uint8)
    for r, deck in enumerate(decks):
        for name, qty in deck.items():
            mat[r, idx[name]] = int(qty)
    split_key = torch.tensor([0, 1, 5000, 6000], dtype=torch.int64)
    payload = {
        "names": list(names),
        "tensors": {"decks": mat, "pools": mat.clone(), "split_key": split_key},
    }
    torch.save(payload, path)


def test_run_belief_experiment_trains_and_saves_artifacts(tmp_path: Path) -> None:
    decks_path = tmp_path / "tiny_decks.pt"
    _write_tiny_deck_catalog(decks_path)
    out_dir = tmp_path / "out"
    summary = run_belief_experiment(
        BeliefExperimentConfig(
            decks_path=decks_path,
            out_dir=out_dir,
            format_id="tiny-test",
            checkpoint=None,
            train_games=2,
            val_games=2,
            collect_workers=1,
            sample_stride=3,
            max_actions=60,
            hidden_dim=16,
            epochs=2,
            batch_size=16,
            v0_epochs=2,
            device="cpu",
            seed=17,
            arena_deals=0,
        )
    )
    belief_dir = out_dir / "belief" / "tiny-test"
    assert (belief_dir / "prior.pt").is_file()
    assert (belief_dir / "belief_net.pt").is_file()
    assert (belief_dir / "v0.pt").is_file()
    assert (out_dir / "experiment_summary.json").is_file()
    assert isinstance(summary["train_rows"], int)
    assert summary["train_rows"] > 0
    assert isinstance(summary["val_rows"], int)
    assert summary["val_rows"] > 0
    belief_val = summary["belief_val"]
    assert isinstance(belief_val, dict)
    assert {"unconditioned_mean_prior", "bayesian_pool_prior", "trained_belief_net"} <= set(belief_val)


def test_full_data_live_belief_stream_and_policy_aux_memory(tmp_path: Path) -> None:
    decks_path = tmp_path / "tiny_decks.pt"
    _write_tiny_deck_catalog(decks_path)
    layout = run_full_data_training(
        FullDataConfig(
            runs_dir=tmp_path / "runs",
            experiment="test-belief-stream",
            run_id="smoke-aux",
            decks_path=decks_path,
            updates=2,
            num_envs=4,
            steps_per_env=8,
            minibatch_size=16,
            hidden_dim=16,
            attention_layers=1,
            decision_memory="none",
            collect_workers=0,
            native_workers=2,
            collect_device="cpu",
            learn_device="cpu",
            eval_every=2,
            eval_deals=1,
            eval_workers=1,
            forensics_deals=1,
            probe_count=2,
            record_train_fraction=1.0,
            snapshot_fraction=0.0,
            dense_early_checkpoints=False,
            belief_stream=BeliefStreamSpec(
                enabled=True,
                every=1,
                steps_per_update=2,
                batch_size=16,
                hidden_dim=16,
                eval_games=2,
                policy_aux_weight=0.1,
                policy_aux_batch_size=8,
            ),
            seed=23,
        )
    )
    belief_dir = layout.root / "belief"
    assert (belief_dir / "prior.pt").is_file()
    assert (belief_dir / "belief_net.pt").is_file()
    assert (belief_dir / "v0.pt").is_file()
    metrics_lines = layout.metrics_path.read_text(encoding="utf-8").splitlines()
    assert any('"kind": "belief_stream"' in line for line in metrics_lines)
    assert any('"belief_policy_aux_loss"' in line for line in metrics_lines)
    assert any('"kind": "belief_eval"' in line for line in metrics_lines)

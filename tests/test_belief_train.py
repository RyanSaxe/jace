"""Smoke tests for real-game `BeliefNet` and `MatchupPredictor` training CLI (`jace.train.belief`)."""

from pathlib import Path

import torch

from jace.collect.decks import elemental_decks
from jace.train.belief import BeliefExperimentConfig, run_belief_experiment


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

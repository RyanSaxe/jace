"""Unit tests for 17Lands human-deck loading, even-odds matching, gradient agreement, and resume.

Verifies deterministic held-out splitting, card-pool subset filtering, mirrored deal
building, and AlphaStar-style even-odds snapshot weights on synthetic decks without
touching production datasets. Also exercises cosine gradient-agreement tracking and
atomic checkpoint-resume round-trips on tiny models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch import nn

from jace.eval.human_decks import (
    build_human_deals,
    build_human_deck_source,
    load_human_deck_catalog,
    row_to_deck,
    sample_human_decks,
    top_card_pool,
)
from jace.loops.collect_pool import MultiProcessCollector, WorkerPoolConfig
from jace.loops.train_ten_decks import build_agent
from jace.rl.diagnostics import GradientAgreementTracker, snapshot_win_rate
from jace.rl.pool import SnapshotPool
from jace.runs import create_run, load_resume_state, save_policy_checkpoint, save_resume_state

if TYPE_CHECKING:
    from pathlib import Path


def _write_synthetic_decks_pt(path: Path) -> Path:
    names = ["Plains", "Mountain", "CardA", "CardB", "CardC"]
    decks = torch.tensor(
        [
            [8, 9, 4, 0, 0],
            [9, 8, 3, 1, 0],
            [8, 9, 2, 2, 0],
            [9, 8, 1, 0, 3],
            [8, 9, 4, 0, 0],
            [9, 8, 0, 2, 2],
            [8, 9, 3, 1, 0],
            [9, 8, 2, 0, 2],
            [8, 9, 4, 0, 0],
            [9, 8, 1, 3, 0],
        ],
        dtype=torch.uint8,
    )
    pools = decks.clone()
    pools[:, 4] += 1
    split_key = torch.tensor([10, 50, 150, 200, 300, 400, 500, 600, 700, 800], dtype=torch.int16)
    torch.save(
        {
            "names": names,
            "tensors": {
                "decks": decks,
                "pools": pools,
                "wins": torch.zeros(10, dtype=torch.int16),
                "losses": torch.zeros(10, dtype=torch.int16),
                "split_key": split_key,
            },
        },
        path,
    )
    return path


def test_human_deck_catalog_split_filter_and_deals(tmp_path: Path) -> None:
    cache = _write_synthetic_decks_pt(tmp_path / "decks.pt")
    catalog = load_human_deck_catalog(cache, held_out_share=0.1)
    assert catalog.card_names == ("Plains", "Mountain", "CardA", "CardB", "CardC")
    assert catalog.held_out_decks.shape == (2, 5)
    assert catalog.train_decks.shape == (8, 5)

    first = row_to_deck(catalog.train_decks[0], catalog.train_sideboards[0], card_names=catalog.card_names)
    assert first.main["Plains"] == 8
    assert first.sideboard["CardC"] == 1

    top3 = top_card_pool(catalog, card_count=3)
    assert {"Plains", "Mountain", "CardA"} == top3
    filtered = sample_human_decks(catalog, split="train", count=2, seed=7, allowed_cards=top3)
    assert len(filtered) == 2
    for _, deck in filtered:
        assert set(deck.main).issubset(top3)

    deals = build_human_deals(catalog, split="held_out", count=1, seed=42)
    assert len(deals) == 1
    assert deals[0].decklists == "closed"
    assert deals[0].deck_a_name != deals[0].deck_b_name

    source = build_human_deck_source(catalog, count=4, seed=9)
    assert len(source.decks) == 4
    assert source.decklists == "closed"


def test_even_odds_snapshot_pool_weighting_and_state_roundtrip() -> None:
    pool = SnapshotPool(max_size=4, matching="even_odds", ema_alpha=1.0)
    m1 = nn.Linear(2, 2)
    m2 = nn.Linear(2, 2)
    pool.add(m1, update=10)
    pool.add(m2, update=20)

    # Mark snapshot 10 as a 95% blowout and snapshot 20 as an even 50% match.
    pool.record_outcome(10, win_rate=0.95)
    pool.record_outcome(20, win_rate=0.50)
    weights = pool.sampling_weights()
    assert weights[1] > 4.0 * weights[0]
    assert np.isclose(weights.sum(), 1.0)

    exported = pool.export_state()
    restored = SnapshotPool(max_size=4, matching="even_odds", ema_alpha=1.0)
    restored.restore_state(exported)
    assert restored.updates == (10, 20)
    assert restored.win_rates == pytest.approx({10: 0.95, 20: 0.50})


def test_gradient_agreement_tracker_and_snapshot_win_rate() -> None:
    tracker = GradientAgreementTracker()
    param = nn.Parameter(torch.zeros(4))

    tracker.begin_update()
    unused = nn.Parameter(torch.zeros(2))
    param.grad = torch.tensor([1.0, 0.0, 0.0, 0.0])
    tracker.accumulate([param, unused])
    assert tracker.finish_update() is None

    tracker.begin_update()
    param.grad = torch.tensor([2.0, 0.0, 0.0, 0.0])
    unused.grad = torch.zeros(2)
    tracker.accumulate([param, unused])
    assert tracker.finish_update() == pytest.approx(1.0)

    tracker.begin_update()
    param.grad = torch.tensor([-3.0, 0.0, 0.0, 0.0])
    unused.grad = None
    tracker.accumulate([param, unused])
    assert tracker.finish_update() == pytest.approx(-1.0)

    rewards = np.array([0.0, 1.0, 0.0, -1.0, 0.0], dtype=np.float32)
    dones = np.array([False, True, False, True, True], dtype=bool)
    assert snapshot_win_rate(rewards, dones) == pytest.approx(0.5)


def test_resume_state_roundtrip(tmp_path: Path) -> None:
    agent, obs_space = build_agent(hidden_dim=16, attention_layers=1)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    pool = SnapshotPool(max_size=4, matching="even_odds")
    pool.add(agent, update=10)
    pool.record_outcome(10, win_rate=0.6)

    layout = create_run(runs_dir=tmp_path, experiment="exp", run_id="r1", config={"ok": True})
    save_policy_checkpoint(agent, obs_space=obs_space, layout=layout, iteration=20)
    save_resume_state(layout, iteration=20, optimizer=optimizer, snapshot_pool=pool)

    fresh_agent, _ = build_agent(hidden_dim=16, attention_layers=1)
    fresh_opt = torch.optim.Adam(fresh_agent.parameters(), lr=5e-4)
    fresh_pool = SnapshotPool(max_size=4, matching="even_odds")
    completed = load_resume_state(layout, agent=fresh_agent, optimizer=fresh_opt, snapshot_pool=fresh_pool)

    assert completed == 20
    assert fresh_pool.updates == (10,)
    assert fresh_pool.win_rates[10] == pytest.approx(0.52)
    for key, val in agent.state_dict().items():
        assert torch.allclose(fresh_agent.state_dict()[key], val)


def test_multiprocess_collector_surfaces_worker_error(tmp_path: Path) -> None:
    cfg = WorkerPoolConfig(
        num_workers=1,
        envs_per_worker=1,
        steps_per_env=4,
        hidden_dim=16,
        attention_layers=1,
        human_decks_path=tmp_path / "missing-decks.pt",
    )
    with pytest.raises(RuntimeError, match="FileNotFoundError"):
        MultiProcessCollector(cfg)

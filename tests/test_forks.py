"""Behavior tests for Jace v2's counterfactual fork loss and parallel fork collector.

Verifies that regularized pairwise fork cross-entropy produces a full unvanishing gradient
even when a collapsed policy puts 99.9% probability on the losing action, skips tied forks
with zero gradient, and leaves un-forked actions untouched on wider spaces. Also smoke-tests
closed-deck `managym.Env.clone_env()` collection and updating across spawned CPU workers so
regressions are caught in seconds without touching MPS.
"""

from __future__ import annotations

import torch

from jace.loops.forks import ForkCollectorPool, ForkConfig, build_fork_aux_loss
from jace.loops.train_ten_decks import build_agent
from jace.rl import fork_pairwise_loss, fork_targets_from_returns


def test_fork_targets_filter_ties_and_smooth_decisive_gaps() -> None:
    pair_returns = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.0, 1.0]])
    targets, weights = fork_targets_from_returns(pair_returns, temperature=0.25, min_gap=0.25)
    assert float(weights[0].item()) == 0.0
    assert float(weights[1].item()) == 0.5
    assert float(weights[2].item()) == 1.0
    assert float(targets[1, 1].item()) > 0.85
    assert float(targets[2, 1].item()) > 0.98


def test_fork_pairwise_loss_unfreezes_collapsed_binary_policy() -> None:
    # Saturated binary attack logit: p(attack=0) = 0.999, p(hold=1) = 0.001.
    logits = torch.tensor([[6.9, 0.0, -5.0]], requires_grad=True)
    legal_mask = torch.tensor([[True, True, False]])
    action_pairs = torch.tensor([[0, 1]])
    # Counterfactual playouts prove holding back (action 1) won 4-0 over attacking (action 0).
    pair_returns = torch.tensor([[0.0, 1.0]])

    breakdown = fork_pairwise_loss(
        logits,
        action_pairs=action_pairs,
        pair_returns=pair_returns,
        legal_mask=legal_mask,
        temperature=0.25,
        min_gap=0.25,
        anchor_kl=0.0,
    )
    assert breakdown.active_count == 1
    assert breakdown.mean_gap == 1.0
    breakdown.total.backward()
    assert logits.grad is not None
    # Cross-entropy gradient is (p - target) ~= +0.98 on losing action 0 and -0.98 on winning action 1.
    assert float(logits.grad[0, 0].item()) > 0.9
    assert float(logits.grad[0, 1].item()) < -0.9
    assert float(logits.grad[0, 2].item()) == 0.0


def test_fork_pairwise_loss_zeroes_gradient_when_all_forks_tie() -> None:
    logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
    breakdown = fork_pairwise_loss(
        logits,
        action_pairs=torch.tensor([[0, 1]]),
        pair_returns=torch.tensor([[0.75, 0.75]]),
        legal_mask=torch.tensor([[True, True]]),
        min_gap=0.25,
    )
    assert breakdown.active_count == 0
    breakdown.total.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) == 0.0


def test_fork_collector_and_update_smoke_on_closed_decks() -> None:
    torch.manual_seed(7)
    agent, _ = build_agent(hidden_dim=32, attention_layers=1)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    cfg = ForkConfig(
        forks_per_update=2,
        playouts_per_action=2,
        decision_kinds=("declare_attacker",),
        sample_prob=1.0,
        num_workers=2,
        max_steps=400,
        min_gap=0.0,
    )
    with ForkCollectorPool(hidden_dim=32, attention_layers=1, num_workers=2) as pool:
        examples = pool.collect(agent, config=cfg, seed=101)
    assert len(examples) == 2
    assert all(item.decision_kind == "declare_attacker" for item in examples)
    assert all(item.num_actions == 2 for item in examples)
    assert all(item.valid_pairs >= 1 for item in examples)

    aux_fn, stats = build_fork_aux_loss(agent, examples=examples, config=cfg, device=torch.device("cpu"))
    assert stats.total_forks == 2
    assert 0 <= stats.decisive_forks <= 2
    if aux_fn is not None:
        agent.train()
        loss = aux_fn()
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

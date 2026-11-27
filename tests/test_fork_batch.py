"""Unit tests for cross-fork batched rollouts, continuation value reuse, and horizon audits.

Verifies that `BatchedForkCollectorPool` harvests roots across closed-decklist deals,
steps all active branches together, and collects on-policy continuation WDL value targets.
Also checks that joint policy-plus-value losses backpropagate cleanly and that the paired
horizon-vs-full-game correlation audit finishes without errors on tiny models.
"""

from __future__ import annotations

from typing import NoReturn

import torch

from jace.collect.agent import build_agent
from jace.collect.decks import elemental_decks
from jace.eval.deals import build_deals
from jace.forks.audit import audit_horizon_correlation, build_batched_fork_aux_loss, parse_fork_kinds
from jace.forks.batch import BatchedForkCollectorPool, BatchedForkConfig, safe_clone, safe_step


def test_batched_fork_pool_collects_examples_value_samples_and_backprops() -> None:
    agent, _ = build_agent(hidden_dim=16, attention_layers=1)
    cfg = BatchedForkConfig(
        forks_per_update=4,
        playouts_per_action=2,
        decision_kinds=None,
        sample_prob=0.5,
        num_workers=2,
        max_steps=600,
        value_samples_per_fork=8,
        min_gap=0.0,
        fork_weight=0.1,
        value_reuse_weight=0.25,
    )
    with BatchedForkCollectorPool(hidden_dim=16, attention_layers=1, num_workers=2) as pool:
        result = pool.collect(agent, config=cfg, seed=7)

    assert len(result.examples) > 0
    assert len(result.value_samples) > 0
    aux_fn, stats = build_batched_fork_aux_loss(agent, result=result, config=cfg, device=torch.device("cpu"))
    assert stats.total_forks == len(result.examples)
    assert aux_fn is not None

    agent.train()
    loss = aux_fn()
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    has_grad = any(
        param.grad is not None and float(param.grad.abs().sum().item()) > 0.0 for param in agent.parameters()
    )
    assert has_grad


def test_horizon_correlation_audit_runs_cleanly_on_tiny_agent() -> None:
    agent, obs_space = build_agent(hidden_dim=16, attention_layers=1)
    deals = build_deals(elemental_decks(), count=6, seed=11, decklists="closed")
    summaries = audit_horizon_correlation(
        agent,
        obs_space,
        deals=deals,
        horizons=(4, 8),
        playouts_per_action=2,
        max_forks=3,
        decision_kinds=None,
        seed=11,
    )
    assert [s.horizon for s in summaries] == [4, 8]
    for summary in summaries:
        assert -1.0 <= summary.pearson_r <= 1.0
        assert 0.0 <= summary.sign_agreement <= 1.0


def test_safe_step_and_clone_isolate_native_panic_exception() -> None:
    class PanicException(BaseException):
        pass

    class PanickingEnv:
        def step(self, action: int) -> NoReturn:
            msg = f"native rust panic on action {action}"
            raise PanicException(msg)

        def is_game_over(self) -> bool:
            return False

        def clone_env(self) -> NoReturn:
            msg = "native rust clone panic"
            raise PanicException(msg)

    env = PanickingEnv()
    obs, is_done, is_failed = safe_step(env, 0)
    assert obs is None
    assert is_done is True
    assert is_failed is True
    assert safe_clone(env) is None


def test_parse_fork_kinds_handles_all_and_csv() -> None:
    assert parse_fork_kinds(None) is None
    assert parse_fork_kinds("all") is None
    assert parse_fork_kinds("  ") is None
    assert parse_fork_kinds("declare_attacker") == ("declare_attacker",)
    assert parse_fork_kinds("declare_attacker, declare_blocker") == ("declare_attacker", "declare_blocker")

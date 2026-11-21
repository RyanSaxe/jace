"""Behavior tests for Jace v2 self-play RL math and update loop.

Exercises credit assignment (GAE with split policy/value lambdas, categorical WDL
lambda-returns, episode boundary isolation, and Monte Carlo), advantage selection,
and damped policy plus PPO losses (legal masking, float64 ratio stability, and
policy-only masking with full-batch critic learning). Also runs one complete
self-play update over real v11 `manabot` rollouts to verify end-to-end integration.
"""

from __future__ import annotations

import managym
import numpy as np
import pytest
import torch
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.training.decks import UniformDecks

from jace.rl import (
    PolicyBatch,
    TrainConfig,
    Trajectory,
    UpdateConfig,
    all_rows,
    damped_policy_loss,
    gae_wdl,
    monte_carlo,
    policy_log_probs,
    ppo_loss,
    reverse_kl,
    top_fraction,
    train_self_play,
)


def test_gae_wdl_cuts_credit_strictly_at_episode_boundaries() -> None:
    # Two consecutive 2-step games packed in one column: game 1 wins (+1), game 2 loses (-1).
    rewards = torch.tensor([[0.0], [1.0], [0.0], [-1.0]])
    values = torch.tensor([[0.1], [0.2], [-0.1], [-0.2]])
    dones = torch.tensor([[False], [True], [False], [True]])
    next_value = torch.tensor([0.9])  # Must be ignored because last row is terminal.
    outcome_probs = torch.tensor(
        [
            [[0.3, 0.2, 0.5]],
            [[0.2, 0.2, 0.6]],
            [[0.5, 0.2, 0.3]],
            [[0.6, 0.2, 0.2]],
        ]
    )
    next_outcomes = torch.tensor([[0.0, 0.0, 1.0]])

    credit = gae_wdl(gamma=1.0, lam_policy=0.5, lam_value=0.8)
    targets = credit(
        Trajectory(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
            outcome_probs=outcome_probs,
            next_outcome_probs=next_outcomes,
        )
    )

    # Row 1 (terminal win): delta = 1.0 - 0.2 = 0.8 (no leak from row 2!).
    assert pytest.approx(0.8, abs=1e-5) == float(targets.raw_advantages[1, 0])
    # Row 0: delta_0 = 0 + 0.2 - 0.1 = 0.1; A_0 = 0.1 + 0.5 * 0.8 = 0.5.
    assert pytest.approx(0.5, abs=1e-5) == float(targets.raw_advantages[0, 0])
    # Row 3 (terminal loss): delta = -1.0 - (-0.2) = -0.8.
    assert pytest.approx(-0.8, abs=1e-5) == float(targets.raw_advantages[3, 0])
    # WDL terminal targets are exact one-hot win ([0,0,1]) and loss ([1,0,0]).
    assert torch.allclose(targets.value_targets[1, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(targets.value_targets[3, 0], torch.tensor([1.0, 0.0, 0.0]))
    # Row 0 WDL target mixes 0.2 * p_1 + 0.8 * terminal_win = [0.04, 0.04, 0.92].
    assert torch.allclose(targets.value_targets[0, 0], torch.tensor([0.04, 0.04, 0.92]), atol=1e-5)


def test_monte_carlo_matches_terminal_outcome() -> None:
    rewards = torch.tensor([[0.0], [0.0], [1.0]])
    values = torch.tensor([[0.2], [0.4], [0.6]])
    dones = torch.tensor([[False], [False], [True]])
    next_value = torch.tensor([0.0])
    targets = monte_carlo(Trajectory(rewards=rewards, values=values, dones=dones, next_value=next_value))
    # Every step's Monte Carlo return is +1.0, so advantage is 1.0 - V(s).
    assert torch.allclose(targets.raw_advantages[:, 0], torch.tensor([0.8, 0.6, 0.4]), atol=1e-5)
    assert torch.allclose(targets.value_targets[:, 0], torch.tensor([1.0, 1.0, 1.0]), atol=1e-5)


def test_selection_filters_keep_expected_rows() -> None:
    adv = torch.tensor([0.05, -0.9, 0.4, -0.1, 0.8, -0.2, 0.3, -0.7])
    assert bool(all_rows(adv).all())
    mask = top_fraction(0.25)(adv)
    # Top 25% of 8 rows = 2 rows: indices 1 (|-0.9|) and 4 (|0.8|).
    assert mask.nonzero(as_tuple=False).flatten().tolist() == [1, 4]


def test_damped_and_ppo_losses_mask_illegal_and_backpropagate() -> None:
    logits = torch.tensor([[1.0, 0.0, 99.0], [0.5, -0.5, 99.0]], requires_grad=True)
    value_logits = torch.tensor([[0.1, 0.0, 0.2], [-0.1, 0.0, 0.3]], requires_grad=True)
    legal = torch.tensor([[True, True, False], [True, True, False]])
    beh_logs = policy_log_probs(logits.detach(), legal)
    ref = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    batch = PolicyBatch(
        logits=logits,
        value_output=value_logits,
        actions=torch.tensor([0, 1]),
        advantages=torch.tensor([1.0, -1.0]),
        value_targets=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        behavior_logs=beh_logs,
        reference_probs=ref,
        legal_mask=legal,
        policy_mask=torch.tensor([True, False]),
    )

    damped = damped_policy_loss(clip=0.2, collection_kl=0.5, magnet_kl=0.05, value_coef=0.5)(batch)
    assert torch.isfinite(damped.total)
    # Identical current & behavior logits => collection KL is 0 and clip fraction is 0.
    assert pytest.approx(0.0, abs=1e-5) == damped.collection_kl
    assert pytest.approx(0.0, abs=1e-5) == damped.clip_fraction
    damped.total.backward()
    assert logits.grad is not None
    # Illegal slot (index 2) must receive strictly zero gradient.
    assert torch.all(logits.grad[:, 2] == 0.0)
    # Unselected policy row (row 1) receives zero policy gradient, while value head still learns both rows.
    assert torch.all(logits.grad[1] == 0.0)
    assert value_logits.grad is not None
    assert float(value_logits.grad[1].abs().sum()) > 0.0

    # Reverse KL against identical distribution is zero.
    assert torch.allclose(reverse_kl(beh_logs, beh_logs, legal_mask=legal), torch.zeros(2), atol=1e-6)

    # PPO loss runs cleanly on the same PolicyBatch.
    logits2 = logits.detach().clone().requires_grad_()
    val2 = value_logits.detach().clone().requires_grad_()
    ppo = ppo_loss(clip=0.2, entropy_coef=0.01, value_coef=0.5)(
        PolicyBatch(
            logits=logits2,
            value_output=val2,
            actions=batch.actions,
            advantages=batch.advantages,
            value_targets=batch.value_targets,
            behavior_logs=batch.behavior_logs,
            reference_probs=batch.reference_probs,
            legal_mask=batch.legal_mask,
            policy_mask=batch.policy_mask,
        )
    )
    assert torch.isfinite(ppo.total)
    ppo.total.backward()
    assert logits2.grad is not None


def test_losses_run_on_mps_when_available() -> None:
    if not torch.backends.mps.is_available():
        return
    dev = torch.device("mps")
    logits = torch.tensor([[1.0, 0.0, 99.0], [0.5, -0.5, 99.0]], device=dev, requires_grad=True)
    value_logits = torch.tensor([[0.1, 0.0, 0.2], [-0.1, 0.0, 0.3]], device=dev, requires_grad=True)
    legal = torch.tensor([[True, True, False], [True, True, False]], device=dev)
    beh_logs = policy_log_probs(logits.detach(), legal)
    ref = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], device=dev)
    batch = PolicyBatch(
        logits=logits,
        value_output=value_logits,
        actions=torch.tensor([0, 1], device=dev),
        advantages=torch.tensor([1.0, -1.0], device=dev),
        value_targets=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], device=dev),
        behavior_logs=beh_logs,
        reference_probs=ref,
        legal_mask=legal,
        policy_mask=torch.tensor([True, True], device=dev),
    )
    out = damped_policy_loss()(batch)
    assert torch.isfinite(out.total)
    out.total.backward()
    assert logits.grad is not None
    assert value_logits.grad is not None


def test_train_self_play_runs_end_to_end_on_real_v11_rollout() -> None:
    keys = list(managym.authored_deck_keys("elemental-world-v2"))[:2]
    decks = tuple(
        managym.Deck(dict(setup.decklist), dict(setup.sideboard))
        for setup in (managym.authored_deck_setup("elemental-world-v2", key) for key in keys)
    )
    source = UniformDecks(decks=decks, decklists="closed", sampling="uniform-independent-v1")
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
    agent = ActorCritic(obs_space, model_cfg, bucketed_shapes=True, live_tokens=False)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

    history = train_self_play(
        agent,
        optimizer,
        obs_space=obs_space,
        deck_source=source,
        credit=gae_wdl(gamma=1.0, lam_policy=0.5, lam_value=0.8),
        select=top_fraction(0.25),
        loss=damped_policy_loss(clip=0.2, collection_kl=0.5, magnet_kl=0.05),
        config=TrainConfig(
            updates=1,
            num_envs=2,
            steps_per_env=8,
            seed=7,
            collect_device="cpu",
            update=UpdateConfig(epochs=1, minibatch_size=64, device="cpu"),
        ),
    )
    assert len(history) == 1
    stats = history[0]
    assert stats.rows > 0
    assert stats.completed_games >= 2
    assert 0 < stats.selected_rows <= stats.rows
    assert np.isfinite(stats.policy_loss)
    assert np.isfinite(stats.value_loss)
    assert np.isfinite(stats.grad_norm)

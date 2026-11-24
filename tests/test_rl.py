"""Behavior tests for Jace v2 self-play RL math, schedules, snapshot pool, and loop.

Exercises credit assignment (GAE with split policy/value lambdas, categorical WDL
lambda-returns, episode boundary isolation, and Monte Carlo), advantage selection,
and damped policy plus PPO losses with legal masking and float64 ratio stability.
Also verifies power-law LR/tau schedules, CPU snapshot pooling, unnormalized advantage
scale, and end-to-end self-play plus frozen historical snapshot rollouts on v11 games.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import managym
import numpy as np
import pytest
import torch
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig, RewardConfig
from manabot.training.decks import UniformDecks
from manabot.training.rollout.collection import RolloutBatch, close_collector, collect, open_collector
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

from jace.collect.agent import merge_rollout_batches
from jace.rl import (
    DampedSchedule,
    DecaySchedule,
    LossBreakdown,
    LossFn,
    PolicyBatch,
    SnapshotPool,
    StepRecord,
    TrainConfig,
    Trajectory,
    UpdateConfig,
    all_rows,
    bucketed_minibatch_order,
    damped_policy_loss,
    gae_wdl,
    load_snapshot,
    monte_carlo,
    policy_log_probs,
    ppo_loss,
    reverse_kl,
    top_fraction,
    train_self_play,
)

_collect_rollout: Callable[..., RolloutBatch] = collect


def test_gae_wdl_cuts_credit_strictly_at_episode_boundaries() -> None:
    rewards = torch.tensor([[0.0], [1.0], [0.0], [-1.0]])
    values = torch.tensor([[0.1], [0.2], [-0.1], [-0.2]])
    dones = torch.tensor([[False], [True], [False], [True]])
    next_value = torch.tensor([0.9])
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

    assert pytest.approx(0.8, abs=1e-5) == float(targets.raw_advantages[1, 0])
    assert pytest.approx(0.5, abs=1e-5) == float(targets.raw_advantages[0, 0])
    assert pytest.approx(-0.8, abs=1e-5) == float(targets.raw_advantages[3, 0])
    assert torch.allclose(targets.value_targets[1, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(targets.value_targets[3, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(targets.value_targets[0, 0], torch.tensor([0.04, 0.04, 0.92]), atol=1e-5)


def test_monte_carlo_matches_terminal_outcome() -> None:
    rewards = torch.tensor([[0.0], [0.0], [1.0]])
    values = torch.tensor([[0.2], [0.4], [0.6]])
    dones = torch.tensor([[False], [False], [True]])
    next_value = torch.tensor([0.0])
    targets = monte_carlo(Trajectory(rewards=rewards, values=values, dones=dones, next_value=next_value))
    assert torch.allclose(targets.raw_advantages[:, 0], torch.tensor([0.8, 0.6, 0.4]), atol=1e-5)
    assert torch.allclose(targets.value_targets[:, 0], torch.tensor([1.0, 1.0, 1.0]), atol=1e-5)


def test_selection_filters_keep_expected_rows() -> None:
    adv = torch.tensor([0.05, -0.9, 0.4, -0.1, 0.8, -0.2, 0.3, -0.7])
    assert bool(all_rows(adv).all())
    mask = top_fraction(0.25)(adv)
    assert mask.nonzero(as_tuple=False).flatten().tolist() == [1, 4]
    strict = top_fraction(0.5, minimum=0.75)(adv)
    assert strict.nonzero(as_tuple=False).flatten().tolist() == [1, 4]


def test_damped_schedule_decays_monotonically_within_bounds() -> None:
    schedule = DampedSchedule(total_updates=150)
    lr1, tau1 = schedule.rates(1)
    lr75, tau75 = schedule.rates(75)
    lr150, tau150 = schedule.rates(150)

    assert pytest.approx(1e-4) == lr1
    assert pytest.approx(0.05) == tau1
    assert lr1 >= lr75 > lr150 >= schedule.lr_min
    assert tau1 > tau75 > tau150 > 0.0
    assert pytest.approx(0.5 / (10_000**1.1), rel=1e-5) == lr150
    assert pytest.approx(0.05 / (10_000**0.3), rel=1e-5) == tau150


def test_decay_schedule_decays_cleanly_across_horizon() -> None:
    schedule = DecaySchedule(total_updates=75, lr_initial=2.5e-4, tau_initial=0.01, decay=9.0)
    lr1, tau1 = schedule.rates(1)
    lr38, tau38 = schedule.rates(38)
    lr75, tau75 = schedule.rates(75)

    assert pytest.approx(2.5e-4) == lr1
    assert pytest.approx(0.01) == tau1
    assert lr1 > lr38 > lr75
    assert tau1 > tau38 > tau75
    assert pytest.approx(2.5e-5, rel=1e-5) == lr75
    assert pytest.approx(0.001, rel=1e-5) == tau75


def test_snapshot_pool_clones_and_bounds_history() -> None:
    layer = nn.Linear(3, 2)
    pool = SnapshotPool(max_size=2)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    pool.add(layer, update=10)
    with torch.no_grad():
        layer.weight.fill_(2.0)
    pool.add(layer, update=20)
    with torch.no_grad():
        layer.weight.fill_(3.0)
    pool.add(layer, update=30)

    assert len(pool) == 2
    assert pool.updates == (20, 30)

    target = nn.Linear(3, 2)
    sampled = pool.sample(np.random.default_rng(0))
    load_snapshot(target, sampled)
    expected = 2.0 if sampled.update == 20 else 3.0
    assert torch.allclose(target.weight, torch.full_like(target.weight, expected))


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
        advantages=torch.tensor([0.05, -0.05]),
        value_targets=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        behavior_logs=beh_logs,
        reference_probs=ref,
        legal_mask=legal,
        policy_mask=torch.tensor([True, False]),
    )

    damped = damped_policy_loss(clip=0.2, collection_kl=0.1, magnet_kl=0.05, value_coef=1.0)(batch)
    assert torch.isfinite(damped.total)
    assert pytest.approx(0.0, abs=1e-5) == damped.collection_kl
    assert pytest.approx(0.0, abs=1e-5) == damped.clip_fraction
    damped.total.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad[:, 2] == 0.0)
    assert torch.all(logits.grad[1] == 0.0)
    assert value_logits.grad is not None
    assert float(value_logits.grad[1].abs().sum()) > 0.0

    assert torch.allclose(reverse_kl(beh_logs, beh_logs, legal_mask=legal), torch.zeros(2), atol=1e-6)

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
        advantages=torch.tensor([0.05, -0.05], device=dev),
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


def test_train_self_play_supports_unnormalized_damped_schedule_and_snapshot_pool() -> None:
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
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    schedule = DampedSchedule(total_updates=2)
    seen_adv_max: list[float] = []
    records: list[StepRecord] = []

    def scheduled_loss(update_idx: int) -> tuple[float, float, LossFn]:
        lr, tau = schedule.rates(update_idx)
        base = damped_policy_loss(clip=0.2, collection_kl=0.1, magnet_kl=tau, value_coef=1.0)

        def wrapped(batch: PolicyBatch) -> LossBreakdown:
            seen_adv_max.append(float(batch.advantages.abs().max().item()))
            return base(batch)

        return lr, tau, wrapped

    history = train_self_play(
        agent,
        optimizer,
        obs_space=obs_space,
        deck_source=source,
        credit=gae_wdl(gamma=1.0, lam_policy=0.5, lam_value=0.8),
        select=top_fraction(0.25, minimum=0.01),
        loss=damped_policy_loss(),
        loss_schedule=scheduled_loss,
        config=TrainConfig(
            updates=2,
            num_envs=4,
            steps_per_env=8,
            seed=7,
            collect_device="cpu",
            update=UpdateConfig(
                epochs=1,
                minibatch_size=64,
                max_grad_norm=0.267,
                normalize_advantages=False,
                device="cpu",
            ),
            snapshot_fraction=0.5,
            snapshot_every=1,
            snapshot_pool_size=4,
        ),
        on_update=records.append,
    )
    assert len(history) == 2
    assert records[0].opponent == "self"
    assert records[1].opponent == "snapshot@u1"
    assert records[0].magnet_tau > records[1].magnet_tau
    assert records[0].learning_rate >= records[1].learning_rate
    # Unnormalized win-probability advantages stay bounded near [-2, 2], never z-scored to extreme std=1 spikes.
    assert seen_adv_max
    assert max(seen_adv_max) <= 2.5
    for stats in history:
        assert stats.rows > 0
        assert stats.completed_games >= 4
        assert np.isfinite(stats.policy_loss)
        assert np.isfinite(stats.value_loss)
        assert np.isfinite(stats.grad_norm)


def test_bucketed_minibatch_order_groups_similar_sizes_and_preserves_permutation() -> None:
    rng = np.random.default_rng(123)
    sizes = np.array([10] * 16 + [200] * 16, dtype=np.int64)
    order = bucketed_minibatch_order(len(sizes), minibatch_size=16, row_sizes=sizes, rng=rng)
    assert sorted(order.tolist()) == list(range(len(sizes)))
    # Each 16-row bucket should contain purely short or purely long rows, never mixed.
    first_bucket = sizes[order[:16]]
    second_bucket = sizes[order[16:]]
    assert {int(first_bucket.max()), int(second_bucket.max())} == {10, 200}
    assert {int(first_bucket.min()), int(second_bucket.min())} == {10, 200}


def test_merge_rollout_batches_preserves_rows_and_packed_actions() -> None:
    obs_space = ObservationSpace(
        ObservationConfig(policy_input_version=11, max_cards_per_player=40, max_permanents_per_player=32)
    )
    agent = ActorCritic(
        obs_space,
        ModelConfig(
            world="world",
            hidden_dim=32,
            attention_layers=1,
            num_attention_heads=4,
            policy_film=True,
            value_kind="categorical_wdl",
            value_aggregation="masked_mean",
        ),
        bucketed_shapes=True,
        live_tokens=False,
    ).eval()
    deck = managym.Deck({"Mountain": 24, "Llanowar Elves": 16})
    source = UniformDecks(decks=(deck, deck), decklists="closed", sampling="uniform-independent-v1")
    reward = Reward(RewardConfig())
    batches = []
    for seed in (1, 2):
        collector = open_collector(
            obs_space,
            Match(source.match_config()),
            reward,
            num_envs=2,
            seed=seed,
            opponent_mode="self",
            numerics=CollectionSession(device="cpu", seed=seed),
            learning_seats="both",
            batch_closure="complete-games",
            deck_source=source,
        )
        try:
            batches.append(_collect_rollout(collector, agent, 4))
        finally:
            close_collector(collector)

    merged = merge_rollout_batches(batches)
    expected_rows = sum(int(batch.actions.size) for batch in batches)
    assert int(merged.actions.size) == expected_rows
    assert int(merged.dones.sum()) == sum(int(batch.dones.sum()) for batch in batches)
    np.testing.assert_array_equal(merged.actions, np.concatenate([b.actions for b in batches], axis=0))
    obs_last = merged.observations(np.array([expected_rows - 1], dtype=np.int64))
    obs_orig = batches[-1].observations(np.array([int(batches[-1].actions.size) - 1], dtype=np.int64))
    np.testing.assert_array_equal(obs_last.log_probabilities, obs_orig.log_probabilities)


def test_epoch_catalog_cache_updates_semantic_card_weights() -> None:
    obs_space = ObservationSpace(
        ObservationConfig(policy_input_version=11, max_cards_per_player=40, max_permanents_per_player=32)
    )
    agent = ActorCritic(
        obs_space,
        ModelConfig(
            world="world",
            hidden_dim=32,
            attention_layers=1,
            num_attention_heads=4,
            policy_film=True,
            value_kind="categorical_wdl",
            value_aggregation="masked_mean",
        ),
        bucketed_shapes=True,
        live_tokens=False,
    )
    assert agent.semantic_cards is not None
    before = {name: param.detach().clone() for name, param in agent.semantic_cards.named_parameters()}
    deck = managym.Deck({"Mountain": 24, "Llanowar Elves": 16})
    source = UniformDecks(decks=(deck, deck), decklists="closed", sampling="uniform-independent-v1")
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    train_self_play(
        agent,
        optimizer,
        obs_space=obs_space,
        deck_source=source,
        credit=gae_wdl(gamma=1.0, lam_policy=0.95, lam_value=0.9),
        select=all_rows,
        loss=damped_policy_loss(),
        config=TrainConfig(
            updates=1,
            num_envs=2,
            steps_per_env=6,
            seed=3,
            collect_device="cpu",
            update=UpdateConfig(epochs=1, minibatch_size=64, max_grad_norm=0.267, device="cpu"),
        ),
    )
    deltas = [
        float((param.detach() - before[name]).abs().max().item())
        for name, param in agent.semantic_cards.named_parameters()
    ]
    assert max(deltas) > 0.0
    assert all(np.isfinite(delta) for delta in deltas)

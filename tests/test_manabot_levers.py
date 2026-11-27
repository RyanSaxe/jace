"""Behavioral tests for manabot training levers: filter_scope, EMA, GRU memory, and tau schedule.

Verifies `filter_scope="actor_critic"` row restriction and empty-filter fallback during
real `update_step` calls alongside `DampedSchedule(tau_scale=0.10)` endpoint decay.
Also tests `EmaWeights` exponential smoothing, backwards-compatible atomic resume
round-trips that keep live and shadow weights distinct, and `decision_memory="gru"`
forward/backward updates with compact `MemoryArchive` IPC batch merging and live eval.
"""

from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING

import managym
import numpy as np
import pytest
import torch
from manabot.backends.torch.collection import CollectionSession
from manabot.env.match import Match, Reward
from manabot.infra.config import RewardConfig
from manabot.training.decks import UniformDecks
from manabot.training.rollout.collection import RolloutBatch, close_collector, collect, open_collector
from torch import nn

from jace.collect.agent import (
    DECISION_MEMORY_CAPACITIES,
    build_agent,
    pack_batch_for_ipc,
    unpack_batch_from_ipc,
)
from jace.eval import PolicyPlayer, build_deals, heuristic_player, play_match
from jace.rl import (
    DampedSchedule,
    EmaWeights,
    LossBreakdown,
    PolicyBatch,
    SnapshotPool,
    UpdateConfig,
    damped_policy_loss,
    gae_wdl,
    top_fraction,
    training_row_indexes,
    update_step,
)
from jace.storage.gameplay import create_run, load_resume_state, save_policy_checkpoint, save_resume_state
from jace.train.full_data import _parse_args, resolve_live_tokens, resolve_precision

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_collect_rollout: Callable[..., RolloutBatch] = collect


def _tiny_deck_source() -> UniformDecks:
    deck = managym.Deck({"Mountain": 24, "Llanowar Elves": 16})
    return UniformDecks(decks=(deck, deck), decklists="closed", sampling="uniform-independent-v1")


def test_damped_schedule_tau_scale_0_10_endpoints_and_monotonic_decay() -> None:
    schedule = DampedSchedule(total_updates=1800, lr_max=1e-4, tau_scale=0.10)
    lr1, tau1 = schedule.rates(1)
    lr900, tau900 = schedule.rates(900)
    lr1800, tau1800 = schedule.rates(1800)

    assert pytest.approx(1e-4) == lr1
    assert pytest.approx(0.10) == tau1
    assert lr1 >= lr900 > lr1800 >= schedule.lr_min
    assert tau1 > tau900 > tau1800 > 0.0
    assert pytest.approx(0.5 / (10_000**1.1), rel=1e-5) == lr1800
    assert pytest.approx(0.10 / (10_000**0.3), rel=1e-5) == tau1800


def test_filter_scope_actor_critic_restricts_minibatches_and_falls_back_when_empty() -> None:
    mask = torch.tensor([False, True, False, True, False, False])
    np.testing.assert_array_equal(training_row_indexes(mask, filter_scope="actor"), np.arange(6, dtype=np.int64))
    np.testing.assert_array_equal(
        training_row_indexes(mask, filter_scope="actor_critic"), np.array([1, 3], dtype=np.int64)
    )
    empty_mask = torch.zeros(6, dtype=torch.bool)
    np.testing.assert_array_equal(
        training_row_indexes(empty_mask, filter_scope="actor_critic"), np.arange(6, dtype=np.int64)
    )

    agent, obs_space = build_agent(hidden_dim=32, attention_layers=1)
    source = _tiny_deck_source()
    collector = open_collector(
        obs_space,
        Match(source.match_config()),
        Reward(RewardConfig()),
        num_envs=2,
        seed=5,
        opponent_mode="self",
        numerics=CollectionSession(device="cpu", seed=5),
        learning_seats="both",
        batch_closure="complete-games",
        deck_source=source,
    )
    try:
        batch = _collect_rollout(collector, agent.eval(), 6)
    finally:
        close_collector(collector)

    base_loss = damped_policy_loss()
    seen_actor: list[int] = []
    seen_ac: list[int] = []
    seen_ac_all_selected: list[bool] = []
    seen_fallback: list[int] = []

    def _recording_loss(counts: list[int], flags: list[bool] | None = None) -> Callable[[PolicyBatch], LossBreakdown]:
        def _loss(policy_batch: PolicyBatch) -> LossBreakdown:
            counts.append(int(policy_batch.logits.shape[0]))
            if flags is not None and policy_batch.policy_mask is not None:
                flags.append(bool(policy_batch.policy_mask.all().item()))
            return base_loss(policy_batch)

        return _loss

    agent_actor = copy.deepcopy(agent)
    stats_actor = update_step(
        agent_actor,
        torch.optim.Adam(agent_actor.parameters(), lr=1e-4),
        batch=batch,
        credit=gae_wdl(gamma=1.0, lam_policy=0.5, lam_value=0.8),
        select=top_fraction(0.25, minimum=0.0),
        loss=_recording_loss(seen_actor),
        config=UpdateConfig(epochs=1, minibatch_size=64, filter_scope="actor", device="cpu"),
        rng=np.random.default_rng(11),
    )

    agent_ac = copy.deepcopy(agent)
    stats_ac = update_step(
        agent_ac,
        torch.optim.Adam(agent_ac.parameters(), lr=1e-4),
        batch=batch,
        credit=gae_wdl(gamma=1.0, lam_policy=0.5, lam_value=0.8),
        select=top_fraction(0.25, minimum=0.0),
        loss=_recording_loss(seen_ac, seen_ac_all_selected),
        config=UpdateConfig(epochs=1, minibatch_size=64, filter_scope="actor_critic", device="cpu"),
        rng=np.random.default_rng(11),
    )

    assert sum(seen_actor) == stats_actor.rows
    assert sum(seen_ac) == stats_ac.selected_rows
    assert stats_ac.selected_rows < stats_ac.rows
    assert seen_ac_all_selected
    assert all(seen_ac_all_selected)

    agent_empty = copy.deepcopy(agent)
    stats_empty = update_step(
        agent_empty,
        torch.optim.Adam(agent_empty.parameters(), lr=1e-4),
        batch=batch,
        credit=gae_wdl(gamma=1.0, lam_policy=0.5, lam_value=0.8),
        select=top_fraction(0.25, minimum=10.0),
        loss=_recording_loss(seen_fallback),
        config=UpdateConfig(epochs=1, minibatch_size=64, filter_scope="actor_critic", device="cpu"),
        rng=np.random.default_rng(11),
    )
    assert stats_empty.selected_rows == 0
    assert sum(seen_fallback) == stats_empty.rows
    assert np.isfinite(stats_empty.value_loss)


def test_ema_weights_smooths_parameters_and_copies_to_eval_module() -> None:
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.fill_(0.0)
    with pytest.raises(ValueError, match="decay"):
        EmaWeights(layer, decay=1.0)

    ema_layer = EmaWeights(layer, decay=0.9, warmup=False)
    assert ema_layer.warmup is False
    assert ema_layer.effective_decay() == pytest.approx(0.9)
    with torch.no_grad():
        layer.weight.fill_(3.0)
        layer.bias.fill_(2.0)
    ema_layer.update(layer)
    assert ema_layer.updates == 1
    assert torch.allclose(layer.weight, torch.full_like(layer.weight, 3.0))

    eval_copy = nn.Linear(3, 2)
    ema_layer.copy_to(eval_copy)
    assert torch.allclose(eval_copy.weight, torch.full_like(eval_copy.weight, 1.2))
    assert torch.allclose(eval_copy.bias, torch.full_like(eval_copy.bias, 0.2))


def test_ema_weights_warmup_sheds_init_approaches_decay_and_roundtrips_state() -> None:
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.fill_(0.0)

    ema = EmaWeights(layer, decay=0.999)
    assert ema.warmup is True
    assert ema.effective_decay() == pytest.approx(0.1)

    with torch.no_grad():
        layer.weight.fill_(3.0)
        layer.bias.fill_(2.0)
    ema.update(layer)

    eval_copy = nn.Linear(3, 2)
    ema.copy_to(eval_copy)
    # First update uses decay=0.1, shedding 90% of initial parameters immediately.
    assert torch.allclose(eval_copy.weight, torch.full_like(eval_copy.weight, 2.8))
    assert torch.allclose(eval_copy.bias, torch.full_like(eval_copy.bias, 1.8))

    decays = [ema.effective_decay()]
    for _ in range(89):
        ema.update(layer)
        decays.append(ema.effective_decay())
    assert ema.updates == 90
    assert all(prev < nxt for prev, nxt in itertools.pairwise(decays))
    assert decays[8] == pytest.approx(10.0 / 19.0)
    assert ema.effective_decay() == pytest.approx(0.91)

    capped = EmaWeights(layer, decay=0.5)
    for _ in range(9):
        capped.update(layer)
    assert capped.effective_decay() == pytest.approx(0.5)

    # export_state/restore_state round-trip warmup, and omit-warmup payloads keep existing setting.
    fixed_state = EmaWeights(layer, decay=0.9, warmup=False).export_state()
    assert fixed_state["warmup"] is False
    restored = EmaWeights(layer, decay=0.999, warmup=True)
    restored.restore_state(fixed_state)
    assert restored.warmup is False
    assert restored.effective_decay() == pytest.approx(0.9)

    legacy_payload = {k: v for k, v in ema.export_state().items() if k != "warmup"}
    keep_true = EmaWeights(layer, decay=0.999, warmup=True)
    keep_true.restore_state(legacy_payload)
    assert keep_true.warmup is True
    assert keep_true.updates == 90
    assert keep_true.effective_decay() == pytest.approx(0.91)

    keep_false = EmaWeights(layer, decay=0.999, warmup=False)
    keep_false.restore_state(legacy_payload)
    assert keep_false.warmup is False
    assert keep_false.effective_decay() == pytest.approx(0.999)


def test_ema_resume_roundtrip_preserves_live_and_shadow_weights(tmp_path: Path) -> None:
    # Full ActorCritic resume round-trip keeps live weights and EMA weights distinct.
    agent, obs_space = build_agent(hidden_dim=16, attention_layers=1)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    pool = SnapshotPool(max_size=4, matching="even_odds")
    ema = EmaWeights(agent, decay=0.9)
    with torch.no_grad():
        for param in agent.parameters():
            param.add_(1.0)
    ema.update(agent)

    eval_agent = copy.deepcopy(agent).to("cpu").eval()
    ema.copy_to(eval_agent)
    layout = create_run(runs_dir=tmp_path, experiment="exp", run_id="ema-r1", config={"ema": 0.9})
    save_policy_checkpoint(eval_agent, obs_space=obs_space, layout=layout, iteration=10)
    save_resume_state(layout, iteration=10, optimizer=optimizer, snapshot_pool=pool, agent=agent, ema=ema)

    fresh_agent, _ = build_agent(hidden_dim=16, attention_layers=1)
    fresh_opt = torch.optim.Adam(fresh_agent.parameters(), lr=5e-4)
    fresh_pool = SnapshotPool(max_size=4, matching="even_odds")
    fresh_ema = EmaWeights(fresh_agent, decay=0.5, warmup=False)
    completed = load_resume_state(
        layout, agent=fresh_agent, optimizer=fresh_opt, snapshot_pool=fresh_pool, ema=fresh_ema
    )
    assert completed == 10
    assert fresh_ema.decay == pytest.approx(0.9)
    assert fresh_ema.warmup is True
    assert fresh_ema.updates == 1
    for key, live_val in agent.state_dict().items():
        assert torch.allclose(fresh_agent.state_dict()[key], live_val)
    fresh_eval = copy.deepcopy(fresh_agent)
    fresh_ema.copy_to(fresh_eval)
    for key, ema_val in eval_agent.state_dict().items():
        assert torch.allclose(fresh_eval.state_dict()[key], ema_val)

    # Backwards compatibility: loading a resume file saved without ema_state syncs EMA to agent.
    legacy_layout = create_run(runs_dir=tmp_path, experiment="exp", run_id="legacy", config={"ema": None})
    save_policy_checkpoint(agent, obs_space=obs_space, layout=legacy_layout, iteration=5)
    save_resume_state(legacy_layout, iteration=5, optimizer=optimizer, snapshot_pool=pool)
    legacy_agent, _ = build_agent(hidden_dim=16, attention_layers=1)
    legacy_ema = EmaWeights(legacy_agent, decay=0.999)
    done = load_resume_state(
        legacy_layout,
        agent=legacy_agent,
        optimizer=torch.optim.Adam(legacy_agent.parameters(), lr=1e-3),
        snapshot_pool=None,
        ema=legacy_ema,
    )
    assert done == 5
    synced_eval = copy.deepcopy(legacy_agent)
    legacy_ema.copy_to(synced_eval)
    for key, val in agent.state_dict().items():
        assert torch.allclose(synced_eval.state_dict()[key], val)


def test_decision_memory_gru_ipc_merge_update_step_and_eval() -> None:
    agent, obs_space = build_agent(hidden_dim=32, attention_layers=1, decision_memory="gru")
    assert agent.decision_memory is not None
    assert obs_space.config.max_decision_records == DECISION_MEMORY_CAPACITIES[0]

    source = _tiny_deck_source()
    batches: list[RolloutBatch] = []
    for seed in (1, 2):
        collector = open_collector(
            obs_space,
            Match(source.match_config()),
            Reward(RewardConfig()),
            num_envs=2,
            seed=seed,
            opponent_mode="self",
            numerics=CollectionSession(device="cpu", seed=seed),
            learning_seats="both",
            batch_closure="complete-games",
            deck_source=source,
        )
        try:
            raw = _collect_rollout(collector, agent.eval(), 4)
        finally:
            close_collector(collector)
        packed, archive = pack_batch_for_ipc(raw)
        assert archive is not None
        batches.append(unpack_batch_from_ipc(packed, archive))

    merged = RolloutBatch.concatenate(batches)
    assert merged.recorded.memory_capacities == DECISION_MEMORY_CAPACITIES
    assert len(merged.recorded.memory) == sum(len(b.recorded.memory) for b in batches)

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    stats = update_step(
        agent,
        optimizer,
        batch=merged,
        credit=gae_wdl(gamma=1.0, lam_policy=0.5, lam_value=0.8),
        select=top_fraction(0.25, minimum=0.0),
        loss=damped_policy_loss(),
        config=UpdateConfig(epochs=1, minibatch_size=64, filter_scope="actor_critic", device="cpu"),
        rng=np.random.default_rng(7),
    )
    assert stats.rows == int(merged.actions.size)
    assert stats.selected_rows > 0
    assert np.isfinite(stats.policy_loss)
    assert np.isfinite(stats.value_loss)
    assert np.isfinite(stats.grad_norm)

    deals = build_deals([("a", managym.Deck({"Mountain": 24, "Llanowar Elves": 16}))], count=1, seed=3)
    with PolicyPlayer(agent.eval(), obs_space, seed=3, deterministic=True) as player:
        assert player.needs_decision_memory
        score = play_match(player, heuristic_player(), deals=deals, max_actions=200)
    assert score.games == 2


def test_train_full_data_cli_parses_manabot_lever_flags(tmp_path: Path) -> None:
    cfg = _parse_args(
        [
            "--runs-dir",
            str(tmp_path),
            "--decks-path",
            str(tmp_path / "decks.pt"),
            "--run-id",
            "r69-levers",
            "--select",
            "top25",
            "--filter-scope",
            "actor_critic",
            "--schedule-kind",
            "damped",
            "--magnet-kl",
            "0.10",
            "--ema",
            "0.999",
            "--decision-memory",
            "gru",
        ]
    )
    assert cfg.select_kind == "top25"
    assert cfg.filter_scope == "actor_critic"
    assert cfg.schedule_kind == "damped"
    assert cfg.magnet_kl == pytest.approx(0.10)
    assert cfg.ema == pytest.approx(0.999)
    assert cfg.decision_memory == "gru"


def test_manabot_026_speedup_flags_and_live_tokens_stream_norm_update(tmp_path: Path) -> None:
    assert resolve_precision("auto", device="cuda") == "bf16"
    assert resolve_precision("auto", device="cuda:0") == "bf16"
    assert resolve_precision("auto", device="mps") == "float32"
    assert resolve_precision("auto", device="cpu") == "float32"
    assert resolve_precision("tf32", device="cuda") == "tf32"
    assert resolve_live_tokens(live_tokens=None, device="cuda") is True
    assert resolve_live_tokens(live_tokens=None, device="mps") is False
    assert resolve_live_tokens(live_tokens=False, device="cuda") is False

    cfg = _parse_args(
        [
            "--runs-dir",
            str(tmp_path),
            "--decks-path",
            str(tmp_path / "decks.pt"),
            "--run-id",
            "r82-speedups",
            "--stream-norm",
            "--focus-film",
            "--live-tokens",
            "--precision",
            "bf16",
        ]
    )
    assert cfg.stream_norm is True
    assert cfg.focus_film is True
    assert cfg.live_tokens is True
    assert cfg.precision == "bf16"

    agent, obs_space = build_agent(
        hidden_dim=32,
        attention_layers=1,
        stream_norm=True,
        focus_film=True,
        live_tokens=True,
    )
    assert agent.config.stream_norm is True
    assert agent.config.focus_film is True

    source = _tiny_deck_source()
    collector = open_collector(
        obs_space,
        Match(source.match_config()),
        Reward(RewardConfig()),
        num_envs=2,
        seed=9,
        opponent_mode="self",
        numerics=CollectionSession(device="cpu", seed=9),
        learning_seats="both",
        batch_closure="complete-games",
        deck_source=source,
    )
    try:
        batch = _collect_rollout(collector, agent.eval(), 4)
    finally:
        close_collector(collector)

    stats = update_step(
        agent,
        torch.optim.Adam(agent.parameters(), lr=1e-4),
        batch=batch,
        credit=gae_wdl(gamma=1.0, lam_policy=0.5, lam_value=0.8),
        select=top_fraction(0.5, minimum=0.0),
        loss=damped_policy_loss(),
        config=UpdateConfig(epochs=1, minibatch_size=64, device="cpu", precision="float32"),
        rng=np.random.default_rng(13),
    )
    assert stats.rows > 0
    assert np.isfinite(stats.policy_loss)
    assert np.isfinite(stats.value_loss)
    assert np.isfinite(stats.grad_norm)

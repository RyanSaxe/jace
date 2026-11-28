from pathlib import Path
from types import SimpleNamespace

import managym
import pytest
import torch
from manabot.backends.torch.checkpoint import save_checkpoint
from managym.enums import ZoneEnum

from jace.belief import BeliefOverride, BeliefState, fit_format_prior
from jace.collect.agent import build_agent
from jace.collect.decks import elemental_decks
from jace.eval.arena import (
    SearchConfigSummary,
    ledger_path_for,
    load_match_records,
    player_id_for,
    run_arena_match,
)
from jace.eval.deals import Deal, build_deals
from jace.forks.arena import fit_arena_format_prior, run_search_arena_match
from jace.forks.arena import main as search_arena_main
from jace.forks.search import (
    SearchConfig,
    build_belief_from_obs,
    extract_public_belief_inputs,
    extract_revealed_hand_cards,
    run_local_search,
    sample_opponent_worlds,
)


def _advance_to_branchable_priority(
    deal: Deal,
) -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=deal.seed, skip_trivial=True)
    obs, _ = env.reset(deal.game_setup())
    for _ in range(80):
        if env.is_game_over():
            break
        if len(obs.action_space.actions) >= 2 and int(obs.opponent.zone_counts[int(ZoneEnum.HAND)]) >= 2:
            return env, obs
        obs, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    return env, obs


def test_sample_opponent_worlds_respects_closed_decklists_dynamic_hand_and_overrides() -> None:
    deal = build_deals(elemental_decks()[:2], count=1, seed=11, decklists="closed")[0]
    env, obs = _advance_to_branchable_priority(deal)
    hero_seat = int(obs.agent.player_index)
    opp_seat = int(obs.opponent.player_index)
    prior = fit_arena_format_prior(format_id="elemental-test")

    revealed, signals, opp_hand_size, opp_lib_size = extract_public_belief_inputs(
        obs,
        card_names=prior.card_names,
        hero_seat=hero_seat,
    )
    assert signals.shape == (16,)
    assert opp_hand_size == int(obs.opponent.zone_counts[int(ZoneEnum.HAND)])
    assert opp_lib_size == int(obs.opponent.zone_counts[int(ZoneEnum.LIBRARY)])

    belief, rev_out = build_belief_from_obs(obs, prior=prior, hero_seat=hero_seat, prior_only=True)
    assert torch.equal(rev_out, revealed)
    assert belief.opp_hand_size == opp_hand_size

    # Choose one card known to exist in opponent's deck to pin and one to exclude.
    opp_deck = deal.deck_a if opp_seat == 0 else deal.deck_b
    deck_cards = [name for name in sorted(opp_deck) if name in prior.card_names]
    pin_card = deck_cards[0]
    exclude_card = deck_cards[1]

    # Materialize worlds with pinned card and excluded card.
    override = BeliefOverride(pinned_in_hand={pin_card: 1}, excluded_cards=frozenset({exclude_card}))
    worlds = sample_opponent_worlds(
        env,
        belief,
        opp_seat=opp_seat,
        k=3,
        seed=42,
        revealed_counts=revealed,
        overrides=override,
    )
    assert len(worlds) == 3
    for world in worlds:
        snap = world.hand_snapshot(opp_seat).cards
        assert sum(snap.values()) == opp_hand_size
        assert snap.get(pin_card, 0) >= 1
        assert snap.get(exclude_card, 0) == 0

    # Verify dynamic hand sizes strictly above 7 cards (no 7-card cap).
    big_hand_size = 9
    deck_scale = float(big_hand_size + opp_lib_size) / max(1.0, float(belief.deck_copies.sum()))
    hand_scale = float(big_hand_size) / max(1.0, float(belief.hand_probs.sum()))
    big_belief = BeliefState(
        card_names=belief.card_names,
        deck_copies=belief.deck_copies * deck_scale,
        hand_probs=belief.hand_probs * hand_scale,
        opp_hand_size=big_hand_size,
        opp_library_size=opp_lib_size,
        prior_backend=belief.prior_backend,
    )
    big_worlds = sample_opponent_worlds(env, big_belief, opp_seat=opp_seat, k=2, seed=99)
    assert len(big_worlds) == 2
    for world in big_worlds:
        assert sum(world.hand_snapshot(opp_seat).cards.values()) == big_hand_size

    # Over-pinning beyond opp_hand_size must raise ValueError.
    over_pinned = BeliefOverride(pinned_in_hand={pin_card: opp_hand_size + 1})
    with pytest.raises(ValueError, match="exceeds opp_hand_size"):
        sample_opponent_worlds(
            env,
            belief,
            opp_seat=opp_seat,
            k=1,
            seed=7,
            revealed_counts=revealed,
            overrides=over_pinned,
        )


def test_run_local_search_truncated_and_terminal() -> None:
    deal = build_deals(elemental_decks()[:2], count=1, seed=19, decklists="closed")[0]
    env, obs = _advance_to_branchable_priority(deal)
    hero_seat = int(obs.agent.player_index)
    prior = fit_arena_format_prior(format_id="elemental-test")
    belief, revealed = build_belief_from_obs(obs, prior=prior, hero_seat=hero_seat, prior_only=True)

    agent, obs_space = build_agent(
        hidden_dim=16,
        attention_layers=1,
        decision_memory="none",
        bucketed_shapes=True,
        live_tokens=False,
    )
    agent.eval()
    num_actions = len(obs.action_space.actions)
    assert num_actions >= 2

    with agent.inference_scope() as scope:
        trunc_res = run_local_search(
            env,
            obs,
            scope=scope,
            obs_space=obs_space,
            belief=belief,
            config=SearchConfig(worlds=2, horizon_plies=4, max_plies=16, temperature=0.0, prior_only=True),
            seed=123,
            revealed_counts=revealed,
        )
        assert 0 <= trunc_res.chosen_action < num_actions
        assert len(trunc_res.action_values) == num_actions
        assert len(trunc_res.action_ci95) == num_actions
        assert len(trunc_res.policy_priors) == num_actions
        assert trunc_res.worlds_sampled == 2
        assert trunc_res.rollout_horizon == 4
        assert trunc_res.elapsed_ms > 0.0

        term_res = run_local_search(
            env,
            obs,
            scope=scope,
            obs_space=obs_space,
            belief=belief,
            config=SearchConfig(worlds=2, horizon_plies=None, max_plies=12, temperature=0.0, prior_only=True),
            seed=123,
            revealed_counts=revealed,
        )
        assert 0 <= term_res.chosen_action < num_actions
        assert len(term_res.action_values) == num_actions
        assert term_res.worlds_sampled == 2
        assert term_res.rollout_horizon is None
        assert term_res.elapsed_ms > 0.0


def test_arena_match_wired_search_and_cli_sweep(tmp_path: Path) -> None:
    agent, obs_space = build_agent(
        hidden_dim=16,
        attention_layers=1,
        decision_memory="none",
        bucketed_shapes=True,
        live_tokens=False,
    )
    agent.eval()
    ckpt_path = tmp_path / "tiny-policy.safetensors"
    save_checkpoint(agent, obs_space, ckpt_path, player_configs=None)

    # Verify search_config_a on pure jace.eval.arena.run_arena_match fails clearly without deal_runner,
    # and run_search_arena_match rejects non-checkpoint specs clearly.
    search_cfg = SearchConfigSummary(worlds=2, horizon=4, belief_id="prior-only")
    with pytest.raises(ValueError, match="deal_runner"):
        run_arena_match(
            tmp_path,
            format_id="elemental-search-test",
            model_a="tiny-policy",
            model_b="random",
            deals=1,
            seed=23,
            num_workers=1,
            max_actions=24,
            search_config_a=search_cfg,
            model_a_spec=ckpt_path,
        )
    with pytest.raises(ValueError, match="Search requires a policy checkpoint"):
        run_search_arena_match(
            tmp_path,
            format_id="elemental-search-test",
            model_a="heuristic",
            model_b="random",
            deals=1,
            seed=23,
            num_workers=1,
            search_config_a=search_cfg,
        )

    record = run_search_arena_match(
        tmp_path,
        format_id="elemental-search-test",
        model_a="tiny-policy",
        model_b="random",
        deals=1,
        seed=23,
        num_workers=1,
        max_actions=24,
        search_config_a=search_cfg,
        model_a_spec=ckpt_path,
    )
    assert record.deals == 1
    assert record.wins_a + record.losses_a + record.draws == 2
    assert record.search_config_a == search_cfg
    assert record.ms_per_decision_a > 0.0
    assert player_id_for(record.model_a, record.search_config_a) == "tiny-policy[search:w2,h4,prior-only]"

    loaded = load_match_records(ledger_path_for(tmp_path), format_id="elemental-search-test")
    assert loaded == [record]

    # Exercise CLI multi-world sweep (`python -m jace.forks.arena match --search-worlds 2,3 ...`).
    rc = search_arena_main(
        [
            "match",
            "--runs-dir",
            str(tmp_path),
            "--format",
            "elemental-cli-sweep",
            "--model-a",
            str(ckpt_path),
            "--model-b",
            "random",
            "--deals",
            "1",
            "--seed",
            "29",
            "--num-workers",
            "1",
            "--search-worlds",
            "2,3",
            "--horizon",
            "2",
            "--belief-prior-only",
        ]
    )
    assert rc == 0
    sweep_records = load_match_records(ledger_path_for(tmp_path), format_id="elemental-cli-sweep")
    assert len(sweep_records) == 2
    assert [r.search_config_a for r in sweep_records] == [
        SearchConfigSummary(worlds=2, horizon=2, belief_id="prior-only"),
        SearchConfigSummary(worlds=3, horizon=2, belief_id="prior-only"),
    ]
    assert all(r.ms_per_decision_a > 0.0 for r in sweep_records)
    assert len({r.match_id for r in sweep_records}) == 2


def test_sample_opponent_worlds_preserves_graveyard_and_warns_on_unmaterialized_pin() -> None:
    deal = build_deals(elemental_decks(), count=1, seed=42, decklists="closed")[0]
    env = managym.Env(seed=deal.seed, skip_trivial=True)
    obs, _ = env.reset(deal.game_setup(swapped=False))
    gy_zone = int(ZoneEnum.GRAVEYARD)
    opp_gy_before: list[str] = []
    for _ in range(40):
        if env.is_game_over():
            break
        opp_gy_before = [str(c.characteristics.name) for c in obs.opponent_cards if int(c.zone) == gy_zone]
        if opp_gy_before and int(obs.opponent.zone_counts[int(ZoneEnum.HAND)]) >= 1:
            break
        obs, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    assert opp_gy_before, "expected opponent graveyard card in test game"
    hero_seat, opp_seat = int(obs.agent.player_index), int(obs.opponent.player_index)
    prior = fit_arena_format_prior(format_id="elemental-test")
    belief, revealed = build_belief_from_obs(obs, prior=prior, hero_seat=hero_seat, prior_only=True)

    # Pin a card known to be absent from opp's library (e.g. a card from another deck or graveyard-only copy).
    opp_deck = deal.deck_a if opp_seat == 0 else deal.deck_b
    absent_card = next(name for name in prior.card_names if name not in opp_deck)
    missing: set[str] = set()
    worlds = sample_opponent_worlds(
        env,
        belief,
        opp_seat=opp_seat,
        k=2,
        seed=77,
        revealed_counts=revealed,
        overrides=BeliefOverride(pinned_in_hand={absent_card: 1}),
        unmaterialized_pins=missing,
    )
    assert absent_card in missing
    for world in worlds:
        w_obs = world.observation_for_player(hero_seat)
        opp_gy_after = [str(c.characteristics.name) for c in w_obs.opponent_cards if int(c.zone) == gy_zone]
        assert sorted(opp_gy_after) == sorted(opp_gy_before)


def test_fit_format_prior_roundtrip_smoke() -> None:
    decks = torch.tensor([[4.0, 2.0, 10.0], [1.0, 4.0, 10.0]], dtype=torch.float32)
    prior = fit_format_prior("smoke", ("Card A", "Card B", "Forest"), decks)
    assert prior.card_names == ("Card A", "Card B", "Forest")


def test_extract_revealed_hand_cards_and_auto_pin_in_build_belief() -> None:
    deal = build_deals(elemental_decks()[:2], count=1, seed=11, decklists="closed")[0]
    _env, obs = _advance_to_branchable_priority(deal)
    hero_seat = int(obs.agent.player_index)
    prior = fit_arena_format_prior(format_id="elemental-test")
    assert extract_revealed_hand_cards(obs, hero_seat=hero_seat) == {}

    known_card = prior.card_names[0]
    fake_revealed_card = SimpleNamespace(
        is_token=False,
        zone=int(ZoneEnum.HAND),
        characteristics=SimpleNamespace(name=known_card),
    )
    proxy_obs = SimpleNamespace(
        agent=obs.agent,
        agent_cards=obs.agent_cards,
        opponent_cards=[*obs.opponent_cards, fake_revealed_card],
    )
    assert extract_revealed_hand_cards(proxy_obs, hero_seat=hero_seat) == {known_card: 1}
    belief, revealed = build_belief_from_obs(obs, prior=prior, hero_seat=hero_seat, prior_only=True)
    assert float(belief.hand_probs.sum().item()) == pytest.approx(float(belief.opp_hand_size), abs=1e-4)
    assert revealed.shape == (len(prior.card_names),)

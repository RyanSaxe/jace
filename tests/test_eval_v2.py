"""Behavior tests for mirrored deal evaluation, forensics auditing, and checkpoint diffs.

Exercises `jace.eval` end to end over real `managym` closed-decklist games using the
bundled two-color Limited decks. Verifies mirrored deal scoring with native
`ScalarCapture` JSONL recording, behavioral decision forensics against
`managym.Heuristic`, and deterministic probe-state checkpoint diffs.
"""

from pathlib import Path

from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from managym.recording import read_records

from jace.eval import (
    PolicyPlayer,
    build_deals,
    capture_probes,
    diff_checkpoints,
    elemental_decks,
    heuristic_player,
    play_match,
    random_player,
    run_forensics,
)


def _tiny_agent() -> tuple[ActorCritic, ObservationSpace]:
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
    return ActorCritic(obs_space, model_cfg, bucketed_shapes=True, live_tokens=False), obs_space


def test_mirrored_match_scores_and_records_native_academy_games(tmp_path: Path) -> None:
    decks = elemental_decks()[:2]
    deals = build_deals(decks, count=1, seed=11, decklists="closed")
    games_path = tmp_path / "eval.jsonl"
    score = play_match(
        heuristic_player(),
        random_player(seed=5),
        deals=deals,
        games_path=games_path,
        player_names=("heuristic", "random"),
    )
    assert score.games == 2
    assert score.wins + score.losses + score.draws == 2
    assert 0.0 <= score.ci_low <= score.win_rate <= score.ci_high <= 1.0
    records = list(read_records(games_path))
    assert len(records) == 2
    assert all(rec.games[0].end is not None and rec.games[0].end.status == "complete" for rec in records)


def test_forensics_and_probe_diffs_run_on_real_games(tmp_path: Path) -> None:
    agent_a, obs_space = _tiny_agent()
    agent_b, _ = _tiny_agent()
    deals = build_deals(elemental_decks()[:2], count=1, seed=21, decklists="closed")
    games_path = tmp_path / "forensics.jsonl"

    report = run_forensics(
        agent_a,
        obs_space,
        deals=deals,
        seed=3,
        games_path=games_path,
        max_actions=400,
    )
    assert report.games == 2
    assert report.total_decisions > 0
    assert report.decision_counts_by_kind
    assert 0.0 <= report.attack_rate <= 1.0
    assert 0.0 <= report.block_rate <= 1.0
    assert 0.0 <= report.missed_land_drop_rate <= 1.0
    assert 0.0 <= report.wrong_side_target_rate <= 1.0
    assert len(list(read_records(games_path))) == 2

    probes = capture_probes(deals, max_probes=6, per_deal_cap=6, max_actions=200)
    assert len(probes) > 0
    identical = diff_checkpoints(agent_a, agent_a, obs_space=obs_space, probes=probes, top_k=3)
    assert identical.mean_tv == 0.0
    assert identical.argmax_flip_rate == 0.0

    different = diff_checkpoints(agent_a, agent_b, obs_space=obs_space, probes=probes, top_k=3)
    assert different.probes == len(probes)
    assert different.mean_tv > 0.0
    assert len(different.top_shifts) <= 3

    with PolicyPlayer(agent_a, obs_space, seed=1) as player:
        score = play_match(player, random_player(seed=2), deals=deals, max_actions=200)
        assert score.games == 2

import json
from pathlib import Path

import managym
import torch
from fastapi.testclient import TestClient

from jace.belief import BeliefNet, fit_format_prior, save_belief_net, save_format_prior
from jace.studio import StudioConfig, create_app


def _record_small_game(seed: int = 303) -> dict[str, object]:
    deck_a = {"Plains": 14, "Mountain": 14, "Fire Lord Zuko": 4, "Swamp": 8}
    deck_b = {"Plains": 14, "Mountain": 14, "Boiling Rock Rioter": 4, "Swamp": 8}
    players: list[tuple[str, dict[str, int]]] = [
        ("hero_u400", deck_a),
        ("opp_heuristic", deck_b),
    ]
    env = managym.Env(seed=seed, skip_trivial=True)
    obs, _ = env.reset([managym.PlayerConfig(name, deck) for name, deck in players])
    actions: list[int] = []
    heuristic = managym.Heuristic()
    for _ in range(30):
        chosen = int(heuristic.select_action(obs))
        actions.append(chosen)
        obs, _, done, truncated, _ = env.step(chosen)
        if done or truncated:
            break
    return {
        "skip_trivial": True,
        "games": [
            {
                "seed": seed,
                "players": [{"name": name, "decklist": deck} for name, deck in players],
                "actions": actions,
                "end": {"winner": 0, "turn": int(obs.turn.turn_number)},
            }
        ],
    }


def _seed_studio_run(runs_dir: Path, *, run_id: str = "v2-test/belief-run") -> None:
    run_dir = runs_dir / run_id
    (run_dir / "games").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"hidden_dim": 128, "attention_layers": 2, "updates": 100}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"kind": "eval", "update": 100, "held_out_det_win_rate": 0.65}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "games" / "eval-iter-0100.jsonl").write_text(
        json.dumps(_record_small_game(seed=303)) + "\n",
        encoding="utf-8",
    )


def test_belief_infer_and_project_valid_and_422(tmp_path: Path) -> None:
    _seed_studio_run(tmp_path)
    client = TestClient(create_app(StudioConfig(runs_dir=tmp_path, allow_scryfall_network=False)))

    infer_resp = client.post(
        "/api/belief/infer",
        json={"run_id": "v2-test/belief-run", "iteration": 100, "game_idx": 0, "step": 0},
    )
    assert infer_resp.status_code == 200
    inferred = infer_resp.json()
    assert len(inferred["card_names"]) >= 4
    assert len(inferred["hand_probs"]) == len(inferred["card_names"])
    assert len(inferred["deck_copies"]) == len(inferred["card_names"])
    assert len(inferred["revealed_counts"]) == len(inferred["card_names"])
    assert inferred["opp_hand_size"] >= 0
    assert inferred["opp_library_size"] >= 0
    assert isinstance(inferred["v0_equity"], float)
    assert inferred["v0"] is not None
    assert inferred["v0"]["is_turn0"] is True
    assert inferred["v0"]["recommendation"] in {"Keep", "Mulligan"}

    opp_hand_size = int(inferred["opp_hand_size"])
    pin_card = "Boiling Rock Rioter"
    proj_resp = client.post(
        "/api/belief/project",
        json={
            "run_id": "v2-test/belief-run",
            "iteration": 100,
            "game_idx": 0,
            "step": 0,
            "override": {
                "pinned_in_hand": {pin_card: 1},
                "excluded_cards": [],
                "copy_multipliers": {"Plains": 0.5},
            },
        },
    )
    assert proj_resp.status_code == 200
    projected = proj_resp.json()
    assert projected["pinned_total"] == 1
    assert projected["opp_hand_size"] == opp_hand_size
    assert projected["unedited_count"] == len(inferred["card_names"]) - 2
    assert 0.0 <= projected["unedited_mass"] <= 1.0
    total_hand_prob = sum(card["hand_prob"] for card in projected["cards"])
    assert abs(total_hand_prob - opp_hand_size) < 0.05
    pinned_row = next(card for card in projected["cards"] if card["name"] == pin_card)
    assert pinned_row["pinned"] == 1
    assert pinned_row["hand_prob"] == 1.0

    # Pinning more cards than opp_hand_size must return HTTP 422 with clear detail.
    over_pin = client.post(
        "/api/belief/project",
        json={
            "run_id": "v2-test/belief-run",
            "iteration": 100,
            "game_idx": 0,
            "step": 0,
            "override": {
                "pinned_in_hand": {"Plains": opp_hand_size + 2},
                "excluded_cards": [],
                "copy_multipliers": {},
            },
        },
    )
    assert over_pin.status_code == 422
    assert "detail" in over_pin.json()
    assert "opp_hand_size" in over_pin.json()["detail"]


def test_search_evaluate_and_override_propagation(tmp_path: Path) -> None:
    _seed_studio_run(tmp_path)
    client = TestClient(create_app(StudioConfig(runs_dir=tmp_path, allow_scryfall_network=False)))

    eval_resp = client.post(
        "/api/search/evaluate",
        json={
            "run_id": "v2-test/belief-run",
            "iteration": 100,
            "game_idx": 0,
            "step": 0,
            "ckpt_a": "heuristic",
            "worlds": 4,
            "horizon_plies": 8,
            "seed": 42,
        },
    )
    assert eval_resp.status_code == 200
    body = eval_resp.json()
    assert body["worlds_sampled"] == 4
    assert body["rollout_horizon"] == 8
    assert body["elapsed_ms"] >= 0.0
    assert body["opp_hand_size"] >= 0
    assert body["perspective"] in {"hero", "opponent"}
    assert isinstance(body["warnings"], list)
    assert len(body["actions"]) >= 1
    assert 0 <= body["chosen_action"] < len(body["actions"])
    first = body["actions"][0]
    assert {"index", "label", "search_wr", "ci_low", "ci_high", "policy_prob", "delta_vs_policy"} <= set(first)
    assert 0.0 <= first["ci_low"] <= first["search_wr"] <= first["ci_high"] <= 1.0
    # delta_vs_policy compares search_wr against the raw policy's top-ranked action (same win-rate units).
    top_pol_row = max(body["actions"], key=lambda row: row["policy_prob"])
    assert top_pol_row["delta_vs_policy"] == 0.0

    # Missing named checkpoint returns HTTP 404 (no silent heuristic fallback).
    missing_ckpt = client.post(
        "/api/search/evaluate",
        json={
            "run_id": "v2-test/belief-run",
            "iteration": 100,
            "game_idx": 0,
            "step": 0,
            "ckpt_a": "nonexistent-iter-9999",
            "worlds": 4,
            "horizon_plies": 8,
        },
    )
    assert missing_ckpt.status_code == 404

    # Valid override propagates cleanly into search evaluation.
    with_override = client.post(
        "/api/search/evaluate",
        json={
            "run_id": "v2-test/belief-run",
            "iteration": 100,
            "game_idx": 0,
            "step": 0,
            "ckpt_a": "heuristic",
            "worlds": 4,
            "horizon_plies": 8,
            "seed": 42,
            "override": {
                "pinned_in_hand": {"Boiling Rock Rioter": 1},
                "excluded_cards": ["Swamp"],
                "copy_multipliers": {},
            },
        },
    )
    assert with_override.status_code == 200
    assert with_override.json()["worlds_sampled"] == 4

    # Invalid over-pinned override returns HTTP 422 on /api/search/evaluate too.
    opp_hand = int(body["opp_hand_size"])
    bad_override = client.post(
        "/api/search/evaluate",
        json={
            "run_id": "v2-test/belief-run",
            "iteration": 100,
            "game_idx": 0,
            "step": 0,
            "ckpt_a": "heuristic",
            "worlds": 4,
            "horizon_plies": 8,
            "override": {
                "pinned_in_hand": {"Plains": opp_hand + 3},
                "excluded_cards": [],
                "copy_multipliers": {},
            },
        },
    )
    assert bad_override.status_code == 422
    assert "detail" in bad_override.json()


def test_saved_format_prior_and_invalid_format_id(tmp_path: Path) -> None:
    _seed_studio_run(tmp_path)
    card_names = ("Boiling Rock Rioter", "Fire Lord Zuko", "Mountain", "Plains", "Swamp", "Zuko's Exile")
    decks = torch.tensor(
        [
            [4.0, 2.0, 12.0, 12.0, 8.0, 2.0],
            [2.0, 4.0, 14.0, 10.0, 6.0, 4.0],
        ],
        dtype=torch.float32,
    )
    prior = fit_format_prior("tla-test", card_names, decks)
    save_format_prior(tmp_path / "belief" / "tla-test" / "prior.pt", prior)
    save_belief_net(tmp_path / "belief" / "tla-test" / "belief_net.pt", BeliefNet(hidden_dim=16))

    client = TestClient(create_app(StudioConfig(runs_dir=tmp_path, allow_scryfall_network=False)))
    resp = client.post(
        "/api/belief/infer",
        json={
            "run_id": "v2-test/belief-run",
            "iteration": 100,
            "game_idx": 0,
            "step": 0,
            "format_id": "tla-test",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["card_names"] == list(card_names)
    assert resp.json()["has_belief_net"] is True

    bad_fmt = client.post(
        "/api/belief/infer",
        json={
            "run_id": "v2-test/belief-run",
            "iteration": 100,
            "game_idx": 0,
            "step": 0,
            "format_id": "../escape",
        },
    )
    assert bad_fmt.status_code == 400

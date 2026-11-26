import json
from pathlib import Path

import managym
from fastapi.testclient import TestClient

from jace.studio import StudioConfig, create_app


def _record_small_game(seed: int = 101) -> dict[str, object]:
    deck_a = {"Plains": 12, "Mountain": 12, "Fire Lord Zuko": 4, "Swamp": 12}
    deck_b = {"Plains": 14, "Mountain": 14, "Boiling Rock Rioter": 4, "Swamp": 8}
    players: list[tuple[str, dict[str, int]]] = [
        ("hero_u400", deck_a),
        ("opp_heuristic", deck_b),
    ]
    env = managym.Env(seed=seed, skip_trivial=True)
    obs, _ = env.reset([managym.PlayerConfig(name, deck) for name, deck in players])
    actions: list[int] = []
    heuristic = managym.Heuristic()
    for _ in range(45):
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


def _make_fixture_runs(tmp_path: Path) -> Path:
    run_dir = tmp_path / "v2-test" / "run-alpha"
    (run_dir / "games").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"hidden_dim": 160, "attention_layers": 2, "updates": 400}) + "\n",
        encoding="utf-8",
    )
    metrics_lines = [
        json.dumps({"kind": "train", "update": 1, "tps": 1200.0, "value_loss": 0.5, "entropy": 0.8}),
        json.dumps(
            {
                "kind": "eval",
                "update": 100,
                "live_held_out_det_win_rate": 0.64,
                "held_out_det_win_rate": 0.61,
                "ensemble_held_out_det_win_rate": 0.66,
            }
        ),
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    (run_dir / "games" / "eval-iter-0100.jsonl").write_text(
        json.dumps(_record_small_game(seed=20001)) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_runs_and_duckdb_game_search(tmp_path: Path) -> None:
    runs_dir = _make_fixture_runs(tmp_path)
    client = TestClient(create_app(StudioConfig(runs_dir=runs_dir, allow_scryfall_network=False)))

    runs_resp = client.get("/api/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "v2-test/run-alpha"
    assert runs[0]["latest_live_win"] == 0.64

    detail_resp = client.get("/api/runs/v2-test/run-alpha")
    assert detail_resp.status_code == 200
    assert len(detail_resp.json()["eval_points"]) == 1

    games_resp = client.get("/api/games", params={"card": "Fire Lord Zuko", "result": "win"})
    assert games_resp.status_code == 200
    payload = games_resp.json()
    assert payload["total"] == 1
    assert payload["games"][0]["seed"] == 20001
    assert payload["games"][0]["set_code"] == "TLA"

    # Second refresh should skip already-indexed files.
    refresh = client.post("/api/index/refresh").json()
    assert refresh["indexed_files"] == 0


def test_replay_pov_privacy_and_stateless_fork(tmp_path: Path) -> None:
    runs_dir = _make_fixture_runs(tmp_path)
    client = TestClient(create_app(StudioConfig(runs_dir=runs_dir, allow_scryfall_network=False)))

    overview = client.get(
        "/api/game-detail",
        params={"run_id": "v2-test/run-alpha", "iteration": 100, "game_idx": 0},
    )
    assert overview.status_code == 200
    assert len(overview.json()["timeline"]) >= 1

    hero_pov = client.get(
        "/api/game-step",
        params={"run_id": "v2-test/run-alpha", "iteration": 100, "game_idx": 0, "step": 5, "pov": "hero"},
    ).json()
    assert hero_pov["board"]["opp"]["hand_hidden"] is True
    assert hero_pov["board"]["opp"]["hand"] == []
    assert hero_pov["board"]["opp"]["library_hidden"] is True

    omni_pov = client.get(
        "/api/game-step",
        params={"run_id": "v2-test/run-alpha", "iteration": 100, "game_idx": 0, "step": 5, "pov": "omniscient"},
    ).json()
    assert omni_pov["board"]["opp"]["hand_hidden"] is False
    assert len(omni_pov["board"]["opp"]["hand"]) == omni_pov["board"]["opp"]["zone_counts"]["hand"]

    # Pick the first Priority step in the game for scenario_refresh().
    priority_step = next(
        step_idx
        for step_idx in range(15)
        if client.get(
            "/api/game-step",
            params={"run_id": "v2-test/run-alpha", "iteration": 100, "game_idx": 0, "step": step_idx},
        ).json()["board"]["action_space_kind"]
        == "PRIORITY"
    )
    fork_resp = client.post(
        "/api/fork",
        json={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": priority_step,
            "pov": "omniscient",
            "edits": [
                {"op": "set_life", "seat": 0, "value": 12},
                {"op": "set_life", "seat": 1, "value": 7},
                {"op": "add_battlefield", "seat": 0, "card": "Fire Lord Zuko"},
            ],
        },
    )
    assert fork_resp.status_code == 200
    fork_data = fork_resp.json()
    assert fork_data["board"]["hero"]["life"] == 12
    assert fork_data["board"]["opp"]["life"] == 7
    creature_names = [c["name"] for c in fork_data["board"]["hero"]["creatures"]]
    assert "Fire Lord Zuko" in creature_names


def test_watchlist_and_offline_card_fallback(tmp_path: Path) -> None:
    runs_dir = _make_fixture_runs(tmp_path)
    client = TestClient(create_app(StudioConfig(runs_dir=runs_dir, allow_scryfall_network=False)))

    created = client.post(
        "/api/watchlist",
        json={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": 5,
            "turn": 2,
            "seed": 20001,
            "label": "Zuko combat test",
        },
    ).json()["item"]
    assert len(client.get("/api/watchlist").json()["items"]) == 1
    assert client.delete(f"/api/watchlist/{created['id']}").status_code == 200
    assert client.get("/api/watchlist").json()["items"] == []

    img = client.get("/api/cards/Ally/image")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/svg+xml"
    assert b"Ally" in img.content


def test_replay_divergence_returns_warning_not_500(tmp_path: Path) -> None:
    runs_dir = _make_fixture_runs(tmp_path)
    game_file = runs_dir / "v2-test" / "run-alpha" / "games" / "eval-iter-0100.jsonl"
    payload = json.loads(game_file.read_text(encoding="utf-8"))
    payload["games"][0]["actions"][4] = 9999
    game_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    client = TestClient(create_app(StudioConfig(runs_dir=runs_dir, allow_scryfall_network=False)))
    detail = client.get(
        "/api/game-detail",
        params={"run_id": "v2-test/run-alpha", "iteration": 100, "game_idx": 0},
    )
    assert detail.status_code == 200
    assert detail.json()["replay_warning"] is not None

    step = client.get(
        "/api/game-step",
        params={"run_id": "v2-test/run-alpha", "iteration": 100, "game_idx": 0, "step": 10},
    )
    assert step.status_code == 200
    assert step.json()["board"]["step"] == 4
    assert step.json()["replay_warning"] is not None

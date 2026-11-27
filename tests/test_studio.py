import json
import os
from pathlib import Path

import managym
import torch
from fastapi.testclient import TestClient
from manabot.backends.torch.checkpoint import save_checkpoint
from managym.enums import ActionSpaceEnum

from jace.collect.agent import build_agent
from jace.studio import StudioConfig, create_app
from jace.studio.replay import serialize_observation


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
    assert payload["games"][0]["set_code"] == ""
    assert payload["games"][0]["playable"] is True
    assert "surprise_score" in payload["games"][0]
    assert "turning_point_step" in payload["games"][0]

    facets = client.get("/api/games/facets").json()
    assert facets["total_playable"] == 1
    assert facets["total_unplayable"] == 0
    assert facets["runs"] == ["v2-test/run-alpha"]
    assert "Fire Lord Zuko" in facets["cards"]

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


def test_leg1_seat_swap_and_hero_pov_action_privacy_and_cr304_fork(tmp_path: Path) -> None:
    runs_dir = _make_fixture_runs(tmp_path)
    game_file = runs_dir / "v2-test" / "run-alpha" / "games" / "eval-iter-0100.jsonl"
    rec = _record_small_game(seed=20002)
    rec["context"] = {"players": ["heuristic", "agent"]}
    games_list = rec["games"]
    assert isinstance(games_list, list)
    games_list[0]["end"]["winner"] = 1
    games_list[0]["players"][0]["disclosure"] = "closed"
    games_list[0]["players"][1]["disclosure"] = "closed"
    game_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    client = TestClient(create_app(StudioConfig(runs_dir=runs_dir, allow_scryfall_network=False)))
    games = client.get("/api/games").json()["games"]
    assert games[0]["hero_seat"] == 1
    assert games[0]["result"] == "win"
    reversed_colors = games[0]["hero_colors"][::-1]
    assert client.get("/api/games", params={"colors": reversed_colors}).json()["total"] == 1

    # Find an opponent-acting step (seat 0 acting while hero_seat is 1)
    opp_step = next(
        idx
        for idx in range(25)
        if not client.get(
            "/api/game-step",
            params={"run_id": "v2-test/run-alpha", "iteration": 100, "game_idx": 0, "step": idx, "pov": "hero"},
        ).json()["board"]["actor_is_hero"]
    )
    masked = client.get(
        "/api/game-step",
        params={"run_id": "v2-test/run-alpha", "iteration": 100, "game_idx": 0, "step": opp_step, "pov": "hero"},
    ).json()
    assert masked["board"]["actions_hidden"] is True
    assert masked["board"]["legal_actions"] == []
    assert masked["evaluation"]["primary"]["probs"] == []

    # CR 304.4: Instant/Sorcery cannot be forced onto battlefield in Fork mode
    priority_step = next(
        idx
        for idx in range(25)
        if client.get(
            "/api/game-step",
            params={"run_id": "v2-test/run-alpha", "iteration": 100, "game_idx": 0, "step": idx, "pov": "omniscient"},
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
            "edits": [{"op": "add_battlefield", "seat": 1, "card": "Sold Out"}],
        },
    )
    assert fork_resp.status_code == 200
    assert any("CR 304.4" in w for w in fork_resp.json()["warnings"])


def test_unplayable_games_hidden_by_default_and_any_step_branch_actions(tmp_path: Path) -> None:
    runs_dir = _make_fixture_runs(tmp_path)
    game_file = runs_dir / "v2-test" / "run-alpha" / "games" / "eval-iter-0100.jsonl"
    good = _record_small_game(seed=30001)
    bad = _record_small_game(seed=30002)
    bad_games = bad["games"]
    assert isinstance(bad_games, list)
    bad_games[0]["actions"][2] = 9999
    game_file.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")

    client = TestClient(create_app(StudioConfig(runs_dir=runs_dir, allow_scryfall_network=False)))
    playable_only = client.get("/api/games").json()
    assert playable_only["total"] == 1
    assert playable_only["games"][0]["seed"] == 30001

    facets = client.get("/api/games/facets").json()
    assert facets["total_playable"] == 1
    assert facets["total_unplayable"] == 1
    run_facets = client.get("/api/games/facets", params={"run_id": "v2-test/run-alpha"}).json()
    assert run_facets["total_playable"] == 1
    assert run_facets["total_unplayable"] == 1
    empty_facets = client.get("/api/games/facets", params={"run_id": "v2-test/nonexistent"}).json()
    assert empty_facets["total_playable"] == 0
    assert empty_facets["total_unplayable"] == 0

    # Step 0 (MULLIGAN) finds nearest Priority step scanning forward
    step0 = client.get(
        "/api/game-step",
        params={"run_id": "v2-test/run-alpha", "iteration": 100, "game_idx": 0, "step": 0, "pov": "omniscient"},
    ).json()
    assert step0["nearest_priority_step"] is not None
    assert step0["nearest_priority_step"] > 0

    # Any-step counterfactual action branching works even on step 0 (MULLIGAN, non-Priority),
    # and stage-indexed board edits apply cleanly after branching onto a Priority step
    branch_resp = client.post(
        "/api/fork",
        json={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": 0,
            "pov": "omniscient",
            "edits": [],
            "branch_actions": [0],
        },
    )
    assert branch_resp.status_code == 200
    bdata = branch_resp.json()
    assert len(bdata["branch_events"]) == 1
    assert bdata["board"]["step"] == 1

    # Keep both mulligans (2 branch actions -> Turn 1 PRIORITY) then set Hero life to 12 at after_branch=2
    post_branch_edit = client.post(
        "/api/fork",
        json={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": 0,
            "pov": "omniscient",
            "edits": [{"op": "set_life", "seat": 0, "value": 12, "after_branch": 2}],
            "branch_actions": [0, 0],
        },
    )
    assert post_branch_edit.status_code == 200
    pdata = post_branch_edit.json()
    assert pdata["warnings"] == []
    assert pdata["board"]["hero"]["life"] == 12

    # Hero POV privacy: branching from Opponent's decision to Hero's decision still hides base_evaluation probs
    opp_step = next(
        idx
        for idx in range(15)
        if not client.get(
            "/api/game-step",
            params={"run_id": "v2-test/run-alpha", "iteration": 100, "game_idx": 0, "step": idx, "pov": "hero"},
        ).json()["board"]["actor_is_hero"]
    )
    priv_resp = client.post(
        "/api/fork",
        json={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": opp_step,
            "pov": "hero",
            "edits": [],
            "branch_actions": [0],
        },
    )
    assert priv_resp.status_code == 200
    assert priv_resp.json()["base_evaluation"]["primary"]["probs"] == []


def _save_tiny_checkpoint(path: Path, *, seed: int) -> None:
    torch.manual_seed(seed)
    agent, obs_space = build_agent(hidden_dim=16, attention_layers=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(agent, obs_space, path, player_configs=None)


def test_dynamic_metrics_config_tags_truthful_counts_and_global_checkpoints(tmp_path: Path) -> None:
    run_a = tmp_path / "exp-alpha" / "run-one"
    run_b = tmp_path / "exp-beta" / "run-two"
    for d in (run_a, run_b):
        (d / "games").mkdir(parents=True)
        (d / "checkpoints").mkdir(parents=True)

    (run_a / "config.json").write_text(
        json.dumps(
            {
                "hidden_dim": 64,
                "attention_layers": 2,
                "updates": 200,
                "custom_format": "cube",
                "seed_pool": 19,
                "use_ema": True,
                "nested_blob": {"ignored": 1},
                "data_path": "/abs/path/ignored",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_b / "config.json").write_text(
        json.dumps({"updates": 50, "deck_mode": "constructed"}) + "\n",
        encoding="utf-8",
    )

    metrics_a = [
        json.dumps({"kind": "train", "update": 1, "tps": 800.0, "grad_norm": 1.4, "timestamp": 123}),
        json.dumps(
            {
                "kind": "eval",
                "update": 50,
                "custom_elo": 1520.5,
                "ensemble_held_out_det_win_rate": 0.72,
            }
        ),
        json.dumps(
            {
                "kind": "eval",
                "update": 100,
                "custom_elo": 1610.0,
                "ensemble_held_out_det_win_rate": 0.68,
            }
        ),
    ]
    (run_a / "metrics.jsonl").write_text("\n".join(metrics_a) + "\n", encoding="utf-8")
    (run_b / "metrics.jsonl").write_text(
        json.dumps({"kind": "train", "update": 50, "policy_kl": 0.04})
        + "\n"
        + json.dumps({"kind": "eval", "update": 50, "mirror_win": 0.55})
        + "\n",
        encoding="utf-8",
    )
    # Ensure run_b is strictly newer than run_a for deterministic checkpoint ordering.
    os.utime(run_a / "metrics.jsonl", (1_700_000_000, 1_700_000_000))
    os.utime(run_b / "metrics.jsonl", (1_700_001_000, 1_700_001_000))

    # 1 playable game + 1 legacy (diverging) game in run_a
    good = _record_small_game(seed=40001)
    bad = _record_small_game(seed=40002)
    bad_games = bad["games"]
    assert isinstance(bad_games, list)
    bad_games[0]["actions"][1] = 9999
    (run_a / "games" / "eval-iter-0100.jsonl").write_text(
        json.dumps(good) + "\n" + json.dumps(bad) + "\n",
        encoding="utf-8",
    )

    (run_a / "checkpoints" / "iter-0050.safetensors").write_bytes(b"a50")
    (run_a / "checkpoints" / "iter-0100.safetensors").write_bytes(b"a100")
    (run_b / "checkpoints" / "iter-0050.safetensors").write_bytes(b"b50")

    client = TestClient(create_app(StudioConfig(runs_dir=tmp_path, allow_scryfall_network=False)))
    catalog = client.get("/api/runs").json()
    assert catalog["experiments"] == ["exp-alpha", "exp-beta"]
    assert set(catalog["eval_metric_keys"]) == {"custom_elo", "ensemble_held_out_det_win_rate", "mirror_win"}
    assert set(catalog["train_metric_keys"]) == {"grad_norm", "policy_kl", "tps"}
    assert "update" not in catalog["eval_metric_keys"]
    assert "timestamp" not in catalog["train_metric_keys"]

    by_id = {row["run_id"]: row for row in catalog["runs"]}
    row_a = by_id["exp-alpha/run-one"]
    assert row_a["playable_games"] == 1
    assert row_a["legacy_games"] == 1
    assert row_a["peak_ensemble_win"] == 0.72
    assert row_a["latest_metrics"]["custom_elo"] == 1610.0
    assert row_a["best_metrics"]["ensemble_held_out_det_win_rate"] == 0.72
    assert row_a["config_tags"]["custom_format"] == "cube"
    assert row_a["config_tags"]["seed_pool"] == 19
    assert row_a["config_tags"]["use_ema"] is True
    assert "nested_blob" not in row_a["config_tags"]
    assert "data_path" not in row_a["config_tags"]

    detail_a = client.get("/api/runs/exp-alpha/run-one").json()
    assert detail_a["summary"]["playable_games"] == 1
    assert detail_a["summary"]["legacy_games"] == 1
    assert "custom_elo" in detail_a["eval_metric_keys"]
    assert "grad_norm" in detail_a["train_metric_keys"]

    ckpts = client.get("/api/checkpoints").json()["checkpoints"]
    assert [c["ref"] for c in ckpts] == [
        "exp-beta/run-two::iter-0050",
        "exp-alpha/run-one::iter-0100",
        "exp-alpha/run-one::iter-0050",
    ]


def test_duplicate_permanent_disambiguation_and_hand_card_fork_edits(tmp_path: Path) -> None:
    runs_dir = _make_fixture_runs(tmp_path)
    client = TestClient(create_app(StudioConfig(runs_dir=runs_dir, allow_scryfall_network=False)))

    priority_step = next(
        step_idx
        for step_idx in range(20)
        if client.get(
            "/api/game-step",
            params={
                "run_id": "v2-test/run-alpha",
                "iteration": 100,
                "game_idx": 0,
                "step": step_idx,
                "pov": "omniscient",
            },
        ).json()["board"]["action_space_kind"]
        == "PRIORITY"
    )
    base_board = client.get(
        "/api/game-step",
        params={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": priority_step,
            "pov": "omniscient",
        },
    ).json()["board"]
    hero_hand = base_board["hero"]["hand"]
    assert len(hero_hand) >= 1
    card_to_remove = hero_hand[0]

    # Fork with 2 duplicate creatures on seat 1 + 1 unique creature on seat 0, plus remove_hand_card
    fork_resp = client.post(
        "/api/fork",
        json={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": priority_step,
            "pov": "omniscient",
            "edits": [
                {"op": "add_battlefield", "seat": 1, "card": "Boiling Rock Rioter"},
                {"op": "add_battlefield", "seat": 1, "card": "Boiling Rock Rioter"},
                {"op": "add_battlefield", "seat": 0, "card": "Fire Lord Zuko"},
                {"op": "remove_hand_card", "seat": 0, "card": card_to_remove},
            ],
        },
    ).json()
    assert fork_resp["warnings"] == []
    opp_creatures = fork_resp["board"]["opp"]["creatures"]
    hero_creatures = fork_resp["board"]["hero"]["creatures"]
    rioters = [c for c in opp_creatures if c["name"] == "Boiling Rock Rioter"]
    zukos = [c for c in hero_creatures if c["name"] == "Fire Lord Zuko"]
    assert sorted(c["copy_index"] for c in rioters) == [1, 2]
    assert len(zukos) >= 1
    assert all(c["copy_index"] is None for c in zukos)
    assert len(fork_resp["board"]["hero"]["hand"]) == len(hero_hand) - 1

    # swap_hand_card with a card present in hero's library succeeds cleanly
    lib_cards = [row["name"] for row in base_board["hero"]["library_summary"]]
    replacement = next(name for name in lib_cards if name != card_to_remove)
    swap_ok = client.post(
        "/api/fork",
        json={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": priority_step,
            "pov": "omniscient",
            "edits": [
                {
                    "op": "swap_hand_card",
                    "seat": 0,
                    "card": card_to_remove,
                    "replacement_card": replacement,
                }
            ],
        },
    ).json()
    assert swap_ok["warnings"] == []
    assert len(swap_ok["board"]["hero"]["hand"]) == len(hero_hand)
    assert replacement in swap_ok["board"]["hero"]["hand"]

    # swap_hand_card with a card NOT in deck emits a warning and preserves hand size
    swap_bad = client.post(
        "/api/fork",
        json={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": priority_step,
            "pov": "omniscient",
            "edits": [
                {
                    "op": "swap_hand_card",
                    "seat": 0,
                    "card": card_to_remove,
                    "replacement_card": "Island",
                }
            ],
        },
    ).json()
    assert len(swap_bad["warnings"]) == 1
    assert len(swap_bad["board"]["hero"]["hand"]) == len(hero_hand)

    # Verify combat action labels disambiguate duplicate creatures (#1 vs #2) without doubling #1
    deck = {"Plains": 16, "Mountain": 16, "Boiling Rock Rioter": 8}
    env = managym.Env(seed=77, skip_trivial=True)
    obs, _ = env.reset([managym.PlayerConfig("a", deck), managym.PlayerConfig("b", deck)])
    for _ in range(10):
        if int(obs.action_space.action_space_type) == int(ActionSpaceEnum.PRIORITY):
            break
        obs, _, _, _, _ = env.step(0)
    env.scenario_force_battlefield(0, "Boiling Rock Rioter", ready=True)
    env.scenario_force_battlefield(0, "Boiling Rock Rioter", ready=True)
    obs = env.scenario_refresh()
    # Collect labels across all DECLARE_ATTACKER prompts and verify both #1 and #2 appear cleanly
    attacker_labels: list[str] = []
    for _ in range(20):
        if int(obs.action_space.action_space_type) == int(ActionSpaceEnum.DECLARE_ATTACKER):
            board = serialize_observation(env, obs, decklists=(deck, deck), hero_seat=0, pov="omniscient")
            legal_actions = board["legal_actions"]
            assert isinstance(legal_actions, list)
            attacker_labels.extend(str(act["label"]) for act in legal_actions if isinstance(act, dict))
        obs, _, done, truncated, _ = env.step(0)
        if (
            done
            or truncated
            or (attacker_labels and int(obs.action_space.action_space_type) != int(ActionSpaceEnum.DECLARE_ATTACKER))
        ):
            break
    assert any("Boiling Rock Rioter #1" in label for label in attacker_labels)
    assert any("Boiling Rock Rioter #2" in label for label in attacker_labels)
    assert all("#1 #1" not in label and "#2 #2" not in label for label in attacker_labels)


def _setup_two_checkpoint_runs(tmp_path: Path) -> tuple[Path, str, str]:
    runs_dir = _make_fixture_runs(tmp_path)
    other_run = runs_dir / "v2-test" / "run-beta"
    (other_run / "checkpoints").mkdir(parents=True)
    (other_run / "config.json").write_text(json.dumps({"updates": 100}) + "\n", encoding="utf-8")
    (other_run / "metrics.jsonl").write_text(
        json.dumps({"kind": "train", "update": 100, "tps": 900.0}) + "\n",
        encoding="utf-8",
    )
    _save_tiny_checkpoint(runs_dir / "v2-test" / "run-alpha" / "checkpoints" / "iter-0100.safetensors", seed=11)
    _save_tiny_checkpoint(other_run / "checkpoints" / "iter-0200.safetensors", seed=29)
    return runs_dir, "v2-test/run-alpha::iter-0100", "v2-test/run-beta::iter-0200"


def test_cross_run_checkpoints_matrix_and_playout(tmp_path: Path) -> None:
    runs_dir, ref_a, ref_b = _setup_two_checkpoint_runs(tmp_path)
    client = TestClient(create_app(StudioConfig(runs_dir=runs_dir, allow_scryfall_network=False)))

    step_resp = client.get(
        "/api/game-step",
        params={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": 2,
            "pov": "omniscient",
            "ckpt_a": ref_a,
            "ckpt_b": ref_b,
        },
    )
    assert step_resp.status_code == 200
    step_data = step_resp.json()
    assert step_data["evaluation"]["warnings"] == []
    assert step_data["evaluation"]["primary"]["checkpoint"] == ref_a
    assert step_data["evaluation"]["compare"]["checkpoint"] == ref_b

    matrix_resp = client.post(
        "/api/matrix",
        json={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": 2,
            "pov": "omniscient",
            "models": ["heuristic", ref_a, ref_b],
        },
    )
    assert matrix_resp.status_code == 200
    matrix = matrix_resp.json()
    assert len(matrix["columns"]) == 3
    assert [col["ref"] for col in matrix["columns"]] == ["heuristic", ref_a, ref_b]
    assert all(col["warning"] is None for col in matrix["columns"])
    assert all(len(col["probs"]) == len(matrix["legal_actions"]) for col in matrix["columns"])

    for model_ref in ("heuristic", ref_b):
        playout_resp = client.post(
            "/api/playout",
            json={
                "run_id": "v2-test/run-alpha",
                "iteration": 100,
                "game_idx": 0,
                "step": 2,
                "pov": "omniscient",
                "model_ref": model_ref,
                "deals": 2,
                "batch_offset": 4,
                "action_indices": [0, 1],
                "seed": 7,
            },
        )
        assert playout_resp.status_code == 200
        pdata = playout_resp.json()
        assert pdata["model_ref"] == model_ref
        assert pdata["deals_done"] == 2
        assert pdata["batch_offset"] == 4
        assert pdata["elapsed_ms"] >= 0.0
        assert len(pdata["results"]) >= 1
        for res in pdata["results"]:
            assert res["wins"] + res["draws"] + res["losses"] == res["deals_done"]
            assert 0.0 <= res["ci_low"] <= res["win_rate"] <= res["ci_high"] <= 1.0


def test_compare_models_and_watchlist_patch(tmp_path: Path) -> None:
    runs_dir, ref_a, ref_b = _setup_two_checkpoint_runs(tmp_path)
    client = TestClient(create_app(StudioConfig(runs_dir=runs_dir, allow_scryfall_network=False)))

    compare_resp = client.post(
        "/api/compare",
        json={
            "ckpt_a": ref_a,
            "ckpt_b": ref_b,
            "sample_count": 6,
            "run_id": "v2-test/run-alpha",
            "seed": 13,
        },
    )
    assert compare_resp.status_code == 200
    cdata = compare_resp.json()
    assert cdata["warnings"] == []
    assert 1 <= cdata["sampled_positions"] <= 6
    assert len(cdata["positions"]) == cdata["sampled_positions"]
    assert len(cdata["by_decision_kind"]) >= 1
    assert 0.0 <= cdata["change_rate"] <= 1.0
    first_pos = cdata["positions"][0]
    assert len(first_pos["probs_a"]) == len(first_pos["legal_actions"])
    assert len(first_pos["probs_b"]) == len(first_pos["legal_actions"])

    filtered_cmp = client.post(
        "/api/compare",
        json={
            "ckpt_a": ref_a,
            "ckpt_b": ref_b,
            "sample_count": 4,
            "decision_kind": "PRIORITY",
            "seed": 13,
        },
    ).json()
    assert all(p["decision_kind"] == "PRIORITY" for p in filtered_cmp["positions"])

    created = client.post(
        "/api/watchlist",
        json={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": 2,
            "turn": 1,
            "seed": 20001,
            "label": "Initial label",
            "note": "Initial note",
            "tag": "combat",
            "summary": "Turn 1 decision",
            "key_cards": ["Fire Lord Zuko"],
        },
    ).json()["item"]
    assert created["tag"] == "combat"
    assert created["key_cards"] == ["Fire Lord Zuko"]

    patched = client.patch(
        f"/api/watchlist/{created['id']}",
        json={"label": "Updated label", "note": "Updated note", "tag": "blunder"},
    )
    assert patched.status_code == 200
    pitem = patched.json()["item"]
    assert pitem["label"] == "Updated label"
    assert pitem["note"] == "Updated note"
    assert pitem["tag"] == "blunder"
    assert pitem["summary"] == "Turn 1 decision"
    assert client.patch("/api/watchlist/nonexistent-id", json={"note": "x"}).status_code == 404

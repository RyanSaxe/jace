import json
from pathlib import Path

import managym
import pytest
from fastapi.testclient import TestClient

import jace.studio.benchmarks.promotion as promotion_mod
from jace.eval.arena import ArenaMatchRecord, append_match_record, ledger_path_for
from jace.eval.puzzles import PuzzleRecord, append_puzzle, puzzles_path
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


def _make_run_with_metrics(
    runs_dir: Path,
    *,
    run_id: str,
    early_wr: float,
    peak_wr: float,
    include_game: bool = False,
) -> None:
    run_dir = runs_dir / run_id
    (run_dir / "games").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"hidden_dim": 160, "attention_layers": 2, "updates": 2000}) + "\n",
        encoding="utf-8",
    )
    rows = [
        {"kind": "eval", "update": 100, "held_out_det_win_rate": round(early_wr - 0.08, 4)},
        {"kind": "eval", "update": 200, "held_out_det_win_rate": early_wr},
        {"kind": "eval", "update": 400, "held_out_det_win_rate": round(min(0.95, early_wr + 0.05), 4)},
        {"kind": "eval", "update": 2000, "held_out_det_win_rate": peak_wr},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    if include_game:
        (run_dir / "games" / "eval-iter-0100.jsonl").write_text(
            json.dumps(_record_small_game(seed=20001)) + "\n",
            encoding="utf-8",
        )


def _seed_arena_matches(runs_dir: Path) -> None:
    ledger = ledger_path_for(runs_dir)
    records = [
        ArenaMatchRecord(
            match_id="m-base-heur",
            timestamp_utc="2026-11-27T00:00:00Z",
            format_id="tla-limited-0.2.6",
            model_a="Base-u460",
            model_b="heuristic",
            search_config_a=None,
            search_config_b=None,
            deals=200,
            seed=9901,
            wins_a=314,
            losses_a=86,
            draws=0,
            sweeps_a=130,
            sweeps_b=16,
            splits=54,
            win_rate_a=0.785,
            ci95_low=0.742,
            ci95_high=0.824,
            los_a=1.0,
            ms_per_decision_a=1.2,
            ms_per_decision_b=0.4,
            manabot_version="0.2.6",
            git_sha="abc1234",
        ),
        ArenaMatchRecord(
            match_id="m-flagship-heur",
            timestamp_utc="2026-11-27T01:00:00Z",
            format_id="tla-limited-0.2.6",
            model_a="Flagship-u2000",
            model_b="heuristic",
            search_config_a=None,
            search_config_b=None,
            deals=200,
            seed=9901,
            wins_a=364,
            losses_a=36,
            draws=0,
            sweeps_a=166,
            sweeps_b=2,
            splits=32,
            win_rate_a=0.91,
            ci95_low=0.88,
            ci95_high=0.935,
            los_a=1.0,
            ms_per_decision_a=1.8,
            ms_per_decision_b=0.4,
            manabot_version="0.2.6",
            git_sha="abc1234",
        ),
        ArenaMatchRecord(
            match_id="m-stage2b-flagship",
            timestamp_utc="2026-11-27T02:00:00Z",
            format_id="tla-limited-0.2.6",
            model_a="Stage2B-u2230",
            model_b="Flagship-u2000",
            search_config_a=None,
            search_config_b=None,
            deals=200,
            seed=9901,
            wins_a=220,
            losses_a=180,
            draws=0,
            sweeps_a=25,
            sweeps_b=5,
            splits=170,
            win_rate_a=0.55,
            ci95_low=0.523,
            ci95_high=0.577,
            los_a=0.999,
            ms_per_decision_a=2.1,
            ms_per_decision_b=1.8,
            manabot_version="0.2.6",
            git_sha="abc1234",
        ),
        ArenaMatchRecord(
            match_id="m-stage2b-heur",
            timestamp_utc="2026-11-27T02:30:00Z",
            format_id="tla-limited-0.2.6",
            model_a="Stage2B-u2230",
            model_b="heuristic",
            search_config_a=None,
            search_config_b=None,
            deals=200,
            seed=9901,
            wins_a=358,
            losses_a=42,
            draws=0,
            sweeps_a=160,
            sweeps_b=2,
            splits=38,
            win_rate_a=0.895,
            ci95_low=0.865,
            ci95_high=0.925,
            los_a=1.0,
            ms_per_decision_a=2.1,
            ms_per_decision_b=0.4,
            manabot_version="0.2.6",
            git_sha="abc1234",
        ),
        ArenaMatchRecord(
            match_id="m-other-format",
            timestamp_utc="2026-11-26T00:00:00Z",
            format_id="dsk-limited-0.2.6",
            model_a="Dsk-u500",
            model_b="heuristic",
            search_config_a=None,
            search_config_b=None,
            deals=50,
            seed=100,
            wins_a=70,
            losses_a=30,
            draws=0,
            sweeps_a=25,
            sweeps_b=5,
            splits=20,
            win_rate_a=0.70,
            ci95_low=0.61,
            ci95_high=0.79,
            los_a=0.99,
            ms_per_decision_a=1.0,
            ms_per_decision_b=0.4,
            manabot_version="0.2.6",
            git_sha="abc1234",
        ),
    ]
    for rec in records:
        append_match_record(ledger, rec)


def test_empty_benchmarks_state(tmp_path: Path) -> None:
    client = TestClient(create_app(StudioConfig(runs_dir=tmp_path, allow_scryfall_network=False)))

    lb = client.get("/api/arena/leaderboard")
    assert lb.status_code == 200
    lb_data = lb.json()
    assert lb_data["formats"] == []
    assert lb_data["models"] == []
    assert lb_data["forecast"] is None

    puz = client.get("/api/puzzles")
    assert puz.status_code == 200
    puz_data = puz.json()
    assert puz_data["puzzles"] == []
    assert puz_data["total"] == 0


def test_arena_leaderboard_anchor_shift_matches_and_forecast(tmp_path: Path) -> None:
    _make_run_with_metrics(tmp_path, run_id="v2-exp/smoke-400u", early_wr=0.60, peak_wr=0.76)
    _make_run_with_metrics(tmp_path, run_id="v2-exp/flagship", early_wr=0.78, peak_wr=0.91)
    _seed_arena_matches(tmp_path)
    client = TestClient(create_app(StudioConfig(runs_dir=tmp_path, allow_scryfall_network=False)))

    resp = client.get("/api/arena/leaderboard", params={"format": "tla-limited-0.2.6"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "tla-limited-0.2.6"
    assert set(data["formats"]) == {"dsk-limited-0.2.6", "tla-limited-0.2.6"}
    assert data["anchor_model"] == "heuristic"
    assert data["anchor_elo"] == 1500.0

    by_id = {m["model_id"]: m for m in data["models"]}
    assert by_id["heuristic"]["elo"] == 1500.0
    assert by_id["heuristic"]["ci_half"] == 0.0
    assert by_id["heuristic"]["vs_anchor_wr"] is None
    assert by_id["Stage2B-u2230"]["elo"] > by_id["Flagship-u2000"]["elo"] > by_id["Base-u460"]["elo"] > 1500.0
    assert by_id["Stage2B-u2230"]["vs_anchor_wr"] == 0.895
    assert by_id["Stage2B-u2230"]["updates"] == 2230

    forecast = data["forecast"]
    assert forecast is not None
    assert forecast["n"] == 2
    assert forecast["pearson_r"] == 1.0
    assert forecast["power_law_pred"] is not None

    shifted = client.get(
        "/api/arena/leaderboard",
        params={"format": "tla-limited-0.2.6", "anchor": "Flagship-u2000"},
    )
    assert shifted.status_code == 200
    shifted_data = shifted.json()
    assert shifted_data["anchor_model"] == "Flagship-u2000"
    shifted_by_id = {m["model_id"]: m for m in shifted_data["models"]}
    assert shifted_by_id["Flagship-u2000"]["elo"] == 1500.0
    assert shifted_by_id["Flagship-u2000"]["ci_half"] == 0.0
    assert shifted_by_id["Stage2B-u2230"]["elo"] > 1500.0
    assert shifted_by_id["heuristic"]["elo"] < 1500.0
    assert shifted_by_id["Stage2B-u2230"]["vs_anchor_wr"] == 0.55

    # Check match orientation from both perspectives.
    m_stage = client.get(
        "/api/arena/matches",
        params={"format": "tla-limited-0.2.6", "model": "Stage2B-u2230"},
    ).json()["matches"]
    assert len(m_stage) == 2
    vs_flag = next(m for m in m_stage if m["opponent_id"] == "Flagship-u2000")
    assert vs_flag["winrate"] == 0.55
    assert vs_flag["sweeps_w"] == 25
    assert vs_flag["sweeps_l"] == 5

    m_flag = client.get(
        "/api/arena/matches",
        params={"format": "tla-limited-0.2.6", "model": "Flagship-u2000"},
    ).json()["matches"]
    vs_stage = next(m for m in m_flag if m["opponent_id"] == "Stage2B-u2230")
    assert vs_stage["winrate"] == 0.45
    assert vs_stage["sweeps_w"] == 5
    assert vs_stage["sweeps_l"] == 25
    assert vs_stage["los"] == round(1.0 - 0.999, 4)


def test_puzzles_endpoint_filters_and_cached_scoring(tmp_path: Path) -> None:
    _make_run_with_metrics(tmp_path, run_id="v2-test/run-alpha", early_wr=0.64, peak_wr=0.72, include_game=True)
    append_puzzle(
        tmp_path,
        PuzzleRecord(
            puzzle_id="puz-001",
            format_id="tla-limited-0.2.6",
            source="policy_disagreement",
            run_id="v2-test/run-alpha",
            iteration=100,
            game_idx=0,
            step=5,
            turn=3,
            phase="COMBAT",
            step_name="DECLARE_ATTACKERS",
            decision_kind="DECLARE_ATTACKER",
            hero_seat=0,
            seed=20001,
            hero_summary="Hero: 16 life · 3 lands · 4 cards",
            opp_summary="Opp: 18 life · 3 lands · 4 cards",
            key_cards=("Fire Lord Zuko", "Boiling Rock Rioter"),
            root_value_v=0.55,
            best_action_index=0,
            best_action_label="Attack with Fire Lord Zuko",
            alt_action_index=1,
            alt_action_label="Don't attack",
            best_playout_wr=0.78,
            alt_playout_wr=0.52,
            playouts_per_action=64,
            max_plies=240,
            margin=0.26,
            paired_ci_low=0.14,
            paired_ci_high=0.38,
        ),
    )
    append_puzzle(
        tmp_path,
        PuzzleRecord(
            puzzle_id="puz-002",
            format_id="tla-limited-0.2.6",
            source="watchlist_promotion",
            run_id="v2-test/run-alpha",
            iteration=100,
            game_idx=0,
            step=6,
            turn=4,
            phase="MAIN1",
            step_name="MAIN1",
            decision_kind="PRIORITY",
            hero_seat=0,
            seed=20001,
            hero_summary="Hero: 14 life · 4 lands · 3 cards",
            opp_summary="Opp: 15 life · 4 lands · 3 cards",
            key_cards=("Fire Lord Zuko",),
            root_value_v=0.50,
            best_action_index=0,
            best_action_label="Pass priority",
            alt_action_index=1,
            alt_action_label="Cast spell",
            best_playout_wr=0.71,
            alt_playout_wr=0.49,
            playouts_per_action=64,
            max_plies=240,
            margin=0.22,
            paired_ci_low=0.11,
            paired_ci_high=0.33,
        ),
    )
    client = TestClient(create_app(StudioConfig(runs_dir=tmp_path, allow_scryfall_network=False)))

    all_resp = client.get(
        "/api/puzzles",
        params={"format": "tla-limited-0.2.6", "model_a": "heuristic", "model_b": "heuristic"},
    )
    assert all_resp.status_code == 200
    payload = all_resp.json()
    assert payload["total"] == 2
    assert payload["formats"] == ["tla-limited-0.2.6"]
    assert payload["puzzles"][0]["source"] == "auto_mined"
    assert payload["puzzles"][1]["source"] == "watchlist"
    assert payload["puzzles"][0]["actions"][0]["prob_a"] is not None

    filtered = client.get(
        "/api/puzzles",
        params={"format": "tla-limited-0.2.6", "source": "watchlist"},
    ).json()
    assert filtered["total"] == 1
    assert filtered["puzzles"][0]["puzzle_id"] == "puz-002"

    phase_filtered = client.get(
        "/api/puzzles",
        params={"format": "tla-limited-0.2.6", "phase": "Declare Attacker"},
    ).json()
    assert phase_filtered["total"] == 1
    assert phase_filtered["puzzles"][0]["puzzle_id"] == "puz-001"


def test_watchlist_promote_puzzle_gate_422_and_200(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_run_with_metrics(tmp_path, run_id="v2-test/run-alpha", early_wr=0.64, peak_wr=0.72, include_game=True)
    client = TestClient(create_app(StudioConfig(runs_dir=tmp_path, allow_scryfall_network=False)))

    # Find a step with >= 2 legal actions so promotion runs real replay + verification.
    decision_step = next(
        step_idx
        for step_idx in range(25)
        if len(
            client.get(
                "/api/game-step",
                params={
                    "run_id": "v2-test/run-alpha",
                    "iteration": 100,
                    "game_idx": 0,
                    "step": step_idx,
                    "pov": "omniscient",
                },
            ).json()["board"]["legal_actions"]
        )
        >= 2
    )
    created = client.post(
        "/api/watchlist",
        json={
            "run_id": "v2-test/run-alpha",
            "iteration": 100,
            "game_idx": 0,
            "step": decision_step,
            "turn": 3,
            "seed": 20001,
            "label": "T3 Decision",
            "note": "Check promote gate",
            "key_cards": ["Fire Lord Zuko"],
        },
    ).json()["item"]
    item_id = created["id"]

    # Real playout with impossible 99% margin threshold must reject with HTTP 422.
    rejected = client.post(
        f"/api/watchlist/{item_id}/promote-puzzle",
        json={"format": "tla-limited-0.2.6", "playouts": 4, "min_margin": 0.99},
    )
    assert rejected.status_code == 422
    rej_body = rejected.json()
    assert "did not pass" in rej_body["detail"]
    assert "margin" in rej_body
    assert "ci_low" in rej_body
    assert "ci_high" in rej_body
    assert not puzzles_path(tmp_path, "tla-limited-0.2.6").exists()

    # Deterministic strong paired margin must pass gate (200) and persist to JSONL.
    def _strong_outcomes(
        _env: object,
        *,
        action_indices: list[int],
        playouts: int,
        **_kwargs: object,
    ) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        for pos, idx in enumerate(action_indices):
            out[idx] = [1.0] * playouts if pos == 0 else [0.0] * playouts
        return out

    # create_app builds BenchmarkService internally; stub CRN playouts for deterministic 200 gate coverage.
    # ast-grep-ignore: python.question-monkeypatch
    monkeypatch.setattr(promotion_mod, "_run_paired_crn_outcomes", _strong_outcomes)
    admitted = client.post(
        f"/api/watchlist/{item_id}/promote-puzzle",
        json={"format": "tla-limited-0.2.6", "playouts": 16, "min_margin": 0.15},
    )
    assert admitted.status_code == 200
    adm_body = admitted.json()
    assert adm_body["admitted"] is True
    assert adm_body["puzzle"]["source"] == "watchlist"
    assert adm_body["puzzle"]["margin"] == 1.0

    # Re-promoting the same watchlist item deduplicates cleanly by deterministic puzzle_id.
    dup = client.post(
        f"/api/watchlist/{item_id}/promote-puzzle",
        json={"format": "tla-limited-0.2.6", "playouts": 16, "min_margin": 0.15},
    )
    assert dup.status_code == 200

    # Promoted puzzle is immediately served by GET /api/puzzles (only 1 record after dedup).
    listed = client.get("/api/puzzles", params={"format": "tla-limited-0.2.6"}).json()
    assert listed["total"] == 1
    assert listed["puzzles"][0]["source"] == "watchlist"

    # Invalid format traversal returns HTTP 400 (not 500).
    bad_fmt = client.get("/api/puzzles", params={"format": "../escape"})
    assert bad_fmt.status_code == 400

import json
from pathlib import Path

import pytest

from jace.eval.arena import (
    ArenaMatchRecord,
    SearchConfigSummary,
    append_match_record,
    import_match_summaries,
    ledger_path_for,
    load_match_records,
    paired_los,
    run_arena_match,
    solve_bradley_terry,
)
from jace.eval.arena import main as arena_main
from jace.eval.deals import build_deals, elemental_decks
from jace.eval.parallel import play_deals_parallel
from jace.eval.report import discover_early_forecast_points, fit_early_forecast, generate_local_report


def _sample_record(
    *,
    match_id: str = "m1",
    format_id: str = "tla-limited-0.2.6",
    model_a: str = "alpha",
    model_b: str = "heuristic",
    wins_a: int = 150,
    losses_a: int = 50,
    draws: int = 0,
    sweeps_a: int = 60,
    sweeps_b: int = 10,
    splits: int = 30,
    ms_a: float = 4.2,
    ms_b: float = 0.4,
    search_a: SearchConfigSummary | None = None,
) -> ArenaMatchRecord:
    total = wins_a + losses_a + draws
    wr = (wins_a + 0.5 * draws) / total
    return ArenaMatchRecord(
        match_id=match_id,
        timestamp_utc="2026-04-02T00:00:00Z",
        format_id=format_id,
        model_a=model_a,
        model_b=model_b,
        search_config_a=search_a,
        search_config_b=None,
        deals=sweeps_a + sweeps_b + splits,
        seed=9901,
        wins_a=wins_a,
        losses_a=losses_a,
        draws=draws,
        sweeps_a=sweeps_a,
        sweeps_b=sweeps_b,
        splits=splits,
        win_rate_a=wr,
        ci95_low=max(0.0, wr - 0.05),
        ci95_high=min(1.0, wr + 0.05),
        los_a=paired_los(wr, 0.025),
        ms_per_decision_a=ms_a,
        ms_per_decision_b=ms_b,
        manabot_version="0.2.6",
        git_sha="abc1234",
    )


def test_locked_append_and_reload_round_trip(tmp_path: Path) -> None:
    ledger = ledger_path_for(tmp_path)
    rec_a = _sample_record(
        match_id="m-search",
        search_a=SearchConfigSummary(worlds=16, horizon=24, belief_id="belief-tla-v1"),
    )
    rec_b = _sample_record(match_id="m-other-format", format_id="fdn-limited-0.2.6")
    append_match_record(ledger, rec_a)
    append_match_record(ledger, rec_b)

    all_loaded = load_match_records(ledger)
    assert all_loaded == [rec_a, rec_b]
    filtered = load_match_records(ledger, format_id="tla-limited-0.2.6")
    assert filtered == [rec_a]


def _write_synthetic_matches_jsonl(path: Path) -> None:
    rows = [
        {"match_id": "m-a-h", "model_a": "model-a-u500", "model_b": "heuristic", "wins_a": 320, "losses_a": 80},
        {"match_id": "m-b-h", "model_a": "model-b-u1000", "model_b": "heuristic", "wins_a": 350, "losses_a": 50},
        {"match_id": "m-b-a", "model_a": "model-b-u1000", "model_b": "model-a-u500", "wins_a": 240, "losses_a": 160},
        {"match_id": "m-c-h", "model_a": "model-c-u2000", "model_b": "heuristic", "wins_a": 360, "losses_a": 40},
        {"match_id": "m-c-b", "model_a": "model-c-u2000", "model_b": "model-b-u1000", "wins_a": 220, "losses_a": 180},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_idempotent_import_and_bradley_terry_anchor_invariance(tmp_path: Path) -> None:
    src = tmp_path / "external_matches.jsonl"
    _write_synthetic_matches_jsonl(src)

    first_added = import_match_summaries(tmp_path, input_path=src, format_id="tla-limited-0.2.6")
    second_added = import_match_summaries(tmp_path, input_path=src, format_id="tla-limited-0.2.6")
    assert first_added == 5
    assert second_added == 0

    records = load_match_records(ledger_path_for(tmp_path), format_id="tla-limited-0.2.6")
    assert sum(r.wins_a + r.losses_a + r.draws for r in records) == 2_000

    ratings_h = solve_bradley_terry(records, format_id="tla-limited-0.2.6", bootstrap_samples=400)
    by_id_h = {item.player_id: item for item in ratings_h}

    assert by_id_h["heuristic"].elo == pytest.approx(1500.0)
    assert by_id_h["heuristic"].ci95_low == pytest.approx(1500.0)
    assert by_id_h["heuristic"].ci95_high == pytest.approx(1500.0)
    assert (
        by_id_h["model-c-u2000"].elo
        > by_id_h["model-b-u1000"].elo
        > by_id_h["model-a-u500"].elo
        > by_id_h["heuristic"].elo
    )
    assert by_id_h["model-c-u2000"].vs_anchor_wr == pytest.approx(360 / 400)

    # Switching anchor shifts every rating by the exact same constant and tightens CIs near the anchor.
    ratings_b = solve_bradley_terry(
        records,
        format_id="tla-limited-0.2.6",
        anchor_player="model-b-u1000",
        bootstrap_samples=400,
    )
    by_id_b = {item.player_id: item for item in ratings_b}
    expected_shift = 1500.0 - by_id_h["model-b-u1000"].elo
    for pid, r_h in by_id_h.items():
        assert by_id_b[pid].elo - r_h.elo == pytest.approx(expected_shift, abs=1e-4)
    assert by_id_b["model-b-u1000"].ci95_low == pytest.approx(1500.0)
    assert by_id_b["model-b-u1000"].ci95_high == pytest.approx(1500.0)
    # model-c played directly against model-b with a closer win rate, so its CI tightens when anchored on model-b.
    width_h = by_id_h["model-c-u2000"].ci95_high - by_id_h["model-c-u2000"].ci95_low
    width_b = by_id_b["model-c-u2000"].ci95_high - by_id_b["model-c-u2000"].ci95_low
    assert width_b < width_h


def test_default_anchor_falls_back_to_earliest_player_without_heuristic() -> None:
    rec = _sample_record(model_a="early-ckpt", model_b="later-ckpt", wins_a=120, losses_a=80)
    ratings = solve_bradley_terry([rec], format_id="tla-limited-0.2.6", bootstrap_samples=0)
    by_id = {item.player_id: item for item in ratings}
    assert by_id["early-ckpt"].elo == pytest.approx(1500.0)
    assert by_id["later-ckpt"].elo < 1500.0


def test_paired_los_edge_cases() -> None:
    assert paired_los(0.6, 0.0) == 1.0
    assert paired_los(0.4, 0.0) == 0.0
    assert paired_los(0.5, 0.0) == 0.5
    assert paired_los(0.5, 0.03) == pytest.approx(0.5)
    assert paired_los(0.6, 0.03) > 0.999
    assert paired_los(0.4, 0.03) < 0.001


def test_fit_early_forecast_recovers_power_law_and_validates_inputs() -> None:
    # Exact noiseless power law: 1 - wr = 2.0 * u^(-0.4)
    points = [(u, 1.0 - 2.0 * (u**-0.4)) for u in (50, 100, 200, 300, 400)]
    forecast = fit_early_forecast(points, max_update=400, target_update=2000, run_id="synthetic")
    assert forecast is not None
    assert forecast.points_used == 5
    assert forecast.max_update_used == 400
    assert forecast.c_param == pytest.approx(2.0, rel=1e-5)
    assert forecast.alpha_param == pytest.approx(0.4, rel=1e-5)
    expected = 1.0 - 2.0 * (2000**-0.4)
    assert forecast.projected_win_rate == pytest.approx(expected, rel=1e-5)
    assert forecast.ci95_low <= forecast.projected_win_rate <= forecast.ci95_high

    assert fit_early_forecast([(100, 0.5)], max_update=400) is None
    assert fit_early_forecast([(0, 0.1), (500, 0.8)], max_update=400) is None


def test_parallel_timing_and_arena_match_integration(tmp_path: Path) -> None:
    deals = build_deals(elemental_decks()[:2], count=1, seed=17, decklists="closed")
    deal_results = play_deals_parallel("heuristic", "random", deals=deals, num_workers=1, seed=17)
    assert len(deal_results) == 1
    assert deal_results[0].decisions_a > 0
    assert deal_results[0].decisions_b > 0
    assert deal_results[0].decision_ms_a >= 0.0
    assert deal_results[0].decision_ms_b >= 0.0

    rec = run_arena_match(
        tmp_path,
        format_id="elemental-test",
        model_a="heuristic",
        model_b="random",
        deals=1,
        seed=17,
        num_workers=1,
    )
    assert rec.deals == 1
    assert rec.wins_a + rec.losses_a + rec.draws == 2
    assert rec.ms_per_decision_a >= 0.0
    assert rec.ms_per_decision_b >= 0.0
    assert load_match_records(ledger_path_for(tmp_path), format_id="elemental-test") == [rec]


def test_discover_early_forecast_and_generate_local_report(tmp_path: Path) -> None:
    src = tmp_path / "external_matches.jsonl"
    _write_synthetic_matches_jsonl(src)
    import_match_summaries(tmp_path, input_path=src, format_id="tla-limited-0.2.6")
    # Add one timed search record so Elo-vs-ms frontier figure is also rendered.
    append_match_record(
        ledger_path_for(tmp_path),
        _sample_record(
            match_id="m-search-16w",
            model_a="model-b-u1000",
            model_b="heuristic",
            wins_a=180,
            losses_a=20,
            sweeps_a=82,
            sweeps_b=2,
            splits=16,
            ms_a=28.5,
            ms_b=0.4,
            search_a=SearchConfigSummary(worlds=16, horizon=24, belief_id="belief-v1"),
        ),
    )
    # Add two synthetic training runs with metrics.jsonl for early forecast discovery.
    for name, curve in (
        ("run-fast", [(0, 0.01), (100, 0.65), (200, 0.78), (400, 0.85), (1000, 0.90)]),
        ("run-slow", [(0, 0.01), (100, 0.30), (200, 0.42), (400, 0.55), (1000, 0.64)]),
    ):
        run_dir = tmp_path / "runs" / name
        run_dir.mkdir(parents=True)
        lines = [json.dumps({"kind": "eval", "update": u, "held_out_det_win_rate": wr}) for u, wr in curve]
        (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Add synthetic verified puzzles so puzzle summary chart is also rendered.
    puzzles_dir = tmp_path / "puzzles"
    puzzles_dir.mkdir(parents=True)
    puzzles_lines = [
        json.dumps({"puzzle_id": "p1", "source": "policy_disagreement", "margin": 0.25}),
        json.dumps({"puzzle_id": "p2", "source": "high_swing_loss", "margin": 0.35}),
    ]
    (puzzles_dir / "tla-limited-0.2.6.jsonl").write_text("\n".join(puzzles_lines) + "\n", encoding="utf-8")

    summary = discover_early_forecast_points(tmp_path)
    assert len(summary.points) == 2
    assert summary.pearson_r == pytest.approx(1.0)
    assert summary.spearman_rho == pytest.approx(1.0)

    html_path = generate_local_report(tmp_path, format_id="tla-limited-0.2.6")
    report_dir = tmp_path / "report"
    assert html_path == report_dir / "report.html"
    assert html_path.exists()
    assert (report_dir / "report.md").exists()
    assert "model-c-u2000" in html_path.read_text(encoding="utf-8")
    assert "model-c-u2000" in (report_dir / "report.md").read_text(encoding="utf-8")

    for fig_name in ("elo_progression.png", "elo_vs_ms.png", "early_forecast.png", "puzzle_summary.png"):
        fig_path = report_dir / "figures" / fig_name
        assert fig_path.exists()
        assert fig_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    # Exercise CLI leaderboard and report subcommands cleanly.
    assert arena_main(["leaderboard", "--runs-dir", str(tmp_path), "--format", "tla-limited-0.2.6"]) == 0
    assert arena_main(["report", "--runs-dir", str(tmp_path), "--format", "tla-limited-0.2.6"]) == 0

    # Search-enabled matches require passing a deal_runner from jace.forks.arena.
    with pytest.raises(ValueError, match="deal_runner"):
        run_arena_match(
            tmp_path,
            format_id="tla-limited-0.2.6",
            model_a="heuristic",
            model_b="heuristic",
            deals=1,
            search_config_a=SearchConfigSummary(worlds=2, horizon=2, belief_id="prior-only"),
        )

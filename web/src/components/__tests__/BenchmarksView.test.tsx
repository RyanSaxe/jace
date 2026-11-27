/** Behavioral tests for Benchmarks (Elo Arena, Puzzles, Anchor shift, Match Ledger) and Watchlist promotion. */

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "@/App";
import { WatchlistView } from "@/components/WatchlistView";
import type {
  ArenaLeaderboardResponse,
  ArenaMatchRow,
  PuzzlesResponse,
  WatchlistItem,
} from "@/lib/types";

const STAGE2B_MODEL = {
  model_id: "stage2b-u2230",
  label: "Stage 2B u2,230 EMA",
  elo: 1876.3,
  ci_half: 33.6,
  matches: 3,
  games: 1200,
  vs_anchor_wr: 0.895,
  updates: 2230,
  search_ms_per_dec: 2.1,
  checkpoint_ref: "v2-colab/stage2b::iter-2230",
};

const FLAGSHIP_MODEL = {
  model_id: "flagship-u2000",
  label: "Flagship u2,000 EMA",
  elo: 1867.4,
  ci_half: 23.1,
  matches: 5,
  games: 2000,
  vs_anchor_wr: 0.91,
  updates: 2000,
  search_ms_per_dec: 1.8,
  checkpoint_ref: "v2-colab/flagship::iter-2000",
};

const HEURISTIC_MODEL = {
  model_id: "heuristic",
  label: "Heuristic",
  elo: 1500,
  ci_half: 0,
  matches: 7,
  games: 2800,
  vs_anchor_wr: null,
  updates: 0,
  search_ms_per_dec: 0.4,
  checkpoint_ref: "heuristic",
};

const MOCK_LEADERBOARD_HEURISTIC: ArenaLeaderboardResponse = {
  format: "tla-limited-0.2.6",
  formats: ["tla-limited-0.2.6", "dsk-limited-0.2.6"],
  anchor_model: "heuristic",
  anchor_elo: 1500,
  models: [STAGE2B_MODEL, FLAGSHIP_MODEL, HEURISTIC_MODEL],
  forecast: {
    n: 2,
    pearson_r: 0.965,
    spearman_rho: 0.829,
    points: [
      { run_id: "smoke-400u", early_wr: 0.585, peak_wr: 0.76 },
      { run_id: "flagship", early_wr: 0.788, peak_wr: 0.91 },
    ],
    power_law_pred: 0.917,
  },
};

const MOCK_LEADERBOARD_FLAGSHIP_ANCHOR: ArenaLeaderboardResponse = {
  ...MOCK_LEADERBOARD_HEURISTIC,
  anchor_model: "flagship-u2000",
  models: [
    { ...STAGE2B_MODEL, elo: 1508.9, ci_half: 18.2, vs_anchor_wr: 0.549 },
    { ...FLAGSHIP_MODEL, elo: 1500, ci_half: 0, vs_anchor_wr: null },
    { ...HEURISTIC_MODEL, elo: 1132.6, ci_half: 23.1, vs_anchor_wr: 0.09 },
  ],
};

const MOCK_MATCHES: ArenaMatchRow[] = [
  {
    match_id: "m-1",
    opponent_id: "flagship-u2000",
    opponent_label: "Flagship u2,000 EMA",
    games: 400,
    winrate: 0.549,
    ci_low: 0.523,
    ci_high: 0.575,
    sweeps_w: 25,
    sweeps_l: 5,
    los: 0.999,
    compare_target: { ckptA: "v2-colab/stage2b::iter-2230", ckptB: "v2-colab/flagship::iter-2000" },
  },
];

const MOCK_PUZZLES: PuzzlesResponse = {
  format: "tla-limited-0.2.6",
  formats: ["tla-limited-0.2.6", "dsk-limited-0.2.6"],
  solved_a: 1,
  solved_b: 0,
  total: 1,
  puzzles: [
    {
      puzzle_id: "puz-1",
      format: "tla-limited-0.2.6",
      turn: 16,
      phase: "Declare Attacker",
      source: "auto_mined",
      seed_label: "Seed #20001",
      hero_summary: "Hero: 4 life · 8 lands · 1 card",
      opp_summary: "Opp: 19 life · 6 lands · 0 cards",
      key_cards: ["Appa, Loyal Sky Bison", "Fire Lord Zuko"],
      actions: [
        {
          label: "Don't attack (Appa, Loyal Sky Bison)",
          playout_wr: 0.492,
          prob_a: 0.998,
          prob_b: 0.06,
          is_best: true,
        },
        {
          label: "Attack with Appa, Loyal Sky Bison",
          playout_wr: 0.148,
          prob_a: 0.002,
          prob_b: 0.94,
          is_best: false,
        },
      ],
      margin: 0.344,
      ci_low: 0.287,
      ci_high: 0.401,
      playouts: 64,
      tabletop_target: {
        run_id: "v2-colab/flagship",
        iteration: 2000,
        game_idx: 0,
        step: 142,
        seed: 20001,
      },
    },
  ],
};

function installBenchmarksFetch(): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: string | URL) => {
      const url = String(input);
      let body: unknown = { items: [] };
      if (url === "/api/runs") {
        body = { runs: [], experiments: [], eval_metric_keys: [], train_metric_keys: [] };
      } else if (url === "/api/checkpoints") {
        body = { checkpoints: [] };
      } else if (url.startsWith("/api/arena/leaderboard")) {
        body = url.includes("anchor=flagship-u2000")
          ? MOCK_LEADERBOARD_FLAGSHIP_ANCHOR
          : MOCK_LEADERBOARD_HEURISTIC;
      } else if (url.startsWith("/api/arena/matches")) {
        body = { matches: MOCK_MATCHES };
      } else if (url.startsWith("/api/puzzles")) {
        body = url.includes("format=dsk-limited-0.2.6")
          ? {
              format: "dsk-limited-0.2.6",
              formats: ["tla-limited-0.2.6", "dsk-limited-0.2.6"],
              puzzles: [],
              solved_a: 0,
              solved_b: 0,
              total: 0,
            }
          : MOCK_PUZZLES;
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: "OK",
        json: (): Promise<unknown> => Promise.resolve(body),
      });
    }),
  );
}

describe("BenchmarksView and App routing", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    window.history.replaceState(null, "", "#runs");
    window.localStorage.clear();
  });

  it("navigates to Benchmarks (3), shifts Elo anchor, opens Match Ledger, and switches to Puzzles without Value Probes", async () => {
    installBenchmarksFetch();
    render(<App />);

    const nav = screen.getByRole("navigation", { name: "Primary navigation" });
    fireEvent.click(within(nav).getByRole("button", { name: /^Benchmarks/i }));

    await waitFor(() => {
      expect(screen.getAllByText("Stage 2B u2,230 EMA").length).toBeGreaterThan(0);
    });
    expect(screen.queryByText(/Value Probes/i)).not.toBeInTheDocument();
    expect(screen.getByText(/vs Heuristic/i)).toBeInTheDocument();

    // Change Elo anchor to Flagship u2,000 EMA.
    const anchorSelect = screen.getByLabelText("Elo anchor");
    fireEvent.change(anchorSelect, { target: { value: "flagship-u2000" } });
    await waitFor(() => {
      expect(screen.getByText(/vs Flagship u2,000 EMA/i)).toBeInTheDocument();
    });

    // Click top row to open Match Ledger side panel.
    const table = screen.getByRole("table");
    fireEvent.click(within(table).getByText("Stage 2B u2,230 EMA"));
    await waitFor(() => {
      expect(screen.getAllByText(/25W \/ 5L sweeps/i).length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText(/Compare in Studio/i).length).toBeGreaterThan(0);

    // Switch to Puzzles segment.
    const puzzleBtns = screen.getAllByRole("button", { name: "Puzzles" });
    const firstPuzzleBtn = puzzleBtns[0];
    expect(firstPuzzleBtn).toBeDefined();
    if (firstPuzzleBtn) fireEvent.click(firstPuzzleBtn);
    await waitFor(() => {
      expect(screen.getByText(/T16 · Declare Attacker/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Don't attack \(Appa, Loyal Sky Bison\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Auto-mined · N=64/i)).toBeInTheDocument();
    expect(window.location.hash).toContain("#benchmarks/puzzles");

    // Empty puzzles state prints CLI command.
    const formatSelect = screen.getByLabelText("Benchmark format");
    fireEvent.change(formatSelect, { target: { value: "dsk-limited-0.2.6" } });
    await waitFor(() => {
      expect(screen.getByText(/uv run python -m jace\.eval\.puzzles mine/i)).toBeInTheDocument();
    });
  });

  it("shows Promote to Puzzle status for 422 rejection and 200 admission on Watchlist cards", async () => {
    const item: WatchlistItem = {
      id: "wl-001",
      run_id: "v2-colab/flagship",
      iteration: 2000,
      game_idx: 0,
      step: 42,
      turn: 9,
      seed: 20001,
      label: "T9 Priority",
      note: "Hold trick",
      key_cards: ["Fire Lord Zuko"],
    };

    let attempt = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL) => {
        const url = String(input);
        if (url.startsWith("/api/arena/leaderboard")) {
          return Promise.resolve({
            ok: true,
            status: 200,
            statusText: "OK",
            json: (): Promise<unknown> => Promise.resolve(MOCK_LEADERBOARD_HEURISTIC),
          });
        }
        attempt += 1;
        if (attempt === 1) {
          return Promise.resolve({
            ok: false,
            status: 422,
            statusText: "Unprocessable Entity",
            json: (): Promise<unknown> =>
              Promise.resolve({
                detail: "Margin +6.2% (95% CI [-4.1%, +16.5%]) did not pass +15.0% & CI > 0 gate",
                margin: 0.062,
                ci_low: -0.041,
                ci_high: 0.165,
              }),
          });
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: "OK",
          json: (): Promise<unknown> =>
            Promise.resolve({ admitted: true, puzzle: MOCK_PUZZLES.puzzles[0] }),
        });
      }),
    );

    render(
      <WatchlistView
        items={[item]}
        loading={false}
        onOpenTabletop={vi.fn()}
        onRemove={vi.fn()}
        onUpdate={vi.fn(() => Promise.resolve())}
        onOpenPuzzles={vi.fn()}
      />,
    );

    const promoteBtn = screen.getByRole("button", { name: /Promote to Puzzle/i });
    fireEvent.click(promoteBtn);
    await waitFor(() => {
      expect(screen.getByText(/did not pass \+15\.0% & CI > 0 gate/i)).toBeInTheDocument();
    });

    fireEvent.click(promoteBtn);
    await waitFor(() => {
      expect(screen.getByText(/Promoted to .* \(\+34\.4%\)/i)).toBeInTheDocument();
    });
  });
});

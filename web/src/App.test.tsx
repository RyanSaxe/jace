/** Behavioral UI tests for Jace Studio navigation, runs list, games search, and Tabletop rendering. */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "@/App";

const MOCK_RUNS = {
  runs: [
    {
      run_id: "v2-r89-flagship-w384l6",
      experiment: "v2-r89-flagship-w384l6",
      name: "v2-r89-flagship-w384l6",
      status: "running",
      latest_update: 219,
      total_updates: 2000,
      latest_tps: 2318,
      latest_live_win: 0.755,
      latest_ema_win: 0.72,
      latest_ensemble_win: 0.775,
      peak_live_win: 0.755,
      peak_ema_win: 0.72,
      hidden_dim: 384,
      attention_layers: 6,
      eval_files: 4,
      checkpoints: [100, 400],
      updated_at: 1700000000,
    },
  ],
};

const MOCK_GAMES = {
  total: 1,
  games: [
    {
      run_id: "v2-r89-flagship-w384l6",
      iteration: 400,
      game_idx: 0,
      set_code: "TLA",
      seed: 20001,
      result: "win",
      winner_seat: 0,
      steps: 248,
      turns: 17,
      hero_name: "hero_u400",
      opp_name: "heuristic",
      hero_colors: "WBR",
      opp_colors: "WR",
      cards_seen: ["Fire Lord Zuko", "Plains"],
      key_cards: ["Fire Lord Zuko"],
    },
  ],
};

function installFakeFetch(): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: string | URL) => {
      const url = String(input);
      let body: unknown = { items: [] };
      if (url.startsWith("/api/runs?")) body = MOCK_RUNS;
      else if (url === "/api/runs") body = MOCK_RUNS;
      else if (url.startsWith("/api/runs/")) {
        body = {
          summary: MOCK_RUNS.runs[0],
          config: {},
          eval_points: [{ update: 200, live_held_out_det_win_rate: 0.755 }],
          train_points: [{ update: 219, value_loss: 0.42, entropy: 0.4, tps: 2318 }],
          checkpoints: [{ iteration: 400, name: "iter-0400", path: "/tmp/a", size_bytes: 10 }],
          eval_iterations: [400],
        };
      } else if (url.startsWith("/api/games")) body = MOCK_GAMES;
      else if (url.startsWith("/api/cards-catalog")) body = { cards: ["Fire Lord Zuko"] };
      else if (url.startsWith("/api/game-detail")) {
        body = {
          run_id: "v2-r89-flagship-w384l6",
          iteration: 400,
          game_idx: 0,
          seed: 20001,
          winner: 0,
          total_steps: 248,
          peak_step: 237,
          players: [],
          timeline: [
            {
              turn: 17,
              step: 237,
              hero_life: 19,
              opp_life: 4,
              hero_perms: 13,
              opp_perms: 12,
              stack_len: 2,
              eval_win: 0.84,
            },
          ],
        };
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

describe("Jace Studio App", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders discovered runs and switches to Games Explorer", async () => {
    installFakeFetch();
    render(<App />);

    await waitFor(() => {
      expect(screen.getAllByText("v2-r89-flagship-w384l6").length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText("75.5%").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: /^Games/i }));
    await waitFor(() => {
      expect(screen.getByText(/Seed #20001/i)).toBeInTheDocument();
    });
  });

  it("toggles Warm Paper and Obsidian Dark theme cleanly", async () => {
    installFakeFetch();
    render(<App />);
    const themeBtn = screen.getByRole("button", { name: /Warm Paper/i });
    fireEvent.click(themeBtn);
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(screen.getByRole("button", { name: /Obsidian Dark/i })).toBeInTheDocument();
  });
});

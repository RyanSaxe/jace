/** Behavioral tests for Tabletop Policy/Search toggle, world budget, Turn-0 V0 strip, Belief Editor drawer, and HTTP 422 handling. */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TabletopView } from "@/components/TabletopView";
import type {
  BeliefInferResponse,
  BeliefProjectResponse,
  BoardState,
  EvaluationPair,
  GameTarget,
  SearchEvaluateResponse,
} from "@/lib/types";

const TARGET: GameTarget = {
  run_id: "exp-test/run-a",
  iteration: 2000,
  game_idx: 0,
  step: 0,
  seed: 20001,
  ckpt_a: "exp-test/run-a::iter-2000",
};

function makeBoard(overrides?: Partial<BoardState>): BoardState {
  const emptyPlayer = {
    seat: 0,
    life: 20,
    is_active: true,
    zone_counts: { library: 33, hand: 7, battlefield: 0, graveyard: 0, exile: 0 },
    lands: [],
    creatures: [],
    other_permanents: [],
    hand: ["Plains", "Mountain", "Fire Lord Zuko"],
    hand_hidden: false,
    graveyard: [],
    exile: [],
    library_summary: [],
    library_hidden: false,
  };
  return {
    step: 0,
    total_steps: 40,
    turn: 0,
    phase: "Pregame",
    step_name: "Mulligan",
    action_space_kind: "MULLIGAN",
    hero_seat: 0,
    acting_seat: 0,
    actor_is_hero: true,
    active_is_hero: true,
    actions_hidden: false,
    chosen_action: 0,
    pov: "hero",
    hero: emptyPlayer,
    opp: {
      ...emptyPlayer,
      seat: 1,
      is_active: false,
      zone_counts: { library: 32, hand: 8, battlefield: 0, graveyard: 0, exile: 0 },
      hand: [],
      hand_hidden: true,
      library_hidden: true,
    },
    stack: [],
    legal_actions: [
      { index: 0, label: "Keep 7", action_type: "MULLIGAN_KEEP", focus_ids: [] },
      { index: 1, label: "Mulligan to 6", action_type: "MULLIGAN_MULLIGAN", focus_ids: [] },
    ],
    recent_events: [],
    non_permanent_names: [],
    ...overrides,
  };
}

const EVAL_PAIR: EvaluationPair = {
  primary: {
    checkpoint: "exp-test/run-a::iter-2000",
    probs: [0.72, 0.28],
    wdl: { win: 0.612, draw: 0, loss: 0.388 },
  },
  compare: {
    checkpoint: "exp-test/run-a::iter-2000",
    probs: [0.72, 0.28],
    wdl: { win: 0.612, draw: 0, loss: 0.388 },
  },
  diff: { tv: 0, kl: 0, argmax_changed: false },
  warnings: [],
};

const INFER_RESPONSE: BeliefInferResponse = {
  card_names: ["Fire Lord Zuko", "Boiling Rock Rioter", "Lightning Strike"],
  hand_probs: [0.35, 0.25, 0.18],
  deck_copies: [1.4, 1.1, 0.8],
  revealed_counts: [0, 1, 0],
  cards: [
    {
      name: "Fire Lord Zuko",
      hand_prob: 0.35,
      deck_copies: 1.4,
      revealed_count: 0,
      pinned: 0,
      excluded: false,
      multiplier: 1,
    },
    {
      name: "Boiling Rock Rioter",
      hand_prob: 0.25,
      deck_copies: 1.1,
      revealed_count: 1,
      pinned: 0,
      excluded: false,
      multiplier: 1,
    },
    {
      name: "Lightning Strike",
      hand_prob: 0.18,
      deck_copies: 0.8,
      revealed_count: 0,
      pinned: 0,
      excluded: false,
      multiplier: 1,
    },
  ],
  opp_hand_size: 8,
  opp_library_size: 32,
  pinned_total: 0,
  unedited_count: 3,
  unedited_mass: 1,
  prior_backend: "dynamic_run_decks",
  matchup_equity: 0.54,
  v0_equity: 0.54,
  v0: {
    is_turn0: true,
    v0_equity: 0.54,
    keep_v: 0.612,
    delta: 0.072,
    recommendation: "Keep",
  },
};

const PROJECTED_PIN_1: BeliefProjectResponse = {
  cards: [
    {
      name: "Fire Lord Zuko",
      hand_prob: 1,
      deck_copies: 1.8,
      revealed_count: 0,
      pinned: 1,
      excluded: false,
      multiplier: 1,
    },
    {
      name: "Boiling Rock Rioter",
      hand_prob: 0.22,
      deck_copies: 1.05,
      revealed_count: 1,
      pinned: 0,
      excluded: false,
      multiplier: 1,
    },
    {
      name: "Lightning Strike",
      hand_prob: 0.16,
      deck_copies: 0.76,
      revealed_count: 0,
      pinned: 0,
      excluded: false,
      multiplier: 1,
    },
  ],
  opp_hand_size: 8,
  opp_library_size: 32,
  pinned_total: 1,
  unedited_count: 2,
  unedited_mass: 0.875,
};

function makeSearchResponse(worlds: number, keepWr: number): SearchEvaluateResponse {
  return {
    chosen_action: 0,
    actions: [
      {
        index: 0,
        label: "Keep 7",
        search_wr: keepWr,
        ci_low: Math.max(0, keepWr - 0.09),
        ci_high: Math.min(1, keepWr + 0.09),
        policy_prob: 0.72,
        delta_vs_policy: 0,
      },
      {
        index: 1,
        label: "Mulligan to 6",
        search_wr: 0.44,
        ci_low: 0.34,
        ci_high: 0.54,
        policy_prob: 0.28,
        delta_vs_policy: 0.44 - keepWr,
      },
    ],
    worlds_sampled: worlds,
    rollout_horizon: 24,
    elapsed_ms: 148.4,
    opp_hand_size: 8,
    v0: INFER_RESPONSE.v0,
    v0_equity: 0.54,
  };
}

function makeJsonResp(body: unknown, status = 200): Promise<unknown> {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 422 ? "Unprocessable Entity" : "OK",
    json: (): Promise<unknown> => Promise.resolve(body),
  });
}

const MOCK_RUN_DETAIL = {
  summary: {
    run_id: TARGET.run_id,
    experiment: "exp-test",
    name: "run-a",
    status: "done",
    latest_update: 2000,
    total_updates: 2000,
    latest_tps: 0,
    latest_live_win: 0.8,
    latest_ema_win: 0.82,
    latest_ensemble_win: null,
    peak_live_win: 0.82,
    peak_ema_win: 0.84,
    hidden_dim: 160,
    attention_layers: 2,
    eval_files: 1,
    checkpoints: [2000],
    updated_at: 0,
  },
  config: {},
  eval_points: [],
  train_points: [],
  checkpoints: [{ iteration: 2000, name: "iter-2000", path: "", size_bytes: 1 }],
  eval_iterations: [2000],
};

function parseJsonBody(init?: RequestInit): Record<string, unknown> {
  if (typeof init?.body !== "string") return {};
  const parsed: unknown = JSON.parse(init.body);
  if (typeof parsed !== "object" || parsed === null) return {};
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(parsed)) out[k] = v;
  return out;
}

function resolveStaticEndpoint(url: string): Promise<unknown> | null {
  if (url.startsWith("/api/runs/")) return makeJsonResp(MOCK_RUN_DETAIL);
  if (url.startsWith("/api/cards-catalog")) {
    return makeJsonResp({ cards: INFER_RESPONSE.card_names });
  }
  if (url.startsWith("/api/game-detail")) {
    return makeJsonResp({
      run_id: TARGET.run_id,
      iteration: 2000,
      game_idx: 0,
      seed: 20001,
      winner: 0,
      total_steps: 40,
      max_playable_step: 39,
      peak_step: 0,
      players: [],
      timeline: [],
    });
  }
  if (url.startsWith("/api/game-step")) {
    return makeJsonResp({
      board: makeBoard(),
      evaluation: EVAL_PAIR,
      replay_warning: null,
      nearest_priority_step: 1,
    });
  }
  if (url === "/api/belief/infer") return makeJsonResp(INFER_RESPONSE);
  return null;
}

describe("TabletopBelief & Search UI", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("toggles Policy/Search, switches world budget, displays Turn-0 V0, edits belief with 422 error, and applies overrides to Search", async () => {
    const searchBodies: Array<Record<string, unknown>> = [];
    let projectCalls = 0;

    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL, init?: RequestInit) => {
        const url = String(input);
        const staticResp = resolveStaticEndpoint(url);
        if (staticResp !== null) return staticResp;
        if (url === "/api/belief/project") {
          projectCalls += 1;
          return projectCalls === 1
            ? makeJsonResp(PROJECTED_PIN_1)
            : makeJsonResp({ detail: "Pinned hand cards (9) exceed opp_hand_size (8)." }, 422);
        }
        if (url === "/api/search/evaluate") {
          const rec = parseJsonBody(init);
          searchBodies.push(rec);
          const worlds = typeof rec.worlds === "number" ? rec.worlds : 16;
          return makeJsonResp(makeSearchResponse(worlds, rec.override ? 0.58 : 0.688));
        }
        return makeJsonResp({});
      }),
    );

    render(
      <TabletopView
        target={TARGET}
        checkpoints={[]}
        watchlist={[]}
        onChangeStep={vi.fn()}
        onPinPosition={vi.fn(() => Promise.resolve())}
        onBackToGames={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText(/Turn 0 · Opening Hand vs V₀/i)).toBeInTheDocument();
    });

    // Switch from Policy to Search and inspect Turn-0 V0 + Search results.
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => {
      expect(screen.getByText(/Search \(16w · 24p\)/i)).toBeInTheDocument();
    });
    expect(screen.getByText("68.8%")).toBeInTheDocument();
    expect(screen.getByText("148 ms")).toBeInTheDocument();
    expect(screen.getByText("54.0%")).toBeInTheDocument();
    expect(screen.getByText("+7.2%")).toBeInTheDocument();
    expect(screen.getByText("Keep")).toBeInTheDocument();

    // Switch world budget to 32w.
    fireEvent.click(screen.getByRole("button", { name: "32w" }));
    await waitFor(() => {
      expect(screen.getByText(/Search \(32w · 24p\)/i)).toBeInTheDocument();
    });

    // Open Edit Belief drawer, verify dynamic 8-card hand (no 7-card cap), search cards, and pin one copy.
    fireEvent.click(screen.getByRole("button", { name: /Edit Belief/i }));
    await waitFor(() => {
      expect(screen.getByText(/Opp Hand:/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Opp Hand:/i)).toHaveTextContent(
      /Opp Hand:\s*8 cards \(0 pinned\) · Library:\s*32/i,
    );
    expect(screen.getByText(/Unedited cards kept in distribution:/i)).toBeInTheDocument();

    const searchInput = screen.getByLabelText("Search format cards");
    fireEvent.change(searchInput, { target: { value: "Zuko" } });
    expect(screen.getByText("Fire Lord Zuko")).toBeInTheDocument();
    expect(screen.queryByText("Lightning Strike")).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Increase pin for Fire Lord Zuko"));
    await waitFor(() => {
      expect(screen.getByText(/88% hand mass/i)).toBeInTheDocument();
    });

    // Apply & Re-score sends active BeliefOverride into /api/search/evaluate.
    fireEvent.click(screen.getByRole("button", { name: /Apply & Re-score/i }));
    await waitFor(() => {
      expect(screen.getByText("58.0%")).toBeInTheDocument();
    });
    const lastSearch = searchBodies[searchBodies.length - 1];
    expect(lastSearch?.override).toEqual({
      pinned_in_hand: { "Fire Lord Zuko": 1 },
      excluded_cards: [],
      copy_multipliers: {},
    });

    // Triggering an over-pin returns HTTP 422 and displays the inline alert without corrupting the last valid override.
    fireEvent.click(screen.getByLabelText("Increase pin for Fire Lord Zuko"));
    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        /Pinned hand cards \(9\) exceed opp_hand_size \(8\)/i,
      );
    });
    expect(screen.getByRole("button", { name: /Edit Belief/i })).toHaveTextContent("(1)");
  });
});

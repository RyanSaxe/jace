/** Arena leaderboard, match history, tactical puzzles, and Benchmarks route TypeScript interfaces. */

import type { BenchmarkSubView } from "./runs";

export interface ArenaModelRow {
  model_id: string;
  label: string;
  elo: number;
  ci_half: number;
  matches: number;
  games: number;
  vs_anchor_wr: number | null;
  updates: number | null;
  search_ms_per_dec: number | null;
  checkpoint_ref: string | null;
}

export interface ArenaForecastPoint {
  run_id: string;
  early_wr: number;
  peak_wr: number;
}

export interface ArenaForecast {
  n: number;
  pearson_r: number;
  spearman_rho: number;
  points: ArenaForecastPoint[];
  power_law_pred: number | null;
}

export interface ArenaLeaderboardResponse {
  format: string;
  formats: string[];
  anchor_model: string;
  anchor_elo: number;
  models: ArenaModelRow[];
  forecast: ArenaForecast | null;
}

export interface ArenaMatchRow {
  match_id: string;
  opponent_id: string;
  opponent_label: string;
  games: number;
  winrate: number;
  ci_low: number;
  ci_high: number;
  sweeps_w: number;
  sweeps_l: number;
  los: number;
  compare_target: { ckptA: string; ckptB: string } | null;
}

export type PuzzleSource = "auto_mined" | "search_disagreement" | "watchlist";

export interface PuzzleActionRow {
  label: string;
  playout_wr: number;
  prob_a: number | null;
  prob_b: number | null;
  is_best: boolean;
}

export interface PuzzleCardItem {
  puzzle_id: string;
  format: string;
  turn: number;
  phase: string;
  source: PuzzleSource;
  seed_label: string;
  hero_summary: string;
  opp_summary: string;
  key_cards: string[];
  actions: PuzzleActionRow[];
  margin: number;
  ci_low: number;
  ci_high: number;
  playouts: number;
  tabletop_target: {
    run_id: string;
    iteration: number;
    game_idx: number;
    step: number;
    seed: number;
  };
}

export interface PuzzlesResponse {
  format: string;
  formats: string[];
  puzzles: PuzzleCardItem[];
  solved_a: number;
  solved_b: number;
  total: number;
}

export type PromotePuzzleResult =
  | { admitted: true; puzzle: PuzzleCardItem }
  | { admitted: false; detail: string; margin: number; ci_low: number; ci_high: number };

export interface BenchmarksRouteState {
  subView: BenchmarkSubView;
  format: string;
  anchor: string;
  model: string;
  modelA: string;
  modelB: string;
}

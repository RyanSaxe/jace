/** Shared studio UI, run catalog, game index, watchlist, and navigation TypeScript interfaces. */

export type ThemeMode = "light" | "dark";
export type NavTab = "runs" | "games" | "benchmarks" | "compare" | "tabletop" | "watchlist";
export type PovMode = "hero" | "omniscient";
export type BenchmarkSubView = "arena" | "puzzles";

export interface CheckpointInfo {
  iteration: number;
  name: string;
  path: string;
  size_bytes: number;
}

export interface GlobalCheckpointOption {
  ref: string;
  run_id: string;
  run_name: string;
  experiment: string;
  checkpoint: string;
  iteration: number;
  size_bytes: number;
}

export interface RunSummary {
  run_id: string;
  experiment: string;
  name: string;
  status: string;
  latest_update: number;
  total_updates: number;
  latest_tps: number;
  latest_live_win: number | null;
  latest_ema_win: number | null;
  latest_ensemble_win: number | null;
  peak_live_win: number | null;
  peak_ema_win: number | null;
  peak_ensemble_win?: number | null;
  hidden_dim: number | null;
  attention_layers: number | null;
  eval_files: number;
  checkpoints: number[];
  updated_at: number;
  playable_games?: number;
  legacy_games?: number;
  latest_metrics?: Record<string, number>;
  best_metrics?: Record<string, number>;
  eval_metric_keys?: string[];
  train_metric_keys?: string[];
  config_tags?: Record<string, unknown>;
}

export interface RunsCatalogResponse {
  runs: RunSummary[];
  experiments: string[];
  eval_metric_keys: string[];
  train_metric_keys: string[];
}

export interface MetricPoint {
  update: number;
  [key: string]: unknown;
}

export interface RunDetail {
  summary: RunSummary;
  config: Record<string, unknown>;
  eval_points: MetricPoint[];
  train_points: MetricPoint[];
  checkpoints: CheckpointInfo[];
  eval_iterations: number[];
  eval_metric_keys?: string[];
  train_metric_keys?: string[];
}

export interface GameSummary {
  run_id: string;
  iteration: number;
  game_idx: number;
  set_code: string;
  seed: number;
  result: string;
  winner_seat: number;
  hero_seat: number;
  steps: number;
  turns: number;
  hero_name: string;
  opp_name: string;
  hero_colors: string;
  opp_colors: string;
  cards_seen: string[];
  key_cards: string[];
  playable?: boolean;
  max_pred_win?: number;
  min_pred_win?: number;
  surprise_score?: number;
  max_swing?: number;
  turning_point_step?: number;
  eval_source?: string;
}

export interface GameFacets {
  runs: string[];
  colors: string[];
  sets: string[];
  results: string[];
  cards: string[];
  total_playable: number;
  total_unplayable: number;
}

export interface TimelinePoint {
  turn: number;
  step: number;
  hero_life: number;
  opp_life: number;
  hero_perms: number;
  opp_perms: number;
  stack_len: number;
  eval_win: number;
}

export interface GameOverview {
  run_id: string;
  iteration: number;
  game_idx: number;
  seed: number;
  hero_seat?: number;
  winner: number | null;
  total_steps: number;
  max_playable_step?: number;
  peak_step: number;
  turning_point_step?: number;
  eval_source?: string;
  replay_warning?: string | null;
  players: Array<{ name: string; deck_size: number }>;
  timeline: TimelinePoint[];
}

export interface WatchlistItem {
  id: string;
  run_id: string;
  iteration: number;
  game_idx: number;
  step: number;
  turn: number;
  seed: number;
  label: string;
  note: string;
  tag?: string;
  summary?: string;
  key_cards?: string[];
}

export interface CardMetadata {
  name: string;
  set_code: string;
  mana_cost: string;
  type_line: string;
  oracle_text: string;
  is_token: boolean;
}

export interface GameTarget {
  run_id: string;
  iteration: number;
  game_idx: number;
  step: number;
  seed: number;
  ckpt_a?: string;
  ckpt_b?: string;
}

export interface CompareTarget {
  ckptA: string;
  ckptB: string;
  runId?: string;
}

export interface GamesFilterState {
  runId: string;
  setCode: string;
  result: string;
  colors: string;
  card: string;
  minSurprise: number | null;
  minSwing: number | null;
  search: string;
  sort: string;
  page: number;
}

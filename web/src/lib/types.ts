/** Domain TypeScript interfaces mirroring the Jace Studio FastAPI JSON contracts. */

export type ThemeMode = "light" | "dark";
export type NavTab = "runs" | "games" | "tabletop" | "watchlist";
export type PovMode = "hero" | "omniscient";

export interface CheckpointInfo {
  iteration: number;
  name: string;
  path: string;
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
  hidden_dim: number | null;
  attention_layers: number | null;
  eval_files: number;
  checkpoints: number[];
  updated_at: number;
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

export interface LandPile {
  name: string;
  tapped: boolean;
  count: number;
  id: number | null;
}

export interface PermanentState {
  id: number;
  name: string;
  is_land: boolean;
  is_creature: boolean;
  is_artifact: boolean;
  is_enchantment: boolean;
  is_token: boolean;
  is_legendary: boolean;
  power: number;
  toughness: number;
  damage: number;
  plus1_counters: number;
  tapped: boolean;
  sick: boolean;
  attacking?: boolean;
  blocking?: boolean;
  blocking_id?: number | null;
  mana_value: number;
}

export interface ZoneCounts {
  library: number;
  hand: number;
  battlefield: number;
  graveyard: number;
  exile: number;
}

export interface LibrarySummaryRow {
  name: string;
  count: number;
}

export interface PlayerState {
  seat: number;
  life: number;
  is_active?: boolean;
  mana_pool?: string;
  open_lands?: number;
  zone_counts: ZoneCounts;
  lands: LandPile[];
  creatures: PermanentState[];
  other_permanents: PermanentState[];
  hand: string[];
  hand_hidden: boolean;
  graveyard: string[];
  exile: string[];
  library_summary: LibrarySummaryRow[];
  library_hidden: boolean;
}

export interface StackItem {
  index: number;
  card_name: string;
  description: string;
  kind: string;
  source_permanent_id: number | null;
  controller_id: number;
  controller_side?: "hero" | "opp";
}

export interface LegalAction {
  index: number;
  label: string;
  action_type: string;
  focus_ids: number[];
}

export interface RecentEvent {
  step: number;
  turn: number;
  actor_side: "hero" | "opp";
  text: string;
}

export interface BoardState {
  step: number;
  total_steps: number;
  turn: number;
  phase: string;
  step_name: string;
  action_space_kind: string;
  hero_seat?: number;
  acting_seat: number;
  actor_is_hero?: boolean;
  active_is_hero?: boolean;
  actions_hidden?: boolean;
  chosen_action: number | null;
  pov: PovMode;
  hero: PlayerState;
  opp: PlayerState;
  stack: StackItem[];
  legal_actions: LegalAction[];
  recent_events?: RecentEvent[];
  non_permanent_names?: string[];
}

export interface WdlEstimate {
  win: number;
  draw: number;
  loss: number;
}

export interface CheckpointEval {
  checkpoint: string;
  probs: number[];
  wdl: WdlEstimate;
}

export interface EvaluationPair {
  primary: CheckpointEval;
  compare: CheckpointEval;
  diff: {
    tv: number;
    kl: number;
    argmax_changed: boolean;
  };
  warnings?: string[];
}

export interface GameStepResponse {
  board: BoardState;
  evaluation: EvaluationPair;
  replay_warning?: string | null;
  nearest_priority_step?: number | null;
}

export interface ForkEdit {
  op: "set_life" | "clear_hand" | "add_hand_card" | "add_battlefield";
  seat: 0 | 1;
  value?: number;
  card?: string;
  after_branch?: number;
}

export interface BranchStepEvent {
  branch_index: number;
  turn: number;
  actor_side: "hero" | "opp";
  action_index: number;
  label: string;
}

export interface ForkResponse {
  board: BoardState;
  base_evaluation: EvaluationPair;
  fork_evaluation: EvaluationPair;
  warnings: string[];
  branch_events?: BranchStepEvent[];
  branch_done?: boolean;
  nearest_priority_step?: number | null;
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

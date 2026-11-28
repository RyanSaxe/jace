/** Domain TypeScript interfaces mirroring Jace Studio FastAPI JSON contracts (runs, replays, forks, benchmarks, belief, search). */

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

export interface LandPile {
  name: string;
  tapped: boolean;
  count: number;
  id: number | null;
}

export interface PermanentState {
  id: number;
  name: string;
  copy_index?: number | null;
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

export type ForkEditOp =
  | "set_life"
  | "clear_hand"
  | "add_hand_card"
  | "remove_hand_card"
  | "swap_hand_card"
  | "add_battlefield";

export interface ForkEdit {
  op: ForkEditOp;
  seat: 0 | 1;
  value?: number;
  card?: string;
  replacement_card?: string;
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

export interface PolicyMatrixColumn {
  ref: string;
  label: string;
  probs: number[];
  wdl: WdlEstimate;
  top_action_index: number | null;
  warning?: string | null;
}

export interface PolicyMatrixResponse {
  legal_actions: LegalAction[];
  columns: PolicyMatrixColumn[];
}

export interface ActionPlayoutResult {
  action_index: number;
  label: string;
  deals_done: number;
  wins: number;
  draws: number;
  losses: number;
  win_rate: number;
  ci_low: number;
  ci_high: number;
}

export interface PlayoutBatchResponse {
  model_ref: string;
  device: string;
  deals_done: number;
  batch_offset: number;
  elapsed_ms: number;
  results: ActionPlayoutResult[];
}

export interface CompareDecisionKindStat {
  kind: string;
  count: number;
  changed: number;
  change_rate: number;
  mean_tv: number;
}

export interface ComparePositionRow {
  run_id: string;
  iteration: number;
  game_idx: number;
  seed: number;
  step: number;
  turn: number;
  decision_kind: string;
  actor_side: "hero" | "opp";
  move_changed: boolean;
  tv: number;
  kl: number;
  top_action_a: string;
  top_prob_a: number;
  top_action_b: string;
  top_prob_b: number;
  win_a: number;
  win_b: number;
  win_delta: number;
  hero_life: number;
  opp_life: number;
  key_cards: string[];
  legal_actions: LegalAction[];
  probs_a: number[];
  probs_b: number[];
}

export interface CompareResponse {
  ckpt_a: string;
  ckpt_b: string;
  sampled_positions: number;
  moves_changed: number;
  change_rate: number;
  mean_tv: number;
  mean_kl: number;
  mean_win_delta: number;
  device: string;
  warnings: string[];
  by_decision_kind: CompareDecisionKindStat[];
  positions: ComparePositionRow[];
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

export type DockAnalysisMode = "policy" | "search";

export type SearchWorldBudget = 8 | 16 | 32 | 64;

export interface BeliefOverridePayload {
  pinned_in_hand: Record<string, number>;
  excluded_cards: string[];
  copy_multipliers: Record<string, number>;
}

export interface BeliefCardEntry {
  name: string;
  hand_prob: number;
  deck_copies: number;
  revealed_count: number;
  pinned: number;
  excluded: boolean;
  multiplier: number;
}

export interface TurnZeroV0Info {
  is_turn0: boolean;
  v0_equity: number;
  keep_v: number;
  delta: number;
  recommendation: "Keep" | "Mulligan";
}

export interface BeliefInferResponse {
  card_names: string[];
  hand_probs: number[];
  deck_copies: number[];
  revealed_counts: number[];
  cards: BeliefCardEntry[];
  opp_hand_size: number;
  opp_library_size: number;
  pinned_total: number;
  unedited_count: number;
  unedited_mass: number;
  prior_backend: string;
  has_belief_net?: boolean;
  acting_seat?: number;
  perspective?: "hero" | "opponent";
  matchup_equity: number;
  v0_equity: number;
  v0: TurnZeroV0Info | null;
}

export interface BeliefProjectResponse {
  cards: BeliefCardEntry[];
  opp_hand_size: number;
  opp_library_size: number;
  pinned_total: number;
  unedited_count: number;
  unedited_mass: number;
  has_belief_net?: boolean;
  acting_seat?: number;
  perspective?: "hero" | "opponent";
}

export type BeliefProjectResult =
  | { ok: true; data: BeliefProjectResponse }
  | { ok: false; status: number; detail: string };

export interface SearchActionRow {
  index: number;
  label: string;
  search_wr: number;
  ci_low: number;
  ci_high: number;
  policy_prob: number;
  delta_vs_policy: number;
}

export interface SearchEvaluateResponse {
  chosen_action: number;
  actions: SearchActionRow[];
  worlds_sampled: number;
  rollout_horizon: number;
  elapsed_ms: number;
  opp_hand_size: number;
  acting_seat?: number;
  perspective?: "hero" | "opponent";
  warnings?: string[];
  v0?: TurnZeroV0Info | null;
  v0_equity?: number | null;
}

export type SearchEvaluateResult =
  | { ok: true; data: SearchEvaluateResponse }
  | { ok: false; status: number; detail: string };

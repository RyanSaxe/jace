/** Tabletop board state, policy evaluation, counterfactual fork, playout, and model comparison TypeScript interfaces. */

import type { PovMode } from "./runs";

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

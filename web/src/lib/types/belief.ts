/** Closed-decklist opponent belief, Turn-0 V0 baseline, and local search TypeScript interfaces. */

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

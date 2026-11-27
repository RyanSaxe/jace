/** Typed HTTP client endpoints for opponent belief inference, override projection, and local search evaluation. */

import type {
  BeliefInferResponse,
  BeliefOverridePayload,
  BeliefProjectResponse,
  BeliefProjectResult,
  SearchEvaluateResponse,
  SearchEvaluateResult,
} from "@/lib/types";
import { extractApiDetail, fetchJson, isRecord } from "./core";

function isBeliefInferResponse(data: unknown): data is BeliefInferResponse {
  return (
    isRecord(data) &&
    Array.isArray(data.card_names) &&
    Array.isArray(data.hand_probs) &&
    typeof data.opp_hand_size === "number"
  );
}

function isBeliefProjectResponse(data: unknown): data is BeliefProjectResponse {
  return (
    isRecord(data) &&
    Array.isArray(data.cards) &&
    typeof data.opp_hand_size === "number" &&
    typeof data.pinned_total === "number"
  );
}

function isSearchEvaluateResponse(data: unknown): data is SearchEvaluateResponse {
  return (
    isRecord(data) &&
    Array.isArray(data.actions) &&
    typeof data.worlds_sampled === "number" &&
    typeof data.elapsed_ms === "number"
  );
}

export async function postBeliefInfer(
  args: {
    runId: string;
    iteration: number;
    gameIdx: number;
    step: number;
    formatId?: string;
    priorOnly?: boolean;
  },
  signal?: AbortSignal,
): Promise<BeliefInferResponse> {
  return fetchJson("/api/belief/infer", isBeliefInferResponse, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      run_id: args.runId,
      iteration: args.iteration,
      game_idx: args.gameIdx,
      step: args.step,
      format_id: args.formatId ?? null,
      prior_only: args.priorOnly ?? false,
    }),
  });
}

export async function postBeliefProject(
  args: {
    runId: string;
    iteration: number;
    gameIdx: number;
    step: number;
    formatId?: string;
    priorOnly?: boolean;
    override: BeliefOverridePayload;
  },
  signal?: AbortSignal,
): Promise<BeliefProjectResult> {
  const response = await fetch("/api/belief/project", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      run_id: args.runId,
      iteration: args.iteration,
      game_idx: args.gameIdx,
      step: args.step,
      format_id: args.formatId ?? null,
      prior_only: args.priorOnly ?? false,
      override: args.override,
    }),
  });
  const data: unknown = await response.json();
  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      detail: extractApiDetail(data, `API ${response.status}: ${response.statusText}`),
    };
  }
  if (!isBeliefProjectResponse(data)) {
    return {
      ok: false,
      status: response.status,
      detail: "Unexpected response shape from /api/belief/project",
    };
  }
  return { ok: true, data };
}

export async function postSearchEvaluate(
  args: {
    runId: string;
    iteration: number;
    gameIdx: number;
    step: number;
    ckptA: string;
    worlds?: number;
    horizonPlies?: number;
    priorOnly?: boolean;
    seed?: number;
    override?: BeliefOverridePayload | null;
  },
  signal?: AbortSignal,
): Promise<SearchEvaluateResult> {
  const response = await fetch("/api/search/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      run_id: args.runId,
      iteration: args.iteration,
      game_idx: args.gameIdx,
      step: args.step,
      ckpt_a: args.ckptA,
      worlds: args.worlds ?? 16,
      horizon_plies: args.horizonPlies ?? 24,
      prior_only: args.priorOnly ?? false,
      seed: args.seed ?? 42,
      override: args.override ?? null,
    }),
  });
  const data: unknown = await response.json();
  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      detail: extractApiDetail(data, `API ${response.status}: ${response.statusText}`),
    };
  }
  if (!isSearchEvaluateResponse(data)) {
    return {
      ok: false,
      status: response.status,
      detail: "Unexpected response shape from /api/search/evaluate",
    };
  }
  return { ok: true, data };
}

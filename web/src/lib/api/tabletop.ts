/** Typed HTTP client endpoints for Tabletop replays, counterfactual forks, policy matrix, playouts, and model comparison. */

import type {
  CompareResponse,
  ForkEdit,
  ForkResponse,
  GameOverview,
  GameStepResponse,
  PlayoutBatchResponse,
  PolicyMatrixResponse,
  PovMode,
} from "@/lib/types";
import { fetchJson, isRecord } from "./core";

function isGameOverview(data: unknown): data is GameOverview {
  return isRecord(data) && Array.isArray(data.timeline) && typeof data.total_steps === "number";
}

function isGameStepResponse(data: unknown): data is GameStepResponse {
  return isRecord(data) && isRecord(data.board) && isRecord(data.evaluation);
}

function isForkResponse(data: unknown): data is ForkResponse {
  return isRecord(data) && isRecord(data.board) && isRecord(data.fork_evaluation);
}

function isPolicyMatrixResponse(data: unknown): data is PolicyMatrixResponse {
  return isRecord(data) && Array.isArray(data.legal_actions) && Array.isArray(data.columns);
}

function isPlayoutBatchResponse(data: unknown): data is PlayoutBatchResponse {
  return isRecord(data) && Array.isArray(data.results) && typeof data.deals_done === "number";
}

function isCompareResponse(data: unknown): data is CompareResponse {
  return isRecord(data) && Array.isArray(data.positions) && Array.isArray(data.by_decision_kind);
}

export async function fetchGameOverview(args: {
  runId: string;
  iteration: number;
  gameIdx: number;
}): Promise<GameOverview> {
  const params = new URLSearchParams({
    run_id: args.runId,
    iteration: String(args.iteration),
    game_idx: String(args.gameIdx),
  });
  return fetchJson(`/api/game-detail?${params.toString()}`, isGameOverview);
}

export async function fetchGameStep(args: {
  runId: string;
  iteration: number;
  gameIdx: number;
  step: number;
  pov: PovMode;
  ckptA?: string;
  ckptB?: string;
}): Promise<GameStepResponse> {
  const params = new URLSearchParams({
    run_id: args.runId,
    iteration: String(args.iteration),
    game_idx: String(args.gameIdx),
    step: String(args.step),
    pov: args.pov,
  });
  if (args.ckptA) params.set("ckpt_a", args.ckptA);
  if (args.ckptB) params.set("ckpt_b", args.ckptB);
  return fetchJson(`/api/game-step?${params.toString()}`, isGameStepResponse);
}

export async function postFork(args: {
  runId: string;
  iteration: number;
  gameIdx: number;
  step: number;
  pov: PovMode;
  ckptA?: string;
  ckptB?: string;
  edits: ForkEdit[];
  branchActions?: number[];
}): Promise<ForkResponse> {
  return fetchJson("/api/fork", isForkResponse, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run_id: args.runId,
      iteration: args.iteration,
      game_idx: args.gameIdx,
      step: args.step,
      pov: args.pov,
      ckpt_a: args.ckptA ?? null,
      ckpt_b: args.ckptB ?? null,
      edits: args.edits,
      branch_actions: args.branchActions ?? [],
    }),
  });
}

export async function postPolicyMatrix(args: {
  runId: string;
  iteration: number;
  gameIdx: number;
  step: number;
  pov: PovMode;
  edits?: ForkEdit[];
  branchActions?: number[];
  models: string[];
}): Promise<PolicyMatrixResponse> {
  return fetchJson("/api/matrix", isPolicyMatrixResponse, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run_id: args.runId,
      iteration: args.iteration,
      game_idx: args.gameIdx,
      step: args.step,
      pov: args.pov,
      edits: args.edits ?? [],
      branch_actions: args.branchActions ?? [],
      models: args.models,
    }),
  });
}

export async function postPlayoutBatch(
  args: {
    runId: string;
    iteration: number;
    gameIdx: number;
    step: number;
    pov: PovMode;
    edits?: ForkEdit[];
    branchActions?: number[];
    modelRef: string;
    deals: number;
    batchOffset?: number;
    actionIndices?: number[];
    seed?: number;
  },
  signal?: AbortSignal,
): Promise<PlayoutBatchResponse> {
  return fetchJson("/api/playout", isPlayoutBatchResponse, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      run_id: args.runId,
      iteration: args.iteration,
      game_idx: args.gameIdx,
      step: args.step,
      pov: args.pov,
      edits: args.edits ?? [],
      branch_actions: args.branchActions ?? [],
      model_ref: args.modelRef,
      deals: args.deals,
      batch_offset: args.batchOffset ?? 0,
      action_indices: args.actionIndices ?? null,
      seed: args.seed ?? 42,
    }),
  });
}

export async function postCompareModels(
  args: {
    ckptA: string;
    ckptB: string;
    sampleCount: number;
    runId?: string;
    decisionKind?: string;
    seed?: number;
  },
  signal?: AbortSignal,
): Promise<CompareResponse> {
  return fetchJson("/api/compare", isCompareResponse, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      ckpt_a: args.ckptA,
      ckpt_b: args.ckptB,
      sample_count: args.sampleCount,
      run_id: args.runId || null,
      decision_kind: args.decisionKind || null,
      seed: args.seed ?? 42,
    }),
  });
}

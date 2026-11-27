/** Typed HTTP client for the generalized Jace Studio FastAPI backend without type assertions. */

import type {
  ArenaLeaderboardResponse,
  ArenaMatchRow,
  CardMetadata,
  CompareResponse,
  ForkEdit,
  ForkResponse,
  GameFacets,
  GameOverview,
  GameStepResponse,
  GameSummary,
  GlobalCheckpointOption,
  PlayoutBatchResponse,
  PolicyMatrixResponse,
  PovMode,
  PromotePuzzleResult,
  PuzzleCardItem,
  PuzzlesResponse,
  RunDetail,
  RunSummary,
  RunsCatalogResponse,
  WatchlistItem,
} from "@/lib/types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

async function fetchJson<T>(
  url: string,
  guard: (data: unknown) => data is T,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`API ${response.status}: ${response.statusText}`);
  }
  const data: unknown = await response.json();
  if (!guard(data)) {
    throw new Error(`Unexpected response shape from ${url}`);
  }
  return data;
}

function isRunsEnvelope(data: unknown): data is RunsCatalogResponse {
  return isRecord(data) && Array.isArray(data.runs);
}

function isCheckpointsEnvelope(data: unknown): data is { checkpoints: GlobalCheckpointOption[] } {
  return isRecord(data) && Array.isArray(data.checkpoints);
}

function isRunDetail(data: unknown): data is RunDetail {
  return isRecord(data) && isRecord(data.summary) && Array.isArray(data.eval_points);
}

function isGamesEnvelope(data: unknown): data is { total: number; games: GameSummary[] } {
  return isRecord(data) && typeof data.total === "number" && Array.isArray(data.games);
}

function isGameFacets(data: unknown): data is GameFacets {
  return isRecord(data) && Array.isArray(data.runs) && Array.isArray(data.cards);
}

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

function isWatchlistEnvelope(data: unknown): data is { items: WatchlistItem[] } {
  return isRecord(data) && Array.isArray(data.items);
}

function isWatchlistItemEnvelope(data: unknown): data is { item: WatchlistItem } {
  return isRecord(data) && isRecord(data.item);
}

function isCardMetadata(data: unknown): data is CardMetadata {
  return isRecord(data) && typeof data.name === "string";
}

function isCardsCatalog(data: unknown): data is { cards: string[] } {
  return isRecord(data) && Array.isArray(data.cards);
}

export function cardImageUrl(name: string): string {
  return `/api/cards/${encodeURIComponent(name)}/image`;
}

export async function fetchRunsCatalog(): Promise<RunsCatalogResponse> {
  const env = await fetchJson("/api/runs", isRunsEnvelope);
  return {
    runs: env.runs,
    experiments: Array.isArray(env.experiments) ? env.experiments : [],
    eval_metric_keys: Array.isArray(env.eval_metric_keys) ? env.eval_metric_keys : [],
    train_metric_keys: Array.isArray(env.train_metric_keys) ? env.train_metric_keys : [],
  };
}

export async function fetchRuns(): Promise<RunSummary[]> {
  const catalog = await fetchRunsCatalog();
  return catalog.runs;
}

export async function fetchAllCheckpoints(): Promise<GlobalCheckpointOption[]> {
  const env = await fetchJson("/api/checkpoints", isCheckpointsEnvelope);
  return env.checkpoints;
}

export async function fetchRunDetail(runId: string): Promise<RunDetail> {
  const encoded = runId.split("/").map(encodeURIComponent).join("/");
  return fetchJson(`/api/runs/${encoded}`, isRunDetail);
}

export interface GamesSearchArgs {
  runId?: string;
  setCode?: string;
  card?: string;
  result?: string;
  colors?: string;
  minSurprise?: number;
  minSwing?: number;
  search?: string;
  sort?: string;
  limit?: number;
  offset?: number;
}

export async function fetchGames(
  args: GamesSearchArgs,
): Promise<{ total: number; games: GameSummary[] }> {
  const params = new URLSearchParams();
  const pairs: Array<[string, string | undefined]> = [
    ["run_id", args.runId],
    ["set_code", args.setCode],
    ["card", args.card],
    ["result", args.result],
    ["colors", args.colors],
    ["search", args.search],
    ["sort", args.sort],
  ];
  for (const [key, val] of pairs) {
    if (val) params.set(key, val);
  }
  if (args.minSurprise !== undefined && args.minSurprise > 0) {
    params.set("min_surprise", String(args.minSurprise));
  }
  if (args.minSwing !== undefined && args.minSwing > 0) {
    params.set("min_swing", String(args.minSwing));
  }
  params.set("limit", String(args.limit ?? 100));
  if (args.offset && args.offset > 0) params.set("offset", String(args.offset));
  return fetchJson(`/api/games?${params.toString()}`, isGamesEnvelope);
}

export async function fetchGameFacets(runId?: string): Promise<GameFacets> {
  const qs = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return fetchJson(`/api/games/facets${qs}`, isGameFacets);
}

export async function fetchCardCatalog(runId?: string): Promise<string[]> {
  const qs = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  const env = await fetchJson(`/api/cards-catalog${qs}`, isCardsCatalog);
  return env.cards;
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

export async function fetchWatchlist(): Promise<WatchlistItem[]> {
  const env = await fetchJson("/api/watchlist", isWatchlistEnvelope);
  return env.items;
}

export async function addWatchlistItem(payload: Omit<WatchlistItem, "id">): Promise<WatchlistItem> {
  const env = await fetchJson("/api/watchlist", isWatchlistItemEnvelope, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return env.item;
}

export async function updateWatchlistItem(
  itemId: string,
  patch: { label?: string; note?: string; tag?: string },
): Promise<WatchlistItem> {
  const env = await fetchJson(
    `/api/watchlist/${encodeURIComponent(itemId)}`,
    isWatchlistItemEnvelope,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    },
  );
  return env.item;
}

export async function removeWatchlistItem(itemId: string): Promise<void> {
  const res = await fetch(`/api/watchlist/${encodeURIComponent(itemId)}`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`Failed to remove watchlist item (${res.status})`);
  }
}

export async function fetchCardMetadata(name: string): Promise<CardMetadata> {
  return fetchJson(`/api/cards/${encodeURIComponent(name)}`, isCardMetadata);
}

function isArenaLeaderboardResponse(data: unknown): data is ArenaLeaderboardResponse {
  return (
    isRecord(data) &&
    typeof data.format === "string" &&
    Array.isArray(data.formats) &&
    Array.isArray(data.models)
  );
}

function isArenaMatchesEnvelope(data: unknown): data is { matches: ArenaMatchRow[] } {
  return isRecord(data) && Array.isArray(data.matches);
}

function isPuzzlesResponse(data: unknown): data is PuzzlesResponse {
  return (
    isRecord(data) &&
    typeof data.format === "string" &&
    Array.isArray(data.formats) &&
    Array.isArray(data.puzzles)
  );
}

function isPromoteAccepted(data: unknown): data is { admitted: true; puzzle: PuzzleCardItem } {
  return isRecord(data) && data.admitted === true && isRecord(data.puzzle);
}

function isPromoteRejected(
  data: unknown,
): data is { detail: string; margin: number; ci_low: number; ci_high: number } {
  return isRecord(data) && typeof data.detail === "string";
}

export async function fetchArenaLeaderboard(args?: {
  format?: string;
  anchor?: string;
}): Promise<ArenaLeaderboardResponse> {
  const params = new URLSearchParams();
  if (args?.format) params.set("format", args.format);
  if (args?.anchor) params.set("anchor", args.anchor);
  const qs = params.toString();
  return fetchJson(`/api/arena/leaderboard${qs ? `?${qs}` : ""}`, isArenaLeaderboardResponse);
}

export async function fetchArenaMatches(args: {
  format: string;
  model: string;
}): Promise<ArenaMatchRow[]> {
  const params = new URLSearchParams({ format: args.format, model: args.model });
  const env = await fetchJson(`/api/arena/matches?${params.toString()}`, isArenaMatchesEnvelope);
  return env.matches;
}

export async function fetchPuzzles(args?: {
  format?: string;
  modelA?: string;
  modelB?: string;
  phase?: string;
  source?: string;
}): Promise<PuzzlesResponse> {
  const params = new URLSearchParams();
  const pairs: Array<[string, string | undefined]> = [
    ["format", args?.format],
    ["model_a", args?.modelA],
    ["model_b", args?.modelB],
    ["phase", args?.phase],
    ["source", args?.source],
  ];
  for (const [key, val] of pairs) {
    if (val) params.set(key, val);
  }
  const qs = params.toString();
  return fetchJson(`/api/puzzles${qs ? `?${qs}` : ""}`, isPuzzlesResponse);
}

function parsePromoteResponse(response: Response, data: unknown): PromotePuzzleResult {
  if (response.status === 422 && isPromoteRejected(data)) {
    return {
      admitted: false,
      detail: data.detail,
      margin: typeof data.margin === "number" ? data.margin : 0,
      ci_low: typeof data.ci_low === "number" ? data.ci_low : 0,
      ci_high: typeof data.ci_high === "number" ? data.ci_high : 0,
    };
  }
  if (!response.ok) {
    const detail =
      isRecord(data) && typeof data.detail === "string" ? data.detail : response.statusText;
    throw new Error(`API ${response.status}: ${detail}`);
  }
  if (!isPromoteAccepted(data)) {
    throw new Error("Unexpected response shape from promote-puzzle");
  }
  return data;
}

export async function promoteWatchlistToPuzzle(args: {
  itemId: string;
  format: string;
  playouts?: number;
  minMargin?: number;
  modelRef?: string;
}): Promise<PromotePuzzleResult> {
  const response = await fetch(`/api/watchlist/${encodeURIComponent(args.itemId)}/promote-puzzle`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      format: args.format,
      playouts: args.playouts ?? 64,
      min_margin: args.minMargin ?? 0.15,
      model_ref: args.modelRef ?? null,
    }),
  });
  const data: unknown = await response.json();
  return parsePromoteResponse(response, data);
}

/** Typed HTTP client for the Jace Studio FastAPI backend without type assertions. */

import type {
  CardMetadata,
  ForkEdit,
  ForkResponse,
  GameFacets,
  GameOverview,
  GameStepResponse,
  GameSummary,
  PovMode,
  RunDetail,
  RunSummary,
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

function isRunsEnvelope(data: unknown): data is { runs: RunSummary[] } {
  return isRecord(data) && Array.isArray(data.runs);
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

export async function fetchRuns(): Promise<RunSummary[]> {
  const env = await fetchJson("/api/runs", isRunsEnvelope);
  return env.runs;
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

export async function removeWatchlistItem(itemId: string): Promise<void> {
  const res = await fetch(`/api/watchlist/${encodeURIComponent(itemId)}`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`Failed to remove watchlist item (${res.status})`);
  }
}

export async function fetchCardMetadata(name: string): Promise<CardMetadata> {
  return fetchJson(`/api/cards/${encodeURIComponent(name)}`, isCardMetadata);
}

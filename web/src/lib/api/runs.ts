/** Typed HTTP client endpoints for runs, checkpoints, games search, card catalog, and watchlist. */

import type {
  CardMetadata,
  GameFacets,
  GameSummary,
  GlobalCheckpointOption,
  RunDetail,
  RunSummary,
  RunsCatalogResponse,
  WatchlistItem,
} from "@/lib/types";
import { fetchJson, isRecord } from "./core";

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

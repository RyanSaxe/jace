/** Typed HTTP client endpoints for Arena leaderboard, matches, tactical puzzles, and watchlist promotion. */

import type {
  ArenaLeaderboardResponse,
  ArenaMatchRow,
  PromotePuzzleResult,
  PuzzleCardItem,
  PuzzlesResponse,
} from "@/lib/types";
import { fetchJson, isRecord } from "./core";

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

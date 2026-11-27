/** Shared presets, formatting, card-ordering, and keyboard helpers for Games Explorer. */

import type { GameSummary, GamesFilterState } from "@/lib/types";

export const PAGE_SIZE = 50;

export interface SearchPreset {
  id: string;
  label: string;
  patch: Partial<GamesFilterState>;
}

export const SEARCH_PRESETS: ReadonlyArray<SearchPreset> = [
  {
    id: "all",
    label: "All Playable",
    patch: {
      runId: "",
      setCode: "",
      card: "",
      colors: "",
      search: "",
      result: "",
      minSurprise: null,
      minSwing: null,
      sort: "surprise_desc",
    },
  },
  {
    id: "throws",
    label: "High Win% → Loss",
    patch: { result: "loss", minSurprise: 0.55, minSwing: null, sort: "surprise_desc" },
  },
  {
    id: "comebacks",
    label: "Low Win% → Win",
    patch: { result: "win", minSurprise: 0.55, minSwing: null, sort: "surprise_desc" },
  },
  {
    id: "swings",
    label: "Wildest Swings",
    patch: { result: "", minSurprise: null, minSwing: 0.25, sort: "swing_desc" },
  },
];

function isUnfilteredAll(filters: GamesFilterState): boolean {
  const anyText = [
    filters.runId,
    filters.card,
    filters.colors,
    filters.setCode,
    filters.search,
    filters.result,
  ].some(Boolean);
  return !anyText && filters.minSurprise === null && filters.minSwing === null;
}

export function activePresetId(filters: GamesFilterState): string {
  if (filters.result === "loss" && filters.minSurprise !== null) return "throws";
  if (filters.result === "win" && filters.minSurprise !== null) return "comebacks";
  if (filters.minSwing !== null && filters.sort === "swing_desc") return "swings";
  return isUnfilteredAll(filters) ? "all" : "";
}

export function resultTone(result: string): "win" | "loss" | "neutral" {
  if (result === "win") return "win";
  if (result === "loss") return "loss";
  return "neutral";
}

export function formatCalibrationSummary(g: GameSummary): {
  text: string;
  tone: "win" | "loss" | "neutral";
} {
  const maxW = Math.round((g.max_pred_win ?? 0.5) * 100);
  const minW = Math.round((g.min_pred_win ?? 0.5) * 100);
  if (g.result === "loss") {
    return { text: `Peak ${maxW}% → Loss`, tone: maxW >= 65 ? "loss" : "neutral" };
  }
  if (g.result === "win") {
    return { text: `Low ${minW}% → Win`, tone: minW <= 35 ? "win" : "neutral" };
  }
  return { text: `${minW}%–${maxW}%`, tone: "neutral" };
}

export function orderedPreviewCards(game: GameSummary, highlightQuery: string): string[] {
  const q = highlightQuery.trim().toLowerCase();
  if (!q) return game.key_cards.slice(0, 8);
  const pool = game.cards_seen.length > 0 ? game.cards_seen : game.key_cards;
  const matching = pool.filter((c) => c.toLowerCase().includes(q));
  const seen = new Set(matching);
  const rest = game.key_cards.filter((c) => !seen.has(c));
  return [...matching, ...rest].slice(0, 8);
}

export function nonEmpty(value: string): string | undefined {
  return value === "" ? undefined : value;
}

export function gameIdentity(game: GameSummary | null): [string, number, number] {
  if (!game) return ["", -1, -1];
  return [game.run_id, game.iteration, game.game_idx];
}

export function navDeltaForKey(key: string): number {
  if (key === "j" || key === "ArrowDown") return 1;
  if (key === "k" || key === "ArrowUp") return -1;
  return 0;
}

export function isClickableControl(target: EventTarget | null): boolean {
  return target instanceof HTMLButtonElement || target instanceof HTMLAnchorElement;
}

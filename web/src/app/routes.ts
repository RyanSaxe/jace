/** URL hash parsing, serialization, and navigation constants for the Jace Studio shell. */

import type {
  BenchmarksRouteState,
  CompareTarget,
  GameTarget,
  GamesFilterState,
  NavTab,
} from "@/lib/types";

export interface NavItemSpec {
  id: NavTab;
  label: string;
  shortcut: string;
}

export const NAV_ITEMS: ReadonlyArray<NavItemSpec> = [
  { id: "runs", label: "Runs", shortcut: "1" },
  { id: "games", label: "Games", shortcut: "2" },
  { id: "benchmarks", label: "Benchmarks", shortcut: "3" },
  { id: "compare", label: "Compare", shortcut: "4" },
  { id: "tabletop", label: "Tabletop", shortcut: "5" },
  { id: "watchlist", label: "Watchlist", shortcut: "6" },
];

export const SHORTCUT_TO_TAB: Record<string, NavTab> = {
  "1": "runs",
  "2": "games",
  "3": "benchmarks",
  "4": "compare",
  "5": "tabletop",
  "6": "watchlist",
};

export const INITIAL_BENCHMARKS_STATE: BenchmarksRouteState = {
  subView: "arena",
  format: "",
  anchor: "",
  model: "",
  modelA: "",
  modelB: "",
};

export const INITIAL_GAMES_FILTERS: GamesFilterState = {
  runId: "",
  setCode: "",
  result: "",
  colors: "",
  card: "",
  minSurprise: null,
  minSwing: null,
  search: "",
  sort: "surprise_desc",
  page: 0,
};

export interface ParsedUrlState {
  tab: NavTab;
  deepDiveRunId: string | null;
  benchmarksState: BenchmarksRouteState;
  compareTarget: CompareTarget | null;
  gameTarget: GameTarget | null;
}

const DEFAULT_URL_STATE: ParsedUrlState = {
  tab: "runs",
  deepDiveRunId: null,
  benchmarksState: INITIAL_BENCHMARKS_STATE,
  compareTarget: null,
  gameTarget: null,
};

function parseOptionalInt(raw: string | null, fallback: number): number {
  const parsed = Number.parseInt(raw ?? "", 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseCompareParams(params: URLSearchParams): CompareTarget | null {
  const ckptA = params.get("a") ?? "";
  const ckptB = params.get("b") ?? "";
  if (!ckptA || !ckptB) return null;
  return { ckptA, ckptB, runId: params.get("run") ?? undefined };
}

function parseTabletopParams(params: URLSearchParams): GameTarget | null {
  const runId = params.get("run") ?? "";
  const iteration = Number.parseInt(params.get("iter") ?? "", 10);
  const gameIdx = Number.parseInt(params.get("game") ?? "", 10);
  if (!runId || !Number.isFinite(iteration) || !Number.isFinite(gameIdx)) return null;
  return {
    run_id: runId,
    iteration,
    game_idx: gameIdx,
    step: parseOptionalInt(params.get("step"), 0),
    seed: parseOptionalInt(params.get("seed"), 0),
    ckpt_a: params.get("a") ?? undefined,
    ckpt_b: params.get("b") ?? undefined,
  };
}

function parseBenchmarksHash(pathPart: string, params: URLSearchParams): BenchmarksRouteState {
  const subView = pathPart === "benchmarks/puzzles" ? "puzzles" : "arena";
  return {
    subView,
    format: params.get("format") ?? "",
    anchor: params.get("anchor") ?? "",
    model: params.get("model") ?? "",
    modelA: params.get("a") ?? "",
    modelB: params.get("b") ?? "",
  };
}

function buildBenchmarksHash(state: BenchmarksRouteState): string {
  const params = new URLSearchParams();
  if (state.format) params.set("format", state.format);
  if (state.subView === "arena") {
    if (state.anchor) params.set("anchor", state.anchor);
    if (state.model) params.set("model", state.model);
  } else {
    if (state.modelA) params.set("a", state.modelA);
    if (state.modelB) params.set("b", state.modelB);
  }
  const qs = params.toString();
  return `#benchmarks/${state.subView}${qs ? `?${qs}` : ""}`;
}

function parseRoutePath(pathPart: string, params: URLSearchParams): ParsedUrlState {
  if (pathPart.startsWith("runs/")) {
    const runId = decodeURIComponent(pathPart.slice("runs/".length));
    return { ...DEFAULT_URL_STATE, tab: "runs", deepDiveRunId: runId || null };
  }
  if (pathPart === "benchmarks" || pathPart.startsWith("benchmarks/")) {
    return {
      ...DEFAULT_URL_STATE,
      tab: "benchmarks",
      benchmarksState: parseBenchmarksHash(pathPart, params),
    };
  }
  if (pathPart === "games" || pathPart === "watchlist") {
    return { ...DEFAULT_URL_STATE, tab: pathPart };
  }
  if (pathPart === "compare") {
    return { ...DEFAULT_URL_STATE, tab: "compare", compareTarget: parseCompareParams(params) };
  }
  if (pathPart === "tabletop") {
    return { ...DEFAULT_URL_STATE, tab: "tabletop", gameTarget: parseTabletopParams(params) };
  }
  return DEFAULT_URL_STATE;
}

export function parseHashState(): ParsedUrlState {
  if (typeof window === "undefined") return DEFAULT_URL_STATE;
  const raw = window.location.hash.replace(/^#\/?/, "");
  const [pathPart = "runs", queryPart = ""] = raw.split("?");
  return parseRoutePath(pathPart, new URLSearchParams(queryPart));
}

function buildTabletopHash(target: GameTarget): string {
  const p = new URLSearchParams({
    run: target.run_id,
    iter: String(target.iteration),
    game: String(target.game_idx),
    step: String(target.step),
    seed: String(target.seed),
  });
  if (target.ckpt_a) p.set("a", target.ckpt_a);
  if (target.ckpt_b) p.set("b", target.ckpt_b);
  return `#tabletop?${p.toString()}`;
}

function buildCompareHash(target: CompareTarget): string {
  const p = new URLSearchParams({ a: target.ckptA, b: target.ckptB });
  if (target.runId) p.set("run", target.runId);
  return `#compare?${p.toString()}`;
}

export function buildHashUrl(state: ParsedUrlState): string {
  if (state.tab === "runs") {
    return state.deepDiveRunId ? `#runs/${encodeURIComponent(state.deepDiveRunId)}` : "#runs";
  }
  if (state.tab === "benchmarks") return buildBenchmarksHash(state.benchmarksState);
  if (state.tab === "compare" && state.compareTarget) return buildCompareHash(state.compareTarget);
  if (state.tab === "tabletop" && state.gameTarget) return buildTabletopHash(state.gameTarget);
  return `#${state.tab}`;
}

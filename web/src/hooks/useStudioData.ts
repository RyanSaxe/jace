/** Custom React hooks owning all external-system synchronization (API fetches, theme DOM, keyboard). */

import { useEffect, useState } from "react";
import {
  type GamesSearchArgs,
  fetchCardCatalog,
  fetchCardMetadata,
  fetchGameFacets,
  fetchGameOverview,
  fetchGameStep,
  fetchGames,
  fetchRunDetail,
  fetchRuns,
  fetchWatchlist,
  postFork,
} from "@/lib/api";
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
  ThemeMode,
  WatchlistItem,
} from "@/lib/types";

const THEME_STORAGE_KEY = "jace-studio-theme";

const EMPTY_FACETS: GameFacets = {
  runs: [],
  colors: [],
  sets: [],
  results: [],
  cards: [],
  total_playable: 0,
  total_unplayable: 0,
};

export function loadSavedTheme(): ThemeMode {
  try {
    return window.localStorage.getItem(THEME_STORAGE_KEY) === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

export function useThemeSync(theme: ThemeMode): void {
  useEffect(() => {
    document.documentElement.dataset.theme = theme === "dark" ? "dark" : "";
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Ignore storage errors in private/restricted browsers.
    }
  }, [theme]);
}

async function fetchRunDetailBatch(runIds: string[]): Promise<Record<string, RunDetail>> {
  const pairs = await Promise.all(
    runIds.map(async (id) => {
      try {
        const detail = await fetchRunDetail(id);
        return [id, detail] as const;
      } catch {
        return null;
      }
    }),
  );
  const map: Record<string, RunDetail> = {};
  for (const pair of pairs) {
    if (pair !== null) map[pair[0]] = pair[1];
  }
  return map;
}

export function useRunsData(): {
  runs: RunSummary[];
  details: Record<string, RunDetail>;
  loading: boolean;
  error: string | null;
  reload: () => void;
} {
  const [tick, setTick] = useState(0);
  const [state, setState] = useState<{
    loadedTick: number;
    runs: RunSummary[];
    details: Record<string, RunDetail>;
    error: string | null;
  }>({ loadedTick: -1, runs: [], details: {}, error: null });

  useEffect(() => {
    let active = true;
    const currentTick = tick;
    void fetchRuns()
      .then(async (list) => {
        const preferred = list
          .filter((r) => r.latest_live_win !== null || r.latest_ema_win !== null)
          .slice(0, 8)
          .map((r) => r.run_id);
        const map = await fetchRunDetailBatch(preferred);
        if (!active) return;
        setState({ loadedTick: currentTick, runs: list, details: map, error: null });
      })
      .catch((err: unknown) => {
        if (!active) return;
        const message = err instanceof Error ? err.message : "Failed to load runs";
        setState((prev) => ({ ...prev, loadedTick: currentTick, error: message }));
      });
    return (): void => {
      active = false;
    };
  }, [tick]);

  return {
    runs: state.runs,
    details: state.details,
    loading: state.loadedTick !== tick,
    error: state.error,
    reload: (): void => setTick((n) => n + 1),
  };
}

export function useExtraRunDetails(
  runIds: string[],
  existing: Record<string, RunDetail>,
): Record<string, RunDetail> {
  const [extra, setExtra] = useState<Record<string, RunDetail>>({});
  const missing = runIds.filter((id) => !existing[id] && !extra[id]);
  const missingKey = missing.join("|");

  useEffect(() => {
    if (!missingKey) return undefined;
    let active = true;
    const ids = missingKey.split("|");
    void fetchRunDetailBatch(ids).then((fetched) => {
      if (active && Object.keys(fetched).length > 0) {
        setExtra((prev) => ({ ...prev, ...fetched }));
      }
    });
    return (): void => {
      active = false;
    };
  }, [missingKey]);

  return { ...extra, ...existing };
}

export function useRunDetail(runId: string | null): RunDetail | null {
  const [loaded, setLoaded] = useState<{ id: string; detail: RunDetail | null }>({
    id: "",
    detail: null,
  });

  useEffect(() => {
    if (!runId) return undefined;
    let active = true;
    void fetchRunDetail(runId)
      .then((res) => {
        if (active) setLoaded({ id: runId, detail: res });
      })
      .catch(() => {
        if (active) setLoaded({ id: runId, detail: null });
      });
    return (): void => {
      active = false;
    };
  }, [runId]);

  if (!runId || loaded.id !== runId) return null;
  return loaded.detail;
}

export function useCardCatalog(runId?: string): string[] {
  const [cards, setCards] = useState<string[]>([]);
  useEffect(() => {
    let active = true;
    void fetchCardCatalog(runId)
      .then((list) => {
        if (active) setCards(list);
      })
      .catch(() => undefined);
    return (): void => {
      active = false;
    };
  }, [runId]);
  return cards;
}

export function useDebouncedValue(value: string, delayMs: number): string {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), delayMs);
    return (): void => window.clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}

export function useGamesSearch(args: GamesSearchArgs): {
  games: GameSummary[];
  total: number;
  facets: GameFacets;
  loading: boolean;
  error: string | null;
} {
  const {
    runId,
    setCode,
    card,
    result,
    colors,
    minSurprise,
    minSwing,
    search,
    sort,
    limit,
    offset,
  } = args;
  const debouncedSearch = useDebouncedValue(search ?? "", 180);
  const effectiveSearch = debouncedSearch === "" ? undefined : debouncedSearch;
  const queryKey = JSON.stringify({
    runId,
    setCode,
    card,
    result,
    colors,
    minSurprise,
    minSwing,
    effectiveSearch,
    sort,
    limit,
    offset,
  });
  const [state, setState] = useState<{
    key: string;
    games: GameSummary[];
    total: number;
    error: string | null;
  }>({ key: "", games: [], total: 0, error: null });
  const [facets, setFacets] = useState<GameFacets>(EMPTY_FACETS);

  useEffect(() => {
    let active = true;
    void fetchGames({
      runId,
      setCode,
      card,
      result,
      colors,
      minSurprise,
      minSwing,
      search: effectiveSearch,
      sort,
      limit,
      offset,
    })
      .then((res) => {
        if (!active) return;
        setState({ key: queryKey, games: res.games, total: res.total, error: null });
      })
      .catch((err: unknown) => {
        if (!active) return;
        const message = err instanceof Error ? err.message : "Failed to search games";
        setState((prev) => ({ ...prev, key: queryKey, error: message }));
      });
    return (): void => {
      active = false;
    };
  }, [
    queryKey,
    runId,
    setCode,
    card,
    result,
    colors,
    minSurprise,
    minSwing,
    effectiveSearch,
    sort,
    limit,
    offset,
  ]);

  useEffect(() => {
    let active = true;
    void fetchGameFacets(runId)
      .then((nextFacets) => {
        if (active) setFacets(nextFacets);
      })
      .catch(() => undefined);
    return (): void => {
      active = false;
    };
  }, [runId]);

  return {
    games: state.games,
    total: state.total,
    facets,
    loading: state.key !== queryKey,
    error: state.error,
  };
}

export function useGameOverview(
  runId: string,
  iteration: number,
  gameIdx: number,
): { overview: GameOverview | null; loading: boolean; error: string | null } {
  const valid = runId !== "" && iteration >= 0 && gameIdx >= 0;
  const reqKey = valid ? `${runId}:${iteration}:${gameIdx}` : "";
  const [state, setState] = useState<{
    key: string;
    overview: GameOverview | null;
    error: string | null;
  }>({ key: "", overview: null, error: null });

  useEffect(() => {
    if (!valid || !reqKey) return undefined;
    let active = true;
    void fetchGameOverview({ runId, iteration, gameIdx })
      .then((res) => {
        if (active) setState({ key: reqKey, overview: res, error: null });
      })
      .catch((err: unknown) => {
        if (!active) return;
        const message = err instanceof Error ? err.message : "Failed to load game overview";
        setState({ key: reqKey, overview: null, error: message });
      });
    return (): void => {
      active = false;
    };
  }, [valid, reqKey, runId, iteration, gameIdx]);

  if (!reqKey) return { overview: null, loading: false, error: null };
  return {
    overview: state.key === reqKey ? state.overview : null,
    loading: state.key !== reqKey,
    error: state.key === reqKey ? state.error : null,
  };
}

export interface StepFetchTarget {
  runId: string;
  iteration: number;
  gameIdx: number;
  step: number;
  pov: PovMode;
  ckptA?: string;
  ckptB?: string;
}

export function useGameStepData(args: StepFetchTarget): {
  data: GameStepResponse | null;
  loading: boolean;
  error: string | null;
} {
  const { runId, iteration, gameIdx, step, pov, ckptA, ckptB } = args;
  const reqKey = `${runId}:${iteration}:${gameIdx}:${step}:${pov}:${ckptA ?? ""}:${ckptB ?? ""}`;
  const [state, setState] = useState<{
    key: string;
    data: GameStepResponse | null;
    error: string | null;
  }>({ key: "", data: null, error: null });

  useEffect(() => {
    if (!runId || iteration < 0 || gameIdx < 0) return undefined;
    let active = true;
    void fetchGameStep({ runId, iteration, gameIdx, step, pov, ckptA, ckptB })
      .then((res) => {
        if (active) setState({ key: reqKey, data: res, error: null });
      })
      .catch((err: unknown) => {
        if (!active) return;
        const message = err instanceof Error ? err.message : "Failed to load step";
        setState({ key: reqKey, data: null, error: message });
      });
    return (): void => {
      active = false;
    };
  }, [reqKey, runId, iteration, gameIdx, step, pov, ckptA, ckptB]);

  return {
    data: state.data,
    loading: state.key !== reqKey,
    error: state.error,
  };
}

export function useForkEvaluation(args: {
  target: StepFetchTarget;
  edits: ForkEdit[];
  branchActions: number[];
  enabled: boolean;
}): { result: ForkResponse | null; busy: boolean; error: string | null } {
  const { target, edits, branchActions, enabled } = args;
  const { runId, iteration, gameIdx, step, pov, ckptA, ckptB } = target;
  const scopeKey = `${runId}:${iteration}:${gameIdx}:${step}:${pov}`;
  const editsJson = JSON.stringify(edits);
  const branchJson = JSON.stringify(branchActions);
  const hasMutations = edits.length > 0 || branchActions.length > 0;
  const activeKey =
    enabled && hasMutations
      ? `${scopeKey}:${ckptA ?? ""}:${ckptB ?? ""}:${editsJson}:${branchJson}`
      : "";
  const [state, setState] = useState<{
    key: string;
    scopeKey: string;
    result: ForkResponse | null;
    error: string | null;
  }>({ key: "", scopeKey: "", result: null, error: null });

  useEffect(() => {
    if (!activeKey) return undefined;
    let active = true;
    void postFork({
      runId,
      iteration,
      gameIdx,
      step,
      pov,
      ckptA,
      ckptB,
      edits,
      branchActions,
    })
      .then((res) => {
        if (active) setState({ key: activeKey, scopeKey, result: res, error: null });
      })
      .catch((err: unknown) => {
        if (!active) return;
        const message = err instanceof Error ? err.message : "Fork evaluation failed";
        setState((prev) => ({
          key: activeKey,
          scopeKey,
          result: prev.scopeKey === scopeKey ? prev.result : null,
          error: message,
        }));
      });
    return (): void => {
      active = false;
    };
  }, [
    activeKey,
    scopeKey,
    runId,
    iteration,
    gameIdx,
    step,
    pov,
    ckptA,
    ckptB,
    edits,
    branchActions,
  ]);

  if (!activeKey) return { result: null, busy: false, error: null };
  return {
    result: state.scopeKey === scopeKey ? state.result : null,
    busy: state.key !== activeKey,
    error: state.key === activeKey ? state.error : null,
  };
}

export function useWatchlistData(): {
  items: WatchlistItem[];
  setItems: React.Dispatch<React.SetStateAction<WatchlistItem[]>>;
  loading: boolean;
  error: string | null;
} {
  const [items, setItems] = useState<WatchlistItem[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void fetchWatchlist()
      .then((list) => {
        if (!active) return;
        setItems(list);
        setLoaded(true);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Failed to load watchlist");
        setLoaded(true);
      });
    return (): void => {
      active = false;
    };
  }, []);

  return { items, setItems, loading: !loaded, error };
}

export function useCardMeta(name: string | null): CardMetadata | null {
  const [loaded, setLoaded] = useState<{ name: string; meta: CardMetadata | null }>({
    name: "",
    meta: null,
  });

  useEffect(() => {
    if (!name) return undefined;
    let active = true;
    void fetchCardMetadata(name)
      .then((res) => {
        if (active) setLoaded({ name, meta: res });
      })
      .catch(() => {
        if (active) setLoaded({ name, meta: null });
      });
    return (): void => {
      active = false;
    };
  }, [name]);

  if (!name || loaded.name !== name) return null;
  return loaded.meta;
}

export function useKeyboardShortcut(handler: (event: KeyboardEvent) => void): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target;
      const isField =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        (target instanceof HTMLElement && target.isContentEditable);
      if (isField) return;
      handler(event);
    };
    window.addEventListener("keydown", onKeyDown);
    return (): void => window.removeEventListener("keydown", onKeyDown);
  }, [handler]);
}

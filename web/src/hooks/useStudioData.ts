/** Custom React hooks owning Studio-level synchronization (runs, games search, watchlist, URL hash, theme, abort cleanup, keyboard) and re-exporting Tabletop hooks. */

import { useEffect, useState } from "react";
import {
  type GamesSearchArgs,
  fetchAllCheckpoints,
  fetchGameFacets,
  fetchGames,
  fetchRunDetail,
  fetchRunsCatalog,
  fetchWatchlist,
} from "@/lib/api";
import type {
  GameFacets,
  GameSummary,
  GlobalCheckpointOption,
  RunDetail,
  RunSummary,
  ThemeMode,
  WatchlistItem,
} from "@/lib/types";

export {
  type StepFetchTarget,
  useAbortOnUnmount,
  useCardCatalog,
  useCardMeta,
  useForkEvaluation,
  useGameOverview,
  useGameStepData,
  useKeyboardShortcut,
  useResetOnKey,
} from "./useTabletopData";

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

export function useUrlHashSync(nextHash: string, onHashChange?: () => void): void {
  useEffect(() => {
    if (typeof window !== "undefined" && window.location.hash !== nextHash) {
      window.history.replaceState(null, "", nextHash);
    }
  }, [nextHash]);

  useEffect(() => {
    if (typeof window === "undefined" || !onHashChange) return undefined;
    window.addEventListener("hashchange", onHashChange);
    return (): void => window.removeEventListener("hashchange", onHashChange);
  }, [onHashChange]);
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
  experiments: string[];
  evalMetricKeys: string[];
  trainMetricKeys: string[];
  checkpoints: GlobalCheckpointOption[];
  details: Record<string, RunDetail>;
  loading: boolean;
  error: string | null;
  reload: () => void;
} {
  const [tick, setTick] = useState(0);
  const [state, setState] = useState<{
    loadedTick: number;
    runs: RunSummary[];
    experiments: string[];
    evalMetricKeys: string[];
    trainMetricKeys: string[];
    checkpoints: GlobalCheckpointOption[];
    details: Record<string, RunDetail>;
    error: string | null;
  }>({
    loadedTick: -1,
    runs: [],
    experiments: [],
    evalMetricKeys: [],
    trainMetricKeys: [],
    checkpoints: [],
    details: {},
    error: null,
  });

  useEffect(() => {
    let active = true;
    const currentTick = tick;
    void Promise.all([fetchRunsCatalog(), fetchAllCheckpoints().catch(() => [])])
      .then(async ([catalog, ckpts]) => {
        const list = catalog.runs;
        const preferred = list.slice(0, 8).map((r) => r.run_id);
        const map = await fetchRunDetailBatch(preferred);
        if (!active) return;
        setState({
          loadedTick: currentTick,
          runs: list,
          experiments: catalog.experiments,
          evalMetricKeys: catalog.eval_metric_keys,
          trainMetricKeys: catalog.train_metric_keys,
          checkpoints: ckpts,
          details: map,
          error: null,
        });
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
    experiments: state.experiments,
    evalMetricKeys: state.evalMetricKeys,
    trainMetricKeys: state.trainMetricKeys,
    checkpoints: state.checkpoints,
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

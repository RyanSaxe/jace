/** Custom React hooks owning all external-system synchronization (API fetches, theme DOM, keyboard). */

import { useEffect, useState } from "react";
import {
  type GamesSearchArgs,
  fetchCardCatalog,
  fetchCardMetadata,
  fetchGameOverview,
  fetchGameStep,
  fetchGames,
  fetchRunDetail,
  fetchRuns,
  fetchWatchlist,
} from "@/lib/api";
import type {
  CardMetadata,
  GameOverview,
  GameStepResponse,
  GameSummary,
  PovMode,
  RunDetail,
  RunSummary,
  ThemeMode,
  WatchlistItem,
} from "@/lib/types";

export function useThemeSync(theme: ThemeMode): void {
  useEffect(() => {
    document.documentElement.dataset.theme = theme === "dark" ? "dark" : "";
  }, [theme]);
}

async function fetchTopRunDetails(list: RunSummary[]): Promise<Record<string, RunDetail>> {
  const pairs = await Promise.all(
    list.slice(0, 6).map(async (item) => {
      try {
        const detail = await fetchRunDetail(item.run_id);
        return [item.run_id, detail] as const;
      } catch {
        return null;
      }
    }),
  );
  const map: Record<string, RunDetail> = {};
  for (const pair of pairs) {
    if (pair !== null) {
      map[pair[0]] = pair[1];
    }
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
        const map = await fetchTopRunDetails(list);
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

export function useGamesSearch(args: GamesSearchArgs): {
  games: GameSummary[];
  total: number;
  cards: string[];
  loading: boolean;
  error: string | null;
} {
  const { runId, setCode, card, result, search, sort, limit } = args;
  const queryKey = JSON.stringify({ runId, setCode, card, result, search, sort, limit });
  const [state, setState] = useState<{
    key: string;
    games: GameSummary[];
    total: number;
    error: string | null;
  }>({ key: "", games: [], total: 0, error: null });
  const [cards, setCards] = useState<string[]>([]);

  useEffect(() => {
    let active = true;
    void fetchGames({ runId, setCode, card, result, search, sort, limit })
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
  }, [queryKey, runId, setCode, card, result, search, sort, limit]);

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

  return {
    games: state.games,
    total: state.total,
    cards,
    loading: state.key !== queryKey,
    error: state.error,
  };
}

function overviewKey(target: { runId: string; iteration: number; gameIdx: number } | null): string {
  if (!target || target.iteration < 0 || target.gameIdx < 0) return "";
  return `${target.runId}:${target.iteration}:${target.gameIdx}`;
}

export function useGameOverview(
  target: { runId: string; iteration: number; gameIdx: number } | null,
): { overview: GameOverview | null; loading: boolean } {
  const reqKey = overviewKey(target);
  const [state, setState] = useState<{ key: string; overview: GameOverview | null }>({
    key: "",
    overview: null,
  });

  useEffect(() => {
    if (!target || !reqKey) return undefined;
    let active = true;
    void fetchGameOverview(target)
      .then((res) => {
        if (active) setState({ key: reqKey, overview: res });
      })
      .catch(() => {
        if (active) setState({ key: reqKey, overview: null });
      });
    return (): void => {
      active = false;
    };
  }, [reqKey, target]);

  if (!reqKey) return { overview: null, loading: false };
  return { overview: state.key === reqKey ? state.overview : null, loading: state.key !== reqKey };
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

export function useGameStepData(args: StepFetchTarget | null): {
  data: GameStepResponse | null;
  loading: boolean;
  error: string | null;
} {
  const reqKey = args ? JSON.stringify(args) : "";
  const [state, setState] = useState<{
    key: string;
    data: GameStepResponse | null;
    error: string | null;
  }>({ key: "", data: null, error: null });

  useEffect(() => {
    if (!args || !reqKey) return undefined;
    let active = true;
    void fetchGameStep(args)
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
  }, [reqKey, args]);

  return {
    data: state.data,
    loading: reqKey !== "" && state.key !== reqKey,
    error: state.error,
  };
}

export function useWatchlistData(): {
  items: WatchlistItem[];
  setItems: React.Dispatch<React.SetStateAction<WatchlistItem[]>>;
  loading: boolean;
} {
  const [items, setItems] = useState<WatchlistItem[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let active = true;
    void fetchWatchlist()
      .then((list) => {
        if (!active) return;
        setItems(list);
        setLoaded(true);
      })
      .catch(() => {
        if (active) setLoaded(true);
      });
    return (): void => {
      active = false;
    };
  }, []);

  return { items, setItems, loading: !loaded };
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
      const target = event.target;
      const isField =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement;
      if (isField) return;
      handler(event);
    };
    window.addEventListener("keydown", onKeyDown);
    return (): void => window.removeEventListener("keydown", onKeyDown);
  }, [handler]);
}

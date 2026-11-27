/** React hooks owning Arena leaderboard, match ledger, and verified puzzle fetches for Benchmarks. */

import { useEffect, useState } from "react";
import { fetchArenaLeaderboard, fetchArenaMatches, fetchPuzzles } from "@/lib/api";
import type { ArenaLeaderboardResponse, ArenaMatchRow, PuzzlesResponse } from "@/lib/types";

export function useArenaLeaderboard(args: {
  format: string;
  anchor: string;
  refreshTick: number;
  enabled: boolean;
}): {
  data: ArenaLeaderboardResponse | null;
  loading: boolean;
  error: string | null;
} {
  const { format, anchor, refreshTick, enabled } = args;
  const reqKey = enabled ? `${format}::${anchor}::${refreshTick}` : "";
  const [state, setState] = useState<{
    key: string;
    data: ArenaLeaderboardResponse | null;
    error: string | null;
  }>({ key: "", data: null, error: null });

  useEffect(() => {
    if (!enabled || !reqKey) return undefined;
    let active = true;
    void fetchArenaLeaderboard({
      format: format || undefined,
      anchor: anchor || undefined,
    })
      .then((res) => {
        if (active) setState({ key: reqKey, data: res, error: null });
      })
      .catch((err: unknown) => {
        if (!active) return;
        const msg = err instanceof Error ? err.message : "Failed to load Arena leaderboard";
        setState((prev) => ({ key: reqKey, data: prev.data, error: msg }));
      });
    return (): void => {
      active = false;
    };
  }, [enabled, reqKey, format, anchor]);

  return {
    data: state.data,
    loading: enabled && state.key !== reqKey,
    error: state.error,
  };
}

export function useArenaMatches(args: {
  format: string;
  model: string;
  refreshTick: number;
  enabled: boolean;
}): {
  matches: ArenaMatchRow[];
  loading: boolean;
  error: string | null;
} {
  const { format, model, refreshTick, enabled } = args;
  const valid = enabled && format !== "" && model !== "";
  const reqKey = valid ? `${format}::${model}::${refreshTick}` : "";
  const [state, setState] = useState<{
    key: string;
    matches: ArenaMatchRow[];
    error: string | null;
  }>({ key: "", matches: [], error: null });

  useEffect(() => {
    if (!valid || !reqKey) return undefined;
    let active = true;
    void fetchArenaMatches({ format, model })
      .then((rows) => {
        if (active) setState({ key: reqKey, matches: rows, error: null });
      })
      .catch((err: unknown) => {
        if (!active) return;
        const msg = err instanceof Error ? err.message : "Failed to load match ledger";
        setState({ key: reqKey, matches: [], error: msg });
      });
    return (): void => {
      active = false;
    };
  }, [valid, reqKey, format, model]);

  if (!valid) return { matches: [], loading: false, error: null };
  return {
    matches: state.key === reqKey ? state.matches : [],
    loading: state.key !== reqKey,
    error: state.key === reqKey ? state.error : null,
  };
}

export function usePuzzlesData(args: {
  format: string;
  modelA: string;
  modelB: string;
  phase: string;
  source: string;
  refreshTick: number;
  enabled: boolean;
}): {
  data: PuzzlesResponse | null;
  loading: boolean;
  error: string | null;
} {
  const { format, modelA, modelB, phase, source, refreshTick, enabled } = args;
  const reqKey = enabled
    ? `${format}::${modelA}::${modelB}::${phase}::${source}::${refreshTick}`
    : "";
  const [state, setState] = useState<{
    key: string;
    data: PuzzlesResponse | null;
    error: string | null;
  }>({ key: "", data: null, error: null });

  useEffect(() => {
    if (!enabled || !reqKey) return undefined;
    let active = true;
    void fetchPuzzles({
      format: format || undefined,
      modelA: modelA || undefined,
      modelB: modelB || undefined,
      phase: phase || undefined,
      source: source || undefined,
    })
      .then((res) => {
        if (active) setState({ key: reqKey, data: res, error: null });
      })
      .catch((err: unknown) => {
        if (!active) return;
        const msg = err instanceof Error ? err.message : "Failed to load puzzles";
        setState((prev) => ({ key: reqKey, data: prev.data, error: msg }));
      });
    return (): void => {
      active = false;
    };
  }, [enabled, reqKey, format, modelA, modelB, phase, source]);

  return {
    data: state.data,
    loading: enabled && state.key !== reqKey,
    error: state.error,
  };
}

export function useBenchmarkFormats(): { formats: string[]; defaultFormat: string } {
  const [state, setState] = useState<{ formats: string[]; defaultFormat: string }>({
    formats: [],
    defaultFormat: "",
  });

  useEffect(() => {
    let active = true;
    void fetchArenaLeaderboard()
      .then((res) => {
        if (active) setState({ formats: res.formats, defaultFormat: res.format });
      })
      .catch(() => undefined);
    return (): void => {
      active = false;
    };
  }, []);

  return state;
}

/** Custom React hooks owning Tabletop replay, game overview, fork evaluation, card catalog, and card metadata fetches. */

import { type RefObject, useEffect, useRef, useState } from "react";
import {
  fetchCardCatalog,
  fetchCardMetadata,
  fetchGameOverview,
  fetchGameStep,
  postFork,
} from "@/lib/api";
import type {
  CardMetadata,
  ForkEdit,
  ForkResponse,
  GameOverview,
  GameStepResponse,
  PovMode,
} from "@/lib/types";

export interface StepFetchTarget {
  runId: string;
  iteration: number;
  gameIdx: number;
  step: number;
  pov: PovMode;
  ckptA?: string;
  ckptB?: string;
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

export function useAbortOnUnmount(abortRef: RefObject<AbortController | null>): void {
  useEffect(
    () => (): void => {
      abortRef.current?.abort();
      abortRef.current = null;
    },
    [abortRef],
  );
}

export function useResetOnKey(key: string, onReset: () => void): void {
  const prevKeyRef = useRef(key);
  const onResetRef = useRef(onReset);
  useEffect(() => {
    onResetRef.current = onReset;
  });
  useEffect(() => {
    if (prevKeyRef.current !== key) {
      prevKeyRef.current = key;
      onResetRef.current();
    }
  }, [key]);
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

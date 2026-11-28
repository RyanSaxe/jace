/** State hook, override helpers, and API dispatchers for Tabletop opponent belief and local search. */

import { useState } from "react";
import { useResetOnKey } from "@/hooks/useStudioData";
import { postBeliefInfer, postBeliefProject, postSearchEvaluate } from "@/lib/api";
import type {
  BeliefInferResponse,
  BeliefOverridePayload,
  BeliefProjectResponse,
  DockAnalysisMode,
  GameTarget,
  SearchEvaluateResponse,
  SearchWorldBudget,
} from "@/lib/types";

export const EMPTY_BELIEF_OVERRIDE: BeliefOverridePayload = {
  pinned_in_hand: {},
  excluded_cards: [],
  copy_multipliers: {},
};

export interface BeliefSearchState {
  mode: DockAnalysisMode;
  worlds: SearchWorldBudget;
  priorOnly: boolean;
  editorOpen: boolean;
  override: BeliefOverridePayload;
  belief: BeliefInferResponse | null;
  projection: BeliefProjectResponse | null;
  beliefBusy: boolean;
  beliefError: string | null;
  searchResult: SearchEvaluateResponse | null;
  searchBusy: boolean;
  searchError: string | null;
}

export const INITIAL_BELIEF_SEARCH: BeliefSearchState = {
  mode: "policy",
  worlds: 16,
  priorOnly: false,
  editorOpen: false,
  override: EMPTY_BELIEF_OVERRIDE,
  belief: null,
  projection: null,
  beliefBusy: false,
  beliefError: null,
  searchResult: null,
  searchBusy: false,
  searchError: null,
};

export interface TabletopBeliefDockProps {
  mode: DockAnalysisMode;
  worlds: SearchWorldBudget;
  priorOnly: boolean;
  editorOpen: boolean;
  overrideCount: number;
  belief: BeliefInferResponse | null;
  searchResult: SearchEvaluateResponse | null;
  searchBusy: boolean;
  searchError: string | null;
  onChangeMode: (mode: DockAnalysisMode) => void;
  onChangeWorlds: (worlds: SearchWorldBudget) => void;
  onRunSearch: () => void;
  onInferBelief: () => void;
  onToggleEditor: () => void;
}

export interface BeliefEditorDrawerProps {
  open: boolean;
  belief: BeliefInferResponse | null;
  projection: BeliefProjectResponse | null;
  override: BeliefOverridePayload;
  priorOnly: boolean;
  busy: boolean;
  error: string | null;
  onChangePriorOnly: (priorOnly: boolean) => void;
  onUpdateOverride: (next: BeliefOverridePayload) => void;
  onReset: () => void;
  onApplyAndRescore: () => void;
  onClose: () => void;
}

let nextBeliefSeq = 0;
let nextSearchSeq = 0;

export function bumpBeliefSearchSeq(): void {
  nextBeliefSeq += 1;
  nextSearchSeq += 1;
}

export function countActiveOverrides(override: BeliefOverridePayload): number {
  const pinned = Object.values(override.pinned_in_hand).filter((n) => n > 0).length;
  const mults = Object.values(override.copy_multipliers).filter((m) => m !== 1).length;
  return pinned + override.excluded_cards.length + mults;
}

export function formatSignedDeltaPct(value: number): string {
  const pct = (value * 100).toFixed(1);
  return value > 0 ? `+${pct}%` : `${pct}%`;
}

export function signedToneClass(delta: number): string {
  if (delta > 0.005) return "text-[var(--win)] font-semibold";
  if (delta < -0.005) return "text-[var(--loss)] font-semibold";
  return "text-[var(--text-3)]";
}

interface BeliefDispatchCtx {
  target: GameTarget;
  ckptA: string;
  setState: (fn: (prev: BeliefSearchState) => BeliefSearchState) => void;
}

function stepCoords(target: GameTarget): {
  runId: string;
  iteration: number;
  gameIdx: number;
  step: number;
} {
  return {
    runId: target.run_id,
    iteration: target.iteration,
    gameIdx: target.game_idx,
    step: target.step,
  };
}

function dispatchBeliefInfer(ctx: BeliefDispatchCtx, priorOnly: boolean): void {
  const seq = ++nextBeliefSeq;
  ctx.setState((p) => ({ ...p, priorOnly, beliefBusy: true, beliefError: null }));
  void postBeliefInfer({ ...stepCoords(ctx.target), priorOnly })
    .then((res) => {
      if (nextBeliefSeq !== seq) return;
      ctx.setState((p) => ({
        ...p,
        belief: res,
        projection: null,
        beliefBusy: false,
        beliefError: null,
      }));
    })
    .catch((err: unknown) => {
      if (nextBeliefSeq !== seq) return;
      const msg = err instanceof Error ? err.message : "Belief inference failed";
      ctx.setState((p) => ({ ...p, beliefBusy: false, beliefError: msg }));
    });
}

function dispatchBeliefProject(
  ctx: BeliefDispatchCtx,
  override: BeliefOverridePayload,
  priorOnly: boolean,
): void {
  const seq = ++nextBeliefSeq;
  ctx.setState((p) => ({ ...p, priorOnly, beliefBusy: true, beliefError: null }));
  void postBeliefProject({ ...stepCoords(ctx.target), priorOnly, override })
    .then((res) => {
      if (nextBeliefSeq !== seq) return;
      // Only commit override on HTTP 200 so a rejected 422 edit never desyncs state.override from projection.
      ctx.setState((p) =>
        res.ok
          ? { ...p, override, projection: res.data, beliefBusy: false, beliefError: null }
          : { ...p, beliefBusy: false, beliefError: res.detail },
      );
    })
    .catch((err: unknown) => {
      if (nextBeliefSeq !== seq) return;
      const msg = err instanceof Error ? err.message : "Belief projection failed";
      ctx.setState((p) => ({ ...p, beliefBusy: false, beliefError: msg }));
    });
}

function dispatchSearchEval(
  ctx: BeliefDispatchCtx,
  opts: { worlds: SearchWorldBudget; priorOnly: boolean; override: BeliefOverridePayload },
): void {
  const seq = ++nextSearchSeq;
  ctx.setState((p) => ({
    ...p,
    worlds: opts.worlds,
    priorOnly: opts.priorOnly,
    searchBusy: true,
    searchError: null,
  }));
  void postSearchEvaluate({
    ...stepCoords(ctx.target),
    ckptA: ctx.ckptA,
    worlds: opts.worlds,
    priorOnly: opts.priorOnly,
    override: countActiveOverrides(opts.override) > 0 ? opts.override : null,
  })
    .then((res) => {
      if (nextSearchSeq !== seq) return;
      ctx.setState((p) =>
        res.ok
          ? { ...p, searchResult: res.data, searchBusy: false, searchError: null }
          : { ...p, searchBusy: false, searchError: res.detail },
      );
    })
    .catch((err: unknown) => {
      if (nextSearchSeq !== seq) return;
      const msg = err instanceof Error ? err.message : "Search evaluation failed";
      ctx.setState((p) => ({ ...p, searchBusy: false, searchError: msg }));
    });
}

export function buildBeliefControllers(args: {
  target: GameTarget;
  ckptA: string;
  state: BeliefSearchState;
  setState: (fn: (prev: BeliefSearchState) => BeliefSearchState) => void;
}): { dock: TabletopBeliefDockProps; drawer: BeliefEditorDrawerProps } {
  const { state, setState } = args;
  const ctx: BeliefDispatchCtx = { target: args.target, ckptA: args.ckptA, setState };

  return {
    dock: {
      mode: state.mode,
      worlds: state.worlds,
      priorOnly: state.priorOnly,
      editorOpen: state.editorOpen,
      overrideCount: countActiveOverrides(state.override),
      belief: state.belief,
      searchResult: state.searchResult,
      searchBusy: state.searchBusy,
      searchError: state.searchError,
      onChangeMode: (nextMode): void => {
        setState((p) => ({ ...p, mode: nextMode }));
        if (nextMode !== "search") return;
        if (state.belief === null) dispatchBeliefInfer(ctx, state.priorOnly);
        dispatchSearchEval(ctx, {
          worlds: state.worlds,
          priorOnly: state.priorOnly,
          override: state.override,
        });
      },
      onChangeWorlds: (worlds): void =>
        dispatchSearchEval(ctx, { worlds, priorOnly: state.priorOnly, override: state.override }),
      onRunSearch: (): void =>
        dispatchSearchEval(ctx, {
          worlds: state.worlds,
          priorOnly: state.priorOnly,
          override: state.override,
        }),
      onInferBelief: (): void => dispatchBeliefInfer(ctx, state.priorOnly),
      onToggleEditor: (): void => {
        const opening = !state.editorOpen;
        setState((p) => ({ ...p, editorOpen: opening }));
        if (opening && state.belief === null) dispatchBeliefInfer(ctx, state.priorOnly);
      },
    },
    drawer: {
      open: state.editorOpen,
      belief: state.belief,
      projection: state.projection,
      override: state.override,
      priorOnly: state.priorOnly,
      busy: state.beliefBusy,
      error: state.beliefError,
      onChangePriorOnly: (nextPrior): void => {
        if (countActiveOverrides(state.override) > 0) {
          dispatchBeliefProject(ctx, state.override, nextPrior);
        } else {
          dispatchBeliefInfer(ctx, nextPrior);
        }
      },
      onUpdateOverride: (next): void => dispatchBeliefProject(ctx, next, state.priorOnly),
      onReset: (): void => {
        setState((p) => ({
          ...p,
          override: EMPTY_BELIEF_OVERRIDE,
          projection: null,
          beliefError: null,
        }));
        dispatchBeliefInfer(ctx, state.priorOnly);
        if (state.mode === "search") {
          dispatchSearchEval(ctx, {
            worlds: state.worlds,
            priorOnly: state.priorOnly,
            override: EMPTY_BELIEF_OVERRIDE,
          });
        }
      },
      onApplyAndRescore: (): void => {
        setState((p) => ({ ...p, mode: "search" }));
        dispatchSearchEval(ctx, {
          worlds: state.worlds,
          priorOnly: state.priorOnly,
          override: state.override,
        });
      },
      onClose: (): void => setState((p) => ({ ...p, editorOpen: false })),
    },
  };
}

export function useBeliefSearchState(
  target: GameTarget,
  ckptA: string,
): { dock: TabletopBeliefDockProps; drawer: BeliefEditorDrawerProps } {
  const [state, setState] = useState<BeliefSearchState>(INITIAL_BELIEF_SEARCH);
  const stepKey = `${target.run_id}:${target.iteration}:${target.game_idx}:${target.step}`;
  useResetOnKey(stepKey, (): void => {
    bumpBeliefSearchSeq();
    setState((prev) => ({
      ...INITIAL_BELIEF_SEARCH,
      mode: prev.mode,
      worlds: prev.worlds,
      priorOnly: prev.priorOnly,
    }));
  });
  return buildBeliefControllers({ target, ckptA, state, setState });
}

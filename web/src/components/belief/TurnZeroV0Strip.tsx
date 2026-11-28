/** Turn-0 Opening Hand vs V0 baseline comparison and pre-game V0 summary strip. */

import type { ReactNode } from "react";
import type {
  BeliefInferResponse,
  BoardState,
  EvaluationPair,
  SearchEvaluateResponse,
  TurnZeroV0Info,
} from "@/lib/types";
import {
  type TabletopBeliefDockProps,
  formatSignedDeltaPct,
  signedToneClass,
} from "./useBeliefSearch";

function resolveV0Equity(
  v0: TurnZeroV0Info | null,
  belief: BeliefInferResponse | null,
  searchResult: SearchEvaluateResponse | null,
): number | null {
  if (v0) return v0.v0_equity;
  if (belief) return belief.v0_equity;
  return searchResult?.v0_equity ?? null;
}

function resolveV0Context(
  board: BoardState,
  ctrl: TabletopBeliefDockProps,
): { v0: TurnZeroV0Info | null; isTurnZero: boolean; v0Equity: number | null } {
  const v0 = ctrl.belief?.v0 ?? ctrl.searchResult?.v0 ?? null;
  const isTurnZero =
    board.turn === 0 || board.action_space_kind === "MULLIGAN" || Boolean(v0?.is_turn0);
  const v0Equity = resolveV0Equity(v0, ctrl.belief, ctrl.searchResult);
  return { v0, isTurnZero, v0Equity };
}

function resolveTurnZeroNumbers(
  v0: TurnZeroV0Info | null,
  v0Equity: number | null,
  primaryWin: number,
): { keepV: number; delta: number | null; rec: "Keep" | "Mulligan" } {
  const keepV = v0 ? v0.keep_v : primaryWin;
  let delta: number | null = null;
  if (v0) delta = v0.delta;
  else if (v0Equity !== null) delta = keepV - v0Equity;
  let rec: "Keep" | "Mulligan" = "Keep";
  if (v0) rec = v0.recommendation;
  else if (delta !== null && delta < 0) rec = "Mulligan";
  return { keepV, delta, rec };
}

function OpeningHandV0Strip(props: {
  keepV: number;
  v0Equity: number | null;
  delta: number | null;
  rec: "Keep" | "Mulligan";
  onInferBelief: () => void;
}): ReactNode {
  const recCls =
    props.rec === "Keep" ? "text-[var(--win)] font-semibold" : "text-[var(--loss)] font-semibold";
  const v0Text = props.v0Equity === null ? "—" : `${(props.v0Equity * 100).toFixed(1)}%`;
  const deltaText = props.delta === null ? "—" : formatSignedDeltaPct(props.delta);
  const deltaCls = props.delta === null ? "text-[var(--text-3)]" : signedToneClass(props.delta);

  return (
    <div className="px-3 py-2 border-b border-[var(--border)] flex flex-col gap-1 shrink-0">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)]">
        <span>Turn 0 · Opening Hand vs V₀</span>
        {props.v0Equity === null ? (
          <button
            type="button"
            onClick={props.onInferBelief}
            className="normal-case font-medium text-[var(--accent)] hover:underline cursor-pointer"
          >
            Compute V₀
          </button>
        ) : (
          <span className={`normal-case tabular ${recCls}`}>{props.rec}</span>
        )}
      </div>
      <div className="grid grid-cols-3 gap-2 text-[10.5px] tabular">
        <div className="flex flex-col">
          <span className="text-[9.5px] text-[var(--text-3)]">Hand V(H)</span>
          <strong className="text-[var(--text)]">{(props.keepV * 100).toFixed(1)}%</strong>
        </div>
        <div className="flex flex-col">
          <span className="text-[9.5px] text-[var(--text-3)]">Deck V₀(D, P₀)</span>
          <strong className="text-[var(--text)]">{v0Text}</strong>
        </div>
        <div className="flex flex-col text-right">
          <span className="text-[9.5px] text-[var(--text-3)]">Delta</span>
          <span className={deltaCls}>{deltaText}</span>
        </div>
      </div>
    </div>
  );
}

export function TurnZeroV0Strip(props: {
  board: BoardState;
  evaluation: EvaluationPair;
  ctrl: TabletopBeliefDockProps;
}): ReactNode {
  const { v0, isTurnZero, v0Equity } = resolveV0Context(props.board, props.ctrl);
  if (!isTurnZero && v0Equity === null) return null;
  if (!isTurnZero && v0Equity !== null) {
    return (
      <div className="px-3 py-1.5 border-b border-[var(--border)] flex items-center justify-between text-[10.5px] shrink-0">
        <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)]">
          Pre-game V₀
        </span>
        <span className="tabular font-semibold text-[var(--text-2)]">
          {(v0Equity * 100).toFixed(1)}%
        </span>
      </div>
    );
  }
  const { keepV, delta, rec } = resolveTurnZeroNumbers(
    v0,
    v0Equity,
    props.evaluation.primary.wdl.win,
  );
  return (
    <OpeningHandV0Strip
      keepV={keepV}
      v0Equity={v0Equity}
      delta={delta}
      rec={rec}
      onInferBelief={props.ctrl.onInferBelief}
    />
  );
}

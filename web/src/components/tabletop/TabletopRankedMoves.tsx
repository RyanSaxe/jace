/** Ranked legal action probability list and move cards for the Tabletop Decision Analysis dock. */

import { EyeOff } from "lucide-react";
import { type ReactNode, useState } from "react";
import type { BoardState, EvaluationPair, LegalAction } from "@/lib/types";
import { deltaToneClass, formatSignedPct } from "./tabletopHelpers";

interface RankedMoveRow {
  act: LegalAction;
  pPrimary: number;
  pSecondary: number | null;
}

function buildRankedMoves(args: {
  actions: LegalAction[];
  probsA: number[];
  probsB: number[];
  baseProbs: number[];
  useBeforeAfter: boolean;
  useCompare: boolean;
}): RankedMoveRow[] {
  return args.actions
    .map((act, idx) => {
      const pPrimary = Math.round((args.probsA[idx] ?? 0) * 100);
      let pSecondary: number | null = null;
      if (args.useBeforeAfter) pSecondary = Math.round((args.baseProbs[idx] ?? 0) * 100);
      else if (args.useCompare) pSecondary = Math.round((args.probsB[idx] ?? 0) * 100);
      return { act, pPrimary, pSecondary };
    })
    .toSorted((a, b) => b.pPrimary - a.pPrimary);
}

function movesLegendLabel(args: {
  useBeforeAfter: boolean;
  useCompare: boolean;
  labelA: string;
  labelB: string;
}): string {
  if (args.useBeforeAfter) return "After vs Before";
  if (args.useCompare) return `${args.labelA} vs ${args.labelB}`;
  return "Click to play";
}

function RankedMoveCard(props: {
  rankIdx: number;
  act: LegalAction;
  primaryPct: number;
  secondaryPct: number | null;
  primaryColor: string;
  secondaryColor: string;
  isChosen: boolean;
  disabled: boolean;
  onBranchAction: (actionIdx: number) => void;
  onHoverFocus: (ids: number[]) => void;
}): ReactNode {
  const delta = props.secondaryPct === null ? 0 : props.primaryPct - props.secondaryPct;
  const topClass =
    props.rankIdx === 0
      ? "bg-[var(--accent-soft)]/55 border-[var(--accent)]/35 hover:border-[var(--accent)]"
      : "bg-[var(--bg)]/55 border-[var(--border)]/70 hover:bg-[var(--surface-2)] hover:border-[var(--accent)]/50";
  return (
    <button
      type="button"
      disabled={props.disabled}
      title={`Click to play: ${props.act.label}`}
      aria-label={`Play action ${props.rankIdx + 1}: ${props.act.label}`}
      onMouseEnter={(): void => props.onHoverFocus(props.act.focus_ids)}
      onMouseLeave={(): void => props.onHoverFocus([])}
      onClick={(): void => props.onBranchAction(props.act.index)}
      className={`w-full text-left rounded-md px-2.5 py-2 border transition-colors cursor-pointer disabled:opacity-50 ${topClass}`}
    >
      <div className="flex items-start justify-between gap-2 text-[11px] mb-1.5">
        <span className="font-semibold leading-snug break-words line-clamp-2 min-w-0">
          {props.rankIdx + 1}. {props.act.label}
        </span>
        <span className="tabular shrink-0 flex items-center gap-1 pt-0.5">
          {props.isChosen ? (
            <span title="Action taken in recorded game" className="text-[var(--accent)]">
              ★
            </span>
          ) : null}
          <strong>{props.primaryPct}%</strong>
          {props.secondaryPct !== null && delta !== 0 ? (
            <span className={`text-[10px] ${deltaToneClass(delta)}`}>
              ({formatSignedPct(delta)})
            </span>
          ) : null}
        </span>
      </div>
      <div className="flex flex-col gap-1">
        <div className="h-[5px] rounded-full bg-[var(--surface-2)] overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-150"
            style={{ width: `${Math.max(2, props.primaryPct)}%`, background: props.primaryColor }}
          />
        </div>
        {props.secondaryPct === null ? null : (
          <div className="h-[4px] rounded-full bg-[var(--surface-2)] overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-150 opacity-85"
              style={{
                width: `${Math.max(2, props.secondaryPct)}%`,
                background: props.secondaryColor,
              }}
            />
          </div>
        )}
      </div>
    </button>
  );
}

export function RankedMovesList(props: {
  board: BoardState;
  evaluation: EvaluationPair;
  baseEvaluation: EvaluationPair;
  showCompare: boolean;
  showEditCompare: boolean;
  labelA: string;
  labelB: string;
  forkBusy: boolean;
  branchDone: boolean;
  onBranchAction: (actionIdx: number) => void;
  onHoverFocus: (ids: number[]) => void;
}): ReactNode {
  const [showAll, setShowAll] = useState(false);

  if (props.board.actions_hidden) {
    return (
      <div className="p-3 flex-1">
        <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] mb-2">
          <span>Opponent Deciding</span>
          <span>Hero POV</span>
        </div>
        <div className="rounded-md border border-dashed border-[var(--border-strong)] bg-[var(--surface-2)] p-2.5 text-[11px] text-[var(--text-2)] flex items-start gap-2">
          <EyeOff className="w-4 h-4 shrink-0 mt-0.5 text-[var(--text-3)]" />
          <span>
            Opponent actions are hidden in <strong>Hero POV</strong>. Switch to{" "}
            <strong>Omniscient</strong> to inspect or branch Opponent&apos;s play.
          </span>
        </div>
      </div>
    );
  }

  const actions = props.board.legal_actions;
  const baseProbs = props.baseEvaluation.primary.probs;
  const useBeforeAfter = props.showEditCompare && baseProbs.length === actions.length;
  const useCompare = !useBeforeAfter && props.showCompare;
  const ranked = buildRankedMoves({
    actions,
    probsA: props.evaluation.primary.probs,
    probsB: props.evaluation.compare.probs,
    baseProbs,
    useBeforeAfter,
    useCompare,
  });
  const maxShow = 8;
  const visible = showAll ? ranked : ranked.slice(0, maxShow);
  const whoLabel = props.board.actor_is_hero === false ? "OPP" : "HERO";
  const legendRight = movesLegendLabel({
    useBeforeAfter,
    useCompare,
    labelA: props.labelA,
    labelB: props.labelB,
  });

  return (
    <div className="p-3 flex-1 min-h-0 flex flex-col gap-2 overflow-y-auto">
      <div className="flex items-center justify-between gap-2 text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] shrink-0">
        <span>
          {whoLabel} · {props.board.action_space_kind} ({actions.length})
        </span>
        <span className="truncate normal-case font-medium">{legendRight}</span>
      </div>

      {props.branchDone ? (
        <div className="rounded-md border border-[var(--win)]/40 bg-[var(--win-soft)] px-2.5 py-2 text-[11px] text-[var(--win)] font-semibold">
          Game ended along this branched line. Click Reset or Undo to continue exploring.
        </div>
      ) : null}

      <div className="flex flex-col gap-1.5">
        {visible.map((row, idx) => (
          <RankedMoveCard
            key={row.act.index}
            rankIdx={idx}
            act={row.act}
            primaryPct={row.pPrimary}
            secondaryPct={row.pSecondary}
            primaryColor={useBeforeAfter ? "var(--fork-fill)" : "var(--accent)"}
            secondaryColor="#d97706"
            isChosen={props.board.chosen_action === row.act.index}
            disabled={props.forkBusy || props.branchDone}
            onBranchAction={props.onBranchAction}
            onHoverFocus={props.onHoverFocus}
          />
        ))}
      </div>

      {actions.length > maxShow ? (
        <button
          type="button"
          onClick={(): void => setShowAll((prev) => !prev)}
          className="mt-1 w-full rounded-md border border-dashed border-[var(--border-strong)] py-1 text-[10.5px] font-medium text-[var(--text-2)] hover:text-[var(--text)] hover:border-[var(--accent)] cursor-pointer shrink-0"
        >
          {showAll ? `Show top ${maxShow} ▴` : `Show all ${actions.length} moves ▾`}
        </button>
      ) : null}
    </div>
  );
}

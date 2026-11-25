/** Right Swiss-grid inspector dock for Tabletop: checkpoint diff, Fork controls, legal actions, and WDL. */

import { Bookmark } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/Primitives";
import { cardImageUrl } from "@/lib/api";
import type { BoardState, EvaluationPair, ForkEdit } from "@/lib/types";

const FORK_CARD_PRESETS = [
  "Fire Lord Zuko",
  "Appa, Steadfast Guardian",
  "Airbender's Reversal",
  "Boiling Rock Rioter",
];

function ForkControlsSection(props: {
  forkBusy: boolean;
  editCount: number;
  baseWinPct: number;
  winPct: number;
  warnings: string[];
  onAddEdit: (edit: ForkEdit) => void;
}): ReactNode {
  const deltaWin = props.winPct - props.baseWinPct;
  return (
    <div className="p-3 border-b border-[var(--border)] flex flex-col gap-2">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider">
        <span className="text-[var(--text-3)]">Counterfactual Fork</span>
        <span className="tabular text-[var(--fork)]">
          {props.forkBusy ? "Evaluating…" : "POST /api/fork"}
        </span>
      </div>
      <div className="rounded-md border border-[var(--fork)]/30 bg-[var(--fork-soft)] px-2.5 py-1.5 flex items-center justify-between">
        <span className="text-[10.5px] text-[var(--text-2)]">
          {props.editCount} edit{props.editCount === 1 ? "" : "s"}
        </span>
        <span className="tabular text-xs font-bold text-[var(--fork)]">
          {props.baseWinPct}% → {props.winPct}% ({deltaWin >= 0 ? `+${deltaWin}` : deltaWin}%)
        </span>
      </div>
      <div>
        <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-3)] mb-1">
          + Force onto Hero Battlefield
        </div>
        <div className="grid grid-cols-2 gap-1">
          {FORK_CARD_PRESETS.map((cardName) => (
            <button
              key={cardName}
              type="button"
              onClick={(): void =>
                props.onAddEdit({ op: "add_battlefield", seat: 0, card: cardName })
              }
              className="flex items-center gap-1.5 rounded border border-[var(--border)] bg-[var(--bg)] p-1 text-left text-[10px] font-medium hover:border-[var(--fork)] hover:bg-[var(--fork-soft)] cursor-pointer min-w-0 overflow-hidden"
            >
              <img
                src={cardImageUrl(cardName)}
                alt={cardName}
                className="w-[18px] h-[25px] rounded-[2px] object-cover shrink-0"
              />
              <span className="truncate">{cardName}</span>
            </button>
          ))}
        </div>
      </div>
      {props.warnings.length > 0 ? (
        <div className="rounded bg-[var(--warn-soft)] text-[var(--warn)] px-2 py-1 text-[10px]">
          {props.warnings[0]}
        </div>
      ) : null}
    </div>
  );
}

function LegalActionsSection(props: {
  board: BoardState;
  evaluation: EvaluationPair;
  compact: boolean;
  onHoverFocus: (ids: number[]) => void;
}): ReactNode {
  const [showAll, setShowAll] = useState(false);
  const actions = props.board.legal_actions;
  const probsA = props.evaluation.primary.probs;
  const probsB = props.evaluation.compare.probs;
  const ranked = actions
    .map((act, idx) => ({
      act,
      pA: Math.round((probsA[idx] ?? 0) * 100),
      pB: Math.round((probsB[idx] ?? 0) * 100),
    }))
    .toSorted((a, b) => b.pA - a.pA);
  const maxShow = props.compact ? 4 : 5;
  const visible = showAll ? ranked : ranked.slice(0, maxShow);

  return (
    <div className="p-3 border-b border-[var(--border)]">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] mb-2">
        <span>
          {props.board.action_space_kind} ({showAll ? "ALL" : `TOP ${visible.length}`} OF{" "}
          {actions.length})
        </span>
        <span>Δ vs B</span>
      </div>
      <div className="flex flex-col gap-1.5">
        {visible.map((row, idx) => {
          const delta = row.pA - row.pB;
          const isChosen = props.board.chosen_action === row.act.index;
          return (
            <div
              key={row.act.index}
              onMouseEnter={(): void => props.onHoverFocus(row.act.focus_ids)}
              onMouseLeave={(): void => props.onHoverFocus([])}
              className={`rounded-md px-2 py-1 border cursor-pointer transition-colors ${
                idx === 0
                  ? "bg-[var(--win-soft)] border-[var(--win)]/25"
                  : "border-transparent hover:bg-[var(--surface-2)] hover:border-[var(--border)]"
              }`}
            >
              <div className="flex items-center justify-between gap-2 text-[11px] mb-1">
                <span className="font-semibold truncate">
                  {idx + 1}. {row.act.label} {isChosen ? "★" : ""}
                </span>
                <span className="tabular shrink-0 flex items-center gap-1">
                  <strong>{row.pA}%</strong>
                  {delta === 0 ? null : (
                    <span
                      className={
                        delta > 0
                          ? "text-[var(--win)] font-bold text-[10px]"
                          : "text-[var(--loss)] font-bold text-[10px]"
                      }
                    >
                      ({delta > 0 ? `+${delta}` : delta}%)
                    </span>
                  )}
                </span>
              </div>
              <div className="h-[5px] rounded bg-[var(--surface-2)] overflow-hidden">
                <div
                  className="h-full rounded transition-all"
                  style={{
                    width: `${Math.max(2, row.pA)}%`,
                    background: idx === 0 ? "var(--win-fill)" : "var(--accent)",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
      {actions.length > maxShow ? (
        <button
          type="button"
          onClick={(): void => setShowAll((prev) => !prev)}
          className="mt-2 w-full rounded-md border border-dashed border-[var(--border-strong)] py-1 text-[10.5px] font-medium text-[var(--text-2)] hover:text-[var(--text)] hover:border-[var(--accent)] cursor-pointer"
        >
          {showAll ? `Show top ${maxShow} actions ▴` : `Show all ${actions.length} legal actions ▾`}
        </button>
      ) : null}
    </div>
  );
}

export function TabletopDock(props: {
  board: BoardState;
  evaluation: EvaluationPair;
  baseEvaluation: EvaluationPair;
  checkpoints: string[];
  ckptA: string;
  ckptB: string;
  onChangeCkptA: (v: string) => void;
  onChangeCkptB: (v: string) => void;
  forkMode: boolean;
  forkBusy: boolean;
  forkEdits: ForkEdit[];
  forkWarnings: string[];
  onAddForkEdit: (edit: ForkEdit) => void;
  onHoverActionFocus: (ids: number[]) => void;
  onPinPosition: () => void;
  pinned: boolean;
}): ReactNode {
  const wdl = props.evaluation.primary.wdl;
  const winPct = Math.round(wdl.win * 100);
  const drawPct = Math.round(wdl.draw * 100);
  const lossPct = Math.max(0, 100 - winPct - drawPct);
  const baseWinPct = Math.round(props.baseEvaluation.primary.wdl.win * 100);

  return (
    <aside className="w-full lg:w-[318px] shrink-0 border-t lg:border-t-0 lg:border-l border-[var(--border)] bg-[var(--surface)] flex flex-col min-h-0 overflow-y-auto">
      <div className="p-3 border-b border-[var(--border)]">
        <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] mb-1.5">
          <span>Checkpoint Diff</span>
          <span className="tabular">
            TV {props.evaluation.diff.tv.toFixed(2)} · KL {props.evaluation.diff.kl.toFixed(2)}
          </span>
        </div>
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-1.5">
          <select
            value={props.ckptA}
            onChange={(e): void => props.onChangeCkptA(e.target.value)}
            aria-label="Primary checkpoint"
            className="h-7 rounded-md border border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)] px-1.5 text-[11px] font-semibold tabular"
          >
            <option value="">Heuristic</option>
            {props.checkpoints.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <span className="text-[10px] text-[var(--text-3)]">vs</span>
          <select
            value={props.ckptB}
            onChange={(e): void => props.onChangeCkptB(e.target.value)}
            aria-label="Comparison checkpoint"
            className="h-7 rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-1.5 text-[11px] font-semibold tabular"
          >
            <option value="">Same</option>
            {props.checkpoints.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
      </div>

      {props.forkMode ? (
        <ForkControlsSection
          forkBusy={props.forkBusy}
          editCount={props.forkEdits.length}
          baseWinPct={baseWinPct}
          winPct={winPct}
          warnings={props.forkWarnings}
          onAddEdit={props.onAddForkEdit}
        />
      ) : null}

      <LegalActionsSection
        board={props.board}
        evaluation={props.evaluation}
        compact={props.forkMode}
        onHoverFocus={props.onHoverActionFocus}
      />

      <div className="p-3">
        <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] mb-1.5">
          <span>Value Estimate (WDL)</span>
          <span className="tabular">{props.evaluation.primary.checkpoint}</span>
        </div>
        <div className="h-[18px] rounded-[5px] overflow-hidden flex text-[10px] font-bold text-white tabular mb-1.5">
          <div
            className="bg-[var(--win-fill)] flex items-center justify-center"
            style={{ width: `${winPct}%` }}
          >
            {winPct >= 16 ? `${winPct}%` : ""}
          </div>
          <div className="bg-zinc-600" style={{ width: `${drawPct}%` }} />
          <div
            className="bg-[var(--loss-fill)] flex items-center justify-center"
            style={{ width: `${lossPct}%` }}
          >
            {lossPct >= 14 ? `${lossPct}%` : ""}
          </div>
        </div>
        <div className="tabular text-[10.5px] text-[var(--text-2)] mb-2.5">
          {winPct}% Win · {drawPct}% Draw · {lossPct}% Loss
        </div>
        <Button
          variant={props.pinned ? "primary" : "default"}
          className="w-full"
          onClick={props.onPinPosition}
        >
          <Bookmark className="w-3.5 h-3.5" />
          {props.pinned ? "Pinned to Watchlist" : "Pin position to Watchlist"}
        </Button>
      </div>
    </aside>
  );
}

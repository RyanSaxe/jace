/** Policy Evolution Matrix table, probability cells, and empty/loading body states. */

import { X } from "lucide-react";
import type { ReactNode } from "react";
import type { GlobalCheckpointOption, LegalAction, PolicyMatrixColumn } from "@/lib/types";
import { PlayoutCell, type PlayoutState } from "./PolicyMatrixPlayouts";
import { formatCheckpointShort } from "./tabletopHelpers";

function sortMatrixActions(
  actions: LegalAction[],
  columns: PolicyMatrixColumn[],
): Array<{ action: LegalAction; originalIdx: number }> {
  const primaryCol = columns[0];
  return actions
    .map((action, originalIdx) => ({ action, originalIdx }))
    .toSorted(
      (a, b) => (primaryCol?.probs[b.originalIdx] ?? 0) - (primaryCol?.probs[a.originalIdx] ?? 0),
    );
}

function MatrixCell(props: { prob: number; isTop: boolean }): ReactNode {
  const pct = Math.round(props.prob * 100);
  return (
    <td
      aria-label={`${pct}% probability${props.isTop ? ", top choice" : ""}`}
      className={`px-2.5 py-2 align-middle border-b border-[var(--border)] ${
        props.isTop ? "bg-[var(--accent-soft)]/70" : ""
      }`}
    >
      <div className="flex flex-col gap-1 min-w-[92px]">
        <div className="flex items-center justify-between gap-1 tabular text-[11px]">
          <span className={props.isTop ? "font-bold text-[var(--accent)]" : "font-medium"}>
            {pct}%
          </span>
          {props.isTop ? (
            <span className="rounded bg-[var(--accent)] px-1 text-[8.5px] font-bold text-white uppercase">
              Top
            </span>
          ) : null}
        </div>
        <div className="h-[5px] rounded-full bg-[var(--surface-2)] overflow-hidden">
          <div
            className="h-full rounded-full"
            style={{
              width: `${Math.max(2, pct)}%`,
              background: props.isTop ? "var(--accent)" : "var(--text-3)",
            }}
          />
        </div>
      </div>
    </td>
  );
}

function PolicyMatrixTable(props: {
  actions: LegalAction[];
  columns: PolicyMatrixColumn[];
  checkpoints: GlobalCheckpointOption[];
  currentRunId: string;
  playout: PlayoutState;
  hasPlayoutColumn: boolean;
  onRemoveModel: (ref: string) => void;
}): ReactNode {
  const sorted = sortMatrixActions(props.actions, props.columns);
  return (
    <div className="flex-1 min-h-0 overflow-auto">
      <table className="w-full border-collapse text-left">
        <thead className="sticky top-0 z-10 bg-[var(--surface)] shadow-[0_1px_0_var(--border)]">
          <tr>
            <th className="px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] min-w-[200px]">
              Legal Action ({sorted.length})
            </th>
            {props.columns.map((col) => {
              const title = formatCheckpointShort(col.ref, props.checkpoints, props.currentRunId);
              return (
                <th
                  key={col.ref}
                  className="px-2.5 py-2 text-[10.5px] font-semibold text-[var(--text)] whitespace-nowrap"
                >
                  <div className="flex items-center justify-between gap-1.5">
                    <span className="tabular truncate max-w-[140px]" title={col.label || col.ref}>
                      {title}
                    </span>
                    <button
                      type="button"
                      onClick={(): void => props.onRemoveModel(col.ref)}
                      aria-label={`Remove ${title} column`}
                      className="rounded p-0.5 text-[var(--text-3)] hover:text-[var(--loss)] hover:bg-[var(--loss-soft)] cursor-pointer"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                </th>
              );
            })}
            {props.hasPlayoutColumn ? (
              <th className="px-2.5 py-2 text-[10px] font-bold uppercase tracking-wider text-[var(--win)] whitespace-nowrap">
                Empirical Win% [95% CI]
              </th>
            ) : null}
          </tr>
        </thead>
        <tbody>
          {sorted.map(({ action, originalIdx }) => (
            <tr key={action.index} className="hover:bg-[var(--surface-2)]/40 transition-colors">
              <td className="px-3 py-2 border-b border-[var(--border)] text-[11.5px] font-medium leading-snug">
                {action.label}
              </td>
              {props.columns.map((col) => (
                <MatrixCell
                  key={`${col.ref}-${action.index}`}
                  prob={col.probs[originalIdx] ?? 0}
                  isTop={col.top_action_index === action.index}
                />
              ))}
              {props.hasPlayoutColumn ? (
                <PlayoutCell row={props.playout.byAction[action.index]} />
              ) : null}
            </tr>
          ))}
        </tbody>
        {props.columns.length > 0 ? (
          <tfoot className="sticky bottom-0 bg-[var(--surface-2)] border-t border-[var(--border-strong)]">
            <tr>
              <td className="px-3 py-2 text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-2)]">
                Hero Win% Estimate
              </td>
              {props.columns.map((col) => {
                const winPct = Math.round(col.wdl.win * 100);
                const drawPct = Math.round(col.wdl.draw * 100);
                const lossPct = Math.max(0, 100 - winPct - drawPct);
                return (
                  <td
                    key={`wdl-${col.ref}`}
                    title={`Win ${winPct}% · Draw ${drawPct}% · Loss ${lossPct}%`}
                    className="px-2.5 py-2 tabular text-[11.5px] font-bold text-[var(--text)]"
                  >
                    {winPct}%
                  </td>
                );
              })}
              {props.hasPlayoutColumn ? (
                <td className="px-2.5 py-2 tabular text-[10.5px] text-[var(--text-2)]">
                  {props.playout.completedDeals} deals
                </td>
              ) : null}
            </tr>
          </tfoot>
        ) : null}
      </table>
    </div>
  );
}

export function MatrixBodySection(props: {
  actionsHidden: boolean;
  selectedModels: string[];
  matrixLoading: boolean;
  actions: LegalAction[];
  columns: PolicyMatrixColumn[];
  checkpoints: GlobalCheckpointOption[];
  currentRunId: string;
  playout: PlayoutState;
  hasPlayoutColumn: boolean;
  onRemoveModel: (ref: string) => void;
}): ReactNode {
  if (props.actionsHidden) {
    return (
      <div className="p-6 text-center text-xs text-[var(--text-2)]">
        Opponent actions are hidden in <strong>Hero POV</strong>. Switch to{" "}
        <strong>Omniscient</strong> to inspect the policy matrix for this step.
      </div>
    );
  }
  if (props.selectedModels.length === 0) {
    return (
      <div className="p-6 text-center text-xs text-[var(--text-2)]">
        No models selected. Click <strong>+ Add model</strong> to compare checkpoints side by side.
      </div>
    );
  }
  if (props.matrixLoading && props.columns.length === 0) {
    return (
      <div className="p-8 text-center text-xs text-[var(--text-3)]">
        Evaluating {props.selectedModels.length} checkpoint
        {props.selectedModels.length === 1 ? "" : "s"}…
      </div>
    );
  }
  return (
    <PolicyMatrixTable
      actions={props.actions}
      columns={props.columns}
      checkpoints={props.checkpoints}
      currentRunId={props.currentRunId}
      playout={props.playout}
      hasPlayoutColumn={props.hasPlayoutColumn}
      onRemoveModel={props.onRemoveModel}
    />
  );
}

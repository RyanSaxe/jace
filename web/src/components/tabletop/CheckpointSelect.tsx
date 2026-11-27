/** Cross-run grouped checkpoint selector shared by TabletopDock and PolicyMatrixModal. */

import type { ReactNode } from "react";
import type { GlobalCheckpointOption } from "@/lib/types";
import { groupCheckpointsByRun } from "./tabletopHelpers";

export function CheckpointSelect(props: {
  value: string;
  onChange: (value: string) => void;
  checkpoints: GlobalCheckpointOption[];
  currentRunId: string;
  ariaLabel: string;
  includeNone?: boolean;
  noneLabel?: string;
  excludeRefs?: ReadonlySet<string>;
  className?: string;
}): ReactNode {
  const groups = groupCheckpointsByRun(props.checkpoints, props.currentRunId);
  const exclude = props.excludeRefs;
  return (
    <select
      value={props.value}
      onChange={(e): void => props.onChange(e.target.value)}
      aria-label={props.ariaLabel}
      className={
        props.className ??
        "h-7 rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-1.5 text-[11px] font-semibold tabular min-w-0"
      }
    >
      {props.includeNone ? (
        <option value="none">{props.noneLabel ?? "None (single)"}</option>
      ) : null}
      {exclude?.has("heuristic") ? null : <option value="heuristic">Heuristic</option>}
      {groups.map((group) => {
        const visible = exclude
          ? group.options.filter((opt) => !exclude.has(opt.ref))
          : group.options;
        if (visible.length === 0) return null;
        return (
          <optgroup key={group.runId} label={group.label}>
            {visible.map((opt) => (
              <option key={opt.ref} value={opt.ref}>
                {opt.iteration >= 0 ? `u${opt.iteration} · ${opt.checkpoint}` : opt.checkpoint}
              </option>
            ))}
          </optgroup>
        );
      })}
    </select>
  );
}

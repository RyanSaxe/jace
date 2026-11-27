/** Toolbar, status notices, and sortable position diff table for Compare Models. */

import { GitCompareArrows, Play, SlidersHorizontal, Square, X } from "lucide-react";
import type { ReactNode } from "react";
import { Badge, Button, SegmentedControl } from "@/components/ui/Primitives";
import type { ComparePositionRow, GlobalCheckpointOption, RunSummary } from "@/lib/types";
import {
  BOX,
  CARD,
  type CheckpointGroup,
  type FilterState,
  LBL,
  type OvState,
  PILL,
  PILL_OFF,
  PILL_ON,
  PRESET,
  SEL,
  SORT_OPTIONS,
  type ScanState,
  type StatusFilter,
  type ViewData,
  clampSamples,
  groupCheckpoints,
  humanizeToken,
  shortRef,
  signedDelta,
} from "./compareUtils";

function CheckpointPicker(props: {
  label: string;
  dot: string;
  value: string;
  groups: CheckpointGroup[];
  onChange: (v: string) => void;
}): ReactNode {
  const hasVal = props.groups.some((g) => g.options.some((o) => o.ref === props.value));
  const custom = Boolean(props.value) && props.value !== "heuristic" && !hasVal;
  return (
    <label className="flex items-center gap-1.5 text-[11px] text-[var(--text-2)] min-w-0">
      <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: props.dot }} />
      <span className="font-semibold text-[var(--text)] shrink-0">{props.label}</span>
      <select
        aria-label={props.label}
        value={props.value}
        onChange={(e): void => props.onChange(e.target.value)}
        className={`${SEL} max-w-[190px]`}
      >
        {custom ? <option value={props.value}>{shortRef(props.value)}</option> : null}
        {props.groups.map((g) => (
          <optgroup key={g.runId} label={g.label}>
            {g.options.map((o) => (
              <option key={o.ref} value={o.ref}>
                {o.iteration >= 0 ? `${o.checkpoint} (u${o.iteration})` : o.checkpoint}
              </option>
            ))}
          </optgroup>
        ))}
        <option value="heuristic">Heuristic baseline</option>
      </select>
    </label>
  );
}

export function CompareToolbar(props: {
  sel: { ckptA: string; ckptB: string; runId: string };
  sampleText: string;
  loading: boolean;
  runs: RunSummary[];
  checkpoints: GlobalCheckpointOption[];
  onPatchOv: (patch: Partial<OvState>) => void;
  onSamples: (v: string) => void;
  onRun: () => void;
  onStop: () => void;
}): ReactNode {
  const groups = groupCheckpoints(props.checkpoints);
  const parsed = clampSamples(props.sampleText);
  const { sel } = props;
  return (
    <div className="min-h-[44px] px-3 py-1.5 border-b border-[var(--border)] bg-[var(--surface)] flex flex-wrap items-center gap-x-2.5 gap-y-1.5 shrink-0">
      <div className="flex items-center gap-1.5 pr-1">
        <GitCompareArrows className="w-4 h-4 text-[var(--accent)] shrink-0" />
        <h2 className="text-xs font-semibold tracking-tight m-0">Compare</h2>
      </div>
      <CheckpointPicker
        label="Model A"
        dot="var(--accent)"
        value={sel.ckptA}
        groups={groups}
        onChange={(a): void => props.onPatchOv({ a })}
      />
      <span className="text-[10.5px] text-[var(--text-3)]">vs</span>
      <CheckpointPicker
        label="Model B"
        dot="var(--warn)"
        value={sel.ckptB}
        groups={groups}
        onChange={(b): void => props.onPatchOv({ b })}
      />
      <select
        aria-label="Source run filter"
        value={sel.runId}
        onChange={(e): void => props.onPatchOv({ run: e.target.value })}
        className={`${SEL} max-w-[168px]`}
      >
        <option value="">All playable games</option>
        {props.runs.map((r) => (
          <option key={r.run_id} value={r.run_id}>
            {r.name || r.run_id}
          </option>
        ))}
      </select>
      <label className="flex items-center gap-1 text-[11px] text-[var(--text-2)]">
        Samples
        <input
          type="number"
          min={1}
          value={props.sampleText}
          onChange={(e): void => props.onSamples(e.target.value)}
          onBlur={(): void => props.onSamples(String(parsed))}
          onKeyDown={(e): void => {
            if (e.key === "Enter" && !props.loading) props.onRun();
          }}
          aria-label="Sample count"
          className={`tabular h-7 w-[62px] ${BOX} px-1.5 text-[11.5px] text-[var(--text)]`}
        />
      </label>
      <div className="hidden sm:flex items-center gap-0.5">
        {[50, 120, 250, 500].map((n) => (
          <button
            key={n}
            type="button"
            onClick={(): void => props.onSamples(String(n))}
            className={`${PRESET} ${parsed === n && props.sampleText.trim() !== "" ? PILL_ON : PILL_OFF}`}
          >
            {n}
          </button>
        ))}
      </div>
      <Button
        variant={props.loading ? "default" : "primary"}
        onClick={props.loading ? props.onStop : props.onRun}
        disabled={!props.loading && (!sel.ckptA || !sel.ckptB)}
        className={`ml-auto h-7 ${props.loading ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]" : ""}`}
      >
        {props.loading ? (
          <>
            <span className="w-2 h-2 rounded-full bg-[var(--accent)] animate-pulse" />
            <Square className="w-3 h-3 fill-current" /> Stop
          </>
        ) : (
          <>
            <Play className="w-3 h-3 fill-current" /> Run Compare
          </>
        )}
      </Button>
    </div>
  );
}

export function CompareNotices(props: {
  scan: ScanState;
  warnings: string[];
  scanLabel: string;
}): ReactNode {
  const { scan, warnings } = props;
  if (!scan.loading && !scan.stopped && !scan.error && warnings.length === 0) return null;
  const cls = "rounded-md border px-3 py-1.5";
  return (
    <div className="flex flex-col gap-2 text-xs">
      {scan.loading ? (
        <div
          className={`${cls} border-[var(--accent)]/40 bg-[var(--accent-soft)] flex flex-col gap-1.5`}
        >
          <div className="flex items-center justify-between gap-2 text-[var(--accent)] font-medium">
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[var(--accent)] animate-ping" />
              {props.scanLabel}
            </span>
            <span className="text-[11px] font-normal text-[var(--text-2)]">Stop anytime</span>
          </div>
          <div className="h-1.5 rounded-full bg-[var(--surface)] overflow-hidden">
            <div className="h-full w-1/2 rounded-full bg-[var(--accent)] animate-pulse" />
          </div>
        </div>
      ) : null}
      {scan.stopped && !scan.loading ? (
        <div className={`${cls} border-[var(--border)] bg-[var(--surface)] text-[var(--text-2)]`}>
          Comparison scan stopped.
        </div>
      ) : null}
      {scan.error ? (
        <div className={`${cls} border-[var(--loss)] bg-[var(--loss-soft)] text-[var(--loss)]`}>
          {scan.error}
        </div>
      ) : null}
      {warnings.length > 0 ? (
        <div className={`${cls} border-[var(--warn)]/40 bg-[var(--warn-soft)] text-[var(--warn)]`}>
          {warnings.join(" · ")}
        </div>
      ) : null}
    </div>
  );
}

export function CompareDiffTable(props: {
  view: ViewData;
  filter: FilterState;
  onFilter: (patch: Partial<FilterState>) => void;
  onSelect: (k: string) => void;
  onOpenRow: (r: ComparePositionRow) => void;
}): ReactNode {
  const { view, filter } = props;
  const btns: ReadonlyArray<[StatusFilter, string]> = [
    ["all", `All (${view.total})`],
    ["changed", `Move changed only (${view.changed})`],
  ];
  return (
    <div className={`${CARD} overflow-hidden flex flex-col`}>
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-[var(--border)]">
        <div className="flex flex-wrap items-center gap-1.5">
          {btns.map(([val, label]) => (
            <button
              key={val}
              type="button"
              onClick={(): void => props.onFilter({ status: val })}
              className={`${PILL} ${filter.status === val ? PILL_ON : PILL_OFF}`}
            >
              {label}
            </button>
          ))}
          {filter.kind ? (
            <span className={`inline-flex items-center gap-1 ${PILL} ${PILL_ON}`}>
              <SlidersHorizontal className="w-3 h-3" /> {humanizeToken(filter.kind)}
              <button
                type="button"
                aria-label="Clear decision kind filter"
                onClick={(): void => props.onFilter({ kind: null })}
                className="ml-0.5 cursor-pointer"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          ) : null}
        </div>
        <SegmentedControl
          options={SORT_OPTIONS}
          value={filter.sort}
          onChange={(sort): void => props.onFilter({ sort })}
        />
      </div>
      {view.filteredRows.length === 0 ? (
        <div className="p-8 text-center text-xs text-[var(--text-3)]">
          No sampled positions match the current filters.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-[11.5px]">
            <thead>
              <tr className={`border-b border-[var(--border)] bg-[var(--bg)] ${LBL}`}>
                {[
                  "Turn / Step",
                  "Decision",
                  "Status",
                  "Model A Top Move",
                  "Model B Top Move",
                  "TV",
                  "Win Δ",
                ].map((lbl, i) => (
                  <th
                    key={lbl}
                    className={`py-1.5 px-2.5 whitespace-nowrap ${i >= 5 ? "text-right" : ""}`}
                  >
                    {lbl}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {view.filteredRows.map((row) => {
                const key = `${row.run_id}:${row.iteration}:${row.game_idx}:${row.step}`;
                const bg =
                  key === view.activeKey
                    ? "bg-[var(--accent-soft)]"
                    : "hover:bg-[var(--surface-2)]/70";
                const moves: ReadonlyArray<[string, string, number, string]> = [
                  ["Model A", row.top_action_a, row.top_prob_a, "text-[var(--accent)]"],
                  ["Model B", row.top_action_b, row.top_prob_b, "text-[var(--warn)]"],
                ];
                return (
                  <tr
                    key={key}
                    onClick={(): void => props.onSelect(key)}
                    onDoubleClick={(): void => props.onOpenRow(row)}
                    title="Click to preview · Double-click to open in Tabletop"
                    className={`border-b border-[var(--border)] cursor-pointer transition-colors ${bg}`}
                  >
                    <td className="py-1.5 px-2.5 tabular whitespace-nowrap font-medium">
                      T{row.turn} <span className="text-[var(--text-3)]">#{row.step}</span>
                    </td>
                    <td className="py-1.5 px-2.5 whitespace-nowrap">
                      {humanizeToken(row.decision_kind)}{" "}
                      <span className="ml-1 text-[10px] uppercase text-[var(--text-3)]">
                        {row.actor_side}
                      </span>
                    </td>
                    <td className="py-1.5 px-2.5 whitespace-nowrap">
                      <Badge tone={row.move_changed ? "accent" : "neutral"}>
                        {row.move_changed ? "Move changed" : "Same move"}
                      </Badge>
                    </td>
                    {moves.map(([side, act, prob, color]) => (
                      <td
                        key={side}
                        className="py-1.5 px-2.5 max-w-[180px]"
                        aria-label={`${side}: ${act} (${Math.round(prob * 100)}%)`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="truncate" title={act}>
                            {act}
                          </span>
                          <span className={`tabular text-[10.5px] font-semibold shrink-0 ${color}`}>
                            {Math.round(prob * 100)}%
                          </span>
                        </div>
                      </td>
                    ))}
                    <td className="py-1.5 px-2.5 text-right tabular font-medium">
                      {row.tv.toFixed(2)}
                    </td>
                    <td className="py-1.5 px-2.5 text-right tabular text-[var(--text-2)]">
                      {signedDelta(row.win_delta)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

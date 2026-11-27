/** Plot picker popover, legend strip, and multi-plot grid for the Training Runs view. */

import { X } from "lucide-react";
import { type ReactNode, useState } from "react";
import type { RunDetail, RunSummary } from "@/lib/types";
import type { ChartSeriesInput, DiscoveredMetric, XAxisMode } from "./metrics";
import { SyncedChartCard } from "./SyncedChartCard";

function MetricGroupList(props: {
  heading: string;
  metrics: DiscoveredMetric[];
  selectedIds: Set<string>;
  onToggleMetric: (id: string) => void;
}): ReactNode {
  if (props.metrics.length === 0) return null;
  return (
    <div className="flex flex-col gap-0.5">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-3)] px-1">
        {props.heading}
      </div>
      {props.metrics.map((m) => (
        <label
          key={m.id}
          className="flex items-center gap-2 px-1.5 py-1 rounded hover:bg-[var(--surface-2)] text-xs cursor-pointer"
        >
          <input
            type="checkbox"
            checked={props.selectedIds.has(m.id)}
            onChange={(): void => props.onToggleMetric(m.id)}
            className="accent-[var(--accent)]"
          />
          <span className="truncate" title={m.key}>
            {m.label}
          </span>
        </label>
      ))}
    </div>
  );
}

export function PlotPickerPopover(props: {
  allMetrics: DiscoveredMetric[];
  selectedIds: Set<string>;
  onToggleMetric: (id: string) => void;
  onResetDefaults: () => void;
  onClose: () => void;
}): ReactNode {
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const filtered = needle
    ? props.allMetrics.filter(
        (m) => m.key.toLowerCase().includes(needle) || m.label.toLowerCase().includes(needle),
      )
    : props.allMetrics;

  return (
    <div className="absolute right-0 top-full mt-1.5 z-40 w-[290px] max-w-[90vw] rounded-lg border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-md)] p-2.5 flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-2)]">
          Pick Plots ({props.selectedIds.size}/{props.allMetrics.length})
        </span>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={props.onResetDefaults}
            className="text-[11px] text-[var(--accent)] hover:underline cursor-pointer"
          >
            Defaults
          </button>
          <button
            type="button"
            onClick={props.onClose}
            aria-label="Close plot picker"
            className="p-0.5 rounded hover:bg-[var(--surface-2)] cursor-pointer"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
      <input
        type="search"
        value={query}
        onChange={(e): void => setQuery(e.target.value)}
        placeholder="Search any metric…"
        aria-label="Search metrics"
        className="h-7 w-full rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs"
      />
      <div className="max-h-[260px] overflow-y-auto flex flex-col gap-2 pr-0.5">
        <MetricGroupList
          heading="Eval Metrics"
          metrics={filtered.filter((m) => m.source === "eval")}
          selectedIds={props.selectedIds}
          onToggleMetric={props.onToggleMetric}
        />
        <MetricGroupList
          heading="Train Metrics"
          metrics={filtered.filter((m) => m.source === "train")}
          selectedIds={props.selectedIds}
          onToggleMetric={props.onToggleMetric}
        />
        {filtered.length === 0 ? (
          <div className="py-4 text-center text-xs text-[var(--text-3)]">No matching metrics.</div>
        ) : null}
      </div>
    </div>
  );
}

export function RunsLegendBar(props: {
  checkedRuns: RunSummary[];
  colorByRunId: Map<string, string>;
  hoverX: number | null;
  xMode: XAxisMode;
  onDeepDive: (runId: string) => void;
  onUncheck: (runId: string) => void;
}): ReactNode {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
      <div className="flex flex-wrap items-center gap-1.5 min-w-0">
        {props.checkedRuns.length === 0 ? (
          <span className="text-[var(--text-3)]">
            Check runs in the table below to overlay curves.
          </span>
        ) : (
          props.checkedRuns.map((run) => (
            <span
              key={run.run_id}
              className="inline-flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--surface)] px-2 py-0.5 text-[11px]"
            >
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ background: props.colorByRunId.get(run.run_id) ?? "#4f46e5" }}
              />
              <button
                type="button"
                onClick={(): void => props.onDeepDive(run.run_id)}
                className="max-w-[140px] truncate font-medium hover:text-[var(--accent)] cursor-pointer"
              >
                {run.name}
              </button>
              <button
                type="button"
                onClick={(): void => props.onUncheck(run.run_id)}
                aria-label={`Remove ${run.name} from charts`}
                className="text-[var(--text-3)] hover:text-[var(--text)] cursor-pointer"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          ))
        )}
      </div>
      {props.hoverX === null ? null : (
        <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[10.5px] font-semibold bg-[var(--accent-soft)] text-[var(--accent)] tabular whitespace-nowrap">
          Synced crosshair:{" "}
          {props.xMode === "progress"
            ? `${Math.round(props.hoverX)}%`
            : `u${Math.round(props.hoverX)}`}
        </span>
      )}
    </div>
  );
}

export function PlotsGridSection(props: {
  visibleMetrics: DiscoveredMetric[];
  checkedRuns: RunSummary[];
  details: Record<string, RunDetail>;
  colorByRunId: Map<string, string>;
  smoothing: number;
  showCi: boolean;
  xMode: XAxisMode;
  hoverX: number | null;
  loading: boolean;
  onHoverX: (x: number | null) => void;
}): ReactNode {
  if (props.visibleMetrics.length === 0) {
    return (
      <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-6 text-center text-xs text-[var(--text-3)]">
        {props.loading
          ? "Loading telemetry metrics…"
          : "No plots selected. Use Pick Plots above to choose metrics."}
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
      {props.visibleMetrics.map((metric) => {
        const series: ChartSeriesInput[] = [];
        for (const run of props.checkedRuns) {
          const detail = props.details[run.run_id];
          if (!detail) continue;
          series.push({
            id: run.run_id,
            label: run.name,
            color: props.colorByRunId.get(run.run_id) ?? "#4f46e5",
            totalUpdates: run.total_updates,
            points: metric.source === "eval" ? detail.eval_points : detail.train_points,
          });
        }
        return (
          <SyncedChartCard
            key={metric.id}
            title={metric.label}
            badge={metric.source}
            metricKey={metric.key}
            series={series}
            smoothing={props.smoothing}
            showCi={props.showCi}
            xMode={props.xMode}
            hoverX={props.hoverX}
            onHoverX={props.onHoverX}
          />
        );
      })}
    </div>
  );
}

/** Toolbar, mobile action bottom sheet, and multi-select sticky action bar for Training Runs. */

import {
  Activity,
  ArrowRight,
  GitCompareArrows,
  Layers,
  Plus,
  RefreshCw,
  SlidersHorizontal,
  X,
} from "lucide-react";
import type { ReactNode } from "react";
import { Badge, Button, SegmentedControl } from "@/components/ui/Primitives";
import type { CompareTarget, GlobalCheckpointOption, RunDetail, RunSummary } from "@/lib/types";
import {
  type DiscoveredMetric,
  type XAxisMode,
  formatPctOrDash,
  latestCheckpointRefForRun,
  resolveLatestWin,
  runStatusMeta,
} from "./metrics";
import { PlotPickerPopover } from "./RunsCharts";

export function RunsToolbar(props: {
  runCount: number;
  experiments: string[];
  activeExperiment: string;
  filterText: string;
  smoothing: number;
  showCi: boolean;
  xMode: XAxisMode;
  allMetrics: DiscoveredMetric[];
  selectedMetricIds: Set<string>;
  pickerOpen: boolean;
  loading: boolean;
  onSelectExperiment: (exp: string) => void;
  onChangeFilter: (text: string) => void;
  onToggleSmoothing: () => void;
  onToggleCi: () => void;
  onChangeXMode: (mode: XAxisMode) => void;
  onTogglePicker: () => void;
  onToggleMetric: (id: string) => void;
  onResetMetrics: () => void;
  onReload: () => void;
}): ReactNode {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2 border-b border-[var(--border)] bg-[var(--surface)] shrink-0">
      <div className="flex flex-wrap items-center gap-2 min-w-0">
        <div className="flex items-center gap-1.5 shrink-0">
          <Activity className="w-4 h-4 text-[var(--accent)]" />
          <h2 className="text-sm font-semibold tracking-tight">Training Runs</h2>
          <Badge>{props.runCount}</Badge>
        </div>
        {props.experiments.length > 1 ? (
          <select
            aria-label="Filter by experiment"
            value={props.activeExperiment}
            onChange={(e): void => props.onSelectExperiment(e.target.value)}
            className="h-7 max-w-[200px] rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs text-[var(--text)]"
          >
            <option value="">All experiments ({props.experiments.length})</option>
            {props.experiments.map((exp) => (
              <option key={exp} value={exp}>
                {exp}
              </option>
            ))}
          </select>
        ) : null}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <input
          type="search"
          value={props.filterText}
          onChange={(e): void => props.onChangeFilter(e.target.value)}
          placeholder="Filter runs…"
          aria-label="Filter runs"
          className="h-7 w-32 sm:w-36 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs"
        />
        <Button
          size="sm"
          variant={props.smoothing > 0 ? "primary" : "default"}
          onClick={props.onToggleSmoothing}
          title="Toggle EMA smoothing (0 vs 0.6)"
        >
          {props.smoothing > 0 ? `EMA ${props.smoothing.toFixed(1)}` : "Smooth 0"}
        </Button>
        <Button
          size="sm"
          variant={props.showCi ? "primary" : "default"}
          onClick={props.onToggleCi}
          title="Toggle 95% confidence bands on rate charts"
        >
          95% CI
        </Button>
        <SegmentedControl
          options={[
            { value: "updates", label: "Updates" },
            { value: "progress", label: "Progress %" },
          ]}
          value={props.xMode}
          onChange={props.onChangeXMode}
        />
        <div className="relative">
          <Button size="sm" onClick={props.onTogglePicker}>
            <Plus className="w-3.5 h-3.5" />
            Pick Plots ({props.selectedMetricIds.size}/{props.allMetrics.length})
          </Button>
          {props.pickerOpen ? (
            <PlotPickerPopover
              allMetrics={props.allMetrics}
              selectedIds={props.selectedMetricIds}
              onToggleMetric={props.onToggleMetric}
              onResetDefaults={props.onResetMetrics}
              onClose={props.onTogglePicker}
            />
          ) : null}
        </div>
        <Button size="sm" onClick={props.onReload} disabled={props.loading} title="Reload runs">
          <RefreshCw className={`w-3.5 h-3.5 ${props.loading ? "animate-spin" : ""}`} />
          <span className="sr-only">Reload runs</span>
        </Button>
      </div>
    </div>
  );
}

export function MobileRunBottomSheet(props: {
  run: RunSummary | null;
  checked: boolean;
  onClose: () => void;
  onDeepDive: (runId: string) => void;
  onBrowseGames: (runId: string) => void;
  onToggleChart: (runId: string) => void;
}): ReactNode {
  if (!props.run) return null;
  const run = props.run;
  const status = runStatusMeta(run.status);
  return (
    <div className="fixed inset-0 z-50 flex flex-col justify-end md:hidden">
      <button
        type="button"
        aria-label="Dismiss run actions"
        onClick={props.onClose}
        className="flex-1 bg-black/45 border-0"
      />
      <div className="rounded-t-2xl border-t border-[var(--border-strong)] bg-[var(--surface)] p-4 pb-6 shadow-[var(--shadow-lg)] flex flex-col gap-3">
        <div className="w-10 h-1 rounded-full bg-[var(--border-strong)] mx-auto" />
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm truncate">{run.name}</span>
              <Badge tone={status.tone}>{status.label}</Badge>
            </div>
            <div className="text-xs text-[var(--text-3)] mt-0.5 tabular">
              {run.experiment} · u{run.latest_update} · Latest{" "}
              {formatPctOrDash(resolveLatestWin(run))}
            </div>
          </div>
          <button
            type="button"
            onClick={props.onClose}
            aria-label="Close"
            className="p-1.5 rounded-md hover:bg-[var(--surface-2)] cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="flex flex-col gap-2">
          <Button
            variant="primary"
            size="md"
            className="w-full justify-between"
            onClick={(): void => {
              props.onClose();
              props.onDeepDive(run.run_id);
            }}
          >
            <span>Open Deep Dive</span>
            <ArrowRight className="w-4 h-4" />
          </Button>
          <Button
            size="md"
            className="w-full justify-between"
            onClick={(): void => {
              props.onClose();
              props.onBrowseGames(run.run_id);
            }}
          >
            <span>Browse Games</span>
            <Layers className="w-4 h-4" />
          </Button>
          <Button
            size="md"
            className="w-full justify-between"
            onClick={(): void => {
              props.onToggleChart(run.run_id);
              props.onClose();
            }}
          >
            <span>{props.checked ? "Hide from Charts" : "Show on Charts"}</span>
            <Activity className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

function resolveCompareMeta(
  checkedRuns: RunSummary[],
  details: Record<string, RunDetail>,
  checkpoints: GlobalCheckpointOption[],
): { target: CompareTarget | null; label: string; title: string } {
  const runA = checkedRuns[0];
  const runB = checkedRuns[1];
  const nameA = runA?.name ?? "A";
  const nameB = runB?.name ?? "B";
  const ckptA = runA ? latestCheckpointRefForRun(runA, details[runA.run_id], checkpoints) : null;
  const ckptB = runB ? latestCheckpointRefForRun(runB, details[runB.run_id], checkpoints) : null;
  const target = ckptA && ckptB ? { ckptA, ckptB } : null;
  const title = target
    ? `Compare ${nameA} vs ${nameB}`
    : "Both runs need at least one saved checkpoint to compare";
  return { target, label: `Compare ${nameA.slice(0, 12)} vs ${nameB.slice(0, 12)} →`, title };
}

export function StickySelectionBar(props: {
  checkedRuns: RunSummary[];
  colorByRunId: Map<string, string>;
  details: Record<string, RunDetail>;
  checkpoints: GlobalCheckpointOption[];
  soloMode: boolean;
  onToggleSolo: () => void;
  onDiffConfigs: () => void;
  onComparePair: (target: CompareTarget) => void;
  onClear: () => void;
}): ReactNode {
  if (props.checkedRuns.length < 2) return null;
  const compare = resolveCompareMeta(props.checkedRuns, props.details, props.checkpoints);
  return (
    <div className="sticky bottom-3 z-20 rounded-xl border border-[var(--border-strong)] bg-[var(--surface)]/95 backdrop-blur px-3.5 py-2.5 shadow-[var(--shadow-md)] flex flex-wrap items-center justify-between gap-2">
      <div className="flex flex-wrap items-center gap-1.5 min-w-0">
        <span className="text-[11px] font-semibold text-[var(--text-2)] mr-1">
          {props.checkedRuns.length} selected
        </span>
        {props.checkedRuns.slice(0, 4).map((run) => (
          <span
            key={run.run_id}
            className="inline-flex items-center gap-1 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 py-0.5 text-[11px] max-w-[130px]"
          >
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ background: props.colorByRunId.get(run.run_id) ?? "var(--accent)" }}
            />
            <span className="truncate">{run.name}</span>
          </span>
        ))}
        {props.checkedRuns.length > 4 ? <Badge>+{props.checkedRuns.length - 4}</Badge> : null}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <Button
          size="sm"
          variant={props.soloMode ? "primary" : "default"}
          onClick={props.onToggleSolo}
        >
          Solo on charts
        </Button>
        <Button size="sm" onClick={props.onDiffConfigs}>
          <SlidersHorizontal className="w-3.5 h-3.5" />
          Diff Configs
        </Button>
        <Button
          size="sm"
          variant="primary"
          disabled={compare.target === null}
          title={compare.title}
          onClick={(): void => {
            if (compare.target) props.onComparePair(compare.target);
          }}
        >
          <GitCompareArrows className="w-3.5 h-3.5" />
          {compare.label}
        </Button>
        <Button size="sm" variant="ghost" onClick={props.onClear} title="Clear selection">
          <X className="w-3.5 h-3.5" />
          <span className="sr-only">Clear selection</span>
        </Button>
      </div>
    </div>
  );
}

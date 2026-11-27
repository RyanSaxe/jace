/** Training Runs Option A: Full-width multi-plot grid, multi-select sortable table, mobile cards + sheet, and Deep Dive. */

import {
  Activity,
  ArrowRight,
  ChevronRight,
  GitCompareArrows,
  Layers,
  Plus,
  RefreshCw,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";
import { RunDeepDive } from "@/components/RunDeepDive";
import {
  type DiscoveredMetric,
  PlotPickerPopover,
  PlotsGridSection,
  RunsLegendBar,
  type XAxisMode,
  colorForIndex,
  discoverCatalogMetrics,
  discoverPopulatedMetricIds,
  formatMetricValue,
  formatPctOrDash,
  humanizeMetricKey,
  isRateMetric,
  isWinRateKey,
  latestCheckpointRefForRun,
  numericFromPoint,
  runStatusMeta,
  selectDefaultMetricIds,
} from "@/components/RunsCharts";
import { Badge, Button, SegmentedControl } from "@/components/ui/Primitives";
import { useExtraRunDetails } from "@/hooks/useStudioData";
import type {
  CompareTarget,
  GlobalCheckpointOption,
  MetricPoint,
  RunDetail,
  RunSummary,
} from "@/lib/types";

export interface RunsViewProps {
  runs: RunSummary[];
  details: Record<string, RunDetail>;
  checkpoints: GlobalCheckpointOption[];
  experiments: string[];
  evalMetricKeys: string[];
  trainMetricKeys: string[];
  selectedRunId: string | null;
  onSelectRunDeepDive: (runId: string | null) => void;
  loading: boolean;
  error: string | null;
  onReload: () => void;
  onBrowseGames: (runId: string) => void;
  onComparePair: (target: CompareTarget) => void;
}

type SortKey = "updated" | "name" | "update" | "bestWin" | "latestWin" | "secondary" | "games";

function resolveBestWin(run: RunSummary): number | null {
  const standard = [run.peak_live_win, run.peak_ema_win, run.peak_ensemble_win].filter(
    (val): val is number => typeof val === "number" && Number.isFinite(val),
  );
  if (standard.length > 0) return Math.max(...standard);
  const fallback: number[] = [];
  if (run.best_metrics) {
    for (const [key, val] of Object.entries(run.best_metrics)) {
      if (isWinRateKey(key) && Number.isFinite(val)) fallback.push(val);
    }
  }
  return fallback.length > 0 ? Math.max(...fallback) : null;
}

function resolveLatestWin(run: RunSummary): number | null {
  const standard = run.latest_ema_win ?? run.latest_ensemble_win ?? run.latest_live_win;
  if (typeof standard === "number" && Number.isFinite(standard)) return standard;
  if (run.latest_metrics) {
    for (const [key, val] of Object.entries(run.latest_metrics)) {
      if (isWinRateKey(key) && Number.isFinite(val)) return val;
    }
  }
  return null;
}

function latestNumericFromPoints(points: MetricPoint[], key: string): number | null {
  for (let i = points.length - 1; i >= 0; i -= 1) {
    const pt = points[i];
    if (!pt) continue;
    const val = numericFromPoint(pt, key);
    if (val !== null) return val;
  }
  return null;
}

function secondaryMetricForRun(
  run: RunSummary,
  detail: RunDetail | undefined,
  key: string | null,
): number | null {
  if (!key) return run.latest_tps > 0 ? run.latest_tps : null;
  const fromSummary = run.latest_metrics?.[key];
  if (typeof fromSummary === "number" && Number.isFinite(fromSummary)) return fromSummary;
  return detail ? latestNumericFromPoints(detail.eval_points, key) : null;
}

function PlayableGamesBadge(props: {
  run: RunSummary;
  onBrowseGames: (runId: string) => void;
}): ReactNode {
  const { run } = props;
  const hasPlayableField = typeof run.playable_games === "number";
  const count = hasPlayableField ? (run.playable_games ?? 0) : run.eval_files;
  if (count > 0) {
    return (
      <button
        type="button"
        onClick={(e): void => {
          e.stopPropagation();
          props.onBrowseGames(run.run_id);
        }}
        className="cursor-pointer"
      >
        <Badge tone="win">{hasPlayableField ? `${count} playable` : `${count} evals`}</Badge>
      </button>
    );
  }
  if (hasPlayableField && (run.legacy_games ?? 0) > 0) {
    return <Badge tone="neutral">{run.legacy_games} legacy</Badge>;
  }
  return <span className="text-[11px] text-[var(--text-3)]">No games</span>;
}

function sortMetricValue(args: {
  run: RunSummary;
  detail: RunDetail | undefined;
  sortKey: SortKey;
  secondaryKey: string | null;
}): number | string {
  const { run, detail, secondaryKey } = args;
  const getters: Record<SortKey, () => number | string> = {
    name: (): string => run.name.toLowerCase(),
    update: (): number => run.latest_update,
    bestWin: (): number => resolveBestWin(run) ?? -1,
    latestWin: (): number => resolveLatestWin(run) ?? -1,
    secondary: (): number => secondaryMetricForRun(run, detail, secondaryKey) ?? -1,
    games: (): number => run.playable_games ?? run.eval_files,
    updated: (): number => run.updated_at,
  };
  return getters[args.sortKey]();
}

function filterAndSortRuns(args: {
  runs: RunSummary[];
  details: Record<string, RunDetail>;
  experiment: string;
  filterText: string;
  sortKey: SortKey;
  sortDir: "desc" | "asc";
  secondaryKey: string | null;
}): RunSummary[] {
  const needle = args.filterText.trim().toLowerCase();
  const matched = args.runs.filter((r) => {
    if (args.experiment && r.experiment !== args.experiment) return false;
    if (!needle) return true;
    return (
      r.run_id.toLowerCase().includes(needle) ||
      r.name.toLowerCase().includes(needle) ||
      r.experiment.toLowerCase().includes(needle) ||
      r.status.toLowerCase().includes(needle)
    );
  });
  const sign = args.sortDir === "desc" ? -1 : 1;
  return matched.toSorted((a, b) => {
    const va = sortMetricValue({
      run: a,
      detail: args.details[a.run_id],
      sortKey: args.sortKey,
      secondaryKey: args.secondaryKey,
    });
    const vb = sortMetricValue({
      run: b,
      detail: args.details[b.run_id],
      sortKey: args.sortKey,
      secondaryKey: args.secondaryKey,
    });
    if (typeof va === "string" && typeof vb === "string") return va.localeCompare(vb) * sign;
    return (Number(va) - Number(vb)) * sign;
  });
}

function RunsToolbar(props: {
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

function SortHeaderButton(props: {
  colKey: SortKey;
  label: string;
  activeKey: SortKey;
  arrow: string;
  onSort: (key: SortKey) => void;
}): ReactNode {
  return (
    <button
      type="button"
      onClick={(): void => props.onSort(props.colKey)}
      className="cursor-pointer hover:text-[var(--text)] whitespace-nowrap"
    >
      {props.label}
      {props.activeKey === props.colKey ? props.arrow : ""}
    </button>
  );
}

function DesktopRunRow(props: {
  run: RunSummary;
  detail: RunDetail | undefined;
  checked: boolean;
  seriesColor: string | undefined;
  secondaryKey: string | null;
  onToggleCheck: (runId: string) => void;
  onDeepDive: (runId: string) => void;
  onBrowseGames: (runId: string) => void;
}): ReactNode {
  const { run, checked, seriesColor } = props;
  const status = runStatusMeta(run.status);
  const secVal = secondaryMetricForRun(run, props.detail, props.secondaryKey);
  const secIsPct =
    props.secondaryKey !== null && secVal !== null && isRateMetric(props.secondaryKey, [secVal]);

  return (
    <tr
      className="border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--surface-2)] transition-colors"
      style={checked ? { background: "var(--accent-soft)" } : undefined}
    >
      <td className="py-2 px-3">
        <label className="inline-flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={checked}
            onChange={(): void => props.onToggleCheck(run.run_id)}
            aria-label={`Plot ${run.name}`}
            style={seriesColor ? { accentColor: seriesColor } : undefined}
            className="cursor-pointer"
          />
          {seriesColor ? (
            <span className="w-2 h-2 rounded-full shrink-0" style={{ background: seriesColor }} />
          ) : null}
        </label>
      </td>
      <td className="py-2 px-2">
        <button
          type="button"
          aria-label={`Open deep dive for ${run.name}`}
          onClick={(): void => props.onDeepDive(run.run_id)}
          className="group inline-flex items-center gap-1.5 text-left font-semibold text-[var(--text)] hover:text-[var(--accent)] cursor-pointer max-w-[240px]"
        >
          <span className="truncate">{run.name}</span>
          <span className="text-[10.5px] font-medium text-[var(--accent)] opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
            Deep dive →
          </span>
        </button>
      </td>
      <td className="py-2 px-2 text-[var(--text-2)] max-w-[140px] truncate">{run.experiment}</td>
      <td className="py-2 px-2">
        <Badge tone={status.tone}>{status.label}</Badge>
      </td>
      <td className="py-2 px-2 tabular text-right">
        {run.latest_update}
        {run.total_updates > 0 ? (
          <span className="text-[var(--text-3)]">/{run.total_updates}</span>
        ) : null}
      </td>
      <td className="py-2 px-2 tabular text-right font-semibold">
        {formatPctOrDash(resolveBestWin(run))}
      </td>
      <td className="py-2 px-2 tabular text-right font-semibold text-[var(--win)]">
        {formatPctOrDash(resolveLatestWin(run))}
      </td>
      <td className="py-2 px-2 tabular text-right text-[var(--text-2)]">
        {secVal === null ? "—" : formatMetricValue(secVal, secIsPct)}
      </td>
      <td className="py-2 px-3 text-right">
        <PlayableGamesBadge run={run} onBrowseGames={props.onBrowseGames} />
      </td>
    </tr>
  );
}

function DesktopRunsTable(props: {
  runs: RunSummary[];
  details: Record<string, RunDetail>;
  checkedSet: Set<string>;
  colorByRunId: Map<string, string>;
  secondaryKey: string | null;
  sortKey: SortKey;
  sortDir: "desc" | "asc";
  loading: boolean;
  onSort: (key: SortKey) => void;
  onToggleCheck: (runId: string) => void;
  onToggleAll: () => void;
  onDeepDive: (runId: string) => void;
  onBrowseGames: (runId: string) => void;
}): ReactNode {
  const arrow = props.sortDir === "desc" ? " ↓" : " ↑";
  const allChecked =
    props.runs.length > 0 && props.runs.every((r) => props.checkedSet.has(r.run_id));
  const secondaryHeader = props.secondaryKey ? humanizeMetricKey(props.secondaryKey) : "TPS";
  const sortableCols: ReadonlyArray<{ key: SortKey; label: string; padEnd?: boolean }> = [
    { key: "update", label: "Updates" },
    { key: "bestWin", label: "Best Win%" },
    { key: "latestWin", label: "Latest Win%" },
    { key: "secondary", label: secondaryHeader },
    { key: "games", label: "Playable Games", padEnd: true },
  ];

  return (
    <div className="hidden md:block rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-x-auto">
      <table className="w-full min-w-[740px] border-collapse text-left text-xs">
        <thead>
          <tr className="border-b border-[var(--border)] bg-[var(--surface-2)] text-[10.5px] uppercase tracking-wider text-[var(--text-3)]">
            <th className="py-2 px-3 w-10">
              <input
                type="checkbox"
                checked={allChecked}
                onChange={props.onToggleAll}
                aria-label="Toggle all visible runs"
                className="accent-[var(--accent)] cursor-pointer"
              />
            </th>
            <th className="py-2 px-2 font-semibold">
              <SortHeaderButton
                colKey="name"
                label="Run"
                activeKey={props.sortKey}
                arrow={arrow}
                onSort={props.onSort}
              />
            </th>
            <th className="py-2 px-2 font-semibold">Experiment</th>
            <th className="py-2 px-2 font-semibold">Status</th>
            {sortableCols.map((col) => (
              <th
                key={col.key}
                className={`py-2 ${col.padEnd ? "px-3" : "px-2"} font-semibold text-right`}
              >
                <SortHeaderButton
                  colKey={col.key}
                  label={col.label}
                  activeKey={props.sortKey}
                  arrow={arrow}
                  onSort={props.onSort}
                />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {props.runs.length === 0 ? (
            <tr>
              <td colSpan={9} className="py-8 text-center text-[var(--text-3)]">
                {props.loading ? "Scanning runs directory…" : "No runs match that filter."}
              </td>
            </tr>
          ) : (
            props.runs.map((run) => (
              <DesktopRunRow
                key={run.run_id}
                run={run}
                detail={props.details[run.run_id]}
                checked={props.checkedSet.has(run.run_id)}
                seriesColor={props.colorByRunId.get(run.run_id)}
                secondaryKey={props.secondaryKey}
                onToggleCheck={props.onToggleCheck}
                onDeepDive={props.onDeepDive}
                onBrowseGames={props.onBrowseGames}
              />
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

function MobileRunCards(props: {
  runs: RunSummary[];
  checkedSet: Set<string>;
  colorByRunId: Map<string, string>;
  loading: boolean;
  onToggleCheck: (runId: string) => void;
  onOpenSheet: (run: RunSummary) => void;
  onBrowseGames: (runId: string) => void;
}): ReactNode {
  return (
    <div className="flex flex-col gap-2 md:hidden">
      {props.runs.length === 0 ? (
        <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-6 text-center text-xs text-[var(--text-3)]">
          {props.loading ? "Scanning runs directory…" : "No runs match that filter."}
        </div>
      ) : (
        props.runs.map((run) => {
          const checked = props.checkedSet.has(run.run_id);
          const color = props.colorByRunId.get(run.run_id);
          const status = runStatusMeta(run.status);
          return (
            <div
              key={run.run_id}
              className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3 flex flex-col gap-2"
              style={checked ? { borderColor: color ?? "var(--accent)" } : undefined}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-start gap-2 min-w-0">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(): void => props.onToggleCheck(run.run_id)}
                    aria-label={`Plot ${run.name}`}
                    style={color ? { accentColor: color } : undefined}
                    className="mt-0.5 cursor-pointer"
                  />
                  <button
                    type="button"
                    onClick={(): void => props.onOpenSheet(run)}
                    className="text-left min-w-0 cursor-pointer"
                  >
                    <div className="font-semibold text-xs truncate">{run.name}</div>
                    <div className="text-[11px] text-[var(--text-3)] truncate">
                      {run.experiment}
                    </div>
                  </button>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <Badge tone={status.tone}>{status.label}</Badge>
                  <button
                    type="button"
                    onClick={(): void => props.onOpenSheet(run)}
                    aria-label={`Actions for ${run.name}`}
                    className="p-1 rounded hover:bg-[var(--surface-2)] text-[var(--text-3)] cursor-pointer"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2 pt-1 border-t border-[var(--border)] text-[11px]">
                <div>
                  <div className="text-[10px] text-[var(--text-3)] uppercase">Updates</div>
                  <div className="tabular font-medium">
                    u{run.latest_update}
                    {run.total_updates > 0 ? `/${run.total_updates}` : ""}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-[var(--text-3)] uppercase">Latest Win</div>
                  <div className="tabular font-semibold text-[var(--win)]">
                    {formatPctOrDash(resolveLatestWin(run))}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] text-[var(--text-3)] uppercase mb-0.5">Games</div>
                  <PlayableGamesBadge run={run} onBrowseGames={props.onBrowseGames} />
                </div>
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}

function MobileRunBottomSheet(props: {
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

function StickySelectionBar(props: {
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

function toggleStringId(base: string[], target: string): string[] {
  return base.includes(target) ? base.filter((id) => id !== target) : [...base, target];
}

export function RunsView(props: RunsViewProps): ReactNode {
  const [activeExperiment, setActiveExperiment] = useState("");
  const [filterText, setFilterText] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("updated");
  const [sortDir, setSortDir] = useState<"desc" | "asc">("desc");
  const [smoothing, setSmoothing] = useState(0);
  const [showCi, setShowCi] = useState(true);
  const [xMode, setXMode] = useState<XAxisMode>("updates");
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [customCheckedIds, setCustomCheckedIds] = useState<string[] | null>(null);
  const [customMetricIds, setCustomMetricIds] = useState<string[] | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [soloMode, setSoloMode] = useState(false);
  const [mobileSheetRunId, setMobileSheetRunId] = useState<string | null>(null);
  const [pendingDiffRunId, setPendingDiffRunId] = useState<string | null>(null);

  const experiments = useMemo(
    () =>
      props.experiments.length > 0
        ? props.experiments
        : [...new Set(props.runs.map((r) => r.experiment).filter(Boolean))].toSorted((a, b) =>
            a.localeCompare(b),
          ),
    [props.experiments, props.runs],
  );
  const prelimMetrics = useMemo(
    () =>
      discoverCatalogMetrics({
        evalMetricKeys: props.evalMetricKeys,
        trainMetricKeys: props.trainMetricKeys,
        details: props.details,
        runs: props.runs,
      }),
    [props.evalMetricKeys, props.trainMetricKeys, props.details, props.runs],
  );
  const secondaryKey = useMemo(
    () => prelimMetrics.find((m) => m.source === "eval" && !isWinRateKey(m.key))?.key ?? null,
    [prelimMetrics],
  );
  const filteredRuns = useMemo(
    () =>
      filterAndSortRuns({
        runs: props.runs,
        details: props.details,
        experiment: activeExperiment,
        filterText,
        sortKey,
        sortDir,
        secondaryKey,
      }),
    [props.runs, props.details, activeExperiment, filterText, sortKey, sortDir, secondaryKey],
  );
  const checkedIds = useMemo(
    () => customCheckedIds ?? filteredRuns.slice(0, 4).map((r) => r.run_id),
    [customCheckedIds, filteredRuns],
  );
  const allDetails = useExtraRunDetails(checkedIds, props.details);
  const allMetrics = useMemo(
    () =>
      discoverCatalogMetrics({
        evalMetricKeys: props.evalMetricKeys,
        trainMetricKeys: props.trainMetricKeys,
        details: allDetails,
        runs: props.runs,
      }),
    [props.evalMetricKeys, props.trainMetricKeys, allDetails, props.runs],
  );
  const selectedMetricIds = useMemo(() => {
    const valid = new Set(allMetrics.map((m) => m.id));
    if (customMetricIds !== null) return new Set(customMetricIds.filter((id) => valid.has(id)));
    return new Set(
      selectDefaultMetricIds(allMetrics, 6, discoverPopulatedMetricIds(checkedIds, allDetails)),
    );
  }, [customMetricIds, allMetrics, checkedIds, allDetails]);

  const visibleMetrics = useMemo(
    () => allMetrics.filter((m) => selectedMetricIds.has(m.id)),
    [allMetrics, selectedMetricIds],
  );
  const checkedSet = useMemo(() => new Set(checkedIds), [checkedIds]);
  const colorByRunId = useMemo(
    () => new Map(checkedIds.map((id, idx) => [id, colorForIndex(idx)])),
    [checkedIds],
  );
  const checkedRuns = useMemo(() => {
    const byId = new Map(props.runs.map((r) => [r.run_id, r]));
    return checkedIds.map((id) => byId.get(id)).filter((r): r is RunSummary => r !== undefined);
  }, [props.runs, checkedIds]);
  const effectiveSolo = soloMode && checkedRuns.length >= 2;
  const tableRuns = effectiveSolo
    ? filteredRuns.filter((r) => checkedSet.has(r.run_id))
    : filteredRuns;

  const deepDiveRun = props.selectedRunId
    ? (props.runs.find((r) => r.run_id === props.selectedRunId) ?? null)
    : null;
  if (deepDiveRun) {
    return (
      <RunDeepDive
        run={deepDiveRun}
        runs={props.runs}
        details={allDetails}
        initialDiffRunId={pendingDiffRunId}
        onBack={(): void => {
          setPendingDiffRunId(null);
          props.onSelectRunDeepDive(null);
        }}
        onBrowseGames={props.onBrowseGames}
        onComparePair={props.onComparePair}
      />
    );
  }

  const openDeepDive = (runId: string): void => {
    setPendingDiffRunId(null);
    props.onSelectRunDeepDive(runId);
  };
  const toggleCheck = (runId: string): void =>
    setCustomCheckedIds((prev) => toggleStringId(prev ?? checkedIds, runId));
  const mobileSheetRun = props.runs.find((r) => r.run_id === mobileSheetRunId) ?? null;

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      <RunsToolbar
        runCount={filteredRuns.length}
        experiments={experiments}
        activeExperiment={activeExperiment}
        filterText={filterText}
        smoothing={smoothing}
        showCi={showCi}
        xMode={xMode}
        allMetrics={allMetrics}
        selectedMetricIds={selectedMetricIds}
        pickerOpen={pickerOpen}
        loading={props.loading}
        onSelectExperiment={setActiveExperiment}
        onChangeFilter={setFilterText}
        onToggleSmoothing={(): void => setSmoothing((s) => (s > 0 ? 0 : 0.6))}
        onToggleCi={(): void => setShowCi((v) => !v)}
        onChangeXMode={setXMode}
        onTogglePicker={(): void => setPickerOpen((v) => !v)}
        onToggleMetric={(id): void =>
          setCustomMetricIds((prev) => toggleStringId(prev ?? [...selectedMetricIds], id))
        }
        onResetMetrics={(): void => setCustomMetricIds(null)}
        onReload={props.onReload}
      />

      <div className="flex-1 overflow-y-auto p-3.5 sm:p-4 flex flex-col gap-3">
        {props.error ? (
          <div className="rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] p-3 text-xs text-[var(--loss)]">
            {props.error}
          </div>
        ) : null}

        <RunsLegendBar
          checkedRuns={checkedRuns}
          colorByRunId={colorByRunId}
          hoverX={hoverX}
          xMode={xMode}
          onDeepDive={openDeepDive}
          onUncheck={(runId): void =>
            setCustomCheckedIds((prev) => (prev ?? checkedIds).filter((id) => id !== runId))
          }
        />

        <PlotsGridSection
          visibleMetrics={visibleMetrics}
          checkedRuns={checkedRuns}
          details={allDetails}
          colorByRunId={colorByRunId}
          smoothing={smoothing}
          showCi={showCi}
          xMode={xMode}
          hoverX={hoverX}
          loading={props.loading}
          onHoverX={setHoverX}
        />

        <DesktopRunsTable
          runs={tableRuns}
          details={allDetails}
          checkedSet={checkedSet}
          colorByRunId={colorByRunId}
          secondaryKey={secondaryKey}
          sortKey={sortKey}
          sortDir={sortDir}
          loading={props.loading}
          onSort={(nextKey): void => {
            if (sortKey === nextKey) setSortDir((d) => (d === "desc" ? "asc" : "desc"));
            else {
              setSortKey(nextKey);
              setSortDir("desc");
            }
          }}
          onToggleCheck={toggleCheck}
          onToggleAll={(): void => {
            const all =
              filteredRuns.length > 0 && filteredRuns.every((r) => checkedSet.has(r.run_id));
            setCustomCheckedIds(all ? [] : filteredRuns.map((r) => r.run_id));
          }}
          onDeepDive={openDeepDive}
          onBrowseGames={props.onBrowseGames}
        />

        <MobileRunCards
          runs={tableRuns}
          checkedSet={checkedSet}
          colorByRunId={colorByRunId}
          loading={props.loading}
          onToggleCheck={toggleCheck}
          onOpenSheet={(run): void => setMobileSheetRunId(run.run_id)}
          onBrowseGames={props.onBrowseGames}
        />

        {customCheckedIds === null ? null : (
          <StickySelectionBar
            checkedRuns={checkedRuns}
            colorByRunId={colorByRunId}
            details={allDetails}
            checkpoints={props.checkpoints}
            soloMode={effectiveSolo}
            onToggleSolo={(): void => setSoloMode((v) => !v)}
            onDiffConfigs={(): void => {
              const first = checkedRuns[0];
              const second = checkedRuns[1];
              if (!first || !second) return;
              setPendingDiffRunId(second.run_id);
              props.onSelectRunDeepDive(first.run_id);
            }}
            onComparePair={props.onComparePair}
            onClear={(): void => {
              setCustomCheckedIds(null);
              setSoloMode(false);
            }}
          />
        )}
      </div>

      <MobileRunBottomSheet
        run={mobileSheetRun}
        checked={mobileSheetRun ? checkedSet.has(mobileSheetRun.run_id) : false}
        onClose={(): void => setMobileSheetRunId(null)}
        onDeepDive={openDeepDive}
        onBrowseGames={props.onBrowseGames}
        onToggleChart={toggleCheck}
      />
    </div>
  );
}

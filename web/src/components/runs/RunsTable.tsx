/** Sortable desktop runs table, mobile run cards, and run filtering/sorting helpers. */

import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";
import { Badge } from "@/components/ui/Primitives";
import type { MetricPoint, RunDetail, RunSummary } from "@/lib/types";
import {
  formatMetricValue,
  formatPctOrDash,
  humanizeMetricKey,
  isRateMetric,
  numericFromPoint,
  resolveBestWin,
  resolveLatestWin,
  runStatusMeta,
} from "./metrics";

export type SortKey =
  | "updated"
  | "name"
  | "update"
  | "bestWin"
  | "latestWin"
  | "secondary"
  | "games";

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

export function filterAndSortRuns(args: {
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

export function DesktopRunsTable(props: {
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

export function MobileRunCards(props: {
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

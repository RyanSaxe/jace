/** Linear-inspired Runs Overview: on-demand multi-run Eval Win% curves, sortable table, config & MTG diagnostics. */

import { Activity, ArrowRight, RefreshCw } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";
import { Badge, Button, SegmentedControl } from "@/components/ui/Primitives";
import { useExtraRunDetails, useRunDetail } from "@/hooks/useStudioData";
import type { EvalHead, MetricPoint, RunDetail, RunSummary } from "@/lib/types";

const SERIES_COLORS = ["#4f46e5", "#15803d", "#b45309", "#7c3aed", "#0284c7", "#be123c"];

const HEAD_OPTIONS: ReadonlyArray<{ value: EvalHead; label: string }> = [
  { value: "live", label: "Live Win%" },
  { value: "ema", label: "EMA Win%" },
  { value: "ensemble", label: "Ensemble" },
];

type SortKey = "updated" | "update" | "live" | "ema" | "ensemble" | "tps" | "games";

const SORT_COLUMNS: ReadonlyArray<{ key: SortKey; label: string; padEnd?: boolean }> = [
  { key: "update", label: "Update" },
  { key: "live", label: "Live" },
  { key: "ema", label: "EMA" },
  { key: "ensemble", label: "Ens" },
  { key: "tps", label: "TPS" },
  { key: "games", label: "Games", padEnd: true },
];

function metricForHead(point: MetricPoint, head: EvalHead): number | null {
  if (head === "live") return point.live_held_out_det_win_rate ?? null;
  if (head === "ensemble") return point.ensemble_held_out_det_win_rate ?? null;
  return point.held_out_det_win_rate ?? null;
}

function formatPct(value: number | null): string {
  if (value === null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function statusTone(status: string): "win" | "accent" | "neutral" {
  if (status === "running") return "win";
  if (status === "completed") return "accent";
  return "neutral";
}

function sortValue(run: RunSummary, key: SortKey): number {
  if (key === "update") return run.latest_update;
  if (key === "live") return run.latest_live_win ?? -1;
  if (key === "ema") return run.latest_ema_win ?? -1;
  if (key === "ensemble") return run.latest_ensemble_win ?? -1;
  if (key === "tps") return run.latest_tps;
  if (key === "games") return run.eval_files;
  return run.updated_at;
}

interface CurveSeries {
  runId: string;
  label: string;
  color: string;
  points: Array<{ x: number; y: number; key: string }>;
}

function buildCurveSeries(
  runs: RunSummary[],
  details: Record<string, RunDetail>,
  head: EvalHead,
): { series: CurveSeries[]; maxUpdate: number } {
  const series: CurveSeries[] = [];
  let maxUpdate = 100;
  runs.slice(0, 6).forEach((run, idx) => {
    const detail = details[run.run_id];
    if (!detail) return;
    const pts: Array<{ x: number; y: number; key: string }> = [];
    const seenX = new Map<number, number>();
    for (const pt of detail.eval_points) {
      const val = metricForHead(pt, head);
      if (val !== null) {
        const occ = (seenX.get(pt.update) ?? 0) + 1;
        seenX.set(pt.update, occ);
        pts.push({ x: pt.update, y: val, key: `${run.run_id}-${pt.update}#${occ}` });
        if (pt.update > maxUpdate) maxUpdate = pt.update;
      }
    }
    if (pts.length > 0) {
      series.push({
        runId: run.run_id,
        label: run.name,
        color: SERIES_COLORS[idx % SERIES_COLORS.length] ?? "#4f46e5",
        points: pts,
      });
    }
  });
  return { series, maxUpdate };
}

function EvalCurvesChart(props: {
  runs: RunSummary[];
  details: Record<string, RunDetail>;
  head: EvalHead;
  hiddenRuns: Set<string>;
  onToggleRun: (runId: string) => void;
}): ReactNode {
  const { series, maxUpdate } = buildCurveSeries(props.runs, props.details, props.head);
  const w = 780;
  const h = 192;
  const padL = 44;
  const padR = 16;
  const padT = 14;
  const padB = 24;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const toX = (u: number): number => padL + (u / Math.max(1, maxUpdate)) * plotW;
  const toY = (v: number): number => padT + (1 - Math.max(0, Math.min(1, v))) * plotH;

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-3)]">
          Held-out Eval Win Rate vs. Update ({props.head.toUpperCase()})
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {series.map((s) => {
            const hidden = props.hiddenRuns.has(s.runId);
            return (
              <button
                key={s.runId}
                type="button"
                onClick={(): void => props.onToggleRun(s.runId)}
                className="inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-[11px] border border-[var(--border)] cursor-pointer transition-opacity"
                style={{ opacity: hidden ? 0.4 : 1 }}
              >
                <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
                <span className="max-w-[140px] truncate">{s.label}</span>
              </button>
            );
          })}
        </div>
      </div>
      {series.length === 0 ? (
        <div className="h-[140px] flex items-center justify-center text-xs text-[var(--text-3)]">
          No {props.head.toUpperCase()} eval points recorded for the currently displayed runs.
        </div>
      ) : (
        <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[180px] block">
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const y = toY(tick);
            return (
              <g key={tick}>
                <line
                  x1={padL}
                  y1={y}
                  x2={w - padR}
                  y2={y}
                  stroke="var(--border)"
                  strokeDasharray={tick === 0.5 ? "4 3" : undefined}
                />
                <text
                  x={padL - 8}
                  y={y + 4}
                  textAnchor="end"
                  fontSize="11"
                  fill="var(--text-3)"
                  className="tabular"
                >
                  {Math.round(tick * 100)}%
                </text>
              </g>
            );
          })}
          {series
            .filter((s) => !props.hiddenRuns.has(s.runId))
            .map((s) => {
              const polyline = s.points
                .map((p) => `${toX(p.x).toFixed(1)},${toY(p.y).toFixed(1)}`)
                .join(" ");
              return (
                <g key={s.runId}>
                  <polyline
                    points={polyline}
                    fill="none"
                    stroke={s.color}
                    strokeWidth="2.2"
                    strokeLinejoin="round"
                  />
                  {s.points.map((p) => (
                    <circle
                      key={p.key}
                      cx={toX(p.x)}
                      cy={toY(p.y)}
                      r="3.4"
                      fill={s.color}
                      stroke="var(--surface)"
                      strokeWidth="1.2"
                    >
                      <title>{`${s.label} · u${p.x}: ${(p.y * 100).toFixed(1)}%`}</title>
                    </circle>
                  ))}
                </g>
              );
            })}
          {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
            const u = Math.round(maxUpdate * frac);
            return (
              <text
                key={frac}
                x={toX(u)}
                y={h - 5}
                textAnchor="middle"
                fontSize="11"
                fill="var(--text-3)"
                className="tabular"
              >
                u{u}
              </text>
            );
          })}
        </svg>
      )}
    </div>
  );
}

function formatSparkValue(value: number, asPct: boolean, unit: string): string {
  if (asPct) return `${(value * 100).toFixed(1)}%`;
  return `${value.toFixed(value >= 10 ? 0 : 3)}${unit}`;
}

function MiniSparkline(props: {
  label: string;
  values: number[];
  color: string;
  unit?: string;
  asPct?: boolean;
}): ReactNode {
  const vals = props.values;
  const last = vals.length > 0 ? vals[vals.length - 1] : undefined;
  const min = vals.length > 0 ? Math.min(...vals) : 0;
  const max = vals.length > 0 ? Math.max(...vals) : 1;
  const span = Math.max(1e-6, max - min);
  const points = vals
    .map((v, i) => {
      const x = vals.length > 1 ? (i / (vals.length - 1)) * 220 : 110;
      const y = 34 - ((v - min) / span) * 28;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const asPct = Boolean(props.asPct);
  const unit = props.unit ?? "";

  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-2">
      <div className="flex items-center justify-between text-[11px] mb-1">
        <span className="text-[var(--text-2)]">{props.label}</span>
        <span className="tabular font-semibold">
          {last === undefined ? "—" : formatSparkValue(last, asPct, unit)}
        </span>
      </div>
      <svg viewBox="0 0 220 38" className="w-full h-9 block">
        {vals.length > 1 ? (
          <polyline points={points} fill="none" stroke={props.color} strokeWidth="1.8" />
        ) : (
          <line x1="0" y1="19" x2="220" y2="19" stroke="var(--border)" />
        )}
      </svg>
      {vals.length > 1 ? (
        <div className="flex justify-between text-[9.5px] text-[var(--text-3)] tabular mt-0.5">
          <span>min {formatSparkValue(min, asPct, unit)}</span>
          <span>max {formatSparkValue(max, asPct, unit)}</span>
        </div>
      ) : null}
    </div>
  );
}

function RunConfigSummary(props: { config: Record<string, unknown> }): ReactNode {
  const entries = Object.entries(props.config).filter(
    ([, v]) => typeof v === "string" || typeof v === "number" || typeof v === "boolean",
  );
  if (entries.length === 0) return null;
  return (
    <details className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-2">
      <summary className="text-[10.5px] font-semibold uppercase tracking-wider text-[var(--text-3)] cursor-pointer">
        Hyperparameters &amp; Config ({entries.length})
      </summary>
      <div className="mt-2 grid grid-cols-2 gap-x-2 gap-y-1 text-[10.5px]">
        {entries.slice(0, 24).map(([key, val]) => (
          <div key={key} className="truncate" title={`${key}: ${String(val)}`}>
            <span className="text-[var(--text-3)]">{key}: </span>
            <span className="tabular font-medium">{String(val)}</span>
          </div>
        ))}
      </div>
    </details>
  );
}

function numericSeries(
  points: MetricPoint[] | undefined,
  pick: (pt: MetricPoint) => number | undefined,
): number[] {
  if (!points) return [];
  return points.map(pick).filter((v): v is number => typeof v === "number");
}

function RunSparklines(props: { detail: RunDetail | null }): ReactNode {
  const vloss = numericSeries(props.detail?.train_points, (p) => p.value_loss);
  const entropy = numericSeries(props.detail?.train_points, (p) => p.entropy);
  const tps = numericSeries(props.detail?.train_points, (p) => p.tps);
  const missedLand = numericSeries(props.detail?.eval_points, (p) => p.missed_land_drop_rate);
  return (
    <div className="flex flex-col gap-2">
      <MiniSparkline label="Value Loss" values={vloss} color="var(--accent)" />
      <MiniSparkline label="Policy Entropy" values={entropy} color="var(--win)" />
      <MiniSparkline label="Throughput" values={tps} color="var(--warn)" unit=" TPS" />
      {missedLand.length > 0 ? (
        <MiniSparkline label="Missed Land Drop" values={missedLand} color="var(--loss)" asPct />
      ) : null}
    </div>
  );
}

function RunDiagnosticsDrawer(props: {
  run: RunSummary | null;
  detail: RunDetail | null;
  onBrowseGames: (runId: string) => void;
}): ReactNode {
  const run = props.run;
  if (!run) {
    return (
      <aside
        aria-label="Run diagnostics"
        className="w-full lg:w-[320px] shrink-0 border-t lg:border-t-0 lg:border-l border-[var(--border)] bg-[var(--surface)] p-4 text-xs text-[var(--text-3)]"
      >
        Select a training run to inspect its diagnostics and checkpoints.
      </aside>
    );
  }
  const modelLabel =
    run.hidden_dim && run.attention_layers
      ? `w${run.hidden_dim}l${run.attention_layers}`
      : "Custom config";
  const updateRange = run.total_updates > 0 ? ` / ${run.total_updates}` : "";

  return (
    <aside
      aria-label="Run diagnostics"
      className="w-full lg:w-[320px] shrink-0 border-t lg:border-t-0 lg:border-l border-[var(--border)] bg-[var(--surface)] p-3.5 overflow-y-auto flex flex-col gap-3"
    >
      <div>
        <div className="flex items-center justify-between gap-2 mb-1">
          <Badge tone={statusTone(run.status)}>{run.status}</Badge>
          <span className="tabular text-[11px] text-[var(--text-3)]">
            u{run.latest_update}
            {updateRange}
          </span>
        </div>
        <h3 className="text-sm font-semibold tracking-tight break-all">{run.run_id}</h3>
        <div className="text-[11px] text-[var(--text-2)] mt-0.5 tabular">
          {modelLabel} · {Math.round(run.latest_tps)} TPS · {run.checkpoints.length} ckpts
        </div>
      </div>

      <div className="grid grid-cols-3 gap-1.5">
        <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-2">
          <div className="text-[10px] text-[var(--text-3)] uppercase">Live</div>
          <div className="tabular text-sm font-bold text-[var(--win)]">
            {formatPct(run.latest_live_win)}
          </div>
        </div>
        <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-2">
          <div className="text-[10px] text-[var(--text-3)] uppercase">EMA</div>
          <div className="tabular text-sm font-bold">{formatPct(run.latest_ema_win)}</div>
        </div>
        <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-2">
          <div className="text-[10px] text-[var(--text-3)] uppercase">Ensemble</div>
          <div className="tabular text-sm font-bold text-[var(--accent)]">
            {formatPct(run.latest_ensemble_win)}
          </div>
        </div>
      </div>

      <RunSparklines detail={props.detail} />
      {props.detail ? <RunConfigSummary config={props.detail.config} /> : null}

      <div>
        <div className="text-[10.5px] font-semibold uppercase tracking-wider text-[var(--text-3)] mb-1.5">
          Recorded Games ({run.eval_files} eval snapshots)
        </div>
        <Button
          variant="primary"
          size="md"
          className="w-full"
          disabled={run.eval_files === 0}
          onClick={(): void => props.onBrowseGames(run.run_id)}
        >
          Browse Games for This Run
          <ArrowRight className="w-3.5 h-3.5" />
        </Button>
      </div>
    </aside>
  );
}

function RunsTableCard(props: {
  runs: RunSummary[];
  loading: boolean;
  activeRunId: string | null;
  sortKey: SortKey;
  sortDir: "desc" | "asc";
  onSort: (key: SortKey) => void;
  onSelect: (runId: string) => void;
}): ReactNode {
  const arrow = props.sortDir === "desc" ? " ↓" : " ↑";
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-x-auto">
      <table className="w-full min-w-[600px] border-collapse text-left text-xs">
        <thead>
          <tr className="border-b border-[var(--border)] bg-[var(--surface-2)] text-[10.5px] uppercase tracking-wider text-[var(--text-3)]">
            <th className="py-2 px-2.5 font-semibold">Run</th>
            <th className="py-2 px-1.5 font-semibold">Status</th>
            <th className="py-2 px-1.5 font-semibold">Model</th>
            {SORT_COLUMNS.map((col) => (
              <th
                key={col.key}
                className={`py-2 ${col.padEnd ? "px-2.5" : "px-1.5"} font-semibold text-right`}
              >
                <button
                  type="button"
                  onClick={(): void => props.onSort(col.key)}
                  className="cursor-pointer hover:text-[var(--text)] whitespace-nowrap"
                >
                  {col.label}
                  {props.sortKey === col.key ? arrow : ""}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {props.loading && props.runs.length === 0 ? (
            <tr>
              <td colSpan={9} className="py-8 text-center text-[var(--text-3)]">
                Scanning jace-runs/…
              </td>
            </tr>
          ) : (
            props.runs.map((run) => (
              <tr
                key={run.run_id}
                onClick={(): void => props.onSelect(run.run_id)}
                className="border-b border-[var(--border)] last:border-b-0 cursor-pointer transition-colors hover:bg-[var(--surface-2)]"
                style={
                  props.activeRunId === run.run_id
                    ? { background: "var(--accent-soft)" }
                    : undefined
                }
              >
                <td className="py-2 px-2.5 font-medium max-w-[185px] truncate" title={run.run_id}>
                  {run.run_id}
                </td>
                <td className="py-2 px-1.5">
                  <Badge tone={statusTone(run.status)}>{run.status}</Badge>
                </td>
                <td className="py-2 px-1.5 tabular text-[var(--text-2)]">
                  {run.hidden_dim && run.attention_layers
                    ? `w${run.hidden_dim}l${run.attention_layers}`
                    : "—"}
                </td>
                <td className="py-2 px-1.5 tabular text-right">
                  {run.latest_update}
                  {run.total_updates > 0 ? `/${run.total_updates}` : ""}
                </td>
                <td className="py-2 px-1.5 tabular text-right font-semibold text-[var(--win)]">
                  {formatPct(run.latest_live_win)}
                </td>
                <td className="py-2 px-1.5 tabular text-right">{formatPct(run.latest_ema_win)}</td>
                <td className="py-2 px-1.5 tabular text-right text-[var(--accent)]">
                  {formatPct(run.latest_ensemble_win)}
                </td>
                <td className="py-2 px-1.5 tabular text-right text-[var(--text-2)]">
                  {Math.round(run.latest_tps)}
                </td>
                <td className="py-2 px-2.5 tabular text-right text-[var(--text-2)]">
                  {run.eval_files}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export function RunsView(props: {
  runs: RunSummary[];
  details: Record<string, RunDetail>;
  loading: boolean;
  onReload: () => void;
  onBrowseGames: (runId: string) => void;
}): ReactNode {
  const [head, setHead] = useState<EvalHead>("live");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filterText, setFilterText] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("updated");
  const [sortDir, setSortDir] = useState<"desc" | "asc">("desc");
  const [hiddenRuns, setHiddenRuns] = useState<Set<string>>(() => new Set());

  const handleSort = (key: SortKey): void => {
    if (key === sortKey) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const filtered = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    const list = props.runs.filter((run) => q === "" || run.run_id.toLowerCase().includes(q));
    const dir = sortDir === "desc" ? 1 : -1;
    return list.toSorted((a, b) => (sortValue(b, sortKey) - sortValue(a, sortKey)) * dir);
  }, [props.runs, filterText, sortKey, sortDir]);

  const activeRun = filtered.find((r) => r.run_id === selectedId) ?? filtered[0] ?? null;

  const chartCandidates = useMemo(() => {
    const withEval = filtered.filter(
      (r) => r.latest_live_win !== null || r.latest_ema_win !== null,
    );
    const base = (withEval.length > 0 ? withEval : filtered).slice(0, 6);
    if (activeRun && !base.some((r) => r.run_id === activeRun.run_id)) {
      return [activeRun, ...base.slice(0, 5)];
    }
    return base;
  }, [filtered, activeRun]);

  const allDetails = useExtraRunDetails(
    chartCandidates.map((r) => r.run_id),
    props.details,
  );
  const fetchedDetail = useRunDetail(activeRun?.run_id ?? null);
  const activeDetail = (activeRun ? allDetails[activeRun.run_id] : undefined) ?? fetchedDetail;

  const toggleSeries = (runId: string): void => {
    setHiddenRuns((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  };

  const countLabel =
    filterText.trim() === ""
      ? `${props.runs.length} runs`
      : `${filtered.length} of ${props.runs.length}`;

  return (
    <div className="flex flex-col lg:flex-row h-full min-h-0 overflow-y-auto lg:overflow-hidden">
      <div className="flex-none lg:flex-1 min-w-0 flex flex-col lg:min-h-0 lg:overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 border-b border-[var(--border)] bg-[var(--surface)]">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[var(--accent)]" />
            <h2 className="text-sm font-semibold tracking-tight">Training Runs</h2>
            <Badge>{countLabel}</Badge>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="search"
              value={filterText}
              onChange={(e): void => setFilterText(e.target.value)}
              placeholder="Filter runs..."
              aria-label="Filter runs"
              className="h-7 w-44 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2.5 text-xs"
            />
            <SegmentedControl options={HEAD_OPTIONS} value={head} onChange={setHead} />
            <Button onClick={props.onReload} title="Refresh runs from disk">
              <RefreshCw className="w-3.5 h-3.5" />
              Refresh
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
          <EvalCurvesChart
            runs={chartCandidates}
            details={allDetails}
            head={head}
            hiddenRuns={hiddenRuns}
            onToggleRun={toggleSeries}
          />
          <RunsTableCard
            runs={filtered}
            loading={props.loading}
            activeRunId={activeRun?.run_id ?? null}
            sortKey={sortKey}
            sortDir={sortDir}
            onSort={handleSort}
            onSelect={setSelectedId}
          />
        </div>
      </div>

      <RunDiagnosticsDrawer
        run={activeRun}
        detail={activeDetail}
        onBrowseGames={props.onBrowseGames}
      />
    </div>
  );
}

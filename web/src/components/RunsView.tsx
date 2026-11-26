/** Payload-driven Runs Overview: dynamic multi-run metric curves, sortable table, and auto-discovered diagnostics. */

import { Activity, ArrowRight, RefreshCw, Search } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";
import { Badge, Button } from "@/components/ui/Primitives";
import { useExtraRunDetails, useRunDetail } from "@/hooks/useStudioData";
import type { MetricPoint, RunDetail, RunSummary } from "@/lib/types";

const SERIES_COLORS = ["#4f46e5", "#15803d", "#b45309", "#7c3aed", "#0284c7", "#be123c"];
const SKIPPED_METRIC_KEYS = new Set(["update", "kind", "step", "epoch"]);
const PREFERRED_DIAG_ORDER = [
  "value_loss",
  "entropy",
  "tps",
  "missed_land_drop_rate",
  "wrong_side_target_rate",
  "policy_loss",
  "clip_fraction",
  "grad_norm",
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

function humanizeMetricKey(key: string): string {
  const known: Record<string, string> = {
    live_held_out_det_win_rate: "Live Held-Out Win%",
    held_out_det_win_rate: "EMA Held-Out Win%",
    ensemble_held_out_det_win_rate: "Ensemble Held-Out Win%",
    win_rate_vs_heuristic: "Win% vs Heuristic",
    missed_land_drop_rate: "Missed Land Drop%",
    wrong_side_target_rate: "Wrong-Side Target%",
    value_loss: "Value Loss",
    policy_loss: "Policy Loss",
    entropy: "Policy Entropy",
    tps: "Throughput (TPS)",
    clip_fraction: "Clip Fraction",
    grad_norm: "Grad Norm",
    grad_cosine: "Grad Cosine",
    collection_kl: "Collection KL",
    magnet_kl: "Magnet KL",
  };
  const mapped = known[key];
  if (mapped) return mapped;
  return key
    .split("_")
    .filter((p) => p.length > 0)
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join(" ");
}

function isRateMetric(key: string, values: number[]): boolean {
  if (/(rate|win|fraction|ratio|pct|percent|acc)/i.test(key)) {
    return values.every((v) => v >= 0 && v <= 1.001);
  }
  return false;
}

function numericFromPoint(pt: MetricPoint, key: string): number | null {
  const val = pt[key];
  return typeof val === "number" && Number.isFinite(val) ? val : null;
}

function discoverNumericKeys(points: MetricPoint[]): string[] {
  const keys = new Set<string>();
  for (const pt of points) {
    for (const [k, v] of Object.entries(pt)) {
      if (!SKIPPED_METRIC_KEYS.has(k) && typeof v === "number" && Number.isFinite(v)) {
        keys.add(k);
      }
    }
  }
  return [...keys].toSorted((a, b) => a.localeCompare(b));
}

function discoverEvalMetricKeys(details: Record<string, RunDetail>): string[] {
  const allPts: MetricPoint[] = [];
  for (const d of Object.values(details)) {
    allPts.push(...d.eval_points);
  }
  const keys = discoverNumericKeys(allPts);
  const priority = [
    "live_held_out_det_win_rate",
    "held_out_det_win_rate",
    "ensemble_held_out_det_win_rate",
    "win_rate_vs_heuristic",
    "missed_land_drop_rate",
  ];
  return keys.toSorted((a, b) => {
    const ia = priority.indexOf(a);
    const ib = priority.indexOf(b);
    if (ia !== -1 && ib !== -1) return ia - ib;
    if (ia !== -1) return -1;
    if (ib !== -1) return 1;
    return a.localeCompare(b);
  });
}

function formatPct(value: number | null): string {
  if (value === null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatMetricVal(value: number, asPct: boolean): string {
  if (asPct) return `${(value * 100).toFixed(1)}%`;
  const abs = Math.abs(value);
  if (abs > 0 && abs < 0.001) return value.toExponential(2);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(3);
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

function resolveAxisBounds(asPct: boolean, allVals: number[]): { minVal: number; maxVal: number } {
  if (asPct) return { minVal: 0, maxVal: 1 };
  if (allVals.length === 0) return { minVal: 0, maxVal: 1 };
  return { minVal: Math.min(...allVals), maxVal: Math.max(...allVals) };
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
  metricKey: string,
): { series: CurveSeries[]; maxUpdate: number; minVal: number; maxVal: number; asPct: boolean } {
  const series: CurveSeries[] = [];
  const allVals: number[] = [];
  let maxUpdate = 100;
  runs.slice(0, 6).forEach((run, idx) => {
    const detail = details[run.run_id];
    if (!detail) return;
    const pts: Array<{ x: number; y: number; key: string }> = [];
    const seenX = new Map<number, number>();
    for (const pt of detail.eval_points) {
      const val = numericFromPoint(pt, metricKey);
      if (val !== null) {
        const occ = (seenX.get(pt.update) ?? 0) + 1;
        seenX.set(pt.update, occ);
        pts.push({ x: pt.update, y: val, key: `${run.run_id}-${pt.update}#${occ}` });
        allVals.push(val);
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
  const asPct = isRateMetric(metricKey, allVals);
  const { minVal, maxVal } = resolveAxisBounds(asPct, allVals);
  return { series, maxUpdate, minVal, maxVal, asPct };
}

function EvalCurvesChart(props: {
  runs: RunSummary[];
  details: Record<string, RunDetail>;
  metricKey: string;
  hiddenRuns: Set<string>;
  onToggleRun: (runId: string) => void;
}): ReactNode {
  const { series, maxUpdate, minVal, maxVal, asPct } = buildCurveSeries(
    props.runs,
    props.details,
    props.metricKey,
  );
  const w = 780;
  const h = 192;
  const padL = 52;
  const padR = 16;
  const padT = 14;
  const padB = 24;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const span = Math.max(1e-6, maxVal - minVal);
  const toX = (u: number): number => padL + (u / Math.max(1, maxUpdate)) * plotW;
  const toY = (v: number): number => padT + (1 - (v - minVal) / span) * plotH;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((frac) => minVal + frac * span);

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-3)]">
          {humanizeMetricKey(props.metricKey)} vs. Update
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
          No points logged for {humanizeMetricKey(props.metricKey)} in the displayed runs.
        </div>
      ) : (
        <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[180px] block">
          {ticks.map((tick, idx) => {
            const y = toY(tick);
            return (
              <g key={`tick-${tick.toFixed(6)}`}>
                <line
                  x1={padL}
                  y1={y}
                  x2={w - padR}
                  y2={y}
                  stroke="var(--border)"
                  strokeDasharray={asPct && idx === 2 ? "4 3" : undefined}
                />
                <text
                  x={padL - 8}
                  y={y + 4}
                  textAnchor="end"
                  fontSize="10.5"
                  fill="var(--text-3)"
                  className="tabular"
                >
                  {formatMetricVal(tick, asPct)}
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
                      r="3.2"
                      fill={s.color}
                      stroke="var(--surface)"
                      strokeWidth="1.2"
                    >
                      <title>{`${s.label} · u${p.x}: ${formatMetricVal(p.y, asPct)}`}</title>
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

interface DiscoveredSeries {
  key: string;
  source: "eval" | "train";
  label: string;
  values: number[];
  asPct: boolean;
  color: string;
}

function discoverAllDiagnosticSeries(detail: RunDetail | null): DiscoveredSeries[] {
  if (!detail) return [];
  const evalKeys = discoverNumericKeys(detail.eval_points);
  const trainKeys = discoverNumericKeys(detail.train_points);
  const seen = new Set<string>();
  const out: DiscoveredSeries[] = [];

  const pushSeries = (key: string, source: "eval" | "train", points: MetricPoint[]): void => {
    const uid = `${source}:${key}`;
    if (seen.has(uid)) return;
    const vals = points
      .map((pt) => numericFromPoint(pt, key))
      .filter((v): v is number => v !== null);
    if (vals.length === 0) return;
    seen.add(uid);
    const colorIdx = out.length % SERIES_COLORS.length;
    out.push({
      key,
      source,
      label: humanizeMetricKey(key),
      values: vals,
      asPct: isRateMetric(key, vals),
      color: SERIES_COLORS[colorIdx] ?? "var(--accent)",
    });
  };

  for (const k of trainKeys) pushSeries(k, "train", detail.train_points);
  for (const k of evalKeys) pushSeries(k, "eval", detail.eval_points);

  return out.toSorted((a, b) => {
    const ia = PREFERRED_DIAG_ORDER.indexOf(a.key);
    const ib = PREFERRED_DIAG_ORDER.indexOf(b.key);
    if (ia !== -1 && ib !== -1) return ia - ib;
    if (ia !== -1) return -1;
    if (ib !== -1) return 1;
    if (a.source !== b.source) return a.source === "eval" ? -1 : 1;
    return a.key.localeCompare(b.key);
  });
}

function MiniSparkline(props: { series: DiscoveredSeries }): ReactNode {
  const { series } = props;
  const vals = series.values;
  const last = vals[vals.length - 1];
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = Math.max(1e-6, max - min);
  const points = vals
    .map((v, i) => {
      const x = vals.length > 1 ? (i / (vals.length - 1)) * 220 : 110;
      const y = 32 - ((v - min) / span) * 26;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-2">
      <div className="flex items-center justify-between gap-2 text-[11px] mb-1">
        <span className="text-[var(--text-2)] truncate" title={`${series.key} (${series.source})`}>
          {series.label}
        </span>
        <span className="tabular font-semibold shrink-0">
          {last === undefined ? "—" : formatMetricVal(last, series.asPct)}
        </span>
      </div>
      <svg viewBox="0 0 220 36" className="w-full h-8 block">
        {vals.length > 1 ? (
          <polyline points={points} fill="none" stroke={series.color} strokeWidth="1.8" />
        ) : (
          <line x1="0" y1="18" x2="220" y2="18" stroke="var(--border)" />
        )}
      </svg>
      {vals.length > 1 ? (
        <div className="flex justify-between text-[9.5px] text-[var(--text-3)] tabular mt-0.5">
          <span>min {formatMetricVal(min, series.asPct)}</span>
          <span>max {formatMetricVal(max, series.asPct)}</span>
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
        Config ({entries.length})
      </summary>
      <div className="mt-2 grid grid-cols-2 gap-x-2 gap-y-1 text-[10.5px]">
        {entries.map(([key, val]) => (
          <div key={key} className="truncate" title={`${key}: ${String(val)}`}>
            <span className="text-[var(--text-3)]">{key}: </span>
            <span className="tabular font-medium">{String(val)}</span>
          </div>
        ))}
      </div>
    </details>
  );
}

function DynamicDiagnosticsList(props: { detail: RunDetail | null }): ReactNode {
  const [filterText, setFilterText] = useState("");
  const [showAll, setShowAll] = useState(false);
  const allSeries = useMemo(() => discoverAllDiagnosticSeries(props.detail), [props.detail]);
  const needle = filterText.trim().toLowerCase();
  const filtered = needle
    ? allSeries.filter(
        (s) => s.key.toLowerCase().includes(needle) || s.label.toLowerCase().includes(needle),
      )
    : allSeries;
  const visible = showAll || needle !== "" ? filtered : filtered.slice(0, 4);

  if (allSeries.length === 0) {
    return <div className="text-[11px] text-[var(--text-3)]">No numeric telemetry logged yet.</div>;
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-3)]">
          Logged Metrics ({allSeries.length})
        </span>
        {allSeries.length > 4 ? (
          <button
            type="button"
            onClick={(): void => setShowAll((prev) => !prev)}
            className="text-[10.5px] font-medium text-[var(--accent)] hover:underline cursor-pointer"
          >
            {showAll ? "Show top 4" : `Show all ${allSeries.length}`}
          </button>
        ) : null}
      </div>
      {allSeries.length > 4 ? (
        <div className="relative">
          <Search className="w-3 h-3 absolute left-2 top-1/2 -translate-y-1/2 text-[var(--text-3)]" />
          <input
            type="search"
            value={filterText}
            onChange={(e): void => setFilterText(e.target.value)}
            placeholder="Filter any logged metric…"
            aria-label="Filter logged metrics"
            className="w-full h-6 rounded border border-[var(--border)] bg-[var(--bg)] pl-6 pr-2 text-[10.5px]"
          />
        </div>
      ) : null}
      {visible.map((s) => (
        <MiniSparkline key={`${s.source}:${s.key}`} series={s} />
      ))}
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
        Select a training run to inspect its logged metrics and checkpoints.
      </aside>
    );
  }
  const modelLabel =
    run.hidden_dim && run.attention_layers
      ? `w${run.hidden_dim}l${run.attention_layers}`
      : "Custom";
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

      <DynamicDiagnosticsList detail={props.detail} />
      {props.detail ? <RunConfigSummary config={props.detail.config} /> : null}

      {run.eval_files > 0 ? (
        <Button
          variant="primary"
          size="md"
          className="w-full"
          onClick={(): void => props.onBrowseGames(run.run_id)}
        >
          Browse Games ({run.eval_files} evals)
          <ArrowRight className="w-3.5 h-3.5" />
        </Button>
      ) : null}
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
          {props.runs.length === 0 ? (
            <tr>
              <td colSpan={9} className="py-8 text-center text-[var(--text-3)]">
                {props.loading ? "Scanning runs directory…" : "No runs match that filter."}
              </td>
            </tr>
          ) : (
            props.runs.map((run) => {
              const selected = run.run_id === props.activeRunId;
              const arch =
                run.hidden_dim && run.attention_layers
                  ? `w${run.hidden_dim}l${run.attention_layers}`
                  : "—";
              return (
                <tr
                  key={run.run_id}
                  onClick={(): void => props.onSelect(run.run_id)}
                  className="border-b border-[var(--border)] last:border-b-0 cursor-pointer hover:bg-[var(--surface-2)] transition-colors"
                  style={selected ? { background: "var(--accent-soft)" } : undefined}
                >
                  <td className="py-2 px-2.5">
                    <div className="font-semibold truncate max-w-[200px]">{run.name}</div>
                    <div className="text-[10.5px] text-[var(--text-3)] truncate max-w-[200px]">
                      {run.experiment}
                    </div>
                  </td>
                  <td className="py-2 px-1.5">
                    <Badge tone={statusTone(run.status)}>{run.status}</Badge>
                  </td>
                  <td className="py-2 px-1.5 tabular text-[var(--text-2)]">{arch}</td>
                  <td className="py-2 px-1.5 tabular text-right">
                    {run.latest_update}
                    {run.total_updates > 0 ? (
                      <span className="text-[var(--text-3)]">/{run.total_updates}</span>
                    ) : null}
                  </td>
                  <td className="py-2 px-1.5 tabular text-right font-semibold text-[var(--win)]">
                    {formatPct(run.latest_live_win)}
                  </td>
                  <td className="py-2 px-1.5 tabular text-right">
                    {formatPct(run.latest_ema_win)}
                  </td>
                  <td className="py-2 px-1.5 tabular text-right text-[var(--accent)] font-semibold">
                    {formatPct(run.latest_ensemble_win)}
                  </td>
                  <td className="py-2 px-1.5 tabular text-right text-[var(--text-2)]">
                    {run.latest_tps > 0 ? Math.round(run.latest_tps) : "—"}
                  </td>
                  <td className="py-2 px-2.5 tabular text-right text-[var(--text-2)]">
                    {run.eval_files}
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}

function RunsHeaderBar(props: {
  runCount: number;
  filterText: string;
  evalMetricKeys: string[];
  activeMetric: string;
  loading: boolean;
  onChangeFilter: (next: string) => void;
  onChangeMetric: (next: string) => void;
  onReload: () => void;
}): ReactNode {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 border-b border-[var(--border)] bg-[var(--surface)]">
      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4 text-[var(--accent)]" />
        <h2 className="text-sm font-semibold tracking-tight">Training Runs</h2>
        <Badge>{props.runCount}</Badge>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="search"
          value={props.filterText}
          onChange={(e): void => props.onChangeFilter(e.target.value)}
          placeholder="Filter runs…"
          aria-label="Filter runs"
          className="h-7 w-44 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2.5 text-xs"
        />
        {props.evalMetricKeys.length > 0 ? (
          <select
            value={props.activeMetric}
            onChange={(e): void => props.onChangeMetric(e.target.value)}
            aria-label="Chart metric"
            className="h-7 max-w-[200px] rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs font-medium"
          >
            {props.evalMetricKeys.map((key) => (
              <option key={key} value={key}>
                {humanizeMetricKey(key)}
              </option>
            ))}
          </select>
        ) : null}
        <Button onClick={props.onReload} disabled={props.loading}>
          <RefreshCw className={`w-3.5 h-3.5 ${props.loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>
    </div>
  );
}

function filterAndSortRuns(args: {
  runs: RunSummary[];
  filterText: string;
  sortKey: SortKey;
  sortDir: "desc" | "asc";
}): RunSummary[] {
  const q = args.filterText.trim().toLowerCase();
  const matched = q
    ? args.runs.filter(
        (r) =>
          r.run_id.toLowerCase().includes(q) ||
          r.experiment.toLowerCase().includes(q) ||
          r.status.toLowerCase().includes(q),
      )
    : args.runs;
  const sign = args.sortDir === "desc" ? -1 : 1;
  return matched.toSorted(
    (a, b) => (sortValue(a, args.sortKey) - sortValue(b, args.sortKey)) * sign,
  );
}

function resolveActiveMetric(selectedMetric: string | null, evalMetricKeys: string[]): string {
  if (selectedMetric && evalMetricKeys.includes(selectedMetric)) return selectedMetric;
  return evalMetricKeys[0] ?? "live_held_out_det_win_rate";
}

function resolveActiveRun(
  filteredRuns: RunSummary[],
  runs: RunSummary[],
  selectedRunId: string | null,
): RunSummary | null {
  return filteredRuns.find((r) => r.run_id === selectedRunId) ?? filteredRuns[0] ?? runs[0] ?? null;
}

export function RunsView(props: {
  runs: RunSummary[];
  details: Record<string, RunDetail>;
  loading: boolean;
  error: string | null;
  onReload: () => void;
  onBrowseGames: (runId: string) => void;
}): ReactNode {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [filterText, setFilterText] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("updated");
  const [sortDir, setSortDir] = useState<"desc" | "asc">("desc");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [hiddenRuns, setHiddenRuns] = useState<Set<string>>(() => new Set());

  const filteredRuns = useMemo(
    () => filterAndSortRuns({ runs: props.runs, filterText, sortKey, sortDir }),
    [props.runs, filterText, sortKey, sortDir],
  );
  const topChartIds = useMemo(() => filteredRuns.slice(0, 6).map((r) => r.run_id), [filteredRuns]);
  const allDetails = useExtraRunDetails(topChartIds, props.details);
  const evalMetricKeys = useMemo(() => discoverEvalMetricKeys(allDetails), [allDetails]);
  const activeMetric = resolveActiveMetric(selectedMetric, evalMetricKeys);
  const activeRun = resolveActiveRun(filteredRuns, props.runs, selectedRunId);
  const cachedActive = activeRun ? (allDetails[activeRun.run_id] ?? null) : null;
  const fetchedDetail = useRunDetail(activeRun && cachedActive === null ? activeRun.run_id : null);
  const activeDetail = cachedActive ?? fetchedDetail;

  const toggleSort = (nextKey: SortKey): void => {
    if (sortKey === nextKey) setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    else {
      setSortKey(nextKey);
      setSortDir("desc");
    }
  };

  const toggleRunVisibility = (runId: string): void => {
    setHiddenRuns((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  };

  return (
    <div className="flex flex-col lg:flex-row h-full min-h-0 overflow-hidden">
      <div className="flex-1 min-w-0 flex flex-col min-h-0 overflow-hidden">
        <RunsHeaderBar
          runCount={props.runs.length}
          filterText={filterText}
          evalMetricKeys={evalMetricKeys}
          activeMetric={activeMetric}
          loading={props.loading}
          onChangeFilter={setFilterText}
          onChangeMetric={setSelectedMetric}
          onReload={props.onReload}
        />

        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
          {props.error ? (
            <div className="rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] p-3 text-xs text-[var(--loss)]">
              {props.error}
            </div>
          ) : null}

          <EvalCurvesChart
            runs={filteredRuns}
            details={allDetails}
            metricKey={activeMetric}
            hiddenRuns={hiddenRuns}
            onToggleRun={toggleRunVisibility}
          />

          <RunsTableCard
            runs={filteredRuns}
            loading={props.loading}
            activeRunId={activeRun?.run_id ?? null}
            sortKey={sortKey}
            sortDir={sortDir}
            onSort={toggleSort}
            onSelect={setSelectedRunId}
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

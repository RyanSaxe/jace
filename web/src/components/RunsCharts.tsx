/** Zero-dependency SVG telemetry charts with synced crosshair, EMA smoothing, 95% CI bands, and dynamic metric discovery. */

import { X } from "lucide-react";
import { type MouseEvent, type ReactNode, useState } from "react";
import type { GlobalCheckpointOption, MetricPoint, RunDetail, RunSummary } from "@/lib/types";

export const RUN_PALETTE: readonly string[] = [
  "#4f46e5",
  "#b45309",
  "#059669",
  "#0284c7",
  "#7c3aed",
  "#dc2626",
  "#0d9488",
  "#db2777",
];

export type XAxisMode = "updates" | "progress";

export interface DiscoveredMetric {
  id: string;
  key: string;
  source: "eval" | "train";
  label: string;
}

export interface ChartSeriesInput {
  id: string;
  label: string;
  color: string;
  totalUpdates: number;
  points: MetricPoint[];
  dashed?: boolean;
}

interface PlotPoint {
  key: string;
  x: number;
  y: number;
  rawY: number;
  ciHalf: number;
  update: number;
}

interface PreparedSeries {
  id: string;
  label: string;
  color: string;
  dashed: boolean;
  points: PlotPoint[];
}

const SKIPPED_METRIC_KEYS = new Set(["update", "iteration", "kind", "step", "epoch", "timestamp"]);
const ACRONYMS = new Set([
  "kl",
  "ema",
  "tps",
  "lr",
  "ppo",
  "wdl",
  "tv",
  "ev",
  "iqm",
  "sps",
  "std",
  "se",
]);
const SAMPLE_COUNT_KEYS = ["eval_games", "num_games", "n_games", "games", "episodes", "n"];

export function runStatusMeta(status: string): {
  label: string;
  tone: "win" | "accent" | "neutral";
} {
  if (status === "running") return { label: "Live", tone: "win" };
  if (status === "completed") return { label: "Done", tone: "accent" };
  return { label: "Stopped", tone: "neutral" };
}

export function colorForIndex(index: number): string {
  const safeIdx = ((index % RUN_PALETTE.length) + RUN_PALETTE.length) % RUN_PALETTE.length;
  return RUN_PALETTE[safeIdx] ?? "#4f46e5";
}

function formatTokenWord(word: string): string {
  const lower = word.toLowerCase();
  if (ACRONYMS.has(lower)) return lower.toUpperCase();
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

export function humanizeMetricKey(key: string): string {
  const trimmed = key.trim();
  if (!trimmed) return "Metric";
  const normalized = trimmed.replaceAll("-", "_");
  const isWinSuffix = /_win_rate$/i.test(normalized);
  const base = isWinSuffix ? normalized.replace(/_win_rate$/i, "") : normalized;
  const words = base
    .split("_")
    .filter((part) => part.length > 0)
    .map(formatTokenWord);
  if (isWinSuffix) {
    return words.length > 0 ? `${words.join(" ")} Win%` : "Win%";
  }
  return words.join(" ");
}

export function isWinRateKey(key: string): boolean {
  return /(^|_)win(_rate|$)|win_rate/i.test(key);
}

export function isRateMetric(key: string, values: number[]): boolean {
  if (!/(rate|win|fraction|ratio|pct|percent|acc)/i.test(key)) return false;
  if (values.length === 0) return true;
  return values.every((v) => v >= 0 && v <= 1.001);
}

export function numericFromPoint(pt: MetricPoint, key: string): number | null {
  const val = pt[key];
  return typeof val === "number" && Number.isFinite(val) ? val : null;
}

export function formatMetricValue(value: number, asPct: boolean): string {
  if (asPct) return `${(value * 100).toFixed(1)}%`;
  const abs = Math.abs(value);
  if (abs > 0 && abs < 0.001) return value.toExponential(2);
  if (abs >= 1000) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(3);
}

export function formatPctOrDash(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function discoverNumericKeys(points: MetricPoint[]): string[] {
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

function metricPriorityScore(metric: DiscoveredMetric): number {
  const k = metric.key.toLowerCase();
  if (metric.source === "eval") {
    if (isWinRateKey(k)) return 10;
    if (/(rate|pct|percent|acc)/i.test(k)) return 20;
    return 65;
  }
  if (/loss/i.test(k)) return 30;
  if (/entropy/i.test(k)) return 40;
  if (/(clip|kl)/i.test(k)) return 50;
  if (/(tps|throughput|sps|grad)/i.test(k)) return 60;
  return 80;
}

function addMetricKeys(args: {
  target: Map<string, DiscoveredMetric>;
  source: "eval" | "train";
  keys: ReadonlyArray<string> | undefined;
}): void {
  if (!args.keys) return;
  for (const key of args.keys) {
    if (SKIPPED_METRIC_KEYS.has(key)) continue;
    const id = `${args.source}:${key}`;
    if (!args.target.has(id)) {
      args.target.set(id, { id, key, source: args.source, label: humanizeMetricKey(key) });
    }
  }
}

export function discoverCatalogMetrics(args: {
  evalMetricKeys: string[];
  trainMetricKeys: string[];
  details: Record<string, RunDetail>;
  runs: RunSummary[];
}): DiscoveredMetric[] {
  const map = new Map<string, DiscoveredMetric>();
  addMetricKeys({ target: map, source: "eval", keys: args.evalMetricKeys });
  addMetricKeys({ target: map, source: "train", keys: args.trainMetricKeys });
  for (const run of args.runs) {
    addMetricKeys({ target: map, source: "eval", keys: run.eval_metric_keys });
    addMetricKeys({ target: map, source: "train", keys: run.train_metric_keys });
  }
  for (const detail of Object.values(args.details)) {
    addMetricKeys({ target: map, source: "eval", keys: detail.eval_metric_keys });
    addMetricKeys({ target: map, source: "train", keys: detail.train_metric_keys });
    addMetricKeys({ target: map, source: "eval", keys: discoverNumericKeys(detail.eval_points) });
    addMetricKeys({
      target: map,
      source: "train",
      keys: discoverNumericKeys(detail.train_points),
    });
  }
  return [...map.values()].toSorted((a, b) => {
    const diff = metricPriorityScore(a) - metricPriorityScore(b);
    return diff === 0 ? a.key.localeCompare(b.key) : diff;
  });
}

function metricBucket(metric: DiscoveredMetric): string {
  const k = metric.key.toLowerCase();
  if (metric.source === "eval") {
    if (isWinRateKey(k)) return "eval_win";
    if (/missed_land|land|mull|error|blunder/i.test(k)) return "eval_forensics";
    return "eval_other";
  }
  if (/entropy/i.test(k)) return "train_entropy";
  if (/clip/i.test(k)) return "train_clip";
  if (/kl/i.test(k)) return "train_kl";
  if (/value_loss|v_loss|loss/i.test(k)) return "train_loss";
  return "train_other";
}

const DEFAULT_BUCKET_ORDER = [
  "eval_win",
  "eval_forensics",
  "train_entropy",
  "train_clip",
  "train_kl",
  "train_loss",
  "eval_other",
  "train_other",
] as const;

export function discoverPopulatedMetricIds(
  runIds: string[],
  details: Record<string, RunDetail>,
): Set<string> {
  const set = new Set<string>();
  for (const id of runIds) {
    const d = details[id];
    if (!d) continue;
    for (const k of discoverNumericKeys(d.eval_points)) set.add(`eval:${k}`);
    for (const k of discoverNumericKeys(d.train_points)) set.add(`train:${k}`);
  }
  return set;
}

export function selectDefaultMetricIds(
  metrics: DiscoveredMetric[],
  maxCount = 6,
  populatedIds?: ReadonlySet<string>,
): string[] {
  const ordered =
    populatedIds && populatedIds.size > 0
      ? [
          ...metrics.filter((m) => populatedIds.has(m.id)),
          ...metrics.filter((m) => !populatedIds.has(m.id)),
        ]
      : metrics;
  const firstByBucket = new Map<string, string>();
  for (const m of ordered) {
    const b = metricBucket(m);
    if (!firstByBucket.has(b)) firstByBucket.set(b, m.id);
  }
  const picked = DEFAULT_BUCKET_ORDER.flatMap((b) => {
    const id = firstByBucket.get(b);
    return id ? [id] : [];
  });
  const extras = ordered.map((m) => m.id).filter((id) => !picked.includes(id));
  return [...picked, ...extras].slice(0, maxCount);
}

export function formatCheckpointRef(runId: string, checkpointName: string): string {
  return `${runId}::${checkpointName}`;
}

export function latestCheckpointRefForRun(
  run: RunSummary,
  detail?: RunDetail,
  checkpoints?: GlobalCheckpointOption[],
): string | null {
  if (detail && detail.checkpoints.length > 0) {
    const latest = detail.checkpoints[detail.checkpoints.length - 1];
    if (latest) return formatCheckpointRef(run.run_id, latest.name);
  }
  const fromGlobal = checkpoints?.filter((c) => c.run_id === run.run_id) ?? [];
  if (fromGlobal.length > 0) {
    const latest = fromGlobal[fromGlobal.length - 1];
    if (latest) return latest.ref;
  }
  const positive = run.checkpoints.filter((iter) => iter >= 0);
  if (positive.length > 0) {
    const padded = String(Math.max(...positive)).padStart(4, "0");
    return formatCheckpointRef(run.run_id, `iter-${padded}`);
  }
  return null;
}

function estimateSampleCount(pt: MetricPoint): number | null {
  for (const key of SAMPLE_COUNT_KEYS) {
    const n = numericFromPoint(pt, key);
    if (n !== null && n > 1) return n;
  }
  return null;
}

function computeCiHalfWidth(pt: MetricPoint, key: string, val: number): number {
  const ciBase = key.replace(/_win_rate$/i, "");
  const ciLow = numericFromPoint(pt, `${ciBase}_ci_low`) ?? numericFromPoint(pt, `${key}_ci_low`);
  const ciHigh =
    numericFromPoint(pt, `${ciBase}_ci_high`) ?? numericFromPoint(pt, `${key}_ci_high`);
  if (ciLow !== null && ciHigh !== null && ciHigh >= ciLow) return (ciHigh - ciLow) / 2;
  const se = numericFromPoint(pt, `${key}_se`) ?? numericFromPoint(pt, `${key}_std`);
  if (se !== null && se >= 0) return 1.96 * se;
  const n = estimateSampleCount(pt);
  if (n === null) return 0;
  const clamped = Math.max(0, Math.min(1, val));
  return 1.96 * Math.sqrt((clamped * (1 - clamped)) / n);
}

function resolveXCoord(update: number, totalUpdates: number, xMode: XAxisMode): number {
  if (xMode === "progress") {
    const denom = totalUpdates > 0 ? totalUpdates : 100;
    return Math.max(0, Math.min(100, (update / denom) * 100));
  }
  return update;
}

function applyEmaToPoints(rawPts: PlotPoint[], smoothing: number): PlotPoint[] {
  if (rawPts.length <= 1 || smoothing <= 0) return rawPts;
  const alpha = Math.max(0, Math.min(0.92, smoothing));
  const smoothed: PlotPoint[] = [];
  let ema = rawPts[0]?.y ?? 0;
  for (let i = 0; i < rawPts.length; i += 1) {
    const cur = rawPts[i];
    if (!cur) continue;
    ema = i === 0 ? cur.rawY : alpha * ema + (1 - alpha) * cur.rawY;
    smoothed.push({ ...cur, y: ema });
  }
  return smoothed;
}

function buildPreparedPoints(args: {
  input: ChartSeriesInput;
  metricKey: string;
  xMode: XAxisMode;
  smoothing: number;
}): PlotPoint[] {
  const rawPts: PlotPoint[] = [];
  const seen = new Map<number, number>();
  for (const pt of args.input.points) {
    const val = numericFromPoint(pt, args.metricKey);
    if (val === null) continue;
    const occ = (seen.get(pt.update) ?? 0) + 1;
    seen.set(pt.update, occ);
    rawPts.push({
      key: `${pt.update}#${occ}`,
      x: resolveXCoord(pt.update, args.input.totalUpdates, args.xMode),
      y: val,
      rawY: val,
      ciHalf: computeCiHalfWidth(pt, args.metricKey, val),
      update: pt.update,
    });
  }
  return applyEmaToPoints(rawPts, args.smoothing);
}

function prepareAllSeries(args: {
  series: ChartSeriesInput[];
  metricKey: string;
  xMode: XAxisMode;
  smoothing: number;
}): PreparedSeries[] {
  const out: PreparedSeries[] = [];
  for (const item of args.series) {
    const pts = buildPreparedPoints({
      input: item,
      metricKey: args.metricKey,
      xMode: args.xMode,
      smoothing: args.smoothing,
    });
    if (pts.length > 0) {
      out.push({
        id: item.id,
        label: item.label,
        color: item.color,
        dashed: item.dashed ?? false,
        points: pts,
      });
    }
  }
  return out;
}

interface AxisBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  asPct: boolean;
}

function computeAxisBounds(args: {
  prepared: PreparedSeries[];
  metricKey: string;
  xMode: XAxisMode;
}): AxisBounds {
  const allVals: number[] = [];
  let maxX = args.xMode === "progress" ? 100 : 10;
  for (const s of args.prepared) {
    for (const p of s.points) {
      allVals.push(p.y);
      if (p.x > maxX) maxX = p.x;
    }
  }
  const asPct = isRateMetric(args.metricKey, allVals);
  if (allVals.length === 0) return { minX: 0, maxX, minY: 0, maxY: 1, asPct };
  const rawMin = Math.min(...allVals);
  const rawMax = Math.max(...allVals);
  if (asPct) {
    const pad = Math.max(0.05, (rawMax - rawMin) * 0.12);
    return {
      minX: 0,
      maxX,
      minY: Math.max(0, Math.floor((rawMin - pad) * 20) / 20),
      maxY: Math.min(1, Math.ceil((rawMax + pad) * 20) / 20),
      asPct,
    };
  }
  const pad = Math.max(1e-6, rawMax - rawMin) * 0.1;
  return { minX: 0, maxX, minY: rawMin - pad, maxY: rawMax + pad, asPct };
}

function nearestPoint(points: PlotPoint[], targetX: number): PlotPoint | null {
  let best: PlotPoint | null = null;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const pt of points) {
    const dist = Math.abs(pt.x - targetX);
    if (dist < bestDist) {
      bestDist = dist;
      best = pt;
    }
  }
  return best;
}

function formatXTick(value: number, xMode: XAxisMode): string {
  if (xMode === "progress") return `${Math.round(value)}%`;
  if (value >= 1000) return `u${(value / 1000).toFixed(1)}k`;
  return `u${Math.round(value)}`;
}

interface ChartGeom {
  w: number;
  h: number;
  padL: number;
  padR: number;
  padT: number;
  padB: number;
  toX: (x: number) => number;
  toY: (y: number) => number;
}

function buildChartGeom(args: { w: number; h: number; bounds: AxisBounds }): ChartGeom {
  const padL = 46;
  const padR = 12;
  const padT = 10;
  const padB = 22;
  const plotW = args.w - padL - padR;
  const plotH = args.h - padT - padB;
  const xSpan = Math.max(1e-6, args.bounds.maxX - args.bounds.minX);
  const ySpan = Math.max(1e-6, args.bounds.maxY - args.bounds.minY);
  return {
    w: args.w,
    h: args.h,
    padL,
    padR,
    padT,
    padB,
    toX: (x: number): number => padL + ((x - args.bounds.minX) / xSpan) * plotW,
    toY: (y: number): number => padT + (1 - (y - args.bounds.minY) / ySpan) * plotH,
  };
}

function ChartAxesLayer(props: {
  geom: ChartGeom;
  bounds: AxisBounds;
  xMode: XAxisMode;
}): ReactNode {
  const { geom, bounds, xMode } = props;
  const yFracs = [0, 0.333, 0.667, 1];
  const xFracs = [0, 0.25, 0.5, 0.75, 1];
  const showFiftyLine = bounds.asPct && bounds.minY < 0.5 && bounds.maxY > 0.5;

  return (
    <g>
      {yFracs.map((frac) => {
        const tick = bounds.minY + frac * (bounds.maxY - bounds.minY);
        const y = geom.toY(tick);
        return (
          <g key={`yt-${frac}`}>
            <line
              x1={geom.padL}
              y1={y}
              x2={geom.w - geom.padR}
              y2={y}
              stroke="var(--border)"
              strokeWidth="1"
            />
            <text
              x={geom.padL - 6}
              y={y + 3.5}
              textAnchor="end"
              fontSize="9.5"
              fill="var(--text-3)"
              className="tabular"
            >
              {formatMetricValue(tick, bounds.asPct)}
            </text>
          </g>
        );
      })}
      {showFiftyLine ? (
        <line
          x1={geom.padL}
          y1={geom.toY(0.5)}
          x2={geom.w - geom.padR}
          y2={geom.toY(0.5)}
          stroke="var(--border-strong)"
          strokeDasharray="3 3"
          strokeWidth="1"
        />
      ) : null}
      {xFracs.map((frac) => {
        const tick = bounds.minX + frac * (bounds.maxX - bounds.minX);
        return (
          <text
            key={`xt-${frac}`}
            x={geom.toX(tick)}
            y={geom.h - 5}
            textAnchor="middle"
            fontSize="9.5"
            fill="var(--text-3)"
            className="tabular"
          >
            {formatXTick(tick, xMode)}
          </text>
        );
      })}
    </g>
  );
}

function buildCiPolygonPoints(series: PreparedSeries, geom: ChartGeom): string {
  const upper = series.points.map(
    (p) => `${geom.toX(p.x).toFixed(1)},${geom.toY(Math.min(1, p.y + p.ciHalf)).toFixed(1)}`,
  );
  const lower = series.points
    .toReversed()
    .map((p) => `${geom.toX(p.x).toFixed(1)},${geom.toY(Math.max(0, p.y - p.ciHalf)).toFixed(1)}`);
  return [...upper, ...lower].join(" ");
}

function ChartSeriesLayer(props: {
  series: PreparedSeries[];
  geom: ChartGeom;
  showCi: boolean;
  asPct: boolean;
  smoothed: boolean;
}): ReactNode {
  const { geom, showCi, asPct, smoothed } = props;
  return (
    <g>
      {props.series.map((s) => {
        const linePts = s.points
          .map((p) => `${geom.toX(p.x).toFixed(1)},${geom.toY(p.y).toFixed(1)}`)
          .join(" ");
        const rawLinePts = smoothed
          ? s.points
              .map((p) => `${geom.toX(p.x).toFixed(1)},${geom.toY(p.rawY).toFixed(1)}`)
              .join(" ")
          : "";
        return (
          <g key={s.id}>
            {showCi && asPct && s.points.length > 1 ? (
              <polygon
                points={buildCiPolygonPoints(s, geom)}
                fill={s.color}
                fillOpacity="0.12"
                stroke="none"
              />
            ) : null}
            {smoothed && s.points.length > 1 ? (
              <polyline
                points={rawLinePts}
                fill="none"
                stroke={s.color}
                strokeOpacity="0.22"
                strokeWidth="1"
                strokeLinejoin="round"
              />
            ) : null}
            {s.points.length > 1 ? (
              <polyline
                points={linePts}
                fill="none"
                stroke={s.color}
                strokeWidth="1.9"
                strokeDasharray={s.dashed ? "5 3" : undefined}
                strokeLinejoin="round"
              />
            ) : null}
            {s.points.length <= 16
              ? s.points.map((p) => (
                  <circle
                    key={`${s.id}-${p.key}`}
                    cx={geom.toX(p.x)}
                    cy={geom.toY(p.y)}
                    r="2.4"
                    fill={s.color}
                    stroke="var(--surface)"
                    strokeWidth="1"
                  />
                ))
              : null}
          </g>
        );
      })}
    </g>
  );
}

function ChartCrosshairLayer(props: {
  series: PreparedSeries[];
  geom: ChartGeom;
  hoverX: number | null;
}): ReactNode {
  if (props.hoverX === null) return null;
  const cx = props.geom.toX(props.hoverX);
  const clampedCx = Math.max(props.geom.padL, Math.min(props.geom.w - props.geom.padR, cx));
  return (
    <g pointerEvents="none">
      <line
        x1={clampedCx}
        y1={props.geom.padT}
        x2={clampedCx}
        y2={props.geom.h - props.geom.padB}
        stroke="var(--accent)"
        strokeWidth="1"
        strokeDasharray="3 2"
        opacity="0.7"
      />
      {props.series.map((s) => {
        const pt = nearestPoint(s.points, props.hoverX ?? 0);
        if (!pt) return null;
        return (
          <circle
            key={`hov-${s.id}`}
            cx={props.geom.toX(pt.x)}
            cy={props.geom.toY(pt.y)}
            r="3.8"
            fill={s.color}
            stroke="var(--surface)"
            strokeWidth="1.5"
          />
        );
      })}
    </g>
  );
}

function HoverReadoutStrip(props: {
  series: PreparedSeries[];
  hoverX: number | null;
  asPct: boolean;
}): ReactNode {
  if (props.series.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-[10.5px] min-h-[16px]">
      {props.series.slice(0, 5).map((s) => {
        const pt =
          props.hoverX === null
            ? (s.points[s.points.length - 1] ?? null)
            : nearestPoint(s.points, props.hoverX);
        return (
          <span key={`rd-${s.id}`} className="inline-flex items-center gap-1 min-w-0">
            <span className="w-2 h-2 rounded-full shrink-0" style={{ background: s.color }} />
            <span className="truncate max-w-[96px] text-[var(--text-2)]">{s.label}</span>
            <span className="tabular font-semibold text-[var(--text)]">
              {pt ? formatMetricValue(pt.y, props.asPct) : "—"}
            </span>
          </span>
        );
      })}
    </div>
  );
}

export interface SyncedChartCardProps {
  title: string;
  badge?: string;
  metricKey: string;
  series: ChartSeriesInput[];
  smoothing: number;
  showCi: boolean;
  xMode: XAxisMode;
  hoverX: number | null;
  onHoverX: (x: number | null) => void;
  hero?: boolean;
}

export function SyncedChartCard(props: SyncedChartCardProps): ReactNode {
  const prepared = prepareAllSeries({
    series: props.series,
    metricKey: props.metricKey,
    xMode: props.xMode,
    smoothing: props.smoothing,
  });
  const bounds = computeAxisBounds({ prepared, metricKey: props.metricKey, xMode: props.xMode });
  const w = props.hero ? 640 : 420;
  const h = props.hero ? 220 : 164;
  const geom = buildChartGeom({ w, h, bounds });

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3 flex flex-col gap-1.5 min-w-0">
      <div className="flex items-center justify-between gap-2 min-w-0">
        <div className="flex items-center gap-1.5 min-w-0">
          <span
            className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-2)] truncate"
            title={props.metricKey}
          >
            {props.title}
          </span>
          {props.badge ? (
            <span className="rounded px-1.5 py-0.5 text-[9.5px] font-semibold uppercase bg-[var(--surface-2)] text-[var(--text-3)] shrink-0">
              {props.badge}
            </span>
          ) : null}
        </div>
        {props.hoverX === null ? null : (
          <span className="tabular text-[10px] font-medium text-[var(--accent)] shrink-0">
            {formatXTick(props.hoverX, props.xMode)}
          </span>
        )}
      </div>

      <HoverReadoutStrip series={prepared} hoverX={props.hoverX} asPct={bounds.asPct} />

      {prepared.length === 0 ? (
        <div
          className={`${props.hero ? "h-[200px]" : "h-[144px]"} flex items-center justify-center text-xs text-[var(--text-3)]`}
        >
          No logged values for {props.title}.
        </div>
      ) : (
        <svg
          viewBox={`0 0 ${w} ${h}`}
          className={`w-full ${props.hero ? "h-[210px]" : "h-[154px]"} block select-none`}
          onMouseMove={(event: MouseEvent<SVGSVGElement>): void => {
            const rect = event.currentTarget.getBoundingClientRect();
            if (rect.width <= 0) return;
            const svgX = ((event.clientX - rect.left) / rect.width) * w;
            const plotW = Math.max(1, w - geom.padL - geom.padR);
            const frac = Math.max(0, Math.min(1, (svgX - geom.padL) / plotW));
            props.onHoverX(bounds.minX + frac * (bounds.maxX - bounds.minX));
          }}
          onMouseLeave={(): void => props.onHoverX(null)}
        >
          <ChartAxesLayer geom={geom} bounds={bounds} xMode={props.xMode} />
          <ChartSeriesLayer
            series={prepared}
            geom={geom}
            showCi={props.showCi}
            asPct={bounds.asPct}
            smoothed={props.smoothing > 0}
          />
          <ChartCrosshairLayer series={prepared} geom={geom} hoverX={props.hoverX} />
        </svg>
      )}
    </div>
  );
}

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

/** Pure series preparation, EMA smoothing, 95% CI calculation, and SVG geometry for run charts. */

import type { MetricPoint } from "@/lib/types";
import { type ChartSeriesInput, type XAxisMode, isRateMetric, numericFromPoint } from "./metrics";

export interface PlotPoint {
  key: string;
  x: number;
  y: number;
  rawY: number;
  ciHalf: number;
  update: number;
}

export interface PreparedSeries {
  id: string;
  label: string;
  color: string;
  dashed: boolean;
  points: PlotPoint[];
}

export interface AxisBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  asPct: boolean;
}

export interface ChartGeom {
  w: number;
  h: number;
  padL: number;
  padR: number;
  padT: number;
  padB: number;
  toX: (x: number) => number;
  toY: (y: number) => number;
}

const SAMPLE_COUNT_KEYS = ["eval_games", "num_games", "n_games", "games", "episodes", "n"];

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

export function prepareAllSeries(args: {
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

export function computeAxisBounds(args: {
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

export function nearestPoint(points: PlotPoint[], targetX: number): PlotPoint | null {
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

export function formatXTick(value: number, xMode: XAxisMode): string {
  if (xMode === "progress") return `${Math.round(value)}%`;
  if (value >= 1000) return `u${(value / 1000).toFixed(1)}k`;
  return `u${Math.round(value)}`;
}

export function buildChartGeom(args: { w: number; h: number; bounds: AxisBounds }): ChartGeom {
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

export function buildCiPolygonPoints(series: PreparedSeries, geom: ChartGeom): string {
  const upper = series.points.map(
    (p) => `${geom.toX(p.x).toFixed(1)},${geom.toY(Math.min(1, p.y + p.ciHalf)).toFixed(1)}`,
  );
  const lower = series.points
    .toReversed()
    .map((p) => `${geom.toX(p.x).toFixed(1)},${geom.toY(Math.max(0, p.y - p.ciHalf)).toFixed(1)}`);
  return [...upper, ...lower].join(" ");
}

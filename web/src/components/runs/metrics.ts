/** Pure metric discovery, formatting, checkpoint resolution, and run-summary helpers. */

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

export function resolveBestWin(run: RunSummary): number | null {
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

export function resolveLatestWin(run: RunSummary): number | null {
  const standard = run.latest_ema_win ?? run.latest_ensemble_win ?? run.latest_live_win;
  if (typeof standard === "number" && Number.isFinite(standard)) return standard;
  if (run.latest_metrics) {
    for (const [key, val] of Object.entries(run.latest_metrics)) {
      if (isWinRateKey(key) && Number.isFinite(val)) return val;
    }
  }
  return null;
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

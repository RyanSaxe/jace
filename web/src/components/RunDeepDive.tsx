/** Single-Run Deep Dive: dynamic multi-track eval chart, checkpoint ladder, and recursive config diff. */

import { ArrowLeft, ArrowRight, GitCompareArrows, Layers, SlidersHorizontal } from "lucide-react";
import { type ReactNode, useMemo, useRef, useState } from "react";
import {
  type ChartSeriesInput,
  SyncedChartCard,
  type XAxisMode,
  colorForIndex,
  discoverNumericKeys,
  formatCheckpointRef,
  formatMetricValue,
  humanizeMetricKey,
  isRateMetric,
  isWinRateKey,
  numericFromPoint,
  runStatusMeta,
} from "@/components/RunsCharts";
import { Badge, Button, SegmentedControl } from "@/components/ui/Primitives";
import { useExtraRunDetails, useRunDetail } from "@/hooks/useStudioData";
import type {
  CheckpointInfo,
  CompareTarget,
  MetricPoint,
  RunDetail,
  RunSummary,
} from "@/lib/types";

export interface RunDeepDiveProps {
  run: RunSummary;
  runs: RunSummary[];
  details: Record<string, RunDetail>;
  initialDiffRunId?: string | null;
  onBack: () => void;
  onBrowseGames: (runId: string) => void;
  onComparePair: (target: CompareTarget) => void;
}

interface TrackSummary {
  key: string;
  label: string;
  color: string;
  latest: number | null;
  peak: number | null;
  peakUpdate: number | null;
  asPct: boolean;
}

interface DiffRow {
  key: string;
  left: string;
  right: string;
  changed: boolean;
}

function isScalar(value: unknown): value is string | number | boolean {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean";
}

function formatBytes(bytes: number): string {
  if (bytes <= 0) return "—";
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}

function flattenArray(items: unknown[], prefix: string, out: Map<string, string>): void {
  const allScalar = items.every((item) => item === null || isScalar(item));
  if (allScalar && prefix) {
    out.set(prefix, `[${items.map((v) => String(v)).join(", ")}]`);
    return;
  }
  items.forEach((item, idx) => {
    flattenConfigValue(item, `${prefix}[${idx}]`, out);
  });
}

function flattenConfigValue(value: unknown, prefix: string, out: Map<string, string>): void {
  if (value === null || value === undefined || isScalar(value)) {
    if (prefix) out.set(prefix, String(value));
    return;
  }
  if (Array.isArray(value)) {
    flattenArray(value, prefix, out);
    return;
  }
  if (typeof value === "object") {
    for (const [k, v] of Object.entries(value)) {
      flattenConfigValue(v, prefix ? `${prefix}.${k}` : k, out);
    }
  }
}

function flattenConfig(config: Record<string, unknown> | undefined): Map<string, string> {
  const out = new Map<string, string>();
  if (config) flattenConfigValue(config, "", out);
  return out;
}

function formatConfigTag(key: string, val: unknown): string | null {
  if (!isScalar(val)) return null;
  const text = String(val);
  if (text.length === 0 || text.length > 24 || text.includes("/")) return null;
  return `${key}=${text}`;
}

function extractConfigTags(run: RunSummary, config: Record<string, unknown> | undefined): string[] {
  const tags: string[] = [];
  const source = run.config_tags ?? config ?? {};
  for (const [key, val] of Object.entries(source)) {
    if (tags.length >= 4) break;
    const tag = formatConfigTag(key, val);
    if (tag) tags.push(tag);
  }
  return tags;
}

function buildTrackSummary(args: {
  key: string;
  points: MetricPoint[];
  color: string;
}): TrackSummary {
  const values: number[] = [];
  let latest: number | null = null;
  let peak: number | null = null;
  let peakUpdate: number | null = null;
  for (const pt of args.points) {
    const val = numericFromPoint(pt, args.key);
    if (val === null) continue;
    values.push(val);
    latest = val;
    if (peak === null || val > peak) {
      peak = val;
      peakUpdate = pt.update;
    }
  }
  return {
    key: args.key,
    label: humanizeMetricKey(args.key),
    color: args.color,
    latest,
    peak,
    peakUpdate,
    asPct: isRateMetric(args.key, values),
  };
}

function findEvalPointForIter(points: MetricPoint[], iteration: number): MetricPoint | null {
  if (iteration < 0) return null;
  let best: MetricPoint | null = null;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const pt of points) {
    const dist = Math.abs(pt.update - iteration);
    if (dist < bestDist) {
      bestDist = dist;
      best = pt;
    }
  }
  return bestDist <= 50 ? best : null;
}

function resolvePairIndices(
  selectedNames: string[],
  checkpoints: CheckpointInfo[],
): [string, string] | null {
  const valid = selectedNames.filter((name) => checkpoints.some((c) => c.name === name));
  if (valid.length >= 2) {
    const a = valid[0];
    const b = valid[1];
    if (a && b) return [a, b];
  }
  if (checkpoints.length >= 2) {
    const first = checkpoints[0];
    const last = checkpoints[checkpoints.length - 1];
    if (first && last) return [first.name, last.name];
  }
  return null;
}

function CheckpointLadderCard(props: {
  runId: string;
  checkpoints: CheckpointInfo[];
  evalPoints: MetricPoint[];
  evalIterations: number[];
  selectedCkpts: string[];
  onToggleCkpt: (name: string) => void;
  onComparePair: (target: CompareTarget) => void;
}): ReactNode {
  const evalKeys = useMemo(
    () => discoverNumericKeys(props.evalPoints).slice(0, 4),
    [props.evalPoints],
  );
  const evalSet = useMemo(() => new Set(props.evalIterations), [props.evalIterations]);
  const pair = resolvePairIndices(props.selectedCkpts, props.checkpoints);

  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 border-b border-[var(--border)]">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-2)]">
            Checkpoint Ladder
          </h3>
          <Badge>{props.checkpoints.length}</Badge>
        </div>
        <Button
          variant="primary"
          size="sm"
          disabled={pair === null}
          onClick={(): void => {
            if (!pair) return;
            props.onComparePair({
              ckptA: formatCheckpointRef(props.runId, pair[0]),
              ckptB: formatCheckpointRef(props.runId, pair[1]),
              runId: props.runId,
            });
          }}
        >
          <GitCompareArrows className="w-3.5 h-3.5" />
          {pair ? `Compare ${pair[0]} vs ${pair[1]}` : "Select 2 checkpoints"}
          <ArrowRight className="w-3.5 h-3.5" />
        </Button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[540px] border-collapse text-left text-xs">
          <thead>
            <tr className="border-b border-[var(--border)] bg-[var(--surface-2)] text-[10.5px] uppercase tracking-wider text-[var(--text-3)]">
              <th className="py-2 px-3 w-10">Pick</th>
              <th className="py-2 px-2">Checkpoint</th>
              <th className="py-2 px-2 text-right">Update</th>
              {evalKeys.map((key) => (
                <th key={key} className="py-2 px-2 text-right whitespace-nowrap">
                  {humanizeMetricKey(key)}
                </th>
              ))}
              <th className="py-2 px-2 text-right">Size</th>
              <th className="py-2 px-3 text-right">Games</th>
            </tr>
          </thead>
          <tbody>
            {props.checkpoints.length === 0 ? (
              <tr>
                <td colSpan={5 + evalKeys.length} className="py-6 text-center text-[var(--text-3)]">
                  No checkpoints saved for this run yet.
                </td>
              </tr>
            ) : (
              props.checkpoints.toReversed().map((ckpt) => {
                const pickIdx = props.selectedCkpts.indexOf(ckpt.name);
                const evalPt = findEvalPointForIter(props.evalPoints, ckpt.iteration);
                return (
                  <tr
                    key={ckpt.name}
                    className="border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--surface-2)] transition-colors"
                    style={pickIdx === -1 ? undefined : { background: "var(--accent-soft)" }}
                  >
                    <td className="py-2 px-3">
                      <input
                        type="checkbox"
                        checked={pickIdx !== -1}
                        onChange={(): void => props.onToggleCkpt(ckpt.name)}
                        aria-label={`Select checkpoint ${ckpt.name}`}
                        className="accent-[var(--accent)] cursor-pointer"
                      />
                    </td>
                    <td className="py-2 px-2 font-medium tabular">
                      <span className="inline-flex items-center gap-1.5">
                        {ckpt.name}
                        {pickIdx === 0 ? <Badge tone="accent">A</Badge> : null}
                        {pickIdx === 1 ? <Badge tone="warn">B</Badge> : null}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-right tabular text-[var(--text-2)]">
                      {ckpt.iteration >= 0 ? `u${ckpt.iteration}` : "—"}
                    </td>
                    {evalKeys.map((key) => {
                      const val = evalPt ? numericFromPoint(evalPt, key) : null;
                      const asPct = val !== null && isRateMetric(key, [val]);
                      return (
                        <td key={key} className="py-2 px-2 text-right tabular">
                          {val === null ? "—" : formatMetricValue(val, asPct)}
                        </td>
                      );
                    })}
                    <td className="py-2 px-2 text-right tabular text-[var(--text-3)]">
                      {formatBytes(ckpt.size_bytes)}
                    </td>
                    <td className="py-2 px-3 text-right">
                      {evalSet.has(ckpt.iteration) ? (
                        <Badge tone="win">Eval</Badge>
                      ) : (
                        <span className="text-[var(--text-3)]">—</span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function matchesConfigNeedle(row: DiffRow, needle: string): boolean {
  if (!needle) return true;
  return (
    row.key.toLowerCase().includes(needle) ||
    row.left.toLowerCase().includes(needle) ||
    row.right.toLowerCase().includes(needle)
  );
}

function buildConfigDiffRows(args: {
  config: Record<string, unknown> | undefined;
  diffConfig: Record<string, unknown> | undefined;
  diffRunId: string;
  changedOnly: boolean;
  filterText: string;
}): DiffRow[] {
  const leftMap = flattenConfig(args.config);
  const rightMap = args.diffRunId ? flattenConfig(args.diffConfig) : new Map<string, string>();
  const allKeys = new Set<string>([...leftMap.keys(), ...rightMap.keys()]);
  const needle = args.filterText.trim().toLowerCase();
  const out: DiffRow[] = [];
  for (const key of [...allKeys].toSorted((a, b) => a.localeCompare(b))) {
    const left = leftMap.get(key) ?? "—";
    const right = rightMap.get(key) ?? "—";
    const changed = args.diffRunId !== "" && left !== right;
    const row: DiffRow = { key, left, right, changed };
    if (args.diffRunId && args.changedOnly && !changed) continue;
    if (matchesConfigNeedle(row, needle)) out.push(row);
  }
  return out;
}

function ConfigDiffTable(props: {
  runName: string;
  diffLabel: string | null;
  rows: DiffRow[];
  emptyMessage: string;
}): ReactNode {
  return (
    <div className="max-h-[340px] overflow-auto">
      <table className="w-full border-collapse text-left text-xs">
        <thead>
          <tr className="border-b border-[var(--border)] bg-[var(--surface-2)] text-[10.5px] uppercase tracking-wider text-[var(--text-3)]">
            <th className="py-1.5 px-3">Key</th>
            <th className="py-1.5 px-3">{props.runName}</th>
            {props.diffLabel ? <th className="py-1.5 px-3">{props.diffLabel}</th> : null}
          </tr>
        </thead>
        <tbody>
          {props.rows.length === 0 ? (
            <tr>
              <td
                colSpan={props.diffLabel ? 3 : 2}
                className="py-6 text-center text-[var(--text-3)]"
              >
                {props.emptyMessage}
              </td>
            </tr>
          ) : (
            props.rows.map((row) => (
              <tr
                key={row.key}
                className="border-b border-[var(--border)] last:border-b-0"
                style={row.changed ? { background: "var(--warn-soft)" } : undefined}
              >
                <td className="py-1.5 px-3 tabular text-[var(--text-2)] font-medium">{row.key}</td>
                <td className="py-1.5 px-3 tabular break-all">{row.left}</td>
                {props.diffLabel ? (
                  <td className="py-1.5 px-3 tabular break-all text-[var(--text-2)]">
                    {row.right}
                  </td>
                ) : null}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

function ConfigDiffCard(props: {
  run: RunSummary;
  runs: RunSummary[];
  config: Record<string, unknown> | undefined;
  diffRunId: string;
  diffConfig: Record<string, unknown> | undefined;
  onChangeDiffRunId: (id: string) => void;
}): ReactNode {
  const [changedOnly, setChangedOnly] = useState(props.diffRunId !== "");
  const [filterText, setFilterText] = useState("");
  const otherRuns = props.runs.filter((r) => r.run_id !== props.run.run_id);

  const rows = useMemo(
    () =>
      buildConfigDiffRows({
        config: props.config,
        diffConfig: props.diffConfig,
        diffRunId: props.diffRunId,
        changedOnly,
        filterText,
      }),
    [props.config, props.diffConfig, props.diffRunId, changedOnly, filterText],
  );
  const changedCount = rows.filter((r) => r.changed).length;
  const diffLabel = props.diffRunId
    ? (otherRuns.find((r) => r.run_id === props.diffRunId)?.name ?? props.diffRunId)
    : null;
  const emptyMessage =
    props.diffRunId && changedOnly
      ? "No differing config keys between these runs."
      : "No config keys logged.";

  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 border-b border-[var(--border)]">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="w-3.5 h-3.5 text-[var(--accent)]" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-2)]">
            Config &amp; Diff
          </h3>
          <Badge>{rows.length} keys</Badge>
          {props.diffRunId && changedCount > 0 ? (
            <Badge tone="warn">{changedCount} diff</Badge>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="search"
            value={filterText}
            onChange={(e): void => setFilterText(e.target.value)}
            placeholder="Filter keys…"
            aria-label="Filter config keys"
            className="h-7 w-36 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs"
          />
          <select
            value={props.diffRunId}
            onChange={(e): void => {
              const next = e.target.value;
              props.onChangeDiffRunId(next);
              if (next !== "") setChangedOnly(true);
            }}
            aria-label="Compare config against run"
            className="h-7 max-w-[200px] rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs"
          >
            <option value="">No comparison run</option>
            {otherRuns.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                vs {r.name}
              </option>
            ))}
          </select>
          {props.diffRunId ? (
            <Button
              size="sm"
              variant={changedOnly ? "primary" : "default"}
              onClick={(): void => setChangedOnly((prev) => !prev)}
            >
              Changed only
            </Button>
          ) : null}
        </div>
      </div>
      <ConfigDiffTable
        runName={props.run.name}
        diffLabel={diffLabel}
        rows={rows}
        emptyMessage={emptyMessage}
      />
    </section>
  );
}

function buildHeroChartData(args: { run: RunSummary; detail: RunDetail | null }): {
  primaryKey: string;
  heroSeries: ChartSeriesInput[];
  tracks: TrackSummary[];
  secondaryPanels: Array<{ key: string; source: "eval" | "train"; points: MetricPoint[] }>;
} {
  const evalPts = args.detail?.eval_points ?? [];
  const trainPts = args.detail?.train_points ?? [];
  const evalKeys = discoverNumericKeys(evalPts);
  const trainKeys = discoverNumericKeys(trainPts);
  const winKeys = evalKeys.filter(isWinRateKey);
  const heroKeys = winKeys.length > 0 ? winKeys : evalKeys.slice(0, 3);
  const heroKeySet = new Set(heroKeys);
  const primaryKey =
    winKeys.length > 0 || heroKeys.every((k) => /(rate|fraction|ratio|pct|percent|acc)/i.test(k))
      ? "__track_win_rate__"
      : "__track_val__";

  const heroSeries: ChartSeriesInput[] = heroKeys.map((key, idx) => {
    const ciBase = key.replace(/_win_rate$/i, "");
    const primaryBase = primaryKey.replace(/_win_rate$/i, "");
    return {
      id: key,
      label: humanizeMetricKey(key),
      color: colorForIndex(idx),
      totalUpdates: args.run.total_updates,
      points: evalPts.map((pt) => ({
        ...pt,
        update: pt.update,
        [primaryKey]: numericFromPoint(pt, key),
        [`${primaryBase}_ci_low`]: numericFromPoint(pt, `${ciBase}_ci_low`),
        [`${primaryBase}_ci_high`]: numericFromPoint(pt, `${ciBase}_ci_high`),
      })),
      dashed: idx >= 2,
    };
  });

  const tracks = heroKeys.map((key, idx) =>
    buildTrackSummary({ key, points: evalPts, color: colorForIndex(idx) }),
  );
  const secondaryEval = evalKeys
    .filter((item) => !heroKeySet.has(item))
    .slice(0, 2)
    .map((key) => ({ key, source: "eval" as const, points: evalPts }));
  const secondaryTrain = trainKeys
    .slice(0, Math.max(0, 3 - secondaryEval.length))
    .map((key) => ({ key, source: "train" as const, points: trainPts }));
  return { primaryKey, heroSeries, tracks, secondaryPanels: [...secondaryEval, ...secondaryTrain] };
}

function DeepDiveHeader(props: {
  run: RunSummary;
  configTags: string[];
  comparePair: [string, string] | null;
  onBack: () => void;
  onScrollToDiff: () => void;
  onBrowseGames: (runId: string) => void;
  onComparePair: (target: CompareTarget) => void;
}): ReactNode {
  const status = runStatusMeta(props.run.status);
  const totalSuffix = props.run.total_updates > 0 ? `/${props.run.total_updates}` : "";

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 border-b border-[var(--border)] bg-[var(--surface)] shrink-0">
      <div className="flex flex-wrap items-center gap-2 min-w-0">
        <Button size="sm" variant="ghost" onClick={props.onBack}>
          <ArrowLeft className="w-3.5 h-3.5" />
          Runs
        </Button>
        <span className="text-[var(--text-3)]">/</span>
        <h2 className="text-sm font-semibold tracking-tight truncate max-w-[260px]">
          {props.run.name}
        </h2>
        <Badge tone={status.tone}>{status.label}</Badge>
        <span className="tabular text-xs text-[var(--text-2)]">
          u{props.run.latest_update}
          {totalSuffix}
        </span>
        {props.run.latest_tps > 0 ? (
          <span className="tabular text-xs text-[var(--text-3)]">
            {Math.round(props.run.latest_tps)} TPS
          </span>
        ) : null}
        {props.configTags.map((tag) => (
          <Badge key={tag}>{tag}</Badge>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <Button size="sm" onClick={props.onScrollToDiff}>
          <SlidersHorizontal className="w-3.5 h-3.5" />
          Diff Config vs…
        </Button>
        <Button size="sm" onClick={(): void => props.onBrowseGames(props.run.run_id)}>
          <Layers className="w-3.5 h-3.5" />
          Browse Games
        </Button>
        <Button
          size="sm"
          variant="primary"
          disabled={props.comparePair === null}
          onClick={(): void => {
            if (!props.comparePair) return;
            props.onComparePair({
              ckptA: formatCheckpointRef(props.run.run_id, props.comparePair[0]),
              ckptB: formatCheckpointRef(props.run.run_id, props.comparePair[1]),
              runId: props.run.run_id,
            });
          }}
        >
          Compare checkpoints
          <ArrowRight className="w-3.5 h-3.5" />
        </Button>
      </div>
    </div>
  );
}

function DeepDiveKpiStrip(props: {
  tracks: TrackSummary[];
  smoothing: number;
  showCi: boolean;
  xMode: XAxisMode;
  onToggleSmoothing: () => void;
  onToggleCi: () => void;
  onChangeXMode: (mode: XAxisMode) => void;
}): ReactNode {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 flex-1 min-w-[260px]">
        {props.tracks.slice(0, 4).map((t) => (
          <div
            key={t.key}
            className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2"
          >
            <div className="flex items-center gap-1.5 text-[10.5px] text-[var(--text-3)] truncate">
              <span className="w-2 h-2 rounded-full shrink-0" style={{ background: t.color }} />
              <span className="truncate">{t.label}</span>
            </div>
            <div className="flex items-baseline justify-between gap-2 mt-1">
              <span className="tabular text-base font-bold">
                {t.latest === null ? "—" : formatMetricValue(t.latest, t.asPct)}
              </span>
              <span className="tabular text-[10.5px] text-[var(--text-3)]">
                peak {t.peak === null ? "—" : formatMetricValue(t.peak, t.asPct)}
                {t.peakUpdate === null ? "" : ` @u${t.peakUpdate}`}
              </span>
            </div>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        <Button
          size="sm"
          variant={props.smoothing > 0 ? "primary" : "default"}
          onClick={props.onToggleSmoothing}
        >
          {props.smoothing > 0 ? `EMA ${props.smoothing.toFixed(1)}` : "Raw"}
        </Button>
        <Button size="sm" variant={props.showCi ? "primary" : "default"} onClick={props.onToggleCi}>
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
      </div>
    </div>
  );
}

function DeepDiveChartsGrid(props: {
  run: RunSummary;
  primaryKey: string;
  heroSeries: ChartSeriesInput[];
  secondaryPanels: Array<{ key: string; source: "eval" | "train"; points: MetricPoint[] }>;
  smoothing: number;
  showCi: boolean;
  xMode: XAxisMode;
  hoverX: number | null;
  onHoverX: (x: number | null) => void;
}): ReactNode {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
      <div className="xl:col-span-2 min-w-0">
        <SyncedChartCard
          title="Evaluation Win-Rate Tracks"
          badge={`${props.heroSeries.length} tracks`}
          metricKey={props.primaryKey}
          series={props.heroSeries}
          smoothing={props.smoothing}
          showCi={props.showCi}
          xMode={props.xMode}
          hoverX={props.hoverX}
          onHoverX={props.onHoverX}
          hero
        />
      </div>
      <div className="flex flex-col gap-3 min-w-0">
        {props.secondaryPanels.length === 0 ? (
          <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4 text-xs text-[var(--text-3)]">
            No secondary diagnostic metrics logged yet.
          </div>
        ) : (
          props.secondaryPanels.map((panel, idx) => (
            <SyncedChartCard
              key={`${panel.source}:${panel.key}`}
              title={humanizeMetricKey(panel.key)}
              badge={panel.source}
              metricKey={panel.key}
              series={[
                {
                  id: props.run.run_id,
                  label: props.run.name,
                  color: colorForIndex(idx + 1),
                  totalUpdates: props.run.total_updates,
                  points: panel.points,
                },
              ]}
              smoothing={props.smoothing}
              showCi={props.showCi}
              xMode={props.xMode}
              hoverX={props.hoverX}
              onHoverX={props.onHoverX}
            />
          ))
        )}
      </div>
    </div>
  );
}

function toggleCkptSelection(prev: string[], name: string): string[] {
  return prev.includes(name) ? prev.filter((item) => item !== name) : [...prev.slice(-1), name];
}

function DeepDiveContent(props: {
  run: RunSummary;
  runs: RunSummary[];
  detail: RunDetail | null;
  diffRunId: string;
  diffDetail: RunDetail | null;
  diffRef: React.RefObject<HTMLDivElement | null>;
  selectedCkpts: string[];
  onToggleCkpt: (name: string) => void;
  onChangeDiffRunId: (id: string) => void;
  onComparePair: (target: CompareTarget) => void;
}): ReactNode {
  const [smoothing, setSmoothing] = useState(0.4);
  const [showCi, setShowCi] = useState(true);
  const [xMode, setXMode] = useState<XAxisMode>("updates");
  const [hoverX, setHoverX] = useState<number | null>(null);
  const { primaryKey, heroSeries, tracks, secondaryPanels } = useMemo(
    () => buildHeroChartData({ run: props.run, detail: props.detail }),
    [props.run, props.detail],
  );

  return (
    <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
      <DeepDiveKpiStrip
        tracks={tracks}
        smoothing={smoothing}
        showCi={showCi}
        xMode={xMode}
        onToggleSmoothing={(): void => setSmoothing((s) => (s > 0 ? 0 : 0.6))}
        onToggleCi={(): void => setShowCi((v) => !v)}
        onChangeXMode={setXMode}
      />
      <DeepDiveChartsGrid
        run={props.run}
        primaryKey={primaryKey}
        heroSeries={heroSeries}
        secondaryPanels={secondaryPanels}
        smoothing={smoothing}
        showCi={showCi}
        xMode={xMode}
        hoverX={hoverX}
        onHoverX={setHoverX}
      />
      <CheckpointLadderCard
        runId={props.run.run_id}
        checkpoints={props.detail?.checkpoints ?? []}
        evalPoints={props.detail?.eval_points ?? []}
        evalIterations={props.detail?.eval_iterations ?? []}
        selectedCkpts={props.selectedCkpts}
        onToggleCkpt={props.onToggleCkpt}
        onComparePair={props.onComparePair}
      />
      <div ref={props.diffRef}>
        <ConfigDiffCard
          run={props.run}
          runs={props.runs}
          config={props.detail?.config}
          diffRunId={props.diffRunId}
          diffConfig={props.diffDetail?.config}
          onChangeDiffRunId={props.onChangeDiffRunId}
        />
      </div>
    </div>
  );
}

function pickDefaultDiffRun(
  diffRunId: string,
  runs: RunSummary[],
  currentRunId: string,
): string | null {
  if (diffRunId) return null;
  return runs.find((r) => r.run_id !== currentRunId)?.run_id ?? null;
}

export function RunDeepDive(props: RunDeepDiveProps): ReactNode {
  const [selectedCkpts, setSelectedCkpts] = useState<string[]>([]);
  const [diffRunId, setDiffRunId] = useState(props.initialDiffRunId ?? "");
  const diffRef = useRef<HTMLDivElement | null>(null);

  const cachedDetail = props.details[props.run.run_id] ?? null;
  const fetchedDetail = useRunDetail(cachedDetail === null ? props.run.run_id : null);
  const detail = cachedDetail ?? fetchedDetail;
  const extraDiffDetails = useExtraRunDetails(diffRunId ? [diffRunId] : [], props.details);
  const diffDetail = extraDiffDetails[diffRunId] ?? null;

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      <DeepDiveHeader
        run={props.run}
        configTags={extractConfigTags(props.run, detail?.config)}
        comparePair={resolvePairIndices(selectedCkpts, detail?.checkpoints ?? [])}
        onBack={props.onBack}
        onScrollToDiff={(): void => {
          const next = pickDefaultDiffRun(diffRunId, props.runs, props.run.run_id);
          if (next) setDiffRunId(next);
          diffRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        }}
        onBrowseGames={props.onBrowseGames}
        onComparePair={props.onComparePair}
      />
      <DeepDiveContent
        run={props.run}
        runs={props.runs}
        detail={detail}
        diffRunId={diffRunId}
        diffDetail={diffDetail}
        diffRef={diffRef}
        selectedCkpts={selectedCkpts}
        onToggleCkpt={(name): void => setSelectedCkpts((prev) => toggleCkptSelection(prev, name))}
        onChangeDiffRunId={setDiffRunId}
        onComparePair={props.onComparePair}
      />
    </div>
  );
}

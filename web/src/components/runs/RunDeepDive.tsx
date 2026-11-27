/** Single-Run Deep Dive: dynamic multi-track eval chart, checkpoint ladder, and recursive config diff. */

import { ArrowLeft, ArrowRight, Layers, SlidersHorizontal } from "lucide-react";
import { type ReactNode, type RefObject, useMemo, useRef, useState } from "react";
import { Badge, Button, SegmentedControl } from "@/components/ui/Primitives";
import { useExtraRunDetails, useRunDetail } from "@/hooks/useStudioData";
import type { CompareTarget, MetricPoint, RunDetail, RunSummary } from "@/lib/types";
import {
  type ChartSeriesInput,
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
} from "./metrics";
import {
  CheckpointLadderCard,
  ConfigDiffCard,
  isScalar,
  resolvePairIndices,
} from "./RunDeepDiveCards";
import { SyncedChartCard } from "./SyncedChartCard";

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
  diffRef: RefObject<HTMLDivElement | null>;
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

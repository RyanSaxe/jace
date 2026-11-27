/** Bradley-Terry Elo progression SVG chart with frontier CI ribbon and deconflicted labels. */

import type { ReactNode } from "react";
import { SegmentedControl } from "@/components/ui/Primitives";
import type { ArenaModelRow } from "@/lib/types";
import {
  type EloXAxis,
  type LabelCandidate,
  computeNiceTicks,
  formatElo,
  placeDeconflictedLabels,
  truncateLabel,
} from "./benchmarksUtils";

interface PlotPoint {
  model: ArenaModelRow;
  xVal: number;
  isFrontier: boolean;
}

interface EloGeometry {
  width: number;
  height: number;
  padL: number;
  padR: number;
  yTicks: number[];
  xTicks: number[];
  fx: (v: number) => number;
  fy: (e: number) => number;
}

const COMPACT_DIMS = {
  width: 334,
  height: 104,
  padL: 36,
  padR: 14,
  padT: 16,
  padB: 20,
  yStep: 200,
};
const FULL_DIMS = { width: 680, height: 210, padL: 54, padR: 32, padT: 20, padB: 26, yStep: 100 };

function buildEloSeries(models: ArenaModelRow[], xAxis: EloXAxis): PlotPoint[] {
  const raw = models
    .map((model) => {
      const xVal = xAxis === "updates" ? model.updates : model.search_ms_per_dec;
      return xVal === null ? null : { model, xVal };
    })
    .filter((entry): entry is { model: ArenaModelRow; xVal: number } => entry !== null)
    .toSorted((a, b) => a.xVal - b.xVal || a.model.elo - b.model.elo);

  const bestByX = new Map<number, ArenaModelRow>();
  for (const item of raw) {
    const existing = bestByX.get(item.xVal);
    if (!existing || item.model.elo > existing.elo) {
      bestByX.set(item.xVal, item.model);
    }
  }
  return raw.map((item) => ({
    model: item.model,
    xVal: item.xVal,
    isFrontier: bestByX.get(item.xVal)?.model_id === item.model.model_id,
  }));
}

function formatXAxisTick(val: number, xAxis: EloXAxis): string {
  if (xAxis === "updates") return `u${Math.round(val).toLocaleString("en-US")}`;
  return val >= 100 ? `${Math.round(val)}ms` : `${val.toFixed(1)}ms`;
}

function selectXStep(maxX: number, xAxis: EloXAxis): number {
  if (xAxis === "updates") {
    if (maxX <= 500) return 100;
    if (maxX <= 1200) return 250;
    return 500;
  }
  if (maxX <= 10) return 2;
  if (maxX <= 50) return 10;
  if (maxX <= 200) return 50;
  return 100;
}

function buildEloGeometry(args: {
  points: PlotPoint[];
  anchorElo: number;
  xAxis: EloXAxis;
  compact: boolean;
}): EloGeometry {
  const { points, anchorElo, xAxis, compact } = args;
  const dims = compact ? COMPACT_DIMS : FULL_DIMS;
  const plotW = dims.width - dims.padL - dims.padR;
  const plotH = dims.height - dims.padT - dims.padB;

  const minX = Math.min(0, ...points.map((p) => p.xVal));
  const maxX = Math.max(1, ...points.map((p) => p.xVal)) * 1.04;
  const minElo = Math.min(anchorElo - 40, ...points.map((p) => p.model.elo - p.model.ci_half));
  const maxElo = Math.max(anchorElo + 60, ...points.map((p) => p.model.elo + p.model.ci_half));
  const yTicks = computeNiceTicks(minElo, maxElo, dims.yStep);
  const yMin = (yTicks[0] ?? 1460) - 30;
  const yMax = (yTicks[yTicks.length - 1] ?? 1920) + 30;
  const ySpan = Math.max(1, yMax - yMin);
  const xSpan = Math.max(1e-6, maxX - minX);
  const xStep = selectXStep(maxX, xAxis);
  const rawXTicks = computeNiceTicks(0, Math.max(xStep, Math.floor(maxX / xStep) * xStep), xStep);

  return {
    width: dims.width,
    height: dims.height,
    padL: dims.padL,
    padR: dims.padR,
    yTicks,
    xTicks: compact ? rawXTicks.filter((_, idx) => idx % 2 === 0) : rawXTicks,
    fx: (v: number): number => dims.padL + ((v - minX) / xSpan) * plotW,
    fy: (e: number): number => dims.padT + plotH - ((e - yMin) / ySpan) * plotH,
  };
}

function eloLabelPriority(p: PlotPoint, topModelId: string, anchorModel: string): number {
  if (p.model.model_id === topModelId) return 4;
  if (p.model.model_id === anchorModel) return 3;
  return p.isFrontier ? 2 : 1;
}

function eloLabelFill(isTop: boolean, isFrontier: boolean): string {
  if (isTop) return "var(--text)";
  return isFrontier ? "var(--text-2)" : "var(--win)";
}

function buildEloLabelCandidates(args: {
  points: PlotPoint[];
  geo: EloGeometry;
  anchorModel: string;
}): LabelCandidate[] {
  const { points, geo, anchorModel } = args;
  const topModelId = points.toSorted((a, b) => b.model.elo - a.model.elo)[0]?.model.model_id ?? "";
  const ordered = points.toSorted(
    (a, b) =>
      eloLabelPriority(b, topModelId, anchorModel) - eloLabelPriority(a, topModelId, anchorModel) ||
      b.model.elo - a.model.elo,
  );
  return ordered.map((p) => {
    const isTop = p.model.model_id === topModelId;
    return {
      id: p.model.model_id,
      cx: geo.fx(p.xVal),
      cy: geo.fy(p.model.elo),
      text: truncateLabel(p.model.label, 18),
      bold: isTop,
      fill: eloLabelFill(isTop, p.isFrontier),
    };
  });
}

function EloFrontierPoints(props: {
  frontier: PlotPoint[];
  geo: EloGeometry;
  anchorModel: string;
  xAxis: EloXAxis;
}): ReactNode {
  const { frontier, geo, anchorModel, xAxis } = props;
  const lastId = frontier[frontier.length - 1]?.model.model_id ?? "";
  return (
    <>
      {frontier.map((p) => {
        const cx = geo.fx(p.xVal);
        const cy = geo.fy(p.model.elo);
        const isAnchor = p.model.model_id === anchorModel;
        const isTop = p.model.model_id === lastId;
        return (
          <circle
            key={p.model.model_id}
            cx={cx}
            cy={cy}
            r={isTop ? 4 : 3.2}
            fill={isAnchor ? "var(--text-3)" : "var(--accent)"}
            stroke="var(--surface)"
            strokeWidth={1.4}
          >
            <title>
              {`${p.model.label}: ${formatElo(p.model.elo, p.model.ci_half)} Elo (${formatXAxisTick(p.xVal, xAxis)})`}
            </title>
          </circle>
        );
      })}
    </>
  );
}

function EloSeriesSvg(props: {
  points: PlotPoint[];
  anchorModel: string;
  anchorElo: number;
  xAxis: EloXAxis;
  compact: boolean;
}): ReactNode {
  const { points, anchorModel, anchorElo, xAxis, compact } = props;
  const geo = buildEloGeometry({ points, anchorElo, xAxis, compact });
  const frontier = points.filter((p) => p.isFrontier);
  const others = compact ? [] : points.filter((p) => !p.isFrontier);
  const labels = compact
    ? []
    : placeDeconflictedLabels({
        candidates: buildEloLabelCandidates({ points, geo, anchorModel }),
        minX: geo.padL - 2,
        maxX: geo.width - 8,
        minY: 10,
        maxY: geo.height - 20,
        maxLabels: 6,
      });
  const lineD = frontier
    .map(
      (p, i) =>
        `${i === 0 ? "M" : "L"} ${geo.fx(p.xVal).toFixed(1)} ${geo.fy(p.model.elo).toFixed(1)}`,
    )
    .join(" ");
  const polyTop = frontier
    .map((p) => `${geo.fx(p.xVal).toFixed(1)},${geo.fy(p.model.elo + p.model.ci_half).toFixed(1)}`)
    .join(" ");
  const polyBot = frontier
    .toReversed()
    .map((p) => `${geo.fx(p.xVal).toFixed(1)},${geo.fy(p.model.elo - p.model.ci_half).toFixed(1)}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${geo.width} ${geo.height}`}
      className={compact ? "w-full h-[96px] block mt-1" : "w-full h-[198px] block mt-1.5"}
      role="img"
      aria-label="Bradley-Terry Elo progression chart"
    >
      {geo.yTicks.map((tick) => {
        const y = geo.fy(tick);
        const isAnchor = Math.abs(tick - anchorElo) < 0.5;
        return (
          <g key={tick}>
            <line
              x1={geo.padL - 6}
              y1={y}
              x2={geo.width - geo.padR + 8}
              y2={y}
              stroke={isAnchor ? "var(--border-strong)" : "var(--border)"}
              strokeDasharray={isAnchor ? "3 3" : "2 3"}
              strokeWidth={1}
            />
            <text
              x={geo.padL - 10}
              y={y + 3.5}
              textAnchor="end"
              fontSize={compact ? 8.5 : 9.5}
              fill="var(--text-3)"
              className="tabular"
            >
              {tick.toLocaleString("en-US")}
            </text>
          </g>
        );
      })}
      {geo.xTicks.map((tick) => (
        <text
          key={tick}
          x={geo.fx(tick)}
          y={geo.height - 6}
          textAnchor="middle"
          fontSize={compact ? 8.5 : 9.5}
          fill="var(--text-3)"
          className="tabular"
        >
          {formatXAxisTick(tick, xAxis)}
        </text>
      ))}
      {frontier.length >= 2 ? (
        <>
          <polygon points={`${polyTop} ${polyBot}`} fill="var(--accent-soft)" />
          <path
            d={lineD}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        </>
      ) : null}
      {others.map((p) => {
        const cx = geo.fx(p.xVal);
        const cy = geo.fy(p.model.elo);
        return (
          <circle
            key={p.model.model_id}
            cx={cx}
            cy={cy}
            r={2.9}
            fill="var(--win)"
            opacity={0.78}
            stroke="var(--surface)"
            strokeWidth={1.2}
          >
            <title>
              {`${p.model.label}: ${formatElo(p.model.elo, p.model.ci_half)} Elo (${formatXAxisTick(p.xVal, xAxis)})`}
            </title>
          </circle>
        );
      })}
      <EloFrontierPoints frontier={frontier} geo={geo} anchorModel={anchorModel} xAxis={xAxis} />
      {labels.map((lbl) => (
        <text
          key={lbl.id}
          x={lbl.x}
          y={lbl.y}
          textAnchor={lbl.textAnchor}
          fontSize={9.5}
          fontWeight={lbl.bold ? 600 : 500}
          fill={lbl.fill}
        >
          {lbl.text}
        </text>
      ))}
    </svg>
  );
}

function EloLegendRow(props: {
  topModel: ArenaModelRow | null;
  secondModel: ArenaModelRow | null;
  anchorRow: ArenaModelRow | null;
  compact: boolean;
}): ReactNode {
  const { topModel, secondModel, anchorRow, compact } = props;
  return (
    <div
      className={`flex items-center flex-wrap gap-x-3 gap-y-1 text-[var(--text-2)] ${
        compact ? "mt-1 text-[10.5px]" : "mt-1.5 text-[11px]"
      }`}
    >
      {topModel ? (
        <span className="inline-flex items-center gap-1.5">
          <span className="w-[7px] h-[7px] rounded-full bg-[var(--accent)] shrink-0" />
          <span>{topModel.label}</span>
          {compact ? null : (
            <span className="tabular font-semibold text-[var(--text)]">
              {formatElo(topModel.elo, topModel.ci_half)}
            </span>
          )}
        </span>
      ) : null}
      {!compact && secondModel ? (
        <span className="inline-flex items-center gap-1.5">
          <span className="w-[7px] h-[7px] rounded-full bg-[var(--win)] shrink-0" />
          <span>{secondModel.label}</span>
          <span className="tabular font-semibold text-[var(--text)]">
            {formatElo(secondModel.elo, secondModel.ci_half)}
          </span>
        </span>
      ) : null}
      {anchorRow ? (
        <span className="inline-flex items-center gap-1.5">
          <span className="w-[7px] h-[7px] rounded-full bg-[var(--text-3)] shrink-0" />
          <span>{anchorRow.label}</span>
          <span className="tabular font-semibold text-[var(--text)]">
            {Math.round(anchorRow.elo).toLocaleString("en-US")}
          </span>
        </span>
      ) : null}
    </div>
  );
}

export function BradleyTerryEloChart(props: {
  models: ArenaModelRow[];
  anchorModel: string;
  anchorElo: number;
  xAxis: EloXAxis;
  onChangeXAxis: (next: EloXAxis) => void;
  compact?: boolean;
}): ReactNode {
  const { models, anchorModel, anchorElo, xAxis, onChangeXAxis, compact = false } = props;
  const points = buildEloSeries(models, xAxis);
  const topModel = models[0] ?? null;
  const secondModel =
    models.find((m) => m.model_id !== topModel?.model_id && m.model_id !== anchorModel) ?? null;
  const anchorRow = models.find((m) => m.model_id === anchorModel) ?? null;

  return (
    <section
      className={`rounded-lg border border-[var(--border)] bg-[var(--surface)] flex flex-col ${
        compact ? "px-3 py-2.5 shrink-0" : "px-3.5 py-3"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] font-semibold uppercase tracking-[0.04em] text-[var(--text-3)]">
          Bradley-Terry Elo
        </div>
        {compact && topModel ? (
          <span className="tabular text-[11px] font-semibold text-[var(--accent)]">
            {formatElo(topModel.elo, topModel.ci_half)}
          </span>
        ) : (
          <SegmentedControl
            value={xAxis}
            onChange={onChangeXAxis}
            options={[
              { value: "updates", label: "Updates" },
              { value: "search_ms", label: "Search ms" },
            ]}
          />
        )}
      </div>

      <EloLegendRow
        topModel={topModel}
        secondModel={secondModel}
        anchorRow={anchorRow}
        compact={compact}
      />

      {points.length > 0 ? (
        <EloSeriesSvg
          points={points}
          anchorModel={anchorModel}
          anchorElo={anchorElo}
          xAxis={xAxis}
          compact={compact}
        />
      ) : (
        <div className="h-[198px] flex items-center justify-center text-xs text-[var(--text-3)]">
          {xAxis === "search_ms"
            ? "No search latency timing recorded in this format yet."
            : "No update progression points recorded for this format."}
        </div>
      )}
    </section>
  );
}

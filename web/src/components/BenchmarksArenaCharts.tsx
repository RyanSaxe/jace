/** SVG charts for the Benchmarks Elo Arena view: Bradley-Terry Elo progression and Early-Run Forecast. */

import type { ReactNode } from "react";
import { SegmentedControl } from "@/components/ui/Primitives";
import type { ArenaForecast, ArenaModelRow } from "@/lib/types";

export type EloXAxis = "updates" | "search_ms";

export function formatElo(elo: number, ciHalf: number): string {
  const rounded = Math.round(elo).toLocaleString("en-US");
  const ci = Math.round(ciHalf);
  return ci > 0 ? `${rounded} ± ${ci}` : rounded;
}

export function formatPct(val: number): string {
  return `${(val * 100).toFixed(1)}%`;
}

export function shortAnchorLabel(label: string): string {
  if (label.toLowerCase() === "heuristic") return "Heur";
  return label.length > 10 ? `${label.slice(0, 9)}…` : label;
}

interface PlotPoint {
  model: ArenaModelRow;
  xVal: number;
  isFrontier: boolean;
}

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

function computeNiceTicks(minVal: number, maxVal: number, step: number): number[] {
  const start = Math.floor(minVal / step) * step;
  const end = Math.ceil(maxVal / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= end; v += step) {
    ticks.push(v);
  }
  return ticks.length > 0 ? ticks : [minVal, maxVal];
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

interface LabelCandidate {
  id: string;
  cx: number;
  cy: number;
  text: string;
  bold: boolean;
  fill: string;
}

interface PlacedLabel {
  id: string;
  x: number;
  y: number;
  textAnchor: "start" | "end";
  text: string;
  bold: boolean;
  fill: string;
}

interface LabelBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

function boxesOverlap(a: LabelBox, b: LabelBox): boolean {
  return a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;
}

function truncateLabel(text: string, maxLen: number): string {
  return text.length > maxLen ? `${text.slice(0, maxLen - 1)}…` : text;
}

const LABEL_OFFSETS: ReadonlyArray<{ dx: number; dy: number; anchor: "start" | "end" }> = [
  { dx: 7, dy: -8, anchor: "start" },
  { dx: -7, dy: -8, anchor: "end" },
  { dx: 7, dy: 13, anchor: "start" },
  { dx: -7, dy: 13, anchor: "end" },
];

function tryPlaceCandidate(args: {
  cand: LabelCandidate;
  bounds: LabelBox;
  boxes: LabelBox[];
}): PlacedLabel | null {
  const { cand, bounds, boxes } = args;
  const widthEst = cand.text.length * 5.6 + 6;
  const heightEst = 13;
  for (const off of LABEL_OFFSETS) {
    const tx = cand.cx + off.dx;
    const ty = cand.cy + off.dy;
    const box: LabelBox = {
      x0: off.anchor === "start" ? tx - 2 : tx - widthEst,
      x1: off.anchor === "start" ? tx + widthEst : tx + 2,
      y0: ty - heightEst + 2,
      y1: ty + 3,
    };
    const outOfBounds =
      box.x0 < bounds.x0 || box.x1 > bounds.x1 || box.y0 < bounds.y0 || box.y1 > bounds.y1;
    if (outOfBounds || boxes.some((existing) => boxesOverlap(existing, box))) continue;
    boxes.push(box);
    return {
      id: cand.id,
      x: tx,
      y: ty,
      textAnchor: off.anchor,
      text: cand.text,
      bold: cand.bold,
      fill: cand.fill,
    };
  }
  return null;
}

function placeDeconflictedLabels(args: {
  candidates: LabelCandidate[];
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  maxLabels: number;
}): PlacedLabel[] {
  const { candidates, minX, maxX, minY, maxY, maxLabels } = args;
  const placed: PlacedLabel[] = [];
  const boxes: LabelBox[] = [];
  const bounds: LabelBox = { x0: minX, y0: minY, x1: maxX, y1: maxY };
  for (const cand of candidates) {
    if (placed.length >= maxLabels) break;
    const hit = tryPlaceCandidate({ cand, bounds, boxes });
    if (hit) placed.push(hit);
  }
  return placed;
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

function fitLinearLine(
  points: ArenaForecast["points"],
): { slope: number; intercept: number } | null {
  if (points.length < 2) return null;
  const mx = points.reduce((acc, p) => acc + p.early_wr, 0) / points.length;
  const my = points.reduce((acc, p) => acc + p.peak_wr, 0) / points.length;
  const denom = points.reduce((acc, p) => acc + (p.early_wr - mx) ** 2, 0);
  if (denom <= 1e-9) return null;
  const slope = points.reduce((acc, p) => acc + (p.early_wr - mx) * (p.peak_wr - my), 0) / denom;
  return { slope, intercept: my - slope * mx };
}

function buildForecastHighlightLabels(args: {
  points: ArenaForecast["points"];
  topRunId: string;
  sx: (pct: number) => number;
  sy: (pct: number) => number;
}): PlacedLabel[] {
  const { points, topRunId, sx, sy } = args;
  const byPeakDesc = points.toSorted((a, b) => b.peak_wr - a.peak_wr);
  const byEarlyAsc = points.toSorted((a, b) => a.early_wr - b.early_wr);
  const picks: ArenaForecast["points"] = [];
  const seen = new Set<string>();
  const addPick = (pt: ArenaForecast["points"][number] | undefined): void => {
    if (!pt || seen.has(pt.run_id)) return;
    seen.add(pt.run_id);
    picks.push(pt);
  };
  addPick(points.find((p) => p.run_id === topRunId));
  addPick(byPeakDesc.at(-1));
  addPick(byEarlyAsc.at(-1));
  addPick(byEarlyAsc[0]);
  if (byPeakDesc.length >= 3) {
    addPick(byPeakDesc[Math.floor(byPeakDesc.length / 2)]);
  }
  return placeDeconflictedLabels({
    candidates: picks.map((pt) => {
      const isTop = pt.run_id === topRunId;
      return {
        id: pt.run_id,
        cx: sx(pt.early_wr * 100),
        cy: sy(pt.peak_wr * 100),
        text: truncateLabel(pt.run_id, 16),
        bold: isTop,
        fill: isTop ? "var(--text)" : "var(--text-2)",
      };
    }),
    minX: 40,
    maxX: 422,
    minY: 12,
    maxY: 176,
    maxLabels: 4,
  });
}

function ForecastSvg(props: { forecast: ArenaForecast; topRunId: string }): ReactNode {
  const { points } = props.forecast;
  const xs = points.map((p) => p.early_wr * 100);
  const ys = points.map((p) => p.peak_wr * 100);
  const xMin = Math.max(0, Math.floor((Math.min(...xs) - 4) / 5) * 5);
  const xMax = Math.min(100, Math.ceil((Math.max(...xs) + 4) / 5) * 5);
  const yMin = Math.max(0, Math.floor((Math.min(...ys) - 4) / 5) * 5);
  const yMax = Math.min(100, Math.ceil((Math.max(...ys) + 4) / 5) * 5);
  const xSpan = Math.max(10, xMax - xMin);
  const ySpan = Math.max(10, yMax - yMin);

  const sx = (pct: number): number => 42 + ((pct - xMin) / xSpan) * 366;
  const sy = (pct: number): number => 174 - ((pct - yMin) / ySpan) * 150;
  const xTicks = computeNiceTicks(xMin, xMax, xSpan > 35 ? 10 : 5);
  const yTicks = computeNiceTicks(yMin, yMax, ySpan > 35 ? 10 : 5);
  const line = fitLinearLine(points);
  const highlightLabels = buildForecastHighlightLabels({
    points,
    topRunId: props.topRunId,
    sx,
    sy,
  });

  return (
    <svg
      viewBox="0 0 430 210"
      className="w-full h-[198px] block mt-1.5"
      role="img"
      aria-label="Early-run u200 vs peak win rate scatter plot"
    >
      {yTicks.map((yv) => {
        const y = sy(yv);
        return (
          <g key={yv}>
            <line
              x1={38}
              y1={y}
              x2={414}
              y2={y}
              stroke="var(--border)"
              strokeDasharray="2 3"
              strokeWidth={1}
            />
            <text
              x={33}
              y={y + 3}
              textAnchor="end"
              fontSize={9.5}
              fill="var(--text-3)"
              className="tabular"
            >
              {yv}%
            </text>
          </g>
        );
      })}
      {xTicks.map((xv) => (
        <text
          key={xv}
          x={sx(xv)}
          y={191}
          textAnchor="middle"
          fontSize={9.5}
          fill="var(--text-3)"
          className="tabular"
        >
          {xv}%
        </text>
      ))}
      {line ? (
        <line
          x1={sx(xMin)}
          y1={sy((line.intercept + line.slope * (xMin / 100)) * 100)}
          x2={sx(xMax)}
          y2={sy((line.intercept + line.slope * (xMax / 100)) * 100)}
          stroke="var(--accent)"
          strokeWidth={1.5}
          strokeDasharray="4 3"
          opacity={0.75}
        />
      ) : null}
      {points.map((pt) => {
        const cx = sx(pt.early_wr * 100);
        const cy = sy(pt.peak_wr * 100);
        const isTop = pt.run_id === props.topRunId;
        return (
          <circle
            key={pt.run_id}
            cx={cx}
            cy={cy}
            r={isTop ? 4.2 : 3}
            fill={isTop ? "var(--accent)" : "var(--text-2)"}
            opacity={isTop ? 1 : 0.72}
            stroke="var(--surface)"
            strokeWidth={1.2}
          >
            <title>
              {`${pt.run_id}: u200 ${formatPct(pt.early_wr)} -> peak ${formatPct(pt.peak_wr)}`}
            </title>
          </circle>
        );
      })}
      {highlightLabels.map((lbl) => (
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
      <text x={226} y={206} textAnchor="middle" fontSize={9} fill="var(--text-3)">
        u200 Eval Win% vs Final Peak Win%
      </text>
    </svg>
  );
}

export function EarlyRunForecastChart(props: { forecast: ArenaForecast | null }): ReactNode {
  const { forecast } = props;
  const topPoint = forecast
    ? (forecast.points.toSorted((a, b) => b.peak_wr - a.peak_wr)[0] ?? null)
    : null;

  return (
    <section className="hidden md:flex rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3.5 py-3 flex-col">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] font-semibold uppercase tracking-[0.04em] text-[var(--text-3)]">
          Early-Run Forecast (u200 vs Peak)
        </div>
        {forecast ? (
          <span className="tabular text-[11px] text-[var(--text-3)]">
            n={forecast.n} · r={forecast.pearson_r.toFixed(3)} · rho=
            {forecast.spearman_rho.toFixed(3)}
          </span>
        ) : null}
      </div>

      {forecast && topPoint ? (
        <>
          <div className="flex items-center flex-wrap gap-x-3 gap-y-1 mt-1.5 text-[11px] text-[var(--text-2)]">
            <span className="inline-flex items-center gap-1.5">
              <span className="w-[7px] h-[7px] rounded-full bg-[var(--accent)] shrink-0" />
              <span>{topPoint.run_id} peak</span>
              <span className="tabular font-semibold text-[var(--text)]">
                {formatPct(topPoint.peak_wr)}
              </span>
            </span>
            {forecast.power_law_pred === null ? null : (
              <span className="inline-flex items-center gap-1.5">
                <span className="w-[7px] h-[7px] rounded-full bg-[var(--text-3)] shrink-0" />
                <span>Power-law fit</span>
                <span className="tabular font-semibold text-[var(--text)]">
                  {formatPct(forecast.power_law_pred)}
                </span>
              </span>
            )}
          </div>
          <ForecastSvg forecast={forecast} topRunId={topPoint.run_id} />
        </>
      ) : (
        <div className="h-[198px] flex items-center justify-center text-xs text-[var(--text-3)] text-center px-4">
          Need at least 2 runs with early (u200) and peak eval metrics to display the forecast.
        </div>
      )}
    </section>
  );
}

/** Early-run u200 vs final peak win rate scatter plot and power-law summary for Elo Arena. */

import type { ReactNode } from "react";
import type { ArenaForecast } from "@/lib/types";
import {
  type PlacedLabel,
  computeNiceTicks,
  formatPct,
  placeDeconflictedLabels,
  truncateLabel,
} from "./benchmarksUtils";

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

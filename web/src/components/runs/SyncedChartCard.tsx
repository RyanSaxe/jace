/** Synced SVG telemetry chart card with axes, CI bands, EMA overlay, and crosshair readout. */

import type { MouseEvent, ReactNode } from "react";
import {
  type AxisBounds,
  type ChartGeom,
  type PreparedSeries,
  buildChartGeom,
  buildCiPolygonPoints,
  computeAxisBounds,
  formatXTick,
  nearestPoint,
  prepareAllSeries,
} from "./chartMath";
import { type ChartSeriesInput, type XAxisMode, formatMetricValue } from "./metrics";

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

/** KPI summary strip and SVG analytics charts for the Compare Models workbench. */

import { X } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/Primitives";
import type { CompareDecisionKindStat, ComparePositionRow, CompareResponse } from "@/lib/types";
import { CARD, LBL, PILL_ON, dotVisual, humanizeToken } from "./compareUtils";

export function SummaryKpiStrip(props: { result: CompareResponse }): ReactNode {
  const r = props.result;
  const kpis: ReadonlyArray<[string, string, string]> = [
    ["Sampled Turns", String(r.sampled_positions), ""],
    ["Moves Changed", String(r.moves_changed), `${(r.change_rate * 100).toFixed(1)}%`],
    ["Mean TV", r.mean_tv.toFixed(3), `KL ${r.mean_kl.toFixed(2)}`],
    ["Mean Win% Shift", `${(r.mean_win_delta * 100).toFixed(1)} pp`, ""],
    ["Device", r.device || "cpu", ""],
  ];
  return (
    <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
      {kpis.map(([label, val, sub], idx) => (
        <div key={label} className={`${CARD} px-3 py-2 flex flex-col gap-0.5`}>
          <span className={LBL}>{label}</span>
          <div className="flex items-baseline gap-1.5">
            <span
              className={`tabular text-base font-semibold leading-tight ${idx === 1 && r.moves_changed > 0 ? "text-[var(--accent)]" : ""}`}
            >
              {val}
            </span>
            {sub ? <span className="tabular text-[11px] text-[var(--text-3)]">{sub}</span> : null}
          </div>
        </div>
      ))}
    </div>
  );
}

export function DecisionKindChart(props: {
  stats: CompareDecisionKindStat[];
  activeKind: string | null;
  onToggle: (k: string) => void;
}): ReactNode {
  const sorted = props.stats.toSorted((a, b) => b.change_rate - a.change_rate || b.count - a.count);
  const maxRate = Math.max(0.05, ...sorted.map((s) => s.change_rate));
  return (
    <div className={`${CARD} p-3 flex flex-col gap-2`}>
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold m-0">Disagreement by Decision Kind</h3>
        {props.activeKind ? (
          <Button
            variant="ghost"
            onClick={(): void => props.onToggle(props.activeKind ?? "")}
            className="text-[10.5px] px-2 py-0.5"
          >
            Clear <X className="w-3 h-3" />
          </Button>
        ) : (
          <span className="text-[10.5px] text-[var(--text-3)]">Click to filter</span>
        )}
      </div>
      {sorted.length === 0 ? (
        <div className="py-6 text-center text-xs text-[var(--text-3)]">
          No decision-kind breakdown.
        </div>
      ) : (
        <div className="flex flex-col gap-1.5 max-h-[196px] overflow-y-auto pr-0.5">
          {sorted.map((s) => {
            const label = humanizeToken(s.kind);
            const active = props.activeKind === s.kind;
            const pct = `${Math.round(s.change_rate * 100)}%`;
            const barW = Math.max(2, (s.change_rate / maxRate) * 100);
            const cls = active
              ? PILL_ON
              : "border-transparent bg-[var(--bg)] hover:border-[var(--border-strong)]";
            return (
              <button
                key={s.kind}
                type="button"
                aria-label={`Filter by ${label}: ${pct} move changed`}
                onClick={(): void => props.onToggle(s.kind)}
                className={`w-full text-left rounded-md border px-2 py-1.5 cursor-pointer ${cls}`}
              >
                <div className="flex justify-between gap-2 text-[11px] mb-1">
                  <span className="font-medium truncate text-[var(--text)]">{label}</span>
                  <span className="tabular text-[10.5px] text-[var(--text-2)] shrink-0">
                    {pct} ({s.changed}/{s.count}) · TV {s.mean_tv.toFixed(2)}
                  </span>
                </div>
                <svg viewBox="0 0 100 6" className="w-full h-1.5 block" preserveAspectRatio="none">
                  <rect width="100" height="6" rx="3" fill="var(--surface-2)" />
                  <rect
                    width={barW}
                    height="6"
                    rx="3"
                    fill="var(--accent)"
                    opacity={active ? 1 : 0.82}
                  />
                </svg>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function WinScatterChart(props: {
  positions: ComparePositionRow[];
  selectedKey: string | null;
  onSelect: (k: string) => void;
}): ReactNode {
  const [w, h, l, r, t, b] = [280, 186, 32, 12, 10, 24];
  const [pw, ph] = [w - l - r, h - t - b];
  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((v) => `M${l} ${t + (1 - v) * ph}H${w - r}M${l + v * pw} ${t}V${h - b}`)
    .join("");
  const dots = props.positions
    .map((p) => {
      const key = `${p.run_id}:${p.iteration}:${p.game_idx}:${p.step}`;
      const sel = key === props.selectedKey;
      const d = dotVisual(p, sel);
      return {
        key,
        sel,
        d,
        cx: l + Math.max(0, Math.min(1, p.win_a)) * pw,
        cy: t + (1 - Math.max(0, Math.min(1, p.win_b))) * ph,
        tip: `T${p.turn} #${p.step} · ${humanizeToken(p.decision_kind)} | A ${(p.win_a * 100).toFixed(1)}% (${p.top_action_a}) vs B ${(p.win_b * 100).toFixed(1)}% (${p.top_action_b})`,
      };
    })
    .toSorted((x, y) => x.d.prio - y.d.prio);
  const midY = t + ph / 2;

  return (
    <div className={`${CARD} p-3 flex flex-col gap-2`}>
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold m-0">Hero Win% Scatter (Model A vs Model B)</h3>
        <div className="flex items-center gap-2 text-[10.5px] shrink-0">
          <span className="text-[var(--accent)]">● Move changed</span>
          <span className="text-[var(--text-3)]">● Same move</span>
        </div>
      </div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full h-[186px] block select-none"
        role="img"
        aria-label="Hero win percentage scatter plot comparing Model A and Model B"
      >
        <path d={grid} stroke="var(--border)" strokeWidth="0.8" fill="none" />
        <line
          x1={l}
          y1={t + ph}
          x2={l + pw}
          y2={t}
          stroke="var(--border-strong)"
          strokeDasharray="3 3"
          strokeWidth="1.2"
        />
        {[0, 0.5, 1].map((v) => (
          <g key={v} fontSize="8.5" fill="var(--text-3)">
            <text x={l - 5} y={t + (1 - v) * ph + 3} textAnchor="end">
              {Math.round(v * 100)}%
            </text>
            <text x={l + v * pw} y={h - b + 11} textAnchor="middle">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        <text x={l + pw / 2} y={h - 2} textAnchor="middle" fontSize="8.5" fill="var(--text-2)">
          Model A Win%
        </text>
        <text
          x={9}
          y={midY}
          textAnchor="middle"
          fontSize="8.5"
          fill="var(--text-2)"
          transform={`rotate(-90 9 ${midY})`}
        >
          Model B Win%
        </text>
        {dots.map((dot) => (
          <circle
            key={dot.key}
            cx={dot.cx}
            cy={dot.cy}
            r={dot.d.r}
            fill={dot.d.fill}
            fillOpacity={dot.d.op}
            stroke={dot.sel ? "var(--text)" : "var(--surface)"}
            strokeWidth={dot.sel ? 1.8 : 0.7}
            className="cursor-pointer"
            onClick={(): void => props.onSelect(dot.key)}
          >
            <title>{dot.tip}</title>
          </circle>
        ))}
      </svg>
    </div>
  );
}

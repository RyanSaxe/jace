/** 3-Pane Compare Models workbench: cross-run checkpoint scanner, SVG analytics, diff table, and docked position preview. */

import { ArrowRight, GitCompareArrows, Play, SlidersHorizontal, Square, X } from "lucide-react";
import { type ReactNode, useRef, useState } from "react";
import { Badge, Button, SegmentedControl } from "@/components/ui/Primitives";
import { useAbortOnUnmount } from "@/hooks/useStudioData";
import { cardImageUrl, postCompareModels } from "@/lib/api";
import type {
  CompareDecisionKindStat,
  ComparePositionRow,
  CompareResponse,
  CompareTarget,
  GameTarget,
  GlobalCheckpointOption,
  RunSummary,
} from "@/lib/types";

export interface CompareViewProps {
  runs: RunSummary[];
  checkpoints: GlobalCheckpointOption[];
  initialTarget: CompareTarget | null;
  onOpenTabletop: (target: GameTarget) => void;
}

type StatusFilter = "all" | "changed";
type SortMode = "tv_desc" | "win_delta_desc" | "turn_asc";
type FilterState = { status: StatusFilter; kind: string | null; sort: SortMode };
type CheckpointGroup = { runId: string; label: string; options: GlobalCheckpointOption[] };

const INIT_FILTER: FilterState = { status: "all", kind: null, sort: "tv_desc" };
const SORT_OPTIONS: ReadonlyArray<{ value: SortMode; label: string }> = [
  { value: "tv_desc", label: "Largest TV" },
  { value: "win_delta_desc", label: "Largest Win% Shift" },
  { value: "turn_asc", label: "Turn" },
];
const CARD = "rounded-lg border border-[var(--border)] bg-[var(--surface)]";
const BOX = "rounded-md border border-[var(--border)] bg-[var(--bg)]";
const SEL = `h-7 ${BOX} px-2 text-[11.5px] text-[var(--text)] truncate cursor-pointer`;
const PILL_ON = "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)] font-semibold";
const PILL_OFF =
  "border-[var(--border)] bg-[var(--bg)] text-[var(--text-2)] hover:text-[var(--text)]";
const PILL =
  "px-2 py-0.5 rounded-md text-[11px] font-medium border cursor-pointer transition-colors";
const PRESET =
  "tabular h-6 px-1.5 rounded text-[10.5px] font-medium border cursor-pointer transition-colors";
const ASIDE =
  "w-full lg:w-[320px] shrink-0 border-t lg:border-t-0 lg:border-l border-[var(--border)] bg-[var(--surface)]";
const LBL = "text-[10px] font-semibold uppercase tracking-wider text-[var(--text-3)]";

function posKey(r: ComparePositionRow): string {
  return `${r.run_id}:${r.iteration}:${r.game_idx}:${r.step}`;
}

function humanizeToken(raw: string): string {
  const s = raw.trim();
  if (!s) return "Decision";
  if (!s.includes("_") && s !== s.toUpperCase()) return s;
  return s
    .toLowerCase()
    .split("_")
    .filter(Boolean)
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join(" ");
}

function clampSamples(raw: string): number {
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n >= 1 ? n : 120;
}

function signedDelta(delta: number): string {
  const pts = delta * 100;
  return `${pts > 0.05 ? "+" : ""}${pts.toFixed(1)}%`;
}

function shortRef(ref: string): string {
  if (ref === "heuristic") return "Heuristic baseline";
  const parts = ref.split("::");
  return parts.length === 2
    ? `${(parts[0] ?? "").split("/").pop() ?? ""} · ${parts[1] ?? ""}`
    : ref;
}

function groupCheckpoints(checkpoints: GlobalCheckpointOption[]): CheckpointGroup[] {
  const map = new Map<string, CheckpointGroup>();
  for (const c of checkpoints) {
    const g = map.get(c.run_id);
    if (g) g.options.push(c);
    else map.set(c.run_id, { runId: c.run_id, label: c.run_name || c.run_id, options: [c] });
  }
  return Array.from(map.values());
}

function pickRef(ckpts: GlobalCheckpointOption[], pref: string | undefined, skip?: string): string {
  if (pref) return pref;
  return (skip ? ckpts.find((c) => c.ref !== skip) : ckpts[0])?.ref ?? "heuristic";
}

function resolveSelection(
  props: CompareViewProps,
  ov: OvState,
): { ckptA: string; ckptB: string; runId: string; targetKey: string } {
  const t = props.initialTarget;
  const defA = pickRef(props.checkpoints, t?.ckptA);
  const initRun = t?.runId ?? "";
  return {
    ckptA: ov.a ?? defA,
    ckptB: ov.b ?? pickRef(props.checkpoints, t?.ckptB, defA),
    runId: ov.run ?? initRun,
    targetKey: t ? `${t.ckptA}|${t.ckptB}|${initRun}` : "",
  };
}

function dotVisual(
  pos: ComparePositionRow,
  sel: boolean,
): { prio: number; r: number; fill: string; op: number } {
  const fill = pos.move_changed ? "var(--accent)" : "var(--text-3)";
  if (sel) return { prio: 2, r: 5.2, fill, op: 1 };
  if (pos.move_changed) return { prio: 1, r: 3.8, fill, op: 0.85 };
  return { prio: 0, r: 2.9, fill, op: 0.42 };
}

function filterAndSort(positions: ComparePositionRow[], filter: FilterState): ComparePositionRow[] {
  return positions
    .filter((r) => {
      if (filter.status === "changed" && !r.move_changed) return false;
      if (filter.kind && r.decision_kind !== filter.kind) return false;
      return true;
    })
    .toSorted((a, b) => {
      if (filter.sort === "win_delta_desc") {
        return Math.abs(b.win_delta) - Math.abs(a.win_delta) || b.tv - a.tv;
      }
      if (filter.sort === "turn_asc") return a.turn - b.turn || a.step - b.step;
      return b.tv - a.tv || Math.abs(b.win_delta) - Math.abs(a.win_delta);
    });
}

function CheckpointPicker(props: {
  label: string;
  dot: string;
  value: string;
  groups: CheckpointGroup[];
  onChange: (v: string) => void;
}): ReactNode {
  const hasVal = props.groups.some((g) => g.options.some((o) => o.ref === props.value));
  const custom = Boolean(props.value) && props.value !== "heuristic" && !hasVal;
  return (
    <label className="flex items-center gap-1.5 text-[11px] text-[var(--text-2)] min-w-0">
      <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: props.dot }} />
      <span className="font-semibold text-[var(--text)] shrink-0">{props.label}</span>
      <select
        aria-label={props.label}
        value={props.value}
        onChange={(e): void => props.onChange(e.target.value)}
        className={`${SEL} max-w-[190px]`}
      >
        {custom ? <option value={props.value}>{shortRef(props.value)}</option> : null}
        {props.groups.map((g) => (
          <optgroup key={g.runId} label={g.label}>
            {g.options.map((o) => (
              <option key={o.ref} value={o.ref}>
                {o.iteration >= 0 ? `${o.checkpoint} (u${o.iteration})` : o.checkpoint}
              </option>
            ))}
          </optgroup>
        ))}
        <option value="heuristic">Heuristic baseline</option>
      </select>
    </label>
  );
}

function SummaryKpiStrip(props: { result: CompareResponse }): ReactNode {
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

function DecisionKindChart(props: {
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

function WinScatterChart(props: {
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

function CompareDiffTable(props: {
  view: ViewData;
  filter: FilterState;
  onFilter: (patch: Partial<FilterState>) => void;
  onSelect: (k: string) => void;
  onOpenRow: (r: ComparePositionRow) => void;
}): ReactNode {
  const { view, filter } = props;
  const btns: ReadonlyArray<[StatusFilter, string]> = [
    ["all", `All (${view.total})`],
    ["changed", `Move changed only (${view.changed})`],
  ];
  return (
    <div className={`${CARD} overflow-hidden flex flex-col`}>
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-[var(--border)]">
        <div className="flex flex-wrap items-center gap-1.5">
          {btns.map(([val, label]) => (
            <button
              key={val}
              type="button"
              onClick={(): void => props.onFilter({ status: val })}
              className={`${PILL} ${filter.status === val ? PILL_ON : PILL_OFF}`}
            >
              {label}
            </button>
          ))}
          {filter.kind ? (
            <span className={`inline-flex items-center gap-1 ${PILL} ${PILL_ON}`}>
              <SlidersHorizontal className="w-3 h-3" /> {humanizeToken(filter.kind)}
              <button
                type="button"
                aria-label="Clear decision kind filter"
                onClick={(): void => props.onFilter({ kind: null })}
                className="ml-0.5 cursor-pointer"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          ) : null}
        </div>
        <SegmentedControl
          options={SORT_OPTIONS}
          value={filter.sort}
          onChange={(sort): void => props.onFilter({ sort })}
        />
      </div>
      {view.filteredRows.length === 0 ? (
        <div className="p-8 text-center text-xs text-[var(--text-3)]">
          No sampled positions match the current filters.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-[11.5px]">
            <thead>
              <tr className={`border-b border-[var(--border)] bg-[var(--bg)] ${LBL}`}>
                {[
                  "Turn / Step",
                  "Decision",
                  "Status",
                  "Model A Top Move",
                  "Model B Top Move",
                  "TV",
                  "Win Δ",
                ].map((lbl, i) => (
                  <th
                    key={lbl}
                    className={`py-1.5 px-2.5 whitespace-nowrap ${i >= 5 ? "text-right" : ""}`}
                  >
                    {lbl}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {view.filteredRows.map((row) => {
                const key = `${row.run_id}:${row.iteration}:${row.game_idx}:${row.step}`;
                const bg =
                  key === view.activeKey
                    ? "bg-[var(--accent-soft)]"
                    : "hover:bg-[var(--surface-2)]/70";
                const moves: ReadonlyArray<[string, string, number, string]> = [
                  ["Model A", row.top_action_a, row.top_prob_a, "text-[var(--accent)]"],
                  ["Model B", row.top_action_b, row.top_prob_b, "text-[var(--warn)]"],
                ];
                return (
                  <tr
                    key={key}
                    onClick={(): void => props.onSelect(key)}
                    onDoubleClick={(): void => props.onOpenRow(row)}
                    title="Click to preview · Double-click to open in Tabletop"
                    className={`border-b border-[var(--border)] cursor-pointer transition-colors ${bg}`}
                  >
                    <td className="py-1.5 px-2.5 tabular whitespace-nowrap font-medium">
                      T{row.turn} <span className="text-[var(--text-3)]">#{row.step}</span>
                    </td>
                    <td className="py-1.5 px-2.5 whitespace-nowrap">
                      {humanizeToken(row.decision_kind)}{" "}
                      <span className="ml-1 text-[10px] uppercase text-[var(--text-3)]">
                        {row.actor_side}
                      </span>
                    </td>
                    <td className="py-1.5 px-2.5 whitespace-nowrap">
                      <Badge tone={row.move_changed ? "accent" : "neutral"}>
                        {row.move_changed ? "Move changed" : "Same move"}
                      </Badge>
                    </td>
                    {moves.map(([side, act, prob, color]) => (
                      <td
                        key={side}
                        className="py-1.5 px-2.5 max-w-[180px]"
                        aria-label={`${side}: ${act} (${Math.round(prob * 100)}%)`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="truncate" title={act}>
                            {act}
                          </span>
                          <span className={`tabular text-[10.5px] font-semibold shrink-0 ${color}`}>
                            {Math.round(prob * 100)}%
                          </span>
                        </div>
                      </td>
                    ))}
                    <td className="py-1.5 px-2.5 text-right tabular font-medium">
                      {row.tv.toFixed(2)}
                    </td>
                    <td className="py-1.5 px-2.5 text-right tabular text-[var(--text-2)]">
                      {signedDelta(row.win_delta)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function LegalMoveBars(props: { position: ComparePositionRow }): ReactNode {
  const rows = props.position.legal_actions
    .map((action, i) => ({
      action,
      a: props.position.probs_a[i] ?? 0,
      b: props.position.probs_b[i] ?? 0,
    }))
    .toSorted((x, y) => Math.max(y.a, y.b) - Math.max(x.a, x.b));
  const [maxA, maxB] = [
    Math.max(-1, ...rows.map((r) => r.a)),
    Math.max(-1, ...rows.map((r) => r.b)),
  ];
  if (rows.length === 0) {
    return (
      <div className={`${BOX} p-3 text-xs text-[var(--text-3)]`}>
        No legal action probabilities recorded.
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <div className={`flex items-center justify-between ${LBL}`}>
        <span>Legal Moves ({rows.length})</span>
        <span className="inline-flex gap-2 normal-case font-medium">
          <span className="text-[var(--accent)]">● A</span>
          <span className="text-[var(--warn)]">● B</span>
        </span>
      </div>
      {rows.map(({ action, a, b }) => {
        const bars: ReadonlyArray<[string, number, string, boolean]> = [
          ["A", a, "var(--accent)", a > 0 && a === maxA],
          ["B", b, "var(--warn)", b > 0 && b === maxB],
        ];
        return (
          <div key={`${action.index}-${action.label}`} className={`${BOX} p-2 flex flex-col gap-1`}>
            <div className="text-[11.5px] font-medium leading-snug break-words">{action.label}</div>
            {bars.map(([lbl, prob, clr, top]) => {
              const pct = Math.max(0, Math.min(100, prob * 100)).toFixed(1);
              return (
                <div
                  key={lbl}
                  className="grid grid-cols-[14px_1fr_40px] items-center gap-1.5 text-[10.5px]"
                >
                  <span className="font-semibold" style={{ color: clr }}>
                    {lbl}
                  </span>
                  <div className="h-2 rounded-full bg-[var(--surface-2)] overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${pct}%`, backgroundColor: clr, opacity: top ? 1 : 0.72 }}
                    />
                  </div>
                  <span
                    className={`tabular text-right ${top ? "font-semibold" : "text-[var(--text-2)]"}`}
                  >
                    {pct}%
                  </span>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

function PositionPreviewPane(props: {
  position: ComparePositionRow | null;
  onOpen: (pos: ComparePositionRow) => void;
}): ReactNode {
  const pos = props.position;
  if (!pos) {
    return (
      <aside
        className={`${ASIDE} p-4 flex items-center justify-center text-center text-xs text-[var(--text-3)]`}
      >
        Select a sampled position from the scatter plot or diff table to inspect legal moves and
        jump into Tabletop.
      </aside>
    );
  }
  return (
    <aside className={`${ASIDE} flex flex-col min-h-0`}>
      <div className="flex-1 lg:overflow-y-auto p-3.5 flex flex-col gap-3">
        <div className="flex flex-col gap-1.5 pb-3 border-b border-[var(--border)]">
          <div className="flex items-center justify-between gap-2">
            <span className="tabular text-xs font-semibold">
              Turn {pos.turn} · Step #{pos.step}
            </span>
            <Badge tone={pos.move_changed ? "accent" : "neutral"}>
              {pos.move_changed ? "Move changed" : "Same move"}
            </Badge>
          </div>
          <div className="text-[11px] text-[var(--text-2)]">
            <strong className="font-medium text-[var(--text)]">
              {humanizeToken(pos.decision_kind)}
            </strong>
            <span className="tabular">
              {" "}
              · Seed #{pos.seed} · u{pos.iteration}
            </span>
          </div>
          <div className={`flex items-center justify-between text-[11px] ${BOX} px-2.5 py-1.5`}>
            <span className="text-[var(--text-2)]">
              Hero <strong className="tabular text-[var(--text)]">{pos.hero_life}</strong> vs Opp{" "}
              <strong className="tabular text-[var(--text)]">{pos.opp_life}</strong>
            </span>
            <span className="tabular text-[10.5px] text-[var(--text-3)]">
              TV {pos.tv.toFixed(2)}
            </span>
          </div>
          <div
            className={`flex items-center justify-between text-[11px] ${BOX} px-2.5 py-1.5 tabular`}
          >
            <span>
              <span className="text-[var(--accent)] font-semibold">A</span>{" "}
              {(pos.win_a * 100).toFixed(1)}%
            </span>
            <span>
              <span className="text-[var(--warn)] font-semibold">B</span>{" "}
              {(pos.win_b * 100).toFixed(1)}%{" "}
              <span className="text-[10px] text-[var(--text-3)]">
                ({signedDelta(pos.win_delta)})
              </span>
            </span>
          </div>
          <div className="text-[10.5px] text-[var(--text-3)] truncate" title={pos.run_id}>
            {pos.run_id}
          </div>
        </div>
        {pos.key_cards.length > 0 ? (
          <div className="flex flex-col gap-1.5">
            <span className={LBL}>Key Cards on Board</span>
            <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5">
              {pos.key_cards.slice(0, 5).map((name) => (
                <div
                  key={name}
                  title={name}
                  className="relative w-[54px] aspect-[5/7] shrink-0 rounded overflow-hidden border border-[var(--border)] bg-[var(--surface-2)] flex items-center justify-center p-1 text-center text-[9px] text-[var(--text-2)]"
                >
                  <span className="line-clamp-3">{name}</span>
                  <img
                    src={cardImageUrl(name)}
                    alt={name}
                    loading="lazy"
                    onError={(e): void => {
                      e.currentTarget.style.display = "none";
                    }}
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                </div>
              ))}
            </div>
          </div>
        ) : null}
        <LegalMoveBars position={pos} />
      </div>
      <div className="p-3 border-t border-[var(--border)] shrink-0">
        <Button
          variant="primary"
          size="md"
          className="w-full"
          onClick={(): void => props.onOpen(pos)}
        >
          Open in Tabletop (A vs B) <ArrowRight className="w-3.5 h-3.5" />
        </Button>
      </div>
    </aside>
  );
}

type OvState = { a: string | null; b: string | null; run: string | null };

function CompareToolbar(props: {
  sel: { ckptA: string; ckptB: string; runId: string };
  sampleText: string;
  loading: boolean;
  runs: RunSummary[];
  checkpoints: GlobalCheckpointOption[];
  onPatchOv: (patch: Partial<OvState>) => void;
  onSamples: (v: string) => void;
  onRun: () => void;
  onStop: () => void;
}): ReactNode {
  const groups = groupCheckpoints(props.checkpoints);
  const parsed = clampSamples(props.sampleText);
  const { sel } = props;
  return (
    <div className="min-h-[44px] px-3 py-1.5 border-b border-[var(--border)] bg-[var(--surface)] flex flex-wrap items-center gap-x-2.5 gap-y-1.5 shrink-0">
      <div className="flex items-center gap-1.5 pr-1">
        <GitCompareArrows className="w-4 h-4 text-[var(--accent)] shrink-0" />
        <h2 className="text-xs font-semibold tracking-tight m-0">Compare</h2>
      </div>
      <CheckpointPicker
        label="Model A"
        dot="var(--accent)"
        value={sel.ckptA}
        groups={groups}
        onChange={(a): void => props.onPatchOv({ a })}
      />
      <span className="text-[10.5px] text-[var(--text-3)]">vs</span>
      <CheckpointPicker
        label="Model B"
        dot="var(--warn)"
        value={sel.ckptB}
        groups={groups}
        onChange={(b): void => props.onPatchOv({ b })}
      />
      <select
        aria-label="Source run filter"
        value={sel.runId}
        onChange={(e): void => props.onPatchOv({ run: e.target.value })}
        className={`${SEL} max-w-[168px]`}
      >
        <option value="">All playable games</option>
        {props.runs.map((r) => (
          <option key={r.run_id} value={r.run_id}>
            {r.name || r.run_id}
          </option>
        ))}
      </select>
      <label className="flex items-center gap-1 text-[11px] text-[var(--text-2)]">
        Samples
        <input
          type="number"
          min={1}
          value={props.sampleText}
          onChange={(e): void => props.onSamples(e.target.value)}
          onBlur={(): void => props.onSamples(String(parsed))}
          onKeyDown={(e): void => {
            if (e.key === "Enter" && !props.loading) props.onRun();
          }}
          aria-label="Sample count"
          className={`tabular h-7 w-[62px] ${BOX} px-1.5 text-[11.5px] text-[var(--text)]`}
        />
      </label>
      <div className="hidden sm:flex items-center gap-0.5">
        {[50, 120, 250, 500].map((n) => (
          <button
            key={n}
            type="button"
            onClick={(): void => props.onSamples(String(n))}
            className={`${PRESET} ${parsed === n && props.sampleText.trim() !== "" ? PILL_ON : PILL_OFF}`}
          >
            {n}
          </button>
        ))}
      </div>
      <Button
        variant={props.loading ? "default" : "primary"}
        onClick={props.loading ? props.onStop : props.onRun}
        disabled={!props.loading && (!sel.ckptA || !sel.ckptB)}
        className={`ml-auto h-7 ${props.loading ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]" : ""}`}
      >
        {props.loading ? (
          <>
            <span className="w-2 h-2 rounded-full bg-[var(--accent)] animate-pulse" />
            <Square className="w-3 h-3 fill-current" /> Stop
          </>
        ) : (
          <>
            <Play className="w-3 h-3 fill-current" /> Run Compare
          </>
        )}
      </Button>
    </div>
  );
}

type ScanState = { loading: boolean; stopped: boolean; error: string | null };
const INIT_SCAN: ScanState = { loading: false, stopped: false, error: null };

function CompareNotices(props: {
  scan: ScanState;
  warnings: string[];
  scanLabel: string;
}): ReactNode {
  const { scan, warnings } = props;
  if (!scan.loading && !scan.stopped && !scan.error && warnings.length === 0) return null;
  const cls = "rounded-md border px-3 py-1.5";
  return (
    <div className="flex flex-col gap-2 text-xs">
      {scan.loading ? (
        <div
          className={`${cls} border-[var(--accent)]/40 bg-[var(--accent-soft)] flex flex-col gap-1.5`}
        >
          <div className="flex items-center justify-between gap-2 text-[var(--accent)] font-medium">
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[var(--accent)] animate-ping" />
              {props.scanLabel}
            </span>
            <span className="text-[11px] font-normal text-[var(--text-2)]">Stop anytime</span>
          </div>
          <div className="h-1.5 rounded-full bg-[var(--surface)] overflow-hidden">
            <div className="h-full w-1/2 rounded-full bg-[var(--accent)] animate-pulse" />
          </div>
        </div>
      ) : null}
      {scan.stopped && !scan.loading ? (
        <div className={`${cls} border-[var(--border)] bg-[var(--surface)] text-[var(--text-2)]`}>
          Comparison scan stopped.
        </div>
      ) : null}
      {scan.error ? (
        <div className={`${cls} border-[var(--loss)] bg-[var(--loss-soft)] text-[var(--loss)]`}>
          {scan.error}
        </div>
      ) : null}
      {warnings.length > 0 ? (
        <div className={`${cls} border-[var(--warn)]/40 bg-[var(--warn-soft)] text-[var(--warn)]`}>
          {warnings.join(" · ")}
        </div>
      ) : null}
    </div>
  );
}

type ViewData = {
  total: number;
  changed: number;
  filteredRows: ComparePositionRow[];
  activePos: ComparePositionRow | null;
  activeKey: string | null;
  ckptA: string;
  ckptB: string;
  warnings: string[];
};

function executeCompareScan(
  req: { ckptA: string; ckptB: string; sampleCount: number; runId: string },
  abortRef: React.RefObject<AbortController | null>,
  handlers: { onResult: (r: CompareResponse) => void; onDone: (error: string | null) => void },
): void {
  abortRef.current?.abort();
  const controller = new AbortController();
  abortRef.current = controller;
  void postCompareModels({ ...req, runId: req.runId || undefined }, controller.signal)
    .then((res) => {
      if (abortRef.current !== controller) return;
      abortRef.current = null;
      handlers.onResult(res);
    })
    .catch((err: unknown) => {
      if (abortRef.current === controller) abortRef.current = null;
      const aborted =
        controller.signal.aborted || (err instanceof DOMException && err.name === "AbortError");
      const msg = err instanceof Error ? err.message : "Failed to compare";
      handlers.onDone(aborted ? null : msg);
    });
}

function deriveViewData(
  result: CompareResponse | null,
  sel: { ckptA: string; ckptB: string },
  opts: { filter: FilterState; selectedPosKey: string | null },
): ViewData {
  const positions = result ? result.positions : [];
  const filteredRows = filterAndSort(positions, opts.filter);
  const activePos =
    filteredRows.find((r) => posKey(r) === opts.selectedPosKey) ?? filteredRows[0] ?? null;
  return {
    total: positions.length,
    changed: positions.filter((p) => p.move_changed).length,
    filteredRows,
    activePos,
    activeKey: activePos ? posKey(activePos) : null,
    ckptA: result ? result.ckpt_a : sel.ckptA,
    ckptB: result ? result.ckpt_b : sel.ckptB,
    warnings: result ? result.warnings : [],
  };
}

export function CompareView(props: CompareViewProps): ReactNode {
  const [ov, setOv] = useState<OvState>({ a: null, b: null, run: null });
  const sel = resolveSelection(props, ov);
  const [prevKey, setPrevKey] = useState(sel.targetKey);
  const [sampleText, setSampleText] = useState("120");
  const [scan, setScan] = useState<ScanState>({ loading: false, stopped: false, error: null });
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [filter, setFilter] = useState<FilterState>(INIT_FILTER);
  const [selectedPosKey, setSelectedPosKey] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  useAbortOnUnmount(abortRef);
  if (sel.targetKey !== prevKey) {
    setPrevKey(sel.targetKey);
    setOv({ a: null, b: null, run: null });
  }
  const sampleCount = clampSamples(sampleText);
  const view = deriveViewData(result, sel, { filter, selectedPosKey });

  const openPosition = (pos: ComparePositionRow): void => {
    props.onOpenTabletop({
      run_id: pos.run_id,
      iteration: pos.iteration,
      game_idx: pos.game_idx,
      step: pos.step,
      seed: pos.seed,
      ckpt_a: view.ckptA,
      ckpt_b: view.ckptB,
    });
  };

  const runScan = (): void => {
    setSampleText(String(sampleCount));
    setScan({ loading: true, stopped: false, error: null });
    executeCompareScan(
      { ckptA: sel.ckptA, ckptB: sel.ckptB, sampleCount, runId: sel.runId },
      abortRef,
      {
        onResult: (res): void => {
          setResult(res);
          setScan(INIT_SCAN);
          setFilter(INIT_FILTER);
          setSelectedPosKey(null);
        },
        onDone: (error): void => setScan((p) => ({ ...p, loading: false, error })),
      },
    );
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <CompareToolbar
        sel={sel}
        sampleText={sampleText}
        loading={scan.loading}
        runs={props.runs}
        checkpoints={props.checkpoints}
        onPatchOv={(patch): void => setOv((p) => ({ ...p, ...patch }))}
        onSamples={setSampleText}
        onRun={runScan}
        onStop={(): void => {
          abortRef.current?.abort();
          abortRef.current = null;
          setScan({ loading: false, stopped: true, error: null });
        }}
      />
      <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
        <div className="flex-1 min-w-0 lg:overflow-y-auto p-3.5 flex flex-col gap-3">
          <CompareNotices
            scan={scan}
            warnings={view.warnings}
            scanLabel={`Scanning ${sampleCount} sampled turns (${shortRef(sel.ckptA)} vs ${shortRef(sel.ckptB)})…`}
          />
          {result ? (
            <>
              <SummaryKpiStrip result={result} />
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                <DecisionKindChart
                  stats={result.by_decision_kind}
                  activeKind={filter.kind}
                  onToggle={(k): void =>
                    setFilter((p) => ({ ...p, kind: p.kind === k ? null : k }))
                  }
                />
                <WinScatterChart
                  positions={view.filteredRows}
                  selectedKey={view.activeKey}
                  onSelect={setSelectedPosKey}
                />
              </div>
              <CompareDiffTable
                view={view}
                filter={filter}
                onFilter={(patch): void => setFilter((p) => ({ ...p, ...patch }))}
                onSelect={setSelectedPosKey}
                onOpenRow={openPosition}
              />
            </>
          ) : null}
          {!result && !scan.loading ? (
            <div className="rounded-lg border border-dashed border-[var(--border-strong)] bg-[var(--surface)] p-7 text-center flex flex-col items-center gap-2.5 max-w-xl mx-auto my-8">
              <GitCompareArrows className="w-6 h-6 text-[var(--accent)]" />
              <h3 className="text-sm font-semibold m-0">
                Compare any two checkpoints across real game turns
              </h3>
              <p className="text-xs text-[var(--text-2)] m-0">
                Pick Model A and Model B above, set turns to sample, and inspect move changes and
                win% shifts.
              </p>
              <Button variant="primary" size="md" onClick={runScan}>
                <Play className="w-3.5 h-3.5 fill-current" /> Run Compare ({sampleCount} turns)
              </Button>
            </div>
          ) : null}
        </div>
        <PositionPreviewPane position={view.activePos} onOpen={openPosition} />
      </div>
    </div>
  );
}

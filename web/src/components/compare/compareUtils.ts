/** Shared types, style tokens, filtering, and scan helpers for the Compare Models workbench. */

import type { RefObject } from "react";
import { postCompareModels } from "@/lib/api";
import type {
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

export type StatusFilter = "all" | "changed";
export type SortMode = "tv_desc" | "win_delta_desc" | "turn_asc";
export type FilterState = { status: StatusFilter; kind: string | null; sort: SortMode };
export type CheckpointGroup = { runId: string; label: string; options: GlobalCheckpointOption[] };
export type OvState = { a: string | null; b: string | null; run: string | null };
export type ScanState = { loading: boolean; stopped: boolean; error: string | null };
export type ViewData = {
  total: number;
  changed: number;
  filteredRows: ComparePositionRow[];
  activePos: ComparePositionRow | null;
  activeKey: string | null;
  ckptA: string;
  ckptB: string;
  warnings: string[];
};

export const INIT_FILTER: FilterState = { status: "all", kind: null, sort: "tv_desc" };
export const INIT_SCAN: ScanState = { loading: false, stopped: false, error: null };
export const SORT_OPTIONS: ReadonlyArray<{ value: SortMode; label: string }> = [
  { value: "tv_desc", label: "Largest TV" },
  { value: "win_delta_desc", label: "Largest Win% Shift" },
  { value: "turn_asc", label: "Turn" },
];
export const CARD = "rounded-lg border border-[var(--border)] bg-[var(--surface)]";
export const BOX = "rounded-md border border-[var(--border)] bg-[var(--bg)]";
export const SEL = `h-7 ${BOX} px-2 text-[11.5px] text-[var(--text)] truncate cursor-pointer`;
export const PILL_ON =
  "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)] font-semibold";
export const PILL_OFF =
  "border-[var(--border)] bg-[var(--bg)] text-[var(--text-2)] hover:text-[var(--text)]";
export const PILL =
  "px-2 py-0.5 rounded-md text-[11px] font-medium border cursor-pointer transition-colors";
export const PRESET =
  "tabular h-6 px-1.5 rounded text-[10.5px] font-medium border cursor-pointer transition-colors";
export const ASIDE =
  "w-full lg:w-[320px] shrink-0 border-t lg:border-t-0 lg:border-l border-[var(--border)] bg-[var(--surface)]";
export const LBL = "text-[10px] font-semibold uppercase tracking-wider text-[var(--text-3)]";

export function posKey(r: ComparePositionRow): string {
  return `${r.run_id}:${r.iteration}:${r.game_idx}:${r.step}`;
}

export function humanizeToken(raw: string): string {
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

export function clampSamples(raw: string): number {
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n >= 1 ? n : 120;
}

export function signedDelta(delta: number): string {
  const pts = delta * 100;
  return `${pts > 0.05 ? "+" : ""}${pts.toFixed(1)}%`;
}

export function shortRef(ref: string): string {
  if (ref === "heuristic") return "Heuristic baseline";
  const parts = ref.split("::");
  return parts.length === 2
    ? `${(parts[0] ?? "").split("/").pop() ?? ""} · ${parts[1] ?? ""}`
    : ref;
}

export function groupCheckpoints(checkpoints: GlobalCheckpointOption[]): CheckpointGroup[] {
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

export function resolveSelection(
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

export function dotVisual(
  pos: ComparePositionRow,
  sel: boolean,
): { prio: number; r: number; fill: string; op: number } {
  const fill = pos.move_changed ? "var(--accent)" : "var(--text-3)";
  if (sel) return { prio: 2, r: 5.2, fill, op: 1 };
  if (pos.move_changed) return { prio: 1, r: 3.8, fill, op: 0.85 };
  return { prio: 0, r: 2.9, fill, op: 0.42 };
}

export function filterAndSort(
  positions: ComparePositionRow[],
  filter: FilterState,
): ComparePositionRow[] {
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

export function executeCompareScan(
  req: { ckptA: string; ckptB: string; sampleCount: number; runId: string },
  abortRef: RefObject<AbortController | null>,
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

export function deriveViewData(
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

/** Tabletop chrome bars: top navigation/POV bar, alerts, edit mode banner, center turn divider, and turn timeline scrubber. */

import {
  AlertTriangle,
  Bookmark,
  ChevronLeft,
  ChevronRight,
  Info,
  Pencil,
  Plus,
  RotateCcw,
  Sparkles,
  X,
} from "lucide-react";
import type { ReactNode } from "react";
import { Button, SegmentedControl } from "@/components/ui/Primitives";
import type { BoardState, ForkEdit, GameTarget, PovMode, TimelinePoint } from "@/lib/types";
import { describeForkEdit } from "./tabletopHelpers";
import type { PlaymatController } from "./tabletopTypes";

const POV_OPTIONS: ReadonlyArray<{ value: PovMode; label: string }> = [
  { value: "hero", label: "Hero POV" },
  { value: "omniscient", label: "Omniscient" },
];

function formatTurnBanner(board: BoardState): string {
  const step = board.step_name.replace(/([a-z])([A-Z])/g, "$1 $2");
  if (board.turn === 0) return `${board.phase.replace(/([a-z])([A-Z])/g, "$1 $2")} · ${step}`;
  return `${board.active_is_hero ? "Hero" : "Opp"} Turn ${board.turn} · ${step}`;
}

export function EditModeBanner(props: {
  heroSeat: number;
  canEdit: boolean;
  nearestPriorityStep: number | null;
  edits: ForkEdit[];
  ctrl: PlaymatController;
}): ReactNode {
  const { ctrl } = props;
  const chipBtn =
    "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[10.5px] font-medium cursor-pointer";
  return (
    <div className="relative z-30 flex flex-wrap items-center justify-between gap-1.5 px-2.5 sm:px-3 py-1 sm:py-1.5 bg-[var(--fork-soft)] border-b border-[var(--fork)]/30 text-[10px] sm:text-[10.5px] shrink-0">
      <div className="flex flex-wrap items-center gap-1.5 min-w-0">
        <span className="font-semibold text-[var(--fork)]">
          {props.canEdit ? (
            <>
              Edit Mode
              <span className="hidden sm:inline"> · Click life or hand cards to fork</span>
              <span className="sm:hidden"> · Tap life or hand</span>
            </>
          ) : (
            "Edit Mode · Board forks require a Priority step"
          )}
        </span>
        {!props.canEdit && props.nearestPriorityStep !== null ? (
          <button
            type="button"
            onClick={(): void => {
              if (props.nearestPriorityStep !== null) ctrl.selectStep(props.nearestPriorityStep);
            }}
            className="rounded-md border border-[var(--fork)] bg-[var(--fork)] text-white px-2 py-0.5 text-[10px] font-semibold cursor-pointer"
          >
            Jump to Priority #{props.nearestPriorityStep}
          </button>
        ) : null}
        {props.edits.map((e, i) => (
          <span
            key={`${e.op}-${e.seat}-${e.card ?? e.value ?? "x"}-${String(i)}`}
            className="inline-flex items-center gap-1 rounded bg-[var(--surface)] border border-[var(--fork)]/35 px-1.5 py-0.5 text-[10px]"
          >
            <span className="max-w-[140px] truncate">{describeForkEdit(e, props.heroSeat)}</span>
            <button
              type="button"
              aria-label="Remove edit"
              onClick={(): void => ctrl.removeForkEdit(i)}
              className="text-[var(--text-3)] hover:text-[var(--loss)] cursor-pointer"
            >
              <X className="w-2.5 h-2.5" />
            </button>
          </span>
        ))}
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        <button
          type="button"
          onClick={(): void => ctrl.setOverlay(ctrl.overlay === "addCard" ? "none" : "addCard")}
          className={`${chipBtn} ${ctrl.overlay === "addCard" ? "border-[var(--fork)] bg-[var(--fork)] text-white" : "border-[var(--border)] bg-[var(--surface)]"}`}
        >
          <Plus className="w-3 h-3" /> Add card
        </button>
        {props.edits.length > 0 ? (
          <button
            type="button"
            onClick={ctrl.resetFork}
            className={`${chipBtn} border-[var(--border)] bg-[var(--surface)]`}
          >
            <RotateCcw className="w-3 h-3" /> Reset
          </button>
        ) : null}
      </div>
    </div>
  );
}

export function TabletopTopBar(props: {
  target: GameTarget;
  turn: number | null;
  pov: PovMode;
  editMode: boolean;
  editCount: number;
  hasMutations: boolean;
  pinned: boolean;
  pinOpen: boolean;
  onBack: () => void;
  onChangePov: (pov: PovMode) => void;
  onToggleEdit: () => void;
  onResetFork: () => void;
  onTogglePin: () => void;
}): ReactNode {
  const runShort = props.target.run_id.split("/").pop() ?? props.target.run_id;
  const barBtnBase =
    "items-center justify-center gap-1 rounded-md border px-2 py-1 min-h-[34px] min-w-[34px] lg:min-h-0 lg:min-w-0 text-[11px] font-medium cursor-pointer";
  const pinLabel = props.pinned ? "Pinned" : "Pin";
  const pinTone =
    props.pinned || props.pinOpen
      ? "border-[var(--accent)] bg-[var(--accent)] text-white"
      : "border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-2)]";
  return (
    <div className="relative h-[44px] flex items-center justify-between gap-2 px-2.5 sm:px-3 border-b border-[var(--border)] bg-[var(--surface)] shrink-0">
      <div className="flex items-center gap-1.5 min-w-0">
        <button
          type="button"
          onClick={props.onBack}
          aria-label="Back to games"
          className="inline-flex items-center justify-center gap-1 rounded-md px-2 py-1 min-h-[34px] min-w-[34px] lg:min-h-0 lg:min-w-0 text-[11.5px] font-medium hover:bg-[var(--surface-2)] cursor-pointer shrink-0"
        >
          ← <span className="hidden sm:inline">Games</span>
        </button>
        <span className="inline-flex items-center gap-1 rounded-full border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-0.5 text-[11px] font-semibold truncate tabular">
          <span>Seed #{props.target.seed}</span>
          <span className="hidden md:inline text-[var(--text-3)]">· {runShort}</span>
          <span>· u{props.target.iteration}</span>
          {props.turn === null ? null : (
            <span className="text-[var(--text-3)]">· T{props.turn}</span>
          )}
        </span>
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        <SegmentedControl
          options={POV_OPTIONS}
          value={props.pov}
          onChange={props.onChangePov}
          className="hidden sm:inline-flex"
        />
        <button
          type="button"
          onClick={(): void => props.onChangePov(props.pov === "hero" ? "omniscient" : "hero")}
          className="sm:hidden inline-flex items-center justify-center min-h-[34px] min-w-[34px] rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1 text-[10.5px] font-semibold cursor-pointer"
        >
          {props.pov === "hero" ? "Hero" : "Omni"}
        </button>
        <button
          type="button"
          onClick={props.onToggleEdit}
          className={`hidden lg:inline-flex ${barBtnBase} ${props.editMode ? "border-[var(--fork)] bg-[var(--fork-soft)] text-[var(--fork)]" : "border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-2)]"}`}
        >
          <Pencil className="w-3 h-3" /> Edit Board
          {props.editCount > 0 ? ` (${props.editCount})` : ""}
        </button>
        {props.hasMutations ? (
          <button
            type="button"
            onClick={props.onResetFork}
            className="hidden lg:inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-[var(--fork)] hover:bg-[var(--fork-soft)] cursor-pointer"
          >
            <RotateCcw className="w-3 h-3" /> Reset
          </button>
        ) : null}
        <button
          type="button"
          onClick={props.onTogglePin}
          aria-label={`${pinLabel} position`}
          className={`inline-flex ${barBtnBase} ${pinTone}`}
        >
          <Bookmark className="w-3 h-3" />
          <span className="hidden sm:inline">{pinLabel}</span>
        </button>
      </div>
    </div>
  );
}

export function TabletopAlerts(props: { warning: string | null; error: string | null }): ReactNode {
  if (!props.warning && !props.error) return null;
  return (
    <>
      {props.warning ? (
        <div className="flex items-center gap-2 px-3.5 py-1.5 bg-[var(--warn-soft)] border-b border-[var(--warn)]/30 text-[11px] text-[var(--warn)] font-medium shrink-0">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span className="truncate">{props.warning}</span>
        </div>
      ) : null}
      {props.error ? (
        <div className="m-3 rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] p-3 text-xs text-[var(--loss)]">
          {props.error}
        </div>
      ) : null}
    </>
  );
}

export function CenterTurnDivider(props: {
  board: BoardState;
  stackOpen: boolean;
  detailsOpen: boolean;
  onToggleStack: () => void;
  onToggleDetails: () => void;
}): ReactNode {
  const { board } = props;
  return (
    <div className="flex flex-nowrap items-center gap-2 min-w-0 z-10">
      <div className="flex-1 h-px bg-[var(--border)] min-w-[12px]" />
      <span className="rounded-full border border-[var(--border)] bg-[var(--surface)]/95 px-3 py-0.5 text-[11px] font-semibold truncate">
        {formatTurnBanner(board)}
      </span>
      {board.stack.length > 0 ? (
        <button
          type="button"
          onClick={props.onToggleStack}
          className="tabular rounded-full border border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)] px-2.5 py-0.5 text-[10.5px] font-bold cursor-pointer shrink-0"
        >
          Stack ({board.stack.length}) {props.stackOpen ? "▴" : "▾"}
        </button>
      ) : null}
      <button
        type="button"
        onClick={props.onToggleDetails}
        aria-label="Board details"
        className={`w-5 h-5 rounded-full border flex items-center justify-center cursor-pointer shrink-0 ${props.detailsOpen ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]" : "border-[var(--border)] bg-[var(--surface)] text-[var(--text-3)]"}`}
      >
        <Info className="w-3 h-3" />
      </button>
      <div className="flex-1 h-px bg-[var(--border)] min-w-[12px]" />
    </div>
  );
}

function TimelineCurveSvg(props: {
  timeline: TimelinePoint[];
  currentTurn: number;
  onSelectStep: (step: number) => void;
}): ReactNode {
  const pts = props.timeline;
  if (pts.length < 2) return null;
  const stepX = 816 / (pts.length - 1);
  const coords = pts.map((p, i) => ({
    turn: p.turn,
    step: p.step,
    x: 22 + i * stepX,
    y: 25 - Math.max(0, Math.min(1, p.eval_win)) * 21,
  }));
  const poly = coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");
  const area = `${coords[0]?.x ?? 22},25 ${poly} ${coords[coords.length - 1]?.x ?? 22},25`;
  const stride = Math.max(1, Math.ceil(coords.length / 18));
  return (
    <svg viewBox="0 0 860 38" className="w-full h-[38px] block">
      <line x1={22} y1={15} x2={838} y2={15} stroke="var(--border)" strokeDasharray="3 3" />
      <polygon points={area} fill="var(--accent-soft)" />
      <polyline
        points={poly}
        fill="none"
        stroke="var(--accent)"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      {coords.map((c, idx) => {
        const active = c.turn === props.currentTurn;
        return (
          <g
            key={c.turn}
            onClick={(): void => props.onSelectStep(c.step)}
            className="cursor-pointer"
          >
            <circle
              cx={c.x}
              cy={c.y}
              r={active ? 4.5 : 3}
              fill={active ? "var(--accent)" : "var(--text-2)"}
              stroke="var(--surface)"
              strokeWidth="1.4"
            />
            {active || idx === coords.length - 1 || idx % stride === 0 ? (
              <text
                x={c.x}
                y={37}
                textAnchor="middle"
                fontSize="9"
                fill={active ? "var(--text)" : "var(--text-3)"}
                fontWeight={active ? 700 : 500}
                className="tabular"
              >
                T{c.turn}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

export function TurnTimelineBar(props: {
  timeline: TimelinePoint[];
  evalSource?: string;
  currentStep: number;
  totalSteps: number;
  maxPlayableStep: number;
  turningPointStep?: number;
  currentTurn: number;
  onSelectStep: (step: number) => void;
}): ReactNode {
  const upper = Math.max(0, Math.min(Math.max(0, props.totalSteps - 1), props.maxPlayableStep));
  const hasTurnPt =
    props.turningPointStep !== undefined &&
    props.turningPointStep >= 0 &&
    props.turningPointStep <= upper;
  return (
    <div className="relative shrink-0 border-t border-[var(--border)] bg-[var(--surface)] px-3.5 py-1.5 hidden lg:flex flex-col gap-1">
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-[var(--text-2)]">
        <div className="flex items-center gap-2 min-w-0">
          <strong className="text-[var(--text)] whitespace-nowrap">
            {props.evalSource === "model" ? "Model Win%" : "Turn Timeline"}
          </strong>
          <span className="tabular whitespace-nowrap">
            T{props.currentTurn} · Step {props.currentStep}/{upper}
          </span>
          {hasTurnPt ? (
            <button
              type="button"
              onClick={(): void => props.onSelectStep(props.turningPointStep ?? 0)}
              className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] border cursor-pointer ${props.currentStep === props.turningPointStep ? "border-[var(--warn)] bg-[var(--warn-soft)] text-[var(--warn)] font-semibold" : "border-[var(--border)] text-[var(--text-2)]"}`}
            >
              <Sparkles className="w-3 h-3" />
              Turning Pt #{props.turningPointStep}
            </button>
          ) : null}
        </div>
        <div className="flex items-center gap-1.5">
          <input
            type="range"
            min={0}
            max={Math.max(0, upper)}
            value={Math.min(props.currentStep, upper)}
            aria-label="Scrub game step"
            onChange={(e): void => props.onSelectStep(Number(e.target.value))}
            className="w-32 sm:w-44 accent-[var(--accent)] cursor-pointer"
          />
          <Button
            size="sm"
            disabled={props.currentStep <= 0}
            onClick={(): void => props.onSelectStep(Math.max(0, props.currentStep - 1))}
            title="Previous step"
            className="px-1.5 py-1"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </Button>
          <Button
            size="sm"
            disabled={props.currentStep >= upper}
            onClick={(): void => props.onSelectStep(Math.min(upper, props.currentStep + 1))}
            title="Next step"
            className="px-1.5 py-1"
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>
      <TimelineCurveSvg
        timeline={props.timeline}
        currentTurn={props.currentTurn}
        onSelectStep={props.onSelectStep}
      />
    </div>
  );
}

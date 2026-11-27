/** Top bar, alerts, edit banner, responsive battlefield halves, overlays, timeline bar, and full playmat for Tabletop. */

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
import { type ReactNode, useSyncExternalStore } from "react";
import { BoardDetailsPopover, CardZoomPreview } from "@/components/PolicyMatrixModal";
import { describeForkEdit, type resolveOverviewInfo } from "@/components/TabletopDock";
import {
  AddCardPopover,
  type DrawerTarget,
  HandCardEditPopover,
  type HandCardEditTarget,
  HandFanView,
  type HoverCardInfo,
  LandPilesRow,
  PermanentsRow,
  StackOverlay,
  type TabletopScale,
  ZoneDrawerSheet,
  ZoneFlankView,
} from "@/components/TabletopParts";
import { Button, SegmentedControl } from "@/components/ui/Primitives";
import type { BoardState, ForkEdit, GameTarget, PovMode, TimelinePoint } from "@/lib/types";

export type ActiveOverlay = "none" | "details" | "addCard" | "pin" | "mobilePolicy";
export type OverviewInfo = ReturnType<typeof resolveOverviewInfo>;

const POV_OPTIONS: ReadonlyArray<{ value: PovMode; label: string }> = [
  { value: "hero", label: "Hero POV" },
  { value: "omniscient", label: "Omniscient" },
];

export interface PlaymatController {
  pov: PovMode;
  editMode: boolean;
  overlay: ActiveOverlay;
  stackOpen: boolean;
  handEditTarget: HandCardEditTarget | null;
  drawer: DrawerTarget | null;
  hoverInfo: HoverCardInfo | null;
  pinnedCard: HoverCardInfo | null;
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  setOverlay: (next: ActiveOverlay) => void;
  toggleStack: () => void;
  setHandEditTarget: (target: HandCardEditTarget | null) => void;
  addForkEdit: (edit: ForkEdit) => void;
  removeForkEdit: (idx: number) => void;
  resetFork: () => void;
  setDrawer: (drawer: DrawerTarget | null) => void;
  setHover: (info: HoverCardInfo | null) => void;
  setPinnedCard: (info: HoverCardInfo | null) => void;
  setHoverSource: (permId: number | null) => void;
  adjustLife: (seat: 0 | 1, delta: number) => void;
  selectStep: (step: number) => void;
}

function formatTurnBanner(board: BoardState): string {
  const step = board.step_name.replace(/([a-z])([A-Z])/g, "$1 $2");
  if (board.turn === 0) return `${board.phase.replace(/([a-z])([A-Z])/g, "$1 $2")} · ${step}`;
  return `${board.active_is_hero ? "Hero" : "Opp"} Turn ${board.turn} · ${step}`;
}

function badgeTone(
  controls: boolean,
  isActive: boolean,
  isActing: boolean,
): { border: string; dot: string } {
  let border = "border-[var(--border)]";
  if (controls) border = "border-[var(--fork)]";
  else if (isActive) border = "border-[var(--accent)]";
  let dot = "bg-[var(--border-strong)]";
  if (isActing) dot = "bg-[var(--win-fill)]";
  else if (isActive) dot = "bg-[var(--accent)]";
  return { border, dot };
}

function subscribeResize(onStoreChange: () => void): () => void {
  if (typeof window === "undefined") return (): void => undefined;
  window.addEventListener("resize", onStoreChange);
  return (): void => window.removeEventListener("resize", onStoreChange);
}

function readTabletopScale(): TabletopScale {
  if (typeof window === "undefined" || window.innerWidth >= 1024) return "desktop";
  return window.innerHeight <= 740 ? "mobile-short" : "mobile-tall";
}

function PlayerCornerBadge(props: {
  label: string;
  player: BoardState["hero"];
  isActing: boolean;
  editMode: boolean;
  canEdit: boolean;
  compact?: boolean;
  onAdjust: (d: number) => void;
}): ReactNode {
  const { player } = props;
  const controls = props.editMode && props.canEdit;
  const tone = badgeTone(controls, Boolean(player.is_active), props.isActing);
  const stepCls = props.compact
    ? "w-5 h-5 inline-flex items-center justify-center rounded-full bg-[var(--fork-soft)] text-[var(--fork)] cursor-pointer disabled:opacity-35"
    : "w-4 h-4 inline-flex items-center justify-center rounded-full bg-[var(--fork-soft)] text-[var(--fork)] cursor-pointer disabled:opacity-35";
  const padCls = props.compact
    ? "gap-1 px-1.5 py-0.5 text-[10.5px]"
    : "gap-1.5 px-2 py-0.5 text-[11px]";
  return (
    <div
      className={`inline-flex items-center rounded-full border bg-[var(--surface)]/95 shadow-[var(--shadow-sm)] ${padCls} ${tone.border}`}
    >
      <span className={`w-2 h-2 rounded-full shrink-0 ${tone.dot}`} />
      <span className="font-semibold">{props.label}</span>
      <span
        className={`tabular inline-flex items-center gap-1 font-bold ${player.life <= 5 ? "text-[var(--loss)]" : ""}`}
      >
        {controls ? (
          <button
            type="button"
            disabled={player.life <= 1}
            onClick={(): void => props.onAdjust(-1)}
            aria-label={`Decrease ${props.label} life`}
            className={stepCls}
          >
            −
          </button>
        ) : null}
        <span>{player.life}</span>
        {controls ? (
          <button
            type="button"
            onClick={(): void => props.onAdjust(1)}
            aria-label={`Increase ${props.label} life`}
            className={stepCls}
          >
            +
          </button>
        ) : null}
      </span>
      {player.mana_pool ? (
        <span className="tabular rounded bg-[var(--warn-soft)] text-[var(--warn)] px-1 text-[10px] font-bold">
          {player.mana_pool}
        </span>
      ) : null}
    </div>
  );
}

function BattlefieldHalf(props: {
  side: "hero" | "opp";
  player: BoardState["hero"];
  isActing: boolean;
  canEdit: boolean;
  scale: TabletopScale;
  permNames: ReadonlyMap<number, string>;
  ctrl: PlaymatController;
  onAdjustLife: (delta: number) => void;
}): ReactNode {
  const { player, ctrl, scale } = props;
  const isOpp = props.side === "opp";
  const hand = (
    <HandFanView
      side={props.side}
      cards={player.hand}
      hiddenCount={player.zone_counts.hand}
      hidden={isOpp ? player.hand_hidden : false}
      scale={scale}
      editMode={ctrl.editMode}
      selectedEditKey={ctrl.handEditTarget?.key ?? null}
      onHover={ctrl.setHover}
      onPinCard={ctrl.setPinnedCard}
      onSelectCardForEdit={ctrl.setHandEditTarget}
    />
  );
  const handLane =
    scale === "desktop" ? (
      hand
    ) : (
      <div className="flex items-center gap-1.5 min-w-0 shrink-0">
        <div className="shrink-0">
          <PlayerCornerBadge
            label={isOpp ? "Opponent" : "Hero"}
            player={player}
            isActing={props.isActing}
            editMode={ctrl.editMode}
            canEdit={props.canEdit}
            compact
            onAdjust={props.onAdjustLife}
          />
        </div>
        <div className="flex-1 min-w-0">{hand}</div>
      </div>
    );
  const hasOther = player.other_permanents.length > 0;
  const lands = hasOther ? (
    <div className="w-full min-w-0 overflow-x-auto no-scrollbar py-1 min-h-[44px] lg:min-h-[52px] shrink-0">
      <div className="w-max mx-auto flex items-center gap-2.5 lg:gap-3 flex-nowrap px-1.5">
        <LandPilesRow piles={player.lands} compact scale={scale} onHover={ctrl.setHover} />
        <div className="pl-2.5 lg:pl-3 shrink-0 border-l border-dashed border-[var(--border-strong)]">
          <PermanentsRow
            permanents={player.other_permanents}
            compact
            scale={scale}
            permNames={props.permNames}
            highlightIds={ctrl.highlightIds}
            highlightTone={ctrl.highlightTone}
            onHover={ctrl.setHover}
            onPinCard={ctrl.setPinnedCard}
          />
        </div>
      </div>
    </div>
  ) : (
    <div className="w-full min-w-0 min-h-[44px] lg:min-h-[52px] flex items-center shrink-0">
      <LandPilesRow piles={player.lands} scale={scale} onHover={ctrl.setHover} />
    </div>
  );
  const creatures = (
    <PermanentsRow
      permanents={player.creatures}
      scale={scale}
      permNames={props.permNames}
      highlightIds={ctrl.highlightIds}
      highlightTone={ctrl.highlightTone}
      onHover={ctrl.setHover}
      onPinCard={ctrl.setPinnedCard}
    />
  );
  return (
    <div className="h-full min-h-0 grid grid-cols-[minmax(0,1fr)_64px] sm:grid-cols-[minmax(0,1fr)_70px] gap-1.5 sm:gap-2">
      <div className="flex flex-col justify-between min-w-0 min-h-0 h-full gap-0.5 lg:gap-1">
        {isOpp ? handLane : null}
        <div className="flex-1 min-h-0 flex flex-col justify-center gap-0.5 lg:gap-1">
          {isOpp ? (
            <>
              {lands}
              {creatures}
            </>
          ) : (
            <>
              {creatures}
              {lands}
            </>
          )}
        </div>
        {isOpp ? null : handLane}
      </div>
      <ZoneFlankView side={props.side} player={player} onOpenDrawer={ctrl.setDrawer} />
    </div>
  );
}

function EditModeBanner(props: {
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

function PlaymatOverlays(props: {
  board: BoardState;
  heroSeat: 0 | 1;
  oppSeat: 0 | 1;
  cardCatalog: string[];
  nearestPriorityStep: number | null;
  ctrl: PlaymatController;
}): ReactNode {
  const { board: b, ctrl } = props;
  const canEdit = b.action_space_kind === "PRIORITY";
  const nonPerm = new Set(b.non_permanent_names ?? []);
  const handTarget = ctrl.editMode ? ctrl.handEditTarget : null;
  return (
    <>
      <StackOverlay
        stack={b.stack}
        open={ctrl.stackOpen && ctrl.drawer === null}
        onClose={ctrl.toggleStack}
        onHoverSource={ctrl.setHoverSource}
        onHoverCard={ctrl.setHover}
      />
      {ctrl.overlay === "details" ? (
        <BoardDetailsPopover board={b} onClose={(): void => ctrl.setOverlay("none")} />
      ) : null}
      {ctrl.editMode && ctrl.overlay === "addCard" ? (
        <AddCardPopover
          heroSeat={props.heroSeat}
          canEditBoard={canEdit}
          nearestPriorityStep={props.nearestPriorityStep}
          nonPermanentNames={nonPerm}
          cardCatalog={props.cardCatalog}
          onAddEdit={ctrl.addForkEdit}
          onJumpToStep={ctrl.selectStep}
          onClose={(): void => ctrl.setOverlay("none")}
        />
      ) : null}
      {handTarget ? (
        <HandCardEditPopover
          target={handTarget}
          seat={handTarget.side === "hero" ? props.heroSeat : props.oppSeat}
          canEditBoard={canEdit}
          nearestPriorityStep={props.nearestPriorityStep}
          cardCatalog={props.cardCatalog}
          onEdit={ctrl.addForkEdit}
          onJumpToStep={ctrl.selectStep}
          onClose={(): void => ctrl.setHandEditTarget(null)}
        />
      ) : null}
      <ZoneDrawerSheet
        target={ctrl.drawer}
        hero={b.hero}
        opp={b.opp}
        pov={ctrl.pov}
        onClose={(): void => {
          ctrl.setHover(null);
          ctrl.setDrawer(null);
        }}
        onHover={ctrl.setHover}
      />
      <CardZoomPreview
        info={ctrl.pinnedCard ?? ctrl.hoverInfo}
        pinned={ctrl.pinnedCard !== null}
        onDismiss={(): void => ctrl.setPinnedCard(null)}
      />
    </>
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

function TurnTimelineBar(props: {
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

function CenterTurnDivider(props: {
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

export function TabletopPlaymat(props: {
  board: BoardState;
  recordedStep: number;
  forkEdits: ForkEdit[];
  cardCatalog: string[];
  nearestPriorityStep: number | null;
  overview: OverviewInfo;
  ctrl: PlaymatController;
}): ReactNode {
  const { board: b, ctrl } = props;
  const scale = useSyncExternalStore(
    subscribeResize,
    readTabletopScale,
    (): TabletopScale => "desktop",
  );
  const heroSeat: 0 | 1 = b.hero_seat === 1 ? 1 : 0;
  const oppSeat: 0 | 1 = heroSeat === 1 ? 0 : 1;
  const canEdit = b.action_space_kind === "PRIORITY";
  const permNames = new Map<number, string>([
    ...b.hero.creatures.map((p): [number, string] => [p.id, p.name]),
    ...b.opp.creatures.map((p): [number, string] => [p.id, p.name]),
  ]);
  const stackOpen = ctrl.stackOpen && ctrl.drawer === null;
  const oppActing = b.actor_is_hero === false;
  const heroActing = b.actor_is_hero !== false;
  return (
    <div className="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden">
      {ctrl.editMode ? (
        <EditModeBanner
          heroSeat={heroSeat}
          canEdit={canEdit}
          nearestPriorityStep={props.nearestPriorityStep}
          edits={props.forkEdits}
          ctrl={ctrl}
        />
      ) : null}
      <div
        className={`flex-1 min-h-0 relative felt-surface px-2.5 sm:px-3 py-1 lg:py-1.5 grid grid-rows-[minmax(0,1fr)_auto_minmax(0,1fr)] gap-1 overflow-y-auto lg:overflow-hidden ${ctrl.editMode ? "ring-2 ring-inset ring-[var(--fork)]/45 border-2 border-dashed border-[var(--fork)]/45" : ""}`}
      >
        {scale === "desktop" ? (
          <div className="absolute top-2 left-2.5 sm:left-3 z-20">
            <PlayerCornerBadge
              label="Opponent"
              player={b.opp}
              isActing={oppActing}
              editMode={ctrl.editMode}
              canEdit={canEdit}
              onAdjust={(d): void => ctrl.adjustLife(oppSeat, d)}
            />
          </div>
        ) : null}
        <BattlefieldHalf
          side="opp"
          player={b.opp}
          isActing={oppActing}
          canEdit={canEdit}
          scale={scale}
          permNames={permNames}
          ctrl={ctrl}
          onAdjustLife={(d): void => ctrl.adjustLife(oppSeat, d)}
        />
        <CenterTurnDivider
          board={b}
          stackOpen={stackOpen}
          detailsOpen={ctrl.overlay === "details"}
          onToggleStack={ctrl.toggleStack}
          onToggleDetails={(): void =>
            ctrl.setOverlay(ctrl.overlay === "details" ? "none" : "details")
          }
        />
        <BattlefieldHalf
          side="hero"
          player={b.hero}
          isActing={heroActing}
          canEdit={canEdit}
          scale={scale}
          permNames={permNames}
          ctrl={ctrl}
          onAdjustLife={(d): void => ctrl.adjustLife(heroSeat, d)}
        />
        {scale === "desktop" ? (
          <div className="absolute bottom-2 left-2.5 sm:left-3 z-20">
            <PlayerCornerBadge
              label="Hero"
              player={b.hero}
              isActing={heroActing}
              editMode={ctrl.editMode}
              canEdit={canEdit}
              onAdjust={(d): void => ctrl.adjustLife(heroSeat, d)}
            />
          </div>
        ) : null}
        <PlaymatOverlays
          board={b}
          heroSeat={heroSeat}
          oppSeat={oppSeat}
          cardCatalog={props.cardCatalog}
          nearestPriorityStep={props.nearestPriorityStep}
          ctrl={ctrl}
        />
      </div>
      <TurnTimelineBar
        timeline={props.overview.timeline}
        evalSource={props.overview.evalSource}
        currentStep={props.recordedStep}
        totalSteps={b.total_steps}
        maxPlayableStep={props.overview.maxPlayableStep}
        turningPointStep={props.overview.turningPointStep}
        currentTurn={b.turn}
        onSelectStep={ctrl.selectStep}
      />
    </div>
  );
}

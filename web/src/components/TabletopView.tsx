/** Crowded MTG Tabletop & Inline Counterfactual Fork workspace with Hero-normalized eval, active/priority badges & play-by-play. */

import { AlertTriangle, GitBranch, RotateCcw } from "lucide-react";
import { type ReactNode, useCallback, useMemo, useState } from "react";
import { TabletopDock } from "@/components/TabletopDock";
import {
  CardZoomPreview,
  type DrawerTarget,
  HandFanView,
  type HoverCardInfo,
  LandPilesRow,
  PermanentsRow,
  StackColumn,
  TurnTimelineBar,
  ZoneDrawerSheet,
  ZoneFlankView,
} from "@/components/TabletopParts";
import { Badge, Button, SegmentedControl } from "@/components/ui/Primitives";
import {
  useForkEvaluation,
  useGameOverview,
  useGameStepData,
  useKeyboardShortcut,
  useRunDetail,
} from "@/hooks/useStudioData";
import type {
  BoardState,
  EvaluationPair,
  ForkEdit,
  ForkResponse,
  GameTarget,
  PovMode,
  RecentEvent,
  TimelinePoint,
  WatchlistItem,
} from "@/lib/types";

type TabletopMode = "inspect" | "fork";

const POV_OPTIONS: ReadonlyArray<{ value: PovMode; label: string }> = [
  { value: "hero", label: "Hero POV" },
  { value: "omniscient", label: "Omniscient" },
];

const MODE_OPTIONS: ReadonlyArray<{
  value: TabletopMode;
  label: string;
  accent?: "default" | "fork";
}> = [
  { value: "inspect", label: "Inspect" },
  { value: "fork", label: "Fork", accent: "fork" },
];

function PlayerStripHeader(props: {
  label: string;
  life: number;
  permCount: number;
  isActive: boolean;
  isActing: boolean;
  actingBadge: string;
  openLands: number;
  manaPool: string;
  forkMode: boolean;
  canEditLife: boolean;
  onAdjustLife: (delta: number) => void;
  hand: ReactNode;
}): ReactNode {
  const lowLife = props.life <= 5;
  return (
    <div className="flex items-center justify-between gap-2 min-h-[32px] min-w-0">
      <div className="flex items-center gap-1 min-w-0 shrink-0 flex-nowrap">
        <span className="text-[11.5px] font-semibold whitespace-nowrap">{props.label}</span>
        {props.isActive ? (
          <span className="rounded bg-[var(--accent-soft)] text-[var(--accent)] px-1 py-0.5 text-[9px] font-bold uppercase tracking-wide whitespace-nowrap">
            Turn
          </span>
        ) : null}
        {props.isActing ? (
          <span className="rounded bg-[var(--win-soft)] text-[var(--win)] px-1 py-0.5 text-[9px] font-bold uppercase tracking-wide whitespace-nowrap">
            {props.actingBadge}
          </span>
        ) : null}
        <span
          className={`tabular inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[11px] font-bold whitespace-nowrap ${
            lowLife
              ? "border-[var(--loss)] bg-[var(--loss-soft)] text-[var(--loss)]"
              : "border-[var(--border)] bg-[var(--surface)]"
          }`}
        >
          {props.forkMode && props.canEditLife ? (
            <button
              type="button"
              disabled={props.life <= 1}
              onClick={(): void => props.onAdjustLife(-1)}
              aria-label={`Decrease ${props.label} life`}
              className="w-4 h-4 rounded-full bg-[var(--fork-soft)] text-[var(--fork)] flex items-center justify-center cursor-pointer disabled:opacity-35"
            >
              −
            </button>
          ) : null}
          <span>{props.life} life</span>
          {props.forkMode && props.canEditLife ? (
            <button
              type="button"
              onClick={(): void => props.onAdjustLife(1)}
              aria-label={`Increase ${props.label} life`}
              className="w-4 h-4 rounded-full bg-[var(--fork-soft)] text-[var(--fork)] flex items-center justify-center cursor-pointer"
            >
              +
            </button>
          ) : null}
        </span>
        <span className="tabular rounded border border-[var(--border)] bg-[var(--surface)] px-1.5 py-0.5 text-[10px] text-[var(--text-3)] whitespace-nowrap">
          {props.openLands} open · {props.permCount} perms
        </span>
        {props.manaPool.length > 0 ? (
          <span
            title="Floating mana in mana pool"
            className="tabular rounded border border-[var(--warn)]/40 bg-[var(--warn-soft)] px-1.5 py-0.5 text-[10px] font-bold text-[var(--warn)] whitespace-nowrap"
          >
            Pool {props.manaPool}
          </span>
        ) : null}
      </div>
      <div className="flex-1 min-w-[90px] flex justify-end">{props.hand}</div>
    </div>
  );
}

function RecentEventsStrip(props: {
  events: RecentEvent[];
  onSelectStep: (step: number) => void;
}): ReactNode {
  if (props.events.length === 0) return null;
  const newestFirst = props.events.toReversed();
  return (
    <div className="shrink-0 border-t border-[var(--border)] bg-[var(--surface-2)]/70 px-3 py-1 flex items-center gap-1.5 overflow-x-auto text-[10.5px] text-[var(--text-2)]">
      <span className="font-bold uppercase tracking-wider text-[9.5px] text-[var(--text-3)] shrink-0">
        Recent
      </span>
      {newestFirst.map((ev) => (
        <button
          key={`${ev.step}-${ev.text}`}
          type="button"
          title={`Jump to step #${ev.step}`}
          onClick={(): void => props.onSelectStep(ev.step)}
          className="inline-flex items-center gap-1 rounded bg-[var(--surface)] border border-[var(--border)] px-1.5 py-0.5 whitespace-nowrap shrink-0 cursor-pointer hover:border-[var(--accent)] hover:text-[var(--text)] transition-colors"
        >
          <span className="tabular text-[9.5px] text-[var(--text-3)]">T{ev.turn}</span>
          <span
            className={
              ev.actor_side === "hero"
                ? "font-semibold text-[var(--win)]"
                : "font-semibold text-[var(--loss)]"
            }
          >
            {ev.actor_side === "hero" ? "Hero" : "Opp"}:
          </span>
          <span>{ev.text}</span>
        </button>
      ))}
    </div>
  );
}

function BattlefieldHalf(props: {
  side: "hero" | "opp";
  player: BoardState["hero"];
  seatNum: 0 | 1;
  isActing: boolean;
  actingBadge: string;
  forkMode: boolean;
  canEditLife: boolean;
  narrowStack: boolean;
  permNames: ReadonlyMap<number, string>;
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  onOpenDrawer: (target: DrawerTarget) => void;
  onSetHover: (info: HoverCardInfo | null) => void;
  onPinCard: (info: HoverCardInfo | null) => void;
  onAdjustLife: (seat: 0 | 1, delta: number) => void;
}): ReactNode {
  const { player } = props;
  const header = (
    <PlayerStripHeader
      label={props.side === "hero" ? "Hero" : "Opponent"}
      life={player.life}
      permCount={player.zone_counts.battlefield}
      isActive={Boolean(player.is_active)}
      isActing={props.isActing}
      actingBadge={props.actingBadge}
      openLands={player.open_lands ?? 0}
      manaPool={player.mana_pool ?? ""}
      forkMode={props.forkMode}
      canEditLife={props.canEditLife}
      onAdjustLife={(d): void => props.onAdjustLife(props.seatNum, d)}
      hand={
        <HandFanView
          cards={player.hand}
          hiddenCount={player.zone_counts.hand}
          hidden={props.side === "opp" ? player.hand_hidden : false}
          onHover={props.onSetHover}
          onPinCard={props.onPinCard}
        />
      }
    />
  );

  const landsRow = (
    <div className="flex items-center justify-between gap-2 min-h-[60px] min-w-0 overflow-x-auto">
      <LandPilesRow
        piles={player.lands}
        compact={player.other_permanents.length > 0 || props.narrowStack}
        onHover={props.onSetHover}
      />
      {player.other_permanents.length > 0 ? (
        <div className="pl-2 shrink-0 border-l border-dashed border-[var(--border-strong)]">
          <PermanentsRow
            permanents={player.other_permanents}
            compact
            permNames={props.permNames}
            highlightIds={props.highlightIds}
            highlightTone={props.highlightTone}
            onHover={props.onSetHover}
            onPinCard={props.onPinCard}
          />
        </div>
      ) : null}
    </div>
  );

  const creaturesRow = (
    <PermanentsRow
      permanents={player.creatures}
      narrow={props.narrowStack}
      permNames={props.permNames}
      highlightIds={props.highlightIds}
      highlightTone={props.highlightTone}
      onHover={props.onSetHover}
      onPinCard={props.onPinCard}
    />
  );

  const isOpp = props.side === "opp";
  const rows = isOpp ? [landsRow, creaturesRow] : [creaturesRow, landsRow];
  return (
    <div className="flex-none lg:flex-1 lg:min-h-0 grid grid-cols-[minmax(0,1fr)_74px] gap-2">
      <div className="flex flex-col justify-between min-w-0 lg:min-h-0 gap-1.5">
        {isOpp ? header : null}
        <div className="flex-none lg:flex-1 lg:min-h-0 flex flex-col justify-center gap-1.5">
          {rows}
        </div>
        {isOpp ? null : header}
      </div>
      <ZoneFlankView side={props.side} player={player} onOpenDrawer={props.onOpenDrawer} />
    </div>
  );
}

function formatTurnBanner(board: BoardState): string {
  if (board.turn === 0) return `${board.phase} · ${board.step_name}`;
  const who = board.active_is_hero ? "Hero" : "Opp";
  return `${who} Turn ${board.turn} · ${board.phase} · ${board.step_name}`;
}

function TabletopPlaymat(props: {
  board: BoardState;
  pov: PovMode;
  forkMode: boolean;
  stackOpen: boolean;
  drawer: DrawerTarget | null;
  hoverInfo: HoverCardInfo | null;
  pinnedCard: HoverCardInfo | null;
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  timeline: TimelinePoint[];
  maxPlayableStep: number;
  onToggleStack: () => void;
  onSetDrawer: (drawer: DrawerTarget | null) => void;
  onSetHover: (info: HoverCardInfo | null) => void;
  onPinCard: (info: HoverCardInfo | null) => void;
  onHoverSource: (permId: number | null) => void;
  onAdjustLife: (seat: 0 | 1, delta: number) => void;
  onSelectStep: (step: number) => void;
}): ReactNode {
  const b = props.board;
  const heroSeat: 0 | 1 = b.hero_seat === 1 ? 1 : 0;
  const oppSeat: 0 | 1 = heroSeat === 1 ? 0 : 1;
  const canEditLife = b.action_space_kind === "PRIORITY";
  const actingBadge = canEditLife ? "Priority" : "Deciding";
  const narrowStack = b.stack.length > 0 && props.stackOpen;
  const permNames = new Map<number, string>([
    ...b.hero.creatures.map((p): [number, string] => [p.id, p.name]),
    ...b.opp.creatures.map((p): [number, string] => [p.id, p.name]),
  ]);
  const turnLabel = formatTurnBanner(b);

  return (
    <div className="flex-none lg:flex-1 min-w-0 lg:min-h-0 flex flex-col">
      <div className="flex-none lg:flex-1 lg:min-h-0 relative felt-surface p-2.5 grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_auto] gap-3 lg:gap-2.5 lg:overflow-hidden">
        <div className="flex flex-col justify-between min-w-0 lg:min-h-0 gap-3 lg:gap-1.5">
          <BattlefieldHalf
            side="opp"
            player={b.opp}
            seatNum={oppSeat}
            isActing={b.actor_is_hero === false}
            actingBadge={actingBadge}
            forkMode={props.forkMode}
            canEditLife={canEditLife}
            narrowStack={narrowStack}
            permNames={permNames}
            highlightIds={props.highlightIds}
            highlightTone={props.highlightTone}
            onOpenDrawer={props.onSetDrawer}
            onSetHover={props.onSetHover}
            onPinCard={props.onPinCard}
            onAdjustLife={props.onAdjustLife}
          />

          <div className="flex flex-nowrap items-center gap-2 my-0.5 min-w-0">
            <div className="flex-1 h-px bg-[var(--border)] min-w-[12px]" />
            <span className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 py-0.5 text-[11px] font-semibold shadow-[var(--shadow-sm)] truncate">
              {turnLabel}
            </span>
            {b.stack.length > 0 ? (
              <button
                type="button"
                onClick={props.onToggleStack}
                className="tabular rounded-full border border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)] px-2.5 py-0.5 text-[10.5px] font-bold cursor-pointer whitespace-nowrap shrink-0"
              >
                Stack ({b.stack.length}) {props.stackOpen ? "▴" : "▾"}
              </button>
            ) : null}
            <div className="flex-1 h-px bg-[var(--border)] min-w-[12px]" />
          </div>

          <BattlefieldHalf
            side="hero"
            player={b.hero}
            seatNum={heroSeat}
            isActing={b.actor_is_hero !== false}
            actingBadge={actingBadge}
            forkMode={props.forkMode}
            canEditLife={canEditLife}
            narrowStack={narrowStack}
            permNames={permNames}
            highlightIds={props.highlightIds}
            highlightTone={props.highlightTone}
            onOpenDrawer={props.onSetDrawer}
            onSetHover={props.onSetHover}
            onPinCard={props.onPinCard}
            onAdjustLife={props.onAdjustLife}
          />
        </div>

        <StackColumn
          stack={b.stack}
          open={props.stackOpen}
          onClose={props.onToggleStack}
          onHoverSource={props.onHoverSource}
          onHoverCard={props.onSetHover}
        />

        <ZoneDrawerSheet
          target={props.drawer}
          hero={b.hero}
          opp={b.opp}
          pov={props.pov}
          onClose={(): void => {
            props.onSetHover(null);
            props.onSetDrawer(null);
          }}
          onHover={props.onSetHover}
        />

        <CardZoomPreview
          info={props.pinnedCard ?? props.hoverInfo}
          pinned={props.pinnedCard !== null}
          onDismiss={(): void => props.onPinCard(null)}
        />
      </div>

      <RecentEventsStrip events={b.recent_events ?? []} onSelectStep={props.onSelectStep} />

      <TurnTimelineBar
        timeline={props.timeline}
        currentStep={b.step}
        totalSteps={b.total_steps}
        maxPlayableStep={props.maxPlayableStep}
        currentTurn={b.turn}
        onSelectStep={props.onSelectStep}
      />
    </div>
  );
}

function TabletopWorkspaceBody(props: {
  board: BoardState;
  evaluation: EvaluationPair;
  baseEvaluation: EvaluationPair;
  pov: PovMode;
  mode: TabletopMode;
  stackOpen: boolean;
  drawer: DrawerTarget | null;
  hoverInfo: HoverCardInfo | null;
  pinnedCard: HoverCardInfo | null;
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  timeline: TimelinePoint[];
  maxPlayableStep: number;
  checkpoints: string[];
  ckptA: string;
  ckptB: string;
  forkBusy: boolean;
  forkEdits: ForkEdit[];
  forkWarnings: string[];
  forkError: string | null;
  pinned: boolean;
  onToggleStack: () => void;
  onSetDrawer: (drawer: DrawerTarget | null) => void;
  onSetHover: (info: HoverCardInfo | null) => void;
  onPinCard: (info: HoverCardInfo | null) => void;
  onHoverSource: (permId: number | null) => void;
  onHoverActionFocus: (ids: number[]) => void;
  onAdjustLife: (seat: 0 | 1, delta: number) => void;
  onSelectStep: (step: number) => void;
  onChangeCkptA: (v: string) => void;
  onChangeCkptB: (v: string) => void;
  onAddForkEdit: (edit: ForkEdit) => void;
  onRemoveForkEdit: (idx: number) => void;
  onPinPosition: (note: string) => void;
}): ReactNode {
  const winPct = Math.round(props.evaluation.primary.wdl.win * 100);
  const drawPct = Math.round(props.evaluation.primary.wdl.draw * 100);
  const lossPct = Math.max(0, 100 - winPct - drawPct);

  return (
    <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
      {/* Compact horizontal Hero Win% bar on tablet/mobile */}
      <div className="flex lg:hidden items-center gap-2 px-3 py-1.5 border-b border-[var(--border)] bg-[var(--surface)] shrink-0">
        <span className="tabular text-[10px] font-bold text-[var(--win)] whitespace-nowrap">
          Hero {winPct}%
        </span>
        <div className="flex-1 h-2.5 rounded-full overflow-hidden bg-[#27272a] flex border border-[var(--border)]">
          <div className="h-full bg-[var(--win-fill)]" style={{ width: `${winPct}%` }} />
        </div>
        <span className="tabular text-[10px] font-bold text-[var(--text-2)] whitespace-nowrap">
          Opp {lossPct}%
        </span>
      </div>

      {/* Vertical Hero Win% bar on desktop */}
      <aside
        aria-label="Hero win probability bar"
        className="hidden lg:flex w-[40px] shrink-0 border-r border-[var(--border)] bg-[var(--surface)] py-2 px-1.5 flex-col items-center gap-1.5"
      >
        <span className="tabular text-[10px] font-bold text-[var(--text-2)]">{lossPct}%</span>
        <div className="flex-1 w-[22px] rounded-[5px] overflow-hidden bg-[#27272a] relative flex flex-col justify-end border border-[var(--border)]">
          <div className="absolute top-1/2 inset-x-0 h-px bg-white/45 pointer-events-none" />
          <div
            className="w-full bg-[var(--win-fill)] transition-all duration-200"
            style={{ height: `${winPct}%` }}
          />
        </div>
        <span className="tabular text-[10px] font-bold text-[var(--win)]">{winPct}%</span>
      </aside>

      <TabletopPlaymat
        board={props.board}
        pov={props.pov}
        forkMode={props.mode === "fork"}
        stackOpen={props.stackOpen}
        drawer={props.drawer}
        hoverInfo={props.hoverInfo}
        pinnedCard={props.pinnedCard}
        highlightIds={props.highlightIds}
        highlightTone={props.highlightTone}
        timeline={props.timeline}
        maxPlayableStep={props.maxPlayableStep}
        onToggleStack={props.onToggleStack}
        onSetDrawer={props.onSetDrawer}
        onSetHover={props.onSetHover}
        onPinCard={props.onPinCard}
        onHoverSource={props.onHoverSource}
        onAdjustLife={props.onAdjustLife}
        onSelectStep={props.onSelectStep}
      />

      <TabletopDock
        board={props.board}
        evaluation={props.evaluation}
        baseEvaluation={props.baseEvaluation}
        checkpoints={props.checkpoints}
        ckptA={props.ckptA}
        ckptB={props.ckptB}
        onChangeCkptA={props.onChangeCkptA}
        onChangeCkptB={props.onChangeCkptB}
        forkMode={props.mode === "fork"}
        forkBusy={props.forkBusy}
        forkEdits={props.forkEdits}
        forkWarnings={props.forkWarnings}
        forkError={props.forkError}
        onAddForkEdit={props.onAddForkEdit}
        onRemoveForkEdit={props.onRemoveForkEdit}
        onHoverActionFocus={props.onHoverActionFocus}
        onPinPosition={props.onPinPosition}
        pinned={props.pinned}
      />
    </div>
  );
}

function resolveDefaultCheckpoints(args: {
  checkpoints: string[];
  gameIteration: number;
  ckptA: string | null;
  ckptB: string | null;
}): { effectiveA: string; effectiveB: string } {
  const { checkpoints, gameIteration, ckptA, ckptB } = args;
  const iterName = `iter-${String(gameIteration).padStart(4, "0")}`;
  const defaultA = checkpoints.includes(iterName)
    ? iterName
    : (checkpoints[checkpoints.length - 1] ?? "heuristic");
  const idxA = checkpoints.indexOf(defaultA);
  let defaultB = "same";
  if (idxA > 0) defaultB = checkpoints[idxA - 1] ?? "same";
  else if (checkpoints.length > 1) defaultB = checkpoints[0] ?? "same";
  return {
    effectiveA: ckptA ?? defaultA,
    effectiveB: ckptB ?? defaultB,
  };
}

function TabletopTopBar(props: {
  target: GameTarget;
  board: BoardState | null;
  replayWarning: string | null;
  pov: PovMode;
  mode: TabletopMode;
  onBack: () => void;
  onChangePov: (pov: PovMode) => void;
  onChangeMode: (mode: TabletopMode) => void;
  onResetFork: () => void;
}): ReactNode {
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-[var(--border)] bg-[var(--surface)] shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <Button variant="ghost" onClick={props.onBack}>
            ← Games
          </Button>
          <span className="text-[var(--border-strong)]">/</span>
          <span className="text-xs font-semibold truncate">
            Seed #{props.target.seed} · {props.target.run_id.split("/").pop()} · u
            {props.target.iteration}
          </span>
          {props.board ? (
            <Badge tone="accent">
              Turn {props.board.turn} · {props.board.action_space_kind}
            </Badge>
          ) : null}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <SegmentedControl options={POV_OPTIONS} value={props.pov} onChange={props.onChangePov} />
          <SegmentedControl
            options={MODE_OPTIONS}
            value={props.mode}
            onChange={props.onChangeMode}
          />
        </div>
      </div>

      {props.replayWarning ? (
        <div className="flex items-center gap-2 px-3.5 py-1.5 bg-[var(--warn-soft)] border-b border-[var(--warn)]/30 text-[11px] text-[var(--warn)] font-medium shrink-0">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span className="truncate">{props.replayWarning}</span>
        </div>
      ) : null}

      {props.mode === "fork" ? (
        <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-1.5 bg-[var(--fork-soft)] border-b border-[var(--fork)]/25 text-[11px] text-[var(--fork)] font-medium shrink-0">
          <div className="flex items-center gap-1.5">
            <GitBranch className="w-3.5 h-3.5 shrink-0" />
            <span>
              Counterfactual Fork active — edits are stateless and never modify recorded games.
            </span>
          </div>
          <Button size="sm" onClick={props.onResetFork}>
            <RotateCcw className="w-3 h-3" />
            Reset fork
          </Button>
        </div>
      ) : null}
    </>
  );
}

interface ReadyPayload {
  board: BoardState;
  evaluation: EvaluationPair;
  base: EvaluationPair;
}

function resolveActivePayload(
  mode: TabletopMode,
  data: { board: BoardState; evaluation: EvaluationPair } | null,
  forkResult: ForkResponse | null,
): ReadyPayload | null {
  if (mode === "fork" && forkResult !== null) {
    return {
      board: forkResult.board,
      evaluation: forkResult.fork_evaluation,
      base: forkResult.base_evaluation,
    };
  }
  if (data === null) return null;
  return { board: data.board, evaluation: data.evaluation, base: data.evaluation };
}

function resolveHighlight(
  stackSourceId: number | null,
  actionFocusIds: number[],
): { ids: Set<number>; tone: "target" | "source" } {
  if (stackSourceId === null) {
    return { ids: new Set(actionFocusIds), tone: "target" };
  }
  return { ids: new Set([stackSourceId]), tone: "source" };
}

function updateLifeForkEdits(args: {
  edits: ForkEdit[];
  board: BoardState;
  seat: 0 | 1;
  delta: number;
}): ForkEdit[] {
  const { edits, board, seat, delta } = args;
  const heroSeat = board.hero_seat === 1 ? 1 : 0;
  const baseLife = seat === heroSeat ? board.hero.life : board.opp.life;
  const existing = edits.find((e) => e.op === "set_life" && e.seat === seat);
  const currentLife = existing?.value ?? baseLife;
  const nextLife = Math.max(1, Math.min(99, currentLife + delta));
  if (nextLife === currentLife) return edits;
  const kept = edits.filter((e) => !(e.op === "set_life" && e.seat === seat));
  return [...kept, { op: "set_life", seat, value: nextLife }];
}

function isTargetPinned(watchlist: WatchlistItem[], target: GameTarget): boolean {
  return watchlist.some(
    (item) =>
      item.run_id === target.run_id &&
      item.iteration === target.iteration &&
      item.game_idx === target.game_idx &&
      item.step === target.step,
  );
}

type PinPositionFn = (args: {
  board: BoardState;
  target: GameTarget;
  note?: string;
}) => Promise<void>;

function resolveOverviewInfo(
  overview: ReturnType<typeof useGameOverview>["overview"],
  stepWarn: string | null | undefined,
): { timeline: TimelinePoint[]; maxPlayableStep: number; replayWarning: string | null } {
  const timeline = overview?.timeline ?? [];
  const fallbackMax = Math.max(0, (overview?.total_steps ?? 1) - 1);
  const maxPlayableStep = overview?.max_playable_step ?? fallbackMax;
  const replayWarning = stepWarn ?? overview?.replay_warning ?? null;
  return { timeline, maxPlayableStep, replayWarning };
}

function LoadedTabletop(props: {
  target: GameTarget;
  watchlist: WatchlistItem[];
  onChangeStep: (step: number) => void;
  onPinPosition: PinPositionFn;
  onBackToGames: () => void;
}): ReactNode {
  const [pov, setPov] = useState<PovMode>("hero");
  const [mode, setMode] = useState<TabletopMode>("inspect");
  const [stackOpen, setStackOpen] = useState(true);
  const [drawer, setDrawer] = useState<DrawerTarget | null>(null);
  const [hoverInfo, setHoverInfo] = useState<HoverCardInfo | null>(null);
  const [pinnedCard, setPinnedCard] = useState<HoverCardInfo | null>(null);
  const [actionFocusIds, setActionFocusIds] = useState<number[]>([]);
  const [stackSourceId, setStackSourceId] = useState<number | null>(null);
  const [ckptA, setCkptA] = useState<string | null>(null);
  const [ckptB, setCkptB] = useState<string | null>(null);
  const [forkEdits, setForkEdits] = useState<ForkEdit[]>([]);

  const { target, onChangeStep } = props;
  const runDetail = useRunDetail(target.run_id);
  const checkpoints = (runDetail?.checkpoints ?? []).map((c) => c.name);
  const { effectiveA, effectiveB } = resolveDefaultCheckpoints({
    checkpoints,
    gameIteration: target.iteration,
    ckptA,
    ckptB,
  });

  const { overview } = useGameOverview(target.run_id, target.iteration, target.game_idx);

  const stepTarget = useMemo(
    () => ({
      runId: target.run_id,
      iteration: target.iteration,
      gameIdx: target.game_idx,
      step: target.step,
      pov,
      ckptA: effectiveA,
      ckptB: effectiveB,
    }),
    [target.run_id, target.iteration, target.game_idx, target.step, pov, effectiveA, effectiveB],
  );
  const { data, loading, error } = useGameStepData(stepTarget);
  const {
    result: forkResult,
    busy: forkBusy,
    error: forkError,
  } = useForkEvaluation(stepTarget, forkEdits, mode === "fork");
  const { timeline, maxPlayableStep, replayWarning } = resolveOverviewInfo(
    overview,
    data?.replay_warning,
  );

  const handleKey = useCallback(
    (event: KeyboardEvent): void => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setForkEdits([]);
        onChangeStep(Math.max(0, target.step - 1));
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        setForkEdits([]);
        onChangeStep(Math.min(maxPlayableStep, target.step + 1));
      } else if (event.key === "Escape") {
        setHoverInfo(null);
        setPinnedCard(null);
        if (drawer === null) setStackOpen(false);
        else setDrawer(null);
      }
    },
    [target.step, maxPlayableStep, drawer, onChangeStep],
  );
  useKeyboardShortcut(handleKey);

  const active = resolveActivePayload(mode, data, forkResult);
  const highlight = resolveHighlight(stackSourceId, actionFocusIds);
  const isPinned = isTargetPinned(props.watchlist, target);
  const forkWarnings = forkResult?.warnings ?? [];

  return (
    <div className="flex flex-col h-full min-h-0">
      <TabletopTopBar
        target={target}
        board={active === null ? null : active.board}
        replayWarning={replayWarning}
        pov={pov}
        mode={mode}
        onBack={props.onBackToGames}
        onChangePov={setPov}
        onChangeMode={(next): void => {
          setMode(next);
          if (next === "inspect") setForkEdits([]);
        }}
        onResetFork={(): void => setForkEdits([])}
      />

      {error ? (
        <div className="m-4 rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] p-3 text-xs text-[var(--loss)]">
          {error}
        </div>
      ) : null}

      {active ? (
        <TabletopWorkspaceBody
          board={active.board}
          evaluation={active.evaluation}
          baseEvaluation={active.base}
          pov={pov}
          mode={mode}
          stackOpen={stackOpen}
          drawer={drawer}
          hoverInfo={hoverInfo}
          pinnedCard={pinnedCard}
          highlightIds={highlight.ids}
          highlightTone={highlight.tone}
          timeline={timeline}
          maxPlayableStep={maxPlayableStep}
          checkpoints={checkpoints}
          ckptA={effectiveA}
          ckptB={effectiveB}
          forkBusy={forkBusy}
          forkEdits={forkEdits}
          forkWarnings={forkWarnings}
          forkError={forkError}
          pinned={isPinned}
          onToggleStack={(): void => setStackOpen((v) => !v)}
          onSetDrawer={setDrawer}
          onSetHover={setHoverInfo}
          onPinCard={setPinnedCard}
          onHoverSource={setStackSourceId}
          onHoverActionFocus={setActionFocusIds}
          onAdjustLife={(seat, delta): void =>
            setForkEdits((prev) =>
              updateLifeForkEdits({ edits: prev, board: active.board, seat, delta }),
            )
          }
          onSelectStep={(step): void => {
            setForkEdits([]);
            onChangeStep(step);
          }}
          onChangeCkptA={setCkptA}
          onChangeCkptB={setCkptB}
          onAddForkEdit={(edit): void => setForkEdits((prev) => [...prev, edit])}
          onRemoveForkEdit={(idx): void => setForkEdits((prev) => prev.filter((_, i) => i !== idx))}
          onPinPosition={(note): void => {
            void props.onPinPosition({ board: active.board, target, note }).catch(() => undefined);
          }}
        />
      ) : (
        <div className="flex-1 flex items-center justify-center text-xs text-[var(--text-3)]">
          {loading ? "Replaying managym board state…" : "Preparing Tabletop…"}
        </div>
      )}
    </div>
  );
}

export function TabletopView(props: {
  target: GameTarget | null;
  watchlist: WatchlistItem[];
  onChangeStep: (step: number) => void;
  onPinPosition: PinPositionFn;
  onBackToGames: () => void;
}): ReactNode {
  if (!props.target) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3 p-6 text-center">
        <div className="text-sm font-semibold">No game selected yet</div>
        <p className="text-xs text-[var(--text-2)] max-w-sm">
          Open any recorded game from the Games Explorer or a saved Watchlist position to inspect
          every zone and run live counterfactual forks.
        </p>
        <Button variant="primary" size="md" onClick={props.onBackToGames}>
          Open Games Explorer
        </Button>
      </div>
    );
  }
  return (
    <LoadedTabletop
      key={`${props.target.run_id}:${props.target.iteration}:${props.target.game_idx}`}
      target={props.target}
      watchlist={props.watchlist}
      onChangeStep={props.onChangeStep}
      onPinPosition={props.onPinPosition}
      onBackToGames={props.onBackToGames}
    />
  );
}

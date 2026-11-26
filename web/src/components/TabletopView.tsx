/** Centered Magic-authentic Tabletop workspace with peek-up Arena hands, zero-shift overlay Stack, and any-step Fork. */

import { AlertTriangle, Info, X } from "lucide-react";
import { type ReactNode, useCallback, useMemo, useState } from "react";
import { TabletopDock } from "@/components/TabletopDock";
import {
  CardZoomPreview,
  type DrawerTarget,
  HandFanView,
  type HoverCardInfo,
  LandPilesRow,
  PermanentsRow,
  StackOverlay,
  TurnTimelineBar,
  ZoneDrawerSheet,
  ZoneFlankView,
} from "@/components/TabletopParts";
import { Button, SegmentedControl } from "@/components/ui/Primitives";
import {
  useForkEvaluation,
  useGameOverview,
  useGameStepData,
  useKeyboardShortcut,
  useRunDetail,
} from "@/hooks/useStudioData";
import type {
  BoardState,
  BranchStepEvent,
  EvaluationPair,
  ForkEdit,
  ForkResponse,
  GameTarget,
  PovMode,
  TimelinePoint,
  WatchlistItem,
} from "@/lib/types";

type TabletopMode = "inspect" | "fork";

const POV_OPTIONS: ReadonlyArray<{ value: PovMode; label: string }> = [
  { value: "hero", label: "Hero POV" },
  { value: "omniscient", label: "Omniscient" },
];

const MODE_OPTIONS: ReadonlyArray<{ value: TabletopMode; label: string }> = [
  { value: "inspect", label: "Inspect" },
  { value: "fork", label: "Fork" },
];

function cornerStatusIndicator(
  isActing: boolean,
  isActive: boolean,
): { title?: string; dotClass: string } {
  if (isActing) return { title: "Currently deciding", dotClass: "bg-[var(--win-fill)]" };
  if (isActive) return { title: "Active player", dotClass: "bg-[var(--accent)]" };
  return { dotClass: "bg-[var(--border-strong)]" };
}

function PlayerCornerBadge(props: {
  label: string;
  life: number;
  isActive: boolean;
  isActing: boolean;
  manaPool: string;
  forkMode: boolean;
  canEditLife: boolean;
  onAdjustLife: (delta: number) => void;
}): ReactNode {
  const showLifeControls = props.forkMode && props.canEditLife;
  const status = cornerStatusIndicator(props.isActing, props.isActive);
  return (
    <div
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] bg-[var(--surface)]/95 shadow-[var(--shadow-sm)] ${
        props.isActive ? "border-[var(--accent)]" : "border-[var(--border)]"
      }`}
    >
      <span title={status.title} className={`w-2 h-2 rounded-full shrink-0 ${status.dotClass}`} />
      <span className="font-semibold whitespace-nowrap">{props.label}</span>
      <span
        className={`tabular inline-flex items-center gap-1 font-bold whitespace-nowrap ${
          props.life <= 5 ? "text-[var(--loss)]" : "text-[var(--text)]"
        }`}
      >
        {showLifeControls ? (
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
        <span>{props.life}</span>
        {showLifeControls ? (
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
      {props.manaPool.length > 0 ? (
        <span
          title="Floating mana in mana pool"
          className="tabular rounded bg-[var(--warn-soft)] text-[var(--warn)] px-1 text-[10px] font-bold whitespace-nowrap"
        >
          {props.manaPool}
        </span>
      ) : null}
    </div>
  );
}

function BattlefieldHalf(props: {
  side: "hero" | "opp";
  player: BoardState["hero"];
  permNames: ReadonlyMap<number, string>;
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  onOpenDrawer: (target: DrawerTarget) => void;
  onSetHover: (info: HoverCardInfo | null) => void;
  onPinCard: (info: HoverCardInfo | null) => void;
}): ReactNode {
  const { player } = props;
  const isOpp = props.side === "opp";

  const handEdge = (
    <div className="relative flex items-center justify-center min-w-0 shrink-0">
      <div className="w-full flex justify-center">
        <HandFanView
          side={props.side}
          cards={player.hand}
          hiddenCount={player.zone_counts.hand}
          hidden={isOpp ? player.hand_hidden : false}
          onHover={props.onSetHover}
          onPinCard={props.onPinCard}
        />
      </div>
    </div>
  );

  const landsRow = (
    <div className="flex items-center justify-center gap-3 min-h-[54px] min-w-0 overflow-x-auto no-scrollbar shrink-0">
      <LandPilesRow
        piles={player.lands}
        compact={player.other_permanents.length > 0}
        onHover={props.onSetHover}
      />
      {player.other_permanents.length > 0 ? (
        <div className="pl-3 shrink-0 border-l border-dashed border-[var(--border-strong)]">
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
      permNames={props.permNames}
      highlightIds={props.highlightIds}
      highlightTone={props.highlightTone}
      onHover={props.onSetHover}
      onPinCard={props.onPinCard}
    />
  );

  const rows = isOpp ? [landsRow, creaturesRow] : [creaturesRow, landsRow];
  return (
    <div className="h-full min-h-0 grid grid-cols-[minmax(0,1fr)_70px] gap-2">
      <div className="flex flex-col justify-between min-w-0 min-h-0 h-full gap-1">
        {isOpp ? handEdge : null}
        <div className="flex-none lg:flex-1 lg:min-h-0 flex flex-col justify-center gap-1">
          {rows}
        </div>
        {isOpp ? null : handEdge}
      </div>
      <ZoneFlankView side={props.side} player={player} onOpenDrawer={props.onOpenDrawer} />
    </div>
  );
}

function formatStepName(raw: string): string {
  return raw.replace(/([a-z])([A-Z])/g, "$1 $2");
}

function formatTurnBanner(board: BoardState): string {
  const stepLabel = formatStepName(board.step_name);
  if (board.turn === 0) return `${formatStepName(board.phase)} · ${stepLabel}`;
  const who = board.active_is_hero ? "Hero" : "Opp";
  return `${who} Turn ${board.turn} · ${stepLabel}`;
}

function BoardDetailsPopover(props: { board: BoardState; onClose: () => void }): ReactNode {
  const b = props.board;
  return (
    <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-40 w-[260px] rounded-lg border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-lg)] p-3 text-[11px] flex flex-col gap-2">
      <div className="flex items-center justify-between font-semibold text-xs border-b border-[var(--border)] pb-1.5">
        <span>Board Details</span>
        <button
          type="button"
          onClick={props.onClose}
          aria-label="Close board details"
          className="p-0.5 text-[var(--text-3)] hover:text-[var(--text)] cursor-pointer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="grid grid-cols-2 gap-1.5 tabular">
        <div className="text-[var(--text-3)]">Phase / Step</div>
        <div className="text-right font-medium">
          {b.phase} · {b.step_name}
        </div>
        <div className="text-[var(--text-3)]">Decision</div>
        <div className="text-right font-medium">{b.action_space_kind}</div>
        <div className="text-[var(--text-3)]">Hero Open Lands</div>
        <div className="text-right font-medium">{b.hero.open_lands ?? 0}</div>
        <div className="text-[var(--text-3)]">Hero Permanents</div>
        <div className="text-right font-medium">{b.hero.zone_counts.battlefield}</div>
        <div className="text-[var(--text-3)]">Opp Open Lands</div>
        <div className="text-right font-medium">{b.opp.open_lands ?? 0}</div>
        <div className="text-[var(--text-3)]">Opp Permanents</div>
        <div className="text-right font-medium">{b.opp.zone_counts.battlefield}</div>
        <div className="text-[var(--text-3)]">Hero Seat</div>
        <div className="text-right font-medium">Seat {b.hero_seat ?? 0}</div>
      </div>
    </div>
  );
}

function CenterTurnDivider(props: {
  turnLabel: string;
  stackCount: number;
  stackOpen: boolean;
  detailsOpen: boolean;
  onToggleStack: () => void;
  onToggleDetails: () => void;
}): ReactNode {
  return (
    <div className="flex flex-nowrap items-center gap-2 min-w-0 z-10">
      <div className="flex-1 h-px bg-[var(--border)] min-w-[12px]" />
      <span className="rounded-full border border-[var(--border)] bg-[var(--surface)]/95 px-3 py-0.5 text-[11px] font-semibold shadow-[var(--shadow-sm)] truncate">
        {props.turnLabel}
      </span>
      {props.stackCount > 0 ? (
        <button
          type="button"
          onClick={props.onToggleStack}
          className="tabular rounded-full border border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)] px-2.5 py-0.5 text-[10.5px] font-bold cursor-pointer whitespace-nowrap shrink-0"
        >
          Stack ({props.stackCount}) {props.stackOpen ? "▴" : "▾"}
        </button>
      ) : null}
      <button
        type="button"
        onClick={props.onToggleDetails}
        title="Board details (open mana, permanent counts, seat)"
        aria-label="Board details"
        className={`w-5 h-5 rounded-full border flex items-center justify-center cursor-pointer shrink-0 transition-colors ${
          props.detailsOpen
            ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
            : "border-[var(--border)] bg-[var(--surface)] text-[var(--text-3)] hover:text-[var(--text)]"
        }`}
      >
        <Info className="w-3 h-3" />
      </button>
      <div className="flex-1 h-px bg-[var(--border)] min-w-[12px]" />
    </div>
  );
}

function TabletopPlaymat(props: {
  board: BoardState;
  recordedStep: number;
  pov: PovMode;
  forkMode: boolean;
  stackOpen: boolean;
  detailsOpen: boolean;
  historyOpen: boolean;
  drawer: DrawerTarget | null;
  hoverInfo: HoverCardInfo | null;
  pinnedCard: HoverCardInfo | null;
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  timeline: TimelinePoint[];
  evalSource?: string;
  maxPlayableStep: number;
  onToggleStack: () => void;
  onToggleDetails: () => void;
  onCloseDetails: () => void;
  onToggleHistory: () => void;
  onCloseHistory: () => void;
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
  const permNames = new Map<number, string>([
    ...b.hero.creatures.map((p): [number, string] => [p.id, p.name]),
    ...b.opp.creatures.map((p): [number, string] => [p.id, p.name]),
  ]);

  return (
    <div className="flex-none lg:flex-1 min-w-0 lg:min-h-0 flex flex-col">
      <div className="flex-none lg:flex-1 lg:min-h-0 relative felt-surface px-3 py-1.5 flex flex-col lg:grid lg:grid-rows-[minmax(0,1fr)_auto_minmax(0,1fr)] gap-1 overflow-hidden">
        <div className="absolute top-2 left-3 z-20">
          <PlayerCornerBadge
            label="Opponent"
            life={b.opp.life}
            isActive={Boolean(b.opp.is_active)}
            isActing={b.actor_is_hero === false}
            manaPool={b.opp.mana_pool ?? ""}
            forkMode={props.forkMode}
            canEditLife={canEditLife}
            onAdjustLife={(d): void => props.onAdjustLife(oppSeat, d)}
          />
        </div>

        <BattlefieldHalf
          side="opp"
          player={b.opp}
          permNames={permNames}
          highlightIds={props.highlightIds}
          highlightTone={props.highlightTone}
          onOpenDrawer={props.onSetDrawer}
          onSetHover={props.onSetHover}
          onPinCard={props.onPinCard}
        />

        <CenterTurnDivider
          turnLabel={formatTurnBanner(b)}
          stackCount={b.stack.length}
          stackOpen={props.stackOpen && props.drawer === null}
          detailsOpen={props.detailsOpen}
          onToggleStack={props.onToggleStack}
          onToggleDetails={props.onToggleDetails}
        />

        <BattlefieldHalf
          side="hero"
          player={b.hero}
          permNames={permNames}
          highlightIds={props.highlightIds}
          highlightTone={props.highlightTone}
          onOpenDrawer={props.onSetDrawer}
          onSetHover={props.onSetHover}
          onPinCard={props.onPinCard}
        />

        <div className="absolute bottom-2 left-3 z-20">
          <PlayerCornerBadge
            label="Hero"
            life={b.hero.life}
            isActive={Boolean(b.hero.is_active)}
            isActing={b.actor_is_hero !== false}
            manaPool={b.hero.mana_pool ?? ""}
            forkMode={props.forkMode}
            canEditLife={canEditLife}
            onAdjustLife={(d): void => props.onAdjustLife(heroSeat, d)}
          />
        </div>

        <StackOverlay
          stack={b.stack}
          open={props.stackOpen && props.drawer === null}
          onClose={props.onToggleStack}
          onHoverSource={props.onHoverSource}
          onHoverCard={props.onSetHover}
        />

        {props.detailsOpen ? (
          <BoardDetailsPopover board={b} onClose={props.onCloseDetails} />
        ) : null}

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

      <TurnTimelineBar
        timeline={props.timeline}
        evalSource={props.evalSource}
        recentEvents={b.recent_events ?? []}
        currentStep={props.recordedStep}
        totalSteps={b.total_steps}
        maxPlayableStep={props.maxPlayableStep}
        currentTurn={b.turn}
        historyOpen={props.historyOpen}
        onToggleHistory={props.onToggleHistory}
        onCloseHistory={props.onCloseHistory}
        onSelectStep={props.onSelectStep}
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
  replayWarning: string | null;
  pov: PovMode;
  mode: TabletopMode;
  onBack: () => void;
  onChangePov: (pov: PovMode) => void;
  onChangeMode: (mode: TabletopMode) => void;
}): ReactNode {
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-1.5 border-b border-[var(--border)] bg-[var(--surface)] shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <Button variant="ghost" onClick={props.onBack}>
            ← Games
          </Button>
          <span className="text-[var(--border-strong)]">/</span>
          <span className="text-xs font-semibold truncate tabular">
            Seed #{props.target.seed} · {props.target.run_id.split("/").pop()} · u
            {props.target.iteration}
          </span>
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
    </>
  );
}

interface ReadyPayload {
  board: BoardState;
  evaluation: EvaluationPair;
  base: EvaluationPair;
  branchEvents: BranchStepEvent[];
  branchDone: boolean;
  nearestPriorityStep: number | null;
}

function resolveActivePayload(
  mode: TabletopMode,
  data: {
    board: BoardState;
    evaluation: EvaluationPair;
    nearest_priority_step?: number | null;
  } | null,
  forkResult: ForkResponse | null,
): ReadyPayload | null {
  if (mode === "fork" && forkResult !== null) {
    return {
      board: forkResult.board,
      evaluation: forkResult.fork_evaluation,
      base: forkResult.base_evaluation,
      branchEvents: forkResult.branch_events ?? [],
      branchDone: Boolean(forkResult.branch_done),
      nearestPriorityStep: forkResult.nearest_priority_step ?? data?.nearest_priority_step ?? null,
    };
  }
  if (data === null) return null;
  return {
    board: data.board,
    evaluation: data.evaluation,
    base: data.evaluation,
    branchEvents: [],
    branchDone: false,
    nearestPriorityStep: data.nearest_priority_step ?? null,
  };
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
  afterBranch: number;
}): ForkEdit[] {
  const { edits, board, seat, delta, afterBranch } = args;
  const heroSeat = board.hero_seat === 1 ? 1 : 0;
  const visibleLife = seat === heroSeat ? board.hero.life : board.opp.life;
  const nextLife = Math.max(1, Math.min(99, visibleLife + delta));
  if (nextLife === visibleLife) return edits;
  const kept = edits.filter(
    (e) => !(e.op === "set_life" && e.seat === seat && (e.after_branch ?? 0) === afterBranch),
  );
  return [...kept, { op: "set_life", seat, value: nextLife, after_branch: afterBranch }];
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
): {
  timeline: TimelinePoint[];
  evalSource?: string;
  maxPlayableStep: number;
  replayWarning: string | null;
} {
  if (!overview) {
    return {
      timeline: [],
      evalSource: undefined,
      maxPlayableStep: 0,
      replayWarning: stepWarn ?? null,
    };
  }
  const fallbackMax = Math.max(0, overview.total_steps - 1);
  return {
    timeline: overview.timeline,
    evalSource: overview.eval_source,
    maxPlayableStep: overview.max_playable_step ?? fallbackMax,
    replayWarning: stepWarn ?? overview.replay_warning ?? null,
  };
}

function topLegalActionIndex(board: BoardState, evaluation: EvaluationPair): number | null {
  const probs = evaluation.primary.probs;
  const actions = board.legal_actions;
  if (actions.length === 0 || probs.length === 0) return null;
  let bestIdx = 0;
  let bestProb = probs[0] ?? -1;
  for (let i = 1; i < actions.length; i += 1) {
    const p = probs[i] ?? -1;
    if (p > bestProb) {
      bestProb = p;
      bestIdx = i;
    }
  }
  return actions[bestIdx]?.index ?? null;
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
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [drawer, setDrawer] = useState<DrawerTarget | null>(null);
  const [hoverInfo, setHoverInfo] = useState<HoverCardInfo | null>(null);
  const [pinnedCard, setPinnedCard] = useState<HoverCardInfo | null>(null);
  const [actionFocusIds, setActionFocusIds] = useState<number[]>([]);
  const [stackSourceId, setStackSourceId] = useState<number | null>(null);
  const [ckptA, setCkptA] = useState<string | null>(null);
  const [ckptB, setCkptB] = useState<string | null>(null);
  const [forkEdits, setForkEdits] = useState<ForkEdit[]>([]);
  const [branchActions, setBranchActions] = useState<number[]>([]);

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
  } = useForkEvaluation({
    target: stepTarget,
    edits: forkEdits,
    branchActions,
    enabled: mode === "fork",
  });
  const { timeline, evalSource, maxPlayableStep, replayWarning } = resolveOverviewInfo(
    overview,
    data?.replay_warning,
  );

  const clearFork = useCallback((): void => {
    setForkEdits([]);
    setBranchActions([]);
  }, []);

  const handleKey = useCallback(
    (event: KeyboardEvent): void => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        clearFork();
        onChangeStep(Math.max(0, target.step - 1));
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        clearFork();
        onChangeStep(Math.min(maxPlayableStep, target.step + 1));
      } else if (event.key === "Escape") {
        setHoverInfo(null);
        setPinnedCard(null);
        setDetailsOpen(false);
        setHistoryOpen(false);
        if (drawer === null) setStackOpen(false);
        else setDrawer(null);
      }
    },
    [target.step, maxPlayableStep, drawer, onChangeStep, clearFork],
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
        replayWarning={replayWarning}
        pov={pov}
        mode={mode}
        onBack={props.onBackToGames}
        onChangePov={setPov}
        onChangeMode={(next): void => {
          setMode(next);
          if (next === "inspect") clearFork();
        }}
      />

      {error ? (
        <div className="m-4 rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] p-3 text-xs text-[var(--loss)]">
          {error}
        </div>
      ) : null}

      {active ? (
        <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
          <TabletopPlaymat
            board={active.board}
            recordedStep={target.step}
            pov={pov}
            forkMode={mode === "fork"}
            stackOpen={stackOpen}
            detailsOpen={detailsOpen}
            historyOpen={historyOpen}
            drawer={drawer}
            hoverInfo={hoverInfo}
            pinnedCard={pinnedCard}
            highlightIds={highlight.ids}
            highlightTone={highlight.tone}
            timeline={timeline}
            evalSource={evalSource}
            maxPlayableStep={maxPlayableStep}
            onToggleStack={(): void => setStackOpen((v) => !v)}
            onToggleDetails={(): void => setDetailsOpen((v) => !v)}
            onCloseDetails={(): void => setDetailsOpen(false)}
            onToggleHistory={(): void => setHistoryOpen((v) => !v)}
            onCloseHistory={(): void => setHistoryOpen(false)}
            onSetDrawer={setDrawer}
            onSetHover={setHoverInfo}
            onPinCard={setPinnedCard}
            onHoverSource={setStackSourceId}
            onAdjustLife={(seat, delta): void =>
              setForkEdits((prev) =>
                updateLifeForkEdits({
                  edits: prev,
                  board: active.board,
                  seat,
                  delta,
                  afterBranch: branchActions.length,
                }),
              )
            }
            onSelectStep={(step): void => {
              clearFork();
              onChangeStep(step);
            }}
          />

          <TabletopDock
            board={active.board}
            evaluation={active.evaluation}
            baseEvaluation={active.base}
            checkpoints={checkpoints}
            ckptA={effectiveA}
            ckptB={effectiveB}
            onChangeCkptA={setCkptA}
            onChangeCkptB={setCkptB}
            forkMode={mode === "fork"}
            forkBusy={forkBusy}
            forkEdits={forkEdits}
            branchEvents={active.branchEvents}
            branchDone={active.branchDone}
            nearestPriorityStep={active.nearestPriorityStep}
            forkWarnings={forkWarnings}
            forkError={forkError}
            onAddForkEdit={(edit): void =>
              setForkEdits((prev) => [...prev, { ...edit, after_branch: branchActions.length }])
            }
            onRemoveForkEdit={(idx): void =>
              setForkEdits((prev) => prev.filter((_, i) => i !== idx))
            }
            onBranchAction={(actionIdx): void => setBranchActions((prev) => [...prev, actionIdx])}
            onStepModelTopMove={(): void => {
              const topIdx = topLegalActionIndex(active.board, active.evaluation);
              if (topIdx !== null) setBranchActions((prev) => [...prev, topIdx]);
            }}
            onPopBranchAction={(): void => {
              const nextLen = Math.max(0, branchActions.length - 1);
              setBranchActions((prev) => prev.slice(0, -1));
              setForkEdits((prev) => prev.filter((e) => (e.after_branch ?? 0) <= nextLen));
            }}
            onResetFork={clearFork}
            onJumpToStep={(step): void => {
              clearFork();
              onChangeStep(step);
            }}
            onHoverActionFocus={setActionFocusIds}
            onPinPosition={(note): void => {
              void props
                .onPinPosition({
                  board: { ...active.board, step: target.step },
                  target,
                  note,
                })
                .catch(() => undefined);
            }}
            pinned={isPinned}
          />
        </div>
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
          Open any playable game from Games &amp; Mistakes or a saved Watchlist position to inspect
          the board and run counterfactual branches.
        </p>
        <Button variant="primary" size="md" onClick={props.onBackToGames}>
          Open Games &amp; Mistakes
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

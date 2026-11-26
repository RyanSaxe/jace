/** Crowded MTG Tabletop & Inline Counterfactual Fork workspace with live checkpoint diffing. */

import { GitBranch, RotateCcw } from "lucide-react";
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
  useGameOverview,
  useGameStepData,
  useKeyboardShortcut,
  useRunDetail,
} from "@/hooks/useStudioData";
import { postFork } from "@/lib/api";
import type {
  BoardState,
  EvaluationPair,
  ForkEdit,
  ForkResponse,
  GameTarget,
  PovMode,
  TimelinePoint,
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
  forkMode: boolean;
  onAdjustLife: (delta: number) => void;
  hand: ReactNode;
}): ReactNode {
  const lowLife = props.life <= 5;
  return (
    <div className="flex items-center justify-between gap-2 min-h-[30px] min-w-0">
      <div className="flex items-center gap-1.5 shrink-0">
        <span className="text-[11.5px] font-semibold whitespace-nowrap">{props.label}</span>
        <span
          className={`tabular inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11.5px] font-bold ${
            lowLife
              ? "border-[var(--loss)] bg-[var(--loss-soft)] text-[var(--loss)]"
              : "border-[var(--border)] bg-[var(--surface)]"
          }`}
        >
          {props.forkMode ? (
            <button
              type="button"
              onClick={(): void => props.onAdjustLife(-1)}
              aria-label={`Decrease ${props.label} life`}
              className="w-4 h-4 rounded-full bg-[var(--fork-soft)] text-[var(--fork)] flex items-center justify-center cursor-pointer"
            >
              −
            </button>
          ) : null}
          <span>{props.life} life</span>
          {props.forkMode ? (
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
        <span className="tabular rounded border border-[var(--border)] bg-[var(--surface)] px-1.5 py-0.5 text-[10px] text-[var(--text-3)]">
          {props.permCount} perms
        </span>
      </div>
      <div className="flex-1 min-w-0 flex justify-end">{props.hand}</div>
    </div>
  );
}

function TabletopPlaymat(props: {
  board: BoardState;
  pov: PovMode;
  forkMode: boolean;
  stackOpen: boolean;
  drawer: DrawerTarget | null;
  hoverInfo: HoverCardInfo | null;
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  timeline: TimelinePoint[];
  onToggleStack: () => void;
  onSetDrawer: (drawer: DrawerTarget | null) => void;
  onSetHover: (info: HoverCardInfo | null) => void;
  onHoverSource: (permId: number | null) => void;
  onAdjustLife: (seat: 0 | 1, life: number, delta: number) => void;
  onSelectStep: (step: number) => void;
}): ReactNode {
  const b = props.board;
  return (
    <div className="flex-1 min-w-0 min-h-0 flex flex-col">
      <div className="flex-1 min-h-0 relative felt-surface p-2.5 grid grid-cols-[minmax(0,1fr)_auto] gap-2.5 overflow-hidden">
        <div className="flex flex-col justify-between min-w-0 min-h-0">
          <div className="flex-1 min-h-0 grid grid-cols-[minmax(0,1fr)_74px] gap-2">
            <div className="flex flex-col justify-between min-w-0 min-h-0 gap-1">
              <PlayerStripHeader
                label="Opponent"
                life={b.opp.life}
                permCount={b.opp.zone_counts.battlefield}
                forkMode={props.forkMode}
                onAdjustLife={(d): void => props.onAdjustLife(1, b.opp.life, d)}
                hand={
                  <HandFanView
                    cards={b.opp.hand}
                    hiddenCount={b.opp.zone_counts.hand}
                    hidden={b.opp.hand_hidden}
                    onHover={props.onSetHover}
                  />
                }
              />
              <div className="flex-1 min-h-0 flex flex-col justify-center gap-1.5">
                <div className="flex items-center justify-between gap-2 h-[66px] min-w-0 overflow-x-auto">
                  <LandPilesRow
                    piles={b.opp.lands}
                    compact={b.opp.other_permanents.length > 0}
                    onHover={props.onSetHover}
                  />
                  {b.opp.other_permanents.length > 0 ? (
                    <div className="pl-2 shrink-0 border-l border-dashed border-[var(--border-strong)]">
                      <PermanentsRow
                        permanents={b.opp.other_permanents}
                        compact
                        highlightIds={props.highlightIds}
                        highlightTone={props.highlightTone}
                        onHover={props.onSetHover}
                      />
                    </div>
                  ) : null}
                </div>
                <PermanentsRow
                  permanents={b.opp.creatures}
                  highlightIds={props.highlightIds}
                  highlightTone={props.highlightTone}
                  onHover={props.onSetHover}
                />
              </div>
            </div>
            <ZoneFlankView side="opp" player={b.opp} onOpenDrawer={props.onSetDrawer} />
          </div>

          <div className="flex items-center gap-2 my-1">
            <div className="flex-1 h-px bg-[var(--border)]" />
            <span className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 py-0.5 text-[11px] font-semibold shadow-[var(--shadow-sm)] whitespace-nowrap">
              Turn {b.turn} · {b.phase} · {b.step_name}
            </span>
            {b.stack.length > 0 ? (
              <button
                type="button"
                onClick={props.onToggleStack}
                className="tabular rounded-full border border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)] px-2.5 py-0.5 text-[10.5px] font-bold cursor-pointer whitespace-nowrap"
              >
                Stack ({b.stack.length}) {props.stackOpen ? "▴" : "▾"}
              </button>
            ) : null}
            <div className="flex-1 h-px bg-[var(--border)]" />
          </div>

          <div className="flex-1 min-h-0 grid grid-cols-[minmax(0,1fr)_74px] gap-2">
            <div className="flex flex-col justify-between min-w-0 min-h-0 gap-1">
              <div className="flex-1 min-h-0 flex flex-col justify-center gap-1.5">
                <PermanentsRow
                  permanents={b.hero.creatures}
                  highlightIds={props.highlightIds}
                  highlightTone={props.highlightTone}
                  onHover={props.onSetHover}
                />
                <div className="flex items-center justify-between gap-2 h-[66px] min-w-0 overflow-x-auto">
                  <LandPilesRow
                    piles={b.hero.lands}
                    compact={b.hero.other_permanents.length > 0}
                    onHover={props.onSetHover}
                  />
                  {b.hero.other_permanents.length > 0 ? (
                    <div className="pl-2 shrink-0 border-l border-dashed border-[var(--border-strong)]">
                      <PermanentsRow
                        permanents={b.hero.other_permanents}
                        compact
                        highlightIds={props.highlightIds}
                        highlightTone={props.highlightTone}
                        onHover={props.onSetHover}
                      />
                    </div>
                  ) : null}
                </div>
              </div>
              <PlayerStripHeader
                label="Hero"
                life={b.hero.life}
                permCount={b.hero.zone_counts.battlefield}
                forkMode={props.forkMode}
                onAdjustLife={(d): void => props.onAdjustLife(0, b.hero.life, d)}
                hand={
                  <HandFanView
                    cards={b.hero.hand}
                    hiddenCount={b.hero.zone_counts.hand}
                    hidden={false}
                    onHover={props.onSetHover}
                  />
                }
              />
            </div>
            <ZoneFlankView side="hero" player={b.hero} onOpenDrawer={props.onSetDrawer} />
          </div>
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
          onClose={(): void => props.onSetDrawer(null)}
          onHover={props.onSetHover}
        />

        <CardZoomPreview info={props.hoverInfo} />
      </div>

      <TurnTimelineBar
        timeline={props.timeline}
        currentStep={b.step}
        totalSteps={b.total_steps}
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
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  timeline: TimelinePoint[];
  checkpoints: string[];
  ckptA: string;
  ckptB: string;
  forkBusy: boolean;
  forkEdits: ForkEdit[];
  forkWarnings: string[];
  pinned: boolean;
  onToggleStack: () => void;
  onSetDrawer: (drawer: DrawerTarget | null) => void;
  onSetHover: (info: HoverCardInfo | null) => void;
  onHoverSource: (permId: number | null) => void;
  onHoverActionFocus: (ids: number[]) => void;
  onAdjustLife: (seat: 0 | 1, life: number, delta: number) => void;
  onSelectStep: (step: number) => void;
  onChangeCkptA: (v: string) => void;
  onChangeCkptB: (v: string) => void;
  onAddForkEdit: (edit: ForkEdit) => void;
  onPinPosition: () => void;
}): ReactNode {
  const winPct = Math.round(props.evaluation.primary.wdl.win * 100);
  const drawPct = Math.round(props.evaluation.primary.wdl.draw * 100);
  const lossPct = Math.max(0, 100 - winPct - drawPct);

  return (
    <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
      <div className="hidden sm:flex">
        <aside
          aria-label="Win probability bar"
          className="w-[40px] shrink-0 border-r border-[var(--border)] bg-[var(--surface)] py-2 px-1.5 flex flex-col items-center gap-1.5"
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
      </div>

      <TabletopPlaymat
        board={props.board}
        pov={props.pov}
        forkMode={props.mode === "fork"}
        stackOpen={props.stackOpen}
        drawer={props.drawer}
        hoverInfo={props.hoverInfo}
        highlightIds={props.highlightIds}
        highlightTone={props.highlightTone}
        timeline={props.timeline}
        onToggleStack={props.onToggleStack}
        onSetDrawer={props.onSetDrawer}
        onSetHover={props.onSetHover}
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
        onAddForkEdit={props.onAddForkEdit}
        onHoverActionFocus={props.onHoverActionFocus}
        onPinPosition={props.onPinPosition}
        pinned={props.pinned}
      />
    </div>
  );
}

function resolveDefaultCheckpoints(
  checkpoints: string[],
  ckptA: string,
  ckptB: string,
): { effectiveA: string; effectiveB: string } {
  const last = checkpoints[checkpoints.length - 1] ?? "";
  const first = checkpoints.length > 1 ? (checkpoints[0] ?? "") : "";
  return {
    effectiveA: ckptA === "" ? last : ckptA,
    effectiveB: ckptB === "" ? first : ckptB,
  };
}

function TabletopTopBar(props: {
  target: GameTarget;
  board: BoardState | null;
  pov: PovMode;
  mode: TabletopMode;
  onBack: () => void;
  onChangePov: (pov: PovMode) => void;
  onChangeMode: (mode: TabletopMode) => void;
  onResetFork: () => void;
}): ReactNode {
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-[var(--border)] bg-[var(--surface)]">
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

      {props.mode === "fork" ? (
        <div className="flex items-center justify-between px-3.5 py-1.5 bg-[var(--fork-soft)] border-b border-[var(--fork)]/25 text-[11px] text-[var(--fork)] font-medium">
          <div className="flex items-center gap-1.5">
            <GitBranch className="w-3.5 h-3.5" />
            <span>
              Counterfactual Fork active — use <strong>− / +</strong> life steppers or add cards in
              the right dock. Recorded games on disk are never mutated.
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

function listOrEmpty<T>(items: T[] | undefined): T[] {
  return items ?? [];
}

function LoadedTabletop(props: {
  target: GameTarget;
  onChangeStep: (step: number) => void;
  onPinPosition: (board: BoardState, target: GameTarget) => Promise<void>;
  onBackToGames: () => void;
}): ReactNode {
  const [pov, setPov] = useState<PovMode>("hero");
  const [mode, setMode] = useState<TabletopMode>("inspect");
  const [stackOpen, setStackOpen] = useState(true);
  const [drawer, setDrawer] = useState<DrawerTarget | null>(null);
  const [hoverInfo, setHoverInfo] = useState<HoverCardInfo | null>(null);
  const [actionFocusIds, setActionFocusIds] = useState<number[]>([]);
  const [stackSourceId, setStackSourceId] = useState<number | null>(null);
  const [ckptA, setCkptA] = useState("");
  const [ckptB, setCkptB] = useState("");
  const [forkEdits, setForkEdits] = useState<ForkEdit[]>([]);
  const [forkResult, setForkResult] = useState<ForkResponse | null>(null);
  const [forkBusy, setForkBusy] = useState(false);
  const [pinnedKey, setPinnedKey] = useState("");

  const { target } = props;
  const runDetail = useRunDetail(target.run_id);
  const checkpoints = listOrEmpty(runDetail?.checkpoints).map((c) => c.name);
  const { effectiveA, effectiveB } = resolveDefaultCheckpoints(checkpoints, ckptA, ckptB);

  const { overview } = useGameOverview(target.run_id, target.iteration, target.game_idx);

  const stepTarget = useMemo(
    () => ({
      runId: target.run_id,
      iteration: target.iteration,
      gameIdx: target.game_idx,
      step: target.step,
      pov,
      ckptA: effectiveA || undefined,
      ckptB: effectiveB || undefined,
    }),
    [target.run_id, target.iteration, target.game_idx, target.step, pov, effectiveA, effectiveB],
  );
  const { data, loading, error } = useGameStepData(stepTarget);

  const handleKey = useCallback(
    (event: KeyboardEvent): void => {
      if (!overview) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setForkEdits([]);
        setForkResult(null);
        props.onChangeStep(Math.max(0, target.step - 1));
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        setForkEdits([]);
        setForkResult(null);
        props.onChangeStep(Math.min(Math.max(0, overview.total_steps - 1), target.step + 1));
      } else if (event.key === "Escape") {
        setDrawer(null);
      }
    },
    [target.step, overview, props],
  );
  useKeyboardShortcut(handleKey);

  const active = resolveActivePayload(mode, data, forkResult);
  const applyFork = (nextEdits: ForkEdit[]): void => {
    setForkEdits(nextEdits);
    if (nextEdits.length === 0) {
      setForkResult(null);
      return;
    }
    setForkBusy(true);
    void postFork({ ...stepTarget, edits: nextEdits })
      .then((res) => {
        setForkResult(res);
      })
      .finally(() => {
        setForkBusy(false);
      });
  };

  const highlight = resolveHighlight(stackSourceId, actionFocusIds);
  const currentPinKey = `${target.run_id}:${target.iteration}:${target.game_idx}:${target.step}`;
  const timeline = listOrEmpty(overview?.timeline);
  const warnings = listOrEmpty(forkResult?.warnings);
  const headerBoard = active === null ? null : active.board;

  return (
    <div className="flex flex-col h-full min-h-0">
      <TabletopTopBar
        target={target}
        board={headerBoard}
        pov={pov}
        mode={mode}
        onBack={props.onBackToGames}
        onChangePov={setPov}
        onChangeMode={(next): void => {
          setMode(next);
          if (next === "inspect") {
            setForkEdits([]);
            setForkResult(null);
          }
        }}
        onResetFork={(): void => {
          applyFork([]);
        }}
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
          highlightIds={highlight.ids}
          highlightTone={highlight.tone}
          timeline={timeline}
          checkpoints={checkpoints}
          ckptA={effectiveA}
          ckptB={effectiveB}
          forkBusy={forkBusy}
          forkEdits={forkEdits}
          forkWarnings={warnings}
          pinned={pinnedKey === currentPinKey}
          onToggleStack={(): void => setStackOpen((v) => !v)}
          onSetDrawer={setDrawer}
          onSetHover={setHoverInfo}
          onHoverSource={setStackSourceId}
          onHoverActionFocus={setActionFocusIds}
          onAdjustLife={(seat, life, delta): void => {
            const nextLife = Math.max(1, Math.min(99, life + delta));
            const kept = forkEdits.filter((e) => !(e.op === "set_life" && e.seat === seat));
            applyFork([...kept, { op: "set_life", seat, value: nextLife }]);
          }}
          onSelectStep={(step): void => {
            setForkEdits([]);
            setForkResult(null);
            props.onChangeStep(step);
          }}
          onChangeCkptA={setCkptA}
          onChangeCkptB={setCkptB}
          onAddForkEdit={(edit): void => {
            applyFork([...forkEdits, edit]);
          }}
          onPinPosition={(): void => {
            void props.onPinPosition(active.board, target).then(() => setPinnedKey(currentPinKey));
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
  onChangeStep: (step: number) => void;
  onPinPosition: (board: BoardState, target: GameTarget) => Promise<void>;
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
      target={props.target}
      onChangeStep={props.onChangeStep}
      onPinPosition={props.onPinPosition}
      onBackToGames={props.onBackToGames}
    />
  );
}

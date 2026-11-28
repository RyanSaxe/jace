/** Desktop Tabletop Analysis Board, Direct-on-Board Edit Mode, Belief & Search inspection, Policy Matrix, and 390px 100dvh Mobile Tabletop. */

import {
  ChevronLeft,
  ChevronRight,
  LayoutGrid,
  Pencil,
  Sliders,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { type ReactNode, useCallback, useMemo, useRef, useState } from "react";
import { PinPositionPopover, PolicyMatrixModal } from "@/components/PolicyMatrixModal";
import {
  BeliefEditorDrawer,
  type BeliefEditorDrawerProps,
  countActiveOverrides,
  useBeliefSearchState,
} from "@/components/belief";
import {
  type ReadyPayload,
  TabletopDock,
  type TabletopDockProps,
  buildInitialMatrixModels,
  defaultPinLabel,
  extractBoardKeyCards,
  mergeCheckpoints,
  resolveActivePayload,
  resolveDefaultCheckpoints,
  resolveOverviewInfo,
  topLegalActionIndex,
  updateLifeForkEdits,
} from "@/components/TabletopDock";
import {
  type ActiveOverlay,
  type OverviewInfo,
  type PlaymatController,
  TabletopAlerts,
  TabletopPlaymat,
  TabletopTopBar,
} from "@/components/TabletopOverlays";
import type { DrawerTarget, HandCardEditTarget, HoverCardInfo } from "@/components/TabletopParts";
import { Button } from "@/components/ui/Primitives";
import {
  useCardCatalog,
  useForkEvaluation,
  useGameOverview,
  useGameStepData,
  useKeyboardShortcut,
  useRunDetail,
} from "@/hooks/useStudioData";
import { postPolicyMatrix } from "@/lib/api";
import type {
  BoardState,
  ForkEdit,
  GameTarget,
  GlobalCheckpointOption,
  PolicyMatrixResponse,
  PovMode,
  WatchlistItem,
} from "@/lib/types";

export interface TabletopViewProps {
  target: GameTarget | null;
  checkpoints: GlobalCheckpointOption[];
  watchlist: WatchlistItem[];
  onChangeStep: (step: number) => void;
  onPinPosition: (args: {
    board: BoardState;
    target: GameTarget;
    label?: string;
    note?: string;
    tag?: string;
    keyCards?: string[];
    summary?: string;
  }) => Promise<void>;
  onBackToGames: () => void;
}

interface MatrixUiState {
  open: boolean;
  models: string[];
  data: PolicyMatrixResponse | null;
  loading: boolean;
  error: string | null;
}

const INITIAL_MATRIX: MatrixUiState = {
  open: false,
  models: [],
  data: null,
  loading: false,
  error: null,
};

function resolveHighlight(
  stackSourceId: number | null,
  actionFocusIds: number[],
): { ids: ReadonlySet<number>; tone: "target" | "source" } {
  if (stackSourceId === null) return { ids: new Set(actionFocusIds), tone: "target" };
  return { ids: new Set([stackSourceId]), tone: "source" };
}

function submitWatchlistPin(args: {
  board: BoardState;
  target: GameTarget;
  payload: { label: string; note: string; tag: string };
  onPinPosition: TabletopViewProps["onPinPosition"];
  setPinSaving: (v: boolean) => void;
  closePin: () => void;
}): void {
  const { board, target, payload } = args;
  args.setPinSaving(true);
  const summary = `Hero ${board.hero.life} vs Opp ${board.opp.life} · ${board.hero.creatures.length + board.opp.creatures.length} creatures`;
  void args
    .onPinPosition({
      board: { ...board, step: target.step },
      target,
      label: payload.label.trim() || defaultPinLabel(board),
      note: payload.note.trim() || undefined,
      tag: payload.tag.trim() || undefined,
      keyCards: extractBoardKeyCards(board),
      summary,
    })
    .then(() => {
      args.setPinSaving(false);
      args.closePin();
    })
    .catch(() => args.setPinSaving(false));
}

function handleTabletopKey(
  event: KeyboardEvent,
  ctx: {
    step: number;
    maxStep: number;
    matrixOpen: boolean;
    onEscape: () => void;
    onStep: (step: number) => void;
  },
): void {
  if (event.key === "Escape") {
    ctx.onEscape();
    return;
  }
  if (ctx.matrixOpen) return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    ctx.onStep(Math.max(0, ctx.step - 1));
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    ctx.onStep(Math.min(ctx.maxStep, ctx.step + 1));
  }
}

function requestPolicyMatrix(args: {
  target: GameTarget;
  pov: PovMode;
  edits: ForkEdit[];
  branchActions: number[];
  models: string[];
  seqRef: { current: number };
  setMatrix: (fn: (prev: MatrixUiState) => MatrixUiState) => void;
}): void {
  const { models, seqRef, setMatrix, target } = args;
  if (models.length === 0) {
    setMatrix((p) => ({ ...p, models, data: null, error: null, loading: false }));
    return;
  }
  const seq = seqRef.current + 1;
  seqRef.current = seq;
  setMatrix((p) => ({ ...p, models, loading: true, error: null }));
  void postPolicyMatrix({
    runId: target.run_id,
    iteration: target.iteration,
    gameIdx: target.game_idx,
    step: target.step,
    pov: args.pov,
    edits: args.edits,
    branchActions: args.branchActions,
    models,
  })
    .then((res) => {
      if (seqRef.current === seq) setMatrix((p) => ({ ...p, data: res, loading: false }));
    })
    .catch((err: unknown) => {
      if (seqRef.current === seq) {
        setMatrix((p) => ({
          ...p,
          error: err instanceof Error ? err.message : "Policy matrix failed",
          loading: false,
        }));
      }
    });
}

function MobilePolicySheet(props: {
  open: boolean;
  onClose: () => void;
  dockProps: TabletopDockProps;
}): ReactNode {
  if (!props.open) return null;
  return (
    <div className="fixed inset-0 z-40 flex flex-col justify-end lg:hidden">
      <button
        type="button"
        aria-label="Dismiss policy sheet"
        onClick={props.onClose}
        className="flex-1 bg-black/45 border-0"
      />
      <div className="h-[74dvh] max-h-[620px] rounded-t-[14px] border-t border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-lg)] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-3 py-1.5 border-b border-[var(--border)] shrink-0">
          <span className="text-[11px] font-bold text-[var(--text-2)]">Decision Analysis</span>
          <button
            type="button"
            onClick={props.onClose}
            aria-label="Close policy sheet"
            className="p-1 rounded text-[var(--text-2)] hover:bg-[var(--surface-2)] cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="flex-1 min-h-0 overflow-hidden">
          <TabletopDock
            {...props.dockProps}
            className="w-full h-full flex flex-col min-h-0 overflow-hidden bg-[var(--surface)]"
          />
        </div>
      </div>
    </div>
  );
}

function ReadyWorkspace(props: {
  target: GameTarget;
  active: ReadyPayload;
  pinSaving: boolean;
  branchActions: number[];
  cardCatalog: string[];
  overview: OverviewInfo;
  playmatCtrl: PlaymatController;
  dockProps: TabletopDockProps;
  beliefDrawerProps: BeliefEditorDrawerProps;
  matrix: MatrixUiState;
  onToggleEdit: () => void;
  onPinSave: (p: { label: string; note: string; tag: string }) => void;
  onLoadMatrix: (models: string[]) => void;
  onCloseMatrix: () => void;
}): ReactNode {
  const { active, playmatCtrl: ctrl, dockProps, target, matrix, beliefDrawerProps } = props;
  const upperStep = Math.max(
    0,
    Math.min(Math.max(0, active.board.total_steps - 1), props.overview.maxPlayableStep),
  );
  const btnCls =
    "min-h-[34px] h-[34px] px-2 rounded-md border border-[var(--border)] bg-[var(--surface-2)] text-[11px] font-medium inline-flex items-center gap-1 cursor-pointer disabled:opacity-40";
  const beliefCount = countActiveOverrides(beliefDrawerProps.override);
  return (
    <>
      {ctrl.overlay === "pin" ? (
        <PinPositionPopover
          board={active.board}
          saving={props.pinSaving}
          onClose={(): void => ctrl.setOverlay("none")}
          onSave={props.onPinSave}
        />
      ) : null}
      <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-hidden">
        <TabletopPlaymat
          board={active.board}
          recordedStep={target.step}
          forkEdits={dockProps.forkEdits}
          cardCatalog={props.cardCatalog}
          nearestPriorityStep={active.nearestPriorityStep}
          overview={props.overview}
          ctrl={ctrl}
        />
        <BeliefEditorDrawer {...beliefDrawerProps} />
        <div className="hidden lg:flex lg:w-[340px] shrink-0 h-full min-h-0">
          <TabletopDock {...dockProps} />
        </div>
      </div>
      <div className="flex lg:hidden h-[44px] shrink-0 border-t border-[var(--border)] bg-[var(--surface)] px-2 items-center justify-between gap-1">
        <div className="flex items-center gap-1 shrink-0">
          <button
            type="button"
            disabled={target.step <= 0}
            aria-label="Previous step"
            onClick={(): void => ctrl.selectStep(Math.max(0, target.step - 1))}
            className={`${btnCls} min-w-[34px] w-[34px] justify-center px-0`}
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="tabular text-[10.5px] font-semibold text-[var(--text-2)] px-1">
            T{active.board.turn} #{target.step}
          </span>
          <button
            type="button"
            disabled={target.step >= upperStep}
            aria-label="Next step"
            onClick={(): void => ctrl.selectStep(Math.min(upperStep, target.step + 1))}
            className={`${btnCls} min-w-[34px] w-[34px] justify-center px-0`}
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <button
            type="button"
            onClick={(): void =>
              ctrl.setOverlay(ctrl.overlay === "mobilePolicy" ? "none" : "mobilePolicy")
            }
            className={
              ctrl.overlay === "mobilePolicy"
                ? `${btnCls} !bg-[var(--accent)] !text-white !border-[var(--accent)]`
                : btnCls
            }
          >
            <SlidersHorizontal className="w-3 h-3" /> Policy ({active.board.legal_actions.length})
          </button>
          <button
            type="button"
            onClick={dockProps.belief?.onToggleEditor}
            className={
              beliefDrawerProps.open || beliefCount > 0
                ? `${btnCls} !bg-[var(--fork-soft)] !text-[var(--fork)] !border-[var(--fork)]`
                : btnCls
            }
          >
            <Sliders className="w-3 h-3" /> Belief
            {beliefCount > 0 ? ` (${beliefCount})` : ""}
          </button>
          <button
            type="button"
            onClick={props.onToggleEdit}
            className={
              ctrl.editMode
                ? `${btnCls} !bg-[var(--fork-soft)] !text-[var(--fork)] !border-[var(--fork)]`
                : btnCls
            }
          >
            <Pencil className="w-3 h-3" /> Edit
            {dockProps.forkEdits.length > 0 ? ` (${dockProps.forkEdits.length})` : ""}
          </button>
          <button type="button" onClick={dockProps.onOpenPolicyMatrix} className={btnCls}>
            <LayoutGrid className="w-3 h-3" /> Matrix
          </button>
        </div>
      </div>
      <MobilePolicySheet
        open={ctrl.overlay === "mobilePolicy"}
        onClose={(): void => ctrl.setOverlay("none")}
        dockProps={dockProps}
      />
      <PolicyMatrixModal
        open={matrix.open}
        board={active.board}
        target={target}
        pov={ctrl.pov}
        edits={dockProps.forkEdits}
        branchActions={props.branchActions}
        checkpoints={dockProps.checkpoints}
        selectedModels={matrix.models}
        matrixData={matrix.data}
        matrixLoading={matrix.loading}
        matrixError={matrix.error}
        defaultPlayoutModel={dockProps.ckptA}
        onAddModel={(ref): void => {
          if (!matrix.models.includes(ref)) props.onLoadMatrix([...matrix.models, ref]);
        }}
        onRemoveModel={(ref): void => props.onLoadMatrix(matrix.models.filter((m) => m !== ref))}
        onRefreshMatrix={(): void => props.onLoadMatrix(matrix.models)}
        onClose={props.onCloseMatrix}
      />
    </>
  );
}

function LoadedTabletop(
  props: Omit<TabletopViewProps, "target"> & { target: GameTarget },
): ReactNode {
  const [pov, setPov] = useState<PovMode>("hero");
  const [editMode, setEditMode] = useState(false);
  const [overlay, setOverlay] = useState<ActiveOverlay>("none");
  const [stackOpen, setStackOpen] = useState<boolean>(() =>
    typeof window === "undefined" ? true : window.innerWidth >= 640,
  );
  const [handEditTarget, setHandEditTarget] = useState<HandCardEditTarget | null>(null);
  const [pinSaving, setPinSaving] = useState(false);
  const [drawer, setDrawer] = useState<DrawerTarget | null>(null);
  const [hoverInfo, setHoverInfo] = useState<HoverCardInfo | null>(null);
  const [pinnedCard, setPinnedCard] = useState<HoverCardInfo | null>(null);
  const [actionFocusIds, setActionFocusIds] = useState<number[]>([]);
  const [stackSourceId, setStackSourceId] = useState<number | null>(null);
  const [ckptA, setCkptA] = useState<string | null>(null);
  const [ckptB, setCkptB] = useState<string | null>(null);
  const [forkEdits, setForkEdits] = useState<ForkEdit[]>([]);
  const [branchActions, setBranchActions] = useState<number[]>([]);
  const [matrix, setMatrix] = useState<MatrixUiState>(INITIAL_MATRIX);
  const matrixReqSeq = useRef(0);

  const { target, onChangeStep } = props;
  const runDetail = useRunDetail(target.run_id);
  const availableCheckpoints = useMemo(
    () => mergeCheckpoints(props.checkpoints, runDetail, target.run_id),
    [props.checkpoints, runDetail, target.run_id],
  );
  const cardCatalog = useCardCatalog(target.run_id);
  const { effectiveA, effectiveB } = resolveDefaultCheckpoints({
    checkpoints: availableCheckpoints,
    runId: target.run_id,
    gameIteration: target.iteration,
    initialA: target.ckpt_a,
    initialB: target.ckpt_b,
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
      ckptB: effectiveB === "none" ? undefined : effectiveB,
    }),
    [target.run_id, target.iteration, target.game_idx, target.step, pov, effectiveA, effectiveB],
  );
  const { data, loading, error } = useGameStepData(stepTarget);
  const hasMutations = forkEdits.length > 0 || branchActions.length > 0;
  const {
    result: forkResult,
    busy: forkBusy,
    error: forkError,
  } = useForkEvaluation({
    target: stepTarget,
    edits: forkEdits,
    branchActions,
    enabled: hasMutations,
  });
  const overviewInfo = resolveOverviewInfo(overview, data?.replay_warning);
  const beliefControllers = useBeliefSearchState(target, effectiveA);

  const clearFork = useCallback((): void => {
    setForkEdits([]);
    setBranchActions([]);
    setHandEditTarget(null);
    setOverlay((cur) => (cur === "addCard" ? "none" : cur));
  }, []);

  const loadPolicyMatrix = (models: string[]): void =>
    requestPolicyMatrix({
      target,
      pov,
      edits: forkEdits,
      branchActions,
      models,
      seqRef: matrixReqSeq,
      setMatrix,
    });

  const openMatrix = (): void => {
    const models =
      matrix.models.length > 0
        ? matrix.models
        : buildInitialMatrixModels({
            effectiveA,
            effectiveB,
            checkpoints: availableCheckpoints,
            runId: target.run_id,
          });
    setMatrix((prev) => ({ ...prev, open: true, models }));
    setOverlay("none");
    loadPolicyMatrix(models);
  };

  const handleKey = useCallback(
    (event: KeyboardEvent): void =>
      handleTabletopKey(event, {
        step: target.step,
        maxStep: overviewInfo.maxPlayableStep,
        matrixOpen: matrix.open,
        onEscape: (): void => {
          setMatrix((prev) => ({ ...prev, open: false }));
          setOverlay("none");
          setHandEditTarget(null);
          setHoverInfo(null);
          setPinnedCard(null);
          if (drawer === null) setStackOpen(false);
          else setDrawer(null);
        },
        onStep: (next): void => {
          clearFork();
          onChangeStep(next);
        },
      }),
    [target.step, overviewInfo.maxPlayableStep, drawer, matrix.open, onChangeStep, clearFork],
  );
  useKeyboardShortcut(handleKey);

  const active = resolveActivePayload(hasMutations, data, forkResult);
  const highlight = resolveHighlight(stackSourceId, actionFocusIds);
  const pinned = props.watchlist.some(
    (i) =>
      i.run_id === target.run_id &&
      i.iteration === target.iteration &&
      i.game_idx === target.game_idx &&
      i.step === target.step,
  );
  const toggleEdit = (): void => {
    setEditMode((v) => !v);
    setHandEditTarget(null);
    setOverlay((cur) => (cur === "addCard" ? "none" : cur));
  };
  const selectStep = (step: number): void => {
    clearFork();
    onChangeStep(step);
  };

  return (
    <div className="relative flex flex-col h-full min-h-0 overflow-hidden">
      <TabletopTopBar
        target={target}
        turn={active?.board.turn ?? null}
        pov={pov}
        editMode={editMode}
        editCount={forkEdits.length}
        hasMutations={hasMutations}
        pinned={pinned}
        pinOpen={overlay === "pin"}
        onBack={props.onBackToGames}
        onChangePov={setPov}
        onToggleEdit={toggleEdit}
        onResetFork={clearFork}
        onTogglePin={(): void => {
          if (active) setOverlay((c) => (c === "pin" ? "none" : "pin"));
        }}
      />
      <TabletopAlerts warning={overviewInfo.replayWarning} error={error} />
      {active ? (
        <ReadyWorkspace
          target={target}
          active={active}
          pinSaving={pinSaving}
          branchActions={branchActions}
          cardCatalog={cardCatalog}
          overview={overviewInfo}
          matrix={matrix}
          onToggleEdit={toggleEdit}
          onPinSave={(payload): void =>
            submitWatchlistPin({
              board: active.board,
              target,
              payload,
              onPinPosition: props.onPinPosition,
              setPinSaving,
              closePin: (): void => setOverlay("none"),
            })
          }
          onLoadMatrix={loadPolicyMatrix}
          onCloseMatrix={(): void => setMatrix((prev) => ({ ...prev, open: false }))}
          playmatCtrl={{
            pov,
            editMode,
            overlay,
            stackOpen,
            handEditTarget,
            drawer,
            hoverInfo,
            pinnedCard,
            highlightIds: highlight.ids,
            highlightTone: highlight.tone,
            setOverlay,
            toggleStack: (): void => setStackOpen((v) => !v),
            setHandEditTarget,
            addForkEdit: (edit): void =>
              setForkEdits((prev) => [...prev, { ...edit, after_branch: branchActions.length }]),
            removeForkEdit: (idx): void => setForkEdits((prev) => prev.filter((_, i) => i !== idx)),
            resetFork: clearFork,
            setDrawer,
            setHover: setHoverInfo,
            setPinnedCard,
            setHoverSource: setStackSourceId,
            adjustLife: (seat, delta): void =>
              setForkEdits((prev) =>
                updateLifeForkEdits({
                  edits: prev,
                  board: active.board,
                  seat,
                  delta,
                  afterBranch: branchActions.length,
                }),
              ),
            selectStep,
          }}
          dockProps={{
            runId: target.run_id,
            board: active.board,
            evaluation: active.evaluation,
            baseEvaluation: active.base,
            checkpoints: availableCheckpoints,
            ckptA: effectiveA,
            ckptB: effectiveB,
            onChangeCkptA: setCkptA,
            onChangeCkptB: setCkptB,
            forkBusy,
            forkEdits,
            branchEvents: active.branchEvents,
            branchDone: active.branchDone,
            forkWarnings: forkResult?.warnings ?? [],
            forkError,
            onBranchAction: (idx): void => setBranchActions((prev) => [...prev, idx]),
            onStepModelTopMove: (): void => {
              const topIdx = topLegalActionIndex(active.board, active.evaluation);
              if (topIdx !== null) setBranchActions((prev) => [...prev, topIdx]);
            },
            onPopBranchAction: (): void => {
              const n = Math.max(0, branchActions.length - 1);
              setBranchActions((prev) => prev.slice(0, -1));
              setForkEdits((prev) => prev.filter((e) => (e.after_branch ?? 0) <= n));
            },
            onResetFork: clearFork,
            onSelectStep: selectStep,
            onHoverActionFocus: setActionFocusIds,
            onOpenPolicyMatrix: openMatrix,
            belief: beliefControllers.dock,
          }}
          beliefDrawerProps={beliefControllers.drawer}
        />
      ) : (
        <div className="flex-1 flex items-center justify-center text-xs text-[var(--text-3)]">
          {loading ? "Replaying managym board state…" : "Preparing Tabletop…"}
        </div>
      )}
    </div>
  );
}

export function TabletopView(props: TabletopViewProps): ReactNode {
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
      checkpoints={props.checkpoints}
      watchlist={props.watchlist}
      onChangeStep={props.onChangeStep}
      onPinPosition={props.onPinPosition}
      onBackToGames={props.onBackToGames}
    />
  );
}

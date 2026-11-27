/** Floating Policy Evolution Matrix modal with arbitrary N checkpoints and progressive MPS playouts. */

import { LayoutGrid, Plus, RefreshCw, X } from "lucide-react";
import { type ReactNode, useRef, useState } from "react";
import { Button } from "@/components/ui/Primitives";
import { useAbortOnUnmount, useResetOnKey } from "@/hooks/useStudioData";
import type { postPlayoutBatch } from "@/lib/api";
import type {
  BoardState,
  ForkEdit,
  GameTarget,
  GlobalCheckpointOption,
  PolicyMatrixResponse,
  PovMode,
} from "@/lib/types";
import { CheckpointSelect } from "./CheckpointSelect";
import {
  EMPTY_PLAYOUT,
  PlayoutControlBar,
  type PlayoutState,
  clampDeals,
  streamPlayoutChunks,
} from "./PolicyMatrixPlayouts";
import { MatrixBodySection } from "./PolicyMatrixTable";

export interface PolicyMatrixModalProps {
  open: boolean;
  board: BoardState;
  target: GameTarget;
  pov: PovMode;
  edits: ForkEdit[];
  branchActions: number[];
  checkpoints: GlobalCheckpointOption[];
  selectedModels: string[];
  matrixData: PolicyMatrixResponse | null;
  matrixLoading: boolean;
  matrixError: string | null;
  defaultPlayoutModel: string;
  onAddModel: (ref: string) => void;
  onRemoveModel: (ref: string) => void;
  onRefreshMatrix: () => void;
  onClose: () => void;
}

function MatrixAddModelControl(props: {
  checkpoints: GlobalCheckpointOption[];
  selectedModels: string[];
  currentRunId: string;
  onAddModel: (ref: string) => void;
}): ReactNode {
  const [pickerOpen, setPickerOpen] = useState(false);
  const [candidate, setCandidate] = useState("");
  const excluded = new Set(props.selectedModels);
  const firstAvailable = props.checkpoints.find((c) => !excluded.has(c.ref))?.ref;
  const fallbackRef = excluded.has("heuristic") ? "" : "heuristic";
  const defaultNext = firstAvailable ?? fallbackRef;
  const effectiveCandidate = candidate && !excluded.has(candidate) ? candidate : defaultNext;

  if (!pickerOpen) {
    return (
      <Button
        size="sm"
        disabled={!effectiveCandidate}
        onClick={(): void => setPickerOpen(true)}
        className="text-[11px]"
      >
        <Plus className="w-3.5 h-3.5" />
        Add model
      </Button>
    );
  }

  return (
    <div className="inline-flex items-center gap-1.5 rounded-md border border-[var(--accent)] bg-[var(--surface)] px-1.5 py-0.5">
      <CheckpointSelect
        value={effectiveCandidate}
        onChange={setCandidate}
        checkpoints={props.checkpoints}
        currentRunId={props.currentRunId}
        excludeRefs={excluded}
        ariaLabel="Choose model to add to matrix"
        className="h-6 rounded border-0 bg-transparent text-[11px] font-semibold tabular max-w-[180px]"
      />
      <Button
        size="sm"
        variant="primary"
        disabled={!effectiveCandidate}
        onClick={(): void => {
          if (effectiveCandidate) props.onAddModel(effectiveCandidate);
          setCandidate("");
          setPickerOpen(false);
        }}
        className="px-2 py-0.5 text-[10.5px]"
      >
        Add
      </Button>
      <button
        type="button"
        onClick={(): void => setPickerOpen(false)}
        aria-label="Cancel adding model"
        className="p-0.5 text-[var(--text-3)] hover:text-[var(--text)] cursor-pointer"
      >
        <X className="w-3 h-3" />
      </button>
    </div>
  );
}

function MatrixModalHeader(props: {
  turn: number;
  stepName: string;
  checkpoints: GlobalCheckpointOption[];
  selectedModels: string[];
  currentRunId: string;
  matrixLoading: boolean;
  onAddModel: (ref: string) => void;
  onRefreshMatrix: () => void;
  onClose: () => void;
}): ReactNode {
  const count = props.selectedModels.length;
  return (
    <div className="px-4 py-2.5 border-b border-[var(--border)] flex flex-wrap items-center justify-between gap-2 shrink-0">
      <div className="flex items-center gap-2 min-w-0">
        <LayoutGrid className="w-4 h-4 text-[var(--accent)] shrink-0" />
        <div className="min-w-0">
          <h2 className="text-xs sm:text-[13px] font-bold m-0 truncate">Policy Evolution Matrix</h2>
          <div className="text-[10.5px] text-[var(--text-3)] tabular truncate">
            T{props.turn} · {props.stepName} · {count} model{count === 1 ? "" : "s"}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-1.5 flex-wrap">
        <MatrixAddModelControl
          checkpoints={props.checkpoints}
          selectedModels={props.selectedModels}
          currentRunId={props.currentRunId}
          onAddModel={props.onAddModel}
        />
        <Button
          size="sm"
          disabled={props.matrixLoading || count === 0}
          onClick={props.onRefreshMatrix}
          title="Re-evaluate matrix columns"
        >
          <RefreshCw className={`w-3 h-3 ${props.matrixLoading ? "animate-spin" : ""}`} />
        </Button>
        <button
          type="button"
          onClick={props.onClose}
          aria-label="Close Policy Matrix"
          className="w-7 h-7 rounded-md flex items-center justify-center text-[var(--text-2)] hover:bg-[var(--surface-2)] cursor-pointer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

function resolvePlayoutModel(
  override: string | null,
  defaultModel: string,
  selectedModels: string[],
): string {
  if (override) return override;
  if (defaultModel) return defaultModel;
  return selectedModels[0] ?? "heuristic";
}

function makePlayoutRequest(
  props: PolicyMatrixModalProps,
  modelRef: string,
): Omit<Parameters<typeof postPlayoutBatch>[0], "deals" | "batchOffset"> {
  return {
    runId: props.target.run_id,
    iteration: props.target.iteration,
    gameIdx: props.target.game_idx,
    step: props.target.step,
    pov: props.pov,
    edits: props.edits,
    branchActions: props.branchActions,
    modelRef,
  };
}

export function PolicyMatrixModal(props: PolicyMatrixModalProps): ReactNode {
  const [playoutModelOverride, setPlayoutModelOverride] = useState<string | null>(null);
  const [dealsInput, setDealsInput] = useState("32");
  const [playout, setPlayout] = useState<PlayoutState>(EMPTY_PLAYOUT);
  const abortRef = useRef<AbortController | null>(null);
  const effectivePlayoutModel = resolvePlayoutModel(
    playoutModelOverride,
    props.defaultPlayoutModel,
    props.selectedModels,
  );
  const resetKey = `${props.open ? "1" : "0"}|${props.target.run_id}|${props.target.iteration}|${props.target.game_idx}|${props.target.step}|${props.pov}|${props.edits.length}|${props.branchActions.join(",")}|${effectivePlayoutModel}`;
  useAbortOnUnmount(abortRef);
  useResetOnKey(resetKey, (): void => {
    abortRef.current?.abort();
    abortRef.current = null;
    setPlayout(EMPTY_PLAYOUT);
  });

  if (!props.open) return null;
  const actions = props.matrixData?.legal_actions ?? props.board.legal_actions;
  const columns = props.matrixData?.columns ?? [];
  const hasPlayoutColumn = playout.running || playout.completedDeals > 0;
  const canRunPlayout = !props.board.actions_hidden && actions.length > 0;

  const abortAndStop = (): void => {
    abortRef.current?.abort();
    abortRef.current = null;
    setPlayout((prev) => ({ ...prev, running: false }));
  };

  const closeModal = (): void => {
    abortAndStop();
    props.onClose();
  };

  const startPlayout = (): void => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const totalDeals = clampDeals(dealsInput);
    setDealsInput(String(totalDeals));
    setPlayout({ ...EMPTY_PLAYOUT, running: true, targetDeals: totalDeals });
    void streamPlayoutChunks({
      baseReq: makePlayoutRequest(props, effectivePlayoutModel),
      targetDeals: totalDeals,
      signal: controller.signal,
      onUpdate: setPlayout,
      onFinish: (): void => {
        if (abortRef.current === controller) abortRef.current = null;
      },
    });
  };

  return (
    <dialog
      open
      aria-modal="true"
      aria-label="Policy Evolution Matrix"
      className="fixed inset-0 z-50 m-0 h-full w-full max-h-none max-w-none border-0 bg-transparent p-2 sm:p-5 flex items-center justify-center text-[var(--text)]"
    >
      <button
        type="button"
        aria-label="Close Policy Matrix backdrop"
        onClick={closeModal}
        className="fixed inset-0 bg-black/45 backdrop-blur-[1px] border-0 cursor-default"
      />
      <div className="relative z-10 w-full max-w-[1040px] max-h-[88dvh] rounded-[12px] border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-lg)] flex flex-col overflow-hidden">
        <MatrixModalHeader
          turn={props.board.turn}
          stepName={props.board.step_name}
          checkpoints={props.checkpoints}
          selectedModels={props.selectedModels}
          currentRunId={props.target.run_id}
          matrixLoading={props.matrixLoading}
          onAddModel={props.onAddModel}
          onRefreshMatrix={props.onRefreshMatrix}
          onClose={closeModal}
        />

        {props.matrixError ? (
          <div className="mx-4 mt-3 rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] px-3 py-2 text-xs text-[var(--loss)]">
            {props.matrixError}
          </div>
        ) : null}

        <MatrixBodySection
          actionsHidden={Boolean(props.board.actions_hidden)}
          selectedModels={props.selectedModels}
          matrixLoading={props.matrixLoading}
          actions={actions}
          columns={columns}
          checkpoints={props.checkpoints}
          currentRunId={props.target.run_id}
          playout={playout}
          hasPlayoutColumn={hasPlayoutColumn}
          onRemoveModel={props.onRemoveModel}
        />

        <PlayoutControlBar
          checkpoints={props.checkpoints}
          currentRunId={props.target.run_id}
          playoutModel={effectivePlayoutModel}
          dealsInput={dealsInput}
          playout={playout}
          canRunPlayout={canRunPlayout}
          onChangeModel={setPlayoutModelOverride}
          onChangeDeals={setDealsInput}
          onStart={startPlayout}
          onStop={abortAndStop}
        />
      </div>
    </dialog>
  );
}

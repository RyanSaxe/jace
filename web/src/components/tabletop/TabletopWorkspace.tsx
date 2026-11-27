/** Ready Tabletop workspace layout combining playmat, Belief drawer, desktop/mobile Decision Analysis dock, and Policy Matrix modal. */

import {
  ChevronLeft,
  ChevronRight,
  LayoutGrid,
  Pencil,
  Sliders,
  SlidersHorizontal,
  X,
} from "lucide-react";
import type { ReactNode } from "react";
import {
  BeliefEditorDrawer,
  type BeliefEditorDrawerProps,
  countActiveOverrides,
} from "@/components/belief";
import type { GameTarget } from "@/lib/types";
import { PolicyMatrixModal } from "./PolicyMatrixModal";
import { TabletopDock, type TabletopDockProps } from "./TabletopDock";
import { TabletopPlaymat } from "./TabletopPlaymat";
import { PinPositionPopover } from "./TabletopPopovers";
import type { ReadyPayload } from "./tabletopHelpers";
import type { MatrixUiState, OverviewInfo, PlaymatController } from "./tabletopTypes";

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

export function ReadyWorkspace(props: {
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

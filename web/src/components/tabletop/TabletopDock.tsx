/** Right 340px single-purpose Decision Analysis pane (zero tabs): cross-run model diff, Policy/Search switch, Turn-0 V0, win bars, ranked moves & Move Log. */

import {
  ChevronDown,
  ChevronUp,
  FastForward,
  GitBranch,
  LayoutGrid,
  RotateCcw,
  Undo2,
} from "lucide-react";
import { type ReactNode, useState } from "react";
import { BeliefDockBody, type TabletopBeliefDockProps } from "@/components/belief";
import { Button } from "@/components/ui/Primitives";
import type {
  BoardState,
  BranchStepEvent,
  EvaluationPair,
  ForkEdit,
  GlobalCheckpointOption,
  RecentEvent,
} from "@/lib/types";
import { CheckpointSelect } from "./CheckpointSelect";
import { RankedMovesList } from "./TabletopRankedMoves";
import { deltaToneClass, formatCheckpointShort, formatSignedPct } from "./tabletopHelpers";

export interface TabletopDockProps {
  runId: string;
  board: BoardState;
  evaluation: EvaluationPair;
  baseEvaluation: EvaluationPair;
  checkpoints: GlobalCheckpointOption[];
  ckptA: string;
  ckptB: string;
  onChangeCkptA: (v: string) => void;
  onChangeCkptB: (v: string) => void;
  forkBusy: boolean;
  forkEdits: ForkEdit[];
  branchEvents: BranchStepEvent[];
  branchDone: boolean;
  forkWarnings: string[];
  forkError: string | null;
  onBranchAction: (actionIdx: number) => void;
  onStepModelTopMove: () => void;
  onPopBranchAction: () => void;
  onResetFork: () => void;
  onSelectStep: (step: number) => void;
  onHoverActionFocus: (ids: number[]) => void;
  onOpenPolicyMatrix: () => void;
  belief?: TabletopBeliefDockProps;
  className?: string;
}

function WinEstimateBar(props: {
  label: string;
  winPct: number;
  barColor: string;
  delta?: number;
}): ReactNode {
  const showDelta = props.delta !== undefined && props.delta !== 0;
  const barWidth = `${Math.max(2, props.winPct)}%`;
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center justify-between gap-2 text-[10.5px]">
        <span className="font-semibold text-[var(--text-2)] truncate">{props.label}</span>
        <span className="tabular flex items-center gap-1.5 shrink-0">
          <strong className="text-[var(--text)]">{props.winPct}%</strong>
          {showDelta ? (
            <span className={deltaToneClass(props.delta ?? 0)}>
              {formatSignedPct(props.delta ?? 0)}
            </span>
          ) : null}
        </span>
      </div>
      <div className="h-[7px] rounded-full bg-[var(--surface-2)] overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-200"
          style={{ width: barWidth, background: props.barColor }}
        />
      </div>
    </div>
  );
}

function WinEstimateSection(props: {
  labelA: string;
  labelB: string;
  showCompare: boolean;
  showEditShift: boolean;
  evaluation: EvaluationPair;
  baseEvaluation: EvaluationPair;
  forkBusy: boolean;
}): ReactNode {
  const curA = Math.round(props.evaluation.primary.wdl.win * 100);
  const curB = Math.round(props.evaluation.compare.wdl.win * 100);
  const baseA = Math.round(props.baseEvaluation.primary.wdl.win * 100);
  const drawPct = Math.round(props.evaluation.primary.wdl.draw * 100);

  if (props.showEditShift) {
    return (
      <div className="px-3 py-2.5 border-b border-[var(--border)] bg-[var(--fork-soft)]/30 flex flex-col gap-1.5">
        <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--fork)]">
          <span>Hero Win Estimate · Edit Shift</span>
          {props.forkBusy ? <span className="tabular">Re-scoring…</span> : null}
        </div>
        <WinEstimateBar label={`Before edit (${props.labelA})`} winPct={baseA} barColor="#d97706" />
        <WinEstimateBar
          label={`After edit (${props.labelA})`}
          winPct={curA}
          barColor="var(--fork-fill)"
          delta={curA - baseA}
        />
      </div>
    );
  }

  return (
    <div className="px-3 py-2.5 border-b border-[var(--border)] flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)]">
        <span>Hero Win Estimate</span>
        <span className="tabular text-[var(--text-2)]">
          {drawPct > 0 ? `${drawPct}% Draw` : `${Math.max(0, 100 - curA)}% Loss`}
        </span>
      </div>
      <WinEstimateBar
        label={props.labelA}
        winPct={curA}
        barColor="var(--accent)"
        delta={props.showCompare ? curA - curB : undefined}
      />
      {props.showCompare ? (
        <WinEstimateBar label={props.labelB} winPct={curB} barColor="#d97706" />
      ) : null}
    </div>
  );
}

function BranchStrip(props: {
  branchEvents: BranchStepEvent[];
  editCount: number;
  canStepTop: boolean;
  forkBusy: boolean;
  onStepModelTopMove: () => void;
  onPopBranchAction: () => void;
  onResetFork: () => void;
}): ReactNode {
  if (props.branchEvents.length === 0 && props.editCount === 0) return null;
  const parts: string[] = [];
  if (props.editCount > 0) parts.push(`${props.editCount} edit${props.editCount === 1 ? "" : "s"}`);
  if (props.branchEvents.length > 0) parts.push(`+${props.branchEvents.length} branched`);
  return (
    <div className="px-3 py-2 border-b border-[var(--border)] bg-[var(--fork-soft)]/25 flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2 text-[10.5px]">
        <span className="inline-flex items-center gap-1 font-semibold text-[var(--fork)] truncate">
          <GitBranch className="w-3 h-3 shrink-0" />
          {parts.join(" · ")}
        </span>
        <div className="flex items-center gap-1 shrink-0">
          {props.canStepTop ? (
            <Button
              size="sm"
              variant="fork"
              disabled={props.forkBusy}
              onClick={props.onStepModelTopMove}
              title="Play model's top ranked move"
              className="px-2 py-0.5 text-[10px]"
            >
              <FastForward className="w-2.5 h-2.5" />
              Top
            </Button>
          ) : null}
          {props.branchEvents.length > 0 ? (
            <button
              type="button"
              onClick={props.onPopBranchAction}
              title="Undo last branched action"
              className="inline-flex items-center gap-0.5 rounded border border-[var(--border)] bg-[var(--surface)] px-1.5 py-0.5 text-[10px] font-medium hover:border-[var(--border-strong)] cursor-pointer"
            >
              <Undo2 className="w-2.5 h-2.5" />
              Undo
            </button>
          ) : null}
          <button
            type="button"
            onClick={props.onResetFork}
            title="Reset all edits and branches"
            className="inline-flex items-center gap-0.5 rounded border border-[var(--border)] bg-[var(--surface)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-2)] hover:text-[var(--loss)] cursor-pointer"
          >
            <RotateCcw className="w-2.5 h-2.5" />
            Reset
          </button>
        </div>
      </div>
      {props.branchEvents.length > 0 ? (
        <div className="flex flex-wrap gap-1">
          {props.branchEvents.map((ev) => (
            <span
              key={ev.branch_index}
              className="inline-flex items-center gap-1 rounded bg-[var(--surface)] border border-[var(--fork)]/30 px-1.5 py-0.5 text-[9.5px]"
            >
              <span className="tabular font-bold text-[var(--fork)]">+{ev.branch_index + 1}</span>
              <span className="max-w-[140px] truncate">{ev.label}</span>
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function CollapsibleMoveLog(props: {
  recentEvents: RecentEvent[];
  branchEvents: BranchStepEvent[];
  onSelectStep: (step: number) => void;
}): ReactNode {
  const [open, setOpen] = useState(false);
  const totalCount = props.recentEvents.length + props.branchEvents.length;

  return (
    <div className="shrink-0 border-t border-[var(--border)] bg-[var(--surface)]">
      <button
        type="button"
        onClick={(): void => setOpen((v) => !v)}
        aria-expanded={open}
        className="w-full px-3 py-2 flex items-center justify-between text-[11px] font-semibold text-[var(--text-2)] hover:text-[var(--text)] hover:bg-[var(--surface-2)]/60 cursor-pointer transition-colors"
      >
        <span className="tabular">Move Log ({totalCount})</span>
        {open ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3.5 h-3.5" />}
      </button>

      {open ? (
        <div className="max-h-[180px] overflow-y-auto border-t border-[var(--border)] px-2 py-1.5 flex flex-col gap-1 bg-[var(--bg)]/60">
          {totalCount === 0 ? (
            <div className="py-3 text-center text-[10.5px] text-[var(--text-3)]">
              No prior moves recorded yet.
            </div>
          ) : null}
          {props.branchEvents.toReversed().map((ev) => (
            <div
              key={`branch-${ev.branch_index}`}
              className="flex items-center gap-1.5 rounded px-1.5 py-1 text-[10.5px] bg-[var(--fork-soft)]/50"
            >
              <span className="tabular text-[9.5px] font-bold text-[var(--fork)] shrink-0">
                +{ev.branch_index + 1}
              </span>
              <span
                className={`font-semibold shrink-0 ${
                  ev.actor_side === "hero" ? "text-[var(--win)]" : "text-[var(--loss)]"
                }`}
              >
                {ev.actor_side === "hero" ? "Hero" : "Opp"}:
              </span>
              <span className="truncate">{ev.label}</span>
            </div>
          ))}
          {props.recentEvents.toReversed().map((ev) => (
            <button
              key={`${ev.step}-${ev.text}`}
              type="button"
              onClick={(): void => props.onSelectStep(ev.step)}
              className="flex items-center gap-1.5 rounded px-1.5 py-1 text-left text-[10.5px] hover:bg-[var(--surface-2)] cursor-pointer"
            >
              <span className="tabular text-[9.5px] text-[var(--text-3)] shrink-0">
                T{ev.turn} #{ev.step}
              </span>
              <span
                className={`font-semibold shrink-0 ${
                  ev.actor_side === "hero" ? "text-[var(--win)]" : "text-[var(--loss)]"
                }`}
              >
                {ev.actor_side === "hero" ? "Hero" : "Opp"}:
              </span>
              <span className="truncate">{ev.text}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function ModelSelectorsHeader(props: {
  runId: string;
  checkpoints: GlobalCheckpointOption[];
  ckptA: string;
  ckptB: string;
  showCompare: boolean;
  actionsHidden: boolean;
  evaluation: EvaluationPair;
  winDelta: number;
  forkError: string | null;
  warnings: string[];
  onChangeCkptA: (v: string) => void;
  onChangeCkptB: (v: string) => void;
}): ReactNode {
  return (
    <div className="px-3 py-2 border-b border-[var(--border)] flex flex-col gap-1.5 shrink-0">
      <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-1.5">
        <CheckpointSelect
          value={props.ckptA}
          onChange={props.onChangeCkptA}
          checkpoints={props.checkpoints}
          currentRunId={props.runId}
          ariaLabel="Model A checkpoint"
          className="h-7 rounded-md border border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)] px-1.5 text-[11px] font-semibold tabular min-w-0"
        />
        <span className="text-[10px] font-medium text-[var(--text-3)]">vs</span>
        <CheckpointSelect
          value={props.ckptB}
          onChange={props.onChangeCkptB}
          checkpoints={props.checkpoints}
          currentRunId={props.runId}
          includeNone
          noneLabel="None"
          ariaLabel="Model B checkpoint"
          className="h-7 rounded-md border border-amber-500/50 bg-[var(--warn-soft)]/50 text-[var(--text)] px-1.5 text-[11px] font-semibold tabular min-w-0"
        />
      </div>

      {!props.actionsHidden && props.showCompare ? (
        <div className="flex items-center justify-between text-[10px] tabular text-[var(--text-3)]">
          <span title="Total Variation and KL(A || B) in nats">
            TV {props.evaluation.diff.tv.toFixed(2)} · KL {props.evaluation.diff.kl.toFixed(2)}
          </span>
          <span className={deltaToneClass(props.winDelta)}>
            Win Δ {formatSignedPct(props.winDelta)}
          </span>
        </div>
      ) : null}

      {props.forkError ? (
        <div className="rounded bg-[var(--loss-soft)] text-[var(--loss)] px-2 py-1 text-[10px]">
          {props.forkError}
        </div>
      ) : null}
      {props.warnings.map((warn) => (
        <div
          key={warn}
          className="rounded bg-[var(--warn-soft)] text-[var(--warn)] px-2 py-1 text-[10px]"
        >
          {warn}
        </div>
      ))}
    </div>
  );
}

function canStepTopMove(props: TabletopDockProps): boolean {
  if (props.branchDone || props.board.actions_hidden) return false;
  return props.board.legal_actions.length > 0 && props.evaluation.primary.probs.length > 0;
}

export function TabletopDock(props: TabletopDockProps): ReactNode {
  const showCompare = props.ckptB !== "none" && props.ckptB !== "same";
  const hasMutations = props.forkEdits.length > 0 || props.branchEvents.length > 0;
  const showEditCompare = props.forkEdits.length > 0 && props.branchEvents.length === 0;
  const labelA = formatCheckpointShort(props.ckptA, props.checkpoints, props.runId);
  const labelB = formatCheckpointShort(props.ckptB, props.checkpoints, props.runId);
  const winDelta =
    Math.round(props.evaluation.primary.wdl.win * 100) -
    Math.round(props.evaluation.compare.wdl.win * 100);
  const allWarnings = [...(props.evaluation.warnings ?? []), ...props.forkWarnings];

  return (
    <aside
      aria-label="Decision analysis"
      className={
        props.className ??
        "w-full lg:w-[340px] shrink-0 border-t lg:border-t-0 lg:border-l border-[var(--border)] bg-[var(--surface)] flex flex-col min-h-0 h-full overflow-hidden"
      }
    >
      <div className="px-3 py-2 border-b border-[var(--border)] flex items-center justify-between gap-2 shrink-0">
        <span className="text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-2)]">
          Decision Analysis
        </span>
        <Button size="sm" onClick={props.onOpenPolicyMatrix} className="px-2 py-1 text-[10.5px]">
          <LayoutGrid className="w-3 h-3" />
          Policy Matrix
        </Button>
      </div>

      <ModelSelectorsHeader
        runId={props.runId}
        checkpoints={props.checkpoints}
        ckptA={props.ckptA}
        ckptB={props.ckptB}
        showCompare={showCompare}
        actionsHidden={Boolean(props.board.actions_hidden)}
        evaluation={props.evaluation}
        winDelta={winDelta}
        forkError={props.forkError}
        warnings={allWarnings}
        onChangeCkptA={props.onChangeCkptA}
        onChangeCkptB={props.onChangeCkptB}
      />

      <WinEstimateSection
        labelA={labelA}
        labelB={labelB}
        showCompare={showCompare}
        showEditShift={hasMutations}
        evaluation={props.evaluation}
        baseEvaluation={props.baseEvaluation}
        forkBusy={props.forkBusy}
      />

      <BranchStrip
        branchEvents={props.branchEvents}
        editCount={props.forkEdits.length}
        canStepTop={canStepTopMove(props)}
        forkBusy={props.forkBusy}
        onStepModelTopMove={props.onStepModelTopMove}
        onPopBranchAction={props.onPopBranchAction}
        onResetFork={props.onResetFork}
      />

      <BeliefDockBody
        board={props.board}
        evaluation={props.evaluation}
        ctrl={props.belief}
        disabled={props.forkBusy || props.branchDone}
        onBranchAction={props.onBranchAction}
        policyFallback={
          <RankedMovesList
            board={props.board}
            evaluation={props.evaluation}
            baseEvaluation={props.baseEvaluation}
            showCompare={showCompare}
            showEditCompare={showEditCompare}
            labelA={labelA}
            labelB={labelB}
            forkBusy={props.forkBusy}
            branchDone={props.branchDone}
            onBranchAction={props.onBranchAction}
            onHoverFocus={props.onHoverActionFocus}
          />
        }
      />

      <CollapsibleMoveLog
        recentEvents={props.board.recent_events ?? []}
        branchEvents={props.branchEvents}
        onSelectStep={props.onSelectStep}
      />
    </aside>
  );
}

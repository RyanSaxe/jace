/** Right 340px single-purpose Decision Analysis pane (zero tabs): cross-run model diff, win bars, ranked moves & Move Log. */

import {
  ChevronDown,
  ChevronUp,
  EyeOff,
  FastForward,
  GitBranch,
  LayoutGrid,
  RotateCcw,
  Undo2,
  X,
} from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/Primitives";
import type {
  BoardState,
  BranchStepEvent,
  EvaluationPair,
  ForkEdit,
  ForkResponse,
  GameOverview,
  GlobalCheckpointOption,
  LegalAction,
  RecentEvent,
  RunDetail,
  TimelinePoint,
} from "@/lib/types";

export interface CheckpointGroup {
  runId: string;
  label: string;
  options: GlobalCheckpointOption[];
}

export function groupCheckpointsByRun(
  checkpoints: GlobalCheckpointOption[],
  currentRunId: string,
): CheckpointGroup[] {
  const byRun = new Map<string, CheckpointGroup>();
  for (const opt of checkpoints) {
    const existing = byRun.get(opt.run_id);
    if (existing) {
      existing.options.push(opt);
    } else {
      const tag = opt.run_id === currentRunId ? `${opt.run_name} (this game)` : opt.run_name;
      byRun.set(opt.run_id, { runId: opt.run_id, label: tag, options: [opt] });
    }
  }
  const groups = [...byRun.values()];
  for (const group of groups) {
    group.options.sort((a, b) => b.iteration - a.iteration);
  }
  return groups.toSorted((a, b) => {
    if (a.runId === currentRunId) return -1;
    if (b.runId === currentRunId) return 1;
    return a.label.localeCompare(b.label);
  });
}

export function formatCheckpointShort(
  ref: string,
  checkpoints: GlobalCheckpointOption[],
  currentRunId: string,
): string {
  if (ref === "heuristic") return "Heuristic";
  if (ref === "none" || ref === "same" || ref === "") return "None";
  const match = checkpoints.find((c) => c.ref === ref || c.checkpoint === ref);
  if (match) {
    const iterLabel = match.iteration >= 0 ? `u${match.iteration}` : match.checkpoint;
    return match.run_id === currentRunId ? iterLabel : `${match.run_name} · ${iterLabel}`;
  }
  const parts = ref.split("::");
  return parts[parts.length - 1] ?? ref;
}

function pickPrimaryDefault(
  sameRun: GlobalCheckpointOption[],
  all: GlobalCheckpointOption[],
  gameIteration: number,
): string {
  const exact = sameRun.find((c) => c.iteration === gameIteration);
  if (exact) return exact.ref;
  const latestRun = sameRun[sameRun.length - 1];
  if (latestRun) return latestRun.ref;
  return all[0]?.ref ?? "heuristic";
}

function pickSecondaryDefault(sameRun: GlobalCheckpointOption[], primaryRef: string): string {
  const idx = sameRun.findIndex((c) => c.ref === primaryRef);
  if (idx > 0) return sameRun[idx - 1]?.ref ?? "none";
  if (sameRun.length > 1) return sameRun[0]?.ref ?? "none";
  return "none";
}

export function resolveDefaultCheckpoints(args: {
  checkpoints: GlobalCheckpointOption[];
  runId: string;
  gameIteration: number;
  initialA?: string;
  initialB?: string;
  ckptA: string | null;
  ckptB: string | null;
}): { effectiveA: string; effectiveB: string } {
  const sameRun = args.checkpoints
    .filter((c) => c.run_id === args.runId)
    .toSorted((a, b) => a.iteration - b.iteration);
  const defaultA =
    args.initialA ?? pickPrimaryDefault(sameRun, args.checkpoints, args.gameIteration);
  const defaultB = args.initialB ?? pickSecondaryDefault(sameRun, defaultA);
  return { effectiveA: args.ckptA ?? defaultA, effectiveB: args.ckptB ?? defaultB };
}

export function buildInitialMatrixModels(args: {
  effectiveA: string;
  effectiveB: string;
  checkpoints: GlobalCheckpointOption[];
  runId: string;
}): string[] {
  const picked: string[] = [];
  const candidates = [
    args.effectiveA,
    args.effectiveB,
    ...args.checkpoints
      .filter((c) => c.run_id === args.runId)
      .toSorted((a, b) => b.iteration - a.iteration)
      .map((c) => c.ref),
  ];
  for (const ref of candidates) {
    if (picked.length >= 4) break;
    if (ref && ref !== "none" && ref !== "same" && !picked.includes(ref)) picked.push(ref);
  }
  return picked.length > 0 ? picked : ["heuristic"];
}

export function topLegalActionIndex(board: BoardState, evaluation: EvaluationPair): number | null {
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

export function describeForkEdit(edit: ForkEdit, heroSeat: number): string {
  const who = edit.seat === heroSeat ? "Hero" : "Opp";
  const card = edit.card ?? "";
  if (edit.op === "set_life") return `${who} life = ${edit.value ?? "?"}`;
  if (edit.op === "remove_hand_card") return `${who} −Hand: ${card}`;
  if (edit.op === "swap_hand_card") return `${who} Swap: ${card} → ${edit.replacement_card ?? ""}`;
  if (edit.op === "add_battlefield") return `${who} +Board: ${card}`;
  if (edit.op === "add_hand_card") return `${who} +Hand: ${card}`;
  return `${who} clear hand`;
}

export function extractBoardKeyCards(board: BoardState): string[] {
  const names = [
    ...board.stack.map((item) => item.card_name),
    ...board.hero.creatures.map((p) => p.name),
    ...board.opp.creatures.map((p) => p.name),
    ...board.hero.other_permanents.map((p) => p.name),
    ...board.hero.hand,
  ];
  return [...new Set(names.filter((name) => name.length > 0))].slice(0, 4);
}

export function defaultPinLabel(board: BoardState): string {
  const stepText = (board.step_name || board.action_space_kind).replace(/([a-z])([A-Z])/g, "$1 $2");
  const topStack = board.stack[0]?.card_name;
  return topStack
    ? `T${board.turn} ${stepText} · ${topStack}`
    : `T${board.turn} ${stepText} · ${board.hero.life} vs ${board.opp.life}`;
}

export function updateLifeForkEdits(args: {
  edits: ForkEdit[];
  board: BoardState;
  seat: 0 | 1;
  delta: number;
  afterBranch: number;
}): ForkEdit[] {
  const isSame = (e: ForkEdit): boolean =>
    e.op === "set_life" && e.seat === args.seat && (e.after_branch ?? 0) === args.afterBranch;
  const existing = args.edits.find(isSame);
  const visible =
    args.seat === (args.board.hero_seat === 1 ? 1 : 0) ? args.board.hero.life : args.board.opp.life;
  const baseLife = typeof existing?.value === "number" ? existing.value : visible;
  const nextLife = Math.max(1, Math.min(99, baseLife + args.delta));
  if (nextLife === baseLife) return args.edits;
  return [
    ...args.edits.filter((e) => !isSame(e)),
    { op: "set_life", seat: args.seat, value: nextLife, after_branch: args.afterBranch },
  ];
}

export interface ReadyPayload {
  board: BoardState;
  evaluation: EvaluationPair;
  base: EvaluationPair;
  branchEvents: BranchStepEvent[];
  branchDone: boolean;
  nearestPriorityStep: number | null;
}

export function resolveActivePayload(
  hasMutations: boolean,
  data: {
    board: BoardState;
    evaluation: EvaluationPair;
    nearest_priority_step?: number | null;
  } | null,
  forkResult: ForkResponse | null,
): ReadyPayload | null {
  if (hasMutations && forkResult !== null) {
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

export function resolveOverviewInfo(
  overview: GameOverview | null,
  stepWarn: string | null | undefined,
): {
  timeline: TimelinePoint[];
  evalSource?: string;
  maxPlayableStep: number;
  turningPointStep?: number;
  replayWarning: string | null;
} {
  if (!overview) {
    return {
      timeline: [],
      evalSource: undefined,
      maxPlayableStep: 0,
      turningPointStep: undefined,
      replayWarning: stepWarn ?? null,
    };
  }
  return {
    timeline: overview.timeline,
    evalSource: overview.eval_source,
    maxPlayableStep: overview.max_playable_step ?? Math.max(0, overview.total_steps - 1),
    turningPointStep: overview.turning_point_step,
    replayWarning: stepWarn ?? overview.replay_warning ?? null,
  };
}

export function mergeCheckpoints(
  globalList: GlobalCheckpointOption[],
  runDetail: RunDetail | null,
  runId: string,
): GlobalCheckpointOption[] {
  if (globalList.length > 0) return globalList;
  const runName = runDetail?.summary.name ?? runId.split("/").pop() ?? runId;
  const experiment = runDetail?.summary.experiment ?? runName;
  return (runDetail?.checkpoints ?? []).map((c) => ({
    ref: `${runId}::${c.name}`,
    run_id: runId,
    run_name: runName,
    experiment,
    checkpoint: c.name,
    iteration: c.iteration,
    size_bytes: c.size_bytes,
  }));
}

export function CheckpointSelect(props: {
  value: string;
  onChange: (value: string) => void;
  checkpoints: GlobalCheckpointOption[];
  currentRunId: string;
  ariaLabel: string;
  includeNone?: boolean;
  noneLabel?: string;
  excludeRefs?: ReadonlySet<string>;
  className?: string;
}): ReactNode {
  const groups = groupCheckpointsByRun(props.checkpoints, props.currentRunId);
  const exclude = props.excludeRefs;
  return (
    <select
      value={props.value}
      onChange={(e): void => props.onChange(e.target.value)}
      aria-label={props.ariaLabel}
      className={
        props.className ??
        "h-7 rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-1.5 text-[11px] font-semibold tabular min-w-0"
      }
    >
      {props.includeNone ? (
        <option value="none">{props.noneLabel ?? "None (single)"}</option>
      ) : null}
      {exclude?.has("heuristic") ? null : <option value="heuristic">Heuristic</option>}
      {groups.map((group) => {
        const visible = exclude
          ? group.options.filter((opt) => !exclude.has(opt.ref))
          : group.options;
        if (visible.length === 0) return null;
        return (
          <optgroup key={group.runId} label={group.label}>
            {visible.map((opt) => (
              <option key={opt.ref} value={opt.ref}>
                {opt.iteration >= 0 ? `u${opt.iteration} · ${opt.checkpoint}` : opt.checkpoint}
              </option>
            ))}
          </optgroup>
        );
      })}
    </select>
  );
}

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
  className?: string;
}

function formatSignedPct(delta: number): string {
  return delta > 0 ? `+${delta}%` : `${delta}%`;
}

function deltaToneClass(delta: number): string {
  if (delta > 0) return "text-[var(--win)] font-semibold";
  if (delta < 0) return "text-[var(--loss)] font-semibold";
  return "text-[var(--text-3)]";
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

interface RankedMoveRow {
  act: LegalAction;
  pPrimary: number;
  pSecondary: number | null;
}

function buildRankedMoves(args: {
  actions: LegalAction[];
  probsA: number[];
  probsB: number[];
  baseProbs: number[];
  useBeforeAfter: boolean;
  useCompare: boolean;
}): RankedMoveRow[] {
  return args.actions
    .map((act, idx) => {
      const pPrimary = Math.round((args.probsA[idx] ?? 0) * 100);
      let pSecondary: number | null = null;
      if (args.useBeforeAfter) pSecondary = Math.round((args.baseProbs[idx] ?? 0) * 100);
      else if (args.useCompare) pSecondary = Math.round((args.probsB[idx] ?? 0) * 100);
      return { act, pPrimary, pSecondary };
    })
    .toSorted((a, b) => b.pPrimary - a.pPrimary);
}

function movesLegendLabel(args: {
  useBeforeAfter: boolean;
  useCompare: boolean;
  labelA: string;
  labelB: string;
}): string {
  if (args.useBeforeAfter) return "After vs Before";
  if (args.useCompare) return `${args.labelA} vs ${args.labelB}`;
  return "Click to play";
}

function RankedMoveCard(props: {
  rankIdx: number;
  act: LegalAction;
  primaryPct: number;
  secondaryPct: number | null;
  primaryColor: string;
  secondaryColor: string;
  isChosen: boolean;
  disabled: boolean;
  onBranchAction: (actionIdx: number) => void;
  onHoverFocus: (ids: number[]) => void;
}): ReactNode {
  const delta = props.secondaryPct === null ? 0 : props.primaryPct - props.secondaryPct;
  const topClass =
    props.rankIdx === 0
      ? "bg-[var(--accent-soft)]/55 border-[var(--accent)]/35 hover:border-[var(--accent)]"
      : "bg-[var(--bg)]/55 border-[var(--border)]/70 hover:bg-[var(--surface-2)] hover:border-[var(--accent)]/50";
  return (
    <button
      type="button"
      disabled={props.disabled}
      title={`Click to play: ${props.act.label}`}
      aria-label={`Play action ${props.rankIdx + 1}: ${props.act.label}`}
      onMouseEnter={(): void => props.onHoverFocus(props.act.focus_ids)}
      onMouseLeave={(): void => props.onHoverFocus([])}
      onClick={(): void => props.onBranchAction(props.act.index)}
      className={`w-full text-left rounded-md px-2.5 py-2 border transition-colors cursor-pointer disabled:opacity-50 ${topClass}`}
    >
      <div className="flex items-start justify-between gap-2 text-[11px] mb-1.5">
        <span className="font-semibold leading-snug break-words line-clamp-2 min-w-0">
          {props.rankIdx + 1}. {props.act.label}
        </span>
        <span className="tabular shrink-0 flex items-center gap-1 pt-0.5">
          {props.isChosen ? (
            <span title="Action taken in recorded game" className="text-[var(--accent)]">
              ★
            </span>
          ) : null}
          <strong>{props.primaryPct}%</strong>
          {props.secondaryPct !== null && delta !== 0 ? (
            <span className={`text-[10px] ${deltaToneClass(delta)}`}>
              ({formatSignedPct(delta)})
            </span>
          ) : null}
        </span>
      </div>
      <div className="flex flex-col gap-1">
        <div className="h-[5px] rounded-full bg-[var(--surface-2)] overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-150"
            style={{ width: `${Math.max(2, props.primaryPct)}%`, background: props.primaryColor }}
          />
        </div>
        {props.secondaryPct === null ? null : (
          <div className="h-[4px] rounded-full bg-[var(--surface-2)] overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-150 opacity-85"
              style={{
                width: `${Math.max(2, props.secondaryPct)}%`,
                background: props.secondaryColor,
              }}
            />
          </div>
        )}
      </div>
    </button>
  );
}

function RankedMovesList(props: {
  board: BoardState;
  evaluation: EvaluationPair;
  baseEvaluation: EvaluationPair;
  showCompare: boolean;
  showEditCompare: boolean;
  labelA: string;
  labelB: string;
  forkBusy: boolean;
  branchDone: boolean;
  onBranchAction: (actionIdx: number) => void;
  onHoverFocus: (ids: number[]) => void;
}): ReactNode {
  const [showAll, setShowAll] = useState(false);

  if (props.board.actions_hidden) {
    return (
      <div className="p-3 flex-1">
        <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] mb-2">
          <span>Opponent Deciding</span>
          <span>Hero POV</span>
        </div>
        <div className="rounded-md border border-dashed border-[var(--border-strong)] bg-[var(--surface-2)] p-2.5 text-[11px] text-[var(--text-2)] flex items-start gap-2">
          <EyeOff className="w-4 h-4 shrink-0 mt-0.5 text-[var(--text-3)]" />
          <span>
            Opponent actions are hidden in <strong>Hero POV</strong>. Switch to{" "}
            <strong>Omniscient</strong> to inspect or branch Opponent&apos;s play.
          </span>
        </div>
      </div>
    );
  }

  const actions = props.board.legal_actions;
  const baseProbs = props.baseEvaluation.primary.probs;
  const useBeforeAfter = props.showEditCompare && baseProbs.length === actions.length;
  const useCompare = !useBeforeAfter && props.showCompare;
  const ranked = buildRankedMoves({
    actions,
    probsA: props.evaluation.primary.probs,
    probsB: props.evaluation.compare.probs,
    baseProbs,
    useBeforeAfter,
    useCompare,
  });
  const maxShow = 8;
  const visible = showAll ? ranked : ranked.slice(0, maxShow);
  const whoLabel = props.board.actor_is_hero === false ? "OPP" : "HERO";
  const legendRight = movesLegendLabel({
    useBeforeAfter,
    useCompare,
    labelA: props.labelA,
    labelB: props.labelB,
  });

  return (
    <div className="p-3 flex-1 min-h-0 flex flex-col gap-2 overflow-y-auto">
      <div className="flex items-center justify-between gap-2 text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] shrink-0">
        <span>
          {whoLabel} · {props.board.action_space_kind} ({actions.length})
        </span>
        <span className="truncate normal-case font-medium">{legendRight}</span>
      </div>

      {props.branchDone ? (
        <div className="rounded-md border border-[var(--win)]/40 bg-[var(--win-soft)] px-2.5 py-2 text-[11px] text-[var(--win)] font-semibold">
          Game ended along this branched line. Click Reset or Undo to continue exploring.
        </div>
      ) : null}

      <div className="flex flex-col gap-1.5">
        {visible.map((row, idx) => (
          <RankedMoveCard
            key={row.act.index}
            rankIdx={idx}
            act={row.act}
            primaryPct={row.pPrimary}
            secondaryPct={row.pSecondary}
            primaryColor={useBeforeAfter ? "var(--fork-fill)" : "var(--accent)"}
            secondaryColor="#d97706"
            isChosen={props.board.chosen_action === row.act.index}
            disabled={props.forkBusy || props.branchDone}
            onBranchAction={props.onBranchAction}
            onHoverFocus={props.onHoverFocus}
          />
        ))}
      </div>

      {actions.length > maxShow ? (
        <button
          type="button"
          onClick={(): void => setShowAll((prev) => !prev)}
          className="mt-1 w-full rounded-md border border-dashed border-[var(--border-strong)] py-1 text-[10.5px] font-medium text-[var(--text-2)] hover:text-[var(--text)] hover:border-[var(--accent)] cursor-pointer shrink-0"
        >
          {showAll ? `Show top ${maxShow} ▴` : `Show all ${actions.length} moves ▾`}
        </button>
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

      <CollapsibleMoveLog
        recentEvents={props.board.recent_events ?? []}
        branchEvents={props.branchEvents}
        onSelectStep={props.onSelectStep}
      />
    </aside>
  );
}

export function MobilePolicySheet(props: {
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

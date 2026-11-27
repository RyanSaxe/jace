/** Pure helper functions for Tabletop checkpoint grouping, fork editing, active payload resolution, and formatting. */

import type {
  BoardState,
  BranchStepEvent,
  EvaluationPair,
  ForkEdit,
  ForkResponse,
  GameOverview,
  GlobalCheckpointOption,
  RunDetail,
  TimelinePoint,
} from "@/lib/types";

export interface CheckpointGroup {
  runId: string;
  label: string;
  options: GlobalCheckpointOption[];
}

export function keyByOccurrence(
  names: string[],
): Array<{ key: string; name: string; order: number }> {
  const seen = new Map<string, number>();
  return names.map((name, order) => {
    const count = (seen.get(name) ?? 0) + 1;
    seen.set(name, count);
    return { key: `${name}#${count}`, name, order };
  });
}

export function formatSignedPct(delta: number): string {
  return delta > 0 ? `+${delta}%` : `${delta}%`;
}

export function deltaToneClass(delta: number): string {
  if (delta > 0) return "text-[var(--win)] font-semibold";
  if (delta < 0) return "text-[var(--loss)] font-semibold";
  return "text-[var(--text-3)]";
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

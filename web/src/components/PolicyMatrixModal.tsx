/** Floating Policy Evolution Matrix modal with arbitrary N checkpoints and progressive MPS playouts. */

import { Bookmark, LayoutGrid, Play, Plus, RefreshCw, Square, X } from "lucide-react";
import { type ReactNode, useRef, useState } from "react";
import {
  CheckpointSelect,
  defaultPinLabel,
  formatCheckpointShort,
} from "@/components/TabletopDock";
import type { HoverCardInfo } from "@/components/TabletopParts";
import { Button } from "@/components/ui/Primitives";
import { useAbortOnUnmount, useCardMeta, useResetOnKey } from "@/hooks/useStudioData";
import { cardImageUrl, postPlayoutBatch } from "@/lib/api";
import type {
  ActionPlayoutResult,
  BoardState,
  ForkEdit,
  GameTarget,
  GlobalCheckpointOption,
  LegalAction,
  PolicyMatrixColumn,
  PolicyMatrixResponse,
  PovMode,
} from "@/lib/types";

const PLAYOUT_CHUNK_SIZE = 8;
const DEAL_PRESETS: ReadonlyArray<number> = [16, 32, 64];

interface CumulativePlayoutRow {
  action_index: number;
  label: string;
  deals_done: number;
  wins: number;
  draws: number;
  losses: number;
  win_rate: number;
  ci_low: number;
  ci_high: number;
}

interface PlayoutState {
  running: boolean;
  completedDeals: number;
  targetDeals: number;
  device: string;
  elapsedMs: number;
  error: string | null;
  byAction: Record<number, CumulativePlayoutRow>;
}

const EMPTY_PLAYOUT: PlayoutState = {
  running: false,
  completedDeals: 0,
  targetDeals: 32,
  device: "",
  elapsedMs: 0,
  error: null,
  byAction: {},
};

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

function wilson95(successes: number, total: number): { rate: number; low: number; high: number } {
  if (total <= 0) return { rate: 0, low: 0, high: 0 };
  const z = 1.959963984540054;
  const p = successes / total;
  const z2 = z * z;
  const denom = 1 + z2 / total;
  const center = (p + z2 / (2 * total)) / denom;
  const half = (z * Math.sqrt((p * (1 - p) + z2 / (4 * total)) / total)) / denom;
  return { rate: p, low: Math.max(0, center - half), high: Math.min(1, center + half) };
}

function mergePlayoutChunk(
  prev: Record<number, CumulativePlayoutRow>,
  batch: ActionPlayoutResult[],
): Record<number, CumulativePlayoutRow> {
  const next: Record<number, CumulativePlayoutRow> = { ...prev };
  for (const row of batch) {
    const existing = next[row.action_index];
    const wins = (existing?.wins ?? 0) + row.wins;
    const draws = (existing?.draws ?? 0) + row.draws;
    const losses = (existing?.losses ?? 0) + row.losses;
    const deals = (existing?.deals_done ?? 0) + row.deals_done;
    const ci = wilson95(wins + 0.5 * draws, deals);
    next[row.action_index] = {
      action_index: row.action_index,
      label: row.label,
      deals_done: deals,
      wins,
      draws,
      losses,
      win_rate: ci.rate,
      ci_low: ci.low,
      ci_high: ci.high,
    };
  }
  return next;
}

function clampDeals(raw: string): number {
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed >= 1 ? parsed : 32;
}

function sortMatrixActions(
  actions: LegalAction[],
  columns: PolicyMatrixColumn[],
): Array<{ action: LegalAction; originalIdx: number }> {
  const primaryCol = columns[0];
  return actions
    .map((action, originalIdx) => ({ action, originalIdx }))
    .toSorted(
      (a, b) => (primaryCol?.probs[b.originalIdx] ?? 0) - (primaryCol?.probs[a.originalIdx] ?? 0),
    );
}

async function streamPlayoutChunks(args: {
  baseReq: Omit<Parameters<typeof postPlayoutBatch>[0], "deals" | "batchOffset">;
  targetDeals: number;
  signal: AbortSignal;
  onUpdate: (updater: (prev: PlayoutState) => PlayoutState) => void;
  onFinish: () => void;
}): Promise<void> {
  const runChunk = async (
    completed: number,
    elapsedMs: number,
    accum: Record<number, CumulativePlayoutRow>,
  ): Promise<void> => {
    if (args.signal.aborted || completed >= args.targetDeals) return;
    const chunk = Math.min(PLAYOUT_CHUNK_SIZE, args.targetDeals - completed);
    const res = await postPlayoutBatch(
      { ...args.baseReq, deals: chunk, batchOffset: completed },
      args.signal,
    );
    if (args.signal.aborted) return;
    const nextCompleted = completed + res.deals_done;
    const nextElapsed = elapsedMs + res.elapsed_ms;
    const nextAccum = mergePlayoutChunk(accum, res.results);
    const stillRunning = nextCompleted < args.targetDeals && res.deals_done > 0;
    args.onUpdate(() => ({
      running: stillRunning,
      completedDeals: nextCompleted,
      targetDeals: args.targetDeals,
      device: res.device,
      elapsedMs: nextElapsed,
      error: null,
      byAction: nextAccum,
    }));
    if (stillRunning) await runChunk(nextCompleted, nextElapsed, nextAccum);
  };

  try {
    await runChunk(0, 0, {});
  } catch (err: unknown) {
    if (!(err instanceof DOMException && err.name === "AbortError")) {
      const message = err instanceof Error ? err.message : "Playout failed";
      args.onUpdate((prev) => ({ ...prev, running: false, error: message }));
    }
  }
  args.onUpdate((prev) => (prev.running ? { ...prev, running: false } : prev));
  args.onFinish();
}

function MatrixCell(props: { prob: number; isTop: boolean }): ReactNode {
  const pct = Math.round(props.prob * 100);
  return (
    <td
      aria-label={`${pct}% probability${props.isTop ? ", top choice" : ""}`}
      className={`px-2.5 py-2 align-middle border-b border-[var(--border)] ${
        props.isTop ? "bg-[var(--accent-soft)]/70" : ""
      }`}
    >
      <div className="flex flex-col gap-1 min-w-[92px]">
        <div className="flex items-center justify-between gap-1 tabular text-[11px]">
          <span className={props.isTop ? "font-bold text-[var(--accent)]" : "font-medium"}>
            {pct}%
          </span>
          {props.isTop ? (
            <span className="rounded bg-[var(--accent)] px-1 text-[8.5px] font-bold text-white uppercase">
              Top
            </span>
          ) : null}
        </div>
        <div className="h-[5px] rounded-full bg-[var(--surface-2)] overflow-hidden">
          <div
            className="h-full rounded-full"
            style={{
              width: `${Math.max(2, pct)}%`,
              background: props.isTop ? "var(--accent)" : "var(--text-3)",
            }}
          />
        </div>
      </div>
    </td>
  );
}

function PlayoutCell(props: { row: CumulativePlayoutRow | undefined }): ReactNode {
  const row = props.row;
  if (!row || row.deals_done === 0) {
    return (
      <td className="px-2.5 py-2 border-b border-[var(--border)] text-[10.5px] text-[var(--text-3)] tabular">
        —
      </td>
    );
  }
  const ratePct = (row.win_rate * 100).toFixed(1);
  const lowPct = Math.round(row.ci_low * 100);
  const highPct = Math.round(row.ci_high * 100);
  return (
    <td
      aria-label={`${ratePct}% win rate, 95% CI ${lowPct}% to ${highPct}%`}
      className="px-2.5 py-2 border-b border-[var(--border)] bg-[var(--win-soft)]/25"
    >
      <div className="flex flex-col gap-1 min-w-[144px]">
        <div className="flex items-baseline justify-between gap-1.5 tabular text-[11px]">
          <strong className="text-[var(--win)]">{ratePct}%</strong>
          <span className="text-[9.5px] text-[var(--text-3)]">
            [{lowPct}–{highPct}%]
          </span>
        </div>
        <div className="relative h-[6px] rounded-full bg-[var(--surface-2)] overflow-hidden">
          <div
            className="absolute top-0 bottom-0 rounded-full bg-[var(--win-fill)]/35"
            style={{ left: `${lowPct}%`, width: `${Math.max(2, highPct - lowPct)}%` }}
          />
          <div
            className="absolute top-0 bottom-0 w-[2px] bg-[var(--win-fill)]"
            style={{ left: `${Math.round(row.win_rate * 100)}%` }}
          />
        </div>
        <div className="tabular text-[9.5px] text-[var(--text-3)]">
          {row.wins}W-{row.losses}L{row.draws > 0 ? `-${row.draws}D` : ""} (n={row.deals_done})
        </div>
      </div>
    </td>
  );
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

function PlayoutProgressStrip(props: { playout: PlayoutState }): ReactNode {
  const { playout } = props;
  if (!playout.running && playout.completedDeals === 0) return null;
  const pct =
    playout.targetDeals > 0
      ? Math.min(100, Math.round((playout.completedDeals / playout.targetDeals) * 100))
      : 0;
  const deviceTag = playout.device ? playout.device.toUpperCase() : "";
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-[10.5px] tabular text-[var(--text-2)]">
        <span>
          {playout.completedDeals} / {playout.targetDeals} deals
          {deviceTag ? ` · ${deviceTag}` : ""}
          {playout.elapsedMs > 0 ? ` · ${(playout.elapsedMs / 1000).toFixed(1)}s` : ""}
        </span>
        <span className="font-semibold">{playout.running ? `${pct}%` : "Complete"}</span>
      </div>
      <div className="h-[6px] rounded-full bg-[var(--surface-2)] overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${
            playout.running ? "bg-[var(--accent)] animate-pulse" : "bg-[var(--win-fill)]"
          }`}
          style={{ width: `${Math.max(playout.running ? 4 : 0, pct)}%` }}
        />
      </div>
    </div>
  );
}

function PlayoutControlBar(props: {
  checkpoints: GlobalCheckpointOption[];
  currentRunId: string;
  playoutModel: string;
  dealsInput: string;
  playout: PlayoutState;
  canRunPlayout: boolean;
  onChangeModel: (ref: string) => void;
  onChangeDeals: (val: string) => void;
  onStart: () => void;
  onStop: () => void;
}): ReactNode {
  const { playout } = props;
  const parsedDeals = clampDeals(props.dealsInput);
  return (
    <div className="border-t border-[var(--border)] bg-[var(--bg)] px-4 py-2.5 flex flex-col gap-2 shrink-0">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-3)]">
            Playout Model
          </span>
          <CheckpointSelect
            value={props.playoutModel}
            onChange={props.onChangeModel}
            checkpoints={props.checkpoints}
            currentRunId={props.currentRunId}
            ariaLabel="Model to play out"
            className="h-7 rounded-md border border-[var(--border)] bg-[var(--surface)] px-2 text-[11px] font-semibold tabular max-w-[190px]"
          />
          <label className="inline-flex items-center gap-1.5 text-[11px] font-medium text-[var(--text-2)]">
            <span>Deals</span>
            <input
              type="number"
              min={1}
              value={props.dealsInput}
              disabled={playout.running}
              onChange={(e): void => props.onChangeDeals(e.target.value)}
              aria-label="Number of playout deals"
              className="w-16 h-7 rounded-md border border-[var(--border)] bg-[var(--surface)] px-2 text-[11px] font-semibold tabular"
            />
          </label>
          <div className="inline-flex items-center gap-1">
            {DEAL_PRESETS.map((preset) => (
              <button
                key={preset}
                type="button"
                disabled={playout.running}
                onClick={(): void => props.onChangeDeals(String(preset))}
                className={`tabular rounded px-1.5 py-0.5 text-[10.5px] font-semibold border cursor-pointer disabled:opacity-45 ${
                  parsedDeals === preset
                    ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                    : "border-[var(--border)] bg-[var(--surface)] text-[var(--text-2)] hover:text-[var(--text)]"
                }`}
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {playout.running ? (
            <Button
              size="md"
              onClick={props.onStop}
              className="border-[var(--loss)] bg-[var(--loss-soft)] text-[var(--loss)] hover:opacity-90"
            >
              <Square className="w-3 h-3 fill-current" />
              Stop
            </Button>
          ) : (
            <Button
              size="md"
              variant="primary"
              disabled={!props.canRunPlayout}
              onClick={props.onStart}
            >
              <Play className="w-3.5 h-3.5" />
              Play out all actions
            </Button>
          )}
        </div>
      </div>

      <PlayoutProgressStrip playout={playout} />

      {playout.error ? (
        <div className="rounded bg-[var(--loss-soft)] text-[var(--loss)] px-2.5 py-1 text-[10.5px]">
          {playout.error}
        </div>
      ) : null}
    </div>
  );
}

function PolicyMatrixTable(props: {
  actions: LegalAction[];
  columns: PolicyMatrixColumn[];
  checkpoints: GlobalCheckpointOption[];
  currentRunId: string;
  playout: PlayoutState;
  hasPlayoutColumn: boolean;
  onRemoveModel: (ref: string) => void;
}): ReactNode {
  const sorted = sortMatrixActions(props.actions, props.columns);
  return (
    <div className="flex-1 min-h-0 overflow-auto">
      <table className="w-full border-collapse text-left">
        <thead className="sticky top-0 z-10 bg-[var(--surface)] shadow-[0_1px_0_var(--border)]">
          <tr>
            <th className="px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] min-w-[200px]">
              Legal Action ({sorted.length})
            </th>
            {props.columns.map((col) => {
              const title = formatCheckpointShort(col.ref, props.checkpoints, props.currentRunId);
              return (
                <th
                  key={col.ref}
                  className="px-2.5 py-2 text-[10.5px] font-semibold text-[var(--text)] whitespace-nowrap"
                >
                  <div className="flex items-center justify-between gap-1.5">
                    <span className="tabular truncate max-w-[140px]" title={col.label || col.ref}>
                      {title}
                    </span>
                    <button
                      type="button"
                      onClick={(): void => props.onRemoveModel(col.ref)}
                      aria-label={`Remove ${title} column`}
                      className="rounded p-0.5 text-[var(--text-3)] hover:text-[var(--loss)] hover:bg-[var(--loss-soft)] cursor-pointer"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                </th>
              );
            })}
            {props.hasPlayoutColumn ? (
              <th className="px-2.5 py-2 text-[10px] font-bold uppercase tracking-wider text-[var(--win)] whitespace-nowrap">
                Empirical Win% [95% CI]
              </th>
            ) : null}
          </tr>
        </thead>
        <tbody>
          {sorted.map(({ action, originalIdx }) => (
            <tr key={action.index} className="hover:bg-[var(--surface-2)]/40 transition-colors">
              <td className="px-3 py-2 border-b border-[var(--border)] text-[11.5px] font-medium leading-snug">
                {action.label}
              </td>
              {props.columns.map((col) => (
                <MatrixCell
                  key={`${col.ref}-${action.index}`}
                  prob={col.probs[originalIdx] ?? 0}
                  isTop={col.top_action_index === action.index}
                />
              ))}
              {props.hasPlayoutColumn ? (
                <PlayoutCell row={props.playout.byAction[action.index]} />
              ) : null}
            </tr>
          ))}
        </tbody>
        {props.columns.length > 0 ? (
          <tfoot className="sticky bottom-0 bg-[var(--surface-2)] border-t border-[var(--border-strong)]">
            <tr>
              <td className="px-3 py-2 text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-2)]">
                Hero Win% Estimate
              </td>
              {props.columns.map((col) => {
                const winPct = Math.round(col.wdl.win * 100);
                const drawPct = Math.round(col.wdl.draw * 100);
                const lossPct = Math.max(0, 100 - winPct - drawPct);
                return (
                  <td
                    key={`wdl-${col.ref}`}
                    title={`Win ${winPct}% · Draw ${drawPct}% · Loss ${lossPct}%`}
                    className="px-2.5 py-2 tabular text-[11.5px] font-bold text-[var(--text)]"
                  >
                    {winPct}%
                  </td>
                );
              })}
              {props.hasPlayoutColumn ? (
                <td className="px-2.5 py-2 tabular text-[10.5px] text-[var(--text-2)]">
                  {props.playout.completedDeals} deals
                </td>
              ) : null}
            </tr>
          </tfoot>
        ) : null}
      </table>
    </div>
  );
}

function MatrixBodySection(props: {
  actionsHidden: boolean;
  selectedModels: string[];
  matrixLoading: boolean;
  actions: LegalAction[];
  columns: PolicyMatrixColumn[];
  checkpoints: GlobalCheckpointOption[];
  currentRunId: string;
  playout: PlayoutState;
  hasPlayoutColumn: boolean;
  onRemoveModel: (ref: string) => void;
}): ReactNode {
  if (props.actionsHidden) {
    return (
      <div className="p-6 text-center text-xs text-[var(--text-2)]">
        Opponent actions are hidden in <strong>Hero POV</strong>. Switch to{" "}
        <strong>Omniscient</strong> to inspect the policy matrix for this step.
      </div>
    );
  }
  if (props.selectedModels.length === 0) {
    return (
      <div className="p-6 text-center text-xs text-[var(--text-2)]">
        No models selected. Click <strong>+ Add model</strong> to compare checkpoints side by side.
      </div>
    );
  }
  if (props.matrixLoading && props.columns.length === 0) {
    return (
      <div className="p-8 text-center text-xs text-[var(--text-3)]">
        Evaluating {props.selectedModels.length} checkpoint
        {props.selectedModels.length === 1 ? "" : "s"}…
      </div>
    );
  }
  return (
    <PolicyMatrixTable
      actions={props.actions}
      columns={props.columns}
      checkpoints={props.checkpoints}
      currentRunId={props.currentRunId}
      playout={props.playout}
      hasPlayoutColumn={props.hasPlayoutColumn}
      onRemoveModel={props.onRemoveModel}
    />
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

const QUICK_PIN_TAGS: ReadonlyArray<string> = ["combat", "blunder", "tactical", "fork"];

export function BoardDetailsPopover(props: { board: BoardState; onClose: () => void }): ReactNode {
  const b = props.board;
  const rows: ReadonlyArray<[string, string]> = [
    ["Phase / Step", `${b.phase} · ${b.step_name}`],
    ["Decision", b.action_space_kind],
    ["Hero Open / Perms", `${b.hero.open_lands ?? 0} / ${b.hero.zone_counts.battlefield}`],
    ["Opp Open / Perms", `${b.opp.open_lands ?? 0} / ${b.opp.zone_counts.battlefield}`],
    ["Hero Seat", `Seat ${b.hero_seat ?? 0}`],
  ];
  return (
    <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-40 w-[250px] rounded-lg border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-lg)] p-3 text-[11px] flex flex-col gap-2">
      <div className="flex items-center justify-between font-semibold text-xs border-b border-[var(--border)] pb-1.5">
        <span>Board Details</span>
        <button
          type="button"
          onClick={props.onClose}
          aria-label="Close details"
          className="p-0.5 text-[var(--text-3)] cursor-pointer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="grid grid-cols-2 gap-1.5 tabular">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <div className="text-[var(--text-3)]">{k}</div>
            <div className="text-right font-medium">{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function PinPositionPopover(props: {
  board: BoardState;
  saving: boolean;
  onSave: (p: { label: string; note: string; tag: string }) => void;
  onClose: () => void;
}): ReactNode {
  const [label, setLabel] = useState(() => defaultPinLabel(props.board));
  const [note, setNote] = useState("");
  const [tag, setTag] = useState("");
  return (
    <dialog
      open
      aria-label="Pin position"
      className="absolute right-2 left-auto top-[44px] z-50 m-0 w-[280px] max-w-[calc(100vw-16px)] rounded-[10px] border border-[var(--border-strong)] bg-[var(--surface)] text-[var(--text)] shadow-[var(--shadow-lg)] p-3 flex flex-col gap-2"
    >
      <div className="flex items-center justify-between text-xs font-bold">
        <span>Pin to Watchlist</span>
        <button
          type="button"
          onClick={props.onClose}
          aria-label="Close pin"
          className="p-0.5 text-[var(--text-3)] cursor-pointer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
      <input
        type="text"
        value={label}
        onChange={(e): void => setLabel(e.target.value)}
        aria-label="Label"
        placeholder="Label…"
        className="h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-[11px]"
      />
      <input
        type="text"
        value={note}
        onChange={(e): void => setNote(e.target.value)}
        aria-label="Note"
        placeholder="Optional note…"
        className="h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-[11px]"
      />
      <div className="flex flex-wrap gap-1">
        {QUICK_PIN_TAGS.map((q) => (
          <button
            key={q}
            type="button"
            onClick={(): void => setTag((p) => (p === q ? "" : q))}
            className={`rounded px-1.5 py-0.5 text-[10px] font-semibold border cursor-pointer ${tag === q ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]" : "border-[var(--border)] bg-[var(--bg)] text-[var(--text-2)]"}`}
          >
            {q}
          </button>
        ))}
      </div>
      <div className="flex justify-end gap-1.5">
        <Button size="sm" variant="ghost" onClick={props.onClose}>
          Cancel
        </Button>
        <Button
          size="sm"
          variant="primary"
          disabled={props.saving}
          onClick={(): void => props.onSave({ label, note, tag })}
        >
          <Bookmark className="w-3 h-3" />
          {props.saving ? "Saving…" : "Save Pin"}
        </Button>
      </div>
    </dialog>
  );
}

function combatTags(info: HoverCardInfo): string[] {
  const tags: string[] = [];
  if (info.damage && info.damage > 0) tags.push(`${info.damage} dmg`);
  if (info.counters) tags.push(`+${info.counters}/+${info.counters}`);
  if (info.attacking) tags.push("Attacking");
  if (info.blocking) tags.push(info.blockingName ? `Blocking ${info.blockingName}` : "Blocking");
  if (info.tapped) tags.push("Tapped");
  if (info.token) tags.push("Token");
  return tags;
}

function zoomTagList(info: HoverCardInfo, manaCost?: string, setCode?: string): string[] {
  const tags: string[] = [];
  if (manaCost) tags.push(manaCost);
  if (info.pt) tags.push(info.pt);
  if (info.copyIndex !== null && info.copyIndex !== undefined) tags.push(`Copy #${info.copyIndex}`);
  tags.push(...combatTags(info));
  if (setCode) tags.push(setCode);
  return tags;
}

function resolveCardCopy(meta: ReturnType<typeof useCardMeta>): {
  typeLine: string;
  bodyText: string;
  manaCost?: string;
  setCode?: string;
} {
  if (!meta) return { typeLine: "", bodyText: "Loading oracle text…" };
  return {
    typeLine: meta.type_line,
    bodyText: meta.oracle_text || meta.type_line || "No oracle text.",
    manaCost: meta.mana_cost,
    setCode: meta.set_code || undefined,
  };
}

function computeZoomStyle(info: HoverCardInfo): { left: number; top: number } {
  if (typeof window === "undefined") return { left: 16, top: 64 };
  const pad = 14;
  const w = Math.min(368, window.innerWidth - pad * 2);
  if (info.anchorX === undefined || info.anchorY === undefined) return { left: pad, top: 60 };
  const rawLeft =
    info.anchorX < window.innerWidth * 0.48 ? info.anchorX + 26 : info.anchorX - w - 26;
  return {
    left: Math.max(pad, Math.min(window.innerWidth - w - pad, rawLeft)),
    top: Math.max(pad, Math.min(window.innerHeight - 274, info.anchorY - 96)),
  };
}

export function CardZoomPreview(props: {
  info: HoverCardInfo | null;
  pinned?: boolean;
  onDismiss?: () => void;
}): ReactNode {
  const meta = useCardMeta(props.info?.name ?? null);
  if (!props.info) return null;
  const copy = resolveCardCopy(meta);
  const tags = zoomTagList(props.info, copy.manaCost, copy.setCode);
  const pos = computeZoomStyle(props.info);
  const showClose = Boolean(props.pinned && props.onDismiss);
  return (
    <div
      style={{ left: pos.left, top: pos.top }}
      className={`${props.pinned ? "pointer-events-auto" : "pointer-events-none"} fixed z-50 w-[368px] max-w-[calc(100vw-28px)] rounded-[10px] border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-lg)] p-2.5 flex gap-2.5 items-start`}
    >
      <img
        src={cardImageUrl(props.info.name)}
        alt={props.info.name}
        className="w-[122px] h-[170px] rounded-[6px] object-cover border border-[var(--border)] shrink-0"
      />
      <div className="min-w-0 flex-1 flex flex-col gap-1 max-h-[340px] overflow-y-auto pr-0.5">
        <div className="flex items-start justify-between gap-1">
          <div className="text-xs font-bold leading-tight">{props.info.name}</div>
          {showClose ? (
            <button
              type="button"
              onClick={props.onDismiss}
              aria-label="Close card preview"
              className="rounded p-0.5 text-[var(--text-2)] hover:bg-[var(--surface-2)] cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          ) : null}
        </div>
        {copy.typeLine ? (
          <div className="text-[10px] text-[var(--text-3)]">{copy.typeLine}</div>
        ) : null}
        <div className="flex flex-wrap gap-1">
          {tags.map((tag) => (
            <span
              key={tag}
              className="tabular rounded bg-[var(--surface-2)] px-1.5 py-0.5 text-[9.5px] font-semibold text-[var(--text-2)]"
            >
              {tag}
            </span>
          ))}
        </div>
        <p className="text-[10.5px] leading-snug text-[var(--text-2)] whitespace-pre-line m-0">
          {copy.bodyText}
        </p>
      </div>
    </div>
  );
}

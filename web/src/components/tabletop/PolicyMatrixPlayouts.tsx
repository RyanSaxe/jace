/** Progressive playout state, Wilson CI math, chunk streaming, and playout table/control components for PolicyMatrixModal. */

import { Play, Square } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/Primitives";
import { postPlayoutBatch } from "@/lib/api";
import type { ActionPlayoutResult, GlobalCheckpointOption } from "@/lib/types";
import { CheckpointSelect } from "./CheckpointSelect";

const PLAYOUT_CHUNK_SIZE = 8;
const DEAL_PRESETS: ReadonlyArray<number> = [16, 32, 64];

export interface CumulativePlayoutRow {
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

export interface PlayoutState {
  running: boolean;
  completedDeals: number;
  targetDeals: number;
  device: string;
  elapsedMs: number;
  error: string | null;
  byAction: Record<number, CumulativePlayoutRow>;
}

export const EMPTY_PLAYOUT: PlayoutState = {
  running: false,
  completedDeals: 0,
  targetDeals: 32,
  device: "",
  elapsedMs: 0,
  error: null,
  byAction: {},
};

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

export function clampDeals(raw: string): number {
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed >= 1 ? parsed : 32;
}

export async function streamPlayoutChunks(args: {
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

export function PlayoutCell(props: { row: CumulativePlayoutRow | undefined }): ReactNode {
  const row = props.row;
  if (!row || row.deals_done === 0) {
    return (
      <td className="px-2.5 py-2 border-b border-[var(--border)] text-[10.5px] text-[var(--text-3)] tabular">
        -
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

export function PlayoutControlBar(props: {
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

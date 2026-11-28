/** Tabletop Policy/Search switcher, world budget selector, Search action cards with CI bars, and dock body wrapper. */

import { EyeOff, Play, Sliders } from "lucide-react";
import type { ReactNode } from "react";
import { Button, SegmentedControl, type SegmentedOption } from "@/components/ui/Primitives";
import type {
  BoardState,
  DockAnalysisMode,
  EvaluationPair,
  SearchActionRow,
  SearchWorldBudget,
} from "@/lib/types";
import { TurnZeroV0Strip } from "./TurnZeroV0Strip";
import {
  type TabletopBeliefDockProps,
  formatSignedDeltaPct,
  signedToneClass,
} from "./useBeliefSearch";

const MODE_OPTIONS: ReadonlyArray<SegmentedOption<DockAnalysisMode>> = [
  { value: "policy", label: "Policy" },
  { value: "search", label: "Search" },
];
const WORLD_BUDGETS: ReadonlyArray<SearchWorldBudget> = [8, 16, 32, 64];

export function BeliefSearchBar(props: { ctrl: TabletopBeliefDockProps }): ReactNode {
  const { ctrl } = props;
  const editCls =
    ctrl.editorOpen || ctrl.overrideCount > 0
      ? "border-[var(--fork)] bg-[var(--fork-soft)] text-[var(--fork)]"
      : "border-[var(--border)] bg-[var(--surface)] text-[var(--text-2)] hover:border-[var(--border-strong)] hover:text-[var(--text)]";

  return (
    <div className="px-3 py-2 border-b border-[var(--border)] flex flex-col gap-1.5 shrink-0">
      <div className="flex items-center justify-between gap-2">
        <SegmentedControl options={MODE_OPTIONS} value={ctrl.mode} onChange={ctrl.onChangeMode} />
        <button
          type="button"
          onClick={ctrl.onToggleEditor}
          aria-expanded={ctrl.editorOpen}
          className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[10.5px] font-medium cursor-pointer transition-colors ${editCls}`}
        >
          <Sliders className="w-3 h-3" />
          <span>Edit Belief</span>
          {ctrl.overrideCount > 0 ? <span className="tabular">({ctrl.overrideCount})</span> : null}
        </button>
      </div>

      {ctrl.mode === "search" ? (
        <div className="flex items-center justify-between gap-1.5">
          <fieldset
            aria-label="Search world budget"
            className="inline-flex items-center rounded-md border border-[var(--border)] bg-[var(--surface-2)] p-0.5 m-0"
          >
            {WORLD_BUDGETS.map((budget) => (
              <button
                key={budget}
                type="button"
                onClick={(): void => ctrl.onChangeWorlds(budget)}
                aria-pressed={ctrl.worlds === budget}
                className={`px-1.5 py-0.5 rounded text-[10.5px] tabular cursor-pointer transition-colors ${
                  ctrl.worlds === budget
                    ? "bg-[var(--accent)] text-white font-semibold"
                    : "text-[var(--text-2)] hover:text-[var(--text)]"
                }`}
              >
                {budget}w
              </button>
            ))}
          </fieldset>
          <Button
            size="sm"
            variant="primary"
            disabled={ctrl.searchBusy}
            onClick={ctrl.onRunSearch}
            className="px-2 py-0.5 text-[10.5px]"
          >
            <Play className="w-2.5 h-2.5" />
            {ctrl.searchBusy ? "Searching…" : "Run Search"}
          </Button>
        </div>
      ) : null}

      {ctrl.searchError ? (
        <div
          role="alert"
          className="rounded border border-[var(--loss)]/40 bg-[var(--loss-soft)] px-2 py-1 text-[10.5px] text-[var(--loss)]"
        >
          {ctrl.searchError}
        </div>
      ) : null}
    </div>
  );
}

function SearchActionCard(props: {
  rankIdx: number;
  row: SearchActionRow;
  isSearchChosen: boolean;
  isRecordedChosen: boolean;
  hasOverrides: boolean;
  disabled: boolean;
  onBranchAction: (actionIdx: number) => void;
}): ReactNode {
  const { row } = props;
  const wrPct = Math.max(0, Math.min(100, row.search_wr * 100));
  const lowPct = Math.max(0, Math.min(100, row.ci_low * 100));
  const highPct = Math.max(lowPct, Math.min(100, row.ci_high * 100));
  const accentColor = props.hasOverrides ? "var(--fork-fill)" : "var(--accent)";
  const boxCls = props.isSearchChosen
    ? "bg-[var(--accent-soft)]/55 border-[var(--accent)]/45 hover:border-[var(--accent)]"
    : "bg-[var(--bg)]/55 border-[var(--border)]/70 hover:bg-[var(--surface-2)] hover:border-[var(--accent)]/40";

  return (
    <button
      type="button"
      disabled={props.disabled}
      onClick={(): void => props.onBranchAction(row.index)}
      title={`Click to play: ${row.label}`}
      aria-label={`Play search action ${props.rankIdx + 1}: ${row.label}`}
      className={`w-full text-left rounded-md px-2.5 py-2 border transition-colors cursor-pointer disabled:opacity-50 ${boxCls}`}
    >
      <div className="flex items-start justify-between gap-2 text-[11px] mb-1">
        <span className="font-semibold leading-snug break-words line-clamp-2 min-w-0">
          {props.rankIdx + 1}. {row.label}
        </span>
        <span className="tabular shrink-0 flex items-center gap-1 pt-0.5">
          {props.isRecordedChosen ? (
            <span title="Action taken in recorded game" className="text-[var(--accent)]">
              ★
            </span>
          ) : null}
          <strong>{(row.search_wr * 100).toFixed(1)}%</strong>
        </span>
      </div>
      <div className="relative h-[6px] rounded-full bg-[var(--surface-2)] overflow-hidden mb-1">
        <div
          className="h-full rounded-full transition-all duration-150"
          style={{ width: `${Math.max(2, wrPct)}%`, background: accentColor }}
        />
        <div
          aria-hidden="true"
          className="absolute top-0 bottom-0 border-x border-[var(--text)]/50 bg-[var(--text)]/10"
          style={{ left: `${lowPct}%`, width: `${Math.max(1, highPct - lowPct)}%` }}
        />
      </div>
      <div className="flex items-center justify-between text-[9.5px] tabular text-[var(--text-3)]">
        <span>
          95% CI [{Math.round(lowPct)}–{Math.round(highPct)}%]
        </span>
        <span className="flex items-center gap-1.5">
          <span>Policy {Math.round(row.policy_prob * 100)}%</span>
          <span
            title="Search win-rate delta vs raw policy's top-ranked action"
            className={signedToneClass(row.delta_vs_policy)}
          >
            vs top {formatSignedDeltaPct(row.delta_vs_policy)}
          </span>
        </span>
      </div>
    </button>
  );
}

function SearchWarningsBanner(props: { warnings?: string[] }): ReactNode {
  if (!props.warnings || props.warnings.length === 0) return null;
  return (
    <output className="block rounded border border-[var(--fork)]/40 bg-[var(--fork-soft)]/40 px-2 py-1 text-[10px] text-[var(--text-2)] shrink-0">
      {props.warnings.join(" ")}
    </output>
  );
}

export function SearchMovesList(props: {
  board: BoardState;
  ctrl: TabletopBeliefDockProps;
  disabled: boolean;
  onBranchAction: (actionIdx: number) => void;
}): ReactNode {
  const { board, ctrl } = props;
  if (board.actions_hidden) {
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
            <strong>Omniscient</strong> to inspect or search Opponent&apos;s play.
          </span>
        </div>
      </div>
    );
  }
  const res = ctrl.searchResult;
  const sorted = res ? res.actions.toSorted((a, b) => b.search_wr - a.search_wr) : [];
  const title = res
    ? `Search (${res.worlds_sampled}w · ${res.rollout_horizon}p)`
    : `Search (${ctrl.worlds}w)`;
  let status = "Ready";
  if (res) status = `${Math.round(res.elapsed_ms)} ms`;
  else if (ctrl.searchBusy) status = "Running…";

  return (
    <div className="p-3 flex-1 min-h-0 flex flex-col gap-2 overflow-y-auto">
      <div className="flex items-center justify-between gap-2 text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] shrink-0">
        <span>{title}</span>
        <span className="tabular normal-case font-medium">{status}</span>
      </div>
      <SearchWarningsBanner warnings={res?.warnings} />
      {sorted.length === 0 ? (
        <div className="py-6 text-center text-[11px] text-[var(--text-3)]">
          {ctrl.searchBusy
            ? "Sampling opponent worlds and evaluating legal actions…"
            : "Click Run Search to score legal actions across sampled worlds."}
        </div>
      ) : (
        <div className="flex flex-col gap-1.5">
          {sorted.map((row, idx) => (
            <SearchActionCard
              key={row.index}
              rankIdx={idx}
              row={row}
              isSearchChosen={res?.chosen_action === row.index}
              isRecordedChosen={board.chosen_action === row.index}
              hasOverrides={ctrl.overrideCount > 0}
              disabled={props.disabled || ctrl.searchBusy}
              onBranchAction={props.onBranchAction}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function BeliefDockBody(props: {
  board: BoardState;
  evaluation: EvaluationPair;
  ctrl?: TabletopBeliefDockProps;
  disabled: boolean;
  onBranchAction: (actionIdx: number) => void;
  policyFallback: ReactNode;
}): ReactNode {
  const { ctrl } = props;
  if (!ctrl) return props.policyFallback;
  return (
    <>
      <BeliefSearchBar ctrl={ctrl} />
      <TurnZeroV0Strip board={props.board} evaluation={props.evaluation} ctrl={ctrl} />
      {ctrl.mode === "search" ? (
        <SearchMovesList
          board={props.board}
          ctrl={ctrl}
          disabled={props.disabled}
          onBranchAction={props.onBranchAction}
        />
      ) : (
        props.policyFallback
      )}
    </>
  );
}

/** Right inspector dock for Tabletop: live checkpoint diff, any-step Fork/Branch lab, legal actions & Watchlist. */

import {
  Bookmark,
  EyeOff,
  FastForward,
  GitBranch,
  RotateCcw,
  Search,
  Undo2,
  X,
} from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/Primitives";
import { cardImageUrl } from "@/lib/api";
import type { BoardState, BranchStepEvent, EvaluationPair, ForkEdit } from "@/lib/types";

const BASIC_LANDS = new Set(["Plains", "Island", "Swamp", "Mountain", "Forest"]);

function deckCardPoolForFork(args: {
  board: BoardState;
  seatSide: "hero" | "opp";
  zone: "add_battlefield" | "add_hand_card";
  filterText: string;
}): string[] {
  const { board, seatSide, zone, filterText } = args;
  const player = seatSide === "hero" ? board.hero : board.opp;
  const nonPerms = new Set(board.non_permanent_names ?? []);
  const needle = filterText.trim().toLowerCase();
  const deckRaw = [...player.library_summary.map((row) => row.name), ...player.graveyard];
  return [...new Set(deckRaw)]
    .filter((name) => {
      if (BASIC_LANDS.has(name) || name === "Unknown") return false;
      if (zone === "add_battlefield" && nonPerms.has(name)) return false;
      return needle === "" || name.toLowerCase().includes(needle);
    })
    .slice(0, 12);
}

function describeForkEdit(edit: ForkEdit, heroSeat: number): string {
  const who = edit.seat === heroSeat ? "Hero" : "Opp";
  if (edit.op === "set_life") return `${who} life = ${edit.value ?? "?"}`;
  if (edit.op === "add_battlefield") return `${who} +Board: ${edit.card ?? ""}`;
  if (edit.op === "add_hand_card") return `${who} +Hand: ${edit.card ?? ""}`;
  return `${who} clear hand`;
}

function resolveTargetSeat(targetSide: "hero" | "opp", heroSeat: number): 0 | 1 {
  const heroNum: 0 | 1 = heroSeat === 1 ? 1 : 0;
  const oppNum: 0 | 1 = heroNum === 1 ? 0 : 1;
  return targetSide === "hero" ? heroNum : oppNum;
}

const SIDE_OPTIONS: ReadonlyArray<{ value: "hero" | "opp"; label: string }> = [
  { value: "hero", label: "Hero" },
  { value: "opp", label: "Opp" },
];

const ZONE_OPTIONS: ReadonlyArray<{ value: "add_battlefield" | "add_hand_card"; label: string }> = [
  { value: "add_battlefield", label: "+ Board" },
  { value: "add_hand_card", label: "+ Hand" },
];

function ForkBoardEditor(props: {
  board: BoardState;
  heroSeat: number;
  nearestPriorityStep: number | null;
  onJumpToStep: (step: number) => void;
  onAddEdit: (edit: ForkEdit) => void;
}): ReactNode {
  const [targetZone, setTargetZone] = useState<"add_battlefield" | "add_hand_card">(
    "add_battlefield",
  );
  const [targetSide, setTargetSide] = useState<"hero" | "opp">("hero");
  const [cardFilter, setCardFilter] = useState("");
  const isPriority = props.board.action_space_kind === "PRIORITY";

  if (!isPriority) {
    return (
      <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-2 flex flex-col gap-1.5 text-[10.5px] text-[var(--text-2)]">
        <span>
          Life &amp; card injections apply on <strong>Priority</strong> steps (current:{" "}
          <span className="tabular">{props.board.action_space_kind}</span>). Action branching works
          right here—click any legal action below.
        </span>
        {props.nearestPriorityStep === null ? null : (
          <Button
            size="sm"
            onClick={(): void => props.onJumpToStep(props.nearestPriorityStep ?? 0)}
          >
            Jump to Priority step #{props.nearestPriorityStep}
          </Button>
        )}
      </div>
    );
  }

  const seatNum = resolveTargetSeat(targetSide, props.heroSeat);
  const deckCards = deckCardPoolForFork({
    board: props.board,
    seatSide: targetSide,
    zone: targetZone,
    filterText: cardFilter,
  });
  const oppHidden = targetSide === "opp" && props.board.opp.library_hidden;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-1">
        <div className="inline-flex rounded border border-[var(--border)] bg-[var(--bg)] p-0.5 text-[9.5px] font-semibold">
          {SIDE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={(): void => setTargetSide(opt.value)}
              className={`px-1.5 py-0.5 rounded-sm cursor-pointer ${
                targetSide === opt.value
                  ? "bg-[var(--fork-soft)] text-[var(--fork)]"
                  : "text-[var(--text-3)]"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <div className="inline-flex rounded border border-[var(--border)] bg-[var(--bg)] p-0.5 text-[9.5px] font-semibold">
          {ZONE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={(): void => setTargetZone(opt.value)}
              className={`px-1.5 py-0.5 rounded-sm cursor-pointer ${
                targetZone === opt.value
                  ? "bg-[var(--fork-soft)] text-[var(--fork)]"
                  : "text-[var(--text-3)]"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="relative">
        <Search className="w-3 h-3 absolute left-2 top-1/2 -translate-y-1/2 text-[var(--text-3)]" />
        <input
          type="search"
          value={cardFilter}
          onChange={(e): void => setCardFilter(e.target.value)}
          placeholder="Search player's deck cards…"
          aria-label="Filter fork cards"
          className="w-full h-6 rounded border border-[var(--border)] bg-[var(--bg)] pl-6 pr-2 text-[10.5px]"
        />
      </div>

      {deckCards.length === 0 ? (
        <div className="text-[10px] text-[var(--text-3)] italic">
          {oppHidden
            ? "Switch to Omniscient to inspect and inject cards from Opponent's library."
            : "No matching cards in this player's library or graveyard."}
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-1">
          {deckCards.map((cardName) => (
            <button
              key={cardName}
              type="button"
              onClick={(e): void => {
                e.currentTarget.blur();
                props.onAddEdit({ op: targetZone, seat: seatNum, card: cardName });
              }}
              className="flex items-center gap-1.5 rounded border border-[var(--border)] bg-[var(--bg)] p-1 text-left text-[10px] font-medium hover:border-[var(--fork)] hover:bg-[var(--fork-soft)] cursor-pointer min-w-0 overflow-hidden"
            >
              <img
                src={cardImageUrl(cardName)}
                alt={cardName}
                className="w-[18px] h-[25px] rounded-[2px] object-cover shrink-0"
              />
              <span className="truncate">{cardName}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function formatForkSummary(editCount: number, branchCount: number): string {
  const parts: string[] = [];
  if (editCount > 0) parts.push(`${editCount} edit${editCount === 1 ? "" : "s"}`);
  if (branchCount > 0) parts.push(`+${branchCount} branched`);
  return parts.length > 0 ? parts.join(" · ") : "0 edits";
}

function ForkShiftSummary(props: {
  editCount: number;
  branchCount: number;
  evaluation: EvaluationPair;
  baseEvaluation: EvaluationPair;
  showCompareDelta: boolean;
}): ReactNode {
  const baseA = Math.round(props.baseEvaluation.primary.wdl.win * 100);
  const curA = Math.round(props.evaluation.primary.wdl.win * 100);
  const deltaA = curA - baseA;
  const baseB = Math.round(props.baseEvaluation.compare.wdl.win * 100);
  const curB = Math.round(props.evaluation.compare.wdl.win * 100);
  const deltaB = curB - baseB;
  return (
    <div className="rounded-md border border-[var(--fork)]/30 bg-[var(--surface)] px-2.5 py-1.5 flex flex-col gap-0.5">
      <div className="flex items-center justify-between gap-2 text-[11px]">
        <span className="text-[var(--text-2)] truncate">
          {formatForkSummary(props.editCount, props.branchCount)}
        </span>
        <span className="tabular font-bold text-[var(--fork)] shrink-0">
          A: {baseA}% → {curA}% ({deltaA >= 0 ? `+${deltaA}` : deltaA}%)
        </span>
      </div>
      {props.showCompareDelta ? (
        <div className="flex items-center justify-end text-[10px] tabular text-[var(--text-2)]">
          B: {baseB}% → {curB}% ({deltaB >= 0 ? `+${deltaB}` : deltaB}%)
        </div>
      ) : null}
    </div>
  );
}

function ForkChipsList(props: {
  heroSeat: number;
  edits: ForkEdit[];
  branchEvents: BranchStepEvent[];
  onRemoveEdit: (idx: number) => void;
}): ReactNode {
  return (
    <>
      {props.branchEvents.length > 0 ? (
        <div className="flex flex-wrap gap-1">
          {props.branchEvents.map((ev) => (
            <span
              key={ev.branch_index}
              className="inline-flex items-center gap-1 rounded bg-[var(--surface)] border border-[var(--fork)]/35 px-1.5 py-0.5 text-[10px]"
            >
              <span className="tabular text-[var(--fork)] font-bold">+{ev.branch_index + 1}</span>
              <span className="max-w-[150px] truncate">{ev.label}</span>
            </span>
          ))}
        </div>
      ) : null}

      {props.edits.length > 0 ? (
        <div className="flex flex-wrap gap-1">
          {props.edits.map((edit, idx) => {
            const label = describeForkEdit(edit, props.heroSeat);
            return (
              <span
                key={`${edit.op}-${edit.seat}-${edit.card ?? edit.value ?? "x"}-${String(idx)}`}
                className="inline-flex items-center gap-1 rounded bg-[var(--surface)] border border-[var(--border)] px-1.5 py-0.5 text-[10px] font-medium"
              >
                <span className="max-w-[140px] truncate">{label}</span>
                <button
                  type="button"
                  aria-label={`Undo ${label}`}
                  onClick={(): void => props.onRemoveEdit(idx)}
                  className="text-[var(--text-3)] hover:text-[var(--loss)] cursor-pointer"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            );
          })}
        </div>
      ) : null}
    </>
  );
}

function canStepModelTop(args: {
  branchDone: boolean;
  board: BoardState;
  evaluation: EvaluationPair;
}): boolean {
  return (
    !args.branchDone &&
    !args.board.actions_hidden &&
    args.board.legal_actions.length > 0 &&
    args.evaluation.primary.probs.length > 0
  );
}

function ForkAlerts(props: { error: string | null; warnings: string[] }): ReactNode {
  return (
    <>
      {props.error ? (
        <div className="rounded bg-[var(--loss-soft)] text-[var(--loss)] px-2 py-1 text-[10px]">
          {props.error}
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
    </>
  );
}

function ForkControlsSection(props: {
  board: BoardState;
  evaluation: EvaluationPair;
  baseEvaluation: EvaluationPair;
  showCompareDelta: boolean;
  forkBusy: boolean;
  edits: ForkEdit[];
  branchEvents: BranchStepEvent[];
  branchDone: boolean;
  nearestPriorityStep: number | null;
  warnings: string[];
  error: string | null;
  onAddEdit: (edit: ForkEdit) => void;
  onRemoveEdit: (idx: number) => void;
  onStepModelTopMove: () => void;
  onPopBranchAction: () => void;
  onResetFork: () => void;
  onJumpToStep: (step: number) => void;
}): ReactNode {
  const [boardEditorOpen, setBoardEditorOpen] = useState(true);
  const heroSeat = props.board.hero_seat ?? 0;
  const hasMutations = props.edits.length > 0 || props.branchEvents.length > 0;
  const canStepTop = canStepModelTop({
    branchDone: props.branchDone,
    board: props.board,
    evaluation: props.evaluation,
  });

  return (
    <div className="p-3 border-b border-[var(--border)] flex flex-col gap-2 bg-[var(--fork-soft)]/35">
      <div className="flex items-center justify-between gap-2 text-[10px] font-bold uppercase tracking-wider">
        <span className="inline-flex items-center gap-1 text-[var(--fork)]">
          <GitBranch className="w-3 h-3" />
          What-If &amp; Branch
        </span>
        <div className="flex items-center gap-1.5">
          {props.forkBusy ? <span className="tabular text-[var(--fork)]">Evaluating…</span> : null}
          {hasMutations ? (
            <button
              type="button"
              onClick={props.onResetFork}
              className="inline-flex items-center gap-0.5 text-[10px] font-semibold normal-case text-[var(--text-2)] hover:text-[var(--loss)] cursor-pointer"
            >
              <RotateCcw className="w-2.5 h-2.5" />
              Reset
            </button>
          ) : null}
        </div>
      </div>

      <ForkShiftSummary
        editCount={props.edits.length}
        branchCount={props.branchEvents.length}
        evaluation={props.evaluation}
        baseEvaluation={props.baseEvaluation}
        showCompareDelta={props.showCompareDelta}
      />

      <div className="flex items-center gap-1.5">
        <Button
          size="sm"
          variant="primary"
          className="flex-1"
          disabled={!canStepTop || props.forkBusy}
          onClick={props.onStepModelTopMove}
        >
          <FastForward className="w-3 h-3" />
          Play Model Top Move
        </Button>
        {props.branchEvents.length > 0 ? (
          <Button size="sm" onClick={props.onPopBranchAction} title="Undo last branched action">
            <Undo2 className="w-3 h-3" />
            Undo
          </Button>
        ) : null}
      </div>

      <ForkChipsList
        heroSeat={heroSeat}
        edits={props.edits}
        branchEvents={props.branchEvents}
        onRemoveEdit={props.onRemoveEdit}
      />

      <button
        type="button"
        onClick={(): void => setBoardEditorOpen((v) => !v)}
        className="text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--text-3)] hover:text-[var(--text)] cursor-pointer"
      >
        {boardEditorOpen ? "▾ Add Cards / Edit Board" : "▸ Add Cards / Edit Board"}
      </button>

      {boardEditorOpen ? (
        <ForkBoardEditor
          board={props.board}
          heroSeat={heroSeat}
          nearestPriorityStep={props.nearestPriorityStep}
          onJumpToStep={props.onJumpToStep}
          onAddEdit={props.onAddEdit}
        />
      ) : null}

      <ForkAlerts error={props.error} warnings={props.warnings} />
    </div>
  );
}

function ActionDeltaTag(props: { delta: number }): ReactNode {
  if (props.delta === 0) return null;
  const cls =
    props.delta > 0
      ? "text-[var(--win)] font-bold text-[10px]"
      : "text-[var(--loss)] font-bold text-[10px]";
  return <span className={cls}>({props.delta > 0 ? `+${props.delta}` : props.delta}%)</span>;
}

function LegalActionRowButton(props: {
  rankIdx: number;
  act: BoardState["legal_actions"][number];
  pA: number;
  delta: number;
  isChosen: boolean;
  forkMode: boolean;
  forkBusy: boolean;
  onBranchAction: (actionIdx: number) => void;
  onHoverFocus: (ids: number[]) => void;
}): ReactNode {
  const { act, rankIdx, pA, delta, isChosen, forkMode, forkBusy } = props;
  const actionAria = forkMode
    ? `Branch action ${rankIdx + 1}: ${act.label}`
    : `Action ${rankIdx + 1}: ${act.label}`;
  return (
    <button
      type="button"
      aria-label={actionAria}
      disabled={forkMode && forkBusy}
      title={forkMode ? `Branch: play "${act.label}"` : act.label}
      onMouseEnter={(): void => props.onHoverFocus(act.focus_ids)}
      onMouseLeave={(): void => props.onHoverFocus([])}
      onClick={(): void => {
        if (forkMode && !forkBusy) props.onBranchAction(act.index);
      }}
      className={`w-full text-left rounded-md px-2 py-1.5 border transition-colors disabled:opacity-50 ${
        forkMode ? "cursor-pointer hover:border-[var(--fork)]" : "cursor-default"
      } ${
        rankIdx === 0
          ? "bg-[var(--win-soft)] border-[var(--win)]/25"
          : "border-transparent hover:bg-[var(--surface-2)] hover:border-[var(--border)]"
      }`}
    >
      <div className="flex items-start justify-between gap-2 text-[11px] mb-1">
        <span className="font-semibold leading-snug break-words line-clamp-2 min-w-0">
          {rankIdx + 1}. {act.label}
        </span>
        <span className="tabular shrink-0 flex items-center gap-1 pt-0.5">
          {isChosen ? (
            <span title="Action taken in recorded game" className="text-[var(--accent)]">
              ★
            </span>
          ) : null}
          <strong>{pA}%</strong>
          <ActionDeltaTag delta={delta} />
        </span>
      </div>
      <div className="h-[5px] rounded bg-[var(--surface-2)] overflow-hidden">
        <div
          className="h-full rounded transition-all"
          style={{
            width: `${Math.max(2, pA)}%`,
            background: rankIdx === 0 ? "var(--win-fill)" : "var(--accent)",
          }}
        />
      </div>
    </button>
  );
}

function LegalActionsSection(props: {
  board: BoardState;
  evaluation: EvaluationPair;
  forkMode: boolean;
  forkBusy: boolean;
  onBranchAction: (actionIdx: number) => void;
  onHoverFocus: (ids: number[]) => void;
}): ReactNode {
  const [showAll, setShowAll] = useState(false);

  if (props.board.actions_hidden) {
    return (
      <div className="p-3 border-b border-[var(--border)]">
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
  const probsA = props.evaluation.primary.probs;
  const probsB = props.evaluation.compare.probs;
  const ranked = actions
    .map((act, idx) => ({
      act,
      pA: Math.round((probsA[idx] ?? 0) * 100),
      pB: Math.round((probsB[idx] ?? 0) * 100),
    }))
    .toSorted((a, b) => b.pA - a.pA);
  const maxShow = props.forkMode ? 5 : 6;
  const visible = showAll ? ranked : ranked.slice(0, maxShow);
  const whoLabel = props.board.actor_is_hero === false ? "OPP" : "HERO";

  return (
    <div className="p-3 border-b border-[var(--border)]">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] mb-2">
        <span>
          {whoLabel} · {props.board.action_space_kind} ({actions.length})
        </span>
        <span>{props.forkMode ? "Click to play" : "Δ vs B"}</span>
      </div>
      <div className="flex flex-col gap-1.5">
        {visible.map((row, idx) => (
          <LegalActionRowButton
            key={row.act.index}
            rankIdx={idx}
            act={row.act}
            pA={row.pA}
            delta={row.pA - row.pB}
            isChosen={props.board.chosen_action === row.act.index}
            forkMode={props.forkMode}
            forkBusy={props.forkBusy}
            onBranchAction={props.onBranchAction}
            onHoverFocus={props.onHoverFocus}
          />
        ))}
      </div>
      {actions.length > maxShow ? (
        <button
          type="button"
          onClick={(): void => setShowAll((prev) => !prev)}
          className="mt-2 w-full rounded-md border border-dashed border-[var(--border-strong)] py-1 text-[10.5px] font-medium text-[var(--text-2)] hover:text-[var(--text)] hover:border-[var(--accent)] cursor-pointer"
        >
          {showAll ? `Show top ${maxShow} ▴` : `Show all ${actions.length} actions ▾`}
        </button>
      ) : null}
    </div>
  );
}

function WdlAndWatchlistFooter(props: {
  checkpointName: string;
  winPct: number;
  drawPct: number;
  lossPct: number;
  wdlDelta: number;
  showDelta: boolean;
  pinned: boolean;
  onPinPosition: (note: string) => void;
}): ReactNode {
  const [pinNote, setPinNote] = useState("");
  const deltaText = props.wdlDelta > 0 ? `+${props.wdlDelta}%` : `${props.wdlDelta}%`;
  return (
    <div className="p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)]">
        <span>Hero Win Estimate</span>
        <span className="tabular">{props.checkpointName}</span>
      </div>
      <div className="h-[18px] rounded-[5px] overflow-hidden flex text-[10px] font-bold text-white tabular">
        <div
          className="bg-[var(--win-fill)] flex items-center justify-center"
          style={{ width: `${props.winPct}%` }}
        >
          {props.winPct >= 22 ? `${props.winPct}% Win` : ""}
        </div>
        {props.drawPct > 0 ? (
          <div className="bg-zinc-600" style={{ width: `${props.drawPct}%` }} />
        ) : null}
        <div
          className="bg-[var(--loss-fill)] flex items-center justify-center"
          style={{ width: `${props.lossPct}%` }}
        >
          {props.lossPct >= 22 ? `${props.lossPct}% Loss` : ""}
        </div>
      </div>
      <div className="flex items-center justify-between tabular text-[10.5px] text-[var(--text-2)]">
        <span>
          Hero {props.winPct}% Win
          {props.drawPct > 0 ? ` · ${props.drawPct}% Draw` : ""} · {props.lossPct}% Loss
        </span>
        {props.showDelta ? (
          <span
            className={
              props.wdlDelta > 0
                ? "text-[var(--win)] font-semibold"
                : "text-[var(--loss)] font-semibold"
            }
          >
            {deltaText} vs B
          </span>
        ) : null}
      </div>
      <div className="flex items-center gap-1.5">
        <input
          type="text"
          value={pinNote}
          onChange={(e): void => setPinNote(e.target.value)}
          placeholder="Watchlist note…"
          aria-label="Watchlist note"
          className="flex-1 min-w-0 h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-[11px]"
        />
        <Button
          variant={props.pinned ? "primary" : "default"}
          size="md"
          onClick={(): void => {
            props.onPinPosition(pinNote);
            setPinNote("");
          }}
        >
          <Bookmark className="w-3.5 h-3.5" />
          {props.pinned ? "Pinned" : "Pin"}
        </Button>
      </div>
    </div>
  );
}

export function TabletopDock(props: {
  board: BoardState;
  evaluation: EvaluationPair;
  baseEvaluation: EvaluationPair;
  checkpoints: string[];
  ckptA: string;
  ckptB: string;
  onChangeCkptA: (v: string) => void;
  onChangeCkptB: (v: string) => void;
  forkMode: boolean;
  forkBusy: boolean;
  forkEdits: ForkEdit[];
  branchEvents: BranchStepEvent[];
  branchDone: boolean;
  nearestPriorityStep: number | null;
  forkWarnings: string[];
  forkError: string | null;
  onAddForkEdit: (edit: ForkEdit) => void;
  onRemoveForkEdit: (idx: number) => void;
  onBranchAction: (actionIdx: number) => void;
  onStepModelTopMove: () => void;
  onPopBranchAction: () => void;
  onResetFork: () => void;
  onJumpToStep: (step: number) => void;
  onHoverActionFocus: (ids: number[]) => void;
  onPinPosition: (note: string) => void;
  pinned: boolean;
}): ReactNode {
  const wdl = props.evaluation.primary.wdl;
  const winPct = Math.round(wdl.win * 100);
  const drawPct = Math.round(wdl.draw * 100);
  const lossPct = Math.max(0, 100 - winPct - drawPct);
  const compareWinPct = Math.round(props.evaluation.compare.wdl.win * 100);
  const wdlDelta = winPct - compareWinPct;
  const showCompare = props.ckptB !== "same";

  return (
    <aside
      aria-label="Policy and counterfactual inspector"
      className="w-full lg:w-[330px] shrink-0 border-t lg:border-t-0 lg:border-l border-[var(--border)] bg-[var(--surface)] flex flex-col min-h-0 lg:overflow-y-auto"
    >
      <div className="p-3 border-b border-[var(--border)]">
        <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] mb-1.5">
          <span>Checkpoint</span>
          {props.board.actions_hidden ? (
            <span className="normal-case font-medium text-[var(--text-3)]">Hidden in Hero POV</span>
          ) : (
            <span className="tabular" title="Total Variation and KL(A || B) in nats">
              TV {props.evaluation.diff.tv.toFixed(2)} · KL {props.evaluation.diff.kl.toFixed(2)}
            </span>
          )}
        </div>
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-1.5">
          <select
            value={props.ckptA}
            onChange={(e): void => props.onChangeCkptA(e.target.value)}
            aria-label="Primary checkpoint"
            className="h-7 rounded-md border border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)] px-1.5 text-[11px] font-semibold tabular"
          >
            <option value="heuristic">Heuristic</option>
            {props.checkpoints.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <span className="text-[10px] text-[var(--text-3)]">vs</span>
          <select
            value={props.ckptB}
            onChange={(e): void => props.onChangeCkptB(e.target.value)}
            aria-label="Comparison checkpoint"
            className="h-7 rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-1.5 text-[11px] font-semibold tabular"
          >
            <option value="same">Same</option>
            <option value="heuristic">Heuristic</option>
            {props.checkpoints.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        {(props.evaluation.warnings ?? []).map((w) => (
          <div
            key={w}
            className="mt-1.5 rounded bg-[var(--warn-soft)] text-[var(--warn)] px-2 py-1 text-[10px]"
          >
            {w}
          </div>
        ))}
      </div>

      {props.forkMode ? (
        <ForkControlsSection
          board={props.board}
          evaluation={props.evaluation}
          baseEvaluation={props.baseEvaluation}
          showCompareDelta={showCompare}
          forkBusy={props.forkBusy}
          edits={props.forkEdits}
          branchEvents={props.branchEvents}
          branchDone={props.branchDone}
          nearestPriorityStep={props.nearestPriorityStep}
          warnings={props.forkWarnings}
          error={props.forkError}
          onAddEdit={props.onAddForkEdit}
          onRemoveEdit={props.onRemoveForkEdit}
          onStepModelTopMove={props.onStepModelTopMove}
          onPopBranchAction={props.onPopBranchAction}
          onResetFork={props.onResetFork}
          onJumpToStep={props.onJumpToStep}
        />
      ) : null}

      <LegalActionsSection
        board={props.board}
        evaluation={props.evaluation}
        forkMode={props.forkMode}
        forkBusy={props.forkBusy}
        onBranchAction={props.onBranchAction}
        onHoverFocus={props.onHoverActionFocus}
      />

      <WdlAndWatchlistFooter
        checkpointName={props.evaluation.primary.checkpoint}
        winPct={winPct}
        drawPct={drawPct}
        lossPct={lossPct}
        wdlDelta={wdlDelta}
        showDelta={showCompare && wdlDelta !== 0}
        pinned={props.pinned}
        onPinPosition={props.onPinPosition}
      />
    </aside>
  );
}

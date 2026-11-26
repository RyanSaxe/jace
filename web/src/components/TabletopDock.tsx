/** Right Swiss-grid inspector dock for Tabletop: checkpoint diff, What-If Fork controls, legal actions, WDL & Watchlist. */

import { Bookmark, EyeOff, X } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/Primitives";
import { cardImageUrl } from "@/lib/api";
import type { BoardState, EvaluationPair, ForkEdit } from "@/lib/types";

const BASIC_LANDS = new Set(["Plains", "Island", "Swamp", "Mountain", "Forest"]);

function deckForkCandidates(args: {
  board: BoardState;
  seatSide: "hero" | "opp";
  zone: "add_battlefield" | "add_hand_card";
  filterText: string;
}): string[] {
  const { board, seatSide, zone, filterText } = args;
  const player = seatSide === "hero" ? board.hero : board.opp;
  const nonPerms = new Set(board.non_permanent_names ?? []);
  const rawPool = [...player.library_summary.map((row) => row.name), ...player.graveyard];
  const needle = filterText.trim().toLowerCase();
  const unique = [...new Set(rawPool)].filter((name) => {
    if (BASIC_LANDS.has(name) || name === "Unknown") return false;
    if (zone === "add_battlefield" && nonPerms.has(name)) return false;
    return needle === "" || name.toLowerCase().includes(needle);
  });
  return unique.slice(0, 8);
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

function ForkEditChips(props: {
  edits: ForkEdit[];
  heroSeat: number;
  onRemoveEdit: (idx: number) => void;
}): ReactNode {
  if (props.edits.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1">
      {props.edits.map((edit, idx) => {
        const label = describeForkEdit(edit, props.heroSeat);
        return (
          <span
            key={`${edit.op}-${edit.seat}-${edit.card ?? edit.value ?? "x"}-${String(idx)}`}
            className="inline-flex items-center gap-1 rounded bg-[var(--surface-2)] border border-[var(--border)] px-1.5 py-0.5 text-[10px] font-medium"
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
  );
}

const SIDE_OPTIONS: ReadonlyArray<{ value: "hero" | "opp"; label: string }> = [
  { value: "hero", label: "Hero" },
  { value: "opp", label: "Opp" },
];

const ZONE_OPTIONS: ReadonlyArray<{ value: "add_battlefield" | "add_hand_card"; label: string }> = [
  { value: "add_battlefield", label: "+ Board" },
  { value: "add_hand_card", label: "+ Hand" },
];

function ForkCardPicker(props: {
  board: BoardState;
  heroSeat: number;
  onAddEdit: (edit: ForkEdit) => void;
}): ReactNode {
  const [targetZone, setTargetZone] = useState<"add_battlefield" | "add_hand_card">(
    "add_battlefield",
  );
  const [targetSide, setTargetSide] = useState<"hero" | "opp">("hero");
  const [cardFilter, setCardFilter] = useState("");
  const seatNum = resolveTargetSeat(targetSide, props.heroSeat);
  const candidates = deckForkCandidates({
    board: props.board,
    seatSide: targetSide,
    zone: targetZone,
    filterText: cardFilter,
  });
  const oppHidden = targetSide === "opp" && props.board.opp.library_hidden;

  return (
    <>
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

      <input
        type="search"
        value={cardFilter}
        onChange={(e): void => setCardFilter(e.target.value)}
        placeholder={
          targetZone === "add_battlefield"
            ? "Filter permanents in deck/GY…"
            : "Filter cards in deck/GY…"
        }
        aria-label="Filter fork cards"
        className="h-6 rounded border border-[var(--border)] bg-[var(--bg)] px-2 text-[10.5px]"
      />

      {candidates.length === 0 ? (
        <div className="text-[10px] text-[var(--text-3)] italic">
          {oppHidden
            ? "Switch to Omniscient POV to pull cards from Opponent's library."
            : "No matching cards available in library or graveyard."}
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-1">
          {candidates.map((cardName) => (
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
    </>
  );
}

function ForkControlsSection(props: {
  board: BoardState;
  forkBusy: boolean;
  edits: ForkEdit[];
  baseWinPct: number;
  winPct: number;
  warnings: string[];
  error: string | null;
  onAddEdit: (edit: ForkEdit) => void;
  onRemoveEdit: (idx: number) => void;
}): ReactNode {
  const isPriority = props.board.action_space_kind === "PRIORITY";
  const heroSeat = props.board.hero_seat ?? 0;
  const deltaWin = props.winPct - props.baseWinPct;
  const deltaText = deltaWin >= 0 ? `+${deltaWin}` : String(deltaWin);

  return (
    <div className="p-3 border-b border-[var(--border)] flex flex-col gap-2">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider">
        <span className="text-[var(--text-3)]">Counterfactual What-If</span>
        <span className="tabular text-[var(--fork)]">
          {props.forkBusy ? "Evaluating…" : "Live Fork"}
        </span>
      </div>

      {isPriority ? (
        <>
          <div className="rounded-md border border-[var(--fork)]/30 bg-[var(--fork-soft)] px-2.5 py-1.5 flex items-center justify-between">
            <span className="text-[10.5px] text-[var(--text-2)]">
              {props.edits.length} edit{props.edits.length === 1 ? "" : "s"}
            </span>
            <span className="tabular text-xs font-bold text-[var(--fork)]">
              Hero {props.baseWinPct}% → {props.winPct}% ({deltaText}%)
            </span>
          </div>
          <ForkEditChips
            edits={props.edits}
            heroSeat={heroSeat}
            onRemoveEdit={props.onRemoveEdit}
          />
          <ForkCardPicker board={props.board} heroSeat={heroSeat} onAddEdit={props.onAddEdit} />
        </>
      ) : (
        <div className="rounded-md border border-[var(--warn)]/40 bg-[var(--warn-soft)] px-2.5 py-2 text-[10.5px] text-[var(--warn)] leading-snug">
          Fork edits apply only on <strong>PRIORITY</strong> steps (current:{" "}
          <span className="tabular">{props.board.action_space_kind}</span>). Step to a Priority step
          to test what-if scenarios.
        </div>
      )}

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
    </div>
  );
}

function LegalActionsSection(props: {
  board: BoardState;
  evaluation: EvaluationPair;
  compact: boolean;
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
            {props.board.action_space_kind === "PRIORITY"
              ? "Opponent has priority."
              : `Opponent is deciding (${props.board.action_space_kind}).`}{" "}
            Legal actions are hidden in <strong>Hero POV</strong> to avoid leaking Opponent&apos;s
            hand. Switch to <strong>Omniscient</strong> to inspect Opponent&apos;s policy.
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
  const maxShow = props.compact ? 4 : 5;
  const visible = showAll ? ranked : ranked.slice(0, maxShow);
  const whoLabel = props.board.actor_is_hero === false ? "OPP" : "HERO";

  return (
    <div className="p-3 border-b border-[var(--border)]">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] mb-2">
        <span>
          {whoLabel} · {props.board.action_space_kind} ({showAll ? "ALL" : `TOP ${visible.length}`}{" "}
          OF {actions.length})
        </span>
        <span>Δ vs B</span>
      </div>
      <div className="flex flex-col gap-1.5">
        {visible.map((row, idx) => {
          const delta = row.pA - row.pB;
          const isChosen = props.board.chosen_action === row.act.index;
          return (
            <div
              key={row.act.index}
              title={row.act.label}
              onMouseEnter={(): void => props.onHoverFocus(row.act.focus_ids)}
              onMouseLeave={(): void => props.onHoverFocus([])}
              className={`rounded-md px-2 py-1.5 border cursor-pointer transition-colors ${
                idx === 0
                  ? "bg-[var(--win-soft)] border-[var(--win)]/25"
                  : "border-transparent hover:bg-[var(--surface-2)] hover:border-[var(--border)]"
              }`}
            >
              <div className="flex items-start justify-between gap-2 text-[11px] mb-1">
                <span className="font-semibold leading-snug break-words line-clamp-2 min-w-0">
                  {idx + 1}. {row.act.label}
                </span>
                <span className="tabular shrink-0 flex items-center gap-1 pt-0.5">
                  {isChosen ? (
                    <span title="Action taken in recorded game" className="text-[var(--accent)]">
                      ★
                    </span>
                  ) : null}
                  <strong>{row.pA}%</strong>
                  {delta === 0 ? null : (
                    <span
                      className={
                        delta > 0
                          ? "text-[var(--win)] font-bold text-[10px]"
                          : "text-[var(--loss)] font-bold text-[10px]"
                      }
                    >
                      ({delta > 0 ? `+${delta}` : delta}%)
                    </span>
                  )}
                </span>
              </div>
              <div className="h-[5px] rounded bg-[var(--surface-2)] overflow-hidden">
                <div
                  className="h-full rounded transition-all"
                  style={{
                    width: `${Math.max(2, row.pA)}%`,
                    background: idx === 0 ? "var(--win-fill)" : "var(--accent)",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
      {actions.length > maxShow ? (
        <button
          type="button"
          onClick={(): void => setShowAll((prev) => !prev)}
          className="mt-2 w-full rounded-md border border-dashed border-[var(--border-strong)] py-1 text-[10.5px] font-medium text-[var(--text-2)] hover:text-[var(--text)] hover:border-[var(--accent)] cursor-pointer"
        >
          {showAll ? `Show top ${maxShow} actions ▴` : `Show all ${actions.length} legal actions ▾`}
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
        <span>Hero Value Estimate (WDL)</span>
        <span className="tabular">{props.checkpointName}</span>
      </div>
      <div className="h-[18px] rounded-[5px] overflow-hidden flex text-[10px] font-bold text-white tabular">
        <div
          className="bg-[var(--win-fill)] flex items-center justify-center"
          style={{ width: `${props.winPct}%` }}
        >
          {props.winPct >= 16 ? `${props.winPct}%` : ""}
        </div>
        <div className="bg-zinc-600" style={{ width: `${props.drawPct}%` }} />
        <div
          className="bg-[var(--loss-fill)] flex items-center justify-center"
          style={{ width: `${props.lossPct}%` }}
        >
          {props.lossPct >= 14 ? `${props.lossPct}%` : ""}
        </div>
      </div>
      <div className="flex items-center justify-between tabular text-[10.5px] text-[var(--text-2)]">
        <span>
          Hero {props.winPct}% Win · {props.drawPct}% Draw · {props.lossPct}% Loss
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
      <input
        type="text"
        value={pinNote}
        onChange={(e): void => setPinNote(e.target.value)}
        placeholder="Optional Watchlist note (e.g. missed lethal block?)"
        aria-label="Watchlist note"
        className="h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-[11px]"
      />
      <Button
        variant={props.pinned ? "primary" : "default"}
        className="w-full"
        onClick={(): void => {
          props.onPinPosition(pinNote);
          setPinNote("");
        }}
      >
        <Bookmark className="w-3.5 h-3.5" />
        {props.pinned ? "Pinned to Watchlist" : "Pin position to Watchlist"}
      </Button>
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
  forkWarnings: string[];
  forkError: string | null;
  onAddForkEdit: (edit: ForkEdit) => void;
  onRemoveForkEdit: (idx: number) => void;
  onHoverActionFocus: (ids: number[]) => void;
  onPinPosition: (note: string) => void;
  pinned: boolean;
}): ReactNode {
  const wdl = props.evaluation.primary.wdl;
  const winPct = Math.round(wdl.win * 100);
  const drawPct = Math.round(wdl.draw * 100);
  const lossPct = Math.max(0, 100 - winPct - drawPct);
  const baseWinPct = Math.round(props.baseEvaluation.primary.wdl.win * 100);
  const compareWinPct = Math.round(props.evaluation.compare.wdl.win * 100);
  const wdlDelta = winPct - compareWinPct;

  return (
    <aside
      aria-label="Policy and counterfactual inspector"
      className="w-full lg:w-[328px] shrink-0 border-t lg:border-t-0 lg:border-l border-[var(--border)] bg-[var(--surface)] flex flex-col min-h-0 lg:overflow-y-auto"
    >
      <div className="p-3 border-b border-[var(--border)]">
        <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] mb-1.5">
          <span>Checkpoint Diff</span>
          {props.board.actions_hidden ? (
            <span className="normal-case font-medium text-[var(--text-3)]">Hidden in Hero POV</span>
          ) : (
            <span className="tabular" title="Total Variation and KL(A || B) in nats">
              TV {props.evaluation.diff.tv.toFixed(2)} · KL(A‖B){" "}
              {props.evaluation.diff.kl.toFixed(2)}
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
          forkBusy={props.forkBusy}
          edits={props.forkEdits}
          baseWinPct={baseWinPct}
          winPct={winPct}
          warnings={props.forkWarnings}
          error={props.forkError}
          onAddEdit={props.onAddForkEdit}
          onRemoveEdit={props.onRemoveForkEdit}
        />
      ) : null}

      <LegalActionsSection
        board={props.board}
        evaluation={props.evaluation}
        compact={props.forkMode}
        onHoverFocus={props.onHoverActionFocus}
      />

      <WdlAndWatchlistFooter
        checkpointName={props.evaluation.primary.checkpoint}
        winPct={winPct}
        drawPct={drawPct}
        lossPct={lossPct}
        wdlDelta={wdlDelta}
        showDelta={props.ckptB !== "same" && wdlDelta !== 0}
        pinned={props.pinned}
        onPinPosition={props.onPinPosition}
      />
    </aside>
  );
}

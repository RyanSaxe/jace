/** Non-covering Opponent Belief Editor drawer with card search, Pin/Exclude/Weight controls, unedited mass footer, and inline 422 alert. */

import { RotateCcw, Search, X } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button, SegmentedControl, type SegmentedOption } from "@/components/ui/Primitives";
import type {
  BeliefCardEntry,
  BeliefInferResponse,
  BeliefOverridePayload,
  BeliefProjectResponse,
} from "@/lib/types";
import type { BeliefEditorDrawerProps } from "./useBeliefSearch";

const SOURCE_OPTIONS: ReadonlyArray<SegmentedOption<"model" | "prior">> = [
  { value: "model", label: "Model" },
  { value: "prior", label: "Prior" },
];
const COPY_MULTIPLIERS: ReadonlyArray<number> = [0.5, 1, 2];
const DEFAULT_VISIBLE_CARDS = 60;

function buildCardsFromInfer(belief: BeliefInferResponse): BeliefCardEntry[] {
  if (belief.cards.length > 0) return belief.cards;
  return belief.card_names.map((name, idx) => ({
    name,
    hand_prob: belief.hand_probs[idx] ?? 0,
    deck_copies: belief.deck_copies[idx] ?? 0,
    revealed_count: belief.revealed_counts[idx] ?? 0,
    pinned: 0,
    excluded: false,
    multiplier: 1,
  }));
}

function mutateOverride(
  prev: BeliefOverridePayload,
  cardName: string,
  edit:
    | { kind: "pin"; count: number }
    | { kind: "exclude"; current: boolean }
    | { kind: "mult"; mult: number },
): BeliefOverridePayload {
  const pinned = { ...prev.pinned_in_hand };
  const mults = { ...prev.copy_multipliers };
  let excluded = prev.excluded_cards;
  if (edit.kind === "pin") {
    excluded = excluded.filter((c) => c !== cardName);
    if (edit.count <= 0) delete pinned[cardName];
    else pinned[cardName] = edit.count;
  } else if (edit.kind === "exclude") {
    delete pinned[cardName];
    excluded = edit.current ? excluded.filter((c) => c !== cardName) : [...excluded, cardName];
  } else if (edit.mult === 1) {
    delete mults[cardName];
  } else {
    mults[cardName] = edit.mult;
  }
  return { pinned_in_hand: pinned, excluded_cards: excluded, copy_multipliers: mults };
}

function BeliefCardRow(props: {
  card: BeliefCardEntry;
  busy: boolean;
  onChangePin: (nextCount: number) => void;
  onToggleExclude: () => void;
  onChangeMultiplier: (mult: number) => void;
}): ReactNode {
  const { card } = props;
  const edited = card.pinned > 0 || card.excluded || card.multiplier !== 1;
  const rowBorder = edited
    ? "border-[var(--fork)]/40 bg-[var(--fork-soft)]/20"
    : "border-[var(--border)]/70";
  return (
    <div className={`rounded-md border px-2 py-1.5 flex flex-col gap-1 ${rowBorder}`}>
      <div className="flex items-center justify-between gap-2 text-[11px]">
        <span
          className={`font-medium truncate ${card.excluded ? "line-through text-[var(--text-3)]" : ""}`}
        >
          {card.name}
        </span>
        <span className="tabular text-[10.5px] text-[var(--text-2)] shrink-0">
          {(card.hand_prob * 100).toFixed(1)}% · {card.deck_copies.toFixed(2)}d
          {card.revealed_count > 0 ? ` · ${card.revealed_count}r` : ""}
        </span>
      </div>
      <div className="flex items-center justify-between gap-1.5 text-[10px]">
        <div className="inline-flex items-center gap-1">
          <span className="text-[var(--text-3)]">Pin</span>
          <button
            type="button"
            disabled={props.busy || card.pinned <= 0}
            aria-label={`Decrease pin for ${card.name}`}
            onClick={(): void => props.onChangePin(Math.max(0, card.pinned - 1))}
            className="w-5 h-5 rounded border border-[var(--border)] bg-[var(--surface)] text-[var(--text)] disabled:opacity-35 cursor-pointer"
          >
            −
          </button>
          <span className="tabular w-4 text-center font-semibold">{card.pinned}</span>
          <button
            type="button"
            disabled={props.busy}
            aria-label={`Increase pin for ${card.name}`}
            onClick={(): void => props.onChangePin(card.pinned + 1)}
            className="w-5 h-5 rounded border border-[var(--border)] bg-[var(--surface)] text-[var(--text)] disabled:opacity-35 cursor-pointer"
          >
            +
          </button>
        </div>
        <button
          type="button"
          disabled={props.busy}
          aria-pressed={card.excluded}
          onClick={props.onToggleExclude}
          className={`px-1.5 py-0.5 rounded border cursor-pointer ${
            card.excluded
              ? "border-[var(--loss)] bg-[var(--loss-soft)] text-[var(--loss)] font-semibold"
              : "border-[var(--border)] bg-[var(--surface)] text-[var(--text-2)]"
          }`}
        >
          Exclude
        </button>
        <div className="inline-flex items-center rounded border border-[var(--border)] bg-[var(--surface-2)] p-0.5">
          {COPY_MULTIPLIERS.map((mult) => (
            <button
              key={mult}
              type="button"
              disabled={props.busy || card.excluded}
              onClick={(): void => props.onChangeMultiplier(mult)}
              className={`px-1 py-0.5 rounded tabular cursor-pointer disabled:opacity-40 ${
                card.multiplier === mult
                  ? "bg-[var(--surface)] text-[var(--text)] font-semibold"
                  : "text-[var(--text-3)] hover:text-[var(--text)]"
              }`}
            >
              {mult}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

interface DrawerMetrics {
  cards: BeliefCardEntry[];
  oppHandSize: number;
  oppLibrarySize: number;
  pinnedTotal: number;
  uneditedCount: number;
  uneditedMass: number;
}

function resolveDrawerMetrics(
  projection: BeliefProjectResponse | null,
  belief: BeliefInferResponse | null,
): DrawerMetrics {
  const src = projection ?? belief;
  if (!src) {
    return {
      cards: [],
      oppHandSize: 0,
      oppLibrarySize: 0,
      pinnedTotal: 0,
      uneditedCount: 0,
      uneditedMass: 1,
    };
  }
  let cards: BeliefCardEntry[] = [];
  if (projection) cards = projection.cards;
  else if (belief) cards = buildCardsFromInfer(belief);
  return {
    cards,
    oppHandSize: src.opp_hand_size,
    oppLibrarySize: src.opp_library_size,
    pinnedTotal: src.pinned_total,
    uneditedCount: src.unedited_count,
    uneditedMass: src.unedited_mass,
  };
}

function BeliefDrawerHeader(
  props: BeliefEditorDrawerProps & {
    metrics: DrawerMetrics;
    query: string;
    onChangeQuery: (q: string) => void;
  },
): ReactNode {
  const { metrics } = props;
  const hasBeliefNet = props.belief?.has_belief_net ?? false;
  const effectiveSource = !hasBeliefNet || props.priorOnly ? "prior" : "model";
  return (
    <div className="px-3 py-2 border-b border-[var(--border)] flex flex-col gap-1.5 shrink-0">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-2)]">
          Opponent Belief
        </span>
        <div className="flex items-center gap-1.5">
          {hasBeliefNet ? (
            <SegmentedControl
              options={SOURCE_OPTIONS}
              value={effectiveSource}
              onChange={(val): void => props.onChangePriorOnly(val === "prior")}
            />
          ) : (
            <span
              title="No trained belief_net.pt loaded for this format; using format prior"
              className="rounded border border-[var(--border)] bg-[var(--surface-2)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--text-3)]"
            >
              Prior only
            </span>
          )}
          <button
            type="button"
            onClick={props.onReset}
            disabled={props.busy}
            className="inline-flex items-center gap-0.5 rounded border border-[var(--border)] bg-[var(--surface)] px-1.5 py-1 text-[10px] font-medium text-[var(--text-2)] hover:text-[var(--text)] cursor-pointer disabled:opacity-40"
          >
            <RotateCcw className="w-2.5 h-2.5" />
            Reset
          </button>
          <button
            type="button"
            onClick={props.onClose}
            aria-label="Close belief editor"
            className="p-1 rounded text-[var(--text-2)] hover:bg-[var(--surface-2)] cursor-pointer"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
      <div className="text-[10.5px] tabular text-[var(--text-2)]">
        Opp Hand: <strong>{metrics.oppHandSize}</strong> cards ({metrics.pinnedTotal} pinned) ·
        Library: <strong>{metrics.oppLibrarySize}</strong>
      </div>
      <label className="relative flex items-center">
        <Search className="w-3.5 h-3.5 absolute left-2 text-[var(--text-3)] pointer-events-none" />
        <input
          type="search"
          value={props.query}
          onChange={(e): void => props.onChangeQuery(e.target.value)}
          placeholder={`Search all ${metrics.cards.length} format cards…`}
          aria-label="Search format cards"
          className="w-full h-7 rounded-md border border-[var(--border)] bg-[var(--surface-2)] pl-7 pr-2 text-[11px] text-[var(--text)] placeholder:text-[var(--text-3)]"
        />
      </label>
      {props.error ? (
        <div
          role="alert"
          className="rounded border border-[var(--loss)]/40 bg-[var(--loss-soft)] px-2 py-1 text-[10.5px] text-[var(--loss)]"
        >
          {props.error}
        </div>
      ) : null}
    </div>
  );
}

function BeliefDrawerCardList(props: {
  cards: BeliefCardEntry[];
  query: string;
  showAll: boolean;
  busy: boolean;
  override: BeliefOverridePayload;
  onToggleShowAll: () => void;
  onUpdateOverride: (next: BeliefOverridePayload) => void;
}): ReactNode {
  const trimmed = props.query.trim().toLowerCase();
  const filtered = trimmed
    ? props.cards.filter((c) => c.name.toLowerCase().includes(trimmed))
    : props.cards;
  const visible =
    props.showAll || trimmed.length > 0 ? filtered : filtered.slice(0, DEFAULT_VISIBLE_CARDS);

  return (
    <div className="flex-1 min-h-0 overflow-y-auto px-3 py-2 flex flex-col gap-1.5">
      {props.cards.length === 0 ? (
        <div className="py-6 text-center text-[11px] text-[var(--text-3)]">
          {props.busy ? "Inferring format belief distribution…" : "No belief distribution loaded."}
        </div>
      ) : (
        visible.map((card) => (
          <BeliefCardRow
            key={card.name}
            card={card}
            busy={props.busy}
            onChangePin={(count): void =>
              props.onUpdateOverride(
                mutateOverride(props.override, card.name, { kind: "pin", count }),
              )
            }
            onToggleExclude={(): void =>
              props.onUpdateOverride(
                mutateOverride(props.override, card.name, {
                  kind: "exclude",
                  current: card.excluded,
                }),
              )
            }
            onChangeMultiplier={(mult): void =>
              props.onUpdateOverride(
                mutateOverride(props.override, card.name, { kind: "mult", mult }),
              )
            }
          />
        ))
      )}
      {trimmed.length === 0 && filtered.length > DEFAULT_VISIBLE_CARDS ? (
        <button
          type="button"
          onClick={props.onToggleShowAll}
          className="w-full rounded-md border border-dashed border-[var(--border-strong)] py-1 text-[10.5px] font-medium text-[var(--text-2)] hover:text-[var(--text)] cursor-pointer shrink-0"
        >
          {props.showAll
            ? `Show top ${DEFAULT_VISIBLE_CARDS} cards ▴`
            : `Show all ${filtered.length} cards ▾`}
        </button>
      ) : null}
    </div>
  );
}

export function BeliefEditorDrawer(props: BeliefEditorDrawerProps): ReactNode {
  const [query, setQuery] = useState("");
  const [showAll, setShowAll] = useState(false);
  if (!props.open) return null;
  const metrics = resolveDrawerMetrics(props.projection, props.belief);
  return (
    <aside
      aria-label="Opponent belief editor"
      className="fixed inset-x-0 bottom-0 z-50 h-[58dvh] max-h-[480px] rounded-t-[14px] border-t border-[var(--border-strong)] shadow-[var(--shadow-lg)] lg:static lg:inset-auto lg:z-auto lg:h-full lg:max-h-none lg:w-[316px] lg:rounded-none lg:border-t-0 lg:border-l lg:border-[var(--border)] lg:shadow-none shrink-0 min-h-0 bg-[var(--surface)] flex flex-col overflow-hidden"
    >
      <BeliefDrawerHeader {...props} metrics={metrics} query={query} onChangeQuery={setQuery} />
      <BeliefDrawerCardList
        cards={metrics.cards}
        query={query}
        showAll={showAll}
        busy={props.busy}
        override={props.override}
        onToggleShowAll={(): void => setShowAll((prev) => !prev)}
        onUpdateOverride={props.onUpdateOverride}
      />
      <div className="px-3 py-2 border-t border-[var(--border)] bg-[var(--surface)] flex flex-col gap-1.5 shrink-0">
        <div className="text-[10px] tabular text-[var(--text-2)]">
          Unedited cards kept in distribution: <strong>{metrics.uneditedCount}</strong> (
          {(metrics.uneditedMass * 100).toFixed(0)}% hand mass)
        </div>
        <Button
          size="sm"
          variant="fork"
          disabled={props.busy || Boolean(props.error)}
          onClick={props.onApplyAndRescore}
          className="w-full py-1.5 text-[11px]"
        >
          Apply &amp; Re-score
        </Button>
      </div>
    </aside>
  );
}

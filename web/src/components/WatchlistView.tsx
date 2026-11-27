/** Saved Watchlist view: searchable, tag-filterable bookmarks with inline title/note editing, puzzle promotion, and Tabletop jump. */

import {
  ArrowRight,
  Bookmark,
  Check,
  Pencil,
  Search,
  Sparkles,
  Tag,
  Trash2,
  X,
} from "lucide-react";
import { type ReactNode, useState } from "react";
import { Badge, Button } from "@/components/ui/Primitives";
import { useBenchmarkFormats } from "@/hooks/useBenchmarksData";
import { cardImageUrl, promoteWatchlistToPuzzle } from "@/lib/api";
import type { GameTarget, PromotePuzzleResult, WatchlistItem } from "@/lib/types";

export interface WatchlistViewProps {
  items: WatchlistItem[];
  loading: boolean;
  onOpenTabletop: (target: GameTarget) => void;
  onRemove: (itemId: string) => void;
  onUpdate: (
    itemId: string,
    patch: { label?: string; note?: string; tag?: string },
  ) => Promise<void>;
  onOpenPuzzles?: (format: string) => void;
}

function humanizeTitle(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return "Pinned Position";
  return trimmed.replace(/\b([A-Z]{2,}(?:_[A-Z0-9]+)+|[A-Z]{5,})\b/g, (token) =>
    token
      .toLowerCase()
      .split("_")
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" "),
  );
}

function matchesSearch(item: WatchlistItem, query: string): boolean {
  if (!query) return true;
  const needle = query.toLowerCase();
  const fields = [
    item.label,
    humanizeTitle(item.label),
    item.note,
    item.tag ?? "",
    item.summary ?? "",
    item.run_id,
    `seed ${item.seed}`,
    `#${item.seed}`,
    `u${item.iteration}`,
    `t${item.turn}`,
    ...(item.key_cards ?? []),
  ];
  return fields.some((field) => field.toLowerCase().includes(needle));
}

function collectDistinctTags(items: WatchlistItem[]): Array<{ tag: string; count: number }> {
  const counts = new Map<string, number>();
  for (const item of items) {
    const tag = item.tag?.trim() ?? "";
    if (tag) counts.set(tag, (counts.get(tag) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([tag, count]) => ({ tag, count }))
    .toSorted((a, b) => a.tag.localeCompare(b.tag));
}

function BookmarkCardThumb(props: { name: string }): ReactNode {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div
        title={props.name}
        className="w-[46px] aspect-[5/7] shrink-0 rounded border border-[var(--border)] bg-[var(--surface-2)] p-1 flex items-center justify-center text-center text-[8.5px] leading-tight text-[var(--text-2)]"
      >
        <span className="line-clamp-3">{props.name}</span>
      </div>
    );
  }
  return (
    <div
      title={props.name}
      className="w-[46px] aspect-[5/7] shrink-0 rounded overflow-hidden border border-[var(--border)] bg-[#18181b]"
    >
      <img
        src={cardImageUrl(props.name)}
        alt={props.name}
        loading="lazy"
        onError={(): void => setFailed(true)}
        className="w-full h-full object-cover block"
      />
    </div>
  );
}

function BookmarkCardEditor(props: {
  label: string;
  note: string;
  tag: string;
  saving: boolean;
  error: string | null;
  onChangeLabel: (val: string) => void;
  onChangeNote: (val: string) => void;
  onChangeTag: (val: string) => void;
  onSave: () => void;
  onCancel: () => void;
}): ReactNode {
  return (
    <div className="flex flex-col gap-2">
      <label className="flex flex-col gap-1 text-[10.5px] text-[var(--text-3)]">
        <span>Title</span>
        <input
          type="text"
          value={props.label}
          onChange={(e): void => props.onChangeLabel(e.target.value)}
          onKeyDown={(e): void => {
            if (e.key === "Enter") props.onSave();
            if (e.key === "Escape") props.onCancel();
          }}
          placeholder="Tactical position title"
          className="h-8 rounded-md border border-[var(--border-strong)] bg-[var(--bg)] px-2 text-xs font-medium text-[var(--text)]"
        />
      </label>

      <label className="flex flex-col gap-1 text-[10.5px] text-[var(--text-3)]">
        <span>Note</span>
        <textarea
          rows={2}
          value={props.note}
          onChange={(e): void => props.onChangeNote(e.target.value)}
          onKeyDown={(e): void => {
            if (e.key === "Escape") props.onCancel();
          }}
          placeholder="Why does this board matter?"
          className="rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 py-1.5 text-xs text-[var(--text)] resize-y"
        />
      </label>

      <label className="flex flex-col gap-1 text-[10.5px] text-[var(--text-3)]">
        <span>Tag (optional)</span>
        <input
          type="text"
          value={props.tag}
          onChange={(e): void => props.onChangeTag(e.target.value)}
          onKeyDown={(e): void => {
            if (e.key === "Enter") props.onSave();
            if (e.key === "Escape") props.onCancel();
          }}
          placeholder="e.g. combat, lethal, mulligan"
          className="h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-[11.5px] text-[var(--text)]"
        />
      </label>

      {props.error ? <div className="text-[11px] text-[var(--loss)]">{props.error}</div> : null}

      <div className="flex items-center gap-1.5 pt-0.5">
        <Button variant="primary" disabled={props.saving} onClick={props.onSave}>
          <Check className="w-3.5 h-3.5" />
          {props.saving ? "Saving…" : "Save"}
        </Button>
        <Button variant="ghost" disabled={props.saving} onClick={props.onCancel}>
          <X className="w-3.5 h-3.5" />
          Cancel
        </Button>
      </div>
    </div>
  );
}

function BookmarkCardReadBody(props: {
  item: WatchlistItem;
  displayTitle: string;
  onStartEdit: () => void;
}): ReactNode {
  const { item } = props;
  const thumbs = (item.key_cards ?? []).slice(0, 3);
  return (
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0 flex-1 flex flex-col gap-1">
        <button
          type="button"
          onClick={props.onStartEdit}
          title="Click to edit title and note"
          className="group text-left flex items-center gap-1.5 cursor-pointer bg-transparent border-0 p-0"
        >
          <span className="text-sm font-semibold text-[var(--text)] leading-snug group-hover:text-[var(--accent)] transition-colors">
            {props.displayTitle}
          </span>
          <Pencil className="w-3 h-3 text-[var(--text-3)] opacity-0 group-hover:opacity-100 shrink-0 transition-opacity" />
        </button>

        {item.summary ? (
          <div className="text-[11.5px] text-[var(--text-2)] leading-snug">{item.summary}</div>
        ) : null}

        <div className="text-[10.5px] text-[var(--text-3)] truncate" title={item.run_id}>
          {item.run_id}
        </div>

        <button
          type="button"
          onClick={props.onStartEdit}
          title="Click to edit note"
          className="mt-1 w-full text-left rounded-md border border-[var(--border)] bg-[var(--bg)] hover:border-[var(--border-strong)] px-2.5 py-1.5 text-xs transition-colors cursor-pointer"
        >
          {item.note ? (
            <span className="text-[var(--text-2)] whitespace-pre-wrap break-words">
              {item.note}
            </span>
          ) : (
            <span className="text-[var(--text-3)] italic">Add a note…</span>
          )}
        </button>
      </div>

      {thumbs.length > 0 ? (
        <div className="flex items-center gap-1 shrink-0 pt-0.5">
          {thumbs.map((cardName) => (
            <BookmarkCardThumb key={cardName} name={cardName} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function PromoteStatusBanner(props: {
  result: PromotePuzzleResult | null;
  error: string | null;
  format: string;
  onOpenPuzzles?: (format: string) => void;
}): ReactNode {
  if (props.error) {
    return <div className="text-[11px] text-[var(--loss)] leading-snug">{props.error}</div>;
  }
  if (!props.result) return null;
  if (props.result.admitted) {
    const marginPct = (props.result.puzzle.margin * 100).toFixed(1);
    return (
      <div className="flex items-center justify-between gap-2 text-[11px] text-[var(--win)]">
        <span className="tabular font-medium">
          Promoted to {props.format} (+{marginPct}%)
        </span>
        {props.onOpenPuzzles ? (
          <button
            type="button"
            onClick={(): void => props.onOpenPuzzles?.(props.format)}
            className="text-[11px] font-semibold text-[var(--accent)] hover:underline cursor-pointer bg-transparent border-0 p-0"
          >
            View Puzzles →
          </button>
        ) : null}
      </div>
    );
  }
  return (
    <div className="tabular text-[11px] text-[var(--loss)] leading-snug">{props.result.detail}</div>
  );
}

function WatchlistCard(props: {
  item: WatchlistItem;
  promoteFormat: string;
  onSelectTag: (tag: string) => void;
  onOpenTabletop: (target: GameTarget) => void;
  onRemove: (itemId: string) => void;
  onUpdate: (
    itemId: string,
    patch: { label?: string; note?: string; tag?: string },
  ) => Promise<void>;
  onOpenPuzzles?: (format: string) => void;
}): ReactNode {
  const { item } = props;
  const displayTitle = humanizeTitle(item.label);
  const [editing, setEditing] = useState(false);
  const [draftLabel, setDraftLabel] = useState(displayTitle);
  const [draftNote, setDraftNote] = useState(item.note);
  const [draftTag, setDraftTag] = useState(item.tag ?? "");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [promoting, setPromoting] = useState(false);
  const [promoteResult, setPromoteResult] = useState<PromotePuzzleResult | null>(null);
  const [promoteError, setPromoteError] = useState<string | null>(null);

  const openEditor = (): void => {
    setDraftLabel(displayTitle);
    setDraftNote(item.note);
    setDraftTag(item.tag ?? "");
    setSaveError(null);
    setEditing(true);
  };

  const handlePromote = (): void => {
    const targetFormat = props.promoteFormat.trim() || "default";
    setPromoting(true);
    setPromoteError(null);
    setPromoteResult(null);
    void promoteWatchlistToPuzzle({ itemId: item.id, format: targetFormat })
      .then((res) => {
        setPromoting(false);
        setPromoteResult(res);
      })
      .catch((err: unknown) => {
        setPromoting(false);
        setPromoteError(err instanceof Error ? err.message : "Failed to promote puzzle");
      });
  };

  const itemTag = item.tag?.trim() ?? "";

  return (
    <article className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3.5 flex flex-col justify-between gap-3">
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center justify-between gap-1.5">
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone="accent">
              Turn {item.turn} · Step #{item.step}
            </Badge>
            {itemTag ? (
              <button
                type="button"
                onClick={(): void => props.onSelectTag(itemTag)}
                title={`Filter by tag "${itemTag}"`}
                className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] font-semibold border border-[var(--border)] bg-[var(--surface-2)] text-[var(--text-2)] hover:border-[var(--accent)] hover:text-[var(--accent)] cursor-pointer transition-colors"
              >
                <Tag className="w-2.5 h-2.5" />
                {itemTag}
              </button>
            ) : null}
          </div>
          <span className="tabular text-[11px] text-[var(--text-3)]">
            Seed #{item.seed} · u{item.iteration}
          </span>
        </div>

        {editing ? (
          <BookmarkCardEditor
            label={draftLabel}
            note={draftNote}
            tag={draftTag}
            saving={saving}
            error={saveError}
            onChangeLabel={setDraftLabel}
            onChangeNote={setDraftNote}
            onChangeTag={setDraftTag}
            onSave={(): void => {
              setSaving(true);
              setSaveError(null);
              void props
                .onUpdate(item.id, {
                  label: draftLabel.trim() || displayTitle,
                  note: draftNote.trim(),
                  tag: draftTag.trim(),
                })
                .then(() => {
                  setSaving(false);
                  setEditing(false);
                })
                .catch((err: unknown) => {
                  setSaving(false);
                  setSaveError(err instanceof Error ? err.message : "Failed to save bookmark");
                });
            }}
            onCancel={(): void => setEditing(false)}
          />
        ) : (
          <BookmarkCardReadBody item={item} displayTitle={displayTitle} onStartEdit={openEditor} />
        )}

        <PromoteStatusBanner
          result={promoteResult}
          error={promoteError}
          format={props.promoteFormat.trim() || "default"}
          onOpenPuzzles={props.onOpenPuzzles}
        />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-[var(--border)]">
        <div className="flex flex-wrap items-center gap-1.5">
          <Button
            variant="primary"
            onClick={(): void =>
              props.onOpenTabletop({
                run_id: item.run_id,
                iteration: item.iteration,
                game_idx: item.game_idx,
                step: item.step,
                seed: item.seed,
              })
            }
          >
            Open in Tabletop
            <ArrowRight className="w-3.5 h-3.5" />
          </Button>
          <Button
            variant="default"
            disabled={promoting}
            title="Verify with N=64 paired playouts and add to format puzzles"
            onClick={handlePromote}
          >
            <Sparkles className="w-3.5 h-3.5" />
            {promoting ? "Verifying…" : "Promote to Puzzle"}
          </Button>
        </div>

        <div className="flex items-center gap-1">
          {editing ? null : (
            <Button variant="ghost" title="Edit bookmark" onClick={openEditor}>
              <Pencil className="w-3.5 h-3.5" />
            </Button>
          )}
          <Button
            variant="ghost"
            title="Remove from watchlist"
            onClick={(): void => props.onRemove(item.id)}
          >
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>
    </article>
  );
}

function WatchlistBody(props: {
  items: WatchlistItem[];
  filtered: WatchlistItem[];
  loading: boolean;
  promoteFormat: string;
  hasActiveFilters: boolean;
  onClearFilters: () => void;
  onSelectTag: (tag: string) => void;
  onOpenTabletop: (target: GameTarget) => void;
  onRemove: (itemId: string) => void;
  onUpdate: (
    itemId: string,
    patch: { label?: string; note?: string; tag?: string },
  ) => Promise<void>;
  onOpenPuzzles?: (format: string) => void;
}): ReactNode {
  if (props.loading) {
    return <div className="text-xs text-[var(--text-3)]">Loading pinned positions…</div>;
  }
  if (props.items.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border-strong)] bg-[var(--surface)] p-8 text-center text-xs text-[var(--text-2)] max-w-xl mx-auto my-8">
        No positions pinned yet. Open any game in <strong>Tabletop</strong> and click{" "}
        <strong>Pin position to Watchlist</strong> to bookmark tactical boards here.
      </div>
    );
  }
  if (props.filtered.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border-strong)] bg-[var(--surface)] p-8 text-center flex flex-col items-center gap-2.5 max-w-md mx-auto my-8">
        <div className="text-xs text-[var(--text-2)]">
          No pinned bookmarks match your current search or tag filter.
        </div>
        {props.hasActiveFilters ? (
          <Button variant="default" onClick={props.onClearFilters}>
            Clear filters
          </Button>
        ) : null}
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      {props.filtered.map((item) => (
        <WatchlistCard
          key={item.id}
          item={item}
          promoteFormat={props.promoteFormat}
          onSelectTag={props.onSelectTag}
          onOpenTabletop={props.onOpenTabletop}
          onRemove={props.onRemove}
          onUpdate={props.onUpdate}
          onOpenPuzzles={props.onOpenPuzzles}
        />
      ))}
    </div>
  );
}

export function WatchlistView(props: WatchlistViewProps): ReactNode {
  const [search, setSearch] = useState("");
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [customFormat, setCustomFormat] = useState("");
  const { formats, defaultFormat } = useBenchmarkFormats();
  const promoteFormat = customFormat || defaultFormat || "default";

  const distinctTags = collectDistinctTags(props.items);
  const effectiveTag =
    selectedTag && distinctTags.some((entry) => entry.tag === selectedTag) ? selectedTag : null;

  const filteredItems = props.items.filter((item) => {
    if (effectiveTag && (item.tag?.trim() ?? "") !== effectiveTag) return false;
    return matchesSearch(item, search.trim());
  });

  const hasActiveFilters = search.trim() !== "" || effectiveTag !== null;

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="min-h-[44px] px-3.5 py-1.5 border-b border-[var(--border)] bg-[var(--surface)] flex flex-wrap items-center gap-2.5 shrink-0">
        <div className="flex items-center gap-2">
          <Bookmark className="w-4 h-4 text-[var(--accent)] shrink-0" />
          <h2 className="text-xs font-semibold tracking-tight m-0">Watchlist</h2>
          <Badge>{props.items.length} pinned</Badge>
        </div>

        <div className="relative flex-1 min-w-[180px] max-w-[320px]">
          <Search className="w-3.5 h-3.5 text-[var(--text-3)] absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="text"
            value={search}
            onChange={(e): void => setSearch(e.target.value)}
            placeholder="Search title, note, run, seed, cards…"
            aria-label="Search watchlist"
            className="h-7 w-full rounded-md border border-[var(--border)] bg-[var(--bg)] pl-8 pr-7 text-[11.5px] text-[var(--text)] placeholder:text-[var(--text-3)]"
          />
          {search ? (
            <button
              type="button"
              onClick={(): void => setSearch("")}
              aria-label="Clear search"
              className="absolute right-1.5 top-1/2 -translate-y-1/2 p-0.5 text-[var(--text-3)] hover:text-[var(--text)] cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          ) : null}
        </div>

        {distinctTags.length > 0 ? (
          <div className="flex flex-wrap items-center gap-1">
            <button
              type="button"
              onClick={(): void => setSelectedTag(null)}
              className={`h-6 px-2 rounded-md text-[11px] font-medium border cursor-pointer transition-colors ${
                effectiveTag === null
                  ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)] font-semibold"
                  : "border-[var(--border)] bg-[var(--bg)] text-[var(--text-2)] hover:text-[var(--text)]"
              }`}
            >
              All
            </button>
            {distinctTags.map(({ tag, count }) => {
              const active = effectiveTag === tag;
              return (
                <button
                  key={tag}
                  type="button"
                  onClick={(): void => setSelectedTag((prev) => (prev === tag ? null : tag))}
                  className={`h-6 px-2 rounded-md text-[11px] font-medium border cursor-pointer transition-colors inline-flex items-center gap-1 ${
                    active
                      ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)] font-semibold"
                      : "border-[var(--border)] bg-[var(--bg)] text-[var(--text-2)] hover:text-[var(--text)]"
                  }`}
                >
                  <span>{tag}</span>
                  <span className="tabular text-[10px] opacity-75">{count}</span>
                </button>
              );
            })}
          </div>
        ) : null}
        {formats.length > 0 ? (
          <label className="flex items-center gap-1.5 text-[11px] text-[var(--text-3)] ml-auto">
            <span>Puzzle format</span>
            <select
              aria-label="Puzzle format"
              value={promoteFormat}
              onChange={(e): void => setCustomFormat(e.target.value)}
              className="h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-[11.5px] font-medium text-[var(--text)]"
            >
              {formats.map((fmt) => (
                <option key={fmt} value={fmt}>
                  {fmt}
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <WatchlistBody
          items={props.items}
          filtered={filteredItems}
          loading={props.loading}
          promoteFormat={promoteFormat}
          hasActiveFilters={hasActiveFilters}
          onClearFilters={(): void => {
            setSearch("");
            setSelectedTag(null);
          }}
          onSelectTag={(tag): void => setSelectedTag((prev) => (prev === tag ? null : tag))}
          onOpenTabletop={props.onOpenTabletop}
          onRemove={props.onRemove}
          onUpdate={props.onUpdate}
          onOpenPuzzles={props.onOpenPuzzles}
        />
      </div>
    </div>
  );
}

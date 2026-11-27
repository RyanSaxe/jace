/** Saved Watchlist view: searchable, tag-filterable bookmarks with inline title/note editing, puzzle promotion, and Tabletop jump. */

import { Bookmark, Search, X } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Badge, Button } from "@/components/ui/Primitives";
import { useBenchmarkFormats } from "@/hooks/useBenchmarksData";
import type { GameTarget, WatchlistItem } from "@/lib/types";
import { WatchlistCard, humanizeTitle } from "./WatchlistCard";

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

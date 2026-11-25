/** Saved Watchlist view listing pinned Magic positions with direct jump back into Tabletop. */

import { ArrowRight, Bookmark, Trash2 } from "lucide-react";
import type { ReactNode } from "react";
import { Badge, Button } from "@/components/ui/Primitives";
import type { GameTarget, WatchlistItem } from "@/lib/types";

function WatchlistContent(props: {
  items: WatchlistItem[];
  loading: boolean;
  onOpenTabletop: (target: GameTarget) => void;
  onRemove: (itemId: string) => void;
}): ReactNode {
  if (props.loading) {
    return <div className="text-xs text-[var(--text-3)]">Loading pinned positions…</div>;
  }
  if (props.items.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border-strong)] bg-[var(--surface)] p-8 text-center text-xs text-[var(--text-2)]">
        No positions pinned yet. Open any game in the Tabletop and click{" "}
        <strong>Pin position to Watchlist</strong> to bookmark tactical moments here.
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      {props.items.map((item) => (
        <div
          key={item.id}
          className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3.5 flex flex-col justify-between gap-3"
        >
          <div>
            <div className="flex items-center justify-between gap-2 mb-1">
              <Badge tone="accent">
                Turn {item.turn} · Step #{item.step}
              </Badge>
              <span className="tabular text-[11px] text-[var(--text-3)]">
                Seed #{item.seed} · u{item.iteration}
              </span>
            </div>
            <div className="text-sm font-semibold">{item.label}</div>
            <div className="text-[11px] text-[var(--text-2)] truncate mt-0.5">{item.run_id}</div>
            {item.note ? <p className="text-xs text-[var(--text-2)] mt-1.5">{item.note}</p> : null}
          </div>
          <div className="flex items-center justify-between gap-2">
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
              variant="ghost"
              title="Remove from watchlist"
              onClick={(): void => props.onRemove(item.id)}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}

export function WatchlistView(props: {
  items: WatchlistItem[];
  loading: boolean;
  onOpenTabletop: (target: GameTarget) => void;
  onRemove: (itemId: string) => void;
}): ReactNode {
  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[var(--border)] bg-[var(--surface)]">
        <div className="flex items-center gap-2">
          <Bookmark className="w-4 h-4 text-[var(--accent)]" />
          <h2 className="text-sm font-semibold tracking-tight">Watchlist Positions</h2>
          <Badge>{props.items.length} pinned</Badge>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        <WatchlistContent
          items={props.items}
          loading={props.loading}
          onOpenTabletop={props.onOpenTabletop}
          onRemove={props.onRemove}
        />
      </div>
    </div>
  );
}

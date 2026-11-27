/** Tabletop side zone flank tiles (library/graveyard/exile), zone drawer sheet, and stack overlay popover. */

import { X } from "lucide-react";
import type { ReactNode } from "react";
import { cardImageUrl } from "@/lib/api";
import type { PlayerState, PovMode, StackItem } from "@/lib/types";
import { keyByOccurrence } from "./tabletopHelpers";
import type { DrawerTarget, HoverCardInfo } from "./tabletopTypes";

export function ZoneFlankView(props: {
  side: "hero" | "opp";
  player: PlayerState;
  onOpenDrawer: (target: DrawerTarget) => void;
}): ReactNode {
  const tiles: Array<{ zone: DrawerTarget["zone"]; label: string; count: number; thumb?: string }> =
    [
      { zone: "library", label: "Lib", count: props.player.zone_counts.library },
      {
        zone: "graveyard",
        label: "GY",
        count: props.player.zone_counts.graveyard,
        thumb: props.player.graveyard[0],
      },
      {
        zone: "exile",
        label: "Exile",
        count: props.player.zone_counts.exile,
        thumb: props.player.exile[0],
      },
    ];
  return (
    <div className="flex flex-col justify-center gap-1.5 w-[64px] sm:w-[70px] shrink-0">
      {tiles.map((t) => (
        <button
          key={t.zone}
          type="button"
          onClick={(): void => props.onOpenDrawer({ side: props.side, zone: t.zone })}
          className="flex items-center justify-between gap-1 rounded-[7px] border border-[var(--border)] bg-[var(--surface)]/90 px-1.5 py-1 text-left cursor-pointer hover:border-[var(--accent)]"
          style={{ opacity: t.count === 0 ? 0.55 : 1 }}
        >
          <div>
            <div className="text-[8.5px] font-semibold uppercase text-[var(--text-3)]">
              {t.label}
            </div>
            <div className="tabular text-[12px] font-bold leading-tight">{t.count}</div>
          </div>
          {t.thumb ? (
            <img
              src={cardImageUrl(t.thumb)}
              alt={t.thumb}
              className="w-[17px] h-[24px] rounded-[2px] object-cover border border-[var(--border-strong)] shrink-0"
            />
          ) : null}
        </button>
      ))}
    </div>
  );
}

function ZoneDrawerList(props: {
  zone: DrawerTarget["zone"];
  player: PlayerState;
  onHover: (info: HoverCardInfo | null) => void;
}): ReactNode {
  if (props.zone === "library" && props.player.library_hidden) {
    return (
      <div className="rounded-md border border-dashed border-[var(--border-strong)] bg-[var(--surface-2)] p-3 text-[11px] text-[var(--text-2)]">
        <strong>Hidden in Hero POV.</strong> Switch to <strong>Omniscient</strong> to inspect.
      </div>
    );
  }
  const rows =
    props.zone === "library"
      ? props.player.library_summary.map((r) => ({
          key: r.name,
          prefix: `×${r.count}`,
          name: r.name,
          count: r.count,
        }))
      : keyByOccurrence(
          props.zone === "graveyard" ? props.player.graveyard : props.player.exile,
        ).map((e) => ({ key: e.key, prefix: `#${e.order + 1}`, name: e.name, count: undefined }));
  if (rows.length === 0) {
    return <div className="py-6 text-center text-xs text-[var(--text-3)]">Zone is empty.</div>;
  }
  return rows.map((row) => (
    <div
      key={row.key}
      onMouseEnter={(e): void =>
        props.onHover({ name: row.name, count: row.count, anchorX: e.clientX, anchorY: e.clientY })
      }
      onMouseLeave={(): void => props.onHover(null)}
      className="flex items-center gap-2 rounded-md border border-[var(--border)] bg-[var(--bg)] p-1.5 cursor-pointer hover:border-[var(--accent)]"
    >
      <span className="tabular text-[10px] text-[var(--text-3)] w-6 shrink-0">{row.prefix}</span>
      <img
        src={cardImageUrl(row.name)}
        alt={row.name}
        className="w-[24px] h-[34px] rounded-[2px] object-cover shrink-0"
      />
      <span className="text-[11.5px] font-medium truncate">{row.name}</span>
    </div>
  ));
}

export function ZoneDrawerSheet(props: {
  target: DrawerTarget | null;
  hero: PlayerState;
  opp: PlayerState;
  pov: PovMode;
  onClose: () => void;
  onHover: (info: HoverCardInfo | null) => void;
}): ReactNode {
  if (!props.target) return null;
  const player = props.target.side === "hero" ? props.hero : props.opp;
  const who = props.target.side === "hero" ? "Hero" : "Opponent";
  let title = `${who} Exile (${player.exile.length})`;
  if (props.target.zone === "library") {
    title = `${who} Library (${player.zone_counts.library})`;
  } else if (props.target.zone === "graveyard") {
    title = `${who} Graveyard (${player.graveyard.length})`;
  }
  return (
    <div className="absolute top-2 right-2 bottom-2 w-[274px] max-w-[calc(100%-16px)] rounded-[9px] border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-lg)] z-40 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border)]">
        <strong className="text-xs">{title}</strong>
        <button
          type="button"
          onClick={(): void => {
            props.onHover(null);
            props.onClose();
          }}
          aria-label="Close zone drawer"
          className="rounded p-0.5 text-[var(--text-2)] hover:bg-[var(--surface-2)] cursor-pointer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 flex flex-col gap-1.5">
        <ZoneDrawerList zone={props.target.zone} player={player} onHover={props.onHover} />
      </div>
    </div>
  );
}

export function StackOverlay(props: {
  stack: StackItem[];
  open: boolean;
  onClose: () => void;
  onHoverSource: (permId: number | null) => void;
  onHoverCard: (info: HoverCardInfo | null) => void;
}): ReactNode {
  if (props.stack.length === 0 || !props.open) return null;
  return (
    <div className="absolute top-[52%] left-2 right-2 sm:left-auto sm:top-3 sm:right-[78px] z-30 sm:w-[236px] sm:max-w-[calc(100%-92px)] max-h-[44%] sm:max-h-[68%] rounded-[9px] border border-[var(--border-strong)] bg-[var(--surface)]/95 backdrop-blur-sm shadow-[var(--shadow-lg)] p-2 flex flex-col gap-1.5 overflow-y-auto">
      <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-2)]">
        <span>Stack ({props.stack.length})</span>
        <button
          type="button"
          onClick={props.onClose}
          aria-label="Collapse stack"
          className="rounded bg-[var(--surface-2)] p-0.5 cursor-pointer"
        >
          <X className="w-3 h-3" />
        </button>
      </div>
      {props.stack.map((item, idx) => (
        <div
          key={item.index}
          title={item.description}
          onMouseEnter={(e): void => {
            props.onHoverSource(item.source_permanent_id);
            props.onHoverCard({ name: item.card_name, anchorX: e.clientX, anchorY: e.clientY });
          }}
          onMouseLeave={(): void => {
            props.onHoverSource(null);
            props.onHoverCard(null);
          }}
          className="flex items-start gap-1.5 rounded-md border border-[var(--border)] bg-[var(--surface-2)] p-1.5 cursor-pointer hover:border-[var(--accent)]"
          style={idx === 0 ? { borderLeft: "3px solid var(--accent)" } : undefined}
        >
          <img
            src={cardImageUrl(item.card_name)}
            alt={item.card_name}
            className="w-[24px] h-[34px] rounded-[2px] object-cover shrink-0 mt-0.5"
          />
          <div className="min-w-0 flex-1">
            <div className="flex items-center justify-between gap-1 text-[10.5px] font-semibold">
              <span className="truncate">
                #{item.index} {item.card_name}
              </span>
              <span
                className={`rounded px-1 text-[8.5px] font-bold uppercase ${item.controller_side === "hero" ? "bg-[var(--win-soft)] text-[var(--win)]" : "bg-[var(--loss-soft)] text-[var(--loss)]"}`}
              >
                {item.controller_side === "hero" ? "Hero" : "Opp"}
              </span>
            </div>
            <div className="text-[9.5px] text-[var(--text-2)] leading-snug line-clamp-3 mt-0.5">
              {item.description}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

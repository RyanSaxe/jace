/** Tabletop auxiliary popovers: board details summary, watchlist pin form, and floating card zoom preview. */

import { Bookmark, X } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/Primitives";
import { useCardMeta } from "@/hooks/useStudioData";
import { cardImageUrl } from "@/lib/api";
import type { BoardState } from "@/lib/types";
import { defaultPinLabel } from "./tabletopHelpers";
import type { HoverCardInfo } from "./tabletopTypes";

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

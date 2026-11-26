/** Tabletop MTG zone composites: centered lands & creatures, peek-up Arena hands, overlay stack, zoom & timeline. */

import { ChevronLeft, ChevronRight, History, X } from "lucide-react";
import type { ReactNode } from "react";
import { useCardMeta } from "@/hooks/useStudioData";
import { cardImageUrl } from "@/lib/api";
import type {
  LandPile,
  PermanentState,
  PlayerState,
  PovMode,
  RecentEvent,
  StackItem,
  TimelinePoint,
} from "@/lib/types";

export interface HoverCardInfo {
  name: string;
  subtitle?: string;
  pt?: string;
  damage?: number;
  counters?: number;
  tapped?: boolean;
  token?: boolean;
  sick?: boolean;
  attacking?: boolean;
  blocking?: boolean;
  blockingName?: string;
  count?: number;
}

export interface DrawerTarget {
  side: "hero" | "opp";
  zone: "library" | "graveyard" | "exile";
}

function keyByOccurrence(names: string[]): Array<{ key: string; name: string; order: number }> {
  const seen = new Map<string, number>();
  return names.map((name, order) => {
    const count = (seen.get(name) ?? 0) + 1;
    seen.set(name, count);
    return { key: `${name}#${count}`, name, order };
  });
}

export function LandPilesRow(props: {
  piles: LandPile[];
  compact?: boolean;
  onHover: (info: HoverCardInfo | null) => void;
}): ReactNode {
  const seen = new Map<string, number>();
  const dense = Boolean(props.compact) || props.piles.length >= 6;
  const sizeClass = dense ? "w-[42px] h-[58px]" : "w-[50px] h-[68px]";
  return (
    <div
      className={`flex items-center justify-center flex-nowrap ${dense ? "gap-1.5" : "gap-2.5"}`}
    >
      {props.piles.map((pile) => {
        const baseKey = `${pile.name}-${pile.tapped ? "t" : "u"}-${pile.id ?? "basic"}`;
        const occ = (seen.get(baseKey) ?? 0) + 1;
        seen.set(baseKey, occ);
        return (
          <div
            key={`${baseKey}#${occ}`}
            onMouseEnter={(): void =>
              props.onHover({ name: pile.name, tapped: pile.tapped, count: pile.count })
            }
            onMouseLeave={(): void => props.onHover(null)}
            className={`relative ${sizeClass} shrink-0 cursor-pointer transition-transform ${
              pile.tapped ? "card-tapped" : "hover:-translate-y-1"
            }`}
          >
            {pile.count >= 3 ? (
              <div className="absolute inset-0 translate-x-[5px] -translate-y-[3px] rounded-[5px] bg-[#27272a] border border-white/20 opacity-55" />
            ) : null}
            {pile.count >= 2 ? (
              <div className="absolute inset-0 translate-x-[2.5px] -translate-y-[1.5px] rounded-[5px] bg-[#27272a] border border-white/20 opacity-75" />
            ) : null}
            <div className="relative w-full h-full rounded-[5px] overflow-hidden border border-black/45 shadow-[var(--shadow-sm)] bg-[#1c1917]">
              <img
                src={cardImageUrl(pile.name)}
                alt={pile.name}
                className="w-full h-full object-cover block"
              />
              <span className="tabular absolute top-0.5 right-0.5 rounded bg-black/88 px-1 text-[9.5px] font-bold text-white border border-white/20">
                ×{pile.count}
              </span>
              {pile.tapped ? (
                <span className="tabular absolute bottom-0.5 left-0.5 rounded bg-black/85 px-1 text-[8px] font-bold text-amber-400">
                  T
                </span>
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function cardWidthForCount(count: number, compact: boolean): number {
  if (compact) return 48;
  if (count >= 9) return 56;
  if (count >= 7) return 64;
  if (count >= 5) return 74;
  if (count >= 3) return 82;
  return 90;
}

function ringClassForPerm(
  perm: PermanentState,
  isHighlighted: boolean,
  tone: "target" | "source",
): string {
  if (isHighlighted) {
    return tone === "target"
      ? "ring-2 ring-[var(--win)] border-[var(--win)] -translate-y-1 z-20"
      : "ring-2 ring-[var(--accent)] border-[var(--accent)] -translate-y-1 z-20";
  }
  if (perm.attacking) return "ring-2 ring-amber-500 border-amber-500 z-10";
  if (perm.blocking) return "ring-2 ring-sky-500 border-sky-500 z-10";
  return "border-black/50";
}

function topLeftBadge(perm: PermanentState): { label: string; cls: string } | null {
  if (perm.attacking) return { label: "ATK", cls: "bg-amber-600 text-white shadow" };
  if (perm.blocking) return { label: "BLK", cls: "bg-sky-600 text-white shadow" };
  if (perm.is_token) return { label: "TOK", cls: "bg-black/85 text-zinc-100" };
  return null;
}

function PermCardBadges(props: { perm: PermanentState; pt: string | undefined }): ReactNode {
  const { perm, pt } = props;
  const topLeft = topLeftBadge(perm);
  return (
    <>
      {topLeft ? (
        <span
          className={`absolute top-0.5 left-0.5 rounded px-1 text-[7.5px] font-bold ${topLeft.cls}`}
        >
          {topLeft.label}
        </span>
      ) : null}
      {perm.plus1_counters > 0 ? (
        <span className="tabular absolute top-0.5 right-0.5 rounded-full bg-[var(--win-fill)] px-1 text-[8.5px] font-bold text-white shadow">
          +{perm.plus1_counters}/+{perm.plus1_counters}
        </span>
      ) : null}
      {perm.damage > 0 ? (
        <span className="tabular absolute bottom-0.5 left-0.5 rounded bg-[var(--loss-fill)] px-1 text-[8px] font-bold text-white shadow">
          -{perm.damage}
        </span>
      ) : null}
      {perm.damage === 0 && perm.sick ? (
        <span className="absolute bottom-0.5 left-1 rounded bg-amber-700/90 px-1 text-[7.5px] font-bold text-white">
          SICK
        </span>
      ) : null}
      {pt ? (
        <span
          className={`tabular absolute bottom-0.5 right-0.5 rounded bg-black/90 px-1 text-[9.5px] font-bold border border-white/20 ${
            perm.plus1_counters > 0 ? "text-emerald-300" : "text-white"
          }`}
        >
          {pt}
        </span>
      ) : null}
    </>
  );
}

export function PermanentsRow(props: {
  permanents: PermanentState[];
  compact?: boolean;
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  permNames?: ReadonlyMap<number, string>;
  onHover: (info: HoverCardInfo | null) => void;
  onPinCard?: (info: HoverCardInfo) => void;
}): ReactNode {
  const width = cardWidthForCount(props.permanents.length, Boolean(props.compact));
  const height = Math.round(width * 1.38);

  return (
    <div className="flex items-center justify-center gap-2 flex-nowrap min-w-0 overflow-x-auto no-scrollbar py-1">
      {props.permanents.map((perm) => {
        const pt = perm.is_creature ? `${perm.power}/${perm.toughness}` : undefined;
        const ringClass = ringClassForPerm(
          perm,
          props.highlightIds.has(perm.id),
          props.highlightTone,
        );
        const blockingName =
          perm.blocking_id !== null && perm.blocking_id !== undefined
            ? props.permNames?.get(perm.blocking_id)
            : undefined;
        const info: HoverCardInfo = {
          name: perm.name,
          pt,
          damage: perm.damage,
          counters: perm.plus1_counters,
          tapped: perm.tapped,
          token: perm.is_token,
          sick: perm.sick,
          attacking: perm.attacking,
          blocking: perm.blocking,
          blockingName,
        };
        return (
          <button
            key={perm.id}
            type="button"
            aria-label={`Inspect ${perm.name}`}
            style={{ width, height }}
            onMouseEnter={(): void => props.onHover(info)}
            onMouseLeave={(): void => props.onHover(null)}
            onClick={(): void => props.onPinCard?.(info)}
            className={`relative p-0 shrink-0 rounded-[7px] overflow-hidden border-[1.5px] bg-[#18181b] shadow-[var(--shadow-sm)] cursor-pointer transition-all ${ringClass} ${
              perm.tapped ? "card-tapped mx-1" : "hover:-translate-y-1.5 hover:z-20"
            }`}
          >
            <img
              src={cardImageUrl(perm.name)}
              alt={perm.name}
              className="w-full h-full object-cover block"
            />
            <PermCardBadges perm={perm} pt={pt} />
          </button>
        );
      })}
    </div>
  );
}

function handOverlapPx(count: number, compact: boolean): number {
  if (compact) return count >= 7 ? -16 : -10;
  if (count >= 8) return -26;
  if (count >= 6) return -18;
  return -10;
}

export function HandFanView(props: {
  side: "hero" | "opp";
  cards: string[];
  hiddenCount: number;
  hidden: boolean;
  onHover: (info: HoverCardInfo | null) => void;
  onPinCard?: (info: HoverCardInfo) => void;
}): ReactNode {
  const count = props.hidden ? props.hiddenCount : props.cards.length;
  if (count === 0) {
    return (
      <div className="h-[28px] flex items-center justify-center text-[10.5px] italic text-[var(--text-3)]">
        No cards in hand
      </div>
    );
  }

  if (props.hidden) {
    const overlap = handOverlapPx(count, true);
    const slots = Array.from({ length: count }, (_, slot) => `hidden-${slot + 1}`);
    return (
      <div className="flex items-start justify-center h-[36px] overflow-visible">
        {slots.map((slotKey, idx) => (
          <div
            key={slotKey}
            style={idx > 0 ? { marginLeft: overlap } : undefined}
            title="Hidden in Hero POV — switch to Omniscient to reveal"
            className="card-back w-[44px] h-[60px] -translate-y-[24px] rounded-[5px] shrink-0 shadow-[var(--shadow-sm)]"
          />
        ))}
      </div>
    );
  }

  const isHero = props.side === "hero";
  const overlap = handOverlapPx(count, !isHero);
  const keyed = keyByOccurrence(props.cards);
  const cardClasses = isHero
    ? "w-[82px] h-[114px] translate-y-[34px] hover:-translate-y-4 hover:scale-110"
    : "w-[70px] h-[98px] -translate-y-[32px] hover:translate-y-3 hover:scale-110";

  return (
    <div
      className={`flex justify-center overflow-visible ${
        isHero ? "items-end h-[74px]" : "items-start h-[58px]"
      }`}
    >
      {keyed.map((entry) => (
        <button
          key={entry.key}
          type="button"
          aria-label={`Inspect ${entry.name}`}
          style={entry.order > 0 ? { marginLeft: overlap } : undefined}
          onMouseEnter={(): void => props.onHover({ name: entry.name })}
          onMouseLeave={(): void => props.onHover(null)}
          onClick={(): void => props.onPinCard?.({ name: entry.name })}
          className={`relative p-0 ${cardClasses} rounded-[7px] overflow-hidden border border-black/45 bg-[#1c1917] shrink-0 shadow-[var(--shadow-md)] cursor-pointer transition-all duration-150 hover:z-40`}
        >
          <img
            src={cardImageUrl(entry.name)}
            alt={entry.name}
            className="w-full h-full object-cover object-top block"
          />
        </button>
      ))}
    </div>
  );
}

export function ZoneFlankView(props: {
  side: "hero" | "opp";
  player: PlayerState;
  onOpenDrawer: (target: DrawerTarget) => void;
}): ReactNode {
  const tiles: Array<{
    zone: "library" | "graveyard" | "exile";
    label: string;
    count: number;
    thumb?: string;
  }> = [
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
    <div className="flex flex-col justify-center gap-1.5 w-[70px] shrink-0">
      {tiles.map((tile) => (
        <button
          key={tile.zone}
          type="button"
          onClick={(): void => props.onOpenDrawer({ side: props.side, zone: tile.zone })}
          className="flex items-center justify-between gap-1 rounded-[7px] border border-[var(--border)] bg-[var(--surface)]/90 px-1.5 py-1 text-left cursor-pointer hover:border-[var(--accent)] hover:bg-[var(--accent-soft)] transition-colors"
          style={{ opacity: tile.count === 0 ? 0.55 : 1 }}
        >
          <div>
            <div className="text-[8.5px] font-semibold uppercase tracking-wide text-[var(--text-3)]">
              {tile.label}
            </div>
            <div className="tabular text-[12.5px] font-bold leading-tight">{tile.count}</div>
          </div>
          {tile.thumb ? (
            <img
              src={cardImageUrl(tile.thumb)}
              alt={tile.thumb}
              className="w-[18px] h-[25px] rounded-[2px] object-cover border border-[var(--border-strong)] shrink-0"
            />
          ) : null}
        </button>
      ))}
    </div>
  );
}

function zoneDrawerTitle(who: string, zone: DrawerTarget["zone"], player: PlayerState): string {
  if (zone === "library") return `${who} Library (${player.zone_counts.library})`;
  if (zone === "graveyard") return `${who} Graveyard (${player.graveyard.length})`;
  return `${who} Exile (${player.exile.length})`;
}

function ZoneDrawerBody(props: {
  zone: DrawerTarget["zone"];
  player: PlayerState;
  pov: PovMode;
  onHover: (info: HoverCardInfo | null) => void;
}): ReactNode {
  if (props.zone === "library") {
    if (props.player.library_hidden) {
      return (
        <div className="rounded-md border border-dashed border-[var(--border-strong)] bg-[var(--surface-2)] p-3 text-[11px] text-[var(--text-2)] leading-relaxed">
          <strong>Hidden in Hero POV.</strong> Opponent has{" "}
          <strong className="tabular">{props.player.zone_counts.library}</strong> cards remaining in
          library. Switch to <strong>Omniscient</strong> to inspect deck composition.
        </div>
      );
    }
    return (
      <>
        <div className="rounded-md border border-dashed border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-1.5 text-[10.5px] text-[var(--text-2)]">
          Unordered remaining-deck summary ({props.pov === "hero" ? "Hero POV" : "Omniscient"}).
        </div>
        {props.player.library_summary.map((row) => (
          <div
            key={row.name}
            onMouseEnter={(): void => props.onHover({ name: row.name, count: row.count })}
            onMouseLeave={(): void => props.onHover(null)}
            className="flex items-center gap-2 rounded-md border border-[var(--border)] bg-[var(--bg)] p-1.5 cursor-pointer hover:border-[var(--accent)]"
          >
            <span className="tabular text-[10.5px] text-[var(--text-3)] w-6">×{row.count}</span>
            <img
              src={cardImageUrl(row.name)}
              alt={row.name}
              className="w-[24px] h-[34px] rounded-[2px] object-cover shrink-0"
            />
            <span className="text-[11.5px] font-medium truncate">{row.name}</span>
          </div>
        ))}
      </>
    );
  }

  const list = props.zone === "graveyard" ? props.player.graveyard : props.player.exile;
  if (list.length === 0) {
    return <div className="py-6 text-center text-xs text-[var(--text-3)]">Zone is empty.</div>;
  }
  return keyByOccurrence(list).map((entry) => (
    <div
      key={entry.key}
      onMouseEnter={(): void => props.onHover({ name: entry.name })}
      onMouseLeave={(): void => props.onHover(null)}
      className="flex items-center gap-2 rounded-md border border-[var(--border)] bg-[var(--bg)] p-1.5 cursor-pointer hover:border-[var(--accent)]"
    >
      <span className="tabular text-[10px] text-[var(--text-3)] w-5">#{entry.order + 1}</span>
      <img
        src={cardImageUrl(entry.name)}
        alt={entry.name}
        className="w-[24px] h-[34px] rounded-[2px] object-cover shrink-0"
      />
      <div className="min-w-0">
        <div className="text-[11.5px] font-medium truncate">{entry.name}</div>
        <div className="text-[10px] text-[var(--text-3)]">
          {entry.order === 0 ? "Top of zone" : `#${entry.order + 1}`}
        </div>
      </div>
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
  return (
    <div className="absolute top-2 right-2 bottom-2 w-[274px] max-w-[calc(100%-16px)] rounded-[9px] border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-lg)] z-40 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-[var(--border)]">
        <strong className="text-xs">{zoneDrawerTitle(who, props.target.zone, player)}</strong>
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
        <ZoneDrawerBody
          zone={props.target.zone}
          player={player}
          pov={props.pov}
          onHover={props.onHover}
        />
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
    <div className="absolute top-3 right-[84px] z-30 w-[236px] max-w-[calc(100%-96px)] max-h-[68%] flex flex-col">
      <div className="rounded-[9px] border border-[var(--border-strong)] bg-[var(--surface)]/95 backdrop-blur-sm shadow-[var(--shadow-lg)] p-2 flex flex-col gap-1.5 overflow-y-auto">
        <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-2)]">
          <span>Stack ({props.stack.length})</span>
          <div className="flex items-center gap-1.5">
            <span className="tabular text-[var(--accent)]">#1 Top</span>
            <button
              type="button"
              onClick={props.onClose}
              aria-label="Collapse stack"
              className="rounded bg-[var(--surface-2)] p-0.5 cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        </div>
        {props.stack.map((item, idx) => {
          const isHeroCtrl = item.controller_side === "hero";
          return (
            <div
              key={item.index}
              title={item.description}
              onMouseEnter={(): void => {
                props.onHoverSource(item.source_permanent_id);
                props.onHoverCard({ name: item.card_name });
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
                className="w-[24px] h-[34px] rounded-[2px] object-cover border border-[var(--border)] shrink-0 mt-0.5"
              />
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-1">
                  <span className="text-[10.5px] font-semibold truncate">
                    #{item.index} {item.card_name}
                  </span>
                  <span
                    className={`rounded px-1 text-[8.5px] font-bold uppercase shrink-0 ${
                      isHeroCtrl
                        ? "bg-[var(--win-soft)] text-[var(--win)]"
                        : "bg-[var(--loss-soft)] text-[var(--loss)]"
                    }`}
                  >
                    {isHeroCtrl ? "Hero" : "Opp"}
                  </span>
                </div>
                <div className="text-[9.5px] text-[var(--text-2)] leading-snug break-words line-clamp-3 mt-0.5">
                  {item.description}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function blockingTagLabel(info: HoverCardInfo): string | false {
  if (!info.blocking) return false;
  return info.blockingName ? `Blocking ${info.blockingName}` : "Blocking";
}

function zoomTagList(info: HoverCardInfo, manaCost?: string, setCode?: string): string[] {
  const optional: Array<string | false | undefined> = [
    manaCost,
    info.pt,
    info.damage && info.damage > 0 ? `${info.damage} dmg` : false,
    info.counters ? `+${info.counters}/+${info.counters}` : false,
    info.attacking ? "Attacking" : false,
    blockingTagLabel(info),
    info.tapped ? "Tapped" : false,
    info.token ? "Token" : false,
    setCode,
  ];
  return optional.filter((t): t is string => typeof t === "string" && t.length > 0);
}

function CardZoomBody(props: {
  name: string;
  typeLine: string;
  tags: string[];
  bodyText: string;
  showClose: boolean;
  onDismiss?: () => void;
}): ReactNode {
  return (
    <div className="min-w-0 flex-1 flex flex-col gap-1 max-h-[360px] overflow-y-auto pr-0.5">
      <div className="flex items-start justify-between gap-1">
        <div className="text-xs font-bold leading-tight">{props.name}</div>
        {props.showClose ? (
          <button
            type="button"
            onClick={props.onDismiss}
            aria-label="Close card preview"
            className="rounded p-0.5 text-[var(--text-2)] hover:bg-[var(--surface-2)] cursor-pointer shrink-0"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        ) : null}
      </div>
      {props.typeLine ? (
        <div className="text-[10px] text-[var(--text-3)]">{props.typeLine}</div>
      ) : null}
      <div className="flex flex-wrap gap-1">
        {props.tags.map((tag) => (
          <span
            key={tag}
            className="tabular rounded bg-[var(--surface-2)] px-1.5 py-0.5 text-[9.5px] font-semibold text-[var(--text-2)]"
          >
            {tag}
          </span>
        ))}
      </div>
      <p className="text-[10.5px] leading-snug text-[var(--text-2)] whitespace-pre-line m-0">
        {props.bodyText}
      </p>
    </div>
  );
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

export function CardZoomPreview(props: {
  info: HoverCardInfo | null;
  pinned?: boolean;
  onDismiss?: () => void;
}): ReactNode {
  const meta = useCardMeta(props.info?.name ?? null);
  if (!props.info) return null;
  const copy = resolveCardCopy(meta);
  const tags = zoomTagList(props.info, copy.manaCost, copy.setCode);

  return (
    <div
      className={`${
        props.pinned ? "pointer-events-auto" : "pointer-events-none"
      } absolute left-3 top-11 z-50 w-[360px] max-w-[calc(100%-24px)] rounded-[10px] border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-lg)] p-2.5 flex gap-2.5 items-start`}
    >
      <img
        src={cardImageUrl(props.info.name)}
        alt={props.info.name}
        className="w-[118px] h-[165px] rounded-[6px] object-cover border border-[var(--border)] shrink-0"
      />
      <CardZoomBody
        name={props.info.name}
        typeLine={copy.typeLine}
        tags={tags}
        bodyText={copy.bodyText}
        showClose={Boolean(props.pinned && props.onDismiss)}
        onDismiss={props.onDismiss}
      />
    </div>
  );
}

function TimelineCurveSvg(props: {
  timeline: TimelinePoint[];
  currentTurn: number;
  onSelectStep: (step: number) => void;
}): ReactNode {
  const pts = props.timeline;
  if (pts.length < 2) return null;
  const w = 860;
  const h = 38;
  const padX = 22;
  const stepX = (w - padX * 2) / (pts.length - 1);
  const coords = pts.map((p, i) => ({
    turn: p.turn,
    step: p.step,
    x: padX + i * stepX,
    y: h - 13 - Math.max(0, Math.min(1, p.eval_win)) * 21,
  }));
  const poly = coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");
  const firstX = coords[0]?.x ?? padX;
  const lastX = coords[coords.length - 1]?.x ?? padX;
  const area = `${firstX},${h - 13} ${poly} ${lastX},${h - 13}`;
  const labelStride = Math.max(1, Math.ceil(coords.length / 18));

  return (
    <div className="h-[38px] w-full">
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[38px] block">
        <line
          x1={padX}
          y1={h - 23}
          x2={w - padX}
          y2={h - 23}
          stroke="var(--border)"
          strokeDasharray="3 3"
          className="pointer-events-none"
        />
        <polygon points={area} fill="var(--accent-soft)" className="pointer-events-none" />
        <polyline
          points={poly}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="1.8"
          strokeLinejoin="round"
          className="pointer-events-none"
        />
        {coords.map((c, idx) => {
          const active = c.turn === props.currentTurn;
          const showLabel = active || idx === coords.length - 1 || idx % labelStride === 0;
          return (
            <g
              key={c.turn}
              onClick={(): void => props.onSelectStep(c.step)}
              className="cursor-pointer"
            >
              <rect
                x={c.x - Math.max(10, stepX / 2)}
                y={0}
                width={Math.max(20, stepX)}
                height={h}
                fill="transparent"
              />
              <circle
                cx={c.x}
                cy={c.y}
                r={active ? 4.5 : 3}
                fill={active ? "var(--accent)" : "var(--text-2)"}
                stroke="var(--surface)"
                strokeWidth="1.4"
              />
              {showLabel ? (
                <text
                  x={c.x}
                  y={h - 1}
                  textAnchor="middle"
                  fontSize="9"
                  fill={active ? "var(--text)" : "var(--text-3)"}
                  fontWeight={active ? 700 : 500}
                  className="tabular"
                >
                  T{c.turn}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function TurnTimelineBar(props: {
  timeline: TimelinePoint[];
  evalSource?: string;
  recentEvents: RecentEvent[];
  currentStep: number;
  totalSteps: number;
  maxPlayableStep: number;
  currentTurn: number;
  historyOpen: boolean;
  onToggleHistory: () => void;
  onCloseHistory: () => void;
  onSelectStep: (step: number) => void;
}): ReactNode {
  const upperStep = Math.max(0, Math.min(Math.max(0, props.totalSteps - 1), props.maxPlayableStep));
  const curveTitle = props.evalSource === "model" ? "Model Win%" : "Turn Timeline";

  return (
    <div className="relative shrink-0 border-t border-[var(--border)] bg-[var(--surface)] px-3.5 py-1.5 flex flex-col gap-1">
      {props.historyOpen && props.recentEvents.length > 0 ? (
        <div className="absolute left-3 bottom-full mb-1.5 z-40 w-[340px] max-w-[calc(100%-24px)] rounded-lg border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-lg)] p-2 flex flex-col gap-1">
          <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-[var(--text-3)] px-1 pb-1 border-b border-[var(--border)]">
            <span>Recent Actions (Newest First)</span>
            <button
              type="button"
              onClick={props.onCloseHistory}
              aria-label="Close recent actions"
              className="p-0.5 hover:text-[var(--text)] cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
          {props.recentEvents.toReversed().map((ev) => (
            <button
              key={`${ev.step}-${ev.text}`}
              type="button"
              onClick={(): void => {
                props.onSelectStep(ev.step);
                props.onCloseHistory();
              }}
              className="flex items-center gap-1.5 rounded px-1.5 py-1 text-left text-[11px] hover:bg-[var(--surface-2)] cursor-pointer"
            >
              <span className="tabular text-[10px] text-[var(--text-3)] shrink-0">
                T{ev.turn} #{ev.step}
              </span>
              <span
                className={`font-semibold shrink-0 ${
                  ev.actor_side === "hero" ? "text-[var(--win)]" : "text-[var(--loss)]"
                }`}
              >
                {ev.actor_side === "hero" ? "Hero" : "Opp"}:
              </span>
              <span className="truncate">{ev.text}</span>
            </button>
          ))}
        </div>
      ) : null}

      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-[var(--text-2)]">
        <div className="flex items-center gap-2 min-w-0">
          <strong className="text-[var(--text)] whitespace-nowrap">{curveTitle}</strong>
          <span className="tabular whitespace-nowrap">
            T{props.currentTurn} · Step {props.currentStep}/{upperStep}
          </span>
          {props.recentEvents.length > 0 ? (
            <button
              type="button"
              onClick={props.onToggleHistory}
              className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] border cursor-pointer transition-colors ${
                props.historyOpen
                  ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                  : "border-[var(--border)] text-[var(--text-2)] hover:text-[var(--text)]"
              }`}
            >
              <History className="w-3 h-3" />
              History
            </button>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          <input
            type="range"
            min={0}
            max={Math.max(0, upperStep)}
            value={Math.min(props.currentStep, upperStep)}
            aria-label="Scrub game step"
            onChange={(e): void => props.onSelectStep(Number(e.target.value))}
            className="w-28 sm:w-44 accent-[var(--accent)] cursor-pointer"
          />
          <button
            type="button"
            title="Previous step (←)"
            aria-label="Previous step"
            disabled={props.currentStep <= 0}
            onClick={(): void => props.onSelectStep(Math.max(0, props.currentStep - 1))}
            className="w-6 h-6 rounded border border-[var(--border)] bg-[var(--surface-2)] flex items-center justify-center cursor-pointer disabled:opacity-40 hover:border-[var(--accent)]"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            title="Next step (→)"
            aria-label="Next step"
            disabled={props.currentStep >= upperStep}
            onClick={(): void => props.onSelectStep(Math.min(upperStep, props.currentStep + 1))}
            className="w-6 h-6 rounded border border-[var(--border)] bg-[var(--surface-2)] flex items-center justify-center cursor-pointer disabled:opacity-40 hover:border-[var(--accent)]"
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <TimelineCurveSvg
        timeline={props.timeline}
        currentTurn={props.currentTurn}
        onSelectStep={props.onSelectStep}
      />
    </div>
  );
}

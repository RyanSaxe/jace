/** Tabletop MTG zone composites: grouped lands, permanents, hand fan, flank tiles, zone sheet, stack & timeline. */

import { ChevronLeft, ChevronRight, X } from "lucide-react";
import type { ReactNode } from "react";
import { useCardMeta } from "@/hooks/useStudioData";
import { cardImageUrl } from "@/lib/api";
import type {
  LandPile,
  PermanentState,
  PlayerState,
  PovMode,
  StackItem,
  TimelinePoint,
} from "@/lib/types";

export interface HoverCardInfo {
  name: string;
  subtitle?: string;
  pt?: string;
  counters?: number;
  tapped?: boolean;
  token?: boolean;
  sick?: boolean;
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
  onHover: (info: HoverCardInfo | null) => void;
}): ReactNode {
  const seen = new Map<string, number>();
  return (
    <div className="flex items-center gap-2.5 flex-nowrap">
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
            className={`relative w-[48px] h-[64px] shrink-0 cursor-pointer transition-transform ${
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
  if (count >= 9) return 52;
  if (count >= 7) return 62;
  if (count >= 5) return 72;
  if (count >= 3) return 80;
  return 88;
}

function ringClassForPerm(isHighlighted: boolean, tone: "target" | "source"): string {
  if (!isHighlighted) return "border-black/50";
  if (tone === "target") return "ring-2 ring-[var(--win)] border-[var(--win)] -translate-y-1 z-20";
  return "ring-2 ring-[var(--accent)] border-[var(--accent)] -translate-y-1 z-20";
}

export function PermanentsRow(props: {
  permanents: PermanentState[];
  compact?: boolean;
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  onHover: (info: HoverCardInfo | null) => void;
}): ReactNode {
  const width = cardWidthForCount(props.permanents.length, Boolean(props.compact));
  const height = Math.round(width * 1.38);

  return (
    <div className="flex items-center gap-1.5 flex-nowrap min-w-0">
      {props.permanents.map((perm) => {
        const pt = perm.is_creature ? `${perm.power}/${perm.toughness}` : undefined;
        const ringClass = ringClassForPerm(props.highlightIds.has(perm.id), props.highlightTone);
        return (
          <div
            key={perm.id}
            style={{ width, height }}
            onMouseEnter={(): void =>
              props.onHover({
                name: perm.name,
                pt,
                counters: perm.plus1_counters,
                tapped: perm.tapped,
                token: perm.is_token,
                sick: perm.sick,
              })
            }
            onMouseLeave={(): void => props.onHover(null)}
            className={`relative shrink-0 rounded-[6px] overflow-hidden border-[1.5px] bg-[#18181b] shadow-[var(--shadow-sm)] cursor-pointer transition-all ${ringClass} ${
              perm.tapped ? "card-tapped mx-0.5" : "hover:-translate-y-1 hover:z-20"
            }`}
          >
            <img
              src={cardImageUrl(perm.name)}
              alt={perm.name}
              className="w-full h-full object-cover block"
            />
            {perm.is_token ? (
              <span className="absolute top-0.5 left-0.5 rounded bg-black/85 px-1 text-[7.5px] font-bold text-zinc-200">
                TOK
              </span>
            ) : null}
            {perm.plus1_counters > 0 ? (
              <span className="tabular absolute top-0.5 right-0.5 rounded-full bg-[var(--win-fill)] px-1 text-[8.5px] font-bold text-white shadow">
                +{perm.plus1_counters}/+{perm.plus1_counters}
              </span>
            ) : null}
            {perm.sick ? (
              <span className="absolute bottom-0.5 left-0.5 rounded bg-amber-700/90 px-1 text-[7.5px] font-bold text-white">
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
          </div>
        );
      })}
    </div>
  );
}

function handOverlapPx(count: number): number {
  if (count >= 9) return -16;
  if (count >= 6) return -11;
  return -5;
}

export function HandFanView(props: {
  cards: string[];
  hiddenCount: number;
  hidden: boolean;
  onHover: (info: HoverCardInfo | null) => void;
}): ReactNode {
  const count = props.hidden ? props.hiddenCount : props.cards.length;
  if (count === 0) {
    return <div className="text-[11px] italic text-[var(--text-3)]">Hand empty (0)</div>;
  }
  const overlap = handOverlapPx(count);

  if (props.hidden) {
    const slots = Array.from({ length: count }, (_, slot) => `hidden-${slot + 1}`);
    return (
      <div className="flex items-center justify-end h-[52px]">
        {slots.map((slotKey, idx) => (
          <div
            key={slotKey}
            style={idx > 0 ? { marginLeft: overlap } : undefined}
            title="Hidden in Hero POV — switch to Omniscient to reveal"
            className="card-back w-[38px] h-[52px] rounded shrink-0 shadow-[var(--shadow-sm)]"
          />
        ))}
      </div>
    );
  }

  const keyed = keyByOccurrence(props.cards);
  return (
    <div className="flex items-center justify-end h-[52px]">
      {keyed.map((entry) => (
        <div
          key={entry.key}
          style={entry.order > 0 ? { marginLeft: overlap } : undefined}
          onMouseEnter={(): void => props.onHover({ name: entry.name })}
          onMouseLeave={(): void => props.onHover(null)}
          className="relative w-[38px] h-[52px] rounded overflow-hidden border border-black/40 bg-[#1c1917] shrink-0 shadow-[var(--shadow-sm)] cursor-pointer transition-transform hover:-translate-y-2.5 hover:scale-115 hover:z-30"
        >
          <img
            src={cardImageUrl(entry.name)}
            alt={entry.name}
            className="w-full h-full object-cover block"
          />
        </div>
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
    <div className="flex flex-col justify-center gap-1.5 w-[74px] shrink-0">
      {tiles.map((tile) => (
        <button
          key={tile.zone}
          type="button"
          onClick={(): void => props.onOpenDrawer({ side: props.side, zone: tile.zone })}
          className="flex items-center justify-between gap-1 rounded-[7px] border border-[var(--border)] bg-[var(--surface)] px-1.5 py-1 text-left cursor-pointer hover:border-[var(--accent)] hover:bg-[var(--accent-soft)] transition-colors"
          style={{ opacity: tile.count === 0 ? 0.6 : 1 }}
        >
          <div>
            <div className="text-[9px] font-semibold uppercase tracking-wide text-[var(--text-3)]">
              {tile.label}
            </div>
            <div className="tabular text-[13px] font-bold leading-tight">{tile.count}</div>
          </div>
          {tile.thumb ? (
            <img
              src={cardImageUrl(tile.thumb)}
              alt={tile.thumb}
              className="w-[20px] h-[28px] rounded-[2px] object-cover border border-[var(--border-strong)] shrink-0"
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
          library. Switch to <strong>Omniscient</strong> in the top bar to inspect deck composition.
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
          {entry.order === 0 ? "Top of zone" : `Order #${entry.order + 1}`}
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
    <div className="absolute top-2 right-2 bottom-2 w-[274px] rounded-[9px] border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-lg)] z-40 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-[var(--border)]">
        <strong className="text-xs">{zoneDrawerTitle(who, props.target.zone, player)}</strong>
        <button
          type="button"
          onClick={props.onClose}
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

export function StackColumn(props: {
  stack: StackItem[];
  open: boolean;
  onClose: () => void;
  onHoverSource: (permId: number | null) => void;
  onHoverCard: (info: HoverCardInfo | null) => void;
}): ReactNode {
  if (props.stack.length === 0 || !props.open) return null;
  return (
    <div className="w-[214px] shrink-0 flex flex-col justify-center min-h-0">
      <div className="rounded-[9px] border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-md)] p-2 flex flex-col gap-1.5 max-h-full overflow-y-auto">
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
        {props.stack.map((item, idx) => (
          <div
            key={item.index}
            onMouseEnter={(): void => {
              props.onHoverSource(item.source_permanent_id);
              props.onHoverCard({ name: item.card_name });
            }}
            onMouseLeave={(): void => {
              props.onHoverSource(null);
              props.onHoverCard(null);
            }}
            className="flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--surface-2)] p-1.5 cursor-pointer hover:border-[var(--accent)]"
            style={idx === 0 ? { borderLeft: "3px solid var(--accent)" } : undefined}
          >
            <img
              src={cardImageUrl(item.card_name)}
              alt={item.card_name}
              className="w-[22px] h-[30px] rounded-[2px] object-cover border border-[var(--border)] shrink-0"
            />
            <div className="min-w-0 flex-1">
              <div className="text-[10.5px] font-semibold truncate">
                #{item.index} {item.card_name}
              </div>
              <div className="text-[9.5px] text-[var(--text-2)] truncate">{item.description}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function zoomTagList(info: HoverCardInfo, setCode?: string): string[] {
  const tags: string[] = [];
  if (info.pt) tags.push(info.pt);
  if (info.counters) tags.push(`+${info.counters}/+${info.counters}`);
  if (info.tapped) tags.push("Tapped");
  if (info.token) tags.push("Token");
  if (setCode) tags.push(setCode);
  return tags;
}

export function CardZoomPreview(props: { info: HoverCardInfo | null }): ReactNode {
  const meta = useCardMeta(props.info?.name ?? null);
  if (!props.info) return null;
  const info = props.info;
  const tags = zoomTagList(info, meta?.set_code);
  const bodyText = meta?.oracle_text || meta?.type_line || "Loading oracle text…";

  return (
    <div className="pointer-events-none absolute left-3 bottom-3 z-50 w-[300px] rounded-[10px] border border-[var(--border-strong)] bg-[var(--surface)] shadow-[var(--shadow-lg)] p-2.5 flex gap-2.5 items-start">
      <img
        src={cardImageUrl(info.name)}
        alt={info.name}
        className="w-[106px] h-[148px] rounded-[6px] object-cover border border-[var(--border)] shrink-0"
      />
      <div className="min-w-0 flex-1 flex flex-col gap-1">
        <div className="text-xs font-bold leading-tight">{info.name}</div>
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
        <p className="text-[10.5px] leading-snug text-[var(--text-2)] line-clamp-5 m-0">
          {bodyText}
        </p>
      </div>
    </div>
  );
}

export function TurnTimelineBar(props: {
  timeline: TimelinePoint[];
  currentStep: number;
  totalSteps: number;
  currentTurn: number;
  onSelectStep: (step: number) => void;
}): ReactNode {
  const pts = props.timeline;
  const w = 860;
  const h = 40;
  const padX = 18;
  const stepX = pts.length > 1 ? (w - padX * 2) / (pts.length - 1) : 0;
  const coords = pts.map((p, i) => ({
    turn: p.turn,
    step: p.step,
    x: padX + i * stepX,
    y: h - 13 - p.eval_win * 23,
  }));
  const poly = coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");
  const firstX = coords[0]?.x ?? padX;
  const lastX = coords[coords.length - 1]?.x ?? padX;
  const area = `${firstX},${h - 13} ${poly} ${lastX},${h - 13}`;

  return (
    <div className="h-[72px] shrink-0 border-t border-[var(--border)] bg-[var(--surface)] px-3.5 py-1.5 flex flex-col justify-between">
      <div className="flex items-center justify-between text-[11px] text-[var(--text-2)]">
        <div>
          <strong className="text-[var(--text)]">Turn-grouped Critic Win%</strong>{" "}
          <span className="tabular">
            · Turn {props.currentTurn} · Step {props.currentStep} /{" "}
            {Math.max(0, props.totalSteps - 1)}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            title="Previous step"
            aria-label="Previous step"
            onClick={(): void => props.onSelectStep(Math.max(0, props.currentStep - 1))}
            className="w-6 h-5 rounded border border-[var(--border)] bg-[var(--surface-2)] flex items-center justify-center cursor-pointer hover:border-[var(--accent)]"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            title="Next step"
            aria-label="Next step"
            onClick={(): void =>
              props.onSelectStep(Math.min(Math.max(0, props.totalSteps - 1), props.currentStep + 1))
            }
            className="w-6 h-5 rounded border border-[var(--border)] bg-[var(--surface-2)] flex items-center justify-center cursor-pointer hover:border-[var(--accent)]"
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
      <div className="h-[40px] w-full">
        {coords.length > 0 ? (
          <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[40px] block">
            <line
              x1={padX}
              y1={h - 24.5}
              x2={w - padX}
              y2={h - 24.5}
              stroke="var(--border)"
              strokeDasharray="3 3"
            />
            <polygon points={area} fill="var(--win-soft)" />
            <polyline
              points={poly}
              fill="none"
              stroke="var(--win)"
              strokeWidth="2"
              strokeLinejoin="round"
            />
            {coords.map((c) => {
              const active = c.turn === props.currentTurn;
              return (
                <g
                  key={c.turn}
                  onClick={(): void => props.onSelectStep(c.step)}
                  className="cursor-pointer"
                >
                  <circle
                    cx={c.x}
                    cy={c.y}
                    r={active ? 4.8 : 3.2}
                    fill={active ? "var(--accent)" : "var(--win)"}
                    stroke="var(--surface)"
                    strokeWidth="1.5"
                  />
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
                </g>
              );
            })}
          </svg>
        ) : null}
      </div>
    </div>
  );
}

/** Responsive tabletop MTG board zones: land piles, battlefield permanents, and hand fan renderers. */

import type { ReactNode } from "react";
import { cardImageUrl } from "@/lib/api";
import type { LandPile, PermanentState } from "@/lib/types";
import { keyByOccurrence } from "./tabletopHelpers";
import type { HandCardEditTarget, HoverCardInfo, TabletopScale } from "./tabletopTypes";

function landSizeCls(scale: TabletopScale, dense: boolean): string {
  if (scale === "mobile-short") return dense ? "w-[32px] h-[44px]" : "w-[34px] h-[47px]";
  if (scale === "mobile-tall") return dense ? "w-[36px] h-[50px]" : "w-[38px] h-[52px]";
  return dense ? "w-[42px] h-[56px]" : "w-[48px] h-[66px]";
}

export function LandPilesRow(props: {
  piles: LandPile[];
  compact?: boolean;
  scale?: TabletopScale;
  onHover: (info: HoverCardInfo | null) => void;
}): ReactNode {
  const seen = new Map<string, number>();
  const scale = props.scale ?? "desktop";
  const dense = Boolean(props.compact) || props.piles.length >= 6;
  const size = landSizeCls(scale, dense);
  const gap = dense || scale !== "desktop" ? "gap-1.5" : "gap-2";
  const items = props.piles.map((pile) => {
    const key = `${pile.name}-${pile.tapped ? "t" : "u"}-${pile.id ?? "basic"}`;
    const occ = (seen.get(key) ?? 0) + 1;
    seen.set(key, occ);
    return (
      <div
        key={`${key}#${occ}`}
        onMouseEnter={(e): void =>
          props.onHover({
            name: pile.name,
            tapped: pile.tapped,
            count: pile.count,
            anchorX: e.clientX,
            anchorY: e.clientY,
          })
        }
        onMouseLeave={(): void => props.onHover(null)}
        className={`relative ${size} shrink-0 cursor-pointer transition-transform ${pile.tapped ? "card-tapped" : "hover:-translate-y-1"}`}
      >
        <div className="relative w-full h-full rounded-[5px] overflow-hidden border border-black/45 shadow-[var(--shadow-sm)] bg-[#1c1917]">
          <img
            src={cardImageUrl(pile.name)}
            alt={pile.name}
            className="w-full h-full object-cover"
          />
          <span className="tabular absolute top-0.5 right-0.5 rounded bg-black/88 px-1 text-[9px] font-bold text-white">
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
  });
  if (props.compact) {
    return <div className={`flex items-center flex-nowrap ${gap}`}>{items}</div>;
  }
  return (
    <div className="w-full min-w-0 overflow-x-auto no-scrollbar py-1">
      <div className={`w-max mx-auto flex items-center flex-nowrap px-1.5 ${gap}`}>{items}</div>
    </div>
  );
}

function mobileCardWidth(count: number, compact: boolean, short: boolean): number {
  if (compact) return short ? 34 : 40;
  if (count >= 7) return short ? 44 : 48;
  if (count >= 5) return short ? 48 : 54;
  if (count >= 3) return short ? 50 : 58;
  return short ? 52 : 64;
}

function cardWidth(count: number, compact: boolean, scale: TabletopScale): number {
  if (scale !== "desktop") return mobileCardWidth(count, compact, scale === "mobile-short");
  if (compact) return 48;
  if (count >= 9) return 56;
  if (count >= 7) return 64;
  if (count >= 5) return 74;
  if (count >= 3) return 84;
  return 92;
}

function permRing(perm: PermanentState, highlighted: boolean, tone: "target" | "source"): string {
  if (highlighted) {
    return tone === "target"
      ? "ring-2 ring-[var(--win)] border-[var(--win)] -translate-y-1 z-20"
      : "ring-2 ring-[var(--accent)] border-[var(--accent)] -translate-y-1 z-20";
  }
  if (perm.attacking) return "ring-2 ring-amber-500 border-amber-500 z-10";
  if (perm.blocking) return "ring-2 ring-sky-500 border-sky-500 z-10";
  return "border-black/50";
}

function permTopLeft(perm: PermanentState): { label: string; cls: string } | null {
  if (perm.attacking) return { label: "ATK", cls: "bg-amber-600 text-white" };
  if (perm.blocking) return { label: "BLK", cls: "bg-sky-600 text-white" };
  if (perm.is_token) return { label: "TOK", cls: "bg-black/85 text-zinc-100" };
  return null;
}

function permBottomLeft(perm: PermanentState): { text: string; cls: string } | null {
  if (perm.damage > 0) return { text: `-${perm.damage}`, cls: "bg-[var(--loss-fill)] text-[8px]" };
  if (perm.sick) return { text: "SICK", cls: "bg-amber-700/95 text-[7.5px]" };
  return null;
}

function PermCardBadges(props: { perm: PermanentState; pt: string | undefined }): ReactNode {
  const { perm, pt } = props;
  const tl = permTopLeft(perm);
  const bl = permBottomLeft(perm);
  const hasCopy = perm.copy_index !== null && perm.copy_index !== undefined;
  return (
    <>
      {tl ? (
        <span className={`absolute top-0.5 left-0.5 rounded px-1 text-[7.5px] font-bold ${tl.cls}`}>
          {tl.label}
        </span>
      ) : null}
      {hasCopy || perm.plus1_counters > 0 ? (
        <div className="absolute top-0.5 right-0.5 flex flex-col items-end gap-0.5">
          {hasCopy ? (
            <span className="tabular rounded bg-black/90 px-1 text-[8px] font-bold text-amber-200 border border-white/25">
              #{perm.copy_index}
            </span>
          ) : null}
          {perm.plus1_counters > 0 ? (
            <span className="tabular rounded-full bg-[var(--win-fill)] px-1 text-[8px] font-bold text-white">
              +{perm.plus1_counters}/+{perm.plus1_counters}
            </span>
          ) : null}
        </div>
      ) : null}
      {bl ? (
        <span
          className={`tabular absolute bottom-0.5 left-0.5 rounded px-1 font-bold text-white ${bl.cls}`}
        >
          {bl.text}
        </span>
      ) : null}
      {pt ? (
        <span
          className={`tabular absolute bottom-0.5 right-0.5 rounded bg-black/90 px-1 text-[9.5px] font-bold border border-white/20 ${perm.plus1_counters > 0 ? "text-emerald-300" : "text-white"}`}
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
  scale?: TabletopScale;
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  permNames?: ReadonlyMap<number, string>;
  onHover: (info: HoverCardInfo | null) => void;
  onPinCard?: (info: HoverCardInfo) => void;
}): ReactNode {
  const scale = props.scale ?? "desktop";
  const width = cardWidth(props.permanents.length, Boolean(props.compact), scale);
  const height = Math.round(width * 1.38);
  const cards = props.permanents.map((perm) => {
    const pt = perm.is_creature ? `${perm.power}/${perm.toughness}` : undefined;
    const blockingName =
      perm.blocking_id === null || perm.blocking_id === undefined
        ? undefined
        : props.permNames?.get(perm.blocking_id);
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
      copyIndex: perm.copy_index,
    };
    const suffix =
      perm.copy_index === null || perm.copy_index === undefined ? "" : ` #${perm.copy_index}`;
    return (
      <button
        key={perm.id}
        type="button"
        aria-label={`Inspect ${perm.name}${suffix}`}
        style={{ width, height }}
        onMouseEnter={(e): void =>
          props.onHover({ ...info, anchorX: e.clientX, anchorY: e.clientY })
        }
        onMouseLeave={(): void => props.onHover(null)}
        onClick={(e): void =>
          props.onPinCard?.({ ...info, anchorX: e.clientX, anchorY: e.clientY })
        }
        className={`relative p-0 shrink-0 rounded-[7px] overflow-hidden border-[1.5px] bg-[#18181b] shadow-[var(--shadow-sm)] cursor-pointer transition-all ${permRing(perm, props.highlightIds.has(perm.id), props.highlightTone)} ${perm.tapped ? "card-tapped mx-1" : "hover:-translate-y-1.5 hover:z-20"}`}
      >
        <img src={cardImageUrl(perm.name)} alt={perm.name} className="w-full h-full object-cover" />
        <PermCardBadges perm={perm} pt={pt} />
      </button>
    );
  });
  if (props.compact) {
    return <div className="flex items-center gap-1.5 lg:gap-2 flex-nowrap">{cards}</div>;
  }
  return (
    <div className="w-full min-w-0 overflow-x-auto no-scrollbar py-1.5">
      <div className="w-max mx-auto flex items-center gap-1.5 lg:gap-2 flex-nowrap px-2">
        {cards}
      </div>
    </div>
  );
}

function handOverlapPx(count: number, compact: boolean, mobile: boolean): number {
  if (mobile) {
    if (compact) return count >= 6 ? -16 : -12;
    if (count >= 7) return -22;
    return count >= 5 ? -16 : -12;
  }
  if (compact) return count >= 7 ? -16 : -10;
  if (count >= 8) return -26;
  if (count >= 6) return -18;
  return -10;
}

function editRingCls(editMode: boolean, selected: boolean, mobile: boolean): string {
  if (!editMode) return "";
  if (selected) {
    return mobile
      ? "ring-2 ring-[var(--fork)] z-40"
      : "ring-2 ring-[var(--fork)] -translate-y-4 z-40";
  }
  return "hover:ring-2 hover:ring-[var(--fork)]";
}

function handCardLayout(
  isHero: boolean,
  scale: TabletopScale,
): { cardCls: string; rowCls: string } {
  if (scale === "desktop") {
    return isHero
      ? {
          cardCls:
            "w-[74px] h-[104px] sm:w-[84px] sm:h-[116px] translate-y-[32px] hover:-translate-y-4 hover:scale-110",
          rowCls: "items-end h-[70px] sm:h-[76px]",
        }
      : {
          cardCls:
            "w-[64px] h-[90px] sm:w-[70px] sm:h-[98px] -translate-y-[30px] hover:translate-y-3 hover:scale-110",
          rowCls: "items-start h-[54px] sm:h-[58px]",
        };
  }
  if (isHero) {
    return scale === "mobile-short"
      ? { cardCls: "w-[50px] h-[69px]", rowCls: "items-center h-[70px]" }
      : { cardCls: "w-[58px] h-[80px]", rowCls: "items-center h-[82px]" };
  }
  return scale === "mobile-short"
    ? { cardCls: "w-[44px] h-[60px]", rowCls: "items-center h-[62px]" }
    : { cardCls: "w-[52px] h-[72px]", rowCls: "items-center h-[74px]" };
}

export function HandFanView(props: {
  side: "hero" | "opp";
  cards: string[];
  hiddenCount: number;
  hidden: boolean;
  scale?: TabletopScale;
  editMode?: boolean;
  selectedEditKey?: string | null;
  onHover: (info: HoverCardInfo | null) => void;
  onPinCard?: (info: HoverCardInfo) => void;
  onSelectCardForEdit?: (target: HandCardEditTarget) => void;
}): ReactNode {
  const scale = props.scale ?? "desktop";
  const mobile = scale !== "desktop";
  const count = props.hidden ? props.hiddenCount : props.cards.length;
  if (count === 0) {
    return (
      <div className="h-[26px] flex items-center justify-center text-[10.5px] italic text-[var(--text-3)]">
        No cards in hand
      </div>
    );
  }
  if (props.hidden) {
    const overlap = handOverlapPx(count, true, mobile);
    const backs = Array.from({ length: count }, (_, idx) => (
      <div
        key={`h-${idx + 1}`}
        style={idx > 0 ? { marginLeft: overlap } : undefined}
        title="Hidden in Hero POV"
        className={`card-back rounded-[5px] shrink-0 ${mobile ? "w-[32px] h-[44px]" : "w-[42px] h-[58px] -translate-y-[24px]"}`}
      />
    ));
    if (mobile) {
      return (
        <div className="w-full min-w-0 overflow-x-auto no-scrollbar">
          <div className="w-max mx-auto flex items-center h-[44px] px-1">{backs}</div>
        </div>
      );
    }
    return <div className="flex items-start justify-center h-[34px] overflow-visible">{backs}</div>;
  }
  const isHero = props.side === "hero";
  const overlap = handOverlapPx(count, !isHero, mobile);
  const { cardCls, rowCls } = handCardLayout(isHero, scale);
  const buttons = keyByOccurrence(props.cards).map((entry) => {
    const editKey = `${props.side}:${entry.key}`;
    return (
      <button
        key={entry.key}
        type="button"
        aria-label={props.editMode ? `Edit ${entry.name} in hand` : `Inspect ${entry.name}`}
        style={entry.order > 0 ? { marginLeft: overlap } : undefined}
        onMouseEnter={(e): void =>
          props.onHover({ name: entry.name, anchorX: e.clientX, anchorY: e.clientY })
        }
        onMouseLeave={(): void => props.onHover(null)}
        onClick={(e): void => {
          if (props.editMode && props.onSelectCardForEdit) {
            const r = e.currentTarget.getBoundingClientRect();
            props.onHover(null);
            props.onSelectCardForEdit({
              side: props.side,
              cardName: entry.name,
              key: editKey,
              anchorX: r.left + r.width / 2,
              anchorY: r.top,
            });
            return;
          }
          props.onPinCard?.({ name: entry.name, anchorX: e.clientX, anchorY: e.clientY });
        }}
        className={`relative p-0 ${cardCls} rounded-[7px] overflow-hidden border border-black/45 bg-[#1c1917] shrink-0 shadow-[var(--shadow-md)] cursor-pointer transition-all duration-150 hover:z-40 ${editRingCls(Boolean(props.editMode), props.selectedEditKey === editKey, mobile)}`}
      >
        <img
          src={cardImageUrl(entry.name)}
          alt={entry.name}
          className="w-full h-full object-cover object-top"
        />
      </button>
    );
  });
  if (mobile) {
    return (
      <div className="w-full min-w-0 overflow-x-auto no-scrollbar">
        <div className={`w-max mx-auto flex px-1 ${rowCls}`}>{buttons}</div>
      </div>
    );
  }
  return <div className={`flex justify-center overflow-visible ${rowCls}`}>{buttons}</div>;
}

/** Responsive tabletop MTG zone composites, scale helpers, stack popover, and direct-on-board edit popovers. */

import { Plus, Repeat2, Trash2, X } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/Primitives";
import { cardImageUrl } from "@/lib/api";
import type {
  ForkEdit,
  LandPile,
  PermanentState,
  PlayerState,
  PovMode,
  StackItem,
} from "@/lib/types";

export interface HoverCardInfo {
  name: string;
  pt?: string;
  damage?: number;
  counters?: number;
  tapped?: boolean;
  token?: boolean;
  sick?: boolean;
  attacking?: boolean;
  blocking?: boolean;
  blockingName?: string;
  copyIndex?: number | null;
  count?: number;
  anchorX?: number;
  anchorY?: number;
}

export interface DrawerTarget {
  side: "hero" | "opp";
  zone: "library" | "graveyard" | "exile";
}

export interface HandCardEditTarget {
  side: "hero" | "opp";
  cardName: string;
  key: string;
  anchorX: number;
  anchorY: number;
}

function keyByOccurrence(names: string[]): Array<{ key: string; name: string; order: number }> {
  const seen = new Map<string, number>();
  return names.map((name, order) => {
    const count = (seen.get(name) ?? 0) + 1;
    seen.set(name, count);
    return { key: `${name}#${count}`, name, order };
  });
}

function chipCls(active: boolean): string {
  return active
    ? "px-2 py-0.5 rounded-sm cursor-pointer bg-[var(--fork-soft)] text-[var(--fork)]"
    : "px-2 py-0.5 rounded-sm cursor-pointer text-[var(--text-3)]";
}

export type TabletopScale = "desktop" | "mobile-tall" | "mobile-short";

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

function clampPopoverPos(args: {
  anchorX: number;
  anchorY: number;
  width: number;
  height: number;
}): { left: number; top: number } {
  if (typeof window === "undefined") return { left: 16, top: 64 };
  const pad = 12;
  const left = Math.max(
    pad,
    Math.min(window.innerWidth - args.width - pad, args.anchorX - args.width / 2),
  );
  const above = args.anchorY - args.height - 10;
  const top =
    above >= pad ? above : Math.min(window.innerHeight - args.height - pad, args.anchorY + 24);
  return { left, top: Math.max(pad, top) };
}

function CatalogCardPicker(props: {
  cards: string[];
  excludeName?: string;
  excludeSet?: ReadonlySet<string>;
  placeholder: string;
  onPick: (cardName: string) => void;
}): ReactNode {
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const matches = props.cards
    .filter(
      (n) =>
        n !== props.excludeName &&
        !props.excludeSet?.has(n) &&
        (needle === "" || n.toLowerCase().includes(needle)),
    )
    .slice(0, 28);
  return (
    <div className="flex flex-col gap-1.5">
      <input
        type="search"
        value={query}
        onChange={(e): void => setQuery(e.target.value)}
        placeholder={props.placeholder}
        aria-label={props.placeholder}
        className="w-full h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-[11px]"
      />
      <div className="max-h-[152px] overflow-y-auto flex flex-col gap-1">
        {matches.length === 0 ? (
          <div className="py-2 text-center text-[10.5px] text-[var(--text-3)]">
            No matching cards.
          </div>
        ) : (
          matches.map((name) => (
            <button
              key={name}
              type="button"
              onClick={(): void => props.onPick(name)}
              className="flex items-center gap-1.5 rounded px-1.5 py-1 text-left text-[10.5px] hover:bg-[var(--fork-soft)] cursor-pointer"
            >
              <img
                src={cardImageUrl(name)}
                alt={name}
                className="w-[18px] h-[25px] rounded-[2px] object-cover shrink-0"
              />
              <span className="truncate font-medium">{name}</span>
            </button>
          ))
        )}
      </div>
    </div>
  );
}

function PriorityNotice(props: {
  text: string;
  step: number | null;
  onJump: (step: number) => void;
}): ReactNode {
  return (
    <div className="flex flex-col gap-1.5 text-[10.5px] text-[var(--text-2)]">
      <span>{props.text}</span>
      {props.step === null ? null : (
        <Button size="sm" variant="fork" onClick={(): void => props.onJump(props.step ?? 0)}>
          Jump to Priority #{props.step}
        </Button>
      )}
    </div>
  );
}

export function HandCardEditPopover(props: {
  target: HandCardEditTarget;
  seat: 0 | 1;
  canEditBoard: boolean;
  nearestPriorityStep: number | null;
  cardCatalog: string[];
  onEdit: (edit: ForkEdit) => void;
  onJumpToStep: (step: number) => void;
  onClose: () => void;
}): ReactNode {
  const [swapOpen, setSwapOpen] = useState(false);
  const pos = clampPopoverPos({
    anchorX: props.target.anchorX,
    anchorY: props.target.anchorY,
    width: 260,
    height: swapOpen ? 300 : 144,
  });
  return (
    <dialog
      open
      aria-label={`Edit ${props.target.cardName}`}
      style={{ left: pos.left, top: pos.top, width: 260 }}
      className="fixed z-50 m-0 rounded-[10px] border border-[var(--fork)] bg-[var(--surface)] text-[var(--text)] shadow-[var(--shadow-lg)] p-2.5 flex flex-col gap-2"
    >
      <div className="flex items-center justify-between gap-2 pb-1.5 border-b border-[var(--border)]">
        <span className="text-[11px] font-bold truncate">{props.target.cardName}</span>
        <button
          type="button"
          onClick={props.onClose}
          aria-label="Close"
          className="p-0.5 text-[var(--text-3)] hover:text-[var(--text)] cursor-pointer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
      {props.canEditBoard ? (
        <>
          <button
            type="button"
            onClick={(): void => {
              props.onEdit({
                op: "remove_hand_card",
                seat: props.seat,
                card: props.target.cardName,
              });
              props.onClose();
            }}
            className="flex items-center gap-2 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2.5 py-1.5 text-[11px] font-medium text-[var(--loss)] hover:bg-[var(--loss-soft)] cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Remove from hand
          </button>
          <button
            type="button"
            onClick={(): void => setSwapOpen((v) => !v)}
            className={`flex items-center justify-between rounded-md border px-2.5 py-1.5 text-[11px] font-medium cursor-pointer ${swapOpen ? "border-[var(--fork)] bg-[var(--fork-soft)] text-[var(--fork)]" : "border-[var(--border)] bg-[var(--bg)]"}`}
          >
            <span className="flex items-center gap-2">
              <Repeat2 className="w-3.5 h-3.5" />
              Swap card…
            </span>
            <span>{swapOpen ? "▴" : "▾"}</span>
          </button>
          {swapOpen ? (
            <CatalogCardPicker
              cards={props.cardCatalog}
              excludeName={props.target.cardName}
              placeholder="Search replacement card…"
              onPick={(rep): void => {
                props.onEdit({
                  op: "swap_hand_card",
                  seat: props.seat,
                  card: props.target.cardName,
                  replacement_card: rep,
                });
                props.onClose();
              }}
            />
          ) : null}
        </>
      ) : (
        <PriorityNotice
          text="Hand edits require a Priority step."
          step={props.nearestPriorityStep}
          onJump={(s): void => {
            props.onJumpToStep(s);
            props.onClose();
          }}
        />
      )}
    </dialog>
  );
}

export function AddCardPopover(props: {
  heroSeat: number;
  canEditBoard: boolean;
  nearestPriorityStep: number | null;
  nonPermanentNames: ReadonlySet<string>;
  cardCatalog: string[];
  onAddEdit: (edit: ForkEdit) => void;
  onJumpToStep: (step: number) => void;
  onClose: () => void;
}): ReactNode {
  const [side, setSide] = useState<"hero" | "opp">("hero");
  const [zone, setZone] = useState<"add_hand_card" | "add_battlefield">("add_hand_card");
  const heroNum: 0 | 1 = props.heroSeat === 1 ? 1 : 0;
  const oppNum: 0 | 1 = heroNum === 1 ? 0 : 1;
  const seat: 0 | 1 = side === "hero" ? heroNum : oppNum;
  return (
    <dialog
      open
      aria-label="Add card to board or hand"
      className="absolute right-2 top-10 left-auto z-40 m-0 w-[276px] max-w-[calc(100%-16px)] rounded-[10px] border border-[var(--fork)] bg-[var(--surface)] text-[var(--text)] shadow-[var(--shadow-lg)] p-2.5 flex flex-col gap-2"
    >
      <div className="flex items-center justify-between pb-1.5 border-b border-[var(--border)]">
        <span className="inline-flex items-center gap-1 text-[11px] font-bold text-[var(--fork)]">
          <Plus className="w-3.5 h-3.5" />
          Add card
        </span>
        <button
          type="button"
          onClick={props.onClose}
          aria-label="Close"
          className="p-0.5 text-[var(--text-3)] hover:text-[var(--text)] cursor-pointer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
      {props.canEditBoard ? (
        <>
          <div className="flex items-center justify-between gap-1 text-[10px] font-semibold">
            <div className="inline-flex rounded border border-[var(--border)] bg-[var(--bg)] p-0.5">
              <button
                type="button"
                onClick={(): void => setSide("hero")}
                className={chipCls(side === "hero")}
              >
                Hero
              </button>
              <button
                type="button"
                onClick={(): void => setSide("opp")}
                className={chipCls(side === "opp")}
              >
                Opp
              </button>
            </div>
            <div className="inline-flex rounded border border-[var(--border)] bg-[var(--bg)] p-0.5">
              <button
                type="button"
                onClick={(): void => setZone("add_hand_card")}
                className={chipCls(zone === "add_hand_card")}
              >
                + Hand
              </button>
              <button
                type="button"
                onClick={(): void => setZone("add_battlefield")}
                className={chipCls(zone === "add_battlefield")}
              >
                + Board
              </button>
            </div>
          </div>
          <CatalogCardPicker
            cards={props.cardCatalog}
            excludeSet={zone === "add_battlefield" ? props.nonPermanentNames : undefined}
            placeholder="Search cards…"
            onPick={(card): void => {
              props.onAddEdit({ op: zone, seat, card });
              props.onClose();
            }}
          />
        </>
      ) : (
        <PriorityNotice
          text="Card injections apply on Priority steps."
          step={props.nearestPriorityStep}
          onJump={(s): void => {
            props.onJumpToStep(s);
            props.onClose();
          }}
        />
      )}
    </dialog>
  );
}

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

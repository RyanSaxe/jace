/** Direct-on-board Tabletop fork edit popovers for hand card removal/swap and adding cards to hand or battlefield. */

import { Plus, Repeat2, Trash2, X } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/Primitives";
import { cardImageUrl } from "@/lib/api";
import type { ForkEdit } from "@/lib/types";
import type { HandCardEditTarget } from "./tabletopTypes";

function chipCls(active: boolean): string {
  return active
    ? "px-2 py-0.5 rounded-sm cursor-pointer bg-[var(--fork-soft)] text-[var(--fork)]"
    : "px-2 py-0.5 rounded-sm cursor-pointer text-[var(--text-3)]";
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

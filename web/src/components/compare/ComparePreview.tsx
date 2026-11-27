/** Docked position preview pane and legal-action probability comparison bars. */

import { ArrowRight } from "lucide-react";
import type { ReactNode } from "react";
import { Badge, Button } from "@/components/ui/Primitives";
import { cardImageUrl } from "@/lib/api";
import type { ComparePositionRow } from "@/lib/types";
import { ASIDE, BOX, LBL, humanizeToken, signedDelta } from "./compareUtils";

function LegalMoveBars(props: { position: ComparePositionRow }): ReactNode {
  const rows = props.position.legal_actions
    .map((action, i) => ({
      action,
      a: props.position.probs_a[i] ?? 0,
      b: props.position.probs_b[i] ?? 0,
    }))
    .toSorted((x, y) => Math.max(y.a, y.b) - Math.max(x.a, x.b));
  const [maxA, maxB] = [
    Math.max(-1, ...rows.map((r) => r.a)),
    Math.max(-1, ...rows.map((r) => r.b)),
  ];
  if (rows.length === 0) {
    return (
      <div className={`${BOX} p-3 text-xs text-[var(--text-3)]`}>
        No legal action probabilities recorded.
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <div className={`flex items-center justify-between ${LBL}`}>
        <span>Legal Moves ({rows.length})</span>
        <span className="inline-flex gap-2 normal-case font-medium">
          <span className="text-[var(--accent)]">● A</span>
          <span className="text-[var(--warn)]">● B</span>
        </span>
      </div>
      {rows.map(({ action, a, b }) => {
        const bars: ReadonlyArray<[string, number, string, boolean]> = [
          ["A", a, "var(--accent)", a > 0 && a === maxA],
          ["B", b, "var(--warn)", b > 0 && b === maxB],
        ];
        return (
          <div key={`${action.index}-${action.label}`} className={`${BOX} p-2 flex flex-col gap-1`}>
            <div className="text-[11.5px] font-medium leading-snug break-words">{action.label}</div>
            {bars.map(([lbl, prob, clr, top]) => {
              const pct = Math.max(0, Math.min(100, prob * 100)).toFixed(1);
              return (
                <div
                  key={lbl}
                  className="grid grid-cols-[14px_1fr_40px] items-center gap-1.5 text-[10.5px]"
                >
                  <span className="font-semibold" style={{ color: clr }}>
                    {lbl}
                  </span>
                  <div className="h-2 rounded-full bg-[var(--surface-2)] overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${pct}%`, backgroundColor: clr, opacity: top ? 1 : 0.72 }}
                    />
                  </div>
                  <span
                    className={`tabular text-right ${top ? "font-semibold" : "text-[var(--text-2)]"}`}
                  >
                    {pct}%
                  </span>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

export function PositionPreviewPane(props: {
  position: ComparePositionRow | null;
  onOpen: (pos: ComparePositionRow) => void;
}): ReactNode {
  const pos = props.position;
  if (!pos) {
    return (
      <aside
        className={`${ASIDE} p-4 flex items-center justify-center text-center text-xs text-[var(--text-3)]`}
      >
        Select a sampled position from the scatter plot or diff table to inspect legal moves and
        jump into Tabletop.
      </aside>
    );
  }
  return (
    <aside className={`${ASIDE} flex flex-col min-h-0`}>
      <div className="flex-1 lg:overflow-y-auto p-3.5 flex flex-col gap-3">
        <div className="flex flex-col gap-1.5 pb-3 border-b border-[var(--border)]">
          <div className="flex items-center justify-between gap-2">
            <span className="tabular text-xs font-semibold">
              Turn {pos.turn} · Step #{pos.step}
            </span>
            <Badge tone={pos.move_changed ? "accent" : "neutral"}>
              {pos.move_changed ? "Move changed" : "Same move"}
            </Badge>
          </div>
          <div className="text-[11px] text-[var(--text-2)]">
            <strong className="font-medium text-[var(--text)]">
              {humanizeToken(pos.decision_kind)}
            </strong>
            <span className="tabular">
              {" "}
              · Seed #{pos.seed} · u{pos.iteration}
            </span>
          </div>
          <div className={`flex items-center justify-between text-[11px] ${BOX} px-2.5 py-1.5`}>
            <span className="text-[var(--text-2)]">
              Hero <strong className="tabular text-[var(--text)]">{pos.hero_life}</strong> vs Opp{" "}
              <strong className="tabular text-[var(--text)]">{pos.opp_life}</strong>
            </span>
            <span className="tabular text-[10.5px] text-[var(--text-3)]">
              TV {pos.tv.toFixed(2)}
            </span>
          </div>
          <div
            className={`flex items-center justify-between text-[11px] ${BOX} px-2.5 py-1.5 tabular`}
          >
            <span>
              <span className="text-[var(--accent)] font-semibold">A</span>{" "}
              {(pos.win_a * 100).toFixed(1)}%
            </span>
            <span>
              <span className="text-[var(--warn)] font-semibold">B</span>{" "}
              {(pos.win_b * 100).toFixed(1)}%{" "}
              <span className="text-[10px] text-[var(--text-3)]">
                ({signedDelta(pos.win_delta)})
              </span>
            </span>
          </div>
          <div className="text-[10.5px] text-[var(--text-3)] truncate" title={pos.run_id}>
            {pos.run_id}
          </div>
        </div>
        {pos.key_cards.length > 0 ? (
          <div className="flex flex-col gap-1.5">
            <span className={LBL}>Key Cards on Board</span>
            <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5">
              {pos.key_cards.slice(0, 5).map((name) => (
                <div
                  key={name}
                  title={name}
                  className="relative w-[54px] aspect-[5/7] shrink-0 rounded overflow-hidden border border-[var(--border)] bg-[var(--surface-2)] flex items-center justify-center p-1 text-center text-[9px] text-[var(--text-2)]"
                >
                  <span className="line-clamp-3">{name}</span>
                  <img
                    src={cardImageUrl(name)}
                    alt={name}
                    loading="lazy"
                    onError={(e): void => {
                      e.currentTarget.style.display = "none";
                    }}
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                </div>
              ))}
            </div>
          </div>
        ) : null}
        <LegalMoveBars position={pos} />
      </div>
      <div className="p-3 border-t border-[var(--border)] shrink-0">
        <Button
          variant="primary"
          size="md"
          className="w-full"
          onClick={(): void => props.onOpen(pos)}
        >
          Open in Tabletop (A vs B) <ArrowRight className="w-3.5 h-3.5" />
        </Button>
      </div>
    </aside>
  );
}

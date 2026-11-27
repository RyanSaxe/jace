/** Game preview drawer with win-probability curve, key card thumbnails, and Tabletop jump actions. */

import { ArrowRight } from "lucide-react";
import type { ReactNode } from "react";
import { Badge, Button } from "@/components/ui/Primitives";
import { cardImageUrl } from "@/lib/api";
import type { GameOverview, GameSummary, GameTarget } from "@/lib/types";
import { formatCalibrationSummary, orderedPreviewCards, resultTone } from "./gamesUtils";

function PreviewCurve(props: {
  overview: GameOverview | null;
  loading: boolean;
  turningStep: number;
}): ReactNode {
  if (props.loading) {
    return (
      <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-3 text-center text-[11px] text-[var(--text-3)]">
        Computing win curve…
      </div>
    );
  }
  const overview = props.overview;
  if (!overview || overview.timeline.length < 2) return null;
  const pts = overview.timeline;
  const w = 310;
  const h = 76;
  const padX = 10;
  const padY = 10;
  const toX = (i: number): number => padX + (i / (pts.length - 1)) * (w - padX * 2);
  const toY = (v: number): number => h - padY - Math.max(0, Math.min(1, v)) * (h - padY * 2);
  const poly = pts.map((p, i) => `${toX(i).toFixed(1)},${toY(p.eval_win).toFixed(1)}`).join(" ");
  const turnIdx = pts.reduce((best, p, i) => {
    const bestDiff = Math.abs((pts[best]?.step ?? 0) - props.turningStep);
    return Math.abs(p.step - props.turningStep) < bestDiff ? i : best;
  }, 0);
  const turnPt = pts[turnIdx];
  const sourceLabel = overview.eval_source === "model" ? "Model Win%" : "Est. Win%";

  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-2">
      <div className="flex items-center justify-between text-[10.5px] text-[var(--text-3)] mb-1">
        <span>{sourceLabel} by Turn</span>
        <span className="tabular font-medium text-[var(--text-2)]">
          Turning pt: T{turnPt?.turn ?? 0} (#{props.turningStep})
        </span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[70px] block">
        <line
          x1={padX}
          y1={h / 2}
          x2={w - padX}
          y2={h / 2}
          stroke="var(--border-strong)"
          strokeDasharray="3 3"
        />
        <polyline
          points={poly}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
          strokeLinejoin="round"
        />
        {turnPt ? (
          <circle
            cx={toX(turnIdx)}
            cy={toY(turnPt.eval_win)}
            r="4"
            fill="var(--loss)"
            stroke="var(--surface)"
            strokeWidth="1.5"
          >
            <title>{`Turn ${turnPt.turn} (step #${turnPt.step}): ${Math.round(turnPt.eval_win * 100)}%`}</title>
          </circle>
        ) : null}
      </svg>
    </div>
  );
}

export function GamePreviewDrawer(props: {
  game: GameSummary | null;
  overview: GameOverview | null;
  loadingOverview: boolean;
  highlightCard: string;
  onOpenTabletop: (target: GameTarget) => void;
}): ReactNode {
  const game = props.game;
  if (!game) {
    return (
      <aside
        aria-label="Game preview"
        className="order-first lg:order-last w-full lg:w-[340px] shrink-0 border-b lg:border-b-0 lg:border-l border-[var(--border)] bg-[var(--surface)] p-4 text-xs text-[var(--text-3)]"
      >
        Select a game (or press j / k) to inspect its turning point and open the Tabletop.
      </aside>
    );
  }
  const turningStep = props.overview?.turning_point_step ?? game.turning_point_step ?? 0;
  const cal = formatCalibrationSummary(game);
  const swingPct = Math.round((game.max_swing ?? 0) * 100);

  return (
    <aside
      aria-label="Game preview"
      className="order-first lg:order-last w-full lg:w-[340px] shrink-0 border-b lg:border-b-0 lg:border-l border-[var(--border)] bg-[var(--surface)] p-3.5 lg:overflow-y-auto flex flex-col gap-3"
    >
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <Badge tone={resultTone(game.result)}>{game.result.toUpperCase()}</Badge>
            <span className="tabular text-xs font-semibold">Seed #{game.seed}</span>
          </div>
          <span className="tabular text-[11px] text-[var(--text-3)]">
            u{game.iteration} · {game.turns}T
          </span>
        </div>
        <div className="text-[11px] text-[var(--text-2)] truncate">
          {game.run_id.split("/").pop()} · {game.hero_colors} vs {game.opp_colors}
        </div>
      </div>

      <div className="flex items-center justify-between rounded-md border border-[var(--border)] bg-[var(--bg)] px-2.5 py-1.5 text-[11px]">
        <span className="font-semibold tabular">{cal.text}</span>
        {swingPct > 0 ? (
          <span className="tabular text-[var(--text-3)]">Max swing {swingPct}%</span>
        ) : null}
      </div>

      <PreviewCurve
        overview={props.overview}
        loading={props.loadingOverview}
        turningStep={turningStep}
      />

      <div className="grid grid-cols-4 sm:grid-cols-8 lg:grid-cols-4 gap-1.5">
        {orderedPreviewCards(game, props.highlightCard).map((cardName) => (
          <div
            key={cardName}
            className="rounded overflow-hidden border border-[var(--border)] bg-[#18181b] aspect-[5/7]"
            title={cardName}
          >
            <img
              src={cardImageUrl(cardName)}
              alt={cardName}
              loading="lazy"
              className="w-full h-full object-cover block"
            />
          </div>
        ))}
      </div>

      <div className="flex items-center gap-1.5 pt-0.5">
        <Button
          variant="primary"
          size="md"
          className="flex-1"
          onClick={(): void =>
            props.onOpenTabletop({
              run_id: game.run_id,
              iteration: game.iteration,
              game_idx: game.game_idx,
              step: turningStep,
              seed: game.seed,
            })
          }
        >
          Turning Point (#{turningStep})
          <ArrowRight className="w-3.5 h-3.5" />
        </Button>
        <Button
          size="md"
          onClick={(): void =>
            props.onOpenTabletop({
              run_id: game.run_id,
              iteration: game.iteration,
              game_idx: game.game_idx,
              step: 0,
              seed: game.seed,
            })
          }
        >
          Turn 1
        </Button>
      </div>
    </aside>
  );
}

/** Linear-style Games & Moments Explorer backed by DuckDB/Parquet faceted search and right board preview. */

import { ArrowRight, Layers, Search } from "lucide-react";
import { type ReactNode, useCallback, useState } from "react";
import { Badge, Button } from "@/components/ui/Primitives";
import { useGameOverview, useGamesSearch, useKeyboardShortcut } from "@/hooks/useStudioData";
import { cardImageUrl } from "@/lib/api";
import type { GameOverview, GameSummary, GameTarget, RunSummary } from "@/lib/types";

function resultTone(result: string): "win" | "loss" | "neutral" {
  if (result === "win") return "win";
  if (result === "loss") return "loss";
  return "neutral";
}

function PreviewCurve(props: { overview: GameOverview | null; loading: boolean }): ReactNode {
  if (props.loading) {
    return (
      <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-4 text-center text-[11px] text-[var(--text-3)]">
        Replaying turn trajectory…
      </div>
    );
  }
  const overview = props.overview;
  if (!overview || overview.timeline.length === 0) return null;
  const pts = overview.timeline;
  const w = 310;
  const h = 68;
  const poly = pts
    .map((p, i) => {
      const x = pts.length > 1 ? 8 + (i / (pts.length - 1)) * (w - 16) : w / 2;
      const y = h - 10 - p.eval_win * (h - 20);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-2">
      <div className="flex justify-between text-[10.5px] text-[var(--text-3)] mb-1">
        <span>Turn-grouped Critic Win%</span>
        <span className="tabular">Peak step #{overview.peak_step}</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-16 block">
        <line
          x1="8"
          y1={h / 2}
          x2={w - 8}
          y2={h / 2}
          stroke="var(--border)"
          strokeDasharray="3 3"
        />
        <polyline
          points={poly}
          fill="none"
          stroke="var(--win)"
          strokeWidth="2"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

function GamePreviewDrawer(props: {
  game: GameSummary | null;
  overview: GameOverview | null;
  loadingOverview: boolean;
  onOpenTabletop: (target: GameTarget) => void;
}): ReactNode {
  const game = props.game;
  if (!game) {
    return (
      <aside className="w-full lg:w-[350px] shrink-0 border-t lg:border-t-0 lg:border-l border-[var(--border)] bg-[var(--surface)] p-4 text-xs text-[var(--text-3)]">
        Select a game row (or press j / k) to preview its curve and cards.
      </aside>
    );
  }
  const peakStep = props.overview?.peak_step ?? Math.max(0, Math.floor(game.steps * 0.7));
  const lastTurn = props.overview?.timeline[props.overview.timeline.length - 1];

  return (
    <aside className="w-full lg:w-[350px] shrink-0 border-t lg:border-t-0 lg:border-l border-[var(--border)] bg-[var(--surface)] p-3.5 overflow-y-auto flex flex-col gap-3">
      <div>
        <div className="flex items-center justify-between gap-2 mb-1">
          <Badge tone={resultTone(game.result)}>{game.result.toUpperCase()}</Badge>
          <span className="tabular text-[11px] text-[var(--text-3)]">
            {game.set_code} · Seed #{game.seed}
          </span>
        </div>
        <h3 className="text-sm font-semibold tracking-tight truncate">{game.run_id}</h3>
        <div className="text-[11px] text-[var(--text-2)] tabular mt-0.5">
          iter-{String(game.iteration).padStart(4, "0")} · Game #{game.game_idx} · {game.steps}{" "}
          steps
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-2">
          <div className="text-[10px] uppercase text-[var(--text-3)]">
            Hero ({game.hero_colors})
          </div>
          <div className="font-semibold truncate">{game.hero_name}</div>
          {lastTurn ? (
            <div className="tabular text-[11px] text-[var(--text-2)]">
              End: {lastTurn.hero_life} life · {lastTurn.hero_perms} perms
            </div>
          ) : null}
        </div>
        <div className="rounded-md border border-[var(--border)] bg-[var(--bg)] p-2">
          <div className="text-[10px] uppercase text-[var(--text-3)]">Opp ({game.opp_colors})</div>
          <div className="font-semibold truncate">{game.opp_name}</div>
          {lastTurn ? (
            <div className="tabular text-[11px] text-[var(--text-2)]">
              End: {lastTurn.opp_life} life · {lastTurn.opp_perms} perms
            </div>
          ) : null}
        </div>
      </div>

      <PreviewCurve overview={props.overview} loading={props.loadingOverview} />

      <div>
        <div className="text-[10.5px] font-semibold uppercase tracking-wider text-[var(--text-3)] mb-1.5">
          Featured Cards in Matchup
        </div>
        <div className="grid grid-cols-4 gap-1.5">
          {game.key_cards.slice(0, 8).map((cardName) => (
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
      </div>

      <div className="flex flex-col gap-1.5 pt-1">
        <Button
          variant="primary"
          size="md"
          className="w-full"
          onClick={(): void =>
            props.onOpenTabletop({
              run_id: game.run_id,
              iteration: game.iteration,
              game_idx: game.game_idx,
              step: peakStep,
              seed: game.seed,
            })
          }
        >
          Open Tabletop at Peak Step (#{peakStep})
          <ArrowRight className="w-3.5 h-3.5" />
        </Button>
        <Button
          size="sm"
          className="w-full"
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
          Start from Turn 1 (Step 0)
        </Button>
      </div>
    </aside>
  );
}

function GamesTableBody(props: {
  games: GameSummary[];
  loading: boolean;
  selectedIdx: number;
  peakStep: number;
  onSelect: (idx: number) => void;
  onOpenTabletop: (target: GameTarget) => void;
}): ReactNode {
  if (props.loading && props.games.length === 0) {
    return (
      <tr>
        <td colSpan={6} className="py-8 text-center text-[var(--text-3)]">
          Querying DuckDB Parquet index…
        </td>
      </tr>
    );
  }
  if (props.games.length === 0) {
    return (
      <tr>
        <td colSpan={6} className="py-8 text-center text-[var(--text-3)]">
          No games match those filters.
        </td>
      </tr>
    );
  }
  return props.games.map((g, idx) => (
    <tr
      key={`${g.run_id}-${g.iteration}-${g.game_idx}`}
      onClick={(): void => props.onSelect(idx)}
      onDoubleClick={(): void =>
        props.onOpenTabletop({
          run_id: g.run_id,
          iteration: g.iteration,
          game_idx: g.game_idx,
          step: props.peakStep,
          seed: g.seed,
        })
      }
      className="border-b border-[var(--border)] last:border-b-0 cursor-pointer hover:bg-[var(--surface-2)]"
      style={idx === props.selectedIdx ? { background: "var(--accent-soft)" } : undefined}
    >
      <td className="py-2 px-3">
        <div className="font-semibold tabular">
          #{g.seed}{" "}
          <span className="font-normal text-[var(--text-2)]">· {g.run_id.split("/").pop()}</span>
        </div>
      </td>
      <td className="py-2 px-2 tabular text-[var(--text-2)]">u{g.iteration}</td>
      <td className="py-2 px-2">
        <Badge tone={resultTone(g.result)}>{g.result.toUpperCase()}</Badge>
      </td>
      <td className="py-2 px-2 tabular text-[var(--text-2)]">
        {g.hero_colors} vs {g.opp_colors}
      </td>
      <td className="py-2 px-2 tabular text-right font-semibold">{g.steps}</td>
      <td className="py-2 px-3 text-[var(--text-2)] max-w-[240px] truncate">
        {g.key_cards.slice(0, 3).join(", ")}
      </td>
    </tr>
  ));
}

function nonEmpty(value: string): string | undefined {
  return value === "" ? undefined : value;
}

function gameIdentity(game: GameSummary | null): [string, number, number] {
  if (!game) return ["", -1, -1];
  return [game.run_id, game.iteration, game.game_idx];
}

export function GamesExplorerView(props: {
  runs: RunSummary[];
  initialRunId: string;
  onOpenTabletop: (target: GameTarget) => void;
}): ReactNode {
  const [runId, setRunId] = useState(props.initialRunId);
  const [result, setResult] = useState("");
  const [card, setCard] = useState("");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("steps_desc");
  const [selectedIdx, setSelectedIdx] = useState(0);

  const { games, total, cards, loading, error } = useGamesSearch({
    runId: nonEmpty(runId),
    result: nonEmpty(result),
    card: nonEmpty(card),
    search: nonEmpty(search),
    sort,
    limit: 120,
  });

  const safeIdx = Math.min(selectedIdx, Math.max(0, games.length - 1));
  const activeGame = games[safeIdx] ?? null;
  const [ovRunId, ovIter, ovGameIdx] = gameIdentity(activeGame);
  const { overview, loading: loadingOverview } = useGameOverview(ovRunId, ovIter, ovGameIdx);

  const handleKey = useCallback(
    (event: KeyboardEvent): void => {
      if (games.length === 0) return;
      if (event.key === "j" || event.key === "ArrowDown") {
        event.preventDefault();
        setSelectedIdx((i) => Math.min(games.length - 1, i + 1));
      } else if (event.key === "k" || event.key === "ArrowUp") {
        event.preventDefault();
        setSelectedIdx((i) => Math.max(0, i - 1));
      } else if (event.key === "Enter" && activeGame) {
        event.preventDefault();
        props.onOpenTabletop({
          run_id: activeGame.run_id,
          iteration: activeGame.iteration,
          game_idx: activeGame.game_idx,
          step: overview?.peak_step ?? 0,
          seed: activeGame.seed,
        });
      }
    },
    [games.length, activeGame, overview, props],
  );
  useKeyboardShortcut(handleKey);

  return (
    <div className="flex flex-col lg:flex-row h-full min-h-0 overflow-y-auto lg:overflow-hidden">
      <div className="flex-none lg:flex-1 min-w-0 flex flex-col lg:min-h-0 lg:overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 border-b border-[var(--border)] bg-[var(--surface)]">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-[var(--accent)]" />
            <h2 className="text-sm font-semibold tracking-tight">Games &amp; Moments</h2>
            <Badge>{total} indexed</Badge>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-3)]" />
              <input
                type="search"
                value={search}
                onChange={(e): void => {
                  setSearch(e.target.value);
                  setSelectedIdx(0);
                }}
                placeholder="Search card, seed, run…"
                aria-label="Search games"
                className="h-7 w-48 rounded-md border border-[var(--border)] bg-[var(--bg)] pl-7 pr-2 text-xs"
              />
            </div>

            <select
              value={runId}
              onChange={(e): void => {
                setRunId(e.target.value);
                setSelectedIdx(0);
              }}
              aria-label="Filter by run"
              className="h-7 max-w-[180px] rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs"
            >
              <option value="">All Runs</option>
              {props.runs
                .filter((r) => r.eval_files > 0)
                .map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.name}
                  </option>
                ))}
            </select>

            <select
              value={card}
              onChange={(e): void => {
                setCard(e.target.value);
                setSelectedIdx(0);
              }}
              aria-label="Filter by card"
              className="h-7 max-w-[165px] rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs"
            >
              <option value="">All Cards</option>
              {cards.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>

            <select
              value={result}
              onChange={(e): void => {
                setResult(e.target.value);
                setSelectedIdx(0);
              }}
              aria-label="Filter by outcome"
              className="h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs"
            >
              <option value="">Any Outcome</option>
              <option value="win">Hero Win</option>
              <option value="loss">Hero Loss</option>
              <option value="draw">Draw</option>
            </select>

            <select
              value={sort}
              onChange={(e): void => setSort(e.target.value)}
              aria-label="Sort games"
              className="h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs"
            >
              <option value="steps_desc">Longest Games</option>
              <option value="iter_desc">Latest Iter</option>
              <option value="turns_desc">Most Turns</option>
              <option value="seed_asc">Seed Asc</option>
            </select>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-3">
          {error ? (
            <div className="rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] p-3 text-xs text-[var(--loss)] mb-3">
              {error}
            </div>
          ) : null}
          <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-x-auto">
            <table className="w-full min-w-[620px] border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-[var(--border)] bg-[var(--surface-2)] text-[10.5px] uppercase tracking-wider text-[var(--text-3)]">
                  <th className="py-2 px-3 font-semibold">Seed / Run</th>
                  <th className="py-2 px-2 font-semibold">Iter</th>
                  <th className="py-2 px-2 font-semibold">Outcome</th>
                  <th className="py-2 px-2 font-semibold">Colors</th>
                  <th className="py-2 px-2 font-semibold text-right">Steps</th>
                  <th className="py-2 px-3 font-semibold">Key Cards</th>
                </tr>
              </thead>
              <tbody>
                <GamesTableBody
                  games={games}
                  loading={loading}
                  selectedIdx={safeIdx}
                  peakStep={overview?.peak_step ?? 0}
                  onSelect={setSelectedIdx}
                  onOpenTabletop={props.onOpenTabletop}
                />
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <GamePreviewDrawer
        game={activeGame}
        overview={overview}
        loadingOverview={loadingOverview}
        onOpenTabletop={props.onOpenTabletop}
      />
    </div>
  );
}

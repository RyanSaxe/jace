/** Payload-driven Games & Moments Explorer with mistake/surprise search, dynamic facets, and clean preview drawer. */

import { ArrowRight, ChevronLeft, ChevronRight, Layers, Search, Sparkles } from "lucide-react";
import { type ReactNode, useCallback, useState } from "react";
import { Badge, Button } from "@/components/ui/Primitives";
import { useGameOverview, useGamesSearch, useKeyboardShortcut } from "@/hooks/useStudioData";
import { cardImageUrl } from "@/lib/api";
import type {
  GameOverview,
  GameSummary,
  GameTarget,
  GamesFilterState,
  RunSummary,
} from "@/lib/types";

const PAGE_SIZE = 50;

interface SearchPreset {
  id: string;
  label: string;
  patch: Partial<GamesFilterState>;
}

const SEARCH_PRESETS: ReadonlyArray<SearchPreset> = [
  {
    id: "all",
    label: "All Playable",
    patch: {
      runId: "",
      setCode: "",
      card: "",
      colors: "",
      search: "",
      result: "",
      minSurprise: null,
      minSwing: null,
      sort: "surprise_desc",
    },
  },
  {
    id: "throws",
    label: "High Win% → Loss",
    patch: { result: "loss", minSurprise: 0.55, minSwing: null, sort: "surprise_desc" },
  },
  {
    id: "comebacks",
    label: "Low Win% → Win",
    patch: { result: "win", minSurprise: 0.55, minSwing: null, sort: "surprise_desc" },
  },
  {
    id: "swings",
    label: "Wildest Swings",
    patch: { result: "", minSurprise: null, minSwing: 0.25, sort: "swing_desc" },
  },
];

function isUnfilteredAll(filters: GamesFilterState): boolean {
  const anyText = [
    filters.runId,
    filters.card,
    filters.colors,
    filters.setCode,
    filters.search,
    filters.result,
  ].some(Boolean);
  return !anyText && filters.minSurprise === null && filters.minSwing === null;
}

function activePresetId(filters: GamesFilterState): string {
  if (filters.result === "loss" && filters.minSurprise !== null) return "throws";
  if (filters.result === "win" && filters.minSurprise !== null) return "comebacks";
  if (filters.minSwing !== null && filters.sort === "swing_desc") return "swings";
  return isUnfilteredAll(filters) ? "all" : "";
}

function resultTone(result: string): "win" | "loss" | "neutral" {
  if (result === "win") return "win";
  if (result === "loss") return "loss";
  return "neutral";
}

function formatCalibrationSummary(g: GameSummary): {
  text: string;
  tone: "win" | "loss" | "neutral";
} {
  const maxW = Math.round((g.max_pred_win ?? 0.5) * 100);
  const minW = Math.round((g.min_pred_win ?? 0.5) * 100);
  if (g.result === "loss") {
    return { text: `Peak ${maxW}% → Loss`, tone: maxW >= 65 ? "loss" : "neutral" };
  }
  if (g.result === "win") {
    return { text: `Low ${minW}% → Win`, tone: minW <= 35 ? "win" : "neutral" };
  }
  return { text: `${minW}%–${maxW}%`, tone: "neutral" };
}

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

function orderedPreviewCards(game: GameSummary, highlightQuery: string): string[] {
  const q = highlightQuery.trim().toLowerCase();
  if (!q) return game.key_cards.slice(0, 8);
  const pool = game.cards_seen.length > 0 ? game.cards_seen : game.key_cards;
  const matching = pool.filter((c) => c.toLowerCase().includes(q));
  const seen = new Set(matching);
  const rest = game.key_cards.filter((c) => !seen.has(c));
  return [...matching, ...rest].slice(0, 8);
}

function GamePreviewDrawer(props: {
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

function GamesTableBody(props: {
  games: GameSummary[];
  loading: boolean;
  selectedIdx: number;
  onSelect: (idx: number) => void;
  onOpenTabletop: (target: GameTarget) => void;
}): ReactNode {
  if (props.loading && props.games.length === 0) {
    return (
      <tr>
        <td colSpan={6} className="py-8 text-center text-[var(--text-3)]">
          Indexing playable games…
        </td>
      </tr>
    );
  }
  if (props.games.length === 0) {
    return (
      <tr>
        <td colSpan={6} className="py-8 text-center text-[var(--text-3)]">
          No playable games match those filters.
        </td>
      </tr>
    );
  }
  return props.games.map((g, idx) => {
    const cal = formatCalibrationSummary(g);
    const swingPct = Math.round((g.max_swing ?? 0) * 100);
    return (
      <tr
        key={`${g.run_id}-${g.iteration}-${g.game_idx}`}
        onClick={(): void => props.onSelect(idx)}
        onDoubleClick={(): void =>
          props.onOpenTabletop({
            run_id: g.run_id,
            iteration: g.iteration,
            game_idx: g.game_idx,
            step: g.turning_point_step ?? 0,
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
          <div className="text-[10px] text-[var(--text-3)] tabular">
            u{g.iteration} · {g.hero_colors} vs {g.opp_colors}
          </div>
        </td>
        <td className="py-2 px-2">
          <Badge tone={resultTone(g.result)}>{g.result.toUpperCase()}</Badge>
        </td>
        <td className="py-2 px-2 tabular">
          <Badge tone={cal.tone}>{cal.text}</Badge>
        </td>
        <td className="py-2 px-2 tabular text-right text-[var(--text-2)]">
          {swingPct > 0 ? `${swingPct}%` : "—"}
        </td>
        <td className="py-2 px-2 tabular text-right text-[var(--text-2)]">
          {g.turns}T / {g.steps}
        </td>
        <td className="py-2 px-3 text-[var(--text-2)] max-w-[220px] truncate">
          {g.key_cards.slice(0, 3).join(", ")}
        </td>
      </tr>
    );
  });
}

function nonEmpty(value: string): string | undefined {
  return value === "" ? undefined : value;
}

function gameIdentity(game: GameSummary | null): [string, number, number] {
  if (!game) return ["", -1, -1];
  return [game.run_id, game.iteration, game.game_idx];
}

function navDeltaForKey(key: string): number {
  if (key === "j" || key === "ArrowDown") return 1;
  if (key === "k" || key === "ArrowUp") return -1;
  return 0;
}

function isClickableControl(target: EventTarget | null): boolean {
  return target instanceof HTMLButtonElement || target instanceof HTMLAnchorElement;
}

function GamesHeaderControls(props: {
  total: number;
  loading: boolean;
  filters: GamesFilterState;
  facets: ReturnType<typeof useGamesSearch>["facets"];
  onPatchFilters: (patch: Partial<GamesFilterState>) => void;
}): ReactNode {
  const { filters, facets, onPatchFilters } = props;
  const currentPreset = activePresetId(filters);
  const runOptions =
    filters.runId && !facets.runs.includes(filters.runId)
      ? [filters.runId, ...facets.runs]
      : facets.runs;

  return (
    <div className="flex flex-col gap-2 px-4 py-2.5 border-b border-[var(--border)] bg-[var(--surface)]">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-[var(--accent)]" />
          <h2 className="text-sm font-semibold tracking-tight">Games &amp; Mistakes</h2>
          <Badge>{props.total} playable</Badge>
          {props.loading ? (
            <span className="text-[11px] text-[var(--text-3)]">Updating…</span>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {SEARCH_PRESETS.map((preset) => {
            const active = currentPreset === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                onClick={(): void => onPatchFilters(preset.patch)}
                className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium border cursor-pointer transition-colors ${
                  active
                    ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                    : "border-[var(--border)] bg-[var(--bg)] text-[var(--text-2)] hover:text-[var(--text)]"
                }`}
              >
                {preset.id === "all" ? null : <Sparkles className="w-3 h-3" />}
                {preset.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-1.5 min-w-0">
        <div className="relative flex-1 min-w-[150px] sm:flex-none sm:w-44">
          <Search className="w-3.5 h-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-[var(--text-3)]" />
          <input
            type="search"
            value={filters.search}
            onChange={(e): void => onPatchFilters({ search: e.target.value })}
            placeholder="Card, seed, color, run…"
            aria-label="Search games"
            className="h-7 w-full rounded-md border border-[var(--border)] bg-[var(--bg)] pl-6 pr-2 text-xs"
          />
        </div>

        {runOptions.length > 0 ? (
          <select
            value={filters.runId}
            onChange={(e): void =>
              onPatchFilters({ runId: e.target.value, card: "", colors: "", setCode: "" })
            }
            aria-label="Filter by run"
            className="h-7 min-w-0 max-w-full sm:max-w-[148px] rounded-md border border-[var(--border)] bg-[var(--bg)] px-1.5 text-xs"
          >
            <option value="">All Runs ({facets.runs.length})</option>
            {runOptions.map((runId) => (
              <option key={runId} value={runId}>
                {runId.split("/").pop()}
              </option>
            ))}
          </select>
        ) : null}

        {facets.sets.length > 0 ? (
          <select
            value={filters.setCode}
            onChange={(e): void => onPatchFilters({ setCode: e.target.value })}
            aria-label="Filter by set"
            className="h-7 min-w-0 max-w-full rounded-md border border-[var(--border)] bg-[var(--bg)] px-1.5 text-xs"
          >
            <option value="">All Sets</option>
            {facets.sets.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        ) : null}

        {facets.cards.length > 0 ? (
          <select
            value={filters.card}
            onChange={(e): void => onPatchFilters({ card: e.target.value })}
            aria-label="Filter by card"
            className="h-7 min-w-0 max-w-full sm:max-w-[156px] rounded-md border border-[var(--border)] bg-[var(--bg)] px-1.5 text-xs"
          >
            <option value="">All Cards ({facets.cards.length})</option>
            {facets.cards.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        ) : null}

        {facets.colors.length > 0 ? (
          <select
            value={filters.colors}
            onChange={(e): void => onPatchFilters({ colors: e.target.value })}
            aria-label="Filter by colors"
            className="h-7 min-w-0 max-w-full rounded-md border border-[var(--border)] bg-[var(--bg)] px-1.5 text-xs"
          >
            <option value="">Colors</option>
            {facets.colors.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        ) : null}

        <select
          value={filters.result}
          onChange={(e): void => onPatchFilters({ result: e.target.value })}
          aria-label="Filter by outcome"
          className="h-7 min-w-0 max-w-full rounded-md border border-[var(--border)] bg-[var(--bg)] px-1.5 text-xs"
        >
          <option value="">Outcome</option>
          {(facets.results.length > 0 ? facets.results : ["win", "loss", "draw"]).map((res) => (
            <option key={res} value={res}>
              {res.charAt(0).toUpperCase() + res.slice(1)}
            </option>
          ))}
        </select>

        <select
          value={filters.sort}
          onChange={(e): void => onPatchFilters({ sort: e.target.value })}
          aria-label="Sort games"
          className="h-7 min-w-0 max-w-full shrink-0 rounded-md border border-[var(--border)] bg-[var(--bg)] px-1.5 text-xs font-medium"
        >
          <option value="surprise_desc">Most Wrong ↓</option>
          <option value="swing_desc">Biggest Swing ↓</option>
          <option value="steps_desc">Longest Games</option>
          <option value="steps_asc">Shortest Games</option>
          <option value="turns_desc">Most Turns</option>
          <option value="iter_desc">Latest Iter</option>
          <option value="seed_asc">Seed Asc</option>
        </select>
      </div>
    </div>
  );
}

export function GamesExplorerView(props: {
  runs: RunSummary[];
  filters: GamesFilterState;
  onChangeFilters: (next: GamesFilterState) => void;
  onOpenTabletop: (target: GameTarget) => void;
}): ReactNode {
  const { filters, onChangeFilters, onOpenTabletop } = props;
  const [selectedIdx, setSelectedIdx] = useState(0);

  const patchFilters = (patch: Partial<GamesFilterState>): void => {
    setSelectedIdx(0);
    onChangeFilters({ ...filters, ...patch, page: patch.page ?? 0 });
  };

  const { games, total, facets, loading, error } = useGamesSearch({
    runId: nonEmpty(filters.runId),
    setCode: nonEmpty(filters.setCode),
    result: nonEmpty(filters.result),
    colors: nonEmpty(filters.colors),
    card: nonEmpty(filters.card),
    minSurprise: filters.minSurprise ?? undefined,
    minSwing: filters.minSwing ?? undefined,
    search: nonEmpty(filters.search),
    sort: filters.sort,
    limit: PAGE_SIZE,
    offset: filters.page * PAGE_SIZE,
  });

  const safeIdx = Math.min(selectedIdx, Math.max(0, games.length - 1));
  const activeGame = games[safeIdx] ?? null;
  const [ovRunId, ovIter, ovGameIdx] = gameIdentity(activeGame);
  const { overview, loading: loadingOverview } = useGameOverview(ovRunId, ovIter, ovGameIdx);

  const handleKey = useCallback(
    (event: KeyboardEvent): void => {
      if (games.length === 0) return;
      const delta = navDeltaForKey(event.key);
      if (delta !== 0) {
        event.preventDefault();
        setSelectedIdx((i) => Math.max(0, Math.min(games.length - 1, i + delta)));
      } else if (event.key === "Enter" && activeGame && !isClickableControl(event.target)) {
        event.preventDefault();
        onOpenTabletop({
          run_id: activeGame.run_id,
          iteration: activeGame.iteration,
          game_idx: activeGame.game_idx,
          step: overview?.turning_point_step ?? activeGame.turning_point_step ?? 0,
          seed: activeGame.seed,
        });
      }
    },
    [games.length, activeGame, overview, onOpenTabletop],
  );
  useKeyboardShortcut(handleKey);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const rangeStart = total === 0 ? 0 : filters.page * PAGE_SIZE + 1;
  const rangeEnd = Math.min(total, (filters.page + 1) * PAGE_SIZE);

  return (
    <div className="flex flex-col lg:flex-row h-full min-h-0 overflow-y-auto lg:overflow-hidden">
      <div className="flex-none lg:flex-1 min-w-0 flex flex-col lg:min-h-0 lg:overflow-hidden">
        <GamesHeaderControls
          total={total}
          loading={loading}
          filters={filters}
          facets={facets}
          onPatchFilters={patchFilters}
        />

        <div className="flex-1 lg:overflow-y-auto p-3 flex flex-col gap-2">
          {error ? (
            <div className="rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] p-3 text-xs text-[var(--loss)]">
              {error}
            </div>
          ) : null}
          <div
            className={`rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-x-auto transition-opacity ${
              loading ? "opacity-65" : "opacity-100"
            }`}
          >
            <table className="w-full min-w-[600px] border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-[var(--border)] bg-[var(--surface-2)] text-[10.5px] uppercase tracking-wider text-[var(--text-3)]">
                  <th className="py-2 px-3 font-semibold">Seed / Run</th>
                  <th className="py-2 px-2 font-semibold">Result</th>
                  <th className="py-2 px-2 font-semibold">Prediction vs Outcome</th>
                  <th className="py-2 px-2 font-semibold text-right">Swing</th>
                  <th className="py-2 px-2 font-semibold text-right">Length</th>
                  <th className="py-2 px-3 font-semibold">Key Cards</th>
                </tr>
              </thead>
              <tbody>
                <GamesTableBody
                  games={games}
                  loading={loading}
                  selectedIdx={safeIdx}
                  onSelect={setSelectedIdx}
                  onOpenTabletop={onOpenTabletop}
                />
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-[var(--text-2)] px-1">
            <span className="tabular">
              Showing {rangeStart}–{rangeEnd} of {total} playable games
              {facets.total_unplayable > 0
                ? ` (${facets.total_unplayable.toLocaleString()} older-engine hidden)`
                : ""}
            </span>
            <div className="flex items-center gap-1.5">
              <Button
                size="sm"
                disabled={filters.page <= 0}
                onClick={(): void => patchFilters({ page: Math.max(0, filters.page - 1) })}
              >
                <ChevronLeft className="w-3.5 h-3.5" />
                Prev
              </Button>
              <span className="tabular px-1">
                Page {filters.page + 1} / {totalPages}
              </span>
              <Button
                size="sm"
                disabled={filters.page + 1 >= totalPages}
                onClick={(): void => patchFilters({ page: filters.page + 1 })}
              >
                Next
                <ChevronRight className="w-3.5 h-3.5" />
              </Button>
            </div>
          </div>
        </div>
      </div>

      <GamePreviewDrawer
        game={activeGame}
        overview={overview}
        loadingOverview={loadingOverview}
        highlightCard={filters.card || filters.search}
        onOpenTabletop={onOpenTabletop}
      />
    </div>
  );
}

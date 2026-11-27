/** Payload-driven Games & Moments Explorer with mistake/surprise search, dynamic facets, and clean preview drawer. */

import { ChevronLeft, ChevronRight } from "lucide-react";
import { type ReactNode, useCallback, useState } from "react";
import { Button } from "@/components/ui/Primitives";
import { useGameOverview, useGamesSearch, useKeyboardShortcut } from "@/hooks/useStudioData";
import type { GameTarget, GamesFilterState, RunSummary } from "@/lib/types";
import { GamePreviewDrawer } from "./GamePreviewDrawer";
import { GamesHeaderControls, GamesTableBody } from "./GamesExplorerParts";
import {
  PAGE_SIZE,
  gameIdentity,
  isClickableControl,
  navDeltaForKey,
  nonEmpty,
} from "./gamesUtils";

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

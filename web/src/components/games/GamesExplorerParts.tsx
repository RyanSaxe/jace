/** Header filter controls and playable games table body for Games Explorer. */

import { Layers, Search, Sparkles } from "lucide-react";
import type { ReactNode } from "react";
import { Badge } from "@/components/ui/Primitives";
import type { useGamesSearch } from "@/hooks/useStudioData";
import type { GameSummary, GameTarget, GamesFilterState } from "@/lib/types";
import { SEARCH_PRESETS, activePresetId, formatCalibrationSummary, resultTone } from "./gamesUtils";

export function GamesTableBody(props: {
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

export function GamesHeaderControls(props: {
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

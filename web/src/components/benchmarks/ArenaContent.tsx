/** Elo Arena leaderboard (desktop table + mobile cards) paired with charts and Match Ledger. */

import { type ReactNode, useState } from "react";
import type {
  ArenaLeaderboardResponse,
  ArenaMatchRow,
  ArenaModelRow,
  CompareTarget,
} from "@/lib/types";
import { ArenaEmptyState, MatchLedgerPanel } from "./ArenaMatchLedger";
import { type EloXAxis, formatElo, formatPct, shortAnchorLabel } from "./benchmarksUtils";
import { BradleyTerryEloChart } from "./BradleyTerryEloChart";
import { EarlyRunForecastChart } from "./EarlyRunForecastChart";

function formatVsAnchorCell(row: ArenaModelRow, isAnchor: boolean): string {
  if (isAnchor) return "Anchor";
  return row.vs_anchor_wr === null ? "—" : formatPct(row.vs_anchor_wr);
}

function formatMobileVsAnchor(row: ArenaModelRow, isAnchor: boolean, anchorShort: string): string {
  if (isAnchor) return "Anchor";
  return row.vs_anchor_wr === null
    ? "No direct match"
    : `${formatPct(row.vs_anchor_wr)} vs ${anchorShort}`;
}

function vsAnchorClass(row: ArenaModelRow, isAnchor: boolean): string {
  if (isAnchor || row.vs_anchor_wr === null) return "text-[var(--text-2)]";
  return row.vs_anchor_wr >= 0.5
    ? "text-[var(--win)] font-semibold"
    : "text-[var(--loss)] font-semibold";
}

function filterArenaModels(models: ArenaModelRow[], search: string): ArenaModelRow[] {
  const needle = search.trim().toLowerCase();
  if (!needle) return models;
  return models.filter(
    (m) => m.label.toLowerCase().includes(needle) || m.model_id.toLowerCase().includes(needle),
  );
}

function LeaderboardDesktopTable(props: {
  rows: ArenaModelRow[];
  totalMirroredGames: number;
  anchorModel: string;
  anchorLabel: string;
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
}): ReactNode {
  const { rows, totalMirroredGames, anchorModel, anchorLabel, selectedModelId } = props;
  return (
    <section className="flex-1 min-w-0 flex flex-col overflow-hidden">
      <div className="px-3.5 py-2.5 border-b border-[var(--border)] flex items-center justify-between shrink-0">
        <div className="text-[11px] font-semibold uppercase tracking-[0.04em] text-[var(--text-3)]">
          Leaderboard
        </div>
        <span className="tabular text-[11px] text-[var(--text-3)]">
          {rows.length} models · {totalMirroredGames.toLocaleString("en-US")} mirrored games
        </span>
      </div>
      <div className="flex-1 overflow-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr className="bg-[var(--surface-2)] text-[10.5px] font-semibold uppercase tracking-[0.04em] text-[var(--text-3)]">
              <th className="text-left px-3.5 py-2 border-b border-[var(--border)] w-12">Rank</th>
              <th className="text-left px-3.5 py-2 border-b border-[var(--border)]">Model</th>
              <th className="text-right px-3.5 py-2 border-b border-[var(--border)]">
                BT Elo (95% CI)
              </th>
              <th className="text-right px-3.5 py-2 border-b border-[var(--border)]">Matches</th>
              <th className="text-right px-3.5 py-2 border-b border-[var(--border)]">Games</th>
              <th className="text-right px-3.5 py-2 border-b border-[var(--border)]">
                vs {anchorLabel}
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => {
              const isSelected = row.model_id === selectedModelId;
              const isAnchor = row.model_id === anchorModel;
              return (
                <tr
                  key={row.model_id}
                  onClick={(): void => props.onSelectModel(row.model_id)}
                  className={`cursor-pointer border-b border-[var(--border)] last:border-b-0 transition-colors ${
                    isSelected ? "bg-[var(--accent-soft)]" : "hover:bg-[var(--surface-2)]/60"
                  }`}
                >
                  <td className="tabular px-3.5 py-2.5 text-[var(--text-3)]">{idx + 1}</td>
                  <td className="px-3.5 py-2.5 font-semibold text-[var(--text)]">{row.label}</td>
                  <td
                    className={`tabular text-right px-3.5 py-2.5 font-semibold ${
                      isSelected ? "text-[var(--accent)]" : "text-[var(--text)]"
                    }`}
                  >
                    {formatElo(row.elo, row.ci_half)}
                  </td>
                  <td className="tabular text-right px-3.5 py-2.5 text-[var(--text-2)]">
                    {row.matches}
                  </td>
                  <td className="tabular text-right px-3.5 py-2.5 text-[var(--text-2)]">
                    {row.games.toLocaleString("en-US")}
                  </td>
                  <td
                    className={`tabular text-right px-3.5 py-2.5 ${vsAnchorClass(row, isAnchor)}`}
                  >
                    {formatVsAnchorCell(row, isAnchor)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function LeaderboardMobileCards(props: {
  rows: ArenaModelRow[];
  anchorModel: string;
  anchorShort: string;
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
}): ReactNode {
  const { rows, anchorModel, anchorShort, selectedModelId } = props;
  return (
    <>
      <div className="flex items-center justify-between px-0.5">
        <div className="text-[11px] font-semibold uppercase tracking-[0.04em] text-[var(--text-3)]">
          Leaderboard
        </div>
        <span className="tabular text-[10.5px] text-[var(--text-3)]">{rows.length} models</span>
      </div>
      {rows.map((row, idx) => {
        const isSelected = row.model_id === selectedModelId;
        const isAnchor = row.model_id === anchorModel;
        const mutedMeta = isAnchor || row.vs_anchor_wr === null;
        return (
          <button
            key={row.model_id}
            type="button"
            aria-label={`Inspect ${row.label} match ledger`}
            onClick={(): void => props.onSelectModel(isSelected ? "" : row.model_id)}
            className={`w-full text-left rounded-lg border px-2.5 py-2 flex flex-col gap-0.5 cursor-pointer transition-colors ${
              isSelected
                ? "border-[var(--accent)] bg-[var(--accent-soft)]"
                : "border-[var(--border)] bg-[var(--surface)]"
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 min-w-0">
                <span className="tabular text-[11px] text-[var(--text-3)] w-3.5 shrink-0">
                  {idx + 1}
                </span>
                <span className="text-xs font-semibold text-[var(--text)] truncate">
                  {row.label}
                </span>
              </span>
              <span
                className={`tabular text-xs font-semibold shrink-0 ${
                  isSelected ? "text-[var(--accent)]" : "text-[var(--text)]"
                }`}
              >
                {formatElo(row.elo, row.ci_half)}
              </span>
            </div>
            <div className="tabular flex items-center justify-between gap-2 text-[10.5px]">
              <span className="text-[var(--text-3)]">
                {row.matches} matches · {row.games.toLocaleString("en-US")} games
              </span>
              <span className={mutedMeta ? "text-[var(--text-3)]" : vsAnchorClass(row, isAnchor)}>
                {formatMobileVsAnchor(row, isAnchor, anchorShort)}
              </span>
            </div>
          </button>
        );
      })}
    </>
  );
}

function ArenaLoadedView(props: {
  leaderboard: ArenaLeaderboardResponse;
  search: string;
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  matches: ArenaMatchRow[];
  matchesLoading: boolean;
  matchesError: string | null;
  onComparePair: (target: CompareTarget) => void;
}): ReactNode {
  const [xAxis, setXAxis] = useState<EloXAxis>("updates");
  const { leaderboard, search, selectedModelId } = props;
  const { models, anchor_model: anchorModel, anchor_elo: anchorElo, forecast } = leaderboard;
  const filteredModels = filterArenaModels(models, search);
  const anchorRow = models.find((m) => m.model_id === anchorModel) ?? null;
  const anchorLabel = anchorRow?.label ?? anchorModel;
  const totalMirroredGames = Math.round(models.reduce((acc, m) => acc + m.games, 0) / 2);
  const selectedModel = models.find((m) => m.model_id === selectedModelId) ?? null;

  return (
    <div className="flex flex-col gap-3 h-full min-h-0">
      <div className="grid grid-cols-1 lg:grid-cols-[1.46fr_1fr] gap-3 shrink-0">
        <div className="hidden md:block">
          <BradleyTerryEloChart
            models={models}
            anchorModel={anchorModel}
            anchorElo={anchorElo}
            xAxis={xAxis}
            onChangeXAxis={setXAxis}
          />
        </div>
        <div className="block md:hidden">
          <BradleyTerryEloChart
            models={models}
            anchorModel={anchorModel}
            anchorElo={anchorElo}
            xAxis="updates"
            onChangeXAxis={setXAxis}
            compact
          />
        </div>
        <EarlyRunForecastChart forecast={forecast} />
      </div>

      <div className="hidden md:flex flex-1 min-h-[260px] overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
        <LeaderboardDesktopTable
          rows={filteredModels}
          totalMirroredGames={totalMirroredGames}
          anchorModel={anchorModel}
          anchorLabel={anchorLabel}
          selectedModelId={selectedModelId}
          onSelectModel={props.onSelectModel}
        />
        {selectedModel ? (
          <MatchLedgerPanel
            selected={selectedModel}
            anchorRow={anchorRow}
            matches={props.matches}
            loading={props.matchesLoading}
            error={props.matchesError}
            onClose={(): void => props.onSelectModel("")}
            onComparePair={props.onComparePair}
          />
        ) : null}
      </div>

      <div className="flex md:hidden flex-col gap-1.5">
        <LeaderboardMobileCards
          rows={filteredModels}
          anchorModel={anchorModel}
          anchorShort={shortAnchorLabel(anchorLabel)}
          selectedModelId={selectedModelId}
          onSelectModel={props.onSelectModel}
        />
        {selectedModel ? (
          <div className="mt-1 rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
            <MatchLedgerPanel
              selected={selectedModel}
              anchorRow={anchorRow}
              matches={props.matches}
              loading={props.matchesLoading}
              error={props.matchesError}
              onClose={(): void => props.onSelectModel("")}
              onComparePair={props.onComparePair}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function ArenaContent(props: {
  leaderboard: ArenaLeaderboardResponse | null;
  loading: boolean;
  error: string | null;
  search: string;
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  matches: ArenaMatchRow[];
  matchesLoading: boolean;
  matchesError: string | null;
  onComparePair: (target: CompareTarget) => void;
}): ReactNode {
  if (props.loading && !props.leaderboard) {
    return <div className="text-xs text-[var(--text-3)] p-4">Loading Elo Arena…</div>;
  }
  if (props.error && !props.leaderboard) {
    return <div className="text-xs text-[var(--loss)] p-4">{props.error}</div>;
  }
  if (!props.leaderboard || props.leaderboard.models.length === 0) {
    return <ArenaEmptyState format={props.leaderboard?.format ?? ""} />;
  }
  return (
    <ArenaLoadedView
      leaderboard={props.leaderboard}
      search={props.search}
      selectedModelId={props.selectedModelId}
      onSelectModel={props.onSelectModel}
      matches={props.matches}
      matchesLoading={props.matchesLoading}
      matchesError={props.matchesError}
      onComparePair={props.onComparePair}
    />
  );
}

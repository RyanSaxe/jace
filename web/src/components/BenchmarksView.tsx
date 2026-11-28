/** Benchmarks workbench: Elo Arena (Bradley-Terry ratings, forecast, Match Ledger) and Tactical Puzzles. */

import { ArrowRight, RefreshCw, Trophy, X } from "lucide-react";
import { type ReactNode, useState } from "react";
import {
  BradleyTerryEloChart,
  type EloXAxis,
  EarlyRunForecastChart,
  formatElo,
  formatPct,
  shortAnchorLabel,
} from "@/components/BenchmarksArenaCharts";
import { PuzzlesContent } from "@/components/BenchmarksPuzzles";
import { Button, SegmentedControl } from "@/components/ui/Primitives";
import { useArenaLeaderboard, useArenaMatches, usePuzzlesData } from "@/hooks/useBenchmarksData";
import type {
  ArenaLeaderboardResponse,
  ArenaMatchRow,
  ArenaModelRow,
  BenchmarkSubView,
  BenchmarksRouteState,
  CompareTarget,
  GameTarget,
  GlobalCheckpointOption,
  PuzzlesResponse,
} from "@/lib/types";

export interface BenchmarksViewProps {
  checkpoints: GlobalCheckpointOption[];
  routeState: BenchmarksRouteState;
  onChangeRouteState: (next: BenchmarksRouteState) => void;
  onOpenTabletop: (target: GameTarget) => void;
  onComparePair: (target: CompareTarget) => void;
}

function matchBarColor(match: ArenaMatchRow): string {
  if (match.winrate > 0.5 && match.ci_low >= 0.5) return "var(--win)";
  if (match.winrate < 0.5 && match.ci_high <= 0.5) return "var(--loss)";
  return "var(--accent)";
}

function resolveCompareTargetForModel(
  selected: ArenaModelRow,
  matches: ArenaMatchRow[],
  anchorRow: ArenaModelRow | null,
): CompareTarget | null {
  for (const row of matches) {
    if (row.compare_target) return row.compare_target;
  }
  if (
    selected.checkpoint_ref &&
    anchorRow?.checkpoint_ref &&
    selected.checkpoint_ref !== anchorRow.checkpoint_ref
  ) {
    return { ckptA: selected.checkpoint_ref, ckptB: anchorRow.checkpoint_ref };
  }
  return null;
}

function resolveFormatOptions(formats: string[], format: string): string[] {
  if (formats.length > 0) return formats;
  return format ? [format] : [];
}

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

function FormatSelect(props: {
  ariaLabel: string;
  format: string;
  options: string[];
  className: string;
  onChange: (fmt: string) => void;
}): ReactNode {
  return (
    <select
      aria-label={props.ariaLabel}
      value={props.format}
      disabled={props.options.length === 0}
      onChange={(e): void => props.onChange(e.target.value)}
      className={props.className}
    >
      {props.options.length === 0 ? (
        <option value="">No formats discovered</option>
      ) : (
        props.options.map((fmt) => (
          <option key={fmt} value={fmt}>
            {fmt}
          </option>
        ))
      )}
    </select>
  );
}

function AnchorSelect(props: {
  ariaLabel: string;
  anchor: string;
  anchorElo: number;
  models: ArenaModelRow[];
  labelClassName: string;
  selectClassName: string;
  onChange: (anchor: string) => void;
}): ReactNode {
  const eloStr = Math.round(props.anchorElo).toLocaleString("en-US");
  return (
    <label className={props.labelClassName}>
      <span className="font-medium shrink-0">Anchor:</span>
      <select
        aria-label={props.ariaLabel}
        value={props.anchor}
        onChange={(e): void => props.onChange(e.target.value)}
        className={props.selectClassName}
      >
        {props.models.map((m) => (
          <option key={m.model_id} value={m.model_id}>
            {m.label} ({eloStr})
          </option>
        ))}
      </select>
    </label>
  );
}

function BenchmarksToolbar(props: {
  subView: BenchmarkSubView;
  format: string;
  formats: string[];
  anchor: string;
  anchorElo: number;
  models: ArenaModelRow[];
  search: string;
  onChangeSubView: (subView: BenchmarkSubView) => void;
  onChangeFormat: (fmt: string) => void;
  onChangeAnchor: (anchor: string) => void;
  onChangeSearch: (q: string) => void;
  onRefresh: () => void;
}): ReactNode {
  const { subView, format, formats, anchor, anchorElo, models, search } = props;
  const formatOptions = resolveFormatOptions(formats, format);
  const showAnchor = subView === "arena" && models.length > 0;

  return (
    <>
      <header className="hidden md:flex h-[44px] shrink-0 bg-[var(--surface)] border-b border-[var(--border)] px-4 items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="flex items-center gap-1.5 text-[13.5px] font-semibold tracking-tight text-[var(--text)] whitespace-nowrap">
            <Trophy className="w-4 h-4 text-[var(--accent)] shrink-0" />
            <span>Benchmarks</span>
          </div>
          <FormatSelect
            ariaLabel="Benchmark format"
            format={format}
            options={formatOptions}
            onChange={props.onChangeFormat}
            className="h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-[11.5px] font-medium text-[var(--text)] max-w-[190px]"
          />
          {showAnchor ? (
            <AnchorSelect
              ariaLabel="Elo anchor"
              anchor={anchor}
              anchorElo={anchorElo}
              models={models}
              onChange={props.onChangeAnchor}
              labelClassName="inline-flex items-center h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 gap-1 text-[11.5px] text-[var(--text-3)]"
              selectClassName="bg-transparent border-0 text-[11.5px] font-medium text-[var(--text)] focus:outline-none max-w-[180px]"
            />
          ) : null}
          <SegmentedControl
            value={subView}
            onChange={props.onChangeSubView}
            options={[
              { value: "arena", label: "Elo Arena" },
              { value: "puzzles", label: "Puzzles" },
            ]}
          />
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <input
            type="text"
            value={search}
            onChange={(e): void => props.onChangeSearch(e.target.value)}
            placeholder={subView === "arena" ? "Filter models..." : "Filter puzzles..."}
            aria-label={subView === "arena" ? "Filter models" : "Filter puzzles"}
            className="h-7 w-[164px] rounded-md border border-[var(--border)] bg-[var(--bg)] px-2.5 text-[11.5px] text-[var(--text)] placeholder:text-[var(--text-3)]"
          />
          <button
            type="button"
            title="Refresh benchmarks"
            aria-label="Refresh benchmarks"
            onClick={props.onRefresh}
            className="w-7 h-7 rounded-md border border-[var(--border)] bg-[var(--surface)] text-[var(--text-2)] hover:border-[var(--border-strong)] hover:text-[var(--text)] flex items-center justify-center cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </header>

      <div className="flex md:hidden flex-col gap-2 px-3 pt-2.5 pb-1 shrink-0">
        <div className="flex items-center gap-2">
          <FormatSelect
            ariaLabel="Benchmark format mobile"
            format={format}
            options={formatOptions}
            onChange={props.onChangeFormat}
            className="flex-1 h-8 rounded-md border border-[var(--border)] bg-[var(--surface)] px-2.5 text-xs font-medium text-[var(--text)]"
          />
          <button
            type="button"
            aria-label="Refresh benchmarks mobile"
            onClick={props.onRefresh}
            className="w-8 h-8 rounded-md border border-[var(--border)] bg-[var(--surface)] text-[var(--text-2)] flex items-center justify-center cursor-pointer shrink-0"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
        {showAnchor ? (
          <AnchorSelect
            ariaLabel="Elo anchor mobile"
            anchor={anchor}
            anchorElo={anchorElo}
            models={models}
            onChange={props.onChangeAnchor}
            labelClassName="flex items-center h-8 rounded-md border border-[var(--border)] bg-[var(--surface)] px-2.5 gap-1.5 text-xs text-[var(--text-3)]"
            selectClassName="flex-1 bg-transparent border-0 text-xs font-medium text-[var(--text)] focus:outline-none min-w-0"
          />
        ) : null}
        <div className="grid grid-cols-2 rounded-md bg-[var(--surface-2)] p-0.5 border border-[var(--border)]">
          {(
            [
              ["arena", "Elo Arena"],
              ["puzzles", "Puzzles"],
            ] as const
          ).map(([val, lbl]) => (
            <button
              key={val}
              type="button"
              onClick={(): void => props.onChangeSubView(val)}
              className={`py-1.5 rounded text-xs font-medium cursor-pointer transition-colors ${
                subView === val
                  ? "bg-[var(--surface)] text-[var(--text)] font-semibold shadow-[var(--shadow-sm)]"
                  : "text-[var(--text-2)]"
              }`}
            >
              {lbl}
            </button>
          ))}
        </div>
      </div>
    </>
  );
}

function MatchLedgerRows(props: {
  matches: ArenaMatchRow[];
  loading: boolean;
  error: string | null;
}): ReactNode {
  if (props.loading) {
    return <div className="py-4 text-xs text-[var(--text-3)]">Loading match ledger…</div>;
  }
  if (props.error) {
    return <div className="py-4 text-xs text-[var(--loss)]">{props.error}</div>;
  }
  if (props.matches.length === 0) {
    return <div className="py-4 text-xs text-[var(--text-3)]">No mirrored matches recorded.</div>;
  }
  return (
    <div>
      {props.matches.map((m) => {
        const col = matchBarColor(m);
        const pct = Math.max(0, Math.min(100, m.winrate * 100));
        return (
          <div
            key={m.match_id}
            className="py-3 border-b border-[var(--border)] last:border-b-0 flex flex-col gap-1.5"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-[var(--text)] truncate">
                vs {m.opponent_label}
              </span>
              <span className="tabular text-[11.5px] font-semibold shrink-0" style={{ color: col }}>
                {formatPct(m.winrate)} [{(m.ci_low * 100).toFixed(1)}–{(m.ci_high * 100).toFixed(1)}
                %]
              </span>
            </div>
            <div className="relative h-1 rounded-sm bg-[var(--surface-2)] overflow-hidden">
              <div className="h-full rounded-sm" style={{ width: `${pct}%`, background: col }} />
              <div className="absolute top-0 bottom-0 left-1/2 w-px bg-[var(--border-strong)]" />
            </div>
            <div className="tabular text-[11px] text-[var(--text-3)]">
              {m.sweeps_w === 0 && m.sweeps_l === 0
                ? `${m.games} games · LOS ${(m.los * 100).toFixed(1)}%`
                : `${m.sweeps_w}W / ${m.sweeps_l}L sweeps · LOS ${(m.los * 100).toFixed(1)}%`}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function MatchLedgerPanel(props: {
  selected: ArenaModelRow;
  anchorRow: ArenaModelRow | null;
  matches: ArenaMatchRow[];
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onComparePair: (target: CompareTarget) => void;
}): ReactNode {
  const { selected, anchorRow, matches } = props;
  const compareTarget = resolveCompareTargetForModel(selected, matches, anchorRow);

  return (
    <aside
      aria-label="Match ledger"
      className="w-full md:w-[360px] shrink-0 bg-[var(--surface)] md:border-l border-[var(--border)] flex flex-col justify-between"
    >
      <div className="min-h-0 flex-1 flex flex-col overflow-hidden">
        <div className="px-4 py-3 border-b border-[var(--border)] flex items-start justify-between gap-2 shrink-0">
          <div className="min-w-0">
            <div className="text-[13.5px] font-semibold text-[var(--text)] tracking-tight truncate">
              {selected.label}
            </div>
            <div className="tabular mt-0.5 text-[11px] text-[var(--text-2)]">
              {formatElo(selected.elo, selected.ci_half)} Elo · {selected.matches} matches ·{" "}
              {selected.games.toLocaleString("en-US")} games
            </div>
          </div>
          <button
            type="button"
            aria-label="Close match ledger"
            onClick={props.onClose}
            className="w-6 h-6 rounded flex items-center justify-center text-[var(--text-3)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] cursor-pointer shrink-0"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.04em] text-[var(--text-3)] mb-0.5">
            Match Ledger
          </div>
          <MatchLedgerRows matches={matches} loading={props.loading} error={props.error} />
        </div>
      </div>

      {compareTarget ? (
        <div className="px-4 py-3 border-t border-[var(--border)] shrink-0">
          <Button
            variant="default"
            size="md"
            className="w-full"
            onClick={(): void => props.onComparePair(compareTarget)}
          >
            <span>Compare in Studio</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Button>
        </div>
      ) : null}
    </aside>
  );
}

function ArenaEmptyState(props: { format: string }): ReactNode {
  const fmt = props.format || "<format>";
  return (
    <div className="rounded-lg border border-dashed border-[var(--border-strong)] bg-[var(--surface)] p-8 text-center max-w-2xl mx-auto my-6 flex flex-col items-center gap-3">
      <div className="text-sm font-semibold text-[var(--text)]">No Arena matches recorded yet</div>
      <div className="text-xs text-[var(--text-2)]">
        Run a mirrored Arena match or import external match summaries to populate the Bradley-Terry
        leaderboard:
      </div>
      <pre className="w-full overflow-x-auto rounded-md border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-left text-[11px] text-[var(--text)] tabular">
        {`uv run python -m jace.eval.arena match --runs-dir <runs-dir> --format ${fmt} --model-a <checkpoint> --model-b heuristic`}
      </pre>
    </div>
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

function filterArenaModels(models: ArenaModelRow[], search: string): ArenaModelRow[] {
  const needle = search.trim().toLowerCase();
  if (!needle) return models;
  return models.filter(
    (m) => m.label.toLowerCase().includes(needle) || m.model_id.toLowerCase().includes(needle),
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

function ArenaContent(props: {
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

function pickActiveFormat(
  routeState: BenchmarksRouteState,
  arenaData: ArenaLeaderboardResponse | null,
  puzzlesData: PuzzlesResponse | null,
): string {
  if (routeState.format) return routeState.format;
  if (routeState.subView === "arena") return arenaData?.format ?? "";
  return puzzlesData?.format ?? "";
}

function pickDiscoveredFormats(
  subView: BenchmarkSubView,
  arenaData: ArenaLeaderboardResponse | null,
  puzzlesData: PuzzlesResponse | null,
): string[] {
  if (subView === "arena") return arenaData?.formats ?? [];
  return puzzlesData?.formats ?? arenaData?.formats ?? [];
}

function pickSelectedModel(routeModel: string, models: ArenaModelRow[]): string {
  return routeModel && models.some((m) => m.model_id === routeModel) ? routeModel : "";
}

const NONE_MODEL_SENTINEL = "__none__";

function resolvePuzzleSlot(routeVal: string, fallback: string): string {
  if (routeVal === NONE_MODEL_SENTINEL) return "";
  return routeVal || fallback;
}

function resolveDefaultPuzzleModels(
  routeState: BenchmarksRouteState,
  checkpoints: GlobalCheckpointOption[],
): { modelA: string; modelB: string } {
  const ranked = checkpoints.toSorted((a, b) => b.iteration - a.iteration);
  return {
    modelA: resolvePuzzleSlot(routeState.modelA, ranked[0]?.ref ?? ""),
    modelB: resolvePuzzleSlot(routeState.modelB, "heuristic"),
  };
}

function resolveToolbarArenaProps(
  routeState: BenchmarksRouteState,
  arenaData: ArenaLeaderboardResponse | null,
): { anchor: string; anchorElo: number; models: ArenaModelRow[] } {
  return {
    anchor: routeState.anchor || arenaData?.anchor_model || "",
    anchorElo: arenaData?.anchor_elo ?? 1500,
    models: arenaData?.models ?? [],
  };
}

export function BenchmarksView(props: BenchmarksViewProps): ReactNode {
  const { checkpoints, routeState, onChangeRouteState } = props;
  const [search, setSearch] = useState("");
  const [phaseFilter, setPhaseFilter] = useState("");
  const [sourceFilter, setSourceFilter] = useState("");
  const [refreshTick, setRefreshTick] = useState(0);

  const isArena = routeState.subView === "arena";
  const { modelA: defaultModelA, modelB: defaultModelB } = resolveDefaultPuzzleModels(
    routeState,
    checkpoints,
  );

  const arena = useArenaLeaderboard({
    format: routeState.format,
    anchor: routeState.anchor,
    refreshTick,
    enabled: isArena,
  });
  const puzzles = usePuzzlesData({
    format: routeState.format,
    modelA: defaultModelA,
    modelB: defaultModelB,
    phase: phaseFilter,
    source: sourceFilter,
    refreshTick,
    enabled: !isArena,
  });

  const activeFormat = pickActiveFormat(routeState, arena.data, puzzles.data);
  const formats = pickDiscoveredFormats(routeState.subView, arena.data, puzzles.data);
  const toolbarArena = resolveToolbarArenaProps(routeState, arena.data);
  const activeModelId = pickSelectedModel(routeState.model, toolbarArena.models);
  const matches = useArenaMatches({
    format: activeFormat,
    model: activeModelId,
    refreshTick,
    enabled: isArena && activeModelId !== "",
  });

  return (
    <div className="flex flex-col h-full min-h-0">
      <BenchmarksToolbar
        subView={routeState.subView}
        format={activeFormat}
        formats={formats}
        anchor={toolbarArena.anchor}
        anchorElo={toolbarArena.anchorElo}
        models={toolbarArena.models}
        search={search}
        onChangeSubView={(subView): void => onChangeRouteState({ ...routeState, subView })}
        onChangeFormat={(format): void =>
          onChangeRouteState({ ...routeState, format, anchor: "", model: "" })
        }
        onChangeAnchor={(anchor): void => onChangeRouteState({ ...routeState, anchor })}
        onChangeSearch={setSearch}
        onRefresh={(): void => setRefreshTick((n) => n + 1)}
      />
      <div className="flex-1 min-h-0 overflow-y-auto px-3 md:px-4 py-3">
        {isArena ? (
          <ArenaContent
            leaderboard={arena.data}
            loading={arena.loading}
            error={arena.error}
            search={search}
            selectedModelId={activeModelId}
            onSelectModel={(model): void => onChangeRouteState({ ...routeState, model })}
            matches={matches.matches}
            matchesLoading={matches.loading}
            matchesError={matches.error}
            onComparePair={props.onComparePair}
          />
        ) : (
          <PuzzlesContent
            data={puzzles.data}
            loading={puzzles.loading}
            error={puzzles.error}
            checkpoints={checkpoints}
            modelA={defaultModelA}
            modelB={defaultModelB}
            phase={phaseFilter}
            source={sourceFilter}
            search={search}
            onChangeModelA={(modelA): void =>
              onChangeRouteState({ ...routeState, modelA: modelA || NONE_MODEL_SENTINEL })
            }
            onChangeModelB={(modelB): void =>
              onChangeRouteState({ ...routeState, modelB: modelB || NONE_MODEL_SENTINEL })
            }
            onChangePhase={setPhaseFilter}
            onChangeSource={setSourceFilter}
            onOpenTabletop={props.onOpenTabletop}
          />
        )}
      </div>
    </div>
  );
}

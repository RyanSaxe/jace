/** Match ledger side panel and empty state for the Elo Arena leaderboard. */

import { ArrowRight, X } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/Primitives";
import type { ArenaMatchRow, ArenaModelRow, CompareTarget } from "@/lib/types";
import { formatElo, formatPct } from "./benchmarksUtils";

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

export function MatchLedgerPanel(props: {
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

export function ArenaEmptyState(props: { format: string }): ReactNode {
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

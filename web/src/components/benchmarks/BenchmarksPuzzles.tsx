/** Tactical Puzzles sub-view for Benchmarks: checkpoint filter bar, puzzle cards, and Tabletop links. */

import { ArrowRight } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/Primitives";
import { cardImageUrl } from "@/lib/api";
import type {
  GameTarget,
  GlobalCheckpointOption,
  PuzzleCardItem,
  PuzzleSource,
  PuzzlesResponse,
} from "@/lib/types";
import { formatPct } from "./benchmarksUtils";

const SOURCE_LABELS: Record<PuzzleSource, string> = {
  auto_mined: "Auto-mined",
  search_disagreement: "Search disagreement",
  watchlist: "Watchlist",
};

function shortModelName(ref: string, checkpoints: GlobalCheckpointOption[]): string {
  if (!ref) return "None";
  if (ref === "heuristic") return "Heuristic";
  const match = checkpoints.find((c) => c.ref === ref);
  if (match) return `u${match.iteration.toLocaleString("en-US")}`;
  const tail = ref.split("::").at(-1) ?? ref;
  return tail.length > 14 ? `${tail.slice(0, 13)}…` : tail;
}

function probToneClass(prob: number | null, isBest: boolean): string {
  if (prob === null) return "text-[var(--text-3)]";
  if (isBest) return prob >= 0.5 ? "font-semibold text-[var(--win)]" : "text-[var(--text-3)]";
  return prob >= 0.5 ? "font-semibold text-[var(--loss)]" : "text-[var(--text-3)]";
}

function PuzzleCardThumb(props: { name: string }): ReactNode {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div
        title={props.name}
        className="w-[68px] aspect-[5/7] rounded-[5px] border border-[var(--border-strong)] bg-[var(--surface-2)] p-1 flex items-center justify-center text-center text-[9px] leading-tight text-[var(--text-2)]"
      >
        <span className="line-clamp-4">{props.name}</span>
      </div>
    );
  }
  return (
    <div
      title={props.name}
      className="w-[68px] aspect-[5/7] rounded-[5px] overflow-hidden border border-[var(--border-strong)] bg-[#18181b]"
    >
      <img
        src={cardImageUrl(props.name)}
        alt={props.name}
        loading="lazy"
        onError={(): void => setFailed(true)}
        className="w-full h-full object-cover block"
      />
    </div>
  );
}

function PuzzleCard(props: {
  puzzle: PuzzleCardItem;
  labelA: string;
  labelB: string;
  onOpenTabletop: (target: GameTarget) => void;
}): ReactNode {
  const { puzzle, labelA, labelB } = props;
  const thumbs = puzzle.key_cards.slice(0, 2);

  return (
    <article className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-4 py-3.5 flex flex-col gap-3">
      <div className="flex items-start justify-between gap-3.5">
        <div className="flex-1 min-w-0 flex flex-col gap-1">
          <div className="flex items-baseline justify-between gap-2 flex-wrap">
            <span className="text-[13.5px] font-semibold text-[var(--text)] tracking-tight">
              T{puzzle.turn} · {puzzle.phase}
            </span>
            <span className="tabular text-[11.5px] font-semibold text-[var(--win)]">
              <span className="font-medium text-[var(--text-3)]">{puzzle.seed_label} · </span>+
              {(puzzle.margin * 100).toFixed(1)}%
            </span>
          </div>
          <div className="tabular text-[11px] text-[var(--text-2)] leading-snug">
            {puzzle.hero_summary}
          </div>
          <div className="tabular text-[11px] text-[var(--text-2)] leading-snug">
            {puzzle.opp_summary}
          </div>
          <div className="tabular text-[10.5px] text-[var(--text-3)]">
            95% CI +{(puzzle.ci_low * 100).toFixed(1)}% to +{(puzzle.ci_high * 100).toFixed(1)}%
          </div>
        </div>
        {thumbs.length > 0 ? (
          <div className="flex items-center gap-1.5 shrink-0">
            {thumbs.map((card) => (
              <PuzzleCardThumb key={card} name={card} />
            ))}
          </div>
        ) : null}
      </div>

      <table className="w-full border-collapse text-[11.5px] border-t border-[var(--border)]">
        <thead>
          <tr className="bg-[var(--bg)] text-[10px] font-semibold uppercase tracking-[0.04em] text-[var(--text-3)]">
            <th className="text-left px-2 py-1.5 border-b border-[var(--border)]">Action</th>
            <th className="text-right px-2 py-1.5 border-b border-[var(--border)]">Playout</th>
            <th className="text-right px-2 py-1.5 border-b border-[var(--border)]">
              <span className="inline-flex items-center gap-1">
                <span className="w-[7px] h-[7px] rounded-full bg-[var(--accent)]" />
                {labelA}
              </span>
            </th>
            <th className="text-right px-2 py-1.5 border-b border-[var(--border)]">
              <span className="inline-flex items-center gap-1">
                <span className="w-[7px] h-[7px] rounded-full bg-[var(--warn)]" />
                {labelB}
              </span>
            </th>
          </tr>
        </thead>
        <tbody>
          {puzzle.actions.map((act) => (
            <tr
              key={act.label}
              className={`border-b border-[var(--border)] last:border-b-0 ${
                act.is_best ? "bg-[var(--win-soft)]/30" : ""
              }`}
            >
              <td className="px-2 py-2 font-medium text-[var(--text)]">{act.label}</td>
              <td className="tabular text-right px-2 py-2 font-semibold text-[var(--text)]">
                {formatPct(act.playout_wr)}
              </td>
              <td
                className={`tabular text-right px-2 py-2 ${probToneClass(act.prob_a, act.is_best)}`}
              >
                {act.prob_a === null ? "—" : formatPct(act.prob_a)}
              </td>
              <td
                className={`tabular text-right px-2 py-2 ${probToneClass(act.prob_b, act.is_best)}`}
              >
                {act.prob_b === null ? "—" : formatPct(act.prob_b)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="flex items-center justify-between pt-2.5 border-t border-[var(--border)]">
        <span className="tabular text-[11px] text-[var(--text-3)]">
          {SOURCE_LABELS[puzzle.source]} · N={puzzle.playouts}
        </span>
        <Button
          variant="default"
          onClick={(): void => props.onOpenTabletop(puzzle.tabletop_target)}
        >
          <span>Open in Tabletop</span>
          <ArrowRight className="w-3 h-3" />
        </Button>
      </div>
    </article>
  );
}

function PuzzlesEmptyState(props: { format: string }): ReactNode {
  const fmt = props.format || "<format>";
  return (
    <div className="rounded-lg border border-dashed border-[var(--border-strong)] bg-[var(--surface)] p-8 text-center max-w-2xl mx-auto my-6 flex flex-col items-center gap-3">
      <div className="text-sm font-semibold text-[var(--text)]">No verified puzzles for {fmt}</div>
      <div className="text-xs text-[var(--text-2)]">
        Mine verified tactical puzzles from recorded games or promote bookmarked Watchlist boards:
      </div>
      <pre className="w-full overflow-x-auto rounded-md border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-left text-[11px] text-[var(--text)] tabular">
        {`uv run python -m jace.eval.puzzles mine --runs-dir <runs-dir> --format ${fmt} --checkpoint <path/to/checkpoint.pt> --sources policy,swing --playouts 64 --max-plies 240`}
      </pre>
    </div>
  );
}

function PuzzleModelSelect(props: {
  ariaLabel: string;
  dotClassName: string;
  value: string;
  checkpoints: GlobalCheckpointOption[];
  onChange: (val: string) => void;
}): ReactNode {
  return (
    <label className="inline-flex items-center gap-1.5 h-7 rounded-md border border-[var(--border)] bg-[var(--surface)] px-2">
      <span className={`w-[7px] h-[7px] rounded-full shrink-0 ${props.dotClassName}`} />
      <select
        aria-label={props.ariaLabel}
        value={props.value}
        onChange={(e): void => props.onChange(e.target.value)}
        className="bg-transparent border-0 text-[11.5px] font-medium text-[var(--text)] focus:outline-none max-w-[230px]"
      >
        <option value="">None</option>
        <option value="heuristic">Heuristic</option>
        {props.checkpoints.map((c) => (
          <option key={c.ref} value={c.ref}>
            {c.run_name} · u{c.iteration}
          </option>
        ))}
      </select>
    </label>
  );
}

function PuzzlesFilterBar(props: {
  data: PuzzlesResponse | null;
  checkpoints: GlobalCheckpointOption[];
  phases: string[];
  modelA: string;
  modelB: string;
  labelA: string;
  labelB: string;
  phase: string;
  source: string;
  onChangeModelA: (val: string) => void;
  onChangeModelB: (val: string) => void;
  onChangePhase: (val: string) => void;
  onChangeSource: (val: string) => void;
}): ReactNode {
  const { data } = props;
  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2 flex flex-wrap items-center justify-between gap-2.5 shrink-0">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] font-medium text-[var(--text-3)]">Model A</span>
        <PuzzleModelSelect
          ariaLabel="Puzzle Model A"
          dotClassName="bg-[var(--accent)]"
          value={props.modelA}
          checkpoints={props.checkpoints}
          onChange={props.onChangeModelA}
        />
        <span className="text-[11px] font-medium text-[var(--text-3)]">vs</span>
        <PuzzleModelSelect
          ariaLabel="Puzzle Model B"
          dotClassName="bg-[var(--warn)]"
          value={props.modelB}
          checkpoints={props.checkpoints}
          onChange={props.onChangeModelB}
        />
        {data && data.total > 0 ? (
          <span className="tabular text-[11.5px] text-[var(--text-2)] ml-1">
            {props.labelA}:{" "}
            <strong className="text-[var(--text)]">
              {data.solved_a}/{data.total}
            </strong>{" "}
            / {props.labelB}:{" "}
            <strong className="text-[var(--text)]">
              {data.solved_b}/{data.total}
            </strong>
          </span>
        ) : null}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <select
          aria-label="Filter puzzle decision"
          value={props.phase}
          onChange={(e): void => props.onChangePhase(e.target.value)}
          className="h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-[11.5px] font-medium text-[var(--text)]"
        >
          <option value="">All decisions</option>
          {props.phases.map((ph) => (
            <option key={ph} value={ph}>
              {ph}
            </option>
          ))}
        </select>
        <select
          aria-label="Filter puzzle source"
          value={props.source}
          onChange={(e): void => props.onChangeSource(e.target.value)}
          className="h-7 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-[11.5px] font-medium text-[var(--text)]"
        >
          <option value="">All sources</option>
          <option value="auto_mined">Auto-mined</option>
          <option value="search_disagreement">Search disagreement</option>
          <option value="watchlist">Watchlist</option>
        </select>
      </div>
    </section>
  );
}

function filterPuzzlesBySearch(puzzles: PuzzleCardItem[], search: string): PuzzleCardItem[] {
  const needle = search.trim().toLowerCase();
  if (!needle) return puzzles;
  return puzzles.filter((p) =>
    [
      p.phase,
      p.seed_label,
      p.hero_summary,
      p.opp_summary,
      ...p.key_cards,
      ...p.actions.map((a) => a.label),
    ].some((field) => field.toLowerCase().includes(needle)),
  );
}

function PuzzlesGridBody(props: {
  visible: PuzzleCardItem[];
  format: string;
  loading: boolean;
  error: string | null;
  hasData: boolean;
  labelA: string;
  labelB: string;
  onOpenTabletop: (target: GameTarget) => void;
}): ReactNode {
  if (props.loading && !props.hasData) {
    return <div className="text-xs text-[var(--text-3)] p-4">Loading puzzles…</div>;
  }
  if (props.error && !props.hasData) {
    return <div className="text-xs text-[var(--loss)] p-4">{props.error}</div>;
  }
  if (props.visible.length === 0) {
    return <PuzzlesEmptyState format={props.format} />;
  }
  return (
    <section className="grid grid-cols-1 xl:grid-cols-2 gap-3 items-start">
      {props.visible.map((puzzle) => (
        <PuzzleCard
          key={puzzle.puzzle_id}
          puzzle={puzzle}
          labelA={props.labelA}
          labelB={props.labelB}
          onOpenTabletop={props.onOpenTabletop}
        />
      ))}
    </section>
  );
}

function useObservedPhases(
  activeFormat: string,
  puzzles: PuzzleCardItem[],
  currentPhase: string,
): string[] {
  const [seenByFormat, setSeenByFormat] = useState<Record<string, string[]>>({});
  const prevPhases = seenByFormat[activeFormat] ?? [];
  const merged = new Set([
    ...prevPhases,
    ...puzzles.map((p) => p.phase),
    ...(currentPhase ? [currentPhase] : []),
  ]);
  const phases = Array.from(merged).toSorted((a, b) => a.localeCompare(b));
  if (activeFormat && phases.length > prevPhases.length) {
    setSeenByFormat((prev) => ({ ...prev, [activeFormat]: phases }));
  }
  return phases;
}

export function PuzzlesContent(props: {
  data: PuzzlesResponse | null;
  loading: boolean;
  error: string | null;
  checkpoints: GlobalCheckpointOption[];
  modelA: string;
  modelB: string;
  phase: string;
  source: string;
  search: string;
  onChangeModelA: (val: string) => void;
  onChangeModelB: (val: string) => void;
  onChangePhase: (val: string) => void;
  onChangeSource: (val: string) => void;
  onOpenTabletop: (target: GameTarget) => void;
}): ReactNode {
  const { data, checkpoints, modelA, modelB, phase } = props;
  const allPuzzles = data?.puzzles ?? [];
  const phases = useObservedPhases(data?.format ?? "", allPuzzles, phase);
  const visible = filterPuzzlesBySearch(allPuzzles, props.search);
  const labelA = shortModelName(modelA, checkpoints);
  const labelB = shortModelName(modelB, checkpoints);

  return (
    <div className="flex flex-col gap-3">
      <PuzzlesFilterBar
        data={data}
        checkpoints={checkpoints}
        phases={phases}
        modelA={modelA}
        modelB={modelB}
        labelA={labelA}
        labelB={labelB}
        phase={props.phase}
        source={props.source}
        onChangeModelA={props.onChangeModelA}
        onChangeModelB={props.onChangeModelB}
        onChangePhase={props.onChangePhase}
        onChangeSource={props.onChangeSource}
      />
      <PuzzlesGridBody
        visible={visible}
        format={data?.format ?? ""}
        loading={props.loading}
        error={props.error}
        hasData={data !== null}
        labelA={labelA}
        labelB={labelB}
        onOpenTabletop={props.onOpenTabletop}
      />
    </div>
  );
}

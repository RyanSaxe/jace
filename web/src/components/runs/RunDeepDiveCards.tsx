/** Checkpoint ladder and recursive config diff cards for the Run Deep Dive view. */

import { ArrowRight, GitCompareArrows, SlidersHorizontal } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";
import { Badge, Button } from "@/components/ui/Primitives";
import type { CheckpointInfo, CompareTarget, MetricPoint, RunSummary } from "@/lib/types";
import {
  discoverNumericKeys,
  formatCheckpointRef,
  formatMetricValue,
  humanizeMetricKey,
  isRateMetric,
  numericFromPoint,
} from "./metrics";

interface DiffRow {
  key: string;
  left: string;
  right: string;
  changed: boolean;
}

export function isScalar(value: unknown): value is string | number | boolean {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean";
}

function formatBytes(bytes: number): string {
  if (bytes <= 0) return "—";
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}

function flattenArray(items: unknown[], prefix: string, out: Map<string, string>): void {
  const allScalar = items.every((item) => item === null || isScalar(item));
  if (allScalar && prefix) {
    out.set(prefix, `[${items.map((v) => String(v)).join(", ")}]`);
    return;
  }
  items.forEach((item, idx) => {
    flattenConfigValue(item, `${prefix}[${idx}]`, out);
  });
}

function flattenConfigValue(value: unknown, prefix: string, out: Map<string, string>): void {
  if (value === null || value === undefined || isScalar(value)) {
    if (prefix) out.set(prefix, String(value));
    return;
  }
  if (Array.isArray(value)) {
    flattenArray(value, prefix, out);
    return;
  }
  if (typeof value === "object") {
    for (const [k, v] of Object.entries(value)) {
      flattenConfigValue(v, prefix ? `${prefix}.${k}` : k, out);
    }
  }
}

function flattenConfig(config: Record<string, unknown> | undefined): Map<string, string> {
  const out = new Map<string, string>();
  if (config) flattenConfigValue(config, "", out);
  return out;
}

function findEvalPointForIter(points: MetricPoint[], iteration: number): MetricPoint | null {
  if (iteration < 0) return null;
  let best: MetricPoint | null = null;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const pt of points) {
    const dist = Math.abs(pt.update - iteration);
    if (dist < bestDist) {
      bestDist = dist;
      best = pt;
    }
  }
  return bestDist <= 50 ? best : null;
}

export function resolvePairIndices(
  selectedNames: string[],
  checkpoints: CheckpointInfo[],
): [string, string] | null {
  const valid = selectedNames.filter((name) => checkpoints.some((c) => c.name === name));
  if (valid.length >= 2) {
    const a = valid[0];
    const b = valid[1];
    if (a && b) return [a, b];
  }
  if (checkpoints.length >= 2) {
    const first = checkpoints[0];
    const last = checkpoints[checkpoints.length - 1];
    if (first && last) return [first.name, last.name];
  }
  return null;
}

export function CheckpointLadderCard(props: {
  runId: string;
  checkpoints: CheckpointInfo[];
  evalPoints: MetricPoint[];
  evalIterations: number[];
  selectedCkpts: string[];
  onToggleCkpt: (name: string) => void;
  onComparePair: (target: CompareTarget) => void;
}): ReactNode {
  const evalKeys = useMemo(
    () => discoverNumericKeys(props.evalPoints).slice(0, 4),
    [props.evalPoints],
  );
  const evalSet = useMemo(() => new Set(props.evalIterations), [props.evalIterations]);
  const pair = resolvePairIndices(props.selectedCkpts, props.checkpoints);

  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 border-b border-[var(--border)]">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-2)]">
            Checkpoint Ladder
          </h3>
          <Badge>{props.checkpoints.length}</Badge>
        </div>
        <Button
          variant="primary"
          size="sm"
          disabled={pair === null}
          onClick={(): void => {
            if (!pair) return;
            props.onComparePair({
              ckptA: formatCheckpointRef(props.runId, pair[0]),
              ckptB: formatCheckpointRef(props.runId, pair[1]),
              runId: props.runId,
            });
          }}
        >
          <GitCompareArrows className="w-3.5 h-3.5" />
          {pair ? `Compare ${pair[0]} vs ${pair[1]}` : "Select 2 checkpoints"}
          <ArrowRight className="w-3.5 h-3.5" />
        </Button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[540px] border-collapse text-left text-xs">
          <thead>
            <tr className="border-b border-[var(--border)] bg-[var(--surface-2)] text-[10.5px] uppercase tracking-wider text-[var(--text-3)]">
              <th className="py-2 px-3 w-10">Pick</th>
              <th className="py-2 px-2">Checkpoint</th>
              <th className="py-2 px-2 text-right">Update</th>
              {evalKeys.map((key) => (
                <th key={key} className="py-2 px-2 text-right whitespace-nowrap">
                  {humanizeMetricKey(key)}
                </th>
              ))}
              <th className="py-2 px-2 text-right">Size</th>
              <th className="py-2 px-3 text-right">Games</th>
            </tr>
          </thead>
          <tbody>
            {props.checkpoints.length === 0 ? (
              <tr>
                <td colSpan={5 + evalKeys.length} className="py-6 text-center text-[var(--text-3)]">
                  No checkpoints saved for this run yet.
                </td>
              </tr>
            ) : (
              props.checkpoints.toReversed().map((ckpt) => {
                const pickIdx = props.selectedCkpts.indexOf(ckpt.name);
                const evalPt = findEvalPointForIter(props.evalPoints, ckpt.iteration);
                return (
                  <tr
                    key={ckpt.name}
                    className="border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--surface-2)] transition-colors"
                    style={pickIdx === -1 ? undefined : { background: "var(--accent-soft)" }}
                  >
                    <td className="py-2 px-3">
                      <input
                        type="checkbox"
                        checked={pickIdx !== -1}
                        onChange={(): void => props.onToggleCkpt(ckpt.name)}
                        aria-label={`Select checkpoint ${ckpt.name}`}
                        className="accent-[var(--accent)] cursor-pointer"
                      />
                    </td>
                    <td className="py-2 px-2 font-medium tabular">
                      <span className="inline-flex items-center gap-1.5">
                        {ckpt.name}
                        {pickIdx === 0 ? <Badge tone="accent">A</Badge> : null}
                        {pickIdx === 1 ? <Badge tone="warn">B</Badge> : null}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-right tabular text-[var(--text-2)]">
                      {ckpt.iteration >= 0 ? `u${ckpt.iteration}` : "—"}
                    </td>
                    {evalKeys.map((key) => {
                      const val = evalPt ? numericFromPoint(evalPt, key) : null;
                      const asPct = val !== null && isRateMetric(key, [val]);
                      return (
                        <td key={key} className="py-2 px-2 text-right tabular">
                          {val === null ? "—" : formatMetricValue(val, asPct)}
                        </td>
                      );
                    })}
                    <td className="py-2 px-2 text-right tabular text-[var(--text-3)]">
                      {formatBytes(ckpt.size_bytes)}
                    </td>
                    <td className="py-2 px-3 text-right">
                      {evalSet.has(ckpt.iteration) ? (
                        <Badge tone="win">Eval</Badge>
                      ) : (
                        <span className="text-[var(--text-3)]">—</span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function matchesConfigNeedle(row: DiffRow, needle: string): boolean {
  if (!needle) return true;
  return (
    row.key.toLowerCase().includes(needle) ||
    row.left.toLowerCase().includes(needle) ||
    row.right.toLowerCase().includes(needle)
  );
}

function buildConfigDiffRows(args: {
  config: Record<string, unknown> | undefined;
  diffConfig: Record<string, unknown> | undefined;
  diffRunId: string;
  changedOnly: boolean;
  filterText: string;
}): DiffRow[] {
  const leftMap = flattenConfig(args.config);
  const rightMap = args.diffRunId ? flattenConfig(args.diffConfig) : new Map<string, string>();
  const allKeys = new Set<string>([...leftMap.keys(), ...rightMap.keys()]);
  const needle = args.filterText.trim().toLowerCase();
  const out: DiffRow[] = [];
  for (const key of [...allKeys].toSorted((a, b) => a.localeCompare(b))) {
    const left = leftMap.get(key) ?? "—";
    const right = rightMap.get(key) ?? "—";
    const changed = args.diffRunId !== "" && left !== right;
    const row: DiffRow = { key, left, right, changed };
    if (args.diffRunId && args.changedOnly && !changed) continue;
    if (matchesConfigNeedle(row, needle)) out.push(row);
  }
  return out;
}

function ConfigDiffTable(props: {
  runName: string;
  diffLabel: string | null;
  rows: DiffRow[];
  emptyMessage: string;
}): ReactNode {
  return (
    <div className="max-h-[340px] overflow-auto">
      <table className="w-full border-collapse text-left text-xs">
        <thead>
          <tr className="border-b border-[var(--border)] bg-[var(--surface-2)] text-[10.5px] uppercase tracking-wider text-[var(--text-3)]">
            <th className="py-1.5 px-3">Key</th>
            <th className="py-1.5 px-3">{props.runName}</th>
            {props.diffLabel ? <th className="py-1.5 px-3">{props.diffLabel}</th> : null}
          </tr>
        </thead>
        <tbody>
          {props.rows.length === 0 ? (
            <tr>
              <td
                colSpan={props.diffLabel ? 3 : 2}
                className="py-6 text-center text-[var(--text-3)]"
              >
                {props.emptyMessage}
              </td>
            </tr>
          ) : (
            props.rows.map((row) => (
              <tr
                key={row.key}
                className="border-b border-[var(--border)] last:border-b-0"
                style={row.changed ? { background: "var(--warn-soft)" } : undefined}
              >
                <td className="py-1.5 px-3 tabular text-[var(--text-2)] font-medium">{row.key}</td>
                <td className="py-1.5 px-3 tabular break-all">{row.left}</td>
                {props.diffLabel ? (
                  <td className="py-1.5 px-3 tabular break-all text-[var(--text-2)]">
                    {row.right}
                  </td>
                ) : null}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export function ConfigDiffCard(props: {
  run: RunSummary;
  runs: RunSummary[];
  config: Record<string, unknown> | undefined;
  diffRunId: string;
  diffConfig: Record<string, unknown> | undefined;
  onChangeDiffRunId: (id: string) => void;
}): ReactNode {
  const [changedOnly, setChangedOnly] = useState(props.diffRunId !== "");
  const [filterText, setFilterText] = useState("");
  const otherRuns = props.runs.filter((r) => r.run_id !== props.run.run_id);

  const rows = useMemo(
    () =>
      buildConfigDiffRows({
        config: props.config,
        diffConfig: props.diffConfig,
        diffRunId: props.diffRunId,
        changedOnly,
        filterText,
      }),
    [props.config, props.diffConfig, props.diffRunId, changedOnly, filterText],
  );
  const changedCount = rows.filter((r) => r.changed).length;
  const diffLabel = props.diffRunId
    ? (otherRuns.find((r) => r.run_id === props.diffRunId)?.name ?? props.diffRunId)
    : null;
  const emptyMessage =
    props.diffRunId && changedOnly
      ? "No differing config keys between these runs."
      : "No config keys logged.";

  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 border-b border-[var(--border)]">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="w-3.5 h-3.5 text-[var(--accent)]" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-2)]">
            Config &amp; Diff
          </h3>
          <Badge>{rows.length} keys</Badge>
          {props.diffRunId && changedCount > 0 ? (
            <Badge tone="warn">{changedCount} diff</Badge>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="search"
            value={filterText}
            onChange={(e): void => setFilterText(e.target.value)}
            placeholder="Filter keys…"
            aria-label="Filter config keys"
            className="h-7 w-36 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs"
          />
          <select
            value={props.diffRunId}
            onChange={(e): void => {
              const next = e.target.value;
              props.onChangeDiffRunId(next);
              if (next !== "") setChangedOnly(true);
            }}
            aria-label="Compare config against run"
            className="h-7 max-w-[200px] rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 text-xs"
          >
            <option value="">No comparison run</option>
            {otherRuns.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                vs {r.name}
              </option>
            ))}
          </select>
          {props.diffRunId ? (
            <Button
              size="sm"
              variant={changedOnly ? "primary" : "default"}
              onClick={(): void => setChangedOnly((prev) => !prev)}
            >
              Changed only
            </Button>
          ) : null}
        </div>
      </div>
      <ConfigDiffTable
        runName={props.run.name}
        diffLabel={diffLabel}
        rows={rows}
        emptyMessage={emptyMessage}
      />
    </section>
  );
}

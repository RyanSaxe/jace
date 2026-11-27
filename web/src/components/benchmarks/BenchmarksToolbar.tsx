/** Desktop and mobile toolbars for selecting Benchmark format, Elo anchor, sub-view, and search filter. */

import { RefreshCw, Trophy } from "lucide-react";
import type { ReactNode } from "react";
import { SegmentedControl } from "@/components/ui/Primitives";
import type { ArenaModelRow, BenchmarkSubView } from "@/lib/types";

function resolveFormatOptions(formats: string[], format: string): string[] {
  if (formats.length > 0) return formats;
  return format ? [format] : [];
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

export function BenchmarksToolbar(props: {
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

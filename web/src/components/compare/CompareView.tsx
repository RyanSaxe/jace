/** 3-Pane Compare Models workbench: cross-run checkpoint scanner, SVG analytics, diff table, and docked position preview. */

import { GitCompareArrows, Play } from "lucide-react";
import { type ReactNode, useRef, useState } from "react";
import { Button } from "@/components/ui/Primitives";
import { useAbortOnUnmount } from "@/hooks/useStudioData";
import type { ComparePositionRow, CompareResponse } from "@/lib/types";
import { DecisionKindChart, SummaryKpiStrip, WinScatterChart } from "./CompareCharts";
import { CompareDiffTable, CompareNotices, CompareToolbar } from "./CompareParts";
import { PositionPreviewPane } from "./ComparePreview";
import {
  type CompareViewProps,
  type FilterState,
  INIT_FILTER,
  INIT_SCAN,
  type OvState,
  type ScanState,
  clampSamples,
  deriveViewData,
  executeCompareScan,
  resolveSelection,
  shortRef,
} from "./compareUtils";

export function CompareView(props: CompareViewProps): ReactNode {
  const [ov, setOv] = useState<OvState>({ a: null, b: null, run: null });
  const sel = resolveSelection(props, ov);
  const [prevKey, setPrevKey] = useState(sel.targetKey);
  const [sampleText, setSampleText] = useState("120");
  const [scan, setScan] = useState<ScanState>(INIT_SCAN);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [filter, setFilter] = useState<FilterState>(INIT_FILTER);
  const [selectedPosKey, setSelectedPosKey] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  useAbortOnUnmount(abortRef);
  if (sel.targetKey !== prevKey) {
    setPrevKey(sel.targetKey);
    setOv({ a: null, b: null, run: null });
  }
  const sampleCount = clampSamples(sampleText);
  const view = deriveViewData(result, sel, { filter, selectedPosKey });

  const openPosition = (pos: ComparePositionRow): void => {
    props.onOpenTabletop({
      run_id: pos.run_id,
      iteration: pos.iteration,
      game_idx: pos.game_idx,
      step: pos.step,
      seed: pos.seed,
      ckpt_a: view.ckptA,
      ckpt_b: view.ckptB,
    });
  };

  const runScan = (): void => {
    setSampleText(String(sampleCount));
    setScan({ loading: true, stopped: false, error: null });
    executeCompareScan(
      { ckptA: sel.ckptA, ckptB: sel.ckptB, sampleCount, runId: sel.runId },
      abortRef,
      {
        onResult: (res): void => {
          setResult(res);
          setScan(INIT_SCAN);
          setFilter(INIT_FILTER);
          setSelectedPosKey(null);
        },
        onDone: (error): void => setScan((p) => ({ ...p, loading: false, error })),
      },
    );
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <CompareToolbar
        sel={sel}
        sampleText={sampleText}
        loading={scan.loading}
        runs={props.runs}
        checkpoints={props.checkpoints}
        onPatchOv={(patch): void => setOv((p) => ({ ...p, ...patch }))}
        onSamples={setSampleText}
        onRun={runScan}
        onStop={(): void => {
          abortRef.current?.abort();
          abortRef.current = null;
          setScan({ loading: false, stopped: true, error: null });
        }}
      />
      <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
        <div className="flex-1 min-w-0 lg:overflow-y-auto p-3.5 flex flex-col gap-3">
          <CompareNotices
            scan={scan}
            warnings={view.warnings}
            scanLabel={`Scanning ${sampleCount} sampled turns (${shortRef(sel.ckptA)} vs ${shortRef(sel.ckptB)})…`}
          />
          {result ? (
            <>
              <SummaryKpiStrip result={result} />
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                <DecisionKindChart
                  stats={result.by_decision_kind}
                  activeKind={filter.kind}
                  onToggle={(k): void =>
                    setFilter((p) => ({ ...p, kind: p.kind === k ? null : k }))
                  }
                />
                <WinScatterChart
                  positions={view.filteredRows}
                  selectedKey={view.activeKey}
                  onSelect={setSelectedPosKey}
                />
              </div>
              <CompareDiffTable
                view={view}
                filter={filter}
                onFilter={(patch): void => setFilter((p) => ({ ...p, ...patch }))}
                onSelect={setSelectedPosKey}
                onOpenRow={openPosition}
              />
            </>
          ) : null}
          {!result && !scan.loading ? (
            <div className="rounded-lg border border-dashed border-[var(--border-strong)] bg-[var(--surface)] p-7 text-center flex flex-col items-center gap-2.5 max-w-xl mx-auto my-8">
              <GitCompareArrows className="w-6 h-6 text-[var(--accent)]" />
              <h3 className="text-sm font-semibold m-0">
                Compare any two checkpoints across real game turns
              </h3>
              <p className="text-xs text-[var(--text-2)] m-0">
                Pick Model A and Model B above, set turns to sample, and inspect move changes and
                win% shifts.
              </p>
              <Button variant="primary" size="md" onClick={runScan}>
                <Play className="w-3.5 h-3.5 fill-current" /> Run Compare ({sampleCount} turns)
              </Button>
            </div>
          ) : null}
        </div>
        <PositionPreviewPane position={view.activePos} onOpen={openPosition} />
      </div>
    </div>
  );
}

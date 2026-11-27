/** Training Runs Option A: Full-width multi-plot grid, multi-select sortable table, mobile cards + sheet, and Deep Dive. */

import { type ReactNode, useMemo, useState } from "react";
import { useExtraRunDetails } from "@/hooks/useStudioData";
import type { CompareTarget, GlobalCheckpointOption, RunDetail, RunSummary } from "@/lib/types";
import {
  type XAxisMode,
  colorForIndex,
  discoverCatalogMetrics,
  discoverPopulatedMetricIds,
  isWinRateKey,
  selectDefaultMetricIds,
} from "./metrics";
import { RunDeepDive } from "./RunDeepDive";
import { PlotsGridSection, RunsLegendBar } from "./RunsCharts";
import { MobileRunBottomSheet, RunsToolbar, StickySelectionBar } from "./RunsOverlays";
import { DesktopRunsTable, MobileRunCards, type SortKey, filterAndSortRuns } from "./RunsTable";

export interface RunsViewProps {
  runs: RunSummary[];
  details: Record<string, RunDetail>;
  checkpoints: GlobalCheckpointOption[];
  experiments: string[];
  evalMetricKeys: string[];
  trainMetricKeys: string[];
  selectedRunId: string | null;
  onSelectRunDeepDive: (runId: string | null) => void;
  loading: boolean;
  error: string | null;
  onReload: () => void;
  onBrowseGames: (runId: string) => void;
  onComparePair: (target: CompareTarget) => void;
}

function toggleStringId(base: string[], target: string): string[] {
  return base.includes(target) ? base.filter((id) => id !== target) : [...base, target];
}

export function RunsView(props: RunsViewProps): ReactNode {
  const [activeExperiment, setActiveExperiment] = useState("");
  const [filterText, setFilterText] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("updated");
  const [sortDir, setSortDir] = useState<"desc" | "asc">("desc");
  const [smoothing, setSmoothing] = useState(0);
  const [showCi, setShowCi] = useState(true);
  const [xMode, setXMode] = useState<XAxisMode>("updates");
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [customCheckedIds, setCustomCheckedIds] = useState<string[] | null>(null);
  const [customMetricIds, setCustomMetricIds] = useState<string[] | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [soloMode, setSoloMode] = useState(false);
  const [mobileSheetRunId, setMobileSheetRunId] = useState<string | null>(null);
  const [pendingDiffRunId, setPendingDiffRunId] = useState<string | null>(null);

  const experiments = useMemo(
    () =>
      props.experiments.length > 0
        ? props.experiments
        : [...new Set(props.runs.map((r) => r.experiment).filter(Boolean))].toSorted((a, b) =>
            a.localeCompare(b),
          ),
    [props.experiments, props.runs],
  );
  const prelimMetrics = useMemo(
    () =>
      discoverCatalogMetrics({
        evalMetricKeys: props.evalMetricKeys,
        trainMetricKeys: props.trainMetricKeys,
        details: props.details,
        runs: props.runs,
      }),
    [props.evalMetricKeys, props.trainMetricKeys, props.details, props.runs],
  );
  const secondaryKey = useMemo(
    () => prelimMetrics.find((m) => m.source === "eval" && !isWinRateKey(m.key))?.key ?? null,
    [prelimMetrics],
  );
  const filteredRuns = useMemo(
    () =>
      filterAndSortRuns({
        runs: props.runs,
        details: props.details,
        experiment: activeExperiment,
        filterText,
        sortKey,
        sortDir,
        secondaryKey,
      }),
    [props.runs, props.details, activeExperiment, filterText, sortKey, sortDir, secondaryKey],
  );
  const checkedIds = useMemo(
    () => customCheckedIds ?? filteredRuns.slice(0, 4).map((r) => r.run_id),
    [customCheckedIds, filteredRuns],
  );
  const allDetails = useExtraRunDetails(checkedIds, props.details);
  const allMetrics = useMemo(
    () =>
      discoverCatalogMetrics({
        evalMetricKeys: props.evalMetricKeys,
        trainMetricKeys: props.trainMetricKeys,
        details: allDetails,
        runs: props.runs,
      }),
    [props.evalMetricKeys, props.trainMetricKeys, allDetails, props.runs],
  );
  const selectedMetricIds = useMemo(() => {
    const valid = new Set(allMetrics.map((m) => m.id));
    if (customMetricIds !== null) return new Set(customMetricIds.filter((id) => valid.has(id)));
    return new Set(
      selectDefaultMetricIds(allMetrics, 6, discoverPopulatedMetricIds(checkedIds, allDetails)),
    );
  }, [customMetricIds, allMetrics, checkedIds, allDetails]);

  const visibleMetrics = useMemo(
    () => allMetrics.filter((m) => selectedMetricIds.has(m.id)),
    [allMetrics, selectedMetricIds],
  );
  const checkedSet = useMemo(() => new Set(checkedIds), [checkedIds]);
  const colorByRunId = useMemo(
    () => new Map(checkedIds.map((id, idx) => [id, colorForIndex(idx)])),
    [checkedIds],
  );
  const checkedRuns = useMemo(() => {
    const byId = new Map(props.runs.map((r) => [r.run_id, r]));
    return checkedIds.map((id) => byId.get(id)).filter((r): r is RunSummary => r !== undefined);
  }, [props.runs, checkedIds]);
  const effectiveSolo = soloMode && checkedRuns.length >= 2;
  const tableRuns = effectiveSolo
    ? filteredRuns.filter((r) => checkedSet.has(r.run_id))
    : filteredRuns;

  const deepDiveRun = props.selectedRunId
    ? (props.runs.find((r) => r.run_id === props.selectedRunId) ?? null)
    : null;
  if (deepDiveRun) {
    return (
      <RunDeepDive
        run={deepDiveRun}
        runs={props.runs}
        details={allDetails}
        initialDiffRunId={pendingDiffRunId}
        onBack={(): void => {
          setPendingDiffRunId(null);
          props.onSelectRunDeepDive(null);
        }}
        onBrowseGames={props.onBrowseGames}
        onComparePair={props.onComparePair}
      />
    );
  }

  const openDeepDive = (runId: string): void => {
    setPendingDiffRunId(null);
    props.onSelectRunDeepDive(runId);
  };
  const toggleCheck = (runId: string): void =>
    setCustomCheckedIds((prev) => toggleStringId(prev ?? checkedIds, runId));
  const mobileSheetRun = props.runs.find((r) => r.run_id === mobileSheetRunId) ?? null;

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      <RunsToolbar
        runCount={filteredRuns.length}
        experiments={experiments}
        activeExperiment={activeExperiment}
        filterText={filterText}
        smoothing={smoothing}
        showCi={showCi}
        xMode={xMode}
        allMetrics={allMetrics}
        selectedMetricIds={selectedMetricIds}
        pickerOpen={pickerOpen}
        loading={props.loading}
        onSelectExperiment={setActiveExperiment}
        onChangeFilter={setFilterText}
        onToggleSmoothing={(): void => setSmoothing((s) => (s > 0 ? 0 : 0.6))}
        onToggleCi={(): void => setShowCi((v) => !v)}
        onChangeXMode={setXMode}
        onTogglePicker={(): void => setPickerOpen((v) => !v)}
        onToggleMetric={(id): void =>
          setCustomMetricIds((prev) => toggleStringId(prev ?? [...selectedMetricIds], id))
        }
        onResetMetrics={(): void => setCustomMetricIds(null)}
        onReload={props.onReload}
      />

      <div className="flex-1 overflow-y-auto p-3.5 sm:p-4 flex flex-col gap-3">
        {props.error ? (
          <div className="rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] p-3 text-xs text-[var(--loss)]">
            {props.error}
          </div>
        ) : null}

        <RunsLegendBar
          checkedRuns={checkedRuns}
          colorByRunId={colorByRunId}
          hoverX={hoverX}
          xMode={xMode}
          onDeepDive={openDeepDive}
          onUncheck={(runId): void =>
            setCustomCheckedIds((prev) => (prev ?? checkedIds).filter((id) => id !== runId))
          }
        />

        <PlotsGridSection
          visibleMetrics={visibleMetrics}
          checkedRuns={checkedRuns}
          details={allDetails}
          colorByRunId={colorByRunId}
          smoothing={smoothing}
          showCi={showCi}
          xMode={xMode}
          hoverX={hoverX}
          loading={props.loading}
          onHoverX={setHoverX}
        />

        <DesktopRunsTable
          runs={tableRuns}
          details={allDetails}
          checkedSet={checkedSet}
          colorByRunId={colorByRunId}
          secondaryKey={secondaryKey}
          sortKey={sortKey}
          sortDir={sortDir}
          loading={props.loading}
          onSort={(nextKey): void => {
            if (sortKey === nextKey) setSortDir((d) => (d === "desc" ? "asc" : "desc"));
            else {
              setSortKey(nextKey);
              setSortDir("desc");
            }
          }}
          onToggleCheck={toggleCheck}
          onToggleAll={(): void => {
            const all =
              filteredRuns.length > 0 && filteredRuns.every((r) => checkedSet.has(r.run_id));
            setCustomCheckedIds(all ? [] : filteredRuns.map((r) => r.run_id));
          }}
          onDeepDive={openDeepDive}
          onBrowseGames={props.onBrowseGames}
        />

        <MobileRunCards
          runs={tableRuns}
          checkedSet={checkedSet}
          colorByRunId={colorByRunId}
          loading={props.loading}
          onToggleCheck={toggleCheck}
          onOpenSheet={(run): void => setMobileSheetRunId(run.run_id)}
          onBrowseGames={props.onBrowseGames}
        />

        {customCheckedIds === null ? null : (
          <StickySelectionBar
            checkedRuns={checkedRuns}
            colorByRunId={colorByRunId}
            details={allDetails}
            checkpoints={props.checkpoints}
            soloMode={effectiveSolo}
            onToggleSolo={(): void => setSoloMode((v) => !v)}
            onDiffConfigs={(): void => {
              const first = checkedRuns[0];
              const second = checkedRuns[1];
              if (!first || !second) return;
              setPendingDiffRunId(second.run_id);
              props.onSelectRunDeepDive(first.run_id);
            }}
            onComparePair={props.onComparePair}
            onClear={(): void => {
              setCustomCheckedIds(null);
              setSoloMode(false);
            }}
          />
        )}
      </div>

      <MobileRunBottomSheet
        run={mobileSheetRun}
        checked={mobileSheetRun ? checkedSet.has(mobileSheetRun.run_id) : false}
        onClose={(): void => setMobileSheetRunId(null)}
        onDeepDive={openDeepDive}
        onBrowseGames={props.onBrowseGames}
        onToggleChart={toggleCheck}
      />
    </div>
  );
}

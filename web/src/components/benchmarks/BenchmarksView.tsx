/** Benchmarks workbench: Elo Arena (Bradley-Terry ratings, forecast, Match Ledger) and Tactical Puzzles. */

import { type ReactNode, useState } from "react";
import { useArenaLeaderboard, useArenaMatches, usePuzzlesData } from "@/hooks/useBenchmarksData";
import type {
  ArenaLeaderboardResponse,
  ArenaModelRow,
  BenchmarkSubView,
  BenchmarksRouteState,
  CompareTarget,
  GameTarget,
  GlobalCheckpointOption,
  PuzzlesResponse,
} from "@/lib/types";
import { ArenaContent } from "./ArenaContent";
import { PuzzlesContent } from "./BenchmarksPuzzles";
import { BenchmarksToolbar } from "./BenchmarksToolbar";

export interface BenchmarksViewProps {
  checkpoints: GlobalCheckpointOption[];
  routeState: BenchmarksRouteState;
  onChangeRouteState: (next: BenchmarksRouteState) => void;
  onOpenTabletop: (target: GameTarget) => void;
  onComparePair: (target: CompareTarget) => void;
}

const NONE_MODEL_SENTINEL = "__none__";

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

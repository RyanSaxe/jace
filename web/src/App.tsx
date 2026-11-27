/** Top-level Jace Studio shell: adaptive Linear rail, URL deep links, and full-workbench routing. */

import { Menu } from "lucide-react";
import { type ReactNode, useCallback, useState } from "react";
import { INITIAL_GAMES_FILTERS, SHORTCUT_TO_TAB, buildHashUrl, parseHashState } from "@/app/routes";
import { SidebarContent } from "@/app/Sidebar";
import { BenchmarksView } from "@/components/benchmarks";
import { CompareView } from "@/components/compare";
import { GamesExplorerView } from "@/components/games";
import { RunsView } from "@/components/runs";
import { TabletopView } from "@/components/tabletop";
import { Badge } from "@/components/ui/Primitives";
import { WatchlistView } from "@/components/watchlist";
import {
  loadSavedTheme,
  useKeyboardShortcut,
  useRunsData,
  useThemeSync,
  useUrlHashSync,
  useWatchlistData,
} from "@/hooks/useStudioData";
import { addWatchlistItem, removeWatchlistItem, updateWatchlistItem } from "@/lib/api";
import type {
  BenchmarksRouteState,
  BoardState,
  CompareTarget,
  GameTarget,
  GamesFilterState,
  NavTab,
  ThemeMode,
} from "@/lib/types";

export function App(): ReactNode {
  const initialUrl = parseHashState();
  const [tab, setTab] = useState<NavTab>(initialUrl.tab);
  const [deepDiveRunId, setDeepDiveRunId] = useState<string | null>(initialUrl.deepDiveRunId);
  const [benchmarksState, setBenchmarksState] = useState<BenchmarksRouteState>(
    initialUrl.benchmarksState,
  );
  const [compareTarget, setCompareTarget] = useState<CompareTarget | null>(
    initialUrl.compareTarget,
  );
  const [theme, setTheme] = useState<ThemeMode>(loadSavedTheme);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [gamesFilters, setGamesFilters] = useState<GamesFilterState>(INITIAL_GAMES_FILTERS);
  const [activeGame, setActiveGame] = useState<GameTarget | null>(initialUrl.gameTarget);
  const [actionError, setActionError] = useState<string | null>(null);

  useThemeSync(theme);
  const {
    runs,
    experiments,
    evalMetricKeys,
    trainMetricKeys,
    checkpoints,
    details,
    loading: runsLoading,
    error: runsError,
    reload,
  } = useRunsData();
  const {
    items: watchlist,
    setItems: setWatchlist,
    loading: watchlistLoading,
  } = useWatchlistData();

  const handleHashChange = useCallback((): void => {
    const parsed = parseHashState();
    setTab(parsed.tab);
    setDeepDiveRunId(parsed.deepDiveRunId);
    if (parsed.tab === "benchmarks") setBenchmarksState(parsed.benchmarksState);
    if (parsed.compareTarget) setCompareTarget(parsed.compareTarget);
    if (parsed.gameTarget) setActiveGame(parsed.gameTarget);
  }, []);

  useUrlHashSync(
    buildHashUrl({ tab, deepDiveRunId, benchmarksState, compareTarget, gameTarget: activeGame }),
    handleHashChange,
  );

  const handleShortcut = useCallback((event: KeyboardEvent): void => {
    if (event.key === "Escape") {
      setMobileMenuOpen(false);
      return;
    }
    const nextTab = SHORTCUT_TO_TAB[event.key];
    if (nextTab) setTab(nextTab);
  }, []);
  useKeyboardShortcut(handleShortcut);

  const handleSelectNavTab = (nextTab: NavTab): void => {
    if (nextTab === "runs") setDeepDiveRunId(null);
    setTab(nextTab);
    setMobileMenuOpen(false);
  };

  const openTabletop = (target: GameTarget): void => {
    setActiveGame(target);
    setTab("tabletop");
    setMobileMenuOpen(false);
  };

  const handlePinPosition = async (args: {
    board: BoardState;
    target: GameTarget;
    label?: string;
    note?: string;
    tag?: string;
    keyCards?: string[];
    summary?: string;
  }): Promise<void> => {
    const { board, target, label, note, tag, keyCards, summary } = args;
    setActionError(null);
    const created = await addWatchlistItem({
      run_id: target.run_id,
      iteration: target.iteration,
      game_idx: target.game_idx,
      step: board.step,
      turn: board.turn,
      seed: target.seed,
      label: label?.trim() || `T${board.turn} · ${board.step_name || board.action_space_kind}`,
      note: note?.trim() || `Hero ${board.hero.life} vs Opp ${board.opp.life}`,
      tag: tag || undefined,
      summary:
        summary ||
        `Hero ${board.hero.life} vs Opp ${board.opp.life} · ${board.hero.creatures.length + board.opp.creatures.length} creatures`,
      key_cards: keyCards,
    });
    setWatchlist((prev) => [created, ...prev]);
  };

  const handleRemoveWatchlist = (itemId: string): void => {
    setActionError(null);
    void removeWatchlistItem(itemId)
      .then(() => {
        setWatchlist((prev) => prev.filter((item) => item.id !== itemId));
      })
      .catch((err: unknown) => {
        setActionError(err instanceof Error ? err.message : "Failed to remove watchlist item");
      });
  };

  const handleComparePair = (target: CompareTarget): void => {
    setCompareTarget(target);
    setTab("compare");
    setMobileMenuOpen(false);
  };

  const handleOpenPuzzles = (format: string): void => {
    setBenchmarksState((prev) => ({ ...prev, subView: "puzzles", format }));
    setTab("benchmarks");
    setMobileMenuOpen(false);
  };

  const runningCount = runs.filter((r) => r.status === "running").length;
  const desktopExpanded = tab !== "tabletop";
  const bannerError = runsError ? `Backend connection error: ${runsError}` : actionError;

  const activeMainView = (): ReactNode => {
    switch (tab) {
      case "runs":
        return (
          <RunsView
            runs={runs}
            details={details}
            checkpoints={checkpoints}
            experiments={experiments}
            evalMetricKeys={evalMetricKeys}
            trainMetricKeys={trainMetricKeys}
            selectedRunId={deepDiveRunId}
            onSelectRunDeepDive={setDeepDiveRunId}
            loading={runsLoading}
            error={runsError}
            onReload={reload}
            onBrowseGames={(runId): void => {
              setGamesFilters({ ...INITIAL_GAMES_FILTERS, runId });
              setTab("games");
            }}
            onComparePair={handleComparePair}
          />
        );
      case "games":
        return (
          <GamesExplorerView
            runs={runs}
            filters={gamesFilters}
            onChangeFilters={setGamesFilters}
            onOpenTabletop={openTabletop}
          />
        );
      case "benchmarks":
        return (
          <BenchmarksView
            checkpoints={checkpoints}
            routeState={benchmarksState}
            onChangeRouteState={setBenchmarksState}
            onOpenTabletop={openTabletop}
            onComparePair={handleComparePair}
          />
        );
      case "compare":
        return (
          <CompareView
            runs={runs}
            checkpoints={checkpoints}
            initialTarget={compareTarget}
            onOpenTabletop={openTabletop}
          />
        );
      case "tabletop":
        return (
          <TabletopView
            target={activeGame}
            checkpoints={checkpoints}
            watchlist={watchlist}
            onChangeStep={(step): void =>
              setActiveGame((prev) => (prev ? { ...prev, step } : null))
            }
            onPinPosition={handlePinPosition}
            onBackToGames={(): void => setTab("games")}
          />
        );
      case "watchlist":
        return (
          <WatchlistView
            items={watchlist}
            loading={watchlistLoading}
            onOpenTabletop={openTabletop}
            onRemove={handleRemoveWatchlist}
            onUpdate={async (itemId, patch): Promise<void> => {
              setActionError(null);
              const updated = await updateWatchlistItem(itemId, patch);
              setWatchlist((prev) => prev.map((item) => (item.id === itemId ? updated : item)));
            }}
            onOpenPuzzles={handleOpenPuzzles}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex h-[100dvh] w-full overflow-hidden bg-[var(--bg)] text-[var(--text)]">
      <aside
        aria-label="Sidebar navigation"
        className={`hidden lg:flex shrink-0 flex-col border-r border-[var(--border)] bg-[var(--surface)] transition-[width] duration-150 ${
          desktopExpanded ? "w-[224px]" : "w-[52px]"
        }`}
      >
        <SidebarContent
          tab={tab}
          theme={theme}
          expanded={desktopExpanded}
          isDrawer={false}
          runningCount={runningCount}
          watchlistCount={watchlist.length}
          onSelectTab={handleSelectNavTab}
          onCloseDrawer={(): void => setMobileMenuOpen(false)}
          onToggleTheme={(): void => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
        />
      </aside>

      {mobileMenuOpen ? (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div className="w-[260px] bg-[var(--surface)] border-r border-[var(--border)] shadow-xl">
            <SidebarContent
              tab={tab}
              theme={theme}
              expanded
              isDrawer
              runningCount={runningCount}
              watchlistCount={watchlist.length}
              onSelectTab={handleSelectNavTab}
              onCloseDrawer={(): void => setMobileMenuOpen(false)}
              onToggleTheme={(): void => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
            />
          </div>
          <button
            type="button"
            aria-label="Dismiss navigation"
            className="flex-1 bg-black/40 border-0"
            onClick={(): void => setMobileMenuOpen(false)}
          />
        </div>
      ) : null}

      <div className="flex-1 min-w-0 flex flex-col h-full overflow-hidden">
        <header className="flex lg:hidden items-center justify-between px-3 py-1.5 border-b border-[var(--border)] bg-[var(--surface)] shrink-0">
          <button
            type="button"
            onClick={(): void => setMobileMenuOpen(true)}
            aria-label="Open navigation"
            className="w-10 h-10 rounded-md flex items-center justify-center border border-[var(--border)] cursor-pointer"
          >
            <Menu className="w-4 h-4" />
          </button>
          <span className="text-xs font-semibold">Jace Studio</span>
          <Badge tone="accent">{tab}</Badge>
        </header>

        {bannerError ? (
          <div className="mx-4 mt-3 rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] p-2.5 text-xs text-[var(--loss)]">
            {bannerError}
          </div>
        ) : null}

        <main className="flex-1 min-h-0 overflow-hidden">{activeMainView()}</main>
      </div>
    </div>
  );
}

/** Top-level Jace Studio shell: 224 px Linear navigation rail, persisted theme, and full-lifecycle routing. */

import { Activity, Bookmark, Layers, Menu, Moon, Sparkles, Sun, Swords, X } from "lucide-react";
import { type ReactNode, useCallback, useState } from "react";
import { GamesExplorerView } from "@/components/GamesExplorerView";
import { RunsView } from "@/components/RunsView";
import { TabletopView } from "@/components/TabletopView";
import { WatchlistView } from "@/components/WatchlistView";
import { Badge } from "@/components/ui/Primitives";
import {
  loadSavedTheme,
  useKeyboardShortcut,
  useRunsData,
  useThemeSync,
  useWatchlistData,
} from "@/hooks/useStudioData";
import { addWatchlistItem, removeWatchlistItem } from "@/lib/api";
import type { BoardState, GameTarget, GamesFilterState, NavTab, ThemeMode } from "@/lib/types";

const NAV_ITEMS: ReadonlyArray<{ id: NavTab; label: string; shortcut: string }> = [
  { id: "runs", label: "Runs", shortcut: "1" },
  { id: "games", label: "Games", shortcut: "2" },
  { id: "tabletop", label: "Tabletop", shortcut: "3" },
  { id: "watchlist", label: "Watchlist", shortcut: "4" },
];

const INITIAL_GAMES_FILTERS: GamesFilterState = {
  runId: "",
  result: "",
  colors: "",
  card: "",
  search: "",
  sort: "steps_desc",
  page: 0,
};

function NavIcon(props: { id: NavTab }): ReactNode {
  if (props.id === "runs") return <Activity className="w-4 h-4" />;
  if (props.id === "games") return <Layers className="w-4 h-4" />;
  if (props.id === "tabletop") return <Swords className="w-4 h-4" />;
  return <Bookmark className="w-4 h-4" />;
}

export function App(): ReactNode {
  const [tab, setTab] = useState<NavTab>("runs");
  const [theme, setTheme] = useState<ThemeMode>(loadSavedTheme);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [gamesFilters, setGamesFilters] = useState<GamesFilterState>(INITIAL_GAMES_FILTERS);
  const [activeGame, setActiveGame] = useState<GameTarget | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  useThemeSync(theme);
  const { runs, details, loading: runsLoading, error: runsError, reload } = useRunsData();
  const {
    items: watchlist,
    setItems: setWatchlist,
    loading: watchlistLoading,
  } = useWatchlistData();

  const handleShortcut = useCallback((event: KeyboardEvent): void => {
    if (event.key === "Escape") {
      setMobileMenuOpen(false);
    } else if (event.key === "1") {
      setTab("runs");
    } else if (event.key === "2") {
      setTab("games");
    } else if (event.key === "3") {
      setTab("tabletop");
    } else if (event.key === "4") {
      setTab("watchlist");
    }
  }, []);
  useKeyboardShortcut(handleShortcut);

  const openTabletop = (target: GameTarget): void => {
    setActiveGame(target);
    setTab("tabletop");
    setMobileMenuOpen(false);
  };

  const handlePinPosition = async (args: {
    board: BoardState;
    target: GameTarget;
    note?: string;
  }): Promise<void> => {
    const { board, target, note } = args;
    setActionError(null);
    const created = await addWatchlistItem({
      run_id: target.run_id,
      iteration: target.iteration,
      game_idx: target.game_idx,
      step: board.step,
      turn: board.turn,
      seed: target.seed,
      label: `Turn ${board.turn} · ${board.action_space_kind}`,
      note:
        note?.trim() ||
        `Hero ${board.hero.life} vs Opp ${board.opp.life} (${board.hero.zone_counts.battlefield + board.opp.zone_counts.battlefield} perms)`,
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

  const runningCount = runs.filter((r) => r.status === "running").length;

  const renderSidebar = (isDrawer: boolean): ReactNode => (
    <div className="flex flex-col h-full justify-between p-3">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between gap-2 px-1">
          <h1 className="flex items-center gap-2 font-semibold text-[13.5px] tracking-tight m-0">
            <span className="w-2.5 h-2.5 rounded-[3px] bg-[var(--accent)]" />
            <span>Jace Studio</span>
          </h1>
          <div className="flex items-center gap-1.5">
            {runningCount > 0 ? <Badge tone="win">{runningCount} live</Badge> : null}
            {isDrawer ? (
              <button
                type="button"
                onClick={(): void => setMobileMenuOpen(false)}
                aria-label="Close navigation"
                className="w-9 h-9 rounded-md flex items-center justify-center text-[var(--text-2)] hover:bg-[var(--surface-2)] cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            ) : null}
          </div>
        </div>

        <nav className="flex flex-col gap-0.5" aria-label="Primary navigation">
          <div className="px-2 pb-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-3)]">
            Gameplay RL
          </div>
          {NAV_ITEMS.map((item) => {
            const active = tab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                aria-current={active ? "page" : undefined}
                onClick={(): void => {
                  setTab(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`flex items-center justify-between rounded-md px-2.5 py-2 lg:py-1.5 text-xs font-medium cursor-pointer transition-colors ${
                  active
                    ? "bg-[var(--accent-soft)] text-[var(--accent)] font-semibold"
                    : "text-[var(--text-2)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
                }`}
              >
                <span className="flex items-center gap-2">
                  <NavIcon id={item.id} />
                  {item.label}
                  {item.id === "watchlist" && watchlist.length > 0 ? (
                    <Badge tone="accent">{watchlist.length}</Badge>
                  ) : null}
                </span>
                <kbd className="tabular text-[10px] text-[var(--text-3)] font-sans">
                  {item.shortcut}
                </kbd>
              </button>
            );
          })}
        </nav>

        <div className="flex flex-col gap-1 pt-2 border-t border-[var(--border)]">
          <div className="px-2 pb-0.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-3)]">
            Full Lifecycle (Next)
          </div>
          <div className="flex items-center justify-between px-2.5 py-1 text-xs text-[var(--text-3)]">
            <span className="flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5" />
              Draft Pods
            </span>
            <Badge>Soon</Badge>
          </div>
          <div className="flex items-center justify-between px-2.5 py-1 text-xs text-[var(--text-3)]">
            <span className="flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5" />
              Deck Builder
            </span>
            <Badge>Soon</Badge>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-2 pt-3 border-t border-[var(--border)]">
        <button
          type="button"
          onClick={(): void => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
          className="flex items-center justify-between rounded-md border border-[var(--border)] bg-[var(--bg)] px-2.5 py-1.5 text-xs font-medium hover:border-[var(--border-strong)] cursor-pointer"
        >
          <span className="flex items-center gap-2">
            {theme === "light" ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
            {theme === "light" ? "Warm Paper" : "Obsidian Dark"}
          </span>
          <span className="text-[10px] text-[var(--text-3)]">Theme</span>
        </button>
      </div>
    </div>
  );

  return (
    <div className="flex h-full w-full overflow-hidden bg-[var(--bg)] text-[var(--text)]">
      {/* Desktop 224 px fixed left rail (collapsed below lg so tablet gets full width) */}
      <aside
        aria-label="Sidebar navigation"
        className="hidden lg:flex w-[224px] shrink-0 flex-col border-r border-[var(--border)] bg-[var(--surface)]"
      >
        {renderSidebar(false)}
      </aside>

      {/* Tablet / Mobile slide-over nav */}
      {mobileMenuOpen ? (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div className="w-[260px] bg-[var(--surface)] border-r border-[var(--border)] shadow-xl">
            {renderSidebar(true)}
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

        {runsError || actionError ? (
          <div className="mx-4 mt-3 rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] p-2.5 text-xs text-[var(--loss)]">
            {runsError ? `Backend connection error: ${runsError}` : actionError}
          </div>
        ) : null}

        <main className="flex-1 min-h-0 overflow-hidden">
          {tab === "runs" ? (
            <RunsView
              runs={runs}
              details={details}
              loading={runsLoading}
              onReload={reload}
              onBrowseGames={(runId): void => {
                setGamesFilters({ ...INITIAL_GAMES_FILTERS, runId });
                setTab("games");
              }}
            />
          ) : null}

          {tab === "games" ? (
            <GamesExplorerView
              runs={runs}
              filters={gamesFilters}
              onChangeFilters={setGamesFilters}
              onOpenTabletop={openTabletop}
            />
          ) : null}

          {tab === "tabletop" ? (
            <TabletopView
              target={activeGame}
              watchlist={watchlist}
              onChangeStep={(step): void =>
                setActiveGame((prev) => (prev ? { ...prev, step } : null))
              }
              onPinPosition={handlePinPosition}
              onBackToGames={(): void => setTab("games")}
            />
          ) : null}

          {tab === "watchlist" ? (
            <WatchlistView
              items={watchlist}
              loading={watchlistLoading}
              onOpenTabletop={openTabletop}
              onRemove={handleRemoveWatchlist}
            />
          ) : null}
        </main>
      </div>
    </div>
  );
}

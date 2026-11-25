/** Top-level Jace Studio shell: 224 px Linear navigation rail, theme toggle, and full-lifecycle routing. */

import { Activity, Bookmark, Layers, Menu, Moon, Sparkles, Sun, Swords, X } from "lucide-react";
import { type ReactNode, useCallback, useState } from "react";
import { GamesExplorerView } from "@/components/GamesExplorerView";
import { RunsView } from "@/components/RunsView";
import { TabletopView } from "@/components/TabletopView";
import { WatchlistView } from "@/components/WatchlistView";
import { Badge } from "@/components/ui/Primitives";
import {
  useKeyboardShortcut,
  useRunsData,
  useThemeSync,
  useWatchlistData,
} from "@/hooks/useStudioData";
import { addWatchlistItem, removeWatchlistItem } from "@/lib/api";
import type { BoardState, GameTarget, NavTab, ThemeMode } from "@/lib/types";

const NAV_ITEMS: ReadonlyArray<{ id: NavTab; label: string; shortcut: string }> = [
  { id: "runs", label: "Runs", shortcut: "1" },
  { id: "games", label: "Games", shortcut: "2" },
  { id: "tabletop", label: "Tabletop", shortcut: "3" },
  { id: "watchlist", label: "Watchlist", shortcut: "4" },
];

function NavIcon(props: { id: NavTab }): ReactNode {
  if (props.id === "runs") return <Activity className="w-4 h-4" />;
  if (props.id === "games") return <Layers className="w-4 h-4" />;
  if (props.id === "tabletop") return <Swords className="w-4 h-4" />;
  return <Bookmark className="w-4 h-4" />;
}

export function App(): ReactNode {
  const [tab, setTab] = useState<NavTab>("runs");
  const [theme, setTheme] = useState<ThemeMode>("light");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [gamesRunFilter, setGamesRunFilter] = useState("");
  const [activeGame, setActiveGame] = useState<GameTarget | null>(null);

  useThemeSync(theme);
  const { runs, details, loading: runsLoading, error: runsError, reload } = useRunsData();
  const {
    items: watchlist,
    setItems: setWatchlist,
    loading: watchlistLoading,
  } = useWatchlistData();

  const handleShortcut = useCallback((event: KeyboardEvent): void => {
    if (event.key === "1") setTab("runs");
    else if (event.key === "2") setTab("games");
    else if (event.key === "3") setTab("tabletop");
    else if (event.key === "4") setTab("watchlist");
  }, []);
  useKeyboardShortcut(handleShortcut);

  const openTabletop = (target: GameTarget): void => {
    setActiveGame(target);
    setTab("tabletop");
    setMobileMenuOpen(false);
  };

  const handlePinPosition = async (board: BoardState, target: GameTarget): Promise<void> => {
    const created = await addWatchlistItem({
      run_id: target.run_id,
      iteration: target.iteration,
      game_idx: target.game_idx,
      step: board.step,
      turn: board.turn,
      seed: target.seed,
      label: `Turn ${board.turn} · ${board.action_space_kind}`,
      note: `Hero ${board.hero.life} vs Opp ${board.opp.life} (${board.hero.zone_counts.battlefield + board.opp.zone_counts.battlefield} perms)`,
    });
    setWatchlist((prev) => [created, ...prev]);
  };

  const handleRemoveWatchlist = (itemId: string): void => {
    void removeWatchlistItem(itemId).then(() => {
      setWatchlist((prev) => prev.filter((item) => item.id !== itemId));
    });
  };

  const runningCount = runs.filter((r) => r.status === "running").length;

  const sidebarContent = (
    <div className="flex flex-col h-full justify-between p-3">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between px-1">
          <div className="flex items-center gap-2 font-semibold text-[13.5px] tracking-tight">
            <span className="w-2.5 h-2.5 rounded-[3px] bg-[var(--accent)]" />
            <span>Jace Studio</span>
          </div>
          {runningCount > 0 ? <Badge tone="win">{runningCount} live</Badge> : null}
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
                onClick={(): void => {
                  setTab(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`flex items-center justify-between rounded-md px-2.5 py-1.5 text-xs font-medium cursor-pointer transition-colors ${
                  active
                    ? "bg-[var(--accent-soft)] text-[var(--accent)] font-semibold"
                    : "text-[var(--text-2)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
                }`}
              >
                <span className="flex items-center gap-2">
                  <NavIcon id={item.id} />
                  {item.label}
                </span>
                <span className="tabular text-[10px] text-[var(--text-3)]">
                  {item.id === "watchlist" && watchlist.length > 0
                    ? watchlist.length
                    : item.shortcut}
                </span>
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
      {/* Desktop 224 px fixed left rail */}
      <aside className="hidden md:flex w-[224px] shrink-0 flex-col border-r border-[var(--border)] bg-[var(--surface)]">
        {sidebarContent}
      </aside>

      {/* Mobile slide-over nav */}
      {mobileMenuOpen ? (
        <div className="fixed inset-0 z-50 flex md:hidden">
          <div className="w-[240px] bg-[var(--surface)] border-r border-[var(--border)] shadow-xl relative">
            <button
              type="button"
              onClick={(): void => setMobileMenuOpen(false)}
              aria-label="Close navigation"
              className="absolute top-2.5 right-2.5 rounded p-1 text-[var(--text-2)]"
            >
              <X className="w-4 h-4" />
            </button>
            {sidebarContent}
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
        <header className="flex md:hidden items-center justify-between px-3 py-2 border-b border-[var(--border)] bg-[var(--surface)]">
          <button
            type="button"
            onClick={(): void => setMobileMenuOpen(true)}
            aria-label="Open navigation"
            className="rounded p-1 border border-[var(--border)]"
          >
            <Menu className="w-4 h-4" />
          </button>
          <span className="text-xs font-semibold">Jace Studio</span>
          <Badge tone="accent">{tab}</Badge>
        </header>

        {runsError ? (
          <div className="mx-4 mt-3 rounded-md border border-[var(--loss)] bg-[var(--loss-soft)] p-2.5 text-xs text-[var(--loss)]">
            Backend connection error: {runsError}
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
                setGamesRunFilter(runId);
                setTab("games");
              }}
            />
          ) : null}

          {tab === "games" ? (
            <GamesExplorerView
              runs={runs}
              initialRunId={gamesRunFilter}
              onOpenTabletop={openTabletop}
            />
          ) : null}

          {tab === "tabletop" ? (
            <TabletopView
              target={activeGame}
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

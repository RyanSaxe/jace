/** Top-level Jace Studio shell: adaptive Linear rail, URL deep links, and full-workbench routing. */

import {
  Activity,
  Bookmark,
  GitCompareArrows,
  Layers,
  Menu,
  Moon,
  Sun,
  Swords,
  X,
} from "lucide-react";
import { type ReactNode, useCallback, useState } from "react";
import { CompareView } from "@/components/CompareView";
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
  useUrlHashSync,
  useWatchlistData,
} from "@/hooks/useStudioData";
import { addWatchlistItem, removeWatchlistItem, updateWatchlistItem } from "@/lib/api";
import type {
  BoardState,
  CompareTarget,
  GameTarget,
  GamesFilterState,
  NavTab,
  ThemeMode,
} from "@/lib/types";

interface NavItemSpec {
  id: NavTab;
  label: string;
  shortcut: string;
}

const NAV_ITEMS: ReadonlyArray<NavItemSpec> = [
  { id: "runs", label: "Runs", shortcut: "1" },
  { id: "games", label: "Games", shortcut: "2" },
  { id: "compare", label: "Compare", shortcut: "3" },
  { id: "tabletop", label: "Tabletop", shortcut: "4" },
  { id: "watchlist", label: "Watchlist", shortcut: "5" },
];

const SHORTCUT_TO_TAB: Record<string, NavTab> = {
  "1": "runs",
  "2": "games",
  "3": "compare",
  "4": "tabletop",
  "5": "watchlist",
};

const INITIAL_GAMES_FILTERS: GamesFilterState = {
  runId: "",
  setCode: "",
  result: "",
  colors: "",
  card: "",
  minSurprise: null,
  minSwing: null,
  search: "",
  sort: "surprise_desc",
  page: 0,
};

interface ParsedUrlState {
  tab: NavTab;
  deepDiveRunId: string | null;
  compareTarget: CompareTarget | null;
  gameTarget: GameTarget | null;
}

const DEFAULT_URL_STATE: ParsedUrlState = {
  tab: "runs",
  deepDiveRunId: null,
  compareTarget: null,
  gameTarget: null,
};

function NavIcon(props: { id: NavTab }): ReactNode {
  if (props.id === "runs") return <Activity className="w-4 h-4 shrink-0" />;
  if (props.id === "games") return <Layers className="w-4 h-4 shrink-0" />;
  if (props.id === "compare") return <GitCompareArrows className="w-4 h-4 shrink-0" />;
  if (props.id === "tabletop") return <Swords className="w-4 h-4 shrink-0" />;
  return <Bookmark className="w-4 h-4 shrink-0" />;
}

function parseOptionalInt(raw: string | null, fallback: number): number {
  const parsed = Number.parseInt(raw ?? "", 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseCompareParams(params: URLSearchParams): CompareTarget | null {
  const ckptA = params.get("a") ?? "";
  const ckptB = params.get("b") ?? "";
  if (!ckptA || !ckptB) return null;
  return { ckptA, ckptB, runId: params.get("run") ?? undefined };
}

function parseTabletopParams(params: URLSearchParams): GameTarget | null {
  const runId = params.get("run") ?? "";
  const iteration = Number.parseInt(params.get("iter") ?? "", 10);
  const gameIdx = Number.parseInt(params.get("game") ?? "", 10);
  if (!runId || !Number.isFinite(iteration) || !Number.isFinite(gameIdx)) return null;
  return {
    run_id: runId,
    iteration,
    game_idx: gameIdx,
    step: parseOptionalInt(params.get("step"), 0),
    seed: parseOptionalInt(params.get("seed"), 0),
    ckpt_a: params.get("a") ?? undefined,
    ckpt_b: params.get("b") ?? undefined,
  };
}

function parseHashState(): ParsedUrlState {
  if (typeof window === "undefined") return DEFAULT_URL_STATE;
  const raw = window.location.hash.replace(/^#\/?/, "");
  const [pathPart = "runs", queryPart = ""] = raw.split("?");
  const params = new URLSearchParams(queryPart);
  if (pathPart.startsWith("runs/")) {
    const runId = decodeURIComponent(pathPart.slice("runs/".length));
    return { ...DEFAULT_URL_STATE, tab: "runs", deepDiveRunId: runId || null };
  }
  if (pathPart === "games" || pathPart === "watchlist") {
    return { ...DEFAULT_URL_STATE, tab: pathPart };
  }
  if (pathPart === "compare") {
    return { ...DEFAULT_URL_STATE, tab: "compare", compareTarget: parseCompareParams(params) };
  }
  if (pathPart === "tabletop") {
    return { ...DEFAULT_URL_STATE, tab: "tabletop", gameTarget: parseTabletopParams(params) };
  }
  return DEFAULT_URL_STATE;
}

function buildHashUrl(state: ParsedUrlState): string {
  if (state.tab === "runs") {
    return state.deepDiveRunId ? `#runs/${encodeURIComponent(state.deepDiveRunId)}` : "#runs";
  }
  if (state.tab === "compare" && state.compareTarget) {
    const p = new URLSearchParams({ a: state.compareTarget.ckptA, b: state.compareTarget.ckptB });
    if (state.compareTarget.runId) p.set("run", state.compareTarget.runId);
    return `#compare?${p.toString()}`;
  }
  if (state.tab === "tabletop" && state.gameTarget) {
    const p = new URLSearchParams({
      run: state.gameTarget.run_id,
      iter: String(state.gameTarget.iteration),
      game: String(state.gameTarget.game_idx),
      step: String(state.gameTarget.step),
      seed: String(state.gameTarget.seed),
    });
    if (state.gameTarget.ckpt_a) p.set("a", state.gameTarget.ckpt_a);
    if (state.gameTarget.ckpt_b) p.set("b", state.gameTarget.ckpt_b);
    return `#tabletop?${p.toString()}`;
  }
  return `#${state.tab}`;
}

function SidebarNavButton(props: {
  item: NavItemSpec;
  active: boolean;
  expanded: boolean;
  watchlistCount: number;
  onSelect: (id: NavTab) => void;
}): ReactNode {
  const { item, active, expanded, watchlistCount, onSelect } = props;
  const count = item.id === "watchlist" ? watchlistCount : 0;
  const toneCls = active
    ? "bg-[var(--accent-soft)] text-[var(--accent)] font-semibold"
    : "text-[var(--text-2)] hover:bg-[var(--surface-2)] hover:text-[var(--text)]";
  if (!expanded) {
    return (
      <button
        type="button"
        title={`${item.label} (${item.shortcut})`}
        aria-label={item.label}
        aria-current={active ? "page" : undefined}
        onClick={(): void => onSelect(item.id)}
        className={`relative flex items-center justify-center w-9 h-9 mx-auto rounded-md text-xs font-medium cursor-pointer transition-colors ${toneCls}`}
      >
        <NavIcon id={item.id} />
        {count > 0 ? (
          <span className="absolute -top-0.5 -right-0.5 min-w-[15px] h-[15px] px-1 rounded-full bg-[var(--accent)] text-white text-[9px] font-semibold flex items-center justify-center tabular">
            {count}
          </span>
        ) : null}
      </button>
    );
  }
  return (
    <button
      type="button"
      aria-current={active ? "page" : undefined}
      onClick={(): void => onSelect(item.id)}
      className={`relative flex items-center justify-between px-2.5 py-2 lg:py-1.5 rounded-md text-xs font-medium cursor-pointer transition-colors ${toneCls}`}
    >
      <span className="flex items-center gap-2">
        <NavIcon id={item.id} />
        {item.label}
        {count > 0 ? <Badge tone="accent">{count}</Badge> : null}
      </span>
      <kbd className="tabular text-[10px] text-[var(--text-3)] font-sans">{item.shortcut}</kbd>
    </button>
  );
}

function SidebarThemeButton(props: {
  theme: ThemeMode;
  expanded: boolean;
  onToggleTheme: () => void;
}): ReactNode {
  const { theme, expanded, onToggleTheme } = props;
  const icon =
    theme === "light" ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />;
  const label = theme === "light" ? "Warm Paper" : "Obsidian Dark";
  const title = theme === "light" ? "Switch to Obsidian Dark" : "Switch to Warm Paper";
  if (!expanded) {
    return (
      <button
        type="button"
        title={title}
        aria-label="Toggle color theme"
        onClick={onToggleTheme}
        className="flex items-center justify-center w-9 h-9 rounded-md border border-[var(--border)] bg-[var(--bg)] text-xs font-medium hover:border-[var(--border-strong)] cursor-pointer"
      >
        {icon}
      </button>
    );
  }
  return (
    <button
      type="button"
      title={title}
      onClick={onToggleTheme}
      className="flex items-center justify-between px-2.5 py-1.5 rounded-md border border-[var(--border)] bg-[var(--bg)] text-xs font-medium hover:border-[var(--border-strong)] cursor-pointer"
    >
      <span className="flex items-center gap-2">
        {icon}
        {label}
      </span>
      <span className="text-[10px] text-[var(--text-3)]">Theme</span>
    </button>
  );
}

function SidebarHeader(props: {
  expanded: boolean;
  isDrawer: boolean;
  runningCount: number;
  onCloseDrawer: () => void;
}): ReactNode {
  const { expanded, isDrawer, runningCount, onCloseDrawer } = props;
  if (!expanded) {
    return (
      <div className="flex items-center justify-center">
        <span className="w-3 h-3 rounded-[4px] bg-[var(--accent)] shrink-0" title="Jace Studio" />
      </div>
    );
  }
  return (
    <div className="flex items-center justify-between gap-2 px-1">
      <h1 className="flex items-center gap-2 font-semibold text-[13.5px] tracking-tight m-0">
        <span className="w-3 h-3 rounded-[4px] bg-[var(--accent)] shrink-0" />
        <span>Jace Studio</span>
      </h1>
      <div className="flex items-center gap-1.5">
        {runningCount > 0 ? <Badge tone="win">{runningCount} live</Badge> : null}
        {isDrawer ? (
          <button
            type="button"
            onClick={onCloseDrawer}
            aria-label="Close navigation"
            className="w-9 h-9 rounded-md flex items-center justify-center text-[var(--text-2)] hover:bg-[var(--surface-2)] cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        ) : null}
      </div>
    </div>
  );
}

function SidebarContent(props: {
  tab: NavTab;
  theme: ThemeMode;
  expanded: boolean;
  isDrawer: boolean;
  runningCount: number;
  watchlistCount: number;
  onSelectTab: (id: NavTab) => void;
  onCloseDrawer: () => void;
  onToggleTheme: () => void;
}): ReactNode {
  const {
    tab,
    theme,
    expanded,
    isDrawer,
    runningCount,
    watchlistCount,
    onSelectTab,
    onCloseDrawer,
    onToggleTheme,
  } = props;
  return (
    <div
      className={`flex flex-col h-full justify-between ${expanded ? "p-3" : "p-2 items-center"}`}
    >
      <div className={`flex flex-col gap-4 ${expanded ? "" : "items-center w-full"}`}>
        <SidebarHeader
          expanded={expanded}
          isDrawer={isDrawer}
          runningCount={runningCount}
          onCloseDrawer={onCloseDrawer}
        />
        <nav className="flex flex-col gap-0.5 w-full" aria-label="Primary navigation">
          {expanded ? (
            <div className="px-2 pb-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-3)]">
              Workbench
            </div>
          ) : null}
          {NAV_ITEMS.map((item) => (
            <SidebarNavButton
              key={item.id}
              item={item}
              active={tab === item.id}
              expanded={expanded}
              watchlistCount={watchlistCount}
              onSelect={onSelectTab}
            />
          ))}
        </nav>
      </div>
      <div
        className={`flex flex-col gap-2 pt-3 border-t border-[var(--border)] ${expanded ? "" : "w-full items-center"}`}
      >
        <SidebarThemeButton theme={theme} expanded={expanded} onToggleTheme={onToggleTheme} />
      </div>
    </div>
  );
}

export function App(): ReactNode {
  const initialUrl = parseHashState();
  const [tab, setTab] = useState<NavTab>(initialUrl.tab);
  const [deepDiveRunId, setDeepDiveRunId] = useState<string | null>(initialUrl.deepDiveRunId);
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
    if (parsed.compareTarget) setCompareTarget(parsed.compareTarget);
    if (parsed.gameTarget) setActiveGame(parsed.gameTarget);
  }, []);

  useUrlHashSync(
    buildHashUrl({ tab, deepDiveRunId, compareTarget, gameTarget: activeGame }),
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

  const runningCount = runs.filter((r) => r.status === "running").length;
  const desktopExpanded = tab !== "tabletop";
  const bannerError = runsError ? `Backend connection error: ${runsError}` : actionError;

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

        <main className="flex-1 min-h-0 overflow-hidden">
          {tab === "runs" ? (
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
              onComparePair={(target): void => {
                setCompareTarget(target);
                setTab("compare");
                setMobileMenuOpen(false);
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

          {tab === "compare" ? (
            <CompareView
              runs={runs}
              checkpoints={checkpoints}
              initialTarget={compareTarget}
              onOpenTabletop={openTabletop}
            />
          ) : null}

          {tab === "tabletop" ? (
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
          ) : null}

          {tab === "watchlist" ? (
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
            />
          ) : null}
        </main>
      </div>
    </div>
  );
}

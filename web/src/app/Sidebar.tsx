/** Adaptive sidebar rail and mobile navigation drawer content for Jace Studio. */

import {
  Activity,
  Bookmark,
  GitCompareArrows,
  Layers,
  Moon,
  Sun,
  Swords,
  Trophy,
  X,
} from "lucide-react";
import type { ReactNode } from "react";
import { Badge } from "@/components/ui/Primitives";
import type { NavTab, ThemeMode } from "@/lib/types";
import { NAV_ITEMS, type NavItemSpec } from "./routes";

function NavIcon(props: { id: NavTab }): ReactNode {
  if (props.id === "runs") return <Activity className="w-4 h-4 shrink-0" />;
  if (props.id === "games") return <Layers className="w-4 h-4 shrink-0" />;
  if (props.id === "benchmarks") return <Trophy className="w-4 h-4 shrink-0" />;
  if (props.id === "compare") return <GitCompareArrows className="w-4 h-4 shrink-0" />;
  if (props.id === "tabletop") return <Swords className="w-4 h-4 shrink-0" />;
  return <Bookmark className="w-4 h-4 shrink-0" />;
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

export function SidebarContent(props: {
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

/** Shared Tabletop UI state, controller, overlay, hover card, and policy matrix interfaces. */

import type { ForkEdit, PolicyMatrixResponse, PovMode } from "@/lib/types";
import type { resolveOverviewInfo } from "./tabletopHelpers";

export interface HoverCardInfo {
  name: string;
  pt?: string;
  damage?: number;
  counters?: number;
  tapped?: boolean;
  token?: boolean;
  sick?: boolean;
  attacking?: boolean;
  blocking?: boolean;
  blockingName?: string;
  copyIndex?: number | null;
  count?: number;
  anchorX?: number;
  anchorY?: number;
}

export interface DrawerTarget {
  side: "hero" | "opp";
  zone: "library" | "graveyard" | "exile";
}

export interface HandCardEditTarget {
  side: "hero" | "opp";
  cardName: string;
  key: string;
  anchorX: number;
  anchorY: number;
}

export type TabletopScale = "desktop" | "mobile-tall" | "mobile-short";

export type ActiveOverlay = "none" | "details" | "addCard" | "pin" | "mobilePolicy";

export type OverviewInfo = ReturnType<typeof resolveOverviewInfo>;

export interface PlaymatController {
  pov: PovMode;
  editMode: boolean;
  overlay: ActiveOverlay;
  stackOpen: boolean;
  handEditTarget: HandCardEditTarget | null;
  drawer: DrawerTarget | null;
  hoverInfo: HoverCardInfo | null;
  pinnedCard: HoverCardInfo | null;
  highlightIds: ReadonlySet<number>;
  highlightTone: "target" | "source";
  setOverlay: (next: ActiveOverlay) => void;
  toggleStack: () => void;
  setHandEditTarget: (target: HandCardEditTarget | null) => void;
  addForkEdit: (edit: ForkEdit) => void;
  removeForkEdit: (idx: number) => void;
  resetFork: () => void;
  setDrawer: (drawer: DrawerTarget | null) => void;
  setHover: (info: HoverCardInfo | null) => void;
  setPinnedCard: (info: HoverCardInfo | null) => void;
  setHoverSource: (permId: number | null) => void;
  adjustLife: (seat: 0 | 1, delta: number) => void;
  selectStep: (step: number) => void;
}

export interface MatrixUiState {
  open: boolean;
  models: string[];
  data: PolicyMatrixResponse | null;
  loading: boolean;
  error: string | null;
}

export const INITIAL_MATRIX: MatrixUiState = {
  open: false,
  models: [],
  data: null,
  loading: false,
  error: null,
};

/** Behavioral tests for responsive mobile Tabletop card sizing, safe-centering scroll, inline life badges, stack popover, and toolbar controls. */

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  LandPilesRow,
  PermanentsRow,
  type PlaymatController,
  TabletopPlaymat,
  TabletopTopBar,
  TabletopView,
} from "@/components/tabletop";
import type { BoardState, GameTarget, LandPile, PermanentState, PlayerState } from "@/lib/types";

function setViewport(width: number, height: number): void {
  Object.defineProperty(window, "innerWidth", { configurable: true, writable: true, value: width });
  Object.defineProperty(window, "innerHeight", {
    configurable: true,
    writable: true,
    value: height,
  });
  act(() => {
    window.dispatchEvent(new Event("resize"));
  });
}

function makeCreature(id: number, name: string, tapped = false): PermanentState {
  return {
    id,
    name,
    is_land: false,
    is_creature: true,
    is_artifact: false,
    is_enchantment: false,
    is_token: false,
    is_legendary: false,
    power: 2,
    toughness: 2,
    damage: 0,
    plus1_counters: 0,
    tapped,
    sick: false,
    mana_value: 2,
  };
}

function makeLands(count: number): LandPile[] {
  return Array.from({ length: count }, (_, idx) => ({
    name: idx % 2 === 0 ? "Plains" : "Island",
    tapped: idx % 2 === 1,
    count: 1,
    id: idx + 1,
  }));
}

function makePlayer(overrides: Partial<PlayerState> = {}): PlayerState {
  return {
    seat: 0,
    life: 20,
    is_active: true,
    zone_counts: { library: 25, hand: 4, battlefield: 6, graveyard: 3, exile: 0 },
    lands: makeLands(3),
    creatures: [
      makeCreature(1, "Aang, the Last Airbender"),
      makeCreature(2, "Sokka, Lateral Strategist"),
    ],
    other_permanents: [],
    hand: ["Plains", "Island", "Water Whip", "Airbend"],
    hand_hidden: false,
    graveyard: ["Plains"],
    exile: [],
    library_summary: [],
    library_hidden: false,
    ...overrides,
  };
}

function makeBoard(overrides: Partial<BoardState> = {}): BoardState {
  return {
    step: 182,
    total_steps: 240,
    turn: 16,
    phase: "PrecombatMain",
    step_name: "PrecombatMainStep",
    action_space_kind: "PRIORITY",
    hero_seat: 0,
    acting_seat: 0,
    actor_is_hero: true,
    active_is_hero: true,
    chosen_action: 0,
    pov: "hero",
    hero: makePlayer(),
    opp: makePlayer({
      seat: 1,
      life: 14,
      is_active: false,
      zone_counts: { library: 23, hand: 6, battlefield: 6, graveyard: 4, exile: 0 },
      hand: [],
      hand_hidden: true,
      creatures: [
        makeCreature(10, "Treetop Freedom Fighters", true),
        makeCreature(11, "Sparring Dummy"),
        makeCreature(12, "Dai Li Agents"),
      ],
    }),
    stack: [
      {
        index: 1,
        card_name: "Avatar Enthusiasts",
        description: "Avatar Enthusiasts",
        kind: "spell",
        source_permanent_id: null,
        controller_id: 0,
        controller_side: "hero",
      },
    ],
    legal_actions: [{ index: 0, label: "Pass priority", action_type: "pass", focus_ids: [] }],
    ...overrides,
  };
}

function makeController(overrides: Partial<PlaymatController> = {}): PlaymatController {
  return {
    pov: "hero",
    editMode: false,
    overlay: "none",
    stackOpen: false,
    handEditTarget: null,
    drawer: null,
    hoverInfo: null,
    pinnedCard: null,
    highlightIds: new Set<number>(),
    highlightTone: "target",
    setOverlay: vi.fn(),
    toggleStack: vi.fn(),
    setHandEditTarget: vi.fn(),
    addForkEdit: vi.fn(),
    removeForkEdit: vi.fn(),
    resetFork: vi.fn(),
    setDrawer: vi.fn(),
    setHover: vi.fn(),
    setPinnedCard: vi.fn(),
    setHoverSource: vi.fn(),
    adjustLife: vi.fn(),
    selectStep: vi.fn(),
    ...overrides,
  };
}

const TARGET: GameTarget = {
  run_id: "runs/v2-r89",
  iteration: 200,
  game_idx: 0,
  step: 182,
  seed: 36145,
};

function installTabletopFetch(board: BoardState): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: string | URL) => {
      const url = String(input);
      let body: unknown = {};
      if (url.startsWith("/api/runs/")) {
        body = {
          summary: { run_id: TARGET.run_id },
          config: {},
          eval_points: [],
          train_points: [],
          checkpoints: [{ iteration: 200, name: "iter-0200", path: "/tmp/c", size_bytes: 10 }],
          eval_iterations: [200],
        };
      } else if (url.startsWith("/api/cards-catalog")) {
        body = { cards: ["Plains", "Island"] };
      } else if (url.startsWith("/api/game-detail")) {
        body = {
          run_id: TARGET.run_id,
          iteration: 200,
          game_idx: 0,
          seed: 36145,
          winner: 0,
          total_steps: 240,
          max_playable_step: 239,
          peak_step: 182,
          players: [],
          timeline: [],
        };
      } else if (url.startsWith("/api/game-step")) {
        body = {
          board,
          evaluation: {
            primary: {
              checkpoint: "iter-0200",
              probs: [1],
              wdl: { win: 0.6, draw: 0, loss: 0.4 },
            },
            compare: {
              checkpoint: "iter-0200",
              probs: [1],
              wdl: { win: 0.6, draw: 0, loss: 0.4 },
            },
            diff: { tv: 0, kl: 0, argmax_changed: false },
          },
          nearest_priority_step: 182,
        };
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: "OK",
        json: (): Promise<unknown> => Promise.resolve(body),
      });
    }),
  );
}

const OVERVIEW = { timeline: [], maxPlayableStep: 239, replayWarning: null };

function expectSafeCenteredScroll(container: HTMLElement): void {
  const scroll = container.firstElementChild;
  const inner = scroll?.firstElementChild;
  expect(scroll?.className).toContain("overflow-x-auto");
  expect(scroll?.className).not.toContain("justify-center");
  expect(inner?.className).toContain("w-max");
  expect(inner?.className).toContain("mx-auto");
}

describe("Mobile Tabletop responsiveness", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    setViewport(1440, 900);
  });

  it("scales creatures, lands, and hands compactly on short mobile without bottom-fold hand push", () => {
    setViewport(390, 664);
    const board = makeBoard();
    render(
      <TabletopPlaymat
        board={board}
        recordedStep={182}
        forkEdits={[]}
        cardCatalog={[]}
        nearestPriorityStep={182}
        overview={OVERVIEW}
        ctrl={makeController()}
      />,
    );

    const heroCreature = screen.getByRole("button", { name: "Inspect Aang, the Last Airbender" });
    expect(heroCreature).toHaveStyle({ width: "52px", height: "72px" });

    const heroHandCard = screen.getByRole("button", { name: "Inspect Water Whip" });
    expect(heroHandCard.className).toContain("w-[50px]");
    expect(heroHandCard.className).toContain("h-[69px]");
    expect(heroHandCard.className).not.toContain("translate-y-[32px]");

    const oppCardBacks = screen.getAllByTitle("Hidden in Hero POV");
    expect(oppCardBacks[0]?.className).toContain("w-[32px]");
    expect(oppCardBacks[0]?.className).toContain("h-[44px]");
    expect(oppCardBacks[0]?.className).not.toContain("-translate-y-[24px]");
  });

  it("preserves full desktop card dimensions and corner badges at 1440x900", () => {
    setViewport(1440, 900);
    const board = makeBoard();
    render(
      <TabletopPlaymat
        board={board}
        recordedStep={182}
        forkEdits={[]}
        cardCatalog={[]}
        nearestPriorityStep={182}
        overview={OVERVIEW}
        ctrl={makeController()}
      />,
    );

    const heroCreature = screen.getByRole("button", { name: "Inspect Aang, the Last Airbender" });
    expect(heroCreature).toHaveStyle({ width: "92px", height: "127px" });

    const heroHandCard = screen.getByRole("button", { name: "Inspect Water Whip" });
    expect(heroHandCard.className).toContain("translate-y-[32px]");

    const oppLabel = screen.getByText("Opponent");
    expect(oppLabel.closest(".absolute")).not.toBeNull();
  });

  it("renders Opponent and Hero life badges inline (non-absolute) on mobile with Edit Mode life controls", () => {
    setViewport(390, 664);
    const adjustLife = vi.fn();
    render(
      <TabletopPlaymat
        board={makeBoard()}
        recordedStep={182}
        forkEdits={[]}
        cardCatalog={[]}
        nearestPriorityStep={182}
        overview={OVERVIEW}
        ctrl={makeController({ editMode: true, adjustLife })}
      />,
    );

    const oppLabel = screen.getByText("Opponent");
    const heroLabel = screen.getByText("Hero");
    expect(oppLabel.closest(".absolute")).toBeNull();
    expect(heroLabel.closest(".absolute")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Decrease Opponent life" }));
    fireEvent.click(screen.getByRole("button", { name: "Increase Hero life" }));
    expect(adjustLife).toHaveBeenCalledWith(1, -1);
    expect(adjustLife).toHaveBeenCalledWith(0, 1);
  });

  it("uses safe-centering scroll containers on wide creature and land rows without justify-center clipping", () => {
    const swarm = [
      makeCreature(1, "Momo, Friendly Flier"),
      makeCreature(2, "Ally #1"),
      makeCreature(3, "Ally #2"),
      makeCreature(4, "Ally #3"),
      makeCreature(5, "Ally #4"),
      makeCreature(6, "Ally #5"),
      makeCreature(7, "Ally #6"),
    ];
    const { container: permContainer } = render(
      <PermanentsRow
        permanents={swarm}
        scale="mobile-short"
        highlightIds={new Set<number>()}
        highlightTone="target"
        onHover={vi.fn()}
      />,
    );
    expectSafeCenteredScroll(permContainer);
    const momo = screen.getByRole("button", { name: "Inspect Momo, Friendly Flier" });
    expect(momo).toHaveStyle({ width: "44px", height: "61px" });

    const { container: landContainer } = render(
      <LandPilesRow piles={makeLands(6)} scale="mobile-short" onHover={vi.fn()} />,
    );
    expectSafeCenteredScroll(landContainer);
  });

  it("keeps Stack collapsed by default on narrow viewports and anchors it below the divider when toggled", async () => {
    setViewport(390, 664);
    installTabletopFetch(makeBoard());
    render(
      <TabletopView
        target={TARGET}
        checkpoints={[]}
        watchlist={[]}
        onChangeStep={vi.fn()}
        onPinPosition={vi.fn().mockResolvedValue(undefined)}
        onBackToGames={vi.fn()}
      />,
    );

    const toggle = await waitFor(() => screen.getByRole("button", { name: /Stack \(1\)/i }));
    expect(screen.queryByRole("button", { name: "Collapse stack" })).toBeNull();

    fireEvent.click(toggle);
    const closeBtn = screen.getByRole("button", { name: "Collapse stack" });
    const stackBox = closeBtn.closest("div.absolute");
    expect(stackBox?.className).toContain("top-[52%]");
    expect(stackBox?.className).toContain("left-2");
    expect(stackBox?.className).toContain("right-2");
  });

  it("hides desktop-only Edit Board on mobile and enforces >=34px top-bar touch targets", () => {
    render(
      <TabletopTopBar
        target={TARGET}
        turn={16}
        pov="hero"
        editMode={false}
        editCount={0}
        hasMutations={false}
        pinned={false}
        pinOpen={false}
        onBack={vi.fn()}
        onChangePov={vi.fn()}
        onToggleEdit={vi.fn()}
        onResetFork={vi.fn()}
        onTogglePin={vi.fn()}
      />,
    );

    const editBtn = screen.getByRole("button", { name: /Edit Board/i });
    const editClasses = editBtn.className.split(/\s+/);
    expect(editClasses).toContain("hidden");
    expect(editClasses).toContain("lg:inline-flex");
    expect(editClasses).not.toContain("inline-flex");

    const backBtn = screen.getByRole("button", { name: "Back to games" });
    const pinBtn = screen.getByRole("button", { name: "Pin position" });
    expect(backBtn.className).toContain("min-h-[34px]");
    expect(backBtn.className).toContain("min-w-[34px]");
    expect(pinBtn.className).toContain("min-h-[34px]");
    expect(pinBtn.className).toContain("min-w-[34px]");
  });
});

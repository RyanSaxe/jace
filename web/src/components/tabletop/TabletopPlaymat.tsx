/** Responsive Tabletop playmat: player life badges, opponent/hero battlefield halves, and overlay container. */

import { type ReactNode, useSyncExternalStore } from "react";
import type { BoardState, ForkEdit } from "@/lib/types";
import { HandFanView, LandPilesRow, PermanentsRow } from "./TabletopBoardZones";
import { CenterTurnDivider, EditModeBanner, TurnTimelineBar } from "./TabletopChrome";
import { AddCardPopover, HandCardEditPopover } from "./TabletopEditPopovers";
import { BoardDetailsPopover, CardZoomPreview } from "./TabletopPopovers";
import { StackOverlay, ZoneDrawerSheet, ZoneFlankView } from "./TabletopZoneDrawers";
import type { OverviewInfo, PlaymatController, TabletopScale } from "./tabletopTypes";

function badgeTone(
  controls: boolean,
  isActive: boolean,
  isActing: boolean,
): { border: string; dot: string } {
  let border = "border-[var(--border)]";
  if (controls) border = "border-[var(--fork)]";
  else if (isActive) border = "border-[var(--accent)]";
  let dot = "bg-[var(--border-strong)]";
  if (isActing) dot = "bg-[var(--win-fill)]";
  else if (isActive) dot = "bg-[var(--accent)]";
  return { border, dot };
}

function subscribeResize(onStoreChange: () => void): () => void {
  if (typeof window === "undefined") return (): void => undefined;
  window.addEventListener("resize", onStoreChange);
  return (): void => window.removeEventListener("resize", onStoreChange);
}

function readTabletopScale(): TabletopScale {
  if (typeof window === "undefined" || window.innerWidth >= 1024) return "desktop";
  return window.innerHeight <= 740 ? "mobile-short" : "mobile-tall";
}

function PlayerCornerBadge(props: {
  label: string;
  player: BoardState["hero"];
  isActing: boolean;
  editMode: boolean;
  canEdit: boolean;
  compact?: boolean;
  onAdjust: (d: number) => void;
}): ReactNode {
  const { player } = props;
  const controls = props.editMode && props.canEdit;
  const tone = badgeTone(controls, Boolean(player.is_active), props.isActing);
  const stepCls = props.compact
    ? "w-5 h-5 inline-flex items-center justify-center rounded-full bg-[var(--fork-soft)] text-[var(--fork)] cursor-pointer disabled:opacity-35"
    : "w-4 h-4 inline-flex items-center justify-center rounded-full bg-[var(--fork-soft)] text-[var(--fork)] cursor-pointer disabled:opacity-35";
  const padCls = props.compact
    ? "gap-1 px-1.5 py-0.5 text-[10.5px]"
    : "gap-1.5 px-2 py-0.5 text-[11px]";
  return (
    <div
      className={`inline-flex items-center rounded-full border bg-[var(--surface)]/95 shadow-[var(--shadow-sm)] ${padCls} ${tone.border}`}
    >
      <span className={`w-2 h-2 rounded-full shrink-0 ${tone.dot}`} />
      <span className="font-semibold">{props.label}</span>
      <span
        className={`tabular inline-flex items-center gap-1 font-bold ${player.life <= 5 ? "text-[var(--loss)]" : ""}`}
      >
        {controls ? (
          <button
            type="button"
            disabled={player.life <= 1}
            onClick={(): void => props.onAdjust(-1)}
            aria-label={`Decrease ${props.label} life`}
            className={stepCls}
          >
            −
          </button>
        ) : null}
        <span>{player.life}</span>
        {controls ? (
          <button
            type="button"
            onClick={(): void => props.onAdjust(1)}
            aria-label={`Increase ${props.label} life`}
            className={stepCls}
          >
            +
          </button>
        ) : null}
      </span>
      {player.mana_pool ? (
        <span className="tabular rounded bg-[var(--warn-soft)] text-[var(--warn)] px-1 text-[10px] font-bold">
          {player.mana_pool}
        </span>
      ) : null}
    </div>
  );
}

function BattlefieldHalf(props: {
  side: "hero" | "opp";
  player: BoardState["hero"];
  isActing: boolean;
  canEdit: boolean;
  scale: TabletopScale;
  permNames: ReadonlyMap<number, string>;
  ctrl: PlaymatController;
  onAdjustLife: (delta: number) => void;
}): ReactNode {
  const { player, ctrl, scale } = props;
  const isOpp = props.side === "opp";
  const hand = (
    <HandFanView
      side={props.side}
      cards={player.hand}
      hiddenCount={player.zone_counts.hand}
      hidden={isOpp ? player.hand_hidden : false}
      scale={scale}
      editMode={ctrl.editMode}
      selectedEditKey={ctrl.handEditTarget?.key ?? null}
      onHover={ctrl.setHover}
      onPinCard={ctrl.setPinnedCard}
      onSelectCardForEdit={ctrl.setHandEditTarget}
    />
  );
  const handLane =
    scale === "desktop" ? (
      hand
    ) : (
      <div className="flex items-center gap-1.5 min-w-0 shrink-0">
        <div className="shrink-0">
          <PlayerCornerBadge
            label={isOpp ? "Opponent" : "Hero"}
            player={player}
            isActing={props.isActing}
            editMode={ctrl.editMode}
            canEdit={props.canEdit}
            compact
            onAdjust={props.onAdjustLife}
          />
        </div>
        <div className="flex-1 min-w-0">{hand}</div>
      </div>
    );
  const hasOther = player.other_permanents.length > 0;
  const lands = hasOther ? (
    <div className="w-full min-w-0 overflow-x-auto no-scrollbar py-1 min-h-[44px] lg:min-h-[52px] shrink-0">
      <div className="w-max mx-auto flex items-center gap-2.5 lg:gap-3 flex-nowrap px-1.5">
        <LandPilesRow piles={player.lands} compact scale={scale} onHover={ctrl.setHover} />
        <div className="pl-2.5 lg:pl-3 shrink-0 border-l border-dashed border-[var(--border-strong)]">
          <PermanentsRow
            permanents={player.other_permanents}
            compact
            scale={scale}
            permNames={props.permNames}
            highlightIds={ctrl.highlightIds}
            highlightTone={ctrl.highlightTone}
            onHover={ctrl.setHover}
            onPinCard={ctrl.setPinnedCard}
          />
        </div>
      </div>
    </div>
  ) : (
    <div className="w-full min-w-0 min-h-[44px] lg:min-h-[52px] flex items-center shrink-0">
      <LandPilesRow piles={player.lands} scale={scale} onHover={ctrl.setHover} />
    </div>
  );
  const creatures = (
    <PermanentsRow
      permanents={player.creatures}
      scale={scale}
      permNames={props.permNames}
      highlightIds={ctrl.highlightIds}
      highlightTone={ctrl.highlightTone}
      onHover={ctrl.setHover}
      onPinCard={ctrl.setPinnedCard}
    />
  );
  return (
    <div className="h-full min-h-0 grid grid-cols-[minmax(0,1fr)_64px] sm:grid-cols-[minmax(0,1fr)_70px] gap-1.5 sm:gap-2">
      <div className="flex flex-col justify-between min-w-0 min-h-0 h-full gap-0.5 lg:gap-1">
        {isOpp ? handLane : null}
        <div className="flex-1 min-h-0 flex flex-col justify-center gap-0.5 lg:gap-1">
          {isOpp ? (
            <>
              {lands}
              {creatures}
            </>
          ) : (
            <>
              {creatures}
              {lands}
            </>
          )}
        </div>
        {isOpp ? null : handLane}
      </div>
      <ZoneFlankView side={props.side} player={player} onOpenDrawer={ctrl.setDrawer} />
    </div>
  );
}

function PlaymatOverlays(props: {
  board: BoardState;
  heroSeat: 0 | 1;
  oppSeat: 0 | 1;
  cardCatalog: string[];
  nearestPriorityStep: number | null;
  ctrl: PlaymatController;
}): ReactNode {
  const { board: b, ctrl } = props;
  const canEdit = b.action_space_kind === "PRIORITY";
  const nonPerm = new Set(b.non_permanent_names ?? []);
  const handTarget = ctrl.editMode ? ctrl.handEditTarget : null;
  return (
    <>
      <StackOverlay
        stack={b.stack}
        open={ctrl.stackOpen && ctrl.drawer === null}
        onClose={ctrl.toggleStack}
        onHoverSource={ctrl.setHoverSource}
        onHoverCard={ctrl.setHover}
      />
      {ctrl.overlay === "details" ? (
        <BoardDetailsPopover board={b} onClose={(): void => ctrl.setOverlay("none")} />
      ) : null}
      {ctrl.editMode && ctrl.overlay === "addCard" ? (
        <AddCardPopover
          heroSeat={props.heroSeat}
          canEditBoard={canEdit}
          nearestPriorityStep={props.nearestPriorityStep}
          nonPermanentNames={nonPerm}
          cardCatalog={props.cardCatalog}
          onAddEdit={ctrl.addForkEdit}
          onJumpToStep={ctrl.selectStep}
          onClose={(): void => ctrl.setOverlay("none")}
        />
      ) : null}
      {handTarget ? (
        <HandCardEditPopover
          target={handTarget}
          seat={handTarget.side === "hero" ? props.heroSeat : props.oppSeat}
          canEditBoard={canEdit}
          nearestPriorityStep={props.nearestPriorityStep}
          cardCatalog={props.cardCatalog}
          onEdit={ctrl.addForkEdit}
          onJumpToStep={ctrl.selectStep}
          onClose={(): void => ctrl.setHandEditTarget(null)}
        />
      ) : null}
      <ZoneDrawerSheet
        target={ctrl.drawer}
        hero={b.hero}
        opp={b.opp}
        pov={ctrl.pov}
        onClose={(): void => {
          ctrl.setHover(null);
          ctrl.setDrawer(null);
        }}
        onHover={ctrl.setHover}
      />
      <CardZoomPreview
        info={ctrl.pinnedCard ?? ctrl.hoverInfo}
        pinned={ctrl.pinnedCard !== null}
        onDismiss={(): void => ctrl.setPinnedCard(null)}
      />
    </>
  );
}

export function TabletopPlaymat(props: {
  board: BoardState;
  recordedStep: number;
  forkEdits: ForkEdit[];
  cardCatalog: string[];
  nearestPriorityStep: number | null;
  overview: OverviewInfo;
  ctrl: PlaymatController;
}): ReactNode {
  const { board: b, ctrl } = props;
  const scale = useSyncExternalStore(
    subscribeResize,
    readTabletopScale,
    (): TabletopScale => "desktop",
  );
  const heroSeat: 0 | 1 = b.hero_seat === 1 ? 1 : 0;
  const oppSeat: 0 | 1 = heroSeat === 1 ? 0 : 1;
  const canEdit = b.action_space_kind === "PRIORITY";
  const permNames = new Map<number, string>([
    ...b.hero.creatures.map((p): [number, string] => [p.id, p.name]),
    ...b.opp.creatures.map((p): [number, string] => [p.id, p.name]),
  ]);
  const stackOpen = ctrl.stackOpen && ctrl.drawer === null;
  const oppActing = b.actor_is_hero === false;
  const heroActing = b.actor_is_hero !== false;
  return (
    <div className="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden">
      {ctrl.editMode ? (
        <EditModeBanner
          heroSeat={heroSeat}
          canEdit={canEdit}
          nearestPriorityStep={props.nearestPriorityStep}
          edits={props.forkEdits}
          ctrl={ctrl}
        />
      ) : null}
      <div
        className={`flex-1 min-h-0 relative felt-surface px-2.5 sm:px-3 py-1 lg:py-1.5 grid grid-rows-[minmax(0,1fr)_auto_minmax(0,1fr)] gap-1 overflow-y-auto lg:overflow-hidden ${ctrl.editMode ? "ring-2 ring-inset ring-[var(--fork)]/45 border-2 border-dashed border-[var(--fork)]/45" : ""}`}
      >
        {scale === "desktop" ? (
          <div className="absolute top-2 left-2.5 sm:left-3 z-20">
            <PlayerCornerBadge
              label="Opponent"
              player={b.opp}
              isActing={oppActing}
              editMode={ctrl.editMode}
              canEdit={canEdit}
              onAdjust={(d): void => ctrl.adjustLife(oppSeat, d)}
            />
          </div>
        ) : null}
        <BattlefieldHalf
          side="opp"
          player={b.opp}
          isActing={oppActing}
          canEdit={canEdit}
          scale={scale}
          permNames={permNames}
          ctrl={ctrl}
          onAdjustLife={(d): void => ctrl.adjustLife(oppSeat, d)}
        />
        <CenterTurnDivider
          board={b}
          stackOpen={stackOpen}
          detailsOpen={ctrl.overlay === "details"}
          onToggleStack={ctrl.toggleStack}
          onToggleDetails={(): void =>
            ctrl.setOverlay(ctrl.overlay === "details" ? "none" : "details")
          }
        />
        <BattlefieldHalf
          side="hero"
          player={b.hero}
          isActing={heroActing}
          canEdit={canEdit}
          scale={scale}
          permNames={permNames}
          ctrl={ctrl}
          onAdjustLife={(d): void => ctrl.adjustLife(heroSeat, d)}
        />
        {scale === "desktop" ? (
          <div className="absolute bottom-2 left-2.5 sm:left-3 z-20">
            <PlayerCornerBadge
              label="Hero"
              player={b.hero}
              isActing={heroActing}
              editMode={ctrl.editMode}
              canEdit={canEdit}
              onAdjust={(d): void => ctrl.adjustLife(heroSeat, d)}
            />
          </div>
        ) : null}
        <PlaymatOverlays
          board={b}
          heroSeat={heroSeat}
          oppSeat={oppSeat}
          cardCatalog={props.cardCatalog}
          nearestPriorityStep={props.nearestPriorityStep}
          ctrl={ctrl}
        />
      </div>
      <TurnTimelineBar
        timeline={props.overview.timeline}
        evalSource={props.overview.evalSource}
        currentStep={props.recordedStep}
        totalSteps={b.total_steps}
        maxPlayableStep={props.overview.maxPlayableStep}
        turningPointStep={props.overview.turningPointStep}
        currentTurn={b.turn}
        onSelectStep={ctrl.selectStep}
      />
    </div>
  );
}

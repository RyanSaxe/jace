/** Shared formatting, axis ticks, and SVG label deconfliction utilities for Benchmarks charts. */

export type EloXAxis = "updates" | "search_ms";

export interface LabelCandidate {
  id: string;
  cx: number;
  cy: number;
  text: string;
  bold: boolean;
  fill: string;
}

export interface PlacedLabel {
  id: string;
  x: number;
  y: number;
  textAnchor: "start" | "end";
  text: string;
  bold: boolean;
  fill: string;
}

interface LabelBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const LABEL_OFFSETS: ReadonlyArray<{ dx: number; dy: number; anchor: "start" | "end" }> = [
  { dx: 7, dy: -8, anchor: "start" },
  { dx: -7, dy: -8, anchor: "end" },
  { dx: 7, dy: 13, anchor: "start" },
  { dx: -7, dy: 13, anchor: "end" },
];

export function formatElo(elo: number, ciHalf: number): string {
  const rounded = Math.round(elo).toLocaleString("en-US");
  const ci = Math.round(ciHalf);
  return ci > 0 ? `${rounded} ± ${ci}` : rounded;
}

export function formatPct(val: number): string {
  return `${(val * 100).toFixed(1)}%`;
}

export function shortAnchorLabel(label: string): string {
  if (label.toLowerCase() === "heuristic") return "Heur";
  return label.length > 10 ? `${label.slice(0, 9)}…` : label;
}

export function computeNiceTicks(minVal: number, maxVal: number, step: number): number[] {
  const start = Math.floor(minVal / step) * step;
  const end = Math.ceil(maxVal / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= end; v += step) {
    ticks.push(v);
  }
  return ticks.length > 0 ? ticks : [minVal, maxVal];
}

export function truncateLabel(text: string, maxLen: number): string {
  return text.length > maxLen ? `${text.slice(0, maxLen - 1)}…` : text;
}

function boxesOverlap(a: LabelBox, b: LabelBox): boolean {
  return a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;
}

function tryPlaceCandidate(args: {
  cand: LabelCandidate;
  bounds: LabelBox;
  boxes: LabelBox[];
}): PlacedLabel | null {
  const { cand, bounds, boxes } = args;
  const widthEst = cand.text.length * 5.6 + 6;
  const heightEst = 13;
  for (const off of LABEL_OFFSETS) {
    const tx = cand.cx + off.dx;
    const ty = cand.cy + off.dy;
    const box: LabelBox = {
      x0: off.anchor === "start" ? tx - 2 : tx - widthEst,
      x1: off.anchor === "start" ? tx + widthEst : tx + 2,
      y0: ty - heightEst + 2,
      y1: ty + 3,
    };
    const outOfBounds =
      box.x0 < bounds.x0 || box.x1 > bounds.x1 || box.y0 < bounds.y0 || box.y1 > bounds.y1;
    if (outOfBounds || boxes.some((existing) => boxesOverlap(existing, box))) continue;
    boxes.push(box);
    return {
      id: cand.id,
      x: tx,
      y: ty,
      textAnchor: off.anchor,
      text: cand.text,
      bold: cand.bold,
      fill: cand.fill,
    };
  }
  return null;
}

export function placeDeconflictedLabels(args: {
  candidates: LabelCandidate[];
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  maxLabels: number;
}): PlacedLabel[] {
  const { candidates, minX, maxX, minY, maxY, maxLabels } = args;
  const placed: PlacedLabel[] = [];
  const boxes: LabelBox[] = [];
  const bounds: LabelBox = { x0: minX, y0: minY, x1: maxX, y1: maxY };
  for (const cand of candidates) {
    if (placed.length >= maxLabels) break;
    const hit = tryPlaceCandidate({ cand, bounds, boxes });
    if (hit) placed.push(hit);
  }
  return placed;
}

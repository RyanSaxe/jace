"""Pure-PNG raster canvas and Arena report figure renderers.

Draws 8-bit RGB PNG figures directly via NumPy and zlib with zero external plotting
dependencies for the local Arena report (`elo_progression.png`, `elo_vs_ms.png`,
`early_forecast.png`, and `puzzle_summary.png`).
"""

from __future__ import annotations

import itertools
import math
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from jace.eval.ratings import BradleyTerryRating
    from jace.eval.report import EarlyForecastSummary, PuzzleSourceSummary

__all__ = (
    "render_elo_figure",
    "render_forecast_figure",
    "render_pareto_figure",
    "render_puzzle_figure",
)

_FIG_WIDTH = 960
_FIG_HEIGHT = 540
_BG_COLOR = (252, 251, 248)
_PANEL_COLOR = (255, 255, 255)
_GRID_COLOR = (231, 228, 220)
_AXIS_COLOR = (148, 163, 184)
_TEXT_COLOR = (28, 25, 23)
_MUTED_COLOR = (87, 83, 78)
_INDIGO = (79, 70, 229)
_EMERALD = (5, 150, 105)
_AMBER = (217, 119, 6)
_SLATE = (100, 116, 139)
_LIGHT_INDIGO = (199, 210, 254)

# 5x7 uppercase/ASCII bitmap glyphs (each row is a 5-bit mask, MSB on left).
_FONT_5X7: dict[str, tuple[int, ...]] = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    "!": (4, 4, 4, 4, 4, 0, 4),
    "%": (25, 26, 4, 4, 11, 19, 0),
    "(": (2, 4, 8, 8, 8, 4, 2),
    ")": (8, 4, 2, 2, 2, 4, 8),
    "+": (0, 4, 4, 31, 4, 4, 0),
    ",": (0, 0, 0, 0, 6, 4, 8),
    "-": (0, 0, 0, 31, 0, 0, 0),
    ".": (0, 0, 0, 0, 0, 12, 12),
    "/": (1, 2, 4, 4, 8, 16, 0),
    ":": (0, 12, 12, 0, 12, 12, 0),
    "<": (2, 4, 8, 16, 8, 4, 2),
    "=": (0, 0, 31, 0, 31, 0, 0),
    ">": (8, 4, 2, 1, 2, 4, 8),
    "_": (0, 0, 0, 0, 0, 0, 31),
    "0": (14, 17, 19, 21, 25, 17, 14),
    "1": (4, 12, 4, 4, 4, 4, 14),
    "2": (14, 17, 1, 6, 8, 16, 31),
    "3": (30, 1, 1, 14, 1, 1, 30),
    "4": (2, 6, 10, 18, 31, 2, 2),
    "5": (31, 16, 30, 1, 1, 17, 14),
    "6": (6, 8, 16, 30, 17, 17, 14),
    "7": (31, 1, 2, 4, 8, 8, 8),
    "8": (14, 17, 17, 14, 17, 17, 14),
    "9": (14, 17, 17, 15, 1, 2, 12),
    "A": (14, 17, 17, 31, 17, 17, 17),
    "B": (30, 17, 17, 30, 17, 17, 30),
    "C": (14, 17, 16, 16, 16, 17, 14),
    "D": (28, 18, 17, 17, 17, 18, 28),
    "E": (31, 16, 16, 30, 16, 16, 31),
    "F": (31, 16, 16, 30, 16, 16, 16),
    "G": (14, 17, 16, 23, 17, 17, 15),
    "H": (17, 17, 17, 31, 17, 17, 17),
    "I": (14, 4, 4, 4, 4, 4, 14),
    "J": (7, 2, 2, 2, 18, 18, 12),
    "K": (17, 18, 20, 24, 20, 18, 17),
    "L": (16, 16, 16, 16, 16, 16, 31),
    "M": (17, 27, 21, 21, 17, 17, 17),
    "N": (17, 25, 21, 19, 17, 17, 17),
    "O": (14, 17, 17, 17, 17, 17, 14),
    "P": (30, 17, 17, 30, 16, 16, 16),
    "Q": (14, 17, 17, 17, 21, 18, 13),
    "R": (30, 17, 17, 30, 20, 18, 17),
    "S": (15, 16, 16, 14, 1, 1, 30),
    "T": (31, 4, 4, 4, 4, 4, 4),
    "U": (17, 17, 17, 17, 17, 17, 14),
    "V": (17, 17, 17, 17, 17, 10, 4),
    "W": (17, 17, 17, 21, 21, 27, 17),
    "X": (17, 17, 10, 4, 10, 17, 17),
    "Y": (17, 17, 10, 4, 4, 4, 4),
    "Z": (31, 1, 2, 4, 8, 16, 31),
}


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class _PlotCanvas:
    def __init__(self, width: int = _FIG_WIDTH, height: int = _FIG_HEIGHT) -> None:
        self.width = width
        self.height = height
        self.rgb = np.full((height, width, 3), _BG_COLOR, dtype=np.uint8)

    def fill_rect(self, x0: int, y0: int, *, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        xa = max(0, min(self.width, x0, x1))
        xb = max(0, min(self.width, max(x0, x1)))
        ya = max(0, min(self.height, y0, y1))
        yb = max(0, min(self.height, max(y0, y1)))
        if xa < xb and ya < yb:
            self.rgb[ya:yb, xa:xb] = color

    def draw_line(
        self,
        p0: tuple[int, int],
        p1: tuple[int, int],
        *,
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        x0, y0 = p0
        x1, y1 = p1
        steps = max(1, abs(x1 - x0), abs(y1 - y0))
        xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(np.int32)
        ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(np.int32)
        rad = max(0, thickness // 2)
        for dx in range(-rad, rad + 1):
            for dy in range(-rad, rad + 1):
                px = np.clip(xs + dx, 0, self.width - 1)
                py = np.clip(ys + dy, 0, self.height - 1)
                self.rgb[py, px] = color

    def draw_circle(self, cx: int, cy: int, *, radius: int, color: tuple[int, int, int]) -> None:
        x0, x1 = max(0, cx - radius), min(self.width - 1, cx + radius)
        y0, y1 = max(0, cy - radius), min(self.height - 1, cy + radius)
        if x0 > x1 or y0 > y1:
            return
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        self.rgb[y0 : y1 + 1, x0 : x1 + 1][mask] = color

    def draw_text(
        self,
        x: int,
        y: int,
        *,
        text: str,
        color: tuple[int, int, int] = _TEXT_COLOR,
        scale: int = 1,
    ) -> None:
        cursor = x
        for ch in text.upper():
            glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
            for row_idx, bits in enumerate(glyph):
                for col_idx in range(5):
                    if bits & (1 << (4 - col_idx)):
                        px = cursor + col_idx * scale
                        py = y + row_idx * scale
                        self.fill_rect(px, py, x1=px + scale, y1=py + scale, color=color)
            cursor += 6 * scale


@dataclass(frozen=True, slots=True)
class _PlotFrame:
    left: int
    top: int
    right: int
    bottom: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def map_x(self, val: float) -> int:
        span = self.x_max - self.x_min or 1.0
        return round(self.left + (val - self.x_min) / span * (self.right - self.left))

    def map_y(self, val: float) -> int:
        span = self.y_max - self.y_min or 1.0
        return round(self.bottom - (val - self.y_min) / span * (self.bottom - self.top))


def _init_panel(canvas: _PlotCanvas, frame: _PlotFrame, *, title: str, y_fmt: str = "{:.0f}") -> None:
    canvas.draw_text(frame.left, 18, text=title, color=_TEXT_COLOR, scale=2)
    canvas.fill_rect(frame.left, frame.top, x1=frame.right, y1=frame.bottom, color=_PANEL_COLOR)
    for step in range(5):
        frac = step / 4.0
        y_val = frame.y_min + frac * (frame.y_max - frame.y_min)
        py = frame.map_y(y_val)
        canvas.draw_line((frame.left, py), (frame.right, py), color=_GRID_COLOR, thickness=1)
        label = y_fmt.format(y_val)
        canvas.draw_text(max(6, frame.left - len(label) * 6 - 8), py - 3, text=label, color=_MUTED_COLOR)
    canvas.draw_line((frame.left, frame.top), (frame.left, frame.bottom), color=_AXIS_COLOR, thickness=1)
    canvas.draw_line((frame.left, frame.bottom), (frame.right, frame.bottom), color=_AXIS_COLOR, thickness=1)


def render_elo_figure(ratings: Sequence[BradleyTerryRating], out_path: Path, *, format_id: str) -> bool:
    if not ratings:
        return False
    ordered = list(reversed(ratings[:18]))
    y_min = min(item.ci95_low for item in ordered) - 25.0
    y_max = max(item.ci95_high for item in ordered) + 35.0
    frame = _PlotFrame(
        left=84,
        top=64,
        right=924,
        bottom=448,
        x_min=-0.5,
        x_max=len(ordered) - 0.5,
        y_min=y_min,
        y_max=y_max,
    )
    canvas = _PlotCanvas()
    _init_panel(canvas, frame, title=f"ARENA BRADLEY-TERRY ELO ({format_id})")
    anchor_y = frame.map_y(1500.0)
    if frame.top <= anchor_y <= frame.bottom:
        canvas.draw_line((frame.left, anchor_y), (frame.right, anchor_y), color=_SLATE, thickness=1)
    points: list[tuple[int, int]] = []
    for idx, item in enumerate(ordered):
        px = frame.map_x(float(idx))
        py = frame.map_y(item.elo)
        points.append((px, py))
        ci_p0 = (px, frame.map_y(item.ci95_low))
        ci_p1 = (px, frame.map_y(item.ci95_high))
        canvas.draw_line(ci_p0, ci_p1, color=_LIGHT_INDIGO, thickness=3)
    for p0, p1 in itertools.pairwise(points):
        canvas.draw_line(p0, p1, color=_INDIGO, thickness=2)
    for idx, (item, (px, py)) in enumerate(zip(ordered, points, strict=True)):
        dot_color = _EMERALD if idx == len(ordered) - 1 else (_SLATE if item.ci95_low == item.ci95_high else _INDIGO)
        canvas.draw_circle(px, py, radius=5, color=dot_color)
        elo_x = max(frame.left, px - 14)
        elo_y = max(frame.top + 4, py - 16)
        canvas.draw_text(elo_x, elo_y, text=f"{item.elo:.0f}", color=_TEXT_COLOR)
        short = item.player_id[:11]
        y_lbl = frame.bottom + 12 + (idx % 2) * 14
        canvas.draw_text(max(4, px - len(short) * 3), y_lbl, text=short, color=_MUTED_COLOR)
    _write_png(out_path, canvas.rgb)
    return True


def render_pareto_figure(ratings: Sequence[BradleyTerryRating], out_path: Path, *, format_id: str) -> bool:
    timed = [item for item in ratings if item.mean_ms_per_decision is not None and item.mean_ms_per_decision > 0.0]
    if not timed:
        return False
    xs = [float(item.mean_ms_per_decision or 0.0) for item in timed]
    x_max = max(xs) * 1.15 + 1.0
    y_min = min(item.ci95_low for item in timed) - 20.0
    y_max = max(item.ci95_high for item in timed) + 25.0
    frame = _PlotFrame(left=84, top=64, right=924, bottom=468, x_min=0.0, x_max=x_max, y_min=y_min, y_max=y_max)
    canvas = _PlotCanvas()
    _init_panel(canvas, frame, title=f"ELO VS MS/DECISION FRONTIER ({format_id})")
    frontier = sorted(timed, key=lambda item: float(item.mean_ms_per_decision or 0.0))
    best_so_far = -math.inf
    pareto_pts: list[tuple[int, int]] = []
    for item in frontier:
        ms_val = float(item.mean_ms_per_decision or 0.0)
        px = frame.map_x(ms_val)
        py = frame.map_y(item.elo)
        is_pareto = item.elo > best_so_far
        if is_pareto:
            best_so_far = item.elo
            pareto_pts.append((px, py))
        canvas.draw_circle(px, py, radius=6 if is_pareto else 4, color=_EMERALD if is_pareto else _INDIGO)
        lbl_x = min(frame.right - 60, px + 8)
        lbl_y = max(frame.top + 4, py - 5)
        canvas.draw_text(lbl_x, lbl_y, text=item.player_id[:12], color=_TEXT_COLOR)
    for p0, p1 in itertools.pairwise(pareto_pts):
        canvas.draw_line(p0, p1, color=_EMERALD, thickness=2)
    canvas.draw_text(frame.left + 320, frame.bottom + 24, text="MEAN MS PER DECISION", color=_MUTED_COLOR)
    _write_png(out_path, canvas.rgb)
    return True


def render_forecast_figure(summary: EarlyForecastSummary, out_path: Path) -> bool:
    if not summary.points:
        return False
    frame = _PlotFrame(left=84, top=64, right=924, bottom=468, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
    canvas = _PlotCanvas()
    stats_tag = ""
    if summary.pearson_r is not None and summary.spearman_rho is not None:
        stats_tag = f" (R={summary.pearson_r:.2f}, RHO={summary.spearman_rho:.2f})"
    _init_panel(canvas, frame, title=f"EARLY U<=200 VS PEAK WIN RATE{stats_tag}", y_fmt="{:.0%}")
    diag_p0 = (frame.map_x(0.0), frame.map_y(0.0))
    diag_p1 = (frame.map_x(1.0), frame.map_y(1.0))
    canvas.draw_line(diag_p0, diag_p1, color=_GRID_COLOR, thickness=1)
    for pt in summary.points:
        px = frame.map_x(pt.early_win_rate)
        py = frame.map_y(pt.peak_win_rate)
        canvas.draw_circle(px, py, radius=5, color=_AMBER)
        lbl_x = min(frame.right - 60, px + 7)
        lbl_y = max(frame.top + 4, py - 4)
        canvas.draw_text(lbl_x, lbl_y, text=pt.run_id.split("/")[-1][:12], color=_TEXT_COLOR)
    for step in range(5):
        val = step / 4.0
        canvas.draw_text(frame.map_x(val) - 10, frame.bottom + 10, text=f"{val:.0%}", color=_MUTED_COLOR)
    canvas.draw_text(frame.left + 310, frame.bottom + 32, text="EARLY WIN RATE (U <= 200)", color=_MUTED_COLOR)
    _write_png(out_path, canvas.rgb)
    return True


def render_puzzle_figure(puzzles: Sequence[PuzzleSourceSummary], out_path: Path, *, format_id: str) -> bool:
    if not puzzles:
        return False
    max_count = max(item.count for item in puzzles)
    frame = _PlotFrame(
        left=84,
        top=64,
        right=924,
        bottom=448,
        x_min=-0.5,
        x_max=len(puzzles) - 0.5,
        y_min=0.0,
        y_max=max(1.0, max_count * 1.2),
    )
    canvas = _PlotCanvas()
    _init_panel(canvas, frame, title=f"VERIFIED PUZZLES BY SOURCE ({format_id})")
    bar_half = max(14, min(48, (frame.right - frame.left) // max(1, len(puzzles) * 3)))
    for idx, item in enumerate(puzzles):
        cx = frame.map_x(float(idx))
        top_y = frame.map_y(float(item.count))
        canvas.fill_rect(cx - bar_half, top_y, x1=cx + bar_half, y1=frame.bottom, color=_INDIGO)
        canvas.draw_text(cx - 10, max(frame.top + 4, top_y - 14), text=str(item.count), color=_TEXT_COLOR)
        label = item.source[:14]
        canvas.draw_text(max(4, cx - len(label) * 3), frame.bottom + 14, text=label, color=_MUTED_COLOR)
    _write_png(out_path, canvas.rgb)
    return True

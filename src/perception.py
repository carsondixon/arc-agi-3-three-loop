"""Grid -> text rendering + structured object extraction.

Stage 1 used hex grid only. Path 1 (Stage 4.5) adds structured perception:
connected-component analysis per color, returning a list of named objects
that Claude can reference by ID for click actions. This fixes the
coordinate-blindness failure mode observed across all 6 Phase 1 click
games.

The grid coming from arc_agi is a 64x64 int8 ndarray with values 0-15.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

GRID_W = 64
GRID_H = 64

# ARC-AGI standard palette (mostly for documentation, not used in encoding).
COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "fuchsia",
    7: "orange",
    8: "teal",
    9: "brown",
    10: "white",
    11: "light_blue",
    12: "magenta",
    13: "olive",
    14: "navy",
    15: "maroon",
}


def grid_to_hex(grid: np.ndarray) -> str:
    """Render a 64x64 int8 grid as a multiline hex string.

    Each row is a string of GRID_W hex digits. Rows separated by newlines.
    Total: ~4160 chars (~1100 tokens).
    """
    if grid.ndim != 2:
        raise ValueError(f"expected 2D grid, got shape {grid.shape}")
    lines = []
    for y in range(grid.shape[0]):
        row = grid[y]
        # vectorized: convert each int to hex; clip to [0, 15]
        row_clipped = np.clip(row, 0, 15).astype(int)
        lines.append("".join(f"{v:x}" for v in row_clipped))
    return "\n".join(lines)


def color_legend(grid: np.ndarray) -> str:
    """Build a per-frame color legend showing only the colors actually present."""
    present = sorted({int(v) for v in np.unique(np.clip(grid, 0, 15))})
    parts = [f"{v:x}={COLOR_NAMES.get(v, '?')}" for v in present]
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Connected-component object extraction (Path 1 perception upgrade)
# --------------------------------------------------------------------------- #


@dataclass
class DetectedObject:
    """One connected-component blob in the grid."""
    obj_id: str           # e.g. "O_c_1" -- color hex + index
    color: int            # 0-15
    centroid_y: int       # integer row
    centroid_x: int       # integer col
    min_y: int
    max_y: int
    min_x: int
    max_x: int
    size: int             # cell count
    shape: str            # "rect", "line_h", "line_v", "blob", "single"

    @property
    def color_name(self) -> str:
        return COLOR_NAMES.get(self.color, "?")

    def render(self) -> str:
        return (
            f"[{self.obj_id}] color={self.color:x}({self.color_name}) "
            f"centroid=({self.centroid_y},{self.centroid_x}) "
            f"bbox=({self.min_y}-{self.max_y},{self.min_x}-{self.max_x}) "
            f"size={self.size} shape={self.shape}"
        )


def _classify_shape(min_y: int, max_y: int, min_x: int, max_x: int, size: int) -> str:
    h = max_y - min_y + 1
    w = max_x - min_x + 1
    bbox_area = h * w
    if size == 1:
        return "single"
    if h == 1 and w > 1:
        return "line_h"
    if w == 1 and h > 1:
        return "line_v"
    # Mostly-filled rectangle
    if size >= 0.85 * bbox_area and bbox_area >= 4:
        return "rect"
    return "blob"


def extract_objects(
    grid: np.ndarray,
    min_size: int = 2,
    ignore_color: int | None = 0,
    max_objects: int = 24,
) -> list[DetectedObject]:
    """Return a list of connected-component objects in the grid.

    Implementation: for each color value present in the grid (skipping
    `ignore_color`, typically 0 = black background), label connected
    components and emit one DetectedObject per component with size >=
    min_size. Returns at most `max_objects`, sorted by size descending so
    the most salient objects come first.

    Cells are 4-connected (von Neumann), matching how a player typically
    perceives "the same object".
    """
    if grid.ndim != 2:
        raise ValueError(f"expected 2D grid, got shape {grid.shape}")

    out: list[DetectedObject] = []
    grid_clipped = np.clip(grid, 0, 15).astype(int)
    unique_colors = sorted({int(c) for c in np.unique(grid_clipped)})
    if ignore_color is not None and ignore_color in unique_colors:
        unique_colors.remove(ignore_color)

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # 4-connectivity

    for color in unique_colors:
        mask = grid_clipped == color
        if not mask.any():
            continue
        labeled, n = ndimage.label(mask, structure=structure)
        for idx in range(1, n + 1):
            cells = np.argwhere(labeled == idx)  # (size, 2): rows of (y, x)
            size = int(cells.shape[0])
            if size < min_size:
                continue
            min_y, min_x = cells.min(axis=0).tolist()
            max_y, max_x = cells.max(axis=0).tolist()
            cy = int(round(cells[:, 0].mean()))
            cx = int(round(cells[:, 1].mean()))
            shape = _classify_shape(min_y, max_y, min_x, max_x, size)
            out.append(
                DetectedObject(
                    obj_id=f"O_{color:x}_{idx}",
                    color=color,
                    centroid_y=cy,
                    centroid_x=cx,
                    min_y=int(min_y),
                    max_y=int(max_y),
                    min_x=int(min_x),
                    max_x=int(max_x),
                    size=size,
                    shape=shape,
                )
            )

    out.sort(key=lambda o: -o.size)
    return out[:max_objects]


def render_object_inventory(objects: list[DetectedObject]) -> str:
    """Multi-line rendering of detected objects, ready for prompt injection."""
    if not objects:
        return "(no objects detected; grid is empty or uniform background)"
    lines = [o.render() for o in objects]
    return "\n".join(lines)


def object_index(objects: list[DetectedObject]) -> dict[str, DetectedObject]:
    """Return {obj_id: DetectedObject} for fast lookup at action-execution time."""
    return {o.obj_id: o for o in objects}


# --------------------------------------------------------------------------- #
# Frame differencing (Stage 4.6 perception upgrade)
# --------------------------------------------------------------------------- #


@dataclass
class FrameDelta:
    """What changed between two consecutive frames."""
    moved: list[tuple[DetectedObject, DetectedObject, int, int]]  # (prev, curr, dy, dx)
    appeared: list[DetectedObject]
    disappeared: list[DetectedObject]
    changed_cells: int


def diff_objects(
    prev_objects: list[DetectedObject],
    curr_objects: list[DetectedObject],
    prev_grid: np.ndarray | None = None,
    curr_grid: np.ndarray | None = None,
) -> FrameDelta:
    """Match objects across two frames and report movement / appearance / disappearance.

    Matching heuristic: same color, similar size (within max(3, 50%)), nearest
    centroid (Manhattan). Greedy, largest-first. This is deliberately simple --
    it just needs to surface "the player moved" reliably enough for Claude to
    learn the action->direction mapping.
    """
    from collections import defaultdict

    prev_by_color: dict[int, list[DetectedObject]] = defaultdict(list)
    for o in prev_objects:
        prev_by_color[o.color].append(o)
    curr_by_color: dict[int, list[DetectedObject]] = defaultdict(list)
    for o in curr_objects:
        curr_by_color[o.color].append(o)

    moved: list[tuple[DetectedObject, DetectedObject, int, int]] = []
    appeared: list[DetectedObject] = []
    disappeared: list[DetectedObject] = []

    all_colors = set(prev_by_color) | set(curr_by_color)
    for color in all_colors:
        prev_list = sorted(prev_by_color.get(color, []), key=lambda o: -o.size)
        curr_list = sorted(curr_by_color.get(color, []), key=lambda o: -o.size)
        used_prev: set[int] = set()
        for c in curr_list:
            best_i = -1
            best_dist = 10**9
            for i, p in enumerate(prev_list):
                if i in used_prev:
                    continue
                if abs(p.size - c.size) > max(3, 0.5 * c.size):
                    continue
                dist = abs(p.centroid_y - c.centroid_y) + abs(p.centroid_x - c.centroid_x)
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
            if best_i >= 0:
                used_prev.add(best_i)
                p = prev_list[best_i]
                dy = c.centroid_y - p.centroid_y
                dx = c.centroid_x - p.centroid_x
                if dy != 0 or dx != 0:
                    moved.append((p, c, dy, dx))
            else:
                appeared.append(c)
        for i, p in enumerate(prev_list):
            if i not in used_prev:
                disappeared.append(p)

    changed_cells = 0
    if prev_grid is not None and curr_grid is not None and prev_grid.shape == curr_grid.shape:
        changed_cells = int(np.count_nonzero(prev_grid != curr_grid))

    return FrameDelta(moved=moved, appeared=appeared, disappeared=disappeared, changed_cells=changed_cells)


def render_delta(delta: FrameDelta, last_action: str) -> str:
    """Render a FrameDelta as a CHANGES section for the prompt."""
    lines = [f"CHANGES SINCE YOUR LAST ACTION ({last_action}):"]
    if not delta.moved and not delta.appeared and not delta.disappeared and delta.changed_cells == 0:
        lines.append("  NOTHING CHANGED -- this action had zero visible effect on the grid.")
        return "\n".join(lines)
    for p, c, dy, dx in delta.moved:
        parts = []
        if dy:
            parts.append(f"{dy:+d} in y ({'down' if dy > 0 else 'up'})")
        if dx:
            parts.append(f"{dx:+d} in x ({'right' if dx > 0 else 'left'})")
        lines.append(
            f"  object color={c.color:x}({c.color_name}) size={c.size} MOVED "
            f"({p.centroid_y},{p.centroid_x})->({c.centroid_y},{c.centroid_x})  [{', '.join(parts)}]"
        )
    for c in delta.appeared:
        lines.append(f"  object color={c.color:x}({c.color_name}) size={c.size} APPEARED at ({c.centroid_y},{c.centroid_x})")
    for p in delta.disappeared:
        lines.append(f"  object color={p.color:x}({p.color_name}) size={p.size} DISAPPEARED from ({p.centroid_y},{p.centroid_x})")
    lines.append(f"  ({delta.changed_cells} cells changed total)")
    return "\n".join(lines)

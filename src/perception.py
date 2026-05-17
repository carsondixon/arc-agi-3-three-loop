"""Grid -> text rendering.

Stage 1 keeps perception naked: a hex grid string, nothing fancy. Stage 3
will add structured perception (connected-component analysis, object
identification, geometric primitives).

The grid coming from arc_agi is a 64x64 int8 ndarray with values 0-15.
We render each cell as a single hex digit (0-9, a-f), one row per line.
"""

from __future__ import annotations

import numpy as np

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

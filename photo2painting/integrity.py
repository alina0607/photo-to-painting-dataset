"""Every saved painting holds only finite numbers.

A NaN spreads: any sum or product with it is NaN, so one bad stroke turns every pixel
painted after it NaN. Worse, NaN turned into an integer token can land on a valid
token id, so a format check passes. Each painting is therefore checked directly with
`np.isfinite`: its canvas and every stroke field.

A painting's stage file (``stage<k>.npz``) holds the canvas after the stage
(``canvas``) and, per layer ``j``, the strokes' fields ``<j>_<field>``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STROKE_FIELDS = ("xy", "lw", "angle", "colour", "alpha")


@dataclass(frozen=True)
class StageCheck:
    """One stage file checked.

    Attributes:
        canvas_finite: every canvas pixel is finite.
        bad_strokes: per layer index, how many strokes have a non-finite field.
    """

    canvas_finite: bool
    bad_strokes: dict[int, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Nothing non-finite anywhere."""
        return self.canvas_finite and not self.bad_strokes


def bad_stroke_mask(arrays: dict[str, np.ndarray], layer: int) -> np.ndarray:
    """For one layer, True for each stroke with any non-finite field."""
    n = len(arrays[f"{layer}_xy"])
    finite = np.ones(n, bool)
    for f in STROKE_FIELDS:
        finite &= np.isfinite(arrays[f"{layer}_{f}"].reshape(n, -1)).all(1)
    return ~finite


def check_stage(path: Path) -> StageCheck:
    """Check one stage file."""
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    layers = sorted({int(k.split("_")[0]) for k in arrays if k.endswith("_xy")})
    bad = {j: int(bad_stroke_mask(arrays, j).sum()) for j in layers}
    return StageCheck(
        canvas_finite=bool(np.isfinite(arrays["canvas"]).all()),
        bad_strokes={j: n for j, n in bad.items() if n},
    )


@dataclass(frozen=True)
class ScanSummary:
    """A folder of paintings scanned.

    Attributes:
        paintings: how many paintings were checked.
        bad_canvas: names of the paintings whose canvas is not finite.
        bad_strokes: names of the paintings with a non-finite stroke.
    """

    paintings: int
    bad_canvas: list[str]
    bad_strokes: list[str]


def scan(folders: Iterable[Path], stage: int = 7) -> ScanSummary:
    """Check the given stage of every painting folder (one folder per cat)."""
    n, canvas, strokes = 0, [], []
    for d in sorted(folders):
        f = d / f"stage{stage}.npz"
        if not f.exists():
            continue
        n += 1
        c = check_stage(f)
        if not c.canvas_finite:
            canvas.append(d.name)
        if c.bad_strokes:
            strokes.append(d.name)
    return ScanSummary(paintings=n, bad_canvas=canvas, bad_strokes=strokes)

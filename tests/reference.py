"""The research code this package was extracted from, for equivalence tests.

Set SELL_ANYTHING_ML to the research repo's ``ml`` folder to run them; without it
they are skipped (CI runs the synthetic tests only).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from photo2painting.ingest import RawCat, afhq_cats

# An AFHQ cat that every step of the research code has processed.
REAL_CAT = "pixabay_cat_001129"


def reference(module: str) -> ModuleType:
    """A module of the research code; the calling test is skipped when it is absent."""
    root = os.environ.get("SELL_ANYTHING_ML")
    if not root:
        pytest.skip("SELL_ANYTHING_ML is not set")
    for path in (root, os.path.join(root, "tools")):
        if path not in sys.path:
            sys.path.append(path)
    return importlib.import_module(module)


def ml_root() -> Path:
    """The research repo's ``ml`` folder."""
    return Path(reference("paint_stages").ML)


def real_cat() -> RawCat:
    """`REAL_CAT` with its photo, mask and predicted face points."""
    m = reference("mask_audit")
    return next(
        c for c in afhq_cats(m.LABELS, m.AFHQ, m.AFHQ_MASKS) if c.cat_id == REAL_CAT
    )

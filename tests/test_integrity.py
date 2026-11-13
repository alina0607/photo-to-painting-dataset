from pathlib import Path

import numpy as np

from photo2painting.integrity import check_stage, scan
from tests.reference import ml_root


def write_stage(path: Path, alpha: np.ndarray, canvas_value: float = 0.5) -> None:
    path.mkdir(parents=True, exist_ok=True)
    n = len(alpha)
    arrays = {
        "canvas": np.full((4, 4, 3), canvas_value, np.float32),
        "0_xy": np.zeros((n, 2)),
        "0_lw": np.ones((n, 2)),
        "0_angle": np.zeros(n),
        "0_colour": np.zeros((n, 3)),
        "0_alpha": alpha,
    }
    np.savez(path / "stage7.npz", **arrays)  # type: ignore[arg-type]


def test_finite_painting_is_ok(tmp_path):
    write_stage(tmp_path / "a", np.array([0.5, 0.7]))
    assert check_stage(tmp_path / "a" / "stage7.npz").ok


def test_nan_stroke_and_canvas_are_found(tmp_path):
    write_stage(tmp_path / "good", np.array([0.5]))
    write_stage(tmp_path / "stroke", np.array([0.5, np.nan, np.nan]))
    write_stage(tmp_path / "canvas", np.array([0.5]), canvas_value=np.nan)
    c = check_stage(tmp_path / "stroke" / "stage7.npz")
    assert c.bad_strokes == {0: 2}
    s = scan(p for p in tmp_path.iterdir())
    assert s.paintings == 3
    assert s.bad_canvas == ["canvas"]
    assert s.bad_strokes == ["stroke"]


def test_the_kept_cuda_paintings():
    """The 450 paintings first painted on CUDA, kept before they were repainted."""
    folder = ml_root() / "runs/paint_stages_kaggle_cuda"
    s = scan(p for p in folder.iterdir() if p.is_dir())
    assert (s.paintings, len(s.bad_canvas), len(s.bad_strokes)) == (450, 55, 450)

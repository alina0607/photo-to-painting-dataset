import json

import numpy as np
import pytest
from PIL import Image

from photo2painting.duplicates import (
    SAME_PHOTO_NCC,
    aligned_similarity,
    candidates,
    cheap_scores,
    cross_split_pairs,
    group_split,
    groups,
    thumbnails,
)
from tests.reference import ml_root


def textured(seed: int, side: int = 128) -> np.ndarray:
    """A smooth random RGB image (blurred noise), uint8."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (side // 8, side // 8, 3)).astype(np.uint8)
    im = Image.fromarray(small).resize((side, side), Image.Resampling.BICUBIC)
    return np.asarray(im)


def recropped(img: np.ndarray, zoom_px: int) -> np.ndarray:
    """The same picture cut tighter and resized back: a re-cropped copy."""
    side = img.shape[0]
    crop = Image.fromarray(img).crop((zoom_px, zoom_px, side - zoom_px, side - zoom_px))
    return np.asarray(crop.resize((side, side), Image.Resampling.BILINEAR))


def test_a_recropped_and_mirrored_copy_is_the_same_photo():
    a = textured(0)
    b = recropped(a, 4)[:, ::-1]
    r = aligned_similarity(a, b)
    assert r.mirrored
    assert r.ncc >= SAME_PHOTO_NCC
    assert aligned_similarity(a, textured(1)).ncc < 0.6


def test_cheap_scores_find_the_copy():
    imgs = [textured(k) for k in range(5)] + [recropped(textured(2), 4)]
    grey = np.stack(
        [np.asarray(Image.fromarray(x).convert("L").resize((64, 64))) for x in imgs]
    )
    s = cheap_scores(grey)
    assert len(s.corr) == 15
    best = int(np.argmax(s.corr))
    assert (s.i[best], s.j[best]) == (2, 5)
    assert best in candidates(s)


def test_groups_and_group_split():
    pairs = [("a", "b"), ("b", "c"), ("d", "e")]
    g = groups(pairs)
    assert g == [["a", "b", "c"], ["d", "e"]]
    split = {
        "a": "train",
        "b": "val",
        "c": "train",
        "d": "train",
        "e": "train",
        "f": "test",
    }
    assert cross_split_pairs(pairs, split) == [("a", "b"), ("b", "c")]
    new = group_split(split, g)
    assert new["a"] == new["b"] == new["c"] == "val"
    assert new["f"] == "test"
    assert not cross_split_pairs(pairs, new)


def test_the_known_pair_matches_the_stored_measurement():
    """The pair that started the search: one photo in train and in val."""
    root = ml_root()
    stored = json.loads((root / "runs/afhq_duplicates/pairs.json").read_text())["pairs"]
    row = next(
        p
        for p in stored
        if {p["a"], p["b"]} == {"pixabay_cat_000709", "pixabay_cat_003118"}
    )
    names = sorted([row["a"], row["b"]])
    paths = [next((root / "data/raw/afhq").glob(f"*/cat/{n}.png")) for n in names]
    grey, rgb = thumbnails(paths)
    r = aligned_similarity(rgb[0], rgb[1])
    assert r.ncc == pytest.approx(row["aligned_ncc"], abs=1e-4)
    assert r.ssim == pytest.approx(row["ssim"], abs=1e-4)
    assert r.mad_rgb == pytest.approx(row["mean_abs_rgb_diff"], abs=0.01)
    assert cheap_scores(grey).corr[0] == pytest.approx(row["corr16"], abs=1e-4)

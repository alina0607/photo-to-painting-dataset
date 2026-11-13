import numpy as np
import pytest

from photo2painting.ingest import (
    SIDE,
    cutout,
    face_cover,
    head_shape,
    load,
    measure,
    parse_cat_points,
)
from tests.reference import ml_root, real_cat, reference
from tests.synthetic import synthetic_cat


def test_parse_cat_points():
    text = "9 " + " ".join(str(v) for v in range(18))
    p = parse_cat_points(text)
    assert p.shape == (9, 2)
    assert p[8].tolist() == [16.0, 17.0]


def test_parse_cat_points_rejects_other_counts():
    with pytest.raises(ValueError, match="expected 9 points"):
        parse_cat_points("8 1 2")


def test_face_cover_is_one_when_the_mask_is_full():
    _, mask, p = synthetic_cat()
    assert face_cover(np.ones_like(mask), p) == 1.0
    assert face_cover(np.zeros_like(mask), p) == 0.0


def test_measure_on_a_drawn_face():
    photo, mask, p = synthetic_cat()
    m = measure(photo, mask, p)
    assert m.eye_dist_px == 60.0
    assert m.face_cover == 1.0
    assert m.tips_in_photo
    assert m.eye_contrast > 0.035


def test_head_shape_contains_the_face_points_but_not_the_corners():
    _, mask, p = synthetic_cat()
    head = head_shape(p, mask.shape)
    for x, y in p[[0, 1, 2]]:
        assert head[int(y), int(x)]
    assert not head[0, 0]
    assert not head[-1, -1]


def test_head_shape_turns_with_the_head():
    _, mask, p = synthetic_cat()
    c = p[:2].mean(0)
    turn = np.radians(30)
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    q = (p - c) @ rot.T + c
    # the same head turned 30 degrees covers the same area, within the grid's rounding
    a = int(head_shape(p, mask.shape).sum())
    b = int(head_shape(q, mask.shape).sum())
    assert abs(a - b) / a < 0.01


def test_cutout_fields():
    photo, mask, p = synthetic_cat()
    c = cutout(photo, mask, p)
    assert c.target.shape == (SIDE, SIDE, 3)
    assert c.target.dtype == np.uint8
    assert set(np.unique(c.tri)) <= {0, 1, 2}
    assert (c.tri == 1).sum() > 0.2 * SIDE * SIDE
    assert c.points.shape == (9, 2)
    # a square up to the rounding of its two edges
    assert abs((c.box[2] - c.box[0]) - (c.box[3] - c.box[1])) <= 1


# ---------------------------------------------- against the research code
def test_measure_matches_the_research_code():
    rx = reference("review_export")
    raw = real_cat()
    photo, mask = load(raw)
    old = rx.measure(photo, mask, raw.points)
    assert measure(photo, mask, raw.points).__dict__ == old


def test_cutout_matches_the_research_code():
    rp = reference("repaint_prep")
    raw = real_cat()
    old = rp.prepare((raw.source, raw.cat_id, raw.photo, raw.mask, raw.points))
    new = cutout(*load(raw), raw.points)
    for key in ("target", "tri", "points"):
        assert np.array_equal(getattr(new, key), old[key]), key
    assert list(new.box) == old["box"].tolist()
    assert ml_root().exists()

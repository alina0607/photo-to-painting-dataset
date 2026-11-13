import numpy as np

from photo2painting.ingest import load, measure
from photo2painting.rules import MODELS, Thresholds, ear_reason, reasons
from tests.reference import real_cat, reference
from tests.synthetic import synthetic_cat


def test_a_clean_drawn_face_is_kept_everywhere():
    photo, mask, p = synthetic_cat()
    r = reasons(measure(photo, mask, p), p, mask)
    assert set(r) == set(MODELS)
    assert all(v == "" for v in r.values())


def test_thresholds_give_the_reasons():
    photo, mask, p = synthetic_cat()
    m = measure(photo, mask, p)
    strict = Thresholds(min_face=1.1, min_contrast=1.0, min_eye_px=100.0)
    r = reasons(m, p, mask, strict)
    assert r["whole"] == "face cut; too dark"
    assert r["eyes"] == "eyes too small"
    assert r["fur"] == "face cut; too dark; eyes too small"


def test_an_ear_outside_the_mask_is_hidden_or_cut():
    _, mask, p = synthetic_cat()
    assert ear_reason(p, (3, 4, 5), mask) == ""
    assert ear_reason(p, (3, 4, 5), np.zeros_like(mask)) == "ear hidden or cut"


def test_reasons_match_the_research_code():
    rx = reference("review_export")
    raw = real_cat()
    photo, mask = load(raw)
    m = measure(photo, mask, raw.points)

    class Args:
        min_face, min_tip, min_contrast, min_eye_px = 0.97, 0.5, 0.035, 40.0

    assert reasons(m, raw.points, mask) == rx.offers(m.__dict__, raw.points, mask, Args)

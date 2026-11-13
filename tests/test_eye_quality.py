import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from photo2painting.eye_quality import (
    EyeMeasure,
    bootstrap,
    calibration,
    crop_features,
    crops,
    cut,
    cv_scores,
    eye_measures,
    level,
    measured_row,
    nested_threshold,
    pick_threshold,
    standardise,
)
from tests.reference import ml_root, reference
from tests.synthetic import synthetic_cat


def separable(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two overlapping Gaussian classes in 4 dimensions."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2 == 0
    x = rng.normal(size=(n, 4)) + 1.2 * y[:, None]
    return x, y


def test_level_puts_the_eyes_on_one_row():
    photo, _, p = synthetic_cat()
    q = p.copy()
    q[1] = [175.0, 160.0]  # tilt the right eye down
    _, _, d, eyes = level(np.asarray(photo), q)
    assert eyes[0][1] == pytest.approx(eyes[1][1])
    assert np.linalg.norm(eyes[1] - eyes[0]) == pytest.approx(d)


def test_crops_have_the_expected_shapes():
    photo, _, p = synthetic_cat()
    both, left, right = crops(np.asarray(photo), p)
    assert both.size == (120, 54)
    assert left.size == right.size


def test_crop_features_ignore_which_eye_is_which():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(3, 512))
    f = crop_features(e)
    assert f.shape == (1536,)
    assert np.array_equal(f, crop_features(e[[0, 2, 1]]))


def test_measured_row_and_standardise():
    def eye(px: int, lab: tuple[float, float, float], score: float) -> EyeMeasure:
        return EyeMeasure(px, True, score, score, lab)

    row = measured_row([eye(100, (50, 0, 10), 20.0), eye(80, (40, 0, 10), 30.0)], 20.0)
    assert row.tolist() == pytest.approx([0.2, 0.8, 10.0, 10.0, 20.0])
    z = standardise(np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]]))
    assert z[:, 0].tolist() == pytest.approx([-1.2247, 0.0, 1.2247], abs=1e-4)
    assert z[1, 1] == 0
    assert z[:, 2].tolist() == [0, 1, 0]


def test_bootstrap_interval_contains_the_value():
    x, y = separable()
    p = cv_scores(x, y)
    ci = bootstrap(y, p, "auc", n=300)
    assert ci.low <= ci.value <= ci.high
    assert ci.value == pytest.approx(roc_auc_score(y, p))


def test_cut_and_threshold():
    y = np.array([True, True, False, False])
    p = np.array([0.9, 0.6, 0.7, 0.1])
    c = cut(y, p, 0.5)
    assert (c.removed, c.unclear_caught, c.clear_lost) == (3, 2, 1)
    assert c.wrongly_removed_share == pytest.approx(1 / 3)
    assert pick_threshold(y, p, 0.0, [0.5, 0.65, 0.8]) == 0.8
    assert pick_threshold(y, p, 0.4, [0.5, 0.65, 0.8]) == 0.5


def test_nested_threshold_chooses_per_fold():
    x, y = separable()
    chosen, pooled = nested_threshold(x, y, 0.1, [0.5, 0.6, 0.7, 0.8, 0.9])
    assert len(chosen) == 5
    assert pooled.removed > 0


def test_calibration_counts_every_cat():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200)
    y = rng.uniform(size=200) < p
    assert sum(n for _, _, n in calibration(y, p)) == 200


# ---------------------------------------------- against the research code
REAL = "afhq__pixabay_cat_001129"


def real_cutout_and_plan() -> tuple[np.ndarray, np.ndarray, dict]:
    ec, ps = reference("eye_clip"), reference("paint_stages")
    z = np.load(ec.DIR / "clip_eyes.npz")
    i = [str(n) for n in z["names"]].index(REAL)
    shard = np.load(str(z["shards"][i]), allow_pickle=True)
    k = int(z["index"][i])
    plan = np.load(next(ps.PLAN.glob(f"*/shard_*/{REAL}.npz")), allow_pickle=True)
    return (
        shard["target"][k],
        shard["points"][k],
        {"eyes": plan["eyes"], "pupils": plan["pupils"]},
    )


def test_crops_match_the_research_code():
    ec = reference("eye_clip")
    img, p, _ = real_cutout_and_plan()
    for a, b in zip(crops(img, p), ec.crops(img, p), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_eye_measures_match_the_research_code():
    pa = reference("pupil_audit")
    img, _, plan = real_cutout_and_plan()
    old = pa.eye_scores(img / 255.0, plan)
    new = eye_measures(img / 255.0, plan["eyes"], plan["pupils"])
    for o, m in zip(old, new, strict=True):
        assert (m.eye_px, m.pupil_ok) == (o["eye_px"], o["pupil_ok"])
        assert m.contrast == pytest.approx(o["contrast"])
        assert m.iris_lab == pytest.approx((o["iris_L"], o["iris_a"], o["iris_b"]))


def test_cross_validation_matches_the_research_code():
    ec = reference("eye_clip")
    z = np.load(ec.DIR / "clip_eyes.npz")
    names = [str(n) for n in z["names"]]
    pos = {n: i for i, n in enumerate(names)}
    owner: dict[str, str] = {}
    for f in ("labels.jsonl", "labels_r2.jsonl", "labels_r3.jsonl"):
        for line in (ec.DIR / f).read_text().splitlines():
            r = json.loads(line)
            owner[r["name"]] = r["answer"]
    lab = {n: a for n, a in owner.items() if a in ("clear", "unclear") and n in pos}
    x = np.c_[z["feats"], ec.measured(names)][[pos[n] for n in lab]]
    y = np.array([lab[n] == "unclear" for n in lab])
    p = cv_scores(x, y)
    # notebook 12 of the research repo: 435 labels, accuracy 0.795, AUC 0.890
    assert len(y) == 435
    assert roc_auc_score(y, p) == pytest.approx(0.890, abs=5e-4)
    assert ml_root().exists()

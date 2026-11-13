"""Which cats' eyes are too poor to paint from: a learned filter.

A painting model learns eye colour, pupils and highlights from the eyes it is shown.
Photos whose eyes are grainy, whose pupil cannot be told from the iris, with one eye
hidden by the angle, with flash eye-shine, in greyscale, squinted very small, or with
the two eyes differing because of a shadow, teach it wrong colours. Simple measures
cannot see most of these, so a person labels a sample of cats "clear" or "unclear" and
a classifier learns the rest:

- features: CLIP image embeddings of three crops (both eyes, and each eye on its own,
  taken from the photo turned so the eyes are level), plus six measured numbers from
  the sketch plan's eye outlines and pupils (`eye_measures`);
- model: logistic regression, P(unclear);
- evaluation: stratified k-fold cross-validation (every cat's score comes from a model
  that did not see it), with bootstrap confidence intervals, and a threshold chosen
  inside each training fold (nested), so the reported cost of the threshold is honest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as Polygon
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

C_REG = 30.0  # the logistic regression's inverse regularisation (best of 5-fold CV)
EXCLUDE_ABOVE = 0.8  # the threshold in use: P(unclear) above it leaves the dataset
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])
CLIP_SIDE = 224
BOTH_W, BOTH_H, ONE = 1.0, 0.45, 0.32  # crop half-sizes, eye distances
MIN_EYE_PX, MIN_RADIUS, MIN_SHARE = 12, 0.75, 0.03  # a pupil the plan really found
MEASURED = (
    "smaller_eye_size",
    "size_ratio",
    "lightness_gap",
    "colour_gap",
    "pupil_contrast",
)


# ------------------------------------------------------------------ crops
def level(
    img: np.ndarray, points: np.ndarray
) -> tuple[Image.Image, np.ndarray, float, list[np.ndarray]]:
    """The photo turned about the eyes' midpoint so the eyes are level.

    Returns:
        The turned image, the eyes' midpoint, the eye distance and the two eyes'
        positions after turning.
    """
    c, v = points[:2].mean(0), points[1] - points[0]
    d = float(np.linalg.norm(v))
    im = Image.fromarray(img).rotate(
        float(np.degrees(np.arctan2(v[1], v[0]))),
        center=(float(c[0]), float(c[1])),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )
    return im, c, d, [c - [d / 2, 0], c + [d / 2, 0]]


def crops(img: np.ndarray, points: np.ndarray) -> list[Image.Image]:
    """Three crops of the levelled photo: both eyes, the image's left eye, the right."""
    im, c, d, eyes = level(img, points)
    out = [im.crop(_box(c, BOTH_W * d, BOTH_H * d))]
    out += [im.crop(_box(q, ONE * d, ONE * d)) for q in eyes]
    return out


def _box(centre: np.ndarray, half_w: float, half_h: float) -> tuple[int, int, int, int]:
    x, y = float(centre[0]), float(centre[1])
    return int(x - half_w), int(y - half_h), int(x + half_w), int(y + half_h)


def clip_input(images: Sequence[Image.Image]) -> np.ndarray:
    """Images as CLIP's input: 224 px, scaled to 0-1, normalised, channels first."""
    return np.stack(
        [
            (
                (
                    np.asarray(
                        im.convert("RGB").resize(
                            (CLIP_SIDE, CLIP_SIDE), Image.Resampling.BICUBIC
                        )
                    )
                    / 255.0
                    - CLIP_MEAN
                )
                / CLIP_STD
            ).transpose(2, 0, 1)
            for im in images
        ]
    )


def crop_features(embeddings: np.ndarray) -> np.ndarray:
    """One cat's feature from its three unit-length crop embeddings (3, 512): both eyes,
    then the element-wise min and max of the two single eyes (so left and right do not
    matter), 1536 numbers."""
    both, a, b = embeddings
    out: np.ndarray = np.concatenate([both, np.minimum(a, b), np.maximum(a, b)])
    return out


# ------------------------------------------------------------------ measured features
@dataclass(frozen=True)
class EyeMeasure:
    """One eye measured inside the sketch plan's outline (256 px cut-out).

    Attributes:
        eye_px: the outline's area in pixels.
        pupil_ok: the plan found a real pupil (not a sliver).
        contrast: the iris's median lightness (L*) minus the pupil's.
        score: ``contrast`` when ``pupil_ok``, else 0.
        iris_lab: the iris's median L*, a*, b*.
    """

    eye_px: int
    pupil_ok: bool
    contrast: float
    score: float
    iris_lab: tuple[float, float, float]


def eye_measures(
    photo: np.ndarray, eyes: Sequence[np.ndarray], pupils: Sequence[np.ndarray]
) -> list[EyeMeasure]:
    """Measure each eye of a cut-out photo (RGB 0-1) inside the plan's outlines.

    Args:
        photo: the 256 px cut-out, float RGB in 0-1.
        eyes: each eye's outline (m, 2) on the 256 grid.
        pupils: each pupil's ellipse: centre x, y, the two radii and the angle.
    """
    side = photo.shape[0]
    lab = rgb2lab(photo)
    lightness = lab[..., 0]
    yy, xx = np.mgrid[:side, :side] + 0.5
    q = np.c_[xx.ravel(), yy.ravel()]
    out = []
    for outline, p in zip(eyes, pupils, strict=False):
        eye = Polygon(outline).contains_points(q).reshape(side, side)
        nan = float("nan")
        m = EyeMeasure(int(eye.sum()), False, nan, 0.0, (nan, nan, nan))
        if eye.sum() >= MIN_EYE_PX and np.all(np.isfinite(p[:5])):
            cx, cy, ra, rb, a = p[:5]
            u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
            v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
            pupil = ((u / max(ra, 0.5)) ** 2 + (v / max(rb, 0.5)) ** 2 <= 1) & eye
            iris = eye & ~ndimage.binary_dilation(pupil, iterations=1)
            if pupil.sum() and iris.sum() > 5:
                # leave out the iris's lightest 10%: the highlights
                iris &= lightness <= np.percentile(lightness[iris], 90)
                contrast = float(
                    np.median(lightness[iris]) - np.median(lightness[pupil])
                )
                ok = bool(
                    min(ra, rb) >= MIN_RADIUS and pupil.sum() / eye.sum() >= MIN_SHARE
                )
                med = [float(np.median(lab[..., c][iris])) for c in range(3)]
                iris_lab = (med[0], med[1], med[2])
                score = contrast if ok else 0.0
                m = EyeMeasure(int(eye.sum()), ok, contrast, score, iris_lab)
        out.append(m)
    return out


def measured_row(two: Sequence[EyeMeasure], eye_dist: float) -> np.ndarray:
    """The five measured numbers of a cat (see `MEASURED`), unscaled.

    The smaller eye's area over the eye distance squared (squinting), the smaller over
    the larger (one eye hidden), the irises' lightness and colour gaps (a shadow over
    one eye), and the worse eye's pupil score.
    """
    px = sorted(e.eye_px for e in two)
    lab = np.array([e.iris_lab for e in two])
    return np.array(
        [
            px[0] / max(eye_dist, 1) ** 2,
            px[0] / max(px[1], 1),
            abs(lab[0, 0] - lab[1, 0]),
            float(np.linalg.norm(lab[0] - lab[1])),
            min(e.score for e in two),
        ]
    )


def standardise(rows: np.ndarray) -> np.ndarray:
    """Measured rows (n, 5, NaN where missing) standardised over all cats, missing
    values set to 0, and a sixth column flagging the cats with a missing value."""
    missing = np.isnan(rows).any(1)
    z = (rows - np.nanmean(rows, 0)) / np.nanstd(rows, 0)
    z[np.isnan(z)] = 0
    out: np.ndarray = np.column_stack([z, missing])
    return out


# ------------------------------------------------------------------ evaluation
def model() -> LogisticRegression:
    """The classifier."""
    return LogisticRegression(C=C_REG, max_iter=5000)


def cv_scores(
    x: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0
) -> np.ndarray:
    """Out-of-fold P(unclear) for every labelled cat (stratified k-fold)."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    p: np.ndarray = cross_val_predict(model(), x, y, cv=cv, method="predict_proba")[
        :, 1
    ]
    return p


@dataclass(frozen=True)
class Interval:
    """A statistic with a percentile bootstrap interval."""

    value: float
    low: float
    high: float


def bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    stat: str,
    n: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """A bootstrap interval for "auc" or "accuracy" (at 0.5) of out-of-fold scores.

    The labelled cats are resampled with replacement ``n`` times; the interval is the
    middle ``level`` of the statistic over the resamples.
    """

    def value(yy: np.ndarray, pp: np.ndarray) -> float:
        if stat == "auc":
            return float(roc_auc_score(yy, pp))
        return float(((pp > 0.5) == yy).mean())

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if stat == "auc" and len(set(y[k])) < 2:
            continue
        out.append(value(y[k], p[k]))
    lo, hi = np.quantile(out, [(1 - level) / 2, (1 + level) / 2])
    return Interval(value(y, p), float(lo), float(hi))


@dataclass(frozen=True)
class Cut:
    """What removing every cat scored above a threshold does to the labelled cats.

    Attributes:
        threshold: the cut.
        removed: cats removed.
        wrongly_removed_share: of the removed, the share labelled clear.
        unclear_caught: labelled-unclear cats removed.
        clear_lost: labelled-clear cats removed.
    """

    threshold: float
    removed: int
    wrongly_removed_share: float
    unclear_caught: int
    clear_lost: int


def cut(y: np.ndarray, p: np.ndarray, threshold: float) -> Cut:
    """The effect of removing the cats with P(unclear) above ``threshold``."""
    drop = p > threshold
    return Cut(
        threshold=threshold,
        removed=int(drop.sum()),
        wrongly_removed_share=float((~y[drop]).mean()) if drop.any() else 0.0,
        unclear_caught=int((y & drop).sum()),
        clear_lost=int((~y & drop).sum()),
    )


def pick_threshold(
    y: np.ndarray, p: np.ndarray, max_wrong: float, grid: Sequence[float]
) -> float:
    """The lowest threshold whose removed cats are at most ``max_wrong`` clear ones
    (so the most unclear cats are caught at that cost); the highest of the grid when
    none qualifies."""
    for t in sorted(grid):
        c = cut(y, p, t)
        if c.removed and c.wrongly_removed_share <= max_wrong:
            return float(t)
    return float(max(grid))


def nested_threshold(
    x: np.ndarray,
    y: np.ndarray,
    max_wrong: float,
    grid: Sequence[float],
    folds: int = 5,
    seed: int = 0,
) -> tuple[list[float], Cut]:
    """Choose the threshold inside each outer fold and measure it on that fold.

    For every outer fold, the threshold is picked (`pick_threshold`) from inner
    out-of-fold scores of the other folds, then a model fitted on those folds scores
    the held fold and the cut is applied there. Nothing about the held fold informs its
    threshold.

    Returns:
        The threshold each fold chose, and the pooled cut over all held folds.
    """
    outer = StratifiedKFold(folds, shuffle=True, random_state=seed)
    chosen, held_p, held_t = [], np.zeros(len(y)), np.zeros(len(y))
    for train, test in outer.split(x, y):
        t = pick_threshold(
            y[train], cv_scores(x[train], y[train], folds, seed), max_wrong, grid
        )
        chosen.append(t)
        held_p[test] = model().fit(x[train], y[train]).predict_proba(x[test])[:, 1]
        held_t[test] = t
    drop = held_p > held_t
    pooled = Cut(
        threshold=float("nan"),
        removed=int(drop.sum()),
        wrongly_removed_share=float((~y[drop]).mean()) if drop.any() else 0.0,
        unclear_caught=int((y & drop).sum()),
        clear_lost=int((~y & drop).sum()),
    )
    return chosen, pooled


def calibration(
    y: np.ndarray, p: np.ndarray, bins: int = 10
) -> list[tuple[float, float, int]]:
    """Reliability: for each score bin, the mean score, the share truly unclear and the
    count (empty bins left out)."""
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [
        (float(p[k == b].mean()), float(y[k == b].mean()), int((k == b).sum()))
        for b in range(bins)
        if (k == b).any()
    ]

"""The first, cheapest filter: thresholds on measured numbers.

Each photo can teach several models: the whole-face model (it needs the whole head),
an ear model per ear, the eye model, the nose-and-mouth model and the fur model. For
each, `reasons` says why the photo is left out, or "" when it is kept. The thresholds
are set at the extremes on purpose (for example, "too dark" was chosen by looking at
the darkest faces that still show their eyes): the rest is left to a person's review
and to the learned filters that follow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ingest import EARS, Measures

MIN_EAR_PX = 20.0  # an ear's base at least this wide to be worth a crop
EAR_MIDDLE = 0.35  # where an ear's mask is read: from its base's middle to its tip
MODELS = ("whole", "ear_1", "ear_2", "eyes", "nose_mouth", "fur")


@dataclass(frozen=True)
class Thresholds:
    """The rules' thresholds.

    Attributes:
        min_face: the face hull's share inside the mask; below it, "face cut".
        min_tip: the mask's value near the ear tips; below it, "ear cut".
        min_contrast: the luminance spread near the eyes; below it, "too dark".
        min_eye_px: the eye distance in pixels; below it, "eyes too small".
    """

    min_face: float = 0.97
    min_tip: float = 0.5
    min_contrast: float = 0.035
    min_eye_px: float = 40.0


def ear_reason(points: np.ndarray, ear: tuple[int, int, int], mask: np.ndarray) -> str:
    """Why an ear cannot teach the ear model, or "" when it can.

    Its base must be inside the photo and wide enough, and its middle inside the mask
    (not hidden or cut away). A cut-off tip is fine.
    """
    h, w = mask.shape
    b0, tip, b1 = (points[k] for k in ear)
    if not all(0 <= x < w and 0 <= y < h for x, y in (b0, b1)):
        return "ear base outside photo"
    if np.linalg.norm(b0 - b1) < MIN_EAR_PX:
        return "ear too small"
    x, y = (b0 + b1) / 2 + EAR_MIDDLE * (tip - (b0 + b1) / 2)
    if not (0 <= x < w and 0 <= y < h) or mask[int(y), int(x)] < 0.5:
        return "ear hidden or cut"
    return ""


def reasons(
    m: Measures, points: np.ndarray, mask: np.ndarray, t: Thresholds | None = None
) -> dict[str, str]:
    """For each model in `MODELS`, why the photo is left out ("" = kept).

    Args:
        m: the photo's measures (`photo2painting.ingest.measure`).
        points: the nine face points in the photo's pixels.
        mask: the subject mask, 0-1, the photo's size.
        t: the thresholds; the defaults when None.

    Returns:
        A dict from model name to the reasons, joined by "; ".
    """
    t = t or Thresholds()
    face = [
        k
        for k, bad in (
            ("face cut", m.face_cover < t.min_face),
            ("too dark", m.eye_contrast < t.min_contrast),
        )
        if bad
    ]
    small = m.eye_dist_px < t.min_eye_px
    ears = [
        ("ear outside photo", not m.tips_in_photo),
        ("ear cut", m.tips_in_photo and m.tip_cover < t.min_tip),
    ]
    h, w = mask.shape
    mouth_in = 0 <= points[2][0] < w and 0 <= points[2][1] < h
    return {
        "whole": "; ".join(face + [k for k, bad in ears if bad]),
        "ear_1": ear_reason(points, EARS[0], mask),
        "ear_2": ear_reason(points, EARS[1], mask),
        "eyes": "eyes too small" if small else "",
        "nose_mouth": "eyes too small"
        if small
        else ("" if mouth_in else "mouth outside photo"),
        "fur": "; ".join(face + (["eyes too small"] if small else [])),
    }

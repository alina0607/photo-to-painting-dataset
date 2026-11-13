"""From a raw photo to the painter's input: the cut-out head on a 256 px grid.

Every photo comes with a subject mask (each pixel 0-1, how sure the mask is that it
is the cat) and nine face points in the CAT dataset's order: the left eye, the right
eye, the mouth, then each ear's outer base, tip and inner base ("left" and "right" are
the image's). AFHQ ships no points, so its points come from a landmark model; the CAT
dataset's are hand-placed.

The head is cut out in the face's own frame (the "straightened" head): origin at the
eyes' midpoint, x along the line between the eyes, y towards the mouth, one unit per
eye distance. In that frame every cat's eyes sit at (-0.5, 0) and (0.5, 0), so one
shape fits every head however it is tilted: a rounded face (a superellipse from the
highest ear base to below the mouth) plus a generous triangle per ear. The photo is
never rotated; the shape is drawn in the rotated frame instead.

The cut-out (`cutout`) is that head inside the mask, in a square around it, resized to
256 px, with a tri map: 0 background, 1 head, 2 uncertain (the mask says cat but the
pixel is outside the head shape, such as the chest below the chin, or outside the
photo). `brushgpt.cat.load_inputs` reads exactly these fields.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from brushgpt.cat import face_frame
from matplotlib.path import Path as Polygon
from PIL import Image
from scipy.spatial import ConvexHull

SIDE = 256  # the cut-out's grid
EARS = ((3, 4, 5), (6, 7, 8))  # each ear's face points: outer base, tip, inner base
TIPS = (4, 7)  # the ear tips
FACE = (0, 1, 2, 3, 5, 6, 8)  # eyes, mouth and ear bases: the points the face hull uses
# The head shape, in the face's frame (eye distances).
HEAD_CHIN, HEAD_SIDE, HEAD_POWER = 0.45, 0.10, 2.2  # below the mouth, cheeks, roundness
EAR_SCALE, EAR_PAD = 1.3, 0.08  # how generous the ear triangles are
CROP_MARGIN = 1.08  # the square around the head, this much wider than the head
# Measures.
TIP_SLACK = 0.10  # eye distances an ear tip may lie outside the photo
TIP_INWARD = 0.15  # an ear's mask is read this far from its tip towards its base
EYE_NEAR = 0.35  # eye distances around each eye where the contrast is measured
LUMA = np.array([0.299, 0.587, 0.114])


@dataclass(frozen=True)
class RawCat:
    """One raw photo and what comes with it.

    Attributes:
        source: "afhq" or "cat" (the CAT dataset).
        cat_id: the photo's name within its source.
        photo: path to the photo file.
        mask: path to the subject mask (a grey image, white = cat).
        points: the nine face points in the photo's pixels, shape (9, 2).
    """

    source: str
    cat_id: str
    photo: Path
    mask: Path
    points: np.ndarray

    @property
    def name(self) -> str:
        """The name used everywhere downstream: ``<source>__<cat_id>``."""
        return f"{self.source}__{self.cat_id}"


def afhq_cats(labels: Path, photos: Path, masks: Path) -> Iterator[RawCat]:
    """The AFHQ cats with predicted face points.

    Args:
        labels: an .npz with ``ids`` (paths relative to ``photos``) and ``points``.
        photos: the AFHQ root folder.
        masks: the folder of subject masks, ``<stem>_mask1.png``.

    Yields:
        One RawCat per photo, in the order of the labels file.
    """
    z = np.load(labels)
    for rel, p in zip([str(i) for i in z["ids"]], z["points"], strict=True):
        stem = Path(rel).stem
        yield RawCat(
            "afhq",
            stem,
            photos / rel,
            masks / f"{stem}_mask1.png",
            np.asarray(p, float),
        )


def cat_dataset_cats(zips: list[Path], photos: Path, masks: Path) -> Iterator[RawCat]:
    """The CAT dataset's cats that have a subject mask, with their hand-placed points.

    Args:
        zips: the dataset's zip files; each photo has a ``.jpg.cat`` text file next to
            it: the number of points (9), then 18 numbers, x and y of each point.
        photos: where the photos were extracted, named ``<zip>__<folder>__<file>.jpg``.
        masks: the folder of subject masks, ``<stem>_mask1.png``.

    Yields:
        One RawCat per photo with a mask, sorted by name.
    """
    members: dict[str, tuple[str, zipfile.ZipFile]] = {}
    for zp in zips:
        zf = zipfile.ZipFile(zp)
        members.update(
            {n: (zp.stem, zf) for n in zf.namelist() if n.endswith(".jpg.cat")}
        )
    for member, (zip_name, zf) in sorted(members.items()):
        folder, file = member.split("/")[-2], member.split("/")[-1][: -len(".jpg.cat")]
        stem = f"{zip_name}__{folder}__{file}"
        mask = masks / f"{stem}_mask1.png"
        if mask.exists():
            yield RawCat(
                "cat",
                stem,
                photos / f"{stem}.jpg",
                mask,
                parse_cat_points(zf.read(member).decode()),
            )


def parse_cat_points(text: str) -> np.ndarray:
    """The nine points of a CAT dataset ``.cat`` file, shape (9, 2)."""
    nums = [float(x) for x in text.split()]
    if int(nums[0]) != 9:
        raise ValueError(f"expected 9 points, got {int(nums[0])}")
    return np.array(nums[1:19]).reshape(9, 2)


def load(raw: RawCat) -> tuple[Image.Image, np.ndarray]:
    """The photo (RGB) and its subject mask resized to the photo, 0-1."""
    photo = Image.open(raw.photo).convert("RGB")
    mask = (
        Image.open(raw.mask).convert("L").resize(photo.size, Image.Resampling.BILINEAR)
    )
    return photo, np.asarray(mask, np.float32) / 255


# ------------------------------------------------------------------ measures
def luminance(img: np.ndarray) -> np.ndarray:
    """Luminance (ITU-R 601 weights) of an RGB image in 0-1."""
    out: np.ndarray = img[..., :3] @ LUMA
    return out


def face_cover(mask: np.ndarray, points: np.ndarray) -> float:
    """The share of the face that the subject mask calls cat.

    The face is the convex hull of the eyes, the mouth and the four ear bases (the ear
    tips are left out: between two tips the hull is background). A mask that misses
    part of the face would put white paper into the painting's face.
    """
    q = points[list(FACE)]
    hull = Polygon(q[ConvexHull(q).vertices])
    x0, y0 = np.floor(q.min(0)).astype(int).clip(0)
    x1, y1 = np.ceil(q.max(0)).astype(int)
    x1, y1 = min(x1, mask.shape[1]), min(y1, mask.shape[0])
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
    inside = hull.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(yy.shape)
    return float(mask[y0:y1, x0:x1][inside].mean()) if inside.any() else 1.0


@dataclass(frozen=True)
class Measures:
    """The numbers the quality rules read (see `photo2painting.rules`).

    Attributes:
        eye_dist_px: distance between the eyes in the photo's pixels.
        face_cover: share of the face hull inside the mask (`face_cover`).
        tip_cover: the mask's value near each ear tip, the smaller of the two
            (0 when a tip is outside the photo).
        tips_in_photo: both ear tips within `TIP_SLACK` eye distances of the photo.
        eye_contrast: standard deviation of the luminance near the eyes; a photo too
            dark to see the eyes has almost none.
        eye_lum: mean luminance near the eyes.
        photo_w: the photo's width in pixels.
        photo_h: the photo's height in pixels.
    """

    eye_dist_px: float
    face_cover: float
    tip_cover: float
    tips_in_photo: bool
    eye_contrast: float
    eye_lum: float
    photo_w: int
    photo_h: int


def measure(photo: Image.Image, mask: np.ndarray, points: np.ndarray) -> Measures:
    """Measure one photo for the quality rules."""
    w, h = photo.size
    d = float(np.linalg.norm(points[0] - points[1]))
    img = np.asarray(photo, np.float32) / 255
    yy, xx = np.mgrid[:h, :w] + 0.5
    near = np.min([np.hypot(xx - x, yy - y) for x, y in points[:2]], 0) < EYE_NEAR * d
    slack = TIP_SLACK * d
    tips_in_photo = all(
        -slack <= x <= w + slack and -slack <= y <= h + slack
        for x, y in points[list(TIPS)]
    )
    inner = [
        points[t] + TIP_INWARD * ((points[b0] + points[b1]) / 2 - points[t])
        for t, b0, b1 in ((4, 3, 5), (7, 6, 8))
    ]
    tip_cover = 0.0
    if tips_in_photo and all(0 <= x < w and 0 <= y < h for x, y in inner):
        tip_cover = min(float(mask[int(y), int(x)]) for x, y in inner)
    lum = luminance(img)[near]
    return Measures(
        eye_dist_px=round(d, 1),
        face_cover=round(face_cover(mask, points), 4),
        tip_cover=round(tip_cover, 3),
        tips_in_photo=tips_in_photo,
        eye_contrast=round(float(lum.std()), 4),
        eye_lum=round(float(lum.mean()), 4),
        photo_w=w,
        photo_h=h,
    )


# ------------------------------------------------------------------ the head
def head_shape(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The head alone as a boolean mask of ``shape`` (height, width).

    In the face's frame: a superellipse (power `HEAD_POWER`) from the highest ear base
    down to `HEAD_CHIN` below the mouth, as wide as the widest ear base or half an eye
    distance plus `HEAD_SIDE`; plus each ear's triangle scaled by `EAR_SCALE` about its
    centre and widened by `EAR_PAD`. Generous on purpose: the ear's real edge comes
    from the subject mask it is intersected with.
    """
    h, w = shape
    d, mid, ex, ey, p = face_frame(points)
    bases = p[[3, 5, 6, 8]]
    a = max(float(np.abs(bases[:, 0]).max()), 1.0) + HEAD_SIDE
    top, bottom = float(bases[:, 1].min()), float(p[2, 1]) + HEAD_CHIN
    cy, b = (top + bottom) / 2, (bottom - top) / 2
    yy, xx = np.mgrid[:h, :w] + 0.5
    rel = np.stack([xx - mid[0], yy - mid[1]], -1)
    qx, qy = rel @ ex / d, rel @ ey / d
    head: np.ndarray = (
        np.abs(qx / a) ** HEAD_POWER + np.abs((qy - cy) / b) ** HEAD_POWER <= 1
    )
    q = np.stack([qx, qy], -1).reshape(-1, 2)
    for ear in EARS:
        t = p[list(ear)]
        c = t.mean(0)
        v = t - c
        t = c + v * EAR_SCALE + v / np.linalg.norm(v, axis=1, keepdims=True) * EAR_PAD
        head |= Polygon(t).contains_points(q).reshape(h, w)
    return head


def square(centre: np.ndarray, side: float) -> tuple[int, int, int, int]:
    """A square box (left, top, right, bottom) of ``side`` pixels about ``centre``."""
    x0, y0 = centre[0] - side / 2, centre[1] - side / 2
    return (
        round(x0),
        round(y0),
        round(centre[0] + side / 2),
        round(centre[1] + side / 2),
    )


def head_box(head: np.ndarray) -> tuple[int, int, int, int]:
    """The square around a head mask, `CROP_MARGIN` times its larger side."""
    ys, xs = np.nonzero(head)
    lo, hi = (
        np.array([xs.min(), ys.min()], float),
        np.array([xs.max(), ys.max()], float) + 1,
    )
    return square((lo + hi) / 2, CROP_MARGIN * float((hi - lo).max()))


@dataclass(frozen=True)
class Cutout:
    """The painter's input for one cat (the fields `brushgpt.cat.load_inputs` reads).

    Attributes:
        target: the head on white, (256, 256, 3) uint8.
        tri: (256, 256) uint8, 0 background, 1 head, 2 uncertain.
        points: the nine face points on the 256 grid, float32.
        box: the square in the photo's pixels (left, top, right, bottom).
    """

    target: np.ndarray
    tri: np.ndarray
    points: np.ndarray
    box: tuple[int, int, int, int]


def cutout(photo: Image.Image, mask: np.ndarray, points: np.ndarray) -> Cutout:
    """Cut the head out of a photo onto the 256 px grid (see the module docstring)."""
    w, h = photo.size
    box = head_box(head_shape(points, (h, w)))
    side = box[2] - box[0]

    def cut(arr: np.ndarray, fill: int) -> np.ndarray:
        img = Image.fromarray(arr)
        sheet = Image.new(
            img.mode, (side, side), fill if img.mode == "L" else (fill,) * 3
        )
        sheet.paste(img, (-box[0], -box[1]))
        resample = (
            Image.Resampling.LANCZOS if img.mode == "RGB" else Image.Resampling.BILINEAR
        )
        return np.asarray(sheet.resize((SIDE, SIDE), resample))

    in_photo = cut(np.full((h, w), 255, np.uint8), 0) > 127
    cat_mask = cut((mask * 255).astype(np.uint8), 0) > 127
    in_head = cut((head_shape(points, (h, w)) * 255).astype(np.uint8), 0) > 127
    target = cut(np.asarray(photo), 255).copy()
    background = in_photo & ~cat_mask
    target[background] = 255
    tri = np.full((SIDE, SIDE), 2, np.uint8)
    tri[background] = 0
    tri[cat_mask & in_head & in_photo] = 1
    pts = ((points - np.array(box[:2], float)) * (SIDE / side)).astype(np.float32)
    return Cutout(target=target, tri=tri, points=pts, box=box)

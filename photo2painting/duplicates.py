"""Near-duplicate photos, and a split that keeps each duplicate group on one side.

AFHQ holds some photos twice: re-cropped, re-compressed or re-coloured. If one copy is
in training and the other in validation or test, the model is evaluated on a photo it
has seen, and the evaluation is too kind. All-pairs comparison of 5,558 photos is 15
million pairs, so it runs in two steps:

1. Cheap scores for every pair (`cheap_scores`): the correlation of 16 x 16 grey
   thumbnails (also against the mirrored photo) and the Hamming distance of a 255-bit
   DCT perceptual hash. Pairs above loose thresholds, or with a high CLIP cosine,
   become candidates (`candidates`).
2. A precise score for each candidate (`aligned_similarity`): the second photo is
   scaled and shifted onto the first (and mirrored, when that fits better), and the
   overlap is compared by normalised cross-correlation (NCC), SSIM and mean absolute
   RGB difference.

Pairs above an NCC threshold are the same photo; pairs just below it are looked at by
eye. `groups` joins the pairs into groups of copies and `group_split` assigns whole
groups to one split.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import shift as nd_shift
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

GREY = np.array([0.299, 0.587, 0.114], np.float32)
# step 1
CHEAP_SIDE, CORR_SIDE, HASH_SIDE = 64, 16, 16  # px of the grey thumbnail, the
# correlation's grid and the DCT block the hash keeps
CORR_MIN, HASH_MAX = 0.85, 70  # candidate: correlation at least, or Hamming at most
CLIP_MIN, CLIP_NEAREST_MIN = 0.97, 0.95  # candidate: CLIP cosine, or nearest neighbour
# step 2
PRECISE_SIDE = 128  # px of the RGB thumbnail aligned in step 2
# 29 scales evenly spaced in log from 0.70 to 1.40. The grid has no 1.0 (its middle
# is 0.99): an exact copy still aligns at NCC 0.996, but a copy re-cropped by more
# than about 5% can fall between two scales and score below `SAME_PHOTO_NCC`.
SCALES = np.round(np.exp(np.linspace(np.log(0.70), np.log(1.40), 29)), 4)
MIN_OVERLAP = 0.35  # an alignment must keep this share of the grid
EDGE = 4  # px left out at the overlap's edges (interpolation)
SAME_PHOTO_NCC = 0.96  # aligned NCC at least this: the same photo


def thumbnails(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    """The 64 px grey and 128 px RGB thumbnails (uint8) of each photo."""
    grey, rgb = [], []
    for p in paths:
        im = Image.open(p).convert("RGB")
        small = im.convert("L").resize(
            (CHEAP_SIDE, CHEAP_SIDE), Image.Resampling.BILINEAR
        )
        grey.append(np.asarray(small, np.uint8))
        size = (PRECISE_SIDE, PRECISE_SIDE)
        rgb.append(np.asarray(im.resize(size, Image.Resampling.BILINEAR), np.uint8))
    return np.stack(grey), np.stack(rgb)


def _unit_rows(a: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1).astype(np.float64)
    a = a - a.mean(1, keepdims=True)
    out: np.ndarray = a / np.linalg.norm(a, axis=1, keepdims=True)
    return out


@dataclass(frozen=True)
class CheapScores:
    """Step 1's scores for every pair i < j (flat arrays in `np.triu_indices` order).

    Attributes:
        i: the first photo's index.
        j: the second photo's index.
        corr: the 16 x 16 grey correlation.
        corr_mirror: the same against the second photo mirrored left-right.
        hamming: the perceptual hashes' Hamming distance (0 = identical hash).
    """

    i: np.ndarray
    j: np.ndarray
    corr: np.ndarray
    corr_mirror: np.ndarray
    hamming: np.ndarray


def cheap_scores(grey64: np.ndarray) -> CheapScores:
    """Step 1 for every pair of the (n, 64, 64) grey thumbnails."""
    g = grey64.astype(np.float32)
    n = len(g)
    k = CHEAP_SIDE // CORR_SIDE
    g16 = g.reshape(n, CORR_SIDE, k, CORR_SIDE, k).mean((2, 4))
    u, um = _unit_rows(g16), _unit_rows(g16[:, :, ::-1])
    corr, corr_m = u @ u.T, u @ um.T
    d = dctn(g, axes=(1, 2), norm="ortho")[:, :HASH_SIDE, :HASH_SIDE]
    d = d.reshape(n, -1)[:, 1:]  # without the DC term (the mean brightness)
    bits = (d > np.median(d, 1, keepdims=True)).astype(np.float32)
    ham = bits.shape[1] - (bits @ bits.T + (1 - bits) @ (1 - bits).T)
    i, j = np.triu_indices(n, 1)
    return CheapScores(
        i=i.astype(np.int32),
        j=j.astype(np.int32),
        corr=corr[i, j].astype(np.float32),
        corr_mirror=corr_m[i, j].astype(np.float32),
        hamming=np.rint(ham[i, j]).astype(np.int16),
    )


def candidates(s: CheapScores, clip: np.ndarray | None = None) -> np.ndarray:
    """Indices into ``s`` of the pairs worth a precise score.

    Args:
        s: step 1's scores.
        clip: optional unit-length CLIP image features, one row per photo. Any two cat
            photos have a high CLIP cosine, so only very high ones count, and a
            photo's nearest neighbour a little lower.

    Returns:
        The selected pair indices, sorted.
    """
    sel = (s.corr >= CORR_MIN) | (s.corr_mirror >= CORR_MIN) | (s.hamming <= HASH_MAX)
    if clip is not None:
        cos = clip @ clip.T
        cc = cos[s.i, s.j]
        np.fill_diagonal(cos, -1)
        nearest = cos.argmax(1)
        is_nearest = (nearest[s.i] == s.j) | (nearest[s.j] == s.i)
        sel |= (cc >= CLIP_MIN) | (is_nearest & (cc >= CLIP_NEAREST_MIN))
    return np.flatnonzero(sel)


def _scaled(img: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """``img`` zoomed by ``s`` about its centre, back on its grid, and where real
    pixels landed."""
    big = zoom(img, (s, s) + (1,) * (img.ndim - 2), order=1)
    out = np.zeros_like(img)
    m = np.zeros(img.shape[:2], bool)
    n, big_n = img.shape[0], big.shape[0]
    if big_n >= n:
        o = (big_n - n) // 2
        out[:] = big[o : o + n, o : o + n]
        m[:] = True
    else:
        o = (n - big_n) // 2
        out[o : o + big_n, o : o + big_n] = big
        m[o : o + big_n, o : o + big_n] = True
    return out, m


def _placed(
    b: np.ndarray, scale: float, sh: np.ndarray, trim: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """``b`` scaled and shifted, and where it covers the grid (without the grid's
    `EDGE` px border when ``trim``)."""
    bs, m = _scaled(b, scale)
    moved = nd_shift(bs, list(sh) + [0] * (b.ndim - 2), order=1)
    inside = nd_shift(m.astype(np.float32), sh, order=0) > 0.5
    if trim:
        inside[:EDGE] = inside[-EDGE:] = False
        inside[:, :EDGE] = inside[:, -EDGE:] = False
    return moved, inside


@dataclass(frozen=True)
class Alignment:
    """Step 2's result for one pair.

    Attributes:
        ncc: normalised cross-correlation of the grey overlap (-1 when no alignment
            keeps `MIN_OVERLAP` of the grid).
        scale: the second photo's scale onto the first.
        shift: its shift in pixels of the 128 grid (rows, columns).
        mirrored: whether the second photo was mirrored.
        overlap: the share of the grid both photos cover.
        ssim: SSIM of the grey overlap.
        mad_rgb: mean absolute RGB difference of the overlap, 0-255.
    """

    ncc: float
    scale: float | None
    shift: tuple[float, float] | None
    mirrored: bool | None
    overlap: float
    ssim: float
    mad_rgb: float


def aligned_similarity(rgb_a: np.ndarray, rgb_b: np.ndarray) -> Alignment:
    """Step 2: align photo b onto photo a (128 px RGB thumbnails) and compare."""
    ca, cb = rgb_a.astype(np.float32), rgb_b.astype(np.float32)
    a, b = ca @ GREY, cb @ GREY
    best: tuple[float, float, np.ndarray, bool, float] | None = None
    for mirror in (False, True):
        bb = b[:, ::-1] if mirror else b
        for s in SCALES:
            bs, _ = _scaled(bb, float(s))
            sh = phase_cross_correlation(a, bs, upsample_factor=2, normalization=None)[
                0
            ]
            moved, inside = _placed(bb, float(s), sh)
            if inside.mean() < MIN_OVERLAP:
                continue
            x, y = a[inside] - a[inside].mean(), moved[inside] - moved[inside].mean()
            ncc = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
            if best is None or ncc > best[0]:
                best = (ncc, float(s), sh, mirror, float(inside.mean()))
    if best is None:
        return Alignment(-1.0, None, None, None, 0.0, 0.0, 255.0)
    ncc, s, sh, mirror, overlap = best
    bb, cbb = (b[:, ::-1], cb[:, ::-1]) if mirror else (b, cb)
    moved, inside = _placed(bb, s, sh, trim=False)
    moved_rgb, _ = _placed(cbb, s, sh, trim=False)
    ys, xs = np.where(inside)
    y0, y1, x0, x1 = ys.min() + EDGE, ys.max() - EDGE, xs.min() + EDGE, xs.max() - EDGE
    box = (slice(y0, y1), slice(x0, x1))
    ssim = float(structural_similarity(a[box], moved[box], data_range=255))
    mad = float(np.abs(ca[box] - moved_rgb[box]).mean())
    return Alignment(ncc, s, (float(sh[0]), float(sh[1])), mirror, overlap, ssim, mad)


def groups(pairs: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Join pairs of copies into groups (connected components), largest first."""
    parent: dict[str, str] = {}

    def root(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[root(a)] = root(b)
    out: dict[str, list[str]] = defaultdict(list)
    for x in list(parent):
        out[root(x)].append(x)
    return sorted((sorted(v) for v in out.values()), key=len, reverse=True)


def cross_split_pairs(
    pairs: Iterable[tuple[str, str]], split: dict[str, str]
) -> list[tuple[str, str]]:
    """The pairs whose two photos are in different splits (the leaks)."""
    return [(a, b) for a, b in pairs if split[a] != split[b]]


def group_split(
    split: dict[str, str],
    copies: list[list[str]],
    order: Sequence[str] = ("test", "val", "train"),
) -> dict[str, str]:
    """A split in which every group of copies is on one side.

    A group goes to the first split in ``order`` that any of its photos already has
    (so evaluation photos stay where they are and their training copies move out of
    training); photos in no group keep their split.

    Args:
        split: photo name to split name.
        copies: the duplicate groups (`groups`).
        order: which split wins when a group spans several.

    Returns:
        The new photo-to-split mapping.
    """
    out = dict(split)
    for g in copies:
        sides = {split[x] for x in g}
        side = next(s for s in order if s in sides)
        for x in g:
            out[x] = side
    return out

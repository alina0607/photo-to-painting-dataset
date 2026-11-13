"""A drawn cat face for tests that need no data: a grey head with two eyes."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# The nine face points (CAT order) of the drawn face on a 300 x 260 photo.
POINTS = np.array(
    [
        [120.0, 140.0],  # left eye
        [180.0, 140.0],  # right eye
        [150.0, 190.0],  # mouth
        [95.0, 110.0],  # left ear: outer base, tip, inner base
        [100.0, 50.0],
        [130.0, 100.0],
        [170.0, 100.0],  # right ear: inner base, tip, outer base
        [200.0, 50.0],
        [205.0, 110.0],
    ]
)


def synthetic_cat() -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """A photo, its subject mask (0-1) and its face points."""
    photo = Image.new("RGB", (300, 260), (40, 90, 160))
    mask = Image.new("L", photo.size, 0)
    for im, fill in ((photo, (150, 140, 130)), (mask, 255)):
        d = ImageDraw.Draw(im)
        d.ellipse((80, 80, 220, 230), fill=fill)
        d.polygon([(90, 115), (100, 45), (135, 100)], fill=fill)
        d.polygon([(165, 100), (200, 45), (210, 115)], fill=fill)
    d = ImageDraw.Draw(photo)
    for x, y in POINTS[:2]:
        d.ellipse((x - 9, y - 6, x + 9, y + 6), fill=(30, 25, 20))
    return photo, np.asarray(mask, np.float32) / 255, POINTS.copy()

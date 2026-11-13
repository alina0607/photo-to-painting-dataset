"""Turning cat photos into a stroke-by-stroke painting dataset.

The steps, in order: read the raw photos (AFHQ and the CAT dataset), straighten
and cut out each head, drop the photos that cannot teach a painting model, paint
the rest in seven stages, and make before/after pairs for learned edits.
"""

__version__ = "0.1.0"

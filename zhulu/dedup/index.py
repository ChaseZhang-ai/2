"""Near-duplicate index over perceptual fingerprints.

Naive dedup is O(n^2): compare every new asset against everything published in
the campaign so far. At the volume this platform runs (a few thousand assets per
campaign, refreshed continuously) that explodes. We use a BK-tree per hash so a
"find everything within hamming distance <= r" query is sub-linear in practice,
then intersect the three candidate sets (ahash / dhash / phash).

The threshold is deliberately small. Marketing assets shot from the same product
photo are *supposed* to look similar; we only want to reject the ones a platform
would flag as a re-upload. Empirically r=6 on the 64-bit dhash catches
truncated/re-encoded copies while leaving intentionally varied shots alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .phash import Fingerprint, hamming

__all__ = ["BKTree", "NearDuplicateIndex", "DuplicateMatch"]



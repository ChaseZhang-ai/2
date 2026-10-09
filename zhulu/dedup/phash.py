"""Perceptual hashing used to fingerprint generated marketing assets.

The pipeline generates hundreds of near-identical assets per day from the same
product photo. Two assets that are visually too close get down-ranked by short
video platforms (dedup / throttle), so before an asset is published we need a
cheap, robust visual fingerprint.

We deliberately avoid pulling in OpenCV / PyTorch for this step: the fingerprint
has to run on the scheduler's control node (CPU, no GPU) for every generated
asset, so it must be pure Python and O(microseconds), not a tensor op.

Two hashes are produced:

* ``ahash`` - average hash, 64 bit. Cheap, sensitive to global luminance shifts.
* ``dhash`` - difference hash, 64 bit. Robust to brightness/gamma changes and
  to small crops, which is what actually matters for "is this the same shot".

``phash`` (DCT-based) is also implemented because it is the hash we use when the
asset comes back from the video model at a different resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["average_hash", "difference_hash", "phash", "hamming", "GrayImage"]


# A grayscale image is just a row-major list of ints in [0, 255].
GrayImage = Sequence[int]



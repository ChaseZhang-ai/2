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


def _resize_bilinear(src: GrayImage, w: int, h: int, nw: int, nh: int) -> list[float]:
    """Bilinear resize of a row-major grayscale buffer.

    Implemented by hand rather than with PIL so the control node has zero native
    dependencies. ``src`` is ``w*h`` long.
    """
    out: list[float] = [0.0] * (nw * nh)
    x_ratio = (w - 1) / (nw - 1) if nw > 1 else 0.0
    y_ratio = (h - 1) / (nh - 1) if nh > 1 else 0.0
    for j in range(nh):
        fy = j * y_ratio
        y0 = int(fy)
        y1 = min(y0 + 1, h - 1)
        wy = fy - y0
        for i in range(nw):
            fx = i * x_ratio
            x0 = int(fx)
            x1 = min(x0 + 1, w - 1)
            wx = fx - x0
            top = src[y0 * w + x0] * (1 - wx) + src[y0 * w + x1] * wx
            bot = src[y1 * w + x0] * (1 - wx) + src[y1 * w + x1] * wx
            out[j * nw + i] = top * (1 - wy) + bot * wy
    return out


def _to_gray_grid(img: GrayImage, w: int, h: int, size: int) -> list[float]:
    if len(img) != w * h:
        raise ValueError(f"buffer length {len(img)} != {w}x{h}")
    return _resize_bilinear(img, w, h, size, size)


def _bits_to_int(bits: Iterable[bool]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(bool(b))
    return v


def average_hash(img: GrayImage, w: int, h: int, size: int = 8) -> int:
    """Average hash: bit = pixel above the mean luminance."""
    g = _to_gray_grid(img, w, h, size)
    mean = sum(g) / len(g)
    return _bits_to_int(px > mean for px in g)


def difference_hash(img: GrayImage, w: int, h: int, size: int = 8) -> int:
    """Difference hash: bit = pixel brighter than its right neighbour.

    Uses a (size+1) x size grid so we get size*size comparisons.
    """
    g = _to_gray_grid(img, w, h, size + 1)
    row = size + 1
    bits: list[bool] = []
    for j in range(size):
        base = j * row
        for i in range(size):
            bits.append(g[base + i] > g[base + i + 1])
    return _bits_to_int(bits)


# Precomputed 8x8 DCT-II basis. phash keeps the top-left 8x8 of the DCT of a
# 32x32 grayscale image; the DC term is dropped before comparing to the median.
_DCT_N = 32
_DCT_COS: list[list[float]] = [
    [math.cos(math.pi * (2 * x + 1) * u / (2 * _DCT_N)) for x in range(_DCT_N)]
    for u in range(_DCT_N)
]


def phash(img: GrayImage, w: int, h: int, size: int = 8, dct_size: int = 32) -> int:
    """DCT perceptual hash (a.k.a. pHash), 64 bit."""
    g = _to_gray_grid(img, w, h, dct_size)
    # Separable DCT: rows then columns, keeping only the low-frequency block.
    rows: list[list[float]] = [[0.0] * size for _ in range(dct_size)]
    for y in range(dct_size):
        base = y * dct_size
        for u in range(size):
            cu = _DCT_COS[u]
            s = 0.0
            for x in range(dct_size):
                s += g[base + x] * cu[x]
            rows[y][u] = s
    low: list[float] = []
    for u in range(size):
        for v in range(size):
            cu = _DCT_COS[u]
            s = 0.0
            for y in range(dct_size):
                s += rows[y][v] * cu[y]
            low.append(s)
    # Drop DC (low[0]) from the median computation, standard pHash behaviour.
    body = low[1:]
    srt = sorted(body)
    n = len(srt)
    median = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    return _bits_to_int(px > median for px in body)



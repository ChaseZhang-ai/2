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


class BKTree:
    """Burkhard-Keller tree over 64-bit integers with hamming distance."""

    def __init__(self) -> None:
        self._root: tuple[int, dict[int, "BKTree"]] | None = None

    def add(self, key: int) -> None:
        if self._root is None:
            self._root = (key, {})
            return
        node = self
        while True:
            assert node._root is not None
            value, children = node._root
            d = hamming(key, value)
            if d == 0:
                return
            if d in children:
                node = children[d]
            else:
                children[d] = BKTree()
                children[d]._root = (key, {})
                return

    def search(self, key: int, radius: int) -> list[tuple[int, int]]:
        """Return ``(stored_key, distance)`` for all keys within ``radius``."""
        if self._root is None:
            return []
        results: list[tuple[int, int]] = []
        stack = [self]
        while stack:
            node = stack.pop()
            if node._root is None:
                continue
            value, children = node._root
            d = hamming(key, value)
            if d <= radius:
                results.append((value, d))
            lo, hi = d - radius, d + radius
            for edge, child in children.items():
                if lo <= edge <= hi:
                    stack.append(child)
        return results


@dataclass
class DuplicateMatch:
    asset_id: str
    distance: int



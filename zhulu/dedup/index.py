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


@dataclass
class NearDuplicateIndex:
    """Three BK-trees keyed by the three perceptual hashes."""

    radius: int = 6
    _trees: dict[str, BKTree] = field(default_factory=lambda: {"a": BKTree(), "d": BKTree(), "p": BKTree()})
    _by_hash: dict[tuple[str, int], list[str]] = field(default_factory=dict)
    _fp_by_id: dict[str, Fingerprint] = field(default_factory=dict)

    def _insert(self, tag: str, key: int, asset_id: str) -> None:
        self._trees[tag].add(key)
        self._by_hash.setdefault((tag, key), []).append(asset_id)

    def add(self, asset_id: str, fp: Fingerprint) -> None:
        self._fp_by_id[asset_id] = fp
        self._insert("a", fp.ahash, asset_id)
        self._insert("d", fp.dhash, asset_id)
        self._insert("p", fp.phash, asset_id)

    def _candidates(self, tag: str, key: int) -> dict[str, int]:
        out: dict[str, int] = {}
        for stored, dist in self._trees[tag].search(key, self.radius):
            for asset_id in self._by_hash.get((tag, stored), ()):
                prev = out.get(asset_id)
                if prev is None or dist < prev:
                    out[asset_id] = dist
        return out

    def query(self, fp: Fingerprint, exclude: Iterable[str] = ()) -> list[DuplicateMatch]:
        """Return existing assets whose combined distance is within threshold.

        A candidate must clear *every* hash (max-distance rule), so we start from
        the tightest candidate set and filter.
        """
        skip = set(exclude)
        merged: dict[str, int] = {}
        for tag, key in (("d", fp.dhash), ("p", fp.phash), ("a", fp.ahash)):
            for asset_id, dist in self._candidates(tag, key).items():
                if asset_id in skip:
                    continue
                merged[asset_id] = max(merged.get(asset_id, 0), dist)
        matches = [
            DuplicateMatch(asset_id=aid, distance=dist)
            for aid, dist in merged.items()
            if dist <= self.radius
        ]
        matches.sort(key=lambda m: m.distance)
        return matches

    def is_duplicate(self, fp: Fingerprint) -> bool:
        return bool(self.query(fp))

    def __len__(self) -> int:
        return len(self._fp_by_id)

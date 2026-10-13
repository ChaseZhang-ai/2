"""Semantic de-duplication on top of the visual fingerprint layer.

The perceptual index only rejects *visually* near-identical assets. A second
failure mode is *semantically* repetitive output: the same talking points, the
same hook, the same script skeleton rendered as visibly different clips. Platforms
detect this too, and more importantly viewers tune out.

We keep one centroid vector per "content angle" the campaign has already used and
penalise new candidates by their cosine similarity to the nearest centroid. This
is a greedy max-coverage style selection: pick the asset whose embedding is
farthest (in the complementary sense) from everything already chosen.

Embeddings are injected (the caller supplies a real CLIP / text encoder); the
index itself is model-agnostic and pure Python so it stays testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = ["cosine", "SemanticDedup", "SemanticScore"]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


@dataclass
class SemanticScore:
    asset_id: str
    max_similarity: float
    nearest: str | None



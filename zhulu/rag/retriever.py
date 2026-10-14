"""Hybrid retriever: BM25 (lexical) + dense (semantic) fused with RRF.

Reciprocal Rank Fusion is used instead of score normalisation because BM25 scores
and cosine similarities are on incomparable scales and their distributions shift
per language and per tenant. RRF only needs ranks:

    score(d) = sum_over_retrievers 1 / (k + rank_r(d))

with k=60 (the value from the original Cormack et al. paper). This is what makes
the same retriever config work for a Chinese FAQ and an English part-number
lookup without per-tenant tuning - which is the whole point of a multilingual
service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .bm25 import BM25, tokenize

__all__ = ["Document", "HybridRetriever", "rrf_fuse"]

EmbeddingFn = Callable[[str], Sequence[float]]


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def rrf_fuse(
    ranked_lists: Iterable[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into a single ranking."""
    lists = list(ranked_lists)
    if weights is None:
        weights = [1.0] * len(lists)
    scores: dict[str, float] = {}
    for w, ranked in zip(weights, lists):
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))



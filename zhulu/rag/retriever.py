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



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


@dataclass
class HybridRetriever:
    embed: EmbeddingFn | None = None
    bm25: BM25 = field(default_factory=BM25)
    _docs: dict[str, Document] = field(default_factory=dict)
    _vectors: dict[str, Sequence[float]] = field(default_factory=dict)
    rrf_k: int = 60
    lexical_weight: float = 1.0
    dense_weight: float = 1.0

    def add(self, doc: Document) -> None:
        self._docs[doc.doc_id] = doc
        self.bm25.add(doc.doc_id, doc.text)
        if self.embed is not None:
            self._vectors[doc.doc_id] = list(self.embed(doc.text))

    def add_all(self, docs: Iterable[Document]) -> None:
        for d in docs:
            self.add(d)

    def _dense_rank(self, query: str) -> list[str]:
        if self.embed is None or not self._vectors:
            return []
        qv = self.embed(query)
        scored = []
        for doc_id, vec in self._vectors.items():
            dot = sum(a * b for a, b in zip(qv, vec))
            na = sum(a * a for a in qv) ** 0.5
            nb = sum(b * b for b in vec) ** 0.5
            sim = dot / (na * nb) if na and nb else 0.0
            scored.append((doc_id, sim))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [d for d, _ in scored]

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        lexical = [d for d, _ in self.bm25.search(query, top_k=max(top_k * 4, 20))]
        dense = self._dense_rank(query)[: max(top_k * 4, 20)]
        fused = rrf_fuse(
            [lexical, dense],
            k=self.rrf_k,
            weights=[self.lexical_weight, self.dense_weight],
        )
        out: list[tuple[Document, float]] = []
        for doc_id, score in fused[:top_k]:
            if doc_id in self._docs:
                out.append((self._docs[doc_id], score))
        return out

    def __len__(self) -> int:
        return len(self._docs)

"""A self-contained BM25 implementation.

The customer-service knowledge base is small (a few thousand chunks per tenant)
but queries are high-QPS and multilingual. A lexical retriever is cheap and, for
product/FAQ text full of rare part numbers and SKUs, often *more* precise than a
dense retriever. We therefore run both and fuse (see ``retriever.py``).

Character-level tokenisation is used because the corpus mixes Chinese, English,
Japanese and Korean; word segmentation for CJK would add a dependency and a
failure mode for the same recall.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = ["tokenize", "BM25"]


def tokenize(text: str) -> list[str]:
    """CJK-aware tokenizer: latin words as-is, CJK as overlapping bigrams.

    Bigrams give CJK the same "phrase-ish" signal latin words get from spaces
    without a segmentation dictionary.
    """
    tokens: list[str] = []
    buf: list[str] = []
    cjk_prev: str | None = None
    for ch in text.lower():
        if ch.isascii() and (ch.isalnum()):
            buf.append(ch)
            cjk_prev = None
            continue
        if buf:
            tokens.append("".join(buf))
            buf = []
        code = ord(ch)
        is_cjk = (
            0x4E00 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7A3
        )
        if is_cjk:
            if cjk_prev is not None:
                tokens.append(cjk_prev + ch)
            cjk_prev = ch
        else:
            cjk_prev = None
    if buf:
        tokens.append("".join(buf))
    return tokens


@dataclass
class BM25:
    k1: float = 1.5
    b: float = 0.75
    _doc_ids: list[str] = field(default_factory=list)
    _tf: list[Counter] = field(default_factory=list)
    _len: list[int] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _avgdl: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        tf = Counter(toks)
        self._doc_ids.append(doc_id)
        self._tf.append(tf)
        self._len.append(len(toks))
        for term in tf:
            self._df[term] += 1
        self._avgdl = sum(self._len) / len(self._len)

    def add_all(self, docs: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in docs:
            self.add(doc_id, text)

    def _idf(self, term: str) -> float:
        n = len(self._doc_ids)
        df = self._df.get(term, 0)
        # BM25+ style idf, always positive.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_terms = tokenize(query)
        if not q_terms or not self._doc_ids:
            return []
        scores: list[tuple[str, float]] = []
        for i, doc_id in enumerate(self._doc_ids):
            score = 0.0
            tf = self._tf[i]
            dl = self._len[i] or 1
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                idf = self._idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1e-9))
                score += idf * f * (self.k1 + 1) / denom
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:top_k]

    def __len__(self) -> int:
        return len(self._doc_ids)

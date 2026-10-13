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



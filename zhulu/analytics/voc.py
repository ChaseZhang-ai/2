"""Voice-of-customer (VOC) analytics.

Customer reviews are collected automatically per store, then clustered into
themes so the content-ops team knows what to put in the next batch of videos.
We use a light approach: normalise text, extract aspect terms by pattern, then
score sentiment per aspect with a small lexicon and a negation window.

Doing this offline (rather than with an LLM call per review) keeps the per-review
cost at zero and lets the number be recomputed over the whole history on demand.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..rag.bm25 import tokenize

__all__ = ["classify_sentiment", "AspectSummary", "summarize_voc", "top_negative_aspects"]

_POS = {
    "好", "👍", "喜欢", "满意", "推荐", "棒", "赞", "快", "实惠", "耐用", "值", "不错", "精细",
    "good", "great", "love", "fast", "nice", "perfect", "recommend", "worth",
}
_NEG = {
    "差", "慢", "贵", "失望", "退货", "坏", "破", "假", "难用", "踩雷", "破损",
    "bad", "slow", "expensive", "broken", "disappointed", "refund", "poor",
}
_NEGATORS = {"不", "没", "无", "别", "not", "no", "never", "without"}

# Aspect lexicon: token -> canonical aspect.
_ASPECTS = {
    "价格": "price", "贵": "price", "便宜": "price", "实惠": "price", "price": "price",
    "物流": "shipping", "快递": "shipping", "发货": "shipping", "shipping": "shipping",
    "质量": "quality", "材质": "quality", "做工": "quality", "quality": "quality",
    "客服": "service", "服务": "service", "service": "service",
    "效果": "efficacy", "味道": "efficacy", "口感": "efficacy", "efficacy": "efficacy",
}


def _is_cjk_2(tok: str) -> bool:
    return len(tok) == 2 and all(ord(c) > 0x2E80 for c in tok)


def _sentiment_tokens(text: str) -> list[str]:
    """Latin words plus CJK bigrams *and* their constituent characters.

    Bigrams let multi-character terms (不错/失望/破损) fire; the characters let
    single-character polarity words (好/差/贵/快) fire. Used for aspect matching.
    """
    out: list[str] = []
    for tok in tokenize(text):
        out.append(tok)
        if _is_cjk_2(tok):
            out.extend(tok)
    return out



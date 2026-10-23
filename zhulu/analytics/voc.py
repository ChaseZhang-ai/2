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


def _scan(tokens: Sequence[str], negators: set[str]) -> tuple[int, int]:
    """Sum polarity over tokens, flipping sign inside a negation window."""
    score = 0
    hits = 0
    for i, tok in enumerate(tokens):
        polarity = 1 if tok in _POS else -1 if tok in _NEG else 0
        if polarity == 0:
            continue
        window = set(tokens[max(0, i - 3) : i])
        if window & negators:
            polarity = -polarity
        score += polarity
        hits += 1
    return score, hits


def classify_sentiment(text: str) -> float:
    """Return sentiment in [-1, 1] using a lexicon with a negation window.

    Scored in two independent passes - multi-char/latin words, then CJK
    characters - so that the unigrams inside a bigram cannot leak into its
    negation window (e.g. the 不 in 不错 must not negate 不错 itself).
    """
    words = tokenize(text)
    word_score, word_hits = _scan(words, _NEGATORS)
    chars = [c for tok in words if _is_cjk_2(tok) for c in tok]
    char_score, char_hits = _scan(chars, _NEGATORS)
    score = word_score + char_score
    hits = word_hits + char_hits
    if hits == 0:
        return 0.0
    # Squash to [-1, 1]; more hits push further from neutral.
    return math.tanh(score / math.sqrt(hits))


@dataclass
class AspectSummary:
    aspect: str
    mentions: int
    mean_sentiment: float
    examples: list[str] = field(default_factory=list)


def summarize_voc(reviews: Iterable[str], top_examples: int = 3) -> list[AspectSummary]:
    buckets: dict[str, list[float]] = defaultdict(list)
    examples: dict[str, list[str]] = defaultdict(list)
    for review in reviews:
        toks = _sentiment_tokens(review)
        aspects_in_review = {_ASPECTS[t] for t in toks if t in _ASPECTS}
        if not aspects_in_review:
            buckets["other"].append(classify_sentiment(review))
            if len(examples["other"]) < top_examples:
                examples["other"].append(review)
            continue
        s = classify_sentiment(review)
        for aspect in aspects_in_review:
            buckets[aspect].append(s)
            if len(examples[aspect]) < top_examples:
                examples[aspect].append(review)
    out: list[AspectSummary] = []
    for aspect, scores in buckets.items():
        mean = sum(scores) / len(scores)
        out.append(AspectSummary(aspect, len(scores), round(mean, 4), examples[aspect][:top_examples]))
    out.sort(key=lambda a: (a.mean_sentiment, -a.mentions))
    return out



"""Multilingual customer-service agent built on the hybrid retriever.

The service layer adds the parts a raw retriever lacks:

* **Language detection** (script-based, cheap) so replies are generated in the
  customer's language without a model round-trip.
* **Grounded answering with an abstention gate.** If the top fused retrieval
  score is below ``min_score`` the agent escalates to a human instead of letting
  the LLM hallucinate a policy. Refusing to answer is a feature for B2B support.
* **Ticket/tool hand-off.** The agent can emit a structured action (create order
  lookup, open ticket) rather than only free text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .retriever import Document, HybridRetriever

__all__ = ["Answer", "CustomerServiceAgent", "detect_language"]

# Script -> language heuristic. Good enough to pick a reply language for the
# languages this platform supports (zh/en/ja/ko + latin fallback).
_RANGES = [
    ("zh", 0x4E00, 0x9FFF),
    ("ja", 0x3040, 0x30FF),
    ("ko", 0xAC00, 0xD7A3),
    ("ru", 0x0400, 0x04FF),
]


def detect_language(text: str) -> str:
    counts = {lang: 0 for lang, _, _ in _RANGES}
    latin = 0
    for ch in text:
        code = ord(ch)
        if ch.isascii() and ch.isalpha():
            latin += 1
            continue
        for lang, lo, hi in _RANGES:
            if lo <= code <= hi:
                counts[lang] += 1
                break
    best = max(counts.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return "en" if latin else "en"
    # Japanese text also contains kanji in the zh range; kana presence wins.
    if best[0] == "zh" and counts["ja"] > 0:
        return "ja"
    return best[0]


@dataclass
class Answer:
    text: str
    language: str
    sources: list[str]
    escalated: bool
    action: dict | None = None



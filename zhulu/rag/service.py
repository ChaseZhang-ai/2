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


@dataclass
class CustomerServiceAgent:
    retriever: HybridRetriever
    generate: Callable[[str, Sequence[Document], str], str] | None = None
    min_score: float = 0.008
    top_k: int = 4
    tenant: str = "default"

    def answer(self, query: str, history: Sequence[str] = ()) -> Answer:
        lang = detect_language(query)
        hits = self.retriever.search(query, top_k=self.top_k)
        if not hits or hits[0][1] < self.min_score:
            return Answer(
                text=_escalation_message(lang),
                language=lang,
                sources=[],
                escalated=True,
                action={"type": "open_ticket", "tenant": self.tenant, "reason": "low_confidence"},
            )
        docs = [d for d, _ in hits]
        sources = [d.doc_id for d in docs]
        if self.generate is not None:
            text = self.generate(query, docs, lang)
        else:
            text = docs[0].text
        return Answer(text=text, language=lang, sources=sources, escalated=False)

    def can_answer(self, query: str) -> bool:
        hits = self.retriever.search(query, top_k=1)
        return bool(hits) and hits[0][1] >= self.min_score


def _escalation_message(lang: str) -> str:
    return {
        "zh": "抱歉，这个问题我需要转接人工客服为您确认，请稍候。",
        "en": "I'd like to have a specialist confirm this for you - transferring you now.",
        "ja": "申し訳ございません。担当者に確認いたしますので少々お待ちください。",
        "ko": "죄송합니다. 담당자에게 확인 후 안내드리겠습니다.",
        "ru": "Уточню у специалиста и вернусь к вам.",
    }.get(lang, "Let me have a specialist confirm this for you.")

"""Content-angle planning: make a batch diverse *before* it is generated.

The dedup gate (`zhulu.dedup`) rejects redundant output, but rejecting is
expensive: every rejected asset already cost GPU time. The cheaper move is to
plan for diversity up front, so the batch that reaches the GPU spans distinct
*angles* - a lifestyle shot, a macro detail, an unboxing, a price/value frame -
instead of ten paraphrases of the same prompt.

`plan_angles` returns deterministic, unique prompt variants for one product. A
`ContentAngle.key` doubles as a stable content-angle id, which is what the
campaign report groups performance by (`analytics/roi.py`) - so "which angle
converts best" is answerable later.

Capacity is bounded on purpose: templates x (1 + modifiers). Beyond that the
prompts would start repeating, which is exactly what we are trying to avoid, so
the function raises instead of silently returning duplicates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "ANGLE_TEMPLATES",
    "ANGLE_MODIFIERS",
    "ContentAngle",
    "plan_angles",
    "capacity",
    "prompt_diversity",
]

# Each template is a distinct *marketing angle*, not a paraphrase of the others.
ANGLE_TEMPLATES: dict[str, str] = {
    "problem-solution": "{product} solving a real everyday problem, close-up on the moment of use, natural light",
    "lifestyle": "{product} in a calm weekend scene, wide shot, soft window light",
    "unboxing": "hands unboxing {product}, top-down, clean background",
    "detail-macro": "macro detail of {product}, texture and stitching visible, shallow depth of field",
    "price-value": "{product} beside its price tag, retail shelf, honest daylight",
    "comparison": "{product} next to a generic alternative, side by side, neutral studio",
    "testimonial": "portrait of a satisfied customer holding {product}, warm indoor light",
    "seasonal": "{product} styled for the season, seasonal props, soft shadows",
    "motion-broll": "{product} rotating slowly on a pedestal, studio rim light",
    "honest-review": "casual phone-shot style photo of {product} on a desk, unretouched",
}

ANGLE_MODIFIERS: tuple[str, ...] = (
    "morning light",
    "overcast daylight",
    "warm tungsten",
    "minimal white background",
    "outdoor park",
    "kitchen counter",
    "commute scene",
    "evening window light",
)


@dataclass(frozen=True)
class ContentAngle:
    """One planned prompt, tagged with the angle it belongs to."""

    name: str
    prompt: str
    key: str

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.prompt


def capacity() -> int:
    """Number of unique angles this planner can produce."""
    return len(ANGLE_TEMPLATES) * (1 + len(ANGLE_MODIFIERS))


def plan_angles(
    product: str,
    n: int,
    seed: int = 0,
    avoid: Iterable[str] = (),
) -> list[ContentAngle]:
    """Plan ``n`` unique, deterministic content angles for ``product``.

    Args:
        product: product name or short descriptor, substituted into templates.
        n: how many angles to plan.
        seed: makes the selection reproducible; the same seed and corpus always
            yield the same plan.
        avoid: angle names already used in a previous campaign. They are moved
            to the back of the queue so a repeat campaign leads with fresh angles
            rather than re-treading old ground.

    Raises:
        ValueError: if ``n`` exceeds :func:`capacity`, or is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not product.strip():
        raise ValueError("product must be a non-empty string")
    cap = capacity()
    if n > cap:
        raise ValueError(f"asked for {n} angles but only {cap} unique combinations exist")

    rng = random.Random(seed)
    avoided = set(avoid)
    fresh = [name for name in ANGLE_TEMPLATES if name not in avoided]
    used = [name for name in ANGLE_TEMPLATES if name in avoided]
    rng.shuffle(fresh)
    rng.shuffle(used)
    order = fresh + used

    modifiers: list[str | None] = [None, *ANGLE_MODIFIERS]
    angles: list[ContentAngle] = []
    for modifier in modifiers:
        for name in order:
            if len(angles) == n:
                return angles
            prompt = ANGLE_TEMPLATES[name].format(product=product)
            key = f"{name}::base" if modifier is None else f"{name}::{modifier}"
            if modifier is not None:
                prompt = f"{prompt}, {modifier}"
            angles.append(ContentAngle(name=name, prompt=prompt, key=key))
    return angles


def prompt_diversity(angles: Sequence[ContentAngle]) -> float:
    """Mean pairwise Jaccard distance between prompts, in [0, 1].

    A crude but cheap proxy for how different the planned prompts are. 0 means
    every pair is word-identical; 1 means no shared words at all.
    """
    def words(text: str) -> set[str]:
        return {w.strip(".,;:").lower() for w in text.split() if w.strip(".,;:")}

    sets = [words(a.prompt) for a in angles]
    if len(sets) < 2:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            total += 0.0 if not union else 1.0 - len(inter) / len(union)
            pairs += 1
    return total / pairs

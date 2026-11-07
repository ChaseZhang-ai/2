"""Reproduce the headline throughput/cost numbers from docs/benchmark.md.

Runs on CPU only - this measures the *control-plane* decisions (batching, dedup),
not GPU sampling, which the numbers in docs/benchmark.md combine with measured
4090 timings from the production run.

    python examples/bench_numbers.py
"""

from __future__ import annotations

import json
import math
import sys
import time

from zhulu.dedup import NearDuplicateIndex, SemanticDedup, fingerprint
from zhulu.scheduler import CostAwareScheduler, Job, WorkerPool

GPU_HOURLY_CNY = 1.59          # RTX 4090, CNY/hour (2025 market rate)
STUDIO_IMAGE_CNY = 100.0       # traditional commercial photo, per shot
STUDIO_VIDEO_CNY_PER_MIN = 14.0


def _lcg(seed: int) -> callable:
    state = (seed * 2654435761 + 1) & 0xFFFFFFFF

    def nxt() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return (state >> 8) / 0x7FFFFF

    return nxt


def synth_image(seed: int, w: int = 32, h: int = 32) -> list[int]:
    """A seed-dependent image built by upsampling a random 8x8 thumbnail.

    Distinct seeds give genuinely distinct images (unique perceptual hashes);
    the same seed reproduces the same image (a re-render). A smooth analytic ramp
    would hash identically for every seed, which is not what a product photo
    library looks like.
    """
    rng = _lcg(seed)
    thumb = [[int(rng() * 256) for _ in range(8)] for _ in range(8)]
    out: list[int] = []
    for y in range(h):
        fy = y * 7 / (h - 1)
        y0 = int(fy)
        y1 = min(y0 + 1, 7)
        wy = fy - y0
        for x in range(w):
            fx = x * 7 / (w - 1)
            x0 = int(fx)
            x1 = min(x0 + 1, 7)
            wx = fx - x0
            top = thumb[y0][x0] * (1 - wx) + thumb[y0][x1] * wx
            bot = thumb[y1][x0] * (1 - wx) + thumb[y1][x1] * wx
            out.append(int(max(0, min(255, top * (1 - wy) + bot * wy))))
    return out



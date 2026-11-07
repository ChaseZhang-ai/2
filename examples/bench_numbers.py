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


def bench_scheduler(n_jobs: int = 96) -> dict:
    sched = CostAwareScheduler(vram_budget_mb=24576, max_batch_size=16)
    for i in range(n_jobs):
        video = (i % 3 == 0)
        sched.submit(Job(
            job_id=f"j{i}",
            workflow_signature="wan_video" if video else "flux_model_image",
            vram_mb=2600 if video else 1800,
            est_seconds=4.2 if video else 1.4,
        ))
    plan = sched.plan()
    pool = WorkerPool(vram_budget_mb=24576)
    result = pool.run(plan.batches)
    return {
        "jobs": plan.total_jobs,
        "batches": len(plan.batches),
        "model_loads": result.model_loads,
        "batched_seconds": round(plan.makespan_seconds, 1),
        "unbatched_seconds": round(plan.single_job_makespan_seconds, 1),
        "speedup": round(plan.speedup, 2),
    }


def bench_dedup(n_images: int = 2000, duplicate_rate: float = 0.3) -> dict:
    idx = NearDuplicateIndex(radius=6)
    t0 = time.perf_counter()
    dupes = 0
    for i in range(n_images):
        is_dup = (i % 10) < 10 * duplicate_rate
        seed = (i // 4) if is_dup else i
        img = synth_image(seed)
        fp = fingerprint(img, 32, 32)
        matches = idx.query(fp)
        if matches:
            dupes += 1
        else:
            idx.add(f"img{i}", fp)
    dt = time.perf_counter() - t0
    return {
        "indexed": len(idx),
        "flagged_duplicates": dupes,
        "duplicate_rate": duplicate_rate,
        "seconds": round(dt, 3),
        "per_1000_images_ms": round(dt / n_images * 1000 * 1000, 2),
    }



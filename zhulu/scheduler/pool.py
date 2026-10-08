"""Simulated GPU worker pool.

In production this owns one ComfyUI process per GPU and drives it over the HTTP
API. Here we provide a deterministic simulator so the scheduler's batching
behaviour can be measured and regression-tested in CI without a GPU.

The simulator models the two costs that actually matter: a fixed per-batch model
load and a per-job sampling time, both scaled by the batch's resident memory
utilisation (a batch that nearly fills the card runs slightly slower per item).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .queue import Batch

__all__ = ["WorkerPool", "PoolResult"]


@dataclass
class PoolResult:
    batches_run: int
    jobs_run: int
    wall_seconds: float
    model_loads: int
    per_job_seconds: list[float] = field(default_factory=list)


class WorkerPool:
    def __init__(self, vram_budget_mb: int = 24576, load_seconds: float = 8.0) -> None:
        self.vram_budget_mb = vram_budget_mb
        self.load_seconds = load_seconds

    def _batch_seconds(self, batch: Batch) -> float:
        if not batch.jobs:
            return 0.0
        # Memory pressure factor: full card -> up to 25% slower per item.
        pressure = batch.vram_mb / max(self.vram_budget_mb, 1)
        factor = 1.0 + 0.25 * pressure
        sampling = sum(j.est_seconds for j in batch.jobs) * factor
        return self.load_seconds + sampling

    def run(self, batches: Iterable[Batch]) -> PoolResult:
        total = 0.0
        jobs = 0
        loads = 0
        per_job: list[float] = []
        n = 0
        for batch in batches:
            secs = self._batch_seconds(batch)
            total += secs
            jobs += len(batch.jobs)
            loads += 1
            n += 1
            if batch.jobs:
                share = secs / len(batch.jobs)
                per_job.extend([share] * len(batch.jobs))
        return PoolResult(
            batches_run=n,
            jobs_run=jobs,
            wall_seconds=total,
            model_loads=loads,
            per_job_seconds=per_job,
        )

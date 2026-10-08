"""Cost-aware batching scheduler for generation jobs.

The economics of this platform only work if the GPU stays saturated. A single
RTX 4090 at ~1.6 CNY/h has to produce enough assets to beat the 100 CNY/shot
studio cost by two orders of magnitude. The naive approach - run each request as
it arrives - wastes the GPU on per-job model loading and leaves it idle during
prompt-engineering stalls.

This scheduler does three things the "just use a Celery queue" approach misses:

1. **Workflow-signature batching.** Jobs whose ComfyUI graph differs only in
   prompt/seed share loaded weights, so they are grouped into one batch. Jobs
   with a different graph start a new batch (re-loading a 12 GB checkpoint costs
   ~8 s, which dwarfs the ~1.4 s of actual sampling for a 1024px image).

2. **VRAM-constrained bin packing.** Batches are packed so their peak resident
   memory fits the card with a safety margin; the packer is a first-fit-decreasing
   over per-job estimated VRAM with a cap on batch size.

3. **Priority + deadline ordering.** Interactive customer requests (an operator
   waiting on a preview) preempt bulk campaign backfill. We sort by
   ``(priority desc, deadline asc)`` and apply an ageing term so low-priority
   backfill cannot starve forever.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Iterable, Iterator, Sequence

__all__ = [
    "Priority",
    "Job",
    "Batch",
    "SchedulePlan",
    "CostAwareScheduler",
]


class Priority(IntEnum):
    INTERACTIVE = 0
    CAMPAIGN = 1
    BACKFILL = 2


@dataclass(order=True)
class Job:
    """A single generation request."""

    # sort key; the actual payload is excluded from comparisons
    sort_key: tuple = field(init=False, repr=False)
    seq: int = field(default=0)
    job_id: str = field(default="", compare=False)
    workflow_signature: str = field(default="", compare=False)
    vram_mb: int = field(default=2048, compare=False)
    priority: Priority = field(default=Priority.CAMPAIGN, compare=False)
    submitted_at: float = field(default_factory=time.monotonic, compare=False)
    deadline: float | None = field(default=None, compare=False)
    est_seconds: float = field(default=1.5, compare=False)

    def __post_init__(self) -> None:
        self.sort_key = (self._effective_priority(), self._deadline_key(), self.seq)

    def _deadline_key(self) -> float:
        return self.deadline if self.deadline is not None else float("inf")

    def _effective_priority(self) -> float:
        # Ageing: every 30 s a waiting job gains 0.5 priority so backfill cannot
        # starve behind an endless stream of interactive jobs.
        waited = time.monotonic() - self.submitted_at
        age_bonus = 0.5 * (waited / 30.0)
        return float(self.priority) - min(age_bonus, 2.0)


@dataclass
class Batch:
    signature: str
    jobs: list[Job] = field(default_factory=list)
    vram_mb: int = 0

    def add(self, job: Job) -> None:
        self.jobs.append(job)
        self.vram_mb += job.vram_mb

    def est_seconds(self) -> float:
        # One shared load per batch plus the sum of per-job sampling time.
        if not self.jobs:
            return 0.0
        load = 8.0
        return load + sum(j.est_seconds for j in self.jobs)


@dataclass
class SchedulePlan:
    batches: list[Batch]
    makespan_seconds: float
    single_job_makespan_seconds: float
    total_jobs: int

    @property
    def speedup(self) -> float:
        if self.makespan_seconds <= 0:
            return float("inf")
        return self.single_job_makespan_seconds / self.makespan_seconds



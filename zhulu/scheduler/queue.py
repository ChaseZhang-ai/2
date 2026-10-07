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



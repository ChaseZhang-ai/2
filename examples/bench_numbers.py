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



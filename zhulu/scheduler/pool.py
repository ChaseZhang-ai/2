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



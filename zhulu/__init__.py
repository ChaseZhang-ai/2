"""ZhuluAI - generative content-ops and enterprise-automation engine.

A composable toolkit for automated marketing-content generation and B-end
workflow automation:

* :mod:`zhulu.generation` - ComfyUI-compatible workflow graphs for product photos,
  model shots, marketing video and digital humans.
* :mod:`zhulu.scheduler`  - cost-aware GPU batching and a worker-pool simulator.
* :mod:`zhulu.dedup`      - perceptual + semantic near-duplicate rejection.
* :mod:`zhulu.rag`        - hybrid (BM25 + dense, RRF-fused) multilingual
  customer-service retrieval.
* :mod:`zhulu.workflow`   - a dependency-free DAG engine for approval, routing
  and data-sync flows.
* :mod:`zhulu.analytics`  - VOC clustering and full-funnel ROI attribution.
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import analytics, dedup, generation, rag, scheduler, workflow

__all__ = [
    "__version__",
    "analytics",
    "dedup",
    "generation",
    "rag",
    "scheduler",
    "workflow",
]

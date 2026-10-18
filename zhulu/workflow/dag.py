"""A minimal, dependency-free DAG execution engine.

The enterprise side of the platform ("企业自动化工作流") lets customers compose
approval chains, ticket routing and data-sync jobs. Off-the-shelf orchestrators
(Airflow, Temporal) were rejected for the B-end deployments here because they
need a control plane the customer's IT team will not run on-prem; the whole engine
has to fit inside the delivered product.

Design choices that matter at this scale:

* **Explicit DAG, no cron DSL.** Nodes declare their dependencies; the engine
  topologically sorts and runs independent nodes concurrently.
* **Per-node retry with exponential backoff + jitter.** Data-sync nodes talk to
  flaky customer APIs; a fixed retry stampedes them.
* **Conditional edges.** A node can declare a predicate over its parent's output;
  if false, the downstream subtree is skipped (not failed) - important for
  approval flows where half the branches are intentionally not taken.
* **Idempotency keys.** Each node execution carries a deterministic key so a
  retried run does not double-post a ticket.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

__all__ = ["NodeStatus", "Task", "TaskResult", "WorkflowEngine", "DAGError"]



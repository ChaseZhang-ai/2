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


class DAGError(Exception):
    pass


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """One node: an action plus its scheduling metadata."""

    task_id: str
    action: Callable[[dict[str, Any]], Any]
    depends_on: tuple[str, ...] = ()
    max_retries: int = 2
    base_backoff: float = 0.2
    # predicate over the resolved upstream outputs; None means "always run"
    condition: Callable[[dict[str, Any]], bool] | None = None

    def idempotency_key(self, run_id: str) -> str:
        return hashlib.sha256(f"{run_id}:{self.task_id}".encode()).hexdigest()[:16]


@dataclass
class TaskResult:
    task_id: str
    status: NodeStatus
    output: Any = None
    error: str | None = None
    attempts: int = 0
    duration: float = 0.0


@dataclass
class WorkflowEngine:
    max_workers: int = 4
    sleep: Callable[[float], None] = time.sleep
    jitter: float = 0.15

    def _backoff(self, task: Task, attempt: int) -> float:
        base = task.base_backoff * (2 ** attempt)
        return base + random.uniform(0, self.jitter * base)

    def topological_layers(self, tasks: dict[str, Task]) -> list[list[str]]:
        indeg = {tid: 0 for tid in tasks}
        adj: dict[str, list[str]] = {tid: [] for tid in tasks}
        for tid, task in tasks.items():
            for dep in task.depends_on:
                if dep not in tasks:
                    raise DAGError(f"task {tid!r} depends on unknown {dep!r}")
                adj[dep].append(tid)
                indeg[tid] += 1
        layers: list[list[str]] = []
        ready = sorted(t for t, d in indeg.items() if d == 0)
        seen = 0
        while ready:
            layers.append(ready)
            seen += len(ready)
            nxt: list[str] = []
            for n in ready:
                for m in adj[n]:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        nxt.append(m)
            ready = sorted(nxt)
        if seen != len(tasks):
            raise DAGError("workflow contains a cycle")
        return layers

    def run(
        self,
        tasks: Iterable[Task],
        run_id: str = "run",
        context: dict[str, Any] | None = None,
    ) -> dict[str, TaskResult]:
        """Execute ``tasks``.

        Each node's action receives a context that is the seeded ``context``
        merged with ``{dependency_id: dependency_output}``. So a node can read
        run-level values directly (``ctx["amount"]``) and upstream outputs by id
        (``ctx["approval"]``).
        """
        base: dict[str, Any] = dict(context or {})
        task_map = {t.task_id: t for t in tasks}
        layers = self.topological_layers(task_map)
        results: dict[str, TaskResult] = {}
        outputs: dict[str, Any] = {}

        for layer in layers:
            # Decide per node whether to run, skip (condition false) or skip
            # (an upstream dependency did not succeed).
            runnable: list[Task] = []
            for tid in layer:
                task = task_map[tid]
                failed_dep = next(
                    (
                        d
                        for d in task.depends_on
                        if results.get(d, TaskResult(d, NodeStatus.SUCCEEDED)).status is not NodeStatus.SUCCEEDED
                    ),
                    None,
                )
                if failed_dep is not None:
                    results[tid] = TaskResult(
                        tid, NodeStatus.SKIPPED, error=f"upstream {failed_dep!r} not succeeded"
                    )
                    continue
                if task.condition is not None:
                    upstream = {d: outputs.get(d) for d in task.depends_on}
                    if not task.condition(upstream):
                        results[tid] = TaskResult(tid, NodeStatus.SKIPPED)
                        continue
                runnable.append(task)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(self._run_task, t, base, outputs, run_id): t for t in runnable}
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    results[res.task_id] = res
                    if res.status is NodeStatus.SUCCEEDED:
                        outputs[res.task_id] = res.output
        return results

    def _run_task(
        self, task: Task, base: dict[str, Any], outputs: dict[str, Any], run_id: str
    ) -> TaskResult:
        ctx = dict(base)
        for dep in task.depends_on:
            ctx[dep] = outputs.get(dep)
        ctx["_idempotency_key"] = task.idempotency_key(run_id)
        ctx["_run_id"] = run_id
        start = time.monotonic()
        last_err: str | None = None
        for attempt in range(task.max_retries + 1):
            try:
                out = task.action(ctx)
                return TaskResult(task.task_id, NodeStatus.SUCCEEDED, out, None, attempt + 1, time.monotonic() - start)
            except Exception as exc:  # noqa: BLE001 - engine must not die on a node
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < task.max_retries:
                    self.sleep(self._backoff(task, attempt))
        return TaskResult(task.task_id, NodeStatus.FAILED, None, last_err, task.max_retries + 1, time.monotonic() - start)

    @staticmethod
    def describe(results: dict[str, TaskResult]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in results.values():
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        return counts

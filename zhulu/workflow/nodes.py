"""Ready-made workflow nodes: approval, ticket routing and data sync.

These are the three flows shipped to customers in the first release ("AI 智能审批",
"AI 智能工单分发", "AI 数据同步"). Each is a small, testable callable so the
engine stays generic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .dag import Task, WorkflowEngine

__all__ = [
    "ApprovalPolicy",
    "make_approval_flow",
    "make_ticket_router",
    "make_data_sync_flow",
]


@dataclass
class ApprovalPolicy:
    """Thresholds that decide whether an approval auto-passes or needs a human."""

    auto_approve_below: float = 1000.0
    escalate_above: float = 50_000.0

    def decide(self, amount: float, risk_score: float) -> str:
        if amount >= self.escalate_above or risk_score >= 0.7:
            return "escalate"
        if amount < self.auto_approve_below and risk_score < 0.3:
            return "auto_approve"
        return "manager_review"


def make_approval_flow(policy: ApprovalPolicy | None = None) -> Task:
    """Return a task that classifies an approval request by amount and risk."""
    policy = policy or ApprovalPolicy()

    def action(ctx: dict[str, Any]) -> dict[str, Any]:
        amount = float(ctx.get("amount", 0.0))
        risk = float(ctx.get("risk_score", 0.0))
        decision = policy.decide(amount, risk)
        return {"decision": decision, "amount": amount, "risk_score": risk}

    return Task("approval", action)


def make_ticket_router(
    routes: Sequence[tuple[str, Callable[[str], bool]]],
    default_queue: str = "general",
) -> Task:
    """Route a ticket to a queue by keyword predicates (first match wins)."""
    compiled = [(queue, pred) for queue, pred in routes]

    def action(ctx: dict[str, Any]) -> dict[str, Any]:
        text = str(ctx.get("ticket_text", ""))
        for queue, pred in compiled:
            if pred(text):
                return {"queue": queue, "matched": True}
        return {"queue": default_queue, "matched": False}

    return Task("route", action)


def make_data_sync_flow(
    fetch: Callable[[dict[str, Any]], Any],
    transform: Callable[[Any], Any],
    load: Callable[[Any], Any],
    depends: tuple[str, ...] = (),
) -> list[Task]:
    """A three-stage ETL: fetch -> transform -> load, with retries on load."""

    def t_fetch(ctx: dict[str, Any]) -> Any:
        return fetch(ctx)

    def t_transform(ctx: dict[str, Any]) -> Any:
        return transform(ctx.get("fetch"))

    def t_load(ctx: dict[str, Any]) -> Any:
        return load(ctx.get("transform"))

    return [
        Task("fetch", t_fetch, depends_on=depends, max_retries=3),
        Task("transform", t_transform, depends_on=("fetch",)),
        Task("load", t_load, depends_on=("transform",), max_retries=3, base_backoff=0.5),
    ]

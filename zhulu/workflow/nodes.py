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



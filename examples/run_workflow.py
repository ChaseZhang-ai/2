"""Run the three enterprise workflow flavours end to end.

    PYTHONPATH=. python examples/run_workflow.py

Approval routing, ticket routing and a fetch->transform->load sync, executed by
the DAG engine with retries and conditional branches.
"""

from __future__ import annotations

from zhulu.workflow import (
    ApprovalPolicy,
    Task,
    WorkflowEngine,
    make_approval_flow,
    make_data_sync_flow,
    make_ticket_router,
)



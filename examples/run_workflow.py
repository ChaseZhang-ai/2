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


def main() -> None:
    engine = WorkflowEngine(max_workers=4)

    # --- approval: auto-approve small low-risk requests, escalate big ones ----
    approval = make_approval_flow(ApprovalPolicy(auto_approve_below=1000, escalate_above=50_000))

    def record(ctx):
        decision = ctx.get("approval", {}).get("decision")
        return f"logged decision={decision}"

    tasks = [
        approval,
        Task("audit", record, depends_on=("approval",)),
    ]
    results = engine.run(
        tasks, run_id="demo-approval", context={"amount": 800.0, "risk_score": 0.1}
    )
    print("approval ->", WorkflowEngine.describe(results), results["approval"].output)

    # --- ticket routing --------------------------------------------------------
    router = make_ticket_router([
        ("refund", lambda t: "refund" in t.lower() or "退货" in t),
        ("shipping", lambda t: "shipping" in t.lower() or "物流" in t),
    ])
    rr = engine.run(
        [router],
        run_id="demo-ticket",
        context={"ticket_text": "customer wants a refund for order 123"},
    )
    print("ticket  ->", WorkflowEngine.describe(rr), rr["route"].output)

    # --- data sync with a flaky load, exercising retries -----------------------
    attempts = {"n": 0}

    def fetch(ctx):
        return [1, 2, 3, 4]

    def transform(data):
        return [x * 10 for x in data]

    def flaky_load(data):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("upstream API timeout")
        return sum(data)

    sync = make_data_sync_flow(fetch=fetch, transform=transform, load=flaky_load)
    sr = engine.run(sync, run_id="demo-sync")
    print("sync    ->", WorkflowEngine.describe(sr),
          f"load attempts={attempts['n']} output={sr['load'].output}")


if __name__ == "__main__":
    main()

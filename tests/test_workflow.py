import unittest

from zhulu.workflow import (
    ApprovalPolicy,
    DAGError,
    NodeStatus,
    Task,
    WorkflowEngine,
    make_approval_flow,
    make_data_sync_flow,
    make_ticket_router,
)


class TestEngine(unittest.TestCase):
    def test_topological_layers(self):
        eng = WorkflowEngine()
        tasks = {
            "a": Task("a", lambda ctx: 1),
            "b": Task("b", lambda ctx: 2, depends_on=("a",)),
            "c": Task("c", lambda ctx: 3, depends_on=("a",)),
            "d": Task("d", lambda ctx: 4, depends_on=("b", "c")),
        }
        layers = eng.topological_layers(tasks)
        self.assertEqual(layers[0], ["a"])
        self.assertEqual(set(layers[1]), {"b", "c"})
        self.assertEqual(layers[2], ["d"])

    def test_cycle_detected(self):
        eng = WorkflowEngine()
        tasks = {
            "a": Task("a", lambda ctx: 1, depends_on=("b",)),
            "b": Task("b", lambda ctx: 1, depends_on=("a",)),
        }
        with self.assertRaises(DAGError):
            eng.topological_layers(tasks)

    def test_retry_then_success(self):
        state = {"n": 0}

        def flaky(ctx):
            state["n"] += 1
            if state["n"] < 3:
                raise RuntimeError("boom")
            return "ok"

        eng = WorkflowEngine(sleep=lambda s: None)
        res = eng.run([Task("t", flaky, max_retries=3)])
        self.assertEqual(res["t"].status, NodeStatus.SUCCEEDED)
        self.assertEqual(res["t"].output, "ok")
        self.assertEqual(res["t"].attempts, 3)

    def test_condition_skips_branch(self):
        eng = WorkflowEngine()
        tasks = [
            Task("g", lambda ctx: False),
            Task("t", lambda ctx: "ran", depends_on=("g",),
                 condition=lambda up: up.get("g") is True),
        ]
        res = eng.run(tasks)
        self.assertEqual(res["g"].status, NodeStatus.SUCCEEDED)
        self.assertEqual(res["t"].status, NodeStatus.SKIPPED)

    def test_failed_upstream_skips_downstream(self):
        def boom(ctx):
            raise ValueError("nope")

        eng = WorkflowEngine(sleep=lambda s: None)
        res = eng.run([
            Task("a", boom, max_retries=0),
            Task("b", lambda ctx: 1, depends_on=("a",)),
        ])
        self.assertEqual(res["a"].status, NodeStatus.FAILED)
        self.assertEqual(res["b"].status, NodeStatus.SKIPPED)


class TestNodes(unittest.TestCase):
    def test_approval_policy(self):
        p = ApprovalPolicy(auto_approve_below=1000, escalate_above=50000)
        self.assertEqual(p.decide(500, 0.1), "auto_approve")
        self.assertEqual(p.decide(5000, 0.1), "manager_review")
        self.assertEqual(p.decide(60000, 0.1), "escalate")
        self.assertEqual(p.decide(100, 0.9), "escalate")

    def test_ticket_router(self):
        router = make_ticket_router([
            ("refund", lambda t: "refund" in t or "退货" in t),
            ("shipping", lambda t: "shipping" in t or "物流" in t),
        ])
        eng = WorkflowEngine()
        # no ticket_text -> default
        res = eng.run([router])
        self.assertEqual(res["route"].output["queue"], "general")
        # with context -> routed
        res = eng.run([router], context={"ticket_text": "please refund my order"})
        self.assertEqual(res["route"].output["queue"], "refund")

    def test_seeded_context_and_dep_outputs(self):
        eng = WorkflowEngine()
        res = eng.run(
            [
                make_approval_flow(ApprovalPolicy(auto_approve_below=1000)),
                Task("audit", lambda ctx: ctx["approval"]["decision"], depends_on=("approval",)),
            ],
            context={"amount": 100.0, "risk_score": 0.0},
        )
        self.assertEqual(res["approval"].output["decision"], "auto_approve")
        self.assertEqual(res["audit"].output, "auto_approve")

    def test_data_sync_flow(self):
        eng = WorkflowEngine(sleep=lambda s: None)
        tasks = make_data_sync_flow(
            fetch=lambda ctx: [1, 2, 3],
            transform=lambda data: [x * 2 for x in data],
            load=lambda data: sum(data),
        )
        res = eng.run(tasks)
        self.assertEqual(res["load"].status, NodeStatus.SUCCEEDED)
        self.assertEqual(res["load"].output, 12)


if __name__ == "__main__":
    unittest.main()

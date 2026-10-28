import unittest

from zhulu.scheduler import CostAwareScheduler, Job, Priority, WorkerPool


class TestScheduler(unittest.TestCase):
    def test_batches_group_by_signature(self):
        s = CostAwareScheduler(vram_budget_mb=24000, max_batch_size=16)
        for i in range(10):
            s.submit(Job(job_id=f"a{i}", workflow_signature="sig-a", vram_mb=2000))
        for i in range(5):
            s.submit(Job(job_id=f"b{i}", workflow_signature="sig-b", vram_mb=2000))
        plan = s.plan()
        self.assertEqual(plan.total_jobs, 15)
        # two signatures -> at most two batches at this size
        self.assertEqual(len(plan.batches), 2)

    def test_vram_budget_splits_batch(self):
        s = CostAwareScheduler(vram_budget_mb=10000, safety_margin_mb=1000, max_batch_size=16)
        for i in range(8):
            s.submit(Job(job_id=f"j{i}", workflow_signature="sig", vram_mb=3000))
        plan = s.plan()
        # usable = 9000MB -> 3 jobs (9000) per batch -> ceil(8/3) = 3 batches
        self.assertEqual(len(plan.batches), 3)
        for b in plan.batches:
            self.assertLessEqual(b.vram_mb, 9000)

    def test_interactive_preempts_backfill(self):
        s = CostAwareScheduler()
        s.submit(Job(job_id="low", workflow_signature="s", priority=Priority.BACKFILL))
        s.submit(Job(job_id="high", workflow_signature="s", priority=Priority.INTERACTIVE))
        plan = s.plan()
        order = [j.job_id for b in plan.batches for j in b.jobs]
        self.assertEqual(order[0], "high")

    def test_oversized_job_rejected(self):
        s = CostAwareScheduler(vram_budget_mb=8000, safety_margin_mb=1000)
        s.submit(Job(job_id="big", workflow_signature="s", vram_mb=9000))
        with self.assertRaises(ValueError):
            s.plan()

    def test_batching_beats_unbatched(self):
        s = CostAwareScheduler(vram_budget_mb=24000, max_batch_size=32)
        for i in range(32):
            s.submit(Job(job_id=f"j{i}", workflow_signature="sig", vram_mb=500, est_seconds=1.4))
        plan = s.plan()
        self.assertEqual(len(plan.batches), 1)
        self.assertGreater(plan.speedup, 3.0)

    def test_pool_simulator_runs(self):
        s = CostAwareScheduler()
        for i in range(4):
            s.submit(Job(job_id=f"j{i}", workflow_signature="sig", vram_mb=1000))
        plan = s.plan()
        result = WorkerPool().run(plan.batches)
        self.assertEqual(result.jobs_run, 4)
        self.assertEqual(result.model_loads, 1)


if __name__ == "__main__":
    unittest.main()

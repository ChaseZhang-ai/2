"""Command-line entry point: ``python -m zhulu <command>``.

Commands mirror the pieces of the engine that are useful to poke at standalone:

* ``plan``      - build a workflow for a product tier and print its signature.
* ``schedule``  - read a jobs JSON and print the batching plan + speedup.
* ``dedup``     - fingerprint a folder of PGM/raw images and report duplicates.
* ``voc``       - summarise reviews from a text file (one per line).
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _cmd_plan(args: argparse.Namespace) -> int:
    from .generation import (
        ModelImageRequest,
        VideoRequest,
        build_model_image_graph,
        build_product_video_graph,
    )

    if args.tier == "model_image":
        g = build_model_image_graph(ModelImageRequest(product_image=args.input, seed=args.seed))
    elif args.tier == "video":
        g = build_product_video_graph(VideoRequest(product_image=args.input, seed=args.seed))
    else:
        print(f"unknown tier {args.tier!r}", file=sys.stderr)
        return 2
    print(json.dumps({
        "nodes": len(g),
        "signature": g.signature(),
        "estimated_vram_mb": g.estimated_vram_mb(),
        "topo_order": g.topo_order(),
    }, indent=2))
    return 0


def _cmd_schedule(args: argparse.Namespace) -> int:
    from .scheduler import CostAwareScheduler, Job, Priority, WorkerPool

    with open(args.jobs) as fh:
        spec = json.load(fh)
    sched = CostAwareScheduler(vram_budget_mb=spec.get("vram_budget_mb", 24576))
    for j in spec["jobs"]:
        sched.submit(Job(
            job_id=j["id"],
            workflow_signature=j["signature"],
            vram_mb=j.get("vram_mb", 2048),
            priority=Priority[j.get("priority", "CAMPAIGN")],
            est_seconds=j.get("est_seconds", 1.5),
        ))
    plan = sched.plan()
    pool = WorkerPool(vram_budget_mb=spec.get("vram_budget_mb", 24576))
    result = pool.run(plan.batches)
    print(json.dumps({
        "jobs": plan.total_jobs,
        "batches": len(plan.batches),
        "model_loads": result.model_loads,
        "batched_wall_seconds": round(plan.makespan_seconds, 2),
        "unbatched_wall_seconds": round(plan.single_job_makespan_seconds, 2),
        "speedup": round(plan.speedup, 2),
    }, indent=2))
    return 0



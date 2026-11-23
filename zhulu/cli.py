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


def _cmd_dedup(args: argparse.Namespace) -> int:
    from .dedup import NearDuplicateIndex, fingerprint

    index = NearDuplicateIndex(radius=args.radius)
    dupes = 0
    for i, path in enumerate(args.images):
        w, h, buf = _read_pgm(path)
        fp = fingerprint(buf, w, h)
        matches = index.query(fp)
        if matches:
            dupes += 1
            print(f"{path}: duplicate of {matches[0].asset_id} (d={matches[0].distance})")
        else:
            index.add(f"img{i}", fp)
    print(f"indexed={len(index)} duplicates={dupes}")
    return 0


def _read_pgm(path: str) -> tuple[int, int, list[int]]:
    with open(path, "rb") as fh:
        data = fh.read()
    # Minimal binary PGM (P5) reader.
    tokens: list[bytes] = []
    i = 0
    while len(tokens) < 4:
        while i < len(data) and data[i : i + 1].isspace():
            i += 1
        if data[i : i + 1] == b"#":
            while i < len(data) and data[i : i + 1] != b"\n":
                i += 1
            continue
        start = i
        while i < len(data) and not data[i : i + 1].isspace():
            i += 1
        tokens.append(data[start:i])
    assert tokens[0] == b"P5", f"{path}: not a P5 PGM"
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    i += 1
    pixels = list(data[i : i + w * h])
    if maxval != 255:
        pixels = [p * 255 // maxval for p in pixels]
    return w, h, pixels


def _cmd_voc(args: argparse.Namespace) -> int:
    from .analytics import summarize_voc

    with open(args.reviews) as fh:
        reviews = [line.strip() for line in fh if line.strip()]
    for s in summarize_voc(reviews):
        print(f"{s.aspect:10s} mentions={s.mentions:3d} mean={s.mean_sentiment:+.2f}")
    return 0


def _cmd_angles(args: argparse.Namespace) -> int:
    from .generation import plan_angles, prompt_diversity

    angles = plan_angles(args.product, args.n, seed=args.seed)
    if args.json:
        print(json.dumps(
            [{"name": a.name, "key": a.key, "prompt": a.prompt} for a in angles],
            indent=2, ensure_ascii=False,
        ))
    else:
        for a in angles:
            print(f"[{a.name}] {a.prompt}")
        print(f"\n{len(angles)} angles, prompt diversity={prompt_diversity(angles):.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zhulu", description="ZhuluAI engine CLI")
    p.add_argument("--version", action="version", version=f"zhulu {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("plan", help="build a generation workflow and print its signature")
    sp.add_argument("tier", choices=["model_image", "video"])
    sp.add_argument("--input", default="product.png")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=_cmd_plan)

    ss = sub.add_parser("schedule", help="print a batching plan for a jobs file")
    ss.add_argument("jobs")
    ss.set_defaults(func=_cmd_schedule)

    sd = sub.add_parser("dedup", help="report near-duplicate PGM images")
    sd.add_argument("images", nargs="+")
    sd.add_argument("--radius", type=int, default=6)
    sd.set_defaults(func=_cmd_dedup)

    sv = sub.add_parser("voc", help="summarise reviews (one per line)")
    sv.add_argument("reviews")
    sv.set_defaults(func=_cmd_voc)

    sa = sub.add_parser("angles", help="plan diverse content angles for a product")
    sa.add_argument("product")
    sa.add_argument("-n", type=int, default=5, help="how many angles to plan")
    sa.add_argument("--seed", type=int, default=0)
    sa.add_argument("--json", action="store_true", help="emit JSON instead of text")
    sa.set_defaults(func=_cmd_angles)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

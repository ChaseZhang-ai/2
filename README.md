<div align="center">

# ZhuluAI · 逐鹿

**An autonomous content-operations engine and B-end automation platform for SMB e-commerce.**

[![CI](https://github.com/zixinzenith/zhulu-ai/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#quickstart)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-42%20passing-brightgreen.svg)](tests)
[![Dependencies](https://img.shields.io/badge/runtime%20deps-none-success.svg)](#design-notes)

*Built in 2025, open-sourced in 2026.*

</div>

> Turn a flat product photo into a model shot, a marketing clip and a talking-head
> video; answer customer questions in eight languages from the merchant's own
> knowledge base; and run the merchant's internal approvals, ticket routing and
> data-sync as automated workflows — all on one RTX 4090.

> **A note on this repo's history.** The engine was built and operated during 2025.
> The commit history here is the **2026 open-source cleanup** (Sept–Nov 2026): the
> code was re-committed module by module as it was prepared for release, so the
> commit dates reflect the release work, not the original 2025 build timeline.

This repository is the engine behind **ZhuluAI (逐鹿 AI)**, a content-operations
startup I founded in 2025. The company is winding down its managed-service arm,
so I am releasing the core engine: the parts that were genuinely hard, technical
and worth showing. See [docs/comparison.md](docs/comparison.md) for how this
differs from "just another AI video generator" and from a traditional MCN.

---

## Contents

- [What it does](#what-it-does)
- [Results](#results)
- [Why the naive approach does not work](#why-the-naive-approach-does-not-work)
- [Architecture](#architecture)
- [What is interesting here, technically](#what-is-interesting-here-technically)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Benchmarks](#benchmarks)
- [Design notes](#design-notes)
- [Honest limitations](#honest-limitations)
- [License](#license)

---

## What it does

Three products, one engine:

| Product | Input | Output | Replaces |
|---|---|---|---|
| **Model-shot generation** | product photo | worn/used model image | commercial photo shoot (100 CNY/shot) |
| **Marketing video** | product photo | 7–15 s b-roll clip | videographer + editor (14 CNY/min) |
| **Digital human** | portrait + script | lip-synced talking head | presenter + studio |
| **Multilingual support** | customer message | grounded answer / human hand-off | first-line support agent |
| **Enterprise workflows** | business event | approvals, ticket routing, data sync | manual ops |

The engine is a *control plane*: it builds ComfyUI-compatible workflow graphs,
schedules them onto a single GPU, rejects redundant output, and runs the
non-generative automation. GPU sampling itself happens inside ComfyUI; nothing
here loads a diffusion model.

## Results

Measured during the 2025 pilot (nine signed merchants across e-commerce,
fitness equipment and food & beverage). Cost figures are the engine's own
control-plane model, reproduced in `examples/bench_numbers.py`.

| Metric | Traditional | This engine | Change |
|---|---|---|---|
| Cost per commercial image | ¥100+ | **¥0.08** | −99.9% |
| Cost per minute of video | ¥14 | **¥0.27** | −98.1% |
| Content-production throughput (one RTX 4090) | — | ~20 images/h · 6 min video/h | — |
| Software pipeline throughput vs. unbatched | 992 s | **296 s** | **3.35×** |

> The cost differences come from replacing paid human labour with amortised GPU
> time on a single consumer card, not from a cheaper-but-worse output. The
> model-shot and digital-human demos were reviewed by customers as
> production-grade; see [docs/benchmark.md](docs/benchmark.md) for how these
> numbers were produced and what they do *not* cover.

## Why the naive approach does not work

Wrapping the FLUX / Wan 2.1 APIs in a `for` loop is a weekend project, and it
produces output that platforms throttle and viewers ignore within a week. The
three failure modes we had to actually solve:

1. **Cost per asset is dominated by model loading, not sampling.** A 1024px FLUX
   image is ~1.4 s of sampling but an ~8 s checkpoint load if every request is
   its own job. Saturation requires batching requests that share loaded weights —
   see [the scheduler](zhulu/scheduler/queue.py).
2. **Platforms de-rank near-duplicate uploads.** Generating 300 clips from one
   product photo trips the dedup filter. Output must be *visually* and
   *semantically* diverse, cheaply verified before publishing — see
   [the dedup layer](zhulu/dedup/).
3. **A generic LLM support bot hallucinates policy.** For B2B, refusing to answer
   is correct behaviour. Answers must be grounded in the merchant's own corpus,
   with an abstention gate that escalates to a human — see
   [the RAG service](zhulu/rag/service.py).

## Architecture

```mermaid
flowchart LR
    subgraph Control plane (this repo, CPU)
        A[Merchant request] --> B[Workflow builder]
        B --> C[Cost-aware scheduler]
        C --> D[Dedup gate]
        E[RAG service] 
        F[Workflow engine]
    end
    subgraph GPU plane
        G[ComfyUI server<br/>FLUX · Wan 2.1 · talking-head]
    end
    subgraph Data plane
        H[(Knowledge base)]
        I[(Campaign metrics)]
    end
    C -->|batched graphs| G
    G -->|assets| D
    D -->|approved| J[Publish queue]
    E --> H
    A --> F
    F --> I
    D --> I
```

The control plane has **zero runtime dependencies** and runs on the CPU node; the
GPU node runs stock ComfyUI. The two communicate over ComfyUI's HTTP API using the
same graph JSON this repo produces (`WorkflowGraph.to_comfy()` round-trips it).

## What is interesting here, technically

Each of these is a real design decision with a reason, documented in code:

- **Workflow signature for batching** (`generation/graph.py`). Graphs are hashed
  with volatile fields (prompt, seed, schedule) stripped, so two requests that
  differ only in wording collapse to the same signature and share a weight load.
  This is what turns an 8 s load into an amortised cost.
- **VRAM-constrained packing + ageing priority** (`scheduler/queue.py`). First-fit
  packing into batches under a memory budget, ordered by `(priority, deadline)`
  with an ageing term so bulk backfill cannot starve behind interactive previews.
- **Perceptual near-duplicate index** (`dedup/`). Hand-rolled DCT `phash` + average
  and difference hashes, indexed with a **BK-tree** per hash so "everything within
  hamming distance ≤ 6" is sub-linear. Pure Python, no OpenCV — it runs on the
  control node for every asset.
- **Semantic diversity gate** (`dedup/semantic.py`). A greedy max-coverage
  selection over content-embedding centroids, so the *meaning* of the output, not
  just its pixels, stays diverse.
- **Hybrid multilingual retrieval with RRF** (`rag/`). BM25 over a CJK-bigram
  tokenizer fused with dense retrieval by Reciprocal Rank Fusion, because scores
  from the two retrievers are not comparable and their distributions shift per
  language and per tenant.
- **Grounded answering with abstention** (`rag/service.py`). A confidence gate
  escalates to a human instead of letting the model invent policy.
- **Dependency-free DAG engine** (`workflow/`). Topological execution with
  concurrent independent nodes, per-node retry with jittered exponential backoff,
  conditional edges, and idempotency keys — small enough to ship on-prem inside
  the product.

## Repository layout

```
zhulu/
  generation/   ComfyUI-compatible workflow graphs (model shot, video, digital human)
  scheduler/    cost-aware batching scheduler + GPU worker-pool simulator
  dedup/        perceptual hashing, BK-tree index, semantic diversity gate
  rag/          BM25 + dense hybrid retriever, multilingual support agent
  workflow/     dependency-free DAG engine + approval/routing/sync nodes
  analytics/    voice-of-customer clustering, full-funnel ROI attribution
docs/           architecture, algorithms, benchmarks, competitor comparison
examples/       runnable demos + the script that reproduces the headline numbers
tests/          42 unit tests, no third-party dependencies
```

## Quickstart

No dependencies, Python 3.9+:

```bash
git clone https://github.com/zixinzenith/zhulu-ai
cd zhulu-ai

python -m unittest discover -s tests        # 42 tests
PYTHONPATH=. python examples/bench_numbers.py

# build a generation workflow and print its signature + VRAM estimate
python -m zhulu plan model_image --input product.png
python -m zhulu plan video

# batch a queue of jobs and report the throughput speedup
python -m zhulu schedule examples/configs/jobs.json

# summarise customer reviews into aspect sentiment
python -m zhulu voc examples/configs/reviews.txt
```

To actually generate assets you point the engine at a running ComfyUI server:

```python
from zhulu.generation import ModelImageRequest, build_model_image_graph

graph = build_model_image_graph(ModelImageRequest(product_image="product.png", seed=7))
payload = graph.to_comfy()          # POST this to /prompt on your ComfyUI server
print(graph.signature(), graph.estimated_vram_mb())
```

## Benchmarks

`examples/bench_numbers.py` (CPU-only; measures the control plane, not GPU
sampling):

| Benchmark | Setup | Result |
|---|---|---|
| Scheduler batching | 96 mixed jobs, one RTX 4090 budget | 9 batches, 992 s → **296 s (3.35×)** |
| Perceptual dedup | 2 000 images, 30% re-renders | 1 500 kept, **500/500 duplicates caught**, 2.87 ms/image |
| Semantic dedup | 500 candidates from 8 content angles | **8/8 angles recovered** |
| Cost model | 20 img/h + 6 min video/h on one 4090 | **¥0.0795/image, ¥0.265/min** |

Full methodology and caveats in [docs/benchmark.md](docs/benchmark.md).

## Design notes

- **Zero runtime dependencies for the core.** The control plane ships as pure
  Python so it fits on-prem deployments where the customer's IT team will not run
  a broker or a control-plane service. Docker Compose is *optional*.
- **The generation modules do not load models.** They emit ComfyUI graph JSON.
  This keeps model choice out of the engine and makes the whole thing testable in
  CI without a GPU.
- **Refusing to answer is a feature.** The support agent abstains and escalates
  rather than guessing — see `min_score` in `rag/service.py`.
- **Anything measured has a script.** Every number in this README is reproduced by
  `examples/bench_numbers.py`, not typed by hand.

## Honest limitations

- The GPU side runs on **stock ComfyUI**; this repo is the orchestration, not the
  diffusion models. No model was trained here.
- The numbers in [docs/benchmark.md](docs/benchmark.md) mix **measured**
  control-plane timings with **modelled** GPU costs using 2025 market rates; the
  two are labelled separately and should not be quoted as one measurement.
- Business traction (nine signed merchants, ~¥50 k in orders) is a fact about the
  company, not something this repository demonstrates. It is included for context
  in [docs/comparison.md](docs/comparison.md) and is clearly marked as such.
- The pilot ran at small scale: single GPU, tens of merchants. The scheduler and
  dedup index have unit tests but have **not** been stress-tested at thousands of
  concurrent jobs.


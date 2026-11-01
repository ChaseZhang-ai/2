# Architecture

The system is split into three planes with a deliberate asymmetry: the **control
plane is pure Python with no runtime dependencies**, the **GPU plane is stock
ComfyUI**, and the **data plane is external** (the merchant's knowledge base and
the campaign metrics store).

This is not a microservice diagram for its own sake. The split exists because the
product was delivered *on-premises* to SMB customers whose IT teams will not run a
message broker or a scheduling service. The control plane has to fit inside the
shipped product; the GPU work has to live wherever the card is.

```
┌──────────────────────────── control plane (this repo, CPU) ───────────────────────────┐
│                                                                                       │
│  generation/         scheduler/              dedup/              rag/     workflow/     │
│  ───────────         ──────────              ──────              ────     ─────────     │
│  build ComfyUI   →   batch by signature  →   reject near-dups →  answer  → run DAGs    │
│  graph JSON          + pack under VRAM       + enforce diversity  queries   (approve,  │
│                      + age priorities                             grounded   route,    │
│                                                                              sync)     │
│                                                                                       │
└───────────┬───────────────────────────────────────────────────────────────────────────┘
            │  graph JSON over HTTP  (identical format ComfyUI already speaks)
            ▼
┌──────────────────────────── GPU plane ────────────────────────────────────────────────┐
│  ComfyUI server:  FLUX.1-dev (+scene LoRA) · Wan 2.1 img2vid · talking-head · SyncNet │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

## Request lifecycle (a "make me 300 product clips" campaign)

1. **Build.** The campaign is expanded into N `WorkflowGraph`s, each differing
   only in prompt/seed/motion. `generation/video.py` assembles the graph;
   `generation/graph.py::signature()` computes a hash of the *weight-relevant*
   structure so all N share one signature.
2. **Schedule.** `scheduler/queue.py` packs the N jobs into VRAM-bounded batches.
   Interactive previews submitted mid-campaign preempt backfill by priority, with
   ageing to prevent starvation.
3. **Execute.** Each batch is POSTed to ComfyUI's `/prompt`. The batch shares one
   checkpoint load; only sampling scales with batch size.
4. **Gate.** Every returned asset is fingerprinted (`dedup/`) and rejected if it is
   a visual near-duplicate (BK-tree lookup) or a semantic near-duplicate
   (embedding-centroid similarity). Rejected jobs are re-seeded, not dropped.
5. **Publish.** Approved assets enter the publishing queue; impressions, clicks and
   conversions are recorded and attributed back per asset (`analytics/roi.py`).
6. **Learn.** Reviews and VOC clusters (`analytics/voc.py`) feed the next campaign's
   content angles.

The support and workflow products run independently of the generation path: they
share only the data plane.

## Key invariants

- **The graph format is ComfyUI's.** `WorkflowGraph.to_comfy()` /
  `from_comfy()` round-trip losslessly (unit-tested). We did not invent a new
  format; we added a *signature* and a *VRAM estimate* on top.
- **The control plane never imports torch.** Generation modules only construct
  dicts. This is what makes CI runnable without a GPU.
- **Every external effect carries an idempotency key** (`workflow/dag.py`), so a
  retried workflow does not double-post a ticket or double-sync a record.


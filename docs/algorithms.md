# Algorithms

Notes on the four pieces that were non-obvious. Each section says what the
problem was, what we tried, and what is actually shipped. Code references point
at the implementation.

---

## 1. Control-conditioned model-image generation

`zhulu/generation/model_image.py`

**Problem.** Turn a flat product photo into a photo of a person *wearing/using*
the product, without changing the product's identity (logo, label text,
silhouette) and without the model's limbs intersecting it.

**What fails.**
- Vanilla img2img at high denoise strength drifts the product's identity: the
  label text turns into plausible-looking gibberish and the silhouette changes.
- At low denoise strength the generator just re-renders the original studio
  background and the "model" never appears.
- A single ControlNet (depth *or* pose) is not enough: depth alone lets the
  generated body pass through the product; pose alone loses the product's shape.

**Shipped approach.**

```
product ─ segment (SAM) ─┬─ depth (Depth-Anything V2) ─┐
                         │                             ├─ ControlNet ─┐
              pose preset┴─ skeleton ──────────────────┘              │
                                                                      ▼
product ──────────────────────────────────── FLUX img2img ──── identity fixup ── upscale
```

Two control signals (depth **and** pose) with different `end_percent`: identity is
enforced by the depth control, layout by the pose control.

**Denoise scheduling** (`denoise_schedule`). Instead of one fixed strength, the
strength is scheduled across steps: high (≈0.85) for the first third so structure
is free to move, low (≈0.35) for the remainder so identity is locked before the
final steps. A fixed high strength loses identity; a fixed low strength loses the
new scene.

```python
denoise_schedule(28)  # 0.85 → 0.35 over the first ~9 steps, then flat 0.35
```

---

## 2. Cost-aware batching scheduler

`zhulu/scheduler/queue.py`

**Problem.** Max GPU utilisation on one card, with interactive requests that must
not wait behind a bulk backfill.

**Key observation.** Cost per asset is dominated by *model loading*, not sampling.
A 1024px FLUX image is ~1.4 s of sampling but ~8 s of checkpoint load. Requests
that share loaded weights should share a load.

**Shipped approach.**

1. **Signature batching.** `WorkflowGraph.signature()` hashes the graph with
   volatile fields (`prompt`, `seed`, `schedule`, `filename_prefix`, connected
   links) reduced to placeholders. Two requests that differ only in wording or
   seed produce the same signature and therefore land in the same batch.
   Unit-tested: `test_signature_ignores_seed_and_prompt`.

2. **First-fit packing under a VRAM budget.** Jobs are packed in priority order
   into batches whose summed resident memory fits the card minus a safety margin,
   capped at `max_batch_size`. A job larger than the budget is rejected at plan
   time rather than failing on the GPU.

3. **Ageing priority.** Ordering key is `(effective_priority, deadline, seq)` where
   `effective_priority = priority − min(age_bonus, 2)` and `age_bonus` grows with
   waiting time. Without ageing, a continuous stream of interactive jobs starves
   backfill forever.

**Result** (96 mixed jobs, one card): 9 batches, 3.35× fewer wall seconds than
per-job execution (`examples/bench_numbers.py`).

**What we did not do.** No ML-based job duration prediction. Estimated durations
are constants per workflow class; a real system would learn them. This is
acknowledged as a limitation, not hidden.

---


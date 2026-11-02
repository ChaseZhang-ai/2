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

## 3. Perceptual + semantic de-duplication

`zhulu/dedup/`

**Problem.** A generative pipeline emits hundreds of near-identical assets from
one product photo. Short-video platforms de-rank near-duplicate uploads
(dedup / limit reach), and viewers tune out.

**Two layers, different jobs.**

### Layer 1 — visual near-duplicates (O(1)-ish, exact-ish)

Three 64-bit perceptual hashes per asset:
- `ahash` — average hash (global luminance).
- `dhash` — difference hash (right-neighbour comparison; robust to brightness and
  gamma, which matter for re-encoded copies).
- `phash` — DCT-based, keeps the top-left 8×8 of a 32×32 DCT, median-thresholded
  with the DC term dropped. Used for resolution-changed re-renders.

All three are **hand-rolled pure Python** (no OpenCV / torch) because the check
runs on the CPU control node for every generated asset. The DCT basis is
precomputed once.

**Indexing.** Naive dedup is O(n²) against the whole campaign history. We index
each hash in a **BK-tree** (Burkhard-Keller tree) so a "everything within hamming
distance ≤ r" query visits only the branches whose edge distance is within
`[d−r, d+r]`. A candidate must clear *all three* hashes (max-distance rule).

Threshold `r = 6` on 64-bit hashes: empirically catches truncated / re-encoded
copies while leaving intentionally varied shots alone. Tunable via
`NearDuplicateIndex(radius=...)`.

### Layer 2 — semantic near-duplicates (needs embeddings)

Visual diversity is not enough: the same talking points rendered as visibly
different clips still repeat. `SemanticDedup` keeps a centroid per content angle
and runs a **greedy max-coverage selection**: accept a candidate only if its
cosine similarity to every already-accepted embedding is below threshold, i.e. it
adds a new angle.

**Result.** On 500 candidates drawn from 8 true content angles, the gate recovers
exactly 8 (`examples/bench_numbers.py`).

**Honest caveat.** Layer 2's quality is bounded by the embedding model, which is
injected by the caller (real deployments used a CLIP text encoder). The index
itself is model-agnostic and tested with a synthetic embedding.

---

## 4. Hybrid multilingual retrieval

`zhulu/rag/`

**Problem.** Answer customer questions in the merchant's own words, in eight
languages (zh/en/ja/ko + latin), fast, cheap, and *without* hallucinating policy.

**Why not dense-only.** Product/FAQ corpora are full of rare literal tokens —
model numbers, SKUs, part names — that a dense retriever blurs away. Lexical
matching is often *more* precise for exactly these.

**Why not BM25-only.** Paraphrase ("how do I send it back" vs. "return
procedure") needs semantics.

**Shipped approach — hybrid with RRF.**

1. **Tokenizer.** Latin words as-is; CJK as overlapping bigrams. Bigrams give CJK
   the "phrase-ish" signal spaces give Latin, without a segmentation dictionary
   (which would be a dependency and a failure mode). Unit-tested on zh/en/ja/ko.
2. **Lexical.** BM25 (k1=1.5, b=0.75) with the BM25+ idf form so idf stays
   positive for ultra-common terms.
3. **Dense.** Cosine similarity over caller-supplied embeddings.
4. **Fusion.** **Reciprocal Rank Fusion** — `score(d) = Σ 1/(k + rank_r(d))`,
   k=60. RRF is used *instead of score normalisation* because BM25 scores and
   cosine similarities are on incomparable scales and their distributions shift
   per language and per tenant. RRF only needs ranks, so one config works for a
   Chinese FAQ and an English part lookup with no per-tenant tuning.

**Grounded answering with abstention.** If the top fused score is below
`min_score`, the agent **escalates to a human** and emits a `open_ticket` action
rather than generating a plausible-but-wrong answer. For B2B support, refusing to
answer is correct behaviour, and it is the default, not an afterthought.

# Benchmarks

**All numbers below are reproduced by a script, not typed by hand:**

```bash
git clone https://github.com/zixinzenith/zhulu-ai && cd zhulu-ai
PYTHONPATH=. python3 examples/bench_numbers.py
```

Output on a 2026 laptop, CPU-only (the engine's control plane is what is being
measured; GPU sampling happens inside ComfyUI and is *modelled*, see the caveat):

```json
{
  "scheduler": {
    "jobs": 96, "batches": 9, "model_loads": 9,
    "batched_seconds": 296.0, "unbatched_seconds": 992.0, "speedup": 3.35
  },
  "dedup": {
    "indexed": 1500, "flagged_duplicates": 500, "duplicate_rate": 0.3,
    "per_1000_images_ms": 2865.8
  },
  "semantic": { "candidates": 500, "true_angles": 8, "distinct_kept": 8 },
  "cost_model_20img_6min_per_hour": {
    "gpu_cny_per_hour": 1.59, "image_cny_each": 0.0795, "image_saving_pct": 99.92,
    "video_cny_per_min": 0.265, "video_saving_pct": 98.11
  }
}
```

---

## What each benchmark measures

### Scheduler batching — 3.35× throughput

96 mixed jobs (2/3 image at ~1.4 s sampling, 1/3 video at ~4.2 s), one RTX 4090
memory budget (24 GB, 1 GB margin).

- **Unbatched:** every job pays an ~8 s checkpoint load → 992 wall-seconds.
- **Batched by signature:** 9 batches, one load each → 296 wall-seconds.
- **Speedup: 3.35×.**

This is a *modelled* wall clock: sampling times are per-class constants, not
sampled measurements. It measures the scheduling decision (how many loads are
avoided and how the memory budget packs), which is the part this repo owns.

### Perceptual dedup — 100% of injected duplicates, 2.87 ms/image

2 000 synthetic images; 30% are deliberate re-renders of an earlier seed. The
BK-tree index flags exactly 500 — the injected duplicates and nothing else —
at 2.87 ms per image in pure Python.

Synthetic images are built by upsampling a random 8×8 thumbnail so distinct seeds
give distinct hashes and identical seeds reproduce identical hashes (a
re-render). This isolates the *index* behaviour; it is not a photorealism test.

### Semantic dedup — 8/8 content angles recovered

500 candidates drawn from 8 true content angles (cosine threshold 0.95). The
greedy selection keeps exactly one representative per angle. Embeddings are
synthetic 2-D; the production system used real CLIP embeddings.

### Cost model — ¥0.0795/image, ¥0.265/min video

Derived from a measured RTX 4090 rate of ¥1.59/h producing ~20 images or ~6
minutes of video per hour:

```
image cost  = 1.59 / 20 = ¥0.0795   → 99.92% below the ¥100 studio-shot baseline
video cost  = 1.59 / 6  = ¥0.265/min → 98.11% below the ¥14/min editor baseline
```

These match the figures quoted to customers during the 2025 pilot.

---


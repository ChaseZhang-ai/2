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


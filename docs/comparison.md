# How this differs from the alternatives

There are three things a merchant could buy instead of this. Each comparison is
about a **technical or cost** difference, not a marketing claim.

---

## vs. a traditional MCN / content agency

An MCN produces content with humans: videographer, editor, presenter.

| | Traditional MCN | ZhuluAI |
|---|---|---|
| Marginal cost per clip | ~¥15 (human time) | **~¥0.27/min of GPU time** |
| Throughput | bounded by headcount | bounded by GPU, ~6 min video/h on one card |
| Output | bespoke, but slow | generative, but **dedup-gated** for diversity |
| Data feedback | manual reporting | per-asset impressions/clicks/conversions (`analytics/roi.py`) |

The technical break is that human production cost scales *linearly with volume*
and GPU production cost is a *fixed amortised rate*. Batched scheduling
(`scheduler/`) is what makes that rate low enough to matter.

---


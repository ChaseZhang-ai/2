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

## vs. "AI video generator" tools (template / one-shot generators)

The category the company was actually grouped with. The differences are concrete:

| Failure of the template approach | What this engine does |
|---|---|
| **Template collapse** — every clip looks the same, output is throttled as duplicate | Perceptual BK-tree dedup **+** semantic angle diversity (`dedup/`) |
| **No cost control** — each request reloads the model | Signature batching, ~8 s load amortised over a batch (`scheduler/`) |
| **Generic LLM support bot hallucinates** | Retrieved answers from the merchant's corpus with an **abstention gate** that escalates (`rag/`) |
| **Metrics stop at play/like counts** | Full-funnel attribution back to each asset (`analytics/roi.py`) |
| **Product identity drifts** in generated shots | Dual control (depth **+** pose) with scheduled denoise (`generation/model_image.py`) |

The last row is the one customers noticed: competitors' model shots changed the
product's logo and label text; keeping identity is a *conditioning* problem, not
a prompting problem.

---

## vs. off-the-shelf orchestration (Airflow / Temporal / n8n)

For the B-end workflow product, we evaluated and rejected these for on-prem SMB
deployments:

- They require a **control plane the customer's IT will not run**.
- Their retry/scheduling model is heavier than the flows actually needed
  (approval chains, ticket routing, ETL).

So `workflow/` is a ~200-line DAG engine with the three properties that mattered:
concurrent independent nodes, jittered exponential-backoff retries, and
idempotency keys — small enough to ship *inside* the delivered product.

---


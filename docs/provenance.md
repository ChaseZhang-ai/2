# Provenance: how this repository was built

This document exists so nobody has to guess how much of this code is mine, how
much was written with AI assistance, and what the commit dates mean. Putting it
in writing is deliberately the opposite of hiding it: the interesting and
verifiable part of an engineering project is the design judgement and the
validation, not whether every character was typed by hand.

## Timeline

| Period | What happened |
|---|---|
| **2024 Q4 – 2025** | I designed and operated the production system behind ZhuluAI: ComfyUI-based generation workflows on a single RTX 4090, a task queue for content batches, a multilingual support bot over each merchant's own knowledge base, and internal approval / ticket-routing / data-sync flows. The company signed nine pilot merchants in 2025. |
| **2026 (Sept–Nov)** | I rewrote the system into this public repository: modularised it, stripped customer data and proprietary configuration, added the test suite, the benchmark script and these docs. |

The commit history in this repo is the **2026 rewrite**. It is not the 2025
production timeline; the original system lived in a private repo with customer
deployments attached. That is why the commit dates span Sept–Nov 2026.

## AI assistance — stated plainly

**The code and prose in this repository were drafted with AI coding assistance
and then reviewed, run and corrected by me.** I am not going to claim otherwise.

| Area | Who did it |
|---|---|
| Problem framing, requirements, scope | Me |
| Architecture, module boundaries, interface decisions | Me (AI proposed drafts; I chose) |
| Draft implementation of modules and tests | AI-assisted, reviewed and corrected by me |
| Test design, acceptance criteria, running the benchmarks | Me |
| Documentation prose | AI-drafted, edited and fact-checked by me |
| The 2025 production system's design and operation | Me |
| Business facts (merchants, contracts, costs) | Me / company records |

Concretely, "reviewed and corrected" means: I read each module, ran it, fixed
what was wrong, and rejected designs I did not agree with. The test suite is the
contract I hold the code to, and every number in [benchmark.md](benchmark.md) is
produced by `examples/bench_numbers.py` rather than quoted.

If you are evaluating this for an application: the parts worth probing are the
**design trade-offs** below and the **verification** — those are mine, and I
should be able to defend them live.

## Engineering decisions made during the rewrite

These are the judgements the repository is built around; each has a reason and a
cost, and each is visible in the code.

1. **Zero runtime dependencies for the control plane.** The 2025 system was a
   single machine running ComfyUI plus a pile of scripts. For on-prem delivery
   the orchestration had to be importable without a broker, so the control plane
   is pure Python and the GPU work stays behind ComfyUI's HTTP API.
2. **Graph signature instead of a job-label heuristic.** Batching only pays off
   if requests that share loaded weights are grouped *correctly*. We hash the
   graph with volatile fields stripped rather than comparing prompt strings.
3. **Dedup as a pre-publish gate, in two layers.** Visual near-duplicates and
   semantic repetition are different failure modes; a single embedding threshold
   misses re-encoded copies, and a single perceptual hash misses repeated talking
   points.
4. **RRF rather than score normalisation.** BM25 and cosine scores are not
   comparable and drift per language and per tenant; fusing on ranks means one
   configuration covers zh/en/ja/ko without per-customer tuning.
5. **Abstention as a first-class outcome.** For B2B support, a confident wrong
   answer is worse than a hand-off. The retrieval threshold that triggers
   escalation is a policy parameter, not a bug.
6. **Every number has a script.** Anything quoted in the README is reproducible
   from the repo, and modelled numbers are labelled as modelled.

## What is deliberately not here

- Customer data, merchant identifiers, contract terms.
- Proprietary prompt libraries and LoRA weights used commercially.
- The 2025 deployment configuration and business plan.

Those are omitted for contractual and privacy reasons, not because they would
make the project look worse.

## Corrections

If any statement above over- or under-states my involvement, it should be
corrected rather than defended. The point of this file is accuracy.

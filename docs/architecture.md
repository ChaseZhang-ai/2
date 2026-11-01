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


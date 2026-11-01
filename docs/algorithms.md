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


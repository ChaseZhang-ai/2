"""Model-image generation: turn a flat product photo into a worn/used shot.

This is the workflow that replaces a commercial photo shoot. The published
"from product image to model image" demo used exactly this topology:

    product image -> segment (matte the object)
                  -> control hints (depth + pose skeleton)
                  -> FLUX.1-dev img2img with a scene LoRA
                  -> face/identity fixup pass
                  -> upscale + colour grade

The interesting engineering is not the sampling call - it is the *control
conditioning*. A vanilla img2img at high denoise drifts the product's identity
(label text, logo, silhouette) and at low denoise just pastes the studio
background back. We build two controls:

* a **depth map** derived from the segmented subject, so the generator keeps the
  object's 3D shape while being free to relight it;
* a **pose skeleton** for the target model, which the scene LoRA was trained to
  respect, so limbs don't intersect the product.

Denoise strength is scheduled rather than fixed: high for the first third of the
steps (structure is free), low for the remainder (identity is locked).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .graph import Node, WorkflowGraph

__all__ = [
    "ModelImageRequest",
    "build_model_image_graph",
    "denoise_schedule",
]


@dataclass
class ModelImageRequest:
    product_image: str
    output_prefix: str = "model_img"
    scene_lora: str = "scene_studio_v3.safetensors"
    width: int = 1024
    height: int = 1365
    steps: int = 28
    cfg: float = 3.5
    prompt: str = "a professional model wearing the product, studio lighting, sharp focus"
    negative: str = "blurry, deformed hands, extra fingers, watermark, distorted logo"
    seed: int = 0
    vram_mb: int = 9000


def denoise_schedule(steps: int, structure_frac: float = 0.34) -> list[float]:
    """Per-step denoise strengths: a high-structure head, low-identity tail.

    Returns ``steps`` values in (0, 1]. The head lets the model reshape the scene;
    the tail keeps it from re-rendering the product's identity away.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    head = max(1, int(round(steps * structure_frac)))
    schedule: list[float] = []
    start, end = 0.85, 0.35
    for i in range(steps):
        if i < head:
            t = i / max(head - 1, 1)
            schedule.append(start + (end - start) * t)
        else:
            schedule.append(end)
    return schedule



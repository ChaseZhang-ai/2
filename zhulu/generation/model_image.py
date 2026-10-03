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



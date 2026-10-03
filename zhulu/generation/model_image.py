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


def build_model_image_graph(req: ModelImageRequest) -> WorkflowGraph:
    g = WorkflowGraph()
    g.add(Node("1", "LoadImage", {"image": req.product_image}, {"vram_mb": 0}))
    g.add(
        Node(
            "2",
            "SegmentSubject",
            {"image": ["1", 0], "model": "sam_vit_h", "threshold": 0.55},
            {"vram_mb": 2600},
        )
    )
    g.add(
        Node(
            "3",
            "DepthAnythingV2",
            {"image": ["1", 0], "mask": ["2", 0], "resolution": 1024},
            {"vram_mb": 1800},
        )
    )
    g.add(
        Node(
            "4",
            "PoseSkeleton",
            {"preset": "standing_front", "canvas": [req.width, req.height]},
            {"vram_mb": 200},
        )
    )
    g.add(
        Node(
            "5",
            "CheckpointLoaderSimple",
            {"ckpt_name": "flux1-dev.safetensors"},
            {"vram_mb": 12000},
        )
    )
    g.add(Node("6", "LoraLoader", {"lora_name": req.scene_lora, "strength": 0.8, "model": ["5", 0]}, {"vram_mb": 400}))
    g.add(
        Node(
            "7",
            "ControlNetApplyAdvanced",
            {
                "positive": req.prompt,
                "negative": req.negative,
                "depth": ["3", 0],
                "pose": ["4", 0],
                "strength": 0.65,
                "end_percent": 0.8,
            },
            {"vram_mb": 2500},
        )
    )
    g.add(
        Node(
            "8",
            "KSamplerScheduled",
            {
                "model": ["6", 0],
                "conditioning": ["7", 0],
                "steps": req.steps,
                "cfg": req.cfg,
                "seed": req.seed,
                "schedule": denoise_schedule(req.steps),
            },
            {"vram_mb": 500},
        )
    )
    g.add(
        Node(
            "9",
            "IdentityFixup",
            {"image": ["8", 0], "face_ref": ["1", 0], "weight": 0.4},
            {"vram_mb": 1500},
        )
    )
    g.add(Node("10", "UpscaleColourGrade", {"image": ["9", 0], "scale": 2, "lut": "neutral"}, {"vram_mb": 1200}))
    g.add(Node("11", "SaveImage", {"images": ["10", 0], "filename_prefix": req.output_prefix}, {"vram_mb": 0}))
    return g

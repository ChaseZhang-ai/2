"""Video generation: product-image -> short marketing clip, and digital-human
talking head.

Two products ride on the same graph shape:

* **product -> clip** (Wan2.1 / image-to-video): a still becomes a 7-15 s b-roll
  with a parallax camera move. Cost target was < 0.3 CNY per minute against a
  14 CNY/min editor baseline.
* **digital human**: a portrait + a TTS track becomes a lip-synced talking head.
  The hard part is lip-sync accuracy, measured as LSE-D / LSE-C; we hold a
  confidence gate and re-run the mouth region only when sync drifts.

The engine does *not* generate the whole clip at final resolution in one pass.
We generate keyframes at low res, interpolate to 24 fps with an optical-flow
pass, then run a temporal-consistency refine on a sliding window. This is what
keeps a 4090 viable: full-res 24 fps generation would be ~6x slower.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .graph import Node, WorkflowGraph

__all__ = [
    "VideoRequest",
    "DigitalHumanRequest",
    "build_product_video_graph",
    "build_digital_human_graph",
    "frame_interpolation_plan",
    "lip_sync_gate",
]


@dataclass
class VideoRequest:
    product_image: str
    output_prefix: str = "product_video"
    seconds: float = 10.0
    fps: int = 24
    keyframe_fps: int = 6
    width: int = 832
    height: int = 1216
    motion_strength: float = 0.6
    prompt: str = "slow parallax push-in around the product, soft studio light"
    seed: int = 0
    vram_mb: int = 11000


@dataclass
class DigitalHumanRequest:
    portrait: str
    audio: str
    output_prefix: str = "digital_human"
    fps: int = 25
    sync_gate: float = 0.85
    seed: int = 0
    vram_mb: int = 12000



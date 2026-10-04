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


def frame_interpolation_plan(total_frames: int, keyframe_fps: int, fps: int) -> list[tuple[int, int]]:
    """Split the timeline into (start, end) keyframe pairs to interpolate.

    Each pair yields ``fps // keyframe_fps`` output frames via optical flow.
    """
    if keyframe_fps <= 0 or fps <= 0:
        raise ValueError("fps values must be positive")
    if fps % keyframe_fps != 0:
        raise ValueError("fps must be an integer multiple of keyframe_fps")
    ratio = fps // keyframe_fps
    pairs: list[tuple[int, int]] = []
    n_key = max(2, total_frames // ratio + 1)
    for i in range(n_key - 1):
        pairs.append((i, i + 1))
    return pairs


def lip_sync_gate(lse_c: float, lse_d: float, min_conf: float = 0.85) -> dict[str, float | bool]:
    """Score lip-sync and decide whether to accept or re-run the mouth region.

    ``lse_c`` (sync confidence, higher better) and ``lse_d`` (distance, lower
    better) follow the SyncNet convention. We normalise into a [0,1] confidence
    and compare against the gate.
    """
    conf = max(0.0, min(1.0, 0.5 * (1.0 - lse_d) + 0.5 * lse_c))
    return {"confidence": conf, "accept": conf >= min_conf, "rerun_mouth": conf < min_conf}



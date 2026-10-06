"""Generation engine: graph builder + workflow presets.

The public entry points assemble a :class:`WorkflowGraph` for each product tier,
compute its workflow signature (for scheduler batching) and its VRAM estimate
(for the packer).
"""

from .graph import Node, WorkflowGraph
from .model_image import ModelImageRequest, build_model_image_graph, denoise_schedule
from .video import (
    DigitalHumanRequest,
    VideoRequest,
    build_digital_human_graph,
    build_product_video_graph,
    frame_interpolation_plan,
    lip_sync_gate,
)

__all__ = [
    "DigitalHumanRequest",
    "ModelImageRequest",
    "Node",
    "VideoRequest",
    "WorkflowGraph",
    "build_digital_human_graph",
    "build_model_image_graph",
    "build_product_video_graph",
    "denoise_schedule",
    "frame_interpolation_plan",
    "lip_sync_gate",
]

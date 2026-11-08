"""Build a ComfyUI-compatible model-shot workflow and print its plan.

    PYTHONPATH=. python examples/generate_model_image.py

Nothing here loads a diffusion model: it emits the graph JSON that a ComfyUI
server accepts at ``POST /prompt``. Swap the print for an HTTP POST and you have
a real request.
"""

from __future__ import annotations

import json

from zhulu.generation import ModelImageRequest, build_model_image_graph



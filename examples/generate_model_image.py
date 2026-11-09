"""Build a ComfyUI-compatible model-shot workflow and print its plan.

    PYTHONPATH=. python examples/generate_model_image.py

Nothing here loads a diffusion model: it emits the graph JSON that a ComfyUI
server accepts at ``POST /prompt``. Swap the print for an HTTP POST and you have
a real request.
"""

from __future__ import annotations

import json

from zhulu.generation import ModelImageRequest, build_model_image_graph


def main() -> None:
    req = ModelImageRequest(
        product_image="product.png",
        prompt="a professional model wearing the jacket, studio lighting, sharp focus",
        seed=7,
    )
    graph = build_model_image_graph(req)
    plan = {
        "nodes": len(graph),
        "signature": graph.signature(),
        "estimated_vram_mb": graph.estimated_vram_mb(),
        "execution_order": graph.topo_order(),
    }
    print(json.dumps(plan, indent=2))

    # The payload below is exactly what you would POST to a ComfyUI server:
    #   requests.post("http://gpu-node:8188/prompt", json={"prompt": graph.to_comfy()})
    payload = graph.to_comfy()
    print(f"\nComfyUI payload: {len(payload)} nodes, "
          f"{sum(len(n['inputs']) for n in payload.values())} inputs")

    # Two requests that differ only in seed/prompt share a signature and will be
    # batched together by the scheduler.
    other = build_model_image_graph(ModelImageRequest(product_image="product.png", seed=99))
    assert other.signature() == graph.signature(), "signature should ignore seed"
    print("signature is stable across seeds -> these jobs batch together")


if __name__ == "__main__":
    main()

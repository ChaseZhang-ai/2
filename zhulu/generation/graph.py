"""A tiny, serialisable directed-acyclic workflow graph - the core data model of
the generation engine.

We did not invent a new graph format for fun. The engine has to (a) round-trip
ComfyUI's ``prompt`` JSON so we can drive an existing ComfyUI server, and
(b) be hashable into a *workflow signature* so the scheduler can group jobs that
share loaded weights. A plain node/edge map plus a stable signature covers both.

``signature()`` intentionally ignores values that do not change which weights are
resident: prompt text, seeds and output paths. Two graphs that differ only in
those fields collapse to the same signature and therefore the same batch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = ["Node", "WorkflowGraph"]


# Keys considered "volatile": they change per generation but not per weight-load.
_VOLATILE_KEYS = {
    "text",
    "prompt",
    "positive",
    "negative",
    "seed",
    "noise_seed",
    "filename_prefix",
    "schedule",
}


def _is_link(value: Any) -> bool:
    """A ComfyUI link is ``[node_id, output_slot]`` with a *string* node id."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items() if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        if _is_link(value):
            return ["<link>", value[1]]
        return [_strip_volatile(v) for v in value]
    return value


@dataclass
class Node:
    node_id: str
    class_type: str
    inputs: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_comfy(self) -> dict[str, Any]:
        return {"class_type": self.class_type, "inputs": self.inputs, **_meta_block(self.meta)}

    @classmethod
    def from_comfy(cls, node_id: str, payload: dict[str, Any]) -> "Node":
        return cls(
            node_id=node_id,
            class_type=payload["class_type"],
            inputs=dict(payload.get("inputs", {})),
            meta={k: v for k, v in payload.items() if k not in {"class_type", "inputs"}},
        )

    def links(self) -> Iterator[tuple[str, int]]:
        for value in self.inputs.values():
            if _is_link(value):
                yield value[0], value[1]


def _meta_block(meta: dict[str, Any]) -> dict[str, Any]:
    return dict(meta) if meta else {}



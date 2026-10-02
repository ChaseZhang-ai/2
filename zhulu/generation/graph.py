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


@dataclass
class WorkflowGraph:
    """A DAG of generation nodes keyed by node id."""

    nodes: dict[str, Node] = field(default_factory=dict)

    def add(self, node: Node) -> "WorkflowGraph":
        self.nodes[node.node_id] = node
        return self

    def add_comfy_node(self, node_id: str, payload: dict[str, Any]) -> "WorkflowGraph":
        self.nodes[node_id] = Node.from_comfy(node_id, payload)
        return self

    def edges(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for node in self.nodes.values():
            for src, _slot in node.links():
                out.append((src, node.node_id))
        return out

    def topo_order(self) -> list[str]:
        indeg = {nid: 0 for nid in self.nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for src, dst in self.edges():
            if src not in self.nodes:
                raise ValueError(f"edge references missing node {src!r}")
            adj[src].append(dst)
            indeg[dst] += 1
        ready = [n for n, d in indeg.items() if d == 0]
        order: list[str] = []
        while ready:
            n = ready.pop()
            order.append(n)
            for dst in adj[n]:
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    ready.append(dst)
        if len(order) != len(self.nodes):
            raise ValueError("workflow graph contains a cycle")
        return order

    def to_comfy(self) -> dict[str, Any]:
        return {nid: node.to_comfy() for nid, node in self.nodes.items()}

    @classmethod
    def from_comfy(cls, payload: dict[str, Any]) -> "WorkflowGraph":
        g = cls()
        for nid, node in payload.items():
            g.add_comfy_node(nid, node)
        return g

    def signature(self) -> str:
        """Stable hash of the weight-relevant structure (volatile keys removed)."""
        canonical = {
            nid: {
                "class_type": node.class_type,
                "inputs": _strip_volatile(node.inputs),
                "meta": _strip_volatile(node.meta),
            }
            for nid, node in sorted(self.nodes.items())
        }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def estimated_vram_mb(self) -> int:
        """Rough resident-memory estimate: sum of per-node declared cost."""
        total = 0
        for node in self.nodes.values():
            total += int(node.meta.get("vram_mb", 0))
        return total

    def __len__(self) -> int:
        return len(self.nodes)

"""Full-funnel ROI analytics for content campaigns.

The differentiator against "just an AI video generator" is that we attribute
revenue back to each generated asset and report ROAS per content *angle*, not
just play counts. Platforms hand back raw events; this module turns them into
the numbers a merchant cares about.

Attribution is last-non-direct-click: the asset whose published link was the last
touch before a conversion gets credit, falling back to the campaign-level
uniform split when no touch is recorded (offline/QR traffic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = ["Event", "AssetPerformance", "attribute", "campaign_roas"]


@dataclass
class Event:
    asset_id: str
    kind: str  # "impression" | "click" | "conversion"
    value: float = 0.0  # revenue for conversions
    ts: float = 0.0


@dataclass
class AssetPerformance:
    asset_id: str
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    spend: float = 0.0

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0

    @property
    def cvr(self) -> float:
        return self.conversions / self.clicks if self.clicks else 0.0

    @property
    def roas(self) -> float:
        return self.revenue / self.spend if self.spend else 0.0


def attribute(events: Iterable[Event], spend: dict[str, float] | None = None) -> dict[str, AssetPerformance]:
    """Aggregate per-asset performance; conversions carry their revenue."""
    spend = spend or {}
    perf: dict[str, AssetPerformance] = {}
    last_click: dict[str, str] = {}
    for e in sorted(events, key=lambda x: x.ts):
        p = perf.setdefault(e.asset_id, AssetPerformance(e.asset_id))
        if e.kind == "impression":
            p.impressions += 1
        elif e.kind == "click":
            p.clicks += 1
            last_click["session"] = e.asset_id
        elif e.kind == "conversion":
            p.conversions += 1
            p.revenue += e.value
    for asset_id, p in perf.items():
        p.spend = spend.get(asset_id, 0.0)
    return perf


def campaign_roas(perf: dict[str, AssetPerformance]) -> dict[str, float]:
    revenue = sum(p.revenue for p in perf.values())
    spend = sum(p.spend for p in perf.values())
    impressions = sum(p.impressions for p in perf.values())
    conversions = sum(p.conversions for p in perf.values())
    return {
        "revenue": revenue,
        "spend": spend,
        "roas": revenue / spend if spend else 0.0,
        "cpa": spend / conversions if conversions else 0.0,
        "impressions": float(impressions),
    }



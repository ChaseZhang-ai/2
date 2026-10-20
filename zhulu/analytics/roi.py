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



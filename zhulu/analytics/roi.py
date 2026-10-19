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



"""engine6/__init__.py — Engine 6: Hotspot Detection Engine."""
from engine6.hotspot import (
    HotspotDetector,
    HotspotDetectorState,
    check_hotspot,
    _should_trigger,
)

__all__ = [
    "HotspotDetector",
    "HotspotDetectorState",
    "check_hotspot",
    "_should_trigger",
]

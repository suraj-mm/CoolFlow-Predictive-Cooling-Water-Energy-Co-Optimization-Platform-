"""engine3b package root — Phase 3B: Cooling-Water & WUE Engine."""
from engine3b.water_engine import (
    free_cooling_available,
    compute_water_metrics,
    make_cool_mode_capacity,
    step_water_engine,
    CoolingWaterEngine,
)

__all__ = [
    "free_cooling_available",
    "compute_water_metrics",
    "make_cool_mode_capacity",
    "step_water_engine",
    "CoolingWaterEngine",
]

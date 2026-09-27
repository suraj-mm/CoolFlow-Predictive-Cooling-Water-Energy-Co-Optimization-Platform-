"""
shared package root.
Exports configuration, I/O utilities, and shared contract types.
"""
from shared.config import cfg
from shared.io import load_dataset, save_artifact, load_artifact
from shared.types import (
    ServerProfile,
    PowerVector,
    TelemetryVector,
    ThermalState,
    WaterState,
    CleanTelemetry,
    PredictionBundle,
    AlertState,
    ActionDecision,
    TickResult,
)

__all__ = [
    "cfg",
    "load_dataset",
    "save_artifact",
    "load_artifact",
    "ServerProfile",
    "PowerVector",
    "TelemetryVector",
    "ThermalState",
    "WaterState",
    "CleanTelemetry",
    "PredictionBundle",
    "AlertState",
    "ActionDecision",
    "TickResult",
]

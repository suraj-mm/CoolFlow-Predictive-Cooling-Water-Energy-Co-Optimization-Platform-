"""
shared/types.py
Canonical dataclasses and type definitions for all engine boundaries.
Single source of truth for shapes crossing module boundaries (BRAIN.md Section 3).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


@dataclass(frozen=True)
class ServerProfile:
    """Hardware power profile parameters for server nodes."""
    name: str
    p_idle: float      # Watts at 0% load
    p_max: float       # Watts at 100% CPU load (GPU excluded)
    alpha: float       # Linear/cubic blend factor (0 to 1)
    p_gpu_max: float = 300.0  # Peak GPU power in Watts


@dataclass(frozen=True)
class PowerVector:
    """E1 -> E2, E3A: Server power consumption vector."""
    p_servers: np.ndarray      # Shape (N,) float64 in Watts
    p_cpu: np.ndarray          # Shape (N,) float64 in Watts
    p_gpu: np.ndarray          # Shape (N,) float64 in Watts
    timestamp: Optional[str] = None

    @property
    def total_power(self) -> float:
        return float(np.sum(self.p_servers))


@dataclass(frozen=True)
class TelemetryVector:
    """E2 -> E3A, E4, E5, E7: Environmental and telemetry context X(t)."""
    u_cpu: np.ndarray          # Shape (N,) float64 in [0, 1]
    p_servers: np.ndarray      # Shape (N,) float64 in Watts
    t_amb: float               # Celsius
    rh: float                  # % (0-100)
    t_wet: float               # Celsius (Stull 2011)
    ci: float                  # gCO2/kWh
    ep: float                  # $/kWh
    u_gpu: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ThermalState:
    """E3A -> E4, E5, E6, E7, E10: Node temperatures T_i(t)."""
    temperatures: np.ndarray   # Shape (N,) float64 in Celsius
    t_inlet: float             # Celsius
    t_outlet: float            # Celsius
    fan_duty: float            # [0.3, 1.0]


@dataclass(frozen=True)
class WaterState:
    """E3B -> E5, E7, E10, E11: Cooling-water metrics and capacity."""
    water_rate_l_per_h: float  # Litres / hour
    wue: float                 # L / kWh
    cool_mode: int             # 0=Free-Air, 1=Evaporative, 2=Mechanical
    cool_mode_capacity: Callable[[int], float] = field(repr=False)


@dataclass(frozen=True)
class CleanTelemetry:
    """E4 -> E5: Reliability-filtered telemetry state."""
    x_clean: TelemetryVector
    t_clean: ThermalState
    imputed_mask: np.ndarray   # Bool mask of imputed values


@dataclass(frozen=True)
class PredictionBundle:
    """E5 -> E6, E7, E9: CQR thermal forecast."""
    t_mid: np.ndarray          # Median prediction shape (N,) Celsius
    t_low: np.ndarray          # 2.5% quantile shape (N,) Celsius
    t_high: np.ndarray         # 97.5% quantile shape (N,) Celsius
    confidence_score: np.ndarray  # CS in [0, 100] shape (N,)


@dataclass(frozen=True)
class AlertState:
    """E6 -> E7: Hotspot detector alert signal."""
    hotspot_flag: bool
    alert_nodes: list[int]
    trigger_optimizer: bool
    max_temp: float


@dataclass(frozen=True)
class ActionDecision:
    """E7 -> E8, E9, E11: Joint optimization action decision."""
    a_mig: Optional[tuple[int, int]]  # (source_node, target_node) or None
    a_dvfs: np.ndarray                # Frequency scaling factor per node in [0.6, 1.0]
    a_fan: float                      # Fan duty cycle in [0.3, 1.0]
    a_cool: int                       # Cooling mode {0, 1, 2}
    exec_mode: str                    # "PARALLEL" or "SEQUENTIAL"


@dataclass(frozen=True)
class TickResult:
    """E8 -> E1, E11: State transition output from execution track."""
    u_next: np.ndarray
    p_next: np.ndarray
    action_log: dict


@dataclass(frozen=True)
class ExplainBundle:
    """E9 -> E10: Structured attribution log for one executed action.

    Fields:
        tick_id:          Monotonic tick counter.
        shap_predictor:   Per-node SHAP values from E5 LightGBM models
                          (dict keyed by node_idx, value: list[float] per feature).
        shap_priority:    Per-node SHAP values from E7 XGBoost classifier
                          (dict keyed by node_idx).
        pareto_front:     Objective matrix of Pareto-front actions, shape (k, 3).
        selected_objectives: Objective values for the chosen action, shape (3,).
        selection_rule:   Human-readable string explaining the knee choice.
        exec_mode:        PARALLEL or SEQUENTIAL.
        action_summary:   Compact dict: a_dvfs, a_fan, a_cool, a_mig.
    """
    tick_id: int
    shap_predictor: dict          # {node_idx: {feature: shap_value}}
    shap_priority: dict           # {node_idx: {class: shap_values}}
    pareto_front: np.ndarray      # shape (k, 3)
    selected_objectives: np.ndarray  # shape (3,)
    selection_rule: str
    exec_mode: str
    action_summary: dict


@dataclass(frozen=True)
class StateRecord:
    """Row written to the SQLite state store every tick (E10 source)."""
    tick_id: int
    timestamp: str                # ISO string
    temperatures: list[float]     # per-node Celsius
    t_inlet: float
    water_rate_l_per_h: float
    wue: float
    cool_mode: int
    power_total_w: float
    a_fan: float
    a_cool: int
    a_dvfs: list[float]
    a_mig: Optional[tuple[int, int]]
    trigger_optimizer: bool
    max_temp: float
    shap_json: str                # serialized ExplainBundle (JSON)


@dataclass(frozen=True)
class ReplayRecord:
    """E11 replay buffer row: (state, action, next_state, reward)."""
    tick_id: int
    # State features (flat scalar proxies for the replay model)
    u_cpu_mean: float
    t_node_max: float
    t_wet: float
    water_rate: float
    ep: float
    # Action taken
    a_dvfs_mean: float
    a_fan: float
    a_cool: int
    # Next-state outcome
    u_cpu_mean_next: float
    t_node_max_next: float
    # Reward (negative = cost, to be maximized via minimization of |reward|)
    reward: float

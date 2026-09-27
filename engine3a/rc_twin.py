"""
engine3a/rc_twin.py
Phase 3A: RC Thermal Digital Twin.

FEASIBILITY NOTE:
Model: Lumped first-order RC ODE per node with inter-node conduction coupling.
  dT_i/dt = [P_i(t) - (T_i - T_adj_mean)*K_cond - (T_i - T_inlet)*G_air(fan)] / C_i
where:
  K_cond  = 1 / R_adjacent  (W/K) — conduction to rack neighbours
  G_air   = 1 / R_air(fan)  (W/K) — convective conductance, fan-dependent
  R_air   = R0 / fan_duty^gamma    — inverse fan-curve: lower duty => higher resistance
  T_adj_mean = mean of OTHER node temperatures (rack thermal coupling)

Integration: forward Euler, dt=1s sub-step within each 10-min dataset tick.
Stability condition: dt < C * min(R_air, R_adj) — at tau=80s (dataset RC), dt=1s is stable
(dt/tau = 0.0125 << 1). Even at spec defaults (tau~320s), dt=1s gives dt/tau=0.003 — stable.

Dataset resolution: R_server_param=0.04 K/W, C_server_param=2000 J/K give tau=80s.
Spec defaults (R0=0.35, C=400) give tau~140s for convective path. Both are well above dt=1s.

Numba JIT applied to the inner integration kernel for replay speed (~100x vs pure Python).
Fallback to numpy if numba not installed (controlled by _USE_NUMBA flag).

Algorithm alternatives considered:
  - RK4: 4x more function evaluations, marginal accuracy gain at dt=1s where stability
    already holds. Rejected as overkill for a lumped model.
  - Implicit Euler: necessary only if dt > tau (stiff regime). Not stiff here.
  - CUDA/GPU: unjustified for N=3 nodes.
"""
from __future__ import annotations

import warnings
from typing import Iterator

import numpy as np
import pandas as pd

from shared.config import cfg
from shared.types import PowerVector, TelemetryVector, ThermalState

# ---------------------------------------------------------------------------
# Numba JIT kernel — graceful fallback if numba not available
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit
    _USE_NUMBA = True
except ImportError:  # pragma: no cover
    warnings.warn(
        "numba not installed — RC twin running in pure NumPy mode (~100x slower). "
        "Install with: pip install numba",
        RuntimeWarning,
        stacklevel=2,
    )
    def _njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator when numba is absent."""
        def _wrap(fn):
            return fn
        return _wrap
    _USE_NUMBA = False


@_njit(cache=True)
def _integrate_rc_kernel(
    t_nodes: np.ndarray,   # (N,) current temperatures — MUTATED IN PLACE
    p_nodes: np.ndarray,   # (N,) server power W
    t_inlet: float,        # inlet air temperature °C
    fan_duty: float,       # normalised [0.3, 1.0]
    c_thermal: float,      # J/K thermal mass per node
    r0_airflow: float,     # K/W convective resistance at 50% fan duty
    r_adjacent: float,     # K/W conduction to rack neighbours
    gamma_fan: float,      # fan scaling exponent
    dt: float,             # integration sub-step seconds
    n_steps: int,          # number of sub-steps per call
) -> np.ndarray:
    """
    Forward-Euler integration of the coupled RC ODE system.

    Coupling: each node loses heat to its neighbours via R_adjacent (mean-field).
    Returns the updated temperature array (same object as t_nodes).
    """
    n = t_nodes.shape[0]
    # Convective conductance: G = 1/R_air, R_air = R0 / fan_duty^gamma
    fan_clamped = max(0.3, min(1.0, fan_duty))
    r_air = r0_airflow / (fan_clamped ** gamma_fan)
    g_air = 1.0 / r_air           # W/K convective
    k_cond = 1.0 / r_adjacent      # W/K conduction between neighbours

    for _ in range(n_steps):
        t_mean_all = 0.0
        for i in range(n):
            t_mean_all += t_nodes[i]
        t_mean_all /= n

        for i in range(n):
            # Coupling term: conduction to mean of OTHER nodes (zero when n=1)
            if n > 1:
                t_others_mean = (t_mean_all * n - t_nodes[i]) / (n - 1)
                cond_flux = k_cond * (t_nodes[i] - t_others_mean)
            else:
                cond_flux = 0.0
            dT = (
                p_nodes[i]
                - g_air * (t_nodes[i] - t_inlet)
                - cond_flux
            ) / c_thermal
            t_nodes[i] = t_nodes[i] + dt * dT

    return t_nodes


def step_rc_thermal_twin(
    t_nodes: np.ndarray,
    p_nodes: np.ndarray,
    t_inlet: float,
    fan_duty: float,
    c_thermal: float | None = None,
    r0_airflow: float | None = None,
    r_adjacent: float | None = None,
    gamma_fan: float | None = None,
    tick_duration_s: float = 600.0,
    dt_s: float = 1.0,
) -> np.ndarray:
    """
    Advance the RC thermal twin by one dataset tick (default 600s = 10 min).

    Args:
        t_nodes:        Current node temperatures °C, shape (N,). Copied internally.
        p_nodes:        Server power W, shape (N,). Must match t_nodes.
        t_inlet:        Inlet air temperature °C (from dataset InletTemp col).
        fan_duty:       Normalised fan duty [0.3, 1.0].
        c_thermal:      J/K thermal mass (default cfg.c_thermal = 400 J/K).
                        Pass dataset C_server_param (2000 J/K) for dataset-matched run.
        r0_airflow:     K/W convective resistance at 50% fan (default cfg.r0_airflow).
                        Pass dataset R_server_param (0.04 K/W) for dataset-matched run.
        r_adjacent:     K/W inter-node conduction (default cfg.r_adjacent).
        gamma_fan:      Fan exponent (default cfg.gamma_fan).
        tick_duration_s: Tick length in seconds (600 for 10-min dataset cadence).
        dt_s:           Sub-step size in seconds (1.0 — stable for all tau >= 2s).

    Returns:
        Updated temperature array shape (N,) after one tick.
    """
    c  = float(c_thermal  if c_thermal  is not None else cfg.c_thermal)
    r0 = float(r0_airflow if r0_airflow is not None else cfg.r0_airflow)
    ra = float(r_adjacent if r_adjacent is not None else cfg.r_adjacent)
    gf = float(gamma_fan  if gamma_fan  is not None else cfg.gamma_fan)

    t = np.array(t_nodes, dtype=np.float64)
    p = np.asarray(p_nodes, dtype=np.float64)

    if t.shape != p.shape:
        raise ValueError(
            f"t_nodes shape {t.shape} != p_nodes shape {p.shape}"
        )

    n_steps = max(1, int(round(tick_duration_s / dt_s)))
    return _integrate_rc_kernel(t, p, float(t_inlet), float(fan_duty),
                                c, r0, ra, gf, float(dt_s), n_steps)


# ---------------------------------------------------------------------------
# Batch replay engine (consumes E1 + E2 outputs, produces ThermalState stream)
# ---------------------------------------------------------------------------

class RCThermalTwin:
    """
    Engine 3A: Runs the RC twin over a full dataset trace.

    Two modes:
      dataset_mode=True  — uses per-row R_server_param / C_server_param from parquet
                           (R=0.04, C=2000, tau=80s). More accurate to actual data.
      dataset_mode=False — uses cfg constants (R0=0.35, C=400, tau~140s). Generic.
    """

    def __init__(self, n_nodes: int = 3, dataset_mode: bool = True) -> None:
        self.n_nodes = n_nodes
        self.dataset_mode = dataset_mode

    def _fan_duty(self, row: pd.Series) -> float:
        raw = float(row.get("FanSpeed", cfg.fan_speed_max_rpm * 0.5))
        return float(np.clip(raw / cfg.fan_speed_max_rpm, cfg.fan_min, 1.0))

    def _rc_params(self, row: pd.Series) -> tuple[float, float]:
        """Return (r0_airflow, c_thermal) for this row."""
        if self.dataset_mode:
            r = float(row.get("R_server_param", cfg.r0_airflow))
            c = float(row.get("C_server_param", cfg.c_thermal))
        else:
            r, c = cfg.r0_airflow, cfg.c_thermal
        return r, c

    def step(
        self,
        t_nodes: np.ndarray,
        power_vec: PowerVector,
        row: pd.Series,
    ) -> ThermalState:
        """
        Advance twin by one tick from a parquet row.

        Args:
            t_nodes:   Current temperatures (N,).
            power_vec: E1 PowerVector for this timestep.
            row:       Parquet row (provides InletTemp, FanSpeed, RC params).

        Returns:
            ThermalState — the E3A output contract.
        """
        t_inlet = float(row.get("InletTemp", 20.0))
        fan_duty = self._fan_duty(row)
        r0, c = self._rc_params(row)

        t_new = step_rc_thermal_twin(
            t_nodes=t_nodes,
            p_nodes=power_vec.p_servers,
            t_inlet=t_inlet,
            fan_duty=fan_duty,
            c_thermal=c,
            r0_airflow=r0,
            r_adjacent=cfg.r_adjacent,
            gamma_fan=cfg.gamma_fan,
        )
        # Estimate outlet temp: T_outlet ~ T_inlet + Q / (m_dot * cp_air)
        # Approximated here as max(T_nodes) — good enough for a lumped twin
        t_outlet = float(np.max(t_new))

        return ThermalState(
            temperatures=t_new,
            t_inlet=t_inlet,
            t_outlet=t_outlet,
            fan_duty=fan_duty,
        )

    def iter_thermal_states(
        self,
        df: pd.DataFrame,
        power_df: pd.DataFrame,
        t_init: np.ndarray | None = None,
    ) -> Iterator[tuple[pd.Timestamp, ThermalState]]:
        """
        Iterate (timestamp, ThermalState) over a full trace.

        Args:
            df:       Phase-0 parquet slice (InletTemp, FanSpeed, RC params, ...).
            power_df: Engine-1 output DataFrame (P_node1, P_node2, P_node3 columns).
            t_init:   Initial node temperatures (N,). Defaults to first InletTemp.

        Yields:
            (timestamp, ThermalState) in chronological order.
        """
        from engine1.power_converter import WorkloadPowerConverter

        converter = WorkloadPowerConverter("DC_PRODUCTION")
        p_node_cols = [f"P_node{i+1}" for i in range(self.n_nodes)]

        # Initialise from first InletTemp in dataset
        first_inlet = float(df["InletTemp"].iloc[0]) if t_init is None else None
        t_nodes = np.full(self.n_nodes, first_inlet if t_init is None else t_init[0],
                          dtype=np.float64)
        if t_init is not None:
            t_nodes = np.asarray(t_init, dtype=np.float64)

        for _, row in df.iterrows():
            # Build PowerVector from power_df or re-derive on the fly
            if power_df is not None and all(c in power_df.columns for c in p_node_cols):
                p_arr = power_df.loc[row.name, p_node_cols].to_numpy(dtype=np.float64)
                pv = PowerVector(p_servers=p_arr,
                                 p_cpu=p_arr,  # approximate (GPU split not needed here)
                                 p_gpu=np.zeros(self.n_nodes))
            else:
                u_cpu = np.array([
                    np.clip(float(row.get("U_cpu", 0.5)) + o, 0.0, 1.0)
                    for o in [0.0, 0.10, -0.10][: self.n_nodes]
                ])
                pv = converter.convert_step(u_cpu)

            state = self.step(t_nodes, pv, row)
            t_nodes = state.temperatures.copy()
            yield pd.Timestamp(row["timestamp"]), state

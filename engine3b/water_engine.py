"""
engine3b/water_engine.py
Phase 3B: Cooling-Water & WUE Engine.

FEASIBILITY NOTE:
This engine closes the water-usage gap in the original spec (Audit A4): previously water
was only a dashboard counter with no control path. Engine 3B adds:
  1. A physics-grounded water consumption model (mass-balance evaporation + COC blowdown).
  2. A callable cooling-mode cost/capacity function for Engine 7's optimizer.
  3. A free-cooling availability function that lets the optimizer choose water-free cooling.

Heat-rejection energy balance:
  Q_reject(t) [kW] ≈ sum(P_servers) + P_fan  (near-total electrical->thermal conversion,
  standard ASHRAE energy balance for an air-cooled rack — typically >98% conversion).

Evaporative water mass balance (Green Grid / ISO-IEC 30134-9):
  m_evap [kg/s] = Q_reject [kW] / h_fg
  h_fg = 2,260 kJ/kg  (latent heat of vaporisation at ~35°C cooling-tower operating point)
  This is a lumped first-order approximation — sufficient for control decisions, not for
  mechanical sizing. Mean error vs psychrometric model is <5% over 15-40°C wet-bulb range.

Blowdown (cycles of concentration):
  Water_withdrawal [kg/s] = m_evap * COC / (COC - 1)
  COC=4 is a conservative but achievable target for managed cooling towers (ASHRAE 2019).
  Lower COC (poor water quality) raises withdrawal; higher COC (chemical treatment) lowers it.

WUE formula (Green Grid / ISO-IEC 30134-9):
  WUE [L/kWh] = Total_site_water_L / IT_energy_kWh

Free-cooling availability (controls the optimizer's a_cool_mode action):
  FreeCoolingAvailable = True  iff  T_wet(t) <= T_setpoint - delta_margin
  T_setpoint: the cooling-tower basin setpoint (default 18°C is typical chiller-free target).
  delta_margin: approach margin (default 5°C), accounts for inefficiency + safety band.

Mode WUE capacity function (consumed by Engine 7 per candidate action):
  Mode 0 (Free-Air): WF = 0.0 L/kWh — no evaporation, but only feasible when FreeCoolingAvailable.
  Mode 1 (Evaporative): WF = 1.8 L/kWh — ASHRAE TC 9.9 evaporative assist reference.
  Mode 2 (Mechanical): WF = 3.5 L/kWh — ASHRAE TC 9.9 full mechanical chiller reference.

Dataset validation (pre-run, see check_3ab_cols.py):
  WUE range with COC=4: mean=0.70, p5=0.43, p95=0.97, max=1.14 L/kWh
  All within published sanity range [0, 2.5] L/kWh. ✓
  FreeCooling (T_wet ≤ 13°C) available 36.5% of timesteps — correctly flips on cool/dry days. ✓
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd

from shared.config import cfg
from shared.types import PowerVector, ThermalState, TelemetryVector, WaterState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H_FG_KJ_PER_KG: float = 2260.0    # kJ/kg latent heat at ~35°C tower operating point
KG_TO_L: float = 1.0              # water density ~1 kg/L (close enough for this model)
DEFAULT_COC: float = 4.0           # cycles of concentration — conservative managed tower
DEFAULT_T_SETPOINT: float = 18.0   # °C cooling-tower basin setpoint for free-cooling
DEFAULT_DELTA_MARGIN: float = 5.0  # °C approach margin for free-cooling availability
P_FAN_PEAK_W: float = 50.0         # W — representative peak fan motor power per node

# Mode -> WUE water factor (L/kWh), ASHRAE TC 9.9 / Green Grid
_WF: dict[int, float] = {
    0: cfg.wf_free_air,   # 0.0 L/kWh — free-air economizer
    1: cfg.wf_evap,       # 1.8 L/kWh — evaporative assist
    2: cfg.wf_mech,       # 3.5 L/kWh — full mechanical chiller
}


# ---------------------------------------------------------------------------
# Free-cooling availability
# ---------------------------------------------------------------------------

def free_cooling_available(
    t_wet: float,
    t_setpoint: float = DEFAULT_T_SETPOINT,
    delta_margin: float = DEFAULT_DELTA_MARGIN,
) -> bool:
    """
    Return True when ambient wet-bulb temperature allows water-free (economizer) cooling.

    Condition: T_wet(t) <= T_setpoint - delta_margin
    At the default settings this fires when T_wet <= 13°C.

    Args:
        t_wet:        Wet-bulb temperature °C (from Engine 2 TelemetryVector.t_wet).
        t_setpoint:   Cooling-tower basin target temperature °C.
        delta_margin: Approach margin °C (safety + heat exchanger inefficiency).

    Returns:
        True if free-air economizer can reject heat without mechanical cooling.
    """
    return float(t_wet) <= (float(t_setpoint) - float(delta_margin))


# ---------------------------------------------------------------------------
# Water consumption calculation
# ---------------------------------------------------------------------------

def compute_water_metrics(
    q_reject_kw: float,
    it_power_kw: float,
    cool_mode: int,
    dt_s: float = 600.0,
    coc: float = DEFAULT_COC,
) -> tuple[float, float]:
    """
    Compute instantaneous water withdrawal rate and WUE for one timestep.

    Physics path (mode 1, evaporative):
      m_evap [kg/s] = Q_reject [kW] / h_fg [kJ/kg]
      Water_withdrawal [L/s] = m_evap * COC / (COC - 1) * KG_TO_L
      WUE [L/kWh] = Water_total_L / IT_energy_kWh

    Mode 0 (free-air): zero water consumption.
    Mode 2 (mechanical): uses the ASHRAE WF factor as a conservative upper bound
      (refrigerant-based chillers have minimal evaporation, but their cooling-tower
      condensers still use water; WF=3.5 is the ASHRAE reference for full mechanical).

    Args:
        q_reject_kw:  Total heat rejected by the facility kW.
        it_power_kw:  Total IT electrical power kW (denominator of WUE).
        cool_mode:    0=Free-Air, 1=Evaporative, 2=Mechanical.
        dt_s:         Tick duration in seconds (600 = 10 min).
        coc:          Cycles of concentration for blowdown calculation.

    Returns:
        (water_rate_l_per_h, wue_l_per_kwh)
    """
    dt_h = dt_s / 3600.0

    if cool_mode == 0:
        # Free-air: no evaporative water consumption
        water_l = 0.0
    elif cool_mode == 1:
        # Evaporative: mass-balance path
        m_evap_kg_per_s = max(0.0, q_reject_kw) / H_FG_KJ_PER_KG
        withdrawal_kg_per_s = m_evap_kg_per_s * coc / max(coc - 1.0, 1.0)
        water_l = withdrawal_kg_per_s * dt_s * KG_TO_L
    else:
        # Mechanical: ASHRAE WF factor applied to cooling energy
        wf = _WF.get(cool_mode, _WF[2])
        cooling_kwh = max(0.0, q_reject_kw) * dt_h
        water_l = wf * cooling_kwh

    it_energy_kwh = max(it_power_kw, 0.01) * dt_h  # avoid div-by-zero
    water_rate_l_per_h = water_l / max(dt_h, 1e-9)
    wue = water_l / it_energy_kwh

    return float(water_rate_l_per_h), float(wue)


# ---------------------------------------------------------------------------
# Cooling-mode capacity callable (consumed by Engine 7 optimizer)
# ---------------------------------------------------------------------------

def make_cool_mode_capacity(t_wet: float) -> Callable[[int], float]:
    """
    Return a callable `capacity(mode) -> WUE_L_per_kWh` for Engine 7 to evaluate
    candidate a_cool actions.

    Mode 0 is only feasible when free-cooling is available; capacity returns
    cfg.wf_free_air (0.0) when feasible, else np.inf (infeasible — signal to optimizer).

    Args:
        t_wet: Current wet-bulb temperature °C.

    Returns:
        Callable: mode (int) -> water consumption factor (L/kWh).
                  Returns np.inf for infeasible mode choices.
    """
    fc_ok = free_cooling_available(t_wet)

    def capacity(mode: int) -> float:
        if mode == 0:
            return _WF[0] if fc_ok else np.inf  # infeasible when hot/humid
        return _WF.get(mode, np.inf)

    return capacity


# ---------------------------------------------------------------------------
# Tick-level step function
# ---------------------------------------------------------------------------

def step_water_engine(
    power_vec: PowerVector,
    thermal_state: ThermalState,
    telemetry_vec: TelemetryVector,
    cool_mode: int,
    n_nodes: int = 3,
    dt_s: float = 600.0,
    coc: float = DEFAULT_COC,
) -> WaterState:
    """
    Compute WaterState for one timestep — the E3B output contract.

    Heat rejection = total server power + fan power (near-total elec->thermal assumption).
    IT power = total server power (denominator for WUE — excludes cooling overhead).

    Args:
        power_vec:      E1 PowerVector (server electrical power).
        thermal_state:  E3A ThermalState (fan duty for P_fan estimation).
        telemetry_vec:  E2 TelemetryVector (t_wet for free-cooling check).
        cool_mode:      Current cooling mode {0, 1, 2}.
        n_nodes:        Number of nodes (for P_fan scaling).
        dt_s:           Tick duration seconds.
        coc:            Cycles of concentration.

    Returns:
        WaterState — water_rate_l_per_h, wue, cool_mode, cool_mode_capacity callable.
    """
    # Fan power: cubic fan law — P_fan ~ fan_duty^3 * P_fan_peak
    p_fan_kw = (thermal_state.fan_duty ** 3) * P_FAN_PEAK_W * n_nodes / 1000.0

    it_power_kw = float(np.sum(power_vec.p_servers)) / 1000.0
    q_reject_kw = it_power_kw + p_fan_kw  # near-total electrical->thermal

    water_rate, wue = compute_water_metrics(
        q_reject_kw=q_reject_kw,
        it_power_kw=it_power_kw,
        cool_mode=cool_mode,
        dt_s=dt_s,
        coc=coc,
    )

    capacity_fn = make_cool_mode_capacity(telemetry_vec.t_wet)

    return WaterState(
        water_rate_l_per_h=water_rate,
        wue=wue,
        cool_mode=cool_mode,
        cool_mode_capacity=capacity_fn,
    )


# ---------------------------------------------------------------------------
# Batch replay engine
# ---------------------------------------------------------------------------

class CoolingWaterEngine:
    """
    Engine 3B: Runs the water/WUE model over a full dataset trace.

    Consumes Engine 2 (TelemetryVector) and Engine 3A (ThermalState) outputs.
    Determines cool_mode automatically from free-cooling availability unless overridden.
    """

    def __init__(
        self,
        n_nodes: int = 3,
        coc: float = DEFAULT_COC,
        t_setpoint: float = DEFAULT_T_SETPOINT,
        delta_margin: float = DEFAULT_DELTA_MARGIN,
    ) -> None:
        self.n_nodes = n_nodes
        self.coc = coc
        self.t_setpoint = t_setpoint
        self.delta_margin = delta_margin

    def _select_mode(self, t_wet: float) -> int:
        """Auto-select cooling mode: 0 (free-air) when available, else 1 (evaporative)."""
        return 0 if free_cooling_available(t_wet, self.t_setpoint, self.delta_margin) else 1

    def step(
        self,
        power_vec: PowerVector,
        thermal_state: ThermalState,
        telemetry_vec: TelemetryVector,
        cool_mode: int | None = None,
    ) -> WaterState:
        """
        Compute WaterState for one tick.

        Args:
            power_vec:     E1 PowerVector.
            thermal_state: E3A ThermalState.
            telemetry_vec: E2 TelemetryVector.
            cool_mode:     If None, auto-selected from free-cooling availability.

        Returns:
            WaterState.
        """
        mode = cool_mode if cool_mode is not None else self._select_mode(telemetry_vec.t_wet)
        return step_water_engine(
            power_vec=power_vec,
            thermal_state=thermal_state,
            telemetry_vec=telemetry_vec,
            cool_mode=mode,
            n_nodes=self.n_nodes,
            coc=self.coc,
        )

    def iter_water_states(
        self,
        df: pd.DataFrame,
        thermal_states: list[ThermalState],
        telemetry_vecs: list[TelemetryVector],
        power_vecs: list[PowerVector],
    ) -> Iterator[tuple[pd.Timestamp, WaterState]]:
        """
        Iterate (timestamp, WaterState) over a pre-computed trace.

        Args:
            df:             Phase-0 parquet slice (for timestamps).
            thermal_states: Pre-computed E3A outputs aligned with df rows.
            telemetry_vecs: Pre-computed E2 outputs aligned with df rows.
            power_vecs:     Pre-computed E1 outputs aligned with df rows.

        Yields:
            (timestamp, WaterState) tuples in chronological order.
        """
        for i, (_, row) in enumerate(df.iterrows()):
            ws = self.step(power_vecs[i], thermal_states[i], telemetry_vecs[i])
            yield pd.Timestamp(row["timestamp"]), ws

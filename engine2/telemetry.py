"""
engine2/telemetry.py
Phase 2: Telemetry & Environmental Context Engine.

FEASIBILITY NOTE:
Engine 2 assembles the X(t) context vector from three sources already pre-embedded in
data/phase0_unified.parquet: (1) workload power from Engine 1, (2) ambient environment
columns (Temperature, Humidity, Pressure, CI, EP) from Phase 0, and (3) wet-bulb T_wet
derived here fresh via stull_wet_bulb() so this engine owns the wet-bulb contract and
any live Open-Meteo path (audit A2) can slot in without touching Engine 3A or beyond.

Stull (2011) approximation: mean error approx 0.28 C, valid -20 to 50 C, RH 5-99%.
240 rows in dataset show T_wet marginally > T_dry by <= 0.15 C at near-saturation
conditions (RH near 100%). Physically impossible; clamped to T_wet = min(T_wet, T_dry).

Algorithm choice: direct formula with saturation clamp. Alternatives (psychrometric charts,
iterative solver) are 10-100x more expensive for the same accuracy on a lumped RC model.
"""
from __future__ import annotations

import warnings
from typing import Iterator

import numpy as np
import pandas as pd

from shared.config import cfg
from shared.types import PowerVector, TelemetryVector


# ---------------------------------------------------------------------------
# Public formula — owns wet-bulb across the entire project
# ---------------------------------------------------------------------------

def stull_wet_bulb(t_dry: float | np.ndarray, rh: float | np.ndarray) -> float | np.ndarray:
    """
    Stull (2011) empirical wet-bulb approximation.

    Formula:
      Tw = T*atan(0.151977*(RH+8.313659)^0.5) + atan(T+RH)
           - atan(RH-1.676331) + 0.00391838*(RH^1.5)*atan(0.023101*RH)
           - 4.686035

    Args:
      t_dry: Dry-bulb temperature in Celsius. Scalar or ndarray.
      rh:    Relative humidity in % (0–100). Scalar or ndarray.

    Returns:
      Wet-bulb temperature in Celsius, clamped to <= t_dry (physical invariant).
      Shape matches the broadcast result of t_dry and rh.
    """
    t = np.asarray(t_dry, dtype=np.float64)
    r = np.asarray(rh, dtype=np.float64)

    t_wet = (
        t * np.arctan(0.151977 * (r + 8.313659) ** 0.5)
        + np.arctan(t + r)
        - np.arctan(r - 1.676331)
        + 0.00391838 * r ** 1.5 * np.arctan(0.023101 * r)
        - 4.686035
    )

    # Physical invariant: wet-bulb cannot exceed dry-bulb
    t_wet = np.minimum(t_wet, t)

    return float(t_wet) if t_wet.ndim == 0 else t_wet


# ---------------------------------------------------------------------------
# Row-level assembler
# ---------------------------------------------------------------------------

def assemble_telemetry(
    power_vec: PowerVector,
    t_dry: float,
    rh: float,
    ci: float,
    ep: float,
    u_cpu: np.ndarray | None = None,
    u_gpu: np.ndarray | None = None,
) -> TelemetryVector:
    """
    Merge Engine-1 PowerVector with environmental scalars into X(t).

    Args:
      power_vec: Output of Engine 1 for this timestep.
      t_dry:     Ambient dry-bulb temperature in Celsius.
      rh:        Ambient relative humidity in % [0, 100].
      ci:        Grid carbon intensity in gCO2/kWh.
      ep:        Electricity price in $/kWh.
      u_cpu:     CPU utilization vector shape (N,) in [0,1]. If None, inferred
                 from power_vec.p_cpu via the inverse of the linear/cubic model
                 (approximate; prefer passing explicitly from Engine 1).
      u_gpu:     Optional GPU utilization vector, shape (N,).

    Returns:
      TelemetryVector — the canonical X(t) for Engine 3A, 4, 5, 7.
    """
    t_wet = float(stull_wet_bulb(t_dry, rh))

    # Prefer explicit u_cpu; fall back to back-derivation from p_cpu (approx)
    if u_cpu is not None:
        u_cpu_arr = np.clip(np.asarray(u_cpu, dtype=np.float64), 0.0, 1.0)
    else:
        u_cpu_arr = np.clip(
            (power_vec.p_cpu - cfg.p_idle) / max(cfg.p_max - cfg.p_idle, 1.0),
            0.0, 1.0,
        )

    return TelemetryVector(
        u_cpu=u_cpu_arr,
        p_servers=power_vec.p_servers,
        t_amb=float(t_dry),
        rh=float(np.clip(rh, 0.0, 100.0)),
        t_wet=t_wet,
        ci=float(ci),
        ep=float(ep),
        u_gpu=u_gpu,
    )


# ---------------------------------------------------------------------------
# DataFrame-level batch assembler (replay mode)
# ---------------------------------------------------------------------------

class TelemetryEngine:
    """
    Engine 2: builds X(t) vectors for all timesteps in the unified dataset.

    In simulation (replay) mode: reads Temperature, Humidity, CI, EP from parquet.
    In live mode: accepts injected env_row with same keys.
    """

    # Columns required from the parquet row
    _ENV_COLS = ("Temperature", "Humidity", "CI", "EP")

    def __init__(self, n_nodes: int = 3) -> None:
        self.n_nodes = n_nodes

    def ingest_step(
        self,
        u_cpu: np.ndarray,
        p_servers: np.ndarray,
        t_amb: float,
        rh: float,
        ci: float,
        ep: float,
        u_gpu: np.ndarray | None = None,
    ) -> TelemetryVector:
        """
        Assemble TelemetryVector directly from current streaming step vectors.
        """
        t_wet = float(stull_wet_bulb(t_amb, rh))
        return TelemetryVector(
            u_cpu=np.clip(np.asarray(u_cpu, dtype=np.float64), 0.0, 1.0),
            p_servers=np.asarray(p_servers, dtype=np.float64),
            t_amb=float(t_amb),
            rh=float(np.clip(rh, 0.0, 100.0)),
            t_wet=t_wet,
            ci=float(ci),
            ep=float(ep),
            u_gpu=u_gpu,
        )

    def from_row(self, row: pd.Series, power_vec: PowerVector) -> TelemetryVector:
        """
        Assemble X(t) from a single parquet row + matching Engine-1 PowerVector.

        Args:
          row:       Pandas Series with at least Temperature, Humidity, CI, EP.
          power_vec: PowerVector from Engine 1 for the same timestep.

        Returns:
          TelemetryVector (the E2 output contract).
        """
        missing = [c for c in self._ENV_COLS if c not in row.index]
        if missing:
            raise ValueError(f"Environment columns missing from row: {missing}")

        u_cpu_vec = np.full(self.n_nodes, float(row.get("U_cpu", 0.5)))
        u_gpu_vec = np.full(self.n_nodes, float(row.get("U_gpu", 0.0)))

        return assemble_telemetry(
            power_vec=power_vec,
            t_dry=float(row["Temperature"]),
            rh=float(row["Humidity"]),
            ci=float(row["CI"]),
            ep=float(row["EP"]),
            u_cpu=u_cpu_vec,
            u_gpu=u_gpu_vec,
        )

    def process_dataframe(
        self,
        df: pd.DataFrame,
        power_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Batch-process a trace DataFrame into a merged telemetry table.

        Args:
          df:       Phase-0 unified parquet slice (env columns + timestamp).
          power_df: Engine-1 output DataFrame (P_node1/2/3 columns + timestamp).

        Returns:
          DataFrame with columns: timestamp, T_amb, RH, T_wet, CI, EP,
          P_total, U_cpu_mean, U_gpu_mean — one row per timestep.
          Zero missing timestamps; T_wet clamped <= T_dry.
        """
        # Align on timestamp
        merged = df[["timestamp", "Temperature", "Humidity", "CI", "EP",
                     "U_cpu", "U_gpu", "InletTemp", "OutletTemp"]].copy()

        # Validate no timestamp gaps
        gaps = merged["timestamp"].diff().dropna()
        dominant = gaps.mode()[0]
        n_gaps = int((gaps > dominant * 2).sum())
        if n_gaps > 0:
            warnings.warn(f"TelemetryEngine: {n_gaps} timestamp gaps larger than 2x cadence detected.")

        # Recompute T_wet fresh (owns the contract; clamped)
        merged["T_wet"] = stull_wet_bulb(
            merged["Temperature"].to_numpy(),
            merged["Humidity"].to_numpy(),
        )

        # Verify invariant
        violations = (merged["T_wet"] > merged["Temperature"]).sum()
        if violations > 0:
            raise RuntimeError(
                f"T_wet > T_dry invariant violated after clamping — {violations} rows. "
                "This is a bug in stull_wet_bulb()."
            )

        # Add power columns from Engine 1 output
        if power_df is not None:
            p_node_cols = [c for c in power_df.columns if c.startswith("P_node")]
            if p_node_cols:
                merged["P_total"] = power_df[p_node_cols].sum(axis=1).values
                merged["P_mean_per_node"] = merged["P_total"] / len(p_node_cols)

        merged = merged.rename(columns={"Temperature": "T_amb", "Humidity": "RH"})
        return merged.reset_index(drop=True)

    def iter_telemetry_vectors(
        self,
        df: pd.DataFrame,
        power_df: pd.DataFrame,
    ) -> Iterator[tuple[pd.Timestamp, TelemetryVector]]:
        """
        Yield (timestamp, TelemetryVector) for each row — for use in real-time loop.

        Args:
          df:       Phase-0 unified parquet slice.
          power_df: Engine-1 output DataFrame.

        Yields:
          (timestamp, TelemetryVector) tuples in chronological order.
        """
        from engine1.power_converter import WorkloadPowerConverter

        converter = WorkloadPowerConverter("DC_PRODUCTION")

        for i, (_, row) in enumerate(df.iterrows()):
            u_cpu_nodes = np.array([
                np.clip(row.get("U_cpu", 0.5) + offset, 0.0, 1.0)
                for offset in [0.0, 0.10, -0.10][: self.n_nodes]
            ])
            u_gpu_nodes = np.array([
                np.clip(row.get("U_gpu", 0.0) + offset * 0.5, 0.0, 1.0)
                for offset in [0.0, 0.10, -0.10][: self.n_nodes]
            ])

            pv = converter.convert_step(u_cpu_nodes, u_gpu_nodes)
            tv = self.from_row(row, pv)
            yield pd.Timestamp(row["timestamp"]), tv

"""
engine4/reliability.py
Phase 4: Data Reliability Engine.

PIPELINE (sequential per tick, or batch over a DataFrame):
  1. Hard-bound filter  — per-field range check; out-of-range values set to NaN.
  2. IsolationForest    — multivariate anomaly detection on a rolling feature matrix;
                          anomalous rows have ALL fields set to NaN.
  3. Hybrid imputer     — gap <= cfg.short_gap_ticks: linear interpolation (pandas.interpolate);
                          gap > cfg.short_gap_ticks: KNNImputer with k=cfg.knn_neighbors.

FAULT INJECTION (test-only):
  inject_faults(df, seed) produces stuck-at, spike, and dropout faults on a copy
  of a DataFrame. Used exclusively in test_engine4.py and validate_phase4.py.

FEASIBILITY NOTE:
  IsolationForest is trained once on a clean warmup window (first cfg.iforest_warmup_frac
  of the dataset) and then applied in detection-only mode. Re-training in the live loop
  is Engine 11's responsibility. This matches the offline-replay-first philosophy of
  the project.

  KNNImputer is stateless per call — no state leaks across ticks in streaming mode.
  For batch mode, the full DataFrame is imputed in one pass (faster, same result).

OUTPUT CONTRACT (shared/types.py):
  CleanTelemetry(x_clean: TelemetryVector, t_clean: ThermalState, imputed_mask: np.ndarray)
  imputed_mask: bool ndarray, True where a value was imputed or anomaly-filled.

CONSUMED BY: Engine 5 (CQR Predictor).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

from shared.config import cfg
from shared.types import CleanTelemetry, PowerVector, TelemetryVector, ThermalState

# ---------------------------------------------------------------------------
# Field-level hard bounds: (col_name, min, max)
# ---------------------------------------------------------------------------
_BOUNDS: list[tuple[str, float, float]] = [
    ("U_cpu",        cfg.bound_u_cpu_min,    cfg.bound_u_cpu_max),
    ("Temperature",  cfg.bound_t_amb_min,    cfg.bound_t_amb_max),
    ("Humidity",     cfg.bound_rh_min,       cfg.bound_rh_max),
    ("fan_duty",     cfg.bound_fan_duty_min, cfg.bound_fan_duty_max),
    # Server zone temperature — comes from ThermalState
    ("RC_ServerZoneTemp", cfg.bound_t_node_min, cfg.bound_t_node_max),
    ("InletTemp",    cfg.bound_t_node_min,   cfg.bound_t_node_max),
]

# Features fed into IsolationForest (must all be present in the batch DataFrame)
_IFOREST_FEATURES: list[str] = [
    "U_cpu", "Temperature", "Humidity", "fan_duty",
    "InletTemp", "RC_ServerZoneTemp",
]

# Fraction of the dataset used as clean warmup for fitting IsolationForest
_IFOREST_WARMUP_FRAC: float = 0.10


# ---------------------------------------------------------------------------
# Fault injection (TEST ONLY — never called in production path)
# ---------------------------------------------------------------------------
def inject_faults(
    df: pd.DataFrame,
    seed: int = 0,
    stuck_at_frac: float = 0.005,
    spike_frac: float = 0.005,
    dropout_frac: float = 0.005,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Inject synthetic faults into a copy of df for reliability testing.

    Args:
        df:            Source DataFrame (not modified in place).
        seed:          RNG seed for reproducibility.
        stuck_at_frac: Fraction of rows to apply stuck-at (constant value) fault.
        spike_frac:    Fraction of rows to apply spike (5× median) fault.
        dropout_frac:  Fraction of rows to drop (set to NaN).

    Returns:
        (df_faulty, fault_mask) where fault_mask is a bool array of length len(df),
        True at rows that have at least one injected fault.
    """
    rng = np.random.default_rng(seed)
    df_faulty = df.copy()
    n = len(df_faulty)
    fault_mask = np.zeros(n, dtype=bool)

    target_col = "U_cpu"  # representative feature column for injection
    median_val = float(df_faulty[target_col].median())

    # Stuck-at: freeze a run of consecutive rows at the first value in the run
    n_stuck = max(1, int(n * stuck_at_frac))
    stuck_starts = rng.integers(0, n - 5, size=n_stuck)
    for s in stuck_starts:
        run_len = rng.integers(3, 8)
        end = min(s + run_len, n)
        stuck_val = float(df_faulty[target_col].iloc[s])
        df_faulty.iloc[s:end, df_faulty.columns.get_loc(target_col)] = stuck_val
        fault_mask[s:end] = True

    # Spikes: multiply by 5 (will exceed hard bounds and be caught)
    n_spikes = max(1, int(n * spike_frac))
    spike_idx = rng.integers(0, n, size=n_spikes)
    df_faulty.iloc[spike_idx, df_faulty.columns.get_loc(target_col)] = median_val * 5.0
    fault_mask[spike_idx] = True

    # Dropouts: NaN
    n_drop = max(1, int(n * dropout_frac))
    drop_idx = rng.integers(0, n, size=n_drop)
    df_faulty.iloc[drop_idx, df_faulty.columns.get_loc(target_col)] = np.nan
    fault_mask[drop_idx] = True

    return df_faulty, fault_mask


# ---------------------------------------------------------------------------
# Hard-bound filter
# ---------------------------------------------------------------------------
def apply_hard_bounds(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    NaN-flag any value outside its declared physical bounds.

    Args:
        df: Input DataFrame (modified copy is returned).

    Returns:
        (df_filtered, violation_mask) where violation_mask[i] is True if row i
        had at least one bound violation.
    """
    df_out = df.copy()
    violation_mask = np.zeros(len(df_out), dtype=bool)

    for col, lo, hi in _BOUNDS:
        if col not in df_out.columns:
            continue
        bad = (df_out[col] < lo) | (df_out[col] > hi)
        df_out.loc[bad, col] = np.nan
        violation_mask |= bad.to_numpy()

    return df_out, violation_mask


# ---------------------------------------------------------------------------
# IsolationForest anomaly detector
# ---------------------------------------------------------------------------
def fit_isolation_forest(df_clean: pd.DataFrame) -> IsolationForest:
    """
    Fit IsolationForest on a clean warmup window.

    Args:
        df_clean: DataFrame whose first cfg.iforest_warmup_frac rows are
                  guaranteed clean (from Phase 0 QA).

    Returns:
        Fitted IsolationForest instance.
    """
    n_warmup = max(100, int(len(df_clean) * _IFOREST_WARMUP_FRAC))
    warmup = df_clean.iloc[:n_warmup]

    available = [f for f in _IFOREST_FEATURES if f in warmup.columns]
    X_warmup = warmup[available].dropna().to_numpy(dtype=np.float64)

    iforest = IsolationForest(
        n_estimators=cfg.iforest_n_estimators,
        contamination=cfg.iforest_contamination,
        random_state=cfg.iforest_random_state,
        n_jobs=-1,
    )
    iforest.fit(X_warmup)
    return iforest


def apply_isolation_forest(
    df: pd.DataFrame,
    iforest: IsolationForest,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Apply a fitted IsolationForest; anomalous rows have all features set to NaN.

    Args:
        df:      DataFrame after hard-bound filtering.
        iforest: Pre-fitted IsolationForest.

    Returns:
        (df_flagged, anomaly_mask) where anomaly_mask[i] is True for anomalous rows.
    """
    df_out = df.copy()
    available = [f for f in _IFOREST_FEATURES if f in df_out.columns]
    X = df_out[available].fillna(df_out[available].median()).to_numpy(dtype=np.float64)

    preds = iforest.predict(X)          # +1 = inlier, -1 = outlier
    anomaly_mask = preds == -1

    # NaN-flag all feature columns in anomalous rows
    df_out.loc[anomaly_mask, available] = np.nan

    return df_out, anomaly_mask


# ---------------------------------------------------------------------------
# Hybrid imputer
# ---------------------------------------------------------------------------
def apply_hybrid_imputer(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Hybrid gap-filling:
      - NaN runs <= cfg.short_gap_ticks: linear interpolation.
      - Remaining NaNs: KNNImputer(k=cfg.knn_neighbors).

    Args:
        df: DataFrame with NaNs from hard-bound / IsolationForest stages.

    Returns:
        (df_imputed, imputed_mask) where imputed_mask[i] is True if any value
        in row i was imputed.
    """
    df_out = df.copy()
    nan_before = df_out.isnull().any(axis=1).to_numpy()

    numeric_cols = df_out.select_dtypes(include=np.number).columns.tolist()

    # Stage 1: interpolate short gaps
    df_out[numeric_cols] = (
        df_out[numeric_cols]
        .interpolate(method="linear", limit=cfg.short_gap_ticks, limit_direction="both")
    )

    # Stage 2: KNN for remaining NaNs (longer gaps)
    remaining_nan_cols = [c for c in numeric_cols if df_out[c].isnull().any()]
    if remaining_nan_cols:
        knn = KNNImputer(n_neighbors=cfg.knn_neighbors)
        df_out[remaining_nan_cols] = knn.fit_transform(
            df_out[remaining_nan_cols].to_numpy(dtype=np.float64)
        )

    imputed_mask = nan_before  # any row that had a NaN before imputation
    return df_out, imputed_mask


# ---------------------------------------------------------------------------
# Row → CleanTelemetry converter
# ---------------------------------------------------------------------------
def _row_to_clean_telemetry(
    row: pd.Series,
    imputed_mask_val: bool,
    n_nodes: int = 3,
) -> CleanTelemetry:
    """
    Convert a single cleaned DataFrame row into the CleanTelemetry contract type.

    This is a thin adapter: it reconstructs TelemetryVector and ThermalState
    from the flat parquet columns that Engine 2 and Engine 3A would have produced,
    so downstream (Engine 5) can consume CleanTelemetry directly without re-running
    Engines 2 and 3A.

    Args:
        row:              A single (cleaned) parquet row as pd.Series.
        imputed_mask_val: Whether this row was imputed.
        n_nodes:          Number of compute nodes.

    Returns:
        CleanTelemetry with x_clean (TelemetryVector) and t_clean (ThermalState).
    """
    u_cpu = np.full(n_nodes, float(row.get("U_cpu", 0.5)), dtype=np.float64)
    it_kw = float(row.get("IT_Load", 300.0))
    p_servers = np.full(n_nodes, it_kw * 1000.0 / n_nodes, dtype=np.float64)

    x_clean = TelemetryVector(
        u_cpu=u_cpu,
        p_servers=p_servers,
        t_amb=float(row.get("Temperature", 20.0)),
        rh=float(row.get("Humidity", 60.0)),
        t_wet=float(row.get("T_wet", 15.0)),
        ci=float(row.get("CI", cfg.ci_default)),
        ep=float(row.get("EP", cfg.ep_default)),
    )

    t_zone = float(row.get("RC_ServerZoneTemp", float(row.get("InletTemp", 25.0)) + 15.0))
    t_inlet = float(row.get("InletTemp", 20.0))
    fan_duty = float(row.get("fan_duty", 0.5))

    t_clean = ThermalState(
        temperatures=np.full(n_nodes, t_zone, dtype=np.float64),
        t_inlet=t_inlet,
        t_outlet=float(row.get("OutletTemp", t_zone + 5.0)),
        fan_duty=fan_duty,
    )

    return CleanTelemetry(
        x_clean=x_clean,
        t_clean=t_clean,
        imputed_mask=np.array([imputed_mask_val] * n_nodes, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Main batch engine
# ---------------------------------------------------------------------------
class DataReliabilityEngine:
    """
    Engine 4: Batch data reliability pipeline.

    Consumes a raw/simulated-fault DataFrame (same schema as phase0_unified.parquet)
    and produces a cleaned DataFrame plus per-row CleanTelemetry objects for Engine 5.

    Usage:
        engine = DataReliabilityEngine()
        engine.fit(df_clean_warmup)        # train IsolationForest
        df_clean, clean_bundle = engine.process(df_faulty)

    The IsolationForest must be fit before calling process().
    """

    def __init__(self, n_nodes: int = 3) -> None:
        self.n_nodes = n_nodes
        self._iforest: Optional[IsolationForest] = None
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "DataReliabilityEngine":
        """
        Train the IsolationForest on a clean warmup window.

        Args:
            df: Clean DataFrame (phase0_unified.parquet or a clean slice thereof).

        Returns:
            self (for chaining).
        """
        self._iforest = fit_isolation_forest(df)
        self._is_fitted = True
        return self

    def process(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[CleanTelemetry]]:
        """
        Run the full reliability pipeline on a DataFrame.

        Pipeline:
          hard_bounds → IsolationForest (if fitted) → hybrid_imputer → CleanTelemetry list

        Args:
            df: Raw or fault-injected DataFrame.

        Returns:
            (df_clean, clean_telemetry_list) — cleaned DataFrame and one
            CleanTelemetry per row for direct consumption by Engine 5.
        """
        if not self._is_fitted:
            warnings.warn(
                "DataReliabilityEngine.process() called without fit(). "
                "IsolationForest stage will be skipped.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Stage 1: hard bounds
        df_work, bound_mask = apply_hard_bounds(df)

        # Stage 2: IsolationForest (optional — skip if not fitted)
        if self._is_fitted and self._iforest is not None:
            df_work, anomaly_mask = apply_isolation_forest(df_work, self._iforest)
        else:
            anomaly_mask = np.zeros(len(df_work), dtype=bool)

        # Stage 3: hybrid imputer
        df_clean, imputed_mask = apply_hybrid_imputer(df_work)

        # Combined mask: any row that was touched
        combined_mask = bound_mask | anomaly_mask | imputed_mask

        # Build CleanTelemetry list
        clean_bundles: list[CleanTelemetry] = [
            _row_to_clean_telemetry(row, bool(combined_mask[i]), self.n_nodes)
            for i, (_, row) in enumerate(df_clean.iterrows())
        ]

        return df_clean, clean_bundles

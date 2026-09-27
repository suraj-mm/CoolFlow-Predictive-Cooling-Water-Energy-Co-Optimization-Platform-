"""
engine5/cqr_predictor.py
Phase 5: CQR Thermal Predictive Forecasting Engine.

ARCHITECTURE:
  One ConformalizedQuantileRegressor (MAPIE 1.5.0) per horizon per node, wrapping
  a LGBMRegressor(objective='quantile'). This is the native, supported combination
  documented in MAPIE and produces calibrated intervals without two separate pipelines.

  Horizons: tau=1 tick (10 min), tau=2 ticks (20 min).
  Per-node models: 3 separate forecasters (one per compute node in the 3-node cluster).

WORKFLOW:
  1. build_feature_matrix(df)  — extract lag features from the cleaned parquet.
  2. CQRPredictor.fit(df)      — split 80/10/10, train + conformalize each model.
  3. CQRPredictor.predict(ct)  — predict from a live CleanTelemetry → PredictionBundle.
  4. CQRPredictor.evaluate_coverage(df_test) — the mandatory coverage verification.

FEATURE SET (per node):
  Lag-1 T_node, Lag-2 T_node, Lag-1 P_server, Lag-2 P_server,
  Lag-1 U_cpu,  Lag-2 U_cpu,
  T_amb, RH, T_wet, fan_duty, CI, EP
  → 12 + (4 * lag_steps additional features) = 12 features total at lag_steps=2

SPLIT STRATEGY (time-ordered, no data leakage):
  train_end  = int(N * cfg.cqr_train_frac)
  cal_end    = int(N * (cfg.cqr_train_frac + cfg.cqr_cal_frac))
  test       = df[cal_end:]
  MAPIE.fit() on train, MAPIE.conformalize() on cal, evaluate on test.

EXIT CRITERION:
  Empirical coverage on held-out test window must be within ±cfg.cqr_coverage_tol
  of cfg.cqr_confidence_level (e.g. 0.90 ± 0.05). Verified by evaluate_coverage().

OUTPUT CONTRACT (shared/types.py):
  PredictionBundle(t_mid, t_low, t_high, confidence_score) — shape (N,) per field.
  confidence_score = 100 * max(0, 1 - interval_width / cfg.delta_max)
                   where cfg.delta_max is the width that maps to 0% confidence.

CONSUMED BY: Engine 6 (Hotspot Detector), Engine 7 (Optimizer), Engine 9 (Explainability).
RETRAINED BY: Engine 11 (Experience Replay).
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mapie.regression import ConformalizedQuantileRegressor

from shared.config import cfg
from shared.io import save_artifact, load_artifact
from shared.types import CleanTelemetry, PredictionBundle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Feature columns read directly from the (cleaned) parquet
_STATIC_FEATURES: list[str] = [
    "Temperature", "Humidity", "T_wet", "fan_duty", "CI", "EP",
]
# Columns for which lag features are derived
_LAG_SOURCE_COLS: list[str] = ["RC_ServerZoneTemp", "IT_Load", "U_cpu"]

# Mapping: node index → target column in parquet (proxy for T_node)
_NODE_TARGET_COLS: list[str] = [
    "RC_ServerZoneTemp",   # node 0 (zone average)
    "RC_CPUTemp",          # node 1 (CPU die temp — higher, different dynamics)
    "InletTemp",           # node 2 (inlet, lower bound — tests interval spread)
]

# Prediction horizons in ticks (10-min cadence → tick × 10 min)
_HORIZONS: list[int] = [1, 2]   # 10 min, 20 min
_HORIZON_LABELS: list[str] = ["tau1", "tau2"]

_ARTIFACT_NAME = "engine5_cqr_models"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the lag-feature matrix from a cleaned DataFrame.

    Produces one row per original row (NaN in the first cfg.cqr_lag_steps rows
    due to lag generation — these are dropped before model fitting).

    Args:
        df: Cleaned parquet DataFrame (phase0_unified.parquet schema).

    Returns:
        Feature DataFrame with columns: static features + lag features.
        Index-aligned with df; first cfg.cqr_lag_steps rows are NaN.
    """
    feat = df[_STATIC_FEATURES].copy()

    for col in _LAG_SOURCE_COLS:
        if col not in df.columns:
            continue
        for lag in range(1, cfg.cqr_lag_steps + 1):
            feat[f"{col}_lag{lag}"] = df[col].shift(lag)

    return feat


def _get_targets(df: pd.DataFrame, node_idx: int, horizon: int) -> pd.Series:
    """
    Future temperature for a given node and horizon.

    T_node[t + horizon] — shifted backward so index aligns with current features.
    """
    col = _NODE_TARGET_COLS[node_idx]
    return df[col].shift(-horizon)


# ---------------------------------------------------------------------------
# Confidence score formula
# ---------------------------------------------------------------------------
def _compute_confidence_scores(t_low: np.ndarray, t_high: np.ndarray) -> np.ndarray:
    """
    Map prediction interval width to a [0, 100] confidence score.

    Score = 100 × max(0, 1 - width / cfg.delta_max)
    A narrow interval (width → 0) gives score ≈ 100.
    At width = cfg.delta_max, score = 0.
    """
    width = t_high - t_low
    score = 100.0 * np.clip(1.0 - width / cfg.delta_max, 0.0, 1.0)
    return score


# ---------------------------------------------------------------------------
# Single-horizon, single-node forecaster
# ---------------------------------------------------------------------------
def _make_lgbm_base() -> LGBMRegressor:
    """Construct the LGBMRegressor base for MAPIE's ConformalizedQuantileRegressor."""
    return LGBMRegressor(
        objective="quantile",
        alpha=0.5,                          # initial alpha; MAPIE overrides per quantile
        n_estimators=cfg.cqr_lgbm_n_estimators,
        num_leaves=cfg.cqr_lgbm_num_leaves,
        learning_rate=cfg.cqr_lgbm_learning_rate,
        verbose=-1,
        n_jobs=-1,
    )


def _fit_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> ConformalizedQuantileRegressor:
    """
    Fit and conformalize one CQR model.

    Args:
        X_train, y_train: Training features and target.
        X_cal, y_cal:     Conformalization features and target.

    Returns:
        Fitted + conformalized ConformalizedQuantileRegressor.
    """
    cqr = ConformalizedQuantileRegressor(
        estimator=_make_lgbm_base(),
        confidence_level=cfg.cqr_confidence_level,
    )
    cqr.fit(X_train, y_train)
    cqr.conformalize(X_cal, y_cal)
    return cqr


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------
class CQRPredictor:
    """
    Engine 5: CQR Thermal Predictive Forecasting Engine.

    Trains one ConformalizedQuantileRegressor per (horizon × node) combination
    and produces calibrated prediction intervals for downstream engines.

    Attributes:
        models: dict mapping (horizon_label, node_idx) → ConformalizedQuantileRegressor.
        n_nodes: Number of compute nodes.
        _feature_cols: Ordered list of feature column names (set after fit).
    """

    def __init__(self, n_nodes: int = 3) -> None:
        self.n_nodes = n_nodes
        self.models: dict[tuple[str, int], ConformalizedQuantileRegressor] = {}
        self._feature_cols: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "CQRPredictor":
        """
        Build feature matrix, split 80/10/10 (time-ordered), and train all models.

        Args:
            df: Full cleaned DataFrame (phase0_unified.parquet schema).

        Returns:
            self (for chaining).
        """
        feat_df = build_feature_matrix(df)

        n = len(feat_df)
        train_end = int(n * cfg.cqr_train_frac)
        cal_end = int(n * (cfg.cqr_train_frac + cfg.cqr_cal_frac))

        # Store ordered feature columns for later prediction alignment
        self._feature_cols = feat_df.columns.tolist()

        X_all = feat_df.to_numpy(dtype=np.float64)
        X_train = X_all[:train_end]
        X_cal = X_all[train_end:cal_end]

        for h_idx, (label, horizon) in enumerate(zip(_HORIZON_LABELS, _HORIZONS)):
            for node_idx in range(self.n_nodes):
                y_series = _get_targets(df, node_idx, horizon)
                y_all = y_series.to_numpy(dtype=np.float64)

                # Align valid mask: drop rows where X has NaN (first lag_steps rows)
                # OR where y has NaN (last horizon rows)
                valid = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all)

                # Train split valid mask
                train_valid = valid[:train_end]
                X_tr = X_train[train_valid]
                y_tr = y_all[:train_end][train_valid]

                # Cal split valid mask
                cal_valid = valid[train_end:cal_end]
                X_ca = X_cal[cal_valid]
                y_ca = y_all[train_end:cal_end][cal_valid]

                if len(X_tr) < 100 or len(X_ca) < 10:
                    warnings.warn(
                        f"Insufficient data for model ({label}, node {node_idx}): "
                        f"train={len(X_tr)}, cal={len(X_ca)}. Skipping.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue

                self.models[(label, node_idx)] = _fit_one(X_tr, y_tr, X_ca, y_ca)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict from CleanTelemetry (live streaming path)
    # ------------------------------------------------------------------
    def predict(
        self,
        clean_telemetry: CleanTelemetry,
        history_window: Optional[np.ndarray] = None,
        horizon_label: str = "tau1",
    ) -> PredictionBundle:
        """
        Produce a PredictionBundle from a single CleanTelemetry tick.

        In streaming mode, lag features require a short history buffer. When
        history_window is None, lags default to the current tick's values
        (conservative approximation for the first few ticks after startup).

        Args:
            clean_telemetry: E4 CleanTelemetry for the current tick.
            history_window:  Optional ndarray shape (lag_steps, n_features) with
                             prior feature rows for lag construction.
            horizon_label:   "tau1" (10 min) or "tau2" (20 min).

        Returns:
            PredictionBundle with t_mid, t_low, t_high, confidence_score (shape (N,)).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "CQRPredictor.predict() called before fit(). Call fit(df) first."
            )

        ct = clean_telemetry
        t_amb = ct.x_clean.t_amb
        rh = ct.x_clean.rh
        t_wet = ct.x_clean.t_wet
        fan_duty = ct.x_clean.u_cpu.mean()   # proxy for rack-level duty in stream mode
        ci = ct.x_clean.ci
        ep = ct.x_clean.ep

        t_mid_list, t_low_list, t_high_list = [], [], []

        for node_idx in range(self.n_nodes):
            key = (horizon_label, node_idx)
            if key not in self.models:
                # Model missing — emit NaN-safe defaults
                t_mid_list.append(ct.t_clean.t_inlet + 15.0)
                t_low_list.append(ct.t_clean.t_inlet + 10.0)
                t_high_list.append(ct.t_clean.t_inlet + 25.0)
                continue

            t_node_now = float(ct.t_clean.temperatures[node_idx])
            p_node_now = float(ct.x_clean.p_servers[node_idx])
            u_cpu_now = float(ct.x_clean.u_cpu[node_idx])

            # Build static features
            static = [t_amb, rh, t_wet, fan_duty, ci, ep]

            # Lag features: use history if available, else current as approximation
            lags: list[float] = []
            if history_window is not None and history_window.shape[0] >= cfg.cqr_lag_steps:
                # history_window[-1] = most recent past tick, [-2] = two ticks ago
                for lag in range(1, cfg.cqr_lag_steps + 1):
                    hw_row = history_window[-(lag)]
                    lags.extend([
                        hw_row[0],   # RC_ServerZoneTemp_lag{lag}
                        hw_row[1],   # IT_Load_lag{lag}
                        hw_row[2],   # U_cpu_lag{lag}
                    ])
            else:
                # Fallback: use current values for all lags (cold-start safe)
                for _ in range(cfg.cqr_lag_steps):
                    lags.extend([t_node_now, p_node_now / 1000.0, u_cpu_now])

            x_row = np.array([static + lags], dtype=np.float64)

            # Pad or trim to match training feature count if needed
            n_expected = len(self._feature_cols)
            if x_row.shape[1] < n_expected:
                pad = np.zeros((1, n_expected - x_row.shape[1]))
                x_row = np.concatenate([x_row, pad], axis=1)
            elif x_row.shape[1] > n_expected:
                x_row = x_row[:, :n_expected]

            y_pred, intervals = self.models[key].predict_interval(x_row)
            t_mid_list.append(float(y_pred[0]))
            t_low_list.append(float(intervals[0, 0, 0]))
            t_high_list.append(float(intervals[0, 1, 0]))

        t_mid = np.array(t_mid_list, dtype=np.float64)
        t_low = np.array(t_low_list, dtype=np.float64)
        t_high = np.array(t_high_list, dtype=np.float64)
        cs = _compute_confidence_scores(t_low, t_high)

        return PredictionBundle(
            t_mid=t_mid,
            t_low=t_low,
            t_high=t_high,
            confidence_score=cs,
        )

    # ------------------------------------------------------------------
    # Batch predict over a test DataFrame (used by validate_phase5.py)
    # ------------------------------------------------------------------
    def predict_batch(
        self,
        feat_df: pd.DataFrame,
        horizon_label: str = "tau1",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch prediction over a pre-built feature DataFrame.

        Args:
            feat_df:       Feature DataFrame from build_feature_matrix().
            horizon_label: "tau1" or "tau2".

        Returns:
            (y_mid, y_low, y_high) — each shape (n_valid, n_nodes).
            Rows with NaN features are excluded; valid_mask is returned implicitly
            (align by calling np.isfinite on feat_df before comparing to y_true).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_batch().")

        X = feat_df.to_numpy(dtype=np.float64)
        valid = np.isfinite(X).all(axis=1)
        X_valid = X[valid]

        n_valid = X_valid.shape[0]
        y_mid = np.full((n_valid, self.n_nodes), np.nan)
        y_low = np.full((n_valid, self.n_nodes), np.nan)
        y_high = np.full((n_valid, self.n_nodes), np.nan)

        for node_idx in range(self.n_nodes):
            key = (horizon_label, node_idx)
            if key not in self.models:
                continue
            y_pred, intervals = self.models[key].predict_interval(X_valid)
            y_mid[:, node_idx] = y_pred
            y_low[:, node_idx] = intervals[:, 0, 0]
            y_high[:, node_idx] = intervals[:, 1, 0]

        return y_mid, y_low, y_high

    # ------------------------------------------------------------------
    # Coverage evaluation (exit criterion)
    # ------------------------------------------------------------------
    def evaluate_coverage(
        self,
        df_test: pd.DataFrame,
        horizon_label: str = "tau1",
    ) -> dict[str, float]:
        """
        Compute empirical coverage on a held-out test DataFrame.

        Empirical coverage = fraction of true values inside [t_low, t_high].
        Must be within ±cfg.cqr_coverage_tol of cfg.cqr_confidence_level.

        Args:
            df_test:       Held-out test DataFrame (same schema as training data).
            horizon_label: "tau1" or "tau2".

        Returns:
            dict with keys:
              "coverage_<node_idx>" — per-node empirical coverage
              "coverage_mean"       — mean across nodes
              "nominal"             — cfg.cqr_confidence_level
              "tolerance"           — cfg.cqr_coverage_tol
              "all_pass"            — bool: all nodes within tolerance
        """
        feat_df = build_feature_matrix(df_test)
        X = feat_df.to_numpy(dtype=np.float64)
        valid_feat = np.isfinite(X).all(axis=1)

        horizon_idx = _HORIZON_LABELS.index(horizon_label)
        horizon = _HORIZONS[horizon_idx]

        results: dict[str, float] = {
            "nominal": cfg.cqr_confidence_level,
            "tolerance": cfg.cqr_coverage_tol,
        }
        coverages: list[float] = []

        for node_idx in range(self.n_nodes):
            key = (horizon_label, node_idx)
            if key not in self.models:
                continue

            y_true_series = _get_targets(df_test, node_idx, horizon)
            y_true_all = y_true_series.to_numpy(dtype=np.float64)
            valid_y = np.isfinite(y_true_all)
            valid = valid_feat & valid_y

            if valid.sum() == 0:
                continue

            X_valid = X[valid]
            y_true = y_true_all[valid]

            _, intervals = self.models[key].predict_interval(X_valid)
            y_low = intervals[:, 0, 0]
            y_high = intervals[:, 1, 0]

            covered = np.mean((y_true >= y_low) & (y_true <= y_high))
            results[f"coverage_node{node_idx}"] = float(covered)
            coverages.append(covered)

        if coverages:
            mean_cov = float(np.mean(coverages))
            results["coverage_mean"] = mean_cov
            nominal = cfg.cqr_confidence_level
            tol = cfg.cqr_coverage_tol
            results["all_pass"] = all(
                abs(c - nominal) <= tol
                for c in coverages
            )
        else:
            results["coverage_mean"] = float("nan")
            results["all_pass"] = False

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, name: str = _ARTIFACT_NAME) -> Path:
        """Persist the fitted predictor to data/artifacts/."""
        return save_artifact(self, name)

    @classmethod
    def load(cls, name: str = _ARTIFACT_NAME) -> "CQRPredictor":
        """Load a previously saved CQRPredictor."""
        obj = load_artifact(name)
        if not isinstance(obj, cls):
            raise TypeError(f"Artifact '{name}' is not a CQRPredictor.")
        return obj

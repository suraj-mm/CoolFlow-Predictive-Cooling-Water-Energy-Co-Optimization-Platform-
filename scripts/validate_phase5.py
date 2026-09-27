"""
scripts/validate_phase5.py
End-to-end validation for Phase 5 (CQR Thermal Predictive Forecasting Engine).

Checks (all must pass):
  5-1. Feature matrix has correct column count.
  5-2. Fit completes on full dataset (all 6 models fitted).
  5-3. y_low <= y_mid everywhere on test batch (ordering invariant).
  5-4. y_mid <= y_high everywhere on test batch (ordering invariant).
  5-5. tau1 empirical coverage within ±5% of 90% nominal on held-out test.
  5-6. tau2 empirical coverage within ±5% of 90% nominal on held-out test.
  5-7. Confidence scores in [0, 100].
  5-8. Predicted temperatures physically plausible (-20 to 200 °C).
  5-9. predict() from CleanTelemetry returns PredictionBundle with shape (3,).
  5-10. save() + load() round-trip preserves predict output.
"""
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from shared.io import load_dataset
from shared.config import cfg
from shared.types import CleanTelemetry, TelemetryVector, ThermalState, PredictionBundle
from engine5.cqr_predictor import (
    CQRPredictor,
    build_feature_matrix,
    _compute_confidence_scores,
    _HORIZON_LABELS,
)

PASS_STR = "  \033[92m[PASS]\033[0m"
FAIL_STR = "  \033[91m[FAIL]\033[0m"
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"{PASS_STR} {name}" + (f" — {detail}" if detail else ""))
    else:
        print(f"{FAIL_STR} {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


print("\n--- Phase 5: CQR Thermal Predictive Forecasting Engine ---")

# Use 10k rows for a realistic train/cal/test split (8000/1000/1000)
_N = 10000
df = load_dataset().iloc[:_N].reset_index(drop=True)

n = len(df)
cal_end = int(n * (cfg.cqr_train_frac + cfg.cqr_cal_frac))
df_test = df.iloc[cal_end:].reset_index(drop=True)
feat_df_test = build_feature_matrix(df_test)

# ---------------------------------------------------------------------------
# 5-1: Feature matrix column count
# ---------------------------------------------------------------------------
feat_df_full = build_feature_matrix(df)
n_static = 6
n_lag_cols = 3 * cfg.cqr_lag_steps   # 3 source cols × lag_steps
expected_cols = n_static + n_lag_cols
check("5-1 Feature matrix has correct column count",
      len(feat_df_full.columns) == expected_cols,
      f"expected={expected_cols}, got={len(feat_df_full.columns)}")

# ---------------------------------------------------------------------------
# 5-2: Fit completes; all model keys present
# ---------------------------------------------------------------------------
print("  Fitting CQR models (this may take ~30-60s)...")
t0 = time.time()
predictor = CQRPredictor(n_nodes=3)
predictor.fit(df)
fit_time = time.time() - t0

expected_keys = {(label, node_idx) for label in _HORIZON_LABELS for node_idx in range(3)}
keys_ok = expected_keys.issubset(set(predictor.models.keys()))
check("5-2 All 6 CQR models fitted (2 horizons × 3 nodes)",
      keys_ok and predictor._is_fitted,
      f"fitted in {fit_time:.1f}s, keys={len(predictor.models)}/6")

# ---------------------------------------------------------------------------
# 5-3 & 5-4: Ordering invariants on batch predictions
# ---------------------------------------------------------------------------
y_mid, y_low, y_high = predictor.predict_batch(feat_df_test, horizon_label="tau1")

check("5-3 y_low <= y_mid everywhere (tau1 batch)",
      np.all(y_low <= y_mid + 1e-6),
      f"violations: {int(np.sum(y_low > y_mid + 1e-6))}")

check("5-4 y_mid <= y_high everywhere (tau1 batch)",
      np.all(y_mid <= y_high + 1e-6),
      f"violations: {int(np.sum(y_mid > y_high + 1e-6))}")

# ---------------------------------------------------------------------------
# 5-5: tau1 empirical coverage (THE KEY EXIT CRITERION)
# ---------------------------------------------------------------------------
cov_results_1 = predictor.evaluate_coverage(df_test, horizon_label="tau1")
mean_cov_1 = cov_results_1.get("coverage_mean", float("nan"))
nominal = cfg.cqr_confidence_level
tol = cfg.cqr_coverage_tol

check("5-5 tau1 empirical coverage within ±5% of 90% nominal",
      cov_results_1.get("all_pass", False),
      f"mean_coverage={mean_cov_1:.3f}, nominal={nominal}, tol=±{tol}")

# Also print per-node coverage for diagnostics
for node_idx in range(3):
    key = f"coverage_node{node_idx}"
    if key in cov_results_1:
        c = cov_results_1[key]
        status = "ok" if abs(c - nominal) <= tol else "FAIL"
        print(f"    node{node_idx}: coverage={c:.3f} [{status}]")

# ---------------------------------------------------------------------------
# 5-6: tau2 empirical coverage
# ---------------------------------------------------------------------------
cov_results_2 = predictor.evaluate_coverage(df_test, horizon_label="tau2")
mean_cov_2 = cov_results_2.get("coverage_mean", float("nan"))
check("5-6 tau2 empirical coverage within ±5% of 90% nominal",
      cov_results_2.get("all_pass", False),
      f"mean_coverage={mean_cov_2:.3f}, nominal={nominal}, tol=±{tol}")

# ---------------------------------------------------------------------------
# 5-7: Confidence scores in [0, 100]
# ---------------------------------------------------------------------------
cs = _compute_confidence_scores(y_low.flatten(), y_high.flatten())
check("5-7 All confidence scores in [0, 100]",
      bool(np.all((cs >= 0.0) & (cs <= 100.0))),
      f"min={cs.min():.2f}, max={cs.max():.2f}")

# ---------------------------------------------------------------------------
# 5-8: Predicted temperatures physically plausible
# ---------------------------------------------------------------------------
check("5-8 All tau1 predictions in physical range (-20, 200) °C",
      bool(np.all(y_mid > -20.0) and np.all(y_mid < 200.0)),
      f"min={y_mid.min():.1f}°C, max={y_mid.max():.1f}°C")

# ---------------------------------------------------------------------------
# 5-9: predict() from CleanTelemetry
# ---------------------------------------------------------------------------
row = df.iloc[200]
ct = CleanTelemetry(
    x_clean=TelemetryVector(
        u_cpu=np.array([0.7, 0.75, 0.65]),
        p_servers=np.array([300.0, 310.0, 290.0]),
        t_amb=float(row["Temperature"]),
        rh=float(row["Humidity"]),
        t_wet=float(row["T_wet"]),
        ci=float(row["CI"]),
        ep=float(row["EP"]),
    ),
    t_clean=ThermalState(
        temperatures=np.array([40.0, 75.0, 23.0]),
        t_inlet=float(row["InletTemp"]),
        t_outlet=float(row["OutletTemp"]),
        fan_duty=float(row["fan_duty"]),
    ),
    imputed_mask=np.array([False, False, False]),
)

bundle = predictor.predict(ct, horizon_label="tau1")
shape_ok = (
    bundle.t_mid.shape == (3,) and
    bundle.t_low.shape == (3,) and
    bundle.t_high.shape == (3,) and
    bundle.confidence_score.shape == (3,)
)
check("5-9 predict(CleanTelemetry) returns PredictionBundle shape (3,)",
      isinstance(bundle, PredictionBundle) and shape_ok,
      f"t_mid={bundle.t_mid.round(1)}, cs={bundle.confidence_score.round(1)}")

# ---------------------------------------------------------------------------
# 5-10: Save + load round-trip
# ---------------------------------------------------------------------------
try:
    saved_path = predictor.save()
    loaded = CQRPredictor.load()
    bundle2 = loaded.predict(ct, horizon_label="tau1")
    roundtrip_ok = np.allclose(bundle.t_mid, bundle2.t_mid, atol=1e-6)
    check("5-10 save()/load() round-trip preserves predict output",
          roundtrip_ok,
          f"saved to {saved_path.name}")
except Exception as e:
    check("5-10 save()/load() round-trip preserves predict output",
          False, f"exception: {e}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
if not failures:
    print("  ALL CHECKS PASSED — Phase 5 exit criteria met.")
else:
    print(f"  {len(failures)} CHECK(S) FAILED: {failures}")
print(f"{'='*55}\n")
sys.exit(0 if not failures else 1)

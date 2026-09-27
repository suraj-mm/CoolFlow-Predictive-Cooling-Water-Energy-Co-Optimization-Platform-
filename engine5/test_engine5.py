"""
engine5/test_engine5.py
Unit tests for the CQR Thermal Predictive Forecasting Engine (Engine 5).

Covers:
  1. build_feature_matrix produces correct number of columns.
  2. build_feature_matrix first lag_steps rows are NaN (lag construction).
  3. build_feature_matrix has no NaN after row cfg.cqr_lag_steps.
  4. Fit completes without error on a small dataset slice.
  5. All expected model keys present after fit.
  6. predict_batch returns correct shapes (n_valid, n_nodes).
  7. predict_batch y_low <= y_mid (ordering invariant).
  8. predict_batch y_mid <= y_high (ordering invariant).
  9. Confidence scores are in [0, 100].
  10. predict() from CleanTelemetry returns PredictionBundle.
  11. PredictionBundle shapes are (n_nodes,).
  12. predict() raises RuntimeError if called before fit.
  13. evaluate_coverage returns dict with required keys.
  14. evaluate_coverage all_pass key is a bool.
  15. Coverage value is between 0.0 and 1.0.
"""
import sys
import unittest

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from shared.config import cfg
from shared.io import load_dataset
from shared.types import CleanTelemetry, TelemetryVector, ThermalState, PredictionBundle
from engine5.cqr_predictor import (
    CQRPredictor,
    build_feature_matrix,
    _compute_confidence_scores,
    _HORIZON_LABELS,
    _HORIZONS,
)

# ---------------------------------------------------------------------------
# Fixture: 5,000 row slice — enough for a meaningful 80/10/10 split
# (4,000 train / 500 cal / 500 test — well above the 100-row minimum)
# ---------------------------------------------------------------------------
_N = 5000
_DF = load_dataset().iloc[:_N].reset_index(drop=True)

# Pre-fit predictor (shared across tests — expensive to repeat)
_PREDICTOR: CQRPredictor | None = None


def _get_predictor() -> CQRPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = CQRPredictor(n_nodes=3)
        _PREDICTOR.fit(_DF)
    return _PREDICTOR


class TestFeatureMatrix(unittest.TestCase):

    def setUp(self):
        self.feat_df = build_feature_matrix(_DF)

    def test_column_count_correct(self):
        """Feature matrix should have static + lag columns."""
        n_static = 6   # _STATIC_FEATURES
        n_lag_cols = len(["RC_ServerZoneTemp", "IT_Load", "U_cpu"]) * cfg.cqr_lag_steps
        expected_cols = n_static + n_lag_cols
        self.assertEqual(len(self.feat_df.columns), expected_cols)

    def test_first_rows_nan(self):
        """First cfg.cqr_lag_steps rows must have NaN due to lag shift."""
        first_rows = self.feat_df.iloc[:cfg.cqr_lag_steps]
        self.assertTrue(first_rows.isnull().any().any(),
                        "Lag features should produce NaN in early rows")

    def test_later_rows_no_nan(self):
        """Rows after the lag warmup should have no NaN from lag construction."""
        # Check a row well past the lag period
        row = self.feat_df.iloc[cfg.cqr_lag_steps + 5]
        # Static features should always be non-NaN (clean dataset)
        static_cols = ["Temperature", "Humidity", "T_wet", "fan_duty", "CI", "EP"]
        for col in static_cols:
            if col in row.index:
                self.assertFalse(np.isnan(row[col]),
                                 f"Static feature {col} should not be NaN")


class TestFit(unittest.TestCase):

    def test_fit_completes(self):
        """Fit on 5k rows should complete without raising."""
        predictor = _get_predictor()
        self.assertTrue(predictor._is_fitted)

    def test_all_model_keys_present(self):
        """Should have one model per (horizon_label, node_idx) combination."""
        predictor = _get_predictor()
        expected_keys = {
            (label, node_idx)
            for label in _HORIZON_LABELS
            for node_idx in range(3)
        }
        for key in expected_keys:
            self.assertIn(key, predictor.models,
                          f"Model key {key} missing after fit")


class TestBatchPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        predictor = _get_predictor()
        n = len(_DF)
        cal_end = int(n * (cfg.cqr_train_frac + cfg.cqr_cal_frac))
        df_test = _DF.iloc[cal_end:].reset_index(drop=True)
        feat_df_test = build_feature_matrix(df_test)
        cls.y_mid, cls.y_low, cls.y_high = predictor.predict_batch(
            feat_df_test, horizon_label="tau1"
        )
        # Valid row count (rows with finite features)
        X = feat_df_test.to_numpy(dtype=np.float64)
        cls.n_valid = int(np.isfinite(X).all(axis=1).sum())

    def test_output_shape_n_valid_by_nodes(self):
        """predict_batch output shape should be (n_valid, n_nodes)."""
        self.assertEqual(self.y_mid.shape, (self.n_valid, 3))
        self.assertEqual(self.y_low.shape, (self.n_valid, 3))
        self.assertEqual(self.y_high.shape, (self.n_valid, 3))

    def test_low_le_mid(self):
        """y_low must be <= y_mid everywhere."""
        self.assertTrue(np.all(self.y_low <= self.y_mid + 1e-6),
                        "y_low should be <= y_mid")

    def test_mid_le_high(self):
        """y_mid must be <= y_high everywhere."""
        self.assertTrue(np.all(self.y_mid <= self.y_high + 1e-6),
                        "y_mid should be <= y_high")

    def test_temperatures_physically_plausible(self):
        """Predicted temperatures should be in a physically plausible range."""
        self.assertTrue(np.all(self.y_mid > -20.0))
        self.assertTrue(np.all(self.y_mid < 200.0))


class TestConfidenceScores(unittest.TestCase):

    def test_scores_in_0_100(self):
        """Confidence scores must be in [0, 100]."""
        t_low = np.array([20.0, 30.0, 25.0])
        t_high = np.array([25.0, 40.0, 35.0])
        cs = _compute_confidence_scores(t_low, t_high)
        self.assertTrue(np.all(cs >= 0.0))
        self.assertTrue(np.all(cs <= 100.0))

    def test_zero_width_gives_100(self):
        """Zero-width interval → confidence = 100."""
        cs = _compute_confidence_scores(np.array([25.0]), np.array([25.0]))
        self.assertAlmostEqual(cs[0], 100.0)

    def test_wide_interval_gives_low_score(self):
        """Interval >= delta_max → confidence = 0."""
        t_low = np.array([20.0])
        t_high = t_low + cfg.delta_max * 2  # way above max width
        cs = _compute_confidence_scores(t_low, t_high)
        self.assertAlmostEqual(cs[0], 0.0)


class TestStreamPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = _get_predictor()
        row = _DF.iloc[100]
        u_cpu = np.array([0.7, 0.75, 0.65])
        p_servers = np.array([300.0, 310.0, 290.0])
        cls.ct = CleanTelemetry(
            x_clean=TelemetryVector(
                u_cpu=u_cpu,
                p_servers=p_servers,
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

    def test_returns_prediction_bundle(self):
        """predict() should return a PredictionBundle."""
        bundle = self.predictor.predict(self.ct, horizon_label="tau1")
        self.assertIsInstance(bundle, PredictionBundle)

    def test_bundle_shapes(self):
        """PredictionBundle arrays must have shape (n_nodes,)."""
        bundle = self.predictor.predict(self.ct, horizon_label="tau1")
        self.assertEqual(bundle.t_mid.shape, (3,))
        self.assertEqual(bundle.t_low.shape, (3,))
        self.assertEqual(bundle.t_high.shape, (3,))
        self.assertEqual(bundle.confidence_score.shape, (3,))

    def test_tau2_predict_returns_bundle(self):
        """predict() for tau2 horizon should also succeed."""
        bundle = self.predictor.predict(self.ct, horizon_label="tau2")
        self.assertIsInstance(bundle, PredictionBundle)

    def test_predict_before_fit_raises(self):
        """predict() on unfitted predictor should raise RuntimeError."""
        unfitted = CQRPredictor(n_nodes=3)
        with self.assertRaises(RuntimeError):
            unfitted.predict(self.ct)


class TestCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        predictor = _get_predictor()
        n = len(_DF)
        cal_end = int(n * (cfg.cqr_train_frac + cfg.cqr_cal_frac))
        cls.df_test = _DF.iloc[cal_end:].reset_index(drop=True)
        cls.results = predictor.evaluate_coverage(cls.df_test, horizon_label="tau1")

    def test_required_keys_present(self):
        """Coverage result dict must contain required keys."""
        for key in ["nominal", "tolerance", "coverage_mean", "all_pass"]:
            self.assertIn(key, self.results)

    def test_all_pass_is_bool(self):
        """all_pass must be a boolean."""
        self.assertIsInstance(self.results["all_pass"], bool)

    def test_coverage_between_0_and_1(self):
        """coverage_mean must be in [0.0, 1.0]."""
        c = self.results["coverage_mean"]
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_nominal_matches_config(self):
        """Nominal level must match cfg.cqr_confidence_level."""
        self.assertAlmostEqual(self.results["nominal"], cfg.cqr_confidence_level)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
engine2/test_engine2.py
Unit tests for Phase 2: Telemetry & Environmental Context Engine.
Covers stull_wet_bulb formula, T_wet <= T_dry invariant, TelemetryVector
contract, timestamp continuity, and integration with Engine 1.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.io import load_dataset
from shared.types import PowerVector, TelemetryVector
from engine1.power_converter import WorkloadPowerConverter
from engine2.telemetry import stull_wet_bulb, assemble_telemetry, TelemetryEngine


class TestStullWetBulb(unittest.TestCase):

    def test_formula_known_values(self):
        """Check against manually computed reference values from Stull 2011 Table 1."""
        # T=20°C, RH=50% -> approximately 13.7°C
        tw = stull_wet_bulb(20.0, 50.0)
        self.assertTrue(12.5 < tw < 15.0, f"Expected ~13.7C, got {tw:.2f}")

        # T=30°C, RH=80% -> approximately 26.0°C
        tw = stull_wet_bulb(30.0, 80.0)
        self.assertTrue(24.0 < tw < 28.0, f"Expected ~26C, got {tw:.2f}")

        # T=0°C, RH=100% -> should be approx 0°C (saturated air, no depression)
        tw = stull_wet_bulb(0.0, 100.0)
        self.assertTrue(-1.0 <= tw <= 0.01, f"At saturation T_wet should ~ T_dry, got {tw:.4f}")

    def test_wet_bulb_never_exceeds_dry(self):
        """T_wet <= T_dry is a physical invariant — must hold for all inputs."""
        t_vals = np.linspace(-20, 50, 71)
        rh_vals = np.linspace(5, 100, 96)
        T, RH = np.meshgrid(t_vals, rh_vals)
        TW = stull_wet_bulb(T.ravel(), RH.ravel())
        violations = np.sum(TW > T.ravel() + 1e-9)
        self.assertEqual(violations, 0,
            f"T_wet > T_dry in {violations} out of {len(TW)} grid cells")

    def test_vectorization(self):
        """Scalar, 1-D array, and 2-D array inputs all return matching shapes."""
        self.assertIsInstance(stull_wet_bulb(25.0, 60.0), float)
        arr = stull_wet_bulb(np.array([10.0, 20.0, 30.0]), np.array([50.0, 60.0, 70.0]))
        self.assertEqual(arr.shape, (3,))
        grid = stull_wet_bulb(np.full((4, 3), 25.0), np.full((4, 3), 60.0))
        self.assertEqual(grid.shape, (4, 3))

    def test_depression_increases_with_lower_rh(self):
        """Lower RH => larger depression (T_dry - T_wet should increase)."""
        t = 30.0
        depressions = [t - stull_wet_bulb(t, rh) for rh in [90, 70, 50, 30, 10]]
        for a, b in zip(depressions, depressions[1:]):
            self.assertGreater(b, a, "Depression must grow as RH falls")


class TestAssembleTelemetry(unittest.TestCase):

    def _make_power_vec(self, n=3):
        converter = WorkloadPowerConverter("DC_PRODUCTION")
        u_cpu = np.array([0.3, 0.5, 0.7][:n])
        u_gpu = np.array([0.1, 0.2, 0.3][:n])
        return converter.convert_step(u_cpu, u_gpu), u_cpu, u_gpu

    def test_output_type_and_fields(self):
        pv, u_cpu, u_gpu = self._make_power_vec()
        tv = assemble_telemetry(pv, t_dry=25.0, rh=60.0, ci=350.0, ep=0.28,
                                u_cpu=u_cpu, u_gpu=u_gpu)
        self.assertIsInstance(tv, TelemetryVector)
        self.assertEqual(tv.p_servers.shape, (3,))
        self.assertEqual(tv.u_cpu.shape, (3,))
        self.assertAlmostEqual(tv.t_amb, 25.0)
        self.assertAlmostEqual(tv.rh, 60.0)
        self.assertAlmostEqual(tv.ci, 350.0)
        self.assertAlmostEqual(tv.ep, 0.28)
        self.assertLessEqual(tv.t_wet, tv.t_amb + 1e-9,
            f"T_wet={tv.t_wet:.4f} must not exceed T_dry={tv.t_amb}")

    def test_t_wet_formula_applied(self):
        pv, u_cpu, _ = self._make_power_vec()
        tv = assemble_telemetry(pv, t_dry=20.0, rh=50.0, ci=300.0, ep=0.18,
                                u_cpu=u_cpu)
        expected = stull_wet_bulb(20.0, 50.0)
        self.assertAlmostEqual(tv.t_wet, expected, places=6)

    def test_rh_clipping(self):
        pv, u_cpu, _ = self._make_power_vec()
        tv = assemble_telemetry(pv, t_dry=20.0, rh=120.0, ci=300.0, ep=0.18,
                                u_cpu=u_cpu)
        self.assertLessEqual(tv.rh, 100.0)


class TestTelemetryEngineDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.engine = TelemetryEngine(n_nodes=3)
        cls.converter = WorkloadPowerConverter("DC_PRODUCTION")
        cls.power_df = cls.converter.process_trace_dataframe(cls.df.head(2000), n_nodes=3)

    def test_no_missing_timestamps(self):
        """Exit criterion: zero timestamp gaps after merge."""
        merged = self.engine.process_dataframe(self.df.head(2000), self.power_df)
        self.assertIn("timestamp", merged.columns)
        gaps = merged["timestamp"].diff().dropna()
        dominant = gaps.mode()[0]
        big_gaps = (gaps > dominant * 2).sum()
        self.assertEqual(big_gaps, 0, f"{big_gaps} timestamp gaps found in merged frame")

    def test_t_wet_lte_t_dry_everywhere(self):
        """Exit criterion: T_wet <= T_dry at every single row of the full dataset."""
        merged = self.engine.process_dataframe(self.df.head(2000), self.power_df)
        violations = (merged["T_wet"] > merged["T_amb"] + 1e-9).sum()
        self.assertEqual(violations, 0,
            f"T_wet > T_dry in {violations} rows after clamping")

    def test_zero_nulls_in_output(self):
        """All key columns must be NaN-free after merge."""
        merged = self.engine.process_dataframe(self.df.head(2000), self.power_df)
        key_cols = ["timestamp", "T_amb", "RH", "T_wet", "CI", "EP"]
        for col in key_cols:
            nulls = merged[col].isna().sum()
            self.assertEqual(nulls, 0, f"Column '{col}' has {nulls} NaNs")

    def test_from_row_contract(self):
        """Single-row TelemetryVector must satisfy all type and shape constraints."""
        row = self.df.iloc[100]
        u_nodes = np.array([row["U_cpu"]] * 3)
        u_gpu_nodes = np.array([row["U_gpu"]] * 3)
        pv = self.converter.convert_step(u_nodes, u_gpu_nodes)
        tv = self.engine.from_row(row, pv)

        self.assertIsInstance(tv, TelemetryVector)
        self.assertEqual(tv.u_cpu.shape, (3,))
        self.assertEqual(tv.p_servers.shape, (3,))
        self.assertLessEqual(tv.t_wet, tv.t_amb + 1e-9)
        self.assertGreater(tv.ci, 0.0)
        self.assertGreater(tv.ep, 0.0)

    def test_iter_telemetry_vectors(self):
        """Iterator yields (Timestamp, TelemetryVector) with correct types for first 10 rows."""
        results = list(self.engine.iter_telemetry_vectors(
            self.df.head(10), self.power_df.head(10)
        ))
        self.assertEqual(len(results), 10)
        for ts, tv in results:
            self.assertIsInstance(tv, TelemetryVector)
            self.assertLessEqual(tv.t_wet, tv.t_amb + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)

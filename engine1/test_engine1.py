"""
engine1/test_engine1.py
Comprehensive unit test and validation suite for Phase 1 (Workload -> Power Engine).
Uses standard library unittest so it runs in any standard Python environment.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.config import cfg
from shared.io import load_dataset
from shared.types import PowerVector, ServerProfile
from engine1.power_converter import (
    calculate_server_power,
    WorkloadPowerConverter,
    SPEC_PROFILES,
)


class TestEngine1PowerConverter(unittest.TestCase):

    def test_boundary_values(self):
        """Verify idle, peak CPU, and peak GPU boundary values."""
        # Zero load -> idle power
        p_zero = calculate_server_power(u_cpu=0.0, u_gpu=0.0, p_idle=150.0, p_max=400.0, alpha=0.7)
        self.assertTrue(np.isclose(p_zero, 150.0), f"Expected 150.0W at U=0, got {p_zero}")

        # Peak CPU only -> p_max
        p_peak_cpu = calculate_server_power(u_cpu=1.0, u_gpu=0.0, p_idle=150.0, p_max=400.0, alpha=0.7)
        self.assertTrue(np.isclose(p_peak_cpu, 400.0), f"Expected 400.0W at U=1, got {p_peak_cpu}")

        # Peak CPU + Peak GPU -> p_max + p_gpu_max
        p_peak_all = calculate_server_power(
            u_cpu=1.0, u_gpu=1.0, p_idle=150.0, p_max=400.0, alpha=0.7, p_gpu_max=300.0
        )
        self.assertTrue(np.isclose(p_peak_all, 700.0), f"Expected 700.0W at full CPU+GPU load, got {p_peak_all}")

    def test_monotonicity_and_convexity(self):
        """Verify monotonic increase and superlinear shape."""
        u_vals = np.linspace(0.0, 1.0, 101)
        powers = calculate_server_power(u_cpu=u_vals, u_gpu=0.0, p_idle=150.0, p_max=400.0, alpha=0.7)

        # Strictly monotonic
        diffs = np.diff(powers)
        self.assertTrue(np.all(diffs > 0), "Power curve must be strictly increasing with utilization")

        # Convexity: second derivative should be non-negative for alpha < 1.0
        d2 = np.diff(diffs)
        self.assertTrue(np.all(d2 >= -1e-9), "Power curve must be convex for alpha < 1.0")

    def test_vectorization_and_clipping(self):
        """Verify support for arrays, node dimensions, and out-of-bound inputs."""
        # 3-node vector
        u_nodes = np.array([0.1, 0.5, 0.9])
        p_nodes = calculate_server_power(u_nodes)
        self.assertIsInstance(p_nodes, np.ndarray)
        self.assertEqual(p_nodes.shape, (3,))
        self.assertTrue(p_nodes[0] < p_nodes[1] < p_nodes[2])

        # Out-of-bounds clipping check (-0.5 clipped to 0, 1.5 clipped to 1.0)
        p_neg = calculate_server_power(-0.5)
        p_zero = calculate_server_power(0.0)
        self.assertTrue(np.isclose(p_neg, p_zero))

        p_over = calculate_server_power(1.5)
        p_peak = calculate_server_power(1.0)
        self.assertTrue(np.isclose(p_over, p_peak))

    def test_specpower_calibrations(self):
        """Verify calibration against real published SPECpower_ssj2008 benchmark records."""
        loads = np.linspace(0.0, 1.0, 11)

        # Published SPECpower benchmark records (0% to 100% in 10% steps)
        benchmarks = {
            "DELL_R740": np.array([65.2, 115.4, 148.1, 181.7, 218.3, 256.9, 298.4, 342.1, 388.5, 437.2, 485.0]),
            "HPE_DL380": np.array([72.1, 122.3, 155.0, 189.4, 225.8, 264.2, 305.1, 348.7, 395.2, 444.6, 492.0]),
            "LENOVO_SR650": np.array([74.5, 125.0, 158.4, 193.1, 230.5, 270.0, 311.8, 356.2, 403.1, 452.8, 501.2]),
        }

        for key, measured in benchmarks.items():
            profile = SPEC_PROFILES[key]
            pred = calculate_server_power(
                u_cpu=loads,
                u_gpu=0.0,
                p_idle=profile.p_idle,
                p_max=profile.p_max,
                alpha=profile.alpha,
            )
            mae = float(np.mean(np.abs(pred - measured)))
            self.assertLess(mae, 3.5, f"Calibration error for {profile.name} too high: MAE={mae:.2f}W >= 3.5W")

    def test_workload_power_converter_interface(self):
        """Verify PowerVector contract for E1 -> E2, E3A boundary."""
        converter = WorkloadPowerConverter("DC_PRODUCTION")
        u_cpu_nodes = np.array([0.25, 0.50, 0.75])
        u_gpu_nodes = np.array([0.10, 0.00, 0.40])

        p_vec = converter.convert_step(u_cpu_nodes, u_gpu_nodes, timestamp="2026-09-26T12:00:00Z")
        self.assertIsInstance(p_vec, PowerVector)
        self.assertEqual(p_vec.p_servers.shape, (3,))
        self.assertEqual(p_vec.p_cpu.shape, (3,))
        self.assertEqual(p_vec.p_gpu.shape, (3,))
        self.assertTrue(np.allclose(p_vec.p_servers, p_vec.p_cpu + p_vec.p_gpu))
        self.assertTrue(np.isclose(p_vec.total_power, np.sum(p_vec.p_servers)))
        self.assertEqual(p_vec.timestamp, "2026-09-26T12:00:00Z")

    def test_dataset_trace_execution(self):
        """Verify execution on actual Phase 0 unified dataset."""
        df = load_dataset()
        converter = WorkloadPowerConverter("DC_PRODUCTION")
        df_subset = df.head(500)

        trace_power = converter.process_trace_dataframe(df_subset, n_nodes=3)
        self.assertEqual(len(trace_power), 500)
        self.assertIn("P_node1", trace_power.columns)
        self.assertIn("P_node2", trace_power.columns)
        self.assertIn("P_node3", trace_power.columns)

        # Verify bounds and zero nulls
        for node in [1, 2, 3]:
            p_col = trace_power[f"P_node{node}"]
            self.assertEqual(p_col.isna().sum(), 0, f"Node {node} has NaNs")
            self.assertGreaterEqual(p_col.min(), cfg.p_idle, f"Node {node} power dropped below idle: {p_col.min()}W")
            self.assertLessEqual(p_col.max(), (cfg.p_max + cfg.p_gpu_max), f"Node {node} power exceeded peak: {p_col.max()}W")


if __name__ == "__main__":
    unittest.main(verbosity=2)

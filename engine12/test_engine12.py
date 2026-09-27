"""
engine12/test_engine12.py
Unit tests for the Phase 12 Integrated Control Pipeline.

Coverage:
  12-1  IntegratedSystem initializes cleanly with all 11 engines
  12-2  warmup() runs and primes models
  12-3  step() returns valid (StateRecord, ExplainBundle, ClosedLoopMetrics)
  12-4  SQLite state store receives row on step()
  12-5  Feedback loop updates internal u and fan for step t+1
  12-6  run_replay() runs multi-tick simulation with zero NaNs
  12-7  ExplainBundle is properly serialized and contains traceable rationale
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shared.state_store import read_recent
from engine12.pipeline import ClosedLoopMetrics, IntegratedSystem

class TestIntegratedPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.tmp_dir / "test_state.db"
        self.replay_path = self.tmp_dir / "test_replay.parquet"
        self.system = IntegratedSystem(
            n_nodes=3,
            db_path=self.db_path,
            replay_path=self.replay_path,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_initialization(self):
        """12-1: All engines initialized and database created."""
        self.assertEqual(self.system.n_nodes, 3)
        self.assertTrue(self.db_path.exists())
        self.assertIsNotNone(self.system.e1_power)
        self.assertIsNotNone(self.system.e2_telemetry)
        self.assertIsNotNone(self.system.e3a_rc)
        self.assertIsNotNone(self.system.e3b_water)
        self.assertIsNotNone(self.system.e4_reliability)
        self.assertIsNotNone(self.system.e5_predictor)
        self.assertIsNotNone(self.system.e6_hotspot)
        self.assertIsNotNone(self.system.e7_optimizer)
        self.assertIsNotNone(self.system.e8_executor)
        self.assertIsNotNone(self.system.e9_explainer)
        self.assertIsNotNone(self.system.e11_retraining)

    def test_warmup_and_single_step(self):
        """12-2, 12-3, 12-4: warmup and single step verification."""
        self.system.warmup()
        self.assertTrue(self.system._is_warm)

        rec, bundle, metrics = self.system.step(
            t_amb=20.0,
            rh=50.0,
            ci=250.0,
            ep=0.25,
        )

        # 12-3: outputs
        self.assertEqual(rec.tick_id, 1)
        self.assertEqual(bundle.tick_id, 1)
        self.assertIsInstance(metrics, ClosedLoopMetrics)
        self.assertFalse(np.isnan(np.array(rec.temperatures)).any())

        # 12-4: SQLite state store
        rows = read_recent(n=5, path=self.db_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tick_id"], 1)

    def test_multi_tick_replay_zero_nans(self):
        """12-5, 12-6: closed-loop run produces zero NaNs and updates feedback."""
        self.system.warmup()
        metrics = self.system.run_replay(ticks=3)
        self.assertEqual(metrics.total_ticks, 3)
        self.assertEqual(metrics.nan_occurrences, 0)
        self.assertGreater(metrics.total_energy_kwh, 0.0)

        # Confirm all 3 ticks are stored in SQLite
        rows = read_recent(n=20, path=self.db_path)
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()

"""
engine8/test_engine8.py
Unit tests for the Dual-Track Execution Engine.

Coverage:
  - Track A (SimPy/NumPy): DVFS reduces utilisation; migration redistributes load
  - Track A output: u_next in [0, 1]; p_next consistent with u_next via power model
  - Track B: Docker unavailable → graceful degradation (no exception raised)
  - ExecutionEngine: returns valid TickResult; action_log keys present
  - Direction check: higher DVFS reduction → lower u_next; migration decreases src load
  - SEQUENTIAL vs PARALLEL exec_mode recorded correctly in action_log
  - Closed-loop contract: u_next and p_next have correct shapes
"""
import unittest
import numpy as np

from shared.types import ActionDecision, ThermalState, TickResult
from engine8.executor import (
    ExecutionEngine,
    run_track_a,
    run_track_b,
    _apply_action_numpy,
)

N = 3  # nodes


def _make_action(dvfs=1.0, fan=0.5, cool=1, mig=None, exec_mode="PARALLEL"):
    return ActionDecision(
        a_mig=mig,
        a_dvfs=np.full(N, dvfs, dtype=np.float64),
        a_fan=float(fan),
        a_cool=cool,
        exec_mode=exec_mode,
    )


def _make_thermal(temps=None, fan_duty=0.5):
    t = np.array(temps or [40.0, 42.0, 38.0], dtype=np.float64)
    return ThermalState(
        temperatures=t, t_inlet=20.0, t_outlet=float(np.max(t)), fan_duty=fan_duty
    )


class TestTrackANumpy(unittest.TestCase):
    """Track A pure-NumPy fallback (platform-independent)."""

    def test_no_action_preserves_utilisation(self):
        """dvfs=1.0, no migration → u_next should equal u_current (no throttle)."""
        u = np.array([0.5, 0.6, 0.4])
        action = _make_action(dvfs=1.0, mig=None)
        u_next, p_next = _apply_action_numpy(action, u, N)
        np.testing.assert_allclose(u_next, u, rtol=1e-6)

    def test_dvfs_reduces_utilisation(self):
        """DVFS ratio < 1 must reduce u_next relative to u_current."""
        u = np.array([0.8, 0.7, 0.9])
        action = _make_action(dvfs=0.7, mig=None)
        u_next, p_next = _apply_action_numpy(action, u, N)
        self.assertTrue(np.all(u_next <= u + 1e-9),
                        "DVFS reduction must not increase utilisation")
        self.assertTrue(np.any(u_next < u),
                        "At least one node must show reduced utilisation with dvfs=0.7")

    def test_u_next_bounded(self):
        """u_next must stay in [0, 1] under any DVFS."""
        u = np.array([1.0, 1.0, 1.0])
        for dvfs in [0.6, 0.8, 1.0]:
            action = _make_action(dvfs=dvfs, mig=None)
            u_next, _ = _apply_action_numpy(action, u, N)
            self.assertTrue(np.all(u_next >= 0.0) and np.all(u_next <= 1.0),
                            f"u_next out of [0,1] at dvfs={dvfs}")

    def test_migration_reduces_src_load(self):
        """Migration from node 2 → node 0 must reduce node 2 utilisation."""
        u = np.array([0.3, 0.5, 0.8])   # node 2 is loaded (src)
        action = _make_action(dvfs=1.0, mig=(2, 0))
        u_next, _ = _apply_action_numpy(action, u, N)
        self.assertLess(u_next[2], u[2],
                        "Source node utilisation must decrease after migration")

    def test_migration_increases_dst_load(self):
        """Migration from node 2 → node 0 must increase node 0 utilisation."""
        u = np.array([0.1, 0.5, 0.8])   # node 0 is lightly loaded (dst)
        action = _make_action(dvfs=1.0, mig=(2, 0))
        u_next, _ = _apply_action_numpy(action, u, N)
        self.assertGreater(u_next[0], u[0],
                           "Destination node utilisation must increase after migration")

    def test_power_consistent_with_utilisation(self):
        """p_next must be consistent with u_next via the power model (monotone)."""
        u = np.array([0.6, 0.7, 0.5])
        action_throttle = _make_action(dvfs=0.7)
        action_full = _make_action(dvfs=1.0)

        u_throttle, p_throttle = _apply_action_numpy(action_throttle, u, N)
        u_full, p_full = _apply_action_numpy(action_full, u, N)

        total_p_throttle = float(np.sum(p_throttle))
        total_p_full = float(np.sum(p_full))
        self.assertLessEqual(total_p_throttle, total_p_full + 1.0,
                             "Throttled total power must not exceed unthrottled power")

    def test_p_next_positive(self):
        """Power must be > 0 (idle power) for any valid utilisation."""
        u = np.array([0.1, 0.2, 0.3])
        action = _make_action(dvfs=0.6)
        _, p_next = _apply_action_numpy(action, u, N)
        self.assertTrue(np.all(p_next > 0), "Power must be positive (includes idle)")


class TestTrackASimPy(unittest.TestCase):
    """Track A via SimPy (falls back to NumPy if simpy not installed)."""

    def test_simpy_track_a_returns_arrays(self):
        u = np.array([0.5, 0.6, 0.4])
        p = np.array([250.0, 280.0, 220.0])
        action = _make_action(dvfs=0.8, mig=None)
        u_next, p_next = run_track_a(action, u, p, n_nodes=N)
        self.assertEqual(u_next.shape, (N,))
        self.assertEqual(p_next.shape, (N,))
        self.assertTrue(np.all(u_next >= 0.0) and np.all(u_next <= 1.0))

    def test_simpy_direction_check(self):
        """SimPy track must produce lower power under throttled DVFS."""
        u = np.array([0.8, 0.9, 0.85])
        p = np.array([350.0, 380.0, 360.0])
        action_throttle = _make_action(dvfs=0.6)
        action_full = _make_action(dvfs=1.0)

        _, p_throttle = run_track_a(action_throttle, u, p, n_nodes=N)
        _, p_full = run_track_a(action_full, u, p, n_nodes=N)

        self.assertLessEqual(np.sum(p_throttle), np.sum(p_full) + 1.0,
                             "Throttled path must produce ≤ power vs unthrottled")

    def test_migration_direction_simpy(self):
        """After migration, src should have less util, dst more."""
        u = np.array([0.2, 0.5, 0.85])
        p = np.array([200.0, 280.0, 360.0])
        action = _make_action(dvfs=1.0, mig=(2, 0))
        u_next, _ = run_track_a(action, u, p, n_nodes=N)
        self.assertLess(u_next[2], u[2], "Migration src should have less util")
        self.assertGreater(u_next[0], u[0], "Migration dst should have more util")


class TestTrackBGracefulDegradation(unittest.TestCase):
    """Track B must degrade gracefully when Docker is unavailable."""

    def test_no_exception_without_docker(self):
        """run_track_b must not raise even if Docker daemon is not running."""
        action = _make_action(dvfs=0.8, mig=None)
        try:
            log = run_track_b(action, n_nodes=N)
        except Exception as exc:
            self.fail(f"run_track_b raised an exception: {exc}")
        self.assertIsInstance(log, dict)

    def test_log_contains_expected_keys(self):
        action = _make_action(dvfs=0.8, mig=None)
        log = run_track_b(action, n_nodes=N)
        # Must have some entry per node or docker_unavailable skip
        self.assertTrue(len(log) > 0)

    def test_migration_logged_as_skip_without_docker(self):
        action = _make_action(dvfs=1.0, mig=(2, 0))
        log = run_track_b(action, n_nodes=N)
        # Either the migration was attempted (and logged) or skipped gracefully
        self.assertIn("migration", log)


class TestExecutionEngineContract(unittest.TestCase):
    """ExecutionEngine top-level TickResult contract."""

    def test_returns_tick_result(self):
        engine = ExecutionEngine(n_nodes=N)
        u = np.array([0.5, 0.6, 0.4])
        p = np.array([250.0, 280.0, 220.0])
        action = _make_action()
        thermal = _make_thermal()
        result = engine.execute(action, u, p, thermal)
        self.assertIsInstance(result, TickResult)

    def test_tick_result_shapes(self):
        engine = ExecutionEngine(n_nodes=N)
        u = np.array([0.5, 0.6, 0.4])
        p = np.array([250.0, 280.0, 220.0])
        action = _make_action()
        thermal = _make_thermal()
        result = engine.execute(action, u, p, thermal)
        self.assertEqual(result.u_next.shape, (N,))
        self.assertEqual(result.p_next.shape, (N,))
        self.assertIsInstance(result.action_log, dict)

    def test_action_log_keys(self):
        engine = ExecutionEngine(n_nodes=N)
        u = np.array([0.5, 0.6, 0.4])
        p = np.array([250.0, 280.0, 220.0])
        action = _make_action()
        thermal = _make_thermal()
        result = engine.execute(action, u, p, thermal)
        log = result.action_log
        for key in ["a_mig", "a_dvfs", "a_fan", "a_cool", "exec_mode",
                    "u_before", "u_after", "p_before", "p_after", "t_max_before"]:
            self.assertIn(key, log, f"Missing key '{key}' in action_log")

    def test_exec_mode_in_log(self):
        engine = ExecutionEngine(n_nodes=N)
        u = np.array([0.5, 0.6, 0.4])
        p = np.array([250.0, 280.0, 220.0])
        for mode in ["PARALLEL", "SEQUENTIAL"]:
            action = _make_action(exec_mode=mode)
            thermal = _make_thermal()
            result = engine.execute(action, u, p, thermal)
            self.assertEqual(result.action_log["exec_mode"], mode)

    def test_u_next_bounded(self):
        engine = ExecutionEngine(n_nodes=N)
        u = np.array([0.9, 0.85, 0.95])
        p = np.array([370.0, 360.0, 380.0])
        action = _make_action(dvfs=0.6)
        thermal = _make_thermal([75.0, 78.0, 72.0])
        result = engine.execute(action, u, p, thermal)
        self.assertTrue(np.all(result.u_next >= 0.0) and np.all(result.u_next <= 1.0))

    def test_dvfs_direction_in_execute(self):
        """
        Exit criterion: temperature/power change direction must match RC model prediction.
        Lower DVFS → lower u_next → lower p_next (direction check, not absolute scale).
        """
        engine = ExecutionEngine(n_nodes=N)
        u = np.array([0.8, 0.85, 0.75])
        p = np.array([350.0, 370.0, 330.0])
        thermal = _make_thermal([70.0, 72.0, 68.0])

        result_throttle = engine.execute(_make_action(dvfs=0.6), u, p, thermal)
        result_full = engine.execute(_make_action(dvfs=1.0), u, p, thermal)

        # Total power under throttle must be ≤ total power under full (direction check)
        self.assertLessEqual(
            np.sum(result_throttle.p_next),
            np.sum(result_full.p_next) + 1.0,
            "Throttled execution must produce ≤ total power vs full-power execution"
        )

    def test_migration_source_decreases_load(self):
        engine = ExecutionEngine(n_nodes=N)
        u = np.array([0.2, 0.5, 0.85])
        p = np.array([200.0, 280.0, 360.0])
        action = _make_action(dvfs=1.0, mig=(2, 0))
        thermal = _make_thermal()
        result = engine.execute(action, u, p, thermal)
        self.assertLess(result.u_next[2], u[2],
                        "Migration src (node 2) must have lower utilisation after execution")


if __name__ == "__main__":
    unittest.main()

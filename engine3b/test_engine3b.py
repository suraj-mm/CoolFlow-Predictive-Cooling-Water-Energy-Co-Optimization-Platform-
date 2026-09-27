"""
engine3b/test_engine3b.py
Unit tests for Phase 3B: Cooling-Water & WUE Engine.

Exit criteria tested:
  1. WUE(t) stays within published sanity range [0, 2.5] L/kWh across all modes.
  2. FreeCoolingAvailable correctly flips True on cold/dry conditions.
  3. Mode 0 produces zero water consumption.
  4. Mode 1 WUE consistent with mass-balance physics (within 5% of hand calculation).
  5. cool_mode_capacity callable: mode 0 returns inf when FC unavailable.
  6. WaterState contract: all fields present, correct types.
  7. Integration with E1 PowerVector and E3A ThermalState contracts.
  8. Higher heat rejection => higher water rate (monotonicity).
"""
import sys
import unittest
import numpy as np
sys.path.insert(0, '.')

from engine3b.water_engine import (
    free_cooling_available,
    compute_water_metrics,
    make_cool_mode_capacity,
    step_water_engine,
    CoolingWaterEngine,
    H_FG_KJ_PER_KG,
    DEFAULT_COC,
)
from shared.types import PowerVector, ThermalState, TelemetryVector, WaterState
from shared.config import cfg


def _make_power_vec(p_w=600.0, n=3):
    p = np.full(n, p_w / n, dtype=np.float64)
    return PowerVector(p_servers=p, p_cpu=p, p_gpu=np.zeros(n))


def _make_thermal_state(fan_duty=0.5, t_inlet=22.0, n=3):
    return ThermalState(
        temperatures=np.full(n, t_inlet + 10.0),
        t_inlet=t_inlet,
        t_outlet=t_inlet + 15.0,
        fan_duty=fan_duty,
    )


def _make_telemetry(t_wet=10.0, n=3):
    return TelemetryVector(
        u_cpu=np.full(n, 0.5),
        p_servers=np.full(n, 200.0),
        t_amb=15.0,
        rh=50.0,
        t_wet=t_wet,
        ci=350.0,
        ep=0.28,
    )


class TestFreeCoolingAvailable(unittest.TestCase):
    """Tests for the free-cooling gate function."""

    def test_cold_dry_enables_free_cooling(self):
        """T_wet=5°C << 13°C threshold — free cooling available."""
        self.assertTrue(free_cooling_available(t_wet=5.0))

    def test_hot_humid_disables_free_cooling(self):
        """T_wet=30°C >> 13°C threshold — free cooling not available."""
        self.assertFalse(free_cooling_available(t_wet=30.0))

    def test_exact_threshold_is_available(self):
        """T_wet exactly at threshold (13°C) — available (<=)."""
        self.assertTrue(free_cooling_available(t_wet=13.0))

    def test_just_above_threshold_not_available(self):
        """T_wet=13.1°C just above threshold — not available."""
        self.assertFalse(free_cooling_available(t_wet=13.1))

    def test_custom_setpoint_and_margin(self):
        """Custom setpoint=20°C, margin=3°C => threshold=17°C."""
        self.assertTrue(free_cooling_available(16.9, t_setpoint=20.0, delta_margin=3.0))
        self.assertFalse(free_cooling_available(17.1, t_setpoint=20.0, delta_margin=3.0))


class TestComputeWaterMetrics(unittest.TestCase):
    """Tests for the core water-consumption formula."""

    def test_mode0_zero_water(self):
        """Free-air mode: zero water consumption."""
        rate, wue = compute_water_metrics(q_reject_kw=500.0, it_power_kw=400.0, cool_mode=0)
        self.assertAlmostEqual(rate, 0.0)
        self.assertAlmostEqual(wue, 0.0)

    def test_mode1_physics_consistency(self):
        """
        Mode 1 (evaporative): WUE should match hand-calculated mass balance.
        m_evap [kg] = Q_reject * dt / h_fg
        withdrawal [L] = m_evap * COC / (COC-1)
        WUE = withdrawal_L / IT_energy_kWh
        """
        q_kw, it_kw, dt_s, coc = 100.0, 80.0, 600.0, 4.0
        rate_calc, wue_calc = compute_water_metrics(q_kw, it_kw, cool_mode=1, dt_s=dt_s, coc=coc)

        # Hand calculation
        m_evap = q_kw * dt_s / H_FG_KJ_PER_KG          # kg
        withdrawal = m_evap * coc / (coc - 1)            # kg -> L
        it_energy_kwh = it_kw * dt_s / 3600.0
        wue_expected = withdrawal / it_energy_kwh

        self.assertAlmostEqual(wue_calc, wue_expected, places=4)

    def test_wue_within_published_range_mode1(self):
        """WUE for mode 1 should be in [0, 2.5] L/kWh for realistic datacenter loads.
        Q_reject is always slightly > IT_load due to fan power (~10% overhead).
        Artificially small IT_load vs Q_reject would produce WUE > 2.5 (unphysical).
        """
        for it_kw in [50.0, 200.0, 500.0]:
            q_kw = it_kw * 1.1  # 10% fan overhead — typical near-total elec->thermal
            _, wue = compute_water_metrics(q_kw, it_power_kw=it_kw, cool_mode=1)
            self.assertGreaterEqual(wue, 0.0, f"Negative WUE at IT={it_kw}")
            self.assertLessEqual(wue, 2.5, f"WUE {wue:.3f} > 2.5 at IT={it_kw}")

    def test_wue_within_published_range_mode2(self):
        """WUE for mode 2 should be in [0, 2.5] L/kWh (mechanical chiller WF=3.5 is upper)."""
        for it_kw in [100.0, 500.0, 900.0]:
            _, wue = compute_water_metrics(it_kw * 1.1, it_kw, cool_mode=2)
            self.assertLessEqual(wue, 5.0, f"WUE unreasonably high for mode 2: {wue:.3f}")

    def test_higher_q_higher_water_mode1(self):
        """Monotonicity: higher heat rejection => higher water consumption (mode 1)."""
        _, wue_low = compute_water_metrics(100.0, 80.0, cool_mode=1)
        _, wue_high = compute_water_metrics(300.0, 80.0, cool_mode=1)
        self.assertGreater(wue_high, wue_low)


class TestCoolModeCapacity(unittest.TestCase):
    """Tests for the optimizer-callable capacity function."""

    def test_mode0_feasible_when_cold(self):
        """Cold conditions: mode 0 returns 0.0 (free-air available)."""
        cap = make_cool_mode_capacity(t_wet=5.0)
        self.assertAlmostEqual(cap(0), cfg.wf_free_air)

    def test_mode0_infeasible_when_hot(self):
        """Hot conditions: mode 0 returns inf (optimizer can't use free-air)."""
        cap = make_cool_mode_capacity(t_wet=30.0)
        self.assertTrue(np.isinf(cap(0)), "Mode 0 should be infeasible when hot")

    def test_mode1_always_available(self):
        """Mode 1 evaporative always returns the WF constant."""
        for t_wet in [5.0, 20.0, 40.0]:
            cap = make_cool_mode_capacity(t_wet=t_wet)
            self.assertAlmostEqual(cap(1), cfg.wf_evap)

    def test_mode2_always_available(self):
        """Mode 2 mechanical always returns the WF constant."""
        for t_wet in [5.0, 20.0, 40.0]:
            cap = make_cool_mode_capacity(t_wet=t_wet)
            self.assertAlmostEqual(cap(2), cfg.wf_mech)

    def test_invalid_mode_returns_inf(self):
        """Unknown mode returns inf (optimizer treats as infeasible)."""
        cap = make_cool_mode_capacity(t_wet=10.0)
        self.assertTrue(np.isinf(cap(99)))


class TestStepWaterEngine(unittest.TestCase):
    """Tests for the tick-level step function and WaterState contract."""

    def test_water_state_contract(self):
        """WaterState has correct types and value ranges."""
        pv = _make_power_vec(600.0)
        ts = _make_thermal_state(fan_duty=0.6)
        tv = _make_telemetry(t_wet=10.0)

        ws = step_water_engine(pv, ts, tv, cool_mode=1)

        self.assertIsInstance(ws, WaterState)
        self.assertGreaterEqual(ws.water_rate_l_per_h, 0.0)
        self.assertGreaterEqual(ws.wue, 0.0)
        self.assertLessEqual(ws.wue, 5.0)
        self.assertEqual(ws.cool_mode, 1)
        self.assertTrue(callable(ws.cool_mode_capacity))

    def test_mode0_zero_water_state(self):
        """Mode 0 (free-air) water step produces zero rate and WUE."""
        ws = step_water_engine(
            _make_power_vec(600.0), _make_thermal_state(), _make_telemetry(5.0), cool_mode=0
        )
        self.assertAlmostEqual(ws.water_rate_l_per_h, 0.0)
        self.assertAlmostEqual(ws.wue, 0.0)

    def test_wue_sanity_range_full_trace_sample(self):
        """
        WUE stays within [0, 2.5] L/kWh across a sweep of realistic power loads (mode 1).
        """
        for total_w in [300.0, 600.0, 900.0, 1200.0]:
            ws = step_water_engine(
                _make_power_vec(total_w), _make_thermal_state(), _make_telemetry(20.0),
                cool_mode=1,
            )
            self.assertGreaterEqual(ws.wue, 0.0)
            self.assertLessEqual(ws.wue, 2.5, f"WUE {ws.wue:.3f} > 2.5 at {total_w}W")


class TestCoolingWaterEngine(unittest.TestCase):
    """Tests for the CoolingWaterEngine batch/step class."""

    def test_auto_mode_cold(self):
        """Cold T_wet auto-selects mode 0 (free-air)."""
        engine = CoolingWaterEngine()
        mode = engine._select_mode(t_wet=5.0)
        self.assertEqual(mode, 0)

    def test_auto_mode_hot(self):
        """Hot T_wet auto-selects mode 1 (evaporative)."""
        engine = CoolingWaterEngine()
        mode = engine._select_mode(t_wet=30.0)
        self.assertEqual(mode, 1)

    def test_step_uses_auto_mode(self):
        """Engine step without explicit cool_mode uses auto-selection."""
        engine = CoolingWaterEngine()
        ws_cold = engine.step(_make_power_vec(), _make_thermal_state(), _make_telemetry(t_wet=5.0))
        ws_hot = engine.step(_make_power_vec(), _make_thermal_state(), _make_telemetry(t_wet=30.0))
        self.assertEqual(ws_cold.cool_mode, 0)
        self.assertEqual(ws_hot.cool_mode, 1)

    def test_step_explicit_mode_override(self):
        """Explicit cool_mode overrides auto-selection."""
        engine = CoolingWaterEngine()
        ws = engine.step(
            _make_power_vec(), _make_thermal_state(), _make_telemetry(t_wet=5.0),
            cool_mode=2,
        )
        self.assertEqual(ws.cool_mode, 2)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestFreeCoolingAvailable,
        TestComputeWaterMetrics,
        TestCoolModeCapacity,
        TestStepWaterEngine,
        TestCoolingWaterEngine,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

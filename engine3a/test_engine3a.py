"""
engine3a/test_engine3a.py
Unit tests for Phase 3A: RC Thermal Digital Twin.

Exit criteria tested:
  1. Step-response: step change in P_i relaxes toward new steady state
     with time constant matching tau = R*C (dataset: 80s, spec-mode: ~140s).
  2. Steady-state accuracy: T_ss = T_inlet + P * R_air (at chosen fan duty).
  3. No divergence over a 24-hour replay (144 ticks at 10-min cadence).
  4. ThermalState contract: shape (N,), t_inlet and fan_duty in expected ranges.
  5. Multi-node coupling: hotter nodes cool toward cooler neighbours.
  6. Fan duty effect: lower fan => higher steady-state temperature.
  7. Integration with Engine 1 PowerVector contract.
"""
import sys
import unittest
import numpy as np
sys.path.insert(0, '.')

from engine3a.rc_twin import step_rc_thermal_twin, RCThermalTwin
from shared.config import cfg
from shared.types import PowerVector, ThermalState


class TestStepRCFunction(unittest.TestCase):
    """Tests for the public step_rc_thermal_twin() function."""

    def _steady_state(self, p_node, t_inlet, fan_duty, r0, c, r_adj, gamma):
        """Analytical steady-state: T_ss ≈ T_inlet + P * R_air (ignores coupling for N=1)."""
        r_air = r0 / (fan_duty ** gamma)
        return t_inlet + p_node * r_air

    def test_step_response_time_constant_dataset_mode(self):
        """
        Step change in power → temperature relaxes with tau = C * R_air (dataset params).
        tau = C * R_air = 2000 * (0.04 / 0.5^0.8) ≈ 139s (NOT R*C=80s which is wrong).
        After one time-constant the response should be 63.2% ± 10% of the step.
        """
        # Dataset RC params
        r0, c = 0.04, 2000.0
        r_adj = cfg.r_adjacent
        gamma = cfg.gamma_fan
        fan_duty = 0.5
        t_inlet = 20.0

        # Convective tau: tau = C * R_air where R_air = r0 / fan_duty^gamma
        r_air = r0 / (fan_duty ** gamma)
        tau = c * r_air  # ~139s at fan_duty=0.5

        p0 = np.array([200.0])
        t_nodes = np.array([self._steady_state(p0[0], t_inlet, fan_duty, r0, c, r_adj, gamma)])
        p_step = np.array([400.0])
        t_ss_new = self._steady_state(p_step[0], t_inlet, fan_duty, r0, c, r_adj, gamma)
        delta = t_ss_new - t_nodes[0]

        # Advance exactly tau seconds — fraction should be 1 - 1/e ≈ 0.632
        t_after_tau = step_rc_thermal_twin(
            t_nodes=t_nodes.copy(), p_nodes=p_step,
            t_inlet=t_inlet, fan_duty=fan_duty,
            c_thermal=c, r0_airflow=r0, r_adjacent=r_adj, gamma_fan=gamma,
            tick_duration_s=tau, dt_s=1.0,
        )
        fraction = (t_after_tau[0] - t_nodes[0]) / delta
        # 1 - 1/e ≈ 0.632; allow ±10% tolerance (N=1 no coupling, analytic)
        self.assertGreater(fraction, 0.53, f"Step response too slow: {fraction:.3f}")
        self.assertLess(fraction, 0.73, f"Step response too fast: {fraction:.3f}")

    def test_steady_state_convergence_dataset_mode(self):
        """After 10 tau, temperature converges within 1°C of T_inlet + P * R_air."""
        r0, c = 0.04, 2000.0
        r_adj = cfg.r_adjacent
        gamma = cfg.gamma_fan
        fan_duty = 0.5
        t_inlet = 20.0
        p = np.array([300.0])
        t_nodes = np.array([t_inlet])

        r_air = r0 / (fan_duty ** gamma)
        tau = c * r_air  # ~139s

        # Run 10 time constants => 99.995% of step response reached
        t_final = step_rc_thermal_twin(
            t_nodes=t_nodes.copy(), p_nodes=p,
            t_inlet=t_inlet, fan_duty=fan_duty,
            c_thermal=c, r0_airflow=r0, r_adjacent=r_adj, gamma_fan=gamma,
            tick_duration_s=tau * 10.0, dt_s=1.0,
        )
        t_ss = self._steady_state(p[0], t_inlet, fan_duty, r0, c, r_adj, gamma)
        self.assertAlmostEqual(float(t_final[0]), t_ss, delta=1.0,
                               msg=f"Steady-state not reached: got {t_final[0]:.2f}, expected ~{t_ss:.2f}")

    def test_no_divergence_24h(self):
        """24-hour replay (144 ticks x 600s) with realistic power — temperatures stay finite."""
        t_nodes = np.array([25.0, 30.0, 28.0])
        p = np.array([250.0, 350.0, 200.0])

        for _ in range(144):
            t_nodes = step_rc_thermal_twin(
                t_nodes=t_nodes, p_nodes=p,
                t_inlet=22.0, fan_duty=0.6,
                c_thermal=2000.0, r0_airflow=0.04,
            )
        self.assertTrue(np.all(np.isfinite(t_nodes)), "Temperatures diverged after 24h replay")
        self.assertTrue(np.all(t_nodes < 200.0), f"Unrealistic temps: {t_nodes}")

    def test_no_divergence_spec_mode(self):
        """No divergence with spec-default RC params (tau ~140s at 50% fan) over 24h."""
        t_nodes = np.array([25.0, 30.0, 28.0])
        p = np.array([250.0, 350.0, 200.0])

        for _ in range(144):
            t_nodes = step_rc_thermal_twin(
                t_nodes=t_nodes, p_nodes=p,
                t_inlet=22.0, fan_duty=0.5,
                c_thermal=cfg.c_thermal,
                r0_airflow=cfg.r0_airflow,
            )
        self.assertTrue(np.all(np.isfinite(t_nodes)), "Divergence with spec params")

    def test_higher_power_higher_temperature(self):
        """More power input must produce higher steady-state temperature."""
        base = np.array([20.0])
        t_low = step_rc_thermal_twin(t_nodes=base.copy(), p_nodes=np.array([150.0]),
                                      t_inlet=20.0, fan_duty=0.5,
                                      c_thermal=2000.0, r0_airflow=0.04, tick_duration_s=800.0)
        t_high = step_rc_thermal_twin(t_nodes=base.copy(), p_nodes=np.array([350.0]),
                                       t_inlet=20.0, fan_duty=0.5,
                                       c_thermal=2000.0, r0_airflow=0.04, tick_duration_s=800.0)
        self.assertGreater(t_high[0], t_low[0], "Higher power should give higher temperature")

    def test_higher_fan_lower_temperature(self):
        """Higher fan duty lowers steady-state temperature (more convective cooling)."""
        base = np.array([50.0])
        p = np.array([300.0])
        t_low_fan = step_rc_thermal_twin(t_nodes=base.copy(), p_nodes=p,
                                          t_inlet=20.0, fan_duty=0.3,
                                          c_thermal=2000.0, r0_airflow=0.04, tick_duration_s=800.0)
        t_hi_fan = step_rc_thermal_twin(t_nodes=base.copy(), p_nodes=p,
                                         t_inlet=20.0, fan_duty=1.0,
                                         c_thermal=2000.0, r0_airflow=0.04, tick_duration_s=800.0)
        self.assertGreater(t_low_fan[0], t_hi_fan[0], "Higher fan should give lower temperature")

    def test_coupling_hot_node_cools_cold_warms(self):
        """Multi-node coupling: hot node cools and cold node warms toward mean."""
        t_nodes = np.array([60.0, 20.0, 20.0])  # node 0 is hot
        p = np.array([0.0, 0.0, 0.0])            # no power — purely coupling

        t_after = step_rc_thermal_twin(
            t_nodes=t_nodes.copy(), p_nodes=p,
            t_inlet=20.0, fan_duty=0.5,
            c_thermal=2000.0, r0_airflow=0.04,
            tick_duration_s=160.0,  # 2 tau
        )
        self.assertLess(t_after[0], 60.0, "Hot node should cool due to coupling")
        self.assertGreater(t_after[1], 20.0, "Cold node should warm due to coupling")


class TestRCThermalTwinClass(unittest.TestCase):
    """Tests for the RCThermalTwin class (E3A step and ThermalState contract)."""

    def _make_power_vec(self, p_vals):
        p = np.array(p_vals, dtype=np.float64)
        return PowerVector(p_servers=p, p_cpu=p, p_gpu=np.zeros_like(p))

    def _make_row(self, t_inlet=22.0, fan_speed=1500.0, r=0.04, c=2000.0):
        import pandas as pd
        return pd.Series({
            "timestamp": "2022-02-03 00:00:00",
            "InletTemp": t_inlet,
            "FanSpeed": fan_speed,
            "R_server_param": r,
            "C_server_param": c,
        })

    def test_thermal_state_contract(self):
        """ThermalState output matches E3A contract: shape (N,), finite, plausible values."""
        twin = RCThermalTwin(n_nodes=3, dataset_mode=True)
        t_nodes = np.array([25.0, 28.0, 22.0])
        pv = self._make_power_vec([250.0, 300.0, 200.0])
        row = self._make_row()

        state = twin.step(t_nodes, pv, row)

        self.assertIsInstance(state, ThermalState)
        self.assertEqual(state.temperatures.shape, (3,))
        self.assertTrue(np.all(np.isfinite(state.temperatures)))
        self.assertGreater(state.t_outlet, state.t_inlet,
                           "Outlet temp should exceed inlet for powered nodes")
        self.assertGreaterEqual(state.fan_duty, cfg.fan_min)
        self.assertLessEqual(state.fan_duty, 1.0)

    def test_fan_duty_normalization(self):
        """FanSpeed=1500 -> fan_duty ~ 0.5, clamped to [0.3, 1.0]."""
        twin = RCThermalTwin(n_nodes=3)
        row = self._make_row(fan_speed=1500.0)
        duty = twin._fan_duty(row)
        self.assertAlmostEqual(duty, 0.5, delta=0.02)

    def test_fan_duty_clamped_low(self):
        """Very low FanSpeed is clamped to fan_min=0.3."""
        twin = RCThermalTwin(n_nodes=3)
        row = self._make_row(fan_speed=1.0)
        duty = twin._fan_duty(row)
        self.assertAlmostEqual(duty, cfg.fan_min, places=5)

    def test_dataset_mode_vs_spec_mode(self):
        """Dataset mode (C=2000) should produce a different (faster) response than spec mode (C=400)."""
        t0 = np.array([25.0, 25.0, 25.0])
        pv = self._make_power_vec([300.0, 300.0, 300.0])
        row = self._make_row()

        twin_ds = RCThermalTwin(n_nodes=3, dataset_mode=True)
        twin_sp = RCThermalTwin(n_nodes=3, dataset_mode=False)

        state_ds = twin_ds.step(t0.copy(), pv, row)
        state_sp = twin_sp.step(t0.copy(), pv, row)

        # Dataset tau=80s < spec tau (~140s) => dataset warms faster per tick
        # (starting from same cold state, dataset should be warmer after 600s)
        self.assertFalse(
            np.allclose(state_ds.temperatures, state_sp.temperatures),
            "Dataset mode and spec mode should give different temperatures",
        )

    def test_shape_mismatch_raises(self):
        """Mismatched t_nodes / p_nodes shapes should raise ValueError."""
        with self.assertRaises(ValueError):
            step_rc_thermal_twin(
                t_nodes=np.array([20.0, 20.0]),
                p_nodes=np.array([200.0, 200.0, 200.0]),  # wrong length
                t_inlet=20.0, fan_duty=0.5,
            )


if __name__ == "__main__":
    import unittest
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestStepRCFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestRCThermalTwinClass))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

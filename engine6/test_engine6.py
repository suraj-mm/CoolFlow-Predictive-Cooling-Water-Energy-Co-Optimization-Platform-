"""
engine6/test_engine6.py
Unit tests for the Hysteresis Deadband Hotspot Detection Engine.

Coverage:
  - Hysteresis entry: alert fires when T_high >= T_crit
  - Hysteresis exit:  alert clears ONLY when T_actual < T_crit - deadband
  - No chattering:    alert stays active during noisy oscillations
  - No false positive: alert never fires when temps are well below threshold
  - trigger_optimizer: fires on alert raise, on confidence drop, on cadence
  - Pure-function check_hotspot API
  - AlertState contract (field types and value ranges)
"""
import unittest
import numpy as np

from shared.config import cfg
from shared.types import AlertState, PredictionBundle, ThermalState
from engine6.hotspot import (
    HotspotDetector,
    HotspotDetectorState,
    check_hotspot,
    _should_trigger,
    _T_CRIT,
    _DEADBAND,
    _CLEAR_THRESHOLD,
    _CONF_GATE,
)


def _make_thermal(temps, fan_duty=0.5, t_inlet=20.0):
    t = np.array(temps, dtype=np.float64)
    return ThermalState(
        temperatures=t,
        t_inlet=t_inlet,
        t_outlet=float(np.max(t)),
        fan_duty=fan_duty,
    )


def _make_pred(t_high, t_mid=None, confidence=95.0):
    t_h = np.array(t_high, dtype=np.float64)
    t_m = t_h - 5.0 if t_mid is None else np.array(t_mid, dtype=np.float64)
    t_l = t_m - 5.0
    cs = np.full(len(t_h), confidence, dtype=np.float64)
    return PredictionBundle(t_mid=t_m, t_low=t_l, t_high=t_h, confidence_score=cs)


class TestHotspotEntryEdge(unittest.TestCase):
    """Alert must fire when ANY T_high >= T_crit."""

    def test_alert_fires_on_t_high_breach(self):
        det = HotspotDetector()
        # T_high[0] == T_crit → should trigger
        pred = _make_pred([_T_CRIT, 50.0, 55.0])
        thermal = _make_thermal([70.0, 50.0, 55.0])
        alert = det.step(thermal, pred)
        self.assertTrue(alert.hotspot_flag, "Alert should fire when T_high >= T_crit")

    def test_alert_does_not_fire_below_threshold(self):
        det = HotspotDetector()
        pred = _make_pred([_T_CRIT - 0.1, 50.0, 55.0])
        thermal = _make_thermal([65.0, 50.0, 55.0])
        alert = det.step(thermal, pred)
        self.assertFalse(alert.hotspot_flag, "Alert should not fire when T_high < T_crit")

    def test_alert_fires_on_any_node_breach(self):
        det = HotspotDetector()
        # Only node 2 breaches
        pred = _make_pred([60.0, 62.0, _T_CRIT + 1.0])
        thermal = _make_thermal([60.0, 62.0, 78.0])
        alert = det.step(thermal, pred)
        self.assertTrue(alert.hotspot_flag)


class TestHotspotHysteresisExit(unittest.TestCase):
    """Alert must NOT clear unless T_actual drops below T_crit - deadband."""

    def _raise_alert(self, det: HotspotDetector):
        """Helper: raise the alert by feeding a breaching tick."""
        pred = _make_pred([_T_CRIT + 2.0, 50.0, 50.0])
        thermal = _make_thermal([78.0, 50.0, 50.0])
        det.step(thermal, pred)

    def test_alert_does_not_clear_at_T_crit(self):
        det = HotspotDetector()
        self._raise_alert(det)
        # Actual temp == T_crit (above clear threshold) — should stay active
        pred = _make_pred([_T_CRIT - 1.0, 50.0, 50.0])   # predicted no longer breaching
        thermal = _make_thermal([_T_CRIT, 50.0, 50.0])    # actual == 80°C, above 76°C clear
        alert = det.step(thermal, pred)
        self.assertTrue(alert.hotspot_flag,
                        "Alert must stay active at T_actual == T_crit (above clear threshold)")

    def test_alert_does_not_clear_at_boundary_minus_epsilon(self):
        det = HotspotDetector()
        self._raise_alert(det)
        # Actual temp just above clear threshold (76.01°C > 76°C)
        pred = _make_pred([60.0, 50.0, 50.0])
        thermal = _make_thermal([_CLEAR_THRESHOLD + 0.01, 50.0, 50.0])
        alert = det.step(thermal, pred)
        self.assertTrue(alert.hotspot_flag,
                        "Alert must stay active at T > clear_threshold")

    def test_alert_clears_below_deadband(self):
        det = HotspotDetector()
        self._raise_alert(det)
        # All temps below clear threshold AND predicted no breach
        pred = _make_pred([60.0, 50.0, 50.0])
        thermal = _make_thermal([_CLEAR_THRESHOLD - 0.1, 50.0, 50.0])
        alert = det.step(thermal, pred)
        self.assertFalse(alert.hotspot_flag,
                         "Alert must clear when T_actual < clear_threshold AND T_high < T_crit")

    def test_clear_threshold_value(self):
        """Verify clear threshold is exactly T_crit - deadband."""
        self.assertAlmostEqual(_CLEAR_THRESHOLD, _T_CRIT - _DEADBAND, places=6)
        self.assertAlmostEqual(_CLEAR_THRESHOLD, 80.0 - 4.0, places=6)


class TestNoChattering(unittest.TestCase):
    """Alert must remain stable under oscillating noisy input."""

    def test_no_chattering_oscillation(self):
        """
        Simulate noisy temperature that oscillates around T_crit without truly cooling.
        Alert should raise and STAY raised (no chatter), because actual temps never
        drop below the clear threshold.
        """
        det = HotspotDetector()
        flags = []
        for tick in range(20):
            # Noisy T_high: alternates above/below T_crit
            t_high_0 = _T_CRIT + (1.0 if tick % 2 == 0 else -0.5)
            # Actual temps: stay above clear threshold (76.5°C)
            t_actual_0 = _CLEAR_THRESHOLD + 0.5

            pred = _make_pred([t_high_0, 55.0, 55.0])
            thermal = _make_thermal([t_actual_0, 55.0, 55.0])
            alert = det.step(thermal, pred)
            flags.append(alert.hotspot_flag)

        # After the first alert fires, it must never flip False while temps stay hot
        first_true = next((i for i, f in enumerate(flags) if f), None)
        if first_true is not None:
            self.assertTrue(
                all(flags[first_true:]),
                "Alert chattering detected: flag went False while temps stayed above clear_threshold"
            )


class TestOptimizerTrigger(unittest.TestCase):
    """trigger_optimizer must follow the cadence rules."""

    def test_trigger_on_alert_raise(self):
        det = HotspotDetector()
        pred = _make_pred([_T_CRIT + 2.0, 50.0, 50.0])
        thermal = _make_thermal([78.0, 50.0, 50.0])
        alert = det.step(thermal, pred)
        self.assertTrue(alert.hotspot_flag)
        self.assertTrue(alert.trigger_optimizer, "trigger_optimizer must be True when alert first fires")

    def test_trigger_on_low_confidence(self):
        det = HotspotDetector()
        pred = _make_pred([50.0, 50.0, 50.0], confidence=_CONF_GATE - 1.0)
        thermal = _make_thermal([50.0, 50.0, 50.0])
        alert = det.step(thermal, pred)
        self.assertTrue(alert.trigger_optimizer,
                        "trigger_optimizer must be True when min confidence < gate threshold")

    def test_no_trigger_when_idle_and_confident(self):
        det = HotspotDetector()
        pred = _make_pred([50.0, 50.0, 50.0], confidence=95.0)
        thermal = _make_thermal([50.0, 50.0, 50.0])
        alert = det.step(thermal, pred)
        self.assertFalse(alert.hotspot_flag)
        self.assertFalse(alert.trigger_optimizer, "No trigger when idle and high confidence")

    def test_should_trigger_logic(self):
        # alert just raised
        self.assertTrue(_should_trigger(True, True, 0, 90.0))
        # low confidence
        self.assertTrue(_should_trigger(False, False, 10, _CONF_GATE - 1.0))
        # idle + confident
        self.assertFalse(_should_trigger(False, False, 10, 95.0))


class TestCheckHotspotStateless(unittest.TestCase):
    """Pure-function check_hotspot API."""

    def test_entry_when_not_active(self):
        t_actual = np.array([70.0, 65.0, 60.0])
        t_high = np.array([_T_CRIT + 1.0, 65.0, 60.0])
        alert_now, nodes = check_hotspot(t_actual, t_high, alert_was_active=False)
        self.assertTrue(alert_now)

    def test_no_entry_below_threshold(self):
        t_actual = np.array([60.0, 55.0, 50.0])
        t_high = np.array([60.0, 55.0, 50.0])
        alert_now, _ = check_hotspot(t_actual, t_high, alert_was_active=False)
        self.assertFalse(alert_now)

    def test_hysteresis_stays_active_when_hot(self):
        """Alert stays active if actual temps above clear threshold, even if T_high drops."""
        t_actual = np.array([_CLEAR_THRESHOLD + 1.0, 60.0, 60.0])
        t_high = np.array([60.0, 60.0, 60.0])   # no new breach
        alert_now, _ = check_hotspot(t_actual, t_high, alert_was_active=True)
        self.assertTrue(alert_now)

    def test_hysteresis_clears_below_threshold(self):
        t_actual = np.array([_CLEAR_THRESHOLD - 1.0, 60.0, 60.0])
        t_high = np.array([60.0, 60.0, 60.0])
        alert_now, _ = check_hotspot(t_actual, t_high, alert_was_active=True)
        self.assertFalse(alert_now)


class TestAlertStateContract(unittest.TestCase):
    """Verify AlertState type contract."""

    def test_alert_state_types(self):
        det = HotspotDetector()
        pred = _make_pred([50.0, 50.0, 50.0])
        thermal = _make_thermal([50.0, 50.0, 50.0])
        alert = det.step(thermal, pred)

        self.assertIsInstance(alert, AlertState)
        self.assertIsInstance(alert.hotspot_flag, bool)
        self.assertIsInstance(alert.alert_nodes, list)
        self.assertIsInstance(alert.trigger_optimizer, bool)
        self.assertIsInstance(alert.max_temp, float)
        self.assertGreaterEqual(alert.max_temp, 0.0)

    def test_alert_nodes_correct(self):
        det = HotspotDetector()
        pred = _make_pred([_T_CRIT + 5.0, 50.0, 50.0])
        thermal = _make_thermal([_T_CRIT + 2.0, 50.0, 50.0])  # node 0 above T_crit
        alert = det.step(thermal, pred)
        self.assertIn(0, alert.alert_nodes)
        self.assertNotIn(1, alert.alert_nodes)

    def test_reset_clears_state(self):
        det = HotspotDetector()
        pred_hot = _make_pred([_T_CRIT + 2.0, 50.0, 50.0])
        thermal_hot = _make_thermal([78.0, 50.0, 50.0])
        det.step(thermal_hot, pred_hot)
        self.assertTrue(det._state.alert_active)

        det.reset()
        self.assertFalse(det._state.alert_active)
        self.assertEqual(det._state.tick_count, 0)

    def test_max_temp_is_max_of_actuals(self):
        det = HotspotDetector()
        temps = [72.0, 68.0, 65.0]
        pred = _make_pred([60.0, 60.0, 60.0])
        thermal = _make_thermal(temps)
        alert = det.step(thermal, pred)
        self.assertAlmostEqual(alert.max_temp, max(temps))


class TestMultiTickReplay(unittest.TestCase):
    """Multi-tick trace: verify hysteresis holds over extended replay."""

    def test_sustained_alert_then_clear(self):
        det = HotspotDetector()
        results = []

        # Phase 1: 5 hot ticks (T_high breaches, T_actual > clear threshold)
        for _ in range(5):
            pred = _make_pred([_T_CRIT + 3.0, 60.0, 60.0])
            thermal = _make_thermal([78.0, 60.0, 60.0])
            results.append(det.step(thermal, pred).hotspot_flag)

        # Phase 2: 5 cooling ticks (T_high drops, T_actual between deadband and T_crit)
        for _ in range(5):
            pred = _make_pred([70.0, 60.0, 60.0])
            thermal = _make_thermal([77.0, 60.0, 60.0])   # 77 > 76 = clear threshold
            results.append(det.step(thermal, pred).hotspot_flag)

        # Phase 3: 3 fully-cooled ticks (T_actual < clear_threshold)
        for _ in range(3):
            pred = _make_pred([60.0, 60.0, 60.0])
            thermal = _make_thermal([_CLEAR_THRESHOLD - 2.0, 60.0, 60.0])
            results.append(det.step(thermal, pred).hotspot_flag)

        # All Phase 1 + Phase 2 ticks must have flag=True (hysteresis holds)
        self.assertTrue(all(results[:10]),
                        "Alert should stay active during phases 1 and 2")

        # Phase 3: alert should have cleared
        self.assertFalse(results[-1],
                         "Alert should clear when temps fall below deadband threshold")


if __name__ == "__main__":
    unittest.main()

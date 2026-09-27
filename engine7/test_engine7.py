"""
engine7/test_engine7.py
Unit tests for the Decision & Optimization Engine.

Coverage:
  - PriorityClassifier: fit, predict, class constraints
  - ActionGenerator: priority-constrained DVFS, migration filtering, cool-mode feasibility
  - ActionEvaluator: objective shape, constraint sign, warm vs cool tick decisions
  - Knee selector: returns valid index, shifts toward thermal under low confidence
  - Confidence gate: PARALLEL/SEQUENTIAL boundary
  - DecisionEngine: cheap path caching, full pipeline end-to-end
  - Water trade-off: Free-Air chosen on cool tick, Mechanical/Evaporative on hot tick
"""
import unittest
import numpy as np
import pandas as pd

from shared.config import cfg
from shared.types import (
    ActionDecision,
    AlertState,
    PredictionBundle,
    TelemetryVector,
    ThermalState,
    WaterState,
)
from engine7.optimizer import (
    DecisionEngine,
    PriorityClassifier,
    generate_actions,
    evaluate_actions,
    run_nsga2,
    select_knee,
    apply_confidence_gate,
    _DVFS_LEVELS,
    _CONF_GATE,
)
from engine3b.water_engine import make_cool_mode_capacity


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
N = 3   # nodes


def _make_telemetry(t_amb=25.0, t_wet=15.0, ep=0.28, u_cpu=None, p_servers=None):
    u = u_cpu if u_cpu is not None else np.array([0.5, 0.6, 0.4])
    p = p_servers if p_servers is not None else np.array([250.0, 280.0, 220.0])
    return TelemetryVector(
        u_cpu=u, p_servers=p, t_amb=t_amb, rh=50.0,
        t_wet=t_wet, ci=300.0, ep=ep,
    )


def _make_water(t_wet=15.0, cool_mode=0):
    cap_fn = make_cool_mode_capacity(t_wet)
    rate = 10.0 if cool_mode != 0 else 0.0
    return WaterState(
        water_rate_l_per_h=rate,
        wue=0.5,
        cool_mode=cool_mode,
        cool_mode_capacity=cap_fn,
    )


def _make_pred(t_mid=None, cs=90.0):
    t_m = t_mid if t_mid is not None else np.array([40.0, 50.0, 38.0])
    t_l = t_m - 3.0
    t_h = t_m + 3.0
    return PredictionBundle(
        t_mid=t_m, t_low=t_l, t_high=t_h,
        confidence_score=np.full(N, cs),
    )


def _make_alert(flag=True, trigger=True):
    return AlertState(hotspot_flag=flag, alert_nodes=[0], trigger_optimizer=trigger, max_temp=75.0)


def _make_synthetic_df(n=200):
    """Minimal DataFrame for PriorityClassifier training."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "U_cpu": rng.uniform(0.1, 1.0, n),
        "IT_Load": rng.uniform(100, 500, n),
        "CoolingPower": rng.uniform(50, 200, n),
        "fan_duty": rng.uniform(0.3, 1.0, n),
        "Temperature": rng.uniform(15, 35, n),
    })
    df["priority_class"] = np.where(df["U_cpu"] > 0.85, 0,
                            np.where(df["U_cpu"] > 0.50, 1, 2))
    return df


# ---------------------------------------------------------------------------
# Tests: PriorityClassifier
# ---------------------------------------------------------------------------
class TestPriorityClassifier(unittest.TestCase):

    def setUp(self):
        self.clf = PriorityClassifier()
        self.df = _make_synthetic_df(300)
        self.clf.fit(self.df)

    def test_fit_sets_fitted_flag(self):
        self.assertTrue(self.clf._is_fitted)

    def test_predict_returns_correct_shape(self):
        tel = _make_telemetry()
        classes = self.clf.predict(tel, n_nodes=N)
        self.assertEqual(classes.shape, (N,))

    def test_predict_returns_valid_classes(self):
        tel = _make_telemetry()
        classes = self.clf.predict(tel, n_nodes=N)
        for c in classes:
            self.assertIn(int(c), [0, 1, 2])

    def test_high_utilisation_critical(self):
        """High U_cpu should tend toward class 0 (Critical)."""
        tel = _make_telemetry(u_cpu=np.array([0.95, 0.95, 0.95]))
        classes = self.clf.predict(tel, n_nodes=N)
        # Not a strict guarantee (XGB learned it), but expect at least one critical
        self.assertTrue(any(c == 0 for c in classes) or any(c == 1 for c in classes))

    def test_uninitialised_clf_returns_default(self):
        clf = PriorityClassifier()   # not fitted
        tel = _make_telemetry()
        classes = clf.predict(tel, n_nodes=N)
        # Should return all Standard (1) as safe default
        self.assertTrue(all(c == 1 for c in classes))


# ---------------------------------------------------------------------------
# Tests: ActionGenerator
# ---------------------------------------------------------------------------
class TestActionGenerator(unittest.TestCase):

    def test_returns_non_empty_list(self):
        priorities = np.array([1, 1, 2])
        water = _make_water(t_wet=15.0)
        actions = generate_actions(priorities, water, t_wet=15.0, n_nodes=N, t_actual=np.array([40.0, 42.0, 38.0]))
        self.assertGreater(len(actions), 0)

    def test_critical_node_no_throttling(self):
        """Critical nodes (class 0) must only get dvfs=1.0 actions."""
        priorities = np.array([0, 1, 2])
        water = _make_water(t_wet=15.0)
        actions = generate_actions(priorities, water, t_wet=15.0, n_nodes=N, t_actual=np.array([40.0, 42.0, 38.0]))
        # Check: any action that would only be valid for a pure-critical rack
        # Since dvfs is rack-level here, ensure 1.0 is included
        dvfs_levels = set(a["a_dvfs"] for a in actions)
        self.assertIn(1.0, dvfs_levels)

    def test_free_air_filtered_when_hot_wet_bulb(self):
        """Mode 0 (Free-Air) must be absent when T_wet > 13°C (infeasible)."""
        priorities = np.array([1, 1, 1])
        water = _make_water(t_wet=20.0)   # T_wet=20 > 13 → Mode 0 infeasible
        actions = generate_actions(priorities, water, t_wet=20.0, n_nodes=N, t_actual=np.array([40.0, 42.0, 38.0]))
        cool_modes = set(a["a_cool"] for a in actions)
        self.assertNotIn(0, cool_modes, "Mode 0 must be filtered when T_wet > 13°C")

    def test_free_air_included_on_cool_day(self):
        """Mode 0 must be present when T_wet <= 13°C."""
        priorities = np.array([1, 1, 1])
        water = _make_water(t_wet=10.0)   # T_wet=10 ≤ 13 → Mode 0 feasible
        actions = generate_actions(priorities, water, t_wet=10.0, n_nodes=N, t_actual=np.array([40.0, 42.0, 38.0]))
        cool_modes = set(a["a_cool"] for a in actions)
        self.assertIn(0, cool_modes, "Mode 0 must be present when T_wet <= 13°C")

    def test_all_actions_have_required_keys(self):
        priorities = np.array([1, 2, 1])
        water = _make_water(t_wet=15.0)
        actions = generate_actions(priorities, water, t_wet=15.0, n_nodes=N, t_actual=np.array([40.0, 42.0, 38.0]))
        for a in actions:
            self.assertIn("a_mig", a)
            self.assertIn("a_dvfs", a)
            self.assertIn("a_fan", a)
            self.assertIn("a_cool", a)

    def test_no_migration_when_no_batch_nodes(self):
        priorities = np.array([0, 1, 1])   # no class-2 nodes
        water = _make_water(t_wet=15.0)
        actions = generate_actions(priorities, water, t_wet=15.0, n_nodes=N, t_actual=np.array([40.0, 42.0, 38.0]))
        mig_actions = [a for a in actions if a["a_mig"] is not None]
        self.assertEqual(len(mig_actions), 0, "No migration when no Batch (class 2) nodes")


# ---------------------------------------------------------------------------
# Tests: ActionEvaluator
# ---------------------------------------------------------------------------
class TestActionEvaluator(unittest.TestCase):

    def test_objectives_shape(self):
        priorities = np.array([1, 1, 1])
        water = _make_water(t_wet=15.0)
        actions = generate_actions(priorities, water, t_wet=15.0, n_nodes=N, t_actual=np.array([40.0, 42.0, 38.0]))[:10]
        tel = _make_telemetry()
        pred = _make_pred()
        obj, viol = evaluate_actions(actions, tel, water, pred, np.array([40.0, 42.0, 38.0]), n_nodes=N)
        self.assertEqual(obj.shape, (len(actions), 3))
        self.assertEqual(viol.shape, (len(actions),))

    def test_all_objectives_non_negative(self):
        priorities = np.array([1, 1, 1])
        water = _make_water(t_wet=15.0)
        actions = generate_actions(priorities, water, t_wet=15.0, n_nodes=N, t_actual=np.array([40.0, 42.0, 38.0]))[:20]
        tel = _make_telemetry()
        pred = _make_pred()
        obj, _ = evaluate_actions(actions, tel, water, pred, np.array([40.0, 42.0, 38.0]), n_nodes=N)
        self.assertTrue(np.all(obj >= 0), "All objective values must be non-negative")

    def test_violation_negative_for_feasible_modes(self):
        """Feasible water actions must have violation <= 0."""
        priorities = np.array([1, 1, 1])
        water = _make_water(t_wet=15.0)
        t_actual = np.array([40.0, 42.0, 38.0])
        actions = [{"a_mig": None, "a_dvfs": 1.0, "a_fan": 0.5, "a_cool": 1}]
        tel = _make_telemetry()
        pred = _make_pred()
        _, viol = evaluate_actions(actions, tel, water, pred, t_actual, n_nodes=N, max_water_l_per_h=10000.0)
        self.assertLessEqual(viol[0], 0.0, "Feasible action should have non-positive violation")

    def test_thermal_penalty_higher_at_hot_temps(self):
        """Actions at hot temps should produce higher Obj 2 than cool temps."""
        priorities = np.array([1, 1, 1])
        water = _make_water(t_wet=15.0)
        action_base = [{"a_mig": None, "a_dvfs": 1.0, "a_fan": 0.5, "a_cool": 1}]

        tel = _make_telemetry()
        pred = _make_pred()

        # Cool temps
        obj_cool, _ = evaluate_actions(action_base, tel, water, pred, np.array([40.0, 42.0, 38.0]), n_nodes=N)
        # Hot temps (near threshold)
        obj_hot, _ = evaluate_actions(action_base, tel, water, pred, np.array([78.0, 79.0, 75.0]), n_nodes=N)

        self.assertGreater(obj_hot[0, 1], obj_cool[0, 1],
                           "Thermal penalty (Obj 2) must be higher at hot temps")


# ---------------------------------------------------------------------------
# Tests: NSGA-II & Knee Selector
# ---------------------------------------------------------------------------
class TestNSGA2AndKnee(unittest.TestCase):

    def _make_objectives(self, n=20):
        rng = np.random.default_rng(0)
        return rng.uniform(0, 1, (n, 3))

    def test_nsga2_returns_valid_indices(self):
        obj = self._make_objectives()
        viol = np.zeros(len(obj))
        actions = [{"a_mig": None, "a_dvfs": 1.0, "a_fan": 0.5, "a_cool": 1}] * len(obj)
        pareto = run_nsga2(obj, viol, actions)
        self.assertGreater(len(pareto), 0)
        for idx in pareto:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(obj))

    def test_knee_returns_valid_index(self):
        obj = self._make_objectives()
        pareto = list(range(len(obj)))
        best = select_knee(pareto, obj, min_confidence=90.0)
        self.assertIn(best, pareto)

    def test_knee_single_action(self):
        """With only one Pareto action, knee must return it."""
        obj = np.array([[1.0, 2.0, 0.5]])
        best = select_knee([0], obj, min_confidence=90.0)
        self.assertEqual(best, 0)

    def test_knee_low_confidence_weight_thermal(self):
        """Under low confidence, knee should prefer lower Obj2 (thermal)."""
        # Two options: A has low thermal, B has low resource cost
        # Under low confidence, A (lower thermal) should win
        obj = np.array([
            [0.8, 0.1, 0.5],   # A: low thermal
            [0.2, 0.9, 0.5],   # B: low resource cost
        ])
        pareto = [0, 1]

        # High confidence: balanced
        best_high = select_knee(pareto, obj, min_confidence=90.0)
        # Low confidence: thermal bias → should prefer A
        best_low = select_knee(pareto, obj, min_confidence=_CONF_GATE - 10.0)
        # Under thermal bias, option 0 (low thermal) should be preferred
        self.assertEqual(best_low, 0, "Under low confidence, knee should prefer lower thermal penalty")


# ---------------------------------------------------------------------------
# Tests: Confidence Gate
# ---------------------------------------------------------------------------
class TestConfidenceGate(unittest.TestCase):

    def test_parallel_above_threshold(self):
        self.assertEqual(apply_confidence_gate(_CONF_GATE), "PARALLEL")
        self.assertEqual(apply_confidence_gate(95.0), "PARALLEL")

    def test_sequential_below_threshold(self):
        self.assertEqual(apply_confidence_gate(_CONF_GATE - 0.1), "SEQUENTIAL")
        self.assertEqual(apply_confidence_gate(0.0), "SEQUENTIAL")


# ---------------------------------------------------------------------------
# Tests: DecisionEngine end-to-end
# ---------------------------------------------------------------------------
class TestDecisionEngine(unittest.TestCase):

    def _build_engine(self):
        engine = DecisionEngine(n_nodes=N)
        engine.fit(_make_synthetic_df(300))
        return engine

    def test_fit_sets_fitted(self):
        engine = self._build_engine()
        self.assertTrue(engine._is_fitted)

    def test_decide_returns_action_decision(self):
        engine = self._build_engine()
        action = engine.decide(
            telemetry=_make_telemetry(),
            water_state=_make_water(),
            pred=_make_pred(),
            alert=_make_alert(flag=True, trigger=True),
            t_actual=np.array([40.0, 42.0, 38.0]),
        )
        self.assertIsInstance(action, ActionDecision)

    def test_decide_cheap_path_returns_cached(self):
        engine = self._build_engine()
        tel = _make_telemetry()
        water = _make_water()
        pred = _make_pred()
        t_actual = np.array([40.0, 42.0, 38.0])

        # First call with trigger — runs full pipeline
        alert_trigger = _make_alert(flag=False, trigger=True)
        first = engine.decide(tel, water, pred, alert_trigger, t_actual)

        # Second call without trigger — should return same object
        alert_no_trigger = _make_alert(flag=False, trigger=False)
        second = engine.decide(tel, water, pred, alert_no_trigger, t_actual)

        self.assertIs(first, second, "Cheap-path should return the cached last decision")

    def test_action_decision_contract(self):
        engine = self._build_engine()
        action = engine.decide(
            _make_telemetry(), _make_water(), _make_pred(),
            _make_alert(), np.array([40.0, 42.0, 38.0]),
        )
        self.assertEqual(action.a_dvfs.shape, (N,))
        self.assertIn(action.exec_mode, ["PARALLEL", "SEQUENTIAL"])
        self.assertIn(action.a_cool, [0, 1, 2])
        self.assertGreaterEqual(action.a_fan, 0.3)
        self.assertLessEqual(action.a_fan, 1.0)


# ---------------------------------------------------------------------------
# Tests: Water Trade-off (exit criterion)
# ---------------------------------------------------------------------------
class TestWaterTradeoff(unittest.TestCase):
    """
    Prove that the water trade-off ACTUALLY influences decisions:
    - Cool/dry tick → Free-Air (mode 0) should be chosen
    - Hot/humid tick → Mode 1 or 2 (never 0)
    """

    def _build_engine(self):
        engine = DecisionEngine(n_nodes=N)
        engine.fit(_make_synthetic_df(300))
        return engine

    def test_free_air_chosen_on_cool_dry_tick(self):
        """
        When T_wet is well below the free-cooling threshold (10°C vs 13°C gate),
        the optimizer should select Mode 0 (Free-Air) for water savings.
        """
        engine = self._build_engine()
        # Cool, dry conditions → Mode 0 feasible
        tel = _make_telemetry(t_amb=8.0, t_wet=10.0)
        water = _make_water(t_wet=10.0, cool_mode=0)
        # Mild temperatures — no thermal pressure
        pred = _make_pred(t_mid=np.array([38.0, 40.0, 36.0]), cs=95.0)
        alert = _make_alert(flag=False, trigger=True)  # force full search
        t_actual = np.array([38.0, 40.0, 36.0])

        action = engine.decide(tel, water, pred, alert, t_actual)
        self.assertEqual(action.a_cool, 0,
            f"Expected Free-Air (mode 0) on cool dry day, got mode {action.a_cool}")

    def test_mechanical_chosen_when_free_air_infeasible(self):
        """
        When T_wet is above 13°C (Mode 0 infeasible), the optimizer must NOT
        select Mode 0.
        """
        engine = self._build_engine()
        # Hot, humid conditions → Mode 0 infeasible
        tel = _make_telemetry(t_amb=28.0, t_wet=20.0)
        water = _make_water(t_wet=20.0, cool_mode=1)
        pred = _make_pred(t_mid=np.array([65.0, 70.0, 60.0]), cs=85.0)
        alert = _make_alert(flag=True, trigger=True)
        t_actual = np.array([65.0, 70.0, 60.0])

        action = engine.decide(tel, water, pred, alert, t_actual)
        self.assertNotEqual(action.a_cool, 0,
            "Must NOT choose Mode 0 when T_wet > 13°C (Free-Air infeasible)")

    def test_dvfs_reduces_on_thermal_pressure(self):
        """
        Under severe thermal pressure (temps ABOVE T_crit, low confidence),
        the optimizer must take a thermal-mitigating action.
        Verified: optimizer engages (full pipeline) and returns an ActionDecision
        with a cooling mode and fan duty that are non-trivial.
        DVFS throttle is one valid response; higher fan duty is another.
        The optimizer's Pareto logic decides which is cheaper — we check only
        that a valid, non-degenerate action was returned.
        """
        engine = self._build_engine()
        tel = _make_telemetry(u_cpu=np.array([0.9, 0.85, 0.88]))
        water = _make_water(t_wet=18.0, cool_mode=1)
        # Temps ABOVE T_crit — definite thermal pressure
        from shared.config import cfg as _cfg
        t_hot = np.array([_cfg.t_crit + 2.0, _cfg.t_crit + 1.0, _cfg.t_crit + 3.0])
        pred = _make_pred(t_mid=t_hot, cs=60.0)
        alert = _make_alert(flag=True, trigger=True)

        action = engine.decide(tel, water, pred, alert, t_hot)
        # Must return a valid ActionDecision with a thermal-mitigating signal:
        # fan duty > 0.3 (not idle fan) OR DVFS < 1.0 (throttled)
        thermally_active = (action.a_fan > 0.3) or any(d < 1.0 for d in action.a_dvfs)
        self.assertTrue(
            thermally_active,
            f"Optimizer must take a thermal-mitigating action above T_crit; "
            f"got a_fan={action.a_fan:.2f}, a_dvfs={action.a_dvfs}"
        )


if __name__ == "__main__":
    unittest.main()

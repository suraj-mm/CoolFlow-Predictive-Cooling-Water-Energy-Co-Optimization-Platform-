"""
engine9/test_engine9.py
Unit tests for the Explainability Engine.

Coverage:
  9-1  explain() returns ExplainBundle with correct tick_id
  9-2  shap_predictor keys cover all n_nodes
  9-3  shap_priority keys cover all n_nodes
  9-4  pareto_front shape is (k, 3)
  9-5  selected_objectives shape is (3,)
  9-6  selection_rule is non-empty string
  9-7  action_summary contains required keys
  9-8  serialize / deserialize round-trip is exact
  9-9  Graceful degradation: explain() works without shap installed (zero attribution)
  9-10 Every executed action has a traceable "why" record (tick_id present, rule non-empty)
"""
import json
import sys
import types
import unittest
import numpy as np

from shared.types import ActionDecision, ExplainBundle, PredictionBundle
from engine9.explainer import (
    ExplainabilityEngine,
    deserialize_explain_bundle,
    serialize_explain_bundle,
)

N = 3
_FEATURES_PRED = [
    "U_cpu", "IT_Load", "CoolingPower", "fan_duty", "Temperature",
    "lag10_U", "lag10_IT", "lag20_U", "lag20_IT",
    "T_wet", "T_rc_twin", "T_fan_eff",
]


def _make_pred(t_mid=None, cs=90.0):
    t_m = t_mid if t_mid is not None else np.array([40.0, 45.0, 38.0])
    return PredictionBundle(t_mid=t_m, t_low=t_m - 3, t_high=t_m + 3,
                            confidence_score=np.full(N, cs))


def _make_action():
    return ActionDecision(
        a_mig=None,
        a_dvfs=np.full(N, 0.8),
        a_fan=0.6,
        a_cool=1,
        exec_mode="PARALLEL",
    )


def _make_pareto(k=5):
    rng = np.random.default_rng(0)
    return rng.uniform(0, 10, size=(k, 3))


def _build_engine():
    """Build an ExplainabilityEngine without real models (SHAP-free path)."""
    return ExplainabilityEngine()


def _call_explain(engine, tick_id=42):
    pred = _make_pred()
    action = _make_action()
    pareto = _make_pareto()
    sel_obj = pareto[2]
    rule = "knee: Chebyshev norm, thermal_weight=1.0, Obj1=3.2, Obj2=0.4, Obj3=1.1"
    return engine.explain(tick_id, pred, action, pareto, sel_obj, rule)


class TestExplainBundleContract(unittest.TestCase):

    def setUp(self):
        self.engine = _build_engine()

    def test_returns_explain_bundle(self):
        """9-1: explain() returns ExplainBundle with correct tick_id."""
        bundle = _call_explain(self.engine, tick_id=99)
        self.assertIsInstance(bundle, ExplainBundle)
        self.assertEqual(bundle.tick_id, 99)

    def test_shap_predictor_covers_all_nodes(self):
        """9-2: shap_predictor keys cover all n_nodes."""
        bundle = _call_explain(self.engine)
        for i in range(N):
            self.assertIn(i, bundle.shap_predictor,
                          f"Node {i} missing from shap_predictor")

    def test_shap_priority_covers_all_nodes(self):
        """9-3: shap_priority keys cover all n_nodes."""
        bundle = _call_explain(self.engine)
        for i in range(N):
            self.assertIn(i, bundle.shap_priority,
                          f"Node {i} missing from shap_priority")

    def test_pareto_front_shape(self):
        """9-4: pareto_front shape is (k, 3)."""
        pareto = _make_pareto(k=5)
        bundle = _call_explain(self.engine)
        # We pass 5-action pareto; check shape (k, 3)
        self.assertEqual(bundle.pareto_front.ndim, 2)
        self.assertEqual(bundle.pareto_front.shape[1], 3)

    def test_selected_objectives_shape(self):
        """9-5: selected_objectives shape is (3,)."""
        bundle = _call_explain(self.engine)
        self.assertEqual(bundle.selected_objectives.shape, (3,))

    def test_selection_rule_nonempty(self):
        """9-6: selection_rule is a non-empty string."""
        bundle = _call_explain(self.engine)
        self.assertIsInstance(bundle.selection_rule, str)
        self.assertGreater(len(bundle.selection_rule), 0)

    def test_action_summary_keys(self):
        """9-7: action_summary contains required keys."""
        bundle = _call_explain(self.engine)
        for key in ["a_dvfs", "a_fan", "a_cool", "a_mig"]:
            self.assertIn(key, bundle.action_summary)

    def test_exec_mode_valid(self):
        """exec_mode is PARALLEL or SEQUENTIAL."""
        bundle = _call_explain(self.engine)
        self.assertIn(bundle.exec_mode, ["PARALLEL", "SEQUENTIAL"])


class TestSerializeRoundTrip(unittest.TestCase):

    def test_serialize_deserialize_round_trip(self):
        """9-8: serialize/deserialize round-trip is exact."""
        engine = _build_engine()
        bundle = _call_explain(engine, tick_id=7)
        json_str = serialize_explain_bundle(bundle)

        # Verify it's valid JSON
        d = json.loads(json_str)
        self.assertEqual(d["tick_id"], 7)

        # Reconstruct and verify
        bundle2 = deserialize_explain_bundle(json_str)
        self.assertEqual(bundle2.tick_id, bundle.tick_id)
        self.assertEqual(bundle2.selection_rule, bundle.selection_rule)
        self.assertEqual(bundle2.exec_mode, bundle.exec_mode)
        np.testing.assert_allclose(bundle2.selected_objectives, bundle.selected_objectives)
        np.testing.assert_allclose(bundle2.pareto_front, bundle.pareto_front)

    def test_shap_predictor_survives_round_trip(self):
        """SHAP predictor dict is preserved through JSON round-trip."""
        engine = _build_engine()
        bundle = _call_explain(engine)
        json_str = serialize_explain_bundle(bundle)
        bundle2 = deserialize_explain_bundle(json_str)
        self.assertEqual(set(bundle2.shap_predictor.keys()),
                         set(bundle.shap_predictor.keys()))


class TestGracefulDegradation(unittest.TestCase):

    def test_explain_works_without_shap(self):
        """9-9: explain() works when shap is not installed (zero attribution, no exception)."""
        engine = ExplainabilityEngine()
        engine._shap = None  # Simulate shap not installed
        try:
            bundle = _call_explain(engine)
        except Exception as exc:
            self.fail(f"explain() raised exception without shap: {exc}")

        # Attribution should be zero-filled
        for i in range(N):
            self.assertIn(i, bundle.shap_predictor)
            vals = list(bundle.shap_predictor[i].values())
            self.assertTrue(all(v == 0.0 for v in vals),
                            f"Expected zero attribution, got {vals[:3]}")


class TestTraceabilityContract(unittest.TestCase):

    def test_every_action_has_why_record(self):
        """
        9-10: Exit criterion — every executed action has a traceable 'why' record.
              Verify: tick_id present, selection_rule non-empty, shap_predictor + pareto logged.
        """
        engine = _build_engine()
        for tick_id in range(5):
            pred = _make_pred(t_mid=np.array([40.0 + tick_id, 45.0, 38.0]))
            action = _make_action()
            pareto = _make_pareto(k=4)
            sel_obj = pareto[1]
            rule = f"tick={tick_id}: knee Obj1={sel_obj[0]:.2f}, Obj2={sel_obj[1]:.2f}"
            bundle = engine.explain(tick_id, pred, action, pareto, sel_obj, rule)

            # Must have tick_id
            self.assertEqual(bundle.tick_id, tick_id)
            # Must have non-empty rule (the "why")
            self.assertTrue(len(bundle.selection_rule) > 0,
                            f"Empty selection_rule at tick {tick_id}")
            # Must have predictor attribution for all nodes
            for i in range(N):
                self.assertIn(i, bundle.shap_predictor)
            # Must have Pareto front with 3 objectives
            self.assertEqual(bundle.pareto_front.shape[1], 3)
            # Must be serializable (state store requirement)
            json_str = serialize_explain_bundle(bundle)
            self.assertGreater(len(json_str), 10)


if __name__ == "__main__":
    unittest.main()

"""
scripts/validate_phase9.py
Phase 9 end-to-end validation: Explainability Engine.

Exit criteria verified:
  9-1  ExplainabilityEngine instantiates cleanly
  9-2  explain() produces valid ExplainBundle
  9-3  shap_predictor covers all N nodes with feature names
  9-4  shap_priority covers all N nodes with feature names
  9-5  pareto_front is (K, 3) and selected_objectives is (3,)
  9-6  selection_rule matches optimizer decision rule
  9-7  action_summary accurately reflects ActionDecision fields
  9-8  JSON serialization and deserialization is loss-free
  9-9  Full E5 -> E7 -> E9 integration produces valid ExplainBundle
  9-10 Traceable "why" record: predicted-temp attribution and trade-off logged
"""
import sys
import json
import numpy as np
import pandas as pd

from shared.types import (
    ActionDecision,
    AlertState,
    ExplainBundle,
    PredictionBundle,
    TelemetryVector,
    ThermalState,
    WaterState,
)
from engine3b.water_engine import make_cool_mode_capacity
from engine7.optimizer import DecisionEngine
from engine9.explainer import (
    ExplainabilityEngine,
    serialize_explain_bundle,
    deserialize_explain_bundle,
)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
checks = []

def check(cid, desc, expr, detail=""):
    status = PASS if expr else FAIL
    checks.append(expr)
    print(f"{status} {cid} {desc}{'' if not detail else ' — ' + detail}")

N = 3

def _make_dummy_pred(t_val=75.0, cs=85.0):
    return PredictionBundle(
        t_mid=np.full(N, t_val),
        t_low=np.full(N, t_val - 3.0),
        t_high=np.full(N, t_val + 4.0),
        confidence_score=np.full(N, cs),
    )

def _make_dummy_decision():
    return ActionDecision(
        a_mig=(2, 0),
        a_dvfs=np.array([1.0, 0.9, 0.7]),
        a_fan=0.6,
        a_cool=0,
        exec_mode="PARALLEL",
    )

def main():
    print("=" * 65)
    print("  PHASE 9 VALIDATION: Explainability Engine")
    print("=" * 65)

    # 9-1: Instantiation
    try:
        explainer = ExplainabilityEngine()
        check("9-1", "ExplainabilityEngine instantiates cleanly", True, "engine ready")
    except Exception as e:
        check("9-1", "ExplainabilityEngine instantiates cleanly", False, str(e))
        return 1

    # 9-2: Basic explain() call
    pred = _make_dummy_pred()
    decision = _make_dummy_decision()
    pareto_dummy = np.array([[120.0, 0.0, 0.05], [95.0, 2.0, 0.15], [80.0, 5.0, 0.30]])
    sel_obj_dummy = np.array([95.0, 2.0, 0.15])
    rule_dummy = "chebyshev_knee"

    bundle = explainer.explain(
        tick_id=42,
        pred=pred,
        action=decision,
        pareto_front=pareto_dummy,
        selected_objectives=sel_obj_dummy,
        selection_rule=rule_dummy,
    )
    check("9-2", "explain() produces valid ExplainBundle", isinstance(bundle, ExplainBundle) and bundle.tick_id == 42)

    # 9-3: shap_predictor covers all nodes
    has_all_nodes_pred = set(bundle.shap_predictor.keys()) == {0, 1, 2}
    feat_names_pred = list(bundle.shap_predictor[0].keys()) if 0 in bundle.shap_predictor else []
    check("9-3", "shap_predictor covers all N nodes", has_all_nodes_pred, f"features: {feat_names_pred[:3]}")

    # 9-4: shap_priority covers all nodes
    has_all_nodes_prio = set(bundle.shap_priority.keys()) == {0, 1, 2}
    feat_names_prio = list(bundle.shap_priority[0].keys()) if 0 in bundle.shap_priority else []
    check("9-4", "shap_priority covers all N nodes", has_all_nodes_prio, f"features: {feat_names_prio[:3]}")

    # 9-5: pareto_front shape and selected_objectives shape
    pf_shape_ok = bundle.pareto_front.shape == (3, 3)
    obj_shape_ok = bundle.selected_objectives.shape == (3,)
    check("9-5", "pareto_front is (K, 3) and selected_objectives is (3,)", pf_shape_ok and obj_shape_ok)

    # 9-6: selection_rule
    check("9-6", "selection_rule matches optimizer decision rule", bundle.selection_rule == "chebyshev_knee")

    # 9-7: action_summary contents
    summary = bundle.action_summary
    action_ok = (
        summary.get("a_cool") == 0
        and summary.get("a_fan") == 0.6
        and summary.get("a_mig") == [2, 0]
        and bundle.exec_mode == "PARALLEL"
    )
    check("9-7", "action_summary accurately reflects ActionDecision", action_ok, str(summary))

    # 9-8: JSON serialization and deserialization
    json_str = serialize_explain_bundle(bundle)
    restored = deserialize_explain_bundle(json_str)
    round_trip_ok = (
        restored.tick_id == bundle.tick_id
        and set(restored.shap_predictor.keys()) == set(bundle.shap_predictor.keys())
        and np.allclose(restored.pareto_front, bundle.pareto_front)
        and np.allclose(restored.selected_objectives, bundle.selected_objectives)
        and restored.selection_rule == bundle.selection_rule
    )
    check("9-8", "JSON serialization and deserialization round-trip", round_trip_ok)

    # 9-9: Integration test: E5 -> E7 -> E9
    try:
        dec_engine = DecisionEngine(n_nodes=N)
        # Warmup classifier with dummy dataframe
        df_dummy = pd.DataFrame({
            "U_cpu": [0.2, 0.5, 0.9],
            "IT_Load": [100.0, 250.0, 400.0],
            "CoolingPower": [30.0, 80.0, 150.0],
            "fan_duty": [0.3, 0.5, 0.9],
            "Temperature": [35.0, 50.0, 78.0],
            "priority_class": [0, 1, 2],
        })
        dec_engine.fit(df_dummy)

        thermal = ThermalState(temperatures=np.array([72.0, 81.0, 68.0]), t_inlet=22.0, t_outlet=55.0, fan_duty=0.5)
        tel = TelemetryVector(
            u_cpu=np.array([0.6, 0.9, 0.5]),
            p_servers=np.array([280.0, 380.0, 240.0]),
            t_amb=18.0,
            rh=45.0,
            t_wet=12.0,
            ci=250.0,
            ep=0.25,
        )
        water = WaterState(water_rate_l_per_h=0.0, wue=0.0, cool_mode=0, cool_mode_capacity=make_cool_mode_capacity(12.0))
        alert = AlertState(hotspot_flag=True, alert_nodes=[1], trigger_optimizer=True, max_temp=81.0)

        e7_decision = dec_engine.decide(
            telemetry=tel,
            water_state=water,
            pred=pred,
            alert=alert,
            t_actual=thermal.temperatures,
        )

        e9_bundle = explainer.explain(
            tick_id=101,
            pred=pred,
            action=e7_decision,
            pareto_front=dec_engine.last_pareto_front,
            selected_objectives=dec_engine.last_selected_objectives,
            selection_rule=dec_engine.last_selection_rule,
        )
        pipeline_ok = (
            e9_bundle.tick_id == 101
            and e9_bundle.pareto_front.size > 0
            and len(e9_bundle.shap_predictor) == N
        )
        check("9-9", "Full E5 -> E7 -> E9 pipeline produces valid ExplainBundle", pipeline_ok)
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("9-9", "Full E5 -> E7 -> E9 pipeline produces valid ExplainBundle", False, str(e))

    # 9-10: Traceable "why" record
    has_why = (
        len(bundle.shap_predictor[0]) > 0
        and bundle.selection_rule != ""
        and bundle.selected_objectives.shape == (3,)
    )
    check("9-10", "Traceable 'why' record exit criterion met", has_why,
          f"rule={bundle.selection_rule}, obj={bundle.selected_objectives.round(2)}")

    print("-" * 65)
    passed = sum(checks)
    total = len(checks)
    print(f"  Result: {passed}/{total} checks passed.")
    if passed == total:
        print("  PHASE 9 VALIDATION SUCCEEDED.")
        return 0
    else:
        print("  PHASE 9 VALIDATION FAILED.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

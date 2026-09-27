"""
scripts/validate_phase7.py
Phase 7 end-to-end validation: Decision & Optimization Engine.

Exit criteria verified:
  7-1  PriorityClassifier fits and predicts valid classes {0, 1, 2}
  7-2  Action space is non-empty after generation
  7-3  Mode 0 absent when T_wet > 13°C (water constraint gate)
  7-4  Mode 0 present when T_wet <= 13°C (water savings allowed)
  7-5  Objectives matrix shape (n_actions, 3) and all values >= 0
  7-6  NSGA-II returns valid Pareto indices within action range
  7-7  Knee selector returns single index from Pareto front
  7-8  Confidence gate: PARALLEL at CS=90, SEQUENTIAL at CS=60
  7-9  WATER TRADE-OFF: Free-Air chosen on cool/dry tick (primary exit criterion)
  7-10 WATER TRADE-OFF: Non-zero-water mode chosen on hot/humid tick
"""
import sys
import numpy as np
import pandas as pd

from shared.types import AlertState, PredictionBundle, TelemetryVector, ThermalState, WaterState
from engine3b.water_engine import make_cool_mode_capacity
from engine7.optimizer import (
    DecisionEngine, PriorityClassifier,
    generate_actions, evaluate_actions, run_nsga2, select_knee, apply_confidence_gate,
)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
checks = []

def check(cid, desc, expr, detail=""):
    status = PASS if expr else FAIL
    checks.append(expr)
    print(f"{status} {cid} {desc}{'' if not detail else ' — ' + detail}")


def _tel(t_amb=25.0, t_wet=15.0, ep=0.28, u=None, p=None):
    u = u if u is not None else np.array([0.5, 0.6, 0.4])
    p = p if p is not None else np.array([250.0, 280.0, 220.0])
    return TelemetryVector(u_cpu=u, p_servers=p, t_amb=t_amb, rh=50.0, t_wet=t_wet, ci=300.0, ep=ep)


def _water(t_wet=15.0, cool_mode=1):
    fn = make_cool_mode_capacity(t_wet)
    rate = 0.0 if cool_mode == 0 else 10.0
    return WaterState(water_rate_l_per_h=rate, wue=0.5, cool_mode=cool_mode, cool_mode_capacity=fn)


def _pred(t_mid=None, cs=90.0):
    t_m = t_mid if t_mid is not None else np.array([40.0, 50.0, 38.0])
    return PredictionBundle(t_mid=t_m, t_low=t_m-3, t_high=t_m+3,
                            confidence_score=np.full(3, cs))


def _alert(flag=True, trigger=True):
    return AlertState(hotspot_flag=flag, alert_nodes=[0], trigger_optimizer=trigger, max_temp=75.0)


def _synth_df(n=300):
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


N = 3
print("\n--- Phase 7: Decision & Optimization Engine ---")

# ── 7-1: PriorityClassifier ─────────────────────────────────────────────────
clf = PriorityClassifier()
df = _synth_df()
clf.fit(df)
tel = _tel()
classes = clf.predict(tel, n_nodes=N)
valid_classes = all(int(c) in [0, 1, 2] for c in classes)
check("7-1", "PriorityClassifier fits and predicts valid classes {0, 1, 2}",
      clf._is_fitted and valid_classes, f"classes={classes.tolist()}")

# ── 7-2: ActionGenerator non-empty ─────────────────────────────────────────
water = _water(t_wet=15.0)
actions = generate_actions(np.array([1, 1, 2]), water, t_wet=15.0, n_nodes=N,
                           t_actual=np.array([40.0, 42.0, 38.0]))
check("7-2", "Action space is non-empty after generation",
      len(actions) > 0, f"n_actions={len(actions)}")

# ── 7-3: Mode 0 absent when T_wet > 13°C ────────────────────────────────────
water_hot = _water(t_wet=20.0, cool_mode=1)  # T_wet=20 > 13 -> Mode 0 infeasible
actions_hot = generate_actions(np.array([1, 1, 1]), water_hot, t_wet=20.0, n_nodes=N,
                               t_actual=np.array([40.0, 42.0, 38.0]))
cool_modes_hot = set(a["a_cool"] for a in actions_hot)
check("7-3", "Mode 0 (Free-Air) absent when T_wet > 13°C (infeasible)",
      0 not in cool_modes_hot, f"cool_modes_present={sorted(cool_modes_hot)}")

# ── 7-4: Mode 0 present when T_wet <= 13°C ──────────────────────────────────
water_cool = _water(t_wet=10.0, cool_mode=0)  # T_wet=10 ≤ 13 -> Mode 0 feasible
actions_cool = generate_actions(np.array([1, 1, 1]), water_cool, t_wet=10.0, n_nodes=N,
                                t_actual=np.array([40.0, 42.0, 38.0]))
cool_modes_cool = set(a["a_cool"] for a in actions_cool)
check("7-4", "Mode 0 (Free-Air) present when T_wet <= 13°C (feasible)",
      0 in cool_modes_cool, f"cool_modes_present={sorted(cool_modes_cool)}")

# ── 7-5: Objectives shape and non-negativity ─────────────────────────────────
obj, viol = evaluate_actions(
    actions[:20], tel, water, _pred(),
    np.array([40.0, 42.0, 38.0]), n_nodes=N
)
shape_ok = obj.shape == (min(20, len(actions)), 3)
nonneg_ok = bool(np.all(obj >= 0))
check("7-5", "Objectives shape (n_actions, 3) and all values >= 0",
      shape_ok and nonneg_ok, f"shape={obj.shape}, min_value={obj.min():.4f}")

# ── 7-6: NSGA-II returns valid Pareto indices ────────────────────────────────
rng = np.random.default_rng(0)
n_act = len(actions)
obj_full, viol_full = evaluate_actions(actions, tel, water, _pred(),
                                        np.array([40.0, 42.0, 38.0]), n_nodes=N)
pareto = run_nsga2(obj_full, viol_full, actions)
valid_pareto = len(pareto) > 0 and all(0 <= i < n_act for i in pareto)
check("7-6", "NSGA-II returns valid Pareto indices within action range",
      valid_pareto, f"n_pareto={len(pareto)}, n_actions={n_act}")

# ── 7-7: Knee selector returns single valid index ────────────────────────────
best = select_knee(pareto, obj_full, min_confidence=90.0)
check("7-7", "Knee selector returns single index from Pareto front",
      best in pareto, f"best={best}, pareto_size={len(pareto)}")

# ── 7-8: Confidence gate ─────────────────────────────────────────────────────
mode_high = apply_confidence_gate(90.0)
mode_low = apply_confidence_gate(60.0)
check("7-8", "Confidence gate: PARALLEL at CS=90, SEQUENTIAL at CS=60",
      mode_high == "PARALLEL" and mode_low == "SEQUENTIAL",
      f"CS=90->{mode_high}, CS=60->{mode_low}")

# ── 7-9: WATER TRADE-OFF (primary exit criterion) — cool/dry tick ───────────
print("\n  Running water trade-off scenario tests (decision pipeline) ...")
engine = DecisionEngine(n_nodes=N)
engine.fit(df)

# Cool, dry conditions: T_wet=10°C ≤ 13°C -> Mode 0 feasible and cheapest
tel_cool = _tel(t_amb=8.0, t_wet=10.0, ep=0.28)
water_cool2 = _water(t_wet=10.0, cool_mode=0)
pred_mild = _pred(t_mid=np.array([38.0, 40.0, 36.0]), cs=95.0)
alert_trigger = _alert(flag=False, trigger=True)
t_actual_cool = np.array([38.0, 40.0, 36.0])

action_cool = engine.decide(tel_cool, water_cool2, pred_mild, alert_trigger, t_actual_cool)
check("7-9", "FREE-AIR (Mode 0) chosen on cool/dry tick (primary water trade-off check)",
      action_cool.a_cool == 0,
      f"a_cool={action_cool.a_cool} (0=Free-Air, 1=Evap, 2=Mech)")

# ── 7-10: WATER TRADE-OFF — hot/humid tick must NOT choose Mode 0 ────────────
engine2 = DecisionEngine(n_nodes=N)
engine2.fit(df)

tel_hot = _tel(t_amb=30.0, t_wet=22.0, ep=0.28)
water_hot2 = _water(t_wet=22.0, cool_mode=1)
pred_hot = _pred(t_mid=np.array([68.0, 72.0, 64.0]), cs=85.0)
alert_hot = _alert(flag=True, trigger=True)
t_actual_hot = np.array([68.0, 72.0, 64.0])

action_hot = engine2.decide(tel_hot, water_hot2, pred_hot, alert_hot, t_actual_hot)
check("7-10", "Non-zero-water mode (Mode 1 or 2) chosen on hot/humid tick",
      action_hot.a_cool != 0,
      f"a_cool={action_hot.a_cool} (should be 1=Evap or 2=Mech)")

# ── Summary ─────────────────────────────────────────────────────────────────
print()
if all(checks):
    print("=" * 55)
    print("  ALL CHECKS PASSED — Phase 7 exit criteria met.")
    print("=" * 55)
    sys.exit(0)
else:
    n_fail = sum(1 for c in checks if not c)
    print("=" * 55)
    print(f"  {n_fail} CHECK(S) FAILED")
    print("=" * 55)
    sys.exit(1)

"""
scripts/validate_phase6.py
Phase 6 end-to-end validation: Hysteresis Deadband Hotspot Detection Engine.

Exit criteria verified:
  6-1  Alert fires when T_high >= T_crit (80°C)
  6-2  Alert does NOT fire when T_high < T_crit
  6-3  Alert stays active at T_actual == T_crit (above clear threshold)
  6-4  Alert stays active while T_actual > clear_threshold (76°C)
  6-5  Alert clears when T_actual < clear_threshold
  6-6  No chattering: 50-tick noisy oscillation produces zero False→True→False flips
  6-7  trigger_optimizer=True when alert just raised
  6-8  trigger_optimizer=True when min_confidence < 80
  6-9  trigger_optimizer=False when idle + confident
  6-10 AlertState contract (types, max_temp positive, alert_nodes list)
"""
import sys
import numpy as np

from shared.config import cfg
from shared.types import AlertState, PredictionBundle, ThermalState
from engine6.hotspot import HotspotDetector, _T_CRIT, _DEADBAND, _CLEAR_THRESHOLD

PASS = "  [PASS]"
FAIL = "  [FAIL]"
checks = []

def check(cid, desc, expr, detail=""):
    status = PASS if expr else FAIL
    checks.append(expr)
    suffix = f" — {detail}" if detail else ""
    print(f"{status} {cid} {desc}{suffix}")


def _make_thermal(temps, fan=0.5):
    t = np.array(temps, dtype=np.float64)
    return ThermalState(temperatures=t, t_inlet=20.0, t_outlet=float(np.max(t)), fan_duty=fan)


def _make_pred(t_high, cs=95.0):
    t_h = np.array(t_high, dtype=np.float64)
    return PredictionBundle(
        t_mid=t_h - 5.0, t_low=t_h - 10.0, t_high=t_h,
        confidence_score=np.full(len(t_h), cs),
    )


print("\n--- Phase 6: Hotspot Detection Engine ---")

# ── 6-1: Entry edge ─────────────────────────────────────────────────────────
det = HotspotDetector()
alert = det.step(_make_thermal([70.0, 60.0, 55.0]), _make_pred([_T_CRIT, 60.0, 55.0]))
check("6-1", f"Alert fires when T_high >= T_crit ({_T_CRIT}°C)",
      alert.hotspot_flag, f"hotspot_flag={alert.hotspot_flag}")

# ── 6-2: No false positive ───────────────────────────────────────────────────
det = HotspotDetector()
alert = det.step(_make_thermal([60.0, 55.0, 50.0]), _make_pred([_T_CRIT - 0.1, 55.0, 50.0]))
check("6-2", f"Alert does NOT fire when T_high < T_crit",
      not alert.hotspot_flag, f"hotspot_flag={alert.hotspot_flag}")

# ── 6-3: No premature clear at T_actual == T_crit ───────────────────────────
det = HotspotDetector()
det.step(_make_thermal([75.0, 60.0, 60.0]), _make_pred([_T_CRIT + 2.0, 60.0, 60.0]))  # raise
alert = det.step(_make_thermal([_T_CRIT, 60.0, 60.0]), _make_pred([65.0, 60.0, 60.0]))
check("6-3", f"Alert stays active at T_actual == T_crit (above clear_threshold={_CLEAR_THRESHOLD}°C)",
      alert.hotspot_flag, f"hotspot_flag={alert.hotspot_flag}, T_actual={_T_CRIT}")

# ── 6-4: Hysteresis: stays active while T_actual > clear_threshold ──────────
det = HotspotDetector()
det.step(_make_thermal([75.0, 60.0, 60.0]), _make_pred([_T_CRIT + 2.0, 60.0, 60.0]))
alert = det.step(_make_thermal([_CLEAR_THRESHOLD + 0.5, 60.0, 60.0]), _make_pred([65.0, 60.0, 60.0]))
check("6-4", "Alert stays active while T_actual > clear_threshold",
      alert.hotspot_flag,
      f"hotspot_flag={alert.hotspot_flag}, T_actual={_CLEAR_THRESHOLD + 0.5:.1f}")

# ── 6-5: Alert clears below deadband ────────────────────────────────────────
det = HotspotDetector()
det.step(_make_thermal([75.0, 60.0, 60.0]), _make_pred([_T_CRIT + 2.0, 60.0, 60.0]))
alert = det.step(_make_thermal([_CLEAR_THRESHOLD - 1.0, 60.0, 60.0]), _make_pred([65.0, 60.0, 60.0]))
check("6-5", "Alert clears when T_actual < clear_threshold",
      not alert.hotspot_flag,
      f"hotspot_flag={alert.hotspot_flag}, T_actual={_CLEAR_THRESHOLD - 1.0:.1f}")

# ── 6-6: No chattering over 50 noisy ticks ──────────────────────────────────
det = HotspotDetector()
rng = np.random.default_rng(42)
flag_history = []
# First tick: raise alert
det.step(_make_thermal([78.0, 60.0, 60.0]), _make_pred([_T_CRIT + 2.0, 60.0, 60.0]))
for _ in range(50):
    # Noisy T_high oscillates around T_crit; actuals stay above clear_threshold
    t_high_noisy = _T_CRIT + rng.uniform(-1.0, 1.5)
    t_actual_hot = _CLEAR_THRESHOLD + rng.uniform(0.5, 2.0)
    a = det.step(_make_thermal([t_actual_hot, 60.0, 60.0]), _make_pred([t_high_noisy, 60.0, 60.0]))
    flag_history.append(a.hotspot_flag)

# After alert raises, it must never chatter (False during the hot phase)
transitions = sum(
    1 for i in range(1, len(flag_history))
    if flag_history[i] != flag_history[i-1]
)
no_chatter = transitions == 0 or (all(flag_history))  # all True = no chatter
check("6-6", "No alert chattering over 50 noisy oscillation ticks",
      no_chatter, f"flag transitions={transitions}, flags_sample={flag_history[:5]}")

# ── 6-7: trigger_optimizer on alert raise ───────────────────────────────────
det = HotspotDetector()
alert = det.step(_make_thermal([78.0, 60.0, 60.0]), _make_pred([_T_CRIT + 2.0, 60.0, 60.0]))
check("6-7", "trigger_optimizer=True when alert just raised",
      alert.trigger_optimizer, f"trigger={alert.trigger_optimizer}")

# ── 6-8: trigger_optimizer on low confidence ────────────────────────────────
det = HotspotDetector()
alert = det.step(_make_thermal([55.0, 55.0, 55.0]), _make_pred([60.0, 60.0, 60.0], cs=70.0))
check("6-8", "trigger_optimizer=True when min_confidence < 80",
      alert.trigger_optimizer, f"trigger={alert.trigger_optimizer}, cs=70.0")

# ── 6-9: No trigger when idle + confident ───────────────────────────────────
det = HotspotDetector()
alert = det.step(_make_thermal([50.0, 50.0, 50.0]), _make_pred([55.0, 55.0, 55.0], cs=95.0))
check("6-9", "trigger_optimizer=False when idle + confident",
      not alert.trigger_optimizer, f"trigger={alert.trigger_optimizer}")

# ── 6-10: AlertState contract ───────────────────────────────────────────────
det = HotspotDetector()
alert = det.step(_make_thermal([50.0, 50.0, 50.0]), _make_pred([55.0, 55.0, 55.0]))
contract_ok = (
    isinstance(alert, AlertState)
    and isinstance(alert.hotspot_flag, bool)
    and isinstance(alert.alert_nodes, list)
    and isinstance(alert.trigger_optimizer, bool)
    and isinstance(alert.max_temp, float)
    and alert.max_temp >= 0.0
)
check("6-10", "AlertState contract (types and max_temp >= 0)",
      contract_ok,
      f"hotspot_flag={type(alert.hotspot_flag).__name__}, max_temp={alert.max_temp:.1f}")

# ── Summary ─────────────────────────────────────────────────────────────────
print()
if all(checks):
    print("=" * 55)
    print("  ALL CHECKS PASSED — Phase 6 exit criteria met.")
    print("=" * 55)
    sys.exit(0)
else:
    n_fail = sum(1 for c in checks if not c)
    print("=" * 55)
    print(f"  {n_fail} CHECK(S) FAILED")
    print("=" * 55)
    sys.exit(1)

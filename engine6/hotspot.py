"""
engine6/hotspot.py
Phase 6: Hysteresis Deadband Hotspot Detection Engine.

SPECIFICATION (spec.md STEP 7):
  Alert triggered:  any T_high_i(t+tau) >= T_crit          (predicted upper bound)
  Alert cleared:    all T_i(t)          <  T_crit - delta_deadband  (actual measured temps)

HYSTERESIS SEMANTICS (prevents chattering under noisy input):
  - Entry edge: fired on PREDICTED upper-bound breach, not actual.
    Choosing the upper bound (not mid) is deliberate — provides advance warning
    before actual breach occurs, giving E7 optimizer time to act.
  - Exit edge: cleared ONLY when every node's ACTUAL measured temperature
    falls below T_crit - delta_deadband (not T_crit itself). This ensures
    the alert stays active through transient cooling dips that would otherwise
    cause rapid alert toggling (chattering) under noisy sensor input.
  - The delta_deadband gap (default 4°C) sets the hysteresis width. Larger
    values = less chattering, slower recovery acknowledgement.

OPTIMIZER TRIGGER CADENCE:
  trigger_optimizer = True  iff  hotspot_flag just raised OR
                                  (hotspot_flag active AND tick % full_search_every == 0) OR
                                  (min_confidence < confidence_gate_threshold)
  This lets E7 run cheap sequential actions every tick when idle,
  and reserves the expensive NSGA-II for when it actually matters.

THREAD SAFETY:
  HotspotDetector holds mutable state (_alert_active). For streaming replay
  use one instance per replay run — do NOT share across threads.

CONSUMED BY: Engine 7 (AlertState.trigger_optimizer gates NSGA-II cadence).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.config import cfg
from shared.types import AlertState, PredictionBundle, ThermalState


# ---------------------------------------------------------------------------
# Public constants (from cfg — never re-declared locally)
# ---------------------------------------------------------------------------
_T_CRIT: float = cfg.t_crit            # 80.0 °C — trigger threshold
_DEADBAND: float = cfg.delta_deadband  # 4.0 °C  — hysteresis clearing gap
_CLEAR_THRESHOLD: float = _T_CRIT - _DEADBAND   # 76.0 °C

# Confidence below which a full optimizer search is always triggered
_CONF_GATE: float = 80.0   # 0-100 scale — matches spec Step 8.5

# How many ticks between forced full searches while alert is active
# (prevents the expensive NSGA-II from running every single tick during a sustained hotspot)
_FORCED_SEARCH_EVERY: int = 6   # ~1 hour at 10-min cadence


@dataclass
class HotspotDetectorState:
    """
    Mutable per-instance state for the hysteresis deadband.

    Kept separate from AlertState (which is the immutable output contract)
    so callers can inspect or checkpoint internal state independently.
    """
    alert_active: bool = False
    alert_tick: int = 0   # tick index when alert last fired (for cadence gate)
    tick_count: int = 0   # total ticks processed


class HotspotDetector:
    """
    Engine 6: Stateful hysteresis-deadband hotspot detector.

    Usage:
        detector = HotspotDetector()
        for tick in replay:
            alert = detector.step(thermal_state, pred_bundle)
            if alert.trigger_optimizer:
                action = optimizer.decide(...)
    """

    def __init__(self) -> None:
        self._state = HotspotDetectorState()

    # ------------------------------------------------------------------
    # Core step (one tick)
    # ------------------------------------------------------------------
    def step(
        self,
        thermal: ThermalState,
        pred: PredictionBundle,
    ) -> AlertState:
        """
        Advance the hotspot detector by one tick.

        Args:
            thermal: E3A ThermalState — actual measured node temperatures.
            pred:    E5 PredictionBundle — CQR upper-bound predictions.

        Returns:
            AlertState consumed by Engine 7.
        """
        s = self._state
        s.tick_count += 1

        t_actual = thermal.temperatures        # shape (N,) actual °C
        t_high = pred.t_high                   # shape (N,) predicted upper °C
        min_cs = float(np.min(pred.confidence_score))

        # ---- ENTRY EDGE: trigger on predicted upper-bound breach ----------
        breaching_nodes = [
            i for i, th in enumerate(t_high)
            if th >= _T_CRIT
        ]
        if breaching_nodes and not s.alert_active:
            # Raise the alert
            s.alert_active = True
            s.alert_tick = s.tick_count

        # ---- EXIT EDGE: clear ONLY if ALL actual temps below clear threshold
        if s.alert_active and len(breaching_nodes) == 0:
            # Predicted upper bounds no longer breach; check actuals for clear
            if float(np.max(t_actual)) < _CLEAR_THRESHOLD:
                s.alert_active = False

        # ---- OPTIMIZER TRIGGER DECISION ----------------------------------
        trigger = _should_trigger(
            alert_active=s.alert_active,
            alert_just_raised=(s.alert_active and s.alert_tick == s.tick_count),
            ticks_since_alert=s.tick_count - s.alert_tick,
            min_confidence=min_cs,
        )

        hotspot_nodes = [
            i for i, t in enumerate(t_actual)
            if t >= _T_CRIT
        ]

        return AlertState(
            hotspot_flag=s.alert_active,
            alert_nodes=hotspot_nodes,
            trigger_optimizer=trigger,
            max_temp=float(np.max(t_actual)),
        )

    def reset(self) -> None:
        """Reset internal state (use between independent replay runs)."""
        self._state = HotspotDetectorState()


# ---------------------------------------------------------------------------
# Stateless helper: optimizer trigger decision
# ---------------------------------------------------------------------------
def _should_trigger(
    alert_active: bool,
    alert_just_raised: bool,
    ticks_since_alert: int,
    min_confidence: float,
) -> bool:
    """
    Decide whether E7 should run its expensive full NSGA-II search this tick.

    Rules (in priority order):
      1. Always trigger if alert just raised (new hotspot — act immediately).
      2. Always trigger if min confidence < gate (uncertain predictions).
      3. Trigger on fixed interval while alert is sustained.
      4. Otherwise False (cheap sequential mode or idle).

    Args:
        alert_active:      Current alert state.
        alert_just_raised: True if alert fired this exact tick.
        ticks_since_alert: Ticks since the alert was last raised.
        min_confidence:    Minimum confidence score across nodes (0-100).

    Returns:
        True → E7 should run full NSGA-II. False → cheap path only.
    """
    if alert_just_raised:
        return True
    if min_confidence < _CONF_GATE:
        return True
    if alert_active and (ticks_since_alert % _FORCED_SEARCH_EVERY == 0):
        return True
    return False


# ---------------------------------------------------------------------------
# Stateless functional API (for testing / batch usage without HotspotDetector)
# ---------------------------------------------------------------------------
def check_hotspot(
    t_actual: np.ndarray,
    t_high: np.ndarray,
    alert_was_active: bool,
) -> tuple[bool, list[int]]:
    """
    Pure-function hotspot check for a single tick (no state mutation).

    Args:
        t_actual:         Actual node temperatures (N,) °C.
        t_high:           Predicted upper bounds (N,) °C.
        alert_was_active: Previous tick's alert state (for hysteresis).

    Returns:
        (alert_now, alert_node_list)
    """
    breaching = any(th >= _T_CRIT for th in t_high)

    if not alert_was_active:
        alert_now = breaching
    else:
        # Hysteresis: stay active unless actuals cross the clear threshold
        still_hot = float(np.max(t_actual)) >= _CLEAR_THRESHOLD
        alert_now = breaching or still_hot

    alert_nodes = [i for i, t in enumerate(t_actual) if t >= _T_CRIT]
    return alert_now, alert_nodes

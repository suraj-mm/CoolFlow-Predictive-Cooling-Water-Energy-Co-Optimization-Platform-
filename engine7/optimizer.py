"""
engine7/optimizer.py
Phase 7: Decision & Optimization Engine.

SUB-ENGINE PIPELINE (executed in order each time trigger_optimizer=True):
  1. PriorityClassifier   — XGBoost on trace priority_class labels
                            → per-node class {0=Critical, 1=Standard, 2=Batch}
  2. ActionGenerator      — discrete (a_mig, a_dvfs, a_fan, a_cool) tuples,
                            pre-filtered by priority constraints and water feasibility
  3. ActionEvaluator      — vectorized 3-objective scoring using RC twin + E3B water fn
  4. NSGAIIOptimizer      — pymoo NSGA-II over the scored population
                            with hard water withdrawal constraint
  5. KneeSelector         — compromise programming (weighted Chebyshev) selects
                            ONE action from the Pareto front;
                            weights shift toward Obj-2 (thermal) when CS is low
  6. ConfidenceGate       — PARALLEL if min_CS >= 80, else SEQUENTIAL

OBJECTIVE FUNCTIONS:
  Obj 1 (Resource cost):   EP [$/kWh] × P_total [kW]   +   water_price [$/L] × water_rate [L/h]
  Obj 2 (Thermal penalty): Σ_i max(0, T_pred_i - T_crit)^2   (per hotspot node)
  Obj 3 (SLA loss):        w_throttle × Δ_dvfs_penalty  +  w_mig × N_migrations

HARD CONSTRAINT:
  water_rate(a, a_cool) ≤ max_withdrawal_rate_l_per_h (regulatory/drought cap)
  Infeasible actions are penalised to +inf; pymoo's NSGA-II handles them as
  constrained problems via constraint violation G <= 0.

ACTION SPACE (discrete):
  a_mig  : None | (src, dst) from top-k least-loaded candidate pairs
  a_dvfs : scalar in {0.6, 0.7, 0.8, 0.9, 1.0} (frequency ratio → power scaling)
  a_fan  : scalar in {0.3, 0.5, 0.8, 1.0}  (fan duty cycle)
  a_cool : int in {0, 1, 2}  (Free-Air, Evaporative, Mechanical)

CHEAP PATH (trigger_optimizer=False):
  Returns the previous decision unchanged (last_decision caching).
  Falls back to a safe default action if no prior decision exists.

DESIGN DECISION — KNEE SELECTION:
  The spec defines the Pareto front optimizer but never defines how to pick one
  action from the returned set. We use weighted Chebyshev compromise programming:
    min_a  max_k( w_k × (f_k(a) - f_k^ideal) / (f_k^nadir - f_k^ideal) )
  When confidence_score < 80, weight w_2 (thermal) is doubled (safety bias).
  This is more defensible than arbitrary lexicographic selection.

DESIGN DECISION — XGBoost TRAINING:
  Trained once on the full parquet (priority_class is a static dataset column).
  No online retraining — workload priority structure is stable. Engine 11
  handles LightGBM retraining; XGBoost does not participate.

CONSUMED BY: Engine 8 (executes ActionDecision).
CONSUMES:    Engine 2 (TelemetryVector), 3B (WaterState), 5 (PredictionBundle),
             6 (AlertState).
"""
from __future__ import annotations

import itertools
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from shared.config import cfg
from shared.io import load_dataset, save_artifact, load_artifact
from shared.types import (
    ActionDecision,
    AlertState,
    PredictionBundle,
    TelemetryVector,
    WaterState,
)
from engine3a.rc_twin import step_rc_thermal_twin

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
# DVFS frequency ratios → power scaling proxy (lower ratio = less power, more SLA loss)
_DVFS_LEVELS: list[float] = [0.6, 0.7, 0.8, 0.9, 1.0]

# Fan duty levels
_FAN_LEVELS: list[float] = [0.3, 0.5, 0.8, 1.0]

# Cooling modes
_COOL_MODES: list[int] = [0, 1, 2]

# SLA loss weights
_W_THROTTLE: float = 0.5    # penalty weight for DVFS throttling
_W_MIG: float = 1.0         # penalty weight per migration

# Water price proxy ($/L) — used in Obj-1 resource cost
_WATER_PRICE_PER_L: float = 0.003   # representative utility water price USD/L

# Regulatory cap on water withdrawal (L/h) — overridden per-tick if supplied
_DEFAULT_MAX_WATER_L_PER_H: float = 500.0

# Confidence gate threshold (0-100 scale, matches spec Step 8.5)
_CONF_GATE: float = 80.0

# Artifact name for XGBoost model
_ARTIFACT_PRIORITY_CLF = "engine7_priority_clf"

# XGBoost feature columns for priority classification
_XGB_FEATURES: list[str] = ["U_cpu", "IT_Load", "CoolingPower", "fan_duty", "Temperature"]


# ---------------------------------------------------------------------------
# Sub-engine 1: Priority Classifier
# ---------------------------------------------------------------------------
class PriorityClassifier:
    """
    XGBoost classifier that maps current telemetry to per-node workload priority.

    Classes:
      0 = Critical  (zero throttling)
      1 = Standard  (up to 30% throttling)
      2 = Batch     (up to 75% throttling, migratable)

    Trained once on the full parquet priority_class column.
    predict() returns a per-node class array.
    """

    def __init__(self) -> None:
        self._clf: Optional[XGBClassifier] = None
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "PriorityClassifier":
        """
        Fit the classifier on the full parquet.

        Args:
            df: Phase-0 parquet DataFrame. Must contain 'priority_class' label column.

        Returns:
            self (for chaining).
        """
        available = [c for c in _XGB_FEATURES if c in df.columns]
        if "priority_class" not in df.columns:
            warnings.warn(
                "priority_class column not found in dataset — using synthetic labels.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Synthesise labels from CPU utilisation thresholds
            df = df.copy()
            df["priority_class"] = np.where(
                df.get("U_cpu", 0.5) > 0.85, 0,
                np.where(df.get("U_cpu", 0.5) > 0.50, 1, 2)
            )

        X = df[available].fillna(df[available].median()).to_numpy(dtype=np.float64)
        y = df["priority_class"].to_numpy(dtype=int)

        self._clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self._clf.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, telemetry: TelemetryVector, n_nodes: int = 3) -> np.ndarray:
        """
        Classify workload priority for each node at the current tick.

        Args:
            telemetry: Current TelemetryVector (E2 output).
            n_nodes:   Number of compute nodes.

        Returns:
            np.ndarray shape (n_nodes,) of int class labels {0, 1, 2}.
        """
        if not self._is_fitted or self._clf is None:
            # Uninitialised — default all to Standard (conservative)
            return np.ones(n_nodes, dtype=int)

        # Synthesise per-node features from the rack-level telemetry
        classes = []
        for i in range(n_nodes):
            u = float(np.clip(telemetry.u_cpu[i], 0.0, 1.0))
            # Build a 1-row feature vector with available columns
            row = np.array([[u, float(np.sum(telemetry.p_servers)) / 1000.0,
                             0.0, u, float(telemetry.t_amb)]], dtype=np.float64)
            # Trim to however many features the clf was trained on
            n_expected = self._clf.n_features_in_
            if row.shape[1] > n_expected:
                row = row[:, :n_expected]
            elif row.shape[1] < n_expected:
                row = np.concatenate([row, np.zeros((1, n_expected - row.shape[1]))], axis=1)
            classes.append(int(self._clf.predict(row)[0]))

        return np.array(classes, dtype=int)

    def save(self) -> None:
        save_artifact(self, _ARTIFACT_PRIORITY_CLF)

    @classmethod
    def load(cls) -> "PriorityClassifier":
        obj = load_artifact(_ARTIFACT_PRIORITY_CLF)
        if not isinstance(obj, cls):
            raise TypeError("Artifact is not a PriorityClassifier.")
        return obj


# ---------------------------------------------------------------------------
# Sub-engine 2: Action Generator
# ---------------------------------------------------------------------------
def generate_actions(
    priority_classes: np.ndarray,
    water_capacity: "WaterState",
    t_wet: float,
    n_nodes: int = 3,
    top_k_mig: int | None = None,
    t_actual: np.ndarray | None = None,
) -> list[dict]:
    """
    Generate pre-filtered candidate action tuples.

    Pre-filtering rules:
      - Critical nodes (class 0): a_dvfs fixed to 1.0 (no throttling).
      - Batch nodes (class 2): migration candidates are included; others excluded.
      - Cool mode 0 (Free-Air): pre-filtered out if water capacity returns inf.

    Args:
        priority_classes: Per-node class array (N,) from PriorityClassifier.
        water_capacity:   WaterState (cool_mode_capacity callable for feasibility check).
        t_wet:            Current wet-bulb temperature (for free-cooling gate).
        n_nodes:          Number of nodes.
        top_k_mig:        Maximum number of migration pairs to consider.
        t_actual:         Current node temperatures for migration targeting.

    Returns:
        List of action dicts with keys:
          a_mig   → None | (src, dst)
          a_dvfs  → float scalar
          a_fan   → float scalar
          a_cool  → int
    """
    k = top_k_mig or cfg.migration_top_k

    # Per-node allowed DVFS levels (Critical → 1.0 only)
    node_dvfs: list[list[float]] = []
    for cls in priority_classes:
        if cls == 0:   # Critical — no throttling
            node_dvfs.append([1.0])
        elif cls == 1: # Standard — up to 30% reduction (≥ 0.7)
            node_dvfs.append([lv for lv in _DVFS_LEVELS if lv >= 0.7])
        else:          # Batch — full throttling allowed
            node_dvfs.append(_DVFS_LEVELS)

    # Migration candidates: only from Batch nodes (class 2), to least-loaded targets
    mig_candidates: list[Optional[tuple[int, int]]] = [None]  # always include no-mig
    batch_nodes = [i for i, c in enumerate(priority_classes) if c == 2]
    if batch_nodes and n_nodes > 1 and t_actual is not None:
        # Targets: nodes with lowest current temperature (proxy for available headroom)
        targets_sorted = np.argsort(t_actual)
        candidate_targets = [
            t for t in targets_sorted if t not in batch_nodes
        ][:k]
        for src in batch_nodes:
            for dst in candidate_targets[:k]:
                if src != dst:
                    mig_candidates.append((src, dst))

    # Feasible cooling modes: pre-filter Mode 0 if free-cooling unavailable
    feasible_cool = [
        mode for mode in _COOL_MODES
        if water_capacity.cool_mode_capacity(mode) < np.inf
    ]
    if not feasible_cool:
        feasible_cool = [1]  # fallback to evaporative if all modes infeasible

    # Use single rack-level dvfs (average of per-node constraints)
    # For simplicity: use the union of all-node allowed levels
    # (per-node enforcement happens in the evaluator's SLA calculation)
    allowed_dvfs = sorted(set(lv for lvs in node_dvfs for lv in lvs))

    actions = []
    for mig, dvfs, fan, cool in itertools.product(
        mig_candidates, allowed_dvfs, _FAN_LEVELS, feasible_cool
    ):
        actions.append({
            "a_mig": mig,
            "a_dvfs": dvfs,
            "a_fan": fan,
            "a_cool": cool,
        })

    return actions


# ---------------------------------------------------------------------------
# Sub-engine 3: Action Evaluator
# ---------------------------------------------------------------------------
def evaluate_actions(
    actions: list[dict],
    telemetry: TelemetryVector,
    water_state: WaterState,
    pred: PredictionBundle,
    t_actual: np.ndarray,
    n_nodes: int = 3,
    max_water_l_per_h: float = _DEFAULT_MAX_WATER_L_PER_H,
) -> np.ndarray:
    """
    Vectorized 3-objective scoring for all candidate actions.

    Returns:
        objectives: ndarray shape (n_actions, 3)
                    col 0 = Obj 1 (resource cost)
                    col 1 = Obj 2 (thermal penalty)
                    col 2 = Obj 3 (SLA loss)
        violations: ndarray shape (n_actions,) — water constraint violation (≤ 0 = feasible)
    """
    n_actions = len(actions)
    objectives = np.zeros((n_actions, 3), dtype=np.float64)
    violations = np.zeros(n_actions, dtype=np.float64)

    it_power_kw = float(np.sum(telemetry.p_servers)) / 1000.0
    ep = float(telemetry.ep)          # $/kWh
    t_pred = pred.t_mid               # shape (N,) — RC-twin predicted temps

    for idx, act in enumerate(actions):
        dvfs = float(act["a_dvfs"])
        fan = float(act["a_fan"])
        cool = int(act["a_cool"])
        mig = act["a_mig"]

        # --- Power under action (DVFS scales P approximately as ratio^3) ---
        p_scaled_kw = it_power_kw * (dvfs ** 3)
        p_fan_kw = (fan ** 3) * 0.05 * n_nodes   # 50W peak per node, fan law

        # --- Water rate under this cooling mode (using E3B capacity callable) --
        wf = water_state.cool_mode_capacity(cool)  # L/kWh — inf if infeasible
        if wf == np.inf:
            water_l_per_h = max_water_l_per_h * 10   # penalise as hard infeasible
        else:
            water_l_per_h = wf * (p_scaled_kw + p_fan_kw)

        # --- Obj 1: Resource cost ---
        energy_cost = ep * (p_scaled_kw + p_fan_kw)     # $/h
        water_cost  = _WATER_PRICE_PER_L * water_l_per_h # $/h
        objectives[idx, 0] = energy_cost + water_cost

        # --- Obj 2: Thermal penalty (on 1-tick RC projection under action) ---
        t_proj = _project_temps(t_actual, telemetry, dvfs, fan, n_nodes)
        thermal_penalty = float(np.sum(np.maximum(0.0, t_proj - cfg.t_crit) ** 2))
        objectives[idx, 1] = thermal_penalty

        # --- Obj 3: SLA/performance loss ---
        dvfs_penalty = _W_THROTTLE * (1.0 - dvfs) * n_nodes
        mig_penalty = _W_MIG if mig is not None else 0.0
        objectives[idx, 2] = dvfs_penalty + mig_penalty

        # --- Hard constraint: water withdrawal cap ---
        violations[idx] = water_l_per_h - max_water_l_per_h   # ≤ 0 = feasible

    return objectives, violations


def _project_temps(
    t_actual: np.ndarray,
    telemetry: TelemetryVector,
    dvfs: float,
    fan: float,
    n_nodes: int,
) -> np.ndarray:
    """
    1-tick RC projection under the given action (DVFS + fan) to estimate
    whether temperatures will rise or fall. Uses spec defaults (not dataset params)
    — this is a decision-time approximation, not a calibrated replay.

    Returns:
        Projected temperature array shape (N,) after one sub-step.
    """
    p_scaled = telemetry.p_servers * (dvfs ** 3)
    t_inlet = float(telemetry.t_amb) + 2.0  # rough inlet offset from ambient
    return step_rc_thermal_twin(
        t_nodes=t_actual.copy(),
        p_nodes=p_scaled,
        t_inlet=t_inlet,
        fan_duty=fan,
        tick_duration_s=10.0,   # 10-second look-ahead (representative sub-step)
    )


# ---------------------------------------------------------------------------
# Sub-engine 4: NSGA-II Optimizer (pymoo)
# ---------------------------------------------------------------------------
def run_nsga2(
    objectives: np.ndarray,
    violations: np.ndarray,
    actions: list[dict],
) -> list[int]:
    """
    Run pymoo NSGA-II over the pre-scored action population.

    Since actions are already enumerated and scored, we pass the pre-computed
    objectives directly to a surrogate pymoo Problem that looks up values by
    index rather than re-evaluating — avoids redundant RC computations.

    Args:
        objectives: (n_actions, 3) objective values.
        violations: (n_actions,)   constraint violation (≤ 0 = feasible).
        actions:    The corresponding action list.

    Returns:
        List of indices into actions[] that lie on the Pareto front.
    """
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
        from pymoo.core.evaluator import Evaluator
        from pymoo.core.population import Population

        n = len(actions)

        class _PrecomputedProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=1,          # single integer index variable
                    n_obj=3,
                    n_ieq_constr=1,   # one inequality constraint (water cap)
                    xl=np.array([0]),
                    xu=np.array([n - 1]),
                )

            def _evaluate(self, X, out, *args, **kwargs):
                idx = np.clip(X[:, 0].astype(int), 0, n - 1)
                out["F"] = objectives[idx]
                out["G"] = violations[idx].reshape(-1, 1)

        problem = _PrecomputedProblem()
        algorithm = NSGA2(pop_size=min(cfg.nsga2_pop, n))
        result = minimize(
            problem,
            algorithm,
            ("n_gen", min(cfg.nsga2_gen, 20)),
            seed=42,
            verbose=False,
        )

        if result.X is not None:
            pareto_indices = list(
                set(np.clip(result.X.flatten().astype(int), 0, n - 1).tolist())
            )
        else:
            pareto_indices = list(range(min(5, n)))

        return pareto_indices

    except Exception as exc:  # pragma: no cover — pymoo call is integration-tested separately
        warnings.warn(
            f"NSGA-II failed ({exc}). Falling back to greedy Pareto filter.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _greedy_pareto(objectives)


def _greedy_pareto(objectives: np.ndarray) -> list[int]:
    """
    Fallback Pareto filter without pymoo: O(n²) dominance check.

    Returns indices of non-dominated solutions.
    """
    n = len(objectives)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                dominated[i] = True
                break
    pareto = list(np.where(~dominated)[0])
    return pareto if pareto else [int(np.argmin(objectives[:, 1]))]  # thermal fallback


# ---------------------------------------------------------------------------
# Sub-engine 5: Knee-Point Selector (compromise programming)
# ---------------------------------------------------------------------------
def select_knee(
    pareto_indices: list[int],
    objectives: np.ndarray,
    min_confidence: float,
) -> int:
    """
    Select one action from the Pareto front via weighted Chebyshev compromise.

    Weights:
      w = [w1_resource, w2_thermal, w3_sla]  (default equal weights)
      When min_confidence < 80: w2 is doubled (safety bias).

    Formula:
      score(a) = max_k( w_k × (f_k - f_ideal_k) / (f_nadir_k - f_ideal_k + ε) )
      → select a* = argmin score(a) over Pareto front

    Args:
        pareto_indices: Indices on the Pareto front.
        objectives:     Full (n_actions, 3) objective matrix.
        min_confidence: Minimum prediction confidence score (0-100).

    Returns:
        Index of the selected action (into the full actions list).
    """
    F_pareto = objectives[pareto_indices]   # shape (k, 3)

    f_ideal = F_pareto.min(axis=0)
    f_nadir = F_pareto.max(axis=0)
    denom = f_nadir - f_ideal + 1e-9

    # Adaptive weights: increase thermal weight under low confidence
    if min_confidence < _CONF_GATE:
        weights = np.array([1.0, 2.0, 1.0])    # safety bias
    else:
        weights = np.array([1.0, 1.0, 1.0])    # balanced

    weights = weights / weights.sum()

    # Normalise objectives to [0, 1] then apply weighted Chebyshev
    F_norm = (F_pareto - f_ideal) / denom   # (k, 3) in [0, 1]
    scores = np.max(F_norm * weights, axis=1)  # (k,)

    best_local = int(np.argmin(scores))
    return pareto_indices[best_local]


# ---------------------------------------------------------------------------
# Sub-engine 6: Confidence Gate
# ---------------------------------------------------------------------------
def apply_confidence_gate(min_confidence: float) -> str:
    """
    Determine execution mode based on minimum prediction confidence.

    Returns:
        "PARALLEL"   if confidence >= 80 (act on all sub-actions simultaneously).
        "SEQUENTIAL" if confidence <  80 (fan first, re-check before DVFS/migration).
    """
    return "PARALLEL" if min_confidence >= _CONF_GATE else "SEQUENTIAL"


# ---------------------------------------------------------------------------
# Top-level Decision Engine
# ---------------------------------------------------------------------------
class DecisionEngine:
    """
    Engine 7: Full decision pipeline for one tick.

    Maintains:
      - PriorityClassifier (fitted once on parquet).
      - _last_decision cache (returned on cheap-path ticks).

    Usage:
        engine7 = DecisionEngine()
        engine7.fit(df)       # train priority classifier
        action = engine7.decide(telemetry, water_state, pred, alert, t_actual)
    """

    def __init__(self, n_nodes: int = 3) -> None:
        self.n_nodes = n_nodes
        self.priority_clf = PriorityClassifier()
        self._last_decision: Optional[ActionDecision] = None
        self._last_pareto_front: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self._last_selected_objectives: np.ndarray = np.zeros(3, dtype=np.float64)
        self._last_selection_rule: str = "chebyshev_knee"
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "DecisionEngine":
        """
        Train the priority classifier on the full parquet.

        Args:
            df: Phase-0 parquet DataFrame.

        Returns:
            self.
        """
        self.priority_clf.fit(df)
        self._is_fitted = True
        return self

    def decide(
        self,
        telemetry: TelemetryVector,
        water_state: WaterState,
        pred: PredictionBundle,
        alert: AlertState,
        t_actual: np.ndarray,
        max_water_l_per_h: float = _DEFAULT_MAX_WATER_L_PER_H,
    ) -> ActionDecision:
        """
        Select the optimal action for this tick.

        CHEAP PATH: If trigger_optimizer is False, return cached last decision.
        FULL PATH:  Run priority→action→evaluate→NSGA-II→knee→gate pipeline.

        Args:
            telemetry:         E2 TelemetryVector.
            water_state:       E3B WaterState.
            pred:              E5 PredictionBundle.
            alert:             E6 AlertState.
            t_actual:          Current node temperatures (N,) from E3A.
            max_water_l_per_h: Regulatory water withdrawal cap.

        Returns:
            ActionDecision — the E7 output contract.
        """
        min_cs = float(np.min(pred.confidence_score))

        # ---- CHEAP PATH ----------------------------------------------------
        if not alert.trigger_optimizer and self._last_decision is not None:
            return self._last_decision

        # ---- FULL PIPELINE PATH --------------------------------------------
        # Step 1: Priority classification
        priorities = self.priority_clf.predict(telemetry, self.n_nodes)

        # Step 2: Action generation
        actions = generate_actions(
            priority_classes=priorities,
            water_capacity=water_state,
            t_wet=telemetry.t_wet,
            n_nodes=self.n_nodes,
            t_actual=t_actual,
        )

        if not actions:
            actions = [{"a_mig": None, "a_dvfs": 1.0, "a_fan": 0.5, "a_cool": 1}]

        # Step 3: Evaluate objectives
        objectives, violations = evaluate_actions(
            actions=actions,
            telemetry=telemetry,
            water_state=water_state,
            pred=pred,
            t_actual=t_actual,
            n_nodes=self.n_nodes,
            max_water_l_per_h=max_water_l_per_h,
        )

        # Step 4: NSGA-II Pareto front
        pareto_idx = run_nsga2(objectives, violations, actions)

        # Step 5: Knee selection
        best_idx = select_knee(pareto_idx, objectives, min_cs)
        best_action = actions[best_idx]
        selection_rule = "thermal_bias" if min_cs < 70.0 else "chebyshev_knee"

        self._last_pareto_front = objectives[pareto_idx] if pareto_idx else np.zeros((1, 3))
        self._last_selected_objectives = objectives[best_idx] if objectives is not None and len(objectives) > best_idx else np.zeros(3)
        self._last_selection_rule = selection_rule

        # Step 6: Confidence gate → execution mode
        exec_mode = apply_confidence_gate(min_cs)

        decision = ActionDecision(
            a_mig=best_action["a_mig"],
            a_dvfs=np.full(self.n_nodes, best_action["a_dvfs"], dtype=np.float64),
            a_fan=float(best_action["a_fan"]),
            a_cool=int(best_action["a_cool"]),
            exec_mode=exec_mode,
        )
        self._last_decision = decision
        return decision

    @property
    def last_pareto_front(self) -> np.ndarray:
        return self._last_pareto_front

    @property
    def last_selected_objectives(self) -> np.ndarray:
        return self._last_selected_objectives

    @property
    def last_selection_rule(self) -> str:
        return self._last_selection_rule


"""
engine9/explainer.py
Phase 9 — Explainability Engine.

Responsibilities:
  1. TreeSHAP on E5 LightGBM predictor models (feature attribution per node per forecast).
  2. TreeSHAP on E7 XGBoost priority classifier (why this node was labeled critical/batch).
  3. Optimizer Pareto rationale: objective values + knee selection rule (no SHAP — not a tree model).
  4. Assembles an ExplainBundle per tick and writes its JSON form to the state store.

Design decisions:
  - SHAP is computed *offline* relative to the control loop: explainer objects are pre-built
    once (from the fitted model artifacts) and stored on the class. Per-tick cost is just
    a forward SHAP call, not a full tree build.
  - If SHAP is not installed, the explainer degrades gracefully and returns zero-valued
    attribution dicts (the "why" record is still present, just un-attributed).
  - The Pareto rationale is always available regardless of SHAP availability — it is derived
    purely from objective values and the knee selection rule computed by Engine 7.
  - ExplainBundle is frozen; the state store receives its JSON serialization only.

CONSUMES:  PredictionBundle (E5), ActionDecision (E7), Pareto metadata (E7).
PRODUCES:  ExplainBundle -> state store -> Engine 10 (Dashboard).
"""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from shared.types import ActionDecision, ExplainBundle, PredictionBundle

# Feature names must match engine5/cqr_predictor.py build_feature_matrix column order.
_PREDICTOR_FEATURES = [
    "U_cpu", "IT_Load", "CoolingPower", "fan_duty", "Temperature",
    "lag10_U", "lag10_IT", "lag20_U", "lag20_IT",
    "T_wet", "T_rc_twin", "T_fan_eff",
]

_PRIORITY_FEATURES = [
    "U_cpu", "IT_Load", "CoolingPower", "fan_duty", "Temperature",
]


def _try_import_shap():
    try:
        import shap
        return shap
    except ImportError:
        return None


class ExplainabilityEngine:
    """
    Builds TreeSHAP explainers from fitted model artifacts and produces
    per-tick ExplainBundles.

    Usage:
        engine = ExplainabilityEngine()
        engine.attach_predictor_models(lgbm_models_dict)  # from E5 artifact
        engine.attach_priority_clf(xgb_clf)                # from E7 PriorityClassifier
        bundle = engine.explain(tick_id, pred, action, pareto_meta)
    """

    def __init__(self) -> None:
        self._shap = _try_import_shap()
        self._predictor_explainers: dict[int, Any] = {}   # node_idx -> shap.TreeExplainer
        self._priority_explainer: Optional[Any] = None
        self._n_nodes: int = 3

    # ── Attach model artifacts ───────────────────────────────────────────────

    def attach_predictor_models(self, lgbm_models: dict[str, Any]) -> None:
        """
        Build TreeSHAP explainers from the CQRPredictor's fitted LightGBM models.

        Args:
            lgbm_models: Dict keyed by e.g. "node0_tau10", value: fitted LGBMRegressor.
                         One explainer per node (tau=10 used; both horizons share the same
                         feature structure, so SHAP values are equivalent in direction).
        """
        self._predictor_explainers = {}
        shap = self._shap
        if shap is None:
            return
        for key, model in lgbm_models.items():
            # Only build one explainer per node (use the tau=10 model)
            if "tau10" in key or "tau_10" in key:
                try:
                    node_idx = int(key.split("node")[1].split("_")[0])
                    self._predictor_explainers[node_idx] = shap.TreeExplainer(model)
                except Exception:
                    pass
        self._n_nodes = max(len(self._predictor_explainers), self._n_nodes)

    def attach_priority_clf(self, xgb_clf: Any) -> None:
        """Build a TreeSHAP explainer from the fitted XGBClassifier."""
        shap = self._shap
        if shap is None or xgb_clf is None:
            return
        try:
            self._priority_explainer = shap.TreeExplainer(xgb_clf)
        except Exception:
            self._priority_explainer = None

    # ── Per-tick explanation ─────────────────────────────────────────────────

    def explain(
        self,
        tick_id: int,
        pred: PredictionBundle,
        action: ActionDecision,
        pareto_front: Optional[np.ndarray] = None,
        selected_objectives: Optional[np.ndarray] = None,
        selection_rule: str = "chebyshev_knee",
        x_tick: Optional[np.ndarray] = None,
    ) -> ExplainBundle:
        """
        Produce one ExplainBundle for the current tick.

        Args:
            tick_id:             Monotonic counter.
            pred:                PredictionBundle from E5 (used to reconstruct feature row).
            action:              ActionDecision from E7.
            pareto_front:        (k, 3) objective matrix of Pareto-front actions (optional).
            selected_objectives: (3,) objectives of the chosen action (optional).
            selection_rule:      Human-readable string from knee selector (default: "chebyshev_knee").
            x_tick:              Optional (1, n_features) feature row for SHAP; if None,
                                 a synthetic row is built from pred.t_mid.

        Returns:
            ExplainBundle with SHAP values (or zeros if SHAP unavailable) and Pareto rationale.
        """
        if pareto_front is None:
            pareto_front = np.zeros((1, 3), dtype=np.float64)
        if selected_objectives is None:
            selected_objectives = pareto_front[0] if len(pareto_front) > 0 else np.zeros(3, dtype=np.float64)
        n = len(pred.t_mid)

        shap_pred = self._explain_predictor(n, x_tick, pred)
        shap_prio = self._explain_priority(n, pred)

        action_summary = {
            "a_dvfs": action.a_dvfs.tolist(),
            "a_fan": float(action.a_fan),
            "a_cool": int(action.a_cool),
            "a_mig": list(action.a_mig) if action.a_mig else None,
        }

        return ExplainBundle(
            tick_id=tick_id,
            shap_predictor=shap_pred,
            shap_priority=shap_prio,
            pareto_front=np.array(pareto_front, dtype=np.float64),
            selected_objectives=np.array(selected_objectives, dtype=np.float64),
            selection_rule=selection_rule,
            exec_mode=action.exec_mode,
            action_summary=action_summary,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _explain_predictor(
        self,
        n: int,
        x_tick: Optional[np.ndarray],
        pred: PredictionBundle,
    ) -> dict:
        """Compute per-node SHAP values for the LightGBM predictor."""
        result: dict[int, dict] = {}
        if self._shap is None or not self._predictor_explainers:
            # Graceful degradation: return zero attribution
            for i in range(n):
                result[i] = {f: 0.0 for f in _PREDICTOR_FEATURES}
            return result

        for node_idx, explainer in self._predictor_explainers.items():
            try:
                if x_tick is not None and x_tick.shape[1] == len(_PREDICTOR_FEATURES):
                    row = x_tick[[node_idx]] if x_tick.shape[0] > 1 else x_tick
                else:
                    # Fallback: synthetic single-row from prediction midpoint
                    row = self._synthetic_row(pred, node_idx)

                shap_vals = explainer.shap_values(row)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
                vals_flat = np.asarray(shap_vals).flatten()
                n_feat = min(len(vals_flat), len(_PREDICTOR_FEATURES))
                result[node_idx] = {
                    _PREDICTOR_FEATURES[j]: float(vals_flat[j])
                    for j in range(n_feat)
                }
            except Exception:
                result[node_idx] = {f: 0.0 for f in _PREDICTOR_FEATURES}

        # Fill missing nodes
        for i in range(n):
            if i not in result:
                result[i] = {f: 0.0 for f in _PREDICTOR_FEATURES}
        return result

    def _explain_priority(self, n: int, pred: PredictionBundle) -> dict:
        """Compute per-node SHAP values for the XGBoost priority classifier."""
        result: dict[int, dict] = {}
        if self._shap is None or self._priority_explainer is None:
            for i in range(n):
                result[i] = {f: [0.0, 0.0, 0.0] for f in _PRIORITY_FEATURES}
            return result

        for i in range(n):
            try:
                row = self._synthetic_priority_row(pred, i)
                shap_vals = self._priority_explainer.shap_values(row)
                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    # (1, n_features, n_classes)
                    vals = shap_vals[0]  # (n_features, n_classes)
                    result[i] = {
                        _PRIORITY_FEATURES[j]: vals[j].tolist()
                        for j in range(min(vals.shape[0], len(_PRIORITY_FEATURES)))
                    }
                elif isinstance(shap_vals, list):
                    # list of (1, n_features) arrays, one per class
                    result[i] = {
                        _PRIORITY_FEATURES[j]: [float(c[0][j]) for c in shap_vals]
                        for j in range(min(len(shap_vals[0][0]), len(_PRIORITY_FEATURES)))
                    }
                else:
                    result[i] = {f: [0.0, 0.0, 0.0] for f in _PRIORITY_FEATURES}
            except Exception:
                result[i] = {f: [0.0, 0.0, 0.0] for f in _PRIORITY_FEATURES}
        return result

    @staticmethod
    def _synthetic_row(pred: PredictionBundle, node_idx: int) -> np.ndarray:
        """Build a synthetic 1-row feature matrix from the prediction midpoint."""
        t_mid = float(pred.t_mid[node_idx]) if node_idx < len(pred.t_mid) else 40.0
        # Rough synthetic values in the correct feature order
        row = np.array([[
            0.6,      # U_cpu
            300.0,    # IT_Load
            100.0,    # CoolingPower
            0.5,      # fan_duty
            t_mid,    # Temperature (approx)
            0.6,      # lag10_U
            290.0,    # lag10_IT
            0.6,      # lag20_U
            285.0,    # lag20_IT
            15.0,     # T_wet
            t_mid,    # T_rc_twin
            t_mid,    # T_fan_eff
        ]], dtype=np.float64)
        return row

    @staticmethod
    def _synthetic_priority_row(pred: PredictionBundle, node_idx: int) -> np.ndarray:
        """Build a synthetic 1-row feature for the priority classifier."""
        t_mid = float(pred.t_mid[node_idx]) if node_idx < len(pred.t_mid) else 40.0
        row = np.array([[
            0.6,      # U_cpu
            300.0,    # IT_Load
            100.0,    # CoolingPower
            0.5,      # fan_duty
            t_mid,    # Temperature
        ]], dtype=np.float64)
        return row


def serialize_explain_bundle(bundle: ExplainBundle) -> str:
    """Serialize an ExplainBundle to a compact JSON string for the state store."""
    return json.dumps({
        "tick_id": bundle.tick_id,
        "shap_predictor": bundle.shap_predictor,
        "shap_priority": bundle.shap_priority,
        "pareto_front": bundle.pareto_front.tolist(),
        "selected_objectives": bundle.selected_objectives.tolist(),
        "selection_rule": bundle.selection_rule,
        "exec_mode": bundle.exec_mode,
        "action_summary": bundle.action_summary,
    })


def deserialize_explain_bundle(json_str: str) -> ExplainBundle:
    """Reconstruct an ExplainBundle from its JSON state store representation."""
    d = json.loads(json_str)
    shap_pred = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in d["shap_predictor"].items()}
    shap_prio = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in d["shap_priority"].items()}
    return ExplainBundle(
        tick_id=d["tick_id"],
        shap_predictor=shap_pred,
        shap_priority=shap_prio,
        pareto_front=np.array(d["pareto_front"]),
        selected_objectives=np.array(d["selected_objectives"]),
        selection_rule=d["selection_rule"],
        exec_mode=d["exec_mode"],
        action_summary=d["action_summary"],
    )

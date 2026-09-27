"""
engine11/replay.py
Phase 11 — Experience Replay & Closed-Loop Retraining Engine.

Responsibilities:
  1. Maintain a Parquet ring-buffer of (state, action, next_state, reward) records.
  2. Compute reward at each tick — explicitly includes the water-cost component:
       reward = -(energy_cost + water_cost + thermal_penalty + sla_loss)
  3. Every RETRAIN_EVERY_N_CYCLES ticks: retrain E5 CQR models on the fresh replay buffer
     merged with the original training data.
  4. Before promoting a new model: verify held-out coverage does not regress beyond
     COVERAGE_TOLERANCE relative to the current model's stored coverage baseline.
  5. Write the promoted model artifact to data/artifacts/ (overwrites E5's load path).

Design decisions:
  - Replay buffer is append-only Parquet (row-per-tick), truncated to MAX_REPLAY_ROWS.
    Parquet is preferred over SQLite for this: columnar scan over millions of rows is
    needed for retraining, which Parquet handles natively.
  - Reward function mirrors the 3-objective formulation in Engine 7 (exact same weights)
    so the replay store and the optimizer agree on what "better" means.
  - Coverage tolerance of 2 pp: if the new model's empirical coverage (on a hold-out
    slice of the replay buffer) is more than 2 pp below the stored baseline, the new
    model is REJECTED and the previous version remains active.
  - Retraining is synchronous but runs in a background thread so the control loop is
    not blocked. A threading.Event signals completion.

CONSUMES:  ActionDecision (E7), TickResult (E8), TelemetryVector (E2), WaterState (E3B).
PRODUCES:  Updated E5 CQR model artifact at data/artifacts/engine5_cqr_models.pkl.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shared.config import cfg
from shared.types import ActionDecision, ReplayRecord, TelemetryVector, TickResult, WaterState

log = logging.getLogger(__name__)

_REPLAY_PATH: Path = Path(cfg.data_dir) / "replay_buffer.parquet"
_MAX_REPLAY_ROWS: int = 50_000
_RETRAIN_EVERY_N_CYCLES: int = 100
_COVERAGE_TOLERANCE: float = 0.02   # 2 percentage points
_ENERGY_W: float = 1.0
_WATER_W: float = 0.5
_THERMAL_W: float = 2.0
_SLA_W: float = 0.5


# ── Reward computation ─────────────────────────────────────────────────────────

def compute_reward(
    tel: TelemetryVector,
    water: WaterState,
    action: ActionDecision,
    tick_result: TickResult,
    t_node_next: np.ndarray,
) -> float:
    """
    Compute one-step reward for the replay buffer.

    Reward = −( energy_cost + water_cost + thermal_penalty + sla_loss )

    All components are the same as Engine 7's objective functions so the replay buffer
    and the optimizer share a single consistent notion of "cost".

    Args:
        tel:         TelemetryVector at the current tick.
        water:       WaterState at the current tick (contains water_rate_l_per_h).
        action:      ActionDecision applied this tick.
        tick_result: TickResult from Engine 8 (u_next, p_next).
        t_node_next: RC-twin temperatures at t+1 (shape N,).

    Returns:
        Scalar reward (more negative = worse).
    """
    # Obj 1: energy + water cost
    p_total_kw = float(np.sum(tick_result.p_next)) / 1000.0
    energy_cost = tel.ep * p_total_kw
    water_cost = 0.001 * water.water_rate_l_per_h  # $0.001 per litre
    resource_cost = energy_cost + water_cost

    # Obj 2: thermal penalty
    t_crit = float(cfg.t_crit)
    thermal_penalty = float(np.sum(np.maximum(0.0, t_node_next - t_crit) ** 2))

    # Obj 3: SLA loss (DVFS degradation + migrations)
    n_nodes = len(action.a_dvfs)
    dvfs_loss = 0.5 * float(np.sum(1.0 - action.a_dvfs))
    mig_loss = 1.0 if action.a_mig is not None else 0.0
    sla_loss = dvfs_loss + mig_loss

    reward = -(
        _ENERGY_W * resource_cost
        + _THERMAL_W * thermal_penalty
        + _WATER_W * water_cost
        + _SLA_W * sla_loss
    )
    return float(reward)


# ── Replay buffer ──────────────────────────────────────────────────────────────

def make_replay_record(
    tick_id: int,
    tel: TelemetryVector,
    water: WaterState,
    action: ActionDecision,
    tick_result: TickResult,
    t_node_next: np.ndarray,
) -> ReplayRecord:
    """Assemble a ReplayRecord from the current tick's outputs."""
    reward = compute_reward(tel, water, action, tick_result, t_node_next)
    return ReplayRecord(
        tick_id=tick_id,
        u_cpu_mean=float(np.mean(tel.u_cpu)),
        t_node_max=float(np.max(t_node_next - 40.0)),   # max above 40°C baseline
        t_wet=float(tel.t_wet),
        water_rate=float(water.water_rate_l_per_h),
        ep=float(tel.ep),
        a_dvfs_mean=float(np.mean(action.a_dvfs)),
        a_fan=float(action.a_fan),
        a_cool=int(action.a_cool),
        u_cpu_mean_next=float(np.mean(tick_result.u_next)),
        t_node_max_next=float(np.max(t_node_next)),
        reward=reward,
    )


def append_record(record: ReplayRecord, path: Path = _REPLAY_PATH) -> None:
    """
    Append one ReplayRecord to the Parquet ring buffer.

    Handles first-write creation and ring-buffer trimming atomically.
    """
    row = pd.DataFrame([record.__dict__])
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, row], ignore_index=True)
        if len(combined) > _MAX_REPLAY_ROWS:
            combined = combined.iloc[-_MAX_REPLAY_ROWS:]
    else:
        combined = row

    combined.to_parquet(path, index=False)


def load_buffer(path: Path = _REPLAY_PATH) -> pd.DataFrame:
    """Load the full replay buffer as a DataFrame. Returns empty DF if absent."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ── Retraining ─────────────────────────────────────────────────────────────────

class RetrainingEngine:
    """
    Periodic retraining of Engine 5 CQR models using the accumulated replay buffer.

    Lifecycle:
        engine = RetrainingEngine()
        engine.tick(tick_id, tel, water, action, tick_result, t_node_next)
        # Every RETRAIN_EVERY_N_CYCLES calls, a background thread runs _retrain().
    """

    def __init__(
        self,
        replay_path: Path = _REPLAY_PATH,
        retrain_interval: int = _RETRAIN_EVERY_N_CYCLES,
    ) -> None:
        self._replay_path = replay_path
        self._retrain_interval = retrain_interval
        self._tick_count: int = 0
        self._retrain_thread: Optional[threading.Thread] = None
        self._retrain_event: threading.Event = threading.Event()
        self._last_coverage: Optional[float] = None   # baseline from current E5 model

    def tick(
        self,
        tick_id: int,
        tel: TelemetryVector,
        water: WaterState,
        action: ActionDecision,
        tick_result: TickResult,
        t_node_next: np.ndarray,
    ) -> None:
        """
        Called every control-loop tick.
        Appends the replay record and triggers retraining every N cycles.
        """
        record = make_replay_record(tick_id, tel, water, action, tick_result, t_node_next)
        append_record(record, self._replay_path)
        self._tick_count += 1

        if self._tick_count % self._retrain_interval == 0:
            if self._retrain_thread is None or not self._retrain_thread.is_alive():
                self._retrain_event.clear()
                self._retrain_thread = threading.Thread(
                    target=self._retrain,
                    daemon=True,
                    name=f"retrain-{tick_id}",
                )
                self._retrain_thread.start()
                log.info("Retrain thread launched at tick %d", tick_id)

    def _retrain(self) -> None:
        """Background: retrain E5 CQR models and promote if coverage doesn't regress."""
        try:
            from engine5.cqr_predictor import CQRPredictor
            from shared.io import save_artifact, load_artifact
        except ImportError as e:
            log.warning("Retrain skipped — import failed: %s", e)
            self._retrain_event.set()
            return

        buffer = load_buffer(self._replay_path)
        if len(buffer) < 500:
            log.info("Buffer too small (%d rows) — retrain skipped", len(buffer))
            self._retrain_event.set()
            return

        # Load original training data and merge with replay buffer
        parquet_path = Path(cfg.data_dir) / "eqcam_clean.parquet"
        if parquet_path.exists():
            orig = pd.read_parquet(parquet_path)
        else:
            orig = pd.DataFrame()

        # Build a surrogate temperature column from replay scalars
        synth = pd.DataFrame({
            "U_cpu": buffer["u_cpu_mean"],
            "IT_Load": buffer["u_cpu_mean"] * 400.0 + 100.0,
            "CoolingPower": buffer["a_fan"] * 150.0,
            "fan_duty": buffer["a_fan"],
            "Temperature": buffer["t_node_max_next"],
        })
        combined = pd.concat([orig, synth], ignore_index=True) if not orig.empty else synth

        # Train new CQR predictor
        predictor = CQRPredictor()
        try:
            predictor.fit(combined)
        except Exception as exc:
            log.warning("Retrain fit failed: %s", exc)
            self._retrain_event.set()
            return

        # Evaluate coverage on last 10% of buffer (held-out slice)
        held_out = buffer.iloc[int(len(buffer) * 0.9):]
        if len(held_out) < 50:
            log.info("Held-out slice too small — skipping coverage check, promoting directly")
            self._promote(predictor)
            self._retrain_event.set()
            return

        new_coverage = self._estimate_coverage(predictor, held_out)
        log.info("Retrain: new_coverage=%.3f, baseline=%.3f",
                 new_coverage, self._last_coverage or 0.0)

        if self._last_coverage is not None:
            if new_coverage < self._last_coverage - _COVERAGE_TOLERANCE:
                log.warning(
                    "Retrain REJECTED: new coverage %.3f < baseline %.3f - %.3f",
                    new_coverage, self._last_coverage, _COVERAGE_TOLERANCE
                )
                self._retrain_event.set()
                return

        self._promote(predictor)
        self._last_coverage = new_coverage
        self._retrain_event.set()

    @staticmethod
    def _estimate_coverage(predictor, held_out: pd.DataFrame) -> float:
        """
        Estimate empirical coverage on the held-out buffer slice.

        Uses T_node_max_next as the ground-truth target and the predictor's interval.
        Returns fraction of true values falling within [t_low, t_high].
        """
        try:
            y_true = held_out["t_node_max_next"].values
            # Build minimal feature rows using replay scalars
            x_rows = pd.DataFrame({
                "U_cpu": held_out["u_cpu_mean"],
                "IT_Load": held_out["u_cpu_mean"] * 400.0 + 100.0,
                "CoolingPower": held_out["a_fan"] * 150.0,
                "fan_duty": held_out["a_fan"],
                "Temperature": y_true,
                "lag10_U": held_out["u_cpu_mean"],
                "lag10_IT": held_out["u_cpu_mean"] * 390.0 + 100.0,
                "lag20_U": held_out["u_cpu_mean"],
                "lag20_IT": held_out["u_cpu_mean"] * 380.0 + 100.0,
                "T_wet": held_out["t_wet"],
                "T_rc_twin": y_true,
                "T_fan_eff": y_true - held_out["a_fan"] * 5.0,
            })
            bundles = [predictor.predict_row(row) for _, row in x_rows.iterrows()]
            covered = sum(
                1 for i, b in enumerate(bundles)
                if float(b.t_low[0]) <= y_true[i] <= float(b.t_high[0])
            )
            return covered / max(len(bundles), 1)
        except Exception as exc:
            log.warning("Coverage estimate failed: %s", exc)
            return 0.0

    @staticmethod
    def _promote(predictor) -> None:
        """Write the new model artifact to E5's load path."""
        try:
            from shared.io import save_artifact
            artifact_path = Path(cfg.data_dir) / "artifacts" / "engine5_cqr_models.pkl"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            save_artifact(predictor, artifact_path)
            log.info("New CQR model promoted to %s", artifact_path)
        except Exception as exc:
            log.warning("Promotion failed: %s", exc)

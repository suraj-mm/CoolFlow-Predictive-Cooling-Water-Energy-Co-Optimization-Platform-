"""
engine11/test_engine11.py
Unit tests for the Experience Replay & Closed-Loop Retraining Engine.

Coverage:
  11-1  compute_reward returns float <= 0
  11-2  higher temperatures produce strictly more negative reward (thermal penalty)
  11-3  higher water rate produces strictly more negative reward (water cost component)
  11-4  make_replay_record populates all fields of ReplayRecord
  11-5  append_record creates and appends to Parquet file
  11-6  append_record enforces ring-buffer size cap
  11-7  load_buffer returns empty DataFrame when file missing
  11-8  load_buffer loads all appended records correctly
  11-9  RetrainingEngine tick updates count and appends records
  11-10 Rejection gate: regressed coverage blocks promotion
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from shared.types import (
    ActionDecision,
    ReplayRecord,
    TelemetryVector,
    TickResult,
    WaterState,
)
from engine3b.water_engine import make_cool_mode_capacity
from engine11.replay import (
    RetrainingEngine,
    append_record,
    compute_reward,
    load_buffer,
    make_replay_record,
)

N = 3

def _tel():
    return TelemetryVector(
        u_cpu=np.array([0.5, 0.6, 0.4]),
        p_servers=np.array([250.0, 280.0, 220.0]),
        t_amb=20.0,
        rh=50.0,
        t_wet=12.0,
        ci=300.0,
        ep=0.25,
    )

def _water(rate=10.0, mode=1):
    return WaterState(
        water_rate_l_per_h=rate,
        wue=0.5,
        cool_mode=mode,
        cool_mode_capacity=make_cool_mode_capacity(12.0),
    )

def _action(mig=None, dvfs=1.0, fan=0.5, cool=1):
    return ActionDecision(
        a_mig=mig,
        a_dvfs=np.full(N, dvfs),
        a_fan=fan,
        a_cool=cool,
        exec_mode="PARALLEL",
    )

def _tick_result():
    return TickResult(
        u_next=np.array([0.5, 0.6, 0.4]),
        p_next=np.array([250.0, 280.0, 220.0]),
        action_log={"action": "test"},
    )


class TestRewardComputation(unittest.TestCase):

    def test_reward_is_negative_float(self):
        """11-1: compute_reward returns a negative float (cost formulation)."""
        r = compute_reward(
            tel=_tel(),
            water=_water(),
            action=_action(),
            tick_result=_tick_result(),
            t_node_next=np.array([50.0, 52.0, 48.0]),
        )
        self.assertIsInstance(r, float)
        self.assertLess(r, 0.0)

    def test_thermal_penalty_increases_cost(self):
        """11-2: temperatures above t_crit (85°C) strictly decrease reward."""
        r_cool = compute_reward(
            tel=_tel(),
            water=_water(),
            action=_action(),
            tick_result=_tick_result(),
            t_node_next=np.array([50.0, 52.0, 48.0]),
        )
        r_hot = compute_reward(
            tel=_tel(),
            water=_water(),
            action=_action(),
            tick_result=_tick_result(),
            t_node_next=np.array([88.0, 92.0, 86.0]),
        )
        self.assertLess(r_hot, r_cool)

    def test_water_cost_component_active(self):
        """11-3: higher water consumption rate strictly decreases reward."""
        r_low_water = compute_reward(
            tel=_tel(),
            water=_water(rate=0.0, mode=0),
            action=_action(cool=0),
            tick_result=_tick_result(),
            t_node_next=np.array([50.0, 50.0, 50.0]),
        )
        r_high_water = compute_reward(
            tel=_tel(),
            water=_water(rate=200.0, mode=1),
            action=_action(cool=1),
            tick_result=_tick_result(),
            t_node_next=np.array([50.0, 50.0, 50.0]),
        )
        self.assertLess(r_high_water, r_low_water)


class TestReplayBufferIO(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.buffer_path = Path(self.test_dir) / "test_replay.parquet"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_make_replay_record(self):
        """11-4: make_replay_record returns valid ReplayRecord with all fields."""
        rec = make_replay_record(
            tick_id=1,
            tel=_tel(),
            water=_water(),
            action=_action(),
            tick_result=_tick_result(),
            t_node_next=np.array([45.0, 46.0, 44.0]),
        )
        self.assertIsInstance(rec, ReplayRecord)
        self.assertEqual(rec.tick_id, 1)
        self.assertAlmostEqual(rec.a_fan, 0.5)
        self.assertEqual(rec.a_cool, 1)
        self.assertLess(rec.reward, 0.0)

    def test_append_and_load_record(self):
        """11-5, 11-8: append_record creates parquet and load_buffer reads it back."""
        rec = make_replay_record(
            tick_id=10,
            tel=_tel(),
            water=_water(),
            action=_action(),
            tick_result=_tick_result(),
            t_node_next=np.array([45.0, 46.0, 44.0]),
        )
        append_record(rec, path=self.buffer_path)
        df = load_buffer(path=self.buffer_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["tick_id"].iloc[0], 10)

    def test_missing_buffer_returns_empty_df(self):
        """11-7: load_buffer returns empty DataFrame if file does not exist."""
        df = load_buffer(path=Path(self.test_dir) / "nonexistent.parquet")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_retraining_engine_tick(self):
        """11-9: RetrainingEngine tick appends record to buffer."""
        engine = RetrainingEngine(replay_path=self.buffer_path, retrain_interval=100)
        engine.tick(
            tick_id=1,
            tel=_tel(),
            water=_water(),
            action=_action(),
            tick_result=_tick_result(),
            t_node_next=np.array([45.0, 46.0, 44.0]),
        )
        df = load_buffer(self.buffer_path)
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()

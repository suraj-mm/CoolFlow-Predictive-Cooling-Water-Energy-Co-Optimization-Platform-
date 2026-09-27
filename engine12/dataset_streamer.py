"""
engine12/dataset_streamer.py
Real-Time Dataset Telemetry Streamer for CoolFlow Dashboard.

Reads sequential records from `data/dataset_stream.json` (extracted from `phase0_unified.parquet`)
and streams them in real-time into the SQLite WAL state store.
Provides interactive controls: play, pause, step, seek, and variable playback speeds.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.state_store import write_state
from shared.types import StateRecord

log = logging.getLogger(__name__)

class DatasetStreamer:
    """Manages real-time replay of dataset telemetry records."""

    def __init__(self, json_path: Optional[Path] = None):
        self.json_path = json_path or Path("data/dataset_stream.json")
        self._records: List[Dict[str, Any]] = []
        self._current_idx: int = 0
        self._is_playing: bool = True
        self._speed: float = 1.0  # 1.0x = 1 tick / second
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._load_records()

    def _load_records(self):
        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as f:
                self._records = json.load(f)
            log.info(f"Loaded {len(self._records)} dataset stream records from {self.json_path}")
        else:
            log.warning(f"Dataset stream file {self.json_path} not found.")

    @property
    def total_records(self) -> int:
        return len(self._records)

    @property
    def current_index(self) -> int:
        return self._current_idx

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def speed(self) -> float:
        return self._speed

    def get_current_record(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._records:
                return None
            return self._records[self._current_idx % len(self._records)]

    def get_stream_records(self, count: int = 4320) -> List[Dict[str, Any]]:
        with self._lock:
            if not self._records:
                return []
            if count <= 0 or count >= len(self._records):
                return list(self._records)
            idx = self._current_idx % len(self._records)
            start = max(0, idx - count + 1)
            return self._records[start : idx + 1]

    def play(self):
        with self._lock:
            self._is_playing = True

    def pause(self):
        with self._lock:
            self._is_playing = False

    def step(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._records:
                return None
            self._current_idx = (self._current_idx + 1) % len(self._records)
            rec = self._records[self._current_idx]
            self._write_to_state_store(rec)
            return rec

    def seek(self, index: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._records:
                return None
            self._current_idx = max(0, min(index, len(self._records) - 1))
            rec = self._records[self._current_idx]
            self._write_to_state_store(rec)
            return rec

    def set_speed(self, speed: float):
        with self._lock:
            self._speed = max(0.2, min(speed, 10.0))

    def _write_to_state_store(self, rec: Dict[str, Any]):
        """Write record into shared state_store SQLite WAL bus."""
        nodes = rec.get("nodes", [])
        t0 = float(nodes[0]["temp"]) if len(nodes) > 0 else 65.0
        t1 = float(nodes[1]["temp"]) if len(nodes) > 1 else 65.0
        t2 = float(nodes[2]["temp"]) if len(nodes) > 2 else 65.0

        fan = float((nodes[0]["fan"] if len(nodes) > 0 else 65) / 100.0)
        peak_t = float(rec.get("peakTemp", max(t0, t1, t2)))

        shap_bundle = {
            "attributions": rec.get("shap", {}),
            "selection_rule": "chebyshev_knee",
            "selected_objectives": [rec.get("power", 320) * 0.0003, max(0.0, (peak_t - 80.0) ** 2), 0.07]
        }

        sr = StateRecord(
            tick_id=int(rec["tick"]),
            timestamp=str(rec.get("timestamp", rec.get("time", "2022-02-03 16:40:00"))),
            temperatures=[t0, t1, t2],
            t_inlet=float(rec.get("ambient", 20.0) + 2.0),
            water_rate_l_per_h=float(rec.get("water", 150.0)),
            wue=float(rec.get("wue", 0.52)),
            cool_mode=int(rec.get("mode", 1)),
            power_total_w=float(rec.get("power", 300.0) * 1000.0),
            a_fan=fan,
            a_cool=int(rec.get("mode", 1)),
            a_dvfs=[0.9, 1.0, 1.0],
            a_mig=None,
            trigger_optimizer=bool(peak_t >= 76.0),
            max_temp=peak_t,
            shap_json=json.dumps(shap_bundle),
        )
        try:
            write_state(sr)
        except Exception as e:
            log.debug(f"State store write error: {e}")

    def start_background_loop(self):
        """Start daemon thread advancing stream ticks."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("DatasetStreamer background thread started.")

    def stop_background_loop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_loop(self):
        while not self._stop_event.is_set():
            delay = 1.0 / max(0.1, self._speed)
            time.sleep(delay)
            if self._is_playing and self._records:
                with self._lock:
                    self._current_idx = (self._current_idx + 1) % len(self._records)
                    rec = self._records[self._current_idx]
                    self._write_to_state_store(rec)

# Singleton global instance
streamer = DatasetStreamer()

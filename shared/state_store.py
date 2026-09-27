"""
shared/state_store.py
SQLite ring-buffer state store — the single shared data bus between the control loop
and the dashboard (Engine 10).

DESIGN:
  - One connection per writer (control loop); readers open read-only connections
    with WAL journal mode so the dashboard never blocks the control loop.
  - Ring buffer: keeps the last MAX_ROWS rows; older rows are deleted in the same
    transaction as the insert (no separate vacuum job needed).
  - Schema is fixed at module import; forward-compatible (new columns → ignored on read).
  - Thread-safe for single-writer / multiple-reader (WAL mode guarantee).

CONSUMED BY: Engine 10 (Dashboard poll), Engine 12 (Integration validation).
WRITTEN BY:  Control loop tick (assembles StateRecord from E3A, E3B, E9 outputs).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from shared.config import cfg

_DB_PATH: Path = Path(cfg.data_dir) / "state_store.db"
_MAX_ROWS: int = 8_640   # 60 days at 10-min cadence


def _connect(path: Path = _DB_PATH, readonly: bool = False) -> sqlite3.Connection:
    abs_path = path.resolve()
    if readonly:
        uri = f"file:{abs_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(abs_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(path: Path = _DB_PATH) -> None:
    """Create the state_log table if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state_log (
            tick_id          INTEGER PRIMARY KEY,
            timestamp        TEXT,
            temp_node0       REAL,
            temp_node1       REAL,
            temp_node2       REAL,
            t_inlet          REAL,
            water_rate_l_per_h REAL,
            wue              REAL,
            cool_mode        INTEGER,
            power_total_w    REAL,
            a_fan            REAL,
            a_cool           INTEGER,
            a_dvfs_json      TEXT,
            a_mig_json       TEXT,
            trigger_optimizer INTEGER,
            max_temp         REAL,
            shap_json        TEXT
        )
    """)
    conn.commit()
    conn.close()


def write_state(record: "StateRecord", path: Path = _DB_PATH) -> None:
    """
    Insert one StateRecord row and trim the ring buffer.

    Args:
        record: StateRecord from the current tick.
        path:   Path to the SQLite database.
    """
    from shared.types import StateRecord  # local import avoids circular at module level
    temps = record.temperatures
    conn = _connect(path)
    conn.execute("""
        INSERT OR REPLACE INTO state_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record.tick_id,
        record.timestamp,
        float(temps[0]) if len(temps) > 0 else None,
        float(temps[1]) if len(temps) > 1 else None,
        float(temps[2]) if len(temps) > 2 else None,
        float(record.t_inlet),
        float(record.water_rate_l_per_h),
        float(record.wue),
        int(record.cool_mode),
        float(record.power_total_w),
        float(record.a_fan),
        int(record.a_cool),
        json.dumps(record.a_dvfs),
        json.dumps(list(record.a_mig) if record.a_mig else None),
        int(record.trigger_optimizer),
        float(record.max_temp),
        record.shap_json,
    ))
    # Ring buffer: delete oldest rows beyond MAX_ROWS
    conn.execute("""
        DELETE FROM state_log
        WHERE tick_id NOT IN (
            SELECT tick_id FROM state_log ORDER BY tick_id DESC LIMIT ?
        )
    """, (_MAX_ROWS,))
    conn.commit()
    conn.close()


def read_recent(n: int = 100, path: Path = _DB_PATH) -> list[dict]:
    """
    Fetch the last n state rows in ascending tick order.

    Returns:
        List of dicts with all column values. Empty list if DB doesn't exist.
    """
    if not path.exists():
        return []
    try:
        conn = _connect(path, readonly=True)
        cursor = conn.execute("""
            SELECT * FROM state_log ORDER BY tick_id DESC LIMIT ?
        """, (n,))
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        conn.close()
        return list(reversed(rows))  # chronological order
    except Exception:
        return []


def get_db_path() -> Path:
    """Return the resolved database path (for dashboard connection string)."""
    return _DB_PATH

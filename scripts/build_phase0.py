"""
scripts/build_phase0.py
Phase 0 dataset assembly — run ONCE; all engines read the output parquet.

Produces: data/phase0_unified.parquet
Columns added beyond raw EQCAM:
  U_cpu, U_gpu        — utilisation [0,1]
  fan_duty            — normalised duty [0.3, 1.0]
  T_wet               — wet-bulb temp (Stull 2011)
  priority_class      — 0=Critical,1=Standard,2=Batch (from RAC column)
  CI                  — grid carbon intensity gCO2/kWh (historical or synthetic)
  EP                  — electricity price $/kWh (TOU synthetic)

Exit criteria: parquet on disk, no NaNs in key columns, timestamp monotonic.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.config import cfg

# ---------------------------------------------------------------------------
# CI fetch — UK Carbon Intensity API (free, no key). 14-day chunks.
# Fallback to synthetic if unreachable.
# ---------------------------------------------------------------------------
_CI_API = "https://api.carbonintensity.org.uk/intensity/{start}/{end}"
_CI_CHUNK_DAYS = 14


def _fetch_ci_historical(timestamps: pd.DatetimeIndex) -> pd.Series:
    """Attempt real CI fetch; fall back to synthetic on any error."""
    start_dt = timestamps.min().tz_localize("UTC")
    end_dt = timestamps.max().tz_localize("UTC")

    chunks, cur = [], start_dt
    while cur < end_dt:
        nxt = min(cur + pd.Timedelta(days=_CI_CHUNK_DAYS), end_dt)
        chunks.append((cur, nxt))
        cur = nxt

    records: list[dict] = []
    print(f"  Fetching CI data in {len(chunks)} chunks from Carbon Intensity API...")
    for i, (s, e) in enumerate(chunks):
        url = _CI_API.format(
            start=s.strftime("%Y-%m-%dT%H:%MZ"),
            end=e.strftime("%Y-%m-%dT%H:%MZ"),
        )
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            for entry in r.json().get("data", []):
                records.append(
                    {
                        "timestamp": pd.to_datetime(entry["from"], utc=True),
                        "CI": float(entry["intensity"]["actual"] or entry["intensity"]["forecast"] or cfg.ci_default),
                    }
                )
            if (i + 1) % 10 == 0:
                print(f"    chunk {i+1}/{len(chunks)} done")
            time.sleep(0.1)   # polite rate-limit
        except Exception as exc:
            print(f"  CI API failed at chunk {i+1}: {exc}. Using synthetic fallback.")
            return _synthetic_ci(timestamps)

    if not records:
        print("  No CI records returned. Using synthetic fallback.")
        return _synthetic_ci(timestamps)

    ci_df = (
        pd.DataFrame(records)
        .drop_duplicates("timestamp")
        .set_index("timestamp")
        .sort_index()
    )
    # Resample from 30-min to 10-min by linear interpolation
    ts_utc = timestamps.tz_localize("UTC")
    ci_resampled = (
        ci_df["CI"]
        .reindex(ci_df.index.union(ts_utc))
        .interpolate("time")
        .reindex(ts_utc)
        .values
    )
    ci_s = pd.Series(ci_resampled, index=timestamps, name="CI")
    # Fill any remaining NaN with synthetic
    mask = ci_s.isna()
    if mask.any():
        ci_s[mask] = _synthetic_ci(timestamps[mask])
    print(f"  CI fetched: {ci_s.mean():.1f} gCO2/kWh avg (real data).")
    return ci_s


def _synthetic_ci(timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    Seasonal + diurnal Fourier signal around 350 gCO2/kWh.
    Annual: peak winter (Jan), trough summer (Jul).
    Daily: peaks at 08:00 and 19:00 (morning/evening demand).
    Seeded for reproducibility.
    """
    rng = np.random.default_rng(42)
    doy = timestamps.dayofyear.to_numpy()
    hour = (timestamps.hour + timestamps.minute / 60).to_numpy()
    annual = 40.0 * np.cos(2 * np.pi * (doy - 15) / 365)
    daily = 20.0 * np.cos(2 * np.pi * (hour - 8) / 24) + 12.0 * np.cos(
        2 * np.pi * (hour - 19) / 12
    )
    noise = rng.normal(0, 5, len(timestamps))
    ci = np.clip(350.0 + annual + daily + noise, 80.0, 600.0)
    return pd.Series(ci, index=timestamps, name="CI")


def _synthetic_ep(timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    California PG&E TOU-A tariff approximation.
    On-peak  16:00-21:00 weekdays  : 0.38 $/kWh
    Mid-peak 09:00-16:00 weekdays  : 0.28 $/kWh
    Off-peak all other times        : 0.18 $/kWh
    """
    hour = np.array(timestamps.hour)
    weekday = np.array(timestamps.dayofweek) < 5  # Mon-Fri
    ep = np.where(
        weekday & (hour >= 16) & (hour < 21), 0.38,
        np.where(weekday & (hour >= 9) & (hour < 16), 0.28, 0.18),
    )
    return pd.Series(ep, index=timestamps, name="EP")


def _wet_bulb(t_amb: pd.Series, rh: pd.Series) -> pd.Series:
    """
    Stull (2011) wet-bulb approximation.
    Validated range: T in [-20, 50] C, RH in [5%, 99%].
    """
    t = t_amb.to_numpy()
    r = rh.to_numpy()
    tw = (
        t * np.arctan(0.151977 * (r + 8.313659) ** 0.5)
        + np.arctan(t + r)
        - np.arctan(r - 1.676331)
        + 0.00391838 * r ** 1.5 * np.arctan(0.023101 * r)
        - 4.686035
    )
    return pd.Series(tw, index=t_amb.index, name="T_wet")


_PRIORITY_MAP = {"Low": 2, "Med": 1, "Medium": 1, "High": 0}


def build(subset: int | None = None) -> None:
    print("=" * 60)
    print("Phase 0 — Dataset Assembly")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load raw CSV
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading {cfg.raw_csv.name} ...")
    df = pd.read_csv(cfg.raw_csv, parse_dates=["timestamp"])
    if subset:
        df = df.head(subset)
        print(f"  DEBUG subset: {subset} rows")
    print(f"  Loaded {len(df):,} rows x {df.shape[1]} cols.")

    # ------------------------------------------------------------------
    # 2. Validate timestamps
    # ------------------------------------------------------------------
    print("\n[2/6] Validating timestamps ...")
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].is_monotonic_increasing, "Timestamps not monotonic!"
    diffs = df["timestamp"].diff().dropna()
    dominant = diffs.mode()[0]
    gaps = diffs[diffs > dominant * 2]
    print(f"  Cadence: {dominant}  |  Gaps > 2x cadence: {len(gaps)}")
    print(f"  Range: {df['timestamp'].min()} ? {df['timestamp'].max()}")

    # ------------------------------------------------------------------
    # 3. Derive workload / hardware columns
    # ------------------------------------------------------------------
    print("\n[3/6] Deriving workload columns ...")
    df["U_cpu"] = (df["CPU_Util"] / 100.0).clip(0.0, 1.0)
    df["U_gpu"] = (df["GPU_Util"] / 100.0).clip(0.0, 1.0)
    df["fan_duty"] = (df["FanSpeed"] / cfg.fan_speed_max_rpm).clip(
        cfg.fan_min, 1.0
    )
    df["priority_class"] = (
        df["RAC"].str.strip().map(_PRIORITY_MAP).fillna(1).astype(int)
    )
    print(f"  priority_class distribution:\n{df['priority_class'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 4. Derive T_wet
    # ------------------------------------------------------------------
    print("\n[4/6] Deriving T_wet (Stull 2011) ...")
    df["T_wet"] = _wet_bulb(df["Temperature"], df["Humidity"])
    print(f"  T_wet range: {df['T_wet'].min():.2f} ? {df['T_wet'].max():.2f} °C")

    # ------------------------------------------------------------------
    # 5. Fetch / synthesise CI and EP
    # ------------------------------------------------------------------
    print("\n[5/6] Building CI and EP time series ...")
    ts_idx = pd.DatetimeIndex(df["timestamp"])
    ci_series = _fetch_ci_historical(ts_idx)
    df["CI"] = ci_series.values
    df["EP"] = _synthetic_ep(ts_idx).values
    print(f"  CI range: {df['CI'].min():.1f} ? {df['CI'].max():.1f} gCO2/kWh")
    print(f"  EP values: {sorted(df['EP'].unique())}")

    # ------------------------------------------------------------------
    # 6. Final validation and save
    # ------------------------------------------------------------------
    print("\n[6/6] Validating and saving ...")
    key_cols = [
        "U_cpu", "U_gpu", "fan_duty", "T_wet", "CI", "EP",
        "InletTemp", "OutletTemp", "IT_Load", "CoolingPower",
        "RC_ServerZoneTemp", "RC_CPUTemp", "RC_GPUTemp",
        "P_cpu", "P_gpu", "P_other", "PUE",
    ]
    null_counts = df[key_cols].isnull().sum()
    if null_counts.any():
        print(f"  WARNING — NaNs in key columns:\n{null_counts[null_counts > 0]}")
        # Forward-fill the 2 known dT_dt_inlet/outlet nulls (row 0 artefact)
        df[key_cols] = df[key_cols].ffill().bfill()
        print("  NaNs resolved with forward/back fill.")
    else:
        print("  No NaNs in key columns. ?")

    cfg.phase0_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.phase0_parquet, index=False, compression="snappy")
    size_mb = cfg.phase0_parquet.stat().st_size / 1e6
    print(f"\n  Saved ? {cfg.phase0_parquet}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Rows: {len(df):,}  Cols: {df.shape[1]}")
    print("\nPhase 0 complete. ?")


if __name__ == "__main__":
    build()

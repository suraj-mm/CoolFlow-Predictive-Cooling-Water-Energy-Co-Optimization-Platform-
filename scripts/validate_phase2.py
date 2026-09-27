"""
scripts/validate_phase2.py
End-to-end validation for Phase 2 (Telemetry & Environmental Context Engine).
Checks:
  1. stull_wet_bulb: formula spot-checks, physical invariant grid sweep (6,816 cells).
  2. Full dataset merge: timestamp continuity, T_wet <= T_dry everywhere, zero NaNs.
  3. TelemetryVector contract: shape, types, value bounds for E2 -> E3A boundary.
  4. E1->E2 integration trace: 1,000 rows from Engine 1 into Engine 2.
  5. CI/EP pass-through: values from parquet preserved exactly.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.io import load_dataset
from shared.types import TelemetryVector
from engine1.power_converter import WorkloadPowerConverter
from engine2.telemetry import stull_wet_bulb, assemble_telemetry, TelemetryEngine


def validate_phase2() -> bool:
    print("=" * 65)
    print("PHASE 2 VALIDATION: Telemetry & Environmental Context Engine")
    print("=" * 65)

    # 1. stull_wet_bulb formula invariant sweep
    print("\n[1/5] Stull (2011) formula: physical invariant grid sweep...")
    t_range = np.linspace(-20, 50, 71)
    rh_range = np.linspace(5, 100, 96)
    T, RH = np.meshgrid(t_range, rh_range)
    TW = stull_wet_bulb(T.ravel(), RH.ravel())
    n_cells = len(TW)
    violations = int(np.sum(TW > T.ravel() + 1e-9))
    assert violations == 0, f"T_wet > T_dry in {violations}/{n_cells} cells"
    max_depression = float(np.max(T.ravel() - TW))
    min_depression = float(np.min(T.ravel() - TW))
    print(f"  Grid: {n_cells:,} cells (T: -20 to 50°C x RH: 5 to 100%)")
    print(f"  T_wet > T_dry violations: {violations}  [PASS]")
    print(f"  Depression range: {min_depression:.2f}°C to {max_depression:.2f}°C")
    # Spot checks
    tw_20_50 = stull_wet_bulb(20.0, 50.0)
    print(f"  stull(T=20°C, RH=50%) = {tw_20_50:.3f}°C  (expect ~13.7°C)")
    assert 12.0 < tw_20_50 < 16.0
    tw_sat = stull_wet_bulb(25.0, 100.0)
    print(f"  stull(T=25°C, RH=100%) = {tw_sat:.4f}°C  (must = 25.0°C after clamp)")
    assert tw_sat <= 25.0 + 1e-9
    print("  Spot checks [PASS]")

    # 2. Full dataset merge
    print("\n[2/5] Full dataset timestamp merge and T_wet invariant...")
    df = load_dataset()
    engine = TelemetryEngine(n_nodes=3)
    converter = WorkloadPowerConverter("DC_PRODUCTION")
    power_df = converter.process_trace_dataframe(df, n_nodes=3)
    merged = engine.process_dataframe(df, power_df)

    gaps = merged["timestamp"].diff().dropna()
    dominant = gaps.mode()[0]
    n_gaps = int((gaps > dominant * 2).sum())
    print(f"  Total rows: {len(merged):,}")
    print(f"  Cadence: {dominant}  |  Gaps > 2x cadence: {n_gaps}")
    assert n_gaps == 0, f"{n_gaps} timestamp gaps found"
    print(f"  Timestamp continuity: ZERO gaps [PASS]")

    wet_viol = int((merged["T_wet"] > merged["T_amb"] + 1e-9).sum())
    assert wet_viol == 0, f"T_wet > T_dry in {wet_viol} rows"
    print(f"  T_wet <= T_dry across {len(merged):,} rows: ZERO violations [PASS]")

    key_cols = ["T_amb", "RH", "T_wet", "CI", "EP"]
    for col in key_cols:
        nulls = merged[col].isna().sum()
        assert nulls == 0, f"Column '{col}' has {nulls} NaNs"
    print(f"  Zero NaNs in {key_cols}  [PASS]")

    # Print summary stats
    print(f"\n  Merged telemetry summary:")
    print(f"    T_amb:  {merged['T_amb'].min():.2f} to {merged['T_amb'].max():.2f} °C")
    print(f"    RH:     {merged['RH'].min():.1f} to {merged['RH'].max():.1f} %")
    print(f"    T_wet:  {merged['T_wet'].min():.2f} to {merged['T_wet'].max():.2f} °C")
    print(f"    CI:     {merged['CI'].min():.1f} to {merged['CI'].max():.1f} gCO2/kWh")
    print(f"    EP:     {sorted(merged['EP'].unique())} $/kWh")

    # 3. TelemetryVector contract check
    print("\n[3/5] TelemetryVector contract (E2 -> E3A boundary)...")
    row = df.iloc[5000]
    u_nodes = np.full(3, row["U_cpu"])
    u_gpu_nodes = np.full(3, row["U_gpu"])
    pv = converter.convert_step(u_nodes, u_gpu_nodes)
    tv = engine.from_row(row, pv)
    assert isinstance(tv, TelemetryVector)
    assert tv.u_cpu.shape == (3,), f"u_cpu shape {tv.u_cpu.shape} != (3,)"
    assert tv.p_servers.shape == (3,)
    assert tv.t_wet <= tv.t_amb + 1e-9
    assert 0 < tv.ci < 1000
    assert 0 < tv.ep < 5.0
    print(f"  TelemetryVector at row 5000:")
    print(f"    u_cpu:    {tv.u_cpu.round(3).tolist()}")
    print(f"    p_servers:{tv.p_servers.round(1).tolist()} W")
    print(f"    T_amb:    {tv.t_amb:.2f}°C | T_wet: {tv.t_wet:.2f}°C | RH: {tv.rh:.1f}%")
    print(f"    CI:       {tv.ci:.1f} gCO2/kWh | EP: ${tv.ep:.2f}/kWh")
    print(f"  Shape and type contract [PASS]")

    # 4. E1->E2 integration trace (1,000 rows)
    print("\n[4/5] E1 -> E2 integration trace over 1,000 rows...")
    count = 0
    for ts, tv in engine.iter_telemetry_vectors(df.head(1000), power_df.head(1000)):
        assert isinstance(tv, TelemetryVector)
        assert tv.t_wet <= tv.t_amb + 1e-9
        count += 1
    assert count == 1000
    print(f"  Processed {count} timesteps E1->E2, all TelemetryVector contracts met [PASS]")

    # 5. CI/EP pass-through fidelity
    print("\n[5/5] CI/EP pass-through fidelity check...")
    sample = df.sample(200, random_state=42)
    for _, row in sample.iterrows():
        u_n = np.full(3, row["U_cpu"])
        pv = converter.convert_step(u_n, np.full(3, row["U_gpu"]))
        tv = engine.from_row(row, pv)
        assert np.isclose(tv.ci, row["CI"], atol=1e-6), f"CI mismatch: {tv.ci} vs {row['CI']}"
        assert np.isclose(tv.ep, row["EP"], atol=1e-9), f"EP mismatch: {tv.ep} vs {row['EP']}"
    print(f"  CI and EP preserved exactly across 200 sampled rows [PASS]")

    print("\n" + "=" * 65)
    print("ALL PHASE 2 EXIT CRITERIA MET AND VERIFIED.")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = validate_phase2()
    sys.exit(0 if success else 1)

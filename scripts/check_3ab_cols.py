"""Inspect dataset columns needed for Engine 3A and 3B feasibility checks."""
import sys; sys.path.insert(0, '.')
import numpy as np
from shared.io import load_dataset

df = load_dataset()

# --- RC tau check ---
tau = df['R_server_param'] * df['C_server_param']
print(f"RC params: R={df['R_server_param'].iloc[0]:.4f} K/W, C={df['C_server_param'].iloc[0]:.0f} J/K")
print(f"tau = R*C = {tau.mean():.1f}s (expected 140-320s range from spec)")

# --- WUE sanity ---
# IT_Load and CoolingPower — check units (kW assumed)
dt_hours = 10.0 / 60.0  # 10-min interval
it_energy_kwh = df['IT_Load'] * dt_hours

# Mass-balance evaporation: m_evap[kg] = Q_reject[kJ] / h_fg[kJ/kg]
# Q_reject = CoolingPower * dt_s (kJ) -- CoolingPower in kW, dt_s=600s
h_fg = 2260.0  # kJ/kg
dt_s = 600.0
q_reject_kj = df['CoolingPower'] * dt_s  # kJ
evap_kg = q_reject_kj / h_fg             # kg

# COC blowdown
coc = 4.0
withdrawal_L = evap_kg * coc / (coc - 1)  # kg -> L (density ~1 kg/L)

wue = withdrawal_L / it_energy_kwh
print("\nWUE (full evap, COC=4):")
print(f"  mean={wue.mean():.3f}  p5={wue.quantile(0.05):.3f}  p95={wue.quantile(0.95):.3f}")
print(f"  min={wue.min():.3f}  max={wue.max():.3f}  (sanity: 0-2.5 L/kWh)")

# --- Free-cooling feasibility ---
if 'T_wet' in df.columns:
    t_wet = df['T_wet']
    t_setpoint = 18.0  # default setpoint
    margin = 5.0
    free_pct = (t_wet <= t_setpoint - margin).mean() * 100
    print(f"\nT_wet range: {t_wet.min():.2f} to {t_wet.max():.2f} C")
    print(f"FreeCooling available (T_wet <= {t_setpoint - margin:.0f}C): {free_pct:.1f}% of timesteps")

# --- InletTemp for twin seeding ---
print(f"\nInletTemp: min={df['InletTemp'].min():.2f} max={df['InletTemp'].max():.2f}")
print(f"FanSpeed:  min={df['FanSpeed'].min():.2f} max={df['FanSpeed'].max():.2f}")
print(f"RC_ServerZoneTemp: min={df['RC_ServerZoneTemp'].min():.2f} max={df['RC_ServerZoneTemp'].max():.2f}")
print(f"CoolingPower: min={df['CoolingPower'].min():.2f} max={df['CoolingPower'].max():.2f} kW")
print(f"IT_Load: min={df['IT_Load'].min():.2f} max={df['IT_Load'].max():.2f} kW")

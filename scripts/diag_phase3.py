"""Diagnose RC MAE and WUE range failures."""
import sys; sys.path.insert(0, '.')
import numpy as np
from shared.io import load_dataset
from engine3b.water_engine import H_FG_KJ_PER_KG, DEFAULT_COC

df = load_dataset()
sample = df.sample(1000, random_state=42).sort_index().reset_index(drop=True)

# RC_ServerZoneTemp vs InletTemp
rc = sample['RC_ServerZoneTemp']
ti = sample['InletTemp']
diff = rc - ti
print(f"RC_ServerZoneTemp: mean={rc.mean():.2f} std={rc.std():.2f} min={rc.min():.2f} max={rc.max():.2f}")
print(f"InletTemp:         mean={ti.mean():.2f} std={ti.std():.2f}")
print(f"Delta RC-Inlet: mean={diff.mean():.2f} std={diff.std():.2f}")

# WUE > 2.5 analysis
it_kw = df['IT_Load']
q_kw = df['CoolingPower']
ratio = q_kw / it_kw
print(f"\nCoolingPower/IT_Load ratio: mean={ratio.mean():.3f} max={ratio.max():.3f}")
print(f"Sample values:")
print(f"  CoolingPower: {q_kw.head(5).values}")
print(f"  IT_Load:      {it_kw.head(5).values}")

dt_s = 600.0
coc = DEFAULT_COC
m_evap = q_kw * dt_s / H_FG_KJ_PER_KG
wd = m_evap * coc / (coc - 1)
it_kwh = it_kw * dt_s / 3600
wue = wd / it_kwh
print(f"\nWUE mass-balance: mean={wue.mean():.3f} max={wue.max():.3f}")
print(f"WUE > 2.5 pct: {(wue > 2.5).mean()*100:.1f}%")
print(f"\nFor IT_Load=100kW, q_reject=110kW:")
wue_ex = (110 * dt_s / H_FG_KJ_PER_KG * coc / (coc-1)) / (100 * dt_s / 3600)
print(f"  WUE = {wue_ex:.3f}")

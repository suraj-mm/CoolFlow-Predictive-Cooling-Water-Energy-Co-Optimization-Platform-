from shared.io import load_dataset
import numpy as np

df = load_dataset()
cols = ['Temperature','Humidity','Pressure','T_wet','CI','EP',
        'InletTemp','OutletTemp','U_cpu','U_gpu','P_cpu','P_gpu','P_other','IT_Load']
print('Cols relevant to E2:')
for c in cols:
    if c in df.columns:
        print(f'  {c}: min={df[c].min():.3f}, max={df[c].max():.3f}, nulls={df[c].isna().sum()}')
    else:
        print(f'  {c}: MISSING')

violations = (df['T_wet'] > df['Temperature']).sum()
delta = df['T_wet'] - df['Temperature']
print(f'\nT_wet > T_dry violations: {violations}')
print(f'Max delta (T_wet - T_dry): {delta.max():.4f}')
print(f'Min delta (T_wet - T_dry): {delta.min():.4f}')
print(f'\nTimestamp gaps:')
gaps = df['timestamp'].diff().dropna()
dominant = gaps.mode()[0]
big_gaps = (gaps > dominant * 2).sum()
print(f'  Dominant cadence: {dominant}')
print(f'  Gaps > 2x cadence: {big_gaps}')
print(f'  Total rows: {len(df)}')

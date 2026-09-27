# CoolFlow 30-Day Dataset Stream Deployment Instructions

## Overview
The CoolFlow command center has been upgraded from a 1-day sample to a **complete 30-day continuous dataset timeline (March 01 to March 30, 2022)** from the production dataset `EQCAM_Dataset_FINAL_with_RC.csv` / `data/phase0_unified.parquet`.

---

## 1. Selected Month: March 2022 (Comprehensive Multi-Level Coverage)
March 2022 was selected because it captures **every single operational level and extreme regime** across all physical cooling physics:

| Level / Parameter | Range in March 2022 | System Response |
| :--- | :--- | :--- |
| **Ambient Temperature** | **-4.7°C to 44.9°C** | Sub-zero arctic nights to extreme daytime heatwaves |
| **Relative Humidity** | **13% to 100%** | Arid desert dry air to dense moisture saturation |
| **Stull Wet-Bulb ($T_{\text{wet}}$)** | **-6.7°C to 43.9°C** | Traverses all 3 ASHRAE psychrometric operating regimes |
| **Mode 0: Free-Air Economizer** | **1,612 intervals (37.3%)** | Active when $T_{\text{wet}} \le 13^\circ\text{C}$; water consumption = 0 L/h |
| **Mode 1: Evaporative Cooling** | **1,107 intervals (25.6%)** | Active when $13^\circ\text{C} < T_{\text{wet}} \le 22^\circ\text{C}$; water = 120–580 L/h |
| **Mode 2: Mechanical Chiller** | **1,601 intervals (37.1%)** | Active when $T_{\text{wet}} > 22^\circ\text{C}$; cooling tower bypassed |
| **CPU Workload** | **15.6% to 99.9%** | Idle baseline to maximum compute stress |
| **Peak Server Temp** | **48.0°C to 88.5°C** | Normal (<76°C), Pre-hotspot (76–80°C), and Critical Hotspot (>80°C) |
| **Cumulative Water Saved** | **149,520 Liters** | Co-optimization vs constant evaporative baseline |

---

## 2. Interactive 30-Day Controls
The dashboard features an interactive control bar:
1. **Play / Pause (`⏸` / `▶`)**: Start or freeze live streaming at any tick.
2. **Step (`⏭`)**: Advance by a single 10-minute dataset record.
3. **Day Skip (`⏮ -1D` / `⏭ +1D`)**: Jump backward or forward by 24 hours (144 ticks).
4. **Timeline Scrubber**: Scrub continuously across all 30 days (`min=0`, `max=4319` locally or `max=359` in Lovable), displaying `DAY XX/30 · YYYY-MM-DD HH:MM:SS`.
5. **Quick Jump Dropdown**: Instantly jump to key representative days:
   - **Day 01**: Sub-zero Free-Air Economizer (-4.7°C, 0 L/h water)
   - **Day 05**: Diurnal transition from night free-air to daytime evaporative
   - **Day 10**: Evaporative cooling peak load
   - **Day 15**: Chiller Heatwave (44.9°C ambient, $T_{\text{wet}} > 22^\circ\text{C}$)
   - **Day 20**: High compute workload saturation (99.9% CPU, hotspot mitigation)
   - **Day 25**: Mixed psychrometric gate transition
   - **Day 30**: Month-end cumulative efficiency summary (149,520 L saved)
6. **Multi-Speed Selector**: `1x`, `5x`, `20x`, `60x` (at 60x, a full 30-day month replays smoothly in 1 minute).

---

## 3. How to Update in Lovable

1. Open your Lovable editor for this project:
   `https://lovable.dev/projects/6eb2a6e1-235d-408d-b6e8-a9eb785c6924`
2. Open **`src/lib/aura-data.ts`** and replace its content with [`lovable_frontend/src/lib/aura-data.ts`](file:///c:/Users/harish/dcproto/lovable_frontend/src/lib/aura-data.ts).
3. Open **`src/routes/index.tsx`** and replace its content with [`lovable_frontend/src/routes/index.tsx`](file:///c:/Users/harish/dcproto/lovable_frontend/src/routes/index.tsx).
4. Lovable will automatically rebuild and deploy to:
   `https://aura-dc-command.lovable.app/`

---

## 4. Local Execution & Live Verification
The standalone engine runs locally at `http://127.0.0.1:8000/`:
- Serves the complete 4,320-record stream directly from `data/dataset_stream.json`.
- Automatically streams live ticks into the SQLite WAL state store.
- Supports Private Network Access (PNA) and CORS.

# Technical Specification: Modified AI-Driven Predictive Thermal Management Framework

## System Architecture & End-to-End Workflow

The **Modified AI-Driven Predictive Thermal Management Framework** is a practical, closed-loop, uncertainty-aware predictive thermal controller designed for simulation-based research and local physical testbed validation. It eliminates over-engineered deep learning components (TFT, GNN synergy, 7-objective genetic algorithms) in favor of lightweight, fast-converging models (Quantile LightGBM, 2D Lumped RC state-space physics, 3-Objective NSGA-II, and Streamlit visualization).

---

```
                       [STEP 1: Workload Traces (Alibaba/Google)]
                                       │
                                       ▼
                 [STEP 2: Telemetry Daemon & Power Converter]
                                       │
                                       ▼
                 [STEP 3: 2D Lumped-RC Thermal Digital Twin]
                                       │
                                       ▼
              [STEP 4: Isolation Forest & KNN Imputer Layer]
                                       │
                                       ▼
             [STEP 5 & 6: Quantile LightGBM + MAPIE Predictor]
                   (Outputs: T_pred + 95% Confidence Bounds)
                                       │
                                       ▼
                   [STEP 7: Hysteresis Hotspot Detector]
                                       │
                                       ▼
            ┌─────────────────────────────────────────────────┐
            │        STEP 8: INTELLIGENT DECISION ENGINE      │
            │  • 8.1 Workload Priority Classifier (XGBoost)   │
            │  • 8.2 & 8.3 Joint Action Space Generator       │
            │  • 8.4 3-Objective Pareto Optimizer (pymoo)     │
            │  • 8.5 Confidence Gate (Parallel / Sequential)  │
            └────────────────────────┬────────────────────────┘
                                     │
                                     ▼
             ┌───────────────────────┴───────────────────────┐
             │                                               │
             ▼                                               ▼
 [STEP 9A: SimPy Macro Simulator]             [STEP 9B: Local 3-Node Docker]
 (Data-Center Scale Execution)               (cgroups CPU & DVFS Scaling)
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     │
                                     ▼
                    [STEP 10: TreeSHAP Explainability Log]
                                     │
                                     ▼
                   [STEP 11: Streamlit Interactive Dashboard]
                                     │
                                     ▼
              [STEP 12 & 13: Experience Replay & Closed Loop]
```

---

## Complete Step-by-Step Implementation Details

---

### STEP 1: Workload Generation & Power Conversion Layer

#### 1. Description
Extracts task utilization logs from public cluster datasets and maps raw CPU/GPU utilization to physical thermal power generation in Watts.

#### 2. Inputs & Outputs
* **Input:** CSV/Parquet trace files from Alibaba Cluster Trace 2018 or Google Cluster Data 2019.
* **Output:** Continuous time-series trace vector $\mathbf{U}(t) = [U_1(t), U_2(t), \dots, U_N(t)]$ where $U_i(t) \in [0, 1]$.

#### 3. Mathematical & Algorithmic Formulation
The power consumption $P_i(t)$ of server node $i$ is calculated using a non-linear CPU-to-Power conversion model derived from SPECpower benchmark data:
$$P_i(t) = P_{idle} + (P_{max} - P_{idle}) \cdot \left( \alpha \cdot U_i(t) + (1 - \alpha) \cdot U_i(t)^3 \right)$$
* **Default Parameters:** $P_{idle} = 150\text{ W}$, $P_{max} = 400\text{ W}$, $\alpha = 0.7$ (voltage scaling factor).
* **GPU Power Addition (if active):** $P_{GPU, i}(t) = P_{GPU, idle} + (P_{GPU, max} - P_{GPU, idle}) \cdot U_{GPU, i}(t)$ ($P_{GPU, max} = 300\text{ W}$).

#### 4. Implementation Stack
* **Language/Libraries:** Python 3.10+, `pandas`, `numpy`, `pyarrow`.
* **Code Structure:**
  ```python
  def calculate_server_power(u_cpu, u_gpu=0.0, p_idle=150.0, p_max=400.0, alpha=0.7):
      p_cpu = p_idle + (p_max - p_idle) * (alpha * u_cpu + (1 - alpha) * (u_cpu ** 3))
      p_gpu = u_gpu * 300.0
      return p_cpu + p_gpu
  ```

---

### STEP 2: Infrastructure Monitoring & Telemetry Daemon

#### 1. Description
Simulates an enterprise telemetry collection bus by merging real-time ambient weather and grid carbon data from external APIs with server power telemetry.

#### 2. Inputs & Outputs
* **Input:** Server power array $\mathbf{P}(t)$, fan duty cycle $V_{fan}(t)$, and external API responses.
* **Output:** Unified Telemetry DataFrame / State Vector $\mathbf{X}(t)$.

#### 3. Mathematical & Algorithmic Formulation
Applies timestamp-aligned merging of ambient temperature $T_{amb}(t)$ (from weather API), Grid Carbon Intensity $CI(t)$ ($\text{gCO}_2/\text{kWh}$), and Electricity Price $EP(t)$ ($\$/\text{kWh}$):
$$\mathbf{X}(t) = \left[ U_i(t), P_i(t), T_{amb}(t), CI(t), EP(t), V_{fan}(t) \right]_{i=1}^N$$

#### 4. Implementation Stack
* **Libraries:** `requests`, `aiohttp`, Open-Meteo API (`api.open-meteo.com`).
* **Code Structure:**
  ```python
  import requests

  def fetch_ambient_context(lat=37.7749, lon=-122.4194):
      url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
      res = requests.get(url).json()
      return res['current_weather']['temperature'] # Ambient temp in Celsius
  ```

---

### STEP 3: 2D Lumped-RC Thermal Digital Twin

#### 1. Description
A physics-based discrete-time thermal model that estimates heat accumulation, inter-server heat transfer, and convective heat removal across server racks without requiring physical sensors.

#### 2. Inputs & Outputs
* **Input:** Server power vector $\mathbf{P}(t)$, fan duty cycle $V_{fan}(t)$, ambient inlet temperature $T_{inlet}(t)$.
* **Output:** Server spatial temperature matrix $\mathbf{T}(t) = [T_1(t), T_2(t), \dots, T_N(t)]^\top$.

#### 3. Mathematical Formulation
Modeled using a 2D Lumped Resistance-Capacitance (RC) State-Space Network solved via discrete Euler integration ($\Delta t = 1\text{ s}$):
$$C_i \frac{T_i(t+\Delta t) - T_i(t)}{\Delta t} = P_i(t) - \sum_{j \in \text{adj}(i)} \frac{T_i(t) - T_j(t)}{R_{ij}} - \frac{T_i(t) - T_{inlet}(t)}{R_{air}(V_{fan})}$$

Where convective airflow resistance $R_{air}$ scales dynamically with fan duty cycle $V_{fan} \in [0.3, 1.0]$:
$$R_{air}(V_{fan}) = R_0 \cdot \left( \frac{V_{fan, 0}}{V_{fan}} \right)^\gamma$$

* **Calibrated Constants:** $C_i = 400.0\text{ J/K}$ (server thermal mass), $R_{ij} = 0.8\text{ K/W}$ (rack conduction resistance), $R_0 = 0.35\text{ K/W}$, $\gamma = 0.8$.

#### 4. Implementation Stack
* **Libraries:** `numpy`, `numba` (JIT acceleration).
* **Code Structure:**
  ```python
  from numba import njit
  import numpy as np

  @njit
  def step_rc_thermal_twin(T_curr, P_vec, T_inlet, V_fan, C=400.0, R_adj=0.8, R0=0.35, dt=1.0):
      N = len(T_curr)
      T_next = np.empty_like(T_curr)
      R_air = R0 * ((0.5 / max(V_fan, 0.1)) ** 0.8)
      
      for i in range(N):
          q_adj = 0.0
          if i > 0: q_adj += (T_curr[i] - T_curr[i-1]) / R_adj
          if i < N - 1: q_adj += (T_curr[i] - T_curr[i+1]) / R_adj
          q_cool = (T_curr[i] - T_inlet) / R_air
          
          dT = (P_vec[i] - q_adj - q_cool) / C
          T_next[i] = T_curr[i] + dT * dt
      return T_next
  ```

---

### STEP 4: Data Validation & Sensor Reliability Layer

#### 1. Description
Identifies faulty sensor readings (stuck-at, noise spikes) and imputes missing telemetry values before feeding downstream ML models.

#### 2. Sub-Modules & Algorithms
* **4.1 Fault Detection:** `IsolationForest(n_estimators=100, contamination=0.02)` flags anomalous temperature readings ($T > 100^\circ\text{C}$ or $T < 10^\circ\text{C}$).
* **4.2 Missing Data Recovery:** `KNNImputer(n_neighbors=5)` or `IterativeImputer(estimator=BayesianRidge())` restores missing `NaN` values.

#### 3. Code Structure
```python
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

class DataValidationLayer:
    def __init__(self):
        self.detector = IsolationForest(contamination=0.02, random_state=42)
        self.imputer = KNNImputer(n_neighbors=5)
        
    def validate_and_clean(self, X_raw):
        # 1. Detect anomalies and replace with NaN
        preds = self.detector.fit_predict(X_raw)
        X_cleaned = X_raw.copy()
        X_cleaned[preds == -1] = np.nan
        
        # 2. Impute NaNs using KNN
        X_restored = self.imputer.fit_transform(X_cleaned)
        return X_restored
```

---

### STEP 5 & 6: AI-Based Thermal Prediction & Confidence Estimation

#### 1. Description
Forecasts future server temperatures $\hat{T}_i(t + \tau)$ ($\tau = 5, 10\text{ min}$) and computes uncertainty confidence intervals.

#### 2. Models Chosen
* **Primary Forecaster:** **Quantile LightGBM Regressor** (`lightgbm.LGBMRegressor`).
* **Uncertainty Quantification:** Quantile Loss bounds ($q = 0.05, 0.50, 0.95$) or Conformal Prediction (`MAPIE`).

#### 3. Mathematical Formulation
The model predicts three target values per server:
$$\hat{y}_{lower} = f_{q=0.05}(\mathbf{x}), \quad \hat{y}_{mid} = f_{q=0.50}(\mathbf{x}), \quad \hat{y}_{upper} = f_{q=0.95}(\mathbf{x})$$

Interval Width $\delta_i = \hat{y}_{upper} - \hat{y}_{lower}$.  
Confidence Score $CS_i$:
$$CS_i = \max\left(0, \, 1 - \frac{\delta_i}{\delta_{max}}\right) \times 100\% \quad (\text{where } \delta_{max} = 10.0^\circ\text{C})$$

#### 4. Code Structure
```python
import lightgbm as lgb

class QuantileThermalPredictor:
    def __init__(self):
        self.models = {
            'low': lgb.LGBMRegressor(objective='quantile', alpha=0.05, n_estimators=100),
            'mid': lgb.LGBMRegressor(objective='regression', n_estimators=100),
            'high': lgb.LGBMRegressor(objective='quantile', alpha=0.95, n_estimators=100)
        }
        
    def train(self, X, y):
        for q, model in self.models.items():
            model.fit(X, y)
            
    def predict_with_confidence(self, X_input):
        y_low = self.models['low'].predict(X_input)
        y_mid = self.models['mid'].predict(X_input)
        y_high = self.models['high'].predict(X_input)
        
        delta = y_high - y_low
        confidence = np.maximum(0.0, 1.0 - (delta / 10.0)) * 100.0
        return y_mid, confidence
```

---

### STEP 7: Hotspot Detection Engine

#### 1. Description
Monitors predicted temperature upper bounds and triggers emergency mitigation states when safety thresholds are breached.

#### 2. Logic & Formulation
Applies a Hysteresis Deadband Filter to prevent alert chattering:
$$\text{Hotspot Alert Triggered} \iff \hat{T}_{upper, i}(t+\tau) \ge T_{crit} \quad (T_{crit} = 80.0^\circ\text{C})$$
$$\text{Hotspot Alert Cleared} \iff T_i(t) < T_{crit} - \Delta_{deadband} \quad (\Delta_{deadband} = 4.0^\circ\text{C})$$

---

### STEP 8: Intelligent Decision Engine

#### 1. Description
Evaluates candidate thermal mitigation actions and selects Pareto-optimal mitigation strategies balancing energy, thermal safety, and performance.

#### 2. Sub-Modules Breakdown
* **8.1 Workload Priority Classifier:** `XGBoostClassifier` categorizes workloads into:
  - `Class 0 (Critical)`: Zero throttling allowed.
  - `Class 1 (Standard)`: Up to 30% throttling allowed.
  - `Class 2 (Batch/Delayable)`: Deferrable or 75% throttling allowed.
* **8.2 Candidate Action Generator:** Generates discrete action tuples $\mathbf{a} = (a_{mig}, a_{dvfs}, a_{fan})$:
  - $a_{mig} \in \{\text{No Migration}, \text{Node}_i \rightarrow \text{Node}_j\}$.
  - $a_{dvfs} \in \{1.2\text{ GHz}, 1.8\text{ GHz}, 2.4\text{ GHz}, 3.0\text{ GHz}\}$.
  - $a_{fan} \in \{30\%, 50\%, 80\%, 100\%\}$.
* **8.3 Joint Action Matrix Evaluation (Replaces GNN):** Evaluates combined action power reductions directly using vectorized RC state-space equations.
* **8.4 3-Objective Pareto Optimizer (`pymoo` NSGA-II):**
  - **Objective 1 (Minimize Energy):** $f_1(\mathbf{a}) = P_{servers}(\mathbf{a}) + P_{fans}(V_{fan})$.
  - **Objective 2 (Minimize Thermal Penalty):** $f_2(\mathbf{a}) = \sum_i \max(0, \hat{T}_i(\mathbf{a}) - T_{crit})^2$.
  - **Objective 3 (Minimize Performance/SLA Loss):** $f_3(\mathbf{a}) = w_{throttle} \cdot \Delta U + w_{mig} \cdot N_{migrations}$.
* **8.5 Confidence Gate Logic:**
  ```python
  if min_confidence >= 80.0:
      execution_mode = "PARALLEL"   # Execute migration + DVFS + fan adjustments simultaneously
  else:
      execution_mode = "SEQUENTIAL" # Execute fan adjustment first; re-check thermal state before migration
  ```

---

### STEP 9: Action Execution Layer (Dual-Track)

#### 1. Description
Implements the selected mitigation strategy in both the data-center scale macro simulator and a local 3-node hardware testbed.

#### 2. Dual Track Specifications
* **Track A (Macro Simulator - SimPy):** Updates global state vectors ($V_{fan}$, $P_i$, $U_i$) inside the `SimPy` discrete event loop.
* **Track B (Hardware Demo Testbed - Linux/Docker):**
  - **CPU Throttling:** Executes `docker update --cpus="0.5" container_id` via `docker-py` SDK.
  - **DVFS Frequency Scaling:** Executes `cpupower frequency-set -u 1.8GHz` via sub-process call.
  - **Container Relocation:** Executes `docker stop container_id` on Node 1 and `docker run` on Node 2.

---

### STEP 10: SHAP Explainability Module

#### 1. Description
Generates feature attributions explaining *why* a hotspot was predicted and *why* a specific action was chosen.

#### 2. Implementation
Uses `shap.TreeExplainer` on the LightGBM thermal predictor to output feature importance rankings (e.g., "Server 2 CPU load contributed +6.4°C, Ambient Temp contributed +2.1°C"). Logs outputs to structured JSON.

```python
import shap

def explain_prediction(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values
```

---

### STEP 11: Dashboard and Visualization Layer

#### 1. Description
Provides a real-time web interface built with **Streamlit** and **Plotly** for live system monitoring.

#### 2. Display Features
* Live 2D Spatial Thermal Heatmap of Server Racks.
* Real-time temperature time-series with shaded 95% confidence intervals.
* Decision Audit Table (Predicted Temp, Confidence %, Selected Pareto Action, Energy Saved).
* Carbon & Water Footprint Savings Counters.

---

### STEP 12 & 13: Experience Replay & Closed-Loop Cycle

#### 1. Description
Stores closed-loop transition tuples $(\mathbf{s}_t, \mathbf{a}_t, \mathbf{s}_{t+1}, r_t)$ into a local Parquet/SQLite database.

#### 2. Retraining Logic
Every 100 simulation cycles, the system initiates an offline batch retraining job for the LightGBM predictor using the updated replay buffer, ensuring adaptive long-term accuracy without live gradient drift.

---

# Key Parameters & System Configuration Table

| Parameter | Symbol | Value | Unit | Description |
| :--- | :---: | :---: | :---: | :--- |
| Server Idle Power | $P_{idle}$ | 150.0 | Watts | Baseline power when unutilized |
| Server Peak Power | $P_{max}$ | 400.0 | Watts | Power consumption at 100% CPU/GPU load |
| Thermal Capacitance | $C_i$ | 400.0 | J/K | Heat capacity per server node |
| Adjacent Conduction Resistance | $R_{ij}$ | 0.8 | K/W | Thermal resistance between rack neighbors |
| Base Airflow Resistance | $R_0$ | 0.35 | K/W | Convective cooling resistance at 50% fan speed |
| Critical Thermal Threshold | $T_{crit}$ | 80.0 | ${}^\circ\text{C}$ | Maximum safe server operating temperature |
| Deadband Hysteresis | $\Delta_{deadband}$ | 4.0 | ${}^\circ\text{C}$ | Temperature delta required to clear alarm |
| Forecasting Horizon | $\tau$ | 5 – 10 | Minutes | Future prediction lookahead window |
| Loop Step Latency | $t_{step}$ | ~145 | Milliseconds | Compute latency per closed-loop iteration |

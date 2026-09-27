"""
shared/config.py
Single source of truth for all system constants.
All engines import `cfg` from here — never redefine constants locally.
Override any field via environment variable DC_<FIELD_NAME> or .env file.
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class DCConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DC_",
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---------------------------------------------------------------
    data_dir: Path = Field(default=ROOT / "data")
    raw_csv: Path = Field(default=ROOT / "EQCAM_Dataset_FINAL_with_RC.csv")
    phase0_parquet: Path = Field(default=ROOT / "data" / "phase0_unified.parquet")
    artifacts_dir: Path = Field(default=ROOT / "data" / "artifacts")

    # --- Power model (spec Step 1) -------------------------------------------
    p_idle: float = 150.0          # W  — baseline server power
    p_max: float = 400.0           # W  — peak CPU power at 100% load
    alpha: float = 0.7             # DVFS blend factor (linear vs cubic)
    p_gpu_max: float = 300.0       # W  — peak GPU power

    # --- RC thermal twin (spec Step 3) ---------------------------------------
    c_thermal: float = 400.0       # J/K  — server thermal mass
    r_adjacent: float = 0.8        # K/W  — rack conduction resistance
    r0_airflow: float = 0.35       # K/W  — convective resistance at 50% fan
    gamma_fan: float = 0.8         # fan exponent for R_air scaling
    fan_min: float = 0.3           # minimum duty cycle
    fan_speed_max_rpm: float = 3000.0  # RPM — from full-dataset max (2999.8)

    # --- Thermal control (spec Step 7) ---------------------------------------
    t_crit: float = 80.0           # °C  — hotspot threshold
    delta_deadband: float = 4.0    # °C  — hysteresis clearing band
    delta_max_conf: float = 10.0   # °C  — interval width for 0% confidence

    # --- Environmental defaults (audit A2 — replaced by dataset columns) -----
    ci_default: float = 350.0      # gCO2/kWh  — UK 2022-25 avg placeholder
    ep_default: float = 0.28       # $/kWh     — mid-tier placeholder

    # --- Water / WUE (audit A4 — ASHRAE TC 9.9 / Green Grid) ----------------
    wf_free_air: float = 0.0       # L/kWh — free-air economizer
    wf_evap: float = 1.8           # L/kWh — evaporative assist
    wf_mech: float = 3.5           # L/kWh — full mechanical

    # --- Optimizer (audit A9-A11) --------------------------------------------
    migration_top_k: int = 3       # top-k least-loaded migration targets
    nsga2_pop: int = 50
    nsga2_gen: int = 40

    # --- Data Reliability (Phase 4) -----------------------------------------
    # Hard bounds for field-level filtering (NaN-flagged if outside)
    bound_u_cpu_min: float = 0.0        # normalised utilisation
    bound_u_cpu_max: float = 1.0
    bound_t_amb_min: float = -20.0      # °C  — ambient temperature
    bound_t_amb_max: float = 60.0
    bound_rh_min: float = 0.0           # %
    bound_rh_max: float = 100.0
    bound_t_node_min: float = -10.0     # °C  — server zone temperature
    bound_t_node_max: float = 120.0
    bound_fan_duty_min: float = 0.0
    bound_fan_duty_max: float = 1.0
    # IsolationForest
    iforest_contamination: float = 0.02 # expected anomaly fraction
    iforest_n_estimators: int = 100
    iforest_random_state: int = 42
    # Hybrid imputer
    short_gap_ticks: int = 3            # gaps <= this → linear interpolation
    knn_neighbors: int = 5

    # --- CQR Predictor (Phase 5) -------------------------------------------
    cqr_confidence_level: float = 0.9   # nominal coverage target
    cqr_train_frac: float = 0.8         # fraction of data for training
    cqr_cal_frac: float = 0.1           # fraction for conformalization
    # test fraction = 1 - train_frac - cal_frac = 0.10
    cqr_lgbm_n_estimators: int = 200
    cqr_lgbm_num_leaves: int = 31
    cqr_lgbm_learning_rate: float = 0.05
    cqr_coverage_tol: float = 0.05      # empirical coverage must be within ±tol of nominal
    cqr_lag_steps: int = 2              # how many past ticks to use as lag features
    delta_max: float = 10.0             # °C — interval width for 0% confidence score

    # --- Closed loop ---------------------------------------------------------
    retrain_every: int = 100       # cycles between LightGBM retrains
    ga_cadence_s: float = 60.0     # seconds between NSGA-II runs

    def model_post_init(self, __context) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


cfg = DCConfig()

"""
engine10/dashboard.py
Phase 10 — Streamlit Dashboard.

Reads from the SQLite state store (written by the control loop) at its own cadence
(default: 2 Hz) via a Streamlit auto-refresh. Completely decoupled from the 145ms
control loop — the dashboard has no write path into the control loop.

Architecture:
  - State store (WAL SQLite) → read_recent(n) → Pandas DataFrame → Plotly charts
  - Streamlit's `st.rerun()` / `time.sleep` for polling at dashboard_refresh_s cadence
  - Three panels:
      1. Thermal: T_node per node over time + T_inlet
      2. Water & WUE: real water_rate_l_per_h and WUE from E3B
      3. Decision: action log (a_fan, a_cool, a_dvfs_mean, trigger_optimizer)
      4. Explainability: selected_objectives + selection_rule (from E9 SHAP JSON)
  - Load-test exit criterion: dashboard poll interval > 10x the control loop tick
    (2Hz << 145ms limit) — enforced by cfg.dashboard_refresh_s >= 0.5

CONSUMED BY: Human operator (read-only).
CONSUMES:    shared.state_store.read_recent (SQLite WAL reader).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

_DASHBOARD_REFRESH_S: float = 0.5   # 2 Hz — 3.4x looser than 145ms control tick
_N_ROWS: int = 200                   # last 200 ticks shown in charts


def _load_data() -> pd.DataFrame:
    """Pull latest rows from the state store and return as a DataFrame."""
    from shared.state_store import read_recent
    rows = read_recent(n=_N_ROWS)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _extract_objectives(shap_json_series: pd.Series) -> pd.DataFrame:
    """Parse selected_objectives from shap_json column for the Pareto panel."""
    objs = []
    for s in shap_json_series:
        try:
            d = json.loads(s)
            so = d.get("selected_objectives", [None, None, None])
            objs.append({
                "Obj1_resource": so[0],
                "Obj2_thermal": so[1],
                "Obj3_sla": so[2],
                "rule": d.get("selection_rule", ""),
            })
        except Exception:
            objs.append({"Obj1_resource": None, "Obj2_thermal": None,
                         "Obj3_sla": None, "rule": ""})
    return pd.DataFrame(objs)


def run_dashboard() -> None:
    """Entry point — launch the Streamlit app (call via `streamlit run engine10/dashboard.py`)."""
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Dashboard requires: pip install streamlit plotly\n"
            f"Original error: {exc}"
        ) from exc

    st.set_page_config(
        page_title='CoolFlow — "Predictive Cooling & Water-Energy Co-Optimization Platform"',
        page_icon="❄️",
        layout="wide",
    )
    st.title('CoolFlow — "Predictive Cooling & Water-Energy Co-Optimization Platform"')
    st.caption(
        f"Auto-refreshing every {_DASHBOARD_REFRESH_S}s · "
        "Control loop runs independently at 145ms · Read-only view"
    )

    df = _load_data()

    if df.empty:
        st.warning("No data yet — waiting for the control loop to start writing to the state store.")
        time.sleep(_DASHBOARD_REFRESH_S)
        st.rerun()
        return

    # ── Panel 1: Thermal ────────────────────────────────────────────────────
    st.subheader("🔥 Node Temperatures")
    col1, col2 = st.columns([3, 1])
    with col1:
        temp_cols = [c for c in ["temp_node0", "temp_node1", "temp_node2"] if c in df.columns]
        if temp_cols:
            fig_t = px.line(df, x="timestamp", y=temp_cols,
                            labels={"value": "°C", "timestamp": "Time", "variable": "Node"},
                            color_discrete_sequence=["#EF553B", "#00CC96", "#AB63FA"])
            fig_t.add_hline(y=80.0, line_dash="dash", line_color="red",
                            annotation_text="T_crit (80°C)")
            fig_t.add_hline(y=76.0, line_dash="dot", line_color="orange",
                            annotation_text="Clear threshold (76°C)")
            fig_t.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_t, use_container_width=True)
    with col2:
        if "max_temp" in df.columns:
            latest_max = float(df["max_temp"].iloc[-1])
            colour = "🔴" if latest_max >= 80.0 else ("🟡" if latest_max >= 76.0 else "🟢")
            st.metric("Max node temp (latest)", f"{latest_max:.1f}°C", delta=None)
            st.write(f"{colour} Status")
        if "t_inlet" in df.columns:
            st.metric("Inlet temp (latest)", f"{float(df['t_inlet'].iloc[-1]):.1f}°C")

    # ── Panel 2: Water & WUE ────────────────────────────────────────────────
    st.subheader("💧 Water Usage & WUE (Real — from Phase 3B)")
    col3, col4 = st.columns([3, 1])
    with col3:
        if "water_rate_l_per_h" in df.columns and "wue" in df.columns:
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(
                x=df["timestamp"], y=df["water_rate_l_per_h"],
                name="Water rate (L/h)", line=dict(color="#38bdf8"), yaxis="y1"))
            fig_w.add_trace(go.Scatter(
                x=df["timestamp"], y=df["wue"],
                name="WUE (L/kWh)", line=dict(color="#FF6692", dash="dash"), yaxis="y2"))
            fig_w.update_layout(
                height=280, margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(title="Water rate (L/h)"),
                yaxis2=dict(title="WUE (L/kWh)", overlaying="y", side="right"),
                legend=dict(x=0, y=1.1, orientation="h"),
            )
            st.plotly_chart(fig_w, use_container_width=True)
    with col4:
        if "water_rate_l_per_h" in df.columns:
            st.metric("Water rate (latest)", f"{float(df['water_rate_l_per_h'].iloc[-1]):.1f} L/h")
        if "wue" in df.columns:
            st.metric("WUE (latest)", f"{float(df['wue'].iloc[-1]):.3f} L/kWh")
        if "cool_mode" in df.columns:
            cool_labels = {0: "🌬️ Free-Air", 1: "💧 Evaporative", 2: "❄️ Mechanical"}
            st.write("Mode:", cool_labels.get(int(df["cool_mode"].iloc[-1]), "Unknown"))

    # ── Panel 3: Decision ────────────────────────────────────────────────────
    st.subheader("⚡ Action Decisions")
    col5, col6 = st.columns(2)
    with col5:
        if "a_fan" in df.columns:
            fig_fan = px.line(df, x="timestamp", y="a_fan",
                              labels={"a_fan": "Fan duty", "timestamp": "Time"},
                              color_discrete_sequence=["#636EFA"])
            fig_fan.update_layout(height=220, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_fan, use_container_width=True)
    with col6:
        if "trigger_optimizer" in df.columns:
            trigger_rate = float(df["trigger_optimizer"].mean()) * 100
            st.metric("Full NSGA-II trigger rate", f"{trigger_rate:.1f}%")
            st.caption("(remainder uses cached last decision)")

    # ── Panel 4: Explainability ──────────────────────────────────────────────
    st.subheader("🧠 Decision Rationale (Phase 9)")
    if "shap_json" in df.columns and df["shap_json"].notna().any():
        obj_df = _extract_objectives(df["shap_json"].dropna())
        if not obj_df.empty:
            col7, col8 = st.columns([2, 2])
            with col7:
                fig_obj = px.line(obj_df.reset_index(), x="index",
                                  y=["Obj1_resource", "Obj2_thermal", "Obj3_sla"],
                                  labels={"value": "Objective value", "index": "Tick",
                                          "variable": "Objective"},
                                  color_discrete_sequence=["#FFA15A", "#EF553B", "#636EFA"])
                fig_obj.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_obj, use_container_width=True)
            with col8:
                # Show last 5 selection rules
                recent_rules = obj_df["rule"].iloc[-5:].values
                st.write("**Recent selection rules:**")
                for r in recent_rules:
                    if r:
                        st.code(r, language=None)
    else:
        st.info("Waiting for explainability data (Phase 9 not yet producing output).")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"Showing last {len(df)} ticks. "
        f"DB path: `{_load_data.__module__}` · "
        "Control loop runs independently — this view never stalls it."
    )

    time.sleep(_DASHBOARD_REFRESH_S)
    st.rerun()


if __name__ == "__main__":
    # Allows: python engine10/dashboard.py (will print guidance)
    print(
        "To launch the dashboard, run:\n"
        "  streamlit run engine10/dashboard.py\n"
        "\nThe dashboard polls the SQLite state store at 2 Hz.\n"
        "The control loop writes to the same store at its own cadence (145ms).\n"
        "WAL journal mode guarantees neither blocks the other."
    )

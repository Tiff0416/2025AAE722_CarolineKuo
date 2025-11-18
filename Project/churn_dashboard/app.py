import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Helper: resolve base directory (normal vs PyInstaller)
# ---------------------------------------------------------
def get_base_dir():
    """
    When bundled with PyInstaller, data files are unpacked into a
    temporary folder accessible via sys._MEIPASS.
    Otherwise, use the folder where this script lives.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # Running in a PyInstaller bundle
        return sys._MEIPASS
    else:
        # Running in a normal Python environment
        return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = get_base_dir()

# Data paths (relative to BASE_DIR)
CLUSTER_PATH = os.path.join(BASE_DIR, "data", "processed", "cluster_tableau.csv")
SHAP_PATH    = os.path.join(BASE_DIR, "data", "processed", "shap_summary.csv")
ROI_PATH     = os.path.join(BASE_DIR, "data", "processed", "roi_simulation.csv")

# ---------------------------------------------------------
# Streamlit Layout 設定
# ---------------------------------------------------------
st.set_page_config(
    page_title="Churn Dashboard",
    layout="wide"
)

st.title("Customer Churn Dashboard")

# ---------------------------------------------------------
# 載入資料
# ---------------------------------------------------------
@st.cache_data
def load_data():
    cluster_df = pd.read_csv(CLUSTER_PATH)
    shap_df = pd.read_csv(SHAP_PATH)
    roi_df = pd.read_csv(ROI_PATH)
    return cluster_df, shap_df, roi_df

cluster, shap_summary, roi_df = load_data()

# ---------------------------------------------------------
# Section 1 — WHO: Segmentation Summary
# ---------------------------------------------------------
st.header("WHO — Segmentation Summary")

cluster_display = (
    cluster
    .groupby("cluster")
    .agg({
        "customer_id": "count",
        "Churn_Flag": "mean",
        "Total_Trans_Ct": "mean",
        "Total_Trans_Amt": "mean"
    })
    .reset_index()
)

cluster_display.columns = [
    "Cluster",
    "N_Customers",
    "Churn_Rate",
    "Avg_Trans_Ct",
    "Avg_Trans_Amt"
]

st.dataframe(cluster_display, use_container_width=True)

# ---------------------------------------------------------
# Section 2 — WHY: SHAP Drivers
# ---------------------------------------------------------
st.header("WHY — Key Drivers (SHAP Values)")

# SHAP CSV 欄位名稱：feature, mean_abs_shap
shap_summary = shap_summary.sort_values("mean_abs_shap", ascending=True)

shap_plot = px.bar(
    shap_summary,
    x="mean_abs_shap",
    y="feature",
    orientation="h",
    title="Top Feature Importance (SHAP)",
    labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"}
)

st.plotly_chart(shap_plot, use_container_width=True)

# ---------------------------------------------------------
# Section 3 — WHAT: Intervention + Interactive ROI
# ---------------------------------------------------------
st.header("WHAT — Proposed Intervention & ROI Simulator")

st.subheader("Interactive ROI Calculator")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Input Assumptions")

    segment_size = st.number_input(
        "Segment Size (customers)",
        min_value=100,
        max_value=20000,
        value=1234,
        step=50
    )

    baseline_churn = st.number_input(
        "Baseline Churn Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.16,
        step=0.01
    )

    retention_lift = st.slider(
        "Retention Lift (%)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.5
    ) / 100.0  # convert % -> decimal

with col2:
    value_per_customer = st.number_input(
        "Value per Retained Customer ($)",
        min_value=50,
        max_value=2000,
        value=300,
        step=10
    )

    cost_per_customer = st.number_input(
        "Campaign Cost per Customer ($)",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )

# ---------------------------------------------------------
# Dynamic ROI calculation
# ---------------------------------------------------------
incremental_customers = segment_size * retention_lift
revenue = incremental_customers * value_per_customer
cost = segment_size * cost_per_customer
net_profit = revenue - cost
roi_value = net_profit / cost if cost > 0 else 0

st.markdown("### Output Metrics")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Retention Lift", f"{retention_lift*100:.1f}%")
k2.metric("Incremental Customers", f"{incremental_customers:,.0f}")
k3.metric("Net Profit", f"${net_profit:,.0f}")
k4.metric("ROI", f"{roi_value:.2f}x")

# ---------------------------------------------------------
# ROI Curve from CSV + Current Slider Point
# ---------------------------------------------------------
st.subheader("ROI vs Retention Lift (Scenario Curve + Current Point)")

roi_df = roi_df.copy()
roi_df.columns = roi_df.columns.str.lower()

# 確保必要欄位為數字
for col in ["retention_lift", "net_profit", "campaign_cost"]:
    roi_df[col] = roi_df[col].astype(float)

# 在 app 裡重新計算 ROI（避免 CSV 內欄位有誤）
roi_df["roi_calc"] = roi_df["net_profit"] / roi_df["campaign_cost"]
roi_df = roi_df.sort_values("retention_lift")

fig = go.Figure()

# 模擬情境的折線
fig.add_trace(
    go.Scatter(
        x=roi_df["retention_lift"],
        y=roi_df["roi_calc"],
        mode="lines+markers",
        name="Simulated scenarios"
    )
)

# 目前 slider 對應的點
fig.add_trace(
    go.Scatter(
        x=[retention_lift],
        y=[roi_value],
        mode="markers+text",
        name="Current setting",
        text=[f"{roi_value:.2f}x"],
        textposition="top center",
        marker=dict(size=12)
    )
)

fig.update_layout(
    title="ROI vs Retention Lift (Simulated Scenarios)",
    xaxis_title="Retention lift",
    yaxis_title="ROI (x)"
)

st.plotly_chart(fig, use_container_width=True)

st.success("Dashboard Loaded Successfully!")

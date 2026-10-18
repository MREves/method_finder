"""Streamlit Page: Machine Learning, Algorithmic Equity & Operational Demand Forecasting.

Pillar 4 of the Method Finder Platform.
Provides an interactive biostatistical and operational framework for:
1. Supervised Clinical Classification (Severe class imbalance, PR-AUC, decision threshold tuning).
2. Algorithmic Fairness & Health Equity Audits (Four-Fifths 80% Rule across IMD deprivation deciles).
3. Operational Time Series Demand Forecasting (A&E arrivals, bed surges, staff sickness absences).
4. Unsupervised Patient Phenotyping (K-Means clustering, PCA variance extraction).
"""

import math
from typing import Dict, Any, List
import streamlit as st

from src.ml_models.ml_schema import (
    ML_TASK_FAMILIES,
    ML_MODELS,
    ML_SCENARIOS,
)
from src.ml_models.ml_engine import (
    generate_ml_demo_data,
    calculate_algorithmic_fairness_metrics,
    calculate_time_series_forecast,
    calculate_classification_threshold_metrics,
    generate_decision_boundary_surface,
    calculate_cluster_metrics,
    calculate_pca_decomposition,
    route_ml_recommendation,
    evaluate_ml_prerequisites,
    generate_ml_python_pipeline,
)
from src.ui_components.ml_widgets import (
    render_ml_plain_english_guide,
    render_ml_navigation_header,
    render_ml_guided_wizard,
    render_ml_direct_explorer,
    render_ml_scenario_and_parameters,
    render_ml_prerequisite_cards,
    render_ml_kpi_dashboard,
    render_ml_visualizations,
    render_ml_educational_tabs,
)
from src.ui_components.sidebar_navigation import (
    render_custom_sidebar_navigation,
    render_roundtrip_navigation_banner
)
from src.ui_components.sample_size_badge import render_sample_size_deep_link_badge

# 1. Page Configuration
st.set_page_config(
    page_title="Machine Learning & Operational Forecasting",
    page_icon="🤖",
    layout="wide"
)

# Render Custom Grouped Sidebar Navigation
render_custom_sidebar_navigation("pages/3_🤖_Machine_Learning.py")

# 2. Header & Title Banner
st.title("🤖 Machine Learning, Algorithmic Equity & Operational Forecasting")
render_roundtrip_navigation_banner()
st.markdown("""
Welcome to the **Machine Learning & Predictive Analytics** module. This tool bridges the gap between 
theoretical algorithms and frontline healthcare delivery. Whether you are predicting **rare clinical events** 
(30-day heart failure readmissions), auditing **algorithmic bias across socio-economic deprivation deciles** 
(outpatient appointment attendance), or forecasting **daily hospital capacity surges** (A&E arrivals and bed demand), 
this guide ensures rigorous biostatistical methodology, zero temporal leakage, and ethical AI deployment.
""")

# Quick Navigation Link back to Core Hub
c_nav1, c_nav2, c_nav3, c_nav4 = st.columns(4)
with c_nav1:
    st.page_link("app.py", label="← Home Overview Hub", icon="🏠")
with c_nav2:
    st.page_link("pages/1_📊_Statistical_Tests.py", label="Statistical Tests", icon="📊")
with c_nav3:
    st.page_link("pages/2_🏥_Epidemiology_Methods.py", label="Epidemiology Methods", icon="🏥")
with c_nav4:
    st.page_link("pages/4_📐_Sample_Size_and_Power.py", label="Sample Size & Power", icon="📐")

st.markdown("---")

# 3. Plain English Pedagogical Introductory Guide (Zero Knowledge)
render_ml_plain_english_guide()

st.markdown("---")

# 4. Navigation Mode Selection (Guided Decision Wizard vs Direct Architecture Explorer)
is_direct_mode = render_ml_navigation_header()

if not is_direct_mode:
    selected_model_key, rec_dict = render_ml_guided_wizard()
else:
    selected_model_key = render_ml_direct_explorer()

st.markdown("---")

# 5. Clinical / Operational Scenario Selection & Live Simulation Settings
scenario_key, meta, params = render_ml_scenario_and_parameters(selected_model_key)
render_sample_size_deep_link_badge(
    method_key=selected_model_key,
    method_name=meta.get("recommended_algorithm", "Machine Learning Architecture"),
    pillar="Machine Learning & AI",
    origin_page="pages/3_🤖_Machine_Learning.py",
    origin_title="Machine Learning",
    origin_icon="🤖"
)

# 6. Execute Mathematical / Algorithmic Calculations (Dynamically with N!)
n_samples = params.get("sample_size")
demo_data = generate_ml_demo_data(scenario_key, n_samples=n_samples)
raw_dict = demo_data["raw_data"]
calc_results: Dict[str, Any] = {}

task_family = meta.get("task_family", "classification")

if task_family == "fairness_audit" or scenario_key in ("operational_dna_equity", "marketing_conversion_scoring"):
    # Equity Audit (Healthcare Outpatient DNA or Omnichannel Marketing Conversion)
    rows = raw_dict["rows"]
    y_true = [r.get("Did_Not_Attend", r.get("Purchased_Conversion", r.get("Target", 0))) for r in rows]
    y_prob = [r.get("Predicted_DNA_Risk_Score", r.get("Predicted_Conversion_Score", r.get("Predicted_Risk_Score", 0.5))) for r in rows]
    threshold = params.get("threshold", 0.28)
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    groups = [r.get("IMD_Deprivation_Tier", r.get("Customer_Demographic_Tier", r.get("Group", "Tier1"))) for r in rows]

    equity_res = calculate_algorithmic_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_groups=groups,
        favorable_outcome=1
    )
    clf_res = calculate_classification_threshold_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold
    )
    calc_results["equity"] = equity_res
    calc_results["classification"] = clf_res

elif task_family == "time_series" or scenario_key in ("ed_demand_forecasting", "workforce_sickness_forecasting", "supply_chain_demand_forecasting"):
    # Time Series Demand, Sickness Absence, or Supply Chain Dispatch Forecasting
    series = raw_dict["series_values"]
    horizon = params.get("horizon", 14)
    seasonal_period = params.get("seasonal_period", 7)
    alpha = params.get("alpha", 0.28)

    ts_res = calculate_time_series_forecast(
        series=series,
        horizon=horizon,
        seasonal_period=seasonal_period,
        alpha=alpha,
        beta=0.05,
        gamma=0.30
    )
    calc_results["time_series"] = ts_res

elif task_family in ("clustering", "unsupervised") or selected_model_key in ("kmeans_clustering", "dbscan_clustering"):
    # Unsupervised Patient Subtyping & Clustering (K-Means / DBSCAN)
    rows = raw_dict.get("rows", [])
    if rows:
        pts = [
            {
                "X1": float(r.get("Age_Years", r.get("Vibration_mm_s", r.get("Travel_Distance_Miles", r.get("Session_Dwell_Mins", 50.0))))),
                "X2": float(r.get("Baseline_LVEF_Pct", r.get("Operating_Temp_C", r.get("Public_Transit_Time_Mins", r.get("Past_Purchases_12m", 35.0)))))
            }
            for r in rows
        ]
    else:
        pts = [{"X1": 50.0, "X2": 35.0}]

    target_k = params.get("n_clusters", 3)
    clust_res = calculate_cluster_metrics(points=pts, k=target_k)
    calc_results["clustering"] = clust_res

else:
    # Supervised Classification (Clinical Readmission, Manufacturing Failure, Subscriber Churn, Custom)
    rows = raw_dict["rows"]
    y_true = [r.get("Readmitted_30d", r.get("Equipment_Failure", r.get("Subscriber_Churn", r.get("Target", 0)))) for r in rows]
    y_prob = [r.get("Predicted_Risk_Score", r.get("Predicted_Failure_Risk", r.get("Predicted_Churn_Risk", r.get("Predicted_Score", 0.5)))) for r in rows]
    threshold = params.get("threshold", 0.50)

    clf_res = calculate_classification_threshold_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold
    )
    calc_results["classification"] = clf_res

# 7. Methodological Prerequisite Evaluation (5-Point Audit)
render_ml_prerequisite_cards(meta, params)

st.markdown("---")

# 8. Primary Metric Dashboard & Executive KPI Summary
render_ml_kpi_dashboard(scenario_key, meta, params, calc_results)

st.markdown("---")

# 9. Plotly Interactive Visualizers
render_ml_visualizations(scenario_key, meta, params, calc_results)

st.markdown("---")

# 10. Comprehensive 4-Tab Educational Architecture
render_ml_educational_tabs(scenario_key, meta, params, calc_results)

st.markdown("---")
st.caption("NHS Method Finder: Machine Learning & Predictive Analytics Engine | Fictional UK Healthcare Case Studies | Pure Standard Library Verification")

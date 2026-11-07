"""Streamlit Page: Time Series Analysis, Decomposition, Forecasting & Evaluation Metrics.

Provides a comprehensive, interactive learning environment covering:
- Time series anatomy: Trend, Seasonality, Residuals, Stationarity, Autocorrelation (ACF/PACF), and ADF Testing.
- Classical Decomposition: Additive vs. Multiplicative modes.
- Forecasting Model Hierarchy: Naive, Seasonal Naive, Drift, SES, Holt's Damped, Holt-Winters, Lagged Feature ML, and Croston SBA.
- Forecast Evaluation Metrics: MAE, MSE, RMSE, MAPE (division-by-zero risks), sMAPE, MASE, and Directional Accuracy (MDA).
- Outlier & Zero Shock Laboratory: Interactively demonstrating why MAPE fails on zeroes and RMSE explodes on outliers.
- Temporal Validation: Rolling-Origin (Expanding-Window) Cross-Validation preventing data leakage.
"""

import os
import sys
import streamlit as st

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.ui_components.sidebar_navigation import (
    render_custom_sidebar_navigation,
    render_roundtrip_navigation_banner
)
from src.ui_components.time_series_widgets import (
    render_time_series_header,
    render_anatomy_tab,
    render_models_tab,
    render_evaluation_metrics_tab,
    render_validation_tab
)
from src.time_series.ts_engine import (
    generate_ed_demand_series,
    generate_seasonal_flu_series,
    generate_ambulance_callout_series,
    generate_intermittent_stockout_series
)

st.set_page_config(
    page_title="Time Series & Forecasting | MethodFinder",
    page_icon="⏱️",
    layout="wide"
)

# Render Custom Sidebar Navigation
render_custom_sidebar_navigation("pages/7_⏱️_Time_Series_and_Forecasting.py")

# Top Header & Contextual Breadcrumb
render_time_series_header()
render_roundtrip_navigation_banner()

st.info("💡 **Educational Disclaimer:** All operational time series, emergency department attendances, laboratory positivity rates, and surgical inventory records across this module are completely synthetic and fabricated solely for education and methodological training purposes.")

# Scenario Selector
st.markdown("### 🗂️ Step 1: Select Real-World Operational Scenario")
c_scen1, c_scen2 = st.columns([1.5, 2.5])

with c_scen1:
    selected_scenario_id = st.selectbox(
        "Choose an operational time series scenario:",
        options=["ed_demand", "respiratory_flu", "ambulance_surges", "intermittent_supplies"],
        format_func=lambda k: {
            "ed_demand": "🏥 Emergency Department Daily Attendances (Weekly Seasonality + Winter Surge)",
            "respiratory_flu": "🦠 Influenza Surveillance Specimen Positivity Rate (Annual 52-Week Cycle)",
            "ambulance_surges": "🚑 Ambulance Emergency 999 Dispatches (Heatwave Crisis Shock)",
            "intermittent_supplies": "💊 Specialized Surgical Implants & Antidotes (Zero-Inflated Intermittent)"
        }[k],
        key="ts_scenario_master_select"
    )

# Load scenario dataset
if selected_scenario_id == "ed_demand":
    active_dataset = generate_ed_demand_series(days=90, seed=42)
elif selected_scenario_id == "respiratory_flu":
    active_dataset = generate_seasonal_flu_series(weeks=104, seed=101)
elif selected_scenario_id == "ambulance_surges":
    active_dataset = generate_ambulance_callout_series(days=70, seed=202)
else:
    active_dataset = generate_intermittent_stockout_series(days=90, seed=303)

with c_scen2:
    st.markdown(f"**Dataset Focus:** `{active_dataset['title']}`")
    st.markdown(f"*{active_dataset['description']}*")
    st.caption(f"**Sequence Length:** {len(active_dataset['series'])} {active_dataset['frequency']} observations • **Seasonal Period ($m$):** {active_dataset['seasonal_period']} steps • **Units:** {active_dataset['unit']}")

st.markdown("---")

# Main Interactive Learning Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 1. Anatomy, Decomposition & Stationarity",
    "🔮 2. Forecasting Model Hierarchy Sandbox",
    "🎯 3. Evaluation Metrics Deep Dive & Shock Lab",
    "🛡️ 4. Temporal Validation & Rolling-Origin CV"
])

with tab1:
    render_anatomy_tab(active_dataset)

with tab2:
    render_models_tab(active_dataset)

with tab3:
    render_evaluation_metrics_tab(active_dataset)

with tab4:
    render_validation_tab(active_dataset)

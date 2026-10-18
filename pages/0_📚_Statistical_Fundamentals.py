"""Streamlit Page: Statistical Fundamentals & Core Metrics.

Provides zero-prior-knowledge, intuition-first clarity on core statistical foundations
prior to hypothesis testing, structured into 6 focused modular components:
1. 🏷️ Measurement Scales & Data Types (Nominal, Ordinal, Interval, Ratio)
2. ⚖️ Measures of Central Tendency (Mean, Median, Mode, Trimmed Mean)
3. 📏 Measures of Dispersion & Spread (Variance, SD, Quartiles, IQR, Tukey's Fences)
4. 🔬 Standard Deviation (SD) vs. Standard Error of the Mean (SEM)
5. 🔔 Distribution Shapes, Skewness & Central Limit Theorem (CLT)
6. 🎯 Confidence Intervals (Continuous Mean: z vs t; Binary Proportion: Wald vs Wilson vs Agresti-Coull)
"""

import os
import sys
from typing import Dict, Any, Optional
import streamlit as st

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.ui_components.sidebar_navigation import (
    render_custom_sidebar_navigation,
    render_roundtrip_navigation_banner
)
from src.ui_components.fundamentals_widgets import (
    render_scales_data_types_module,
    render_central_tendency_module,
    render_dispersion_iqr_module,
    render_sd_vs_sem_module,
    render_distributions_clt_module,
    render_confidence_intervals_module
)

st.set_page_config(
    page_title="Statistical Fundamentals | MethodFinder",
    page_icon="📚",
    layout="wide"
)

# Render Custom Grouped Sidebar Navigation
render_custom_sidebar_navigation("pages/0_📚_Statistical_Fundamentals.py")

# Render contextual breadcrumb if navigated from Selector
render_roundtrip_navigation_banner()

st.title("📚 Statistical Fundamentals & Core Metrics")
st.markdown(r"""
Master the foundational building blocks of quantitative analysis with **plain-English intuition**, 
real-world clinical analogies, interactive live playgrounds, and our **4-tab educational guides** (Explanation, Excel formulas, Python code, and Practice Data).
""")

# ----------------- Implementation Tracker -----------------
if st.session_state.get("debug_mode", False) or st.session_state.get("debug_model", False):
    with st.expander("📌 Statistical Fundamentals Implementation & Topic Tracker", expanded=False):
        st.markdown("""
        **Project Status: Phase 8 — Statistical Fundamentals & Baseline Concepts (Complete)**  
        This tracker monitors the 6 core foundational modules, interactive visual simulations, and cross-platform guides.
        """)
        st.markdown(r"""
        #### 📚 Core Modules Covered
        * [x] **1. Measurement Scales & Data Types** (Nominal, Ordinal, Interval, Ratio, permitted operations)
        * [x] **2. Measures of Central Tendency** (Mean, Median, Mode, 10% Trimmed Mean, 0.20 SD Skew Pull Rule)
        * [x] **3. Measures of Dispersion & Spread** (Sample Variance $s^2$, Bessel's $N-1$, SD, Quartiles, IQR, Tukey's Fences)
        * [x] **4. Standard Deviation (SD) vs. Standard Error (SEM)** (Biological variability vs. estimator precision, sample size scaling)
        * [x] **5. Distribution Shapes, Skewness & CLT** (Fisher-Pearson skewness, kurtosis, $N \ge 30$ Central Limit Theorem simulation)
        * [x] **6. Confidence Intervals (Mean & Proportion)** (Normal $z$ vs. Student's $t$ with $df = n-1$; Wald vs. Wilson Score & Agresti-Coull)
        * [x] **Sidebar Menu Integration** (Horizontal separator and dedicated `"Fundamentals Topic Selector:"` dropdown)
        * [x] **4-Tab Educational Architecture** (Full formula derivations, Excel cell layouts `A1:F20`, Python scripts, and CSV downloads)
        """)

# ----------------- Topic Switcher Controller -----------------
TOPIC_KEYS = [
    "scales_data_types",
    "central_tendency",
    "dispersion_iqr",
    "sd_vs_sem",
    "distributions_clt",
    "confidence_intervals"
]

TOPIC_LABELS = {
    "scales_data_types": "🏷️ 1. Measurement Scales & Data Types",
    "central_tendency": "⚖️ 2. Central Tendency (Mean, Median, Mode)",
    "dispersion_iqr": "📏 3. Dispersion, Spread & IQR (Tukey's Fences)",
    "sd_vs_sem": "🔬 4. SD vs. Standard Error (SEM)",
    "distributions_clt": "🔔 5. Distribution Shapes, Skewness & CLT",
    "confidence_intervals": "🎯 6. Confidence Intervals (Mean & Proportion)"
}

# Resolve active topic from session state or query
current_active_topic = st.session_state.get("fundamentals_active_topic")
if current_active_topic not in TOPIC_KEYS:
    current_active_topic = TOPIC_KEYS[0]

default_idx = TOPIC_KEYS.index(current_active_topic)

selected_label = st.selectbox(
    "Choose Fundamental Statistical Concept to Explore:",
    options=[TOPIC_LABELS[k] for k in TOPIC_KEYS],
    index=default_idx,
    help="Select an individual concept to inspect its intuition, interactive simulation, Excel guide, Python script, and sample data."
)

# Sync back key
label_to_key = {v: k for k, v in TOPIC_LABELS.items()}
active_key = label_to_key.get(selected_label, TOPIC_KEYS[0])
st.session_state["fundamentals_active_topic"] = active_key

st.markdown("---")

# ----------------- Render Active Concept Module -----------------
if active_key == "scales_data_types":
    render_scales_data_types_module()
elif active_key == "central_tendency":
    render_central_tendency_module()
elif active_key == "dispersion_iqr":
    render_dispersion_iqr_module()
elif active_key == "sd_vs_sem":
    render_sd_vs_sem_module()
elif active_key == "distributions_clt":
    render_distributions_clt_module()
elif active_key == "confidence_intervals":
    render_confidence_intervals_module()

# ----------------- Sequential Topic Walkthrough Footer -----------------
st.markdown("---")
cur_idx = TOPIC_KEYS.index(active_key)
col_prev, col_spacer, col_next = st.columns([1.5, 2.0, 1.5])

with col_prev:
    if cur_idx > 0:
        prev_key = TOPIC_KEYS[cur_idx - 1]
        prev_label = TOPIC_LABELS[prev_key]
        if st.button(f"← Previous: {prev_label.split('.')[1].strip()}", key="btn_prev_topic", use_container_width=True):
            st.session_state["fundamentals_active_topic"] = prev_key
            st.rerun()

with col_next:
    if cur_idx < len(TOPIC_KEYS) - 1:
        next_key = TOPIC_KEYS[cur_idx + 1]
        next_label = TOPIC_LABELS[next_key]
        if st.button(f"Next: {next_label.split('.')[1].strip()} 👉", key="btn_next_topic", use_container_width=True):
            st.session_state["fundamentals_active_topic"] = next_key
            st.rerun()

# ----------------- Quick Navigation Hub -----------------
st.markdown("---")
st.markdown("#### 🧭 Continue Your Learning Pathway:")
c_h1, c_h2, c_h3, c_h4, c_h5, c_h6, c_h7, c_h8 = st.columns(8)
with c_h1:
    st.page_link("app.py", label="Home Overview Hub", icon="🏠")
with c_h2:
    st.page_link("pages/7_🎯_Study_Design_and_Method_Selector.py", label="Method Selector", icon="🎯")
with c_h3:
    st.page_link("pages/8_🗺️_Decision_Flowcharts.py", label="Decision Flowcharts", icon="🗺️")
with c_h4:
    st.page_link("pages/1_📊_Statistical_Tests.py", label="Statistical Tests", icon="📊")
with c_h5:
    st.page_link("pages/2_🏥_Epidemiology_Methods.py", label="Epidemiology Methods", icon="🏥")
with c_h6:
    st.page_link("pages/4_📐_Sample_Size_and_Power.py", label="Sample Size & Power", icon="📐")
with c_h7:
    st.page_link("pages/5_📋_Survey_and_Questionnaire_Analysis.py", label="Survey Analysis", icon="📋")
with c_h8:
    st.page_link("pages/6_🗣️_Natural_Language_Processing.py", label="NLP", icon="🗣️")


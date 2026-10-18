"""Streamlit Page: Survey & Questionnaire Analysis.

Guides researchers, clinicians, and analysts through Likert scale metrics,
internal consistency (Cronbach's Alpha, McDonald's Omega), inter-rater agreement (Cohen's Kappa),
factor analysis dimensionality, finite population survey sizing, and post-stratification weighting.
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
from src.ui_components.survey_widgets import (
    render_survey_mode_selector,
    render_survey_method_module
)
from src.survey_analysis.survey_schema import SURVEY_METHODS

st.set_page_config(
    page_title="Survey & Questionnaire Analysis",
    page_icon="📋",
    layout="wide"
)

# Render Custom Grouped Sidebar Navigation
render_custom_sidebar_navigation("pages/5_📋_Survey_and_Questionnaire_Analysis.py")

# Top Header Banner
st.title("📋 Survey & Questionnaire Analysis")
render_roundtrip_navigation_banner()

st.markdown("""
Navigate the questions below to evaluate **Likert scale distributions**, assess **scale reliability** 
(Cronbach's Alpha vs. McDonald's Omega), measure **inter-rater agreement** (Cohen's Kappa), 
discover **latent subscales** via Factor Analysis, calculate **finite population sample size**, 
or apply **post-stratification weighting** to correct for non-response bias.
""")

# ----------------- Implementation Progress Tracker (Gated behind debug_mode) -----------------
if st.session_state.get("debug_mode", False) or st.session_state.get("debug_model", False):
    with st.expander("📌 Survey Analysis Implementation Tracker", expanded=False):
        st.markdown("""
        | Cluster | Tool / Method | Theoretical Derivation | Excel Implementation | Python Pipeline | Synthetic Data Generator | Status |
        | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
        | **1. Scales** | Likert Scale Analysis (Ordinal vs Interval) | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **2. Reliability** | Cronbach's Alpha (α) & Item-Total Purification | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **2. Reliability** | McDonald's Omega (ω) & Tau-Equivalence | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **3. Agreement** | Cohen's Kappa (κ) & Quadratic Weighted Kappa | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **4. Validity** | Exploratory Factor Analysis (EFA) & Scree Plot | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **5. Sizing** | Survey Sample Size with FPC & Response Buffer | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **5. Weighting** | Post-Stratification Weighting & Kish Neff | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        """)

st.markdown("---")

# Navigation Mode & Method Execution
nav_mode, selected_method_key = render_survey_mode_selector()
render_survey_method_module(selected_method_key)


"""Streamlit Page: Longitudinal & Hierarchical Mixed-Effects Models (Multilevel Data).

Pillar 12 of Method Finder.
Guides clinical trialists, biostatisticians, epidemiologists, and data scientists through:
1. Linear Mixed-Effects Models (LMM): Random Intercepts & Slopes (Laird & Ware)
2. Generalized Linear Mixed Models (GLMM): Repeated Binary & Count Outcomes
3. Repeated Measures ANOVA vs. Mixed Models: Sphericity & Missing Data Tradeoffs
4. Hierarchical 3-Level Clustered Models: Patients within Clinicians within Hospitals
5. Covariance Structure Selection & Model Diagnostics: ML vs. REML, AIC & Nakagawa R²
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
from src.ui_components.mixed_widgets import (
    render_mixed_plain_english_guide,
    render_mixed_mode_selector,
    render_mixed_data_method_module
)
from src.mixed_models.mixed_schema import MIXED_MODELS_METHODS

st.set_page_config(
    page_title="Longitudinal & Mixed Models | MethodFinder",
    page_icon="🪜",
    layout="wide"
)

# Render Custom Sidebar Navigation
render_custom_sidebar_navigation("pages/12_🪜_Longitudinal_and_Mixed_Models.py")

# Top Header & Contextual Breadcrumb
st.title("🪜 Longitudinal & Hierarchical Mixed-Effects Models (Multilevel Data)")
render_roundtrip_navigation_banner()

st.info("💡 **Educational Disclaimer:** All cognitive trial decline scores, ICU ward septic relapse events, physiotherapy VAS pain ratings, NHS 3-level surgical complications, and serial cardiac troponin draws across this module are completely synthetic and simulated solely for educational and methodological training purposes.")

st.markdown("""
Welcome to the **Longitudinal & Hierarchical Mixed-Effects Models** module. In medical science, clinical trials, and healthcare 
quality audits, data is rarely composed of independent individuals. Patients are followed repeatedly across time, and individuals 
are naturally nested inside care environments (wards, clinics, general practices, and hospital trusts).

This module delivers publication-grade mixed-effects architectures aligned with **BMJ**, **JAMA**, **NEJM**, and the **ICH E9** 
international regulatory standard. Master the Laird & Ware formulation of random intercepts and slopes; unpack why Generalized Linear 
Mixed Models (GLMM) produce subject-specific odds ratios that differ from marginal population averages; discover why mixed models 
dominate classical Repeated Measures ANOVA by recovering 100% of patients with missing visits; partition variance across 3-level 
organizational hierarchies; and evaluate model goodness-of-fit using the **Nakagawa & Schielzeth (2013) marginal and conditional $R^2$** metrics.
""")

# 1. Plain English Zero-Knowledge Guide
render_mixed_plain_english_guide()

st.markdown("---")

# 2. Navigation Mode Selector & Execution Controller
nav_mode, selected_method_key = render_mixed_mode_selector()

# 3. Comprehensive Method Module Execution
render_mixed_data_method_module(selected_method_key)

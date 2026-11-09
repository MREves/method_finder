"""Streamlit Page: Missing Data Mechanisms, Imputation & Sensitivity Audits.

Pillar 11 of Method Finder.
Guides clinical trialists, biostatisticians, epidemiologists, and data scientists through:
1. Rubin's Missingness Taxonomy & Diagnostic Audits (MCAR vs. MAR vs. MNAR, Little's Test)
2. Complete Case Analysis (Listwise Deletion) vs. Single Imputation Hazards (Variance Collapse & LOCF)
3. Multiple Imputation by Chained Equations (MICE) & Rubin's Rules Pooling (Q-bar, U-bar, B, T, FMI)
4. Algorithmic & Distance-Based Imputation (k-NN & Random Forest MissForest)
5. MNAR Sensitivity Audits: Pattern-Mixture & Delta-Adjustment Tipping Point Analysis (ICH E9 R1)
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
from src.ui_components.missing_widgets import (
    render_missing_plain_english_guide,
    render_missing_mode_selector,
    render_missing_data_method_module
)
from src.missing_data.missing_schema import MISSING_DATA_METHODS

st.set_page_config(
    page_title="Missing Data & Imputation | MethodFinder",
    page_icon="🧩",
    layout="wide"
)

# Render Custom Sidebar Navigation
render_custom_sidebar_navigation("pages/11_🧩_Missing_Data_and_Imputation.py")

# Top Header & Contextual Breadcrumb
st.title("🧩 Missing Data Mechanisms, Imputation & Sensitivity Audits")
render_roundtrip_navigation_banner()

st.info("💡 **Educational Disclaimer:** All ICU arterial lactate values, multivariable cardiovascular risk factors, diabetes HbA1c trial cohorts, and depression dropout tipping point simulations across this module are completely synthetic and fabricated solely for educational and methodological training purposes.")

st.markdown("""
Welcome to the **Missing Data Mechanisms, Imputation & Sensitivity Audits** module. Missing data is an unavoidable reality in clinical 
trials, observational epidemiology, and electronic health registries. How you handle missing data determines whether your statistical 
conclusions are rigorous and reproducible, or severely distorted by selection bias and manufactured Type I errors.

This module provides publication-grade missing data frameworks aligned with **BMJ**, **JAMA**, **NEJM**, and the **ICH E9 (R1)** 
international regulatory standard. Diagnose whether your data is MCAR, MAR, or MNAR; eliminate the hazards of complete-case deletion 
and mean imputation; master Multiple Imputation by Chained Equations (MICE) and Rubin's Rules; and stress-test trial conclusions against 
informative dropout using pattern-mixture tipping point analysis.
""")

# 1. Plain English Zero-Knowledge Guide
render_missing_plain_english_guide()

st.markdown("---")

# 2. Navigation Mode Selector & Execution Controller
nav_mode, selected_method_key = render_missing_mode_selector()

# 3. Comprehensive Method Module Execution
render_missing_data_method_module(selected_method_key)

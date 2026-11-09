"""Streamlit Page: Health Economics & Health Technology Assessment (HTA / NICE).

Pillar 10 of Method Finder.
Guides health economists, clinical trialists, market access leads, and NHS commissioners through:
1. Incremental Cost-Effectiveness Ratio (ICER) & The 4-Quadrant Cost-Effectiveness Plane
2. Net Monetary Benefit (NMB) & Net Health Benefit (NHB) Linear Multi-Arm Ranking
3. Markov Decision Analytic Cohort Models (State Transitions, Half-Cycle Correction & 3.5% NICE Discounting)
4. Probabilistic Sensitivity Analysis (PSA) & Cost-Effectiveness Acceptability Curves (CEAC) & EVPI
5. Budget Impact Analysis (BIA) & NHS 5-Year Payer Affordability Forecasting
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
from src.ui_components.hta_widgets import (
    render_hta_plain_english_guide,
    render_hta_mode_selector,
    render_hta_method_module
)
from src.health_economics.hta_schema import HTA_METHODS

st.set_page_config(
    page_title="Health Economics & HTA | MethodFinder",
    page_icon="💷",
    layout="wide"
)

# Render Custom Sidebar Navigation
render_custom_sidebar_navigation("pages/10_💷_Health_Economics_and_HTA.py")

# Top Header & Contextual Breadcrumb
st.title("💷 Health Economics & Health Technology Assessment (HTA / NICE)")
render_roundtrip_navigation_banner()

st.info("💡 **Educational Disclaimer:** All biologic drug acquisition costs, surgical implant prices, cancer progression Markov transition matrices, and NHS CKD budget impact figures across this module are completely synthetic and fabricated solely for educational and health economic training purposes.")

st.markdown("""
Welcome to the **Health Economics & Health Technology Assessment (HTA / NICE)** module. In taxpayer-funded healthcare systems 
such as the UK NHS, resources are finite. Every pound committed to a novel intervention incurs an **opportunity cost**: the health 
benefits that would have been generated elsewhere in the healthcare system had those funds been deployed elsewhere.

This module provides publication-grade health economic decision models aligned with the **NICE Health Technology Appraisal Reference Case**, 
**CADTH**, and **ISPOR** international standards. Master the mathematical derivation of Incremental Cost-Effectiveness Ratios (ICER), 
linear Net Monetary Benefit ranking, 10-year Markov cohort state-transition modeling with 3.5% discounting, Monte Carlo Probabilistic 
Sensitivity Analysis (PSA) with CEAC curves, and 5-year Budget Impact Analysis (BIA) for healthcare commissioners.
""")

# 1. Plain English Zero-Knowledge Guide
render_hta_plain_english_guide()

st.markdown("---")

# 2. Navigation Mode Selector & Execution Controller
nav_mode, selected_method_key = render_hta_mode_selector()

# 3. Comprehensive Method Module Execution
render_hta_method_module(selected_method_key)

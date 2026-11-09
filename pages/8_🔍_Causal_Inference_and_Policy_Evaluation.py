"""Streamlit Page: Causal Inference & Quasi-Experimental Policy Evaluation.

Pillar 8 of Method Finder.
Guides researchers, clinicians, and health policy analysts through:
1. Difference-in-Differences (DiD) & Event Studies (Parallel Trends & Policy Shocks)
2. Interrupted Time Series (ITS) & Segmented Regression (Level Shifts & Trajectory Changes)
3. Propensity Score Matching (PSM) & Inverse Probability Weighting (IPTW)
4. Regression Discontinuity Design (RDD: Sharp & Fuzzy Local Treatment Effects)
5. Instrumental Variables (IV) & Mendelian Randomization (MR: Overcoming Unmeasured Confounding)
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
from src.ui_components.causal_widgets import (
    render_causal_plain_english_guide,
    render_causal_mode_selector,
    render_causal_method_module
)
from src.causal_inference.causal_schema import CAUSAL_METHODS

st.set_page_config(
    page_title="Causal Inference & Policy Evaluation | MethodFinder",
    page_icon="🔍",
    layout="wide"
)

# Render Custom Sidebar Navigation
render_custom_sidebar_navigation("pages/8_🔍_Causal_Inference_and_Policy_Evaluation.py")

# Top Header & Contextual Breadcrumb
st.title("🔍 Causal Inference & Quasi-Experimental Policy Evaluation")
render_roundtrip_navigation_banner()

st.info("💡 **Educational Disclaimer:** All hospital panel mortality figures, opioid prescribing time series, surgical cohort records, biomarker threshold cutoffs, and genetic alleles across this module are completely synthetic and fabricated solely for educational and methodological training purposes.")

st.markdown("""
Welcome to the **Causal Inference & Quasi-Experimental Evaluation** module. In real-world healthcare and public policy, 
randomized controlled trials (RCTs) are often unfeasible, unethical, or prohibitively expensive. This module equips you 
with the econometric and epidemiological tools to rigorously infer **cause-and-effect relationships from observational data**, 
eliminating confounding by indication, isolating true policy shocks from background secular drift, and auditing identification assumptions.
""")

# 1. Plain English Zero-Knowledge Guide
render_causal_plain_english_guide()

st.markdown("---")

# 2. Navigation Mode Selector & Execution Controller
nav_mode, selected_method_key = render_causal_mode_selector()

# 3. Comprehensive Method Module Execution
render_causal_method_module(selected_method_key)

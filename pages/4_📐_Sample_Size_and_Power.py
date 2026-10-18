"""Streamlit Page: Sample Size & Statistical Power Calculations.

Biostatistical design and power analysis platform for clinical trials,
epidemiological investigations, diagnostic accuracy validations, and cluster-randomized studies.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.ui_components.sample_size_widgets import render_sample_size_module
from src.ui_components.sidebar_navigation import (
    render_custom_sidebar_navigation,
    render_roundtrip_navigation_banner
)


st.set_page_config(
    page_title="Sample Size & Statistical Power | MethodFinder",
    page_icon="📐",
    layout="wide"
)

# Render Custom Grouped Sidebar Navigation
render_custom_sidebar_navigation("pages/4_📐_Sample_Size_and_Power.py")

st.title("📐 Sample Size & Statistical Power Calculations")
render_roundtrip_navigation_banner()
st.markdown("""
Plan, size, and justify your clinical trial, observational epidemiology study, or diagnostic evaluation with biostatistical rigor.
Features closed-form mathematical sizing, interactive power curves, sensitivity tables, and automated publication-ready protocol justification statements for NIHR, MRC, and NHS REC / IRB submissions.
""")

# ----------------- Implementation & Method Tracker -----------------
if st.session_state.get("debug_mode", False) or st.session_state.get("debug_model", False):
    with st.expander("📌 Sample Size & Statistical Power Implementation Tracker", expanded=False):
        st.markdown("""
        **Project Status: Phase 4 — Sample Size & Statistical Power (Complete)**  
        This tracker monitors the 10 biostatistical calculation frameworks across 4 outcome families, trial modifiers, and educational guides.
        """)
        st.markdown(r"""
        #### 📏 Family 1: Continuous Outcome Measures
        * [x] **Two Independent Means (Two-Sample t-Test)** (Equal and unequal allocation $k = n_2/n_1$, Cohen's $d$)
        * [x] **Paired / Repeated Measures (Paired t-Test)** (Mean differences and within-pair variance, Cohen's $d_z$)
        * [x] **Multiple Groups (One-Way ANOVA)** (Cohen's $f$ effect size, numerator/denominator degrees of freedom)
        * [x] **Linear Correlation (Pearson's r)** (Fisher's $z$-transformation, normal approximation)

        #### 🎲 Family 2: Binary Outcomes & Epidemiological Risk Metrics
        * [x] **Two Independent Proportions** (Fleiss unpooled asymptotic & Casagrande-Pike continuity correction)
        * [x] **Prospective Cohort Study (Relative Risk RR)** (Kelsey & Fleiss formulation with baseline incidence $p_0$)
        * [x] **Retrospective Case-Control Study (Odds Ratio OR)** (Control exposure $p_0$ and $1:k$ matching ratios)
        * [x] **Matched Pairs (McNemar Discordant Pairs)** (Proportion of discordant pairs and exposure odds)

        #### ⏱️ Family 3: Longitudinal Survival Outcomes (Time-to-Event)
        * [x] **Cox Proportional Hazards Log-Rank Test** (Schoenfeld 1981 required event formula $E$)
        * [x] **Event-to-Enrollment Accrual Conversion** (Cumulative event probability $P(\text{Event})$, patient recruitment $N$)

        #### 🎯 Family 4: Diagnostic & Screening Accuracy Studies
        * [x] **Target Sensitivity / Specificity Estimation** (Wilson score margin of error $W$, 95% confidence intervals)
        * [x] **Prevalence-Adjusted Screening Cohort** ($N_{\text{total}} = n_{\text{diseased}} / \text{Prevalence}$)

        #### 🛠️ Trial Design Adjustments & Real-World Pragmatics
        * [x] **Attrition & Loss-to-Follow-Up Inflation** ($N_{\text{recruited}} = \lceil N_{\text{evaluable}} / (1 - d) \rceil$)
        * [x] **Cluster Randomization Design Effect** ($\text{VIF} = 1 + (m - 1) \times \text{ICC}$, required cluster count $C$)

        #### 📊 Interactive Visualizations & Protocol Generator
        * [x] **Plotly Power Curves** (Power vs. $N$ across Planned, Small, and Large effect size benchmarks)
        * [x] **Plotly Sensitivity Curves** (Minimum Detectable Effect Size MDES vs. $N$)
        * [x] **Automated Protocol Justification Generator** (CONSORT 2010 Item 7a, STROBE, NIHR / MRC / IRB Section 6 text)
        * [x] **Full 4-Tab Educational Architecture** (Derivations, Excel `A1:P20` Guide, Python Pipelines, Synthetic Data Downloads)
        """)

# ----------------- Sidebar Reference Context -----------------
with st.sidebar:
    st.markdown("### 📐 Study Design & Sizing")
    st.info("""
    **Core Guidelines Followed:**
    * **ICH-E9**: Statistical Principles for Clinical Trials
    * **CONSORT 2010**: Item 7a (Sample Size Determination)
    * **STROBE**: Observational Epidemiology Study Sizing
    * **NIHR / MRC**: UK Health Research Protocol Standards
    """)
    st.caption("All calculation algorithms utilize closed-form biostatistical formulations with high-precision normal quantiles.")


# ----------------- Render Main Sample Size Module -----------------
render_sample_size_module()


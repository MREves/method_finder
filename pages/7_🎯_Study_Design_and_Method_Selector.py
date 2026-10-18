"""Streamlit Page: Universal Study Design & Method Selector.

Translates real-world research problems, trial architectures, operational constraints,
and governance requirements into the optimal analytical framework across Classical Statistics,
Epidemiological Methods, Machine Learning, and Sample Size & Statistical Power Calculations.
"""

import os
import sys
from typing import Dict, Any, Optional
import streamlit as st

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.selector.study_design_schema import (
    DOMAIN_PROFILES,
    PROBLEM_DEFINITIONS,
    PROBLEM_BLUEPRINTS
)
from src.selector.selector_engine import (
    get_problem_blueprint,
    route_universal_method_recommendation
)
from src.ui_components.selector_widgets import (
    render_selector_plain_english_guide,
    render_problem_blueprints_bar,
    render_problem_definition_form,
    render_recommendation_card,
    render_decision_path_flowchart,
    render_methodological_comparison_matrix,
    render_study_protocol_checklist
)
from src.ui_components.sidebar_navigation import render_custom_sidebar_navigation

st.set_page_config(
    page_title="Study Design & Method Selector | MethodFinder",
    page_icon="🎯",
    layout="wide"
)

# Render Custom Grouped Sidebar Navigation
render_custom_sidebar_navigation("pages/7_🎯_Study_Design_and_Method_Selector.py")

st.title("🎯 Universal Study Design & Method Selector")
st.markdown("""
Map your clinical, epidemiological, industrial, marketing, or operational research question to the mathematically justified analytical framework.
This universal selector coordinates across **Classical Statistics**, **Epidemiology Methods**, **Machine Learning**, and **Sample Size Determination**, providing cross-pillar comparisons, protocol verification checklists, and one-click deep links directly into focused training modules.
""")

# ----------------- Implementation & Method Tracker -----------------
if st.session_state.get("debug_mode", False) or st.session_state.get("debug_model", False):
    with st.expander("📌 Universal Study Design & Selector Implementation Tracker", expanded=False):
        st.markdown("""
        **Project Status: Universal Method Selector & Study Design Engine (Complete)**  
        This tracker monitors the 14 universal problem classifications, 8 real-world blueprints, 5-point protocol audits, and cross-pillar deep-linking continuity.
        """)
        st.markdown(r"""
        #### 🌐 Cross-Pillar Quantitative Decision Graph
        * [x] **Pillar 1: Classical Statistical Hypothesis Testing** (19 parametric and non-parametric tests)
        * [x] **Pillar 2: Observational Epidemiology & Public Health** (14 risk ratios, standardization, screening & meta-analysis)
        * [x] **Pillar 3: Machine Learning & Operational Analytics** (16 architectures, equity audits, time series & clustering)
        * [x] **Pillar 4: Biostatistical Sample Size & Statistical Power** (10 closed-form formulas, attrition & cluster cushions)

        #### ⚡ Quick-Start Real-World Blueprints
        * [x] **All Meadows Phase III Diabetes RCT** (Welch's t-test $\rightarrow$ Two Means Sample Size)
        * [x] **Riverside Outbreak Investigation** (Case-Control OR $\rightarrow$ Control Matching Sizing)
        * [x] **Inner City Sepsis ICU Triage** (Random Forest $\rightarrow$ PR-AUC / Brier Score)
        * [x] **NHS ICB Outpatient DNA Equity Audit** (Algorithmic Fairness $\rightarrow$ Four-Fifths 80% Rule)
        * [x] **Apex Precision Industrial Spindle Failure** (Gradient Boosting $\rightarrow$ Rare Class Handling)
        * [x] **MetroLogix Regional Freight Surge** (Lagged Ensemble Time Series $\rightarrow$ Rolling Origin CV)
        * [x] **OmniChannel Marketing Disparate Impact** (Fairness Audit $\rightarrow$ Demography Disparate Impact)
        * [x] **PulseStream SaaS Subscriber Churn** (Elastic Net Logistic $\rightarrow$ Odds Ratio Interpretability)

        #### 🛡️ Protocol Governance & Educational Matrix
        * [x] **5-Point Methodological Protocol Checklist** (CONSORT 2010, STROBE, TRIPOD, NHS AI Ethics standards)
        * [x] **3-Pillar Pedagogical Comparison Matrix** (Statistics vs. Epidemiology vs. Machine Learning)
        * [x] **One-Click Deep-Linking Navigation** (Preserves isolated training environments while auto-configuring pre-selected state)
        """)

# ----------------- Quick Navigation Hub -----------------
c_nav1, c_nav2, c_nav3, c_nav4, c_nav5, c_nav6, c_nav7, c_nav8 = st.columns(8)
with c_nav1:
    st.page_link("app.py", label="Home Overview Hub", icon="🏠")
with c_nav2:
    st.page_link("pages/8_🗺️_Decision_Flowcharts.py", label="Decision Flowcharts", icon="🗺️")
with c_nav3:
    st.page_link("pages/1_📊_Statistical_Tests.py", label="Statistical Tests", icon="📊")
with c_nav4:
    st.page_link("pages/2_🏥_Epidemiology_Methods.py", label="Epidemiology Methods", icon="🏥")
with c_nav5:
    st.page_link("pages/3_🤖_Machine_Learning.py", label="Machine Learning", icon="🤖")
with c_nav6:
    st.page_link("pages/4_📐_Sample_Size_and_Power.py", label="Sample Size & Power", icon="📐")
with c_nav7:
    st.page_link("pages/5_📋_Survey_and_Questionnaire_Analysis.py", label="Survey Analysis", icon="📋")
with c_nav8:
    st.page_link("pages/6_🗣️_Natural_Language_Processing.py", label="NLP", icon="🗣️")

st.markdown("---")

# ----------------- Sidebar Reference Context -----------------
with st.sidebar:
    st.markdown("### 🎯 Study Design Framework")
    st.info("""
    **Core Reporting & Design Standards:**
    * **CONSORT 2010**: Parallel Group Randomized Trials
    * **STROBE**: Observational Studies in Epidemiology
    * **TRIPOD**: Multivariable Clinical Prediction Models
    * **ICH-E9**: Statistical Principles for Clinical Trials
    * **NHS AI Lab Ethics**: Algorithmic Fairness & Health Equity
    """)
    st.caption("Cross-pillar recommendations evaluate parametric assumptions, sample magnitude, confounding risk, and explainability priorities.")

# 1. Plain English Pedagogical Guide
render_selector_plain_english_guide()

st.markdown("---")

# 2. Real-World Problem Blueprints Bar
render_problem_blueprints_bar()

# Active Blueprint Indicator & Reset Control
active_bp_key = st.session_state.get("active_blueprint_key")
active_bp = None
if active_bp_key:
    active_bp = get_problem_blueprint(active_bp_key)
    if active_bp:
        col_b1, col_b2 = st.columns([4, 1])
        with col_b1:
            dom_name = active_bp.get("domain_name") or DOMAIN_PROFILES.get(active_bp.get("domain", ""), {}).get("name", active_bp.get("domain", "Domain"))
            st.success(
                f"⚡ **Active Blueprint Loaded:** {active_bp.get('icon', '🎯')} **{active_bp.get('title', 'Blueprint')}** — "
                f"*{active_bp.get('organisation', '')} ({dom_name})*"
            )
        with col_b2:
            if st.button("🔄 Reset Blueprint", key="btn_reset_blueprint", use_container_width=True):
                st.session_state["active_blueprint_key"] = None
                st.session_state["last_loaded_blueprint_key"] = "__RESET__"
                active_bp = None
                st.rerun()

# 3. Interactive Problem Definition Form
form_inputs = render_problem_definition_form(active_blueprint=active_bp)

# 4. Route Recommendation
rec = route_universal_method_recommendation(
    domain=form_inputs["domain"],
    analytical_goal=form_inputs["analytical_goal"],
    study_design=form_inputs["study_design"],
    sample_scale=form_inputs["sample_scale"],
    problem_key=form_inputs["problem_key"]
)

# 5. Top Recommendation Card with Deep-Link Launchers
render_recommendation_card(rec)

# 6. Interactive Visual Decision Pathway Flowchart
render_decision_path_flowchart(rec, form_inputs)

# 7. Pedagogical Cross-Pillar Comparison Matrix
render_methodological_comparison_matrix(rec["comparison"])

# 8. Methodological Study Protocol Verification Checklist
render_study_protocol_checklist(rec["checklist"])


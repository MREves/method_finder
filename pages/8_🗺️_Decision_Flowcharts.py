"""Streamlit Page: Methodology Decision Flowcharts & Process Explainer.

Provides a standalone, bird's-eye architectural roadmap and interactive visual
decision trees across:
1. Master Cross-Pillar Triage (Statistics vs Epidemiology vs Machine Learning)
2. Classical Statistical Hypothesis Testing (18 Tests, Parametric & Non-Parametric)
3. Observational Epidemiology & Public Health (17 Methods, Risk, Rates & Screening)
4. Machine Learning & Predictive Modeling (16 Architectures, Rare Events & Clustering)
5. Biostatistical Sample Size & Power Planning (10 Frameworks, Attrition & Clusters)
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
from src.ui_components.flowchart_widgets import (
    render_master_triage_flowchart,
    render_statistics_decision_tree,
    render_epidemiology_decision_tree,
    render_machine_learning_decision_tree,
    render_sample_size_decision_tree
)

st.set_page_config(
    page_title="Methodology Decision Flowcharts | MethodFinder",
    page_icon="🗺️",
    layout="wide"
)

# Render Custom Grouped Sidebar Navigation
render_custom_sidebar_navigation("pages/8_🗺️_Decision_Flowcharts.py")

# Render contextual breadcrumb if navigated from Selector
render_roundtrip_navigation_banner()

st.title("🗺️ Methodology Decision Flowcharts & Process Explainer")
st.markdown(r"""
A comprehensive, bird's-eye architectural roadmap across **Classical Statistics**, **Observational Epidemiology**, 
**Machine Learning**, and **Sample Size Determination**. 

Follow the visual decision trees below to understand how data types, study designs, sample scale boundaries ($N < 30$ vs. $N \ge 30$), 
and assumption violations guide you to the mathematically justified analytical method.
""")

# ----------------- Implementation Tracker -----------------
if st.session_state.get("debug_mode", False) or st.session_state.get("debug_model", False):
    with st.expander("📌 Decision Flowchart Architecture & Pillar Coverage Tracker", expanded=False):
        st.markdown("""
        **Project Status: Standalone Methodology Process Explainer (Complete)**  
        This module provides the global architectural perspective of all quantitative methods available in MethodFinder.
        """)
        st.markdown(r"""
        #### 🌐 Comprehensive Decision Trees & Flowcharts Covered
        * [x] **Master Cross-Pillar Triage Router**: High-level triage guiding between hypothesis testing, observational risk, predictive modeling, and sample size planning.
        * [x] **Classical Statistics Decision Tree**: 18 parametric and non-parametric tests, Central Limit Theorem gates ($N \ge 30$), and outlier fallbacks.
        * [x] **Observational Epidemiology Decision Tree**: 17 methods spanning prospective cohorts (RR, IRR, AR), retrospective case-controls (OR, Mantel-Haenszel), cross-sectional diagnostic accuracy (Sens, Spec, ROC), demographic standardization (DSR, SMR), and meta-analysis.
        * [x] **Machine Learning Architecture Tree**: 16 models across regularized regression, severe class imbalance ($< 10\%$), algorithmic fairness audits (Four-Fifths Rule), sequential time series, and unsupervised clustering evaluated via **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Index**.
        * [x] **Biostatistical Sample Size Planning Tree**: 10 analytical sizing frameworks, minimal clinically important differences (MCID), attrition cushions ($N / (1-L)$), and cluster randomization variance inflation ($\text{VIF} = 1 + (m-1)\rho$).
        * [x] **Interactive Simulators & 1-Click Launchers**: Direct deep links from leaf nodes into individual training modules.
        """)

# ----------------- Quick Navigation Hub -----------------
c_nav1, c_nav2, c_nav3, c_nav4, c_nav5, c_nav6, c_nav7, c_nav8 = st.columns(8)
with c_nav1:
    st.page_link("app.py", label="Home Overview Hub", icon="🏠")
with c_nav2:
    st.page_link("pages/7_🎯_Study_Design_and_Method_Selector.py", label="Study Design Selector", icon="🎯")
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

# ----------------- 5 Interactive Pillar Tabs -----------------
tab_master, tab_stat, tab_epi, tab_ml, tab_power = st.tabs([
    "🌐 Master Cross-Pillar Triage",
    "📊 Classical Statistics Tree",
    "🏥 Epidemiology Tree",
    "🤖 Machine Learning Tree",
    "📐 Sample Size & Power Tree"
])

with tab_master:
    render_master_triage_flowchart()

with tab_stat:
    render_statistics_decision_tree()

with tab_epi:
    render_epidemiology_decision_tree()

with tab_ml:
    render_machine_learning_decision_tree()

with tab_power:
    render_sample_size_decision_tree()

st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.85rem; color: #64748b; padding: 1rem 0;">
    🧭 <strong>MethodFinder Quantitative Architecture</strong> • Cross-Pillar Decision Support & Educational Guidance
</div>
""", unsafe_allow_html=True)

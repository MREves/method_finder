"""Streamlit Page: Methodology Decision Flowcharts & Process Explainer.

Provides a standalone, bird's-eye architectural roadmap and interactive visual
decision trees across:
1. Master Cross-Pillar Triage (Statistics vs Epidemiology vs Machine Learning vs NLP vs Causal Inference)
2. Classical Statistical Hypothesis Testing (18 Tests, Parametric & Non-Parametric)
3. Observational Epidemiology & Public Health (17 Methods, Risk, Rates & Screening)
4. Causal Inference & Quasi-Experimental Evaluation (5 Architectures, Policy Shocks & Endogeneity)
5. Machine Learning & Predictive Modeling (16 Architectures, Rare Events & Clustering)
6. Biostatistical Sample Size & Power Planning (10 Frameworks, Attrition & Clusters)
7. Qualitative Analysis & Natural Language Processing (7 Methods, Rule-Based to LLMs)
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
    render_causal_inference_decision_tree,
    render_machine_learning_decision_tree,
    render_sample_size_decision_tree,
    render_nlp_decision_tree,
    render_spc_qi_decision_tree,
    render_hta_decision_tree,
    render_missing_data_decision_tree,
    render_mixed_models_decision_tree
)

st.set_page_config(
    page_title="Methodology Decision Flowcharts | MethodFinder",
    page_icon="🗺️",
    layout="wide"
)

# Render Custom Grouped Sidebar Navigation
render_custom_sidebar_navigation("pages/14_🗺️_Decision_Flowcharts.py")

# Render contextual breadcrumb if navigated from Selector
render_roundtrip_navigation_banner()

st.title("🗺️ Methodology Decision Flowcharts & Process Explainer")
st.markdown(r"""
A comprehensive, bird's-eye architectural roadmap across **Classical Statistics**, **Observational Epidemiology**, 
**Causal Inference & Policy Evaluation**, **Machine Learning**, **Sample Size Determination**, and **Qualitative Analysis (Natural Language Processing)**. 

Follow the visual decision trees below to understand how data types, study designs, sample scale boundaries ($N < 30$ vs. $N \ge 30$), 
annotation availability, and assumption violations guide you to the mathematically and architecturally justified analytical method.
""")

# ----------------- Implementation Tracker -----------------
if st.session_state.get("debug_mode", False) or st.session_state.get("debug_model", False):
    with st.expander("📌 Decision Flowchart Architecture & Pillar Coverage Tracker", expanded=False):
        st.markdown("""
        **Project Status: Standalone Methodology Process Explainer (Complete)**  
        This module provides the global architectural perspective of all quantitative and qualitative methods available in MethodFinder.
        """)
        st.markdown(r"""
        #### 🌐 Comprehensive Decision Trees & Flowcharts Covered
        * [x] **Master Cross-Pillar Triage Router**: High-level triage guiding between hypothesis testing, observational risk, causal policy evaluation, predictive modeling, sample size planning, and qualitative text analysis.
        * [x] **Classical Statistics Decision Tree**: 18 parametric and non-parametric tests, Central Limit Theorem gates ($N \ge 30$), and outlier fallbacks.
        * [x] **Observational Epidemiology Decision Tree**: 18 methods spanning prospective cohorts (RR, IRR, AR), retrospective case-controls (OR, Mantel-Haenszel), cross-sectional diagnostic accuracy (Sens, Spec, ROC), demographic standardization (DSR, SMR), health inequalities (SII, RII), and meta-analysis.
        * [x] **Causal Inference & Policy Evaluation Tree**: 5 quasi-experimental architectures spanning Difference-in-Differences (DiD), Interrupted Time Series (ITS), Propensity Score Matching (PSM / IPTW), Regression Discontinuity (Sharp & Fuzzy RDD), and Instrumental Variables (IV / 2SLS & Mendelian Randomization).
        * [x] **Statistical Process Control & QI Tree**: 5 healthcare QI architectures spanning Healthcare Run Charts (Shift, Trend, Runs, Astronomical Point), Shewhart XmR / I-MR, Laney p' & u' overdispersion corrections (sigma_z), and NHS Making Data Count capability (Cp, Cpk).
        * [x] **Health Economics & HTA Tree**: 5 health economic architectures spanning ICER & Cost-Effectiveness Plane, Net Monetary Benefit (NMB) linear ranking, 10-year Markov state-transition cohort models, Monte Carlo PSA with CEAC curves, and 5-year NHS Budget Impact Analysis (BIA).
        * [x] **Missing Data & Imputation Tree**: 5 missing data architectures spanning Rubin's Taxonomy & Little's MCAR audit, Complete Case retention collapse & mean imputation variance deflation, MICE with Rubin's Rules pooling, non-parametric k-NN / MissForest, and ICH E9 (R1) pattern-mixture tipping point analysis.
        * [x] **Longitudinal & Mixed Models Tree**: 5 repeated measures architectures spanning Laird & Ware LMM with random intercepts & slopes, GLMM non-collapsible odds ratios, RM-ANOVA sphericity & missing data recovery, 3-level hierarchical organizational variance partitioning, and Nakagawa marginal/conditional R2.
        * [x] **Machine Learning Architecture Tree**: 16 models across regularized regression, severe class imbalance ($< 10\%$), algorithmic fairness audits (Four-Fifths Rule), sequential time series, and unsupervised clustering evaluated via **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Index**.
        * [x] **Biostatistical Sample Size Planning Tree**: 10 analytical sizing frameworks, minimal clinically important differences (MCID), attrition cushions ($N / (1-L)$), and cluster randomization variance inflation ($\text{VIF} = 1 + (m-1)\rho$).
        * [x] **Qualitative Analysis & NLP Decision Tree**: 7 methods spanning 5 evolutionary levels from rule-based VADER & Clinical NegEx, TF-IDF, Naive Bayes / Linear SVM, LDA topic modeling, and 384D dense embeddings to Clinical NER and LLM Zero-Shot RAG with Faithfulness audits.
        * [x] **Interactive Simulators & 1-Click Launchers**: Direct deep links from leaf nodes into individual training modules.
        """)

# ----------------- 11 Interactive Pillar Tabs -----------------
tab_master, tab_stat, tab_epi, tab_causal, tab_spc, tab_hta, tab_missing, tab_mixed, tab_ml, tab_power, tab_nlp = st.tabs([
    "🌐 Master Cross-Pillar Triage",
    "📊 Classical Statistics Tree",
    "🏥 Epidemiology Tree",
    "🔍 Causal Inference Tree",
    "📈 SPC & Quality Improvement Tree",
    "💷 Health Economics & HTA Tree",
    "🧩 Missing Data & Imputation Tree",
    "🪜 Longitudinal & Mixed Models Tree",
    "🤖 Machine Learning Tree",
    "📐 Sample Size & Power Tree",
    "🗣️ Qualitative Analysis (NLP) Tree"
])

with tab_master:
    render_master_triage_flowchart()

with tab_stat:
    render_statistics_decision_tree()

with tab_epi:
    render_epidemiology_decision_tree()

with tab_causal:
    render_causal_inference_decision_tree()

with tab_spc:
    render_spc_qi_decision_tree()

with tab_hta:
    render_hta_decision_tree()

with tab_missing:
    render_missing_data_decision_tree()

with tab_mixed:
    render_mixed_models_decision_tree()

with tab_ml:
    render_machine_learning_decision_tree()

with tab_power:
    render_sample_size_decision_tree()

with tab_nlp:
    render_nlp_decision_tree()

st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.85rem; color: #64748b; padding: 1rem 0;">
    🧭 <strong>MethodFinder Quantitative Architecture</strong> • Cross-Pillar Decision Support & Educational Guidance
</div>
""", unsafe_allow_html=True)

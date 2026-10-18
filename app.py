"""Method Finder: Main Entry Point & Welcome Dashboard.

Interactive decision-support system guiding researchers and analysts to the
right statistical test, machine learning algorithm, or epidemiological method.
"""

import streamlit as st
import sys
import os

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.stats_calcs.decision_schema import STATISTICAL_TESTS
from src.ui_components.sidebar_navigation import render_custom_sidebar_navigation

st.set_page_config(
    page_title="Method Finder",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Custom Grouped Sidebar Navigation
render_custom_sidebar_navigation("app.py")

# Header & Developer / Debug Mode Controls
col_title, col_debug = st.columns([3, 1])
with col_title:
    st.title("🧭 Method Finder")
    st.subheader("Interactive Decision-Support for Data Analysis & Research")
with col_debug:
    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    if "debug_mode" not in st.session_state:
        st.session_state["debug_mode"] = False
    if "debug_model" not in st.session_state:
        st.session_state["debug_model"] = False

    debug_val = st.toggle(
        "🛠️ Debug / Dev Mode",
        value=st.session_state.get("debug_mode", False) or st.session_state.get("debug_model", False),
        key="debug_mode_toggle",
        help="When enabled, displays internal implementation and validation progress trackers at the top of each page."
    )
    st.session_state["debug_mode"] = debug_val
    st.session_state["debug_model"] = debug_val
    if debug_val:
        st.caption("🟢 *Debug mode ON: Trackers visible*")


st.markdown("""
Welcome to **Method Finder**. Choosing the appropriate quantitative method is one of the most critical steps in any data project. 
This tool guides you step-by-step through research questions, data types, and design constraints to recommend the correct method, 
verify its assumptions, and implement it in both **Python** and **Microsoft Excel**.
""")

st.markdown("""
<div style="background-color: #f0fdf4; border: 2px solid #16a34a80; border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.2rem;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h3 style="color: #166534; margin: 0 0 0.4rem 0;">🎯 Not Sure Which Pillar or Method Fits Your Problem?</h3>
            <p style="color: #1e293b; margin: 0; font-size: 0.95rem; line-height: 1.5;">
                Start with the <strong>Universal Study Design & Method Selector</strong>. Translate your research or operational question across 7 real-world domains (clinical care, public health, manufacturing, marketing, subscriptions, logistics), evaluate cross-pillar tradeoffs (Statistics vs. Epidemiology vs. Machine Learning), pair with sample size planning, and deep-link directly into the dedicated module.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_hub1, col_hub2, col_hub3 = st.columns(3)
with col_hub1:
    st.page_link("pages/0_📚_Statistical_Fundamentals.py", label="Statistical Fundamentals & Baseline Concepts 📚", icon="📚")
with col_hub2:
    st.page_link("pages/7_🎯_Study_Design_and_Method_Selector.py", label="Universal Study Design Selector 🎯", icon="🎯")
with col_hub3:
    st.page_link("pages/8_🗺️_Decision_Flowcharts.py", label="Methodology Decision Flowcharts 🗺️", icon="🗺️")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### 📊 Statistical Tests
    **Status:** 🟢 **Active (18 Tests)**
    
    Identify the right hypothesis test:
    - Compare 1, 2, or 3+ groups (t-tests, ANOVA, Mann-Whitney, Kruskal-Wallis)
    - Paired & repeated measures (Paired t-test, Wilcoxon, RM-ANOVA, Friedman)
    - Categorical tables (Chi-square, Fisher's Exact, McNemar)
    - Relationships (Pearson, Spearman, Linear Regression)
    
    *Includes native Excel formulas, Toolpak steps, and live simulations.*
    """)
    st.page_link("pages/1_📊_Statistical_Tests.py", label="Open Statistical Tests 🎯", icon="🎯")
    
    with st.expander("⚡ Quick Jump to Test", expanded=False):
        selected_quick_key = st.selectbox(
            "Select method to explore directly:",
            options=list(STATISTICAL_TESTS.keys()),
            format_func=lambda k: STATISTICAL_TESTS[k]["name"],
            key="home_quick_select"
        )
        if st.button("Explore This Method 👉", key="btn_quick_explore"):
            st.session_state["nav_mode"] = "direct"
            st.session_state["direct_test_key"] = selected_quick_key
            st.switch_page("pages/1_📊_Statistical_Tests.py")

with col2:
    st.markdown("""
    ### 🏥 Epidemiology
    **Status:** 🟢 **Active (17 Methods)**
    
    Tailored for public health & observational studies:
    - Study designs (Cohort, Case-Control, RCT)
    - Risk & effect metrics (OR, RR, IRR, RD, NNT)
    - Rate standardization (Direct DSR & Indirect SMR)
    - Outbreak analytics & Meta-Analysis (PRISMA 2020)
    - Diagnostic screening (Sensitivity, Specificity, ROC)
    
    *Full 4-tab architecture with DerSimonian-Laird & GIS maps.*
    """)
    st.page_link("pages/2_🏥_Epidemiology_Methods.py", label="Open Epidemiology Methods 👉", icon="🏥")

with col3:
    st.markdown("""
    ### 📐 Sample Size & Power
    **Status:** 🟢 **Active (10 Frameworks)**
    
    Biostatistical sizing & clinical trial planning:
    - Continuous endpoints (Two-sample t, paired, ANOVA)
    - Binary risk metrics (Two proportions, Cohort RR, Case-Control OR)
    - Survival outcomes (Schoenfeld log-rank events)
    - Modifiers (Attrition cushion & Cluster RCT VIF)
    - NIHR / MRC protocol justification text generator
    
    *Dual Plotly power curves & Excel sensitivity grids.*
    """)
    st.page_link("pages/4_📐_Sample_Size_and_Power.py", label="Open Sample Size & Power 📐", icon="📐")

with col4:
    st.markdown("""
    ### 🤖 Machine Learning
    **Status:** 🟢 **Active (16 Architectures)**
    
    Predictive modeling & operational analytics:
    - Supervised classification (PR-AUC, calibrated probabilities, rare 8% readmissions)
    - Operational demand & bed surge forecasting (Holt-Winters triple smoothing, SARIMA)
    - Health equity & algorithmic fairness audits (Four-Fifths 80% Rule across IMD deciles)
    - Dynamic classification threshold tuning (Precision vs. Sensitivity tradeoffs)
    - Unsupervised patient phenotyping & PCA scree variance extraction
    """)
    st.page_link("pages/3_🤖_Machine_Learning.py", label="Open Machine Learning & Forecasting 🤖", icon="🤖")

st.markdown("---")

col_row2_1, col_row2_2 = st.columns(2)

with col_row2_1:
    st.markdown("""
    ### 📋 Survey & Questionnaire Analysis
    **Status:** 🟢 **Active (7 Methods)**
    
    Psychometrics, scale reliability & representative survey sampling:
    - **Scale Analysis:** Likert scale analysis, Ordinal vs. Interval controversy, Diverging 100% stacked bar charts
    - **Scale Reliability:** Cronbach's Alpha (α), Alpha-if-Item-Deleted purification table, McDonald's Omega (ω)
    - **Inter-Rater Concordance:** Cohen's Kappa (κ), Quadratic Weighted Kappa (κw), Fleiss' Kappa
    - **Construct Validity:** Exploratory Factor Analysis (EFA), Cattell's Scree Plot, KMO sampling adequacy
    - **Sampling & Weighting:** Finite Population Correction (FPC) sizing & Post-stratification cell weighting with Kish's Effective Sample Size (Neff)
    """)
    st.page_link("pages/5_📋_Survey_and_Questionnaire_Analysis.py", label="Open Survey & Questionnaire Analysis 📋", icon="📋")

with col_row2_2:
    st.markdown("""
    ### 🗣️ Natural Language Processing (Classical to LLMs)
    **Status:** 🟢 **Active (7 Methods)**
    
    Clinical text, incident reports (Datix) & patient comments:
    - **Level 1:** Classical & Rule-Based NLP (tokenization, regex entity extraction, VADER sentiment & clinical NegEx)
    - **Level 2:** Statistical Vectorization (TF-IDF term weighting, Naive Bayes & Linear SVM text classifiers)
    - **Level 3:** Unsupervised Topic Modelling (Latent Dirichlet Allocation, thematic discovery, coherence $C_v$)
    - **Level 4:** Dense Semantic Embeddings (384-dim vector representations, cosine similarity, semantic KNN)
    - **Level 5:** State-of-the-Art Generative AI & LLMs (Clinical NER, Zero-Shot, RAG hallucination audit)
    """)
    st.page_link("pages/6_🗣️_Natural_Language_Processing.py", label="Open Natural Language Processing 🗣️", icon="🗣️")

st.markdown("---")

st.markdown("""
### 💡 Guiding Principles
1. **Plain Language First:** We ask about your research questions and variable characteristics without upfront statistical jargon.
2. **Dual Implementation Pathways:** Every recommendation includes ready-to-use Python snippets and step-by-step Microsoft Excel guidance.
3. **Transparent Assumptions:** We highlight critical distribution and independence assumptions along with fallback alternatives.
4. **Reproducible Live Visuals:** Test out how sample sizes and variances impact p-values on simulated synthetic data.
""")


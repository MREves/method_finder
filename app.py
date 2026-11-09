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
from src.ml_models.ml_schema import ML_MODELS
from src.survey_analysis.survey_schema import SURVEY_METHODS
from src.nlp.nlp_schema import NLP_METHODS
from src.time_series.ts_engine import TIME_SERIES_ALGORITHMS
from src.causal_inference.causal_schema import CAUSAL_METHODS
from src.spc_qi.spc_schema import SPC_METHODS
from src.health_economics.hta_schema import HTA_METHODS
from src.missing_data.missing_schema import MISSING_DATA_METHODS
from src.mixed_models.mixed_schema import MIXED_MODELS_METHODS
from src.ui_components.sidebar_navigation import render_custom_sidebar_navigation

EPI_QUICK_METHODS = {
    "risk_stratification": "Clinical Risk Stratification (TRIPOD, DCA, Points)",
    "risk_difference": "Risk Difference (RD) & NNT / NNH",
    "health_inequalities": "Health Inequalities (Slope Index SII & RII)",
    "relative_risk": "Relative Risk / Risk Ratio (RR)",
    "odds_ratio": "Odds Ratio (OR in Case-Control)",
    "direct_standardization": "Direct Age-Standardized Rate (DSR & Map)",
    "indirect_standardization": "Indirect Standardization & SMR",
    "kaplan_meier": "Kaplan-Meier Survival Analysis",
    "cox_regression": "Cox Proportional Hazards Regression",
    "diagnostic_accuracy": "Diagnostic Accuracy (Sens / Spec / ROC)",
    "meta_analysis": "Random-Effects Meta-Analysis (PRISMA)",
    "outbreak_investigation": "Outbreak Investigation & Attack Rates"
}

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
    st.page_link("pages/13_🎯_Study_Design_and_Method_Selector.py", label="Universal Study Design Selector 🎯", icon="🎯")
with col_hub3:
    st.page_link("pages/14_🗺️_Decision_Flowcharts.py", label="Methodology Decision Flowcharts 🗺️", icon="🗺️")

st.markdown("---")

col1, col2, col3 = st.columns(3)

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
    **Status:** 🟢 **Active (18 Methods)**
    
    Tailored for public health & observational studies:
    - Study designs (Cohort, Case-Control, RCT)
    - Risk & effect metrics (OR, RR, IRR, RD, NNT)
    - Clinical risk stratification (Framingham/QRISK points, TRIPOD, DCA)
    - Rate standardization (Direct DSR & Indirect SMR)
    - Health inequalities (Slope Index SII & Relative Index RII)
    - Outbreak analytics & Meta-Analysis (PRISMA 2020)
    - Diagnostic screening (Sensitivity, Specificity, ROC)
    
    *Full 4-tab architecture with DerSimonian-Laird & GIS maps.*
    """)
    st.page_link("pages/2_🏥_Epidemiology_Methods.py", label="Open Epidemiology Methods 👉", icon="🏥")

    with st.expander("⚡ Quick Jump to Method", expanded=False):
        selected_epi_quick = st.selectbox(
            "Select method to explore directly:",
            options=list(EPI_QUICK_METHODS.keys()),
            format_func=lambda k: EPI_QUICK_METHODS[k],
            key="home_quick_epi_select"
        )
        if st.button("Explore This Method 👉", key="btn_quick_epi_explore"):
            st.session_state["epi_nav_mode"] = "direct"
            st.session_state["epi_direct_method_key"] = selected_epi_quick
            st.switch_page("pages/2_🏥_Epidemiology_Methods.py")

with col3:
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

    with st.expander("⚡ Quick Jump to Method", expanded=False):
        selected_ml_quick = st.selectbox(
            "Select method to explore directly:",
            options=list(ML_MODELS.keys()),
            format_func=lambda k: ML_MODELS[k]["name"],
            key="home_quick_ml_select"
        )
        if st.button("Explore This Method 👉", key="btn_quick_ml_explore"):
            st.session_state["ml_nav_mode"] = "direct"
            st.session_state["ml_direct_model_key"] = selected_ml_quick
            st.switch_page("pages/3_🤖_Machine_Learning.py")

st.markdown("---")

col_row2_1, col_row2_2, col_row2_3 = st.columns(3)

with col_row2_1:
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

with col_row2_2:
    st.markdown("""
    ### 📋 Survey & Questionnaire Analysis
    **Status:** 🟢 **Active (7 Methods)**
    
    Psychometrics, scale reliability & representative survey sampling:
    - **Scale Analysis:** Likert scale analysis, Diverging 100% stacked bar charts
    - **Scale Reliability:** Cronbach's Alpha (α), Item-Deleted purification, McDonald's Omega (ω)
    - **Inter-Rater Concordance:** Cohen's Kappa (κ), Quadratic Weighted Kappa, Fleiss' Kappa
    - **Construct Validity:** Exploratory Factor Analysis (EFA), Cattell's Scree, KMO test
    - **Sampling & Weighting:** FPC sizing & Post-stratification cell weighting with Kish's Neff
    """)
    st.page_link("pages/5_📋_Survey_and_Questionnaire_Analysis.py", label="Open Survey & Questionnaire Analysis 📋", icon="📋")

    with st.expander("⚡ Quick Jump to Method", expanded=False):
        selected_survey_quick = st.selectbox(
            "Select method to explore directly:",
            options=list(SURVEY_METHODS.keys()),
            format_func=lambda k: f"{SURVEY_METHODS[k]['icon']} {SURVEY_METHODS[k]['name']}",
            key="home_quick_survey_select"
        )
        if st.button("Explore This Method 👉", key="btn_quick_survey_explore"):
            st.session_state["survey_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.session_state["survey_direct_method_key"] = selected_survey_quick
            st.switch_page("pages/5_📋_Survey_and_Questionnaire_Analysis.py")

with col_row2_3:
    st.markdown("""
    ### 🗣️ Natural Language Processing
    **Status:** 🟢 **Active (7 Methods)**
    
    Clinical text, incident reports (Datix) & patient comments:
    - **Rule-Based NLP:** VADER sentiment valence & clinical NegEx negation scope
    - **Vectorization:** TF-IDF keyword extraction, Naive Bayes & Linear SVM classifiers
    - **Topic Modelling:** Latent Dirichlet Allocation (LDA) thematic discovery
    - **Dense Embeddings:** 384-dim semantic vectors, cosine similarity & KNN
    - **Generative AI & LLMs:** Clinical NER, Zero-Shot classification & RAG audits
    """)
    st.page_link("pages/6_🗣️_Natural_Language_Processing.py", label="Open Natural Language Processing 🗣️", icon="🗣️")

    with st.expander("⚡ Quick Jump to Method", expanded=False):
        selected_nlp_quick = st.selectbox(
            "Select method to explore directly:",
            options=list(NLP_METHODS.keys()),
            format_func=lambda k: f"{NLP_METHODS[k]['icon']} {NLP_METHODS[k]['name']}",
            key="home_quick_nlp_select"
        )
        if st.button("Explore This Method 👉", key="btn_quick_nlp_explore"):
            st.session_state["nlp_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.session_state["nlp_direct_method_key"] = selected_nlp_quick
            st.session_state["active_nlp_method"] = selected_nlp_quick
            st.switch_page("pages/6_🗣️_Natural_Language_Processing.py")

st.markdown("---")

col_row3_1, col_row3_2, col_row3_3 = st.columns(3)

with col_row3_1:
    st.markdown("""
    ### ⏱️ Time Series & Forecasting
    **Status:** 🟢 **Active (8 Models & 7 Metrics)**
    
    Sequential temporal modeling, decomposition & evaluation metrics:
    - **Anatomy & Diagnostics:** Additive/Multiplicative decomposition, ACF, PACF, ADF test
    - **Model Hierarchy:** Naive, Seasonal Naive, Drift, SES, Holt's Damped, Holt-Winters, Lagged ML, Croston
    - **Evaluation Metrics Deep Dive:** MAE, MSE, RMSE, MAPE (zero division risks), sMAPE, MASE, MDA
    - **Metric Decision Matrix:** When to pick which metric over another based on operational costs
    - **Temporal Validation:** Rolling-Origin (Expanding-Window) CV preventing lookahead leakage
    """)
    st.page_link("pages/7_⏱️_Time_Series_and_Forecasting.py", label="Open Time Series & Forecasting ⏱️", icon="⏱️")

    with st.expander("⚡ Quick Jump to Method", expanded=False):
        selected_ts_quick = st.selectbox(
            "Select method to explore directly:",
            options=list(TIME_SERIES_ALGORITHMS.keys()),
            format_func=lambda k: f"⏱️ {TIME_SERIES_ALGORITHMS[k]['name']}",
            key="home_quick_ts_select"
        )
        if st.button("Explore This Method 👉", key="btn_quick_ts_explore"):
            st.session_state["ts_direct_method_key"] = selected_ts_quick
            st.switch_page("pages/7_⏱️_Time_Series_and_Forecasting.py")

with col_row3_2:
    st.markdown("""
    ### 🔍 Causal Inference & Policy Evaluation
    **Status:** 🟢 **Active (5 Architectures)**
    
    Quasi-experimental evaluation & counterfactual identification:
    - **Parallel Trends:** Difference-in-Differences (DiD) & event studies
    - **Temporal Policy Shocks:** Interrupted Time Series (ITS) segmented regression
    - **Confounding by Indication:** Propensity Score Matching & IPTW
    - **Cutoff Decisions:** Sharp & Fuzzy Regression Discontinuity Design (RDD)
    - **Unmeasured Confounding:** 2SLS Instrumental Variables & Mendelian Randomization
    """)
    st.page_link("pages/8_🔍_Causal_Inference_and_Policy_Evaluation.py", label="Open Causal Inference & Policy 🔍", icon="🔍")

    with st.expander("⚡ Quick Jump to Method", expanded=False):
        selected_causal_quick = st.selectbox(
            "Select method to explore directly:",
            options=list(CAUSAL_METHODS.keys()),
            format_func=lambda k: f"{CAUSAL_METHODS[k]['icon']} {CAUSAL_METHODS[k]['name']}",
            key="home_quick_causal_select"
        )
        if st.button("Explore This Method 👉", key="btn_quick_causal_explore"):
            st.session_state["causal_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.session_state["causal_direct_method_key"] = selected_causal_quick
            st.switch_page("pages/8_🔍_Causal_Inference_and_Policy_Evaluation.py")

with col_row3_3:
    st.markdown("""
    ### 📈 Statistical Process Control & QI
    **Status:** 🟢 **Active (5 Architectures)**
    
    Healthcare quality improvement & Shewhart control limits:
    - **Run Charts:** Baseline median, shift, trend, runs & astronomical points
    - **Shewhart XmR / I-MR:** Individual measures & moving ranges with 3σ limits
    - **Laney p'-Charts:** Binomial proportions with overdispersion correction
    - **Laney u'-Charts:** Poisson infection rates per 1,000 bed-days
    - **NHS Making Data Count:** System variation & assurance icons ($C_p, C_{pk}$)
    """)
    st.page_link("pages/9_📈_Statistical_Process_Control_and_Quality_Improvement.py", label="Open Statistical Process Control 📈", icon="📈")

    with st.expander("⚡ Quick Jump to Method", expanded=False):
        selected_spc_quick = st.selectbox(
            "Select method to explore directly:",
            options=list(SPC_METHODS.keys()),
            format_func=lambda k: f"{SPC_METHODS[k]['icon']} {SPC_METHODS[k]['name']}",
            key="home_quick_spc_select"
        )
        if st.button("Explore This Method 👉", key="btn_quick_spc_explore"):
            st.session_state["spc_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.session_state["spc_direct_method_key"] = selected_spc_quick
            st.switch_page("pages/9_📈_Statistical_Process_Control_and_Quality_Improvement.py")

st.markdown("---")

col_row4_1, col_row4_2, col_row4_3 = st.columns(3)

with col_row4_1:
    st.markdown("""
    ### 💷 Health Economics & HTA (NICE)
    **Status:** 🟢 **Active (5 Architectures)**
    
    Cost-effectiveness, QALYs & healthcare payer decision models:
    - **Cost-Effectiveness Plane:** ICER (£/QALY), 4 quadrants & £20k-£30k NICE threshold
    - **Linear Net Benefits:** Net Monetary Benefit (NMB) & Net Health Benefit (NHB)
    - **Markov Cohort Models:** 10-year state transitions, half-cycle correction & 3.5% discounting
    - **Decision Uncertainty:** Monte Carlo PSA, parameter distributions & CEAC curves
    - **Financial Affordability:** 5-year NHS Budget Impact Analysis (BIA) & hospital offsets
    """)
    st.page_link("pages/10_💷_Health_Economics_and_HTA.py", label="Open Health Economics & HTA 💷", icon="💷")

    with st.expander("⚡ Quick Jump to Method", expanded=False):
        selected_hta_quick = st.selectbox(
            "Select method to explore directly:",
            options=list(HTA_METHODS.keys()),
            format_func=lambda k: f"{HTA_METHODS[k]['icon']} {HTA_METHODS[k]['name']}",
            key="home_quick_hta_select"
        )
        if st.button("Explore This Method 👉", key="btn_quick_hta_explore"):
            st.session_state["hta_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.session_state["hta_direct_method_key"] = selected_hta_quick
            st.switch_page("pages/10_💷_Health_Economics_and_HTA.py")

with col_row4_2:
    st.markdown(r"""
    ### 🧩 Missing Data & Imputation
    **Status:** 🟢 **Active (5 Architectures)**
    
    Rubin's taxonomy, MICE pooling & regulatory stress-tests:
    - **Rubin's Taxonomy:** Diagnostic audits, Little's MCAR test & logistic missingness
    - **Methodological Hazards:** Complete case attrition & mean imputation variance collapse
    - **MICE Imputation:** Fully conditional specification & Rubin's Rules pooling ($T = \bar{U} + (1+1/m)B$)
    - **Algorithmic Imputation:** Non-parametric $k$-NN & MissForest random forest ensembles
    - **MNAR Stress-Testing:** ICH E9 (R1) pattern-mixture $\delta$-adjustment tipping point analysis
    """)
    st.page_link("pages/11_🧩_Missing_Data_and_Imputation.py", label="Open Missing Data & Imputation 🧩", icon="🧩")

    with st.expander("⚡ Quick Jump to Method", expanded=False):
        selected_missing_quick = st.selectbox(
            "Select method to explore directly:",
            options=list(MISSING_DATA_METHODS.keys()),
            format_func=lambda k: f"{MISSING_DATA_METHODS[k]['icon']} {MISSING_DATA_METHODS[k]['name']}",
            key="home_quick_missing_select"
        )
        if st.button("Explore This Method 👉", key="btn_quick_missing_explore"):
            st.session_state["missing_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.session_state["missing_direct_method_key"] = selected_missing_quick
            st.switch_page("pages/11_🧩_Missing_Data_and_Imputation.py")

with col_row4_3:
    st.markdown(r"""
    ### 🪜 Longitudinal & Mixed Models
    **Status:** 🟢 **Active (5 Architectures)**
    
    Repeated measures, subject trajectories & organizational nesting:
    - **LMM Random Slopes:** Laird & Ware formulation, personalized rates of change & ICC
    - **GLMM Repeated Outcomes:** Mixed logistic/Poisson regressions & non-collapsible ORs
    - **RM-ANOVA Tradeoff:** Mauchly's sphericity, Greenhouse-Geisser & LMM MAR recovery
    - **3-Level Hierarchies:** Patients nested in clinicians in trusts & variance partitioning (VPC)
    - **Covariance Selection:** AR(1) error decay, ML vs REML & Nakagawa $R^2_m$ / $R^2_c$
    """)
    st.page_link("pages/12_🪜_Longitudinal_and_Mixed_Models.py", label="Open Longitudinal & Mixed Models 🪜", icon="🪜")

    with st.expander("⚡ Quick Jump to Method", expanded=False):
        selected_mixed_quick = st.selectbox(
            "Select method to explore directly:",
            options=list(MIXED_MODELS_METHODS.keys()),
            format_func=lambda k: f"{MIXED_MODELS_METHODS[k]['icon']} {MIXED_MODELS_METHODS[k]['name']}",
            key="home_quick_mixed_select"
        )
        if st.button("Explore This Method 👉", key="btn_quick_mixed_explore"):
            st.session_state["mixed_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.session_state["mixed_direct_method_key"] = selected_mixed_quick
            st.switch_page("pages/12_🪜_Longitudinal_and_Mixed_Models.py")

st.markdown("---")

st.markdown("""
### 💡 Guiding Principles
1. **Plain Language First:** We ask about your research questions and variable characteristics without upfront statistical jargon.
2. **Dual Implementation Pathways:** Every recommendation includes ready-to-use Python snippets and step-by-step Microsoft Excel guidance.
3. **Transparent Assumptions:** We highlight critical distribution and independence assumptions along with fallback alternatives.
4. **Reproducible Live Visuals:** Test out how sample sizes and variances impact p-values on simulated synthetic data.
""")


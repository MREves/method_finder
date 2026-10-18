"""Interactive UI Components & Visualizations for Machine Learning & Predictive Analytics.

Provides:
1. Zero-Knowledge Pedagogical Guide:
   - Fundamentals of Machine Learning, Predictive Modeling, Bias-Variance Tradeoff,
     The Accuracy Paradox in imbalanced clinical data, ROC vs. PR-AUC discrimination,
     Algorithmic Equity (Four-Fifths Rule & IMD Deprivation), and Temporal Leakage.
2. Interactive Scenario & Model Architecture Selector:
   - All Meadows Cardiovascular Institute (30-day HF readmission, severe 8% imbalance).
   - Riverside Community Health Trust (Outpatient DNA risk & IMD equity audit).
   - Inner City Teaching Hospital (Daily ED demand & bed capacity forecasting).
   - NHS Integrated Care Board (Frontline nursing sickness absence forecasting).
   - Custom Guided ML & Analytics Playground with step-by-step decision funnel.
3. Methodological Prerequisite Evaluation (5-Point NHS AI & Biostatistical Audit):
   - Sample Size Adequacy (EPV >= 10:1), Class Imbalance & Metric Appropriateness,
     Temporal Independence, Model Linearity vs. Non-Linear Capacity, Algorithmic Fairness.
4. Primary Metrics & Executive KPI Dashboard:
   - Dynamic threshold metrics (Precision, Recall, Specificity, F1, Balanced Accuracy, Brier).
   - Algorithmic disparity metrics (Disparate Impact Ratio, Parity Differences).
   - Time series forecast accuracy (MAE, RMSE, MAPE).
5. Plotly Interactive Visualizers:
   - Dual ROC & Precision-Recall curves with dynamic threshold marker.
   - Algorithmic Equity & Disparate Impact audit across IMD deprivation deciles.
   - Time Series Demand & Bed Surge Forecast with 95% prediction intervals.
   - 2D Decision Boundary Surface Contours.
6. Full 4-Tab Educational Architecture:
   - Tab 1: Model Rationale & Biostatistical Formulations.
   - Tab 2: Clinical & Operational Implementation Guide.
   - Tab 3: Executable Python Pipeline (scikit-learn / statsmodels).
   - Tab 4: Limitations, Bias Pitfalls & NHS AI Guidelines.
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Union
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ml_models.ml_schema import (
    ML_TASK_FAMILIES,
    ML_MODELS,
    ML_DECISION_NODES,
    ML_SCENARIOS,
)
from src.ml_models.ml_engine import (
    generate_ml_demo_data,
    calculate_algorithmic_fairness_metrics,
    calculate_time_series_forecast,
    calculate_classification_threshold_metrics,
    generate_decision_boundary_surface,
    calculate_cluster_metrics,
    calculate_pca_decomposition,
    route_ml_recommendation,
    evaluate_ml_prerequisites,
    generate_ml_python_pipeline,
)


# =====================================================================
# 1. Plain English Pedagogical Introductory Guide
# =====================================================================

def render_ml_plain_english_guide():
    """Renders a collapsible plain English introductory guide to machine learning & predictive analytics."""
    with st.expander("📖 Zero-Knowledge Guide: Machine Learning, Algorithmic Equity & Operational Forecasting", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. Machine Learning vs. Classical Biostatistics
            * **Classical Hypothesis Testing:** Asks *"Did this intervention cause a statistically significant difference?"* Focuses on $p$-values, confidence intervals, and unbiased population parameter estimation under strict parametric assumptions.
            * **Predictive Machine Learning:** Asks *"Given this individual patient or hospital's current features, what is the exact probability of an event occurring in the future?"* Optimizes out-of-sample discrimination, calibration, and generalization without requiring rigid distributional assumptions.
            * **The Core Supervised Paradigms:**
              - **Classification:** Predicting discrete outcomes (e.g., 30-day heart failure emergency readmission: *Yes* vs. *No*).
              - **Continuous Regression:** Predicting numerical quantities (e.g., inpatient length of stay: *days*, clinical treatment cost: *£*).
              - **Time Series Forecasting:** Predicting sequential temporal demand across future horizons (e.g., daily A&E attendances, winter bed requirements, staff sickness absences).
              - **Unsupervised Learning:** Discovering hidden patient subgroups or disease phenotypes without predefined labels (e.g., K-Means clustering, PCA).

            #### 2. The Accuracy Paradox in Healthcare
            * In clinical epidemiology, target events are frequently rare. For example, in heart failure registries, only **8% of discharged patients** suffer emergency 30-day readmission.
            * **The Trap:** A naive algorithm predicting that *zero patients will ever be readmitted* achieves **92.0% Overall Accuracy**!
            * While statistically impressive on paper, this model has **0% Sensitivity (Recall)**, identifies zero deteriorating patients, and is clinically useless.
            * **The Golden Rule:** For imbalanced healthcare outcomes, **never rely on accuracy**. Evaluate models using **Precision-Recall AUC (PR-AUC)**, **F1-Score**, and **Calibrated Probabilities** (Brier score).
            """)

        with col_g2:
            st.markdown(r"""
            #### 3. Algorithmic Equity & Health Disparities (The Four-Fifths Rule)
            * When predictive models inform operational scheduling—such as identifying patients likely to **Did Not Attend (DNA)** an outpatient clinic to overbook slots—models can inadvertently entrench socio-economic bias.
            * **The Transport & Deprivation Gradient:** Patients residing in high-deprivation areas (Indices of Multiple Deprivation IMD Deciles 1–3) encounter greater public transit delays, childcare barriers, and zero-hour contract inflexibility. As a result, predictive algorithms produce **2.4x higher False Positive Rates** in deprived cohorts.
            * If an algorithm flags these patients for double-booking, vulnerable patients face punitive clinic waiting times and clinician fatigue.
            * **The Four-Fifths (80%) Disparate Impact Standard:** Under international fairness guidelines, the selection rate of a protected cohort must be at least **80% ($\ge 0.80$)** of the most favored cohort. Auditing Disparate Impact ensures predictive models allocate **supportive interventions** (e.g. transport subsidies, SMS reminders) rather than punitive operational penalties.

            #### 4. The Danger of Temporal Data Leakage in Time Series
            * In standard tabular cross-validation, data rows are randomly shuffled across $K$ folds.
            * **The Catastrophic Flaw:** In healthcare time series (e.g. daily A&E attendances or winter bed surges), shuffling trains the model on future observations (e.g. February 2025) to predict past attendances (e.g. January 2025).
            * This causes **future-to-past temporal leakage**, artificially inflating test accuracy while causing disastrous model collapse when deployed live.
            * **The Solution:** Time series forecasting must strictly enforce **Rolling-Origin / Expanding Window Evaluation**, training solely on historical sequences to project forward into unseen horizons.

            #### 5. Cross-Industry Parallels: Manufacturing, Marketing, Subscriptions & Logistics
            The core mathematical principles of machine learning apply identically across diverse industrial and commercial sectors:
            * **The Accuracy Paradox in Manufacturing (Predictive Maintenance):**
              - Industrial CNC spindle bearings fail at a low rate (e.g. 4.2% failure rate). A naive model predicting that *zero spindles ever break down* achieves **95.8% accuracy** while causing catastrophic assembly line stoppage costing £45,000/hour. Just as in clinical readmissions, **Precision-Recall AUC (PR-AUC)** and **Recall** are the only valid metrics.
            * **Algorithmic Fairness in Omnichannel Marketing & Digital Redlining:**
              - Auditing conversion propensity models using the **Four-Fifths (80%) Rule** prevents discriminatory exclusion. If promotional discount codes or installment terms are disproportionately withheld from price-sensitive demographic segments ($DIR < 0.80$), automated marketing algorithms deepen socio-economic inequalities.
            * **Threshold Tuning & Unit Economics in Digital SaaS Subscriptions:**
              - Operating a subscriber churn model at the default decision threshold of $p^* = 0.50$ misses over 60% of cancellations. Tuning $p^*$ to balance the cost of proactive customer success concierge outreach (£15/user) against annual customer lifetime value (£360/year) maximizes net retention revenue.
            * **Temporal Data Leakage in Supply Chain & Logistics Forecasting:**
              - Shuffling warehouse dispatch orders randomly across cross-validation folds leaks future holiday demand spikes into past baseline weeks. Enforcing **Rolling-Origin Temporal Partitioning** ensures freight trucks and loading bay allocations match actual physical capacity.

            #### 6. Unsupervised Patient Phenotyping & Cluster Validation Metrics
            * In unsupervised clustering (e.g. K-Means, DBSCAN), there are **no predefined outcome labels** ($Y$) and **no null hypothesis** ($H_0$). Therefore, classical statistical sample size and power formulas ($\alpha = 0.05, 1 - \beta \ge 80\%$) do not apply to cluster discovery.
            * **Sample Scale Rules of Thumb for Clustering:**
              - $N \ge 20\text{–}50$ observations per expected cluster ($N \ge 20k$) to ensure stable, reproducible subgroup boundaries and avoid single-patient idiosyncratic clusters.
              - $N \ge 10 \times p$ observations per feature to prevent the curse of dimensionality where Euclidean distances flatten.
            * **The 3 Core Mathematical Cluster Quality Metrics:**
              1. **Silhouette Score ($s \in [-1, +1]$):** Measures intra-cluster cohesion relative to nearest-cluster separation. Scores $> 0.50$ indicate dense, well-isolated phenotypes; scores near $0$ indicate boundary overlap.
              2. **Davies-Bouldin Index ($\text{DBI} \ge 0$):** Measures the maximum ratio of within-cluster scatter to centroid separation. **Lower is better** ($\text{DBI} < 1.0$ indicates compact, well-separated clusters; the trough on the curve identifies optimal $k$).
              3. **Calinski-Harabasz Index ($\text{CHI} \ge 0$ / Variance Ratio Criterion):** Measures the ratio of between-cluster variance to within-cluster variance (analogous to the ANOVA $F$-statistic). **Higher is better**; peaks on the curve identify the optimal cluster partition $k^*$.
            """)


# =====================================================================
# 2. Navigation Header, Guided Wizard & Direct Explorer
# =====================================================================

def render_ml_navigation_header() -> bool:
    """Renders the top navigation mode radio between Guided Wizard and Direct Explorer."""
    st.subheader("🧭 Choose Navigation Mode")

    if "ml_nav_mode" not in st.session_state:
        st.session_state["ml_nav_mode"] = "guided"

    mode_guided = "🎯 Guided Decision Wizard (Route by research aim, data structure & constraints)"
    mode_direct = "⚡ Direct Architecture Explorer (Select any model directly from the 16-model catalog)"

    mode_options = [mode_guided, mode_direct]
    default_idx = 0 if st.session_state["ml_nav_mode"] == "guided" else 1

    selected_mode = st.radio(
        "How would you like to explore machine learning models?",
        options=mode_options,
        index=default_idx,
        horizontal=True,
        help="Switch between guided step-by-step decision routing and direct selection of any machine learning method."
    )

    is_direct = (selected_mode == mode_direct)
    st.session_state["ml_nav_mode"] = "direct" if is_direct else "guided"
    return is_direct


def render_ml_guided_wizard() -> Tuple[str, Dict[str, Any]]:
    """Renders the step-by-step guided questions and model recommendation card."""
    st.subheader("🎯 Step 1: Define Research Aim & Algorithmic Requirements")
    st.markdown("Answer the questions below to receive a biostatistically justified machine learning recommendation:")

    c1, c2 = st.columns(2)
    with c1:
        q_task = st.selectbox(
            "What is your target outcome or analytical task?",
            options=["binary_event", "continuous_count", "time_series", "unlabelled_cohort", "dim_reduction", "equity_audit"],
            format_func=lambda x: {
                "binary_event": "🎯 Binary Clinical Event (e.g. 30-day Readmission, DNA: 0 or 1)",
                "continuous_count": "📈 Continuous Quantity (e.g. Inpatient Length of Stay, Healthcare Cost)",
                "time_series": "⏱️ Sequential Operational Time Series (e.g. Daily A&E Attendances, Staff Sickness)",
                "unlabelled_cohort": "🔍 Unlabelled Patient Cohort (e.g. Discovering Disease Subtypes / Phenotypes)",
                "dim_reduction": "📉 High-Dimensional Feature Space (e.g. Compressing Correlated Biomarkers / PCA)",
                "equity_audit": "⚖️ Algorithmic Equity Audit (e.g. Auditing Disparate Impact across IMD Deciles)"
            }[x],
            index=0,
            help="Select the fundamental epidemiological or operational structure of your primary endpoint."
        )

        q_interpret = st.selectbox(
            "What is your clinical explainability & governance priority?",
            options=["high", "balanced", "max_accuracy"],
            format_func=lambda x: {
                "high": "High Interpretability: Must publish direct Odds Ratios / linear slopes (Governance priority)",
                "balanced": "Balanced: Non-linear interaction capture with robust variance reduction (e.g. Random Forest)",
                "max_accuracy": "Maximum Discrimination: Prioritize top-tier predictive power (e.g. Gradient Boosting)"
            }[x],
            index=1,
            help="Clinical trials often require linear odds ratios; operational triage can leverage non-linear tree ensembles."
        )

    with c2:
        q_imbalance = st.selectbox(
            "What is the target event incidence / prevalence rate?",
            options=["severe", "moderate", "balanced"],
            format_func=lambda x: {
                "severe": "Severely Imbalanced (< 10% event rate - e.g. Heart Failure Readmission 8%)",
                "moderate": "Moderately Imbalanced (10% to 20% event incidence)",
                "balanced": "Balanced Target (>= 20% event incidence)"
            }[x],
            index=0,
            help="Determines whether standard accuracy is misleading (Accuracy Paradox) and enforces PR-AUC evaluation."
        )

        q_temporal = st.checkbox(
            "Sequential temporal dependencies present (Must prevent future-to-past data leakage!)",
            value=(q_task == "time_series"),
            help="Enforces rolling-origin TimeSeriesSplit evaluation over standard randomized K-Fold splits."
        )

    task_map = {
        "binary_event": "classification",
        "continuous_count": "regression",
        "time_series": "time_series",
        "unlabelled_cohort": "clustering",
        "dim_reduction": "dim_reduction",
        "equity_audit": "fairness_audit"
    }

    rec = route_ml_recommendation(
        task=task_map[q_task],
        interpretability_priority=q_interpret,
        imbalance_severity=q_imbalance,
        has_temporal_structure=q_temporal,
        is_equity_audit=(q_task == "equity_audit")
    )

    p_mod = rec["primary_model"]
    c_mod = rec["comparator_model"]
    p_key = rec["primary_model_key"]
    c_key = rec["comparator_model_key"]

    st.markdown(f"""
    <div style="background-color: #f0fdf4; border: 1px solid #16a34a50; border-radius: 8px; padding: 1rem 1.2rem; margin-top: 0.8rem; margin-bottom: 0.8rem;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span style="font-weight: 700; color: #166534; font-size: 1.05rem;">
                🎯 Recommended Primary Architecture: {p_mod['name']}
            </span>
            <span style="background-color: #dcfce7; color: #15803d; font-size: 0.8rem; font-weight: 600; padding: 0.2rem 0.5rem; border-radius: 4px;">
                Family: {p_mod.get('family', 'Classification').title()}
            </span>
        </div>
        <div style="font-size: 0.9rem; color: #1f2937; margin-top: 0.4rem; line-height: 1.4;">
            <strong>Biostatistical Rationale:</strong> {rec.get('rationale', '')}
        </div>
        <div style="font-size: 0.85rem; color: #374151; margin-top: 0.4rem;">
            <strong>Alternative Benchmark Comparator:</strong> {c_mod['name']} ({c_mod.get('interpretability', '')} Interpretability)
        </div>
        <div style="font-size: 0.82rem; color: #4b5563; margin-top: 0.3rem;">
            <strong>Primary Recommended Evaluation Metric:</strong> <code>{p_mod.get('primary_metric', p_mod.get('default_metric', 'PR-AUC / F1-Score'))}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_options = [
        (p_key, f"🎯 Recommended Primary Model: {p_mod['name']}"),
        (c_key, f"🔄 Benchmark Alternative: {c_mod['name']}")
    ]
    chosen_model_tuple = st.radio(
        "Select Active Model to Configure and Simulate:",
        options=model_options,
        format_func=lambda t: t[1],
        index=0,
        horizontal=True
    )
    selected_model_key = chosen_model_tuple[0]

    return selected_model_key, rec


def render_ml_direct_explorer() -> str:
    """Renders the direct architecture explorer across the 16 model profiles."""
    st.subheader("⚡ Direct Architecture Explorer")
    st.markdown(
        "Select any machine learning architecture from our catalog to directly inspect its methodology, "
        "mathematical derivations, prerequisite diagnostics, production Python pipeline, and live interactive simulation."
    )

    ML_CATEGORY_GROUPS = {
        "All Architectures (Show All 16 Models)": None,
        "🎯 Supervised Binary Classification (6 Models)": [
            "random_forest_classifier", "penalized_logistic_regression", "gradient_boosting_classifier",
            "decision_tree_classifier", "svm_classifier", "knn_classifier"
        ],
        "📈 Supervised Continuous Regression (3 Models)": [
            "random_forest_regressor", "elastic_net_regression", "gradient_boosting_regressor"
        ],
        "⏱️ Time Series & Operational Forecasting (3 Models)": [
            "holt_winters_exponential_smoothing", "sarima_forecasting", "lagged_ensemble_forecasting"
        ],
        "🔍 Unsupervised Patient Phenotyping (2 Models)": [
            "kmeans_clustering", "dbscan_clustering"
        ],
        "📉 Dimensionality Reduction & Variance (1 Model)": [
            "pca_reduction"
        ],
        "⚖️ Algorithmic Fairness & Health Equity (1 Model)": [
            "algorithmic_fairness_audit"
        ]
    }

    col_cat, col_mod = st.columns([1, 2])
    with col_cat:
        selected_group = st.selectbox(
            "Filter by Learning Family:",
            options=list(ML_CATEGORY_GROUPS.keys()),
            index=0,
            help="Filter by learning paradigm or view all 16 supported models."
        )

    allowed_keys = ML_CATEGORY_GROUPS[selected_group]
    available_keys = list(ML_MODELS.keys()) if allowed_keys is None else allowed_keys

    preselected = st.session_state.get("ml_direct_model_key", "random_forest_classifier")
    if preselected not in available_keys:
        preselected = available_keys[0]

    def format_model_option(key: str) -> str:
        info = ML_MODELS[key]
        interp = info.get("interpretability", "Moderate")
        return f"{info['name']} ({interp} Interpretability)"

    with col_mod:
        chosen_key = st.selectbox(
            "Select Machine Learning Architecture:",
            options=available_keys,
            index=available_keys.index(preselected),
            format_func=format_model_option,
            help="Select an algorithm to configure parameters and run live synthetic simulation."
        )
        st.session_state["ml_direct_model_key"] = chosen_key

    info = ML_MODELS[chosen_key]
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.info(f"📁 **Family:** {info.get('family', 'Classification').title()}")
    with c_m2:
        non_lin = "Yes (Non-Linear Splits / Curves)" if info.get("handles_non_linear", True) else "Linear Boundary Only"
        st.info(f"🔄 **Boundary:** {non_lin}")
    with c_m3:
        st.info(f"🔍 **Sample Efficiency:** {info.get('sample_efficiency', 'Moderate')}")

    return chosen_key


def render_ml_scenario_and_parameters(selected_model_key: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Renders clinical and cross-industry scenario selection for the chosen model and interactive simulation sliders."""
    st.subheader("🏥 Step 2: Clinical / Operational Scenario & Live Simulation Settings")

    model_info = ML_MODELS.get(selected_model_key, ML_MODELS["random_forest_classifier"])
    model_family = model_info.get("family", "classification")

    scenario_options = {
        # Clinical & NHS Operational Scenarios
        "clinical_readmission": "🫀 Healthcare: All Meadows Cardiovascular Institute (30-Day Emergency HF Readmission Risk)",
        "operational_dna_equity": "⚖️ Healthcare: Riverside Community Health Trust (Outpatient DNA Risk & Algorithmic Equity Audit)",
        "ed_demand_forecasting": "⏱️ Healthcare: Inner City Teaching Hospital (Daily A&E Demand & Inpatient Bed Surge)",
        "workforce_sickness_forecasting": "👥 Healthcare: NHS Integrated Care Board (Staff Sickness Absence & Rota Depletion)",
        # Cross-Industry Enterprise Scenarios
        "manufacturing_predictive_maintenance": "🏭 Manufacturing: Apex Precision Robotics (CNC Spindle Failure & Predictive Maintenance)",
        "marketing_conversion_scoring": "📢 Marketing: OmniChannel Retail Group (Customer Purchase Propensity & Equity Audit)",
        "subscription_churn_forecasting": "📱 Subscriptions: PulseStream Digital (Monthly SaaS Churn & Retention Analytics)",
        "supply_chain_demand_forecasting": "🚛 Supply Chain: MetroLogix Distribution (Warehouse Pallet Dispatch & Freight Demand)",
        # Custom
        "custom": "🧪 Custom Guided Machine Learning & Predictive Analytics Playground"
    }

    # Contextual default scenario based on selected model family
    if model_family == "time_series":
        default_idx = 2  # Inner City ED Demand
    elif model_family == "fairness_audit":
        default_idx = 1  # Riverside DNA Equity
    elif model_family == "regression":
        default_idx = 2  # Continuous forecasting / admissions
    elif model_family in ("clustering", "dim_reduction"):
        default_idx = 8  # Custom playground
    else:
        default_idx = 0  # All Meadows HF readmissions

    selected_scenario_key = st.selectbox(
        "Select Clinical / Operational Scenario to Test on this Model:",
        options=list(scenario_options.keys()),
        format_func=lambda k: scenario_options[k],
        index=default_idx,
        help="Choose the clinical or cross-industry scenario to test with your selected machine learning architecture."
    )

    meta = dict(ML_SCENARIOS.get(selected_scenario_key, ML_SCENARIOS["clinical_readmission"]))

    # Domain Icons & Enterprise Badge Banner
    domain_icons = {
        "clinical_readmission": "🏥",
        "operational_dna_equity": "⚖️",
        "ed_demand_forecasting": "⏱️",
        "workforce_sickness_forecasting": "👥",
        "manufacturing_predictive_maintenance": "🏭",
        "marketing_conversion_scoring": "📢",
        "subscription_churn_forecasting": "📱",
        "supply_chain_demand_forecasting": "🚛",
        "custom": "🧪"
    }
    icon = domain_icons.get(selected_scenario_key, "🏥")
    org_label = meta.get("organisation_name", meta.get("fictional_trust", "Enterprise Analytics"))

    st.markdown(f"""
    <div style="background-color: #f0fdf4; border-left: 4px solid #16a34a; padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1rem;">
        <span style="font-weight: 600; color: #166534; font-size: 0.95rem;">{icon} {org_label}</span>
        <span style="margin-left: 0.5rem; color: #15803d; font-size: 0.85rem;">| {meta.get('domain', 'Predictive Analytics')}</span>
        <p style="margin: 0.35rem 0 0 0; font-size: 0.88rem; color: #374151;">
            {meta.get('description', '')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    params: Dict[str, Any] = {"model_key": selected_model_key}

    with st.expander("⚙️ Interactive Algorithmic Parameters & Model Tuning", expanded=False):
        c1, c2, c3 = st.columns(3)

        # 1. Sample Size Slider dynamically tailored to domain
        if selected_scenario_key in ("ed_demand_forecasting", "supply_chain_demand_forecasting"):
            with c1:
                label = "Simulated Daily Dispatch Horizon (Days):" if selected_scenario_key == "supply_chain_demand_forecasting" else "Simulated Observation Horizon (Days):"
                help_text = "Controls the exact number of daily dispatch records generated." if selected_scenario_key == "supply_chain_demand_forecasting" else "Controls the exact number of daily attendance records generated. Moving this slider dynamically updates historical fitting, trend estimates, and rolling-origin validation metrics in real time."
                params["sample_size"] = st.slider(
                    label,
                    min_value=90,
                    max_value=730,
                    value=365,
                    step=30,
                    help=help_text
                )
        elif selected_scenario_key == "workforce_sickness_forecasting":
            with c1:
                params["sample_size"] = st.slider(
                    "Simulated Observation Horizon (Weeks):",
                    min_value=26,
                    max_value=156,
                    value=52,
                    step=4,
                    help="Controls the exact number of weekly staff sickness absence records generated."
                )
        elif selected_scenario_key == "manufacturing_predictive_maintenance":
            with c1:
                params["sample_size"] = st.slider(
                    "Monitored CNC Machine Tool Batch Size (N):",
                    min_value=100,
                    max_value=2000,
                    value=500,
                    step=50,
                    help="Controls the number of CNC spindle telemetry records generated. Moving this slider updates machine telemetry, rare failure counts (4.2%), and classification diagnostics in real time."
                )
        elif selected_scenario_key == "marketing_conversion_scoring":
            with c1:
                params["sample_size"] = st.slider(
                    "Targeted Customer Campaign Cohort Size (N):",
                    min_value=100,
                    max_value=2000,
                    value=600,
                    step=50,
                    help="Controls the number of e-commerce customer browsing and conversion profiles generated across demographic tiers."
                )
        elif selected_scenario_key == "subscription_churn_forecasting":
            with c1:
                params["sample_size"] = st.slider(
                    "Active Digital Subscriber Cohort Size (N):",
                    min_value=100,
                    max_value=2000,
                    value=500,
                    step=50,
                    help="Controls the number of digital subscription product telemetry records generated."
                )
        else:
            with c1:
                params["sample_size"] = st.slider(
                    "Simulated Patient Cohort Size (N):",
                    min_value=100,
                    max_value=2000,
                    value=500,
                    step=50,
                    help="Controls the exact number of synthetic patient records generated. Moving this slider dynamically updates cohort sample size, event counts, confusion matrices, and model metrics in real-time."
                )

        # 2. Decision Threshold / Forecasting Horizon
        is_ts = (model_family == "time_series" or selected_scenario_key in ("ed_demand_forecasting", "workforce_sickness_forecasting", "supply_chain_demand_forecasting"))
        if is_ts:
            with c2:
                params["horizon"] = st.slider(
                    "Out-of-Sample Forward Forecast Horizon:",
                    min_value=7,
                    max_value=28,
                    value=14,
                    step=1,
                    help="Number of forward periods (days or weeks) to forecast into the future."
                )
            with c3:
                params["seasonal_period"] = st.selectbox(
                    "Seasonal Rhythm / Period (m):",
                    options=[7, 14, 52],
                    format_func=lambda s: f"{s} Days (Day-of-Week Rhythm / Logistics Cycle)" if s == 7 else (f"{s} Days (Bi-Weekly Cycle)" if s == 14 else f"{s} Weeks (Annual Winter / Peak Cycle)"),
                    index=0 if ("demand" in selected_scenario_key) else 2
                )
                params["alpha"] = st.slider(
                    "Holt-Winters Level Smoothing (α):",
                    min_value=0.05,
                    max_value=0.80,
                    value=0.28,
                    step=0.01,
                    help="Weights recent throughput / attendance levels vs historical baseline."
                )
        else:
            with c2:
                default_th = 0.50
                if selected_scenario_key in ("operational_dna_equity", "marketing_conversion_scoring"):
                    default_th = 0.28
                elif selected_scenario_key == "manufacturing_predictive_maintenance":
                    default_th = 0.15
                elif selected_scenario_key == "subscription_churn_forecasting":
                    default_th = 0.26

                params["threshold"] = st.slider(
                    "Classification Decision Threshold (p*):",
                    min_value=0.05,
                    max_value=0.95,
                    value=default_th,
                    step=0.01,
                    help="Adjust probability threshold p* for assigning positive class. Lowering threshold maximizes sensitivity/recall at the cost of false positives."
                )
            with c3:
                if model_family in ("clustering", "dim_reduction"):
                    params["n_clusters"] = st.slider("Target Number of Phenotype Clusters (k):", 2, 6, 3)
                    params["n_components"] = st.slider("Retained Principal Components:", 2, 5, 2)
                else:
                    st.markdown(f"**Active Model:** `{model_info.get('name', selected_model_key)}`")
                    st.markdown(f"**Interpretability:** `{model_info.get('interpretability', 'Standard')} ({model_info.get('interpretability_score', 3)}/5)`")
                    st.caption("Threshold changes dynamically update the confusion matrix and ROC/PR curve points live.")

    return selected_scenario_key, meta, params


def render_ml_scenario_controls():
    """Backward-compatible alias for render_ml_scenario_and_parameters."""
    return render_ml_scenario_and_parameters("random_forest_classifier")


def render_ml_decision_wizard():
    """Backward-compatible alias for render_ml_guided_wizard."""
    return render_ml_guided_wizard()


# =====================================================================
# 3. Methodological Prerequisite Evaluation (5-Point Audit)
# =====================================================================

def render_ml_prerequisite_cards(meta: Dict[str, Any], params: Dict[str, Any]):
    """Renders the 5-point NHS AI and biostatistical methodological prerequisite audit."""
    task = meta.get("task_family", "classification")
    model_key = params.get("model_key", meta.get("recommended_model", "random_forest_classifier"))

    audit = evaluate_ml_prerequisites(
        model_key=model_key,
        task=task,
        n_samples=params.get("sample_size", 500),
        n_features=len(meta.get("clinical_features", [1, 2, 3, 4, 5])),
        n_events=int(params.get("sample_size", 500) * 0.08) if task == "classification" else None,
        is_imbalanced="imbalance_note" in meta,
        is_time_series=task == "time_series",
        has_protected_attrs="protected_attribute" in meta
    )

    with st.expander("📋 Biostatistical & NHS AI Ethics 5-Point Governance Audit", expanded=False):
        st.markdown(f"**Methodological Audit Verdict:** {audit['summary']}")

        cols = st.columns(len(audit["checks"]))
        for i, check in enumerate(audit["checks"]):
            with cols[i]:
                status_color = "#16a34a" if check["status"] == "pass" else "#d97706"
                bg_color = "#f0fdf4" if check["status"] == "pass" else "#fffbeb"

                st.markdown(f"""
                <div style="background-color: {bg_color}; border: 1px solid {status_color}40; border-radius: 6px; padding: 0.6rem; height: 100%;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: {status_color}; margin-bottom: 0.3rem;">
                        {check['badge']}
                    </div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #1e293b;">
                        {check['title']}
                    </div>
                    <div style="font-size: 0.78rem; color: #475569; margin-top: 0.3rem;">
                        {check['finding']}
                    </div>
                    <p style="font-size: 0.72rem; color: #64748b; margin-top: 0.4rem; line-height: 1.25;">
                        {check['explanation']}
                    </p>
                </div>
                """, unsafe_allow_html=True)


# =====================================================================
# 4. Primary Metric Dashboard & Executive KPI Summary
# =====================================================================

def render_ml_performance_metric_math_breakdown(
    scenario_key: str,
    task: str,
    params: Dict[str, Any],
    calc_results: Dict[str, Any]
):
    """Renders mathematical formulae and plain English step-by-step explanations for the active scenario metrics."""
    with st.expander("📐 Performance Metric Formulae & Step-by-Step Plain English Breakdown", expanded=False):
        if task == "fairness_audit" or scenario_key == "operational_dna_equity":
            st.markdown(r"""
            ### ⚖️ Algorithmic Fairness & Equity Audit Metrics

            #### 1. Disparate Impact Ratio (DIR) & Four-Fifths (80%) Rule
            ##### Mathematical Representation:
            $$\text{DIR} = \frac{\min_{a \in A} P(\hat{Y} = 1 \mid A = a)}{\max_{a \in A} P(\hat{Y} = 1 \mid A = a)} = \frac{\min_{a \in A} \left( \frac{\text{Flagged}_a}{N_a} \right)}{\max_{a \in A} \left( \frac{\text{Flagged}_a}{N_a} \right)}$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Calculate Group Selection Rate):** For each demographic or socio-economic subgroup $a \in A$ (e.g. Index of Multiple Deprivation IMD Deciles 1–3 vs 8–10), count how many patients were flagged by the algorithm for overbooking/support ($\text{Flagged}_a$), and divide by total scheduled patients in that group ($N_a$).
            * **Step 2 (Identify Minimum and Maximum Rates):** Scan across all demographic subgroups to find the group with the lowest selection rate ($\min$) and the group with the highest selection rate ($\max$).
            * **Step 3 (Divide Minimum by Maximum):** Divide the minimum rate by the maximum rate to compute the ratio.
            * **Step 4 (Evaluate Against the Benchmark):** Compare the resulting ratio against the international Four-Fifths standard ($\ge 0.80$):
              * **If $\text{DIR} \ge 0.80$:** The algorithm satisfies the demographic parity threshold.
              * **If $\text{DIR} < 0.80$:** Adverse disparate impact is present; the algorithm systematically flags one demographic at an unfairly depressed or elevated rate.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Mandated by the NHS AI Ethics Code of Conduct and NICE Evidence Standards Framework. When predictive models inform outpatient slot allocation, vulnerable patients facing transport barriers must not be penalised with disproportionate double-booking delays.

            ---

            #### 2. Demographic Parity Difference ($\Delta_{\text{DP}}$)
            ##### Mathematical Representation:
            $$\Delta_{\text{DP}} = \max_{a \in A} P(\hat{Y} = 1 \mid A = a) - \min_{a \in A} P(\hat{Y} = 1 \mid A = a)$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1:** Calculate the positive flagging percentage in each socio-economic tier.
            * **Step 2:** Subtract the minimum percentage from the maximum percentage.
            * **Step 3:** The resulting percentage gap quantifies the absolute disparity in intervention assignment across cohorts (e.g. a $14.2\%$ gap indicates a 14.2 percentage point difference between the most and least flagged tiers).

            ##### Why This Calculation is Being Used:
            * Provides an absolute percentage gap that is readily interpretable by NHS Trust health equity boards and clinical governance directors.

            ---

            #### 3. False Positive Rate (FPR) Parity & Double-Booking Overburden
            ##### Mathematical Representation:
            $$\text{FPR}_a = P(\hat{Y} = 1 \mid Y = 0, A = a) = \frac{\text{FP}_a}{\text{FP}_a + \text{TN}_a}$$
            $$\text{FPR Ratio} = \frac{\min_a \text{FPR}_a}{\max_a \text{FPR}_a}, \quad \Delta_{\text{FPR}} = |\text{FPR}_{\text{deprived}} - \text{FPR}_{\text{affluent}}|$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Filter to Attending Patients):** Filter exclusively to reliable patients who will actually attend their hospital appointments ($Y = 0$, unaffected cohort).
            * **Step 2 (Count False Positives):** Within each socio-economic tier, count how many attending patients were falsely misclassified as DNA risks ($\text{FP}_a$).
            * **Step 3 (Divide by Total Attenders):** Divide $\text{FP}_a$ by total attenders in that tier ($\text{FP}_a + \text{TN}_a$) to find the False Positive Rate.
            * **Step 4 (Compare Across Demographic Tiers):** If high-deprivation patients have an FPR of $28\%$ while affluent patients have an FPR of $11\%$, the algorithm inflicts a $2.5\times$ higher double-booking penalty on attending vulnerable patients.

            ##### Why This Calculation is Being Used:
            * A false positive in appointment overbooking causes two patients to arrive for the same consultation slot, generating prolonged waiting room delays and clinician rush. Auditing FPR ensures double-booking penalties do not fall disproportionately on vulnerable patients.

            ---

            #### 4. Equalized Odds Disparity ($\Delta_{\text{EO}}$)
            ##### Mathematical Representation:
            $$\Delta_{\text{EO}} = |\text{TPR}_{\text{group1}} - \text{TPR}_{\text{group2}}| + |\text{FPR}_{\text{group1}} - \text{FPR}_{\text{group2}}|$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1:** Calculate True Positive Rate (Sensitivity) for each group: $\frac{\text{TP}}{\text{TP} + \text{FN}}$.
            * **Step 2:** Calculate False Positive Rate for each group: $\frac{\text{FP}}{\text{FP} + \text{TN}}$.
            * **Step 3:** Compute the absolute differences in TPR and FPR between groups.
            * **Step 4:** Sum the two differences. Equalized Odds is satisfied when $\Delta_{\text{EO}} = 0$, meaning the algorithm exhibits equal accuracy and equal error rates across all patient backgrounds.

            ##### Why This Calculation is Being Used:
            * Recognized as the gold standard biostatistical fairness definition because it requires equal benefit (catching DNAs) and equal protection from harm (avoiding false alarms) across patient populations.
            """)

        elif task == "time_series" or scenario_key in ("ed_demand_forecasting", "workforce_sickness_forecasting", "supply_chain_demand_forecasting"):
            st.markdown(r"""
            ### ⏱️ Time Series & Operational Demand Forecasting Metrics

            #### 1. Mean Absolute Error (MAE)
            ##### Mathematical Representation:
            $$\text{MAE} = \frac{1}{H} \sum_{t=1}^H |y_t - \hat{y}_t|$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Calculate Error):** For each future time period $t$ across forward test horizon $H$ (e.g. Days 1 to 14), subtract the model's projected demand $\hat{y}_t$ from actual observed demand $y_t$.
            * **Step 2 (Absolute Value):** Take the absolute value $|y_t - \hat{y}_t|$, ensuring that under-predicting and over-predicting are both treated as positive operational errors.
            * **Step 3 (Sum Errors):** Sum these absolute errors across all $H$ test periods.
            * **Step 4 (Compute Average):** Divide by total periods $H$ to obtain the average error in natural units (e.g. $\pm 14.2$ patients/day for A&E demand, or $\pm 28.5$ pallets/day for warehouse freight dispatch).

            ##### Why This Calculation is Being Used & Real-World Context:
            * MAE represents the most intuitive operational metric for resource managers: *"On average, our daily A&E arrival or warehouse pallet dispatch forecast is off by 14 units."*

            ---

            #### 2. Root Mean Squared Error (RMSE)
            ##### Mathematical Representation:
            $$\text{RMSE} = \sqrt{\frac{1}{H} \sum_{t=1}^H (y_t - \hat{y}_t)^2}$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Calculate Error):** Compute the forecast error $(y_t - \hat{y}_t)$ for each period.
            * **Step 2 (Square the Error):** Square each error $(y_t - \hat{y}_t)^2$. This exponentially penalizes large forecast misses compared to small routine fluctuations.
            * **Step 3 (Average Squared Errors):** Sum the squared errors and divide by $H$ (Mean Squared Error).
            * **Step 4 (Take Square Root):** Take the square root to convert the metric back to the original operational scale (e.g. patients/day or pallets/day).

            ##### Why This Calculation is Being Used:
            * In hospital bed capacity and freight logistics, under-predicting an attendance surge or shipping bottleneck causes catastrophic gridlock and carrier demurrage charges. RMSE heavily penalizes large forecast misses.

            ---

            #### 3. Mean Absolute Percentage Error (MAPE)
            ##### Mathematical Representation:
            $$\text{MAPE} = \frac{100\%}{H} \sum_{t=1}^H \left| \frac{y_t - \hat{y}_t}{y_t} \right|$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Relative Error):** Divide the absolute error $|y_t - \hat{y}_t|$ by actual observed demand $y_t$ for each period.
            * **Step 2 (Sum Relative Errors):** Sum all relative errors across horizon $H$.
            * **Step 3 (Express as Percentage):** Divide by $H$ and multiply by $100\%$ to produce a dimensionless percentage.

            ##### Why This Calculation is Being Used:
            * Enables standardized cross-facility comparison regardless of facility scale (e.g. comparing a major regional teaching hospital seeing 450 attendances/day against a community hospital seeing 80 attendances/day, or a central distribution hub dispatching 1,200 pallets/day vs a local depot dispatching 250 pallets/day). In operational planning, $\text{MAPE} < 10\%$ is considered an operational pass.

            ---

            #### 4. Rolling-Origin (Expanding Window) Temporal Evaluation
            ##### Mathematical Representation:
            $$\mathcal{D}_{\text{train}}^{(k)} = \{y_1, y_2, \dots, y_{t_k}\}, \quad \mathcal{D}_{\text{test}}^{(k)} = \{y_{t_k + 1}, \dots, y_{t_k + h}\}$$
            $$\text{Validation Metric} = \frac{1}{K} \sum_{k=1}^K \text{Metric}\left(\mathcal{D}_{\text{test}}^{(k)}, \hat{y}^{(k)}\right)$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Fix Historical Baseline):** Train the model on historical attendance or dispatch data up to time $t_k$.
            * **Step 2 (Forecast Unseen Future):** Project forward $h$ periods into the immediate future without updating weights.
            * **Step 3 (Advance the Origin):** Roll the origin forward in time to include newly observed data, never shuffling future periods into the past.
            * **Step 4 (Average Across Windows):** Repeat across $K$ folds and average test errors.

            ##### Why This Calculation is Being Used:
            * Randomly shuffling time series data causes future-to-past data leakage, falsely inflating validation accuracy. Rolling-origin evaluation strictly mirrors real-world deployment.
            """)

        elif task in ("clustering", "unsupervised") or "clustering" in calc_results or params.get("model_key") in ("kmeans_clustering", "dbscan_clustering"):
            st.markdown(r"""
            ### 🔍 Unsupervised Clustering Validation & Partition Metrics

            #### 1. Silhouette Score ($s$)
            ##### Mathematical Representation:
            $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad \text{Silhouette} = \frac{1}{N} \sum_{i=1}^N s(i)$$
            where for patient observation $i$ assigned to cluster $C_I$:
            $$a(i) = \frac{1}{|C_I| - 1} \sum_{j \in C_I, j \neq i} \|x_i - x_j\|, \quad b(i) = \min_{J \neq I} \frac{1}{|C_J|} \sum_{j \in C_J} \|x_i - x_j\|$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Calculate Intra-Cluster Distance $a(i)$):** For patient observation $i$, compute the average Euclidean distance to all other patients in the *same* cluster. This measures internal cluster cohesion.
            * **Step 2 (Calculate Nearest-Cluster Distance $b(i)$):** For every other cluster $C_J$ ($J \neq I$), compute the average distance from patient $i$ to all patients in $C_J$. Find the minimum among all other clusters; this is the "neighboring cluster".
            * **Step 3 (Normalize Difference):** Subtract internal scatter from neighbor separation: $b(i) - a(i)$, and divide by $\max(a(i), b(i))$ to scale the metric strictly between $-1$ and $+1$.
            * **Step 4 (Average Across Dataset):** Average $s(i)$ over all $N$ patients.
            * **Interpretation Guide:**
              - **$s > 0.50$:** Strong, dense, and well-isolated cluster phenotypes.
              - **$0.25 \le s \le 0.50$:** Reasonable cluster structure; some overlap at boundaries.
              - **$s < 0.25$:** Weak structure, substantial cluster overlap, or artificial subdivision.
              - **$s < 0$:** Points are on average closer to a neighboring cluster than to their assigned cluster (misclustering).

            ##### Why This Calculation is Being Used:
            * Identifies whether discovered patient phenotypes (e.g. cardiac sub-phenotypes) reflect genuine biological separation rather than arbitrary geometric partitioning.

            ---

            #### 2. Davies-Bouldin Index ($\text{DBI}$)
            ##### Mathematical Representation:
            $$R_{i,j} = \frac{S_i + S_j}{M_{i,j}}, \quad R_i = \max_{j \neq i} R_{i,j}, \quad \text{DBI} = \frac{1}{k} \sum_{i=1}^k R_i$$
            where:
            $$S_i = \frac{1}{|C_i|} \sum_{x \in C_i} \|x - \mu_i\|, \quad M_{i,j} = \|\mu_i - \mu_j\|$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Compute Cluster Scatter $S_i$):** For each cluster $i$, compute the average Euclidean distance of all points in cluster $i$ to their centroid $\mu_i$.
            * **Step 2 (Compute Centroid Separation $M_{i,j}$):** For every pair of clusters $(i, j)$, measure the Euclidean distance between their centroids $\|\mu_i - \mu_j\|$.
            * **Step 3 (Compute Pairwise Similarity Ratio $R_{i,j}$):** Divide the combined internal scatter $(S_i + S_j)$ by the distance between their centroids $M_{i,j}$. High ratios indicate overlapping, poorly separated clusters.
            * **Step 4 (Find Worst-Case Similarity $R_i$):** For each cluster $i$, identify its worst-case (maximum) overlap ratio with any other cluster.
            * **Step 5 (Average Across Clusters):** Compute the arithmetic mean of $R_i$ across all $k$ clusters.
            * **Interpretation Guide:**
              - **$\text{DBI} < 1.0$:** Good clustering; clusters are compact and well-separated from each other.
              - **$\text{DBI} \ge 1.0$:** Clusters exhibit excessive dispersion or close proximity.
              - **Lower is Better:** The optimal number of clusters $k^*$ corresponds to the global minimum (trough) of the DBI curve.

            ##### Why This Calculation is Being Used:
            * Provides an intuitive ratio of within-cluster spread to between-cluster separation that is computationally efficient and penalizes overlapping phenotypes.

            ---

            #### 3. Calinski-Harabasz Index ($\text{CHI}$ / Variance Ratio Criterion)
            ##### Mathematical Representation:
            $$\text{CHI} = \frac{\text{SSB} / (k - 1)}{\text{SSW} / (N - k)} = \frac{\sum_{q=1}^k n_q \|\mu_q - \mu\|^2 / (k - 1)}{\sum_{q=1}^k \sum_{x \in C_q} \|x - \mu_q\|^2 / (N - k)}$$
            where $\mu$ is the overall center of all $N$ observations, $\mu_q$ is the centroid of cluster $q$, and $n_q = |C_q|$.

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Compute Between-Cluster Dispersion $\text{SSB}$):** Calculate how far each cluster centroid $\mu_q$ is from the overall dataset mean $\mu$, square this distance, multiply by cluster size $n_q$, and sum over all clusters. Divide by degrees of freedom $(k - 1)$.
            * **Step 2 (Compute Within-Cluster Dispersion $\text{SSW}$):** Calculate the squared Euclidean distance of each observation $x$ to its assigned cluster centroid $\mu_q$, and sum over all observations (this equals the total Inertia / WCSS). Divide by degrees of freedom $(N - k)$.
            * **Step 3 (Compute Variance Ratio):** Divide the normalized between-cluster variance by the normalized within-cluster variance.
            * **Interpretation Guide:**
              - **Higher is Better:** Higher values indicate tight clusters that are widely separated from each other.
              - The optimal number of clusters $k^*$ is typically identified by the maximum peak in the CHI curve.
              - Analogous to the $F$-statistic in One-Way ANOVA: it directly tests whether between-group variance significantly exceeds within-group noise.

            ##### Why This Calculation is Being Used:
            * Serves as an omnibus signal-to-noise ratio. Because it directly mirrors the $F$-ratio in classical variance decomposition, it is widely trusted by biostatisticians and machine learning researchers.

            ---

            #### 4. Within-Cluster Sum of Squares (WCSS / Inertia) & The Elbow Method
            ##### Mathematical Representation:
            $$\text{WCSS} = \sum_{q=1}^k \sum_{x \in C_q} \|x - \mu_q\|^2$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1:** For each point in cluster $q$, compute its squared distance to centroid $\mu_q$.
            * **Step 2:** Sum across all points and all clusters.
            * **The Elbow Inflection:** As $k$ increases, WCSS monotonically decreases. The "elbow" inflection point is where adding an additional cluster produces diminishing returns in variance reduction.
            """)

        else:
            st.markdown(r"""
            ### 🎯 Supervised Classification & Clinical Diagnostic Metrics

            #### 1. Precision-Recall Area Under the Curve (PR-AUC / Average Precision)
            ##### Mathematical Representation:
            $$\text{PR-AUC} = \sum_{k=1}^M (R_k - R_{k-1}) \cdot P_k$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Sort Predicted Probabilities):** Rank all samples in descending order of predicted positive event probability $\hat{p}_i$ (e.g. 30-day readmission, CNC spindle breakdown, or subscriber churn).
            * **Step 2 (Compute PR Points):** Sweep decision threshold $p^*$ from $1.0$ down to $0.0$, computing Precision ($P_k$) and Recall ($R_k$) at each operating point $k$.
            * **Step 3 (Calculate Incremental Recall):** Calculate the recall gain $(R_k - R_{k-1})$ between consecutive thresholds.
            * **Step 4 (Multiply by Precision):** Multiply the recall gain by the precision $P_k$ achieved at that threshold.
            * **Step 5 (Integrate Area):** Sum these trapezoidal slices across the entire curve.
            * **Baseline Benchmark:** A random uninformative model achieves $\text{PR-AUC} = \text{Prevalence}$ (e.g. $0.080$ for an $8\%$ clinical event rate, or $0.042$ for a $4.2\%$ machinery failure rate).

            ##### Why This Calculation is Being Used & Real-World Context:
            * In rare events (8% heart failure readmission, 4.2% industrial spindle failure), ROC-AUC is artificially inflated by the overwhelming majority of non-events (the Accuracy Paradox). PR-AUC focuses strictly on the minority target, evaluating true alert reliability.

            ---

            #### 2. Receiver Operating Characteristic Area (ROC-AUC / Concordance C-Statistic)
            ##### Mathematical Representation:
            $$\text{ROC-AUC} = P(\hat{p}_{\text{event}} > \hat{p}_{\text{non-event}}) = \frac{1}{N_{\text{pos}} \cdot N_{\text{neg}}} \sum_{i \in \text{Pos}} \sum_{j \in \text{Neg}} \left[ \mathbb{I}(\hat{p}_i > \hat{p}_j) + 0.5 \cdot \mathbb{I}(\hat{p}_i = \hat{p}_j) \right]$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Form All Cohort Pairs):** Pair every sample that experienced the event ($i \in \text{Pos}$) with every sample that remained negative ($j \in \text{Neg}$).
            * **Step 2 (Compare Scores):** If the positive sample was assigned a higher risk score than the negative sample, assign $1$ point; if scores are identical, assign $0.5$ points.
            * **Step 3 (Compute Proportion):** Sum points across all pairs and divide by the total number of pairs ($N_{\text{pos}} \times N_{\text{neg}}$).
            * **Interpretation:** $\text{ROC-AUC} = 0.50$ is random coin flipping; $\text{ROC-AUC} \ge 0.80$ indicates strong operational discrimination.

            ##### Why This Calculation is Being Used:
            * The universally accepted benchmark ($C$-statistic) for evaluating whether a prognostic score correctly ranks high-risk cases above low-risk cases.

            ---

            #### 3. F1-Score & Operational Threshold Optimization
            ##### Mathematical Representation:
            $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Compute Recall / Sensitivity):** Divide true positives caught ($\text{TP}$) by total positive events ($\text{TP} + \text{FN}$).
            * **Step 2 (Compute Precision / PPV):** Divide true positives caught ($\text{TP}$) by all alerts triggered ($\text{TP} + \text{FP}$).
            * **Step 3 (Multiply and Scale):** Multiply Precision by Recall, then multiply by $2$.
            * **Step 4 (Divide by Sum):** Divide by $(\text{Precision} + \text{Recall})$.
            * **Why Harmonic Mean?** If a model catches $90\%$ of events but has only $5\%$ precision (triggering hundreds of false alarms), the arithmetic mean is $47.5\%$, but the harmonic $F_1$ is only $9.5\%$, correctly identifying alert fatigue.

            ##### Why This Calculation is Being Used:
            * Balances operator alert fatigue (false alarms) against missed critical failures or deteriorations (false negatives) at a chosen decision threshold $p^*$.

            ---

            #### 4. Brier Score (Probabilistic Calibration)
            ##### Mathematical Representation:
            $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Calculate Probability Error):** For each observation $i$, subtract actual binary outcome $y_i \in \{0, 1\}$ from predicted probability $\hat{p}_i \in [0, 1]$.
            * **Step 2 (Square the Error):** Square the error $(\hat{p}_i - y_i)^2$.
            * **Step 3 (Average Across Cohort):** Sum squared errors across all $N$ observations and divide by $N$.
            * **Interpretation:** Perfect calibration yields $\text{BS} = 0.000$. Lower values indicate superior probabilistic reliability.

            ##### Why This Calculation is Being Used:
            * Discrimination (ROC-AUC) only measures ranking, not absolute risk. If a predictive model alerts maintenance engineers to a $30\%$ failure risk or clinicians to a $30\%$ readmission risk, that probability must be calibrated. Brier score evaluates probabilistic honesty for operational decision-making.

            ---

            #### 5. Foundational 2x2 Diagnostic Confusion Matrix Metrics
            ##### Mathematical Representation:
            $$\text{Sensitivity (Recall)} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
            $$\text{PPV (Precision)} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{NPV} = \frac{\text{TN}}{\text{TN} + \text{FN}}$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Sensitivity (True Positive Rate):** Of all actual positive events, what proportion did our algorithm successfully detect?
            * **Specificity (True Negative Rate):** Of all stable/normal cases, what proportion did our algorithm correctly identify as low risk?
            * **Positive Predictive Value (PPV):** When the algorithm sounds an alarm, what is the probability that the event actually occurs?
            * **Negative Predictive Value (NPV):** When the algorithm gives the green light, what is the probability that the case remains safely uneventful?

            ##### Why This Calculation is Being Used:
            * The gold-standard diagnostic framework required for operational and clinical decision support systems.
            """)


def render_ml_kpi_dashboard(
    scenario_key: str,
    meta: Dict[str, Any],
    params: Dict[str, Any],
    calc_results: Dict[str, Any]
):
    """Renders executive KPI metrics tailored to classification, time series, or equity audits."""
    task = meta.get("task_family", "classification")

    st.subheader("3. Executive Model Performance & Methodological Diagnostics")

    if task == "fairness_audit" or scenario_key in ("operational_dna_equity", "marketing_conversion_scoring"):
        # Equity Audit Dashboard
        eq = calc_results.get("equity", {})
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            ratio = eq.get("disparate_impact_ratio", 1.0)
            status_text = "Satisfied (Pass)" if ratio >= 0.80 else "Violated (Adverse Bias)"
            st.metric(
                label="Disparate Impact Ratio",
                value=f"{ratio:.2f}",
                delta=">= 0.80 (Four-Fifths Rule)" if ratio >= 0.80 else "- Disparity Alert (< 0.80)",
                delta_color="normal" if ratio >= 0.80 else "inverse"
            )
            st.caption(f"Status: **{status_text}**")

        with c2:
            dp_diff = eq.get("demographic_parity_difference_pct", 0.0)
            st.metric(
                label="Demographic Parity Gap",
                value=f"{dp_diff:.1f}%",
                delta="Selection Rate Difference",
                delta_color="off"
            )
            st.caption("Max vs Min group flag rate")

        with c3:
            fpr_diff = eq.get("fpr_disparity_difference_pct", 0.0)
            st.metric(
                label="False Positive Parity Gap",
                value=f"{fpr_diff:.1f}%",
                delta="High vs Low Tier FPR",
                delta_color="inverse" if fpr_diff > 10.0 else "normal"
            )
            st.caption("Higher FPR = Disproportionate penalty risk")

        with c4:
            eo_diff = eq.get("equalized_odds_difference_pct", 0.0)
            st.metric(
                label="Equalized Odds Disparity",
                value=f"{eo_diff:.1f}%",
                delta="Combined TPR + FPR Error Divergence",
                delta_color="off"
            )
            st.caption("Total error rate divergence")

    elif task == "time_series" or scenario_key in ("ed_demand_forecasting", "workforce_sickness_forecasting", "supply_chain_demand_forecasting"):
        # Time Series Forecasting Dashboard
        ts = calc_results.get("time_series", {})
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            horizon = len(ts.get("forecast_values", []))
            st.metric(
                label="Forecast Planning Horizon",
                value=f"{horizon} Periods",
                delta=f"Seasonal Cycle m = {ts.get('seasonal_period', 7)}",
                delta_color="off"
            )
            st.caption("Forward out-of-sample projection")

        with c2:
            mae = ts.get("validation_mae", 0.0)
            if scenario_key == "supply_chain_demand_forecasting":
                unit = "pallets/day"
            elif "demand" in scenario_key:
                unit = "patients/day"
            else:
                unit = "% staff absence"
            st.metric(
                label="Validation MAE",
                value=f"± {mae:.2f} {unit}",
                delta="Rolling-Origin Test Window",
                delta_color="off"
            )
            st.caption("Mean Absolute Prediction Error")

        with c3:
            rmse = ts.get("validation_rmse", 0.0)
            st.metric(
                label="Validation RMSE",
                value=f"{rmse:.2f}",
                delta="Root Mean Squared Error",
                delta_color="off"
            )
            st.caption("Penalizes large demand surges")

        with c4:
            mape = ts.get("validation_mape_pct", 0.0)
            st.metric(
                label="Validation MAPE",
                value=f"{mape:.1f}%",
                delta="Standard Capacity Error < 10%",
                delta_color="normal" if mape < 10.0 else "inverse"
            )
            st.caption("Percentage error relative to observed demand")

    elif task in ("clustering", "unsupervised") or "clustering" in calc_results or params.get("model_key") in ("kmeans_clustering", "dbscan_clustering"):
        # Unsupervised Clustering Dashboard
        clust = calc_results.get("clustering", {})
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            sil = clust.get("silhouette_score", 0.0)
            st.metric(
                label="Silhouette Score (Cohesion)",
                value=f"{sil:.3f}",
                delta=">= 0.50 (Dense & Separated)" if sil >= 0.50 else ("Fair (0.25 - 0.50)" if sil >= 0.25 else "Weak (< 0.25)"),
                delta_color="normal" if sil >= 0.50 else "off"
            )
            st.caption("Intra-cluster compactness vs nearest separation (-1 to +1)")

        with c2:
            dbi = clust.get("davies_bouldin_index", 0.0)
            st.metric(
                label="Davies-Bouldin Index (DBI)",
                value=f"{dbi:.3f}",
                delta="< 1.00 (Lower is Better)" if dbi < 1.0 else ">= 1.00 (Higher Dispersion)",
                delta_color="normal" if dbi < 1.0 else "inverse"
            )
            st.caption("Similarity ratio; lower values indicate superior clustering")

        with c3:
            chi = clust.get("calinski_harabasz_index", 0.0)
            st.metric(
                label="Calinski-Harabasz Index (CHI)",
                value=f"{chi:.1f}",
                delta="Variance Ratio Criterion (Higher is Better)",
                delta_color="off"
            )
            st.caption("Between-cluster to within-cluster dispersion ratio")

        with c4:
            inertia = clust.get("inertia_wcss", clust.get("inertia", 0.0))
            k_val = clust.get("k", params.get("n_clusters", 3))
            st.metric(
                label=f"Inertia (WCSS for k={k_val})",
                value=f"{inertia:.1f}",
                delta="Elbow Inflection Criterion",
                delta_color="off"
            )
            st.caption("Within-cluster sum of squared distances to centroids")

    else:
        # Supervised Classification Dashboard (Imbalanced Readmissions)
        clf = calc_results.get("classification", {})
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            pr_auc = clf.get("pr_auc", 0.0)
            base_prev = clf.get("base_positive_prevalence_pct", 8.0)
            gain = (pr_auc / (base_prev / 100.0)) if base_prev > 0 else 1.0
            st.metric(
                label="PR-AUC (Precision-Recall Area)",
                value=f"{pr_auc:.3f}",
                delta=f"{gain:.1f}x Over Baseline Random Chance",
                delta_color="normal"
            )
            st.caption(f"Baseline prevalence: **{base_prev:.1f}%** (PR-AUC is paramount)")

        with c2:
            roc_auc = clf.get("roc_auc", 0.0)
            st.metric(
                label="ROC-AUC Discrimination",
                value=f"{roc_auc:.3f}",
                delta="Concordance Index (C-Statistic)",
                delta_color="off"
            )
            st.caption("Pairwise ranking probability")

        with c3:
            f1 = clf.get("f1_score", 0.0)
            sens = clf.get("sensitivity_recall_pct", 0.0)
            prec = clf.get("precision_pct", 0.0)
            st.metric(
                label=f"F1-Score (Threshold {params.get('threshold', 0.5):.2f})",
                value=f"{f1:.3f}",
                delta=f"Recall: {sens:.1f}% | Precision: {prec:.1f}%",
                delta_color="off"
            )
            st.caption("Harmonic mean of Precision & Sensitivity")

        with c4:
            brier = clf.get("brier_score", 0.0)
            st.metric(
                label="Brier Score (Calibration)",
                value=f"{brier:.4f}",
                delta="Optimal = 0.000 (Mean Squared Error)",
                delta_color="normal" if brier < 0.10 else "inverse"
            )
            st.caption("Probabilistic risk score accuracy")

    # Expandable Metric Mathematical Formulae & Step-by-Step Breakdown
    render_ml_performance_metric_math_breakdown(scenario_key, task, params, calc_results)


# =====================================================================
# 5. Plotly Interactive Visualizers
# =====================================================================

def render_ml_visualizations(
    scenario_key: str,
    meta: Dict[str, Any],
    params: Dict[str, Any],
    calc_results: Dict[str, Any]
):
    """Renders high-fidelity Plotly interactive visualizers based on the clinical task."""
    st.subheader("4. Interactive Mathematical Visualizers")

    task = meta.get("task_family", "classification")

    if task == "fairness_audit" or scenario_key in ("operational_dna_equity", "marketing_conversion_scoring"):
        # Visualization 1: Algorithmic Fairness & Disparate Impact Bar Chart
        eq = calc_results.get("equity", {})
        group_stats = eq.get("group_stats", {})

        if group_stats:
            groups = list(group_stats.keys())
            sel_rates = [group_stats[g]["selection_rate_pct"] for g in groups]
            fpr_rates = [group_stats[g]["false_positive_rate_fpr_pct"] for g in groups]
            tpr_rates = [group_stats[g]["true_positive_rate_tpr_pct"] for g in groups]

            sel_name = "Selection Rate (% Flagged for Overbooking)" if scenario_key == "operational_dna_equity" else "Selection Rate (% Targeted with Promotion)"
            fpr_name = "False Positive Rate (% Wrongly Penalized)" if scenario_key == "operational_dna_equity" else "False Positive Rate (% Non-Buyers Promoted)"
            tpr_name = "True Positive Rate (% DNA Detected)" if scenario_key == "operational_dna_equity" else "True Positive Rate (% Buyers Detected)"

            fig = go.Figure()

            fig.add_trace(go.Bar(
                name=sel_name,
                x=groups,
                y=sel_rates,
                marker_color="#0284c7",
                text=[f"{v:.1f}%" for v in sel_rates],
                textposition="outside"
            ))

            fig.add_trace(go.Bar(
                name=fpr_name,
                x=groups,
                y=fpr_rates,
                marker_color="#dc2626",
                text=[f"{v:.1f}%" for v in fpr_rates],
                textposition="outside"
            ))

            fig.add_trace(go.Bar(
                name=tpr_name,
                x=groups,
                y=tpr_rates,
                marker_color="#16a34a",
                text=[f"{v:.1f}%" for v in tpr_rates],
                textposition="outside"
            ))

            # Reference baseline line for Four-Fifths threshold
            max_sel = max(sel_rates) if sel_rates else 100.0
            four_fifths_threshold = max_sel * 0.80

            fig.add_hline(
                y=four_fifths_threshold,
                line_dash="dash",
                line_color="#d97706",
                annotation_text=f"Four-Fifths (80%) Parity Threshold ({four_fifths_threshold:.1f}%)",
                annotation_position="top right"
            )

            chart_title = "Algorithmic Disparity & Equity Audit across Index of Multiple Deprivation (IMD) Tiers" if scenario_key == "operational_dna_equity" else f"Algorithmic Disparity & Equity Audit across Demographic Segments ({meta.get('fictional_trust', 'Retail Analytics')})"
            xaxis_title = "Deprivation Cohort (Index of Multiple Deprivation)" if scenario_key == "operational_dna_equity" else "Customer Demographic Segment"

            fig.update_layout(
                title=dict(
                    text=chart_title,
                    font=dict(size=15, color="#1e3a8a")
                ),
                barmode="group",
                xaxis_title=xaxis_title,
                yaxis_title="Percentage Rate (%)",
                yaxis=dict(range=[0, max(max(sel_rates + fpr_rates + tpr_rates) * 1.25, 40)]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=480,
                template="plotly_white",
                margin=dict(l=40, r=40, t=80, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

            if scenario_key == "marketing_conversion_scoring":
                st.markdown(f"""
                <div style="background-color: #fffbeb; border-left: 4px solid #d97706; padding: 0.6rem 0.9rem; border-radius: 4px; font-size: 0.86rem; color: #92400e;">
                    ⚖️ <strong>Algorithmic Equity Insight:</strong> {eq.get('verdict', '')}
                    Auditing Selection Rates and False Positive Rates across customer demographic tiers prevents digital redlining.
                    Notice whether promotional incentives or credit offerings are disproportionately concentrated in affluent segments at the exclusion of price-sensitive rural or low-income cohorts.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #fffbeb; border-left: 4px solid #d97706; padding: 0.6rem 0.9rem; border-radius: 4px; font-size: 0.86rem; color: #92400e;">
                    ⚖️ <strong>Health Equity Insight:</strong> {eq.get('verdict', '')}
                    Notice the disparity in False Positive Rate between High Deprivation (Deciles 1–3) and Low Deprivation (Deciles 8–10).
                    If clinics use this algorithm to overbook slots without providing travel aid, deprived patients suffer disproportionate waiting penalties.
                </div>
                """, unsafe_allow_html=True)

    elif task == "time_series" or scenario_key in ("ed_demand_forecasting", "workforce_sickness_forecasting", "supply_chain_demand_forecasting"):
        # Visualization 2: Time Series Demand Forecast with 95% Intervals
        ts = calc_results.get("time_series", {})
        obs = ts.get("historical_observed", [])
        fitted = ts.get("historical_fitted", [])
        forecast = ts.get("forecast_values", [])
        low_95 = ts.get("lower_95_prediction_interval", [])
        high_95 = ts.get("upper_95_prediction_interval", [])

        n_hist = len(obs)
        n_fc = len(forecast)

        x_hist = list(range(1, n_hist + 1))
        x_fc = list(range(n_hist + 1, n_hist + n_fc + 1))

        # Show last 45 historical periods to make the plot clean and readable
        display_window = min(45, n_hist)
        x_hist_disp = x_hist[-display_window:]
        obs_disp = obs[-display_window:]
        fitted_disp = fitted[-display_window:]

        if scenario_key == "supply_chain_demand_forecasting":
            chart_title = f"Regional Warehouse Dispatch Projection ({meta.get('fictional_trust', 'MetroLogix')}): 14-Day Demand & 95% Confidence Corridor"
            y_title = "Daily Regional Warehouse Pallets (Count)"
            obs_name = "Observed Pallet Dispatches"
            fit_name = "Holt-Winters Fitted Dispatch Trend & Cycle"
            fc_name = f"{n_fc}-Day Forward Dispatch Forecast"
            x_title = "Sequential Day Index"
        elif scenario_key == "workforce_sickness_forecasting":
            chart_title = "Clinical Workforce Absence Projection: Weekly Absence Rate & 95% Confidence Corridor"
            y_title = "Frontline Staff Sickness Absence (%)"
            obs_name = "Observed Sickness Rate"
            fit_name = "SARIMA Fitted Baseline & Seasonality"
            fc_name = f"{n_fc}-Week Forward Absence Forecast"
            x_title = "Sequential Week Index"
        else:
            chart_title = "Hospital Emergency Attendance Projection: Forward Arrivals & 95% Confidence Corridor"
            y_title = "Daily Emergency Department Attendances (Count)"
            obs_name = "Observed Attendances"
            fit_name = "Holt-Winters Fitted Level & Seasonality"
            fc_name = f"{n_fc}-Day Forward Forecast"
            x_title = "Sequential Day Index"

        fig = go.Figure()

        # Shaded 95% prediction interval band
        fig.add_trace(go.Scatter(
            x=x_fc + x_fc[::-1],
            y=high_95 + low_95[::-1],
            fill="toself",
            fillcolor="rgba(2, 132, 199, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Forward Prediction Interval"
        ))

        # Observed historical points
        fig.add_trace(go.Scatter(
            x=x_hist_disp,
            y=obs_disp,
            mode="lines+markers",
            name=obs_name,
            line=dict(color="#334155", width=1.5),
            marker=dict(size=4)
        ))

        # Historical fitted values
        fig.add_trace(go.Scatter(
            x=x_hist_disp,
            y=fitted_disp,
            mode="lines",
            name=fit_name,
            line=dict(color="#10b981", width=1.5, dash="dot")
        ))

        # Forward forecast
        fig.add_trace(go.Scatter(
            x=x_fc,
            y=forecast,
            mode="lines+markers",
            name=fc_name,
            line=dict(color="#0284c7", width=2.5),
            marker=dict(size=6, symbol="diamond")
        ))

        fig.update_layout(
            title=dict(
                text=chart_title,
                font=dict(size=15, color="#1e3a8a")
            ),
            xaxis_title=x_title,
            yaxis_title=y_title,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=480,
            template="plotly_white",
            margin=dict(l=40, r=40, t=80, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

    elif task in ("clustering", "unsupervised") or "clustering" in calc_results or params.get("model_key") in ("kmeans_clustering", "dbscan_clustering"):
        # Visualization 3: 2D Feature Partitioning & Multi-Metric Cluster Validation Curves (All 3 Metrics)
        clust = calc_results.get("clustering", {})
        pts = clust.get("points", [])
        assignments = clust.get("assignments", [])
        centroids = clust.get("centroids", [])
        k_eval = clust.get("k_evaluation_curve", [])

        cur_k = clust.get("k", params.get("n_clusters", 3))
        cur_sil = clust.get("silhouette_score", 0.5)
        cur_dbi = clust.get("davies_bouldin_index", 0.8)
        cur_chi = clust.get("calinski_harabasz_index", 300.0)

        c_chart1, c_chart2 = st.columns(2)

        # Left Chart: 2D Feature Space & Discovered Clusters
        with c_chart1:
            fig1 = go.Figure()
            palette = ["#0284c7", "#16a34a", "#d97706", "#dc2626", "#8b5cf6", "#06b6d4"]
            unique_c = sorted(list(set(assignments))) if assignments else [0]

            for c_idx in unique_c:
                c_pts = [pts[i] for i in range(len(pts)) if i < len(assignments) and assignments[i] == c_idx]
                if c_pts:
                    xs = [p["X1"] for p in c_pts]
                    ys = [p["X2"] for p in c_pts]
                    color = palette[c_idx % len(palette)]
                    fig1.add_trace(
                        go.Scatter(
                            x=xs, y=ys,
                            mode="markers",
                            name=f"Cluster {c_idx + 1} (N={len(c_pts)})",
                            marker=dict(size=7, color=color, opacity=0.7)
                        )
                    )

            if centroids:
                c_xs = [c[0] for c in centroids]
                c_ys = [c[1] for c in centroids]
                fig1.add_trace(
                    go.Scatter(
                        x=c_xs, y=c_ys,
                        mode="markers+text",
                        name="Centroids",
                        text=[f"C{i+1}" for i in range(len(centroids))],
                        textposition="top center",
                        marker=dict(size=14, color="#0f172a", symbol="diamond-wide", line=dict(width=2, color="#ffffff"))
                    )
                )

            fig1.update_layout(
                title=dict(
                    text=f"2D Feature Partitioning (k = {cur_k}, Silhouette = {cur_sil:.3f})",
                    font=dict(size=14, color="#1e3a8a")
                ),
                xaxis_title="Feature X1 (Standardized)",
                yaxis_title="Feature X2 (Standardized)",
                height=490,
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11)
                ),
                margin=dict(l=40, r=20, t=50, b=85)
            )
            st.plotly_chart(fig1, use_container_width=True)

        # Right Chart: Multi-Metric Cluster Validation Curves (All 3 Metrics: Silhouette, DBI, CHI)
        with c_chart2:
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])

            if k_eval:
                ks = [item["k"] for item in k_eval]
                sils = [item["Silhouette_Score"] for item in k_eval]
                dbis = [item["Davies_Bouldin_Index"] for item in k_eval]
                chis = [item["Calinski_Harabasz_Index"] for item in k_eval]

                # Metric 1: Silhouette Score (Left Axis, Peak is Optimal)
                fig2.add_trace(
                    go.Scatter(
                        x=ks, y=sils,
                        mode="lines+markers",
                        name="Silhouette Score (Left Axis, Peak = Optimal)",
                        line=dict(color="#16a34a", width=2.5),
                        marker=dict(size=8, symbol="circle")
                    ),
                    secondary_y=False
                )

                # Metric 2: Davies-Bouldin Index (Left Axis, Trough is Optimal)
                fig2.add_trace(
                    go.Scatter(
                        x=ks, y=dbis,
                        mode="lines+markers",
                        name="Davies-Bouldin (Left Axis, Trough = Optimal)",
                        line=dict(color="#dc2626", width=2.5, dash="dash"),
                        marker=dict(size=8, symbol="square")
                    ),
                    secondary_y=False
                )

                # Metric 3: Calinski-Harabasz Index (Right Secondary Axis, Peak is Optimal)
                fig2.add_trace(
                    go.Scatter(
                        x=ks, y=chis,
                        mode="lines+markers",
                        name="Calinski-Harabasz (Right Axis, Peak = Optimal)",
                        line=dict(color="#7c3aed", width=2.5, dash="dot"),
                        marker=dict(size=8, symbol="diamond")
                    ),
                    secondary_y=True
                )

            fig2.update_xaxes(title_text="Number of Clusters (k)", tickvals=[2, 3, 4, 5, 6])
            fig2.update_yaxes(title_text="Silhouette & Davies-Bouldin Score", secondary_y=False)
            fig2.update_yaxes(title_text="Calinski-Harabasz Index (CHI)", secondary_y=True)

            fig2.update_layout(
                title=dict(
                    text="Validation Metrics across k = 2 to 6 (All 3 Metrics)",
                    font=dict(size=14, color="#1e3a8a")
                ),
                height=490,
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                ),
                margin=dict(l=40, r=40, t=50, b=85)
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
        <div style="background-color: #f8fafc; border-left: 4px solid #16a34a; padding: 0.65rem 0.95rem; border-radius: 4px; font-size: 0.86rem; color: #1e293b; line-height: 1.5;">
            💡 <strong>Unsupervised Cluster Diagnostics (k = {cur_k}):</strong> 
            <div style="display: flex; gap: 1.5rem; margin: 0.35rem 0; flex-wrap: wrap;">
                <span>🟢 <strong>Silhouette Score:</strong> <code>{cur_sil:.3f}</code> (Left Axis, peak = optimal, &gt; 0.50 denotes strong cohesion)</span>
                <span>🔴 <strong>Davies-Bouldin:</strong> <code>{cur_dbi:.3f}</code> (Left Axis, trough = optimal, &lt; 1.00 denotes compact separation)</span>
                <span>🟣 <strong>Calinski-Harabasz:</strong> <code>{cur_chi:.1f}</code> (Right Axis, peak = optimal, variance ratio criterion)</span>
            </div>
            The right-hand curve traces all 3 validation metrics across candidate values of $k$: optimal clustering is indicated where the <strong>Silhouette Score</strong> and <strong>Calinski-Harabasz Index</strong> peak, and where the <strong>Davies-Bouldin Index</strong> reaches its minimum trough.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Visualization 3: Dual ROC Curve & Precision-Recall Curve with Dynamic Operating Point
        clf = calc_results.get("classification", {})
        roc_pts = clf.get("roc_points", [])
        pr_pts = clf.get("pr_points", [])

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Receiver Operating Characteristic (ROC-AUC = {clf.get('roc_auc', 0.85):.3f})",
                f"Precision-Recall Curve (PR-AUC = {clf.get('pr_auc', 0.48):.3f})"
            )
        )

        # 1. ROC Curve
        fprs = [p["FPR"] for p in roc_pts]
        tprs = [p["TPR"] for p in roc_pts]
        fig.add_trace(
            go.Scatter(x=fprs, y=tprs, mode="lines", name="ROC Trajectory", line=dict(color="#2563eb", width=2.5)),
            row=1, col=1
        )
        # ROC Chance diagonal
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Chance", line=dict(color="#94a3b8", dash="dash")),
            row=1, col=1
        )
        # Current operating point on ROC
        cur_fpr = clf.get("fpr", 0.15)
        cur_tpr = clf.get("recall_sensitivity", 0.70)
        fig.add_trace(
            go.Scatter(
                x=[cur_fpr], y=[cur_tpr],
                mode="markers",
                name=f"Current Threshold ({params.get('threshold', 0.5):.2f})",
                marker=dict(size=11, color="#dc2626", symbol="star")
            ),
            row=1, col=1
        )

        # 2. Precision-Recall Curve
        recalls = [p["Recall"] for p in pr_pts]
        precisions = [p["Precision"] for p in pr_pts]
        fig.add_trace(
            go.Scatter(x=recalls, y=precisions, mode="lines", name="PR Trajectory", line=dict(color="#059669", width=2.5)),
            row=1, col=2
        )
        # Baseline prevalence horizontal line
        base_prev = clf.get("base_positive_prevalence_pct", 8.0) / 100.0
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[base_prev, base_prev], mode="lines", name="Prevalence Chance", line=dict(color="#cbd5e1", dash="dot")),
            row=1, col=2
        )
        # Current operating point on PR
        cur_prec = clf.get("precision", 0.40)
        fig.add_trace(
            go.Scatter(
                x=[cur_tpr], y=[cur_prec],
                mode="markers",
                name="Operating Point",
                showlegend=False,
                marker=dict(size=11, color="#dc2626", symbol="star")
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="False Positive Rate (1 - Specificity)", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate (Sensitivity / Recall)", range=[0, 1.05], row=1, col=1)
        fig.update_xaxes(title_text="Recall (Sensitivity)", range=[0, 1], row=1, col=2)
        fig.update_yaxes(title_text="Precision (Positive Predictive Value)", range=[0, 1.05], row=1, col=2)

        fig.update_layout(
            height=460,
            template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        target_entity = "unplanned equipment failures" if "manufacturing" in scenario_key else ("subscriber cancellations" if "subscription" in scenario_key else "30-day HF readmissions")
        pos_label = "tool breakdown alerts" if "manufacturing" in scenario_key else ("churn risk accounts" if "subscription" in scenario_key else "true high-risk patients")

        st.markdown(f"""
        <div style="background-color: #f8fafc; border-left: 4px solid #0284c7; padding: 0.6rem 0.9rem; border-radius: 4px; font-size: 0.86rem; color: #1e293b;">
            💡 <strong>Discrimination Diagnostics:</strong> 
            At decision threshold <code>p* = {params.get('threshold', 0.50):.2f}</code>, the classifier achieves 
            <strong>{clf.get('sensitivity_recall_pct', 70.0):.1f}% Sensitivity</strong> (identifying {clf.get('tp', 0)} {pos_label}) 
            with <strong>{clf.get('precision_pct', 40.0):.1f}% Precision</strong>. 
            Because target events ({target_entity}) are relatively rare ({base_prev*100:.1f}%), the PR curve is far more informative than the ROC curve: 
            it proves the model achieves a <strong>{clf.get('pr_auc', 0.48) / max(0.01, base_prev):.1f}x precision improvement</strong> over random guessing.
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# 6. Full 4-Tab Educational Architecture
# =====================================================================

def render_ml_model_mathematical_framework(
    model_key: str,
    model_info: Dict[str, Any],
    meta: Dict[str, Any],
    params: Dict[str, Any]
):
    """Renders comprehensive mathematical formulations with plain English step-by-step explanations for the active model."""
    with st.expander("📐 Mathematical Formulations & Loss Functions", expanded=False):
        if model_key in ("penalized_logistic_regression", "logistic_regression"):
            st.markdown(r"""
            #### 1. Logistic Sigmoid Function & Linear Log-Odds Formulation
            ##### Mathematical Representation:
            $$P(Y = 1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = \beta_0 + \sum_{j=1}^p \beta_j X_j$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Calculate Linear Log-Odds $z$):** For an individual patient, multiply each recorded clinical biomarker $X_j$ (e.g. systolic BP, prior admissions, ejection fraction) by its estimated regression coefficient $\beta_j$, and add baseline intercept $\beta_0$. This calculates linear risk score $z \in (-\infty, +\infty)$.
            * **Step 2 (Negative Exponentiation $e^{-z}$):** Calculate the mathematical inverse exponential of the risk score. As risk score $z$ increases, $e^{-z}$ decays exponentially towards $0$.
            * **Step 3 (Denominator Formation $1 + e^{-z}$):** Add $1$ to the exponential factor, guaranteeing the denominator is strictly greater than $1$.
            * **Step 4 (Sigmoid Probability Mapping $\sigma(z)$):** Divide $1$ by $(1 + e^{-z})$. This mathematically bounds the unbounded score $z$ into a valid clinical probability $P \in (0, 1)$.
            * **Step 5 (Clinical Classification):** Compare predicted probability $P$ against classification threshold $p^*$ (e.g. $p^* = 0.50$ or $0.28$) to assign high-risk flags.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * In clinical trials and NHS governance, logistic regression is the gold standard because the exponential of each coefficient $\text{OR}_j = e^{\beta_j}$ represents an exact **Adjusted Odds Ratio**. Clinicians can directly audit why an individual was flagged (e.g. each 10 mmHg increase in systolic BP multiplies readmission odds by $1.15$).

            ---

            #### 2. Elastic Net Regularized Objective & Log-Loss Function
            ##### Mathematical Representation:
            $$\mathcal{L}_{\text{ElasticNet}}(\beta) = -\sum_{i=1}^n \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right] + \lambda \left[ \alpha \sum_{j=1}^p |\beta_j| + \frac{1 - \alpha}{2} \sum_{j=1}^p \beta_j^2 \right]$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Binary Cross-Entropy Loss):** For each patient $i$, compute the log-loss between their true clinical outcome $y_i \in \{0, 1\}$ and predicted probability $\hat{p}_i$. If a patient who readmits was predicted $99\%$, the penalty is near zero; if predicted $1\%$, the penalty is huge.
            * **Step 2 (L1 Lasso Penalty $\sum |\beta_j|$):** Sum the absolute values of all clinical weights. This drives negligible variables strictly to zero, providing automated feature selection.
            * **Step 3 (L2 Ridge Penalty $\sum \beta_j^2$):** Sum the squared coefficients. This shrinks correlated clinical features jointly, stabilizing collinear markers (e.g. urea, creatinine, and eGFR).
            * **Step 4 (Elastic Net Mixing $\alpha$ & Strength $\lambda$):** Parameter $\alpha \in [0, 1]$ balances Lasso ($\alpha=1$) versus Ridge ($\alpha=0$), while $\lambda$ controls total penalty strength.

            ##### Why This Calculation is Being Used:
            * Electronic Health Record (EHR) datasets have high collinearity. Standard unpenalized regression produces unstable, wildly oscillating coefficients. Elastic Net stabilizes estimation while pruning redundant laboratory tests.
            """)

        elif model_key == "decision_tree_classifier":
            st.markdown(r"""
            #### 1. Recursive Binary Splitting via Gini Impurity Reduction
            ##### Mathematical Representation:
            $$\Delta I_G(s, t) = I_G(t) - \left[ \frac{N_L}{N_t} I_G(t_L) + \frac{N_R}{N_t} I_G(t_R) \right], \quad \text{where } I_G(t) = 1 - \sum_{k=1}^K p(k \mid t)^2$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Measure Parent Node Impurity):** In parent patient node $t$, calculate the proportion of patients in each outcome class $p(k \mid t)$ (e.g. $8\%$ readmitted, $92\%$ stable). Square the proportions, sum them, and subtract from $1.0$ to determine parent Gini impurity $I_G(t)$.
            * **Step 2 (Evaluate Clinical Threshold $s$):** Test a prospective decision split on clinical feature $X_j$ (e.g. Left Ventricular Ejection Fraction $\text{LVEF} < 35\%$).
            * **Step 3 (Partition Cohort):** Split the $N_t$ patients into Left child node $t_L$ ($N_L$ patients) and Right child node $t_R$ ($N_R$ patients).
            * **Step 4 (Weighted Child Impurity):** Compute Gini impurity in each child node, and calculate their sample-weighted sum: $\frac{N_L}{N_t} I_G(t_L) + \frac{N_R}{N_t} I_G(t_R)$.
            * **Step 5 (Maximize Impurity Drop):** Subtract weighted child impurity from parent impurity to find Gini Gain $\Delta I_G$. The algorithm selects the biomarker split that produces the greatest purity gain.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Single decision trees generate transparent, flowchart-style clinical decision ladders (like NEWS2 triage protocols) that clinicians can audit and follow step-by-step at the bedside without software.

            ---

            #### 2. Cost-Complexity Pruning Objective
            ##### Mathematical Representation:
            $$R_\alpha(T) = R(T) + \alpha |T|$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1:** Calculate the training misclassification error rate $R(T)$ of full tree $T$.
            * **Step 2:** Count the total number of terminal leaf nodes $|T|$ (tree complexity).
            * **Step 3:** Multiply leaf count by complexity penalty parameter $\alpha \ge 0$.
            * **Step 4:** Select the pruned subtree that minimizes $R_\alpha(T)$ via cross-validation, cutting back noisy branches that do not generalize.

            ##### Why This Calculation is Being Used:
            * Unpruned trees easily memorize noise in small hospital cohorts. Pruning prevents overfitting and ensures clinical rules remain simple and generalizable.
            """)

        elif model_key == "random_forest_classifier":
            st.markdown(r"""
            #### 1. Bootstrap Aggregation (Bagging) Ensemble Probability
            ##### Mathematical Representation:
            $$\hat{P}(Y = 1 \mid x) = \frac{1}{B} \sum_{b=1}^B T_b(x; \Theta_b)$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Bootstrap Resampling):** Randomly draw $B$ synthetic training cohorts (each of size $N$, sampled with replacement) from the master registry.
            * **Step 2 (Decorrelated Tree Training):** Grow an independent decision tree $T_b$ on each bootstrap bag. At each split, evaluate only a random subset of candidate features ($m \approx \sqrt{p}$) to decorrelate the trees.
            * **Step 3 (Individual Tree Voting):** For a new patient with clinical profile $x$, pass their features down all $B$ trees to terminal leaf nodes.
            * **Step 4 (Ensemble Averaging):** Average the predicted probabilities across all $B$ trees. Individual tree variance cancels out, yielding a stable, low-variance risk prediction.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * A single decision tree has high variance and overfits to local hospital idiosyncrasies. Bagging decorrelated trees reduces prediction error without increasing bias, while capturing multi-way clinical biomarker interactions (e.g. low ejection fraction combined with renal impairment).

            ---

            #### 2. Node Splitting Criterion — Gini Impurity
            ##### Mathematical Representation:
            $$I_G(t) = 1 - \sum_{k=1}^K p(k \mid t)^2$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1:** In tree node $t$, calculate the proportion $p(k \mid t)$ of patients in each outcome class (e.g. readmitted vs non-readmitted).
            * **Step 2:** Square each proportion ($p_0^2$ and $p_1^2$).
            * **Step 3:** Sum the squared proportions. If a node is completely mixed (50/50), the sum is $0.5^2 + 0.5^2 = 0.50$. If completely pure (100% readmitted), the sum is $1.0^2 = 1.00$.
            * **Step 4:** Subtract from $1.0$. A pure node yields $I_G = 0$; maximum impurity yields $I_G = 0.50$.
            * **Step 5:** Choose the biomarker split that maximizes impurity reduction (Gini gain) between the parent node and the child nodes.

            ##### Why This Calculation is Being Used:
            * Computationally efficient splitting criterion that partitions complex patient cohorts into homogeneous risk groups without requiring linear boundaries.
            """)

        elif model_key == "gradient_boosting_classifier":
            st.markdown(r"""
            #### 1. Sequential Functional Gradient Descent Update
            ##### Mathematical Representation:
            $$F_m(x) = F_{m-1}(x) + \eta \sum_{j=1}^{J_m} \gamma_{jm} \mathbb{I}(x \in R_{jm})$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Initial Prediction):** Start with a constant baseline model $F_0(x)$ equal to the log-odds of the overall training cohort event rate.
            * **Step 2 (Compute Pseudo-Residuals):** At iteration $m$, compute gradient residuals $r_{im} = y_i - \sigma(F_{m-1}(x_i))$ for every patient, measuring remaining prediction errors.
            * **Step 3 (Fit Weak Learner):** Train a shallow regression tree to predict these residuals, partitioning patients into terminal regions $R_{jm}$.
            * **Step 4 (Optimal Leaf Step Size):** Compute optimal leaf values $\gamma_{jm}$ using Newton-Raphson approximation.
            * **Step 5 (Shrinkage & Update):** Multiply leaf values by learning rate $\eta \in (0, 1]$ (e.g. $\eta = 0.05$) and add to current model $F_{m-1}(x)$.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Unlike Random Forest where trees are built independently in parallel, Gradient Boosting focuses computational effort specifically on patients misclassified by prior trees, achieving highest discrimination on challenging tabular records.

            ---

            #### 2. Log-Loss Negative Gradient (Pseudo-Residuals)
            ##### Mathematical Representation:
            $$r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)} \right]_{F=F_{m-1}} = y_i - \hat{p}_i^{(m-1)}$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1:** For patient $i$, take their actual outcome $y_i \in \{0, 1\}$.
            * **Step 2:** Subtract the current model's predicted probability $\hat{p}_i^{(m-1)} = \sigma(F_{m-1}(x_i))$.
            * **Step 3:** The difference $(y_i - \hat{p}_i)$ represents the direction and magnitude of error. If a readmitted patient ($y=1$) only had predicted probability $0.20$, the residual is $+0.80$, compelling the next tree to focus on this patient.

            ##### Why This Calculation is Being Used:
            * Converts complex non-linear classification into a sequence of straightforward residual fitting steps.
            """)

        elif model_key == "svm_classifier":
            st.markdown(r"""
            #### 1. Soft-Margin Support Vector Optimization with Slack Variables
            ##### Mathematical Representation:
            $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{subject to } y_i \left( w^T \phi(x_i) + b \right) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Maximize Geometric Margin):** The physical distance between the separating boundary and the nearest patient data points is $\frac{2}{\|w\|}$. Minimizing $\frac{1}{2}\|w\|^2$ maximizes this safety margin.
            * **Step 2 (Introduce Slack $\xi_i$):** Real-world clinical biomarkers overlap; slack variables $\xi_i \ge 0$ permit controlled misclassifications.
            * **Step 3 (Penalty Weight $C$):** Regularization parameter $C$ dictates the trade-off: large $C$ heavily penalizes misclassified patients; small $C$ prioritizes a wider, more generalizable margin.
            * **Step 4 (Kernel Mapping $\phi(x)$):** Transforms non-linear clinical features into a higher-dimensional space where classes become linearly separable.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Support Vector Machines excel in high-dimensional biomarker datasets ($p > n$, e.g. genomic expression or proteomics) where finding a wide geometric safety margin prevents overfitting.

            ---

            #### 2. Radial Basis Function (RBF) Non-Linear Kernel
            ##### Mathematical Representation:
            $$K(x, x') = \exp\left( -\gamma \| x - x' \|^2 \right)$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1:** Calculate squared Euclidean distance $\|x - x'\|^2$ between two patients' biomarker vectors.
            * **Step 2:** Multiply by kernel scale parameter $\gamma > 0$.
            * **Step 3:** Take negative exponential. Two identical patients yield $K(x, x') = e^0 = 1.0$; patients with vastly different physiology approach $0.0$.

            ##### Why This Calculation is Being Used:
            * Computes inner products in infinite-dimensional space without ever explicitly computing coordinate transformations (the "Kernel Trick").
            """)

        elif model_key == "knn_classifier":
            st.markdown(r"""
            #### 1. Standardized Multidimensional Euclidean Distance Metric
            ##### Mathematical Representation:
            $$d(x, x_i) = \sqrt{\sum_{j=1}^p \left( \frac{x_j - x_{ij}}{\sigma_j} \right)^2}$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Biomarker Normalization):** Divide the difference in clinical marker $j$ by its standard deviation $\sigma_j$. This prevents variables with large numerical values (e.g. Platelets $250,000$) from drowning out small values (e.g. Creatinine $1.2$).
            * **Step 2 (Square Standardized Differences):** Square each feature difference $(\frac{x_j - x_{ij}}{\sigma_j})^2$.
            * **Step 3 (Sum Across Features):** Sum squared differences across all $p$ clinical variables.
            * **Step 4 (Square Root):** Take the square root to calculate the exact geometric distance $d(x, x_i)$ between query patient $x$ and historical patient $x_i$.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Non-parametric case-based reasoning: directly measures physiological similarity between a newly admitted patient and historical electronic health records.

            ---

            #### 2. k-Nearest Neighbors Probability Voting
            ##### Mathematical Representation:
            $$\hat{P}(Y = 1 \mid x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} \mathbb{I}(y_i = 1)$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1:** Sort all historical patients by distance $d(x, x_i)$ and select the $k$ closest individuals ($\mathcal{N}_k(x)$).
            * **Step 2:** Count how many of these $k$ closest patients experienced 30-day emergency readmission ($y_i = 1$).
            * **Step 3:** Divide by $k$ to compute the estimated probability.

            ##### Why This Calculation is Being Used:
            * Provides an intuitive clinical answer: *"Among the 9 patients in our registry most physiologically identical to this patient, 3 required emergency readmission ($33\%$ risk)."*
            """)

        elif model_key == "elastic_net_regression":
            st.markdown(r"""
            #### 1. Elastic Net Continuous Regularized Least Squares Objective
            ##### Mathematical Representation:
            $$\min_{\beta} \frac{1}{2n} \sum_{i=1}^n \left( y_i - \beta_0 - \sum_{j=1}^p \beta_j X_{ij} \right)^2 + \lambda \left[ \alpha \sum_{j=1}^p |\beta_j| + \frac{1 - \alpha}{2} \sum_{j=1}^p \beta_j^2 \right]$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Mean Squared Error):** Measure the squared difference between actual hospital length of stay $y_i$ (days) and predicted stay $\hat{y}_i$.
            * **Step 2 (L1 Lasso Penalty):** Sum absolute coefficient values $\alpha \sum |\beta_j|$ to shrink uninformative predictors to zero.
            * **Step 3 (L2 Ridge Penalty):** Sum squared coefficients $\frac{1-\alpha}{2} \sum \beta_j^2$ to stabilize grouped collinear predictors.
            * **Step 4 (Regularization Parameter $\lambda$):** Balances data fitting against complexity penalties.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Predicts continuous hospital length of stay and bed-day consumption while automatically selecting sparse predictors from hundreds of routine clinical laboratory variables.
            """)

        elif model_key in ("random_forest_regressor", "gradient_boosting_regressor"):
            st.markdown(r"""
            #### 1. Continuous Ensemble Prediction & Variance Reduction
            ##### Mathematical Representation:
            $$\hat{y}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x), \quad \text{Node Splitting Criterion: } \text{Var}(t) = \frac{1}{N_t} \sum_{i \in S_t} (y_i - \bar{y}_t)^2$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Compute Node Variance):** At node $t$, calculate the variance of the continuous outcome (e.g. inpatient length of stay) around the node mean $\bar{y}_t$.
            * **Step 2 (Evaluate Feature Splits):** Find the clinical feature split that maximizes the reduction in variance from parent to child nodes.
            * **Step 3 (Terminal Leaf Assignment):** In terminal leaves, assign the average value of training patients in that leaf.
            * **Step 4 (Ensemble Combination):** Average predictions across all $B$ trees to produce smooth continuous predictions with low variance.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Accurately captures non-linear thresholds in patient length of stay and hospital treatment costs without assuming normal error distributions.
            """)

        elif model_key == "holt_winters_exponential_smoothing":
            st.markdown(r"""
            #### 1. Triple Exponential Smoothing State Equations (Additive Formulation)
            ##### Mathematical Representation:
            $$\ell_t = \alpha (y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1})$$
            $$b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}$$
            $$s_t = \gamma (y_t - \ell_t) + (1 - \gamma) s_{t-m}$$
            $$\hat{y}_{t+h \mid t} = \ell_t + h b_t + s_{t+h - m(k+1)}$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Level $\ell_t$):** Update baseline attendance volume by combining today's deseasonalized attendance $(y_t - s_{t-m})$ weighted by $\alpha$, with yesterday's expected level $(\ell_{t-1} + b_{t-1})$ weighted by $(1 - \alpha)$.
            * **Step 2 (Trend $b_t$):** Update growth trajectory by combining today's change in baseline $(\ell_t - \ell_{t-1})$ weighted by $\beta$, with yesterday's slope $b_{t-1}$ weighted by $(1 - \beta)$.
            * **Step 3 (Seasonality $s_t$):** Update the 7-day cyclical pattern by combining today's seasonal residual $(y_t - \ell_t)$ weighted by $\gamma$, with the seasonal index from exactly one week ago ($s_{t-m}$) weighted by $(1 - \gamma)$.
            * **Step 4 (Forecast $\hat{y}_{t+h}$):** Project $h$ days into the future: current level $\ell_t$ $+$ $(h \times \text{trend } b_t)$ $+$ future day-of-week seasonal component.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Hospital emergency departments experience massive, predictable weekly rhythms (Monday surges, weekend dips). Holt-Winters decomposes and projects these dynamics in real time with zero training latency.
            """)

        elif model_key == "sarima_forecasting":
            st.markdown(r"""
            #### 1. Seasonal ARIMA Difference & Polynomial Formulation
            ##### Mathematical Representation:
            $$\Phi_P(B^s) \phi_p(B) (1 - B)^d (1 - B^s)^D y_t = \Theta_Q(B^s) \theta_q(B) \epsilon_t$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Consecutive Differencing $(1-B)^d$):** Subtract consecutive daily observations $(y_t - y_{t-1})$ to remove linear growth trends.
            * **Step 2 (Seasonal Differencing $(1-B^s)^D$):** Subtract attendance from the same day last week $(y_t - y_{t-7})$ to remove weekly seasonality.
            * **Step 3 (Autoregressive Polynomial $\phi_p(B)$):** Model serial dependencies on recent preceding days ($t-1, t-2$).
            * **Step 4 (Seasonal Autoregressive $\Phi_P(B^s)$):** Model correlation with previous weekly cycles ($t-7, t-14$).
            * **Step 5 (Moving Average $\theta_q(B), \Theta_Q(B^s)$):** Model persistence of sudden unexpected demand shocks ($\epsilon_t$).

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Box-Jenkins SARIMA provides formal statistical confidence bounds for annual hospital capacity contracts and winter surge bed allocation.
            """)

        elif model_key == "lagged_ensemble_forecasting":
            st.markdown(r"""
            #### 1. Autoregressive Tabular Lag Embedding Function
            ##### Mathematical Representation:
            $$\hat{y}_{t+1} = f\left( y_t, y_{t-1}, y_{t-7}, \bar{y}_{7d}, \sigma_{7d}, \text{CalendarFeatures}_t \right)$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Create Lag Features):** Convert the raw time series into a supervised tabular matrix by creating lag columns ($y_t, y_{t-1}, y_{t-7}$).
            * **Step 2 (Rolling Window Statistics):** Compute moving averages ($\bar{y}_{7d}$) and rolling standard deviations ($\sigma_{7d}$).
            * **Step 3 (Exogenous Calendar Indicators):** Append bank holiday flags, day-of-week dummy variables, and ambient weather indicators.
            * **Step 4 (Fit Tree Ensemble):** Train a gradient boosted decision tree to map feature vectors directly to next-period hospital attendance.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Allows integration of external non-linear covariates (e.g. freezing weather causing respiratory admissions, flu epidemic indices) that classical linear time series models cannot ingest.
            """)

        elif model_key == "kmeans_clustering":
            st.markdown(r"""
            #### 1. Within-Cluster Sum of Squares (WCSS / Inertia) Optimization
            ##### Mathematical Representation:
            $$\arg\min_S \sum_{k=1}^K \sum_{x \in S_k} \| x - \mu_k \|^2, \quad \text{where } \mu_k = \frac{1}{|S_k|} \sum_{x \in S_k} x$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Centroid Initialization):** Place $K$ initial centroid vectors $\mu_1, \dots, \mu_K$ in multidimensional feature space.
            * **Step 2 (Patient Assignment):** For every patient $x$, calculate Euclidean distance to all $K$ centroids and assign them to the closest cluster $S_k$.
            * **Step 3 (Centroid Recomputation):** Compute new centroid $\mu_k$ as the vector mean of all patients currently assigned to cluster $S_k$.
            * **Step 4 (Iterative Convergence):** Repeat Assignment and Recomputation until patient cluster memberships stabilize and total WCSS converges.

            ##### Why This Calculation is Being Used & Real-World Context:
            * Discovers novel clinical phenotypes (e.g. identifying distinct sub-phenotypes of sepsis or heart failure with preserved ejection fraction) from unlabelled electronic health records.

            ---

            #### 2. Silhouette Score ($s \in [-1, +1]$)
            ##### Mathematical Representation:
            $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad \text{Silhouette Score} = \frac{1}{N} \sum_{i=1}^N s(i)$$
            * **Intra-cluster cohesion $a(i)$:** Average distance between observation $i$ and all other members in the same cluster.
            * **Nearest-cluster separation $b(i)$:** Average distance from observation $i$ to all members in the nearest neighboring cluster.
            * **Interpretation:** Scores $> 0.50$ indicate well-demarcated cluster boundaries; near $0$ indicates heavy overlap; negative values indicate misassigned points.

            ---

            #### 3. Davies-Bouldin Index ($\text{DBI} \ge 0$)
            ##### Mathematical Representation:
            $$R_{i,j} = \frac{S_i + S_j}{M_{i,j}}, \quad \text{DBI} = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} R_{i,j}$$
            * **Cluster scatter $S_i$:** Mean distance of points in cluster $i$ to their centroid.
            * **Centroid distance $M_{i,j}$:** Euclidean distance between centroids of clusters $i$ and $j$.
            * **Interpretation:** Lower values indicate superior clustering (tight clusters that are far apart). The minimum of the DBI curve across $k$ indicates optimal $k$.

            ---

            #### 4. Calinski-Harabasz Index ($\text{CHI} \ge 0$ / Variance Ratio Criterion)
            ##### Mathematical Representation:
            $$\text{CHI} = \frac{\text{SSB} / (k - 1)}{\text{SSW} / (N - k)} = \frac{\sum_{q=1}^k n_q \|\mu_q - \mu\|^2 / (k - 1)}{\sum_{q=1}^k \sum_{x \in C_q} \|x - \mu_q\|^2 / (N - k)}$$
            * **Between-cluster dispersion ($\text{SSB}$):** Weighted sum of squared distances from cluster centroids $\mu_q$ to overall centroid $\mu$.
            * **Within-cluster dispersion ($\text{SSW}$):** Total Inertia / sum of squared distances to assigned centroids.
            * **Interpretation:** Higher is better. Peaks identify optimal cluster partitions maximizing between-group separation relative to internal variance (mirroring the ANOVA $F$-test).

            ---

            #### 5. Sample Scale & Stability Guidance for Unsupervised Learning
            * **No Null Hypothesis ($H_0$):** In unsupervised clustering, classical $\alpha/\beta$ hypothesis testing sample size formulas do not apply because there are no pre-specified groups to contrast.
            * **Heuristic Sample Rules:** Target at least $N \ge 20\text{–}50$ observations per expected cluster ($N \ge 20k$) and $N \ge 10 \times p$ observations per feature to guarantee centroid stability and avoid high-dimensional metric collapse.
            """)

        elif model_key == "dbscan_clustering":
            st.markdown(r"""
            #### 1. Density-Based Spatial Clustering & Neighborhood Reachability
            ##### Mathematical Representation:
            $$N_\epsilon(p) = \{ q \in D \mid \text{dist}(p, q) \le \epsilon \}, \quad \text{Core Point if } |N_\epsilon(p)| \ge \text{MinPts}$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Count Density Neighbors):** For patient $p$, count how many other patients lie within Euclidean radius $\epsilon$ ($N_\epsilon(p)$).
            * **Step 2 (Core Point Designation):** If the neighbor count meets or exceeds threshold $\text{MinPts}$, designate patient $p$ as a Core Point.
            * **Step 3 (Density-Reachable Chaining):** Chain mutually reachable core points and their boundary points together into arbitrary cluster shapes.
            * **Step 4 (Outlier Isolation):** Any point that is not density-reachable from any core point is classified strictly as Noise / Outlier (Cluster $-1$).

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Identifies geographic disease outbreak clusters and isolates anomalous atypical patient cases without imposing artificial spherical shapes.
            """)

        elif model_key == "pca_reduction":
            st.markdown(r"""
            #### 1. Sample Covariance Eigen-Decomposition & Explained Variance Ratio
            ##### Mathematical Representation:
            $$\Sigma = \frac{1}{n-1} X^T X, \quad \Sigma v_j = \lambda_j v_j, \quad \text{EVR}_j = \frac{\lambda_j}{\sum_{k=1}^p \lambda_k}$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Standardize Data Matrix):** Center each clinical biomarker to mean $0$ and unit variance $1$.
            * **Step 2 (Compute Covariance Matrix $\Sigma$):** Calculate the $p \times p$ sample covariance matrix measuring pairwise associations between all biomarkers.
            * **Step 3 (Solve Eigen-System):** Compute eigenvalues $\lambda_j$ and orthogonal eigenvectors $v_j$.
            * **Step 4 (Rank Components):** Sort eigenvectors in descending order of eigenvalue; $v_1$ points in the direction of maximum patient variance.
            * **Step 5 (Calculate Explained Variance):** Divide $\lambda_j$ by the sum of all eigenvalues to find the proportion of total clinical information captured by component $j$.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Compresses hundreds of correlated clinical parameters into 2 or 3 orthogonal summary axes for visual exploratory analysis and multicollinearity elimination.
            """)

        else:
            # Algorithmic Fairness Audit / Generic Fallback
            st.markdown(r"""
            #### 1. Disparate Impact Ratio (DIR) & Four-Fifths Standard
            ##### Mathematical Representation:
            $$\text{DIR} = \frac{\min_{a \in A} P(\hat{Y} = 1 \mid A = a)}{\max_{a \in A} P(\hat{Y} = 1 \mid A = a)}$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1 (Calculate Selection Rate):** For each socio-economic tier $a \in A$, compute the percentage of patients flagged positive by the model.
            * **Step 2 (Find Extremes):** Scan to locate the minimum group selection rate and maximum group selection rate.
            * **Step 3 (Divide Minimum by Maximum):** Compute the Disparate Impact Ratio.
            * **Step 4 (Evaluate Four-Fifths Standard):** If $\text{DIR} \ge 0.80$, demographic parity is met; if $< 0.80$, adverse disparate impact exists.

            ##### Why This Calculation is Being Used & Real-World NHS Context:
            * Mandated by NHS AI ethics standards to ensure automated clinical and scheduling algorithms do not disadvantage vulnerable demographic groups.

            ---

            #### 2. False Positive Rate Parity & Equalized Odds
            ##### Mathematical Representation:
            $$\text{FPR}_a = \frac{\text{FP}_a}{\text{FP}_a + \text{TN}_a}, \quad \Delta_{\text{EO}} = |\text{TPR}_1 - \text{TPR}_2| + |\text{FPR}_1 - \text{FPR}_2|$$

            ##### Plain English Step-by-Step Breakdown of the Formula:
            * **Step 1:** Among attenders ($Y=0$), compute the proportion incorrectly flagged as DNA risk ($\text{FP}_a$).
            * **Step 2:** Compare FPR across tiers to audit overbooking penalties.
            * **Step 3:** Sum absolute differences in TPR and FPR to measure total Equalized Odds divergence.

            ##### Why This Calculation is Being Used:
            * Guarantees that algorithm prediction errors are distributed equitably across patient populations.
            """)


def render_ml_educational_tabs(
    scenario_key: str,
    meta: Dict[str, Any],
    params: Dict[str, Any],
    calc_results: Dict[str, Any]
):
    """Renders the comprehensive 4-tab pedagogical and technical architecture."""
    st.subheader("5. Methodological Architecture & Technical Specifications")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Rationale & Biostatistical Derivations",
        "🏥 Clinical & Operational Implementation Guide",
        "🐍 Executable Python Pipeline (scikit-learn / statsmodels)",
        "⚠️ Limitations, Bias Pitfalls & NHS AI Guidelines"
    ])

    model_key = params.get("model_key", meta.get("recommended_model", "random_forest_classifier"))
    model_info = ML_MODELS.get(model_key, ML_MODELS["random_forest_classifier"])

    # -------------------------------------------------------------
    # TAB 1: Rationale & Formulations
    # -------------------------------------------------------------
    with tab1:
        model_name = model_info.get("name", "Machine Learning Model")
        desc = model_info.get("short_desc", model_info.get("description", "Predictive machine learning algorithm."))
        st.markdown(f"### {model_name}: Mathematical & Epidemiological Framework")
        st.markdown(f"**Methodological Description:** {desc}")
        if "primary_use_case" in model_info:
            st.markdown(f"**Primary Healthcare Use Case:** {model_info['primary_use_case']}")

        # Comprehensive Mathematical Formulations with Plain English Step-by-Step Breakdown
        render_ml_model_mathematical_framework(model_key, model_info, meta, params)

        with st.expander("🌐 Cross-Industry Applications: Manufacturing, Marketing, Subscriptions & Supply Chain", expanded=False):
            st.markdown(r"""
            #### How this Architecture Operates Across Commercial & Industrial Sectors
            Machine learning algorithms share identical mathematical loss optimization whether analyzing hospital patient registries or enterprise industrial telemetry:

            * **🏭 Industry 4.0 & Advanced Manufacturing (Apex Precision Robotics):**
              - **Application:** Predictive condition monitoring of high-speed CNC spindle bearings and robotic joints using multi-sensor vibration, acoustic emission, and temperature telemetry.
              - **Architecture Implementation:** Handles extreme class imbalance (rare failures, 4.2% rate) via balanced tree ensembles or positive-class weighting. Captures non-linear resonance interactions before catastrophic factory downtime occurs (£45,000/hr).

            * **📢 Omnichannel Marketing & Retail (OmniChannel Retail Group):**
              - **Application:** Customer purchase conversion propensity scoring and algorithmic equity auditing across demographic cohorts.
              - **Architecture Implementation:** Ranks browsing session trajectories (dwell time, cart abandonment, past spend) to allocate personalized promotional discount incentives while enforcing the **Four-Fifths Rule (DIR $\ge 0.80$)** to eliminate algorithmic redlining.

            * **📱 Digital Subscriptions & SaaS (PulseStream Digital):**
              - **Application:** Subscriber cancellation / churn prevention and lifetime value optimization.
              - **Architecture Implementation:** Ingests product telemetry (streaming hours, buffering glitches, login recency, support tickets) 30 days prior to billing renewal. Calibrated decision thresholds ($p^* = 0.26$) optimize retention ROI by balancing concierge outreach cost against recurring revenue.

            * **🚛 Supply Chain, Freight & Logistics (MetroLogix Distribution):**
              - **Application:** Regional distribution center pallet dispatch and freight truck fleet capacity forecasting.
              - **Architecture Implementation:** Models rigid 7-day operational cycles and holiday surge dynamics using rolling-origin temporal evaluation, preventing loading bay bottlenecks and carrier demurrage penalties.
            """)

        with st.expander("💡 Pedagogical Biostatistical Takeaways", expanded=False):
            st.markdown(meta.get("takeaway", "Machine learning models must balance predictive power with clinical interpretability."))

    # -------------------------------------------------------------
    # TAB 2: Clinical & Operational Implementation Guide
    # -------------------------------------------------------------
    with tab2:
        org_label = meta.get('organisation_name', meta.get('fictional_trust', 'Enterprise Deployment'))
        st.markdown(f"### {org_label}: Operational Deployment Workflow")

        with st.expander("🏥 Step 1: Clinical Cohort Ingestion & Data Cleansing", expanded=False):
            st.markdown(r"""
            1. **Electronic Health Record (EHR) Linkage:** Query the secondary care data warehouse (Epic, Cerner, or EMIS) for patient discharge records.
            2. **Observation Window Definition:** For 30-day readmission models, index discharge date is $t_0$. Exclude planned elective admissions and in-hospital mortalities.
            3. **Handling Missing Clinical Biomarkers:** 
               - If missingness $< 5\%$, apply median/modal imputation within cross-validation folds.
               - If missingness $\ge 5\%$ (e.g. BNP or ejection fraction missing when not clinically indicated), apply Multiple Imputation by Chained Equations (MICE) or use models natively handling missing values (Histogram Gradient Boosting).
            """)

        with st.expander("🛡️ Step 2: Leakage-Free Preprocessing Architecture", expanded=False):
            st.markdown(r"""
            1. **Never Normalize Prior to Splitting:** Continuous features (e.g. Serum Creatinine, Age) must be scaled inside a `ColumnTransformer` fitted strictly on training folds.
            2. **Categorical Encoding:** One-hot encode clinical categorical features (`drop='first'`, `handle_unknown='ignore'`) inside the cross-validation pipeline to prevent vocabulary leakage.
            3. **Temporal Partitioning for Forecasting:** In time series demand forecasting, strictly utilize expanding-window `TimeSeriesSplit` rather than randomized K-Fold splits.
            """)

        with st.expander("🎯 Step 3: Threshold Calibration & Operational Governance", expanded=False):
            st.markdown(r"""
            1. **Brier Score Reliability Curves:** Verify that a predicted probability of 30% corresponds to an observed 30% event rate in calibration plots.
            2. **Decision Curve Analysis (DCA):** Compare model-guided intervention net benefit against "intervene on everyone" and "intervene on no one" policies.
            3. **Fairness Threshold Recalibration:** If the Four-Fifths rule is violated across IMD deprivation deciles, calculate group-specific decision thresholds ($p^*_a$) to equalize False Positive Rates before operational scheduling deployment.
            """)

        with st.expander("🏭 Cross-Industry Deployment: Industrial IoT, E-Commerce, SaaS & Logistics Pipelines", expanded=False):
            st.markdown(r"""
            1. **Manufacturing Edge Ingestion (Apex Precision Robotics):**
               - High-frequency sensor streams (10 kHz vibration, acoustic noise) are ingested via MQTT/Kafka into edge gateways.
               - Fast Fourier Transform (FFT) feature extractors compute spectral energy and root-mean-square (RMS) metrics for low-latency (<5ms) edge model inference on PLC controllers.
            2. **Omnichannel E-Commerce Scoring (OmniChannel Retail Group):**
               - User session clicks and dwell times stream through a low-latency Redis feature store.
               - Real-time propensity inference scores shoppers as they add items to their cart, dynamically rendering targeted promotions while auditing group selection rates hourly.
            3. **SaaS Retention Webhooks (PulseStream Digital):**
               - Automated nightly batch inference scans all active subscribers within 30 days of subscription renewal.
               - Accounts exceeding risk threshold $p^* = 0.26$ trigger automated webhook dispatches to customer success platforms (Salesforce/HubSpot) for proactive concierge outreach.
            4. **Supply Chain ERP Integration (MetroLogix Distribution):**
               - Weekly rolling-origin demand forecasts feed directly into enterprise ERP (SAP/Oracle SCM) via automated REST APIs.
               - Warehouse capacity managers dynamically allocate third-party freight truck contracts and loading bay shifts 14 days in advance.
            """)

    # -------------------------------------------------------------
    # TAB 3: Executable Python Pipeline
    # -------------------------------------------------------------
    with tab3:
        st.markdown("### Production-Grade Python Pipeline (scikit-learn / statsmodels)")
        st.markdown("Copy and run this reproducible, leak-free pipeline directly in your analytical environment:")

        pipeline_code = generate_ml_python_pipeline(
            model_key=model_key,
            scenario_meta=meta
        )

        st.code(pipeline_code, language="python")

        st.download_button(
            label="💾 Download Pipeline Script (.py)",
            data=pipeline_code,
            file_name=f"ml_pipeline_{scenario_key}_{model_key}.py",
            mime="text/x-python",
            help="Download the complete executable Python script for your analytics data mart."
        )

    # -------------------------------------------------------------
    # TAB 4: Limitations, Bias Pitfalls & NHS AI Guidelines
    # -------------------------------------------------------------
    with tab4:
        st.markdown("### Clinical AI Governance, Ethical Pitfalls & Cross-Industry Guidelines")

        with st.expander("⚖️ NHS AI Ethics & Health Equity Pitfalls", expanded=False):
            st.markdown(r"""
            * **Automating Socio-Economic Disparities:** If high-deprivation patients face systemic barriers attending appointments (lack of transport, inflexible jobs), training an algorithm on historical attendance records will flag low-income patients as high DNA risks. If the clinic overbooks these slots, the algorithm actively punishes the vulnerable cohort, widening health inequalities.
            * **Proactive vs. Punitive Interventions:** Operational algorithms should be used to provide **supportive interventions** (subsidized bus passes, proactive nurse telephone check-ins) rather than punitive double-booking.
            * **NICE Evidence Standards Framework (ESF):** Tier C digital health technologies utilizing machine learning must provide audited evidence of algorithmic fairness across protected characteristics (Equality Act 2010).
            """)

        with st.expander("⚠️ Machine Learning Methodological Pitfalls", expanded=False):
            st.markdown(r"""
            * **The Accuracy Trap:** Relying on accuracy in imbalanced clinical registries hides complete model failure. Always report PR-AUC, F1-macro, and calibration curves.
            * **Overfitting Small Clinical Samples:** When Events Per Variable (EPV) $< 10$, complex machine learning models overfit noise. Regularized regression (ElasticNet) or Firth's penalized logistic regression is mandatory.
            * **Concept Drift in Healthcare:** Clinical protocols, patient demographics, and seasonal pathogens evolve. Models deployed in operational environments require continuous monitoring of discrimination degradation over time.
            """)

        with st.expander("🌐 Cross-Industry Governance & Failure Modes (Manufacturing, Marketing, Subscriptions & Logistics)", expanded=False):
            st.markdown(r"""
            * **Industrial Sensor Drift (Apex Precision Robotics):** Physical sensors accumulate dust, calibration drift, and thermal bias over time. Maintenance models require periodic zero-point recalibration and drift detection tests (Kolmogorov-Smirnov feature drift tests) to prevent spurious maintenance shutdowns.
            * **Commercial Algorithmic Redlining (OmniChannel Retail Group):** Propensity algorithms trained naively on historical spending patterns exacerbate socio-economic divides by systematically denying promotional financing to lower-income zip codes. Regular disparate impact audits are mandatory under consumer protection guidelines.
            * **Retention Unit Economics Trap (PulseStream Digital):** Intervening on every predicted churner erodes profit margins if the retention incentive cost exceeds the customer's discounted residual lifetime value. Thresholds must be chosen by maximizing net monetary benefit: $\text{NMB} = (\text{TP} \times \text{CLV}_{\text{retained}}) - ((\text{TP} + \text{FP}) \times \text{Cost}_{\text{outreach}})$.
            * **The Bullwhip Effect in Supply Chains (MetroLogix Distribution):** Small fluctuations in end-consumer retail demand amplify into violent swings up the wholesale supply chain. Using unregularized multi-step autoregressive models can cause wild over-ordering; seasonal exponential smoothing or constrained SARIMA dampens spurious oscillation.
            """)


# =====================================================================
# 7. Interactive Decision Wizard (Funnel)
# =====================================================================

def render_ml_decision_wizard():
    """Renders an interactive decision funnel helping analysts choose the right ML architecture."""
    with st.expander("🧭 Interactive Machine Learning Decision Wizard: Find Your Architecture", expanded=False):
        st.markdown("Answer 4 high-level questions about your clinical outcome and operational constraints to receive a tailored biostatistical recommendation:")

        c1, c2 = st.columns(2)
        with c1:
            q_task = st.selectbox(
                "1. What is the nature of your target outcome?",
                options=["binary_event", "continuous_count", "time_series", "unlabelled_cohort", "equity_audit"],
                format_func=lambda x: {
                    "binary_event": "🎯 Binary Clinical Event (e.g. Readmission, Sepsis, DNA: 0 or 1)",
                    "continuous_count": "📈 Continuous Quantity (e.g. Length of Stay, Healthcare Cost)",
                    "time_series": "⏱️ Sequential Temporal Demand (e.g. Daily Attendances, Weekly Bed Surge)",
                    "unlabelled_cohort": "🔍 Unlabelled Cohort (e.g. Discovering Patient Phenotypes)",
                    "equity_audit": "⚖️ Algorithmic Equity Audit (e.g. Auditing Disparate Impact across IMD Deciles)"
                }[x]
            )

            q_interpret = st.selectbox(
                "2. Is publication-ready linear interpretability (Odds Ratios / Slopes) required?",
                options=["high", "balanced", "max_accuracy"],
                format_func=lambda x: {
                    "high": "High: Must publish exact Odds Ratios / linear coefficients (Governance priority)",
                    "balanced": "Balanced: Balance predictive accuracy with feature importance explanations",
                    "max_accuracy": "Maximum Predictive Power: Prioritize top-tier discrimination (e.g. gradient boosting)"
                }[x]
            )

        with c2:
            q_imbalance = st.selectbox(
                "3. What is the target event incidence rate?",
                options=["balanced", "moderate", "severe"],
                format_func=lambda x: {
                    "balanced": "Balanced (>= 20% event incidence)",
                    "moderate": "Moderately Imbalanced (10% to 20% incidence)",
                    "severe": "Severely Imbalanced (< 10% incidence - e.g. Heart Failure Readmission)"
                }[x]
            )

            q_temporal = st.checkbox("Sequential time-dependent data (Must prevent future-to-past leakage)", value=(q_task == "time_series"))

        # Map funnel choices to recommendation
        task_map = {
            "binary_event": "classification",
            "continuous_count": "regression",
            "time_series": "time_series",
            "unlabelled_cohort": "clustering",
            "equity_audit": "fairness_audit"
        }

        rec = route_ml_recommendation(
            task=task_map[q_task],
            interpretability_priority=q_interpret,
            imbalance_severity=q_imbalance,
            has_temporal_structure=q_temporal,
            is_equity_audit=(q_task == "equity_audit")
        )

        p_mod = rec["primary_model"]
        c_mod = rec["comparator_model"]

        st.markdown(f"""
        <div style="background-color: #f0fdf4; border: 1px solid #16a34a40; border-radius: 6px; padding: 0.8rem 1rem; margin-top: 0.8rem;">
            <div style="font-weight: 700; color: #166534; font-size: 1rem;">
                🎯 Recommended Primary Model: {p_mod['name']}
            </div>
            <div style="font-size: 0.88rem; color: #374151; margin-top: 0.35rem;">
                <strong>Biostatistical Rationale:</strong> {rec['rationale']}
            </div>
            <div style="font-size: 0.84rem; color: #475569; margin-top: 0.35rem;">
                <strong>Secondary Benchmark Comparator:</strong> {c_mod['name']}
            </div>
            <div style="font-size: 0.80rem; color: #64748b; margin-top: 0.35rem;">
                <strong>Recommended Evaluation Metric:</strong> {p_mod.get('primary_metric', 'PR-AUC / F1-Score')}
            </div>
        </div>
        """, unsafe_allow_html=True)


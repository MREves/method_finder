"""Interactive UI Components for the Unified Study Design & Method Selector.

Provides:
1. Zero-Knowledge Pedagogical Guide to Study Design & Quantitative Paradigms.
2. 1-Click Real-World Problem Blueprints across Healthcare and Industry.
3. Interactive Problem Specification Funnel (Sector, Goal, Endpoints, Constraints).
4. Top Recommendation Cards with Biostatistical Rationale & Alternative Fallbacks.
5. One-Click Deep-Linking Navigation Buttons into Modules 1, 2, 3, and 4.
6. 3-Pillar Pedagogical Comparison Matrix (Statistics vs. Epidemiology vs. Machine Learning).
7. 5-Point Methodological Study Protocol Verification Checklist.
"""

import html
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st

from src.selector.study_design_schema import (
    DOMAIN_PROFILES,
    PROBLEM_DEFINITIONS,
    PROBLEM_BLUEPRINTS
)
from src.selector.selector_engine import (
    get_problem_blueprint,
    route_universal_method_recommendation
)


def render_html(html_content: str) -> None:
    """Renders HTML cleanly without CommonMark indentation code-block interference."""
    cleaned = "\n".join(line.strip() for line in html_content.splitlines() if line.strip())
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


# =====================================================================
# 1. Plain English Pedagogical Introductory Guide
# =====================================================================

def render_selector_plain_english_guide():
    """Renders a collapsible plain English guide explaining how to choose between the 3 quantitative pillars."""
    with st.expander("📖 Zero-Knowledge Guide: How to Choose Between Statistics, Epidemiology & Machine Learning", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(r"""
            #### 1. The Core Methodological Paradigms
            Before picking a formula or writing code, identify the fundamental nature of your scientific or operational question:

            * **📊 1. Classical Statistical Hypothesis Testing (Pillar 1):**
              - **The Question:** *"Did an intervention or exposure cause a statistically significant difference, or could this result have arisen by random chance?"*
              - **Focus:** $p$-values, confidence intervals, unbiased parameter estimation ($\mu_1 - \mu_2$), and controlling Type I False Positive error ($\alpha = 0.05$).
              - **Best For:** Randomized clinical trials (RCTs), parallel treatment comparisons, pre-post laboratory assays, and confirmatory regulatory submissions.

            * **🏥 2. Epidemiological & Public Health Methods (Pillar 2):**
              - **The Question:** *"What is the strength of association between an exposure and disease risk in an observational population, and how many cases can be prevented?"*
              - **Focus:** Relative Risk (RR), Odds Ratios (OR), Hazard Ratios (HR), Population Attributable Fractions (PAF), Direct Age-Standardization (DSR), and Diagnostic Screening (Sensitivity, Specificity, Likelihood Ratios).
              - **Best For:** Cohort studies, retrospective case-control investigations, outbreak tracking, geographic health variation, and clinical diagnostic validation.
            """)

        with c2:
            st.markdown(r"""
            * **🤖 3. Machine Learning & Operational Analytics (Pillar 3):**
              - **The Question:** *"Given an individual's multi-dimensional features right now, what is the exact probability of an event occurring in the future, or what will tomorrow's demand be?"*
              - **Focus:** Out-of-sample generalization, discrimination (PR-AUC / ROC-AUC), probabilistic calibration (Brier score), time series demand forecasting (MAE/RMSE), and Algorithmic Fairness (Four-Fifths Rule).
              - **Best For:** Predicting rare clinical readmissions, industrial equipment failure, subscriber churn, daily emergency department surges, and customer propensity.

            #### 2. The Indispensable Role of Sample Size Planning (Pillar 4)
            * **The Golden Rule:** Regardless of which method you select, **sample size planning must precede data collection**.
            * Recruiting too few participants causes an underpowered study that misses true effects ($p > 0.05$). Recruiting too many participants needlessly wastes resources and exposes volunteers to inferior treatments.
            * This selector automatically pairs every recommended method with its matching sample size calculation framework, complete with attrition and cluster randomization cushions.
            """)


# =====================================================================
# 2. Real-World Problem Blueprints Bar
# =====================================================================

def render_problem_blueprints_bar() -> Optional[str]:
    """Renders 1-click real-world problem blueprint buttons across healthcare and industry in a 2-column by 4-row layout."""
    st.markdown("#### ⚡ Quick-Start: Pre-Configured Real-World Problem Blueprints")
    st.caption("Click any blueprint below to instantly populate the problem framework, inspect the end-to-end analytical roadmap, and launch interactive modules:")

    blueprints = list(PROBLEM_BLUEPRINTS.values())
    selected_bp = None

    # Render in 4 rows of 2 columns so button labels have ample width and full text visibility
    for row_idx in range(0, len(blueprints), 2):
        row_blueprints = blueprints[row_idx:row_idx + 2]
        cols = st.columns(2)
        for col_idx, bp in enumerate(row_blueprints):
            with cols[col_idx]:
                button_label = f"{bp['icon']} {bp['title']}"
                if st.button(
                    button_label,
                    key=f"btn_bp_{bp['id']}",
                    use_container_width=True,
                    help=f"{bp['organisation']} — {bp['description']}"
                ):
                    st.session_state["active_blueprint_key"] = bp["id"]
                    selected_bp = bp["id"]
                    st.rerun()

    return selected_bp


# =====================================================================
# 3. Interactive Problem Specification Funnel
# =====================================================================

def render_problem_definition_form(active_blueprint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Renders the step-by-step problem definition form allowing users to configure custom study requirements."""
    st.markdown("---")
    st.subheader("🎯 Step 1: Define Your Research Problem & Operational Context")
    st.markdown("Specify your domain, research objective, endpoint structure, and study constraints below:")

    bp = active_blueprint if active_blueprint is not None else {}

    domain_keys = list(DOMAIN_PROFILES.keys())

    goal_options = {
        "compare_groups": "⚖️ Compare 2 or 3+ Groups (Means / Continuous Outcomes)",
        "paired_change": "🔄 Measure Within-Subject Change (Pre- vs. Post-Intervention)",
        "correlation_association": "📈 Quantify Association / Correlation Between 2 Measures",
        "contingency_rates": "🗂️ Compare Categorical Frequencies / Proportions (2x2 Tables)",
        "cohort_prospective": "⏱️ Follow an Exposed vs. Unexposed Cohort Forward (Relative Risk / IRR)",
        "case_control_retrospective": "🔍 Sample Diseased Cases vs. Healthy Controls (Odds Ratio in Rare Disease)",
        "diagnostic_accuracy": "🎯 Evaluate a Diagnostic Test, Biomarker or Screening Device",
        "survival_time_to_event": "⏳ Follow Longitudinal Cohorts with Right-Censoring (Time-to-Event)",
        "demand_forecasting": "⏱️ Forecast Sequential Daily/Weekly Operational Demand (Time Series)",
        "predict_rare_event": "🔮 Predict Individual Binary High-Risk Events (Rare Clinical Event / Failure / Churn)",
        "equity_fairness": "⚖️ Audit Algorithmic Disparate Impact & Fairness (Four-Fifths 80% Rule)",
        "clustering_phenotyping": "🔍 Discover Unlabelled Patient Phenotypes or Customer Clusters",
        "dimension_reduction": "📉 Compress High-Dimensional Multicollinear Predictors (PCA)"
    }

    prob_to_goal = {
        "two_group_comparison": "compare_groups",
        "multi_group_comparison": "compare_groups",
        "paired_pre_post": "paired_change",
        "linear_correlation": "correlation_association",
        "categorical_contingency": "contingency_rates",
        "prospective_cohort_study": "cohort_prospective",
        "retrospective_case_control": "case_control_retrospective",
        "diagnostic_screening_evaluation": "diagnostic_accuracy",
        "survival_time_to_event": "survival_time_to_event",
        "time_series_forecasting": "demand_forecasting",
        "rare_event_classification": "predict_rare_event",
        "algorithmic_fairness_audit": "equity_fairness",
        "unsupervised_phenotyping": "clustering_phenotyping",
        "dimension_reduction_pca": "dimension_reduction",
    }

    design_options = {
        "parallel_rct": "Double-Blind Parallel Randomized Controlled Trial (RCT)",
        "paired_pre_post": "Single-Cohort Pre-Post Interventional Design",
        "multi_arm": "Multi-Arm Parallel Factorial Trial (3+ Groups)",
        "observational_cohort": "Prospective Longitudinal Observational Cohort",
        "case_control": "Retrospective Case-Control Design (1:k Matching)",
        "diagnostic_accuracy": "Cross-Sectional Diagnostic Accuracy Study against Gold Standard",
        "longitudinal_series": "Sequential Operational Time Series (Chronological)",
        "cross_sectional": "Cross-Sectional Observational Survey",
        "sensor_telemetry": "Industrial IoT Condition Monitoring Sensor Telemetry",
        "web_telemetry": "Omnichannel Digital Web Session & Propensity Telemetry",
        "product_telemetry": "Digital Product Usage & Subscription Telemetry"
    }

    sample_scale_options = {
        "small": "Small Cohort (N < 30 participants / records)",
        "medium": "Moderate Sample (30 <= N < 500 participants / records)",
        "large": "Large Registry or Streaming Telemetry (N >= 500 records)"
    }

    # Check if a new blueprint was loaded, reset, or existing session selections should be preserved
    current_bp_key = bp.get("id") if bp else None
    last_bp_key = None
    if hasattr(st, "session_state") and hasattr(st.session_state, "get"):
        val = st.session_state.get("last_loaded_blueprint_key")
        if isinstance(val, str):
            last_bp_key = val

    if current_bp_key is not None and current_bp_key != last_bp_key:
        st.session_state["saved_selector_domain"] = bp.get("domain", domain_keys[0])
        st.session_state["saved_selector_goal"] = prob_to_goal.get(bp.get("problem_key", ""), "compare_groups")
        st.session_state["saved_selector_design"] = bp.get("study_design", "parallel_rct")
        st.session_state["saved_selector_scale"] = bp.get("sample_scale", "medium")
        st.session_state["last_loaded_blueprint_key"] = current_bp_key
    elif last_bp_key == "__RESET__":
        st.session_state["saved_selector_domain"] = domain_keys[0]
        st.session_state["saved_selector_goal"] = "compare_groups"
        st.session_state["saved_selector_design"] = "parallel_rct"
        st.session_state["saved_selector_scale"] = "medium"
        st.session_state["last_loaded_blueprint_key"] = None

    c1, c2 = st.columns(2)

    with c1:
        saved_domain = st.session_state.get("saved_selector_domain") if hasattr(st, "session_state") else None
        default_domain_idx = domain_keys.index(saved_domain) if (isinstance(saved_domain, str) and saved_domain in domain_keys) else 0

        selected_domain = st.selectbox(
            "1. Real-World Domain / Analytical Sector:",
            options=domain_keys,
            format_func=lambda k: DOMAIN_PROFILES[k]["name"],
            index=default_domain_idx,
            help="Select the operational or scientific environment for your project."
        )
        st.session_state["saved_selector_domain"] = selected_domain

        goal_keys = list(goal_options.keys())
        saved_goal = st.session_state.get("saved_selector_goal") if hasattr(st, "session_state") else None
        default_goal_idx = goal_keys.index(saved_goal) if (isinstance(saved_goal, str) and saved_goal in goal_keys) else 0

        selected_goal = st.selectbox(
            "2. Primary Research Objective / Problem Statement:",
            options=goal_keys,
            format_func=lambda k: goal_options[k],
            index=default_goal_idx,
            help="Select the fundamental scientific or operational question you are solving."
        )
        st.session_state["saved_selector_goal"] = selected_goal

    with c2:
        study_design_help = (
            "Plain-English Guide to Study Designs & Data Architectures:\n\n"
            "• Sequential Operational Time Series (Chronological):\n"
            "  Historical events or measurements recorded step-by-step over regular time intervals (e.g. daily, weekly, monthly). "
            "Choose this if you have past data (such as a previous year of hospital attendances, sales transactions, or utility demand) "
            "that you wish to use for a time series forecasting model to predict future events.\n\n"
            "• Double-Blind Parallel Randomized Controlled Trial (RCT):\n"
            "  Two separate groups created by random lottery (e.g. Treatment vs. Control/Placebo). Each participant receives only one treatment. "
            "Choose this when testing a new medicine or intervention under controlled experimental conditions.\n\n"
            "• Single-Cohort Pre-Post Interventional Design:\n"
            "  One single group of participants measured twice: once before an intervention and once after. "
            "Choose this when measuring within-person change (e.g. pain scores before and after physiotherapy, or staff stress before and after training).\n\n"
            "• Multi-Arm Parallel Factorial Trial (3+ Groups):\n"
            "  Three or more independent groups tested simultaneously. "
            "Choose this when comparing multiple treatment doses or options against a control (e.g. Low Dose vs. High Dose vs. Placebo).\n\n"
            "• Prospective Longitudinal Observational Cohort:\n"
            "  Following a group of individuals forward through time without assigning treatments, observing who naturally develops an outcome. "
            "Choose this when tracking exposed vs. unexposed groups over months or years to compute Relative Risk (RR).\n\n"
            "• Retrospective Case-Control Design (1:k Matching):\n"
            "  Looking backwards in time. Starts with people who already have the outcome/disease (Cases) and matches them to healthy people (Controls) to check past exposures. "
            "Choose this for rare conditions or outbreak vehicle tracing (e.g. comparing 30 food poisoning patients with 60 healthy diners) to compute Odds Ratios (OR).\n\n"
            "• Cross-Sectional Diagnostic Accuracy Study against Gold Standard:\n"
            "  Testing every individual with both a new rapid test and a reference 'gold standard' test at the same point in time. "
            "Choose this when evaluating sensitivity, specificity, or ROC curves for a diagnostic swab, imaging tool, or screening test.\n\n"
            "• Cross-Sectional Observational Survey:\n"
            "  A single point-in-time 'snapshot' measuring exposures and outcomes concurrently. "
            "Choose this when analyzing one-off survey questionnaires or census records (e.g. surveying 500 workers on daily screen time and headache frequency).\n\n"
            "• Industrial IoT Condition Monitoring Sensor Telemetry:\n"
            "  Automated, high-frequency readings recorded continuously by physical machine sensors (vibration, heat, pressure, electrical draw). "
            "Choose this when training predictive maintenance models to anticipate machine wear or breakdown before failure occurs.\n\n"
            "• Omnichannel Digital Web Session & Propensity Telemetry:\n"
            "  Online clickstream events, page visits, session duration, and marketing interactions logged on websites or digital channels. "
            "Choose this when predicting customer conversion probability or auditing promotional algorithms for demographic equity.\n\n"
            "• Digital Product Usage & Subscription Telemetry:\n"
            "  Software activity metrics recorded within a web/mobile application (login frequency, feature usage, active user seats). "
            "Choose this when training predictive models to detect user disengagement and prevent subscriber churn before renewal."
        )

        design_keys = list(design_options.keys())
        saved_design = st.session_state.get("saved_selector_design") if hasattr(st, "session_state") else None
        default_design_idx = design_keys.index(saved_design) if (isinstance(saved_design, str) and saved_design in design_keys) else 0

        selected_design = st.selectbox(
            "3. Study Design / Data Collection Architecture:",
            options=design_keys,
            format_func=lambda k: design_options[k],
            index=default_design_idx,
            help=study_design_help
        )
        st.session_state["saved_selector_design"] = selected_design

        # Plain-English dynamic caption reflecting current selection
        design_summaries = {
            "parallel_rct": "👥 **Parallel RCT:** Two independent groups created by random lottery (e.g. Treatment vs. Control). Each participant is in one group only.",
            "paired_pre_post": "🔄 **Pre-Post:** One single group measured before and after an intervention (each participant is their own baseline comparison).",
            "multi_arm": "📊 **Multi-Arm Trial:** Three or more separate groups compared simultaneously (e.g. Low Dose vs. High Dose vs. Placebo).",
            "observational_cohort": "⏱️ **Prospective Cohort:** Tracking participants forward in time from exposure to outcome without assigning treatments (computes Relative Risk).",
            "case_control": "🔍 **Retrospective Case-Control:** Looking backwards; starts with diseased cases and matches to healthy controls to investigate past exposures (computes Odds Ratios).",
            "diagnostic_accuracy": "🎯 **Diagnostic Accuracy:** Evaluating a new rapid test or screening biomarker against a definitive 'gold standard' reference test.",
            "longitudinal_series": "📈 **Sequential Time Series:** Chronological historical records (e.g. past year of daily/weekly events) used to train forecasting models for future demand.",
            "cross_sectional": "📸 **Cross-Sectional Survey:** A single point-in-time snapshot measuring exposure and outcome together (e.g. a one-off questionnaire).",
            "sensor_telemetry": "🏭 **Sensor Telemetry:** Automated high-frequency machine streams (vibration, heat) to detect equipment degradation or predict breakdown.",
            "web_telemetry": "🌐 **Web Telemetry:** Online clickstreams, browsing patterns, and session activity used to model user propensity or audit fairness.",
            "product_telemetry": "📱 **Product Telemetry:** In-app SaaS usage logs (logins, feature engagement) to predict customer churn before renewal."
        }
        if selected_design in design_summaries:
            st.caption(design_summaries[selected_design])

        scale_keys = list(sample_scale_options.keys())
        saved_scale = st.session_state.get("saved_selector_scale") if hasattr(st, "session_state") else None
        default_scale_idx = scale_keys.index(saved_scale) if (isinstance(saved_scale, str) and saved_scale in scale_keys) else 1

        selected_scale = st.selectbox(
            "4. Sample Size Scale / Cohort Magnitude:",
            options=scale_keys,
            format_func=lambda k: sample_scale_options[k],
            index=default_scale_idx,
            help="Indicates whether asymptotic normal approximations apply or non-parametric / exact tests are required."
        )
        st.session_state["saved_selector_scale"] = selected_scale

    # Map inputs to internal problem key
    inferred_problem_key = None
    if selected_goal == "compare_groups":
        if selected_design == "paired_pre_post":
            inferred_problem_key = "paired_pre_post"
        elif selected_design == "multi_arm":
            inferred_problem_key = "multi_group_comparison"
        else:
            inferred_problem_key = "two_group_comparison"
    elif selected_goal == "paired_change":
        inferred_problem_key = "paired_pre_post"
    elif selected_goal == "correlation_association":
        inferred_problem_key = "linear_correlation"
    elif selected_goal == "contingency_rates":
        if selected_design == "case_control":
            inferred_problem_key = "retrospective_case_control"
        elif selected_design in ("cohort", "observational_cohort", "prospective", "parallel_rct"):
            inferred_problem_key = "prospective_cohort_study"
        else:
            inferred_problem_key = "categorical_contingency"
    elif selected_goal == "cohort_prospective":
        if selected_design == "case_control":
            inferred_problem_key = "retrospective_case_control"
        else:
            inferred_problem_key = "prospective_cohort_study"
    elif selected_goal == "case_control_retrospective":
        inferred_problem_key = "retrospective_case_control"
    elif selected_goal == "diagnostic_accuracy":
        inferred_problem_key = "diagnostic_screening_evaluation"
    elif selected_goal == "survival_time_to_event":
        inferred_problem_key = "survival_time_to_event"
    elif selected_goal == "demand_forecasting":
        inferred_problem_key = "time_series_forecasting"
    elif selected_goal == "predict_rare_event":
        inferred_problem_key = "rare_event_classification"
    elif selected_goal == "equity_fairness":
        inferred_problem_key = "algorithmic_fairness_audit"
    elif selected_goal == "clustering_phenotyping":
        inferred_problem_key = "unsupervised_phenotyping"
    elif selected_goal == "dimension_reduction":
        inferred_problem_key = "dimension_reduction_pca"

    return {
        "domain": selected_domain,
        "analytical_goal": selected_goal,
        "study_design": selected_design,
        "sample_scale": selected_scale,
        "problem_key": inferred_problem_key
    }


# =====================================================================
# 4. Top Recommendation Cards & Deep-Link Launcher
# =====================================================================

def render_recommendation_card(rec: Dict[str, Any]):
    """Renders the top recommendation card with biostatistical rationale and one-click deep link buttons."""
    st.markdown("---")
    st.subheader("🎯 Step 2: Biostatistical Recommendation & Analytical Roadmap")

    dom = rec["domain"]
    pillar_label = rec["primary_pillar_label"]
    prim_name = rec["primary_method_name"]
    alt_name = rec["alternative_method_name"]
    alt_cond = rec["alternative_condition"]
    ss_label = rec["sample_size_label"]
    rationale = rec["rationale"]
    is_unsupervised = rec.get("sample_size_framework") == "unsupervised_clustering" or rec.get("primary_method_key") in ("kmeans_clustering", "dbscan_clustering")
    is_time_series = (
        rec.get("sample_size_framework") == "time_series_capacity"
        or rec.get("primary_method_key") in ("holt_winters_exponential_smoothing", "sarima_forecasting", "lagged_ensemble_forecasting")
        or rec.get("problem_key") == "time_series_forecasting"
    )

    if is_unsupervised:
        sizing_html = f"""
        <div style="font-size: 0.9rem; color: #854d0e; background-color: #fefce8; border: 1px solid #fef08a; border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.6rem;">
            <div style="font-weight: 700; color: #713f12; margin-bottom: 0.25rem;">
                📐 Machine Learning Sample Scale & Cluster Stability Guidance: <code>{ss_label}</code>
            </div>
            <div style="color: #854d0e; font-size: 0.84rem; line-height: 1.45;">
                • <strong>No Prior Null Hypothesis (H&#8320;):</strong> Classical statistical power formulas (&alpha; = 0.05, 1 - &beta; &ge; 80%) do not apply to unsupervised discovery because there are no pre-existing ground-truth groups to contrast. (A two-means t-test only applies if you subsequently run a <em>post-hoc hypothesis test</em> comparing the discovered clusters on an external continuous clinical outcome).<br>
                • <strong>Sample Scale Heuristics:</strong> Sizing requires at least N &ge; 20–50 samples per expected cluster (N &ge; 20k) and N &ge; 10 &times; p observations per feature to ensure geometric stability and avoid local-minima traps.<br>
                • <strong>Unsupervised Validation Metrics:</strong> Evaluate cluster partition quality using <strong>Silhouette Score</strong> (s &gt; 0.50: cohesive and separated), <strong>Davies-Bouldin Index</strong> (DBI &lt; 1.0: compact prototypes), and <strong>Calinski-Harabasz Index</strong> (Variance Ratio Criterion; higher is better).
            </div>
        </div>
        """
    elif is_time_series:
        sizing_html = f"""
        <div style="font-size: 0.9rem; color: #1e3a8a; background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.6rem;">
            <div style="font-weight: 700; color: #1e40af; margin-bottom: 0.25rem;">
                📈 Time Series Sequence Sizing & Rolling Validation: <code>{ss_label}</code>
            </div>
            <div style="color: #1e3a8a; font-size: 0.84rem; line-height: 1.45;">
                • <strong>No Classical Two-Sample Hypothesis Testing:</strong> Hypothesis testing power formulas (&alpha; = 0.05, 1 - &beta; &ge; 80%) do not apply to chronological time series forecasting because data points are serially autocorrelated rather than independent cross-sectional samples.<br>
                • <strong>Historical Sequence Length Requirements:</strong> Reliable level, trend, and seasonal index decomposition requires a minimum of <strong>4 to 8 full seasonal cycles</strong> (e.g., &ge; 28–56 daily observations for weekly cycles; N &ge; 50–100+ points) to prevent state parameter instability.<br>
                • <strong>Temporal Out-of-Sample Validation:</strong> Model accuracy and confidence intervals are evaluated using <strong>Rolling-Origin Cross-Validation</strong> (expanding/sliding windows) and tested against forward forecast horizons (e.g., 14-day lead times) using MAE, MAPE, and RMSE.
            </div>
        </div>
        """
    else:
        sizing_html = f"""
        <div style="font-size: 0.9rem; color: #1e3a8a; margin-top: 0.6rem; padding-top: 0.5rem; border-top: 1px dashed #cbd5e1;">
            <strong>📐 Matching Sample Size & Power Framework:</strong> <code>{ss_label}</code>
            <br>
            <span style="color: #475569; font-size: 0.84rem;">Pre-specifies significance level (&alpha; = 0.05: a 5% maximum risk of declaring a false discovery), target statistical power (1 - &beta; &ge; 80%: an 80% or greater chance of detecting a real effect), Minimal Clinically Important Difference (MCID), and attrition cushion under ICH-E9.</span>
        </div>
        """

    card_html = f"""
    <div style="background-color: #f0fdf4; border: 2px solid #16a34a60; border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: 700; color: #166534; font-size: 1.15rem;">
                🎯 Recommended Primary Method: {prim_name}
            </span>
            <span style="background-color: #dcfce7; color: #15803d; font-size: 0.85rem; font-weight: 700; padding: 0.25rem 0.65rem; border-radius: 6px;">
                {pillar_label}
            </span>
        </div>
        <div style="font-size: 0.95rem; color: #1f2937; margin-top: 0.5rem; line-height: 1.5;">
            <strong>Biostatistical Rationale:</strong> {rationale}
        </div>
        <div style="font-size: 0.9rem; color: #374151; margin-top: 0.5rem;">
            <strong>Alternative / Non-Parametric Fallback:</strong> <code>{alt_name}</code>
            <br>
            <span style="color: #6b7280; font-size: 0.85rem;"><em>When to switch:</em> {alt_cond}</span>
        </div>
        {sizing_html}
    </div>
    """
    render_html(card_html)

    # Prominent One-Click Deep-Link Action Buttons
    c_btn1, c_btn2 = st.columns(2)

    with c_btn1:
        st.markdown("#### 🚀 Step 3A: Launch Primary Method Module")
        st.caption(f"Opens the dedicated training module for **{prim_name}** with worked guides, formulas, and synthetic simulations.")
        btn_label = f"Launch {prim_name} 👉"
        if st.button(btn_label, key="btn_launch_primary", use_container_width=True):
            st.session_state["selector_consultation"] = {
                "active": True,
                "problem_name": rec.get("problem_name", "Research Problem"),
                "method_name": prim_name,
                "method_key": rec.get("primary_method_key", ""),
                "pillar_label": pillar_label,
                "domain_name": dom.get("name", "") if isinstance(dom, dict) else str(dom),
                "target_type": "method",
                "sample_scale": rec.get("checklist", [{}])[0].get("badge", "").replace("1. Sizing:", "").strip() if rec.get("checklist") else ""
            }
            deep_link = rec["deep_link"]
            for k, v in deep_link["session_state_updates"].items():
                st.session_state[k] = v
            st.switch_page(deep_link["page_route"])

    with c_btn2:
        if is_unsupervised:
            st.markdown("#### 🔬 Step 3B: Inspect Cluster Evaluation & Stability Metrics")
            st.caption("Opens the Machine Learning module with cluster validation metrics (Silhouette, Davies-Bouldin, and Calinski-Harabasz curves).")
            btn_ss_label = "Inspect Cluster Validation Metrics 👉"
            if st.button(btn_ss_label, key="btn_launch_sample_size", use_container_width=True):
                st.session_state["selector_consultation"] = {
                    "active": True,
                    "problem_name": rec.get("problem_name", "Research Problem"),
                    "method_name": prim_name,
                    "method_key": rec.get("primary_method_key", ""),
                    "pillar_label": pillar_label,
                    "domain_name": dom.get("name", "") if isinstance(dom, dict) else str(dom),
                    "target_type": "method",
                    "sample_scale": rec.get("checklist", [{}])[0].get("badge", "").replace("1. Sizing:", "").strip() if rec.get("checklist") else ""
                }
                ss_deep = rec["sample_size_deep_link"]
                for k, v in ss_deep["session_state_updates"].items():
                    st.session_state[k] = v
                st.switch_page(ss_deep["page_route"])
        elif is_time_series:
            st.markdown("#### 📈 Step 3B: Inspect Time Series Forecasting Horizon & Temporal Metrics")
            st.caption("Opens the Machine Learning module with Holt-Winters forecast decomposition, 95% prediction intervals, and rolling-origin temporal validation.")
            btn_ss_label = "Inspect Time Series Forecasting & Horizon Metrics 👉"
            if st.button(btn_ss_label, key="btn_launch_sample_size", use_container_width=True):
                st.session_state["selector_consultation"] = {
                    "active": True,
                    "problem_name": rec.get("problem_name", "Research Problem"),
                    "method_name": prim_name,
                    "method_key": rec.get("primary_method_key", ""),
                    "pillar_label": pillar_label,
                    "domain_name": dom.get("name", "") if isinstance(dom, dict) else str(dom),
                    "target_type": "method",
                    "sample_scale": rec.get("checklist", [{}])[0].get("badge", "").replace("1. Sizing:", "").strip() if rec.get("checklist") else ""
                }
                ss_deep = rec["sample_size_deep_link"]
                for k, v in ss_deep["session_state_updates"].items():
                    st.session_state[k] = v
                st.switch_page(ss_deep["page_route"])
        else:
            st.markdown("#### 📐 Step 3B: Calculate Required Sample Size & Power")
            st.caption(f"Opens Pillar 4 with **{ss_label}** pre-selected to generate power curves and protocol justification statements.")
            btn_ss_label = f"Calculate Sample Size for {ss_label} 👉"
            if st.button(btn_ss_label, key="btn_launch_sample_size", use_container_width=True):
                st.session_state["selector_consultation"] = {
                    "active": True,
                    "problem_name": rec.get("problem_name", "Research Problem"),
                    "method_name": f"Sample Size: {ss_label}",
                    "method_key": ss_label,
                    "pillar_label": "📐 Sample Size & Power (Pillar 4)",
                    "domain_name": dom.get("name", "") if isinstance(dom, dict) else str(dom),
                    "target_type": "sample_size",
                    "sample_scale": rec.get("checklist", [{}])[0].get("badge", "").replace("1. Sizing:", "").strip() if rec.get("checklist") else ""
                }
                ss_deep = rec["sample_size_deep_link"]
                for k, v in ss_deep["session_state_updates"].items():
                    st.session_state[k] = v
                st.switch_page(ss_deep["page_route"])


# =====================================================================
# 5. Interactive Visual Decision Pathway Flowchart
# =====================================================================

def render_decision_path_flowchart(rec: Dict[str, Any], form_inputs: Dict[str, Any]) -> None:
    """Renders an interactive visual flowchart depicting how the 4 input choices
    (Domain -> Objective -> Study Design -> Sample Scale) dynamically navigate
    the quantitative decision graph to produce the primary recommendation, alternative fallback,
    and matching sample size calculation framework.
    """
    with st.expander("🗺️ Visual Decision Pathway: How Your Method Was Routed", expanded=False):
        st.markdown(
            "Follow the dynamic visual pathway below to trace how your operational sector, research objective, "
            "study design architecture, and sample magnitude constraints navigated the quantitative decision graph "
            "to arrive at the mathematically justified recommendation."
        )

        domain_key = form_inputs.get("domain", "healthcare_clinical") if isinstance(form_inputs, dict) else "healthcare_clinical"
        domain_info = DOMAIN_PROFILES.get(domain_key, {})
        domain_name = domain_info.get("name", "Domain Context")
        domain_org = domain_info.get("organisation", "")
        is_unsupervised = (rec.get("sample_size_framework") == "unsupervised_clustering" or rec.get("primary_method_key") in ("kmeans_clustering", "dbscan_clustering")) if isinstance(rec, dict) else False
        is_time_series = (
            rec.get("sample_size_framework") == "time_series_capacity"
            or rec.get("primary_method_key") in ("holt_winters_exponential_smoothing", "sarima_forecasting", "lagged_ensemble_forecasting")
            or rec.get("problem_key") == "time_series_forecasting"
        ) if isinstance(rec, dict) else False

        goal_key = form_inputs.get("analytical_goal", "compare_groups") if isinstance(form_inputs, dict) else "compare_groups"
        goal_metadata = {
            "compare_groups": ("Compare 2 Groups", "⚖️", "Assess between-group differential effects"),
            "paired_change": ("Paired Pre-Post Change", "🔄", "Assess intra-subject repeated measures"),
            "correlation_association": ("Continuous Association", "📏", "Evaluate bivariate correlation between 2 metrics"),
            "contingency_rates": ("Categorical Proportions", "📊", "Cross-tabulate rates or events across cohorts"),
            "cohort_prospective": ("Prospective Cohort Incidence", "⏱️", "Track exposure to outcome forward in time (RR)"),
            "case_control_retrospective": ("Retrospective Case-Control", "🔍", "Outcome-dependent sampling for past exposures (OR)"),
            "diagnostic_accuracy": ("Diagnostic Accuracy", "🎯", "Evaluate sensitivity, specificity & AUC vs Gold Standard"),
            "survival_time_to_event": ("Survival / Time-to-Event", "⏳", "Analyze censored event times & hazard ratios"),
            "demand_forecasting": ("Demand & Volume Forecasting", "📈", "Forecast sequential chronological demand over time"),
            "predict_rare_event": ("Rare Event Classification", "🚨", "Predict low-prevalence binary outcomes"),
            "equity_fairness": ("Algorithmic Fairness Audit", "🛡️", "Audit protected demographic parity & Four-Fifths Rule"),
            "clustering_phenotyping": ("Unsupervised Phenotyping", "🧩", "Discover latent sub-phenotypes or clusters"),
            "dimension_reduction": ("Dimension Reduction (PCA)", "📉", "Compress high-dimensional collinear predictors")
        }
        goal_title, goal_icon, goal_desc = goal_metadata.get(
            goal_key,
            (goal_key.replace("_", " ").title() if isinstance(goal_key, str) else "Analytical Goal", "🎯", "Analytical Objective")
        )

        design_key = form_inputs.get("study_design", "parallel_rct") if isinstance(form_inputs, dict) else "parallel_rct"
        design_metadata = {
            "parallel_rct": ("Parallel RCT", "👥", "Randomized independent comparison arms"),
            "paired_pre_post": ("Pre-Post Repeated Measures", "🔄", "Within-subject matched pre-post measurements"),
            "multi_arm": ("Multi-Arm Trial (3+ Groups)", "📊", "Three or more independent experimental arms"),
            "observational_cohort": ("Prospective Longitudinal Cohort", "⏱️", "Exposed vs unexposed followed over time"),
            "case_control": ("Retrospective Case-Control", "🔍", "Cases matched to controls (1:k matching)"),
            "diagnostic_accuracy": ("Diagnostic Cross-Sectional", "🎯", "Index test vs reference Gold Standard"),
            "longitudinal_series": ("Sequential Operational Time Series", "📈", "Chronological regular interval records"),
            "cross_sectional": ("Cross-Sectional Survey", "📸", "Single point-in-time observational snapshot"),
            "sensor_telemetry": ("Sensor Telemetry Streams", "🏭", "High-frequency streaming machine sensor metrics"),
            "web_telemetry": ("Web & Clickstream Telemetry", "🌐", "Online user browsing & interaction logs"),
            "product_telemetry": ("Product SaaS Telemetry", "📱", "In-app digital product activity metrics")
        }
        design_title, design_icon, design_desc = design_metadata.get(
            design_key,
            (design_key.replace("_", " ").title() if isinstance(design_key, str) else "Study Design", "📐", "Study Architecture")
        )

        scale_key = form_inputs.get("sample_scale", "medium") if isinstance(form_inputs, dict) else "medium"
        scale_metadata = {
            "small": ("Small Cohort (N < 30)", "⚡", "CLT Untenable ➔ Non-Parametric Trigger", "#dc2626", "#fef2f2", "#b91c1c"),
            "medium": ("Moderate Sample (30 ≤ N < 500)", "📐", "Asymptotic Normality ➔ Parametric & Sizing", "#2563eb", "#eff6ff", "#1d4ed8"),
            "large": ("Large Scale (N ≥ 500)", "📊", "High Power ➔ Prioritize Effect Size & CIs", "#059669", "#ecfdf5", "#047857")
        }
        scale_title, scale_icon, scale_desc, scale_accent, scale_bg, scale_text = scale_metadata.get(
            scale_key,
            (scale_key.title() if isinstance(scale_key, str) else "Sample Scale", "📐", "Sample Magnitude", "#4b5563", "#f9fafb", "#1f2937")
        )

        prim_name = rec.get("primary_method_name") or "Primary Recommended Method"
        prim_pillar = rec.get("primary_pillar_label") or "Analytical Pillar"
        prim_param = rec.get("parametric_type") or "Analytical Framework"
        alt_name = rec.get("alternative_method_name") or "Alternative Method"
        alt_cond = rec.get("alternative_condition") or "Specific assumption violations."
        ss_label = rec.get("sample_size_label") or "Sample Size Framework"

        # HTML Escaping for safe rendering inside unsafe_allow_html
        safe_domain_name = html.escape(str(domain_name))
        safe_domain_org = html.escape(str(domain_org))
        safe_goal_title = html.escape(str(goal_title))
        safe_goal_desc = html.escape(str(goal_desc))
        safe_design_title = html.escape(str(design_title))
        safe_design_desc = html.escape(str(design_desc))
        safe_scale_title = html.escape(str(scale_title))
        safe_scale_desc = html.escape(str(scale_desc))
        safe_prim_name = html.escape(str(prim_name))
        safe_prim_pillar = html.escape(str(prim_pillar))
        safe_prim_param = html.escape(str(prim_param))
        safe_alt_name = html.escape(str(alt_name))
        safe_alt_cond = html.escape(str(alt_cond))
        safe_ss_label = html.escape(str(ss_label))

        pipeline_html = f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.1rem; margin-bottom: 1.25rem;">
            <div style="font-weight: 700; color: #1e293b; font-size: 1rem; margin-bottom: 0.25rem;">
                🧭 Dynamic Method Routing Pipeline
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 0.85rem;">
                Active path generated from your 4 operational inputs:
            </div>

            <div style="display: flex; flex-wrap: wrap; align-items: stretch; justify-content: flex-start; gap: 8px;">
                <!-- Gate 1: Sector -->
                <div style="flex: 1 1 170px; min-width: 155px; background: #ffffff; border: 1px solid #cbd5e1; border-top: 4px solid #3b82f6; border-radius: 8px; padding: 0.75rem 0.8rem; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.68rem; font-weight: 800; color: #3b82f6; letter-spacing: 0.05em; text-transform: uppercase;">
                            Gate 1 • Sector
                        </div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: #1e293b; margin-top: 0.25rem;">
                            {safe_domain_name}
                        </div>
                    </div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 0.45rem; border-top: 1px solid #f1f5f9; padding-top: 0.35rem;">
                        <em>{safe_domain_org}</em>
                    </div>
                </div>

                <div style="display: flex; align-items: center; justify-content: center; font-size: 1.1rem; color: #94a3b8; font-weight: 800; padding: 0 1px;">➔</div>

                <!-- Gate 2: Objective -->
                <div style="flex: 1 1 170px; min-width: 155px; background: #ffffff; border: 1px solid #cbd5e1; border-top: 4px solid #8b5cf6; border-radius: 8px; padding: 0.75rem 0.8rem; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.68rem; font-weight: 800; color: #8b5cf6; letter-spacing: 0.05em; text-transform: uppercase;">
                            Gate 2 • Objective
                        </div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: #1e293b; margin-top: 0.25rem;">
                            {goal_icon} {safe_goal_title}
                        </div>
                    </div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 0.45rem; border-top: 1px solid #f1f5f9; padding-top: 0.35rem;">
                        {safe_goal_desc}
                    </div>
                </div>

                <div style="display: flex; align-items: center; justify-content: center; font-size: 1.1rem; color: #94a3b8; font-weight: 800; padding: 0 1px;">➔</div>

                <!-- Gate 3: Study Architecture -->
                <div style="flex: 1 1 170px; min-width: 155px; background: #ffffff; border: 1px solid #cbd5e1; border-top: 4px solid #f59e0b; border-radius: 8px; padding: 0.75rem 0.8rem; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.68rem; font-weight: 800; color: #d97706; letter-spacing: 0.05em; text-transform: uppercase;">
                            Gate 3 • Architecture
                        </div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: #1e293b; margin-top: 0.25rem;">
                            {design_icon} {safe_design_title}
                        </div>
                    </div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 0.45rem; border-top: 1px solid #f1f5f9; padding-top: 0.35rem;">
                        {safe_design_desc}
                    </div>
                </div>

                <div style="display: flex; align-items: center; justify-content: center; font-size: 1.1rem; color: #94a3b8; font-weight: 800; padding: 0 1px;">➔</div>

                <!-- Gate 4: Scale Constraint -->
                <div style="flex: 1 1 170px; min-width: 155px; background: {scale_bg}; border: 1px solid #cbd5e1; border-top: 4px solid {scale_accent}; border-radius: 8px; padding: 0.75rem 0.8rem; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.68rem; font-weight: 800; color: {scale_text}; letter-spacing: 0.05em; text-transform: uppercase;">
                            Gate 4 • Scale Boundary
                        </div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: #1e293b; margin-top: 0.25rem;">
                            {scale_icon} {safe_scale_title}
                        </div>
                    </div>
                    <div style="font-size: 0.72rem; color: {scale_text}; margin-top: 0.45rem; border-top: 1px solid rgba(0,0,0,0.06); padding-top: 0.35rem;">
                        <strong>{safe_scale_desc}</strong>
                    </div>
                </div>

                <div style="display: flex; align-items: center; justify-content: center; font-size: 1.1rem; color: #94a3b8; font-weight: 800; padding: 0 1px;">➔</div>

                <!-- Final Route: Recommended Method Card -->
                <div style="flex: 1.3 1 230px; min-width: 215px; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 2px solid #16a34a; border-radius: 8px; padding: 0.8rem 0.9rem; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 2px 5px rgba(22, 163, 74, 0.12);">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.68rem; font-weight: 800; color: #15803d; letter-spacing: 0.05em; text-transform: uppercase;">
                                🎯 Final Route
                            </span>
                            <span style="background: #16a34a; color: #ffffff; font-size: 0.65rem; font-weight: 700; padding: 0.15rem 0.4rem; border-radius: 4px;">
                                {safe_prim_param}
                            </span>
                        </div>
                        <div style="font-size: 0.92rem; font-weight: 800; color: #14532d; margin-top: 0.25rem;">
                            {safe_prim_name}
                        </div>
                        <div style="font-size: 0.75rem; color: #166534; margin-top: 0.15rem;">
                            {safe_prim_pillar}
                        </div>
                    </div>

                    <div style="margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px dashed #86efac; font-size: 0.72rem; color: #1e3a8a;">
                        <div style="margin-bottom: 0.2rem;">
                            <strong>🔀 Fallback:</strong> <code>{safe_alt_name}</code>
                        </div>
                        <div>
                            <strong>📐 {"Stability Sizing:" if is_unsupervised else ("Temporal Sequence Sizing:" if is_time_series else "Power Sizing:")}</strong> <code>{safe_ss_label}</code>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        render_html(pipeline_html)

        st.markdown("#### 🔍 Step-by-Step Decision Routing Rationale")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            render_html(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.85rem; margin-bottom: 0.8rem;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #3b82f6; text-transform: uppercase;">
                    Gate 1 & 2: Domain Context & Analytical Objective
                </div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #1e293b; margin-top: 0.2rem;">
                    {goal_icon} {safe_goal_title} ({safe_domain_name})
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 0.35rem; line-height: 1.4;">
                    <strong>Routing Principle:</strong> The primary objective defines whether the analysis seeks to test a confirmatory hypothesis (Pillar 1), estimate population disease risk and association (Pillar 2), or train predictive supervised / unsupervised models (Pillar 3).
                </div>
            </div>
            """)

            # Gate 3 dynamic explanation
            if design_key == "paired_pre_post":
                design_expl = "Within-subject repeated measurements mean observations are non-independent. The model analyzes paired differences (Di = X_post,i - X_pre,i) rather than pooled variance."
            elif design_key == "multi_arm":
                design_expl = "Three or more independent experimental arms require omnibus hypothesis testing with multiplicity adjustments to prevent Family-Wise Error Rate (FWER) inflation."
            elif design_key == "case_control":
                design_expl = "Outcome-dependent sampling (Cases vs Controls) means disease incidence cannot be directly measured. Confounding is controlled and risk is estimated via Odds Ratios (OR)."
            elif design_key == "observational_cohort":
                design_expl = "Prospective tracking from exposure to outcome provides true risk denominators, enabling direct calculation of Incidence Rates and Relative Risk (RR)."
            elif design_key == "longitudinal_series":
                design_expl = "Chronological observations exhibit temporal autocorrelation and seasonal cycles, requiring lag structures and rolling-origin cross-validation rather than random shuffling."
            elif design_key == "sensor_telemetry":
                design_expl = "High-frequency streaming IoT sensor data contains non-linear degradation signals and severe class imbalance, requiring robust tree ensembles or anomaly scoring."
            elif design_key == "web_telemetry":
                design_expl = "Online digital session clickstreams require propensity modeling and fairness auditing (Disparate Impact Ratio) across demographic tiers."
            else:
                design_expl = "Randomized independent comparison arms ensure exchangeability between treatment and control, justifying unbiased between-group differential testing."

            safe_design_expl = html.escape(design_expl)

            render_html(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #d97706; text-transform: uppercase;">
                    Gate 3: Study Architecture & Dependency Gate
                </div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #1e293b; margin-top: 0.2rem;">
                    {design_icon} {safe_design_title}
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 0.35rem; line-height: 1.4;">
                    <strong>Structural Constraint:</strong> {safe_design_expl}
                </div>
            </div>
            """)

        with col_g2:
            # Gate 4 scale explanation
            if is_unsupervised:
                scale_expl = f"Unsupervised Clustering Sample Scale ({safe_scale_title}): Classical statistical power does not apply without a null hypothesis. Scale adequacy requires N >= 20–50 samples per expected cluster and N >= 10 * features. Partition quality is evaluated via Silhouette Score (s > 0.50), Davies-Bouldin Index (DBI < 1.0), and Calinski-Harabasz Index."
            elif is_time_series:
                scale_expl = f"Time Series Historical Sequence Sizing ({safe_scale_title}): Classical two-sample hypothesis testing power does not apply to sequential chronological series. Data adequacy mandates at least 4 to 8 full seasonal cycles (N >= 50–100+ observations) for stable level, trend, and seasonal index decomposition under rolling-origin temporal evaluation."
            elif scale_key == "small":
                scale_expl = f"Small Cohort (N < 30): The Central Limit Theorem does not guarantee asymptotic normality. Parametric tests are vulnerable to skewness and outliers. The engine automatically routed to {prim_name} to safeguard Type I error."
            elif scale_key == "large":
                scale_expl = f"Large Registry / Telemetry (N >= 500): With large cohorts, asymptotic normality is guaranteed, but statistical power is so high that clinically trivial effects reach p < 0.001. The analysis must prioritize Effect Size and 95% Confidence Intervals over raw p-values."
            else:
                scale_expl = f"Moderate Sample (30 <= N < 500): Asymptotic normality applies under the Central Limit Theorem. The parametric framework ({prim_name}) provides optimal statistical power and direct parameter interpretability."

            safe_scale_expl = html.escape(scale_expl)

            render_html(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.85rem; margin-bottom: 0.8rem;">
                <div style="font-size: 0.8rem; font-weight: 700; color: {scale_text}; text-transform: uppercase;">
                    Gate 4: Sample Scale & Distributional Boundary Gate
                </div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #1e293b; margin-top: 0.2rem;">
                    {scale_icon} {safe_scale_title}
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 0.35rem; line-height: 1.4;">
                    <strong>Distributional Rationale:</strong> {safe_scale_expl}
                </div>
            </div>
            """)

            if is_unsupervised:
                sizing_label_text = "Matching Stability Framework:"
                sizing_rationale_text = "ensures sufficient geometric density (N >= 20k) and cluster partition separation."
            elif is_time_series:
                sizing_label_text = "Matching Sequence Sizing:"
                sizing_rationale_text = "ensures minimum historical sequence length (4–8 seasonal cycles, N >= 50–100) for stable rolling-origin forecasting."
            else:
                sizing_label_text = "Matching Power Sizing:"
                sizing_rationale_text = "ensures study design pre-specifies alpha=0.05 and power >= 80% under ICH-E9."

            render_html(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.85rem;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #15803d; text-transform: uppercase;">
                    Contingency & Fallback Branching Rule
                </div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #1e293b; margin-top: 0.2rem;">
                    🔀 When to Switch to: {safe_alt_name}
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-top: 0.35rem; line-height: 1.4;">
                    <strong>Trigger Condition:</strong> {safe_alt_cond}<br>
                    <strong>{sizing_label_text}</strong> <code>{safe_ss_label}</code> {sizing_rationale_text}
                </div>
            </div>
            """)


# =====================================================================
# 6. Methodological Comparison Matrix (Pedagogical Framework)
# =====================================================================

def render_methodological_comparison_matrix(comparison: List[Dict[str, str]]):
    """Renders the pedagogical 3-pillar comparison matrix."""
    with st.expander("⚖️ Methodological Comparison: Why Statistics vs. Epidemiology vs. Machine Learning?", expanded=False):
        st.markdown("Understand how the three analytical pillars approach this research problem differently:")

        cols = st.columns(3)
        for i, item in enumerate(comparison):
            with cols[i]:
                is_prim = "Recommended" in item["status"]
                bg_color = "#f0fdf4" if is_prim else "#f8fafc"
                border_color = "#16a34a" if is_prim else "#cbd5e1"
                status_color = "#15803d" if is_prim else "#64748b"

                render_html(f"""
                <div style="background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 6px; padding: 0.9rem; height: 100%;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #1e293b; margin-bottom: 0.3rem;">
                        {item['pillar']}
                    </div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: {status_color}; margin-bottom: 0.5rem;">
                        {item['status']}
                    </div>
                    <div style="font-size: 0.85rem; color: #374151; margin-bottom: 0.4rem;">
                        <strong>Analytical Paradigm:</strong><br>{item['approach']}
                    </div>
                    <div style="font-size: 0.82rem; color: #475569; margin-bottom: 0.4rem;">
                        <strong>Optimizes For:</strong><br><code>{item['optimizes_for']}</code>
                    </div>
                    <p style="font-size: 0.78rem; color: #64748b; margin-top: 0.5rem; line-height: 1.3;">
                        <strong>When to Choose:</strong> {item['when_to_choose']}
                    </p>
                </div>
                """)


# =====================================================================
# 7. Study Protocol Verification Checklist
# =====================================================================

def render_study_protocol_checklist(checklist: List[Dict[str, str]]):
    """Renders the 5-point study planning and protocol verification checklist."""
    with st.expander("📋 5-Point Study Protocol & Methodological Verification Checklist", expanded=False):
        st.markdown("Ensure your study design satisfies international reporting and governance guidelines (CONSORT 2010, STROBE, TRIPOD, NHS AI Ethics):")

        cols = st.columns(len(checklist))
        for i, check in enumerate(checklist):
            with cols[i]:
                render_html(f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem; height: 100%;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #0369a1; margin-bottom: 0.25rem;">
                        {check['badge']}
                    </div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #1e293b;">
                        {check['title']}
                    </div>
                    <div style="font-size: 0.78rem; color: #475569; margin-top: 0.3rem;">
                        {check['requirement']}
                    </div>
                    <p style="font-size: 0.72rem; color: #64748b; margin-top: 0.4rem; line-height: 1.25;">
                        {check['guidance']}
                    </p>
                </div>
                """)

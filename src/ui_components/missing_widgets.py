"""Missing Data Mechanisms, Imputation & Sensitivity Audits UI Components.

Provides:
1. Zero-knowledge plain English introductory guide on Missingness Mechanisms, the Broken Thermometer analogy, and Rubin's Rules.
2. Mode selector (Guided Missing Data Scenario Wizard vs. Direct Method Explorer).
3. Dynamic interactive simulation playgrounds with Plotly distribution comparisons, variance shrinkage curves, Rubin's variance decomposition, and ICH E9 tipping point sensitivity curves.
4. Standard 4-tab container (Mathematical Formulation, Python Pipeline, Excel Guide with 3 expanders, Practice CSV Data).
"""

from typing import Dict, Any, List, Tuple, Optional
import streamlit as st

try:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    HAS_UI_STACK = True
except ImportError:
    pd = None
    np = None
    go = None
    HAS_UI_STACK = False

from src.missing_data.missing_schema import MISSING_DATA_METHODS, MISSING_DATA_SCENARIOS
from src.missing_data.missing_engine import (
    calculate_rubin_taxonomy_model,
    generate_rubin_dataset,
    calculate_complete_case_vs_single_model,
    generate_complete_case_dataset,
    calculate_mice_rubin_model,
    generate_mice_dataset,
    calculate_knn_imputation_model,
    generate_knn_dataset,
    calculate_mnar_tipping_point_model,
    generate_tipping_point_dataset
)


# =====================================================================
# 1. Plain-English Zero-Knowledge Pedagogical Guide
# =====================================================================

def render_missing_plain_english_guide() -> None:
    """Renders intuition-first conceptual guide to Rubin's missing data taxonomy,
    the broken thermometer analogy, single imputation traps, and Rubin's Rules.
    """
    with st.expander("📖 Plain English Guide: MCAR vs. MAR vs. MNAR & The Hazard of Naive Imputation", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 🌡️ The "Broken Thermometer" Analogy:
            Imagine a hospital clinic monitoring patient fevers. A temperature reading is missing:
            * **MCAR (Missing Completely at Random):**  
              The nurse dropped the thermometer on the floor and it shattered.  
              *Missingness is pure bad luck.* It has nothing to do with patient age or how sick they feel. Throwing away missing rows causes loss of power, but no systematic bias.
            * **MAR (Missing at Random):**  
              The clinic ran out of thermometers during the busy night shift when acute elderly admissions were highest.  
              *Missingness depends on what shift it was and patient age.* Once we control for age and admission time, missingness is random. **Multiple Imputation (MICE)** recovers valid, unbiased estimates.
            * **MNAR (Missing Not at Random):**  
              The patient's fever was so burning hot that the mercury exceeded the maximum 42°C scale, or the patient felt too delirious to hold the probe.  
              *The measurement itself caused the missingness.* Standard models fail; we must perform **Tipping Point Sensitivity Analysis** (ICH E9).
            """)
        with col_g2:
            st.markdown(r"""
            #### ⚠️ The Two Dangerous Traps in Medical Research:
            * **Trap 1: Complete Case Analysis (Listwise Deletion):**  
              If you delete every patient who has even one missing test result across 8 lab tests, you routinely throw away **40% to 60% of your patients**. This wastes public research funds and produces severe clinical selection bias.
            * **Trap 2: Mean Imputation (The Variance Killer):**  
              Plugging the average value into every missing blank makes your spreadsheet look complete, but **artificially crushes the data variance**. Your standard errors become falsely tiny, manufacturing **false-positive scientific breakthroughs ($p < 0.05$)** out of thin air.
            * **The Principled Solution (Rubin's Rules):**  
              Fill in each blank $m$ times with realistic random draws (MICE). Analyze all $m$ copies, then pool results. Rubin's Rules explicitly add a **Between-Imputation Penalty ($B$)** to your standard error to honestly reflect that you had to guess the missing values.
            """)


# =====================================================================
# 2. Navigation Mode Selector & Execution Controller
# =====================================================================

def render_missing_mode_selector() -> Tuple[str, str]:
    """Renders mode controller allowing toggling between:
    - Guided Missing Data Scenario Wizard
    - Direct Method Explorer
    """
    if "missing_nav_mode_radio" not in st.session_state:
        st.session_state["missing_nav_mode_radio"] = "🧭 Guided Missing Data Scenario Wizard"

    col_m1, col_m2 = st.columns([1.8, 2.2])
    with col_m1:
        nav_mode = st.radio(
            "Select Exploration Pathway:",
            options=["🧭 Guided Missing Data Scenario Wizard", "🔍 Direct Method Explorer"],
            key="missing_nav_mode_radio",
            horizontal=True
        )

    selected_method_key = "rubin_taxonomy"

    if nav_mode == "🧭 Guided Missing Data Scenario Wizard":
        with col_m2:
            scenario_names = {k: f"{v['title']}" for k, v in MISSING_DATA_SCENARIOS.items()}
            sel_scen_key = st.selectbox(
                "Choose a real-world clinical missingness scenario:",
                options=list(scenario_names.keys()),
                format_func=lambda k: scenario_names[k],
                key="missing_scenario_select"
            )
            selected_method_key = MISSING_DATA_SCENARIOS[sel_scen_key]["method_key"]
            st.caption(f"💡 *Scenario:* {MISSING_DATA_SCENARIOS[sel_scen_key]['description']}")
    else:
        with col_m2:
            method_options = {k: f"{v['icon']} {v['name']}" for k, v in MISSING_DATA_METHODS.items()}
            default_idx = 0
            if "missing_direct_method_key" in st.session_state and st.session_state["missing_direct_method_key"] in method_options:
                default_idx = list(method_options.keys()).index(st.session_state["missing_direct_method_key"])

            selected_method_key = st.selectbox(
                "Choose a Missing Data & Imputation method:",
                options=list(method_options.keys()),
                format_func=lambda k: method_options[k],
                index=default_idx,
                key="missing_direct_method_select"
            )
            st.session_state["missing_direct_method_key"] = selected_method_key

    return nav_mode, selected_method_key


# =====================================================================
# 3. Interactive Method Simulation Playgrounds
# =====================================================================

def _render_rubin_taxonomy_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 🔍 Interactive Playground: Rubin's Missingness Taxonomy & Diagnostic Audits")
    st.markdown(r"""
    Evaluate whether clinical missingness is **MCAR** (unrelated to patient state) or **MAR** (driven by disease acuity) 
    by testing mean differences and running Little's MCAR diagnostic test statistic.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        mechanism = st.selectbox(
            "True Underlying Missingness Mechanism:",
            options=["MAR_by_sofa", "MCAR", "MNAR"],
            format_func=lambda x: "MAR (Driven by SOFA Acuity)" if x == "MAR_by_sofa" else ("MCAR (Completely Random)" if x == "MCAR" else "MNAR (Driven by Lactate Itself)"),
            key="rubin_mech_select"
        )
    with c2:
        pct_missing = st.slider("Missingness Proportion (%):", min_value=10, max_value=50, value=25, step=5, key="rubin_pct_miss") / 100.0
    with c3:
        n_records = st.select_slider("ICU Patient Cohort Size (N):", options=[500, 1000, 2000], value=1000, key="rubin_n_records")

    results = calculate_rubin_taxonomy_model(n_records=n_records, pct_missing=pct_missing, mechanism=mechanism)

    # Metric Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Observed Records", f"{results['n_observed']:,}")
    with m2:
        st.metric("Missing Records", f"{results['n_missing']:,} ({results['actual_pct_missing']:.1f}%)")
    with m3:
        st.metric("SOFA Mean: Observed", f"{results['mean_sofa_observed']:.2f}")
    with m4:
        st.metric("SOFA Mean: Missing", f"{results['mean_sofa_missing']:.2f}")
    with m5:
        st.metric("Little's / Audit p-value", f"{results['p_value']:.4f}")

    # Decision Banner
    if "Reject MCAR" in results["diagnosis"] or "Evidence Against MCAR" in results["diagnosis"]:
        st.warning(f"### {results['diagnosis']}")
        st.info("💡 **Methodological Directive:** Do NOT use Complete Case analysis. Missingness is covariate-dependent (MAR); deploy **Multiple Imputation by Chained Equations (MICE)**.")
    else:
        st.success(f"### {results['diagnosis']}")
        st.caption("Missingness appears independent of patient illness severity. Complete case analysis is unbiased, though Multiple Imputation still recovers lost statistical power.")

    if HAS_UI_STACK and go is not None:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Mean SOFA Acuity Score",
            x=["Observed Patients (Lactate Recorded)", "Missing Patients (Lactate Missing)"],
            y=[results["mean_sofa_observed"], results["mean_sofa_missing"]],
            marker_color=["#16a34a", "#dc2626"],
            text=[f"{results['mean_sofa_observed']:.2f}", f"{results['mean_sofa_missing']:.2f}"],
            textposition="auto"
        ))
        fig.update_layout(
            title=f"Diagnostic Audit: SOFA Acuity Score Comparison (Δ = {results['delta_sofa']:+.2f}, p = {results['p_value']:.4f})",
            yaxis_title="Mean SOFA Score",
            template="plotly_white",
            height=380,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    df_sample = pd.DataFrame(results["sample_records"]) if HAS_UI_STACK and pd is not None else []
    _render_rubin_taxonomy_tabs(meta, df_sample, results)


def _render_complete_case_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### ⚠️ Interactive Playground: Complete Case Attrition & Mean Imputation Hazards")
    st.markdown(r"""
    Observe how combining modest missingness across multiple variables decimates your usable sample size in Complete Case analysis, 
    and how naive Mean Imputation collapses standard errors to manufacture false-positive findings.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_covariates = st.slider("Number of Model Covariates (K):", min_value=2, max_value=12, value=6, step=1, key="cc_k_vars")
    with c2:
        pct_miss = st.slider("Missingness per Covariate (%):", min_value=5, max_value=30, value=12, step=1, key="cc_pct_miss") / 100.0
    with c3:
        n_records = st.number_input("Starting Study Cohort Size (N):", min_value=100, max_value=5000, value=500, step=50, key="cc_n_records")

    results = calculate_complete_case_vs_single_model(n_records=n_records, n_covariates=n_covariates, pct_missing_per_var=pct_miss)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Retained Patients (Complete Case)", f"{results['empirical_retained_count']} ({results['empirical_retained_pct']:.1f}%)")
        st.caption(f"Discarded: {results['empirical_discarded_count']} patients ({results['empirical_discarded_pct']:.1f}%)")
    with m2:
        st.metric("Theoretical Retention (1-p)^K", f"{results['theoretical_retained_pct']:.1f}%")
        st.caption(f"Exponential sample decay")
    with m3:
        st.metric("Mean Imputation Variance Shrinkage", f"-{results['variance_shrinkage_pct']:.1f}%")
        st.caption(f"Retained Variance: {results['retained_variance_pct']:.1f}%")
    with m4:
        st.metric("SE Deflation Factor", f"{results['se_deflation_factor']:.3f}")
        st.caption(f"Standard error falsely shrunken by {results['se_reduction_pct']:.1f}%")

    if HAS_UI_STACK and go is not None:
        fig = go.Figure()
        k_range = list(range(1, 13))
        ret_curve = [((1.0 - pct_miss) ** k) * 100.0 for k in k_range]

        fig.add_trace(go.Scatter(
            x=k_range, y=ret_curve,
            mode="lines+markers",
            line=dict(color="#dc2626", width=2.5),
            name=f"Retention at {results['pct_missing_per_var']:.0f}% Missingness"
        ))
        fig.add_vline(x=n_covariates, line=dict(color="#16a34a", dash="dash"), annotation_text=f"Selected K={n_covariates}")

        fig.update_layout(
            title="Complete Case Sample Size Attrition Curve: P(Retained) = (1 - p)^K",
            xaxis_title="Number of Multivariable Covariates (K)",
            yaxis_title="Retained Study Cohort (%)",
            yaxis=dict(range=[0, 105]),
            template="plotly_white",
            height=380,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    df_cc = pd.DataFrame(results["sample_records"]) if HAS_UI_STACK and pd is not None else []
    _render_complete_case_tabs(meta, df_cc, results)


def _render_mice_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 🎲 Interactive Playground: MICE Imputation & Rubin's Rules Pooling")
    st.markdown(r"""
    Simulate Multiple Imputation across $m$ parallel datasets and explore how **Rubin's Rules** combine within-imputation variance 
    and between-imputation disagreement into the authoritative pooled standard error.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        m_imputations = st.slider("Number of Imputed Datasets (m):", min_value=3, max_value=25, value=10, step=1, key="mice_m_slider")
    with c2:
        pct_missing = st.slider("Missingness on Primary Outcome (%):", min_value=5, max_value=45, value=20, step=5, key="mice_miss_pct") / 100.0
    with c3:
        true_effect = st.number_input("True Treatment Effect (ΔHbA1c):", value=-0.65, step=0.05, key="mice_effect")

    results = calculate_mice_rubin_model(m_imputations=m_imputations, pct_missing=pct_missing, true_treatment_effect=true_effect)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Pooled Effect (Q-bar)", f"{results['q_bar_pooled']:.3f}")
    with m2:
        st.metric("Pooled SE", f"{results['pooled_se']:.4f}")
    with m3:
        st.metric("Between Variance (B)", f"{results['b_between_variance']:.5f}")
    with m4:
        st.metric("Fraction Missing Info (FMI)", f"{results['fmi_pct']:.1f}%")
    with m5:
        st.metric("Degrees of Freedom (ν)", f"{results['degrees_of_freedom']:.1f}")

    st.success(f"### 🎯 Pooled 95% Confidence Interval: [{results['ci_95'][0]:.3f}, {results['ci_95'][1]:.3f}] | p = {results['p_value']:.4e}")

    if HAS_UI_STACK and go is not None:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Within-Imputation Variance (U-bar)",
            x=["Variance Decomposition"],
            y=[results["u_bar_within_variance"]],
            marker_color="#2563eb"
        ))
        finite_correction_b = (1.0 + 1.0 / m_imputations) * results["b_between_variance"]
        fig.add_trace(go.Bar(
            name="Between-Imputation Penalty (1 + 1/m)B",
            x=["Variance Decomposition"],
            y=[finite_correction_b],
            marker_color="#f59e0b"
        ))
        fig.update_layout(
            barmode="stack",
            title=f"Rubin's Total Variance Decomposition: T = U-bar + (1 + 1/m)B (FMI: {results['fmi_pct']:.1f}%)",
            yaxis_title="Variance Magnitude",
            template="plotly_white",
            height=360,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    df_mice = pd.DataFrame(results["imputation_details"]) if HAS_UI_STACK and pd is not None else []
    _render_mice_tabs(meta, df_mice, results)


def _render_knn_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 🤖 Interactive Playground: k-NN Distance-Based Imputation")
    st.markdown(r"""
    Explore non-parametric donor matching: finding the $k$ most similar complete patient records in multidimensional space 
    to impute missing biomarker values without assuming linear relationships.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        k_neighbors = st.slider("Number of Donor Neighbors (k):", min_value=1, max_value=15, value=5, step=1, key="knn_k_slider")
    with c2:
        pct_missing = st.slider("Biomarker Missingness (%):", min_value=5, max_value=30, value=15, step=5, key="knn_pct_miss") / 100.0
    with c3:
        n_samples = st.select_slider("Patient Cohort Size (N):", options=[200, 400, 800], value=400, key="knn_n_samples")

    results = calculate_knn_imputation_model(n_samples=n_samples, k_neighbors=k_neighbors, pct_missing=pct_missing)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Complete Donors", f"{results['n_donors']}")
    with m2:
        st.metric("Imputed Recipients", f"{results['n_imputed']}")
    with m3:
        st.metric("Imputation RMSE", f"{results['imputation_rmse']:.2f}")
    with m4:
        st.metric("Mean Absolute % Error", f"{results['imputation_mape_pct']:.1f}%")

    if HAS_UI_STACK and go is not None and results["imputed_records"]:
        fig = go.Figure()
        true_vals = [r["true_biomarker"] for r in results["imputed_records"]]
        imp_vals = [r["imputed_biomarker"] for r in results["imputed_records"]]

        fig.add_trace(go.Scatter(
            x=true_vals, y=imp_vals,
            mode="markers",
            marker=dict(size=8, color="#2563eb", opacity=0.7),
            name="Imputed vs True"
        ))
        # 45-degree identity line
        min_v = min(min(true_vals), min(imp_vals))
        max_v = max(max(true_vals), max(imp_vals))
        fig.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode="lines",
            line=dict(color="#16a34a", dash="dash"),
            name="Perfect Agreement (45°)"
        ))
        fig.update_layout(
            title=f"k-NN Imputation Fidelity: True vs Imputed Values (k={k_neighbors}, RMSE={results['imputation_rmse']:.2f})",
            xaxis_title="True Masked Biomarker Level",
            yaxis_title="k-NN Imputed Biomarker Level",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    df_knn = pd.DataFrame(results["imputed_records"]) if HAS_UI_STACK and pd is not None else []
    _render_knn_tabs(meta, df_knn, results)


def _render_tipping_point_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 🎯 Interactive Playground: MNAR Tipping Point Sensitivity Analysis")
    st.markdown(r"""
    Stress-test clinical trial robustness against informative dropout under **ICH E9 (R1)** guidelines. 
    Systematically apply clinical deterioration penalties ($\delta$) to dropouts until the treatment effect crosses $p \ge 0.05$.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        baseline_diff = st.number_input("Baseline MAR Treatment Effect (Δ Score):", value=-3.40, step=0.20, key="tip_diff")
        se_mar = st.number_input("Standard Error of Difference (SE):", value=1.15, step=0.05, key="tip_se")
    with c2:
        dropout_treated = st.slider("Treated Arm Dropout Rate (%):", min_value=5, max_value=40, value=22, step=1, key="tip_drop_t") / 100.0
    with c3:
        max_delta = st.slider("Max Tested Clinical Penalty (δ):", min_value=2.0, max_value=10.0, value=6.0, step=0.5, key="tip_max_delta")

    results = calculate_mnar_tipping_point_model(
        baseline_mar_diff=baseline_diff,
        se_mar=se_mar,
        dropout_rate_treated=dropout_treated,
        delta_max=max_delta
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Tipping Point (δ*)", results["tipping_point_str"])
    with m2:
        st.metric("Trial Robustness Classification", results["robustness_tier"])
    with m3:
        st.metric("Treated Dropouts", f"{results['dropout_rate_treated']:.1f}%")

    if "Fragile" in results["robustness_tier"]:
        st.error(f"### {results['robustness_verdict']}")
    else:
        st.success(f"### {results['robustness_verdict']}")

    if HAS_UI_STACK and go is not None:
        fig = go.Figure()
        deltas = [g["delta_penalty"] for g in results["grid_points"]]
        effs = [g["adjusted_treatment_effect"] for g in results["grid_points"]]
        upper_ci = [g["ci_95_upper"] for g in results["grid_points"]]
        lower_ci = [g["ci_95_lower"] for g in results["grid_points"]]

        fig.add_trace(go.Scatter(x=deltas, y=effs, mode="lines+markers", name="Adjusted Treatment Effect", line=dict(color="#2563eb", width=2.5)))
        fig.add_trace(go.Scatter(x=deltas, y=upper_ci, mode="lines", name="Upper 95% CI", line=dict(color="#dc2626", dash="dash")))
        fig.add_trace(go.Scatter(x=deltas, y=lower_ci, mode="lines", name="Lower 95% CI", line=dict(color="#64748b", dash="dash")))

        fig.add_hline(y=0.0, line=dict(color="#0f172a", width=1.5), annotation_text="Null Effect (p = 0.05 Threshold)")
        if results["tipping_point_delta"] is not None:
            fig.add_vline(x=results["tipping_point_delta"], line=dict(color="#dc2626", width=2), annotation_text=f"Tipping Point δ* = {results['tipping_point_delta']:.2f}")

        fig.update_layout(
            title="ICH E9 (R1) Tipping Point Curve: Effect Attenuation across Dropout Penalty δ",
            xaxis_title="Dropout Deterioration Penalty (δ points)",
            yaxis_title="Treatment Effect (95% CI)",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    df_tip = pd.DataFrame(results["grid_points"]) if HAS_UI_STACK and pd is not None else []
    _render_tipping_point_tabs(meta, df_tip, results)


# =====================================================================
# 4. Standard 4-Tab Pedagogical Container
# =====================================================================

def _render_rubin_taxonomy_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs([
        "📖 Mathematical Formulation & Causal Derivation",
        "💻 Implementation in Python",
        "📊 Excel Implementation",
        "📥 Download Practice Data (CSV)"
    ])
    with t1:
        st.markdown(meta["mathematical_formulation"])
        with st.expander("🔍 Plain English Glossary of Terms", expanded=False):
            for term, desc in meta["plain_english_glossary"].items():
                st.markdown(f"- **{term}**: {desc}")
        with st.expander("🛡️ Critical Assumptions & Failure Modes", expanded=False):
            for a in meta["assumptions"]:
                st.markdown(f"- {a}")
    with t2:
        st.markdown("#### Production Python Pipeline: Diagnostic Audit & Little's MCAR Test")
        st.code("""import pandas as pd
import statsmodels.api as sm
from scipy import stats

df = pd.read_csv("icu_sepsis_missing_audit.csv")
df["m_lactate"] = df["recorded_lactate"].isna().astype(int)

# Logistic regression audit: M ~ SOFA + Age
X = sm.add_constant(df[["sofa_score", "age"]])
logit_model = sm.Logit(df["m_lactate"], X).fit()
print(logit_model.summary())

# If SOFA p < 0.05, MCAR is rejected -> Data is MAR or MNAR!
sofa_p = logit_model.pvalues["sofa_score"]
print(f"Missingness Audit p-value for SOFA: {sofa_p:.4e}")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Missingness Diagnostic Audit")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Column A (`Patient_ID`):** `ICU-0001` to `ICU-1000`.
            * **Column B (`SOFA_Score`):** Patient acuity score (0-20).
            * **Column C (`Recorded_Lactate`):** Arterial lactate or empty blank.
            * **Column D (`Missing_Indicator_M`):** `=IF(ISBLANK(C2), 1, 0)`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame) and not df.empty:
                st.dataframe(df.head(6), use_container_width=True, height=240, hide_index=True)
            df_cells = pd.DataFrame({
                "Cell / Range": ["Cell F2", "Cell F3", "Cell F4", "Cell F5"],
                "Target Metric": ["Mean SOFA: Observed", "Mean SOFA: Missing", "Difference t-test p-value", "Diagnostic Verdict"],
                "Exact Excel Formula": ['=AVERAGEIFS(B2:B1001, D2:D1001, 0)', '=AVERAGEIFS(B2:B1001, D2:D1001, 1)', '=T.TEST(FILTER(B2:B1001, D2:D1001=0), FILTER(B2:B1001, D2:D1001=1), 2, 3)', '=IF(F4<0.05, "Reject MCAR (MAR/MNAR)", "Fail to Reject MCAR")'],
                "Calculated Output Value": [f"{results['mean_sofa_observed']:.2f}", f"{results['mean_sofa_missing']:.2f}", f"{results['p_value']:.4f}", results["mechanism_verdict"]]
            }) if pd is not None else None
            if df_cells is not None:
                st.dataframe(df_cells, use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Flag Blanks:** In `Cell D2`, enter `=IF(ISBLANK(C2), 1, 0)` and drag down.
            2. **Compute Group Means:** Use `=AVERAGEIFS` to compare observed vs missing patient characteristics.
            3. **Run Two-Sample t-test:** Enter `=T.TEST(range_obs, range_mis, 2, 3)` (Welch unequal variances).
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Clinical Audit Interpretation:
            * **Audit Finding:** Missing lactate patients have a mean SOFA score of **{results['mean_sofa_missing']:.2f}** vs **{results['mean_sofa_observed']:.2f}** for observed patients (p = {results['p_value']:.4f}).
            * **Guidance:** Because missingness is correlated with clinical acuity, listwise deletion will drop the sickest patients, biasing mortality studies. **MICE imputation is required**.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: ICU Sepsis Missing Lactate Audit")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download Missing Lactate Dataset (CSV)", csv_bytes, "icu_missing_lactate_audit.csv", "text/csv", key="dl_rubin_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)


def _render_complete_case_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs([
        "📖 Mathematical Formulation & Causal Derivation",
        "💻 Implementation in Python",
        "📊 Excel Implementation",
        "📥 Download Practice Data (CSV)"
    ])
    with t1:
        st.markdown(meta["mathematical_formulation"])
        with st.expander("🔍 Plain English Glossary of Terms", expanded=False):
            for term, desc in meta["plain_english_glossary"].items():
                st.markdown(f"- **{term}**: {desc}")
        with st.expander("🛡️ Critical Assumptions & Failure Modes", expanded=False):
            for a in meta["assumptions"]:
                st.markdown(f"- {a}")
    with t2:
        st.markdown("#### Production Python Pipeline: Auditing Complete Case Attrition")
        st.code("""import pandas as pd

df = pd.read_csv("cohort_missingness.csv")
n_initial = len(df)
df_clean = df.dropna()
n_retained = len(df_clean)

attrition_pct = (1.0 - n_retained / n_initial) * 100
print(f"Initial N: {n_initial} | Retained N: {n_retained} | Discarded: {attrition_pct:.1f}%")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Complete Case Attrition")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Column A to F:** 6 multivariable clinical predictors.
            * **Column G (`Missing_Count`):** `=COUNTBLANK(A2:F2)`.
            * **Column H (`Retained_Complete`):** `=IF(G2=0, 1, 0)`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                st.dataframe(df.head(6), use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Count Blanks across Row:** In `Cell G2`, enter `=COUNTBLANK(A2:F2)`.
            2. **Sum Retained Rows:** In summary cell, enter `=SUM(H2:H501) / 500`.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Methodological Hazard Interpretation:
            * **Sample Attrition:** With {results['n_covariates']} covariates missing {results['pct_missing_per_var']:.0f}% each, listwise deletion **discards {results['empirical_discarded_pct']:.1f}% of all records**.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: Complete Case Attrition Table")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download Attrition Data (CSV)", csv_bytes, "complete_case_attrition.csv", "text/csv", key="dl_cc_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)


def _render_mice_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs([
        "📖 Mathematical Formulation & Causal Derivation",
        "💻 Implementation in Python",
        "📊 Excel Implementation",
        "📥 Download Practice Data (CSV)"
    ])
    with t1:
        st.markdown(meta["mathematical_formulation"])
        with st.expander("🔍 Plain English Glossary of Terms", expanded=False):
            for term, desc in meta["plain_english_glossary"].items():
                st.markdown(f"- **{term}**: {desc}")
        with st.expander("🛡️ Critical Assumptions & Failure Modes", expanded=False):
            for a in meta["assumptions"]:
                st.markdown(f"- {a}")
    with t2:
        st.markdown("#### Production Python Pipeline: MICE Imputation & Rubin's Rules Pooling")
        st.code("""import numpy as np

# Rubin's Rules Pooling Engine
def rubins_rules(estimates, variances):
    m = len(estimates)
    q_bar = np.mean(estimates)
    u_bar = np.mean(variances)
    b = np.var(estimates, ddof=1)
    t = u_bar + (1.0 + 1.0 / m) * b
    se = np.sqrt(t)
    fmi = (1.0 + 1.0 / m) * b / t
    return q_bar, se, fmi

# Example pooled output across m=10 imputations
q_bar, se, fmi = rubins_rules([-0.68, -0.62, -0.64, -0.66, -0.61], [0.014, 0.015, 0.014, 0.013, 0.014])
print(f"Pooled Treatment Effect: {q_bar:.3f} | SE: {se:.4f} | FMI: {fmi*100:.1f}%")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Rubin's Rules Pooling")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Column A (`Imputation`):** Imputation index `1` to `10`.
            * **Column B (`Q_l`):** Treatment effect estimate $\hat{Q}_l$.
            * **Column C (`U_l`):** Within-imputation variance $U_l = SE_l^2$.
            * **Cell B12 (`Q_bar`):** `=AVERAGE(B2:B11)`.
            * **Cell B13 (`U_bar`):** `=AVERAGE(C2:C11)`.
            * **Cell B14 (`B_var`):** `=VAR.S(B2:B11)`.
            * **Cell B15 (`Total_Var_T`):** `=B13 + (1 + 1/10)*B14`.
            * **Cell B16 (`Pooled_SE`):** `=SQRT(B15)`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                st.dataframe(df, use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Compute Point Estimate:** `=AVERAGE(B2:B11)`.
            2. **Compute Within-Variance:** `=AVERAGE(C2:C11)`.
            3. **Compute Between-Variance:** `=VAR.S(B2:B11)`.
            4. **Compute Total Pooled SE:** `=SQRT(U_bar + (1 + 1/m)*B)`.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Clinical Trial Interpretation:
            * **Pooled Result:** The pooled treatment effect is **{results['q_bar_pooled']:.3f}** (SE: {results['pooled_se']:.4f}, 95% CI: [{results['ci_95'][0]:.3f}, {results['ci_95'][1]:.3f}]).
            * **Missing Data Impact:** The Fraction of Missing Information is **{results['fmi_pct']:.1f}%**, confirming that Rubin's Rules accurately penalize the standard error for missingness uncertainty.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: MICE Imputation Pooling Table")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download MICE Pooling Table (CSV)", csv_bytes, "mice_rubin_pooling_table.csv", "text/csv", key="dl_mice_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df, use_container_width=True, height=240, hide_index=True)


def _render_knn_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs([
        "📖 Mathematical Formulation & Causal Derivation",
        "💻 Implementation in Python",
        "📊 Excel Implementation",
        "📥 Download Practice Data (CSV)"
    ])
    with t1:
        st.markdown(meta["mathematical_formulation"])
        with st.expander("🔍 Plain English Glossary of Terms", expanded=False):
            for term, desc in meta["plain_english_glossary"].items():
                st.markdown(f"- **{term}**: {desc}")
        with st.expander("🛡️ Critical Assumptions & Failure Modes", expanded=False):
            for a in meta["assumptions"]:
                st.markdown(f"- {a}")
    with t2:
        st.markdown("#### Production Python Pipeline: scikit-learn KNNImputer")
        st.code("""from sklearn.impute import KNNImputer
import pandas as pd

df = pd.read_csv("clinical_metabolomics.csv")
imputer = KNNImputer(n_neighbors=5, weights="distance")
imputed_array = imputer.fit_transform(df[["age", "bmi", "systolic_bp", "biomarker"]])
df_imputed = pd.DataFrame(imputed_array, columns=["age", "bmi", "systolic_bp", "biomarker"])
print(df_imputed.head())
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: k-NN Donor Matching")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Columns A-C:** Standardized predictor $z$-scores `=STANDARDIZE(X, mean, sd)`.
            * **Column D:** Target biomarker value.
            * **Column E:** Euclidean distance to donor recipient `=SQRT(SUMXMY2(A2:C2, $A$15:$C$15))`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                st.dataframe(df.head(6), use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Compute Distances:** Use `=SQRT(SUMXMY2(...))` to calculate Euclidean distance to candidate donors.
            2. **Find k Nearest:** Use `=SMALL(distance_range, {1,2,3,4,5})` with `=INDEX(MATCH(...))` to retrieve donor values.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Machine Learning Imputation Interpretation:
            * **Fidelity Benchmark:** The k-NN imputation achieved an RMSE of **{results['imputation_rmse']:.2f}** (MAPE: {results['imputation_mape_pct']:.1f}%), accurately reproducing non-linear multi-biomarker trajectories.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: k-NN Imputed Biomarker Cohort")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download k-NN Imputed Data (CSV)", csv_bytes, "knn_imputed_biomarkers.csv", "text/csv", key="dl_knn_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)


def _render_tipping_point_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs([
        "📖 Mathematical Formulation & Causal Derivation",
        "💻 Implementation in Python",
        "📊 Excel Implementation",
        "📥 Download Practice Data (CSV)"
    ])
    with t1:
        st.markdown(meta["mathematical_formulation"])
        with st.expander("🔍 Plain English Glossary of Terms", expanded=False):
            for term, desc in meta["plain_english_glossary"].items():
                st.markdown(f"- **{term}**: {desc}")
        with st.expander("🛡️ Critical Assumptions & Failure Modes", expanded=False):
            for a in meta["assumptions"]:
                st.markdown(f"- {a}")
    with t2:
        st.markdown("#### Production Python Pipeline: ICH E9 (R1) Tipping Point Sensitivity Analysis")
        st.code("""import numpy as np

def run_tipping_point(baseline_diff=-3.40, se=1.15, dropout_rate=0.22, deltas=np.linspace(0, 6, 25)):
    for delta in deltas:
        eff_adj = baseline_diff + (dropout_rate * delta)
        ci_upper = eff_adj + 1.96 * se
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(eff_adj / se)))
        if ci_upper >= 0.0 or p_val >= 0.05:
            print(f"Tipping Point Reached at delta* = {delta:.2f} points (p = {p_val:.4f})")
            return delta
    return None

run_tipping_point()
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Tipping Point Sensitivity Grid")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Column A (`Delta_Penalty`):** `0.0` to `6.0` in steps of `0.25`.
            * **Column B (`Adjusted_Effect`):** `=$B$1 + ($B$2 * A2)` where B1 is baseline difference and B2 is dropout rate.
            * **Column C (`Upper_95_CI`):** `=B2 + (1.96 * $B$3)`.
            * **Column D (`P_Value`):** `=2 * (1 - NORM.S.DIST(ABS(B2/$B$3), TRUE))`.
            * **Column E (`Is_Significant`):** `=IF(C2<0, "YES", "TIPPED (NO)")`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                st.dataframe(df.head(6), use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Delta Progression Grid:** Fill Column A from `0.00` to `6.00`.
            2. **Compute Adjusted Effect:** Enter `=$B$1 + ($B$2 * A2)`.
            3. **Locate Tipping Point:** Enter `=INDEX(A2:A26, MATCH("TIPPED (NO)", E2:E26, 0))`.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Regulatory Sensitivity Interpretation (ICH E9):
            * **Tipping Point:** Statistical significance tips at **δ* = {results['tipping_point_str']}**.
            * **Regulatory Conclusion:** {results['robustness_verdict']}
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: Tipping Point Sensitivity Grid")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download Tipping Point Grid (CSV)", csv_bytes, "ich_e9_tipping_point_grid.csv", "text/csv", key="dl_tip_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)


# =====================================================================
# 5. Master Module Router
# =====================================================================

def render_missing_data_method_module(method_key: str) -> None:
    """Master router executing the chosen Missing Data & Imputation method module."""
    meta = MISSING_DATA_METHODS.get(method_key, MISSING_DATA_METHODS["rubin_taxonomy"])

    st.markdown(f"## {meta['icon']} {meta['name']}")
    st.markdown(f"*{meta['tagline']}*")

    if method_key == "rubin_taxonomy":
        _render_rubin_taxonomy_sandbox(meta)
    elif method_key == "complete_case_vs_single":
        _render_complete_case_sandbox(meta)
    elif method_key == "mice_rubin_rules":
        _render_mice_sandbox(meta)
    elif method_key == "knn_missforest":
        _render_knn_sandbox(meta)
    elif method_key == "mnar_tipping_point":
        _render_tipping_point_sandbox(meta)

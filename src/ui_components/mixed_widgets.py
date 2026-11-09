"""Longitudinal & Hierarchical Mixed-Effects Models UI Components.

Provides interactive pedagogical simulators, diagnostic playbacks,
3-step Microsoft Excel implementations, and publication-grade Python pipelines across:
1. Linear Mixed-Effects Models (LMM): Random Intercepts & Slopes
2. Generalized Linear Mixed Models (GLMM): Repeated Binary & Count Outcomes
3. Repeated Measures ANOVA vs. Mixed Models: Sphericity & Missing Data Tradeoffs
4. Hierarchical 3-Level Clustered Models: Patients within Clinicians within Hospitals
5. Covariance Structure Selection & Model Diagnostics: ML vs. REML, AIC & Nakagawa R²
"""

import html
import io
import math
import random
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st

from src.mixed_models.mixed_schema import MIXED_MODELS_METHODS, MIXED_MODELS_SCENARIOS
from src.mixed_models.mixed_engine import (
    calculate_lmm_random_slopes_model,
    generate_lmm_dataset,
    calculate_glmm_binary_count_model,
    generate_glmm_dataset,
    calculate_rm_anova_vs_lmm_model,
    generate_rm_anova_dataset,
    calculate_hierarchical_3level_model,
    generate_3level_dataset,
    calculate_covariance_selection_model,
    generate_covariance_dataset
)


def render_html(html_str: str) -> None:
    """Safely renders HTML content, stripping excessive indentation to prevent
    CommonMark code-block misinterpretation.
    """
    clean_lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    cleaned_html = "".join(clean_lines)
    if hasattr(st, "html"):
        st.html(cleaned_html)
    else:
        st.markdown(cleaned_html, unsafe_allow_html=True)


# =====================================================================
# 1. Plain English Zero-Knowledge Guide
# =====================================================================

def render_mixed_plain_english_guide() -> None:
    """Renders plain-English conceptual foundation for longitudinal & hierarchical modeling."""
    with st.expander("📖 Intuition First: Why Longitudinal & Multilevel Models Dominate Classical Statistics", expanded=False):
        guide_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #4f46e5; padding: 1rem 1.25rem; border-radius: 4px; font-size: 0.90rem; line-height: 1.6; color: #1e293b;">
            <div style="font-weight: 800; color: #3730a3; font-size: 1.05rem; margin-bottom: 0.5rem;">
                🪜 The Core Problem: Why Ordinary Regression Fails on Clustered and Repeated Data
            </div>
            Standard Ordinary Least Squares (OLS) regression relies on one non-negotiable rule: 
            <strong>every single observation in your spreadsheet must be completely independent of every other observation</strong>. 
            In healthcare and real-world science, this assumption is broken immediately:<br>
            • <strong>Repeated Patient Measures:</strong> If you measure Patient Smith's blood pressure 4 times across a 12-month drug trial, Smith's Visit 3 measurement is biologically bound to Smith's Visit 1 measurement. They share the same genetics, lifestyle, and disease severity.<br>
            • <strong>Organizational Clustering:</strong> Patients treated by the same surgeon in the same hospital ward experience similar nursing ratios, infection control standards, and surgical techniques.<br><br>
            
            <strong>What Happens If You Ignore Clustering?</strong><br>
            If you run standard OLS or standard logistic regression on repeated measures, the computer mistakenly believes it has far more independent information than it really possesses. 
            Standard errors collapse, confidence intervals become artificially narrow, and <strong>you manufacture false-positive "statistically significant" discoveries ($p < 0.05$)</strong> where none exist in reality.<br><br>

            <strong>The Mixed-Effects Solution: Fixed Effects + Random Effects</strong><br>
            Mixed models cleanly partition your data into two complementary components:
            <ol style="margin-top: 0.35rem; margin-bottom: 0.5rem; padding-left: 1.5rem;">
                <li><strong>Fixed Effects ($eta$):</strong> The average population trajectory (e.g. "On average across all patients, Drug X lowers symptom severity by 1.8 points per year").</li>
                <li><strong>Random Effects ($u_i$):</strong> Individual deviations from that average (e.g. "Patient 12 starts 4 points worse than average at baseline, but improves twice as fast as the average patient").</li>
            </ol>

            <strong>Why Mixed Models Crush Repeated Measures ANOVA:</strong><br>
            In classical Repeated Measures ANOVA, if a patient misses even 1 follow-up checkup, <strong>all of their valid visits are thrown away</strong> (listwise deletion). 
            Linear Mixed Models use maximum likelihood under Missing at Random (MAR) to retain 100% of your participants, recovering massive statistical power while properly handling irregular appointment schedules and decaying correlation over time.
        </div>
        """
        render_html(guide_html)


# =====================================================================
# 2. Navigation Mode Selector
# =====================================================================

def render_mixed_mode_selector() -> Tuple[str, str]:
    """Renders mode controller allowing toggling between:
    - Guided Longitudinal & Multilevel Scenario Wizard
    - Direct Method Explorer
    """
    if "mixed_nav_mode_radio" not in st.session_state:
        st.session_state["mixed_nav_mode_radio"] = "🧭 Guided Longitudinal & Multilevel Scenario Wizard"

    col_m1, col_m2 = st.columns([1.8, 2.2])
    with col_m1:
        nav_mode = st.radio(
            "Select Exploration Pathway:",
            options=["🧭 Guided Longitudinal & Multilevel Scenario Wizard", "🔍 Direct Method Explorer"],
            key="mixed_nav_mode_radio",
            horizontal=True
        )

    selected_method_key = "lmm_random_slopes"

    if nav_mode == "🧭 Guided Longitudinal & Multilevel Scenario Wizard":
        with col_m2:
            scenario_names = {k: f"{v['title']}" for k, v in MIXED_MODELS_SCENARIOS.items()}
            sel_scen_key = st.selectbox(
                "Choose a real-world clinical or organizational scenario:",
                options=list(scenario_names.keys()),
                format_func=lambda k: scenario_names[k],
                key="mixed_scenario_select"
            )
            selected_method_key = MIXED_MODELS_SCENARIOS[sel_scen_key]["method_key"]
            st.caption(f"💡 *Scenario:* {MIXED_MODELS_SCENARIOS[sel_scen_key]['description']}")
    else:
        with col_m2:
            method_options = {k: f"{v['icon']} {v['name']}" for k, v in MIXED_MODELS_METHODS.items()}
            default_idx = 0
            if "mixed_direct_method_key" in st.session_state and st.session_state["mixed_direct_method_key"] in method_options:
                default_idx = list(method_options.keys()).index(st.session_state["mixed_direct_method_key"])

            selected_method_key = st.selectbox(
                "Choose a Longitudinal & Mixed Model method:",
                options=list(method_options.keys()),
                format_func=lambda k: method_options[k],
                index=default_idx,
                key="mixed_direct_method_select"
            )
            st.session_state["mixed_direct_method_key"] = selected_method_key

    return nav_mode, selected_method_key


# =====================================================================
# 3. Interactive Method Simulation Playgrounds
# =====================================================================

def _render_lmm_random_slopes_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 🔍 Interactive Playground: Linear Mixed-Effects Model (Random Intercepts & Slopes)")
    st.markdown(r"""
    Evaluate how **patient baseline heterogeneity** (random intercept $\sigma_{u0}$) and **individual rates of change** 
    (random slope $\sigma_{u1}$) shape clinical trajectories, Intraclass Correlation ($	ext{ICC}$), and Empirical Bayes shrinkage.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_subj = st.slider("Number of Patients (N):", min_value=20, max_value=200, value=60, step=10, key="lmm_n_subj")
        n_vis = st.slider("Visits per Patient:", min_value=3, max_value=8, value=4, step=1, key="lmm_n_vis")
    with c2:
        beta_tx = st.slider("Treatment Main Effect (Baseline):", min_value=-5.0, max_value=2.0, value=-0.5, step=0.5, key="lmm_beta_tx")
        beta_inter = st.slider("Treatment × Time Interaction (Slope Difference):", min_value=-3.0, max_value=1.0, value=-1.2, step=0.1, key="lmm_beta_inter")
    with c3:
        sigma_u0 = st.slider("Random Intercept SD (Patient Baseline Spread):", min_value=1.0, max_value=8.0, value=4.2, step=0.2, key="lmm_sig_u0")
        sigma_u1 = st.slider("Random Slope SD (Patient Trajectory Spread):", min_value=0.2, max_value=3.0, value=0.9, step=0.1, key="lmm_sig_u1")

    res = calculate_lmm_random_slopes_model(
        n_subjects=n_subj,
        n_visits=n_vis,
        beta_0=24.5,
        beta_time=-1.8,
        beta_tx=beta_tx,
        beta_interaction=beta_inter,
        sigma_u0=sigma_u0,
        sigma_u1=sigma_u1,
        rho_01=-0.25,
        sigma_eps=1.8,
        seed=42
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Repeated Records", f"{res['total_observations']:,}")
    with m2:
        st.metric("Intraclass Correlation (ICC)", f"{res['icc']:.3f}", help="Proportion of total baseline variance attributable to between-patient biological differences.")
    with m3:
        st.metric("Treatment × Time Effect", f"{res['beta_interaction']:.2f} pts/yr", f"p = {res['p_interaction']:.4f}")
    with m4:
        st.metric("Empirical Bayes Shrinkage", f"{res['shrinkage_factor']:.3f}", help="Shrinkage factor lambda: values near 1 indicate reliable individual patient random estimates.")

    st.markdown("#### 📋 Simulated Longitudinal Patient Trajectories (First 10 Records)")
    st.dataframe(res["records"][:10], height=240, hide_index=True)


def _render_glmm_binary_count_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 🎯 Interactive Playground: GLMM Binary & Count Repeated Outcomes")
    st.markdown(r"""
    Explore the **non-collapsibility of the odds ratio**: why the subject-specific conditional odds ratio ($	ext{OR}_{	ext{SS}}$) 
    is always systematically larger in magnitude than the population-averaged marginal odds ratio ($	ext{OR}_{	ext{marginal}}$).
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_clusters = st.slider("Number of Hospital Wards / Clusters:", min_value=10, max_value=80, value=40, step=5, key="glmm_n_clust")
        clust_sz = st.slider("Patients per Cluster:", min_value=5, max_value=30, value=15, step=5, key="glmm_clust_sz")
    with c2:
        beta_tx = st.slider("Log-Odds Treatment Effect (β_tx):", min_value=-2.5, max_value=-0.1, value=-0.95, step=0.05, key="glmm_beta_tx")
        outcome_t = st.selectbox("Outcome Framework:", options=["binary", "count"], format_func=lambda x: "Binary Relapse Event (0/1)" if x == "binary" else "Repeated Infection Count (Poisson)", key="glmm_out_type")
    with c3:
        sigma_u = st.slider("Ward Frailty SD (Cluster Random Effect σ_u):", min_value=0.2, max_value=2.0, value=0.85, step=0.05, key="glmm_sig_u")

    res = calculate_glmm_binary_count_model(
        n_clusters=n_clusters,
        cluster_size=clust_sz,
        beta_0=-0.8,
        beta_tx=beta_tx,
        sigma_u=sigma_u,
        outcome_type=outcome_t,
        seed=42
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Subject-Specific OR", f"{res['or_subject_specific']:.3f}", help="Odds ratio for a single patient holding their personal biological frailty constant.")
    with m2:
        st.metric("Marginal (Pop-Average) OR", f"{res['or_marginal']:.3f}", help="Odds ratio averaged across random patients from the wider community.")
    with m3:
        st.metric("Non-Collapsibility Gap", f"-{res['attenuation_pct']:.1f}%", help="Percentage by which population-level odds ratio is attenuated towards 1 due to non-linearity.")
    with m4:
        st.metric("Cluster Random Variance (σ_u²)", f"{res['sigma_u_sq']:.3f}", f"z = {res['z_stat']:.2f}")

    st.markdown("#### 📋 Simulated Clustered Patient Records (First 10 Records)")
    st.dataframe(res["records"][:10], height=240, hide_index=True)


def _render_rm_anova_vs_lmm_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### ⚖️ Interactive Playground: Repeated Measures ANOVA vs. Mixed Models")
    st.markdown(r"""
    Observe what happens when **Mauchly's sphericity assumption fails** and participants miss visits. 
    Compare the devastating statistical power loss of **RM-ANOVA listwise deletion** against **LMM likelihood retention**.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_pat = st.slider("Total Enrolled Patients (N):", min_value=20, max_value=120, value=60, step=10, key="rm_n_pat")
        n_tp = st.slider("Timepoints / Visits (k):", min_value=3, max_value=6, value=4, step=1, key="rm_n_tp")
    with c2:
        mis_rate = st.slider("Missed Appointment Visit Rate:", min_value=0.05, max_value=0.40, value=0.20, step=0.05, key="rm_mis_rate")
    with c3:
        sphericity_violation = st.checkbox("Simulate Sphericity Violation (Autoregressive Error Decay)", value=True, key="rm_spher_viol")

    res = calculate_rm_anova_vs_lmm_model(
        n_patients=n_pat,
        n_timepoints=n_tp,
        missing_rate=mis_rate,
        sphericity_violation=sphericity_violation,
        seed=42
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Mauchly's W", f"{res['w_mauchly']:.3f}", f"p = {res['p_mauchly']:.4f}")
    with m2:
        st.metric("Greenhouse-Geisser ε", f"{res['gg_epsilon']:.3f}", "Severe" if res['gg_epsilon'] < 0.75 else "Moderate")
    with m3:
        st.metric("RM-ANOVA Discarded Patients", f"{res['dropped_patients']} / {res['n_patients']}", f"-{res['pct_dropped_rm_anova']:.1f}% Sample Loss")
    with m4:
        st.metric("LMM Retained Patients", f"{res['lmm_retained_patients']} (100%)", f"{res['lmm_observed_visits']} Valid Visits Kept")

    if res["sphericity_rejected"]:
        st.warning("⚠️ **Mauchly's Sphericity Rejected (p < 0.05):** Standard unadjusted RM-ANOVA produces inflated Type I errors. Greenhouse-Geisser degrees of freedom penalty is required.")
    else:
        st.success("🟢 **Sphericity Maintained:** Unadjusted RM-ANOVA degrees of freedom are valid.")

    st.markdown("#### 📋 Simulated Longitudinal Patient Records & Exclusion Audits (First 10 Records)")
    st.dataframe(res["records"][:10], height=240, hide_index=True)


def _render_hierarchical_3level_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 🏥 Interactive Playground: 3-Level Hierarchical Clustered Models")
    st.markdown(r"""
    Simulate a multi-tiered NHS healthcare system: **Patients** nested within **Surgeons** nested within **Hospital Trusts**. 
    Decompose total outcome variance across all three organizational levels.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_hosp = st.slider("Hospital Trusts (Level 3):", min_value=10, max_value=40, value=20, step=5, key="h3_n_hosp")
        clin_per_h = st.slider("Surgeons per Hospital (Level 2):", min_value=2, max_value=8, value=5, step=1, key="h3_clin_per_h")
    with c2:
        pat_per_c = st.slider("Patients per Surgeon (Level 1):", min_value=5, max_value=25, value=15, step=5, key="h3_pat_per_c")
        sig_hosp = st.slider("Hospital Trust SD (σ_hosp):", min_value=0.5, max_value=3.0, value=1.4, step=0.1, key="h3_sig_hosp")
    with c3:
        sig_clin = st.slider("Surgeon SD (σ_clin):", min_value=0.3, max_value=2.0, value=0.9, step=0.1, key="h3_sig_clin")
        sig_err = st.slider("Patient Residual SD (σ_error):", min_value=1.0, max_value=4.0, value=2.2, step=0.2, key="h3_sig_err")

    res = calculate_hierarchical_3level_model(
        n_hospitals=n_hosp,
        clinicians_per_hosp=clin_per_h,
        patients_per_clin=pat_per_c,
        sigma_hosp=sig_hosp,
        sigma_clin=sig_clin,
        sigma_error=sig_err,
        seed=42
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Patients Audited", f"{res['n_patients']:,}", f"{res['n_clinicians']} Clinicians / {res['n_hospitals']} Trusts")
    with m2:
        st.metric("Hospital-Level VPC", f"{res['vpc_hosp']:.1f}%", help="Proportion of recovery variation attributable to hospital trust policies.")
    with m3:
        st.metric("Surgeon-Level VPC", f"{res['vpc_clin']:.1f}%", help="Proportion of recovery variation attributable to individual surgical skill.")
    with m4:
        st.metric("Patient Biology VPC", f"{res['vpc_error']:.1f}%", help="Unexplained variation driven by patient biology and unmeasured health traits.")

    st.markdown("#### 📋 Simulated 3-Level Hierarchical Dataset (First 10 Records)")
    st.dataframe(res["records"][:10], height=240, hide_index=True)


def _render_covariance_selection_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 📐 Interactive Playground: Covariance Structure Selection & Nakagawa R²")
    st.markdown(r"""
    Compare competing within-subject residual covariance structures: **Independent (VC)**, **Compound Symmetry (CS)**, 
    **First-Order Autoregressive AR(1)**, and **Unstructured (UN)** using AIC, BIC, and **Nakagawa & Schielzeth (2013) R²**.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_subj = st.slider("Number of Monitored Patients:", min_value=20, max_value=100, value=50, step=10, key="cov_n_subj")
        n_tp = st.slider("Serial Biomarker Draws (Timepoints):", min_value=3, max_value=6, value=4, step=1, key="cov_n_tp")
    with c2:
        rho_ar1 = st.slider("Serial Lag Correlation (AR1 ρ):", min_value=0.20, max_value=0.90, value=0.65, step=0.05, key="cov_rho_ar1")
    with c3:
        sig_sq = st.slider("Total Residual Variance (σ²):", min_value=1.0, max_value=10.0, value=4.5, step=0.5, key="cov_sig_sq")

    res = calculate_covariance_selection_model(
        n_subjects=n_subj,
        n_timepoints=n_tp,
        rho_ar1=rho_ar1,
        sigma_sq=sig_sq,
        seed=42
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Best Model by AIC", res["best_model_by_aic"])
    with m2:
        st.metric("ΔAIC (CS vs AR1)", f"+{res['delta_aic_cs_vs_ar1']:.1f}", "AR(1) Substantially Superior" if res['delta_aic_cs_vs_ar1'] > 10 else "Modest Difference")
    with m3:
        st.metric("Marginal R² (R²_m)", f"{res['r2_marginal']:.3f}", help="Variance explained purely by the fixed clinical covariates alone.")
    with m4:
        st.metric("Conditional R² (R²_c)", f"{res['r2_conditional']:.3f}", help="Total variance explained by combining fixed clinical covariates with subject random baselines.")

    st.markdown("#### 📊 Covariance Model Comparison Matrix (AIC / BIC Selection)")
    st.dataframe(res["models"], height=240, hide_index=True)


# =====================================================================
# 4. Standard 4-Tab Educational Module Container
# =====================================================================

def render_mixed_data_method_module(method_key: str) -> None:
    """Renders the comprehensive 4-tab educational container for the selected mixed models method."""
    if method_key not in MIXED_MODELS_METHODS:
        st.error(f"Unknown method key: {method_key}")
        return

    meta = MIXED_MODELS_METHODS[method_key]

    st.markdown(f"## {meta['icon']} {meta['name']}")
    st.caption(f"**Pillar 12: Longitudinal & Hierarchical Mixed-Effects Models** • *{meta['category']}*")
    st.info(f"💡 **Method Summary:** {meta['tagline']}")

    # 1. Interactive Simulation Playground
    if method_key == "lmm_random_slopes":
        _render_lmm_random_slopes_sandbox(meta)
    elif method_key == "glmm_binary_count":
        _render_glmm_binary_count_sandbox(meta)
    elif method_key == "rm_anova_vs_lmm":
        _render_rm_anova_vs_lmm_sandbox(meta)
    elif method_key == "hierarchical_3level":
        _render_hierarchical_3level_sandbox(meta)
    elif method_key == "covariance_model_selection":
        _render_covariance_selection_sandbox(meta)

    st.markdown("---")

    # 2. Standard 4-Tab Pedagogical Container
    tab_math, tab_py, tab_xl, tab_data = st.tabs([
        "📋 Mathematical Formulation",
        "🐍 Python Pipeline",
        "📗 Microsoft Excel Guide",
        "💾 Practice CSV Dataset"
    ])

    with tab_math:
        st.markdown(meta["mathematical_formulation"])

        st.markdown("#### 📖 Plain-English Conceptual Glossary")
        for term, explanation in meta["plain_english_glossary"].items():
            st.markdown(f"• **{term}:** {explanation}")

        st.markdown("#### 🛡️ Methodological Assumptions & Quality Criteria")
        for a in meta["assumptions"]:
            st.markdown(f"• {a}")

        col_p, col_c = st.columns(2)
        with col_p:
            st.markdown("##### 🟢 Methodological Strengths (Pros)")
            for pro in meta["pros_cons"]["Pros"]:
                st.markdown(f"• {pro}")
        with col_c:
            st.markdown("##### 🔴 Limitations & Edge Cases (Cons)")
            for con in meta["pros_cons"]["Cons"]:
                st.markdown(f"• {con}")

    with tab_py:
        st.markdown("### 🐍 Production Python Implementation Pipeline")
        st.caption("Publication-grade script utilizing `statsmodels` and `scipy` for rigorous mixed-effects modeling.")

        if method_key == "lmm_random_slopes":
            py_code = """import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. Load longitudinal patient trial data
# Columns: patient_id, treatment_arm, visit_year, adas_cog_score
df = pd.read_csv("longitudinal_alzheimers_trial.csv")

# 2. Fit Linear Mixed-Effects Model (Random Intercepts & Random Slopes)
# Formula: Y ~ time + treatment + time*treatment
# Grouping variable: patient_id
# Random slope on visit_year
model = smf.mixedlm(
    "adas_cog_score ~ visit_year * C(treatment_arm, Treatment('Placebo Standard'))",
    data=df,
    groups=df["patient_id"],
    re_formula="~visit_year"
)
result = model.fit(method=["lbfgs"])

# 3. Print publication-ready summary
print(result.summary())

# 4. Extract Intraclass Correlation (ICC at baseline)
sigma_u0_sq = float(result.cov_re.iloc[0, 0])
sigma_eps_sq = float(result.scale)
icc_baseline = sigma_u0_sq / (sigma_u0_sq + sigma_eps_sq)
print(f"\n[DIAGNOSTIC] Baseline ICC: {icc_baseline:.3f}")

# 5. Extract Empirical Bayes BLUPs (subject-specific random deviations)
blups = result.random_effects
print(f"[DIAGNOSTIC] First 3 Patient Random Devs: {list(blups.items())[:3]}")
"""
        elif method_key == "glmm_binary_count":
            py_code = """import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

# 1. Load clustered binary infection relapse data
# Columns: hospital_ward, treatment_group, relapse_event
df = pd.read_csv("sepsis_icu_relapse_cluster.csv")

# 2. Fit Generalized Estimating Equations (GEE) or Mixed Logistic GLMM
# Using GEE with exchangeable correlation structure as population-averaged model
gee_model = smf.gee(
    "relapse_event ~ C(treatment_group, Treatment('Control Ward'))",
    groups="hospital_ward",
    data=df,
    family=sm.families.Binomial(sm.families.links.Logit()),
    cov_struct=sm.cov_struct.Exchangeable()
)
gee_res = gee_model.fit()
print(gee_res.summary())

# 3. Calculate Marginal Odds Ratio and 95% Confidence Interval
beta_tx = gee_res.params[1]
ci_low, ci_high = gee_res.conf_int().iloc[1]
print(f"\nMarginal OR: {np.exp(beta_tx):.3f} (95% CI: {np.exp(ci_low):.3f} - {np.exp(ci_high):.3f})")
"""
        elif method_key == "rm_anova_vs_lmm":
            py_code = """import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pingouin import rm_anova, sphericity

# 1. Load longitudinal pain score dataset
df = pd.read_csv("physiotherapy_pain_recovery.csv")

# 2. Evaluate Sphericity on complete cases
df_clean = df.dropna()
spher_stat, p_spher, eps_gg = sphericity(df_clean, dv='pain_vas_score', within='week_number', subject='patient_id')
print(f"[DIAGNOSTIC] Mauchly Sphericity W p-value: {p_spher:.4f}")
print(f"[DIAGNOSTIC] Greenhouse-Geisser Epsilon: {eps_gg:.3f}")

# 3. Fit Linear Mixed Model (handles all observed visits without dropping patients)
lmm_model = smf.mixedlm("pain_vas_score ~ week_number", data=df, groups=df["patient_id"])
lmm_res = lmm_model.fit()
print(lmm_res.summary())
print(f"\nTotal patients retained in LMM: {df['patient_id'].nunique()} (100% of cohort)")
"""
        elif method_key == "hierarchical_3level":
            py_code = """import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. Load 3-level surgical recovery dataset
# Hierarchical nesting: patient -> surgeon -> hospital trust
df = pd.read_csv("nhs_surgical_recovery_3level.csv")

# 2. Fit 2-level or 3-level mixed model
# Using statsmodels mixedlm with hospital clustering
model = smf.mixedlm(
    "recovery_days ~ patient_age + charlson_comorbidity",
    data=df,
    groups=df["hospital_id"]
)
res = model.fit()
print(res.summary())

# 3. Decompose Variance Components
sigma_hosp_sq = float(res.cov_re.iloc[0, 0])
sigma_eps_sq = float(res.scale)
total_var = sigma_hosp_sq + sigma_eps_sq
vpc_hosp = (sigma_hosp_sq / total_var) * 100.0
print(f"\n[VPC AUDIT] Hospital-Level Variance Partition: {vpc_hosp:.2f}%")
"""
        else:
            py_code = """import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. Load serial biomarker decay dataset
df = pd.read_csv("cardiac_troponin_serial.csv")

# 2. Fit Linear Mixed Model under Compound Symmetry
model_cs = smf.mixedlm("troponin_t_ng_l ~ hours_post_mi", data=df, groups=df["patient_id"])
res_cs = model_cs.fit(reml=False)

# 3. Calculate Nakagawa & Schielzeth (2013) R2
var_fixed = np.var(res_cs.fittedvalues)
var_random = float(res_cs.cov_re.iloc[0, 0])
var_resid = float(res_cs.scale)
total_v = var_fixed + var_random + var_resid

r2_marginal = var_fixed / total_v
r2_conditional = (var_fixed + var_random) / total_v

print(f"Model AIC: {res_cs.aic:.2f}")
print(f"Nakagawa Marginal R2 (Fixed Effects): {r2_marginal:.4f}")
print(f"Nakagawa Conditional R2 (Full Model): {r2_conditional:.4f}")
"""
        st.code(py_code, language="python")

    with tab_xl:
        st.markdown("### 📗 Step-by-Step Microsoft Excel Implementation Guide")
        st.markdown(r"""
        This guide provides a structured 3-expander workbook architecture for auditing repeated measures 
        and multilevel clustered data in Microsoft Excel.
        """)

        with st.expander("Step 1: Data Structuring & Baseline Repeated Measures Diagnostics", expanded=False):
            st.markdown(r"""
            1. **Long Format vs. Wide Format:** Arrange patient visits in stacked longitudinal format (`Col A: Patient_ID`, `Col B: Visit_Number`, `Col C: Time_Months`, `Col D: Treatment_Arm`, `Col E: Score`).
            2. **Compute Per-Patient Baseline Means:** Use `=AVERAGEIFS(E:E, A:A, A2)` in `Col F` to calculate each patient's mean score across their observed visits.
            3. **Compute Grand Mean:** In cell `H2`, enter `=AVERAGE(E2:E241)`.
            4. **Audit Missingness Pattern:** In cell `H3`, enter `=COUNTBLANK(E2:E241) / COUNTA(A2:A241)` to determine the cohort attrition rate.
            """)

        with st.expander("Step 2: Primary Mixed Model & Random Effects Variance Decomposition", expanded=False):
            st.markdown(r"""
            1. **Between-Patient Sum of Squares ($SS_{	ext{between}}$):**
               In cell `H5`, calculate patient variability: `=SUMPRODUCT((F2:F241 - H2)^2)`.
            2. **Within-Patient Residual Sum of Squares ($SS_{	ext{within}}$):**
               In cell `H6`, calculate measurement noise: `=SUMPRODUCT((E2:E241 - F2:F241)^2)`.
            3. **Variance Components:**
               * $\sigma_{	ext{patient}}^2$: `=(H5 / (N_patients - 1) - H6 / (N_patients * (k - 1))) / k`
               * $\sigma_{\epsilon}^2$: `=H6 / (N_patients * (k - 1))`
            4. **Intraclass Correlation (ICC):**
               In cell `H8`, enter `=sigma_patient_sq / (sigma_patient_sq + sigma_eps_sq)`.
            """)

        with st.expander("Step 3: Advanced Covariance / Sphericity Sensitivity & ICC Audits", expanded=False):
            st.markdown(r"""
            1. **Pairwise Difference Variances (Sphericity Check):**
               Create difference columns for `Visit 2 - Visit 1`, `Visit 3 - Visit 1`, `Visit 4 - Visit 1`.
               Compute `=VAR.S(...)` for each column. If the variances differ by $> 25\%$, sphericity is violated.
            2. **Greenhouse-Geisser Correction ($\hat{\epsilon}$):**
               Compute the correlation matrix across visits using Excel Data Analysis Toolpak. Enter the GG formula to discount ANOVA degrees of freedom.
            3. **Decision Rule:**
               * If missing data $> 5\%$ or sphericity $p < 0.05$, **cease classical RM-ANOVA** and report Linear Mixed Model results.
            """)

        st.markdown("#### 📋 Microsoft Excel Spreadsheet Preview Layout (`A1:G15`)")
        if method_key == "lmm_random_slopes":
            sample_df = generate_lmm_dataset(10, 4)
        elif method_key == "glmm_binary_count":
            sample_df = generate_glmm_dataset(5, 10)
        elif method_key == "rm_anova_vs_lmm":
            sample_df = generate_rm_anova_dataset(10, 4)
        elif method_key == "hierarchical_3level":
            sample_df = generate_3level_dataset(2, 2, 5)
        else:
            sample_df = generate_covariance_dataset(10, 4)

        st.dataframe(sample_df[:15], height=240, hide_index=True)

    with tab_data:
        st.markdown("### 💾 Practice Clinical CSV Dataset Download")
        st.markdown(r"""
        Download the fully populated synthetic dataset for this method to test your Python scripts 
        or verify formulas in Microsoft Excel.
        """)

        if method_key == "lmm_random_slopes":
            export_records = generate_lmm_dataset(60, 4)
            file_name = "longitudinal_adas_cog_trial.csv"
        elif method_key == "glmm_binary_count":
            export_records = generate_glmm_dataset(40, 15)
            file_name = "glmm_sepsis_icu_clusters.csv"
        elif method_key == "rm_anova_vs_lmm":
            export_records = generate_rm_anova_dataset(50, 4)
            file_name = "physiotherapy_rm_anova_pain.csv"
        elif method_key == "hierarchical_3level":
            export_records = generate_3level_dataset(20, 5, 15)
            file_name = "nhs_3level_surgical_audit.csv"
        else:
            export_records = generate_covariance_dataset(50, 4)
            file_name = "serial_cardiac_troponin_decay.csv"

        st.dataframe(export_records[:20], height=240, hide_index=True)

        # Convert records to CSV string
        if export_records:
            header = list(export_records[0].keys())
            csv_lines = [",".join(header)]
            for r in export_records:
                csv_lines.append(",".join(str(r.get(h, "")) if r.get(h) is not None else "" for h in header))
            csv_data = "\n".join(csv_lines)

            st.download_button(
                label=f"📥 Download {file_name} ({len(export_records)} rows)",
                data=csv_data,
                file_name=file_name,
                mime="text/csv",
                key=f"btn_download_mixed_{method_key}"
            )

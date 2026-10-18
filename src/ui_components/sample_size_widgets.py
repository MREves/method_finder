"""Interactive UI Components & Visualizations for Sample Size & Statistical Power Calculations.

Provides:
1. Plain English Pedagogical Introductory Guide to Sample Size, Statistical Power (1 - beta),
   Type I (alpha) and Type II (beta) errors, Minimal Clinically Important Differences (MCID),
   Allocation Ratios, Dropout Inflation, and Cluster Randomization Design Effects (Zero prior knowledge).
2. On-page and sidebar scenario controls featuring 3 realistic clinical scenarios:
   - All Meadows Cardiovascular Institute (SGLT2 Phase III RCT: HbA1c reduction & MACE prevention).
   - Riverside Health Protection Team (Outbreak Case-Control 1:2 & School Health Cluster Trial).
   - Inner City Teaching Hospital (NSCLC Survival Trial HR=0.70 & Diagnostic Accuracy Precision).
   - Custom Mode with Guided Study Design Selector covering 10 distinct biostatistical methods.
3. Methodological Prerequisite Evaluation (5 ICH-E9 / CONSORT / STROBE standards: Power sufficiency,
   Alpha level rigor, MCID plausibility, Attrition protection, and Cluster unit of analysis).
4. Primary Metric Dashboard (Required Evaluable N, Recruited Target N, Target vs. Achieved Power,
   Standardized Effect Size, Required Events / Clusters).
5. Dual Plotly Interactive Visualizations:
   - Plotly Power Curve (Power vs. N across Target, Small, and Large effect benchmarks).
   - Plotly Minimum Detectable Effect Size (MDES vs. N sensitivity curve).
6. Comprehensive Sensitivity Analysis Grid (Power across sample size and effect variations).
7. Automated Protocol Justification Statement Card (Formatted for NIHR / MRC / Ethics submissions).
8. Full 4-Tab Educational Architecture:
   - 📋 Rationale & Epidemiological Framework (Mathematical derivations, term-by-term plain English explanations).
   - 📗 Excel Implementation Guide (Cell coordinate mapping A1:P20, collapsible expanders for every step, formulas, sensitivity tables).
   - 🐍 Python Implementation Guide (Collapsible expanders for each stage, scipy/statsmodels pipelines, standalone script).
   - 💾 Data Downloads (Protocol planning workbook CSV, synthetic cohort microdata CSV, variable dictionary).
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Union
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.stats_calcs.sample_size_engine import (
    calculate_sample_size_two_means,
    calculate_sample_size_paired_means,
    calculate_sample_size_anova,
    calculate_sample_size_correlation,
    calculate_sample_size_two_proportions,
    calculate_sample_size_cohort_rr,
    calculate_sample_size_case_control_or,
    calculate_sample_size_paired_proportions_mcnemar,
    calculate_sample_size_survival_logrank,
    calculate_sample_size_diagnostic_accuracy,
    apply_attrition_inflation,
    calculate_unpadded_power_after_dropout,
    apply_cluster_design_effect,
    generate_power_curve_data,
    generate_protocol_justification_statement,
    evaluate_sample_size_prerequisites,
    generate_sample_size_demo_data
)


# =====================================================================
# 1. Plain English Pedagogical Introductory Guide
# =====================================================================

def render_sample_size_plain_english_guide():
    """Renders a collapsible plain English introductory guide to sample size and power."""
    with st.expander("📖 Zero-Knowledge Guide: Fundamentals of Sample Size, Statistical Power & Study Design", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. The High-Stakes Dilemma: Why Sample Size Planning is Mandatory
            * **The Danger of Underpowered Studies:** If a clinical trial recruits too few participants (e.g. 30 patients when 200 were required), it lacks the statistical precision to detect even a lifesaving treatment effect. The trial concludes "no statistically significant difference" ($p > 0.05$), abandoning a viable therapy, wasting public research funding, and subjecting volunteers to invasive procedures for zero scientific gain.
            * **The Danger of Over-Recruitment:** Conversely, enrolling 5,000 patients when 500 would suffice is ethically unacceptable: it needlessly exposes thousands of participants to potential experimental side effects or inferior placebo arms, while inflating healthcare research budgets by millions of pounds.
            * **The Biostatistical Objective:** Sample size calculation determines the **exact minimum number of evaluable participants** necessary to test a scientific hypothesis with predetermined mathematical rigor.
            """)

            st.markdown(r"""
            #### 2. The Four Pillars of Every Power Calculation
            Every classical sample size calculation balances four interconnected mathematical quantities:
            * **1. Significance Level ($\alpha$, Type I Error Rate):**
              * The risk of a **False Positive** finding: claiming the treatment works when in reality there is zero true benefit.
              * By universal regulatory convention (ICH-E9, FDA, EMA), $\alpha$ is fixed at **$0.05$ (5%)**, typically evaluated two-sided ($z_{1-\alpha/2} = 1.96$).
            * **2. Statistical Power ($1 - \beta$, Complement of Type II Error):**
              * The probability of a **True Positive** finding: successfully detecting a true treatment effect if it genuinely exists in the population.
              * $\beta$ is the False Negative rate. Modern medical research standards mandate Power of at least **$80\%$ ($\beta = 0.20, z_{1-\beta} = 0.84$)** or **$90\%$ ($\beta = 0.10, z_{1-\beta} = 1.28$)** for confirmatory Phase III trials.
            * **3. Minimal Clinically Important Difference (MCID, $\Delta$):**
              * The smallest difference in outcome that clinicians, guideline panels, and patients would consider meaningful in medical practice (e.g. a $0.5\%$ reduction in HbA1c or a $15\%$ drop in mortality).
            * **4. Population Variability / Background Noise ($\sigma$):**
              * The standard deviation of the outcome. Higher physiological variation requires substantially larger sample sizes to uncover the underlying signal.
            """)

            st.markdown(r"""
            #### 3. Cohen's Effect Size Benchmarks ($d, f, r$)
            When historical pilot data are scarce, biostatisticians standardise differences using Cohen's benchmarks:
            * **Cohen's $d$ (Two Means):** $d = |\mu_1 - \mu_2| / \sigma_{\text{pooled}}$
              * Small: $d = 0.20$ (Requires $\approx 786$ total patients for 80% power).
              * Medium: $d = 0.50$ (Requires $\approx 128$ total patients for 80% power).
              * Large: $d = 0.80$ (Requires $\approx 52$ total patients for 80% power).
            * **Cohen's $f$ (One-Way ANOVA):** Small ($0.10$), Medium ($0.25$), Large ($0.40$).
            * **Pearson's $r$ (Correlation):** Small ($0.10$), Medium ($0.30$), Large ($0.50$).
            """)

        with col_g2:
            st.markdown(r"""
            #### 4. Allocation Ratios ($k = n_2 / n_1$): Statistical vs. Clinical Efficiency
            * **Equal Allocation ($1:1$, $k = 1.0$):**
              * Mathematically minimizes the total required sample size ($N_{\text{total}}$) for a given statistical power.
            * **Unequal Allocation ($1:2$ or $1:3$, $k > 1.0$):**
              * In rare diseases, paediatric medicine, or surgical oncology, assigning more patients to the novel experimental arm or matching multiple healthy controls per case in case-control studies increases clinical acceptability or statistical power without needing more rare cases.
              * *Statistical Penalty:* A $1:2$ ratio increases total required enrollment by approximately $12.5\%$, but only requires two-thirds as many experimental patients.
            """)

            st.markdown(r"""
            #### 5. Real-World Pragmatics: Attrition & Loss-to-Follow-Up
            * Calculations yield the number of **evaluable** participants required at final analysis ($N_{\text{evaluable}}$).
            * In clinical practice, participants relocate, withdraw consent, develop adverse drug reactions, or miss follow-up appointments.
            * To safeguard the trial's statistical power under Intention-to-Treat (ITT) principles, the initial recruitment target ($N_{\text{recruited}}$) must be inflated using the non-linear formula:
              $$N_{\text{recruited}} = \left\lceil \frac{N_{\text{evaluable}}}{1 - d} \right\rceil$$
              where $d$ is the anticipated dropout proportion (e.g. $15\%$). Simply adding $15\%$ to $N_{\text{evaluable}}$ underestimates required enrollment!
            """)

            st.markdown(r"""
            #### 6. Cluster Randomization & The Design Effect (VIF)
            * When treatments are delivered at a group level (e.g. training whole GP practices, hospital wards, or schools), individuals within the same cluster respond more similarly than individuals from different clusters.
            * **Intracluster Correlation Coefficient (ICC, $\rho$):** Measures this within-cluster clustering ($0.01$ to $0.08$ in primary care).
            * **Variance Inflation Factor (VIF / Design Effect):**
              $$\text{VIF} = 1 + (m - 1) \times \text{ICC}$$
              where $m$ is the average cluster size.
            * *Real-World Impact:* If an average GP practice enrolls $m = 30$ patients with an $\text{ICC} = 0.04$, $\text{VIF} = 1 + 29 \times 0.04 = 2.16$. The required sample size **more than doubles** ($N_{\text{clustered}} = N_{\text{individual}} \times 2.16$)!
            """)

            st.markdown(r"""
            #### 7. International Regulatory Standards (ICH-E9 & CONSORT)
            * All modern health research grant proposals (NIHR, MRC, Wellcome Trust, NIH) and ethics committee submissions mandate an explicit, reproducible sample size justification statement.
            * Under **ICH-E9 (Statistical Principles for Clinical Trials)** and **CONSORT 2010 (Item 7a)**, researchers must state: primary outcome, effect size, variance assumptions, $\alpha$, power, allocation ratio, expected dropout, and software used.
            """)


# =====================================================================
# 2. Methodological & Protocol Prerequisite Evaluation Cards
# =====================================================================

def render_sample_size_prerequisite_cards(evaluations: Dict[str, Any]):
    """Renders 5 core study planning prerequisite cards based on ICH-E9 and CONSORT standards."""
    checks = evaluations.get("checks", [])
    if not checks:
        return

    st.markdown("### 🔍 Methodological & Study Protocol Prerequisite Checks")
    st.caption("Verifying statistical power adequacy, significance level control, effect size plausibility, attrition inflation, and cluster unit of analysis under ICH-E9 and CONSORT 2010 guidelines.")

    cols = st.columns(len(checks))
    for i, check in enumerate(checks):
        with cols[i]:
            status = check.get("status", "pass")
            badge_label = check.get("badge", "")
            title = check.get("title", "")
            finding = check.get("finding", "")
            explanation = check.get("explanation", "")

            with st.container(border=True):
                if status == "pass":
                    st.success(f"**{badge_label}**")
                elif status == "warn":
                    st.warning(f"**{badge_label}**")
                else:
                    st.error(f"**{badge_label}**")

                st.markdown(f"**{title}**")
                st.caption(finding)
                with st.expander("Methodological Details", expanded=False):
                    st.write(explanation)


# =====================================================================
# 3. Primary Metric Dashboard
# =====================================================================

def render_sample_size_metric_dashboard(
    calc_results: Dict[str, Any],
    attrition: Dict[str, Any],
    cluster: Optional[Dict[str, Any]] = None,
    unpadded_power_info: Optional[Dict[str, Any]] = None
):
    """Renders top-level biostatistical summary metric cards."""
    c1, c2, c3, c4, c5 = st.columns(5)

    n_eval_indiv = calc_results.get("n_total", calc_results.get("required_total_screening_cohort", calc_results.get("n_matched_pairs", 100)))
    n1 = calc_results.get("n1", calc_results.get("n_cases", calc_results.get("n_unexposed", n_eval_indiv // 2)))
    n2 = calc_results.get("n2", calc_results.get("n_controls", calc_results.get("n_exposed", n_eval_indiv // 2)))

    is_cluster = cluster is not None and cluster.get("icc", 0.0) > 0
    if is_cluster:
        n_eval = cluster.get("n_clustered_total", n_eval_indiv)
        k_clust = cluster.get("clusters_required", 0)
        m_size = cluster.get("cluster_size_m", cluster.get("cluster_size", 30))
        vif = cluster.get("design_effect_vif", 1.0)
        rec_clust = cluster.get("clusters_required_recruited", k_clust)
    else:
        n_eval = n_eval_indiv
        k_clust = m_size = vif = rec_clust = None

    n_rec = attrition.get("n_recruited", n_eval)
    drop_pct = attrition.get("dropout_rate_pct", 0.0)
    extra_pts = attrition.get("additional_subjects_needed", 0)
    inflation_factor = attrition.get("inflation_factor", 1.0)

    achieved_pwr = calc_results.get("achieved_power", calc_results.get("target_power", 0.80)) * 100.0
    alpha = calc_results.get("alpha", 0.05)

    # Degraded power calculation if study does not inflate for dropout
    unpad_pwr = unpadded_power_info.get("degraded_power_pct", achieved_pwr) if unpadded_power_info else achieved_pwr
    power_loss = unpadded_power_info.get("power_loss_pct", 0.0) if unpadded_power_info else 0.0

    # Standardized Effect Size formatting
    if "cohen_d" in calc_results:
        eff_label = "Cohen's d (Std Diff)"
        eff_val = f"{calc_results['cohen_d']:.2f}"
        eff_delta = calc_results.get("cohen_tier", "Continuous Effect")
    elif "hazard_ratio" in calc_results:
        eff_label = "Target Hazard Ratio"
        eff_val = f"{calc_results['hazard_ratio']:.2f}"
        eff_delta = f"ln(HR) = {calc_results.get('ln_hazard_ratio', 0.0):.3f}"
    elif "odds_ratio" in calc_results:
        eff_label = "Target Odds Ratio"
        eff_val = f"{calc_results['odds_ratio']:.2f}"
        eff_delta = f"Exposure p0 = {calc_results.get('p0_control_exposure', 0.2)*100:.0f}%"
    elif "relative_risk" in calc_results:
        eff_label = "Target Relative Risk"
        eff_val = f"{calc_results['relative_risk']:.2f}"
        eff_delta = f"Risk Diff = {calc_results.get('risk_difference', 0.0)*100:.1f}%"
    elif "target_sensitivity_pct" in calc_results:
        eff_label = "Target Sensitivity"
        eff_val = f"{calc_results['target_sensitivity_pct']:.1f}%"
        eff_delta = f"Margin W = +/-{calc_results.get('margin_of_error_pct', 5.0):.1f}%"
    elif "cohen_f" in calc_results:
        eff_label = "Cohen's f (ANOVA)"
        eff_val = f"{calc_results['cohen_f']:.2f}"
        eff_delta = f"{calc_results.get('num_groups', 3)} Groups"
    elif "r" in calc_results:
        eff_label = "Pearson Correlation (r)"
        eff_val = f"{calc_results['r']:.2f}"
        eff_delta = f"Fisher z = {calc_results.get('fisher_z', 0.0):.3f}"
    else:
        eff_label = "Target Effect Size"
        eff_val = "Specified"
        eff_delta = "Primary Hypothesis"

    # Contextual metric 5 (Events, Clusters, or Diseased Cases)
    if "required_events" in calc_results:
        m5_label = "Required Events (E)"
        m5_val = f"{calc_results['required_events']} Events"
        m5_delta = f"Event Prob P(E) = {calc_results.get('cumulative_event_prob', 0.5)*100:.0f}%"
        m5_sub = "Schoenfeld (1981) log-rank formulation"
    elif is_cluster:
        m5_label = "Cluster Design Effect (VIF)"
        m5_val = f"VIF = {vif:.2f}"
        m5_delta = f"ICC = {cluster.get('icc', 0.0):.3f} | Size m = {m_size}"
        m5_sub = f"Inflates {n_eval_indiv:,} individual N to {n_eval:,} clustered N"
    elif "required_diseased_cases" in calc_results:
        m5_label = "Required Confirmed Cases"
        m5_val = f"{calc_results['required_diseased_cases']} Cases"
        m5_delta = f"At {calc_results.get('disease_prevalence_pct', 10.0):.1f}% prevalence"
        m5_sub = "Wilson score precision formulation"
    else:
        m5_label = "Allocation Structure"
        alloc_ratio = calc_results.get("allocation_ratio", 1.0)
        m5_val = f"Ratio k = {alloc_ratio:.1f}"
        m5_delta = f"Arm 1: {n1} | Arm 2: {n2}"
        m5_sub = "Equal 1:1" if alloc_ratio == 1.0 else f"Unequal 1:{alloc_ratio:.1f}"

    with c1:
        with st.container(border=True):
            if is_cluster:
                st.metric(
                    label="Evaluable Target (Clustered)",
                    value=f"{n_eval:,}",
                    delta=f"{k_clust} Clusters of size {m_size}"
                )
                st.caption(f"Unclustered individual N = {n_eval_indiv:,} (VIF = {vif:.2f})")
            else:
                st.metric(
                    label="Evaluable Target (N)",
                    value=f"{n_eval:,}",
                    delta=f"Arm 1: {n1} | Arm 2: {n2}" if n1 != n_eval else f"{n_eval} Observations"
                )
                st.caption("Final evaluable participants required at primary endpoint")

    with c2:
        with st.container(border=True):
            if drop_pct > 0:
                cushion_str = f"+{extra_pts} Cushion ({inflation_factor:.2f}x)" if not is_cluster else f"+{extra_pts} Cushion ({rec_clust} Clusters)"
                st.metric(
                    label=f"Recruitment Target (+{drop_pct:.0f}% Drop)",
                    value=f"{n_rec:,}",
                    delta=cushion_str
                )
                st.caption(f"Protects statistical power against {drop_pct:.0f}% loss-to-follow-up")
            else:
                st.metric(
                    label="Recruitment Target (0% Drop)",
                    value=f"{n_rec:,}",
                    delta="No Cushion Needed (+0)"
                )
                st.caption(f"Matches evaluable target ({n_eval:,})")

    with c3:
        with st.container(border=True):
            if drop_pct > 0 and power_loss > 0:
                pwr_delta = f"Drops to {unpad_pwr:.1f}% if unpadded (-{power_loss:.1f}%)"
                pwr_delta_col = "inverse"
                pwr_caption = "Guaranteed with cushion"
            else:
                pwr_delta = f"Type I α = {alpha} (2-sided)"
                pwr_delta_col = "normal"
                pwr_caption = f"β = {100.0 - achieved_pwr:.1f}% risk of false-negative inference"

            st.metric(
                label="Statistical Power (1 - β)",
                value=f"{achieved_pwr:.1f}%",
                delta=pwr_delta,
                delta_color=pwr_delta_col
            )
            st.caption(pwr_caption)

    with c4:
        with st.container(border=True):
            st.metric(
                label=eff_label,
                value=eff_val,
                delta=eff_delta
            )
            st.caption("Hypothesized Minimal Clinically Important Difference")

    with c5:
        with st.container(border=True):
            st.metric(
                label=m5_label,
                value=m5_val,
                delta=m5_delta
            )
            st.caption(m5_sub)

    # Contextual Recommendation Text
    if is_cluster:
        if drop_pct == 0:
            rec_body = (
                f"To achieve <strong>{achieved_pwr:.1f}% statistical power</strong> at a two-sided significance level of <strong>α = {alpha}</strong>, "
                f"an individual unclustered design would require <strong>{n_eval_indiv:,} participants</strong> ({n1:,} in Group 1, {n2:,} in Group 2). "
                f"Accounting for cluster randomization across schools / GP practices (average cluster size <em>m</em> = {m_size}, "
                f"ICC = {cluster.get('icc', 0):.3f}, Design Effect VIF = {vif:.2f}), the trial requires a clustered evaluable cohort of "
                f"<strong>{n_eval:,} participants across {k_clust} clusters</strong>. "
                f"With <strong>0% anticipated loss-to-follow-up</strong>, the recruitment target remains <strong>{n_eval:,} enrolled participants across {k_clust} clusters</strong> "
                f"(no attrition cushion required)."
            )
        else:
            rec_body = (
                f"To achieve <strong>{achieved_pwr:.1f}% statistical power</strong> at a two-sided significance level of <strong>α = {alpha}</strong>, "
                f"an individual unclustered design would require <strong>{n_eval_indiv:,} participants</strong> ({n1:,} in Group 1, {n2:,} in Group 2). "
                f"Accounting for cluster randomization across schools / GP practices (average cluster size <em>m</em> = {m_size}, "
                f"ICC = {cluster.get('icc', 0):.3f}, Design Effect VIF = {vif:.2f}), the trial requires a clustered evaluable cohort of "
                f"<strong>{n_eval:,} participants across {k_clust} clusters</strong>. "
                f"Accounting for the selected <strong>{drop_pct:.0f}% loss-to-follow-up</strong>, the recruitment target is inflated to "
                f"<strong>{n_rec:,} enrolled participants across {rec_clust} clusters</strong> (+{extra_pts} recruitment cushion)."
            )
    else:
        if drop_pct == 0:
            rec_body = (
                f"To achieve <strong>{achieved_pwr:.1f}% statistical power</strong> at a two-sided significance level of <strong>α = {alpha}</strong>, "
                f"the clinical trial must retain exactly <strong>{n_eval:,} evaluable participants</strong> ({n1:,} in Group 1, {n2:,} in Group 2). "
                f"With <strong>0% anticipated loss-to-follow-up</strong>, the recruitment target remains <strong>{n_eval:,} enrolled participants</strong> "
                f"(no attrition cushion required)."
            )
        else:
            rec_body = (
                f"To achieve <strong>{achieved_pwr:.1f}% statistical power</strong> at a two-sided significance level of <strong>α = {alpha}</strong>, "
                f"the clinical trial must retain exactly <strong>{n_eval:,} evaluable participants</strong> ({n1:,} in Group 1, {n2:,} in Group 2). "
                f"Accounting for the selected <strong>{drop_pct:.0f}% loss-to-follow-up</strong>, the recruitment target is inflated to "
                f"<strong>{n_rec:,} enrolled participants</strong> (+{extra_pts} recruitment cushion)."
            )

    dropout_warn_html = ""
    if drop_pct > 0 and unpadded_power_info and power_loss > 0:
        eff_n = int(math.floor(n_eval * (1.0 - drop_pct / 100.0)))
        dropout_warn_html = (
            f'<div style="margin-top: 0.75rem; font-size: 0.88rem; color: #92400e; background-color: #fffbeb; '
            f'padding: 0.6rem 0.85rem; border-radius: 4px; border: 1px solid #fde68a;">'
            f'⚠️ <strong>Power Degradation Risk:</strong> If the study only enrolls {n_eval:,} participants without this +{extra_pts} cushion, '
            f'a {drop_pct:.0f}% loss-to-follow-up will reduce evaluable data to {eff_n:,} individuals, causing statistical power to '
            f'<strong>collapse from {achieved_pwr:.1f}% down to {unpad_pwr:.1f}%</strong> (-{power_loss:.1f}% power deficit), '
            f'critically elevating the risk of a false-negative trial.'
            f'</div>'
        )

    banner_html = (
        f'<div style="background-color: #f0fdf4; border-left: 5px solid #10b981; padding: 1rem; border-radius: 6px; margin-top: 1rem; margin-bottom: 1.5rem;">'
        f'<h4 style="margin: 0 0 0.35rem 0; color: #065f46; font-size: 1.05rem;">🎯 Biostatistical Recommendation & Enrollment Target</h4>'
        f'<p style="margin: 0; font-size: 0.92rem; color: #334155;">{rec_body}</p>'
        f'{dropout_warn_html}'
        f'</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)


# =====================================================================
# 4. Interactive Plotly Visualizations
# =====================================================================

def build_power_curve_plot(
    curve_data: List[Dict[str, Any]],
    planned_n: int,
    planned_power: float,
    design_type: str = "two_means",
    recruited_n: Optional[int] = None,
    dropout_pct: float = 0.0,
    effective_n: Optional[int] = None,
    unpadded_power: Optional[float] = None
) -> go.Figure:
    """Builds an interactive Plotly Power vs. Sample Size (N) curve across multiple effect benchmarks."""
    fig = go.Figure()
    if not curve_data:
        return fig

    ns = [pt["N"] for pt in curve_data]
    p_target = [pt["Power_Target"] for pt in curve_data]
    p_small = [pt["Power_Small"] for pt in curve_data]
    p_large = [pt["Power_Large"] for pt in curve_data]

    # Target Planned Effect Curve (Bold Dark Teal)
    fig.add_trace(go.Scatter(
        x=ns,
        y=p_target,
        mode="lines",
        name="Planned Effect (Target Hypothesis)",
        line=dict(color="#0284c7", width=3.0),
        hovertemplate="<b>Sample Size (N)</b>: %{x:,}<br><b>Statistical Power</b>: %{y:.1f}%<extra></extra>"
    ))

    # Small Effect Benchmark (Amber Dotted)
    fig.add_trace(go.Scatter(
        x=ns,
        y=p_small,
        mode="lines",
        name="Small Effect Size (Benchmark)",
        line=dict(color="#d97706", width=2.0, dash="dot"),
        hovertemplate="<b>Sample Size (N)</b>: %{x:,}<br><b>Small Effect Power</b>: %{y:.1f}%<extra></extra>"
    ))

    # Large Effect Benchmark (Emerald Dotted)
    fig.add_trace(go.Scatter(
        x=ns,
        y=p_large,
        mode="lines",
        name="Large Effect Size (Benchmark)",
        line=dict(color="#059669", width=2.0, dash="dash"),
        hovertemplate="<b>Sample Size (N)</b>: %{x:,}<br><b>Large Effect Power</b>: %{y:.1f}%<extra></extra>"
    ))

    # Target Operating Point Marker (Evaluable N)
    planned_pwr_pct = planned_power * 100.0 if planned_power <= 1.0 else planned_power
    fig.add_trace(go.Scatter(
        x=[planned_n],
        y=[planned_pwr_pct],
        mode="markers+text",
        name=f"Evaluable Target (N = {planned_n:,}, Power = {planned_pwr_pct:.0f}%)",
        marker=dict(color="#2563eb", size=12, symbol="diamond", line=dict(color="white", width=2)),
        text=[f" Evaluable (N={planned_n:,})"],
        textposition="top right",
        textfont=dict(color="#2563eb", size=11, family="sans-serif"),
        hovertemplate="<b>Evaluable Target Operating Point</b><br>N = %{x:,}<br>Power = %{y:.1f}%<extra></extra>"
    ))

    # Dynamic Attrition Cushion: Shaded Box, Recruited Marker, and Degraded Power Point
    if recruited_n is not None and recruited_n > planned_n and dropout_pct > 0:
        # Shaded Attrition Buffer Region
        fig.add_vrect(
            x0=planned_n, x1=recruited_n,
            fillcolor="rgba(245, 158, 11, 0.12)",
            layer="below", line_width=0,
            annotation_text=f"Attrition Cushion (+{dropout_pct:.0f}%)",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#b45309")
        )
        # Vertical recruitment target line
        fig.add_shape(
            type="line",
            x0=recruited_n, x1=recruited_n,
            y0=0, y1=100,
            line=dict(color="#f59e0b", width=2.0, dash="dot")
        )
        # Recruited target marker
        fig.add_trace(go.Scatter(
            x=[recruited_n],
            y=[planned_pwr_pct],
            mode="markers+text",
            name=f"Recruitment Target (N = {recruited_n:,}, +{dropout_pct:.0f}% Drop)",
            marker=dict(color="#d97706", size=11, symbol="square", line=dict(color="white", width=2)),
            text=[f" Enrolled Target (N={recruited_n:,})"],
            textposition="top right",
            textfont=dict(color="#d97706", size=10),
            hovertemplate="<b>Enrolled Recruitment Target</b><br>N = %{x:,} (accounts for " + f"{dropout_pct:.0f}% dropout)<extra></extra>"
        ))

    # Degraded Power Risk if study is unpadded
    if unpadded_power is not None and effective_n is not None and dropout_pct > 0 and unpadded_power < planned_pwr_pct:
        fig.add_trace(go.Scatter(
            x=[effective_n],
            y=[unpadded_power],
            mode="markers+text",
            name=f"Degraded Power if Unpadded ({unpadded_power:.1f}% at N={effective_n:,})",
            marker=dict(color="#dc2626", size=11, symbol="triangle-down", line=dict(color="white", width=2)),
            text=[f" Unpadded ({unpadded_power:.1f}%)"],
            textposition="bottom right",
            textfont=dict(color="#dc2626", size=10),
            hovertemplate="<b>Degraded Power Risk (No Cushion)</b><br>Remaining N = %{x:,}<br>Power collapses to %{y:.1f}%<extra></extra>"
        ))

    # Horizontal 80% / Target Power Reference Line
    fig.add_shape(
        type="line",
        x0=min(ns), x1=max(max(ns), recruited_n if recruited_n else planned_n) * 1.05,
        y0=planned_pwr_pct, y1=planned_pwr_pct,
        line=dict(color="#94a3b8", width=1.2, dash="dash")
    )

    # Vertical Evaluable Reference Line
    fig.add_shape(
        type="line",
        x0=planned_n, x1=planned_n,
        y0=0, y1=100,
        line=dict(color="#94a3b8", width=1.2, dash="dash")
    )

    max_x_val = max(max(ns), recruited_n if recruited_n else planned_n) * 1.08
    min_x_val = max(0, min(min(ns), effective_n if effective_n else planned_n) * 0.90)

    fig.update_layout(
        title=dict(
            text="<b>Statistical Power vs. Total Sample Size (N)</b>",
            font=dict(size=14, color="#1e293b", family="sans-serif")
        ),
        xaxis=dict(
            title="<b>Total Sample Size (N)</b>",
            range=[min_x_val, max_x_val],
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            showline=True,
            linecolor="#cbd5e1"
        ),
        yaxis=dict(
            title="<b>Statistical Power (%)</b>",
            range=[0, 105],
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            showline=True,
            linecolor="#cbd5e1",
            ticksuffix="%"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.32,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=45, r=30, t=45, b=75),
        height=390
    )

    return fig


def build_mdes_curve_plot(
    curve_data: List[Dict[str, Any]],
    planned_n: int,
    planned_effect: float,
    effect_label: str = "Cohen's d"
) -> go.Figure:
    """Builds an interactive Plotly Minimum Detectable Effect Size (MDES) vs. Sample Size curve."""
    fig = go.Figure()
    if not curve_data:
        return fig

    ns = [pt["N"] for pt in curve_data if pt["N"] >= 10]
    if not ns:
        return fig

    # Compute MDES for each N at alpha=0.05, power=0.80
    # MDES = (1.96 + 0.8416) * sqrt(2 / (N / 2)) = 2.8016 * 2 / sqrt(N)
    mdes_vals = [round(5.603 / math.sqrt(n), 3) for n in ns]

    fig.add_trace(go.Scatter(
        x=ns,
        y=mdes_vals,
        mode="lines",
        name=f"Minimum Detectable {effect_label}",
        line=dict(color="#7c3aed", width=2.8),
        hovertemplate="<b>Sample Size (N)</b>: %{x:,}<br><b>Min Detectable Effect</b>: %{y:.3f}<extra></extra>"
    ))

    # Planned target effect marker
    planned_mdes = 5.603 / math.sqrt(max(4, planned_n))
    fig.add_trace(go.Scatter(
        x=[planned_n],
        y=[planned_mdes],
        mode="markers+text",
        name=f"Planned Study (N = {planned_n:,}, MDES = {planned_mdes:.2f})",
        marker=dict(color="#dc2626", size=11, symbol="circle", line=dict(color="white", width=2)),
        text=[f" Target (MDES={planned_mdes:.2f})"],
        textposition="top right",
        textfont=dict(color="#dc2626", size=10),
        hovertemplate="<b>Planned Study MDES</b><br>N = %{x:,}<br>MDES = %{y:.3f}<extra></extra>"
    ))

    # Reference lines
    fig.add_shape(
        type="line",
        x0=planned_n, x1=planned_n,
        y0=0, y1=max(mdes_vals) * 1.05,
        line=dict(color="#cbd5e1", width=1.2, dash="dash")
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Sensitivity Curve: Minimum Detectable Effect ({effect_label}) vs. N</b>",
            font=dict(size=14, color="#1e293b", family="sans-serif")
        ),
        xaxis=dict(
            title="<b>Total Evaluable Sample Size (N)</b>",
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            showline=True,
            linecolor="#cbd5e1"
        ),
        yaxis=dict(
            title=f"<b>Min Detectable Effect ({effect_label})</b>",
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            showline=True,
            linecolor="#cbd5e1"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=45, r=30, t=45, b=65),
        height=380
    )

    return fig


# =====================================================================
# 5. Automated Protocol Justification Statement Card
# =====================================================================

def render_sample_size_protocol_card(statement: str, calc_results: Dict[str, Any]):
    """Renders a copyable biostatistical protocol justification statement formatted for NIHR/MRC/IRB."""
    with st.container(border=True):
        st.markdown("### 📋 Formal Biostatistical Sample Size Justification Statement")
        st.caption("Publication-ready protocol methodology paragraph formatted according to CONSORT 2010 (Item 7a), STROBE, and UK NIHR / MRC / Ethics Committee guidelines. Ready for direct inclusion in Protocol Section 6.")

        st.info(statement)

        st.warning(
            "⚠️ **User Responsibility & Methodology Disclaimer:** It is the user's responsibility to check that this sample size justification statement is correct and accurate, and conforms to their specific requirements or use case — no liability or responsibility is taken for any errors in calculation or presentational format. Independent verification by a qualified biostatistician is strongly recommended prior to formal protocol submission, ethics review, or trial registration."
        )

        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            st.caption("💡 *Tip: Copy this paragraph directly into your study protocol, clinical trial registry entry (ClinicalTrials.gov / ISRCTN), or ethics application.*")
        with col_c2:
            export_text = (
                f"{statement}\n\n"
                f"--------------------------------------------------------------------------------\n"
                f"DISCLAIMER:\n"
                f"It is the user's responsibility to check that this statement is correct and accurate,\n"
                f"and conforms to their specific requirements or use case — no liability or\n"
                f"responsibility is taken for any errors in calculation or presentational format.\n"
                f"--------------------------------------------------------------------------------"
            )
            st.download_button(
                label="📄 Export Protocol Statement (.txt)",
                data=export_text,
                file_name="sample_size_protocol_justification.txt",
                mime="text/plain",
                use_container_width=True
            )


# =====================================================================
# 6. Sensitivity Analysis Grid
# =====================================================================

def render_sample_size_sensitivity_grid(
    curve_data: List[Dict[str, Any]],
    planned_n: int,
    calc_results: Optional[Dict[str, Any]] = None,
    current_dropout_pct: float = 15.0
):
    """Renders a sensitivity grid showing statistical power across varying sample sizes and dropout rates."""
    with st.expander("📊 Power & Loss-to-Follow-Up Sensitivity Analysis Matrix", expanded=False):
        c_g1, c_g2 = st.columns([1, 1.25])
        with c_g1:
            st.markdown("#### 1. Sample Size (N) vs. Effect Benchmarks")
            st.caption("Power shifting if evaluable sample size falls short or exceeds target by ±20% to ±40%.")
            if curve_data:
                df_grid = pd.DataFrame(curve_data)
                st.dataframe(
                    df_grid.style.format({
                        "N": "{:,}",
                        "Power_Target": "{:.1f}%",
                        "Power_Small": "{:.1f}%",
                        "Power_Large": "{:.1f}%"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.write("Sensitivity data not available.")

        with c_g2:
            st.markdown("#### 2. Loss-to-Follow-Up (Dropout) Sensitivity")
            st.caption("Recruitment targets and degraded statistical power if dropouts occur without a recruitment cushion.")
            dropout_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
            drop_rows = []
            for d in dropout_rates:
                attr = apply_attrition_inflation(planned_n, d)
                unpad = calculate_unpadded_power_after_dropout(calc_results or {"n_total": planned_n}, d)
                d_pct = round(d * 100.0, 0)
                is_selected = abs(d_pct - current_dropout_pct) < 1.0
                drop_label = f"{d_pct:.0f}%" + (" 🎯 [Active]" if is_selected else "")
                status_icon = "🟢" if unpad["degraded_power_pct"] >= 80.0 else ("🟡" if unpad["degraded_power_pct"] >= 70.0 else "🔴")
                drop_rows.append({
                    "Dropout": drop_label,
                    "Enrolled Target N": attr["n_recruited"],
                    "Cushion": f"+{attr['additional_subjects_needed']}",
                    "Remaining N": unpad["effective_evaluable_n"],
                    "Unpadded Power": f"{unpad['degraded_power_pct']:.1f}%",
                    "Deficit": f"-{unpad['power_loss_pct']:.1f}%",
                    "Status": f"{status_icon} {'Optimal (≥80%)' if unpad['degraded_power_pct'] >= 80.0 else ('At Risk (<80%)' if unpad['degraded_power_pct'] >= 70.0 else 'Inadequate (<70%)')}"
                })
            df_drop = pd.DataFrame(drop_rows)
            st.dataframe(
                df_drop.style.format({
                    "Enrolled Target N": "{:,}",
                    "Remaining N": "{:,}"
                }),
                use_container_width=True,
                hide_index=True
            )


# =====================================================================
# 7. Full 4-Tab Educational Architecture
# =====================================================================

def render_sample_size_educational_tabs(
    calc_results: Dict[str, Any],
    meta: Dict[str, Any],
    demo_data: Dict[str, Any],
    curve_data: List[Dict[str, Any]],
    attrition: Dict[str, Any],
    cluster: Optional[Dict[str, Any]] = None
):
    """Renders the comprehensive 4-tab educational architecture for Sample Size & Power."""
    tab_names = [
        "📋 Rationale & Epidemiological Framework",
        "📗 Excel Implementation Guide",
        "🐍 Python Implementation Guide",
        "💾 Data Downloads"
    ]
    tab_math, tab_xls, tab_py, tab_dl = st.tabs(tab_names)

    # -------------------------------------------------------------
    # TAB 1: Rationale & Epidemiological Framework
    # -------------------------------------------------------------
    with tab_math:
        st.markdown(r"""
        ### 📋 Biostatistical Rationale & Mathematical Derivations

        This guide presents the complete mathematical formulations governing sample size and statistical power calculations
        across continuous, binary, longitudinal survival, and diagnostic precision designs, accompanied by term-by-term plain English breakdowns.
        """)

        with st.expander("1. Master Sample Size Formula for Comparing Two Independent Means (Two-Sample t-Test)", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            For a two-sided test with significance level $\alpha$ and power $1 - \beta$, assuming equal allocation ($n_1 = n_2 = n$):
            $$n = \frac{2 \left(Z_{1 - \alpha/2} + Z_{1 - \beta}\right)^2 \sigma_p^2}{\left(\mu_1 - \mu_2\right)^2} = \frac{2 \left(Z_{1 - \alpha/2} + Z_{1 - \beta}\right)^2}{d^2}$$
            $$N_{\text{total}} = 2n$$
            where $d = \frac{|\mu_1 - \mu_2|}{\sigma_p}$ is Cohen's standardized effect size, and $\sigma_p$ is the pooled standard deviation:
            $$\sigma_p = \sqrt{\frac{\sigma_1^2 + \sigma_2^2}{2}}$$

            #### Plain English Breakdown of Every Term:
            * **$n$**: The required number of evaluable participants in **each** treatment group.
            * **$Z_{1 - \alpha/2}$**: The critical value from the standard normal distribution corresponding to the two-sided false-positive threshold. For $\alpha = 0.05$, $Z_{0.975} = 1.960$.
            * **$Z_{1 - \beta}$**: The standard normal quantile corresponding to target statistical power. For $80\%$ power, $Z_{0.80} = 0.842$; for $90\%$ power, $Z_{0.90} = 1.282$.
            * **$(Z_{1 - \alpha/2} + Z_{1 - \beta})^2$**: The **power factor**. For $\alpha = 0.05$ and $80\%$ power, $(1.960 + 0.842)^2 = (2.802)^2 \approx 7.85$. For $90\%$ power, $(1.960 + 1.282)^2 = (3.242)^2 \approx 10.51$.
            * **$\sigma_p^2$**: Background physiological noise (pooled variance). If patient measurements vary widely, more patients are required to separate the true drug effect from noise.
            * **$(\mu_1 - \mu_2)^2$**: The squared Minimal Clinically Important Difference (MCID). Because effect size is squared in the denominator, **halving the expected difference quadruples the required sample size**!

            #### Why This Calculation is Being Used:
            * The two-sample t-test is the gold-standard analysis for parallel-group randomized clinical trials with continuous endpoints (e.g. blood pressure, HbA1c reduction, LDL cholesterol).
            * It guarantees that the trial has sufficient statistical precision to reject the null hypothesis of equal means with controlled error rates.
            """)

        with st.expander("2. Unequal Treatment Allocation Ratio Formulation (k = n2 / n1)", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            When participants are allocated with ratio $k = n_2 / n_1$ (e.g. $k = 2.0$ for $1:2$ allocation):
            $$n_1 = \frac{\left(1 + \frac{1}{k}\right) \left(Z_{1 - \alpha/2} + Z_{1 - \beta}\right)^2 \sigma_p^2}{\left(\mu_1 - \mu_2\right)^2}$$
            $$n_2 = k \cdot n_1, \quad N_{\text{total}} = n_1 + n_2 = n_1 (1 + k)$$

            #### Plain English Breakdown of Every Term:
            * **$k = n_2 / n_1$**: The allocation ratio. $k = 1.0$ is equal allocation. $k = 2.0$ assigns twice as many patients to Group 2.
            * **$(1 + 1/k)$**: The **imbalance penalty factor**. For $k = 1$, $(1 + 1) = 2.0$. For $k = 2$, $(1 + 0.5) = 1.5$.
            * **Efficiency Comparison:**
              * At $1:1$ allocation ($k = 1$): $N = 2 \times n_1$.
              * At $1:2$ allocation ($k = 2$): $n_1$ drops to $0.75 \times n_{\text{equal}}$, but $n_2 = 1.5 \times n_{\text{equal}}$, yielding $N_{\text{total}} = 2.25 \times n_{\text{equal}}$ (a $12.5\%$ increase in total patients).

            #### Why This Calculation is Being Used:
            * In early-phase trials of expensive or experimental drugs, researchers may wish to maximize patient exposure to active treatment while limiting the control group.
            * In case-control outbreak investigations where cases are scarce, matching $2$ to $4$ controls per case significantly boosts statistical power without requiring more rare disease cases.
            """)

        with st.expander("3. Binary Proportions: Fleiss Unpooled & Casagrande-Pike Continuity Correction", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            * **Unpooled Asymptotic Formula (Fleiss 1981):**
              $$n_{\text{uncorr}} = \frac{\left(Z_{1 - \alpha/2} \sqrt{2 \bar{p} (1 - \bar{p})} + Z_{1 - \beta} \sqrt{p_1 (1 - p_1) + p_2 (1 - p_2)}\right)^2}{\left(p_1 - p_2\right)^2}$$
              where $\bar{p} = \frac{p_1 + p_2}{2}$ is the average event probability under the null hypothesis.
            * **Casagrande-Pike Continuity Correction (Adjusting for Discrete Binomial Steps):**
              $$n_{\text{corrected}} = \frac{n_{\text{uncorr}}}{4} \left(1 + \sqrt{1 + \frac{4}{n_{\text{uncorr}} |p_1 - p_2|}}\right)^2$$

            #### Plain English Breakdown of Every Term:
            * **$p_1, p_2$**: Event rates in experimental and control arms (e.g. $10\%$ vs. $20\%$).
            * **$\bar{p} (1 - \bar{p})$**: Pooled binomial variance under the null hypothesis of no difference.
            * **Casagrande-Pike Term:** Discrete participant counts ($N = 1, 2, 3\dots$) jump in discrete steps rather than following a smooth continuous curve. The continuity correction adds a small sample size cushion to prevent actual power from dipping below nominal target power ($80\%$).

            #### Why This Calculation is Being Used:
            * Binary endpoints (cured/not cured, alive/dead, infection present/absent) are the most frequent outcome in Phase III clinical trials and epidemiology.
            * Uncorrected asymptotic formulas systematically underestimate required sample size when sample sizes are small or event proportions are extreme ($< 0.10$ or $> 0.90$).
            """)

        with st.expander("4. Time-to-Event / Survival Outcomes: Schoenfeld's (1981) Proportional Hazards Formulation", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            Under the Cox proportional hazards assumption, statistical power in survival analysis is driven by the **number of observed events ($E$)**, not total patient count:
            $$E = \frac{4 \left(Z_{1 - \alpha/2} + Z_{1 - \beta}\right)^2}{\left(\ln \text{HR}\right)^2}$$
            For unequal allocation ratio $k = n_2 / n_1$:
            $$E = \frac{\left(1 + k\right)^2}{k} \frac{\left(Z_{1 - \alpha/2} + Z_{1 - \beta}\right)^2}{\left(\ln \text{HR}\right)^2}$$
            * **Patient Enrollment Conversion ($N_{\text{total}}$):**
              $$N_{\text{total}} = \left\lceil \frac{E}{P(\text{Event})}\right\rceil$$
              where $P(\text{Event}) = 1 - \frac{S_1(T) + S_2(T)}{2}$ is the cumulative probability of observing an event during the trial follow-up period $T$.

            #### Plain English Breakdown of Every Term:
            * **$E$**: The required number of primary events (e.g. cardiovascular deaths, cancer progressions).
            * **$\text{HR}$**: The hypothesized Hazard Ratio (e.g. $0.70$, indicating a $30\%$ reduction in the instantaneous event rate).
            * **$\ln(\text{HR})$**: The natural logarithm of the Hazard Ratio. The log transformation linearizes proportional hazards.
            * **$P(\text{Event})$**: The anticipated cumulative event rate over the trial duration. If only $30\%$ of patients experience an event over 2 years, enrolling 300 patients yields approximately $90$ events.

            #### Why This Calculation is Being Used:
            * Survival trials (oncology, cardiovascular disease) evaluate time until event occurrence.
            * If a trial enrolls 10,000 patients but closes after only 10 events occur, it will have near zero statistical power. Schoenfeld's formula establishes the exact required event count $E$, allowing the Data Monitoring Committee (DMC) to trigger final analysis when event $E$ is observed.
            """)

        with st.expander("5. Diagnostic Accuracy & Screening Precision: Wilson Score Formulation", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            To estimate diagnostic sensitivity ($S$) or specificity within a maximum allowable margin of error $W$ with $95\%$ confidence:
            $$n_{\text{diseased}} = \left\lceil \frac{Z_{1 - \alpha/2}^2 \cdot S (1 - S)}{W^2} \right\rceil$$
            * **Prevalence-Adjusted Total Screening Cohort ($N_{\text{screening}}$):**
              $$N_{\text{screening}} = \left\lceil \frac{n_{\text{diseased}}}{\text{Prevalence}} \right\rceil$$

            #### Plain English Breakdown of Every Term:
            * **$S$**: Hypothesized sensitivity (e.g. $0.92$ or $92\%$).
            * **$W$**: Allowable half-width precision margin of error (e.g. $\pm 0.04$ or $\pm 4\%$, yielding a $95\%$ CI of $[88\%, 96\%]$).
            * **$\text{Prevalence}$**: The proportion of individuals in the target clinical population who have the true disease condition. If disease prevalence is $15\%$, only $15$ out of every $100$ screened patients will have the disease.

            #### Why This Calculation is Being Used:
            * Diagnostic trials (evaluating rapid PCR tests, point-of-care antigen devices, or AI imaging algorithms) must recruit enough diseased patients to confirm high sensitivity, and enough non-diseased individuals to confirm high specificity.
            """)

        with st.expander("6. Attrition / Dropout Inflation & Cluster Randomization Design Effects", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            * **Attrition Inflation:**
              $$N_{\text{recruited}} = \left\lceil \frac{N_{\text{evaluable}}}{1 - d} \right\rceil$$
              where $d \in [0, 1)$ is the anticipated withdrawal/dropout rate.
            * **Cluster Randomization Variance Inflation Factor (VIF / Design Effect):**
              $$\text{VIF} = 1 + (m - 1) \times \text{ICC}$$
              $$N_{\text{clustered}} = N_{\text{individual}} \times \text{VIF}, \quad C = \left\lceil \frac{N_{\text{clustered}}}{m} \right\rceil$$
              where $m$ is cluster size, $\text{ICC}$ is the intracluster correlation coefficient, and $C$ is the required number of clusters.

            #### Plain English Breakdown of Every Term:
            * **$1 / (1 - d)$**: The non-linear recruitment inflation multiplier. If $d = 0.20$ ($20\%$ dropout), the multiplier is $1 / 0.80 = 1.25$ (a $25\%$ increase, not $20\%$).
            * **$\text{ICC} = \sigma_b^2 / (\sigma_b^2 + \sigma_w^2)$**: The proportion of total variation attributable to differences between clusters (e.g. GP practices).
            * **$m - 1$**: The cluster size penalty. Larger clusters suffer worse loss of statistical efficiency than smaller clusters with identical ICC.

            #### Why This Calculation is Being Used:
            * In real clinical trials, participant dropout threatens statistical power.
            * In cluster-randomized trials, ignoring clustering leads to massively deflated $p$-values and spurious false-positive discoveries (Type I error inflation).
            """)

    # -------------------------------------------------------------
    # TAB 2: Excel Implementation Guide
    # -------------------------------------------------------------
    with tab_xls:
        st.markdown(r"""
        ### 📗 Step-by-Step Implementation in Microsoft Excel

        This tutorial guides you through building a complete, formula-driven Sample Size & Statistical Power workbook in Microsoft Excel,
        including normal quantiles, continuous two-sample formulas, binary proportions, Schoenfeld survival events,
        attrition cushions, cluster design effects, and sensitivity data tables.
        """)

        with st.expander("Spreadsheet Layout & Cell Coordinate Reference Table (A1:P20)", expanded=False):
            st.markdown(r"""
            Set up your primary parameters worksheet with the following cells:

            | Cell | Label | Description | Example Formula / Value |
            | :---: | :--- | :--- | :--- |
            | **B1** | `Alpha` ($\alpha$) | Two-sided Type I error rate | `0.05` |
            | **B2** | `Power` ($1 - \beta$) | Target statistical power | `0.80` |
            | **B3** | `Z_Alpha` | Normal quantile for two-sided $\alpha$ | `=NORM.S.INV(1 - B1/2)` (Returns `1.95996`) |
            | **B4** | `Z_Beta` | Normal quantile for statistical power | `=NORM.S.INV(B2)` (Returns `0.84162`) |
            | **B5** | `Mean_Group1` ($\mu_1$) | Hypothesized mean of experimental arm | `7.75` |
            | **B6** | `Mean_Group2` ($\mu_2$) | Hypothesized mean of control arm | `8.20` |
            | **B7** | `SD_Pooled` ($\sigma_p$) | Common standard deviation | `1.10` |
            | **B8** | `Allocation_Ratio` ($k$) | Ratio $n_2 / n_1$ | `1.0` |
            | **B9** | `Dropout_Rate` ($d$) | Anticipated loss-to-follow-up | `0.15` (15%) |
            | **B10** | `Cluster_ICC` ($\rho$) | Intracluster correlation coefficient | `0.04` |
            | **B11** | `Cluster_Size` ($m$) | Average participants per cluster | `30` |
            | **B13** | `Cohen_d` ($d$) | Standardized effect size | `=ABS(B5 - B6) / B7` |
            | **B14** | `N1_Evaluable` ($n_1$) | Required evaluable in Group 1 | `=ROUNDUP((1 + 1/B8) * ((B3 + B4)^2) * (B7^2) / ((B5 - B6)^2), 0)` |
            | **B15** | `N2_Evaluable` ($n_2$) | Required evaluable in Group 2 | `=ROUNDUP(B14 * B8, 0)` |
            | **B16** | `N_Total_Evaluable` | Total evaluable sample size | `=B14 + B15` |
            | **B17** | `N_Recruited` | Total recruitment target (+dropout) | `=ROUNDUP(B16 / (1 - B9), 0)` |
            | **B18** | `VIF` | Variance Inflation Factor for clustering | `=1 + (B11 - 1) * B10` |
            | **B19** | `N_Clustered_Total` | Total sample size adjusted for clusters | `=ROUNDUP(B16 * B18, 0)` |
            | **B20** | `Clusters_Required` | Number of distinct clusters needed | `=ROUNDUP(B19 / B11, 0)` |
            """)

        with st.expander("Step 1: Setting Up Critical Normal Quantiles in Excel", expanded=False):
            st.markdown(r"""
            1. In cell `B1`, type `0.05` (Significance level $\alpha$).
            2. In cell `B2`, type `0.80` (Target power $80\%$).
            3. In cell `B3`, compute the critical two-sided normal quantile $Z_{1 - \alpha/2}$:
               `=NORM.S.INV(1 - B1/2)` $\rightarrow$ Returns `1.960`.
            4. In cell `B4`, compute the one-sided power quantile $Z_{1 - \beta}$:
               `=NORM.S.INV(B2)` $\rightarrow$ Returns `0.842`.
            5. *For Student's t distribution approximation (small N):*
               Use `=T.INV.2T(B1, df)` where `df` is the estimated degrees of freedom ($N - 2$).
            """)

        with st.expander("Step 2: Implementing the Continuous Two-Sample t-Test Formula", expanded=False):
            st.markdown(r"""
            1. Enter $\mu_1$ in cell `B5` (e.g. `7.75`), $\mu_2$ in `B6` (e.g. `8.20`), and $\sigma_p$ in `B7` (e.g. `1.10`).
            2. In cell `B13`, compute Cohen's $d$:
               `=ABS(B5 - B6) / B7` $\rightarrow$ Returns `0.409`.
            3. In cell `B14`, compute required sample size per arm ($n_1$):
               `=ROUNDUP((1 + 1/B8) * ((B3 + B4)^2) * (B7^2) / ((B5 - B6)^2), 0)`
            4. In cell `B15`, compute $n_2$: `=ROUNDUP(B14 * B8, 0)`.
            5. In cell `B16`, compute total evaluable $N$: `=B14 + B15`.
            """)

        with st.expander("Step 3: Implementing Binary Proportions & Continuity Correction", expanded=False):
            st.markdown(r"""
            For comparing two independent event proportions $p_1$ and $p_2$:
            1. Enter $p_1$ in cell `E5` (e.g. `0.10`) and $p_2$ in `E6` (e.g. `0.20`).
            2. In cell `E7`, compute pooled average $\bar{p}$: `=(E5 + E6) / 2`.
            3. In cell `E8`, compute uncorrected Fleiss sample size per arm:
               `=ROUNDUP(((B3 * SQRT(2 * E7 * (1 - E7)) + B4 * SQRT(E5 * (1 - E5) + E6 * (1 - E6)))^2) / ((E1 - E2)^2), 0)`
            4. In cell `E9`, apply Casagrande-Pike continuity correction:
               `=ROUNDUP((E8 / 4) * ((1 + SQRT(1 + 4 / (E8 * ABS(E5 - E6))))^2), 0)`
            """)

        with st.expander("Step 4: Implementing Schoenfeld's Survival Events Formula", expanded=False):
            st.markdown(r"""
            For survival analysis under Cox proportional hazards:
            1. Enter target Hazard Ratio ($\text{HR}$) in cell `G5` (e.g. `0.70`).
            2. Enter cumulative event probability $P(\text{Event})$ in cell `G6` (e.g. `0.55`).
            3. In cell `G7`, compute required event count $E$:
               `=ROUNDUP(4 * ((B3 + B4)^2) / ((LN(G5))^2), 0)` $\rightarrow$ Returns `284 Events`.
            4. In cell `G8`, convert events to patient enrollment $N$:
               `=ROUNDUP(G7 / G6, 0)` $\rightarrow$ Returns `517 Patients`.
            """)

        with st.expander("Step 5: Implementing Real-World Attrition Inflation & Cluster Design Effects", expanded=False):
            st.markdown(r"""
            1. In cell `B9`, enter anticipated dropout proportion (e.g. `0.15` for 15%).
            2. In cell `B17`, compute the inflation recruitment target:
               `=ROUNDUP(B16 / (1 - B9), 0)`
               *Never simply multiply by 1.15; dividing by $(1 - d)$ correctly compensates for dropouts.*
            3. In cell `B18`, calculate Variance Inflation Factor (VIF):
               `=1 + (B11 - 1) * B10`
            4. In cell `B19`, compute total clustered sample size:
               `=ROUNDUP(B16 * B18, 0)`
            5. In cell `B20`, compute the number of clusters needed:
               `=ROUNDUP(B19 / B11, 0)`
            """)

        with st.expander("Step 6: Building a Two-Way Sensitivity Table in Excel using Data Tables", expanded=False):
            st.markdown(r"""
            To model power across varying sample sizes ($N$) and effect sizes ($\Delta$):
            1. Across row 25, enter sample sizes: `50, 75, 100, 125, 150, 200, 250`.
            2. Down column A (cells `A26:A32`), enter effect sizes: `0.20, 0.30, 0.40, 0.50, 0.60`.
            3. In cell `A25` (the top-left corner), enter the power formula reference: `=NORM.S.DIST((B13 * SQRT(B16 / 2) - B3), TRUE)`.
            4. Highlight the range `A25:H32`.
            5. Go to Excel Ribbon: `Data -> What-If Analysis -> Data Table...`
            6. For **Row input cell**, select your sample size parameter (`$B$16`).
            7. For **Column input cell**, select your Cohen's d parameter (`$B$13`).
            8. Click **OK**. Excel automatically populates the entire matrix with exact statistical power values!
            """)

    # -------------------------------------------------------------
    # TAB 3: Python Implementation Guide
    # -------------------------------------------------------------
    with tab_py:
        st.markdown(r"""
        ### 🐍 Production-Ready Python Script Guide

        This guide presents vectorized, production-grade Python pipelines utilizing standard scientific libraries
        (`scipy.stats` and `statsmodels.stats.power`) to execute sample size planning.
        """)

        with st.expander("Stage 1: Continuous Outcomes Pipeline (Two-Sample t-Test & Cohen's d)", expanded=False):
            st.markdown(r"""
            **Purpose:** Calculates sample size and power for independent groups using both exact normal approximations and `statsmodels`.
            """)
            st.code(r'''
import math
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

def calculate_two_means_sample_size(mean1, mean2, sd1, sd2=None, alpha=0.05, power=0.80, ratio=1.0):
    """Calculates required sample size for two independent means with allocation ratio k."""
    sd_pooled = sd1 if sd2 is None else math.sqrt((sd1**2 + sd2**2) / 2.0)
    cohen_d = abs(mean1 - mean2) / sd_pooled
    
    # 1. Closed-form normal approximation
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    factor = 1.0 + (1.0 / ratio)
    n1_approx = math.ceil(factor * ((z_alpha + z_beta)**2) / (cohen_d**2))
    n2_approx = math.ceil(n1_approx * ratio)
    
    # 2. Exact t-distribution calculation via statsmodels
    power_analysis = TTestIndPower()
    n1_exact = power_analysis.solve_power(
        effect_size=cohen_d,
        power=power,
        alpha=alpha,
        ratio=ratio,
        alternative='two-sided'
    )
    
    return {
        "cohen_d": round(cohen_d, 3),
        "n1_approx": n1_approx,
        "n2_approx": n2_approx,
        "n_total_approx": n1_approx + n2_approx,
        "n1_exact_t": math.ceil(n1_exact),
        "n_total_exact": math.ceil(n1_exact * (1.0 + ratio))
    }

# Example: HbA1c reduction from 8.20% to 7.75% (SD = 1.10)
res = calculate_two_means_sample_size(mean1=8.20, mean2=7.75, sd1=1.10, power=0.85)
print(res)
''', language="python")

        with st.expander("Stage 2: Binary Proportions & Risk Metrics Pipeline (Casagrande-Pike)", expanded=False):
            st.markdown(r"""
            **Purpose:** Computes sample sizes for binary trials with Fleiss unpooled and Casagrande-Pike continuity corrections.
            """)
            st.code(r'''
import math
from scipy import stats

def calculate_two_proportions_sample_size(p1, p2, alpha=0.05, power=0.80):
    """Calculates sample size for two proportions with Casagrande-Pike continuity correction."""
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2.0
    diff = abs(p1 - p2)
    
    # Fleiss uncorrected formula
    num = (z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar)) + 
           z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))) ** 2
    n_uncorr = math.ceil(num / (diff ** 2))
    
    # Casagrande-Pike continuity correction
    term = 4.0 / (n_uncorr * diff)
    n_corr = math.ceil((n_uncorr / 4.0) * ((1.0 + math.sqrt(1.0 + term)) ** 2))
    
    return {
        "p1": p1, "p2": p2, "risk_diff": round(diff, 4),
        "n_per_arm_uncorrected": n_uncorr,
        "n_per_arm_corrected": n_corr,
        "n_total_corrected": n_corr * 2
    }

# Example: MACE reduction from 20% to 10%
res_prop = calculate_two_proportions_sample_size(p1=0.20, p2=0.10, alpha=0.05, power=0.80)
print(res_prop)
''', language="python")

        with st.expander("Stage 3: Longitudinal Survival Schoenfeld Formulation", expanded=False):
            st.markdown(r"""
            **Purpose:** Computes required event count $E$ and patient accrual $N$ under proportional hazards.
            """)
            st.code(r'''
import math
from scipy import stats

def calculate_survival_sample_size(hazard_ratio, alpha=0.05, power=0.80, event_prob=0.50, ratio=1.0):
    """Calculates Schoenfeld required events E and patient enrollment N for Cox regression."""
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    ln_hr = math.log(hazard_ratio)
    
    # Schoenfeld formula for events E
    allocation_factor = ((1.0 + ratio)**2) / ratio
    events_required = math.ceil(allocation_factor * ((z_alpha + z_beta)**2) / (ln_hr**2))
    
    # Convert events to patient enrollment
    patients_required = math.ceil(events_required / event_prob)
    n1 = math.ceil(patients_required / (1.0 + ratio))
    n2 = math.ceil(n1 * ratio)
    
    return {
        "hazard_ratio": hazard_ratio,
        "required_events": events_required,
        "event_probability": event_prob,
        "n_total_patients": n1 + n2,
        "n1_patients": n1,
        "n2_patients": n2
    }

# Example: Oncology trial with HR = 0.70, power = 85%, event incidence = 55%
res_surv = calculate_survival_sample_size(hazard_ratio=0.70, power=0.85, event_prob=0.55)
print(res_surv)
''', language="python")

        with st.expander("Stage 4: Real-World Modifiers: Dropout Inflation & Cluster VIF", expanded=False):
            st.markdown(r"""
            **Purpose:** Adjusts evaluable sample sizes for participant dropout and intracluster correlation.
            """)
            st.code(r'''
import math

def apply_trial_adjustments(n_evaluable, dropout_rate=0.15, cluster_size=None, icc=None):
    """Applies dropout cushion and cluster design effect (VIF)."""
    # 1. Attrition inflation
    n_recruited = math.ceil(n_evaluable / (1.0 - dropout_rate)) if dropout_rate < 1.0 else n_evaluable
    
    # 2. Cluster design effect
    vif = 1.0
    n_clustered = n_evaluable
    clusters_needed = None
    if cluster_size is not None and icc is not None and icc > 0:
        vif = 1.0 + (cluster_size - 1) * icc
        n_clustered = math.ceil(n_evaluable * vif)
        clusters_needed = math.ceil(n_clustered / cluster_size)
        n_clustered = clusters_needed * cluster_size  # rounded to whole clusters
        
    return {
        "n_evaluable": n_evaluable,
        "n_recruited_individual": n_recruited,
        "dropout_rate_pct": round(dropout_rate * 100.0, 1),
        "cluster_vif": round(vif, 2),
        "n_clustered_total": n_clustered,
        "clusters_required": clusters_needed
    }

# Example: 200 evaluable patients, 15% dropout, 30 patients/GP practice, ICC = 0.04
adj = apply_trial_adjustments(n_evaluable=200, dropout_rate=0.15, cluster_size=30, icc=0.04)
print(adj)
''', language="python")

        with st.expander("Stage 5: Complete Standalone End-to-End Production Script", expanded=False):
            st.code(r'''
# Complete Standalone Production Script for Sample Size Planning
from src.stats_calcs.sample_size_engine import (
    calculate_sample_size_two_means,
    apply_attrition_inflation,
    apply_cluster_design_effect,
    generate_protocol_justification_statement,
    evaluate_sample_size_prerequisites
)

# 1. Configure Trial Parameters (All Meadows Cardiovascular Institute)
calc = calculate_sample_size_two_means(
    mean1=8.20, mean2=7.75, sd1=1.10, sd2=1.10,
    alpha=0.05, power=0.85, allocation_ratio=1.0
)

# 2. Apply Real-World Adjustments
attrition = apply_attrition_inflation(calc['n_total'], dropout_rate=0.15)
evals = evaluate_sample_size_prerequisites(calc)

# 3. Generate Formal NIHR Protocol Statement
statement = generate_protocol_justification_statement(
    calc, primary_endpoint_label="HbA1c reduction at 24 weeks", dropout_rate=0.15
)

# 4. Display Results
print("=" * 65)
print("BIOSTATISTICAL SAMPLE SIZE & POWER REPORT")
print("=" * 65)
print(f"Evaluable Sample Size:  N = {calc['n_total']} ({calc['n1']} per arm)")
print(f"Standardized Cohen's d: d = {calc['cohen_d']:.3f} ({calc['cohen_tier']})")
print(f"Target Statistical Power: {calc['target_power']*100:.0f}% (Achieved: {calc['achieved_power']*100:.1f}%)")
print(f"Recruitment Target:     N = {attrition['n_recruited']} (Safeguarding 15% dropout)")
print("=" * 65)
print("PROTOCOL JUSTIFICATION PARAGRAPH:")
print(statement)
''', language="python")

    # -------------------------------------------------------------
    # TAB 4: Data Downloads
    # -------------------------------------------------------------
    with tab_dl:
        st.markdown(r"""
        ### 💾 Live Synthetic Demonstration Data & Workbook Downloads

        All datasets provided below are **100% synthetic, computer-generated demonstration files** created for teaching,
        protocol drafting, and code rehearsal. They contain **zero real patient data or protected health information**.
        """)

        st.warning("""
        ⚠️ **EDUCATIONAL & METHODOLOGICAL NOTICE**:
        * **100% Synthetic / Artificial Data:** All patient records, baseline measurements, and simulated endpoints were synthesized via mathematical models.
        * **Educational Rehearsal Only:** Designed for academic instruction, clinical trial workshops, and practicing power curves in Excel, R (`pwr`), Python, Stata, or SAS.
        * **No Clinical Utility:** Must never be used for actual clinical trial licensing or regulatory submissions without independent statistical review.
        """)

        c_d1, c_d2, c_d3 = st.columns(3)

        df_micro = demo_data.get("df", pd.DataFrame())
        if not isinstance(df_micro, pd.DataFrame):
            df_micro = pd.DataFrame(df_micro)

        with c_d1:
            with st.container(border=True):
                st.markdown(f"#### 📄 Synthetic Patient Microdata ({len(df_micro)} Rows)")
                st.markdown(f"Contains **{len(df_micro)} simulated patient records** matching active study design with arm, biomarkers, and endpoints.")
                st.caption("*100% synthetic dataset generated purely for demonstration.*")
                csv_micro = df_micro.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇️ Download Patient Microdata CSV",
                    data=csv_micro,
                    file_name=f"synthetic_{meta['scenario_key']}_patient_microdata.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                with st.expander("👁️ Preview First 5 Rows", expanded=False):
                    st.dataframe(df_micro.head(5), use_container_width=True, hide_index=True)

        with c_d2:
            with st.container(border=True):
                st.markdown("#### 📊 Power Curve Matrix (CSV)")
                st.markdown("Includes full power curve coordinate matrix across varying sample sizes ($N$) and small, target, and large effect benchmarks.")
                st.caption("*Reconstruct power curves and sensitivity surfaces in Excel or R.*")
                df_curve = pd.DataFrame(curve_data) if curve_data else pd.DataFrame([{"N": 100, "Power": 80.0}])
                csv_curve = df_curve.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Power Curve Matrix CSV",
                    data=csv_curve,
                    file_name=f"power_curve_{meta['scenario_key']}_matrix.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                with st.expander("👁️ Preview Power Coordinates", expanded=False):
                    st.dataframe(df_curve.head(5), use_container_width=True, hide_index=True)

        with c_d3:
            with st.container(border=True):
                st.markdown("#### 📉 Loss-to-Follow-Up Matrix (CSV)")
                st.markdown("Full 0% to 40% attrition matrix showing recruitment targets, cushion sizes, and degraded unpadded statistical power.")
                st.caption(f"*Active slider setting: {attrition.get('dropout_rate_pct', 15.0):.0f}% attrition.*")
                drop_grid = []
                base_sens_n = cluster["n_clustered_total"] if cluster else calc_results.get("n_total", 100)
                for d_rate in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
                    att_row = apply_attrition_inflation(base_sens_n, d_rate)
                    unpad_row = calculate_unpadded_power_after_dropout(calc_results, d_rate)
                    drop_grid.append({
                        "Dropout_Rate_Pct": round(d_rate * 100, 0),
                        "Evaluable_Target_N": base_sens_n,
                        "Recruitment_Target_N": att_row["n_recruited"],
                        "Recruitment_Cushion": att_row["additional_subjects_needed"],
                        "Remaining_Evaluable_N": unpad_row["effective_evaluable_n"],
                        "Guaranteed_Nominal_Power_Pct": unpad_row["guaranteed_power_pct"],
                        "Degraded_Unpadded_Power_Pct": unpad_row["degraded_power_pct"],
                        "Statistical_Power_Loss_Pct": unpad_row["power_loss_pct"],
                        "Status": "Optimal" if unpad_row["degraded_power_pct"] >= 80.0 else ("At Risk" if unpad_row["degraded_power_pct"] >= 70.0 else "Inadequate")
                    })
                df_drop_csv = pd.DataFrame(drop_grid)
                csv_attr = df_drop_csv.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Attrition Matrix CSV",
                    data=csv_attr,
                    file_name=f"attrition_sensitivity_{meta['scenario_key']}_matrix.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                with st.expander("👁️ Preview Attrition Matrix", expanded=False):
                    st.dataframe(df_drop_csv.head(5), use_container_width=True, hide_index=True)

        with st.expander("📋 Synthetic Data Generation Methodology & Variable Dictionary", expanded=False):
            st.markdown(r"""
            #### How This Synthetic Dataset Was Created
            1. **Cohort Simulation:** Patient identifiers and baseline covariates (e.g. age, sex, ECOG status, baseline HbA1c) are synthesized to mirror realistic clinical trial demographics.
            2. **Treatment Assignment:** Simulated patients are assigned to active or control arms following the trial's specified allocation ratio $k$.
            3. **Outcome Generation:** Outcomes are sampled from probability distributions parameterized with the exact effect size specified in the sample size calculation, ensuring empirical estimates align with theoretical assumptions.
            4. **Information Governance:** Because this dataset is entirely synthetic, it is safe to publish, distribute, and utilize in educational workshops without Information Governance, NHS Caldicott Guardian, or GDPR restrictions.
            """)


# =====================================================================
# 8. Main Module Rendering Controller
# =====================================================================

def render_sample_size_module(
    active_method_key: Optional[str] = None,
    method_info: Optional[Dict[str, Any]] = None
):
    """Main rendering controller for the Sample Size & Statistical Power Calculations Module."""

    # 1. Plain English Pedagogical Introductory Guide
    render_sample_size_plain_english_guide()

    # 2. Scenario & Design Selection Controls
    scenario_options = {
        "cardiovascular_rct": "All Meadows Cardiovascular Institute: Double-Blind Phase III RCT (HbA1c & MACE) [Synthetic]",
        "public_health_case_control": "Riverside Health Protection Team: Outbreak Case-Control & School Cluster Trial [Synthetic]",
        "oncology_survival": "Inner City Teaching Hospital: Metastatic NSCLC Survival (HR = 0.70) & Diagnostic Precision [Synthetic]",
        "custom": "Custom Guided Study Design & Parameter Exploration [Synthetic]"
    }

    col_s1, col_s2 = st.columns([2.0, 1.4])
    with col_s1:
        scenario_keys = list(scenario_options.keys())
        default_scenario_idx = 0
        preset_scenario = st.session_state.get("sample_size_scenario")
        if preset_scenario in scenario_keys:
            default_scenario_idx = scenario_keys.index(preset_scenario)

        selected_scenario = st.selectbox(
            "Select Clinical / Epidemiological Study Scenario:",
            options=scenario_keys,
            format_func=lambda k: scenario_options[k],
            index=default_scenario_idx
        )

    cluster_adj = None
    if selected_scenario != "custom":
        demo_data = generate_sample_size_demo_data(scenario_key=selected_scenario)

        with col_s2:
            st.info(f"📍 **{demo_data['organisation_name']}**\n\n*{demo_data['domain']}*")

        meta = {
            "scenario_key": selected_scenario,
            "title": demo_data["title"],
            "organisation": demo_data["organisation"],
            "domain": demo_data["domain"],
            "intervention": demo_data["intervention"],
            "comparator": demo_data["comparator"],
            "primary_endpoint": demo_data["primary_endpoint"],
            "description": demo_data["description"],
            "takeaway": demo_data["takeaway"]
        }

        with st.container(border=True):
            st.markdown("#### ⚙️ Interactive Protocol Parameters & Power Trade-Off Controls")
            st.caption("Adjust statistical power (1 - β), significance level (α), effect size, study design modifiers, and loss-to-follow-up to explore real-time biostatistical trade-offs for this scenario.")

            if selected_scenario == "cardiovascular_rct":
                c_sc1, c_sc2, c_sc3, c_sc4 = st.columns(4)
                with c_sc1:
                    alpha_val = st.selectbox("Significance Level (α):", options=[0.01, 0.05, 0.10], index=1, key="am_alpha")
                    power_val = st.slider("Target Statistical Power (1 - β):", min_value=0.70, max_value=0.99, value=0.85, step=0.05, key="am_power")
                with c_sc2:
                    delta_val = st.slider("Target Difference (Δ HbA1c %):", min_value=0.20, max_value=1.00, value=0.45, step=0.05, key="am_delta")
                    sd_val = st.slider("Standard Deviation (σ %):", min_value=0.60, max_value=2.00, value=1.10, step=0.05, key="am_sd")
                with c_sc3:
                    alloc_val = st.slider("Allocation Ratio (k = n2 / n1):", min_value=0.5, max_value=3.0, value=1.0, step=0.5, key="am_alloc")
                with c_sc4:
                    dropout_slider = st.slider("Anticipated Dropout (%):", min_value=0, max_value=35, value=15, step=5, key="am_drop", help="Proportion of participants expected to withdraw or be lost to follow-up before the primary endpoint.")

                calc_results = calculate_sample_size_two_means(
                    mean1=7.75 + delta_val, mean2=7.75, sd1=sd_val, sd2=sd_val,
                    alpha=alpha_val, power=power_val, allocation_ratio=alloc_val
                )
                base_n = calc_results["n_total"]
                attrition = apply_attrition_inflation(base_n, dropout_slider / 100.0)
                unpadded_info = calculate_unpadded_power_after_dropout(calc_results, dropout_slider / 100.0)
                design_type = "two_means"
                curve_data = generate_power_curve_data(design_type, calc_results)

            elif selected_scenario == "public_health_case_control":
                c_sc1, c_sc2, c_sc3, c_sc4 = st.columns(4)
                with c_sc1:
                    alpha_val = st.selectbox("Significance Level (α):", options=[0.01, 0.05, 0.10], index=1, key="rs_alpha")
                    power_val = st.slider("Target Statistical Power (1 - β):", min_value=0.70, max_value=0.99, value=0.80, step=0.05, key="rs_power")
                with c_sc2:
                    or_val = st.slider("Hypothesized Odds Ratio (OR):", min_value=1.30, max_value=4.50, value=2.50, step=0.10, key="rs_or")
                    k_ctrl = st.slider("Controls per Case (k):", min_value=1.0, max_value=4.0, value=2.0, step=0.5, key="rs_ctrl")
                with c_sc3:
                    clust_size_val = st.slider("School Cluster Size (m):", min_value=10, max_value=60, value=30, step=5, key="rs_size")
                    icc_val = st.slider("Intracluster Correlation (ICC):", min_value=0.000, max_value=0.100, value=0.040, step=0.005, format="%.3f", key="rs_icc")
                with c_sc4:
                    dropout_slider = st.slider("Anticipated Dropout (%):", min_value=0, max_value=35, value=10, step=5, key="rs_drop", help="Proportion of participants expected to withdraw or be lost to follow-up before the primary endpoint.")

                calc_results = calculate_sample_size_case_control_or(
                    p0_exposure=0.20, odds_ratio=or_val, alpha=alpha_val, power=power_val, controls_per_case=k_ctrl
                )
                cluster_adj = apply_cluster_design_effect(
                    n_individual=calc_results["n_total"], cluster_size=int(clust_size_val), icc=icc_val
                )
                base_n = cluster_adj["n_clustered_total"]
                attrition = apply_attrition_inflation(base_n, dropout_slider / 100.0)
                cluster_adj["clusters_required_recruited"] = math.ceil(attrition["n_recruited"] / cluster_adj["cluster_size_m"])
                unpadded_info = calculate_unpadded_power_after_dropout(calc_results, dropout_slider / 100.0)
                design_type = "case_control"
                curve_data = generate_power_curve_data(design_type, calc_results)

            else:  # oncology_survival
                c_sc1, c_sc2, c_sc3, c_sc4 = st.columns(4)
                with c_sc1:
                    alpha_val = st.selectbox("Significance Level (α):", options=[0.01, 0.05, 0.10], index=1, key="ic_alpha")
                    power_val = st.slider("Target Statistical Power (1 - β):", min_value=0.70, max_value=0.99, value=0.85, step=0.05, key="ic_power")
                with c_sc2:
                    hr_val = st.slider("Target Hazard Ratio (HR):", min_value=0.40, max_value=0.90, value=0.70, step=0.05, key="ic_hr")
                    ev_prob = st.slider("Cumulative Event Prob P(Event):", min_value=0.25, max_value=0.90, value=0.65, step=0.05, key="ic_ev")
                with c_sc3:
                    alloc_val = st.slider("Allocation Ratio (k = n2 / n1):", min_value=0.5, max_value=2.0, value=1.0, step=0.5, key="ic_alloc")
                with c_sc4:
                    dropout_slider = st.slider("Anticipated Dropout (%):", min_value=0, max_value=35, value=15, step=5, key="ic_drop", help="Proportion of participants expected to withdraw or be lost to follow-up before the primary endpoint.")

                calc_results = calculate_sample_size_survival_logrank(
                    hazard_ratio=hr_val, alpha=alpha_val, power=power_val,
                    allocation_ratio=alloc_val, cumulative_event_prob=ev_prob
                )
                base_n = calc_results["n_total"]
                attrition = apply_attrition_inflation(base_n, dropout_slider / 100.0)
                unpadded_info = calculate_unpadded_power_after_dropout(calc_results, dropout_slider / 100.0)
                design_type = "survival"
                curve_data = generate_power_curve_data(design_type, calc_results)

        # Dynamic Takeaway Banner
        if cluster_adj and cluster_adj.get("icc", 0) > 0:
            if dropout_slider == 0:
                meta["takeaway"] = (
                    f"To achieve {calc_results.get('achieved_power', 0.80)*100:.1f}% statistical power, "
                    f"an unclustered individual study requires {calc_results.get('n_total', 100):,} participants. "
                    f"Accounting for cluster randomization across schools/wards (cluster size m = {cluster_adj['cluster_size_m']}, "
                    f"ICC = {cluster_adj['icc']:.3f}, VIF = {cluster_adj['design_effect_vif']:.2f}), "
                    f"the trial requires {cluster_adj['n_clustered_total']:,} evaluable participants across {cluster_adj['clusters_required']} clusters. "
                    f"With 0% anticipated loss-to-follow-up, the final enrollment target remains {cluster_adj['n_clustered_total']:,} participants."
                )
            else:
                meta["takeaway"] = (
                    f"To achieve {calc_results.get('achieved_power', 0.80)*100:.1f}% statistical power, "
                    f"the clustered trial requires {cluster_adj['n_clustered_total']:,} evaluable participants across {cluster_adj['clusters_required']} clusters. "
                    f"Accounting for {dropout_slider}% anticipated loss-to-follow-up, the recruitment target is inflated to "
                    f"{attrition['n_recruited']:,} participants across {cluster_adj['clusters_required_recruited']} clusters "
                    f"(+{attrition['additional_subjects_needed']:,} cushion). "
                    f"Without this cushion, power would degrade to {unpadded_info['degraded_power_pct']:.1f}%."
                )
        else:
            if dropout_slider == 0:
                meta["takeaway"] = (
                    f"Achieving {calc_results.get('achieved_power', 0.85)*100:.1f}% statistical power requires "
                    f"{base_n:,} evaluable participants ({calc_results.get('n1', base_n//2):,} per arm). "
                    f"With 0% anticipated loss-to-follow-up, the recruitment target remains {base_n:,} participants "
                    f"(no attrition cushion required)."
                )
            else:
                meta["takeaway"] = (
                    f"Achieving {calc_results.get('achieved_power', 0.85)*100:.1f}% statistical power requires "
                    f"{base_n:,} evaluable participants ({calc_results.get('n1', base_n//2):,} per arm). "
                    f"With {dropout_slider}% anticipated loss-to-follow-up, the recruitment target is inflated to "
                    f"{attrition['n_recruited']:,} participants (+{attrition['additional_subjects_needed']:,} cushion). "
                    f"Without this cushion, power would degrade to {unpadded_info['degraded_power_pct']:.1f}%."
                )

    else:
        # Custom Mode with Guided Study Design Selector
        with col_s2:
            custom_options = [
                "Continuous: Two Independent Means (t-test)",
                "Continuous: Paired / Pre-Post Means",
                "Continuous: One-Way ANOVA (Multiple Groups)",
                "Continuous: Pearson Correlation (r)",
                "Binary: Two Independent Proportions",
                "Binary: Prospective Cohort (Relative Risk)",
                "Binary: Retrospective Case-Control (Odds Ratio)",
                "Survival: Cox Log-Rank Test (Hazard Ratio)",
                "Diagnostic: Target Sensitivity Precision"
            ]
            default_custom_idx = 0
            preset_custom = st.session_state.get("sample_size_custom_design")
            if preset_custom in custom_options:
                default_custom_idx = custom_options.index(preset_custom)

            custom_design = st.selectbox(
                "Select Statistical Method & Endpoint Family:",
                options=custom_options,
                index=default_custom_idx
            )

        st.markdown("#### ⚙️ Custom Study Design Parameters")
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)

        with c_p1:
            alpha_val = st.selectbox("Significance Level (α):", options=[0.01, 0.05, 0.10], index=1)
            power_val = st.slider("Target Statistical Power (1 - β):", min_value=0.70, max_value=0.99, value=0.85, step=0.05)

        with c_p2:
            alloc_ratio = st.slider("Allocation Ratio (k = n2 / n1):", min_value=0.5, max_value=4.0, value=1.0, step=0.5)
            dropout_val = st.slider("Anticipated Dropout (%):", min_value=0, max_value=40, value=15, step=5)

        with c_p3:
            enable_cluster = st.checkbox("Include Cluster Randomization (VIF)", value=False)
            if enable_cluster:
                clust_size = st.number_input("Average Cluster Size (m):", min_value=5, max_value=200, value=30, step=5)
                clust_icc = st.number_input("Intracluster Correlation (ICC):", min_value=0.001, max_value=0.20, value=0.04, step=0.01, format="%.3f")
            else:
                clust_size, clust_icc = 30, 0.0

        with c_p4:
            # Custom method specific inputs
            if "Two Independent Means" in custom_design:
                m1 = st.number_input("Mean Arm 1 (μ1):", value=8.20, step=0.10)
                m2 = st.number_input("Mean Arm 2 (μ2):", value=7.75, step=0.10)
                sd_val = st.number_input("Standard Deviation (σ):", min_value=0.10, value=1.10, step=0.10)
                calc_results = calculate_sample_size_two_means(m1, m2, sd_val, sd_val, alpha=alpha_val, power=power_val, allocation_ratio=alloc_ratio)
                curve_data = generate_power_curve_data("two_means", calc_results)
                endpoint_label = "continuous outcome measure"
            elif "Paired" in custom_design:
                m_diff = st.number_input("Mean Difference (μ_diff):", value=2.0, step=0.5)
                sd_d = st.number_input("SD of Differences (σ_d):", min_value=0.1, value=4.0, step=0.5)
                calc_results = calculate_sample_size_paired_means(m_diff, sd_d, alpha=alpha_val, power=power_val)
                curve_data = generate_power_curve_data("two_means", {"mean1": m_diff, "mean2": 0.0, "sd1": sd_d, "alpha": alpha_val, "power": power_val})
                endpoint_label = "pre-post mean change"
            elif "ANOVA" in custom_design:
                c_f = st.slider("Cohen's f Effect Size:", min_value=0.10, max_value=0.60, value=0.25, step=0.05)
                n_grp = st.slider("Number of Groups:", min_value=3, max_value=6, value=3)
                calc_results = calculate_sample_size_anova(num_groups=n_grp, cohen_f=c_f, alpha=alpha_val, power=power_val)
                curve_data = generate_power_curve_data("two_means", {"mean1": c_f, "mean2": 0.0, "sd1": 1.0, "alpha": alpha_val, "power": power_val})
                endpoint_label = "multi-arm continuous outcome"
            elif "Correlation" in custom_design:
                r_val = st.slider("Hypothesized Correlation (r):", min_value=0.10, max_value=0.80, value=0.30, step=0.05)
                calc_results = calculate_sample_size_correlation(r=r_val, alpha=alpha_val, power=power_val)
                curve_data = generate_power_curve_data("correlation", calc_results)
                endpoint_label = "linear correlation"
            elif "Two Independent Proportions" in custom_design:
                p1_val = st.slider("Prop Arm 1 (p1):", min_value=0.02, max_value=0.95, value=0.20, step=0.02)
                p2_val = st.slider("Prop Arm 2 (p2):", min_value=0.02, max_value=0.95, value=0.10, step=0.02)
                calc_results = calculate_sample_size_two_proportions(p1_val, p2_val, alpha=alpha_val, power=power_val, allocation_ratio=alloc_ratio)
                curve_data = generate_power_curve_data("two_proportions", calc_results)
                endpoint_label = "binary clinical event"
            elif "Cohort" in custom_design:
                p0_c = st.slider("Baseline Incidence (p0):", min_value=0.02, max_value=0.80, value=0.15, step=0.02)
                rr_c = st.slider("Hypothesized Relative Risk (RR):", min_value=0.20, max_value=3.0, value=1.60, step=0.10)
                calc_results = calculate_sample_size_cohort_rr(p0_c, rr_c, alpha=alpha_val, power=power_val, allocation_ratio=alloc_ratio)
                curve_data = generate_power_curve_data("two_proportions", {"p1": p0_c, "p2": min(0.99, p0_c * rr_c), "alpha": alpha_val, "power": power_val})
                endpoint_label = "cohort incident risk"
            elif "Case-Control" in custom_design:
                p0_cc = st.slider("Control Exposure (p0):", min_value=0.05, max_value=0.80, value=0.20, step=0.05)
                or_cc = st.slider("Hypothesized Odds Ratio (OR):", min_value=1.2, max_value=5.0, value=2.50, step=0.1)
                calc_results = calculate_sample_size_case_control_or(p0_cc, or_cc, alpha=alpha_val, power=power_val, controls_per_case=alloc_ratio)
                curve_data = generate_power_curve_data("two_proportions", {"p1": calc_results["p1_case_exposure"], "p2": p0_cc, "alpha": alpha_val, "power": power_val})
                endpoint_label = "case exposure odds"
            elif "Survival" in custom_design:
                hr_val = st.slider("Target Hazard Ratio (HR):", min_value=0.30, max_value=0.90, value=0.70, step=0.05)
                ev_prob = st.slider("Cumulative Event Prob P(Event):", min_value=0.10, max_value=0.90, value=0.55, step=0.05)
                calc_results = calculate_sample_size_survival_logrank(hr_val, alpha=alpha_val, power=power_val, cumulative_event_prob=ev_prob)
                curve_data = generate_power_curve_data("survival", calc_results)
                endpoint_label = "progression-free survival"
            else:
                sens_val = st.slider("Target Sensitivity (%):", min_value=70.0, max_value=99.0, value=92.0, step=1.0)
                margin_val = st.slider("Margin of Error (±%):", min_value=1.0, max_value=10.0, value=4.0, step=0.5)
                prev_val = st.slider("Disease Prevalence (%):", min_value=1.0, max_value=50.0, value=15.0, step=1.0)
                calc_results = calculate_sample_size_diagnostic_accuracy(sens_val / 100.0, margin_val / 100.0, prev_val / 100.0, alpha=alpha_val)
                curve_data = generate_power_curve_data("two_means", {"mean1": sens_val / 100.0, "mean2": 0.5, "sd1": 0.25, "alpha": alpha_val, "power": 0.80})
                endpoint_label = "diagnostic test sensitivity"

        base_n = calc_results.get("n_total", 100)
        if enable_cluster and clust_icc > 0:
            cluster_adj = apply_cluster_design_effect(base_n, int(clust_size), clust_icc)
            base_n = cluster_adj["n_clustered_total"]
        attrition = apply_attrition_inflation(base_n, dropout_val / 100.0)
        if cluster_adj:
            cluster_adj["clusters_required_recruited"] = math.ceil(attrition["n_recruited"] / cluster_adj["cluster_size_m"])
        unpadded_info = calculate_unpadded_power_after_dropout(calc_results, dropout_val / 100.0)

        if enable_cluster and clust_icc > 0 and cluster_adj:
            if dropout_val == 0:
                cust_takeaway = (
                    f"Custom study sized at {power_val*100:.0f}% statistical power (two-sided alpha = {alpha_val}). "
                    f"Accounting for cluster randomization (cluster size m = {cluster_adj['cluster_size_m']}, "
                    f"ICC = {cluster_adj['icc']:.3f}, VIF = {cluster_adj['design_effect_vif']:.2f}), "
                    f"requires {base_n:,} evaluable participants across {cluster_adj['clusters_required']} clusters. "
                    f"With 0% anticipated loss-to-follow-up, the final enrollment target remains {base_n:,} participants."
                )
            else:
                cust_takeaway = (
                    f"Custom study sized at {power_val*100:.0f}% statistical power (two-sided alpha = {alpha_val}). "
                    f"Requires {base_n:,} evaluable participants across {cluster_adj['clusters_required']} clusters, "
                    f"inflated to {attrition['n_recruited']:,} recruitment target across {cluster_adj['clusters_required_recruited']} clusters "
                    f"to protect against {dropout_val}% loss-to-follow-up (power would drop to {unpadded_info['degraded_power_pct']:.1f}% if unpadded)."
                )
        else:
            if dropout_val == 0:
                cust_takeaway = (
                    f"Custom study sized at {power_val*100:.0f}% statistical power (two-sided alpha = {alpha_val}). "
                    f"Requires {base_n:,} evaluable participants. With 0% anticipated loss-to-follow-up, "
                    f"the final recruitment target remains {base_n:,} participants (no attrition cushion required)."
                )
            else:
                cust_takeaway = (
                    f"Custom study sized at {power_val*100:.0f}% statistical power (two-sided alpha = {alpha_val}). "
                    f"Requires {base_n:,} evaluable participants, inflated to {attrition['n_recruited']:,} recruitment target "
                    f"(+{attrition['additional_subjects_needed']:,} cushion) to protect against {dropout_val}% loss-to-follow-up "
                    f"(power would drop to {unpadded_info['degraded_power_pct']:.1f}% if unpadded)."
                )

        meta = {
            "scenario_key": "custom",
            "title": f"Custom Sizing: {custom_design}",
            "organisation": "All Meadows Cardiovascular Institute (Clinical Trials & Biostatistics Unit)",
            "domain": "Custom Clinical Trial & Epidemiological Investigation",
            "intervention": "Custom Evaluated Intervention",
            "comparator": "Standard Control Comparator",
            "primary_endpoint": endpoint_label,
            "description": "User-configured sample size and statistical power planning model.",
            "takeaway": cust_takeaway
        }
        demo_data = {
            "df": pd.DataFrame({"Simulated_Subject_ID": [f"SUBJ-{i:03d}" for i in range(1, 51)], "Assigned_Arm": ["Active" if i % 2 == 1 else "Control" for i in range(1, 51)]}),
            "meta": meta
        }

    # 3. Methodological Prerequisite Evaluation
    evaluations = evaluate_sample_size_prerequisites(calc_results)

    # 4. Header Banner & Prominent Synthetic Disclaimer
    st.markdown(f"## 📐 {meta['title']}")
    st.markdown(f"**Organisation:** {meta['organisation']} | **Domain:** {meta['domain']}")
    st.markdown(f"**Intervention:** `{meta['intervention']}` | **Comparator:** `{meta['comparator']}` | **Primary Endpoint:** `{meta['primary_endpoint']}`")
    st.markdown(f"*{meta['description']}*")

    with st.container(border=True):
        st.markdown(r"""
        > [!IMPORTANT]
        > **EDUCATIONAL & SYNTHETIC DATA DISCLAIMER**: The study scenarios, patient counts, biomarkers, and biostatistical metrics displayed below are **100% synthetic and computer-generated** purely for academic training, clinical trial design education, and software demonstration. Contains no real patient identifiers or protected health information.
        """)

    # 5. Methodological Prerequisite Cards
    render_sample_size_prerequisite_cards(evaluations)

    # 6. Primary Metric Dashboard
    st.markdown("### 📈 Biostatistical Sizing Dashboard & Recruitment Target")
    render_sample_size_metric_dashboard(calc_results, attrition, cluster_adj, unpadded_power_info=unpadded_info)

    # 7. Dual Plotly Visualizations
    col_v1, col_v2 = st.columns([1.1, 0.9])
    planned_n = cluster_adj["n_clustered_total"] if cluster_adj else calc_results.get("n_total", calc_results.get("required_total_screening_cohort", 100))
    planned_pwr = calc_results.get("achieved_power", calc_results.get("target_power", 0.85))

    with col_v1:
        with st.container(border=True):
            design_type_val = "two_means" if "cohen_d" in calc_results else ("survival" if "hazard_ratio" in calc_results else "two_proportions")
            fig_power = build_power_curve_plot(
                curve_data,
                planned_n=planned_n,
                planned_power=planned_pwr,
                design_type=design_type_val,
                recruited_n=attrition["n_recruited"],
                dropout_pct=attrition["dropout_rate_pct"],
                effective_n=unpadded_info["effective_evaluable_n"],
                unpadded_power=unpadded_info["degraded_power_pct"]
            )
            st.plotly_chart(fig_power, use_container_width=True)

    with col_v2:
        with st.container(border=True):
            if "cohen_d" in calc_results:
                mdes_label = "Cohen's d"
                planned_eff = calc_results["cohen_d"]
            elif "hazard_ratio" in calc_results:
                mdes_label = "ln(Hazard Ratio)"
                planned_eff = abs(calc_results.get("ln_hazard_ratio", math.log(calc_results["hazard_ratio"])))
            elif "odds_ratio" in calc_results:
                mdes_label = "Odds Ratio (Std Diff)"
                planned_eff = abs(math.log(max(0.01, calc_results["odds_ratio"])))
            elif "relative_risk" in calc_results:
                mdes_label = "Risk Difference"
                planned_eff = abs(calc_results.get("risk_difference", 0.10))
            else:
                mdes_label = "Standardized Effect"
                planned_eff = 0.50
            fig_mdes = build_mdes_curve_plot(curve_data, planned_n, planned_eff, effect_label=mdes_label)
            st.plotly_chart(fig_mdes, use_container_width=True)

    # 8. Protocol Justification Statement Card
    drop_pct_for_stmt = attrition.get("dropout_rate_pct", 15.0) / 100.0
    protocol_statement = generate_protocol_justification_statement(
        calc_results,
        primary_endpoint_label=meta["primary_endpoint"],
        dropout_rate=drop_pct_for_stmt,
        cluster_icc=cluster_adj.get("icc") if cluster_adj else None,
        cluster_size=cluster_adj.get("cluster_size_m", cluster_adj.get("cluster_size")) if cluster_adj else None
    )
    render_sample_size_protocol_card(protocol_statement, calc_results)

    # 9. Sensitivity Grid
    render_sample_size_sensitivity_grid(
        curve_data,
        planned_n,
        calc_results=calc_results,
        current_dropout_pct=attrition.get("dropout_rate_pct", 15.0)
    )

    # 10. Full 4-Tab Educational Architecture
    st.markdown("---")
    render_sample_size_educational_tabs(calc_results, meta, demo_data, curve_data, attrition, cluster_adj)


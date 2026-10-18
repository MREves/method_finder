"""Interactive UI Components & Visualizations for Cochran-Mantel-Haenszel (CMH) Stratified Analysis & Effect Modification.

Provides:
1. Plain English Pedagogical Introductory Guide to Stratified Analysis, Confounding, Simpson's Paradox,
   and Effect Measure Modification (Zero prior statistical knowledge).
2. Sidebar scenario controls with 3 realistic clinical scenarios and custom cohort exploration.
3. Methodological Prerequisite Evaluation (Homogeneity of Odds Ratios, Confounder Evaluation via 10% rule,
   Simpson's Paradox detection, Stratum cell density, and Conditional Independence).
4. Primary Metric Dashboard (Crude OR, Mantel-Haenszel Pooled OR with Robins-Breslow-Greenland CI,
   Confounding Delta %, Breslow-Day Homogeneity p-value, CMH Chi-Square p-value).
5. Plotly Stratified Forest Plot (stratum ORs, pooled MH diamond, crude reference line) and
   Stratified Event Rate Comparison Plot illustrating Simpson's reversal / confounding distortion.
6. Full-Width Pedagogical Explanatory Guide beneath plots explaining how to interpret forest plots and effect modification.
7. Clinical Strata Performance & Weights Table (Observed cells, stratum risks, Woolf CIs, and MH percentage weights).
8. Full 4-Tab Educational Architecture:
   - 📋 Rationale & Epidemiological Framework (Robins-Breslow-Greenland variance, Greenland-Robins RR, Breslow-Day/Tarone quadratic, STROBE).
   - 📗 Microsoft Excel Implementation Guide (Collapsible expanders for each step, formulas, chart construction).
   - 🐍 Python Production Script Guide (Collapsible expanders for each stage, clean reusable code).
   - 💾 Live Synthetic Data Downloads (Patient microdata CSV, strata contingency tables CSV, and MH diagnostic summary CSV).
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Union
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.mantel_haenszel import (
    calculate_mantel_haenszel,
    evaluate_mantel_haenszel_prerequisites,
    generate_mantel_haenszel_demo_data
)


def render_mantel_haenszel_plain_english_guide():
    """Renders a collapsible plain English introductory guide to stratified analysis and effect modification."""
    with st.expander("📖 Zero-Knowledge Guide: What is Cochran-Mantel-Haenszel (CMH) Stratified Analysis?", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. The Epidemiological Dilemma: Why Crude Associations Mislead
            * When studying the relationship between an exposure (e.g. a new medical drug or lifestyle habit) and a health outcome (e.g. recovery or mortality), looking at the overall population average—the **Crude Table**—can be deeply misleading.
            * A third factor—a **Confounder**—may be unevenly distributed between the exposed and unexposed groups. For example, if older or more severely ill patients are preferentially given a new drug, the drug might falsely look ineffective or even dangerous!
            * **Stratification** is the foundational epidemiological technique of slicing the cohort into homogeneous subgroups (strata)—such as Intensive Care Unit vs. General Ward, or Young vs. Old—so the exposure and outcome can be compared fairly within like-for-like groups.
            """)

            st.markdown(r"""
            #### 2. Simpson's Paradox: The Ultimate Statistical Illusion
            * **Simpson's Paradox** is an extreme mathematical phenomenon where a treatment appears to increase mortality in the overall crude population, yet simultaneously **reduces** mortality in *every single subgroup*!
            * How is this possible? Because the baseline risk of death in the ICU is $50\%$, while in the general ward it is only $15\%$. If $70\%$ of patients given the drug were in the ICU (indication bias), the crude mortality is artificially dragged upward by the high-risk setting.
            * Stratifying by admission ward immediately exposes this illusion: within the ICU, the drug cuts death odds by $55\%$; within the general ward, the drug also cuts death odds by $55\%$!
            * Failing to perform stratified analysis in this scenario would cause clinicians to abandon a life-saving therapy!
            """)

            st.markdown(r"""
            #### 3. How the Mantel-Haenszel Summary Works
            * Once you have divided your study into $K$ separate $2 \times 2$ tables, you need a single, mathematically rigorous summary estimate of the true adjusted effect.
            * In 1959, epidemiologists **Nathan Mantel and William Haenszel** developed a precision-weighted pooling method:
              $$\text{OR}_{MH} = \frac{\sum_{k=1}^K \frac{a_k d_k}{n_k}}{\sum_{k=1}^K \frac{b_k c_k}{n_k}}$$
            * Rather than weighting by sample size alone, Mantel-Haenszel assigns the greatest weight to strata that provide the most statistical information, while protecting against small-cell distortion.
            """)

        with col_g2:
            st.markdown(r"""
            #### 4. Confounding vs. Effect Measure Modification (Interaction)
            Epidemiology makes a sharp, critical distinction between two completely different biological phenomena:
            * **Confounding (A Distortion to be Controlled):**
              * The stratifying variable distorts the crude estimate.
              * When stratified, the odds ratio is **the same across all strata** ($\text{OR}_1 \approx \text{OR}_2$).
              * We *want* to remove confounding by reporting the single adjusted summary $\text{OR}_{MH}$.
              * **The 10% Rule:** If $|\text{OR}_{\text{crude}} - \text{OR}_{MH}| / \text{OR}_{MH} \ge 10\%$, confounding is clinically meaningful and the adjusted $\text{OR}_{MH}$ must be reported.
            * **Effect Measure Modification / Interaction (A Biological Reality to be Reported):**
              * The true effect of the exposure **differs** depending on the stratum ($\text{OR}_1 \ne \text{OR}_2$).
              * For example, an industrial toxin has an odds ratio of $1.5$ in non-smokers, but an odds ratio of $6.9$ in smokers (synergy).
              * **Statistical Rule:** Pooling heterogeneous strata into a single $\text{OR}_{MH}$ is scientifically invalid! You must report the stratum-specific effects separately.
            """)

            st.markdown(r"""
            #### 5. The Breslow-Day Test for Homogeneity
            * How do you objectively prove whether odds ratios are homogeneous or if effect modification is present?
            * The **Breslow-Day Test** (with Tarone's modern degrees-of-freedom adjustment) statistically tests the null hypothesis:
              $$H_0: \text{OR}_1 = \text{OR}_2 = \dots = \text{OR}_K = \text{OR}_{MH}$$
            * If $p \ge 0.05$: Strata are homogeneous. Pooling into a single summary $\text{OR}_{MH}$ is valid.
            * If $p < 0.05$: Effect measure modification is present. Do **not** pool; report each stratum independently.
            """)

            st.markdown(r"""
            #### 6. STROBE Guidelines for Reporting Stratified Analyses
            * Under the international **STROBE** statement (Strengthening the Reporting of Observational Studies in Epidemiology, Item 16):
              * Authors must present both unadjusted (crude) and confounder-adjusted estimates with 95% confidence intervals.
              * If interaction / effect modification was investigated, authors must present stratum-specific estimates rather than hiding heterogeneity behind a pooled summary.
            """)


def render_mantel_haenszel_prerequisite_cards(evaluations: Dict[str, Any]):
    """Renders diagnostic prerequisite cards based on STROBE / epidemiological standards."""
    checks = evaluations.get("checks", [])
    if not checks:
        return

    st.markdown("### 🔍 Methodological & Epidemiological Prerequisite Checks")
    st.caption("Verifying homogeneity of odds ratios, confounding magnitude, Simpson's paradox, cell density, and conditional association.")

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
                with st.expander("Details", expanded=False):
                    st.write(explanation)


def render_mantel_haenszel_metric_dashboard(results: Dict[str, Any]):
    """Renders the top-level metric cards for crude vs adjusted OR, confounding, and tests."""
    c1, c2, c3, c4, c5 = st.columns(5)

    crude_or = results.get("crude_or", 1.0)
    crude_ci = results.get("crude_or_ci_str", "")
    or_mh = results.get("or_mh", 1.0)
    or_mh_ci = results.get("or_mh_ci_str", "")
    delta_pct = results.get("confounding_delta_pct", 0.0)
    bd_p = results.get("breslow_day_p_value", 1.0)
    cmh_p = results.get("cmh_p_value", 1.0)
    cmh_stat = results.get("cmh_statistic_corrected", 0.0)
    simpson = results.get("simpsons_paradox_detected", False)

    with c1:
        with st.container(border=True):
            st.metric(
                label="Crude Odds Ratio",
                value=f"{crude_or:.2f}",
                delta="Unadjusted"
            )
            st.caption(f"95% CI: [{results.get('crude_or_ci_lower', 1.0):.2f}, {results.get('crude_or_ci_upper', 1.0):.2f}]")

    with c2:
        with st.container(border=True):
            st.metric(
                label="Mantel-Haenszel OR (Pooled)",
                value=f"{or_mh:.2f}",
                delta=f"{delta_pct:+.1f}% vs. Crude",
                delta_color="inverse" if simpson else "normal"
            )
            st.caption(f"RBG 95% CI: [{results.get('or_mh_ci_lower', 1.0):.2f}, {results.get('or_mh_ci_upper', 1.0):.2f}]")

    with c3:
        with st.container(border=True):
            badge = "🚨 Simpson's Inversion" if simpson else ("⚠️ Confounded" if delta_pct >= 10.0 else "✅ Negligible")
            st.metric(
                label="Confounding Magnitude (Δ%)",
                value=f"{delta_pct:.1f}%",
                delta=badge
            )
            st.caption("10% Rule: |Crude - OR_MH| / OR_MH")

    with c4:
        with st.container(border=True):
            homog_badge = "❌ Heterogeneous (Interaction)" if bd_p < 0.05 else "✅ Homogeneous (Valid Pool)"
            st.metric(
                label="Breslow-Day Homogeneity p",
                value=f"{bd_p:.4f}",
                delta=homog_badge
            )
            st.caption(f"Tarone χ² = {results.get('breslow_day_tarone_statistic', 0.0):.2f} (df={results.get('breslow_day_df', 1)})")

    with c5:
        with st.container(border=True):
            sig_badge = "✅ Significant (p < 0.05)" if cmh_p < 0.05 else "⚠️ Not Significant"
            st.metric(
                label="CMH χ² Test p-value",
                value=f"{cmh_p:.4f}",
                delta=sig_badge
            )
            st.caption(f"Corrected χ² = {cmh_stat:.2f} (df=1)")

    # Clinical Recommendation Alert
    recommendation = results.get("clinical_recommendation", "")
    if results.get("effect_modification_present", False):
        st.error(recommendation)
    elif simpson:
        st.error(recommendation)
    elif results.get("confounding_present", False):
        st.warning(recommendation)
    else:
        st.success(recommendation)


def build_stratified_forest_plot(results: Dict[str, Any], meta: Dict[str, Any]) -> go.Figure:
    """Builds a high-precision Plotly Stratified Forest Plot comparing stratum ORs, crude OR, and pooled MH OR."""
    strata_list = results.get("strata_list", [])
    crude_or = results.get("crude_or", 1.0)
    crude_lower = results.get("crude_or_ci_lower", 1.0)
    crude_upper = results.get("crude_or_ci_upper", 1.0)
    or_mh = results.get("or_mh", 1.0)
    or_mh_lower = results.get("or_mh_ci_lower", 1.0)
    or_mh_upper = results.get("or_mh_ci_upper", 1.0)

    # Reverse order so first stratum appears at the top
    labels = []
    ors = []
    ci_lowers = []
    ci_uppers = []
    colors = []
    text_labels = []

    for s in reversed(strata_list):
        labels.append(f"{s['Stratum']} (N={s['N_Stratum']})")
        ors.append(s["OR_Stratum"])
        ci_lowers.append(s["OR_CI_Lower"])
        ci_uppers.append(s["OR_CI_Upper"])
        colors.append("#2563EB")  # Blue for strata
        text_labels.append(f"OR {s['OR_Stratum']:.2f} [{s['OR_CI_Lower']:.2f}, {s['OR_CI_Upper']:.2f}] (Weight: {s['Weight_MH_Pct']}%)")

    # Add Crude Row
    labels.append("Crude Overall (Unadjusted)")
    ors.append(crude_or)
    ci_lowers.append(crude_lower)
    ci_uppers.append(crude_upper)
    colors.append("#DC2626")  # Red for crude
    text_labels.append(f"Crude OR {crude_or:.2f} [{crude_lower:.2f}, {crude_upper:.2f}] (Unadjusted)")

    # Add Pooled MH Row
    labels.append("Mantel-Haenszel Summary (Pooled OR)")
    ors.append(or_mh)
    ci_lowers.append(or_mh_lower)
    ci_uppers.append(or_mh_upper)
    colors.append("#059669")  # Green for MH summary
    text_labels.append(f"OR_MH {or_mh:.2f} [{or_mh_lower:.2f}, {or_mh_upper:.2f}] (RBG 95% CI)")

    fig = go.Figure()

    # Vertical reference line at null effect (OR = 1.0)
    fig.add_shape(
        type="line",
        x0=1.0, x1=1.0,
        y0=-0.5, y1=len(labels) - 0.5,
        line=dict(color="#6B7280", width=1.5, dash="dash")
    )

    # Vertical reference line at Crude OR (to visually illustrate confounding shift)
    fig.add_shape(
        type="line",
        x0=crude_or, x1=crude_or,
        y0=-0.5, y1=len(labels) - 0.5,
        line=dict(color="#DC2626", width=1.2, dash="dot")
    )

    # Error bars for stratum and crude rows
    error_x_minus = [ors[i] - ci_lowers[i] for i in range(len(ors))]
    error_x_plus = [ci_uppers[i] - ors[i] for i in range(len(ors))]

    # Stratum points and lines
    fig.add_trace(go.Scatter(
        x=ors[:-1],
        y=labels[:-1],
        mode="markers+text",
        error_x=dict(
            type="data",
            symmetric=False,
            array=error_x_plus[:-1],
            arrayminus=error_x_minus[:-1],
            color="#374151",
            thickness=2,
            width=6
        ),
        marker=dict(
            size=11,
            color=colors[:-1],
            symbol=["square" if "Crude" not in l else "circle" for l in labels[:-1]]
        ),
        text=[f"  {ors[i]:.2f}" for i in range(len(ors)-1)],
        textposition="top right",
        textfont=dict(size=11),
        hoverinfo="text",
        hovertext=text_labels[:-1],
        name="Strata & Crude Estimates"
    ))

    # Pooled MH Diamond / Marker at the bottom
    fig.add_trace(go.Scatter(
        x=[or_mh],
        y=[labels[-1]],
        mode="markers+text",
        error_x=dict(
            type="data",
            symmetric=False,
            array=[or_mh_upper - or_mh],
            arrayminus=[or_mh - or_mh_lower],
            color="#059669",
            thickness=3.5,
            width=10
        ),
        marker=dict(
            size=14,
            color="#059669",
            symbol="diamond"
        ),
        text=[f"  {or_mh:.2f} (Summary)"],
        textposition="top right",
        textfont=dict(size=12, color="#059669"),
        hoverinfo="text",
        hovertext=[text_labels[-1]],
        name="Mantel-Haenszel Pooled OR"
    ))

    # Compute reasonable x-axis limits on log-scale
    all_vals = [v for v in ci_lowers + ci_uppers + ors + [1.0, crude_or, or_mh] if v > 0]
    min_x = max(0.05, min(all_vals) * 0.7)
    max_x = min(30.0, max(all_vals) * 1.4)

    fig.update_layout(
        title=dict(
            text=f"<b>Stratified Forest Plot: Odds Ratios & Confounder Shift</b><br><sup>{meta.get('title', 'Mantel-Haenszel Analysis')} — Blue: Strata | Red: Crude | Green: MH Pooled</sup>",
            font=dict(size=14)
        ),
        xaxis=dict(
            title="Odds Ratio (Logarithmic Scale)",
            type="log",
            range=[math.log10(min_x), math.log10(max_x)],
            gridcolor="#E5E7EB",
            zeroline=False
        ),
        yaxis=dict(
            title="",
            gridcolor="#F3F4F6"
        ),
        height=380 + (len(strata_list) * 45),
        margin=dict(l=220, r=40, t=70, b=50),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def build_strata_rate_comparison_plot(results: Dict[str, Any], meta: Dict[str, Any]) -> go.Figure:
    """Builds a grouped bar chart comparing Exposed vs. Unexposed event rates across strata and crude."""
    strata_list = results.get("strata_list", [])
    crude_table = results.get("crude_table", {})

    crude_exp_rate = (crude_table.get("a", 0) / max(1, crude_table.get("n1", 1))) * 100.0
    crude_unexp_rate = (crude_table.get("c", 0) / max(1, crude_table.get("n0", 1))) * 100.0

    categories = [s["Stratum"] for s in strata_list] + ["Crude Overall (Unadjusted)"]
    exp_rates = [s["Exposed_Risk_Pct"] for s in strata_list] + [round(crude_exp_rate, 2)]
    unexp_rates = [s["Unexposed_Risk_Pct"] for s in strata_list] + [round(crude_unexp_rate, 2)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Exposed Group Event Rate (%)",
        x=categories,
        y=exp_rates,
        marker_color="#2563EB",
        text=[f"{r:.1f}%" for r in exp_rates],
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        name="Unexposed / Control Event Rate (%)",
        x=categories,
        y=unexp_rates,
        marker_color="#9CA3AF",
        text=[f"{r:.1f}%" for r in unexp_rates],
        textposition="outside"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Within-Stratum vs. Crude Disease Event Rates (%)</b><br><sup>Comparing Exposed vs. Unexposed risk across strata reveals confounding distortion or Simpson's reversal</sup>",
            font=dict(size=14)
        ),
        barmode="group",
        xaxis=dict(title="Stratum / Cohort Setting", gridcolor="#F3F4F6"),
        yaxis=dict(title="Observed Disease Event Rate (%)", range=[0, max(exp_rates + unexp_rates) * 1.25], gridcolor="#E5E7EB"),
        height=380,
        margin=dict(l=50, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def render_stratified_explanatory_box(results: Dict[str, Any]):
    """Renders a full-width pedagogical explanatory guide beneath the plots."""
    with st.container(border=True):
        st.markdown("### 📊 How to Interpret the Forest Plot & Rate Comparison")
        st.markdown(r"""
        * **1. The Blue Strata Squares & Error Bars:**
          * Each blue square marks the stratum-specific Odds Ratio ($\text{OR}_k$). The horizontal line shows its Woolf 95% Confidence Interval.
          * If an error bar stays entirely to the left of the dashed gray line ($\text{OR} < 1.0$), the exposure is statistically protective within that specific subgroup.
          * If the blue squares align vertically with each other (e.g. both near $0.44$), the true biological effect is **homogeneous** across subgroups.
        * **2. The Red Crude Line vs. The Green MH Diamond:**
          * The vertical red dotted line shows the **Crude (unadjusted) Odds Ratio**.
          * The green diamond shows the **Mantel-Haenszel Summary Pooled OR** ($\text{OR}_{MH}$) with its Robins-Breslow-Greenland 95% CI.
          * **Confounding Shift:** The distance between the red crude line and the green diamond represents the magnitude of confounding removed by stratification ($\Delta\%$).
          * **Simpson's Paradox:** In our ICU scenario, notice how the red crude line lies on the harmful side ($\text{OR}_{\text{crude}} = 1.34 > 1.0$), while the green diamond and all blue strata lie on the protective side ($\text{OR} = 0.44 < 1.0$)!
        * **3. The Rate Comparison Bar Chart:**
          * Observe how the crude rate blends disparate clinical populations. Because the high-mortality ICU stratum had far more treated patients, the crude treated bar is pulled upward, hiding the fact that the treated bar is lower than the control bar in *both* individual settings.
        """)


def render_strata_comparison_table(results: Dict[str, Any]):
    """Renders a comprehensive contingency table breakdown across all strata."""
    st.markdown("### 📋 Stratum-by-Stratum Contingency Breakdown & Mantel-Haenszel Weights")
    st.caption("Observed 2x2 cell counts, stratum risks, Woolf odds ratio confidence intervals, and normalized Mantel-Haenszel percentage weights.")

    strata_list = results.get("strata_list", [])
    crude_table = results.get("crude_table", {})
    crude_or = results.get("crude_or", 1.0)
    crude_ci = results.get("crude_or_ci_str", "")
    crude_rr = results.get("crude_rr", 1.0)
    crude_rr_ci = results.get("crude_rr_ci_str", "")
    or_mh = results.get("or_mh", 1.0)
    or_mh_ci = results.get("or_mh_ci_str", "")
    rr_mh = results.get("rr_mh", 1.0)
    rr_mh_ci = results.get("rr_mh_ci_str", "")

    table_data = []
    for s in strata_list:
        table_data.append({
            "Stratum": s["Stratum"],
            "Exposed Cases (a)": s["Exposed_Cases"],
            "Exposed Non-Cases (b)": s["Exposed_NonCases"],
            "Unexposed Cases (c)": s["Unexposed_Cases"],
            "Unexposed Non-Cases (d)": s["Unexposed_NonCases"],
            "Total N": s["N_Stratum"],
            "Exposed Risk": f"{s['Exposed_Risk_Pct']:.1f}%",
            "Unexposed Risk": f"{s['Unexposed_Risk_Pct']:.1f}%",
            "Stratum OR (95% CI)": s["OR_95_CI_Str"],
            "Stratum RR (95% CI)": s["RR_95_CI_Str"],
            "MH Weight (%)": f"{s['Weight_MH_Pct']:.1f}%"
        })

    # Add Crude row
    n1_c = crude_table.get("n1", 1)
    n0_c = crude_table.get("n0", 1)
    exp_risk_c = (crude_table.get("a", 0) / max(1, n1_c)) * 100.0
    unexp_risk_c = (crude_table.get("c", 0) / max(1, n0_c)) * 100.0

    table_data.append({
        "Stratum": "CRUDE TOTAL (Unadjusted)",
        "Exposed Cases (a)": crude_table.get("a", 0),
        "Exposed Non-Cases (b)": crude_table.get("b", 0),
        "Unexposed Cases (c)": crude_table.get("c", 0),
        "Unexposed Non-Cases (d)": crude_table.get("d", 0),
        "Total N": crude_table.get("n_total", 0),
        "Exposed Risk": f"{exp_risk_c:.1f}%",
        "Unexposed Risk": f"{unexp_risk_c:.1f}%",
        "Stratum OR (95% CI)": crude_ci,
        "Stratum RR (95% CI)": crude_rr_ci,
        "MH Weight (%)": "—"
    })

    # Add Summary MH row
    table_data.append({
        "Stratum": "MANTEL-HAENSZEL POOLED (Adjusted)",
        "Exposed Cases (a)": crude_table.get("a", 0),
        "Exposed Non-Cases (b)": crude_table.get("b", 0),
        "Unexposed Cases (c)": crude_table.get("c", 0),
        "Unexposed Non-Cases (d)": crude_table.get("d", 0),
        "Total N": crude_table.get("n_total", 0),
        "Exposed Risk": f"—",
        "Unexposed Risk": f"—",
        "Stratum OR (95% CI)": or_mh_ci,
        "Stratum RR (95% CI)": rr_mh_ci,
        "MH Weight (%)": "100.0%"
    })

    df_display = pd.DataFrame(table_data)
    st.dataframe(df_display, use_container_width=True, hide_index=True)


def render_mantel_haenszel_educational_tabs(
    results: Dict[str, Any],
    meta: Dict[str, Any],
    demo_data: Dict[str, Any]
):
    """Renders the comprehensive 4-Tab Educational Architecture."""
    tab_rat, tab_xls, tab_py, tab_dl = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Microsoft Excel Implementation Guide",
        "🐍 Python Production Script Guide",
        "💾 Live Synthetic Data Downloads"
    ])

    # -------------------------------------------------------------
    # TAB 1: Rationale & Epidemiological Framework
    # -------------------------------------------------------------
    with tab_rat:
        st.markdown(r"""
        ### 📋 Epidemiological & Mathematical Foundations of Mantel-Haenszel Analysis

        The Cochran-Mantel-Haenszel (CMH) method is the cornerstone of non-parametric confounding control
        and interaction testing in observational epidemiology across stratified $2 \times 2$ contingency tables.
        """)

        with st.expander("1. Mathematical Formulation of the Stratified 2x2 Contingency Table", expanded=False):
            st.markdown(r"""
            For each stratum $k \in \{1, 2, \dots, K\}$ formed by dividing the study cohort by the stratifying variable:

            | Exposure Status | Disease Event ($D=1$) | No Event ($D=0$) | Total Margin |
            | :--- | :---: | :---: | :---: |
            | **Exposed ($E=1$)** | $a_k$ | $b_k$ | $n_{1k} = a_k + b_k$ |
            | **Unexposed ($E=0$)** | $c_k$ | $d_k$ | $n_{0k} = c_k + d_k$ |
            | **Total Margin** | $m_{1k} = a_k + c_k$ | $m_{0k} = b_k + d_k$ | $n_k = n_{1k} + n_{0k}$ |

            * **Stratum-Specific Odds Ratio ($\text{OR}_k$):**
              $$\text{OR}_k = \frac{a_k d_k}{b_k c_k}$$
            * **Woolf's Logit Standard Error & Greenland 95% Confidence Interval:**
              $$\text{SE}(\ln \text{OR}_k) = \sqrt{\frac{1}{a_k} + \frac{1}{b_k} + \frac{1}{c_k} + \frac{1}{d_k}}$$
              $$\text{CI}_{95\%}(\text{OR}_k) = \exp\left(\ln \text{OR}_k \pm 1.96 \times \text{SE}(\ln \text{OR}_k)\right)$$
            * *(Note: When any cell contains zero, Haldane's $0.5$ continuity correction is added to all four cells of that stratum).*
            """)

        with st.expander("2. Mantel-Haenszel Summary Pooled Odds Ratio & Robins-Breslow-Greenland Variance", expanded=False):
            st.markdown(r"""
            The Mantel-Haenszel summary pooled odds ratio ($\text{OR}_{MH}$) weights each stratum by the precision factor $b_k c_k / n_k$:

            $$\text{OR}_{MH} = \frac{\sum_{k=1}^K R_k}{\sum_{k=1}^K S_k} = \frac{\sum_{k=1}^K \frac{a_k d_k}{n_k}}{\sum_{k=1}^K \frac{b_k c_k}{n_k}}$$

            #### Robins-Breslow-Greenland (RBG 1986) Asymptotic Variance
            To construct valid $95\%$ confidence intervals that remain robust under both large-stratum asymptotic sampling and sparse-data (many small strata) conditions, **Robins, Breslow, and Greenland (1986)** derived the landmark triple-term variance:

            $$\text{Var}(\ln \text{OR}_{MH}) = \frac{\sum_{k=1}^K P_k R_k}{2 R_+^2} + \frac{\sum_{k=1}^K (P_k S_k + Q_k R_k)}{2 R_+ S_+} + \frac{\sum_{k=1}^K Q_k S_k}{2 S_+^2}$$

            Where:
            * $R_k = \frac{a_k d_k}{n_k}$, \quad $S_k = \frac{b_k c_k}{n_k}$, \quad $R_+ = \sum R_k$, \quad $S_+ = \sum S_k$
            * $P_k = \frac{a_k + d_k}{n_k}$, \quad $Q_k = \frac{b_k + c_k}{n_k}$

            The two-sided $95\%$ confidence interval is computed on the natural log scale and exponentiated:
            $$\text{CI}_{95\%}(\text{OR}_{MH}) = \exp\left(\ln \text{OR}_{MH} \pm 1.96 \times \sqrt{\text{Var}(\ln \text{OR}_{MH})}\right)$$
            """)

        with st.expander("3. Cochran-Mantel-Haenszel Chi-Square Test of Conditional Independence", expanded=False):
            st.markdown(r"""
            Under the null hypothesis $H_0: \text{OR}_1 = \text{OR}_2 = \dots = \text{OR}_K = 1.0$ (exposure is conditionally independent of outcome given the stratifier), the cell count $a_k$ follows a hypergeometric distribution conditional on fixed marginals $n_{1k}, n_{0k}, m_{1k}, m_{0k}$:

            * **Hypergeometric Expected Cell Count:**
              $$E(a_k) = \frac{n_{1k} m_{1k}}{n_k}$$
            * **Hypergeometric Variance:**
              $$\text{Var}(a_k) = \frac{n_{1k} n_{0k} m_{1k} m_{0k}}{n_k^2 (n_k - 1)}$$

            #### CMH Chi-Square Statistic with Continuity Correction:
            $$\chi^2_{CMH} = \frac{\left(\left|\sum_{k=1}^K a_k - \sum_{k=1}^K E(a_k)\right| - 0.5\right)^2}{\sum_{k=1}^K \text{Var}(a_k)}, \quad \text{df} = 1$$

            If $\chi^2_{CMH} > 3.841$ ($p < 0.05$), the null hypothesis of conditional independence is rejected: exposure and outcome remain significantly associated after controlling for the stratifying variable.
            """)

        with st.expander("4. Breslow-Day & Tarone Homogeneity Tests for Effect Measure Modification", expanded=False):
            st.markdown(r"""
            Before relying on the summary $\text{OR}_{MH}$, the epidemiologist must test the assumption of **homogeneity of odds ratios** across strata:
            $$H_0: \text{OR}_1 = \text{OR}_2 = \dots = \text{OR}_K = \text{OR}_{MH}$$

            For each stratum, the Breslow-Day test solves the quadratic equation for the expected cell count $A_k^*$ under $H_0$:
            $$\frac{A_k^* (n_{0k} - m_{1k} + A_k^*)}{(n_{1k} - A_k^*) (m_{1k} - A_k^*)} = \text{OR}_{MH}$$
            $$\text{Standard form: } (\text{OR}_{MH} - 1) (A_k^*)^2 - \left[(\text{OR}_{MH} - 1)(n_{1k} + m_{1k}) + n_k\right] A_k^* + \text{OR}_{MH} n_{1k} m_{1k} = 0$$

            #### Breslow-Day Statistic:
            $$\chi^2_{BD} = \sum_{k=1}^K \frac{(a_k - A_k^*)^2}{\text{Var}(A_k^*)}, \quad \text{where } \text{Var}(A_k^*) = \left(\frac{1}{A_k^*} + \frac{1}{n_{1k} - A_k^*} + \frac{1}{m_{1k} - A_k^*} + \frac{1}{n_{0k} - m_{1k} + A_k^*}\right)^{-1}$$

            #### Tarone's Asymptotic Adjustment:
            Because $\text{OR}_{MH}$ is an estimated rather than known true parameter, Breslow-Day slightly inflates the type-I error rate. **Tarone (1985)** derived the exact adjustment:
            $$\chi^2_{\text{Tarone}} = \chi^2_{BD} - \frac{\left(\sum a_k - \sum A_k^*\right)^2}{\sum \text{Var}(A_k^*)}, \quad \text{df} = K - 1$$

            * **If $p \ge 0.05$:** The null hypothesis of homogeneous odds ratios is retained. Stratum variation is attributable to random sampling noise. Reporting $\text{OR}_{MH}$ is methodologically sound.
            * **If $p < 0.05$:** Effect Measure Modification (Interaction) is statistically established. The exposure effect genuinely varies by stratum. Pooling is scientifically invalid; report stratum-specific estimates.
            """)

        with st.expander("5. STROBE Statement Guidelines for Observational Epidemiology", expanded=False):
            st.markdown(r"""
            The **STROBE** statement (Strengthening the Reporting of Observational Studies in Epidemiology) provides international standards for reporting stratified cohort and case-control studies:
            * **Item 16 (Main Results):**
              * Give unadjusted estimates and, if applicable, confounder-adjusted estimates and their precision (e.g., 95% confidence intervals). Make clear which confounders were adjusted for and why they were included.
            * **Item 17 (Other Analyses):**
              * Report other analyses done—e.g. analyses of subgroups and interactions, and sensitivity analyses.
            * **Best Practice:** When reporting stratified analyses, present the crude table, all stratum-specific tables, the test of homogeneity (Breslow-Day / Tarone), and either the pooled summary (if homogeneous) or separate subgroup interpretations (if modified).
            """)

    # -------------------------------------------------------------
    # TAB 2: Microsoft Excel Step-by-Step Implementation Guide
    # -------------------------------------------------------------
    with tab_xls:
        st.markdown("""
        ### 📗 Step-by-Step Implementation in Microsoft Excel

        This tutorial guides you through building a complete Cochran-Mantel-Haenszel analysis workbook in Microsoft Excel,
        including stratum-specific Woolf confidence intervals, Robins-Breslow-Greenland variance, CMH Chi-Square, and a forest plot.
        """)

        with st.expander("Step 1: Setting Up Stratified 2x2 Contingency Tables in Excel", expanded=False):
            st.markdown("""
            1. Open a new Excel worksheet and set up the following column headers across row 3:
               * `A3`: `Stratum_ID`
               * `B3`: `Stratum_Name`
               * `C3`: `a_Exp_Cases`
               * `D3`: `b_Exp_NonCases`
               * `E3`: `c_Unexp_Cases`
               * `F3`: `d_Unexp_NonCases`
               * `G3`: `n1_Exposed_Total` (`=C4+D4`)
               * `H3`: `n0_Unexposed_Total` (`=E4+F4`)
               * `I3`: `m1_Cases_Total` (`=C4+E4`)
               * `J3`: `m0_NonCases_Total` (`=D4+F4`)
               * `K3`: `n_Stratum_Total` (`=G4+H4`)
            2. In rows 4 and 5, enter the observed cell counts for Stratum 1 (ICU) and Stratum 2 (General Ward).
            3. In row 6, compute the crude totals using `=SUM(C4:C5)`, `=SUM(D4:D5)`, etc.
            """)

        with st.expander("Step 2: Calculating Stratum-Specific Odds Ratios & Woolf 95% Confidence Intervals", expanded=False):
            st.markdown("""
            1. In column `L4`, compute the Stratum Odds Ratio:
               `=(C4*F4)/(D4*E4)`
            2. In column `M4`, compute Woolf's standard error of the log-odds ratio:
               `=SQRT(1/C4 + 1/D4 + 1/E4 + 1/F4)`
            3. In column `N4`, compute the 95% Lower Confidence Limit:
               `=EXP(LN(L4) - 1.95996 * M4)`
            4. In column `O4`, compute the 95% Upper Confidence Limit:
               `=EXP(LN(L4) + 1.95996 * M4)`
            5. Drag formulas down for all strata and for the Crude row in row 6.
            """)

        with st.expander("Step 3: Calculating Mantel-Haenszel Numerator (R_k) and Denominator (S_k)", expanded=False):
            st.markdown("""
            1. In column `P4`, compute the Mantel-Haenszel numerator term $R_k = (a_k \times d_k) / n_k$:
               `=(C4*F4)/K4`
            2. In column `Q4`, compute the Mantel-Haenszel denominator term $S_k = (b_k \times c_k) / n_k$:
               `=(D4*E4)/K4`
            3. In row 7, sum columns `P` and `Q`:
               * Cell `P7`: `=SUM(P4:P5)` (This is $R_+$)
               * Cell `Q7`: `=SUM(Q4:Q5)` (This is $S_+$)
            4. In cell `P8`, compute the Pooled Mantel-Haenszel Odds Ratio:
               `=P7/Q7`
            """)

        with st.expander("Step 4: Computing Robins-Breslow-Greenland (RBG) Standard Error & 95% CI", expanded=False):
            st.markdown("""
            1. To compute the triple-term RBG (1986) variance, create three helper columns:
               * `R4` (Term 1 numerator $P_k \times R_k$): `=((C4+F4)/K4)*P4`
               * `S4` (Term 2 numerator $P_k \times S_k + Q_k \times R_k$): `=((C4+F4)/K4)*Q4 + ((D4+E4)/K4)*P4`
               * `T4` (Term 3 numerator $Q_k \times S_k$): `=((D4+E4)/K4)*Q4`
            2. In row 7, sum each column:
               * `R7`: `=SUM(R4:R5)`
               * `S7`: `=SUM(S4:S5)`
               * `T7`: `=SUM(T4:T5)`
            3. In cell `P9`, compute the asymptotic variance $\\text{Var}(\\ln \\text{OR}_{MH})$:
               `=(R7/(2*(P7^2))) + (S7/(2*P7*Q7)) + (T7/(2*(Q7^2)))`
            4. In cell `P10`, compute the Standard Error:
               `=SQRT(P9)`
            5. In cells `P11` and `P12`, compute the summary 95% CI:
               * Lower 95%: `=EXP(LN(P8) - 1.95996 * P10)`
               * Upper 95%: `=EXP(LN(P8) + 1.95996 * P10)`
            """)

        with st.expander("Step 5: Computing Cochran-Mantel-Haenszel Chi-Square (χ²_CMH) & p-value", expanded=False):
            st.markdown("""
            1. In column `U4`, compute the Hypergeometric Expected Count $E(a_k)$:
               `=(G4*I4)/K4`
            2. In column `V4`, compute the Hypergeometric Variance $\\text{Var}(a_k)$:
               `=(G4*H4*I4*J4)/((K4^2)*(K4-1))`
            3. In cell `U7`, compute sum of observed cases: `=SUM(C4:C5)`
            4. In cell `U8`, compute sum of expected cases: `=SUM(U4:U5)`
            5. In cell `V8`, compute sum of variances: `=SUM(V4:V5)`
            6. In cell `U9`, compute the continuity-corrected CMH Chi-Square:
               `=((ABS(U7 - U8) - 0.5)^2) / V8`
            7. In cell `U10`, compute the two-sided p-value using Excel's chi-square distribution function:
               `=CHISQ.DIST.RT(U9, 1)`
            """)

        with st.expander("Step 6: Assessing Confounding (10% Rule) & Testing Homogeneity", expanded=False):
            st.markdown("""
            1. In cell `W4`, calculate the Confounding Delta Percentage:
               `=ABS(L6 - P8) / P8 * 100` (where `L6` is Crude OR and `P8` is $\\text{OR}_{MH}$)
            2. In cell `W5`, apply an IF statement to flag confounding:
               `=IF(W4>=10, "CONFOUNDING PRESENT (Report Adjusted OR_MH)", "UNCONFOUNDED")`
            3. In cell `W6`, check for Simpson's Paradox:
               `=IF(AND(L6>1, L4<1, L5<1), "SIMPSON'S PARADOX DETECTED!", "No Paradox")`
            """)

        with st.expander("Step 7: Building a Stratified Forest Plot in Excel", expanded=False):
            st.markdown("""
            1. Select the Stratum Names, Odds Ratios, and Lower/Upper Confidence limits.
            2. Insert a **2D Clustered Bar** or **Scatter Plot**.
            3. Add custom error bars:
               * Click on the data series -> **Add Chart Element** -> **Error Bars** -> **More Error Bar Options**.
               * Select **Custom** -> **Specify Value**.
               * Set Positive Error to `Upper_Limit - OR` and Negative Error to `OR - Lower_Limit`.
            4. Add a vertical reference line at $X = 1.0$ using a secondary scatter line to highlight the null effect.
            """)

    # -------------------------------------------------------------
    # TAB 3: Python Production Script Guide
    # -------------------------------------------------------------
    with tab_py:
        st.markdown("""
        ### 🐍 Production-Ready Python Script Guide

        This guide presents clean, vectorized, production-grade Python code utilizing standard scientific libraries
        (`numpy`, `pandas`, `scipy.stats`) to execute the complete Mantel-Haenszel analysis pipeline.
        """)

        with st.expander("Stage 1: Ingesting Stratified Data & Computing Stratum-Specific 2x2 Metrics", expanded=False):
            st.code(r'''
import numpy as np
import pandas as pd
from scipy import stats

def compute_stratum_metrics(strata_data):
    """Computes stratum-specific 2x2 tables, odds ratios, and Woolf 95% CIs."""
    df = pd.DataFrame(strata_data)
    
    # Calculate margins
    df['n1'] = df['a'] + df['b']
    df['n0'] = df['c'] + df['d']
    df['m1'] = df['a'] + df['c']
    df['m0'] = df['b'] + df['d']
    df['n'] = df['n1'] + df['n0']
    
    # Haldane 0.5 correction for zero cells
    has_zero = (df['a'] == 0) | (df['b'] == 0) | (df['c'] == 0) | (df['d'] == 0)
    adj_a = np.where(has_zero, df['a'] + 0.5, df['a'])
    adj_b = np.where(has_zero, df['b'] + 0.5, df['b'])
    adj_c = np.where(has_zero, df['c'] + 0.5, df['c'])
    adj_d = np.where(has_zero, df['d'] + 0.5, df['d'])
    
    # Odds Ratios and Woolf Standard Errors
    df['OR'] = (adj_a * adj_d) / (adj_b * adj_c)
    df['SE_ln_OR'] = np.sqrt(1/adj_a + 1/adj_b + 1/adj_c + 1/adj_d)
    df['OR_Lower'] = np.exp(np.log(df['OR']) - 1.96 * df['SE_ln_OR'])
    df['OR_Upper'] = np.exp(np.log(df['OR']) + 1.96 * df['SE_ln_OR'])
    
    return df
''', language="python")

        with st.expander("Stage 2: Calculating Mantel-Haenszel Summary Pooled OR and Robins-Breslow-Greenland CI", expanded=False):
            st.code(r'''
def compute_mantel_haenszel_pooled_or(df):
    """Calculates OR_MH and Robins-Breslow-Greenland (1986) 95% Confidence Interval."""
    # Terms for OR_MH
    R_k = (df['a'] * df['d']) / df['n']
    S_k = (df['b'] * df['c']) / df['n']
    R_plus = R_k.sum()
    S_plus = S_k.sum()
    
    or_mh = R_plus / S_plus
    
    # Robins-Breslow-Greenland variance components
    P_k = (df['a'] + df['d']) / df['n']
    Q_k = (df['b'] + df['c']) / df['n']
    
    term1 = (P_k * R_k).sum() / (2 * (R_plus ** 2))
    term2 = ((P_k * S_k) + (Q_k * R_k)).sum() / (2 * R_plus * S_plus)
    term3 = (Q_k * S_k).sum() / (2 * (S_plus ** 2))
    
    var_ln_or_mh = term1 + term2 + term3
    se_ln_or_mh = np.sqrt(var_ln_or_mh)
    
    ci_lower = np.exp(np.log(or_mh) - 1.96 * se_ln_or_mh)
    ci_upper = np.exp(np.log(or_mh) + 1.96 * se_ln_or_mh)
    
    return or_mh, ci_lower, ci_upper, se_ln_or_mh
''', language="python")

        with st.expander("Stage 3: Computing Cochran-Mantel-Haenszel Chi-Square Test with Continuity Correction", expanded=False):
            st.code(r'''
def compute_cmh_chi_square(df):
    """Computes Cochran-Mantel-Haenszel Chi-Square test of conditional independence."""
    exp_a = (df['n1'] * df['m1']) / df['n']
    var_a = (df['n1'] * df['n0'] * df['m1'] * df['m0']) / ((df['n'] ** 2) * (df['n'] - 1))
    
    sum_a = df['a'].sum()
    sum_exp_a = exp_a.sum()
    sum_var_a = var_a.sum()
    
    diff = abs(sum_a - sum_exp_a)
    diff_corrected = max(0.0, diff - 0.5)
    
    chi2_stat = (diff_corrected ** 2) / sum_var_a
    p_value = stats.chi2.sf(chi2_stat, df=1)
    
    return chi2_stat, p_value
''', language="python")

        with st.expander("Stage 4: Breslow-Day & Tarone Homogeneity Tests for Effect Modification", expanded=False):
            st.code(r'''
import math

def solve_expected_cell_bd(row, theta):
    """Solves quadratic equation for Breslow-Day expected cell A_k*."""
    n1, n0, m1, m0, n = row['n1'], row['n0'], row['m1'], row['m0'], row['n']
    if abs(theta - 1.0) < 1e-7:
        return (n1 * m1) / n
    A = theta - 1.0
    B = -((theta - 1.0) * (n1 + m1) + n)
    C = theta * n1 * m1
    disc = max(0.0, B * B - 4.0 * A * C)
    r1 = (-B - math.sqrt(disc)) / (2.0 * A)
    r2 = (-B + math.sqrt(disc)) / (2.0 * A)
    low = max(0.0, n1 - m0)
    high = min(n1, m1)
    for r in (r1, r2):
        if low - 1e-5 <= r <= high + 1e-5:
            return r
    return (n1 * m1) / n

def test_breslow_day_homogeneity(df, or_mh):
    """Calculates Breslow-Day and Tarone-adjusted test of homogeneity of odds ratios."""
    k = len(df)
    exp_cells = []
    vars_cells = []
    bd_stat = 0.0
    
    for _, row in df.iterrows():
        a_star = solve_expected_cell_bd(row, or_mh)
        b_star = row['n1'] - a_star
        c_star = row['m1'] - a_star
        d_star = row['n0'] - c_star
        
        var_a = 1.0 / (1.0/a_star + 1.0/b_star + 1.0/c_star + 1.0/d_star)
        bd_stat += ((row['a'] - a_star) ** 2) / var_a
        exp_cells.append(a_star)
        vars_cells.append(var_a)
        
    sum_a = df['a'].sum()
    sum_exp_a = sum(exp_cells)
    sum_var_a = sum(vars_cells)
    
    # Tarone adjustment
    tarone_stat = max(0.0, bd_stat - ((sum_a - sum_exp_a) ** 2) / sum_var_a)
    df_bd = max(1, k - 1)
    p_value = stats.chi2.sf(tarone_stat, df=df_bd)
    
    return tarone_stat, df_bd, p_value
''', language="python")

        with st.expander("Stage 5: Complete Standalone End-to-End Production Script", expanded=False):
            st.code(r'''
# Complete Cochran-Mantel-Haenszel Analysis Pipeline
import pandas as pd
from src.epi_methods.mantel_haenszel import calculate_mantel_haenszel, generate_mantel_haenszel_demo_data

# 1. Load Synthetic Demonstration Cohort
demo = generate_mantel_haenszel_demo_data("simpsons_paradox_icu")
print(f"Dataset: {demo['scenario_name']}")

# 2. Run Comprehensive Mantel-Haenszel Pipeline
results = calculate_mantel_haenszel(demo['strata_tables'])

# 3. Print Clinical & Epidemiological Summary
print("=" * 60)
print(f"Crude Odds Ratio:            {results['crude_or']:.3f} {results['crude_or_ci_str']}")
print(f"Mantel-Haenszel Summary OR:  {results['or_mh']:.3f} {results['or_mh_ci_str']}")
print(f"Confounding Delta %:         {results['confounding_delta_pct']:.1f}%")
print(f"Simpson's Paradox Detected:  {results['simpsons_paradox_detected']}")
print(f"Breslow-Day Homogeneity p:   {results['breslow_day_p_value']:.4f}")
print(f"CMH Chi-Square p-value:      {results['cmh_p_value']:.4f}")
print("=" * 60)
print(results['clinical_recommendation'])
''', language="python")

    # -------------------------------------------------------------
    # TAB 4: Live Synthetic Data Downloads
    # -------------------------------------------------------------
    with tab_dl:
        st.markdown(r"""
        ### 💾 Live Synthetic Demonstration Data Downloads

        All datasets provided below are **100% synthetic, computer-generated demonstration files** created for teaching
        and code rehearsal. They contain **zero real patient data or protected health information**.
        """)

        st.warning("""
        ⚠️ **EDUCATIONAL DISCLAIMER & NOTICE**:
        * **100% Synthetic / Artificial Data:** All rows, medical records, wards, and outcomes were synthesized via mathematical simulation.
        * **Educational Rehearsal Only:** Designed for university teaching, biostatistical coursework, and practicing Mantel-Haenszel stratification in Excel, R, Python, Stata, or SAS.
        * **No Clinical Utility:** Must never be used for real patient treatment decisions or healthcare policy.
        """)

        c_d1, c_d2 = st.columns(2)

        df_micro = demo_data.get("df", pd.DataFrame())
        if not isinstance(df_micro, pd.DataFrame):
            df_micro = pd.DataFrame(df_micro)

        with c_d1:
            with st.container(border=True):
                st.markdown(f"#### 📄 Synthetic Patient Microdata ({len(df_micro)} Simulated Records)")
                st.markdown(f"Contains **{len(df_micro)} individual patient microdata records** including simulated patient IDs, admission ward/stratum, binary exposure status, and 30-day mortality outcome.")
                st.caption("*100% synthetic / computer-generated cohort. Contains zero real patient data.*")
                csv_micro = df_micro.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇️ Download Synthetic Patient Microdata ({len(df_micro)} Rows CSV)",
                    data=csv_micro,
                    file_name=f"synthetic_{meta['scenario_key']}_patient_microdata.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                with st.expander("👁️ Preview First 5 Rows of Synthetic Microdata", expanded=False):
                    st.dataframe(df_micro.head(5), use_container_width=True, hide_index=True)

        with c_d2:
            with st.container(border=True):
                st.markdown("#### 📊 Stratified 2x2 Contingency Tables Summary (CSV)")
                st.markdown("Aggregated summary of simulated patient numbers, observed 2x2 cell counts ($a, b, c, d$), stratum risks, Woolf odds ratio 95% CIs, and Mantel-Haenszel percentage weights.")
                st.caption("*Derived from the simulated synthetic cohort for tutorial exercises.*")
                df_strata = pd.DataFrame(results.get("strata_list", []))
                csv_strata = df_strata.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Stratified Contingency Summary CSV",
                    data=csv_strata,
                    file_name=f"synthetic_{meta['scenario_key']}_strata_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                with st.expander("👁️ Preview Stratified Contingency Summary Table", expanded=False):
                    st.dataframe(df_strata, use_container_width=True, hide_index=True)

        with st.expander("📋 Synthetic Data Generation Methodology & Variable Dictionary", expanded=False):
            st.markdown(r"""
            #### How This Synthetic Dataset Was Created
            1. **Cohort Simulation:** Patient microdata records are generated by allocating simulated patients across distinct clinical strata (e.g. Intensive Care Unit vs. General Medical Ward).
            2. **Unequal Confounder Distribution (Indication Bias):** To realistically demonstrate confounding and Simpson's Paradox, exposure allocation is correlated with stratum acuity (e.g. severely ill ICU patients preferentially received the novel therapy).
            3. **Ground-Truth Stratum Association:** Outcomes are simulated using fixed within-stratum odds ratios ($\text{OR}_k = 0.444$).
            4. **Reproducibility:** A fixed deterministic algorithm assigns records so the cohort perfectly matches the verifiable contingency counts across repeated runs.
            5. **Information Governance:** Because this dataset is entirely synthetic, it is safe to share, publish, and use in educational workshops without information governance, Caldicott Guardian, or GDPR restrictions.
            """)


def render_mantel_haenszel_module(active_method_key: str, method_info: Dict[str, Any]):
    """Main rendering controller for Cochran-Mantel-Haenszel Stratified Analysis & Effect Modification."""

    # 1. Plain English Pedagogical Introductory Guide
    render_mantel_haenszel_plain_english_guide()

    # 2. Sidebar Scenario Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Mantel-Haenszel Controls")

    scenario_options = {
        "simpsons_paradox_icu": "Inner City Teaching Hospital (Severe Simpson's Paradox) [Synthetic]",
        "age_confounding_cardiac": "All Meadows Cardiovascular Institute (Age Confounding) [Synthetic]",
        "effect_modification_smoking": "Riverside Health Trust (Smoking Interaction / Effect Modification) [Synthetic]",
        "custom": "Custom User-Defined Stratified Contingency Tables [Synthetic]"
    }

    selected_scenario = st.sidebar.selectbox(
        "Select Epidemiological Scenario:",
        options=list(scenario_options.keys()),
        format_func=lambda k: scenario_options[k]
    )
    st.sidebar.caption("⚠️ All clinical scenarios utilize purely synthetic datasets simulated for educational demonstration.")

    if selected_scenario != "custom":
        demo_data = generate_mantel_haenszel_demo_data(scenario_key=selected_scenario)
        strata_input = demo_data["strata_tables"]
        meta = {
            "scenario_key": selected_scenario,
            "title": demo_data["scenario_name"],
            "organisation": demo_data["organisation_name"],
            "domain": demo_data["domain"],
            "exposure_name": demo_data["exposure_name"],
            "outcome_name": demo_data["outcome_name"],
            "confounder_name": demo_data["confounder_name"],
            "description": demo_data["description"]
        }
    else:
        st.sidebar.markdown("#### ⚙️ Custom Strata Contingency Parameters")
        num_strata = st.sidebar.slider("Number of Strata (K)", min_value=2, max_value=4, value=2, step=1)
        strata_input = []
        for i in range(num_strata):
            st.sidebar.markdown(f"**Stratum {i+1} Cells:**")
            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                a_val = st.number_input(f"Exposed Cases (a_{i+1})", min_value=0, max_value=1000, value=40 if i==0 else 10, key=f"cust_a_{i}")
                c_val = st.number_input(f"Unexposed Cases (c_{i+1})", min_value=0, max_value=1000, value=30 if i==0 else 40, key=f"cust_c_{i}")
            with col_b:
                b_val = st.number_input(f"Exposed Non-Cases (b_{i+1})", min_value=0, max_value=1000, value=60 if i==0 else 90, key=f"cust_b_{i}")
                d_val = st.number_input(f"Unexposed Non-Cases (d_{i+1})", min_value=0, max_value=1000, value=20 if i==0 else 160, key=f"cust_d_{i}")
            strata_input.append({
                "Stratum": f"Custom Stratum {i+1}",
                "Exposed_Cases": a_val,
                "Exposed_NonCases": b_val,
                "Unexposed_Cases": c_val,
                "Unexposed_NonCases": d_val
            })

        # Create synthetic microdata for custom strata
        rows_cust = []
        pt_idx = 1
        for s_idx, s in enumerate(strata_input):
            s_name = s["Stratum"]
            for _ in range(int(s["Exposed_Cases"])):
                rows_cust.append({"Patient_ID": f"CUST-PT-{pt_idx:04d}", "Stratum": s_name, "Exposure": 1, "Outcome": 1})
                pt_idx += 1
            for _ in range(int(s["Exposed_NonCases"])):
                rows_cust.append({"Patient_ID": f"CUST-PT-{pt_idx:04d}", "Stratum": s_name, "Exposure": 1, "Outcome": 0})
                pt_idx += 1
            for _ in range(int(s["Unexposed_Cases"])):
                rows_cust.append({"Patient_ID": f"CUST-PT-{pt_idx:04d}", "Stratum": s_name, "Exposure": 0, "Outcome": 1})
                pt_idx += 1
            for _ in range(int(s["Unexposed_NonCases"])):
                rows_cust.append({"Patient_ID": f"CUST-PT-{pt_idx:04d}", "Stratum": s_name, "Exposure": 0, "Outcome": 0})
                pt_idx += 1

        df_custom = pd.DataFrame(rows_cust)
        demo_data = {
            "df": df_custom,
            "strata_tables": strata_input,
            "scenario_name": "Custom User-Defined Stratified Contingency Tables",
            "organisation_name": "Riverside Health Trust Community Health Directorate"
        }
        meta = {
            "scenario_key": "custom",
            "title": "Custom User-Defined Stratified Contingency Tables",
            "organisation": "Riverside Health Trust Community Health Directorate",
            "domain": "Custom Observational Epidemiology",
            "exposure_name": "Custom Evaluated Exposure",
            "outcome_name": "Custom Evaluated Clinical Outcome",
            "confounder_name": "Custom Stratifying Confounder",
            "description": "User-configured stratified contingency table evaluation."
        }

    # 3. Execute Core Calculations
    results = calculate_mantel_haenszel(strata_input)
    evaluations = evaluate_mantel_haenszel_prerequisites(results)

    # 4. Header Banner & Prominent Synthetic Disclaimer
    st.markdown(f"## 🏥 {meta['title']}")
    st.markdown(f"**Organisation:** {meta['organisation']} | **Domain:** {meta['domain']}")
    st.markdown(f"**Exposure:** `{meta['exposure_name']}` | **Outcome:** `{meta['outcome_name']}` | **Stratifying Variable:** `{meta['confounder_name']}`")
    st.markdown(f"*{meta['description']}*")

    with st.container(border=True):
        st.markdown(r"""
        > [!IMPORTANT]
        > **EDUCATIONAL & SYNTHETIC DATA DISCLAIMER**: The patient records, contingency tables, and clinical metrics displayed below are **100% synthetic and computer-generated** purely for academic training, epidemiological education, and software demonstration. Contains no real patient identifiers or protected health information.
        """)

    # 5. Methodological Prerequisite Cards
    render_mantel_haenszel_prerequisite_cards(evaluations)

    # 6. Primary Metric Dashboard
    st.markdown("### 📈 Primary Effect Measures & Confounding Diagnostics")
    render_mantel_haenszel_metric_dashboard(results)

    # 7. Interactive Visualizations
    col_v1, col_v2 = st.columns([1, 1])
    with col_v1:
        with st.container(border=True):
            fig_forest = build_stratified_forest_plot(results, meta)
            st.plotly_chart(fig_forest, use_container_width=True)

    with col_v2:
        with st.container(border=True):
            fig_rates = build_strata_rate_comparison_plot(results, meta)
            st.plotly_chart(fig_rates, use_container_width=True)

    # Explanatory Guide Beneath Plots
    render_stratified_explanatory_box(results)

    # 8. Detailed Stratum Contingency Breakdown Table
    render_strata_comparison_table(results)

    # 9. Full 4-Tab Educational Architecture
    st.markdown("---")
    st.markdown("### 🎓 Educational Guides & Reference Materials")
    render_mantel_haenszel_educational_tabs(results, meta, demo_data)

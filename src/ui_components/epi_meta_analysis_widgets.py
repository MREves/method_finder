"""Interactive UI Components & Visualizations for Meta-Analysis & Evidence Synthesis.

Provides:
1. Plain English Pedagogical Introductory Guide to Systematic Reviews, Meta-Analysis,
   Fixed vs. Random Effects, Between-Study Heterogeneity, and Publication Bias (Zero prior knowledge).
2. On-page and sidebar scenario controls with 3 realistic clinical meta-analysis scenarios and custom matrix exploration.
3. Methodological Prerequisite Evaluation (5 PRISMA 2020 / Cochrane standards: Study Volume,
   Heterogeneity magnitude, Model Selection concordance, Publication Bias, and Sensitivity stability).
4. Primary Metric Dashboard (Pooled Effect Size, Higgins I^2, Cochran's Q, tau^2, Egger test p-value).
5. Plotly Meta-Analysis Forest Plot with dual summary diamonds (Fixed-Effect & Random-Effects) and
   study weights represented by marker area.
6. Plotly Funnel Plot with inverted standard error y-axis and 95% pseudo-confidence limits for publication bias.
7. Full-Width Pedagogical Explanatory Guide beneath plots explaining forest diamonds, funnel symmetry,
   and term-by-term formula breakdowns.
8. Comprehensive Study-by-Study Evidence Synthesis Table with leave-one-out sensitivity metrics.
9. Full 4-Tab Educational Architecture:
   - 📋 Rationale & Epidemiological Framework (Complete formula breakdowns, plain English meanings, clinical rationale, PRISMA 2020).
   - 📗 Excel Implementation Guide (Cell coordinate mapping A1:Q25, collapsible expanders for each step, formulas, chart construction).
   - 🐍 Python Implementation Guide (Collapsible expanders for each stage, clean vectorized code, standalone pipeline).
   - 💾 Data Downloads (Patient microdata CSV, study matrix CSV, leave-one-out sensitivity CSV, variable dictionary).
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Union
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.meta_analysis import (
    calculate_meta_analysis,
    evaluate_meta_analysis_prerequisites,
    generate_meta_analysis_demo_data
)


def render_meta_analysis_plain_english_guide():
    """Renders a collapsible plain English introductory guide to systematic reviews and meta-analysis."""
    with st.expander("📖 Zero-Knowledge Guide: What is Meta-Analysis & Evidence Synthesis?", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. The Clinical Dilemma: Why One Trial is Never Enough
            * A single clinical trial—even a well-designed one—often lacks statistical power or may produce conflicting results due to random sampling variation.
            * Clinicians and national guideline panels (such as NICE or the WHO) cannot base medical practice on an isolated study.
            * **Meta-Analysis** is the statistical discipline of combining the quantitative findings of multiple independent studies addressing the exact same clinical question into a single, highly powered summary estimate of effect.
            * By pooling thousands of patients across dozens of clinical trials, meta-analysis reduces random noise and reveals the true underlying treatment effect with narrow confidence intervals.
            """)

            st.markdown(r"""
            #### 2. Fixed-Effect vs. Random-Effects: Two Fundamental Philosophies
            Epidemiology provides two distinct mathematical models for combining studies:
            * **Fixed-Effect Model (Common Effect):**
              * Assumes there is **one single true treatment effect** ($\theta$) that applies universally to all patients everywhere.
              * Any observed differences in effect sizes between studies are assumed to be purely due to within-study random sampling error.
              * Weights each study strictly by the **inverse of its variance** ($w_i = 1 / \text{SE}_i^2$). Large multi-thousand patient trials dominate the weighting.
            * **Random-Effects Model (DerSimonian & Laird):**
              * Acknowledges that real-world clinical trials differ: patients have different average ages, different comorbidities, slightly different drug dosages, or different healthcare settings.
              * Assumes the true effect varies across study populations ($\theta_i \sim N(\mu, \tau^2)$).
              * Incorporates **between-study variance ($\tau^2$)** into each study's weight: $w_i^* = 1 / (\text{SE}_i^2 + \tau^2)$.
              * **Clinical Rule:** When between-study heterogeneity is present ($I^2 \ge 30\%$), the Random-Effects model must be preferred because the Fixed-Effect model produces artificially narrow confidence intervals that underestimate clinical uncertainty.
            """)

            st.markdown(r"""
            #### 3. Higgins & Thompson's $I^2$: Measuring Heterogeneity in Plain English
            * How do we measure whether studies agree with each other or disagree?
            * **Cochran's $Q$:** A chi-square test testing the null hypothesis that all studies share the exact same true effect ($H_0: \theta_1 = \dots = \theta_K$).
            * **Higgins $I^2$ Statistic:** Quantifies the percentage of total variability across studies that is due to genuine clinical or methodological heterogeneity rather than random chance:
              $$I^2 = \frac{Q - (K - 1)}{Q} \times 100\%$$
              * **Plain English Meaning:** The numerator ($Q - (K - 1)$) represents the **excess variation** beyond what would be expected from random chance alone ($K - 1$). Dividing by total variation ($Q$) gives the proportion of real disagreement.
              * **$I^2 < 25\%$:** Low / negligible heterogeneity (studies are highly consistent).
              * **$I^2 = 25–50\%$:** Moderate heterogeneity.
              * **$I^2 = 50–75\%$:** Substantial heterogeneity (random-effects model essential).
              * **$I^2 > 75\%$:** Severe heterogeneity (pooling must be interpreted with extreme caution).
            """)

        with col_g2:
            st.markdown(r"""
            #### 4. Publication Bias & The "File Drawer Problem"
            * In biomedical research, studies demonstrating large, exciting, statistically significant treatment benefits are far more likely to be submitted and published in prestigious journals than trials finding neutral or negative results.
            * Small negative studies often remain unpublished—locked away in the researcher's "file drawer".
            * If an evidence synthesis pools only published studies while ignoring hidden negative trials, the resulting meta-analysis will **dramatically exaggerate the true benefit of the drug**!
            """)

            st.markdown(r"""
            #### 5. How to Detect Publication Bias: The Funnel Plot & Egger's Test
            * **The Funnel Plot:** A scatter plot displaying each study's effect size on the horizontal axis against its precision (standard error $\text{SE}$) on the vertical axis (inverted so the largest, most precise trials appear at the top).
              * In the absence of bias, the studies form a symmetrical upside-down triangle (funnel) centered around the true pooled effect.
              * If smaller studies at the bottom of the plot are missing from the negative side, the funnel becomes lopsided and asymmetrical.
            * **Egger's Linear Regression Test:** Statistically tests funnel plot asymmetry by regressing the standardized effect size ($\text{SND} = y_i / \text{SE}_i$) against precision ($1 / \text{SE}_i$).
              * **Why this works:** If the funnel is symmetrical, the regression line passes through the origin (intercept $\alpha = 0$). If small studies have exaggerated benefits, the intercept shifts away from zero ($\alpha \ne 0, p < 0.10$), providing objective statistical proof of publication bias or small-study effects!
            """)

            st.markdown(r"""
            #### 6. PRISMA 2020 Guidelines for Evidence Synthesis
            * All rigorous systematic reviews and meta-analyses are reported according to the international **PRISMA 2020** statement (Preferred Reporting Items for Systematic Reviews and Meta-Analyses).
            * Requires transparent reporting of search strategy, study eligibility criteria, assessment of risk of bias, heterogeneity statistics ($I^2, \tau^2, Q$), both fixed and random effects, sensitivity analyses, and investigation of publication bias.
            """)


def render_meta_analysis_prerequisite_cards(evaluations: Dict[str, Any]):
    """Renders PRISMA 2020 / Cochrane evidence synthesis prerequisite cards."""
    checks = evaluations.get("checks", [])
    if not checks:
        return

    st.markdown("### 🔍 Methodological & Evidence Synthesis Prerequisite Checks")
    st.caption("Verifying study volume power, between-study heterogeneity, statistical model choice, publication bias, and sensitivity stability under PRISMA 2020 standards.")

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


def render_meta_analysis_metric_dashboard(results: Dict[str, Any]):
    """Renders the top-level metric cards for pooled summary effects, I^2, tau^2, and Egger bias."""
    c1, c2, c3, c4, c5 = st.columns(5)

    pref = results.get("preferred_model", "Fixed-Effect")
    eff_rand = results.get("random_effect_natural", 1.0)
    eff_rand_ci = results.get("random_effect_ci_str", "")
    eff_fixed = results.get("fixed_effect_natural", 1.0)
    eff_fixed_ci = results.get("fixed_effect_ci_str", "")

    i2 = results.get("i2_pct", 0.0)
    q_stat = results.get("cochran_q", 0.0)
    q_p = results.get("cochran_q_p_value", 1.0)
    tau2 = results.get("tau2", 0.0)
    tau = results.get("tau", 0.0)

    egger_p = results.get("egger_p_value", 1.0)
    egger_alpha = results.get("egger_intercept", 0.0)
    egger_bias = results.get("egger_bias_detected", False)

    with c1:
        with st.container(border=True):
            st.metric(
                label="Random-Effects Pooled OR",
                value=f"{eff_rand:.2f}",
                delta=f"95% CI: {eff_rand_ci}"
            )
            st.caption(f"DerSimonian-Laird Model | Z = {results.get('random_z_statistic', 0.0):.2f} (p = {results.get('random_p_value', 1.0):.4f})")

    with c2:
        with st.container(border=True):
            st.metric(
                label="Fixed-Effect Pooled OR",
                value=f"{eff_fixed:.2f}",
                delta=f"95% CI: {eff_fixed_ci}"
            )
            st.caption(f"Inverse-Variance Model | Z = {results.get('fixed_z_statistic', 0.0):.2f} (p = {results.get('fixed_p_value', 1.0):.4f})")

    with c3:
        with st.container(border=True):
            st.metric(
                label="Higgins Heterogeneity (I²)",
                value=f"{i2:.1f}%",
                delta=results.get("heterogeneity_category", "Low Heterogeneity")
            )
            st.caption(f"Cochran's Q = {q_stat:.2f} (df={results.get('cochran_q_df', 1)}, p = {q_p:.4f})")

    with c4:
        with st.container(border=True):
            st.metric(
                label="Between-Study Variance (τ²)",
                value=f"{tau2:.4f}",
                delta=f"τ = {tau:.3f} (Log Scale)"
            )
            st.caption("DerSimonian-Laird between-study variance estimator")

    with c5:
        with st.container(border=True):
            st.metric(
                label="Publication Bias (Egger Test)",
                value=f"p = {egger_p:.4f}",
                delta=f"Intercept α = {egger_alpha:.2f}" + (" (Bias Detected!)" if egger_bias else " (Symmetric)")
            )
            st.caption("Egger's linear regression test for funnel plot asymmetry")

    # Recommendation Callout Banner
    st.markdown(f"""
    <div style="background-color: {'#fffbeb' if pref == 'Random-Effects' else '#f0fdf4'}; 
                border-left: 5px solid {'#f59e0b' if pref == 'Random-Effects' else '#10b981'}; 
                padding: 1rem; border-radius: 6px; margin-top: 1rem; margin-bottom: 1.5rem;">
        <h4 style="margin: 0 0 0.35rem 0; color: {'#92400e' if pref == 'Random-Effects' else '#065f46'}; font-size: 1.05rem;">
            {results.get('clinical_recommendation', '')}
        </h4>
        <p style="margin: 0; font-size: 0.9rem; color: #334155;">
            <strong>Higgins 95% Prediction Interval for a Future Study:</strong> {results.get('prediction_interval_str', '—')} 
            <em>(Reflects the plausible true effect size if a new trial is conducted in an independent patient population).</em>
        </p>
    </div>
    """, unsafe_allow_html=True)


def build_meta_analysis_forest_plot(results: Dict[str, Any], meta: Dict[str, Any]) -> go.Figure:
    """Builds a Plotly forest plot displaying individual study estimates and pooled summary diamonds."""
    studies = results.get("studies_list", [])
    if not studies:
        return go.Figure()

    is_ratio = results.get("is_ratio_metric", True)
    null_val = 1.0 if is_ratio else 0.0

    study_labels = [f"{s['Study']} ({s['Year']})" for s in studies]
    effects = [s["Effect_Natural"] for s in studies]
    ci_lows = [s["CI_Lower_Natural"] for s in studies]
    ci_upps = [s["CI_Upper_Natural"] for s in studies]
    weights = [s["Weight_Random_Pct"] for s in studies]

    # Marker sizes scaled proportionally to study weights (sleeker bounds)
    max_w = max(weights) if weights else 1.0
    marker_sizes = [max(7, min(18, int(math.sqrt(w / max_w) * 16))) for w in weights]

    y_pos = list(range(len(studies), 0, -1))

    fig = go.Figure()

    # Vertical reference line at null effect (1.0 for OR/RR)
    fig.add_shape(
        type="line",
        x0=null_val, x1=null_val,
        y0=0, y1=len(studies) + 2.5,
        line=dict(color="#9CA3AF", width=1.0, dash="dash")
    )

    # Individual study error bars and markers (reduced line width and cap width)
    fig.add_trace(go.Scatter(
        x=effects,
        y=y_pos,
        mode="markers",
        marker=dict(
            symbol="square",
            size=marker_sizes,
            color="#2563EB",
            line=dict(color="#1D4ED8", width=1.0)
        ),
        error_x=dict(
            type="data",
            symmetric=False,
            array=[u - e for u, e in zip(ci_upps, effects)],
            arrayminus=[e - l for l, e in zip(ci_lows, effects)],
            color="#2563EB",
            thickness=1.2,
            width=3
        ),
        text=[f"<b>{s['Study']}</b><br>Effect: {e:.2f} [{l:.2f}, {u:.2f}]<br>Weight: {w:.1f}%"
              for s, e, l, u, w in zip(studies, effects, ci_lows, ci_upps, weights)],
        hoverinfo="text",
        name="Individual Studies"
    ))

    # Add Fixed-Effect Summary Diamond (at y = 0.5)
    f_eff = results.get("fixed_effect_natural", 1.0)
    f_low = results.get("fixed_effect_ci_lower", 0.8)
    f_upp = results.get("fixed_effect_ci_upper", 1.2)
    fig.add_trace(go.Scatter(
        x=[f_low, f_eff, f_upp, f_eff, f_low],
        y=[0.5, 0.7, 0.5, 0.3, 0.5],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.5)",
        line=dict(color="#047857", width=1.5),
        name="Fixed-Effect Pooled Summary",
        hoverinfo="text",
        text=f"<b>Fixed-Effect Summary</b><br>OR: {f_eff:.2f} [{f_low:.2f}, {f_upp:.2f}]"
    ))

    # Add Random-Effects Summary Diamond (at y = -0.5)
    r_eff = results.get("random_effect_natural", 1.0)
    r_low = results.get("random_effect_ci_lower", 0.7)
    r_upp = results.get("random_effect_ci_upper", 1.3)
    fig.add_trace(go.Scatter(
        x=[r_low, r_eff, r_upp, r_eff, r_low],
        y=[-0.5, -0.3, -0.5, -0.7, -0.5],
        fill="toself",
        fillcolor="rgba(239, 68, 68, 0.5)",
        line=dict(color="#B91C1C", width=1.5),
        name="Random-Effects Pooled Summary",
        hoverinfo="text",
        text=f"<b>Random-Effects Summary</b><br>OR: {r_eff:.2f} [{r_low:.2f}, {r_upp:.2f}]"
    ))

    # Add Higgins 95% Prediction Interval as dashed horizontal bar beneath random diamond
    pred_low = results.get("prediction_interval_natural_lower", r_low)
    pred_upp = results.get("prediction_interval_natural_upper", r_upp)
    fig.add_shape(
        type="line",
        x0=pred_low, x1=pred_upp,
        y0=-1.3, y1=-1.3,
        line=dict(color="#DC2626", width=1.5, dash="dot")
    )
    fig.add_trace(go.Scatter(
        x=[(pred_low + pred_upp) / 2.0],
        y=[-1.3],
        mode="markers",
        marker=dict(symbol="line-ns", size=8, color="#DC2626"),
        hoverinfo="text",
        text=f"<b>95% Prediction Interval</b><br>[{pred_low:.2f}, {pred_upp:.2f}]",
        name="95% Prediction Interval"
    ))

    all_y_labels = ["Prediction Interval", "Random-Effects Summary", "Fixed-Effect Summary"] + study_labels
    all_y_ticks = [-1.3, -0.5, 0.5] + y_pos

    fig.update_layout(
        title=dict(
            text=f"<b>Meta-Analysis Forest Plot: {meta.get('intervention', 'Intervention')} vs. {meta.get('comparator', 'Comparator')}</b>",
            font=dict(size=14)
        ),
        xaxis=dict(
            title=dict(
                text=f"Treatment Effect ({results.get('effect_metric', 'Odds Ratio')} — Log Scale)",
                font=dict(size=11),
                standoff=12
            ),
            type="log" if is_ratio else "linear",
            gridcolor="#E5E7EB",
            zeroline=False
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=all_y_ticks,
            ticktext=all_y_labels,
            autorange="reversed",
            gridcolor="#F3F4F6",
            gridwidth=0.7
        ),
        height=max(420, 100 + len(studies) * 35),
        margin=dict(l=195, r=160, t=55, b=65),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#E5E7EB",
            borderwidth=1
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def build_meta_analysis_funnel_plot(results: Dict[str, Any], meta: Dict[str, Any]) -> go.Figure:
    """Builds a Plotly Funnel Plot to assess publication bias and small-study effects."""
    studies = results.get("studies_list", [])
    if not studies:
        return go.Figure()

    is_ratio = results.get("is_ratio_metric", True)
    theta_pooled = results.get("fixed_effect_log" if is_ratio else "fixed_effect_natural", 0.0)

    study_names = [s["Study"] for s in studies]
    y_effects = [s["Effect_Log"] if is_ratio else s["Effect_Natural"] for s in studies]
    x_ses = [s["SE_Log"] for s in studies]

    max_se = max(x_ses) * 1.25 if x_ses else 0.50

    # Generate pseudo 95% confidence limits triangle
    se_grid = np.linspace(0.001, max_se, 100)
    ci_left = theta_pooled - 1.95996 * se_grid
    ci_right = theta_pooled + 1.95996 * se_grid

    fig = go.Figure()

    # Shaded 95% confidence funnel region
    fig.add_trace(go.Scatter(
        x=list(ci_left) + list(ci_right[::-1]),
        y=list(se_grid) + list(se_grid[::-1]),
        fill="toself",
        fillcolor="rgba(229, 231, 235, 0.5)",
        line=dict(color="rgba(156, 163, 175, 0.6)", width=1, dash="dash"),
        name="95% Pseudo-Confidence Limits",
        hoverinfo="skip"
    ))

    # Centerline at pooled effect
    fig.add_shape(
        type="line",
        x0=theta_pooled, x1=theta_pooled,
        y0=0, y1=max_se,
        line=dict(color="#DC2626", width=1.5, dash="dot")
    )

    # Null reference line (0.0 on log scale)
    fig.add_shape(
        type="line",
        x0=0.0, x1=0.0,
        y0=0, y1=max_se,
        line=dict(color="#6B7280", width=1, dash="solid")
    )

    # Scatter points for studies
    hover_texts = [
        f"<b>{name}</b><br>Observed Effect (Log): {eff:.2f} (Natural: {math.exp(eff):.2f})<br>SE: {se:.3f}"
        for name, eff, se in zip(study_names, y_effects, x_ses)
    ]

    fig.add_trace(go.Scatter(
        x=y_effects,
        y=x_ses,
        mode="markers",
        marker=dict(
            size=11,
            color="#2563EB",
            line=dict(color="#1E40AF", width=1.5)
        ),
        text=study_names,
        hoverinfo="text",
        hovertext=hover_texts,
        name="Included Studies"
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Funnel Plot for Publication Bias & Small-Study Effects</b><br><sup>Egger's Test Intercept: {results.get('egger_intercept', 0.0):.2f} (p = {results.get('egger_p_value', 1.0):.4f})</sup>",
            font=dict(size=14)
        ),
        xaxis=dict(
            title=f"Log {results.get('effect_metric', 'Odds Ratio')} (Symmetrical around red pooled line)",
            gridcolor="#E5E7EB"
        ),
        yaxis=dict(
            title="Standard Error (SE) — Inverted Precision Scale",
            autorange="reversed",  # Invert so 0 is at top
            gridcolor="#F3F4F6",
            range=[max_se, 0.0]
        ),
        height=380,
        margin=dict(l=60, r=40, t=70, b=50),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def render_meta_analysis_explanatory_box(results: Dict[str, Any]):
    """Renders a full-width pedagogical explanatory guide beneath the plots with term-by-term formula explanations."""
    with st.container(border=True):
        st.markdown("### 📊 How to Interpret the Forest Plot and Funnel Plot")
        st.markdown(r"""
        * **1. Reading the Forest Plot:**
          * **Square Size & Weights:** The area of each blue square is directly proportional to that study's statistical weight in the meta-analysis. Larger trials with more confirmed events and smaller standard errors receive the largest squares.
          * **Horizontal Lines:** The whiskers represent the 95% Confidence Interval for each individual study. If a whisker crosses the dashed gray vertical line ($1.0$ for Odds Ratios), that individual study was not statistically significant on its own.
          * **The Pooled Diamonds:** The width of the diamond represents the 95% Confidence Interval of the pooled estimate.
            * **Green Diamond (Fixed-Effect):** Weights trials strictly by inverse variance ($w_i = 1 / \text{SE}_i^2$). Assumes all trials share one single universal truth.
            * **Red Diamond (Random-Effects):** Incorporates between-study variance ($w_i^* = 1 / (\text{SE}_i^2 + \tau^2)$). When $I^2 > 0$, the red diamond is wider than the green diamond, correctly reflecting the real-world clinical diversity across hospitals and populations.
          * **Dashed Red Line (Higgins 95% Prediction Interval):** Predicts the true effect size in a *future* clinical trial or hospital setting. If the prediction interval crosses 1.0 while the confidence interval diamond is statistically significant, it warns clinicians that although the treatment works on average across past trials, it might fail to benefit patients in some future healthcare settings.
        * **2. Reading the Funnel Plot:**
          * The inverted funnel has the most precise, massive studies at the very top (near $\text{SE} = 0$), clustering closely around the red pooled center line.
          * As sample size decreases, studies have larger standard errors and naturally scatter more widely toward the base of the plot.
          * **Spotting Publication Bias:** If the bottom of the triangle is missing studies on the neutral/harmful side, while small studies are clustered exclusively on the beneficial side, **publication bias** is present. This is formally verified by **Egger's linear regression test** ($p < 0.10$).
        """)


def render_meta_analysis_evidence_table(results: Dict[str, Any]):
    """Renders the comprehensive study-by-study evidence synthesis and leave-one-out table."""
    st.markdown("### 📋 Study-by-Study Evidence Synthesis Table & Weights")
    st.caption("Detailed breakdown of included studies, event counts, effect sizes with 95% CIs, and percentage weights in Fixed and Random models.")

    studies = results.get("studies_list", [])
    loo_list = results.get("leave_one_out_list", [])
    loo_dict = {r["Omitted_Study"]: r for r in loo_list}

    rows = []
    for s in studies:
        st_name = s["Study"]
        loo_info = loo_dict.get(st_name, {})
        rows.append({
            "Study": st_name,
            "Year": s["Year"],
            "Treated Events / Total": f"{s['Treated_Events']} / {s['Treated_Events'] + s['Treated_NonEvents']}",
            "Control Events / Total": f"{s['Control_Events']} / {s['Control_Events'] + s['Control_NonEvents']}",
            "Effect Size (95% CI)": s["CI_Str"],
            "Fixed Weight (%)": f"{s['Weight_Fixed_Pct']:.1f}%",
            "Random Weight (%)": f"{s['Weight_Random_Pct']:.1f}%",
            "Omit-Study Pooled Effect": loo_info.get("CI_Str", "—"),
            "Omit-Study I² (%)": f"{loo_info.get('Heterogeneity_I2_Pct', 0.0):.1f}%"
        })

    # Summary rows
    rows.append({
        "Study": "FIXED-EFFECT SUMMARY",
        "Year": "—",
        "Treated Events / Total": "—",
        "Control Events / Total": "—",
        "Effect Size (95% CI)": results.get("fixed_effect_ci_str", ""),
        "Fixed Weight (%)": "100.0%",
        "Random Weight (%)": "—",
        "Omit-Study Pooled Effect": "—",
        "Omit-Study I² (%)": "—"
    })
    rows.append({
        "Study": "RANDOM-EFFECTS SUMMARY",
        "Year": "—",
        "Treated Events / Total": "—",
        "Control Events / Total": "—",
        "Effect Size (95% CI)": results.get("random_effect_ci_str", ""),
        "Fixed Weight (%)": "—",
        "Random Weight (%)": "100.0%",
        "Omit-Study Pooled Effect": "—",
        "Omit-Study I² (%)": "—"
    })

    df_disp = pd.DataFrame(rows)
    st.dataframe(df_disp, use_container_width=True, hide_index=True)


def render_meta_analysis_educational_tabs(
    results: Dict[str, Any],
    meta: Dict[str, Any],
    demo_data: Dict[str, Any],
    method_info: Dict[str, Any]
):
    """Renders the comprehensive 4-Tab Educational Architecture."""
    tab_rat, tab_xls, tab_py, tab_dl = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Excel Implementation Guide",
        "🐍 Python Implementation Guide",
        "💾 Data Downloads"
    ])

    # -------------------------------------------------------------
    # TAB 1: Rationale & Epidemiological Framework
    # -------------------------------------------------------------
    with tab_rat:
        st.subheader(f"Why Systematic Reviews & Meta-Analysis? ({method_info.get('name', 'Evidence Synthesis')})")
        st.markdown(r"""
        In modern evidence-based medicine and public health, decisions regarding patient care, clinical guidelines, 
        and health technology appraisals cannot rely on individual trials. Individual studies may yield conflicting results 
        due to small sample sizes, varying patient baseline demographics, or random sampling fluctuations. 
        **Meta-Analysis** provides the rigorous mathematical framework to systematically pool quantitative estimates across multiple trials, 
        quantify clinical heterogeneity, and assess potential reporting biases.
        """)

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown(f"#### ✅ When to use {method_info.get('name', 'Meta-Analysis')}")
            for item in method_info.get("when_to_use", []):
                st.markdown(f"- {item}")
        with col_w2:
            st.markdown("#### ⚠️ Key Assumptions to Verify")
            for item in method_info.get("key_assumptions", []):
                st.markdown(f"- {item}")
            st.info(f"**Alternative if assumptions fail:** {method_info.get('alternative_test', 'Individual Patient Data (IPD) Meta-Analysis')}")

        st.markdown("---")
        st.markdown("### 🔍 In-Depth Epidemiological Framework: Fixed vs. Random Effects vs. IPD Meta-Analysis")
        st.markdown(r"""
        | Dimension | Fixed-Effect Model (Common Effect) | Random-Effects Model (DerSimonian-Laird) | Individual Patient Data (IPD) Meta-Analysis |
        | :--- | :--- | :--- | :--- |
        | **Underlying Hypothesis** | Assumes **one single true effect** ($\theta$) applies to all trials; observed variance is 100% within-study sampling error. | Assumes true effects vary across populations ($\theta_i \sim N(\mu, \tau^2)$); real clinical diversity exists. | Models patient-level covariates directly using raw microdata from trial registries. |
        | **Study Weighting Formula** | $w_i = \frac{1}{\text{SE}_i^2}$ (strictly precision-driven). | $w_i^* = \frac{1}{\text{SE}_i^2 + \tau^2}$ (balanced across studies). | Multilevel mixed-effects regression across patient rows. |
        | **Weight Distribution** | Massive trials dominate; small studies receive negligible weight. | Weights are more evenly distributed across large and small studies. | Weighted by individual patient-level information. |
        | **When Recommended** | $I^2 < 25–30\%$ and Cochran's $Q$ $p \ge 0.10$ (homogeneous trials). | $I^2 \ge 30\%$ or Cochran's $Q$ $p < 0.10$ (substantial clinical heterogeneity). | When raw patient records can be shared and pooled across clinical centers. |
        | **Confidence Interval Behavior** | Produces narrower, highly optimistic confidence intervals. | Widens confidence intervals to incorporate between-study uncertainty ($\tau^2$). | Incorporates both patient-level and trial-level clustering. |
        """)

        st.markdown("#### 📖 Key Terms, Formulae & Plain English Explanations (Zero Prior Knowledge Assumed)")
        st.markdown("Below is a step-by-step mathematical and conceptual breakdown of every formula used in evidence synthesis:")

        with st.expander("1. Logarithmic Transformation of Odds Ratios (Why and How We Work on the Log Scale)", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            $$y_i = \ln(\text{OR}_i) = \ln\left(\frac{a_i \cdot d_i}{b_i \cdot c_i}\right)$$

            #### Plain English Breakdown of the Formula:
            * **$\text{OR}_i$**: The raw Odds Ratio computed from study $i$'s $2 \times 2$ contingency table (where $a_i$ = treated events, $b_i$ = treated non-events, $c_i$ = control events, $d_i$ = control non-events).
            * **$\ln$**: The natural logarithm function.

            #### Why This Calculation is Being Used:
            * **The Skewness Problem:** Raw Odds Ratios are severely asymmetrical! A protective effect is compressed between $0.00$ and $1.00$, while a harmful effect stretches from $1.00$ all the way to $\infty$. For example, an $\text{OR} = 0.50$ (halving the risk) and an $\text{OR} = 2.00$ (doubling the risk) represent equal biological effects in opposite directions, but their arithmetic average is $(0.5 + 2.0)/2 = 1.25$ (a false finding of net harm!).
            * **Logarithmic Symmetry:** Taking the natural logarithm converts the scale so that $0.00$ represents the null effect, $\ln(0.50) = -0.693$, and $\ln(2.00) = +0.693$. The distribution becomes symmetrical and Gaussian (normal), allowing classical inverse-variance weighting and $Z$-score calculations!
            * **Conversion Back:** After pooling on the log scale, we exponentiate ($\exp(\theta)$) to return to the natural Odds Ratio scale.
            """)

        with st.expander("2. Woolf's Formula for Standard Error & Within-Study Variance", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            $$v_i = \text{SE}_i^2 = \frac{1}{a_i} + \frac{1}{b_i} + \frac{1}{c_i} + \frac{1}{d_i}$$
            $$\text{SE}_i = \sqrt{\frac{1}{a_i} + \frac{1}{b_i} + \frac{1}{c_i} + \frac{1}{d_i}}$$

            #### Plain English Breakdown of the Formula:
            * **$v_i$**: The within-study sampling variance of the log odds ratio.
            * **$\text{SE}_i$**: The standard error (the square root of the variance).
            * **$a_i, b_i, c_i, d_i$**: The counts of participants in the four cells of study $i$'s $2 \times 2$ table.

            #### Why This Calculation is Being Used:
            * **Measuring Study Precision:** Woolf's formula sums the reciprocal of every single cell in the contingency table.
            * **The Reciprocal Logic:** If any cell in a study has very few patients (e.g. $a_i = 2$ events), its reciprocal is huge ($1/2 = 0.50$), inflating the standard error and signaling high statistical noise. If a study has hundreds of events ($a_i = 500$), the reciprocal is tiny ($1/500 = 0.002$), producing a small standard error and high precision.
            * **Haldane-Anscombe 0.5 Continuity Correction:** If any cell is zero ($0$), dividing by zero is mathematically impossible ($1/0 \rightarrow \infty$). The software automatically adds $0.5$ to all four cells of that trial, stabilizing the calculation without distorting clinical conclusions.
            """)

        with st.expander("3. Fixed-Effect Model: Inverse-Variance Weighting & Summary Effect", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            * **Study Weight ($w_i$):**
              $$w_i = \frac{1}{v_i} = \frac{1}{\text{SE}_i^2}$$
            * **Summary Pooled Log-Effect ($\theta_{\text{fixed}}$):**
              $$\theta_{\text{fixed}} = \frac{\sum_{i=1}^K w_i y_i}{\sum_{i=1}^K w_i}$$
            * **Pooled Standard Error & 95% Confidence Interval:**
              $$\text{SE}(\theta_{\text{fixed}}) = \sqrt{\frac{1}{\sum_{i=1}^K w_i}}, \quad \text{CI}_{95\%} = \exp\left(\theta_{\text{fixed}} \pm 1.96 \times \text{SE}(\theta_{\text{fixed}})\right)$$

            #### Plain English Breakdown of the Formula:
            * **The Inversion of Variance ($1 / v_i$):** Variance measures uncertainty. Inverting variance turns uncertainty into credibility: a trial with tiny uncertainty gets a massive weight, while a noisy, imprecise trial gets a tiny weight.
            * **The Weighted Average ($\sum w_i y_i / \sum w_i$):** Exactly like calculating a weighted GPA in university, we multiply each study's log-effect ($y_i$) by its credibility weight ($w_i$), sum them up across all $K$ studies, and divide by the total weight.

            #### Why This Calculation is Being Used:
            * A simple unweighted mean would treat a 10-patient pilot study with equal importance as a 5,000-patient multicenter trial. Inverse-variance weighting guarantees that the most statistically reliable studies have the greatest influence on clinical guidelines.
            * **Assumption:** Assumes that every trial is estimating the exact same true underlying parameter $\theta$, and all differences between studies are pure random sampling error.
            """)

        with st.expander("4. Cochran's Q Test Statistic for Between-Study Heterogeneity", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            $$Q = \sum_{i=1}^K w_i \left(y_i - \theta_{\text{fixed}}\right)^2, \quad \text{df} = K - 1$$
            $$p\text{-value} = 1 - F_{\chi^2}(Q, \text{df} = K - 1)$$

            #### Plain English Breakdown of the Formula:
            * **$\left(y_i - \theta_{\text{fixed}}\right)$**: The observed gap between study $i$'s result and the pooled summary consensus.
            * **$\left(y_i - \theta_{\text{fixed}}\right)^2$**: The gap is squared so that positive and negative deviations do not cancel each other out, and large disagreements are penalized heavily.
            * **$\times w_i$**: Each squared gap is multiplied by the study's weight. If a massive, precise trial strongly disagrees with the rest of the literature, it adds a huge penalty to $Q$.
            * **$\text{df} = K - 1$**: Degrees of freedom (number of studies minus 1). Under the null hypothesis that all studies share the exact same true effect, the expected value of $Q$ is exactly equal to $K - 1$.

            #### Why This Calculation is Being Used:
            * Cochran's $Q$ formally tests whether the differences between clinical trials can be explained by pure random chance alone ($H_0$), or whether true clinical/methodological differences exist between trials ($H_1$).
            * If $Q$ is substantially larger than $K - 1$, the $p$-value drops below $0.10$, rejecting the common-effect assumption and proving that studies disagree!
            """)

        with st.expander("5. DerSimonian-Laird Estimator of Between-Study Variance (Tau-Squared, τ²)", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            $$C = \sum_{i=1}^K w_i - \frac{\sum_{i=1}^K w_i^2}{\sum_{i=1}^K w_i}$$
            $$\tau^2 = \max\left(0, \frac{Q - (K - 1)}{C}\right), \quad \tau = \sqrt{\tau^2}$$

            #### Plain English Breakdown of the Formula:
            * **Numerator ($Q - (K - 1)$):** The total observed variation ($Q$) minus the variation expected by pure chance alone ($K - 1$). This subtraction isolates the **excess variation** that cannot be explained by sampling noise!
            * **Denominator ($C$):** A scaling factor that converts the excess sum-of-squares back onto the squared effect size scale.
            * **$\max(0, \dots)$:** If $Q < K - 1$, there is less variation than expected by chance. Variance cannot be negative, so $\tau^2$ is truncated at zero ($0.000$).

            #### Why This Calculation is Being Used:
            * While Cochran's $Q$ tells us *whether* heterogeneity is present (a hypothesis test), $\tau^2$ (tau-squared) measures *how much* true variance exists between study populations in squared log units.
            * $\tau^2$ is the core parameter required to calculate Random-Effects weights and prediction intervals.
            """)

        with st.expander("6. Higgins & Thompson's I² Statistic: Quantifying Heterogeneity as a Percentage", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            $$I^2 = \max\left(0\%, \frac{Q - (K - 1)}{Q} \times 100\%\right)$$

            #### Plain English Breakdown of the Formula:
            * **Numerator ($Q - (K - 1)$):** The excess variation between studies beyond chance.
            * **Denominator ($Q$):** The total observed variation between studies.
            * Dividing excess by total gives the proportion of variation that is genuine clinical diversity rather than random noise. Multiplying by $100\%$ converts it into an intuitive percentage.

            #### Why This Calculation is Being Used:
            * Cochran's $Q$ has low statistical power when $K$ is small and excessive power when $K$ is large, making its $p$-value hard to interpret clinically. $\tau^2$ is on an abstract log-squared scale.
            * **Higgins $I^2$ solves this by providing a universal 0% to 100% benchmark:**
              * **$I^2 < 25\%$ (Low):** Over 75% of the differences between studies are just random sampling noise; trials are saying the same thing.
              * **$I^2 = 25–50\%$ (Moderate):** Mild clinical variation (e.g. slight differences in patient age or drug dosing).
              * **$I^2 = 50–75\%$ (Substantial):** Substantial heterogeneity; random-effects model is mandatory.
              * **$I^2 > 75\%$ (Severe):** Major disagreement; pooling into a single number may be clinically misleading!
            """)

        with st.expander("7. Random-Effects Model (DerSimonian-Laird): Study Weighting & Pooled Summary", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            * **Adjusted Random-Effects Weight ($w_i^*$):**
              $$w_i^* = \frac{1}{v_i + \tau^2} = \frac{1}{\text{SE}_i^2 + \tau^2}$$
            * **Summary Pooled Log-Effect ($\theta_{\text{random}}$):**
              $$\theta_{\text{random}} = \frac{\sum_{i=1}^K w_i^* y_i}{\sum_{i=1}^K w_i^*}$$
            * **Pooled Standard Error & 95% Confidence Interval:**
              $$\text{SE}(\theta_{\text{random}}) = \sqrt{\frac{1}{\sum_{i=1}^K w_i^*}}, \quad \text{CI}_{95\%} = \exp\left(\theta_{\text{random}} \pm 1.96 \times \text{SE}(\theta_{\text{random}})\right)$$

            #### Plain English Breakdown of the Formula:
            * **Why Add $\tau^2$ to the Denominator?** Under the Random-Effects philosophy, there are now **two separate sources of uncertainty**:
              1. Within-study sampling noise ($\text{SE}_i^2$)
              2. Real-world between-study clinical diversity ($\tau^2$)
            * Adding $\tau^2$ into the denominator levels the playing field: it prevents a single massive trial from receiving 95% of the weight and completely drowning out smaller clinical trials. It balances the weights across studies.

            #### Why This Calculation is Being Used:
            * When real-world clinical heterogeneity exists, the Fixed-Effect model produces artificially narrow confidence intervals that create a false sense of certainty.
            * The Random-Effects model correctly widens the confidence interval to incorporate between-study variance, providing an honest, clinically realistic estimate of treatment efficacy.
            """)

        with st.expander("8. Higgins 95% Prediction Interval: The Future Patient Reality Check", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            $$\text{PI}_{95\%} = \exp\left(\theta_{\text{random}} \pm t_{K-2, 0.975} \times \sqrt{\tau^2 + \text{SE}(\theta_{\text{random}})^2}\right)$$

            #### Plain English Breakdown of the Formula:
            * **$\theta_{\text{random}}$**: The center of the distribution of true study effects.
            * **$\text{SE}(\theta_{\text{random}})^2$**: The uncertainty in estimating that center average.
            * **$\tau^2$**: The spread of true study effects around that center across different hospitals.
            * **$t_{K-2, 0.975}$**: Student's $t$ critical value with $K - 2$ degrees of freedom (which penalizes small numbers of studies).

            #### Why This Calculation is Being Used:
            * The standard 95% Confidence Interval tells you where the *average* effect lies across past clinical trials.
            * But a clinician in clinic needs to know: *"If I use this drug tomorrow in my local hospital on a new patient population, what true effect might I expect?"*
            * The **Prediction Interval** answers that question. If the pooled 95% CI is $0.75 [0.65, 0.86]$ (statistically significant benefit), but the 95% Prediction Interval is $[0.40, 1.35]$ (crossing $1.00$), it warns doctors that in some future hospital populations, the treatment may fail to provide benefit!
            """)

        with st.expander("9. Egger's Linear Regression Test for Funnel Plot Asymmetry", expanded=False):
            st.markdown(r"""
            #### Mathematical Representation:
            $$\frac{y_i}{\text{SE}_i} = \alpha + \beta \left(\frac{1}{\text{SE}_i}\right) + \epsilon_i$$

            #### Plain English Breakdown of the Formula:
            * **$\frac{y_i}{\text{SE}_i}$ (Standard Normal Deviate, SND):** The study's effect size divided by its standard error (equivalent to a $Z$-statistic).
            * **$\frac{1}{\text{SE}_i}$ (Precision):** How large and statistically reliable the study is.
            * **$\beta$ (Slope):** Represents the overall pooled effect size.
            * **$\alpha$ (Intercept):** The value where the regression line hits the vertical axis as precision approaches zero.

            #### Why This Calculation is Being Used:
            * In the absence of publication bias, small studies with low precision should scatter evenly on both sides of the pooled effect, meaning the regression line will pass directly through the origin ($\alpha = 0$).
            * However, if researchers hide negative small studies in their file drawers, only small studies with exaggerated positive effects get published. This drags the regression line away from the origin, producing a statistically significant non-zero intercept ($\alpha \ne 0, p < 0.10$).
            * Egger's test provides an objective mathematical decision rule for publication bias rather than relying on subjective visual inspection of funnel plots.
            """)

    # -------------------------------------------------------------
    # TAB 2: Microsoft Excel Implementation Guide
    # -------------------------------------------------------------
    with tab_xls:
        st.markdown(r"""
        ### 📗 Step-by-Step Implementation in Microsoft Excel

        This tutorial guides you through building a complete, formula-driven Meta-Analysis spreadsheet in Microsoft Excel,
        including log transformations, Woolf variances, inverse-variance weights, Cochran's Q, DerSimonian-Laird tau-squared,
        Higgins I^2, Random-Effects summary estimates, and Egger's linear regression test.
        """)

        with st.expander("Spreadsheet Layout & Cell Coordinate Reference Table (A1:Q25)", expanded=False):
            st.markdown(r"""
            Set up your worksheet with the following column headers across row 3:

            | Column | Header | Description | Example Cell Formula (Row 4) |
            | :---: | :--- | :--- | :--- |
            | **A** | `Study_Name` | Name of clinical trial | `DAPA-HF 2019` |
            | **B** | `Year` | Publication year | `2019` |
            | **C** | `Treated_Events` ($a$) | Events in experimental arm | `386` |
            | **D** | `Treated_Total` ($n_1$) | Total patients in experimental arm | `2373` |
            | **E** | `Control_Events` ($c$) | Events in control arm | `502` |
            | **F** | `Control_Total` ($n_0$) | Total patients in control arm | `2371` |
            | **G** | `Odds_Ratio` | Individual trial Odds Ratio | `=(C4*(F4-E4)) / ((D4-C4)*E4)` |
            | **H** | `Log_OR` ($y_i$) | Natural log of Odds Ratio | `=LN(G4)` |
            | **I** | `Woolf_Variance` ($v_i$) | Within-study variance ($\text{SE}_i^2$) | `=1/C4 + 1/(D4-C4) + 1/E4 + 1/(F4-E4)` |
            | **J** | `SE_Log_OR` ($\text{SE}_i$) | Standard error of log effect | `=SQRT(I4)` |
            | **K** | `Fixed_Weight` ($w_i$) | Inverse-variance weight ($1 / v_i$) | `=1/I4` |
            | **L** | `Weighted_Effect` ($w_i \cdot y_i$) | Product of weight and log effect | `=K4*H4` |
            | **M** | `Q_Contribution` | Study contribution to Cochran's Q | `=K4 * ((H4 - $K$15)^2)` |
            | **N** | `Weight_Squared` ($w_i^2$) | Squared weight for DL denominator | `=K4^2` |
            | **O** | `Random_Weight` ($w_i^*$) | Adjusted weight ($1 / (v_i + \tau^2)$) | `=1 / (I4 + $K$18)` |
            | **P** | `Random_Weighted_Effect` | Product of random weight and effect | `=O4*H4` |
            | **Q** | `SND_i` | Standard Normal Deviate for Egger | `=H4/J4` |
            """)

        with st.expander("Step 1: Ingesting Trial Data & Computing 2x2 Contingency Margins", expanded=False):
            st.markdown(r"""
            1. Enter trial identifiers in columns `A` and `B` from row 4 to row 13.
            2. Enter raw 2x2 event counts in columns `C`, `D`, `E`, and `F`:
               * `C4`: Experimental events ($a$)
               * `D4`: Experimental total ($n_1 = a + b$)
               * `E4`: Control events ($c$)
               * `F4`: Control total ($n_0 = c + d$)
            3. *Continuity Correction Note:* If any trial contains 0 events in either arm, add `0.5` to all four cells (`C4+0.5`, `D4+1.0`, etc.) to prevent division-by-zero errors.
            """)

        with st.expander("Step 2: Calculating Log-Odds Ratios and Woolf Variances in Excel", expanded=False):
            st.markdown(r"""
            1. In cell `G4`, calculate the trial Odds Ratio:
               `=(C4*(F4-E4)) / ((D4-C4)*E4)`
            2. In cell `H4`, compute the natural log-effect size $y_i$:
               `=LN(G4)`
            3. In cell `I4`, compute the Woolf within-study variance $v_i = \\text{SE}_i^2$:
               `=1/C4 + 1/(D4-C4) + 1/E4 + 1/(F4-E4)`
            4. In cell `J4`, compute the Standard Error:
               `=SQRT(I4)`
            5. In cell `K4`, compute the Fixed-Effect Inverse-Variance Weight $w_i$:
               `=1/I4`
            6. In cell `L4`, compute the weighted effect $w_i \\times y_i$:
               `=K4*H4`
            7. Copy formulas down across all study rows (`G4:L13`).
            """)

        with st.expander("Step 3: Calculating Fixed-Effect Pooled Summary Estimate & Confidence Intervals", expanded=False):
            st.markdown(r"""
            In row 15, calculate the summary sums:
            1. In cell `K14`, sum the weights $\\sum w_i$:
               `=SUM(K4:K13)`
            2. In cell `L14`, sum the weighted effects $\\sum w_i y_i$:
               `=SUM(L4:L13)`
            3. In cell `K15`, compute the Fixed-Effect pooled log summary $\\theta_{\\text{fixed}}$:
               `=L14/K14`
            4. In cell `K16`, compute the Standard Error of the pooled effect:
               `=SQRT(1/K14)`
            5. In cell `K17`, exponentiate to return to the natural Odds Ratio scale:
               `=EXP(K15)`
            6. In cells `L17` and `M17`, compute the 95% Wald Confidence Limits:
               * Lower 95% CI: `=EXP(K15 - 1.96 * K16)`
               * Upper 95% CI: `=EXP(K15 + 1.96 * K16)`
            """)

        with st.expander("Step 4: Calculating Cochran's Q Heterogeneity Statistic & Chi-Square P-Value", expanded=False):
            st.markdown(r"""
            1. In column `M4`, compute each individual study's contribution to Cochran's Q:
               `=K4 * ((H4 - $K$15)^2)`
            2. Copy down across all study rows (`M4:M13`).
            3. In cell `M14`, compute the total Cochran's Q statistic:
               `=SUM(M4:M13)`
            4. In cell `M15`, compute the degrees of freedom: `=COUNT(A4:A13) - 1` (e.g. `9`).
            5. In cell `M16`, compute the right-tailed Chi-square p-value using Excel's native function:
               `=CHISQ.DIST.RT(M14, M15)`
            6. *Interpretation:* If cell `M16` is $< 0.10$, between-study heterogeneity is statistically significant!
            """)

        with st.expander("Step 5: Computing DerSimonian-Laird Tau-Squared (τ²) and Higgins I²", expanded=False):
            st.markdown(r"""
            1. In column `N4`, compute squared weights $w_i^2$:
               `=K4^2`
            2. In cell `N14`, sum the squared weights: `=SUM(N4:N13)`.
            3. In cell `K18`, compute the DerSimonian-Laird scaling constant $C$:
               `=K14 - (N14 / K14)`
            4. In cell `K19`, compute between-study variance $\\tau^2$:
               `=MAX(0, (M14 - M15) / K18)`
            5. In cell `K20`, compute Higgins $I^2$ percentage:
               `=MAX(0, (M14 - M15) / M14) * 100`
            6. *Interpretation:* If $I^2 \\ge 30\\%$, you must report the Random-Effects summary rather than the Fixed-Effect estimate!
            """)

        with st.expander("Step 6: Calculating Random-Effects Pooled Summary & Higgins 95% Prediction Interval", expanded=False):
            st.markdown(r"""
            1. In column `O4`, compute adjusted Random-Effects weights $w_i^*$:
               `=1 / (I4 + $K$19)`
            2. In column `P4`, compute weighted random effect $w_i^* \\times y_i$:
               `=O4*H4`
            3. In row 14:
               * Cell `O14`: `=SUM(O4:O13)` (Total random weight)
               * Cell `P14`: `=SUM(P4:P13)` (Total weighted random effect)
            4. In cell `O15`, compute the Random-Effects summary log-effect $\\theta_{\\text{random}}$:
               `=P14/O14`
            5. In cell `O16`, compute the Standard Error:
               `=SQRT(1/O14)`
            6. In cell `O17`, exponentiate to obtain the Random-Effects Pooled Odds Ratio:
               `=EXP(O15)`
            7. In cell `O18` and `P18`, compute 95% Random-Effects Confidence Limits:
               * Lower CI: `=EXP(O15 - 1.96 * O16)`
               * Upper CI: `=EXP(O15 + 1.96 * O16)`
            8. In cells `O19` and `P19`, compute Higgins 95% Prediction Interval using Student's t critical value:
               * Critical t value: `=T.INV.2T(0.05, M15 - 1)`
               * Prediction Lower: `=EXP(O15 - T.INV.2T(0.05, M15-1) * SQRT($K$19 + O16^2))`
               * Prediction Upper: `=EXP(O15 + T.INV.2T(0.05, M15-1) * SQRT($K$19 + O16^2))`
            """)

        with st.expander("Step 7: Executing Egger's Linear Regression Test for Publication Bias in Excel", expanded=False):
            st.markdown(r"""
            1. In column `Q4`, compute Standard Normal Deviate $SND_i = y_i / \\text{SE}_i$:
               `=H4/J4`
            2. In column `R4`, compute precision $1 / \\text{SE}_i$:
               `=1/J4`
            3. In cell `Q21`, compute the regression intercept using Excel's native `INTERCEPT` function:
               `=INTERCEPT(Q4:Q13, R4:R13)`
            4. In cell `Q22`, compute the slope using `SLOPE`:
               `=SLOPE(Q4:Q13, R4:R13)`
            5. Use Excel's `Data Analysis Toolpak -> Regression` selecting `Q4:Q13` as Y and `R4:R13` as X to view the two-tailed t-test and p-value for the intercept.
            6. *Decision Rule:* If the intercept p-value is $< 0.10$, funnel plot asymmetry and publication bias are statistically detected!
            """)

        with st.expander("Step 8: Constructing a Forest Plot in Excel", expanded=False):
            st.markdown(r"""
            1. Create a 3-column summary table containing: `Study_Name`, `Odds_Ratio`, `CI_Lower`, `CI_Upper`.
            2. Compute positive error: `CI_Upper - Odds_Ratio`.
            3. Compute negative error: `Odds_Ratio - CI_Lower`.
            4. Select `Study_Name` and `Odds_Ratio`, and insert a **Scatter Plot with Straight Lines and Markers** or a **Horizontal Bar Chart**.
            5. Click on data series -> `Add Chart Element -> Error Bars -> More Error Bar Options`.
            6. Set error bar direction to `Both` and specify `Custom Error Amounts` referencing your positive and negative error columns.
            7. Add a vertical reference line at $X = 1.00$ to demarcate the null effect.
            """)

    # -------------------------------------------------------------
    # TAB 3: Python Implementation Guide
    # -------------------------------------------------------------
    with tab_py:
        st.markdown(r"""
        ### 🐍 Production-Ready Python Script Guide

        This guide presents vectorized, production-grade Python code utilizing standard scientific libraries
        (`numpy`, `pandas`, `scipy.stats`) to execute the complete Meta-Analysis pipeline.
        """)

        with st.expander("Stage 1: Ingesting Trial Microdata & Calculating Study Log-Effects and SEs", expanded=False):
            st.markdown(r"""
            **Purpose:** Takes raw 2x2 counts ($a, b, c, d$), applies Haldane 0.5 continuity correction for sparse cells,
            and calculates log-odds ratios ($y_i$) and Woolf standard errors ($\text{SE}_i$).
            """)
            st.code(r'''
import numpy as np
import pandas as pd
from scipy import stats

def prepare_study_effects(studies_df, effect_metric="OR"):
    """Computes log-effect sizes (yi) and standard errors (sei) from 2x2 counts."""
    df = studies_df.copy()
    
    a = df['treated_events'].astype(float)
    n1 = df['treated_total'].astype(float)
    b = n1 - a
    c = df['control_events'].astype(float)
    n0 = df['control_total'].astype(float)
    d = n0 - c
    
    # 0.5 continuity correction for zero cells
    has_zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    adj_a = np.where(has_zero, a + 0.5, a)
    adj_b = np.where(has_zero, b + 0.5, b)
    adj_c = np.where(has_zero, c + 0.5, c)
    adj_d = np.where(has_zero, d + 0.5, d)
    
    if effect_metric == "OR":
        df['yi'] = np.log((adj_a * adj_d) / (adj_b * adj_c))
        df['vi'] = 1/adj_a + 1/adj_b + 1/adj_c + 1/adj_d
        df['sei'] = np.sqrt(df['vi'])
    else:  # RR
        risk1 = adj_a / (adj_a + adj_b)
        risk0 = adj_c / (adj_c + adj_d)
        df['yi'] = np.log(risk1 / risk0)
        df['vi'] = (1/adj_a - 1/(adj_a + adj_b)) + (1/adj_c - 1/(adj_c + adj_d))
        df['sei'] = np.sqrt(df['vi'])
        
    return df
''', language="python")

        with st.expander("Stage 2: Vectorized Calculation of Fixed & Random Effects and Heterogeneity", expanded=False):
            st.markdown(r"""
            **Purpose:** Vectorized computation of inverse-variance weights, Cochran's $Q$, DerSimonian-Laird $\tau^2$,
            Higgins $I^2$, and random-effects pooled estimates.
            """)
            st.code(r'''
def compute_meta_analysis(df):
    """Calculates Fixed and DerSimonian-Laird Random Effects, Cochran Q, tau2, and I2."""
    k = len(df)
    
    # Fixed effect weights
    w_fixed = 1.0 / df['vi']
    sum_w = w_fixed.sum()
    sum_w_sq = (w_fixed ** 2).sum()
    theta_fixed = (w_fixed * df['yi']).sum() / sum_w
    se_fixed = np.sqrt(1.0 / sum_w)
    
    # Cochran's Q test for heterogeneity
    q_stat = (w_fixed * ((df['yi'] - theta_fixed) ** 2)).sum()
    df_q = max(1, k - 1)
    p_q = stats.chi2.sf(q_stat, df=df_q)
    
    # DerSimonian-Laird tau^2 and Higgins I^2
    c_factor = sum_w - (sum_w_sq / sum_w)
    tau2 = max(0.0, (q_stat - df_q) / c_factor) if c_factor > 0 else 0.0
    i2 = max(0.0, (q_stat - df_q) / q_stat * 100.0) if q_stat > 0 else 0.0
    
    # Random effects weights
    w_rand = 1.0 / (df['vi'] + tau2)
    sum_w_rand = w_rand.sum()
    theta_rand = (w_rand * df['yi']).sum() / sum_w_rand
    se_rand = np.sqrt(1.0 / sum_w_rand)
    
    # Higgins 95% Prediction Interval
    t_crit = stats.t.ppf(0.975, df=max(1, k - 2))
    se_pred = np.sqrt(tau2 + se_rand**2)
    pred_low = np.exp(theta_rand - t_crit * se_pred)
    pred_upp = np.exp(theta_rand + t_crit * se_pred)
    
    return {
        'fixed_effect_or': np.exp(theta_fixed),
        'fixed_ci': [np.exp(theta_fixed - 1.96 * se_fixed), np.exp(theta_fixed + 1.96 * se_fixed)],
        'random_effect_or': np.exp(theta_rand),
        'random_ci': [np.exp(theta_rand - 1.96 * se_rand), np.exp(theta_rand + 1.96 * se_rand)],
        'prediction_interval_95': [pred_low, pred_upp],
        'cochran_q': q_stat, 'q_p_value': p_q,
        'tau2': tau2, 'i2_pct': i2
    }
''', language="python")

        with st.expander("Stage 3: Leave-One-Out Sensitivity Analysis Loop", expanded=False):
            st.markdown(r"""
            **Purpose:** Sequentially drops one study at a time and recalculates pooled estimates to detect outlier trials.
            """)
            st.code(r'''
def run_leave_one_out_sensitivity(df):
    """Sequentially recalculates random-effects pooled estimate omitting each study."""
    loo_results = []
    k = len(df)
    
    for i in range(k):
        sub_df = df.drop(df.index[i])
        res = compute_meta_analysis(sub_df)
        loo_results.append({
            'Omitted_Study': df.iloc[i]['Study'],
            'Pooled_OR': res['random_effect_or'],
            'CI_Lower': res['random_ci'][0],
            'CI_Upper': res['random_ci'][1],
            'I2_Pct': res['i2_pct']
        })
        
    return pd.DataFrame(loo_results)
''', language="python")

        with st.expander("Stage 4: Egger's Linear Regression Test for Funnel Plot Asymmetry", expanded=False):
            st.markdown(r"""
            **Purpose:** Regresses standardized effect sizes against precision to detect small-study publication bias.
            """)
            st.code(r'''
def egger_regression_test(yi, sei):
    """Calculates Egger's linear regression test for funnel plot asymmetry."""
    snd = yi / sei
    precision = 1.0 / sei
    k = len(yi)
    
    # OLS regression of SND on Precision
    slope, intercept, r_val, p_val, std_err = stats.linregress(precision, snd)
    
    # Two-tailed t-test on intercept
    df_res = k - 2
    residuals = snd - (intercept + slope * precision)
    s2 = (residuals ** 2).sum() / max(1, df_res)
    ss_prec = ((precision - precision.mean()) ** 2).sum()
    se_intercept = np.sqrt(s2 * (1.0/k + (precision.mean()**2)/ss_prec))
    t_stat = intercept / se_intercept
    p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=df_res)
    
    return {
        'intercept': intercept,
        't_statistic': t_stat,
        'p_value': p_value,
        'bias_detected': p_value < 0.10
    }
''', language="python")

        with st.expander("Stage 5: Complete Standalone End-to-End Production Meta-Analysis Script", expanded=False):
            st.code(r'''
# Complete Standalone Meta-Analysis Pipeline
from src.epi_methods.meta_analysis import (
    calculate_meta_analysis,
    evaluate_meta_analysis_prerequisites,
    generate_meta_analysis_demo_data
)

# 1. Load Synthetic Demonstration Trial Matrix
demo = generate_meta_analysis_demo_data("cardiovascular_sglt2")
print(f"Evidence Synthesis: {demo['scenario_name']}")

# 2. Run Meta-Analysis Calculations
results = calculate_meta_analysis(demo['studies'], effect_metric="OR")
evals = evaluate_meta_analysis_prerequisites(results)

# 3. Print Clinical & Methodological Summary
print("=" * 65)
print(f"Fixed-Effect Pooled OR:       {results['fixed_effect_natural']:.2f} {results['fixed_effect_ci_str']}")
print(f"Random-Effects Pooled OR:     {results['random_effect_natural']:.2f} {results['random_effect_ci_str']}")
print(f"Heterogeneity I²:             {results['i2_pct']:.1f}% ({results['heterogeneity_category']})")
print(f"Cochran's Q Statistic:        {results['cochran_q']:.2f} (p = {results['cochran_q_p_value']:.4f})")
print(f"Between-Study Variance τ²:    {results['tau2']:.4f}")
print(f"Higgins 95% Prediction Int:   {results['prediction_interval_str']}")
print(f"Egger's Test Intercept:       {results['egger_intercept']:.2f} (p = {results['egger_p_value']:.4f})")
print(f"Preferred Model:              {results['preferred_model']}")
print("=" * 65)
print(results['clinical_recommendation'])
''', language="python")

    # -------------------------------------------------------------
    # TAB 4: Data Downloads
    # -------------------------------------------------------------
    with tab_dl:
        st.markdown(r"""
        ### 💾 Live Synthetic Demonstration Data Downloads

        All datasets provided below are **100% synthetic, computer-generated demonstration files** created for teaching
        and code rehearsal. They contain **zero real patient data or protected health information**.
        """)

        st.warning("""
        ⚠️ **EDUCATIONAL DISCLAIMER & NOTICE**:
        * **100% Synthetic / Artificial Data:** All patient records, event numbers, trial labels, and follow-up data were synthesized via deterministic mathematical simulation.
        * **Educational Rehearsal Only:** Designed for university teaching, Cochrane evidence synthesis workshops, and practicing meta-analysis in Excel, R (`metafor`), Python, Stata, or SAS.
        * **No Clinical Utility:** Must never be used for real patient treatment guidelines, drug licensing, or commercial pharmaceutical submissions.
        """)

        c_d1, c_d2 = st.columns(2)

        df_micro = demo_data.get("df", pd.DataFrame())
        if not isinstance(df_micro, pd.DataFrame):
            df_micro = pd.DataFrame(df_micro)

        with c_d1:
            with st.container(border=True):
                st.markdown(f"#### 📄 Synthetic Patient Microdata ({len(df_micro)} Simulated Records)")
                st.markdown(f"Contains **{len(df_micro)} individual patient microdata records** including simulated patient IDs, assigned trial arm, binary treatment status, and cardiovascular outcome.")
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
                st.markdown("#### 📊 Meta-Analysis Study Evidence Matrix (CSV)")
                st.markdown("Aggregated summary of included clinical trials, 2x2 event numbers, natural effect sizes, Woolf confidence intervals, and Fixed/Random percentage weights.")
                st.caption("*Derived from the simulated synthetic cohort for tutorial exercises.*")
                df_studies = pd.DataFrame(results.get("studies_list", []))
                csv_studies = df_studies.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Study Evidence Matrix CSV",
                    data=csv_studies,
                    file_name=f"synthetic_{meta['scenario_key']}_evidence_matrix.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                with st.expander("👁️ Preview Study Evidence Matrix Table", expanded=False):
                    st.dataframe(df_studies, use_container_width=True, hide_index=True)

        # Download Card 3: Leave-One-Out Sensitivity Analysis
        with st.container(border=True):
            st.markdown("#### 🔄 Leave-One-Out Sensitivity Analysis Dataset (CSV)")
            st.markdown("Recalculated pooled effect sizes, 95% confidence intervals, and Higgins $I^2$ percentages sequentially omitting each study.")
            st.caption("*Used to assess whether a single trial acts as an influential outlier.*")
            df_loo = pd.DataFrame(results.get("leave_one_out_list", []))
            csv_loo = df_loo.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Leave-One-Out Sensitivity Summary CSV",
                data=csv_loo,
                file_name=f"synthetic_{meta['scenario_key']}_leave_one_out.csv",
                mime="text/csv",
                use_container_width=True
            )
            with st.expander("👁️ Preview Leave-One-Out Table", expanded=False):
                st.dataframe(df_loo, use_container_width=True, hide_index=True)

        with st.expander("📋 Synthetic Data Generation Methodology & Variable Dictionary", expanded=False):
            st.markdown(r"""
            #### How This Synthetic Dataset Was Created
            1. **Trial Simulation:** Clinical trials are parameterized with varying sample sizes ($N = 100$ to $3,000$) and baseline event risks reflecting realistic randomized multicenter cohorts.
            2. **Effect Size Sampling:** True log-odds ratios are sampled from Gaussian distributions with specified between-study variance ($\tau^2$) to simulate homogeneous, moderate, or severe heterogeneity scenarios.
            3. **Publication Bias Mechanism:** In the publication bias scenario, small studies with neutral or adverse effects are selectively omitted to realistically demonstrate funnel plot asymmetry and Egger regression significance.
            4. **Information Governance:** Because this dataset is entirely synthetic, it is safe to share, publish, and use in educational workshops without information governance, Caldicott Guardian, or GDPR restrictions.
            """)


def render_meta_analysis_module(active_method_key: str, method_info: Dict[str, Any]):
    """Main rendering controller for Module 9: Meta-Analysis & Evidence Synthesis."""

    # 1. Plain English Pedagogical Introductory Guide
    render_meta_analysis_plain_english_guide()

    # 2. Scenario Selection (Available on Main Page & Sidebar)
    scenario_options = {
        "cardiovascular_sglt2": "All Meadows Cardiovascular Institute (SGLT2 Inhibitors: 10 RCTs, Low I²) [Synthetic]",
        "respiratory_masks": "Riverside Health Protection Team (Community Masks: 8 Cohorts, Substantial I²) [Synthetic]",
        "anticoagulation_elderly": "Inner City Teaching Hospital (Anticoagulation in Elderly: Publication Bias) [Synthetic]",
        "custom": "Custom User-Defined Meta-Analysis Matrix [Synthetic]"
    }

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        selected_scenario = st.selectbox(
            "Select Evidence Synthesis Clinical Scenario:",
            options=list(scenario_options.keys()),
            format_func=lambda k: scenario_options[k],
            index=0
        )
    with col_s2:
        eff_metric = st.selectbox(
            "Effect Measure:",
            options=["OR", "RR"],
            index=0,
            help="Odds Ratio (OR) is standard for case-control and trial meta-analysis. Relative Risk (RR) is used for cohort trials."
        )

    if selected_scenario != "custom":
        demo_data = generate_meta_analysis_demo_data(scenario_key=selected_scenario)
        studies_input = demo_data["studies"]
        meta = {
            "scenario_key": selected_scenario,
            "title": demo_data["scenario_name"],
            "organisation": demo_data["organisation_name"],
            "domain": demo_data["domain"],
            "intervention": demo_data["intervention"],
            "comparator": demo_data["comparator"],
            "outcome": demo_data["outcome"],
            "description": demo_data["description"]
        }
    else:
        st.markdown("#### ⚙️ Custom Study Matrix Parameters")
        num_studies = st.slider("Number of Studies (K)", min_value=3, max_value=8, value=4, step=1)
        studies_input = []
        for i in range(num_studies):
            st.markdown(f"**Study {i+1}:**")
            c_a, c_b = st.columns(2)
            with c_a:
                t_ev = st.number_input(f"Treated Events (Study {i+1})", min_value=1, max_value=1000, value=30 + i*10, key=f"ma_tev_{i}")
                t_tot = st.number_input(f"Treated Total (Study {i+1})", min_value=10, max_value=5000, value=200 + i*50, key=f"ma_ttot_{i}")
            with c_b:
                c_ev = st.number_input(f"Control Events (Study {i+1})", min_value=1, max_value=1000, value=45 + i*15, key=f"ma_cev_{i}")
                c_tot = st.number_input(f"Control Total (Study {i+1})", min_value=10, max_value=5000, value=200 + i*50, key=f"ma_ctot_{i}")
            studies_input.append({
                "Study": f"Custom Trial {i+1}",
                "Year": 2020 + i,
                "treated_events": t_ev,
                "treated_total": t_tot,
                "control_events": c_ev,
                "control_total": c_tot
            })

        demo_data = {
            "df": pd.DataFrame({"Custom_Data": [1]}),
            "studies": studies_input,
            "effect_metric": eff_metric
        }
        meta = {
            "scenario_key": "custom",
            "title": "Custom User-Defined Meta-Analysis Simulation",
            "organisation": "Riverside Health Protection Team (Evidence Synthesis)",
            "domain": "Custom Clinical Systematic Review",
            "intervention": "Custom Medical Intervention",
            "comparator": "Custom Comparator",
            "outcome": "Custom Evaluated Clinical Outcome",
            "description": "User-configured systematic review evidence synthesis."
        }

    # 3. Core Calculations
    results = calculate_meta_analysis(studies_input, effect_metric=eff_metric)
    evaluations = evaluate_meta_analysis_prerequisites(results)

    # 4. Header Banner & Prominent Synthetic Disclaimer
    st.markdown(f"## 📚 {meta['title']}")
    st.markdown(f"**Organisation:** {meta['organisation']} | **Domain:** {meta['domain']}")
    st.markdown(f"**Intervention:** `{meta['intervention']}` | **Comparator:** `{meta['comparator']}` | **Primary Outcome:** `{meta['outcome']}`")
    st.markdown(f"*{meta['description']}*")

    with st.container(border=True):
        st.markdown(r"""
        > [!IMPORTANT]
        > **EDUCATIONAL & SYNTHETIC DATA DISCLAIMER**: The clinical trials, patient numbers, and meta-analytic metrics displayed below are **100% synthetic and computer-generated** purely for academic training, systematic review education, and software demonstration. Contains no real patient identifiers or protected health information.
        """)

    # 5. Methodological Prerequisite Cards
    render_meta_analysis_prerequisite_cards(evaluations)

    # 6. Primary Metric Dashboard
    st.markdown("### 📈 Primary Pooled Effect Measures & Heterogeneity Diagnostics")
    render_meta_analysis_metric_dashboard(results)

    # 7. Interactive Visualizations
    col_v1, col_v2 = st.columns([1.25, 0.75])
    with col_v1:
        with st.container(border=True):
            fig_forest = build_meta_analysis_forest_plot(results, meta)
            st.plotly_chart(fig_forest, use_container_width=True)

    with col_v2:
        with st.container(border=True):
            fig_funnel = build_meta_analysis_funnel_plot(results, meta)
            st.plotly_chart(fig_funnel, use_container_width=True)

    # Explanatory Guide Beneath Plots
    render_meta_analysis_explanatory_box(results)

    # 8. Detailed Study Evidence Table
    render_meta_analysis_evidence_table(results)

    # 9. Full 4-Tab Educational Architecture
    st.markdown("---")
    render_meta_analysis_educational_tabs(results, meta, demo_data, method_info)

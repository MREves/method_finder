"""Interactive UI Components & Visualizations for Kaplan-Meier Survival Analysis & Log-Rank Test.

Provides:
1. Plain English Pedagogical Guide to Survival Analysis and Right-Censoring.
2. Sidebar scenario controls with three realistic clinical scenarios and custom time-to-event entry.
3. Methodological Prerequisite Evaluation (STROBE/CONSORT non-informative censoring, proportional hazards).
4. Interactive Metric Dashboard (Median survival, Log-Rank Chi2/p-value, Peto Hazard Ratio, NNT).
5. High-precision Plotly Step-plots with shaded Greenwood confidence bands, censoring tick-marks, and Number-at-Risk table.
6. Interactive Landmark Survival Time Simulator (evaluating S(t) and NNT at any time point).
7. Complete 4-Tab Educational Architecture:
   - 📋 Rationale & Epidemiological Framework (Zero prior knowledge, LaTeX + Plain English).
   - 📗 Microsoft Excel Implementation Guide (Step-by-step formulas: COUNTIFS, PRODUCT, Greenwood, CHISQ.DIST.RT).
   - 🐍 Python Production Script Guide (Clean standalone script with lifelines / scipy).
   - 💾 Live Data Downloads (Patient-level microdata and at-risk summary life tables).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.kaplan_meier import (
    calculate_kaplan_meier_single_group,
    calculate_log_rank_test,
    calculate_landmark_survival_probability,
    calculate_kaplan_meier_two_groups,
    evaluate_kaplan_meier_prerequisites,
    generate_kaplan_meier_demo_data
)


def render_kaplan_meier_plain_english_guide():
    """Renders a collapsible plain English introductory guide to survival analysis."""
    with st.expander("📖 Zero-Knowledge Guide: What is Survival Analysis, Right-Censoring & The Log-Rank Test?", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. Why Time Matters: Beyond "Did It Happen?"
            * In standard epidemiological studies (like Relative Risk or Odds Ratio), we ask: **"Did the patient experience the event?"** (A simple Yes/No headcount).
            * But in chronic disease, clinical oncology, and surgery, the crucial question is: **"WHEN did the event happen, and HOW LONG did the patient remain healthy?"**
            * Two treatments might both have a 50% mortality rate at 5 years. But if Treatment A keeps patients alive and healthy for 4.5 years before relapse, while Treatment B leads to relapse at 6 months, Treatment A is clinically superior!
            * **Survival Analysis (Time-to-Event Analysis)** tracks the trajectory of health over time, accounting for every single day of event-free survival.
            """)

            st.markdown(r"""
            #### 2. The Unique Challenge: "Right-Censoring"
            * In longitudinal research, patients enter at staggered dates and leave at different times.
            * When the study ends, many patients are **still alive and healthy**. Other patients may move away or be lost to follow-up.
            * We know these patients survived *at least* until their last clinic visit, but we do not know when they will eventually experience the event. This incomplete observation is called **Right-Censoring**.
            * **Why Standard Statistics Fail:**
              * Throwing away censored patients introduces massive bias (ignoring healthy survivors).
              * Treating censored patients as if they died today falsely exaggerates danger.
              * Treating them as surviving forever underestimates risk.
            * **Kaplan & Meier (1958)** solved this with the **Product-Limit Estimator**, which computes survival conditionally at each event time using everyone still under observation!
            """)

        with col_g2:
            st.markdown(r"""
            #### 3. How the Kaplan-Meier Curve Works
            * The curve starts at **100% ($1.0$)** at baseline ($t = 0$).
            * **Between events, the curve is flat.**
            * **Whenever a confirmed event occurs**, the curve steps downward by a factor equal to the proportion of patients at risk who just experienced the event ($1 - d_i / n_i$).
            * **Whenever a patient is censored**, the curve does *not* step downward; instead, the number of patients at risk simply decreases for future time points (indicated by small vertical tick marks).
            * **Median Survival Time:** The exact point in time when survival drops to $50\%$ (half the cohort has experienced the event).
            """)

            st.markdown(r"""
            #### 4. The Log-Rank Test: "Observed vs. Expected Events"
            * How do we prove that a new drug or surgical technique significantly extends life compared to standard care?
            * The **Mantel-Cox Log-Rank Test** compares the two survival curves across their entire length.
            * At every single moment an event occurs, it asks: *If there were truly no difference between the treatments, how many events would we expect in each group given how many patients were at risk?*
            * It compares **Total Observed Events ($O$)** against **Total Expected Events ($E$)**.
            * If the new treatment has far fewer observed events than expected ($O \ll E$), the test generates a statistically significant Chi-Square ($\chi^2$) and p-value ($p < 0.05$).
            """)


def render_kaplan_meier_diagnostics(evaluations: Dict[str, Any]):
    """Renders diagnostic evaluation cards for survival analysis assumptions."""
    st.markdown("#### 🛡️ Methodological Prerequisite Evaluation (STROBE / CONSORT Standards)")

    cols = st.columns(len(evaluations))
    for i, (key, item) in enumerate(evaluations.items()):
        with cols[i]:
            status = item["status"]
            if status == "pass":
                border_color = "#10b981"
                bg_color = "#ecfdf5"
                text_color = "#065f46"
            elif status == "warning":
                border_color = "#f59e0b"
                bg_color = "#fffbeb"
                text_color = "#92400e"
            else:
                border_color = "#ef4444"
                bg_color = "#fef2f2"
                text_color = "#991b1b"

            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1.5px solid {border_color}; border-radius: 8px; padding: 0.85rem; height: 100%; min-height: 140px;">
                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: {text_color}; margin-bottom: 0.25rem;">
                    {item['badge']}
                </div>
                <div style="font-size: 0.90rem; font-weight: 600; color: #1e293b; margin-bottom: 0.4rem; line-height: 1.25;">
                    {item['label']}
                </div>
                <div style="font-size: 0.80rem; color: #475569; line-height: 1.35;">
                    {item['details']}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_kaplan_meier_dashboard(km_analysis: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the primary metrics dashboard comparing both survival arms."""
    st.markdown("### 📊 Longitudinal Survival Analysis Dashboard")

    km1 = km_analysis["km_group_1"]
    km2 = km_analysis["km_group_2"]
    lr = km_analysis["log_rank"]
    time_unit = meta.get("time_unit", "Months")

    g1_name = km_analysis["group_1_name"]
    g2_name = km_analysis["group_2_name"]

    med1_str = f"{km1['median_survival_time']:.1f} {time_unit}" if km1["median_reached"] else "Not Reached (>50% alive)"
    med2_str = f"{km2['median_survival_time']:.1f} {time_unit}" if km2["median_reached"] else "Not Reached (>50% alive)"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container(border=True):
            st.metric(
                label=f"Median Survival: {g1_name}",
                value=med1_str,
                help="The earliest follow-up time where cumulative survival drops to 50%."
            )
            ci1 = km1["median_ci"]
            if ci1[0] is not None and ci1[1] is not None:
                st.caption(f"95% CI: [{ci1[0]:.1f}, {ci1[1]:.1f}] {time_unit}")
            else:
                st.caption("95% CI: Not reached in follow-up")

    with c2:
        with st.container(border=True):
            st.metric(
                label=f"Median Survival: {g2_name}",
                value=med2_str,
                help="The earliest follow-up time where cumulative survival drops to 50%."
            )
            ci2 = km2["median_ci"]
            if ci2[0] is not None and ci2[1] is not None:
                st.caption(f"95% CI: [{ci2[0]:.1f}, {ci2[1]:.1f}] {time_unit}")
            else:
                st.caption("95% CI: Not reached in follow-up")

    with c3:
        with st.container(border=True):
            p_val = lr["p_value"]
            p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
            st.metric(
                label="Log-Rank Test (Mantel-Cox)",
                value=f"p = {p_str}",
                delta=f"Chi2 = {lr['chisq']:.2f} (df=1)",
                delta_color="normal"
            )
            st.caption("🟢 Statistically Significant" if lr["is_significant"] else "⚪ Not Significant (p >= 0.05)")

    with c4:
        with st.container(border=True):
            hr = lr["peto_hazard_ratio"]
            hr_ci = lr["peto_hr_ci"]
            reduction = (1.0 - hr) * 100.0 if hr < 1.0 else (hr - 1.0) * 100.0
            delta_label = f"-{reduction:.1f}% Hazard" if hr < 1.0 else f"+{reduction:.1f}% Hazard"
            st.metric(
                label=f"Peto Hazard Ratio ({g1_name} vs {g2_name})",
                value=f"{hr:.2f}",
                delta=delta_label,
                delta_color="inverse" if hr < 1.0 else "normal"
            )
            st.caption(f"95% CI: [{hr_ci[0]:.2f}, {hr_ci[1]:.2f}]")


def render_kaplan_meier_visualizations(km_analysis: Dict[str, Any], meta: Dict[str, Any]):
    """Renders side-by-side Plotly Kaplan-Meier step-curves with Greenwood bands and balanced explanations."""
    st.markdown("### 📈 Kaplan-Meier Curves & Longitudinal Survival Trajectories")

    km1 = km_analysis["km_group_1"]
    km2 = km_analysis["km_group_2"]
    lr = km_analysis["log_rank"]
    time_unit = meta.get("time_unit", "Months")
    g1_name = km_analysis["group_1_name"]
    g2_name = km_analysis["group_2_name"]

    col_plot, col_info = st.columns([1.5, 1.0])

    with col_plot:
        fig = go.Figure()

        # Step line for Group 1
        fig.add_trace(go.Scatter(
            x=km1["table_times"],
            y=km1["survival_probabilities"],
            mode="lines",
            name=f"{g1_name}",
            line=dict(color="#0284c7", width=3, shape="hv"),
            hovertemplate=f"<b>{g1_name}</b><br>Time: %{{x:.1f}} {time_unit}<br>Survival: %{{y:.1%}}<extra></extra>"
        ))

        # Upper and lower Greenwood confidence bounds for Group 1
        fig.add_trace(go.Scatter(
            x=km1["table_times"],
            y=km1["ci_upper_loglog"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=km1["table_times"],
            y=km1["ci_lower_loglog"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(2, 132, 199, 0.15)",
            name=f"{g1_name} 95% CI",
            hoverinfo="skip"
        ))

        # Step line for Group 2
        fig.add_trace(go.Scatter(
            x=km2["table_times"],
            y=km2["survival_probabilities"],
            mode="lines",
            name=f"{g2_name}",
            line=dict(color="#e11d48", width=3, shape="hv", dash="dash"),
            hovertemplate=f"<b>{g2_name}</b><br>Time: %{{x:.1f}} {time_unit}<br>Survival: %{{y:.1%}}<extra></extra>"
        ))

        # Upper and lower Greenwood confidence bounds for Group 2
        fig.add_trace(go.Scatter(
            x=km2["table_times"],
            y=km2["ci_upper_loglog"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=km2["table_times"],
            y=km2["ci_lower_loglog"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(225, 29, 72, 0.12)",
            name=f"{g2_name} 95% CI",
            hoverinfo="skip"
        ))

        # Reference 50% Median Line
        fig.add_hline(
            y=0.50,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text="50% Median Survival Threshold",
            annotation_position="bottom right"
        )

        fig.update_layout(
            title=f"Kaplan-Meier Survival Estimates ({meta['title']})",
            xaxis_title=f"Follow-up Duration ({time_unit})",
            yaxis_title="Cumulative Survival Probability S(t)",
            yaxis=dict(range=[0.0, 1.05], tickformat=".0%"),
            xaxis=dict(range=[0.0, max(km1['max_follow_up'], km2['max_follow_up']) * 1.05]),
            template="plotly_white",
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            margin=dict(l=40, r=40, t=50, b=40),
            height=460
        )

        st.plotly_chart(fig, use_container_width=True)

        # Number-at-Risk Table underneath the chart
        st.markdown(f"##### 👥 Number of Patients at Risk Across Follow-up ({time_unit})")
        landmarks = meta.get("landmark_times", [6.0, 12.0, 24.0, 36.0])

        risk_rows = []
        for grp_label, km_obj in [(g1_name, km1), (g2_name, km2)]:
            row = {"Study Group": grp_label}
            for lm in [0.0] + landmarks:
                # Find number at risk at landmark time
                matched_risk = km_obj["total_participants"]
                for idx, t_val in enumerate(km_obj["table_times"]):
                    if t_val <= lm:
                        matched_risk = km_obj["life_table_df"]["Number_At_Risk"].iloc[idx]
                    else:
                        break
                row[f"t = {lm:.0f}"] = matched_risk
            risk_rows.append(row)

        df_risk_table = pd.DataFrame(risk_rows)
        st.dataframe(df_risk_table, use_container_width=True, hide_index=True)

    with col_info:
        with st.container(border=True):
            st.markdown("#### 🧭 Interpreting the Kaplan-Meier Trajectories")
            p_val = lr["p_value"]
            hr = lr["peto_hazard_ratio"]

            if lr["is_significant"]:
                st.markdown(f"""
                * **Statistically Significant Curve Divergence ($p = {p_val:.4f}$):**
                  The survival curves demonstrate clear, statistically confirmed separation across follow-up. 
                  The null hypothesis ($H_0: S_1(t) = S_2(t)$) is rejected at $\\alpha = 0.05$.
                * **Direction of Benefit:**
                  Patients assigned to **{g1_name}** achieved superior survival compared to **{g2_name}**.
                * **Hazard Ratio ($\text{{HR}} = {hr:.2f}$):**
                  At any given point in time, patients in {g1_name} experienced approximately **{(1.0 - hr)*100:.1f}% lower hazard** of experiencing the event ({meta['event_name']}).
                """)
            else:
                st.markdown(f"""
                * **No Significant Difference ($p = {p_val:.4f}$):**
                  The survival curves track closely together without statistically significant divergence.
                  Any observed differences may be attributable to random variation ($p \\ge 0.05$).
                """)

            st.markdown("---")
            st.markdown(f"#### ⏱️ Landmark Survival Probability Comparison")
            st.markdown(f"Comparing cumulative survival rates at key clinical milestones:")
            lm_df = km_analysis["landmarks_df"]
            st.dataframe(lm_df[[
                "Landmark_Time",
                f"Survival_{g1_name}",
                f"Survival_{g2_name}",
                "Absolute_Difference",
                "NNT_At_Landmark"
            ]], use_container_width=True, hide_index=True)


def render_kaplan_meier_landmark_simulator(km_analysis: Dict[str, Any], meta: Dict[str, Any]):
    """Renders an interactive follow-up slider allowing clinicians to query S(t) at any time."""
    with st.container(border=True):
        st.markdown("### 🎛️ Interactive Survival Milestone & Landmark Probability Calculator")
        st.caption("Slide to any specific follow-up time point to evaluate estimated survival probabilities and Number Needed to Treat (NNT).")

        km1 = km_analysis["km_group_1"]
        km2 = km_analysis["km_group_2"]
        max_t = float(max(km1["max_follow_up"], km2["max_follow_up"]))
        time_unit = meta.get("time_unit", "Months")
        g1_name = km_analysis["group_1_name"]
        g2_name = km_analysis["group_2_name"]

        target_time = st.slider(
            f"Select Target Follow-Up Time ({time_unit}):",
            min_value=0.0,
            max_value=max_t,
            value=float(round(max_t * 0.50, 1)),
            step=0.5 if max_t <= 20 else 1.0
        )

        s1_res = calculate_landmark_survival_probability(km1, target_time)
        s2_res = calculate_landmark_survival_probability(km2, target_time)

        p1 = s1_res["survival_probability"]
        p2 = s2_res["survival_probability"]
        diff = p1 - p2
        nnt = 1.0 / abs(diff) if abs(diff) > 0.001 else np.nan

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                label=f"{g1_name} Survival at t={target_time:.1f}",
                value=f"{p1 * 100.0:.1f}%",
                help=f"Estimated probability of surviving past {target_time:.1f} {time_unit}."
            )
            st.caption(f"95% CI: [{s1_res['ci_lower']*100:.1f}%, {s1_res['ci_upper']*100:.1f}%]")

        with c2:
            st.metric(
                label=f"{g2_name} Survival at t={target_time:.1f}",
                value=f"{p2 * 100.0:.1f}%",
                help=f"Estimated probability of surviving past {target_time:.1f} {time_unit}."
            )
            st.caption(f"95% CI: [{s2_res['ci_lower']*100:.1f}%, {s2_res['ci_upper']*100:.1f}%]")

        with c3:
            st.metric(
                label=f"Absolute Survival Difference",
                value=f"{diff * 100.0:+.1f}%",
                delta=f"{diff * 100.0:+.1f}% Points",
                delta_color="normal" if diff > 0 else "inverse"
            )
            st.caption(f"Survival benefit with {g1_name}")

        with c4:
            nnt_str = f"{nnt:.1f} Patients" if not np.isnan(nnt) else "Undefined"
            st.metric(
                label=f"Number Needed to Treat (NNT at t={target_time:.1f})",
                value=nnt_str,
                help=f"Number of patients who must receive {g1_name} to prevent 1 event by time {target_time:.1f} {time_unit}."
            )
            st.caption("Lower NNT indicates higher clinical treatment potency.")


def render_kaplan_meier_rationale_tab(method_info: Dict[str, Any], km_analysis: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the comprehensive Rationale & Epidemiological Framework tab for Kaplan-Meier & Log-Rank."""
    st.subheader(f"Why Kaplan-Meier Survival Analysis & Log-Rank Test? ({method_info['name']})")
    st.markdown("""
    In clinical medicine, surgical trials, and public health follow-up cohorts, researchers rarely evaluate fixed, closed groups where 
    every individual is followed for the exact same number of days. Participants enter trials at staggered calendar dates, some move away 
    or are lost to follow-up, and many remain alive and disease-free when the study reaches its planned termination.
    
    If you analyze this longitudinal cohort using simple percentages or 2×2 tables (Relative Risk or Odds Ratio), you commit fatal methodological errors:
    1. **Survivor Selection Bias:** If you only analyze participants who finished the study, you exclude censored individuals who contributed months or years of healthy observation.
    2. **Immortal Time Distortion:** If you ignore the exact calendar date of each event, you fail to distinguish between a treatment that prevents relapse for 5 years versus one that delays it for only 2 months.
    
    **Kaplan-Meier Survival Analysis (Kaplan & Meier, 1958)** is the international gold standard for time-to-event research. 
    It computes cumulative survival probability incrementally at each distinct event time using the **Product-Limit Estimator**, 
    while the **Mantel-Cox Log-Rank Test** provides the non-parametric hypothesis test to compare survival curves across groups.
    """)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown(f"#### ✅ When to use {method_info['name']}")
        for item in method_info.get("when_to_use", []):
            st.markdown(f"- {item}")
    with col_w2:
        st.markdown("#### ⚠️ Key Assumptions to Verify")
        for item in method_info.get("key_assumptions", []):
            st.markdown(f"- {item}")
        st.info(f"**Alternative if assumptions fail:** {method_info.get('alternative_test', 'Cox Proportional Hazards Regression or Peto-Wilcoxon Test')}")

    st.markdown("---")
    st.markdown("### 🔍 The Longitudinal Survival Analysis Framework: Comparing Time-to-Event Methods")
    st.markdown(r"""
    | Survival Method | Primary Purpose | Key Advantage | Major Limitation | Type of Censoring Handled |
    | :--- | :--- | :--- | :--- | :--- |
    | **Kaplan-Meier Estimator** | Non-parametric visual estimation of $S(t)$ over continuous time. | Makes **zero parametric assumptions** about the underlying shape of hazard over time. | Univariable only; cannot adjust for multiple confounding variables simultaneously. | Right-censoring (independent/non-informative). |
    | **Mantel-Cox Log-Rank Test** | Hypothesis testing comparing survival distributions between two or more groups ($H_0: S_1(t) = S_2(t)$). | Optimal statistical power when hazards are **proportional** across time. | Loses power if survival curves **cross** (hazard ratio reverses over time). | Right-censoring. |
    | **Wilcoxon / Gehan-Breslow Test** | Hypothesis test weighting event times by the number at risk ($w_j = n_j$). | Emphasizes **early treatment differences**; more robust to late curve convergence. | Sensitive to heavy early censoring patterns. | Right-censoring. |
    | **Cox Proportional Hazards Model** | Multivariable semi-parametric regression estimating **Hazard Ratios (HR)**. | Adjusts for patient age, comorbidities, and baseline stage simultaneously. | Requires the **Proportional Hazards Assumption** ($\text{HR}(t) = \text{constant}$). | Right, left, and interval censoring. |
    | **Restricted Mean Survival (RMST)** | Area under the survival curve up to a fixed clinical time horizon ($\tau$). | Clinically intuitive (measures average event-free months gained); valid even when curves cross! | Requires pre-specifying the clinical truncation time horizon $\tau$. | Right-censoring. |
    """)

    st.markdown("#### 📖 Mathematical Formulae Explained in Plain English (Zero Prior Knowledge Assumed)")
    st.markdown(r"""
    Survival analysis mathematics can appear complex, but every equation is built upon simple, intuitive principles of conditional probability:

    1. **The Kaplan-Meier Product-Limit Survival Formula:**
       $$\hat{S}(t) = \prod_{i: t_i \le t} \left(1 - \frac{d_i}{n_i}\right) = \prod_{i: t_i \le t} \left(\frac{n_i - d_i}{n_i}\right)$$
       * **What each symbol represents:**
         * $t_i$: A distinct, sorted time point where at least one event occurred ($t_1 < t_2 < \dots < t_k$).
         * $d_i$: The number of patients who experienced the event (e.g. death or relapse) at time $t_i$.
         * $n_i$: The number of patients **at risk** immediately before time $t_i$ (those still alive, under observation, and not yet censored).
         * $(1 - d_i / n_i)$: The **conditional probability** of surviving past time $t_i$, given that the patient survived up to that moment.
       * **In Plain English:** To survive past year 3, you must first survive year 1, THEN survive year 2, THEN survive year 3. The formula simply multiplies these conditional step-probabilities together!

    2. **Greenwood's Formula for Standard Error of Survival Probability:**
       $$\widehat{\text{Var}}(\hat{S}(t)) = [\hat{S}(t)]^2 \sum_{i: t_i \le t} \frac{d_i}{n_i (n_i - d_i)}, \quad \text{SE}(\hat{S}(t)) = \sqrt{\widehat{\text{Var}}(\hat{S}(t))}$$
       * **In Plain English:** Major Greenwood (1926) developed this variance formula. Notice that as time progresses and fewer patients remain at risk ($n_i$ shrinks), each additional event contributes a much larger fraction to the summation. This causes the shaded confidence bands to widen naturally toward the tail of the curve.

    3. **Log-Log Transformed 95% Confidence Intervals (Kalbfleisch & Prentice / Borgan & Liestøl):**
       $$W(t) = \ln(-\ln \hat{S}(t)), \quad \text{SE}(W(t)) = \frac{1}{|\ln \hat{S}(t)|} \sqrt{\sum_{i: t_i \le t} \frac{d_i}{n_i (n_i - d_i)}}$$
       $$95\% \text{ Confidence Interval} = [\hat{S}(t)]^{\exp\left(\pm 1.96 \cdot \text{SE}(W(t))\right)}$$
       * **Why Standard Wald Intervals Fail:** The standard linear interval ($\hat{S}(t) \pm 1.96 \cdot \text{SE}$) often calculates upper bounds greater than $100\%$ ($1.05$) or lower bounds below $0\%$, which is mathematically impossible.
       * **The Log-Log Solution:** By transforming survival to the double-log scale, constructing the interval, and transforming back, the resulting confidence boundaries are guaranteed to stay strictly within $[0, 1]$.

    4. **The Mantel-Cox Log-Rank Test Statistic ($\chi^2$):**
       At each distinct event time $t_j$ across the combined cohort:
       * **Expected Events in Group 1 ($E_{1j}$):**
         $$E_{1j} = n_{1j} \times \left(\frac{d_j}{n_j}\right)$$
         *(Where $n_{1j}$ is the number at risk in Group 1, $d_j$ is total events at $t_j$, and $n_j$ is total combined patients at risk).*
       * **Hypergeometric Variance ($V_{1j}$):**
         $$V_{1j} = \frac{n_{1j} n_{2j} d_j (n_j - d_j)}{n_j^2 (n_j - 1)}$$
       * **Overall Chi-Square Statistic:**
         $$\chi^2 = \frac{(O_1 - E_1)^2}{V_1} \sim \chi^2(\text{df} = 1), \quad \text{p-value} = 1 - F_{\chi^2_1}(\chi^2)$$
       * **In Plain English:** The log-rank test evaluates whether Group 1 observed significantly fewer (or more) events than would be expected by pure chance under the null hypothesis of equal survival.

    5. **Peto Hazard Ratio (HR) Approximation:**
       $$\text{HR}_{\text{Peto}} = \exp\left(\frac{O_1 - E_1}{V_1}\right), \quad 95\% \text{ CI} = \exp\left[\ln(\text{HR}_{\text{Peto}}) \pm \frac{1.96}{\sqrt{V_1}}\right]$$
       * **In Plain English:** Richard Peto's log-rank hazard ratio approximation. If $\text{HR} = 0.58$, patients in Group 1 experience a $42\%$ lower instantaneous rate of death or relapse at any given moment compared to Group 2.
    """)


def render_kaplan_meier_excel_guide(km_analysis: Dict[str, Any], meta: Dict[str, Any]):
    """Renders step-by-step Microsoft Excel formulas for Kaplan-Meier and Log-Rank."""
    st.subheader("📗 Microsoft Excel Implementation Guide: Kaplan-Meier & Log-Rank")
    st.markdown("""
    You can construct a complete Kaplan-Meier life table and run the Mantel-Cox Log-Rank test in standard Microsoft Excel 
    without installing any third-party add-ins. Follow this step-by-step formula layout:
    """)

    with st.expander("📌 Step 1: Excel Sheet Architecture & Life Table Columns", expanded=False):
        st.markdown(r"""
        Organize your sorted event times in columns `A` through `H`:

        | Column | Column Header | Excel Formula (Row 3 Example) | Plain English Purpose |
        | :--- | :--- | :--- | :--- |
        | **A** | `Time_t` | `4.0` | Distinct ordered follow-up time point. |
        | **B** | `At_Risk_n` | `=COUNTIF(Raw_Times, ">=" & A3)` | Number of patients with follow-up $\ge t$. |
        | **C** | `Events_d` | `=COUNTIFS(Raw_Times, A3, Raw_Events, 1)` | Confirmed events occurring exactly at time $t$. |
        | **D** | `Censored_c`| `=COUNTIFS(Raw_Times, A3, Raw_Events, 0)` | Censored dropouts occurring at time $t$. |
        | **E** | `Conditional_Surv` | `=1 - (C3 / B3)` | Probability of surviving step $t$ given survival to $t$. |
        | **F** | `Cumulative_S_t` | `=F2 * E3` *(where F2 = 1.0)* | Cumulative Kaplan-Meier product-limit survival probability. |
        | **G** | `Greenwood_Term` | `=C3 / (B3 * (B3 - C3))` | Individual summand for Greenwood's variance formula. |
        | **H** | `Greenwood_SE` | `=F3 * SQRT(SUM($G$3:G3))` | Greenwood standard error for 95% confidence intervals. |
        """)

    with st.expander("📌 Step 2: Log-Rank Test Chi-Square Formula & Peto Hazard Ratio in Excel", expanded=False):
        st.markdown(r"""
        To compare two groups in Excel:
        1. **Expected Events in Group 1 (Cell `E_1j`):**
           `=B_1j * (D_total_j / N_total_j)`
        2. **Variance in Group 1 (Cell `V_1j`):**
           `=(B_1j * B_2j * D_total_j * (N_total_j - D_total_j)) / (N_total_j^2 * (N_total_j - 1))`
        3. **Log-Rank Chi-Square Statistic:**
           `=(SUM(Observed_G1) - SUM(Expected_G1))^2 / SUM(Variance_G1)`
        4. **Two-Tailed p-value:**
           `=CHISQ.DIST.RT(ChiSquare_Cell, 1)`
        5. **Peto Hazard Ratio:**
           `=EXP((SUM(Observed_G1) - SUM(Expected_G1)) / SUM(Variance_G1))`
        """)


def render_kaplan_meier_python_guide(km_analysis: Dict[str, Any], meta: Dict[str, Any]):
    """Renders production-ready Python code using lifelines and scipy."""
    st.subheader("🐍 Python Production Script Guide: Kaplan-Meier & Log-Rank")
    st.markdown("""
    Below is a modular, step-by-step Python workflow demonstrating how to calculate Kaplan-Meier curves, 
    Greenwood confidence intervals, and the Log-Rank test using both the gold-standard `lifelines` library and pure `scipy`:
    """)

    g1_name = km_analysis["group_1_name"]
    g2_name = km_analysis["group_2_name"]
    time_unit = meta.get("time_unit", "Months")

    with st.expander("📌 Step 1: Load Cohort Microdata & Fit Kaplan-Meier Curves in Python", expanded=False):
        st.code(f"""import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

# 1. Load Cohort Microdata
df = pd.read_csv("kaplan_meier_microdata_{meta.get('scenario_key', 'demo')}.csv")
time_col = "{meta.get('time_col', 'Time')}"
event_col = "{meta.get('event_col', 'Event')}"
group_col = "{meta.get('group_col', 'Group')}"

df_g1 = df[df[group_col] == "{g1_name}"]
df_g2 = df[df[group_col] == "{g2_name}"]

# 2. Fit Product-Limit Curves
kmf_g1 = KaplanMeierFitter()
kmf_g1.fit(durations=df_g1[time_col], event_observed=df_g1[event_col], label="{g1_name}")

kmf_g2 = KaplanMeierFitter()
kmf_g2.fit(durations=df_g2[time_col], event_observed=df_g2[event_col], label="{g2_name}")
""", language="python")

    with st.expander("📌 Step 2: Extract Milestone Survival Probabilities & Median Survival", expanded=False):
        st.code(f"""# Extract key clinical milestone survival metrics
print(f"=== {g1_name} ===")
print(f"Median Survival: {{kmf_g1.median_survival_time_:.1f}} {time_unit}")
print(f"12-Month Survival: {{kmf_g1.predict(12.0):.1%}}")

print(f"\\n=== {g2_name} ===")
print(f"Median Survival: {{kmf_g2.median_survival_time_:.1f}} {time_unit}")
print(f"12-Month Survival: {{kmf_g2.predict(12.0):.1%}}")
""", language="python")

    with st.expander("📌 Step 3: Perform Two-Group Mantel-Cox Log-Rank Test", expanded=False):
        st.code(f"""from lifelines.statistics import logrank_test

# Hypothesis test: H0: S1(t) == S2(t)
results = logrank_test(
    durations_A=df_g1[time_col],
    durations_B=df_g2[time_col],
    event_observed_A=df_g1[event_col],
    event_observed_B=df_g2[event_col]
)

print("=== Mantel-Cox Log-Rank Test Results ===")
print(f"Chi-Square Statistic: {{results.test_statistic:.3f}} (df=1)")
print(f"p-value:              {{results.p_value:.4e}}")
print(f"Significant at 0.05:  {{results.p_value < 0.05}}")
""", language="python")

    with st.expander("📌 Step 4: Plot Professional Survival Curves with at-Risk Table", expanded=False):
        st.code(f"""import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
kmf_g1.plot_survival_function(ax=ax, ci_show=True)
kmf_g2.plot_survival_function(ax=ax, ci_show=True)
plt.axhline(y=0.50, color="gray", linestyle=":", label="50% Median Line")
plt.title("{meta['title']}")
plt.xlabel("Follow-up Duration ({time_unit})")
plt.ylabel("Cumulative Survival Probability S(t)")
plt.legend(loc="lower left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
""", language="python")

    with st.expander("📌 Step 5: Consolidated Production Kaplan-Meier Pipeline Script", expanded=False):
        st.code(f"""import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

df = pd.read_csv("kaplan_meier_microdata_{meta.get('scenario_key', 'demo')}.csv")
time_col = "{meta.get('time_col', 'Time')}"
event_col = "{meta.get('event_col', 'Event')}"
group_col = "{meta.get('group_col', 'Group')}"

df_g1 = df[df[group_col] == "{g1_name}"]
df_g2 = df[df[group_col] == "{g2_name}"]

kmf1 = KaplanMeierFitter().fit(df_g1[time_col], df_g1[event_col], label="{g1_name}")
kmf2 = KaplanMeierFitter().fit(df_g2[time_col], df_g2[event_col], label="{g2_name}")

lr = logrank_test(df_g1[time_col], df_g2[time_col], df_g1[event_col], df_g2[event_col])
print(f"Log-Rank Chi2 = {{lr.test_statistic:.2f}}, p = {{lr.p_value:.4f}}")

fig, ax = plt.subplots(figsize=(8, 5))
kmf1.plot(ax=ax)
kmf2.plot(ax=ax)
plt.show()
""", language="python")


def render_kaplan_meier_download_tab(df_microdata: pd.DataFrame, df_summary: pd.DataFrame, meta: Dict[str, Any]):
    """Renders sample dataset download controls for survival analysis."""
    st.subheader("💾 Download Survival Analysis Datasets")
    st.markdown("Download the synthetic trial cohort microdata and group summary tables for practice in Excel, R, or Python:")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Cohort Summary Table (Sample Size, Events, Censored %)**")
        csv_sum = df_summary.to_csv(index=False)
        st.download_button(
            label="📥 Download Group Summary CSV",
            data=csv_sum,
            file_name=f"kaplan_meier_summary_{meta.get('scenario_key', 'custom')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    with c2:
        st.markdown(f"**2. Patient Microdata ({len(df_microdata)} Individual Records)**")
        csv_micro = df_microdata.to_csv(index=False)
        st.download_button(
            label=f"📥 Download Patient Microdata ({len(df_microdata)} Rows CSV)",
            data=csv_micro,
            file_name=f"kaplan_meier_microdata_{meta.get('scenario_key', 'custom')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df_microdata.head(6), use_container_width=True, hide_index=True)


def render_kaplan_meier_module(active_method_key: str, method_info: Dict[str, Any]):
    """Top-level controller for Kaplan-Meier Survival Analysis & Log-Rank Test."""
    st.markdown("## ⏱️ Kaplan-Meier Survival Curves & Log-Rank Test")
    st.caption("Evaluates longitudinal time-to-event outcomes, right-censoring, product-limit survival trajectories, and two-group log-rank comparisons.")

    # 1. Plain English Pedagogical Guide
    render_kaplan_meier_plain_english_guide()

    # 2. Scenario Selection Sidebar / Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🩺 Longitudinal Survival Scenario")

    scenario_options = {
        "oncology_immunotherapy": "🎗️ All Meadows Cancer Centre: Immunotherapy vs Chemotherapy (Metastatic NSCLC)",
        "cardiac_surgery_graft": "🫀 Inner City Hospital: Radial Artery vs Saphenous Vein CABG Patency",
        "pediatric_leukemia_induction": "🩸 Riverside Children's Hospital: Intensified Induction in Pediatric ALL",
        "custom": "✏️ Custom Time-to-Event Data Input (Enter Patient Records)"
    }

    selected_scenario = st.sidebar.selectbox(
        "Choose Clinical Scenario:",
        options=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
        index=0
    )

    if selected_scenario != "custom":
        df_microdata, df_summary, meta = generate_kaplan_meier_demo_data(selected_scenario)
        time_col = meta["time_col"]
        event_col = meta["event_col"]
        group_col = meta["group_col"]
    else:
        st.sidebar.markdown("**Input Custom Time-to-Event Parameters:**")
        n_g1 = st.sidebar.slider("Sample Size: Treatment Arm 1", min_value=20, max_value=500, value=100, step=10)
        n_g2 = st.sidebar.slider("Sample Size: Control Arm 2", min_value=20, max_value=500, value=100, step=10)
        h1 = st.sidebar.slider("Monthly Hazard Rate: Arm 1", min_value=0.01, max_value=0.20, value=0.04, step=0.01)
        h2 = st.sidebar.slider("Monthly Hazard Rate: Arm 2", min_value=0.01, max_value=0.20, value=0.08, step=0.01)
        max_t = st.sidebar.slider("Maximum Study Duration (Months)", min_value=12, max_value=60, value=36, step=6)

        np.random.seed(42)
        u1 = np.random.uniform(0.0001, 0.9999, size=n_g1)
        t1 = -np.log(u1) / h1
        censor_u1 = np.random.uniform(0.0001, 0.9999, size=n_g1)
        loss1 = -np.log(censor_u1) / 0.015

        obs1 = [round(float(min(t1[i], loss1[i], max_t)), 1) for i in range(n_g1)]
        ev1 = [1 if (t1[i] <= loss1[i] and t1[i] <= max_t) else 0 for i in range(n_g1)]

        u2 = np.random.uniform(0.0001, 0.9999, size=n_g2)
        t2 = -np.log(u2) / h2
        censor_u2 = np.random.uniform(0.0001, 0.9999, size=n_g2)
        loss2 = -np.log(censor_u2) / 0.015

        obs2 = [round(float(min(t2[i], loss2[i], max_t)), 1) for i in range(n_g2)]
        ev2 = [1 if (t2[i] <= loss2[i] and t2[i] <= max_t) else 0 for i in range(n_g2)]

        time_col = "Time_Months"
        event_col = "Event"
        group_col = "Group"

        df_microdata = pd.DataFrame({
            "Patient_ID": [f"PT-G1-{i+1:03d}" for i in range(n_g1)] + [f"PT-G2-{i+1:03d}" for i in range(n_g2)],
            "Group": ["Investigational Arm"] * n_g1 + ["Standard Control Arm"] * n_g2,
            time_col: obs1 + obs2,
            event_col: ev1 + ev2,
            "Status": ["Event Occurred" if e == 1 else "Censored" for e in (ev1 + ev2)],
            "Censoring_Detail": ["Event Occurred" if e == 1 else "Censored / End of Follow-up" for e in (ev1 + ev2)]
        })

        df_summary = pd.DataFrame({
            "Treatment_Group": ["Investigational Arm", "Standard Control Arm", "Combined Total"],
            "Sample_Size_N": [n_g1, n_g2, n_g1 + n_g2],
            "Events_Count": [sum(ev1), sum(ev2), sum(ev1) + sum(ev2)],
            "Censored_Count": [n_g1 - sum(ev1), n_g2 - sum(ev2), n_g1 + n_g2 - (sum(ev1) + sum(ev2))],
            "Event_Rate_Pct": [f"{(sum(ev1)/n_g1)*100:.1f}%", f"{(sum(ev2)/n_g2)*100:.1f}%", f"{((sum(ev1)+sum(ev2))/(n_g1+n_g2))*100:.1f}%"]
        })

        meta = {
            "scenario_key": "custom",
            "title": "Custom User-Defined Survival Trial",
            "domain": "User-Defined Longitudinal Cohort",
            "description": f"Custom time-to-event trial simulating {n_g1} patients in Investigational Arm vs {n_g2} in Standard Control Arm over {max_t} months.",
            "time_col": time_col,
            "time_unit": "Months",
            "event_col": event_col,
            "group_col": group_col,
            "group_names": ("Investigational Arm", "Standard Control Arm"),
            "event_name": "Primary Clinical Endpoint",
            "landmark_times": [6.0, 12.0, 24.0, max_t],
            "landmark_labels": ["6-Mo Survival", "12-Mo Survival", "24-Mo Survival", f"{max_t}-Mo Survival"],
            "expected_hr": h1 / h2 if h2 > 0 else 1.0,
            "clinical_takeaway": "Custom simulated survival parameters calculated dynamically from user controls.",
            "disclaimer": "User-entered customized time-to-event data."
        }

    # 3. Calculate Comparative Survival Analysis
    km_analysis = calculate_kaplan_meier_two_groups(
        df=df_microdata,
        time_col=time_col,
        event_col=event_col,
        group_col=group_col,
        landmark_times=meta.get("landmark_times"),
        alpha=0.05
    )

    # 4. Clinical Investigation Header Banner
    st.markdown(f"### 📋 Active Investigation: {meta['title']}")
    st.markdown(f"**Clinical Domain:** *{meta['domain']}*")
    st.markdown(f"{meta['description']}")

    # 5. Clinical Takeaway Callout Card
    with st.container(border=True):
        st.markdown("#### 💡 Clinical & Epidemiological Takeaway")
        st.markdown(f"{meta['clinical_takeaway']}")

    # 6. Methodological Prerequisite Evaluation (STROBE / CONSORT)
    evaluations = evaluate_kaplan_meier_prerequisites(
        km_analysis, df_microdata, time_col=time_col, event_col=event_col, group_col=group_col
    )
    render_kaplan_meier_diagnostics(evaluations)

    st.markdown("---")

    # 7. Primary Metrics Dashboard
    render_kaplan_meier_dashboard(km_analysis, meta)

    st.markdown("---")

    # 8. Side-by-Side Visualizations & Balanced Explanatory Containers
    render_kaplan_meier_visualizations(km_analysis, meta)

    st.markdown("---")

    # 9. Interactive Landmark Survival Milestone Simulator
    render_kaplan_meier_landmark_simulator(km_analysis, meta)

    st.markdown("---")

    # 10. Methodology, Implementation Guides & Downloads Tabs
    tab_rationale, tab_excel, tab_python, tab_data = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Microsoft Excel Implementation Guide",
        "🐍 Python Production Script Guide",
        "💾 Live Data Downloads"
    ])

    with tab_rationale:
        render_kaplan_meier_rationale_tab(method_info, km_analysis, meta)

    with tab_excel:
        render_kaplan_meier_excel_guide(km_analysis, meta)

    with tab_python:
        render_kaplan_meier_python_guide(km_analysis, meta)

    with tab_data:
        render_kaplan_meier_download_tab(df_microdata, df_summary, meta)


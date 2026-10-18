"""Interactive UI Components & Visualizations for Cox Proportional Hazards Regression.

Provides:
1. Plain English Pedagogical Introductory Guide to Multivariable Survival Analysis.
2. Sidebar scenario controls with three realistic clinical scenarios and custom covariate exploration.
3. Methodological Prerequisite Evaluation (Events Per Variable [EPV], Schoenfeld proportionality,
   independent censoring, collinearity, Harrell's C-index).
4. Primary Metric Dashboard (Adjusted Hazard Ratio, LRT Chi2, C-index, EPV).
5. Plotly Forest Plot of Adjusted Hazard Ratios with 95% confidence intervals and neutral reference line (HR = 1.0).
6. Interactive Individual Patient Survival Predictor (calculating personalized predicted survival curves
   and landmark probabilities based on user-adjusted clinical profile).
7. Full 4-Tab Educational Architecture:
   - 📋 Rationale & Epidemiological Framework (Zero prior knowledge, LaTeX + Plain English).
   - 📗 Microsoft Excel Implementation Guide (Formulas, Solver partial likelihood, forest plot setup).
   - 🐍 Python Production Script Guide (Production script with lifelines CoxPHFitter and diagnostics).
   - 💾 Live Data Downloads (Patient-level microdata and regression summary CSVs).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.cox_regression import (
    fit_cox_proportional_hazards,
    calculate_harrell_c_index,
    calculate_schoenfeld_diagnostics,
    predict_cox_survival_curve,
    evaluate_cox_regression_prerequisites,
    generate_cox_regression_demo_data
)


def render_cox_plain_english_guide():
    """Renders a collapsible plain English introductory guide to multivariable survival modeling."""
    with st.expander("📖 Zero-Knowledge Guide: What is Cox Proportional Hazards Regression?", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. Why Kaplan-Meier Isn't Enough: The Confounding Dilemma
            * In clinical medicine, a new treatment might look beneficial in a simple univariable test (like Kaplan-Meier or Log-Rank).
            * But what if younger, healthier patients were disproportionately given the new drug, while older, frailer patients received the standard therapy?
            * A simple Kaplan-Meier curve cannot separate the true effect of the drug from the confounding effects of age, disease severity, or comorbidities.
            * **Sir David Cox (1972)** solved this monumental challenge by creating **Cox Proportional Hazards Regression**, allowing researchers to measure the independent effect of a treatment while simultaneously adjusting for multiple confounding variables.
            """)

            st.markdown(r"""
            #### 2. What is a "Hazard" and a "Hazard Ratio (HR)"?
            * **Hazard ($h(t)$):** The instantaneous event rate at time $t$ for someone who has survived up to that moment. Think of it as the immediate "velocity" of experiencing an event (e.g. relapse or death).
            * **Hazard Ratio ($\text{HR}$):** The relative difference in event speed between two groups:
              * $\text{HR} = 1.0$: Neutral effect (equal event rate in both groups).
              * $\text{HR} = 0.60$: Protective effect ($40\%$ lower rate of experiencing the event at any given moment).
              * $\text{HR} = 1.75$: Elevated risk ($75\%$ higher rate of experiencing the event at any moment).
            """)

        with col_g2:
            st.markdown(r"""
            #### 3. Why is it Called "Semi-Parametric"?
            * Traditional regression models (like Weibull or Exponential survival models) force survival curves to follow strict mathematical curves (bell curves, exponential decay, etc.).
            * Cox regression makes **no assumptions** about the shape of the baseline hazard curve over time ($h_0(t)$ is completely non-parametric).
            * However, it assumes that the relative effect of covariates ($\exp(X\beta)$) acts as a constant multiplier across time (the parametric proportional hazards assumption).
            * This gives researchers the best of both worlds: extreme flexibility over time with clean, interpretable relative risk multipliers!
            """)

            st.markdown(r"""
            #### 4. The Core Assumption: "Proportional Hazards"
            * The model assumes that the Hazard Ratio between two patients remains constant over time.
            * For example, if a drug halves the risk of death in Month 1 ($\text{HR} = 0.50$), it must also approximately halve the risk in Month 12 and Month 24.
            * If the curves cross (e.g., an invasive surgery causes high early mortality but provides long-term cure), the proportional hazards assumption is violated, requiring specialized diagnostic evaluation (Schoenfeld residuals).
            """)


def render_cox_diagnostics(evaluations: Dict[str, Any]):
    """Renders diagnostic evaluation cards for Cox regression assumptions."""
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


def render_cox_dashboard(model: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the primary metrics dashboard for the fitted Cox model."""
    st.markdown("### 📊 Multivariable Survival Model Summary Dashboard")

    df_coef = model["df_coefficients"]
    primary_var = meta.get("primary_exposure", model["covariate_names"][0])
    match_row = df_coef[df_coef["Covariate"] == primary_var]

    if not match_row.empty:
        primary_hr = float(match_row["Hazard_Ratio"].iloc[0])
        primary_ci_str = match_row["HR_95_CI_Str"].iloc[0]
        primary_pct = float(match_row["Pct_Change_Hazard"].iloc[0])
        primary_p = float(match_row["p_value"].iloc[0])
        primary_sig = match_row["Significance"].iloc[0]
    else:
        primary_hr = float(df_coef["Hazard_Ratio"].iloc[0])
        primary_ci_str = df_coef["HR_95_CI_Str"].iloc[0]
        primary_pct = float(df_coef["Pct_Change_Hazard"].iloc[0])
        primary_p = float(df_coef["p_value"].iloc[0])
        primary_sig = df_coef["Significance"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container(border=True):
            delta_str = f"{primary_pct:+.1f}% Hazard"
            st.metric(
                label=f"Adjusted HR: {primary_var}",
                value=f"{primary_hr:.2f}",
                delta=delta_str,
                delta_color="inverse" if primary_hr < 1.0 else "normal",
                help=f"Adjusted Hazard Ratio for {primary_var} controlling for all other model covariates."
            )
            st.caption(f"95% CI: {primary_ci_str} (p = {primary_p:.4f})")

    with c2:
        with st.container(border=True):
            lrt_stat = model["lrt_statistic"]
            lrt_p = model["lrt_p_value"]
            lrt_p_str = f"{lrt_p:.4f}" if lrt_p >= 0.0001 else "< 0.0001"
            st.metric(
                label="Overall Model Test (LRT)",
                value=f"Chi2 = {lrt_stat:.2f}",
                delta=f"p = {lrt_p_str} (df={model['degrees_of_freedom']})",
                delta_color="normal",
                help="Likelihood Ratio Test comparing the multivariable model against the null model."
            )
            st.caption("🟢 Statistically Significant" if lrt_p < 0.05 else "⚪ Null Model Not Rejected")

    with c3:
        with st.container(border=True):
            c_idx = model["c_index"]
            st.metric(
                label="Harrell's C-Index",
                value=f"{c_idx:.3f}",
                delta="Strong Discrimination" if c_idx >= 0.70 else "Moderate Discrimination",
                delta_color="normal",
                help="Concordance index: proportion of evaluable subject pairs where the patient with higher risk experienced the event earlier."
            )
            st.caption("Theoretical Range: 0.50 (Chance) to 1.00 (Perfect)")

    with c4:
        with st.container(border=True):
            total_n = model["n_samples"]
            events_n = model["total_events"]
            epv = events_n / max(1, model["degrees_of_freedom"])
            st.metric(
                label="Cohort Events & EPV",
                value=f"{events_n} / {total_n}",
                delta=f"{epv:.1f} Events/Variable",
                delta_color="normal" if epv >= 10.0 else "inverse",
                help="Confirmed event count out of total sample size and Events Per Variable (EPV)."
            )
            st.caption(f"Censored: {model['censoring_rate_pct']:.1f}% ({model['total_censored']} subjects)")


def render_cox_forest_plot(model: Dict[str, Any], meta: Dict[str, Any]):
    """Renders a Plotly Forest Plot of Adjusted Hazard Ratios with 95% confidence intervals."""
    st.markdown("### 🌲 Forest Plot of Multivariable Adjusted Hazard Ratios")

    df_coef = model["df_coefficients"].copy()
    labels_dict = meta.get("covariate_labels", {})

    # Display friendly names if available
    df_coef["Display_Label"] = df_coef["Covariate"].apply(lambda c: labels_dict.get(c, c))

    # Reverse order so first covariate appears at the top of the plot
    df_coef = df_coef.iloc[::-1].reset_index(drop=True)

    col_forest, col_table = st.columns([1.4, 1.1])

    with col_forest:
        fig = go.Figure()

        # Vertical reference dashed line at HR = 1.0 (neutral)
        fig.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="#64748b",
            line_width=1.5,
            annotation_text="Neutral Line (HR = 1.0)",
            annotation_position="top"
        )

        for i, row in df_coef.iterrows():
            hr = row["Hazard_Ratio"]
            hr_l = row["HR_95_CI_Lower"]
            hr_u = row["HR_95_CI_Upper"]
            cov_name = row["Display_Label"]
            is_sig = (row["p_value"] < 0.05)

            # Color coding
            if not is_sig:
                marker_color = "#94a3b8"  # Slate gray for non-significant
            elif hr < 1.0:
                marker_color = "#10b981"  # Emerald green for protective
            else:
                marker_color = "#ef4444"  # Rose red for hazard

            # Error bar horizontal trace
            fig.add_trace(go.Scatter(
                x=[hr],
                y=[cov_name],
                mode="markers",
                marker=dict(size=12, color=marker_color, symbol="square"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[hr_u - hr],
                    arrayminus=[hr - hr_l],
                    color=marker_color,
                    thickness=2.5,
                    width=6
                ),
                name=cov_name,
                hovertemplate=(
                    f"<b>{cov_name}</b><br>"
                    f"Adjusted Hazard Ratio: {hr:.2f}<br>"
                    f"95% CI: [{hr_l:.2f}, {hr_u:.2f}]<br>"
                    f"p-value: {row['p_value_str']}<extra></extra>"
                ),
                showlegend=False
            ))

        # Determine reasonable x-axis limits
        min_x = max(0.1, min(0.4, float(df_coef["HR_95_CI_Lower"].min()) * 0.8))
        max_x = min(15.0, max(2.5, float(df_coef["HR_95_CI_Upper"].max()) * 1.2))

        fig.update_layout(
            title="Adjusted Hazard Ratios (aHR) & 95% Confidence Intervals",
            xaxis_title="Adjusted Hazard Ratio (Logarithmic Scale)",
            xaxis_type="log",
            xaxis=dict(
                tickvals=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                ticktext=["0.25", "0.50", "0.75", "1.00", "1.50", "2.00", "3.00", "4.00", "6.00"],
                range=[np.log10(min_x), np.log10(max_x)]
            ),
            yaxis_title=None,
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=40),
            height=340
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        with st.container(border=True):
            st.markdown("#### 📋 Regression Coefficients Table")
            display_df = model["df_coefficients"][[
                "Covariate",
                "Hazard_Ratio",
                "HR_95_CI_Str",
                "p_value_str",
                "Significance"
            ]].rename(columns={
                "Hazard_Ratio": "Adjusted HR",
                "HR_95_CI_Str": "95% CI",
                "p_value_str": "p-value"
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.caption("""
            * **Adjusted HR:** Multiplier of the instantaneous event rate per unit change, adjusting for all other covariates.
            * **95% Confidence Interval (CI):** Range of plausible true population values. Excludes 1.0 if significant.
            * **p-value:** Two-tailed Wald test probability under the null hypothesis (beta = 0, HR = 1.0).
            """)

    # Full-width explanation container spanning the whole width of the screen
    with st.container(border=True):
        st.markdown("#### 📖 How to Read & Interpret this Hazard Ratio Forest Plot (Full-Width Guide)")

        c_exp1, c_exp2 = st.columns(2)

        with c_exp1:
            st.markdown(r"""
            ##### 1. The Visual Components of the Forest Plot
            * **Square Markers (Point Estimates):**
              Each marker represents the **Adjusted Hazard Ratio** ($\text{aHR} = \exp(\beta)$) for that variable. It indicates how much the instantaneous event rate changes when that predictor increases by 1 unit (or is present vs. absent), holding all other predictors constant.
            * **Horizontal Error Whiskers (95% Confidence Intervals):**
              The whiskers show the **95% Confidence Interval** $[\text{HR}_{\text{lower}}, \text{HR}_{\text{upper}}]$. There is a 95% probability that the true population hazard ratio falls within this span. Narrower whiskers reflect larger event counts and higher statistical precision.
            * **Vertical Dashed Line at $\text{HR} = 1.0$ (Line of No Effect):**
              $\text{HR} = 1.0$ is the neutral benchmark (null hypothesis, $\beta = 0$). It signifies identical event rates in exposed and unexposed groups.
            * **Logarithmic Horizontal Axis:**
              The horizontal axis is plotted on a **logarithmic scale**. This guarantees that a halving of risk ($\text{HR} = 0.50$, ratio $1/2$) and a doubling of risk ($\text{HR} = 2.00$, ratio $2/1$) are visually equidistant from the neutral line $\text{HR} = 1.0$.
            """)

        with c_exp2:
            st.markdown(r"""
            ##### 2. The Golden Rule of Statistical Significance & Direction
            * **Does the Whisker Cross $\text{HR} = 1.0$?**
              * **Whiskers CROSS $1.0$ (Touches or overlaps dashed line):** The covariate is **not statistically significant** ($p \ge 0.05$). We cannot rule out that the observed effect arose by pure random chance.
              * **Whiskers DO NOT Cross $1.0$ (Entirely to the left or right):** The covariate is **statistically significant** ($p < 0.05$). The direction of effect is statistically confirmed.
            * **Interpreting Direction Depends on the Clinical Endpoint:**
              * **Adverse Endpoints (Death, Relapse, Disease Progression):**
                * **$\text{HR} < 1.0$ (Left of line):** **Protective / Beneficial.** Associated with prolonged survival or delayed disease progression.
                * **$\text{HR} > 1.0$ (Right of line):** **Harmful / Risk Factor.** Associated with accelerated event rates or increased hazard.
              * **Desirable / Recovery Endpoints (Return to Work [RTW], Hospital Discharge):**
                * **$\text{HR} > 1.0$ (Right of line):** **Beneficial / Accelerated Resumption.** Employees or patients experience a higher daily velocity of recovery, leading to shorter absence or faster recovery.
                * **$\text{HR} < 1.0$ (Left of line):** **Delayed Resumption / Barrier.** Slower daily recovery rate, leading to prolonged sickness absence.
            """)

        st.markdown("---")
        st.markdown(f"##### 🔍 Plain English Walkthrough: Active Investigation ({meta['title']})")

        is_rtw = ("Return" in meta.get("event_name", "") or meta.get("scenario_key") == "occupational_return_to_work")

        interpret_bullets = []
        for _, row in model["df_coefficients"].iterrows():
            c_name = row["Covariate"]
            c_disp = labels_dict.get(c_name, c_name)
            hr_val = row["Hazard_Ratio"]
            hr_l = row["HR_95_CI_Lower"]
            hr_u = row["HR_95_CI_Upper"]
            p_val = row["p_value"]
            pct = row["Pct_Change_Hazard"]

            if p_val < 0.05:
                sig_note = f"statistically significant ($p = {p_val:.4f}$)"
                if is_rtw:
                    if hr_val > 1.0:
                        meaning = f"associated with **{pct:+.1f}% faster return to work** per day (accelerated resumption), with the true benefit likely between {hr_l:.2f} and {hr_u:.2f}."
                    else:
                        meaning = f"associated with **{abs(pct):.1f}% slower return to work** per day (prolonged sickness absence), with the true rate ratio between {hr_l:.2f} and {hr_u:.2f}."
                else:
                    if hr_val < 1.0:
                        meaning = f"associated with a **{abs(pct):.1f}% lower instantaneous hazard** of {meta.get('event_name', 'the event')} (protective effect), with true reduction between {(1.0-hr_u)*100:.1f}% and {(1.0-hr_l)*100:.1f}%."
                    else:
                        meaning = f"associated with a **{pct:+.1f}% higher instantaneous hazard** of {meta.get('event_name', 'the event')} (elevated risk factor), with true elevation between {(hr_l-1.0)*100:.1f}% and {(hr_u-1.0)*100:.1f}%."
            else:
                sig_note = f"not statistically significant ($p = {p_val:.4f}$, 95% CI: [{hr_l:.2f}, {hr_u:.2f}] spans 1.0)"
                meaning = "no statistically confirmed independent effect after multivariable adjustment for all other model covariates."

            interpret_bullets.append(f"* **{c_disp}:** **Adjusted HR = {hr_val:.2f}** (95% CI: [{hr_l:.2f}, {hr_u:.2f}]). This is {sig_note} — {meaning}")

        for bullet in interpret_bullets:
            st.markdown(bullet)


def render_cox_individual_predictor(model: Dict[str, Any], meta: Dict[str, Any]):
    """Renders an interactive patient risk profiling tool that computes personalized survival curves."""
    with st.container(border=True):
        st.markdown("### 🎛️ Individual Patient Survival Trajectory Predictor")
        st.caption("Adjust the patient profile below to calculate their individualized survival trajectory $S(t \\mid X^*)$ vs. baseline survival $S_0(t)$.")

        covar_names = model["covariate_names"]
        labels_dict = meta.get("covariate_labels", {})
        time_unit = meta.get("time_unit", "Months")

        cols = st.columns(len(covar_names))
        patient_inputs = {}

        for i, c_name in enumerate(covar_names):
            with cols[i]:
                label_text = labels_dict.get(c_name, c_name)
                # Determine input type based on column name or values
                if "Age" in c_name:
                    patient_inputs[c_name] = st.slider(label_text, min_value=20, max_value=85, value=48, step=1)
                elif "LVEF" in c_name:
                    patient_inputs[c_name] = st.slider(label_text, min_value=15, max_value=60, value=30, step=1)
                elif "Creatinine" in c_name:
                    patient_inputs[c_name] = st.slider(label_text, min_value=0.7, max_value=3.0, value=1.4, step=0.1)
                elif "eGFR" in c_name:
                    patient_inputs[c_name] = st.slider(label_text, min_value=15, max_value=70, value=35, step=1)
                else:
                    # Binary flag (0 or 1)
                    val_bool = st.selectbox(label_text, options=["0 (No / Standard / Reference)", "1 (Yes / Exposed / Present)"], index=0)
                    patient_inputs[c_name] = 1.0 if "1 (" in val_bool else 0.0

        # Predict survival curve for this specific patient
        patient_curve = predict_cox_survival_curve(model, patient_inputs)

        col_p1, col_p2 = st.columns([1.5, 1.0])

        with col_p1:
            fig_p = go.Figure()

            # Baseline Survival S0(t) curve
            baseline_times = np.insert(model["unique_event_times"], 0, 0.0)
            baseline_surv = np.insert(model["baseline_survival"], 0, 1.0)
            fig_p.add_trace(go.Scatter(
                x=baseline_times,
                y=baseline_surv,
                mode="lines",
                name="Baseline Cohort Survival S0(t)",
                line=dict(color="#94a3b8", width=2, dash="dash"),
                hovertemplate=f"Baseline S0(t): %{{y:.1%}}<br>Time: %{{x:.1f}} {time_unit}<extra></extra>"
            ))

            # Patient Personalized Curve
            fig_p.add_trace(go.Scatter(
                x=patient_curve["times"],
                y=patient_curve["survival_probabilities"],
                mode="lines",
                name="Patient Predicted Survival S(t|X*)",
                line=dict(color="#0284c7", width=3.5),
                hovertemplate=f"Patient Survival S(t|X*): %{{y:.1%}}<br>Time: %{{x:.1f}} {time_unit}<extra></extra>"
            ))

            # 50% Median Line
            fig_p.add_hline(
                y=0.50,
                line_dash="dot",
                line_color="#cbd5e1",
                annotation_text="50% Median Line",
                annotation_position="bottom right"
            )

            fig_p.update_layout(
                title=f"Predicted Survival Curve for Selected Patient Profile",
                xaxis_title=f"Follow-up Duration ({time_unit})",
                yaxis_title="Predicted Cumulative Survival S(t)",
                yaxis=dict(range=[0.0, 1.05], tickformat=".0%"),
                template="plotly_white",
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
                margin=dict(l=40, r=40, t=50, b=40),
                height=380
            )

            st.plotly_chart(fig_p, use_container_width=True)

        with col_p2:
            with st.container(border=True):
                st.markdown("#### ⏱️ Predicted Milestone Survival Rates")
                risk_mult = patient_curve["risk_multiplier"]
                st.markdown(f"**Patient Relative Hazard Multiplier:** $\\exp(X^* \\beta) = {risk_mult:.2f}$")

                # Extract landmark probabilities at 6, 12, 24, 36 months if available
                times_arr = patient_curve["times"]
                surv_arr = patient_curve["survival_probabilities"]

                landmarks = [6.0, 12.0, 24.0, 36.0]
                rows = []
                for lm in landmarks:
                    if lm <= np.max(times_arr):
                        # Find closest time <= lm
                        idx = np.where(times_arr <= lm)[0][-1]
                        prob = surv_arr[idx]
                        rows.append({
                            "Milestone": f"{lm:.0f} {time_unit}",
                            "Predicted Survival": f"{prob*100:.1f}%",
                            "Event Risk": f"{(1.0 - prob)*100:.1f}%"
                        })

                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                med_t = patient_curve["median_survival_time"]
                if med_t is not None:
                    st.success(f"**Predicted 50% Median Survival:** {med_t:.1f} {time_unit}")
                else:
                    st.info(f"**Predicted 50% Median Survival:** Not reached in follow-up (>50% survive at {np.max(times_arr):.0f} {time_unit})")


def render_cox_rationale_tab(method_info: Dict[str, Any], model: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the comprehensive Rationale & Epidemiological Framework tab for Cox PH regression."""
    st.subheader(f"Why Cox Proportional Hazards Regression? ({method_info['name']})")
    st.markdown("""
    In non-randomized observational cohorts, registry studies, and pragmatic clinical trials, treatment assignment is rarely 
    independent of patient prognosis. Older patients with advanced stage or extensive cardiovascular comorbidities are frequently 
    allocated to conservative therapies, while fitter candidates receive aggressive interventions.
    
    If researchers rely exclusively on univariable survival tools (such as the Kaplan-Meier estimator or the two-group Mantel-Cox Log-Rank test), 
    the resulting comparison is severely distorted by **confounding by indication**.
    
    **Cox Proportional Hazards Regression (David Cox, 1972)** revolutionized biomedical epidemiology by introducing a **semi-parametric** 
    framework that simultaneously adjusts for continuous, ordinal, and binary confounders without forcing survival times to adhere to restrictive 
    parametric probability distributions (such as Weibull, Exponential, or Gompertz).
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
        st.info(f"**Alternative if assumptions fail:** {method_info.get('alternative_test', 'Time-Stratified Cox Regression or Accelerated Failure Time [AFT] models')}")

    st.markdown("---")
    st.markdown("### 🔍 The Epidemiological Survival Spectrum: Comparing Analytical Approaches")
    st.markdown(r"""
    | Survival Method | Mathematical Model | Handling of Confounders | Baseline Hazard Form | Proportional Hazards Required? |
    | :--- | :--- | :--- | :--- | :--- |
    | **Kaplan-Meier Estimator** | Non-parametric product-limit $\hat{S}(t) = \prod (1 - d_i/n_i)$ | **None** (Univariable only; must stratify if adjusting). | Completely unspecified empirical steps. | **No** (Valid even when curves cross). |
    | **Mantel-Cox Log-Rank Test** | Non-parametric rank test $\chi^2 = (O_1 - E_1)^2 / V_1$ | **None** (Tests $H_0: S_1(t) = S_2(t)$ across two univariable arms). | Unspecified. | **Yes** (Maximum statistical power under proportional hazards). |
    | **Cox Proportional Hazards** | Semi-parametric model $h(t \mid X) = h_0(t) \exp(X\beta)$ | **Full multivariable adjustment** (continuous, binary, categorical). | **Non-parametric** (completely arbitrary $h_0(t)$). | **Yes** ($\text{HR}_k = \exp(\beta_k)$ constant over time). |
    | **Accelerated Failure Time (AFT)** | Fully parametric model $\ln(T) = \mu + X\beta + \sigma \epsilon$ | **Full multivariable adjustment**. | Fully parametric (Weibull, Lognormal, Log-logistic). | **No** (Assumes acceleration factor rather than proportional hazards). |
    | **Restricted Mean Survival (RMST)** | Area under the survival curve $\int_0^\tau S(t) dt$ | Univariable or multivariable pseudovalues. | Unspecified empirical steps. | **No** (Directly measures average event-free months gained). |
    """)

    st.markdown("#### 📖 Mathematical Formulae Explained in Plain English (Zero Prior Knowledge Assumed)")
    st.markdown(r"""
    The mathematical foundations of Cox regression are elegant and intuitive when broken down term-by-term:

    1. **The Semi-Parametric Hazard Equation:**
       $$h(t \mid X) = h_0(t) \exp(\beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p) = h_0(t) \exp(X\beta)$$
       * **What each symbol represents:**
         * $h(t \mid X)$: The **instantaneous hazard rate** at time $t$ for an individual with covariate profile $X = (X_1, X_2, \dots, X_p)$.
         * $h_0(t)$: The **baseline hazard function**. This represents the event rate over time when all covariates equal zero ($X = 0$). It can take any shape imaginable (increasing, decreasing, bathtub, or oscillating) without restricting the model.
         * $\exp(X\beta)$: The **relative risk multiplier**. This scales the baseline hazard up or down based on the patient's individual risk factors.
       * **In Plain English:** Your immediate risk of experiencing an event today equals the general baseline risk of the disease multiplied by your personal risk score!

    2. **The Adjusted Hazard Ratio (aHR):**
       $$\text{aHR}_k = \exp(\beta_k), \quad 95\% \text{ CI} = \left[\exp(\beta_k - 1.96 \cdot \text{SE}(\beta_k)), \exp(\beta_k + 1.96 \cdot \text{SE}(\beta_k))\right]$$
       * **What it means:** For a binary treatment ($1 = \text{Treated}, 0 = \text{Control}$), $\text{aHR} = \exp(\beta)$ indicates the ratio of the hazard in the treated group versus the control group, **holding all other covariates completely constant**.
       * If $\text{aHR} = 0.62$, treated patients experience a $38\%$ lower rate of the event at any given point in time compared to identical untreated patients.

    3. **Cox's Partial Likelihood (David Cox, 1972):**
       At each distinct failure time $t_i$ where subject $i$ experiences the event:
       $$L_i(\beta) = \frac{\exp(X_i \beta)}{\sum_{j \in R(t_i)} \exp(X_j \beta)}$$
       $$\ell(\beta) = \sum_{i: d_i = 1} \left[ X_i \beta - \ln \left(\sum_{j \in R(t_i)} \exp(X_j \beta)\right) \right]$$
       * **What each term represents:**
         * $R(t_i)$: The **risk set** immediately prior to time $t_i$ (all patients alive, under follow-up, and not yet censored).
         * $\exp(X_i \beta)$: The personal risk score of the patient who actually had the event.
         * $\sum_{j \in R(t_i)} \exp(X_j \beta)$: The combined sum of risk scores across every patient at risk at that moment.
       * **In Plain English:** At the exact second an event happens, Cox asks: *Given that someone in the risk pool was going to experience an event right now, what is the conditional probability that it was patient $i$ rather than anyone else?* Notice that the unknown baseline hazard $h_0(t)$ cancels out completely from the numerator and denominator!

    4. **Breslow's Baseline Cumulative Hazard ($H_0(t)$) & Baseline Survival ($S_0(t)$):**
       $$h_{0m} = \frac{d_m}{\sum_{j \in R(t_m)} \exp(X_j \hat{\beta})}, \quad H_0(t) = \sum_{t_m \le t} h_{0m}, \quad S_0(t) = \exp(-H_0(t))$$
       * **Predicting Individual Patient Survival Curves:**
         $$S(t \mid X^*) = [S_0(t)]^{\exp(X^* \hat{\beta})} = \exp\left(-H_0(t) \cdot \exp(X^* \hat{\beta})\right)$$
       * **In Plain English:** Once the regression coefficients $\hat{\beta}$ are estimated, Norman Breslow's formula calculates the underlying baseline survival curve. Any individual patient's personal survival curve is obtained simply by raising the baseline curve to the power of their personal risk multiplier!

    5. **Harrell's Concordance Index (C-Index):**
       $$C = \frac{\text{Concordant Pairs} + 0.5 \times \text{Ties}}{\text{Total Evaluable Pairs}}$$
       * **In Plain English:** Measures the predictive discrimination of the model. For every pair of patients where one had an event earlier than the other, did the model assign a higher hazard score to the patient who had the event first? A score of $0.50$ represents a coin flip, while $1.00$ indicates perfect predictive ordering.
    """)


def render_cox_excel_guide(model: Dict[str, Any], meta: Dict[str, Any]):
    """Renders step-by-step Microsoft Excel guidance for Cox regression concepts."""
    st.subheader("📗 Microsoft Excel Implementation Guide: Cox Proportional Hazards")
    st.markdown("""
    While standard Microsoft Excel does not feature a native one-click `COX.REG` function, researchers can 
    implement Cox partial likelihood optimization using **Excel Solver**, or compute multivariable Hazard Ratios 
    and construct Forest Plots using standard spreadsheet functions:
    """)

    with st.expander("📌 Step 1: Excel Layout for Partial Likelihood via Solver", expanded=False):
        st.markdown(r"""
        | Column | Header | Excel Formula Example | Description |
        | :--- | :--- | :--- | :--- |
        | **A** | `Time_t` | `4.5` | Follow-up duration to event or censoring. |
        | **B** | `Status_d` | `1` | `1` = Event observed, `0` = Censored. |
        | **C** | `Treatment_X1`| `1` | Covariate 1 (e.g. 1 = Active Drug, 0 = Standard). |
        | **D** | `Age_X2` | `62` | Covariate 2 (Continuous age in years). |
        | **E** | `Risk_Score` | `=EXP($K$1*C2 + $K$2*D2)` | Individual hazard multiplier $\exp(X_i \beta)$. |
        | **F** | `Risk_Set_Sum` | `=SUMIF($A$2:$A$200, ">=" & A2, $E$2:$E$200)` | Sum of risk scores across all patients at risk $\sum_{j \in R(t_i)} \exp(X_j \beta)$. |
        | **G** | `LogLik_i` | `=IF(B2=1, LN(E2 / F2), 0)` | Partial log-likelihood contribution for event $i$. |
        | **H** | **Total LogLik** | `=SUM(G2:G200)` | Objective cell to **Maximize** in Excel Solver by changing cells `$K$1:$K$2`. |
        """)

    with st.expander("📌 Step 2: Excel Formulas for Hazard Ratio Confidence Intervals & P-Values", expanded=False):
        st.markdown(r"""
        Once coefficients ($\beta_k$) and standard errors ($\text{SE}_k$) are computed:
        1. **Adjusted Hazard Ratio:**
           `=EXP(Beta_Cell)`
        2. **95% Confidence Interval Lower Bound:**
           `=EXP(Beta_Cell - 1.95996 * SE_Cell)`
        3. **95% Confidence Interval Upper Bound:**
           `=EXP(Beta_Cell + 1.95996 * SE_Cell)`
        4. **Two-Tailed Wald p-value:**
           `=2 * (1 - NORM.S.DIST(ABS(Beta_Cell / SE_Cell), TRUE))`
        5. **Likelihood Ratio Test (LRT) p-value:**
           `=CHISQ.DIST.RT(2 * (LogLik_Model - LogLik_Null), Degrees_of_Freedom)`
        """)

    with st.expander("📌 Step 3: Constructing the Hazard Ratio Forest Plot in Microsoft Excel (Component by Component)", expanded=False):
        st.markdown(r"""
        A professional forest plot requires five core visual components: square point estimates, horizontal error whiskers, 
        a dashed vertical reference line at $\text{HR} = 1.0$, a logarithmic horizontal axis, and vertical covariate labels. 
        Follow this exact spreadsheet setup:

        1. **Organize your Plot Data Table in Excel:**
           | Cell | Column Header | Example Row 2 | Purpose |
           | :--- | :--- | :--- | :--- |
           | **A** | `Covariate_Label` | `Early Occ Health Intervention` | Predictor variable display name. |
           | **B** | `Adjusted_HR` | `1.65` | Square marker center point estimate. |
           | **C** | `HR_95_CI_Lower` | `1.25` | Left boundary of 95% confidence interval. |
           | **D** | `HR_95_CI_Upper` | `2.18` | Right boundary of 95% confidence interval. |
           | **E** | `Error_Minus` | `=B2 - C2` *(0.40)* | Left whisker length ($HR - Lower$). |
           | **F** | `Error_Plus` | `=D2 - B2` *(0.53)* | Right whisker length ($Upper - HR$). |
           | **G** | `Y_Position` | `1` | Vertical spacing coordinate (1, 2, 3, 4...). |

        2. **Insert the Scatter Plot:**
           * Select the range for `Adjusted_HR` (Column B) and `Y_Position` (Column G).
           * In the ribbon, go to **Insert** → **Charts** → **Scatter (X, Y)** → choose **Scatter with only Markers**.

        3. **Add the Custom Horizontal Error Whiskers:**
           * Click on any marker in the chart to select the series.
           * Click **Chart Design** → **Add Chart Element** → **Error Bars** → **More Error Bar Options...**
           * Excel will insert both vertical and horizontal error bars by default. Click on the vertical error bars and press **Delete**.
           * Click on the horizontal error bars. In the **Format Error Bars** side panel:
             * **Direction:** Select `Both`.
             * **End Style:** Select `Cap`.
             * **Error Amount:** Select `Custom` and click the **Specify Value** button.
             * In **Positive Error Value**, select the `Error_Plus` range (`=$F$2:$F$5`).
             * In **Negative Error Value**, select the `Error_Minus` range (`=$E$2:$E$5`).
             * Click **OK**.

        4. **Add the Vertical Neutral Reference Line ($HR = 1.0$):**
           * In an empty area of your sheet, enter two coordinates for the neutral line:
             `X: 1.0, Y: 0` and `X: 1.0, Y: 5` (or one unit above your highest Y position).
           * Right-click the chart → **Select Data** → click **Add** under Legend Entries (Series).
           * Series Name: `"Neutral Line (HR=1.0)"`.
           * Series X values: select the two `1.0` cells; Series Y values: select the `0` and `5` cells.
           * Right-click this new series → **Format Data Series** → add a line (`Solid line`, color: `#64748b`, `Dash type: Dash`, width `1.5 pt`) and set Marker to `None`.

        5. **Configure the Logarithmic Horizontal Axis:**
           * Right-click the horizontal (X) axis numbers → **Format Axis**.
           * Under Axis Options, check the box for **Logarithmic scale** (Base 10).
           * Set Minimum Bound to `0.2` and Maximum Bound to `5.0`.
           * Under Tick Marks, set Major type to `Outside`.

        6. **Add Covariate Text Labels beside Markers:**
           * Click the primary data points → click the green `+` icon (or Chart Design) → check **Data Labels**.
           * Right-click the data labels → **Format Data Labels** → check **Value From Cells** → select the `Covariate_Label` range (`=$A$2:$A$5`).
           * Uncheck `Y Value` and `Show Leader Lines`. Set Label Position to `Left`.
        """)


def render_cox_python_guide(model: Dict[str, Any], meta: Dict[str, Any]):
    """Renders production-ready Python script using lifelines, matplotlib, and plotly."""
    st.subheader("🐍 Python Production Script Guide: Cox Proportional Hazards")
    st.markdown("""
    Below is a complete, modular Python workflow demonstrating how to fit a multivariable Cox model, 
    evaluate proportional hazards, construct publication-grade forest plots component-by-component, and generate predicted survival curves:
    """)

    covars_str = ", ".join([f'"{c}"' for c in model["covariate_names"]])

    with st.expander("📌 Step 1: Load Cohort Microdata & Define Covariates in Python", expanded=False):
        st.code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Install lifelines if needed: pip install lifelines
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# Load microdata
df = pd.read_csv("cox_regression_microdata_{meta.get('scenario_key', 'demo')}.csv")

time_col = "{meta.get('time_col', 'Follow_Up_Months')}"
event_col = "{meta.get('event_col', 'Event')}"
covariates = [{covars_str}]

# Clean and prepare subset
df_analysis = df[[time_col, event_col] + covariates].dropna()
print(f"Cohort loaded: {{len(df_analysis)}} observations, {{df_analysis[event_col].sum()}} confirmed events.")
""", language="python")

    with st.expander("📌 Step 2: Fit Multivariable Cox Proportional Hazards Model & Extract Coefficients", expanded=False):
        st.code("""# Initialize and fit Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(
    df=df_analysis,
    duration_col=time_col,
    event_col=event_col,
    show_progress=True
)

# Print full multivariable regression summary
print("=== Multivariable Cox Proportional Hazards Model Summary ===")
cph.print_summary()

# Extract model goodness-of-fit metrics
print(f"\\nHarrell's C-Index: {{cph.concordance_index_:.3f}}")
print(f"Log Partial Likelihood: {{cph.log_likelihood_:.2f}}")
print(f"AIC: {{cph.AIC_partial_:.2f}}")
""", language="python")

    with st.expander("📌 Step 3: Test Proportional Hazards Assumption (Schoenfeld Residuals)", expanded=False):
        st.code("""# Grambsch-Therneau test of Schoenfeld residuals correlation with time
print("=== Testing Proportional Hazards Assumption (Schoenfeld Residuals) ===")
ph_results = proportional_hazard_test(cph, df_analysis, time_transform="rank")
ph_results.print_summary()

# If any covariate has p < 0.05, the proportional hazards assumption is violated.
# Consider time-stratified Cox regression or adding time-interaction terms.
""", language="python")

    with st.expander("📌 Step 4: Construct the Hazard Ratio Forest Plot (Component by Component)", expanded=False):
        st.code("""# Component-by-Component Matplotlib Forest Plot on Logarithmic Scale
summary_df = cph.summary.copy()
summary_df["Hazard_Ratio"] = np.exp(summary_df["coef"])
summary_df["HR_95_CI_Lower"] = np.exp(summary_df["coef lower 95%"])
summary_df["HR_95_CI_Upper"] = np.exp(summary_df["coef upper 95%"])

# Reverse order so first covariate appears at the top
plot_df = summary_df.iloc[::-1].reset_index()
y_positions = np.arange(len(plot_df))

# Component 1: Asymmetric confidence error whisker lengths
xerr_minus = plot_df["Hazard_Ratio"] - plot_df["HR_95_CI_Lower"]
xerr_plus  = plot_df["HR_95_CI_Upper"] - plot_df["Hazard_Ratio"]

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

# Component 2: Vertical Neutral Line at HR = 1.0 (Null Hypothesis)
ax.axvline(x=1.0, color="#64748b", linestyle="--", linewidth=1.5, zorder=1, label="Neutral Line (HR = 1.0)")

# Component 3 & 4: Square Point Estimates and 95% CI Whiskers
for i, row in plot_df.iterrows():
    hr = row["Hazard_Ratio"]
    p_val = row["p"]
    color = "#94a3b8" if p_val >= 0.05 else ("#10b981" if hr < 1.0 else "#ef4444")
    
    ax.errorbar(
        x=hr, y=i,
        xerr=[[xerr_minus[i]], [xerr_plus[i]]],
        fmt='s', color=color, ecolor=color, elinewidth=2.2, capsize=5, capthick=1.8, markersize=9, zorder=3
    )
    ci_label = f"{hr:.2f} [{row['HR_95_CI_Lower']:.2f}, {row['HR_95_CI_Upper']:.2f}] (p={p_val:.3f})"
    ax.text(x=row["HR_95_CI_Upper"] * 1.08, y=i, s=ci_label, va="center", fontsize=9, color="#1e293b", fontweight="medium")

# Component 5: Configure Logarithmic Scale
ax.set_xscale("log")
ax.set_yticks(y_positions)
ax.set_yticklabels(plot_df["covariate"], fontsize=10, fontweight="medium")
ax.set_xlabel("Adjusted Hazard Ratio (Logarithmic Scale)", fontsize=11, fontweight="bold", labelpad=10)
ax.set_title("Multivariable Adjusted Hazard Ratios (aHR) & 95% Confidence Intervals", fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="x", linestyle=":", alpha=0.6, zorder=0)
ax.set_ylim(-0.8, len(plot_df) - 0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()
""", language="python")

    with st.expander("📌 Step 5: Predict Individual Patient Survival Trajectories & Milestone Rates", expanded=False):
        st.code(f"""# Predict survival trajectory for an individual patient profile
new_patient = pd.DataFrame([{{
    {", ".join([f'"{c}": df_analysis["{c}"].median()' for c in model["covariate_names"]])}
}}])

predicted_surv = cph.predict_survival_function(new_patient)
print("=== Predicted Milestone Survival Probabilities ===")
for milestone in [6.0, 12.0, 24.0, 36.0]:
    if milestone <= predicted_surv.index.max():
        p_val = predicted_surv.loc[milestone].values[0]
        print(f"{{milestone:.0f}}-Unit Survival Probability: {{p_val:.1%}}")
""", language="python")

    with st.expander("📌 Step 6: Consolidated Production Cox Proportional Hazards Pipeline Script", expanded=False):
        st.code(f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# 1. Load data
df = pd.read_csv("cox_regression_microdata_{meta.get('scenario_key', 'demo')}.csv")
time_col = "{meta.get('time_col', 'Follow_Up_Months')}"
event_col = "{meta.get('event_col', 'Event')}"
covariates = [{covars_str}]
df_clean = df[[time_col, event_col] + covariates].dropna()

# 2. Fit model
cph = CoxPHFitter()
cph.fit(df_clean, duration_col=time_col, event_col=event_col)
cph.print_summary()

# 3. Test Proportional Hazards
ph_test = proportional_hazard_test(cph, df_clean, time_transform="rank")
ph_test.print_summary()

# 4. Plot Forest Plot
plt.figure(figsize=(9, 4))
cph.plot()
plt.axvline(x=0.0, color="gray", linestyle="--")
plt.title("Adjusted Hazard Ratios (95% CI)")
plt.tight_layout()
plt.show()
""", language="python")


def render_cox_download_tab(df_microdata: pd.DataFrame, df_coefficients: pd.DataFrame, meta: Dict[str, Any]):
    """Renders download controls for Cox regression data and results."""
    st.subheader("💾 Download Cox Regression Cohort Microdata & Coefficients")
    st.markdown("Download the synthetic trial cohort microdata and fitted regression coefficients for practice in R, Stata, SAS, or Python:")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Multivariable Regression Coefficients & Hazard Ratios**")
        csv_coef = df_coefficients.to_csv(index=False)
        st.download_button(
            label="📥 Download Coefficients Table CSV",
            data=csv_coef,
            file_name=f"cox_coefficients_{meta.get('scenario_key', 'custom')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df_coefficients[[
            "Covariate", "Hazard_Ratio", "HR_95_CI_Str", "p_value_str", "Significance"
        ]], use_container_width=True, hide_index=True)

    with c2:
        st.markdown(f"**2. Patient Microdata ({len(df_microdata)} Individual Records)**")
        csv_micro = df_microdata.to_csv(index=False)
        st.download_button(
            label=f"📥 Download Cohort Microdata ({len(df_microdata)} Rows CSV)",
            data=csv_micro,
            file_name=f"cox_regression_microdata_{meta.get('scenario_key', 'custom')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df_microdata.head(6), use_container_width=True, hide_index=True)


def render_cox_regression_module(active_method_key: str, method_info: Dict[str, Any]):
    """Top-level controller for Cox Proportional Hazards Regression."""
    st.markdown("## ⏱️ Cox Proportional Hazards Regression (Adjusted Hazard Ratios)")
    st.caption("Semi-parametric multivariable survival modeling for time-to-event outcomes, non-informative right-censoring, and confounding adjustment.")

    # 1. Plain English Pedagogical Guide
    render_cox_plain_english_guide()

    # 2. Scenario Selection Sidebar / Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🩺 Longitudinal Clinical Cohort Scenario")

    scenario_options = {
        "oncology_multivariable": "🎗️ All Meadows Cancer Centre: Metastatic Colorectal Cancer (TKI vs Chemotherapy)",
        "cardiology_heart_failure": "🫀 Inner City Teaching Hospital: Post-Discharge Heart Failure (SGLT2i)",
        "nephrology_ckd": "🩸 Riverside Health Trust: Chronic Kidney Disease Progression (Intensive BP Target)",
        "occupational_return_to_work": "💼 All Meadows Health Trust: Sickness Absence & Return to Work (Early Occ Health)",
        "custom": "✏️ Custom Time-to-Event Cohort Simulation"
    }

    selected_scenario = st.sidebar.selectbox(
        "Choose Clinical Scenario:",
        options=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
        index=0
    )

    if selected_scenario != "custom":
        df_microdata, meta = generate_cox_regression_demo_data(selected_scenario)
        time_col = meta["time_col"]
        event_col = meta["event_col"]
        covariate_cols = meta["covariate_cols"]
    else:
        st.sidebar.markdown("**Input Custom Cohort Parameters:**")
        n_cust = st.sidebar.slider("Sample Size (N)", min_value=100, max_value=600, value=300, step=25)
        hr_treat = st.sidebar.slider("Hypothesized Treatment Hazard Ratio", min_value=0.30, max_value=2.00, value=0.65, step=0.05)
        hr_age = st.sidebar.slider("Hazard Ratio per 10 Years Older", min_value=1.00, max_value=1.80, value=1.25, step=0.05)
        hr_comorb = st.sidebar.slider("Hazard Ratio: Severe Comorbidity", min_value=1.00, max_value=3.00, value=1.70, step=0.10)

        np.random.seed(42)
        treat_vec = np.random.binomial(1, 0.50, size=n_cust)
        age_vec = np.clip(np.round(np.random.normal(63, 10, size=n_cust)), 38, 88)
        comorb_vec = np.random.binomial(1, 0.40, size=n_cust)

        b_treat = np.log(hr_treat)
        b_age = np.log(hr_age) / 10.0
        b_comorb = np.log(hr_comorb)

        X_cust = np.column_stack([treat_vec, age_vec - 60.0, comorb_vec])
        log_hazard = X_cust @ np.array([b_treat, b_age, b_comorb])

        u_cust = np.random.uniform(0.0001, 0.9999, size=n_cust)
        base_h = 0.022
        t_event = -np.log(u_cust) / (base_h * np.exp(log_hazard))

        censor_u = np.random.uniform(0.0001, 0.9999, size=n_cust)
        t_censor = -np.log(censor_u) / 0.012
        max_t = 36.0

        obs_t = np.round(np.minimum(np.minimum(t_event, t_censor), max_t), 1)
        obs_t = np.maximum(obs_t, 0.5)
        ev_status = ((t_event <= t_censor) & (t_event <= max_t)).astype(int)

        time_col = "Follow_Up_Months"
        event_col = "Event"
        covariate_cols = ["Investigational_Treatment", "Age_Years", "Comorbidity_Present"]

        df_microdata = pd.DataFrame({
            "Patient_ID": [f"CUST-PT-{i+1:03d}" for i in range(n_cust)],
            "Investigational_Treatment": treat_vec,
            "Age_Years": age_vec,
            "Comorbidity_Present": comorb_vec,
            time_col: obs_t,
            event_col: ev_status
        })

        meta = {
            "scenario_key": "custom",
            "title": "Custom User-Defined Multivariable Survival Cohort",
            "organisation": "User-Defined Healthcare Cohort",
            "domain": "Biostatistical & Epidemiological Simulation",
            "description": f"Custom cohort simulating {n_cust} patients with multivariable adjustment for treatment assignment, continuous age, and comorbidity burden.",
            "time_col": time_col,
            "time_unit": "Months",
            "event_col": event_col,
            "event_name": "Primary Clinical Endpoint",
            "covariate_cols": covariate_cols,
            "covariate_labels": {
                "Investigational_Treatment": "Investigational Treatment (1 = Yes, 0 = No)",
                "Age_Years": "Age (Years)",
                "Comorbidity_Present": "High Comorbidity Burden (1 = Yes, 0 = No)"
            },
            "primary_exposure": "Investigational_Treatment",
            "clinical_takeaway": "Custom simulated survival parameters calculated dynamically from user controls.",
            "disclaimer": "User-entered customized time-to-event data."
        }

    # 3. Fit Multivariable Cox Proportional Hazards Model
    model = fit_cox_proportional_hazards(
        df=df_microdata,
        time_col=time_col,
        event_col=event_col,
        covariate_cols=covariate_cols,
        alpha=0.05
    )

    # 4. Clinical Investigation Header Banner
    st.markdown(f"### 📋 Active Investigation: {meta['title']}")
    st.markdown(f"**Clinical Domain:** *{meta['domain']}* | **Fictional Institution:** *{meta['organisation']}*")
    st.markdown(f"{meta['description']}")

    # 5. Clinical Takeaway Callout Card
    with st.container(border=True):
        st.markdown("#### 💡 Clinical & Epidemiological Takeaway")
        st.markdown(f"{meta['clinical_takeaway']}")

    # 6. Methodological Prerequisite Evaluation (STROBE / CONSORT)
    evaluations = evaluate_cox_regression_prerequisites(model, df_microdata, covariate_cols)
    render_cox_diagnostics(evaluations)

    st.markdown("---")

    # 7. Primary Metrics Dashboard
    render_cox_dashboard(model, meta)

    st.markdown("---")

    # 8. Forest Plot of Adjusted Hazard Ratios
    render_cox_forest_plot(model, meta)

    st.markdown("---")

    # 9. Interactive Individual Patient Survival Predictor
    render_cox_individual_predictor(model, meta)

    st.markdown("---")

    # 10. Methodology, Implementation Guides & Downloads Tabs
    tab_rationale, tab_excel, tab_python, tab_data = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Microsoft Excel Implementation Guide",
        "🐍 Python Production Script Guide",
        "💾 Live Data Downloads"
    ])

    with tab_rationale:
        render_cox_rationale_tab(method_info, model, meta)

    with tab_excel:
        render_cox_excel_guide(model, meta)

    with tab_python:
        render_cox_python_guide(model, meta)

    with tab_data:
        render_cox_download_tab(df_microdata, model["df_coefficients"], meta)


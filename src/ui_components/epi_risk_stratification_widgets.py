"""Interactive UI Components & Visualizations for Clinical Risk Stratification & Prognostic Prediction Models.

Provides:
1. Plain English Pedagogical Introductory Guide to Clinical Risk Stratification (Zero prior knowledge).
2. Sidebar scenario controls with three realistic clinical scenarios and custom cohort exploration.
3. Methodological Prerequisite Evaluation (TRIPOD standards: Calibration slope, AUROC discrimination,
   stratum sample size stability, DCA clinical utility window, Brier score accuracy).
4. Primary Metric Dashboard (AUROC / C-Index, Calibration Slope & Intercept, Brier Score, High-Risk Event Rate).
5. Plotly Calibration Curve (Observed vs. Expected Deciles with 45-degree ideal line) and Decision Curve Analysis (DCA) Net Benefit Plot.
6. Full-Width Pedagogical Explanatory Guide beneath plots explaining how to interpret calibration and decision curves.
7. Clinical Risk Strata Performance & Action Table (Low, Moderate, High tiers with Rose Prevention Paradox insights).
8. Interactive Individual Patient Point Score & Risk Calculator.
9. Full 4-Tab Educational Architecture:
   - 📋 Rationale & Epidemiological Framework (Sullivan Framingham point formulation, logistic conversion, DCA, TRIPOD).
   - 📗 Microsoft Excel Implementation Guide (Collapsible expanders for each step, formulas, chart construction).
   - 🐍 Python Production Script Guide (Collapsible expanders for each stage, clean reusable code).
   - 💾 Live Data Downloads (Patient microdata CSV and model performance summary CSVs).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.risk_stratification import (
    build_integer_scoring_system,
    calculate_risk_strata_performance,
    evaluate_model_calibration,
    calculate_decision_curve_analysis,
    evaluate_risk_stratification_prerequisites,
    generate_risk_stratification_demo_data
)


def render_risk_stratification_plain_english_guide():
    """Renders a collapsible plain English introductory guide to clinical risk stratification and prediction models."""
    with st.expander("📖 Zero-Knowledge Guide: What is Clinical Risk Stratification & Prognostic Prediction?", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. The Clinical Dilemma: Why Population Averages Fail Individual Patients
            * In everyday healthcare, a doctor or nurse cannot simply treat a patient based on the "average mortality rate" of the entire population.
            * A 25-year-old non-smoker with normal blood pressure and an 82-year-old smoker with diabetes both walk into the clinic. Saying "the general risk of heart attack is 5%" is useless for both of them!
            * **Clinical Risk Stratification** is the science of combining multiple individual risk factors (such as age, blood pressure, smoking status, and laboratory biomarkers) into a single personalized prediction of future disease risk.
            * It allows healthcare systems to separate patients into actionable tiers: **Low Risk** (lifestyle advice), **Moderate Risk** (close monitoring and shared decision-making), and **High Risk** (immediate drug therapy or specialist admission).
            """)

            st.markdown(r"""
            #### 2. The Sullivan / Framingham Integer Point System: Clinical Usability
            * Advanced statistical models (like Multivariable Logistic Regression or Cox Proportional Hazards) produce mathematical regression $\beta$-coefficients (log-odds or log-hazards) that look like $\beta = 0.842$ or $\beta = 0.287$.
            * A busy emergency doctor or general practitioner cannot calculate exponential functions like $\frac{1}{1 + \exp(-(-4.2 + 0.842 \times \text{Age} + 0.551 \times \text{Smoking}))}$ in their head during an 8-minute consultation!
            * In 2004, biostatisticians **Sullivan, Massaro, and D'Agostino** (developers of the landmark Framingham Risk Score) created a standardized mathematical technique:
              * Pick a single reference factor to represent **1 point** (the Base Unit $B$).
              * Divide every other risk factor's $\beta$-coefficient by $B$ and round to the nearest whole integer:
                $$\text{Points}_k = \text{round}\left(\frac{\beta_k}{B}\right)$$
              * Clinicians can now simply sum up whole points on a bedside paper chart or triage app!
            """)

            st.markdown(r"""
            #### 3. Geoffrey Rose's Prevention Paradox in Risk Stratification
            * In 1981, eminent epidemiologist **Geoffrey Rose** proved a counter-intuitive public health truth:
              * *A large number of people exposed to a small risk generates far more total cases of disease than a small number of people exposed to a high risk!*
            * For example, the **High-Risk Stratum** might have a very high personal event rate ($25\%$), but contains only $10\%$ of the population (accounting for 25 cases per 1,000).
            * Meanwhile, the **Moderate-Risk Stratum** has a personal event rate of only $12\%$, but contains $60\%$ of the population (accounting for 72 cases per 1,000)!
            * Therefore, a purely high-risk clinical strategy misses the majority of disease burden in the community. Effective healthcare requires combining targeted high-risk medical treatment with broad population-level health initiatives.
            """)

        with col_g2:
            st.markdown(r"""
            #### 4. Discrimination vs. Calibration: The Two Pillars of Prediction
            A prognostic model must be evaluated on two completely separate dimensions before it can ever be trusted on real patients:
            * **Discrimination (AUROC / Harrell's C-Index):**
              * Answers: *Does the model assign higher risk scores to patients who actually get sick than to patients who stay healthy?*
              * Measured by the Area Under the Receiver Operating Characteristic Curve (AUROC). An AUROC of $0.50$ is no better than flipping a coin; $0.70$ is acceptable; $\ge 0.80$ is strong.
            * **Calibration (Slope & Intercept):**
              * Answers: *Does the numerical risk percentage match the real world?*
              * If the model predicts that a group of 100 patients each have a $20\%$ chance of a heart attack, do exactly 20 of them actually have a heart attack?
              * A model can have great discrimination but awful calibration (e.g. predicting $80\%$ risk when the true rate is $20\%$). Calibration is verified using a calibration curve, calibration slope (ideal = $1.0$), and the Brier score.
            """)

            st.markdown(r"""
            #### 5. Decision Curve Analysis (DCA): Measuring Real-World Net Benefit
            * Traditional statistics (p-values, AUROC, R-squared) cannot tell a hospital chief or guideline panel whether implementing a model will actually do more clinical good than harm.
            * Every medical test or intervention has potential harms: false alarms cause patient anxiety, invasive biopsies, unnecessary drug side-effects, and healthcare costs.
            * In 2006, **Andrew Vickers and Elena Elkin** invented **Decision Curve Analysis (DCA)**:
              * DCA calculates the **Net Benefit (NB)** of using the model across different clinical threshold probabilities ($p_t$), penalizing false-positive diagnoses proportionally to the harm of unnecessary treatment:
                $$\text{Net Benefit} = \frac{\text{True Positives}}{N} - \frac{\text{False Positives}}{N} \left(\frac{p_t}{1 - p_t}\right)$$
              * It compares the model against two default strategies: "Treat All" (giving drugs to every single patient) and "Treat None" (giving drugs to nobody).
            """)

            st.markdown(r"""
            #### 6. Reporting Guidelines: TRIPOD vs. STROBE
            * While observational risk factor studies follow the **STROBE** statement (Strengthening the Reporting of Observational Studies in Epidemiology), clinical prediction and risk scoring models are governed by the **TRIPOD** statement:
              * **TRIPOD:** *Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis*.
              * Requires full reporting of predictor definitions, handling of missing data, internal/external validation, calibration slope/intercept, and decision curves.
            """)

            st.markdown(r"""
            > [!NOTE]
            > **Educational & Demonstration Framework:**
            > All case studies, patient cohorts, and numerical examples in this section use **completely synthetic, computer-simulated data** for pedagogical training. They do not represent real patients, genuine NHS trust audits, or actual clinical trials.
            """)


def render_risk_stratification_diagnostics(evaluations: Dict[str, Any]):
    """Renders diagnostic evaluation cards for TRIPOD risk stratification assumptions."""
    st.markdown("#### 🛡️ Methodological Prerequisite Evaluation (TRIPOD Clinical Prediction Standards)")

    checks = evaluations.get("checks", [])
    if not checks:
        return

    cols = st.columns(len(checks))
    for i, item in enumerate(checks):
        with cols[i]:
            status = item["status"]
            if status == "pass":
                border_color = "#10b981"
                bg_color = "#ecfdf5"
                text_color = "#065f46"
                badge = "PASS ✅"
            elif status == "warn":
                border_color = "#f59e0b"
                bg_color = "#fffbeb"
                text_color = "#92400e"
                badge = "CAUTION ⚠️"
            else:
                border_color = "#ef4444"
                bg_color = "#fef2f2"
                text_color = "#991b1b"
                badge = "FAIL ❌"

            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1.5px solid {border_color}; border-radius: 8px; padding: 0.85rem; height: 100%; min-height: 150px;">
                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: {text_color}; margin-bottom: 0.25rem;">
                    {badge}
                </div>
                <div style="font-size: 0.88rem; font-weight: 600; color: #1e293b; margin-bottom: 0.35rem; line-height: 1.25;">
                    {item['title']}
                </div>
                <div style="font-size: 0.78rem; font-weight: 600; color: {text_color}; margin-bottom: 0.35rem;">
                    {item['finding']}
                </div>
                <div style="font-size: 0.78rem; color: #475569; line-height: 1.35;">
                    {item['explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_risk_stratification_dashboard(
    cal_metrics: Dict[str, Any],
    strata_metrics: Dict[str, Any],
    dca_metrics: Dict[str, Any],
    meta: Dict[str, Any]
):
    """Renders the primary metrics dashboard for discrimination, calibration, and clinical tiers."""
    st.markdown("### 📊 Model Discrimination, Calibration & Stratification Dashboard")
    st.caption(f"Prognostic evaluation computed on synthetic demonstration cohort of N = {strata_metrics.get('n_total', 0)} simulated patient records (Educational Demonstration Only).")

    c_idx = cal_metrics.get("c_index", 0.50)
    cal_slope = cal_metrics.get("calibration_slope", 1.0)
    cal_intercept = cal_metrics.get("calibration_intercept", 0.0)
    brier = cal_metrics.get("brier_score", 0.0)
    scaled_brier = cal_metrics.get("scaled_brier_score", 0.0)

    # Highest risk stratum
    strata_list = strata_metrics.get("strata", [])
    high_stratum = strata_list[-1] if strata_list else {}
    high_rate_pct = high_stratum.get("observed_rate_pct", 0.0)
    high_oe = high_stratum.get("oe_ratio", 1.0)
    high_label = high_stratum.get("label", "High Risk")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container(border=True):
            st.metric(
                label="Discrimination (AUROC / C-Index)",
                value=f"{c_idx:.3f}",
                delta="Strong Discrimination" if c_idx >= 0.75 else ("Acceptable" if c_idx >= 0.70 else "Poor Discrimination"),
                delta_color="normal" if c_idx >= 0.70 else "inverse",
                help="Area Under the Receiver Operating Characteristic Curve (AUROC / Harrell's C-Index). Evaluates the ability of the score to rank higher-risk patients above lower-risk patients."
            )
            st.caption("Benchmark: 0.50 (Chance) | >= 0.70 (Acceptable) | >= 0.80 (Strong)")

    with c2:
        with st.container(border=True):
            st.metric(
                label="Calibration Slope & Intercept",
                value=f"Slope: {cal_slope:.2f}",
                delta=f"Intercept: {cal_intercept:+.2f} (Ideal: 0.0)",
                delta_color="normal" if abs(cal_slope - 1.0) <= 0.15 else "inverse",
                help="Calibration slope (ideal = 1.0) and intercept (ideal = 0.0) from logistic recalibration. Slope < 1 indicates overfitting (predictions too extreme); Slope > 1 indicates underfitting."
            )
            st.caption("Ideal Slope: 1.00 | Well-Calibrated: 0.85 – 1.15")

    with c3:
        with st.container(border=True):
            st.metric(
                label="Overall Accuracy (Brier Score)",
                value=f"{brier:.3f}",
                delta=f"Scaled Brier: {scaled_brier*100:+.1f}%",
                delta_color="normal" if scaled_brier > 0 else "inverse",
                help="Brier Score: Mean squared error between predicted probabilities and observed binary outcomes (0.0 = perfect accuracy). Scaled Brier measures % improvement over null prevalence."
            )
            st.caption(f"Null Prevalence Brier: {cal_metrics.get('null_brier_score', 0.0):.3f}")

    with c4:
        with st.container(border=True):
            st.metric(
                label=f"Highest Tier: {high_label}",
                value=f"{high_rate_pct:.1f}% Observed",
                delta=f"O/E Ratio: {high_oe:.2f}",
                delta_color="normal" if abs(high_oe - 1.0) <= 0.20 else "inverse",
                help=f"Observed event rate in the highest risk stratum ({high_label}) and Observed-to-Expected (O/E) calibration ratio."
            )
            st.caption(f"{high_stratum.get('observed_events', 0)} events / {high_stratum.get('n_patients', 0)} patients ({high_stratum.get('cohort_share_pct', 0.0):.1f}% cohort)")


def render_risk_stratification_visualisations(
    cal_metrics: Dict[str, Any],
    dca_metrics: Dict[str, Any],
    meta: Dict[str, Any]
):
    """Renders side-by-side Plotly Calibration Curve and Decision Curve Analysis (DCA) plot with explanatory guide."""
    st.markdown("### 📈 Visualizing Model Calibration & Clinical Decision Curves")

    col_cal, col_dca = st.columns(2)

    with col_cal:
        st.markdown("#### 🎯 Calibration Curve (Observed vs. Expected Deciles)")
        deciles_df = cal_metrics.get("deciles_table", pd.DataFrame())

        fig_cal = go.Figure()

        if not deciles_df.empty:
            # 45-degree dashed ideal line
            max_val = max(deciles_df["Mean_Predicted_Pct"].max(), deciles_df["Observed_Event_Rate_Pct"].max())
            max_axis = min(100.0, max(25.0, np.ceil(max_val / 10.0) * 10.0))

            fig_cal.add_trace(go.Scatter(
                x=[0, max_axis],
                y=[0, max_axis],
                mode="lines",
                line=dict(color="#94a3b8", dash="dash", width=2),
                name="Ideal Calibration (45° Line)"
            ))

            # Decile points with Wilson CI error bars
            ci_minus = deciles_df["Observed_Event_Rate_Pct"] - deciles_df["CI_Lower_Pct"]
            ci_plus = deciles_df["CI_Upper_Pct"] - deciles_df["Observed_Event_Rate_Pct"]

            fig_cal.add_trace(go.Scatter(
                x=deciles_df["Mean_Predicted_Pct"],
                y=deciles_df["Observed_Event_Rate_Pct"],
                mode="markers+lines",
                marker=dict(size=10, color="#2563eb", symbol="circle"),
                line=dict(color="#2563eb", width=2),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=ci_plus,
                    arrayminus=ci_minus,
                    color="#2563eb",
                    thickness=1.5,
                    width=4
                ),
                name="Risk Deciles (Observed vs Expected)",
                hovertemplate=(
                    "<b>Decile %{text}</b><br>"
                    "Mean Predicted Risk: %{x:.1f}%<br>"
                    "Observed Event Rate: %{y:.1f}%<br>"
                    "<extra></extra>"
                ),
                text=deciles_df["Decile"]
            ))

            fig_cal.update_layout(
                xaxis_title="Mean Predicted Probability (%)",
                yaxis_title="Observed Event Rate (%)",
                xaxis=dict(range=[0, max_axis], showgrid=True, gridcolor="#f1f5f9"),
                yaxis=dict(range=[0, max_axis], showgrid=True, gridcolor="#f1f5f9"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=420,
                margin=dict(l=40, r=20, t=30, b=40)
            )

        st.plotly_chart(fig_cal, use_container_width=True)

    with col_dca:
        st.markdown("#### ⚖️ Decision Curve Analysis (DCA): Clinical Net Benefit")
        dca_df = dca_metrics.get("dca_table", pd.DataFrame())

        fig_dca = go.Figure()

        if not dca_df.empty:
            pts_pct = dca_df["Threshold_Pt"] * 100.0

            # Model Net Benefit
            fig_dca.add_trace(go.Scatter(
                x=pts_pct,
                y=dca_df["Net_Benefit_Model"],
                mode="lines+markers",
                line=dict(color="#10b981", width=3),
                marker=dict(size=6, color="#10b981"),
                name="Risk Model Strategy",
                hovertemplate="<b>Model Strategy</b><br>Threshold p_t: %{x:.0f}%<br>Net Benefit: %{y:.4f}<extra></extra>"
            ))

            # Treat All Strategy
            fig_dca.add_trace(go.Scatter(
                x=pts_pct,
                y=dca_df["Net_Benefit_Treat_All"],
                mode="lines",
                line=dict(color="#f59e0b", dash="dash", width=2),
                name="Treat All Patients",
                hovertemplate="<b>Treat All</b><br>Threshold p_t: %{x:.0f}%<br>Net Benefit: %{y:.4f}<extra></extra>"
            ))

            # Treat None Strategy (horizontal line at 0.0)
            fig_dca.add_trace(go.Scatter(
                x=pts_pct,
                y=dca_df["Net_Benefit_Treat_None"],
                mode="lines",
                line=dict(color="#64748b", dash="dot", width=1.5),
                name="Treat None (Baseline = 0)",
                hovertemplate="<b>Treat None</b><br>Net Benefit: 0.0000<extra></extra>"
            ))

            # Set reasonable Y bounds
            y_min = max(-0.05, float(dca_df["Net_Benefit_Treat_All"].min()) * 1.1)
            y_max = float(dca_df["Net_Benefit_Model"].max()) * 1.2

            fig_dca.update_layout(
                xaxis_title="Decision Threshold Probability p_t (%)",
                yaxis_title="Clinical Net Benefit (True Positives / N)",
                xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
                yaxis=dict(range=[y_min, y_max], showgrid=True, gridcolor="#f1f5f9"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=420,
                margin=dict(l=40, r=20, t=30, b=40)
            )

        st.plotly_chart(fig_dca, use_container_width=True)

    # Full-Width Explanatory Card beneath the visualisations
    with st.container(border=True):
        st.markdown("#### 📖 How to Interpret the Calibration Curve and Decision Curve (Zero-Knowledge Guide)")
        st.markdown(r"""
        * **Interpreting the Calibration Curve (Left Panel):**
          * **The 45-Degree Grey Dashed Line:** Represents the theoretical ideal of **perfect calibration**. Points lying directly on this line indicate that a predicted $10\%$ risk corresponded to exactly $10$ cases out of $100$ patients in reality.
          * **Points Above the 45° Line:** The model **under-estimated risk** (the disease occurred more often than predicted).
          * **Points Below the 45° Line:** The model **over-estimated risk** (the disease occurred less often than predicted).
          * **Calibration Slope:** Indicates whether predictions are too extreme or too conservative. A slope of **$1.00$** is ideal. A slope $< 1.00$ (e.g. $0.80$) means high-risk predictions were exaggerated and low-risk predictions were too low (classic statistical overfitting).
        * **Interpreting the Decision Curve Analysis (DCA) Plot (Right Panel):**
          * **What is Threshold Probability ($p_t$)?** It represents a clinician's or patient's willingness to accept treatment harms. A threshold of $10\%$ means: *"I am willing to treat 10 patients to prevent 1 true disease event (accepting 9 unnecessary treatments)."* A threshold of $20\%$ means: *"I require at least a 1 in 5 chance before starting medication."*
          * **The Clinical Utility Window:** The model is clinically useful across the range of thresholds where the green **Model** line sits **strictly above both the orange "Treat All" line and the grey "Treat None" line**.
          * In this clinical window, using the prediction score to guide medical decisions results in a higher Net Benefit than either giving medication to everyone blindly or withholding medication from everyone.
        """)


def render_risk_stratification_action_table(strata_metrics: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the detailed clinical strata performance table and Rose's Prevention Paradox breakdown."""
    st.markdown("### 🗂️ Clinical Risk Strata Performance & Action Pathway")
    st.markdown("""
    The continuous risk predictions are segmented into pre-specified clinical action tiers (Low, Moderate, High).
    Notice how each tier translates into a distinct clinical pathway, and how the **Moderate Risk** group frequently 
    harbors the majority of absolute disease events in the population (Geoffrey Rose's Prevention Paradox):
    """)
    st.caption("⚠️ Educational Demonstration: Clinical action pathways and expected vs. observed rates are derived from a simulated synthetic cohort for method illustration.")

    strata_df = strata_metrics.get("strata_table", pd.DataFrame())
    if strata_df.empty:
        st.info("No strata data available.")
        return

    # Add clinical recommendation column based on active scenario
    scenario_key = meta.get("scenario_key", "cardiovascular_qrisk")
    if scenario_key == "cardiovascular_qrisk":
        clinical_actions = [
            "Routine lifestyle advice (diet, exercise, smoking cessation); re-screen in 5 years.",
            "Shared decision-making; consider atorvastatin 20mg if target lifestyle changes fail; re-assess in 2 years.",
            "Immediate high-intensity statin therapy (atorvastatin 20-80mg), intensive BP control, and urgent cardiology review."
        ]
    elif scenario_key == "emergency_curb65":
        clinical_actions = [
            "Low risk of 30-day mortality: Suitable for outpatient home management with oral antibiotics.",
            "Intermediate risk: Consider short-stay hospital admission or closely supervised outpatient care.",
            "High / Severe mortality risk: Immediate hospital admission; urgent assessment for Intensive Care Unit (ICU) level care."
        ]
    else:
        clinical_actions = [
            "Low readmission risk: Standard community discharge with routine GP follow-up in 4 weeks.",
            "Intermediate frailty: Comprehensive Geriatric Assessment (CGA), medication review, and virtual ward monitoring.",
            "Severe frailty / High readmission: Urgent multidisciplinary team (MDT) case management, social care package, and rapid response team."
        ]

    # Assign actions if matching length
    display_df = strata_df.copy()
    if len(clinical_actions) == len(display_df):
        display_df["Recommended_Clinical_Action"] = clinical_actions
    else:
        display_df["Recommended_Clinical_Action"] = "Clinical review per local guideline."

    # Rename columns for presentation
    presentation_df = pd.DataFrame({
        "Risk Stratum": display_df["Stratum"],
        "Predicted Range": display_df["Threshold_Range"],
        "Patients (N)": display_df["N_Patients"],
        "Cohort Share (%)": display_df["Cohort_Share_Pct"].apply(lambda v: f"{v:.1f}%"),
        "Observed Events": display_df["Observed_Events"],
        "Observed Rate (95% CI)": display_df.apply(
            lambda r: f"{r['Observed_Event_Rate_Pct']:.1f}% [{r['CI_Lower_Pct']:.1f}%, {r['CI_Upper_Pct']:.1f}%]", axis=1
        ),
        "Expected Rate": display_df["Expected_Rate_Pct"].apply(lambda v: f"{v:.1f}%"),
        "O/E Ratio": display_df["OE_Ratio"].apply(lambda v: f"{v:.2f}"),
        "Share of Total Events": display_df["Share_Of_Total_Events_Pct"].apply(lambda v: f"{v:.1f}%"),
        "Clinical Action Pathway": display_df["Recommended_Clinical_Action"]
    })

    st.dataframe(presentation_df, use_container_width=True, hide_index=True)

    # Highlight Rose's Prevention Paradox callout
    total_events = strata_metrics.get("total_events", 0)
    strata_list = strata_metrics.get("strata", [])
    if len(strata_list) >= 2:
        mod_tier = strata_list[1] if len(strata_list) == 3 else strata_list[0]
        high_tier = strata_list[-1]

        mod_events = mod_tier.get("observed_events", 0)
        mod_share = mod_tier.get("share_of_total_events_pct", 0.0)
        high_events = high_tier.get("observed_events", 0)
        high_share = high_tier.get("share_of_total_events_pct", 0.0)

        with st.container(border=True):
            st.markdown(f"""
            **💡 The Rose Prevention Paradox in Action:**
            Notice that the **{mod_tier.get('label', 'Moderate Risk')}** stratum accounts for **{mod_events} out of {total_events} total events ({mod_share:.1f}%)**, 
            compared to **{high_events} events ({high_share:.1f}%)** in the **{high_tier.get('label', 'High Risk')}** stratum. 
            Because the moderate risk group comprises a far larger proportion of the population ({mod_tier.get('cohort_share_pct', 0.0):.1f}%), 
            the absolute burden of disease occurring in "moderate" patients exceeds that of "high risk" patients!
            """)


def render_risk_stratification_calculator(demo_data: Dict[str, Any], meta: Dict[str, Any]):
    """Renders an interactive individual patient risk score and probability calculator."""
    st.markdown("### 🧮 Interactive Individual Patient Risk Score Calculator")
    st.markdown("""
    Use the clinical controls below to simulate a specific patient presentation.
    The calculator dynamically adds the Sullivan integer point weights, computes the predicted absolute probability, 
    and assigns the patient to their validated clinical risk stratum:
    """)
    st.caption("⚠️ Educational Simulation Only: This interactive calculator is simulated for educational demonstration. It must not be used for real medical decision-making or individual patient care.")


    scenario_key = meta.get("scenario_key", "cardiovascular_qrisk")
    point_weights = demo_data.get("point_weights", {})
    base_unit = demo_data.get("base_unit", 1.0)

    col_calc_inputs, col_calc_result = st.columns([1.3, 1.0])

    with col_calc_inputs:
        with st.container(border=True):
            st.markdown("#### 🩺 Patient Clinical Characteristics")
            patient_points = 0
            user_inputs = {}

            if scenario_key == "cardiovascular_qrisk":
                c_a1, c_a2 = st.columns(2)
                with c_a1:
                    age_val = st.slider("Age (Years)", min_value=35, max_value=84, value=66, step=1)
                    if age_val >= 70:
                        patient_points += point_weights.get("Age_70_Plus", 4)
                    elif age_val >= 60:
                        patient_points += point_weights.get("Age_60_69", 3)
                    elif age_val >= 50:
                        patient_points += point_weights.get("Age_50_59", 2)

                    sbp_val = st.slider("Systolic Blood Pressure (mmHg)", min_value=100, max_value=190, value=152, step=2)
                    if sbp_val >= 160:
                        patient_points += point_weights.get("SBP_160_Plus", 3)
                    elif sbp_val >= 140:
                        patient_points += point_weights.get("SBP_140_159", 2)

                with c_a2:
                    is_smoker = st.checkbox("Current Tobacco Smoker", value=True)
                    if is_smoker:
                        patient_points += point_weights.get("Current_Smoker", 3)

                    is_diabetic = st.checkbox("Type 2 Diabetes Mellitus", value=False)
                    if is_diabetic:
                        patient_points += point_weights.get("Diabetes_Mellitus", 3)

                    is_ckd = st.checkbox("Chronic Kidney Disease (Stage 3+)", value=False)
                    if is_ckd:
                        patient_points += point_weights.get("Chronic_Kidney_Disease", 2)

            elif scenario_key == "emergency_curb65":
                st.markdown("**CURB-65 Severe Pneumonia Criteria (1 Point Each):**")
                c_c1, c_c2 = st.columns(2)
                with c_c1:
                    c_conf = st.checkbox("Confusion (AMT score <= 8)", value=False)
                    c_urea = st.checkbox("Blood Urea > 7.0 mmol/L", value=True)
                    c_rr = st.checkbox("Respiratory Rate >= 30 /min", value=True)
                with c_c2:
                    c_bp = st.checkbox("Blood Pressure Low (SBP < 90 or DBP <= 60)", value=False)
                    c_age = st.checkbox("Age >= 65 Years", value=True)

                if c_conf:
                    patient_points += point_weights.get("Confusion", 1)
                if c_urea:
                    patient_points += point_weights.get("Urea_Over_7", 1)
                if c_rr:
                    patient_points += point_weights.get("Resp_Rate_30_Plus", 1)
                if c_bp:
                    patient_points += point_weights.get("Low_Blood_Pressure", 1)
                if c_age:
                    patient_points += point_weights.get("Age_65_Plus", 1)

            else:  # geriatric_frailty
                st.markdown("**Electronic Frailty Index Deficit Items:**")
                c_f1, c_f2 = st.columns(2)
                with c_f1:
                    f_poly = st.checkbox("Polypharmacy (>= 5 regular medications)", value=True)
                    f_fall = st.checkbox("History of Falls in past 12 months", value=True)
                    f_mobil = st.checkbox("Severe Mobility Impairment", value=False)
                with c_f2:
                    f_cog = st.checkbox("Cognitive Impairment / Dementia", value=False)
                    f_social = st.checkbox("Lives Alone / Social Isolation", value=True)
                    f_hosp = st.checkbox("Emergency Admission in Past Year", value=True)

                if f_poly:
                    patient_points += point_weights.get("Polypharmacy", 2)
                if f_fall:
                    patient_points += point_weights.get("Recent_Falls", 2)
                if f_mobil:
                    patient_points += point_weights.get("Mobility_Impairment", 3)
                if f_cog:
                    patient_points += point_weights.get("Cognitive_Impairment", 3)
                if f_social:
                    patient_points += point_weights.get("Social_Isolation", 1)
                if f_hosp:
                    patient_points += point_weights.get("Prior_Emergency_Admission", 3)

    with col_calc_result:
        with st.container(border=True):
            st.markdown("#### 🎯 Personalized Prognostic Output")

            # Map point score to probability using logistic link
            # P = 1 / (1 + exp(-(alpha + beta_score * points)))
            # Standard realistic parameters per scenario:
            if scenario_key == "cardiovascular_qrisk":
                alpha = -3.85
                beta_pt = 0.28
                thresholds = [0.10, 0.20]
                labels = ["Low Risk (<10%)", "Moderate Risk (10-20%)", "High Risk (>=20%)"]
            elif scenario_key == "emergency_curb65":
                alpha = -3.50
                beta_pt = 0.82
                thresholds = [0.05, 0.15]
                labels = ["Low Risk: CURB 0-1", "Intermediate Risk: CURB 2", "High Risk: CURB 3-5"]
            else:
                alpha = -2.80
                beta_pt = 0.32
                thresholds = [0.15, 0.30]
                labels = ["Mild Frailty (<15%)", "Moderate Frailty (15-30%)", "Severe Frailty (>=30%)"]

            logit_val = alpha + beta_pt * patient_points
            pred_prob = 1.0 / (1.0 + np.exp(-logit_val))
            pred_pct = pred_prob * 100.0

            # Determine stratum
            if pred_prob < thresholds[0]:
                stratum_idx = 0
                stratum_label = labels[0]
                alert_func = st.success
                alert_icon = "🟢"
            elif pred_prob < thresholds[1]:
                stratum_idx = 1
                stratum_label = labels[1]
                alert_func = st.warning
                alert_icon = "🟡"
            else:
                stratum_idx = 2
                stratum_label = labels[2]
                alert_func = st.error
                alert_icon = "🔴"

            st.metric(
                label="Total Integer Point Score",
                value=f"{patient_points} Points",
                delta=f"Predicted Probability: {pred_pct:.1f}%",
                delta_color="normal"
            )

            alert_func(f"**{alert_icon} Stratum: {stratum_label}**")

            st.markdown(f"**Predicted Event Probability:** **{pred_pct:.1f}%**")

            # Management action
            if scenario_key == "cardiovascular_qrisk":
                if stratum_idx == 0:
                    action_text = "Lifestyle and dietary guidance; reassess in 5 years."
                elif stratum_idx == 1:
                    action_text = "Shared decision-making regarding statin initiation (atorvastatin 20mg); review in 12 months."
                else:
                    action_text = "Immediate high-intensity statin initiation, blood pressure optimization, and cardiology referral."
            elif scenario_key == "emergency_curb65":
                if stratum_idx == 0:
                    action_text = "Suitable for home treatment with oral amoxicillin / doxycycline."
                elif stratum_idx == 1:
                    action_text = "Inpatient hospital admission for IV/oral antibiotics and respiratory monitoring."
                else:
                    action_text = "Urgent inpatient admission; immediate alert to Intensive Care / Medical High Dependency Unit."
            else:
                if stratum_idx == 0:
                    action_text = "Routine primary care management and annual frailty screening."
                elif stratum_idx == 1:
                    action_text = "Comprehensive Geriatric Assessment (CGA), medication review, and virtual ward check-in."
                else:
                    action_text = "Multidisciplinary team intervention, daily community nursing, urgent falls prevention, and social care plan."

            st.info(f"**Recommended Clinical Action:** {action_text}")


def render_risk_stratification_rationale_tab(method_info: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the comprehensive Rationale & Epidemiological Framework tab for Risk Stratification."""
    st.subheader(f"Why Clinical Risk Stratification & Prognostic Models? ({method_info['name']})")
    st.markdown("""
    In clinical medicine and public health epidemiology, identifying individual risk factors (such as smoking or hypertension) 
    is only the first step. To make practical clinical decisions — such as deciding which patient should be prescribed lifelong 
    statins, who can safely recover from pneumonia at home, or who requires urgent intensive care — clinicians require **multivariable prognostic models** 
    that synthesize multiple simultaneous risk markers into a single calibrated probability.
    
    Rather than treating disease prevention as a one-size-fits-all rule, **Risk Stratification** segments populations into 
    evidence-based risk tiers (Low, Moderate, High), directing intensive medical interventions to those who will derive the greatest 
    absolute benefit while shielding low-risk individuals from unnecessary medicalization and treatment side-effects.
    """)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown(f"#### ✅ When to use {method_info['name']}")
        for item in method_info.get("when_to_use", []):
            st.markdown(f"- {item}")
    with col_w2:
        st.markdown("#### ⚠️ Key Assumptions to Verify (TRIPOD Guidelines)")
        for item in method_info.get("key_assumptions", []):
            st.markdown(f"- {item}")
        st.info(f"**Alternative if assumptions fail:** {method_info.get('alternative_test', 'Parametric Accelerated Failure Time or Machine Learning Ensembles with Recalibration')}")

    st.markdown("---")
    st.markdown("### 🔍 The Prognostic Prediction Hierarchy: Comparing Modeling Strategies")
    st.markdown(r"""
    | Modeling Strategy | Mathematical Output | Bedside Usability | Handling of Non-Linearity | Primary Evaluation Metrics |
    | :--- | :--- | :--- | :--- | :--- |
    | **Integer Point Score (Sullivan / Framingham)** | Summed integer points mapped to probability tiers | **Highest** (Can be calculated on paper in 30 seconds without computer). | Binned continuous variables into discrete step scores. | Stratum separation, AUROC, Calibration Intercept/Slope. |
    | **Multivariable Logistic Regression** | Continuous predicted probability $P = \frac{1}{1 + \exp(-X\beta)}$ | Moderate (Requires calculator or Electronic Health Record [EHR] integration). | Linear in the log-odds; requires restricted cubic splines for curves. | AUROC / C-Index, Brier Score, Hosmer-Lemeshow $\chi^2$, Decision Curve Net Benefit. |
    | **Cox Proportional Hazards Risk Model** | Absolute survival probability $S(t) = [S_0(t)]^{\exp(X\beta)}$ | Moderate (EHR-integrated calculators such as QRISK3). | Semi-parametric time baseline; linear in log-hazard. | Harrell's C-index, Time-dependent AUROC, Calibration curves at milestone years. |
    | **Machine Learning Ensembles (XGBoost / Random Forest)** | Complex non-linear risk probabilities | Lower (Requires dedicated clinical decision support software). | Handles arbitrary high-order interactions and non-linearities automatically. | Often prone to overfitting; requires rigorous post-hoc isotonic or Platt recalibration. |
    """)

    st.markdown("#### 📖 Mathematical Formulae Explained in Plain English (Zero Prior Knowledge Assumed)")
    st.markdown(r"""
    The mathematical foundations of clinical risk scoring and stratification are structured across six sequential steps:

    1. **Converting Regression $\beta$-Coefficients into Integer Point Scores (Sullivan et al., 2004):**
       $$\text{Points}_k = \text{round}\left(\frac{\beta_k}{B}\right), \quad \text{Total Score} = \sum_{k=1}^K \text{Points}_k$$
       * **What each term represents:**
         * $\beta_k$: The multivariable regression coefficient (log-odds from logistic regression or log-hazard from Cox regression) for risk factor $k$.
         * $B$: The **Base Beta Unit**, representing the regression weight assigned to exactly $1$ point score (typically chosen as the smallest positive $\beta$-coefficient in the model).
         * $\text{Points}_k$: The rounded whole integer weight assigned to predictor $k$.
         * $\text{Total Score}$: The simple arithmetic sum of all earned points across the patient's clinical profile.
       * **In Plain English:** Instead of forcing a doctor to multiply decimal log-odds in an emergency, we divide every risk factor's weight by a common base number and round to neat whole numbers (e.g. Age $>65 = 3\text{ pts}$, Smoking $= 2\text{ pts}$, Normal BP $= 0\text{ pts}$).

    2. **Mapping Total Integer Points to Predicted Absolute Probability:**
       $$P(\text{Event} \mid \text{Total Score}) = \frac{1}{1 + \exp\left(-(\alpha + \beta_{\text{score}} \times \text{Total Score})\right)}$$
       * **What each term represents:**
         * $\alpha$: The logistic intercept (representing the baseline disease log-odds when the score is zero).
         * $\beta_{\text{score}}$: The slope multiplier per unit increase in total score.
         * $P(\text{Event})$: The predicted absolute risk of experiencing the clinical endpoint (ranging strictly from $0.00$ to $1.00$).

    3. **Logistic Recalibration: Measuring Calibration Intercept and Slope:**
       $$\text{logit}(Y_i) = \ln\left(\frac{P(Y_i=1)}{1 - P(Y_i=1)}\right) = a + b \cdot \text{logit}(\hat{p}_i)$$
       * **What each term represents:**
         * $a$ (**Calibration Intercept**): Measures whether the model systematically over- or under-predicts risk on average. The ideal value is **$a = 0.0$**. A negative intercept ($a < 0$) indicates the model over-predicts (predicts higher risk than actually occurs).
         * $b$ (**Calibration Slope**): Measures the spread of risk predictions. The ideal value is **$b = 1.0$**. A slope $< 1.0$ (e.g. $0.75$) indicates **overfitting** (extreme predictions are too high, low predictions are too low).

    4. **The Brier Score (Overall Mean Squared Error):**
       $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \left(Y_i - \hat{p}_i\right)^2, \quad \text{Scaled Brier} = 1 - \frac{\text{BS}}{\text{BS}_{\text{null}}}$$
       * **What each term represents:**
         * $Y_i$: The confirmed binary outcome for patient $i$ ($1 = \text{experienced event}, 0 = \text{no event}$).
         * $\hat{p}_i$: The model's predicted probability for patient $i$ (between $0.0$ and $1.0$).
         * $\text{BS}_{\text{null}} = \bar{p}(1 - \bar{p})$: The benchmark Brier score obtained by simply assigning the cohort average prevalence $\bar{p}$ to everyone.
       * **In Plain English:** The Brier score measures the average squared difference between what the model predicted and what actually happened. A score of $0.0$ represents a clairvoyant crystal ball. Scaled Brier measures the percentage improvement over a naive model that just guesses the cohort average.

    5. **Decision Curve Analysis: Net Benefit (Andrew Vickers & Elena Elkin, 2006):**
       $$\text{Net Benefit (NB)} = \frac{\text{True Positives}}{N} - \frac{\text{False Positives}}{N} \left(\frac{p_t}{1 - p_t}\right)$$
       * **What each term represents:**
         * $p_t$: The **decision threshold probability** at which a patient or clinician chooses to initiate treatment.
         * $\frac{p_t}{1 - p_t}$: The **harm-to-benefit exchange rate**. For example, if $p_t = 0.10$ ($10\%$), the exchange rate is $\frac{0.10}{0.90} = \frac{1}{9}$, meaning a false-positive treatment is penalized as being $\frac{1}{9}$th as bad as missing a true positive case.
         * $\text{Net Benefit}$: The net proportion of true-positive patients correctly identified and treated, after subtracting a proportional penalty for false alarms.

    6. **Net Interventions Avoided per 100 Patients:**
       $$\text{Interventions Avoided} = \left(\frac{\text{Net Benefit}_{\text{Model}} - \text{Net Benefit}_{\text{Treat All}}}{\frac{p_t}{1 - p_t}}\right) \times 100$$
       * **In Plain English:** Tells a healthcare commissioner: *For every 100 patients screened with this model, how many unnecessary treatments, biopsies, or hospitalizations did we safely avoid without missing a single true disease case?*
    """)


def render_risk_stratification_excel_guide(meta: Dict[str, Any]):
    """Renders step-by-step Microsoft Excel guidance for Clinical Risk Stratification with collapsible expanders."""
    st.subheader("📗 Microsoft Excel Implementation Guide: Risk Scoring & Stratification")
    st.markdown("""
    Microsoft Excel is widely used across healthcare and public health units to construct 
    risk stratification tools, audit clinical registers, and validate prognostic prediction scores.
    The step-by-step guidance below uses the **synthetic educational cohort** available in the data downloads tab (purely for training and practice):
    """)

    with st.expander("📌 Step 1: Excel Layout for Integer Point Scoring & Total Score Calculation", expanded=False):
        st.markdown(r"""
        Create a patient cohort sheet structured as follows:

        | Column | Header | Excel Formula Example | Description |
        | :--- | :--- | :--- | :--- |
        | **A** | `Patient_ID` | `PT-1001` | Simulated synthetic subject identifier (educational only). |
        | **B** | `Age_Years` | `68` | Continuous patient age in years. |
        | **C** | `Smoker_Status` | `1` | `1` = Current smoker, `0` = Non-smoker. |
        | **D** | `SBP_mmHg` | `154` | Systolic blood pressure in mmHg. |
        | **E** | `Diabetes` | `0` | `1` = Diagnosed diabetes, `0` = None. |
        | **F** | `Pts_Age` | `=IF(B2>=70, 4, IF(B2>=60, 3, IF(B2>=50, 2, 0)))` | Points awarded for age category. |
        | **G** | `Pts_Smoke` | `=IF(C2=1, 3, 0)` | Points awarded for smoking status. |
        | **H** | `Pts_SBP` | `=IF(D2>=160, 3, IF(D2>=140, 2, 0))` | Points awarded for blood pressure. |
        | **I** | `Pts_Diabetes` | `=IF(E2=1, 3, 0)` | Points awarded for diabetes. |
        | **J** | **Total_Score** | `=SUM(F2:I2)` | Total integer points for patient $i$. |
        """)

    with st.expander("📌 Step 2: Excel Formulas for Logistic Probability Conversion & Risk Tiers", expanded=False):
        st.markdown(r"""
        Once total integer points are computed in Column J, map them to predicted absolute probabilities and clinical tiers:

        1. **Logistic Predicted Probability (Column K):**
           `=1 / (1 + EXP(-($P$1 + $P$2 * J2)))`
           *(Where cell `$P$1` holds the model intercept $\alpha$, and cell `$P$2` holds the score multiplier $\beta_{\text{score}}$).*

        2. **Clinical Risk Stratum Tier (Column L):**
           `=IFS(K2 < 0.10, "Low Risk (<10%)", K2 < 0.20, "Moderate Risk (10-20%)", TRUE, "High Risk (>=20%)")`

        3. **Clinical Action Pathway Lookup (Column M):**
           `=VLOOKUP(L2, $R$2:$S$4, 2, FALSE)`
           *(Where range `$R$2:$S$4` maps each risk tier to its clinical recommendation).*
        """)

    with st.expander("📌 Step 3: Excel Formulas for Risk Strata Performance (O/E Ratio & Deciles)", expanded=False):
        st.markdown(r"""
        To verify calibration across clinical strata in a summary table:

        1. **Count Patients in Stratum ($N$):**
           `=COUNTIF($L$2:$L$1001, "Low Risk (<10%)")`

        2. **Observed Events in Stratum ($O$):**
           `=COUNTIFS($L$2:$L$1001, "Low Risk (<10%)", $N$2:$N$1001, 1)`
           *(Where Column N holds the confirmed binary event $1/0$).*

        3. **Observed Event Rate (%):**
           `=Observed_Events_Cell / N_Patients_Cell`

        4. **Expected Events in Stratum ($E$):**
           `=SUMIF($L$2:$L$1001, "Low Risk (<10%)", $K$2:$K$1001)`
           *(Sum of predicted probabilities across patients in that stratum).*

        5. **Observed-to-Expected (O/E) Ratio:**
           `=Observed_Events_Cell / Expected_Events_Cell`
           *(Values near 1.0 indicate perfect stratum calibration; $>1$ indicates under-prediction; $<1$ indicates over-prediction).*
        """)

    with st.expander("📌 Step 4: Excel Formulas for Overall Accuracy (Brier Score)", expanded=False):
        st.markdown(r"""
        1. **Individual Squared Prediction Error (Column O):**
           `=(N2 - K2)^2`
           *(Where `N2` is actual binary outcome $1/0$, and `K2` is predicted probability).*

        2. **Overall Brier Score:**
           `=AVERAGE(O2:O1001)`

        3. **Cohort Prevalence ($\bar{p}$):**
           `=AVERAGE(N2:N1001)`

        4. **Null Benchmark Brier Score:**
           `=Prev_Cell * (1 - Prev_Cell)`

        5. **Scaled Brier Score (% Improvement over Null):**
           `=1 - (Brier_Score_Cell / Null_Brier_Cell)`
        """)

    with st.expander("📌 Step 5: Excel Formulas for Decision Curve Analysis (DCA Net Benefit)", expanded=False):
        st.markdown(r"""
        Set up a Decision Curve Analysis calculation grid for threshold probabilities $p_t \in \{0.05, 0.10, 0.15, 0.20, 0.25, 0.30\}$:

        | Column | Header | Excel Formula Example |
        | :--- | :--- | :--- |
        | **AA** | `Threshold_Pt` | `0.10` |
        | **AB** | `True_Positives (TP)` | `=COUNTIFS($K$2:$K$1001, ">=" & AA2, $N$2:$N$1001, 1)` |
        | **AC** | `False_Positives (FP)` | `=COUNTIFS($K$2:$K$1001, ">=" & AA2, $N$2:$N$1001, 0)` |
        | **AD** | **Net_Benefit_Model** | `=(AB2 / $Z$1) - (AC2 / $Z$1) * (AA2 / (1 - AA2))` |
        | **AE** | **Net_Benefit_Treat_All** | `=($Z$2 / $Z$1) - (($Z$1 - $Z$2) / $Z$1) * (AA2 / (1 - AA2))` |
        | **AF** | **Net_Benefit_Treat_None** | `0.0` |

        *(Where cell `$Z$1` is total cohort size $N = 1000$, and cell `$Z$2` is total cohort events $E = \text{COUNTIF}(N2:N1001, 1)$).*
        """)

    with st.expander("📌 Step 6: Step-by-Step Excel Chart Creation for Calibration & Decision Curves", expanded=False):
        st.markdown(r"""
        1. **Creating the Calibration Plot in Excel:**
           * Select your Deciles Summary table (Mean Predicted Probability and Observed Event Rate).
           * Click **Insert > Charts > Scatter with Straight Lines and Markers**.
           * Add a secondary data series for the **Ideal 45° Line** by creating two coordinates: $(0, 0)$ and $(50\%, 50\%)$. Set its line style to **Dashed Grey**.
           * Format both X and Y axes to have identical minimum ($0.0$) and maximum values (e.g. $0.50$ or $50\%$) to maintain a square $1:1$ aspect ratio.

        2. **Creating the Decision Curve (DCA) Plot in Excel:**
           * Select columns `Threshold_Pt`, `Net_Benefit_Model`, and `Net_Benefit_Treat_All`.
           * Click **Insert > Charts > Scatter with Smooth Lines**.
           * Format `Net_Benefit_Model` as a solid **Emerald Green Line** (width 2.5 pt).
           * Format `Net_Benefit_Treat_All` as a dashed **Orange Line**.
           * Add a horizontal reference line at $Y = 0.0$ for `Treat None` (**Grey Dotted Line**).
        """)


def render_risk_stratification_python_guide(meta: Dict[str, Any]):
    """Renders production Python script guidance for Clinical Risk Stratification with collapsible expanders."""
    st.subheader("🐍 Python Production Script Guide: Clinical Risk Stratification")
    st.markdown("""
    This production-grade Python script implements the complete clinical risk stratification workflow:
    integer point score generation, stratum performance calculation, calibration metrics (slope, intercept, Brier score),
    and Decision Curve Analysis (DCA Net Benefit).
    """)

    with st.expander("📌 Step 1: Environment Setup & Data Loading", expanded=False):
        st.code(r"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

# 1. Load synthetic patient-level clinical cohort (simulated for demonstration)
# Note: This is an artificial synthetic dataset for training; not real patient data.
df = pd.read_csv("synthetic_cardiovascular_risk_cohort.csv")
print(f"Loaded synthetic cohort: {len(df)} simulated patients with {df['Event_10Yr_CVD'].sum()} events.")
        """, language="python")

    with st.expander("📌 Step 2: Sullivan / Framingham Integer Point Score Formulation", expanded=False):
        st.code(r"""
def build_integer_scoring_system(coefficients: dict, base_unit: float = None):
    '''Converts multivariable regression beta-coefficients into integer point scores.
    Formula: Points_k = round(beta_k / base_unit)
    '''
    if base_unit is None:
        # Default base unit is the minimum positive non-zero beta
        non_zero = [abs(b) for b in coefficients.values() if abs(b) > 1e-5]
        base_unit = min(non_zero)
        
    point_weights = {var: int(np.round(beta / base_unit)) for var, beta in coefficients.items()}
    return {"base_unit": base_unit, "point_weights": point_weights}

# Example multivariable logistic regression log-odds coefficients
beta_coefs = {
    "Age_60_69": 0.56,
    "Age_70_Plus": 1.12,
    "SBP_140_159": 0.55,
    "SBP_160_Plus": 0.88,
    "Current_Smoker": 0.82,
    "Diabetes": 0.79
}

scoring_sys = build_integer_scoring_system(beta_coefs, base_unit=0.28)
print("Integer Point Weights:", scoring_sys["point_weights"])
        """, language="python")

    with st.expander("📌 Step 3: Risk Stratification & Stratum Performance Calculation", expanded=False):
        st.code(r"""
def calculate_risk_strata(df, prob_col, event_col, thresholds=[0.10, 0.20], labels=None):
    '''Categorizes continuous predicted probabilities into clinical risk tiers.'''
    if labels is None:
        labels = [f"Tier {i+1}" for i in range(len(thresholds) + 1)]
        
    bins = [-np.inf] + thresholds + [np.inf]
    df['Stratum'] = pd.cut(df[prob_col], bins=bins, labels=labels, right=False)
    
    summary = []
    total_n = len(df)
    total_events = df[event_col].sum()
    
    for label in labels:
        sub = df[df['Stratum'] == label]
        n_tier = len(sub)
        ev_tier = sub[event_col].sum() if n_tier > 0 else 0
        obs_rate = (ev_tier / n_tier) * 100 if n_tier > 0 else 0.0
        exp_rate = sub[prob_col].mean() * 100 if n_tier > 0 else 0.0
        oe_ratio = (ev_tier / sub[prob_col].sum()) if sub[prob_col].sum() > 0 else 1.0
        
        summary.append({
            "Stratum": label,
            "N_Patients": n_tier,
            "Cohort_Share_Pct": (n_tier / total_n) * 100,
            "Observed_Events": ev_tier,
            "Observed_Rate_Pct": obs_rate,
            "Expected_Rate_Pct": exp_rate,
            "OE_Ratio": oe_ratio,
            "Share_Of_Total_Events": (ev_tier / total_events) * 100
        })
        
    return pd.DataFrame(summary)

strata_table = calculate_risk_strata(df, "Predicted_Prob", "Event_10Yr_CVD")
print(strata_table)
        """, language="python")

    with st.expander("📌 Step 4: Model Calibration Metrics (Slope, Intercept, Brier Score)", expanded=False):
        st.code(r"""
def evaluate_calibration(df, prob_col, event_col):
    '''Computes calibration intercept and slope via logistic recalibration, plus Brier score.'''
    y_true = df[event_col].values
    y_prob = np.clip(df[prob_col].values, 1e-6, 1.0 - 1e-6)
    
    # Logistic recalibration: logit(Y) = a + b * logit(p_hat)
    logit_p = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
    recal_model = LogisticRegression(penalty=None, solver="lbfgs")
    recal_model.fit(logit_p, y_true)
    
    cal_intercept = recal_model.intercept_[0]
    cal_slope = recal_model.coef_[0][0]
    
    # Brier scores
    bs = brier_score_loss(y_true, y_prob)
    prev = np.mean(y_true)
    bs_null = prev * (1.0 - prev)
    scaled_bs = 1.0 - (bs / bs_null) if bs_null > 0 else 0.0
    auroc = roc_auc_score(y_true, y_prob)
    
    return {
        "AUROC": auroc,
        "Calibration_Intercept": cal_intercept,
        "Calibration_Slope": cal_slope,
        "Brier_Score": bs,
        "Scaled_Brier_Score": scaled_bs
    }

cal_eval = evaluate_calibration(df, "Predicted_Prob", "Event_10Yr_CVD")
print("Calibration & Accuracy Metrics:", cal_eval)
        """, language="python")

    with st.expander("📌 Step 5: Decision Curve Analysis (DCA Net Benefit)", expanded=False):
        st.code(r"""
def calculate_dca(df, prob_col, event_col, thresholds=np.linspace(0.02, 0.35, 18)):
    '''Computes clinical Net Benefit across threshold probabilities p_t.'''
    n_total = len(df)
    total_events = df[event_col].sum()
    rows = []
    
    for pt in thresholds:
        tp = ((df[prob_col] >= pt) & (df[event_col] == 1)).sum()
        fp = ((df[prob_col] >= pt) & (df[event_col] == 0)).sum()
        
        weight = pt / (1.0 - pt)
        nb_model = (tp / n_total) - (fp / n_total) * weight
        nb_treat_all = (total_events / n_total) - ((n_total - total_events) / n_total) * weight
        
        rows.append({
            "Threshold_Pt": pt,
            "Net_Benefit_Model": nb_model,
            "Net_Benefit_Treat_All": nb_treat_all,
            "Net_Benefit_Treat_None": 0.0
        })
        
    return pd.DataFrame(rows)

dca_table = calculate_dca(df, "Predicted_Prob", "Event_10Yr_CVD")
print(dca_table.head())
        """, language="python")


def render_risk_stratification_download_tab(
    df_microdata: pd.DataFrame,
    strata_metrics: Dict[str, Any],
    cal_metrics: Dict[str, Any],
    dca_metrics: Dict[str, Any],
    meta: Dict[str, Any]
):
    """Renders the Live Data Downloads tab providing CSV exports of synthetic cohort microdata and model results."""
    st.subheader("💾 Synthetic Cohort Datasets & Demonstration Downloads")

    st.warning(r"""
    ### ⚠️ IMPORTANT NOTICE: 100% SYNTHETIC DATA FOR DEMONSTRATION & TRAINING ONLY

    The datasets available for download below are **completely made up, computer-generated (synthetic)**, and provided **purely for educational, training, and methodological illustration purposes**.

    * **Zero Real Patients:** None of the rows represent actual human patients, real NHS clinical audit data, or genuine medical records. All patient IDs, demographic variables, physiological biomarkers, point totals, and disease outcomes were generated algorithmically using mathematical probability distributions.
    * **Educational Code Rehearsal:** These files are designed specifically for students, clinicians, and health data analysts to practice implementing risk scores, logistic calibration curves, Harrell's C-index, and Decision Curve Analysis (DCA) in Microsoft Excel, Python, R, Stata, or SAS.
    * **No Medical Utility:** Point weights, risk cutoffs, and event rates from these simulated datasets must **never** be used for clinical diagnosis, patient triage, or real-world medical decision-making.
    """)

    c_d1, c_d2 = st.columns(2)

    with c_d1:
        with st.container(border=True):
            st.markdown(f"#### 📄 Synthetic Patient Microdata ({len(df_microdata)} Simulated Records)")
            st.markdown(f"Contains **{len(df_microdata)} artificially generated synthetic patient records** with simulated clinical predictors, calculated integer points, predicted probabilities, and simulated binary outcomes.")
            st.caption("*100% synthetic / computer-generated cohort. Contains zero real patient data.*")
            csv_micro = df_microdata.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇️ Download Synthetic Patient Microdata ({len(df_microdata)} Rows CSV)",
                data=csv_micro,
                file_name=f"synthetic_{meta['scenario_key']}_patient_microdata.csv",
                mime="text/csv",
                use_container_width=True
            )
            with st.expander("👁️ Preview First 5 Rows of Synthetic Microdata", expanded=False):
                st.dataframe(df_microdata.head(5), use_container_width=True, hide_index=True)

    with c_d2:
        with st.container(border=True):
            st.markdown("#### 📊 Synthetic Risk Strata Performance Summary (CSV)")
            st.markdown("Aggregated summary of simulated patient numbers, observed event rates with Wilson score 95% CIs, expected rates, and Observed-to-Expected (O/E) calibration ratios across risk tiers.")
            st.caption("*Derived from the simulated synthetic cohort for tutorial exercises.*")
            strata_df = strata_metrics.get("strata_table", pd.DataFrame())
            csv_strata = strata_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Synthetic Strata Performance CSV",
                data=csv_strata,
                file_name=f"synthetic_{meta['scenario_key']}_strata_performance.csv",
                mime="text/csv",
                use_container_width=True
            )
            with st.expander("👁️ Preview Synthetic Strata Summary Table", expanded=False):
                st.dataframe(strata_df, use_container_width=True, hide_index=True)

    c_d3, c_d4 = st.columns(2)

    with c_d3:
        with st.container(border=True):
            st.markdown("#### 🎯 Synthetic Calibration Deciles Summary (CSV)")
            st.markdown("Contains 10 risk deciles with mean predicted probabilities and observed event rates calculated from the synthetic cohort for calibration curve plotting.")
            st.caption("*Simulated decile event rates for calibration curve construction.*")
            deciles_df = cal_metrics.get("deciles_table", pd.DataFrame())
            csv_cal = deciles_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Synthetic Calibration Deciles CSV",
                data=csv_cal,
                file_name=f"synthetic_{meta['scenario_key']}_calibration_deciles.csv",
                mime="text/csv",
                use_container_width=True
            )
            with st.expander("👁️ Preview Synthetic Deciles Table", expanded=False):
                st.dataframe(deciles_df, use_container_width=True, hide_index=True)

    with c_d4:
        with st.container(border=True):
            st.markdown("#### ⚖️ Synthetic Decision Curve Analysis (DCA) Net Benefit (CSV)")
            st.markdown("Contains Net Benefit across decision threshold probabilities comparing Model vs. Treat All vs. Treat None and net interventions avoided per 100.")
            st.caption("*Net Benefit calculated from simulated patient cohort.*")
            dca_df = dca_metrics.get("dca_table", pd.DataFrame())
            csv_dca = dca_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Synthetic Decision Curve CSV",
                data=csv_dca,
                file_name=f"synthetic_{meta['scenario_key']}_decision_curve_net_benefit.csv",
                mime="text/csv",
                use_container_width=True
            )
            with st.expander("👁️ Preview Synthetic Decision Curve Table", expanded=False):
                st.dataframe(dca_df.head(6), use_container_width=True, hide_index=True)

    with st.expander("📋 Synthetic Data Generation Methodology & Variable Dictionary", expanded=False):
        st.markdown(r"""
        #### How This Synthetic Dataset Was Created
        1. **Simulation Model:** Predictors are drawn from Gaussian (normal) and Bernoulli (binomial) probability distributions tuned to realistic clinical epidemiological parameters (e.g. mean Age $\approx 58$, SBP $\approx 136\text{ mmHg}$, $28\%$ smoking prevalence).
        2. **True Event Probability Generation:** A multivariable logistic link function converts individual predictors into a ground-truth probability of disease:
           $$P_i = \frac{1}{1 + \exp\left(-\left(\beta_0 + \sum_{k} \beta_k X_{ik}\right)\right)}$$
        3. **Outcome Generation:** Binary clinical outcomes ($1 = \text{Event}, 0 = \text{No Event}$) are sampled using Bernoulli random trials where $Y_i \sim \text{Bernoulli}(P_i)$.
        4. **Point Score Mapping:** Points are assigned using the Sullivan et al. (Framingham) integer formulation ($\text{round}(\beta_k / B)$) and summed into total risk scores.
        5. **Reproducibility:** A fixed random seed (`np.random.seed(42)`) is applied so the dataset remains perfectly reproducible across sessions for teaching, classroom demonstrations, and code verification.

        > [!NOTE]
        > Because this dataset is entirely synthetic, it is safe to share, download, publish, and use in educational workshops without information governance, Caldicott Guardian, or GDPR restrictions.
        """)


def render_risk_stratification_module(active_method_key: str, method_info: Dict[str, Any]):
    """Main rendering controller for Module 6: Clinical Risk Stratification & Prognostic Prediction Models."""

    # 1. Plain English Pedagogical Introductory Guide
    render_risk_stratification_plain_english_guide()

    # 2. Sidebar Scenario Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Risk Stratification Controls")

    scenario_options = {
        "cardiovascular_qrisk": "All Meadows Cardiovascular Institute (10-Yr CVD Risk) [Synthetic]",
        "emergency_curb65": "Inner City Teaching Hospital (CURB-65 Pneumonia Mortality) [Synthetic]",
        "geriatric_frailty": "Riverside Health Trust (eFI Hospital Readmission) [Synthetic]",
        "custom": "Custom User-Defined Prognostic Cohort Simulation [Synthetic]"
    }

    selected_scenario = st.sidebar.selectbox(
        "Select Clinical Prediction Scenario:",
        options=list(scenario_options.keys()),
        format_func=lambda k: scenario_options[k]
    )
    st.sidebar.caption("⚠️ All clinical scenarios utilize purely synthetic datasets simulated for educational demonstration.")


    if selected_scenario != "custom":
        demo_data = generate_risk_stratification_demo_data(scenario_key=selected_scenario)
        df_microdata = demo_data["df"]
        pred_col = demo_data["predicted_prob_col"]
        event_col = demo_data["event_col"]
        score_col = demo_data["score_col"]
        thresholds = demo_data["strata_thresholds"]
        labels = demo_data["strata_labels"]

        meta = {
            "scenario_key": selected_scenario,
            "title": demo_data["scenario_name"],
            "organisation": demo_data["organisation_name"],
            "domain": "Clinical Risk Stratification & Prognostic Modeling",
            "description": f"Evaluating prognostic performance, discrimination, calibration, and clinical action tiers for {demo_data['scenario_name']}.",
            "pred_col": pred_col,
            "event_col": event_col,
            "score_col": score_col
        }
    else:
        st.sidebar.markdown("#### ⚙️ Custom Cohort Parameters")
        n_cust = st.sidebar.slider("Cohort Sample Size (N)", min_value=200, max_value=2500, value=800, step=100)
        base_prev = st.sidebar.slider("Baseline Disease Event Prevalence", min_value=0.05, max_value=0.40, value=0.15, step=0.01)
        c_index_target = st.sidebar.slider("Target Model Discrimination (AUROC)", min_value=0.60, max_value=0.90, value=0.78, step=0.02)

        np.random.seed(42)
        # Generate correlated probabilities and outcomes
        z = np.random.normal(0, 1, size=n_cust)
        # Adjust spread for target c-index
        scaled_z = z * (c_index_target - 0.50) * 4.0
        logit_p = np.log(base_prev / (1.0 - base_prev)) + scaled_z
        probs = 1.0 / (1.0 + np.exp(-logit_p))
        probs = np.clip(probs, 0.005, 0.995)
        outcomes = (np.random.rand(n_cust) < probs).astype(int)

        score_points = np.round(probs * 15.0).astype(int)

        df_microdata = pd.DataFrame({
            "Patient_ID": [f"CUST-PT-{i+1:04d}" for i in range(n_cust)],
            "Total_Point_Score": score_points,
            "Predicted_Risk_Prob": np.round(probs, 4),
            "Confirmed_Event": outcomes
        })

        pred_col = "Predicted_Risk_Prob"
        event_col = "Confirmed_Event"
        score_col = "Total_Point_Score"
        thresholds = [0.10, 0.20]
        labels = ["Low Risk (<10%)", "Moderate Risk (10-20%)", "High Risk (>=20%)"]

        demo_data = {
            "df": df_microdata,
            "scenario_name": "Custom User-Defined Prognostic Cohort",
            "organisation_name": "Riverside Health Trust Community Health Directorate",
            "predicted_prob_col": pred_col,
            "event_col": event_col,
            "score_col": score_col,
            "strata_thresholds": thresholds,
            "strata_labels": labels,
            "point_weights": {"Custom_Factor_1": 2, "Custom_Factor_2": 3, "Custom_Factor_3": 4},
            "base_unit": 0.30
        }

        meta = {
            "scenario_key": "custom",
            "title": "Custom User-Defined Prognostic Cohort Simulation",
            "organisation": "Riverside Health Trust Community Health Directorate",
            "domain": "Prognostic Simulation",
            "description": f"Evaluating prognostic performance on a simulated synthetic cohort of {n_cust} patients with target AUROC {c_index_target:.2f} and baseline prevalence {base_prev*100:.1f}% (Educational Demonstration Only).",
            "pred_col": pred_col,
            "event_col": event_col,
            "score_col": score_col
        }

    # 3. Calculate Performance Metrics
    cal_metrics = evaluate_model_calibration(df_microdata, pred_col, event_col, n_bins=10)
    strata_metrics = calculate_risk_strata_performance(df_microdata, pred_col, event_col, thresholds, labels)
    dca_metrics = calculate_decision_curve_analysis(df_microdata, pred_col, event_col)

    # 4. Header Banner
    st.markdown(f"### 📋 Active Prognostic Model: {meta['title']}")
    st.markdown(f"**Clinical Domain:** *{meta['domain']}* | **Fictional Institution:** *{meta['organisation']}*")
    st.markdown(f"{meta['description']}")

    st.markdown(r"""
    > [!IMPORTANT]
    > **⚠️ Synthetic Demonstration Data Notice:**
    > All patient records, clinical predictor variables, point totals, and outcome rates in this section are **completely synthetic and computer-simulated**. They are provided **purely for statistical education, method demonstration, and tutorial exercises**. They do not represent real patients, genuine NHS trust audits, or actual clinical trials, and must never be used for individual clinical diagnosis, prognosis, or medical decision-making.
    """)

    # 5. Methodological Prerequisite Evaluation (TRIPOD)
    prereqs = evaluate_risk_stratification_prerequisites(
        calibration_metrics=cal_metrics,
        strata_metrics=strata_metrics,
        dca_metrics=dca_metrics,
        sample_size=len(df_microdata),
        total_events=int(df_microdata[event_col].sum())
    )
    render_risk_stratification_diagnostics(prereqs)

    st.markdown("---")

    # 6. Primary Metrics Dashboard
    render_risk_stratification_dashboard(cal_metrics, strata_metrics, dca_metrics, meta)

    st.markdown("---")

    # 7. Visualizations (Calibration Curve & DCA Plot)
    render_risk_stratification_visualisations(cal_metrics, dca_metrics, meta)

    st.markdown("---")

    # 8. Clinical Strata Performance Table & Action Pathway
    render_risk_stratification_action_table(strata_metrics, meta)

    st.markdown("---")

    # 9. Interactive Individual Patient Calculator
    render_risk_stratification_calculator(demo_data, meta)

    st.markdown("---")

    # 10. Educational 4-Tab Architecture
    tab_rationale, tab_excel, tab_python, tab_data = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Microsoft Excel Implementation Guide",
        "🐍 Python Production Script Guide",
        "💾 Live Data Downloads"
    ])

    with tab_rationale:
        render_risk_stratification_rationale_tab(method_info, meta)

    with tab_excel:
        render_risk_stratification_excel_guide(meta)

    with tab_python:
        render_risk_stratification_python_guide(meta)

    with tab_data:
        render_risk_stratification_download_tab(df_microdata, strata_metrics, cal_metrics, dca_metrics, meta)


"""Streamlit UI Components & Visualizations for Incidence Rate Ratio (IRR) with Person-Time.

Provides:
1. render_irr_plain_english_guide: Zero-prior-knowledge walkthrough with odometer/taxi meter analogies and Poisson offset explanation.
2. render_irr_diagnostics: Tri-color evaluation cards (PASS, WARNING, FAIL) for dynamic cohort assumptions.
3. render_irr_dashboard: Metric cards (IR_e, IR_u, IRR, 95% Wald CI, IRD, Wald and Exact Poisson p-values).
4. render_irr_visualizations: Scaled incidence rate bar comparisons, Forest Plot with 95% CIs, and Person-Time contingency table.
5. render_irr_excel_guide: Strict 3-step sequence with cell mapping and explicit formulas.
6. render_irr_python_guide: Strict 4-step sequence with exhaustive docstrings and conditional prints.
7. render_irr_download_tab: Live CSV download for both individual microdata and aggregated Person-Time table.
8. render_incidence_rate_ratio_module: Top-level controller for Incidence Rate Ratio with Person-Time.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.incidence_rate_ratio import (
    calculate_incidence_rate_ratio,
    evaluate_irr_prerequisites,
    generate_incidence_rate_ratio_demo_data
)


def render_irr_plain_english_guide():
    """Renders the zero-prior-knowledge pedagogical guide for Incidence Rate Ratio with Person-Time."""
    with st.expander("💡 Plain English Guide: Incidence Rate Ratio (IRR) with Person-Time & Poisson Regression", expanded=False):
        st.markdown("""
        In classical epidemiology and clinical trials, we often study closed cohorts where everyone is enrolled on day 1 
        and followed for the exact same duration. But in real-world public health, occupational monitoring, and care homes, 
        **populations are dynamic and open**: people move in and out, get hired, retire, or drop out at different times. 
        **Incidence Rate Ratio (IRR) with Person-Time** solves this fundamental challenge.
        """)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. Why Simple Headcounts Fail: The Need for Person-Time
            * **The Flaw of Cumulative Risk (Headcount):** If you count 10 cases among 100 people, the risk looks like $10\%$. But what if 50 of those people were followed for **10 years** and the other 50 were only followed for **2 weeks**? Treating them equally in the denominator creates severe bias!
            * **Person-Time Denominators:** Person-Time measures the **exact sum of time** each participant was observed while event-free:
              $$\text{Total Person-Time} = \sum_{i=1}^{n} \text{Observation Time of Person } i$$
            * **Equivalence:** 
              * **100 people followed for 1 year** = 100 person-years.
              * **10 people followed for 10 years** = 100 person-years.
              * Both accumulate the exact same amount of population exposure time!
            """)

            st.markdown(r"""
            #### 2. The Core Analogy: "The Taxi Meter / Odometer vs. The Headcount"
            * Imagine evaluating car reliability across a fleet:
              * **Headcount Approach:** Count how many cars broke down this year. (Unfair: a taxi driving 80,000 miles is far more likely to break down than a car parked in a garage driving 500 miles!).
              * **Person-Time Approach:** Read the **odometer / taxi meter**! Count breakdowns per **100,000 miles driven**.
            * In epidemiology, **Person-Time is the odometer of human life**: whether measured in **person-years**, **person-months**, or **resident-days**, it guarantees fair comparisons regardless of staggered entry or dropouts.
            """)

        with col_g2:
            st.markdown(r"""
            #### 3. What is the Incidence Rate Ratio (IRR)?
            * **Incidence Rate ($IR$):** The speed at which new cases occur per unit of person-time:
              $$IR_e = \frac{a}{PT_e} \quad \text{and} \quad IR_u = \frac{c}{PT_u}$$
              *(Scaled per convenient multiplier, e.g. per 1,000 person-years or per 1,000 resident-days).*
            * **Incidence Rate Ratio ($IRR$):** The ratio of the two incidence rates:
              $$IRR = \frac{IR_e}{IR_u} = \frac{a / PT_e}{c / PT_u}$$
            * **Interpreting IRR:**
              * **$IRR = 1.0$ (Null Parity):** Identical rate of disease in both groups.
              * **$IRR > 1.0$ (Harmful Exposure):** E.g., $IRR = 2.50$ means exposed workers develop illness at **2.5 times the speed** of unexposed workers.
              * **$IRR < 1.0$ (Protective Intervention):** E.g., $IRR = 0.42$ represents a **58% relative rate reduction** ($1 - 0.42 = 0.58$).
            """)

            st.markdown(r"""
            #### 4. Multivariable Extension: Poisson Regression & The Offset $\ln(PT)$
            * In real studies, we must adjust for confounders (age, smoking, comorbidities). For person-time data, we use **Poisson Regression**:
              $$\ln(E[\text{Events}_i]) = \beta_0 + \beta_1 X_i + \beta_2 \text{Age}_i + \mathbf{\ln(PT_i)}$$
            * **What is an Offset?** Notice the term $\ln(PT_i)$. It is entered into the model with its regression coefficient **fixed strictly at 1.0**!
            * Mathematically, subtracting the offset converts counts to rates:
              $$\ln(E[\text{Events}_i]) - \ln(PT_i) = \ln\left(\frac{E[\text{Events}_i]}{PT_i}\right) = \beta_0 + \beta_1 X_i + \beta_2 \text{Age}_i$$
            * Exponentiating $\exp(\beta_1)$ directly yields the **Adjusted Incidence Rate Ratio (aIRR)**!
            """)

        st.markdown(r"""
        #### 5. Key Assumptions & The Constant Hazard Rule
        * **Constant Hazard Assumption (Poisson Assumption):** Poisson rate calculations assume the incidence rate remains relatively uniform across the follow-up window. If the risk spikes dramatically with age or duration of exposure, person-time must be stratified into age bands or modeled using **Cox Proportional Hazards regression**.
        """)

        st.info(r"""
        📐 **Why are Confidence Intervals Calculated on the Natural Log Scale?**  
        Because the Incidence Rate Ratio ($IRR$) is a multiplicative ratio bounded by zero, its sampling distribution is skewed to the right. 
        Taking the natural log $\ln(IRR)$ symmetrizes the distribution into a normal bell curve! 
        Using the **Rothman-Greenland method**, the log-scale standard error is:
        $$SE(\ln(IRR)) = \sqrt{\frac{1}{a} + \frac{1}{c}}$$
        We calculate the 95% confidence interval on the log scale and then exponentiate back:
        $$\text{95\% Wald CI for IRR} = \exp\left(\ln(IRR) \pm 1.96 \times \sqrt{\frac{1}{a} + \frac{1}{c}}\right)$$
        """)


def render_irr_diagnostics(evaluations: List[Dict[str, Any]]):
    """Renders tri-color prerequisite diagnostic cards for Incidence Rate Ratio."""
    st.markdown("### 🚦 Tri-Color Bias & Methodological Diagnostics")
    st.caption("Verifies person-time metric suitability, constant hazard assumption, event numerator volume, and denominator balance.")

    cols = st.columns(len(evaluations))
    for idx, ev in enumerate(evaluations):
        status = ev["status"]
        if status == "PASS":
            badge = "✅ PASS"
            border_color = "#10b981"
            bg_color = "#f0fdf4"
            text_color = "#065f46"
        elif status == "WARNING":
            badge = "🟡 CAUTION"
            border_color = "#f59e0b"
            bg_color = "#fffbeb"
            text_color = "#92400e"
        else:
            badge = "❌ INVALID"
            border_color = "#ef4444"
            bg_color = "#fef2f2"
            text_color = "#991b1b"

        with cols[idx]:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1.5px solid {border_color}; border-radius: 8px; padding: 0.85rem; height: 100%; min-height: 140px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="font-size: 0.85rem; color: #1e293b;">{ev['name']}</strong>
                    <span style="font-size: 0.75rem; font-weight: 700; color: {text_color}; background: white; padding: 0.15rem 0.4rem; border-radius: 4px; border: 1px solid {border_color};">{badge}</span>
                </div>
                <p style="font-size: 0.8rem; color: #334155; margin-bottom: 0.4rem; line-height: 1.35;">{ev['details']}</p>
                <div style="font-size: 0.75rem; font-weight: 600; color: {text_color};">💡 {ev['recommendation']}</div>
            </div>
            """, unsafe_allow_html=True)


def render_irr_dashboard(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders summary metric cards for Incidence Rate Ratio with Person-Time."""
    st.markdown("### 📊 Primary Epidemiological Effect Measures")

    irr_val = calcs["irr"]
    irr_ll = calcs["irr_ci_lower"]
    irr_ul = calcs["irr_ci_upper"]
    mult_label = meta.get("rate_multiplier_label", f"per {int(calcs['rate_multiplier']):,} Person-Time")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label=f"Incidence Rate (Exposed): {mult_label}",
            value=f"{calcs['ir_exposed_scaled']:.2f}",
            help=f"Exposed group incidence rate: (a / PT_e) * multiplier = ({calcs['a']} / {calcs['pt_e']:,.1f})"
        )
    with c2:
        st.metric(
            label=f"Incidence Rate (Unexposed): {mult_label}",
            value=f"{calcs['ir_unexposed_scaled']:.2f}",
            help=f"Unexposed group incidence rate: (c / PT_u) * multiplier = ({calcs['c']} / {calcs['pt_u']:,.1f})"
        )
    with c3:
        st.metric(
            label="Incidence Rate Ratio (IRR)",
            value=f"{irr_val:.2f}",
            delta=f"95% CI: [{irr_ll:.2f}, {irr_ul:.2f}]",
            delta_color="off",
            help="Incidence Rate Ratio = (a / PT_e) / (c / PT_u). Null benchmark is 1.00."
        )
    with c4:
        sign = "+" if calcs["ird_scaled"] > 0 else ""
        st.metric(
            label=f"Rate Difference (IRD): {mult_label}",
            value=f"{sign}{calcs['ird_scaled']:.2f}",
            delta=f"95% CI: [{calcs['ird_ci_lower']:.2f}, {calcs['ird_ci_upper']:.2f}]",
            delta_color="off",
            help=f"Rate Difference = IR_e - IR_u = ({calcs['ir_exposed_scaled']:.2f} - {calcs['ir_unexposed_scaled']:.2f})"
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric(
            label="Log-Scale SE: SE(ln(IRR))",
            value=f"{calcs['se_ln_irr']:.4f}",
            delta=f"z = {calcs['z_score']:.2f}",
            delta_color="off",
            help="Rothman-Greenland standard error on natural log scale: sqrt(1/a + 1/c)"
        )
    with c6:
        tot_cases = calcs["a"] + calcs["c"]
        tot_pt = calcs["pt_total"]
        st.metric(
            label="Total Person-Time Follow-Up",
            value=f"{tot_pt:,.1f}",
            help=f"Total accumulated person-time across all participants ({tot_cases} total incident cases)"
        )
    with c7:
        p_val = calcs["p_value_wald"]
        p_txt = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        st.metric(
            label="Wald Test (H0: IRR = 1.0)",
            value=f"p = {p_txt}",
            help="Wald hypothesis test statistic on log scale: ln(IRR) / SE(ln(IRR))"
        )
    with c8:
        p_exact = calcs["p_value_exact_poisson"]
        p_exact_txt = "< 0.0001" if p_exact < 0.0001 else f"{p_exact:.4f}"
        st.metric(
            label="Exact Conditional Poisson Test",
            value=f"p = {p_exact_txt}",
            help="Exact binomial test conditioning on total events: Binomial(n = a + c, p = PT_e / PT_total)"
        )


def render_irr_visualizations(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders Plotly incidence rate comparisons, Forest plot for IRR, and Person-Time contingency table."""
    st.markdown("### 📈 Visualizing Incidence Rates & Relative Rate Ratios")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**1. Incidence Rate Comparison (Exposed vs. Unexposed)**")
        exp_label, unexp_label = meta.get("exposure_levels", ("Exposed", "Unexposed"))
        rate_exp = calcs["ir_exposed_scaled"]
        rate_unexp = calcs["ir_unexposed_scaled"]
        mult_label = meta.get("rate_multiplier_label", f"per {int(calcs['rate_multiplier']):,} Person-Time")

        bar_colors = ["#ef4444", "#3b82f6"] if calcs["effect_direction"] == "harmful" else ["#10b981", "#64748b"]

        fig_rate = go.Figure()
        fig_rate.add_trace(go.Bar(
            x=[exp_label, unexp_label],
            y=[rate_exp, rate_unexp],
            text=[f"{rate_exp:.2f}", f"{rate_unexp:.2f}"],
            textposition="auto",
            marker_color=bar_colors,
            width=0.5
        ))
        fig_rate.update_layout(
            yaxis_title=f"Incidence Rate ({mult_label})",
            xaxis_title="Cohort Stratum",
            height=330,
            margin=dict(l=30, r=30, t=30, b=40),
            template="plotly_white",
            yaxis=dict(range=[0, max(rate_exp, rate_unexp) * 1.35 + 0.5])
        )
        st.plotly_chart(fig_rate, use_container_width=True)

    with col_chart2:
        st.markdown("**2. Incidence Rate Ratio Forest Plot (Point Estimate & 95% Wald CI)**")
        irr = calcs["irr"]
        irr_ll = calcs["irr_ci_lower"]
        irr_ul = calcs["irr_ci_upper"]

        fig_forest = go.Figure()
        err_plus = irr_ul - irr
        err_minus = irr - irr_ll

        fig_forest.add_trace(go.Scatter(
            x=[irr],
            y=["Active Dynamic Cohort"],
            mode="markers",
            marker=dict(
                size=14,
                color="#0284c7" if calcs["effect_direction"] == "protective" else "#dc2626",
                symbol="diamond"
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[err_plus],
                arrayminus=[err_minus],
                color="#1e293b",
                thickness=2.5,
                width=8
            ),
            name="Incidence Rate Ratio"
        ))

        bound_upper = max(irr_ul, 3.0) * 1.25
        fig_forest.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="#64748b",
            annotation_text="Null Line (IRR = 1.0)",
            annotation_position="top right"
        )

        fig_forest.add_vrect(
            x0=0.0, x1=1.0,
            fillcolor="#ecfdf5", opacity=0.3, layer="below", line_width=0
        )
        fig_forest.add_vrect(
            x0=1.0, x1=bound_upper,
            fillcolor="#fef2f2", opacity=0.3, layer="below", line_width=0
        )

        fig_forest.update_layout(
            xaxis_title="Incidence Rate Ratio (IRR)",
            yaxis_title="",
            height=330,
            margin=dict(l=30, r=30, t=30, b=40),
            template="plotly_white",
            xaxis=dict(range=[0.0, bound_upper]),
            showlegend=False
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # 3. Person-Time Contingency Structure Display
    st.markdown("**3. Person-Time Contingency Structure (Exposed vs. Unexposed Marginals)**")
    a = calcs["a"]
    c = calcs["c"]
    pt_e = calcs["pt_e"]
    pt_u = calcs["pt_u"]
    time_unit = meta.get("time_unit", "Person-Years")
    mult_label = meta.get("rate_multiplier_label", "per 1,000 Person-Years")
    exp_label, unexp_label = meta.get("exposure_levels", ("Exposed", "Unexposed"))

    df_pt_display = pd.DataFrame([
        {
            "Cohort Stratum": f"🟢 {exp_label}",
            "Incident Events (a / c)": f"{a:,}",
            f"Person-Time ({time_unit})": f"{pt_e:,.1f}",
            f"Incidence Rate ({mult_label})": f"{calcs['ir_exposed_scaled']:.2f}",
            "Comparative Metric": f"IR_e = {calcs['ir_exposed_scaled']:.2f}"
        },
        {
            "Cohort Stratum": f"⚪ {unexp_label}",
            "Incident Events (a / c)": f"{c:,}",
            f"Person-Time ({time_unit})": f"{pt_u:,.1f}",
            f"Incidence Rate ({mult_label})": f"{calcs['ir_unexposed_scaled']:.2f}",
            "Comparative Metric": f"IR_u = {calcs['ir_unexposed_scaled']:.2f}"
        },
        {
            "Cohort Stratum": "📊 Total Cohort",
            "Incident Events (a / c)": f"{a + c:,} (a + c)",
            f"Person-Time ({time_unit})": f"{pt_e + pt_u:,.1f} (PT_total)",
            f"Incidence Rate ({mult_label})": f"{((a + c) / (pt_e + pt_u) * calcs['rate_multiplier']):.2f}",
            "Comparative Metric": f"IRR = {calcs['irr']:.2f} (95% CI: [{calcs['irr_ci_lower']:.2f}, {calcs['irr_ci_upper']:.2f}])"
        }
    ])

    st.dataframe(df_pt_display, use_container_width=True, hide_index=True)


def render_irr_excel_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 3-step Microsoft Excel Implementation Guide with cell coordinates."""
    st.markdown("### 📗 Microsoft Excel Implementation Workflow for Incidence Rate Ratio")
    st.markdown("Follow the 3-step sequence: **Set up Person-Time Layout** ➔ **Calculate Rates & Rate Ratio (IRR)** ➔ **Calculate Rothman-Greenland Log-scale SE, 95% Wald CI & Wald Test**.")

    a = calcs["a"]
    c = calcs["c"]
    pt_e = calcs["pt_e"]
    pt_u = calcs["pt_u"]
    mult = calcs["rate_multiplier"]
    exp_label, unexp_label = meta.get("exposure_levels", ("Exposed", "Unexposed"))
    time_unit = meta.get("time_unit", "Person-Years")

    with st.expander("📌 Step 1: Excel Person-Time Data Layout & Cell References", expanded=False):
        st.markdown(f"""
        Structure your Excel worksheet with the Person-Time cohort data in `Cells A1:E4` and calculation outputs in `Cells A6:B15`:
        
        * **Column A (Rows 2:4):** Exposure Group (`Row 2: Exposed`, `Row 3: Unexposed`, `Row 4: Total`)
        * **Column B (Rows 2:4):** Incident Events (`B2`: $a = {a}$, `B3`: $c = {c}$, `B4`: `=SUM(B2:B3)`)
        * **Column C (Rows 2:4):** Person-Time (`C2`: $PT_e = {pt_e}$, `C3`: $PT_u = {pt_u}$, `C4`: `=SUM(C2:C3)`)
        * **Column D (Rows 2:4):** Unscaled Rate (`D2`: `=B2/C2`, `D3`: `=B3/C3`, `D4`: `=B4/C4`)
        * **Column E (Rows 2:4):** Scaled Rate (`E2`: `=D2*{int(mult)}`, `E3`: `=D3*{int(mult)}`, `E4`: `=D4*{int(mult)}`)
        """)

        preview_grid = pd.DataFrame([
            {"Excel_Row": "Row 1 (Headers)", "Col_A": "Exposure_Group", "Col_B": "Incident_Events", "Col_C": f"Person_Time_{time_unit}", "Col_D": "Raw_Rate_Per_Unit", "Col_E": f"Scaled_Rate_x{int(mult)}"},
            {"Excel_Row": "Row 2 (Exposed)", "Col_A": exp_label, "Col_B": str(a), "Col_C": f"{pt_e:,.1f}", "Col_D": f"{calcs['ir_exposed']:.6f} (=B2/C2)", "Col_E": f"{calcs['ir_exposed_scaled']:.2f} (=D2*{int(mult)})"},
            {"Excel_Row": "Row 3 (Unexposed)", "Col_A": unexp_label, "Col_B": str(c), "Col_C": f"{pt_u:,.1f}", "Col_D": f"{calcs['ir_unexposed']:.6f} (=B3/C3)", "Col_E": f"{calcs['ir_unexposed_scaled']:.2f} (=D3*{int(mult)})"},
            {"Excel_Row": "Row 4 (Totals)", "Col_A": "Total Cohort", "Col_B": f"{a + c} (=SUM(B2:B3))", "Col_C": f"{pt_e + pt_u:,.1f} (=SUM(C2:C3))", "Col_D": f"{(a+c)/(pt_e+pt_u):.6f} (=B4/C4)", "Col_E": f"{((a+c)/(pt_e+pt_u)*mult):.2f} (=D4*{int(mult)})"}
        ])
        st.dataframe(preview_grid, use_container_width=True, hide_index=True)

        st.markdown("**Summary Output Cells Mapping (`Cells A6:B15`):**")
        df_summary_cells = pd.DataFrame({
            "Cell": ["B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15"],
            "Metric Name": [
                "Incidence Rate Exposed (IR_e)",
                "Incidence Rate Unexposed (IR_u)",
                "Incidence Rate Ratio (IRR)",
                "Natural Log of IRR [ln(IRR)]",
                "Log-Scale Standard Error [SE]",
                "95% CI Lower Bound for IRR",
                "95% CI Upper Bound for IRR",
                "Rate Difference (IRD unscaled)",
                "Rate Difference (IRD scaled)",
                "Wald Two-Sided p-value"
            ],
            "Excel Formula": [
                "=B2/C2",
                "=B3/C3",
                "=B6/B7",
                "=LN(B8)",
                "=SQRT((1/B2) + (1/B3))",
                "=EXP(B9 - 1.96 * B10)",
                "=EXP(B9 + 1.96 * B10)",
                "=B6 - B7",
                f"=B13 * {int(mult)}",
                "=2 * (1 - NORM.S.DIST(ABS(B9 / B10), TRUE))"
            ],
            "Calculated Value": [
                f"{calcs['ir_exposed']:.6f}",
                f"{calcs['ir_unexposed']:.6f}",
                f"{calcs['irr']:.4f}",
                f"{calcs['ln_irr']:.4f}",
                f"{calcs['se_ln_irr']:.4f}",
                f"{calcs['irr_ci_lower']:.4f}",
                f"{calcs['irr_ci_upper']:.4f}",
                f"{calcs['ird']:.6f}",
                f"{calcs['ird_scaled']:.2f}",
                f"{calcs['p_value_wald']:.4e}"
            ]
        })
        st.dataframe(df_summary_cells, use_container_width=True, hide_index=True)

    with st.expander("📌 Step 2: Calculate Incidence Rates & IRR in Excel", expanded=False):
        st.markdown(f"""
        1. **Compute Incidence Rate in Exposed ($IR_e$):** In cell `B6`, enter `=B2/C2`.
        2. **Compute Incidence Rate in Unexposed ($IR_u$):** In cell `B7`, enter `=B3/C3`.
        3. **Compute Incidence Rate Ratio (IRR):** In cell `B8`, enter:
           ```excel
           =B6 / B7
           ```
        4. **Compute Natural Logarithm of IRR:** In cell `B9`, enter `=LN(B8)`.
        5. **Compute Rate Difference (IRD):** In cell `B13`, enter `=B6 - B7`. Multiply by `{int(mult)}` in cell `B14` for scaled rate difference.
        """)

    with st.expander("📌 Step 3: Compute Rothman-Greenland Log-scale SE & 95% Wald CI in Excel", expanded=False):
        st.markdown("""
        Because Rate Ratio is multiplicative, confidence intervals are computed on the natural log scale and exponentiated back:

        1. **Compute Log-Scale Standard Error:** In cell `B10`, enter the Rothman-Greenland formula:
           ```excel
           =SQRT((1 / B2) + (1 / B3))
           ```
        2. **Compute 95% CI Lower Bound for IRR:** In cell `B11`, enter:
           ```excel
           =EXP(B9 - 1.96 * B10)
           ```
        3. **Compute 95% CI Upper Bound for IRR:** In cell `B12`, enter:
           ```excel
           =EXP(B9 + 1.96 * B10)
           ```
        4. **Compute Wald Two-Sided p-value:** In cell `B15`, enter:
           ```excel
           =2 * (1 - NORM.S.DIST(ABS(B9 / B10), TRUE))
           ```
        """)


def render_irr_python_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 4-step Python Implementation Workflow with docstrings and conditional prints."""
    st.markdown("### 🐍 Python Implementation Workflow for Incidence Rate Ratio with Person-Time")
    st.markdown("This guide follows a strict 4-step sequence: **Evaluate Prerequisites & Poisson Hazard Assumptions** ➔ **Calculate Rates & IRR / IRD Point Estimates** ➔ **Compute Log-Scale SE, 95% Wald CI & Exact Poisson Test** ➔ **Consolidated Executable Pipeline Script**.")

    a = calcs["a"]
    c = calcs["c"]
    pt_e = calcs["pt_e"]
    pt_u = calcs["pt_u"]
    mult = calcs["rate_multiplier"]

    with st.expander("📌 Step 1: Evaluate Prerequisites & Poisson Hazard Assumptions in Python", expanded=False):
        st.code('''import numpy as np

def check_irr_prerequisites(
    study_design: str,
    a: int,
    pt_e: float,
    c: int,
    pt_u: float,
    constant_hazard_verified: bool = True
) -> dict:
    """Verifies epidemiological prerequisites, person-time suitability, and constant hazard for IRR.

    Args:
        study_design (str): Must be 'dynamic_cohort', 'open_cohort', 'prospective_cohort', or 'registry'.
            (If 'case_control' or 'cross_sectional', person-time cannot be measured; use Odds Ratio or Prevalence Ratio).
        a (int): Observed new incident events in exposed group.
        pt_e (float): Total person-time accumulated by exposed cohort.
        c (int): Observed new incident events in unexposed group.
        pt_u (float): Total person-time accumulated by unexposed cohort.
        constant_hazard_verified (bool, optional): Whether disease hazard is approximately uniform across time. Defaults to True.

    Returns:
        dict: Diagnostic pass/warning/fail flags and recommended actions.
    """
    valid_designs = {"dynamic_cohort", "open_cohort", "prospective_cohort", "registry", "retrospective_cohort"}
    if study_design.lower() not in valid_designs:
        print("[INAPPROPRIATE - FAIL] Person-time cannot be measured in Case-Control or Cross-Sectional designs! Use Odds Ratio instead.")
        return {"status": "FAIL"}

    if pt_e <= 0 or pt_u <= 0:
        print("[INAPPROPRIATE - FAIL] Person-time denominators must be strictly positive numbers.")
        return {"status": "FAIL"}

    print(f"[ASSURANCE - PASS] Study design ({study_design}) correctly tracks longitudinal Person-Time exposure.")

    min_events = min(a, c)
    if min_events < 5:
        print(f"[WARNING] Event counts are sparse (minimum = {min_events} < 5). Wald CI may exhibit asymmetry; verify with Exact Poisson test.")
    else:
        print(f"[ASSURANCE - PASS] Event counts robust (minimum = {min_events} >= 5). Asymptotic normal log-scale approximation is reliable.")

    if not constant_hazard_verified:
        print("[WARNING] Non-constant hazard detected! Incidence rates accelerate or decay across follow-up time. Stratify into age bands or use Cox Proportional Hazards regression.")
    else:
        print("[ASSURANCE - PASS] Constant hazard verified. Summary Incidence Rate Ratio is a faithful measure of effect.")

    return {"status": "PASS"}
''', language="python")

    with st.expander("📌 Step 2: Calculate Rates & IRR / IRD Point Estimates in Python", expanded=False):
        st.code('''def calculate_irr_point_estimates(
    a: int,
    pt_e: float,
    c: int,
    pt_u: float,
    rate_multiplier: float = 1000.0
) -> dict:
    """Calculates unscaled rates, scaled rates, Incidence Rate Ratio (IRR), and Rate Difference (IRD).

    Args:
        a (int): Exposed incident events.
        pt_e (float): Exposed person-time.
        c (int): Unexposed incident events.
        pt_u (float): Unexposed person-time.
        rate_multiplier (float, optional): Multiplier for reporting (e.g. 1,000 for per 1,000 person-years). Defaults to 1000.0.

    Returns:
        dict: Dictionary containing:
            - 'ir_exposed': a / pt_e
            - 'ir_unexposed': c / pt_u
            - 'ir_exposed_scaled': (a / pt_e) * multiplier
            - 'ir_unexposed_scaled': (c / pt_u) * multiplier
            - 'irr': (a / pt_e) / (c / pt_u)
            - 'ird_scaled': (ir_exposed - ir_unexposed) * multiplier
            - 'effect_direction': 'harmful', 'protective', or 'null'
    """
    ir_exp = a / pt_e
    ir_unexp = c / pt_u
    irr = ir_exp / ir_unexp
    ird = ir_exp - ir_unexp

    direction = "harmful" if irr > 1.0001 else ("protective" if irr < 0.9999 else "null")

    return {
        "ir_exposed": ir_exp,
        "ir_unexposed": ir_unexp,
        "ir_exposed_scaled": ir_exp * rate_multiplier,
        "ir_unexposed_scaled": ir_unexp * rate_multiplier,
        "irr": irr,
        "ird_scaled": ird * rate_multiplier,
        "effect_direction": direction
    }
''', language="python")

    with st.expander("📌 Step 3: Compute Rothman-Greenland Log-Scale SE, Wald 95% CI & Exact Poisson Test in Python", expanded=False):
        st.code('''from scipy import stats

def compute_irr_confidence_interval_and_tests(
    a: int,
    pt_e: float,
    c: int,
    pt_u: float,
    alpha: float = 0.05
) -> dict:
    """Computes Rothman-Greenland log-scale SE, Wald 95% CI for IRR, and Exact Poisson test p-value.

    Args:
        a (int): Exposed incident events.
        pt_e (float): Exposed person-time.
        c (int): Unexposed incident events.
        pt_u (float): Unexposed person-time.
        alpha (float, optional): Significance level. Defaults to 0.05 (95% CI).

    Returns:
        dict: se_ln_irr, irr_ci_lower, irr_ci_upper, z_score, p_value_wald, p_value_exact_poisson.
    """
    ir_exp = a / pt_e
    ir_unexp = c / pt_u
    irr = ir_exp / ir_unexp
    ln_irr = np.log(irr)

    # Rothman-Greenland log-scale standard error: sqrt(1/a + 1/c)
    se_ln_irr = np.sqrt((1.0 / a) + (1.0 / c))

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    irr_ci_lower = np.exp(ln_irr - z_crit * se_ln_irr)
    irr_ci_upper = np.exp(ln_irr + z_crit * se_ln_irr)

    # Wald z-test on log scale
    z_score = ln_irr / se_ln_irr
    p_wald = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    # Exact Conditional Poisson test: conditional on total cases (a + c),
    # the exposed events follow Binomial(n = a + c, p = PT_e / (PT_e + PT_u))
    total_events = a + c
    p_null = pt_e / (pt_e + pt_u)
    binom_res = stats.binomtest(k=a, n=total_events, p=p_null, alternative="two-sided")
    p_exact = binom_res.pvalue

    return {
        "se_ln_irr": se_ln_irr,
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper,
        "z_score": z_score,
        "p_value_wald": p_wald,
        "p_value_exact_poisson": p_exact
    }
''', language="python")

    with st.expander("📌 Step 4: Consolidated Executable Pipeline Script", expanded=False):
        st.code(f'''import numpy as np
from scipy import stats

def run_incidence_rate_ratio_pipeline(
    a: int,
    pt_e: float,
    c: int,
    pt_u: float,
    study_design: str = "dynamic_cohort",
    rate_multiplier: float = {mult},
    constant_hazard_verified: bool = True
) -> dict:
    """Executes end-to-end Incidence Rate Ratio (IRR) analysis with dynamic cohort checks.

    Args:
        a (int): Exposed incident events.
        pt_e (float): Exposed person-time.
        c (int): Unexposed incident events.
        pt_u (float): Unexposed person-time.
        study_design (str, optional): Architecture. Defaults to 'dynamic_cohort'.
        rate_multiplier (float, optional): Multiplier for rates. Defaults to {mult}.
        constant_hazard_verified (bool, optional): Poisson hazard check. Defaults to True.

    Returns:
        dict: Complete results dictionary with rates, IRR, CIs, IRD, and test statistics.
    """
    print("=" * 65)
    print("EPIDEMIOLOGICAL INCIDENCE RATE RATIO (IRR) WITH PERSON-TIME PIPELINE")
    print("=" * 65)

    # Step 1: Diagnostics
    valid_designs = {{"dynamic_cohort", "open_cohort", "prospective_cohort", "registry", "retrospective_cohort"}}
    if study_design.lower() not in valid_designs:
        print("[INAPPROPRIATE - FAIL] Person-time invalid for Case-Control designs! Use Odds Ratio.")
        return {{}}

    min_events = min(a, c)
    if min_events < 5:
        print(f"[WARNING] Minimum event count is {{min_events}} (< 5). Cross-reference with Exact Poisson test.")
    else:
        print(f"[ASSURANCE - PASS] Minimum event count is {{min_events}} (>= 5). Log-scale Wald CI robust.")

    # Step 2: Rates & Point Estimates
    ir_exp = a / pt_e
    ir_unexp = c / pt_u
    ir_exp_scaled = ir_exp * rate_multiplier
    ir_unexp_scaled = ir_unexp * rate_multiplier
    irr = ir_exp / ir_unexp
    ln_irr = np.log(irr)
    ird_scaled = (ir_exp - ir_unexp) * rate_multiplier

    # Step 3: Rothman-Greenland Log-Scale SE & 95% Wald CI
    se_ln_irr = np.sqrt((1.0 / a) + (1.0 / c))
    z_crit = 1.95996
    irr_ci_lower = np.exp(ln_irr - z_crit * se_ln_irr)
    irr_ci_upper = np.exp(ln_irr + z_crit * se_ln_irr)

    # Hypothesis testing: Wald test and Exact Conditional Poisson test
    z_score = ln_irr / se_ln_irr
    p_wald = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    total_events = a + c
    p_null = pt_e / (pt_e + pt_u)
    binom_res = stats.binomtest(k=a, n=total_events, p=p_null, alternative="two-sided")
    p_exact = binom_res.pvalue

    print("\\n--- RESULTS ---")
    print(f"Exposed Incidence Rate:     {{ir_exp_scaled:.2f}} per {{int(rate_multiplier):,}} Person-Time")
    print(f"Unexposed Incidence Rate:   {{ir_unexp_scaled:.2f}} per {{int(rate_multiplier):,}} Person-Time")
    print(f"Incidence Rate Ratio (IRR): {{irr:.4f}}")
    print(f"95% Wald CI for IRR:        [{{irr_ci_lower:.4f}}, {{irr_ci_upper:.4f}}]")
    print(f"Rate Difference (IRD):      {{ird_scaled:+.2f}} per {{int(rate_multiplier):,}} Person-Time")
    print(f"Wald Hypothesis Test:       p = {{p_wald:.4e}} (z = {{z_score:.2f}})")
    print(f"Exact Conditional Poisson:  p = {{p_exact:.4e}}")

    return {{
        "ir_exposed_scaled": ir_exp_scaled,
        "ir_unexposed_scaled": ir_unexp_scaled,
        "irr": irr,
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper,
        "ird_scaled": ird_scaled,
        "p_value_wald": p_wald,
        "p_value_exact_poisson": p_exact
    }}

# Execute with scenario data ({meta['title']}):
if __name__ == "__main__":
    results = run_incidence_rate_ratio_pipeline(
        a={a}, pt_e={pt_e}, c={c}, pt_u={pt_u},
        study_design="{meta.get('study_design', 'dynamic_cohort')}",
        rate_multiplier={mult},
        constant_hazard_verified=True
    )
''', language="python")


def render_irr_download_tab(
    df_contingency: pd.DataFrame,
    df_microdata: pd.DataFrame,
    meta: Dict[str, Any]
):
    """Renders the data download tab with individual dynamic cohort microdata and aggregated Person-Time CSVs."""
    st.markdown("### 📥 Download Sample Datasets")
    st.markdown(f"""
    Download the sample data associated with the active dynamic cohort scenario: **{meta['title']}**.
    You can use these datasets to practice calculating Incidence Rates, person-time accumulation, 
    Rothman-Greenland standard errors, Wald confidence intervals, and Rate Ratios in Microsoft Excel or Python.
    """)

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.markdown("#### 1. Individual Participant Dynamic Microdata (CSV)")
        st.markdown(f"Contains **{len(df_microdata):,} dynamic cohort participant records** with columns: `Participant_ID`, `Cohort_Group`, `Exposure_Binary`, `Observation_Person_Time`, `Event_Occurred`, `Event_Status`, `Age_At_Entry`, `Sex`.")
        csv_micro = df_microdata.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Individual Participant Microdata (CSV)",
            data=csv_micro,
            file_name=f"{meta['scenario_key']}_microdata.csv",
            mime="text/csv",
            key="btn_dl_irr_microdata"
        )
        st.dataframe(df_microdata.head(8), height=220, use_container_width=True, hide_index=True)

    with col_dl2:
        st.markdown("#### 2. Aggregated Person-Time Contingency Table (CSV)")
        st.markdown("Contains the summarized Person-Time contingency table matching the exact Excel tutorial cell coordinates (`A1:E4`).")
        csv_cont = df_contingency.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Aggregated Person-Time Table (CSV)",
            data=csv_cont,
            file_name=f"{meta['scenario_key']}_person_time_table.csv",
            mime="text/csv",
            key="btn_dl_irr_contingency"
        )
        st.dataframe(df_contingency, height=220, use_container_width=True, hide_index=True)


def render_incidence_rate_ratio_module(active_method_key: str, method_info: Dict[str, Any]):
    """Renders the comprehensive Incidence Rate Ratio (IRR with Person-Time) module."""
    # 1. Plain English Guide
    render_irr_plain_english_guide()

    # 2. Sidebar Controls
    st.sidebar.header("🧪 Person-Time Cohort Settings")
    st.sidebar.markdown("**Study Scenarios & Dynamic Follow-Up:**")

    irr_scenario_choice = st.sidebar.selectbox(
        "Study Scenario:",
        options=["derbyshire_lead_foundry", "elderly_care_falls", "telehealth_copd_admissions", "custom"],
        format_func=lambda s: {
            "derbyshire_lead_foundry": "⛏️ Regional Lead Smelting Occupational Cohort (Harmful, IRR = 2.50)",
            "elderly_care_falls": "🛏️ All Meadows Care Home Fall Prevention Flooring (Protective, IRR = 0.42)",
            "telehealth_copd_admissions": "🫁 All Meadows Community COPD Telemonitoring (Protective, IRR = 0.50)",
            "custom": "⚙️ Custom Person-Time Input (Enter Your Own Counts & Time)"
        }[s],
        index=0,
        help="Select a realistic clinical/occupational dynamic cohort scenario or enter custom counts and person-time."
    )
    st.sidebar.caption("⚠️ **Data Notice:** Scenarios, populations, and event counts are made-up examples synthetically simulated for educational and illustrative purposes only.")

    if irr_scenario_choice == "custom":
        st.sidebar.subheader("Custom Person-Time Parameters")
        custom_study_design = st.sidebar.selectbox(
            "Study Design Architecture:",
            options=["dynamic_cohort", "open_cohort", "prospective_cohort", "registry", "case_control", "cross_sectional"],
            format_func=lambda d: {
                "dynamic_cohort": "Dynamic Cohort (Staggered entry / exit)",
                "open_cohort": "Open Cohort (Turnover of residents / beds)",
                "prospective_cohort": "Prospective Cohort with Person-Time",
                "registry": "Longitudinal Disease Registry",
                "case_control": "Case-Control (No person-time follow-up)",
                "cross_sectional": "Cross-Sectional (No person-time follow-up)"
            }[d],
            index=0
        )
        custom_time_unit = st.sidebar.selectbox(
            "Person-Time Measurement Unit:",
            options=["Person-Years", "Resident-Days", "Person-Months", "Patient-Hours"],
            index=0
        )
        custom_multiplier = st.sidebar.select_slider(
            "Rate Multiplier Reporting Base:",
            options=[1.0, 100.0, 1000.0, 10000.0, 100000.0],
            value=1000.0,
            format_func=lambda m: f"per {int(m):,} {custom_time_unit}"
        )
        custom_hazard_verified = st.sidebar.checkbox(
            "Constant Hazard Assumption Verified (Uniform risk over time)",
            value=True,
            help="Uncheck if risk changes strongly with age or duration of follow-up."
        )

        col_sb1, col_sb2 = st.sidebar.columns(2)
        with col_sb1:
            a_val = st.number_input("Exposed Incident Events (a):", min_value=0, max_value=100000, value=50, step=1)
            pt_e_val = st.number_input(f"Exposed Person-Time (PT_e):", min_value=0.1, max_value=1000000.0, value=5000.0, step=50.0)
        with col_sb2:
            c_val = st.number_input("Unexposed Incident Events (c):", min_value=0, max_value=100000, value=40, step=1)
            pt_u_val = st.number_input(f"Unexposed Person-Time (PT_u):", min_value=0.1, max_value=1000000.0, value=8000.0, step=50.0)

        meta = {
            "title": "Custom User-Defined Dynamic Cohort Investigation",
            "domain": "Epidemiological Hypothesis Testing",
            "description": "User-configured dynamic cohort person-time evaluation.",
            "exposure_name": "Exposure / Intervention",
            "exposure_levels": ("Exposed Cohort", "Unexposed Cohort"),
            "outcome_name": "Target Incident Health Event",
            "outcome_levels": ("Incident Event (+)", "Event-Free (-)"),
            "study_design": custom_study_design,
            "expected_effect": "User Defined",
            "clinical_takeaway": "Custom parameters evaluated based on user-entered events and accumulated person-time.",
            "scenario_key": "custom_incidence_rate_ratio",
            "time_unit": custom_time_unit,
            "rate_multiplier": float(custom_multiplier),
            "rate_multiplier_label": f"per {int(custom_multiplier):,} {custom_time_unit}",
            "is_constant_hazard_verified": custom_hazard_verified
        }

        # Build contingency and microdata
        df_contingency = pd.DataFrame([
            {
                "Cohort Stratum": "🟢 Exposed Cohort",
                "Incident Events (a / c)": a_val,
                f"Person-Time ({custom_time_unit})": f"{pt_e_val:,.1f}",
                f"Incidence Rate ({meta['rate_multiplier_label']})": float(a_val / pt_e_val * custom_multiplier) if pt_e_val > 0 else 0.0
            },
            {
                "Cohort Stratum": "⚪ Unexposed Cohort",
                "Incident Events (a / c)": c_val,
                f"Person-Time ({custom_time_unit})": f"{pt_u_val:,.1f}",
                f"Incidence Rate ({meta['rate_multiplier_label']})": float(c_val / pt_u_val * custom_multiplier) if pt_u_val > 0 else 0.0
            },
            {
                "Cohort Stratum": "📊 Total Cohort",
                "Incident Events (a / c)": a_val + c_val,
                f"Person-Time ({custom_time_unit})": f"{pt_e_val + pt_u_val:,.1f}",
                f"Incidence Rate ({meta['rate_multiplier_label']})": float((a_val + c_val) / (pt_e_val + pt_u_val) * custom_multiplier) if (pt_e_val + pt_u_val) > 0 else 0.0
            }
        ])

        micro_list = []
        for i in range(min(a_val * 3, 100)):
            had_event = 1 if i < a_val else 0
            micro_list.append({
                "Participant_ID": f"EXP_{i+1:04d}",
                "Cohort_Group": "Exposed",
                "Exposure_Binary": 1,
                "Observation_Person_Time": round(pt_e_val / max(a_val * 3, 100), 2),
                "Event_Occurred": had_event,
                "Event_Status": "Incident Event (+)" if had_event else "Event-Free (-)",
                "Age_At_Entry": 50,
                "Sex": "Male" if i % 2 == 0 else "Female"
            })
        for i in range(min(c_val * 3, 100)):
            had_event = 1 if i < c_val else 0
            micro_list.append({
                "Participant_ID": f"UNEXP_{i+1:04d}",
                "Cohort_Group": "Unexposed",
                "Exposure_Binary": 0,
                "Observation_Person_Time": round(pt_u_val / max(c_val * 3, 100), 2),
                "Event_Occurred": had_event,
                "Event_Status": "Incident Event (+)" if had_event else "Event-Free (-)",
                "Age_At_Entry": 52,
                "Sex": "Male" if i % 2 == 0 else "Female"
            })
        df_microdata = pd.DataFrame(micro_list)

        calcs = calculate_incidence_rate_ratio(a_val, pt_e_val, c_val, pt_u_val, rate_multiplier=custom_multiplier)
        evaluations = evaluate_irr_prerequisites(
            study_design=custom_study_design,
            a=a_val, pt_e=pt_e_val, c=c_val, pt_u=pt_u_val,
            is_constant_hazard_verified=custom_hazard_verified
        )
    else:
        df_contingency, df_microdata, meta = generate_incidence_rate_ratio_demo_data(scenario=irr_scenario_choice, seed=42)
        calcs = meta["calcs"]
        evaluations = evaluate_irr_prerequisites(
            study_design=meta.get("study_design", "dynamic_cohort"),
            a=meta["a"], pt_e=meta["pt_e"], c=meta["c"], pt_u=meta["pt_u"],
            is_constant_hazard_verified=True
        )

    # Narrative callout for active scenario
    st.markdown(f"""
    <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 1.1rem; border-radius: 6px; margin-bottom: 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.35rem; flex-wrap: wrap; gap: 0.5rem;">
            <h4 style="margin: 0; color: #1e3a8a;">🔬 Active Scenario: {meta['title']}</h4>
            <span style="font-size: 0.75rem; background-color: #fef3c7; color: #92400e; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 600; border: 1px solid #fde68a;">⚠️ Made-Up Illustrative Scenario</span>
        </div>
        <p style="margin: 0 0 0.4rem 0; font-size: 0.95rem; color: #334155;">{meta['description']}</p>
        <div style="font-size: 0.85rem; color: #0369a1; font-weight: 600; margin-bottom: 0.4rem;">💡 Clinical Takeaway: {meta['clinical_takeaway']}</div>
        <div style="font-size: 0.8rem; color: #64748b; font-style: italic; border-top: 1px dashed #cbd5e1; padding-top: 0.4rem;">
            ⚠️ <strong>Disclaimer:</strong> This is a completely made-up example scenario with synthetic data simulated for educational and illustrative purposes only. It does not represent actual medical records or real trial data from NHS trusts or local authorities.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Tri-Color Diagnostics
    render_irr_diagnostics(evaluations)

    # 2. Metric Dashboard
    render_irr_dashboard(calcs, meta)

    # 3. Visualizations & Contingency Table
    render_irr_visualizations(calcs, meta)

    # 4. Methodology Tabs
    st.markdown("---")
    tab_irr_rationale, tab_irr_excel, tab_irr_code, tab_irr_download = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Excel Implementation Guide",
        "🐍 Python Code Implementation",
        "📥 Download Sample Data"
    ])

    with tab_irr_rationale:
        st.subheader(f"Why Incidence Rate Ratio? ({method_info['name']})")
        st.markdown("""
        When study participants enter at staggered calendar dates or have varying lengths of observation time, 
        standard cumulative incidence risk ($RR = I_e / I_u$) and odds ratios ($OR$) fail. 
        **Incidence Rate Ratio (IRR)** provides the gold-standard method for dynamic cohorts, occupational surveillance, and care home monitoring.
        """)

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown(f"#### ✅ When to use {method_info['name']}")
            for item in method_info["when_to_use"]:
                st.markdown(f"- {item}")
        with col_w2:
            st.markdown("#### ⚠️ Key Assumptions to Verify")
            for item in method_info["key_assumptions"]:
                st.markdown(f"- {item}")
            st.info(f"**Alternative if assumptions fail:** {method_info['alternative_test']}")

        st.markdown("---")
        st.markdown("### 🔍 The Complete Epidemiological Effect Measures Matrix")
        st.markdown(r"""
        | Dimension | Incidence Rate Ratio (IRR) | Relative Risk (RR) | Odds Ratio (OR) | Risk Difference (RD) |
        | :--- | :--- | :--- | :--- | :--- |
        | **Study Architecture** | **Dynamic / Open Cohorts** with varying follow-up times | **Closed Fixed Cohorts** & Clinical Trials | **Case-Control** & Cross-Sectional Studies | **Clinical Trials** & Closed Cohorts |
        | **Denominator** | **Person-Time** ($PT = \text{person-years}$, resident-days) | **Headcount** ($n = \text{total participants}$) | **Non-cases / Controls** ($b$ and $d$) | **Headcount** ($n_1$ and $n_0$) |
        | **Mathematical Formula** | $\frac{a / PT_e}{c / PT_u}$ | $\frac{a / (a + b)}{c / (c + d)}$ | $\frac{a \times d}{b \times c}$ | $\frac{a}{a+b} - \frac{c}{c+d}$ |
        | **Scale & CI Method** | **Multiplicative / Log Scale** ($\exp[\ln(IRR) \pm 1.96 \cdot SE]$) | **Multiplicative / Log Scale** ($\exp[\ln(RR) \pm 1.96 \cdot SE]$) | **Multiplicative / Log Scale** ($\exp[\ln(OR) \pm 1.96 \cdot SE]$) | **Additive / Linear Scale** ($RD \pm 1.96 \cdot SE$) |
        | **Null Benchmark** | **$1.00$** (Equal Rates) | **$1.00$** (Equal Risks) | **$1.00$** (Equal Odds) | **$0.00$** (Zero Difference) |
        | **Standard Error** | $\sqrt{\frac{1}{a} + \frac{1}{c}}$ (Rothman-Greenland) | $\sqrt{\frac{1}{a} - \frac{1}{n_1} + \frac{1}{c} - \frac{1}{n_0}}$ | $\sqrt{\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d}}$ (Woolf) | $\sqrt{\frac{I_e(1-I_e)}{n_1} + \frac{I_u(1-I_u)}{n_0}}$ |
        | **Regression Model** | **Poisson Regression** (with $\ln(PT)$ offset) | **Log-Binomial Regression** | **Binary Logistic Regression** | **Linear Probability Model** |
        """)

        st.markdown("#### 📖 Key Concepts Explained (Zero Prior Knowledge Assumed)")
        st.markdown(r"""
        1. **What is an Offset in Poisson Regression and Why is it Necessary?**
           * Suppose you fit a standard linear or logistic model to predict how many heart attacks occur. An individual observed for **10 years** will naturally have more events than an individual observed for **3 months**!
           * In Poisson regression:
             $$\ln(E[Y_i]) = \beta_0 + \beta_1 X_i + \beta_2 \text{Age}_i + \mathbf{1.0 \times \ln(PT_i)}$$
           * By including $\ln(PT_i)$ with its coefficient fixed strictly to **$1.0$** (called an **offset**), we use the log rule $\ln(A) - \ln(B) = \ln(A/B)$:
             $$\ln(E[Y_i]) - \ln(PT_i) = \ln\left(\frac{E[Y_i]}{PT_i}\right) = \beta_0 + \beta_1 X_i + \beta_2 \text{Age}_i$$
           * This guarantees that the model evaluates the **rate of disease per unit of time**, completely neutralizing differences in observation duration!
           * Exponentiating the regression coefficient $\exp(\beta_1)$ directly equals the **Adjusted Incidence Rate Ratio (aIRR)**.

        2. **The Constant Hazard Assumption (The Poisson Assumption):**
           * Poisson rates assume that within any specified exposure group, the probability of an event per unit of time is uniform across the entire follow-up window.
           * *When this assumption fails:* For example, in a 30-year study of lead workers, heart disease risk accelerates dramatically as workers age from 30 to 60.
           * *Solution:* We break down person-time into 5-year age bands (stratified rate standardization) or fit a **Cox Proportional Hazards model** which allows the baseline hazard to fluctuate freely over time.

        3. **Why Exact Poisson Test is Preferred for Small Event Counts:**
           * The standard Wald confidence interval assumes that $\ln(IRR)$ follows an asymptotic normal distribution, which requires reasonably large counts (typically $a \ge 5$ and $c \ge 5$).
           * When events are sparse (e.g. 2 cases vs 1 case), we condition on the total number of events $n = a + c$. Under the null hypothesis that $IRR = 1.0$, the probability of an event falling into the exposed group is exactly proportional to its person-time:
             $$p = \frac{PT_e}{PT_e + PT_u}$$
           * We test whether $a$ differs significantly from $\text{Binomial}(n = a + c, p)$ using `scipy.stats.binomtest`. This gives an exact p-value with zero reliance on large-sample approximations!
        """)

    with tab_irr_excel:
        render_irr_excel_guide(calcs, meta)

    with tab_irr_code:
        render_irr_python_guide(calcs, meta)

    with tab_irr_download:
        render_irr_download_tab(df_contingency, df_microdata, meta)


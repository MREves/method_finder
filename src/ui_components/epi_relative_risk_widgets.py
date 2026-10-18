"""Streamlit UI Components & Visualizations for Relative Risk (Risk Ratio) & Cohort Effect Measures.

Provides:
1. render_relative_risk_plain_english_guide: Zero-prior-knowledge walkthrough with analogies.
2. render_relative_risk_diagnostics: Tri-color evaluation cards (PASS, WARNING, FAIL).
3. render_relative_risk_dashboard: Metric cards (I_e, I_u, RR, RD, AF_e, NNT/NNH, p-values).
4. render_relative_risk_visualizations: Cumulative incidence bar chart, forest plot with 95% CIs, and 2x2 grid.
5. render_relative_risk_excel_guide: Strict 3-step sequence with cell mapping and explicit formulas.
6. render_relative_risk_python_guide: Strict 4-step sequence with exhaustive docstrings and conditional prints.
7. render_relative_risk_download_tab: Live CSV download for both individual microdata and 2x2 table.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.relative_risk import (
    calculate_relative_risk,
    evaluate_relative_risk_prerequisites,
    generate_relative_risk_demo_data
)


def render_relative_risk_plain_english_guide():
    """Renders the zero-prior-knowledge pedagogical guide for Relative Risk."""
    with st.expander("💡 Plain English Guide: Relative Risk (Risk Ratio) & Cohort Study Design", expanded=False):
        st.markdown("""
        In epidemiology and public health, **Relative Risk (also known as the Risk Ratio, or RR)** compares the likelihood 
        of developing a health condition between people who were exposed to a risk factor (or intervention) and people who were not.
        """)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("""
            #### 1. What is "Risk" (Cumulative Incidence)?
            * **Risk** is simply the proportion of disease-free individuals who develop the disease over a defined time window:
              $$\\text{Risk in Exposed } (I_e) = \\frac{\\text{Exposed Cases } (a)}{\\text{Total Exposed } (a + b)}$$
              $$\\text{Risk in Unexposed } (I_u) = \\frac{\\text{Unexposed Cases } (c)}{\\text{Total Unexposed } (c + d)}$$
            * If 18 out of 100 quarry workers develop lung disease over 10 years, their cumulative risk is **18%** ($0.18$).
            """)

            st.markdown("""
            #### 2. What is Relative Risk (RR)?
            * The **Relative Risk** divides the exposed risk by the unexposed risk:
              $$RR = \\frac{I_e}{I_u} = \\frac{a / (a + b)}{c / (c + d)}$$
            * **$RR = 1.0$ (The Null Value):** Parity. Exposure has no effect on disease risk.
            * **$RR > 1.0$ (Harmful Exposure):** Disease risk is higher in the exposed. For example, $RR = 2.5$ means exposed individuals are **2.5 times as likely** (a 150% increase in risk) to develop the condition.
            * **$RR < 1.0$ (Protective Factor / Vaccine):** Disease risk is lower in the exposed. For example, $RR = 0.35$ means a **65% reduction in risk** (Vaccine Efficacy = $1 - RR = 65\\%$).
            """)

        with col_g2:
            st.markdown("""
            #### 3. The Core Analogy: "Looking Forward vs. In the Rearview Mirror"
            * **Looking Forward (Cohort / Clinical Trial):**
              You recruit healthy individuals today, check their exposure status (e.g. occupational dust exposure vs. office work), 
              and follow them forward in time. Because you know the total denominator of exposed ($a+b$) and unexposed ($c+d$) people, 
              you can calculate true cumulative incidence. **Relative Risk is mathematically valid!**
            * **In the Rearview Mirror (Retrospective Case-Control):**
              You start by finding people who are already ill (100 Cases) and healthy people (100 Controls), then look backward 
              into their medical history to see who was exposed. Because the researcher artificially decided to recruit 100 cases and 100 controls, 
              the proportion of diseased participants ($50\\%$) is an artifact of the recruitment quota, not reality! 
              **Calculating cumulative incidence or Relative Risk from a Case-Control study is mathematically invalid; you MUST use the Odds Ratio (OR) instead!**
            """)

            st.markdown(r"""
            #### 4. Relative Risk vs. Absolute Risk Difference & NNT/NNH
            * **The Rare Disease Trap:** A treatment that lowers risk from $0.02\%$ to $0.01\%$ has a huge Relative Risk of $0.50$ (a 50% relative reduction!), but the absolute benefit is only $0.01\%$ ($1$ in $10,000$). You would need to treat 10,000 people to prevent a single event ($\text{NNT} = 10,000$).
            * **Number Needed to Treat (NNT):** For beneficial interventions ($RR < 1$, e.g. a vaccine or lifestyle change), NNT indicates how many people must receive the intervention to prevent exactly $1$ person from developing the condition. A lower NNT means a more effective, high-yield treatment.
            * **Number Needed to Harm (NNH):** For harmful risk factors ($RR > 1$, e.g. quarry dust exposure), NNH indicates how many people must be exposed to the hazard for $1$ additional person to fall ill. A lower NNH means a more hazardous, potent exposure.
            * **Golden Rule of Public Health Reporting:** Always report **both** Relative Risk ($RR$) and Absolute Risk Difference ($RD = I_e - I_u$) together to provide the complete clinical and population-level picture.
            """)

            st.markdown(r"""
            #### 5. Survey & Feedback Analytics: The "Prevalence Ratio (PR)"
            * **Customer / Patient Surveys:** When comparing survey ratings between demographic groups (e.g. patients aged $\le 18$ vs. adults aged $19+$ giving positive vs. negative feedback), you are measuring existing sentiment at a single moment in time (a **Cross-Sectional Survey**).
            * **Different Name, Identical Mathematics:** In survey research, Relative Risk is termed the **Prevalence Ratio (PR)**. The formula $\frac{a/(a+b)}{c/(c+d)}$, Woolf's log standard error, and 95% Wald confidence intervals are **100% identical**!
            * **Example:** If $80\%$ of youth are satisfied vs. $60\%$ of adults, the Prevalence Ratio is $0.80 / 0.60 = 1.33$ (*"Youth are 1.33 times as likely, or 33% more likely, to give positive feedback"*).
            """)


        st.info(r"""
        📐 **Why are Confidence Intervals calculated on the Natural Log Scale ($\ln(RR)$)?**  
        Ratios cannot fall below zero, but can extend to positive infinity. Because of this right-skewed distribution, standard errors 
        cannot be symmetrically added to the point estimate on the linear scale. Epidemiologists transform $RR$ to the natural log scale ($\ln(RR)$), 
        where the distribution is approximately normal, compute the symmetric 95% confidence interval, and then exponentiate ($\exp$) back:
        $$\text{95\% Confidence Interval (CI)} = \exp\left(\ln(RR) \pm 1.96 \times SE(\ln(RR))\right)$$
        """)


def render_relative_risk_diagnostics(evaluations: List[Dict[str, Any]]):
    """Renders tri-color prerequisite diagnostic cards."""
    st.markdown("### 🚦 Tri-Color Bias & Methodological Diagnostics")
    st.caption("Verifies study design suitability, temporal precedence, cell counts, and attrition bias before interpreting results.")

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


def render_relative_risk_dashboard(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the top summary metric cards."""
    st.markdown("### 📊 Primary Epidemiological Effect Measures")

    rr = calcs["rr"]
    rr_ll = calcs["rr_ci_lower"]
    rr_ul = calcs["rr_ci_upper"]
    rd = calcs["risk_difference"] * 100.0  # as percentage
    rd_ll = calcs["rd_ci_lower"] * 100.0
    rd_ul = calcs["rd_ci_upper"] * 100.0
    afe = calcs["attributable_fraction_exposed"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Cumulative Risk (Exposed)",
            value=f"{calcs['risk_exposed'] * 100.0:.1f}%",
            help="Proportion of exposed participants who developed the outcome: a / (a + b)"
        )
    with c2:
        st.metric(
            label="Cumulative Risk (Unexposed)",
            value=f"{calcs['risk_unexposed'] * 100.0:.1f}%",
            help="Proportion of unexposed participants who developed the outcome: c / (c + d)"
        )
    with c3:
        ci_str = f"[{rr_ll:.2f}, {rr_ul:.2f}]" if not np.isnan(rr_ll) else "N/A"
        st.metric(
            label="Relative Risk (RR)",
            value=f"{rr:.2f}" if not np.isnan(rr) else "N/A",
            delta=f"95% CI: {ci_str}",
            delta_color="off",
            help="Risk Ratio = Risk(Exposed) / Risk(Unexposed). Value of 1.0 indicates no association."
        )
    with c4:
        rd_sign = "+" if rd > 0 else ""
        st.metric(
            label="Risk Difference (RD / ARR)",
            value=f"{rd_sign}{rd:.2f}%",
            delta=f"95% CI: [{rd_ll:.2f}%, {rd_ul:.2f}%]",
            delta_color="off",
            help="Absolute Risk Difference = Risk(Exposed) - Risk(Unexposed)."
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        if calcs["effect_direction"] == "harmful":
            st.metric(
                label="Attributable Fraction in Exposed (AF_e)",
                value=f"{afe:.1f}%",
                help="Proportion of disease in the exposed group attributable to the exposure: (RR - 1) / RR"
            )
        else:
            st.metric(
                label="Relative Risk Reduction (RRR / VE)",
                value=f"{afe:.1f}%",
                help="Relative reduction in risk (Vaccine Efficacy): 1 - RR"
            )
    with c6:
        if calcs["nnh"] is not None:
            st.metric(
                label="Number Needed to Harm (NNH)",
                value=f"{calcs['nnh']:.1f}",
                help="Average number of exposed people required to cause 1 additional case of disease: 1 / RD"
            )
        elif calcs["nnt"] is not None:
            st.metric(
                label="Number Needed to Treat (NNT)",
                value=f"{calcs['nnt']:.1f}",
                help="Average number of people treated/vaccinated to prevent 1 case of disease: 1 / |RD|"
            )
        else:
            st.metric(label="NNT / NNH", value="N/A")
    with c7:
        p_val = calcs["p_value_wald"]
        p_txt = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        st.metric(
            label="Wald Test (H0: RR = 1.0)",
            value=f"p = {p_txt}",
            delta=f"z = {calcs['z_score']:.2f}",
            delta_color="off",
            help="Wald z-test statistic on the natural log scale: ln(RR) / SE(ln(RR))"
        )
    with c8:
        p_fish = calcs["p_value_fisher"]
        p_fish_txt = "< 0.0001" if p_fish < 0.0001 else f"{p_fish:.4f}"
        st.metric(
            label="Fisher's Exact Test",
            value=f"p = {p_fish_txt}",
            help="Exact hypergeometric test for the 2x2 contingency table (robust to sparse cells)."
        )


def render_relative_risk_visualizations(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the Plotly grouped bar chart, forest plot, and interactive 2x2 table."""
    st.markdown("### 📈 Visualizing Cohort Risks & Effect Magnitude")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**1. Cumulative Incidence Comparison (Exposed vs. Unexposed)**")
        exp_label, unexp_label = meta.get("exposure_levels", ("Exposed", "Unexposed"))
        risk_exp_pct = calcs["risk_exposed"] * 100.0
        risk_unexp_pct = calcs["risk_unexposed"] * 100.0

        bar_colors = ["#ef4444", "#3b82f6"] if calcs["effect_direction"] == "harmful" else ["#10b981", "#64748b"]

        fig_risk = go.Figure()
        fig_risk.add_trace(go.Bar(
            x=[exp_label, unexp_label],
            y=[risk_exp_pct, risk_unexp_pct],
            text=[f"{risk_exp_pct:.1f}%", f"{risk_unexp_pct:.1f}%"],
            textposition="auto",
            marker_color=bar_colors,
            width=0.5
        ))
        fig_risk.update_layout(
            yaxis_title="Cumulative Incidence Risk (%)",
            xaxis_title="Cohort Exposure Status",
            height=340,
            margin=dict(l=30, r=30, t=30, b=40),
            template="plotly_white",
            yaxis=dict(range=[0, max(risk_exp_pct, risk_unexp_pct) * 1.35 + 1.0])
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_chart2:
        st.markdown("**2. Relative Risk Forest Plot (Point Estimate & 95% Wald CI)**")
        rr = calcs["rr"]
        rr_ll = calcs["rr_ci_lower"]
        rr_ul = calcs["rr_ci_upper"]

        fig_forest = go.Figure()

        if not np.isnan(rr) and not np.isnan(rr_ll) and not np.isnan(rr_ul):
            # Error bar computation
            err_plus = rr_ul - rr
            err_minus = rr - rr_ll

            marker_col = "#dc2626" if rr > 1.0 else "#059669"

            fig_forest.add_trace(go.Scatter(
                x=[rr],
                y=["Relative Risk (RR)"],
                mode="markers+text",
                marker=dict(size=14, color=marker_col, symbol="square"),
                text=[f"RR = {rr:.2f} [{rr_ll:.2f}, {rr_ul:.2f}]"],
                textposition="top center",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[err_plus],
                    arrayminus=[err_minus],
                    color=marker_col,
                    thickness=3,
                    width=10
                ),
                name="Point Estimate & 95% CI"
            ))

        # Vertical line of no effect (RR = 1.0)
        fig_forest.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="#475569",
            line_width=2,
            annotation_text="Null Line (RR = 1.0: No Effect)",
            annotation_position="bottom right"
        )

        x_max = max(3.5, rr_ul * 1.25) if not np.isnan(rr_ul) else 4.0
        fig_forest.update_layout(
            xaxis_title="Relative Risk (RR)",
            yaxis_title="",
            height=340,
            margin=dict(l=30, r=30, t=30, b=40),
            template="plotly_white",
            xaxis=dict(range=[0, x_max]),
            showlegend=False
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # Interactive 2x2 Contingency Table Display
    st.markdown("**3. Contingency Table Structure (2×2 Matrix & Marginals)**")
    a, b, c, d = calcs["a"], calcs["b"], calcs["c"], calcs["d"]
    n1, n0 = calcs["n_exposed"], calcs["n_unexposed"]
    m1, m0 = a + c, b + d
    n_tot = calcs["n_total"]

    exp_label, unexp_label = meta.get("exposure_levels", ("Exposed", "Unexposed"))
    out_pos, out_neg = meta.get("outcome_levels", ("Cases (Outcome +)", "Non-Cases (Outcome -)"))

    df_2x2_display = pd.DataFrame([
        {
            "Cohort Stratum": f"🟢 {exp_label}",
            f"{out_pos} [Cases]": f"{a:,}",
            f"{out_neg} [Non-Cases]": f"{b:,}",
            "Stratum Total (n)": f"{n1:,}",
            "Cumulative Incidence (Risk)": f"{calcs['risk_exposed'] * 100.0:.2f}%"
        },
        {
            "Cohort Stratum": f"⚪ {unexp_label}",
            f"{out_pos} [Cases]": f"{c:,}",
            f"{out_neg} [Non-Cases]": f"{d:,}",
            "Stratum Total (n)": f"{n0:,}",
            "Cumulative Incidence (Risk)": f"{calcs['risk_unexposed'] * 100.0:.2f}%"
        },
        {
            "Cohort Stratum": "📊 Total Cohort",
            f"{out_pos} [Cases]": f"{m1:,} (a + c)",
            f"{out_neg} [Non-Cases]": f"{m0:,} (b + d)",
            "Stratum Total (n)": f"{n_tot:,} (N)",
            "Cumulative Incidence (Risk)": f"{(m1 / n_tot) * 100.0:.2f}% (Crude)"
        }
    ])

    st.dataframe(df_2x2_display, use_container_width=True, hide_index=True)


def render_relative_risk_excel_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 3-step Microsoft Excel Implementation Guide with cell coordinates."""
    st.markdown("### 📗 Microsoft Excel Implementation Workflow for Relative Risk")
    st.markdown("Follow the 3-step sequence: **Set up 2×2 Layout** ➔ **Calculate Incidences & Point Estimates** ➔ **Calculate Log Standard Error & 95% Wald CI**.")

    a, b, c, d = calcs["a"], calcs["b"], calcs["c"], calcs["d"]
    n1, n0 = calcs["n_exposed"], calcs["n_unexposed"]
    m1, m0 = a + c, b + d
    n_tot = calcs["n_total"]
    exp_label, unexp_label = meta.get("exposure_levels", ("Exposed", "Unexposed"))

    with st.expander("📌 Step 1: Excel 2×2 Data Layout & Cell References", expanded=False):
        st.markdown(f"""
        Structure your Excel worksheet with the 2×2 table in `Cells A1:E4` and calculation outputs in `Cells A6:B14`:
        
        * **Column A (Rows 2:4):** Exposure Group (`Row 2: Exposed`, `Row 3: Unexposed`, `Row 4: Total`)
        * **Column B (Rows 2:4):** Cases (`B2`: $a = {a}$, `B3`: $c = {c}$, `B4`: `=SUM(B2:B3)`)
        * **Column C (Rows 2:4):** Non-Cases (`C2`: $b = {b}$, `C3`: $d = {d}$, `C4`: `=SUM(C2:C3)`)
        * **Column D (Rows 2:4):** Cohort Totals (`D2`: `=SUM(B2:C2)`, `D3`: `=SUM(B3:C3)`, `D4`: `=SUM(B4:C4)`)
        * **Column E (Rows 2:4):** Cumulative Risk (`E2`: `=B2/D2`, `E3`: `=B3/D3`, `E4`: `=B4/D4`)
        """)

        preview_grid = pd.DataFrame([
            {"Excel_Row": "Row 1 (Headers)", "Col_A": "Exposure_Group", "Col_B": "Cases_Outcome_Pos", "Col_C": "NonCases_Outcome_Neg", "Col_D": "Total_Cohort_n", "Col_E": "Risk_Incidence"},
            {"Excel_Row": "Row 2 (Exposed)", "Col_A": exp_label, "Col_B": str(a), "Col_C": str(b), "Col_D": f"{n1} (=SUM(B2:C2))", "Col_E": f"{calcs['risk_exposed']:.4f} (=B2/D2)"},
            {"Excel_Row": "Row 3 (Unexposed)", "Col_A": unexp_label, "Col_B": str(c), "Col_C": str(d), "Col_D": f"{n0} (=SUM(B3:C3))", "Col_E": f"{calcs['risk_unexposed']:.4f} (=B3/D3)"},
            {"Excel_Row": "Row 4 (Totals)", "Col_A": "Total Cohort", "Col_B": f"{m1} (=SUM(B2:B3))", "Col_C": f"{m0} (=SUM(C2:C3))", "Col_D": f"{n_tot} (=SUM(D2:D3))", "Col_E": f"{(m1/n_tot):.4f} (=B4/D4)"}
        ])
        st.dataframe(preview_grid, use_container_width=True, hide_index=True)

        st.markdown("**Summary Output Cells Mapping (`Cells A6:B14`):**")
        df_summary_cells = pd.DataFrame({
            "Cell": ["B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14"],
            "Metric Name": [
                "Cumulative Risk Exposed (I_e)",
                "Cumulative Risk Unexposed (I_u)",
                "Relative Risk (RR)",
                "Absolute Risk Difference (RD)",
                "Natural Log of RR [ln(RR)]",
                "Standard Error of ln(RR) [SE]",
                "95% CI Lower Bound",
                "95% CI Upper Bound",
                "Wald Two-Sided p-value"
            ],
            "Excel Formula": [
                "=B2/D2",
                "=B3/D3",
                "=B6/B7",
                "=B6-B7",
                "=LN(B8)",
                "=SQRT((1/B2 - 1/D2) + (1/B3 - 1/D3))",
                "=EXP(B10 - 1.96 * B11)",
                "=EXP(B10 + 1.96 * B11)",
                "=2 * (1 - NORM.S.DIST(ABS(B10 / B11), TRUE))"
            ],
            "Calculated Value": [
                f"{calcs['risk_exposed']:.4f}",
                f"{calcs['risk_unexposed']:.4f}",
                f"{calcs['rr']:.3f}",
                f"{calcs['risk_difference']:.4f}",
                f"{calcs['ln_rr']:.4f}",
                f"{calcs['se_ln_rr']:.4f}",
                f"{calcs['rr_ci_lower']:.3f}",
                f"{calcs['rr_ci_upper']:.3f}",
                f"{calcs['p_value_wald']:.4e}"
            ]
        })
        st.dataframe(df_summary_cells, use_container_width=True, hide_index=True)

    with st.expander("📌 Step 2: Calculate Incidences, Relative Risk, and Risk Difference in Excel", expanded=False):
        st.markdown("""
        1. **Compute Cumulative Incidence in Exposed ($I_e$):** In cell `B6`, enter `=B2/D2`.
        2. **Compute Cumulative Incidence in Unexposed ($I_u$):** In cell `B7`, enter `=B3/D3`.
        3. **Compute Relative Risk (Risk Ratio):** In cell `B8`, enter `=B6/B7`.
        4. **Compute Absolute Risk Difference (RD):** In cell `B9`, enter `=B6-B7`.
        5. **Compute Number Needed to Treat (NNT) / Harm (NNH):** In cell `B15`, enter `=1/ABS(B9)`.
        6. **Compute Attributable Fraction in Exposed (AF_e %):** In cell `B16`, enter `=(B8 - 1) / B8 * 100`.
        """)

    with st.expander("📌 Step 3: Compute Natural Log, Standard Error, and 95% Wald CI in Excel", expanded=False):
        st.markdown("""
        Because ratios are skewed and bounded by zero, confidence intervals are computed on the natural log scale and exponentiated back:

        1. **Compute Natural Log of RR:** In cell `B10`, enter:
           ```excel
           =LN(B8)
           ```
        2. **Compute Standard Error of $\ln(RR)$:** In cell `B11`, enter Greenwood's large-sample formula:
           ```excel
           =SQRT((1/B2 - 1/D2) + (1/B3 - 1/D3))
           ```
           *(Alternative equivalent formula: `=SQRT(C2/(B2*D2) + C3/(B3*D3))`)*
        3. **Compute 95% CI Lower Bound:** In cell `B12`, enter:
           ```excel
           =EXP(B10 - 1.96 * B11)
           ```
        4. **Compute 95% CI Upper Bound:** In cell `B13`, enter:
           ```excel
           =EXP(B10 + 1.96 * B11)
           ```
        5. **Compute Wald Two-Sided p-value:** In cell `B14`, enter:
           ```excel
           =2 * (1 - NORM.S.DIST(ABS(B10 / B11), TRUE))
           ```
        """)


def render_relative_risk_python_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 4-step Python Implementation Workflow with docstrings and conditional prints."""
    st.markdown("### 🐍 Python Implementation Workflow for Relative Risk")
    st.markdown("This guide follows a strict 4-step sequence: **Evaluate Prerequisites** ➔ **Calculate Cumulative Incidences & Point Estimates** ➔ **Calculate Standard Error & Wald Confidence Intervals** ➔ **Consolidated Executable Pipeline Script**.")

    a, b, c, d = calcs["a"], calcs["b"], calcs["c"], calcs["d"]

    with st.expander("📌 Step 1: Evaluate Prerequisites & Study Design in Python", expanded=False):
        st.code('''import numpy as np

def check_relative_risk_prerequisites(
    study_design: str,
    a: int,
    b: int,
    c: int,
    d: int,
    loss_to_follow_up_pct: float = 0.0
) -> dict:
    """Verifies epidemiological criteria for Relative Risk from a 2x2 table.

    Args:
        study_design (str): Must be 'prospective_cohort', 'retrospective_cohort', or 'rct'.
            If 'case_control', Relative Risk is invalid and Odds Ratio must be used.
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        loss_to_follow_up_pct (float, optional): Percentage lost to follow-up. Defaults to 0.0.

    Returns:
        dict: Diagnostic pass/warning/fail flags and recommended next steps.
    """
    valid_designs = {"prospective_cohort", "retrospective_cohort", "rct"}
    if study_design.lower() not in valid_designs:
        return {
            "status": "FAIL",
            "message": "[INAPPROPRIATE - FAIL] Relative Risk is invalid for Case-Control studies! Use Odds Ratio instead."
        }
    
    min_cell = min(a, b, c, d)
    if min_cell == 0:
        return {
            "status": "WARNING",
            "message": "[WARNING] One or more zero cells detected. Haldane-Anscombe (+0.5) correction required."
        }
    elif min_cell < 5:
        return {
            "status": "WARNING",
            "message": "[WARNING] Cell counts < 5. Wald CI may suffer slight skew; cross-reference with Fisher exact test."
        }
        
    return {
        "status": "PASS",
        "message": "[ASSURANCE - PASS] Study design and sample criteria fully satisfied for Relative Risk."
    }
''', language="python")

    with st.expander("📌 Step 2: Calculate Cumulative Incidences & Relative Risk Point Estimate in Python", expanded=False):
        st.code('''def calculate_point_estimates(a: int, b: int, c: int, d: int) -> dict:
    """Calculates cumulative incidence risks, Relative Risk, and Risk Difference.

    Args:
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.

    Returns:
        dict: Dictionary containing:
            - 'risk_exposed': a / (a + b)
            - 'risk_unexposed': c / (c + d)
            - 'rr': Relative Risk point estimate
            - 'risk_difference': Absolute Risk Difference (RD)
            - 'attributable_fraction_exposed': AF_e %
            - 'nnt_or_nnh': Number Needed to Treat/Harm
    """
    n1 = a + b
    n0 = c + d
    
    risk_exp = a / n1 if n1 > 0 else 0.0
    risk_unexp = c / n0 if n0 > 0 else 0.0
    rr = risk_exp / risk_unexp if risk_unexp > 0 else np.nan
    rd = risk_exp - risk_unexp
    
    afe = ((rr - 1.0) / rr) * 100.0 if rr > 1.0 else (1.0 - rr) * 100.0
    nnt_nnh = (1.0 / abs(rd)) if abs(rd) > 1e-6 else np.nan
    
    return {
        "risk_exposed": risk_exp,
        "risk_unexposed": risk_unexp,
        "rr": rr,
        "risk_difference": rd,
        "attributable_fraction_exposed": afe,
        "nnt_or_nnh": nnt_nnh
    }
''', language="python")

    with st.expander("📌 Step 3: Compute Log-scale Standard Error & Wald 95% CI in Python", expanded=False):
        st.code('''from scipy import stats

def compute_rr_confidence_interval(
    a: int,
    b: int,
    c: int,
    d: int,
    alpha: float = 0.05
) -> dict:
    """Computes log-scale Standard Error, Wald 95% CI, and Wald hypothesis test for Relative Risk.

    Args:
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        alpha (float, optional): Significance level. Defaults to 0.05 (95% CI).

    Returns:
        dict: ln_rr, se_ln_rr, ci_lower, ci_upper, z_score, p_value_wald.
    """
    n1 = a + b
    n0 = c + d
    
    risk_exp = a / n1
    risk_unexp = c / n0
    rr = risk_exp / risk_unexp
    
    ln_rr = np.log(rr)
    # SE on natural log scale
    se_ln_rr = np.sqrt((1.0 / a) - (1.0 / n1) + (1.0 / c) - (1.0 / n0))
    
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_lower = np.exp(ln_rr - z_crit * se_ln_rr)
    ci_upper = np.exp(ln_rr + z_crit * se_ln_rr)
    
    z_stat = ln_rr / se_ln_rr
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    
    return {
        "rr": rr,
        "ln_rr": ln_rr,
        "se_ln_rr": se_ln_rr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "z_score": z_stat,
        "p_value_wald": p_value
    }
''', language="python")

    with st.expander("📌 Step 4: Consolidated Executable Pipeline Script", expanded=False):
        st.code(f'''"""Consolidated Epidemiological Analysis Pipeline: Relative Risk (Risk Ratio).

Executes end-to-end verification, point estimation, log-scale variance estimation,
and clinical effect interpretation for a 2x2 contingency table.
"""

import numpy as np
from scipy import stats

def run_relative_risk_pipeline(
    a: int,
    b: int,
    c: int,
    d: int,
    study_design: str = "prospective_cohort",
    loss_to_follow_up_pct: float = 4.0,
    alpha: float = 0.05
) -> dict:
    """Full epidemiological analysis pipeline for Relative Risk.

    Args:
        a (int): Exposed cases (outcome developed).
        b (int): Exposed non-cases (outcome free).
        c (int): Unexposed cases (outcome developed).
        d (int): Unexposed non-cases (outcome free).
        study_design (str, optional): Epidemiological design architecture. Defaults to 'prospective_cohort'.
        loss_to_follow_up_pct (float, optional): Attrition percentage. Defaults to 4.0.
        alpha (float, optional): Type I error rate for confidence intervals. Defaults to 0.05.

    Returns:
        dict: Complete epidemiological findings including risks, RR, RD, CIs, and p-values.
    """
    print("=" * 70)
    print("  EPIDEMIOLOGICAL RELATIVE RISK (RISK RATIO) PIPELINE REPORT")
    print("=" * 70)

    # 1. Study Design Verification
    if study_design.lower() == "case_control":
        print("[INAPPROPRIATE - FAIL] Study is a retrospective Case-Control design!")
        print("  Cumulative incidence cannot be measured. Compute Odds Ratio (OR) instead.")
        return {{"status": "FAIL", "reason": "Invalid design for Relative Risk"}}
    else:
        print(f"[ASSURANCE - PASS] Study design '{{study_design}}' satisfies forward-looking risk criteria.")

    # 2. Check Sparsity & Apply Haldane-Anscombe correction if needed
    applied_cc = False
    if min(a, b, c, d) == 0:
        print("[WARNING] Zero cell detected! Applying Haldane-Anscombe (+0.5) continuity correction.")
        a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        applied_cc = True
    else:
        a_c, b_c, c_c, d_c = float(a), float(b), float(c), float(d)
        print(f"[ASSURANCE - PASS] All 2x2 cells >= 5 (minimum cell = {{min(a, b, c, d)}}). Large-sample Wald CI robust.")

    n1 = a_c + b_c
    n0 = c_c + d_c

    # 3. Cumulative Incidences
    risk_exp = a_c / n1
    risk_unexp = c_c / n0
    rr = risk_exp / risk_unexp
    rd = risk_exp - risk_unexp

    # 4. Log Scale Variance & 95% Wald CI
    ln_rr = np.log(rr)
    se_ln_rr = np.sqrt((1.0 / a_c) - (1.0 / n1) + (1.0 / c_c) - (1.0 / n0))
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_lower = np.exp(ln_rr - z_crit * se_ln_rr)
    ci_upper = np.exp(ln_rr + z_crit * se_ln_rr)

    z_stat = ln_rr / se_ln_rr
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    # 5. Clinical Translation
    if rr > 1.0:
        afe = ((rr - 1.0) / rr) * 100.0
        nnh = 1.0 / rd if rd > 0 else np.nan
        print(f"\\n--- EFFECT SUMMARY: HARMFUL EXPOSURE ---")
        print(f"  Risk in Exposed:   {{risk_exp * 100.0:.2f}}%")
        print(f"  Risk in Unexposed: {{risk_unexp * 100.0:.2f}}%")
        print(f"  Relative Risk:     {{rr:.3f}} (95% CI: [{{ci_lower:.3f}}, {{ci_upper:.3f}}])")
        print(f"  Risk Difference:   +{{rd * 100.0:.2f}}% (NNH = {{nnh:.1f}})")
        print(f"  Attributable Frac: {{afe:.1f}}% of cases in exposed attributable to exposure.")
    else:
        rrr = (1.0 - rr) * 100.0
        nnt = 1.0 / abs(rd) if abs(rd) > 0 else np.nan
        print(f"\\n--- EFFECT SUMMARY: PROTECTIVE FACTOR / INTERVENTION ---")
        print(f"  Risk in Exposed:   {{risk_exp * 100.0:.2f}}%")
        print(f"  Risk in Unexposed: {{risk_unexp * 100.0:.2f}}%")
        print(f"  Relative Risk:     {{rr:.3f}} (95% CI: [{{ci_lower:.3f}}, {{ci_upper:.3f}}])")
        print(f"  Relative Risk Red: {{rrr:.1f}}% protection (Vaccine Efficacy)")
        print(f"  Risk Difference:   -{{abs(rd) * 100.0:.2f}}% (NNT = {{nnt:.1f}})")

    print(f"\\n--- HYPOTHESIS TEST ---")
    print(f"  Wald z-statistic:  {{z_stat:.3f}} (p = {{p_value:.4e}})")
    if p_value < 0.05:
        print("  Conclusion:        Statistically significant association at alpha = 0.05.")
    else:
        print("  Conclusion:        Fail to reject null hypothesis (no significant association).")

    return {{
        "rr": rr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "risk_difference": rd,
        "p_value": p_value
    }}

# Execute with study data ({meta['title']}):
if __name__ == "__main__":
    results = run_relative_risk_pipeline(
        a={a}, b={b}, c={c}, d={d},
        study_design="prospective_cohort",
        loss_to_follow_up_pct={meta.get('loss_to_follow_up_pct', 3.0)}
    )
''', language="python")


def render_relative_risk_download_tab(
    df_contingency: pd.DataFrame,
    df_microdata: pd.DataFrame,
    meta: Dict[str, Any]
):
    """Renders the data download tab with individual microdata and 2x2 contingency table CSVs."""
    st.markdown("### 📥 Download Sample Cohort Datasets")
    st.markdown(f"""
    Download the sample data associated with the active scenario: **{meta['title']}**.
    You can use these datasets to practice calculating Relative Risk, log standard errors, and Wald confidence intervals 
    in Microsoft Excel or with Python scripts.
    """)

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.markdown("#### 1. Individual Participant Microdata (CSV)")
        st.markdown(f"Contains **{len(df_microdata):,} participant records** with columns: `Participant_ID`, `Exposure_Status`, `Exposure_Binary`, `Outcome_Status`, `Outcome_Binary`, `Follow_Up_Years`, `Age`, `Sex`.")
        csv_micro = df_microdata.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Individual Cohort Microdata (CSV)",
            data=csv_micro,
            file_name=f"{meta['scenario_key']}_cohort_microdata.csv",
            mime="text/csv",
            key="btn_dl_rr_microdata"
        )
        st.dataframe(df_microdata.head(8), height=220, use_container_width=True, hide_index=True)

    with col_dl2:
        st.markdown("#### 2. Aggregated 2×2 Contingency Table (CSV)")
        st.markdown("Contains the summarized contingency matrix matching the exact Excel tutorial cell coordinates (`A1:E4`).")
        csv_cont = df_contingency.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Aggregated 2×2 Table (CSV)",
            data=csv_cont,
            file_name=f"{meta['scenario_key']}_2x2_contingency_table.csv",
            mime="text/csv",
            key="btn_dl_rr_contingency"
        )
        st.dataframe(df_contingency, height=220, use_container_width=True, hide_index=True)


def render_relative_risk_module(active_method_key: str, method_info: Dict[str, Any]):
    """Renders the comprehensive Relative Risk (Risk Ratio) module."""
    # 1. Plain English Guide
    render_relative_risk_plain_english_guide()

    # 2. Sidebar Controls
    st.sidebar.header("🧪 Cohort Study Settings")
    st.sidebar.markdown("**Study Scenarios & Contingency Parameters:**")

    rr_scenario_choice = st.sidebar.selectbox(
        "Study Scenario:",
        options=["occupational_dust", "lifestyle_intervention", "patient_feedback_survey", "influenza_vaccine", "custom"],
        format_func=lambda s: {
            "occupational_dust": "🏭 Regional Occupational Dust Cohort (Harmful Exposure, RR > 1)",
            "lifestyle_intervention": "🏃 Physical Activity Community Cohort (Protective Intervention, RR < 1)",
            "patient_feedback_survey": "📋 NHS Patient Experience Survey (Prevalence Ratio in Survey Data)",
            "influenza_vaccine": "💉 Elderly Influenza Vaccine RCT (Vaccine Efficacy, RR < 1)",
            "custom": "⚙️ Custom 2×2 Table Input (Enter Your Own Counts)"
        }[s],
        index=0,
        help="Select a realistic public health scenario or enter custom 2x2 contingency table counts."
    )
    st.sidebar.caption("⚠️ **Data Notice:** Scenarios, populations, and event counts are made-up examples synthetically simulated for educational and illustrative purposes only.")

    if rr_scenario_choice == "custom":
        st.sidebar.subheader("Custom 2×2 Parameters")
        custom_study_design = st.sidebar.selectbox(
            "Study Design Architecture:",
            options=["prospective_cohort", "retrospective_cohort", "rct", "case_control", "cross_sectional"],
            format_func=lambda d: {
                "prospective_cohort": "Prospective Cohort (Forward-looking)",
                "retrospective_cohort": "Retrospective Cohort (Historical records)",
                "rct": "Randomized Controlled Trial (RCT)",
                "case_control": "Case-Control (Outcome-first, Retrospective)",
                "cross_sectional": "Cross-Sectional (Prevalence snapshot)"
            }[d],
            index=0
        )
        custom_loss_pct = st.sidebar.slider(
            "Loss to Follow-up (%):",
            min_value=0.0,
            max_value=30.0,
            value=3.5,
            step=0.5,
            help=(
                "What is Loss to Follow-up? In cohort studies and clinical trials, it is the percentage of enrolled participants "
                "who dropped out, moved away, or could not be reached before the observation window ended. "
                "Effect on scenario: Directly drives the 'Follow-Up Completeness (Attrition Bias)' diagnostic card below "
                "(< 5% Pass, 5–20% Caution, > 20% Fail). If dropouts differ between exposed and unexposed, Attrition Bias distorts the Relative Risk."
            )
        )
        st.sidebar.caption(
            "📉 **What is Loss to Follow-up?** Participants who drop out before study completion. "
            "Moving this slider evaluates **Attrition Bias** in the diagnostics below: "
            "🟢 `<5%` Pass | 🟡 `5–20%` Caution | 🔴 `>20%` Fail."
        )

        col_sb1, col_sb2 = st.sidebar.columns(2)
        with col_sb1:
            a_val = st.number_input("Exposed Cases (a):", min_value=0, max_value=100000, value=75, step=1)
            b_val = st.number_input("Exposed Non-Cases (b):", min_value=0, max_value=100000, value=425, step=5)
        with col_sb2:
            c_val = st.number_input("Unexposed Cases (c):", min_value=0, max_value=100000, value=35, step=1)
            d_val = st.number_input("Unexposed Non-Cases (d):", min_value=0, max_value=100000, value=465, step=5)

        meta = {
            "title": "Custom User-Defined Cohort Investigation",
            "domain": "Epidemiological Hypothesis Testing",
            "description": "User-configured 2x2 contingency cohort evaluation.",
            "exposure_name": "Exposure Factor",
            "exposure_levels": ("Exposed", "Unexposed"),
            "outcome_name": "Target Outcome",
            "outcome_levels": ("Cases (Outcome +)", "Non-Cases (Outcome -)"),
            "expected_effect": "User Defined",
            "clinical_takeaway": "Custom parameters evaluated based on user-entered contingency frequencies.",
            "scenario_key": "custom_cohort",
            "loss_to_follow_up_pct": custom_loss_pct
        }

        # Build contingency and microdata
        df_contingency = pd.DataFrame([
            {
                "Exposure_Group": "Exposed",
                "Cases_Outcome_Present": a_val,
                "Non_Cases_Outcome_Absent": b_val,
                "Total_Cohort_Size": a_val + b_val,
                "Cumulative_Incidence_Risk": a_val / (a_val + b_val) if (a_val + b_val) > 0 else 0.0
            },
            {
                "Exposure_Group": "Unexposed",
                "Cases_Outcome_Present": c_val,
                "Non_Cases_Outcome_Absent": d_val,
                "Total_Cohort_Size": c_val + d_val,
                "Cumulative_Incidence_Risk": c_val / (c_val + d_val) if (c_val + d_val) > 0 else 0.0
            }
        ])
        micro_list = []
        for i in range(min(a_val, 100)):
            micro_list.append({"Participant_ID": f"EXP_{i+1:04d}", "Exposure_Status": "Exposed", "Exposure_Binary": 1, "Outcome_Status": "Case", "Outcome_Binary": 1, "Follow_Up_Years": 5.0, "Age": 55, "Sex": "Male"})
        for i in range(min(b_val, 100)):
            micro_list.append({"Participant_ID": f"EXP_{a_val+i+1:04d}", "Exposure_Status": "Exposed", "Exposure_Binary": 1, "Outcome_Status": "Non-Case", "Outcome_Binary": 0, "Follow_Up_Years": 5.0, "Age": 54, "Sex": "Female"})
        for i in range(min(c_val, 100)):
            micro_list.append({"Participant_ID": f"UNEXP_{i+1:04d}", "Exposure_Status": "Unexposed", "Exposure_Binary": 0, "Outcome_Status": "Case", "Outcome_Binary": 1, "Follow_Up_Years": 5.0, "Age": 56, "Sex": "Male"})
        for i in range(min(d_val, 100)):
            micro_list.append({"Participant_ID": f"UNEXP_{c_val+i+1:04d}", "Exposure_Status": "Unexposed", "Exposure_Binary": 0, "Outcome_Status": "Non-Case", "Outcome_Binary": 0, "Follow_Up_Years": 5.0, "Age": 55, "Sex": "Female"})
        df_microdata = pd.DataFrame(micro_list)

        calcs = calculate_relative_risk(a_val, b_val, c_val, d_val)
        evaluations = evaluate_relative_risk_prerequisites(
            study_design=custom_study_design,
            a=a_val, b=b_val, c=c_val, d=d_val,
            loss_to_follow_up_pct=custom_loss_pct,
            is_temporal_sequence_verified=(custom_study_design != "cross_sectional")
        )
    else:
        df_contingency, df_microdata, meta = generate_relative_risk_demo_data(scenario=rr_scenario_choice, seed=42)
        calcs = meta["calcs"]

        if rr_scenario_choice == "influenza_vaccine":
            active_design = "rct"
        elif rr_scenario_choice == "patient_feedback_survey":
            active_design = "cross_sectional"
        else:
            active_design = "prospective_cohort"

        evaluations = evaluate_relative_risk_prerequisites(
            study_design=active_design,
            a=meta["a"], b=meta["b"], c=meta["c"], d=meta["d"],
            loss_to_follow_up_pct=meta.get("loss_to_follow_up_pct", 0.0),
            is_temporal_sequence_verified=(active_design != "cross_sectional")
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
            ⚠️ <strong>Disclaimer:</strong> This is a completely made-up example scenario with synthetic data simulated for educational and illustrative purposes only. It does not represent actual medical records or real occupational exposure data from employers or NHS trusts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if meta.get("scenario_key") == "patient_feedback_survey":
        st.info("""
        💡 **Methodological Note on Naming Conventions in Survey Research:**  
        In customer, patient, and cross-sectional survey research, Relative Risk is universally referred to as the **Prevalence Ratio (PR)**. 
        This is because you are comparing the **prevalence of existing sentiment** at a single snapshot in time rather than tracking newly developed incident health events over years. 
        **The underlying mathematical formulas, Woolf standard errors, and 95% Wald confidence intervals are 100% identical to Relative Risk!**
        """)


    # 1. Tri-Color Diagnostics
    render_relative_risk_diagnostics(evaluations)

    # 2. Metric Dashboard
    render_relative_risk_dashboard(calcs, meta)

    # 3. Visualizations & 2x2 Grid
    render_relative_risk_visualizations(calcs, meta)

    # 4. Methodology Tabs
    st.markdown("---")
    tab_rr_rationale, tab_rr_excel, tab_rr_code, tab_rr_download = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Excel Implementation Guide",
        "🐍 Python Code Implementation",
        "📥 Download Sample Data"
    ])

    with tab_rr_rationale:
        st.subheader(f"Why Relative Risk (Risk Ratio)? ({method_info['name']})")
        st.markdown("""
        When investigating whether an exposure (e.g. occupational dust, lifestyle interventions, pharmacological treatments) 
        causes or prevents health events, public health scientists follow participants across time to evaluate risk.
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
        st.markdown("### 🔍 In-Depth Epidemiological Framework: Risk Ratio vs. Absolute Risk vs. Odds Ratio")
        st.markdown(r"""
        | Epidemiological Dimension | Relative Risk (Risk Ratio, RR) | Absolute Risk Difference (RD / ARR) | Odds Ratio (OR) |
        | :--- | :--- | :--- | :--- |
        | **Mathematical Formula** | $I_e / I_u = [a / (a+b)] / [c / (c+d)]$ | $I_e - I_u$ | $(a / b) / (c / d) = (ad) / (bc)$ |
        | **Primary Study Design** | Prospective Cohort, RCT | Prospective Cohort, RCT | Retrospective Case-Control, Cross-Sectional |
        | **Core Question Asked** | How many times higher or lower is the risk of illness in the exposed group? | How many additional (or fewer) cases occur per 100 people due to the exposure? | How much higher are the odds of past exposure among people who already have the disease? |
        | **Clinical & Public Health Meaning** | **Biological Strength of Association:** Shows whether a hazard triggers disease, or a drug prevents it. Does *not* show overall population burden. | **Real-World Impact & Clinical Workload:** Directly quantifies how many people are affected. Expressed as **Number Needed to Harm (NNH)** for hazards, or **Number Needed to Treat (NNT)** for treatments (see detailed guide below). | **Relative Exposure Odds:** Useful when disease is already present; only approximates Relative Risk if the disease is rare (< 5–10% prevalence). |
        | **Baseline Null Value** | $1.00$ (No difference in risk) | $0.00$ (Zero difference in risk) | $1.00$ (No difference in odds) |
        """)

        st.markdown("#### 📖 Key Terms & Plain English Explanations (Zero Prior Knowledge Assumed)")
        st.markdown(r"""
        When reading public health research, clinicians and epidemiologists distinguish between **relative** changes (Relative Risk) and **absolute** numbers (Risk Difference, NNT, and NNH). Here is what each term means and how to interpret it without prior statistical training:

        1. **Absolute Risk Difference (RD) / Absolute Risk Reduction (ARR):**
           * **Definition:** The direct subtraction between the disease risk in the exposed group ($I_e$) and the risk in the unexposed group ($I_u$):
             $$\text{Risk Difference } (RD) = I_e - I_u = \frac{a}{a+b} - \frac{c}{c+d}$$
           * **Plain English:** While Relative Risk tells you how many *times* higher the risk is, Risk Difference tells you how many *actual extra people* get sick per 100 individuals.
           * **Example:** In our regional quarry dust scenario, $18.0\%$ of heavy dust workers develop COPD compared to $7.1\%$ of unexposed workers. The Risk Difference is $+10.9\%$ ($0.109$). That means dust exposure causes **10.9 extra cases of COPD for every 100 workers**.
           * **Beneficial Interventions:** When a treatment or vaccine *reduces* risk (e.g. dropping risk from $11\%$ to $5\%$), this subtraction is called **Absolute Risk Reduction (ARR)** ($11\% - 5\% = 6\%$).

        2. **Number Needed to Harm (NNH) — For Harmful Exposures ($RR > 1$):**
           * **Definition:** The number of people who must be exposed to a hazard (e.g. quarry dust, smoking, chemical fumes) for **exactly one additional person** to develop the disease who would otherwise have stayed healthy.
           * **Formula:**
             $$\text{NNH} = \frac{1}{\text{Risk Difference}} = \frac{1}{I_e - I_u}$$
           * **Plain English Example:** In our quarry dust cohort ($RD = +10.9\%$ or $0.109$):
             $$\text{NNH} = \frac{1}{0.109} \approx 9.2$$
             This means: for every **9 to 10 workers** exposed to heavy respirable dust over 10 years, **1 additional worker** develops COPD who would not have developed it without dust exposure.
           * **How to interpret:** A **lower NNH indicates a more dangerous hazard**, because fewer exposed people are needed to cause an extra case of illness!

        3. **Number Needed to Treat (NNT) — For Beneficial Treatments & Vaccines ($RR < 1$):**
           * **Definition:** The number of patients who must receive a preventative treatment, vaccine, or lifestyle programme to prevent **exactly one bad health outcome** (e.g. preventing 1 heart attack, stroke, or severe infection).
           * **Formula:**
             $$\text{NNT} = \frac{1}{|\text{Risk Difference}|} = \frac{1}{\text{Absolute Risk Reduction}}$$
           * **Plain English Example:** In our All Meadows walking initiative, regular exercise reduces 5-year heart disease risk from $11.0\%$ down to $5.0\%$ (an Absolute Risk Reduction of $6.0\%$ or $0.060$):
             $$\text{NNT} = \frac{1}{0.060} \approx 16.7 \quad (\approx 17 \text{ people})$$
             This means a GP practice or public health team needs to enrol **17 patients** in the walking programme to prevent **1 heart attack**.
           * **How to interpret:** A **lower NNT indicates a more potent, high-yield treatment**, because fewer patients need to be treated to save one person from disease.

        4. **Why do clinicians need both RR and NNT/NNH? (The "Rare Disease Trap"):**
           * Imagine a rare condition that affects only **1 in 100,000** people ($0.001\%$).
           * A new drug cuts that risk in half to **1 in 200,000** ($0.0005\%$). The Relative Risk is $0.50$ (a massive $50\%$ relative risk reduction!). In a headline, "Drug cuts risk by 50%" sounds astounding.
           * But the **Absolute Risk Reduction** is tiny: $0.001\% - 0.0005\% = 0.0005\%$ ($1$ in $200,000$).
           * To calculate the **Number Needed to Treat (NNT)**:
             $$\text{NNT} = \frac{1}{0.000005} = 200,000 \text{ patients}$$
           * A healthcare system would need to treat and pay for **200,000 people** to prevent a single event!
           * **Takeaway:** Relative Risk tells you the *biological mechanism*, while Risk Difference and NNT/NNH tell you the *real-world clinical workload and public health cost*. Always report and interpret both together!

        5. **Loss to Follow-up & Attrition Bias:**
           * **What is Loss to Follow-up?** In longitudinal cohort studies and clinical trials, participants are tracked forward over months or years. During this time, some individuals inevitably move house, change phone numbers, withdraw consent, or become untraceable before the study's final evaluation. The proportion of initially enrolled participants who cannot be assessed at the end is the **Loss to Follow-up**.
           * **Why is it dangerous? (Attrition Bias):** If dropouts happen randomly and equally between groups, it simply reduces sample size. But if participants drop out *because* of their exposure or early symptoms (for example, if the sickest quarry workers quit their jobs early and leave the area), the remaining exposed group will look artificially healthy! This systematic distortion is called **Attrition Bias** (a form of Selection Bias).
           * **Standard Epidemiological Thresholds (Tested by the Custom Slider):**
             * **< 5% Loss:** `✅ PASS` — Minimal loss. Highly reliable; very unlikely to bias the Relative Risk.
             * **5% to 20% Loss:** `🟡 CAUTION` — Moderate loss. Selection bias is possible. Epidemiologists recommend conducting *sensitivity analysis* (worst-case vs. best-case imputation for missing participants).
             * **> 20% Loss:** `❌ INVALID` — Severe attrition. The remaining participants represent a non-random "survivor" cohort, making cumulative incidence and Relative Risk conclusions methodologically untrustworthy.
        """)

    with tab_rr_excel:
        render_relative_risk_excel_guide(calcs, meta)

    with tab_rr_code:
        render_relative_risk_python_guide(calcs, meta)

    with tab_rr_download:
        render_relative_risk_download_tab(df_contingency, df_microdata, meta)

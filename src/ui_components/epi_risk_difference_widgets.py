"""Streamlit UI Components & Visualizations for Risk Difference (RD), Attributable Risk (AR), and NNT/NNH.

Provides:
1. render_risk_difference_plain_english_guide: Zero-prior-knowledge walkthrough with analogies.
2. render_risk_difference_diagnostics: Tri-color evaluation cards (PASS, WARNING, FAIL).
3. render_risk_difference_dashboard: Metric cards (I_e, I_u, RD/ARR, NNT/NNH, Wald/Fisher p-values).
4. render_risk_difference_visualizations: Absolute risk comparison, Forest Plot with 95% CIs, NNT reciprocal curve, and 2x2 grid.
5. render_risk_difference_excel_guide: Strict 3-step sequence with cell mapping and explicit formulas.
6. render_risk_difference_python_guide: Strict 4-step sequence with exhaustive docstrings and conditional prints.
7. render_risk_difference_download_tab: Live CSV download for both individual microdata and 2x2 table.
8. render_risk_difference_module: Top-level controller for Risk Difference and NNT/NNH.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.risk_difference import (
    calculate_risk_difference,
    evaluate_risk_difference_prerequisites,
    generate_risk_difference_demo_data
)


def render_risk_difference_plain_english_guide():
    """Renders the zero-prior-knowledge pedagogical guide for Risk Difference and NNT/NNH."""
    with st.expander("💡 Plain English Guide: Risk Difference (RD), Attributable Risk & NNT / NNH", expanded=False):
        st.markdown("""
        While **Relative Risk** and the **Odds Ratio** describe the *biological strength* of an association, 
        **Risk Difference (RD)** and **Number Needed to Treat (NNT) / Harm (NNH)** answer the most crucial clinical, managerial, 
        and financial question: **"How many actual people get sick or get saved per 100 individuals in the real world?"**
        """)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. What is the Risk Difference (RD)?
            * **Risk Difference (also called Absolute Risk Reduction, ARR, or Attributable Risk, AR):**
              The direct subtraction between the disease risk in the exposed group ($I_e$) and the unexposed group ($I_u$):
              $$RD = I_e - I_u = \frac{a}{a + b} - \frac{c}{c + d}$$
            * **$RD = 0.00$ (The Null Parity Benchmark):** Zero difference in absolute disease risk between groups.
            * **$RD > 0$ (Harmful Exposure):** For example, if $RD = +0.10$ ($+10\%$), the hazard causes **10 additional cases of disease for every 100 exposed people**.
            * **$RD < 0$ (Beneficial Treatment):** For example, if a drug drops risk from $9\%$ to $4\%$, the Risk Difference is $-5\%$. In clinical medicine, we drop the minus sign and call it the **Absolute Risk Reduction (ARR = 5%)**.
            """)

            st.markdown(r"""
            #### 2. What is Number Needed to Treat (NNT)?
            * For beneficial treatments ($RD < 0$), **NNT** indicates how many patients must receive the therapy to prevent **exactly 1 bad health event** (such as 1 heart attack, stroke, or death):
              $$\text{NNT} = \frac{1}{|\text{Risk Difference}|} = \frac{1}{\text{Absolute Risk Reduction (ARR)}}$$
            * *Example:* If a cholesterol-lowering statin reduces 5-year cardiac events by $5\%$ ($0.05$):
              $$\text{NNT} = \frac{1}{0.05} = \mathbf{20 \text{ patients}}$$
              A GP practice must treat **20 patients over 5 years** to prevent 1 patient from suffering a heart attack.
            * **Rule:** A **lower NNT** indicates a more potent, high-yield treatment.
            """)

        with col_g2:
            st.markdown(r"""
            #### 3. What is Number Needed to Harm (NNH)?
            * For harmful hazards ($RD > 0$), **NNH** indicates how many people must be exposed to the hazard for **exactly 1 additional person** to develop the illness:
              $$\text{NNH} = \frac{1}{\text{Risk Difference}} = \frac{1}{I_e - I_u}$$
            * *Example:* If heavy tool vibration increases hand-arm vibration syndrome (HAVS) by $18\%$ ($0.18$):
              $$\text{NNH} = \frac{1}{0.18} \approx \mathbf{5.6 \text{ workers}}$$
              For every **5 to 6 workers** exposed to vibration over 8 years, **1 additional worker** develops permanent nerve/vascular damage.
            * **Rule:** A **lower NNH** indicates a more dangerous, potent hazard.
            """)

            st.markdown(r"""
            #### 4. The Core Analogy: "The Clinical Workload & Budget Counter"
            * Imagine a hospital Chief Executive or Public Health Director deciding where to allocate funding:
            * **Relative Risk** tells you the *biological mechanism* (e.g., *"This drug cuts risk in half!"*).
            * **Risk Difference and NNT** tell you the *real-world workload and cost* (e.g., *"Do we need to buy 20 prescriptions to save 1 life, or 20,000 prescriptions?"*).
            * **The Rare Disease Trap:** If a rare illness affects 1 in 100,000 people ($0.001\%$), a drug cutting risk to 1 in 200,000 has a massive Relative Risk of $0.50$ (a 50% relative reduction!). But the Absolute Risk Reduction is just $0.0005\%$ ($1$ in $200,000$). The $\text{NNT} = 200,000$! You would have to treat 200,000 people to prevent a single event.
            """)

        st.markdown(r"""
        #### 5. Survey & Feedback Analytics: The "Percentage Point Difference"
        * **Customer & Patient Experience:** When comparing feedback between demographics (e.g. young patients aged $\le 18$ vs adults aged $19+$), Risk Difference is universally referred to as the **Percentage Point Difference** (or Prevalence Difference).
        * *Example:* If $80\%$ of youth give positive feedback vs. $60\%$ of adults, the difference is $+20.0$ percentage points ($80\% - 60\% = +20\%$).
        """)

        st.info(r"""
        📐 **Why are Confidence Intervals for Risk Difference Linear and Symmetric?**  
        Unlike Relative Risk ($RR$) and the Odds Ratio ($OR$) which are multiplicative ratios bounded by zero, the Risk Difference is an **additive subtraction** ($I_e - I_u$). 
        Its distribution is centered around $0.00$ and is approximately normal on the standard linear scale! 
        Therefore, confidence intervals are computed directly without natural log transformations:
        $$\text{95\% Confidence Interval (CI)} = RD \pm 1.96 \times SE(RD)$$
        where:
        $$SE(RD) = \sqrt{\frac{I_e(1 - I_e)}{n_1} + \frac{I_u(1 - I_u)}{n_0}}$$
        """)


def render_risk_difference_diagnostics(evaluations: List[Dict[str, Any]]):
    """Renders tri-color prerequisite diagnostic cards for Risk Difference."""
    st.markdown("### 🚦 Tri-Color Bias & Methodological Diagnostics")
    st.caption("Verifies study design suitability, baseline risk dependency, cell counts, and attrition bias before interpreting NNT and RD.")

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


def render_risk_difference_dashboard(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders summary metric cards for Risk Difference and NNT/NNH."""
    st.markdown("### 📊 Primary Epidemiological Effect Measures")

    rd_pct = calcs["risk_difference"] * 100.0
    rd_ll = calcs["rd_ci_lower"] * 100.0
    rd_ul = calcs["rd_ci_upper"] * 100.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Cumulative Risk (Exposed / Treated)",
            value=f"{calcs['risk_exposed'] * 100.0:.1f}%",
            help="Proportion of exposed or treated participants who developed the outcome: a / (a + b)"
        )
    with c2:
        st.metric(
            label="Cumulative Risk (Unexposed / Control)",
            value=f"{calcs['risk_unexposed'] * 100.0:.1f}%",
            help="Proportion of unexposed or control participants who developed the outcome: c / (c + d)"
        )
    with c3:
        rd_sign = "+" if rd_pct > 0 else ""
        st.metric(
            label="Risk Difference (RD / ARR)",
            value=f"{rd_sign}{rd_pct:.2f}%",
            delta=f"95% CI: [{rd_ll:.2f}%, {rd_ul:.2f}%]",
            delta_color="off",
            help="Absolute Risk Difference = I_e - I_u. In beneficial trials, called Absolute Risk Reduction (ARR)."
        )
    with c4:
        st.metric(
            label="Linear Standard Error",
            value=f"{calcs['se_rd'] * 100.0:.2f}%",
            delta=f"z = {calcs['z_score']:.2f}",
            delta_color="off",
            help="Binomial standard error on the linear percentage scale: sqrt([I_e(1-I_e)/n1] + [I_u(1-I_u)/n0])"
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        if calcs["nnt"] is not None:
            st.metric(
                label="Number Needed to Treat (NNT)",
                value=f"{calcs['nnt']:.1f}",
                help="Average number of patients required to treat to prevent 1 bad health event: 1 / |RD|"
            )
        elif calcs["nnh"] is not None:
            st.metric(
                label="Number Needed to Harm (NNH)",
                value=f"{calcs['nnh']:.1f}",
                help="Average number of individuals exposed to the hazard to cause 1 additional case of disease: 1 / RD"
            )
        else:
            st.metric(label="NNT / NNH", value="N/A (Null)")
    with c6:
        if calcs["effect_direction"] == "harmful":
            st.metric(
                label="Attributable Fraction (Exposed)",
                value=f"{calcs['attributable_fraction_exposed']:.1f}%",
                help="Proportion of disease in the exposed cohort attributable to exposure: (I_e - I_u) / I_e"
            )
        else:
            st.metric(
                label="Relative Risk Reduction (RRR)",
                value=f"{calcs['attributable_fraction_exposed']:.1f}%",
                help="Relative reduction in disease: (I_u - I_e) / I_u"
            )
    with c7:
        p_val = calcs["p_value_wald"]
        p_txt = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        st.metric(
            label="Wald Test (H0: RD = 0.0)",
            value=f"p = {p_txt}",
            help="Wald z-test statistic on linear scale: RD / SE(RD)"
        )
    with c8:
        p_fish = calcs["p_value_fisher"]
        p_fish_txt = "< 0.0001" if p_fish < 0.0001 else f"{p_fish:.4f}"
        st.metric(
            label="Fisher's Exact Test",
            value=f"p = {p_fish_txt}",
            help="Exact hypergeometric test for the 2x2 contingency table (robust to sparse cells)."
        )


def render_risk_difference_visualizations(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders Plotly risk bar comparison, Forest plot for RD, NNT curve, and 2x2 table."""
    st.markdown("### 📈 Visualizing Absolute Risk Differences & Clinical Workload")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**1. Cumulative Incidence Risk Comparison (Exposed vs. Unexposed)**")
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
            xaxis_title="Cohort Stratum",
            height=330,
            margin=dict(l=30, r=30, t=30, b=40),
            template="plotly_white",
            yaxis=dict(range=[0, max(risk_exp_pct, risk_unexp_pct) * 1.35 + 2.0])
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_chart2:
        st.markdown("**2. Risk Difference Forest Plot (Point Estimate & 95% Wald CI)**")
        rd = calcs["risk_difference"] * 100.0
        rd_ll = calcs["rd_ci_lower"] * 100.0
        rd_ul = calcs["rd_ci_upper"] * 100.0

        fig_forest = go.Figure()
        err_plus = rd_ul - rd
        err_minus = rd - rd_ll

        fig_forest.add_trace(go.Scatter(
            x=[rd],
            y=["Active Investigation"],
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
            name="Risk Difference"
        ))

        bound = max(abs(rd_ll), abs(rd_ul), 5.0) * 1.35
        fig_forest.add_vline(
            x=0.0,
            line_dash="dash",
            line_color="#64748b",
            annotation_text="Null Line (RD = 0%)",
            annotation_position="top right"
        )

        fig_forest.add_vrect(
            x0=-bound, x1=0.0,
            fillcolor="#ecfdf5", opacity=0.3, layer="below", line_width=0
        )
        fig_forest.add_vrect(
            x0=0.0, x1=bound,
            fillcolor="#fef2f2", opacity=0.3, layer="below", line_width=0
        )

        fig_forest.update_layout(
            xaxis_title="Risk Difference (Percentage Points %)",
            yaxis_title="",
            height=330,
            margin=dict(l=30, r=30, t=30, b=40),
            template="plotly_white",
            xaxis=dict(range=[-bound, bound]),
            showlegend=False
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # 3. Interactive NNT Reciprocal Curve
    st.markdown("**3. The Reciprocal Curve: How NNT/NNH Changes with Absolute Risk Difference**")
    arr_grid = np.linspace(0.005, 0.25, 200)
    nnt_grid = 1.0 / arr_grid

    active_arr = calcs["arr"] if calcs["arr"] > 0 else (abs(calcs["risk_difference"]) if abs(calcs["risk_difference"]) > 0 else 0.05)
    active_nnt = 1.0 / active_arr if active_arr > 0 else 20.0

    fig_nnt = go.Figure()
    fig_nnt.add_trace(go.Scatter(
        x=arr_grid * 100.0,
        y=nnt_grid,
        mode="lines",
        line=dict(color="#2563eb", width=2.5),
        name="NNT = 1 / ARR"
    ))
    fig_nnt.add_trace(go.Scatter(
        x=[active_arr * 100.0],
        y=[active_nnt],
        mode="markers+text",
        marker=dict(size=12, color="#dc2626"),
        text=[f"Current: {active_nnt:.1f} (ARR={active_arr*100.0:.1f}%)"],
        textposition="top right",
        name="Current Scenario"
    ))
    fig_nnt.update_layout(
        xaxis_title="Absolute Risk Reduction / Difference (%)",
        yaxis_title="Number Needed to Treat / Harm",
        height=300,
        margin=dict(l=30, r=30, t=30, b=40),
        template="plotly_white",
        yaxis=dict(range=[0, min(100, max(nnt_grid))]),
        showlegend=False
    )
    st.plotly_chart(fig_nnt, use_container_width=True)

    # 4. Interactive 2x2 Contingency Table Display
    st.markdown("**4. Cohort Contingency Structure (2×2 Matrix & Marginals)**")
    a, b, c, d = calcs["a"], calcs["b"], calcs["c"], calcs["d"]
    n1, n0 = calcs["n_exposed"], calcs["n_unexposed"]
    m1, m0 = a + c, b + d
    n_tot = calcs["n_total"]

    exp_label, unexp_label = meta.get("exposure_levels", ("Exposed", "Unexposed"))
    out_pos, out_neg = meta.get("outcome_levels", ("Cases (Outcome +)", "Non-Cases (Outcome -)"))

    df_2x2_display = pd.DataFrame([
        {
            "Cohort Stratum": f"🟢 {exp_label}",
            f"{out_pos}": f"{a:,}",
            f"{out_neg}": f"{b:,}",
            "Stratum Total (n)": f"{n1:,}",
            "Cumulative Incidence (Risk)": f"{calcs['risk_exposed'] * 100.0:.2f}%"
        },
        {
            "Cohort Stratum": f"⚪ {unexp_label}",
            f"{out_pos}": f"{c:,}",
            f"{out_neg}": f"{d:,}",
            "Stratum Total (n)": f"{n0:,}",
            "Cumulative Incidence (Risk)": f"{calcs['risk_unexposed'] * 100.0:.2f}%"
        },
        {
            "Cohort Stratum": "📊 Total Cohort",
            f"{out_pos}": f"{m1:,} (a + c)",
            f"{out_neg}": f"{m0:,} (b + d)",
            "Stratum Total (n)": f"{n_tot:,} (Total N)",
            "Cumulative Incidence (Risk)": f"RD = {calcs['percentage_point_diff']:+.2f}% points"
        }
    ])

    st.dataframe(df_2x2_display, use_container_width=True, hide_index=True)


def render_risk_difference_excel_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 3-step Microsoft Excel Implementation Guide with cell coordinates."""
    st.markdown("### 📗 Microsoft Excel Implementation Workflow for Risk Difference & NNT")
    st.markdown("Follow the 3-step sequence: **Set up 2×2 Layout** ➔ **Calculate Incidences & Risk Difference** ➔ **Calculate Linear Standard Error, 95% CI & NNT/NNH**.")

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
        * **Column D (Rows 2:4):** Group Totals (`D2`: `=SUM(B2:C2)`, `D3`: `=SUM(B3:C3)`, `D4`: `=SUM(B4:C4)`)
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
                "Risk Difference (RD = I_e - I_u)",
                "Absolute Risk Reduction (ARR)",
                "Standard Error of RD [SE]",
                "95% CI Lower Bound",
                "95% CI Upper Bound",
                "Number Needed to Treat (NNT = 1 / ARR)",
                "Wald Two-Sided p-value"
            ],
            "Excel Formula": [
                "=B2/D2",
                "=B3/D3",
                "=B6-B7",
                "=ABS(B8)",
                "=SQRT((B6*(1-B6)/D2) + (B7*(1-B7)/D3))",
                "=B8 - 1.96 * B10",
                "=B8 + 1.96 * B10",
                "=IF(B9>0, 1/B9, \"N/A\")",
                "=2 * (1 - NORM.S.DIST(ABS(B8 / B10), TRUE))"
            ],
            "Calculated Value": [
                f"{calcs['risk_exposed']:.4f}",
                f"{calcs['risk_unexposed']:.4f}",
                f"{calcs['risk_difference']:.4f}",
                f"{calcs['arr']:.4f}",
                f"{calcs['se_rd']:.4f}",
                f"{calcs['rd_ci_lower']:.4f}",
                f"{calcs['rd_ci_upper']:.4f}",
                f"{calcs['nnt']:.1f}" if calcs['nnt'] is not None else (f"{calcs['nnh']:.1f} (NNH)" if calcs['nnh'] is not None else "N/A"),
                f"{calcs['p_value_wald']:.4e}"
            ]
        })
        st.dataframe(df_summary_cells, use_container_width=True, hide_index=True)

    with st.expander("📌 Step 2: Calculate Incidences & Risk Difference in Excel", expanded=False):
        st.markdown("""
        1. **Compute Cumulative Incidence in Exposed ($I_e$):** In cell `B6`, enter `=B2/D2`.
        2. **Compute Cumulative Incidence in Unexposed ($I_u$):** In cell `B7`, enter `=B3/D3`.
        3. **Compute Risk Difference (RD):** In cell `B8`, enter:
           ```excel
           =B6 - B7
           ```
        4. **Compute Absolute Risk Reduction (ARR):** In cell `B9`, enter `=ABS(B8)`.
        """)

    with st.expander("📌 Step 3: Compute Linear Standard Error, 95% Wald CI, and NNT/NNH in Excel", expanded=False):
        st.markdown("""
        Because Risk Difference is an additive linear subtraction, confidence intervals are computed directly on the linear scale:

        1. **Compute Standard Error of Risk Difference:** In cell `B10`, enter:
           ```excel
           =SQRT((B6*(1 - B6)/D2) + (B7*(1 - B7)/D3))
           ```
        2. **Compute 95% CI Lower Bound:** In cell `B11`, enter:
           ```excel
           =B8 - 1.96 * B10
           ```
        3. **Compute 95% CI Upper Bound:** In cell `B12`, enter:
           ```excel
           =B8 + 1.96 * B10
           ```
        4. **Compute Number Needed to Treat (NNT) / Harm (NNH):** In cell `B13`, enter:
           ```excel
           =IF(B9 > 0, 1 / B9, "N/A")
           ```
        5. **Compute Wald Two-Sided p-value:** In cell `B14`, enter:
           ```excel
           =2 * (1 - NORM.S.DIST(ABS(B8 / B10), TRUE))
           ```
        """)


def render_risk_difference_python_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 4-step Python Implementation Workflow with docstrings and conditional prints."""
    st.markdown("### 🐍 Python Implementation Workflow for Risk Difference & NNT/NNH")
    st.markdown("This guide follows a strict 4-step sequence: **Evaluate Prerequisites & Baseline Risk** ➔ **Calculate Cumulative Incidences & Risk Difference** ➔ **Calculate Linear Standard Error, 95% CI & NNT/NNH** ➔ **Consolidated Executable Pipeline Script**.")

    a, b, c, d = calcs["a"], calcs["b"], calcs["c"], calcs["d"]

    with st.expander("📌 Step 1: Evaluate Prerequisites & Baseline Risk in Python", expanded=False):
        st.code('''import numpy as np

def check_risk_difference_prerequisites(
    study_design: str,
    a: int,
    b: int,
    c: int,
    d: int,
    loss_to_follow_up_pct: float = 0.0
) -> dict:
    """Verifies epidemiological prerequisites, baseline dependency, and attrition for Risk Difference.

    Args:
        study_design (str): Must be 'prospective_cohort', 'retrospective_cohort', 'rct', or 'cross_sectional'.
            (If 'case_control', absolute risk cannot be measured and Odds Ratio must be used).
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        loss_to_follow_up_pct (float, optional): Percent lost to follow-up. Defaults to 0.0.

    Returns:
        dict: Diagnostic pass/warning/fail flags and recommended actions.
    """
    valid_designs = {"prospective_cohort", "retrospective_cohort", "rct", "cross_sectional"}
    if study_design.lower() not in valid_designs:
        print("[INAPPROPRIATE - FAIL] Risk Difference is invalid for Case-Control studies! Use Odds Ratio instead.")
        return {"status": "FAIL"}

    if study_design.lower() == "cross_sectional":
        print("[ASSURANCE - PASS] Cross-Sectional survey detected. Risk Difference represents Percentage Point Difference.")

    min_cell = min(a, b, c, d)
    if min_cell < 5:
        print(f"[WARNING] Small cell counts (< 5). Cross-reference normal Wald CI with Fisher exact test.")
    else:
        print("[ASSURANCE - PASS] Cell counts robust (>= 5). Asymptotic linear Wald CI is reliable.")

    if loss_to_follow_up_pct > 20.0:
        print("[INAPPROPRIATE - FAIL] Severe attrition bias (> 20% loss). Cumulative incidence distorted.")
    elif loss_to_follow_up_pct >= 5.0:
        print("[WARNING] Moderate loss to follow-up (5-20%). Conduct sensitivity analysis.")
    else:
        print("[ASSURANCE - PASS] Minimal loss to follow-up (< 5%). Attrition bias is negligible.")

    return {"status": "PASS"}
''', language="python")

    with st.expander("📌 Step 2: Calculate Incidences & Risk Difference Point Estimate in Python", expanded=False):
        st.code('''def calculate_rd_point_estimates(a: int, b: int, c: int, d: int) -> dict:
    """Calculates cumulative incidences, Risk Difference (RD), and Absolute Risk Reduction (ARR).

    Args:
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.

    Returns:
        dict: Dictionary containing:
            - 'risk_exposed': a / (a + b)
            - 'risk_unexposed': c / (c + d)
            - 'risk_difference': I_e - I_u
            - 'arr': Absolute Risk Reduction (|RD|)
            - 'effect_direction': 'harmful', 'protective', or 'null'
    """
    n1 = a + b
    n0 = c + d

    risk_exp = a / n1 if n1 > 0 else 0.0
    risk_unexp = c / n0 if n0 > 0 else 0.0
    rd = risk_exp - risk_unexp

    direction = "harmful" if rd > 1e-6 else ("protective" if rd < -1e-6 else "null")
    arr = abs(rd) if direction == "protective" else 0.0

    return {
        "risk_exposed": risk_exp,
        "risk_unexposed": risk_unexp,
        "risk_difference": rd,
        "arr": arr,
        "effect_direction": direction
    }
''', language="python")

    with st.expander("📌 Step 3: Compute Linear Standard Error, Wald 95% CI & NNT/NNH in Python", expanded=False):
        st.code('''from scipy import stats

def compute_rd_confidence_interval_and_nnt(
    a: int,
    b: int,
    c: int,
    d: int,
    alpha: float = 0.05
) -> dict:
    """Computes linear-scale Standard Error, 95% Wald CI, NNT, NNH, and Wald z-test.

    Args:
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        alpha (float, optional): Significance level. Defaults to 0.05 (95% CI).

    Returns:
        dict: se_rd, ci_lower, ci_upper, nnt, nnh, z_score, p_value_wald.
    """
    n1 = a + b
    n0 = c + d

    risk_exp = a / n1
    risk_unexp = c / n0
    rd = risk_exp - risk_unexp

    # Linear-scale binomial variance
    var_exp = (risk_exp * (1.0 - risk_exp)) / n1
    var_unexp = (risk_unexp * (1.0 - risk_unexp)) / n0
    se_rd = np.sqrt(var_exp + var_unexp)

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_lower = max(-1.0, rd - z_crit * se_rd)
    ci_upper = min(1.0, rd + z_crit * se_rd)

    # NNT / NNH
    nnt = (1.0 / abs(rd)) if rd < -1e-6 else None
    nnh = (1.0 / rd) if rd > 1e-6 else None

    z_score = rd / se_rd if se_rd > 0 else 0.0
    p_value_wald = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return {
        "se_rd": se_rd,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "nnt": nnt,
        "nnh": nnh,
        "z_score": z_score,
        "p_value_wald": p_value_wald
    }
''', language="python")

    with st.expander("📌 Step 4: Consolidated Executable Pipeline Script", expanded=False):
        st.code(f'''import numpy as np
from scipy import stats

def run_risk_difference_pipeline(
    a: int,
    b: int,
    c: int,
    d: int,
    study_design: str = "rct",
    loss_to_follow_up_pct: float = 0.0
) -> dict:
    """Executes end-to-end Risk Difference (RD) and NNT/NNH analysis with bias checks.

    Args:
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        study_design (str, optional): Architecture. Defaults to 'rct'.
        loss_to_follow_up_pct (float, optional): Percent lost to follow-up. Defaults to 0.0.

    Returns:
        dict: Complete results dictionary with incidences, RD, CIs, NNT/NNH, and test stats.
    """
    print("=" * 60)
    print("EPIDEMIOLOGICAL RISK DIFFERENCE & NNT/NNH PIPELINE")
    print("=" * 60)

    # Step 1: Diagnostics
    valid_designs = {{"prospective_cohort", "retrospective_cohort", "rct", "cross_sectional"}}
    if study_design.lower() not in valid_designs:
        print("[INAPPROPRIATE - FAIL] RD invalid for Case-Control studies! Use Odds Ratio.")
        return {{}}

    min_cell = min(a, b, c, d)
    if min_cell < 5:
        print(f"[WARNING] Minimum cell is {{min_cell}} (< 5). Cross-reference with Fisher exact test.")
    else:
        print(f"[ASSURANCE - PASS] Minimum cell is {{min_cell}} (>= 5). Asymptotic normal CI robust.")

    # Step 2: Cumulative Incidences & Point Estimates
    n1 = a + b
    n0 = c + d
    risk_exp = a / n1
    risk_unexp = c / n0
    rd = risk_exp - risk_unexp

    # Step 3: Linear Standard Error & 95% Wald CI
    se_rd = np.sqrt((risk_exp * (1.0 - risk_exp) / n1) + (risk_unexp * (1.0 - risk_unexp) / n0))
    z_crit = 1.95996
    ci_lower = rd - z_crit * se_rd
    ci_upper = rd + z_crit * se_rd

    # Step 4: NNT / NNH
    nnt = (1.0 / abs(rd)) if rd < -1e-6 else None
    nnh = (1.0 / rd) if rd > 1e-6 else None

    z_score = rd / se_rd if se_rd > 0 else 0.0
    p_wald = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    table = np.array([[a, b], [c, d]])
    _, p_fisher = stats.fisher_exact(table)

    print("\\n--- RESULTS ---")
    print(f"Risk in Exposed (I_e):    {{risk_exp * 100.0:.2f}}%")
    print(f"Risk in Unexposed (I_u):  {{risk_unexp * 100.0:.2f}}%")
    print(f"Risk Difference (RD):     {{rd * 100.0:+.2f}}% points")
    print(f"95% Wald CI:              [{{ci_lower * 100.0:+.2f}}%, {{ci_upper * 100.0:+.2f}}%]")
    if nnt is not None:
        print(f"Number Needed to Treat:   {{nnt:.1f}} patients")
    if nnh is not None:
        print(f"Number Needed to Harm:    {{nnh:.1f}} individuals")
    print(f"Wald Test p-value:        {{p_wald:.4e}} (z = {{z_score:.2f}})")
    print(f"Fisher's Exact Test:      {{p_fisher:.4e}}")

    return {{
        "risk_exposed": risk_exp,
        "risk_unexposed": risk_unexp,
        "risk_difference": rd,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "nnt": nnt,
        "nnh": nnh,
        "p_value_wald": p_wald
    }}

# Execute with scenario data ({meta['title']}):
if __name__ == "__main__":
    results = run_risk_difference_pipeline(
        a={a}, b={b}, c={c}, d={d},
        study_design="{meta.get('study_design', 'rct')}",
        loss_to_follow_up_pct={meta.get('loss_to_follow_up_pct', 0.0)}
    )
''', language="python")


def render_risk_difference_download_tab(
    df_contingency: pd.DataFrame,
    df_microdata: pd.DataFrame,
    meta: Dict[str, Any]
):
    """Renders the data download tab with individual microdata and 2x2 contingency table CSVs."""
    st.markdown("### 📥 Download Sample Datasets")
    st.markdown(f"""
    Download the sample data associated with the active scenario: **{meta['title']}**.
    You can use these datasets to practice calculating Risk Differences, linear standard errors, 
    Wald confidence intervals, and NNT/NNH in Microsoft Excel or Python.
    """)

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.markdown("#### 1. Individual Participant Microdata (CSV)")
        st.markdown(f"Contains **{len(df_microdata):,} participant records** with columns: `Participant_ID`, `Exposure_Group`, `Exposure_Binary`, `Outcome_Status`, `Outcome_Binary`, `Follow_Up_Years`, `Age`, `Sex`.")
        csv_micro = df_microdata.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Individual Participant Microdata (CSV)",
            data=csv_micro,
            file_name=f"{meta['scenario_key']}_microdata.csv",
            mime="text/csv",
            key="btn_dl_rd_microdata"
        )
        st.dataframe(df_microdata.head(8), height=220, use_container_width=True, hide_index=True)

    with col_dl2:
        st.markdown("#### 2. Aggregated 2×2 Contingency Table (CSV)")
        st.markdown("Contains the summarized 2×2 contingency matrix matching the exact Excel tutorial cell coordinates (`A1:E4`).")
        csv_cont = df_contingency.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Aggregated 2×2 Table (CSV)",
            data=csv_cont,
            file_name=f"{meta['scenario_key']}_2x2_contingency_table.csv",
            mime="text/csv",
            key="btn_dl_rd_contingency"
        )
        st.dataframe(df_contingency, height=220, use_container_width=True, hide_index=True)


def render_risk_difference_module(active_method_key: str, method_info: Dict[str, Any]):
    """Renders the comprehensive Risk Difference (RD) and NNT/NNH module."""
    # 1. Plain English Guide
    render_risk_difference_plain_english_guide()

    # 2. Sidebar Controls
    st.sidebar.header("🧪 Absolute Risk Study Settings")
    st.sidebar.markdown("**Study Scenarios & Contingency Parameters:**")

    rd_scenario_choice = st.sidebar.selectbox(
        "Study Scenario:",
        options=["statin_cardiovascular", "smoking_cessation", "occupational_vibration", "patient_feedback_survey", "custom"],
        format_func=lambda s: {
            "statin_cardiovascular": "💊 Regional Secondary Prevention Statin Trial (Beneficial, NNT = 20)",
            "smoking_cessation": "🚭 All Meadows Smoking Cessation Pharmacotherapy Trial (Beneficial, NNT = 5)",
            "occupational_vibration": "⛏️ Regional Pneumatic Tool Vibration Cohort (Harmful, NNH ≈ 5.6)",
            "patient_feedback_survey": "📋 NHS Patient Experience Survey (Percentage Point Difference = +20%)",
            "custom": "⚙️ Custom 2×2 Table Input (Enter Your Own Counts)"
        }[s],
        index=0,
        help="Select a realistic clinical/public health scenario or enter custom 2x2 contingency table counts."
    )
    st.sidebar.caption("⚠️ **Data Notice:** Scenarios, populations, and event counts are made-up examples synthetically simulated for educational and illustrative purposes only.")

    if rd_scenario_choice == "custom":
        st.sidebar.subheader("Custom 2×2 Parameters")
        custom_study_design = st.sidebar.selectbox(
            "Study Design Architecture:",
            options=["rct", "prospective_cohort", "retrospective_cohort", "cross_sectional", "case_control"],
            format_func=lambda d: {
                "rct": "Randomized Controlled Trial (RCT)",
                "prospective_cohort": "Prospective Cohort (Forward-looking)",
                "retrospective_cohort": "Retrospective Cohort (Historical records)",
                "cross_sectional": "Cross-Sectional (Prevalence snapshot)",
                "case_control": "Case-Control (Outcome-first, Retrospective)"
            }[d],
            index=0
        )
        custom_loss_pct = st.sidebar.slider(
            "Loss to Follow-up (%):",
            min_value=0.0,
            max_value=30.0,
            value=2.5,
            step=0.5,
            help="Percentage of enrolled participants who dropped out or could not be assessed at completion."
        )
        st.sidebar.caption(
            "📉 **What is Loss to Follow-up?** Drives Attrition Bias diagnostics: "
            "🟢 `<5%` Pass | 🟡 `5–20%` Caution | 🔴 `>20%` Fail."
        )

        col_sb1, col_sb2 = st.sidebar.columns(2)
        with col_sb1:
            a_val = st.number_input("Exposed / Treated Cases (a):", min_value=0, max_value=100000, value=40, step=1)
            b_val = st.number_input("Exposed / Treated Non-Cases (b):", min_value=0, max_value=100000, value=960, step=5)
        with col_sb2:
            c_val = st.number_input("Unexposed / Control Cases (c):", min_value=0, max_value=100000, value=90, step=1)
            d_val = st.number_input("Unexposed / Control Non-Cases (d):", min_value=0, max_value=100000, value=910, step=5)

        meta = {
            "title": "Custom User-Defined Absolute Risk Investigation",
            "domain": "Epidemiological Hypothesis Testing",
            "description": "User-configured 2x2 contingency absolute risk evaluation.",
            "exposure_name": "Intervention / Hazard",
            "exposure_levels": ("Exposed / Treated", "Unexposed / Control"),
            "outcome_name": "Target Health Event",
            "outcome_levels": ("Cases (Event +)", "Non-Cases (Event -)"),
            "study_design": custom_study_design,
            "expected_effect": "User Defined",
            "clinical_takeaway": "Custom parameters evaluated based on user-entered contingency frequencies.",
            "scenario_key": "custom_risk_difference",
            "loss_to_follow_up_pct": custom_loss_pct
        }

        # Build contingency and microdata
        df_contingency = pd.DataFrame([
            {
                "Cohort_Stratum": "🟢 Exposed / Treated",
                "Cases_Outcome_Pos": a_val,
                "Non_Cases_Outcome_Neg": b_val,
                "Stratum_Total_n": a_val + b_val,
                "Cumulative_Incidence_Risk": a_val / (a_val + b_val) if (a_val + b_val) > 0 else 0.0
            },
            {
                "Cohort_Stratum": "⚪ Unexposed / Control",
                "Cases_Outcome_Pos": c_val,
                "Non_Cases_Outcome_Neg": d_val,
                "Stratum_Total_n": c_val + d_val,
                "Cumulative_Incidence_Risk": c_val / (c_val + d_val) if (c_val + d_val) > 0 else 0.0
            },
            {
                "Cohort_Stratum": "📊 Total Cohort",
                "Cases_Outcome_Pos": a_val + c_val,
                "Non_Cases_Outcome_Neg": b_val + d_val,
                "Stratum_Total_n": a_val + b_val + c_val + d_val,
                "Cumulative_Incidence_Risk": (a_val + c_val) / (a_val + b_val + c_val + d_val) if (a_val + b_val + c_val + d_val) > 0 else 0.0
            }
        ])

        micro_list = []
        for i in range(min(a_val, 100)):
            micro_list.append({"Participant_ID": f"EXP_{i+1:04d}", "Exposure_Group": "Exposed", "Exposure_Binary": 1, "Outcome_Status": "Event +", "Outcome_Binary": 1, "Follow_Up_Years": 5.0, "Age": 60, "Sex": "Male"})
        for i in range(min(b_val, 100)):
            micro_list.append({"Participant_ID": f"EXP_{a_val+i+1:04d}", "Exposure_Group": "Exposed", "Exposure_Binary": 1, "Outcome_Status": "Event -", "Outcome_Binary": 0, "Follow_Up_Years": 5.0, "Age": 59, "Sex": "Female"})
        for i in range(min(c_val, 100)):
            micro_list.append({"Participant_ID": f"UNEXP_{i+1:04d}", "Exposure_Group": "Unexposed", "Exposure_Binary": 0, "Outcome_Status": "Event +", "Outcome_Binary": 1, "Follow_Up_Years": 5.0, "Age": 61, "Sex": "Male"})
        for i in range(min(d_val, 100)):
            micro_list.append({"Participant_ID": f"UNEXP_{c_val+i+1:04d}", "Exposure_Group": "Unexposed", "Exposure_Binary": 0, "Outcome_Status": "Event -", "Outcome_Binary": 0, "Follow_Up_Years": 5.0, "Age": 60, "Sex": "Female"})
        df_microdata = pd.DataFrame(micro_list)

        calcs = calculate_risk_difference(a_val, b_val, c_val, d_val)
        evaluations = evaluate_risk_difference_prerequisites(
            study_design=custom_study_design,
            a=a_val, b=b_val, c=c_val, d=d_val,
            loss_to_follow_up_pct=custom_loss_pct,
            baseline_risk_known=True
        )
    else:
        df_contingency, df_microdata, meta = generate_risk_difference_demo_data(scenario=rd_scenario_choice, seed=42)
        calcs = meta["calcs"]
        evaluations = evaluate_risk_difference_prerequisites(
            study_design=meta.get("study_design", "rct"),
            a=meta["a"], b=meta["b"], c=meta["c"], d=meta["d"],
            loss_to_follow_up_pct=meta.get("loss_to_follow_up_pct", 0.0),
            baseline_risk_known=True
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
            ⚠️ <strong>Disclaimer:</strong> This is a completely made-up example scenario with synthetic data simulated for educational and illustrative purposes only. It does not represent actual medical records or real trial data from NHS trusts or healthcare providers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if meta.get("scenario_key") == "patient_feedback_survey":
        st.info("""
        💡 **Methodological Note on Naming Conventions in Survey Research:**  
        In customer, patient, and cross-sectional survey research, Risk Difference is universally referred to as the **Percentage Point Difference** (or Prevalence Difference). 
        This is because you are comparing existing prevalence percentages ($80\% - 60\% = +20\text{ percentage points}$) rather than cumulative disease incidence over time. 
        **The mathematical calculations, binomial standard errors, and linear 95% confidence intervals are 100% identical!**
        """)

    # 1. Tri-Color Diagnostics
    render_risk_difference_diagnostics(evaluations)

    # 2. Metric Dashboard
    render_risk_difference_dashboard(calcs, meta)

    # 3. Visualizations & 2x2 Grid
    render_risk_difference_visualizations(calcs, meta)

    # 4. Methodology Tabs
    st.markdown("---")
    tab_rd_rationale, tab_rd_excel, tab_rd_code, tab_rd_download = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Excel Implementation Guide",
        "🐍 Python Code Implementation",
        "📥 Download Sample Data"
    ])

    with tab_rd_rationale:
        st.subheader(f"Why Risk Difference & NNT/NNH? ({method_info['name']})")
        st.markdown("""
        When clinicians and commissioners decide whether to fund a new drug, public health intervention, or safety guideline, 
        relative measures ($RR$ and $OR$) are not enough. Risk Difference and NNT provide the direct, real-world impact needed for health economics.
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
        st.markdown("### 🔍 In-Depth Epidemiological Framework: Risk Difference vs. Relative Risk vs. Odds Ratio")
        st.markdown(r"""
        | Dimension | Risk Difference (RD / ARR) | Relative Risk (RR) | Odds Ratio (OR) |
        | :--- | :--- | :--- | :--- |
        | **Mathematical Formula** | $I_e - I_u$ | $\frac{I_e}{I_u}$ | $\frac{a \times d}{b \times c}$ |
        | **Scale & CI Calculation** | **Additive & Linear** ($RD \pm 1.96 \times SE$) | **Multiplicative & Log-scale** ($\exp[\ln(RR) \pm 1.96 \times SE]$) | **Multiplicative & Log-scale** ($\exp[\ln(OR) \pm 1.96 \times SE]$) |
        | **Null Benchmark Value** | **$0.00$** ($0.0\%$ difference) | **$1.00$** (Equal risk) | **$1.00$** (Equal odds) |
        | **Clinical Meaning** | **Absolute Clinical Workload & Impact:** How many fewer (or extra) people get sick per 100 individuals? Directly converts to **NNT** or **NNH**. | **Biological Strength of Association:** How many times higher or lower is the disease risk? | **Exposure Odds in Diseased vs. Controls:** Universal currency in logistic regression; approximates RR only when disease is rare. |
        | **Baseline Risk Dependency** | **Highly Dependent:** If baseline risk is low, RD is small and NNT is huge, even if RR is massive. | **Independent:** Often assumed constant across baseline risk strata. | **Independent:** Often assumed constant across baseline risk strata. |
        | **Key Derived Measures** | **Number Needed to Treat (NNT = $1 / ARR$)**<br>**Number Needed to Harm (NNH = $1 / RD$)** | **Vaccine Efficacy / Relative Risk Reduction ($1 - RR$)** | **Adjusted Odds Ratio (aOR)** from multivariable logistic regression |
        """)

        st.markdown("#### 📖 Key Concepts Explained (Zero Prior Knowledge Assumed)")
        st.markdown(r"""
        1. **Absolute vs. Relative Measures: The "Rare Disease Trap"**
           * Imagine Condition A affects **$10\%$** ($100$ per $1,000$) and a drug cuts risk by $50\%$ to **$5\%$** ($50$ per $1,000$).
             * $\text{Relative Risk} = 0.50$ ($50\%$ relative reduction).
             * $\text{Risk Difference (ARR)} = 10\% - 5\% = 5\%$.
             * $\text{NNT} = 1 / 0.05 = \mathbf{20 \text{ patients}}$. (Treat 20 to prevent 1 case).
           * Now imagine Condition B is very rare, affecting only **$0.02\%$** ($20$ per $100,000$). A drug cuts risk by the exact same $50\%$ to **$0.01\%$** ($10$ per $100,000$).
             * $\text{Relative Risk} = 0.50$ (Still $50\%$ relative reduction!).
             * $\text{Risk Difference (ARR)} = 0.02\% - 0.01\% = 0.01\%$ ($0.0001$).
             * $\text{NNT} = 1 / 0.0001 = \mathbf{10,000 \text{ patients}}$!
           * **Conclusion:** Both drugs boast an identical *"50% risk reduction"*, but the second drug requires treating **10,000 patients** to prevent a single event. Never rely on Relative Risk alone without looking at the Risk Difference and NNT!

        2. **What happens to NNT when the 95% Confidence Interval for RD crosses Zero?**
           * When a clinical trial is not statistically significant, the 95% CI for Risk Difference spans zero (e.g., from $-3\%$ to $+1\%$).
           * Because $NNT = 1 / RD$, as $RD$ approaches $0$, $1 / 0$ approaches **infinity**!
           * On the negative side (benefit), NNT extends to infinity ($\text{NNT} = \infty$). As the CI crosses zero into positive territory, the intervention actually causes harm, becoming **Number Needed to Harm (NNH)**!
           * Therefore, if a confidence interval for Risk Difference spans zero, you cannot quote a single finite NNT confidence interval; the treatment is not proven superior to control.

        3. **Why Public Health Policy Demands NNT and NNH:**
           * NNT and NNH allow clinicians and commissioners to directly weigh **Benefit vs. Harm**:
             * *Example:* If a blood-thinner has an $\text{NNT} = 30$ to prevent an ischemic stroke, but an $\text{NNH} = 250$ for severe gastrointestinal bleeding, the benefit clearly outweighs the risk.
             * But if $\text{NNT} = 150$ and $\text{NNH} = 80$, the drug harms more people than it helps!
        """)

    with tab_rd_excel:
        render_risk_difference_excel_guide(calcs, meta)

    with tab_rd_code:
        render_risk_difference_python_guide(calcs, meta)

    with tab_rd_download:
        render_risk_difference_download_tab(df_contingency, df_microdata, meta)


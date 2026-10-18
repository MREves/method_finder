"""Streamlit UI Components & Visualizations for Odds Ratio (OR) & Case-Control Studies.

Provides:
1. render_odds_ratio_plain_english_guide: Zero-prior-knowledge walkthrough with analogies.
2. render_odds_ratio_diagnostics: Tri-color evaluation cards (PASS, WARNING, FAIL).
3. render_odds_ratio_dashboard: Metric cards (Odds in Cases/Controls, OR, 95% CI, p-values).
4. render_odds_ratio_visualizations: Exposure prevalence comparison, Forest Plot with 95% CIs, and 2x2 grid.
5. render_odds_ratio_excel_guide: Strict 3-step sequence with cell mapping and explicit formulas.
6. render_odds_ratio_python_guide: Strict 4-step sequence with exhaustive docstrings and conditional prints.
7. render_odds_ratio_download_tab: Live CSV download for both individual microdata and 2x2 table.
8. render_odds_ratio_module: Top-level controller for Odds Ratio.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.odds_ratio import (
    calculate_odds_ratio,
    evaluate_odds_ratio_prerequisites,
    generate_odds_ratio_demo_data
)


def render_odds_ratio_plain_english_guide():
    """Renders the zero-prior-knowledge pedagogical guide for Odds Ratio."""
    with st.expander("💡 Plain English Guide: Odds Ratio (OR) & Case-Control Study Design", expanded=False):
        st.markdown("""
        In epidemiology and public health, the **Odds Ratio (OR)** is one of the most widely used effect measures. 
        It is essential for investigating rare diseases, acute foodborne outbreaks, and interpreting findings from multivariable logistic regression.
        """)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. What are "Odds" vs. "Probability"?
            * **Probability ($P$):** The chance that an event occurs out of **all possible outcomes**:
              $$P = \frac{\text{Number of times event happens}}{\text{Total number of all opportunities}}$$
              *Example:* If 1 out of 6 dice rolls is a six, the probability is $1/6 \approx 16.7\%$.
            * **Odds ($O$):** The ratio of the chance that an event happens to the chance that it **does NOT happen**:
              $$\text{Odds} = \frac{P}{1 - P} = \frac{\text{Events that happen}}{\text{Events that do not happen}}$$
              *Example:* On a die roll, the odds of rolling a six are **1 to 5** ($1/5 = 0.20$).
            * **Summary Rule:** Probability is *"part-to-whole"*, while Odds is *"part-to-part"*.
            """)

            st.markdown(r"""
            #### 2. What is the Odds Ratio (OR)?
            * In a Case-Control study, we compare **Exposure Odds** between people who already have the condition (Cases) and people who do not (Controls):
              $$\text{Exposure Odds in Cases} = \frac{\text{Exposed Cases } (a)}{\text{Unexposed Cases } (c)} = \frac{a}{c}$$
              $$\text{Exposure Odds in Controls} = \frac{\text{Exposed Controls } (b)}{\text{Unexposed Controls } (d)} = \frac{b}{d}$$
            * The **Odds Ratio (OR)** divides the odds in cases by the odds in controls (the famous *cross-product ratio*):
              $$OR = \frac{a / c}{b / d} = \frac{a \times d}{b \times c}$$
            * **$OR = 1.0$ (The Null Value):** Parity. Past exposure odds are identical between cases and controls (no association).
            * **$OR > 1.0$ (Harmful Exposure):** Diseased cases had higher odds of past exposure than controls (e.g. $OR = 3.0$ means cases had 3 times higher odds of being exposed).
            * **$OR < 1.0$ (Protective Factor):** Diseased cases had lower odds of past exposure (e.g. $OR = 0.40$ indicates a protective factor such as physical exercise or vaccination).
            """)

        with col_g2:
            st.markdown("""
            #### 3. The Core Analogy: "The Detective in the Archive" vs. "The Security Guard at the Door"
            * **The Security Guard at the Door (Cohort Study / Relative Risk):**
              Imagine standing at the door of a factory. You record every worker entering who is exposed to chemical dust versus unexposed office staff. 
              You follow them forward in time for 10 years and count who gets sick. Because you know the total number of exposed ($a+b$) and unexposed ($c+d$) people who entered, 
              you can calculate true cumulative incidence. **Relative Risk (RR) is mathematically valid!**
            * **The Detective in the Archive (Case-Control Study / Odds Ratio):**
              Now imagine a very rare cancer (like mesothelioma). Only 1 in 50,000 people ever get it. A cohort study would have to follow 500,000 people for 20 years just to see 10 cases! 
              Instead, an epidemiological "detective" visits the hospital archives today, gathers 50 patients who already have mesothelioma (**Cases**), 
              and pairs them with 100 healthy hospital patients or neighbours (**Controls**).
            * **Why Relative Risk is impossible here:** Because the detective artificially decided to recruit 50 cases and 100 controls, 
              the proportion of ill people in the study ($50 / 150 = 33\%$) is an artifact of the study design, not real-world disease incidence! 
              **You cannot calculate true risk or Relative Risk. But you CAN ask: "Were the cases more likely to have worked with asbestos in the past than controls?" That is the Odds Ratio!**
            """)

            st.markdown(r"""
            #### 4. The Rare Disease Assumption ($OR \approx RR$)
            * When a disease is **rare in the source population** (typically $< 5\text{--}10\%$ cumulative incidence), 
              the number of cases ($a$ and $c$) is very small compared to the non-cases ($b$ and $d$).
            * Mathematically: $a + b \approx b$ and $c + d \approx d$. Therefore:
              $$RR = \frac{a / (a + b)}{c / (c + d)} \approx \frac{a / b}{c / d} = \frac{ad}{bc} = OR$$
            * **The Cautionary Rule:** When a disease is common ($> 10\%$), the Odds Ratio **exaggerates** the effect size compared to Relative Risk:
              * If harmful: $OR > RR$ (appears more dangerous than it is).
              * If protective: $OR < RR$ (appears more protective than it is).
            """)

            st.markdown(r"""
            #### 5. Survey & Feedback Analytics: The "Prevalence Odds Ratio (POR)"
            * **Customer / Patient Surveys:** When analyzing survey ratings across demographics (e.g. young patients aged $\le 18$ vs. adults aged $19+$ giving positive vs. negative feedback), you are measuring existing sentiment in a **Cross-Sectional Survey**.
            * **Different Name, Identical Mathematics:** In survey research, the Odds Ratio is termed the **Prevalence Odds Ratio (POR)**. The cross-product ratio $\frac{a \times d}{b \times c}$, Woolf's standard error, and 95% Wald confidence intervals are **100% identical**!
            * **⚠️ The Common Outcome Trap:** Because positive reviews are common ($80\%$ in youth vs. $60\%$ in adults), the Rare Disease Assumption does NOT hold:
              * **Prevalence Ratio (RR):** $80\% / 60\% = \mathbf{1.33}$ (*"Youth are 33% more likely to be satisfied"*).
              * **Prevalence Odds Ratio (OR):** $\frac{80/20}{60/40} = \frac{4.0}{1.5} = \mathbf{2.67}$ (*"The odds of satisfaction are 2.67 times higher"*).
              * **Reporting Warning:** Do not misinterpret an Odds Ratio of $2.67$ as meaning young people are "2.7 times as satisfied" — they are only 1.33 times as likely to be satisfied!
            """)


        st.info(r"""
        📐 **Why are Confidence Intervals calculated on the Natural Log Scale ($\ln(OR)$)?**  
        Like the Relative Risk, the Odds Ratio cannot fall below zero, but can extend to infinity ($[0, \infty)$), with $1.0$ as the null centre. 
        Because of this right-skewed distribution, standard errors cannot be symmetrically added to $OR$ on the linear scale. 
        Epidemiologists transform $OR$ to the natural log scale ($\ln(OR)$), where Woolf showed the distribution is normal with standard error:
        $$SE(\ln(OR)) = \sqrt{\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d}}$$
        We compute the symmetric 95% confidence interval on the log scale, and then exponentiate ($\exp$) back:
        $$\text{95\% Confidence Interval (CI)} = \exp\left(\ln(OR) \pm 1.96 \times SE(\ln(OR))\right)$$
        """)


def render_odds_ratio_diagnostics(evaluations: List[Dict[str, Any]]):
    """Renders tri-color prerequisite diagnostic cards for Odds Ratio."""
    st.markdown("### 🚦 Tri-Color Bias & Methodological Diagnostics")
    st.caption("Verifies study design suitability, the rare disease assumption, control sampling (Berkson's bias), recall bias, and cell sparsity.")

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


def render_odds_ratio_dashboard(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the top summary metric cards for Odds Ratio."""
    st.markdown("### 📊 Primary Epidemiological Effect Measures")

    or_val = calcs["or_point"]
    or_ll = calcs["or_ci_lower"]
    or_ul = calcs["or_ci_upper"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Exposure % (Cases)",
            value=f"{calcs['exposure_pct_cases']:.1f}%",
            help="Proportion of diseased cases who had past exposure: a / (a + c)"
        )
    with c2:
        st.metric(
            label="Exposure % (Controls)",
            value=f"{calcs['exposure_pct_controls']:.1f}%",
            help="Proportion of healthy controls who had past exposure: b / (b + d)"
        )
    with c3:
        ci_str = f"[{or_ll:.2f}, {or_ul:.2f}]" if not np.isnan(or_ll) else "N/A"
        st.metric(
            label="Odds Ratio (OR)",
            value=f"{or_val:.2f}" if not np.isnan(or_val) else "N/A",
            delta=f"95% CI: {ci_str}",
            delta_color="off",
            help="Cross-Product Odds Ratio = (a * d) / (b * c). Parity null = 1.0."
        )
    with c4:
        st.metric(
            label="Natural Log of OR [ln(OR)]",
            value=f"{calcs['ln_or']:.3f}" if not np.isnan(calcs['ln_or']) else "N/A",
            delta=f"SE = {calcs['se_ln_or']:.3f}",
            delta_color="off",
            help="Woolf's log-scale transformation and standard error: sqrt(1/a + 1/b + 1/c + 1/d)"
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric(
            label="Exposure Odds (Cases)",
            value=f"{calcs['odds_cases']:.2f}" if not np.isnan(calcs['odds_cases']) else "N/A",
            help="Odds of exposure among cases: a / c"
        )
    with c6:
        st.metric(
            label="Exposure Odds (Controls)",
            value=f"{calcs['odds_controls']:.2f}" if not np.isnan(calcs['odds_controls']) else "N/A",
            help="Odds of exposure among controls: b / d"
        )
    with c7:
        p_val = calcs["p_value_wald"]
        p_txt = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        st.metric(
            label="Wald Test (H0: OR = 1.0)",
            value=f"p = {p_txt}",
            delta=f"z = {calcs['z_score']:.2f}",
            delta_color="off",
            help="Wald z-test statistic on the natural log scale: ln(OR) / SE(ln(OR))"
        )
    with c8:
        p_fish = calcs["p_value_fisher"]
        p_fish_txt = "< 0.0001" if p_fish < 0.0001 else f"{p_fish:.4f}"
        st.metric(
            label="Fisher's Exact Test",
            value=f"p = {p_fish_txt}",
            help="Exact hypergeometric test for the 2x2 contingency table (valid even with small/sparse cell counts)."
        )


def render_odds_ratio_visualizations(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders Plotly grouped bar chart, forest plot, and interactive 2x2 table."""
    st.markdown("### 📈 Visualizing Exposure Distributions & Effect Magnitude")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**1. Exposure Prevalence Comparison (Cases vs. Controls)**")
        out_pos, out_neg = meta.get("outcome_levels", ("Cases", "Controls"))
        exp_pct_cases = calcs["exposure_pct_cases"]
        exp_pct_controls = calcs["exposure_pct_controls"]

        bar_colors = ["#ef4444", "#3b82f6"] if calcs["effect_direction"] == "harmful" else ["#10b981", "#64748b"]

        fig_prev = go.Figure()
        fig_prev.add_trace(go.Bar(
            x=[out_pos, out_neg],
            y=[exp_pct_cases, exp_pct_controls],
            text=[f"{exp_pct_cases:.1f}%", f"{exp_pct_controls:.1f}%"],
            textposition="auto",
            marker_color=bar_colors,
            width=0.5
        ))
        fig_prev.update_layout(
            yaxis_title="Exposure Prevalence (%)",
            xaxis_title="Participant Outcome Status",
            height=340,
            margin=dict(l=30, r=30, t=30, b=40),
            template="plotly_white",
            yaxis=dict(range=[0, max(exp_pct_cases, exp_pct_controls) * 1.35 + 2.0])
        )
        st.plotly_chart(fig_prev, use_container_width=True)

    with col_chart2:
        st.markdown("**2. Odds Ratio Forest Plot (Point Estimate & 95% Wald CI)**")
        or_val = calcs["or_point"]
        or_ll = calcs["or_ci_lower"]
        or_ul = calcs["or_ci_upper"]

        fig_forest = go.Figure()

        if not np.isnan(or_val) and not np.isnan(or_ll) and not np.isnan(or_ul):
            err_plus = or_ul - or_val
            err_minus = or_val - or_ll

            fig_forest.add_trace(go.Scatter(
                x=[or_val],
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
                name="Odds Ratio"
            ))

            max_x = max(or_ul * 1.25, 2.5)
            fig_forest.add_vline(
                x=1.0,
                line_dash="dash",
                line_color="#64748b",
                annotation_text="Null Line (OR = 1.0)",
                annotation_position="top right"
            )

            # Zone annotations
            fig_forest.add_vrect(
                x0=0, x1=1.0,
                fillcolor="#ecfdf5", opacity=0.3, layer="below", line_width=0
            )
            fig_forest.add_vrect(
                x0=1.0, x1=max_x,
                fillcolor="#fef2f2", opacity=0.3, layer="below", line_width=0
            )

            fig_forest.update_layout(
                xaxis_title="Odds Ratio (OR)",
                yaxis_title="",
                height=340,
                margin=dict(l=30, r=30, t=30, b=40),
                template="plotly_white",
                xaxis=dict(range=[0, max_x]),
                showlegend=False
            )
        st.plotly_chart(fig_forest, use_container_width=True)

    # 3. Interactive 2x2 Contingency Table Display
    st.markdown("**3. Case-Control Contingency Matrix (2×2 Matrix & Marginals)**")
    a, b, c, d = calcs["a"], calcs["b"], calcs["c"], calcs["d"]
    n_cases, n_controls = calcs["n_cases"], calcs["n_controls"]
    m1, m0 = a + b, c + d
    n_tot = calcs["n_total"]

    exp_label, unexp_label = meta.get("exposure_levels", ("Exposed", "Unexposed"))
    out_pos, out_neg = meta.get("outcome_levels", ("Cases (Outcome +)", "Controls (Outcome -)"))

    df_2x2_display = pd.DataFrame([
        {
            "Historical Exposure Status": f"🟢 {exp_label}",
            f"{out_pos}": f"{a:,}",
            f"{out_neg}": f"{b:,}",
            "Exposure Margin (Row Total)": f"{m1:,} (a + b)",
            "Exposure Odds Contribution": f"a/c = {calcs['odds_cases']:.2f}"
        },
        {
            "Historical Exposure Status": f"⚪ {unexp_label}",
            f"{out_pos}": f"{c:,}",
            f"{out_neg}": f"{d:,}",
            "Exposure Margin (Row Total)": f"{m0:,} (c + d)",
            "Exposure Odds Contribution": f"b/d = {calcs['odds_controls']:.2f}"
        },
        {
            "Historical Exposure Status": "📊 Study Group Totals",
            f"{out_pos}": f"{n_cases:,} (a + c)",
            f"{out_neg}": f"{n_controls:,} (b + d)",
            "Exposure Margin (Row Total)": f"{n_tot:,} (Total N)",
            "Exposure Odds Contribution": f"Cross-Product OR = {calcs['or_point']:.2f}"
        }
    ])


    st.dataframe(df_2x2_display, use_container_width=True, hide_index=True)


def render_odds_ratio_excel_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 3-step Microsoft Excel Implementation Guide with cell coordinates."""
    st.markdown("### 📗 Microsoft Excel Implementation Workflow for Odds Ratio")
    st.markdown("Follow the 3-step sequence: **Set up 2×2 Layout** ➔ **Calculate Exposure Odds & Point Estimate** ➔ **Calculate Log Standard Error & 95% Wald CI**.")

    a, b, c, d = calcs["a"], calcs["b"], calcs["c"], calcs["d"]
    n_cases, n_controls = calcs["n_cases"], calcs["n_controls"]
    m1, m0 = a + b, c + d
    n_tot = calcs["n_total"]
    exp_label, unexp_label = meta.get("exposure_levels", ("Exposed", "Unexposed"))

    with st.expander("📌 Step 1: Excel 2×2 Data Layout & Cell References", expanded=False):
        st.markdown(f"""
        Structure your Excel worksheet with the 2×2 table in `Cells A1:E4` and calculation outputs in `Cells A6:B14`:
        
        * **Column A (Rows 2:4):** Historical Exposure Status (`Row 2: Exposed`, `Row 3: Unexposed`, `Row 4: Total`)
        * **Column B (Rows 2:4):** Cases (`B2`: $a = {a}$, `B3`: $c = {c}$, `B4`: `=SUM(B2:B3)`)
        * **Column C (Rows 2:4):** Controls (`C2`: $b = {b}$, `C3`: $d = {d}$, `C4`: `=SUM(C2:C3)`)
        * **Column D (Rows 2:4):** Exposure Totals (`D2`: `=SUM(B2:C2)`, `D3`: `=SUM(B3:C3)`, `D4`: `=SUM(B4:C4)`)
        * **Column E (Rows 2:4):** Exposure Proportions (`E2`: `=B2/B$4`, `E3`: `=C2/C$4`, `E4`: `=1`)
        """)

        preview_grid = pd.DataFrame([
            {"Excel_Row": "Row 1 (Headers)", "Col_A": "Exposure_Status", "Col_B": "Cases_Outcome_Pos", "Col_C": "Controls_Outcome_Neg", "Col_D": "Total_Exposed_m", "Col_E": "Exposure_Pct"},
            {"Excel_Row": "Row 2 (Exposed)", "Col_A": exp_label, "Col_B": str(a), "Col_C": str(b), "Col_D": f"{m1} (=SUM(B2:C2))", "Col_E": f"{calcs['exposure_pct_cases']:.1f}% (=B2/B$4)"},
            {"Excel_Row": "Row 3 (Unexposed)", "Col_A": unexp_label, "Col_B": str(c), "Col_C": str(d), "Col_D": f"{m0} (=SUM(B3:C3))", "Col_E": f"{calcs['exposure_pct_controls']:.1f}% (=C2/C$4)"},
            {"Excel_Row": "Row 4 (Totals)", "Col_A": "Study Marginals", "Col_B": f"{n_cases} (=SUM(B2:B3))", "Col_C": f"{n_controls} (=SUM(C2:C3))", "Col_D": f"{n_tot} (=SUM(D2:D3))", "Col_E": "100.0%"}
        ])
        st.dataframe(preview_grid, use_container_width=True, hide_index=True)

        st.markdown("**Summary Output Cells Mapping (`Cells A6:B14`):**")
        df_summary_cells = pd.DataFrame({
            "Cell": ["B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14"],
            "Metric Name": [
                "Exposure Odds in Cases",
                "Exposure Odds in Controls",
                "Odds Ratio (OR Point Estimate)",
                "Percentage Odds Reduction (if protective)",
                "Natural Log of OR [ln(OR)]",
                "Woolf's Standard Error of ln(OR)",
                "95% CI Lower Bound",
                "95% CI Upper Bound",
                "Wald Two-Sided p-value"
            ],
            "Excel Formula": [
                "=B2/B3",
                "=C2/C3",
                "=(B2*C3)/(C2*B3)",
                "=IF(B8<1, (1-B8)*100, 0)",
                "=LN(B8)",
                "=SQRT((1/B2) + (1/C2) + (1/B3) + (1/C3))",
                "=EXP(B10 - 1.96 * B11)",
                "=EXP(B10 + 1.96 * B11)",
                "=2 * (1 - NORM.S.DIST(ABS(B10 / B11), TRUE))"
            ],
            "Calculated Value": [
                f"{calcs['odds_cases']:.4f}",
                f"{calcs['odds_controls']:.4f}",
                f"{calcs['or_point']:.3f}",
                f"{calcs['odds_reduction_pct']:.1f}%",
                f"{calcs['ln_or']:.4f}",
                f"{calcs['se_ln_or']:.4f}",
                f"{calcs['or_ci_lower']:.3f}",
                f"{calcs['or_ci_upper']:.3f}",
                f"{calcs['p_value_wald']:.4e}"
            ]
        })
        st.dataframe(df_summary_cells, use_container_width=True, hide_index=True)

    with st.expander("📌 Step 2: Calculate Exposure Odds & Cross-Product Odds Ratio in Excel", expanded=False):
        st.markdown("""
        1. **Compute Exposure Odds in Cases:** In cell `B6`, enter `=B2/B3` (divides exposed cases $a$ by unexposed cases $c$).
        2. **Compute Exposure Odds in Controls:** In cell `B7`, enter `=C2/C3` (divides exposed controls $b$ by unexposed controls $d$).
        3. **Compute Odds Ratio (Cross-Product Ratio):** In cell `B8`, enter:
           ```excel
           =(B2*C3)/(C2*B3)
           ```
           *(Alternative equivalent formula: `=B6/B7`)*
        4. **Compute Percentage Odds Reduction (if protective):** In cell `B9`, enter `=IF(B8<1, (1-B8)*100, 0)`.
        """)

    with st.expander("📌 Step 3: Compute Woolf's Standard Error & 95% Wald CI in Excel", expanded=False):
        st.markdown("""
        Because ratios are skewed and bounded by zero, confidence intervals are computed on the natural log scale and exponentiated back:

        1. **Compute Natural Log of Odds Ratio:** In cell `B10`, enter:
           ```excel
           =LN(B8)
           ```
        2. **Compute Woolf's Standard Error of $\ln(OR)$:** In cell `B11`, enter Woolf's famous reciprocal sum formula:
           ```excel
           =SQRT((1/B2) + (1/C2) + (1/B3) + (1/C3))
           ```
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


def render_odds_ratio_python_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 4-step Python Implementation Workflow with docstrings and conditional prints."""
    st.markdown("### 🐍 Python Implementation Workflow for Odds Ratio")
    st.markdown("This guide follows a strict 4-step sequence: **Evaluate Prerequisites & Biases** ➔ **Calculate Exposure Odds & Point Estimates** ➔ **Calculate Woolf's Standard Error & Wald Confidence Intervals** ➔ **Consolidated Executable Pipeline Script**.")

    a, b, c, d = calcs["a"], calcs["b"], calcs["c"], calcs["d"]

    with st.expander("📌 Step 1: Evaluate Prerequisites & Study Design in Python", expanded=False):
        st.code('''import numpy as np

def check_odds_ratio_prerequisites(
    study_design: str,
    a: int,
    b: int,
    c: int,
    d: int,
    control_sampling_type: str = "population",
    disease_is_rare: bool = True,
    is_recall_bias_risk: bool = False
) -> dict:
    """Verifies epidemiological criteria and methodological biases for Odds Ratio.

    Args:
        study_design (str): Must be 'case_control' or 'cross_sectional'. 
            (If prospective cohort, Relative Risk can also be directly computed).
        a (int): Count of exposed diseased cases.
        b (int): Count of exposed healthy controls.
        c (int): Count of unexposed diseased cases.
        d (int): Count of unexposed healthy controls.
        control_sampling_type (str, optional): 'population' or 'hospital'. Defaults to 'population'.
        disease_is_rare (bool, optional): Whether disease prevalence is < 5-10%. Defaults to True.
        is_recall_bias_risk (bool, optional): Whether exposure is self-reported memory. Defaults to False.

    Returns:
        dict: Diagnostic pass/warning/fail flags and recommended actions.
    """
    diagnostics = []

    # 1. Study design check
    if study_design.lower() in {"case_control", "cross_sectional"}:
        print("[ASSURANCE - PASS] Study design is outcome-based. Odds Ratio is the mathematically valid measure.")
    else:
        print("[WARNING] Prospective cohort detected. Relative Risk (RR) can be directly calculated.")

    # 2. Rare disease assumption check
    if disease_is_rare:
        print("[ASSURANCE - PASS] Rare disease assumption met (< 5-10% incidence). OR closely approximates Relative Risk (OR ≈ RR).")
    else:
        print("[WARNING] Common disease (> 10%). OR exaggerates effect size. Do not interpret as Relative Risk.")

    # 3. Control selection check (Berkson's bias)
    if control_sampling_type.lower() == "hospital":
        print("[WARNING] Hospital controls used. Check that control admission diagnosis is strictly independent of exposure (Berkson's bias).")
    else:
        print("[ASSURANCE - PASS] Population-based controls reflect source community baseline.")

    # 4. Cell sparsity check
    min_cell = min(a, b, c, d)
    if min_cell == 0:
        print("[WARNING] Zero cell detected. Haldane-Anscombe continuity correction (+0.5) must be applied.")
    elif min_cell < 5:
        print("[WARNING] Sparse cell counts (< 5). Cross-reference Wald CI with Fisher's exact test.")
    else:
        print("[ASSURANCE - PASS] Cell counts robust (all cells >= 5). Asymptotic Woolf SE is reliable.")

    return {"min_cell": min_cell, "status": "EVALUATED"}
''', language="python")

    with st.expander("📌 Step 2: Calculate Exposure Odds & Cross-Product Odds Ratio in Python", expanded=False):
        st.code('''def calculate_odds_ratio_point_estimate(
    a: int,
    b: int,
    c: int,
    d: int,
    continuity_correction: bool = False
) -> dict:
    """Calculates exposure odds in cases and controls and the cross-product Odds Ratio.

    Args:
        a (int): Exposed cases.
        b (int): Exposed controls.
        c (int): Unexposed cases.
        d (int): Unexposed controls.
        continuity_correction (bool, optional): Apply 0.5 addition if cell is zero. Defaults to False.

    Returns:
        dict: Dictionary containing:
            - 'odds_cases': Exposure odds in cases (a / c).
            - 'odds_controls': Exposure odds in controls (b / d).
            - 'or_point': Cross-product Odds Ratio (a * d) / (b * c).
            - 'effect_direction': 'harmful', 'protective', or 'null'.
    """
    applied_cc = continuity_correction or (a == 0 or b == 0 or c == 0 or d == 0)
    a_c = float(a) + (0.5 if applied_cc else 0.0)
    b_c = float(b) + (0.5 if applied_cc else 0.0)
    c_c = float(c) + (0.5 if applied_cc else 0.0)
    d_c = float(d) + (0.5 if applied_cc else 0.0)

    odds_cases = a_c / c_c if c_c > 0 else np.nan
    odds_controls = b_c / d_c if d_c > 0 else np.nan
    or_point = (a_c * d_c) / (b_c * c_c) if (b_c > 0 and c_c > 0) else np.nan

    direction = "harmful" if or_point > 1.0001 else ("protective" if or_point < 0.9999 else "null")

    return {
        "odds_cases": odds_cases,
        "odds_controls": odds_controls,
        "or_point": or_point,
        "effect_direction": direction
    }
''', language="python")

    with st.expander("📌 Step 3: Compute Woolf's Standard Error & Wald 95% CI in Python", expanded=False):
        st.code('''from scipy import stats

def compute_odds_ratio_confidence_interval(
    a: int,
    b: int,
    c: int,
    d: int,
    alpha: float = 0.05
) -> dict:
    """Computes Woolf's log-scale Standard Error, Wald 95% CI, and Wald hypothesis test.

    Args:
        a (int): Exposed cases.
        b (int): Exposed controls.
        c (int): Unexposed cases.
        d (int): Unexposed controls.
        alpha (float, optional): Significance level. Defaults to 0.05 (95% CI).

    Returns:
        dict: ln_or, se_ln_or, ci_lower, ci_upper, z_score, p_value_wald.
    """
    # 0.5 correction if zero
    cc = 0.5 if min(a, b, c, d) == 0 else 0.0
    a_c, b_c, c_c, d_c = a + cc, b + cc, c + cc, d + cc

    or_val = (a_c * d_c) / (b_c * c_c)
    ln_or = np.log(or_val)

    # Woolf's standard error formula: sqrt(1/a + 1/b + 1/c + 1/d)
    se_ln_or = np.sqrt((1.0 / a_c) + (1.0 / b_c) + (1.0 / c_c) + (1.0 / d_c))

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_lower = np.exp(ln_or - z_crit * se_ln_or)
    ci_upper = np.exp(ln_or + z_crit * se_ln_or)

    z_score = ln_or / se_ln_or
    p_value_wald = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return {
        "ln_or": ln_or,
        "se_ln_or": se_ln_or,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "z_score": z_score,
        "p_value_wald": p_value_wald
    }
''', language="python")

    with st.expander("📌 Step 4: Consolidated Executable Pipeline Script", expanded=False):
        st.code(f'''import numpy as np
from scipy import stats

def run_odds_ratio_pipeline(
    a: int,
    b: int,
    c: int,
    d: int,
    study_design: str = "case_control",
    control_sampling_type: str = "population",
    disease_is_rare: bool = True
) -> dict:
    """Executes end-to-end Case-Control Odds Ratio analysis with bias diagnostics.

    Args:
        a (int): Count of exposed diseased cases.
        b (int): Count of exposed healthy controls.
        c (int): Count of unexposed diseased cases.
        d (int): Count of unexposed healthy controls.
        study_design (str, optional): Study architecture. Defaults to 'case_control'.
        control_sampling_type (str, optional): 'population' or 'hospital'. Defaults to 'population'.
        disease_is_rare (bool, optional): Rarity flag for RR approximation. Defaults to True.

    Returns:
        dict: Complete results dictionary with point estimates, CIs, and test statistics.
    """
    print("=" * 60)
    print("EPIDEMIOLOGICAL ODDS RATIO (OR) PIPELINE")
    print("=" * 60)

    # Step 1: Diagnostics
    if study_design.lower() in {{"case_control", "cross_sectional"}}:
        print("[ASSURANCE - PASS] Study design is outcome-based. Odds Ratio is valid.")
    else:
        print("[WARNING] Cohort design detected. Relative Risk can be calculated directly.")

    if disease_is_rare:
        print("[ASSURANCE - PASS] Rare disease assumption satisfied (OR ≈ RR).")
    else:
        print("[WARNING] Common disease (> 10%). OR exaggerates relative risk.")

    min_cell = min(a, b, c, d)
    applied_cc = (min_cell == 0)
    if applied_cc:
        print("[WARNING] Zero cell detected. Haldane-Anscombe (+0.5) applied.")
    elif min_cell < 5:
        print(f"[WARNING] Minimum cell is {{min_cell}} (< 5). Cross-reference with Fisher exact test.")
    else:
        print(f"[ASSURANCE - PASS] Minimum cell count is {{min_cell}} (>= 5). Asymptotic SE robust.")

    # Step 2: Point Estimates
    a_c = float(a) + (0.5 if applied_cc else 0.0)
    b_c = float(b) + (0.5 if applied_cc else 0.0)
    c_c = float(c) + (0.5 if applied_cc else 0.0)
    d_c = float(d) + (0.5 if applied_cc else 0.0)

    odds_cases = a_c / c_c
    odds_controls = b_c / d_c
    or_point = (a_c * d_c) / (b_c * c_c)

    # Step 3: Woolf Standard Error & Wald 95% CI
    ln_or = np.log(or_point)
    se_ln_or = np.sqrt(1.0/a_c + 1.0/b_c + 1.0/c_c + 1.0/d_c)
    z_crit = 1.95996
    ci_lower = np.exp(ln_or - z_crit * se_ln_or)
    ci_upper = np.exp(ln_or + z_crit * se_ln_or)

    z_score = ln_or / se_ln_or
    p_wald = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    # Step 4: Classical Fisher's Exact Test
    table = np.array([[a, b], [c, d]])
    _, p_fisher = stats.fisher_exact(table)

    print("\\n--- RESULTS ---")
    print(f"Exposure Odds in Cases:    {{odds_cases:.4f}}")
    print(f"Exposure Odds in Controls: {{odds_controls:.4f}}")
    print(f"Odds Ratio (OR):           {{or_point:.3f}}")
    print(f"95% Wald CI:               [{{ci_lower:.3f}}, {{ci_upper:.3f}}]")
    print(f"Wald Test p-value:         {{p_wald:.4e}} (z = {{z_score:.2f}})")
    print(f"Fisher's Exact Test p-val: {{p_fisher:.4e}}")

    return {{
        "or_point": or_point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value_wald": p_wald,
        "p_value_fisher": p_fisher
    }}

# Execute with scenario data ({meta['title']}):
if __name__ == "__main__":
    results = run_odds_ratio_pipeline(
        a={a}, b={b}, c={c}, d={d},
        study_design="case_control",
        control_sampling_type="{meta.get('control_sampling_type', 'population')}",
        disease_is_rare={meta.get('disease_is_rare', True)}
    )
''', language="python")


def render_odds_ratio_download_tab(
    df_contingency: pd.DataFrame,
    df_microdata: pd.DataFrame,
    meta: Dict[str, Any]
):
    """Renders the data download tab with individual microdata and 2x2 contingency table CSVs."""
    st.markdown("### 📥 Download Sample Case-Control Datasets")
    st.markdown(f"""
    Download the sample data associated with the active scenario: **{meta['title']}**.
    You can use these datasets to practice calculating Odds Ratios, Woolf's log standard errors, and Wald confidence intervals 
    in Microsoft Excel or with Python scripts.
    """)

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.markdown("#### 1. Individual Participant Microdata (CSV)")
        st.markdown(f"Contains **{len(df_microdata):,} participant records** with columns: `Participant_ID`, `Study_Group` (Case/Control), `Group_Binary`, `Exposure_Status`, `Exposure_Binary`, `Age`, `Sex`.")
        csv_micro = df_microdata.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Case-Control Participant Microdata (CSV)",
            data=csv_micro,
            file_name=f"{meta['scenario_key']}_case_control_microdata.csv",
            mime="text/csv",
            key="btn_dl_or_microdata"
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
            key="btn_dl_or_contingency"
        )
        st.dataframe(df_contingency, height=220, use_container_width=True, hide_index=True)


def render_odds_ratio_module(active_method_key: str, method_info: Dict[str, Any]):
    """Renders the comprehensive Odds Ratio (OR) module for Case-Control studies."""
    # 1. Plain English Guide
    render_odds_ratio_plain_english_guide()

    # 2. Sidebar Controls
    st.sidebar.header("🧪 Case-Control Study Settings")
    st.sidebar.markdown("**Study Scenarios & Contingency Parameters:**")

    or_scenario_choice = st.sidebar.selectbox(
        "Study Scenario:",
        options=["mesothelioma_asbestos", "outbreak_investigation", "cardiovascular_diet", "patient_feedback_survey", "custom"],
        format_func=lambda s: {
            "mesothelioma_asbestos": "🫁 Regional Mesothelioma & Historical Asbestos (Rare Disease, Harmful OR > 1)",
            "outbreak_investigation": "🧀 Food Festival Salmonella Outbreak (Acute Field Epi, Harmful OR > 1)",
            "cardiovascular_diet": "🥗 Myocardial Infarction & Mediterranean Diet (Protective OR < 1, Hospital Controls)",
            "patient_feedback_survey": "📋 NHS Patient Experience Survey (Prevalence Odds Ratio in Survey Data)",
            "custom": "⚙️ Custom 2×2 Table Input (Enter Your Own Counts)"
        }[s],
        index=0,
        help="Select a realistic public health scenario or enter custom 2x2 contingency table counts."
    )
    st.sidebar.caption("⚠️ **Data Notice:** Scenarios, populations, and event counts are made-up examples synthetically simulated for educational and illustrative purposes only.")

    if or_scenario_choice == "custom":
        st.sidebar.subheader("Custom 2×2 Parameters")
        custom_study_design = st.sidebar.selectbox(
            "Study Design Architecture:",
            options=["case_control", "cross_sectional", "prospective_cohort", "retrospective_cohort"],
            format_func=lambda d: {
                "case_control": "Case-Control (Outcome-first, Retrospective)",
                "cross_sectional": "Cross-Sectional (Prevalence snapshot)",
                "prospective_cohort": "Prospective Cohort (Forward-looking)",
                "retrospective_cohort": "Retrospective Cohort (Historical records)"
            }[d],
            index=0
        )
        custom_control_sampling = st.sidebar.selectbox(
            "Control Group Sampling Source:",
            options=["population", "hospital"],
            format_func=lambda c: {
                "population": "Community / Population Matched (Low selection bias risk)",
                "hospital": "Hospital / Clinic Controls (Risk of Berkson's bias)"
            }[c],
            index=0,
            help="Hospital controls are convenient but introduce Berkson's bias if the control illness is also linked to the exposure."
        )
        custom_disease_is_rare = st.sidebar.checkbox(
            "Is Disease Rare in Source Population (< 5–10%)?",
            value=True,
            help="If checked, the Rare Disease Assumption holds and OR closely approximates Relative Risk (OR ≈ RR)."
        )
        custom_is_recall_bias_risk = st.sidebar.checkbox(
            "Exposure Gathered via Retrospective Recall?",
            value=False,
            help="Self-reported exposure histories frequently cause recall bias because ill cases ruminate on past habits more than healthy controls."
        )

        col_sb1, col_sb2 = st.sidebar.columns(2)
        with col_sb1:
            a_val = st.number_input("Exposed Cases (a):", min_value=0, max_value=100000, value=50, step=1)
            c_val = st.number_input("Unexposed Cases (c):", min_value=0, max_value=100000, value=25, step=1)
        with col_sb2:
            b_val = st.number_input("Exposed Controls (b):", min_value=0, max_value=100000, value=30, step=1)
            d_val = st.number_input("Unexposed Controls (d):", min_value=0, max_value=100000, value=120, step=5)

        meta = {
            "title": "Custom User-Defined Case-Control Investigation",
            "domain": "Epidemiological Hypothesis Testing",
            "description": "User-configured 2x2 contingency Case-Control evaluation.",
            "exposure_name": "Exposure Factor",
            "exposure_levels": ("Exposed", "Unexposed"),
            "outcome_name": "Target Condition",
            "outcome_levels": ("Cases (Condition +)", "Controls (Condition -)"),
            "expected_effect": "User Defined",
            "clinical_takeaway": "Custom parameters evaluated based on user-entered contingency frequencies.",
            "scenario_key": "custom_case_control",
            "control_sampling_type": custom_control_sampling,
            "disease_is_rare": custom_disease_is_rare,
            "is_recall_bias_risk": custom_is_recall_bias_risk
        }

        # Build contingency and microdata
        df_contingency = pd.DataFrame([
            {
                "Exposure_Status": "🟢 Exposed",
                "Cases_Outcome_Pos": a_val,
                "Controls_Outcome_Neg": b_val,
                "Total_Exposed": a_val + b_val,
                "Exposure_Percentage": float(a_val / (a_val + b_val) * 100.0) if (a_val + b_val) > 0 else 0.0
            },
            {
                "Exposure_Status": "⚪ Unexposed",
                "Cases_Outcome_Pos": c_val,
                "Controls_Outcome_Neg": d_val,
                "Total_Unexposed": c_val + d_val,
                "Exposure_Percentage": float(c_val / (c_val + d_val) * 100.0) if (c_val + d_val) > 0 else 0.0
            },
            {
                "Exposure_Status": "📊 Total (Marginals)",
                "Cases_Outcome_Pos": a_val + c_val,
                "Controls_Outcome_Neg": b_val + d_val,
                "Total_Exposed": a_val + b_val + c_val + d_val,
                "Exposure_Percentage": 100.0
            }
        ])

        micro_list = []
        for i in range(min(a_val, 100)):
            micro_list.append({"Participant_ID": f"CASE_{i+1:04d}", "Study_Group": "Case", "Group_Binary": 1, "Exposure_Status": "Exposed", "Exposure_Binary": 1, "Age": 62, "Sex": "Male"})
        for i in range(min(c_val, 100)):
            micro_list.append({"Participant_ID": f"CASE_{a_val+i+1:04d}", "Study_Group": "Case", "Group_Binary": 1, "Exposure_Status": "Unexposed", "Exposure_Binary": 0, "Age": 63, "Sex": "Female"})
        for i in range(min(b_val, 100)):
            micro_list.append({"Participant_ID": f"CTRL_{i+1:04d}", "Study_Group": "Control", "Group_Binary": 0, "Exposure_Status": "Exposed", "Exposure_Binary": 1, "Age": 61, "Sex": "Male"})
        for i in range(min(d_val, 100)):
            micro_list.append({"Participant_ID": f"CTRL_{b_val+i+1:04d}", "Study_Group": "Control", "Group_Binary": 0, "Exposure_Status": "Unexposed", "Exposure_Binary": 0, "Age": 62, "Sex": "Female"})
        df_microdata = pd.DataFrame(micro_list)

        calcs = calculate_odds_ratio(a_val, b_val, c_val, d_val)
        evaluations = evaluate_odds_ratio_prerequisites(
            study_design=custom_study_design,
            a=a_val, b=b_val, c=c_val, d=d_val,
            control_sampling_type=custom_control_sampling,
            disease_is_rare=custom_disease_is_rare,
            is_recall_bias_risk=custom_is_recall_bias_risk
        )
    else:
        df_contingency, df_microdata, meta = generate_odds_ratio_demo_data(scenario=or_scenario_choice, seed=42)
        calcs = meta["calcs"]

        active_design = "cross_sectional" if or_scenario_choice == "patient_feedback_survey" else "case_control"
        evaluations = evaluate_odds_ratio_prerequisites(
            study_design=active_design,
            a=meta["a"], b=meta["b"], c=meta["c"], d=meta["d"],
            control_sampling_type=meta.get("control_sampling_type", "population"),
            disease_is_rare=meta.get("disease_is_rare", True),
            is_recall_bias_risk=meta.get("is_recall_bias_risk", False)
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
            ⚠️ <strong>Disclaimer:</strong> This is a completely made-up example scenario with synthetic data simulated for educational and illustrative purposes only. It does not represent actual patient records or genuine disease registries from NHS trusts or healthcare providers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if meta.get("scenario_key") == "patient_feedback_survey":
        st.info("""
        💡 **Methodological Note on Naming Conventions in Survey Research:**  
        In customer, patient, and cross-sectional survey research, the Odds Ratio is known as the **Prevalence Odds Ratio (POR)**. 
        **The underlying mathematical formulas and cross-product calculations are 100% identical!**  
        ⚠️ **The Common Outcome Warning:** Because positive feedback is common (80% vs 60%), the Rare Disease Assumption does not hold. 
        This causes the Prevalence Odds Ratio ($POR = 2.67$) to appear more than double the Prevalence Ratio ($PR = 1.33$), 
        even though young people are only 33% more likely to leave a positive review!
        """)


    # 1. Tri-Color Diagnostics
    render_odds_ratio_diagnostics(evaluations)

    # 2. Metric Dashboard
    render_odds_ratio_dashboard(calcs, meta)

    # 3. Visualizations & 2x2 Grid
    render_odds_ratio_visualizations(calcs, meta)

    # 4. Methodology Tabs
    st.markdown("---")
    tab_or_rationale, tab_or_excel, tab_or_code, tab_or_download = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Excel Implementation Guide",
        "🐍 Python Code Implementation",
        "📥 Download Sample Data"
    ])

    with tab_or_rationale:
        st.subheader(f"Why Odds Ratio (OR)? ({method_info['name']})")
        st.markdown("""
        When researching health conditions that are rare, have long latency periods (e.g. cancers developing 30 years after chemical exposure), 
        or when evaluating sudden acute outbreaks, following a healthy cohort forward in time is often impossible or unethical. 
        Epidemiologists turn to **retrospective Case-Control studies** and compute the **Odds Ratio**.
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
        st.markdown("### 🔍 In-Depth Epidemiological Framework: Odds Ratio vs. Relative Risk vs. Risk Difference")
        st.markdown(r"""
        | Epidemiological Dimension | Odds Ratio (OR) | Relative Risk (Risk Ratio, RR) | Absolute Risk Difference (RD) |
        | :--- | :--- | :--- | :--- |
        | **Mathematical Formula** | $\frac{a \times d}{b \times c} = \frac{a/c}{b/d}$ | $\frac{a / (a+b)}{c / (c+d)} = \frac{I_e}{I_u}$ | $I_e - I_u = \frac{a}{a+b} - \frac{c}{c+d}$ |
        | **Primary Study Design** | Retrospective Case-Control, Cross-Sectional | Prospective Cohort, Randomized Controlled Trial | Prospective Cohort, Randomized Controlled Trial |
        | **Core Question Asked** | How many times higher were the odds of past exposure in diseased cases compared to healthy controls? | How many times higher is the cumulative risk of falling ill in exposed people over time? | How many additional (or fewer) cases occur per 100 people due to the exposure? |
        | **Can Incidence Be Calculated?** | **No.** Total cases ($a+c$) and controls ($b+d$) are fixed by researcher recruitment quotas, not natural population risk. | **Yes.** Cohort denominators ($a+b$ and $c+d$) are known, allowing true cumulative incidence ($I_e, I_u$). | **Yes.** True baseline risks are directly subtracted ($I_e - I_u$). |
        | **When Do They Agree?** | **Rare Disease Assumption:** When disease incidence is $< 5\text{--}10\%$, $OR \approx RR$. | Reference standard for prospective follow-up studies. | Reference standard for clinical workload, public health impact, and NNT/NNH. |
        | **When Disease Is Common (> 10%)** | $OR$ **exaggerates** effect size ($OR > RR$ for harmful exposures; $OR < RR$ for protective factors). | Remains true ratio of cumulative incidences. | Remains true difference in percentage points. |
        | **Null Benchmark Value** | $1.00$ (Parity in exposure odds) | $1.00$ (Parity in disease risk) | $0.00$ (Zero difference in risk) |
        """)

        st.markdown("#### 📖 Key Methodological Concepts Explained (Zero Prior Knowledge Assumed)")
        st.markdown(r"""
        1. **Why Can't We Calculate Relative Risk in a Case-Control Study?**
           * In a clinical trial or cohort study, if 100 people take a drug and 5 get sick, their risk is $5\%$.
           * But in a Case-Control study, the researcher decides in advance: *"I will recruit 50 lung cancer patients from the hospital, and 50 healthy people from the GP register."*
           * If you tried to calculate disease risk from that table, you would get $50 / 100 = 50\%$. But lung cancer does not affect $50\%$ of the population! That $50\%$ figure is purely an artifact of how many people the researcher chose to interview.
           * Because you do not know the true total population denominator, **you cannot calculate risk or Relative Risk**. However, the ratio of exposure odds ($\frac{a/c}{b/d}$) mathematically equals the disease odds ratio ($\frac{ad}{bc}$) and is completely independent of the control-to-case sampling ratio!

        2. **Berkson's Selection Bias (The Danger of Hospital Controls):**
           * If you recruit controls from a hospital instead of the community (e.g. patients in the orthopedic ward), beware: patients in hospitals often have multiple medical conditions linked to smoking, diet, or alcohol.
           * If the exposure you are studying (e.g. smoking) also causes hospital admissions for other conditions, your controls will have artificially high smoking rates! This artificially lowers the Odds Ratio, masking a real danger.
           * **Best Practice:** Select controls from the general community, or ensure hospital controls have diagnoses that are completely unrelated to the exposure.

        3. **Recall / Information Bias:**
           * When someone is diagnosed with a severe illness (like cancer or a child born with a birth defect), they often spend weeks agonizing over why it happened and remember every detail of past chemical or dietary exposure.
           * Healthy controls, on the other hand, frequently forget past exposures because they had no reason to dwell on them.
           * This asymmetric memory creates **Recall Bias**, which artificially inflates the Odds Ratio.
           * **Best Practice:** Verify past exposures using objective historical records (e.g. employment payrolls, prescription databases, atmospheric monitoring) rather than relying solely on memory questionnaires.

        4. **Why the Odds Ratio Rules Modern Medical Statistics (Logistic Regression):**
           * You will see Odds Ratios in almost every medical journal, even in cohort studies and clinical trials!
           * Why? Because when researchers adjust for multiple confounding variables (like age, sex, smoking, and blood pressure) using **Multivariable Binary Logistic Regression**, the mathematical output of the model equation is:
             $$\ln\left(\frac{P}{1 - P}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots$$
           * When you take the exponent ($\exp(\beta_1)$) of a logistic regression coefficient, it yields an **Adjusted Odds Ratio (aOR)**!
           * Understanding the Odds Ratio is therefore your passport to understanding virtually all multivariable medical and epidemiological research.
        """)

    with tab_or_excel:
        render_odds_ratio_excel_guide(calcs, meta)

    with tab_or_code:
        render_odds_ratio_python_guide(calcs, meta)

    with tab_or_download:
        render_odds_ratio_download_tab(df_contingency, df_microdata, meta)


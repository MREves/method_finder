"""Streamlit UI Components & Visualizations for Predictive Values (PPV & NPV) and Bayes' Theorem.

Provides:
1. render_predictive_values_plain_english_guide: Zero-prior-knowledge pedagogical guide with Needle-in-a-Haystack analogy, False Positive Paradox, and Bayes' Theorem.
2. render_predictive_values_diagnostics: Tri-color evaluation cards (PASS, WARNING, FAIL) for prevalence generalizability, work-up bias, incorporation bias, and test marginal counts.
3. render_predictive_values_dashboard: Primary metric cards (Positive Predictive Value, Negative Predictive Value, Prevalence, False Alarms per True Case) with 95% CIs.
4. render_predictive_values_visualizations: Side-by-side Plotly Bayes Prevalence Curve, Population Flow Bar breakdown, balanced explanatory containers, and 2x2 matrix display.
5. render_predictive_values_prevalence_simulator: Interactive live slider allowing instant simulation of Bayes' Theorem across any target prevalence.
6. render_predictive_values_excel_guide: 4-step sequence with cell mapping, explicit formulas, Bayes formula, and Excel charting steps.
7. render_predictive_values_python_guide: 5-step sequence with direct math, Bayes formulation, Mercaldo CIs, consolidated pipeline, and Plotly scripts.
8. render_predictive_values_download_tab: Live CSV downloads for patient microdata and 2x2 predictive matrix.
9. render_predictive_values_module: Top-level controller for Predictive Values & Bayes' Theorem Screening Evaluation.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.predictive_values import (
    calculate_predictive_values,
    evaluate_predictive_value_prerequisites,
    generate_predictive_values_demo_data
)


def render_predictive_values_plain_english_guide():
    """Renders the zero-prior-knowledge pedagogical guide for PPV, NPV, and Bayes' Theorem."""
    with st.expander("💡 Plain English Guide: Predictive Values (PPV & NPV) & The False Positive Paradox", expanded=False):
        st.markdown("""
        In clinical medicine, when a patient takes a diagnostic screening test, they do not ask:
        *"Given that I have the disease, what was the chance my test was positive?"* (**Sensitivity**).
        
        The patient and their doctor ask the reverse conditional question:
        > **"Doctor, my test result came back POSITIVE. What is the actual real-world probability that I truly have the disease?"**
        
        This real-world probability is the **Positive Predictive Value (PPV)**. Crucially, **Predictive Values are NOT fixed properties of the test** — they depend dramatically on the **prevalence of the disease** in the group being tested!
        """)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. Positive Predictive Value (PPV, Post-Test Probability of Disease)
            * **Definition:** The probability that a patient who **tests positive** actually **has the disease**.
            * **Formula (Direct Sample):**
              $$\text{PPV} = P(\text{Disease}+ \mid \text{Test}+) = \frac{\text{True Positives}}{\text{Total Test Positives}} = \frac{TP}{TP + FP}$$
            * **False Discovery Rate (FDR):**
              $$\text{FDR} = 1 - \text{PPV} = \frac{FP}{TP + FP}$$
              The proportion of all positive test results that are **false alarms**.
            * **Why Prevalence Dictates PPV (Bayes' Theorem):**
              $$\text{PPV} = \frac{\text{Sensitivity} \times \text{Prevalence}}{\text{Sensitivity} \times \text{Prevalence} + (1 - \text{Specificity}) \times (1 - \text{Prevalence})}$$
              * When a disease is **rare** in the community (e.g. $0.1\%$), healthy people vastly outnumber sick people. Even a tiny false-positive rate generates more false alarms than true cases!
              * When a disease is **common** (e.g. in a hospital chest pain ward where $30\%$ have heart attacks), a positive test is overwhelmingly likely to be a true positive.
            """)

            st.markdown(r"""
            #### 2. Negative Predictive Value (NPV, Post-Test Probability of Health)
            * **Definition:** The probability that a patient who **tests negative** is truly **disease-free**.
            * **Formula (Direct Sample):**
              $$\text{NPV} = P(\text{Disease}- \mid \text{Test}-) = \frac{\text{True Negatives}}{\text{Total Test Negatives}} = \frac{TN}{FN + TN}$$
            * **False Omission Rate (FOR):**
              $$\text{FOR} = 1 - \text{NPV} = \frac{FN}{FN + TN}$$
              The proportion of all negative test results that are **missed cases**.
            * **Clinical Role:** High NPV provides clinical safety and reassurance, confirming that a patient can safely be discharged without invasive workup.
            """)

        with col_g2:
            st.markdown(r"""
            #### 3. The Core Analogy: "The Needle in a Haystack (The False Positive Paradox)"
            * Imagine a rare blood condition affecting **1 in 1,000 people (Prevalence = 0.1%)**.
            * You invent an extraordinary new test with **99% Sensitivity** and **99% Specificity** (only a 1% false positive rate).
            * Now screen **100,000 citizens** across the population:
              * **100 people are truly sick:** The test catches 99 of them ($TP = 99$), misses 1 ($FN = 1$).
              * **99,900 people are completely healthy:** The test clears 98,901, but the 1% error triggers on healthy people: $0.01 \times 99,900 = \mathbf{999 \text{ False Positives}}$!
            * **The Shocking Result:**
              * Total positive results: $99 + 999 = 1,098$.
              * **Positive Predictive Value:** $\frac{99}{1,098} \approx \mathbf{9.0\%}$!
              * **Over 90% of everyone who tested positive is completely healthy!**
            * That is the **False Positive Paradox**: in rare diseases, false alarms outnumber true cases because the haystack of healthy people is massive compared to the tiny needle of diseased cases.
            """)

            st.markdown(r"""
            #### 4. The Public Health Policy Rule: "Targeted vs. Mass Screening"
            * **Why Universal Whole-Body Scans are Dangerous:**
              * Ordering full-body MRI or rare tumor marker tests on asymptomatic 25-year-olds causes tremendous harm: thousands of false alarms, panic, invasive biopsies, and surgical complications for non-existent diseases.
            * **The Clinical Solution (Pre-Test Probability Enrichment):**
              * Only apply screening tests to **high-risk, symptomatic, or enriched demographic cohorts** (e.g. heavy smokers over 55, or patients presenting with chest pain in A&E).
              * By raising the pre-test probability from $0.1\%$ to $25\%$, the exact same test's **PPV leaps from $9\%$ to over $80\%$**!
            """)

        st.info(r"""
        📐 **Summary Comparison:**  
        * **Sensitivity & Specificity:** Intrinsic laboratory properties of the test instrument. Do not change with prevalence.  
        * **Positive & Negative Predictive Values:** Clinical realities for the patient. Scale dramatically with population disease prevalence.
        """)


def render_predictive_values_diagnostics(evaluations: List[Dict[str, Any]]):
    """Renders tri-color prerequisite diagnostic cards for Predictive Values."""
    st.markdown("### 🚦 Tri-Color Bias & Methodological Diagnostics (STARD & QUADAS-2)")
    st.caption("Verifies setting and prevalence generalizability, work-up / verification bias, reference standard independence, and marginal count stability.")

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


def render_predictive_values_dashboard(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders primary predictive value metric cards with 95% CIs and error rates."""
    st.markdown("### 📊 Primary Predictive Performance Measures")

    ppv_pct = calcs["ppv"] * 100.0
    ppv_ll = calcs["ppv_ci_lower"] * 100.0
    ppv_ul = calcs["ppv_ci_upper"] * 100.0

    npv_pct = calcs["npv"] * 100.0
    npv_ll = calcs["npv_ci_lower"] * 100.0
    npv_ul = calcs["npv_ci_upper"] * 100.0

    fdr_pct = calcs["fdr"] * 100.0
    for_pct = calcs["for_rate"] * 100.0
    eff_prev_pct = calcs["effective_prevalence"] * 100.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Positive Predictive Value (PPV)",
            value=f"{ppv_pct:.1f}%",
            delta=f"False Alarm Rate (FDR): {fdr_pct:.1f}%",
            delta_color="inverse",
            help="Post-Test Probability of Disease given Positive Test: TP / (TP + FP). Or derived via Bayes' Theorem using population prevalence."
        )
        st.caption(f"**95% CI:** [{ppv_ll:.1f}%, {ppv_ul:.1f}%]")

    with c2:
        st.metric(
            label="Negative Predictive Value (NPV)",
            value=f"{npv_pct:.1f}%",
            delta=f"Missed Disease Rate (FOR): {for_pct:.1f}%",
            delta_color="inverse",
            help="Post-Test Probability of No Disease given Negative Test: TN / (FN + TN). Or derived via Bayes' Theorem using population prevalence."
        )
        st.caption(f"**95% CI:** [{npv_ll:.1f}%, {npv_ul:.1f}%]")

    with c3:
        st.metric(
            label="Population Disease Prevalence",
            value=f"{eff_prev_pct:.2f}%",
            delta="Bayes Mode" if calcs["bayes_applied"] else "Sample Cohort",
            delta_color="off",
            help="Pre-test probability of disease in the screened population. Direct driver of Positive Predictive Value."
        )
        st.caption(f"**Sample Size N:** {calcs['total_sample']:,} patients")

    with c4:
        ratio_val = calcs["false_positives_per_true_positive"]
        ratio_str = f"{ratio_val:.1f} : 1" if not np.isnan(ratio_val) else "0.0 : 1"
        st.metric(
            label="False Alarms per True Case",
            value=ratio_str,
            delta=f"Sens: {calcs['sensitivity']*100.0:.1f}% | Spec: {calcs['specificity']*100.0:.1f}%",
            delta_color="off",
            help="Ratio of False Positives to True Positives (FP / TP). Illustrates the real-world clinical false alarm burden."
        )
        st.caption(f"**Total Positives:** {calcs['total_test_positive']:,} (TP={calcs['tp']:,}, FP={calcs['fp']:,})")


def render_predictive_values_visualizations(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders side-by-side Plotly Bayes Prevalence Curve and Population Flow Bar breakdown."""
    st.markdown("### 📈 Visualizing Predictive Values & The False Positive Paradox")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**1. The Bayes Prevalence Curve (PPV & NPV vs. Disease Prevalence)**")

        # Generate smooth prevalence values from 0.1% to 100%
        prev_range = np.linspace(0.001, 0.999, 150)
        sens = calcs["sensitivity"]
        spec = calcs["specificity"]

        # Bayes PPV & NPV curves
        ppv_curve = (sens * prev_range) / ((sens * prev_range) + ((1.0 - spec) * (1.0 - prev_range))) * 100.0
        npv_curve = (spec * (1.0 - prev_range)) / ((spec * (1.0 - prev_range)) + ((1.0 - sens) * prev_range)) * 100.0

        fig_bayes = go.Figure()
        fig_bayes.add_trace(go.Scatter(
            x=prev_range * 100.0,
            y=ppv_curve,
            mode="lines",
            name="Positive Predictive Value (PPV)",
            line=dict(color="#2563eb", width=2.5)
        ))
        fig_bayes.add_trace(go.Scatter(
            x=prev_range * 100.0,
            y=npv_curve,
            mode="lines",
            name="Negative Predictive Value (NPV)",
            line=dict(color="#10b981", width=2.5, dash="dash")
        ))

        # Add active scenario point
        active_prev = calcs["effective_prevalence"] * 100.0
        active_ppv = calcs["ppv"] * 100.0
        fig_bayes.add_trace(go.Scatter(
            x=[active_prev],
            y=[active_ppv],
            mode="markers+text",
            marker=dict(size=13, color="#ef4444", symbol="diamond"),
            text=[f"Current Setting<br>Prev={active_prev:.1f}%, PPV={active_ppv:.1f}%"],
            textposition="top left" if active_prev > 40 else "top right",
            name="Active Clinical Setting"
        ))

        fig_bayes.update_layout(
            xaxis_title="Population Disease Prevalence P(Disease+) (%)",
            yaxis_title="Post-Test Probability (%)",
            yaxis=dict(range=[0, 105]),
            height=340,
            margin=dict(l=30, r=30, t=30, b=30),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bayes, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### 🔍 What is the Bayes Prevalence Curve Showing?")
            st.markdown(
                "This curve reveals the profound mathematical relationship governing all clinical diagnostic testing:\n"
                f"* **Sensitivity ({sens*100.0:.1f}%) and Specificity ({spec*100.0:.1f}%) are Constant:** The test technology itself has not changed at all along this entire horizontal axis.\n"
                f"* **PPV is Non-Linear:** When disease prevalence is rare ($< 2\\%$), PPV collapses toward zero regardless of high specificity. As prevalence rises into symptomatic clinic levels ($> 20\\%$), PPV rapidly accelerates toward $80\\%-95\\%$.\n"
                f"* **NPV is Mirror Opposite:** Negative Predictive Value is near $100\\%$ in rare diseases, but declines when disease prevalence is extremely high.\n"
                f"* **Red Diamond Coordinate:** Your active investigation ($P(D+) = {active_prev:.2f}\\%$, yielding $\\text{{PPV}} = {active_ppv:.1f}\\%$)."
            )

    with col_chart2:
        st.markdown("**2. Test Outcome Flow: True Signals vs. False Alarms**")

        # Visual breakdown of test positives and test negatives
        tp = calcs["tp"]
        fp = calcs["fp"]
        fn = calcs["fn"]
        tn = calcs["tn"]

        categories = ["Positive Test Results (T+)", "Negative Test Results (T-)"]

        fig_breakdown = go.Figure()

        # Correct Classifications (TP in T+, TN in T-)
        fig_breakdown.add_trace(go.Bar(
            name="True Diagnostic Findings (TP & TN)",
            x=categories,
            y=[tp, tn],
            marker_color="#10b981",
            text=[f"TP: {tp:,}<br>({calcs['ppv']*100.0:.1f}%)", f"TN: {tn:,}<br>({calcs['npv']*100.0:.1f}%)"],
            textposition="auto"
        ))

        # Errors (FP in T+, FN in T-)
        fig_breakdown.add_trace(go.Bar(
            name="Diagnostic Errors (FP False Alarms & FN Misses)",
            x=categories,
            y=[fp, fn],
            marker_color="#ef4444",
            text=[f"FP: {fp:,}<br>({calcs['fdr']*100.0:.1f}%)" if fp > 0 else "", f"FN: {fn:,}<br>({calcs['for_rate']*100.0:.1f}%)" if fn > 0 else ""],
            textposition="auto"
        ))

        fig_breakdown.update_layout(
            barmode="stack",
            yaxis_title="Number of Patients in Sample",
            height=340,
            margin=dict(l=30, r=30, t=30, b=30),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### ⚖️ Understanding the False Positive Paradox")
            st.markdown(
                "This chart visualizes the real-world distribution of clinical test results:\n"
                f"* **Positive Test Results ($T+ = {tp + fp:,}$ patients):** Composed of **{tp:,}** True Positives ({calcs['ppv']*100.0:.1f}%) and **{fp:,}** False Alarms ({calcs['fdr']*100.0:.1f}%).\n"
                f"* **False Alarms Ratio:** For every true diseased patient detected, **{calcs['false_positives_per_true_positive']:.1f}** healthy individuals receive an abnormal call requiring further diagnostic workup.\n"
                f"* **Negative Test Results ($T- = {fn + tn:,}$ patients):** Overwhelmingly composed of **{tn:,}** cleared healthy individuals ({calcs['npv']*100.0:.1f}% NPV), with only **{fn:,}** missed cases ({calcs['for_rate']*100.0:.1f}% False Omission Rate).\n"
                f"* **Clinical Takeaway:** High NPV allows rapid rule-out, but low PPV in population screening necessitates confirmatory secondary testing before diagnosis."
            )

    # 3. Interactive Contingency Matrix Display (Row Predictive Marginals)
    st.markdown("**3. Diagnostic 2×2 Contingency Matrix with Row Marginals (Predictive View)**")
    test_pos_label, test_neg_label = meta.get("test_levels", ("Test Positive", "Test Negative"))
    cond_pos_label, cond_neg_label = meta.get("condition_levels", ("Disease Present (Gold +)", "Disease Absent (Gold -)"))

    df_2x2_display = pd.DataFrame([
        {
            "Test Outcome": f"🔴 {test_pos_label}",
            f"{cond_pos_label}": f"{tp:,} (True Positives, TP)",
            f"{cond_neg_label}": f"{fp:,} (False Positives, FP)",
            "Total Test Outcome": f"{tp + fp:,}",
            "Row Predictive Metric": f"PPV = {calcs['ppv']*100.0:.1f}% | FDR = {calcs['fdr']*100.0:.1f}% (False Alarms)"
        },
        {
            "Test Outcome": f"🟢 {test_neg_label}",
            f"{cond_pos_label}": f"{fn:,} (False Negatives, FN)",
            f"{cond_neg_label}": f"{tn:,} (True Negatives, TN)",
            "Total Test Outcome": f"{fn + tn:,}",
            "Row Predictive Metric": f"NPV = {calcs['npv']*100.0:.1f}% | FOR = {calcs['for_rate']*100.0:.1f}% (Miss Rate)"
        },
        {
            "Test Outcome": "📊 Total Cohort",
            f"{cond_pos_label}": f"{calcs['total_diseased']:,} (Diseased)",
            f"{cond_neg_label}": f"{calcs['total_healthy']:,} (Healthy)",
            "Total Test Outcome": f"{calcs['total_sample']:,} (Grand Total N)",
            "Row Predictive Metric": f"Disease Prevalence = {calcs['effective_prevalence']*100.0:.2f}%"
        }
    ])

    st.dataframe(df_2x2_display, use_container_width=True, hide_index=True)


def render_predictive_values_prevalence_simulator(calcs: Dict[str, Any]):
    """Renders an interactive live Bayes' Theorem simulator widget."""
    st.markdown("### 🎛️ Live Bayes' Theorem Prevalence Simulator")
    st.caption("Adjust disease prevalence below to observe how PPV and NPV transform in real time while Sensitivity and Specificity stay fixed.")

    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        sim_prev = st.slider(
            "Hypothetical Setting Prevalence (%):",
            min_value=0.1,
            max_value=50.0,
            value=float(np.clip(calcs["effective_prevalence"] * 100.0, 0.1, 50.0)),
            step=0.1,
            format="%.1f%%",
            help="Simulate testing in different settings: 0.5% (unselected population screening), 5.0% (primary care symptomatic), or 25.0% (hospital specialist referral clinic)."
        ) / 100.0

        sens = calcs["sensitivity"]
        spec = calcs["specificity"]

        # Bayes calculation
        p_t_pos = (sens * sim_prev) + ((1.0 - spec) * (1.0 - sim_prev))
        sim_ppv = (sens * sim_prev) / p_t_pos if p_t_pos > 0 else 0.0

        p_t_neg = (spec * (1.0 - sim_prev)) + ((1.0 - sens) * sim_prev)
        sim_npv = (spec * (1.0 - sim_prev)) / p_t_neg if p_t_neg > 0 else 0.0

    with sim_col2:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label="Simulated PPV",
                value=f"{sim_ppv * 100.0:.1f}%",
                delta=f"{((sim_ppv - calcs['ppv']) * 100.0):+.1f}% vs. active",
                help="Positive Predictive Value under simulated prevalence."
            )
        with m2:
            st.metric(
                label="Simulated NPV",
                value=f"{sim_npv * 100.0:.1f}%",
                delta=f"{((sim_npv - calcs['npv']) * 100.0):+.1f}% vs. active",
                help="Negative Predictive Value under simulated prevalence."
            )
        with m3:
            sim_ratio = (1.0 - sim_ppv) / sim_ppv if sim_ppv > 0 else 0.0
            st.metric(
                label="Simulated False Alarms per Case",
                value=f"{sim_ratio:.1f} : 1",
                help="Ratio of False Positives to True Positives under simulated prevalence."
            )


def render_predictive_values_excel_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 4-step Microsoft Excel Implementation Guide with cell coordinates and Bayes formulas."""
    st.markdown("### 📗 Microsoft Excel Implementation Workflow for Predictive Values & Bayes' Theorem")
    st.markdown("Follow the 4-step sequence: **Set up 2×2 Layout** ➔ **Calculate Direct PPV & NPV** ➔ **Apply Bayes' Theorem with External Prevalence** ➔ **Build the Bayes Prevalence Curve in Excel**.")

    tp, fp, fn, tn = calcs["tp"], calcs["fp"], calcs["fn"], calcs["tn"]
    total_pos = calcs["total_test_positive"]
    total_neg = calcs["total_test_negative"]
    total_d = calcs["total_diseased"]
    total_h = calcs["total_healthy"]
    total_n = calcs["total_sample"]

    test_pos_label, test_neg_label = meta.get("test_levels", ("Test Positive", "Test Negative"))
    cond_pos_label, cond_neg_label = meta.get("condition_levels", ("Condition Present", "Condition Absent"))

    with st.expander("📌 Step 1: Excel 2×2 Data Layout & Cell References", expanded=False):
        st.markdown(f"""
        Structure your Excel worksheet with the 2×2 table in `Cells A1:E4` and calculation outputs in `Cells A6:B16`:
        
        * **Column A (Rows 2:4):** Test Result (`Row 2: Positive`, `Row 3: Negative`, `Row 4: Total`)
        * **Column B (Rows 2:4):** Gold Standard Disease Present (`B2`: $TP = {tp}$, `B3`: $FN = {fn}$, `B4`: `=SUM(B2:B3)`)
        * **Column C (Rows 2:4):** Gold Standard Disease Absent (`C2`: $FP = {fp}$, `C3`: $TN = {tn}$, `C4`: `=SUM(C2:C3)`)
        * **Column D (Rows 2:4):** Row Marginals (`D2`: `=SUM(B2:C2)`, `D3`: `=SUM(B3:C3)`, `D4`: `=SUM(B4:C4)`)
        * **Column E (Rows 2:4):** Row Meaning (`E2`: `Total Test Positives`, `E3`: `Total Test Negatives`, `E4`: `Total Sample N`)
        """)

        preview_grid = pd.DataFrame([
            {"Excel_Row": "Row 1 (Headers)", "Col_A": "Test_Result", "Col_B": "Condition_Present_Gold_Pos", "Col_C": "Condition_Absent_Gold_Neg", "Col_D": "Test_Marginal_Total", "Col_E": "Classification_Role"},
            {"Excel_Row": "Row 2 (Test Positive)", "Col_A": test_pos_label, "Col_B": str(tp), "Col_C": str(fp), "Col_D": f"{total_pos} (=SUM(B2:C2))", "Col_E": "TP (a) & FP (b) -> PPV Row"},
            {"Excel_Row": "Row 3 (Test Negative)", "Col_A": test_neg_label, "Col_B": str(fn), "Col_C": str(tn), "Col_D": f"{total_neg} (=SUM(B3:C3))", "Col_E": "FN (c) & TN (d) -> NPV Row"},
            {"Excel_Row": "Row 4 (Totals)", "Col_A": "Total Cohort", "Col_B": f"{total_d} (=SUM(B2:B3))", "Col_C": f"{total_h} (=SUM(C2:C3))", "Col_D": f"{total_n} (=SUM(D2:D3))", "Col_E": "Grand Total Sample"}
        ])
        st.dataframe(preview_grid, use_container_width=True, hide_index=True)

        st.markdown("**Summary Output Cells Mapping (`Cells A6:B16`):**")
        df_summary_cells = pd.DataFrame({
            "Cell": ["B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15", "B16"],
            "Metric Name": [
                "Sample Disease Prevalence",
                "Sensitivity (TPR = B2/B4)",
                "Specificity (TNR = C3/C4)",
                "Direct Positive Predictive Value (PPV)",
                "False Discovery Rate (FDR = 1 - PPV)",
                "Direct Negative Predictive Value (NPV)",
                "False Omission Rate (FOR = 1 - NPV)",
                "False Positives per True Positive",
                "Target Setting Prevalence (Override)",
                "Bayes' Theorem Model-Based PPV",
                "Bayes' Theorem Model-Based NPV"
            ],
            "Excel Formula": [
                "=B4/D4",
                "=B2/B4",
                "=C3/C4",
                "=B2/D2",
                "=1-B9",
                "=C3/D3",
                "=1-B11",
                "=C2/B2",
                "0.05",
                "=(B7*B14)/((B7*B14)+((1-B8)*(1-B14)))",
                "=(B8*(1-B14))/((B8*(1-B14))+((1-B7)*B14))"
            ],
            "Calculated Value": [
                f"{calcs['sample_prevalence']:.4f}",
                f"{calcs['sensitivity']:.4f}",
                f"{calcs['specificity']:.4f}",
                f"{calcs['ppv']:.4f}",
                f"{calcs['fdr']:.4f}",
                f"{calcs['npv']:.4f}",
                f"{calcs['for_rate']:.4f}",
                f"{calcs['false_positives_per_true_positive']:.2f}",
                "0.0500",
                f"{(calcs['sensitivity']*0.05)/((calcs['sensitivity']*0.05)+((1-calcs['specificity'])*0.95)):.4f}",
                f"{(calcs['specificity']*0.95)/((calcs['specificity']*0.95)+((1-calcs['sensitivity'])*0.05)):.4f}"
            ]
        })
        st.dataframe(df_summary_cells, use_container_width=True, hide_index=True)

    with st.expander("📌 Step 2: Calculate Direct PPV & NPV in Excel", expanded=False):
        st.markdown(r"""
        1. **Compute Direct Positive Predictive Value (PPV):** In cell `B9`, enter:
           ```excel
           =B2 / D2
           ```
           *(True Positives divided by Total Test Positives)*. Format as **Percentage with 1 decimal place** (e.g. `8.3%`).
        2. **Compute False Discovery Rate (FDR):** In cell `B10`, enter:
           ```excel
           =1 - B9
           ```
        3. **Compute Direct Negative Predictive Value (NPV):** In cell `B11`, enter:
           ```excel
           =C3 / D3
           ```
           *(True Negatives divided by Total Test Negatives)*. Format as **Percentage** (e.g. `99.9%`).
        4. **Compute False Omission Rate (FOR):** In cell `B12`, enter:
           ```excel
           =1 - B11
           ```
        5. **Compute False Alarms per True Case Ratio:** In cell `B13`, enter:
           ```excel
           =C2 / B2
           ```
        """)

    with st.expander("📌 Step 3: Apply Bayes' Theorem with External Population Prevalence in Excel", expanded=False):
        st.markdown(r"""
        When you apply test sensitivity and specificity to a new population with an external disease prevalence $p$ (entered in cell `B14`):
        
        1. **Enter Target Setting Prevalence:** In cell `B14`, enter your target population prevalence (e.g. `0.005` for 0.5% population screening, or `0.25` for 25% emergency clinic).
        2. **Compute Bayes' Theorem Positive Predictive Value (PPV):** In cell `B15`, enter:
           ```excel
           =(B7 * B14) / ((B7 * B14) + ((1 - B8) * (1 - B14)))
           ```
        3. **Compute Bayes' Theorem Negative Predictive Value (NPV):** In cell `B16`, enter:
           ```excel
           =(B8 * (1 - B14)) / ((B8 * (1 - B14)) + ((1 - B7) * B14))
           ```
        """)

    with st.expander("📌 Step 4: Build the Bayes Prevalence Curve in Excel", expanded=False):
        st.markdown(r"""
        1. In columns `G`, `H`, and `I`, create a prevalence simulation table:
           * Column `G`: Prevalence values from `0.01` to `0.99` in steps of `0.02` (`G2` = 0.01, `G3` = 0.03, etc.).
           * Column `H`: PPV formula: `=($B$7 * G2) / (($B$7 * G2) + ((1 - $B$8) * (1 - G2)))`. Fill down.
           * Column `I`: NPV formula: `=($B$8 * (1 - G2)) / (($B$8 * (1 - G2)) + ((1 - $B$7) * G2))`. Fill down.
        2. Highlight `Cells G1:I51`.
        3. Go to **Insert ➔ Charts ➔ Scatter with Smooth Lines**.
        4. Label X-Axis as **Disease Prevalence (%)** and Y-Axis as **Post-Test Probability (%)**.
        5. Observe how PPV curves upward non-linearly, clearly revealing the False Positive Paradox!
        """)


def render_predictive_values_python_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 5-step Python implementation guide with direct math, Bayes' theorem, and Plotly scripts."""
    st.markdown("### 🐍 Python Implementation Workflow for Predictive Values & Bayes' Theorem")
    st.markdown("Follow the 5-step sequence: **Define 2×2 Counts & Prerequisites** ➔ **Direct PPV & NPV** ➔ **Bayes' Theorem Prevalence Modeling** ➔ **Consolidated Production Script** ➔ **Plotly Bayes Curves**.")

    tp, fp, fn, tn = calcs["tp"], calcs["fp"], calcs["fn"], calcs["tn"]

    with st.expander("📌 Step 1: Define Confusion Matrix & Methodological Prerequisites", expanded=False):
        st.markdown(f"""
        ```python
        import numpy as np
        import pandas as pd
        from scipy import stats

        # 1. 2x2 Matrix Counts
        tp = {tp}   # True Positives
        fp = {fp}   # False Positives (False Alarms)
        fn = {fn}   # False Negatives (Missed Cases)
        tn = {tn}   # True Negatives

        total_test_pos = tp + fp
        total_test_neg = fn + tn
        total_diseased = tp + fn
        total_healthy = tn + fp
        total_sample = tp + fp + fn + tn

        # 2. Intrinsic Sensitivity & Specificity
        sensitivity = tp / total_diseased if total_diseased > 0 else 0.0
        specificity = tn / total_healthy if total_healthy > 0 else 0.0
        sample_prevalence = total_diseased / total_sample if total_sample > 0 else 0.0

        print(f"Sample Prevalence:  {{sample_prevalence * 100:.2f}}%")
        print(f"Sensitivity (TPR):  {{sensitivity * 100:.2f}}%")
        print(f"Specificity (TNR):  {{specificity * 100:.2f}}%")
        ```
        """)

    with st.expander("📌 Step 2: Compute Direct Predictive Values (PPV, NPV, FDR, FOR)", expanded=False):
        st.markdown("""
        ```python
        # 1. Positive Predictive Value (PPV) & False Discovery Rate (FDR)
        ppv = tp / total_test_pos if total_test_pos > 0 else 0.0
        fdr = 1.0 - ppv  # False alarms proportion

        # 2. Negative Predictive Value (NPV) & False Omission Rate (FOR)
        npv = tn / total_test_neg if total_test_neg > 0 else 0.0
        for_rate = 1.0 - npv  # Missed cases proportion

        # 3. Ratio of False Positives to True Positives
        fp_to_tp_ratio = fp / tp if tp > 0 else np.nan

        print(f"PPV (Direct Sample): {{ppv * 100:.2f}}% (FDR False Alarms: {{fdr * 100:.2f}}%)")
        print(f"NPV (Direct Sample): {{npv * 100:.2f}}% (FOR Missed: {{for_rate * 100:.2f}}%)")
        print(f"False Alarms per True Case: {{fp_to_tp_ratio:.1f}} to 1")
        ```
        """)

    with st.expander("📌 Step 3: Model Predictive Values across Disease Prevalences via Bayes' Theorem", expanded=False):
        st.markdown("""
        ```python
        def bayes_predictive_values(sensitivity: float, specificity: float, prevalence: float):
            \"\"\"Calculates model-based PPV and NPV via Bayes' Theorem.\"\"\"
            p = float(prevalence)
            se = float(sensitivity)
            sp = float(specificity)

            # Bayes PPV = [Se * p] / [Se * p + (1 - Sp) * (1 - p)]
            p_pos = (se * p) + ((1.0 - sp) * (1.0 - p))
            bayes_ppv = (se * p) / p_pos if p_pos > 0 else 0.0

            # Bayes NPV = [Sp * (1 - p)] / [Sp * (1 - p) + (1 - Se) * p]
            p_neg = (sp * (1.0 - p)) + ((1.0 - se) * p)
            bayes_npv = (sp * (1.0 - p)) / p_neg if p_neg > 0 else 0.0

            return bayes_ppv, bayes_npv

        # Demonstration of the False Positive Paradox across settings
        for test_prev in [0.001, 0.005, 0.05, 0.25]:
            p_ppv, p_npv = bayes_predictive_values(sensitivity, specificity, test_prev)
            print(f"Prevalence {{test_prev*100:>5.1f}}% -> PPV: {{p_ppv*100:>5.1f}}%, NPV: {{p_npv*100:>5.1f}}%")
        ```
        """)

    with st.expander("📌 Step 4: Consolidated Production Predictive Values Script", expanded=False):
        st.markdown(f"""
        ```python
        import numpy as np
        import pandas as pd
        from scipy import stats

        def run_predictive_values_pipeline(tp: int, fp: int, fn: int, tn: int, prevalence_override=None, alpha=0.05):
            \"\"\"Runs comprehensive predictive values pipeline with Wilson score CIs.\"\"\"
            total_diseased = tp + fn
            total_healthy = tn + fp
            total_test_pos = tp + fp
            total_test_neg = fn + tn
            total_sample = tp + fp + fn + tn

            sens = tp / total_diseased if total_diseased > 0 else 0.0
            spec = tn / total_healthy if total_healthy > 0 else 0.0
            sample_prev = total_diseased / total_sample if total_sample > 0 else 0.0

            def calc_wilson(k, n):
                if n <= 0: return (0.0, 0.0)
                p = k / n
                z = stats.norm.ppf(1.0 - alpha / 2.0)
                denom = 1.0 + (z**2 / n)
                center = (p + (z**2 / (2.0 * n))) / denom
                spread = (z * np.sqrt((p * (1.0 - p) / n) + (z**2 / (4.0 * n**2)))) / denom
                return (max(0.0, center - spread), min(1.0, center + spread))

            if prevalence_override is None:
                ppv = tp / total_test_pos if total_test_pos > 0 else 0.0
                ppv_ci = calc_wilson(tp, total_test_pos)
                npv = tn / total_test_neg if total_test_neg > 0 else 0.0
                npv_ci = calc_wilson(tn, total_test_neg)
                eff_prev = sample_prev
            else:
                p = prevalence_override
                eff_prev = p
                p_pos = (sens * p) + ((1.0 - spec) * (1.0 - p))
                ppv = (sens * p) / p_pos if p_pos > 0 else 0.0
                p_neg = (spec * (1.0 - p)) + ((1.0 - sens) * p)
                npv = (spec * (1.0 - p)) / p_neg if p_neg > 0 else 0.0
                ppv_ci = (ppv, ppv)
                npv_ci = (npv, npv)

            return {{
                "Prevalence": eff_prev,
                "Sensitivity": sens,
                "Specificity": spec,
                "PPV": ppv,
                "PPV_95CI": ppv_ci,
                "NPV": npv,
                "NPV_95CI": npv_ci,
                "FDR": 1.0 - ppv,
                "FOR": 1.0 - npv,
                "False_Alarms_Per_Case": fp / tp if tp > 0 else np.nan
            }}

        # Execute
        results = run_predictive_values_pipeline(tp={tp}, fp={fp}, fn={fn}, tn={tn})
        print("=== Diagnostic Predictive Values Summary ===")
        for k, v in results.items():
            if isinstance(v, tuple):
                print(f"{{k:<25}}: [{{v[0]*100:.1f}}%, {{v[1]*100:.1f}}%]")
            elif isinstance(v, float):
                print(f"{{k:<25}}: {{v*100:.2f}}%" if "Case" not in k else f"{{k:<25}}: {{v:.2f}} : 1")
        ```
        """)

    with st.expander("📌 Step 5: Generate Plotly Bayes Prevalence Curve Script", expanded=False):
        st.markdown(f"""
        ```python
        import numpy as np
        import plotly.graph_objects as go

        sens = {calcs['sensitivity']}
        spec = {calcs['specificity']}

        prev_range = np.linspace(0.001, 0.999, 100)
        ppv_curve = (sens * prev_range) / ((sens * prev_range) + ((1.0 - spec) * (1.0 - prev_range))) * 100.0
        npv_curve = (spec * (1.0 - prev_range)) / ((spec * (1.0 - prev_range)) + ((1.0 - sens) * prev_range)) * 100.0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prev_range * 100, y=ppv_curve, mode="lines", name="PPV (%)", line=dict(color="#2563eb", width=2)))
        fig.add_trace(go.Scatter(x=prev_range * 100, y=npv_curve, mode="lines", name="NPV (%)", line=dict(color="#10b981", width=2, dash="dash")))

        fig.update_layout(
            title="Bayes' Theorem Prevalence Curve: PPV & NPV vs. Disease Prevalence",
            xaxis_title="Prevalence (%)",
            yaxis_title="Predictive Value (%)"
        )
        fig.show()
        ```
        """)


def render_predictive_values_download_tab(df_contingency: pd.DataFrame, df_microdata: pd.DataFrame, meta: Dict[str, Any]):
    """Renders data download tab for individual patient microdata and 2x2 predictive matrix."""
    st.markdown("### 💾 Live Predictive Values Data Downloads")
    st.caption("Download individual patient microdata or the aggregated 2×2 diagnostic predictive table.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Aggregated 2×2 Predictive Matrix Table (CSV)**")
        csv_matrix = df_contingency.to_csv(index=False)
        st.download_button(
            label="📥 Download 2×2 Predictive Matrix (CSV)",
            data=csv_matrix,
            file_name=f"predictive_values_2x2_matrix_{meta.get('scenario_key', 'custom')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df_contingency, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**2. Synthetic Patient Microdata (Sample of N Rows)**")
        csv_micro = df_microdata.to_csv(index=False)
        st.download_button(
            label=f"📥 Download Patient Microdata ({len(df_microdata)} Rows CSV)",
            data=csv_micro,
            file_name=f"predictive_values_microdata_{meta.get('scenario_key', 'custom')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df_microdata.head(6), use_container_width=True, hide_index=True)


def render_predictive_values_rationale_tab(method_info: Dict[str, Any], calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the comprehensive Rationale & Epidemiological Framework tab for Predictive Values and Bayes' Theorem."""
    st.subheader(f"Why Predictive Values (PPV & NPV) & Bayes' Theorem? ({method_info['name']})")
    st.markdown("""
    When an asymptomatic person receives an abnormal screening result, or a patient in the emergency department 
    has an elevated cardiac biomarker test, their immediate and urgent question is: 
    **"Doctor, what are the chances that I actually have this disease?"**
    
    Clinicians frequently make a critical cognitive error known as the **Prosecutor's Fallacy** or the **Transposition Fallacy**: 
    they confuse **Sensitivity** (the probability of a positive test *given that the patient already has the disease*, $P(\text{Test}+ \mid \text{Disease}+)$) 
    with **Positive Predictive Value (PPV)** (the probability that the patient *actually has the disease given that they just tested positive*, $P(\text{Disease}+ \mid \text{Test}+)$).
    
    A laboratory test may have an outstanding Sensitivity of $99\%$ and Specificity of $99\%$, yet when applied to an unselected general population 
    where the condition is rare (e.g. $0.1\%$ prevalence), **over $90\%$ of all positive results will be false alarms!** 
    This is the celebrated **False Positive Paradox**. **Predictive Values (PPV and NPV)** answer the patient's real question, 
    and **Bayes' Theorem** provides the mathematical framework that connects laboratory accuracy to real-world population disease prevalence.
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
        st.info(f"**Alternative if assumptions fail:** {method_info.get('alternative_test', 'Diagnostic Likelihood Ratios (LR+ & LR-)')}")

    st.markdown("---")
    st.markdown("### 🔍 The Two Perspectives of Diagnostic Evaluation: Laboratory Accuracy vs. Patient Clinical Reality")
    st.markdown(r"""
    | Diagnostic Metric | Full Name & Abbreviation | Direction of Question | Clinical Perspective | Prevalence Dependent? | Primary Public Health Role |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Sensitivity** | True Positive Rate (TPR) | $P(\text{Test}+ \mid \text{Disease}+)$ | **Laboratory Bench:** Looks backward from confirmed disease cases. | **No.** Inherent property of assay. | Triage & Screening: Catching true cases (**SnNOut**). |
    | **Specificity** | True Negative Rate (TNR) | $P(\text{Test}- \mid \text{Disease}-)$ | **Laboratory Bench:** Looks backward from confirmed healthy cases. | **No.** Inherent property of assay. | Confirmatory: Avoiding false alarms (**SpPIn**). |
    | **Positive Predictive Value** | PPV / Post-Test Probability of Disease | $P(\text{Disease}+ \mid \text{Test}+)$ | **Patient in Clinic:** Looks forward from a positive test report. | **Yes, dramatically.** Drops toward zero in rare conditions. | Patient Counseling: Communicating the true probability of having the disease. |
    | **Negative Predictive Value** | NPV / Post-Test Probability of Health | $P(\text{Disease}- \mid \text{Test}-)$ | **Patient in Clinic:** Looks forward from a negative test report. | **Yes, dramatically.** Rises toward $100\%$ in rare conditions. | Patient Reassurance: Safely ruling out disease and clearing patients. |
    | **False Discovery Rate** | FDR ($1 - \text{PPV}$) | $P(\text{Disease}- \mid \text{Test}+)$ | **Health System / Quality:** Proportion of alarms that are false. | **Yes.** | Resource Planning: Auditing unnecessary recalls and biopsies. |
    | **False Omission Rate** | FOR ($1 - \text{NPV}$) | $P(\text{Disease}+ \mid \text{Test}-)$ | **Safety / Risk Management:** Proportion of cleared tests that missed disease. | **Yes.** | Clinical Governance: Monitoring safety of discharge protocols. |
    """)

    st.markdown("#### 📖 Mathematical Formulae Explained in Plain English (Zero Prior Knowledge Assumed)")
    st.markdown(r"""
    Here is every formula for Predictive Values and Bayes' Theorem broken down term-by-term into plain English:

    1. **Positive Predictive Value (PPV):**
       $$\text{Positive Predictive Value } (PPV) = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} = \frac{a}{a + b}$$
       * **What the Formula Measures:** The numerator ($a$) is the number of sick people who correctly tested positive. The denominator ($a + b$) is the total number of ALL people who received a positive test result (sick people plus healthy false alarms).
       * **In Plain English:** PPV answers: *"Out of 100 people who hold a positive test result in their hands, how many actually have the disease?"* If 100 people test positive and 75 have the disease, the PPV is $75\%$.

    2. **Negative Predictive Value (NPV):**
       $$\text{Negative Predictive Value } (NPV) = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Negatives}} = \frac{d}{c + d}$$
       * **What the Formula Measures:** The numerator ($d$) is the number of healthy people who correctly tested negative. The denominator ($c + d$) is the total number of ALL people who received a negative test result (healthy people plus sick patients whose disease was missed).
       * **In Plain English:** NPV answers: *"Out of 100 people who hold a negative test result in their hands, how many are truly healthy and disease-free?"* If 1,000 people test negative and 990 are healthy, the NPV is $99\%$.

    3. **False Discovery Rate (FDR):**
       $$\text{False Discovery Rate } (FDR) = 1 - PPV = \frac{\text{False Positives}}{\text{True Positives} + \text{False Positives}} = \frac{b}{a + b}$$
       * **In Plain English:** The false alarm rate among positive results. If PPV is $8.3\%$, then FDR is $91.7\%$. That means over $91\%$ of people flagged by the test are completely healthy.

    4. **False Omission Rate (FOR):**
       $$\text{False Omission Rate } (FOR) = 1 - NPV = \frac{\text{False Negatives}}{\text{True Negatives} + \text{False Negatives}} = \frac{c}{c + d}$$
       * **In Plain English:** The missed-diagnosis rate among negative results. If NPV is $99.9\%$, then FOR is $0.1\%$, meaning only 1 in 1,000 cleared patients has an undetected condition.

    5. **Bayes' Theorem for Positive Predictive Value (Decoupling Test Accuracy from Prevalence):**
       $$PPV = \frac{\text{Sensitivity} \times \text{Prevalence}}{\text{Sensitivity} \times \text{Prevalence} + (1 - \text{Specificity}) \times (1 - \text{Prevalence})}$$
       * **Term-by-Term Breakdown in Plain English:**
         * **Numerator ($\text{Sensitivity} \times \text{Prevalence}$):** Represents the proportion of the entire population who are truly sick AND correctly detected by the test (**True Positives**).
         * **Denominator Part 1 ($\text{Sensitivity} \times \text{Prevalence}$):** The same True Positives.
         * **Denominator Part 2 ($(1 - \text{Specificity}) \times (1 - \text{Prevalence})$):** The healthy proportion of the population ($(1 - \text{Prevalence})$) multiplied by the test's false alarm rate ($(1 - \text{Specificity})$), representing **False Positives**.
       * **Why Bayes' Theorem is Essential:** You can measure Sensitivity and Specificity in an academic hospital trial, and then use this single equation to calculate the exact PPV for any new clinic, screening van, or community without ever having to re-run the clinical trial!

    6. **Bayes' Theorem for Negative Predictive Value:**
       $$NPV = \frac{\text{Specificity} \times (1 - \text{Prevalence})}{\text{Specificity} \times (1 - \text{Prevalence}) + (1 - \text{Sensitivity}) \times \text{Prevalence}}$$
       * **Term-by-Term Breakdown in Plain English:** The exact mirror image of the PPV equation, dividing correctly identified healthy people by the sum of healthy people and missed cases.

    7. **The False Positive Paradox Explained Step-by-Step (The 100,000-Person Demonstration):**
       * Imagine screening **100,000 citizens** for a rare disease affecting **1 in 1,000 people (Prevalence = 0.1%)**.
       * Your laboratory test has an outstanding **99% Sensitivity** and **99% Specificity** (only a 1% error rate).
       * **The 100 Truly Sick People:** The test detects 99 of them ($TP = 99$), and misses 1 ($FN = 1$).
       * **The 99,900 Healthy People:** The test correctly clears 98,901 people, but the 1% error rate triggers on healthy people: $0.01 \times 99,900 = \mathbf{999 \text{ False Positives}}$!
       * **The Shocking Calculation:** Total positive results = $99 + 999 = 1,098$.
         $$PPV = \frac{99}{1,098} \approx \mathbf{9.0\%}$$
       * **Conclusion:** Even though the test is "99% accurate", **over 91% of everyone who tested positive is completely healthy!** The massive haystack of healthy people overwhelms the tiny needle of sick people.

    8. **Mercaldo, Lau & Zhou (2007) Logit-Scale Delta Confidence Intervals for Model-Based Bayes Estimates:**
       * When prevalence is specified from external census or registry data rather than an internal cross-sectional sample, standard Wilson intervals cannot be applied.
       * Mercaldo et al. transform PPV and NPV to the logit scale: $g = \ln\left(\frac{PPV}{1 - PPV}\right)$.
       * Using the multivariate Delta Method, the variance is calculated from the variances of Sensitivity and Specificity, bounded, and mapped back using the logistic function $\frac{1}{1 + e^{-g}}$. This guarantees mathematically sound 95% confidence intervals that never exceed 100% or drop below 0%.
    """)


def render_predictive_values_module(active_method_key: str, method_info: Dict[str, Any]):
    """Top-level controller for Diagnostic Predictive Values (PPV & NPV) and Bayes' Theorem."""
    st.markdown("## 🎯 Diagnostic Predictive Values: PPV, NPV & Bayes' Theorem")
    st.caption("Evaluates real-world post-test probabilities of disease and health across changing clinical prevalences using Bayes' Theorem.")

    # 1. Plain English Pedagogical Guide
    render_predictive_values_plain_english_guide()

    # 2. Scenario Selection Sidebar / Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🩺 Diagnostic Setting & Prevalence Scenario")

    scenario_options = {
        "breast_screening_low_prev": "🎗️ County Mobile Breast Screening (Low Prevalence 0.5% - False Positive Paradox)",
        "troponin_chest_pain_high_prev": "🏥 Inner City Hospital A&E Troponin (High Prevalence 25.0% - Cardiac ACS Triage)",
        "lateral_flow_flu_surveillance": "🦠 All Meadows Winter Respiratory Lateral Flow (Moderate Prevalence 5.0% - Surveillance)",
        "custom": "✏️ Custom 2×2 Predictive Matrix & Bayes Input"
    }

    selected_scenario = st.sidebar.selectbox(
        "Choose Clinical Setting:",
        options=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
        index=0
    )

    if selected_scenario != "custom":
        df_contingency, df_microdata, meta = generate_predictive_values_demo_data(selected_scenario)
        tp = meta["tp"]
        fp = meta["fp"]
        fn = meta["fn"]
        tn = meta["tn"]
        prev_override = meta.get("prevalence_override", None)
    else:
        st.sidebar.markdown("**Input Custom 2×2 Diagnostic Counts:**")
        tp = st.sidebar.number_input("True Positives (TP): Diseased with Positive Test", min_value=0, max_value=500000, value=25, step=1)
        fp = st.sidebar.number_input("False Positives (FP): Healthy with Positive Test", min_value=0, max_value=500000, value=75, step=1)
        fn = st.sidebar.number_input("False Negatives (FN): Diseased with Negative Test", min_value=0, max_value=500000, value=3, step=1)
        tn = st.sidebar.number_input("True Negatives (TN): Healthy with Negative Test", min_value=0, max_value=500000, value=900, step=1)

        use_bayes = st.sidebar.checkbox("Apply External Population Prevalence Override (Bayes' Theorem)", value=False)
        if use_bayes:
            prev_pct_in = st.sidebar.slider("External Setting Prevalence (%):", min_value=0.01, max_value=50.0, value=1.0, step=0.01)
            prev_override = prev_pct_in / 100.0
        else:
            prev_override = None

        meta = {
            "title": "Custom Diagnostic Setting & Predictive Evaluation",
            "domain": "User-Defined Clinical Investigation",
            "description": "Custom user 2×2 predictive contingency matrix evaluated directly and via Bayes' Theorem.",
            "test_name": "Index Diagnostic Test",
            "condition_name": "Target Condition",
            "test_levels": ("Test Positive", "Test Negative"),
            "condition_levels": ("Condition Present (Gold +)", "Condition Absent (Gold -)"),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "prevalence_override": prev_override,
            "scenario_key": "custom",
            "clinical_takeaway": "Custom parameters evaluated with Wilson / Mercaldo 95% confidence intervals and Bayes' Theorem prevalence modeling.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
        df_contingency = pd.DataFrame([
            {"Test Result": "Test Positive", "Condition Present": tp, "Condition Absent": fp, "Total": tp + fp},
            {"Test Result": "Test Negative", "Condition Present": fn, "Condition Absent": tn, "Total": fn + tn},
            {"Test Result": "Total", "Condition Present": tp + fn, "Condition Absent": fp + tn, "Total": tp + fp + fn + tn}
        ])
        df_microdata = pd.DataFrame([
            {"Patient_ID": f"PT_{i+1:04d}", "Test_Result": "Positive", "Condition": "Present"} for i in range(min(tp, 20))
        ])

    calcs = calculate_predictive_values(tp, fp, fn, tn, prevalence_override=prev_override)
    meta["calcs"] = calcs

    # 3. Scenario Header & Educational Disclaimer Banner
    st.markdown(f"### 📋 Case Study: {meta['title']}")
    st.markdown(f"**Clinical Domain:** *{meta['domain']}*")
    st.markdown(f"{meta['description']}")

    st.markdown(f"""
    > [!NOTE]
    > **Educational Disclaimer:** {meta['disclaimer']}
    """)

    # 4. Clinical Takeaway Callout Card
    with st.container(border=True):
        st.markdown("#### 💡 Clinical & Epidemiological Takeaway")
        st.markdown(f"{meta['clinical_takeaway']}")

    # 5. Methodological Prerequisite Evaluation (STARD / QUADAS-2)
    evaluations = evaluate_predictive_value_prerequisites(tp, fp, fn, tn)
    render_predictive_values_diagnostics(evaluations)

    st.markdown("---")

    # 6. Primary Metrics Dashboard
    render_predictive_values_dashboard(calcs, meta)

    st.markdown("---")

    # 7. Side-by-Side Visualizations & Balanced Explanatory Containers
    render_predictive_values_visualizations(calcs, meta)

    st.markdown("---")

    # 8. Interactive Prevalence Simulator
    render_predictive_values_prevalence_simulator(calcs)

    st.markdown("---")

    # 9. Methodology, Implementation Guides & Downloads Tabs
    tab_rationale, tab_excel, tab_python, tab_data = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Microsoft Excel Implementation Guide",
        "🐍 Python Production Script Guide",
        "💾 Live Data Downloads"
    ])

    with tab_rationale:
        render_predictive_values_rationale_tab(method_info, calcs, meta)

    with tab_excel:
        render_predictive_values_excel_guide(calcs, meta)

    with tab_python:
        render_predictive_values_python_guide(calcs, meta)

    with tab_data:
        render_predictive_values_download_tab(df_contingency, df_microdata, meta)


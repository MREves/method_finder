"""Streamlit UI Components & Visualizations for Diagnostic Test Accuracy & Screening Evaluation.

Provides:
1. render_diagnostic_accuracy_plain_english_guide: Zero-prior-knowledge pedagogical guide with Smoke Alarm analogy, SnNOut/SpPIn mnemonics, and Youden's Index.
2. render_diagnostic_accuracy_diagnostics: Tri-color evaluation cards (PASS, WARNING, FAIL) for reference standard independence, review blinding, spectrum bias, and sample size.
3. render_diagnostic_accuracy_dashboard: Primary metrics (Sensitivity, Specificity, FPR, FNR, Accuracy, Youden's J, DOR) with Wilson 95% CIs.
4. render_diagnostic_accuracy_visualizations: 2x2 diagnostic confusion heatmap, performance balance bar chart, balanced explanatory containers, and 2x2 matrix display.
5. render_diagnostic_accuracy_excel_guide: 4-step sequence with cell mapping, explicit formulas, Wilson CI formulas, and Excel charting steps.
6. render_diagnostic_accuracy_python_guide: 5-step sequence with prerequisite validation, Wilson score math, consolidated pipeline, and Plotly visualization scripts.
7. render_diagnostic_accuracy_download_tab: Live CSV downloads for patient microdata and 2x2 confusion matrix.
8. render_diagnostic_accuracy_module: Top-level controller for Diagnostic Test Accuracy & Screening Evaluation.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.diagnostic_accuracy import (
    calculate_diagnostic_accuracy,
    evaluate_diagnostic_prerequisites,
    generate_diagnostic_accuracy_demo_data,
    calculate_wilson_score_interval
)


def render_diagnostic_accuracy_plain_english_guide():
    """Renders the zero-prior-knowledge pedagogical guide for Sensitivity and Specificity."""
    with st.expander("💡 Plain English Guide: Sensitivity, Specificity & Screening Accuracy", expanded=False):
        st.markdown("""
        In clinical medicine, public health screening, and epidemiology, no diagnostic test or rapid swab is 100% perfect. 
        Evaluating a diagnostic test requires comparing it against an independent **Gold Standard** (e.g. histopathology biopsy, slit-lamp biomicroscopy, or viral RT-PCR).
        
        The two fundamental pillars of diagnostic accuracy measure how well the test classifies diseased versus healthy individuals:
        """)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. Sensitivity (True Positive Rate, $TPR$, Recall)
            * **Definition:** The probability that a patient who **truly has the disease** will test **positive**.
            * **Formula:**
              $$\text{Sensitivity} = \frac{\text{True Positives}}{\text{Total Diseased}} = \frac{TP}{TP + FN} = \frac{a}{a + c}$$
            * **False Negative Rate ($FNR$, Miss Rate):**
              $$FNR = \frac{FN}{TP + FN} = 1 - \text{Sensitivity}$$
            * **The Mnemonic — SnNOut:**
              * When a test has very **high Sensitivity**, a **Negative** test result reliably rules **Out** the disease!
              * *Why?* Because high sensitivity means false negatives ($FN$) are virtually zero. If such a test comes back negative, you can be confident the patient is disease-free.
            * **Clinical Role:** Ideal for **population screening** and **triage** (e.g. blood bank donation screening, community diabetic retinopathy AI, newborn heel-prick tests) where missing a case ($FN$) leads to preventable death or disability.
            """)

            st.markdown(r"""
            #### 2. Specificity (True Negative Rate, $TNR$)
            * **Definition:** The probability that an individual who is **truly healthy / disease-free** will test **negative**.
            * **Formula:**
              $$\text{Specificity} = \frac{\text{True Negatives}}{\text{Total Healthy}} = \frac{TN}{TN + FP} = \frac{d}{b + d}$$
            * **False Positive Rate ($FPR$, False Alarm / Type I Error):**
              $$FPR = \frac{FP}{TN + FP} = 1 - \text{Specificity}$$
            * **The Mnemonic — SpPIn:**
              * When a test has very **high Specificity**, a **Positive** test result reliably rules **In** the disease!
              * *Why?* Because high specificity means false positives ($FP$) are virtually zero. If such a test comes back positive, the patient almost certainly has the condition.
            * **Clinical Role:** Ideal for **confirmatory testing** before initiating risky, invasive, or toxic interventions (e.g. core biopsy before breast surgery, toxic chemotherapy, or HIV Western blot).
            """)

        with col_g2:
            st.markdown(r"""
            #### 3. The Core Analogy: "The Kitchen Smoke Alarm vs. The Sprinkler"
            * **The Smoke Alarm (Ultra-Sensitive Screening Test):**
              * You want your home smoke alarm to be **ultra-sensitive**.
              * If you burn a slice of toast, the smoke alarm shrieks ($FP$). It is mildly annoying, but you would happily tolerate 10 false alarms as long as the alarm **never sleeps through a real bedroom fire ($FN = 0$)**!
              * That is a screening test: prioritize Sensitivity so nobody is missed.
            * **The Fire Sprinkler System (Ultra-Specific Confirmatory Test):**
              * You do **not** want your ceiling fire sprinklers to spray thousands of gallons of water across your living room and ruin all your furniture just because toast burned!
              * The sprinkler must be **ultra-specific**: it should only trigger when intense heat and open flames are unmistakable ($FP = 0$).
              * That is a confirmatory test: prioritize Specificity so healthy patients are never subjected to devastating treatments.
            """)

            st.markdown(r"""
            #### 4. Beyond Raw Accuracy: Youden's Index ($J$) & Wilson CIs
            * **Why Overall Accuracy is Deceptive in Screening:**
              * $\text{Accuracy} = (TP + TN) / N$.
              * If a rare cancer has $1\%$ prevalence, a useless test that always outputs *"Negative"* for everyone achieves **$99.0\%$ Accuracy**, yet misses **$100\%$** of cancer patients!
            * **Youden's Index ($J$, Informedness):**
              $$J = \text{Sensitivity} + \text{Specificity} - 1 = TPR - FPR$$
              * Ranges from $0.0$ (useless coin toss) to $1.0$ (perfect diagnostic test). It captures the net performance above chance regardless of prevalence!
            * **Why Wilson Score 95% Confidence Intervals are Gold Standard:**
              * Standard Wald normal intervals ($\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$) can overshoot $100\%$ when sensitivity or specificity is high (e.g. $98\%$) and fail when sample sizes are modest.
              * CLSI EP12-A2 and the CDC mandate **Wilson score intervals**, which strictly respect the $[0, 1]$ probability boundary and guarantee robust coverage.
            """)

        st.info(r"""
        📐 **Are Sensitivity and Specificity Affected by Disease Prevalence?**  
        In classical diagnostic epidemiology, **Sensitivity and Specificity are intrinsic mathematical properties of the test technology itself** and do not change with disease prevalence (assuming no spectrum bias). 
        However, the *probability that a positive test means the patient actually has the disease* (**Positive Predictive Value, PPV**) depends heavily on population prevalence! We explore PPV, NPV, and Bayes' Theorem in the next module.
        """)


def render_diagnostic_accuracy_diagnostics(evaluations: List[Dict[str, Any]]):
    """Renders tri-color prerequisite diagnostic cards for Diagnostic Test Accuracy."""
    st.markdown("### 🚦 Tri-Color Bias & Methodological Diagnostics (STARD & QUADAS-2)")
    st.caption("Verifies reference standard independence (incorporation bias), blinded review, spectrum representativeness, and sample size stability.")

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


def render_diagnostic_accuracy_dashboard(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders primary diagnostic accuracy metric cards with Wilson 95% CIs."""
    st.markdown("### 📊 Primary Diagnostic Test Performance Measures")

    sens_pct = calcs["sensitivity"] * 100.0
    sens_ll = calcs["sensitivity_ci_lower"] * 100.0
    sens_ul = calcs["sensitivity_ci_upper"] * 100.0

    spec_pct = calcs["specificity"] * 100.0
    spec_ll = calcs["specificity_ci_lower"] * 100.0
    spec_ul = calcs["specificity_ci_upper"] * 100.0

    acc_pct = calcs["accuracy"] * 100.0
    acc_ll = calcs["accuracy_ci_lower"] * 100.0
    acc_ul = calcs["accuracy_ci_upper"] * 100.0

    fnr_pct = calcs["fnr"] * 100.0
    fpr_pct = calcs["fpr"] * 100.0
    prev_pct = calcs["prevalence"] * 100.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Sensitivity (True Positive Rate, TPR)",
            value=f"{sens_pct:.1f}%",
            delta=f"Miss Rate (False Negative Rate, FNR): {fnr_pct:.1f}% (FN={calcs['fn']})",
            delta_color="inverse",
            help="Sensitivity (True Positive Rate, TPR): TP / (TP + FN). Evaluated strictly among diseased patients. Wilson 95% CI displayed in card."
        )
        st.caption(f"**Wilson 95% CI:** [{sens_ll:.1f}%, {sens_ul:.1f}%]")

    with c2:
        st.metric(
            label="Specificity (True Negative Rate, TNR)",
            value=f"{spec_pct:.1f}%",
            delta=f"False Alarm (False Positive Rate, FPR): {fpr_pct:.1f}% (FP={calcs['fp']})",
            delta_color="inverse",
            help="Specificity (True Negative Rate, TNR): TN / (TN + FP). Evaluated strictly among healthy non-diseased individuals."
        )
        st.caption(f"**Wilson 95% CI:** [{spec_ll:.1f}%, {spec_ul:.1f}%]")

    with c3:
        st.metric(
            label="Overall Diagnostic Accuracy",
            value=f"{acc_pct:.1f}%",
            delta=f"Sample Prevalence: {prev_pct:.1f}%",
            delta_color="off",
            help="Overall Proportion Correct: (TP + TN) / Total N. Wilson 95% CI displayed below."
        )
        st.caption(f"**Wilson 95% CI:** [{acc_ll:.1f}%, {acc_ul:.1f}%]")

    with c4:
        dor_val = calcs["dor"]
        dor_str = f"{dor_val:.1f}" if not np.isnan(dor_val) else "N/A"
        st.metric(
            label="Youden's Index (J)",
            value=f"{calcs['youdens_j']:.3f}",
            delta=f"Diagnostic Odds Ratio (DOR): {dor_str}",
            delta_color="off",
            help="Youden's Index J = Sensitivity + Specificity - 1 (ranges from 0 = useless to 1 = perfect test). Diagnostic Odds Ratio (DOR) = (TP*TN)/(FP*FN)."
        )
        if not np.isnan(calcs["dor_ci_lower"]):
            st.caption(f"**Diagnostic Odds Ratio (DOR) 95% CI:** [{calcs['dor_ci_lower']:.1f}, {calcs['dor_ci_upper']:.1f}]")
        else:
            st.caption("Diagnostic Odds Ratio (DOR): Continuity correction applied")


def render_diagnostic_accuracy_visualizations(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders side-by-side Plotly diagnostic visualizations with balanced explanatory containers."""
    st.markdown("### 📈 Visualizing Diagnostic Accuracy & Screening Trade-offs")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**1. 2×2 Diagnostic Confusion Heatmap Matrix**")
        tp = calcs["tp"]
        fp = calcs["fp"]
        fn = calcs["fn"]
        tn = calcs["tn"]
        total_n = calcs["total_sample"]

        # Confusion Matrix Grid
        # Row 0: Test Positive [TP, FP]
        # Row 1: Test Negative [FN, TN]
        z_matrix = [[tp, fp], [fn, tn]]
        text_matrix = [
            [
                f"<b>True Positives (TP)</b><br>{tp:,} patients<br>({tp/total_n*100.0:.1f}% of sample)",
                f"<b>False Positives (FP)</b><br>{fp:,} patients<br>({fp/total_n*100.0:.1f}% of sample)"
            ],
            [
                f"<b>False Negatives (FN)</b><br>{fn:,} patients<br>({fn/total_n*100.0:.1f}% of sample)",
                f"<b>True Negatives (TN)</b><br>{tn:,} patients<br>({tn/total_n*100.0:.1f}% of sample)"
            ]
        ]

        fig_matrix = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=["Disease Present (D+)", "Disease Absent (D-)"],
            y=["Test Positive (T+)", "Test Negative (T-)"],
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            colorscale=[[0, "#3b82f6"], [0.5, "#1d4ed8"], [1.0, "#0f172a"]],
            showscale=False
        ))

        fig_matrix.update_layout(
            height=340,
            margin=dict(l=30, r=30, t=30, b=30),
            template="plotly_white",
            xaxis=dict(side="bottom", tickfont=dict(size=11, family="sans-serif")),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11, family="sans-serif"))
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### 🔍 Understanding the 2×2 Diagnostic Heatmap")
            st.markdown(
                "The 2×2 confusion matrix is the foundational building block for all diagnostic epidemiology:\n"
                f"* **Top-Left Quadrant (True Positives, $TP = {tp:,}$):** Diseased patients correctly flagged by the test. Represents **{calcs['sensitivity']*100.0:.1f}%** of all diseased individuals.\n"
                f"* **Top-Right Quadrant (False Positives, $FP = {fp:,}$):** Healthy people incorrectly flagged positive (Type I error / false alarm). Represents **{calcs['fpr']*100.0:.1f}%** of healthy individuals.\n"
                f"* **Bottom-Left Quadrant (False Negatives, $FN = {fn:,}$):** Diseased patients missed by the test (Type II error / miss rate). Represents **{calcs['fnr']*100.0:.1f}%** of diseased individuals.\n"
                f"* **Bottom-Right Quadrant (True Negatives, $TN = {tn:,}$):** Healthy individuals correctly cleared. Represents **{calcs['specificity']*100.0:.1f}%** of all healthy individuals.\n"
                f"* **Overall Correct Classifications:** $(TP + TN) = {tp + tn:,}$ out of {total_n:,} total patients (**{calcs['accuracy']*100.0:.1f}%** Accuracy)."
            )

    with col_chart2:
        st.markdown("**2. Diagnostic Performance Balance & Error Rates**")

        # Performance comparison bar chart: Sensitivity vs FNR, and Specificity vs FPR
        categories = ["Diseased Cohort (D+)", "Healthy Cohort (D-)", "Composite Informedness"]
        
        correct_rates = [calcs["sensitivity"] * 100.0, calcs["specificity"] * 100.0, calcs["youdens_j"] * 100.0]
        error_rates = [calcs["fnr"] * 100.0, calcs["fpr"] * 100.0, 0.0]

        fig_perf = go.Figure()

        # Correct rates (Sensitivity, Specificity, Youden's J)
        fig_perf.add_trace(go.Bar(
            name="Correct Diagnostic Rate (TPR: Sensitivity / TNR: Specificity / J: Informedness)",
            x=categories,
            y=correct_rates,
            marker_color="#10b981",
            text=[f"{v:.1f}%" for v in correct_rates],
            textposition="auto"
        ))

        # Error rates (FNR / FPR)
        fig_perf.add_trace(go.Bar(
            name="Diagnostic Error Rate (FNR: Miss Rate / FPR: False Alarm Rate)",
            x=categories,
            y=error_rates,
            marker_color="#ef4444",
            text=[f"{v:.1f}%" if v > 0 else "" for v in error_rates],
            textposition="auto"
        ))

        fig_perf.update_layout(
            barmode="stack",
            yaxis=dict(title="Percentage of Cohort (%)", range=[0, 105]),
            height=340,
            margin=dict(l=30, r=30, t=30, b=30),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### ⚖️ The Screening Trade-Off: SnNOut vs. SpPIn")
            st.markdown(
                "Sensitivity and Specificity operate in constant clinical tension:\n"
                f"* **Sensitivity (True Positive Rate = {calcs['sensitivity']*100.0:.1f}%):** Evaluates how thoroughly the test catches the disease. High sensitivity provides **SnNOut** power: with a Miss Rate of only **{calcs['fnr']*100.0:.1f}%**, a negative test reliably rules out disease.\n"
                f"* **Specificity (True Negative Rate = {calcs['specificity']*100.0:.1f}%):** Evaluates how cleanly the test clears healthy people. High specificity provides **SpPIn** power: with a False Alarm Rate of **{calcs['fpr']*100.0:.1f}%**, a positive test rules in disease.\n"
                f"* **Youden's J Index ({calcs['youdens_j']:.3f}):** Combines sensitivity and specificity into a single performance index $(TPR - FPR)$. A score of 1.0 represents a perfect test, while 0.0 reflects a coin toss.\n"
                f"* **Diagnostic Odds Ratio (DOR = {calcs['dor']:.1f}):** Shows that the odds of testing positive are {calcs['dor']:.1f} times higher in diseased patients compared to healthy individuals."
            )

    # 3. Interactive Contingency Matrix Display
    st.markdown("**3. Diagnostic 2×2 Contingency Matrix (Gold Standard Verification)**")
    test_pos_label, test_neg_label = meta.get("test_levels", ("Test Positive", "Test Negative"))
    cond_pos_label, cond_neg_label = meta.get("condition_levels", ("Disease Present (Gold +)", "Disease Absent (Gold -)"))

    df_2x2_display = pd.DataFrame([
        {
            "Test Outcome": f"🔴 {test_pos_label}",
            f"{cond_pos_label}": f"{tp:,} (TP)",
            f"{cond_neg_label}": f"{fp:,} (FP)",
            "Total Test Outcome": f"{tp + fp:,}",
            "Classification Meaning": f"True Positives ({calcs['sensitivity']*100.0:.1f}% caught) & False Alarms ({calcs['fpr']*100.0:.1f}% of healthy)"
        },
        {
            "Test Outcome": f"🟢 {test_neg_label}",
            f"{cond_pos_label}": f"{fn:,} (FN)",
            f"{cond_neg_label}": f"{tn:,} (TN)",
            "Total Test Outcome": f"{fn + tn:,}",
            "Classification Meaning": f"Missed Cases ({calcs['fnr']*100.0:.1f}% of sick) & True Cleared ({calcs['specificity']*100.0:.1f}% of healthy)"
        },
        {
            "Test Outcome": "📊 Total Cohort (Gold Standard)",
            f"{cond_pos_label}": f"{calcs['total_diseased']:,} (Diseased)",
            f"{cond_neg_label}": f"{calcs['total_healthy']:,} (Healthy)",
            "Total Test Outcome": f"{calcs['total_sample']:,} (Total N)",
            "Classification Meaning": f"Sample Prevalence = {calcs['prevalence']*100.0:.1f}% | Overall Accuracy = {calcs['accuracy']*100.0:.1f}%"
        }
    ])

    st.dataframe(df_2x2_display, use_container_width=True, hide_index=True)


def render_diagnostic_accuracy_excel_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 4-step Microsoft Excel Implementation Guide with cell coordinates and Wilson formulas."""
    st.markdown("### 📗 Microsoft Excel Implementation Workflow for Diagnostic Accuracy")
    st.markdown("Follow the 4-step sequence: **Set up 2×2 Matrix Layout** ➔ **Calculate Sensitivity, Specificity & Accuracy** ➔ **Calculate Wilson Score 95% CIs & DOR** ➔ **Build Excel Confusion Heatmap & Performance Charts**.")

    tp, fp, fn, tn = calcs["tp"], calcs["fp"], calcs["fn"], calcs["tn"]
    total_d = calcs["total_diseased"]
    total_h = calcs["total_healthy"]
    total_pos = calcs["total_test_positive"]
    total_neg = calcs["total_test_negative"]
    total_n = calcs["total_sample"]

    test_pos_label, test_neg_label = meta.get("test_levels", ("Test Positive", "Test Negative"))
    cond_pos_label, cond_neg_label = meta.get("condition_levels", ("Condition Present", "Condition Absent"))

    with st.expander("📌 Step 1: Excel 2×2 Data Layout & Cell References", expanded=False):
        st.markdown(f"""
        Structure your Excel worksheet with the 2×2 diagnostic matrix in `Cells A1:E4` and performance formulas in `Cells A6:B16`:
        
        * **Column A (Rows 2:4):** Index Test Result (`Row 2: Positive`, `Row 3: Negative`, `Row 4: Total`)
        * **Column B (Rows 2:4):** Gold Standard Disease Present (`B2`: $TP = {tp}$, `B3`: $FN = {fn}$, `B4`: `=SUM(B2:B3)`)
        * **Column C (Rows 2:4):** Gold Standard Disease Absent (`C2`: $FP = {fp}$, `C3`: $TN = {tn}$, `C4`: `=SUM(C2:C3)`)
        * **Column D (Rows 2:4):** Test Marginals (`D2`: `=SUM(B2:C2)`, `D3`: `=SUM(B3:C3)`, `D4`: `=SUM(B4:C4)`)
        * **Column E (Rows 2:4):** Cell Quadrant Labels (`E2`: `TP / FP`, `E3`: `FN / TN`, `E4`: `Total Sample N`)
        """)

        preview_grid = pd.DataFrame([
            {"Excel_Row": "Row 1 (Headers)", "Col_A": "Index_Test_Result", "Col_B": "Condition_Present_Gold_Pos", "Col_C": "Condition_Absent_Gold_Neg", "Col_D": "Test_Marginal_Total", "Col_E": "Quadrant_Label"},
            {"Excel_Row": "Row 2 (Test Positive)", "Col_A": test_pos_label, "Col_B": str(tp), "Col_C": str(fp), "Col_D": f"{total_pos} (=SUM(B2:C2))", "Col_E": "TP (a) | FP (b)"},
            {"Excel_Row": "Row 3 (Test Negative)", "Col_A": test_neg_label, "Col_B": str(fn), "Col_C": str(tn), "Col_D": f"{total_neg} (=SUM(B3:C3))", "Col_E": "FN (c) | TN (d)"},
            {"Excel_Row": "Row 4 (Condition Totals)", "Col_A": "Total Cohort", "Col_B": f"{total_d} (=SUM(B2:B3))", "Col_C": f"{total_h} (=SUM(C2:C3))", "Col_D": f"{total_n} (=SUM(D2:D3))", "Col_E": "Grand Total N"}
        ])
        st.dataframe(preview_grid, use_container_width=True, hide_index=True)

        st.markdown("**Summary Output Cells Mapping (`Cells A6:B16`):**")
        df_summary_cells = pd.DataFrame({
            "Cell": ["B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15", "B16"],
            "Diagnostic Metric Name": [
                "Sample Disease Prevalence",
                "Sensitivity (TPR = TP / Diseased)",
                "Specificity (TNR = TN / Healthy)",
                "False Positive Rate (FPR = 1 - Spec)",
                "False Negative Rate (FNR = 1 - Sens)",
                "Overall Diagnostic Accuracy",
                "Youden's Index J (Sens + Spec - 1)",
                "Diagnostic Odds Ratio (DOR)",
                "Wilson 95% CI Lower - Sensitivity",
                "Wilson 95% CI Upper - Sensitivity",
                "Wilson 95% CI Lower - Specificity"
            ],
            "Excel Formula": [
                "=B4/D4",
                "=B2/B4",
                "=C3/C4",
                "=1-B8",
                "=1-B7",
                "=(B2+C3)/D4",
                "=B7+B8-1",
                "=(B2*C3)/(C2*B3)",
                "=((B7+(1.96^2)/(2*B4)) - 1.96*SQRT((B7*(1-B7)/B4)+(1.96^2)/(4*B4^2))) / (1+(1.96^2)/B4)",
                "=((B7+(1.96^2)/(2*B4)) + 1.96*SQRT((B7*(1-B7)/B4)+(1.96^2)/(4*B4^2))) / (1+(1.96^2)/B4)",
                "=((B8+(1.96^2)/(2*C4)) - 1.96*SQRT((B8*(1-B8)/C4)+(1.96^2)/(4*C4^2))) / (1+(1.96^2)/C4)"
            ],
            "Calculated Value": [
                f"{calcs['prevalence']:.4f}",
                f"{calcs['sensitivity']:.4f}",
                f"{calcs['specificity']:.4f}",
                f"{calcs['fpr']:.4f}",
                f"{calcs['fnr']:.4f}",
                f"{calcs['accuracy']:.4f}",
                f"{calcs['youdens_j']:.4f}",
                f"{calcs['dor']:.4f}" if not np.isnan(calcs["dor"]) else "N/A",
                f"{calcs['sensitivity_ci_lower']:.4f}",
                f"{calcs['sensitivity_ci_upper']:.4f}",
                f"{calcs['specificity_ci_lower']:.4f}"
            ]
        })
        st.dataframe(df_summary_cells, use_container_width=True, hide_index=True)

    with st.expander("📌 Step 2: Calculate Sensitivity, Specificity & Accuracy in Excel", expanded=False):
        st.markdown(r"""
        1. **Compute Sensitivity ($TPR$):** In cell `B7`, enter:
           ```excel
           =B2 / B4
           ```
           *(True Positives divided by Total Diseased)*. Format cell as **Percentage with 1 decimal place** (e.g. `94.0%`).
        2. **Compute Specificity ($TNR$):** In cell `B8`, enter:
           ```excel
           =C3 / C4
           ```
           *(True Negatives divided by Total Healthy)*. Format as **Percentage** (e.g. `94.4%`).
        3. **Compute False Positive Rate ($FPR$):** In cell `B9`, enter:
           ```excel
           =1 - B8
           ```
        4. **Compute False Negative Rate ($FNR$):** In cell `B10`, enter:
           ```excel
           =1 - B7
           ```
        5. **Compute Overall Diagnostic Accuracy:** In cell `B11`, enter:
           ```excel
           =(B2 + C3) / D4
           ```
        6. **Compute Youden's Index ($J$):** In cell `B12`, enter:
           ```excel
           =B7 + B8 - 1
           ```
        7. **Compute Diagnostic Odds Ratio (DOR):** In cell `B13`, enter:
           ```excel
           =(B2 * C3) / (C2 * B3)
           ```
        """)

    with st.expander("📌 Step 3: Calculate Wilson Score 95% Confidence Intervals in Excel", expanded=False):
        st.markdown(r"""
        The standard Wald normal interval ($p \pm 1.96 \sqrt{p(1-p)/n}$) produces invalid probabilities ($> 100\%$) when sensitivity or specificity is high. 
        In clinical guidelines (CLSI EP12-A2), you must use the **Wilson Score Interval**.
        
        For sample proportion $p$ (e.g. Sensitivity in `B7`), sample size $n$ (Total Diseased in `B4`), and $z = 1.96$:
        
        1. **Sensitivity 95% CI Lower Bound:** In cell `B14`, enter:
           ```excel
           =((B7 + (1.96^2)/(2*B4)) - 1.96 * SQRT((B7*(1-B7)/B4) + (1.96^2)/(4*B4^2))) / (1 + (1.96^2)/B4)
           ```
        2. **Sensitivity 95% CI Upper Bound:** In cell `B15`, enter:
           ```excel
           =((B7 + (1.96^2)/(2*B4)) + 1.96 * SQRT((B7*(1-B7)/B4) + (1.96^2)/(4*B4^2))) / (1 + (1.96^2)/B4)
           ```
        3. **Specificity 95% CI Lower Bound:** In cell `B16`, enter:
           ```excel
           =((B8 + (1.96^2)/(2*C4)) - 1.96 * SQRT((B8*(1-B8)/C4) + (1.96^2)/(4*C4^2))) / (1 + (1.96^2)/C4)
           ```
        4. **Specificity 95% CI Upper Bound:** In cell `B17`, enter:
           ```excel
           =((B8 + (1.96^2)/(2*C4)) + 1.96 * SQRT((B8*(1-B8)/C4) + (1.96^2)/(4*C4^2))) / (1 + (1.96^2)/C4)
           ```
        """)

    with st.expander("📌 Step 4: Create Diagnostic Confusion Matrix & Trade-off Charts in Excel", expanded=False):
        st.markdown(r"""
        #### 1. Building the 2×2 Heatmap Matrix in Excel:
        1. Highlight `Cells B2:C3` containing the 4 confusion counts ($TP, FP, FN, TN$).
        2. Go to **Home ➔ Conditional Formatting ➔ Color Scales**.
        3. Select the **3-Color Blue Scale** (or Blue-White gradient).
        4. In the cell formatting options, center the text and display bold numbers. Add borders around `A1:D4` to create a publication-ready confusion table.

        #### 2. Building the Diagnostic Performance Bar Chart in Excel:
        1. In an open area of the worksheet (e.g. `Cells G1:I3`), create a mini-table:
           * `Cell G1`: `Group`, `Cell H1`: `Correct_Rate`, `Cell I1`: `Error_Rate`
           * `Cell G2`: `Diseased (D+)`, `Cell H2`: `=B7` *(Sensitivity)*, `Cell I2`: `=B10` *(FNR)*
           * `Cell G3`: `Healthy (D-)`, `Cell H3`: `=B8` *(Specificity)*, `Cell I3`: `=B9` *(FPR)*
        2. Select `Cells G1:I3`.
        3. Go to **Insert ➔ Charts ➔ 100% Stacked Column Chart**.
        4. Format the `Correct_Rate` series in emerald green (`#10b981`) and the `Error_Rate` series in red (`#ef4444`).
        5. Add Data Labels to clearly show Sensitivity (TPR) vs Miss Rate (FNR) on diseased individuals, and Specificity (TNR) vs False Alarm Rate (FPR) on healthy individuals!
        """)


def render_diagnostic_accuracy_python_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 5-step Python implementation guide with Wilson interval math, consolidated pipeline, and Plotly scripts."""
    st.markdown("### 🐍 Python Implementation Workflow for Diagnostic Accuracy")
    st.markdown("Follow the 5-step sequence: **Define Confusion Matrix & Validate Biases** ➔ **Calculate Accuracy Metrics** ➔ **Wilson Score Confidence Intervals** ➔ **Consolidated Production Script** ➔ **Interactive Plotly Charts**.")

    tp, fp, fn, tn = calcs["tp"], calcs["fp"], calcs["fn"], calcs["tn"]

    with st.expander("📌 Step 1: Define Confusion Matrix & Methodological Prerequisites", expanded=False):
        st.markdown(f"""
        ```python
        import numpy as np
        import pandas as pd
        from scipy import stats

        # 1. Input 2x2 Confusion Matrix Counts
        # tp: True Positives, fp: False Positives, fn: False Negatives, tn: True Negatives
        tp = {tp}
        fp = {fp}
        fn = {fn}
        tn = {tn}

        total_diseased = tp + fn
        total_healthy = tn + fp
        total_sample = tp + fp + fn + tn

        # 2. Methodological Validity Checks (STARD / QUADAS-2)
        # Verify independence of reference standard (prevent incorporation bias)
        gold_standard_independent = True
        # Verify readers were blinded to disease status during test scoring (prevent review bias)
        review_blinded = True

        print(f"Total Diseased (D+): {{total_diseased}}")
        print(f"Total Healthy (D-):  {{total_healthy}}")
        print(f"Total Sample Size:  {{total_sample}}")
        ```
        """)

    with st.expander("📌 Step 2: Compute Sensitivity, Specificity, Accuracy & Youden's J", expanded=False):
        st.markdown("""
        ```python
        # 1. Sensitivity (True Positive Rate, TPR, Recall)
        sensitivity = tp / total_diseased if total_diseased > 0 else 0.0

        # 2. Specificity (True Negative Rate, TNR)
        specificity = tn / total_healthy if total_healthy > 0 else 0.0

        # 3. Error Rates
        false_positive_rate = 1.0 - specificity  # FPR = FP / (TN + FP)
        false_negative_rate = 1.0 - sensitivity  # FNR = FN / (TP + FN)

        # 4. Overall Diagnostic Accuracy
        accuracy = (tp + tn) / total_sample if total_sample > 0 else 0.0

        # 5. Youden's J Statistic (Informedness)
        youdens_j = sensitivity + specificity - 1.0

        # 6. Diagnostic Odds Ratio (DOR)
        dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else np.nan

        print(f"Sensitivity (TPR):    {sensitivity * 100:.2f}% (FNR: {false_negative_rate * 100:.2f}%)")
        print(f"Specificity (TNR):    {specificity * 100:.2f}% (FPR: {false_positive_rate * 100:.2f}%)")
        print(f"Overall Accuracy:     {accuracy * 100:.2f}%")
        print(f"Youden's Index J:     {youdens_j:.4f}")
        print(f"Diagnostic Odds Ratio:{dor:.2f}")
        ```
        """)

    with st.expander("📌 Step 3: Compute Wilson Score 95% Confidence Intervals", expanded=False):
        st.markdown("""
        ```python
        def wilson_score_interval(k: int, n: int, confidence: float = 0.95):
            \"\"\"Calculates the Wilson score confidence interval for a binomial proportion.
            Recommended by CLSI EP12-A2 and CDC over Wald normal approximation.
            \"\"\"
            if n <= 0:
                return (0.0, 0.0)
            p = k / n
            alpha = 1.0 - confidence
            z = stats.norm.ppf(1.0 - alpha / 2.0)
            z_sq = z * z
            
            denominator = 1.0 + (z_sq / n)
            center = (p + (z_sq / (2.0 * n))) / denominator
            spread = (z * np.sqrt((p * (1.0 - p) / n) + (z_sq / (4.0 * n * n)))) / denominator
            
            lower = max(0.0, center - spread)
            upper = min(1.0, center + spread)
            return (lower, upper)

        sens_ci = wilson_score_interval(tp, total_diseased)
        spec_ci = wilson_score_interval(tn, total_healthy)
        acc_ci = wilson_score_interval(tp + tn, total_sample)

        print(f"Sensitivity 95% CI: [{sens_ci[0]*100:.1f}%, {sens_ci[1]*100:.1f}%]")
        print(f"Specificity 95% CI: [{spec_ci[0]*100:.1f}%, {spec_ci[1]*100:.1f}%]")
        print(f"Accuracy 95% CI:    [{acc_ci[0]*100:.1f}%, {acc_ci[1]*100:.1f}%]")
        ```
        """)

    with st.expander("📌 Step 4: Consolidated Production Diagnostic Accuracy Script", expanded=False):
        st.markdown(f"""
        ```python
        import numpy as np
        import pandas as pd
        from scipy import stats

        def run_diagnostic_accuracy_pipeline(tp: int, fp: int, fn: int, tn: int, alpha: float = 0.05):
            \"\"\"Runs complete diagnostic test accuracy analysis with Wilson score CIs.\"\"\"
            total_diseased = tp + fn
            total_healthy = tn + fp
            total_sample = tp + fp + fn + tn

            # Wilson Score function
            def calc_wilson(k, n):
                if n <= 0: return (0.0, 0.0)
                p = k / n
                z = stats.norm.ppf(1.0 - alpha / 2.0)
                denom = 1.0 + (z**2 / n)
                center = (p + (z**2 / (2.0 * n))) / denom
                spread = (z * np.sqrt((p * (1.0 - p) / n) + (z**2 / (4.0 * n**2)))) / denom
                return (max(0.0, center - spread), min(1.0, center + spread))

            sens = tp / total_diseased if total_diseased > 0 else 0.0
            sens_ci = calc_wilson(tp, total_diseased)

            spec = tn / total_healthy if total_healthy > 0 else 0.0
            spec_ci = calc_wilson(tn, total_healthy)

            acc = (tp + tn) / total_sample if total_sample > 0 else 0.0
            acc_ci = calc_wilson(tp + tn, total_sample)

            youdens_j = sens + spec - 1.0
            dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else np.nan

            return {{
                "Sensitivity": sens,
                "Sensitivity_95CI": sens_ci,
                "Specificity": spec,
                "Specificity_95CI": spec_ci,
                "Accuracy": acc,
                "Accuracy_95CI": acc_ci,
                "FPR": 1.0 - spec,
                "FNR": 1.0 - sens,
                "Youdens_J": youdens_j,
                "DOR": dor
            }}

        # Execute on current clinical scenario
        results = run_diagnostic_accuracy_pipeline(tp={tp}, fp={fp}, fn={fn}, tn={tn})
        print("=== Diagnostic Test Accuracy Summary ===")
        for metric, val in results.items():
            if isinstance(val, tuple):
                print(f"{{metric:<20}}: [{{val[0]*100:.1f}}%, {{val[1]*100:.1f}}%]")
            elif isinstance(val, float):
                print(f"{{metric:<20}}: {{val*100:.2f}}%" if "J" not in metric and "DOR" not in metric else f"{{metric:<20}}: {{val:.3f}}")
        ```
        """)

    with st.expander("📌 Step 5: Generate Plotly Confusion Matrix & Performance Bar Charts", expanded=False):
        st.markdown(f"""
        ```python
        import plotly.graph_objects as go

        # 1. 2x2 Diagnostic Confusion Matrix Heatmap
        tp, fp, fn, tn = {tp}, {fp}, {fn}, {tn}
        total_n = tp + fp + fn + tn

        z_matrix = [[tp, fp], [fn, tn]]
        text_matrix = [
            [f"TP: {{tp:,}} ({{tp/total_n*100:.1f}}%)", f"FP: {{fp:,}} ({{fp/total_n*100:.1f}}%)"],
            [f"FN: {{fn:,}} ({{fn/total_n*100:.1f}}%)", f"TN: {{tn:,}} ({{tn/total_n*100:.1f}}%)"]
        ]

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=["Disease Present (D+)", "Disease Absent (D-)"],
            y=["Test Positive (T+)", "Test Negative (T-)"],
            text=text_matrix,
            texttemplate="%{{text}}",
            colorscale="Blues",
            showscale=False
        ))
        fig_heat.update_layout(title="Diagnostic Confusion Matrix", yaxis=dict(autorange="reversed"))
        fig_heat.show()

        # 2. Performance & Error Rates Stacked Bar Chart
        sens_pct = (tp / (tp + fn)) * 100.0
        spec_pct = (tn / (tn + fp)) * 100.0
        fnr_pct = 100.0 - sens_pct
        fpr_pct = 100.0 - spec_pct

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Correct (TPR / TNR)", x=["Diseased", "Healthy"], y=[sens_pct, spec_pct], marker_color="#10b981"))
        fig_bar.add_trace(go.Bar(name="Error (FNR / FPR)", x=["Diseased", "Healthy"], y=[fnr_pct, fpr_pct], marker_color="#ef4444"))
        fig_bar.update_layout(barmode="stack", title="Diagnostic Accuracy vs. Error Rates")
        fig_bar.show()
        ```
        """)


def render_diagnostic_accuracy_download_tab(df_contingency: pd.DataFrame, df_microdata: pd.DataFrame, meta: Dict[str, Any]):
    """Renders data download tab for individual patient microdata and 2x2 confusion matrix."""
    st.markdown("### 💾 Live Diagnostic Data Downloads")
    st.caption("Download individual patient microdata or the aggregated 2×2 diagnostic confusion table.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Aggregated 2×2 Diagnostic Confusion Matrix (CSV)**")
        csv_matrix = df_contingency.to_csv(index=False)
        st.download_button(
            label="📥 Download 2×2 Diagnostic Matrix (CSV)",
            data=csv_matrix,
            file_name=f"diagnostic_2x2_matrix_{meta.get('scenario_key', 'custom')}.csv",
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
            file_name=f"diagnostic_patient_microdata_{meta.get('scenario_key', 'custom')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df_microdata.head(6), use_container_width=True, hide_index=True)


def render_diagnostic_accuracy_rationale_tab(method_info: Dict[str, Any], calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the comprehensive Rationale & Epidemiological Framework tab for Diagnostic Test Accuracy."""
    st.subheader(f"Why Diagnostic Accuracy & 2×2 Contingency Evaluation? ({method_info['name']})")
    st.markdown("""
    Before any diagnostic assay, clinical screening test, rapid antigen lateral flow strip, or artificial intelligence 
    imaging algorithm can be deployed in the NHS or community healthcare, its analytical and clinical accuracy 
    must be validated against an independent **Gold Standard** (Reference Standard).
    
    The fundamental reality of medical diagnostics is that no test is perfect. Every diagnostic tool produces a trade-off 
    between catching true disease cases (**Sensitivity**) and avoiding false alarms on healthy individuals (**Specificity**). 
    Understanding this 2×2 diagnostic matrix is essential for designing effective screening programs, triage protocols, 
    and confirmatory diagnostic pathways.
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
        st.info(f"**Alternative if assumptions fail:** {method_info.get('alternative_test', 'Predictive Values (PPV & NPV)')}")

    st.markdown("---")
    st.markdown("### 🔍 The Complete Diagnostic Accuracy Matrix: The Six Fundamental Measures")
    st.markdown(r"""
    | Diagnostic Metric | Full Name & Abbreviation | Mathematical Formula | In Plain English | Ideal Clinical Purpose |
    | :--- | :--- | :--- | :--- | :--- |
    | **Sensitivity** | True Positive Rate (TPR) | $\frac{TP}{TP + FN} = \frac{a}{a + c}$ | The percentage of genuinely sick patients who correctly test positive. | **Screening / Triage:** High Sensitivity rules OUT disease (**SnNOut**). |
    | **Specificity** | True Negative Rate (TNR) | $\frac{TN}{TN + FP} = \frac{d}{b + d}$ | The percentage of genuinely healthy patients who correctly test negative. | **Confirmatory Testing:** High Specificity rules IN disease (**SpPIn**). |
    | **False Positive Rate** | FPR (Type I Error / $1 - \text{Spec}$) | $\frac{FP}{TN + FP} = \frac{b}{b + d}$ | The percentage of healthy people mistakenly labeled as diseased (false alarms). | Minimizing patient anxiety, unnecessary biopsies, and overtreatment. |
    | **False Negative Rate** | FNR (Type II Error / $1 - \text{Sens}$) | $\frac{FN}{TP + FN} = \frac{c}{a + c}$ | The percentage of sick patients mistakenly told they are healthy (missed cases). | Preventing delayed care and preventable morbidity in deadly conditions. |
    | **Youden's J Index** | Youden's Informedness ($J$) | $\text{Sens} + \text{Spec} - 1 = \text{TPR} - \text{FPR}$ | A single summary score from $0$ (useless) to $1$ (perfection) measuring net accuracy. | Comparing competing tests or selecting optimal cut-off points on ROC curves. |
    | **Diagnostic Odds Ratio** | DOR | $\frac{TP \times TN}{FP \times FN} = \frac{a \times d}{b \times c}$ | The odds of a positive test in sick patients divided by the odds in healthy patients. | Global measure of discriminating power; portable across meta-analyses. |
    """)

    st.markdown("#### 📖 Mathematical Formulae Explained in Plain English (Zero Prior Knowledge Assumed)")
    st.markdown(r"""
    Here is every foundational diagnostic accuracy formula explained term-by-term in plain English:

    1. **Sensitivity / True Positive Rate (TPR):**
       $$\text{Sensitivity} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} = \frac{a}{a + c}$$
       * **What the Formula Measures:** The numerator ($a$) is the number of diseased patients correctly identified. The denominator ($a + c$) is the total number of genuinely diseased patients in the study.
       * **In Plain English:** Sensitivity answers: *"Out of 100 people who genuinely have the disease, how many does the test catch?"*
       * **The SnNOut Rule:** A test with high **S**e**n**sitivity, when **N**egative, rules **Out** disease. Why? Because a 95% sensitive test has only a 5% false negative rate; if you test negative, it is virtually impossible that you have the disease.

    2. **Specificity / True Negative Rate (TNR):**
       $$\text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}} = \frac{d}{b + d}$$
       * **What the Formula Measures:** The numerator ($d$) is the number of healthy people correctly cleared. The denominator ($b + d$) is the total number of genuinely healthy people in the study.
       * **In Plain English:** Specificity answers: *"Out of 100 healthy people, how many does the test correctly clear as normal?"*
       * **The SpPIn Rule:** A test with high **S**peci**p**icity, when **P**ositive, rules **In** disease. Why? Because a 99% specific test has only a 1% false positive rate; if you test positive, it is virtually certain you have the disease.

    3. **False Positive Rate (FPR / 1 - Specificity / Type I Error):**
       $$\text{False Positive Rate } (FPR) = 1 - \text{Specificity} = \frac{\text{False Positives}}{\text{True Negatives} + \text{False Positives}} = \frac{b}{b + d}$$
       * **In Plain English:** The proportion of healthy people who receive a terrifying false alarm. In breast screening, this represents women recalled for stressful follow-up biopsies for benign cysts.

    4. **False Negative Rate (FNR / 1 - Sensitivity / Type II Error):**
       $$\text{False Negative Rate } (FNR) = 1 - \text{Sensitivity} = \frac{\text{False Negatives}}{\text{True Positives} + \text{False Negatives}} = \frac{c}{a + c}$$
       * **In Plain English:** The proportion of genuinely sick patients who are falsely reassured and sent home without treatment.

    5. **Youden's J Index (Diagnostic Informedness):**
       $$J = \text{Sensitivity} + \text{Specificity} - 1 = \text{True Positive Rate} - \text{False Positive Rate}$$
       * **In Plain English:** A single score between $0.0$ and $1.0$ that captures the net informedness of the test above pure chance:
         * $J = 1.0$: 100% perfect test (100% Sensitivity and 100% Specificity).
         * $J = 0.0$: Completely useless test operating at the level of a coin toss (e.g. 50% Sensitivity and 50% Specificity).
       * In clinical laboratory science, Youden's $J$ is the standard mathematical criterion used to select the optimal diagnostic threshold on a continuous biomarker scale.

    6. **Diagnostic Odds Ratio (DOR):**
       $$\text{Diagnostic Odds Ratio } (DOR) = \frac{\text{Sensitivity} / (1 - \text{Sensitivity})}{(1 - \text{Specificity}) / \text{Specificity}} = \frac{TP \times TN}{FP \times FN} = \frac{a \times d}{b \times c}$$
       * **In Plain English:** Measures how many times higher the odds of testing positive are among sick patients compared to healthy patients. 
       * Unlike overall accuracy, DOR does not change when disease prevalence changes, making it the preferred metric for systematic reviews and meta-analyses.

    7. **Wilson Score 95% Confidence Intervals for Proportions:**
       $$\tilde{p} = \frac{x + \frac{z^2}{2}}{n + z^2}, \quad 95\% \text{ CI} = \tilde{p} \pm \frac{z}{n + z^2} \sqrt{x\left(1 - \frac{x}{n}\right) + \frac{z^2}{4}}$$
       * **Why Standard Wald Intervals Fail:** The standard textbook Wald formula ($\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$) assumes a symmetrical normal distribution. When a test has 98% or 100% Specificity, Wald intervals frequently compute upper bounds greater than $100\%$ (e.g. $[97\%, 102\%]$), which is a mathematical impossibility.
       * **The Wilson Score Solution:** Recommended as the gold standard by CLSI (Clinical and Laboratory Standards Institute) and CDC, the Wilson score interval incorporates a quadratic score adjustment, guaranteeing mathematically valid confidence limits strictly within $[0, 1]$ even with small sample sizes or proportions near $0\%$ or $100\%$.

    8. **Key Methodological Biases in Diagnostic Studies (STARD & QUADAS-2 Guidelines):**
       * **Work-up / Verification Bias:** Occurs when patients with positive test results are preferentially referred for the gold standard procedure, while test-negative patients are sent home without confirmation. This artificially inflates Sensitivity and deflates Specificity.
       * **Incorporation Bias:** Occurs when the index test under evaluation is actually included as part of the criteria that define the gold standard diagnosis. This artificially creates near-perfect accuracy that evaporates in real clinical practice.
       * **Review / Blinding Bias:** Occurs when the doctor or lab technician interpreting the index test already knows whether the patient is sick or healthy. All interpretations must be strictly double-blinded.
       * **Spectrum Bias:** Occurs when a test is validated on an artificial cohort of severely ill hospitalized patients versus young healthy blood donors, rather than a realistic spectrum of mild, moderate, and ambiguous clinical presentations seen in everyday practice.
    """)


def render_diagnostic_accuracy_module(active_method_key: str, method_info: Dict[str, Any]):
    """Top-level controller for Diagnostic Test Accuracy & Screening Evaluation."""
    st.markdown("## 🎯 Diagnostic Test Accuracy: Sensitivity & Specificity (2×2 Matrix)")
    st.caption("Evaluates screening and diagnostic test accuracy against gold standard clinical verification using the STARD / QUADAS-2 framework.")

    # 1. Plain English Pedagogical Guide
    render_diagnostic_accuracy_plain_english_guide()

    # 2. Scenario Selection Sidebar / Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🩺 Diagnostic Test Investigation Scenario")

    scenario_options = {
        "diabetic_retinopathy": "👁️ All Meadows Diabetic Eye Screening (Autonomous AI Camera - SnNOut Screening)",
        "breast_mammography": "🎗️ County Mobile Breast Screening (Mammography - Low Prevalence Trade-off)",
        "rapid_rsv_antigen": "🧪 Inner City Hospital Pediatric A&E RSV (Rapid Lateral Flow - SpPIn Confirmatory)",
        "custom": "✏️ Custom 2×2 Diagnostic Matrix Input"
    }

    selected_scenario = st.sidebar.selectbox(
        "Choose Clinical Scenario:",
        options=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
        index=0
    )

    if selected_scenario != "custom":
        df_contingency, df_microdata, meta = generate_diagnostic_accuracy_demo_data(selected_scenario)
        tp = meta["tp"]
        fp = meta["fp"]
        fn = meta["fn"]
        tn = meta["tn"]
    else:
        st.sidebar.markdown("**Input Custom 2×2 Diagnostic Counts:**")
        tp = st.sidebar.number_input("True Positives (TP, a): Diseased with Positive Test", min_value=0, max_value=500000, value=150, step=1)
        fp = st.sidebar.number_input("False Positives (FP, b): Healthy with Positive Test", min_value=0, max_value=500000, value=30, step=1)
        fn = st.sidebar.number_input("False Negatives (FN, c): Diseased with Negative Test", min_value=0, max_value=500000, value=10, step=1)
        tn = st.sidebar.number_input("True Negatives (TN, d): Healthy with Negative Test", min_value=0, max_value=500000, value=810, step=1)

        meta = {
            "title": "Custom Diagnostic Test Evaluation",
            "domain": "User-Defined Clinical Screening",
            "description": "User-specified 2×2 diagnostic contingency matrix evaluated against an independent reference standard.",
            "test_name": "Index Diagnostic Test",
            "condition_name": "Target Condition",
            "test_levels": ("Test Positive", "Test Negative"),
            "condition_levels": ("Condition Present (Gold +)", "Condition Absent (Gold -)"),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "scenario_key": "custom",
            "clinical_takeaway": "Custom user parameters evaluated with Wilson score 95% confidence intervals and STARD diagnostic validity checks.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
        calcs = calculate_diagnostic_accuracy(tp, fp, fn, tn)
        meta["calcs"] = calcs
        df_contingency = pd.DataFrame([
            {"Test Outcome": "Test Positive", "Condition Present": tp, "Condition Absent": fp, "Total": tp + fp},
            {"Test Outcome": "Test Negative", "Condition Present": fn, "Condition Absent": tn, "Total": fn + tn},
            {"Test Outcome": "Total", "Condition Present": tp + fn, "Condition Absent": fp + tn, "Total": tp + fp + fn + tn}
        ])
        df_microdata = pd.DataFrame([
            {"Patient_ID": f"PT_{i+1:04d}", "Test_Result": "Positive", "Condition": "Present"} for i in range(min(tp, 20))
        ])

    calcs = calculate_diagnostic_accuracy(tp, fp, fn, tn)

    # 3. Scenario Header & Explicit Educational Disclaimer Banner
    st.markdown(f"### 📋 Case Study: {meta['title']}")
    st.markdown(f"**Clinical Domain:** *{meta['domain']}*")
    st.markdown(f"{meta['description']}")

    st.markdown(f"""
    > [!NOTE]
    > **Educational Disclaimer:** {meta['disclaimer']}
    """)

    # 4. Clinical Takeaway Callout Card
    with st.container(border=True):
        st.markdown(f"#### 💡 Clinical & Epidemiological Takeaway")
        st.markdown(f"{meta['clinical_takeaway']}")

    # 5. Methodological Prerequisite Evaluation (STARD / QUADAS-2)
    evaluations = evaluate_diagnostic_prerequisites(tp, fp, fn, tn)
    render_diagnostic_accuracy_diagnostics(evaluations)

    st.markdown("---")

    # 6. Primary Metrics Dashboard
    render_diagnostic_accuracy_dashboard(calcs, meta)

    st.markdown("---")

    # 7. Side-by-Side Visualizations & Balanced Explanatory Containers
    render_diagnostic_accuracy_visualizations(calcs, meta)

    st.markdown("---")

    # 8. Methodology, Implementation Guides & Downloads Tabs
    tab_rationale, tab_excel, tab_python, tab_data = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Microsoft Excel Implementation Guide",
        "🐍 Python Production Script Guide",
        "💾 Live Data Downloads"
    ])

    with tab_rationale:
        render_diagnostic_accuracy_rationale_tab(method_info, calcs, meta)

    with tab_excel:
        render_diagnostic_accuracy_excel_guide(calcs, meta)

    with tab_python:
        render_diagnostic_accuracy_python_guide(calcs, meta)

    with tab_data:
        render_diagnostic_accuracy_download_tab(df_contingency, df_microdata, meta)


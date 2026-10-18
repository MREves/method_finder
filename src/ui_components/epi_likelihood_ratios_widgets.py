"""Streamlit UI Components & Visualizations for Diagnostic Likelihood Ratios (LR+, LR-) and Fagan's Nomogram.

Provides:
1. render_likelihood_ratios_plain_english_guide: Zero-prior-knowledge pedagogical guide with Magnifying Glass analogy, EBM thresholds, and Bayes odds updating.
2. render_likelihood_ratios_diagnostics: Tri-color evaluation cards (PASS, WARNING, FAIL) for reference standard independence, review blinding, spectrum bias, and cell stability.
3. render_likelihood_ratios_dashboard: Primary metrics (Positive LR, Negative LR, Post-Test Probabilities, Simel 95% CIs).
4. render_likelihood_ratios_visualizations: Interactive Plotly Fagan Nomogram, Probability Shift Waterfall bar chart, balanced explanatory containers, and 2x2 matrix display.
5. render_likelihood_ratios_probability_simulator: Interactive live slider allowing dynamic patient-specific pre-test probability updating.
6. render_likelihood_ratios_excel_guide: 4-step sequence with cell mapping, explicit formulas, Simel CIs, and Excel charting steps.
7. render_likelihood_ratios_python_guide: 5-step sequence with direct math, Bayes odds updating, Simel CIs, consolidated pipeline, and Plotly Fagan Nomogram script.
8. render_likelihood_ratios_download_tab: Live CSV downloads for patient microdata and 2x2 contingency matrix.
9. render_likelihood_ratios_module: Top-level controller for Diagnostic Likelihood Ratios & Fagan's Nomogram.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.likelihood_ratios import (
    calculate_likelihood_ratios,
    evaluate_likelihood_ratio_prerequisites,
    generate_likelihood_ratio_demo_data
)


def render_likelihood_ratios_plain_english_guide():
    """Renders the zero-prior-knowledge pedagogical guide for Diagnostic Likelihood Ratios and Fagan's Nomogram."""
    with st.expander("💡 Plain English Guide: Diagnostic Likelihood Ratios & Fagan's Nomogram", expanded=False):
        st.markdown("""
        In clinical epidemiology, while Sensitivity and Specificity describe the laboratory instrument, and Predictive Values (PPV & NPV) apply only to one fixed disease prevalence, 
        **Diagnostic Likelihood Ratios (LRs)** are the clinician's ultimate decision-making tool.
        
        Likelihood Ratios allow doctors to update disease probability for **any individual patient**, starting from their unique clinical suspicion (**Pre-Test Probability**) 
        and arriving at their definitive post-test diagnosis (**Post-Test Probability**) via the **Odds Form of Bayes' Theorem**.
        """)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. Positive Likelihood Ratio (Positive LR, LR+)
            * **Definition:** How many times more likely a **positive test result** is to occur in a patient **with the disease** compared to someone **without the disease**.
            * **Mathematical Formula:**
              $$\text{Positive LR (LR+)} = \frac{\text{Sensitivity}}{1 - \text{Specificity}} = \frac{\text{True Positive Rate (TPR)}}{\text{False Positive Rate (FPR)}} = \frac{a / (a + c)}{b / (b + d)}$$
            * **Evidence-Based Medicine (EBM) Clinical Benchmarks:**
              * $\text{Positive LR (LR+)} \ge \mathbf{10.0}$: **Conclusive Rule-In**. A positive result causes a massive, often definitive increase in disease probability (virtually diagnostic).
              * $\text{Positive LR (LR+)} \in [5.0, 10.0]$: **Moderate Rule-In**. Substantial shift in probability.
              * $\text{Positive LR (LR+)} \in [2.0, 5.0]$: **Small Rule-In**. Helpful but rarely diagnostic alone.
              * $\text{Positive LR (LR+)} = \mathbf{1.0}$: **Useless Test**. Provides zero diagnostic information.
            """)

            st.markdown(r"""
            #### 2. Negative Likelihood Ratio (Negative LR, LR-)
            * **Definition:** How much less likely a **negative test result** is to occur in a patient **with the disease** compared to someone **without the disease**.
            * **Mathematical Formula:**
              $$\text{Negative LR (LR-)} = \frac{1 - \text{Sensitivity}}{\text{Specificity}} = \frac{\text{False Negative Rate (FNR)}}{\text{True Negative Rate (TNR)}} = \frac{c / (a + c)}{d / (b + d)}$$
            * **Evidence-Based Medicine (EBM) Clinical Benchmarks:**
              * $\text{Negative LR (LR-)} \le \mathbf{0.10}$: **Conclusive Rule-Out**. A negative result causes a massive drop in disease probability, safely ruling out the condition.
              * $\text{Negative LR (LR-)} \in [0.10, 0.20]$: **Moderate Rule-Out**. Substantial drop in probability.
              * $\text{Negative LR (LR-)} \in [0.20, 0.50]$: **Small Rule-Out**. Modest drop.
              * $\text{Negative LR (LR-)} = \mathbf{1.0}$: **Useless Test**. No change in probability.
            """)

        with col_g2:
            st.markdown(r"""
            #### 3. The Core Analogy: "The Optical Zoom Lens / Volume Knob"
            * Imagine a camera zoom lens or audio volume knob:
              * Likelihood Ratios do not set the final sound volume on their own; **they multiply whatever initial sound input you feed into them!**
            * **A Faint Whisper (0.5% Pre-Test Risk in Population Screening):**
              * If you amplify a faint whisper with a $10\times$ zoom lens ($\text{Positive LR} = 10$), the sound is still relatively soft ($\approx 5\%$ post-test risk).
            * **A Clear Conversation (30% Pre-Test Risk in Emergency A&E):**
              * If you amplify a clear conversation with that exact same $10\times$ lens ($\text{Positive LR} = 10$), it becomes a deafening roar ($\approx 81\%$ post-test risk)!
            * That is why Likelihood Ratios are universally portable across general practice, emergency triage, and intensive care.
            """)

            st.markdown(r"""
            #### 4. The Odds Form of Bayes' Theorem & Fagan's Nomogram
            * Clinical probability updating follows 3 simple steps:
              1. **Pre-Test Probability ($p$) $\rightarrow$ Pre-Test Odds:**
                 $$\text{Pre-Test Odds} = \frac{p}{1 - p}$$
              2. **Multiply by Likelihood Ratio:**
                 $$\text{Post-Test Odds} = \text{Pre-Test Odds} \times \text{LR}$$
              3. **Post-Test Odds $\rightarrow$ Post-Test Probability:**
                 $$\text{Post-Test Probability} = \frac{\text{Post-Test Odds}}{1 + \text{Post-Test Odds}}$$
            * **Dr. Thomas J. Fagan's Brilliant 1975 Invention:**
              * Published in the *New England Journal of Medicine (NEJM)*, **Fagan's Nomogram** visualizes these three steps on three vertical parallel scales.
              * Drawing a single straight line from the patient's Pre-Test Probability through the test's Likelihood Ratio intersects the right axis precisely at the Post-Test Probability!
            """)

        st.info(r"""
        📐 **Why are Likelihood Ratios Preferred over Sensitivity and Specificity in Bedside Practice?**  
        Sensitivity and Specificity must be considered together as a pair, making it difficult to calculate patient risk at the bedside. 
        A Likelihood Ratio synthesizes both metrics into a **single, clinically actionable multiplier** that directly tells the doctor how far to shift their diagnostic suspicion!
        """)


def render_likelihood_ratios_diagnostics(evaluations: List[Dict[str, Any]]):
    """Renders tri-color prerequisite diagnostic cards for Diagnostic Likelihood Ratios."""
    st.markdown("### 🚦 Tri-Color Bias & Methodological Diagnostics (STARD & QUADAS-2)")
    st.caption("Verifies reference standard independence (incorporation bias), clinical spectrum representativeness, review blinding, and cell count adequacy.")

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


def render_likelihood_ratios_dashboard(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders primary diagnostic likelihood ratio metric cards with 95% CIs and post-test shifts."""
    st.markdown("### 📊 Primary Likelihood Ratio Metrics & Post-Test Shifts")

    lr_pos = calcs["lr_pos"]
    lr_pos_ll = calcs["lr_pos_ci_lower"]
    lr_pos_ul = calcs["lr_pos_ci_upper"]

    lr_neg = calcs["lr_neg"]
    lr_neg_ll = calcs["lr_neg_ci_lower"]
    lr_neg_ul = calcs["lr_neg_ci_upper"]

    pre_prob_pct = calcs["pre_test_prob"] * 100.0
    post_prob_pos_pct = calcs["post_test_prob_pos"] * 100.0
    post_prob_neg_pct = calcs["post_test_prob_neg"] * 100.0

    shift_pos = post_prob_pos_pct - pre_prob_pct
    shift_neg = post_prob_neg_pct - pre_prob_pct

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Positive Likelihood Ratio (Positive LR, LR+)",
            value=f"{lr_pos:.2f}",
            delta=f"Rule-In Power: {calcs['clinical_strength_pos'].split('(')[0].strip()}",
            delta_color="normal",
            help="Ratio of True Positive Rate to False Positive Rate: Sensitivity / (1 - Specificity). Values > 10 indicate conclusive rule-in power."
        )
        st.caption(f"**Simel 95% CI:** [{lr_pos_ll:.2f}, {lr_pos_ul:.2f}]")

    with c2:
        st.metric(
            label="Negative Likelihood Ratio (Negative LR, LR-)",
            value=f"{lr_neg:.3f}",
            delta=f"Rule-Out Power: {calcs['clinical_strength_neg'].split('(')[0].strip()}",
            delta_color="inverse",
            help="Ratio of False Negative Rate to True Negative Rate: (1 - Sensitivity) / Specificity. Values < 0.10 indicate conclusive rule-out power."
        )
        st.caption(f"**Simel 95% CI:** [{lr_neg_ll:.3f}, {lr_neg_ul:.3f}]")

    with c3:
        st.metric(
            label="Post-Test Probability if Positive (T+)",
            value=f"{post_prob_pos_pct:.1f}%",
            delta=f"{shift_pos:+.1f}% from baseline ({pre_prob_pct:.1f}%)",
            delta_color="normal",
            help="Updated patient probability of disease if test is positive: Pre-Test Odds * LR+ converted back to probability."
        )
        st.caption(f"**Post-Test Odds (+):** {calcs['post_test_odds_pos']:.3f}")

    with c4:
        st.metric(
            label="Post-Test Probability if Negative (T-)",
            value=f"{post_prob_neg_pct:.1f}%",
            delta=f"{shift_neg:+.1f}% from baseline ({pre_prob_pct:.1f}%)",
            delta_color="inverse",
            help="Updated patient probability of disease if test is negative: Pre-Test Odds * LR- converted back to probability."
        )
        st.caption(f"**Post-Test Odds (-):** {calcs['post_test_odds_neg']:.4f}")


def render_likelihood_ratios_visualizations(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders interactive Plotly Fagan Nomogram and Pre-to-Post-Test Probability Shift Waterfall chart."""
    st.markdown("### 📈 Visualizing Likelihood Ratios & Fagan's Nomogram")

    pre_p = calcs["pre_test_prob"]
    lr_pos = calcs["lr_pos"]
    lr_neg = calcs["lr_neg"]
    post_p_pos = calcs["post_test_prob_pos"]
    post_p_neg = calcs["post_test_prob_neg"]

    shift_pos = (post_p_pos - pre_p) * 100.0
    shift_neg = (post_p_neg - pre_p) * 100.0

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**1. Interactive Fagan's Nomogram (Pre-to-Post-Test Probability Updating)**")

        # Fagan's Nomogram Plotly Construction
        # Coordinates: X=0 (Pre-Test), X=1 (Likelihood Ratio), X=2 (Post-Test)
        # Probabilities mapped to log-odds coordinates for straight-line alignment:
        # y = logit(p) = ln(p / (1 - p))
        # LR axis y-coordinate = 0.5 * ln(LR) so that straight ray connecting Pre and Post passes through LR

        # Log-odds transformations
        y_pre = np.log(pre_p / (1.0 - pre_p))
        y_post_pos = np.log(post_p_pos / (1.0 - post_p_pos))
        y_post_neg = np.log(post_p_neg / (1.0 - post_p_neg))
        
        # Midpoint coordinate for LR axis: (y_pre + y_post) / 2
        y_lr_pos = (y_pre + y_post_pos) / 2.0
        y_lr_neg = (y_pre + y_post_neg) / 2.0

        fig_nomo = go.Figure()

        # 1. Vertical Axis Guides (Lines at X=0, X=1, X=2)
        axis_y_min, axis_y_max = -6.9, 6.9  # approx 0.1% to 99.9%
        for x_pos in [0, 1, 2]:
            fig_nomo.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[axis_y_min, axis_y_max],
                mode="lines",
                line=dict(color="#cbd5e1", width=2),
                showlegend=False,
                hoverinfo="skip"
            ))

        # 2. Probability Ticks & Labels on Left Axis (X=0) and Right Axis (X=2)
        prob_ticks = [0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999]
        for pt in prob_ticks:
            y_tick = np.log(pt / (1.0 - pt))
            label = f"{pt*100:.1f}%" if pt < 0.1 or pt > 0.9 else f"{pt*100:.0f}%"
            # Left ticks
            fig_nomo.add_trace(go.Scatter(
                x=[-0.08], y=[y_tick], mode="text", text=[label],
                textposition="middle left", textfont=dict(size=9, color="#64748b"),
                showlegend=False, hoverinfo="skip"
            ))
            # Right ticks
            fig_nomo.add_trace(go.Scatter(
                x=[2.08], y=[y_tick], mode="text", text=[label],
                textposition="middle right", textfont=dict(size=9, color="#64748b"),
                showlegend=False, hoverinfo="skip"
            ))

        # 3. Ray Line for Positive Test Result (Pre -> LR+ -> Post+)
        fig_nomo.add_trace(go.Scatter(
            x=[0, 1, 2],
            y=[y_pre, y_lr_pos, y_post_pos],
            mode="lines+markers",
            line=dict(color="#ef4444", width=2.5),
            marker=dict(size=[10, 10, 10], color=["#ef4444", "#ef4444", "#ef4444"]),
            name=f"Positive Test (LR+ = {lr_pos:.2f} ➔ Post = {post_p_pos*100.0:.1f}%)"
        ))

        # 4. Ray Line for Negative Test Result (Pre -> LR- -> Post-)
        fig_nomo.add_trace(go.Scatter(
            x=[0, 1, 2],
            y=[y_pre, y_lr_neg, y_post_neg],
            mode="lines+markers",
            line=dict(color="#10b981", width=2.5, dash="dash"),
            marker=dict(size=[10, 10, 10], color=["#10b981", "#10b981", "#10b981"]),
            name=f"Negative Test (LR- = {lr_neg:.3f} ➔ Post = {post_p_neg*100.0:.1f}%)"
        ))

        # Axis Title annotations
        fig_nomo.add_annotation(x=0, y=7.4, text="<b>Pre-Test Prob</b>", showarrow=False, font=dict(size=11, color="#1e293b"))
        fig_nomo.add_annotation(x=1, y=7.4, text="<b>Likelihood Ratio</b>", showarrow=False, font=dict(size=11, color="#1e293b"))
        fig_nomo.add_annotation(x=2, y=7.4, text="<b>Post-Test Prob</b>", showarrow=False, font=dict(size=11, color="#1e293b"))

        fig_nomo.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 2.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-7.5, 8.2]),
            height=370,
            margin=dict(l=20, r=20, t=30, b=20),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_nomo, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### 🔍 Understanding Fagan's Nomogram")
            st.markdown(
                "Invented by Dr. Thomas Fagan in 1975, the nomogram plots Bayes' Theorem as straight lines:\n"
                f"* **Left Vertical Axis (Pre-Test Probability):** Your baseline suspicion ({pre_p*100.0:.1f}%) based on patient history and risk score.\n"
                f"* **Middle Vertical Axis (Likelihood Ratio):** The intrinsic discriminatory power of the test (Positive LR = {lr_pos:.2f}, Negative LR = {lr_neg:.3f}).\n"
                f"* **Right Vertical Axis (Post-Test Probability):** The definitive post-test probability of disease.\n"
                f"* **Red Solid Ray (Positive Test):** Connects {pre_p*100.0:.1f}% through Positive LR ({lr_pos:.2f}) to arrive at **{post_p_pos*100.0:.1f}%** post-test probability.\n"
                f"* **Green Dashed Ray (Negative Test):** Connects {pre_p*100.0:.1f}% through Negative LR ({lr_neg:.3f}) to drop down to **{post_p_neg*100.0:.1f}%** post-test probability."
            )

    with col_chart2:
        st.markdown("**2. Pre-to-Post-Test Probability Shift Comparison**")

        # Waterfall / Probability Shift Bar Chart
        categories = ["1. Baseline Pre-Test", "2. Post-Test if Positive (T+)", "3. Post-Test if Negative (T-)"]
        probs = [pre_p * 100.0, post_p_pos * 100.0, post_p_neg * 100.0]
        colors = ["#64748b", "#ef4444", "#10b981"]

        fig_shift = go.Figure(go.Bar(
            x=categories,
            y=probs,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probs],
            textposition="auto"
        ))

        fig_shift.update_layout(
            yaxis=dict(title="Disease Probability (%)", range=[0, 105]),
            height=370,
            margin=dict(l=30, r=30, t=30, b=30),
            template="plotly_white"
        )
        st.plotly_chart(fig_shift, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### ⚖️ Clinical Impact of the Likelihood Ratios")
            st.markdown(
                "This chart demonstrates the magnitude of the diagnostic shift:\n"
                f"* **Starting Baseline Suspicion:** {pre_p*100.0:.1f}% pre-test risk based on clinical scoring.\n"
                f"* **If Test is Positive (T+):** Driven by Positive LR ({lr_pos:.2f}), disease risk escalates by **{shift_pos:+.1f}%** to **{post_p_pos*100.0:.1f}%** ({calcs['clinical_strength_pos']}).\n"
                f"* **If Test is Negative (T-):** Driven by Negative LR ({lr_neg:.3f}), disease risk plunges by **{abs(shift_neg):.1f}%** down to **{post_p_neg*100.0:.1f}%** ({calcs['clinical_strength_neg']}).\n"
                f"* **Diagnostic Odds Ratio (DOR = {calcs['dor']:.1f}):** Combines rule-in and rule-out power into a single prevalence-independent metric $(LR+ / LR- = {calcs['dor']:.1f})$."
            )

    # 3. Interactive Contingency Matrix Display
    st.markdown("**3. Diagnostic 2×2 Contingency Matrix with Likelihood Ratio Dimensions**")
    test_pos_label, test_neg_label = meta.get("test_levels", ("Test Positive", "Test Negative"))
    cond_pos_label, cond_neg_label = meta.get("condition_levels", ("Disease Present (Gold +)", "Disease Absent (Gold -)"))

    tp = calcs["tp"]
    fp = calcs["fp"]
    fn = calcs["fn"]
    tn = calcs["tn"]

    df_2x2_display = pd.DataFrame([
        {
            "Test Outcome": f"🔴 {test_pos_label}",
            f"{cond_pos_label}": f"{tp:,} (True Positives, TP)",
            f"{cond_neg_label}": f"{fp:,} (False Positives, FP)",
            "Total Test Marginals": f"{tp + fp:,}",
            "Likelihood Ratio Role": f"Positive LR (LR+) = {calcs['lr_pos']:.2f} (Sensitivity {calcs['sensitivity']*100.0:.1f}% / FPR {calcs['fpr']*100.0:.1f}%)"
        },
        {
            "Test Outcome": f"🟢 {test_neg_label}",
            f"{cond_pos_label}": f"{fn:,} (False Negatives, FN)",
            f"{cond_neg_label}": f"{tn:,} (True Negatives, TN)",
            "Total Test Marginals": f"{fn + tn:,}",
            "Likelihood Ratio Role": f"Negative LR (LR-) = {calcs['lr_neg']:.3f} (FNR {calcs['fnr']*100.0:.1f}% / Specificity {calcs['specificity']*100.0:.1f}%)"
        },
        {
            "Test Outcome": "📊 Total Gold Standard Cohort",
            f"{cond_pos_label}": f"{calcs['total_diseased']:,} (Diseased)",
            f"{cond_neg_label}": f"{calcs['total_healthy']:,} (Healthy)",
            "Total Test Marginals": f"{calcs['total_sample']:,} (Grand Total N)",
            "Likelihood Ratio Role": f"Diagnostic Odds Ratio (DOR) = {calcs['dor']:.1f} (LR+ / LR-)"
        }
    ])

    st.dataframe(df_2x2_display, use_container_width=True, hide_index=True)


def render_likelihood_ratios_probability_simulator(calcs: Dict[str, Any]):
    """Renders an interactive live Pre-Test Probability updating simulator."""
    st.markdown("### 🎛️ Live Pre-Test Probability Updating Simulator")
    st.caption("Adjust the patient's pre-test probability below to observe how Positive LR (LR+) and Negative LR (LR-) update their post-test diagnosis.")

    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        custom_pre_p = st.slider(
            "Patient Pre-Test Probability (%):",
            min_value=0.5,
            max_value=95.0,
            value=float(np.clip(calcs["pre_test_prob"] * 100.0, 0.5, 95.0)),
            step=0.5,
            format="%.1f%%",
            help="Set patient's pre-test probability based on validated risk scores (e.g. Wells Score, Centor Criteria, Alvarado Score)."
        ) / 100.0

        # Odds Bayes updating
        custom_pre_odds = custom_pre_p / (1.0 - custom_pre_p)
        custom_post_odds_pos = custom_pre_odds * calcs["lr_pos"]
        custom_post_prob_pos = custom_post_odds_pos / (1.0 + custom_post_odds_pos)

        custom_post_odds_neg = custom_pre_odds * calcs["lr_neg"]
        custom_post_prob_neg = custom_post_odds_neg / (1.0 + custom_post_odds_neg)

    with sim_col2:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label="Simulated Pre-Test Odds",
                value=f"{custom_pre_odds:.3f}",
                help="p / (1 - p)"
            )
        with m2:
            st.metric(
                label="Simulated Post-Test if Positive (T+)",
                value=f"{custom_post_prob_pos * 100.0:.1f}%",
                delta=f"{(custom_post_prob_pos - custom_pre_p)*100.0:+.1f}%",
                help="Pre-Test Odds * LR+ converted to probability."
            )
        with m3:
            st.metric(
                label="Simulated Post-Test if Negative (T-)",
                value=f"{custom_post_prob_neg * 100.0:.1f}%",
                delta=f"{(custom_post_prob_neg - custom_pre_p)*100.0:+.1f}%",
                delta_color="inverse",
                help="Pre-Test Odds * LR- converted to probability."
            )


def render_likelihood_ratios_excel_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 4-step Microsoft Excel Implementation Guide with cell coordinates and formulas."""
    st.markdown("### 📗 Microsoft Excel Implementation Workflow for Likelihood Ratios & Odds Bayes Updating")
    st.markdown("Follow the 4-step sequence: **Set up 2×2 Layout** ➔ **Calculate Positive LR (LR+) & Negative LR (LR-)** ➔ **Calculate Pre-to-Post Odds & Probabilities** ➔ **Build Excel Probability Shift Waterfall**.")

    tp, fp, fn, tn = calcs["tp"], calcs["fp"], calcs["fn"], calcs["tn"]
    total_d = calcs["total_diseased"]
    total_h = calcs["total_healthy"]
    total_pos = tp + fp
    total_neg = fn + tn
    total_n = calcs["total_sample"]

    test_pos_label, test_neg_label = meta.get("test_levels", ("Test Positive", "Test Negative"))

    with st.expander("📌 Step 1: Excel 2×2 Data Layout & Cell References", expanded=False):
        st.markdown(f"""
        Structure your Excel worksheet with the 2×2 diagnostic matrix in `Cells A1:E4` and calculation outputs in `Cells A6:B18`:
        
        * **Column A (Rows 2:4):** Test Finding (`Row 2: Positive`, `Row 3: Negative`, `Row 4: Total`)
        * **Column B (Rows 2:4):** Gold Standard Diseased (`B2`: $TP = {tp}$, `B3`: $FN = {fn}$, `B4`: `=SUM(B2:B3)`)
        * **Column C (Rows 2:4):** Gold Standard Healthy (`C2`: $FP = {fp}$, `C3`: $TN = {tn}$, `C4`: `=SUM(C2:C3)`)
        * **Column D (Rows 2:4):** Marginal Totals (`D2`: `=SUM(B2:C2)`, `D3`: `=SUM(B3:C3)`, `D4`: `=SUM(B4:C4)`)
        """)

        preview_grid = pd.DataFrame([
            {"Excel_Row": "Row 1 (Headers)", "Col_A": "Test_Finding", "Col_B": "Condition_Present_Gold_Pos", "Col_C": "Condition_Absent_Gold_Neg", "Col_D": "Test_Marginal_Total"},
            {"Excel_Row": "Row 2 (Test Positive)", "Col_A": test_pos_label, "Col_B": str(tp), "Col_C": str(fp), "Col_D": f"{total_pos} (=SUM(B2:C2))"},
            {"Excel_Row": "Row 3 (Test Negative)", "Col_A": test_neg_label, "Col_B": str(fn), "Col_C": str(tn), "Col_D": f"{total_neg} (=SUM(B3:C3))"},
            {"Excel_Row": "Row 4 (Totals)", "Col_A": "Total Gold Standard", "Col_B": f"{total_d} (=SUM(B2:B3))", "Col_C": f"{total_h} (=SUM(C2:C3))", "Col_D": f"{total_n} (=SUM(D2:D3))"}
        ])
        st.dataframe(preview_grid, use_container_width=True, hide_index=True)

        st.markdown("**Summary Output Cells Mapping (`Cells A6:B18`):**")
        df_summary_cells = pd.DataFrame({
            "Cell": ["B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18"],
            "Metric Name": [
                "Sensitivity (TPR)",
                "Specificity (TNR)",
                "False Positive Rate (FPR)",
                "False Negative Rate (FNR)",
                "Positive Likelihood Ratio (LR+)",
                "Negative Likelihood Ratio (LR-)",
                "Diagnostic Odds Ratio (DOR)",
                "Patient Pre-Test Probability",
                "Pre-Test Odds",
                "Post-Test Odds if Positive",
                "Post-Test Probability if Positive",
                "Post-Test Odds if Negative",
                "Post-Test Probability if Negative"
            ],
            "Excel Formula": [
                "=B2/B4",
                "=C3/C4",
                "=1-B7",
                "=1-B6",
                "=B6/B8",
                "=B9/B7",
                "=B10/B11",
                f"{calcs['pre_test_prob']:.2f}",
                "=B13/(1-B13)",
                "=B14*B10",
                "=B15/(1+B15)",
                "=B14*B11",
                "=B17/(1+B17)"
            ],
            "Calculated Value": [
                f"{calcs['sensitivity']:.4f}",
                f"{calcs['specificity']:.4f}",
                f"{calcs['fpr']:.4f}",
                f"{calcs['fnr']:.4f}",
                f"{calcs['lr_pos']:.4f}",
                f"{calcs['lr_neg']:.4f}",
                f"{calcs['dor']:.2f}",
                f"{calcs['pre_test_prob']:.4f}",
                f"{calcs['pre_test_odds']:.4f}",
                f"{calcs['post_test_odds_pos']:.4f}",
                f"{calcs['post_test_prob_pos']:.4f}",
                f"{calcs['post_test_odds_neg']:.4f}",
                f"{calcs['post_test_prob_neg']:.4f}"
            ]
        })
        st.dataframe(df_summary_cells, use_container_width=True, hide_index=True)

    with st.expander("📌 Step 2: Calculate Positive LR (LR+) & Negative LR (LR-) in Excel", expanded=False):
        st.markdown(r"""
        1. **Compute Sensitivity ($TPR$):** In cell `B6`, enter `=B2/B4`.
        2. **Compute Specificity ($TNR$):** In cell `B7`, enter `=C3/C4`.
        3. **Compute False Positive Rate ($FPR$):** In cell `B8`, enter `=1-B7`.
        4. **Compute False Negative Rate ($FNR$):** In cell `B9`, enter `=1-B6`.
        5. **Compute Positive Likelihood Ratio (LR+):** In cell `B10`, enter:
           ```excel
           =B6 / B8
           ```
           *(Sensitivity divided by False Positive Rate)*. Format as **Number with 2 decimal places**.
        6. **Compute Negative Likelihood Ratio (LR-):** In cell `B11`, enter:
           ```excel
           =B9 / B7
           ```
           *(False Negative Rate divided by Specificity)*. Format as **Number with 3 decimal places**.
        7. **Compute Diagnostic Odds Ratio (DOR):** In cell `B12`, enter:
           ```excel
           =B10 / B11
           ```
        """)

    with st.expander("📌 Step 3: Implement Odds Bayes Updating in Excel", expanded=False):
        st.markdown(r"""
        To update probability for an individual patient with Pre-Test Probability entered in cell `B13` (e.g. `0.20`):
        
        1. **Convert Pre-Test Probability to Pre-Test Odds:** In cell `B14`, enter:
           ```excel
           =B13 / (1 - B13)
           ```
        2. **Compute Post-Test Odds if Positive ($T+$):** In cell `B15`, enter:
           ```excel
           =B14 * B10
           ```
        3. **Convert Post-Test Odds to Post-Test Probability if Positive:** In cell `B16`, enter:
           ```excel
           =B15 / (1 + B15)
           ```
           Format as **Percentage** (e.g. `34.8%`).
        4. **Compute Post-Test Odds if Negative ($T-$):** In cell `B17`, enter:
           ```excel
           =B14 * B11
           ```
        5. **Convert Post-Test Odds to Post-Test Probability if Negative:** In cell `B18`, enter:
           ```excel
           =B17 / (1 + B17)
           ```
           Format as **Percentage** (e.g. `1.8%`).
        """)

    with st.expander("📌 Step 4: Build Probability Shift Waterfall Chart in Excel", expanded=False):
        st.markdown(r"""
        1. In an open worksheet area (e.g. `Cells G1:H4`), create a summary table:
           * `Cell G1`: `Diagnostic_Stage`, `Cell H1`: `Disease_Probability`
           * `Cell G2`: `1. Baseline Pre-Test`, `Cell H2`: `=B13`
           * `Cell G3`: `2. Post-Test if Positive (T+)`, `Cell H3`: `=B16`
           * `Cell G4`: `3. Post-Test if Negative (T-)`, `Cell H4`: `=B18`
        2. Highlight `Cells G1:H4`.
        3. Go to **Insert ➔ Charts ➔ Clustered Column Chart**.
        4. Color the Baseline bar in gray (`#64748b`), the Positive bar in red (`#ef4444`), and the Negative bar in green (`#10b981`).
        5. Add Data Labels to clearly demonstrate the diagnostic shift to clinicians!
        """)


def render_likelihood_ratios_python_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 5-step Python implementation guide with odds Bayes math, Simel CIs, and Plotly scripts."""
    st.markdown("### 🐍 Python Implementation Workflow for Diagnostic Likelihood Ratios")
    st.markdown("Follow the 5-step sequence: **Define Confusion Matrix & Validate Biases** ➔ **Calculate Positive LR (LR+) & Negative LR (LR-)** ➔ **Odds Bayes Updating Function** ➔ **Consolidated Production Script** ➔ **Plotly Fagan Nomogram Script**.")

    tp, fp, fn, tn = calcs["tp"], calcs["fp"], calcs["fn"], calcs["tn"]
    pre_p = calcs["pre_test_prob"]

    with st.expander("📌 Step 1: Define Confusion Matrix & Methodological Prerequisites", expanded=False):
        st.markdown(f"""
        ```python
        import numpy as np
        import pandas as pd
        from scipy import stats

        # 1. Input 2x2 Contingency Matrix Counts
        tp = {tp}   # True Positives
        fp = {fp}   # False Positives
        fn = {fn}   # False Negatives
        tn = {tn}   # True Negatives
        pre_test_prob = {pre_p}  # Baseline clinical pre-test suspicion

        total_diseased = tp + fn
        total_healthy = tn + fp
        total_sample = tp + fp + fn + tn

        print(f"Total Diseased (D+): {{total_diseased}}")
        print(f"Total Healthy (D-):  {{total_healthy}}")
        print(f"Pre-Test Probability: {{pre_test_prob * 100:.1f}}%")
        ```
        """)

    with st.expander("📌 Step 2: Compute Sensitivity, Specificity, Positive LR & Negative LR", expanded=False):
        st.markdown("""
        ```python
        # 1. Intrinsic Sensitivity & Specificity
        sensitivity = tp / total_diseased if total_diseased > 0 else 0.0
        specificity = tn / total_healthy if total_healthy > 0 else 0.0
        fpr = 1.0 - specificity  # False Positive Rate
        fnr = 1.0 - sensitivity  # False Negative Rate

        # 2. Positive Likelihood Ratio (LR+) = Sensitivity / FPR
        lr_pos = sensitivity / fpr if fpr > 0 else np.inf

        # 3. Negative Likelihood Ratio (LR-) = FNR / Specificity
        lr_neg = fnr / specificity if specificity > 0 else np.inf

        # 4. Diagnostic Odds Ratio (DOR)
        dor = lr_pos / lr_neg if lr_neg > 0 else np.nan

        print(f"Sensitivity: {{sensitivity * 100:.2f}}% | Specificity: {{specificity * 100:.2f}}%")
        print(f"Positive Likelihood Ratio (LR+): {{lr_pos:.2f}}")
        print(f"Negative Likelihood Ratio (LR-): {{lr_neg:.3f}}")
        print(f"Diagnostic Odds Ratio (DOR):     {{dor:.2f}}")
        ```
        """)

    with st.expander("📌 Step 3: Implement Odds Form of Bayes' Theorem Updating", expanded=False):
        st.markdown("""
        ```python
        def bayes_odds_update(pre_test_probability: float, likelihood_ratio: float):
            \"\"\"Updates pre-test probability to post-test probability using Odds Bayes' Theorem.\"\"\"
            p = float(pre_test_probability)
            # Step 1: Convert probability to odds
            pre_odds = p / (1.0 - p)
            # Step 2: Multiply by Likelihood Ratio
            post_odds = pre_odds * likelihood_ratio
            # Step 3: Convert odds back to probability
            post_prob = post_odds / (1.0 + post_odds)
            return post_odds, post_prob

        post_odds_pos, post_prob_pos = bayes_odds_update(pre_test_prob, lr_pos)
        post_odds_neg, post_prob_neg = bayes_odds_update(pre_test_prob, lr_neg)

        print(f"Baseline Pre-Test Probability:       {{pre_test_prob * 100:.1f}}%")
        print(f"Post-Test Probability if Positive (T+): {{post_prob_pos * 100:.1f}}% (Shift: {{(post_prob_pos - pre_test_prob)*100:+.1f}}%)")
        print(f"Post-Test Probability if Negative (T-): {{post_prob_neg * 100:.1f}}% (Shift: {{(post_prob_neg - pre_test_prob)*100:+.1f}}%)")
        ```
        """)

    with st.expander("📌 Step 4: Consolidated Production Likelihood Ratio Script", expanded=False):
        st.markdown(f"""
        ```python
        import numpy as np
        import pandas as pd
        from scipy import stats

        def run_likelihood_ratio_pipeline(tp: int, fp: int, fn: int, tn: int, pre_test_prob: float = 0.20, alpha: float = 0.05):
            \"\"\"Runs comprehensive likelihood ratio analysis with Simel 95% CIs and odds Bayes updating.\"\"\"
            # Apply 0.5 continuity correction if zero cells
            has_zero = (tp == 0 or fp == 0 or fn == 0 or tn == 0)
            c = 0.5 if has_zero else 0.0
            tp_c, fp_c, fn_c, tn_c = tp + c, fp + c, fn + c, tn + c

            tot_d = tp_c + fn_c
            tot_h = tn_c + fp_c

            sens = tp_c / tot_d
            spec = tn_c / tot_h
            fpr = 1.0 - spec
            fnr = 1.0 - sens

            lr_pos = sens / fpr
            lr_neg = fnr / spec
            dor = lr_pos / lr_neg

            z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

            # Simel log-scale 95% CI for LR+
            se_ln_pos = np.sqrt((fn_c / (tp_c * tot_d)) + (tn_c / (fp_c * tot_h)))
            lr_pos_ci = (np.exp(np.log(lr_pos) - z_crit * se_ln_pos), np.exp(np.log(lr_pos) + z_crit * se_ln_pos))

            # Simel log-scale 95% CI for LR-
            se_ln_neg = np.sqrt((tp_c / (fn_c * tot_d)) + (fp_c / (tn_c * tot_h)))
            lr_neg_ci = (np.exp(np.log(lr_neg) - z_crit * se_ln_neg), np.exp(np.log(lr_neg) + z_crit * se_ln_neg))

            # Odds Bayes updating
            pre_odds = pre_test_prob / (1.0 - pre_test_prob)
            post_odds_pos = pre_odds * lr_pos
            post_prob_pos = post_odds_pos / (1.0 + post_odds_pos)

            post_odds_neg = pre_odds * lr_neg
            post_prob_neg = post_odds_neg / (1.0 + post_odds_neg)

            return {{
                "Sensitivity": sens,
                "Specificity": spec,
                "Positive_LR": lr_pos,
                "Positive_LR_95CI": lr_pos_ci,
                "Negative_LR": lr_neg,
                "Negative_LR_95CI": lr_neg_ci,
                "DOR": dor,
                "Pre_Test_Probability": pre_test_prob,
                "Post_Test_Probability_Pos": post_prob_pos,
                "Post_Test_Probability_Neg": post_prob_neg
            }}

        # Run pipeline
        res = run_likelihood_ratio_pipeline(tp={tp}, fp={fp}, fn={fn}, tn={tn}, pre_test_prob={pre_p})
        print("=== Diagnostic Likelihood Ratios & Odds Bayes Summary ===")
        for k, v in res.items():
            if isinstance(v, tuple):
                print(f"{{k:<26}}: [{{v[0]:.2f}}, {{v[1]:.2f}}]")
            elif isinstance(v, float):
                print(f"{{k:<26}}: {{v:.3f}}" if "LR" in k or "DOR" in k else f"{{k:<26}}: {{v*100:.2f}}%")
        ```
        """)

    with st.expander("📌 Step 5: Generate Plotly Fagan's Nomogram Script", expanded=False):
        st.markdown(f"""
        ```python
        import numpy as np
        import plotly.graph_objects as go

        pre_p = {pre_p}
        lr_pos = {calcs['lr_pos']}
        lr_neg = {calcs['lr_neg']}
        post_pos = {calcs['post_test_prob_pos']}
        post_neg = {calcs['post_test_prob_neg']}

        # Logit coordinates
        y_pre = np.log(pre_p / (1.0 - pre_p))
        y_post_pos = np.log(post_pos / (1.0 - post_pos))
        y_post_neg = np.log(post_neg / (1.0 - post_neg))
        y_mid_pos = (y_pre + y_post_pos) / 2.0
        y_mid_neg = (y_pre + y_post_neg) / 2.0

        fig = go.Figure()
        # Nomogram axes
        for x in [0, 1, 2]:
            fig.add_trace(go.Scatter(x=[x, x], y=[-7, 7], mode="lines", line=dict(color="#cbd5e1", width=2), showlegend=False))

        # Ray lines
        fig.add_trace(go.Scatter(x=[0, 1, 2], y=[y_pre, y_mid_pos, y_post_pos], mode="lines+markers", name=f"T+ (LR+={{lr_pos:.2f}})", line=dict(color="#ef4444", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1, 2], y=[y_pre, y_mid_neg, y_post_neg], mode="lines+markers", name=f"T- (LR-={{lr_neg:.3f}})", line=dict(color="#10b981", width=2, dash="dash")))

        fig.update_layout(title="Fagan's Nomogram", template="plotly_white")
        fig.show()
        ```
        """)


def render_likelihood_ratios_download_tab(df_contingency: pd.DataFrame, df_microdata: pd.DataFrame, meta: Dict[str, Any]):
    """Renders data download tab for individual patient microdata and 2x2 contingency matrix."""
    st.markdown("### 💾 Live Likelihood Ratio Data Downloads")
    st.caption("Download individual patient microdata or the aggregated 2×2 diagnostic contingency table.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Aggregated 2×2 Contingency Matrix Table (CSV)**")
        csv_matrix = df_contingency.to_csv(index=False)
        st.download_button(
            label="📥 Download 2×2 Contingency Table (CSV)",
            data=csv_matrix,
            file_name=f"likelihood_ratios_2x2_matrix_{meta.get('scenario_key', 'custom')}.csv",
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
            file_name=f"likelihood_ratios_microdata_{meta.get('scenario_key', 'custom')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df_microdata.head(6), use_container_width=True, hide_index=True)


def render_likelihood_ratios_rationale_tab(method_info: Dict[str, Any], calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the comprehensive Rationale & Epidemiological Framework tab for Likelihood Ratios."""
    st.subheader(f"Why Diagnostic Likelihood Ratios (LR+, LR-) & Fagan's Nomogram? ({method_info['name']})")
    st.markdown("""
    When evaluating a patient in clinical practice, doctors and nurses do not make decisions in a vacuum. 
    Every clinical consultation begins with a baseline degree of suspicion (**Pre-Test Probability**) formed from 
    the patient's age, medical history, physical examination, and presenting symptoms. 
    
    When a diagnostic test (e.g. an emergency high-sensitivity D-Dimer, a point-of-care rapid Strep A antigen cassette, or an abdominal ultrasound) 
    returns positive or negative, the clinician needs to know: **How much does this result change my suspicion?**
    
    Traditional Predictive Values (Positive Predictive Value [PPV] and Negative Predictive Value [NPV]) cannot easily be transferred 
    from published research studies into a local clinic because **predictive values change dramatically whenever population prevalence changes**. 
    In contrast, **Diagnostic Likelihood Ratios (Positive LR [LR+] and Negative LR [LR-])** are portable, prevalence-independent properties of the test. 
    They serve as mathematical multipliers that update a patient's pre-test odds into post-test odds via **Bayes' Theorem** and **Fagan's Nomogram**.
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
    st.markdown("### 🔍 The Diagnostic Reasoning Framework: Test Properties vs. Clinical Probabilities")
    st.markdown(r"""
    | Diagnostic Dimension | Sensitivity & Specificity | Predictive Values (PPV & NPV) | Diagnostic Likelihood Ratios (LR+ & LR-) | Diagnostic Odds Ratio (DOR) |
    | :--- | :--- | :--- | :--- | :--- |
    | **Primary Question** | How well does the test identify true disease vs. true health in a lab? | If my patient tests positive (or negative), what is the actual chance they are diseased? | By what mathematical factor does this test result multiply my patient's baseline odds of disease? | What is the overall discriminating power of the test across all thresholds combined? |
    | **Perspective** | **Retrospective / Laboratory:** Looks backward from confirmed disease status ($P(\text{Test} \mid \text{Disease})$). | **Prospective / Patient:** Looks forward from test result to true disease state ($P(\text{Disease} \mid \text{Test})$). | **Bayesian / Multiplier:** Quantifies relative probability shifts ($P(\text{Result} \mid D+) / P(\text{Result} \mid D-)$). | **Global Summary:** Combines rule-in and rule-out performance into a single metric. |
    | **Prevalence Dependent?** | **No.** Inherent properties of the diagnostic assay. | **Yes, extremely.** PPV collapses toward zero in low-prevalence screening populations. | **No.** Prevalence-independent; portable across primary care, emergency, and specialist clinics. | **No.** Prevalence-independent odds ratio. |
    | **Mathematical Formula** | $\text{Sens} = \frac{a}{a+c}, \quad \text{Spec} = \frac{d}{b+d}$ | $\text{PPV} = \frac{a}{a+b}, \quad \text{NPV} = \frac{d}{c+d}$ | $\text{LR+} = \frac{\text{Sens}}{1 - \text{Spec}}, \quad \text{LR-} = \frac{1 - \text{Sens}}{\text{Spec}}$ | $\text{DOR} = \frac{\text{LR+}}{\text{LR-}} = \frac{a \times d}{b \times c}$ |
    | **Null Value (Useless Test)** | $\text{Sens} = 0.50, \text{Spec} = 0.50$ (Coin flip) | Equal to Baseline Population Prevalence ($P$) | **$1.00$** (Does not change odds of disease at all) | **$1.00$** (Equal odds in diseased vs. healthy) |
    | **Clinical Utility** | Screening triage (SnNOut) or confirmation (SpPIn). | Counseling individual patients on test report meaning. | Updating individual patient probability on Fagan's Nomogram. | Comparing competing diagnostic technologies in meta-analyses. |
    """)

    st.markdown("#### 📖 Mathematical Formulae Explained in Plain English (Zero Prior Knowledge Assumed)")
    st.markdown(r"""
    Medical diagnostic formulas can appear daunting at first glance. Here is every single formula explained step-by-step in intuitive, jargon-free plain English:

    1. **Positive Likelihood Ratio (Positive LR, LR+):**
       $$\text{Positive Likelihood Ratio } (LR+) = \frac{\text{Sensitivity}}{1 - \text{Specificity}} = \frac{\text{True Positive Rate}}{\text{False Positive Rate}} = \frac{a / (a + c)}{b / (b + d)}$$
       * **What the Formula Measures:** The numerator is the probability that a sick person tests positive. The denominator is the probability that a healthy person mistakenly tests positive (a false alarm).
       * **In Plain English:** $LR+$ answers: *"How many times more likely is a positive test result to come from someone who is truly sick compared to someone who is completely healthy?"*
       * **Evidence-Based Medicine (EBM) Benchmarks:**
         * **$LR+ \ge 10$ (Conclusive Rule-In):** Generates large, conclusive shifts in disease probability (e.g. jumping from 20% to > 85%). A positive test virtually confirms the diagnosis.
         * **$LR+ = 5 \text{ to } 10$ (Moderate Rule-In):** Generates meaningful diagnostic shifts (e.g. jumping from 20% to ~60%).
         * **$LR+ = 2 \text{ to } 5$ (Small Rule-In):** Weak diagnostic shift; requires additional confirmatory testing.
         * **$LR+ = 1.0$ (Zero Diagnostic Value):** The test is completely uninformative. A positive result does not change probability by even a fraction of a percent.

    2. **Negative Likelihood Ratio (Negative LR, LR-):**
       $$\text{Negative Likelihood Ratio } (LR-) = \frac{1 - \text{Sensitivity}}{\text{Specificity}} = \frac{\text{False Negative Rate}}{\text{True Negative Rate}} = \frac{c / (a + c)}{d / (b + d)}$$
       * **What the Formula Measures:** The numerator is the probability that a sick person mistakenly tests negative (a missed case). The denominator is the probability that a healthy person correctly tests negative.
       * **In Plain English:** $LR-$ answers: *"How many times less likely is a negative test result in someone who is sick compared to someone who is healthy?"*
       * **Evidence-Based Medicine (EBM) Benchmarks:**
         * **$LR- \le 0.10$ (Conclusive Rule-Out):** Generates massive downward shifts in disease probability (e.g. dropping an intermediate 20% suspicion down to < 2%). Safely rules out the condition and spares patients invasive or risky procedures.
         * **$LR- = 0.10 \text{ to } 0.20$ (Moderate Rule-Out):** Generates meaningful diagnostic reduction.
         * **$LR- = 0.20 \text{ to } 0.50$ (Small Rule-Out):** Slight reduction in suspicion; cannot safely rule out dangerous illness alone.
         * **$LR- = 1.0$ (Zero Diagnostic Value):** A negative result does not reduce disease suspicion at all.

    3. **Why Can't We Directly Multiply Probabilities? (The Role of Odds):**
       * Probability is bounded strictly between $0\%$ ($0.0$) and $100\%$ ($1.0$).
       * If a patient has an $80\%$ probability of illness and tests positive with an $LR+$ of $15$, multiplying $80\% \times 15 = 1,200\%$, which is mathematically impossible!
       * To overcome this, Bayes' Theorem converts Probability into **Odds** (which can range freely from $0$ to $+\infty$), multiplies by the Likelihood Ratio, and then converts the resulting odds back into a valid Probability.

    4. **Step 1: Converting Probability into Odds (Pre-Test Odds):**
       $$\text{Pre-Test Odds} = \frac{\text{Pre-Test Probability}}{1 - \text{Pre-Test Probability}} = \frac{p}{1 - p}$$
       * **In Plain English:** If 20 out of 100 emergency patients with chest pain have pulmonary embolism, the probability is $20\%$ ($0.20$). That means 20 people are sick and 80 are healthy. The odds are $20 / 80 = 0.25$ (expressed as "1 to 4 odds").

    5. **Step 2: The Odds Form of Bayes' Theorem (The Core Calculation):**
       $$\text{Post-Test Odds} = \text{Pre-Test Odds} \times \text{Likelihood Ratio}$$
       * **In Plain English:** This is the elegant engine of medical diagnosis:
         * If the test is **Positive**, multiply baseline odds by $LR+$: $\text{Post-Test Odds} = \text{Pre-Test Odds} \times LR+$.
         * If the test is **Negative**, multiply baseline odds by $LR-$: $\text{Post-Test Odds} = \text{Pre-Test Odds} \times LR-$.

    6. **Step 3: Converting Post-Test Odds Back into Probability:**
       $$\text{Post-Test Probability} = \frac{\text{Post-Test Odds}}{1 + \text{Post-Test Odds}}$$
       * **In Plain English:** After multiplying your odds, this step turns the updated odds back into an intuitive percentage probability that you can directly explain to the patient.

    7. **How Fagan's Nomogram Works (The Graphical Slide-Rule):**
       * In 1975, Dr. Terence Fagan created a graphical nomogram with three vertical parallel axes:
         * **Left Axis:** Pre-Test Probability (scaled on a logarithmic odds scale).
         * **Middle Axis:** Likelihood Ratio (scaled on a logarithmic scale from 0.001 to 1,000).
         * **Right Axis:** Post-Test Probability (scaled on a logarithmic odds scale).
       * **How to Read It:** By placing a straight ruler connecting your patient's starting suspicion on the left axis through the test's Likelihood Ratio in the center, the ruler's intersection with the right axis instantly gives the exact Bayesian Post-Test Probability without doing any arithmetic!

    8. **Diagnostic Odds Ratio (DOR):**
       $$\text{Diagnostic Odds Ratio } (DOR) = \frac{LR+}{LR-} = \frac{a \times d}{b \times c} = \frac{\text{TP} \times \text{TN}}{\text{FP} \times \text{FN}}$$
       * **In Plain English:** A single summary number of test performance. It tells you how many times higher the odds of a positive test are in diseased patients compared to healthy patients. 
       * A test with $DOR = 1.0$ has zero diagnostic ability. High-quality clinical tests often feature $DOR > 50$ or $100$.

    9. **Log-Scale 95% Confidence Intervals for Likelihood Ratios (Simel et al., 1991):**
       $$\text{SE}(\ln LR+) = \sqrt{\frac{1}{a} - \frac{1}{a+c} + \frac{1}{b} - \frac{1}{b+d}}, \quad 95\% \text{ CI} = \exp\left[\ln(LR+) \pm 1.96 \cdot \text{SE}(\ln LR+)\right]$$
       $$\text{SE}(\ln LR-) = \sqrt{\frac{1}{c} - \frac{1}{a+c} + \frac{1}{d} - \frac{1}{b+d}}, \quad 95\% \text{ CI} = \exp\left[\ln(LR-) \pm 1.96 \cdot \text{SE}(\ln LR-)\right]$$
       * **In Plain English:** Because Likelihood Ratios are skewed ratios that cannot be less than zero, we calculate the standard error on the natural logarithm scale (where distributions are symmetrical and normal), construct the interval, and then exponentiate back. This guarantees mathematically sound bounds that never dip below zero.
    """)


def render_likelihood_ratios_module(active_method_key: str, method_info: Dict[str, Any]):
    """Top-level controller for Diagnostic Likelihood Ratios & Fagan's Nomogram."""
    st.markdown("## 🎯 Diagnostic Likelihood Ratios (LR+, LR-) & Fagan's Nomogram")
    st.caption("Quantifies diagnostic discrimination and pre-to-post-test probability updating using the Odds Form of Bayes' Theorem and Fagan's Nomogram.")

    # 1. Plain English Pedagogical Guide
    render_likelihood_ratios_plain_english_guide()

    # 2. Scenario Selection Sidebar / Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🩺 Clinical Likelihood Ratio Scenario")

    scenario_options = {
        "pulmonary_embolism_d_dimer": "🫁 Inner City Hospital A&E Pulmonary Embolism (D-Dimer: Conclusive Rule-Out, LR- = 0.07)",
        "strep_throat_centor_rapid_antigen": "🧪 All Meadows Primary Care Strep A (Rapid Antigen: Conclusive Rule-In, LR+ = 20.89)",
        "appendicitis_alvarado_ultrasound": "🩺 Riverside Children's Hospital Appendicitis (Ultrasound: Balanced, LR+ = 14.94, LR- = 0.13)",
        "custom": "✏️ Custom 2×2 Likelihood Ratio & Fagan Nomogram Input"
    }

    selected_scenario = st.sidebar.selectbox(
        "Choose Clinical Scenario:",
        options=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
        index=0
    )

    if selected_scenario != "custom":
        df_contingency, df_microdata, meta = generate_likelihood_ratio_demo_data(selected_scenario)
        tp = meta["tp"]
        fp = meta["fp"]
        fn = meta["fn"]
        tn = meta["tn"]
        pre_test = meta.get("pre_test_prob", 0.20)
    else:
        st.sidebar.markdown("**Input Custom 2×2 Diagnostic Counts:**")
        tp = st.sidebar.number_input("True Positives (TP): Diseased with Positive Test", min_value=0, max_value=500000, value=180, step=1)
        fp = st.sidebar.number_input("False Positives (FP): Healthy with Positive Test", min_value=0, max_value=500000, value=30, step=1)
        fn = st.sidebar.number_input("False Negatives (FN): Diseased with Negative Test", min_value=0, max_value=500000, value=20, step=1)
        tn = st.sidebar.number_input("True Negatives (TN): Healthy with Negative Test", min_value=0, max_value=500000, value=770, step=1)

        pre_test_pct = st.sidebar.slider("Patient Pre-Test Probability (%):", min_value=0.5, max_value=95.0, value=20.0, step=0.5)
        pre_test = pre_test_pct / 100.0

        meta = {
            "title": "Custom Diagnostic Likelihood Ratio Investigation",
            "domain": "User-Defined Clinical Investigation",
            "description": "Custom user 2×2 diagnostic matrix evaluated with Positive Likelihood Ratio (LR+), Negative Likelihood Ratio (LR-), and Fagan's Nomogram.",
            "test_name": "Index Diagnostic Test",
            "condition_name": "Target Condition",
            "test_levels": ("Test Positive", "Test Negative"),
            "condition_levels": ("Condition Present (Gold +)", "Condition Absent (Gold -)"),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "pre_test_prob": pre_test,
            "scenario_key": "custom",
            "clinical_takeaway": "Custom parameters evaluated with Simel et al. 95% log-scale confidence intervals and Fagan's Nomogram ray lines.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
        df_contingency = pd.DataFrame([
            {"Test Finding": "Test Positive", "Condition Present": tp, "Condition Absent": fp, "Total": tp + fp},
            {"Test Finding": "Test Negative", "Condition Present": fn, "Condition Absent": tn, "Total": fn + tn},
            {"Test Finding": "Total", "Condition Present": tp + fn, "Condition Absent": fp + tn, "Total": tp + fp + fn + tn}
        ])
        df_microdata = pd.DataFrame([
            {"Patient_ID": f"PT_{i+1:04d}", "Test_Result": "Positive", "Condition": "Present"} for i in range(min(tp, 20))
        ])

    calcs = calculate_likelihood_ratios(tp, fp, fn, tn, pre_test_prob=pre_test)
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
    evaluations = evaluate_likelihood_ratio_prerequisites(tp, fp, fn, tn)
    render_likelihood_ratios_diagnostics(evaluations)

    st.markdown("---")

    # 6. Primary Metrics Dashboard
    render_likelihood_ratios_dashboard(calcs, meta)

    st.markdown("---")

    # 7. Side-by-Side Visualizations & Balanced Explanatory Containers
    render_likelihood_ratios_visualizations(calcs, meta)

    st.markdown("---")

    # 8. Interactive Pre-Test Probability Simulator
    render_likelihood_ratios_probability_simulator(calcs)

    st.markdown("---")

    # 9. Methodology, Implementation Guides & Downloads Tabs
    tab_rationale, tab_excel, tab_python, tab_data = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Microsoft Excel Implementation Guide",
        "🐍 Python Production Script Guide",
        "💾 Live Data Downloads"
    ])

    with tab_rationale:
        render_likelihood_ratios_rationale_tab(method_info, calcs, meta)

    with tab_excel:
        render_likelihood_ratios_excel_guide(calcs, meta)

    with tab_python:
        render_likelihood_ratios_python_guide(calcs, meta)

    with tab_data:
        render_likelihood_ratios_download_tab(df_contingency, df_microdata, meta)


"""Statistical Process Control (SPC) & Quality Improvement (QI) UI Components.

Provides:
1. Zero-knowledge plain English introductory guide on Deming's Common vs Special Cause Variation.
2. Mode selector (Guided Quality Improvement Wizard vs. Direct Method Explorer).
3. Dynamic interactive simulation playgrounds with Plotly visualizations and NHS MDC icon badges.
4. Standard 4-tab container (Mathematical Formulation, Python Pipeline, Excel Guide, Practice Data).
"""

from typing import Dict, Any, List, Tuple, Optional
import streamlit as st

try:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    HAS_UI_STACK = True
except ImportError:
    pd = None
    np = None
    go = None
    HAS_UI_STACK = False

from src.spc_qi.spc_schema import SPC_METHODS, SPC_SCENARIOS
from src.spc_qi.spc_engine import (
    generate_run_chart_dataset,
    calculate_run_chart_model,
    generate_xmr_dataset,
    calculate_xmr_model,
    generate_p_chart_dataset,
    calculate_p_chart_model,
    generate_u_chart_dataset,
    calculate_u_chart_model,
    generate_mdc_dataset,
    calculate_mdc_capability_model
)


# =====================================================================
# 1. Plain-English Zero-Knowledge Pedagogical Guide
# =====================================================================

def render_spc_plain_english_guide() -> None:
    """Renders intuition-first conceptual guide to Deming's common vs special cause variation,
    the peril of tampering, and NHS England Making Data Count principles.
    """
    with st.expander("📖 Plain English Guide: Common vs. Special Cause Variation & The Peril of Tampering", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. The Core Philosophy of Statistical Process Control
            * In healthcare operations and clinical governance, boards and managers routinely ask: *"Are our waiting times getting worse? Did last month's infection spike mean our ward is failing?"*
            * W. Edwards Deming and Walter Shewhart proved that all processes exhibit two distinct types of variation:
              - **Common Cause Variation (Routine Process Noise):** Natural, everyday fluctuation inherent in the way the healthcare system is designed. It is completely predictable within statistical boundaries.
              - **Special Cause Variation (Genuine External Signal):** An unexpected deviation caused by a specific breakdown, equipment failure, seasonal outbreak, or genuine improvement reform.
            
            #### 2. The Golden Trap: Tampering (Rule 1 of Deming's Funnel)
            * **The Executive Tampering Trap:** When a hospital board reacts to a routine fluctuation (e.g. demanding an investigation because infections rose from 3 to 5 within common cause limits), they are **tampering with noise**.
            * Tampering treats common cause variation as if it were a special cause. Mathematically, tampering *doubles* system variance, demoralizes clinical staff, and distracts leadership from structural redesign.
            """)
        with col_g2:
            st.markdown(r"""
            #### 3. Process Stability vs. Process Capability
            * **Process Stability (SPC):** Answers: *"Is this system consistent and predictable over time?"* (Evaluated via control limits $\text{UCL}$ and $\text{LCL}$).
            * **Process Capability (Targets):** Answers: *"Can this system reliably achieve our statutory target?"* (Evaluated via $C_p, C_{pk}$ and NHS Assurance Icons).
            * **A system can be 100% stable but 100% incapable!** If an emergency department consistently admits 82% of patients within 4 hours with stable control limits of $[78\%, 86\%]$, it is predictable, but it is structurally incapable of meeting the NHS 95% target without redesigning the emergency flow pathway.

            #### 4. The 5 Healthcare SPC Architectures
            1. **Run Charts:** Quick non-parametric median analysis with 4 probability rules.
            2. **Shewhart XmR Charts:** Continuous operational metrics with 3-sigma limits based on moving ranges.
            3. **p-Charts & Laney p':** Proportions with overdispersion correction for large hospital trusts.
            4. **u-Charts & Laney u':** Non-conformity rates per unit of exposure (e.g. falls per 1,000 bed-days).
            5. **NHS 'Making Data Count' & Capability:** Variation & Assurance Icons paired with $C_p / C_{pk}$.
            """)


# =====================================================================
# 2. Mode Selector: Guided Wizard vs. Direct Method Explorer
# =====================================================================

def render_spc_mode_selector() -> Tuple[str, str]:
    """Renders top mode toggle between Guided Problem Wizard and Direct Method Explorer."""
    c_m1, c_m2 = st.columns([1.8, 2.2])
    
    preselected = st.session_state.get("spc_direct_method_key")
    if preselected and preselected in SPC_METHODS:
        st.session_state["spc_nav_mode_radio"] = "🔍 Direct Method Explorer"
    elif "spc_nav_mode_radio" not in st.session_state:
        st.session_state["spc_nav_mode_radio"] = "🎯 Guided Healthcare QI Wizard"

    with c_m1:
        nav_mode = st.radio(
            "Navigation Mode:",
            options=["🎯 Guided Healthcare QI Wizard", "🔍 Direct Method Explorer"],
            index=1 if st.session_state.get("spc_nav_mode_radio") == "🔍 Direct Method Explorer" else 0,
            horizontal=False,
            key="spc_nav_mode_radio"
        )
        
    with c_m2:
        if "Guided" in nav_mode:
            st.markdown("#### 🎯 Triage Your Quality Improvement Process:")
            q_metric_type = st.selectbox(
                "1. What type of operational or clinical metric are you monitoring?",
                options=[
                    "Continuous measure (e.g. Waiting hours, turnaround time, length of stay)",
                    "Proportion or percentage with fluctuating denominators (e.g. 30-day readmission %)",
                    "Rate of incidents per unit of exposure (e.g. Falls or infections per 1,000 bed-days)",
                    "Limited baseline data / early-stage project (Non-parametric median analysis)",
                    "Evaluating performance against statutory standard (NHS Making Data Count & Capability)"
                ],
                key="spc_wiz_metric"
            )
            
            if "Continuous" in q_metric_type:
                selected_key = "xmr_individual_chart"
            elif "Proportion" in q_metric_type:
                selected_key = "p_chart_laney"
            elif "Rate of incidents" in q_metric_type:
                selected_key = "u_chart_laney"
            elif "Limited baseline" in q_metric_type:
                selected_key = "healthcare_run_chart"
            else:
                selected_key = "making_data_count_capability"
                
            info = SPC_METHODS[selected_key]
            st.success(f"Recommended Tool: **{info['name']}** — *{info['tag']}*")
        else:
            st.markdown("#### 🔍 Select SPC Architecture Directly:")
            method_keys = list(SPC_METHODS.keys())
            method_labels = [f"{SPC_METHODS[k]['icon']} {SPC_METHODS[k]['name']}" for k in method_keys]
            
            default_idx = 0
            if preselected in method_keys:
                default_idx = method_keys.index(preselected)
                
            selected_label = st.selectbox(
                "Choose Method:",
                options=method_labels,
                index=default_idx,
                key="spc_direct_select_box"
            )
            selected_key = method_keys[method_labels.index(selected_label)]
            
    return nav_mode, selected_key


# =====================================================================
# 3. Interactive Playgrounds (Modules 1 - 5)
# =====================================================================

def _render_run_chart_sandbox(meta: Dict[str, Any]) -> None:
    st.subheader("📈 Module 1: Interactive Healthcare Run Chart Laboratory")
    st.caption("Evaluate 24 consecutive months of surgical site infections (SSI) before and after chlorhexidine prep.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        base_med = st.slider("Baseline Median Infections:", 8.0, 25.0, 14.0, 1.0, key="rc_med_slider")
    with c2:
        intervention_shift = st.slider("Post-Intervention Effect:", -10.0, 4.0, -5.0, 1.0, key="rc_shift_slider")
    with c3:
        baseline_n = st.number_input("Baseline Calibration Periods:", 8, 18, 12, key="rc_base_p_num")
        
    df_raw = generate_run_chart_dataset(n_points=24, baseline_median=base_med, intervention_point=13, shift_magnitude=intervention_shift, seed=42)
    df = pd.DataFrame(df_raw) if HAS_UI_STACK and not isinstance(df_raw, pd.DataFrame) else df_raw
    results = calculate_run_chart_model(df, baseline_periods=baseline_n)
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Centerline Median", f"{results['median']:.1f}", help="Median of calibration baseline")
    with k2:
        st.metric("Useful Points", f"{results['n_useful']} / {results['total_points']}", help="Points not tied with median")
    with k3:
        st.metric("Total Runs (R)", f"{results['runs_count']}", delta=f"Expected: [{results['expected_runs_min']}, {results['expected_runs_max']}]", help="Runs crossing median")
    with k4:
        status_txt = "🚨 SIGNAL" if results["has_special_cause_signal"] else "🟢 COMMON CAUSE"
        st.metric("Detection Signal", status_txt, help="Any of the 4 IHI / British rules triggered")
        
    if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
        periods = df["period"].tolist()
        vals = df["value"].tolist()
        med_line = [results["median"]] * len(periods)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[f"M{p:02d}" for p in periods], y=vals,
            mode="lines+markers", name="Observed Infections",
            line=dict(color="#0284c7", width=2.5),
            marker=dict(size=8, color="#0284c7")
        ))
        fig.add_trace(go.Scatter(
            x=[f"M{p:02d}" for p in periods], y=med_line,
            mode="lines", name=f"Baseline Median ({results['median']:.1f})",
            line=dict(color="#0f172a", width=2, dash="dash")
        ))
        
        # Highlight shift points
        shift_pts = results["rule_1_shift"]["indices"]
        if shift_pts:
            fig.add_trace(go.Scatter(
                x=[f"M{periods[i]:02d}" for i in shift_pts],
                y=[vals[i] for i in shift_pts],
                mode="markers", name="Shift Signal (>= 6 pts)",
                marker=dict(size=12, color="#10b981", symbol="circle")
            ))
            
        fig.add_vline(x=11.5, line_width=2, line_dash="dot", line_color="#10b981")
        fig.add_annotation(x=11.5, y=max(vals) + 1.0, text="Intervention (M13)", showarrow=False, font=dict(color="#047857", size=12))
        
        fig.update_layout(
            title=dict(text="<b>Healthcare Run Chart: Surgical Site Infections Over Time</b>", font=dict(size=15)),
            margin=dict(t=50, b=40, l=40, r=40),
            xaxis_title="Chronological Month",
            yaxis_title="Surgical Site Infections Count",
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)
        
    _render_run_chart_tabs(meta, df, results)


def _render_xmr_sandbox(meta: Dict[str, Any]) -> None:
    st.subheader("🎯 Module 2: Interactive Shewhart XmR / I-MR Control Chart Laboratory")
    st.caption("Monitor 30 consecutive weeks of Emergency Department triage-to-physician consultation hours.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        base_mean = st.slider("Baseline Mean Waiting Time (Hours):", 2.5, 7.0, 4.2, 0.1, key="xmr_mean_slider")
    with c2:
        effect_shift = st.slider("Flow Improvement Effect (Hours):", -2.5, 1.0, -1.1, 0.1, key="xmr_eff_slider")
    with c3:
        baseline_wks = st.number_input("Baseline Weeks Calibration:", 12, 25, 20, key="xmr_base_num")
        
    df_raw = generate_xmr_dataset(n_points=30, baseline_mean=base_mean, intervention_step=20, improvement_effect=effect_shift, seed=42)
    df = pd.DataFrame(df_raw) if HAS_UI_STACK and not isinstance(df_raw, pd.DataFrame) else df_raw
    results = calculate_xmr_model(df, baseline_periods=baseline_wks)
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Process Mean (X̄)", f"{results['x_bar']:.2f} hrs", help="Arithmetic average of calibration baseline")
    with k2:
        st.metric("Average Moving Range (MR̄)", f"{results['mr_bar']:.2f} hrs", help="Mean point-to-point step difference")
    with k3:
        st.metric("Process Limits (±3σ)", f"[{results['lcl_x']:.2f}, {results['ucl_x']:.2f}]", help="UCL = X̄ + 2.66*MR̄, LCL = X̄ - 2.66*MR̄")
    with k4:
        sc_txt = "🚨 SPECIAL CAUSE" if results["special_cause_detected"] else "🔘 COMMON CAUSE"
        st.metric("Variation Status", sc_txt, delta=f"{len(results['special_cause_indices'])} signal weeks", help="Shewhart special cause rules 1-3")
        
    if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
        weeks = df["week"].tolist()
        vals = df["wait_hours"].tolist()
        mr_vals = results["moving_ranges"]
        
        col_p1, col_p2 = st.columns([1.6, 1.0])
        with col_p1:
            fig_x = go.Figure()
            fig_x.add_trace(go.Scatter(x=[f"Wk {w:02d}" for w in weeks], y=vals, mode="lines+markers", name="Waiting Hours", line=dict(color="#0284c7", width=2.5)))
            fig_x.add_hline(y=results["x_bar"], line_width=2, line_dash="solid", line_color="#0f172a", annotation_text="X̄")
            fig_x.add_hline(y=results["ucl_x"], line_width=2, line_dash="dash", line_color="#dc2626", annotation_text="UCL (+3σ)")
            fig_x.add_hline(y=results["lcl_x"], line_width=2, line_dash="dash", line_color="#dc2626", annotation_text="LCL (-3σ)")
            
            # Mark special cause points
            sc_pts = results["special_cause_indices"]
            if sc_pts:
                fig_x.add_trace(go.Scatter(x=[f"Wk {weeks[i]:02d}" for i in sc_pts], y=[vals[i] for i in sc_pts], mode="markers", name="Special Cause Signal", marker=dict(size=12, color="#ea580c", symbol="diamond")))
                
            fig_x.update_layout(title="<b>Individuals (X) Chart: 3-Sigma Limits</b>", margin=dict(t=40, b=30, l=30, r=30), height=360)
            st.plotly_chart(fig_x, use_container_width=True)
            
        with col_p2:
            fig_mr = go.Figure()
            fig_mr.add_trace(go.Scatter(x=[f"Wk {w:02d}" for w in weeks], y=mr_vals, mode="lines+markers", name="Moving Range", line=dict(color="#64748b", width=2)))
            fig_mr.add_hline(y=results["mr_bar"], line_width=1.5, line_dash="solid", line_color="#0f172a", annotation_text="MR̄")
            fig_mr.add_hline(y=results["ucl_mr"], line_width=1.5, line_dash="dash", line_color="#dc2626", annotation_text="UCL_MR")
            fig_mr.update_layout(title="<b>Moving Range (MR) Chart</b>", margin=dict(t=40, b=30, l=30, r=30), height=360)
            st.plotly_chart(fig_mr, use_container_width=True)
            
    _render_xmr_tabs(meta, df, results)


def _render_p_chart_sandbox(meta: Dict[str, Any]) -> None:
    st.subheader("📊 Module 3: Proportions & Laney p'-Chart Laboratory")
    st.caption("Demonstrates the 'Large N Trap': How Laney p' prevents false alarms in large hospital populations.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        base_p_val = st.slider("Baseline Readmission Rate (%):", 4.0, 16.0, 8.5, 0.5, key="p_base_slider") / 100.0
    with c2:
        overdisp = st.slider("Overdispersion Inflation (Cluster Shock):", 1.0, 2.5, 1.5, 0.1, key="p_disp_slider")
    with c3:
        base_periods = st.number_input("Calibration Baseline Months:", 12, 20, 18, key="p_base_m_num")
        
    df_raw = generate_p_chart_dataset(n_periods=24, base_p=base_p_val, overdispersion_inflation=overdisp, seed=42)
    df = pd.DataFrame(df_raw) if HAS_UI_STACK and not isinstance(df_raw, pd.DataFrame) else df_raw
    results = calculate_p_chart_model(df, baseline_periods=base_periods)
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Process Rate (p̄)", f"{results['p_bar_pct']:.2f}%", help="Weighted average readmission rate")
    with k2:
        st.metric("Laney Factor (σ_z)", f"{results['sigma_z']:.2f}", delta="> 1.30 = Overdispersed" if results["is_overdispersed"] else "<= 1.30 = Normal", help="Ratio of between-subgroup variance to binomial variance")
    with k3:
        st.metric("Standard p Alarms", f"{results['std_alarms_count']} months", help="Alarms using narrow binomial limits")
    with k4:
        st.metric("Laney p' Alarms", f"{results['laney_alarms_count']} months", delta=f"-{results['false_alarm_difference']} False Alarms", help="True special causes after overdispersion adjustment")
        
    if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
        periods = df["period"].tolist()
        rates = df["readmission_rate_pct"].tolist()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[f"M{p:02d}" for p in periods], y=rates, mode="lines+markers", name="Observed Readmissions (%)", line=dict(color="#0284c7", width=2.5)))
        fig.add_hline(y=results["p_bar_pct"], line_width=2, line_dash="solid", line_color="#0f172a", annotation_text="p̄ Centerline")
        fig.add_trace(go.Scatter(x=[f"M{p:02d}" for p in periods], y=results["std_ucl"], mode="lines", name="Standard p-Chart UCL (Binomial)", line=dict(color="#ef4444", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=[f"M{p:02d}" for p in periods], y=results["laney_ucl"], mode="lines", name="Laney p' UCL (Overdispersion Adjusted)", line=dict(color="#10b981", width=2.5)))
        
        fig.update_layout(
            title="<b>Proportions: Standard p-Chart vs. Laney p'-Chart Control Limits</b>",
            margin=dict(t=50, b=40, l=40, r=40),
            xaxis_title="Observation Month",
            yaxis_title="30-Day Readmission Rate (%)",
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)
        
    _render_p_chart_tabs(meta, df, results)


def _render_u_chart_sandbox(meta: Dict[str, Any]) -> None:
    st.subheader("🔢 Module 4: Count Rates & Laney u'-Chart Laboratory")
    st.caption("Monitor inpatient falls per 1,000 occupied bed-days with variable ward occupancy.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        base_rate = st.slider("Baseline Fall Rate (per 1,000 bed-days):", 2.0, 9.0, 4.8, 0.2, key="u_base_slider")
    with c2:
        overdisp = st.slider("Poisson Overdispersion Factor:", 1.0, 2.5, 1.4, 0.1, key="u_disp_slider")
    with c3:
        base_periods = st.number_input("Baseline Periods Calibration:", 12, 20, 18, key="u_base_m_num")
        
    df_raw = generate_u_chart_dataset(n_periods=24, base_rate=base_rate, overdispersion_inflation=overdisp, seed=42)
    df = pd.DataFrame(df_raw) if HAS_UI_STACK and not isinstance(df_raw, pd.DataFrame) else df_raw
    results = calculate_u_chart_model(df, baseline_periods=base_periods)
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Process Rate (ū)", f"{results['u_bar']:.2f} / 1k", help="Mean falls per 1,000 bed-days")
    with k2:
        st.metric("Laney Sigma (σ_z)", f"{results['sigma_z']:.2f}", delta="Overdispersed" if results["is_overdispersed"] else "Poisson Expected")
    with k3:
        st.metric("Standard u Alarms", f"{results['std_alarms_count']} months")
    with k4:
        st.metric("Laney u' Alarms", f"{results['laney_alarms_count']} months")
        
    if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
        periods = df["period"].tolist()
        rates = df["falls_rate_per_1k"].tolist()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[f"M{p:02d}" for p in periods], y=rates, mode="lines+markers", name="Falls Rate", line=dict(color="#0284c7", width=2.5)))
        fig.add_hline(y=results["u_bar"], line_width=2, line_dash="solid", line_color="#0f172a", annotation_text="ū Centerline")
        fig.add_trace(go.Scatter(x=[f"M{p:02d}" for p in periods], y=results["std_ucl"], mode="lines", name="Standard u UCL", line=dict(color="#ef4444", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=[f"M{p:02d}" for p in periods], y=results["laney_ucl"], mode="lines", name="Laney u' UCL", line=dict(color="#10b981", width=2.5)))
        
        fig.update_layout(title="<b>Count Rates: u-Chart vs. Laney u'-Chart Limits</b>", margin=dict(t=50, b=40, l=40, r=40), height=380)
        st.plotly_chart(fig, use_container_width=True)
        
    _render_u_chart_tabs(meta, df, results)


def _render_mdc_sandbox(meta: Dict[str, Any]) -> None:
    st.subheader("🧭 Module 5: NHS 'Making Data Count' & Capability Analysis (Cp / Cpk)")
    st.caption("Evaluate Emergency Department 4-hour performance against the statutory NHS 95% target.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        base_compliance = st.slider("Process Compliance Mean (%):", 78.0, 96.0, 88.5, 0.5, key="mdc_mean_slider")
    with c2:
        stat_target = st.slider("NHS Statutory Target Standard (%):", 85.0, 98.0, 95.0, 1.0, key="mdc_tgt_slider")
    with c3:
        direction = st.selectbox("Metric Directionality:", ["Higher is Better (e.g. 4-hour ED standard >= 95%)", "Lower is Better (e.g. Infection rate <= 2%)"], key="mdc_dir_select")
        
    is_higher = "Higher" in direction
    df_raw = generate_mdc_dataset(n_periods=26, baseline_compliance=base_compliance, target_value=stat_target, seed=42)
    df = pd.DataFrame(df_raw) if HAS_UI_STACK and not isinstance(df_raw, pd.DataFrame) else df_raw
    results = calculate_mdc_capability_model(df, target_value=stat_target, is_higher_better=is_higher, baseline_periods=20)
    
    # MDC Badges
    st.markdown("#### 🏷️ NHS England 'Making Data Count' System Icon Badges:")
    b1, b2 = st.columns(2)
    with b1:
        st.info(f"**Variation Icon:**\n\n### {results['variation_icon']}")
    with b2:
        if results["assurance_code"] == "PASS":
            st.success(f"**Assurance Icon:**\n\n### {results['assurance_icon']}\n\n*{results['assurance_desc']}*")
        elif results["assurance_code"] == "FAIL":
            st.error(f"**Assurance Icon:**\n\n### {results['assurance_icon']}\n\n*{results['assurance_desc']}*")
        else:
            st.warning(f"**Assurance Icon:**\n\n### {results['assurance_icon']}\n\n*{results['assurance_desc']}*")
            
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Process Mean (X̄)", f"{results['x_bar']:.1f}%")
    with k2:
        st.metric("Control Limits", f"[{results['lcl']:.1f}%, {results['ucl']:.1f}%]")
    with k3:
        st.metric("Potential Capability (Cp)", f"{results['cp']:.2f}")
    with k4:
        st.metric("Actual Capability (Cpk)", f"{results['cpk']:.2f}", delta=">= 1.33 = Capable" if results["cpk"] >= 1.33 else "< 1.33 = Incapable")
        
    if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
        periods = df["period"].tolist()
        vals = df["compliance_pct"].tolist()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[f"P{p:02d}" for p in periods], y=vals, mode="lines+markers", name="4-Hour Compliance %", line=dict(color="#0284c7", width=2.5)))
        fig.add_hline(y=results["x_bar"], line_width=2, line_dash="solid", line_color="#0f172a", annotation_text="Process Mean")
        fig.add_hline(y=results["ucl"], line_width=2, line_dash="dash", line_color="#64748b", annotation_text="UCL (+3σ)")
        fig.add_hline(y=results["lcl"], line_width=2, line_dash="dash", line_color="#64748b", annotation_text="LCL (-3σ)")
        fig.add_hline(y=stat_target, line_width=2.5, line_dash="dot", line_color="#dc2626", annotation_text=f"Target ({stat_target:.0f}%)")
        
        fig.update_layout(title="<b>Making Data Count: Process Limits vs. Statutory Target</b>", margin=dict(t=50, b=40, l=40, r=40), height=380)
        st.plotly_chart(fig, use_container_width=True)
        
    _render_mdc_tabs(meta, df, results)


# =====================================================================
# 4. Standard 4-Tab Educational Guidance Containers
# =====================================================================

def _render_run_chart_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs([
        "📖 Mathematical Formulation & Causal Derivation",
        "💻 Implementation in Python",
        "📊 Excel Implementation",
        "📥 Download Practice Data (CSV)"
    ])
    with t1:
        st.markdown(meta["mathematical_formulation"])
        with st.expander("🔍 Plain English Glossary of Terms", expanded=False):
            for term, desc in meta["plain_english_glossary"].items():
                st.markdown(f"- **{term}**: {desc}")
        with st.expander("🛡️ Critical Assumptions & Failure Modes", expanded=False):
            for a in meta["assumptions"]:
                st.markdown(f"- {a}")
    with t2:
        st.markdown("#### Production Python Pipeline (Healthcare Run Chart Engine):")
        st.code("""import pandas as pd
import numpy as np

df = pd.read_csv("run_chart_ssi.csv")
median_val = df["value"][:12].median()
df["useful"] = df["value"] != median_val
n_useful = df["useful"].sum()

# Count runs
useful_series = df[df["useful"]]["value"] > median_val
runs = (useful_series != useful_series.shift()).sum()

print(f"Baseline Median: {median_val:.1f}")
print(f"Useful Points: {n_useful}, Total Runs: {runs}")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Healthcare Run Chart")
        st.markdown(r"""
        Healthcare run charts can be implemented in Microsoft Excel using standard array and logical formulas.
        """)
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Column A (`Period`):** Observation month (`1` to `24`).
            * **Column B (`Value`):** Surgical site infections count.
            * **Column C (`Baseline_Median`):** `=MEDIAN($B$2:$B$13)`.
            * **Column D (`Useful`):** `=IF(B2=$C$2, 0, 1)`.
            * **Column E (`Side`):** `=IF(B2>$C$2, 1, IF(B2<$C$2, -1, 0))`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                p_df = df.head(6).copy()
                p_df.insert(0, "Excel_Row", [f"Row {i+2}" for i in range(len(p_df))])
                st.dataframe(p_df, use_container_width=True, height=240, hide_index=True)
            df_cells = pd.DataFrame({
                "Cell / Range": ["Cell H2", "Cell H3", "Cell H4", "Cell H5"],
                "Target Metric": ["Baseline Median", "Useful Points Count", "Runs Count", "Shift Rule (>= 6 pts)"],
                "Exact Excel Formula": ["=MEDIAN(B2:B13)", "=SUM(D2:D25)", "=SUMPRODUCT((E3:E25<>E2:E24)*(E3:E25<>0))+1", "Array formula checking consecutive sides"],
                "Calculated Output Value": [f"{results['median']:.1f}", f"{results['n_useful']}", f"{results['runs_count']}", results["rule_1_shift"]["status"]]
            }) if pd is not None else None
            if df_cells is not None:
                st.dataframe(df_cells, use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Calculate Baseline Median:** In `Cell C2`, enter `=MEDIAN($B$2:$B$13)` and drag down to `C25`.
            2. **Flag Useful Points:** In `Cell D2`, enter `=IF(B2=C2, 0, 1)`.
            3. **Count Crossings:** A new run occurs whenever `Side` changes sign.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Quality Improvement Interpretation:
            * **Non-Random Signal:** The data demonstrates a **{results['rule_1_shift']['status']}**, confirming that the new chlorhexidine skin prep protocol at Month 13 produced a genuine systemic improvement rather than random noise.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: Surgical Site Infections (Run Chart)")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download Run Chart Practice Dataset (CSV)", csv_bytes, "run_chart_ssi_practice.csv", "text/csv", key="dl_rc_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)
        st.markdown(r"""
        #### 📋 Variable Dictionary:
        | Column Name | Data Type | Description |
        | :--- | :--- | :--- |
        | `period` | Integer | Chronological observation month |
        | `period_label` | String | Month code (M01 to M24) |
        | `value` | Integer | Surgical site infections count |
        | `is_post_intervention` | Binary | 1 = Post-chlorhexidine reform |
        """)


def _render_xmr_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs([
        "📖 Mathematical Formulation & Causal Derivation",
        "💻 Implementation in Python",
        "📊 Excel Implementation",
        "📥 Download Practice Data (CSV)"
    ])
    with t1:
        st.markdown(meta["mathematical_formulation"])
        with st.expander("🔍 Plain English Glossary of Terms", expanded=False):
            for term, desc in meta["plain_english_glossary"].items():
                st.markdown(f"- **{term}**: {desc}")
        with st.expander("🛡️ Critical Assumptions & Failure Modes", expanded=False):
            for a in meta["assumptions"]:
                st.markdown(f"- {a}")
    with t2:
        st.markdown("#### Production Python Pipeline (Shewhart XmR Engine):")
        st.code("""import pandas as pd
import numpy as np

df = pd.read_csv("xmr_ed_wait.csv")
k_base = 20
x_bar = df["wait_hours"][:k_base].mean()
mr = df["wait_hours"][:k_base].diff().abs()
mr_bar = mr.mean()

ucl_x = x_bar + 2.66 * mr_bar
lcl_x = max(0.0, x_bar - 2.66 * mr_bar)
print(f"X-bar: {x_bar:.2f}, UCL: {ucl_x:.2f}, LCL: {lcl_x:.2f}")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Shewhart XmR Control Chart")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Column A (`Week`):** Weekly period (`1` to `30`).
            * **Column B (`Wait_Hours`):** Continuous average waiting hours.
            * **Column C (`Moving_Range`):** `=ABS(B3-B2)` starting in row 3.
            * **Column D (`X_bar`):** `=$H$2`.
            * **Column E (`UCL_X`):** `=$H$4`.
            * **Column F (`LCL_X`):** `=$H$5`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                p_df = df.head(6).copy()
                p_df.insert(0, "Excel_Row", [f"Row {i+2}" for i in range(len(p_df))])
                st.dataframe(p_df, use_container_width=True, height=240, hide_index=True)
            df_cells = pd.DataFrame({
                "Cell / Range": ["Cell H2", "Cell H3", "Cell H4", "Cell H5", "Cell H6"],
                "Target Metric": ["Process Mean (X̄)", "Average Moving Range (MR̄)", "Upper Control Limit (UCL_X)", "Lower Control Limit (LCL_X)", "Moving Range UCL (UCL_MR)"],
                "Exact Excel Formula": ["=AVERAGE(B2:B21)", "=AVERAGE(C3:C21)", "=H2 + 2.66*H3", "=MAX(0, H2 - 2.66*H3)", "=3.267*H3"],
                "Calculated Output Value": [f"{results['x_bar']:.2f} hrs", f"{results['mr_bar']:.2f} hrs", f"{results['ucl_x']:.2f} hrs", f"{results['lcl_x']:.2f} hrs", f"{results['ucl_mr']:.2f} hrs"]
            }) if pd is not None else None
            if df_cells is not None:
                st.dataframe(df_cells, use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Compute Moving Range:** In `Cell C3`, enter `=ABS(B3-B2)` and drag down to `C31`.
            2. **Compute Baseline Constants:** In `Cell H2`, `=AVERAGE(B2:B21)`. In `Cell H3`, `=AVERAGE(C3:C21)`.
            3. **Compute 3-Sigma Limits:** UCL = `=H2+2.66*H3`, LCL = `=MAX(0, H2-2.66*H3)`.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Clinical Operations Interpretation:
            * **System Predictability:** Routine common cause variation in ED wait times produces fluctuations between **{results['lcl_x']:.2f} and {results['ucl_x']:.2f} hours**.
            * **Special Cause Action:** {len(results['special_cause_indices'])} weeks triggered special cause signals, confirming that the new acute medical assessment unit improved performance.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: ED Waiting Times (Shewhart XmR)")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download XmR Practice Dataset (CSV)", csv_bytes, "xmr_ed_waiting_hours.csv", "text/csv", key="dl_xmr_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)
        st.markdown(r"""
        #### 📋 Variable Dictionary:
        | Column Name | Data Type | Description |
        | :--- | :--- | :--- |
        | `week` | Integer | Observation week index |
        | `week_label` | String | Week code (Wk 01 to Wk 30) |
        | `wait_hours` | Float | Average triage-to-doctor time (hours) |
        | `is_post_improvement` | Binary | 1 = Post-acute assessment reform |
        """)


def _render_p_chart_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs([
        "📖 Mathematical Formulation & Causal Derivation",
        "💻 Implementation in Python",
        "📊 Excel Implementation",
        "📥 Download Practice Data (CSV)"
    ])
    with t1:
        st.markdown(meta["mathematical_formulation"])
        with st.expander("🔍 Plain English Glossary of Terms", expanded=False):
            for term, desc in meta["plain_english_glossary"].items():
                st.markdown(f"- **{term}**: {desc}")
        with st.expander("🛡️ Critical Assumptions & Failure Modes", expanded=False):
            for a in meta["assumptions"]:
                st.markdown(f"- {a}")
    with t2:
        st.markdown("#### Production Python Pipeline (Laney p'-Chart Engine):")
        st.code("""import pandas as pd
import numpy as np

df = pd.read_csv("p_chart_readmission.csv")
tot_x = df["readmissions_x"].sum()
tot_n = df["discharges_n"].sum()
p_bar = tot_x / tot_n

se_i = np.sqrt(p_bar * (1 - p_bar) / df["discharges_n"])
z_i = (df["readmissions_x"] / df["discharges_n"] - p_bar) / se_i
mr_z = np.abs(np.diff(z_i))
sigma_z = mr_z.mean() / 1.128

laney_ucl = (p_bar + 3 * sigma_z * se_i) * 100
print(f"p-bar: {p_bar*100:.2f}%, Laney sigma_z: {sigma_z:.3f}")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Laney p'-Chart")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Column A (`Period`):** Month index (`1` to `24`).
            * **Column B (`Discharges_n`):** Subgroup denominator.
            * **Column C (`Readmissions_x`):** Event count.
            * **Column D (`Rate_p`):** `=C2/B2*100`.
            * **Column E (`Standard_SE`):** `=SQRT($H$2*(1-$H$2)/B2)`.
            * **Column F (`Z_Score`):** `=(C2/B2 - $H$2)/E2`.
            * **Column G (`Laney_UCL`):** `=( $H$2 + 3 * $H$3 * E2 ) * 100`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                p_df = df.head(6).copy()
                p_df.insert(0, "Excel_Row", [f"Row {i+2}" for i in range(len(p_df))])
                st.dataframe(p_df, use_container_width=True, height=240, hide_index=True)
            df_cells = pd.DataFrame({
                "Cell / Range": ["Cell H2", "Cell H3", "Cell H4"],
                "Target Metric": ["Weighted Proportion (p̄)", "Laney Overdispersion (σ_z)", "Overdispersion Diagnostic"],
                "Exact Excel Formula": ["=SUM(C2:C19)/SUM(B2:B19)", "=AVERAGE(ABS(F3:F19-F2:F18))/1.128", '=IF(H3>1.30, "OVERDISPERSED", "NORMAL")'],
                "Calculated Output Value": [f"{results['p_bar_pct']:.2f}%", f"{results['sigma_z']:.3f}", results["dispersion_status"]]
            }) if pd is not None else None
            if df_cells is not None:
                st.dataframe(df_cells, use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Compute Weighted Centerline:** In `Cell H2`, `=SUM(C2:C19)/SUM(B2:B19)`.
            2. **Compute Z-Scores:** In `Cell F2`, `=((C2/B2) - $H$2) / SQRT($H$2*(1-$H$2)/B2)`.
            3. **Compute Laney Factor:** In `Cell H3`, calculate moving range of Z-scores divided by 1.128.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Quality Governance & False Alarm Reduction:
            * **Large N Trap Resolved:** Standard binomial p-charts sounded **{results['std_alarms_count']} alarms** because hospital volume was large ($n_i > 1,500$). The Laney p'-chart expanded limits appropriately ($\sigma_z = {results['sigma_z']:.2f}$), eliminating **{results['false_alarm_difference']} false alarms** and isolating true special causes.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: 30-Day Emergency Readmissions (p-Chart)")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download p-Chart Practice Dataset (CSV)", csv_bytes, "p_chart_readmissions.csv", "text/csv", key="dl_p_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)


def _render_u_chart_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs([
        "📖 Mathematical Formulation & Causal Derivation",
        "💻 Implementation in Python",
        "📊 Excel Implementation",
        "📥 Download Practice Data (CSV)"
    ])
    with t1:
        st.markdown(meta["mathematical_formulation"])
        with st.expander("🔍 Plain English Glossary of Terms", expanded=False):
            for term, desc in meta["plain_english_glossary"].items():
                st.markdown(f"- **{term}**: {desc}")
        with st.expander("🛡️ Critical Assumptions & Failure Modes", expanded=False):
            for a in meta["assumptions"]:
                st.markdown(f"- {a}")
    with t2:
        st.markdown("#### Production Python Pipeline (Laney u'-Chart Engine):")
        st.code("""import pandas as pd
import numpy as np

df = pd.read_csv("u_chart_falls.csv")
tot_c = df["falls_count"].sum()
tot_n = df["exposure_units_1k"].sum()
u_bar = tot_c / tot_n

se_i = np.sqrt(u_bar / df["exposure_units_1k"])
z_i = (df["falls_rate_per_1k"] - u_bar) / se_i
sigma_z = np.abs(np.diff(z_i)).mean() / 1.128

laney_ucl = u_bar + 3 * sigma_z * se_i
print(f"u-bar: {u_bar:.2f}, Laney sigma_z: {sigma_z:.3f}")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Laney u'-Chart")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Column A (`Period`):** Month index (`1` to `24`).
            * **Column B (`Bed_Days`):** Occupied bed-days count.
            * **Column C (`Exposure_Units`):** `=B2/1000`.
            * **Column D (`Falls_Count`):** Recorded fall incidents.
            * **Column E (`Rate_u`):** `=D2/C2`.
            * **Column F (`Laney_UCL`):** `=$H$2 + 3 * $H$3 * SQRT($H$2/C2)`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                p_df = df.head(6).copy()
                p_df.insert(0, "Excel_Row", [f"Row {i+2}" for i in range(len(p_df))])
                st.dataframe(p_df, use_container_width=True, height=240, hide_index=True)
            df_cells = pd.DataFrame({
                "Cell / Range": ["Cell H2", "Cell H3", "Cell H4"],
                "Target Metric": ["Weighted Mean Rate (ū)", "Laney Dispersion Factor (σ_z)", "Diagnostic Status"],
                "Exact Excel Formula": ["=SUM(D2:D19)/SUM(C2:C19)", "=AVERAGE(ABS(Z_diff))/1.128", '=IF(H3>1.30, "OVERDISPERSED", "NORMAL")'],
                "Calculated Output Value": [f"{results['u_bar']:.2f} / 1k", f"{results['sigma_z']:.3f}", results["dispersion_status"]]
            }) if pd is not None else None
            if df_cells is not None:
                st.dataframe(df_cells, use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Compute Exposure Units:** In `Cell C2`, enter `=B2/1000`.
            2. **Compute Rate:** In `Cell E2`, enter `=D2/C2`.
            3. **Compute Weighted ū:** In `Cell H2`, enter `=SUM(D2:D19)/SUM(C2:C19)`.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Inpatient Safety Interpretation:
            * **Rate per 1,000 Bed-Days:** Baseline falls rate is **{results['u_bar']:.2f} per 1,000 bed-days**. Laney u' limits adapt dynamically to fluctuating monthly bed occupancy.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: Inpatient Falls (u-Chart)")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download u-Chart Practice Dataset (CSV)", csv_bytes, "u_chart_inpatient_falls.csv", "text/csv", key="dl_u_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)


def _render_mdc_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs([
        "📖 Mathematical Formulation & Causal Derivation",
        "💻 Implementation in Python",
        "📊 Excel Implementation",
        "📥 Download Practice Data (CSV)"
    ])
    with t1:
        st.markdown(meta["mathematical_formulation"])
        with st.expander("🔍 Plain English Glossary of Terms", expanded=False):
            for term, desc in meta["plain_english_glossary"].items():
                st.markdown(f"- **{term}**: {desc}")
        with st.expander("🛡️ Critical Assumptions & Failure Modes", expanded=False):
            for a in meta["assumptions"]:
                st.markdown(f"- {a}")
    with t2:
        st.markdown("#### Production Python Pipeline (NHS Making Data Count & Capability Engine):")
        st.code("""import pandas as pd
import numpy as np

df = pd.read_csv("mdc_ed_4hour.csv")
target = 95.0
x_bar = df["compliance_pct"][:20].mean()
mr_bar = df["compliance_pct"][:20].diff().abs().mean()
sigma_hat = mr_bar / 1.128

cpk = (x_bar - target) / (3 * sigma_hat)
ucl = x_bar + 2.66 * mr_bar
lcl = x_bar - 2.66 * mr_bar

print(f"Mean: {x_bar:.1f}%, Cpk: {cpk:.2f}, Limits: [{lcl:.1f}%, {ucl:.1f}%]")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: NHS Making Data Count")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Column A (`Period`):** Bi-weekly period (`1` to `26`).
            * **Column B (`Compliance_Pct`):** 4-hour performance (%).
            * **Column C (`Target`):** `=95.0`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                p_df = df.head(6).copy()
                p_df.insert(0, "Excel_Row", [f"Row {i+2}" for i in range(len(p_df))])
                st.dataframe(p_df, use_container_width=True, height=240, hide_index=True)
            df_cells = pd.DataFrame({
                "Cell / Range": ["Cell H2", "Cell H3", "Cell H4", "Cell H5", "Cell H6"],
                "Target Metric": ["Process Mean", "Lower Control Limit", "Upper Control Limit", "Capability Cpk", "Assurance Category"],
                "Exact Excel Formula": ["=AVERAGE(B2:B21)", "=H2 - 2.66*AVERAGE(ABS(B3:B21-B2:B20))", "=H2 + 2.66*AVERAGE(ABS(B3:B21-B2:B20))", "=(H2 - 95.0)/(3*(AVERAGE(ABS(B3:B21-B2:B20))/1.128))", '=IF(H3>=95, "PASS", IF(H4<95, "FAIL", "FLAPPING"))'],
                "Calculated Output Value": [f"{results['x_bar']:.1f}%", f"{results['lcl']:.1f}%", f"{results['ucl']:.1f}%", f"{results['cpk']:.2f}", results["assurance_code"]]
            }) if pd is not None else None
            if df_cells is not None:
                st.dataframe(df_cells, use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Calculate Process Limits:** In `Cell H2`, `=AVERAGE(B2:B21)`.
            2. **Calculate Cpk:** Min distance from mean to target in units of $3\hat{\sigma}$.
            3. **Classify Assurance:** Compare target to `LCL` and `UCL`.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 NHS Executive Board Interpretation:
            * **Assurance Status:** **{results['assurance_icon']}**
            * **Process Incapable ($C_{{pk}} = {results['cpk']:.2f} < 1.33$):** Demanding that ward staff work harder will not achieve 95% compliance because the process capability upper limit is only **{results['ucl']:.1f}%**. Achieving 95% requires structural emergency department bed flow redesign!
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: NHS 4-Hour Compliance (Making Data Count)")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download MDC Practice Dataset (CSV)", csv_bytes, "mdc_ed_4hour_compliance.csv", "text/csv", key="dl_mdc_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)


# =====================================================================
# 5. Top-Level Module Controller
# =====================================================================

def render_spc_method_module(method_key: str) -> None:
    """Dispatches execution to the corresponding SPC sandbox."""
    meta = SPC_METHODS.get(method_key, SPC_METHODS["healthcare_run_chart"])
    
    if method_key == "healthcare_run_chart":
        _render_run_chart_sandbox(meta)
    elif method_key == "xmr_individual_chart":
        _render_xmr_sandbox(meta)
    elif method_key == "p_chart_laney":
        _render_p_chart_sandbox(meta)
    elif method_key == "u_chart_laney":
        _render_u_chart_sandbox(meta)
    elif method_key == "making_data_count_capability":
        _render_mdc_sandbox(meta)
    else:
        _render_run_chart_sandbox(meta)

"""Health Economics & Health Technology Assessment (HTA / NICE) UI Components.

Provides:
1. Zero-knowledge plain English introductory guide on Opportunity Cost, QALYs, and NICE thresholds.
2. Mode selector (Guided HTA Scenario Wizard vs. Direct Method Explorer).
3. Dynamic interactive simulation playgrounds with Plotly CE planes, Markov cohort traces, CEAC curves, and BIA budget bars.
4. Standard 4-tab container (Mathematical Formulation, Python Pipeline, Excel Guide with 3 expanders, Practice CSV Data).
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

from src.health_economics.hta_schema import HTA_METHODS, HTA_SCENARIOS
from src.health_economics.hta_engine import (
    calculate_icer_model,
    generate_icer_dataset,
    calculate_nmb_model,
    generate_nmb_dataset,
    calculate_markov_model,
    generate_markov_dataset,
    calculate_psa_model,
    generate_psa_dataset,
    calculate_bia_model,
    generate_bia_dataset
)


# =====================================================================
# 1. Plain-English Zero-Knowledge Pedagogical Guide
# =====================================================================

def render_hta_plain_english_guide() -> None:
    """Renders intuition-first conceptual guide to Health Economics, QALYs,
    the Cost-Effectiveness Plane, and NICE decision rules.
    """
    with st.expander("📖 Plain English Guide: Opportunity Costs, QALYs & The NICE £20,000–£30,000 Threshold", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 💡 Why Health Economics Matters in Public Healthcare:
            Unlike private consumer markets, publicly funded healthcare systems (such as the UK NHS) operate under a **fixed, taxpayer-funded budget**.
            
            Every pound spent on an expensive new cancer drug or surgical robot is money that cannot be spent on hip replacements, 
            community mental health teams, or neonatal nursing. Therefore, adopting a new technology creates an inevitable **opportunity cost**: 
            the health that would have been generated elsewhere in the NHS has been displaced.
            
            Health Technology Assessment (HTA) answers two fundamental ethical questions:
            1. **Is the new technology worth the money over a patient's lifetime?** (*Cost-Effectiveness Analysis / ICER*)
            2. **Can the healthcare budget afford the invoice this financial year?** (*Budget Impact Analysis*)
            """)
        with col_g2:
            st.markdown(r"""
            #### 🎯 The Core Building Blocks of NICE Appraisals:
            * **What is a QALY (Quality-Adjusted Life Year)?**  
              A single universal yardstick combining length of life and quality of life.  
              $$\text{QALY} = \text{Years of Life Lived} \times \text{Health Utility Weight } (0.0 \text{ to } 1.0)$$  
              *1 year lived in perfect health (utility 1.0) = 1.0 QALY.*  
              *2 years lived with moderate arthritis (utility 0.5) = 1.0 QALY.*
            * **The NICE Willingness-to-Pay Threshold ($\lambda$):**  
              NICE typically considers a technology cost-effective if the incremental cost per QALY gained is **between £20,000 and £30,000**.  
              Empirical health economics research estimates that the NHS displaces 1 QALY for every ~£15,000–£20,000 of new spending.
            * **Severity Modifier:**  
              For severe, life-threatening conditions with substantial absolute or proportional QALY shortfall, NICE applies an equity weighting 
              allowing thresholds up to **£50,000 per QALY**.
            """)


# =====================================================================
# 2. Navigation Mode Selector & Execution Controller
# =====================================================================

def render_hta_mode_selector() -> Tuple[str, str]:
    """Renders the mode controller allowing toggling between:
    - Guided HTA Scenario Wizard (NICE Appraisal Workflows)
    - Direct Method Explorer (Select from 5 core architectures)
    """
    if "hta_nav_mode_radio" not in st.session_state:
        st.session_state["hta_nav_mode_radio"] = "🧭 Guided HTA Scenario Wizard"

    col_m1, col_m2 = st.columns([1.8, 2.2])
    with col_m1:
        nav_mode = st.radio(
            "Select Exploration Pathway:",
            options=["🧭 Guided HTA Scenario Wizard", "🔍 Direct Method Explorer"],
            key="hta_nav_mode_radio",
            horizontal=True
        )

    selected_method_key = "icer_plane"

    if nav_mode == "🧭 Guided HTA Scenario Wizard":
        with col_m2:
            scenario_names = {k: f"{v['title']} ({v['params'].get('wtp_threshold', '') and '£' + str(int(v['params']['wtp_threshold'])) + '/QALY'})" for k, v in HTA_SCENARIOS.items()}
            sel_scen_key = st.selectbox(
                "Choose a real-world NHS appraisal scenario:",
                options=list(scenario_names.keys()),
                format_func=lambda k: scenario_names[k],
                key="hta_scenario_select"
            )
            selected_method_key = HTA_SCENARIOS[sel_scen_key]["method_key"]
            st.caption(f"💡 *Scenario:* {HTA_SCENARIOS[sel_scen_key]['description']}")
    else:
        with col_m2:
            method_options = {k: f"{v['icon']} {v['name']}" for k, v in HTA_METHODS.items()}
            default_idx = 0
            if "hta_direct_method_key" in st.session_state and st.session_state["hta_direct_method_key"] in method_options:
                default_idx = list(method_options.keys()).index(st.session_state["hta_direct_method_key"])

            selected_method_key = st.selectbox(
                "Choose an HTA / Health Economics method:",
                options=list(method_options.keys()),
                format_func=lambda k: method_options[k],
                index=default_idx,
                key="hta_direct_method_select"
            )
            st.session_state["hta_direct_method_key"] = selected_method_key

    return nav_mode, selected_method_key


# =====================================================================
# 3. Interactive Method Simulation Playgrounds
# =====================================================================

def _render_icer_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### ⚖️ Interactive Playground: Cost-Effectiveness Plane & ICER Thresholds")
    st.markdown(r"""
    Explore how incremental costs ($\Delta C$) and incremental health benefits ($\Delta \text{QALY}$) determine 
    quadrant placement on the Cost-Effectiveness Plane, the resulting ICER, and the maximum recommended price ceiling.
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cost_t = st.number_input("Intervention Total Cost (£):", min_value=0.0, max_value=200000.0, value=28400.0, step=1000.0, key="icer_cost_t")
        cost_c = st.number_input("Comparator Total Cost (£):", min_value=0.0, max_value=200000.0, value=14200.0, step=1000.0, key="icer_cost_c")
    with c2:
        qaly_t = st.number_input("Intervention QALYs:", min_value=0.0, max_value=30.0, value=6.85, step=0.10, key="icer_qaly_t")
        qaly_c = st.number_input("Comparator QALYs:", min_value=0.0, max_value=30.0, value=6.15, step=0.10, key="icer_qaly_c")
    with c3:
        wtp_threshold = st.select_slider(
            "WTP Threshold λ (£/QALY):",
            options=[15000.0, 20000.0, 25000.0, 30000.0, 40000.0, 50000.0],
            value=25000.0,
            key="icer_wtp_slider"
        )
    with c4:
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.info("💡 **NICE Standard Range:** £20k–£30k. Severe conditions with high QALY shortfall qualify for £50k/QALY.")

    results = calculate_icer_model(cost_t, cost_c, qaly_t, qaly_c, wtp_threshold)

    # Metric Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Incremental Cost (ΔC)", f"£{results['delta_cost']:,.0f}")
    with m2:
        st.metric("Incremental QALYs (ΔE)", f"{results['delta_qaly']:+.2f}")
    with m3:
        st.metric("Calculated ICER", results["icer_str"])
    with m4:
        st.metric("Net Monetary Benefit", f"£{results['nmb']:,.0f}")
    with m5:
        st.metric("CE Plane Quadrant", results["quadrant_code"])

    # Decision Banner
    if results["is_cost_effective"]:
        st.success(f"### {results['verdict']}")
    else:
        st.warning(f"### {results['verdict']}")
        st.caption(f"Price reduction required: **£{results['price_gap']:,.0f}** to achieve cost-effectiveness at £{wtp_threshold:,.0f}/QALY (Max justified cost: £{results['max_justified_cost']:,.0f}).")

    # Plotly Visual: Cost-Effectiveness Plane
    if HAS_UI_STACK and go is not None:
        fig = go.Figure()
        # Quadrant background zones
        max_x = max(2.0, abs(results["delta_qaly"]) * 1.8)
        max_y = max(30000.0, abs(results["delta_cost"]) * 1.8)

        # WTP threshold line: y = lambda * x
        x_line = [-max_x, max_x]
        y_line = [-max_x * wtp_threshold, max_x * wtp_threshold]
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color="#16a34a", dash="dash", width=2),
            name=f"WTP Threshold λ (£{wtp_threshold:,.0f}/QALY)"
        ))

        # Origin axes
        fig.add_hline(y=0, line=dict(color="#94a3b8", width=1.5))
        fig.add_vline(x=0, line=dict(color="#94a3b8", width=1.5))

        # Plotted point (delta_qaly, delta_cost)
        pt_color = "#16a34a" if results["is_cost_effective"] else "#dc2626"
        fig.add_trace(go.Scatter(
            x=[results["delta_qaly"]],
            y=[results["delta_cost"]],
            mode="markers+text",
            marker=dict(size=14, color=pt_color, line=dict(color="#0f172a", width=2)),
            text=[f"  ΔE: {results['delta_qaly']:+.2f}, ΔC: £{results['delta_cost']:,.0f}"],
            textposition="top right",
            name="Technology vs Comparator"
        ))

        # Quadrant annotations
        fig.add_annotation(x=max_x*0.7, y=max_y*0.8, text="North-East: Trade-off (Cost-effective if below dashed line)", showarrow=False, font=dict(color="#475569", size=11))
        fig.add_annotation(x=max_x*0.7, y=-max_y*0.8, text="South-East: Strictly Dominant (More effective & cheaper)", showarrow=False, font=dict(color="#166534", size=11))
        fig.add_annotation(x=-max_x*0.7, y=max_y*0.8, text="North-West: Strictly Dominated (Less effective & more expensive)", showarrow=False, font=dict(color="#991b1b", size=11))
        fig.add_annotation(x=-max_x*0.7, y=-max_y*0.8, text="South-West: Trade-off (Cost-effective if above dashed line)", showarrow=False, font=dict(color="#854d0e", size=11))

        fig.update_layout(
            title="Cost-Effectiveness Plane (4 Quadrants & Decision Threshold)",
            xaxis_title="Incremental Health Outcome: ΔQALY",
            yaxis_title="Incremental Lifetime Cost: ΔCost (£)",
            xaxis=dict(range=[-max_x, max_x], zeroline=False),
            yaxis=dict(range=[-max_y, max_y], zeroline=False),
            template="plotly_white",
            height=460,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    df_patients = pd.DataFrame(generate_icer_dataset(cost_t, cost_c, qaly_t, qaly_c)) if HAS_UI_STACK and pd is not None else []
    _render_icer_tabs(meta, df_patients, results)


def _render_nmb_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 💷 Interactive Playground: Net Monetary Benefit & Multi-Arm Selection")
    st.markdown(r"""
    Evaluate how Net Monetary Benefit $\text{NMB}(\lambda) = \lambda \cdot \Delta E - \Delta C$ linearly ranks competing 
    clinical interventions without ratio instability, identifying the strategy that maximizes net population health gain.
    """)

    c1, c2 = st.columns([1.5, 2.5])
    with c1:
        wtp_threshold = st.select_slider(
            "WTP Threshold λ (£/QALY):",
            options=[15000.0, 20000.0, 25000.0, 30000.0, 40000.0, 50000.0],
            value=30000.0,
            key="nmb_wtp_slider"
        )
        t1_cost = st.number_input("Standard Care Cost (£):", value=8500.0, step=500.0, key="nmb_c1")
        t1_qaly = st.number_input("Standard Care QALYs:", value=4.20, step=0.05, key="nmb_q1")
        t2_cost = st.number_input("ARNI Therapy Cost (£):", value=15800.0, step=500.0, key="nmb_c2")
        t2_qaly = st.number_input("ARNI Therapy QALYs:", value=4.62, step=0.05, key="nmb_q2")
        t3_cost = st.number_input("SGLT2i Combo Cost (£):", value=19400.0, step=500.0, key="nmb_c3")
        t3_qaly = st.number_input("SGLT2i Combo QALYs:", value=4.88, step=0.05, key="nmb_q3")

    treatments = [
        {"name": "Standard Care", "cost": t1_cost, "qalys": t1_qaly},
        {"name": "ARNI Therapy", "cost": t2_cost, "qalys": t2_qaly},
        {"name": "SGLT2i Combo", "cost": t3_cost, "qalys": t3_qaly}
    ]

    results = calculate_nmb_model(treatments, wtp_threshold)

    with c2:
        st.success(f"### 🏆 Optimal Clinical Strategy: **{results['optimal_treatment']}** (Max NMB: £{results['max_nmb']:,.0f})")
        if HAS_UI_STACK and pd is not None and go is not None:
            df_nmb = pd.DataFrame(results["ranked_by_nmb"])
            fig = go.Figure()
            colors = ["#16a34a" if t["name"] == results["optimal_treatment"] else "#64748b" for t in results["ranked_by_nmb"]]
            fig.add_trace(go.Bar(
                x=[t["name"] for t in results["ranked_by_nmb"]],
                y=[t["nmb"] for t in results["ranked_by_nmb"]],
                marker_color=colors,
                text=[f"£{t['nmb']:,.0f}" for t in results["ranked_by_nmb"]],
                textposition="auto"
            ))
            fig.update_layout(
                title=f"Net Monetary Benefit by Strategy at λ = £{wtp_threshold:,.0f}/QALY",
                xaxis_title="Treatment Strategy",
                yaxis_title="Net Monetary Benefit (£)",
                template="plotly_white",
                height=380,
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

    df_dataset = pd.DataFrame(generate_nmb_dataset(treatments, wtp_threshold)) if HAS_UI_STACK and pd is not None else []
    _render_nmb_tabs(meta, df_dataset, results)


def _render_markov_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 🔄 Interactive Playground: 10-Year Markov Decision Analytic Cohort Model")
    st.markdown(r"""
    Simulate a multi-state oncology Markov model tracking patient cohort transitions between 
    **Progression-Free Survival (PFS)**, **Progressed Disease (PD)**, and **Death** across 10 annual cycles 
    incorporating half-cycle correction and 3.5% NICE annual discounting.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        cycles = st.slider("Time Horizon (Annual Cycles):", min_value=5, max_value=20, value=10, step=1, key="markov_cycles")
        discount_rate = st.slider("NICE Annual Discount Rate (%):", min_value=0.0, max_value=7.0, value=3.5, step=0.5, key="markov_disc_pct") / 100.0
    with c2:
        immuno_pfs_stay = st.slider("Immunotherapy Annual PFS Retention (%):", min_value=50, max_value=90, value=72, step=2, key="markov_imm_pfs") / 100.0
        chemo_pfs_stay = st.slider("Chemotherapy Annual PFS Retention (%):", min_value=30, max_value=75, value=55, step=2, key="markov_chemo_pfs") / 100.0
    with c3:
        immuno_cost_pfs = st.number_input("Immunotherapy Annual PFS Cost (£):", value=18500.0, step=1000.0, key="markov_imm_cost")
        chemo_cost_pfs = st.number_input("Chemotherapy Annual PFS Cost (£):", value=4500.0, step=500.0, key="markov_chemo_cost")

    chemo_trans = [
        [chemo_pfs_stay, (1.0 - chemo_pfs_stay) * 0.78, (1.0 - chemo_pfs_stay) * 0.22],
        [0.00, 0.60, 0.40],
        [0.00, 0.00, 1.00]
    ]
    immuno_trans = [
        [immuno_pfs_stay, (1.0 - immuno_pfs_stay) * 0.78, (1.0 - immuno_pfs_stay) * 0.22],
        [0.00, 0.65, 0.35],
        [0.00, 0.00, 1.00]
    ]

    results = calculate_markov_model(
        cycles=cycles,
        discount_rate=discount_rate,
        chemo_transitions=chemo_trans,
        immuno_transitions=immuno_trans,
        chemo_costs=[chemo_cost_pfs, 12000.0, 0.0],
        immuno_costs=[immuno_cost_pfs, 12000.0, 0.0]
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Discounted ΔCost", f"£{results['delta_disc_cost']:,.0f}")
    with m2:
        st.metric("Discounted ΔQALYs", f"{results['delta_disc_qaly']:+.2f}")
    with m3:
        st.metric("Markov Model ICER", f"£{results['icer']:,.0f} / QALY")
    with m4:
        icer_badge = "🟢 Cost-Effective (<£30k)" if results["is_cost_effective_30k"] else "🟡 Over £30k/QALY"
        st.metric("NICE Appraisal Verdict", icer_badge)

    if HAS_UI_STACK and go is not None:
        fig = go.Figure()
        yrs = [r["year"] for r in results["immuno_records"]]
        fig.add_trace(go.Scatter(x=yrs, y=[r["hcc_pfs"]*100 for r in results["immuno_records"]], mode="lines+markers", name="Immunotherapy PFS %", line=dict(color="#16a34a", width=2.5)))
        fig.add_trace(go.Scatter(x=yrs, y=[r["hcc_pfs"]*100 for r in results["chemo_records"]], mode="lines+markers", name="Chemotherapy PFS %", line=dict(color="#64748b", dash="dash", width=2)))
        fig.add_trace(go.Scatter(x=yrs, y=[r["hcc_dead"]*100 for r in results["immuno_records"]], mode="lines", name="Immunotherapy Dead %", line=dict(color="#dc2626", width=2)))
        fig.add_trace(go.Scatter(x=yrs, y=[r["hcc_dead"]*100 for r in results["chemo_records"]], mode="lines", name="Chemotherapy Dead %", line=dict(color="#ef4444", dash="dash", width=1.5)))

        fig.update_layout(
            title=f"Markov Cohort Survival & Progression Trace ({cycles}-Year Horizon)",
            xaxis_title="Cycle (Years)",
            yaxis_title="Cohort in Health State (%)",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    df_markov = pd.DataFrame(generate_markov_dataset(results)) if HAS_UI_STACK and pd is not None else []
    _render_markov_tabs(meta, df_markov, results)


def _render_psa_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 🎲 Interactive Playground: Probabilistic Sensitivity Analysis (PSA) & CEAC")
    st.markdown(r"""
    Run Monte Carlo simulations sampling parameter distributions (Gamma for skewed costs, Normal/Beta for utilities) 
    to construct empirical Cost-Effectiveness Acceptability Curves (CEAC) quantifying reimbursement decision risk.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        iterations = st.select_slider("Monte Carlo Iterations (B):", options=[200, 500, 1000, 2000], value=500, key="psa_iter_slider")
        wtp_threshold = st.select_slider("Target Decision Threshold λ (£/QALY):", options=[15000.0, 20000.0, 30000.0, 50000.0], value=30000.0, key="psa_wtp_slider")
    with c2:
        sd_cost_pct = st.slider("Cost Uncertainty (CV %):", min_value=5, max_value=30, value=15, step=1, key="psa_cv_cost") / 100.0
    with c3:
        sd_qaly_val = st.slider("QALY Uncertainty (SD):", min_value=0.10, max_value=0.80, value=0.45, step=0.05, key="psa_sd_qaly")

    results = calculate_psa_model(
        iterations=iterations,
        sd_cost_t=32000.0 * sd_cost_pct,
        sd_cost_c=24000.0 * sd_cost_pct,
        sd_qaly_t=sd_qaly_val,
        sd_qaly_c=sd_qaly_val,
        wtp_threshold=wtp_threshold,
        seed=42
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Mean Incremental Cost", f"£{results['mean_delta_cost']:,.0f}")
        st.caption(f"95% CI: £{results['ci_delta_cost'][0]:,.0f} to £{results['ci_delta_cost'][1]:,.0f}")
    with m2:
        st.metric("Mean Incremental QALYs", f"{results['mean_delta_qaly']:+.2f}")
        st.caption(f"95% CI: {results['ci_delta_qaly'][0]:+.2f} to {results['ci_delta_qaly'][1]:+.2f}")
    with m3:
        st.metric("Probability Cost-Effective", f"{results['prob_ce_pct']:.1f}%")
        st.caption(f"Evaluated at λ = £{wtp_threshold:,.0f}/QALY")
    with m4:
        st.metric("Dominant (SE) Iterations", f"{results['quadrant_breakdown']['SE_dominant_pct']:.1f}%")
        st.caption(f"Dominated (NW): {results['quadrant_breakdown']['NW_dominated_pct']:.1f}%")

    if HAS_UI_STACK and go is not None:
        fig_ceac = go.Figure()
        thresh_x = [p["threshold"] for p in results["ceac_curve"]]
        prob_y = [p["prob_ce_pct"] for p in results["ceac_curve"]]

        fig_ceac.add_trace(go.Scatter(
            x=thresh_x, y=prob_y,
            mode="lines+markers",
            line=dict(color="#2563eb", width=2.5),
            name="Cost-Effectiveness Acceptability Curve (CEAC)"
        ))
        fig_ceac.add_vline(x=20000.0, line=dict(color="#f59e0b", dash="dot", width=1.5), annotation_text="£20k Threshold")
        fig_ceac.add_vline(x=30000.0, line=dict(color="#16a34a", dash="dot", width=1.5), annotation_text="£30k Threshold")

        fig_ceac.update_layout(
            title=f"Cost-Effectiveness Acceptability Curve (CEAC across {iterations} Iterations)",
            xaxis_title="Willingness-to-Pay Threshold λ (£ / QALY)",
            yaxis_title="Probability Technology is Cost-Effective (%)",
            yaxis=dict(range=[0, 105]),
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_ceac, use_container_width=True)

    df_psa = pd.DataFrame(results["sample_records"]) if HAS_UI_STACK and pd is not None else []
    _render_psa_tabs(meta, df_psa, results)


def _render_bia_sandbox(meta: Dict[str, Any]) -> None:
    st.markdown("### 📊 Interactive Playground: 5-Year NHS Budget Impact Analysis (BIA)")
    st.markdown(r"""
    Evaluate the total undiscounted cash expenditure impact on an NHS Integrated Care Board (ICB) or national commissioning budget 
    over a 5-year planning horizon, incorporating disease prevalence, dynamic market uptake, and hospitalization offset savings.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        eligible_pop = st.number_input("Eligible Patient Population:", min_value=1000, max_value=5000000, value=250000, step=10000, key="bia_pop")
        cost_new = st.number_input("New Drug Annual Cost per Patient (£):", value=1650.0, step=50.0, key="bia_c_new")
    with c2:
        cost_current = st.number_input("Current Therapy Annual Cost (£):", value=380.0, step=20.0, key="bia_c_curr")
        hospital_offset = st.number_input("Annual Hospitalization Offset / Patient (£):", value=620.0, step=50.0, key="bia_c_off")
    with c3:
        y1_up = st.slider("Year 1 Uptake (%):", min_value=1, max_value=20, value=5, step=1, key="bia_u1") / 100.0
        y5_up = st.slider("Year 5 Uptake (%):", min_value=20, max_value=80, value=48, step=2, key="bia_u5") / 100.0

    uptakes = [y1_up, y1_up + (y5_up - y1_up)*0.20, y1_up + (y5_up - y1_up)*0.45, y1_up + (y5_up - y1_up)*0.75, y5_up]
    results = calculate_bia_model(
        eligible_population=eligible_pop,
        uptake_rates=uptakes,
        annual_cost_new=cost_new,
        annual_cost_current=cost_current,
        annual_hospital_offset_per_treated=hospital_offset
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Year 1 Net Budget Impact", f"£{results['yearly_results'][0]['net_budget_impact']:,.0f}")
    with m2:
        st.metric("Year 5 Net Budget Impact", f"£{results['yearly_results'][4]['net_budget_impact']:,.0f}")
    with m3:
        st.metric("5-Year Cumulative Net Impact", f"£{results['cumulative_5yr_net_budget_impact']:,.0f}")
    with m4:
        st.metric("5-Year Hospital Bed Offsets", f"£{results['cumulative_5yr_hospital_offsets']:,.0f}")

    if HAS_UI_STACK and go is not None:
        fig_bia = go.Figure()
        yrs = [f"Year {y['year']}" for y in results["yearly_results"]]
        fig_bia.add_trace(go.Bar(name="Status Quo Budget", x=yrs, y=[y["budget_status_quo"] for y in results["yearly_results"]], marker_color="#94a3b8"))
        fig_bia.add_trace(go.Bar(name="Revised Budget with New Drug", x=yrs, y=[y["budget_with_new"] for y in results["yearly_results"]], marker_color="#2563eb"))
        fig_bia.add_trace(go.Scatter(name="Net Budget Impact", x=yrs, y=[y["net_budget_impact"] for y in results["yearly_results"]], mode="lines+markers", yaxis="y2", line=dict(color="#dc2626", width=3)))

        fig_bia.update_layout(
            title="5-Year NHS Payer Budget Comparison & Net Impact",
            xaxis_title="Financial Year",
            yaxis_title="Total Budget Expenditure (£)",
            yaxis2=dict(title="Net Budget Impact (£)", overlaying="y", side="right", showgrid=False),
            barmode="group",
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_bia, use_container_width=True)

    df_bia = pd.DataFrame(generate_bia_dataset(eligible_pop)) if HAS_UI_STACK and pd is not None else []
    _render_bia_tabs(meta, df_bia, results)


# =====================================================================
# 4. Standard 4-Tab Pedagogical Container
# =====================================================================

def _render_icer_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
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
        st.markdown("#### Production Python Pipeline: ICER, Quadrant Classification & NMB Engine")
        st.code("""def compute_icer(cost_t, cost_c, qaly_t, qaly_c, threshold=25000.0):
    delta_c = cost_t - cost_c
    delta_e = qaly_t - qaly_c
    icer = delta_c / delta_e if delta_e != 0 else None
    nmb = (threshold * delta_e) - delta_c
    
    if delta_e > 0 and delta_c < 0:
        verdict = "Dominant"
    elif delta_e < 0 and delta_c > 0:
        verdict = "Dominated"
    elif icer is not None and icer <= threshold:
        verdict = "Cost-Effective"
    else:
        verdict = "Not Cost-Effective"
        
    return {"delta_c": delta_c, "delta_e": delta_e, "icer": icer, "nmb": nmb, "verdict": verdict}

# Example evaluation
res = compute_icer(28400.0, 14200.0, 6.85, 6.15, 25000.0)
print(f"ICER: £{res['icer']:,.2f}/QALY | Status: {res['verdict']}")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: ICER & Cost-Effectiveness Plane")
        st.markdown("Health technology cost-effectiveness models are standardly built in Microsoft Excel for NICE dossier submissions.")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Cell B2 (`Cost_Intervention`):** `=28400`
            * **Cell B3 (`Cost_Comparator`):** `=14200`
            * **Cell B4 (`QALY_Intervention`):** `=6.85`
            * **Cell B5 (`QALY_Comparator`):** `=6.15`
            * **Cell B6 (`WTP_Threshold_Lambda`):** `=25000`
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame) and not df.empty:
                st.dataframe(df.head(6), use_container_width=True, height=240, hide_index=True)
            df_cells = pd.DataFrame({
                "Cell / Range": ["Cell E2", "Cell E3", "Cell E4", "Cell E5", "Cell E6"],
                "Target Metric": ["Incremental Cost (ΔC)", "Incremental QALYs (ΔE)", "Calculated ICER", "Net Monetary Benefit (NMB)", "Decision Verdict"],
                "Exact Excel Formula": ["=B2-B3", "=B4-B5", "=E2/E3", "=(B6*E3)-E2", '=IF(AND(E3>0,E2<0),"Dominant",IF(AND(E3<0,E2>0),"Dominated",IF(E4<=B6,"Cost-Effective","Not Cost-Effective")))'],
                "Calculated Output Value": [f"£{results['delta_cost']:,.0f}", f"{results['delta_qaly']:+.2f}", results["icer_str"], f"£{results['nmb']:,.0f}", "Cost-Effective" if results["is_cost_effective"] else "Not Cost-Effective"]
            }) if pd is not None else None
            if df_cells is not None:
                st.dataframe(df_cells, use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Calculate Incremental Differences:** In `Cell E2`, enter `=B2-B3`. In `Cell E3`, enter `=B4-B5`.
            2. **Compute ICER:** In `Cell E4`, enter `=E2/E3`.
            3. **Compute NMB:** In `Cell E5`, enter `=(B6*E3)-E2`.
            4. **Evaluate Quadrant:** Check whether $\Delta E > 0$ and $\Delta C \le 0$ (Dominance).
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Health Technology Assessment Interpretation:
            * **Primary Outcome:** The novel technology generates an incremental **{results['delta_qaly']:+.2f} QALYs** at an additional lifetime cost of **£{results['delta_cost']:,.0f}**, yielding an ICER of **{results['icer_str']}**.
            * **NICE Appraisal Verdict:** Because the ICER is below the £{results['wtp_threshold']:,.0f}/QALY threshold (NMB = £{results['nmb']:,.0f} > 0), this technology represents an efficient use of NHS resources.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: Biologic DMARD Patient Cohort")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download Patient Cohort Dataset (CSV)", csv_bytes, "hta_dmard_patient_cohort.csv", "text/csv", key="dl_icer_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)
        st.markdown(r"""
        #### 📋 Variable Dictionary:
        | Column Name | Data Type | Description |
        | :--- | :--- | :--- |
        | `patient_id` | String | Unique patient cohort identifier |
        | `arm` | String | Assigned clinical evaluation arm |
        | `treatment_name` | String | Specific therapy administered |
        | `total_lifetime_cost` | Float | Discounted lifetime healthcare cost (£) |
        | `total_qalys` | Float | Discounted lifetime Quality-Adjusted Life Years |
        | `survival_years` | Float | Total years of life lived |
        """)


def _render_nmb_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
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
        st.markdown("#### Production Python Pipeline: Net Monetary Benefit Ranking")
        st.code("""def rank_strategies_by_nmb(strategies, wtp=30000.0):
    for s in strategies:
        s["nmb"] = (wtp * s["qalys"]) - s["cost"]
        s["nhb"] = s["qalys"] - (s["cost"] / wtp)
    return sorted(strategies, key=lambda x: x["nmb"], reverse=True)

strats = [
    {"name": "Standard Care", "cost": 8500.0, "qalys": 4.20},
    {"name": "ARNI", "cost": 15800.0, "qalys": 4.62},
    {"name": "SGLT2i", "cost": 19400.0, "qalys": 4.88}
]
ranked = rank_strategies_by_nmb(strats, 30000.0)
print(f"Optimal Strategy: {ranked[0]['name']} (NMB: £{ranked[0]['nmb']:,.0f})")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Net Monetary Benefit Ranking")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Column A (`Strategy`):** Strategy name (`Row 2` to `Row 4`).
            * **Column B (`Cost`):** Total cost in £.
            * **Column C (`QALYs`):** Mean QALYs.
            * **Column D (`NMB`):** `=(30000*C2)-B2`.
            * **Column E (`NHB`):** `=C2-(B2/30000)`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                st.dataframe(df, use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Calculate NMB:** In `Cell D2`, enter `=(30000*C2)-B2` and drag down.
            2. **Identify Optimal Strategy:** In `Cell D6`, enter `=INDEX(A2:A4, MATCH(MAX(D2:D4), D2:D4, 0))`.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Quality Improvement & Commissioning Interpretation:
            * **Optimal Decision:** **{results['optimal_treatment']}** provides the greatest population health benefit per pound spent, with an NMB of **£{results['max_nmb']:,.0f}**.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: Multi-Arm NMB Table")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download Multi-Arm NMB Data (CSV)", csv_bytes, "hta_multi_arm_nmb.csv", "text/csv", key="dl_nmb_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df, use_container_width=True, height=240, hide_index=True)


def _render_markov_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
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
        st.markdown("#### Production Python Pipeline: 3-State Markov Cohort Model with Discounting")
        st.code("""import numpy as np

def run_markov_cohort(cycles=10, p_matrix=None, utilities=[0.82, 0.54, 0.0], costs=[18500, 12000, 0], disc_rate=0.035):
    state = np.array([1.0, 0.0, 0.0])
    history = [state]
    for _ in range(cycles):
        state = state @ p_matrix
        history.append(state)
        
    tot_cost, tot_qaly = 0.0, 0.0
    for t in range(cycles):
        # Half-cycle correction
        hcc = (history[t] + history[t+1]) / 2.0
        df = 1.0 / ((1.0 + disc_rate) ** t)
        tot_cost += np.dot(hcc, costs) * df
        tot_qaly += np.dot(hcc, utilities) * df
    return tot_cost, tot_qaly

p_imm = np.array([[0.72, 0.22, 0.06], [0.0, 0.65, 0.35], [0.0, 0.0, 1.0]])
c, q = run_markov_cohort(10, p_imm)
print(f"Immunotherapy 10-Yr Discounted Cost: £{c:,.0f} | QALYs: {q:.2f}")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Markov Cohort Model")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Columns A-D:** Cycle 0 to 10, State proportions (PFS, PD, Dead).
            * **Columns E-G:** Half-cycle corrected proportions `=AVERAGE(B2, B3)`.
            * **Column H:** Discount Factor `=(1+$B$1)^(-A2)`.
            * **Column I:** Discounted Costs `=SUMPRODUCT(E2:G2, $B$15:$D$15)*H2`.
            * **Column J:** Discounted QALYs `=SUMPRODUCT(E2:G2, $B$16:$D$16)*H2`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                st.dataframe(df.head(6), use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Markov Trace:** Multiply previous row vector by transition matrix using `MMULT`.
            2. **Half-Cycle Correction:** Take trapezoidal mean of row $t$ and $t+1$.
            3. **Discounting:** Apply 3.5% discount factor `=1/(1.035^t)`.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Clinical & Economic Interpretation:
            * **10-Year Discounted ICER:** **£{results['icer']:,.0f} / QALY** (Immunotherapy vs Chemotherapy).
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: 10-Year Markov Trace")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download Markov Trace (CSV)", csv_bytes, "hta_markov_10yr_trace.csv", "text/csv", key="dl_markov_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)


def _render_psa_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
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
        st.markdown("#### Production Python Pipeline: Monte Carlo PSA & CEAC Generator")
        st.code("""import numpy as np

def run_psa(b=1000, mean_c=(32000, 24000), sd_c=(4500, 3800), mean_q=(5.40, 4.95), sd_q=(0.45, 0.50)):
    # Gamma for costs
    sh_t = (mean_c[0]**2)/(sd_c[0]**2); sc_t = (sd_c[0]**2)/mean_c[0]
    sh_c = (mean_c[1]**2)/(sd_c[1]**2); sc_c = (sd_c[1]**2)/mean_c[1]
    costs_t = np.random.gamma(sh_t, sc_t, b)
    costs_c = np.random.gamma(sh_c, sc_c, b)
    
    qalys_t = np.random.normal(mean_q[0], sd_q[0], b)
    qalys_c = np.random.normal(mean_q[1], sd_q[1], b)
    
    delta_c = costs_t - costs_c
    delta_q = qalys_t - qalys_c
    
    # CEAC at £30k
    p_ce_30k = np.mean((30000 * delta_q - delta_c) > 0)
    return delta_c, delta_q, p_ce_30k

dc, dq, p_ce = run_psa(1000)
print(f"Prob Cost-Effective at £30k/QALY: {p_ce*100:.1f}%")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Monte Carlo PSA")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Column A:** Iteration index `1` to `1000`.
            * **Column B (`Cost_T`):** `=GAMMA.INV(RAND(), alpha, beta)`.
            * **Column C (`Cost_C`):** `=GAMMA.INV(RAND(), alpha, beta)`.
            * **Column D (`QALY_T`):** `=NORM.INV(RAND(), mean, sd)`.
            * **Column E (`QALY_C`):** `=NORM.INV(RAND(), mean, sd)`.
            * **Column F (`NMB_30k`):** `=(30000*(D2-E2))-(B2-C2)`.
            * **Column G (`Is_CE`):** `=IF(F2>0, 1, 0)`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                st.dataframe(df.head(6), use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Random Draws:** Use Excel Data Tables (`Data -> What-If Analysis -> Data Table`) with `=RAND()` to recalculate 1,000 Monte Carlo runs without VBA.
            2. **CEAC Probability:** Enter `=AVERAGE(G2:G1001)` to calculate $P(\text{NMB} > 0)$.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 Decision Risk Interpretation:
            * **Reimbursement Confidence:** Across 1,000 simulated parameter sets, the technology has a **{results['prob_ce_pct']:.1f}% probability of being cost-effective** at £{results['wtp_threshold']:,.0f}/QALY.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: PSA Simulation Draws")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download PSA Iterations (CSV)", csv_bytes, "hta_psa_simulation_draws.csv", "text/csv", key="dl_psa_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df.head(8), use_container_width=True, height=240, hide_index=True)


def _render_bia_tabs(meta: Dict[str, Any], df: Any, results: Dict[str, Any]) -> None:
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
        st.markdown("#### Production Python Pipeline: 5-Year Budget Impact Model")
        st.code("""def run_bia(pop=250000, uptakes=[0.05, 0.12, 0.22, 0.35, 0.48], c_new=1650, c_curr=380, offset=620):
    yearly = []
    tot_impact = 0.0
    for yr, up in enumerate(uptakes, start=1):
        n_treated = int(pop * up)
        net_impact = n_treated * (c_new - c_curr - offset)
        tot_impact += net_impact
        yearly.append({"year": yr, "treated": n_treated, "net_impact": net_impact})
    return yearly, tot_impact

res, tot = run_bia()
print(f"5-Year Cumulative NHS Budget Impact: £{tot:,.0f}")
""", language="python")
    with t3:
        st.markdown("### 📊 Microsoft Excel Implementation Guide: Budget Impact Analysis")
        with st.expander("📌 Step 1: Data Architecture & Spreadsheet Cell Layout", expanded=False):
            st.markdown(r"""
            * **Row 1:** Financial Years (`Year 1` to `Year 5`).
            * **Row 2:** Market Uptake `%`.
            * **Row 3:** Treated Cohort `=Eligible_Pop * Uptake`.
            * **Row 4:** Status Quo Budget `=Eligible_Pop * Cost_Current`.
            * **Row 5:** Revised Budget `=(Row 3 * Cost_New) + ((Eligible_Pop - Row 3) * Cost_Current) - (Row 3 * Hospital_Offset)`.
            * **Row 6:** Net Budget Impact `=Row 5 - Row 4`.
            """)
            if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
                st.dataframe(df, use_container_width=True, height=240, hide_index=True)
        with st.expander("📌 Step 2: Step-by-Step Excel Formulas & Calculations", expanded=False):
            st.markdown(r"""
            1. **Undiscounted Cash Flow:** Do NOT apply discounting formulas to BIA rows (ISPOR requirement).
            2. **Cumulative Total:** Enter `=SUM(B6:F6)` to calculate 5-year budget total.
            """)
        with st.expander("📌 Step 3: Diagnostic Validation & Output Interpretation", expanded=False):
            st.markdown(f"""
            #### 💡 NHS Payer Financial Interpretation:
            * **Net 5-Year Budget Impact:** **£{results['cumulative_5yr_net_budget_impact']:,.0f}** across {results['eligible_population']:,} eligible patients.
            * **Affordability Benchmark:** Year 1 net spend is **£{results['yearly_results'][0]['net_budget_impact']:,.0f}**, well under the standard £20m NHS Budget Impact Test threshold.
            """)
    with t4:
        st.markdown("### 📥 Download Practice Dataset: 5-Year Budget Impact Financial Table")
        csv_bytes = df.to_csv(index=False).encode("utf-8") if HAS_UI_STACK and isinstance(df, pd.DataFrame) else b""
        st.download_button("📥 Download BIA Model Data (CSV)", csv_bytes, "hta_5yr_budget_impact.csv", "text/csv", key="dl_bia_csv")
        if HAS_UI_STACK and pd is not None and isinstance(df, pd.DataFrame):
            st.dataframe(df, use_container_width=True, height=240, hide_index=True)


# =====================================================================
# 5. Master Module Router
# =====================================================================

def render_hta_method_module(method_key: str) -> None:
    """Master router executing the chosen HTA method module."""
    meta = HTA_METHODS.get(method_key, HTA_METHODS["icer_plane"])

    st.markdown(f"## {meta['icon']} {meta['name']}")
    st.markdown(f"*{meta['tagline']}*")

    if method_key == "icer_plane":
        _render_icer_sandbox(meta)
    elif method_key == "nmb_nhb":
        _render_nmb_sandbox(meta)
    elif method_key == "markov_model":
        _render_markov_sandbox(meta)
    elif method_key == "psa_ceac":
        _render_psa_sandbox(meta)
    elif method_key == "budget_impact":
        _render_bia_sandbox(meta)

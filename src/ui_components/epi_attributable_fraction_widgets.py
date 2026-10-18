"""Streamlit UI Components & Visualizations for Attributable Fractions (AF_e) and Population Attributable Fraction (PAF).

Provides:
1. render_attributable_fraction_plain_english_guide: Zero-prior-knowledge walkthrough with Rose's Prevention Paradox and analogies.
2. render_attributable_fraction_diagnostics: Tri-color evaluation cards (PASS, WARNING, FAIL) for causal and confounding assumptions.
3. render_attributable_fraction_dashboard: Metric cards (AF_e, PAF, RR, Pe, PAR, attributable cases, Levin/Miettinen concordance).
4. render_attributable_fraction_visualizations: Preventable vs unpreventable pie breakdown, Rose Paradox sensitivity curve, and 2x2 grid.
5. render_attributable_fraction_excel_guide: Strict 3-step sequence with cell mapping and explicit formulas.
6. render_attributable_fraction_python_guide: Strict 4-step sequence with exhaustive docstrings and conditional prints.
7. render_attributable_fraction_download_tab: Live CSV download for both individual microdata and 2x2 table.
8. render_attributable_fraction_module: Top-level controller for Attributable Fractions & Population Impact.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.attributable_fraction import (
    calculate_attributable_fraction,
    evaluate_attributable_fraction_prerequisites,
    generate_attributable_fraction_demo_data
)


def render_attributable_fraction_plain_english_guide():
    """Renders the zero-prior-knowledge pedagogical guide for AF_e and PAF."""
    with st.expander("💡 Plain English Guide: Attributable Fraction (AF_e) & Population Attributable Fraction (PAF)", expanded=False):
        st.markdown("""
        In epidemiology and public health, knowing that an exposure is harmful ($RR > 1$) is only the beginning. 
        **Attributable Fractions** answer the two most vital questions for individual patients and public health commissioners:
        1. **The Individual Patient's Question:** *"Given that I was exposed and got sick, what proportion of my illness was caused by the exposure?"*
        2. **The Public Health Director's Question:** *"Across our entire county or city, what percentage of all cases would disappear if we eradicated this hazard?"*
        """)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. Attributable Fraction in the Exposed ($AF_e$)
            * **Definition:** The proportion of disease cases among the **exposed individuals** that is attributable to the exposure and would not have occurred without it.
            * **Mathematical Formula:**
              $$AF_e = \frac{I_e - I_u}{I_e} = \frac{RR - 1}{RR} = 1 - \frac{1}{RR}$$
            * *Example (Smoking & Lung Cancer):* If tobacco smokers have a Relative Risk of $15.0$ compared to non-smokers:
              $$AF_e = \frac{15 - 1}{15} = \frac{14}{15} \approx \mathbf{93.3\%}$$
              Among smokers who develop lung cancer, **93.3% of their cancer is directly attributable to smoking**. Only $6.7\%$ is background risk.
            * **Protective Interventions (Vaccine Efficacy):** For protective interventions where $RR < 1$, $AF_e$ becomes the **Prevented Fraction in the Exposed ($PF_e = 1 - RR$)**, universally known in clinical trials as **Vaccine Efficacy (VE)** or **Relative Risk Reduction (RRR)**!
            """)

            st.markdown(r"""
            #### 2. Population Attributable Fraction ($PAF$)
            * **Definition:** The proportion of all disease cases in the **entire population (both exposed and unexposed combined)** that is attributable to the exposure and could be prevented by eliminating it.
            * **Levin's Formula (using Population Prevalence $P_e$):**
              $$PAF = \frac{P_e(RR - 1)}{1 + P_e(RR - 1)}$$
            * **Miettinen's Formula (using Case Exposure Proportion $P_{c|e} = \frac{a}{a + c}$):**
              $$PAF = P_{c|e} \left(\frac{RR - 1}{RR}\right) = P_{c|e} \times AF_e$$
            * **Why Miettinen's Formula is Loved in Epidemiology:** In a case-control study, you do not know the total population prevalence $P_e$, but you *can* observe the percentage of cases that are exposed ($P_{c|e}$)!
            """)

        with col_g2:
            st.markdown(r"""
            #### 3. Geoffrey Rose's "Prevention Paradox"
            * In 1985, British epidemiologist **Geoffrey Rose** articulated a profound public health truth:
              > *"A large number of people at a small risk may give rise to more cases of disease than the small number who are at a high risk."*
            * **The Extreme Hazard vs. Universal Hazard Comparison:**
              * **Extreme Hazard (Asbestos / High-Risk Stunt):** Huge $RR = 15.0$, but only $0.2\%$ of the population is exposed. $AF_e$ is massive ($93\%$), but $PAF$ in the general community is tiny ($< 3\%$)!
              * **Universal Moderate Hazard (Physical Inactivity / Air Pollution):** Modest $RR = 1.30$, but $50\%$ of the population is exposed! $AF_e$ is only $23\%$, but $PAF$ is **$13.0\%$**!
            * **The Public Health Lesson:** Getting the entire sedentary population to take a daily brisk walk prevents far more total heart attacks than only treating a handful of elite high-cholesterol patients in specialist hospital clinics.
            """)

            st.markdown(r"""
            #### 4. The Core Analogy: "The Speed Limit vs. The Stunt Driver"
            * **The Stunt Driver ($AF_e$):** Driving at 140 mph has a huge Relative Risk of crashing ($RR = 50$). If a stunt driver crashes, $98\%$ of the crash is attributable to speeding ($AF_e = 98\%$). But almost nobody drives at 140 mph!
            * **The 35 mph in a 30 mph Zone ($PAF$):** Driving 5 mph over the limit has a modest $RR = 1.25$. But millions of drivers do it every day ($P_e = 60\%$). Across the UK, **thousands of pedestrian injuries** are attributable to minor speeding ($PAF = 13\%$).
            * That is why public health policy targets speed limits and clean air rather than just penalizing stunt drivers!
            """)

        st.info(r"""
        📐 **Why are Causal Assumptions Critical for Attributable Fractions?**  
        Unlike Relative Risk (which merely quantifies statistical association), Attributable Fractions **assume genuine biological causality** (Bradford Hill criteria). 
        If an exposure is merely a bystander or confounded (e.g. yellow fingers and lung cancer), eliminating the exposure will **NOT** prevent any cases! 
        Always verify that confounding has been controlled and biological plausibility is established before interpreting $PAF$ as preventable cases.
        """)


def render_attributable_fraction_diagnostics(evaluations: List[Dict[str, Any]]):
    """Renders tri-color prerequisite diagnostic cards for Attributable Fractions."""
    st.markdown("### 🚦 Tri-Color Bias & Methodological Diagnostics")
    st.caption("Verifies causal validity (Bradford Hill), confounding control, exposure prevalence representativeness, and cell sizes before interpreting AF_e and PAF.")

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


def render_attributable_fraction_dashboard(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders summary metric cards for AF_e and PAF."""
    st.markdown("### 📊 Primary Epidemiological Impact Measures")

    is_protective = (calcs["effect_direction"] == "protective")
    afe_pct = calcs["afe"] * 100.0
    paf_pct = calcs["paf"] * 100.0
    paf_ll = calcs["paf_ci_lower"] * 100.0
    paf_ul = calcs["paf_ci_upper"] * 100.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if not is_protective:
            st.metric(
                label="Attributable Fraction (Exposed) [AF_e]",
                value=f"{afe_pct:.1f}%",
                delta=f"95% CI: [{calcs['afe_ci_lower']*100.0:.1f}%, {calcs['afe_ci_upper']*100.0:.1f}%]",
                delta_color="off",
                help="Proportion of disease among exposed individuals attributable to exposure: (RR - 1) / RR"
            )
        else:
            ve_pct = calcs["prevented_fraction_exposed"] * 100.0 if calcs["prevented_fraction_exposed"] is not None else 0.0
            st.metric(
                label="Vaccine Efficacy / Prevented Fraction [PF_e]",
                value=f"{ve_pct:.1f}%",
                help="Relative reduction in disease among vaccinated/treated individuals: 1 - RR"
            )
    with c2:
        if not is_protective:
            st.metric(
                label="Population Attributable Fraction [PAF]",
                value=f"{paf_pct:.1f}%",
                delta=f"95% CI: [{paf_ll:.1f}%, {paf_ul:.1f}%]",
                delta_color="off",
                help="Proportion of all cases in total population attributable to exposure: Levin's / Miettinen's formula"
            )
        else:
            pfp_pct = calcs["prevented_fraction_population"] * 100.0 if calcs["prevented_fraction_population"] is not None else 0.0
            st.metric(
                label="Prevented Fraction (Population) [PFP]",
                value=f"{pfp_pct:.1f}%",
                help="Proportion of potential population disease burden prevented by vaccination: Pe * (1 - RR)"
            )
    with c3:
        st.metric(
            label="Relative Risk (RR)",
            value=f"{calcs['rr']:.2f}",
            delta=f"95% CI: [{calcs['rr_ci_lower']:.2f}, {calcs['rr_ci_upper']:.2f}]",
            delta_color="off",
            help="Biological strength of association: I_e / I_u"
        )
    with c4:
        st.metric(
            label="Population Exposure Prevalence (P_e)",
            value=f"{calcs['pe'] * 100.0:.1f}%",
            help="Percentage of the target population carrying the exposure: n1 / N (or demographic override)"
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        target_name = meta.get("target_population_name", "Target Population")
        st.metric(
            label=f"Preventable Cases in {target_name}",
            value=f"{calcs['attributable_cases']:,}",
            help=f"Estimated annual cases preventable if exposure is eliminated: Total Cases * PAF"
        )
    with c6:
        st.metric(
            label="Population Attributable Risk (PAR)",
            value=f"{calcs['par'] * 100.0:+.2f}%",
            help="Absolute excess disease incidence in the total community attributable to hazard: I_total - I_u"
        )
    with c7:
        st.metric(
            label="Proportion of Cases Exposed (P_c|e)",
            value=f"{calcs['p_cases_exposed'] * 100.0:.1f}%",
            help="Cases exposed over total cases: a / (a + c). Drives Miettinen's formula: PAF = P_c|e * AF_e"
        )
    with c8:
        lev_diff = abs(calcs["paf_levin"] - calcs["paf_miettinen"]) * 100.0
        st.metric(
            label="Levin vs. Miettinen Concordance",
            value="100% Identical" if lev_diff < 0.1 else f"Δ {lev_diff:.2f}%",
            help="In closed cohorts, Levin's formula (P_e) and Miettinen's formula (P_c|e) are mathematically identical."
        )


def render_attributable_fraction_visualizations(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders Plotly pie breakdown, Rose Prevention Paradox sensitivity curve, and 2x2 grid."""
    st.markdown("### 📈 Visualizing Community Disease Burden & The Rose Prevention Paradox")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**1. Population Disease Burden: Attributable vs. Background Cases**")
        paf_val = max(0.0, min(1.0, calcs["paf"]))
        unattributable = 1.0 - paf_val
        exp_name = meta.get("exposure_name", "Exposure")

        labels = [f"Attributable to {exp_name} ({paf_val*100.0:.1f}%)", f"Background / Other Causes ({unattributable*100.0:.1f}%)"]
        values = [paf_val * 100.0, unattributable * 100.0]
        colors = ["#dc2626", "#3b82f6"] if calcs["effect_direction"] == "harmful" else ["#10b981", "#64748b"]

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker_colors=colors,
            textinfo="label+percent",
            insidetextorientation="radial"
        )])
        fig_pie.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### 🍩 What is the Population Disease Burden Chart?")
            st.markdown(
                f"This donut chart breaks down all disease cases occurring across the total community into two actionable categories:\n\n"
                f"* **Attributable to {exp_name} ({paf_val*100.0:.1f}%):** The proportion of total community disease directly caused by this exposure that would be eliminated if the hazard were eradicated.\n"
                f"* **Background / Other Causes ({unattributable*100.0:.1f}%):** Cases arising from non-modifiable factors like baseline genetics, natural aging, or unrelated environmental causes.\n"
                f"* **Clinical & Policy Meaning:** If public health interventions successfully eliminate this exposure in {meta.get('target_population_name', 'the target population')} ({meta.get('target_population_size', 800000):,} people), approximately **{calcs['attributable_cases']:,} cases** would be prevented every year."
            )

    with col_chart2:
        st.markdown("**2. The Rose Prevention Paradox: PAF vs. Exposure Prevalence**")
        # Generate PAF curves across exposure prevalences from 1% to 100% for various RR levels
        pe_range = np.linspace(0.01, 0.99, 100)
        rr_levels = [1.25, 1.5, 2.0, 5.0, 15.0]
        color_palette = ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1e3a8a"]

        fig_rose = go.Figure()
        for idx, rr_val in enumerate(rr_levels):
            paf_curve = (pe_range * (rr_val - 1.0)) / (1.0 + pe_range * (rr_val - 1.0)) * 100.0
            fig_rose.add_trace(go.Scatter(
                x=pe_range * 100.0,
                y=paf_curve,
                mode="lines",
                name=f"RR = {rr_val}",
                line=dict(color=color_palette[idx], width=2)
            ))

        # Add active scenario point
        active_pe = calcs["pe"] * 100.0
        active_paf = calcs["paf"] * 100.0
        fig_rose.add_trace(go.Scatter(
            x=[active_pe],
            y=[active_paf],
            mode="markers+text",
            marker=dict(size=13, color="#ef4444", symbol="diamond"),
            text=[f"Current: {active_paf:.1f}% (Pe={active_pe:.1f}%)"],
            textposition="top left",
            name="Active Investigation"
        ))

        fig_rose.update_layout(
            xaxis_title="Population Exposure Prevalence P_e (%)",
            yaxis_title="Population Attributable Fraction PAF (%)",
            height=340,
            margin=dict(l=30, r=30, t=30, b=40),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_rose, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### 🔍 What is the Rose Prevention Paradox?")
            st.markdown(
                "Formulated in 1985 by British epidemiologist **Geoffrey Rose**, the paradox states:\n"
                "> *\"A large number of people exposed to a small risk often generates far more total cases of disease in a community than a small number of people exposed to an extreme risk.\"*\n\n"
                "**What this chart is showing:**\n"
                f"* **Horizontal X-Axis ($P_e$):** Exposure prevalence across the community (from 0% to 100%).\n"
                f"* **Vertical Y-Axis ($PAF\\%$):** Percentage of all community cases eliminated if exposure ceased.\n"
                f"* **Colored Curves ($RR = 1.25$ to $15.0$):** Demonstrate how population impact scales under Levin's formula: $PAF = \\frac{{P_e(RR - 1)}}{{1 + P_e(RR - 1)}}$.\n"
                f"* **Red Diamond Marker:** Active investigation coordinate ($P_e = {calcs['pe']*100.0:.1f}\\%, \\, PAF = {calcs['paf']*100.0:.1f}\\%$).\n"
                f"* **The Paradox in Action:** A modest risk ($RR = 1.50$) affecting 60% of people generates $PAF \\approx 23\\%$, matching an extreme risk ($RR = 15.0$) affecting only 2%!"
            )

    # 3. Interactive Contingency Matrix Display
    st.markdown("**3. Contingency Matrix Structure & Incidence Marginals**")
    a = calcs["a"]
    b = calcs["b"]
    c = calcs["c"]
    d = calcs["d"]
    n1 = calcs["n1"]
    n0 = calcs["n0"]
    n_tot = calcs["n_total"]
    m1 = calcs["m1"]
    m0 = calcs["m0"]

    exp_label, unexp_label = meta.get("exposure_levels", ("Exposed", "Unexposed"))
    out_pos, out_neg = meta.get("outcome_levels", ("Cases (Event +)", "Non-Cases (Event -)"))

    df_2x2_display = pd.DataFrame([
        {
            "Cohort Stratum": f"🟢 {exp_label}",
            f"{out_pos}": f"{a:,}",
            f"{out_neg}": f"{b:,}",
            "Stratum Total (n)": f"{n1:,}",
            "Cumulative Incidence": f"{calcs['risk_exposed'] * 100.0:.2f}%",
            "Impact Measure": f"AF_e = {calcs['afe'] * 100.0:.1f}%" if calcs['effect_direction'] == 'harmful' else f"VE = {calcs['prevented_fraction_exposed']*100.0:.1f}%"
        },
        {
            "Cohort Stratum": f"⚪ {unexp_label}",
            f"{out_pos}": f"{c:,}",
            f"{out_neg}": f"{d:,}",
            "Stratum Total (n)": f"{n0:,}",
            "Cumulative Incidence": f"{calcs['risk_unexposed'] * 100.0:.2f}%",
            "Impact Measure": "Reference Baseline (Unexposed)"
        },
        {
            "Cohort Stratum": "📊 Total Population",
            f"{out_pos}": f"{m1:,} (a + c)",
            f"{out_neg}": f"{m0:,} (b + d)",
            "Stratum Total (n)": f"{n_tot:,} (Total N)",
            "Cumulative Incidence": f"{calcs['risk_total'] * 100.0:.2f}%",
            "Impact Measure": f"PAF = {calcs['paf'] * 100.0:.1f}% (95% CI: [{calcs['paf_ci_lower']*100.0:.1f}%, {calcs['paf_ci_upper']*100.0:.1f}%])"
        }
    ])

    st.dataframe(df_2x2_display, use_container_width=True, hide_index=True)


def render_attributable_fraction_excel_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 4-step Microsoft Excel Implementation Guide with cell coordinates and charts."""
    st.markdown("### 📗 Microsoft Excel Implementation Workflow for AF_e & PAF")
    st.markdown("Follow the 4-step sequence: **Set up 2×2 Layout** ➔ **Calculate Incidences, RR & AF_e** ➔ **Calculate PAF (Levin & Miettinen) and Walter 95% CI** ➔ **Create the Population Burden Donut & Rose Paradox Charts**.")

    a, b, c, d = calcs["a"], calcs["b"], calcs["c"], calcs["d"]
    n1, n0 = calcs["n1"], calcs["n0"]
    m1, m0 = calcs["m1"], calcs["m0"]
    n_tot = calcs["n_total"]
    exp_label, unexp_label = meta.get("exposure_levels", ("Exposed", "Unexposed"))

    with st.expander("📌 Step 1: Excel 2×2 Data Layout & Cell References", expanded=False):
        st.markdown(f"""
        Structure your Excel worksheet with the 2×2 table in `Cells A1:E4` and calculation outputs in `Cells A6:B15`:
        
        * **Column A (Rows 2:4):** Exposure Group (`Row 2: Exposed`, `Row 3: Unexposed`, `Row 4: Total`)
        * **Column B (Rows 2:4):** Cases (`B2`: $a = {a}$, `B3`: $c = {c}$, `B4`: `=SUM(B2:B3)`)
        * **Column C (Rows 2:4):** Non-Cases (`C2`: $b = {b}$, `C3`: $d = {d}$, `C4`: `=SUM(C2:C3)`)
        * **Column D (Rows 2:4):** Stratum Totals (`D2`: `=SUM(B2:C2)`, `D3`: `=SUM(B3:C3)`, `D4`: `=SUM(B4:C4)`)
        * **Column E (Rows 2:4):** Cumulative Risk (`E2`: `=B2/D2`, `E3`: `=B3/D3`, `E4`: `=B4/D4`)
        """)

        preview_grid = pd.DataFrame([
            {"Excel_Row": "Row 1 (Headers)", "Col_A": "Exposure_Group", "Col_B": "Cases_Outcome_Pos", "Col_C": "NonCases_Outcome_Neg", "Col_D": "Total_Cohort_n", "Col_E": "Cumulative_Risk"},
            {"Excel_Row": "Row 2 (Exposed)", "Col_A": exp_label, "Col_B": str(a), "Col_C": str(b), "Col_D": f"{n1} (=SUM(B2:C2))", "Col_E": f"{calcs['risk_exposed']:.4f} (=B2/D2)"},
            {"Excel_Row": "Row 3 (Unexposed)", "Col_A": unexp_label, "Col_B": str(c), "Col_C": str(d), "Col_D": f"{n0} (=SUM(B3:C3))", "Col_E": f"{calcs['risk_unexposed']:.4f} (=B3/D3)"},
            {"Excel_Row": "Row 4 (Totals)", "Col_A": "Total Population", "Col_B": f"{m1} (=SUM(B2:B3))", "Col_C": f"{m0} (=SUM(C2:C3))", "Col_D": f"{n_tot} (=SUM(D2:D3))", "Col_E": f"{calcs['risk_total']:.4f} (=B4/D4)"}
        ])
        st.dataframe(preview_grid, use_container_width=True, hide_index=True)

        st.markdown("**Summary Output Cells Mapping (`Cells A6:B15`):**")
        df_summary_cells = pd.DataFrame({
            "Cell": ["B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15"],
            "Metric Name": [
                "Cumulative Risk Exposed (I_e)",
                "Cumulative Risk Unexposed (I_u)",
                "Total Population Risk (I_tot)",
                "Relative Risk (RR = I_e / I_u)",
                "Attributable Fraction Exposed (AF_e)",
                "Population Exposure Prevalence (P_e)",
                "PAF - Levin's Formula",
                "PAF - Miettinen's Formula",
                "Walter's SE for ln(1 - PAF)",
                "95% CI Lower Bound for PAF"
            ],
            "Excel Formula": [
                "=B2/D2",
                "=B3/D3",
                "=B4/D4",
                "=B6/B7",
                "=(B9-1)/B9",
                "=D2/D4",
                "=(B11*(B9-1))/(1+B11*(B9-1))",
                "=(B2/B4)*B10",
                "=SQRT((B3/(B2*B4))+(C2/(D3*D4)))",
                "=1 - EXP(LN(1-B12) + 1.96*B14)"
            ],
            "Calculated Value": [
                f"{calcs['risk_exposed']:.4f}",
                f"{calcs['risk_unexposed']:.4f}",
                f"{calcs['risk_total']:.4f}",
                f"{calcs['rr']:.4f}",
                f"{calcs['afe']:.4f}",
                f"{calcs['pe']:.4f}",
                f"{calcs['paf_levin']:.4f}",
                f"{calcs['paf_miettinen']:.4f}",
                f"{calcs.get('se_ln_comp_paf', 0.05):.4f}",
                f"{calcs['paf_ci_lower']:.4f}"
            ]
        })
        st.dataframe(df_summary_cells, use_container_width=True, hide_index=True)

    with st.expander("📌 Step 2: Calculate Incidences, RR & AF_e in Excel", expanded=False):
        st.markdown("""
        1. **Compute Incidence in Exposed ($I_e$):** In cell `B6`, enter `=B2/D2`.
        2. **Compute Incidence in Unexposed ($I_u$):** In cell `B7`, enter `=B3/D3`.
        3. **Compute Total Incidence ($I_{total}$):** In cell `B8`, enter `=B4/D4`.
        4. **Compute Relative Risk (RR):** In cell `B9`, enter:
           ```excel
           =B6 / B7
           ```
        5. **Compute Attributable Fraction in Exposed ($AF_e$):** In cell `B10`, enter:
           ```excel
           =(B9 - 1) / B9
           ```
        """)

    with st.expander("📌 Step 3: Compute PAF (Levin & Miettinen) and Walter 95% CI in Excel", expanded=False):
        st.markdown("""
        1. **Compute Population Exposure Prevalence ($P_e$):** In cell `B11`, enter `=D2/D4`.
        2. **Compute PAF using Levin's Formula:** In cell `B12`, enter:
           ```excel
           =(B11 * (B9 - 1)) / (1 + B11 * (B9 - 1))
           ```
        3. **Compute PAF using Miettinen's Formula:** In cell `B13`, enter:
           ```excel
           =(B2 / B4) * B10
           ```
        4. **Compute Walter's Log-Scale Standard Error:** In cell `B14`, enter:
           ```excel
           =SQRT((B3 / (B2 * B4)) + (C2 / (D3 * D4)))
           ```
        5. **Compute 95% CI Lower Bound for PAF:** In cell `B15`, enter:
           ```excel
           =1 - EXP(LN(1 - B12) + 1.96 * B14)
           ```
        6. **Compute 95% CI Upper Bound for PAF:** In cell `B16`, enter:
           ```excel
           =1 - EXP(LN(1 - B12) - 1.96 * B14)
           ```
        """)

    with st.expander("📌 Step 4: Create the Population Burden Donut & Rose Paradox Sensitivity Charts in Excel", expanded=False):
        st.markdown(f"""
        ### Chart A: Population Disease Burden Donut Chart (Attributable vs. Background)
        Visualizes the proportion of total community cases attributable to the exposure versus background/other causes.

        1. **Set Up Data Range (Cells `D18:E19`):**
           * In cell `D18`, enter label: `Attributable Cases`
           * In cell `E18`, enter formula: `=B12 * 100` *(PAF percentage)*
           * In cell `D19`, enter label: `Background / Other Causes`
           * In cell `E19`, enter formula: `=(1 - B12) * 100` *(Unattributable percentage)*
        2. **Insert Donut Chart:**
           * Highlight cells `D18:E19`.
           * On the Excel Ribbon, click **Insert** ➔ **Charts** ➔ **Pie or Doughnut Chart** dropdown ➔ click **Doughnut**.
        3. **Format Slices & Labels:**
           * Double-click the Attributable slice: set Fill Color to **Red** (or Green if protective).
           * Double-click the Background slice: set Fill Color to **Blue** or **Slate Gray**.
           * Right-click chart ➔ select **Add Data Labels** ➔ Right-click labels ➔ **Format Data Labels** ➔ check **Category Name** and **Percentage**.

        ---

        ### Chart B: The Rose Prevention Paradox Sensitivity Curves (XY Scatter with Smooth Lines)
        Plots how Population Attributable Fraction ($PAF$) grows as Exposure Prevalence ($P_e$) increases, across multiple Relative Risk levels ($RR = 1.25, 1.50, 2.00, 5.00, 15.00$).

        1. **Set Up Prevalence & RR Table (Columns `G:L`, Rows 1 to 21):**
           * In cell `G1`, enter header: `Exposure_Pe_Pct`
           * In cells `G2:G21`, enter prevalence values from `0.05` to `0.95` in 0.05 increments (`G2` = `0.05`, `G3` = `0.10`, ..., `G21` = `0.95`). Format column G as Percentage (`5%` to `95%`).
           * In cells `H1:L1`, enter the Relative Risk levels: `1.25` in `H1`, `1.50` in `I1`, `2.00` in `J1`, `5.00` in `K1`, `15.00` in `L1`.
           * In cell `H2`, enter Levin's formula referencing the prevalence in Column G and RR in Row 1:
             ```excel
             =($G2 * (H$1 - 1)) / (1 + $G2 * (H$1 - 1)) * 100
             ```
           * Copy cell `H2` across columns `H2:L2`, and drag down through row 21 (`H2:L21`).
        2. **Enter Active Study Coordinates (Cells `N1:O2`):**
           * In cell `N1`, enter: `Active_Pe` | In cell `O1`, enter: `Active_PAF`
           * In cell `N2`, enter: `=B11 * 100` *(Current Pe %)*
           * In cell `O2`, enter: `=B12 * 100` *(Current PAF %)*
        3. **Insert the XY Scatter Chart:**
           * Select data range `G1:L21`.
           * On the Excel Ribbon, click **Insert** ➔ **Charts** ➔ **Scatter (X, Y)** ➔ click **Scatter with Smooth Lines**.
           * Excel plots $P_e$ on the horizontal X-axis and each $RR$ column as a distinct colored curve on the vertical Y-axis.
        4. **Overlay Active Study Marker (Red Diamond):**
           * Right-click inside the chart ➔ click **Select Data...**
           * Under *Legend Entries (Series)*, click **Add**.
           * In **Series name**, enter: `="Active Study"`.
           * In **Series X values**, select: `=Sheet1!$N$2`.
           * In **Series Y values**, select: `=Sheet1!$O$2`.
           * Click **OK**.
           * Click the newly added single data point on the chart ➔ Right-click ➔ **Format Data Series** ➔ **Fill & Line** ➔ **Marker** ➔ **Marker Options** ➔ Built-in: **Diamond**, Size: `10`, Fill: **Solid Red**, Border: **None**.
        5. **Axis Titles & Scaling:**
           * Click chart ➔ **Chart Design** ➔ **Add Chart Element** ➔ **Axis Titles**:
             * Primary Horizontal: `Population Exposure Prevalence P_e (%)`
             * Primary Vertical: `Population Attributable Fraction PAF (%)`
           * Double-click Horizontal X-axis: set Minimum = `0`, Maximum = `100`.
           * Double-click Vertical Y-axis: set Minimum = `0`, Maximum = `100`.
        """)


def render_attributable_fraction_python_guide(calcs: Dict[str, Any], meta: Dict[str, Any]):
    """Renders the 5-step Python Implementation Workflow with docstrings, conditional prints, and visualizations."""
    st.markdown("### 🐍 Python Implementation Workflow for AF_e & PAF")
    st.markdown("This guide follows a strict 5-step sequence: **Evaluate Prerequisites & Causal Assumptions** ➔ **Calculate AF_e, Levin's PAF & Miettinen's PAF** ➔ **Compute Walter / Greenland 95% Confidence Intervals** ➔ **Consolidated Executable Pipeline Script** ➔ **Visualize the Rose Prevention Paradox & Population Burden**.")

    a, b, c, d = calcs["a"], calcs["b"], calcs["c"], calcs["d"]

    with st.expander("📌 Step 1: Evaluate Prerequisites & Causal Assumptions in Python", expanded=False):
        st.code('''def check_attributable_fraction_prerequisites(
    study_design: str,
    a: int,
    b: int,
    c: int,
    d: int,
    causality_established: bool = True,
    confounding_controlled: bool = True
) -> dict:
    """Verifies epidemiological prerequisites, Bradford Hill causality, and cell frequencies for AF_e/PAF.

    Args:
        study_design (str): Architecture: 'prospective_cohort', 'retrospective_cohort', 'cross_sectional', 'case_control'.
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        causality_established (bool, optional): Bradford Hill biological causality. Defaults to True.
        confounding_controlled (bool, optional): Multivariable adjustment for age/sex/lifestyle. Defaults to True.

    Returns:
        dict: Diagnostic pass/warning/fail flags.
    """
    if not causality_established:
        print("[INAPPROPRIATE - FAIL] Exposure is non-causal! AF_e and PAF represent preventable cases ONLY when causality is proven.")
        return {"status": "FAIL"}

    print("[ASSURANCE - PASS] Causal relationship assumed. Attributable fractions represent genuine preventable disease burden.")

    if not confounding_controlled:
        print("[WARNING] Confounding unadjusted! Crude risk ratio will bias the PAF estimate upwards or downwards.")
    else:
        print("[ASSURANCE - PASS] Confounding adjusted. Risk ratio reflects direct etiologic effect.")

    min_cell = min(a, b, c, d)
    if min_cell < 5:
        print(f"[WARNING] Minimum cell count is {min_cell} (< 5). Asymptotic Walter variance may be unstable.")
    else:
        print(f"[ASSURANCE - PASS] Minimum cell count is {min_cell} (>= 5). Asymptotic normal approximations robust.")

    return {"status": "PASS"}
''', language="python")

    with st.expander("📌 Step 2: Calculate AF_e, Levin's PAF & Miettinen's PAF in Python", expanded=False):
        st.code('''def calculate_attributable_fractions(
    a: int,
    b: int,
    c: int,
    d: int,
    pe_override: float = None
) -> dict:
    """Calculates Attributable Fraction in Exposed (AF_e), Levin's PAF, and Miettinen's PAF.

    Args:
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        pe_override (float, optional): Target population exposure prevalence (0 to 1). Defaults to None.

    Returns:
        dict: Risk ratio, AF_e, Levin's PAF, Miettinen's PAF, and PAR.
    """
    n1 = a + b
    n0 = c + d
    n_tot = n1 + n0
    m1 = a + c

    risk_exp = a / n1
    risk_unexp = c / n0
    risk_tot = m1 / n_tot
    rr = risk_exp / risk_unexp

    # Attributable Fraction in Exposed: (RR - 1) / RR
    afe = (rr - 1.0) / rr

    # Population Exposure Prevalence
    pe = pe_override if pe_override is not None else (n1 / n_tot)

    # Levin's Formula: [Pe * (RR - 1)] / [1 + Pe * (RR - 1)]
    paf_levin = (pe * (rr - 1.0)) / (1.0 + pe * (rr - 1.0))

    # Miettinen's Formula: P_{cases|exposed} * (RR - 1) / RR
    p_cases_exposed = a / m1
    paf_miettinen = p_cases_exposed * afe

    # Population Attributable Risk (absolute rate difference)
    par = risk_tot - risk_unexp

    return {
        "risk_exposed": risk_exp,
        "risk_unexposed": risk_unexp,
        "rr": rr,
        "afe": afe,
        "pe": pe,
        "paf_levin": paf_levin,
        "paf_miettinen": paf_miettinen,
        "par": par
    }
''', language="python")

    with st.expander("📌 Step 3: Compute Walter / Greenland 95% Confidence Intervals in Python", expanded=False):
        st.code('''import numpy as np
from scipy import stats

def compute_paf_confidence_intervals(
    a: int,
    b: int,
    c: int,
    d: int,
    paf: float,
    alpha: float = 0.05
) -> dict:
    """Computes Walter (1976) / Greenland (1987) log-scale variance and 95% CI for PAF.

    Args:
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        paf (float): Point estimate of PAF.
        alpha (float, optional): Significance level. Defaults to 0.05 (95% CI).

    Returns:
        dict: Standard error of ln(1 - PAF), lower CI, and upper CI for PAF.
    """
    n0 = c + d
    n_tot = a + b + c + d
    m1 = a + c

    # Walter's formula: Var(ln(1 - PAF)) = c / [a * (a + c)] + b / [n0 * n_tot]
    var_ln_comp = (c / (a * m1)) + (b / (n0 * n_tot))
    se_ln_comp = np.sqrt(var_ln_comp)

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ln_comp = np.log(1.0 - paf)

    # Invert back to PAF scale
    comp_upper = np.exp(ln_comp + z_crit * se_ln_comp)
    comp_lower = np.exp(ln_comp - z_crit * se_ln_comp)

    paf_ci_lower = max(0.0, 1.0 - comp_upper)
    paf_ci_upper = min(1.0, 1.0 - comp_lower)

    return {
        "se_ln_comp_paf": se_ln_comp,
        "paf_ci_lower": paf_ci_lower,
        "paf_ci_upper": paf_ci_upper
    }
''', language="python")

    with st.expander("📌 Step 4: Consolidated Executable Pipeline Script", expanded=False):
        st.code(f'''import numpy as np
from scipy import stats

def run_attributable_fraction_pipeline(
    a: int,
    b: int,
    c: int,
    d: int,
    pe_override: float = {calcs['pe']:.4f},
    target_population_size: int = {calcs.get('total_population_size') or 800000}
) -> dict:
    """Executes end-to-end Attributable Fraction (AF_e) and Population Attributable Fraction (PAF) analysis.

    Args:
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        pe_override (float, optional): Population prevalence of exposure.
        target_population_size (int, optional): Size of the target jurisdiction.

    Returns:
        dict: Full epidemiological impact report.
    """
    print("=" * 65)
    print("EPIDEMIOLOGICAL ATTRIBUTABLE FRACTIONS (AF_e & PAF) PIPELINE")
    print("=" * 65)

    n1 = a + b
    n0 = c + d
    n_tot = n1 + n0
    m1 = a + c

    # Step 1: Incidences & Risk Ratio
    risk_exp = a / n1
    risk_unexp = c / n0
    risk_tot = m1 / n_tot
    rr = risk_exp / risk_unexp

    # Step 2: Attributable Fraction in Exposed (AF_e)
    afe = (rr - 1.0) / rr

    # Step 3: Population Attributable Fraction (PAF)
    pe = pe_override if pe_override is not None else (n1 / n_tot)
    paf_levin = (pe * (rr - 1.0)) / (1.0 + pe * (rr - 1.0))
    paf_miettinen = (a / m1) * afe

    # Step 4: Walter's 95% Confidence Interval for PAF
    var_ln_comp = (c / (a * m1)) + (b / (n0 * n_tot))
    se_ln_comp = np.sqrt(var_ln_comp)
    z_crit = 1.95996
    ln_comp = np.log(1.0 - paf_levin)
    paf_ci_lower = max(0.0, 1.0 - np.exp(ln_comp + z_crit * se_ln_comp))
    paf_ci_upper = min(1.0, 1.0 - np.exp(ln_comp - z_crit * se_ln_comp))

    # Step 5: Preventable Cases in Target Population
    expected_total_cases = int(round(target_population_size * risk_tot))
    preventable_cases = int(round(expected_total_cases * paf_levin))

    print("\\n--- RESULTS ---")
    print(f"Risk in Exposed (I_e):             {{risk_exp * 100.0:.2f}}%")
    print(f"Risk in Unexposed (I_u):           {{risk_unexp * 100.0:.2f}}%")
    print(f"Relative Risk (RR):                {{rr:.2f}}")
    print(f"Attributable Fraction Exposed (AF_e): {{afe * 100.0:.2f}}%")
    print(f"Population Exposure Prevalence:     {{pe * 100.0:.2f}}%")
    print(f"Population Attributable Fraction (PAF): {{paf_levin * 100.0:.2f}}%")
    print(f"95% Walter CI for PAF:             [{{paf_ci_lower * 100.0:.2f}}%, {{paf_ci_upper * 100.0:.2f}}%]")
    print(f"Preventable Cases in Target Pop:    {{preventable_cases:,}} cases (out of {{expected_total_cases:,}} total)")

    return {{
        "rr": rr,
        "afe": afe,
        "pe": pe,
        "paf": paf_levin,
        "paf_ci_lower": paf_ci_lower,
        "paf_ci_upper": paf_ci_upper,
        "preventable_cases": preventable_cases
    }}

# Execute with scenario data ({meta['title']}):
if __name__ == "__main__":
    results = run_attributable_fraction_pipeline(
        a={a}, b={b}, c={c}, d={d},
        pe_override={calcs['pe']:.4f},
        target_population_size={calcs.get('total_population_size') or 800000}
    )
''', language="python")

    with st.expander("📌 Step 5: Visualize the Rose Prevention Paradox & Population Burden in Python", expanded=False):
        st.code('''import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_rose_prevention_paradox(
    current_pe: float,
    current_paf: float,
    active_label: str = "Active Investigation",
    use_plotly: bool = True
):
    """Plots the Rose Prevention Paradox sensitivity curves across exposure prevalences.

    Visualizes Geoffrey Rose's principle: demonstrates that high-prevalence modest hazards
    generate equal or greater community disease burden than low-prevalence extreme hazards.

    Args:
        current_pe (float): Active study exposure prevalence (0.0 to 1.0, e.g. 0.40 for 40%).
        current_paf (float): Active study Population Attributable Fraction (0.0 to 1.0, e.g. 0.138 for 13.8%).
        active_label (str, optional): Legend label for current study marker. Defaults to 'Active Investigation'.
        use_plotly (bool, optional): If True, returns an interactive Plotly figure; if False, renders Matplotlib plot.

    Returns:
        go.Figure or None: Interactive Plotly figure if use_plotly=True, otherwise displays Matplotlib figure.
    """
    pe_range = np.linspace(0.01, 0.99, 100)
    rr_levels = [1.25, 1.5, 2.0, 5.0, 15.0]

    if use_plotly:
        color_palette = ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1e3a8a"]
        fig = go.Figure()

        # Plot Levin's PAF curve for each RR level: PAF = [Pe * (RR - 1)] / [1 + Pe * (RR - 1)]
        for idx, rr in enumerate(rr_levels):
            paf_curve = (pe_range * (rr - 1.0)) / (1.0 + pe_range * (rr - 1.0)) * 100.0
            fig.add_trace(go.Scatter(
                x=pe_range * 100.0,
                y=paf_curve,
                mode="lines",
                name=f"RR = {rr}",
                line=dict(color=color_palette[idx], width=2)
            ))

        # Add active scenario point marker (Red Diamond)
        fig.add_trace(go.Scatter(
            x=[current_pe * 100.0],
            y=[current_paf * 100.0],
            mode="markers+text",
            marker=dict(size=14, color="#ef4444", symbol="diamond"),
            text=[f"Current: {current_paf * 100.0:.1f}% (Pe={current_pe * 100.0:.1f}%)"],
            textposition="top left",
            name=active_label
        ))

        fig.update_layout(
            title="The Rose Prevention Paradox: PAF vs. Exposure Prevalence",
            xaxis_title="Population Exposure Prevalence P_e (%)",
            yaxis_title="Population Attributable Fraction PAF (%)",
            template="plotly_white",
            height=400,
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    else:
        plt.figure(figsize=(9, 6))
        for rr in rr_levels:
            paf_curve = (pe_range * (rr - 1.0)) / (1.0 + pe_range * (rr - 1.0)) * 100.0
            plt.plot(pe_range * 100.0, paf_curve, label=f"RR = {rr}", lw=2)

        # Plot active scenario marker
        plt.scatter(
            [current_pe * 100.0], [current_paf * 100.0],
            color="red", s=130, marker="D", zorder=5,
            label=f"{active_label} ({current_paf * 100.0:.1f}%)"
        )
        plt.title("The Rose Prevention Paradox: PAF vs. Exposure Prevalence", fontsize=12, fontweight="bold")
        plt.xlabel("Population Exposure Prevalence P_e (%)", fontsize=10)
        plt.ylabel("Population Attributable Fraction PAF (%)", fontsize=10)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()


def plot_population_burden_donut(
    paf: float,
    exposure_name: str = "Exposure",
    use_plotly: bool = True
):
    """Generates a donut chart comparing Attributable vs. Background disease burden.

    Args:
        paf (float): Population Attributable Fraction (0.0 to 1.0).
        exposure_name (str, optional): Name of the exposure hazard. Defaults to 'Exposure'.
        use_plotly (bool, optional): If True returns Plotly figure, else Matplotlib.

    Returns:
        go.Figure or None: Donut chart figure.
    """
    paf_val = max(0.0, min(1.0, paf))
    unattributable = 1.0 - paf_val

    labels = [f"Attributable to {exposure_name}", "Background / Other Causes"]
    values = [paf_val * 100.0, unattributable * 100.0]
    colors = ["#dc2626", "#3b82f6"]

    if use_plotly:
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker_colors=colors,
            textinfo="label+percent"
        )])
        fig.update_layout(
            title=f"Population Disease Burden: {exposure_name}",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        return fig
    else:
        plt.figure(figsize=(6, 6))
        plt.pie(
            values, labels=labels, autopct="%1.1f%%",
            startangle=90, colors=colors,
            wedgeprops=dict(width=0.4, edgecolor="white")
        )
        plt.title(f"Population Disease Burden: {exposure_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

# Example usage:
# fig_rose = plot_rose_prevention_paradox(current_pe=0.40, current_paf=0.138, use_plotly=True)
# fig_rose.show()
''', language="python")


def render_attributable_fraction_download_tab(
    df_contingency: pd.DataFrame,
    df_microdata: pd.DataFrame,
    meta: Dict[str, Any]
):
    """Renders the data download tab with individual microdata and 2x2 contingency table CSVs."""
    st.markdown("### 📥 Download Sample Datasets")
    st.markdown(f"""
    Download the sample data associated with the active scenario: **{meta['title']}**.
    You can use these datasets to practice calculating Attributable Fractions ($AF_e$), 
    Levin's and Miettinen's $PAF$, Walter's confidence intervals, and preventable cases in Microsoft Excel or Python.
    """)

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.markdown("#### 1. Individual Participant Microdata (CSV)")
        st.markdown(f"Contains **{len(df_microdata):,} participant records** with columns: `Participant_ID`, `Exposure_Group`, `Exposure_Binary`, `Outcome_Status`, `Outcome_Binary`, `Age`, `Sex`, `District`.")
        csv_micro = df_microdata.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Individual Participant Microdata (CSV)",
            data=csv_micro,
            file_name=f"{meta['scenario_key']}_microdata.csv",
            mime="text/csv",
            key="btn_dl_af_microdata"
        )
        st.dataframe(df_microdata.head(8), height=220, use_container_width=True, hide_index=True)

    with col_dl2:
        st.markdown("#### 2. Aggregated 2×2 Contingency Table (CSV)")
        st.markdown("Contains the summarized 2×2 contingency table matching the exact Excel tutorial cell coordinates (`A1:E4`).")
        csv_cont = df_contingency.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Aggregated 2×2 Table (CSV)",
            data=csv_cont,
            file_name=f"{meta['scenario_key']}_2x2_contingency_table.csv",
            mime="text/csv",
            key="btn_dl_af_contingency"
        )
        st.dataframe(df_contingency, height=220, use_container_width=True, hide_index=True)


def render_attributable_fraction_module(active_method_key: str, method_info: Dict[str, Any]):
    """Renders the comprehensive Attributable Fractions (AF_e & PAF) module."""
    # 1. Plain English Guide
    render_attributable_fraction_plain_english_guide()

    # 2. Sidebar Controls
    st.sidebar.header("🧪 Population Impact Settings")
    st.sidebar.markdown("**Public Health Scenarios & Target Populations:**")

    af_scenario_choice = st.sidebar.selectbox(
        "Study Scenario:",
        options=["smoking_lung_cancer", "physical_inactivity_cvd", "air_pollution_asthma", "hpv_vaccine_cervical", "custom"],
        format_func=lambda s: {
            "smoking_lung_cancer": "🚬 Derbyshire Tobacco Smoking & Lung Cancer (AF_e = 93%, PAF = 66%)",
            "physical_inactivity_cvd": "🏃 Derbyshire Inactivity & CVD: The Rose Paradox (AF_e = 29%, PAF = 14%)",
            "air_pollution_asthma": "🏭 Urban A38/M1 Corridor PM2.5 & Pediatric Asthma (AF_e = 20%, PAF = 16%)",
            "hpv_vaccine_cervical": "💉 Derbyshire HPV Vaccine: Prevented Fraction (VE = 88%, PFP = 75%)",
            "custom": "⚙️ Custom 2×2 Table Input (Enter Your Own Counts & Prevalence)"
        }[s],
        index=0,
        help="Select a realistic public health scenario or enter custom 2x2 contingency table counts."
    )
    st.sidebar.caption("⚠️ **Data Notice:** Scenarios, populations, and event counts are made-up examples synthetically simulated for educational and illustrative purposes only.")

    if af_scenario_choice == "custom":
        st.sidebar.subheader("Custom 2×2 Parameters")
        custom_study_design = st.sidebar.selectbox(
            "Study Design Architecture:",
            options=["prospective_cohort", "retrospective_cohort", "cross_sectional", "case_control", "rct"],
            format_func=lambda d: {
                "prospective_cohort": "Prospective Cohort (Forward-looking)",
                "retrospective_cohort": "Retrospective Cohort (Historical records)",
                "cross_sectional": "Cross-Sectional (Prevalence snapshot)",
                "case_control": "Case-Control (Miettinen formula applied)",
                "rct": "Randomized Controlled Trial (RCT)"
            }[d],
            index=0
        )
        custom_target_pop = st.sidebar.number_input(
            "Target Population Jurisdiction Size:",
            min_value=1000,
            max_value=10000000,
            value=800000,
            step=50000,
            help="Size of total community (e.g. Derbyshire adult population = 800,000)"
        )
        custom_pe_override = st.sidebar.slider(
            "Population Exposure Prevalence P_e (%):",
            min_value=1.0,
            max_value=99.0,
            value=25.0,
            step=1.0,
            help="Prevalence of exposure in the general population (drives Levin's formula)"
        ) / 100.0

        col_sb1, col_sb2 = st.sidebar.columns(2)
        with col_sb1:
            a_val = st.number_input("Exposed Cases (a):", min_value=0, max_value=100000, value=80, step=1)
            b_val = st.number_input("Exposed Non-Cases (b):", min_value=0, max_value=100000, value=920, step=5)
        with col_sb2:
            c_val = st.number_input("Unexposed Cases (c):", min_value=0, max_value=100000, value=30, step=1)
            d_val = st.number_input("Unexposed Non-Cases (d):", min_value=0, max_value=100000, value=2970, step=5)

        meta = {
            "title": "Custom User-Defined Population Impact Investigation",
            "domain": "Epidemiological Hypothesis Testing",
            "description": "User-configured 2x2 contingency evaluation for Attributable Fractions.",
            "exposure_name": "Target Exposure",
            "exposure_levels": ("Exposed Cohort", "Unexposed Cohort"),
            "outcome_name": "Target Health Event",
            "outcome_levels": ("Cases (Event +)", "Non-Cases (Event -)"),
            "study_design": custom_study_design,
            "expected_effect": "User Defined",
            "clinical_takeaway": "Custom parameters evaluated based on user-entered contingency frequencies.",
            "scenario_key": "custom_attributable_fraction",
            "target_population_name": "Custom Target Population",
            "target_population_size": custom_target_pop,
            "pe_override": custom_pe_override
        }

        # Build contingency and microdata
        df_contingency = pd.DataFrame([
            {
                "Cohort Stratum": "🟢 Exposed Cohort",
                "Cases (Event +)": a_val,
                "Non-Cases (Event -)": b_val,
                "Stratum Total (n)": a_val + b_val,
                "Cumulative Risk": float(a_val / (a_val + b_val)) if (a_val + b_val) > 0 else 0.0
            },
            {
                "Cohort Stratum": "⚪ Unexposed Cohort",
                "Cases (Event +)": c_val,
                "Non-Cases (Event -)": d_val,
                "Stratum Total (n)": c_val + d_val,
                "Cumulative Risk": float(c_val / (c_val + d_val)) if (c_val + d_val) > 0 else 0.0
            },
            {
                "Cohort Stratum": "📊 Total Population",
                "Cases (Event +)": a_val + c_val,
                "Non-Cases (Event -)": b_val + d_val,
                "Stratum Total (n)": a_val + b_val + c_val + d_val,
                "Cumulative Risk": float((a_val + c_val) / (a_val + b_val + c_val + d_val)) if (a_val + b_val + c_val + d_val) > 0 else 0.0
            }
        ])

        micro_list = []
        for i in range(min(a_val + b_val, 150)):
            had_event = 1 if i < a_val else 0
            micro_list.append({"Participant_ID": f"EXP_{i+1:04d}", "Exposure_Group": "Exposed", "Exposure_Binary": 1, "Outcome_Status": "Event +" if had_event else "Event -", "Outcome_Binary": had_event, "Age": 55, "Sex": "Male"})
        for i in range(min(c_val + d_val, 150)):
            had_event = 1 if i < c_val else 0
            micro_list.append({"Participant_ID": f"UNEXP_{i+1:04d}", "Exposure_Group": "Unexposed", "Exposure_Binary": 0, "Outcome_Status": "Event +" if had_event else "Event -", "Outcome_Binary": had_event, "Age": 54, "Sex": "Female"})
        df_microdata = pd.DataFrame(micro_list)

        calcs = calculate_attributable_fraction(
            a_val, b_val, c_val, d_val,
            pe_override=custom_pe_override,
            total_population_size=custom_target_pop
        )
        evaluations = evaluate_attributable_fraction_prerequisites(
            study_design=custom_study_design,
            a=a_val, b=b_val, c=c_val, d=d_val,
            is_causality_established=True,
            is_confounding_controlled=True,
            is_exposure_prevalence_representative=True
        )
    else:
        df_contingency, df_microdata, meta = generate_attributable_fraction_demo_data(scenario=af_scenario_choice, seed=42)
        calcs = meta["calcs"]
        evaluations = evaluate_attributable_fraction_prerequisites(
            study_design=meta.get("study_design", "prospective_cohort"),
            a=meta["a"], b=meta["b"], c=meta["c"], d=meta["d"],
            is_causality_established=True,
            is_confounding_controlled=True,
            is_exposure_prevalence_representative=True
        )

    # Narrative callout for active scenario
    st.markdown(f"""
    <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 1.1rem; border-radius: 6px; margin-bottom: 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.35rem; flex-wrap: wrap; gap: 0.5rem;">
            <h4 style="margin: 0; color: #1e3a8a;">🔬 Active Scenario: {meta['title']}</h4>
            <span style="font-size: 0.75rem; background-color: #fef3c7; color: #92400e; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 600; border: 1px solid #fde68a;">⚠️ Made-Up Illustrative Scenario</span>
        </div>
        <p style="margin: 0 0 0.4rem 0; font-size: 0.95rem; color: #334155;">{meta['description']}</p>
        <div style="font-size: 0.85rem; color: #0369a1; font-weight: 600; margin-bottom: 0.4rem;">💡 Clinical & Policy Takeaway: {meta['clinical_takeaway']}</div>
        <div style="font-size: 0.8rem; color: #64748b; font-style: italic; border-top: 1px dashed #cbd5e1; padding-top: 0.4rem;">
            ⚠️ <strong>Disclaimer:</strong> This is a completely made-up example scenario with synthetic data simulated for educational and illustrative purposes only. It does not represent actual medical records or real surveillance data from NHS trusts or public health authorities.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Tri-Color Diagnostics
    render_attributable_fraction_diagnostics(evaluations)

    # 2. Metric Dashboard
    render_attributable_fraction_dashboard(calcs, meta)

    # 3. Visualizations & 2x2 Grid
    render_attributable_fraction_visualizations(calcs, meta)

    # 4. Methodology Tabs
    st.markdown("---")
    tab_af_rationale, tab_af_excel, tab_af_code, tab_af_download = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Excel Implementation Guide",
        "🐍 Python Code Implementation",
        "📥 Download Sample Data"
    ])

    with tab_af_rationale:
        st.subheader(f"Why Attributable Fractions? ({method_info['name']})")
        st.markdown("""
        In public health policy and resource commissioning, Relative Risk tells us only about biological potency. 
        **Attributable Fractions (AF_e and PAF)** translate that biological risk into concrete, actionable numbers of preventable hospitalizations and deaths.
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
        st.markdown("### 🔍 The Epidemiological Impact Spectrum: Individual vs. Population Burden")
        st.markdown(r"""
        | Dimension | Attributable Fraction in Exposed ($AF_e$) | Population Attributable Fraction ($PAF$) | Population Attributable Risk ($PAR$) | Relative Risk ($RR$) |
        | :--- | :--- | :--- | :--- | :--- |
        | **Primary Question** | *"What fraction of disease in exposed people is due to exposure?"* | *"What fraction of all community cases is due to exposure?"* | *"What is the absolute excess rate in the community?"* | *"How many times higher is disease risk in exposed?"* |
        | **Target Audience** | **Clinician & Patient** in individual consultations | **Public Health Director & Policymaker** | **Health Economist & Modeler** | **Etiologic Researcher & Biologist** |
        | **Formula** | $\frac{RR - 1}{RR} = 1 - \frac{1}{RR}$ | $\frac{P_e(RR - 1)}{1 + P_e(RR - 1)}$ (Levin) | $I_{total} - I_u = P_e \times RD$ | $\frac{I_e}{I_u}$ |
        | **Prevalence Dependence** | **Independent of Prevalence** (Only depends on RR) | **Critically Dependent on $P_e$** (Rose Paradox) | **Critically Dependent on $P_e$** | **Independent of Prevalence** |
        | **Null Benchmark** | **$0.0\%$** ($RR = 1.0$) | **$0.0\%$** ($RR = 1.0$) | **$0.00$** ($RD = 0.0$) | **$1.00$** (Equal risk) |
        | **Protective Analog** | **Vaccine Efficacy / $PF_e = 1 - RR$** | **Prevented Fraction in Population ($PFP$)** | Absolute Risk Reduction in Population | Relative Risk ($RR < 1.0$) |
        """)

        st.markdown("#### 📖 Key Concepts Explained (Zero Prior Knowledge Assumed)")
        st.markdown(r"""
        1. **Levin's vs. Miettinen's Formulas: When to Use Which?**
           * **Levin's Formula ($PAF = \frac{P_e(RR - 1)}{1 + P_e(RR - 1)}$):** Use this formula when you know the **prevalence of exposure in the total general population ($P_e$)**, such as from census data, national health surveys, or a representative prospective cohort.
           * **Miettinen's Formula ($PAF = P_{cases|exposed} \times \frac{RR - 1}{RR}$):** In a **case-control study**, you intentionally recruit cases and controls in a fixed ratio (e.g. 100 cases and 100 controls), destroying the natural population prevalence $P_e$. But Miettinen's formula uses the proportion of cases who were exposed ($P_{c|e}$), which is directly observable in case-control studies!
           * In a standard cohort study, both formulas give **100% identical results**!

        2. **Why High RR Does Not Mean High Population Burden (The Rose Paradox):**
           * Imagine Exposure A has an astronomical $RR = 20.0$, but only 1 person in 10,000 is exposed ($P_e = 0.01\%$).
             * $AF_e = (20 - 1) / 20 = 95.0\%$.
             * $PAF = \frac{0.0001 \times 19}{1 + 0.0001 \times 19} = \frac{0.0019}{1.0019} \approx \mathbf{0.19\%}$!
             * Eradicating this chemical prevents less than $0.2\%$ of the total disease burden.
           * Now imagine Exposure B has a modest $RR = 1.30$, but $60\%$ of adults are exposed ($P_e = 60\%$).
             * $AF_e = (1.30 - 1) / 1.30 = 23.1\%$.
             * $PAF = \frac{0.60 \times 0.30}{1 + 0.60 \times 0.30} = \frac{0.18}{1.18} \approx \mathbf{15.3\%}$!
             * Eradicating this mild risk prevents **15.3% of all cases** in the entire county!

        3. **Prevented Fraction for Vaccines and Interventions ($RR < 1.0$):**
           * When an exposure is protective (e.g., childhood HPV vaccination with $RR = 0.12$):
           * The **Prevented Fraction in the Exposed ($PF_e$)** is:
             $$PF_e = 1 - RR = 1 - 0.12 = \mathbf{88\%}$$
             This is what clinicians and clinical trial reports call **Vaccine Efficacy (VE)**.
           * The **Prevented Fraction in the Population ($PFP$)** combines efficacy with coverage ($P_v = 85\%$):
             $$PFP = P_v \times (1 - RR) = 0.85 \times 0.88 = \mathbf{74.8\%}$$
             The school vaccination campaign has prevented **75% of all potential cervical cancer cases** that would have otherwise occurred!
        """)

    with tab_af_excel:
        render_attributable_fraction_excel_guide(calcs, meta)

    with tab_af_code:
        render_attributable_fraction_python_guide(calcs, meta)

    with tab_af_download:
        render_attributable_fraction_download_tab(df_contingency, df_microdata, meta)


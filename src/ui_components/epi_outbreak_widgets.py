"""Interactive UI Components & Visualizations for Outbreak Epidemiology, Attack Rates & Vehicle Incrimination.

Provides:
1. Plain English Pedagogical Introductory Guide to Outbreak Investigations, Attack Rates,
   Epidemic Curves, and Secondary Transmission (Zero prior epidemiological knowledge).
2. Sidebar scenario controls with 3 realistic outbreak scenarios and custom outbreak cohort exploration.
3. Methodological Prerequisite Evaluation (5 CDC / UKHSA standards: Vehicle Incrimination Distinctiveness,
   Cohort Size & Attack Rate Power, Incubation Plausibility, Secondary Transmission, Epi Curve Morphology).
4. Primary Metric Dashboard (Overall Attack Rate, Incriminated Vehicle, Relative Risk with 95% CI,
   Median Incubation Period, Secondary Attack Rate with Wilson 95% CI).
5. Plotly Epidemic Curve (Epi Curve) with onset timeline, exposure event marker, and median incubation annotation.
6. Plotly Food-Specific Attack Rate & Relative Risk Incrimination Plot (Forest plot of RRs on log scale).
7. Full-Width Pedagogical Explanatory Guide beneath plots explaining how to read epi curves and vehicle matrices.
8. Complete Food-Specific Attack Rate Incrimination Matrix Table (Observed cells, attack rates, RRs, ARDs, AF_e, p-values).
9. Full 4-Tab Educational Architecture:
   - 📋 Rationale & Epidemiological Framework (CDC 10-step framework, Attack Rate math, Woolf CIs, SAR, ORION).
   - 📗 Microsoft Excel Implementation Guide (Collapsible expanders for each step, formulas, chart construction).
   - 🐍 Python Production Script Guide (Collapsible expanders for each stage, clean reusable code).
   - 💾 Live Synthetic Data Downloads (Attendee microdata CSV, food matrix CSV, and epi curve onset CSV).
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Union
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.outbreak_investigation import (
    calculate_food_specific_attack_rates,
    calculate_secondary_attack_rate,
    calculate_epidemic_curve,
    evaluate_outbreak_prerequisites,
    generate_outbreak_demo_data
)


def render_outbreak_plain_english_guide():
    """Renders a collapsible plain English introductory guide to outbreak epidemiology."""
    with st.expander("📖 Zero-Knowledge Guide: What is Outbreak Epidemiology & Attack Rate Analysis?", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. The Emergency Dilemma: When Disease Strikes a Population
            * In clinical epidemiology, we often study chronic diseases over years. But in **Outbreak Epidemiology**, an acute surge of illness occurs suddenly in a confined group of people (e.g. wedding guests, hospital ward patients, care home residents, or restaurant patrons).
            * Field epidemiologists from health protection agencies (such as the UK Health Security Agency [UKHSA] or CDC) are deployed to answer four urgent questions:
              1. *What pathogen is causing the illness?*
              2. *What contaminated food, water, or environmental vehicle introduced it?*
              3. *Is the illness spreading from person to person (secondary transmission)?*
              4. *What immediate control actions are needed to stop the outbreak?*
            """)

            st.markdown(r"""
            #### 2. What is an Attack Rate (AR)?
            * In an outbreak setting, the term "incidence rate" is replaced by **Attack Rate (AR)**:
              $$\text{Attack Rate} = \frac{\text{Number of People who Became Ill}}{\text{Total Number of People at Risk}} \times 100\%$$
            * Unlike true rates that accumulate over calendar years, an attack rate is a **cumulative incidence proportion** calculated over the finite duration of the outbreak event.
            * An overall attack rate of $50\%$ means that half of everyone who attended the event developed symptoms.
            """)

            st.markdown(r"""
            #### 3. How to Incriminate a Food Vehicle: Food-Specific Attack Rates
            * To pinpoint which item on a 10-course banquet menu was contaminated, we compare the attack rate among people who **ate** the item against people who **did not eat** the item:
              * **Attack Rate in Exposed ($\text{AR}_{\text{exp}}$):** $\frac{\text{Ill who ate}}{\text{Total who ate}} \times 100\%$
              * **Attack Rate in Unexposed ($\text{AR}_{\text{unexp}}$):** $\frac{\text{Ill who did not eat}}{\text{Total who did not eat}} \times 100\%$
            * **Risk Ratio (Relative Risk):** $\text{RR} = \frac{\text{AR}_{\text{exp}}}{\text{AR}_{\text{unexp}}}$.
            * **Attack Rate Difference (Risk Difference):** $\text{ARD} = \text{AR}_{\text{exp}} - \text{AR}_{\text{unexp}}$.
            * **The Culprit Test:** The incriminated item typically has:
              1. A very high Relative Risk ($\text{RR} > 3.0$ or $5.0$).
              2. A large positive Attack Rate Difference ($\text{ARD} > +40\%$).
              3. A very low attack rate among people who avoided it ($\text{AR}_{\text{unexp}} < 15\%$).
              4. High exposure coverage (most ill persons ate it).
            """)

        with col_g2:
            st.markdown(r"""
            #### 4. The Epidemic Curve (Epi Curve): The Outbreak's Fingerprint
            * An **Epidemic Curve (Epi Curve)** is a time-series histogram showing the number of new cases by date and hour of symptom onset.
            * Its visual shape immediately tells epidemiologists the transmission mechanism:
              * **Point-Source Epidemic:** A single, brief common exposure (e.g. wedding dinner). Produces a sharp, steep rise to a single peak, followed by a gradual right-hand tail. The distance from the meal to the median case is the **Median Incubation Period**.
              * **Continuous Common-Source:** Ongoing exposure to a contaminated well or water supply. Produces a broad, flattened plateau rather than a sharp peak.
              * **Propagated Epidemic:** Person-to-person spread. Produces multiple successive waves of cases, each taller than the last, spaced approximately one incubation period apart.
            """)

            st.markdown(r"""
            #### 5. Secondary Attack Rate (SAR): Human-to-Human Communicability
            * After the contaminated food is identified and discarded, does the outbreak stop? Not if the pathogen can spread from person to person!
            * The **Secondary Attack Rate (SAR)** measures household or close-contact spread:
              $$\text{SAR} = \frac{\text{Secondary Cases}}{\text{Susceptible Close Contacts (excluding primary index cases)}} \times 100\%$$
            * For food intoxications (e.g. *Staphylococcus aureus* toxin or *Bacillus cereus*), $\text{SAR} = 0\%$ because toxins cannot spread between humans.
            * For highly contagious viral agents like **Norovirus** or bacterial pathogens like **Shigella**, $\text{SAR}$ can exceed $40\%$, requiring ward isolation, deep cleaning, and exclusion of healthcare workers.
            """)

            st.markdown(r"""
            #### 6. The CDC 10-Step Outbreak Framework & ORION Guidelines
            * Field investigations follow the standardized **CDC 10-Step Outbreak Framework**:
              1. Prepare for fieldwork; 2. Establish existence of outbreak; 3. Verify diagnosis; 4. Define and identify cases; 5. Perform descriptive epidemiology; 6. Develop hypotheses; 7. Evaluate hypotheses (Analytical Epidemiology / Attack Rates); 8. Refine hypotheses and execute additional studies; 9. Implement control and prevention measures; 10. Communicate findings.
            * Scientific reporting is governed by the **ORION** statement (Outbreak Reports and Intervention Studies Of Nosocomial infection).
            """)


def render_outbreak_prerequisite_cards(evaluations: Dict[str, Any]):
    """Renders CDC / UKHSA field outbreak prerequisite cards."""
    checks = evaluations.get("checks", [])
    if not checks:
        return

    st.markdown("### 🔍 Field Outbreak & Epidemiological Prerequisite Checks")
    st.caption("Verifying vehicle incrimination distinctiveness, cohort investigation power, incubation concordance, secondary transmission, and curve morphology.")

    cols = st.columns(len(checks))
    for i, check in enumerate(checks):
        with cols[i]:
            status = check.get("status", "pass")
            badge_label = check.get("badge", "")
            title = check.get("title", "")
            finding = check.get("finding", "")
            explanation = check.get("explanation", "")

            with st.container(border=True):
                if status == "pass":
                    st.success(f"**{badge_label}**")
                elif status == "warn":
                    st.warning(f"**{badge_label}**")
                else:
                    st.error(f"**{badge_label}**")

                st.markdown(f"**{title}**")
                st.caption(finding)
                with st.expander("Details", expanded=False):
                    st.write(explanation)


def render_outbreak_metric_dashboard(
    ar_results: Dict[str, Any],
    sar_results: Dict[str, Any],
    curve_results: Dict[str, Any]
):
    """Renders the top-level metric cards for overall attack rate, incriminated item, and transmission."""
    c1, c2, c3, c4, c5 = st.columns(5)

    overall_ar = ar_results.get("overall_attack_rate_pct", 0.0)
    primary_cases = ar_results.get("total_primary_cases", 0)
    cohort_n = ar_results.get("total_cohort_at_risk", 0)
    primary_veh = ar_results.get("primary_incriminated_vehicle", "None")
    details = ar_results.get("incriminated_vehicle_details", {}) or {}
    rr_val = details.get("Risk_Ratio", 1.0)
    rr_ci = details.get("RR_95_CI_Str", "")
    ard_val = details.get("Attack_Rate_Difference_Pct", 0.0)

    med_inc = curve_results.get("median_incubation_hours", 0.0)
    iqr_inc = curve_results.get("iqr_hours", 0.0)
    pattern = curve_results.get("curve_pattern", "Point-Source")

    sar_val = sar_results.get("secondary_attack_rate_pct", 0.0)
    sar_ci = sar_results.get("sar_ci_str", "")
    comm_tier = sar_results.get("communicability_tier", "Low")

    with c1:
        with st.container(border=True):
            st.metric(
                label="Overall Attack Rate",
                value=f"{overall_ar:.1f}%",
                delta=f"{primary_cases} / {cohort_n} Ill"
            )
            st.caption("Cumulative incidence in cohort")

    with c2:
        with st.container(border=True):
            st.metric(
                label="Incriminated Vehicle",
                value=primary_veh[:18] + ("…" if len(primary_veh) > 18 else ""),
                delta=f"ARD: +{ard_val:.1f}%"
            )
            st.caption(f"Full name: {primary_veh}")

    with c3:
        with st.container(border=True):
            st.metric(
                label="Vehicle Relative Risk",
                value=f"{rr_val:.2f}x",
                delta="p < 0.001" if details.get("P_Value", 1.0) < 0.001 else f"p={details.get('P_Value', 1.0):.3f}",
                delta_color="inverse"
            )
            st.caption(f"Woolf 95% CI: [{details.get('RR_CI_Lower', 1.0):.2f}, {details.get('RR_CI_Upper', 1.0):.2f}]")

    with c4:
        with st.container(border=True):
            st.metric(
                label="Median Incubation Period",
                value=f"{med_inc:.1f} hrs",
                delta=f"IQR: {iqr_inc:.1f} hrs"
            )
            st.caption(f"Range: {curve_results.get('min_incubation_hours', 0.0)}–{curve_results.get('max_incubation_hours', 0.0)} hrs")

    with c5:
        with st.container(border=True):
            st.metric(
                label="Secondary Attack Rate (SAR)",
                value=f"{sar_val:.1f}%",
                delta=comm_tier.split()[0],
                delta_color="inverse" if sar_val >= 30.0 else "normal"
            )
            st.caption(f"Wilson 95% CI: [{sar_results.get('sar_ci_lower_pct', 0.0):.1f}%, {sar_results.get('sar_ci_upper_pct', 0.0):.1f}%]")

    # Action / Public Health Takeaway Alert
    if details.get("Is_Suspect_Vehicle", False):
        st.success(
            f"🎯 **VEHICLE INCRIMINATED**: Epidemiological investigation decisively implicates **'{primary_veh}'** "
            f"(Relative Risk = {rr_val:.2f}, 95% CI: [{details.get('RR_CI_Lower', 1.0):.2f}, {details.get('RR_CI_Upper', 1.0):.2f}], "
            f"Attack Rate Difference = +{ard_val:.1f}%, p = {details.get('P_Value', 0.0):.4f}). "
            f"{details.get('Case_Exposure_Coverage_Pct', 0.0):.1f}% of all cases consumed this item. "
            "Immediate recall, supplier batch tracing, and kitchen environmental swab testing recommended."
        )
    else:
        st.warning("⚠️ **INCONCLUSIVE VEHICLE IDENTIFICATION**: No individual food item met all strict incrimination criteria.")


def build_epidemic_curve_plot(curve_results: Dict[str, Any], meta: Dict[str, Any]) -> go.Figure:
    """Builds an interactive Plotly Epidemic Curve (Epi Curve) histogram with annotated milestones."""
    bins_list = curve_results.get("bins_list", [])
    med_inc = curve_results.get("median_incubation_hours", 0.0)

    x_labels = [b["Bin_Label"] for b in bins_list]
    y_counts = [b["Case_Count"] for b in bins_list]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_counts,
        marker=dict(
            color="#2563EB",
            line=dict(color="#1E40AF", width=1.5)
        ),
        text=[str(c) if c > 0 else "" for c in y_counts],
        textposition="outside",
        name="Confirmed Primary Cases"
    ))

    # Add vertical reference line near median bin if possible
    fig.update_layout(
        title=dict(
            text=f"<b>Epidemic Curve (Epi Curve): Case Onset Timeline</b><br><sup>{meta.get('title', 'Outbreak Investigation')} — Pattern: {curve_results.get('curve_pattern', 'Point-Source')} (Median Incubation: {med_inc:.1f}h)</sup>",
            font=dict(size=14)
        ),
        xaxis=dict(
            title="Hours Post-Exposure Event (Onset Interval)",
            gridcolor="#F3F4F6"
        ),
        yaxis=dict(
            title="Number of Confirmed Cases",
            range=[0, max(y_counts + [1]) * 1.3],
            gridcolor="#E5E7EB"
        ),
        height=380,
        margin=dict(l=50, r=40, t=70, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def build_food_incrimination_plot(ar_results: Dict[str, Any], meta: Dict[str, Any]) -> go.Figure:
    """Builds a forest plot comparing Relative Risk across all suspected food items on a log scale."""
    results_list = ar_results.get("results_list", [])
    primary_veh = ar_results.get("primary_incriminated_vehicle", "")

    # Sort so suspect items appear at the top
    sorted_items = sorted(results_list, key=lambda x: x["Risk_Ratio"])

    items = [r["Food_Item"] for r in sorted_items]
    rrs = [r["Risk_Ratio"] for r in sorted_items]
    ci_lows = [r["RR_CI_Lower"] for r in sorted_items]
    ci_upps = [r["RR_CI_Upper"] for r in sorted_items]

    colors = ["#DC2626" if r["Food_Item"] == primary_veh else "#2563EB" for r in sorted_items]

    fig = go.Figure()

    # Null reference line at RR = 1.0
    fig.add_shape(
        type="line",
        x0=1.0, x1=1.0,
        y0=-0.5, y1=len(items) - 0.5,
        line=dict(color="#6B7280", width=1.5, dash="dash")
    )

    # Threshold reference line at RR = 2.0 (suspect threshold)
    fig.add_shape(
        type="line",
        x0=2.0, x1=2.0,
        y0=-0.5, y1=len(items) - 0.5,
        line=dict(color="#F59E0B", width=1.2, dash="dot")
    )

    error_minus = [rrs[i] - ci_lows[i] for i in range(len(rrs))]
    error_plus = [ci_upps[i] - rrs[i] for i in range(len(rrs))]

    fig.add_trace(go.Scatter(
        x=rrs,
        y=items,
        mode="markers+text",
        error_x=dict(
            type="data",
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            color="#374151",
            thickness=2,
            width=6
        ),
        marker=dict(
            size=12,
            color=colors,
            symbol=["diamond" if r["Food_Item"] == primary_veh else "circle" for r in sorted_items]
        ),
        text=[f"  {rrs[i]:.2f}x" for i in range(len(rrs))],
        textposition="top right",
        hoverinfo="text",
        hovertext=[f"<b>{r['Food_Item']}</b><br>RR: {r['Risk_Ratio']:.2f} [{r['RR_CI_Lower']:.2f}, {r['RR_CI_Upper']:.2f}]<br>ARD: +{r['Attack_Rate_Difference_Pct']:.1f}%<br>p-value: {r['P_Value']:.4f}" for r in sorted_items],
        name="Food Items"
    ))

    min_x = max(0.1, min(ci_lows + [1.0]) * 0.7)
    max_x = min(40.0, max(ci_upps + [2.0]) * 1.4)

    fig.update_layout(
        title=dict(
            text="<b>Food-Specific Relative Risk (Incrimination Plot)</b><br><sup>Red Diamond: Primary Incriminated Vehicle | Dotted Yellow Line: Suspect Threshold (RR=2.0)</sup>",
            font=dict(size=14)
        ),
        xaxis=dict(
            title="Relative Risk (Logarithmic Scale)",
            type="log",
            range=[math.log10(min_x), math.log10(max_x)],
            gridcolor="#E5E7EB"
        ),
        yaxis=dict(title="", gridcolor="#F3F4F6"),
        height=380,
        margin=dict(l=200, r=40, t=70, b=50),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def render_outbreak_explanatory_box(
    ar_results: Dict[str, Any],
    curve_results: Dict[str, Any]
):
    """Renders a full-width pedagogical explanatory guide beneath the plots."""
    with st.container(border=True):
        st.markdown("### 📊 How to Interpret the Epidemic Curve & Vehicle Forest Plot")
        st.markdown(r"""
        * **1. Reading the Epidemic Curve (Epi Curve):**
          * The shape of the histogram reveals the **transmission dynamics**:
            * A **sharp single peak** with a rapid upslope and gradual downslope (as seen in point-source food poisoning) proves all patients were exposed during a single shared meal event.
            * The time interval from the meal ($T_0$) to the peak or median of the curve represents the **Median Incubation Period**. A median of $20–24$ hours strongly points to enteric bacterial pathogens like *Salmonella*, whereas $2–6$ hours indicates preformed bacterial toxins (*Staphylococcus aureus* enterotoxin), and $48–72$ hours suggests *Campylobacter*.
        * **2. Reading the Vehicle Incrimination Forest Plot:**
          * Each dot represents the **Relative Risk (RR)** of developing illness after eating that specific food item, compared to those who did not eat it.
          * The horizontal error bar represents the **Woolf 95% Confidence Interval**.
          * The dashed gray line at $\text{RR} = 1.0$ indicates no association. The dotted amber line at $\text{RR} = 2.0$ marks the field epidemiology threshold for a strong suspect.
          * The **red diamond** identifies the single incriminated vehicle. Its entire confidence interval sits far to the right of the null line, confirming that eating this item multiplied the risk of illness many-fold.
        """)


def render_food_matrix_table(ar_results: Dict[str, Any]):
    """Renders the complete 2x2 attack rate incrimination matrix table."""
    st.markdown("### 📋 Food-Specific Attack Rate Incrimination Matrix")
    st.caption("Comparing attack rates among exposed and unexposed event attendees across all evaluated menu items.")

    results_list = ar_results.get("results_list", [])
    primary_veh = ar_results.get("primary_incriminated_vehicle", "")

    display_rows = []
    for r in results_list:
        is_suspect = (r["Food_Item"] == primary_veh)
        status_str = "🚨 INCRIMINATED VEHICLE" if is_suspect else ("🟡 Suspect (RR >= 2.0)" if r["Is_Suspect_Vehicle"] else "🟢 Safe / Unassociated")
        display_rows.append({
            "Food / Exposure Vehicle": r["Food_Item"],
            "Ate Ill (a)": r["Exposed_Ill"],
            "Ate Well (b)": r["Exposed_Well"],
            "Ate Total (n1)": r["Exposed_Total"],
            "Attack Rate Ate (%)": f"{r['Attack_Rate_Exposed_Pct']:.1f}%",
            "Did Not Eat Ill (c)": r["Unexposed_Ill"],
            "Did Not Eat Well (d)": r["Unexposed_Well"],
            "Did Not Eat Total (n0)": r["Unexposed_Total"],
            "Attack Rate Did Not Eat (%)": f"{r['Attack_Rate_Unexposed_Pct']:.1f}%",
            "Attack Rate Difference": f"+{r['Attack_Rate_Difference_Pct']:.1f}%",
            "Relative Risk (95% CI)": r["RR_95_CI_Str"],
            "AF_e (%)": f"{r['AF_Exposed_Pct']:.1f}%",
            "Chi-Square (df=1)": f"{r['Chi2_Statistic']:.2f}",
            "p-value": f"{r['P_Value']:.4f}",
            "Case Coverage (%)": f"{r['Case_Exposure_Coverage_Pct']:.1f}%",
            "Status": status_str
        })

    df_disp = pd.DataFrame(display_rows)
    st.dataframe(df_disp, use_container_width=True, hide_index=True)


def render_outbreak_educational_tabs(
    ar_results: Dict[str, Any],
    sar_results: Dict[str, Any],
    curve_results: Dict[str, Any],
    meta: Dict[str, Any],
    demo_data: Dict[str, Any]
):
    """Renders the comprehensive 4-Tab Educational Architecture."""
    tab_rat, tab_xls, tab_py, tab_dl = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Microsoft Excel Implementation Guide",
        "🐍 Python Production Script Guide",
        "💾 Live Synthetic Data Downloads"
    ])

    # -------------------------------------------------------------
    # TAB 1: Rationale & Epidemiological Framework
    # -------------------------------------------------------------
    with tab_rat:
        st.markdown(r"""
        ### 📋 Field Epidemiology & Outbreak Investigation Framework

        Outbreak investigation is an applied epidemiological science combining rapid descriptive epidemiology,
        analytical hypothesis testing ($2 \times 2$ attack rate matrices), environmental microbiology, and prompt public health control.
        """)

        with st.expander("1. Mathematical Formulation of Food-Specific Attack Rates & Incrimination", expanded=False):
            st.markdown(r"""
            For every suspected food, water, or environmental vehicle $j$:

            | Exposure Status | Developed Illness ($D=1$) | Remained Well ($D=0$) | Total Exposed Margin |
            | :--- | :---: | :---: | :---: |
            | **Ate Vehicle ($E=1$)** | $a_j$ | $b_j$ | $n_{1j} = a_j + b_j$ |
            | **Did Not Eat ($E=0$)** | $c_j$ | $d_j$ | $n_{0j} = c_j + d_j$ |
            | **Total Margin** | $m_{1j} = a_j + c_j$ | $m_{0j} = b_j + d_j$ | $n_j = n_{1j} + n_{0j}$ |

            #### Key Epidemiological Metrics:
            * **Attack Rate in Exposed ($\text{AR}_{\text{exp}}$):**
              $$\text{AR}_{\text{exp}} = \frac{a_j}{n_{1j}} \times 100\%$$
            * **Attack Rate in Unexposed ($\text{AR}_{\text{unexp}}$):**
              $$\text{AR}_{\text{unexp}} = \frac{c_j}{n_{0j}} \times 100\%$$
            * **Attack Rate Difference (Risk Difference, ARD):**
              $$\text{ARD}_j = \text{AR}_{\text{exp}} - \text{AR}_{\text{unexp}}$$
            * **Relative Risk / Risk Ratio ($\text{RR}_j$):**
              $$\text{RR}_j = \frac{\text{AR}_{\text{exp}}}{\text{AR}_{\text{unexp}}} = \frac{a_j / n_{1j}}{c_j / n_{0j}}$$
            * **Woolf Standard Error & 95% Confidence Interval for $\text{RR}$:**
              $$\text{SE}(\ln \text{RR}_j) = \sqrt{\left(\frac{1}{a_j} - \frac{1}{n_{1j}}\right) + \left(\frac{1}{c_j} - \frac{1}{n_{0j}}\right)}$$
              $$\text{CI}_{95\%}(\text{RR}_j) = \exp\left(\ln \text{RR}_j \pm 1.96 \times \text{SE}(\ln \text{RR}_j)\right)$$
            * **Attributable Fraction in the Exposed ($AF_e$):**
              $$AF_e = \frac{\text{RR}_j - 1}{\text{RR}_j} \times 100\% = \frac{\text{AR}_{\text{exp}} - \text{AR}_{\text{unexp}}}{\text{AR}_{\text{exp}}} \times 100\%$$
            """)

        with st.expander("2. Secondary Attack Rate (SAR) & Household Communicability Formulation", expanded=False):
            st.markdown(r"""
            While primary foodborne illness results from common-source ingestion, many enteric pathogens can transmit secondarily via the fecal-oral route, fomites, or respiratory aerosols.

            #### Secondary Attack Rate Formulation:
            $$\text{SAR} = \frac{\text{Number of Secondary Cases Among Contacts}}{\text{Total Susceptible Contacts}} \times 100\%$$
            $$\text{Susceptible Contacts} = N_{\text{contacts}} - N_{\text{immune}} - N_{\text{primary index cases}}$$

            * **Wilson Score 95% Confidence Interval:**
              Because household contact samples are often modest in size, the Wilson score interval provides superior coverage without normal approximation degradation:
              $$p \pm \frac{z}{1 + z^2/n} \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}$$
            * **Clinical Application:**
              * $\text{SAR} = 0\%$: Food intoxications (*B. cereus*, *S. aureus*, *C. botulinum*).
              * $\text{SAR} = 5–15\%$: Modest bacterial secondary spread (*Salmonella*, *E. coli* O157).
              * $\text{SAR} > 30\%$: Highly infectious agents with low infectious doses (*Norovirus*, *Shigella*, *Rotavirus*).
            """)

        with st.expander("3. Epidemic Curve Typologies & Incubation Period Mechanics", expanded=False):
            st.markdown(r"""
            The **Epidemic Curve (Epi Curve)** is a visual representation of the outbreak's time dynamics:

            1. **Point-Source Epidemic:**
               * All susceptible individuals are exposed simultaneously during a single brief time window.
               * Produces a steep, rapid upstroke, a sharp peak, and a gradual right-skewed tail representing individuals with longer incubation periods.
               * The time interval from exposure ($T_0$) to the median case is the **Median Incubation Period**.
            2. **Continuous Common-Source Epidemic:**
               * Exposure to the contaminated source persists over an extended duration (e.g. days or weeks).
               * Cases do not decline rapidly; instead, the curve plateaus until the contaminated source is removed.
            3. **Propagated (Intermittent / Person-to-Person) Epidemic:**
               * Initial exposure leads to secondary and tertiary person-to-person spread.
               * Produces multiple successive peaks, each taller than the preceding one, separated by approximately one incubation period.
            """)

        with st.expander("4. The CDC 10-Step Outbreak Framework & ORION Guidelines", expanded=False):
            st.markdown(r"""
            The **CDC Field Epidemiology 10-Step Framework** provides an orderly approach:
            1. **Prepare for field work** (logistics, personal protective equipment, laboratory kits).
            2. **Establish the existence of an outbreak** (verify cases exceed expected baseline incidence).
            3. **Verify the diagnosis** (clinical symptoms and microbiological confirmation).
            4. **Construct a working case definition** (Confirmed vs. Probable vs. Suspect cases).
            5. **Find cases systematically and record information** (create standardized line listings).
            6. **Perform descriptive epidemiology** (Person, Place, and Time; plot the Epi Curve).
            7. **Develop hypotheses** (source, mode of transmission, vehicles).
            8. **Evaluate hypotheses analytically** (Retrospective Cohort Attack Rate Analysis or Case-Control Odds Ratios).
            9. **Implement control and prevention measures** (food recalls, deep sanitization, hygiene exclusion).
            10. **Communicate findings** (formal written outbreak report complying with **ORION** statement standards).
            """)

    # -------------------------------------------------------------
    # TAB 2: Microsoft Excel Implementation Guide
    # -------------------------------------------------------------
    with tab_xls:
        st.markdown("""
        ### 📗 Step-by-Step Implementation in Microsoft Excel

        This tutorial guides you through building a complete Outbreak Investigation workbook in Microsoft Excel,
        including food-specific attack rate calculations, relative risks, Woolf confidence intervals, and epidemic curve histograms.
        """)

        with st.expander("Step 1: Setting Up the Food-Specific Survey Matrix in Excel", expanded=False):
            st.markdown("""
            1. Open a new Excel worksheet and set up column headers across row 3:
               * `A3`: `Food_Item`
               * `B3`: `Ate_Ill` (Exposed Cases, $a$)
               * `C3`: `Ate_Well` (Exposed Non-Cases, $b$)
               * `D3`: `Ate_Total` (`=B4+C4`, $n_1$)
               * `E3`: `AR_Ate_Pct` (`=(B4/D4)*100`, $AR_{\\text{exp}}$)
               * `F3`: `DidNotEat_Ill` (Unexposed Cases, $c$)
               * `G3`: `DidNotEat_Well` (Unexposed Non-Cases, $d$)
               * `H3`: `DidNotEat_Total` (`=F4+G4`, $n_0$)
               * `I3`: `AR_DidNotEat_Pct` (`=(F4/H4)*100`, $AR_{\\text{unexp}}$)
            2. Enter the observed counts for each menu item in rows 4 through 10.
            """)

        with st.expander("Step 2: Calculating Attack Rate Differences & Risk Ratios", expanded=False):
            st.markdown("""
            1. In column `J4`, calculate the Attack Rate Difference (Risk Difference):
               `=E4 - I4`
            2. In column `K4`, calculate the Relative Risk (Risk Ratio):
               `=IF(I4>0, E4/I4, 1.0)`
            3. In column `L4`, calculate the Attributable Fraction in the Exposed ($AF_e$):
               `=IF(K4>1, (K4-1)/K4*100, 0)`
            4. Drag these formulas down for all evaluated food items.
            """)

        with st.expander("Step 3: Calculating Woolf 95% Confidence Intervals for Relative Risk", expanded=False):
            st.markdown("""
            1. In column `M4`, compute the Woolf standard error of $\\ln(\\text{RR})$:
               `=SQRT((1/B4 - 1/D4) + (1/F4 - 1/H4))`
            2. In column `N4`, compute the 95% Lower Confidence Limit:
               `=EXP(LN(K4) - 1.95996 * M4)`
            3. In column `O4`, compute the 95% Upper Confidence Limit:
               `=EXP(LN(K4) + 1.95996 * M4)`
            """)

        with st.expander("Step 4: Performing Pearson Chi-Square Test in Excel", expanded=False):
            st.markdown("""
            1. In column `P4`, calculate the 1-df Chi-Square test statistic:
               `=((D4+H4) * ((B4*G4 - C4*F4)^2)) / (D4 * H4 * (B4+F4) * (C4+G4))`
            2. In column `Q4`, calculate the two-sided p-value using Excel's chi-square distribution function:
               `=CHISQ.DIST.RT(P4, 1)`
            3. Flag items with `P4 < 0.05` and `K4 >= 2.0` as suspect vehicles!
            """)

        with st.expander("Step 5: Calculating the Secondary Attack Rate (SAR) in Excel", expanded=False):
            st.markdown("""
            1. In a separate summary block:
               * Cell `B14`: Total Primary Cases (e.g. `60`)
               * Cell `B15`: Total Household Contacts (e.g. `75`)
               * Cell `B16`: Secondary Cases Among Contacts (e.g. `5`)
            2. In cell `B17`, compute the Secondary Attack Rate:
               `=(B16/B15)*100`
            3. In cell `B18`, display communicability tier:
               `=IF(B17>30, "HIGH / EXPLOSIVE COMMUNICABILITY", IF(B17>=10, "MODERATE", "LOW / CONTAINED"))`
            """)

        with st.expander("Step 6: Constructing the Epidemic Curve (Epi Curve) Histogram", expanded=False):
            st.markdown("""
            1. Create a table with Onset Time Bins (e.g. 0-4h, 4-8h, 8-12h, 12-16h, 16-20h, etc.) in column `A`, and Case Counts in column `B`.
            2. Select the range and click **Insert** -> **2D Column Chart** (Clustered Column).
            3. Right-click any bar on the chart -> **Format Data Series**.
            4. Set **Gap Width** to `0%` (or `5%`). Epidemiological standards require bars to touch each other to represent a continuous time distribution!
            5. Add a vertical reference line or shape marking the meal exposure time.
            """)

    # -------------------------------------------------------------
    # TAB 3: Python Production Script Guide
    # -------------------------------------------------------------
    with tab_py:
        st.markdown("""
        ### 🐍 Production-Ready Python Script Guide

        This guide presents vectorized, production-grade Python code utilizing standard scientific libraries
        (`numpy`, `pandas`, `scipy.stats`) to execute the complete Outbreak Investigation analytical pipeline.
        """)

        with st.expander("Stage 1: Ingesting Outbreak Survey Microdata & Building 2x2 Matrices", expanded=False):
            st.code(r'''
import numpy as np
import pandas as pd
from scipy import stats

def build_attack_rate_matrix(df_microdata, food_columns, outcome_col='Became_Ill'):
    """Builds 2x2 contingency counts for all food columns from attendee survey microdata."""
    matrix_rows = []
    
    for food in food_columns:
        ate_ill = len(df_microdata[(df_microdata[food] == 1) & (df_microdata[outcome_col] == 1)])
        ate_well = len(df_microdata[(df_microdata[food] == 1) & (df_microdata[outcome_col] == 0)])
        not_ill = len(df_microdata[(df_microdata[food] == 0) & (df_microdata[outcome_col] == 1)])
        not_well = len(df_microdata[(df_microdata[food] == 0) & (df_microdata[outcome_col] == 0)])
        
        matrix_rows.append({
            'Food_Item': food,
            'a': ate_ill, 'b': ate_well,
            'c': not_ill, 'd': not_well
        })
        
    return pd.DataFrame(matrix_rows)
''', language="python")

        with st.expander("Stage 2: Vectorized Calculation of Food-Specific Attack Rates, RRs, and ARDs", expanded=False):
            st.code(r'''
def compute_food_attack_rates(df_matrix):
    """Calculates Attack Rates, Relative Risk, Woolf 95% CIs, and Chi-Square p-values."""
    df = df_matrix.copy()
    
    df['n1'] = df['a'] + df['b']
    df['n0'] = df['c'] + df['d']
    df['n'] = df['n1'] + df['n0']
    
    # Attack rates (%)
    df['AR_Ate'] = (df['a'] / df['n1']) * 100.0
    df['AR_DidNotEat'] = (df['c'] / df['n0']) * 100.0
    df['ARD'] = df['AR_Ate'] - df['AR_DidNotEat']
    
    # Relative Risk
    df['RR'] = (df['a'] / df['n1']) / (df['c'] / df['n0'])
    df['SE_ln_RR'] = np.sqrt((1/df['a'] - 1/df['n1']) + (1/df['c'] - 1/df['n0']))
    df['RR_Lower'] = np.exp(np.log(df['RR']) - 1.96 * df['SE_ln_RR'])
    df['RR_Upper'] = np.exp(np.log(df['RR']) + 1.96 * df['SE_ln_RR'])
    
    # Attributable fraction in exposed (%)
    df['AF_e'] = np.where(df['RR'] > 1.0, ((df['RR'] - 1.0) / df['RR']) * 100.0, 0.0)
    
    # Chi-Square test
    diff = np.abs(df['a'] * df['d'] - df['b'] * df['c'])
    denom = df['n1'] * df['n0'] * (df['a'] + df['c']) * (df['b'] + df['d'])
    df['Chi2'] = (df['n'] * (diff ** 2)) / denom
    df['P_Value'] = stats.chi2.sf(df['Chi2'], df=1)
    
    return df
''', language="python")

        with st.expander("Stage 3: Secondary Attack Rate (SAR) & Wilson Confidence Interval", expanded=False):
            st.code(r'''
def compute_secondary_attack_rate(secondary_cases, household_contacts, alpha=0.05):
    """Calculates Secondary Attack Rate and Wilson score 95% confidence interval."""
    sar = (secondary_cases / float(household_contacts)) * 100.0
    
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p_hat = secondary_cases / float(household_contacts)
    n = household_contacts
    
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / denom
    
    ci_lower = max(0.0, center - margin) * 100.0
    ci_upper = min(1.0, center + margin) * 100.0
    
    return sar, ci_lower, ci_upper
''', language="python")

        with st.expander("Stage 4: Epidemic Curve Onset Binning & Incubation Distribution Statistics", expanded=False):
            st.code(r'''
def compute_epidemic_curve(onset_hours, bin_width=4.0):
    """Computes incubation period statistics and bins cases for Epi Curve plotting."""
    inc_array = np.array(sorted(onset_hours))
    
    min_inc = np.min(inc_array)
    q1 = np.percentile(inc_array, 25)
    median_inc = np.median(inc_array)
    q3 = np.percentile(inc_array, 75)
    max_inc = np.max(inc_array)
    iqr = q3 - q1
    
    # Create histogram bins
    bins = np.arange(np.floor(min_inc / bin_width) * bin_width, np.ceil(max_inc / bin_width) * bin_width + bin_width, bin_width)
    counts, edges = np.histogram(inc_array, bins=bins)
    
    df_curve = pd.DataFrame({
        'Bin_Start': edges[:-1],
        'Bin_End': edges[1:],
        'Case_Count': counts
    })
    
    return {
        'median_incubation': median_inc,
        'iqr': iqr,
        'range': (min_inc, max_inc),
        'curve_df': df_curve
    }
''', language="python")

        with st.expander("Stage 5: Complete Standalone End-to-End Production Outbreak Investigation Script", expanded=False):
            st.code(r'''
# Complete Outbreak Investigation & Attack Rate Pipeline
from src.epi_methods.outbreak_investigation import (
    calculate_food_specific_attack_rates,
    calculate_secondary_attack_rate,
    calculate_epidemic_curve,
    generate_outbreak_demo_data
)

# 1. Load Synthetic Field Outbreak Dataset
demo = generate_outbreak_demo_data("wedding_salmonella")
print(f"Investigating: {demo['scenario_name']}")

# 2. Food-Specific Attack Rates
ar_res = calculate_food_specific_attack_rates(demo['food_matrix'])
incriminated = ar_res['primary_incriminated_vehicle']
details = ar_res['incriminated_vehicle_details']

# 3. Secondary Transmission & Epi Curve
sar_res = calculate_secondary_attack_rate(demo['primary_cases'], demo['household_contacts'], demo['secondary_cases'])
curve_res = calculate_epidemic_curve(demo['onset_hours'], demo['exposure_hour'], demo['bin_width'])

# 4. Print Summary Report
print("=" * 60)
print(f"Overall Attack Rate:        {ar_res['overall_attack_rate_pct']:.1f}%")
print(f"Incriminated Vehicle:       {incriminated}")
print(f"Vehicle Relative Risk:      {details['Risk_Ratio']:.2f} {details['RR_95_CI_Str']}")
print(f"Attack Rate Difference:     +{details['Attack_Rate_Difference_Pct']:.1f}% (p = {details['P_Value']:.4f})")
print(f"Median Incubation Period:   {curve_res['median_incubation_hours']:.1f} hours (IQR: {curve_res['iqr_hours']:.1f}h)")
print(f"Outbreak Pattern:           {curve_res['curve_pattern']}")
print(f"Secondary Attack Rate:      {sar_res['secondary_attack_rate_pct']:.1f}% [{sar_res['sar_ci_lower_pct']}%, {sar_res['sar_ci_upper_pct']}%]")
print(f"Communicability Tier:       {sar_res['communicability_tier']}")
print("=" * 60)
''', language="python")

    # -------------------------------------------------------------
    # TAB 4: Live Synthetic Data Downloads
    # -------------------------------------------------------------
    with tab_dl:
        st.markdown(r"""
        ### 💾 Live Synthetic Demonstration Data Downloads

        All datasets provided below are **100% synthetic, computer-generated demonstration files** created for teaching
        and code rehearsal. They contain **zero real patient data or protected health information**.
        """)

        st.warning("""
        ⚠️ **EDUCATIONAL DISCLAIMER & NOTICE**:
        * **100% Synthetic / Artificial Data:** All attendee IDs, food selections, symptom onsets, and household contacts were synthesized via deterministic mathematical simulation.
        * **Educational Rehearsal Only:** Designed for public health training, field epidemiology workshops, and practicing attack rate calculations in Excel, R, Python, Stata, or SAS.
        * **No Clinical Utility:** Must never be used for real patient treatment decisions or commercial catering enforcement.
        """)

        c_d1, c_d2 = st.columns(2)

        df_micro = demo_data.get("df", pd.DataFrame())
        if not isinstance(df_micro, pd.DataFrame):
            df_micro = pd.DataFrame(df_micro)

        with c_d1:
            with st.container(border=True):
                st.markdown(f"#### 📄 Synthetic Attendee Microdata ({len(df_micro)} Simulated Records)")
                st.markdown(f"Contains **{len(df_micro)} individual attendee records** including simulated attendee IDs, food item consumption flags, confirmed illness status, and onset hours post-exposure.")
                st.caption("*100% synthetic / computer-generated cohort. Contains zero real patient data.*")
                csv_micro = df_micro.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇️ Download Synthetic Attendee Microdata ({len(df_micro)} Rows CSV)",
                    data=csv_micro,
                    file_name=f"synthetic_{meta['scenario_key']}_attendee_microdata.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                with st.expander("👁️ Preview First 5 Rows of Synthetic Microdata", expanded=False):
                    st.dataframe(df_micro.head(5), use_container_width=True, hide_index=True)

        with c_d2:
            with st.container(border=True):
                st.markdown("#### 📊 Food-Specific Attack Rate Matrix Summary (CSV)")
                st.markdown("Aggregated summary of evaluated food items, exposed/unexposed attack rates, attack rate differences, relative risks with 95% CIs, and chi-square statistics.")
                st.caption("*Derived from the simulated synthetic cohort for tutorial exercises.*")
                df_matrix = pd.DataFrame(ar_results.get("results_list", []))
                csv_matrix = df_matrix.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Food Attack Rate Matrix CSV",
                    data=csv_matrix,
                    file_name=f"synthetic_{meta['scenario_key']}_food_attack_rates.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                with st.expander("👁️ Preview Food Attack Rate Matrix Table", expanded=False):
                    st.dataframe(df_matrix, use_container_width=True, hide_index=True)

        with st.expander("📋 Synthetic Data Generation Methodology & Variable Dictionary", expanded=False):
            st.markdown(r"""
            #### How This Synthetic Dataset Was Created
            1. **Cohort & Menu Simulation:** Simulated banquet attendees are assigned binary consumption flags across multiple menu items based on realistic event dining preferences.
            2. **Contamination Mechanism:** A single food vehicle (e.g. Hollandaise sauce made with raw eggs) is designated as the contaminated source with a high primary transmission probability ($\sim 85\%$).
            3. **Onset Timing Simulation:** Symptom onset times for ill cases are sampled from log-normal incubation distributions tuned to specific biological pathogens (e.g. mean $22\text{h}$ for *Salmonella*, $30\text{h}$ for *Norovirus*, $72\text{h}$ for *Campylobacter*).
            4. **Secondary Transmission:** Close household contacts are exposed to primary cases with fixed transmission probabilities reflecting real-world pathogen communicability.
            5. **Information Governance:** Because this dataset is entirely synthetic, it is safe to share, publish, and use in educational workshops without information governance, Caldicott Guardian, or GDPR restrictions.
            """)


def render_outbreak_module(active_method_key: str, method_info: Dict[str, Any]):
    """Main rendering controller for Module 8: Outbreak Epidemiology, Attack Rates & Vehicle Incrimination."""

    # 1. Plain English Pedagogical Introductory Guide
    render_outbreak_plain_english_guide()

    # 2. Sidebar Scenario Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Outbreak Investigation Controls")

    scenario_options = {
        "wedding_salmonella": "Riverside Health Protection Team (Salmonella Wedding Banquet) [Synthetic]",
        "hospital_norovirus": "Inner City Teaching Hospital (Ward 4B Norovirus Outbreak) [Synthetic]",
        "community_campylobacter": "All Meadows Community Care Trust (Charity BBQ Campylobacter) [Synthetic]",
        "custom": "Custom User-Defined Food Attack Rate Matrix [Synthetic]"
    }

    selected_scenario = st.sidebar.selectbox(
        "Select Outbreak Investigation Scenario:",
        options=list(scenario_options.keys()),
        format_func=lambda k: scenario_options[k]
    )
    st.sidebar.caption("⚠️ All clinical scenarios utilize purely synthetic datasets simulated for educational demonstration.")

    if selected_scenario != "custom":
        demo_data = generate_outbreak_demo_data(scenario_key=selected_scenario)
        food_matrix_input = demo_data["food_matrix"]
        onset_hours_input = demo_data["onset_hours"]
        exp_hour = demo_data["exposure_hour"]
        bin_w = demo_data["bin_width"]
        prim_cases = demo_data["primary_cases"]
        hh_contacts = demo_data["household_contacts"]
        sec_cases = demo_data["secondary_cases"]

        meta = {
            "scenario_key": selected_scenario,
            "title": demo_data["scenario_name"],
            "organisation": demo_data["organisation_name"],
            "domain": demo_data["domain"],
            "pathogen": demo_data["pathogen"],
            "incriminated_item": demo_data["incriminated_item"],
            "description": demo_data["description"]
        }
    else:
        st.sidebar.markdown("#### ⚙️ Custom Outbreak Matrix Parameters")
        num_items = st.sidebar.slider("Number of Food / Exposure Items", min_value=2, max_value=6, value=3, step=1)
        food_matrix_input = []
        for i in range(num_items):
            st.sidebar.markdown(f"**Item {i+1} Cells:**")
            col_a, col_b = st.sidebar.columns(2)
            with col_a:
                a_val = st.number_input(f"Ate Ill (a_{i+1})", min_value=0, max_value=500, value=40 if i==0 else 15, key=f"ob_a_{i}")
                c_val = st.number_input(f"Did Not Eat Ill (c_{i+1})", min_value=0, max_value=500, value=5 if i==0 else 30, key=f"ob_c_{i}")
            with col_b:
                b_val = st.number_input(f"Ate Well (b_{i+1})", min_value=0, max_value=500, value=10 if i==0 else 35, key=f"ob_b_{i}")
                d_val = st.number_input(f"Did Not Eat Well (d_{i+1})", min_value=0, max_value=500, value=45 if i==0 else 20, key=f"ob_d_{i}")
            food_matrix_input.append({
                "Food_Item": f"Custom Item {i+1}",
                "Exposed_Ill": a_val,
                "Exposed_Well": b_val,
                "Unexposed_Ill": c_val,
                "Unexposed_Well": d_val
            })

        st.sidebar.markdown("#### 👥 Secondary Transmission Parameters")
        prim_cases = st.sidebar.number_input("Primary Cases", min_value=1, max_value=500, value=45)
        hh_contacts = st.sidebar.number_input("Household Contacts", min_value=1, max_value=500, value=60)
        sec_cases = st.sidebar.number_input("Secondary Cases", min_value=0, max_value=500, value=6)

        onset_hours_input = [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 32.0, 36.0, 42.0]
        exp_hour = 0.0
        bin_w = 4.0

        demo_data = {
            "df": pd.DataFrame({"Custom_Data": [1]}),
            "food_matrix": food_matrix_input,
            "onset_hours": onset_hours_input,
            "exposure_hour": exp_hour,
            "bin_width": bin_w,
            "primary_cases": prim_cases,
            "household_contacts": hh_contacts,
            "secondary_cases": sec_cases
        }
        meta = {
            "scenario_key": "custom",
            "title": "Custom User-Defined Outbreak Simulation",
            "organisation": "Riverside Health Protection Team (Environmental Health)",
            "domain": "Custom Field Outbreak Epidemiology",
            "pathogen": "Suspected Enteric Pathogen",
            "incriminated_item": "User Defined",
            "description": "User-configured field outbreak investigation."
        }

    # 3. Core Calculations
    ar_results = calculate_food_specific_attack_rates(food_matrix_input)
    sar_results = calculate_secondary_attack_rate(prim_cases, hh_contacts, sec_cases)
    curve_results = calculate_epidemic_curve(onset_hours_input, exposure_hour=exp_hour, bin_width=bin_w)
    evaluations = evaluate_outbreak_prerequisites({
        **ar_results,
        "sar_results": sar_results,
        "curve_results": curve_results
    })

    # 4. Header Banner & Prominent Synthetic Disclaimer
    st.markdown(f"## 🦠 {meta['title']}")
    st.markdown(f"**Organisation:** {meta['organisation']} | **Domain:** {meta['domain']}")
    st.markdown(f"**Suspected Pathogen:** `{meta['pathogen']}` | **Primary Suspect:** `{meta.get('incriminated_item', 'Under Investigation')}`")
    st.markdown(f"*{meta['description']}*")

    with st.container(border=True):
        st.markdown(r"""
        > [!IMPORTANT]
        > **EDUCATIONAL & SYNTHETIC DATA DISCLAIMER**: The patient records, food exposure tables, and outbreak metrics displayed below are **100% synthetic and computer-generated** purely for academic training, field epidemiology education, and software demonstration. Contains no real patient identifiers or protected health information.
        """)

    # 5. Methodological Prerequisite Cards
    render_outbreak_prerequisite_cards(evaluations)

    # 6. Primary Metric Dashboard
    st.markdown("### 📈 Primary Outbreak Metrics & Attack Rates")
    render_outbreak_metric_dashboard(ar_results, sar_results, curve_results)

    # 7. Interactive Visualizations
    col_v1, col_v2 = st.columns([1, 1])
    with col_v1:
        with st.container(border=True):
            fig_curve = build_epidemic_curve_plot(curve_results, meta)
            st.plotly_chart(fig_curve, use_container_width=True)

    with col_v2:
        with st.container(border=True):
            fig_forest = build_food_incrimination_plot(ar_results, meta)
            st.plotly_chart(fig_forest, use_container_width=True)

    # Explanatory Guide Beneath Plots
    render_outbreak_explanatory_box(ar_results, curve_results)

    # 8. Detailed Food Attack Rate Table
    render_food_matrix_table(ar_results)

    # 9. Full 4-Tab Educational Architecture
    st.markdown("---")
    st.markdown("### 🎓 Educational Guides & Reference Materials")
    render_outbreak_educational_tabs(ar_results, sar_results, curve_results, meta, demo_data)


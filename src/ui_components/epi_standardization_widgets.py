"""Streamlit UI Components & Visualizations for Demographic Rate Standardization (DSR & SMR).

Provides:
- render_rate_standardization_module: Full interactive spatial & demographic standardization suite.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.epi_methods.derbyshire_data import (
    generate_derbyshire_standardization_data,
    get_derbyshire_geojson_path,
    get_esp_2013_standard,
    get_excel_aligned_dataset,
    DISTRICT_PROFILES,
    ESP_2013_BANDS,
    SCENARIO_CONFIGS
)

from src.epi_methods.standardization import (
    evaluate_standardization_prerequisites,
    calculate_direct_standardized_rate,
    calculate_indirect_standardization,
    calculate_dobson_dsr_ci,
    calculate_byar_poisson_ci
)

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


def render_rate_standardization_module(active_method_key: str, method_info: dict):
    """Renders the interactive rate standardization module (DSR, SMR, Folium map, and pyramids)."""
    is_indirect = (active_method_key == "indirect_standardization")

    # ----------------- Plain English Core Concepts Explainer -----------------
    with st.expander("💡 Plain English Guide: Direct Standardization (DSR) vs. Indirect Standardization (SMR)", expanded=False):
        st.markdown("""
        Demographic standardization answers two distinct epidemiological questions depending on whether you standardize **directly** or **indirectly**:
        """)
        col_pe1, col_pe2 = st.columns(2)
        with col_pe1:
            st.markdown("""
            #### 🟢 Direct Age-Standardization (DSR)
            * **The Core Question:** *"What would our local death rate be if our population had the STANDARD age distribution?"*
            * **How it Works:** Takes local age-specific mortality rates ($r_i = d_i / n_i$) and applies them to an external Standard Population (e.g. European Standard Population 2013 weights $w_i$).
            * **Output Metric:** Absolute standardized rate per 1,000 (e.g. $7.85$ per 1,000).
            * **Key Advantage:** **Cross-area comparability**. DSRs from different districts (e.g., Derby City vs. Derbyshire Dales) can be directly compared against one another because they both apply the exact same population weights.
            * **When to Use:** When local event counts are sufficiently large and reliable across every age stratum.
            * **Prerequisite:** Denominators must be positive and death counts cannot be zero or excessively sparse.
            """)
        with col_pe2:
            st.markdown("""
            #### 🔵 Indirect Standardization (SMR)
            * **The Core Question:** *"How many deaths did we actually observe compared to what we would EXPECT if our population experienced the benchmark rates?"*
            * **How it Works:** Takes external standard benchmark rates ($R_{std, i}$) and applies them to the local population structure to calculate Expected Deaths ($E$). Then takes the ratio: $\\text{SMR} = O / E$.
            * **Output Metric:** Relative ratio or percentage (e.g., $\\text{SMR} = 0.88$ or $88.2\\%$, where $100\\%$ is parity with benchmark).
            * **Key Advantage:** **Statistical stability with small numbers**. Works reliably even when local event counts are small, rare, or zero in younger brackets, because it only requires total observed deaths ($O$) and borrows statistical power from the large benchmark reference rates.
            * **Critical Trade-off:** SMRs from two different districts **cannot strictly be compared directly to each other** if their age distributions differ markedly; each is compared against the common benchmark.
            """)
        st.info("""
        💡 **Converting SMR to an Indirectly Standardized Rate (ISR):**  
        $$\\text{Indirect Standardized Rate} = \\text{SMR} \\times (\\text{Crude Rate of the Reference Population})$$  
        This expresses the relative SMR as a pseudo-rate per 1,000 population while retaining the stability of indirect calculation.
        """)


    # ----------------- Sidebar Controls for Rate Standardization -----------------
    st.sidebar.header("🧪 Epidemiological Case Study Settings")
    st.sidebar.markdown("**Derby City & Derbyshire Local Authorities:**")

    selected_scenario = st.sidebar.selectbox(
        "Health Scenario:",
        options=["circulatory", "cancer", "all_cause"],
        format_func=lambda s: {
            "circulatory": "❤️ Circulatory & Cardiovascular Disease (CVD)",
            "cancer": "🎗️ All-Malignant Neoplasms (Cancer)",
            "all_cause": "📈 All-Cause Mortality"
        }[s],
        index=0,
        help="Select the epidemiological disease category to observe how age profiles affect crude vs. standardized rankings."
    )

    focal_district = st.sidebar.selectbox(
        "Focal District to Inspect:",
        options=[p["name"] for p in DISTRICT_PROFILES],
        index=1,  # Derbyshire Dales default
        help="Select an authority to inspect its demographic pyramid and age-specific rate breakdown."
    )

    st.sidebar.caption(f"💡 *Scenario Info:* {SCENARIO_CONFIGS[selected_scenario]['description']}")
    st.sidebar.caption("⚠️ *Data Notice:* Local authority populations reflect authentic census demographics; disease event counts are synthetically simulated for educational demonstration.")


    # ----------------- Generate Case Study Data -----------------
    df_detailed, df_summary, meta = generate_derbyshire_standardization_data(scenario=selected_scenario, seed=42)
    df_excel_focal = get_excel_aligned_dataset(df_detailed, focal_district=focal_district)

    # ----------------- Methodology Tabs -----------------
    tab_rationale, tab_excel, tab_code, tab_download = st.tabs([
        "📋 Rationale & Epidemiological Framework",
        "📗 Excel Implementation Guide",
        "🐍 Python Code Implementation",
        "📥 Download Sample Data"
    ])

    with tab_rationale:
        st.subheader(f"Why Demographic Rate Standardization? ({method_info['name']})")
        st.markdown("""
        When comparing disease occurrence or mortality across geographic areas, healthcare systems, or time periods, 
        **crude rates can be deeply misleading**. 

        Because the risk of disease and death rises steeply with age, any population with an older age structure 
        will inevitably record more deaths per 1,000 people—even if its healthcare services, living standards, 
        and age-specific survival rates are exceptionally high.
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
        st.markdown("### 🌐 Selecting Reference Populations: Standards, Benchmarks & Trade-Offs")
        st.markdown("""
        When performing standardization, the choice of the **Reference Population** directly dictates what your results mean, 
        who you can compare them with, and how statistically stable they are.
        """)

        st.markdown("""
        | Reference Type | Example | Used In | Key Advantages | Trade-Offs & Limitations |
        | :--- | :--- | :--- | :--- | :--- |
        | **External Theoretical Standard** | European Standard Population (ESP 2013), WHO World Standard | Direct Standardization (DSR) | • Fixed, unchanging weights over decades.<br>• Enables comparisons across nations, counties, and historical years. | • Theoretical distribution may not match local demographics.<br>• Arbitrary weights (e.g. ESP 2013 has higher elderly weights than ESP 1976). |
        | **External National Benchmark Rates** | England & Wales national age-specific rates (ONS / Fingertips) | Indirect Standardization (SMR) | • Represents the true national health benchmark.<br>• Stable rates derived from millions of people. | • SMRs between two local districts cannot strictly be compared if their age profiles differ.<br>• Reflects national socio-economic conditions rather than local context. |
        | **Internal Pooled Benchmark (Combined Local Districts)** | Aggregating population and deaths across all 9 Derbyshire authorities ($R_{\\text{pooled}, i} = \\sum d_{i,j} / \\sum n_{i,j}$) | Indirect (SMR) & Direct (DSR) | • **Locally calibrated:** Represents the collective Derbyshire health average.<br>• Summary SMR across the county is exactly $1.00$ ($100\\%$), immediately showing which areas are above/below average.<br>• No dependency on external data tables. | • Cannot be compared to outside regions (e.g. Nottinghamshire or UK national averages).<br>• **Dominated by largest district:** Derby City (~264k) represents 25% of the total population, pulling the pooled average toward its urban profile. |
        """)

        with st.expander("🔎 Deep-Dive: Sub-Population & Subsection Analyses (e.g., 5–9 years or 18–25 years only)", expanded=False):
            st.markdown("""
            When public health investigations focus on a specific sub-population (e.g. pediatric emergency admissions in children aged 5–9, 
            or university-age mental health presentations in young adults aged 18–25), how should standardization be handled?

            ##### 💡 Clarifying the Epidemiological Assumptions:
            1. **Restriction as Confounding Control:**
               * Restricting your study to a narrow age band (e.g., 5–9 years) is itself a fundamental epidemiological design technique called **Restriction**.
               * Restriction automatically eliminates confounding from the rest of the lifespan (e.g., elderly chronic disease patterns).
            2. **Why an Internal Pooled Standard is Preferred for Narrow Age Subsections:**
               * **The Sample Size Problem (Sparsity):** In small rural districts (like Derbyshire Dales or High Peak), health events within a narrow 5-year bracket are often rare (0, 1, or 2 events per year). Calculating an empirical stratum rate locally yields extreme random noise or $0.0$.
               * **Lack of Subsection Weights:** Lifespan standards (like ESP 2013) define weights across all ages from 0 to 85+. They do not have pre-set weights for analyzing *only* 5–9 year olds.
               * **The Power of Pooling:** By pooling population and deaths across **all 9 districts for that specific age window** (e.g., all 5–9 year olds across Derbyshire), you calculate a stable, high-powered local benchmark rate:
                 $$R_{\\text{pooled, 5–9}} = \\frac{\\sum_{\\text{districts}} \\text{Deaths}_{5–9}}{\\sum_{\\text{districts}} \\text{Population}_{5–9}}$$
               * Each district's observed count in that age slice can then be compared against its expected count ($E_j = n_{j, 5–9} \\times R_{\\text{pooled, 5–9}}$), yielding a statistically robust Indirect SMR for that sub-population.
            3. **When Does Age Standardization Still Apply Within a Subsection?**
               * If the age window contains multiple granular sub-strata (e.g. youth aged 15–24 divided into 15–17 school-age and 18–24 working/university adults), internal age confounding can still occur if Derby City has a much higher share of 18–21 year olds than rural districts.
               * In that case, you can either:
                 * **Re-normalize external weights:** Filter ESP weights for 15–24 and scale them so they sum to $1.0$.
                 * **Use the pooled internal age distribution:** Use the combined Derbyshire population distribution across those sub-brackets as the reference weights.
            """)
    with tab_excel:
        if is_indirect:
            st.markdown("### 📗 Microsoft Excel Implementation Workflow for Indirect Rate Standardization (SMR)")
            st.markdown(f"Follow the 3-step sequence for **{focal_district}**: **Set up Data & Layout** ➔ **Calculate Indirect SMR (Primary Method)** ➔ **Calculate Direct DSR (Comparative Benchmark)**.")
        else:
            st.markdown("### 📗 Microsoft Excel Implementation Workflow for Direct Rate Standardization (DSR)")
            st.markdown(f"Follow the 3-step sequence for **{focal_district}**: **Set up Data & Layout** ➔ **Calculate Direct DSR (Primary Method)** ➔ **Calculate Indirect SMR (Comparative Benchmark)**.")

        with st.expander("📌 Step 1: Excel Data Layout & Cell References", expanded=False):
            st.markdown(f"""
            Structure your Excel worksheet with the following 6 age bands (`Rows 2 to 7`) using the data for **{focal_district}**:
            * **Column A (Cells A2:A7):** Age Band (`0-14`, `15-44`, `45-64`, `65-74`, `75-84`, `85+`)
            * **Column B (Cells B2:B7):** Local Stratum Population count $n_i$
            * **Column C (Cells C2:C7):** Local Observed Deaths $d_i$
            * **Column D (Cells D2:D7):** Local Age-Specific Rate per 1,000 `=C2/B2*1000`
            * **Column E (Cells E2:E7):** European Standard Population Weight $w_i$ (ESP 2013, summing to 1.0)
            * **Column F (Cells F2:F7):** Standard Reference Rate per 1,000 $R_{{std, i}}$
            """)

            tot_pop = int(df_excel_focal["Col_B_Population"].sum())
            tot_deaths = int(df_excel_focal["Col_C_Observed_Deaths"].sum())
            crude_rate = (tot_deaths / tot_pop) * 1000.0 if tot_pop > 0 else 0.0
            tot_weights = float(df_excel_focal["Col_E_ESP_Weight"].sum())

            preview_rows = []
            for idx, r in df_excel_focal.iterrows():
                preview_rows.append({
                    "Excel_Row": f"Row {idx + 2}",
                    "Col_A_Age_Band": r["Col_A_Age_Band"],
                    "Col_B_Population": f"{int(r['Col_B_Population']):,}",
                    "Col_C_Observed_Deaths": int(r["Col_C_Observed_Deaths"]),
                    "Col_D_Rate_per_1k": f"{r['Col_D_Rate_per_1k']:.2f}",
                    "Col_E_ESP_Weight": f"{r['Col_E_ESP_Weight']:.3f}",
                    "Col_F_Reference_Rate_per_1k": f"{r['Col_F_Reference_Rate_per_1k']:.2f}"
                })
            preview_rows.append({
                "Excel_Row": "Row 8 (Totals / Crude)",
                "Col_A_Age_Band": "Total / Summary",
                "Col_B_Population": f"{tot_pop:,}  (=SUM(B2:B7))",
                "Col_C_Observed_Deaths": f"{tot_deaths}  (=SUM(C2:C7))",
                "Col_D_Rate_per_1k": f"{crude_rate:.2f}  (=C8/B8*1000)",
                "Col_E_ESP_Weight": f"{tot_weights:.3f}  (=SUM(E2:E7))",
                "Col_F_Reference_Rate_per_1k": "—"
            })

            st.markdown(f"**Worksheet Grid Representation for `{focal_district}` (`Cells A1:F8`):**")
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

            st.markdown("**Formula Mapping & Summary Output Cells:**")
            if is_indirect:
                df_excel_layout = pd.DataFrame({
                    "Cell": ["B8", "C8", "D8", "D14", "D15", "D16", "D17", "D18", "D20", "D21"],
                    "Metric": [
                        "Total Local Population", "Total Observed Deaths", "Crude Death Rate per 1,000",
                        "Expected Deaths (Indirect E)", "Standardized Mortality Ratio (SMR)", "SMR Percentage (%)",
                        "Byar's SMR 95% CI Lower", "Byar's SMR 95% CI Upper",
                        "Direct Standardized Rate (DSR)", "DSR 95% Confidence Interval"
                    ],
                    "Excel_Formula": [
                        "=SUM(B2:B7)", "=SUM(C2:C7)", "=C8/B8*1000",
                        "=SUMPRODUCT(B2:B7, F2:F7)/1000",
                        "=C8/D14",
                        "=D15*100",
                        "=(C8/D14)*(1 - 1/(9*C8) - 1.96/(3*SQRT(C8)))^3",
                        "=((C8+1)/D14)*(1 - 1/(9*(C8+1)) + 1.96/(3*SQRT(C8+1)))^3",
                        "=SUMPRODUCT(D2:D7, E2:E7)",
                        "=D20 & ' (95% CI: ' & ROUND(D20-1.96*D22, 2) & ' - ' & ROUND(D20+1.96*D22, 2) & ')'"
                    ]
                })
            else:
                df_excel_layout = pd.DataFrame({
                    "Cell": ["B8", "C8", "D8", "E8", "D10", "D11", "D12", "D14", "D15"],
                    "Metric": [
                        "Total Local Population", "Total Observed Deaths", "Crude Death Rate per 1,000",
                        "Total Standard Weights", "Direct Standardized Rate (DSR)", "Standard Error SE(DSR)",
                        "DSR 95% Confidence Interval", "Expected Deaths (Indirect)", "Standardized Mortality Ratio (SMR)"
                    ],
                    "Excel_Formula": [
                        "=SUM(B2:B7)", "=SUM(C2:C7)", "=C8/B8*1000",
                        "=SUM(E2:E7)", "=SUMPRODUCT(D2:D7, E2:E7)",
                        "=SQRT(SUMPRODUCT((E2:E7^2)*(C2:C7/(B2:B7^2))))*1000",
                        "=D10 & ' (95% CI: ' & ROUND(D10-1.96*D11, 2) & ' - ' & ROUND(D10+1.96*D11, 2) & ')'",
                        "=SUMPRODUCT(B2:B7, F2:F7)/1000",
                        "=C8/D14"
                    ]
                })
            st.dataframe(df_excel_layout, use_container_width=True, hide_index=True)

        if is_indirect:
            with st.expander("📌 Step 2: Implement Indirect Standardization (SMR & Expected Deaths) in Excel", expanded=False):
                st.markdown("""
                1. **Compute Expected Deaths in Cell D14:** Enter `=SUMPRODUCT(B2:B7, F2:F7)/1000`. This multiplies each local age population by the standard reference rate and sums across strata.
                2. **Compute SMR Point Estimate in Cell D15:** Enter `=C8/D14` (Observed Deaths / Expected Deaths).
                3. **Compute SMR Percentage in Cell D16:** Enter `=D15*100` (e.g., $100\\%$ is parity with benchmark; $>100\\%$ indicates excess mortality).
                4. **Compute Indirectly Standardized Rate (ISR) in Cell D17:** Enter `=D15 * F8` (where F8 is the crude benchmark rate) to scale the ratio into an absolute rate per 1,000.
                5. **Compute Byar's Exact Poisson 95% Confidence Intervals:**
                   * Lower SMR bound in cell `D18`: `=(C8/D14)*(1 - 1/(9*C8) - 1.96/(3*SQRT(C8)))^3`
                   * Upper SMR bound in cell `D19`: `=((C8+1)/D14)*(1 - 1/(9*(C8+1)) + 1.96/(3*SQRT(C8+1)))^3`
                """)
            with st.expander("📌 Step 3: Implement Direct Standardization (DSR) in Excel (Comparative Benchmark)", expanded=False):
                st.markdown("""
                1. **Calculate Age-Specific Rates in Column D:** Enter `=C2/B2*1000` in cell `D2` and drag down to row 7 (`D2:D7`).
                2. **Enter European Standard Population (ESP 2013) Weights in Column E:** Enter `0.150`, `0.380`, `0.260`, `0.110`, `0.070`, `0.030`.
                3. **Compute DSR:** In cell `D20`, enter `=SUMPRODUCT(D2:D7, E2:E7)`.
                4. **Compute Dobson / Normal 95% Confidence Interval:**
                   * Standard Error in cell `D21`: `=SQRT(SUMPRODUCT((E2:E7^2)*(C2:C7/(B2:B7^2))))*1000`
                   * 95% Bounds: `=D20 - 1.96 * D21` and `=D20 + 1.96 * D21`
                """)
        else:
            with st.expander("📌 Step 2: Implement Direct Standardization (DSR) in Excel", expanded=False):
                st.markdown("""
                1. **Calculate Age-Specific Rates in Column D:** Enter `=C2/B2*1000` in cell `D2` and drag down to row 7 (`D2:D7`).
                2. **Enter European Standard Population (ESP 2013) Weights in Column E:**
                   * `0-14`: `0.150` | `15-44`: `0.380` | `45-64`: `0.260` | `65-74`: `0.110` | `75-84`: `0.070` | `85+`: `0.030`.
                3. **Compute DSR:** In cell `D10`, enter `=SUMPRODUCT(D2:D7, E2:E7)`.
                4. **Compute Dobson / Normal 95% Confidence Interval:**
                   * Standard Error in cell `D11`: `=SQRT(SUMPRODUCT((E2:E7^2)*(C2:C7/(B2:B7^2))))*1000`
                   * Lower 95% Bound in cell `D12_Lower`: `=D10 - 1.96 * D11`
                   * Upper 95% Bound in cell `D12_Upper`: `=D10 + 1.96 * D11`
                """)

            with st.expander("📌 Step 3: Implement Indirect Standardization (SMR) in Excel", expanded=False):
                st.markdown("""
                1. **Compute Expected Deaths:** In cell `D14`, enter `=SUMPRODUCT(B2:B7, F2:F7)/1000`.
                2. **Compute SMR Point Estimate:** In cell `D15`, enter `=C8/D14` (or `=C8/D14*100` for percentage).
                3. **Compute Byar's Exact Poisson 95% Confidence Interval:**
                   * Lower SMR bound: `=(C8/D14)*(1 - 1/(9*C8) - 1.96/(3*SQRT(C8)))^3`
                   * Upper SMR bound: `=((C8+1)/D14)*(1 - 1/(9*(C8+1)) + 1.96/(3*SQRT(C8+1)))^3`
                """)

    with tab_code:
        if is_indirect:
            st.markdown("### 🐍 Python Implementation Workflow for Indirect Rate Standardization (SMR)")
            st.markdown("This guide follows a strict 4-step sequence: **Evaluate Prerequisites** ➔ **Implement Indirect SMR (Primary Method)** ➔ **Implement Direct DSR (Comparative Benchmark)** ➔ **Consolidated Executable Pipeline Script**.")
        else:
            st.markdown("### 🐍 Python Implementation Workflow for Direct Rate Standardization (DSR)")
            st.markdown("This guide follows a strict 4-step sequence: **Evaluate Prerequisites** ➔ **Implement Direct DSR (Primary Method)** ➔ **Implement Indirect SMR (Comparative Benchmark)** ➔ **Consolidated Executable Pipeline Script**.")

        # STEP 1: EVALUATE PREREQUISITES
        with st.expander("📌 Step 1: Evaluate Prerequisites & Assumptions in Python (Denominators, Event Volumes, Zero Cells, Rate Ratios)", expanded=False):
            st.code("""import numpy as np
    import pandas as pd

    # Load sample data (aligned with Excel Columns A through F)
    # df has columns: Col_A_Age_Band, Col_B_Population, Col_C_Observed_Deaths, Col_D_Rate_per_1k, Col_E_ESP_Weight, Col_F_Reference_Rate_per_1k
    pop = np.asarray(df["Col_B_Population"], dtype=float)
    deaths = np.asarray(df["Col_C_Observed_Deaths"], dtype=float)
    ref_rates = np.asarray(df["Col_F_Reference_Rate_per_1k"], dtype=float)

    # 1. Denominator Non-Negativity & Positivity Check
    zero_or_neg_pop = int(np.sum(pop <= 0))
    total_pop = int(np.sum(pop))
    if zero_or_neg_pop == 0:
        print(f"[ASSURANCE - PASS] All {len(pop)} age strata have positive populations (Total Population: {total_pop:,}).")
    else:
        print(f"[INAPPROPRIATE - FAIL] {zero_or_neg_pop} age strata have zero or negative population. Rates cannot be computed.")

    # 2. Total Event Numerator Volume Check
    total_deaths = int(np.sum(deaths))
    if total_deaths >= 100:
        print(f"[ASSURANCE - PASS] Total observed deaths = {total_deaths} (>= 100). Ample statistical power for Direct DSR.")
    elif total_deaths >= 25:
        print(f"[WARNING] Total observed deaths = {total_deaths} (between 25 and 99). Use Dobson exact CI or consider Indirect SMR.")
    else:
        print(f"[INAPPROPRIATE - FAIL] Total observed deaths = {total_deaths} (< 25). Direct DSR is unstable; switch to Indirect SMR.")

    # 3. Stratum-Specific Sparsity (Zero Cells & Small Counts)
    zero_cells = int(np.sum(deaths == 0))
    sparse_cells = int(np.sum((deaths > 0) & (deaths < 5)))
    if zero_cells == 0 and sparse_cells == 0:
        print(f"[ASSURANCE - PASS] All {len(deaths)} age strata contain >= 5 observed events. No zero or sparse cells detected.")
    elif zero_cells == 0:
        print(f"[WARNING] {sparse_cells} age strata have < 5 observed deaths. Direct DSR remains usable, but stratum rates have wider variance.")
    else:
        print(f"[WARNING] {zero_cells} age strata have 0 observed deaths, and {sparse_cells} have < 5 deaths. Indirect SMR is strongly recommended.")

    # 4. Homogeneity of Rate Ratios (Proportionality Check)
    local_rates = np.where(pop > 0, (deaths / pop) * 1000.0, 0.0)
    rate_ratios = np.where(ref_rates > 0, local_rates / ref_rates, 0.0)
    valid_rr = rate_ratios[rate_ratios > 0]
    if len(valid_rr) > 1:
        dispersion = float(np.max(valid_rr) / np.min(valid_rr))
        if dispersion <= 3.0:
            print(f"[ASSURANCE - PASS] Rate ratio dispersion across strata = {dispersion:.2f}x (<= 3.0x). Summary standardization is valid.")
        else:
            print(f"[WARNING] Rate ratio dispersion across strata = {dispersion:.2f}x (> 3.0x). Heterogeneity detected across age groups.")
    """, language="python")

            st.markdown("""
            **🔍 How to Interpret the Diagnostic Output:**
            * **Population Denominators:** Must be positive for all age bands. If zero, collapse adjacent age brackets.
            * **Total Event Count Volume:** Total deaths $\ge 100$ provides tight confidence intervals with Normal/Dobson approximations. If $< 25$, Direct DSR suffers severe random noise $\\rightarrow$ switch to Indirect SMR.
            * **Zero Cells:** A count of 0 yields an empirical rate of 0.0. Indirect SMR borrows strength from the external benchmark rates to avoid zero-rate distortions.
            * **Rate Ratio Dispersion:** If the ratio of local to standard rates varies widely across age strata ($> 3.0\\times$), summary standardization masks important age-specific patterns.
            """)

        # STEP 2: IMPLEMENT DIRECT STANDARDIZATION (DSR)
        with st.expander("📌 Step 2: Implement Direct Age-Standardization (DSR) in Python (ESP 2013 & Dobson 95% CIs)", expanded=False):
            st.code("""import numpy as np
    import pandas as pd

    def calculate_direct_standardized_rate(
        pop_age: np.ndarray,
        deaths_age: np.ndarray,
        esp_weights: np.ndarray,
        multiplier: float = 1000.0,
        confidence_level: float = 0.95
    ) -> dict:
        \"\"\"Calculates Direct Age-Standardized Rate (DSR) and Dobson 95% Confidence Intervals.

        Direct standardization applies local stratum-specific rates to an external Standard 
        Population (European Standard Population ESP 2013) to eliminate demographic confounding.

        Args:
            pop_age (np.ndarray): 1D array of local population counts in each age stratum 
                (stratum denominator n_i). Must be non-negative integers or floats.
            deaths_age (np.ndarray): 1D array of observed health events or deaths in each 
                age stratum (stratum numerator d_i). Must be non-negative integers or floats.
            esp_weights (np.ndarray): 1D array of reference standard population weights 
                (e.g., European Standard Population ESP 2013 weights w_i). Automatically 
                normalized so sum equals 1.0.
            multiplier (float, optional): Scaling factor for the rate (e.g., 1000.0 for rate 
                per 1,000 population, or 100000.0 for per 100,000). Defaults to 1000.0.
            confidence_level (float, optional): Two-sided confidence level for interval 
                estimation (e.g. 0.95). Defaults to 0.95.

        Returns:
            dict: Complete standardization results containing:
                - 'dsr': Point estimate of Direct Standardized Rate scaled by multiplier.
                - 'se_dsr': Standard error of the standardized rate.
                - 'ci_lower': Lower bound of Dobson / Normal confidence interval.
                - 'ci_upper': Upper bound of Dobson / Normal confidence interval.
                - 'crude_rate': Unadjusted crude rate for comparison.
                - 'total_deaths': Total observed deaths across all age strata.
                - 'total_population': Total local population denominator.
                - 'age_specific_rates': 1D array of empirical stratum rates per multiplier.
        \"\"\"
        pop = np.asarray(pop_age, dtype=float)
        deaths = np.asarray(deaths_age, dtype=float)
        weights = np.asarray(esp_weights, dtype=float) / np.sum(esp_weights)

        # 1. Stratum-specific empirical rates per multiplier
        rates = np.where(pop > 0, (deaths / pop) * multiplier, 0.0)

        # 2. DSR weighted average
        dsr = float(np.sum(weights * rates))

        # 3. Variance under Poisson distribution: Var(DSR) = M^2 * sum( w_i^2 * d_i / n_i^2 )
        var_terms = np.where(pop > 0, (weights ** 2) * (deaths / (pop ** 2)) * (multiplier ** 2), 0.0)
        var_dsr = float(np.sum(var_terms))
        se_dsr = float(np.sqrt(max(0.0, var_dsr)))

        tot_deaths = float(np.sum(deaths))
        tot_pop = float(np.sum(pop))
        crude_rate = (tot_deaths / tot_pop) * multiplier if tot_pop > 0 else 0.0

        z = 1.959963984540054  # 95% two-sided normal quantile

        # 4. Confidence Interval: Dobson method for small events (< 100) or Normal approximation
        if tot_deaths >= 100:
            ci_lower = max(0.0, dsr - z * se_dsr)
            ci_upper = dsr + z * se_dsr
        elif tot_deaths > 0:
            # Dobson's method scales Byar exact Poisson limits of total deaths by Var(DSR) / D
            d_lower = tot_deaths * ((1.0 - (1.0 / (9.0 * tot_deaths)) - (z / (3.0 * np.sqrt(tot_deaths)))) ** 3)
            d_upper = (tot_deaths + 1.0) * ((1.0 - (1.0 / (9.0 * (tot_deaths + 1.0))) + (z / (3.0 * np.sqrt(tot_deaths + 1.0)))) ** 3)
            scale = np.sqrt(var_dsr / tot_deaths)
            ci_lower = max(0.0, dsr + scale * (d_lower - tot_deaths))
            ci_upper = dsr + scale * (d_upper - tot_deaths)
        else:
            ci_lower = 0.0
            ci_upper = 0.0

        return {
            "dsr": dsr,
            "se_dsr": se_dsr,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "crude_rate": crude_rate,
            "total_deaths": tot_deaths,
            "total_population": tot_pop,
            "age_specific_rates": rates
        }

    # Execute calculation on local authority data
    res_dsr = calculate_direct_standardized_rate(
        pop_age=df["Col_B_Population"].values,
        deaths_age=df["Col_C_Observed_Deaths"].values,
        esp_weights=df["Col_E_ESP_Weight"].values,
        multiplier=1000.0
    )

    print(f"=== DIRECT AGE-STANDARDIZED RATE (DSR) RESULTS ===")
    print(f"Crude Death Rate:           {res_dsr['crude_rate']:.2f} per 1,000")
    print(f"Direct Standardized (DSR):  {res_dsr['dsr']:.2f} per 1,000 (ESP 2013)")
    print(f"Standard Error SE(DSR):     {res_dsr['se_dsr']:.3f}")
    print(f"95% Confidence Interval:    [{res_dsr['ci_lower']:.2f} - {res_dsr['ci_upper']:.2f}] per 1,000")
    """, language="python")

        # STEP 3: IMPLEMENT INDIRECT STANDARDIZATION (SMR)
        with st.expander("📌 Step 3: Implement Indirect Standardization & Standardized Mortality Ratio (SMR) in Python (Byar's Poisson 95% CIs)", expanded=False):
            st.code("""import numpy as np
    import pandas as pd

    def calculate_indirect_standardization(
        pop_age: np.ndarray,
        observed_deaths_total: float,
        reference_rates_per_k: np.ndarray,
        multiplier: float = 1000.0,
        confidence_level: float = 0.95
    ) -> dict:
        \"\"\"Calculates Indirect Standardization, Expected Deaths, SMR, and Byar's 95% Poisson CIs.

        Indirect standardization applies external standard reference rates to the local population 
        age structure to compute Expected Deaths (E). The SMR is the ratio of Observed to Expected.

        Args:
            pop_age (np.ndarray): 1D array of local population counts in each age stratum 
                (stratum denominator n_i). Must be non-negative integers or floats.
            observed_deaths_total (float): Total number of observed health events or deaths 
                aggregated across all age groups (O = sum(d_i)). Must be non-negative.
            reference_rates_per_k (np.ndarray): 1D array of benchmark reference age-specific 
                rates per multiplier from the standard reference population (R_std,i).
            multiplier (float, optional): Scaling denominator for reference rates (e.g. 1000.0). 
                Defaults to 1000.0.
            confidence_level (float, optional): Two-sided confidence level. Defaults to 0.95.

        Returns:
            dict: Calculation results containing:
                - 'expected_deaths': Total expected deaths (E = sum(n_i * R_std,i / multiplier)).
                - 'observed_deaths': Total observed deaths (O).
                - 'smr': Standardized Mortality Ratio (O / E; 1.0 represents benchmark parity).
                - 'smr_percentage': SMR expressed as a percentage (100 * O / E).
                - 'smr_ci_lower': Lower bound of Byar's exact Poisson 95% CI for SMR.
                - 'smr_ci_upper': Upper bound of Byar's exact Poisson 95% CI for SMR.
                - 'smr_pct_ci_lower': Lower bound for SMR percentage.
                - 'smr_pct_ci_upper': Upper bound for SMR percentage.
        \"\"\"
        pop = np.asarray(pop_age, dtype=float)
        ref_rates = np.asarray(reference_rates_per_k, dtype=float)

        # 1. Stratum-specific expected deaths: E_i = n_i * R_std,i / multiplier
        stratum_exp = (pop * ref_rates) / multiplier
        tot_exp = float(np.sum(stratum_exp))
        obs = float(observed_deaths_total)

        # 2. SMR point estimate: O / E
        smr = float(obs / tot_exp) if tot_exp > 0 else 1.0
        smr_pct = smr * 100.0

        # 3. Byar's formula for exact Poisson 95% confidence interval
        z = 1.959963984540054
        if obs > 0 and tot_exp > 0:
            obs_lower = obs * ((1.0 - (1.0 / (9.0 * obs)) - (z / (3.0 * np.sqrt(obs)))) ** 3)
            obs_upper = (obs + 1.0) * ((1.0 - (1.0 / (9.0 * (obs + 1.0))) + (z / (3.0 * np.sqrt(obs + 1.0)))) ** 3)
            ci_lower = max(0.0, float(obs_lower / tot_exp))
            ci_upper = float(obs_upper / tot_exp)
        else:
            ci_lower = 0.0
            ci_upper = 3.688879 / tot_exp if tot_exp > 0 else 0.0

        return {
            "expected_deaths": tot_exp,
            "observed_deaths": obs,
            "smr": smr,
            "smr_percentage": smr_pct,
            "smr_ci_lower": ci_lower,
            "smr_ci_upper": ci_upper,
            "smr_pct_ci_lower": ci_lower * 100.0,
            "smr_pct_ci_upper": ci_upper * 100.0
        }

    # Execute calculation on local authority data
    obs_total = df["Col_C_Observed_Deaths"].sum()
    res_smr = calculate_indirect_standardization(
        pop_age=df["Col_B_Population"].values,
        observed_deaths_total=obs_total,
        reference_rates_per_k=df["Col_F_Reference_Rate_per_1k"].values,
        multiplier=1000.0
    )

    print(f"=== INDIRECT STANDARDIZATION (SMR) RESULTS ===")
    print(f"Observed Deaths (O):     {res_smr['observed_deaths']:.0f}")
    print(f"Expected Deaths (E):     {res_smr['expected_deaths']:.1f}")
    print(f"Standardized Ratio SMR:  {res_smr['smr']:.3f} (Baseline = 1.000)")
    print(f"SMR Percentage:          {res_smr['smr_percentage']:.1f}% (Baseline = 100.0%)")
    print(f"Byar's 95% Poisson CI:   [{res_smr['smr_ci_lower']:.3f} - {res_smr['smr_ci_upper']:.3f}]")
    """, language="python")

        # STEP 4: CONSOLIDATED EXECUTABLE PIPELINE
        with st.expander("📌 Step 4: Consolidated Executable Standardization Pipeline Script", expanded=False):
            st.code("""import numpy as np
    import pandas as pd

    def run_standardization_pipeline(df: pd.DataFrame, area_name: str = "Local Authority"):
        \"\"\"Runs complete demographic rate standardization pipeline with assumption audits.

        Args:
            df (pd.DataFrame): DataFrame containing columns:
                - 'Col_A_Age_Band': Age bracket labels
                - 'Col_B_Population': Local population count in each age bracket (n_i)
                - 'Col_C_Observed_Deaths': Observed deaths in each age bracket (d_i)
                - 'Col_D_Rate_per_1k': Empirical age-specific rate per 1,000
                - 'Col_E_ESP_Weight': European Standard Population weight (w_i)
                - 'Col_F_Reference_Rate_per_1k': Standard reference benchmark rate (R_std,i)
            area_name (str, optional): Name of the local authority. Defaults to "Local Authority".

        Returns:
            dict: Combined results dictionary with DSR and SMR metrics.
        \"\"\"
        pop = np.asarray(df["Col_B_Population"], dtype=float)
        deaths = np.asarray(df["Col_C_Observed_Deaths"], dtype=float)
        weights = np.asarray(df["Col_E_ESP_Weight"], dtype=float) / np.sum(df["Col_E_ESP_Weight"])
        ref_rates = np.asarray(df["Col_F_Reference_Rate_per_1k"], dtype=float)

        print(f"============================================================")
        print(f"DEMOGRAPHIC RATE STANDARDIZATION AUDIT: {area_name.upper()}")
        print(f"============================================================")

        # 1. Prerequisite Checks
        tot_pop = int(np.sum(pop))
        tot_deaths = int(np.sum(deaths))
        zero_cells = int(np.sum(deaths == 0))
        sparse_cells = int(np.sum((deaths > 0) & (deaths < 5)))

        print("\\n--- 1. ASSUMPTION & PREREQUISITE DIAGNOSTICS ---")
        if np.all(pop > 0):
            print(f"[ASSURANCE - PASS] Denominators valid across all {len(pop)} strata (Pop: {tot_pop:,}).")
        else:
            print("[INAPPROPRIATE - FAIL] Found non-positive population denominators.")

        if tot_deaths >= 100:
            print(f"[ASSURANCE - PASS] Event count = {tot_deaths} (>= 100). Robust for Direct DSR.")
        elif tot_deaths >= 25:
            print(f"[WARNING] Event count = {tot_deaths}. Use Dobson exact CI.")
        else:
            print(f"[INAPPROPRIATE - FAIL] Event count = {tot_deaths} (< 25). Switch to Indirect SMR.")

        if zero_cells == 0 and sparse_cells == 0:
            print(f"[ASSURANCE - PASS] No zero or sparse cells detected.")
        else:
            print(f"[WARNING] Found {zero_cells} zero cells and {sparse_cells} sparse cells (< 5 events).")

        # 2. Direct Standardization (DSR)
        rates_per_1k = np.where(pop > 0, (deaths / pop) * 1000.0, 0.0)
        dsr = float(np.sum(weights * rates_per_1k))
        var_dsr = float(np.sum((weights ** 2) * (deaths / (pop ** 2)) * (1000.0 ** 2)))
        se_dsr = float(np.sqrt(max(0.0, var_dsr)))
        crude_rate = (tot_deaths / tot_pop) * 1000.0 if tot_pop > 0 else 0.0

        z = 1.959963984540054
        if tot_deaths >= 100:
            dsr_lower = max(0.0, dsr - z * se_dsr)
            dsr_upper = dsr + z * se_dsr
        elif tot_deaths > 0:
            d_lower = tot_deaths * ((1.0 - (1.0 / (9.0 * tot_deaths)) - (z / (3.0 * np.sqrt(tot_deaths)))) ** 3)
            d_upper = (tot_deaths + 1.0) * ((1.0 - (1.0 / (9.0 * (tot_deaths + 1.0))) + (z / (3.0 * np.sqrt(tot_deaths + 1.0)))) ** 3)
            scale = np.sqrt(var_dsr / tot_deaths)
            dsr_lower = max(0.0, dsr + scale * (d_lower - tot_deaths))
            dsr_upper = dsr + scale * (d_upper - tot_deaths)
        else:
            dsr_lower, dsr_upper = 0.0, 0.0

        print("\\n--- 2. DIRECT AGE-STANDARDIZED RATE (DSR) ---")
        print(f"Crude Death Rate:           {crude_rate:.2f} per 1,000")
        print(f"Direct Standardized (DSR):  {dsr:.2f} per 1,000 (ESP 2013)")
        print(f"Standard Error SE(DSR):     {se_dsr:.3f}")
        print(f"Dobson 95% CI:              [{dsr_lower:.2f} - {dsr_upper:.2f}] per 1,000")

        # 3. Indirect Standardization (SMR)
        exp_deaths = float(np.sum((pop * ref_rates) / 1000.0))
        smr = float(tot_deaths / exp_deaths) if exp_deaths > 0 else 1.0
        if tot_deaths > 0 and exp_deaths > 0:
            o_lower = tot_deaths * ((1.0 - (1.0 / (9.0 * tot_deaths)) - (z / (3.0 * np.sqrt(tot_deaths)))) ** 3)
            o_upper = (tot_deaths + 1.0) * ((1.0 - (1.0 / (9.0 * (tot_deaths + 1.0))) + (z / (3.0 * np.sqrt(tot_deaths + 1.0)))) ** 3)
            smr_lower = float(o_lower / exp_deaths)
            smr_upper = float(o_upper / exp_deaths)
        else:
            smr_lower, smr_upper = 0.0, 0.0

        print("\\n--- 3. INDIRECT STANDARDIZATION (SMR) ---")
        print(f"Observed Deaths (O):        {tot_deaths}")
        print(f"Expected Deaths (E):        {exp_deaths:.1f}")
        print(f"Standardized Ratio (SMR):   {smr:.3f} (Benchmark = 1.000)")
        print(f"SMR Percentage:             {smr * 100.0:.1f}%")
        print(f"Byar's 95% Poisson CI:      [{smr_lower:.3f} - {smr_upper:.3f}]")
        print(f"============================================================")

        return {
            "crude_rate": crude_rate,
            "dsr": dsr,
            "dsr_lower": dsr_lower,
            "dsr_upper": dsr_upper,
            "expected_deaths": exp_deaths,
            "smr": smr,
            "smr_lower": smr_lower,
            "smr_upper": smr_upper
        }
    """, language="python")

    with tab_download:
        st.markdown(f"### 📥 Download Sample Dataset for **{focal_district}**")
        st.markdown("""
        The dataset below is structured specifically to align **1:1 with the cell and column coordinates** 
        (`Col_A_Age_Band` through `Col_F_Reference_Rate_per_1k` across rows 2 to 7) described in the 
        **📗 Microsoft Excel Implementation Guide**.
        """)

        st.dataframe(df_excel_focal, use_container_width=True, hide_index=True)

        col_dl1, col_dl2 = st.columns([1, 1])
        with col_dl1:
            st.download_button(
                f"📥 Download Excel-Aligned Data for {focal_district} (CSV)",
                data=df_excel_focal.to_csv(index=False),
                file_name=f"{focal_district.lower().replace(' ', '_')}_excel_standardization_sample_data.csv",
                mime="text/csv",
                key="btn_dl_focal_excel_data"
            )
        with col_dl2:
            st.caption(f"💡 Contains the 6 standard European Standard Population age bands for **{focal_district}**.")

        with st.expander("📁 Download Full Multi-District Dataset (All 9 Derbyshire Authorities, 54 Rows)", expanded=False):
            st.markdown("""
            If you wish to compare multiple authorities across Derbyshire or practice running comparative 
            loops in Python, download the full multi-area dataset below:
            """)
            st.download_button(
                "📥 Download Complete 9-District Dataset (CSV)",
                data=df_detailed.to_csv(index=False),
                file_name="derbyshire_all_9_districts_standardization_data.csv",
                mime="text/csv",
                key="btn_dl_all_districts_data"
            )
            st.dataframe(df_detailed.head(12), height=220, use_container_width=True, hide_index=True)


    # ----------------- Live Interactive Spatial & Demographic Demonstration -----------------
    st.markdown("---")
    st.subheader("🧪 Live Interactive Demonstration: The Reversal Paradox in Derbyshire")
    st.markdown(f"**Scenario:** *{SCENARIO_CONFIGS[selected_scenario]['title']}*")

    # The Reversal Paradox Educational Callout Banner
    dales_row = df_summary[df_summary["District"] == "Derbyshire Dales"].iloc[0]
    derby_row = df_summary[df_summary["District"] == "Derby"].iloc[0]

    st.markdown(f"""
    <div style="background-color: #fffbeb; border: 2px solid #f59e0b; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem;">💡</span>
            <strong style="font-size: 1.15rem; color: #92400e;">The Reversal Paradox (Simpson's Paradox in Spatial Epidemiology)</strong>
        </div>
        <div style="font-size: 0.95rem; color: #1e293b; line-height: 1.6;">
            Notice what happens when comparing <strong>Derbyshire Dales</strong> and <strong>Derby City</strong>:
            <ul>
                <li><strong>Crude Death Rate:</strong> Derbyshire Dales has a crude rate of <strong>{dales_row['Crude_Rate_per_1k']:.2f} per 1,000</strong> compared to Derby City's <strong>{derby_row['Crude_Rate_per_1k']:.2f} per 1,000</strong>. A naive observer would conclude that living in Derbyshire Dales carries higher mortality risk!</li>
                <li><strong>The Demographic Distortion:</strong> In reality, <strong>{dales_row['Pct_65_Plus']:.1f}%</strong> of Derbyshire Dales residents are aged 65 or older, compared to just <strong>{derby_row['Pct_65_Plus']:.1f}%</strong> in Derby City.</li>
                <li><strong>Direct Standardization (DSR):</strong> Once adjusted for age (ESP 2013), the ranking <strong>completely reverses</strong>: Derbyshire Dales drops to <strong>{dales_row['DSR_per_1k']:.2f} per 1,000</strong>, while Derby City rises to <strong>{derby_row['DSR_per_1k']:.2f} per 1,000</strong>. Derbyshire Dales actually has superior age-specific health outcomes!</li>
                <li><strong>Indirect Standardization (SMR):</strong> Comparing observed deaths against expected deaths ($E$) based on benchmark rates reveals that Derbyshire Dales observed <strong>{int(dales_row['Total_Deaths'])}</strong> deaths vs. <strong>{dales_row['Expected_Deaths_Indirect']:.1f}</strong> expected ($\\text{{SMR}} = {dales_row['SMR']:.3f}$ or <strong>{dales_row['SMR_Percentage']:.1f}%</strong> — {100.0 - dales_row['SMR_Percentage']:.1f}% below benchmark). In contrast, Derby City recorded <strong>{int(derby_row['Total_Deaths'])}</strong> deaths vs. <strong>{derby_row['Expected_Deaths_Indirect']:.1f}</strong> expected ($\\text{{SMR}} = {derby_row['SMR']:.3f}$ or <strong>{derby_row['SMR_Percentage']:.1f}%</strong> — {derby_row['SMR_Percentage'] - 100.0:.1f}% above benchmark). Both methods resolve Simpson's Paradox!</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ----------------- Section 1: Interactive Folium Choropleth Map -----------------
    st.markdown("### 🗺️ Geographical Choropleth Map: Derbyshire & Derby City (9 Local Authorities)")

    default_map_idx = 2 if is_indirect else 0

    col_map_metric, col_map_style = st.columns([3, 2])
    with col_map_metric:
        map_mode = st.selectbox(
            "Select Map Metric to Display:",
            options=["dsr", "crude", "smr"],
            index=default_map_idx,
            format_func=lambda m: {
                "dsr": "🟢 Direct Age-Standardized Rate (DSR per 1,000 ESP 2013) — Age-Adjusted True Risk",
                "crude": "🔴 Crude Death Rate (Unadjusted Deaths per 1,000 Population) — Demographically Confounded",
                "smr": "🔵 Standardized Mortality Ratio (SMR %: 100% = Baseline Benchmark)"
            }[m],
        )
    with col_map_style:
        basemap_choice = st.selectbox(
            "Basemap Style (100% Free & Legal — No API Key Required):",
            options=["osm", "canvas"],
            format_func=lambda b: {
                "osm": "🌍 OpenStreetMap (Standard Free Basemap)",
                "canvas": "⚪ Minimalist Canvas (Clean Vector Boundaries Only)"
            }[b],
            #horizontal=True,
            help="OpenStreetMap is free under Open Data Commons ODbL; Minimalist Canvas eliminates street noise for clean thematic display."
        )

    st.caption("""
    ℹ️ **Basemap Transparency Notice:** Commercial map providers (e.g., Carto, Mapbox) now enforce domain tokens and display 
    an *"API key required"* mask on unauthenticated requests. The two standard, fully legal, and open-access approaches in spatial epidemiology 
    are **OpenStreetMap** (free open-data basemap under ODbL) and **Minimalist Vector Canvas** (renders high-contrast district boundaries 
    without external street tile dependencies, matching ONS and UKHSA public health atlas standards).
    """)

    geojson_path = get_derbyshire_geojson_path()

    if HAS_FOLIUM and os.path.exists(geojson_path):
        with open(geojson_path, "r", encoding="utf-8") as gf:
            geo_data = json.load(gf)

        # Attach metric values to geojson features
        metric_col = "DSR_per_1k" if map_mode == "dsr" else ("Crude_Rate_per_1k" if map_mode == "crude" else "SMR_Percentage")
        metric_title = "DSR per 1k" if map_mode == "dsr" else ("Crude per 1k" if map_mode == "crude" else "SMR (%)")

        val_dict = dict(zip(df_summary["District"], df_summary[metric_col]))

        for feat in geo_data["features"]:
            d_name = feat["properties"]["name"]
            feat["properties"]["metric_val"] = val_dict.get(d_name, 0.0)
            row_match = df_summary[df_summary["District"] == d_name]
            if not row_match.empty:
                r = row_match.iloc[0]
                feat["properties"]["crude_rate"] = r["Crude_Rate_per_1k"]
                feat["properties"]["dsr_rate"] = r["DSR_per_1k"]
                feat["properties"]["smr_pct"] = r["SMR_Percentage"]
                feat["properties"]["pop"] = int(r["Total_Population"])
                feat["properties"]["pct_65"] = r["Pct_65_Plus"]

        # Configure Basemap Tiles without requiring any commercial API keys
        tiles_val = "OpenStreetMap" if basemap_choice == "osm" else None
        attr_val = "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors" if basemap_choice == "osm" else None

        # Initialize Folium Map centered on Derbyshire (53.1° N, -1.5° W)
        m = folium.Map(
            location=[53.12, -1.52],
            zoom_start=9,
            tiles=tiles_val,
            attr=attr_val
        )

        # Fit viewport directly to Derbyshire local authority bounds
        m.fit_bounds([[52.75, -2.15], [53.55, -1.10]])

        color_scheme = "YlGnBu" if map_mode == "dsr" else ("YlOrRd" if map_mode == "crude" else "PuBu")

        folium.Choropleth(
            geo_data=geo_data,
            name="choropleth",
            data=df_summary,
            columns=["District", metric_col],
            key_on="feature.properties.name",
            fill_color=color_scheme,
            fill_opacity=0.80 if basemap_choice == "osm" else 0.88,
            line_opacity=0.9,
            line_color="#1e293b",
            line_weight=1.8,
            legend_name=f"{metric_title} — {SCENARIO_CONFIGS[selected_scenario]['title']}"
        ).add_to(m)

        # Add interactive hover tooltips
        folium.GeoJson(
            geo_data,
            style_function=lambda x: {"fillColor": "transparent", "color": "transparent"},
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "pop", "pct_65", "crude_rate", "dsr_rate", "smr_pct"],
                aliases=["District:", "Population:", "Aged 65+ (%):", "Crude Rate (per 1k):", "DSR Rate (per 1k):", "SMR (%):"],
                localize=True,
                sticky=True
            )
        ).add_to(m)

        # Render Folium map with returned_objects=[] to ensure fast, static load without iframe blanking
        st_folium(
            m,
            use_container_width=True,
            height=500,
            returned_objects=[],
            key=f"derbyshire_folium_{map_mode}_{selected_scenario}_{basemap_choice}"
        )
    else:
        # High-quality spatial fallback if Folium or boundaries are missing
        st.info("ℹ️ Interactive Folium map active. Rendering comparative geospatial bar breakdown below.")


    # ----------------- Section 2: Comparative Analytical Charts -----------------
    st.markdown("---")
    st.markdown("### 📊 Demographic Age Pyramids & Comparative Rate Visualizations")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        pyramid_district = st.selectbox(
            "Select Local Authority to compare against ESP 2013:",
            options=[p["name"] for p in DISTRICT_PROFILES],
            index=1,  # Default to Derbyshire Dales
            key="select_pyramid_authority",
            help="Inspect any authority's demographic age profile against the European Standard Population (ESP 2013)."
        )

        st.markdown(f"**1. Population Age Structure: {pyramid_district} vs. ESP 2013 Standard**")

        # Prepare comparison data
        pyr_df = df_detailed[df_detailed["District"] == pyramid_district].copy()
        pyr_tot = pyr_df["Population"].sum()
        pyr_df["Share_Pct"] = (pyr_df["Population"] / pyr_tot) * 100.0
        pyr_df["ESP_Pct"] = pyr_df["ESP_Weight"] * 100.0

        fig_pyr = go.Figure()
        fig_pyr.add_trace(go.Bar(
            x=pyr_df["Age_Band_Label"],
            y=pyr_df["Share_Pct"],
            name=pyramid_district,
            marker_color="#2563eb"
        ))
        fig_pyr.add_trace(go.Bar(
            x=pyr_df["Age_Band_Label"],
            y=pyr_df["ESP_Pct"],
            name="ESP 2013 Standard",
            marker_color="#94a3b8"
        ))
        fig_pyr.update_layout(
            barmode="group",
            height=380,
            margin=dict(l=30, r=30, t=30, b=40),
            xaxis_title="Age Bracket",
            yaxis_title="% of Total Population",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pyr, use_container_width=True)

    with chart_col2:
        default_comp_idx = 1 if is_indirect else 0
        comp_metric = st.radio(
            "Select Comparative Visualisation:",
            options=["dsr", "smr"],
            index=default_comp_idx,
            format_func=lambda c: {
                "dsr": "🟢 Crude Rate vs. Direct Age-Standardized Rate (DSR)",
                "smr": "🔵 Standardized Mortality Ratio (SMR %) with Byar's 95% CIs"
            }[c],
            horizontal=True,
            key="radio_comparative_chart"
        )

        if comp_metric == "dsr":
            st.markdown("**2. The Reversal Effect: Crude Rate vs. Direct Age-Standardized Rate (DSR)**")
            df_sorted = df_summary.sort_values(by="DSR_per_1k", ascending=True)

            fig_rates = go.Figure()
            fig_rates.add_trace(go.Bar(
                y=df_sorted["District"],
                x=df_sorted["Crude_Rate_per_1k"],
                name="Crude Rate (Unadjusted)",
                orientation="h",
                marker_color="#f87171"
            ))
            fig_rates.add_trace(
                go.Bar(
                    y=df_sorted["District"],
                    x=df_sorted["DSR_per_1k"],
                    name="DSR (Age-Standardized)",
                    orientation="h",
                    marker_color="#34d399"
                )
            )
            fig_rates.update_layout(
                barmode="group",
                height=380,
                margin=dict(l=30, r=30, t=30, b=40),
                xaxis_title="Rate per 1,000 Population",
                yaxis_title="",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_rates, use_container_width=True)
        else:
            st.markdown("**2. Indirect Standardization: Standardized Mortality Ratio (SMR %) vs. Benchmark**")
            df_sorted_smr = df_summary.sort_values(by="SMR_Percentage", ascending=True)

            # Compute asymmetric error bars for Byar's 95% Poisson limits
            error_plus = (df_sorted_smr["SMR_95_Upper"] * 100.0) - df_sorted_smr["SMR_Percentage"]
            error_minus = df_sorted_smr["SMR_Percentage"] - (df_sorted_smr["SMR_95_Lower"] * 100.0)

            fig_smr = go.Figure()
            fig_smr.add_trace(go.Bar(
                y=df_sorted_smr["District"],
                x=df_sorted_smr["SMR_Percentage"],
                name="SMR % (100% = Baseline Benchmark)",
                orientation="h",
                marker_color="#3b82f6",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=error_plus,
                    arrayminus=error_minus,
                    color="#1e3a8a"
                )
            ))
            fig_smr.add_vline(
                x=100.0,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text="100% Benchmark",
                annotation_position="top right"
            )
            fig_smr.update_layout(
                height=380,
                margin=dict(l=30, r=30, t=30, b=40),
                xaxis_title="Standardized Mortality Ratio (SMR %)",
                yaxis_title="",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_smr, use_container_width=True)


    # ----------------- Section 3: Detailed Tables & Audit Microdata -----------------
    st.markdown("---")
    st.markdown("### 📋 Local Authority Summary Metrics (Derby City + 8 Derbyshire Districts)")
    st.caption("Compare Crude Rates, Age-Standardized Rates (DSR with 95% CIs), Expected Deaths, and Standardized Mortality Ratios (SMR).")

    cols_summary_display = [
        "District", "Authority_Type", "Total_Population", "Total_Deaths",
        "Expected_Deaths_Indirect", "SMR", "SMR_Percentage", "SMR_95_Lower", "SMR_95_Upper",
        "Crude_Rate_per_1k", "DSR_per_1k", "DSR_95_Lower", "DSR_95_Upper"
    ] if is_indirect else [
        "District", "Authority_Type", "Total_Population", "Pct_65_Plus",
        "Total_Deaths", "Crude_Rate_per_1k", "DSR_per_1k", "DSR_95_Lower",
        "DSR_95_Upper", "Expected_Deaths_Indirect", "SMR", "SMR_Percentage",
        "Crude_Rank", "DSR_Rank", "Rank_Shift"
    ]

    st.dataframe(
        df_summary[cols_summary_display],
        use_container_width=True,
        height=260,
        hide_index=True
    )

    with st.expander("🔎 View Full Age-Specific Microdata Across All 9 Districts (54 Rows)", expanded=False):
        st.dataframe(
            df_detailed[[
                "District", "Age_Band_Label", "Population", "Observed_Deaths",
                "Age_Specific_Rate_per_1k", "ESP_Weight", "DSR_Component_per_1k",
                "Expected_Deaths_Indirect"
            ]],
            use_container_width=True,
            height=300,
            hide_index=True
        )

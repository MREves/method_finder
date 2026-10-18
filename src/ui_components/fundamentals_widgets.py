"""UI Widgets and Interactive Visualizations for Statistical Fundamentals.

Provides 6 self-contained foundational learning modules:
1. Measurement Scales & Data Types (Nominal, Ordinal, Interval, Ratio)
2. Measures of Central Tendency (Mean, Median, Mode, Trimmed Mean)
3. Measures of Dispersion & Spread (Variance, SD, Quartiles, IQR, Tukey's Fences)
4. Standard Deviation (SD) vs. Standard Error of the Mean (SEM)
5. Distribution Shapes, Skewness & Central Limit Theorem (CLT)
6. Confidence Intervals (Continuous Mean: z vs t; Binary Proportion: Wald vs Wilson vs Agresti-Coull)

Each module adheres strictly to:
- A prominent interactive playground with live visual simulations.
- The standardized 4-Tab educational layout:
  📋 Rationale & Formulas | 📗 Excel Guide | 🐍 Python Code | 💾 Practice Data Download.
- 100% compliance with st.expander(expanded=False).
"""

import math
import random
from typing import Dict, Any, List, Optional
import streamlit as st

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go = None
    px = None

HAS_SCIENTIFIC_STACK = bool(pd is not None and np is not None)

from src.stats_calcs.fundamentals_engine import (
    calculate_central_tendency,
    calculate_dispersion_and_quartiles,
    calculate_sd_vs_sem,
    calculate_distribution_shape_and_clt,
    calculate_confidence_interval_mean,
    calculate_confidence_interval_proportion,
    generate_scales_practice_data,
    generate_central_tendency_practice_data,
    generate_dispersion_practice_data,
    generate_sd_sem_practice_data,
    generate_clt_practice_data,
    generate_confidence_intervals_practice_data,
    _normal_ppf,
    _t_ppf
)


def render_html(html_str: str) -> None:
    """Safely renders HTML content, stripping excessive indentation to prevent

    CommonMark code-block misinterpretation, and falls back to st.markdown.
    """
    clean_lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    cleaned_html = "".join(clean_lines)
    if hasattr(st, "html"):
        st.html(cleaned_html)
    else:
        st.markdown(cleaned_html, unsafe_allow_html=True)


# =====================================================================
# 1. Measurement Scales & Data Types Module
# =====================================================================

def render_scales_data_types_module() -> None:
    """Renders Module 1: Measurement Scales & Data Types."""
    st.markdown("## 🏷️ Measurement Scales & Data Types")
    st.markdown("""
    The mathematical operations you are allowed to perform depend entirely on the **measurement scale** of your data.
    Before calculating an average, standard deviation, or correlation, you must classify your variables correctly.
    """)

    # Interactive Scale Classifier Playground
    st.markdown("### 🎮 Interactive Data Scale Classifier Playground")
    st.caption("Select a real-world healthcare variable below or enter your own to inspect its properties and permitted operations.")

    c1, c2 = st.columns([1.6, 2.4])
    with c1:
        preset_var = st.selectbox(
            "Select Clinical / Operational Variable:",
            options=[
                "Blood Type (O, A, B, AB)",
                "Pain Severity Score (Likert 1 to 5)",
                "NYHA Heart Failure Class (I, II, III, IV)",
                "Body Temperature in Celsius (°C)",
                "Calendar Year of Admission (2024)",
                "Systolic Blood Pressure (mmHg)",
                "Hospital Length of Stay (Days)",
                "Medication Dosage (mg)"
            ],
            key="scale_preset_select"
        )

    # Scale mapping
    scale_info_map = {
        "Blood Type (O, A, B, AB)": {
            "scale": "Nominal (Categorical)",
            "definition": "Unordered categories. Values serve as mutually exclusive naming labels.",
            "zero_type": "No natural zero.",
            "allowed_ops": "Count, Frequency, Mode, Chi-Square Test.",
            "forbidden_ops": "Mean, Standard Deviation, Greater than / Less than.",
            "color": "#2563eb", "bg": "#eff6ff"
        },
        "Pain Severity Score (Likert 1 to 5)": {
            "scale": "Ordinal (Ordered Categorical)",
            "definition": "Ordered ranks where order matters, but spacing between levels is unknown or non-uniform.",
            "zero_type": "Arbitrary lower bound (0 or 1 does not mean 'zero pain units').",
            "allowed_ops": "Median, Interquartile Range (IQR), Rank-order comparisons, Mann-Whitney U.",
            "forbidden_ops": "Arithmetic Mean, Ratio multiplication (Score 4 is NOT 'twice as painful' as Score 2).",
            "color": "#d97706", "bg": "#fffbeb"
        },
        "NYHA Heart Failure Class (I, II, III, IV)": {
            "scale": "Ordinal (Ordered Categorical)",
            "definition": "Clinical severity staging where Class IV is worse than Class III, but the physiological gap is unequal.",
            "zero_type": "No zero stage.",
            "allowed_ops": "Median stage, Proportions, Wilcoxon / Kruskal-Wallis.",
            "forbidden_ops": "Computing 'Mean NYHA Class = 2.4'.",
            "color": "#d97706", "bg": "#fffbeb"
        },
        "Body Temperature in Celsius (°C)": {
            "scale": "Interval (Continuous Numerical)",
            "definition": "Continuous values with equal intervals between degrees, but an arbitrary zero point (0°C is water freezing, not absence of heat).",
            "zero_type": "Arbitrary Zero (0°C does not mean zero thermal energy).",
            "allowed_ops": "Addition, Subtraction, Mean, SD, Differences (38°C is 2°C hotter than 36°C).",
            "forbidden_ops": "Ratios (40°C is NOT 'twice as hot' as 20°C; 40°C = 313.15 K, 20°C = 293.15 K).",
            "color": "#059669", "bg": "#ecfdf5"
        },
        "Calendar Year of Admission (2024)": {
            "scale": "Interval (Discrete Numerical)",
            "definition": "Chronological intervals with an arbitrary epoch benchmark (Year 0 is arbitrary).",
            "zero_type": "Arbitrary Zero.",
            "allowed_ops": "Elapsed years difference, Trend regression.",
            "forbidden_ops": "Direct multiplicative ratios.",
            "color": "#059669", "bg": "#ecfdf5"
        },
        "Systolic Blood Pressure (mmHg)": {
            "scale": "Ratio (Continuous Numerical)",
            "definition": "Continuous measurements with a true physical zero point (0 mmHg represents absolute absence of pressure).",
            "zero_type": "True Absolute Zero.",
            "allowed_ops": "All arithmetic operations: Mean, SD, Differences, Percentages, Multiplicative Ratios (160 mmHg is exactly twice 80 mmHg).",
            "forbidden_ops": "None. Highest measurement scale.",
            "color": "#7c3aed", "bg": "#faf5ff"
        },
        "Hospital Length of Stay (Days)": {
            "scale": "Ratio (Continuous / Count Numerical)",
            "definition": "Duration with absolute zero (0 days = same-day outpatient discharge).",
            "zero_type": "True Absolute Zero.",
            "allowed_ops": "Mean, Median, Ratios (10 days is twice 5 days), Geometric Mean.",
            "forbidden_ops": "None.",
            "color": "#7c3aed", "bg": "#faf5ff"
        },
        "Medication Dosage (mg)": {
            "scale": "Ratio (Continuous Numerical)",
            "definition": "Mass / volume with absolute zero (0 mg = complete absence of drug).",
            "zero_type": "True Absolute Zero.",
            "allowed_ops": "Full mathematical suite: Mean, Standard Deviation, Dose Proportionality ratios.",
            "forbidden_ops": "None.",
            "color": "#7c3aed", "bg": "#faf5ff"
        }
    }
    cur_info = scale_info_map.get(preset_var, scale_info_map["Blood Type (O, A, B, AB)"])

    with c2:
        card_html = f"""
        <div style="background-color: {cur_info['bg']}; border-left: 5px solid {cur_info['color']}; border-radius: 8px; padding: 1rem 1.25rem;">
            <div style="font-size: 0.8rem; font-weight: 800; color: {cur_info['color']}; text-transform: uppercase; letter-spacing: 0.05em;">
                Detected Measurement Scale
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #0f172a; margin-top: 0.2rem;">
                {cur_info['scale']}
            </div>
            <div style="font-size: 0.85rem; color: #334155; margin-top: 0.4rem; line-height: 1.45;">
                {cur_info['definition']}
            </div>
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.5rem; border-top: 1px solid rgba(0,0,0,0.08); padding-top: 0.4rem;">
                <strong>Zero Point:</strong> {cur_info['zero_type']}<br>
                <strong>Allowed Operations:</strong> <code>{cur_info['allowed_ops']}</code><br>
                <strong>Forbidden Trap:</strong> <span style="color: #dc2626;">{cur_info['forbidden_ops']}</span>
            </div>
        </div>
        """
        render_html(card_html)

    st.markdown("---")

    # 4-Tab Educational Architecture
    t1, t2, t3, t4 = st.tabs([
        "📋 Rationale & Framework",
        "📗 Microsoft Excel Guide",
        "🐍 Python Code Guide",
        "💾 Practice Data Download"
    ])

    with t1:
        st.markdown(r"""
        #### 1. Why Measurement Scales Dictate Your Analysis
        Stanley Smith Stevens (1946) established the four levels of measurement: **Nominal**, **Ordinal**, **Interval**, and **Ratio** (NOIR).
        Using a statistical method designed for a higher scale on data from a lower scale produces mathematically invalid conclusions.

        * **The "Finishing Position vs. Race Time" Dilemma:**
          Suppose Runner A finishes 1st in 10.0 seconds, Runner B finishes 2nd in 10.1 seconds, and Runner C finishes 3rd in 15.0 seconds.
          * The **Ordinal ranks** are 1, 2, 3 (equal spacing of 1 rank).
          * The **Ratio times** are 10.0s, 10.1s, 15.0s (vastly unequal spacing).
          * Calculating the mean rank tells you nothing about the 4.9-second gap between 2nd and 3rd place!

        #### 2. Mathematical Formulations & Plain English Rules

        ##### A. The Ratio Scale Multiplier Formula
        $$\text{Ratio Comparison} = \frac{x_A}{x_B}$$

        * **Plain English Translation:** *"Dividing one patient's measurement by another to claim 'Patient A has twice as much as Patient B'."*
        * **Symbol-by-Symbol Breakdown:**
          * $x_A$: The numerical measurement of Patient A (*e.g., blood pressure of 160 mmHg, or weight of 80 kg*).
          * $x_B$: The numerical measurement of Patient B (*e.g., blood pressure of 80 mmHg, or weight of 40 kg*).
          * $\text{Ratio Comparison}$: The relative factor connecting the two measurements ($\frac{160}{80} = 2.0$).
        * **The "Why" & Theoretical Requirement:** This division is **only meaningful when zero represents absolute nothingness** (the total absence of the physical quantity). If zero is arbitrary (like $0^\circ\text{C}$, which is merely when water freezes, not when heat ceases to exist), calculating $\frac{20^\circ\text{C}}{10^\circ\text{C}} = 2.0$ is completely false — in absolute thermodynamics ($293.15\text{ K} / 283.15\text{ K}$), $20^\circ\text{C}$ is only 3.5% warmer than $10^\circ\text{C}$!
        * **Permitted Scales:** **Ratio only**. Forbidden on Nominal, Ordinal, and Interval scales.

        ##### B. The Interval Scale Distance Formula
        $$\Delta = x_A - x_B$$

        * **Plain English Translation:** *"Subtracting Patient B's score from Patient A's score to measure the exact physical distance separating them."*
        * **Symbol-by-Symbol Breakdown:**
          * $\Delta$ (Delta): The difference or gap between the two measurements.
          * $x_A, x_B$: The measured values of the two individuals.
        * **The "Why" & Theoretical Requirement:** Subtraction requires **equal spacing between units** throughout the entire scale. A $5^\circ\text{C}$ increase from $35^\circ\text{C}$ to $40^\circ\text{C}$ represents the exact same amount of thermal energy as an increase from $15^\circ\text{C}$ to $20^\circ\text{C}$. On an **Ordinal scale** (*such as Cancer Stages I, II, III, IV*), the disease progression gap between Stage I and Stage II is NOT equal to the gap between Stage III and Stage IV — therefore, calculating differences or averages on ordinal data is mathematically invalid.
        * **Permitted Scales:** **Interval and Ratio**. Forbidden on Nominal and Ordinal scales.

        #### 3. Permitted Statistical Operations by Scale Matrix
        """)
        summary_rows = [
            {"Scale": "Nominal", "Order?": "No", "Equal Spacing?": "No", "True Zero?": "No", "Central Measure": "Mode", "Dispersion": "Frequency %", "Permitted Tests": "Chi-Square, Fisher's Exact"},
            {"Scale": "Ordinal", "Order?": "Yes", "Equal Spacing?": "No", "True Zero?": "No", "Central Measure": "Median", "Dispersion": "Interquartile Range (IQR)", "Permitted Tests": "Mann-Whitney, Kruskal-Wallis, Spearman"},
            {"Scale": "Interval", "Order?": "Yes", "Equal Spacing?": "Yes", "True Zero?": "No (Arbitrary)", "Central Measure": "Mean or Median", "Dispersion": "Standard Deviation", "Permitted Tests": "t-tests, ANOVA, Pearson r"},
            {"Scale": "Ratio", "Order?": "Yes", "Equal Spacing?": "Yes", "True Zero?": "Yes (Absolute)", "Central Measure": "Mean or Median", "Dispersion": "Standard Deviation, CV%", "Permitted Tests": "All Parametric & Non-Parametric"}
        ]
        summary_table = pd.DataFrame(summary_rows) if pd is not None else summary_rows
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

    with t2:
        st.markdown(r"""
        #### 📗 Microsoft Excel Implementation: Data Types & Validation
        In Excel, ensuring correct measurement scale formatting prevents silent calculation errors (e.g. averaging coded numbers `1=Female, 2=Male`):

        ##### Step 1: Cell Coordinate Layout (`A1:E10`)
        ```text
        Row 1: | Patient_ID (A) | Gender_Nominal (B) | NYHA_Ordinal (C) | Body_Temp_C (D) | Systolic_BP (E) |
        Row 2: | PT-101         | Female             | Class II         | 37.2            | 124             |
        Row 3: | PT-102         | Male               | Class I          | 36.8            | 138             |
        ```

        ##### Step 2: Formulas for Nominal & Ordinal Data
        * **Count frequencies for Nominal categories:**
          `=COUNTIF(B2:B50, "Female")`
        * **Calculate frequency proportion (%):**
          `=COUNTIF(B2:B50, "Female") / COUNTA(B2:B50)`
        * **Compute Median for Ordinal numeric ranks (Likert 1–5):**
          `=MEDIAN(C2:C50)`
        * **Compute Interquartile Range (IQR):**
          `=QUARTILE.EXC(C2:C50, 3) - QUARTILE.EXC(C2:C50, 1)`

        ##### Step 3: Formulas for Interval & Ratio Numerical Data
        * **Compute Arithmetic Mean:**
          `=AVERAGE(E2:E50)`
        * **Compute Sample Standard Deviation:**
          `=STDEV.S(E2:E50)`
        """)

    with t3:
        st.markdown(r"""
        #### 🐍 Python Implementation: Enforcing Measurement Scales in Pandas
        ```python
        import pandas as pd
        import numpy as np

        # Load clinical dataset
        df = pd.read_csv("patient_scales_registry.csv")

        # 1. Cast Nominal Variables to category dtype (Unordered)
        df["Gender_Nominal"] = df["Gender_Nominal"].astype("category")

        # 2. Cast Ordinal Variables with explicit clinical hierarchy
        nyha_order = ["Class I (Mild)", "Class II (Moderate)", "Class III (Severe)", "Class IV (End-Stage)"]
        df["NYHA_Class_Ordinal"] = pd.Categorical(df["NYHA_Class_Ordinal"], categories=nyha_order, ordered=True)

        # 3. Appropriate Central Measures based on Scale
        print("Nominal Central Measure (Mode):", df["Gender_Nominal"].mode()[0])
        print("Ordinal Central Measure (Median Rank):", df["NYHA_Class_Ordinal"].mode()[0])
        print("Ratio Central Measure (Mean):", np.mean(df["Systolic_BP_mmHg_Ratio"]))
        print("Ratio Dispersion (Sample SD):", np.std(df["Systolic_BP_mmHg_Ratio"], ddof=1))
        ```
        """)

    with t4:
        df_scales, csv_scales = generate_scales_practice_data()
        st.markdown("#### 💾 Practice Dataset: Clinical Patient Intake Registry")
        st.caption("Demonstrates all 4 measurement scales in a 40-patient cohort.")
        st.dataframe(df_scales.head(10), use_container_width=True)
        st.download_button(
            label="📥 Download Scales Practice Data (CSV)",
            data=csv_scales,
            file_name="patient_measurement_scales_data.csv",
            mime="text/csv",
            key="dl_btn_scales"
        )


# =====================================================================
# 2. Measures of Central Tendency Module
# =====================================================================

def render_central_tendency_module() -> None:
    """Renders Module 2: Measures of Central Tendency."""
    st.markdown("## ⚖️ Measures of Central Tendency")
    st.markdown("""
    Measures of central tendency attempt to describe an entire distribution of numbers with a single central value.
    The crucial decision is choosing between the **Mean** (center of gravity) and the **Median** (middle rank).
    """)

    # Interactive Outlier Pull Simulator
    st.markdown("### 🎮 Interactive Outlier Pull Simulator")
    st.caption("Adjust the outlier slider below to see how a single prolonged hospital stay distorts the Mean while the Median stands firm.")

    c1, c2 = st.columns([1.5, 2.5])
    with c1:
        base_stay = st.slider("Typical Surgery Stays (N = 40): Mean approx (Days)", min_value=2.0, max_value=8.0, value=3.5, step=0.5)
        outlier_days = st.slider("Inject Extreme ICU Outlier Stay (Days):", min_value=10.0, max_value=180.0, value=90.0, step=10.0)
        inject_toggle = st.checkbox("Include Outlier in Sample", value=True)

    # Build simulated dataset
    if HAS_SCIENTIFIC_STACK and np is not None:
        rng = np.random.RandomState(42)
        sample_vals = list(np.round(rng.lognormal(mean=np.log(base_stay), sigma=0.35, size=40), 1))
    else:
        import random
        rng_py = random.Random(42)
        log_mean = math.log(max(base_stay, 0.1))
        sample_vals = [round(rng_py.lognormvariate(log_mean, 0.35), 1) for _ in range(40)]

    if inject_toggle:
        sample_vals.append(float(outlier_days))

    res = calculate_central_tendency(sample_vals)

    with c2:
        # KPI metrics comparison
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Arithmetic Mean", f"{res['mean']:.2f} d", delta=f"{res['mean'] - res['median']:+.2f} d vs Median" if inject_toggle else None)
        with k2:
            st.metric("Median (50th %ile)", f"{res['median']:.2f} d", help="Robust middle observation")
        with k3:
            st.metric("10% Trimmed Mean", f"{res['trimmed_mean']:.2f} d", help="Discards top and bottom 10% of observations")

        # Plotly Histogram & Density
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=sample_vals,
            nbinsx=25,
            name="Patient Stays",
            marker_color="#93c5fd",
            opacity=0.75
        ))
        fig.add_vline(x=res["mean"], line_width=3, line_dash="solid", line_color="#dc2626", annotation_text=f"Mean: {res['mean']:.1f}d")
        fig.add_vline(x=res["median"], line_width=3, line_dash="dash", line_color="#16a34a", annotation_text=f"Median: {res['median']:.1f}d")
        fig.update_layout(
            title="Distribution of Hospital Length of Stay (Days)",
            xaxis_title="Stay Duration (Days)",
            yaxis_title="Patient Count",
            height=280,
            margin=dict(l=20, r=20, t=35, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    if res["is_noticeably_skewed"]:
        st.warning(f"⚠️ **Severe Skewness Detected:** The Mean ({res['mean']:.2f} days) is pulled {res['relative_shift']:.2f} standard deviations away from the Median ({res['median']:.2f} days) by the outlier. **Under ICMJE medical guidelines, you must report Median & IQR rather than Mean & SD.**")

    st.markdown("---")

    # 4-Tab Educational Architecture
    t1, t2, t3, t4 = st.tabs([
        "📋 Rationale & Framework",
        "📗 Microsoft Excel Guide",
        "🐍 Python Code Guide",
        "💾 Practice Data Download"
    ])

    with t1:
        st.markdown(r"""
        #### 1. Center of Gravity vs. Middle Rank: The Core Formulas

        ##### A. The Arithmetic Mean Formula ($\bar{x}$)
        $$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i = \frac{x_1 + x_2 + x_3 + \dots + x_n}{n}$$

        * **Plain English Translation:** *"Add up every single patient's measurement from first to last, then divide the total sum equally by the number of patients."*
        * **Symbol-by-Symbol Breakdown:**
          * $\bar{x}$ (pronounced *"x-bar"*): The calculated sample mean. Represents the mathematical center of gravity.
          * $\sum_{i=1}^n$ (Greek capital letter *Sigma*): A mathematical shorthand instruction meaning *"sum every term starting at patient $i=1$ and finishing at patient $n$"*.
          * $x_i$: The recorded measurement of the $i$-th patient (*e.g., Patient 1 stayed 3 days, Patient 2 stayed 4 days*).
          * $n$: The total sample size (the total number of patients enrolled).
          * $\frac{1}{n}$: Dividing by $n$ distributes the grand total evenly across all individuals.
        * **The "Why" & Physical Intuition:** Imagine placing all patient values along a balance seesaw. The mean is the exact fulcrum point where the seesaw rests horizontally without tilting. Because every single value enters the numerator, a single extreme outlier far out on the board (*e.g., a 150-day ICU stay*) exerts massive mechanical leverage, dragging the mean far to the right.

        ##### B. The Sample Median Formula ($\tilde{x}$ / $Q_2$)
        $$\text{Sorted Array: } x_{(1)} \le x_{(2)} \le \dots \le x_{(n)}, \quad \text{Middle Rank Position} = \frac{n + 1}{2}$$
        $$\tilde{x} = \begin{cases} x_{\left(\frac{n+1}{2}\right)} & \text{if } n \text{ is odd} \\[8pt] \dfrac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2} + 1\right)}}{2} & \text{if } n \text{ is even} \end{cases}$$

        * **Plain English Translation:** *"Line up all patients from lowest to highest. The median is the physical value sitting exactly in the middle seat, with 50% of patients below it and 50% of patients above it."*
        * **Symbol-by-Symbol Breakdown:**
          * $\tilde{x}$ (pronounced *"x-tilde"*): The sample median (also known as the 50th percentile or second quartile $Q_2$).
          * $x_{(1)}, x_{(2)}, \dots, x_{(n)}$: The patient values after being sorted in ascending numerical order (order statistics).
          * $\frac{n+1}{2}$: The positional index of the middle patient. If $n = 9$, position is $(9+1)/2 = 5$ (the 5th patient). If $n = 10$, position is $(10+1)/2 = 5.5$, meaning we average the 5th and 6th patients.
        * **The "Why" & Physical Intuition:** The median tracks rank order rather than numerical sum. If the sickest patient in your surgical ward stays 15 days, 50 days, or 500 days, the patient sitting in the middle rank does not move by even one millimeter. This makes the median immune to outliers.

        ##### C. The Sample Mode
        $$\text{Mode} = \operatorname{arg\,max}_{v} \, \operatorname{Count}(x_i = v)$$

        * **Plain English Translation:** *"The single most common measurement that appears with the highest frequency in your dataset."*
        * **Symbol-by-Symbol Breakdown:**
          * $\text{Count}(x_i = v)$: The number of times a specific value $v$ appears.
          * $\operatorname{arg\,max}$: Select the value that produces the maximum count.
        * **When to Use:** Essential for categorical/nominal data (*e.g., most common admitting diagnosis = "Chest Pain"*). In continuous measurements (*e.g., blood pressure to 2 decimal places*), true duplicates are rare unless data is binned into intervals.

        ##### D. The 10% Trimmed Mean Formula ($\bar{x}_{\text{trim}}$)
        $$\bar{x}_{\text{trim}} = \frac{1}{n - 2k} \sum_{i=k+1}^{n - k} x_{(i)}, \quad \text{where } k = \operatorname{round}(n \times 0.10)$$

        * **Plain English Translation:** *"Chop off the bottom 10% and top 10% of patients, then calculate the regular average of everyone left in the middle."*
        * **Symbol-by-Symbol Breakdown:**
          * $x_{(i)}$: The patient values sorted from lowest to highest.
          * $k$: The number of extreme observations cut off from each tail (10% of total sample size $n$).
          * $\sum_{i=k+1}^{n-k}$: Sum only the remaining middle 80% of patients, ignoring the $k$ lowest and $k$ highest.
          * $n - 2k$: The count of patients remaining after trimming $k$ from the bottom and $k$ from the top.
        * **The "Why" & Physical Intuition:** Often called the *"Olympic diving score"* method. It provides the mathematical efficiency and precision of a mean, but shields researchers from ICU outliers, measurement artifacts, or data-entry errors.

        #### 2. The 0.20 SD Skew Pull Rule (Objective Reporting Criterion)

        $$\text{Relative Skew Shift} = \frac{|\bar{x} - \tilde{x}|}{s}$$

        * **Plain English Translation:** *"Measure the physical distance between the mean and the median, and divide it by the standard deviation ruler."*
        * **Symbol-by-Symbol Breakdown:**
          * $|\bar{x} - \tilde{x}|$: The absolute distance (gap) in original clinical units (*e.g., days or mmHg*) between the sample average and sample median.
          * $s$: The sample standard deviation (the benchmark measure of natural patient variability).
          * $\text{Relative Skew Shift}$: Expresses the skew pull as a percentage of the standard deviation.
        * **Clinical Decision Standard (ICMJE / CONSORT):**
          * **Shift $\le 0.10$ ($10\%$ of SD):** **Symmetric Distribution.** The mean and median agree. Report **$\text{Mean} \pm \text{SD}$**.
          * **Shift $> 0.20$ ($20\%$ of SD):** **Noticeably Skewed Distribution.** The mean has been substantially dragged by extreme values. Reporting the mean would mislead clinicians about typical patient prognosis. You must report **$\text{Median [IQR]}$**.
        """)

    with t2:
        st.markdown(r"""
        #### 📗 Microsoft Excel Implementation: Central Tendency
        ##### Step 1: Layout (`A1:A50`)
        ```text
        A1: Length_of_Stay_Days
        A2: 3.2
        A3: 2.8
        ...
        A42: 90.0 (ICU Outlier)
        ```

        ##### Step 2: Formulas
        * **Arithmetic Mean:**
          `=AVERAGE(A2:A42)`
        * **Median (50th Percentile):**
          `=MEDIAN(A2:A42)`
        * **Mode:**
          `=MODE.MULT(A2:A42)`
        * **10% Trimmed Mean (discards top/bottom 10%):**
          `=TRIMMEAN(A2:A42, 0.20)` *(Note: percent argument is total % excluded across both tails).*
        * **Automated Skewness Flag in Excel:**
          `=IF(ABS(AVERAGE(A2:A42) - MEDIAN(A2:A42)) > 0.2 * STDEV.S(A2:A42), "Report Median [IQR]", "Report Mean ± SD")`
        """)

    with t3:
        st.markdown(r"""
        #### 🐍 Python Implementation: Central Tendency & Outlier Robustness
        ```python
        import numpy as np
        import scipy.stats as stats

        # Sample hospital length of stay
        stays = np.array([2.5, 3.1, 4.0, 2.8, 3.5, 1.9, 4.2, 3.8, 90.0])

        mean_val = np.mean(stays)
        median_val = np.median(stays)
        std_val = np.std(stays, ddof=1)
        trimmed_mean = stats.trim_mean(stays, proportiontocut=0.10)

        print(f"Mean: {mean_val:.2f} days")
        print(f"Median: {median_val:.2f} days")
        print(f"10% Trimmed Mean: {trimmed_mean:.2f} days")

        # Clinical Reporting Heuristic
        shift = abs(mean_val - median_val) / std_val
        if shift > 0.20:
            print(f"[ICMJE RECOMMENDATION] Report Median ({median_val:.1f}d) & IQR. Distribution is skewed.")
        else:
            print(f"[ICMJE RECOMMENDATION] Report Mean ({mean_val:.1f}d) ± SD ({std_val:.1f}d).")
        ```
        """)

    with t4:
        df_los, csv_los = generate_central_tendency_practice_data(inject_extreme=True)
        st.markdown("#### 💾 Practice Dataset: Hospital Length of Stay (LOS)")
        st.caption("Contains 50 inpatient admissions including prolonged intensive care stays.")
        st.dataframe(df_los.head(10), use_container_width=True)
        st.download_button(
            label="📥 Download Central Tendency Practice Data (CSV)",
            data=csv_los,
            file_name="hospital_length_of_stay_data.csv",
            mime="text/csv",
            key="dl_btn_los"
        )


# =====================================================================
# 3. Measures of Dispersion & Tukey's Fences Module
# =====================================================================

def render_dispersion_iqr_module() -> None:
    """Renders Module 3: Measures of Dispersion, Spread & Outlier Rules."""
    st.markdown("## 📏 Measures of Dispersion & Tukey's Outlier Fences")
    st.markdown("""
    Central tendency tells you where the center is; **dispersion** tells you whether the data clusters tightly 
    or scatters wildly. It defines the difference between predictable stability and extreme clinical volatility.
    """)

    # Interactive Box-Plot & Outlier Builder
    st.markdown("### 🎮 Interactive Box-and-Whisker & Outlier Builder")
    st.caption("Adjust sample variance and inspect Tukey's inner (1.5x) and outer (3.0x) outlier fences dynamically.")

    c1, c2 = st.columns([1.5, 2.5])
    with c1:
        base_chol = st.slider("Mean Serum Cholesterol (mmol/L):", min_value=3.5, max_value=7.0, value=5.10, step=0.10)
        spread_sd = st.slider("Biological Standard Deviation (s):", min_value=0.30, max_value=2.00, value=0.85, step=0.05)
        outlier_opt = st.radio("Outlier Injection:", options=["No Outliers (Clean)", "Mild Outliers (+1.8x IQR)", "Extreme Outliers (+3.5x IQR)"])

    # Generate data
    if HAS_SCIENTIFIC_STACK and np is not None:
        rng = np.random.RandomState(101)
        sample_vals = list(np.round(rng.normal(loc=base_chol, scale=spread_sd, size=46), 2))
    else:
        import random
        rng_py = random.Random(101)
        sample_vals = [round(rng_py.gauss(base_chol, spread_sd), 2) for _ in range(46)]
    if outlier_opt == "Mild Outliers (+1.8x IQR)":
        sample_vals.extend([base_chol + 2.5 * spread_sd, base_chol - 2.5 * spread_sd])
    elif outlier_opt == "Extreme Outliers (+3.5x IQR)":
        sample_vals.extend([base_chol + 4.2 * spread_sd, base_chol + 4.8 * spread_sd])

    disp = calculate_dispersion_and_quartiles(sample_vals)

    with c2:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Sample SD (s)", f"{disp['std_dev']:.2f}")
        with k2:
            st.metric("Q1 (25th %ile)", f"{disp['q1']:.2f}")
        with k3:
            st.metric("Median (Q2)", f"{disp['median']:.2f}")
        with k4:
            st.metric("Q3 (75th %ile)", f"{disp['q3']:.2f}")

        # Plotly Box Plot
        fig = go.Figure()
        fig.add_trace(go.Box(
            x=sample_vals,
            name="Serum Cholesterol",
            boxpoints="outliers",
            jitter=0.3,
            pointpos=-1.8,
            marker_color="#2563eb",
            line_color="#1d4ed8"
        ))
        fig.add_vline(x=disp["tukey_inner_upper"], line_dash="dash", line_color="#d97706", annotation_text=f"Inner Fence: {disp['tukey_inner_upper']:.2f}")
        fig.add_vline(x=disp["tukey_inner_lower"], line_dash="dash", line_color="#d97706", annotation_text=f"Inner Fence: {disp['tukey_inner_lower']:.2f}")
        fig.update_layout(
            title="Box-and-Whisker Plot with Tukey Outlier Fences",
            xaxis_title="Serum Cholesterol (mmol/L)",
            height=260,
            margin=dict(l=20, r=20, t=35, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"📊 **Interquartile Range ($\text{{IQR}}$):** `{disp['iqr']:.2f} mmol/L` | **Mild Outliers:** `{len(disp['mild_outliers'])}` | **Extreme Outliers:** `{len(disp['extreme_outliers'])}`")

    st.markdown("---")

    # 4-Tab Educational Architecture
    t1, t2, t3, t4 = st.tabs([
        "📋 Rationale & Framework",
        "📗 Microsoft Excel Guide",
        "🐍 Python Code Guide",
        "💾 Practice Data Download"
    ])

    with t1:
        st.markdown(r"""
        #### 1. Measuring Spread: The Core Dispersion Formulas

        ##### A. The Sample Variance Formula ($s^2$) & Bessel's Correction
        $$s^2 = \frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2 = \frac{(x_1 - \bar{x})^2 + (x_2 - \bar{x})^2 + \dots + (x_n - \bar{x})^2}{n - 1}$$

        * **Plain English Translation:** *"Find how far each patient's score sits from the group average. Square each distance so negative gaps become positive. Add all those squared gaps together, and divide by the number of patients minus one."*
        * **Symbol-by-Symbol Breakdown:**
          * $s^2$: The sample variance (spread measured in awkward squared units, such as $\text{days}^2$ or $\text{mmHg}^2$).
          * $x_i - \bar{x}$: The individual deviation (gap). Patients below average have negative deviations; patients above average have positive deviations.
          * $(x_i - \bar{x})^2$: The squared deviation. Squaring accomplishes two vital tasks:
            1. It turns all negative distances positive so they don't cancel each other out to zero ($\sum(x_i - \bar{x}) \equiv 0$).
            2. It penalizes large deviations much more heavily than small ones.
          * $\sum_{i=1}^n$: Sum across all $n$ patients in your cohort.
          * $n - 1$ (**Bessel's Correction**): Divide by $n-1$ instead of $n$.
        * **The "Why" Behind Bessel's Correction ($n - 1$):** Because we don't know the unobservable true population mean $\mu$, we calculate deviations from our own sample mean $\bar{x}$. But sample points naturally sit closer to their own average than to the universe average. Dividing by $n$ produces an estimate that is systematically **too small** (biased downwards). Dividing by the slightly smaller denominator $n - 1$ inflates the fraction just enough to make sample variance an **unbiased** estimate of the population!

        ##### B. The Sample Standard Deviation Formula ($s$)
        $$s = \sqrt{s^2} = \sqrt{\frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2}$$

        * **Plain English Translation:** *"Take the square root of the variance to translate the spread back into original, physical clinical units."*
        * **Symbol-by-Symbol Breakdown:**
          * $s$: The sample standard deviation. Expressed in the exact same physical units as the original raw data (*e.g., mmol/L, days, or mmHg*).
          * $\sqrt{\dots}$: The square root reverses the mathematical squaring operation performed during the variance step.
        * **The Empirical Rule (The 68–95–99.7 Rule for Normal Data):**
          * $\bar{x} \pm 1s$: Captures approximately **$68.3\%$** of all patients.
          * $\bar{x} \pm 2s$: Captures approximately **$95.4\%$** of all patients.
          * $\bar{x} \pm 3s$: Captures approximately **$99.7\%$** of all patients (almost everyone).

        ##### C. Quartiles & The Interquartile Range Formula ($\text{IQR}$)
        $$\text{IQR} = Q_3 - Q_1$$

        * **Plain English Translation:** *"Subtract the 25th percentile mark from the 75th percentile mark to measure the exact physical width of the middle 50% of patients."*
        * **Symbol-by-Symbol Breakdown:**
          * $Q_1$ (First Quartile / 25th Percentile): The value below which the lowest $25\%$ of patients sit.
          * $Q_3$ (Third Quartile / 75th Percentile): The value below which $75\%$ of patients sit (meaning $25\%$ sit above it).
          * $\text{IQR}$: The distance spanning the middle half of your sample.
        * **The "Why" & Robustness:** Even if your sample includes an extreme outlier (*e.g., one ICU patient staying 300 days*), that outlier is in the top 1% and is completely ignored by $Q_3$. The $\text{IQR}$ provides an unshakeable, outlier-proof measure of dispersion.

        #### 2. Tukey's Outlier Boundary Fences

        ##### A. Mild Outlier Fences ($1.5 \times \text{IQR}$)
        $$\text{Lower Inner Fence} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Inner Fence} = Q_3 + 1.5 \times \text{IQR}$$

        * **Plain English Translation:** *"Take the width of the middle 50% ($\text{IQR}$), multiply it by one-and-a-half, and step that far beyond the first and third quartiles. Any point outside these posts is a mild outlier."*
        * **Why 1.5?:** John Tukey calculated that for a normal bell curve, stepping $1.5 \times \text{IQR}$ beyond the quartiles lands at $\pm 2.698 \times \sigma$, covering $99.3\%$ of the distribution. Only $0.7\%$ of normal observations (about 1 in 150) fall outside by pure chance.

        ##### B. Extreme Outlier Fences ($3.0 \times \text{IQR}$)
        $$\text{Lower Outer Fence} = Q_1 - 3.0 \times \text{IQR}, \quad \text{Upper Outer Fence} = Q_3 + 3.0 \times \text{IQR}$$

        * **Plain English Translation:** *"Multiply the middle 50% width by three, and step that far beyond the quartiles. Any point outside is an extreme outlier."*
        * **Why 3.0?:** Stepping $3.0 \times \text{IQR}$ beyond the quartiles lands at $\pm 4.718 \times \sigma$. In a normal distribution, only **1 in 500,000 observations** occurs this far out!
        * **Clinical Meaning:** Any observation beyond $3 \times \text{IQR}$ is almost certainly a data-entry error (*e.g., typing 145 instead of 14.5*), an assay malfunction, or severe rare pathology. It severely distorts means and warrants non-parametric testing.
        """)

    with t2:
        st.markdown(r"""
        #### 📗 Microsoft Excel Implementation: Dispersion & Outliers
        ##### Step 1: Formulas in Excel
        * **Sample Variance (ddof = 1):**
          `=VAR.S(A2:A50)` *(Never use VAR.P for research samples!)*
        * **Sample Standard Deviation:**
          `=STDEV.S(A2:A50)`
        * **First Quartile ($Q_1$, 25th percentile):**
          `=QUARTILE.EXC(A2:A50, 1)`
        * **Third Quartile ($Q_3$, 75th percentile):**
          `=QUARTILE.EXC(A2:A50, 3)`
        * **Interquartile Range ($\text{IQR}$):**
          `=QUARTILE.EXC(A2:A50, 3) - QUARTILE.EXC(A2:A50, 1)`
        * **Upper Tukey Outlier Fence:**
          `=QUARTILE.EXC(A2:A50, 3) + 1.5 * (QUARTILE.EXC(A2:A50, 3) - QUARTILE.EXC(A2:A50, 1))`
        * **Outlier Flag Formula:**
          `=IF(OR(A2 < ($E$1 - 1.5*$E$3), A2 > ($E$2 + 1.5*$E$3)), "OUTLIER", "NORMAL")`
        """)

    with t3:
        st.markdown(r"""
        #### 🐍 Python Implementation: Tukey Fences Filter
        ```python
        import numpy as np

        data = np.array([4.2, 5.1, 4.8, 5.3, 4.9, 5.0, 5.2, 8.95, 2.10])

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        inner_lower = q1 - 1.5 * iqr
        inner_upper = q3 + 1.5 * iqr
        outer_lower = q1 - 3.0 * iqr
        outer_upper = q3 + 3.0 * iqr

        mild_outliers = data[(data < inner_lower) | (data > inner_upper)]
        extreme_outliers = data[(data < outer_lower) | (data > outer_upper)]

        print(f"IQR: {iqr:.2f}")
        print(f"Mild Outliers (1.5x IQR): {mild_outliers}")
        print(f"Extreme Outliers (3.0x IQR): {extreme_outliers}")
        ```
        """)

    with t4:
        df_chol, csv_chol = generate_dispersion_practice_data()
        st.markdown("#### 💾 Practice Dataset: Serum Cholesterol Assay Values")
        st.caption("Contains 50 laboratory assay values with simulated clinical variance.")
        st.dataframe(df_chol.head(10), use_container_width=True)
        st.download_button(
            label="📥 Download Dispersion Practice Data (CSV)",
            data=csv_chol,
            file_name="serum_cholesterol_dispersion_data.csv",
            mime="text/csv",
            key="dl_btn_chol"
        )


# =====================================================================
# 4. Standard Deviation (SD) vs. Standard Error (SEM) Module
# =====================================================================

def render_sd_vs_sem_module() -> None:
    """Renders Module 4: Standard Deviation (SD) vs. Standard Error (SEM)."""
    st.markdown("## 🔬 Standard Deviation (SD) vs. Standard Error of the Mean (SEM)")
    st.markdown("""
    Confusing **SD** and **SEM** is one of the most common errors in biomedical research.
    They answer two completely different scientific questions.
    """)

    # Interactive N-Scaling Simulator
    st.markdown("### 🎮 Interactive N-Scaling Precision Simulator")
    st.caption("Drag the Sample Size slider from N = 10 to N = 500: observe that SD remains stable around true patient variability, while SEM collapses toward zero.")

    c1, c2 = st.columns([1.5, 2.5])
    with c1:
        n_slider = st.slider("Sample Size (N):", min_value=10, max_value=500, value=30, step=10)
        pop_sd = st.slider("True Population Variability (σ):", min_value=5.0, max_value=30.0, value=15.0, step=1.0)

    # Generate sample of size N
    if HAS_SCIENTIFIC_STACK and np is not None:
        rng = np.random.RandomState(42)
        sample_data = list(rng.normal(loc=135.0, scale=pop_sd, size=n_slider))
    else:
        import random
        rng_py = random.Random(42)
        sample_data = [rng_py.gauss(135.0, pop_sd) for _ in range(n_slider)]
    sem_res = calculate_sd_vs_sem(sample_data)

    with c2:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Sample SD (s)", f"{sem_res['std_dev']:.1f}", help="Quantifies patient-to-patient biological spread. Stays constant!")
        with m2:
            st.metric("Standard Error (SEM)", f"{sem_res['sem']:.2f}", help="Quantifies precision of the mean estimate. Shrinks with sqrt(N)!")
        with m3:
            st.metric("SEM as % of SD", f"{sem_res['sem_as_pct_of_sd']:.1f}%")

        # Plotly side-by-side comparison chart
        ns_range = [10, 20, 30, 50, 100, 200, 300, 500]
        sems = [pop_sd / math.sqrt(n) for n in ns_range]
        sds = [pop_sd for _ in ns_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ns_range, y=sds, mode="lines", name="Standard Deviation (SD)", line=dict(color="#2563eb", width=3, dash="dash")))
        fig.add_trace(go.Scatter(x=ns_range, y=sems, mode="lines+markers", name="Standard Error (SEM)", line=dict(color="#dc2626", width=3)))
        fig.add_vline(x=n_slider, line_color="#10b981", line_width=2, annotation_text=f"Active N = {n_slider}")
        fig.update_layout(
            title="How SD (Patient Spread) vs. SEM (Precision) React as N Increases",
            xaxis_title="Sample Size (N)",
            yaxis_title="Units (mmHg)",
            height=260,
            margin=dict(l=20, r=20, t=35, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-left: 4px solid #ef4444; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.5rem; font-size: 0.88rem; color: #991b1b;">
        🚨 <strong>The Biomedical Deception Trap:</strong> In scientific papers, authors sometimes report <code>Mean ± SEM</code> instead of <code>Mean ± SD</code> simply because the SEM is smaller, making their experimental bars look tighter. Under CONSORT guidelines, you must report <strong>SD</strong> when describing descriptive baseline characteristics!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 4-Tab Educational Architecture
    t1, t2, t3, t4 = st.tabs([
        "📋 Rationale & Framework",
        "📗 Microsoft Excel Guide",
        "🐍 Python Code Guide",
        "💾 Practice Data Download"
    ])

    with t1:
        st.markdown(r"""
        #### 1. The Core Formulations: Biological Reality vs. Statistical Precision

        ##### A. The Standard Error of the Mean Formula ($\text{SEM}$)
        $$\text{SEM} = \frac{s}{\sqrt{n}} = \frac{\sqrt{\dfrac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2}}{\sqrt{n}}$$

        * **Plain English Translation:** *"Divide the natural patient-to-patient spread ($s$) by the square root of the number of enrolled patients ($\sqrt{n}$)."*
        * **Symbol-by-Symbol Breakdown:**
          * $\text{SEM}$ (Standard Error of the Mean): A measure of **estimation uncertainty**. It tells you how much the sample average $\bar{x}$ would bounce around if you repeated the entire clinical study dozens of times with fresh batches of $n$ patients.
          * $s$ (Sample Standard Deviation): A measure of **biological spread**. It quantifies how much individual human beings differ from one another (*e.g., some patients have blood pressure of 110, others 160*).
          * $n$: The total number of patients enrolled in your study.
          * $\sqrt{n}$: The square root of the sample size.
        * **The "Why": The Square-Root Law of Diminishing Returns:**
          * Why does $n$ sit inside a square root ($\sqrt{n}$) instead of just dividing by $n$? Because random errors in opposite directions partially cancel each other out when averaged.
          * **The 4× Investment Rule:** To cut your estimation uncertainty in half ($\frac{1}{2}$), you cannot just double your sample size — you must recruit **$4\times$ as many patients** ($\sqrt{4} = 2$).
          * **The 100× Investment Rule:** To reduce your margin of error to one-tenth ($\frac{1}{10}$), you must recruit **$100\times$ as many patients** ($\sqrt{100} = 10$)!

        ##### B. The 95% Estimation Margin of Error Formula
        $$\text{Margin of Error}_{95\%} = z_{0.975} \times \text{SEM} \approx 1.960 \times \frac{s}{\sqrt{n}}$$

        * **Plain English Translation:** *"Multiply the standard error by 1.96 to establish the plus-or-minus boundary covering 95% of possible sample means."*
        * **Symbol-by-Symbol Breakdown:**
          * $1.960$: The critical normal multiplier containing $95\%$ of the area under a standard Gaussian curve.
          * $\text{SEM} = \frac{s}{\sqrt{n}}$: The standard error.
          * Result: $\bar{x} \pm \text{Margin of Error}$ forms the 95% Confidence Interval for the mean.

        #### 2. Clinical Reporting Standard: When to Use Which?

        | Metric | What It Answers | Clinical Meaning | Where It Belongs in a Paper |
        | :--- | :--- | :--- | :--- |
        | **Standard Deviation ($s$)** | *"How variable are my patients?"* | Biological diversity. Enrolling 10,000 patients will **not** change $s$ because humans stay just as physiologically diverse. | **Table 1 (Baseline Demographics):** *"Participants had a baseline age of $58.2 \pm 9.4$ years."* |
        | **Standard Error ($\text{SEM}$)** | *"How precise is my average?"* | Scientific measurement uncertainty. Enrolling 10,000 patients collapses $\text{SEM}$ toward zero. | **Results & Discussion (Treatment Effects):** *"Drug A reduced SBP by $8.4\text{ mmHg}$ (SEM $0.62$; 95% CI: $7.2$ to $9.6$)."* |
        """)

    with t2:
        st.markdown(r"""
        #### 📗 Microsoft Excel Implementation: SD vs. SEM
        ##### Step 1: Formulas in Excel
        ```text
        A1: Systolic_BP
        A2: 138.5
        ...
        A61: 142.0 (N = 60)
        ```
        * **Compute Sample SD:**
          `=STDEV.S(A2:A61)`
        * **Compute Sample Size N:**
          `=COUNT(A2:A61)`
        * **Compute Standard Error (SEM):**
          `=STDEV.S(A2:A61) / SQRT(COUNT(A2:A61))`
        * **Compute 95% Error Bar Margin:**
          `=1.96 * (STDEV.S(A2:A61) / SQRT(COUNT(A2:A61)))`
        """)

    with t3:
        st.markdown(r"""
        #### 🐍 Python Implementation: SD vs. SEM
        ```python
        import numpy as np
        import scipy.stats as stats

        bp_data = np.array([135, 142, 128, 150, 138, 144, 132, 140])
        n = len(bp_data)

        mean_val = np.mean(bp_data)
        sd_val = np.std(bp_data, ddof=1)
        sem_val = stats.sem(bp_data)  # Equivalently: sd_val / np.sqrt(n)

        print(f"Descriptive Patient Variation: Mean = {mean_val:.1f} ± {sd_val:.1f} mmHg (SD)")
        print(f"Estimation Precision: Mean = {mean_val:.1f} mmHg [SEM = {sem_val:.2f}]")
        print(f"95% Confidence Interval: [{mean_val - 1.96*sem_val:.1f}, {mean_val + 1.96*sem_val:.1f}]")
        ```
        """)

    with t4:
        df_sd, csv_sd = generate_sd_sem_practice_data()
        st.markdown("#### 💾 Practice Dataset: Clinical Trial Baseline Demographics")
        st.caption("Contains 60 participant records with blood pressure and heart rate measurements.")
        st.dataframe(df_sd.head(10), use_container_width=True)
        st.download_button(
            label="📥 Download SD vs SEM Practice Data (CSV)",
            data=csv_sd,
            file_name="clinical_trial_baseline_sd_sem_data.csv",
            mime="text/csv",
            key="dl_btn_sd_sem"
        )


# =====================================================================
# 5. Distribution Shapes, Skewness & Central Limit Theorem (CLT)
# =====================================================================

def render_distributions_clt_module() -> None:
    """Renders Module 5: Distribution Shapes, Skewness & Central Limit Theorem."""
    st.markdown("## 🔔 Distribution Shapes, Skewness & Central Limit Theorem (CLT)")
    st.markdown("""
    The **Central Limit Theorem (CLT)** is the mathematical bedrock of all parametric statistical inference.
    It explains why we can use $t$-tests and ANOVA even when raw patient data is heavily skewed.
    """)

    # Interactive CLT Simulator
    st.markdown("### 🎮 Interactive Central Limit Theorem Simulator")
    st.caption("Observe an extremely skewed raw distribution (e.g. ED wait times) transform into a clean Gaussian bell curve as you increase sample size per batch.")

    c1, c2 = st.columns([1.5, 2.5])
    with c1:
        dist_choice = st.selectbox("Underlying Population Shape:", options=["Exponential (Heavy Right Skew)", "Uniform (Flat Dice)", "Bimodal (Two Peaks)"])
        batch_n = st.slider("Sample Size per Draw (N):", min_value=1, max_value=60, value=30, step=1)
        n_draws = st.slider("Number of Resamples:", min_value=100, max_value=1000, value=500, step=100)

    # Generate population
    if HAS_SCIENTIFIC_STACK and np is not None:
        rng = np.random.RandomState(42)
        if dist_choice == "Exponential (Heavy Right Skew)":
            pop_raw = list(rng.exponential(scale=40.0, size=2000) + 10.0)
        elif dist_choice == "Uniform (Flat Dice)":
            pop_raw = list(rng.uniform(low=1.0, high=6.0, size=2000))
        else:
            pop_raw = list(np.concatenate([rng.normal(loc=20.0, scale=3.0, size=1000), rng.normal(loc=55.0, scale=5.0, size=1000)]))
    else:
        import random
        rng_py = random.Random(42)
        if dist_choice == "Exponential (Heavy Right Skew)":
            pop_raw = [rng_py.expovariate(1.0 / 40.0) + 10.0 for _ in range(2000)]
        elif dist_choice == "Uniform (Flat Dice)":
            pop_raw = [rng_py.uniform(1.0, 6.0) for _ in range(2000)]
        else:
            p1 = [rng_py.gauss(20.0, 3.0) for _ in range(1000)]
            p2 = [rng_py.gauss(55.0, 5.0) for _ in range(1000)]
            pop_raw = p1 + p2

    clt_res = calculate_distribution_shape_and_clt(list(pop_raw), n_sim_samples=n_draws, sim_sample_size=batch_n)

    with c2:
        # Plot dual subplots: Raw Population vs Sampling Distribution of Means
        f1, f2 = st.columns(2)
        with f1:
            fig_raw = go.Figure()
            fig_raw.add_trace(go.Histogram(x=pop_raw, nbinsx=30, marker_color="#cbd5e1", name="Raw Population"))
            fig_raw.update_layout(title="Raw Population Data", height=230, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_raw, use_container_width=True)
            st.caption(f"Raw Skewness: `{clt_res['skewness']:.2f}` ({clt_res['shape_label']})")

        with f2:
            fig_clt = go.Figure()
            fig_clt.add_trace(go.Histogram(x=clt_res["sim_means"], nbinsx=25, marker_color="#10b981", name="Sample Means"))
            fig_clt.update_layout(title=f"Sample Means Distribution (N = {batch_n})", height=230, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_clt, use_container_width=True)
            st.caption(f"Mean of Means: `{clt_res['sim_mean_of_means']:.1f}` | Theoretical SEM: `{clt_res['theoretical_sem']:.2f}`")

    if batch_n >= 30:
        st.success(rf"✅ **Central Limit Theorem Activated ($N = {batch_n} \ge 30$):** The distribution of the sample mean is approximately normal regardless of raw skewness. Parametric tests ($t$-tests, ANOVA) are statistically valid.")
    else:
        st.warning(f"⚠️ **Small Sample ($N = {batch_n} < 30$):** The Central Limit Theorem does not provide sufficient normality protection. If raw data is skewed, you must use non-parametric tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis).")

    st.markdown("---")

    # 4-Tab Educational Architecture
    t1, t2, t3, t4 = st.tabs([
        "📋 Rationale & Framework",
        "📗 Microsoft Excel Guide",
        "🐍 Python Code Guide",
        "💾 Practice Data Download"
    ])

    with t1:
        st.markdown(r"""
        #### 1. The Central Limit Theorem (CLT) Formulation

        $$\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right), \quad \text{as } n \ge 30$$

        * **Plain English Translation:** *"Regardless of what shape the raw individual patient data has (even if heavily skewed, exponential, or bimodal), the average of a sample batch will follow a symmetric Gaussian bell curve centered at the true population mean $\mu$, with spread shrinking as sample size $n$ grows."*
        * **Symbol-by-Symbol Breakdown:**
          * $\bar{X}$: The sample mean calculated from a random batch of $n$ patients.
          * $\sim$: Mathematical notation meaning *"follows the probability distribution of..."*
          * $\mathcal{N}$: The Gaussian normal distribution (the classic symmetrical bell-shaped curve).
          * $\mu$: The true unobservable center of the overall population. The center of the sampling distribution lands exactly on $\mu$.
          * $\frac{\sigma^2}{n}$: The variance of the sample means. Taking the square root gives the Standard Error ($\text{SEM} = \frac{\sigma}{\sqrt{n}}$).
        * **The "Why" & Physical Intuition:** When you average multiple observations together, random extreme high values (*e.g., an 8-hour ED wait*) are almost always neutralized by ordinary or low values in the same batch. For an average to be extreme, you would need to randomly pick *only* extreme patients in the exact same batch — which becomes exponentially improbable as sample size grows!
        * **Why This Matters to You:** It proves why we can safely run **$t$-tests, ANOVA, and linear regression** on sample means even when raw patient charts are heavily right-skewed, provided $N \ge 30$.

        #### 2. Quantifying Asymmetry: Fisher-Pearson Sample Skewness ($g_1$)

        $$g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3$$

        * **Plain English Translation:** *"Convert each patient's distance from the mean into standard deviation units, cube that distance so direction is preserved, sum them up, and adjust for sample size."*
        * **Symbol-by-Symbol Breakdown:**
          * $\left(\frac{x_i - \bar{x}}{s}\right)$: The standardized distance ($z$-score). Measures how many standard deviations a patient sits above or below the mean.
          * Cubing $\left(\dots\right)^3$: Squaring would make all numbers positive; **cubing preserves the directional sign** ($+3^3 = +27$, $-3^3 = -27$) while heavily magnifying points in the outer tail.
          * $\frac{n}{(n-1)(n-2)}$: An exact small-sample bias correction factor (the formula implemented in Excel's `=SKEW()` and SciPy).
        * **Clinical Interpretation Guidelines:**
          * $g_1 \approx 0$ (between $-0.5$ and $+0.5$): **Symmetric / Approximately Normal.**
          * $g_1 > +1.0$: **Highly Right-Skewed (Positive Skew).** A long tail of high values drags the mean to the right (*e.g., hospital costs, ICU stays, ED triage wait times*).
          * $g_1 < -1.0$: **Highly Left-Skewed (Negative Skew).** A long tail of low values drags the mean to the left (*e.g., gestational age at birth, adult cognitive scores*).

        #### 3. Quantifying Tail Heaviness: Fisher-Pearson Excess Kurtosis ($g_2$)

        $$g_2 = \left[\frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^4\right] - \frac{3(n-1)^2}{(n-2)(n-3)}$$

        * **Plain English Translation:** *"Raise each standardized distance to the 4th power to heavily magnify extreme outlier events, then subtract 3 so a standard normal bell curve equals zero."*
        * **Symbol-by-Symbol Breakdown:**
          * Raising to the 4th power $\left(\dots\right)^4$: Massively amplifies points far from the mean, making kurtosis an index of **tail thickness and outlier propensity**.
          * Subtracting 3: A theoretical normal distribution has a raw kurtosis of 3. Subtracting 3 sets the baseline of a normal distribution to exactly 0 ("Excess" Kurtosis).
        * **Clinical Interpretation Guidelines:**
          * $g_2 = 0$: **Mesokurtic.** Tail thickness matches a standard Gaussian bell curve.
          * $g_2 > 0$: **Leptokurtic (Fat Tails).** Indicates a sharp peak and heavy tails. Outliers and extreme physiological events occur far more frequently than normal theory predicts.
          * $g_2 < 0$: **Platykurtic (Thin Tails).** Flat-topped distribution with light tails (*e.g., uniform birth distribution across calendar months*).

        #### 4. The $N = 30$ Threshold for Clinical Research

        * **If $N \ge 30$:** The Central Limit Theorem guarantees that the distribution of sample averages is approximately Gaussian. Parametric methods ($t$-tests, ANOVA) are robust.
        * **If $N < 30$:** The sample size is insufficient to wash out raw skewness. If raw data is skewed, the sample mean remains skewed, leading to inflated false-positive rates. You must use non-parametric methods (**Mann-Whitney U, Wilcoxon Signed-Rank, Kruskal-Wallis**).
        """)

    with t2:
        st.markdown(r"""
        #### 📗 Microsoft Excel Implementation: Skewness & Histograms
        ##### Step 1: Formulas in Excel
        * **Calculate Fisher-Pearson Skewness:**
          `=SKEW(A2:A100)`
          * $\text{Skew} = 0$: Perfectly symmetric.
          * $\text{Skew} > +1.0$: Highly right-skewed (tail extends right).
          * $\text{Skew} < -1.0$: Highly left-skewed (tail extends left).
        * **Calculate Excess Kurtosis (Tail heaviness):**
          `=KURT(A2:A100)`

        ##### Step 2: Creating a Native Histogram
        Select `A1:A100` $\rightarrow$ **Insert Tab** $\rightarrow$ **Statistic Chart (Histogram)** $\rightarrow$ Right click X-axis $\rightarrow$ **Format Axis** $\rightarrow$ Set Bin Width or Number of Bins.
        """)

    with t3:
        st.markdown(r"""
        #### 🐍 Python Implementation: Central Limit Theorem Simulation
        ```python
        import numpy as np
        import scipy.stats as stats

        # 1. Generate heavy right-skewed raw population (e.g. ED wait times)
        rng = np.random.RandomState(42)
        population = rng.exponential(scale=45.0, size=5000)

        # 2. Simulate drawing 1,000 samples of size N = 30
        n = 30
        sample_means = [np.mean(rng.choice(population, size=n, replace=True)) for _ in range(1000)]

        # 3. Check skewness reduction
        print(f"Raw Population Skewness: {stats.skew(population):.2f}")
        print(f"Sample Means Skewness (N=30): {stats.skew(sample_means):.2f} (Normality Emerges!)")
        ```
        """)

    with t4:
        df_clt, csv_clt = generate_clt_practice_data()
        st.markdown("#### 💾 Practice Dataset: Emergency Department Triage Waiting Times")
        st.caption("Contains 80 patient triage records exhibiting heavy positive skewness.")
        st.dataframe(df_clt.head(10), use_container_width=True)
        st.download_button(
            label="📥 Download CLT Practice Data (CSV)",
            data=csv_clt,
            file_name="ed_triage_wait_times_clt_data.csv",
            mime="text/csv",
            key="dl_btn_clt"
        )


# =====================================================================
# 6. Confidence Intervals (Mean & Proportion) Module
# =====================================================================

def render_confidence_intervals_module() -> None:
    """Renders Module 6: Confidence Intervals (Continuous Mean & Binary Proportion)."""
    st.markdown("## 🎯 Confidence Intervals: Continuous Mean & Binary Proportion")
    st.markdown("""
    A $p$-value only tests whether an effect is statistically distinguishable from zero; 
    a **Confidence Interval** tells you the **direction and clinical magnitude** of the effect.
    """)

    # Interactive CI Calculator & Comparison
    st.markdown("### 🎮 Interactive Confidence Interval Playground")
    st.caption("Switch between Continuous Mean and Binary Proportion to compare normal approximations against exact small-sample distributions.")

    ci_type = st.radio("Select Endpoint Family:", options=["Continuous Mean (z vs. Student's t)", "Binary Proportion (Wald vs. Wilson Score)"], horizontal=True)

    if "Continuous Mean" in ci_type:
        c1, c2 = st.columns([1.5, 2.5])
        with c1:
            mean_in = st.number_input("Sample Mean (x̄):", value=6.80, step=0.10)
            sd_in = st.number_input("Sample SD (s):", min_value=0.10, value=1.40, step=0.10)
            n_in = st.slider("Sample Size (N):", min_value=3, max_value=100, value=12, step=1)
            conf_level = st.selectbox("Confidence Level:", options=[0.90, 0.95, 0.99], index=1)

        # Compute intervals
        sem = sd_in / math.sqrt(n_in)
        alpha = 1.0 - conf_level
        z_crit = _normal_ppf(1.0 - alpha / 2.0)
        t_crit = _t_ppf(1.0 - alpha / 2.0, df=n_in - 1)
        z_lo, z_hi = mean_in - z_crit * sem, mean_in + z_crit * sem
        t_lo, t_hi = mean_in - t_crit * sem, mean_in + t_crit * sem

        with c2:
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Normal z-Interval (Asymptotic)", f"[{z_lo:.2f}, {z_hi:.2f}]", help=f"Margin = ±{z_crit*sem:.2f} (z = {z_crit:.3f})")
            with m2:
                penalty = ((t_crit - z_crit) / z_crit) * 100
                st.metric("Student's t-Interval (Exact)", f"[{t_lo:.2f}, {t_hi:.2f}]", delta=f"{penalty:+.1f}% wider penalty" if n_in < 30 else None)

            # Visual Interval Comparison Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[z_lo, z_hi], y=["Normal z-Interval", "Normal z-Interval"], mode="lines+markers", line=dict(color="#2563eb", width=5), marker=dict(size=10)))
            fig.add_trace(go.Scatter(x=[t_lo, t_hi], y=["Student's t-Interval", "Student's t-Interval"], mode="lines+markers", line=dict(color="#10b981", width=5), marker=dict(size=10)))
            fig.add_vline(x=mean_in, line_dash="dash", line_color="#dc2626", annotation_text=f"Mean: {mean_in:.2f}")
            fig.update_layout(title=f"{int(conf_level*100)}% Confidence Interval Comparison (N = {n_in}, df = {n_in-1})", height=230, margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig, use_container_width=True)

    else:
        c1, c2 = st.columns([1.5, 2.5])
        with c1:
            events_x = st.number_input("Number of Events (x):", min_value=0, max_value=200, value=2, step=1)
            trials_n = st.number_input("Total Sample Size (N):", min_value=1, max_value=500, value=40, step=5)
            conf_p = st.selectbox("Confidence Level:", options=[0.90, 0.95, 0.99], index=1)

        prop_res = calculate_confidence_interval_proportion(successes=events_x, trials=trials_n, confidence=conf_p)

        with c2:
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    "Standard Wald Interval",
                    f"[{prop_res['wald_lower']*100:.1f}%, {prop_res['wald_upper']*100:.1f}%]",
                    help="Textbook formula: p_hat ± z * sqrt(p*(1-p)/n)"
                )
            with m2:
                st.metric(
                    "Wilson Score Interval (Recommended)",
                    f"[{prop_res['wilson_lower']*100:.1f}%, {prop_res['wilson_upper']*100:.1f}%]",
                    help="Inverts score test. Guaranteed [0%, 100%]"
                )

            # Warning if Wald fails
            if not prop_res["wald_valid"]:
                st.error(f"❌ **Wald Formula Invalid:** Expected events $np = {events_x}$ or $n(1-p) < 5$. The Wald interval yields erratic coverage and can produce negative lower bounds! **You must use the Wilson Score or Agresti-Coull interval.**")
            else:
                st.success(r"✅ **Sample Size Adequate ($np \ge 5$ and $n(1-p) \ge 5$):** Wald approximation is acceptable.")

    st.markdown("---")

    # 4-Tab Educational Architecture
    t1, t2, t3, t4 = st.tabs([
        "📋 Rationale & Framework",
        "📗 Microsoft Excel Guide",
        "🐍 Python Code Guide",
        "💾 Practice Data Download"
    ])

    with t1:
        st.markdown(r"""
        #### 1. What a 95% Confidence Interval Actually Means

        * **The Frequentist Definition:** The true population parameter ($\mu$ or $p$) is a fixed, unknown constant of nature. It does *not* move. If we were to repeat this identical clinical trial 100 times with fresh random samples of patients and calculate a 95% CI from each trial, **exactly 95 of those 100 calculated intervals would successfully contain the true parameter**.
        * **The Common Misconception:** *"There is a 95% probability that the true mean lies between 5.2 and 8.4."*
          * **False!** Once an interval is calculated from your observed data, the true mean is either inside it (probability = 1) or outside it (probability = 0). The "95%" describes the long-term reliability of the *process*, not the probability of a specific fixed interval.

        #### 2. Continuous Mean: Normal $z$-Interval vs. Student's $t$-Interval

        ##### A. Large-Sample Normal $z$-Interval Formula ($N \ge 30$)
        $$\text{CI}_{95\%} = \bar{x} \pm z_{1 - \alpha/2} \times \frac{s}{\sqrt{n}} = \bar{x} \pm 1.960 \times \text{SEM}$$

        * **Plain English Translation:** *"Start at your sample average ($\bar{x}$). Step downwards and upwards by a margin of error equal to 1.96 times the standard error ($s/\sqrt{n}$) to capture the range compatible with the data."*
        * **Symbol-by-Symbol Breakdown:**
          * $\bar{x}$: The observed sample mean (your single best guess or point estimate).
          * $\pm$: Subtract the margin to establish the Lower Bound; add the margin to establish the Upper Bound.
          * $z_{1 - \alpha/2} = 1.960$: The standard normal critical multiplier for a two-tailed $95\%$ confidence level ($\alpha = 0.05$). Under a Gaussian curve, $95\%$ of all values sit within $1.960$ standard deviations of the center.
          * $\frac{s}{\sqrt{n}}$: The Standard Error of the Mean ($\text{SEM}$), representing experimental sampling noise.

        ##### B. Small-Sample Student's $t$-Interval Formula ($N < 30$)
        $$\text{CI}_{95\%} = \bar{x} \pm t_{1 - \alpha/2, \, df = n - 1} \times \frac{s}{\sqrt{n}}$$

        * **Plain English Translation:** *"When your sample is small, replace the fixed 1.960 multiplier with a larger, penalized multiplier from Student's $t$-distribution to account for extra uncertainty in your sample standard deviation."*
        * **Symbol-by-Symbol Breakdown:**
          * $t_{1 - \alpha/2, \, df}$: The Student's $t$ critical multiplier. It depends strictly on the degrees of freedom ($df = n - 1$).
          * $df = n - 1$ (Degrees of Freedom): Because 1 degree of freedom was already "spent" estimating the sample mean $\bar{x}$, only $n - 1$ pieces of independent information remain to estimate spread $s$.
        * **The "Why" Behind the Small-Sample $t$-Penalty:** In small trials (*e.g., $N = 5$ patients*), not only is the mean $\bar{x}$ uncertain, but the sample standard deviation $s$ is also noisy and unstable. William Sealy Gosset ("Student", 1908) proved that to maintain a true 95% coverage rate, you must widen the interval. For $N = 5$ ($df = 4$), $t = 2.776$ — making the interval **41.6% wider** than the naive normal formula!

        #### 3. Binary Proportion: Wald vs. Wilson Score vs. Agresti-Coull

        ##### A. The Standard Wald Approximation Formula
        $$\text{Wald CI}_{95\%} = \hat{p} \pm z_{1 - \alpha/2} \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}} \approx \hat{p} \pm 1.960 \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$$

        * **Plain English Translation:** *"Take the observed success rate ($\hat{p}$), and add/subtract 1.96 times the coin-flip standard error."*
        * **Symbol-by-Symbol Breakdown:**
          * $\hat{p} = \frac{x}{n}$: The observed sample proportion (e.g., $x = 3$ surgical complications out of $n = 50$ operations = $0.06$ or $6\%$).
          * $\hat{p}(1 - \hat{p})$: The binomial variance of a yes/no event. Variance peaks at $\hat{p} = 0.50$ ($0.5 \times 0.5 = 0.25$) and drops toward zero as events become rare.
          * $\sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$: The standard error of a proportion.
        * **Why the Wald Interval Fails for Rare Events:**
          * **The Zero-Event Collapse:** If zero complications occur in $N = 30$ patients ($x = 0 \implies \hat{p} = 0$), the Wald margin is $1.96 \times \sqrt{0} = 0$, giving an interval of **$[0\%, 0\%]$**! This nonsensically claims that complications are mathematically impossible in future patients.
          * **Negative Bounds:** In small samples with low incidence, subtracting the margin often produces negative percentages (*e.g., $[-2.1\%, +8.4\%]$*), which is physically impossible for a probability.

        ##### B. The Wilson Score Interval Formula (Gold Standard)
        $$\text{Wilson CI}_{95\%} = \frac{\hat{p} + \dfrac{z^2}{2n} \pm z \sqrt{\dfrac{\hat{p}(1 - \hat{p})}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}}$$

        * **Plain English Translation:** *"Instead of centering strictly on $\hat{p}$, gently pull the center slightly inward toward 50% using small anchor terms, and divide by $(1 + z^2/n)$ so the bounds are mathematically locked within 0% and 100%."*
        * **Symbol-by-Symbol Breakdown:**
          * $\hat{p} = \frac{x}{n}$: Observed sample proportion.
          * $z = 1.960$: Normal critical value for 95% confidence ($z^2 \approx 3.8416$).
          * $\frac{z^2}{2n}$ and $\frac{z^2}{4n^2}$: Asymmetry correction terms that prevent variance from collapsing when $\hat{p} = 0$ or $\hat{p} = 1$.
          * $1 + \frac{z^2}{n}$: Normalizing denominator that guarantees the bounds never cross below $0.0$ or above $1.0$.
        * **Real-World Impact:** With $0$ complications in $30$ patients, Wilson correctly calculates **$[0.0\%, 11.3\%]$**, honestly informing surgeons that the true complication risk could still be as high as 11.3%!

        ##### C. The Agresti-Coull "Plus-Four" Shortcut Formula
        $$\tilde{n} = n + z^2 \approx n + 4, \quad \tilde{x} = x + \frac{z^2}{2} \approx x + 2, \quad \tilde{p} = \frac{x + 2}{n + 4}$$
        $$\text{Agresti-Coull CI}_{95\%} = \tilde{p} \pm 1.960 \sqrt{\frac{\tilde{p}(1 - \tilde{p})}{\tilde{n}}}$$

        * **Plain English Translation:** *"Pretend you ran 4 extra patient trials with 2 complications and 2 successes: add 2 to your event count, add 4 to your sample size, and then run the standard formula with those adjusted numbers."*
        * **Why Clinicians Love It:** Alan Agresti and Brent Coull (1998) showed that this simple "add two successes and two failures" trick almost perfectly matches the complex Wilson score formula, eliminating the zero-event collapse while remaining simple enough to calculate by hand or in Excel!
        """)

    with t2:
        st.markdown(r"""
        #### 📗 Microsoft Excel Implementation: Confidence Intervals
        ##### 1. Continuous Mean ($t$-Interval)
        * Calculate Margin of Error using Student's $t$:
          `=CONFIDENCE.T(0.05, STDEV.S(A2:A50), COUNT(A2:A50))`
        * Lower 95% Bound:
          `=AVERAGE(A2:A50) - CONFIDENCE.T(0.05, STDEV.S(A2:A50), COUNT(A2:A50))`
        * Upper 95% Bound:
          `=AVERAGE(A2:A50) + CONFIDENCE.T(0.05, STDEV.S(A2:A50), COUNT(A2:A50))`

        ##### 2. Binary Proportion (Agresti-Coull "Plus-Four" Shortcut)
        When $x$ events occur in $n$ trials in Excel:
        * Adjusted $n$: `=n + 4`
        * Adjusted $p$: `=(x + 2) / (n + 4)`
        * Margin of Error: `=1.96 * SQRT(((x+2)/(n+4)) * (1 - ((x+2)/(n+4))) / (n + 4))`
        """)

    with t3:
        st.markdown(r"""
        #### 🐍 Python Implementation: Continuous and Proportion CIs
        ```python
        import numpy as np
        import scipy.stats as stats
        from statsmodels.stats.proportion import proportion_confint

        # 1. Continuous Mean: Exact Student's t-interval
        data = np.array([6.5, 7.2, 5.8, 6.9, 8.1, 6.4])
        mean_val = np.mean(data)
        sem_val = stats.sem(data)
        ci_mean = stats.t.interval(0.95, df=len(data)-1, loc=mean_val, scale=sem_val)
        print(f"95% Mean CI: [{ci_mean[0]:.2f}, {ci_mean[1]:.2f}]")

        # 2. Binary Proportion: Wilson Score Interval (Handles x=0 cleanly!)
        events_x = 2
        trials_n = 40
        ci_prop = proportion_confint(count=events_x, nobs=trials_n, alpha=0.05, method="wilson")
        print(f"95% Wilson Proportion CI: [{ci_prop[0]*100:.1f}%, {ci_prop[1]*100:.1f}%]")
        ```
        """)

    with t4:
        df_ci, csv_ci = generate_confidence_intervals_practice_data()
        st.markdown("#### 💾 Practice Dataset: Surgical Recovery Duration & Complications")
        st.caption("Contains 50 laparoscopic cholecystectomy cases with continuous recovery days and binary complications.")
        st.dataframe(df_ci.head(10), use_container_width=True)
        st.download_button(
            label="📥 Download CI Practice Data (CSV)",
            data=csv_ci,
            file_name="surgical_recovery_confidence_intervals_data.csv",
            mime="text/csv",
            key="dl_btn_ci"
        )

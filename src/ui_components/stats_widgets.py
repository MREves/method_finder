"""UI Components for Statistical Test Presentation.

Provides reusable Streamlit UI cards, callouts, and Plotly visualization
builders for the statistical tests page.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import html


def render_table(df: pd.DataFrame, height: Optional[int] = 240):
    """Renders a dataframe using native st.dataframe with resilient fallback handling."""
    try:
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)
    except TypeError:
        try:
            st.dataframe(df, use_container_width=True, height=height)
        except TypeError:
            st.dataframe(df)


def render_html(html_content: str):
    """Renders HTML cleanly without CommonMark indentation code-block interference."""
    cleaned = "\n".join(line.strip() for line in html_content.splitlines() if line.strip())
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)



def render_recommendation_card(rec: Dict[str, Any]):
    """Renders the top recommendation card with badges and summary."""
    parametric_badge = "🔵 Parametric" if rec["parametric"] else "🟠 Non-Parametric"
    
    st.markdown(f"""
    <div style="background-color: #f0f7ff; border-left: 5px solid #2563eb; padding: 1.25rem; border-radius: 6px; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #2563eb;">
                {rec['category']}
            </span>
            <span style="font-size: 0.85rem; font-weight: 600; background: #e0e7ff; color: #3730a3; padding: 0.2rem 0.6rem; border-radius: 9999px;">
                {parametric_badge}
            </span>
        </div>
        <h2 style="margin: 0 0 0.5rem 0; color: #1e3a8a; font-size: 1.6rem;">{rec['name']}</h2>
        <p style="margin: 0; font-size: 1.05rem; color: #1e293b; line-height: 1.5;">{rec['description']}</p>
    </div>
    """, unsafe_allow_html=True)


def render_methodology_tabs(
    rec: Dict[str, Any],
    demo_df: Optional[pd.DataFrame] = None,
    stats_results: Optional[Dict[str, Any]] = None
):
    """Renders tabs for Rationale & Assumptions, Excel Guide, Python Code, and Download Sample Data."""
    tab_rationale, tab_excel, tab_code, tab_download = st.tabs([
        "📋 Rationale & Decision Framework",
        "📗 Excel Implementation Guide",
        "🐍 Python Code Implementation",
        "📥 Download Sample Data"
    ])

    with tab_rationale:
        st.subheader("Why this method?")
        for item in rec["reasoning"]:
            st.markdown(f"- **{item}**")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ When to use")
            for condition in rec["when_to_use"]:
                st.markdown(f"- {condition}")

        with col2:
            st.markdown("#### ⚠️ Key Assumptions to Verify")
            for assumption in rec["key_assumptions"]:
                st.markdown(f"- {assumption}")
            
            st.info(f"**Alternative if assumptions fail:** {rec['alternative_test']}")

    with tab_excel:
        if rec.get("test_key") == "one_sample_t_test":
            render_one_sample_excel_guide(rec)
        elif rec.get("test_key") == "wilcoxon_signed_rank":
            render_wilcoxon_excel_guide(rec)
        elif rec.get("test_key") == "wilcoxon_paired":
            render_wilcoxon_paired_excel_guide(rec)
        elif rec.get("test_key") == "mann_whitney_u":
            render_mann_whitney_excel_guide(rec)
        elif rec.get("test_key") == "welch_t_test":
            render_welch_excel_guide(rec)
        elif rec.get("test_key") == "students_t_test":
            render_students_t_excel_guide(rec)
        elif rec.get("test_key") == "paired_t_test":
            render_paired_t_excel_guide(rec)
        elif rec.get("test_key") == "one_way_anova":
            render_one_way_anova_excel_guide(rec)
        elif rec.get("test_key") == "kruskal_wallis":
            render_kruskal_wallis_excel_guide(rec)
        elif rec.get("test_key") == "repeated_measures_anova":
            render_repeated_measures_anova_excel_guide(rec)
        elif rec.get("test_key") == "friedman_test":
            render_friedman_excel_guide(rec)
        elif rec.get("test_key") in ["chi_square_independence", "contingency_table_2x2"]:
            render_chi_square_excel_guide(rec)
        elif rec.get("test_key") == "fisher_exact":
            render_fisher_exact_excel_guide(rec)
        elif rec.get("test_key") in ["mcnemar_test", "paired_binary"]:
            render_mcnemar_excel_guide(rec)
        elif rec.get("test_key") in ["linear_regression", "multiple_linear_regression"]:
            render_linear_regression_excel_guide(rec)
        elif rec.get("test_key") in ["pearson_correlation", "correlation_continuous"]:
            render_pearson_correlation_excel_guide(rec)
        elif rec.get("test_key") in ["spearman_correlation", "spearman_rank"]:
            render_spearman_correlation_excel_guide(rec)
        elif rec.get("test_key") in ["binary_logistic_regression", "binary_classification", "logistic_regression"]:
            render_binary_logistic_regression_excel_guide(rec)
        else:
            render_generic_excel_guide(rec)

    with tab_code:
        if rec.get("test_key") == "one_sample_t_test":
            render_one_sample_python_guide(rec)
        elif rec.get("test_key") == "wilcoxon_signed_rank":
            render_wilcoxon_python_guide(rec)
        elif rec.get("test_key") == "wilcoxon_paired":
            render_wilcoxon_paired_python_guide(rec)
        elif rec.get("test_key") == "mann_whitney_u":
            render_mann_whitney_python_guide(rec)
        elif rec.get("test_key") == "welch_t_test":
            render_welch_python_guide(rec)
        elif rec.get("test_key") == "students_t_test":
            render_students_t_python_guide(rec)
        elif rec.get("test_key") == "paired_t_test":
            render_paired_t_python_guide(rec)
        elif rec.get("test_key") == "one_way_anova":
            render_one_way_anova_python_guide(rec)
        elif rec.get("test_key") == "kruskal_wallis":
            render_kruskal_wallis_python_guide(rec)
        elif rec.get("test_key") == "repeated_measures_anova":
            render_repeated_measures_anova_python_guide(rec)
        elif rec.get("test_key") == "friedman_test":
            render_friedman_python_guide(rec)
        elif rec.get("test_key") in ["chi_square_independence", "contingency_table_2x2"]:
            render_chi_square_python_guide(rec)
        elif rec.get("test_key") == "fisher_exact":
            render_fisher_exact_python_guide(rec)
        elif rec.get("test_key") in ["mcnemar_test", "paired_binary"]:
            render_mcnemar_python_guide(rec)
        elif rec.get("test_key") in ["linear_regression", "multiple_linear_regression"]:
            render_linear_regression_python_guide(rec)
        elif rec.get("test_key") in ["pearson_correlation", "correlation_continuous"]:
            render_pearson_correlation_python_guide(rec)
        elif rec.get("test_key") in ["spearman_correlation", "spearman_rank"]:
            render_spearman_correlation_python_guide(rec)
        elif rec.get("test_key") in ["binary_logistic_regression", "binary_classification", "logistic_regression"]:
            render_binary_logistic_regression_python_guide(rec)
        else:
            render_generic_python_guide(rec)

    with tab_download:
        render_sample_data_tab(rec, demo_df, stats_results)


def _generate_fallback_dataset(test_key: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates a small deterministic fallback dataset if none was passed from the active demo."""
    if test_key in ["one_sample_t_test", "wilcoxon_signed_rank"]:
        vals = np.array([98.2, 102.5, 105.1, 99.4, 108.2, 110.0, 101.3, 107.5, 103.8, 106.2, 104.5, 109.1, 97.8, 105.4, 108.0])
        df = pd.DataFrame({"Observation_ID": [f"Obs_{i+1:03d}" for i in range(len(vals))], "Value": vals})
        return df, {"benchmark": 100.0, "preset_name": "Standard Benchmark Comparison (μ₀ = 100.0)"}
    elif test_key in ["welch_t_test", "students_t_test", "mann_whitney_u"]:
        a = [102.1, 105.4, 98.7, 104.2, 101.0, 106.3, 99.8, 103.5, 100.2, 105.0]
        b = [112.4, 115.1, 108.9, 114.2, 110.5, 116.8, 111.0, 113.5, 115.2, 117.0]
        df_a = pd.DataFrame({"Observation_ID": [f"A_{i+1:02d}" for i in range(len(a))], "Group": "Group A (Control)", "Value": a})
        df_b = pd.DataFrame({"Observation_ID": [f"B_{i+1:02d}" for i in range(len(b))], "Group": "Group B (Treatment)", "Value": b})
        return pd.concat([df_a, df_b], ignore_index=True), {"preset_name": "Two Independent Groups"}
    elif test_key in ["paired_t_test", "wilcoxon_paired"]:
        pre = np.array([135.0, 142.0, 138.5, 145.0, 140.2, 136.8, 144.1, 139.0, 141.5, 143.0])
        post = np.array([128.0, 134.5, 131.0, 136.2, 133.0, 130.5, 135.8, 132.0, 134.0, 135.2])
        diff = np.round(post - pre, 1)
        df = pd.DataFrame({"Subject_ID": [f"Subj_{i+1:02d}" for i in range(len(pre))], "Pre_Intervention": pre, "Post_Intervention": post, "Difference": diff})
        return df, {"preset_name": "Matched Paired Repeated Measurements"}
    elif test_key in ["one_way_anova", "kruskal_wallis"]:
        g1 = [10.2, 11.5, 12.1, 10.8, 11.0]
        g2 = [14.1, 15.3, 14.8, 15.0, 14.5]
        g3 = [18.2, 19.0, 18.5, 19.5, 18.8]
        df1 = pd.DataFrame({"Observation_ID": [f"G1_{i+1:02d}" for i in range(5)], "Group": "Group 1", "Value": g1})
        df2 = pd.DataFrame({"Observation_ID": [f"G2_{i+1:02d}" for i in range(5)], "Group": "Group 2", "Value": g2})
        df3 = pd.DataFrame({"Observation_ID": [f"G3_{i+1:02d}" for i in range(5)], "Group": "Group 3", "Value": g3})
        return pd.concat([df1, df2, df3], ignore_index=True), {"preset_name": "Three Independent Cohorts"}
    elif test_key in ["repeated_measures_anova", "friedman_test"]:
        Y = np.array([[55.0, 62.0, 71.0], [52.5, 60.0, 68.5], [58.0, 65.2, 74.0], [54.0, 61.5, 70.0], [56.2, 63.8, 72.5]])
        time_names = ["Baseline", "Month 1", "Month 3"]
        rows = []
        for i in range(5):
            for j in range(3):
                rows.append({"Subject_ID": f"Subj_{i+1:03d}", "Time_Point": time_names[j], "Score": Y[i, j]})
        return pd.DataFrame(rows), {"Y_wide": Y, "time_names": time_names, "preset_name": "Repeated Conditions (Baseline, Month 1, Month 3)"}
    elif test_key in ["chi_square_independence", "contingency_table_2x2", "fisher_exact"]:
        table = pd.DataFrame({"Disease (+)": [45, 20], "Healthy (-)": [55, 80]}, index=["Treatment", "Control"])
        cases = []
        for _ in range(45): cases.append({"Subject_ID": f"S_{len(cases)+1:03d}", "Group": "Treatment", "Outcome": "Disease (+)", "Outcome_Binary": 1})
        for _ in range(55): cases.append({"Subject_ID": f"S_{len(cases)+1:03d}", "Group": "Treatment", "Outcome": "Healthy (-)", "Outcome_Binary": 0})
        for _ in range(20): cases.append({"Subject_ID": f"S_{len(cases)+1:03d}", "Group": "Control", "Outcome": "Disease (+)", "Outcome_Binary": 1})
        for _ in range(80): cases.append({"Subject_ID": f"S_{len(cases)+1:03d}", "Group": "Control", "Outcome": "Healthy (-)", "Outcome_Binary": 0})
        return table, {"df_cases": pd.DataFrame(cases), "preset_name": "Standard Contingency Trial (E ≥ 5)"}
    elif test_key in ["mcnemar_test", "paired_binary"]:
        table = pd.DataFrame({"Post (+ / Recovered)": [40, 8], "Post (- / Persistent)": [22, 30]}, index=["Pre (+ / Case)", "Pre (- / Non-Case)"])
        cases = []
        for _ in range(40): cases.append({"Subject_ID": f"S_{len(cases)+1:03d}", "Occasion_1": "Positive (+)", "Occasion_2": "Positive (+)", "Transition": "Concordant (+/+)"})
        for _ in range(22): cases.append({"Subject_ID": f"S_{len(cases)+1:03d}", "Occasion_1": "Positive (+)", "Occasion_2": "Negative (-)", "Transition": "Discordant (+/-)"})
        for _ in range(8): cases.append({"Subject_ID": f"S_{len(cases)+1:03d}", "Occasion_1": "Negative (-)", "Occasion_2": "Positive (+)", "Transition": "Discordant (-/+)"})
        for _ in range(30): cases.append({"Subject_ID": f"S_{len(cases)+1:03d}", "Occasion_1": "Negative (-)", "Occasion_2": "Negative (-)", "Transition": "Concordant (-/-)"})
        return table, {"df_cases": pd.DataFrame(cases), "preset_name": "Paired Binary Transitions"}
    elif test_key in ["pearson_correlation", "correlation_continuous"]:
        x = np.array([2.5, 3.0, 3.8, 4.2, 4.8, 5.1, 5.7, 6.2, 6.8, 7.5, 8.0, 8.4, 9.1, 9.8, 10.5])
        y = np.array([55.0, 58.2, 64.0, 67.5, 71.0, 73.5, 78.0, 81.2, 85.0, 89.1, 91.5, 93.8, 96.0, 98.5, 102.0])
        df = pd.DataFrame({"Participant_ID": [f"P_{i+1:03d}" for i in range(len(x))], "Study_Hours_X": x, "Exam_Score_Y": y})
        return df, {"preset_name": "Study Hours vs. Exam Score (r ≈ +0.82)"}
    elif test_key in ["spearman_correlation", "spearman_rank"]:
        x = np.array([10.2, 12.0, 13.5, 15.0, 16.8, 18.2, 19.5, 21.0, 22.4, 24.0, 25.5, 27.0, 28.6, 30.0, 31.5])
        y = np.array([22.0, 28.5, 34.0, 39.2, 45.0, 49.5, 53.0, 57.2, 60.5, 63.8, 67.0, 70.1, 72.8, 75.5, 78.0])
        rx = np.arange(1, len(x)+1, dtype=float)
        ry = np.arange(1, len(y)+1, dtype=float)
        d_sq = (rx - ry)**2
        df = pd.DataFrame({
            "Participant_ID": [f"P_{i+1:03d}" for i in range(len(x))],
            "Aptitude_Score_X": x,
            "Job_Performance_Y": y,
            "Rank_X": rx,
            "Rank_Y": ry,
            "Squared_Rank_Diff": d_sq
        })
        return df, {"preset_name": "Aptitude vs. Performance (Monotonic Non-Linear, ρ ≈ +0.85)"}
    elif test_key in ["linear_regression", "multiple_linear_regression"]:
        y = np.array([22.1, 24.5, 27.8, 30.2, 33.5, 35.8, 39.0, 41.5, 45.2, 48.0])
        x1 = np.array([12.0, 14.5, 18.0, 21.2, 25.0, 28.5, 32.0, 35.8, 40.0, 44.5])
        x2 = np.array([5.0, 6.2, 7.8, 8.5, 10.0, 11.2, 13.0, 14.5, 16.0, 18.2])
        df = pd.DataFrame({
            "Obs_ID": [f"Obs_{i+1:03d}" for i in range(len(y))],
            "Sales_Revenue_Y": y,
            "TV_Budget_X1": x1,
            "Digital_Budget_X2": x2
        })
        return df, {"preset_name": "Advertising Budget -> Sales Revenue (k = 2)"}
    elif test_key in ["binary_logistic_regression", "binary_classification", "logistic_regression"]:
        y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1])
        x1 = np.array([42.0, 45.0, 48.0, 51.0, 53.0, 55.0, 58.0, 60.0, 62.0, 65.0, 68.0, 70.0, 72.0, 75.0, 78.0])
        x2 = np.array([118.0, 122.0, 125.0, 130.0, 128.0, 135.0, 138.0, 142.0, 145.0, 150.0, 155.0, 160.0, 158.0, 165.0, 172.0])
        x3 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        df = pd.DataFrame({
            "Obs_ID": [f"Obs_{i+1:03d}" for i in range(len(y))],
            "CVD_Event": y,
            "Age_Years": x1,
            "Systolic_BP": x2,
            "Current_Smoker": x3
        })
        return df, {"preset_name": "Cardiology CVD Risk (k = 3 Predictors)"}
    else:
        df = pd.DataFrame({"ID": [1, 2, 3], "Value": [10.0, 20.0, 30.0]})
        return df, {"preset_name": "Generic Reference Dataset"}


def _get_rm_wide_dataframe(demo_df: pd.DataFrame, stats_results: Dict[str, Any]) -> pd.DataFrame:
    """Extracts or pivots a matched wide-format dataset for Repeated Measures ANOVA or Friedman."""
    if "Y_wide" in stats_results and "time_names" in stats_results:
        Y = stats_results["Y_wide"]
        cols = stats_results["time_names"]
        df_wide = pd.DataFrame(Y, columns=cols)
        df_wide.insert(0, "Subject_ID", [f"Subj_{i+1:03d}" for i in range(len(df_wide))])
        return df_wide
    elif "Subject_ID" in demo_df.columns and "Time_Point" in demo_df.columns and "Score" in demo_df.columns:
        piv = demo_df.pivot(index="Subject_ID", columns="Time_Point", values="Score").reset_index()
        piv.columns.name = None
        return piv
    elif "Subject_ID" in demo_df.columns and "Condition" in demo_df.columns and "Score" in demo_df.columns:
        piv = demo_df.pivot(index="Subject_ID", columns="Condition", values="Score").reset_index()
        piv.columns.name = None
        return piv
    return demo_df


def _get_sample_data_guide_info(test_key: str, rec: Dict[str, Any], demo_df: pd.DataFrame, stats_results: Dict[str, Any]) -> Dict[str, str]:
    """Returns test-tailored guidance for Excel cell mapping and Python execution."""
    fn_map = {
        "one_sample_t_test": "one_sample_t_test_sample_data.csv",
        "wilcoxon_signed_rank": "wilcoxon_signed_rank_sample_data.csv",
        "welch_t_test": "welch_t_test_sample_data.csv",
        "students_t_test": "students_t_test_sample_data.csv",
        "mann_whitney_u": "mann_whitney_u_sample_data.csv",
        "paired_t_test": "paired_t_test_sample_data.csv",
        "wilcoxon_paired": "wilcoxon_paired_sample_data.csv",
        "one_way_anova": "one_way_anova_sample_data.csv",
        "kruskal_wallis": "kruskal_wallis_sample_data.csv",
        "repeated_measures_anova": "repeated_measures_anova_wide.csv",
        "friedman_test": "friedman_test_wide.csv",
        "chi_square_independence": "chi_square_2x2_table.csv",
        "fisher_exact": "fisher_exact_2x2_table.csv",
        "mcnemar_test": "mcnemar_2x2_table.csv",
        "pearson_correlation": "pearson_correlation_sample_data.csv",
        "spearman_correlation": "spearman_correlation_sample_data.csv",
        "linear_regression": "linear_regression_sample_data.csv",
        "binary_logistic_regression": "binary_logistic_regression_sample_data.csv"
    }
    default_filename = fn_map.get(test_key, f"{test_key}_sample_data.csv")

    if test_key in ["one_sample_t_test"]:
        mu0 = stats_results.get("benchmark", 100.0)
        n = len(demo_df)
        excel_text = f"""
1. **Open File in Excel:** Navigate to **File > Open** and select `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Observation_ID` (`Obs_001`, `Obs_002`, ...)
   * **Column B:** `Value` (Continuous sample observations)
3. **Spreadsheet Mapping to Excel Guide:**
   * Column B corresponds to the sample measurement range `A2:A{n+1}` documented in the **📗 Excel Implementation Guide** tab.
   * **Step 1 (Assumptions):** Compute Mean via `=AVERAGE(B2:B{n+1})`, Sample SD via `=STDEV.S(B2:B{n+1})`, and Tukey IQR fences (`=QUARTILE.INC(B2:B{n+1}, 1)`, `=QUARTILE.INC(B2:B{n+1}, 3)`).
   * **Step 2 (Proposed Test):** Calculate the t-statistic: `=(AVERAGE(B2:B{n+1}) - {mu0}) / (STDEV.S(B2:B{n+1}) / SQRT(COUNT(B2:B{n+1})))`.
   * Compute the two-sided p-value: `=T.DIST.2T(ABS(t_stat), COUNT(B2:B{n+1}) - 1)`.
"""
        python_text = f"""import pandas as pd

# 1. Load the downloaded sample dataset
df = pd.read_csv("{default_filename}")
print(f"Loaded dataset: {{len(df)}} observations")
print(df.head())

# 2. Extract values and theoretical benchmark
values = df["Value"]
benchmark = {mu0}

# 3. Execute the automated assumption checking pipeline:
# (Copy and paste run_one_sample_pipeline from Step 4 of the Python tab)
results = run_one_sample_pipeline(values, mu_0=benchmark)
"""

    elif test_key in ["wilcoxon_signed_rank"]:
        mu0 = stats_results.get("benchmark", 100.0)
        n = len(demo_df)
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Observation_ID`
   * **Column B:** `Value` (Continuous or ordinal observations)
3. **Spreadsheet Mapping to Excel Guide:**
   * In Column C, compute signed deviations from median benchmark $\\mu_0 = {mu0}$: `=B2-{mu0}`.
   * In Column D, compute absolute deviations: `=ABS(C2)`.
   * In Column E, assign fractional ranks via `=IF(C2=0, "", RANK.AVG(D2, $D$2:$D${n+1}, 1))`.
   * In Column F, isolate positive ranks `=IF(C2>0, E2, 0)` and in Column G, negative ranks `=IF(C2<0, E2, 0)`.
   * Wilcoxon test statistic: `W = MIN(SUM(F2:F{n+1}), SUM(G2:G{n+1}))`.
"""
        python_text = f"""import pandas as pd

# 1. Load the downloaded sample dataset
df = pd.read_csv("{default_filename}")
print(df.head())

# 2. Extract observations and test against benchmark
values = df["Value"]
benchmark = {mu0}

# 3. Run automated assumption-checked pipeline:
results = run_wilcoxon_signed_rank_pipeline(values, mu_0=benchmark)
"""

    elif test_key in ["welch_t_test"]:
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Observation_ID`
   * **Column B:** `Group` (e.g. Group A vs Group B)
   * **Column C:** `Value`
3. **Spreadsheet Mapping to Excel Guide:**
   * Separate the two groups into two side-by-side columns: Group A values in `A2:A36` and Group B values in `B2:B36`.
   * **Step 1 (Assumptions):** Compute sample variances via `=VAR.S(...)` and evaluate Levene's test for heteroscedasticity.
   * **Step 2 (Proposed Test):** Execute Welch's t-test directly: `=T.TEST(A2:A36, B2:B36, 2, 3)` (Type 3 = unequal variances).
"""
        python_text = f"""import pandas as pd

# 1. Load the downloaded sample dataset
df = pd.read_csv("{default_filename}")
print(df.head())

# 2. Split by group indicator
groups = [v["Value"].values for _, v in df.groupby("Group")]
group_a, group_b = groups[0], groups[1]

# 3. Run automated Welch pipeline from the Python tab:
results = run_welch_pipeline(group_a, group_b)
"""

    elif test_key in ["students_t_test"]:
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Observation_ID`, **Column B:** `Group`, **Column C:** `Value`
3. **Spreadsheet Mapping to Excel Guide:**
   * Separate observations into Column A (`Group A`) and Column B (`Group B`).
   * **Step 1 (Assumptions):** Confirm variance equality via `=F.TEST(A2:A36, B2:B36)` ($p > 0.05$).
   * **Step 2 (Proposed Test):** Execute Student's t-test: `=T.TEST(A2:A36, B2:B36, 2, 2)` (Type 2 = equal variances).
"""
        python_text = f"""import pandas as pd

df = pd.read_csv("{default_filename}")
groups = [v["Value"].values for _, v in df.groupby("Group")]
group_a, group_b = groups[0], groups[1]

# Run automated Student's t pipeline:
results = run_students_t_pipeline(group_a, group_b)
"""

    elif test_key in ["mann_whitney_u"]:
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Observation_ID`, **Column B:** `Group`, **Column C:** `Value`
3. **Spreadsheet Mapping to Excel Guide:**
   * In Column D, compute combined fractional ranks: `=RANK.AVG(C2, $C$2:$C$71, 1)`.
   * Sum ranks for Group 1 ($R_1$) and Group 2 ($R_2$).
   * Calculate Mann-Whitney $U = \\min(U_1, U_2)$ using formulas in Step 2 of the Excel tab.
"""
        python_text = f"""import pandas as pd

df = pd.read_csv("{default_filename}")
groups = [v["Value"].values for _, v in df.groupby("Group")]
group_a, group_b = groups[0], groups[1]

results = run_mann_whitney_pipeline(group_a, group_b)
"""

    elif test_key in ["paired_t_test"]:
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Subject_ID`
   * **Column B:** `Pre_Intervention`
   * **Column C:** `Post_Intervention`
   * **Column D:** `Difference` (`=C2-B2`)
3. **Spreadsheet Mapping to Excel Guide:**
   * Matches columns `A` through `D` in the **📗 Excel Implementation Guide** tab.
   * **Step 1 (Assumptions):** Audit difference normality and outliers on Column D using `=AVERAGE(D2:D36)`, `=SKEW(D2:D36)`.
   * **Step 2 (Proposed Test):** Execute paired t-test directly: `=T.TEST(C2:C36, B2:B36, 2, 1)` (Type 1 = paired).
"""
        python_text = f"""import pandas as pd

df = pd.read_csv("{default_filename}")
pre = df["Pre_Intervention"]
post = df["Post_Intervention"]

results = run_paired_t_pipeline(pre, post)
"""

    elif test_key in ["wilcoxon_paired"]:
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Subject_ID`, **Column B:** `Pre_Intervention`, **Column C:** `Post_Intervention`, **Column D:** `Difference`
3. **Spreadsheet Mapping to Excel Guide:**
   * In Column E, compute absolute differences `=ABS(D2)`.
   * In Column F, compute fractional ranks on non-zero pairs: `=IF(D2=0, "", RANK.AVG(E2, $E$2:$E$36, 1))`.
   * Sum positive and negative rank totals and obtain Wilcoxon $W = \\min(W^+, W^-)$.
"""
        python_text = f"""import pandas as pd

df = pd.read_csv("{default_filename}")
pre = df["Pre_Intervention"]
post = df["Post_Intervention"]

results = run_wilcoxon_paired_pipeline(pre, post)
"""

    elif test_key in ["one_way_anova"]:
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Observation_ID`, **Column B:** `Group`, **Column C:** `Value`
3. **Spreadsheet Mapping to Excel Guide:**
   * Pivot or arrange the 3 groups side-by-side into Columns A, B, and C (`A2:C26`).
   * Select **Data > Data Analysis > ANOVA: Single Factor**. Select range `A1:C26` and click OK.
   * Or compute $SS_{{\\text{{Between}}}}$ and $SS_{{\\text{{Within}}}}$ manually following Step 2.
"""
        python_text = f"""import pandas as pd

df = pd.read_csv("{default_filename}")
groups = [v["Value"].values for _, v in df.groupby("Group")]

results = run_one_way_anova_pipeline(*groups)
"""

    elif test_key in ["kruskal_wallis"]:
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Observation_ID`, **Column B:** `Group`, **Column C:** `Value`
3. **Spreadsheet Mapping to Excel Guide:**
   * In Column D, assign pooled fractional ranks: `=RANK.AVG(C2, $C$2:$C$76, 1)`.
   * Sum ranks for each group via `=SUMIFS(D:D, B:B, "Group 1")`.
   * Compute Kruskal-Wallis $H$ and $p$-value via `=CHISQ.DIST.RT(H, 2)`.
"""
        python_text = f"""import pandas as pd

df = pd.read_csv("{default_filename}")
groups = [v["Value"].values for _, v in df.groupby("Group")]

results = run_kruskal_wallis_pipeline(*groups)
"""

    elif test_key in ["repeated_measures_anova"]:
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Subject_ID`
   * **Columns B, C, D:** Repeated condition measurements (e.g. `Baseline`, `Month 1`, `Month 3`).
3. **Spreadsheet Mapping to Excel Guide:**
   * Range `B1:D26` matches the spreadsheet layout in the **📗 Excel Implementation Guide** tab.
   * Go to **Data > Data Analysis > ANOVA: Two-Factor Without Replication**, select `B1:D26`, and check Labels.
   * The "Columns" row represents the within-subject omnibus effect.
"""
        python_text = f"""import pandas as pd

df_wide = pd.read_csv("{default_filename}")
c1, c2, c3 = df_wide.iloc[:, 1], df_wide.iloc[:, 2], df_wide.iloc[:, 3]

results = run_repeated_measures_anova_pipeline(c1, c2, c3)
"""

    elif test_key in ["friedman_test"]:
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Subject_ID`, **Columns B, C, D:** Condition scores.
3. **Spreadsheet Mapping to Excel Guide:**
   * In Columns E, F, G, assign within-subject row ranks: `=RANK.AVG(B2, $B2:$D2, 1)`.
   * Sum column rank totals $R_1, R_2, R_3$.
   * Calculate Friedman $\\chi_F^2$ and $p$-value `=CHISQ.DIST.RT(chi_sq, 2)`.
"""
        python_text = f"""import pandas as pd

df_wide = pd.read_csv("{default_filename}")
c1, c2, c3 = df_wide.iloc[:, 1], df_wide.iloc[:, 2], df_wide.iloc[:, 3]

results = run_friedman_pipeline(c1, c2, c3)
"""

    elif test_key in ["chi_square_independence", "contingency_table_2x2"]:
        excel_text = f"""
1. **Open File in Excel:**
   * Open `chi_square_2x2_table.csv` for direct cross-tabulated calculations.
   * Or open `chi_square_subjects.csv` to practice creating a Pivot Table from raw participant records.
2. **Spreadsheet Mapping to Excel Guide:**
   * In `chi_square_2x2_table.csv`, cells `B2:C3` contain observed counts.
   * In cells `B8:C9`, compute expected frequencies: `=(RowTotal * ColTotal) / GrandTotal`.
   * In cell `E11`, compute Chi-Square p-value: `=CHISQ.TEST(B2:C3, B8:C9)`.
"""
        python_text = f"""import pandas as pd

# Load 2x2 contingency matrix
df_table = pd.read_csv("chi_square_2x2_table.csv", index_col=0)
print(df_table)

# Run automated Chi-Square pipeline:
results = run_chi_square_pipeline(df_table.values)
"""

    elif test_key in ["fisher_exact"]:
        excel_text = f"""
1. **Open File in Excel:** Open `fisher_exact_2x2_table.csv`.
2. **Spreadsheet Mapping to Excel Guide:**
   * Observed frequencies $a, b, c, d$ are in cells `B2:C3`.
   * Marginal totals are in `D2:D3` and `B4:C4`.
   * Compute exact hypergeometric point probability: `=HYPGEOM.DIST(B2, D2, B4, D4, FALSE)`.
"""
        python_text = f"""import pandas as pd

df_table = pd.read_csv("fisher_exact_2x2_table.csv", index_col=0)
print(df_table)

results = run_fisher_exact_pipeline(df_table.values)
"""

    elif test_key in ["mcnemar_test", "paired_binary"]:
        excel_text = f"""
1. **Open File in Excel:** Open `mcnemar_2x2_table.csv`.
2. **Spreadsheet Mapping to Excel Guide:**
   * Off-diagonal discordant switchers are in cells `C2` ($b$) and `B3` ($c$).
   * If $b+c \\ge 25$: Edwards' chi-square `=((ABS(C2-B3)-1)^2)/(C2+B3)`.
   * If $b+c < 25$: Exact Binomial p-value `=2*BINOM.DIST(MIN(B3,C2), B3+C2, 0.5, TRUE)`.
"""
        python_text = f"""import pandas as pd

df_table = pd.read_csv("mcnemar_2x2_table.csv", index_col=0)
print(df_table)

results = run_mcnemar_pipeline(df_table.values)
"""

    elif test_key in ["pearson_correlation", "correlation_continuous"]:
        n = len(demo_df)
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Participant_ID`, **Column B:** Predictor $X$, **Column C:** Outcome $Y$
3. **Spreadsheet Mapping to Excel Guide:**
   * Follow Step 1 for normality & bivariate outlier auditing.
   * In cell `G5`, calculate Pearson's $r$: `=CORREL(B2:B{n+1}, C2:C{n+1})`.
   * In cell `G7`, compute $t$-statistic: `=G5*SQRT(G4-2)/SQRT(1-G5^2)`.
   * In cell `G8`, compute two-sided $p$-value: `=T.DIST.2T(ABS(G7), G4-2)`.
"""
        python_text = f"""import pandas as pd

df = pd.read_csv("{default_filename}")
x = df.iloc[:, 1]
y = df.iloc[:, 2]

results = run_pearson_pipeline(x, y)
"""

    elif test_key in ["spearman_correlation", "spearman_rank"]:
        n = len(demo_df)
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Participant_ID`, **Column B:** Variable $X$, **Column C:** Variable $Y$
   * **Columns D & E:** Fractional ranks (`=RANK.AVG(B2, $B$2:$B${n+1}, 1)`, `=RANK.AVG(C2, $C$2:$C${n+1}, 1)`)
   * **Column F:** Squared rank differences $d_i^2 = (R_X - R_Y)^2$ via `=(D2-E2)^2`.
3. **Spreadsheet Mapping to Excel Guide:**
   * Compute Spearman's $\\rho$: `=CORREL(D2:D{n+1}, E2:E{n+1})`.
"""
        python_text = f"""import pandas as pd

df = pd.read_csv("{default_filename}")
x = df.iloc[:, 1]
y = df.iloc[:, 2]

results = run_spearman_pipeline(x, y)
"""

    elif test_key in ["linear_regression", "multiple_linear_regression"]:
        n = len(demo_df)
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Obs_ID`, **Column B:** Outcome $Y$, **Columns C+:** Predictors $X_1, X_2, \\dots$
3. **Spreadsheet Mapping to Excel Guide:**
   * Navigate to **Data > Data Analysis > Regression**. Select Column B as Input Y Range, and Columns C+ as Input X Range.
   * Or enter array formula `=LINEST(B2:B{n+1}, C2:D{n+1}, TRUE, TRUE)`.
"""
        python_text = f"""import pandas as pd

df = pd.read_csv("{default_filename}")
y = df.iloc[:, 1]
X = df.iloc[:, 2:]

results = run_linear_regression_pipeline(X, y)
"""

    elif test_key in ["binary_logistic_regression", "binary_classification", "logistic_regression"]:
        n = len(demo_df)
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Column Structure:**
   * **Column A:** `Obs_ID`, **Column B:** Dichotomous Outcome $Y$ (0 or 1), **Columns C+:** Predictor Variables $X_1, X_2, \\dots$
3. **Spreadsheet Mapping to Excel Guide:**
   * Range `A1:E{n+1}` corresponds directly to the layout detailed in the **📗 Excel Implementation Guide** tab.
   * Parameter cells `B29:E29` hold the regression intercept and slopes.
   * In Column F, compute linear logit: `=$B$29 + $C$29*C2 + $D$29*D2 + ...`
   * In Column G, compute predicted probability: `=1 / (1 + EXP(-F2))`.
   * In Column H, compute Bernoulli log-likelihood: `=B2*LN(G2) + (1-B2)*LN(1-G2)`.
   * Open **Data > Solver**, set Objective `=SUM(H2:H{n+1})` to **Max**, By Changing Variable Cells `$B$29:$E$29`, and choose **GRG Nonlinear**.
   * Calculate adjusted Odds Ratios: `=EXP(B29)`, `=EXP(C29)`, etc.
"""
        python_text = f"""import pandas as pd

# 1. Load downloaded sample dataset
df = pd.read_csv("{default_filename}")
print(f"Loaded dataset: {{len(df)}} observations")
print(df.head())

# 2. Separate binary outcome y and predictor matrix X
y = df.iloc[:, 1]
X = df.iloc[:, 2:]
feature_names = list(X.columns)

# 3. Execute automated assumption checking and model pipeline:
# (Copy run_binary_logistic_pipeline from Step 4 of the Python Code tab)
results = run_binary_logistic_pipeline(X, y, feature_names=feature_names)
"""

    else:
        excel_text = f"""
1. **Open File in Excel:** Open `{default_filename}`.
2. **Spreadsheet Mapping:** Reference the columns to work through the formulas in the **📗 Excel Implementation Guide** tab.
"""
        python_text = f"""import pandas as pd

df = pd.read_csv("{default_filename}")
print(df.head())
"""

    return {
        "default_filename": default_filename,
        "excel_instructions": excel_text.strip(),
        "python_code": python_text.strip()
    }


def _render_contingency_sample_data_section(
    test_key: str,
    test_name: str,
    demo_df: pd.DataFrame,
    stats_results: Dict[str, Any]
):
    """Renders the sample data download UI for 2x2 categorical & contingency tests."""
    df_cases = stats_results.get("df_cases")
    guide_info = _get_sample_data_guide_info(test_key, {"name": test_name}, demo_df, stats_results)

    st.markdown("##### 📋 Data Previews")
    st.markdown("**1. 2×2 Contingency Table Summary Matrix**")
    render_table(demo_df, height=140)

    if df_cases is not None:
        st.markdown(f"**2. Subject-Level Observational Data Preview (N = {len(df_cases)} Records)**")
        render_table(df_cases.head(15), height=220)

    st.markdown("##### 💾 Download Practice Files")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        if df_cases is not None:
            st.download_button(
                "📥 Download Participant-Level Data (CSV)",
                data=df_cases.to_csv(index=False).encode("utf-8"),
                file_name=f"{test_key}_subjects.csv",
                mime="text/csv",
                key=f"btn_dl_subj_{test_key}",
                type="primary",
                help="Individual participant records (ideal for practicing Pivot Tables & COUNTIFS in Excel or pd.crosstab in Python)."
            )
        else:
            st.download_button(
                "📥 Download Sample Data (CSV)",
                data=demo_df.to_csv(index=True).encode("utf-8"),
                file_name=f"{test_key}_table.csv",
                mime="text/csv",
                key=f"btn_dl_table_only_{test_key}",
                type="primary"
            )
    with col_dl2:
        st.download_button(
            "📥 Download 2×2 Contingency Matrix (CSV)",
            data=demo_df.to_csv(index=True).encode("utf-8"),
            file_name=f"{test_key}_2x2_table.csv",
            mime="text/csv",
            key=f"btn_dl_matrix_{test_key}",
            help="Cross-tabulated 2x2 summary table (ideal for practicing direct cell formulas)."
        )

    with st.expander("📌 How to Work with this CSV in Microsoft Excel", expanded=False):
        st.markdown(guide_info["excel_instructions"])

    with st.expander("📌 How to Work with this CSV in Python", expanded=False):
        st.code(guide_info["python_code"], language="python")


def _render_repeated_measures_sample_data_section(
    test_key: str,
    test_name: str,
    demo_df: pd.DataFrame,
    stats_results: Dict[str, Any]
):
    """Renders the sample data download UI for Repeated Measures ANOVA and Friedman Test."""
    df_wide = _get_rm_wide_dataframe(demo_df, stats_results)
    guide_info = _get_sample_data_guide_info(test_key, {"name": test_name}, df_wide, stats_results)

    st.markdown("##### 📋 Data Previews")
    st.markdown(f"**1. Matched Wide-Format Dataset Preview (N = {len(df_wide)} Subjects)**")
    st.caption("Each subject occupies one row with separate columns for each measurement occasion (ideal for Excel ANOVA).")
    render_table(df_wide.head(15), height=220)

    st.markdown(f"**2. Long / Tidy Format Dataset Preview (N = {len(demo_df)} Observations)**")
    st.caption("Stacked observation rows with Subject_ID, Condition/Time, and Score (ideal for Python / R).")
    render_table(demo_df.head(15), height=220)

    st.markdown("##### 💾 Download Practice Files")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Download Wide-Format Dataset (CSV)",
            data=df_wide.to_csv(index=False).encode("utf-8"),
            file_name=f"{test_key}_wide.csv",
            mime="text/csv",
            key=f"btn_dl_wide_{test_key}",
            type="primary",
            help="Matched repeated measures matrix (ideal for Microsoft Excel)."
        )
    with col_dl2:
        st.download_button(
            "📥 Download Long / Tidy Dataset (CSV)",
            data=demo_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{test_key}_long.csv",
            mime="text/csv",
            key=f"btn_dl_long_{test_key}",
            help="Tidy long format (ideal for Python / statsmodels)."
        )

    with st.expander("📌 How to Work with this CSV in Microsoft Excel", expanded=False):
        st.markdown(guide_info["excel_instructions"])

    with st.expander("📌 How to Work with this CSV in Python", expanded=False):
        st.code(guide_info["python_code"], language="python")


def _render_standard_sample_data_section(
    test_key: str,
    test_name: str,
    demo_df: pd.DataFrame,
    stats_results: Dict[str, Any]
):
    """Renders the sample data download UI for standard columnar datasets."""
    guide_info = _get_sample_data_guide_info(test_key, {"name": test_name}, demo_df, stats_results)
    filename = guide_info["default_filename"]

    st.markdown("##### 📋 Dataset Preview")
    st.caption(f"Displaying the first {min(15, len(demo_df))} observations (Total: {len(demo_df)} rows):")
    render_table(demo_df.head(15), height=240)

    st.markdown("##### 💾 Download Practice File")
    st.download_button(
        f"📥 Download {test_name} Dataset as CSV ({filename})",
        data=demo_df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"btn_dl_main_{test_key}",
        type="primary",
        help=f"Download {len(demo_df)} records as a CSV file to practice in Excel or Python."
    )

    with st.expander("📌 How to Work with this CSV in Microsoft Excel", expanded=False):
        st.markdown(guide_info["excel_instructions"])

    with st.expander("📌 How to Work with this CSV in Python", expanded=False):
        st.code(guide_info["python_code"], language="python")


def render_sample_data_tab(
    rec: Dict[str, Any],
    demo_df: Optional[pd.DataFrame] = None,
    stats_results: Optional[Dict[str, Any]] = None
):
    """Renders the comprehensive '📥 Download Sample Data' tab allowing users to inspect and download
    the active dataset as a CSV file to practice in Excel or Python.
    """
    test_key = rec.get("test_key", "statistical_test")
    test_name = rec.get("name", "Statistical Test")
    stats_results = stats_results or {}

    # 1. Fallback dataset generation if demo_df is not provided
    if demo_df is None:
        demo_df, fallback_res = _generate_fallback_dataset(test_key)
        stats_results = {**fallback_res, **stats_results}

    st.markdown(f"### 📥 Download Practice Sample Data for {test_name}")
    st.markdown(
        "This tab allows you to download the **exact demonstration dataset** currently configured in the sidebar menu on the left. "
        "You can save this CSV file to your local computer to practice executing the step-by-step instructions in the "
        "**📗 Excel Implementation Guide** tab or running the code scripts in the **🐍 Python Code Implementation** tab."
    )

    # 2. Metadata Context Grid
    scenario_label = stats_results.get("preset_name") or stats_results.get("scenario") or "Active Menu Parameters"
    n_records = len(demo_df) if demo_df is not None else 0
    n_cols = len(demo_df.columns) if demo_df is not None else 0

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.info(f"🏷️ **Active Scenario:**\n{scenario_label}")
    with col_meta2:
        st.info(f"🔢 **Sample Size:**\n{n_records} records")
    with col_meta3:
        st.info(f"📊 **Dimensions:**\n{n_records} Rows × {n_cols} Cols")
    with col_meta4:
        st.info(f"💾 **File Format:**\nComma-Separated (.csv)")

    st.markdown("---")

    # 3. Render test family specific download layout
    if test_key in ["chi_square_independence", "contingency_table_2x2", "fisher_exact", "mcnemar_test", "paired_binary"]:
        _render_contingency_sample_data_section(test_key, test_name, demo_df, stats_results)
    elif test_key in ["repeated_measures_anova", "friedman_test"]:
        _render_repeated_measures_sample_data_section(test_key, test_name, demo_df, stats_results)
    else:
        _render_standard_sample_data_section(test_key, test_name, demo_df, stats_results)



def render_one_sample_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for the One-Sample t-test."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Outliers & Normality)", expanded=False):
        st.markdown(
            "Before interpreting a t-test, verify there are no extreme outliers "
            "and that data is approximately normal (or N >= 30)."
        )

        # 1. Generate deterministic sample data
        rng = np.random.default_rng(42)
        sample_vals = np.round(rng.normal(loc=106.0, scale=12.0, size=35), 2)
        benchmark_vals = np.full(35, 100.0)
        diff_vals = np.round(sample_vals - benchmark_vals, 2)

        # 2. Outlier detection using Tukey's fences (IQR)
        q25 = float(np.percentile(sample_vals, 25))
        q75 = float(np.percentile(sample_vals, 75))
        iqr = q75 - q25
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr

        outlier_flags = np.where(
            (sample_vals < lower_fence) | (sample_vals > upper_fence),
            "OUTLIER",
            "OK"
        )

        example_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(len(sample_vals))],
            "Sample_Value_A": sample_vals,
            "Benchmark_mu0": benchmark_vals,
            "Difference": diff_vals,
            "Outlier_Check": outlier_flags
        })

        # 3. Dynamic diagnostics (do not hardcode PASS/FAIL)
        n_count = len(sample_vals)
        outlier_count = int(np.sum(outlier_flags == "OUTLIER"))
        skew_val = float(stats.skew(sample_vals))
        kurt_val = float(stats.kurtosis(sample_vals))

        n_pass = "PASS" if n_count >= 30 else "WARN"
        outlier_pass = "PASS" if outlier_count == 0 else "FAIL"
        skew_pass = "PASS" if abs(skew_val) <= 0.5 else "WARN"
        kurt_pass = "PASS" if abs(kurt_val) <= 1.0 else "WARN"

        summary_df = pd.DataFrame({
            "Assumption_Metric": [
                "Sample Size (N)",
                "Outlier Count (>1.5x IQR)",
                "Sample Skewness",
                "Excess Kurtosis"
            ],
            "Value": [
                f"{n_count}",
                f"{outlier_count}",
                f"{skew_val:.3f}",
                f"{kurt_val:.3f}"
            ],
            "Rule_Threshold": [
                "N >= 30",
                "0 Extreme Outliers",
                "Between -0.5 and +0.5",
                "Between -1.0 and +1.0"
            ],
            "Assumption_Result": [
                n_pass,
                outlier_pass,
                skew_pass,
                kurt_pass
            ]
        })

        st.markdown("##### 📋 Example Spreadsheet Data (Range: A2:A36)")
        render_table(example_df, height=240)

        st.markdown("##### 📊 Assumption Diagnostic Summary")
        render_table(summary_df, height=240)

        st.caption(
            "Excel Formulas: Q1: `=PERCENTILE.INC(A2:A36, 0.25)` | "
            "Q3: `=PERCENTILE.INC(A2:A36, 0.75)` | "
            "Skew: `=SKEW(A2:A36)` | Kurtosis: `=KURT(A2:A36)`"
        )
    
    # STEP 2: IMPLEMENT THE PROPOSED TEST IN EXCEL
    with st.expander("📌 Step 2: Implement One-Sample t-Test in Excel", expanded=False):
        st.markdown("""
        Calculate the test statistic, $p$-value, 95% Confidence Interval, and effect size against your theoretical benchmark $\\mu_0$:
        """)
        st.code("""
Sample Size (n):           =COUNT(A2:A36)
Sample Mean (x̄):           =AVERAGE(A2:A36)
Sample Std Dev (s):         =STDEV.S(A2:A36)
Theoretical Benchmark (μ₀): 100.0   [Enter your reference value in cell D1]

Mean Difference (x̄ - μ₀):   =AVERAGE(A2:A36) - D1
Standard Error (SE):        =STDEV.S(A2:A36) / SQRT(COUNT(A2:A36))
Degrees of Freedom (df):    =COUNT(A2:A36) - 1

t-Statistic:                =(AVERAGE(A2:A36) - D1) / (STDEV.S(A2:A36) / SQRT(COUNT(A2:A36)))
Two-Tailed p-value:         =T.DIST.2T(ABS(t_stat_cell), df_cell)

95% CI Margin (Half-Width): =CONFIDENCE.T(0.05, STDEV.S(A2:A36), COUNT(A2:A36))
95% CI Lower Limit:         =AVERAGE(A2:A36) - ci_margin_cell
95% CI Upper Limit:         =AVERAGE(A2:A36) + ci_margin_cell

Effect Size (Cohen's d):    =(AVERAGE(A2:A36) - D1) / STDEV.S(A2:A36)
""", language="excel")

        st.markdown("""
        **🔍 How to Evaluate & Interpret the Output:**
        * **$p$-value $< 0.05$:** Statistically significant difference between the sample mean and the benchmark $\\mu_0$. Reject the null hypothesis ($H_0: \\mu = \\mu_0$).
        * **$p$-value $\\ge 0.05$:** Fail to reject $H_0$. There is insufficient evidence to conclude the sample mean differs from the benchmark.
        * **95% Confidence Interval:** Represents the range of plausible values for the true population mean. If the theoretical benchmark $\\mu_0$ lies **outside** this interval, the difference is statistically significant at $\\alpha = 0.05$.
        * **Cohen's $d$ Effect Size:**
          * $|d| \\approx 0.2$: Small effect (subtle shift).
          * $|d| \\approx 0.5$: Medium effect (practically meaningful shift).
          * $|d| \\ge 0.8$: Large effect (substantial clinical/practical shift).
        """)

        st.info("""
        🛠️ **Toolpak Alternative (No manual math required):**
        1. Click **Data Tab ➔ Data Analysis ➔ Descriptive Statistics**.
        2. Set Input Range to `A2:A36`, check **Summary statistics** and **Confidence Level for Mean (95%)**.
        3. Excel outputs Mean, Standard Error, and the 95% Confidence Margin automatically!
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TEST IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Test in Excel (When Assumptions Fail)", expanded=False):
        st.markdown("""
        When normality fails or severe outliers distort the mean, switch to a non-parametric test comparing the **median** to the benchmark:
        """)

        st.markdown("#### Option A: Exact Sign Test (Native Single-Formula Method)")
        st.markdown("Tests whether values are equally likely to fall above or below the benchmark $\\mu_0$ using the binomial distribution:")
        st.code("""
In Column B: Differences from Benchmark
=A2 - $D$1

Count Positive Differences: =COUNTIF(B2:B36, ">0")
Count Non-Zero Differences: =COUNTIF(B2:B36, "<>0")

Two-Tailed Sign Test p-value:
=2 * BINOM.DIST(MIN(pos_cell, nonzero_cell - pos_cell), nonzero_cell, 0.5, TRUE)
""", language="excel")

        st.markdown("#### Option B: Wilcoxon Signed-Rank Test (Ranked Method)")
        st.code("""
In Column C: Absolute Differences   =ABS(A2 - $D$1)
In Column D: Average Ranks          =IF(A2=$D$1, "", RANK.AVG(C2, $C$2:$C$36, 1))
In Column E: Positive Signed Ranks  =IF(A2>$D$1, D2, 0)
Wilcoxon W Statistic:               =SUM(E2:E36)
""", language="excel")

        st.markdown("""
        **🔍 How to Interpret Alternative Output:**
        * The Sign and Wilcoxon tests evaluate the **median** rather than the arithmetic mean.
        * If the resulting $p$-value $< 0.05$, the sample median differs significantly from the theoretical benchmark $\\mu_0$.
        * Because these tests rely on signs or rank orderings, they are **immune to skewness and extreme outliers**.
        """)


def render_one_sample_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python script guide for the One-Sample t-test."""
    st.markdown("""
    ### 🐍 Python Code Implementation Workflow
    This script is divided into 3 modular sections: **Evaluate Assumptions** ➔ **Execute Primary Test** ➔ **Execute Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Outliers & Normality)", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

# Given sample_data (numpy array or pandas series) and benchmark mu_0
# sample_data = np.array([...])
# benchmark = 100.0
n = len(sample_data)

# 1. Outlier Check via Tukey's IQR Method
q25, q75 = np.percentile(sample_data, [25, 75])
iqr = q75 - q25
lower_fence = q25 - 1.5 * iqr
upper_fence = q75 + 1.5 * iqr
outliers = sample_data[(sample_data < lower_fence) | (sample_data > upper_fence)]
extreme_outliers = sample_data[(sample_data < q25 - 3 * iqr) | (sample_data > q75 + 3 * iqr)]

print(f"Mild Outliers: {len(outliers)} | Extreme Outliers (>3xIQR): {len(extreme_outliers)}")

# 2. Normality Check via Shapiro-Wilk Test & Skewness
w_stat, shapiro_p = stats.shapiro(sample_data)
skewness = stats.skew(sample_data)
kurtosis = stats.kurtosis(sample_data)

print(f"Shapiro-Wilk Test: W = {w_stat:.4f}, p-value = {shapiro_p:.4f}")
print(f"Skewness: {skewness:.2f} | Kurtosis: {kurtosis:.2f}")
""", language="python")

        st.markdown("""
        **🔍 How to Interpret the Output:**
        * **Outliers:** If `len(extreme_outliers) > 0`, extreme values will distort the sample mean and inflate the standard error.
        * **Normality:**
          * `shapiro_p >= 0.05`: Normality assumption **satisfied**.
          * `shapiro_p < 0.05` and `n >= 30`: Moderate skewness detected, but **Central Limit Theorem applies**. One-Sample t-test is acceptable.
          * `shapiro_p < 0.05` and `n < 30`: Normality assumption **violated**. Switch to Step 3 (Wilcoxon Signed-Rank Test).
        """)

    # STEP 2: IMPLEMENT ONE-SAMPLE T-TEST IN PYTHON
    with st.expander("📌 Step 2: Implement One-Sample t-Test in Python", expanded=False):
        st.code("""# 1. Execute One-Sample t-test
t_stat, p_val = stats.ttest_1samp(sample_data, popmean=benchmark)

# 2. Compute Summary Metrics & 95% Confidence Interval
df = n - 1
mean_val = np.mean(sample_data)
std_val = np.std(sample_data, ddof=1)
se_val = std_val / np.sqrt(n)

t_crit = stats.t.ppf(0.975, df=df)
ci_lower = mean_val - t_crit * se_val
ci_upper = mean_val + t_crit * se_val
cohens_d = (mean_val - benchmark) / std_val

print(f"Sample Mean (x̄): {mean_val:.2f} (Benchmark μ₀: {benchmark})")
print(f"t({df}) = {t_stat:.4f}, p-value = {p_val:.4e}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Cohen's d Effect Size: {cohens_d:.2f}")
""", language="python")

        st.markdown("""
        **🔍 How to Evaluate & Interpret the Output:**
        * `p_val < 0.05`: Reject $H_0$. The sample mean differs significantly from the theoretical benchmark.
        * `p_val >= 0.05`: Fail to reject $H_0$. No statistically significant difference detected.
        * **Confidence Interval:** If `benchmark` falls outside `[ci_lower, ci_upper]`, the difference is statistically significant.
        * **Cohen's d:** $0.2$ = small, $0.5$ = medium, $0.8$ = large effect.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TEST IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Test (Wilcoxon Signed-Rank Test) in Python", expanded=False):
        st.code("""# Use when normality is violated or extreme outliers exist
diffs = sample_data - benchmark
nonzero_diffs = diffs[diffs != 0]

w_stat, wilc_p = stats.wilcoxon(nonzero_diffs, alternative="two-sided")
sample_median = np.median(sample_data)

print(f"Sample Median: {sample_median:.2f} (Benchmark: {benchmark})")
print(f"Wilcoxon Signed-Rank Test: W = {w_stat:.2f}, p-value = {wilc_p:.4f}")
""", language="python")

        st.markdown("""
        **🔍 How to Interpret Alternative Output:**
        * Tests whether the **median** of differences from the benchmark is zero.
        * `wilc_p < 0.05`: The population median differs significantly from the benchmark.
        * Robust against non-normality and outliers because it analyzes ranked differences.
        """)

    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

def run_one_sample_analysis(sample_data, benchmark=100.0, alpha=0.05):
    n = len(sample_data)
    print(f"=== ONE-SAMPLE ANALYSIS (N = {n}, Benchmark = {benchmark}) ===\\n")
    
    # 1. Check Assumptions
    w_stat, shapiro_p = stats.shapiro(sample_data)
    q25, q75 = np.percentile(sample_data, [25, 75])
    iqr = q75 - q25
    extreme_outliers = sample_data[(sample_data < q25 - 3 * iqr) | (sample_data > q75 + 3 * iqr)]
    
    normal_pass = (shapiro_p >= alpha) or (n >= 30 and abs(stats.skew(sample_data)) <= 1.2)
    outlier_pass = len(extreme_outliers) == 0
    
    print(f"[Assumption] Normality: Shapiro p = {shapiro_p:.4f} -> {'PASS' if normal_pass else 'VIOLATED'}")
    print(f"[Assumption] Outliers: {len(extreme_outliers)} extreme outliers -> {'PASS' if outlier_pass else 'VIOLATED'}\\n")
    
    # 2. Decision Routing
    if normal_pass and outlier_pass:
        print("--> Proceeding with One-Sample t-test:")
        t_stat, p_val = stats.ttest_1samp(sample_data, popmean=benchmark)
        mean_val = np.mean(sample_data)
        std_val = np.std(sample_data, ddof=1)
        se = std_val / np.sqrt(n)
        ci = (mean_val - 1.96 * se, mean_val + 1.96 * se)
        d = (mean_val - benchmark) / std_val
        print(f"t({n-1}) = {t_stat:.4f}, p = {p_val:.4e}, 95% CI = [{ci[0]:.2f}, {ci[1]:.2f}], d = {d:.2f}")
    else:
        print("--> Assumptions violated! Switching to Wilcoxon Signed-Rank Test:")
        diffs = sample_data - benchmark
        w_stat, w_p = stats.wilcoxon(diffs[diffs != 0])
        print(f"Median = {np.median(sample_data):.2f}, Wilcoxon W = {w_stat:.2f}, p = {w_p:.4f}")

# Example Run:
# run_one_sample_analysis(my_data, benchmark=100.0)
""", language="python")


def render_wilcoxon_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for the Wilcoxon Signed-Rank Test."""
    st.markdown(r"""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Scale, Independence & Symmetry)", expanded=False):
        st.markdown(r"""
        The Wilcoxon Signed-Rank Test requires that data is at least **ordinal**, observations are **independent**, 
        and differences from the benchmark $\mu_0$ are **approximately symmetric around the median**.
        """)

        # 1. Generate deterministic sample data & derived Wilcoxon ranks
        rng = np.random.default_rng(42)
        raw_demo = rng.laplace(loc=0.0, scale=10.6, size=30)
        sample_vals = np.round(raw_demo + 108.0, 1)
        benchmark_val = 100.0
        diff_vals = np.round(sample_vals - benchmark_val, 1)
        abs_diffs = np.round(np.abs(diff_vals), 1)
        ranks = stats.rankdata(abs_diffs)
        pos_ranks = np.where(diff_vals > 0, ranks, 0.0)
        neg_ranks = np.where(diff_vals < 0, ranks, 0.0)

        example_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(len(sample_vals))],
            "Col_A_Value [A2:A31]": sample_vals,
            "Col_B_Diff (=A2-$D$1)": diff_vals,
            "Col_C_AbsDiff (=ABS(B2))": abs_diffs,
            "Col_D_Rank (=RANK.AVG)": ranks,
            "Col_E_PosRank": pos_ranks,
            "Col_F_NegRank": neg_ranks
        })

        # 2. Dynamic diagnostics (do not hardcode PASS/FAIL)
        n_total = len(sample_vals)
        zero_count = int(np.sum(diff_vals == 0.0))
        n_eff = n_total - zero_count
        nonzero_diffs = diff_vals[diff_vals != 0.0]
        skew_val = float(stats.skew(nonzero_diffs))
        w_plus = float(np.sum(pos_ranks))
        w_minus = float(np.sum(neg_ranks))

        n_pass = "PASS" if n_eff >= 10 else ("WARN" if n_eff >= 6 else "FAIL")
        zero_pass = "PASS" if zero_count <= 2 else "WARN"
        skew_pass = "PASS" if abs(skew_val) <= 0.8 else ("WARN" if abs(skew_val) <= 1.5 else "FAIL")

        summary_df = pd.DataFrame({
            "Assumption_Metric": [
                "Total Observations (N)",
                "Zero Differences (d = 0)",
                "Effective Sample Size (N_eff)",
                "Difference Skewness (Symmetry)",
                "Sum Positive Ranks (W+)",
                "Sum Negative Ranks (W-)"
            ],
            "Value": [
                f"{n_total}",
                f"{zero_count}",
                f"{n_eff}",
                f"{skew_val:.3f}",
                f"{w_plus:.1f}",
                f"{w_minus:.1f}"
            ],
            "Rule_Threshold": [
                "N >= 10",
                "Minimal zero ties",
                "N_eff >= 10 (CLT robust)",
                "Between -0.8 and +0.8 (Symmetric)",
                "Sum of positive ranks",
                "Sum of negative ranks"
            ],
            "Assumption_Result": [
                "PASS",
                zero_pass,
                n_pass,
                skew_pass,
                "INFO",
                "INFO"
            ]
        })

        st.markdown("##### 📋 Example Spreadsheet Data (Range: A2:A31, Benchmark μ₀ = 100.0 in Cell G1)")
        render_table(example_df, height=240)

        st.markdown("##### 📊 Assumption Diagnostic Summary")
        render_table(summary_df, height=240)

        st.markdown("#### 1A. Testing Difference Symmetry around the Median")
        st.markdown(r"""
        Assuming sample data in column `A2:A31` and benchmark $\mu_0$ in cell `G1`:
        In Column B, calculate differences: `=A2 - $G$1`.
        """)
        st.code("""
In Column B (Difference):        =A2 - $G$1
Sample Skewness of Differences: =SKEW(B2:B31)
Count of Zero Differences:      =COUNTIF(B2:B31, 0)
Effective Sample Size (N_eff):  =COUNT(B2:B31) - COUNTIF(B2:B31, 0)
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret Assumption Output:**
        * **Symmetry Check (`=SKEW(B2:B31)`):**
          * **$|\text{SKEW}| \le 0.8$:** Differences are symmetric around the median. Assumption **PASSED**. The Wilcoxon Signed-Rank Test is a pure test of the population median.
          * **$0.8 < |\text{SKEW}| \le 1.5$:** Moderate asymmetry. The test evaluates the *Hodges-Lehmann pseudo-median* (Walsh averages). Proceed to Step 2, but check concordance with the Sign Test in Step 3.
          * **$|\text{SKEW}| > 1.5$:** Severe asymmetry. Symmetry assumption is **VIOLATED**. The Wilcoxon rank sum confounds location with shape. **Skip to Step 3 (Exact Sign Test)**.
        * **Effective Sample Size ($N_{\text{eff}}$):**
          * $N_{\text{eff}} \ge 10$: Adequate statistical power and reliable normal approximation.
          * $N_{\text{eff}} < 6$: Insufficient data after dropping zero ties.
        """)
        st.caption(r"💡 **Tip:** Create a histogram of Column B (`Insert ➔ Charts ➔ Histogram`) to visually inspect symmetry around 0.")

    # STEP 2: IMPLEMENT THE PROPOSED TEST IN EXCEL
    with st.expander("📌 Step 2: Implement Wilcoxon Signed-Rank Test in Excel", expanded=False):
        st.markdown(r"""
        Standard Excel does not have a single `=WILCOXON.TEST()` formula, but you can compute it transparently using ranks:
        """)
        st.code("""
Col A (Values):               [Raw Data]
Col B (Differences):          =A2 - $G$1
Col C (Absolute Differences): =ABS(B2)
Col D (Ranks):                =IF(B2=0, "", RANK.AVG(C2, $C$2:$C$31, 1))
Col E (Positive Ranks):       =IF(B2>0, D2, 0)
Col F (Negative Ranks):       =IF(B2<0, D2, 0)

Summary Statistics:
Sum of Positive Ranks (W+):   =SUM(E2:E31)
Sum of Negative Ranks (W-):   =SUM(F2:F31)
Wilcoxon Test Statistic (W):  =MIN(SUM(E2:E31), SUM(F2:F31))
Effective Sample Size (n):    =COUNTIF(B2:B31, "<>0")

Large-Sample Normal Approximation (for n >= 10):
Expected Mean of W (μ_W):     =(n_eff * (n_eff + 1)) / 4
Std Error of W (σ_W):         =SQRT((n_eff * (n_eff + 1) * (2 * n_eff + 1)) / 24)
Z-Score (with continuity):    =(ABS(W_cell - μ_W_cell) - 0.5) / σ_W_cell
Two-Tailed p-value:           =2 * (1 - NORM.S.DIST(ABS(Z_cell), TRUE))
""", language="excel")

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output:**
        * **$p$-value $< 0.05$:** Statistically significant difference between the population median and the benchmark $\mu_0$. Reject $H_0$.
        * **$p$-value $\ge 0.05$:** Fail to reject $H_0$. Insufficient evidence of a median shift.
        * **Small Sample Note ($N_{\text{eff}} < 20$):** Rather than the large-sample Z approximation, compare the smaller rank sum $W$ directly to a published Wilcoxon Signed-Rank critical values table ($W \le W_{\text{crit}}$ indicates significance).
        * **Hodges-Lehmann Shift:** The median of all pairwise Walsh averages $\frac{d_i + d_j}{2}$ reflects the non-parametric location shift.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TEST IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Test in Excel (When Symmetry Fails: Exact Sign Test)", expanded=False):
        st.markdown(r"""
        If differences from the benchmark are **severely asymmetric** ($|\text{SKEW}| > 1.5$), switch to the **Exact Sign Test**. 
        The Sign Test makes zero symmetry assumptions, testing purely whether values are equally likely to be positive or negative.
        """)
        st.code("""
In Column B: Differences =A2 - $G$1

Count Positive Differences (k): =COUNTIF(B2:B31, ">0")
Count Non-Zero Differences (n): =COUNTIF(B2:B31, "<>0")

Two-Tailed Exact Sign Test p-value:
=2 * BINOM.DIST(MIN(k_cell, n_cell - k_cell), n_cell, 0.5, TRUE)
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret the Sign Test Output:**
        * Evaluates the null hypothesis that $P(X > \mu_0) = 0.5$.
        * If $p < 0.05$, the sample median differs significantly from $\mu_0$.
        * Completely robust against extreme skewness, heavy tails, and asymmetric outliers.
        """)


def render_wilcoxon_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python script guide for the Wilcoxon Signed-Rank Test."""
    st.markdown(r"""
    ### 🐍 Python Code Implementation Workflow
    This script is divided into 3 modular sections: **Evaluate Assumptions** ➔ **Execute Proposed Test** ➔ **Execute Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Scale, Independence & Symmetry)", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

# Given sample_data (numpy array) and theoretical benchmark mu_0
# benchmark = 100.0
diffs = sample_data - benchmark
nonzero_diffs = diffs[diffs != 0]
n_eff = len(nonzero_diffs)
n_zeros = len(sample_data) - n_eff

# 1. Symmetry Check of Differences
skew_diffs = stats.skew(nonzero_diffs)
print(f"Effective Sample Size: {n_eff} (Zero ties dropped: {n_zeros})")
print(f"Skewness of Differences: {skew_diffs:.2f}")

# How to interpret:
# - |skew_diffs| <= 0.8: Differences are symmetric. Assumption PASSED.
# - 0.8 < |skew_diffs| <= 1.5: Moderate asymmetry. Test evaluates Walsh averages.
# - |skew_diffs| > 1.5: Asymmetry VIOLATED -> Proceed to Step 3 (Exact Sign Test).
""", language="python")

    # STEP 2: IMPLEMENT WILCOXON SIGNED-RANK TEST IN PYTHON
    with st.expander("📌 Step 2: Implement Wilcoxon Signed-Rank Test in Python", expanded=False):
        st.code("""# 1. Execute Wilcoxon Signed-Rank Test
# zero_method='wilcox' drops zeros; correction=True applies continuity correction
res = stats.wilcoxon(nonzero_diffs, alternative="two-sided")
w_stat = res.statistic
p_val = res.pvalue

# 2. Compute Median and Hodges-Lehmann Location Shift
sample_median = np.median(sample_data)
median_diff = sample_median - benchmark

# Pairwise Walsh averages for Hodges-Lehmann estimate
walsh_diffs = [(nonzero_diffs[i] + nonzero_diffs[j]) / 2.0 
               for i in range(n_eff) for j in range(i, n_eff)]
hl_estimate = np.median(walsh_diffs)

print(f"Sample Median: {sample_median:.2f} (Benchmark: {benchmark})")
print(f"Median Difference: {median_diff:+.2f}")
print(f"Hodges-Lehmann Shift Estimate: {hl_estimate:+.2f}")
print(f"Wilcoxon W: {w_stat:.1f}, p-value: {p_val:.4e}")

# How to interpret:
# - p_val < 0.05: Statistically significant difference between median and benchmark.
# - Hodges-Lehmann shift gives the robust non-parametric effect magnitude.
""", language="python")

    # STEP 3: IMPLEMENT ALTERNATIVE TEST IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Test (Exact Sign Test) in Python", expanded=False):
        st.code("""# Use when difference distribution is severely asymmetric
pos_count = int(np.sum(nonzero_diffs > 0))

# Exact Binomial Sign Test
sign_res = stats.binomtest(pos_count, n=n_eff, p=0.5, alternative="two-sided")
sign_p = sign_res.pvalue

print(f"Positive Differences: {pos_count} / {n_eff}")
print(f"Exact Sign Test p-value: {sign_p:.4e}")

# How to interpret:
# - Tests H0: P(X > mu_0) = 0.5 without assuming distribution symmetry.
# - sign_p < 0.05: Statistically significant median shift from benchmark.
""", language="python")

    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

def run_wilcoxon_analysis(sample_data, benchmark=100.0, alpha=0.05):
    diffs = sample_data - benchmark
    nonzero_diffs = diffs[diffs != 0]
    n_eff = len(nonzero_diffs)
    
    print(f"=== WILCOXON SIGNED-RANK ANALYSIS (N = {len(sample_data)}, Benchmark = {benchmark}) ===\\n")
    
    # 1. Symmetry check
    skew_val = stats.skew(nonzero_diffs)
    symmetric_pass = abs(skew_val) <= 1.5
    print(f"[Assumption] Difference Symmetry: Skewness = {skew_val:.2f} -> {'PASS' if symmetric_pass else 'VIOLATED'}")
    print(f"[Assumption] Effective Sample Size: N_eff = {n_eff} (dropped {len(sample_data)-n_eff} zeros)\\n")
    
    # 2. Decision routing
    if symmetric_pass:
        print("--> Proceeding with Wilcoxon Signed-Rank Test:")
        res = stats.wilcoxon(nonzero_diffs, alternative="two-sided")
        walsh = [(nonzero_diffs[i] + nonzero_diffs[j]) / 2.0 for i in range(n_eff) for j in range(i, n_eff)]
        hl = np.median(walsh)
        print(f"Median = {np.median(sample_data):.2f}, W = {res.statistic:.1f}, p = {res.pvalue:.4e}, HL Shift = {hl:+.2f}")
    else:
        print("--> Symmetry assumption violated! Switching to Exact Sign Test:")
        pos = int(np.sum(nonzero_diffs > 0))
        sign_res = stats.binomtest(pos, n=n_eff, p=0.5, alternative="two-sided")
        print(f"Positive count = {pos}/{n_eff}, Sign Test p-value = {sign_res.pvalue:.4e}")

# Example:
# run_wilcoxon_analysis(my_data, benchmark=100.0)
""", language="python")


def render_mann_whitney_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for the Mann-Whitney U Test."""
    st.markdown(r"""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Sample Size, Variance & Shape)", expanded=False):
        st.markdown(r"""
        The Mann-Whitney U Test requires independent samples, at least ordinal data, and **similar distribution shapes** 
        if testing for a difference in group medians.
        """)

        # 1. Generate deterministic sample data
        rng = np.random.default_rng(42)
        vals_a = np.round(rng.laplace(loc=0.0, scale=10.6, size=30) + 100.0, 1)
        vals_b = np.round(rng.laplace(loc=0.0, scale=10.6, size=30) + 112.0, 1)

        example_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(30)],
            "Col_A_GroupA [A2:A31]": vals_a,
            "Col_B_GroupB [B2:B31]": vals_b
        })

        # 2. Dynamic diagnostics (do not hardcode PASS/FAIL)
        n1 = len(vals_a)
        n2 = len(vals_b)
        var_a = float(np.var(vals_a, ddof=1))
        var_b = float(np.var(vals_b, ddof=1))
        var_ratio = var_a / var_b if var_b > 0 else 1.0

        f_stat = var_a / var_b if var_a >= var_b else var_b / var_a
        df1 = n1 - 1 if var_a >= var_b else n2 - 1
        df2 = n2 - 1 if var_a >= var_b else n1 - 1
        f_pval = float(2.0 * min(stats.f.cdf(f_stat, df1, df2), 1.0 - stats.f.cdf(f_stat, df1, df2)))

        skew_a = float(stats.skew(vals_a))
        skew_b = float(stats.skew(vals_b))
        delta_skew = abs(skew_b - skew_a)

        n_pass = "PASS" if (n1 >= 15 and n2 >= 15) else "WARN"
        var_pass = "PASS" if f_pval >= 0.05 else "WARN"
        skew_pass = "PASS" if delta_skew <= 0.8 else "WARN"

        summary_df = pd.DataFrame({
            "Assumption_Metric": [
                "Sample Size Group A (n1)",
                "Sample Size Group B (n2)",
                "Group A Skewness",
                "Group B Skewness",
                "Absolute |ΔSkew| (Shape Discrepancy)",
                "Variance Ratio (s1² / s2²)",
                "F-Test for Equal Variances p-value"
            ],
            "Value": [
                f"{n1}",
                f"{n2}",
                f"{skew_a:.3f}",
                f"{skew_b:.3f}",
                f"{delta_skew:.3f}",
                f"{var_ratio:.2f}",
                f"{f_pval:.4f}"
            ],
            "Rule_Threshold": [
                "n1 >= 15",
                "n2 >= 15",
                "Comparable to Group B",
                "Comparable to Group A",
                "|ΔSkew| <= 0.8 (Similar Shapes)",
                "Near 1.0",
                "p >= 0.05 (Homoscedastic)"
            ],
            "Assumption_Result": [
                n_pass,
                n_pass,
                "INFO",
                "INFO",
                skew_pass,
                "INFO",
                var_pass
            ]
        })

        st.markdown("##### 📋 Example Spreadsheet Data (Side-by-Side: A2:A31 & B2:B31)")
        render_table(example_df, height=240)

        st.markdown("##### 📊 Assumption Diagnostic Summary")
        render_table(summary_df, height=240)

        st.markdown("#### 1A. Testing Sample Size & Variance Homogeneity")
        st.markdown(r"""
        Assuming Group A values in `A2:A31` and Group B values in `B2:B31`:
        """)
        st.code("""
Count Group A (n1):             =COUNT(A2:A31)
Count Group B (n2):             =COUNT(B2:B31)

Two-Sample F-Test for Variances: =F.TEST(A2:A31, B2:B31)
Skewness Group A:               =SKEW(A2:A31)
Skewness Group B:               =SKEW(B2:B31)
Absolute Skewness Difference:   =ABS(SKEW(A2:A31) - SKEW(B2:B31))
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret the Assumption Output:**
        * **$F$-test $p \ge 0.05$ and $|\Delta \text{Skew}| \le 0.8$:** Shapes and variances are comparable. Proceed with the Mann-Whitney U test as a direct test of the **difference between population medians**.
        * **$F$-test $p < 0.05$ or $|\Delta \text{Skew}| > 0.8$:** Shapes differ. The test evaluates general **stochastic dominance** ($P(B > A) \ne 0.5$) rather than equal medians.
        * **Severe Heteroscedasticity ($p < 0.01$):** Proceed to **Step 3** to use Welch's Two-Sample t-Test.
        """)

    # STEP 2: IMPLEMENT MANN-WHITNEY U TEST IN EXCEL
    with st.expander("📌 Step 2: Implement Mann-Whitney U Test in Excel (Combined Ranks & U-Statistic)", expanded=False):
        st.markdown(r"""
        To execute the Mann-Whitney U test in Excel, combine both groups into a stacked table:
        * **Column A:** Group identifier (`"Group A"` or `"Group B"`).
        * **Column B:** Continuous / Ordinal score (rows `2` to `61` for total $N = n_1 + n_2$).
        * **Column C:** Combined average ranks.
        """)

        comb_vals = np.concatenate([vals_a, vals_b])
        comb_groups = ["Group A"] * 30 + ["Group B"] * 30
        comb_ranks = stats.rankdata(comb_vals)
        stacked_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(60)],
            "Col_A_Group [A2:A61]": comb_groups,
            "Col_B_Score [B2:B61]": comb_vals,
            "Col_C_MidRank [C2:C61]": comb_ranks
        })
        st.markdown("##### 📋 Combined Stacked Spreadsheet Data (Rows 2 to 61)")
        render_table(stacked_df, height=240)
        st.code("""
In Column C (Combined Mid-Rank):
=RANK.AVG(B2, $B$2:$B$61, 1)

Summary Calculations:
Sample Size n1 (Group A):        =COUNTIF($A$2:$A$61, "Group A")
Sample Size n2 (Group B):        =COUNTIF($A$2:$A$61, "Group B")
Rank Sum R1 (Group A):           =SUMIF($A$2:$A$61, "Group A", $C$2:$C$61)
Rank Sum R2 (Group B):           =SUMIF($A$2:$A$61, "Group B", $C$2:$C$61)

Mann-Whitney U1 Statistic:       =n1_cell * n2_cell + (n1_cell * (n1_cell + 1)) / 2 - R1_cell
Mann-Whitney U2 Statistic:       =n1_cell * n2_cell + (n2_cell * (n2_cell + 1)) / 2 - R2_cell
Test Statistic (U):              =MIN(U1_cell, U2_cell)

Large-Sample Normal Approximation (for n1, n2 >= 10):
Expected Mean of U (μ_U):        =(n1_cell * n2_cell) / 2
Std Error of U (σ_U):            =SQRT((n1_cell * n2_cell * (n1_cell + n2_cell + 1)) / 12)
Z-Score (with continuity):       =(ABS(U_cell - μ_U_cell) - 0.5) / σ_U_cell
Two-Tailed p-value:              =2 * (1 - NORM.S.DIST(ABS(Z_cell), TRUE))

Effect Size Measures:
Rank-Biserial Correlation (r_rb):=(U2_cell - U1_cell) / (n1_cell * n2_cell)
Common Language Effect Size (A): =U2_cell / (n1_cell * n2_cell)
""", language="excel")

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output:**
        * **$p$-value $< 0.05$:** Statistically significant difference in rank distributions between Group A and Group B. Reject $H_0$.
        * **$p$-value $\ge 0.05$:** Fail to reject $H_0$. Insufficient evidence of a difference in distributions.
        * **Common Language Effect Size ($CLES$ / $A$):** Represents $P(B > A)$. An effect size of $0.50$ means complete equality; $0.70$ indicates a $70\%$ probability that a random subject from Group B scores higher than Group A.
        * **Rank-Biserial Correlation ($r_{rb}$):** Ranges from $-1.0$ (all A > B) to $+1.0$ (all B > A). $0.1$ = small, $0.3$ = medium, $0.5$ = large effect.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TEST IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Test in Excel (When Shapes / Variances Differ: Welch's t-Test)", expanded=False):
        st.markdown(r"""
        If group variances differ significantly ($p_{\text{Levene}} < 0.05$) or sample sizes are large ($N \ge 30$), 
        switch to **Welch's Two-Sample t-Test**. Welch's t-test does not assume equal variances and is robust against non-normality 
        in moderate-to-large samples via the Central Limit Theorem.
        """)
        st.code("""
Two-Tailed Welch's t-Test p-value (Type 3 = Unequal Variances):
=T.TEST(A2:A31, B2:B31, 2, 3)
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret Welch's t-Test Output:**
        * Returns the exact two-tailed $p$-value comparing the two group means without pooling variances.
        * If $p < 0.05$, group means differ significantly.
        * Completely immune to the heteroscedasticity problems that inflate Mann-Whitney U error rates.
        """)


def render_mann_whitney_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python script guide for the Mann-Whitney U Test."""
    st.markdown(r"""
    ### 🐍 Python Code Implementation Workflow
    This script is divided into 3 modular sections: **Evaluate Assumptions** ➔ **Execute Proposed Test** ➔ **Execute Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Levene's Test & Distribution Shape)", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

# Given group_a and group_b (1D numpy arrays or pandas Series)
n_a, n_b = len(group_a), len(group_b)

# 1. Variance Homogeneity Check (Levene's Test)
lev_stat, p_levene = stats.levene(group_a, group_b)

# 2. Distribution Shape & Skewness Check
skew_a = stats.skew(group_a)
skew_b = stats.skew(group_b)
delta_skew = abs(skew_b - skew_a)

print(f"Sample Sizes: n_A = {n_a}, n_B = {n_b} (Total N = {n_a + n_b})")
print(f"Levene's Test: Statistic = {lev_stat:.3f}, p-value = {p_levene:.4f}")
print(f"Skewness: Group A = {skew_a:.2f}, Group B = {skew_b:.2f} (|ΔSkew| = {delta_skew:.2f})")

# How to interpret:
# - p_levene >= 0.05 and delta_skew <= 0.8: Similar shapes -> Tests difference in medians.
# - p_levene < 0.05 or delta_skew > 0.8: Differing shapes -> Tests general stochastic dominance (P(B > A) != 0.5).
# - Severe heteroscedasticity (p < 0.01) -> Switch to Brunner-Munzel or Welch's t-test (Step 3).
""", language="python")

    # STEP 2: IMPLEMENT MANN-WHITNEY U TEST IN PYTHON
    with st.expander("📌 Step 2: Implement Mann-Whitney U Test in Python", expanded=False):
        st.code("""# 1. Execute Mann-Whitney U Test
# Note: alternative='two-sided' tests whether distributions are stochastically equal
res = stats.mannwhitneyu(group_b, group_a, alternative="two-sided")
u_b = res.statistic
p_val = res.pvalue
u_a = n_a * n_b - u_b
u_stat = min(u_a, u_b)

# 2. Compute Effect Sizes
r_rb = (u_b - u_a) / (n_a * n_b)  # Rank-Biserial Correlation (-1 to +1)
cles = u_b / (n_a * n_b)           # Common Language Effect Size: P(B > A)

# 3. Compute Hodges-Lehmann Median Difference
pairwise_diffs = np.subtract.outer(group_b, group_a).flatten()
hl_diff = np.median(pairwise_diffs)

print(f"Group A Median: {np.median(group_a):.2f} | Group B Median: {np.median(group_b):.2f}")
print(f"Hodges-Lehmann Location Shift (B - A): {hl_diff:+.2f}")
print(f"Mann-Whitney U: {u_stat:.1f}, p-value: {p_val:.4e}")
print(f"Rank-Biserial r: {r_rb:+.3f} | CLES (P(B > A)): {cles:.3f}")

# How to interpret:
# - p_val < 0.05: Statistically significant difference in rank distributions.
# - Hodges-Lehmann shift gives the median of all pairwise differences.
# - CLES > 0.50 means observations in Group B systematically exceed Group A.
""", language="python")

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Tests in Python (Brunner-Munzel & Welch's t-test)", expanded=False):
        st.code("""# Alternative 1: Brunner-Munzel Test (Generalized Wilcoxon for Heteroscedastic Data)
bm_res = stats.brunnermunzel(group_b, group_a, alternative="two-sided")
print(f"Brunner-Munzel Statistic: {bm_res.statistic:.3f}, p-value: {bm_res.pvalue:.4e}")

# Alternative 2: Welch's Two-Sample t-Test (Parametric, robust against unequal variance)
t_stat, p_welch = stats.ttest_ind(group_b, group_a, equal_var=False)
print(f"Welch's t-test: t = {t_stat:.3f}, p-value: {p_welch:.4e}")

# How to interpret:
# - Use Brunner-Munzel when shapes/variances differ but you still require a non-parametric rank test.
# - Use Welch's t-test when sample size is large enough (N >= 30 per group) for CLT robustness.
""", language="python")

    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

def run_mann_whitney_analysis(group_a, group_b, alpha=0.05):
    n_a, n_b = len(group_a), len(group_b)
    print(f"=== MANN-WHITNEY U ANALYSIS (n_A = {n_a}, n_B = {n_b}) ===\\n")
    
    # 1. Assumption diagnostics
    lev_stat, p_lev = stats.levene(group_a, group_b)
    skew_a, skew_b = stats.skew(group_a), stats.skew(group_b)
    delta_skew = abs(skew_b - skew_a)
    
    equal_var = p_lev >= alpha
    equal_shape = delta_skew <= 0.8
    
    print(f"[Assumption] Variance Homogeneity: Levene p = {p_lev:.4f} -> {'PASS' if equal_var else 'CAUTION'}")
    print(f"[Assumption] Shape Discrepancy: |ΔSkew| = {delta_skew:.2f} -> {'PASS' if equal_shape else 'CAUTION'}\\n")
    
    # 2. Primary Test: Mann-Whitney U
    res = stats.mannwhitneyu(group_b, group_a, alternative="two-sided")
    u_b = res.statistic
    u_a = n_a * n_b - u_b
    u_min = min(u_a, u_b)
    r_rb = (u_b - u_a) / (n_a * n_b)
    cles = u_b / (n_a * n_b)
    hl = np.median(np.subtract.outer(group_b, group_a))
    
    print("--> Mann-Whitney U Test Results:")
    print(f"U = {u_min:.1f}, p = {res.pvalue:.4e}")
    print(f"Medians: A = {np.median(group_a):.2f}, B = {np.median(group_b):.2f}")
    print(f"Hodges-Lehmann Shift = {hl:+.2f}, Rank-Biserial r = {r_rb:+.3f}, CLES = {cles:.3f}\\n")
    
    # 3. Decision routing advice
    if not equal_var or not equal_shape:
        print("--> Notice: Differing shapes/variances detected. Comparing with robust alternatives:")
        bm = stats.brunnermunzel(group_b, group_a, alternative="two-sided")
        welch = stats.ttest_ind(group_b, group_a, equal_var=False)
        print(f"Brunner-Munzel p-value = {bm.pvalue:.4e}")
        print(f"Welch's t-test p-value   = {welch.pvalue:.4e}")

# Example Run:
# run_mann_whitney_analysis(ctrl_data, treat_data)
""", language="python")


def render_welch_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for Welch's Two-Sample t-test."""
    st.markdown(r"""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Normality, Outliers & Variances)", expanded=False):
        st.markdown(r"""
        Welch's t-test requires independent samples and continuous data. 
        It **does not assume equal variances**, but assumes approximate normality within each group (unless $N \ge 30$ per group where the Central Limit Theorem applies).
        """)

        # 1. Generate deterministic sample data
        rng = np.random.default_rng(42)
        vals_a = np.round(rng.normal(loc=100.0, scale=10.0, size=35), 1)
        vals_b = np.round(rng.normal(loc=108.0, scale=18.0, size=35), 1)

        # Outlier detection using Tukey's fences (IQR) for both groups
        q25_a = float(np.percentile(vals_a, 25))
        q75_a = float(np.percentile(vals_a, 75))
        iqr_a = q75_a - q25_a
        lower_a = q25_a - 1.5 * iqr_a
        upper_a = q75_a + 1.5 * iqr_a
        flags_a = np.where((vals_a < lower_a) | (vals_a > upper_a), "OUTLIER", "OK")

        q25_b = float(np.percentile(vals_b, 25))
        q75_b = float(np.percentile(vals_b, 75))
        iqr_b = q75_b - q25_b
        lower_b = q25_b - 1.5 * iqr_b
        upper_b = q75_b + 1.5 * iqr_b
        flags_b = np.where((vals_b < lower_b) | (vals_b > upper_b), "OUTLIER", "OK")

        example_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(35)],
            "Col_A_GroupA [A2:A36]": vals_a,
            "Col_A_Outlier_Check": flags_a,
            "Col_B_GroupB [B2:B36]": vals_b,
            "Col_B_Outlier_Check": flags_b
        })

        # 2. Dynamic diagnostics (do not hardcode PASS/FAIL)
        n1 = len(vals_a)
        n2 = len(vals_b)
        outliers_a = int(np.sum(flags_a == "OUTLIER"))
        outliers_b = int(np.sum(flags_b == "OUTLIER"))
        skew_a = float(stats.skew(vals_a))
        skew_b = float(stats.skew(vals_b))
        kurt_a = float(stats.kurtosis(vals_a))
        kurt_b = float(stats.kurtosis(vals_b))
        var_a = float(np.var(vals_a, ddof=1))
        var_b = float(np.var(vals_b, ddof=1))
        var_ratio = var_b / var_a if var_a > 0 else 1.0

        n_pass = "PASS" if (n1 >= 30 and n2 >= 30) else "WARN"
        outlier_pass = "PASS" if (outliers_a == 0 and outliers_b == 0) else "WARN"
        skew_pass = "PASS" if (abs(skew_a) <= 0.8 and abs(skew_b) <= 0.8) else "WARN"

        summary_df = pd.DataFrame({
            "Assumption_Metric": [
                "Group A Sample Size (n1)",
                "Group B Sample Size (n2)",
                "Group A Outliers (>1.5x IQR)",
                "Group B Outliers (>1.5x IQR)",
                "Group A Skewness",
                "Group B Skewness",
                "Group A Excess Kurtosis",
                "Group B Excess Kurtosis",
                "Variance Ratio (s2² / s1²)"
            ],
            "Value": [
                f"{n1}",
                f"{n2}",
                f"{outliers_a}",
                f"{outliers_b}",
                f"{skew_a:.3f}",
                f"{skew_b:.3f}",
                f"{kurt_a:.3f}",
                f"{kurt_b:.3f}",
                f"{var_ratio:.2f}"
            ],
            "Rule_Threshold": [
                "n1 >= 30 (CLT robust)",
                "n2 >= 30 (CLT robust)",
                "0 Extreme Outliers",
                "0 Extreme Outliers",
                "Between -0.8 and +0.8",
                "Between -0.8 and +0.8",
                "Between -1.0 and +1.0",
                "Between -1.0 and +1.0",
                "Unequal variance expected"
            ],
            "Assumption_Result": [
                n_pass,
                n_pass,
                outlier_pass,
                outlier_pass,
                skew_pass,
                skew_pass,
                "PASS",
                "PASS",
                "PASS (Welch robust)"
            ]
        })

        st.markdown("##### 📋 Example Spreadsheet Data (Group A: A2:A36, Group B: B2:B36)")
        render_table(example_df, height=240)

        st.markdown("##### 📊 Assumption Diagnostic Summary")
        render_table(summary_df, height=240)

        st.markdown("#### 1A. Testing Sample Size, Skewness & Outliers")
        st.markdown(r"""
        Assuming Group A values in `A2:A36` and Group B values in `B2:B36`:
        """)
        st.code("""
Sample Counts:
Count Group A (n1):             =COUNT(A2:A36)
Count Group B (n2):             =COUNT(B2:B36)

Distribution Shape & Skewness:
Skewness Group A:               =SKEW(A2:A36)
Skewness Group B:               =SKEW(B2:B36)
Kurtosis Group A:               =KURT(A2:A36)
Kurtosis Group B:               =KURT(B2:B36)

Two-Sample F-Test for Variances (Informational):
Variance Homogeneity:           =F.TEST(A2:A36, B2:B36)

Outlier Check (Tukey's IQR Method for Group A):
Q1 (25th percentile):           =QUARTILE.INC(A2:A36, 1)
Q3 (75th percentile):           =QUARTILE.INC(A2:A36, 3)
IQR:                            =Q3_cell - Q1_cell
Lower Extreme Fence:            =Q1_cell - 3 * IQR_cell
Upper Extreme Fence:            =Q3_cell + 3 * IQR_cell
Count Extreme Outliers:         =COUNTIF(A2:A36, "<" & Lower_Fence) + COUNTIF(A2:A36, ">" & Upper_Fence)
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret the Assumption Output:**
        * **Normality:** If $|\text{SKEW}| \le 0.8$, the distribution is approximately normal. If $|\text{SKEW}| > 0.8$ but both group sample sizes are $N \ge 30$, **Central Limit Theorem applies** and Welch's t-test remains robust.
        * **Outliers:** If any extreme outliers exist ($> 3 \times \text{IQR}$), they distort group means and inflate standard errors. Switch to **Step 3 (Mann-Whitney U Test)**.
        * **Variance:** Even if `=F.TEST()` shows $p < 0.05$ (heteroscedasticity), Welch's t-test specifically does not assume equal variances and is completely valid.
        """)

    # STEP 2: IMPLEMENT WELCH'S T-TEST IN EXCEL
    with st.expander("📌 Step 2: Implement Welch's Two-Sample t-Test in Excel (Native Formula & Step-by-Step)", expanded=False):
        st.markdown(r"""
        #### 2A. Single Native Formula (Direct p-value)
        Excel provides a built-in function to compute the exact two-tailed Welch's $t$-test $p$-value directly:
        """)
        st.code("""
Two-Tailed Welch's t-Test p-value:
=T.TEST(A2:A36, B2:B36, 2, 3)
""", language="excel")
        st.caption("💡 **Formula arguments:** `array1`, `array2`, `tails=2` (two-tailed), `type=3` (two-sample assuming unequal variances / Welch's t-test).")

        st.markdown("#### 2B. Complete Step-by-Step Breakdown (t-statistic, df, CI & Cohen's d)")
        st.code("""
Sample Means & Variances:
Mean Group A (x̄1):              =AVERAGE(A2:A36)
Mean Group B (x̄2):              =AVERAGE(B2:B36)
Variance Group A (s1²):         =VAR.S(A2:A36)
Variance Group B (s2²):         =VAR.S(B2:B36)
Mean Difference (x̄2 - x̄1):      =MeanB_cell - MeanA_cell

Welch Standard Error of Difference:
SE_diff:                        =SQRT(VarA_cell / n1_cell + VarB_cell / n2_cell)

Welch-Satterthwaite Degrees of Freedom (df):
=((VarA_cell/n1_cell + VarB_cell/n2_cell)^2) / (((VarA_cell/n1_cell)^2)/(n1_cell - 1) + ((VarB_cell/n2_cell)^2)/(n2_cell - 1))

Welch t-Statistic:
t-Stat:                         =(MeanB_cell - MeanA_cell) / SE_diff_cell

Two-Tailed p-value:
p-value:                        =T.DIST.2T(ABS(t_cell), df_cell)

95% Confidence Interval for Difference (μ2 - μ1):
Margin of Error:                =T.INV.2T(0.05, df_cell) * SE_diff_cell
95% CI Lower Bound:             =Diff_cell - Margin_cell
95% CI Upper Bound:             =Diff_cell + Margin_cell

Standardized Effect Size (Cohen's d):
=Diff_cell / SQRT((VarA_cell + VarB_cell) / 2)
""", language="excel")

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output:**
        * **$p$-value $< 0.05$ (Statistically Significant):** There is strong evidence of a genuine difference between the population means of the two groups. Reject the null hypothesis $H_0: \mu_1 = \mu_2$.
        * **$p$-value $\ge 0.05$ (Not Significant):** Insufficient evidence to conclude that the true group means differ. The observed gap could plausibly have arisen through random sampling variation alone.
        * **95% Confidence Interval:** If zero (`0.0`) falls **outside** the interval `[CI_Lower, CI_Upper]`, the difference is statistically significant at $\alpha = 0.05$.
        * **Cohen's $d$:** Measures standardized magnitude: $0.2$ = small, $0.5$ = medium, $0.8$ = large effect.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TEST IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Test in Excel (When Normality Fails: Mann-Whitney U Test)", expanded=False):
        st.markdown(r"""
        If either group violates normality with small sample size ($N < 30$) or contains extreme outliers, 
        switch to the non-parametric **Mann-Whitney U Test**.
        """)
        st.code("""
Combine observations into a single table with Group indicator (Col A = Group, Col B = Value):
Col C (Combined Rank):          =RANK.AVG(B2, $B$2:$B$71, 1)

Rank Sum Group A (R1):          =SUMIF($A$2:$A$71, "Group A", $C$2:$C$71)
Rank Sum Group B (R2):          =SUMIF($A$2:$A$71, "Group B", $C$2:$C$71)

U1 Statistic:                   =n1 * n2 + (n1 * (n1 + 1)) / 2 - R1
U2 Statistic:                   =n1 * n2 + (n2 * (n2 + 1)) / 2 - R2
Mann-Whitney U:                 =MIN(U1_cell, U2_cell)

Normal Approximation p-value:
μ_U = (n1 * n2) / 2
σ_U = SQRT((n1 * n2 * (n1 + n2 + 1)) / 12)
Z   = (ABS(U_cell - μ_U) - 0.5) / σ_U
Two-Tailed p-value:             =2 * (1 - NORM.S.DIST(ABS(Z_cell), TRUE))
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret the Alternative Output:**
        * The Mann-Whitney U test evaluates **ranks** rather than arithmetic means, making it immune to skewness and extreme outliers.
        * If $p < 0.05$, the distributions of the two groups differ significantly.
        """)


def render_welch_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python script guide for Welch's Two-Sample t-test."""
    st.markdown(r"""
    ### 🐍 Python Code Implementation Workflow
    This script is divided into 3 modular sections: **Evaluate Assumptions** ➔ **Execute Proposed Test** ➔ **Execute Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Normality, Outliers & Levene's Test)", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

# Given group_a and group_b (1D numpy arrays or pandas Series)
n_a, n_b = len(group_a), len(group_b)

# 1. Normality Check via Shapiro-Wilk Test & Skewness
w_a, p_shap_a = stats.shapiro(group_a)
w_b, p_shap_b = stats.shapiro(group_b)
skew_a, skew_b = stats.skew(group_a), stats.skew(group_b)

# 2. Outlier Check via Tukey's IQR Method
def get_extreme_outliers(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    return data[(data < q25 - 3 * iqr) | (data > q75 + 3 * iqr)]

ext_a = get_extreme_outliers(group_a)
ext_b = get_extreme_outliers(group_b)

# 3. Informational Variance Homogeneity Check
lev_stat, p_levene = stats.levene(group_a, group_b)

print(f"Sample Sizes: n_A = {n_a}, n_B = {n_b}")
print(f"Shapiro-Wilk Normality: Group A p = {p_shap_a:.4f}, Group B p = {p_shap_b:.4f}")
print(f"Skewness: Group A = {skew_a:.2f}, Group B = {skew_b:.2f}")
print(f"Extreme Outliers (>3xIQR): Group A = {len(ext_a)}, Group B = {len(ext_b)}")
print(f"Levene's Equal Variance: p = {p_levene:.4f} (Welch robust to unequal variances)")

# Decision Logic:
# - Both Shapiro p >= 0.05 OR (n_a >= 30 and n_b >= 30 and |skew| <= 1.2): Normality satisfied -> Step 2.
# - Otherwise or if extreme outliers exist: Switch to Mann-Whitney U test -> Step 3.
""", language="python")

    # STEP 2: IMPLEMENT WELCH'S T-TEST IN PYTHON
    with st.expander("📌 Step 2: Implement Welch's Two-Sample t-Test in Python", expanded=False):
        st.code("""# 1. Execute Welch's t-test (equal_var=False)
res = stats.ttest_ind(group_b, group_a, equal_var=False)
t_stat = res.statistic
p_val = res.pvalue

# 2. Compute Summary Statistics & Welch-Satterthwaite Degrees of Freedom
m_a, m_b = np.mean(group_a), np.mean(group_b)
v_a, v_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
mean_diff = m_b - m_a
se_diff = np.sqrt(v_a / n_a + v_b / n_b)

df = ((v_a/n_a + v_b/n_b)**2) / (((v_a/n_a)**2)/(n_a - 1) + ((v_b/n_b)**2)/(n_b - 1))

# 3. Compute 95% Confidence Interval for Mean Difference (B - A)
t_crit = stats.t.ppf(0.975, df=df)
ci_lower = mean_diff - t_crit * se_diff
ci_upper = mean_diff + t_crit * se_diff

# 4. Compute Standardized Effect Sizes (Cohen's d and Hedges' g)
s_avg = np.sqrt((v_a + v_b) / 2.0)
cohens_d = mean_diff / s_avg
hedges_g = cohens_d * (1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0))

print(f"Group A Mean: {m_a:.2f} (SD: {np.sqrt(v_a):.2f})")
print(f"Group B Mean: {m_b:.2f} (SD: {np.sqrt(v_b):.2f})")
print(f"Mean Difference (B - A): {mean_diff:+.2f} (SE: {se_diff:.2f})")
print(f"Welch t({df:.1f}) = {t_stat:.3f}, p-value: {p_val:.4e}")
print(f"95% CI for Difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Cohen's d: {cohens_d:.2f} | Hedges' g: {hedges_g:.2f}")

# How to interpret:
# - p_val < 0.05: Statistically significant difference between population means.
# - CI excludes 0.0: Strong evidence that Group B differs from Group A.
""", language="python")

    # STEP 3: IMPLEMENT ALTERNATIVE TEST IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Test (Mann-Whitney U Test) in Python", expanded=False):
        st.code("""# Use when normality is violated in small samples or extreme outliers distort means
u_res = stats.mannwhitneyu(group_b, group_a, alternative="two-sided")
u_stat = u_res.statistic
p_mann = u_res.pvalue
cles = u_stat / (n_a * n_b)

print(f"Mann-Whitney U: {u_stat:.1f}, p-value: {p_mann:.4e}")
print(f"Common Language Effect Size (P(B > A)): {cles:.3f}")

# How to interpret:
# - Tests whether the rank distribution of Group B systematically exceeds Group A.
# - Robust against extreme outliers and severe skewness.
""", language="python")

    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

def run_welch_analysis(group_a, group_b, alpha=0.05):
    n_a, n_b = len(group_a), len(group_b)
    print(f"=== WELCH'S TWO-SAMPLE t-TEST ANALYSIS (n_A = {n_a}, n_B = {n_b}) ===\\n")
    
    # 1. Assumption Diagnostics
    _, p_shap_a = stats.shapiro(group_a)
    _, p_shap_b = stats.shapiro(group_b)
    skew_a, skew_b = stats.skew(group_a), stats.skew(group_b)
    
    def count_ext_outliers(d):
        q25, q75 = np.percentile(d, [25, 75])
        iqr = q75 - q25
        return len(d[(d < q25 - 3 * iqr) | (d > q75 + 3 * iqr)])
    
    total_ext = count_ext_outliers(group_a) + count_ext_outliers(group_b)
    
    normal_pass = (p_shap_a >= alpha and p_shap_b >= alpha) or (n_a >= 30 and n_b >= 30 and abs(skew_a) <= 1.2 and abs(skew_b) <= 1.2)
    outlier_pass = total_ext == 0
    
    print(f"[Assumption] Normality: Shapiro A p={p_shap_a:.3f}, B p={p_shap_b:.3f} -> {'PASS' if normal_pass else 'VIOLATED'}")
    print(f"[Assumption] Outliers: {total_ext} extreme outliers -> {'PASS' if outlier_pass else 'VIOLATED'}\\n")
    
    # 2. Decision Routing
    if normal_pass and outlier_pass:
        print("--> Proceeding with Welch's Two-Sample t-Test:")
        res = stats.ttest_ind(group_b, group_a, equal_var=False)
        m_a, m_b = np.mean(group_a), np.mean(group_b)
        v_a, v_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
        mean_diff = m_b - m_a
        se = np.sqrt(v_a / n_a + v_b / n_b)
        df = ((v_a/n_a + v_b/n_b)**2) / (((v_a/n_a)**2)/(n_a - 1) + ((v_b/n_b)**2)/(n_b - 1))
        t_crit = stats.t.ppf(1 - alpha/2, df=df)
        ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
        d = mean_diff / np.sqrt((v_a + v_b) / 2.0)
        
        print(f"Means: A = {m_a:.2f}, B = {m_b:.2f} | Diff = {mean_diff:+.2f} (SE = {se:.2f})")
        print(f"t({df:.1f}) = {res.statistic:.3f}, p = {res.pvalue:.4e}")
        print(f"95% CI for Difference: [{ci[0]:.2f}, {ci[1]:.2f}] | Cohen's d: {d:.2f}")
    else:
        print("--> Assumptions violated! Switching to Mann-Whitney U Test:")
        u_res = stats.mannwhitneyu(group_b, group_a, alternative="two-sided")
        print(f"Mann-Whitney U = {u_res.statistic:.1f}, p-value = {u_res.pvalue:.4e}")
        print(f"Medians: A = {np.median(group_a):.2f}, B = {np.median(group_b):.2f}")

# Example Run:
# run_welch_analysis(ctrl_group, treat_group)
""", language="python")


def render_students_t_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for Student's Two-Sample t-test."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Normality, Outliers & Equal Variances)", expanded=False):
        st.markdown(r"""
        Student's independent two-sample t-test requires independent samples and continuous data. 
        Unlike Welch's t-test, it **strictly assumes equal population variances** ($\sigma_1^2 = \sigma_2^2$, homoscedasticity) 
        in addition to approximate normality within each group (unless $N \ge 30$ per group where the Central Limit Theorem applies).
        """)

        # 1. Generate deterministic sample data (equal variance design)
        rng = np.random.default_rng(42)
        vals_a = np.round(rng.normal(loc=100.0, scale=12.0, size=35), 1)
        vals_b = np.round(rng.normal(loc=108.0, scale=12.0, size=35), 1)

        # Outlier detection using Tukey's fences (IQR) for both groups
        q25_a = float(np.percentile(vals_a, 25))
        q75_a = float(np.percentile(vals_a, 75))
        iqr_a = q75_a - q25_a
        lower_a = q25_a - 1.5 * iqr_a
        upper_a = q75_a + 1.5 * iqr_a
        flags_a = np.where((vals_a < lower_a) | (vals_a > upper_a), "OUTLIER", "OK")

        q25_b = float(np.percentile(vals_b, 25))
        q75_b = float(np.percentile(vals_b, 75))
        iqr_b = q75_b - q25_b
        lower_b = q25_b - 1.5 * iqr_b
        upper_b = q75_b + 1.5 * iqr_b
        flags_b = np.where((vals_b < lower_b) | (vals_b > upper_b), "OUTLIER", "OK")

        example_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(35)],
            "Col_A_GroupA [A2:A36]": vals_a,
            "Col_A_Outlier_Check": flags_a,
            "Col_B_GroupB [B2:B36]": vals_b,
            "Col_B_Outlier_Check": flags_b
        })

        # 2. Dynamic diagnostics (do not hardcode PASS/FAIL)
        n1 = len(vals_a)
        n2 = len(vals_b)
        outliers_a = int(np.sum(flags_a == "OUTLIER"))
        outliers_b = int(np.sum(flags_b == "OUTLIER"))
        skew_a = float(stats.skew(vals_a))
        skew_b = float(stats.skew(vals_b))
        kurt_a = float(stats.kurtosis(vals_a))
        kurt_b = float(stats.kurtosis(vals_b))
        var_a = float(np.var(vals_a, ddof=1))
        var_b = float(np.var(vals_b, ddof=1))
        var_ratio = var_b / var_a if var_a > 0 else 1.0

        # Two-sample F-test p-value (equal variance check)
        f_stat = var_a / var_b if var_b > 0 else 1.0
        f_cdf = float(stats.f.cdf(f_stat, n1 - 1, n2 - 1))
        f_pval = float(2.0 * min(f_cdf, 1.0 - f_cdf))

        n_pass = "PASS" if (n1 >= 30 and n2 >= 30) else "WARN"
        outlier_pass = "PASS" if (outliers_a == 0 and outliers_b == 0) else "WARN"
        skew_pass = "PASS" if (abs(skew_a) <= 0.8 and abs(skew_b) <= 0.8) else "WARN"
        var_pass = "PASS" if f_pval >= 0.05 else "FAIL"

        summary_df = pd.DataFrame({
            "Assumption_Metric": [
                "Group A Sample Size (n1)",
                "Group B Sample Size (n2)",
                "Group A Outliers (>1.5x IQR)",
                "Group B Outliers (>1.5x IQR)",
                "Group A Skewness",
                "Group B Skewness",
                "Group A Excess Kurtosis",
                "Group B Excess Kurtosis",
                "Variance Ratio (s2² / s1²)",
                "F-Test for Equal Variances p-value"
            ],
            "Value": [
                f"{n1}",
                f"{n2}",
                f"{outliers_a}",
                f"{outliers_b}",
                f"{skew_a:.3f}",
                f"{skew_b:.3f}",
                f"{kurt_a:.3f}",
                f"{kurt_b:.3f}",
                f"{var_ratio:.2f}",
                f"{f_pval:.4f}"
            ],
            "Rule_Threshold": [
                "n1 >= 30 (CLT robust)",
                "n2 >= 30 (CLT robust)",
                "0 Extreme Outliers",
                "0 Extreme Outliers",
                "Between -0.8 and +0.8",
                "Between -0.8 and +0.8",
                "Between -1.0 and +1.0",
                "Between -1.0 and +1.0",
                "Near 1.0 (Homoscedastic)",
                "p >= 0.05 (Homoscedastic)"
            ],
            "Assumption_Result": [
                n_pass,
                n_pass,
                outlier_pass,
                outlier_pass,
                skew_pass,
                skew_pass,
                "PASS",
                "PASS",
                "INFO",
                var_pass
            ]
        })

        st.markdown("##### 📋 Example Spreadsheet Data (Group A: A2:A36, Group B: B2:B36)")
        render_table(example_df, height=240)

        st.markdown("##### 📊 Assumption Diagnostic Summary")
        render_table(summary_df, height=240)

        st.markdown("#### 1A. Testing Sample Size, Skewness & Equal Variances")
        st.markdown(r"""
        Assuming Group A values in `A2:A36` and Group B values in `B2:B36`:
        """)
        st.code("""
Sample Counts:
Count Group A (n1):             =COUNT(A2:A36)
Count Group B (n2):             =COUNT(B2:B36)

Distribution Shape & Normality:
Skewness Group A:               =SKEW(A2:A36)
Skewness Group B:               =SKEW(B2:B36)
Kurtosis Group A:               =KURT(A2:A36)
Kurtosis Group B:               =KURT(B2:B36)

Homogeneity of Variance (Critical Assumption):
Two-Sample F-Test:              =F.TEST(A2:A36, B2:B36)

Outlier Detection (Tukey's IQR Method for Group A):
Q1 (25th percentile):           =QUARTILE.INC(A2:A36, 1)
Q3 (75th percentile):           =QUARTILE.INC(A2:A36, 3)
IQR:                            =Q3_cell - Q1_cell
Lower Extreme Fence:            =Q1_cell - 3 * IQR_cell
Upper Extreme Fence:            =Q3_cell + 3 * IQR_cell
Count Extreme Outliers:         =COUNTIF(A2:A36, "<" & Lower_Fence) + COUNTIF(A2:A36, ">" & Upper_Fence)
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret the Assumption Output:**
        * **Homogeneity of Variance ($F$-test $p \ge 0.05$):** Variances are statistically comparable ($\sigma_1^2 = \sigma_2^2$). Proceed with **Step 2 (Student's t-Test with pooled variance)**.
        * **Heteroscedasticity ($F$-test $p < 0.05$):** The assumption of equal variance is **violated**. Proceed to **Step 3 (Alternative A: Welch's Two-Sample t-Test)**.
        * **Normality & Outliers:** If $|\text{SKEW}| \le 0.8$, normality is satisfied. If $N \ge 30$ per group, the Central Limit Theorem provides robustness. If extreme outliers ($> 3 \times \text{IQR}$) or severe skewness exist with small $N$, proceed to **Step 3 (Alternative B: Mann-Whitney U Test)**.
        """)

    # STEP 2: IMPLEMENT STUDENT'S T-TEST IN EXCEL
    with st.expander("📌 Step 2: Implement Student's Two-Sample t-Test in Excel (Native Formula, Pooled Breakdown & Toolpak)", expanded=False):
        st.markdown(r"""
        #### 2A. Single Native Formula (Direct p-value)
        Excel provides a built-in function to compute the exact two-tailed Student's $t$-test $p$-value using a pooled variance estimate:
        """)
        st.code("""
Two-Tailed Student's t-Test p-value (Equal Variance):
=T.TEST(A2:A36, B2:B36, 2, 2)
""", language="excel")
        st.caption("💡 **Formula arguments:** `array1`, `array2`, `tails=2` (two-tailed), `type=2` (two-sample assuming equal variances / Student's t-test with pooled variance). Contrast with `type=3` which runs Welch's test.")

        st.markdown("#### 2B. Complete Step-by-Step Breakdown (Pooled Variance, SE, df, t-stat, CI & Cohen's d)")
        st.code("""
Sample Sizes, Means & Variances:
Count Group A (n1):             =COUNT(A2:A36)
Count Group B (n2):             =COUNT(B2:B36)
Mean Group A (x̄1):              =AVERAGE(A2:A36)
Mean Group B (x̄2):              =AVERAGE(B2:B36)
Variance Group A (s1²):         =VAR.S(A2:A36)
Variance Group B (s2²):         =VAR.S(B2:B36)
Mean Difference (x̄2 - x̄1):      =MeanB_cell - MeanA_cell

Degrees of Freedom (Exact Integer):
df:                             =n1_cell + n2_cell - 2

Pooled Sample Variance (sp²):
sp²:                            =((n1_cell - 1) * VarA_cell + (n2_cell - 1) * VarB_cell) / df_cell

Pooled Standard Deviation (sp):
sp:                             =SQRT(sp2_cell)

Pooled Standard Error of Difference (SE_diff):
SE_diff:                        =sp_cell * SQRT(1 / n1_cell + 1 / n2_cell)

Student's t-Statistic:
t-Stat:                         =(MeanB_cell - MeanA_cell) / SE_diff_cell

Two-Tailed p-value:
p-value:                        =T.DIST.2T(ABS(t_cell), df_cell)

95% Confidence Interval for Difference (μ2 - μ1):
Margin of Error:                =T.INV.2T(0.05, df_cell) * SE_diff_cell
95% CI Lower Bound:             =Diff_cell - Margin_cell
95% CI Upper Bound:             =Diff_cell + Margin_cell

Standardized Effect Size (Pooled Cohen's d):
Cohen's d:                      =Diff_cell / sp_cell
Hedges' g (bias-corrected):     =d_cell * (1 - 3 / (4 * (n1_cell + n2_cell) - 9))
""", language="excel")

        st.markdown("#### 2C. Excel Analysis ToolPak Automated Workflow")
        st.markdown(r"""
        1. Click **Data** tab ➔ **Data Analysis** (top right ribbon).
        2. Select **t-Test: Two-Sample Assuming Equal Variances** and click **OK**.
        3. **Variable 1 Range:** `$A$1:$A$36` (Group A column).
        4. **Variable 2 Range:** `$B$1:$B$36` (Group B column).
        5. **Hypothesized Mean Difference:** Enter `0` (null hypothesis: $\mu_1 = \mu_2$).
        6. Check **Labels** (if header row 1 was selected).
        7. **Alpha:** `0.05`.
        8. **Output Range:** Select `$D$1` and click **OK**.
        
        *The ToolPak automatically outputs: Mean, Variance, Observations, Pooled Variance, Hypothesized Mean Difference, df ($n_1 + n_2 - 2$), t Stat, P(T<=t) two-tail, and t Critical two-tail.*
        """)

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output:**
        * **$p$-value $< 0.05$ (Statistically Significant):** There is strong evidence of a genuine difference between the population means of the two groups. Reject the null hypothesis $H_0: \mu_1 = \mu_2$.
        * **$p$-value $\ge 0.05$ (Not Significant):** Insufficient evidence to conclude that the true group means differ. The observed difference can be attributed to random sampling variation.
        * **95% Confidence Interval:** If zero (`0.0`) is **not** contained within `[CI_Lower, CI_Upper]`, the difference is statistically significant at $\alpha = 0.05$.
        * **Pooled Cohen's $d$:** Measures standardized practical effect magnitude: $0.2$ = small, $0.5$ = medium, $0.8$ = large effect.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Tests in Excel (Welch's t-Test or Mann-Whitney U)", expanded=False):
        st.markdown(r"""
        #### 3A. Alternative When Variances are Unequal: Welch's Two-Sample t-Test
        If the $F$-test shows heteroscedasticity ($p < 0.05$, variances are significantly unequal), do **not** use pooled variance. 
        Instead, execute **Welch's t-test** using `type=3`:
        """)
        st.code("""
Two-Tailed Welch's t-Test (Unequal Variances):
=T.TEST(A2:A36, B2:B36, 2, 3)

Welch Standard Error:           =SQRT(VarA_cell / n1_cell + VarB_cell / n2_cell)
Welch-Satterthwaite df:         =((VarA/n1 + VarB/n2)^2) / (((VarA/n1)^2)/(n1 - 1) + ((VarB/n2)^2)/(n2 - 1))
Welch t-Stat:                   =(MeanB - MeanA) / SE_welch
Welch p-value:                  =T.DIST.2T(ABS(t_welch), df_welch)
""", language="excel")
        st.markdown(r"""
        *ToolPak alternative:* **Data Analysis** ➔ **t-Test: Two-Sample Assuming Unequal Variances**.
        """)

        st.markdown(r"""
        #### 3B. Alternative When Normality Fails or Extreme Outliers Exist: Mann-Whitney U Test
        If either group violates normality with small sample size ($N < 30$) or contains extreme outliers that distort the mean, 
        switch to the non-parametric **Mann-Whitney U Test** (Wilcoxon Rank-Sum):
        """)
        st.code("""
Combine observations into a single stacked table (Col A = Group, Col B = Value):
Col C (Combined Rank):          =RANK.AVG(B2, $B$2:$B$71, 1)

Rank Sum Group A (R1):          =SUMIF($A$2:$A$71, "Group A", $C$2:$C$71)
Rank Sum Group B (R2):          =SUMIF($A$2:$A$71, "Group B", $C$2:$C$71)

U1 Statistic:                   =n1 * n2 + (n1 * (n1 + 1)) / 2 - R1
U2 Statistic:                   =n1 * n2 + (n2 * (n2 + 1)) / 2 - R2
Mann-Whitney U:                 =MIN(U1_cell, U2_cell)

Normal Approximation p-value (for n1, n2 >= 10):
μ_U = (n1 * n2) / 2
σ_U = SQRT((n1 * n2 * (n1 + n2 + 1)) / 12)
Z   = (ABS(U_cell - μ_U) - 0.5) / σ_U
Two-Tailed p-value:             =2 * (1 - NORM.S.DIST(ABS(Z_cell), TRUE))
""", language="excel")
        st.markdown(r"""
        **🔍 How to Interpret the Alternative Output:**
        * **Welch's t-test ($p < 0.05$):** Group means differ significantly without requiring equal variances.
        * **Mann-Whitney U test ($p < 0.05$):** The rank distributions of the two groups differ significantly, completely protected against extreme outliers.
        """)


def render_students_t_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python script guide for Student's Two-Sample t-test."""
    st.markdown(r"""
    ### 🐍 Python Code Implementation Workflow
    This script is divided into 3 modular sections: **Evaluate Assumptions** ➔ **Execute Proposed Test** ➔ **Execute Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Normality, Outliers & Levene's Test)", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

# Given group_a and group_b (1D numpy arrays or pandas Series)
n_a, n_b = len(group_a), len(group_b)

# 1. Normality Check via Shapiro-Wilk Test & Skewness
w_a, p_shap_a = stats.shapiro(group_a)
w_b, p_shap_b = stats.shapiro(group_b)
skew_a, skew_b = stats.skew(group_a), stats.skew(group_b)
kurt_a, kurt_b = stats.kurtosis(group_a), stats.kurtosis(group_b)

# 2. Outlier Check via Tukey's IQR Method
def get_extreme_outliers(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    return data[(data < q25 - 3 * iqr) | (data > q75 + 3 * iqr)]

ext_a = get_extreme_outliers(group_a)
ext_b = get_extreme_outliers(group_b)

# 3. Homogeneity of Variance via Levene's Test (Critical Assumption)
# center='median' is the robust Brown-Forsythe variant
lev_stat, p_levene = stats.levene(group_a, group_b, center="median")

print(f"Sample Sizes: n_A = {n_a}, n_B = {n_b}")
print(f"Shapiro-Wilk Normality: Group A p = {p_shap_a:.4f}, Group B p = {p_shap_b:.4f}")
print(f"Skewness: Group A = {skew_a:.2f}, Group B = {skew_b:.2f}")
print(f"Extreme Outliers (>3xIQR): Group A = {len(ext_a)}, Group B = {len(ext_b)}")
print(f"Levene's Equal Variance Test: W = {lev_stat:.3f}, p = {p_levene:.4f}")

# Decision Logic:
# - If p_levene >= 0.05 AND normality satisfied: Proceed with Student's t-test (equal_var=True) -> Step 2.
# - If p_levene < 0.05: Equal variance VIOLATED -> Switch to Welch's t-test (equal_var=False) -> Step 3.
# - If normality is violated and n < 30, or extreme outliers exist: Switch to Mann-Whitney U -> Step 3.
""", language="python")

    # STEP 2: IMPLEMENT STUDENT'S T-TEST IN PYTHON
    with st.expander("📌 Step 2: Implement Student's Two-Sample t-Test in Python (Equal Variance)", expanded=False):
        st.code("""# 1. Execute Student's t-test (equal_var=True)
res = stats.ttest_ind(group_b, group_a, equal_var=True)
t_stat = res.statistic
p_val = res.pvalue

# 2. Compute Summary Statistics & Exact Integer Degrees of Freedom
m_a, m_b = np.mean(group_a), np.mean(group_b)
v_a, v_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
mean_diff = m_b - m_a

df = n_a + n_b - 2  # Exact integer df for equal variance

# 3. Compute Pooled Variance and Standard Error
pooled_var = ((n_a - 1) * v_a + (n_b - 1) * v_b) / df
pooled_sd = np.sqrt(pooled_var)
se_diff = pooled_sd * np.sqrt(1.0 / n_a + 1.0 / n_b)

# 4. Compute 95% Confidence Interval for Mean Difference (B - A)
t_crit = stats.t.ppf(0.975, df=df)
ci_lower = mean_diff - t_crit * se_diff
ci_upper = mean_diff + t_crit * se_diff

# 5. Compute Standardized Effect Sizes (Pooled Cohen's d and Hedges' g)
cohens_d = mean_diff / pooled_sd
hedges_g = cohens_d * (1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0))

print(f"Group A Mean: {m_a:.2f} (SD: {np.sqrt(v_a):.2f})")
print(f"Group B Mean: {m_b:.2f} (SD: {np.sqrt(v_b):.2f})")
print(f"Pooled SD: {pooled_sd:.2f}")
print(f"Mean Difference (B - A): {mean_diff:+.2f} (SE: {se_diff:.2f})")
print(f"Student's t({df}) = {t_stat:.3f}, p-value: {p_val:.4e}")
print(f"95% CI for Difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Pooled Cohen's d: {cohens_d:.2f} | Hedges' g: {hedges_g:.2f}")

# How to interpret:
# - p_val < 0.05: Statistically significant difference between population means.
# - CI excludes 0.0: Strong evidence that Group B differs from Group A.
# - Cohen's d benchmarks: 0.2 small, 0.5 medium, 0.8 large.
""", language="python")

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Tests (Welch's t-Test or Mann-Whitney U) in Python", expanded=False):
        st.code("""# Alternative A: Welch's Two-Sample t-test (When equal variance assumption fails)
res_welch = stats.ttest_ind(group_b, group_a, equal_var=False)
print("=== Alternative A: Welch's t-Test (Unequal Variances) ===")
print(f"Welch t-statistic: {res_welch.statistic:.3f}, p-value: {res_welch.pvalue:.4e}")

# Alternative B: Mann-Whitney U Test (When normality fails or extreme outliers distort means)
u_res = stats.mannwhitneyu(group_b, group_a, alternative="two-sided")
cles = u_res.statistic / (n_a * n_b)
print("\\n=== Alternative B: Mann-Whitney U Test (Non-parametric) ===")
print(f"Mann-Whitney U: {u_res.statistic:.1f}, p-value: {u_res.pvalue:.4e}")
print(f"Common Language Effect Size (P(B > A)): {cles:.3f}")
""", language="python")

    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

def run_students_t_analysis(group_a, group_b, alpha=0.05):
    n_a, n_b = len(group_a), len(group_b)
    print(f"=== STUDENT'S TWO-SAMPLE t-TEST ANALYSIS (n_A = {n_a}, n_B = {n_b}) ===\\n")
    
    # 1. Assumption Diagnostics
    _, p_shap_a = stats.shapiro(group_a)
    _, p_shap_b = stats.shapiro(group_b)
    skew_a, skew_b = stats.skew(group_a), stats.skew(group_b)
    
    def count_ext_outliers(d):
        q25, q75 = np.percentile(d, [25, 75])
        iqr = q75 - q25
        return len(d[(d < q25 - 3 * iqr) | (d > q75 + 3 * iqr)])
    
    total_ext = count_ext_outliers(group_a) + count_ext_outliers(group_b)
    _, p_levene = stats.levene(group_a, group_b, center="median")
    
    normal_pass = (p_shap_a >= alpha and p_shap_b >= alpha) or (n_a >= 30 and n_b >= 30 and abs(skew_a) <= 1.2 and abs(skew_b) <= 1.2)
    outlier_pass = total_ext == 0
    equal_var_pass = p_levene >= alpha
    
    print(f"[Assumption] Normality: Shapiro A p={p_shap_a:.3f}, B p={p_shap_b:.3f} -> {'PASS' if normal_pass else 'VIOLATED'}")
    print(f"[Assumption] Outliers: {total_ext} extreme outliers -> {'PASS' if outlier_pass else 'VIOLATED'}")
    print(f"[Assumption] Equal Variance: Levene p={p_levene:.3f} -> {'PASS' if equal_var_pass else 'VIOLATED'}\\n")
    
    # 2. Decision Routing
    if not normal_pass or not outlier_pass:
        print("--> Normality / Outlier violation! Routing to Mann-Whitney U Test:")
        u_res = stats.mannwhitneyu(group_b, group_a, alternative="two-sided")
        cles = u_res.statistic / (n_a * n_b)
        print(f"Mann-Whitney U = {u_res.statistic:.1f}, p-value = {u_res.pvalue:.4e}")
        print(f"Common Language Effect Size (P(B > A)): {cles:.3f}")
        print(f"Medians: A = {np.median(group_a):.2f}, B = {np.median(group_b):.2f}")
    elif not equal_var_pass:
        print("--> Equal Variance violated! Routing to Welch's t-Test (Unequal Variances):")
        res_w = stats.ttest_ind(group_b, group_a, equal_var=False)
        m_a, m_b = np.mean(group_a), np.mean(group_b)
        v_a, v_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
        mean_diff = m_b - m_a
        se = np.sqrt(v_a / n_a + v_b / n_b)
        df_w = ((v_a/n_a + v_b/n_b)**2) / (((v_a/n_a)**2)/(n_a - 1) + ((v_b/n_b)**2)/(n_b - 1))
        t_crit = stats.t.ppf(1 - alpha/2, df=df_w)
        ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
        print(f"Welch t({df_w:.1f}) = {res_w.statistic:.3f}, p = {res_w.pvalue:.4e}")
        print(f"Means: A = {m_a:.2f}, B = {m_b:.2f} | Diff = {mean_diff:+.2f} (SE = {se:.2f})")
        print(f"95% CI for Difference: [{ci[0]:.2f}, {ci[1]:.2f}]")
    else:
        print("--> All assumptions satisfied! Executing Student's Two-Sample t-Test:")
        res = stats.ttest_ind(group_b, group_a, equal_var=True)
        m_a, m_b = np.mean(group_a), np.mean(group_b)
        v_a, v_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
        mean_diff = m_b - m_a
        df = n_a + n_b - 2
        sp2 = ((n_a - 1) * v_a + (n_b - 1) * v_b) / df
        sp = np.sqrt(sp2)
        se = sp * np.sqrt(1.0 / n_a + 1.0 / n_b)
        t_crit = stats.t.ppf(1 - alpha/2, df=df)
        ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
        d = mean_diff / sp
        g = d * (1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0))
        
        print(f"Means: A = {m_a:.2f}, B = {m_b:.2f} | Pooled SD = {sp:.2f}")
        print(f"Mean Difference: {mean_diff:+.2f} (SE: {se:.2f})")
        print(f"Student's t({df}) = {res.statistic:.3f}, p = {res.pvalue:.4e}")
        print(f"95% CI for Difference: [{ci[0]:.2f}, {ci[1]:.2f}]")
        print(f"Pooled Cohen's d: {d:.2f} | Hedges' g: {g:.2f}")

# Example Run:
# run_students_t_analysis(group_a, group_b)
""", language="python")


def render_paired_t_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for the Paired Samples t-test."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Normality & Outliers of Differences)", expanded=False):
        st.markdown(r"""
        The paired samples t-test compares means between two dependent/matched conditions (e.g., Pre- vs. Post-intervention). 
        The core assumption is that the **pairwise differences** ($D_i = \text{Post}_i - \text{Pre}_i$) are approximately normally distributed 
        and free of extreme outliers (or $N \ge 30$ pairs where the Central Limit Theorem applies).
        *(Note: The raw Pre and Post scores do NOT need to be normally distributed; only their difference scores must be normal.)*
        """)

        # 1. Generate deterministic sample data (n=35 matched pairs)
        rng = np.random.default_rng(42)
        n_pairs = 35
        pre_vals = np.round(rng.normal(loc=142.0, scale=14.0, size=n_pairs), 1)
        diff_vals = np.round(rng.normal(loc=-8.5, scale=6.0, size=n_pairs), 1)
        post_vals = np.round(pre_vals + diff_vals, 1)
        diff_vals = np.round(post_vals - pre_vals, 1)

        # Outlier detection on difference scores using Tukey's fences (IQR)
        q25_d = float(np.percentile(diff_vals, 25))
        q75_d = float(np.percentile(diff_vals, 75))
        iqr_d = q75_d - q25_d
        lower_d = q25_d - 1.5 * iqr_d
        upper_d = q75_d + 1.5 * iqr_d
        flags_d = np.where((diff_vals < lower_d) | (diff_vals > upper_d), "OUTLIER", "OK")

        example_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(n_pairs)],
            "Col_A_Subject": [f"Subj_{i+1:02d}" for i in range(n_pairs)],
            "Col_B_Pre [B2:B36]": pre_vals,
            "Col_C_Post [C2:C36]": post_vals,
            "Col_D_Diff [D2:D36]": diff_vals,
            "Col_E_Outlier_Check": flags_d
        })

        # 2. Dynamic diagnostics on differences (do not hardcode PASS/FAIL)
        n_cnt = len(diff_vals)
        outliers_cnt = int(np.sum(flags_d == "OUTLIER"))
        skew_d = float(stats.skew(diff_vals))
        kurt_d = float(stats.kurtosis(diff_vals))
        mean_d = float(np.mean(diff_vals))
        std_d = float(np.std(diff_vals, ddof=1))

        n_pass = "PASS" if n_cnt >= 30 else "WARN"
        outlier_pass = "PASS" if outliers_cnt == 0 else "WARN"
        skew_pass = "PASS" if abs(skew_d) <= 0.8 else "WARN"
        kurt_pass = "PASS" if abs(kurt_d) <= 1.0 else "WARN"

        summary_df = pd.DataFrame({
            "Assumption_Metric": [
                "Number of Matched Pairs (n)",
                "Difference Outliers (>1.5x IQR)",
                "Difference Skewness",
                "Difference Excess Kurtosis",
                "Mean Difference (d̄ = Post - Pre)",
                "Std Dev of Difference (s_d)"
            ],
            "Value": [
                f"{n_cnt}",
                f"{outliers_cnt}",
                f"{skew_d:.3f}",
                f"{kurt_d:.3f}",
                f"{mean_d:+.2f}",
                f"{std_d:.2f}"
            ],
            "Rule_Threshold": [
                "n >= 30 (CLT robust)",
                "0 Extreme Outliers",
                "Between -0.8 and +0.8",
                "Between -1.0 and +1.0",
                "Informational Metric",
                "Informational Metric"
            ],
            "Assumption_Result": [
                n_pass,
                outlier_pass,
                skew_pass,
                kurt_pass,
                "INFO",
                "INFO"
            ]
        })

        st.markdown("##### 📋 Example Spreadsheet Data (Pre: B2:B36, Post: C2:C36, Diff: D2:D36)")
        render_table(example_df, height=240)

        st.markdown("##### 📊 Assumption Diagnostic Summary (on Difference Scores)")
        render_table(summary_df, height=240)

        st.markdown("#### 1A. Testing Normality & Outliers of Differences in Excel")
        st.markdown(r"""
        Assuming Subject ID in `A2:A36`, Pre scores in `B2:B36`, and Post scores in `C2:C36`:
        """)
        st.code("""
Pairwise Difference Score (Enter in D2 and copy down to D36):
=C2 - B2

Sample Count of Pairs:
Count of Pairs (n):             =COUNT(D2:D36)

Normality & Distribution Shape of Differences:
Skewness of Differences:        =SKEW(D2:D36)
Kurtosis of Differences:        =KURT(D2:D36)

Outlier Detection on Differences (Tukey's IQR Method):
Q1 of Differences:              =QUARTILE.INC(D2:D36, 1)
Q3 of Differences:              =QUARTILE.INC(D2:D36, 3)
IQR:                            =Q3_cell - Q1_cell
Lower Extreme Fence:            =Q1_cell - 3 * IQR_cell
Upper Extreme Fence:            =Q3_cell + 3 * IQR_cell
Count Extreme Outliers:         =COUNTIF(D2:D36, "<" & Lower_Fence) + COUNTIF(D2:D36, ">" & Upper_Fence)
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret the Assumption Output:**
        * **Normality:** If $|\text{SKEW}| \le 0.8$, differences are approximately normally distributed. If $|\text{SKEW}| > 0.8$ but sample size is $N \ge 30$ pairs, **Central Limit Theorem applies** to the mean difference $\bar{D}$, and the paired t-test remains robust.
        * **Outliers:** If extreme difference outliers exist ($> 3 \times \text{IQR}$), they heavily distort the mean difference and inflate the standard error. Proceed to **Step 3 (Wilcoxon Signed-Rank Test)**.
        * **Dependence:** The paired t-test requires true matched pairs (the same individuals evaluated under two conditions). If groups are independent subjects, use **Student's or Welch's Two-Sample t-Test** instead.
        """)

    # STEP 2: IMPLEMENT PAIRED SAMPLES T-TEST IN EXCEL
    with st.expander("📌 Step 2: Implement Paired Samples t-Test in Excel (Native Formula, Difference Breakdown & Toolpak)", expanded=False):
        st.markdown(r"""
        #### 2A. Single Native Formula (Direct p-value)
        Excel provides a built-in function to compute the exact two-tailed paired $t$-test $p$-value directly:
        """)
        st.code("""
Two-Tailed Paired Samples t-Test p-value:
=T.TEST(B2:B36, C2:C36, 2, 1)
""", language="excel")
        st.caption("💡 **Formula arguments:** `array1` (Pre), `array2` (Post), `tails=2` (two-tailed), `type=1` (paired / dependent t-test). Notice `type=1` specifies paired design.")

        st.markdown("#### 2B. Complete Step-by-Step Breakdown on Difference Scores")
        st.code("""
Difference Score Summary Metrics:
Count of Pairs (n):             =COUNT(D2:D36)
Mean Difference (d̄):            =AVERAGE(D2:D36)
Std Dev of Difference (s_d):    =STDEV.S(D2:D36)

Degrees of Freedom (df):
df:                             =n_cell - 1

Standard Error of Difference (SE_d):
SE_d:                           =s_d_cell / SQRT(n_cell)

Paired t-Statistic:
t-Stat:                         =d̄_cell / SE_d_cell

Two-Tailed p-value:
p-value:                        =T.DIST.2T(ABS(t_cell), df_cell)

95% Confidence Interval for Mean Difference (μ_D):
Margin of Error:                =T.INV.2T(0.05, df_cell) * SE_d_cell
95% CI Lower Bound:             =d̄_cell - Margin_cell
95% CI Upper Bound:             =d̄_cell + Margin_cell

Standardized Effect Size (Cohen's d_z for Paired Samples):
Cohen's d_z:                    =d̄_cell / s_d_cell
Hedges' g_av (bias-corrected):  =d_z_cell * (1 - 3 / (4 * df_cell - 1))
""", language="excel")

        st.markdown("#### 2C. Excel Analysis ToolPak Automated Workflow")
        st.markdown(r"""
        1. Click **Data** tab ➔ **Data Analysis** (top right ribbon).
        2. Select **t-Test: Paired Two Sample for Means** and click **OK**.
        3. **Variable 1 Range:** `$B$1:$B$36` (Pre column with header).
        4. **Variable 2 Range:** `$C$1:$C$36` (Post column with header).
        5. **Hypothesized Mean Difference:** Enter `0` (null hypothesis: $\mu_D = 0$).
        6. Check **Labels** (if header row 1 is selected).
        7. **Alpha:** `0.05`.
        8. **Output Range:** Select `$G$1` and click **OK**.
        
        *The ToolPak automatically outputs: Mean Pre, Mean Post, Variance Pre, Variance Post, Pearson Correlation (r), Hypothesized Mean Difference, df ($n - 1$), t Stat, P(T<=t) two-tail, and t Critical two-tail.*
        """)

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output:**
        * **$p$-value $< 0.05$ (Statistically Significant):** There is strong evidence of a genuine within-subject change or treatment effect. Reject the null hypothesis $H_0: \mu_D = 0$.
        * **$p$-value $\ge 0.05$ (Not Significant):** Insufficient evidence to conclude that a true change occurred between Pre and Post measurements. The observed difference can be attributed to chance.
        * **95% Confidence Interval for $\mu_D$:** If zero (`0.0`) is **not** contained within `[CI_Lower, CI_Upper]`, the change is statistically significant at $\alpha = 0.05$.
        * **Cohen's $d_z$:** Standardized effect size in paired designs: $0.2$ = small, $0.5$ = medium, $0.8$ = large effect.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TEST IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Test in Excel (When Normality Fails: Wilcoxon Signed-Rank Test)", expanded=False):
        st.markdown(r"""
        If the pairwise difference scores violate normality with small sample size ($N < 30$) or contain extreme outliers, 
        switch to the non-parametric **Wilcoxon Signed-Rank Test (Paired)**.
        """)
        st.code("""
In the spreadsheet, add columns next to Differences (D2:D36):
Col E (Absolute Difference):    =ABS(D2)
Col F (Exclude Zeros Check):    =IF(D2 = 0, "", RANK.AVG(E2, $E$2:$E$36, 1))

Rank Sum Calculations:
Positive Rank Sum (W+):         =SUMIF($D$2:$D$36, ">0", $F$2:$F$36)
Negative Rank Sum (W-):         =SUMIF($D$2:$D$36, "<0", $F$2:$F$36)
Wilcoxon Test Statistic (W):    =MIN(W+_cell, W-_cell)

Effective Sample Size (n_eff):  =COUNTIF($D$2:$D$36, "<>0")

Large-Sample Normal Approximation (for n >= 20):
Expected Mean of W (μ_W):       =(n_eff * (n_eff + 1)) / 4
Std Error of W (σ_W):           =SQRT((n_eff * (n_eff + 1) * (2 * n_eff + 1)) / 24)
Z-Score (with continuity):      =(ABS(W_cell - μ_W_cell) - 0.5) / σ_W_cell
Two-Tailed p-value:             =2 * (1 - NORM.S.DIST(ABS(Z_cell), TRUE))
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret the Alternative Output:**
        * The Wilcoxon Signed-Rank test analyzes the **ranks and signs** of pairwise changes rather than raw arithmetic magnitudes.
        * It is completely protected against non-normal skewness and extreme outliers in difference scores.
        * If $p < 0.05$, there is a statistically significant median shift between Pre and Post measurements.
        """)


def render_paired_t_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python script guide for the Paired Samples t-test."""
    st.markdown(r"""
    ### 🐍 Python Code Implementation Workflow
    This script is divided into 3 modular sections: **Evaluate Assumptions** ➔ **Execute Proposed Test** ➔ **Execute Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Normality & Outliers of Differences)", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

# Given pre_vals and post_vals (1D numpy arrays or pandas Series of identical length)
n_pairs = len(pre_vals)
diffs = post_vals - pre_vals

# 1. Normality Check of Difference Scores via Shapiro-Wilk & Skewness
w_stat, p_shap = stats.shapiro(diffs)
skew_d = stats.skew(diffs)
kurt_d = stats.kurtosis(diffs)

# 2. Outlier Check in Difference Scores via Tukey's IQR Method
def get_extreme_outliers(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    return data[(data < q25 - 3 * iqr) | (data > q75 + 3 * iqr)]

ext_outliers = get_extreme_outliers(diffs)

# 3. Pearson Correlation Check between Paired Measures
r_corr, p_corr = stats.pearsonr(pre_vals, post_vals)

print(f"Number of Matched Pairs: n = {n_pairs}")
print(f"Shapiro-Wilk Normality on Differences: W = {w_stat:.4f}, p = {p_shap:.4f}")
print(f"Difference Skewness: {skew_d:.2f}, Excess Kurtosis: {kurt_d:.2f}")
print(f"Extreme Outliers in Differences (>3xIQR): {len(ext_outliers)}")
print(f"Pre-Post Correlation: r = {r_corr:.3f} (p = {p_corr:.4e})")

# Decision Logic:
# - If p_shap >= 0.05 OR (n_pairs >= 30 and |skew_d| <= 1.2): Normality satisfied -> Step 2.
# - If normality is violated with n < 30, or extreme outliers exist: Switch to Wilcoxon Signed-Rank Test -> Step 3.
""", language="python")

    # STEP 2: IMPLEMENT PAIRED SAMPLES T-TEST IN PYTHON
    with st.expander("📌 Step 2: Implement Paired Samples t-Test in Python", expanded=False):
        st.code("""# 1. Execute Paired t-test
res = stats.ttest_rel(post_vals, pre_vals)
t_stat = res.statistic
p_val = res.pvalue

# 2. Compute Summary Statistics on Differences
diffs = post_vals - pre_vals
n = len(diffs)
df = n - 1
mean_diff = np.mean(diffs)
std_diff = np.std(diffs, ddof=1)
se_diff = std_diff / np.sqrt(n)

# 3. Compute 95% Confidence Interval for Mean Difference (Post - Pre)
t_crit = stats.t.ppf(0.975, df=df)
ci_lower = mean_diff - t_crit * se_diff
ci_upper = mean_diff + t_crit * se_diff

# 4. Compute Standardized Effect Sizes (Cohen's dz and Hedges' gav)
cohens_dz = mean_diff / std_diff
hedges_gav = cohens_dz * (1.0 - 3.0 / (4.0 * df - 1.0))

print(f"Baseline Pre Mean: {np.mean(pre_vals):.2f} (SD: {np.std(pre_vals, ddof=1):.2f})")
print(f"Follow-up Post Mean: {np.mean(post_vals):.2f} (SD: {np.std(post_vals, ddof=1):.2f})")
print(f"Mean Difference (Post - Pre): {mean_diff:+.2f} (SE: {se_diff:.2f})")
print(f"Paired t({df}) = {t_stat:.3f}, p-value: {p_val:.4e}")
print(f"95% CI for Mean Difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Cohen's d_z: {cohens_dz:.2f} | Hedges' g_av: {hedges_gav:.2f}")

# How to interpret:
# - p_val < 0.05: Statistically significant within-subject change between Pre and Post.
# - CI excludes 0.0: Strong evidence that true change differs from zero.
# - Cohen's d_z benchmarks: 0.2 small, 0.5 medium, 0.8 large.
""", language="python")

    # STEP 3: IMPLEMENT ALTERNATIVE TEST IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Test (Wilcoxon Signed-Rank Test) in Python", expanded=False):
        st.code("""# Use when difference scores violate normality in small samples or have extreme outliers
wilc_res = stats.wilcoxon(post_vals, pre_vals, alternative="two-sided")
w_stat = wilc_res.statistic
p_wilc = wilc_res.pvalue

print("=== Alternative: Wilcoxon Signed-Rank Test (Paired) ===")
print(f"Wilcoxon W Statistic: {w_stat:.1f}, p-value: {p_wilc:.4e}")
print(f"Median Difference: {np.median(post_vals - pre_vals):+.2f}")

# How to interpret:
# - Tests whether the median difference between paired scores significantly deviates from zero.
# - Robust against extreme outliers and severe non-normality in difference scores.
""", language="python")

    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

def run_paired_t_analysis(pre_vals, post_vals, alpha=0.05):
    n = len(pre_vals)
    diffs = np.array(post_vals) - np.array(pre_vals)
    print(f"=== PAIRED SAMPLES t-TEST ANALYSIS (n = {n} pairs) ===\\n")
    
    # 1. Assumption Diagnostics on Differences
    _, p_shap = stats.shapiro(diffs)
    skew_d = stats.skew(diffs)
    
    def count_ext_outliers(d):
        q25, q75 = np.percentile(d, [25, 75])
        iqr = q75 - q25
        return len(d[(d < q25 - 3 * iqr) | (d > q75 + 3 * iqr)])
    
    total_ext = count_ext_outliers(diffs)
    normal_pass = p_shap >= alpha or (n >= 30 and abs(skew_d) <= 1.2)
    outlier_pass = total_ext == 0
    
    print(f"[Assumption] Normality of Differences: Shapiro p = {p_shap:.4f}, Skew = {skew_d:.2f} -> {'PASS' if normal_pass else 'VIOLATED'}")
    print(f"[Assumption] Difference Outliers: {total_ext} extreme outliers -> {'PASS' if outlier_pass else 'VIOLATED'}\\n")
    
    # 2. Decision Routing
    if not normal_pass or not outlier_pass:
        print("--> Differences violate assumptions! Routing to Wilcoxon Signed-Rank Test:")
        w_res = stats.wilcoxon(post_vals, pre_vals, alternative="two-sided")
        print(f"Wilcoxon W = {w_res.statistic:.1f}, p-value = {w_res.pvalue:.4e}")
        print(f"Median Difference: {np.median(diffs):+.2f}")
    else:
        print("--> Assumptions satisfied! Executing Paired Samples t-Test:")
        res = stats.ttest_rel(post_vals, pre_vals)
        mean_d = np.mean(diffs)
        std_d = np.std(diffs, ddof=1)
        df = n - 1
        se = std_d / np.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha/2, df=df)
        ci = (mean_d - t_crit * se, mean_d + t_crit * se)
        dz = mean_d / std_d
        gav = dz * (1.0 - 3.0 / (4.0 * df - 1.0))
        
        print(f"Pre Mean: {np.mean(pre_vals):.2f} | Post Mean: {np.mean(post_vals):.2f}")
        print(f"Mean Difference (Post - Pre): {mean_d:+.2f} (SE: {se:.2f})")
        print(f"Paired t({df}) = {res.statistic:.3f}, p-value = {res.pvalue:.4e}")
        print(f"95% CI for Mean Difference: [{ci[0]:.2f}, {ci[1]:.2f}]")
        print(f"Cohen's d_z: {dz:.2f} | Hedges' g_av: {gav:.2f}")

# Example Run:
# run_paired_t_analysis(pre_data, post_data)
""", language="python")


def render_wilcoxon_paired_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for the Paired Wilcoxon Signed-Rank Test."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Scale, Independence, Symmetry & Ties)", expanded=False):
        st.markdown(r"""
        The **Wilcoxon Signed-Rank Test (Paired)** is the standard non-parametric alternative to the paired t-test for **both continuous non-normal paired data and ordinal paired data** (e.g. Likert scale pre/post evaluations).
        
        The test analyzes the pairwise difference scores ($D_i = \text{Post}_i - \text{Pre}_i$). Core assumptions to verify:
        1. **Paired Design:** Observations consist of matched pairs or repeated measures from the same subjects.
        2. **Scale of Measurement:** At least ordinal (ranks of absolute differences are meaningful).
        3. **Symmetry of Differences:** The distribution of pairwise differences is approximately symmetric around their median.
        4. **Minimal Zero Ties:** Differences equal to 0 ($Post_i = Pre_i$) provide no directional information and are dropped from ranking, reducing effective sample size ($N_{\text{eff}}$).
        """)

        # 1. Generate deterministic sample data (n=30 matched pairs)
        rng = np.random.default_rng(42)
        n_pairs = 30
        pre_vals = np.round(rng.normal(loc=138.0, scale=12.0, size=n_pairs), 1)
        raw_diffs = rng.laplace(loc=-6.0, scale=5.0, size=n_pairs)
        diff_vals = np.round(raw_diffs, 1)
        # Ensure 1 tie / zero difference to demonstrate tie handling
        diff_vals[12] = 0.0
        post_vals = np.round(pre_vals + diff_vals, 1)
        diff_vals = np.round(post_vals - pre_vals, 1)
        abs_diffs = np.round(np.abs(diff_vals), 1)

        # Nonzero ranking with mid-ranks for ties
        nonzero_mask = diff_vals != 0.0
        ranks = np.zeros(n_pairs, dtype=float)
        if np.sum(nonzero_mask) > 0:
            ranks[nonzero_mask] = stats.rankdata(abs_diffs[nonzero_mask])
        
        pos_ranks = np.where(diff_vals > 0, ranks, 0.0)
        neg_ranks = np.where(diff_vals < 0, ranks, 0.0)

        example_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(n_pairs)],
            "Col_A_Subject": [f"Subj_{i+1:02d}" for i in range(n_pairs)],
            "Col_B_Pre [B2:B31]": pre_vals,
            "Col_C_Post [C2:C31]": post_vals,
            "Col_D_Diff [D2:D31]": diff_vals,
            "Col_E_AbsDiff [E2:E31]": abs_diffs,
            "Col_F_Rank [F2:F31]": ranks,
            "Col_G_PosRank [G2:G31]": pos_ranks,
            "Col_H_NegRank [H2:H31]": neg_ranks
        })

        # 2. Dynamic diagnostics on differences (do not hardcode PASS/FAIL)
        n_total = len(diff_vals)
        zero_cnt = int(np.sum(diff_vals == 0.0))
        n_eff = n_total - zero_cnt
        nonzero_diffs = diff_vals[nonzero_mask]
        skew_d = float(stats.skew(nonzero_diffs)) if n_eff > 2 else 0.0
        w_plus = float(np.sum(pos_ranks))
        w_minus = float(np.sum(neg_ranks))

        n_pass = "PASS" if n_eff >= 10 else ("WARN" if n_eff >= 6 else "FAIL")
        zero_pass = "PASS" if zero_cnt <= 2 else "WARN"
        skew_pass = "PASS" if abs(skew_d) <= 0.8 else ("WARN" if abs(skew_d) <= 1.5 else "FAIL")

        summary_df = pd.DataFrame({
            "Assumption_Metric": [
                "Total Matched Pairs (N)",
                "Zero Differences / Ties (Post = Pre)",
                "Effective Sample Size (N_eff)",
                "Difference Skewness (Symmetry Check)",
                "Sum of Positive Ranks (W+)",
                "Sum of Negative Ranks (W-)"
            ],
            "Value": [
                f"{n_total}",
                f"{zero_cnt}",
                f"{n_eff}",
                f"{skew_d:.3f}",
                f"{w_plus:.1f}",
                f"{w_minus:.1f}"
            ],
            "Rule_Threshold": [
                "N >= 10 pairs",
                "Minimal zero ties",
                "N_eff >= 10 (CLT / Normal approx)",
                "Between -0.8 and +0.8 (Symmetric)",
                "Informational Metric",
                "Informational Metric"
            ],
            "Assumption_Result": [
                "PASS",
                zero_pass,
                n_pass,
                skew_pass,
                "INFO",
                "INFO"
            ]
        })

        st.markdown("##### 📋 Example Spreadsheet Data (Pre: B2:B31, Post: C2:C31, Diff: D2:D31)")
        render_table(example_df, height=240)

        st.markdown("##### 📊 Assumption Diagnostic Summary (on Difference Scores)")
        render_table(summary_df, height=240)

        st.markdown("#### 1A. Testing Difference Symmetry and Ties in Excel")
        st.markdown(r"""
        Assuming Subject ID in `A2:A31`, Pre scores in `B2:B31`, and Post scores in `C2:C31`:
        """)
        st.code("""
Pairwise Difference Score (Enter in D2 and copy down to D31):
=C2 - B2

Absolute Difference Score (Enter in E2 and copy down to E31):
=ABS(D2)

Count of Zero Differences (No-change pairs):
=COUNTIF(D2:D31, 0)

Effective Sample Size (Pairs with non-zero difference):
=COUNT(D2:D31) - COUNTIF(D2:D31, 0)

Skewness of Difference Scores (Symmetry evaluation):
=SKEW(D2:D31)
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret the Assumption Output:**
        * **Difference Symmetry (`=SKEW(D2:D31)`):**
          * **$|\text{SKEW}| \le 0.8$:** Differences are approximately symmetric around the median difference. Assumption **PASSED**. The test is a pure test of the population median difference ($\text{Median}_D = 0$).
          * **$0.8 < |\text{SKEW}| \le 1.5$:** Moderate asymmetry. The test evaluates the *Hodges-Lehmann pseudo-median difference*. Proceed to Step 2, but check concordance with the Exact Sign Test in Step 3.
          * **$|\text{SKEW}| > 1.5$:** Severe asymmetry. Symmetry assumption is **VIOLATED**. Wilcoxon rank sums confound location with shape. **Skip to Step 3 (Exact Sign Test)**.
        * **Zero Differences ($D_i = 0$):** In ordinal surveys (e.g. Likert scales), participants who report no change ($Post = Pre$) receive differences of 0. Standard Wilcoxon drops these ties, reducing effective sample size. If over 25% of cases are zeros, compare results with the Sign Test.
        * **Continuous Non-Normal vs. Ordinal Data:** Both are handled identically by ranking the absolute differences `=RANK.AVG(E2, $E$2:$E$31, 1)`. In ordinal data, extensive tied ranks are resolved by average mid-ranks.
        """)

    # STEP 2: IMPLEMENT THE PROPOSED TEST IN EXCEL
    with st.expander("📌 Step 2: Implement Wilcoxon Signed-Rank Test for Paired Data in Excel", expanded=False):
        st.markdown(r"""
        Standard Excel does not contain a single `=WILCOXON.TEST()` formula, but the test can be computed transparently and accurately using rank formulas:
        """)
        st.code("""
Col B (Pre Scores):               [Raw Baseline Data B2:B31]
Col C (Post Scores):              [Raw Follow-up Data C2:C31]
Col D (Differences):              =C2 - B2
Col E (Absolute Differences):     =ABS(D2)
Col F (Average Mid-Ranks):        =IF(D2=0, "", RANK.AVG(E2, $E$2:$E$31, 1))
Col G (Positive Ranks):           =IF(D2>0, F2, 0)
Col H (Negative Ranks):           =IF(D2<0, F2, 0)

Summary Statistics:
Sum of Positive Ranks (W+):       =SUM(G2:G31)
Sum of Negative Ranks (W-):       =SUM(H2:H31)
Wilcoxon Test Statistic (W):      =MIN(SUM(G2:G31), SUM(H2:H31))
Effective Sample Size (n_eff):    =COUNTIF(D2:D31, "<>0")

Large-Sample Normal Approximation (for n_eff >= 10):
Expected Mean of W (μ_W):         =(n_eff * (n_eff + 1)) / 4
Std Error of W (σ_W):             =SQRT((n_eff * (n_eff + 1) * (2 * n_eff + 1)) / 24)
Z-Score (with continuity):        =(ABS(W_cell - μ_W_cell) - 0.5) / σ_W_cell
Two-Tailed p-value:               =2 * (1 - NORM.S.DIST(ABS(Z_cell), TRUE))

Non-Parametric Effect Sizes:
Rank-Biserial Correlation (r_rb): =(SUM(G2:G31) - SUM(H2:H31)) / (SUM(G2:G31) + SUM(H2:H31))
Correlation Effect Size (r):      =ABS(Z_cell) / SQRT(n_eff)
""", language="excel")

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output:**
        * **Test Statistic ($W$):** The smaller of the positive and negative rank sums ($W = \min(W^+, W^-)$).
        * **$p$-value $< 0.05$:** Statistically significant difference between Pre and Post conditions. Reject $H_0$ that median change is zero.
        * **$p$-value $\ge 0.05$:** Fail to reject $H_0$. Insufficient evidence of a consistent directional shift.
        * **Rank-Biserial Correlation ($r_{rb}$):** Ranges from $-1.0$ (all subjects worsened) to $+1.0$ (all subjects improved). $0.0$ indicates perfect balance between positive and negative shifts.
        * **Effect Size ($r = \frac{|Z|}{\sqrt{N_{\text{eff}}}}$):**
          * $|r| \approx 0.10$: Small effect
          * $|r| \approx 0.30$: Medium effect
          * $|r| \ge 0.50$: Large effect
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TEST IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Test in Excel (Exact Sign Test or Paired t-test)", expanded=False):
        st.markdown(r"""
        If assumptions are violated or your research question dictates an alternative method:
        * **If Difference Symmetry Fails ($|\text{SKEW}| > 1.5$) or Ordinal Scale has unequal intervals:** Switch to the **Exact Sign Test**.
        * **If Difference Scores are Continuous and Normally Distributed:** Switch back to the parametric **Paired Samples t-test**.
        """)

        st.markdown("#### Option 3A: Exact Sign Test in Excel (Zero Symmetry Assumptions)")
        st.markdown("""
        The Sign Test counts solely whether participants improved ($Post > Pre$) or worsened ($Post < Pre$), ignoring the magnitude of change:
        """)
        st.code("""
Count of Positive Changes (k):    =COUNTIF(D2:D31, ">0")
Count of Negative Changes:        =COUNTIF(D2:D31, "<0")
Count of Non-Zero Pairs (n_eff):  =COUNTIF(D2:D31, "<>0")

Two-Tailed Exact Sign Test p-value:
=2 * BINOM.DIST(MIN(k_cell, n_eff_cell - k_cell), n_eff_cell, 0.5, TRUE)
""", language="excel")

        st.markdown("#### Option 3B: Paired Samples t-test in Excel (If Differences are Normal)")
        st.code("""
Native Paired t-test p-value:
=T.TEST(B2:B31, C2:C31, 2, 1)
""", language="excel")


def render_wilcoxon_paired_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python script guide for the Paired Wilcoxon Signed-Rank Test."""
    st.markdown("""
    ### 🐍 Python Code Implementation Workflow
    This script is divided into 3 modular sections: **Evaluate Assumptions** ➔ **Execute Proposed Test** ➔ **Execute Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Scale, Symmetry & Ties)", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

# Given paired observations: pre_vals and post_vals (1D numpy arrays or pandas Series)
diffs = post_vals - pre_vals
nonzero_diffs = diffs[diffs != 0]
n_total = len(diffs)
n_eff = len(nonzero_diffs)
zero_count = n_total - n_eff

# 1. Symmetry check via skewness of difference scores
skew_val = float(stats.skew(nonzero_diffs)) if n_eff > 2 else 0.0
sym_pass = abs(skew_val) <= 0.8
sym_warning = 0.8 < abs(skew_val) <= 1.5

# 2. Effective sample size check
n_pass = n_eff >= 10

# 3. Normality test on difference scores (to determine if paired t-test was viable)
shapiro_stat, shapiro_p = stats.shapiro(diffs) if n_total >= 3 else (0.0, 1.0)

print(f"[Assumption] Total Pairs: {n_total} | Zero Differences (Ties): {zero_count} | Effective N: {n_eff}")
print(f"[Assumption] Difference Skewness: {skew_val:.3f} -> {'PASS (Symmetric)' if sym_pass else ('WARN (Moderate Skew)' if sym_warning else 'VIOLATED (Severe Skew)')}")
print(f"[Assumption] Shapiro-Wilk on Differences: p = {shapiro_p:.4f} -> {'Normal' if shapiro_p >= 0.05 else 'Non-Normal (Signed-Rank Appropriate)'}")
""", language="python")

    # STEP 2: EXECUTE PROPOSED TEST IN PYTHON
    with st.expander("📌 Step 2: Execute Wilcoxon Signed-Rank Test for Paired Data in Python", expanded=False):
        st.code("""# Execute Wilcoxon Signed-Rank Test on paired data
# Note: stats.wilcoxon automatically drops zero differences by default (zero_method='wilcox')
res = stats.wilcoxon(post_vals, pre_vals, alternative="two-sided")

# Extract ranks and compute effect sizes
nonzero_abs = np.abs(nonzero_diffs)
ranks = stats.rankdata(nonzero_abs)
pos_ranks = np.where(nonzero_diffs > 0, ranks, 0.0)
neg_ranks = np.where(nonzero_diffs < 0, ranks, 0.0)
w_plus = float(np.sum(pos_ranks))
w_minus = float(np.sum(neg_ranks))

# Rank-biserial correlation: r_rb = (W+ - W-) / (W+ + W-)
r_rb = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else 0.0

# Hodges-Lehmann median difference estimator (pairwise Walsh averages)
walsh_diffs = [(nonzero_diffs[i] + nonzero_diffs[j]) / 2.0 for i in range(n_eff) for j in range(i, n_eff)]
hl_median_diff = float(np.median(walsh_diffs))

print(f"Wilcoxon W Statistic: {res.statistic:.2f}")
print(f"p-value: {res.pvalue:.4e}")
print(f"Sum of Positive Ranks (W+): {w_plus:.1f} | Sum of Negative Ranks (W-): {w_minus:.1f}")
print(f"Hodges-Lehmann Median Difference: {hl_median_diff:+.2f}")
print(f"Rank-Biserial Effect Size (r_rb): {r_rb:+.3f}")
""", language="python")

    # STEP 3: EXECUTE ALTERNATIVE TEST IN PYTHON
    with st.expander("📌 Step 3: Execute Alternative Tests in Python (Exact Sign Test or Paired t-test)", expanded=False):
        st.code("""# Alternative 1: Exact Sign Test (when difference distribution is severely asymmetric)
pos_count = int(np.sum(nonzero_diffs > 0))
neg_count = int(np.sum(nonzero_diffs < 0))
sign_test_res = stats.binomtest(pos_count, n=n_eff, p=0.5, alternative="two-sided")

print(f"Sign Test: {pos_count} Positive vs {neg_count} Negative shifts")
print(f"Exact Sign Test p-value: {sign_test_res.pvalue:.4e}")

# Alternative 2: Paired Samples t-test (if differences are continuous & normal)
t_stat, t_pval = stats.ttest_rel(post_vals, pre_vals)
print(f"Paired t-test: t = {t_stat:.3f}, p-value = {t_pval:.4e}")
""", language="python")

    # COMPLETE PIPELINE
    with st.expander("📋 Complete Copy-Paste Python Pipeline", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

def run_paired_wilcoxon_analysis(pre_vals, post_vals, alpha=0.05):
    \"\"\"Complete, production-ready pipeline for Paired Wilcoxon Signed-Rank analysis.\"\"\"
    pre = np.asarray(pre_vals, dtype=float)
    post = np.asarray(post_vals, dtype=float)
    diffs = post - pre
    nonzero = diffs[diffs != 0]
    n_eff = len(nonzero)
    
    if n_eff < 5:
        raise ValueError(f"Insufficient non-zero differences ({n_eff}) for Wilcoxon analysis.")
    
    # 1. Assumption check: Symmetry of differences
    skew_val = float(stats.skew(nonzero))
    sym_pass = abs(skew_val) <= 0.8
    
    # 2. Wilcoxon Signed-Rank Test
    wilc_res = stats.wilcoxon(post, pre, alternative="two-sided")
    
    # 3. Effect size: Rank-biserial correlation
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = float(np.sum(np.where(nonzero > 0, ranks, 0.0)))
    w_minus = float(np.sum(np.where(nonzero < 0, ranks, 0.0)))
    r_rb = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else 0.0
    
    # 4. Hodges-Lehmann shift
    walsh = [(nonzero[i] + nonzero[j]) / 2.0 for i in range(n_eff) for j in range(i, n_eff)]
    hl_shift = float(np.median(walsh))
    
    # 5. Sign test fallback
    pos_cnt = int(np.sum(nonzero > 0))
    sign_res = stats.binomtest(pos_cnt, n=n_eff, p=0.5, alternative="two-sided")
    
    print("=" * 60)
    print("PAIRED NON-PARAMETRIC ANALYSIS REPORT")
    print("=" * 60)
    print(f"Pre Median: {np.median(pre):.2f} | Post Median: {np.median(post):.2f}")
    print(f"Median Difference: {np.median(diffs):+.2f} | Hodges-Lehmann Shift: {hl_shift:+.2f}")
    print(f"Difference Skewness: {skew_val:+.3f} -> {'SYMMETRIC (Median valid)' if sym_pass else 'ASYMMETRIC (Pseudo-median)'}")
    print(f"Wilcoxon W: {wilc_res.statistic:.1f}, p-value = {wilc_res.pvalue:.4e}")
    print(f"Rank-Biserial Correlation (r_rb): {r_rb:+.3f}")
    print(f"Exact Sign Test p-value: {sign_res.pvalue:.4e}")
    print("=" * 60)
    
    return {
        "w_stat": wilc_res.statistic,
        "p_wilcoxon": wilc_res.pvalue,
        "r_rb": r_rb,
        "hl_shift": hl_shift,
        "p_sign_test": sign_res.pvalue
    }

# Example Run:
# run_paired_wilcoxon_analysis(pre_scores, post_scores)
""", language="python")


def render_one_way_anova_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for One-Way ANOVA."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Normality, Homoscedasticity & Outliers)", expanded=False):
        st.markdown(r"""
        One-Way Analysis of Variance (ANOVA) tests whether means across three or more independent groups differ significantly. 
        Before interpreting the omnibus $F$-test, verify that:
        1. **Independent Observations & Continuous Scale:** Groups are mutually exclusive and measured on an interval or ratio scale.
        2. **Normality:** Distributions within each group are approximately normal (or sample sizes $n_k \ge 30$ via the Central Limit Theorem).
        3. **Homogeneity of Variance (Homoscedasticity):** Group variances are approximately equal ($F_{\max} = s_{\max}^2 / s_{\min}^2 \le 4.0$).
        4. **Absence of Extreme Outliers:** No extreme outliers ($> 3 \times \text{IQR}$) distorting group means or inflating Mean Square Error.
        """)

        # 1. Generate deterministic sample data across 3 groups (n=25 per group)
        rng = np.random.default_rng(42)
        n_per = 25
        g1_vals = np.round(rng.normal(loc=52.0, scale=7.5, size=n_per), 1)
        g2_vals = np.round(rng.normal(loc=58.5, scale=8.0, size=n_per), 1)
        g3_vals = np.round(rng.normal(loc=65.0, scale=8.2, size=n_per), 1)

        def count_ext_outliers(vals):
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr = q75 - q25
            return int(np.sum((vals < q25 - 3.0 * iqr) | (vals > q75 + 3.0 * iqr)))

        example_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(n_per)],
            "Col_A_Subject": [f"Patient_{i+1:02d}" for i in range(n_per)],
            "Col_B_Standard_Care [B2:B26]": g1_vals,
            "Col_C_Treatment_Low [C2:C26]": g2_vals,
            "Col_D_Treatment_High [D2:D26]": g3_vals
        })

        # 2. Dynamic diagnostics across groups
        m1, m2, m3 = float(np.mean(g1_vals)), float(np.mean(g2_vals)), float(np.mean(g3_vals))
        s1, s2, s3 = float(np.std(g1_vals, ddof=1)), float(np.std(g2_vals, ddof=1)), float(np.std(g3_vals, ddof=1))
        v1, v2, v3 = s1 ** 2, s2 ** 2, s3 ** 2
        skew1, skew2, skew3 = float(stats.skew(g1_vals)), float(stats.skew(g2_vals)), float(stats.skew(g3_vals))
        kurt1, kurt2, kurt3 = float(stats.kurtosis(g1_vals)), float(stats.kurtosis(g2_vals)), float(stats.kurtosis(g3_vals))
        out1, out2, out3 = count_ext_outliers(g1_vals), count_ext_outliers(g2_vals), count_ext_outliers(g3_vals)
        var_ratio = max(v1, v2, v3) / min(v1, v2, v3)

        norm_pass = "PASS" if max(abs(skew1), abs(skew2), abs(skew3)) <= 0.8 else "WARN"
        homo_pass = "PASS" if var_ratio <= 2.5 else ("WARN" if var_ratio <= 4.0 else "FAIL")
        outlier_pass = "PASS" if (out1 + out2 + out3) == 0 else "FAIL"

        summary_df = pd.DataFrame({
            "Assumption_Metric": [
                "Sample Size (n_k)",
                "Group Mean (x̄_k)",
                "Group Std Dev (s_k)",
                "Group Variance (s_k²)",
                "Sample Skewness",
                "Excess Kurtosis",
                "Extreme Outliers (>3x IQR)"
            ],
            "Standard_Care": [
                f"{n_per}", f"{m1:.2f}", f"{s1:.2f}", f"{v1:.2f}", f"{skew1:.2f}", f"{kurt1:.2f}", f"{out1}"
            ],
            "Treatment_Low": [
                f"{n_per}", f"{m2:.2f}", f"{s2:.2f}", f"{v2:.2f}", f"{skew2:.2f}", f"{kurt2:.2f}", f"{out2}"
            ],
            "Treatment_High": [
                f"{n_per}", f"{m3:.2f}", f"{s3:.2f}", f"{v3:.2f}", f"{skew3:.2f}", f"{kurt3:.2f}", f"{out3}"
            ],
            "Rule_Threshold": [
                "n_k >= 20 (robust >= 30)",
                "Informational Metric",
                "Informational Metric",
                "Max/Min Ratio <= 4.0",
                "Between -0.8 and +0.8",
                "Between -1.0 and +1.0",
                "0 Extreme Outliers"
            ],
            "Assumption_Result": [
                "PASS", "INFO", "INFO", homo_pass, norm_pass, "PASS", outlier_pass
            ]
        })

        st.markdown("##### 📋 Example Spreadsheet Data (Group 1: B2:B26, Group 2: C2:C26, Group 3: D2:D26)")
        render_table(example_df, height=240)

        st.markdown("##### 📊 Assumption Diagnostic Summary Across Groups")
        render_table(summary_df, height=240)

        st.markdown("#### 1A. Testing Normality and Homoscedasticity in Excel")
        st.markdown(r"""
        Assuming Group 1 in `B2:B26`, Group 2 in `C2:C26`, and Group 3 in `D2:D26`:
        """)
        st.code("""
Descriptive Metrics per Group:
Group Means:                      =AVERAGE(B2:B26)      | =AVERAGE(C2:C26)      | =AVERAGE(D2:D26)
Group Variances:                  =VAR.S(B2:B26)        | =VAR.S(C2:C26)        | =VAR.S(D2:D26)
Group Std Deviations:             =STDEV.S(B2:B26)      | =STDEV.S(C2:C26)      | =STDEV.S(D2:D26)
Sample Counts:                    =COUNT(B2:B26)        | =COUNT(C2:C26)        | =COUNT(D2:D26)

Normality Diagnostics (Skewness & Kurtosis):
Group Skewness:                   =SKEW(B2:B26)         | =SKEW(C2:C26)         | =SKEW(D2:D26)
Group Kurtosis:                   =KURT(B2:B26)         | =KURT(C2:C26)         | =KURT(D2:D26)

Homogeneity of Variance Check (Hartley's F_max Ratio):
Max Group Variance:               =MAX(VAR.S(B2:B26), VAR.S(C2:C26), VAR.S(D2:D26))
Min Group Variance:               =MIN(VAR.S(B2:B26), VAR.S(C2:C26), VAR.S(D2:D26))
Variance Ratio (F_max):           =Max_Var / Min_Var

Outlier Fences (Tukey's IQR Method for Group 1):
Q1: =QUARTILE.INC(B2:B26, 1)      | Q3: =QUARTILE.INC(B2:B26, 3)
IQR: =Q3 - Q1                     | Extreme Fences: Q1 - 3*IQR  and  Q3 + 3*IQR
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret the Assumption Output:**
        * **Homogeneity of Variance ($F_{\max} \le 4.0$):**
          * **$F_{\max} \le 2.5$:** Variances are homogenous. Assumption **PASSED**. Standard ANOVA $F$-test is fully valid.
          * **$2.5 < F_{\max} \le 4.0$:** Moderate variance difference. Standard ANOVA remains robust when group sample sizes are balanced ($n_1 = n_2 = n_3$).
          * **$F_{\max} > 4.0$:** Heteroscedasticity. Type I error rates are distorted. **Skip to Step 3 (Welch's ANOVA)**.
        * **Normality Check ($|\text{SKEW}| \le 0.8$):**
          * If all $|\text{SKEW}| \le 0.8$, distributions are normal. If $|\text{SKEW}| > 0.8$ but all $n_k \ge 30$, **Central Limit Theorem applies**. If sample sizes are small ($n_k < 30$) and skewed, **Skip to Step 3 (Kruskal-Wallis Test)**.
        * **Outliers:** Extreme outliers inflate $MS_{\text{within}}$ (the error denominator) and falsely reduce the $F$-statistic.
        """)

    # STEP 2: IMPLEMENT THE PROPOSED TEST IN EXCEL
    with st.expander("📌 Step 2: Implement One-Way ANOVA in Excel", expanded=False):
        st.markdown(r"""
        Excel offers two complementary approaches: the automated **Data Analysis Toolpak**, and transparent **native worksheet formulas**.
        """)

        st.markdown("#### Method A: Automated Data Analysis Toolpak (Fastest)")
        st.markdown("""
        1. Click **Data** tab ➔ **Data Analysis** (top right).
        2. Select **Anova: Single Factor** and click **OK**.
        3. **Input Range:** Select `$B$1:$D$26` (including headers).
        4. Check **Grouped By: Columns** and check **Labels in first row**.
        5. Set **Alpha:** `0.05` and choose Output Range ➔ Click **OK**.
        6. Excel outputs the complete ANOVA summary table with Between-Groups and Within-Groups SS, df, MS, F, and p-value.
        """)

        st.markdown("#### Method B: Transparent Native Excel Formulas")
        st.code("""
Summary Statistics:
Total Sample Size (N):            =COUNT(B2:D26)
Number of Groups (k):             =3
Grand Mean (x̄_grand):             =AVERAGE(B2:D26)

Degrees of Freedom:
Between Groups df (df_b):         =k - 1                                 [= 2]
Within Groups df (df_w):          =N - k                                 [= 72]
Total df (df_tot):                =N - 1                                 [= 74]

Sum of Squares (SS):
Between Sum of Squares (SS_b):    =COUNT(B2:B26)*(AVERAGE(B2:B26)-grand_mean)^2
                                  +COUNT(C2:C26)*(AVERAGE(C2:C26)-grand_mean)^2
                                  +COUNT(D2:D26)*(AVERAGE(D2:D26)-grand_mean)^2
Within Sum of Squares (SS_w):     =(COUNT(B2:B26)-1)*VAR.S(B2:B26)
                                  +(COUNT(C2:C26)-1)*VAR.S(C2:C26)
                                  +(COUNT(D2:D26)-1)*VAR.S(D2:D26)
Total Sum of Squares (SS_tot):    =SS_b + SS_w

Mean Squares (MS):
Mean Square Between (MS_b):       =SS_b / df_b
Mean Square Within / Error (MS_w):=SS_w / df_w

F-Statistic and p-value:
Omnibus F-Ratio:                  =MS_b / MS_w
Omnibus p-value:                  =F.DIST.RT(F_cell, df_b, df_w)
Critical F-Value (alpha = 0.05):  =F.INV.RT(0.05, df_b, df_w)

Effect Sizes:
Eta-Squared (η²):                 =SS_b / SS_tot
Omega-Squared (ω²):               =MAX(0, (SS_b - df_b * MS_w) / (SS_tot + MS_w))
""", language="excel")

        st.markdown("#### Method C: Post-Hoc Pairwise Comparisons in Excel (Pinpointing Group Differences)")
        st.markdown(r"""
        The omnibus ANOVA test above tells you that the groups differ overall, but **does not tell you which specific groups differ from each other**. 
        To find out, we must run pairwise comparisons while controlling for the **risk of false positives**.
        """)
        st.code("""
Pairwise Difference in Means:
Care vs Low Dose:                 =AVERAGE(C2:C26) - AVERAGE(B2:B26)
Care vs High Dose:                =AVERAGE(D2:D26) - AVERAGE(B2:B26)
Low Dose vs High Dose:            =AVERAGE(D2:D26) - AVERAGE(C2:C26)

Pooled Standard Error of Difference (SE_diff for balanced groups n=25):
SE_diff = SQRT(2 * MS_w / n):     =SQRT(2 * MS_w_cell / 25)

Bonferroni-Adjusted Pairwise t-Tests (Guarantees overall false positive rate <= 5%):
Care vs Low Dose:                 =MIN(1.0, 3 * T.TEST(B2:B26, C2:C26, 2, 2))
Care vs High Dose:                =MIN(1.0, 3 * T.TEST(B2:B26, D2:D26, 2, 2))
Low Dose vs High Dose:            =MIN(1.0, 3 * T.TEST(C2:C26, D2:D26, 2, 2))

Tukey's Critical Minimum Difference (HSD):
Studentized Range Critical q:     3.40   (from Tukey q table for k=3 groups, df=72 at alpha=0.05)
Tukey HSD Threshold:              =3.40 * SQRT(MS_w_cell / 25)
Significant if:                   =IF(ABS(Mean_Diff) >= HSD_Threshold, "SIGNIFICANT (*)", "Not Significant")
""", language="excel")

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output in Plain English:**
        * **Why Can't We Just Run Normal t-Tests?** Running 3 ordinary t-tests each at a 5% false alarm rate inflates your overall chance of a false positive to **$1 - (1 - 0.05)^3 = 14.3\%$**. Multiplying the p-value by 3 (`=MIN(1.0, 3 * T.TEST(...))`) or using Tukey's HSD threshold mathematically guarantees that your total risk of a false alarm remains strictly capped at **5%**.
        * **Interpreting the Pairwise Findings:**
          * If an adjusted $p$-value is $< 0.05$ (or the mean difference exceeds Tukey's HSD), that specific pair is **statistically significantly different**.
          * A positive difference means the second group scored higher; a negative difference means the first group scored higher.
          * If an adjusted $p$-value is $\ge 0.05$, there is **no genuine difference** between those two groups—any difference observed in your sample is consistent with random chance.
        * **Eta-Squared ($\eta^2$):** Proportion of total variance explained by group membership:
          * $\eta^2 \approx 0.01$: Small effect
          * $\eta^2 \approx 0.06$: Medium effect
          * $\eta^2 \ge 0.14$: Large effect
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Tests in Excel (Welch's ANOVA or Kruskal-Wallis)", expanded=False):
        st.markdown(r"""
        When ANOVA assumptions are violated:
        * **If Variances are Unequal ($F_{\max} > 4.0$):** Switch to **Welch's ANOVA**.
        * **If Data is Non-Normal, Ordinal, or has Extreme Outliers:** Switch to the non-parametric **Kruskal-Wallis H Test**.
        """)

        st.markdown("#### Option 3A: Welch's ANOVA in Excel (Heteroscedasticity)")
        st.markdown("""
        When variances differ, standard ANOVA produces false positive rates up to 30%. Welch's ANOVA weights each group by $w_k = n_k / s_k^2$:
        """)
        st.code("""
Group Weights:                    w1 = n1 / s1^2,  w2 = n2 / s2^2,  w3 = n3 / s3^2
Weighted Grand Mean:              x̄_w = (w1*x̄1 + w2*x̄2 + w3*x̄3) / (w1 + w2 + w3)

Pairwise Welch t-tests in Excel with Bonferroni correction:
Care vs Low Dose (Type=3):        =MIN(1.0, 3 * T.TEST(B2:B26, C2:C26, 2, 3))
Care vs High Dose (Type=3):       =MIN(1.0, 3 * T.TEST(B2:B26, D2:D26, 2, 3))
Low vs High Dose (Type=3):        =MIN(1.0, 3 * T.TEST(C2:C26, D2:D26, 2, 3))
""", language="excel")

        st.markdown("#### Option 3B: Kruskal-Wallis H Test in Excel (Non-Parametric Ranks)")
        st.markdown("""
        The Kruskal-Wallis test ranks all $N$ observations together across all groups and tests rank sum differences:
        """)
        st.code("""
In Stacked Data Column:
Rank all observations:            =RANK.AVG(Score_cell, $Score$Range, 1)

Sum of Ranks per Group:
R1 (Standard Care Rank Sum):      =SUMIF(Group_Range, "Standard Care", Rank_Range)
R2 (Low Dose Rank Sum):           =SUMIF(Group_Range, "Treatment Low", Rank_Range)
R3 (High Dose Rank Sum):          =SUMIF(Group_Range, "Treatment High", Rank_Range)

Kruskal-Wallis H Statistic:
=(12 / (N * (N + 1))) * ((R1^2 / n1) + (R2^2 / n2) + (R3^2 / n3)) - 3 * (N + 1)

Kruskal-Wallis p-value:
=CHISQ.DIST.RT(H_cell, k - 1)
""", language="excel")


def render_one_way_anova_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python script guide for One-Way ANOVA."""
    st.markdown("""
    ### 🐍 Python Code Implementation Workflow
    This script is divided into 3 modular sections: **Evaluate Assumptions** ➔ **Execute Proposed Test** ➔ **Execute Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Normality, Homoscedasticity & Outliers)", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

# Given 3 independent group arrays: g1, g2, g3
groups = [g1, g2, g3]
group_names = ["Standard Care", "Treatment Low Dose", "Treatment High Dose"]

# 1. Normality check per group (Shapiro-Wilk)
shapiro_results = [stats.shapiro(g) for g in groups]
for name, (w, p) in zip(group_names, shapiro_results):
    print(f"[Assumption] Normality ({name}): Shapiro W = {w:.4f}, p = {p:.4f} -> {'PASS' if p >= 0.05 else 'VIOLATED'}")

# 2. Homogeneity of Variance (Levene's Test)
lev_stat, lev_p = stats.levene(*groups, center='median')
vars_arr = [np.var(g, ddof=1) for g in groups]
var_ratio = max(vars_arr) / min(vars_arr)
homo_pass = lev_p >= 0.05 or var_ratio <= 4.0
print(f"[Assumption] Homoscedasticity: Levene p = {lev_p:.4f}, F_max = {var_ratio:.2f} -> {'PASS' if homo_pass else 'VIOLATED'}")

# 3. Outlier Detection per group (Tukey's IQR)
for name, g in zip(group_names, groups):
    q25, q75 = np.percentile(g, [25, 75])
    iqr = q75 - q25
    ext_outliers = g[(g < q25 - 3 * iqr) | (g > q75 + 3 * iqr)]
    print(f"[Assumption] Outliers ({name}): {len(ext_outliers)} extreme outliers")
""", language="python")

    # STEP 2: EXECUTE PROPOSED TEST IN PYTHON
    with st.expander("📌 Step 2: Execute One-Way ANOVA and Post-Hoc Tests in Python", expanded=False):
        st.code("""# 1. Omnibus One-Way ANOVA
f_stat, p_val = stats.f_oneway(g1, g2, g3)
print(f"One-Way ANOVA: F = {f_stat:.4f}, p-value = {p_val:.4e}")

# 2. Effect Sizes (Eta-squared and Omega-squared)
k = len(groups)
n_total = sum(len(g) for g in groups)
grand_mean = np.mean(np.concatenate(groups))

ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
ss_total = ss_between + ss_within
df_between = k - 1
df_within = n_total - k
ms_within = ss_within / df_within

eta_sq = ss_between / ss_total
omega_sq = max(0.0, (ss_between - df_between * ms_within) / (ss_total + ms_within))
print(f"Effect Sizes: Eta-squared (η²) = {eta_sq:.3f}, Omega-squared (ω²) = {omega_sq:.3f}")

# 3. Post-Hoc Pairwise Comparisons (Tukey's HSD)
tukey_res = stats.tukey_hsd(g1, g2, g3)
print("\\nPost-Hoc Tukey HSD Pairwise Comparisons:")
print(tukey_res)
""", language="python")

    # STEP 3: EXECUTE ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Execute Alternative Tests in Python (Welch's ANOVA or Kruskal-Wallis)", expanded=False):
        st.code("""# Alternative 1: Non-Parametric Kruskal-Wallis H Test (if non-normal or extreme outliers)
krusk_stat, krusk_p = stats.kruskal(g1, g2, g3)
print(f"Kruskal-Wallis: H = {krusk_stat:.4f}, p-value = {krusk_p:.4e}")

# Alternative 2: Welch's ANOVA (if heteroscedastic / unequal variances)
# Scipy Alexander-Govern test provides an unequal-variance multi-group test:
ag_res = stats.alexandergovern(g1, g2, g3)
print(f"Alexander-Govern (Welch-type): A = {ag_res.statistic:.4f}, p-value = {ag_res.pvalue:.4e}")
""", language="python")

    # COMPLETE PIPELINE
    with st.expander("📋 Complete Copy-Paste Python Pipeline", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

def run_one_way_anova_pipeline(groups_dict, alpha=0.05):
    \"\"\"Complete, production-ready pipeline for One-Way ANOVA with assumption checking.\"\"\"
    group_names = list(groups_dict.keys())
    groups = [np.asarray(groups_dict[k], dtype=float) for k in group_names]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    
    # 1. Assumptions: Normality & Homoscedasticity
    norm_passes = [stats.shapiro(g)[1] >= alpha for g in groups]
    lev_stat, lev_p = stats.levene(*groups, center='median')
    vars_list = [np.var(g, ddof=1) for g in groups]
    var_ratio = max(vars_list) / (min(vars_list) if min(vars_list) > 0 else 1.0)
    
    homo_pass = (lev_p >= alpha) or (var_ratio <= 4.0)
    all_normal = all(norm_passes)
    
    print("=" * 65)
    print("ONE-WAY ANOVA STATISTICAL REPORT")
    print("=" * 65)
    print(f"Groups Analyzed: {group_names}")
    print(f"Normality Check: {'PASS' if all_normal else 'VIOLATED (Check Kruskal-Wallis)'}")
    print(f"Equal Variances Check: Levene p = {lev_p:.4f}, F_max = {var_ratio:.2f} -> {'PASS' if homo_pass else 'VIOLATED (Check Welch)'}")
    print("-" * 65)
    
    # 2. ANOVA Execution
    f_stat, p_val = stats.f_oneway(*groups)
    grand_mean = np.mean(np.concatenate(groups))
    ss_b = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_w = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
    ss_tot = ss_b + ss_w
    eta_sq = ss_b / ss_tot if ss_tot > 0 else 0.0
    
    print(f"Omnibus F({k-1}, {n_total-k}) = {f_stat:.4f}, p-value = {p_val:.4e}")
    print(f"Eta-Squared (η²): {eta_sq:.3f}")
    
    # 3. Post-Hoc if Significant
    if p_val < alpha:
        print("\\nPost-Hoc Tukey HSD Pairwise Results:")
        tukey = stats.tukey_hsd(*groups)
        print(tukey)
    else:
        print("\\nOmnibus test not significant; pairwise post-hoc tests not indicated.")
    print("=" * 65)
    
    return {
        "f_stat": f_stat,
        "p_val": p_val,
        "eta_squared": eta_sq,
        "levene_p": lev_p,
        "all_normal": all_normal,
        "homo_pass": homo_pass
    }

# Example Run:
# run_one_way_anova_pipeline({
#     "Standard Care": [52.1, 48.3, 55.4, ...],
#     "Treatment Low": [58.2, 59.1, 61.3, ...],
#     "Treatment High": [64.5, 67.2, 63.8, ...]
# })
""", language="python")


def render_kruskal_wallis_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for the Kruskal-Wallis H Test."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Scale, Independence, Group Sizes & Shape)", expanded=False):
        st.markdown(r"""
        The **Kruskal-Wallis H Test** is the premier non-parametric alternative to One-Way ANOVA for comparing 3 or more independent groups. 
        Before evaluating the omnibus $H$-statistic, verify that:
        1. **Independent Random Sampling:** Observations across all $k$ groups are mutually exclusive and sampled independently.
        2. **Measurement Scale:** The outcome variable is measured on at least an **ordinal** scale (ordered categories, Likert ratings) or a **continuous** scale with non-normal distributions or extreme outliers.
        3. **Adequate Group Sample Sizes ($n_k \ge 5$):** For the Chi-Square distribution to accurately model $H$, each group must contain at least 5 observations (ideally $n_k \ge 15$).
        4. **Distribution Shape Similarity:**
           - If distribution shapes are similar across groups $\rightarrow$ Kruskal-Wallis directly tests differences between **population medians**.
           - If distribution shapes differ $\rightarrow$ Kruskal-Wallis tests **stochastic dominance / mean rank differences** (whether one group systematically scores higher than another).
        """)

        # 1. Generate deterministic sample data across 3 groups (n=20 per group, Likert / recovery ratings)
        rng = np.random.default_rng(42)
        n_per = 20
        # Group 1: Moderate scores (Median ~ 5.0)
        g1_raw = 3.5 + rng.gamma(shape=2.0, scale=1.2, size=n_per)
        g1_vals = np.clip(np.round(g1_raw), 1.0, 10.0).astype(float)
        # Group 2: Slightly elevated (Median ~ 6.5)
        g2_raw = 4.8 + rng.gamma(shape=2.0, scale=1.2, size=n_per)
        g2_vals = np.clip(np.round(g2_raw), 1.0, 10.0).astype(float)
        # Group 3: Substantially higher (Median ~ 8.0)
        g3_raw = 6.2 + rng.gamma(shape=2.0, scale=1.2, size=n_per)
        g3_vals = np.clip(np.round(g3_raw), 1.0, 10.0).astype(float)

        example_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(n_per)],
            "Col_A_Subject": [f"Patient_{i+1:02d}" for i in range(n_per)],
            "Col_B_Standard_Care [B2:B21]": g1_vals,
            "Col_C_Treatment_Low [C2:C21]": g2_vals,
            "Col_D_Treatment_High [D2:D21]": g3_vals
        })

        # 2. Dynamic group diagnostics
        med1, med2, med3 = float(np.median(g1_vals)), float(np.median(g2_vals)), float(np.median(g3_vals))
        q25_1, q25_2, q25_3 = float(np.percentile(g1_vals, 25)), float(np.percentile(g2_vals, 25)), float(np.percentile(g3_vals, 25))
        q75_1, q75_2, q75_3 = float(np.percentile(g1_vals, 75)), float(np.percentile(g2_vals, 75)), float(np.percentile(g3_vals, 75))
        iqr1, iqr2, iqr3 = q75_1 - q25_1, q75_2 - q25_2, q75_3 - q25_3
        s1, s2, s3 = float(np.std(g1_vals, ddof=1)), float(np.std(g2_vals, ddof=1)), float(np.std(g3_vals, ddof=1))
        skew1, skew2, skew3 = float(stats.skew(g1_vals)), float(stats.skew(g2_vals)), float(stats.skew(g3_vals))

        size_pass = "PASS" if n_per >= 15 else ("WARN" if n_per >= 5 else "FAIL")
        max_skew_diff = max(skew1, skew2, skew3) - min(skew1, skew2, skew3)
        shape_pass = "PASS" if max_skew_diff <= 1.0 else "WARN"

        summary_df = pd.DataFrame({
            "Assumption_Metric": [
                "Sample Size (n_k)",
                "Group Median (x̃_k)",
                "Interquartile Range (IQR)",
                "Sample Skewness",
                "Sample Std Dev (s_k)",
                "Minimum Sample Rule (>= 5)"
            ],
            "Standard_Care": [
                f"{n_per}", f"{med1:.1f}", f"{iqr1:.1f}", f"{skew1:.2f}", f"{s1:.2f}", f"n = {n_per} (PASS)"
            ],
            "Treatment_Low": [
                f"{n_per}", f"{med2:.1f}", f"{iqr2:.1f}", f"{skew2:.2f}", f"{s2:.2f}", f"n = {n_per} (PASS)"
            ],
            "Treatment_High": [
                f"{n_per}", f"{med3:.1f}", f"{iqr3:.1f}", f"{skew3:.2f}", f"{s3:.2f}", f"n = {n_per} (PASS)"
            ],
            "Rule_Threshold": [
                "n_k >= 5 (robust >= 15)",
                "Primary Location Metric",
                "Measure of Dispersion",
                "Similar across groups",
                "Informational Metric",
                "Required for Chi-Square H"
            ],
            "Assumption_Result": [
                size_pass, "INFO", "INFO", shape_pass, "INFO", "PASS"
            ]
        })

        st.markdown("##### 📋 Example Spreadsheet Data (Unstacked Layout: B2:B21, C2:C21, D2:D21)")
        render_table(example_df, height=240)

        st.markdown("##### 📊 Assumption Diagnostic Summary Across Groups")
        render_table(summary_df, height=240)

        # 3. Stacked Global Ranking Table Preview
        all_unstacked = [g1_vals, g2_vals, g3_vals]
        all_group_names = ["Standard Care", "Treatment Low", "Treatment High"]
        stacked_rows = []
        for g_idx, g_arr in enumerate(all_unstacked):
            for i, val in enumerate(g_arr):
                stacked_rows.append({
                    "Subject": f"Subj_{g_idx+1}_{i+1:02d}",
                    "Group": all_group_names[g_idx],
                    "Score": float(val)
                })
        stacked_demo_df = pd.DataFrame(stacked_rows)
        stacked_demo_df["Global_Rank"] = stats.rankdata(stacked_demo_df["Score"].values, method="average")
        
        sample_slice = pd.concat([
            stacked_demo_df.iloc[:3],
            stacked_demo_df.iloc[20:23],
            stacked_demo_df.iloc[40:43]
        ]).reset_index(drop=True)
        sample_slice["Excel_Row"] = [f"Row {i+2}" for i in range(len(sample_slice))]
        sample_slice = sample_slice[["Excel_Row", "Subject", "Group", "Score", "Global_Rank"]]

        st.markdown("##### 🏷️ Stacked Global Ranking Preview (Columns A-E: Required for =RANK.AVG in Excel)")
        render_table(sample_slice, height=240)

        st.markdown("#### 1A. Testing Assumptions in Excel")
        st.code("""
Descriptive Metrics per Group:
Group Medians:                    =MEDIAN(B2:B21)       | =MEDIAN(C2:C21)       | =MEDIAN(D2:D21)
Group Q1 (25th percentile):       =QUARTILE.INC(B2:B21, 1)
Group Q3 (75th percentile):       =QUARTILE.INC(B2:B21, 3)
Interquartile Range (IQR):        =Q3 - Q1
Group Sample Sizes:               =COUNT(B2:B21)        | =COUNT(C2:C21)        | =COUNT(D2:D21)
Group Skewness:                   =SKEW(B2:B21)         | =SKEW(C2:C21)         | =SKEW(D2:D21)

Minimum Sample Size Check (n_k >= 5):
Smallest Group Count:             =MIN(COUNT(B2:B21), COUNT(C2:C21), COUNT(D2:D21))
Asymptotic Validity:              =IF(Smallest_Count >= 5, "PASS (Chi-Square Valid)", "FAIL (Permutation Required)")
""", language="excel")

        st.markdown(r"""
        **🔍 How to Interpret the Assumption Output:**
        * **Sample Size Rule ($n_k \ge 5$):** If every group has at least 5 observations, the Chi-Square distribution reliably models $H$. If $n_k < 5$, use exact Kruskal-Wallis permutation tables.
        * **Shape Similarity:** When group skewness and dispersion are roughly comparable, rejection of $H_0$ directly indicates that **group medians differ**. When shapes differ significantly, rejection indicates **stochastic dominance / general shift in rank distributions**.
        """)

    # STEP 2: IMPLEMENT THE PROPOSED TEST IN EXCEL
    with st.expander("📌 Step 2: Implement Kruskal-Wallis H Test in Excel (Stacked Ranking & Formulas)", expanded=False):
        st.markdown(r"""
        Because Microsoft Excel's Data Analysis Toolpak does **not** include an automated Kruskal-Wallis tool, 
        the test is performed transparently using native worksheet formulas in **two simple phases**:
        1. **Stack Data and Assign Global Mid-Ranks** using `=RANK.AVG()`.
        2. **Compute Group Rank Sums and the $H$ Statistic**.
        """)

        st.markdown("#### Phase 1: Stack Data and Assign Global Ranks")
        st.markdown("""
        1. Create a stacked table where **Column A** is Subject ID, **Column B** is Group Name, and **Column C** is Score (`C2:C61` for $N=60$).
        2. In **Column D** (Row 2), compute the global mid-rank with ascending order (`order = 1`):
        """)
        st.code("""
Global Mid-Rank Formula (Cell D2):
=RANK.AVG(C2, $C$2:$C$61, 1)

(Drag this formula down from D2 to D61. Tied values automatically receive their average rank).
""", language="excel")

        st.markdown("#### Phase 2: Compute Group Rank Sums, H-Statistic & p-Value")
        st.code("""
Group Sample Sizes (Column B = Group, Column D = Global Rank):
n1 (Standard Care):               =COUNTIF($B$2:$B$61, "Standard Care")          [= 20]
n2 (Treatment Low):               =COUNTIF($B$2:$B$61, "Treatment Low")          [= 20]
n3 (Treatment High):              =COUNTIF($B$2:$B$61, "Treatment High")         [= 20]
Total Sample Size (N):            =SUM(n1_cell, n2_cell, n3_cell)                [= 60]
Number of Groups (k):             =3
Degrees of Freedom (df):          =k - 1                                         [= 2]

Group Rank Sums:
R1 (Standard Care Rank Sum):      =SUMIF($B$2:$B$61, "Standard Care", $D$2:$D$61)
R2 (Treatment Low Rank Sum):      =SUMIF($B$2:$B$61, "Treatment Low", $D$2:$D$61)
R3 (Treatment High Rank Sum):     =SUMIF($B$2:$B$61, "Treatment High", $D$2:$D$61)

Group Mean Ranks:
Mean Rank 1 (R̄1):                 =R1_cell / n1_cell
Mean Rank 2 (R̄2):                 =R2_cell / n2_cell
Mean Rank 3 (R̄3):                 =R3_cell / n3_cell
Expected Mean Rank under H0:      =(N + 1) / 2                                   [= 30.5]

Kruskal-Wallis H Statistic:
H = (12 / (N * (N + 1))) * ((R1^2 / n1) + (R2^2 / n2) + (R3^2 / n3)) - 3 * (N + 1)
In Excel Formula Syntax:
=(12 / (N_cell * (N_cell + 1))) * ((R1_cell^2 / n1_cell) + (R2_cell^2 / n2_cell) + (R3_cell^2 / n3_cell)) - 3 * (N_cell + 1)

Omnibus p-value:
=CHISQ.DIST.RT(H_cell, df_cell)

Critical Chi-Square Value (alpha = 0.05):
=CHISQ.INV.RT(0.05, df_cell)

Effect Size Epsilon-Squared (ε_R²):
=MAX(0, (H_cell - (k - 1)) / (N_cell - k))

Post-Hoc Dunn-Bonferroni Comparisons (Pairwise Rank Differences):
Standard Error of Difference (SE_ij):
=SQRT((N_cell * (N_cell + 1) / 12) * (1 / n1_cell + 1 / n2_cell))

Z-Statistic (Care vs Low Dose):
=ABS(MeanRank1_cell - MeanRank2_cell) / SE_12_cell

Bonferroni-Adjusted p-value (3 comparisons):
=MIN(1.0, 3 * (2 * (1 - NORM.S.DIST(Z_cell, TRUE))))
""", language="excel")

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output in Plain English:**
        * **What is Dunn's Post-Hoc Test?** The omnibus $H$-test only tells you *that* a difference exists somewhere among the groups. **Dunn's post-hoc test** compares each specific pair of groups to identify exactly which treatments differ in their rank ordering.
        * **Why Can't We Just Run Multiple Mann-Whitney Tests?** Running 3 separate unadjusted rank tests increases your combined false positive risk to **14.3%**. Multiplying each pairwise p-value by 3 (`=MIN(1.0, 3 * ...)` via the Bonferroni method) mathematically ensures that the overall false alarm rate across all 3 comparisons remains capped at **5%**.
        * **Interpreting the Pairwise Findings:**
          * If adjusted $p < 0.05$, that specific pair of groups is **statistically significantly different**. The group with the higher average rank scored systematically higher.
          * If adjusted $p \ge 0.05$, there is **no statistically significant difference** between those two groups—any observed gap in ranks is consistent with random chance.
        * **Epsilon-Squared ($\epsilon_R^2$):** Proportion of variance in the ranked outcome accounted for by group membership:
          * $\epsilon_R^2 < 0.06$: Small effect
          * $0.06 \le \epsilon_R^2 < 0.14$: Medium effect
          * $\epsilon_R^2 \ge 0.14$: Large effect
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Tests in Excel (When Parametric Conditions are Met)", expanded=False):
        st.markdown(r"""
        When should you choose an alternative test?
        * **If Data is Continuous, Normal & Homoscedastic:** Switch to **One-Way ANOVA** for ~5% higher statistical power.
        * **If Data is Normal but Heteroscedastic ($F_{\max} > 4.0$):** Switch to **Welch's ANOVA**.
        * **If Distribution Shapes Differ Radically & Pure Median Equality is Required:** Switch to **Mood's Median Test**.
        """)

        st.markdown("#### Option 3A: One-Way ANOVA in Excel (When Normality Holds)")
        st.markdown("""
        If residuals within each group are normally distributed and group sample sizes are moderate-to-large ($n_k \ge 30$), One-Way ANOVA is more powerful than rank tests:
        1. Click **Data** ➔ **Data Analysis** ➔ **Anova: Single Factor**.
        2. Input unstacked columns (`$B$1:$D$21`), select **Grouped By: Columns**, check **Labels in first row**.
        3. Click **OK**.
        """)

        st.markdown("#### Option 3B: Welch's ANOVA in Excel (Unequal Variances)")
        st.code("""
Pairwise Welch t-tests in Excel with Bonferroni correction:
Care vs Low Dose (Type=3, Unequal Variance):  =MIN(1.0, 3 * T.TEST(B2:B21, C2:C21, 2, 3))
Care vs High Dose (Type=3, Unequal Variance): =MIN(1.0, 3 * T.TEST(B2:B21, D2:D21, 2, 3))
Low vs High Dose (Type=3, Unequal Variance):  =MIN(1.0, 3 * T.TEST(C2:C21, D2:D21, 2, 3))
""", language="excel")


def render_kruskal_wallis_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python script guide for Kruskal-Wallis H Test."""
    st.markdown("""
    ### 🐍 Python Code Implementation Workflow
    This script is divided into 3 modular sections: **Evaluate Assumptions** ➔ **Execute Proposed Test** ➔ **Execute Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Sample Sizes, Shape & Normality)", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

# Given 3 independent group arrays: g1, g2, g3
groups = [g1, g2, g3]
group_names = ["Standard Care", "Treatment Low Dose", "Treatment High Dose"]

# 1. Sample Size Check per group (n_k >= 5)
for name, g in zip(group_names, groups):
    n = len(g)
    print(f"[Assumption] Sample Size ({name}): n = {n} -> {'PASS' if n >= 15 else ('BORDERLINE' if n >= 5 else 'FAIL')}")

# 2. Distribution Shape & Homoscedasticity Check (Levene's Test)
lev_stat, lev_p = stats.levene(*groups, center='median')
skews = [stats.skew(g) for g in groups]
max_skew_diff = max(skews) - min(skews)
shape_similar = lev_p >= 0.05 and max_skew_diff <= 1.0
print(f"[Assumption] Shape Similarity: Levene p = {lev_p:.4f}, Max Skew Diff = {max_skew_diff:.2f}")
print(f"             Interpretation: {'Valid test of Medians' if shape_similar else 'Valid test of Stochastic Dominance'}")

# 3. Normality Check (To justify non-parametric vs ANOVA)
shapiro_results = [stats.shapiro(g) for g in groups]
for name, (w, p) in zip(group_names, shapiro_results):
    print(f"[Assumption] Normality ({name}): Shapiro W = {w:.4f}, p = {p:.4f} -> {'NON-NORMAL (Rank Test Justified)' if p < 0.05 else 'Normal'}")
""", language="python")

    # STEP 2: EXECUTE PROPOSED TEST IN PYTHON
    with st.expander("📌 Step 2: Execute Kruskal-Wallis H Test and Post-Hoc Dunn's Test in Python", expanded=False):
        st.code("""# 1. Omnibus Kruskal-Wallis H Test
h_stat, p_val = stats.kruskal(g1, g2, g3)
k = len(groups)
n_total = sum(len(g) for g in groups)
df = k - 1

# Epsilon-squared effect size (Tomczak & Tomczak, 2014)
epsilon_sq = max(0.0, (h_stat - k + 1) / (n_total - k))

print(f"Kruskal-Wallis H({df}) = {h_stat:.4f}, p-value = {p_val:.4e}")
print(f"Epsilon-Squared (ε_R²): {epsilon_sq:.3f}")

# 2. Post-Hoc Pairwise Comparisons (Dunn's Test with Bonferroni Correction)
if p_val < 0.05:
    all_scores = np.concatenate(groups)
    all_ranks = stats.rankdata(all_scores, method='average')
    
    # Mid-rank sums and means
    idx = 0
    mean_ranks = []
    for g in groups:
        r = all_ranks[idx:idx+len(g)]
        mean_ranks.append(np.mean(r))
        idx += len(g)
    
    # Tie correction for standard error
    _, tie_counts = np.unique(all_scores, return_counts=True)
    tie_sum = np.sum(tie_counts**3 - tie_counts)
    sigma_0_sq = (n_total * (n_total + 1) / 12.0) - (tie_sum / (12.0 * (n_total - 1)))
    sigma_0 = np.sqrt(max(0.0001, sigma_0_sq))
    
    pairs = [(0, 1), (0, 2), (1, 2)]
    num_comparisons = len(pairs)
    print("\\nPairwise Post-Hoc Dunn-Bonferroni Results:")
    for i, j in pairs:
        ni, nj = len(groups[i]), len(groups[j])
        se = sigma_0 * np.sqrt(1.0 / ni + 1.0 / nj)
        diff_rank = mean_ranks[j] - mean_ranks[i]
        z = abs(diff_rank) / se
        p_raw = 2.0 * (1.0 - stats.norm.cdf(z))
        p_adj = min(1.0, p_raw * num_comparisons)
        sig = "Significant (*)" if p_adj < 0.05 else "Not Significant"
        print(f"  {group_names[j]} vs {group_names[i]}: Diff = {diff_rank:+.2f}, z = {z:.3f}, p_adj = {p_adj:.4e} -> {sig}")
""", language="python")

    # STEP 3: EXECUTE ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Execute Alternative Tests in Python (One-Way ANOVA & Welch's ANOVA)", expanded=False):
        st.code("""# Alternative A: One-Way ANOVA (Parametric Baseline)
f_stat, p_anova = stats.f_oneway(g1, g2, g3)
print(f"One-Way ANOVA: F = {f_stat:.4f}, p-value = {p_anova:.4e}")

# Alternative B: Pairwise Welch's t-tests with Bonferroni correction
for i, j in [(0, 1), (0, 2), (1, 2)]:
    res = stats.ttest_ind(groups[i], groups[j], equal_var=False)
    p_adj = min(1.0, res.pvalue * 3)
    print(f"Welch t-test ({group_names[j]} vs {group_names[i]}): t = {res.statistic:.3f}, p_adj = {p_adj:.4e}")
""", language="python")

    # COPY CONSOLIDATED PIPELINE
    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any

def run_kruskal_wallis_pipeline(groups_dict: Dict[str, List[float]], alpha: float = 0.05) -> Dict[str, Any]:
    \"\"\"Executes a complete 3-step Kruskal-Wallis pipeline with diagnostics and post-hoc tests.\"\"\"
    group_names = list(groups_dict.keys())
    groups = [np.asarray(groups_dict[k], dtype=float) for k in group_names]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    df = k - 1
    
    print("=" * 65)
    print("STEP 1: ASSUMPTION DIAGNOSTICS")
    print("=" * 65)
    
    # Sample sizes
    for name, g in zip(group_names, groups):
        n = len(g)
        status = "PASS" if n >= 15 else ("BORDERLINE" if n >= 5 else "FAIL")
        print(f"Sample Size ({name}): n = {n} -> {status}")
        
    # Shape similarity via Levene's test
    lev_stat, lev_p = stats.levene(*groups, center='median')
    skews = [stats.skew(g) for g in groups]
    max_skew_diff = max(skews) - min(skews)
    shape_similar = lev_p >= 0.05 and max_skew_diff <= 1.0
    print(f"Shape Similarity: Levene p = {lev_p:.4f}, Max Skew Diff = {max_skew_diff:.2f}")
    print(f"                  -> {'Test of Medians' if shape_similar else 'Test of Stochastic Dominance'}")
    
    print("\\n" + "=" * 65)
    print("STEP 2: KRUSKAL-WALLIS H TEST & POST-HOC DUNN")
    print("=" * 65)
    
    h_stat, p_val = stats.kruskal(*groups)
    epsilon_sq = max(0.0, (h_stat - k + 1) / (n_total - k))
    
    print(f"Kruskal-Wallis H({df}) = {h_stat:.4f}, p-value = {p_val:.4e}")
    print(f"Epsilon-Squared (ε_R²): {epsilon_sq:.3f}")
    
    # Global ranking and pairwise comparisons
    all_scores = np.concatenate(groups)
    all_ranks = stats.rankdata(all_scores, method='average')
    idx = 0
    mean_ranks = []
    for g in groups:
        mean_ranks.append(float(np.mean(all_ranks[idx:idx+len(g)])))
        idx += len(g)
        
    _, tie_counts = np.unique(all_scores, return_counts=True)
    tie_sum = np.sum(tie_counts**3 - tie_counts)
    sigma_0_sq = (n_total * (n_total + 1) / 12.0) - (tie_sum / (12.0 * (n_total - 1)))
    sigma_0 = np.sqrt(max(0.0001, sigma_0_sq))
    
    pairwise_results = []
    pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
    num_comparisons = len(pairs)
    
    if p_val < alpha:
        print("\\nPairwise Dunn-Bonferroni Comparisons:")
        for i, j in pairs:
            ni, nj = len(groups[i]), len(groups[j])
            se = sigma_0 * np.sqrt(1.0 / ni + 1.0 / nj)
            diff_rank = mean_ranks[j] - mean_ranks[i]
            z = abs(diff_rank) / se
            p_raw = 2.0 * (1.0 - stats.norm.cdf(z))
            p_adj = min(1.0, float(p_raw * num_comparisons))
            sig = p_adj < alpha
            pairwise_results.append({
                "comparison": f"{group_names[j]} vs {group_names[i]}",
                "diff_rank": diff_rank,
                "z_stat": z,
                "p_adj": p_adj,
                "significant": sig
            })
            print(f"  {group_names[j]} vs {group_names[i]}: Diff = {diff_rank:+.2f}, z = {z:.3f}, p_adj = {p_adj:.4e} -> {'SIGNIFICANT' if sig else 'Not sig'}")
    else:
        print("\\nOmnibus test not significant; pairwise comparisons not indicated.")
        
    print("=" * 65)
    return {
        "h_stat": float(h_stat),
        "p_val": float(p_val),
        "epsilon_squared": float(epsilon_sq),
        "mean_ranks": mean_ranks,
        "pairwise": pairwise_results
    }

# Example Run:
# run_kruskal_wallis_pipeline({
#     "Standard Care": [4.0, 5.0, 6.0, 4.0, 5.0, 7.0, 3.0, 5.0],
#     "Treatment Low": [5.0, 6.0, 7.0, 6.0, 7.0, 8.0, 6.0, 6.0],
#     "Treatment High": [7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 8.0, 9.0]
# })
""", language="python")


def render_repeated_measures_anova_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for Repeated Measures ANOVA."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Normality of Differences, Sphericity & Outliers)", expanded=False):
        st.markdown(
            "Before interpreting a Repeated Measures ANOVA, verify that: (1) each participant has complete observations "
            "across all conditions, (2) pairwise differences between conditions are approximately normally distributed (or N >= 30), "
            "(3) sphericity is satisfied (variances of pairwise differences are approximately equal), and (4) there are no extreme "
            "within-subject difference outliers."
        )

        # 1. Generate deterministic sample data
        rng = np.random.default_rng(42)
        n_excel = 20
        subj_offset = rng.normal(0.0, 9.0, n_excel)
        base_vals = np.round(55.0 + subj_offset + rng.normal(0.0, 3.5, n_excel), 2)
        m1_vals = np.round(62.0 + subj_offset + rng.normal(0.0, 3.5, n_excel), 2)
        m3_vals = np.round(71.0 + subj_offset + rng.normal(0.0, 3.5, n_excel), 2)

        diff_m1_base = np.round(m1_vals - base_vals, 2)
        diff_m3_base = np.round(m3_vals - base_vals, 2)
        diff_m3_m1 = np.round(m3_vals - m1_vals, 2)

        # Outlier detection on primary difference (Month 3 - Baseline)
        q25 = float(np.percentile(diff_m3_base, 25))
        q75 = float(np.percentile(diff_m3_base, 75))
        iqr = q75 - q25
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr

        outlier_flags = np.where(
            (diff_m3_base < lower_fence) | (diff_m3_base > upper_fence),
            "OUTLIER",
            "OK"
        )

        example_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(n_excel)],
            "Subject_ID": [f"Subj_{i+1:02d}" for i in range(n_excel)],
            "Baseline (Col B)": base_vals,
            "Month_1 (Col C)": m1_vals,
            "Month_3 (Col D)": m3_vals,
            "Diff_M1_Base (Col E)": diff_m1_base,
            "Diff_M3_Base (Col F)": diff_m3_base,
            "Diff_M3_M1 (Col G)": diff_m3_m1,
            "Outlier_Check": outlier_flags
        })

        # Dynamic diagnostics
        n_count = len(diff_m3_base)
        outlier_count = int(np.sum(outlier_flags == "OUTLIER"))
        skew_diff = float(stats.skew(diff_m3_base))
        kurt_diff = float(stats.kurtosis(diff_m3_base))

        var_e = float(np.var(diff_m1_base, ddof=1))
        var_f = float(np.var(diff_m3_base, ddof=1))
        var_g = float(np.var(diff_m3_m1, ddof=1))
        spher_ratio = max(var_e, var_f, var_g) / max(0.001, min(var_e, var_f, var_g))

        n_pass = "PASS" if n_count >= 15 else "WARN"
        outlier_pass = "PASS" if outlier_count == 0 else "FAIL"
        skew_pass = "PASS" if abs(skew_diff) <= 0.6 else "WARN"
        kurt_pass = "PASS" if abs(kurt_diff) <= 1.0 else "WARN"
        spher_pass = "PASS" if spher_ratio <= 3.0 else ("WARN" if spher_ratio <= 4.0 else "FAIL")

        summary_df = pd.DataFrame({
            "Assumption_Metric": [
                "Matched Sample Size (N)",
                "Extreme Difference Outliers (>1.5x IQR)",
                "Difference Skewness (M3 - Base)",
                "Difference Excess Kurtosis",
                "Sphericity Variance Ratio (Max/Min Var)"
            ],
            "Value": [
                f"{n_count}",
                f"{outlier_count}",
                f"{skew_diff:.3f}",
                f"{kurt_diff:.3f}",
                f"{spher_ratio:.2f}"
            ],
            "Rule_Threshold": [
                "N >= 15 matched subjects",
                "0 Extreme Outliers",
                "Between -0.6 and +0.6",
                "Between -1.0 and +1.0",
                "Ratio <= 3.0 (Equal variances)"
            ],
            "Assumption_Result": [
                n_pass,
                outlier_pass,
                skew_pass,
                kurt_pass,
                spher_pass
            ]
        })

        st.markdown("##### 📋 Example Spreadsheet Data (Range: A2:H21)")
        render_table(example_df, height=240)

        st.markdown("##### 📊 Assumption Diagnostic Summary")
        render_table(summary_df, height=240)

        st.caption(
            "Excel Formulas: Diff: `=D2-B2` | Variance: `=VAR.S(F2:F21)` | "
            "Skew: `=SKEW(F2:F21)` | Outlier Q1: `=PERCENTILE.INC(F2:F21, 0.25)` | Q3: `=PERCENTILE.INC(F2:F21, 0.75)`"
        )

    # STEP 2: IMPLEMENT REPEATED MEASURES ANOVA IN EXCEL
    with st.expander("📌 Step 2: Implement Repeated Measures ANOVA & Post-Hoc Tests in Excel", expanded=False):
        st.markdown("""
        #### 🛠️ Method A: Data Analysis Toolpak (Recommended — Fastest & Automated)
        Excel's Data Analysis Toolpak executes Repeated Measures ANOVA under the name **"Anova: Two-Factor Without Replication"**:
        1. Click **Data Tab ➔ Data Analysis ➔ Anova: Two-Factor Without Replication**.
        2. Set **Input Range** to `B1:D21` (encompassing your 3 condition columns across all 20 subject rows).
        3. Check **Labels** (if row 1 contains headers). Set **Alpha** to `0.05`.
        4. Select an output range and click **OK**.
        
        **🔍 How to Read the Toolpak Output Table:**
        * **Rows (Between Subjects):** Quantifies variance attributable to individual baseline differences between participants ($SS_{\\text{subjects}}$, $df = n - 1 = 19$). Excel isolates and partitions this variation out of the error term!
        * **Columns (Within Subjects / Conditions):** **This is your primary hypothesis test!**
          * $SS_{\\text{columns}}$: Sum of Squares across conditions ($SS_{\\text{time}}$).
          * $df = k - 1 = 2$: Degrees of freedom across conditions.
          * $MS_{\\text{columns}}$: Mean Square ($SS / df$).
          * **F:** Omnibus F-statistic ($MS_{\\text{columns}} / MS_{\\text{error}}$).
          * **P-value:** Tests whether mean scores change significantly across repeated time points.
          * **F crit:** Critical threshold for statistical significance ($\approx 3.25$ at $\\alpha = 0.05$).
        * **Error (Residual):** Within-subject random error ($SS_{\\text{error}}$, $df = (n-1)(k-1) = 38$, $MS_{\\text{error}}$).
        
        ---
        #### 📐 Method B: Native Formulas (Without Toolpak)
        If Toolpak is not installed, calculate sums of squares directly:
        """)
        st.code("""
Total Observations:         =COUNT(B2:D21)                         [60]
Grand Mean:                 =AVERAGE(B2:D21)
SS Total:                   =DEVSQ(B2:D21)

SS Subjects (Rows):         =3 * DEVSQ(H2:H21)                     [Compute row averages in Col H first: =AVERAGE(B2:D2)]
df Subjects:                =COUNT(B2:B21) - 1                     [19]

SS Time / Conditions (Cols):=20 * ((AVERAGE(B2:B21)-grand_mean)^2 + (AVERAGE(C2:C21)-grand_mean)^2 + (AVERAGE(D2:D21)-grand_mean)^2)
df Time:                    =3 - 1                                 [2]
MS Time:                    =ss_time / 2

SS Error (Residual):        =ss_total - ss_subjects - ss_time
df Error:                   =(COUNT(B2:B21) - 1) * (3 - 1)         [38]
MS Error:                   =ss_error / 38

Omnibus F-Statistic:        =ms_time / ms_error
Two-Tailed p-value:         =F.DIST.RT(f_stat, 2, 38)
Partial Eta-Squared (ηp²):  =ss_time / (ss_time + ss_error)
""", language="excel")

        st.markdown("""
        ---
        #### 🔍 Post-Hoc Pairwise Comparisons in Excel (Bonferroni-Adjusted Paired t-tests)
        If the omnibus ANOVA above is significant ($p < 0.05$), it tells you that scores changed across time, but **not between which specific time points**. 
        
        To find out exactly where the differences lie, conduct **pairwise paired t-tests** between all pairs with a **Bonferroni correction**:
        """)
        st.code("""
Pairwise Comparison 1 (Month 1 vs Baseline):
Raw Paired t-test p-value:  =T.TEST(C2:C21, B2:B21, 2, 1)        [4th argument '1' = Paired!]
Bonferroni-Adjusted p:      =MIN(1.0, 3 * T.TEST(C2:C21, B2:B21, 2, 1))

Pairwise Comparison 2 (Month 3 vs Baseline):
Raw Paired t-test p-value:  =T.TEST(D2:D21, B2:B21, 2, 1)
Bonferroni-Adjusted p:      =MIN(1.0, 3 * T.TEST(D2:D21, B2:B21, 2, 1))

Pairwise Comparison 3 (Month 3 vs Month 1):
Raw Paired t-test p-value:  =T.TEST(D2:D21, C2:C21, 2, 1)
Bonferroni-Adjusted p:      =MIN(1.0, 3 * T.TEST(D2:D21, C2:C21, 2, 1))
""", language="excel")

        st.info("""
        💡 **Plain English Guide: Why Multiply by 3 (Family-Wise Error Rate)?**
        * Running multiple individual tests without adjustment multiplies your false positive risk. At $\\alpha = 0.05$, running 3 comparisons produces a compound false alarm risk of $1 - (1 - 0.05)^3 \\approx 14.3\\%$.
        * Multiplying each raw $p$-value by the number of comparisons ($k = 3$) caps the cumulative family-wise false positive risk strictly at $5\\%$.
        * **Interpretation Rule:** If the Bonferroni-adjusted $p$-value is $< 0.05$, conclude that the difference between those two specific time points is statistically significant.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Tests in Excel (Greenhouse-Geisser or Friedman Test)", expanded=False):
        st.markdown("""
        #### 🛡️ Alternative A: Greenhouse-Geisser Sphericity Correction
        When difference variances are unequal (Mauchly $p < 0.05$ or variance ratio $> 3.0$), unadjusted degrees of freedom produce too many false alarms.
        
        Adjust degrees of freedom using the Greenhouse-Geisser estimate $\\hat{\\epsilon}$ (typically between $0.50$ and $0.85$):
        """)
        st.code("""
GG Adjusted df1:            =2 * eps_cell                          [e.g. 2 * 0.72 = 1.44]
GG Adjusted df2:            =38 * eps_cell                         [e.g. 38 * 0.72 = 27.36]
Greenhouse-Geisser p-value: =F.DIST.RT(f_stat, df1_adj, df2_adj)
""", language="excel")

        st.markdown("""
        ---
        #### 🛡️ Alternative B: Non-Parametric Friedman Test (Ordinal / Skewed Data)
        If differences are heavily skewed or data is ordinal (e.g. Likert ratings), use the within-subject rank-based **Friedman Test**:
        1. Across each subject row (Row 2 to 21), rank the 3 scores from 1 to 3 using `=RANK.AVG(B2, $B2:$D2, 1)` in Columns I, J, K.
        2. Sum the ranks for each column: $R_1 = \\text{SUM}(I2:I21)$, $R_2 = \\text{SUM}(J2:J21)$, $R_3 = \\text{SUM}(K2:K21)$.
        3. Calculate the Friedman Chi-Square test statistic $\\chi_F^2$:
        """)
        st.code("""
Friedman Chi-Square:        =(12 / (20 * 3 * 4)) * (R1^2 + R2^2 + R3^2) - (3 * 20 * 4)
Friedman p-value:           =CHISQ.DIST.RT(chi_sq_cell, 2)
""", language="excel")


def render_repeated_measures_anova_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python code guide for Repeated Measures ANOVA."""
    st.markdown("""
    ### 🐍 Python Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Normality of Differences, Sphericity & Outliers)", expanded=False):
        st.code("""import numpy as np
import pandas as pd
import scipy.stats as stats

# Example wide-format data (rows = subjects, columns = repeated occasions)
# Y has shape (n_subjects, 3)
n = len(Y)
diff_10 = Y[:, 1] - Y[:, 0]  # Month 1 - Baseline
diff_20 = Y[:, 2] - Y[:, 0]  # Month 3 - Baseline
diff_21 = Y[:, 2] - Y[:, 1]  # Month 3 - Month 1

# 1. Normality of Pairwise Differences (Shapiro-Wilk)
p_shap_10 = stats.shapiro(diff_10).pvalue
p_shap_20 = stats.shapiro(diff_20).pvalue
p_shap_21 = stats.shapiro(diff_21).pvalue

# 2. Sphericity Analysis (Mauchly's Test & Greenhouse-Geisser Epsilon)
# Contrast matrix C for k=3
C = np.array([
    [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
    [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
    [0.0, -2.0 / np.sqrt(6.0)]
])
S = np.cov(Y, rowvar=False)
SC = C.T @ S @ C
eigvals = np.maximum(np.linalg.eigvalsh(SC), 1e-10)
tr_sc = np.sum(eigvals)
det_sc = np.prod(eigvals)
w_mauchly = det_sc / ((tr_sc / 2.0) ** 2)
d_factor = 1.0 - 1.0 / (n - 1)
chi2_mauchly = -(n - 1) * d_factor * np.log(w_mauchly)
p_mauchly = 1.0 - stats.chi2.cdf(chi2_mauchly, df=2)
eps_gg = (tr_sc ** 2) / (2.0 * np.sum(eigvals ** 2))

# 3. Outlier Detection on Difference Scores (> 3x IQR)
def count_ext_outliers(d):
    q25, q75 = np.percentile(d, [25, 75])
    iqr = q75 - q25
    return len(d[(d < q25 - 3 * iqr) | (d > q75 + 3 * iqr)])

total_outliers = sum(count_ext_outliers(d) for d in [diff_10, diff_20, diff_21])

print(f"Normality: Diff 1-0 p={p_shap_10:.4f}, Diff 2-0 p={p_shap_20:.4f}, Diff 2-1 p={p_shap_21:.4f}")
print(f"Mauchly's Sphericity: W = {w_mauchly:.4f}, chi2 = {chi2_mauchly:.3f}, p = {p_mauchly:.4f}")
print(f"Greenhouse-Geisser Epsilon: eps = {eps_gg:.4f} (Threshold: eps >= 0.75)")
print(f"Extreme Difference Outliers: {total_outliers}")
""", language="python")

    # STEP 2: IMPLEMENT REPEATED MEASURES ANOVA IN PYTHON
    with st.expander("📌 Step 2: Implement Repeated Measures ANOVA & Post-Hoc Tests in Python", expanded=False):
        st.code("""from statsmodels.stats.anova import AnovaRM

# Long-format DataFrame required for statsmodels AnovaRM:
# Columns: ['Subject_ID', 'Time_Point', 'Score']
aov = AnovaRM(df_long, depvar='Score', subject='Subject_ID', within=['Time_Point'])
res = aov.fit()
print("=== Repeated Measures ANOVA (statsmodels) ===")
print(res.summary())

# Extract ANOVA metrics
f_stat = res.anova_table['F Value'][0]
df_num = res.anova_table['Num DF'][0]
df_den = res.anova_table['Den DF'][0]
p_val = res.anova_table['Pr > F'][0]

# --- POST-HOC PAIRWISE PAIRED T-TESTS WITH BONFERRONI CORRECTION ---
# If omnibus p < 0.05, isolate which pairs differ:
pairs = [
    ("Month 1", "Baseline", Y[:, 1], Y[:, 0]),
    ("Month 3", "Baseline", Y[:, 2], Y[:, 0]),
    ("Month 3", "Month 1", Y[:, 2], Y[:, 1])
]
num_comparisons = len(pairs)
t_crit = stats.t.ppf(1.0 - 0.05 / (2.0 * num_comparisons), df=n - 1)

print("\\n=== Post-Hoc Pairwise Paired t-tests (Bonferroni Corrected) ===")
for name_b, name_a, y_b, y_a in pairs:
    diff = y_b - y_a
    mean_d = np.mean(diff)
    se_d = np.std(diff, ddof=1) / np.sqrt(n)
    t_pair = mean_d / se_d
    p_raw = 2.0 * (1.0 - stats.t.cdf(abs(t_pair), df=n - 1))
    p_adj = min(1.0, p_raw * num_comparisons)
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d
    sig = p_adj < 0.05
    print(f"{name_b} vs {name_a}: Mean Diff = {mean_d:+.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}], p_adj = {p_adj:.4f} -> {'SIGNIFICANT' if sig else 'No Diff'}")
""", language="python")

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Tests in Python (Greenhouse-Geisser / Friedman Test)", expanded=False):
        st.code("""# Alternative A: Greenhouse-Geisser Sphericity Correction
# Adjust degrees of freedom by multiplying by eps_gg
df_num_gg = df_num * eps_gg
df_den_gg = df_den * eps_gg
p_val_gg = 1.0 - stats.f.cdf(f_stat, df_num_gg, df_den_gg)

print("=== Alternative A: Greenhouse-Geisser Corrected RM-ANOVA ===")
print(f"Adjusted df: ({df_num_gg:.2f}, {df_den_gg:.2f})")
print(f"F({df_num_gg:.2f}, {df_den_gg:.2f}) = {f_stat:.3f}, Greenhouse-Geisser p-value = {p_val_gg:.4e}")

# Alternative B: Friedman Test (Rank-based non-parametric alternative)
# Takes arrays for each repeated condition:
friedman_stat, p_friedman = stats.friedmanchisquare(Y[:, 0], Y[:, 1], Y[:, 2])
print("\\n=== Alternative B: Friedman Test (Non-parametric) ===")
print(f"Friedman Chi-Square(2) = {friedman_stat:.3f}, p-value = {p_friedman:.4e}")
""", language="python")

    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

def run_repeated_measures_anova_pipeline(Y, time_names=("Baseline", "Month 1", "Month 3"), alpha=0.05):
    \"\"\"Complete Repeated Measures ANOVA pipeline with sphericity check, post-hoc paired tests, and Friedman alternative.
    
    Y: 2D numpy array of shape (n_subjects, 3) where columns are repeated occasions.
    \"\"\"
    n, k = Y.shape
    print(f"=== REPEATED MEASURES ANOVA PIPELINE (N = {n} Subjects, k = {k} Conditions) ===\\n")
    
    # 1. Sum of Squares Partitioning
    grand_mean = np.mean(Y)
    subj_means = np.mean(Y, axis=1)
    col_means = np.mean(Y, axis=0)
    
    ss_total = np.sum((Y - grand_mean) ** 2)
    ss_subj = k * np.sum((subj_means - grand_mean) ** 2)
    ss_time = n * np.sum((col_means - grand_mean) ** 2)
    ss_err = ss_total - ss_subj - ss_time
    
    df_time = k - 1
    df_err = (n - 1) * (k - 1)
    ms_time = ss_time / df_time
    ms_err = ss_err / df_err
    f_stat = ms_time / ms_err
    p_anova = 1.0 - stats.f.cdf(f_stat, df_time, df_err)
    eta_p2 = ss_time / (ss_time + ss_err)
    
    # 2. Sphericity (Mauchly & Greenhouse-Geisser)
    C = np.array([
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
        [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
        [0.0, -2.0 / np.sqrt(6.0)]
    ])
    SC = C.T @ np.cov(Y, rowvar=False) @ C
    eigvals = np.maximum(np.linalg.eigvalsh(SC), 1e-10)
    tr_sc = np.sum(eigvals)
    det_sc = np.prod(eigvals)
    w_mauchly = det_sc / ((tr_sc / 2.0) ** 2)
    chi2_m = -(n - 1) * (1.0 - 1.0 / (n - 1)) * np.log(w_mauchly)
    p_mauchly = 1.0 - stats.chi2.cdf(chi2_m, df=2)
    eps_gg = (tr_sc ** 2) / (2.0 * np.sum(eigvals ** 2))
    
    p_gg = 1.0 - stats.f.cdf(f_stat, df_time * eps_gg, df_err * eps_gg)
    
    print(f"Omnibus F({df_time}, {df_err}) = {f_stat:.3f}, p-value = {p_anova:.4e}, Partial Eta2 = {eta_p2:.3f}")
    print(f"Mauchly Sphericity: W = {w_mauchly:.4f}, p = {p_mauchly:.4f}")
    print(f"Greenhouse-Geisser: eps = {eps_gg:.4f}, GG p-value = {p_gg:.4e}")
    
    # 3. Post-Hoc Comparisons
    if p_anova < alpha or p_gg < alpha:
        print("\\n--- Post-Hoc Pairwise Paired t-tests (Bonferroni Corrected) ---")
        pairs = [(1, 0), (2, 0), (2, 1)]
        t_crit = stats.t.ppf(1.0 - alpha / (2.0 * len(pairs)), df=n - 1)
        for j, i in pairs:
            diff = Y[:, j] - Y[:, i]
            mean_d = np.mean(diff)
            se_d = np.std(diff, ddof=1) / np.sqrt(n)
            t_pair = mean_d / se_d
            p_raw = 2.0 * (1.0 - stats.t.cdf(abs(t_pair), df=n - 1))
            p_adj = min(1.0, p_raw * len(pairs))
            ci_l, ci_u = mean_d - t_crit * se_d, mean_d + t_crit * se_d
            sig = p_adj < alpha
            print(f"  {time_names[j]} vs {time_names[i]}: Diff = {mean_d:+.2f}, 95% CI [{ci_l:.2f}, {ci_u:.2f}], p_adj = {p_adj:.4f} -> {'SIGNIFICANT' if sig else 'No Diff'}")
    
    # 4. Non-Parametric Alternative
    fr_stat, p_fr = stats.friedmanchisquare(Y[:, 0], Y[:, 1], Y[:, 2])
    print(f"\\nFriedman Non-Parametric: Chi2 = {fr_stat:.3f}, p = {p_fr:.4e}")
    print("=" * 65)

# Example Run:
# Y_demo = np.random.normal(50, 8, size=(20, 3))
# run_repeated_measures_anova_pipeline(Y_demo)
""", language="python")


def render_friedman_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for the Friedman Test."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Scale, Matched Design & Asymptotic Sample Size)", expanded=False):
        st.markdown(
            "Before interpreting the Friedman Test in Excel, verify that: (1) each participant is evaluated under all 3 conditions "
            "(matched repeated measures design), (2) data is at least ordinal (e.g. Likert ratings 1-7 or continuous scores), "
            "(3) subjects are mutually independent blocks, and (4) sample size satisfies the asymptotic threshold ($N \\ge 10$ matched subjects)."
        )

        # 1. Generate deterministic sample data
        rng = np.random.default_rng(42)
        n_excel = 20
        subj_offset = rng.normal(0.0, 5.0, n_excel)
        raw_c1 = 20.0 + subj_offset + rng.gamma(shape=2.0, scale=2.5, size=n_excel)
        raw_c2 = 23.5 + subj_offset + rng.gamma(shape=2.0, scale=2.5, size=n_excel)
        raw_c3 = 27.0 + subj_offset + rng.gamma(shape=2.0, scale=2.5, size=n_excel)

        # Discretize to 1-7 Likert rating scale
        pooled_min = min(float(np.min(raw_c1)), float(np.min(raw_c2)), float(np.min(raw_c3)))
        pooled_max = max(float(np.max(raw_c1)), float(np.max(raw_c2)), float(np.max(raw_c3)))
        span = pooled_max - pooled_min if pooled_max > pooled_min else 1.0

        c1_vals = np.clip(np.round(1.0 + 6.0 * (raw_c1 - pooled_min) / span), 1.0, 7.0)
        c2_vals = np.clip(np.round(1.0 + 6.0 * (raw_c2 - pooled_min) / span), 1.0, 7.0)
        c3_vals = np.clip(np.round(1.0 + 6.0 * (raw_c3 - pooled_min) / span), 1.0, 7.0)

        # Within-subject ranks
        r1_vals = []
        r2_vals = []
        r3_vals = []
        for i in range(n_excel):
            ranks = stats.rankdata([c1_vals[i], c2_vals[i], c3_vals[i]], method="average")
            r1_vals.append(ranks[0])
            r2_vals.append(ranks[1])
            r3_vals.append(ranks[2])

        example_df = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(n_excel)],
            "Subject_ID": [f"Subj_{i+1:02d}" for i in range(n_excel)],
            "Condition_1 (Col B)": c1_vals,
            "Condition_2 (Col C)": c2_vals,
            "Condition_3 (Col D)": c3_vals,
            "Rank_C1 (Col E)": [f"{r:.1f}" for r in r1_vals],
            "Rank_C2 (Col F)": [f"{r:.1f}" for r in r2_vals],
            "Rank_C3 (Col G)": [f"{r:.1f}" for r in r3_vals]
        })

        # Dynamic diagnostics
        n_count = n_excel
        sum_r1 = sum(r1_vals)
        sum_r2 = sum(r2_vals)
        sum_r3 = sum(r3_vals)
        mean_r1 = sum_r1 / n_count
        mean_r2 = sum_r2 / n_count
        mean_r3 = sum_r3 / n_count

        n_pass = "PASS" if n_count >= 15 else ("WARN" if n_count >= 10 else "FAIL")

        summary_df = pd.DataFrame({
            "Assumption_Metric": [
                "Matched Sample Size (N)",
                "Measurement Scale",
                "Asymptotic Chi-Square Rule (N >= 10)",
                "Expected Mean Rank (Null Baseline)",
                "Observed Mean Ranks (C1 / C2 / C3)"
            ],
            "Value": [
                f"{n_count} subjects",
                "Ordinal Likert (1 to 7)",
                f"N = {n_count} (Exceeds minimum)",
                "2.00 ((k + 1) / 2)",
                f"{mean_r1:.2f} / {mean_r2:.2f} / {mean_r3:.2f}"
            ],
            "Rule_Threshold": [
                "N >= 10 (ideally >= 15)",
                "At least Ordinal",
                "N >= 10",
                "Equal for all groups (2.00)",
                "Differing ranks indicate treatment effect"
            ],
            "Assumption_Result": [
                n_pass,
                "PASS",
                n_pass,
                "PASS",
                "SIGNIFICANT SPREAD" if max(mean_r1, mean_r2, mean_r3) - min(mean_r1, mean_r2, mean_r3) > 0.5 else "SIMILAR"
            ]
        })

        st.markdown("##### 📋 Example Spreadsheet Data (Range: A2:G21)")
        render_table(example_df, height=240)

        st.markdown("##### 📊 Assumption Diagnostic Summary")
        render_table(summary_df, height=240)

        st.caption(
            "Excel Formulas: Within-Row Rank C1: `=RANK.AVG(B2, $B2:$D2, 1)` | "
            "Rank Sum C1: `=SUM(E2:E21)` | Mean Rank C1: `=AVERAGE(E2:E21)`"
        )

    # STEP 2: IMPLEMENT FRIEDMAN TEST IN EXCEL
    with st.expander("📌 Step 2: Implement Friedman Test & Post-Hoc Tests in Excel", expanded=False):
        st.markdown("""
        #### 🛠️ Native Worksheet Formulas (Toolpak does not include Friedman)
        The Friedman Test evaluates differences across 3 or more repeated conditions using within-subject ranks.
        
        1. **Rank within each subject row:** In Columns E, F, G, compute the ascending rank of each condition across row 2 to 21:
           * `E2`: `=RANK.AVG(B2, $B2:$D2, 1)`
           * `F2`: `=RANK.AVG(C2, $B2:$D2, 1)`
           * `G2`: `=RANK.AVG(D2, $B2:$D2, 1)`
        2. **Calculate Rank Sums ($R_j$):**
           * Condition 1 Rank Sum ($R_1$): `=SUM(E2:E21)`
           * Condition 2 Rank Sum ($R_2$): `=SUM(F2:F21)`
           * Condition 3 Rank Sum ($R_3$): `=SUM(G2:G21)`
        3. **Compute the Friedman Chi-Square Test Statistic ($Q$ or $\\chi_F^2$):**
        """)
        st.code("""
Matched Sample Size (N):    =COUNT(B2:B21)                         [20]
Number of Conditions (k):   3

Friedman Chi-Square (Q):    =(12 / (20 * 3 * (3 + 1))) * (SUMSQ(R1, R2, R3)) - (3 * 20 * (3 + 1))
                            [Formula: =(12 / 240) * (R1^2 + R2^2 + R3^2) - 240]

Degrees of Freedom (df):    =3 - 1                                 [2]
Two-Tailed p-value:         =CHISQ.DIST.RT(Q_cell, 2)

Effect Size (Kendall's W):  =Q_cell / (20 * (3 - 1))               [Q / (N * (k - 1))]
""", language="excel")

        st.markdown("""
        ---
        #### 🔍 Post-Hoc Pairwise Comparisons in Excel (Bonferroni-Adjusted Tests)
        If the omnibus Friedman test above is statistically significant ($p < 0.05$), it proves that ranks differ across the 3 conditions, 
        but does **not** specify which pairs differ. Conduct pairwise comparisons with a **Bonferroni correction** ($p_{\\text{adj}} = \\min(1.0, 3 \\times p_{\\text{raw}})$):
        """)
        st.code("""
Pairwise Comparison 1 (Condition 2 vs Condition 1):
Raw Paired Test p-value:    =T.TEST(C2:C21, B2:B21, 2, 1)          [Type 1 = Paired]
Bonferroni-Adjusted p:      =MIN(1.0, 3 * T.TEST(C2:C21, B2:B21, 2, 1))

Pairwise Comparison 2 (Condition 3 vs Condition 1):
Raw Paired Test p-value:    =T.TEST(D2:D21, B2:B21, 2, 1)
Bonferroni-Adjusted p:      =MIN(1.0, 3 * T.TEST(D2:D21, B2:B21, 2, 1))

Pairwise Comparison 3 (Condition 3 vs Condition 2):
Raw Paired Test p-value:    =T.TEST(D2:D21, C2:C21, 2, 1)
Bonferroni-Adjusted p:      =MIN(1.0, 3 * T.TEST(D2:D21, C2:C21, 2, 1))
""", language="excel")

        st.info("""
        💡 **Plain English Guide: Why Multiply by 3 (Family-Wise Error Rate)?**
        * Running multiple individual tests without adjustment multiplies your false positive risk. At $\\alpha = 0.05$, running 3 comparisons produces a compound false alarm risk of $1 - (1 - 0.05)^3 \\approx 14.3\\%$.
        * Multiplying each raw $p$-value by the number of comparisons ($k = 3$) caps the cumulative family-wise false positive risk strictly at $5\\%$.
        * **Interpretation Rule:** If the Bonferroni-adjusted $p$-value is $< 0.05$, conclude that the difference between those two specific conditions is statistically significant.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Tests in Excel (Repeated Measures ANOVA)", expanded=False):
        st.markdown("""
        #### 🛡️ Alternative: Parametric Repeated Measures ANOVA (For Normal Continuous Data)
        If your data is continuous and pairwise differences meet normality and sphericity assumptions, 
        use the parametric **Repeated Measures ANOVA** (Data Analysis Toolpak ➔ **"Anova: Two-Factor Without Replication"**):
        * Repeated Measures ANOVA provides approximately 5% to 15% greater statistical power than the Friedman test when normality holds.
        * However, if data is ordinal (Likert ratings) or heavily skewed, the rank-based Friedman test remains the scientifically valid choice.
        """)


def render_friedman_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python code guide for the Friedman Test."""
    st.markdown("""
    ### 🐍 Python Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Scale, Matched Design & Normality)", expanded=False):
        st.code("""import numpy as np
import pandas as pd
import scipy.stats as stats

# Example wide-format array Y of shape (n_subjects, 3)
n, k = Y.shape
diff_10 = Y[:, 1] - Y[:, 0]
diff_20 = Y[:, 2] - Y[:, 0]
diff_21 = Y[:, 2] - Y[:, 1]

# 1. Normality of Differences (Shapiro-Wilk)
p_shap = [stats.shapiro(d).pvalue for d in [diff_10, diff_20, diff_21]]
min_shap = min(p_shap)

# 2. Asymptotic Sample Size Check (N >= 10)
asymptotic_ok = n >= 10

print(f"Sample Size: N = {n} matched subjects (Asymptotic rule N >= 10: {'PASS' if asymptotic_ok else 'WARN'})")
print(f"Normality on Differences: Min Shapiro p = {min_shap:.4f} -> {'Normal' if min_shap >= 0.05 else 'Non-Normal (Friedman Justified)'}")
""", language="python")

    # STEP 2: IMPLEMENT FRIEDMAN TEST IN PYTHON
    with st.expander("📌 Step 2: Implement Friedman Test & Post-Hoc Tests in Python", expanded=False):
        st.code("""# 1. Friedman Test using scipy.stats
q_stat, p_val = stats.friedmanchisquare(Y[:, 0], Y[:, 1], Y[:, 2])

# 2. Effect Size: Kendall's W (Coefficient of Concordance)
kendalls_w = q_stat / (n * (k - 1))

# 3. Compute Within-Subject Mean Ranks
R = np.zeros_like(Y, dtype=float)
for i in range(n):
    R[i, :] = stats.rankdata(Y[i, :], method="average")
mean_ranks = np.mean(R, axis=0)

print(f"Friedman Chi-Square Q({k - 1}) = {q_stat:.4f}, p-value = {p_val:.4e}")
print(f"Kendall's W (Effect Size) = {kendalls_w:.3f}")
for j in range(k):
    print(f"Condition {j+1} Mean Rank: {mean_ranks[j]:.2f} (Expected Null = 2.00)")

# --- POST-HOC PAIRWISE WILCOXON SIGNED-RANK TESTS (BONFERRONI CORRECTED) ---
if p_val < 0.05:
    print("\\n=== Post-Hoc Pairwise Wilcoxon Signed-Rank Tests ===")
    pairs = [(1, 0), (2, 0), (2, 1)]
    for j, i in pairs:
        res = stats.wilcoxon(Y[:, j], Y[:, i], alternative="two-sided")
        p_adj = min(1.0, res.pvalue * len(pairs))
        sig = p_adj < 0.05
        diff_rank = mean_ranks[j] - mean_ranks[i]
        print(f"Cond {j+1} vs Cond {i+1}: Rank Diff = {diff_rank:+.2f}, Wilcoxon W = {res.statistic:.1f}, p_adj = {p_adj:.4f} -> {'SIGNIFICANT' if sig else 'No Diff'}")
""", language="python")

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Tests in Python (Repeated Measures ANOVA)", expanded=False):
        st.code("""# Parametric Repeated Measures ANOVA (when normality & sphericity hold)
from statsmodels.stats.anova import AnovaRM

# Prepare long-format DataFrame
# df_long: ['Subject_ID', 'Condition', 'Score']
aov = AnovaRM(df_long, depvar='Score', subject='Subject_ID', within=['Condition'])
res_param = aov.fit()
print("=== Alternative: Repeated Measures ANOVA ===")
print(res_param.summary())
""", language="python")

    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
import scipy.stats as stats

def run_friedman_pipeline(Y, condition_names=("Condition 1", "Condition 2", "Condition 3"), alpha=0.05):
    \"\"\"Complete Friedman Test pipeline with within-subject ranking, Kendall's W, and post-hoc Wilcoxon comparisons.
    
    Y: 2D numpy array of shape (n_subjects, 3) where columns are repeated conditions.
    \"\"\"
    n, k = Y.shape
    print(f"=== FRIEDMAN TEST PIPELINE (N = {n} Subjects, k = {k} Conditions) ===\\n")
    
    # 1. Within-Subject Mid-Ranks
    R = np.zeros_like(Y, dtype=float)
    for i in range(n):
        R[i, :] = stats.rankdata(Y[i, :], method="average")
    mean_ranks = np.mean(R, axis=0)
    
    # 2. Friedman Test
    q_stat, p_val = stats.friedmanchisquare(Y[:, 0], Y[:, 1], Y[:, 2])
    kendalls_w = max(0.0, min(1.0, q_stat / (n * (k - 1))))
    
    print(f"Friedman Chi-Square Q({k - 1}) = {q_stat:.4f}, p-value = {p_val:.4e}")
    print(f"Kendall's W = {kendalls_w:.3f}")
    for j in range(k):
        print(f"  {condition_names[j]}: Mean Rank = {mean_ranks[j]:.2f} (Median = {np.median(Y[:, j]):.1f})")
    
    # 3. Post-Hoc Pairwise Comparisons
    if p_val < alpha:
        print("\\n--- Post-Hoc Pairwise Wilcoxon Signed-Rank Tests (Bonferroni Corrected) ---")
        pairs = [(1, 0), (2, 0), (2, 1)]
        for j, i in pairs:
            w_res = stats.wilcoxon(Y[:, j], Y[:, i], alternative="two-sided")
            p_adj = min(1.0, w_res.pvalue * len(pairs))
            diff_r = mean_ranks[j] - mean_ranks[i]
            diff_m = np.median(Y[:, j]) - np.median(Y[:, i])
            sig = p_adj < alpha
            print(f"  {condition_names[j]} vs {condition_names[i]}: Rank Diff = {diff_r:+.2f}, Median Diff = {diff_m:+.1f}, W = {w_res.statistic:.1f}, p_adj = {p_adj:.4f} -> {'SIGNIFICANT' if sig else 'No Diff'}")
    else:
        print("\\nOmnibus test not significant; pairwise comparisons not indicated.")
    print("=" * 65)

# Example Run:
# Y_demo = np.array([[2, 4, 6], [3, 5, 5], [1, 3, 7], [4, 6, 7], [2, 5, 6]])
# run_friedman_pipeline(Y_demo)
""", language="python")


def render_chi_square_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for Pearson's Chi-Square Test of Independence."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow for Pearson's Chi-Square Test
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Cochran's Rule, Expected Frequencies & Cell Counts)", expanded=False):
        st.markdown(r"""
        Pearson's Chi-Square Test ($\chi^2$) relies on a continuous asymptotic distribution that approximates discrete counts. 
        Before trusting the Chi-Square $p$-value, verify these foundational assumptions in Excel:
        1. **Mutually Exclusive & Independent Observations:** Each subject belongs to exactly one cell. No matched pairs or repeated measures.
        2. **Adequate Total Sample Size:** Total count $N \ge 20$ (ideally $N \ge 40$).
        3. **Cochran's Criterion (Expected Frequencies $E_{ij} \ge 5.0$):**
           * In a $2 \times 2$ contingency table, **100% of cells must have an expected count $\ge 5.0$** ($E_{ij} \ge 5.0$).
           * In larger $r \times c$ tables, at least **80% of cells must have $E_{ij} \ge 5.0$**, and **no cell may have $E_{ij} < 1.0$**.
           * *If any expected count is $< 5.0$ in a $2 \times 2$ table, the continuous approximation fails and inflates false alarms.* **Switch to Fisher's Exact Test (Step 3)!**
        """)

        st.markdown("##### 📋 1. Observed 2×2 Contingency Table (Excel Range: A1:D4)")
        st.markdown("""
        Enter your observed frequency counts directly into cells **`B2:C3`**. Compute row totals in column **`D`**, column totals in row **`4`**, and the Grand Total $N$ in cell **`D4`**:
        """)
        df_chi_observed_layout = pd.DataFrame({
            "Excel_Row": [
                "Row 1 (Header)",
                "Row 2 (Treatment / Group 1)",
                "Row 3 (Control / Group 2)",
                "Row 4 (Column Totals)"
            ],
            "Col_A (Exposure / Condition)": [
                "Group Variable",
                "Treatment (Group 1)",
                "Control (Group 2)",
                "Column Marginal Totals"
            ],
            "Col_B (Outcome: Event +)": [
                "Event (+)",
                "45  [Cell a]",
                "20  [Cell c]",
                "65  [Formula: =SUM(B2:B3)]"
            ],
            "Col_C (Outcome: No Event -)": [
                "No Event (-)",
                "55  [Cell b]",
                "80  [Cell d]",
                "135 [Formula: =SUM(C2:C3)]"
            ],
            "Col_D (Row Margin Totals)": [
                "Row Marginal Totals",
                "100 [Formula: =SUM(B2:C2)]",
                "100 [Formula: =SUM(B3:C3)]",
                "200 [Formula: =SUM(D2:D3) -> Grand Total N]"
            ]
        })
        render_table(df_chi_observed_layout, height=180)

        st.markdown("##### 📋 2. Expected Frequencies Grid under Independence H₀ (Excel Range: A6:D9)")
        st.markdown("""
        In cells **`B7:C8`**, calculate expected counts under $H_0$ using formula `=(RowTotal * ColTotal) / GrandTotal`:
        """)
        df_chi_expected_layout = pd.DataFrame({
            "Excel_Row": [
                "Row 6 (Header)",
                "Row 7 (Treatment / Group 1)",
                "Row 8 (Control / Group 2)",
                "Row 9 (Column Totals)"
            ],
            "Col_A (Expected Counts)": [
                "Group Variable",
                "Treatment (Group 1)",
                "Control (Group 2)",
                "Column Marginal Totals"
            ],
            "Col_B (Expected Event +)": [
                "Expected Event (+)",
                "32.50 [Formula: =($D$2 * B$4) / $D$4]",
                "32.50 [Formula: =($D$3 * B$4) / $D$4]",
                "65.00 [Formula: =SUM(B7:B8)]"
            ],
            "Col_C (Expected No Event -)": [
                "Expected No Event (-)",
                "67.50 [Formula: =($D$2 * C$4) / $D$4]",
                "67.50 [Formula: =($D$3 * C$4) / $D$4]",
                "135.00 [Formula: =SUM(C7:C8)]"
            ],
            "Col_D (Expected Row Totals)": [
                "Row Marginal Totals",
                "100.00 [Formula: =SUM(B7:C7)]",
                "100.00 [Formula: =SUM(B8:C8)]",
                "200.00 [Formula: =SUM(D7:D8)]"
            ]
        })
        render_table(df_chi_expected_layout, height=180)

        st.markdown("##### 📊 3. Assumption Diagnostic Checks & Rule Evaluation")
        df_excel_chi_assumptions = pd.DataFrame({
            "Contingency_Assumption": [
                "1. Expected Frequency Calculation",
                "2. Minimum Expected Count (Cochran's Rule)",
                "3. Sparse Cells Percentage (E < 5.0)",
                "4. Total Sample Size Adequacy (N)",
                "5. Data Structure (Independent vs Paired)"
            ],
            "Excel_Formula_or_Workflow": [
                "Expected Cell Formula: =(RowTotal * ColTotal) / GrandTotal  [e.g. =($D$2 * B$4) / $D$4]",
                "Minimum Expected Cell: =MIN(B7:C8)",
                "Count of Sparse Cells: =COUNTIF(B7:C8, \"<5\") / COUNT(B7:C8)",
                "Grand Total: =D4  [or =SUM(B2:C3)]",
                "Cross-sectional sampling: Ensure no participant appears in multiple cells"
            ],
            "Acceptable_Benchmark": [
                "Must be computed under H0 of independence for all cells",
                "Min E ≥ 5.0 (for 2x2 tables, all 4 cells must be ≥ 5.0)",
                "0% sparse cells in 2x2; ≤ 20% sparse cells in RxC (and no cell < 1.0)",
                "N ≥ 20 (ideally N ≥ 40)",
                "Mutually exclusive counts only"
            ],
            "Interpretation_Verdict": [
                "Expected counts represent what would occur by chance if variables were independent.",
                "PASS if Min E ≥ 5.0. If Min E < 5.0 -> VIOLATED: Switch to Fisher's Exact Test (Step 3).",
                "PASS if 0% in 2x2. If violated, asymptotic chi-square approximation inflates Type I errors.",
                "PASS if N ≥ 20. If N < 20, use Fisher's Exact Test.",
                "If data is paired (before/after or matched cases) -> Switch to McNemar's Test (Step 3)."
            ]
        })
        render_table(df_excel_chi_assumptions, height=220)

    # STEP 2: IMPLEMENT CHI-SQUARE IN EXCEL
    with st.expander("📌 Step 2: Implement Pearson's Chi-Square Test in Excel (Formulas & Effect Sizes)", expanded=False):
        st.markdown("""
        ##### 📋 Complete Excel Worksheet Cell Map for Chi-Square Calculations
        The table below shows exactly which cells store the intermediate $(O - E)^2 / E$ calculations, test statistics, $p$-values, and effect sizes:
        """)
        df_chi_calc_map = pd.DataFrame({
            "Target_Statistic": [
                "1. Cell Contribution (Row 1, Col 1)",
                "2. Cell Contribution (Row 1, Col 2)",
                "3. Cell Contribution (Row 2, Col 1)",
                "4. Cell Contribution (Row 2, Col 2)",
                "5. Pearson Chi-Square Statistic (χ²)",
                "6. Degrees of Freedom (df)",
                "7. Chi-Square p-Value (=CHISQ.DIST.RT)",
                "8. Native p-Value (=CHISQ.TEST)",
                "9. Critical Threshold (α = 0.05)",
                "10. Phi Coefficient (φ)",
                "11. Odds Ratio (OR)",
                "12. Relative Risk (RR)"
            ],
            "Excel_Cell": [
                "B12",
                "C12",
                "B13",
                "C13",
                "D14",
                "D15",
                "D16",
                "D17",
                "D18",
                "D19",
                "D20",
                "D21"
            ],
            "Excel_Formula": [
                "=(B2 - B7)^2 / B7",
                "=(C2 - C7)^2 / C7",
                "=(B3 - B8)^2 / B8",
                "=(C3 - C8)^2 / C8",
                "=SUM(B12:C13)",
                "=(COUNT(A2:A3)-1) * (COUNT(B1:C1)-1)",
                "=CHISQ.DIST.RT(D14, D15)",
                "=CHISQ.TEST(B2:C3, B7:C8)",
                "=CHISQ.INV.RT(0.05, D15)",
                "=SQRT(D14 / D4)",
                "=(B2 * C3) / (C2 * B3)",
                "=(B2 / D2) / (B3 / D3)"
            ],
            "Sample_Output_Value": [
                "4.808",
                "2.315",
                "4.808",
                "2.315",
                "14.246",
                "1",
                "0.00016",
                "0.00016",
                "3.841",
                "0.267 (Medium)",
                "3.273",
                "2.250"
            ],
            "Interpretation_Guidance": [
                "Contribution of Treatment Event cell to total Chi-Square.",
                "Contribution of Treatment Non-Event cell to total Chi-Square.",
                "Contribution of Control Event cell to total Chi-Square.",
                "Contribution of Control Non-Event cell to total Chi-Square.",
                "Total Chi-Square statistic. Rejects H0 if χ² > 3.841.",
                "Degrees of freedom: (2 - 1) * (2 - 1) = 1.",
                "Upper-tail p-value from Chi-square distribution; p < 0.001 is highly significant.",
                "Excel native formula directly comparing observed B2:C3 to expected B7:C8.",
                "Critical value at alpha = 0.05; χ² > 3.841 indicates significance.",
                "Phi correlation between binary variables (0.1 small, 0.3 medium, 0.5 large).",
                "Cross-product ratio; Treatment group has 3.27x higher odds of event.",
                "Risk ratio; Treatment group has 2.25x higher probability of event."
            ]
        })
        render_table(df_chi_calc_map, height=280)

        st.markdown(r"""
        ---

        **Method A: Native Excel p-Value Formula (=CHISQ.TEST)**
        * In cell **`D17`**, enter:
        ```excel
        =CHISQ.TEST(B2:C3, B7:C8)
        ```
        * Directly computes the two-tailed upper-tail $p$-value comparing observed `B2:C3` to expected `B7:C8`. Returns **`0.00016`**.

        **Method B: Full Chi-Square Statistic Calculation & Manual Table:**
        1. **Create Cell Contribution Grid (`B12:C13`):** For each cell, enter `=(Observed - Expected)^2 / Expected`:
           * Cell B12: `=(B2 - B7)^2 / B7`
           * Cell C12: `=(C2 - C7)^2 / C7`
           * Cell B13: `=(B3 - B8)^2 / B8`
           * Cell C13: `=(C3 - C8)^2 / C8`
        2. **Sum Contributions:** In cell **`D14`**, enter `=SUM(B12:C13)` ($\chi^2 = 14.246$).
        3. **Degrees of Freedom ($df$):** In cell **`D15`**, enter `1` (or `=(2-1)*(2-1)`).
        4. **Compute p-Value:** In cell **`D16`**, enter `=CHISQ.DIST.RT(D14, D15)`.
        5. **Critical Threshold ($\alpha = 0.05$):** In cell **`D18`**, enter `=CHISQ.INV.RT(0.05, D15)` ($\approx 3.841$).
           * Since calculated $\chi^2 (14.246) > 3.841$ (and $p < 0.05$), reject $H_0$: the association is statistically significant!
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Tests in Excel (Fisher's Exact & McNemar's)", expanded=False):
        st.markdown("""
        **1. Alternative 1: Fisher's Exact Test (When Any Expected Cell < 5.0 or N < 20):**
        * When expected frequencies are sparse ($E_{ij} < 5$), Pearson's continuous approximation breaks down. 
        * Excel computes exact hypergeometric point probabilities using `=HYPGEOM.DIST`:
        ```excel
        =HYPGEOM.DIST(cell_a, Row1_Total, Col1_Total, Grand_Total, FALSE)
        ```
        * **Arguments:** `sample_s` (cell a), `number_sample` (Row 1 Total), `population_s` (Col 1 Total), `number_pop` (Grand Total), `cumulative` (`FALSE` for point probability, `TRUE` for cumulative left-tail).
        * Sum all point probabilities where $P(k) \le P(\text{observed})$ to obtain the exact two-tailed Fisher $p$-value.

        **2. Alternative 2: McNemar's Test (When Binary Data is Paired or Repeated):**
        * If evaluating paired/matched binary data (e.g. pre- vs post-intervention), concordant cells ($a$ and $d$) provide no information on marginal change.
        * Identify discordant cells $b$ (row 1, col 2) and $c$ (row 2, col 1).
        * **Edwards' Continuity-Corrected Formula:**
        ```excel
        =((ABS(b - c) - 1)^2) / (b + c)
        ```
        * **McNemar p-Value:** `=CHISQ.DIST.RT(mcnemar_cell, 1)`.
        * **Small Sample Exact Binomial (if $b + c < 25$):**
        ```excel
        =2 * BINOM.DIST(MIN(b, c), b + c, 0.5, TRUE)
        ```
        """)


def render_chi_square_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python guide for Pearson's Chi-Square Test of Independence,
    including full expected count verification and conditional assurance/warning/switching statements.
    """
    st.markdown("""
    ### 🐍 Python Implementation Workflow for Pearson's Chi-Square Test
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Expected Frequencies & Cochran's Rule)", expanded=False):
        st.code("""import numpy as np
from scipy import stats
from scipy.stats.contingency import expected_freq

# 1. Define Contingency Table: [[a, b], [c, d]]
table = np.array([
    [45, 55],   # Group 1: 45 Events, 55 Non-Events (N = 100)
    [20, 80]    # Group 2: 20 Events, 80 Non-Events (N = 100)
])

# 2. Calculate Expected Cell Frequencies under Independence
expected = expected_freq(table)
n_total = np.sum(table)
min_expected = np.min(expected)
sparse_cells_count = np.sum(expected < 5.0)
pct_sparse = (sparse_cells_count / expected.size) * 100

print(f"Total Sample Size: N = {n_total}")
print(f"Observed Table:\\n{table}")
print(f"Expected Table under H0:\\n{expected.round(2)}")
print(f"Minimum Expected Cell Count: {min_expected:.2f} (Required: >= 5.0)")
print(f"Sparse Cells (< 5.0): {sparse_cells_count} of {expected.size} ({pct_sparse:.1f}%)")
""", language="python")

        st.markdown("""
        **🔍 How to Interpret the Assumption Diagnostic Output:**
        * **Cochran's Rule for $2 \\times 2$ Tables:** Every expected cell must be $\\ge 5.0$. If $\\min(E_{ij}) < 5.0$, the continuous Chi-Square distribution breaks down and inflates Type I errors (false positives).
        * **Cochran's Rule for $r \\times c$ Tables:** $\\ge 80\\%$ of cells must have $E_{ij} \\ge 5.0$, and no cell may have $E_{ij} < 1.0$.
        * **Total Sample Size:** If $N < 20$, Chi-Square should not be used.
        * **Decision Rule:**
          * If $\\min(E_{ij}) \\ge 5.0$ and $N \\ge 20 \\rightarrow$ **Assumptions Satisfied**. Proceed to Step 2 (Pearson's Chi-Square).
          * If $\\min(E_{ij}) < 5.0$ or $N < 20 \\rightarrow$ **Assumptions Violated**. Switch to Step 3 (Fisher's Exact Test).
        """)

    # STEP 2: IMPLEMENT CHI-SQUARE IN PYTHON
    with st.expander("📌 Step 2: Implement Pearson's Chi-Square Test in Python", expanded=False):
        st.code("""# 1. Run Pearson Chi-Square Test
# correction=False: Classic Pearson Chi-Square
# correction=True: Yates' Continuity Correction (conservative for 2x2)
chi2_uncorr, p_uncorr, dof, expected = stats.chi2_contingency(table, correction=False)
chi2_yates, p_yates, _, _ = stats.chi2_contingency(table, correction=True)

# 2. Compute Effect Sizes: Odds Ratio & Phi Coefficient
a, b = table[0]
c, d = table[1]
odds_ratio = (a * d) / (b * c) if (b * c) > 0 else np.nan
phi_coeff = np.sqrt(chi2_uncorr / n_total)

print(f"Pearson Chi-Square (Uncorrected): chi2({dof}) = {chi2_uncorr:.4f}, p = {p_uncorr:.4e}")
print(f"Yates Continuity-Corrected:       chi2({dof}) = {chi2_yates:.4f}, p = {p_yates:.4e}")
print(f"Odds Ratio (OR):                  OR = {odds_ratio:.3f}")
print(f"Phi Coefficient Effect Size:      phi = {phi_coeff:.3f}")
""", language="python")

        st.markdown("""
        **🔍 How to Evaluate & Interpret the Output:**
        * **Statistical Decision:** If $p < 0.05$, reject $H_0$. There is a statistically significant association between the two categorical variables.
        * **Yates' Correction:** For $2 \\times 2$ tables, Yates subtracts $0.5$ from $|O - E|$ before squaring, preventing slight over-rejection in moderate sample sizes.
        * **Phi Coefficient ($\phi$):** Measures association strength on a 0 to 1 scale ($0.10$ small, $0.30$ medium, $0.50$ large).
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Tests in Python (Fisher's Exact & McNemar's)", expanded=False):
        st.code("""# Alternative 1: Fisher's Exact Test (When Any Expected Cell < 5.0)
odds_ratio_exact, p_fisher = stats.fisher_exact(table, alternative='two-sided')
print("=== FISHER'S EXACT TEST (SMALL / SPARSE SAMPLES) ===")
print(f"Exact Odds Ratio: {odds_ratio_exact:.4f}")
print(f"Two-Sided Exact p-value: {p_fisher:.4e}")

# Alternative 2: McNemar's Test (When Binary Data is Paired / Repeated)
from statsmodels.stats.contingency_tables import mcnemar
res_mcnemar = mcnemar(table, exact=(table[0,1] + table[1,0] < 25), correction=True)
print("\\n=== MCNEMAR'S TEST (PAIRED BINARY PROPORTIONS) ===")
print(f"Statistic: {res_mcnemar.statistic:.4f}, p-value: {res_mcnemar.pvalue:.4e}")
""", language="python")

        st.markdown("""
        **🔍 When to Switch to Alternative Tests:**
        * **Use Fisher's Exact Test:** Whenever any expected cell frequency is $< 5.0$ in a $2 \\times 2$ table, or when $N < 20$. Fisher's test calculates exact probabilities from the hypergeometric distribution without continuous approximations.
        * **Use McNemar's Test:** When observations are paired (e.g. before/after measurements on the same subjects or matched case-control pairs).
        """)

    # STEP 4: CONSOLIDATED PYTHON PIPELINE SCRIPT WITH CONDITIONAL ASSURANCE/WARNING/SWITCHING PRINTS
    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
from scipy import stats
from scipy.stats.contingency import expected_freq

def run_chi_square_pipeline(table, alpha=0.05):
    \"\"\"Automated Contingency Analysis Pipeline with Cochran's Rule expected frequency
    verification, conditional assurance/warning printouts, and automated switching to
    Fisher's Exact Test when cell frequencies are sparse.
    \"\"\"
    table = np.asarray(table)
    n_total = int(np.sum(table))
    r, c = table.shape
    
    print("=" * 70)
    print(f"=== CONTINGENCY TABLE ANALYSIS ({r}x{c} Table, Total N = {n_total}) ===")
    print("=" * 70)
    print(f"Observed Counts:\\n{table}")
    
    # 1. Calculate Expected Frequencies under Independence
    expected = expected_freq(table)
    min_e = float(np.min(expected))
    sparse_count = int(np.sum(expected < 5.0))
    pct_sparse = float((sparse_count / expected.size) * 100.0)
    
    print(f"\\nExpected Counts under Independence H0:\\n{expected.round(2)}")
    
    # 2. Evaluate Cochran's Rule & Sample Size
    if r == 2 and c == 2:
        pass_cochran = (min_e >= 5.0)
    else:
        pass_cochran = (pct_sparse <= 20.0) and (min_e >= 1.0)
    pass_sample_size = (n_total >= 20)
    
    print("\\n--- ASSUMPTION DIAGNOSTICS & COCHRAN'S CRITERION ---")
    
    # Cochran's Rule Feedback
    if pass_cochran:
        print(f"[ASSURANCE - PASS] Cochran's rule satisfied: All expected cell frequencies >= 5.0 (Min E = {min_e:.2f}).")
        print("   -> Asymptotic continuous Chi-Square approximation is fully valid.")
    else:
        print(f"[INAPPROPRIATE - FAIL] Sparse expected frequencies detected: Min E = {min_e:.2f} < 5.0 ({sparse_count} cells < 5)!")
        print("   -> Asymptotic Chi-Square test breaks down and inflates Type I errors.")
        print("   -> Switching to Fisher's Exact Test (hypergeometric distribution).")
        
    # Sample Size Feedback
    if pass_sample_size:
        print(f"[ASSURANCE - PASS] Adequate total sample size: N = {n_total} >= 20.")
    else:
        print(f"[INAPPROPRIATE - FAIL] Small total sample size: N = {n_total} < 20!")
        print("   -> Asymptotic continuous approximation is unreliable. Switching to Fisher's Exact Test.")
        
    # 3. Methodological Routing Decision
    print("\\n" + "=" * 70)
    if pass_cochran and pass_sample_size:
        print("--> [VERDICT] Assumptions satisfied: Reporting Pearson's Chi-Square Test of Independence.")
        print("=" * 70)
        chi2_stat, p_val, dof, _ = stats.chi2_contingency(table, correction=False)
        chi2_yates, p_yates, _, _ = stats.chi2_contingency(table, correction=(r == 2 and c == 2))
        
        # Effect Sizes
        if r == 2 and c == 2:
            a, b = table[0]
            c_val, d = table[1]
            odds_ratio = (a * d) / (b * c_val) if (b * c_val) > 0 else np.nan
            phi = np.sqrt(chi2_stat / n_total)
            print(f"Pearson Chi-Square: chi2({dof}) = {chi2_stat:.4f}, p = {p_val:.4e}")
            print(f"Yates Continuity-Corrected: chi2({dof}) = {chi2_yates:.4f}, p = {p_yates:.4e}")
            print(f"Odds Ratio (OR):    OR = {odds_ratio:.3f}")
            print(f"Phi Coefficient:    phi = {phi:.3f}")
        else:
            cramers_v = np.sqrt(chi2_stat / (n_total * min(r - 1, c - 1)))
            print(f"Pearson Chi-Square: chi2({dof}) = {chi2_stat:.4f}, p = {p_val:.4e}")
            print(f"Cramér's V Effect:  V = {cramers_v:.3f}")
            
        is_sig = (p_val < alpha)
        print(f"Conclusion: {'Statistically significant association (Reject H0)' if is_sig else 'No significant association (Fail to reject H0)'} at alpha = {alpha}.")
        return {"method": "pearson_chi2", "statistic": chi2_stat, "p_value": p_val, "dof": dof}
        
    else:
        print("--> [VERDICT] Switching to Fisher's Exact Test due to sparse expected cells / small N.")
        print("=" * 70)
        if r == 2 and c == 2:
            odds_ratio, p_val = stats.fisher_exact(table, alternative='two-sided')
            print(f"Fisher's Exact Test Odds Ratio: OR = {odds_ratio:.4f}")
            print(f"Two-Sided Exact p-value:        p = {p_val:.4e}")
            is_sig = (p_val < alpha)
            print(f"Conclusion: {'Statistically significant association (Reject H0)' if is_sig else 'No significant association (Fail to reject H0)'} at alpha = {alpha}.")
            return {"method": "fisher_exact", "odds_ratio": odds_ratio, "p_value": p_val}
        else:
            print("Note: Fisher's exact test for tables larger than 2x2 requires Freeman-Halton extension.")
            chi2_stat, p_val, dof, _ = stats.chi2_contingency(table, correction=False)
            return {"method": "chi2_with_warning", "statistic": chi2_stat, "p_value": p_val}

# Example Run:
# table_demo = np.array([[45, 55], [20, 80]])
# run_chi_square_pipeline(table_demo)
""", language="python")


def render_fisher_exact_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for Fisher's Exact Test."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow for Fisher's Exact Test
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (2x2 Structure, Independence & Cochran's Sparseness Criterion)", expanded=False):
        st.markdown(r"""
        Fisher's Exact Test is an exact, distribution-free test specifically designed for $2 \times 2$ contingency tables with small sample sizes or sparse cell frequencies.
        Before relying on Fisher's Exact Test, set up your Excel worksheet and verify the foundational assumptions:
        1. **$2 \times 2$ Contingency Design:** Both variables are categorical with exactly two levels each ($2 \times 2$ layout: 2 rows $\times$ 2 columns).
        2. **Mutually Exclusive & Independent Observations:** Each subject belongs to exactly one cell. Data is cross-sectional (no paired before/after or matched designs).
        3. **Cochran's Sparseness Criterion ($E_{ij} < 5.0$ or $N < 20$):**
           * Calculate expected cell frequency for each cell: $E_{ij} = (\text{RowTotal}_i \times \text{ColTotal}_j) / N$.
           * If any cell has an expected count $< 5.0$ (or total sample size $N < 20$), Pearson's continuous $\chi^2$ approximation breaks down and inflates false alarms (Type I error).
           * **Verdict:** When $\min(E_{ij}) < 5.0$, **Fisher's Exact Test is strongly indicated**. If all expected counts are $\ge 5.0$ and $N \ge 20$, Fisher's test remains statistically exact, but Pearson's Chi-Square Test is the standard large-sample method.
        """)

        st.markdown("##### 📋 1. Observed 2×2 Contingency Table (Excel Range: A1:D4)")
        st.markdown("""
        Enter your observed data directly into cells **`B2:C3`**. Row totals are computed in column **`D`**, column totals in row **`4`**, and the Grand Total $N$ in cell **`D4`**:
        """)
        df_observed_layout = pd.DataFrame({
            "Excel_Row": [
                "Row 1 (Header)",
                "Row 2 (Treatment / Group 1)",
                "Row 3 (Control / Group 2)",
                "Row 4 (Column Totals)"
            ],
            "Col_A (Exposure / Condition)": [
                "Group Variable",
                "Treatment (Group 1)",
                "Control (Group 2)",
                "Column Marginal Totals"
            ],
            "Col_B (Outcome: Event +)": [
                "Event (+)",
                "12  [Cell a]",
                "4   [Cell c]",
                "16  [Formula: =SUM(B2:B3)]"
            ],
            "Col_C (Outcome: No Event -)": [
                "No Event (-)",
                "3   [Cell b]",
                "11  [Cell d]",
                "14  [Formula: =SUM(C2:C3)]"
            ],
            "Col_D (Row Margin Totals)": [
                "Row Marginal Totals",
                "15  [Formula: =SUM(B2:C2)]",
                "15  [Formula: =SUM(B3:C3)]",
                "30  [Formula: =SUM(D2:D3) -> Grand Total N]"
            ]
        })
        render_table(df_observed_layout, height=180)

        st.markdown("##### 📋 2. Expected Frequencies Grid under Independence H₀ (Excel Range: A6:D9)")
        st.markdown("""
        In cells **`B7:C8`**, calculate the expected counts under $H_0$ of independence using the formula `=(RowTotal * ColTotal) / GrandTotal`:
        """)
        df_expected_layout = pd.DataFrame({
            "Excel_Row": [
                "Row 6 (Header)",
                "Row 7 (Treatment / Group 1)",
                "Row 8 (Control / Group 2)",
                "Row 9 (Column Totals)"
            ],
            "Col_A (Expected Counts)": [
                "Group Variable",
                "Treatment (Group 1)",
                "Control (Group 2)",
                "Column Marginal Totals"
            ],
            "Col_B (Expected Event +)": [
                "Expected Event (+)",
                "8.00  [Formula: =($D$2 * B$4) / $D$4]",
                "8.00  [Formula: =($D$3 * B$4) / $D$4]",
                "16.00 [Formula: =SUM(B7:B8)]"
            ],
            "Col_C (Expected No Event -)": [
                "Expected No Event (-)",
                "7.00  [Formula: =($D$2 * C$4) / $D$4]",
                "7.00  [Formula: =($D$3 * C$4) / $D$4]",
                "14.00 [Formula: =SUM(C7:C8)]"
            ],
            "Col_D (Expected Row Totals)": [
                "Row Marginal Totals",
                "15.00 [Formula: =SUM(B7:C7)]",
                "15.00 [Formula: =SUM(B8:C8)]",
                "30.00 [Formula: =SUM(D7:D8)]"
            ]
        })
        render_table(df_expected_layout, height=180)

        st.markdown("##### 📊 3. Assumption Diagnostic Checks & Rule Evaluation")
        df_excel_fisher_assumptions = pd.DataFrame({
            "Contingency_Assumption": [
                "1. Expected Frequency Calculation",
                "2. Minimum Expected Count (Cochran's Rule)",
                "3. Total Sample Size Adequacy (N)",
                "4. Table Geometry (2x2 Structure)",
                "5. Data Independence (Non-Paired)"
            ],
            "Excel_Formula_or_Workflow": [
                "Expected Cell Formula: =(RowTotal * ColTotal) / GrandTotal  [e.g. =($D$2 * B$4) / $D$4]",
                "Minimum Expected Cell: =MIN(B7:C8)",
                "Grand Total: =D4  [or =SUM(B2:C3)]",
                "Verify range contains exactly 2 rows and 2 columns: =COUNT(B2:C3) = 4",
                "Cross-sectional check: Verify each observation is an independent subject"
            ],
            "Acceptable_Benchmark": [
                "Must be computed under H0 of independence for all 4 cells",
                "Min E < 5.0 indicates Fisher's Exact Test is required; Min E ≥ 5.0 permits Chi-Square",
                "Small sample size N < 20 or sparse table warrants exact test",
                "Exactly 2 rows x 2 columns",
                "Mutually exclusive counts only"
            ],
            "Interpretation_Verdict": [
                "Calculates theoretical frequency expected purely by random chance.",
                "If Min E < 5.0 -> ASSURANCE: Fisher's Exact Test is strictly required over Chi-Square.",
                "If N < 20 -> ASSURANCE: Asymptotic tests fail; exact hypergeometric method is required.",
                "If table is RxC (> 2x2) -> Switch to Freeman-Halton extension or Chi-Square (Step 3).",
                "If data is paired (before/after or matched pairs) -> Switch to McNemar's Test (Step 3)."
            ]
        })
        render_table(df_excel_fisher_assumptions, height=220)

    # STEP 2: IMPLEMENT FISHER'S EXACT TEST IN EXCEL
    with st.expander("📌 Step 2: Implement Fisher's Exact Test in Excel (Hypergeometric Formulas & Effect Sizes)", expanded=False):
        st.markdown(r"""
        Excel evaluates Fisher's Exact Test using the native **Hypergeometric Distribution** function `=HYPGEOM.DIST`.

        **Hypergeometric Function Syntax:**
        ```excel
        =HYPGEOM.DIST(sample_s, number_sample, population_s, number_pop, cumulative)
        ```
        * `sample_s`: Count in Cell $a$ (cell **`B2`**, Treatment Success = 12).
        * `number_sample`: Row 1 Marginal Total (cell **`D2`**, Total Treatment subjects = 15).
        * `population_s`: Column 1 Marginal Total (cell **`B4`**, Total Successes across both groups = 16).
        * `number_pop`: Grand Total $N$ (cell **`D4`**, all subjects across the table = 30).
        * `cumulative`: `FALSE` for point probability $P(X = a)$; `TRUE` for cumulative left-tail probability $P(X \le a)$.
        """)

        st.markdown("##### 📋 Complete Excel Worksheet Cell Map (Hypothesis Tests, Columns F & H:I)")
        st.markdown("""
        The table below shows exactly which cell stores each calculation, the required Excel formula, the sample output value, and the plain-English interpretation:
        """)
        df_calc_map = pd.DataFrame({
            "Target_Statistic": [
                "1. Observed Point Probability P(Obs)",
                "2. Hypergeometric Grid (Possible a values)",
                "3. Point Probability for each k",
                "4. Two-Sided Fisher Exact p-Value",
                "5. One-Sided Left-Tail p-Value",
                "6. One-Sided Right-Tail p-Value",
                "7. Sample Odds Ratio (OR)",
                "8. Standard Error SE(ln OR)",
                "9. 95% Confidence Interval Lower Limit",
                "10. 95% Confidence Interval Upper Limit",
                "11. Relative Risk (RR)",
                "12. Absolute Risk Difference (RD)"
            ],
            "Excel_Cell": [
                "F2",
                "H2:H16",
                "I2:I16",
                "F4",
                "F5",
                "F6",
                "F8",
                "F9",
                "F10",
                "F11",
                "F13",
                "F14"
            ],
            "Excel_Formula": [
                "=HYPGEOM.DIST(B2, D2, B4, D4, FALSE)",
                "Integer sequence 1 to 15 (MAX(0, B4 - D3) to MIN(D2, B4))",
                "=HYPGEOM.DIST(H2, $D$2, $B$4, $D$4, FALSE)",
                '=SUMIF(I2:I16, "<=" & F2)',
                "=HYPGEOM.DIST(B2, D2, B4, D4, TRUE)",
                "=1 - HYPGEOM.DIST(B2 - 1, D2, B4, D4, TRUE)",
                "=(B2 * C3) / (C2 * B3)",
                "=SQRT(1/B2 + 1/C2 + 1/B3 + 1/C3)",
                "=EXP(LN(F8) - 1.96 * F9)",
                "=EXP(LN(F8) + 1.96 * F9)",
                "=(B2 / D2) / (B3 / D3)",
                "=(B2 / D2) - (B3 / D3)"
            ],
            "Sample_Output_Value": [
                "0.00518",
                "k = 1, 2, ..., 15",
                "Probabilities summing to 1.0",
                "0.00947",
                "0.99965",
                "0.00557",
                "11.000",
                "0.793",
                "2.327",
                "52.007",
                "3.000",
                "+53.3% (+0.533)"
            ],
            "Interpretation_Guidance": [
                "Hypergeometric probability of obtaining cell a = 12 under H0.",
                "List of all table outcomes possible under fixed marginal totals (R1=15, C1=16).",
                "Exact probability for each outcome k; drag formula from I2 to I16.",
                "Sum of all table probabilities <= P(observed). Significant if p < 0.05.",
                "Cumulative probability of obtaining cell a <= 12.",
                "Cumulative probability of obtaining cell a >= 12 (Significant if p < 0.05).",
                "Cross-product ratio (a*d)/(b*c); Treatment odds are 11x Control odds.",
                "Large-sample standard error of log odds ratio.",
                "Lower 95% bound > 1.0 confirms statistically significant increased odds.",
                "Upper 95% bound of the odds ratio.",
                "Treatment event rate (80%) divided by Control event rate (26.7%).",
                "Treatment event rate (80%) minus Control event rate (26.7%)."
            ]
        })
        render_table(df_calc_map, height=280)

        st.markdown(r"""
        ---

        **Step-by-Step Instructions for the Two-Sided Fisher Exact p-Value in Excel:**
        1. **Point Probability of the Observed Table:** In cell `F2`, enter `=HYPGEOM.DIST(B2, D2, B4, D4, FALSE)`. For this table, $P(\text{obs}) = 0.00518$.
        2. **Create the Hypergeometric Column Grid (Columns H & I):**
           * In cells `H2:H16`, enter integers from $1$ to $15$. These represent all possible counts for cell $a$ that preserve the row totals ($R_1 = 15, R_2 = 15$) and column totals ($C_1 = 16, C_2 = 14$).
           * In cell `I2`, enter `=HYPGEOM.DIST(H2, $D$2, $B$4, $D$4, FALSE)`.
           * Double-click or copy cell `I2` down to `I16`.
        3. **Sum Probabilities Less Than or Equal to Observed:** In cell `F4`, enter:
           ```excel
           =SUMIF(I2:I16, "<=" & F2)
           ```
           * Returns the exact two-sided Fisher $p$-value: **`0.00947`** ($p < 0.05 \rightarrow$ Reject $H_0$!).
        4. **One-Sided Tests:**
           * Left-tail ($P(X \le a)$): In cell `F5`, enter `=HYPGEOM.DIST(B2, D2, B4, D4, TRUE)`.
           * Right-tail ($P(X \ge a)$): In cell `F6`, enter `=1 - HYPGEOM.DIST(B2 - 1, D2, B4, D4, TRUE)`.
        5. **Effect Sizes & 95% Confidence Intervals:**
           * Odds Ratio in cell `F8`: `=(B2 * C3) / (C2 * B3)`.
           * SE of $\ln(OR)$ in cell `F9`: `=SQRT(1/B2 + 1/C2 + 1/B3 + 1/C3)`.
           * 95% CI Lower Limit in cell `F10`: `=EXP(LN(F8) - 1.96 * F9)`.
           * 95% CI Upper Limit in cell `F11`: `=EXP(LN(F8) + 1.96 * F9)`.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Tests in Excel (Pearson's Chi-Square, McNemar's & RxC Tables)", expanded=False):
        st.markdown(r"""
        **1. Alternative 1: Pearson's Chi-Square Test (When Min Expected Count $\ge 5.0$ and $N \ge 20$):**
        * If sample size is sufficiently large satisfying Cochran's rule (all $E_{ij} \ge 5.0$), Pearson's Chi-Square Test is standard and asymptotically equivalent:
        ```excel
        =CHISQ.TEST(actual_range, expected_range)
        ```
        * Or compute $(O - E)^2 / E$ cell by cell, sum them with `=SUM()`, and compute $p$-value via `=CHISQ.DIST.RT(chi2_sum, 1)`.

        **2. Alternative 2: McNemar's Test (When Binary Data is Paired or Repeated):**
        * If the $2 \times 2$ table represents paired measurements (e.g. pre- vs post-treatment on the same individuals, or matched case-control pairs), independent sampling is violated.
        * Identify discordant cells $b$ (Row 1, Col 2) and $c$ (Row 2, Col 1).
        * **Small Sample Exact Binomial (if $b + c < 25$):**
        ```excel
        =2 * BINOM.DIST(MIN(b, c), b + c, 0.5, TRUE)
        ```
        * **Edwards' Continuity-Corrected $\chi^2$ (if $b + c \ge 25$):**
        ```excel
        =((ABS(b - c) - 1)^2) / (b + c)
        ```
        * Compute $p$-value: `=CHISQ.DIST.RT(mcnemar_stat, 1)`.

        **3. Alternative 3: Freeman-Halton Extension (When Contingency Table is Larger than $2 \times 2$):**
        * The native `=HYPGEOM.DIST` formula in Excel can only handle $2 \times 2$ tables.
        * For $2 \times 3$ or $3 \times 3$ tables with small cell counts ($E_{ij} < 5$), standard practice is to use Python Scipy, R (`fisher.test`), or Monte Carlo simulated Chi-Square tests.
        """)


def render_fisher_exact_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python guide for Fisher's Exact Test,
    including expected count verification and conditional assurance/warning/switching statements.
    """
    st.markdown("""
    ### 🐍 Python Implementation Workflow for Fisher's Exact Test
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (2x2 Structure, Independence & Expected Frequencies)", expanded=False):
        st.code("""import numpy as np
from scipy import stats
from scipy.stats.contingency import expected_freq

# 1. Define 2x2 Contingency Table: [[a, b], [c, d]]
# Row 1: Treatment (Success, Failure); Row 2: Control (Success, Failure)
table = np.array([
    [12,  3],   # Treatment: 12 Successes, 3 Failures (N1 = 15)
    [ 4, 11]    # Control:    4 Successes, 11 Failures (N2 = 15)
])

# 2. Check 2x2 Structure & Sample Size
r, c = table.shape
n_total = int(np.sum(table))

# 3. Calculate Expected Frequencies under Independence H0
expected = expected_freq(table)
min_expected = float(np.min(expected))
sparse_count = int(np.sum(expected < 5.0))

print(f"Table Shape: {r}x{c} (Required: 2x2)")
print(f"Total Sample Size: N = {n_total}")
print(f"Observed Table:\\n{table}")
print(f"Expected Table under H0:\\n{expected.round(2)}")
print(f"Minimum Expected Cell Count: {min_expected:.2f}")
print(f"Cells with Expected Count < 5: {sparse_count} of {expected.size}")
""", language="python")

        st.markdown(r"""
        **🔍 How to Interpret the Assumption Diagnostic Output:**
        * **When Fisher's Exact Test is Strongly Indicated:** When $\min(E_{ij}) < 5.0$ or $N < 20$. In small or sparse tables, continuous Chi-Square distributions significantly inflate Type I errors. Fisher's exact hypergeometric test eliminates this approximation error.
        * **When Chi-Square is Permissible:** When all expected counts $E_{ij} \ge 5.0$ and $N \ge 20$. Fisher's test remains statistically exact and valid, but Pearson's Chi-Square Test is computationally faster and universally reported.
        * **Independence Requirement:** If table represents matched pairs or before/after repeated measures, switch to McNemar's Test (Step 3).
        """)

    # STEP 2: IMPLEMENT FISHER'S EXACT TEST IN PYTHON
    with st.expander("📌 Step 2: Implement Fisher's Exact Test in Python", expanded=False):
        st.code("""# 1. Run Fisher's Exact Test using Scipy
# alternative: 'two-sided' (default), 'less', or 'greater'
odds_ratio, p_two_sided = stats.fisher_exact(table, alternative='two-sided')
_, p_less = stats.fisher_exact(table, alternative='less')
_, p_greater = stats.fisher_exact(table, alternative='greater')

# 2. Compute 95% Confidence Interval for Odds Ratio using statsmodels
from statsmodels.stats.contingency_tables import Table2x2
t22 = Table2x2(table)
ci_lower, ci_upper = t22.oddsratio_confint(alpha=0.05, method='exact')

print(f"Sample Odds Ratio (OR):          OR = {odds_ratio:.4f}")
print(f"95% Confidence Interval for OR:  [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"Two-Sided Exact p-value:         p = {p_two_sided:.4e}")
print(f"One-Sided Left-tail p-value:     p = {p_less:.4e}")
print(f"One-Sided Right-tail p-value:    p = {p_greater:.4e}")
""", language="python")

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output:**
        * **Exact Two-Sided $p$-Value:** If $p < 0.05$, reject $H_0$. There is a statistically significant association between the treatment and outcome.
        * **Odds Ratio ($OR$):** $OR > 1.0$ indicates higher odds of success in Group 1 compared to Group 2; $OR < 1.0$ indicates lower odds; $OR = 1.0$ represents null effect.
        * **Exact 95% CI:** If the confidence interval excludes $1.0$, the association is statistically significant at $\alpha = 0.05$.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Tests in Python (Chi-Square, McNemar's & Barnard's Exact)", expanded=False):
        st.code("""# Alternative 1: Pearson's Chi-Square Test (When all expected cells >= 5.0 and N >= 20)
chi2_stat, p_chi2, dof, _ = stats.chi2_contingency(table, correction=False)
chi2_yates, p_yates, _, _ = stats.chi2_contingency(table, correction=True)
print("=== ALTERNATIVE 1: PEARSON CHI-SQUARE (LARGE SAMPLES) ===")
print(f"Pearson Chi-Square: chi2({dof}) = {chi2_stat:.4f}, p = {p_chi2:.4e}")
print(f"Yates Corrected:    chi2({dof}) = {chi2_yates:.4f}, p = {p_yates:.4e}")

# Alternative 2: McNemar's Test (When Binary Data is Paired / Repeated)
from statsmodels.stats.contingency_tables import mcnemar
res_mcnemar = mcnemar(table, exact=(table[0,1] + table[1,0] < 25), correction=True)
print("\\n=== ALTERNATIVE 2: MCNEMAR'S TEST (PAIRED DATA) ===")
print(f"Statistic: {res_mcnemar.statistic:.4f}, p-value: {res_mcnemar.pvalue:.4e}")

# Alternative 3: Barnard's Exact Test (Unconditional Exact Test for 2x2 Tables)
# Often more powerful than Fisher's test when marginal totals are not fixed by study design
res_barnard = stats.barnard_exact(table, alternative='two-sided')
print("\\n=== ALTERNATIVE 3: BARNARD'S EXACT TEST (UNCONDITIONAL) ===")
print(f"Barnard Statistic: {res_barnard.statistic:.4f}, p = {res_barnard.pvalue:.4e}")
""", language="python")

        st.markdown(r"""
        **🔍 When to Switch to Alternative Tests:**
        * **Switch to Pearson's Chi-Square:** When sample size is large and all expected counts $\ge 5.0$.
        * **Switch to McNemar's Test:** When data is paired or repeated measurements on the same participants.
        * **Consider Barnard's Exact Test:** When marginal totals are free (not fixed by study design). Fisher's exact test conditions on both margins, making it slightly conservative; Barnard's test is unconditional and often more powerful.
        """)

    # STEP 4: CONSOLIDATED PYTHON PIPELINE SCRIPT WITH CONDITIONAL ASSURANCE/WARNING/SWITCHING PRINTS
    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
from scipy import stats
from scipy.stats.contingency import expected_freq

def run_fisher_exact_pipeline(table, alpha=0.05, alternative='two-sided'):
    \"\"\"Automated Fisher's Exact Test Pipeline with expected frequency verification,
    sample size checks, conditional assurance/warning printouts, and alternative method comparison.
    \"\"\"
    table = np.asarray(table)
    r, c = table.shape
    n_total = int(np.sum(table))
    
    print("=" * 70)
    print(f"=== FISHER'S EXACT TEST PIPELINE (Table Shape: {r}x{c}, Total N = {n_total}) ===")
    print("=" * 70)
    print(f"Observed Contingency Table:\\n{table}")
    
    # 1. Verify Table Geometry
    if r != 2 or c != 2:
        print(f"\\n[INAPPROPRIATE - FAIL] Contingency table is {r}x{c}. Standard Fisher's Exact test requires 2x2!")
        print("   -> SciPy stats.fisher_exact requires a 2x2 table.")
        print("   -> Switching to Pearson Chi-Square test (or Freeman-Halton extension for RxC).")
        chi2_stat, p_val, dof, _ = stats.chi2_contingency(table, correction=False)
        print(f"Pearson Chi-Square: chi2({dof}) = {chi2_stat:.4f}, p = {p_val:.4e}")
        return {"method": "chi2_fallback", "statistic": chi2_stat, "p_value": p_val}
        
    # 2. Calculate Expected Frequencies under Independence H0
    expected = expected_freq(table)
    min_e = float(np.min(expected))
    sparse_count = int(np.sum(expected < 5.0))
    
    print(f"\\nExpected Counts under Independence H0:\\n{expected.round(2)}")
    print(f"Minimum Expected Cell Count: {min_e:.2f}")
    
    # 3. Evaluate Sparseness and Sample Size for Methodological Fit
    print("\\n--- ASSUMPTION DIAGNOSTICS & METHODOLOGICAL FIT ---")
    if min_e < 5.0 or n_total < 20:
        print(f"[ASSURANCE - PASS] Fisher's Exact Test is strongly indicated: Sparse expected frequencies detected (Min E = {min_e:.2f} < 5.0) or small N = {n_total} < 20.")
        print("   -> Asymptotic continuous Chi-Square test breaks down and inflates Type I errors.")
        print("   -> Exact hypergeometric calculation provides guaranteed valid error rates.")
    else:
        print(f"[WARNING] Large sample size detected (Min E = {min_e:.2f} >= 5.0, Total N = {n_total} >= 20).")
        print("   -> Fisher's Exact Test remains statistically valid and conservative.")
        print("   -> Note: Pearson's Chi-Square Test is standard and asymptotically equivalent for this sample.")
        
    # 4. Execute Fisher's Exact Test
    print("\\n" + "=" * 70)
    print("--> [VERDICT] Executing Fisher's Exact Test (Hypergeometric Distribution)")
    print("=" * 70)
    
    odds_ratio, p_val = stats.fisher_exact(table, alternative=alternative)
    
    # Exact or Logit 95% Confidence Interval for Odds Ratio
    a, b = table[0]
    c_val, d = table[1]
    if min(a, b, c_val, d) == 0:
        # Haldane-Anscombe 0.5 continuity correction for CI calculation
        a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c_val + 0.5, d + 0.5
    else:
        a_c, b_c, c_c, d_c = a, b, c_val, d
    log_or = np.log((a_c * d_c) / (b_c * c_c))
    se_log_or = np.sqrt(1/a_c + 1/b_c + 1/c_c + 1/d_c)
    ci_lower = float(np.exp(log_or - 1.96 * se_log_or))
    ci_upper = float(np.exp(log_or + 1.96 * se_log_or))
    
    # Relative Risk & Risk Difference
    r1, r2 = a + b, c_val + d
    p1 = a / r1 if r1 > 0 else 0.0
    p2 = c_val / r2 if r2 > 0 else 0.0
    risk_diff = p1 - p2
    risk_ratio = (p1 / p2) if p2 > 0 else np.nan
    
    print(f"Sample Odds Ratio (OR):           OR = {odds_ratio:.4f}")
    print(f"95% Confidence Interval for OR:   [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"Two-Sided Exact p-value:          p = {p_val:.4e} ({alternative})")
    print(f"Risk Difference (P1 - P2):        {risk_diff*100:+.1f}%")
    if not np.isnan(risk_ratio):
        print(f"Relative Risk (RR):               RR = {risk_ratio:.3f}")
        
    is_sig = (p_val < alpha)
    print(f"\\nConclusion: {'Statistically significant association (Reject H0)' if is_sig else 'No significant association (Fail to reject H0)'} at alpha = {alpha}.")
    
    # Compare with Chi-Square if sample is moderate/large
    if min_e >= 5.0 and n_total >= 20:
        chi2_s, p_chi2, _, _ = stats.chi2_contingency(table, correction=False)
        print(f"\\nComparison with Asymptotic Chi-Square: chi2 = {chi2_s:.4f}, p = {p_chi2:.4e}")
        
    return {
        "method": "fisher_exact",
        "odds_ratio": odds_ratio,
        "ci_95": (ci_lower, ci_upper),
        "p_value": p_val,
        "is_significant": is_sig
    }

# Example Run:
# table_demo = np.array([[12, 3], [4, 11]])
# run_fisher_exact_pipeline(table_demo)
""", language="python")


def render_linear_regression_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for Linear Regression (OLS)."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Linearity, Normality, Homoscedasticity, Multicollinearity & Outliers)", expanded=False):
        st.markdown(r"""
        Before interpreting OLS regression coefficients and $p$-values, verify that the Gauss-Markov assumptions hold:
        1. **Linearity & Homoscedasticity:** Residuals must be randomly scattered around zero with constant spread across fitted values $\hat{Y}$.
        2. **Independence of Errors:** Observations must not exhibit serial correlation (Durbin-Watson $\approx 2.0$).
        3. **Normality of Residuals:** Error terms must be approximately normal (Skewness in $[-0.5, 0.5]$, Kurtosis in $[-1.0, 1.0]$).
        4. **Absence of Multicollinearity:** Predictors must not be highly correlated ($VIF < 5.0$, Tolerance $> 0.20$).
        5. **No Influential Outliers:** Standardized residuals should not exceed $\pm 3.0$ and Cook's distance should remain below $4/N$.
        """)

        df_excel_assumptions = pd.DataFrame({
            "Regression_Assumption": [
                "1. Linearity & Homoscedasticity",
                "2. Normality of Residuals",
                "3. Independence of Errors (Durbin-Watson)",
                "4. Absence of Multicollinearity (VIF)",
                "5. Outliers & Influential Observations"
            ],
            "Excel_Formula_or_Tool": [
                "Scatter Plot: Fitted Ŷ (X-axis) vs Residuals e (Y-axis)",
                "=SKEW(residual_range) and =KURT(residual_range)",
                "=SUMXMY2(residuals_2_to_N, residuals_1_to_N_minus_1) / SUMSQ(residuals)",
                "Auxiliary Regression: =1 / (1 - RSQ(X1_range, other_X_ranges))",
                "Standardized Residuals: =STANDARDIZE(e_i, 0, RMSE_cell)"
            ],
            "Acceptable_Benchmark": [
                "Uniform horizontal band around 0; no funnel or U-curve",
                "Skewness within [-0.5, 0.5]; Kurtosis within [-1.0, 1.0]",
                "Value between 1.5 and 2.5 (Ideal = ~2.0)",
                "VIF < 5.0 (Tolerance > 0.20); VIF ≥ 10 is severe",
                "|Standardized Residual| ≤ 3.0; Cook's D ≤ 4/N"
            ],
            "Interpretation_Verdict": [
                "If funnel appears -> Heteroscedastic (Step 3). If U-shaped -> Non-linear (Step 3).",
                "If non-normal and N < 30 -> Parameter CIs and p-values may be distorted.",
                "If < 1.5 -> Positive autocorrelation; If > 2.5 -> Negative autocorrelation.",
                "If VIF ≥ 10 -> Redundant predictors destabilize slopes (Step 3).",
                "If |Std Res| > 3 -> Investigate leverage points distorting line (Step 3)."
            ]
        })
        render_table(df_excel_assumptions, height=220)

    # STEP 2: IMPLEMENT OLS REGRESSION IN EXCEL
    with st.expander("📌 Step 2: Implement OLS Linear Regression in Excel (Toolpak & LINEST)", expanded=False):
        st.markdown("""
        **Method A (Recommended): Data Analysis Toolpak**
        1. **Arrange Your Data:** Outcome $Y$ in one column (e.g. `D1:D100`); contiguous predictors in adjacent columns (e.g. `A1:C100`).
        2. **Open Regression Tool:** **Data** tab $\\rightarrow$ **Data Analysis** $\\rightarrow$ **Regression** $\\rightarrow$ **OK**.
           *(Enable if missing: File $\\rightarrow$ Options $\\rightarrow$ Add-ins $\\rightarrow$ Excel Add-ins $\\rightarrow$ Check 'Analysis ToolPak').*
        3. **Input Ranges:**
           * **Input Y Range:** `$D$1:$D$100` (outcome with header).
           * **Input X Range:** `$A$1:$C$100` (all predictor columns with headers).
           * Check **Labels** and **Confidence Level (95%)**.
           * Check **Residuals** and **Residual Plots**.
        4. **Run:** Select **Output Range** (e.g. `$F$1`) $\\rightarrow$ **OK**.
        
        **How to interpret Excel's Regression Output:**
        * **Summary Output (Top Table):**
          * **Multiple R:** Multiple correlation coefficient between $Y$ and the predictor set.
          * **R Square ($R^2$):** Proportion of outcome variance explained by the model.
          * **Adjusted R Square:** Model fit adjusted for the number of predictors.
          * **Standard Error:** Root Mean Squared Error (RMSE) in outcome units.
        * **ANOVA Table (Middle Table):**
          * **Regression & Residual Rows:** Sum of squares ($SS$), degrees of freedom ($df$), and mean squares ($MS$).
          * **F & Significance F:** Model $F$-statistic and omnibus $p$-value. If *Significance F* $< 0.05$, the model is statistically significant.
        * **Coefficients Table (Bottom Table):**
          * **Coefficients ($B$):** Unstandardized intercept and slopes.
          * **Standard Error & t Stat:** Estimation uncertainty and $t$-test values.
          * **P-value:** Two-tailed test for each predictor. Values $< 0.05$ indicate statistically significant independent predictors.
          * **Lower 95% & Upper 95%:** 95% confidence intervals for each slope parameter.

        **Method B: Dynamic Array Formula (=LINEST)**
        * In an empty cell, enter:
        ```excel
        =LINEST(D2:D100, A2:C100, TRUE, TRUE)
        ```
        * Spills a 5-row by $(k+1)$-column block with slopes ($B_k, \\dots, B_1$), intercept ($B_0$), standard errors, $R^2$, RMSE, $F$-statistic, $df_{\\text{res}}$, $SS_{\\text{reg}}$, and $SS_{\\text{res}}$.

        **Method C: Simple Bivariate Native Formulas (for 1 Predictor):**
        * Slope ($B_1$): `=SLOPE(D2:D100, A2:A100)`
        * Intercept ($B_0$): `=INTERCEPT(D2:D100, A2:A100)`
        * R-Squared ($R^2$): `=RSQ(D2:D100, A2:A100)`
        * Standard Error of Estimate: `=STEYX(D2:D100, A2:A100)`
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Models in Excel (When Assumptions Fail)", expanded=False):
        st.markdown("""
        **1. When Linearity Fails (Curved Residual Pattern):**
        * **Polynomial Regression:** Insert a new column for $X^2$ using formula `=A2^2`. Include both `$A$2:$A$100` and `$B$2:$B$100` in the Toolpak $X$-range to fit a quadratic curve $\\hat{Y} = B_0 + B_1 X + B_2 X^2$.
        * **Log Transformation:** For exponential growth or diminishing returns, create `=LN(D2)` (log-outcome) or `=LN(A2)` (log-predictor) to linearize the relationship.

        **2. When Homoscedasticity Fails (Funnel-Shaped Residuals):**
        * **Weighted Least Squares (WLS):** Compute absolute residuals `=ABS(D2 - Y_fitted)`, regress them on $X$ to get fitted error standard deviations $\\hat{s}_i$, then create weights `=1 / (s_i^2)`. Multiply $Y$ and $X$ columns by $\\sqrt{w_i}$ before running `=LINEST`.

        **3. When Multicollinearity Exists (VIF ≥ 10.0):**
        * **Variable Removal / Dimension Reduction:** Remove the redundant predictor with the lower individual $t$-statistic, or create a normalized composite index `=AVERAGE(STANDARDIZE(A2,...), STANDARDIZE(B2,...))`.

        **4. When Influential Outliers Distort Slopes (|Std Res| > 3.0):**
        * **Robust Median / Trimmed Regression:** Inspect data points flagged with $|\\text{Std Res}| > 3.0$ for data entry errors. Filter them out or compare slopes with and without outliers to evaluate estimate stability.
        """)


def render_linear_regression_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python guide for Linear Regression (OLS),
    including full assumption verification with conditional assurance, warning, and alternative switching statements.
    """
    st.markdown("""
    ### 🐍 Python Implementation Workflow
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Linearity, Independence, Homoscedasticity, Normality, VIF & Leverage)", expanded=False):
        st.code("""import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Fit initial OLS model to obtain residuals and fitted values
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
residuals = model.resid
fitted_y = model.fittedvalues
n_obs = len(y)
k_preds = X.shape[1]

# 2. Test Linearity (Rainbow Test: H0 = linear relationship)
rb_stat, rb_p = linear_rainbow(model)

# 3. Test Independence of Errors (Durbin-Watson: ideal ~ 2.0)
dw = durbin_watson(residuals)

# 4. Test Homoscedasticity (Breusch-Pagan: H0 = constant variance)
bp_lm, bp_p, _, _ = het_breuschpagan(residuals, X_const)

# 5. Test Residual Normality (Shapiro-Wilk: H0 = normal errors)
sw_stat, sw_p = stats.shapiro(residuals[:min(n_obs, 5000)])

# 6. Test Multicollinearity (Variance Inflation Factor for each predictor)
vifs = [variance_inflation_factor(X_const.values, i) for i in range(1, X_const.shape[1])]
max_vif = max(vifs) if len(vifs) > 0 else 1.0

# 7. Test Outliers and Leverage (Cook's distance threshold 4/N)
influence = model.get_influence()
cooks_d, _ = influence.cooks_distance
high_influence_count = np.sum(cooks_d > (4.0 / n_obs))

print(f"Linearity Rainbow Test: p = {rb_p:.4f}")
print(f"Durbin-Watson Statistic: {dw:.3f} (Ideal: [1.5, 2.5])")
print(f"Breusch-Pagan Homoscedasticity Test: p = {bp_p:.4f}")
print(f"Shapiro-Wilk Residual Normality: p = {sw_p:.4f}")
print(f"Maximum Predictor VIF: {max_vif:.2f} (Threshold: < 5.0)")
print(f"High-Influence Observations (Cook's D > 4/N): {high_influence_count}")
""", language="python")

        st.markdown("""
        **🔍 How to Interpret the Assumption Diagnostic Output:**
        * **Linearity (`rb_p >= 0.05`):** Confirms relationship is linear. If $p < 0.05$, non-linear curvature is detected $\\rightarrow$ consider polynomial terms.
        * **Independence (`1.5 <= dw <= 2.5`):** Confirms error independence. If outside this range, autocorrelation is present $\\rightarrow$ check time-series or cluster structure.
        * **Homoscedasticity (`bp_p >= 0.05`):** Confirms constant residual variance. If $p < 0.05$, heteroscedasticity is detected $\\rightarrow$ apply HC3 robust standard errors.
        * **Normality (`sw_p >= 0.05` or `N >= 30`):** Confirms normally distributed residuals. For $N \\ge 30$, OLS slope estimates remain robust by the Central Limit Theorem.
        * **Multicollinearity (`max_vif < 5.0`):** Confirms low correlation among predictors. If $\\text{VIF} \\ge 10.0$, predictors are redundant $\\rightarrow$ apply Ridge/Lasso regularization.
        * **Influential Points (`cooks_d <= 4/N`):** No points disproportionately pull the regression line. If high leverage outliers exist $\\rightarrow$ apply Huber Robust Regression.
        """)

    # STEP 2: IMPLEMENT OLS LINEAR REGRESSION IN PYTHON
    with st.expander("📌 Step 2: Implement OLS Linear Regression in Python", expanded=False):
        st.code("""# 1. Fit Ordinary Least Squares (OLS) model
X_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_const).fit()

# 2. Display complete model summary table
print(ols_model.summary())

# 3. Extract core summary metrics
r_squared = ols_model.rsquared
adj_r_squared = ols_model.rsquared_adj
f_stat = ols_model.fvalue
f_pvalue = ols_model.f_pvalue
rmse = np.sqrt(ols_model.mse_resid)

print(f"\\nModel R² = {r_squared:.4f} ({r_squared*100:.1f}% variance explained)")
print(f"Adjusted R² = {adj_r_squared:.4f}")
print(f"Omnibus Model F({ols_model.df_model:.0f}, {ols_model.df_resid:.0f}) = {f_stat:.2f}, p = {f_pvalue:.4e}")
print(f"Root Mean Squared Error (RMSE) = {rmse:.3f}")
""", language="python")

        st.markdown("""
        **🔍 How to Evaluate & Interpret OLS Output:**
        * **Omnibus Model Fit ($p < 0.05$):** Reject $H_0$; the combined set of predictors explains significantly more variance than random chance.
        * **$R^2$ vs Adjusted $R^2$:** Percentage of outcome variation explained. Adjusted $R^2$ penalizes non-informative predictors.
        * **Predictor $p$-values ($P > |t|$):** If $p < 0.05$, that specific predictor makes a statistically significant unique contribution after adjusting for all other predictors in the model.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Models in Python (When Assumptions Fail)", expanded=False):
        st.code("""# Alternative 1: HC3 Robust Standard Errors (When Homoscedasticity Fails)
robust_model = ols_model.get_robustcov_results(cov_type='HC3')
print("=== HC3 HETEROSCEDASTICITY-ROBUST RESULTS ===")
print(robust_model.summary())

# Alternative 2: Huber Robust Regression (When Outliers or High Leverage Distort Slopes)
huber_model = sm.RLM(y, X_const, M=sm.robust.norms.HuberT()).fit()
print("\\n=== HUBER ROBUST REGRESSION (RLM) RESULTS ===")
print(huber_model.summary())

# Alternative 3: Ridge Regularization (When Severe Multicollinearity VIF >= 10 Exists)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0).fit(X, y)
print(f"\\nRidge Intercept: {ridge.intercept_:.3f}")
print(f"Ridge Coefficients: {ridge.coef_}")
""", language="python")

        st.markdown("""
        **🔍 When to Switch to Alternative Models:**
        * **Use HC3 Robust Errors:** When Breusch-Pagan test fails ($p < 0.05$). Keeps OLS slopes unchanged but recalculates valid standard errors and $p$-values.
        * **Use Huber Robust Regression (RLM):** When influential outliers ($D_i > 4/N$) distort OLS slopes. Downweights extreme observations without dropping them.
        * **Use Ridge / Lasso Regularization:** When multicollinearity ($\\text{VIF} \\ge 10$) destabilizes individual coefficients.
        """)

    # STEP 4: CONSOLIDATED PYTHON PIPELINE SCRIPT WITH CONDITIONAL ASSURANCE/WARNING/SWITCHING PRINTS
    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

def run_regression_pipeline(X, y, feature_names=None, alpha=0.05):
    \"\"\"Automated Ordinary Least Squares (OLS) Regression Pipeline with rigorous
    Gauss-Markov assumption verification, conditional assurance/warning printouts,
    and automated routing to robust alternatives when assumptions fail.
    \"\"\"
    X = np.asarray(X)
    y = np.asarray(y)
    n, k = X.shape
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(k)]
    
    print("=" * 70)
    print(f"=== OLS REGRESSION PIPELINE (N = {n}, Predictors k = {k}) ===")
    print("=" * 70)
    
    # 1. Fit Baseline OLS Model
    X_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_const).fit()
    residuals = ols_model.resid
    fitted = ols_model.fittedvalues
    
    # 2. Rigorous Assumption Diagnostics
    # A. Linearity (Rainbow Test)
    try:
        _, rb_p = linear_rainbow(ols_model)
        pass_linear = (rb_p >= alpha)
    except Exception:
        rb_p = 1.0
        pass_linear = True
        
    # B. Independence (Durbin-Watson)
    dw = float(durbin_watson(residuals))
    pass_dw = (1.5 <= dw <= 2.5)
    
    # C. Homoscedasticity (Breusch-Pagan)
    try:
        _, bp_p, _, _ = het_breuschpagan(residuals, X_const)
        pass_homo = (bp_p >= alpha)
    except Exception:
        bp_p = 1.0
        pass_homo = True
        
    # D. Residual Normality (Shapiro-Wilk)
    sub_res = residuals[:min(n, 5000)]
    sw_stat, sw_p = stats.shapiro(sub_res)
    pass_norm = (sw_p >= alpha) or (n >= 30 and abs(stats.skew(residuals)) <= 1.0)
    
    # E. Multicollinearity (VIF)
    if k > 1:
        vifs = [variance_inflation_factor(X_const, i) for i in range(1, k + 1)]
        max_vif = max(vifs)
        pass_vif = (max_vif < 5.0)
        severe_vif = (max_vif >= 10.0)
    else:
        vifs = [1.0]
        max_vif = 1.0
        pass_vif = True
        severe_vif = False
        
    # F. Outliers & Leverage (Cook's Distance)
    influence = ols_model.get_influence()
    cooks_d, _ = influence.cooks_distance
    cooks_thresh = 4.0 / n
    n_high_cooks = int(np.sum(cooks_d > cooks_thresh))
    pass_cooks = (n_high_cooks == 0)
    
    # 3. Print Conditional Diagnostics with Assurance / Warning / Action Statements
    print("\\n--- GAUSS-MARKOV ASSUMPTION DIAGNOSTICS ---")
    
    # Linearity Feedback
    if pass_linear:
        print(f"[ASSURANCE - PASS] Linearity verified (Rainbow test p = {rb_p:.4f} >= {alpha}).")
    else:
        print(f"[WARNING - VIOLATED] Non-linear relationship detected (Rainbow test p = {rb_p:.4f} < {alpha})!")
        print("   -> Consider polynomial terms (X²) or Generalized Additive Models (GAM).")
        
    # Independence Feedback
    if pass_dw:
        print(f"[ASSURANCE - PASS] Error independence confirmed: Durbin-Watson = {dw:.3f} (Ideal: ~2.0).")
    else:
        print(f"[WARNING] Potential residual autocorrelation: Durbin-Watson = {dw:.3f} outside [1.5, 2.5].")
        print("   -> Verify sample is cross-sectional; adjust for clustering or time series if applicable.")
        
    # Homoscedasticity Feedback
    if pass_homo:
        print(f"[ASSURANCE - PASS] Homoscedasticity confirmed: Constant variance (Breusch-Pagan p = {bp_p:.4f} >= {alpha}).")
    else:
        print(f"[INAPPROPRIATE - FAIL] Heteroscedasticity detected: Non-constant error variance (Breusch-Pagan p = {bp_p:.4f} < {alpha})!")
        print("   -> Standard OLS standard errors and p-values are invalid! Switching to HC3 Robust Errors.")
        
    # Normality Feedback
    if pass_norm:
        print(f"[ASSURANCE - PASS] Residual normality confirmed: (Shapiro-Wilk p = {sw_p:.4f}, N = {n}).")
    else:
        print(f"[WARNING] Residual non-normality detected: (Shapiro-Wilk p = {sw_p:.4f} < {alpha}).")
        print("   -> For N >= 30, OLS point estimates remain robust by the Central Limit Theorem.")
        
    # Multicollinearity Feedback
    if pass_vif:
        print(f"[ASSURANCE - PASS] Multicollinearity low: Max VIF = {max_vif:.2f} < 5.0 across all predictors.")
    elif not severe_vif:
        print(f"[WARNING] Moderate multicollinearity: Max VIF = {max_vif:.2f} (5.0 to 10.0). Standard errors slightly inflated.")
    else:
        print(f"[INAPPROPRIATE - FAIL] Severe multicollinearity: Max VIF = {max_vif:.2f} >= 10.0!")
        print("   -> Predictors are highly redundant, destabilizing individual slope estimates. Consider Ridge regression.")
        
    # Outliers / Leverage Feedback
    if pass_cooks:
        print(f"[ASSURANCE - PASS] Outliers & Leverage: No points exceed Cook's distance threshold (4/N = {cooks_thresh:.3f}).")
    else:
        print(f"[WARNING] Influential outliers detected: {n_high_cooks} point(s) exceed Cook's D threshold ({cooks_thresh:.3f}).")
        print("   -> Extreme observations may exert undue leverage on OLS slopes. Exploring Huber Robust Regression.")
        
    # 4. Final Methodological Routing Decision
    print("\\n" + "=" * 70)
    if pass_linear and pass_homo and not severe_vif and (pass_cooks or n >= 50):
        print("--> [VERDICT] Assumptions satisfied: Reporting standard Ordinary Least Squares (OLS) model.")
        print("=" * 70)
        final_model = ols_model
        print(final_model.summary())
    elif not pass_homo:
        print("--> [VERDICT] Switching to HC3 Heteroscedasticity-Consistent Standard Errors.")
        print("=" * 70)
        final_model = ols_model.get_robustcov_results(cov_type='HC3')
        print(final_model.summary())
    elif not pass_cooks and n < 50:
        print("--> [VERDICT] Switching to Huber Robust Linear Model (RLM) due to influential outliers.")
        print("=" * 70)
        final_model = sm.RLM(y, X_const, M=sm.robust.norms.HuberT()).fit()
        print(final_model.summary())
    else:
        print("--> [VERDICT] Reporting OLS model with cautionary diagnostic warnings.")
        print("=" * 70)
        final_model = ols_model
        print(final_model.summary())
        
    return final_model

# Example Run:
# X_demo = np.random.randn(80, 2)
# y_demo = 15.0 + 2.5 * X_demo[:, 0] - 1.8 * X_demo[:, 1] + np.random.randn(80)
# run_regression_pipeline(X_demo, y_demo, feature_names=["Study_Hours", "Prior_GPA"])
""", language="python")


def render_mcnemar_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for McNemar's Test for Paired Proportions."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow for McNemar's Test
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Paired Binary Structure & Discordant Count Adequacy)", expanded=False):
        st.markdown(r"""
        McNemar's test evaluates changes in marginal proportions for **paired, dependent binary data** (e.g., repeated measurements on the same participants before and after an intervention, or 1:1 matched case-control pairs).
        
        Before interpreting the McNemar test statistic, verify the foundational assumptions in Excel:
        1. **Paired / Matched Dependency:** Each row represents a subject observed under two conditions (Time 1 vs Time 2). Observations are not independent.
        2. **Binary Dichotomous Outcomes:** Status must have exactly two mutually exclusive states (e.g. Positive `+` vs Negative `-`).
        3. **Role of Concordant vs Discordant Pairs:**
           * **Concordant pairs ($a$ and $d$):** Subjects whose status did not change ($+ \to +$ or $- \to -$). They provide zero information on marginal shift and cancel out of the test!
           * **Discordant pairs ($b$ and $c$):** Subjects who transitioned between states. Cell $b$ represents transitions from Positive to Negative ($+ \to -$); Cell $c$ represents transitions from Negative to Positive ($- \to +$).
        4. **Discordant Sample Size Adequacy ($b + c \ge 25$):**
           * If total discordant pairs $b + c \ge 25$, Edwards' continuity-corrected asymptotic $\chi^2$ distribution is fully valid.
           * If $b + c < 25$, continuous approximations break down and inflate error rates. **Switch to the Exact Binomial McNemar Test (Step 2 & Step 3)!**
        """)

        st.markdown("##### 📋 1. Observed 2×2 Paired Transition Matrix (Excel Range: A1:D4)")
        st.markdown("""
        Enter your paired counts into cells **`B2:C3`**. Compute pre-test totals in column **`D`**, post-test totals in row **`4`**, and Grand Total $N$ in cell **`D4`**:
        """)
        df_mcnemar_observed_layout = pd.DataFrame({
            "Excel_Row": [
                "Row 1 (Header)",
                "Row 2 (Pre-Test: Positive)",
                "Row 3 (Pre-Test: Negative)",
                "Row 4 (Post-Test Totals)"
            ],
            "Col_A (Pre-Intervention Status)": [
                "Pre-Test Condition",
                "Pre-Test: Positive (+)",
                "Pre-Test: Negative (-)",
                "Post-Test Marginal Totals"
            ],
            "Col_B (Post-Test: Positive +)": [
                "Post-Test (+)",
                "15  [Cell a: Both Positive (Concordant)]",
                "25  [Cell c: Pre - / Post + (Discordant Gain)]",
                "40  [Formula: =SUM(B2:B3)]"
            ],
            "Col_C (Post-Test: Negative -)": [
                "Post-Test (-)",
                "5   [Cell b: Pre + / Post - (Discordant Loss)]",
                "55  [Cell d: Both Negative (Concordant)]",
                "60  [Formula: =SUM(C2:C3)]"
            ],
            "Col_D (Pre-Test Row Totals)": [
                "Pre-Test Row Totals",
                "20  [Formula: =SUM(B2:C2)]",
                "80  [Formula: =SUM(B3:C3)]",
                "100 [Formula: =SUM(D2:D3) -> Grand Total N]"
            ]
        })
        render_table(df_mcnemar_observed_layout, height=180)

        st.markdown("##### 📊 2. Assumption Diagnostic Checks & Rule Evaluation")
        df_mcnemar_assumptions = pd.DataFrame({
            "Paired_Assumption": [
                "1. Dependent / Paired Sampling Structure",
                "2. Binary Outcome Dichotomy",
                "3. Discordant Pair Adequacy (b + c)",
                "4. Discordance Benchmark Rule",
                "5. Concordant Pairs Information (a + d)"
            ],
            "Excel_Formula_or_Workflow": [
                "Verify observations represent identical subjects or 1:1 matched pairs",
                "Verify outcomes have exactly 2 mutually exclusive levels (+ vs -)",
                "Discordant Total: =C2 + B3  [Cell b + Cell c]",
                '=IF(C2 + B3 >= 25, "Asymptotic Chi-Square Valid (>= 25)", "Use Exact Binomial (< 25)")',
                "Concordant Total: =B2 + C3  [Cell a + Cell d]"
            ],
            "Acceptable_Benchmark": [
                "Must be paired/repeated (not independent cross-sectional)",
                "Dichotomous binary outcome only",
                "b + c >= 25 for Edwards Chi-Square; b + c < 25 requires exact binomial",
                "Benchmark threshold: 25 discordant transitions",
                "Cells a and d provide zero information on marginal shift"
            ],
            "Interpretation_Verdict": [
                "PASS if paired. If independent groups -> Switch to Pearson Chi-Square (Step 3).",
                "PASS if binary. If ordinal/continuous -> Switch to Wilcoxon Signed-Rank.",
                "In this example, b + c = 5 + 25 = 30 >= 25 -> Edwards Chi-Square is fully valid.",
                "If b + c < 25 -> Use Exact Binomial formula =2*BINOM.DIST(...).",
                "Concordant pairs (70 subjects) stayed unchanged; test evaluates whether 25 gains exceed 5 losses."
            ]
        })
        render_table(df_mcnemar_assumptions, height=220)

    # STEP 2: IMPLEMENT MCNEMAR'S TEST IN EXCEL
    with st.expander("📌 Step 2: Implement McNemar's Test in Excel (Edwards' Chi-Square, Exact Binomial & Effect Sizes)", expanded=False):
        st.markdown(r"""
        Excel evaluates McNemar's Test using either the **Edwards' Continuity-Corrected Chi-Square** formula (when $b + c \ge 25$) or the **Exact Binomial Distribution** `=BINOM.DIST` (when $b + c < 25$).

        **Core Formulas:**
        * **Total Discordant Pairs ($n_{\text{disc}}$):** $b + c$
        * **Edwards' Continuity-Corrected Statistic:**
          $$\chi^2_{\text{Edwards}} = \frac{(|b - c| - 1)^2}{b + c}$$
        * **Exact Binomial $p$-Value ($H_0: p = 0.5$):**
          $$p_{\text{exact}} = 2 \times \text{BINOM.DIST}(\min(b, c), b + c, 0.5, \text{TRUE})$$
        * **Marginal Shift / Risk Difference ($\Delta p$):** $\Delta p = \frac{c - b}{N} = p_{\text{Post}} - p_{\text{Pre}}$
        * **Conditional Odds Ratio ($OR_{\text{paired}}$):** $OR = \frac{c}{b}$
        """)

        st.markdown("##### 📋 Complete Excel Worksheet Cell Map for McNemar Calculations (Columns F:G)")
        st.markdown("""
        The table below shows exactly which cell stores each intermediate calculation, the required Excel formula, the sample output value, and the plain-English interpretation:
        """)
        df_mcnemar_calc_map = pd.DataFrame({
            "Target_Statistic": [
                "1. Discordant Gain (Pre - / Post +)",
                "2. Discordant Loss (Pre + / Post -)",
                "3. Total Discordant Pairs (b + c)",
                "4. Net Discordant Difference (|b - c|)",
                "5. Edwards Continuity-Corrected χ²",
                "6. Edwards Asymptotic p-Value",
                "7. Exact Binomial Two-Sided p-Value",
                "8. Pre-Intervention Rate (p1)",
                "9. Post-Intervention Rate (p2)",
                "10. Marginal Shift / Risk Difference (Δp)",
                "11. Conditional Odds Ratio (OR)",
                "12. Standard Error SE(ln OR)",
                "13. 95% CI Lower Bound",
                "14. 95% CI Upper Bound"
            ],
            "Excel_Cell": [
                "B3",
                "C2",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F8",
                "F9",
                "F10",
                "F11",
                "F12",
                "F13",
                "F14"
            ],
            "Excel_Formula": [
                "25",
                "5",
                "=C2 + B3",
                "=ABS(C2 - B3)",
                "=((ABS(C2 - B3) - 1)^2) / (C2 + B3)",
                "=CHISQ.DIST.RT(F4, 1)",
                "=2 * BINOM.DIST(MIN(C2, B3), C2 + B3, 0.5, TRUE)",
                "=D2 / D4",
                "=B4 / D4",
                "=(B3 - C2) / D4",
                "=B3 / C2",
                "=SQRT(1/C2 + 1/B3)",
                "=EXP(LN(F11) - 1.96 * F12)",
                "=EXP(LN(F11) + 1.96 * F12)"
            ],
            "Sample_Output_Value": [
                "25",
                "5",
                "30",
                "20",
                "12.033",
                "0.00052",
                "0.00033",
                "20.0%",
                "40.0%",
                "+20.0% (+0.200)",
                "5.000",
                "0.490",
                "1.914",
                "13.064"
            ],
            "Interpretation_Guidance": [
                "Count of participants transitioning from Negative to Positive (Gain).",
                "Count of participants transitioning from Positive to Negative (Loss).",
                "Total number of subjects whose binary status changed (30 >= 25).",
                "Absolute difference between discordant transitions (|25 - 5| = 20).",
                "Continuity-corrected test statistic with 1 degree of freedom.",
                "Upper-tail p-value from Chi-Square distribution (Significant: p < 0.001).",
                "Exact two-sided binomial p-value (universally exact; required when b + c < 25).",
                "Proportion of participants positive before intervention: 20 / 100 = 20.0%.",
                "Proportion of participants positive after intervention: 40 / 100 = 40.0%.",
                "Net increase in positive proportion: (25 - 5) / 100 = +20.0 percentage points.",
                "Ratio of discordant transitions: odds of gaining positive status are 5x losing it.",
                "Large-sample standard error of log conditional odds ratio.",
                "Lower 95% bound > 1.0 confirms statistically significant positive shift.",
                "Upper 95% bound of conditional odds ratio."
            ]
        })
        render_table(df_mcnemar_calc_map, height=280)

        st.markdown(r"""
        ---

        **Step-by-Step Instructions for McNemar's Test in Excel:**
        1. **Calculate Total Discordant Pairs:** In cell **`F2`**, enter `=C2 + B3` ($5 + 25 = 30$).
        2. **Method A — Edwards' Continuity-Corrected Chi-Square (When $b + c \ge 25$):**
           * In cell **`F4`**, enter `=((ABS(C2 - B3) - 1)^2) / (C2 + B3)`. This yields $\chi^2 = 12.033$.
           * In cell **`F5`**, compute the $p$-value: `=CHISQ.DIST.RT(F4, 1)`. Returns **`0.00052`** ($p < 0.001 \rightarrow$ Reject $H_0$!).
        3. **Method B — Exact Binomial McNemar Test (When $b + c < 25$):**
           * In cell **`F6`**, enter `=2 * BINOM.DIST(MIN(C2, B3), C2 + B3, 0.5, TRUE)`.
           * Returns **`0.00033`**, providing an exact test without continuous approximations.
        4. **Effect Sizes & 95% Confidence Intervals:**
           * **Marginal Shift / Risk Difference ($\Delta p$):** In cell **`F10`**, enter `=(B3 - C2) / D4`. Shows a **`+20.0%`** net increase.
           * **Conditional Odds Ratio ($OR$):** In cell **`F11`**, enter `=B3 / C2` ($25 / 5 = 5.00$).
           * **95% Confidence Interval for $OR$:**
             * SE in cell `F12`: `=SQRT(1/C2 + 1/B3)`.
             * Lower limit in cell `F13`: `=EXP(LN(F11) - 1.96 * F12)` ($1.914$).
             * Upper limit in cell `F14`: `=EXP(LN(F11) + 1.96 * F12)` ($13.064$). Excludes $1.0$, confirming statistical significance!
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Tests in Excel (Independent Groups, 3+ Timepoints & Ordinal Outcomes)", expanded=False):
        st.markdown(r"""
        **1. Alternative 1: Exact Binomial McNemar Test (When $b + c < 25$):**
        * If total discordant pairs $b + c < 25$, the continuous Chi-Square approximation is inaccurate.
        * Use Excel's native binomial distribution formula:
        ```excel
        =2 * BINOM.DIST(MIN(b, c), b + c, 0.5, TRUE)
        ```
        * Directly computes the exact two-tailed $p$-value under $H_0: p = 0.5$.

        **2. Alternative 2: Pearson's Chi-Square or Fisher's Exact Test (When Groups are Independent):**
        * If the $2 \times 2$ table represents two independent, separate cohorts rather than repeated measurements on the same individuals, do not use McNemar's test.
        * Use `=CHISQ.TEST(actual_range, expected_range)` or Fisher's exact test (`=HYPGEOM.DIST`).

        **3. Alternative 3: Cochran's Q Test (When Measuring Across $k \ge 3$ Repeated Time Points):**
        * If the same subjects are assessed across 3 or more occasions (e.g. Baseline, 3 Months, 6 Months) with binary outcomes, McNemar's test generalizes to Cochran's Q Test.
        * In Excel, calculate row sums $R_i = \sum_{j} X_{ij}$ and column sums $C_j = \sum_{i} X_{ij}$:
        ```excel
        = (k - 1) * (k * SUMSQ(C_range) - (SUM(C_range))^2) / (k * SUM(C_range) - SUMSQ(R_range))
        ```
        * Compute $p$-value: `=CHISQ.DIST.RT(Q_stat, k - 1)`.
        """)


def render_mcnemar_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python guide for McNemar's Test for Paired Proportions,
    including discordant pair verification and conditional assurance/warning/switching statements.
    """
    st.markdown("""
    ### 🐍 Python Implementation Workflow for McNemar's Test
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Paired Matrix & Discordant Counts)", expanded=False):
        st.code("""import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# 1. Define Paired 2x2 Transition Matrix: [[a, b], [c, d]]
# Row 1: Pre-Test (+); Row 2: Pre-Test (-)
# Col 1: Post-Test (+); Col 2: Post-Test (-)
table = np.array([
    [15,  5],   # [a: Pre+, Post+ (Concordant); b: Pre+, Post- (Discordant Loss)]
    [25, 55]    # [c: Pre-, Post+ (Discordant Gain); d: Pre-, Post- (Concordant)]
])

# 2. Extract Concordant and Discordant Pair Counts
a, b = table[0]
c, d = table[1]
n_total = int(np.sum(table))
n_conc = a + d
n_disc = b + c

print(f"Total Sample Size: N = {n_total}")
print(f"Paired Transition Matrix:\\n{table}")
print(f"Concordant Pairs (No change): a + d = {n_conc} ({n_conc/n_total*100:.1f}%)")
print(f"Discordant Pairs (Changed):   b + c = {n_disc} ({n_disc/n_total*100:.1f}%)")
print(f"  - Discordant Losses (+ to -): cell b = {b}")
print(f"  - Discordant Gains  (- to +): cell c = {c}")
""", language="python")

        st.markdown(r"""
        **🔍 How to Interpret the Assumption Diagnostic Output:**
        * **Paired Sampling Design:** Data must represent paired/repeated binary measurements on the same individuals. Concordant pairs ($a, d$) provide zero information about marginal shift and drop out of the test.
        * **Discordant Count Benchmark ($b + c \ge 25$):**
          * If $b + c \ge 25 \rightarrow$ **Assumptions Satisfied**. Proceed to Edwards' continuity-corrected asymptotic Chi-Square test (`exact=False, correction=True`).
          * If $b + c < 25 \rightarrow$ **Sparse Discordance Warning**. The asymptotic approximation fails. Automatically switch to the Exact Binomial McNemar Test (`exact=True`).
        """)

    # STEP 2: IMPLEMENT MCNEMAR'S TEST IN PYTHON
    with st.expander("📌 Step 2: Implement McNemar's Test in Python", expanded=False):
        st.code("""# 1. Execute Edwards Continuity-Corrected McNemar Test (Asymptotic, b + c >= 25)
res_asymp = mcnemar(table, exact=False, correction=True)

# 2. Execute Exact Binomial McNemar Test (Exact, recommended when b + c < 25)
res_exact = mcnemar(table, exact=True)

# 3. Calculate Clinical Effect Sizes: Marginal Shift & Conditional Odds Ratio
p_pre = (a + b) / n_total
p_post = (a + c) / n_total
marginal_shift = p_post - p_pre
cond_or = c / b if b > 0 else np.nan

# 95% Confidence Interval for Conditional Odds Ratio
if b > 0 and c > 0:
    se_log_or = np.sqrt(1/b + 1/c)
    ci_lower = float(np.exp(np.log(cond_or) - 1.96 * se_log_or))
    ci_upper = float(np.exp(np.log(cond_or) + 1.96 * se_log_or))
else:
    ci_lower, ci_upper = np.nan, np.nan

print(f"Edwards Chi2 Statistic:      chi2 = {res_asymp.statistic:.4f}, p = {res_asymp.pvalue:.4e}")
print(f"Exact Binomial p-value:      p = {res_exact.pvalue:.4e}")
print(f"Pre-Test Positive Rate:      {p_pre*100:.1f}%")
print(f"Post-Test Positive Rate:     {p_post*100:.1f}%")
print(f"Marginal Shift (Post - Pre): {marginal_shift*100:+.1f}%")
print(f"Conditional Odds Ratio:      OR = {cond_or:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
""", language="python")

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output:**
        * **Statistical Decision:** If $p < 0.05$, reject $H_0$. There is a statistically significant marginal shift in proportions between Time 1 and Time 2.
        * **Marginal Shift ($\Delta p$):** Quantifies the net percentage point increase (or decrease) in the positive outcome rate after intervention.
        * **Conditional Odds Ratio ($OR_{\text{paired}} = c/b$):** Measures the relative odds of transitioning from Negative to Positive compared to Positive to Negative. $OR > 1.0$ indicates favorable positive transitions.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Tests in Python (Cochran's Q, Chi-Square & Wilcoxon)", expanded=False):
        st.code("""# Alternative 1: Cochran's Q Test (When Evaluating >= 3 Repeated Time Points)
from statsmodels.stats.contingency_tables import cochrans_q
# Example: Binary outcomes for 10 subjects across 3 time points
data_3t = np.array([
    [0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 1],
    [1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]
])
res_q = cochrans_q(data_3t)
print("=== ALTERNATIVE 1: COCHRAN'S Q TEST (>= 3 TIME POINTS) ===")
print(f"Cochran Q Statistic: {res_q.statistic:.4f}, p-value: {res_q.pvalue:.4e}, df: {res_q.df}")

# Alternative 2: Pearson's Chi-Square Test (When Observations are Independent Groups)
from scipy import stats
chi2_stat, p_chi2, _, _ = stats.chi2_contingency(table, correction=False)
print("\\n=== ALTERNATIVE 2: PEARSON CHI-SQUARE (INDEPENDENT GROUPS) ===")
print(f"Chi-Square: {chi2_stat:.4f}, p-value: {p_chi2:.4e}")

# Alternative 3: Wilcoxon Signed-Rank Test (When Outcome is Ordinal or Continuous)
# Use stats.wilcoxon(pre_scores, post_scores) if responses are ordered ratings (e.g. Likert 1-5)
""", language="python")

        st.markdown(r"""
        **🔍 When to Switch to Alternative Tests:**
        * **Use Cochran's Q Test:** When evaluating binary repeated measures across $3$ or more conditions or time points on the same individuals.
        * **Use Pearson Chi-Square or Fisher's Exact:** If observations represent independent, cross-sectional cohorts rather than matched/repeated pairs.
        * **Use Wilcoxon Signed-Rank:** If the paired response variable is ordinal (e.g. 1 to 5 scale) or continuous, rather than binary.
        """)

    # STEP 4: CONSOLIDATED PYTHON PIPELINE SCRIPT WITH CONDITIONAL ASSURANCE/WARNING/SWITCHING PRINTS
    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar_pipeline(table, alpha=0.05):
    \"\"\"Automated McNemar Pipeline for paired binary contingency tables with
    discordant pair verification, conditional assurance/warning printouts,
    and automatic routing between Edwards' Chi-Square and Exact Binomial tests.
    \"\"\"
    table = np.asarray(table)
    r, c = table.shape
    n_total = int(np.sum(table))
    
    print("=" * 70)
    print(f"=== MCNEMAR PAIRED BINARY PIPELINE (Table Shape: {r}x{c}, Total N = {n_total}) ===")
    print("=" * 70)
    print(f"Observed Paired Transition Matrix:\\n{table}")
    
    # 1. Verify Table Geometry
    if r != 2 or c != 2:
        print(f"\\n[INAPPROPRIATE - FAIL] Contingency table is {r}x{c}. McNemar's test requires a 2x2 paired table!")
        print("   -> For RxR paired nominal data (> 2 categories), use Stuart-Maxwell or Bowker's test of symmetry.")
        return {"method": "inappropriate", "error": "Table is not 2x2"}
        
    a, b = table[0]
    c_val, d = table[1]
    n_conc = int(a + d)
    n_disc = int(b + c_val)
    
    print(f"\\nConcordant Pairs (No Transition): a + d = {n_conc} ({n_conc/n_total*100:.1f}%)")
    print(f"Discordant Pairs (Transitioned):    b + c = {n_disc} ({n_disc/n_total*100:.1f}%)")
    print(f"  - Lost Positive (+ to -): cell b = {b}")
    print(f"  - Gained Positive (- to +): cell c = {c_val}")
    
    # 2. Evaluate Discordant Count Adequacy for Methodological Fit
    print("\\n--- ASSUMPTION DIAGNOSTICS & METHODOLOGICAL FIT ---")
    if n_disc == 0:
        print("[WARNING] Zero discordant pairs detected (b = 0, c = 0).")
        print("   -> 100% concordance observed; no marginal shift occurred across timepoints (p = 1.0).")
        return {"method": "mcnemar_zero_discordance", "p_value": 1.0}
    elif n_disc >= 25:
        print(f"[ASSURANCE - PASS] Discordant pair count is adequate: b + c = {n_disc} >= 25.")
        print("   -> Asymptotic continuous Chi-Square approximation with Edwards' continuity correction is fully valid.")
        use_exact = False
    else:
        print(f"[WARNING] Sparse discordant pairs detected: b + c = {n_disc} < 25.")
        print("   -> Asymptotic continuous Chi-Square test breaks down with small discordant sample sizes.")
        print("   -> Automatically routing to Exact Binomial McNemar Test (distribution-free exact p-value).")
        use_exact = True
        
    # 3. Execute McNemar Test
    print("\\n" + "=" * 70)
    if use_exact:
        print("--> [VERDICT] Executing Exact Binomial McNemar Test")
        print("=" * 70)
        res = mcnemar(table, exact=True)
        stat_name = "Exact Binomial"
        stat_val = np.nan
        p_val = float(res.pvalue)
    else:
        print("--> [VERDICT] Executing Edwards' Continuity-Corrected McNemar Chi-Square Test")
        print("=" * 70)
        res = mcnemar(table, exact=False, correction=True)
        stat_name = "Edwards Chi2(1)"
        stat_val = float(res.statistic)
        p_val = float(res.pvalue)
        
    # 4. Effect Sizes & Transition Metrics
    p_pre = (a + b) / n_total
    p_post = (a + c_val) / n_total
    marginal_shift = p_post - p_pre
    cond_or = float(c_val / b) if b > 0 else np.nan
    
    if b > 0 and c_val > 0:
        se_log = np.sqrt(1/b + 1/c_val)
        ci_lower = float(np.exp(np.log(cond_or) - 1.96 * se_log))
        ci_upper = float(np.exp(np.log(cond_or) + 1.96 * se_log))
    else:
        ci_lower, ci_upper = np.nan, np.nan
        
    if not np.isnan(stat_val):
        print(f"{stat_name}:               chi2 = {stat_val:.4f}")
    print(f"Two-Sided p-value:          p = {p_val:.4e}")
    print(f"Pre-Test Positive Rate:     {p_pre*100:.1f}%")
    print(f"Post-Test Positive Rate:    {p_post*100:.1f}%")
    print(f"Marginal Shift (Post-Pre):  {marginal_shift*100:+.1f}%")
    if not np.isnan(cond_or):
        print(f"Conditional Odds Ratio:     OR = {cond_or:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
        
    is_sig = (p_val < alpha)
    print(f"\\nConclusion: {'Statistically significant marginal shift (Reject H0)' if is_sig else 'No significant marginal shift (Fail to reject H0)'} at alpha = {alpha}.")
    
    return {
        "method": "exact_binomial" if use_exact else "edwards_chi2",
        "statistic": stat_val,
        "p_value": p_val,
        "marginal_shift": marginal_shift,
        "conditional_or": cond_or,
        "ci_95": (ci_lower, ci_upper),
        "is_significant": is_sig
    }

# Example Run:
# table_demo = np.array([[15, 5], [25, 55]])
# run_mcnemar_pipeline(table_demo)
""", language="python")


def render_pearson_correlation_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for Pearson Correlation Coefficient."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow for Pearson Correlation
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Linearity, Normality & Bivariate Outliers)", expanded=False):
        st.markdown(r"""
        Pearson's correlation coefficient ($r$) evaluates the strength and direction of a **linear relationship** between two continuous variables.
        
        Before trusting Pearson's $r$ and its $p$-value, verify the foundational assumptions in Excel:
        1. **Continuous Metric Variables:** Both variables $X$ and $Y$ must be measured on an interval or ratio scale.
        2. **Straight-Line Linearity:** The relationship must follow a straight line trajectory. In Excel, insert a **Scatter Plot** (`Insert` ➔ `Scatter (X, Y)`) and add a linear trendline. If the relationship curves, Pearson's $r$ severely underestimates association strength.
        3. **Bivariate Normality:** Both variables should be approximately normally distributed:
           * Skewness within $[-0.5, +0.5]$ (or at least $[-1.0, +1.0]$) via `=SKEW(range)`.
           * Excess Kurtosis within $[-1.0, +1.0]$ via `=KURT(range)`.
        4. **Absence of Influential Outliers:** A single extreme data point can artificially inflate or destroy Pearson's $r$. Verify using standardized $Z$-scores ($|Z| \le 3.0$) or Tukey's fences.
        * *If normality fails, severe outliers exist, or the relationship is non-linear monotonic:* **Switch to Spearman Rank Correlation (Step 3)!**
        """)

        st.markdown("##### 📋 1. Example Spreadsheet Data Layout (Excel Range: A1:D31)")
        st.markdown("""
        Enter your paired continuous observations into columns **`A`** ($X$) and **`B`** ($Y$). Standardized $Z$-scores in columns **`C`** and **`D`** identify high-leverage outliers:
        """)
        
        # Deterministic illustrative dataset of N=30
        np.random.seed(42)
        demo_x = np.array([
            12.5, 14.0, 15.2, 16.8, 17.5, 18.2, 19.0, 20.1, 21.4, 22.0,
            22.8, 23.5, 24.1, 25.0, 25.8, 26.4, 27.2, 28.0, 28.9, 29.5,
            30.2, 31.0, 31.8, 32.5, 33.4, 34.2, 35.0, 36.1, 37.0, 38.5
        ])
        demo_y = np.array([
            75.2, 78.0, 81.5, 84.0, 87.2, 89.0, 91.4, 94.0, 96.5, 98.2,
            100.4, 102.1, 104.5, 107.0, 109.2, 111.0, 113.4, 115.8, 118.0, 120.2,
            122.5, 124.8, 127.0, 129.5, 131.8, 134.0, 136.2, 139.0, 141.5, 145.0
        ])
        z_x = np.round((demo_x - np.mean(demo_x)) / np.std(demo_x, ddof=1), 2)
        z_y = np.round((demo_y - np.mean(demo_y)) / np.std(demo_y, ddof=1), 2)
        
        df_pearson_data_layout = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(10)] + ["...", "Row 31"],
            "Col_A (Predictor X: Study Hours)": [f"{v:.1f}" for v in demo_x[:10]] + ["...", f"{demo_x[-1]:.1f}"],
            "Col_B (Outcome Y: Exam Score)": [f"{v:.1f}" for v in demo_y[:10]] + ["...", f"{demo_y[-1]:.1f}"],
            "Col_C (Z-Score X: Formula)": [f"{z:+.2f}" for z in z_x[:10]] + ["...", f"{z_x[-1]:+.2f}"],
            "Col_D (Z-Score Y: Formula)": [f"{z:+.2f}" for z in z_y[:10]] + ["...", f"{z_y[-1]:+.2f}"]
        })
        render_table(df_pearson_data_layout, height=220)
        st.caption("Excel Formulas: Cell C2: `=(A2 - AVERAGE($A$2:$A$31)) / STDEV.S($A$2:$A$31)` | Cell D2: `=(B2 - AVERAGE($B$2:$B$31)) / STDEV.S($B$2:$B$31)`")

        st.markdown("##### 📊 2. Assumption Diagnostic Checks & Rule Evaluation")
        df_pearson_assumptions = pd.DataFrame({
            "Correlation_Assumption": [
                "1. Continuous Variable Scales",
                "2. Sample Size Adequacy (N)",
                "3. Variable X Distribution (Skewness)",
                "4. Variable Y Distribution (Skewness)",
                "5. Variable X Distribution (Kurtosis)",
                "6. Variable Y Distribution (Kurtosis)",
                "7. Bivariate Outliers (|Z| > 3.0)",
                "8. Linearity Trajectory"
            ],
            "Excel_Formula_or_Workflow": [
                "Verify both columns contain numeric interval/ratio data",
                "Total Count: =COUNT(A2:A31)",
                "Skewness X: =SKEW(A2:A31)",
                "Skewness Y: =SKEW(B2:B31)",
                "Kurtosis X: =KURT(A2:A31)",
                "Kurtosis Y: =KURT(B2:B31)",
                "Outlier Count: =COUNTIF(C2:C31, \">3\") + COUNTIF(C2:C31, \"<-3\") + COUNTIF(D2:D31, \">3\") + COUNTIF(D2:D31, \"<-3\")",
                "Scatter Plot: Insert -> Scatter (X, Y) -> Add Linear Trendline -> Display R-squared"
            ],
            "Acceptable_Benchmark": [
                "Interval or ratio continuous scales",
                "N >= 20 (ideally N >= 30)",
                "Between -0.50 and +0.50 (normal)",
                "Between -0.50 and +0.50 (normal)",
                "Between -1.00 and +1.00 (mesokurtic)",
                "Between -1.00 and +1.00 (mesokurtic)",
                "0 extreme outliers (|Z| > 3.0)",
                "Linear cloud of points without visible curvature"
            ],
            "Interpretation_Verdict": [
                "PASS: Both study hours and exam scores are continuous metrics.",
                "PASS: N = 30 observations provides adequate statistical power.",
                "PASS: SKEW(A2:A31) = +0.09 is near zero (symmetric).",
                "PASS: SKEW(B2:B31) = +0.08 is near zero (symmetric).",
                "PASS: KURT(A2:A31) = -1.18 is close to normal flat peak.",
                "PASS: KURT(B2:B31) = -1.17 is close to normal flat peak.",
                "PASS: 0 data points exceed |Z| = 3.0; no influential outliers distorting slope.",
                "PASS: Scatter plot displays a clean linear upward slope without curvilinearity."
            ]
        })
        render_table(df_pearson_assumptions, height=240)

    # STEP 2: IMPLEMENT PEARSON CORRELATION IN EXCEL
    with st.expander("📌 Step 2: Implement Pearson Correlation in Excel (Formulas, Hypothesis Testing & Fisher's z CI)", expanded=False):
        st.markdown(r"""
        Excel computes sample Pearson's $r$ using the native `=CORREL` or `=PEARSON` functions.
        To test statistical significance ($H_0: \rho = 0$) and compute the 95% confidence interval, use Excel's student-$t$ and Fisher's $z$ transformation functions.

        **Core Mathematical Formulas:**
        * **Pearson Correlation Coefficient ($r$):**
          $$r = \frac{\sum (X - \bar{X})(Y - \bar{Y})}{\sqrt{\sum (X - \bar{X})^2 \sum (Y - \bar{Y})^2}}$$
        * **Coefficient of Determination ($R^2$):** $R^2 = r^2$ (proportion of variance explained).
        * **Significance $t$-Statistic ($df = N - 2$):**
          $$t = \frac{r \sqrt{N - 2}}{\sqrt{1 - r^2}}$$
        * **Two-Tailed $p$-Value:** `=T.DIST.2T(ABS(t), df)`
        * **Fisher's $z$ Transformation for 95% CI:**
          $$z = \frac{1}{2} \ln\left(\frac{1 + r}{1 - r}\right) = \text{FISHER}(r), \quad \text{SE}_z = \frac{1}{\sqrt{N - 3}}$$
          $$\text{CI limits} = \text{FISHERINV}(z \pm 1.96 \cdot \text{SE}_z)$$
        """)

        st.markdown("##### 📋 Complete Excel Worksheet Cell Map for Correlation Metrics (Columns F:G)")
        st.markdown("""
        The table below shows exactly which cell stores each calculation, the required Excel formula, the sample output value, and the plain-English interpretation:
        """)
        df_pearson_calc_map = pd.DataFrame({
            "Target_Statistic": [
                "1. Sample Size (N)",
                "2. Pearson Correlation Coefficient (r)",
                "3. Alternative Correlation Formula",
                "4. Coefficient of Determination (R²)",
                "5. Degrees of Freedom (df)",
                "6. Hypothesis Test t-Statistic",
                "7. Two-Tailed p-Value",
                "8. Critical t-Threshold (α = 0.05)",
                "9. Fisher's z Transformation",
                "10. Standard Error of Fisher's z",
                "11. 95% Confidence Interval Lower Limit",
                "12. 95% Confidence Interval Upper Limit",
                "13. OLS Regression Slope (b1)",
                "14. OLS Regression Intercept (b0)"
            ],
            "Excel_Cell": [
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "F9",
                "F10",
                "F11",
                "F12",
                "F13",
                "F14",
                "F15"
            ],
            "Excel_Formula": [
                "=COUNT(A2:A31)",
                "=CORREL(A2:A31, B2:B31)",
                "=PEARSON(A2:A31, B2:B31)",
                "=RSQ(B2:B31, A2:A31)",
                "=F2 - 2",
                "=F3 * SQRT(F6) / SQRT(1 - F3^2)",
                "=T.DIST.2T(ABS(F7), F6)",
                "=T.INV.2T(0.05, F6)",
                "=FISHER(F3)",
                "=1 / SQRT(F2 - 3)",
                "=FISHERINV(F10 - 1.96 * F11)",
                "=FISHERINV(F10 + 1.96 * F11)",
                "=SLOPE(B2:B31, A2:A31)",
                "=INTERCEPT(B2:B31, A2:A31)"
            ],
            "Sample_Output_Value": [
                "30",
                "+0.999",
                "+0.999",
                "0.998 (99.8%)",
                "28",
                "126.75",
                "0.0000000",
                "2.048",
                "3.800",
                "0.192",
                "+0.998",
                "+1.000",
                "+2.673",
                "+41.87"
            ],
            "Interpretation_Guidance": [
                "Total number of paired continuous observations.",
                "Strength and direction of linear association (-1.0 to +1.0). Near +1.0 = near perfect positive.",
                "Native formula identical to =CORREL; calculates sample Pearson r.",
                "Proportion of variance in Y explained by linear relationship with X (99.8%).",
                "Degrees of freedom: N - 2.",
                "t-statistic testing null hypothesis H0: rho = 0.",
                "p-value testing H0; p < 0.001 indicates highly significant linear correlation.",
                "Two-tailed critical t; since |126.75| > 2.048, reject H0.",
                "Normalizes the skewed sampling distribution of r.",
                "Theoretical standard error of Fisher's z metric: 1 / SQRT(N - 3).",
                "Lower bound of 95% confidence interval for population correlation rho.",
                "Upper bound of 95% confidence interval for population correlation rho.",
                "Expected increase in Exam Score per 1 additional Study Hour (+2.67 points).",
                "Predicted Exam Score when Study Hours = 0."
            ]
        })
        render_table(df_pearson_calc_map, height=280)

        st.markdown(r"""
        ---

        **Step-by-Step Instructions for Pearson Correlation in Excel:**
        1. **Compute Correlation Coefficient ($r$):** In cell **`F3`**, enter `=CORREL(A2:A31, B2:B31)`. Returns the sample correlation coefficient $r$.
        2. **Compute Coefficient of Determination ($R^2$):** In cell **`F5`**, enter `=RSQ(B2:B31, A2:A31)` (or `=F3^2`). Indicates the percentage of shared variance.
        3. **Compute Significance $t$-Statistic:** In cell **`F7`**, enter `=F3 * SQRT(F6) / SQRT(1 - F3^2)`.
        4. **Compute Two-Tailed $p$-Value:** In cell **`F8`**, enter `=T.DIST.2T(ABS(F7), F6)`.
           * If $p < 0.05$, reject $H_0$: the correlation is statistically significant!
        5. **Compute 95% Confidence Interval via Fisher's $z$:**
           * In cell **`F10`**, enter `=FISHER(F3)` (converts $r$ to $z$).
           * In cell **`F11`**, enter `=1 / SQRT(F2 - 3)` ($\text{SE}_z$).
           * In cell **`F12`**, enter `=FISHERINV(F10 - 1.96 * F11)` (Lower 95% limit).
           * In cell **`F13`**, enter `=FISHERINV(F10 + 1.96 * F11)` (Upper 95% limit).
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Tests in Excel (Spearman Rank Correlation, Toolpak & Transformations)", expanded=False):
        st.markdown(r"""
        **1. Alternative 1: Spearman Rank Correlation ($\rho$) (When Normality Fails or Outliers are Present):**
        * If either variable violates normality, has extreme outliers, or follows a non-linear monotonic curve, Pearson's $r$ is compromised.
        * **How to compute Spearman's $\rho$ in Excel:**
          1. Create ranked columns in `C` and `D`:
             * Cell C2: `=RANK.AVG(A2, $A$2:$A$31, 1)` (Ranks Variable X ascending)
             * Cell D2: `=RANK.AVG(B2, $B$2:$B$31, 1)` (Ranks Variable Y ascending)
             * Drag formulas down to row 31.
          2. Compute Pearson correlation on the ranks:
             ```excel
             =CORREL(C2:C31, D2:D31)
             ```
          * This yields Spearman's rank correlation coefficient $\rho$!

        **2. Alternative 2: Excel Data Analysis Toolpak Correlation Matrix:**
        * When you have 3 or more continuous variables and wish to inspect an all-pairs correlation table:
          1. Navigate to **Data** tab ➔ **Data Analysis** ➔ select **Correlation** ➔ click **OK**.
          2. In **Input Range**, select columns `A1:C31`. Check **Labels in first row**.
          3. Set Output Range to an empty cell.
          * Excel automatically outputs the complete correlation matrix!

        **3. Alternative 3: Non-Linear Monotonic Transformations:**
        * If the scatter plot displays exponential or logarithmic curvature (e.g. diminishing returns):
          * Apply natural log: `=LN(A2)` in an adjacent column.
          * Re-test Pearson correlation on `=CORREL(LN_Range, B2:B31)`.
        """)


def render_pearson_correlation_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python guide for Pearson Correlation Coefficient,
    including bivariate normality verification, outlier detection, and conditional assurance/warning/switching statements.
    """
    st.markdown("""
    ### 🐍 Python Implementation Workflow for Pearson Correlation
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Bivariate Normality, Outliers & Linearity)", expanded=False):
        st.code("""import numpy as np
from scipy import stats

# 1. Define Paired Continuous Observations (X and Y)
np.random.seed(42)
x = np.round(np.random.normal(loc=25.0, scale=6.0, size=35), 2)
y = np.round(40.0 + 2.4 * x + np.random.normal(loc=0.0, scale=5.0, size=35), 2)

# 2. Univariate Normality Diagnostics (Shapiro-Wilk Test)
shapiro_x = stats.shapiro(x)
shapiro_y = stats.shapiro(y)
skew_x, kurt_x = float(stats.skew(x)), float(stats.kurtosis(x))
skew_y, kurt_y = float(stats.skew(y)), float(stats.kurtosis(y))

# 3. Outlier Detection via Standardized Z-Scores
z_x = np.abs(stats.zscore(x))
z_y = np.abs(stats.zscore(y))
outliers_count = int(np.sum((z_x > 3.0) | (z_y > 3.0)))

print(f"Sample Size: N = {len(x)}")
print(f"Variable X: Shapiro W = {shapiro_x.statistic:.4f}, p = {shapiro_x.pvalue:.4f} (Skew = {skew_x:+.2f}, Kurt = {kurt_x:+.2f})")
print(f"Variable Y: Shapiro W = {shapiro_y.statistic:.4f}, p = {shapiro_y.pvalue:.4f} (Skew = {skew_y:+.2f}, Kurt = {kurt_y:+.2f})")
print(f"Extreme Bivariate Outliers (|Z| > 3.0): {outliers_count}")
""", language="python")

        st.markdown(r"""
        **🔍 How to Interpret the Assumption Diagnostic Output:**
        * **Normality Requirement:** Pearson's correlation assumes both variables follow a normal distribution. If Shapiro-Wilk $p \ge 0.05$ on both $X$ and $Y$, bivariate normality is supported.
        * **Outlier Sensitivity:** Pearson's $r$ is extraordinarily sensitive to isolated outliers that act as high-leverage points. If $|Z| > 3.0$ points exist, Pearson's $r$ is distorted.
        * **Decision Rule:**
          * If both variables are normal ($p \ge 0.05$) and no extreme outliers $\rightarrow$ **Assumptions Satisfied**. Proceed with Pearson correlation (Step 2).
          * If non-normal ($p < 0.01$) or extreme outliers exist $\rightarrow$ **Assumptions Violated**. Switch to Spearman Rank Correlation (Step 3).
        """)

    # STEP 2: IMPLEMENT PEARSON CORRELATION IN PYTHON
    with st.expander("📌 Step 2: Implement Pearson Correlation in Python", expanded=False):
        st.code("""# 1. Compute Pearson Correlation, Two-Sided p-Value, and 95% CI
res = stats.pearsonr(x, y)
r = res.statistic
p_val = res.pvalue
ci_lower, ci_upper = res.confidence_interval(confidence_level=0.95)

# 2. Coefficient of Determination (R-squared) & Significance t-Statistic
r_squared = r ** 2
n = len(x)
df = n - 2
t_stat = r * np.sqrt(df) / np.sqrt(1 - r_squared)

# 3. OLS Linear Regression Trajectory
reg = stats.linregress(x, y)

print(f"Pearson Correlation (r):       r = {r:+.4f}")
print(f"95% Confidence Interval:       [{ci_lower:+.4f}, {ci_upper:+.4f}] (Fisher's z)")
print(f"Hypothesis Test:               t({df}) = {t_stat:.4f}, p = {p_val:.4e}")
print(f"Coefficient of Determination:  R2 = {r_squared*100:.1f}% shared variance")
print(f"OLS Linear Equation:           Y = {reg.intercept:.2f} + {reg.slope:.2f} * X")
""", language="python")

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output:**
        * **Pearson's $r$ Magnitude (Cohen benchmarks):** $|r| \ge 0.10$ small, $\ge 0.30$ medium, $\ge 0.50$ large linear association.
        * **Statistical Significance:** If $p < 0.05$, reject $H_0: \rho = 0$. The linear correlation is statistically significant.
        * **95% Confidence Interval:** Computed via Fisher's $z$ transformation. If the interval excludes $0.0$, the association is statistically significant at $\alpha = 0.05$.
        * **$R^2$ (Explained Variance):** Represents the percentage of variance in $Y$ that is explained by linear dependence on $X$.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Tests in Python (Spearman, Kendall & Robust Correlation)", expanded=False):
        st.code("""# Alternative 1: Spearman Rank Correlation (Non-parametric / Monotonic)
rho, p_spearman = stats.spearmanr(x, y)
print("=== ALTERNATIVE 1: SPEARMAN RANK CORRELATION (ROBUST) ===")
print(f"Spearman rho: {rho:+.4f}, p-value: {p_spearman:.4e}")

# Alternative 2: Kendall's Tau-b (Small Samples or Heavy Tied Ranks)
tau, p_kendall = stats.kendalltau(x, y)
print("\\n=== ALTERNATIVE 2: KENDALL'S TAU-B (TIED RANKS) ===")
print(f"Kendall tau:  {tau:+.4f}, p-value: {p_kendall:.4e}")

# Alternative 3: Robust Huber / Theil-Sen Median Regression (Outlier-Resistant)
slope_ts, intercept_ts, _, _ = stats.mstats.theilslopes(y, x)
print("\\n=== ALTERNATIVE 3: THEIL-SEN ROBUST MEDIAN FIT ===")
print(f"Robust Slope: {slope_ts:.4f}, Intercept: {intercept_ts:.4f}")
""", language="python")

        st.markdown(r"""
        **🔍 When to Switch to Alternative Tests:**
        * **Use Spearman Rank Correlation:** When either variable is non-normal, contains outliers, or displays a monotonic non-linear pattern. Spearman correlates ranks rather than raw values, neutralizing outlier distortion.
        * **Use Kendall's Tau-b:** When sample size is small ($N < 30$) or data contains a high proportion of tied ranks ($> 25\%$).
        * **Use Theil-Sen Robust Regression:** When interested in predicting $Y$ from $X$ in the presence of extreme bivariate outliers without discarding data points.
        """)

    # STEP 4: CONSOLIDATED PYTHON PIPELINE SCRIPT WITH CONDITIONAL ASSURANCE/WARNING/SWITCHING PRINTS
    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
from scipy import stats

def run_pearson_pipeline(x, y, alpha=0.05):
    \"\"\"Automated Bivariate Correlation Pipeline with normality verification,
    outlier detection, conditional assurance/warning printouts, and automated
    switching to Spearman Rank Correlation when assumptions fail.
    \"\"\"
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Remove any paired NaN values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    n = len(x)
    
    print("=" * 70)
    print(f"=== BIVARIATE CORRELATION ANALYSIS PIPELINE (Sample Size N = {n}) ===")
    print("=" * 70)
    
    if n < 3:
        print("[INAPPROPRIATE - FAIL] Sample size N < 3. Correlation requires at least 3 paired observations!")
        return {"method": "error", "message": "N < 3"}
        
    # 1. Normality Diagnostics (Shapiro-Wilk)
    shapiro_x = stats.shapiro(x) if n >= 3 else None
    shapiro_y = stats.shapiro(y) if n >= 3 else None
    p_norm_x = shapiro_x.pvalue if shapiro_x else 1.0
    p_norm_y = shapiro_y.pvalue if shapiro_y else 1.0
    min_norm_p = min(p_norm_x, p_norm_y)
    
    # 2. Outlier Diagnostics (Z-Scores)
    z_x = np.abs(stats.zscore(x))
    z_y = np.abs(stats.zscore(y))
    outlier_count = int(np.sum((z_x > 3.0) | (z_y > 3.0)))
    
    print(f"Variable X: Mean = {np.mean(x):.2f}, Std = {np.std(x, ddof=1):.2f}, Shapiro p = {p_norm_x:.4f}")
    print(f"Variable Y: Mean = {np.mean(y):.2f}, Std = {np.std(y, ddof=1):.2f}, Shapiro p = {p_norm_y:.4f}")
    print(f"Bivariate Outliers (|Z| > 3.0): {outlier_count} observations")
    
    # 3. Assumption Evaluation & Methodological Routing
    print("\\n--- ASSUMPTION DIAGNOSTICS & METHODOLOGICAL FIT ---")
    if min_norm_p >= 0.05 and outlier_count == 0:
        print(f"[ASSURANCE - PASS] Bivariate normality satisfied (min Shapiro p = {min_norm_p:.4f} >= 0.05) and 0 severe outliers detected.")
        print("   -> Pearson Correlation Coefficient (r) is fully valid and optimal.")
        use_pearson = True
    elif min_norm_p < 0.01 or outlier_count > 0:
        print(f"[INAPPROPRIATE - FAIL] Assumption violations detected: {'Non-normality (p < 0.01)' if min_norm_p < 0.01 else ''} {f'and {outlier_count} severe outliers' if outlier_count > 0 else ''}.")
        print("   -> Pearson's r is heavily distorted by extreme values and distribution tails.")
        print("   -> Automatically switching to Spearman Rank Correlation (rho).")
        use_pearson = False
    else:
        print(f"[WARNING] Mild departure from normality detected (min Shapiro p = {min_norm_p:.4f} in [0.01, 0.05]).")
        print("   -> Pearson's r reported with caution. Review Spearman's rho to verify estimate consistency.")
        use_pearson = True
        
    # 4. Execute Analysis
    print("\\n" + "=" * 70)
    if use_pearson:
        print("--> [VERDICT] Reporting Pearson Linear Correlation Coefficient (r)")
        print("=" * 70)
        res = stats.pearsonr(x, y)
        r = float(res.statistic)
        p_val = float(res.pvalue)
        ci_low, ci_high = res.confidence_interval(confidence_level=0.95)
        r2 = r ** 2
        df = n - 2
        t_stat = r * np.sqrt(df) / np.sqrt(1 - r2)
        
        print(f"Pearson Correlation (r):       r = {r:+.4f}")
        print(f"95% Confidence Interval:       [{ci_low:+.4f}, {ci_high:+.4f}] (Fisher's z)")
        print(f"Hypothesis Test:               t({df}) = {t_stat:.4f}, p = {p_val:.4e}")
        print(f"Shared Variance (R-squared):   R2 = {r2*100:.1f}%")
        
        is_sig = (p_val < alpha)
        print(f"Conclusion: {'Statistically significant linear correlation (Reject H0)' if is_sig else 'No significant correlation (Fail to reject H0)'} at alpha = {alpha}.")
        
        return {
            "method": "pearson_correlation",
            "r": r,
            "r_squared": r2,
            "p_value": p_val,
            "ci_95": (ci_low, ci_high),
            "is_significant": is_sig
        }
    else:
        print("--> [VERDICT] Switching to Non-Parametric Spearman Rank Correlation (rho)")
        print("=" * 70)
        res_spearman = stats.spearmanr(x, y)
        rho = float(res_spearman.statistic)
        p_val = float(res_spearman.pvalue)
        
        print(f"Spearman Rank Correlation:     rho = {rho:+.4f}")
        print(f"Two-Sided p-value:             p = {p_val:.4e}")
        print(f"Shared Rank Variance:          rho^2 = {rho**2*100:.1f}%")
        
        is_sig = (p_val < alpha)
        print(f"Conclusion: {'Statistically significant monotonic correlation (Reject H0)' if is_sig else 'No significant correlation (Fail to reject H0)'} at alpha = {alpha}.")
        
        return {
            "method": "spearman_correlation_fallback",
            "rho": rho,
            "p_value": p_val,
            "is_significant": is_sig
        }

# Example Run:
# x_demo = np.array([12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
# y_demo = np.array([25, 29, 34, 38, 42, 47, 51, 55, 60, 66])
# run_pearson_pipeline(x_demo, y_demo)
""", language="python")


def render_spearman_correlation_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for Spearman Rank Correlation Coefficient."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow for Spearman Rank Correlation
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Monotonicity, Variable Scales & Tied Ranks Audit)", expanded=False):
        st.markdown(r"""
        Spearman's rank correlation coefficient ($\rho$) evaluates the strength and direction of a **monotonic relationship** between two variables using their ranked values.
        Unlike Pearson's $r$, it does not assume normal distributions or a straight-line relationship, and it is robust to extreme outliers.
        
        Before trusting Spearman's $\rho$, verify the foundational assumptions in Excel:
        1. **Ordinal or Continuous Scales:** Both variables must be at least ordinal (ordered ratings, Likert scores, or continuous metrics).
        2. **Monotonic Trajectory:** As $X$ increases, $Y$ consistently increases (or consistently decreases). The relationship can curve (e.g. exponential growth, logarithmic saturation, diminishing returns), but it must not reverse direction (e.g. U-shaped or parabolic curves).
        3. **Tied Ranks Audit:** Spearman's test handles moderate ties smoothly using average rank positions (`=RANK.AVG`). However, if tied values exceed **$25\%$** of the dataset, rank variance is reduced, and **Kendall's $\tau_b$** is preferred.
        4. **Paired Observations:** Each row represents paired measurements from an independent subject.
        """)

        st.markdown("##### 📋 1. Example Spreadsheet Data & Rank Transformation Layout (Excel Range: A1:E26)")
        st.markdown("""
        Enter your paired observations into columns **`A`** ($X$) and **`B`** ($Y$). Compute fractional rank positions in columns **`C`** and **`D`**, and squared rank disparities in column **`E`**:
        """)
        
        # Deterministic illustrative dataset of N=25
        demo_x = np.array([
            10.2, 12.0, 13.5, 15.0, 16.8, 18.2, 19.5, 21.0, 22.4, 24.0,
            25.5, 27.0, 28.6, 30.0, 31.5, 33.0, 34.8, 36.0, 37.5, 39.0,
            40.5, 42.0, 43.8, 45.0, 47.5
        ])
        # Monotonic non-linear curved relationship (diminishing returns / log shape)
        demo_y = np.array([
            22.0, 28.5, 34.0, 39.2, 45.0, 49.5, 53.0, 57.2, 60.5, 63.8,
            67.0, 70.1, 72.8, 75.5, 78.0, 80.2, 82.5, 84.4, 86.0, 87.8,
            89.5, 91.0, 92.5, 93.8, 95.5
        ])
        rank_x = np.arange(1, 26, dtype=float)
        rank_y = np.arange(1, 26, dtype=float)
        d_sq = (rank_x - rank_y) ** 2
        
        df_spearman_data_layout = pd.DataFrame({
            "Excel_Row": [f"Row {i+2}" for i in range(10)] + ["...", "Row 26"],
            "Col_A (Predictor X: Aptitude)": [f"{v:.1f}" for v in demo_x[:10]] + ["...", f"{demo_x[-1]:.1f}"],
            "Col_B (Outcome Y: Performance)": [f"{v:.1f}" for v in demo_y[:10]] + ["...", f"{demo_y[-1]:.1f}"],
            "Col_C (Rank X: =RANK.AVG)": [f"{r:.1f}" for r in rank_x[:10]] + ["...", f"{rank_x[-1]:.1f}"],
            "Col_D (Rank Y: =RANK.AVG)": [f"{r:.1f}" for r in rank_y[:10]] + ["...", f"{rank_y[-1]:.1f}"],
            "Col_E (Squared Diff dᵢ²: =(C2-D2)²)": [f"{d:.1f}" for d in d_sq[:10]] + ["...", f"{d_sq[-1]:.1f}"]
        })
        render_table(df_spearman_data_layout, height=220)
        st.caption("Excel Formulas: Cell C2: `=RANK.AVG(A2, $A$2:$A$26, 1)` | Cell D2: `=RANK.AVG(B2, $B$2:$B$26, 1)` | Cell E2: `=(C2 - D2)^2`")

        st.markdown("##### 📊 2. Assumption Diagnostic Checks & Rule Evaluation")
        df_spearman_assumptions = pd.DataFrame({
            "Spearman_Assumption": [
                "1. Measurement Scale Adequacy",
                "2. Sample Size Adequacy (N)",
                "3. Monotonic Directionality",
                "4. Tied Values in Variable X",
                "5. Tied Values in Variable Y",
                "6. Total Tied Ranks Proportion",
                "7. Independent Observations"
            ],
            "Excel_Formula_or_Workflow": [
                "Verify variables are numeric ordinal ratings or continuous metrics",
                "Total Count: =COUNT(A2:A26)",
                "Scatter Plot: Insert -> Scatter (X, Y) -> Inspect monotonic trajectory",
                "Count Duplicate X: =COUNT(A2:A26) - SUM(1 / COUNTIF(A2:A26, A2:A26))",
                "Count Duplicate Y: =COUNT(B2:B26) - SUM(1 / COUNTIF(B2:B26, B2:B26))",
                "Percentage Tied: Max duplicate count divided by total sample size N",
                "Verify each row represents an independent subject or unit"
            ],
            "Acceptable_Benchmark": [
                "At least ordinal scale",
                "N >= 10 (ideally N >= 20)",
                "Consistently moves in one direction (no U-turn)",
                "Tied scores <= 25%",
                "Tied scores <= 25%",
                "Total tied percentage <= 25% (if > 25%, switch to Kendall)",
                "Independent paired sampling"
            ],
            "Interpretation_Verdict": [
                "PASS: Both aptitude scores and performance ratings are continuous metrics.",
                "PASS: N = 25 observations satisfies sample size requirements.",
                "PASS: Scatter plot displays a monotonic upward curve without direction reversals.",
                "PASS: 0 duplicate values in X; all ranks are unique.",
                "PASS: 0 duplicate values in Y; all ranks are unique.",
                "PASS: Tied ranks proportion = 0.0% <= 25%; standard Spearman rho is fully optimal.",
                "PASS: Each observation represents an independent participant."
            ]
        })
        render_table(df_spearman_assumptions, height=220)

    # STEP 2: IMPLEMENT SPEARMAN RANK CORRELATION IN EXCEL
    with st.expander("📌 Step 2: Implement Spearman Rank Correlation in Excel (Formulas, Hypothesis Testing & Effect Sizes)", expanded=False):
        st.markdown(r"""
        Excel computes Spearman's rank correlation using two mathematically equivalent workflows:

        * **Method A (Standard & Recommended): Pearson Correlation on Fractional Ranks:**
          ```excel
          =CORREL(C2:C26, D2:D26)
          ```
          *(Directly applies `=CORREL` to the ranked columns generated by `=RANK.AVG`. This automatically and correctly handles tied ranks).*
        * **Method B (Classical Untied Formula): Squared Rank Differences ($d_i^2$):**
          $$\rho = 1 - \frac{6 \sum d_i^2}{N(N^2 - 1)}$$
          ```excel
          =1 - (6 * SUM(E2:E26)) / (COUNT(A2:A26) * (COUNT(A2:A26)^2 - 1))
          ```
        * **Significance $t$-Statistic ($df = N - 2$):**
          $$t = \frac{\rho \sqrt{N - 2}}{\sqrt{1 - \rho^2}}$$
        * **Two-Tailed $p$-Value:** `=T.DIST.2T(ABS(t), df)`
        * **Fisher's $z$ Transformation for 95% CI (Fieller adjustment):**
          $$z = \text{FISHER}(\rho), \quad \text{SE}_z = \frac{1.06}{\sqrt{N - 3}}, \quad \text{CI} = \text{FISHERINV}(z \pm 1.96 \cdot \text{SE}_z)$$
        """)

        st.markdown("##### 📋 Complete Excel Worksheet Cell Map for Spearman Metrics (Columns G:H)")
        st.markdown("""
        The table below shows exactly which cell stores each intermediate calculation, the required Excel formula, the sample output value, and the plain-English interpretation:
        """)
        df_spearman_calc_map = pd.DataFrame({
            "Target_Statistic": [
                "1. Sample Size (N)",
                "2. Spearman Rank Correlation (ρ, Method A)",
                "3. Sum of Squared Rank Diffs (Σ dᵢ²)",
                "4. Untied Classical Formula (Method B)",
                "5. Degrees of Freedom (df)",
                "6. Significance t-Statistic",
                "7. Two-Tailed p-Value",
                "8. Critical t-Threshold (α = 0.05)",
                "9. Fisher's z Transformation",
                "10. Adjusted Standard Error of z",
                "11. 95% Confidence Interval Lower Limit",
                "12. 95% Confidence Interval Upper Limit",
                "13. Shared Rank Variance (ρ²)"
            ],
            "Excel_Cell": [
                "G2",
                "G3",
                "G4",
                "G5",
                "G6",
                "G7",
                "G8",
                "G9",
                "G10",
                "G11",
                "G12",
                "G13",
                "G14"
            ],
            "Excel_Formula": [
                "=COUNT(A2:A26)",
                "=CORREL(C2:C26, D2:D26)",
                "=SUM(E2:E26)",
                "=1 - (6 * G4) / (G2 * (G2^2 - 1))",
                "=G2 - 2",
                "=G3 * SQRT(G6) / SQRT(1 - G3^2)",
                "=T.DIST.2T(ABS(G7), G6)",
                "=T.INV.2T(0.05, G6)",
                "=FISHER(G3)",
                "=1.06 / SQRT(G2 - 3)",
                "=FISHERINV(G10 - 1.96 * G11)",
                "=FISHERINV(G10 + 1.96 * G11)",
                "=G3^2"
            ],
            "Sample_Output_Value": [
                "25",
                "+1.000",
                "0.0",
                "+1.000",
                "23",
                "Undefined (rho=1.0)",
                "< 0.0001",
                "2.069",
                "3.800",
                "0.226",
                "+0.998",
                "+1.000",
                "100.0%"
            ],
            "Interpretation_Guidance": [
                "Total number of paired ranked observations.",
                "Strength and direction of monotonic rank association (-1.0 to +1.0). Fully accounts for ties.",
                "Sum of squared rank differences Σ(Rx - Ry)². Zero indicates perfect rank concordance.",
                "Classical formula without ties: 1 - 6*Σd² / (N*(N²-1)). Identical to =CORREL when untied.",
                "Degrees of freedom: N - 2 = 23.",
                "t-statistic testing null hypothesis H0: population rank correlation rho = 0.",
                "Two-tailed p-value testing H0; p < 0.001 indicates highly significant monotonic association.",
                "Critical t-threshold at alpha = 0.05; since |t| > 2.069, reject H0.",
                "Variance-stabilizing Fisher z transformation for rank correlation.",
                "Adjusted standard error of z using Fieller's rank correlation adjustment: 1.06 / SQRT(N - 3).",
                "Lower bound of 95% confidence interval for population rank correlation rho.",
                "Upper bound of 95% confidence interval for population rank correlation rho.",
                "Proportion of variance in rank order explained by monotonic relationship."
            ]
        })
        render_table(df_spearman_calc_map, height=280)

        st.markdown(r"""
        ---

        **Step-by-Step Instructions for Spearman Correlation in Excel:**
        1. **Create Rank Columns (`C2:D26`):**
           * In cell `C2`, enter `=RANK.AVG(A2, $A$2:$A$26, 1)`. Double-click to fill down to `C26`.
           * In cell `D2`, enter `=RANK.AVG(B2, $B$2:$B$26, 1)`. Double-click to fill down to `D26`.
        2. **Calculate Spearman Correlation via Method A (`G3`):**
           * In cell `G3`, enter `=CORREL(C2:C26, D2:D26)`. Returns Spearman's rank correlation coefficient $\rho$.
        3. **Calculate Squared Rank Differences (`E2:E26`):**
           * In cell `E2`, enter `=(C2 - D2)^2`. Fill down to `E26`.
           * In cell `G4`, sum the differences: `=SUM(E2:E26)`.
        4. **Compute Hypothesis Test $t$-Statistic & $p$-Value:**
           * Degrees of freedom in cell `G6`: `=G2 - 2`.
           * $t$-statistic in cell `G7`: `=G3 * SQRT(G6) / SQRT(1 - G3^2)`.
           * Two-tailed $p$-value in cell `G8`: `=T.DIST.2T(ABS(G7), G6)`.
           * If $p < 0.05$, reject $H_0$: the monotonic relationship is statistically significant!
        5. **Compute 95% Confidence Interval with Fieller Adjustment:**
           * In cell `G10`, enter `=FISHER(G3)`.
           * In cell `G11`, enter `=1.06 / SQRT(G2 - 3)` ($\text{SE}_z$).
           * In cell `G12`, enter `=FISHERINV(G10 - 1.96 * G11)` (Lower 95% limit).
           * In cell `G13`, enter `=FISHERINV(G10 + 1.96 * G11)` (Upper 95% limit).
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Tests in Excel (Kendall's Tau, Pearson Correlation & Quadratic Models)", expanded=False):
        st.markdown(r"""
        **1. Alternative 1: Kendall's Tau-b ($\tau_b$) (When Tied Ranks Exceed 25% or N < 20):**
        * If a dataset contains heavy tied ranks (e.g. Likert scales where many respondents pick identical numbers), Spearman's $\rho$ can slightly overestimate association strength.
        * Kendall's $\tau_b$ evaluates concordance directly: $\tau = (C - D) / \sqrt{(C + D + T_x)(C + D + T_y)}$.
        * It is better calibrated for small sample sizes and heavy ties.

        **2. Alternative 2: Pearson Correlation ($r$) (When Relationship is Confirmed Linear and Normal):**
        * If a scatter plot reveals a straight line (no curve) and both variables follow a normal distribution without outliers:
        ```excel
        =CORREL(A2:A26, B2:B26)
        ```
        * Pearson's $r$ provides higher statistical power than Spearman's $\rho$ when bivariate normality holds.

        **3. Alternative 3: Polynomial / Quadratic Models (When Relationship is Non-Monotonic):**
        * If the relationship forms a U-turn (e.g. stress vs performance, where performance rises then drops), Spearman rank correlation drops to near zero despite a strong relationship.
        * In Excel: Right-click scatter plot points ➔ **Add Trendline** ➔ select **Polynomial (Order 2)** ➔ check **Display Equation on chart** and **Display R-squared value on chart**.
        """)


def render_spearman_correlation_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Python guide for Spearman Rank Correlation Coefficient,
    including tied ranks audit, monotonic trajectory verification, and conditional assurance/warning/switching statements.
    """
    st.markdown("""
    ### 🐍 Python Implementation Workflow for Spearman Rank Correlation
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test** ➔ **Implement Alternative Test**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Monotonicity & Tied Ranks Audit)", expanded=False):
        st.code("""import numpy as np
from scipy import stats

# 1. Define Paired Observations (Continuous or Ordinal)
np.random.seed(42)
x = np.round(np.random.normal(loc=30, scale=8, size=30), 2)
# Monotonic non-linear curved relationship (diminishing returns)
y = np.round(15 + 20 * np.log(np.maximum(x, 1.0)) + np.random.normal(loc=0, scale=2, size=30), 2)

# 2. Sample Size and Tied Ranks Audit
n = len(x)
ties_x = int(n - len(np.unique(x)))
ties_y = int(n - len(np.unique(y)))
pct_ties_x = (ties_x / n) * 100.0
pct_ties_y = (ties_y / n) * 100.0
max_ties_pct = max(pct_ties_x, pct_ties_y)

print(f"Sample Size: N = {n}")
print(f"Variable X: {ties_x} tied values ({pct_ties_x:.1f}%)")
print(f"Variable Y: {ties_y} tied values ({pct_ties_y:.1f}%)")
print(f"Maximum Tied Ranks Proportion: {max_ties_pct:.1f}% (Benchmark: <= 25%)")
""", language="python")

        st.markdown(r"""
        **🔍 How to Interpret the Assumption Diagnostic Output:**
        * **Monotonic Relationship:** Spearman's $\rho$ measures monotonic consistency. It does not require a straight line; curved trajectories (e.g. logarithmic or exponential) are evaluated accurately.
        * **Tied Ranks Threshold:**
          * If tied ranks $\le 25\% \rightarrow$ **Assumptions Satisfied**. Spearman's $\rho$ with average rank assignment is optimal.
          * If tied ranks $> 25\%$ or $N < 30 \rightarrow$ **Heavy Ties Warning**. Kendall's $\tau_b$ is recommended for better calibrated standard errors.
        """)

    # STEP 2: IMPLEMENT SPEARMAN RANK CORRELATION IN PYTHON
    with st.expander("📌 Step 2: Implement Spearman Rank Correlation in Python", expanded=False):
        st.code("""# 1. Compute Spearman Rank Correlation and Two-Sided p-Value
res = stats.spearmanr(x, y)
rho = res.statistic
p_val = res.pvalue

# 2. Significance t-Statistic & Fieller Adjusted 95% Confidence Interval
df = n - 2
t_stat = rho * np.sqrt(df) / np.sqrt(1 - rho**2) if abs(rho) < 1.0 else np.inf

# Fisher z transformation with Fieller adjustment (SE_z = 1.06 / sqrt(n - 3))
z = np.arctanh(np.clip(rho, -0.99999, 0.99999))
se_z = 1.06 / np.sqrt(n - 3)
ci_lower = float(np.tanh(z - 1.96 * se_z))
ci_upper = float(np.tanh(z + 1.96 * se_z))

print(f"Spearman Rank Correlation (rho):  rho = {rho:+.4f}")
print(f"95% Confidence Interval:          [{ci_lower:+.4f}, {ci_upper:+.4f}] (Fieller z)")
print(f"Hypothesis Test:                  t({df}) = {t_stat:.4f}, p = {p_val:.4e}")
print(f"Shared Rank Variance:             rho^2 = {rho**2*100:.1f}%")
""", language="python")

        st.markdown(r"""
        **🔍 How to Evaluate & Interpret the Output:**
        * **Spearman's $\rho$ Magnitude (Cohen benchmarks):** $|\rho| \ge 0.10$ small, $\ge 0.30$ medium, $\ge 0.50$ large monotonic association.
        * **Statistical Decision:** If $p < 0.05$, reject $H_0: \rho = 0$. There is a statistically significant monotonic association.
        * **95% Confidence Interval:** Computed via Fisher's $z$ with the Fieller adjustment ($1.06 / \sqrt{N - 3}$). If the interval excludes $0.0$, the association is statistically significant.
        * **Shared Rank Variance ($\rho^2$):** Measures the proportion of variance in rank order shared between the two variables.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Tests in Python (Kendall's Tau-b, Pearson's r & Distance Correlation)", expanded=False):
        st.code("""# Alternative 1: Kendall's Tau-b (Recommended for Heavy Ties > 25% or Small N)
res_tau = stats.kendalltau(x, y)
print("=== ALTERNATIVE 1: KENDALL'S TAU-B (TIED RANKS) ===")
print(f"Kendall tau:  {res_tau.statistic:+.4f}, p-value = {res_tau.pvalue:.4e}")

# Alternative 2: Pearson Linear Correlation (When Relationship is Confirmed Linear & Normal)
res_pearson = stats.pearsonr(x, y)
print("\\n=== ALTERNATIVE 2: PEARSON LINEAR CORRELATION ===")
print(f"Pearson r:    {res_pearson.statistic:+.4f}, p-value = {res_pearson.pvalue:.4e}")

# Alternative 3: Polynomial Quadratic Regression (When Relationship is U-Shaped Non-Monotonic)
poly_coeffs = np.polyfit(x, y, deg=2)
print("\\n=== ALTERNATIVE 3: POLYNOMIAL QUADRATIC FIT (NON-MONOTONIC) ===")
print(f"Quadratic:    Y = {poly_coeffs[0]:.4f}*X^2 + {poly_coeffs[1]:.4f}*X + {poly_coeffs[2]:.4f}")
""", language="python")

        st.markdown(r"""
        **🔍 When to Switch to Alternative Tests:**
        * **Use Kendall's Tau-b:** When data contains many duplicate values (tied ranks $> 25\%$) or when sample size is small ($N < 30$). Kendall's $\tau_b$ provides an unbiased estimate with better small-sample properties.
        * **Use Pearson's r:** When the relationship is straight-line linear and both variables follow a bivariate normal distribution.
        * **Use Polynomial Models:** When the scatterplot shows a U-shaped or inverted U-shaped curve. Monotonic rank correlation fails on non-monotonic data.
        """)

    # STEP 4: CONSOLIDATED PYTHON PIPELINE SCRIPT WITH CONDITIONAL ASSURANCE/WARNING/SWITCHING PRINTS
    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
from scipy import stats

def run_spearman_pipeline(x, y, alpha=0.05):
    \"\"\"Automated Non-Parametric Correlation Pipeline with tied rank auditing,
    monotonicity checks, conditional assurance/warning printouts, and automated
    comparison with Kendall's Tau-b and Pearson's r.
    \"\"\"
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Remove any paired missing values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    n = len(x)
    
    print("=" * 70)
    print(f"=== SPEARMAN RANK CORRELATION PIPELINE (Sample Size N = {n}) ===")
    print("=" * 70)
    
    if n < 3:
        print("[INAPPROPRIATE - FAIL] Sample size N < 3. Correlation requires at least 3 paired observations!")
        return {"method": "error", "message": "N < 3"}
        
    # 1. Tied Ranks Audit
    ties_x = int(n - len(np.unique(x)))
    ties_y = int(n - len(np.unique(y)))
    pct_ties_x = (ties_x / n) * 100.0
    pct_ties_y = (ties_y / n) * 100.0
    max_ties_pct = max(pct_ties_x, pct_ties_y)
    
    print(f"Variable X: {ties_x} tied observations ({pct_ties_x:.1f}%)")
    print(f"Variable Y: {ties_y} tied observations ({pct_ties_y:.1f}%)")
    print(f"Maximum Tied Ranks Proportion: {max_ties_pct:.1f}%")
    
    # 2. Check for Severe Non-Monotonicity (U-turn check via quadratic fit)
    linear_fit = np.polyfit(x, y, deg=1)
    quad_fit = np.polyfit(x, y, deg=2)
    y_pred_lin = np.polyval(linear_fit, x)
    y_pred_quad = np.polyval(quad_fit, x)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2_lin = 1 - (np.sum((y - y_pred_lin)**2) / ss_tot) if ss_tot > 0 else 0
    r2_quad = 1 - (np.sum((y - y_pred_quad)**2) / ss_tot) if ss_tot > 0 else 0
    
    is_strongly_parabolic = (r2_quad > r2_lin + 0.35) and (abs(quad_fit[0]) > 0.05)
    
    # 3. Assumption Diagnostics & Routing
    print("\\n--- ASSUMPTION DIAGNOSTICS & METHODOLOGICAL FIT ---")
    if is_strongly_parabolic:
        print("[INAPPROPRIATE - FAIL] Strong non-monotonic U-shaped relationship detected!")
        print("   -> Monotonic rank correlation fails when the relationship reverses direction.")
        print("   -> Consider polynomial quadratic regression or distance correlation.")
    elif max_ties_pct > 25.0 or n < 30:
        print(f"[WARNING] High proportion of tied ranks ({max_ties_pct:.1f}% > 25%) or small sample size (N = {n} < 30).")
        print("   -> Spearman's rho remains valid, but Kendall's Tau-b is recommended for better calibrated error rates.")
    else:
        print(f"[ASSURANCE - PASS] Low tied ranks proportion ({max_ties_pct:.1f}% <= 25%) and monotonic progression.")
        print("   -> Spearman Rank Correlation Coefficient (rho) is fully optimal and robust.")
        
    # 4. Execute Spearman Rank Correlation
    print("\\n" + "=" * 70)
    print("--> [VERDICT] Reporting Spearman Rank Correlation Coefficient (rho)")
    print("=" * 70)
    
    res = stats.spearmanr(x, y)
    rho = float(res.statistic)
    p_val = float(res.pvalue)
    
    # 95% Confidence Interval via Fisher's z with Fieller adjustment
    df = n - 2
    t_stat = rho * np.sqrt(df) / np.sqrt(1 - rho**2) if abs(rho) < 1.0 else np.inf
    z = np.arctanh(np.clip(rho, -0.99999, 0.99999))
    se_z = 1.06 / np.sqrt(n - 3) if n > 3 else np.nan
    ci_low = float(np.tanh(z - 1.96 * se_z)) if not np.isnan(se_z) else np.nan
    ci_high = float(np.tanh(z + 1.96 * se_z)) if not np.isnan(se_z) else np.nan
    
    print(f"Spearman Rank Correlation:     rho = {rho:+.4f}")
    if not np.isnan(ci_low):
        print(f"95% Confidence Interval:       [{ci_low:+.4f}, {ci_high:+.4f}] (Fieller z)")
    if not np.isinf(t_stat):
        print(f"Hypothesis Test:               t({df}) = {t_stat:.4f}, p = {p_val:.4e}")
    else:
        print(f"Hypothesis Test:               p = {p_val:.4e}")
    print(f"Shared Rank Variance:          rho^2 = {rho**2*100:.1f}%")
    
    # Comparative Kendall Tau-b
    res_tau = stats.kendalltau(x, y)
    print(f"\\nComparison with Kendall's Tau: tau = {res_tau.statistic:+.4f}, p = {res_tau.pvalue:.4e}")
    
    is_sig = (p_val < alpha)
    print(f"Conclusion: {'Statistically significant monotonic association (Reject H0)' if is_sig else 'No significant association (Fail to reject H0)'} at alpha = {alpha}.")
    
    return {
        "method": "spearman_correlation",
        "rho": rho,
        "ci_95": (ci_low, ci_high),
        "p_value": p_val,
        "kendall_tau": float(res_tau.statistic),
        "kendall_p": float(res_tau.pvalue),
        "is_significant": is_sig
    }

# Example Run:
# x_demo = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
# y_demo = np.array([20, 26, 31, 35, 38, 40, 42, 43, 44, 45])
# run_spearman_pipeline(x_demo, y_demo)
""", language="python")


def render_binary_logistic_regression_excel_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 3-step Microsoft Excel guide for Binary Logistic Regression."""
    st.markdown("""
    ### 📗 Microsoft Excel Implementation Workflow for Binary Logistic Regression
    This guide follows a strict 3-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test (Solver Maximum Likelihood & Add-ins)** ➔ **Implement Alternative Models**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN EXCEL
    with st.expander("📌 Step 1: Evaluate Assumptions in Excel (Dichotomous Scale, EPV, Linearity of the Logit, Multicollinearity & Outliers)", expanded=False):
        st.markdown(r"""
        Before interpreting logistic regression parameters and adjusted Odds Ratios, verify that key model assumptions hold:
        1. **Binary Dependent Variable:** Outcome $Y$ must be strictly dichotomous ($0$ or $1$, Negative or Positive, Failure or Success).
        2. **Adequate Sample Size (Events Per Variable / EPV):** Rule of thumb (Peduzzi / Harrell) requires at least $10$ to $20$ events in the smaller outcome class per predictor to prevent coefficient inflation and overfitting.
        3. **Linearity in the Logit:** Continuous predictors must have a linear relationship with the log-odds $\ln(p / (1-p))$ (verified via Box-Tidwell interaction terms $X \ln(X)$ or decile logit plots).
        4. **Absence of Multicollinearity:** Explanatory variables must not be excessively intercorrelated ($\text{VIF} < 5.0$, Tolerance $> 0.20$).
        5. **No Complete Separation & Extreme Outliers:** No single predictor or linear combination should perfectly divide $0$s and $1$s (which sends coefficients and standard errors to infinity), and standardized Pearson residuals should satisfy $|r_i| \le 3.0$.
        """)

        df_excel_assumptions = pd.DataFrame({
            "Regression_Assumption": [
                "1. Binary Outcome Variable",
                "2. Sample Size / Events Per Variable (EPV)",
                "3. Linearity in the Logit (Box-Tidwell)",
                "4. Absence of Multicollinearity (VIF)",
                "5. Separation & Pearson Residual Outliers"
            ],
            "Excel_Formula_or_Tool": [
                "=COUNTIF(B2:B101, \"<>0\") - COUNTIF(B2:B101, \"1\")",
                "=MIN(COUNTIF(B2:B101, 1), COUNTIF(B2:B101, 0)) / k",
                "Decile scatter: Group X into 10 bins, calculate =LN(p_bin / (1 - p_bin))",
                "Auxiliary OLS: =1 / (1 - RSQ(X1_range, other_X_ranges))",
                "Standardized Pearson Residual: =(B2 - G2) / SQRT(G2 * (1 - G2))"
            ],
            "Acceptable_Benchmark": [
                "Must equal 0 (sample contains strictly 0s and 1s)",
                "EPV ≥ 10.0 (EPV ≥ 20.0 ideal for clinical risk scores)",
                "Monotonic linear trend; no parabolic U-curves",
                "VIF < 5.0 (Tolerance > 0.20); VIF ≥ 10.0 is severe",
                "|Pearson Residual| ≤ 3.0; standard errors SE < 20.0"
            ],
            "Interpretation_Verdict": [
                "If non-binary categories exist -> Multinomial or Ordinal Logistic (Step 3).",
                "If EPV < 10.0 -> Standard MLE is biased; switch to Firth Penalized Logistic.",
                "If non-linear -> Apply polynomial term X² or log transformation.",
                "If VIF ≥ 10.0 -> Redundant predictors inflate standard errors; regularize.",
                "If SE explodes (> 25) -> Complete separation; use Firth Penalized Logistic."
            ]
        })
        render_table(df_excel_assumptions, height=220)

        st.markdown("#### 📋 Recommended Spreadsheet Architecture & Cell Coordinates:")
        st.markdown("""
        To replicate maximum likelihood logistic regression in Microsoft Excel, structure your worksheet as follows:
        """)

        df_layout_map = pd.DataFrame({
            "Cell_Range": [
                "A1:A26",
                "B1:B26",
                "C1:E26",
                "F1:F26",
                "G1:G26",
                "H1:H26",
                "B28:E28",
                "B29:E29",
                "H28",
                "B31:E31",
                "B33",
                "B34",
                "B35"
            ],
            "Sheet_Item": [
                "Subject / Observation Identifier",
                "Binary Outcome Y (0 or 1)",
                "Predictor Variables (X1: Age, X2: SBP, X3: Smoker)",
                "Linear Predictor Logit z",
                "Predicted Probability p",
                "Observation Log-Likelihood (LL_i)",
                "Parameter Labels",
                "Estimated Parameters (β0, β1, β2, β3)",
                "Objective Cell: Total Log-Likelihood (LL_model)",
                "Adjusted Odds Ratios (OR = e^β)",
                "Null Log-Likelihood (LL_null)",
                "Omnibus Likelihood Ratio Chi-Square (LR)",
                "Omnibus Model p-value"
            ],
            "Excel_Formula_or_Content": [
                "Obs_001, Obs_002, ...",
                "0 = Non-Event, 1 = Event",
                "Numeric measurements (e.g. Age in C, SBP in D, Smoker in E)",
                "=B$29 + C$29*C2 + D$29*D2 + E$29*E2",
                "=1 / (1 + EXP(-F2))",
                "=B2*LN(G2) + (1-B2)*LN(1-G2)",
                "Intercept (B0) | Age (B1) | SBP (B2) | Smoker (B3)",
                "Initial guess 0.00 (Solved by Excel Solver GRG Nonlinear)",
                "=SUM(H2:H26)",
                "=EXP(B29), =EXP(C29), =EXP(D29), =EXP(E29)",
                "=COUNTIF(B2:B26, 1)*LN(AVERAGE(B2:B26)) + COUNTIF(B2:B26, 0)*LN(1-AVERAGE(B2:B26))",
                "=2 * (H28 - B33)",
                "=CHISQ.DIST.RT(B34, 3)"
            ],
            "Role_in_Analysis": [
                "Audit tracking",
                "Dependent classification target",
                "Explanatory covariate matrix",
                "Linear link function index",
                "Sigmoid-transformed probability",
                "Individual Bernoulli likelihood contribution",
                "Parameter naming header",
                "Solver Decision Variables",
                "Solver Target Objective (Maximize)",
                "Clinically interpretable multiplicative effect per unit change",
                "Baseline intercept-only model likelihood",
                "Omnibus model goodness-of-fit test statistic",
                "Hypothesis test against null intercept-only model"
            ]
        })
        render_table(df_layout_map, height=310)

        st.markdown("#### 👁️ Example Spreadsheet Data Layout (First 5 Rows Preview):")
        df_sheet_preview = pd.DataFrame({
            "A (Obs_ID)": ["Obs_001", "Obs_002", "Obs_003", "Obs_004", "Obs_005"],
            "B (CVD_Event)": [0, 1, 0, 1, 0],
            "C (Age_X1)": [48.0, 64.0, 52.0, 71.0, 55.0],
            "D (SBP_X2)": [124.0, 158.0, 130.0, 168.0, 136.0],
            "E (Smoker_X3)": [0, 1, 0, 1, 0],
            "F (Logit z)": ["-2.14", "+1.85", "-1.42", "+2.90", "-0.95"],
            "G (Prob p)": ["0.1053", "0.8641", "0.1947", "0.9478", "0.2789"],
            "H (Log-Likelihood)": ["-0.1113", "-0.1461", "-0.2165", "-0.0536", "-0.3270"]
        })
        render_table(df_sheet_preview, height=180)

    # STEP 2: IMPLEMENT PROPOSED TEST IN EXCEL
    with st.expander("📌 Step 2: Implement Binary Logistic Regression in Excel (Solver GRG Nonlinear & Real Statistics)", expanded=False):
        st.markdown("""
        **Method A (Recommended Built-in Approach): Excel Solver GRG Nonlinear Maximum Likelihood**
        
        Because Microsoft Excel does not feature a native `=LOGITREG()` worksheet function, Maximum Likelihood Estimation (MLE) is implemented using Excel's built-in **GRG Nonlinear Solver**:

        1. **Enable Excel Solver:**
           * Go to **File > Options > Add-ins**.
           * Under **Manage**, select **Excel Add-ins** and click **Go**.
           * Check **Solver Add-in** and click **OK**.
        
        2. **Enter Data & Establish Parameter Cells:**
           * Place outcome $Y$ in `B2:B26` and predictors $X_1, X_2, X_3$ in `C2:E26`.
           * Create parameter cells in `B29:E29` and initialize each to `0.0`.
        
        3. **Compute Linear Logit $z_i$ in Column F:**
           * In cell `F2`, enter:
           ```excel
           =$B$29 + $C$29*C2 + $D$29*D2 + $E$29*E2
           ```
           * Copy down to row 26.
        
        4. **Compute Predicted Probability $p_i$ in Column G:**
           * In cell `G2`, enter the standard logistic sigmoid function:
           ```excel
           =1 / (1 + EXP(-F2))
           ```
           * Copy down to row 26.
        
        5. **Compute Individual Observation Log-Likelihood in Column H:**
           * In cell `H2`, enter the Bernoulli log-likelihood contribution:
           ```excel
           =B2*LN(G2) + (1-B2)*LN(1-G2)
           ```
           * Copy down to row 26.
        
        6. **Compute Total Log-Likelihood in Cell H28:**
           * In cell `H28`, enter:
           ```excel
           =SUM(H2:H26)
           ```
        
        7. **Configure and Run Solver:**
           * Navigate to the **Data** tab and click **Solver**.
           * **Set Objective:** `$H$28`
           * **To:** Select **Max** (Maximum Likelihood)
           * **By Changing Variable Cells:** `$B$29:$E$29`
           * **Select a Solving Method:** Choose **GRG Nonlinear**
           * Uncheck *Make Unconstrained Variables Non-Negative* (coefficients can be negative).
           * Click **Solve** and accept the Solver solution.
        
        8. **Calculate Adjusted Odds Ratios in Row 31:**
           * Intercept baseline odds: `=EXP(B29)`
           * Age Odds Ratio ($OR_{Age}$): `=EXP(C29)`
           * Systolic BP Odds Ratio ($OR_{SBP}$): `=EXP(D29)`
           * Smoker Odds Ratio ($OR_{Smoker}$): `=EXP(E29)`
        
        9. **Calculate Omnibus Model Significance (Likelihood Ratio Test):**
           * In cell `B33`, compute Null Log-Likelihood:
           ```excel
           =COUNTIF(B2:B26, 1)*LN(AVERAGE(B2:B26)) + COUNTIF(B2:B26, 0)*LN(1-AVERAGE(B2:B26))
           ```
           * In cell `B34`, compute Likelihood Ratio $\\chi^2$:
           ```excel
           =2 * (H28 - B33)
           ```
           * In cell `B35`, compute omnibus $p$-value ($df = 3$):
           ```excel
           =CHISQ.DIST.RT(B34, 3)
           ```
        
        10. **Construct 2x2 Confusion Matrix (0.5 Cutoff):**
           * True Positives (TP): `=COUNTIFS(B2:B26, 1, G2:G26, ">=0.5")`
           * True Negatives (TN): `=COUNTIFS(B2:B26, 0, G2:G26, "<0.5")`
           * False Positives (FP): `=COUNTIFS(B2:B26, 0, G2:G26, ">=0.5")`
           * False Negatives (FN): `=COUNTIFS(B2:B26, 1, G2:G26, "<0.5")`
           * Overall Accuracy: `=(TP + TN) / COUNT(B2:B26)`

        ---
        **Method B (Add-in Approach): Real Statistics Resource Pack (`=LOGITREG`)**
        * If you have the free Real Statistics Resource Pack installed, enter the dynamic array formula:
        ```excel
        =LOGITREG(C1:E26, B1:B26, TRUE)
        ```
        * Instantly outputs standard errors, Wald $z$-statistics, two-tailed $p$-values, McFadden's pseudo-$R^2$, and adjusted Odds Ratios with 95% confidence intervals.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE TESTS IN EXCEL
    with st.expander("📌 Step 3: Implement Alternative Models in Excel (When Assumptions Fail)", expanded=False):
        st.markdown("""
        **1. When Sample Size is Sparse (EPV < 10.0) or Complete Separation Occurs:**
        * **Firth's Penalized Likelihood via Solver:** Add an L2 shrinkage penalty to cell `H28`:
          ```excel
          =SUM(H2:H26) - 0.5 * 1.0 * (C29^2 + D29^2 + E29^2)
          ```
          This penalizes large parameter magnitudes and prevents coefficients from exploding to $\pm \infty$.
        * **Variable Reduction / Screening:** Eliminate non-significant or redundant covariates to increase EPV back above $10.0$.
        
        **2. When Severe Multicollinearity is Present (VIF ≥ 10.0):**
        * **Predictor Removal:** Identify the two collinear predictors (e.g., Systolic BP and Mean Arterial Pressure) and drop the one with lower clinical relevance.
        * **Principal Components / Composite Index:** Combine collinear variables into a standardized composite score:
          ```excel
          =(STANDARDIZE(C2, AVERAGE(C$2:C$26), STDEV.S(C$2:C$26)) + STANDARDIZE(D2, AVERAGE(D$2:D$26), STDEV.S(D$2:D$26))) / 2
          ```
        
        **3. When Linearity in the Logit Fails (Curved Box-Tidwell Diagnostic):**
        * **Polynomial Logit Terms:** Insert a column for $X_1^2$ (`=C2^2`) to allow the log-odds to bend non-linearly.
        * **Log Transformation:** Create a linearized predictor column `=LN(C2)` before running Solver.
        """)


def render_binary_logistic_regression_python_guide(rec: Dict[str, Any]):
    """Renders the comprehensive 4-step Python guide for Binary Logistic Regression."""
    st.markdown("""
    ### 🐍 Python Implementation Workflow for Binary Logistic Regression
    This guide follows a strict 4-step sequence: **Evaluate Assumptions** ➔ **Implement Proposed Test (statsmodels.Logit)** ➔ **Implement Alternative Models (Penalized / Regularized Logistic)** ➔ **Consolidated Executable Pipeline Script**.
    """)

    # STEP 1: EVALUATE ASSUMPTIONS IN PYTHON
    with st.expander("📌 Step 1: Evaluate Assumptions in Python (Dichotomy, EPV, Box-Tidwell Logit Linearity, VIF, Separation & Outliers)", expanded=False):
        st.code("""import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Verify Dichotomous Outcome Scale
unique_outcomes = np.unique(y)
is_binary = len(unique_outcomes) == 2 and set(unique_outcomes).issubset({0, 1})
n_events = int(min(np.sum(y == 1), np.sum(y == 0)))
k_preds = X.shape[1]

# 2. Evaluate Events Per Variable (EPV)
epv = n_events / k_preds

# 3. Evaluate Multicollinearity (Variance Inflation Factor / VIF)
X_const = sm.add_constant(X)
if k_preds > 1:
    vifs = [variance_inflation_factor(X_const.values, i) for i in range(1, k_preds + 1)]
    max_vif = max(vifs)
else:
    vifs = [1.0]
    max_vif = 1.0

# 4. Fit Baseline Logistic Model to Audit Residuals & Separation
logit_mod = sm.Logit(y, X_const).fit(disp=False)
p_pred = logit_mod.predict(X_const)
pearson_residuals = (y - p_pred) / np.sqrt(p_pred * (1.0 - p_pred))
max_pearson = np.max(np.abs(pearson_residuals))
extreme_outliers = np.sum(np.abs(pearson_residuals) > 3.0)

# Check for quasi-complete separation (exploding SEs or extreme coefficients)
is_separated = np.any(logit_mod.bse > 25.0) or np.any(np.abs(logit_mod.params) > 15.0)

print(f"Outcome Dichotomy Verified: {is_binary} (Classes: {unique_outcomes})")
print(f"Events Per Variable (EPV): {epv:.1f} (Benchmark: >= 10.0)")
print(f"Maximum Predictor VIF: {max_vif:.2f} (Threshold: < 5.0)")
print(f"Complete Separation Detected: {is_separated}")
print(f"Extreme Pearson Residual Outliers (|r| > 3.0): {extreme_outliers} (Max |r| = {max_pearson:.2f})")
""", language="python")

        st.markdown("""
        **🔍 How to Interpret the Diagnostic Output:**
        * **Binary Outcome:** Outcome must be coded 0 and 1. If multi-class, switch to Multinomial Logistic Regression.
        * **Events Per Variable (`epv >= 10.0`):** Confirms adequate sample size. If $\\text{EPV} < 5.0$, standard maximum likelihood is biased $\\rightarrow$ switch to Firth's penalized likelihood or L2 regularization.
        * **Multicollinearity (`max_vif < 5.0`):** Low intercorrelation. If $\\text{VIF} \\ge 10.0$, predictors share redundant information $\\rightarrow$ regularize or drop collinear terms.
        * **Separation (`is_separated == False`):** If True, a predictor perfectly divides outcome classes, causing standard errors to explode to infinity $\\rightarrow$ apply penalized logistic regression.
        * **Pearson Residuals:** Observations with $|r| > 3.0$ represent poorly fitted cases with high leverage.
        """)

    # STEP 2: IMPLEMENT BINARY LOGISTIC REGRESSION IN PYTHON
    with st.expander("📌 Step 2: Implement Binary Logistic Regression in Python (statsmodels.api.Logit)", expanded=False):
        st.code("""# 1. Fit Binary Logistic Regression Model
X_const = sm.add_constant(X)
logit_model = sm.Logit(y, X_const).fit()

# 2. Display Model Summary Table
print(logit_model.summary())

# 3. Compute Adjusted Odds Ratios & 95% Confidence Intervals
odds_ratios = np.exp(logit_model.params)
conf_ints = np.exp(logit_model.conf_int())
or_summary = pd.DataFrame({
    "Odds_Ratio": odds_ratios,
    "95%_CI_Lower": conf_ints[0],
    "95%_CI_Upper": conf_ints[1],
    "p_value": logit_model.pvalues
})
print("\\n=== ADJUSTED ODDS RATIOS & 95% CONFIDENCE INTERVALS ===")
print(or_summary)

# 4. Omnibus Goodness-of-Fit & Model Quality
ll_model = logit_model.llf
ll_null = logit_model.llnull
lr_stat = logit_model.llr
lr_p = logit_model.llr_pvalue
pseudo_r2 = logit_model.prsquared  # McFadden's Pseudo-R²

print(f"\\nLikelihood Ratio Chi²({logit_model.df_model:.0f}) = {lr_stat:.2f}, p = {lr_p:.4e}")
print(f"McFadden's Pseudo-R² = {pseudo_r2:.4f}")
print(f"AIC = {logit_model.aic:.2f}, BIC = {logit_model.bic:.2f}")

# 5. Classification Performance & ROC-AUC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

y_probs = logit_model.predict(X_const)
y_preds = (y_probs >= 0.5).astype(int)
auc_score = roc_auc_score(y, y_probs)

print(f"ROC-AUC Score = {auc_score:.4f}")
print("\\nConfusion Matrix (0.5 Cutoff):")
print(confusion_matrix(y, y_preds))
print("\\nClassification Report:")
print(classification_report(y, y_preds))
""", language="python")

        st.markdown("""
        **🔍 How to Evaluate Logistic Regression Output:**
        * **Omnibus Model Fit ($p < 0.05$):** The predictor set significantly improves prediction over an intercept-only null model.
        * **McFadden's Pseudo-$R^2$:** Values between $0.20$ and $0.40$ represent excellent model fit (analogous to higher $R^2$ in OLS).
        * **Adjusted Odds Ratios ($OR = e^B$):**
          * $OR > 1.0$: Predictor increases odds of the event (e.g., $OR = 1.85$ means an 85% increase in odds per 1-unit increase in $X$).
          * $OR < 1.0$: Predictor decreases odds of the event (protective factor).
          * $95\\%\\text{ CI}$ not crossing $1.0$ indicates statistical significance at $\\alpha = 0.05$.
        * **ROC-AUC Score:** $0.50$ is random chance, $0.70-0.80$ is acceptable, $> 0.80$ is excellent discrimination.
        """)

    # STEP 3: IMPLEMENT ALTERNATIVE MODELS IN PYTHON
    with st.expander("📌 Step 3: Implement Alternative Models in Python (When Assumptions Fail)", expanded=False):
        st.code("""# Alternative 1: L2 Ridge Penalized Logistic Regression (When VIF >= 10 or EPV < 10)
from sklearn.linear_model import LogisticRegression

# L2 regularization prevents overfitting and handles collinear predictors
ridge_logit = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
ridge_logit.fit(X, y)

print("=== L2 RIDGE PENALIZED LOGISTIC REGRESSION ===")
print(f"Intercept: {ridge_logit.intercept_[0]:.4f}")
for idx, coef in enumerate(ridge_logit.coef_[0]):
    print(f"Predictor X{idx+1}: Coef = {coef:.4f}, OR = {np.exp(coef):.4f}")

# Alternative 2: Firth's Penalized Likelihood (When Complete Separation Occurs)
# In Python, Firth's correction is available via the 'firthlogist' library:
# from firthlogist import FirthLogisticRegression
# firth_mod = FirthLogisticRegression().fit(X, y)
# print(firth_mod.summary())
""", language="python")

        st.markdown("""
        **🔍 When to Switch to Alternative Models:**
        * **Use L2 Ridge Penalization:** When multicollinearity ($\text{VIF} \ge 10.0$) or sparse sample size ($\text{EPV} < 10.0$) destabilizes maximum likelihood estimates.
        * **Use Firth's Penalized Likelihood:** When complete or quasi-complete separation occurs, producing infinite parameter estimates or exploding standard errors.
        * **Use Multinomial / Ordinal Logistic Regression:** When the outcome variable has 3 or more categories (nominal or ordered).
        """)

    # STEP 4: CONSOLIDATED PYTHON PIPELINE SCRIPT
    with st.expander("📋 Copy Complete Consolidated Python Pipeline Script", expanded=False):
        st.code("""import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

def run_binary_logistic_pipeline(X, y, feature_names=None, alpha=0.05):
    \"\"\"Automated Binary Logistic Regression Pipeline with comprehensive
    assumption verification (Dichotomy, EPV, Multicollinearity, Separation,
    Residual Outliers) and conditional assurance/warning printouts.
    \"\"\"
    X = np.asarray(X)
    y = np.asarray(y)
    n, k = X.shape
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(k)]
    
    print("=" * 70)
    print(f"=== BINARY LOGISTIC REGRESSION PIPELINE (N = {n}, Predictors k = {k}) ===")
    print("=" * 70)
    
    # 1. Assumption Audits
    unique_y = np.unique(y)
    is_binary = len(unique_y) == 2 and set(unique_y).issubset({0, 1})
    n_events = int(min(np.sum(y == 1), np.sum(y == 0)))
    epv = float(n_events / k)
    
    X_const = sm.add_constant(X)
    if k > 1:
        vifs = [variance_inflation_factor(X_const, i) for i in range(1, k + 1)]
        max_vif = float(max(vifs))
        pass_vif = (max_vif < 5.0)
        severe_vif = (max_vif >= 10.0)
    else:
        vifs = [1.0]
        max_vif = 1.0
        pass_vif = True
        severe_vif = False
        
    # Fit initial MLE model
    try:
        model = sm.Logit(y, X_const).fit(disp=False)
        converged = True
    except Exception as e:
        converged = False
        model = None
        
    if converged:
        p_pred = model.predict(X_const)
        p_pred_clipped = np.clip(p_pred, 1e-12, 1.0 - 1e-12)
        pearson_res = (y - p_pred_clipped) / np.sqrt(p_pred_clipped * (1.0 - p_pred_clipped))
        extreme_outliers = int(np.sum(np.abs(pearson_res) > 3.0))
        max_pearson = float(np.max(np.abs(pearson_res)))
        is_separated = bool(np.any(model.bse > 25.0) or np.any(np.abs(model.params) > 15.0))
    else:
        is_separated = True
        extreme_outliers = 0
        max_pearson = 0.0
        
    # 2. Print Conditional Diagnostic Feedback
    print("\\n--- METHODOLOGICAL ASSUMPTION DIAGNOSTICS ---")
    
    # Scale Audit
    if is_binary:
        print(f"[ASSURANCE - PASS] Binary outcome scale verified: {np.sum(y==0)} controls (0) vs {np.sum(y==1)} events (1).")
    else:
        print(f"[INAPPROPRIATE - FAIL] Outcome variable is not strictly dichotomous (Unique values: {unique_y})!")
        print("   -> Switch to Multinomial or Ordinal Logistic Regression.")
        
    # EPV Audit
    if epv >= 10.0:
        print(f"[ASSURANCE - PASS] Adequate sample size: EPV = {epv:.1f} >= 10.0 ({n_events} effective events across {k} predictors).")
    elif epv >= 5.0:
        print(f"[WARNING] Borderline sample size: EPV = {epv:.1f} < 10.0. Slight risk of coefficient overestimation.")
        print("   -> Verify findings against L2 penalized logistic regression.")
    else:
        print(f"[INAPPROPRIATE - FAIL] Sparse sample size: EPV = {epv:.1f} < 5.0 ({n_events} effective events across {k} predictors)!")
        print("   -> Standard Maximum Likelihood suffers from severe small-sample bias. Switching to Penalized Logistic Regression.")
        
    # Multicollinearity Audit
    if pass_vif:
        print(f"[ASSURANCE - PASS] Multicollinearity low: Max VIF = {max_vif:.2f} < 5.0 across all predictors.")
    elif not severe_vif:
        print(f"[WARNING] Moderate multicollinearity: Max VIF = {max_vif:.2f} (5.0 to 10.0). Standard errors slightly widened.")
    else:
        print(f"[INAPPROPRIATE - FAIL] Severe multicollinearity: Max VIF = {max_vif:.2f} >= 10.0!")
        print("   -> High predictor redundancy destabilizes Odds Ratios. Switching to L2 Regularized Logistic Regression.")
        
    # Separation Audit
    if not is_separated:
        print("[ASSURANCE - PASS] Complete separation absent: Coefficients converged to stable, finite estimates.")
    else:
        print("[INAPPROPRIATE - FAIL] Complete or quasi-complete separation detected (Exploding SEs or divergent parameters)!")
        print("   -> Standard MLE estimates are infinite. Switching to Firth's Penalized Logistic Regression.")
        
    # Residual Outlier Audit
    if extreme_outliers == 0:
        print(f"[ASSURANCE - PASS] Pearson residuals well-calibrated (Max |r| = {max_pearson:.2f} <= 3.0).")
    else:
        print(f"[WARNING] Detected {extreme_outliers} observation(s) with extreme Pearson residuals (|r| > 3.0, Max |r| = {max_pearson:.2f}).")
        print("   -> Inspect flagged cases for clinical anomalies or incorrect data entry.")
        
    # 3. Final Methodological Routing Decision
    print("\\n" + "=" * 70)
    if is_binary and epv >= 5.0 and not severe_vif and not is_separated and converged:
        print("--> [VERDICT] Assumptions satisfied: Reporting Binary Logistic Regression Model.")
        print("=" * 70)
        print(model.summary())
        
        # Odds Ratios Table
        or_df = pd.DataFrame({
            "Predictor": ["Intercept"] + feature_names,
            "Beta": model.params,
            "SE": model.bse,
            "z_stat": model.tvalues,
            "p_value": model.pvalues,
            "Odds_Ratio": np.exp(model.params),
            "95%_CI_Lower": np.exp(model.conf_int()[0]),
            "95%_CI_Upper": np.exp(model.conf_int()[1])
        })
        print("\\n--- ADJUSTED ODDS RATIOS & 95% CONFIDENCE INTERVALS ---")
        print(or_df.to_string(index=False))
        
        # Model Evaluation
        p_final = model.predict(X_const)
        auc_val = roc_auc_score(y, p_final)
        print(f"\\nOmnibus Likelihood Ratio Chi² = {model.llr:.2f}, p = {model.llr_pvalue:.4e}")
        print(f"McFadden's Pseudo-R² = {model.prsquared:.4f}")
        print(f"ROC-AUC Discrimination Index = {auc_val:.4f}")
        return model
    else:
        print("--> [VERDICT] Switching to Penalized Logistic Regression (L2 Ridge / Firth).")
        print("=" * 70)
        from sklearn.linear_model import LogisticRegression
        pen_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000).fit(X, y)
        print(f"Penalized Intercept: {pen_model.intercept_[0]:.4f}")
        for idx, coef in enumerate(pen_model.coef_[0]):
            print(f"Penalized {feature_names[idx]}: Coef = {coef:.4f}, Adjusted OR = {np.exp(coef):.4f}")
        return pen_model

# Example Run:
# X_demo = np.column_stack([np.random.normal(55, 8, 120), np.random.normal(130, 15, 120)])
# y_demo = (1 / (1 + np.exp(-(-5.0 + 0.05 * X_demo[:, 0] + 0.02 * X_demo[:, 1]))) > np.random.rand(120)).astype(int)
# run_binary_logistic_pipeline(X_demo, y_demo, feature_names=["Age", "Systolic_BP"])
""", language="python")


def render_generic_excel_guide(rec: Dict[str, Any]):
    """Fallback Excel guide for other tests until customized."""
    excel = rec["excel_guide"]
    st.subheader("Microsoft Excel Implementation")

    formula_syntax = excel.get("formula_syntax", "")
    lines = [ln.strip() for ln in formula_syntax.splitlines() if ln.strip()]

    if len(lines) > 1:
        st.markdown("#### 📗 Native Excel Formulas & Workflow Steps:")
        for line in lines:
            if ":" in line:
                step_title, step_detail = line.split(":", 1)
                st.markdown(f"* **{step_title.strip()}:** {step_detail.strip()}")
            else:
                st.markdown(f"* {line}")
    elif len(lines) == 1:
        if excel.get("has_native_formula"):
            st.success(f"**Native Excel Formula:** `{lines[0]}`")
        else:
            st.warning(f"**Formula Note:** {lines[0]}")
    elif excel.get("has_native_formula"):
        st.success(f"**Native Excel Formula:** `{formula_syntax}`")
    else:
        st.warning(f"**Formula Note:** {formula_syntax}")

    if excel.get("toolpak_available"):
        st.markdown("#### 🛠️ Data Analysis Toolpak Steps:")
        st.info(excel["toolpak_steps"])
    else:
        st.markdown("#### 🛠️ Toolpak Availability:")
        st.markdown(f"*{excel.get('toolpak_steps', 'Not built into Toolpak.')}*")

    if excel.get("notes"):
        st.caption(f"💡 **Tip:** {excel['notes']}")


def render_generic_python_guide(rec: Dict[str, Any]):
    """Fallback Python guide for other tests until customized."""
    st.subheader("Python Script")
    st.code(rec["python_snippet"], language="python")


def render_interactive_demo(demo_type: str, demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders interactive Plotly visual and summary test statistics."""
    st.subheader("🧪 Live Interactive Demonstration")
    st.markdown("Adjust parameters in the sidebar to see how data distributions affect the test results in real time.")

    if demo_type == "one_sample_continuous":
        render_one_sample_t_test_demo(demo_df, stats_results)

    elif demo_type == "wilcoxon_signed_rank":
        render_wilcoxon_demo(demo_df, stats_results)

    elif demo_type == "wilcoxon_paired":
        render_wilcoxon_paired_demo(demo_df, stats_results)

    elif demo_type == "mann_whitney_u":
        render_mann_whitney_demo(demo_df, stats_results)

    elif demo_type == "welch_t_test":
        render_welch_demo(demo_df, stats_results)

    elif demo_type in ["students_t_test", "two_independent_continuous"]:
        render_students_t_demo(demo_df, stats_results)

    elif demo_type in ["paired_t_test", "paired_continuous"]:
        render_paired_t_demo(demo_df, stats_results)

    elif demo_type in ["one_way_anova", "three_independent_continuous"]:
        render_one_way_anova_demo(demo_df, stats_results)

    elif demo_type in ["kruskal_wallis", "three_independent_ordinal"]:
        render_kruskal_wallis_demo(demo_df, stats_results)

    elif demo_type in ["repeated_measures_anova", "three_repeated_continuous"]:
        render_repeated_measures_anova_demo(demo_df, stats_results)

    elif demo_type in ["friedman_test", "repeated_measures_ordinal", "repeated_measures_continuous"]:
        render_friedman_demo(demo_df, stats_results)

    elif demo_type == "fisher_exact":
        render_fisher_exact_demo(demo_df, stats_results)

    elif demo_type in ["mcnemar_test", "paired_binary"]:
        render_mcnemar_demo(demo_df, stats_results)

    elif demo_type in ["contingency_table_2x2", "chi_square_independence"]:
        render_chi_square_demo(demo_df, stats_results)

    elif demo_type in ["pearson_correlation", "correlation_continuous"]:
        render_pearson_correlation_demo(demo_df, stats_results)

    elif demo_type in ["spearman_correlation", "spearman_rank"]:
        render_spearman_correlation_demo(demo_df, stats_results)

    elif demo_type in ["linear_regression", "multiple_linear_regression", "regression_continuous"]:
        render_linear_regression_demo(demo_df, stats_results)

    elif demo_type in ["binary_logistic_regression", "binary_classification", "logistic_regression"]:
        render_binary_logistic_regression_demo(demo_df, stats_results)


def render_one_sample_t_test_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics for One-Sample t-test."""
    st.markdown("### 🎯 Hypothesis Test Results: Sample vs. Benchmark")

    # 1. Metrics Grid with Tooltips
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric(
        "Sample Size (N)",
        f"{stats_results['n']}",
        help="Number of observations. N >= 30 provides Central Limit Theorem robustness."
    )
    mcol1.metric(
        "Benchmark (μ₀)",
        f"{stats_results['benchmark']:.2f}",
        help="The theoretical standard or population mean being compared against."
    )

    mcol2.metric(
        "Sample Mean (x̄)",
        f"{stats_results['mean']:.2f}",
        help="Calculated sample arithmetic mean."
    )
    mcol2.metric(
        "Mean Diff (x̄ - μ₀)",
        f"{stats_results['mean_diff']:+.2f}",
        help="Observed difference between the sample mean and the benchmark."
    )

    mcol3.metric(
        "t-Statistic",
        f"t({stats_results['df']}) = {stats_results['t_stat']:.3f}",
        help="Number of standard errors the sample mean is away from the benchmark."
    )
    sig_text = "p < 0.05 (Significant)" if stats_results["p_val"] < 0.05 else "p ≥ 0.05 (Not Significant)"
    mcol3.metric(
        "p-value",
        f"{stats_results['p_val']:.4f}",
        delta=sig_text,
        delta_color="normal" if stats_results["p_val"] < 0.05 else "off",
        help=(
            "The p-value measures whether the difference between the sample mean (x̄) and the benchmark (μ₀) is likely real or just random luck.\n\n"
            "• p < 0.05 (Statistically Significant): Strong evidence that the true population mean differs from the benchmark "
            "(rejects the null hypothesis H0: μ = μ₀). The observed difference is very unlikely to be a fluke of random sampling.\n\n"
            "• p ≥ 0.05 (Not Significant): Insufficient evidence to claim a true difference; the sample mean is statistically consistent with the benchmark, "
            "and any observed difference could easily occur by chance variation."
        )
    )

    mcol4.metric(
        "95% CI for Mean",
        f"[{stats_results['ci_lower']:.2f}, {stats_results['ci_upper']:.2f}]",
        help="95% Confidence Interval for the true population mean."
    )
    mcol4.metric(
        "Effect Size (Cohen's d)",
        f"{stats_results['cohens_d']:.2f}",
        help="Standardized effect size: 0.2 = small, 0.5 = medium, 0.8 = large."
    )

    st.markdown("---")

    # 2. Interactive Visualization Tabs
    st.markdown("### 📈 Visual Comparison & Distribution")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📊 Sample Distribution vs Benchmark",
        "📐 Q-Q Normality Plot",
        "📦 Boxplot & Outliers"
    ])

    with viz_tab1:
        # Histogram with benchmark reference line
        fig = px.histogram(
            demo_df,
            x="Value",
            nbins=max(10, int(stats_results["n"] / 3)),
            title="Sample Value Distribution vs. Theoretical Benchmark",
            color_discrete_sequence=["#3b82f6"],
            opacity=0.75
        )
        # Vertical line for benchmark
        fig.add_vline(
            x=stats_results["benchmark"],
            line_width=3,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Benchmark μ₀ = {stats_results['benchmark']:.1f}",
            annotation_position="top left",
            annotation_font=dict(color="#ef4444", size=12, family="sans-serif")
        )
        # Vertical line for sample mean
        fig.add_vline(
            x=stats_results["mean"],
            line_width=3,
            line_dash="solid",
            line_color="#1d4ed8",
            annotation_text=f"Sample Mean x̄ = {stats_results['mean']:.1f}",
            annotation_position="top right",
            annotation_font=dict(color="#1d4ed8", size=12, family="sans-serif")
        )
        # Shaded CI band
        fig.add_vrect(
            x0=stats_results["ci_lower"],
            x1=stats_results["ci_upper"],
            fillcolor="#93c5fd",
            opacity=0.25,
            layer="below",
            line_width=0,
            annotation_text="95% CI",
            annotation_position="bottom right"
        )
        fig.update_layout(template="plotly_white", xaxis_title="Value", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "💡 **How to interpret this chart & p-value:** "
            "This chart compares your sample distribution and mean (x̄, solid blue line) against the theoretical benchmark (μ₀, dashed red line). "
            "The shaded blue band represents the 95% Confidence Interval (CI) for the true population mean. "
            "A **p-value < 0.05** indicates a statistically significant difference: there is strong evidence that the true population mean is genuinely different from the benchmark "
            "(the red dashed benchmark line falls outside the 95% CI band), meaning the gap is unlikely to be a fluke of random sampling. "
            "Conversely, a **p-value ≥ 0.05** means the sample mean is statistically consistent with the benchmark, and any observed difference could easily be due to chance variation alone."
        )

    with viz_tab2:
        # Q-Q plot
        qq_fig = go.Figure()
        # Scatter of quantiles
        qq_fig.add_trace(go.Scatter(
            x=stats_results["qq_osm"],
            y=stats_results["qq_osr"],
            mode="markers",
            marker=dict(color="#2563eb", size=7, opacity=0.8),
            name="Sample Quantiles"
        ))
        # 45-degree reference line
        osm_arr = stats_results["qq_osm"]
        x_trend = [min(osm_arr), max(osm_arr)]
        y_trend = [stats_results["qq_slope"] * x + stats_results["qq_intercept"] for x in x_trend]
        qq_fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode="lines",
            line=dict(color="#ef4444", width=2, dash="dash"),
            name="Normal Line"
        ))
        qq_fig.update_layout(
            title="Normal Q-Q (Quantile-Quantile) Plot",
            xaxis_title="Theoretical Normal Quantiles",
            yaxis_title="Ordered Sample Values",
            template="plotly_white"
        )
        st.plotly_chart(qq_fig, width="stretch")
        st.caption("💡 **How to interpret:** If the data is normally distributed, points will fall tightly along the red diagonal line. S-shaped curves or points flaring off at the tails indicate skewness or extreme outliers.")

    with viz_tab3:
        box_fig = px.box(
            demo_df,
            y="Value",
            points="all",
            title="Boxplot with Individual Data Points",
            color_discrete_sequence=["#3b82f6"]
        )
        box_fig.add_hline(
            y=stats_results["benchmark"],
            line_width=3,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Benchmark μ₀ = {stats_results['benchmark']:.1f}"
        )
        box_fig.update_layout(template="plotly_white")
        st.plotly_chart(box_fig, width="stretch")

    st.markdown("---")

    # 3. Comprehensive Assumption Diagnostics Section
    st.markdown("### 🛡️ Diagnostic Tests for Core Assumptions")
    st.markdown("""
    Every statistical hypothesis test rests on fundamental assumptions. Violating these assumptions can lead to 
    unreliable p-values and flawed conclusions. Below are the tests evaluating each assumption for this dataset:
    """)

    # Render assumption cards in 2x2 grid
    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        status_color = "#10b981" if asm["status"] == "pass" else ("#f59e0b" if asm["status"] == "warning" else "#ef4444")
        bg_color = "#f0fdf4" if asm["status"] == "pass" else ("#fffbeb" if asm["status"] == "warning" else "#fef2f2")
        badge_symbol = "✅" if asm["status"] == "pass" else ("🟡" if asm["status"] == "warning" else "❌")

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

        # Highlight Wilcoxon Signed-Rank Test side-by-side
        with st.expander("🔎 View Non-Parametric Alternative: Wilcoxon Signed-Rank Test Details", expanded=False):
            st.markdown(f"""
            - **Hypothesis Tested:** Median of the population = Benchmark ({stats_results['benchmark']:.1f})
            - **Wilcoxon W Statistic:** `{stats_results['wilcoxon_stat']:.2f}`
            - **p-value:** `{stats_results['wilcoxon_p']:.4f}`
            - **Conclusion:** {'Statistically significant difference from benchmark (p < 0.05).' if stats_results['wilcoxon_p'] < 0.05 else 'No statistically significant difference from benchmark (p ≥ 0.05).'}
            - **Why this works:** The Wilcoxon Signed-Rank test ranks the absolute differences $|x_i - \\mu_0|$ and sums the positive/negative signs, making it immune to extreme skewness or outliers.
            """)


def render_wilcoxon_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics for Wilcoxon Signed-Rank Test."""
    st.markdown("### 🎯 Hypothesis Test Results: Sample Median vs. Benchmark")

    # 1. Metrics Grid with Tooltips
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric(
        "Sample Size (N)",
        f"{stats_results['n']}",
        help="Total observations in the dataset."
    )
    mcol1.metric(
        "Benchmark (μ₀)",
        f"{stats_results['benchmark']:.2f}",
        help="The theoretical median benchmark being tested against."
    )

    mcol2.metric(
        "Sample Median",
        f"{stats_results['median']:.2f}",
        help="50th percentile of the sample data."
    )
    mcol2.metric(
        "Median Diff",
        f"{stats_results['median_diff']:+.2f}",
        help="Observed difference: Sample Median - Benchmark."
    )

    mcol3.metric(
        "Wilcoxon Statistic",
        f"W = {stats_results['w_stat']:.1f}",
        help="Sum of signed ranks for positive/negative differences."
    )
    sig_text = "p < 0.05 (Significant)" if stats_results["p_wilcoxon"] < 0.05 else "p ≥ 0.05 (Not Significant)"
    mcol3.metric(
        "p-value",
        f"{stats_results['p_wilcoxon']:.4f}",
        delta=sig_text,
        delta_color="normal" if stats_results["p_wilcoxon"] < 0.05 else "off",
        help=(
            "The p-value measures whether the difference between the sample median and the benchmark (μ₀) is likely real or just random luck.\n\n"
            "• p < 0.05 (Statistically Significant): Strong evidence that the population median truly differs from the theoretical benchmark "
            "(rejects the null hypothesis of zero median difference). Positive and negative ranks around the benchmark are significantly unbalanced.\n\n"
            "• p ≥ 0.05 (Not Significant): Insufficient evidence to conclude a true median shift; the ranks of positive and negative differences are balanced, "
            "meaning any observed difference could easily arise from chance variation."
        )
    )

    mcol4.metric(
        "Hodges-Lehmann Shift",
        f"{stats_results['hl_estimate']:+.2f}",
        help="Median of all pairwise Walsh averages (x_i + x_j)/2 - mu_0."
    )
    mcol4.metric(
        "Effective N (N_eff)",
        f"{stats_results['n_eff']} (ties: {stats_results['n_zeros']})",
        help="Non-zero differences included in rank calculation (exact ties with benchmark dropped)."
    )

    st.markdown("---")

    # 2. Interactive Visualization Tabs
    st.markdown("### 📈 Visual Comparison & Distribution")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📊 Sample Distribution vs Benchmark",
        "🌊 Signed Differences & Ranks",
        "📦 Boxplot & Median Shift"
    ])

    with viz_tab1:
        fig = px.histogram(
            demo_df,
            x="Value",
            nbins=max(10, int(stats_results["n"] / 3)),
            title="Sample Value Distribution vs. Theoretical Median Benchmark",
            color_discrete_sequence=["#8b5cf6"],
            opacity=0.75
        )
        fig.add_vline(
            x=stats_results["benchmark"],
            line_width=3,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Benchmark μ₀ = {stats_results['benchmark']:.1f}",
            annotation_position="top left",
            annotation_font=dict(color="#ef4444", size=12)
        )
        fig.add_vline(
            x=stats_results["median"],
            line_width=3,
            line_dash="solid",
            line_color="#7c3aed",
            annotation_text=f"Sample Median = {stats_results['median']:.1f}",
            annotation_position="top right",
            annotation_font=dict(color="#7c3aed", size=12)
        )
        fig.update_layout(template="plotly_white", xaxis_title="Value", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "💡 **How to interpret this chart & p-value:** "
            "This non-parametric test compares your sample median (solid purple line) against the theoretical median benchmark (μ₀, dashed red line) by analyzing signed ranks. "
            "A **p-value < 0.05** indicates a statistically significant result: there is strong evidence that the true population median differs from the benchmark "
            "(rejecting the null hypothesis that differences are balanced around μ₀) — observations systematically tend to lie on one side of the benchmark rather than varying by random chance. "
            "Conversely, a **p-value ≥ 0.05** means the distribution of differences is consistent with chance, providing insufficient evidence to claim a true shift in the population median."
        )

    with viz_tab2:
        diff_df = demo_df.copy()
        diff_df["Sign"] = diff_df["Difference"].apply(
            lambda x: "Positive Difference" if x > 0 else ("Negative Difference" if x < 0 else "Zero Difference")
        )
        diff_fig = px.bar(
            diff_df,
            x="Observation_ID",
            y="Difference",
            color="Sign",
            title="Observation Differences from Benchmark (xᵢ - μ₀) and Signs",
            color_discrete_map={
                "Positive Difference": "#2563eb",
                "Negative Difference": "#f97316",
                "Zero Difference": "#94a3b8"
            }
        )
        diff_fig.add_hline(y=0, line_width=2, line_color="#334155")
        diff_fig.update_layout(template="plotly_white", xaxis_title="Observation", yaxis_title="Difference (Value - μ₀)")
        st.plotly_chart(diff_fig, width="stretch")
        st.caption(r"💡 **How to interpret:** The Wilcoxon Signed-Rank Test assigns ranks to the absolute bar heights $|x_i - \mu_0|$ and compares the total sum of blue ranks ($W^+$) against orange ranks ($W^-$).")

    with viz_tab3:
        box_fig = px.box(
            demo_df,
            y="Value",
            points="all",
            title="Boxplot with Individual Data Points & Median Benchmark",
            color_discrete_sequence=["#8b5cf6"]
        )
        box_fig.add_hline(
            y=stats_results["benchmark"],
            line_width=3,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Benchmark μ₀ = {stats_results['benchmark']:.1f}"
        )
        box_fig.update_layout(template="plotly_white")
        st.plotly_chart(box_fig, width="stretch")

    st.markdown("---")

    # 3. Comprehensive Assumption Diagnostics Section
    st.markdown("### 🛡️ Diagnostic Tests for Core Assumptions")
    st.markdown("""
    Every statistical hypothesis test rests on fundamental assumptions. Below are the diagnostic evaluations 
    verifying whether the Wilcoxon Signed-Rank Test is appropriate for this dataset:
    """)

    # Render assumption cards in 2x2 grid
    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        status_color = "#10b981" if asm["status"] == "pass" else ("#f59e0b" if asm["status"] == "warning" else "#ef4444")
        bg_color = "#f0fdf4" if asm["status"] == "pass" else ("#fffbeb" if asm["status"] == "warning" else "#fef2f2")
        badge_symbol = "✅" if asm["status"] == "pass" else ("🟡" if asm["status"] == "warning" else "❌")

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

        # Highlight Exact Sign Test side-by-side
        with st.expander("🔎 View Non-Parametric Alternative: Exact Sign Test Details", expanded=False):
            st.markdown(f"""
            - **Hypothesis Tested:** Median of the population = Benchmark ({stats_results['benchmark']:.1f})
            - **Positive Differences Count:** `{stats_results['pos_count']} / {stats_results['n_eff']}`
            - **Sign Test p-value:** `{stats_results['p_sign_test']:.4f}`
            - **Conclusion:** {'Statistically significant median difference from benchmark (p < 0.05).' if stats_results['p_sign_test'] < 0.05 else 'No statistically significant difference from benchmark (p ≥ 0.05).'}
            - **Why this works:** The Exact Sign Test makes **zero assumptions about distribution symmetry**. It tests whether values are equally likely to fall above or below the benchmark using the exact binomial distribution.
            """)


def render_mann_whitney_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics for Mann-Whitney U Test."""
    st.markdown("### 🎯 Hypothesis Test Results: Group A vs. Group B")

    # 1. Metrics Grid with Tooltips
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric(
        f"{stats_results['group_a_name']} Median",
        f"{stats_results['median_a']:.2f}",
        help=f"Sample size n = {stats_results['n_a']}. IQR = {stats_results['iqr_a']:.2f}, Mean = {stats_results['mean_a']:.2f}, Std Dev = {stats_results['std_a']:.2f}."
    )
    mcol1.metric(
        f"{stats_results['group_b_name']} Median",
        f"{stats_results['median_b']:.2f}",
        help=f"Sample size n = {stats_results['n_b']}. IQR = {stats_results['iqr_b']:.2f}, Mean = {stats_results['mean_b']:.2f}, Std Dev = {stats_results['std_b']:.2f}."
    )

    mcol2.metric(
        "Observed Median Diff",
        f"{stats_results['median_diff']:+.2f}",
        help="Sample Median of Group B minus Sample Median of Group A."
    )
    mcol2.metric(
        "Hodges-Lehmann Shift",
        f"{stats_results['hl_diff']:+.2f}",
        help="Median of all pairwise differences (B_j - A_i). Represents the non-parametric location shift."
    )

    mcol3.metric(
        "Mann-Whitney U",
        f"U = {stats_results['u_stat']:.1f}",
        help=f"U_A = {stats_results['u_a']:.1f}, U_B = {stats_results['u_b']:.1f}, Rank Sum A = {stats_results['rank_sum_a']:.1f}, Rank Sum B = {stats_results['rank_sum_b']:.1f}."
    )
    sig_text = "p < 0.05 (Significant)" if stats_results["p_mann"] < 0.05 else "p ≥ 0.05 (Not Significant)"
    mcol3.metric(
        "p-value",
        f"{stats_results['p_mann']:.4f}",
        delta=sig_text,
        delta_color="normal" if stats_results["p_mann"] < 0.05 else "off",
        help=(
            "The p-value measures whether the observed group difference is likely real or just random luck.\n\n"
            "• p < 0.05 (Statistically Significant): There is strong evidence of a genuine difference between the two groups "
            "(rejects the null hypothesis of identical distributions). Observations in one group tend to systematically score higher/lower "
            "rather than differing by random chance. When distribution shapes are similar, this indicates a significant difference in group medians.\n\n"
            "• p ≥ 0.05 (Not Significant): Insufficient evidence to conclude a real difference exists; any observed gap could easily have occurred by random sampling variation."
        )
    )

    mcol4.metric(
        "Rank-Biserial Correlation",
        f"{stats_results['r_rb']:+.3f}",
        help="Standardized effect size r_rb = (U_B - U_A) / (n_A * n_B), ranging from -1.0 to +1.0. 0.1=small, 0.3=medium, 0.5=large."
    )
    mcol4.metric(
        "Prob of Superiority (CLES)",
        f"{stats_results['cles']:.3f}",
        help="Common Language Effect Size: Probability that a randomly chosen observation from Group B exceeds Group A. 0.50 represents equality."
    )

    st.markdown("---")

    # 2. Interactive Visualization Tabs
    st.markdown("### 📈 Visual Comparison & Distribution")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📊 Group Distribution Comparison",
        "📈 Cumulative Distribution (ECDF)",
        "📦 Boxplot & Individual Observations"
    ])

    with viz_tab1:
        fig = px.histogram(
            demo_df,
            x="Value",
            color="Group",
            barmode="overlay",
            opacity=0.65,
            nbins=max(12, int(stats_results["n_total"] / 4)),
            title="Group Distribution Comparison with Medians",
            color_discrete_sequence=["#3b82f6", "#10b981"]
        )
        fig.add_vline(
            x=stats_results["median_a"],
            line_width=3,
            line_dash="dash",
            line_color="#1d4ed8",
            annotation_text=f"Med A = {stats_results['median_a']:.1f}",
            annotation_position="top left",
            annotation_font=dict(color="#1d4ed8", size=12)
        )
        fig.add_vline(
            x=stats_results["median_b"],
            line_width=3,
            line_dash="solid",
            line_color="#059669",
            annotation_text=f"Med B = {stats_results['median_b']:.1f}",
            annotation_position="top right",
            annotation_font=dict(color="#059669", size=12)
        )
        fig.update_layout(template="plotly_white", xaxis_title="Value", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "💡 **How to interpret this chart & p-value:** "
            "The Mann-Whitney U test evaluates whether values in one group systematically rank higher or lower than the other. "
            "A **p-value < 0.05** indicates a statistically significant result: there is strong evidence that the two groups truly differ "
            "(rejecting the null hypothesis that both groups share the same distribution) — one group consistently tends to produce higher values rather than differing simply by random chance. "
            "When both distributions share a similar spread and shape, this directly reflects a significant difference between the group medians. "
            "Conversely, a **p-value ≥ 0.05** means there is insufficient evidence of a true difference; the observed variation could plausibly be due to random sampling alone."
        )

    with viz_tab2:
        ecdf_fig = px.ecdf(
            demo_df,
            x="Value",
            color="Group",
            title="Empirical Cumulative Distribution Functions (ECDF)",
            color_discrete_sequence=["#3b82f6", "#10b981"]
        )
        ecdf_fig.update_layout(
            template="plotly_white",
            xaxis_title="Value",
            yaxis_title="Cumulative Probability P(X ≤ x)"
        )
        st.plotly_chart(ecdf_fig, width="stretch")
        st.caption("💡 **How to interpret ECDF:** When one group's curve is systematically shifted to the right, its values stochastically dominate the other. The distance between curves directly reflects the Common Language Effect Size (Probability of Superiority).")

    with viz_tab3:
        box_fig = px.box(
            demo_df,
            x="Group",
            y="Value",
            color="Group",
            points="all",
            title="Side-by-Side Boxplots with Individual Observations",
            color_discrete_sequence=["#3b82f6", "#10b981"]
        )
        box_fig.update_layout(template="plotly_white")
        st.plotly_chart(box_fig, width="stretch")

    st.markdown("---")

    # 3. Comprehensive Assumption Diagnostics Section
    st.markdown("### 🛡️ Diagnostic Tests for Core Assumptions")
    st.markdown("""
    Every statistical hypothesis test rests on fundamental assumptions. Below are the diagnostic evaluations 
    verifying whether the Mann-Whitney U Test is appropriate for this dataset:
    """)

    # Render assumption cards in 2x2 grid
    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        status_color = "#10b981" if asm["status"] == "pass" else ("#f59e0b" if asm["status"] == "warning" else "#ef4444")
        bg_color = "#f0fdf4" if asm["status"] == "pass" else ("#fffbeb" if asm["status"] == "warning" else "#fef2f2")
        badge_symbol = "✅" if asm["status"] == "pass" else ("🟡" if asm["status"] == "warning" else "❌")

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # Expandable details for robust alternatives
    with st.expander("🔎 View Robust Alternative Methods: Brunner-Munzel & Welch's t-Test Details", expanded=False):
        st.markdown(f"""
        When distribution shapes or variances differ substantially, the standard Mann-Whitney U test can produce inflated Type I errors:
        
        * **1. Brunner-Munzel Test (Generalized Wilcoxon Test):**
          - **Purpose:** Specifically designed for comparing ordinal or continuous groups with **unequal variances and differing distribution shapes**.
          - **Test Statistic:** `{stats_results['bm_stat']:.3f}`
          - **p-value:** `{stats_results['p_bm']:.4f}`
          - **Conclusion:** {'Statistically significant stochastic superiority (p < 0.05).' if stats_results['p_bm'] < 0.05 else 'No statistically significant difference (p ≥ 0.05).'}
          - **Why this works:** It relaxes the assumption of identical distribution shapes under $H_0$.
        
        * **2. Welch's Independent Two-Sample t-Test:**
          - **Purpose:** Compares population means without assuming equal variances.
          - **t-Statistic:** `t = {stats_results['t_welch']:.3f}`
          - **p-value:** `{stats_results['p_welch']:.4f}`
          - **Conclusion:** {'Statistically significant difference between group means (p < 0.05).' if stats_results['p_welch'] < 0.05 else 'No statistically significant difference between group means (p ≥ 0.05).'}
          - **Why this works:** When sample sizes are moderate-to-large ($N \\ge 30$), the Central Limit Theorem ensures that sample means are approximately normal, making Welch's t-test robust even with skewed data.
        """)


def render_welch_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics for Welch's Two-Sample t-test."""
    st.markdown("### 🎯 Hypothesis Test Results: Group Means Comparison")

    # 1. Metrics Grid with Tooltips
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric(
        f"{stats_results['group_a_name']} Mean (x̄₁)",
        f"{stats_results['mean_a']:.2f}",
        help=f"Sample size n = {stats_results['n_a']}. Std Dev = {stats_results['std_a']:.2f}, Variance = {stats_results['var_a']:.2f}, Median = {stats_results['median_a']:.2f}."
    )
    mcol1.metric(
        f"{stats_results['group_b_name']} Mean (x̄₂)",
        f"{stats_results['mean_b']:.2f}",
        help=f"Sample size n = {stats_results['n_b']}. Std Dev = {stats_results['std_b']:.2f}, Variance = {stats_results['var_b']:.2f}, Median = {stats_results['median_b']:.2f}."
    )

    mcol2.metric(
        "Mean Diff (x̄₂ - x̄₁)",
        f"{stats_results['mean_diff']:+.2f}",
        help="Sample Mean of Group B minus Sample Mean of Group A."
    )
    mcol2.metric(
        "Std Error of Diff (SE)",
        f"{stats_results['se_diff']:.2f}",
        help="Welch standard error: SE = sqrt(s1²/n1 + s2²/n2). Accounts directly for unequal sample variances."
    )

    mcol3.metric(
        "Welch t-Statistic",
        f"t({stats_results['df']:.1f}) = {stats_results['t_stat']:.3f}",
        help=f"Number of standard errors the difference in means is away from zero, with Welch-Satterthwaite df = {stats_results['df']:.1f}."
    )
    sig_text = "p < 0.05 (Significant)" if stats_results["p_val"] < 0.05 else "p ≥ 0.05 (Not Significant)"
    mcol3.metric(
        "p-value",
        f"{stats_results['p_val']:.4f}",
        delta=sig_text,
        delta_color="normal" if stats_results["p_val"] < 0.05 else "off",
        help=(
            "The p-value measures whether the observed difference between group means is likely real or just random luck.\n\n"
            "• p < 0.05 (Statistically Significant): Strong evidence that the true population means differ between the two groups "
            "(rejects the null hypothesis H0: μ1 = μ2). The observed difference is very unlikely to be a fluke of random sampling.\n\n"
            "• p ≥ 0.05 (Not Significant): Insufficient evidence to conclude a real difference in population means; any observed difference "
            "is consistent with random sampling variation."
        )
    )

    mcol4.metric(
        "95% CI for Difference",
        f"[{stats_results['ci_lower']:.2f}, {stats_results['ci_upper']:.2f}]",
        help="95% Confidence Interval for the true difference between population means (μ2 - μ1). If 0 is excluded, the difference is statistically significant."
    )
    mcol4.metric(
        "Cohen's d Effect Size",
        f"{stats_results['cohens_d']:.2f}",
        help="Standardized mean difference: 0.2 = small effect, 0.5 = medium effect, 0.8 = large effect. (Hedges' g correction = " f"{stats_results['hedges_g']:.2f})."
    )

    st.markdown("---")

    # 2. Interactive Visualization Tabs
    st.markdown("### 📈 Visual Comparison & Distribution")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📊 Group Distributions & Sample Means",
        "📐 Normal Q-Q Plots (Both Groups)",
        "📦 Boxplots & Individual Observations"
    ])

    with viz_tab1:
        fig = px.histogram(
            demo_df,
            x="Value",
            color="Group",
            barmode="overlay",
            opacity=0.65,
            nbins=max(12, int(stats_results["n_total"] / 4)),
            title="Group Distribution Comparison with Sample Means",
            color_discrete_sequence=["#3b82f6", "#10b981"]
        )
        fig.add_vline(
            x=stats_results["mean_a"],
            line_width=3,
            line_dash="dash",
            line_color="#1d4ed8",
            annotation_text=f"Mean A = {stats_results['mean_a']:.1f}",
            annotation_position="top left",
            annotation_font=dict(color="#1d4ed8", size=12)
        )
        fig.add_vline(
            x=stats_results["mean_b"],
            line_width=3,
            line_dash="solid",
            line_color="#059669",
            annotation_text=f"Mean B = {stats_results['mean_b']:.1f}",
            annotation_position="top right",
            annotation_font=dict(color="#059669", size=12)
        )
        fig.update_layout(template="plotly_white", xaxis_title="Value", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "💡 **How to interpret this chart & p-value:** "
            "This chart displays the distributions of both groups along with their sample means (dashed blue line for Group A, solid green line for Group B). "
            "A **p-value < 0.05** indicates a statistically significant difference: there is strong evidence that the true population means genuinely differ "
            "(rejecting the null hypothesis that both groups have identical means) — the gap between group averages is unlikely to be due to chance sampling variation. "
            "Conversely, a **p-value ≥ 0.05** means the data is consistent with the null hypothesis, and the observed difference could easily have occurred by random variation alone."
        )

    with viz_tab2:
        qq_fig = go.Figure()
        # Group A Q-Q
        qq_fig.add_trace(go.Scatter(
            x=stats_results["qq_osm_a"],
            y=stats_results["qq_osr_a"],
            mode="markers",
            marker=dict(color="#2563eb", size=6, opacity=0.8),
            name=f"{stats_results['group_a_name']} Quantiles"
        ))
        x_trend_a = [min(stats_results["qq_osm_a"]), max(stats_results["qq_osm_a"])]
        y_trend_a = [stats_results["qq_slope_a"] * x + stats_results["qq_intercept_a"] for x in x_trend_a]
        qq_fig.add_trace(go.Scatter(
            x=x_trend_a,
            y=y_trend_a,
            mode="lines",
            line=dict(color="#1d4ed8", width=2, dash="dash"),
            name=f"{stats_results['group_a_name']} Normal Fit"
        ))

        # Group B Q-Q
        qq_fig.add_trace(go.Scatter(
            x=stats_results["qq_osm_b"],
            y=stats_results["qq_osr_b"],
            mode="markers",
            marker=dict(color="#059669", size=6, opacity=0.8),
            name=f"{stats_results['group_b_name']} Quantiles"
        ))
        x_trend_b = [min(stats_results["qq_osm_b"]), max(stats_results["qq_osm_b"])]
        y_trend_b = [stats_results["qq_slope_b"] * x + stats_results["qq_intercept_b"] for x in x_trend_b]
        qq_fig.add_trace(go.Scatter(
            x=x_trend_b,
            y=y_trend_b,
            mode="lines",
            line=dict(color="#047857", width=2, dash="solid"),
            name=f"{stats_results['group_b_name']} Normal Fit"
        ))

        qq_fig.update_layout(
            title="Normal Q-Q (Quantile-Quantile) Diagnostic Plots for Both Groups",
            xaxis_title="Theoretical Normal Quantiles",
            yaxis_title="Ordered Sample Values",
            template="plotly_white"
        )
        st.plotly_chart(qq_fig, width="stretch")
        st.caption("💡 **How to interpret Q-Q plots:** Points that follow the straight diagonal line indicate normally distributed data. S-curves or deviations at the ends indicate skewness or heavy tails.")

    with viz_tab3:
        box_fig = px.box(
            demo_df,
            x="Group",
            y="Value",
            color="Group",
            points="all",
            title="Side-by-Side Boxplots with Jittered Observations",
            color_discrete_sequence=["#3b82f6", "#10b981"]
        )
        box_fig.update_layout(template="plotly_white")
        st.plotly_chart(box_fig, width="stretch")

    st.markdown("---")

    # 3. Comprehensive Assumption Diagnostics Section
    st.markdown("### 🛡️ Diagnostic Tests for Core Assumptions")
    st.markdown("""
    Every statistical hypothesis test rests on fundamental assumptions. Below are the diagnostic evaluations 
    verifying whether Welch's Two-Sample t-Test is appropriate for this dataset:
    """)

    # Render assumption cards in 2x2 grid
    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        status_color = "#10b981" if asm["status"] == "pass" else ("#f59e0b" if asm["status"] == "warning" else "#ef4444")
        bg_color = "#f0fdf4" if asm["status"] == "pass" else ("#fffbeb" if asm["status"] == "warning" else "#fef2f2")
        badge_symbol = "✅" if asm["status"] == "pass" else ("🟡" if asm["status"] == "warning" else "❌")

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # Expandable details for non-parametric alternative
    with st.expander("🔎 View Non-Parametric Alternative: Mann-Whitney U Test Details", expanded=False):
        st.markdown(f"""
        When normality is violated in small samples ($N < 30$) or extreme outliers exist, the **Mann-Whitney U Test** is the recommended non-parametric alternative:
        
        * **Hypothesis Tested:** The rank distribution of {stats_results['group_b_name']} is identical to {stats_results['group_a_name']}.
        * **Mann-Whitney U Statistic:** `{stats_results['u_stat']:.1f}`
        * **p-value:** `{stats_results['p_mann']:.4f}`
        * **Conclusion:** {'Statistically significant difference in rank distributions (p < 0.05).' if stats_results['p_mann'] < 0.05 else 'No statistically significant difference in rank distributions (p ≥ 0.05).'}
        * **Why this works:** The Mann-Whitney U test converts values into ranks across both groups, completely avoiding distortions from non-normal skewness and extreme outliers.
        """)


def render_students_t_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics for Student's Two-Sample t-test."""
    st.markdown("### 🎯 Hypothesis Test Results: Group Means Comparison (Equal Variance)")

    # 1. Metrics Grid with Tooltips
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric(
        f"{stats_results['group_a_name']} Mean (x̄₁)",
        f"{stats_results['mean_a']:.2f}",
        help=f"Sample size n = {stats_results['n_a']}. Std Dev = {stats_results['std_a']:.2f}, Variance = {stats_results['var_a']:.2f}, Median = {stats_results['median_a']:.2f}."
    )
    mcol1.metric(
        f"{stats_results['group_b_name']} Mean (x̄₂)",
        f"{stats_results['mean_b']:.2f}",
        help=f"Sample size n = {stats_results['n_b']}. Std Dev = {stats_results['std_b']:.2f}, Variance = {stats_results['var_b']:.2f}, Median = {stats_results['median_b']:.2f}."
    )

    mcol2.metric(
        "Mean Diff (x̄₂ - x̄₁)",
        f"{stats_results['mean_diff']:+.2f}",
        help="Sample Mean of Group B minus Sample Mean of Group A."
    )
    mcol2.metric(
        "Pooled SE of Diff (SE)",
        f"{stats_results['se_diff']:.2f}",
        help=f"Pooled standard error: SE = sp * sqrt(1/n1 + 1/n2), where pooled SD sp = {stats_results['sp']:.2f}. Assumes equal variances."
    )

    mcol3.metric(
        "Student's t-Statistic",
        f"t({stats_results['df']}) = {stats_results['t_stat']:.3f}",
        help=f"Student's t-statistic with exact integer degrees of freedom df = n1 + n2 - 2 = {stats_results['df']}."
    )
    sig_text = "p < 0.05 (Significant)" if stats_results["p_val"] < 0.05 else "p ≥ 0.05 (Not Significant)"
    mcol3.metric(
        "p-value",
        f"{stats_results['p_val']:.4f}",
        delta=sig_text,
        delta_color="normal" if stats_results["p_val"] < 0.05 else "off",
        help=(
            "The p-value measures whether the observed difference between group means is likely real or due to random chance.\n\n"
            "• p < 0.05 (Statistically Significant): Strong evidence that true population means differ (rejects H0: μ1 = μ2).\n\n"
            "• p ≥ 0.05 (Not Significant): Insufficient evidence to conclude that population means differ."
        )
    )

    mcol4.metric(
        "95% CI for Difference",
        f"[{stats_results['ci_lower']:.2f}, {stats_results['ci_upper']:.2f}]",
        help="95% Confidence Interval for the difference between population means (μ2 - μ1) using pooled variance. If 0 is excluded, the difference is statistically significant."
    )
    mcol4.metric(
        "Pooled Cohen's d",
        f"{stats_results['cohens_d']:.2f}",
        help=f"Standardized mean difference relative to pooled standard deviation sp ({stats_results['sp']:.2f}): 0.2 = small, 0.5 = medium, 0.8 = large effect. (Hedges' g correction = {stats_results['hedges_g']:.2f})."
    )

    st.markdown("---")

    # 2. Interactive Visualization Tabs
    st.markdown("### 📈 Visual Comparison & Distribution")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📊 Group Distributions & Sample Means",
        "📐 Normal Q-Q Plots (Both Groups)",
        "📦 Boxplots & Individual Observations"
    ])

    with viz_tab1:
        fig = px.histogram(
            demo_df,
            x="Value",
            color="Group",
            barmode="overlay",
            opacity=0.65,
            nbins=max(12, int(stats_results["n_total"] / 4)),
            title="Group Distribution Comparison with Sample Means",
            color_discrete_sequence=["#3b82f6", "#10b981"]
        )
        fig.add_vline(
            x=stats_results["mean_a"],
            line_width=3,
            line_dash="dash",
            line_color="#1d4ed8",
            annotation_text=f"Mean A = {stats_results['mean_a']:.1f}",
            annotation_position="top left",
            annotation_font=dict(color="#1d4ed8", size=12)
        )
        fig.add_vline(
            x=stats_results["mean_b"],
            line_width=3,
            line_dash="solid",
            line_color="#059669",
            annotation_text=f"Mean B = {stats_results['mean_b']:.1f}",
            annotation_position="top right",
            annotation_font=dict(color="#059669", size=12)
        )
        fig.update_layout(template="plotly_white", xaxis_title="Value", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "💡 **How to interpret this chart & p-value:** "
            "This chart displays the distributions of both groups along with their sample means (dashed blue line for Group A, solid green line for Group B). "
            "A **p-value < 0.05** indicates a statistically significant difference: there is strong evidence that the true population means genuinely differ "
            "(rejecting the null hypothesis that both groups have identical means). "
            "Conversely, a **p-value ≥ 0.05** means the observed gap is consistent with random sampling variation."
        )

    with viz_tab2:
        qq_fig = go.Figure()
        # Group A Q-Q
        qq_fig.add_trace(go.Scatter(
            x=stats_results["qq_osm_a"],
            y=stats_results["qq_osr_a"],
            mode="markers",
            marker=dict(color="#2563eb", size=6, opacity=0.8),
            name=f"{stats_results['group_a_name']} Quantiles"
        ))
        x_trend_a = [min(stats_results["qq_osm_a"]), max(stats_results["qq_osm_a"])]
        y_trend_a = [stats_results["qq_slope_a"] * x + stats_results["qq_intercept_a"] for x in x_trend_a]
        qq_fig.add_trace(go.Scatter(
            x=x_trend_a,
            y=y_trend_a,
            mode="lines",
            line=dict(color="#1d4ed8", width=2, dash="dash"),
            name=f"{stats_results['group_a_name']} Normal Fit"
        ))

        # Group B Q-Q
        qq_fig.add_trace(go.Scatter(
            x=stats_results["qq_osm_b"],
            y=stats_results["qq_osr_b"],
            mode="markers",
            marker=dict(color="#059669", size=6, opacity=0.8),
            name=f"{stats_results['group_b_name']} Quantiles"
        ))
        x_trend_b = [min(stats_results["qq_osm_b"]), max(stats_results["qq_osm_b"])]
        y_trend_b = [stats_results["qq_slope_b"] * x + stats_results["qq_intercept_b"] for x in x_trend_b]
        qq_fig.add_trace(go.Scatter(
            x=x_trend_b,
            y=y_trend_b,
            mode="lines",
            line=dict(color="#047857", width=2, dash="solid"),
            name=f"{stats_results['group_b_name']} Normal Fit"
        ))

        qq_fig.update_layout(
            title="Normal Q-Q (Quantile-Quantile) Diagnostic Plots for Both Groups",
            xaxis_title="Theoretical Normal Quantiles",
            yaxis_title="Ordered Sample Values",
            template="plotly_white"
        )
        st.plotly_chart(qq_fig, width="stretch")
        st.caption("💡 **How to interpret Q-Q plots:** Points that follow the straight diagonal line indicate normally distributed data. S-curves or deviations at the ends indicate skewness or heavy tails.")

    with viz_tab3:
        box_fig = px.box(
            demo_df,
            x="Group",
            y="Value",
            color="Group",
            points="all",
            title="Side-by-Side Boxplots with Jittered Observations",
            color_discrete_sequence=["#3b82f6", "#10b981"]
        )
        box_fig.update_layout(template="plotly_white")
        st.plotly_chart(box_fig, width="stretch")

    st.markdown("---")

    # 3. Comprehensive Assumption Diagnostics Section
    st.markdown("### 🛡️ Diagnostic Tests for Core Assumptions")
    st.markdown("""
    Every statistical hypothesis test rests on fundamental assumptions. Below are the diagnostic evaluations 
    verifying whether Student's Two-Sample t-Test is appropriate for this dataset:
    """)

    # Render assumption cards in 2-column layout
    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        status_color = "#10b981" if asm["status"] == "pass" else ("#f59e0b" if asm["status"] == "warning" else "#ef4444")
        bg_color = "#f0fdf4" if asm["status"] == "pass" else ("#fffbeb" if asm["status"] == "warning" else "#fef2f2")
        badge_symbol = "✅" if asm["status"] == "pass" else ("🟡" if asm["status"] == "warning" else "❌")

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # Expandable details for alternatives
    has_var_violation = any(a["name"].startswith("5.") and a["status"] == "fail" for a in assumptions)
    with st.expander("🔎 View Alternative A: Welch's Two-Sample t-Test (When Variances are Unequal)", expanded=False):
        st.markdown(f"""
        When the assumption of equal population variances is violated (heteroscedasticity), **Welch's Two-Sample t-Test** is the gold standard parametric solution:
        
        * **Welch t-Statistic:** `{stats_results['t_welch']:.3f}`
        * **Adjusted Welch Degrees of Freedom:** `{stats_results['df_welch']:.1f}`
        * **p-value:** `{stats_results['p_welch']:.4f}`
        * **Conclusion:** {'Statistically significant difference in population means (p < 0.05).' if stats_results['p_welch'] < 0.05 else 'No statistically significant difference in population means (p ≥ 0.05).'}
        * **Why this works:** Welch's test does not pool variances ($s_p^2$). Instead, it computes separate variance estimates for each group ($s_1^2/n_1 + s_2^2/n_2$) and applies the Welch-Satterthwaite adjustment to degrees of freedom, completely preventing Type I error inflation.
        """)

    has_norm_violation = any(a["name"].startswith("3.") and a["status"] == "fail" for a in assumptions) or any(a["name"].startswith("4.") and a["status"] == "fail" for a in assumptions)
    with st.expander("🔎 View Alternative B: Mann-Whitney U Test (When Normality Fails or Outliers Exist)", expanded=False):
        st.markdown(f"""
        When normality is violated in small samples ($N < 30$) or extreme outliers exist, the **Mann-Whitney U Test** is the recommended non-parametric alternative:
        
        * **Hypothesis Tested:** The rank distribution of {stats_results['group_b_name']} is identical to {stats_results['group_a_name']}.
        * **Mann-Whitney U Statistic:** `{stats_results['u_stat']:.1f}`
        * **p-value:** `{stats_results['p_mann']:.4f}`
        * **Conclusion:** {'Statistically significant difference in rank distributions (p < 0.05).' if stats_results['p_mann'] < 0.05 else 'No statistically significant difference in rank distributions (p ≥ 0.05).'}
        * **Why this works:** The Mann-Whitney U test converts values into ranks across both groups, completely avoiding distortions from non-normal skewness and extreme outliers.
        """)


def render_paired_t_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics for the Paired Samples t-test."""
    st.markdown("### 🎯 Hypothesis Test Results: Paired Before-After Comparison")

    # 1. Metrics Grid with Tooltips
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric(
        "Baseline / Pre Mean",
        f"{stats_results['mean_pre']:.2f}",
        help=f"Sample size n = {stats_results['n_pairs']} matched pairs. Baseline Pre Std Dev = {stats_results['std_pre']:.2f}."
    )
    mcol1.metric(
        "Follow-up / Post Mean",
        f"{stats_results['mean_post']:.2f}",
        help=f"Sample size n = {stats_results['n_pairs']} matched pairs. Post Std Dev = {stats_results['std_post']:.2f}."
    )

    mcol2.metric(
        "Mean Difference (d̄)",
        f"{stats_results['mean_difference']:+.2f}",
        help="Sample Mean of Pairwise Differences (Post minus Pre)."
    )
    mcol2.metric(
        "Std Error of Diff (SE)",
        f"{stats_results['se_diff']:.2f}",
        help=f"Standard error of the mean difference: SE = s_d / sqrt(n), where s_d = {stats_results['std_difference']:.2f}."
    )

    mcol3.metric(
        "Paired t-Statistic",
        f"t({stats_results['df']}) = {stats_results['t_stat']:.3f}",
        help=f"Paired t-statistic with exact integer degrees of freedom df = n - 1 = {stats_results['df']}."
    )
    sig_text = "p < 0.05 (Significant)" if stats_results["p_val"] < 0.05 else "p ≥ 0.05 (Not Significant)"
    mcol3.metric(
        "p-value",
        f"{stats_results['p_val']:.4f}",
        delta=sig_text,
        delta_color="normal" if stats_results["p_val"] < 0.05 else "off",
        help=(
            "The p-value measures whether the observed paired change is likely real or due to random chance.\n\n"
            "• p < 0.05 (Statistically Significant): Strong evidence of a genuine within-subject change (rejects H0: μD = 0).\n\n"
            "• p ≥ 0.05 (Not Significant): Insufficient evidence to conclude that the true mean difference differs from zero."
        )
    )

    mcol4.metric(
        "95% CI for Mean Diff",
        f"[{stats_results['ci_lower']:.2f}, {stats_results['ci_upper']:.2f}]",
        help="95% Confidence Interval for the true mean difference (μD). If 0 is excluded, the change is statistically significant."
    )
    mcol4.metric(
        "Cohen's d_z Effect Size",
        f"{stats_results['cohens_dz']:.2f}",
        help=f"Standardized paired effect size d_z = d̄ / s_d: 0.2 = small, 0.5 = medium, 0.8 = large effect. (Hedges' g_av = {stats_results['hedges_gav']:.2f})."
    )

    st.markdown("---")

    # 2. Interactive Visualization Tabs
    st.markdown("### 📈 Visual Comparison & Distribution")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📈 Paired Trajectories per Subject",
        "📊 Distribution of Differences (Post - Pre)",
        "📐 Normal Q-Q Plot of Differences"
    ])

    with viz_tab1:
        fig = go.Figure()
        # Add paired lines
        for _, row in demo_df.iterrows():
            fig.add_trace(go.Scatter(
                x=["Pre", "Post"],
                y=[row["Pre_Intervention"], row["Post_Intervention"]],
                mode="lines+markers",
                line=dict(color="rgba(100, 116, 139, 0.35)", width=1.5),
                marker=dict(size=4, color="#64748b"),
                showlegend=False
            ))
        # Add group means trend
        fig.add_trace(go.Scatter(
            x=["Pre", "Post"],
            y=[stats_results["mean_pre"], stats_results["mean_post"]],
            mode="lines+markers",
            line=dict(color="#ef4444", width=3.5),
            marker=dict(size=12, color="#ef4444"),
            name="Mean Trend"
        ))
        fig.update_layout(
            title="Individual Paired Trajectories (Pre vs Post) with Mean Trend",
            xaxis_title="Measurement Occasion",
            yaxis_title="Score / Value",
            template="plotly_white"
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "💡 **How to interpret this chart:** Each faint gray line tracks a single individual subject across the two occasions. "
            "The bold red line represents the overall mean trajectory from Pre to Post. Consistent slopes indicate a reliable paired effect."
        )

    with viz_tab2:
        diff_fig = px.histogram(
            demo_df,
            x="Difference",
            nbins=max(12, int(stats_results["n_pairs"] / 3)),
            title="Distribution of Pairwise Differences (Post - Pre)",
            color_discrete_sequence=["#3b82f6"],
            opacity=0.75
        )
        diff_fig.add_vline(
            x=stats_results["mean_difference"],
            line_width=3,
            line_dash="dash",
            line_color="#1d4ed8",
            annotation_text=f"Mean Diff = {stats_results['mean_difference']:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="#1d4ed8", size=12)
        )
        diff_fig.add_vline(
            x=0.0,
            line_width=2,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text="Null (0.0)",
            annotation_position="top right",
            annotation_font=dict(color="#dc2626", size=12)
        )
        diff_fig.update_layout(template="plotly_white", xaxis_title="Difference Score (Post - Pre)", yaxis_title="Count")
        st.plotly_chart(diff_fig, width="stretch")
        st.caption(
            "💡 **How to interpret difference scores:** The distribution shows the frequency of change scores ($D = Post - Pre$). "
            "If the distribution is centered away from zero (dashed blue line) and zero (dotted red line) is outside the 95% CI, "
            "there is statistically significant evidence of a treatment effect."
        )

    with viz_tab3:
        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(
            x=stats_results["qq_osm"],
            y=stats_results["qq_osr"],
            mode="markers",
            marker=dict(color="#2563eb", size=6, opacity=0.8),
            name="Difference Quantiles"
        ))
        x_trend = [min(stats_results["qq_osm"]), max(stats_results["qq_osm"])]
        y_trend = [stats_results["qq_slope"] * x + stats_results["qq_intercept"] for x in x_trend]
        qq_fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode="lines",
            line=dict(color="#1d4ed8", width=2, dash="dash"),
            name="Normal Fit"
        ))
        qq_fig.update_layout(
            title="Normal Q-Q (Quantile-Quantile) Plot of Difference Scores",
            xaxis_title="Theoretical Normal Quantiles",
            yaxis_title="Ordered Sample Difference Scores",
            template="plotly_white"
        )
        st.plotly_chart(qq_fig, width="stretch")
        st.caption("💡 **How to interpret Q-Q plot:** Points along the diagonal line indicate that the pairwise differences are normally distributed.")

    st.markdown("---")

    # 3. Comprehensive Assumption Diagnostics Section
    st.markdown("### 🛡️ Diagnostic Tests for Core Assumptions")
    st.markdown("""
    Every statistical hypothesis test rests on fundamental assumptions. Below are the diagnostic evaluations 
    verifying whether the Paired Samples t-Test is appropriate for this dataset:
    """)

    # Render assumption cards in 2-column layout
    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        status_color = "#10b981" if asm["status"] == "pass" else ("#f59e0b" if asm["status"] == "warning" else "#ef4444")
        bg_color = "#f0fdf4" if asm["status"] == "pass" else ("#fffbeb" if asm["status"] == "warning" else "#fef2f2")
        badge_symbol = "✅" if asm["status"] == "pass" else ("🟡" if asm["status"] == "warning" else "❌")

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # Expandable details for non-parametric alternative
    with st.expander("🔎 View Non-Parametric Alternative: Wilcoxon Signed-Rank Test (Paired)", expanded=False):
        st.markdown(f"""
        When pairwise difference scores violate normality in small samples ($N < 30$) or contain extreme outliers, 
        the **Wilcoxon Signed-Rank Test (Paired)** is the recommended non-parametric alternative:
        
        * **Hypothesis Tested:** The median of pairwise differences equals zero ($\text{Median}_D = 0$).
        * **Wilcoxon W Statistic:** `{stats_results['wilcoxon_stat']:.1f}`
        * **p-value:** `{stats_results['p_wilcoxon']:.4f}`
        * **Conclusion:** {'Statistically significant difference between paired measures (p < 0.05).' if stats_results['p_wilcoxon'] < 0.05 else 'No statistically significant difference between paired measures (p ≥ 0.05).'}
        * **Why this works:** The Wilcoxon Signed-Rank test ranks the absolute differences and analyzes their positive and negative signs, making it immune to extreme outliers and severe non-normality in difference scores.
        """)


def render_wilcoxon_paired_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics for Paired Wilcoxon Signed-Rank Test."""
    st.markdown("### 🎯 Hypothesis Test Results: Paired Wilcoxon Signed-Rank Test")

    # 1. Metrics Grid with Tooltips
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric(
        "Sample Size (Matched Pairs)",
        f"{stats_results['n_pairs']}",
        help="Total number of matched or repeated measurement pairs."
    )
    mcol1.metric(
        "Effective Pairs (N_eff)",
        f"{stats_results['n_eff']} (ties: {stats_results['n_zeros']})",
        help="Pairs with non-zero difference included in ranking (pairs where Post == Pre are excluded)."
    )

    mcol2.metric(
        "Pre Median",
        f"{stats_results['median_pre']:.1f}",
        help="Median baseline measurement."
    )
    mcol2.metric(
        "Post Median",
        f"{stats_results['median_post']:.1f}",
        help="Median post-intervention measurement."
    )

    mcol3.metric(
        "Observed Median Diff",
        f"{stats_results['median_diff']:+.1f}",
        help="Sample median of pairwise differences (Post - Pre)."
    )
    mcol3.metric(
        "Hodges-Lehmann Shift",
        f"{stats_results['hl_estimate']:+.2f}",
        help="Median of all pairwise Walsh averages (d_i + d_j)/2."
    )

    mcol4.metric(
        "Wilcoxon Statistic",
        f"W = {stats_results['w_stat']:.1f}",
        help="Minimum of positive rank sum (W+) and negative rank sum (W-)."
    )
    sig_text = "p < 0.05 (Significant)" if stats_results["p_wilcoxon"] < 0.05 else "p ≥ 0.05 (Not Significant)"
    mcol4.metric(
        "p-value",
        f"{stats_results['p_wilcoxon']:.4f}",
        delta=sig_text,
        delta_color="normal" if stats_results["p_wilcoxon"] < 0.05 else "off",
        help=(
            "The p-value evaluates the null hypothesis that the median difference between paired observations equals zero.\n\n"
            "• p < 0.05 (Significant): Statistically significant shift between paired conditions (reject H₀).\n\n"
            "• p ≥ 0.05 (Not Significant): Insufficient evidence of a consistent shift between paired conditions."
        )
    )

    st.markdown("---")

    # 2. Interactive Plotly Visualizations in Tabs
    tab_slopes, tab_diffs, tab_ranks = st.tabs([
        "📉 Paired Trajectories (Slope Chart)",
        "📊 Difference Distribution & Symmetry",
        "🏷️ Signed Ranks Breakdown"
    ])

    with tab_slopes:
        slope_fig = go.Figure()
        # Add individual trajectory lines
        for _, row in demo_df.iterrows():
            d = row["Difference"]
            if d > 0:
                line_color = "rgba(16, 185, 129, 0.7)"  # Emerald Green (Increased)
                line_width = 1.5
            elif d < 0:
                line_color = "rgba(239, 68, 68, 0.7)"   # Coral Red (Decreased)
                line_width = 1.5
            else:
                line_color = "rgba(148, 163, 184, 0.5)" # Gray dashed (No change)
                line_width = 1.0

            slope_fig.add_trace(go.Scatter(
                x=["Pre-Intervention", "Post-Intervention"],
                y=[row["Pre"], row["Post"]],
                mode="lines+markers",
                line=dict(color=line_color, width=line_width),
                marker=dict(size=5),
                hoverinfo="text",
                hovertext=f"{row['Subject_ID']}: Pre={row['Pre']}, Post={row['Post']} (Diff: {d:+.1f})",
                showlegend=False
            ))

        # Add median trend line
        slope_fig.add_trace(go.Scatter(
            x=["Pre-Intervention", "Post-Intervention"],
            y=[stats_results["median_pre"], stats_results["median_post"]],
            mode="lines+markers",
            line=dict(color="#1e3a8a", width=4),
            marker=dict(size=10, color="#1e3a8a"),
            name="Median Trajectory",
            hoverinfo="text",
            hovertext=f"Medians: Pre={stats_results['median_pre']:.1f} ➔ Post={stats_results['median_post']:.1f} (Shift: {stats_results['median_diff']:+.1f})"
        ))

        slope_fig.update_layout(
            title="Individual Participant Pre ➔ Post Trajectories",
            xaxis_title="Condition / Measurement Occasion",
            yaxis_title="Score / Measurement Value",
            template="plotly_white",
            hovermode="closest"
        )
        st.plotly_chart(slope_fig, use_container_width=True)
        st.caption("💡 **Interpretation:** Green lines indicate an increase (Post > Pre); red lines indicate a decrease (Post < Pre); thick blue line tracks group median shift.")

    with tab_diffs:
        diff_fig = go.Figure()
        diff_fig.add_trace(go.Histogram(
            x=demo_df["Difference"],
            nbinsx=15,
            marker_color="#3b82f6",
            opacity=0.75,
            name="Differences (Post - Pre)"
        ))
        diff_fig.add_vline(
            x=0.0,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2,
            annotation_text="Null (Diff = 0)",
            annotation_position="top left"
        )
        diff_fig.add_vline(
            x=stats_results["median_diff"],
            line_dash="solid",
            line_color="#10b981",
            line_width=2,
            annotation_text=f"Median Diff = {stats_results['median_diff']:+.1f}",
            annotation_position="top right"
        )
        diff_fig.update_layout(
            title="Distribution of Pairwise Difference Scores (Post - Pre)",
            xaxis_title="Difference Score (Post - Pre)",
            yaxis_title="Frequency Count",
            template="plotly_white"
        )
        st.plotly_chart(diff_fig, use_container_width=True)
        st.caption(f"💡 **Symmetry Check:** Skewness = {stats_results['skew_diffs']:+.3f}. Symmetry around 0 confirms validity of the Wilcoxon median test.")

    with tab_ranks:
        # Rank comparison bar chart
        pos_sum = stats_results["sum_pos_ranks"]
        neg_sum = stats_results["sum_neg_ranks"]
        rank_fig = go.Figure(data=[
            go.Bar(
                x=["Positive Ranks (W+)", "Negative Ranks (W-)"],
                y=[pos_sum, neg_sum],
                marker_color=["#10b981", "#ef4444"],
                text=[f"W+ = {pos_sum:.1f}", f"W- = {neg_sum:.1f}"],
                textposition="auto"
            )
        ])
        rank_fig.update_layout(
            title="Sum of Signed Ranks: Positive Shifts vs. Negative Shifts",
            xaxis_title="Rank Category",
            yaxis_title="Sum of Mid-Ranks",
            template="plotly_white"
        )
        st.plotly_chart(rank_fig, use_container_width=True)
        st.caption(f"💡 **Rank-Biserial Effect Size:** r_rb = {stats_results['r_rb']:+.3f} (Ranges from -1.0 to +1.0). Wilcoxon W = {stats_results['w_stat']:.1f}.")

    st.markdown("---")

    # 3. Comprehensive Assumption Diagnostics Section
    st.markdown("### 🛡️ Diagnostic Tests for Core Assumptions")
    st.markdown("""
    Below are the diagnostic evaluations verifying whether the Wilcoxon Signed-Rank Test (Paired) 
    is appropriate for this dataset:
    """)

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        status_color = "#10b981" if asm["status"] == "pass" else ("#f59e0b" if asm["status"] == "warning" else "#ef4444")
        bg_color = "#f0fdf4" if asm["status"] == "pass" else ("#fffbeb" if asm["status"] == "warning" else "#fef2f2")
        badge_symbol = "✅" if asm["status"] == "pass" else ("🟡" if asm["status"] == "warning" else "❌")

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # Expandable details for alternative tests
    with st.expander("🔎 View Alternative Test Details (Exact Sign Test & Paired t-test)", expanded=False):
        st.markdown(f"""
        * **Alternative 1: Exact Sign Test (Zero Symmetry Assumptions):**
          * Positive shifts ($Post > Pre$): `{stats_results['pos_count']}` | Negative shifts ($Post < Pre$): `{stats_results['neg_count']}`
          * Two-tailed Exact Sign Test **p-value = `{stats_results['p_sign_test']:.4f}`**
          * Conclusion: {'Statistically significant shift under Sign Test (p < 0.05).' if stats_results['p_sign_test'] < 0.05 else 'No statistically significant shift under Sign Test (p ≥ 0.05).'}
          * Use when difference scores are severely skewed ($|\text{{skew}}| > 1.5$) or ordinal categories lack equal spacing.

        * **Alternative 2: Paired Samples t-test (Parametric Baseline):**
          * Paired t-statistic: `t = {stats_results['t_stat']:.3f}` | **p-value = `{stats_results['p_ttest']:.4f}`**
          * Use when difference scores are verified to be approximately normal and free of extreme outliers.
        """)


def render_one_way_anova_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics for One-Way ANOVA."""
    st.markdown("### 🎯 Hypothesis Test Results: One-Way Analysis of Variance (ANOVA)")

    # 1. Metrics Grid with Tooltips
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric(
        "Omnibus F-Statistic",
        f"F = {stats_results['f_stat']:.3f}",
        help="Ratio of variance between groups to variance within groups (MS_between / MS_within)."
    )
    sig_text = "p < 0.05 (Significant)" if stats_results["p_anova"] < 0.05 else "p ≥ 0.05 (Not Significant)"
    mcol1.metric(
        "ANOVA p-value",
        f"{stats_results['p_anova']:.4e}",
        delta=sig_text,
        delta_color="normal" if stats_results["p_anova"] < 0.05 else "off",
        help=(
            "The omnibus ANOVA p-value evaluates whether all group means are equal (H₀: μ₁ = μ₂ = μ₃).\n\n"
            "• p < 0.05 (Significant): At least one group mean significantly differs from the others (reject H₀).\n\n"
            "• p ≥ 0.05 (Not Significant): Insufficient evidence to reject equality of group means."
        )
    )

    mcol2.metric(
        "Total Sample Size (N)",
        f"{stats_results['n_total']} (k = {stats_results['k_groups']} groups)",
        help=f"Balanced design with n = {stats_results['n_per_group']} participants per group."
    )
    mcol2.metric(
        "Grand Mean",
        f"{stats_results['grand_mean']:.2f}",
        help="Overall mean score across all participants and groups combined."
    )

    eta_sq = stats_results["eta_squared"]
    eta_badge = "Large (≥0.14)" if eta_sq >= 0.14 else ("Medium (≥0.06)" if eta_sq >= 0.06 else "Small (<0.06)")
    mcol3.metric(
        "Eta-Squared (η²)",
        f"{eta_sq:.3f}",
        delta=eta_badge,
        delta_color="normal" if eta_sq >= 0.06 else "off",
        help="Proportion of total score variance explained by group membership (SS_between / SS_total)."
    )
    mcol3.metric(
        "Omega-Squared (ω²)",
        f"{stats_results['omega_squared']:.3f}",
        help="Less biased population effect size estimate that adjusts for sample size."
    )

    mcol4.metric(
        "Levene p (Homoscedasticity)",
        f"{stats_results['p_levene']:.4f}",
        delta=f"F_max = {stats_results['var_ratio']:.2f}",
        delta_color="normal" if stats_results["p_levene"] >= 0.05 else "off",
        help=(
            "Levene's test evaluates whether group variances are equal.\n\n"
            "• p ≥ 0.05 (Homoscedastic): Variances are equal across groups; ANOVA F-test valid.\n\n"
            "• p < 0.05 & F_max > 4.0: Heteroscedastic; switch to Welch's ANOVA."
        )
    )
    mcol4.metric(
        "Kruskal-Wallis p-value",
        f"{stats_results['p_kruskal']:.4e}",
        help="Non-parametric alternative evaluating rank sum differences across the 3 groups."
    )

    st.markdown("---")

    # 2. Dedicated Post-Hoc Pairwise Comparisons Breakdown (Tukey's HSD)
    st.markdown("### 🔍 Which Groups Are Actually Different? (Post-Hoc Pairwise Comparisons)")

    comp_data = stats_results["pairwise_comparisons"]
    sig_count = sum(1 for c in comp_data if c["significant"])

    if stats_results["p_anova"] < 0.05:
        st.markdown(f"""
        The omnibus ANOVA test confirms that group means differ overall ($p < 0.05$). 
        Below, **Tukey's Honestly Significant Difference (HSD) post-hoc test** examines each pair of groups 
        to identify **exactly which groups differ from each other**, while strictly protecting against false alarms.
        """)
    else:
        st.info("""
        ℹ️ **Omnibus test not significant ($p \ge 0.05$):** Under standard scientific testing procedures (Fisher's protection principle), 
        post-hoc tests are not statistically indicated because any observed differences between pairs are consistent with random chance. 
        The pairwise comparisons are shown below for exploratory inspection only.
        """)

    posthoc_rows = []
    for c in comp_data:
        posthoc_rows.append({
            "Pairwise_Comparison": c["comparison"],
            "Mean_Difference": c["diff_str"],
            "95%_CI_Tukey": c["ci_str"],
            "Adjusted_p_value": c["p_adj_str"],
            "Significance_Verdict": c["significance_verdict"],
            "Plain_English_Finding": c["plain_english"]
        })
    posthoc_df = pd.DataFrame(posthoc_rows)
    render_table(posthoc_df, height=180)

    with st.expander("📘 Plain English Guide: What Are Post-Hoc Tests & Why Are They Needed?", expanded=False):
        st.markdown(r"""
        #### 1. What is a Post-Hoc Test?
        * The term **"post-hoc"** is Latin for *"after this"*. 
        * An **omnibus test** (like ANOVA) tells you *whether* an overall difference exists somewhere among the groups, like a building smoke alarm that sounds when smoke is detected anywhere in the building. However, the smoke alarm **cannot tell you which specific room has the fire**.
        * **Post-Hoc tests** are the targeted follow-up inspections that check each room pair-by-pair to pinpoint exactly where the genuine differences are.

        #### 2. Why Can't We Just Run Multiple Standard t-Tests? (The False Positive Trap)
        * In statistics, every standard test carries a small 5% risk of a **false alarm** (finding a difference that doesn't actually exist purely by luck, called a *Type I error* or $\alpha = 0.05$).
        * When you have 3 groups, there are **3 possible pairs** to compare. If you run 3 separate ordinary t-tests without correction, the combined risk of getting at least one false alarm multiplies like buying multiple lottery tickets:
          $$\text{Risk of } \ge 1 \text{ False Alarm} = 1 - (1 - 0.05)^3 = 1 - (0.95)^3 = 1 - 0.857 = 14.3\%$$
        * If you had 5 groups (10 pairwise tests), your false alarm rate would skyrocket to over **40%**!
        * This cumulative risk is called the **Family-Wise Error Rate (FWER)**.

        #### 3. How Tukey's HSD Controls for Multiple Testing
        * **Tukey's Honestly Significant Difference (HSD)** uses the mathematical properties of the *Studentized Range Distribution ($q$)* to establish a wider critical margin.
        * It simultaneously tests all pairs while mathematically guaranteeing that the overall risk of even a single false alarm across the entire set of comparisons remains strictly capped at **5% ($\alpha = 0.05$)**.

        #### 4. How to Read the Numbers in Plain English
        * **Mean Difference ($x̄_B - x̄_A$):** The average score of Group B minus Group A. A positive number means Group B scored higher; a negative number means Group A scored higher.
        * **95% Confidence Interval:** The true real-world difference is 95% likely to fall between these two numbers. If zero ($0.00$) is excluded from this range (e.g. `[+5.20, +14.80]`), you can be confident the groups genuinely differ.
        * **Adjusted p-value:** The probability of observing this gap purely by luck, after protecting against multiple testing. If adjusted $p < 0.05$, we conclude with 95% confidence that the difference is real.
        """)

    st.markdown("---")

    # 3. Interactive Plotly Visualizations in Tabs
    tab_dist, tab_ci, tab_posthoc = st.tabs([
        "📊 Group Distributions & Raw Data",
        "🎯 Group Means with 95% Confidence Intervals",
        "🔍 Post-Hoc Pairwise Comparisons (Tukey HSD)"
    ])

    with tab_dist:
        box_fig = px.box(
            demo_df,
            x="Group",
            y="Score",
            color="Group",
            points="all",
            title="Score Distributions and Individual Data Points Across Groups",
            template="plotly_white",
            color_discrete_sequence=["#2563eb", "#059669", "#7c3aed"]
        )
        # Add group mean markers
        for i, (g_name, g_mean) in enumerate(zip(stats_results["group_names"], stats_results["means"])):
            box_fig.add_trace(go.Scatter(
                x=[g_name],
                y=[g_mean],
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="#b91c1c", line=dict(width=2, color="white")),
                name=f"{g_name} Mean",
                showlegend=False,
                hoverinfo="text",
                hovertext=f"{g_name} Mean: {g_mean:.2f}"
            ))

        box_fig.update_layout(
            xaxis_title="Study Group / Treatment Condition",
            yaxis_title="Continuous Score",
            showlegend=False
        )
        st.plotly_chart(box_fig, use_container_width=True)
        st.caption("💡 **Interpretation:** Red diamonds represent group sample means. Points illustrate individual participant scores.")

    with tab_ci:
        ci_fig = go.Figure()
        grp_names = stats_results["group_names"]
        grp_means = stats_results["means"]
        ci_list = stats_results["ci_groups"]

        err_y = [m - ci[0] for m, ci in zip(grp_means, ci_list)]

        ci_fig.add_trace(go.Scatter(
            x=grp_names,
            y=grp_means,
            error_y=dict(type="data", array=err_y, visible=True, thickness=2.5, width=8, color="#1e3a8a"),
            mode="lines+markers",
            line=dict(color="#93c5fd", width=2, dash="dash"),
            marker=dict(size=10, color="#1e3a8a"),
            name="Group Mean ± 95% CI"
        ))

        # Add horizontal grand mean line
        ci_fig.add_hline(
            y=stats_results["grand_mean"],
            line_dash="dot",
            line_color="#ef4444",
            line_width=2,
            annotation_text=f"Grand Mean = {stats_results['grand_mean']:.2f}",
            annotation_position="bottom right"
        )

        ci_fig.update_layout(
            title="Group Means with 95% Confidence Intervals",
            xaxis_title="Study Group",
            yaxis_title="Mean Score (95% CI)",
            template="plotly_white"
        )
        st.plotly_chart(ci_fig, use_container_width=True)
        st.caption("💡 **Interpretation:** Non-overlapping confidence intervals suggest statistically significant pairwise differences.")

    with tab_posthoc:
        comp_data = stats_results["pairwise_comparisons"]
        post_fig = go.Figure()

        for idx, comp in enumerate(comp_data):
            diff = comp["diff"]
            ci_low = comp["ci_lower"]
            ci_high = comp["ci_upper"]
            is_sig = comp["significant"]
            bar_color = "#10b981" if is_sig else "#94a3b8"

            post_fig.add_trace(go.Scatter(
                x=[diff],
                y=[comp["comparison"]],
                error_x=dict(type="data", array=[ci_high - diff], arrayminus=[diff - ci_low], visible=True, thickness=2.5, width=6, color=bar_color),
                mode="markers",
                marker=dict(size=10, color=bar_color),
                name=comp["comparison"],
                hoverinfo="text",
                hovertext=f"{comp['comparison']}: Diff = {diff:+.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}], Adjusted p = {comp['p_adj']:.4f}"
            ))

        # Add vertical null line at diff = 0
        post_fig.add_vline(
            x=0.0,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2,
            annotation_text="No Difference (Diff = 0)",
            annotation_position="top left"
        )

        post_fig.update_layout(
            title="Post-Hoc Pairwise Mean Differences (Tukey's HSD 95% CIs)",
            xaxis_title="Pairwise Difference Between Means",
            yaxis_title="Group Comparison",
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(post_fig, use_container_width=True)
        st.caption("💡 **Interpretation:** Green intervals that do not cross the vertical red line (0.0) indicate statistically significant pairwise differences at alpha = 0.05.")

    st.markdown("---")

    # 3. Comprehensive Assumption Diagnostics Section
    st.markdown("### 🛡️ Diagnostic Tests for Core Assumptions")
    st.markdown("""
    Below are the diagnostic evaluations verifying whether One-Way ANOVA 
    is appropriate for this dataset:
    """)

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        status_color = "#10b981" if asm["status"] == "pass" else ("#f59e0b" if asm["status"] == "warning" else "#ef4444")
        bg_color = "#f0fdf4" if asm["status"] == "pass" else ("#fffbeb" if asm["status"] == "warning" else "#fef2f2")
        badge_symbol = "✅" if asm["status"] == "pass" else ("🟡" if asm["status"] == "warning" else "❌")

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # Expandable details for alternative tests
    with st.expander("🔎 View Alternative Test Details (Welch's ANOVA & Kruskal-Wallis)", expanded=False):
        st.markdown(f"""
        * **Alternative 1: Welch's ANOVA (Robust to Unequal Variances):**
          * Use when group variances are unequal ($F_{{\max}} = s_{{\max}}^2 / s_{{\min}}^2 > 4.0$ or Levene p < 0.05).
          * Welch's ANOVA weights each group inversely by its sample variance, protecting against inflated false positive rates.

        * **Alternative 2: Kruskal-Wallis H Test (Non-Parametric Multi-Group Test):**
          * Kruskal-Wallis H Statistic: `H = {stats_results['kruskal_stat']:.3f}` | **p-value = `{stats_results['p_kruskal']:.4e}`**
          * Conclusion: {'Statistically significant difference across group distributions (p < 0.05).' if stats_results['p_kruskal'] < 0.05 else 'No statistically significant difference across group distributions (p ≥ 0.05).'}
          * Use when group sample sizes are small ($n_k < 30$) and residuals violate normality or contain extreme outliers.
        """)


def render_kruskal_wallis_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics for Kruskal-Wallis H Test."""
    st.markdown("### 🎯 Hypothesis Test Results: Multi-Group Rank Comparison")

    names = stats_results["group_names"]
    meds = stats_results["medians"]
    iqrs = stats_results["iqrs"]
    m_ranks = stats_results["mean_ranks"]
    r_sums = stats_results["rank_sums"]

    # 1. Metrics Grid with Tooltips
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric(
            f"{names[0]} Median",
            f"{meds[0]:.1f}",
            help=f"Sample size n = {stats_results['n_per_group']}. Q1 = {stats_results['q25s'][0]:.1f}, Q3 = {stats_results['q75s'][0]:.1f}, IQR = {iqrs[0]:.1f}. Mean = {stats_results['means'][0]:.1f}."
        )
        st.metric(
            f"{names[1]} Median",
            f"{meds[1]:.1f}",
            help=f"Sample size n = {stats_results['n_per_group']}. Q1 = {stats_results['q25s'][1]:.1f}, Q3 = {stats_results['q75s'][1]:.1f}, IQR = {iqrs[1]:.1f}. Mean = {stats_results['means'][1]:.1f}."
        )
        st.metric(
            f"{names[2]} Median",
            f"{meds[2]:.1f}",
            help=f"Sample size n = {stats_results['n_per_group']}. Q1 = {stats_results['q25s'][2]:.1f}, Q3 = {stats_results['q75s'][2]:.1f}, IQR = {iqrs[2]:.1f}. Mean = {stats_results['means'][2]:.1f}."
        )

    with mcol2:
        st.metric(
            f"{names[0]} Mean Rank (R̄₁)",
            f"{m_ranks[0]:.1f}",
            help=f"Rank sum R1 = {r_sums[0]:.1f}. Theoretical expected rank under H0 = {stats_results['expected_mean_rank']:.1f}."
        )
        st.metric(
            f"{names[1]} Mean Rank (R̄₂)",
            f"{m_ranks[1]:.1f}",
            help=f"Rank sum R2 = {r_sums[1]:.1f}. Theoretical expected rank under H0 = {stats_results['expected_mean_rank']:.1f}."
        )
        st.metric(
            f"{names[2]} Mean Rank (R̄₃)",
            f"{m_ranks[2]:.1f}",
            help=f"Rank sum R3 = {r_sums[2]:.1f}. Theoretical expected rank under H0 = {stats_results['expected_mean_rank']:.1f}."
        )

    with mcol3:
        st.metric(
            "Kruskal-Wallis Statistic",
            f"H({stats_results['df_kruskal']}) = {stats_results['h_stat']:.3f}",
            help=f"H-statistic with df = k - 1 = {stats_results['df_kruskal']}. Evaluates variance of group mean ranks around expected rank."
        )
        sig_text = "p < 0.05 (Significant)" if stats_results["p_kruskal"] < 0.05 else "p ≥ 0.05 (Not Significant)"
        st.metric(
            "p-value",
            f"{stats_results['p_kruskal']:.4f}",
            delta=sig_text,
            delta_color="normal" if stats_results["p_kruskal"] < 0.05 else "off",
            help=(
                "The p-value evaluates whether the observed group rank distributions differ more than expected by random chance.\n\n"
                "• p < 0.05 (Significant): Strong evidence that population distributions / medians differ across groups (rejects H0).\n\n"
                "• p ≥ 0.05 (Not Significant): Insufficient evidence to conclude group distributions differ."
            )
        )

    with mcol4:
        st.metric(
            "Epsilon-Squared (ε_R²)",
            f"{stats_results['epsilon_squared']:.3f}",
            help=f"Rank-based effect size: {stats_results['effect_magnitude']}. Scale: <0.06 = Small, 0.06-0.14 = Medium, >=0.14 = Large."
        )
        st.metric(
            "Tie Factor (C)",
            f"{stats_results['tie_factor']:.4f}",
            help=f"Tie correction factor: {stats_results['tie_count_total']} tied observations detected across sample."
        )

    st.markdown("---")

    # 2. Dedicated Post-Hoc Pairwise Comparisons Breakdown (Dunn's Test with Bonferroni Correction)
    st.markdown("### 🔍 Which Groups Are Actually Different? (Post-Hoc Pairwise Comparisons)")

    comp_data = stats_results["pairwise_comparisons"]
    sig_count = sum(1 for c in comp_data if c["significant"])

    if stats_results["p_kruskal"] < 0.05:
        st.markdown(f"""
        The omnibus Kruskal-Wallis test confirms that group rank distributions differ overall ($p < 0.05$). 
        Below, **Dunn's post-hoc test with Bonferroni multiple testing correction** examines each pair of groups 
        to identify **exactly which groups differ from each other**, while strictly protecting against false alarms.
        """)
    else:
        st.info("""
        ℹ️ **Omnibus test not significant ($p \ge 0.05$):** Under standard scientific testing procedures (Fisher's protection principle), 
        post-hoc tests are not statistically indicated because any observed rank differences between pairs are consistent with random sampling variability. 
        The pairwise comparisons are shown below for exploratory inspection only.
        """)

    posthoc_rows = []
    for c in comp_data:
        posthoc_rows.append({
            "Pairwise_Comparison": c["comparison"],
            "Rank_Difference": c["diff_rank_str"],
            "Median_Difference": c["diff_median_str"],
            "95%_CI_Rank_Diff": c["ci_rank_str"],
            "Adjusted_p_value": c["p_adj_str"],
            "Significance_Verdict": c["significance_verdict"],
            "Plain_English_Finding": c["plain_english"]
        })
    posthoc_df = pd.DataFrame(posthoc_rows)
    render_table(posthoc_df, height=180)

    with st.expander("📘 Plain English Guide: What is Dunn's Post-Hoc Test & How is Multiple Testing Controlled?", expanded=False):
        st.markdown(r"""
        #### 1. What is Dunn's Post-Hoc Test?
        * The **Kruskal-Wallis omnibus test** tells you that at least one group tends to have higher or lower scores than the others, but it **cannot identify which specific pairs differ**.
        * **Dunn's post-hoc test (1964)** is the non-parametric gold standard follow-up test. It compares the difference between average ranks ($\bar{R}_j - \bar{R}_i$) relative to the pooled ranking variance.

        #### 2. Why Can't We Just Run Multiple Mann-Whitney Tests? (The False Positive Trap)
        * In statistics, testing each pair at the standard 5% significance level ($\alpha = 0.05$) creates a risk of false alarms.
        * With 3 groups (3 pairwise comparisons), running unadjusted tests raises the combined risk of at least one false alarm to:
          $$\text{Risk of } \ge 1 \text{ False Alarm} = 1 - (1 - 0.05)^3 = 14.3\%$$
        * Just like rolling a die multiple times increases your chance of rolling a 6, testing multiple pairs without correction dramatically inflates false positives.

        #### 3. How the Bonferroni Correction Controls for Multiple Testing
        * The **Bonferroni correction** is a mathematically rigorous shield against false positives.
        * When conducting $c = 3$ comparisons, the method adjusts each raw p-value by multiplying it by 3:
          $$p_{\text{adjusted}} = \min(1.0, p_{\text{raw}} \times 3)$$
        * Equivalently, a comparison is only declared statistically significant if its raw p-value is below:
          $$\alpha_{\text{adjusted}} = \frac{0.05}{3} \approx 0.0167$$
        * This strictly guarantees that the overall family-wise false positive rate across all 3 comparisons remains capped at **5% ($\alpha = 0.05$)**.

        #### 4. How to Read the Numbers in Plain English
        * **Rank Difference ($\bar{R}_B - \bar{R}_A$):** The average rank of Group B minus Group A. A positive number means Group B consistently scored higher across the study.
        * **Median Difference ($\tilde{x}_B - \tilde{x}_A$):** The typical (median) score gap between the two groups.
        * **95% Confidence Interval:** The true rank difference is 95% likely to fall within this range. If zero is excluded, the difference is statistically significant.
        * **Adjusted p-value:** The probability of seeing this gap purely by chance after protecting against multiple testing. If adjusted $p < 0.05$, we conclude with 95% confidence that the groups truly differ.
        """)

    st.markdown("---")

    # 3. Interactive Visualization Tabs
    st.markdown("### 📈 Visual Comparison & Rank Distributions")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📦 Group Distributions & Medians",
        "🏷️ Group Mean Ranks vs. Expected Rank",
        "🎯 Pairwise Dunn Post-Hoc Comparisons"
    ])

    with viz_tab1:
        box_fig = px.box(
            demo_df,
            x="Group",
            y="Score",
            color="Group",
            points="all",
            title="Side-by-Side Boxplots with Jittered Observations",
            color_discrete_sequence=["#3b82f6", "#10b981", "#8b5cf6"]
        )
        box_fig.update_layout(template="plotly_white")
        st.plotly_chart(box_fig, use_container_width=True)
        st.caption(
            "💡 **How to interpret boxplots:** "
            "The horizontal center line inside each box marks the group median. "
            "The box boundaries show the 25th (Q1) and 75th (Q3) percentiles (IQR). "
            "A **p-value < 0.05** indicates that group distributions differ significantly."
        )

    with viz_tab2:
        exp_rank = stats_results["expected_mean_rank"]
        rank_fig = go.Figure()
        rank_fig.add_trace(go.Bar(
            x=names,
            y=m_ranks,
            marker_color=["#3b82f6", "#10b981", "#8b5cf6"],
            text=[f"R̄ = {r:.1f}" for r in m_ranks],
            textposition="auto",
            name="Observed Mean Rank"
        ))
        rank_fig.add_hline(
            y=exp_rank,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2.5,
            annotation_text=f"Expected Rank under H₀ = {exp_rank:.1f}",
            annotation_position="top left",
            annotation_font=dict(color="#ef4444", size=12)
        )
        rank_fig.update_layout(
            title="Observed Group Mean Ranks (R̄_k) vs. Theoretical Expected Rank under H₀",
            xaxis_title="Group",
            yaxis_title="Average Rank",
            template="plotly_white"
        )
        st.plotly_chart(rank_fig, use_container_width=True)
        st.caption(
            "💡 **Rank Test Principle:** Under the null hypothesis H₀, all groups share identical distributions, "
            f"so every group's average rank is expected to equal (N + 1) / 2 = {exp_rank:.1f} (dashed red line). "
            "Kruskal-Wallis tests whether group deviations from this expected rank are larger than random sampling variation."
        )

    with viz_tab3:
        pairs = stats_results["pairwise_comparisons"]
        forest_fig = go.Figure()

        y_labels = [p["comparison"] for p in pairs]
        diffs = [p["diff_mean_rank"] for p in pairs]
        ci_lows = [p["ci_lower_rank"] for p in pairs]
        ci_highs = [p["ci_upper_rank"] for p in pairs]
        sigs = [p["significant"] for p in pairs]
        colors = ["#10b981" if s else "#94a3b8" for s in sigs]

        forest_fig.add_trace(go.Scatter(
            x=diffs,
            y=y_labels,
            mode="markers",
            error_x=dict(
                type="data",
                symmetric=False,
                array=[ci_highs[idx] - diffs[idx] for idx in range(len(diffs))],
                arrayminus=[diffs[idx] - ci_lows[idx] for idx in range(len(diffs))],
                color="#334155",
                thickness=2,
                width=6
            ),
            marker=dict(color=colors, size=11),
            text=[f"z = {p['z_stat']:.2f}, p_adj = {p['p_adj']:.4f} ({'Significant' if p['significant'] else 'Not sig'})" for p in pairs],
            hoverinfo="text",
            name="Pairwise Difference"
        ))

        forest_fig.add_vline(
            x=0.0,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2,
            annotation_text="Zero Difference (H₀)",
            annotation_position="top right"
        )

        forest_fig.update_layout(
            title="Dunn's Post-Hoc Pairwise Comparisons (95% Bonferroni Confidence Intervals)",
            xaxis_title="Difference in Mean Ranks (R̄_j - R̄_i)",
            yaxis_title="Pairwise Contrast",
            template="plotly_white"
        )
        st.plotly_chart(forest_fig, use_container_width=True)
        st.caption(
            "💡 **Interpreting Dunn's Post-Hoc Test:** "
            "Green markers denote statistically significant contrasts (p_adj < 0.05). "
            "If a contrast's confidence interval excludes zero (dashed red line), the difference between those two groups is statistically significant after Bonferroni correction."
        )

    st.markdown("---")

    # 3. Comprehensive Assumption Diagnostics Section
    st.markdown("### 🛡️ Diagnostic Tests for Core Assumptions")
    st.markdown("""
    Below are the diagnostic evaluations verifying whether the Kruskal-Wallis H Test 
    is appropriate for this dataset:
    """)

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        status_color = "#10b981" if asm["status"] == "pass" else ("#f59e0b" if asm["status"] == "warning" else "#ef4444")
        bg_color = "#f0fdf4" if asm["status"] == "pass" else ("#fffbeb" if asm["status"] == "warning" else "#fef2f2")
        badge_symbol = "✅" if asm["status"] == "pass" else ("🟡" if asm["status"] == "warning" else "❌")

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 4. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # 5. Expandable details for alternative tests
    with st.expander("🔎 View Alternative Test Details (One-Way ANOVA & Welch's ANOVA)", expanded=False):
        st.markdown(f"""
        * **Alternative 1: One-Way ANOVA (Parametric Gold Standard for Normal Data):**
          * Parametric omnibus F-statistic: `F = {stats_results['f_stat']:.3f}` | **p-value = `{stats_results['p_anova']:.4e}`**
          * Use when dependent data is continuous, normally distributed across groups, and variances are homogeneous ($F_{{\max}} \le 4.0$).
          * ANOVA offers approximately 5% greater statistical power than Kruskal-Wallis when normality holds.

        * **Alternative 2: Welch's ANOVA (Robust to Heteroscedasticity):**
          * Levene's Test of Equal Variances: `statistic = {stats_results['levene_stat']:.3f}`, **p = {stats_results['p_levene']:.4f}**
          * Use when data is continuous and approximately normal, but group variances differ substantially ($F_{{\max}} > 4.0$ or Levene p < 0.05).
        """)


def render_repeated_measures_anova_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visuals, and post-hoc pairwise comparisons for Repeated Measures ANOVA."""
    st.markdown("### 🎯 Hypothesis Test Results: Repeated Measures ANOVA")

    # 1. High-level Summary Metrics (5 Cards)
    col1, col2, col3, col4, col5 = st.columns(5)
    f_stat = stats_results["f_stat"]
    df_t = stats_results["df_time"]
    df_e = stats_results["df_error"]
    col1.metric("Omnibus F-Stat", f"{f_stat:.2f}", f"df=({df_t}, {df_e})")

    p_anova = stats_results["p_anova"]
    p_str = f"{p_anova:.4f}" if p_anova >= 0.0001 else "< 0.0001"
    col2.metric("Omnibus p-value", p_str, "Sig (p < 0.05)" if p_anova < 0.05 else "Not Sig")

    p_gg = stats_results["p_gg"]
    p_gg_str = f"{p_gg:.4f}" if p_gg >= 0.0001 else "< 0.0001"
    eps_gg = stats_results["eps_gg"]
    col3.metric("GG-Adjusted p", p_gg_str, f"ε = {eps_gg:.2f}")

    w_mauchly = stats_results["w_mauchly"]
    p_mauchly = stats_results["p_mauchly"]
    col4.metric("Mauchly Sphericity W", f"{w_mauchly:.3f}", f"p = {p_mauchly:.3f}")

    eta_p2 = stats_results["eta_p2"]
    bench = "Large" if eta_p2 >= 0.14 else ("Medium" if eta_p2 >= 0.06 else "Small")
    col5.metric("Partial Eta² (ηp²)", f"{eta_p2:.3f}", f"{bench} effect")

    # 2. Prominently Featured Post-Hoc Pairwise Comparisons Table
    st.markdown("##### 🔍 Post-Hoc Pairwise Comparisons (Bonferroni-Adjusted Paired t-tests)")
    st.markdown(
        "The omnibus Repeated Measures ANOVA test above tells us *whether* mean scores change significantly over time, "
        "but it does **not** identify *between which specific pairs of time points* the change occurred. "
        "The table below isolates each pairwise comparison, calculating the exact mean change and 95% Bonferroni-adjusted "
        "confidence interval to strictly control the risk of false positives."
    )

    posthoc_df = pd.DataFrame({
        "Comparison": [p["comparison"] for p in stats_results["pairwise_comparisons"]],
        "Mean_Difference": [p["diff_str"] for p in stats_results["pairwise_comparisons"]],
        "Adjusted_95_CI": [p["ci_str"] for p in stats_results["pairwise_comparisons"]],
        "Adjusted_p_value": [p["p_adj_str"] for p in stats_results["pairwise_comparisons"]],
        "Verdict": [p["significance_verdict"] for p in stats_results["pairwise_comparisons"]],
        "Plain_English_Interpretation": [p["plain_english"] for p in stats_results["pairwise_comparisons"]]
    })
    render_table(posthoc_df, height=180)

    with st.expander("💡 Plain-English Guide: Why Post-Hoc Tests & False Positive Control (FWER) Matter", expanded=False):
        st.markdown("""
        **What are post-hoc tests?**
        * "Post-hoc" is Latin for "after this". When Repeated Measures ANOVA shows a statistically significant overall result, post-hoc tests are pairwise comparisons conducted *after* the omnibus test to pin down exactly *where* the significant changes occurred (e.g. did subjects improve between Baseline and Month 1, or did the improvement only emerge by Month 3?).

        **Why must we control for multiple tests (Family-Wise Error Rate)?**
        * When evaluating a single comparison at the standard threshold $\\alpha = 0.05$, there is a 5% chance of getting a "false positive" (crying wolf when no real effect exists).
        * If you test 3 pairs independently without correction, the compound probability of getting **at least one false positive** across the family of tests jumps to:
          $$1 - (1 - 0.05)^3 = 1 - 0.95^3 = 1 - 0.857 = 14.3\\%$$
        * The **Bonferroni correction** multiplies each raw $p$-value by the total number of comparisons ($k = 3$). This guarantees that your cumulative false alarm rate across the entire study stays strictly capped at 5%.

        **How to interpret the table above:**
        * **Adjusted 95% Confidence Interval:** Gives the plausible range of the true average change. If the interval excludes $0.00$, the change is statistically significant.
        * **Adjusted p-value $< 0.05$:** Strong evidence of a real within-subject change between those two time points.
        * **Adjusted p-value $\\ge 0.05$:** The observed change is indistinguishable from random sampling noise.
        """)

    # 3. Interactive Plotly Visualizations (3 Tabs)
    st.markdown("##### 📈 Visual Inspection & Diagnostic Trajectories")
    tab_traj, tab_means, tab_forest = st.tabs([
        "📈 Subject Trajectories (Spaghetti Plot)",
        "📊 Condition Means & 95% CIs",
        "🌲 Post-Hoc Pairwise Differences Forest Plot"
    ])

    time_names = stats_results["time_names"]
    Y_wide = stats_results["Y_wide"]
    n_subj = stats_results["n_subjects"]

    with tab_traj:
        fig_traj = go.Figure()
        # Draw each participant's line as semi-transparent gray
        for i in range(n_subj):
            fig_traj.add_trace(go.Scatter(
                x=time_names,
                y=Y_wide[i, :],
                mode="lines+markers",
                line=dict(color="rgba(148, 163, 184, 0.4)", width=1.5),
                marker=dict(size=4, color="rgba(148, 163, 184, 0.6)"),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"Subject {i+1:02d}"
            ))
        # Overlay bold Group Mean line with error bars
        mean_cis = [1.96 * s / np.sqrt(n_subj) for s in stats_results["stds"]]
        fig_traj.add_trace(go.Scatter(
            x=time_names,
            y=stats_results["means"],
            mode="lines+markers",
            name="Group Mean",
            line=dict(color="#2563eb", width=4),
            marker=dict(size=10, color="#1e40af", symbol="diamond"),
            error_y=dict(
                type="data",
                array=mean_cis,
                visible=True,
                color="#1e40af",
                thickness=2,
                width=6
            )
        ))
        fig_traj.update_layout(
            title="Individual Participant Trajectories (Spaghetti Plot) with Group Mean",
            xaxis_title="Repeated Occasion / Condition",
            yaxis_title="Score",
            template="plotly_white",
            hovermode="closest"
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with tab_means:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=time_names,
            y=stats_results["means"],
            error_y=dict(type="data", array=mean_cis, visible=True),
            marker=dict(color=["#3b82f6", "#10b981", "#8b5cf6"]),
            name="Condition Means"
        ))
        fig_bar.update_layout(
            title="Condition Means with 95% Confidence Intervals",
            xaxis_title="Condition",
            yaxis_title="Mean Score",
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_forest:
        fig_forest = go.Figure()
        comps = [p["comparison"] for p in stats_results["pairwise_comparisons"]]
        diff_means = [p["diff_mean"] for p in stats_results["pairwise_comparisons"]]
        ci_lows = [p["ci_lower"] for p in stats_results["pairwise_comparisons"]]
        ci_highs = [p["ci_upper"] for p in stats_results["pairwise_comparisons"]]
        err_plus = [ci_highs[i] - diff_means[i] for i in range(len(diff_means))]
        err_minus = [diff_means[i] - ci_lows[i] for i in range(len(diff_means))]
        colors = ["#10b981" if p["significant"] else "#64748b" for p in stats_results["pairwise_comparisons"]]

        fig_forest.add_trace(go.Scatter(
            x=diff_means,
            y=comps,
            mode="markers",
            marker=dict(size=12, color=colors, symbol="square"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=err_plus,
                arrayminus=err_minus,
                color="#334155",
                thickness=2,
                width=8
            ),
            name="Mean Difference"
        ))
        fig_forest.add_vline(x=0.0, line_dash="dash", line_color="#ef4444", annotation_text="No Difference (0.0)")
        fig_forest.update_layout(
            title="Post-Hoc Pairwise Differences (95% Bonferroni CIs)",
            xaxis_title="Mean Difference (Change)",
            yaxis_title="Pairwise Comparison",
            template="plotly_white"
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # 4. Core Assumption Evaluation Cards
    st.markdown("### 🛡️ Core Methodological Assumption Diagnostics")
    st.caption("Repeated Measures ANOVA requires within-subject matching, normality of differences, sphericity, and absence of extreme outliers.")

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        if asm["status"] == "pass":
            status_color = "#10b981"
            bg_color = "#f0fdf4"
            badge_symbol = "✅"
        elif asm["status"] == "warning":
            status_color = "#f59e0b"
            bg_color = "#fffbeb"
            badge_symbol = "🟡"
        else:
            status_color = "#ef4444"
            bg_color = "#fef2f2"
            badge_symbol = "❌"

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 5. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # 6. Expandable Details for Alternative Tests
    with st.expander("🔎 View Alternative Test Details (Greenhouse-Geisser Correction & Friedman Test)", expanded=False):
        st.markdown(f"""
        * **Alternative 1: Greenhouse-Geisser Sphericity Adjustment:**
          * Mauchly's Sphericity: `W = {stats_results['w_mauchly']:.4f}`, **p = `{stats_results['p_mauchly']:.4f}`**
          * Greenhouse-Geisser Epsilon: `ε = {stats_results['eps_gg']:.4f}`
          * Adjusted degrees of freedom: `df1 = {stats_results['df_time_gg']:.2f}`, `df2 = {stats_results['df_error_gg']:.2f}`
          * Adjusted omnibus p-value: **`{stats_results['p_gg']:.4e}`**
          * Use when difference variances are unequal (Mauchly p < 0.05 or ε < 0.75) to prevent inflated false positives.

        * **Alternative 2: Friedman Test (Rank-Based Non-Parametric Alternative):**
          * Friedman Chi-Square test statistic: `χ²(2) = {stats_results['friedman_stat']:.3f}` | **p-value = `{stats_results['p_friedman']:.4e}`**
          * Use when pairwise differences violate normality or data is ordinal (e.g. Likert ratings 1-7).
        """)


def render_friedman_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visuals, and post-hoc pairwise comparisons for the Friedman Test."""
    st.markdown("### 🎯 Hypothesis Test Results: Friedman Test")

    # 1. High-level Summary Metrics (5 Cards)
    col1, col2, col3, col4, col5 = st.columns(5)
    q_stat = stats_results["q_stat"]
    df_f = stats_results["df_friedman"]
    col1.metric("Friedman Chi² (Q)", f"{q_stat:.2f}", f"df={df_f}")

    p_val = stats_results["p_friedman"]
    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
    col2.metric("Omnibus p-value", p_str, "Sig (p < 0.05)" if p_val < 0.05 else "Not Sig")

    kendalls_w = stats_results["kendalls_w"]
    col3.metric("Kendall's W (Effect Size)", f"{kendalls_w:.3f}", stats_results["effect_magnitude"])

    # Mean Ranks spread
    mean_ranks = stats_results["mean_ranks"]
    rank_str = " / ".join(f"{r:.2f}" for r in mean_ranks)
    col4.metric("Mean Ranks (1 / 2 / 3)", rank_str, "Baseline = 2.00")

    p_param = stats_results["p_anova_param"]
    p_param_str = f"{p_param:.4f}" if p_param >= 0.0001 else "< 0.0001"
    col5.metric("Parametric RM-ANOVA p", p_param_str, f"F = {stats_results['f_stat_param']:.2f}")

    # 2. Prominently Featured Post-Hoc Pairwise Comparisons Table
    st.markdown("##### 🔍 Post-Hoc Pairwise Comparisons (Wilcoxon Signed-Rank with Bonferroni Correction)")
    st.markdown(
        "The omnibus Friedman test above tells us *whether* within-subject ranks differ significantly across the 3 conditions, "
        "but it does **not** identify *between which specific pairs of conditions* the difference occurred. "
        "The table below isolates each pairwise comparison using Wilcoxon signed-rank tests with a strict Bonferroni adjustment "
        "to prevent false discoveries."
    )

    posthoc_df = pd.DataFrame({
        "Comparison": [p["comparison"] for p in stats_results["pairwise_comparisons"]],
        "Mean_Rank_Difference": [p["diff_rank_str"] for p in stats_results["pairwise_comparisons"]],
        "Median_Difference": [p["diff_median_str"] for p in stats_results["pairwise_comparisons"]],
        "Wilcoxon_W": [f"{p['w_stat']:.1f}" for p in stats_results["pairwise_comparisons"]],
        "Adjusted_p_value": [p["p_adj_str"] for p in stats_results["pairwise_comparisons"]],
        "Verdict": [p["significance_verdict"] for p in stats_results["pairwise_comparisons"]],
        "Plain_English_Interpretation": [p["plain_english"] for p in stats_results["pairwise_comparisons"]]
    })
    render_table(posthoc_df, height=180)

    with st.expander("💡 Plain-English Guide: Why Post-Hoc Tests & False Positive Control (FWER) Matter", expanded=False):
        st.markdown("""
        **What are post-hoc tests?**
        * "Post-hoc" is Latin for "after this". When the Friedman test shows that conditions overall are not equal, post-hoc tests are pairwise comparisons conducted *after* the omnibus test to pin down exactly *where* the significant changes occurred.

        **Why must we control for multiple tests (Family-Wise Error Rate)?**
        * When testing a single comparison at $\\alpha = 0.05$, there is a 5% chance of getting a "false positive" (falsely claiming a difference exists).
        * If you test 3 pairs independently without correction, the compound probability of getting **at least one false positive** across the family of tests jumps to:
          $$1 - (1 - 0.05)^3 = 1 - 0.95^3 = 1 - 0.857 = 14.3\\%$$
        * The **Bonferroni correction** multiplies each raw $p$-value by the total number of comparisons ($k = 3$). This guarantees that your cumulative false alarm rate across the entire study stays strictly capped at 5%.

        **How to interpret the table above:**
        * **Mean Rank Difference:** Quantifies how much higher or lower one condition consistently placed relative to the other across participants.
        * **Adjusted p-value $< 0.05$:** Strong evidence of a real within-subject difference between those two specific conditions.
        * **Adjusted p-value $\\ge 0.05$:** The observed rank difference is consistent with random chance.
        """)

    # 3. Interactive Plotly Visualizations (3 Tabs)
    st.markdown("##### 📈 Visual Inspection & Diagnostic Trajectories")
    tab_traj, tab_ranks, tab_forest = st.tabs([
        "📈 Subject Rank & Score Trajectories",
        "📊 Condition Mean Ranks vs Baseline",
        "🌲 Post-Hoc Pairwise Differences Forest Plot"
    ])

    cond_names = stats_results["condition_names"]
    Y_wide = stats_results["Y_wide"]
    R_wide = stats_results["R_wide"]
    n_subj = stats_results["n_subjects"]

    with tab_traj:
        fig_traj = go.Figure()
        for i in range(n_subj):
            fig_traj.add_trace(go.Scatter(
                x=cond_names,
                y=Y_wide[i, :],
                mode="lines+markers",
                line=dict(color="rgba(148, 163, 184, 0.4)", width=1.5),
                marker=dict(size=5, color="rgba(148, 163, 184, 0.6)"),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"Subject {i+1:02d} (Ranks: {R_wide[i, 0]:.1f}, {R_wide[i, 1]:.1f}, {R_wide[i, 2]:.1f})"
            ))
        # Overlay condition medians line
        fig_traj.add_trace(go.Scatter(
            x=cond_names,
            y=stats_results["medians"],
            mode="lines+markers",
            name="Condition Median",
            line=dict(color="#2563eb", width=4),
            marker=dict(size=12, color="#1e40af", symbol="diamond")
        ))
        fig_traj.update_layout(
            title="Participant Trajectories Across Conditions with Medians",
            xaxis_title="Repeated Condition",
            yaxis_title="Score / Rating",
            template="plotly_white",
            hovermode="closest"
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with tab_ranks:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=cond_names,
            y=stats_results["mean_ranks"],
            marker=dict(color=["#3b82f6", "#10b981", "#8b5cf6"]),
            name="Mean Rank"
        ))
        fig_bar.add_hline(y=2.0, line_dash="dash", line_color="#ef4444", annotation_text="Null Expected Rank (2.00)")
        fig_bar.update_layout(
            title="Within-Subject Mean Ranks by Condition (Expected Null = 2.00)",
            xaxis_title="Condition",
            yaxis_title="Average Rank",
            yaxis=dict(range=[0, 3.2]),
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_forest:
        fig_forest = go.Figure()
        comps = [p["comparison"] for p in stats_results["pairwise_comparisons"]]
        rank_diffs = [p["diff_mean_rank"] for p in stats_results["pairwise_comparisons"]]
        colors = ["#10b981" if p["significant"] else "#64748b" for p in stats_results["pairwise_comparisons"]]
        se_r = np.sqrt(2.0 / n_subj) if n_subj > 0 else 1.0

        fig_forest.add_trace(go.Scatter(
            x=rank_diffs,
            y=comps,
            mode="markers",
            marker=dict(size=12, color=colors, symbol="square"),
            error_x=dict(
                type="data",
                array=[1.96 * se_r] * len(rank_diffs),
                color="#334155",
                thickness=2,
                width=8
            ),
            name="Mean Rank Diff"
        ))
        fig_forest.add_vline(x=0.0, line_dash="dash", line_color="#ef4444", annotation_text="No Difference (0.0)")
        fig_forest.update_layout(
            title="Post-Hoc Pairwise Mean Rank Differences (95% CI Bands)",
            xaxis_title="Mean Rank Difference",
            yaxis_title="Pairwise Comparison",
            template="plotly_white"
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # 4. Core Assumption Evaluation Cards
    st.markdown("### 🛡️ Core Methodological Assumption Diagnostics")
    st.caption("The Friedman Test requires a within-subject matched block design, mutual independence of participants, and adequate sample size ($N \\ge 10$).")

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        if asm["status"] == "pass":
            status_color = "#10b981"
            bg_color = "#f0fdf4"
            badge_symbol = "✅"
        elif asm["status"] == "warning":
            status_color = "#f59e0b"
            bg_color = "#fffbeb"
            badge_symbol = "🟡"
        else:
            status_color = "#ef4444"
            bg_color = "#fef2f2"
            badge_symbol = "❌"

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 5. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # 6. Expandable Details for Alternative Tests
    with st.expander("🔎 View Alternative Test Details (Repeated Measures ANOVA)", expanded=False):
        st.markdown(f"""
        * **Alternative: Parametric Repeated Measures ANOVA:**
          * Parametric omnibus F-statistic: `F = {stats_results['f_stat_param']:.3f}` | **p-value = `{stats_results['p_anova_param']:.4e}`**
          * Use when data is continuous, normally distributed across difference scores, and satisfies the sphericity assumption.
          * Repeated Measures ANOVA provides greater statistical power than the Friedman test when parametric assumptions are met.
        """)


def render_chi_square_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visualizations, and diagnostics for Pearson's Chi-Square Test."""
    st.markdown("### 🎯 Hypothesis Test Results: Pearson's Chi-Square Test of Independence")

    # 1. Metrics Grid
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    mcol1.metric(
        "Grand Sample Size (N)",
        f"{stats_results['n_total']}",
        help="Total number of observations cross-classified across all cells."
    )
    mcol1.metric(
        "Degrees of Freedom (df)",
        f"{stats_results['degrees_of_freedom']}",
        help="df = (Rows - 1) × (Cols - 1) = (2 - 1) × (2 - 1) = 1."
    )

    mcol2.metric(
        "Pearson Chi-Square (χ²)",
        f"{stats_results['chi2_uncorr']:.3f}",
        help="Standard asymptotic Pearson test statistic summing standardized squared cell deviations."
    )
    mcol2.metric(
        "Yates' Corrected χ²",
        f"{stats_results['chi2_yates']:.3f}",
        help="Yates' continuity-corrected statistic subtracting 0.5 from discrepancies (conservative discrete adjustment)."
    )

    sig_delta = "p < 0.05 (Significant)" if stats_results["p_uncorr"] < 0.05 else "p ≥ 0.05 (Not Significant)"
    mcol3.metric(
        "Pearson p-value",
        f"{stats_results['p_uncorr']:.4f}",
        delta=sig_delta,
        delta_color="normal" if stats_results["p_uncorr"] < 0.05 else "off",
        help="Probability of observing this degree of discrepancy if row and column factors were completely independent."
    )
    mcol3.metric(
        "Fisher's Exact p-value",
        f"{stats_results['p_fisher']:.4f}",
        help="Hypergeometric exact p-value. Preferred gold standard whenever expected cell frequencies are small (< 5.0)."
    )

    min_exp = stats_results["min_expected_count"]
    exp_delta = "≥ 5.0 (Cochran Satisfied)" if min_exp >= 5.0 else "< 5.0 (Cochran Violated)"
    mcol4.metric(
        "Min Expected Cell (E_min)",
        f"{min_exp:.2f}",
        delta=exp_delta,
        delta_color="normal" if min_exp >= 5.0 else "inverse",
        help="Smallest expected cell frequency under independence. Cochran's rule requires all 4 cells to be ≥ 5.0."
    )
    or_val = stats_results["odds_ratio"]
    or_ci = stats_results["odds_ratio_ci"]
    mcol4.metric(
        "Odds Ratio (OR)",
        f"{or_val:.2f}",
        help=f"Odds of positive outcome in {stats_results['group1_name']} vs {stats_results['group2_name']}. 95% CI: [{or_ci[0]:.2f}, {or_ci[1]:.2f}]."
    )

    # 2. Clinical & Practical Decision Interpretation in Plain English
    st.markdown("##### 🩺 Clinical & Statistical Decision Interpretation")
    g1 = stats_results["group1_name"]
    g2 = stats_results["group2_name"]
    out_pos = stats_results["outcome_pos_name"]
    p1_pct = stats_results["risk_group1"] * 100.0
    p2_pct = stats_results["risk_group2"] * 100.0
    rd_pct = stats_results["risk_diff"] * 100.0
    rr_val = stats_results["risk_ratio"]
    is_sig = stats_results["p_uncorr"] < 0.05
    min_exp_ok = min_exp >= 5.0

    if is_sig and min_exp_ok:
        st.success(f"""
        **Primary Finding: Statistically Significant Association (χ²(1) = {stats_results['chi2_uncorr']:.3f}, p = {stats_results['p_uncorr']:.4f} < 0.05)**
        * **Hypotheses Tested:** 
          * *Null Hypothesis ($H_0$):* There is no association between **{g1} vs. {g2}** and the occurrence of **{out_pos}** (proportions are identical).
          * *Alternative Hypothesis ($H_1$):* There is a genuine statistical association between groups and outcome occurrence.
        * **Observed Rates:** Individuals in **{g1}** had an event rate of **{p1_pct:.1f}%** compared to **{p2_pct:.1f}%** in **{g2}** (Absolute Risk Difference: **{rd_pct:+.1f}%**).
        * **Effect Sizes:** 
          * **Odds Ratio (OR):** `{or_val:.2f}` (95% CI: `{or_ci[0]:.2f}` – `{or_ci[1]:.2f}`) — The odds of experiencing {out_pos} are {or_val:.2f} times as high in {g1} compared to {g2}.
          * **Relative Risk (RR):** `{rr_val:.2f}` (95% CI: `{stats_results['risk_ratio_ci'][0]:.2f}` – `{stats_results['risk_ratio_ci'][1]:.2f}`).
          * **Strength of Association:** Cramér's $V = {stats_results['cramers_v']:.3f}$ ({stats_results['effect_magnitude']}).
        * **Methodological Validity:** Because all expected cell counts are $\\ge 5.0$ (minimum: `{min_exp:.2f}`), Cochran's criterion is fully met and reporting the standard Pearson Chi-Square test is mathematically valid.
        """)
    elif not min_exp_ok:
        st.warning(f"""
        **Methodological Warning: Small Expected Cell Frequency Detected (Min Expected = {min_exp:.2f} < 5.0)**
        * **Cochran's Criterion Violation:** Standard Pearson Chi-Square relies on large-sample asymptotic approximations that break down when expected frequencies drop below 5.0, potentially distorting $p$-values.
        * **Gold-Standard Recommendation:** **Report Fisher's Exact Test** instead of Pearson Chi-Square.
        * **Fisher's Exact Test Results:** Two-sided exact **$p = {stats_results['p_fisher']:.4f}$** | Exact Odds Ratio = `{stats_results['odds_ratio_fisher']:.2f}`.
        * **Conclusion:** {'Statistically significant association confirmed via exact hypergeometric probability (p < 0.05).' if stats_results['p_fisher'] < 0.05 else 'No statistically significant association detected under exact testing (p ≥ 0.05).'}
        """)
    else:
        st.info(f"""
        **Primary Finding: No Statistically Significant Association (χ²(1) = {stats_results['chi2_uncorr']:.3f}, p = {stats_results['p_uncorr']:.4f} ≥ 0.05)**
        * **Hypotheses Tested:** 
          * *Null Hypothesis ($H_0$):* Group membership and outcome occurrence are independent.
          * *Alternative Hypothesis ($H_1$):* An association exists between groups and outcome occurrence.
        * **Observed Rates:** Individuals in **{g1}** had an event rate of **{p1_pct:.1f}%** vs. **{p2_pct:.1f}%** in **{g2}** (Absolute Difference: **{rd_pct:+.1f}%**).
        * **Odds Ratio (OR):** `{or_val:.2f}` (95% CI: `{or_ci[0]:.2f}` – `{or_ci[1]:.2f}`). Notice that the 95% confidence interval spans `1.0` (the neutral null value).
        * **Conclusion:** We fail to reject the null hypothesis. The observed difference in rates is entirely consistent with random chance sampling variation.
        """)

    # 3. Interactive Plotly Visualizations (3 Tabs)
    st.markdown("##### 📈 Visual Inspection & Contingency Diagnostics")
    tab_bar, tab_heat, tab_forest = st.tabs([
        "📊 Observed vs Expected Counts",
        "🔥 Contingency Heatmap & Haberman Residuals",
        "🌲 Effect Sizes Forest Plot (OR, RR, RD)"
    ])

    with tab_bar:
        # Grouped bar chart comparing Observed vs Expected
        fig_bar = go.Figure()
        cell_labels = [c["cell_name"] for c in stats_results["cells_diagnostics"]]
        obs_vals = [c["observed"] for c in stats_results["cells_diagnostics"]]
        exp_vals = [c["expected"] for c in stats_results["cells_diagnostics"]]

        fig_bar.add_trace(go.Bar(
            x=cell_labels,
            y=obs_vals,
            name="Observed Count (O)",
            marker=dict(color="#2563eb"),
            text=obs_vals,
            textposition="auto"
        ))
        fig_bar.add_trace(go.Bar(
            x=cell_labels,
            y=exp_vals,
            name="Expected Under Independence (E)",
            marker=dict(color="#94a3b8"),
            text=[f"{v:.1f}" for v in exp_vals],
            textposition="auto"
        ))
        fig_bar.update_layout(
            title="Observed vs. Expected Frequencies Under the Independence Null Hypothesis",
            barmode="group",
            xaxis_title="Contingency Cell (Group × Outcome)",
            yaxis_title="Participant Count",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_heat:
        # 2x2 Heatmap annotated with counts and Haberman z
        diag_map = {(c["row_idx"], c["col_idx"]): c for c in stats_results["cells_diagnostics"]}
        z_matrix = [
            [diag_map[(0, 0)]["haberman_residual"], diag_map[(0, 1)]["haberman_residual"]],
            [diag_map[(1, 0)]["haberman_residual"], diag_map[(1, 1)]["haberman_residual"]]
        ]
        text_matrix = [
            [
                f"Obs: {diag_map[(0, 0)]['observed']}<br>Exp: {diag_map[(0, 0)]['expected']:.1f}<br>z: {diag_map[(0, 0)]['haberman_residual']:+.2f} ({diag_map[(0, 0)]['stars']})",
                f"Obs: {diag_map[(0, 1)]['observed']}<br>Exp: {diag_map[(0, 1)]['expected']:.1f}<br>z: {diag_map[(0, 1)]['haberman_residual']:+.2f} ({diag_map[(0, 1)]['stars']})"
            ],
            [
                f"Obs: {diag_map[(1, 0)]['observed']}<br>Exp: {diag_map[(1, 0)]['expected']:.1f}<br>z: {diag_map[(1, 0)]['haberman_residual']:+.2f} ({diag_map[(1, 0)]['stars']})",
                f"Obs: {diag_map[(1, 1)]['observed']}<br>Exp: {diag_map[(1, 1)]['expected']:.1f}<br>z: {diag_map[(1, 1)]['haberman_residual']:+.2f} ({diag_map[(1, 1)]['stars']})"
            ]
        ]

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=[stats_results["outcome_pos_name"], stats_results["outcome_neg_name"]],
            y=[stats_results["group1_name"], stats_results["group2_name"]],
            colorscale="RdBu_r",
            zmid=0.0,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=13),
            colorbar=dict(title="Haberman<br>Adjusted z")
        ))
        fig_heat.update_layout(
            title="2×2 Contingency Heatmap Color-Coded by Adjusted Standardized Residual (z)",
            xaxis_title="Outcome Dimension",
            yaxis_title="Group / Treatment Dimension",
            template="plotly_white"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab_forest:
        # Forest Plot for OR, RR, and Risk Difference
        fig_forest = go.Figure()

        metrics = ["Odds Ratio (OR)", "Relative Risk (RR)"]
        estimates = [stats_results["odds_ratio"], stats_results["risk_ratio"]]
        ci_lows = [stats_results["odds_ratio_ci"][0], stats_results["risk_ratio_ci"][0]]
        ci_highs = [stats_results["odds_ratio_ci"][1], stats_results["risk_ratio_ci"][1]]

        fig_forest.add_trace(go.Scatter(
            y=metrics,
            x=estimates,
            mode="markers",
            marker=dict(size=14, color="#2563eb", symbol="diamond"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[h - e for h, e in zip(ci_highs, estimates)],
                arrayminus=[e - l for l, e in zip(ci_lows, estimates)],
                color="#1e293b",
                thickness=2,
                width=8
            ),
            name="Ratio Estimates (95% CI)"
        ))
        fig_forest.add_vline(x=1.0, line_dash="dash", line_color="#ef4444", annotation_text="No Effect (1.0)")
        fig_forest.update_layout(
            title="Comparative Ratio Effect Sizes (Odds Ratio & Relative Risk with 95% CIs)",
            xaxis_title="Ratio Point Estimate (1.0 = Null Difference)",
            yaxis_title="Ratio Metric",
            template="plotly_white"
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # 4. Statistical Tables
    st.markdown("##### 📋 Detailed Statistical Summary Tables")

    # Table 1: Cross-Tabulation
    st.markdown("**1. 2×2 Contingency Cross-Tabulation (Observed, Expected, and Rates)**")
    crosstab_df = pd.DataFrame({
        "Group_Condition": [
            stats_results["group1_name"],
            stats_results["group2_name"],
            "Column Totals (Margins)"
        ],
        "Event_Observed": [
            f"{stats_results['a']}",
            f"{stats_results['c']}",
            f"{stats_results['c1']}"
        ],
        "Event_Expected": [
            f"{stats_results['expected_matrix'][0, 0]:.1f}",
            f"{stats_results['expected_matrix'][1, 0]:.1f}",
            f"{stats_results['c1']:.1f}"
        ],
        "No_Event_Observed": [
            f"{stats_results['b']}",
            f"{stats_results['d']}",
            f"{stats_results['c2']}"
        ],
        "No_Event_Expected": [
            f"{stats_results['expected_matrix'][0, 1]:.1f}",
            f"{stats_results['expected_matrix'][1, 1]:.1f}",
            f"{stats_results['c2']:.1f}"
        ],
        "Row_Total": [
            f"{stats_results['r1']}",
            f"{stats_results['r2']}",
            f"{stats_results['n_total']}"
        ],
        "Event_Incidence_Rate": [
            f"{p1_pct:.1f}% ({stats_results['risk_group1_ci'][0]*100:.1f}% – {stats_results['risk_group1_ci'][1]*100:.1f}%)",
            f"{p2_pct:.1f}% ({stats_results['risk_group2_ci'][0]*100:.1f}% – {stats_results['risk_group2_ci'][1]*100:.1f}%)",
            f"{(stats_results['c1']/stats_results['n_total'])*100:.1f}% (Overall)"
        ]
    })
    render_table(crosstab_df, height=180)

    # Table 2: Omnibus Statistical Tests Comparison
    st.markdown("**2. Omnibus Association Test Statistics Comparison**")
    tests_df = pd.DataFrame({
        "Test_Method": [
            "Pearson's Chi-Square (Uncorrected)",
            "Yates' Continuity-Corrected Chi-Square",
            "Likelihood Ratio Chi-Square (G²)",
            "Fisher's Exact Test (Two-Sided)"
        ],
        "Test_Statistic": [
            f"χ² = {stats_results['chi2_uncorr']:.3f}",
            f"χ² = {stats_results['chi2_yates']:.3f}",
            f"G² = {stats_results['lr_g2']:.3f}",
            f"Exact OR = {stats_results['odds_ratio_fisher']:.2f}"
        ],
        "df": [
            "1",
            "1",
            "1",
            "N/A (Exact Hypergeometric)"
        ],
        "p_value": [
            f"{stats_results['p_uncorr']:.4f}",
            f"{stats_results['p_yates']:.4f}",
            f"{stats_results['p_lr']:.4f}",
            f"{stats_results['p_fisher']:.4f}"
        ],
        "Significance_Alpha_05": [
            "Significant (p < 0.05)" if stats_results["p_uncorr"] < 0.05 else "Not Significant",
            "Significant (p < 0.05)" if stats_results["p_yates"] < 0.05 else "Not Significant",
            "Significant (p < 0.05)" if stats_results["p_lr"] < 0.05 else "Not Significant",
            "Significant (p < 0.05)" if stats_results["p_fisher"] < 0.05 else "Not Significant"
        ],
        "Recommended_Application": [
            "Primary standard when all expected cell counts ≥ 5.0.",
            "Conservative adjustment for 2×2 tables; prevents over-rejection in modest samples.",
            "Preferred when modeling via log-linear analysis or logistic deviance.",
            "Gold standard when any expected cell count < 5.0 (small samples)."
        ]
    })
    render_table(tests_df, height=180)

    # Table 3: Post-Hoc Cell Residuals & Diagnostics
    st.markdown("**3. Post-Hoc Cell Residual Diagnostics (Standardized & Adjusted Haberman Residuals)**")
    cells_df = pd.DataFrame({
        "Contingency_Cell": [c["cell_name"] for c in stats_results["cells_diagnostics"]],
        "Observed_O": [f"{c['observed']}" for c in stats_results["cells_diagnostics"]],
        "Expected_E": [f"{c['expected']:.1f}" for c in stats_results["cells_diagnostics"]],
        "Raw_Diff_O_minus_E": [f"{c['diff']:+.1f}" for c in stats_results["cells_diagnostics"]],
        "Chi2_Contribution": [f"{c['chi2_contrib']:.3f}" for c in stats_results["cells_diagnostics"]],
        "Pct_Total_Chi2": [f"{c['pct_chi2']:.1f}%" for c in stats_results["cells_diagnostics"]],
        "Pearson_Residual_r": [f"{c['pearson_residual']:+.2f}" for c in stats_results["cells_diagnostics"]],
        "Haberman_Adjusted_z": [f"{c['haberman_residual']:+.2f}" for c in stats_results["cells_diagnostics"]],
        "Residual_p_value": [f"{c['p_value']:.4f}" for c in stats_results["cells_diagnostics"]],
        "Significance": [c["significance"] for c in stats_results["cells_diagnostics"]],
        "Cell_Direction": [c["direction"] for c in stats_results["cells_diagnostics"]]
    })
    render_table(cells_df, height=240)

    with st.expander("💡 Plain-English Guide: Why Adjusted Standardized (Haberman) Residuals Matter in Post-Hoc Testing", expanded=False):
        st.markdown("""
        **What are Haberman Adjusted Standardized Residuals ($z$)?**
        * While the omnibus Chi-Square test tells you *that* the two variables are associated, it does not reveal *which specific combinations of categories drove the result*.
        * **Raw Difference ($O - E$):** Simply subtracts expected from observed counts, but does not account for cell size.
        * **Pearson Residual ($r = (O - E) / \\sqrt{E}$):** Standardizes by expected count variance, but its variance is strictly less than 1.0 because margins are fixed.
        * **Haberman Adjusted Residual ($z$):** Corrects for row and column marginal totals:
          $$z_{ij} = \\frac{O_{ij} - E_{ij}}{\\sqrt{E_{ij} \\left(1 - \\frac{R_i}{N}\\right) \\left(1 - \\frac{C_j}{N}\\right)}}$$
          This transforms each cell residual directly into a standard normal $Z$-score (mean = 0, standard deviation = 1).

        **How to interpret Haberman Residuals in practice:**
        * $|z| > 1.96$: Statistically significant deviation at $\\alpha = 0.05$ ($p < 0.05$).
        * $|z| > 2.58$: Very significant deviation at $\\alpha = 0.01$ ($p < 0.01$).
        * $|z| > 3.29$: Highly significant deviation at $\\alpha = 0.001$ ($p < 0.001$).
        * **Positive $z > 0$ (Excess):** That group experienced significantly **more** events than would be expected if the groups were identical.
        * **Negative $z < 0$ (Deficit):** That group experienced significantly **fewer** events than expected under random independence.
        """)

    # Table 4: Epidemiological Effect Sizes & Strength of Association
    st.markdown("**4. Epidemiological Effect Sizes & Association Strength**")
    effects_df = pd.DataFrame({
        "Effect_Metric": [
            "Odds Ratio (OR)",
            "Relative Risk (RR) / Risk Ratio",
            "Risk Difference (RD) / ARR",
            "Number Needed to Treat (NNT)",
            "Phi Coefficient (φ)",
            "Cramér's V",
            "Cohen's w"
        ],
        "Point_Estimate": [
            f"{or_val:.3f}",
            f"{rr_val:.3f}",
            f"{stats_results['risk_diff']:+.3f} ({rd_pct:+.1f}%)",
            f"{stats_results['nnt']:.1f}" if stats_results["nnt"] else "N/A (RD = 0)",
            f"{stats_results['phi']:.3f}",
            f"{stats_results['cramers_v']:.3f}",
            f"{stats_results['cohens_w']:.3f}"
        ],
        "CI_95_Lower": [
            f"{or_ci[0]:.3f}",
            f"{stats_results['risk_ratio_ci'][0]:.3f}",
            f"{stats_results['risk_diff_ci'][0]:+.3f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ],
        "CI_95_Upper": [
            f"{or_ci[1]:.3f}",
            f"{stats_results['risk_ratio_ci'][1]:.3f}",
            f"{stats_results['risk_diff_ci'][1]:+.3f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ],
        "Clinical_Interpretation_Guideline": [
            "Odds of outcome in Group 1 relative to Group 2 (1.0 = no difference).",
            "Probability of outcome in Group 1 divided by Group 2 (1.0 = no difference).",
            "Absolute percentage difference in risk between groups (0.0 = identical).",
            "Number of participants needed to treat to see one additional positive outcome.",
            "Correlation coefficient for binary variables (0.1 = small, 0.3 = medium, 0.5 = large).",
            "Identical to Phi for 2×2 tables; standard metric for association magnitude.",
            "Standardized effect size used for Chi-square power calculations."
        ]
    })
    render_table(effects_df, height=240)

    # Table 5: Observational Dataset Preview
    st.markdown(f"**5. Subject-Level Observational Data Preview (N = {stats_results['n_total']} Participants)**")
    df_cases = stats_results["df_cases"]
    render_table(df_cases.head(10), height=180)
    st.download_button(
        "📥 Download Simulated Subject Dataset (CSV)",
        data=df_cases.to_csv(index=False),
        file_name="contingency_table_subjects.csv",
        mime="text/csv",
        key="btn_download_contingency"
    )

    # 5. Core Assumption Evaluation Cards
    st.markdown("### 🛡️ Core Methodological Assumption Diagnostics")
    st.caption("Pearson's Chi-Square test assumes categorical scale, independent observations, and adequate expected frequencies ($E_{ij} \\ge 5$).")

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        if asm["status"] == "pass":
            status_color = "#10b981"
            bg_color = "#f0fdf4"
            badge_symbol = "✅"
        elif asm["status"] == "warning":
            status_color = "#f59e0b"
            bg_color = "#fffbeb"
            badge_symbol = "🟡"
        else:
            status_color = "#ef4444"
            bg_color = "#fef2f2"
            badge_symbol = "❌"

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 6. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    elif stats_results["overall_status"] == "warning":
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.error(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # 7. Expandable Details for Alternative Tests
    with st.expander("🔎 View Alternative Test Details (Fisher's Exact & McNemar's Test)", expanded=False):
        st.markdown(f"""
        * **Fisher's Exact Test (Hypergeometric Small-Sample Alternative):**
          * Two-Sided Exact p-value: **`{stats_results['p_fisher']:.4f}`** | Exact Odds Ratio: **`{stats_results['odds_ratio_fisher']:.2f}`**
          * Use when any expected cell count is $< 5.0$. Unlike Pearson Chi-Square, Fisher's Exact test does not rely on an asymptotic continuous approximation to the discrete binomial distribution.
        * **McNemar's Test (Paired Binary Data Alternative):**
          * If your observations were collected on the **same individuals across two time points** (e.g., Before vs After) or on **matched case-control pairs**, do not use Pearson Chi-Square.
          * McNemar's test evaluates discordance exclusively across discordant cells ($b$ vs $c$): $\\chi^2_{{\\text{{McNemar}}}} = (|b - c| - 1)^2 / (b + c)$.
        """)


def render_fisher_exact_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, hypergeometric visualizations, and diagnostics for Fisher's Exact Test."""
    st.markdown("### 🎯 Hypothesis Test Results: Fisher's Exact Test")

    # Extract user-entered labels and 2x2 table parameters
    g1 = stats_results["group1_name"]
    g2 = stats_results["group2_name"]
    out_pos = stats_results["outcome_pos_name"]
    out_neg = stats_results["outcome_neg_name"]
    a = stats_results["a"]
    b = stats_results["b"]
    c = stats_results["c"]
    d = stats_results["d"]
    r1 = stats_results["r1"]
    r2 = stats_results["r2"]
    c1 = stats_results["c1"]
    c2 = stats_results["c2"]
    n_total = stats_results["n_total"]
    exp_mat = stats_results["expected_matrix"]
    exp_a = exp_mat[0, 0]
    exp_b = exp_mat[0, 1]
    exp_c = exp_mat[1, 0]
    exp_d = exp_mat[1, 1]
    p1_pct = stats_results["risk_group1"] * 100.0
    p2_pct = stats_results["risk_group2"] * 100.0
    rd_pct = stats_results["risk_diff"] * 100.0
    or_val = stats_results["odds_ratio_fisher"]
    or_ci = stats_results["odds_ratio_ci"]
    min_exp = stats_results["min_expected_count"]
    is_sig = stats_results["p_two_sided"] < 0.05

    # Safe HTML-escaped strings for UI rendering
    g1_esc = html.escape(str(g1))
    g2_esc = html.escape(str(g2))
    out_pos_esc = html.escape(str(out_pos))
    out_neg_esc = html.escape(str(out_neg))

    # Margin percentages of Grand Total N
    r1_pct_of_n = (r1 / n_total * 100.0) if n_total > 0 else 0.0
    r2_pct_of_n = (r2 / n_total * 100.0) if n_total > 0 else 0.0
    c1_pct_of_n = (c1 / n_total * 100.0) if n_total > 0 else 0.0
    c2_pct_of_n = (c2 / n_total * 100.0) if n_total > 0 else 0.0

    # 1. User-Configured 2x2 Contingency Grid
    grid_html = (
        f'<div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">'
        f'<span style="font-weight: 700; font-size: 1.05rem; color: #1e3a8a;">🏷️ Observed 2×2 Contingency Table (Cross-Classification)</span>'
        f'<span style="font-size: 0.85rem; background: #eff6ff; color: #2563eb; padding: 0.25rem 0.65rem; border-radius: 6px; font-weight: 600;">Fixed Margins: R₁ = {r1}, R₂ = {r2} | C₁ = {c1}, C₂ = {c2} | Total N = {n_total}</span>'
        f'</div>'
        f'<div style="overflow-x: auto;">'
        f'<table style="width: 100%; border-collapse: separate; border-spacing: 8px; text-align: center; margin: 0 auto;">'
        f'<thead>'
        f'<tr>'
        f'<th rowspan="2" style="background: #f8fafc; color: #475569; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; vertical-align: middle; width: 22%;">Row Dimension<br><span style="font-size: 0.75rem; font-weight: normal; color: #64748b;">(Group / Exposure)</span></th>'
        f'<th colspan="2" style="background: #eff6ff; color: #1e40af; padding: 8px; border-radius: 8px; border: 1px solid #bfdbfe; font-size: 0.95rem; font-weight: 700;">Column Dimension: Outcome / Status</th>'
        f'<th rowspan="2" style="background: #f8fafc; color: #475569; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; vertical-align: middle; width: 18%;">Row Margins<br><span style="font-size: 0.75rem; font-weight: normal; color: #64748b;">(Row Totals)</span></th>'
        f'</tr>'
        f'<tr>'
        f'<th style="background: #f0fdf4; color: #166534; padding: 10px; border-radius: 8px; border: 1px solid #bbf7d0; font-size: 0.95rem; font-weight: 700; width: 30%;">{out_pos_esc}</th>'
        f'<th style="background: #f8fafc; color: #334155; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 0.95rem; font-weight: 700; width: 30%;">{out_neg_esc}</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>'
        f'<tr>'
        f'<th style="background: #eff6ff; color: #1e3a8a; padding: 12px; border-radius: 8px; border: 1px solid #bfdbfe; text-align: left; font-size: 0.95rem; font-weight: 700; vertical-align: middle;">{g1_esc}</th>'
        f'<td style="background: #ffffff; border: 2px solid #2563eb; border-radius: 8px; padding: 12px; vertical-align: middle; box-shadow: 0 1px 2px rgba(37,99,235,0.08);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;"><span style="font-size: 0.75rem; font-weight: 700; color: #2563eb; text-transform: uppercase; background: #eff6ff; padding: 1px 6px; border-radius: 4px;">Cell a</span><span style="font-size: 0.75rem; color: #64748b;">Exp: {exp_a:.1f}</span></div>'
        f'<div style="font-size: 1.75rem; font-weight: 800; color: #1e293b; line-height: 1.2;">{a}</div>'
        f'<div style="font-size: 0.8rem; color: #0284c7; font-weight: 600; margin-top: 2px;">Rate: {p1_pct:.1f}%</div>'
        f'</td>'
        f'<td style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; vertical-align: middle;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;"><span style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; background: #f1f5f9; padding: 1px 6px; border-radius: 4px;">Cell b</span><span style="font-size: 0.75rem; color: #64748b;">Exp: {exp_b:.1f}</span></div>'
        f'<div style="font-size: 1.75rem; font-weight: 800; color: #1e293b; line-height: 1.2;">{b}</div>'
        f'<div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">Rate: {100.0 - p1_pct:.1f}%</div>'
        f'</td>'
        f'<td style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 12px; vertical-align: middle;">'
        f'<div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">R₁ = a + b</div>'
        f'<div style="font-size: 1.4rem; font-weight: 700; color: #1e293b; line-height: 1.2;">{r1}</div>'
        f'<div style="font-size: 0.75rem; color: #94a3b8;">{r1_pct_of_n:.1f}% of N</div>'
        f'</td>'
        f'</tr>'
        f'<tr>'
        f'<th style="background: #f8fafc; color: #334155; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: left; font-size: 0.95rem; font-weight: 700; vertical-align: middle;">{g2_esc}</th>'
        f'<td style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; vertical-align: middle;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;"><span style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; background: #f1f5f9; padding: 1px 6px; border-radius: 4px;">Cell c</span><span style="font-size: 0.75rem; color: #64748b;">Exp: {exp_c:.1f}</span></div>'
        f'<div style="font-size: 1.75rem; font-weight: 800; color: #1e293b; line-height: 1.2;">{c}</div>'
        f'<div style="font-size: 0.8rem; color: #0284c7; font-weight: 600; margin-top: 2px;">Rate: {p2_pct:.1f}%</div>'
        f'</td>'
        f'<td style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; vertical-align: middle;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;"><span style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; background: #f1f5f9; padding: 1px 6px; border-radius: 4px;">Cell d</span><span style="font-size: 0.75rem; color: #64748b;">Exp: {exp_d:.1f}</span></div>'
        f'<div style="font-size: 1.75rem; font-weight: 800; color: #1e293b; line-height: 1.2;">{d}</div>'
        f'<div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">Rate: {100.0 - p2_pct:.1f}%</div>'
        f'</td>'
        f'<td style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 12px; vertical-align: middle;">'
        f'<div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">R₂ = c + d</div>'
        f'<div style="font-size: 1.4rem; font-weight: 700; color: #1e293b; line-height: 1.2;">{r2}</div>'
        f'<div style="font-size: 0.75rem; color: #94a3b8;">{r2_pct_of_n:.1f}% of N</div>'
        f'</td>'
        f'</tr>'
        f'<tr>'
        f'<th style="background: #f8fafc; color: #475569; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: left; font-size: 0.85rem; font-weight: 700; vertical-align: middle;">Column Margins<br><span style="font-size: 0.75rem; font-weight: normal; color: #64748b;">(Column Totals)</span></th>'
        f'<td style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 10px; vertical-align: middle;">'
        f'<div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">C₁ = a + c</div>'
        f'<div style="font-size: 1.4rem; font-weight: 700; color: #1e293b; line-height: 1.2;">{c1}</div>'
        f'<div style="font-size: 0.75rem; color: #94a3b8;">{c1_pct_of_n:.1f}% of N</div>'
        f'</td>'
        f'<td style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 10px; vertical-align: middle;">'
        f'<div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">C₂ = b + d</div>'
        f'<div style="font-size: 1.4rem; font-weight: 700; color: #1e293b; line-height: 1.2;">{c2}</div>'
        f'<div style="font-size: 0.75rem; color: #94a3b8;">{c2_pct_of_n:.1f}% of N</div>'
        f'</td>'
        f'<td style="background: #eff6ff; border: 2px solid #3b82f6; border-radius: 8px; padding: 10px; vertical-align: middle;">'
        f'<div style="font-size: 0.75rem; font-weight: 700; color: #1d4ed8; text-transform: uppercase;">Grand Total</div>'
        f'<div style="font-size: 1.6rem; font-weight: 800; color: #1e3a8a; line-height: 1.2;">N = {n_total}</div>'
        f'<div style="font-size: 0.75rem; color: #2563eb; font-weight: 600;">100.0%</div>'
        f'</td>'
        f'</tr>'
        f'</tbody>'
        f'</table>'
        f'</div>'
        f'<div style="font-size: 0.8rem; color: #64748b; margin-top: 0.6rem; padding-top: 0.5rem; border-top: 1px dashed #e2e8f0; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">'
        f'<span>📌 <em>Cell a, b, c, d values and row/column labels dynamically reflect your sidebar settings.</em></span>'
        f'<span>💡 <em>Under H₀, Expected Count (Exp) = (Row Total × Column Total) / Grand Total N</em></span>'
        f'</div>'
        f'</div>'
    )
    render_html(grid_html)

    # 2. Key Statistical Metrics Grid
    st.markdown("##### 📊 Key Statistical Summary Metrics")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    mcol1.metric(
        "Grand Sample Size (N)",
        f"{n_total}",
        help="Total number of observations across the 2×2 table."
    )
    mcol1.metric(
        "Possible Table Set",
        f"{len(stats_results['hypergeom_distribution'])} tables",
        help=f"Number of valid contingency tables preserving fixed margins: Row 1 = {r1}, Row 2 = {r2}, Col 1 = {c1}, Col 2 = {c2}."
    )

    mcol2.metric(
        "Observed P(a)",
        f"{stats_results['p_observed']:.4f}",
        help=f"Exact point probability of obtaining this specific observed table ({a}) under H0."
    )
    mcol2.metric(
        "Exact Odds Ratio",
        f"{or_val:.2f}",
        help=f"Exact conditional maximum likelihood odds ratio. 95% CI: [{or_ci[0]:.2f}, {or_ci[1]:.2f}]."
    )

    sig_delta = "p < 0.05 (Significant)" if stats_results["p_two_sided"] < 0.05 else "p ≥ 0.05 (Not Significant)"
    mcol3.metric(
        "Fisher Exact p (Two-Sided)",
        f"{stats_results['p_two_sided']:.4f}",
        delta=sig_delta,
        delta_color="normal" if stats_results["p_two_sided"] < 0.05 else "off",
        help="Sum of exact probabilities of all possible tables having probability <= observed table probability."
    )
    mcol3.metric(
        "Right-Tail p (Greater)",
        f"{stats_results['p_greater']:.4f}",
        help=f"One-sided exact probability of observing {a} or more events in Group 1 (testing if Group 1 odds > Group 2 odds)."
    )

    exp_delta = "< 5.0 (Mandates Exact)" if min_exp < 5.0 else "≥ 5.0"
    mcol4.metric(
        "Min Expected Cell (E_min)",
        f"{min_exp:.2f}",
        delta=exp_delta,
        delta_color="inverse" if min_exp < 5.0 else "normal",
        help="Expected cell count under independence. When E_min < 5.0, asymptotic Chi-Square is invalid and Fisher's Exact Test is strictly required."
    )
    mcol4.metric(
        "Pearson Chi-Square p",
        f"{stats_results['p_chi2_uncorr']:.4f}",
        delta=f"Diff: {stats_results['p_discrepancy']:.3f}",
        delta_color="off",
        help=f"Asymptotic Pearson Chi-Square p-value ({stats_results['p_chi2_uncorr']:.4f}). Discrepancy from exact p reflects approximation distortion on small cell counts."
    )

    # 3. Clinical & Practical Decision Interpretation in Plain English
    st.markdown("##### 🩺 Clinical & Statistical Decision Interpretation")

    if is_sig:
        st.success(f"""
        **Primary Finding: Statistically Significant Association via Fisher's Exact Test (Two-Sided p = {stats_results['p_two_sided']:.4f} < 0.05)**
        * **Hypotheses Tested:**
          * *Null Hypothesis ($H_0$):* There is no association between **{g1} vs. {g2}** and the occurrence of **{out_pos}** (odds ratio = 1.0).
          * *Alternative Hypothesis ($H_1$):* There is a genuine statistical association between groups and outcome occurrence (odds ratio $\\ne$ 1.0).
        * **Observed Event Rates:** Individuals in **{g1}** experienced an event rate of **{p1_pct:.1f}%** compared to **{p2_pct:.1f}%** in **{g2}** (Absolute Risk Difference: **{rd_pct:+.1f}%**).
        * **Exact Effect Sizes:**
          * **Odds Ratio (OR):** `{or_val:.2f}` (95% CI: `{or_ci[0]:.2f}` – `{or_ci[1]:.2f}`) — The odds of experiencing {out_pos} are {or_val:.2f} times as high in {g1} compared to {g2}.
          * **Relative Risk (RR):** `{stats_results['risk_ratio']:.2f}` (95% CI: `{stats_results['risk_ratio_ci'][0]:.2f}` – `{stats_results['risk_ratio_ci'][1]:.2f}`).
        * **Why Fisher's Exact Test is Essential Here:** 
          Because the minimum expected cell count is `{min_exp:.2f}` (< 5.0), the standard Pearson Chi-Square test is mathematically invalid. Fisher's Exact Test evaluates the exact combinatorial probability under the hypergeometric distribution, guaranteeing zero approximation error.
        """)
    else:
        st.info(f"""
        **Primary Finding: No Statistically Significant Association via Fisher's Exact Test (Two-Sided p = {stats_results['p_two_sided']:.4f} ≥ 0.05)**
        * **Hypotheses Tested:**
          * *Null Hypothesis ($H_0$):* Group membership and outcome occurrence are independent.
          * *Alternative Hypothesis ($H_1$):* Group membership and outcome occurrence are associated.
        * **Observed Event Rates:** Individuals in **{g1}** had an event rate of **{p1_pct:.1f}%** vs. **{p2_pct:.1f}%** in **{g2}** (Absolute Difference: **{rd_pct:+.1f}%**).
        * **Odds Ratio (OR):** `{or_val:.2f}` (95% CI: `{or_ci[0]:.2f}` – `{or_ci[1]:.2f}`). The 95% confidence interval spans `1.0` (the neutral null benchmark).
        * **One-Sided Probabilities:** Right-tail $p = {stats_results['p_greater']:.4f}$ | Left-tail $p = {stats_results['p_less']:.4f}$.
        * **Conclusion:** We fail to reject the null hypothesis. The observed difference in rates is consistent with random chance sampling variability given the small sample size.
        """)

    # 3. Interactive Plotly Visualizations (3 Tabs)
    st.markdown("##### 📈 Visual Inspection & Hypergeometric Distribution Diagnostics")
    tab_dist, tab_bar, tab_forest = st.tabs([
        "🎲 Hypergeometric Distribution & Rejection Region",
        "📊 Observed vs Expected Counts",
        "🌲 Effect Sizes Forest Plot (OR, RR, RD)"
    ])

    with tab_dist:
        st.markdown("""
        **What does this distribution show?**
        * **The Hypergeometric Probability Space:** Unlike continuous tests that assume data follows a smooth theoretical bell curve, Fisher's Exact Test works directly with **discrete counts** (whole numbers of individuals). Keeping the total sizes of both groups ($R_1, R_2$) and the total counts of both outcomes ($C_1, C_2$) strictly fixed, this bar chart maps out the **exact mathematical probability** of every possible way the table could have turned out if there were no genuine connection between group and outcome (the null hypothesis $H_0$).
        * **Horizontal Axis ($k$ = Count in Cell $a$):** The number of positive events in Group 1. Each individual bar represents one complete, distinct $2 \\times 2$ contingency table.
        * **Vertical Axis ($P(X = k)$):** The exact probability of observing that specific table configuration under pure random sampling chance. The probabilities across all bars sum to exactly $1.00000$ ($100\\%$).

        **How to interpret the color coding & decision rule:**
        * 🔵 **Blue Bar (Observed Table):** Highlights your actual experiment ($k = a$). The height of this bar is the exact point probability $P(a)$ of getting this specific result.
        * 🔴 **Red Bars (Rejection Region / Critical Tail):** Tables that are *even more extreme or unlikely* than the observed table (where $P(k) \\le P(a)$).
        * ⚪ **Gray Bars (Non-Rejection Region):** Common, central table configurations expected under the null hypothesis.
        * **Calculating the Two-Sided p-value:** Fisher's Exact Test simply **sums the heights of the Blue bar and all Red bars**. If this total sum is $< 0.05$ (less than a $5\\%$ chance of occurring by fluke), we reject the null hypothesis and conclude that the groups are genuinely different.
        """)

        # Discrete hypergeometric probability distribution of all possible tables
        dist = stats_results["hypergeom_distribution"]
        k_vals = [tbl["k"] for tbl in dist]
        probs = [tbl["prob"] for tbl in dist]
        colors = []
        bar_text = []

        for tbl in dist:
            if tbl["is_observed"]:
                colors.append("#2563eb")  # Highlight observed table in strong blue
                bar_text.append(f"Observed (k={tbl['k']})<br>P={tbl['prob']:.4f}")
            elif tbl["is_two_sided_rejection"]:
                colors.append("#ef4444")  # Two-sided rejection tables in red
                bar_text.append(f"P={tbl['prob']:.4f}<br>(Rejection)")
            else:
                colors.append("#cbd5e1")  # Non-rejection tables in muted slate
                bar_text.append(f"P={tbl['prob']:.4f}")

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=[f"k = {k}" for k in k_vals],
            y=probs,
            marker=dict(color=colors),
            text=bar_text,
            textposition="auto",
            hoverinfo="text",
            hovertext=[
                f"<b>Table configuration with cell a = {tbl['k']}:</b><br>"
                f"{tbl['table_repr']}<br>"
                f"Point Probability P(X={tbl['k']}): {tbl['prob']:.5f}<br>"
                f"Cumulative P(X<={tbl['k']}): {tbl['cum_prob']:.5f}<br>"
                f"In Two-Sided Tail (P <= P_obs): {'Yes' if tbl['is_two_sided_rejection'] else 'No'}"
                for tbl in dist
            ]
        ))
        fig_dist.update_layout(
            title=f"Discrete Hypergeometric Probability Distribution P(X = k) Across All {len(dist)} Possible Tables<br><sup>Blue = Observed Table (k={stats_results['a']}), Red = Rejection Region (P(k) ≤ P(a)), Gray = Non-Rejection Region</sup>",
            xaxis_title=f"Count of Successes in {stats_results['group1_name']} (Cell a)",
            yaxis_title="Exact Probability P(X = k)",
            template="plotly_white"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab_bar:
        st.markdown("""
        **What does this chart compare?**
        * **Observed Counts ($O$ - Blue Bars):** The real headcounts actually recorded in your data for each of the four cells (e.g. Group 1 with positive outcome, Group 1 with negative outcome, etc.).
        * **Expected Frequencies Under Independence ($E$ - Gray Bars):** The theoretical baseline number of participants you would expect to see in that cell if there were **zero connection** between group membership and the outcome. Calculated as:
          $$\\text{Expected Count } (E) = \\frac{\\text{Row Total} \\times \\text{Column Total}}{\\text{Grand Total } N}$$

        **How to interpret differences:**
        * **Divergence (Blue vs. Gray):** If the blue observed bar is noticeably taller than the gray expected bar, that category occurred more often than expected by chance.
        * **Cochran's Small Sample Diagnostic:** Look at the height of the gray bars. If **any gray bar is below 5.0**, standard Pearson Chi-Square tests are mathematically compromised because continuous approximations break down. This small expected count is the precise reason Fisher's Exact Test is required.
        """)

        # Grouped bar chart comparing Observed vs Expected
        fig_bar = go.Figure()
        cell_labels = [
            f"{stats_results['group1_name']} × {stats_results['outcome_pos_name']}",
            f"{stats_results['group1_name']} × {stats_results['outcome_neg_name']}",
            f"{stats_results['group2_name']} × {stats_results['outcome_pos_name']}",
            f"{stats_results['group2_name']} × {stats_results['outcome_neg_name']}"
        ]
        obs_vals = [stats_results["a"], stats_results["b"], stats_results["c"], stats_results["d"]]
        exp_mat = stats_results["expected_matrix"]
        exp_vals = [exp_mat[0, 0], exp_mat[0, 1], exp_mat[1, 0], exp_mat[1, 1]]

        fig_bar.add_trace(go.Bar(
            x=cell_labels,
            y=obs_vals,
            name="Observed Count (O)",
            marker=dict(color="#2563eb"),
            text=obs_vals,
            textposition="auto"
        ))
        fig_bar.add_trace(go.Bar(
            x=cell_labels,
            y=exp_vals,
            name="Expected Count Under H0 (E)",
            marker=dict(color="#94a3b8"),
            text=[f"{v:.1f}" for v in exp_vals],
            textposition="auto"
        ))
        fig_bar.update_layout(
            title="Observed Counts vs Expected Frequencies Under the Independence Null Hypothesis",
            barmode="group",
            xaxis_title="Contingency Cell",
            yaxis_title="Participant Count",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_forest:
        st.markdown("""
        **What does a Forest Plot show?**
        * A **Forest Plot** provides a graphical summary of effect sizes and their precision, allowing clinicians and researchers to see both the strength of the relationship and the uncertainty of the estimate at a single glance.

        **Key Metrics Explained (Assume No Prior Knowledge):**
        * **Exact Odds Ratio (OR):** Compares the odds of having the outcome in Group 1 versus Group 2.
          * $\\text{OR} = 1.0$: Identical odds in both groups (no difference).
          * $\\text{OR} > 1.0$: Higher odds in Group 1 (e.g. $\\text{OR} = 2.5$ means Group 1 has 2.5 times higher odds).
          * $\\text{OR} < 1.0$: Lower odds in Group 1 (protective effect).
        * **Relative Risk (RR / Risk Ratio):** Compares the actual probability (percentage event rate) in Group 1 divided by Group 2. For rare events, RR and OR are very close; for common events, OR exaggerates relative risk.
        * **95% Confidence Intervals (Horizontal Whiskers):** The plausible range where the true population effect lies with 95% certainty.

        **How to interpret the plot:**
        * **Red Dashed Vertical Line at 1.0 (Null Benchmark):** Represents the point of "no difference".
        * **Statistically Significant:** If the entire horizontal line (including whiskers) stays **completely to the right of 1.0** or **completely to the left of 1.0**, the difference is statistically significant ($p < 0.05$).
        * **Not Significant:** If the horizontal line **crosses or touches 1.0**, the observed difference could easily be random sampling noise ($p \\ge 0.05$).
        """)

        # Forest Plot for OR and RR
        fig_forest = go.Figure()
        metrics = ["Exact Odds Ratio (OR)", "Relative Risk (RR)"]
        estimates = [stats_results["odds_ratio_fisher"], stats_results["risk_ratio"]]
        ci_lows = [stats_results["odds_ratio_ci"][0], stats_results["risk_ratio_ci"][0]]
        ci_highs = [stats_results["odds_ratio_ci"][1], stats_results["risk_ratio_ci"][1]]

        fig_forest.add_trace(go.Scatter(
            y=metrics,
            x=estimates,
            mode="markers",
            marker=dict(size=14, color="#2563eb", symbol="diamond"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[h - e for h, e in zip(ci_highs, estimates)],
                arrayminus=[e - l for l, e in zip(ci_lows, estimates)],
                color="#1e293b",
                thickness=2,
                width=8
            ),
            name="Ratio Estimates (95% CI)"
        ))
        fig_forest.add_vline(x=1.0, line_dash="dash", line_color="#ef4444", annotation_text="No Effect (1.0)")
        fig_forest.update_layout(
            title="Comparative Ratio Effect Sizes (Exact Odds Ratio & Relative Risk with 95% CIs)",
            xaxis_title="Ratio Point Estimate (1.0 = Null Difference)",
            yaxis_title="Ratio Metric",
            template="plotly_white"
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # 4. Statistical Tables
    st.markdown("##### 📋 Detailed Statistical Summary Tables")

    # Table 1: Cross-Tabulation
    st.markdown(f"**1. 2×2 Contingency Cross-Tabulation ({g1} vs. {g2} by {out_pos} / {out_neg})**")
    st.markdown(f"""
    **What this table contains:**
    * Cross-classifies all participants across both study groups (**{g1}** and **{g2}**) and outcome categories (**{out_pos}** and **{out_neg}**).
    * Shows both the **Observed counts** (the real headcounts collected in your data) and the theoretical **Expected counts** (what we would expect to see if group and outcome were completely independent of each other).
    * Summarises the row totals ($R_1, R_2$), column totals ($C_1, C_2$), and calculates the empirical **{out_pos} Rate (%)** with 95% Confidence Intervals for each group.

    **How to interpret:**
    * **Observed vs Expected:** Compare each observed cell count to its expected baseline. If observed is higher than expected, that particular outcome occurred more often in that group than expected under pure chance.
    * **Event Rate Difference:** Compare the percentage rates between groups. If {g1} has a substantially higher or lower rate than {g2}, there may be an association.
    * **Cochran's Small-Cell Diagnostic Check:** Look at the Expected counts. If **any expected cell count is below 5.0**, classical continuous Chi-Square tests are statistically compromised because their smooth bell-curve approximation breaks down. Fisher's Exact Test is required because it calculates exact probabilities without relying on large-sample approximations.
    """)
    crosstab_df = pd.DataFrame({
        "Group_Condition": [
            g1,
            g2,
            "Column Margins (Totals)"
        ],
        f"{out_pos} [Observed]": [
            f"{stats_results['a']}",
            f"{stats_results['c']}",
            f"{c1}"
        ],
        f"{out_pos} [Expected]": [
            f"{exp_a:.1f}",
            f"{exp_c:.1f}",
            f"{c1:.1f}"
        ],
        f"{out_neg} [Observed]": [
            f"{stats_results['b']}",
            f"{stats_results['d']}",
            f"{c2}"
        ],
        f"{out_neg} [Expected]": [
            f"{exp_b:.1f}",
            f"{exp_d:.1f}",
            f"{c2:.1f}"
        ],
        "Row_Total": [
            f"{r1}",
            f"{r2}",
            f"{n_total}"
        ],
        f"{out_pos} Rate (%)": [
            f"{p1_pct:.1f}% ({stats_results['risk_group1_ci'][0]*100:.1f}% – {stats_results['risk_group1_ci'][1]*100:.1f}%)",
            f"{p2_pct:.1f}% ({stats_results['risk_group2_ci'][0]*100:.1f}% – {stats_results['risk_group2_ci'][1]*100:.1f}%)",
            f"{(c1/n_total)*100.0:.1f}% (Overall)"
        ]
    })
    render_table(crosstab_df, height=180)

    # Table 2: Exact vs Asymptotic Comparison Table
    st.markdown("**2. Exact vs. Asymptotic Association Tests Comparison**")
    st.markdown(f"""
    **What this table contains:**
    * A direct side-by-side comparison between **Fisher's Exact Test** (under two-sided, right-tail, and left-tail directional hypotheses) and classical large-sample approximations (**Pearson's Chi-Square** and **Yates' Continuity-Corrected Chi-Square**).
    * Details each method's mathematical foundation, test statistic, calculated $p$-value, significance decision at $\\alpha = 0.05$, and methodological reliability.

    **How to interpret:**
    * **Exact vs Asymptotic (Approximation):** Fisher's test calculates exact probabilities directly from the discrete combinatorial hypergeometric distribution. In contrast, Pearson's Chi-Square assumes sample statistics follow a smooth theoretical continuous curve, which often produces distorted $p$-values and higher false positive rates in small or sparse samples.
    * **Two-Sided vs Directional Tails:**
      * **Two-Sided (Standard):** Tests whether *either* group has different odds of the outcome (non-directional; two-tailed).
      * **Right-Tail (Greater):** Evaluates whether the odds of {out_pos} are specifically *greater* in {g1} than in {g2}.
      * **Left-Tail (Less):** Evaluates whether the odds of {out_pos} are specifically *lower* in {g1} than in {g2}.
    * **Yates' Continuity Correction:** A historical compromise that subtracts $0.5$ from cell discrepancies to adjust for using a continuous curve to model discrete integers. While more conservative than uncorrected Chi-Square, it remains an approximation and is superseded by Fisher's exact method.
    """)
    tests_df = pd.DataFrame({
        "Test_Method": [
            "Fisher's Exact Test (Two-Sided)",
            "Fisher's Exact Test (Right-Tail / Greater)",
            "Fisher's Exact Test (Left-Tail / Less)",
            "Pearson's Chi-Square (Uncorrected)",
            "Yates' Continuity-Corrected Chi-Square"
        ],
        "Mathematical_Basis": [
            "Exact Hypergeometric Distribution",
            "Exact Hypergeometric Distribution",
            "Exact Hypergeometric Distribution",
            "Continuous Asymptotic χ² Approximation",
            "Asymptotic χ² with Discrete 0.5 Adjustment"
        ],
        "Test_Statistic": [
            f"Exact OR = {stats_results['odds_ratio_fisher']:.2f}",
            f"Exact OR = {stats_results['odds_ratio_fisher']:.2f}",
            f"Exact OR = {stats_results['odds_ratio_fisher']:.2f}",
            f"χ² = {stats_results['chi2_uncorr']:.3f} (df=1)",
            f"χ² = {stats_results['chi2_yates']:.3f} (df=1)"
        ],
        "p_value": [
            f"{stats_results['p_two_sided']:.4f}",
            f"{stats_results['p_greater']:.4f}",
            f"{stats_results['p_less']:.4f}",
            f"{stats_results['p_chi2_uncorr']:.4f}",
            f"{stats_results['p_chi2_yates']:.4f}"
        ],
        "Significance_Alpha_05": [
            "Significant (p < 0.05)" if stats_results["p_two_sided"] < 0.05 else "Not Significant",
            "Significant (p < 0.05)" if stats_results["p_greater"] < 0.05 else "Not Significant",
            "Significant (p < 0.05)" if stats_results["p_less"] < 0.05 else "Not Significant",
            "Significant (p < 0.05)" if stats_results["p_chi2_uncorr"] < 0.05 else "Not Significant",
            "Significant (p < 0.05)" if stats_results["p_chi2_yates"] < 0.05 else "Not Significant"
        ],
        "Reliability_Assessment": [
            "Gold standard for small sample sizes and sparse tables (E < 5.0).",
            "Use when testing a directional hypothesis that Group 1 odds > Group 2.",
            "Use when testing a directional hypothesis that Group 1 odds < Group 2.",
            "Unreliable when expected cells < 5.0; distorts true p-value.",
            "Conservative approximation; still inferior to exact computation."
        ]
    })
    render_table(tests_df, height=240)

    # Table 3: Complete Hypergeometric Distribution Table
    st.markdown(f"**3. Complete Hypergeometric Distribution of All {len(dist)} Possible Tables Preserving Marginal Totals**")
    st.markdown(f"""
    **What this table contains:**
    * An exhaustive, transparent catalog of all **{len(dist)} mathematically possible** $2 \\times 2$ contingency tables that could possibly exist while keeping the observed row totals ($R_1 = {r1}, R_2 = {r2}$) and column totals ($C_1 = {c1}, C_2 = {c2}$) strictly fixed.
    * For each possible value of $k$ (the count of positive outcomes in {g1}), it displays the full $2 \\times 2$ configuration, its exact **Point Probability** $P(X = k)$, cumulative probability, and whether it contributes to the rejection tails.

    **How to interpret:**
    * **Point Probability ($P$):** The exact combinatorial likelihood of that specific table configuration occurring by pure random chance under the null hypothesis ($H_0$). The sum of point probabilities across all rows equals exactly $1.00000$ ($100\\%$).
    * **⭐ OBSERVED TABLE:** Marks the exact data configuration observed in your experiment ($k = {a}$).
    * **Two-Sided Rejection Region ($P \\le P_{{\\text{{obs}}}}$):** Fisher's two-sided test identifies any table whose point probability is less than or equal to that of the observed table ($P(k) \\le P({a})$). Summing the point probabilities of these flagged rows produces the exact two-sided $p$-value.
    * **Directional Tail Flags:** Shows whether each table contributes to the Right Tail ($k \\ge {a}$, more positive outcomes in {g1}) or Left Tail ($k \\le {a}$, fewer positive outcomes).
    """)
    dist_df = pd.DataFrame({
        "Cell_a_Count": [f"k = {tbl['k']}" for tbl in dist],
        "Contingency_Table_Configuration": [tbl["table_repr"] for tbl in dist],
        "Point_Probability_P": [f"{tbl['prob']:.5f}" for tbl in dist],
        "Cumulative_P": [f"{tbl['cum_prob']:.5f}" for tbl in dist],
        "In_Two_Sided_Rejection": ["Yes (P ≤ P_obs)" if tbl["is_two_sided_rejection"] else "No" for tbl in dist],
        "In_Right_Tail": ["Yes (k ≥ a)" if tbl["is_right_tail"] else "No" for tbl in dist],
        "In_Left_Tail": ["Yes (k ≤ a)" if tbl["is_left_tail"] else "No" for tbl in dist],
        "Status_Notes": [
            "⭐ OBSERVED TABLE IN EXPERIMENT" if tbl["is_observed"] else ("Contributes to Two-Sided p" if tbl["is_two_sided_rejection"] else "Center Distribution")
            for tbl in dist
        ]
    })
    render_table(dist_df, height=240)

    with st.expander("💡 Plain-English Guide: How Fisher's Exact Test Computes Probability Without Approximations", expanded=False):
        st.markdown("""
        **What makes Fisher's Exact Test "exact"?**
        * Tests like the Student's t-test or Pearson's Chi-Square assume that sample statistics follow smooth, continuous bell curves ($\chi^2$ or $t$ distributions).
        * When sample sizes are small or events are rare, this smooth approximation breaks down because counts are discrete integers, not smooth real numbers.
        * Fisher's Exact Test calculates the **exact combinatorial probability** of the data under the **Hypergeometric Distribution**:
          $$P(X = k) = \\frac{\\binom{R_1}{k} \\binom{R_2}{C_1 - k}}{\\binom{N}{C_1}}$$
        * **Fixed Marginals:** Imagine a jar containing $N$ total balls, where $C_1$ are red and $C_2$ are white. If you randomly draw a hand of $R_1$ balls (representing Group 1), the formula tells you the exact chance that your hand contains exactly $k$ red balls.
        * **How the Two-Sided p-value is formed:** 
          Fisher's test evaluates the point probability of your observed table ($P(a)$), and sums the probabilities of **all tables in the distribution whose probability is less than or equal to $P(a)$**. This guarantees that your false positive rate is strictly bounded at $\\alpha = 0.05$.
        """)

    # Table 4: Epidemiological Effect Sizes
    st.markdown("**4. Epidemiological Effect Sizes & Strength of Association**")
    st.markdown(f"""
    **What this table contains:**
    * Comprehensive epidemiological and clinical effect size metrics quantifying the **magnitude, direction, and practical strength** of the relationship between group and outcome, beyond just reporting whether $p < 0.05$.
    * Displays point estimates alongside 95% Confidence Intervals for relative metrics (Odds Ratio, Relative Risk) and absolute metrics (Risk Difference), plus standardized correlation indices (Phi, Cramér's $V$) and Number Needed to Treat (NNT).

    **How to interpret:**
    * **Exact Odds Ratio (OR) & Relative Risk (RR) [Null Benchmark = 1.0]:**
      * These are *ratio* metrics comparing likelihoods between groups.
      * $\\text{{OR}} = 1.0$ or $\\text{{RR}} = 1.0$ means identical risk/odds in both groups (no association).
      * Values $> 1.0$ indicate higher likelihood in {g1}; values $< 1.0$ indicate lower likelihood (protective effect).
      * If the 95% Confidence Interval excludes $1.0$, the effect is statistically significant at $\\alpha = 0.05$.
    * **Risk Difference (RD / ARR) [Null Benchmark = 0.0%]:** The absolute difference in percentage points between groups ({g1} rate minus {g2} rate). For example, a Risk Difference of $+20\\%$ means {g1} had an event rate 20 percentage points higher than {g2}.
    * **Number Needed to Treat / Harm (NNT / NNH):** Calculated as $1 / |\\text{{Risk Difference}}|$. Indicates the average number of individuals who would need to be in {g1} rather than {g2} to see one additional positive outcome (or prevent one negative outcome). Lower numbers indicate a stronger, higher-yield clinical or practical intervention.
    * **Phi Coefficient ($\\phi$) & Cramér's $V$ [Range: 0.0 to 1.0]:** Standardized measures of correlation for categorical data. For a $2 \\times 2$ table, they are identical: $\\approx 0.1$ indicates a weak association, $\\approx 0.3$ indicates a moderate association, and $\\ge 0.5$ indicates a strong association.
    """)
    effects_df = pd.DataFrame({
        "Effect_Metric": [
            "Exact Odds Ratio (OR)",
            "Sample Odds Ratio",
            "Relative Risk (RR) / Risk Ratio",
            "Risk Difference (RD) / ARR",
            "Number Needed to Treat / Harm (NNT)",
            "Phi Coefficient (φ)",
            "Cramér's V"
        ],
        "Point_Estimate": [
            f"{stats_results['odds_ratio_fisher']:.3f}",
            f"{stats_results['odds_ratio']:.3f}",
            f"{stats_results['risk_ratio']:.3f}",
            f"{stats_results['risk_diff']:+.3f} ({rd_pct:+.1f}%)",
            f"{stats_results['nnt']:.1f}" if stats_results["nnt"] else "N/A (RD = 0)",
            f"{stats_results['phi']:.3f}",
            f"{stats_results['cramers_v']:.3f}"
        ],
        "CI_95_Lower": [
            f"{or_ci[0]:.3f}",
            f"{or_ci[0]:.3f}",
            f"{stats_results['risk_ratio_ci'][0]:.3f}",
            f"{stats_results['risk_diff_ci'][0]:+.3f}",
            "N/A",
            "N/A",
            "N/A"
        ],
        "CI_95_Upper": [
            f"{or_ci[1]:.3f}",
            f"{or_ci[1]:.3f}",
            f"{stats_results['risk_ratio_ci'][1]:.3f}",
            f"{stats_results['risk_diff_ci'][1]:+.3f}",
            "N/A",
            "N/A",
            "N/A"
        ],
        "Clinical_Interpretation_Guideline": [
            "Exact conditional maximum likelihood odds ratio (1.0 = no effect).",
            "Standard cross-product odds ratio (a*d)/(b*c).",
            "Probability of outcome in Group 1 divided by Group 2.",
            "Absolute difference in event proportions between groups.",
            "Number of participants needed to see one additional positive outcome.",
            "Correlation coefficient for binary variables (0.1 = small, 0.3 = medium, 0.5 = large).",
            "Strength of association metric (identical to Phi for 2×2 tables)."
        ]
    })
    render_table(effects_df, height=240)

    # Table 5: Observational Dataset Preview
    st.markdown(f"**5. Subject-Level Observational Data Preview (N = {stats_results['n_total']} Participants)**")
    st.markdown("""
    **What this table contains:**
    * The raw, participant-level microdata records (one row per individual subject) that aggregate into the $2 \\times 2$ contingency table.
    * Displays participant identifiers (`Subject_ID`), assigned group condition (`Group`), and observed categorical status (`Outcome`).

    **How to interpret & utilize:**
    * **From Microdata to Cross-Tabulation:** In empirical research, data is recorded participant by participant. Cross-tabulating these individual rows produces the counts in cells $a, b, c, d$.
    * **Standard Data Structure:** Illustrates the canonical tidy/long data format required by statistical packages (such as Python `pandas`, R data frames, SPSS, or Stata) when running categorical association tests.
    * **Dataset Export:** Click the button below to download the complete simulated subject dataset as a CSV file for verification, teaching, or external statistical analysis.
    """)
    df_cases = stats_results["df_cases"]
    render_table(df_cases.head(10), height=180)
    st.download_button(
        "📥 Download Simulated Subject Dataset (CSV)",
        data=df_cases.to_csv(index=False),
        file_name="fisher_exact_subjects.csv",
        mime="text/csv",
        key="btn_download_fisher"
    )

    # 5. Core Assumption Evaluation Cards
    st.markdown("### 🛡️ Core Methodological Assumption Diagnostics")
    st.caption("Fisher's Exact Test is a distribution-free exact test designed for small sample sizes, sparse tables, and rare events.")

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        if asm["status"] == "pass":
            status_color = "#10b981"
            bg_color = "#f0fdf4"
            badge_symbol = "✅"
        elif asm["status"] == "warning":
            status_color = "#f59e0b"
            bg_color = "#fffbeb"
            badge_symbol = "🟡"
        else:
            status_color = "#ef4444"
            bg_color = "#fef2f2"
            badge_symbol = "❌"

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 6. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.warning(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # 7. Expandable Details for Alternative Tests
    with st.expander("🔎 View Alternative Test Details (Pearson Chi-Square & McNemar's Test)", expanded=False):
        st.markdown(f"""
        * **Pearson's Chi-Square Test of Independence (Large-Sample Alternative):**
          * Pearson omnibus test statistic: `χ² = {stats_results['chi2_uncorr']:.3f}` | **p-value = `{stats_results['p_chi2_uncorr']:.4f}`**
          * Yates' continuity corrected: `χ² = {stats_results['chi2_yates']:.3f}` | **p-value = `{stats_results['p_chi2_yates']:.4f}`**
          * Discrepancy from exact test: `{stats_results['p_discrepancy']:.4f}`.
          * When sample sizes are large and all expected cell frequencies are $\\ge 5.0$, Pearson Chi-Square matches Fisher's exact results closely while being computationally lighter.
        * **McNemar's Test (Paired Binary Data Alternative):**
          * If your observations were collected on the **same individuals across two time points** (e.g., Before vs After) or on **matched case-control pairs**, do not use Fisher's Exact test.
          * McNemar's test evaluates discordance exclusively across discordant cells ($b$ vs $c$): $\\chi^2_{{\\text{{McNemar}}}} = (|b - c| - 1)^2 / (b + c)$.
        """)


def render_mcnemar_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, binomial transition visualizations, and diagnostics for McNemar's Test."""
    st.markdown("### 🎯 Hypothesis Test Results: McNemar's Test for Paired Proportions")

    # Extract user-configured labels and paired 2x2 table parameters
    t1 = stats_results["time1_name"]
    t2 = stats_results["time2_name"]
    out_pos = stats_results["outcome_pos_name"]
    out_neg = stats_results["outcome_neg_name"]
    a = stats_results["a"]
    b = stats_results["b"]
    c = stats_results["c"]
    d = stats_results["d"]
    r1 = stats_results["r1"]
    r2 = stats_results["r2"]
    c1 = stats_results["c1"]
    c2 = stats_results["c2"]
    n_total = stats_results["n_total"]
    n_conc = stats_results["n_conc"]
    n_disc = stats_results["n_disc"]
    net_shift = stats_results["net_shift"]
    p1 = stats_results["p1"]
    p2 = stats_results["p2"]
    p1_pct = p1 * 100.0
    p2_pct = p2 * 100.0
    risk_diff = stats_results["risk_diff"]
    risk_diff_ci = stats_results["risk_diff_ci"]
    rd_pct = risk_diff * 100.0
    risk_ratio = stats_results["risk_ratio"]
    risk_ratio_ci = stats_results["risk_ratio_ci"]
    odds_ratio = stats_results["odds_ratio"]
    odds_ratio_ci = stats_results["odds_ratio_ci"]
    nnt = stats_results["nnt"]
    kappa = stats_results["kappa"]
    p_exact_two_sided = stats_results["p_exact_two_sided"]
    p_exact_greater = stats_results["p_exact_greater"]
    p_exact_less = stats_results["p_exact_less"]
    chi2_edwards = stats_results["chi2_edwards"]
    p_edwards = stats_results["p_edwards"]
    chi2_uncorr = stats_results["chi2_uncorr"]
    p_uncorr = stats_results["p_uncorr"]
    primary_p = stats_results["primary_p"]
    primary_method = stats_results["primary_method"]
    is_sig = primary_p < 0.05

    # Safe HTML-escaped strings for UI rendering
    t1_esc = html.escape(str(t1))
    t2_esc = html.escape(str(t2))
    out_pos_esc = html.escape(str(out_pos))
    out_neg_esc = html.escape(str(out_neg))

    conc_pct = (n_conc / n_total * 100.0) if n_total > 0 else 0.0
    disc_pct = (n_disc / n_total * 100.0) if n_total > 0 else 0.0

    # 1. User-Configured 2x2 Paired Transition Grid
    grid_html = (
        f'<div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">'
        f'<span style="font-weight: 700; font-size: 1.05rem; color: #1e3a8a;">🏷️ Observed 2×2 Paired Transition Matrix (Repeated Measures)</span>'
        f'<span style="font-size: 0.85rem; background: #eff6ff; color: #2563eb; padding: 0.25rem 0.65rem; border-radius: 6px; font-weight: 600;">Total N = {n_total} | Concordant: {n_conc} ({conc_pct:.1f}%) | Discordant: {n_disc} ({disc_pct:.1f}%)</span>'
        f'</div>'
        f'<div style="overflow-x: auto;">'
        f'<table style="width: 100%; border-collapse: separate; border-spacing: 8px; text-align: center; margin: 0 auto;">'
        f'<thead>'
        f'<tr>'
        f'<th rowspan="2" style="background: #f8fafc; color: #475569; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; vertical-align: middle; width: 22%;">Row Dimension<br><span style="font-size: 0.75rem; font-weight: normal; color: #64748b;">({t1_esc})</span></th>'
        f'<th colspan="2" style="background: #eff6ff; color: #1e40af; padding: 8px; border-radius: 8px; border: 1px solid #bfdbfe; font-size: 0.95rem; font-weight: 700;">Column Dimension: {t2_esc}</th>'
        f'<th rowspan="2" style="background: #f8fafc; color: #475569; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; vertical-align: middle; width: 18%;">Row Margins<br><span style="font-size: 0.75rem; font-weight: normal; color: #64748b;">({t1_esc} Totals)</span></th>'
        f'</tr>'
        f'<tr>'
        f'<th style="background: #f0fdf4; color: #166534; padding: 10px; border-radius: 8px; border: 1px solid #bbf7d0; font-size: 0.95rem; font-weight: 700; width: 30%;">{out_pos_esc}</th>'
        f'<th style="background: #f8fafc; color: #334155; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 0.95rem; font-weight: 700; width: 30%;">{out_neg_esc}</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>'
        f'<tr>'
        f'<th style="background: #eff6ff; color: #1e3a8a; padding: 12px; border-radius: 8px; border: 1px solid #bfdbfe; text-align: left; font-size: 0.95rem; font-weight: 700; vertical-align: middle;">{out_pos_esc}</th>'
        f'<td style="background: #ffffff; border: 2px solid #2563eb; border-radius: 8px; padding: 12px; vertical-align: middle; box-shadow: 0 1px 2px rgba(37,99,235,0.08);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;"><span style="font-size: 0.75rem; font-weight: 700; color: #2563eb; text-transform: uppercase; background: #eff6ff; padding: 1px 6px; border-radius: 4px;">Cell a (+/+)</span><span style="font-size: 0.75rem; color: #2563eb; font-weight: 600;">Concordant</span></div>'
        f'<div style="font-size: 1.75rem; font-weight: 800; color: #1e293b; line-height: 1.2;">{a}</div>'
        f'<div style="font-size: 0.8rem; color: #2563eb; margin-top: 2px;">Persistent {out_pos_esc} ({(a/n_total*100.0) if n_total > 0 else 0.0:.1f}%)</div>'
        f'</td>'
        f'<td style="background: #fffbeb; border: 2px solid #f59e0b; border-radius: 8px; padding: 12px; vertical-align: middle; box-shadow: 0 1px 2px rgba(245,158,11,0.08);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;"><span style="font-size: 0.75rem; font-weight: 700; color: #b45309; text-transform: uppercase; background: #fef3c7; padding: 1px 6px; border-radius: 4px;">Cell b (+/-)</span><span style="font-size: 0.75rem; color: #b45309; font-weight: 700;">⚡ Discordant</span></div>'
        f'<div style="font-size: 1.75rem; font-weight: 800; color: #92400e; line-height: 1.2;">{b}</div>'
        f'<div style="font-size: 0.8rem; color: #b45309; margin-top: 2px;">Switched to {out_neg_esc} ({(b/n_total*100.0) if n_total > 0 else 0.0:.1f}%)</div>'
        f'</td>'
        f'<td style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 12px; vertical-align: middle;">'
        f'<div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">R₁ = a + b</div>'
        f'<div style="font-size: 1.4rem; font-weight: 700; color: #1e293b; line-height: 1.2;">{r1}</div>'
        f'<div style="font-size: 0.75rem; color: #0284c7; font-weight: 600;">p₁ = {p1_pct:.1f}%</div>'
        f'</td>'
        f'</tr>'
        f'<tr>'
        f'<th style="background: #f8fafc; color: #334155; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: left; font-size: 0.95rem; font-weight: 700; vertical-align: middle;">{out_neg_esc}</th>'
        f'<td style="background: #f0fdf4; border: 2px solid #10b981; border-radius: 8px; padding: 12px; vertical-align: middle; box-shadow: 0 1px 2px rgba(16,185,129,0.08);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;"><span style="font-size: 0.75rem; font-weight: 700; color: #047857; text-transform: uppercase; background: #d1fae5; padding: 1px 6px; border-radius: 4px;">Cell c (-/+)</span><span style="font-size: 0.75rem; color: #047857; font-weight: 700;">⚡ Discordant</span></div>'
        f'<div style="font-size: 1.75rem; font-weight: 800; color: #065f46; line-height: 1.2;">{c}</div>'
        f'<div style="font-size: 0.8rem; color: #047857; margin-top: 2px;">Switched to {out_pos_esc} ({(c/n_total*100.0) if n_total > 0 else 0.0:.1f}%)</div>'
        f'</td>'
        f'<td style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; vertical-align: middle;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;"><span style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; background: #f1f5f9; padding: 1px 6px; border-radius: 4px;">Cell d (-/-)</span><span style="font-size: 0.75rem; color: #64748b;">Concordant</span></div>'
        f'<div style="font-size: 1.75rem; font-weight: 800; color: #1e293b; line-height: 1.2;">{d}</div>'
        f'<div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">Persistent {out_neg_esc} ({(d/n_total*100.0) if n_total > 0 else 0.0:.1f}%)</div>'
        f'</td>'
        f'<td style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 12px; vertical-align: middle;">'
        f'<div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">R₂ = c + d</div>'
        f'<div style="font-size: 1.4rem; font-weight: 700; color: #1e293b; line-height: 1.2;">{r2}</div>'
        f'<div style="font-size: 0.75rem; color: #94a3b8;">{(r2/n_total*100.0) if n_total > 0 else 0.0:.1f}% of N</div>'
        f'</td>'
        f'</tr>'
        f'<tr>'
        f'<th style="background: #f8fafc; color: #475569; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: left; font-size: 0.85rem; font-weight: 700; vertical-align: middle;">Column Margins<br><span style="font-size: 0.75rem; font-weight: normal; color: #64748b;">({t2_esc} Totals)</span></th>'
        f'<td style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 10px; vertical-align: middle;">'
        f'<div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">C₁ = a + c</div>'
        f'<div style="font-size: 1.4rem; font-weight: 700; color: #1e293b; line-height: 1.2;">{c1}</div>'
        f'<div style="font-size: 0.75rem; color: #0284c7; font-weight: 600;">p₂ = {p2_pct:.1f}%</div>'
        f'</td>'
        f'<td style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 10px; vertical-align: middle;">'
        f'<div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">C₂ = b + d</div>'
        f'<div style="font-size: 1.4rem; font-weight: 700; color: #1e293b; line-height: 1.2;">{c2}</div>'
        f'<div style="font-size: 0.75rem; color: #94a3b8;">{(c2/n_total*100.0) if n_total > 0 else 0.0:.1f}% of N</div>'
        f'</td>'
        f'<td style="background: #eff6ff; border: 2px solid #3b82f6; border-radius: 8px; padding: 10px; vertical-align: middle;">'
        f'<div style="font-size: 0.75rem; font-weight: 700; color: #1d4ed8; text-transform: uppercase;">Grand Total</div>'
        f'<div style="font-size: 1.6rem; font-weight: 800; color: #1e3a8a; line-height: 1.2;">N = {n_total}</div>'
        f'<div style="font-size: 0.75rem; color: #2563eb; font-weight: 600;">100.0%</div>'
        f'</td>'
        f'</tr>'
        f'</tbody>'
        f'</table>'
        f'</div>'
        f'<div style="font-size: 0.8rem; color: #64748b; margin-top: 0.6rem; padding-top: 0.5rem; border-top: 1px dashed #e2e8f0; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">'
        f'<span>💡 <em>Concordant pairs (Cells a & d) reflect stability and cancel out of McNemar test statistics.</em></span>'
        f'<span>⚡ <em>Only discordant pairs (Cells b & c) determine whether the proportion change is statistically significant.</em></span>'
        f'</div>'
        f'</div>'
    )
    render_html(grid_html)

    # 2. Key Statistical Metrics Grid
    st.markdown("##### 📊 Key Statistical Summary Metrics")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    mcol1.metric(
        "Grand Paired Size (N)",
        f"{n_total}",
        help="Total number of paired participants / matched pairs."
    )
    mcol1.metric(
        "Concordant Pairs (a + d)",
        f"{n_conc} ({conc_pct:.1f}%)",
        help="Subjects whose binary status remained identical across both measurement occasions."
    )

    mcol2.metric(
        "Discordant Pairs (b + c)",
        f"{n_disc} ({disc_pct:.1f}%)",
        help="Subjects who changed status between occasions. When b + c < 25, Exact Binomial Test is mandated."
    )
    mcol2.metric(
        "Net Directional Shift",
        f"{net_shift:+d} pairs",
        help=f"Net change towards {out_pos}: c (switched to {out_pos}) minus b (switched to {out_neg})."
    )

    sig_delta = f"p < 0.05 ({primary_method})" if is_sig else "p ≥ 0.05 (Not Significant)"
    mcol3.metric(
        "Exact Binomial p (Two-Sided)",
        f"{p_exact_two_sided:.4f}",
        delta=sig_delta,
        delta_color="normal" if is_sig else "off",
        help="Two-sided exact binomial p-value testing H0: P(switch) = 0.5 among discordant pairs."
    )
    mcol3.metric(
        "Edwards' Corrected χ²",
        f"{chi2_edwards:.3f}",
        delta=f"p = {p_edwards:.4f}",
        delta_color="off",
        help="Edwards continuity-corrected McNemar chi-square statistic: (|b - c| - 1)^2 / (b + c)."
    )

    mcol4.metric(
        "Marginal Risk Diff (Δp)",
        f"{rd_pct:+.1f}%",
        help=f"Absolute shift in proportion: {t2} ({p2_pct:.1f}%) minus {t1} ({p1_pct:.1f}%)."
    )
    mcol4.metric(
        "Conditional Odds Ratio",
        f"{odds_ratio:.2f}",
        help=f"Odds of switching to {out_pos} vs switching to {out_neg} (c / b). 95% CI: [{odds_ratio_ci[0]:.2f}, {odds_ratio_ci[1]:.2f}]."
    )

    # 3. Clinical & Practical Decision Interpretation in Plain English
    st.markdown("##### 🩺 Clinical & Statistical Decision Interpretation")

    if is_sig:
        st.success(f"""
        **Primary Finding: Statistically Significant Proportion Change via McNemar's Test ({primary_method}: p = {primary_p:.4f} < 0.05)**
        * **Hypotheses Tested:**
          * *Null Hypothesis ($H_0$):* Marginal homogeneity ($p_1 = p_2$). The probability of switching from positive to negative equals the probability of switching from negative to positive (discordant symmetry: $b = c$).
          * *Alternative Hypothesis ($H_1$):* Marginal proportions differ ($p_1 \\ne p_2$). Transitions between categories are significantly asymmetrical ($b \\ne c$).
        * **Observed Shift in Proportions:** The proportion of participants experiencing **{out_pos}** changed from **{p1_pct:.1f}%** at **{t1}** to **{p2_pct:.1f}%** at **{t2}**, representing an absolute marginal shift of **{rd_pct:+.1f}%** (95% CI: {risk_diff_ci[0]*100:+.1f}% to {risk_diff_ci[1]*100:+.1f}%).
        * **Discordant Dynamics:** Among the {n_disc} individuals who changed status, **{c}** switched to {out_pos} while **{b}** switched to {out_neg} (Net shift: **{net_shift:+d}** pairs).
        * **Conditional Odds Ratio:** `{odds_ratio:.2f}` (95% CI: `{odds_ratio_ci[0]:.2f}` – `{odds_ratio_ci[1]:.2f}`) — The odds of switching to {out_pos} were {odds_ratio:.2f} times as high as switching to {out_neg}.
        * **Methodological Choice:** 
          {"Because the discordant sample was small (b + c < 25), the Exact Binomial McNemar Test is the authoritative standard." if n_disc < 25 else "Because discordant pairs were adequate (b + c ≥ 25), Edwards' continuity-corrected chi-square confirms the finding (χ² = " + f"{chi2_edwards:.3f}, p = {p_edwards:.4f})."}
        """)
    else:
        st.info(f"""
        **Primary Finding: No Statistically Significant Proportion Change via McNemar's Test ({primary_method}: p = {primary_p:.4f} ≥ 0.05)**
        * **Hypotheses Tested:**
          * *Null Hypothesis ($H_0$):* Marginal proportions are identical ($p_1 = p_2$), meaning switches in either direction are symmetrical ($b = c$).
          * *Alternative Hypothesis ($H_1$):* Marginal proportions differ significantly ($p_1 \\ne p_2$).
        * **Observed Proportions:** The proportion of participants with **{out_pos}** was {p1_pct:.1f}% at **{t1}** vs. {p2_pct:.1f}% at **{t2}** (absolute difference: {rd_pct:+.1f}%, 95% CI: {risk_diff_ci[0]*100:+.1f}% to {risk_diff_ci[1]*100:+.1f}%).
        * **Discordant Breakdown:** {b} participants switched to {out_neg} and {c} participants switched to {out_pos}.
        * **Conditional Odds Ratio:** `{odds_ratio:.2f}` (95% CI: `{odds_ratio_ci[0]:.2f}` – `{odds_ratio_ci[1]:.2f}`). The confidence interval spans 1.0 (the neutral symmetry benchmark).
        * **Conclusion:** We fail to reject the null hypothesis of marginal homogeneity. The observed shift is fully consistent with random sampling fluctuation.
        """)

    # 3. Interactive Plotly Visualizations (3 Tabs)
    st.markdown("##### 📈 Visual Inspection & Paired Transition Diagnostics")
    tab_binom, tab_trans, tab_forest = st.tabs([
        "🎲 Discordant Cell Symmetry & Binomial Distribution",
        "🔄 Paired Transition & State Breakdown",
        "🌲 Marginal Effect Sizes & Concordance Forest Plot"
    ])

    with tab_binom:
        st.markdown(f"""
        **What does this distribution show?**
        * **The Binomial Probability Space for Discordant Pairs:** In McNemar's test, concordant pairs ($a, d$) provide evidence of stability but cancel out of test statistics because they did not change. Under the null hypothesis $H_0$ of symmetry, any individual who *did* change is equally likely to switch in either direction (probability $p = 0.5$).
        * **Horizontal Axis ($k$ = Count in Cell $c$):** The number of participants who switched from negative to positive. Each bar represents one possible outcome out of the total $b + c = {n_disc}$ discordant switchers.
        * **Vertical Axis ($P(X = k)$):** The exact mathematical probability of observing exactly $k$ positive switchers under pure chance symmetry ($p = 0.5$). The sum of all bars equals exactly $1.00000$ ($100\\%$).

        **How to interpret the color coding & decision rule:**
        * 🔵 **Blue Bar (Observed Experiment):** Highlights your actual data ($k = c = {c}$). The height represents the point probability of this exact split.
        * 🔴 **Red Bars (Rejection Region / Critical Tails):** Splits that are *as extreme or more extreme* than observed under two-sided symmetry ($k \\le \\min(b, c)$ or $k \\ge \\max(b, c)$).
        * ⚪ **Gray Bars (Non-Rejection Region):** Central, balanced splits expected under random chance symmetry.
        * **Calculating the Two-Sided p-value:** Summing the heights of the Blue bar and all Red bars gives the exact two-sided binomial $p$-value (**{p_exact_two_sided:.4f}**). If this sum is $< 0.05$, we reject symmetry and conclude that change between occasions was genuine.
        """)

        dist = stats_results["binom_distribution"]
        k_vals = [tbl["k"] for tbl in dist]
        probs = [tbl["prob"] for tbl in dist]
        colors = []
        bar_text = []

        for tbl in dist:
            if tbl["is_observed"]:
                colors.append("#2563eb")
                bar_text.append(f"Observed (k={tbl['k']})<br>P={tbl['prob']:.4f}")
            elif tbl["is_two_sided_rejection"]:
                colors.append("#ef4444")
                bar_text.append(f"P={tbl['prob']:.4f}<br>(Tail)")
            else:
                colors.append("#cbd5e1")
                bar_text.append(f"P={tbl['prob']:.4f}")

        fig_binom = go.Figure()
        fig_binom.add_trace(go.Bar(
            x=[f"k = {k}" for k in k_vals],
            y=probs,
            marker=dict(color=colors),
            text=bar_text,
            textposition="auto",
            hoverinfo="text",
            hovertext=[
                f"<b>Discordant split: {tbl['split_repr']}</b><br>"
                f"Positive Switchers (Cell c): {tbl['k']}<br>"
                f"Negative Switchers (Cell b): {tbl['b_k']}<br>"
                f"Point Probability P(X={tbl['k']}): {tbl['prob']:.5f}<br>"
                f"Cumulative P(X<={tbl['k']}): {tbl['cum_prob']:.5f}<br>"
                f"In Two-Sided Tail: {'Yes' if tbl['is_two_sided_rejection'] else 'No'}"
                for tbl in dist
            ]
        ))
        fig_binom.update_layout(
            title=f"Exact Binomial Probability Distribution Binomial(n={n_disc}, p=0.5) for Discordant Switchers<br><sup>Blue = Observed Split (c={c}), Red = Rejection Region (Two-Tailed Asymmetry), Gray = Expected Under Symmetry</sup>",
            xaxis_title=f"Number of Participants Switched to {out_pos} (Cell c)",
            yaxis_title="Exact Probability P(X = k)",
            template="plotly_white"
        )
        st.plotly_chart(fig_binom, use_container_width=True)

    with tab_trans:
        st.markdown(f"""
        **What does this chart compare?**
        * **Marginal Prevalence (% Positive):** Directly compares the overall proportion of positive participants at **{t1}** ($p_1 = {p1_pct:.1f}\\%$) versus **{t2}** ($p_2 = {p2_pct:.1f}\\%$).
        * **Four Clinical Transition Trajectories:** Details how each participant moved between categories across time:
          1. 🟦 **Persistent {out_pos} (Cell a = {a}):** Positive at both time points (Concordant).
          2. 🟧 **Switched to {out_neg} (Cell b = {b}):** Positive at {t1}, switched to negative at {t2} (Discordant relapse/loss).
          3. 🟩 **Switched to {out_pos} (Cell c = {c}):** Negative at {t1}, switched to positive at {t2} (Discordant gain/improvement).
          4. ⬜ **Persistent {out_neg} (Cell d = {d}):** Negative at both time points (Concordant).

        **How to interpret:**
        * **Net Impact:** The net gain in positive outcomes is Cell $c$ minus Cell $b$ ({net_shift:+d} participants).
        * **Stability vs Flux:** Concordant pairs ({n_conc} participants, {conc_pct:.1f}%) represent stability; discordant pairs ({n_disc} participants, {disc_pct:.1f}%) represent behavioral or clinical change.
        """)

        fig_trans = go.Figure()
        categories = [
            f"Persistent {out_pos} (Cell a: +/+)",
            f"Switched {out_neg} (Cell b: +/-)",
            f"Switched {out_pos} (Cell c: -/+)",
            f"Persistent {out_neg} (Cell d: -/-)"
        ]
        counts = [a, b, c, d]
        bar_colors = ["#2563eb", "#f59e0b", "#10b981", "#94a3b8"]

        fig_trans.add_trace(go.Bar(
            x=categories,
            y=counts,
            marker=dict(color=bar_colors),
            text=[f"{cnt} ({cnt/n_total*100.0 if n_total > 0 else 0:.1f}%)" for cnt in counts],
            textposition="auto"
        ))
        fig_trans.update_layout(
            title=f"Individual Subject Transition Breakdown Across {t1} and {t2} (N = {n_total})",
            xaxis_title="Paired Transition Category",
            yaxis_title="Participant Count",
            template="plotly_white"
        )
        st.plotly_chart(fig_trans, use_container_width=True)

    with tab_forest:
        st.markdown("""
        **What does a Forest Plot show for Paired Data?**
        * A **Forest Plot** illustrates the point estimates and 95% Confidence Intervals for effect sizes, revealing both the magnitude and precision of the paired change.

        **Key Metrics Explained (Assume No Prior Knowledge):**
        * **Marginal Risk Difference (Δp):** The net change in positive proportion ($p_2 - p_1$).
          * **Null benchmark = 0.0%:** Indicates identical proportions at both occasions.
          * **Positive value:** Net increase in positive status at Occasion 2.
          * **Negative value:** Net decrease in positive status at Occasion 2.
        * **Conditional Odds Ratio (OR = c / b):** The ratio of positive switchers to negative switchers.
          * **Null benchmark = 1.0:** Indicates symmetrical switching.
          * **OR > 1.0:** Higher odds of switching to positive than negative.
        * **Cohen's Kappa (κ):** Chance-corrected measure of stability / agreement across occasions.

        **How to interpret:**
        * If the 95% CI for Marginal Risk Difference excludes 0.0%, or the Odds Ratio excludes 1.0, the paired shift is statistically significant ($p < 0.05$).
        """)

        fig_forest = go.Figure()
        metrics = ["Conditional Odds Ratio (c / b)", "Marginal Relative Risk (p2 / p1)"]
        estimates = [odds_ratio, risk_ratio]
        ci_lows = [odds_ratio_ci[0], risk_ratio_ci[0]]
        ci_highs = [odds_ratio_ci[1], risk_ratio_ci[1]]

        fig_forest.add_trace(go.Scatter(
            y=metrics,
            x=estimates,
            mode="markers",
            marker=dict(size=14, color="#2563eb", symbol="diamond"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[h - e for h, e in zip(ci_highs, estimates)],
                arrayminus=[e - l for l, e in zip(ci_lows, estimates)],
                color="#1e293b",
                thickness=2,
                width=8
            ),
            name="Ratio Estimates (95% CI)"
        ))
        fig_forest.add_vline(x=1.0, line_dash="dash", line_color="#ef4444", annotation_text="No Effect / Symmetry (1.0)")
        fig_forest.update_layout(
            title="Comparative Paired Ratio Effect Sizes (Conditional Odds Ratio & Relative Risk with 95% CIs)",
            xaxis_title="Ratio Point Estimate (1.0 = Null Symmetry)",
            yaxis_title="Effect Metric",
            template="plotly_white"
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    # 4. Statistical Tables
    st.markdown("##### 📋 Detailed Statistical Summary Tables")

    # Table 1: Cross-Tabulation
    st.markdown(f"**1. 2×2 Paired Transition Cross-Tabulation ({t1} vs. {t2})**")
    st.markdown(f"""
    **What this table contains:**
    * Displays the complete repeated-measures transition matrix crossing {t1} by {t2}.
    * Shows **Concordant pairs** on the diagonal (Cells a and d) where subjects maintained their status, and **Discordant pairs** off the diagonal (Cells b and c) where subjects switched status.
    * Summarises marginal totals ($R_1, R_2, C_1, C_2$), marginal prevalence rates ($p_1, p_2$), and the total paired cohort size $N$.

    **How to interpret:**
    * **Marginal Shift:** Compare $p_1$ ({p1_pct:.1f}%) to $p_2$ ({p2_pct:.1f}%). The difference between these rates is the primary subject of McNemar's test.
    * **Discordant Imbalance:** Compare Cell $b$ ({b}) to Cell $c$ ({c}). McNemar's test specifically examines whether this off-diagonal difference is greater than what random chance symmetry would produce.
    """)
    crosstab_df = pd.DataFrame({
        f"{t1} Condition": [
            f"{out_pos} (Pre)",
            f"{out_neg} (Pre)",
            "Column Margins (Post Totals)"
        ],
        f"{out_pos} [Post]": [
            f"{stats_results['a']} (Persistent +)",
            f"{stats_results['c']} (Switched +)",
            f"{c1}"
        ],
        f"{out_neg} [Post]": [
            f"{stats_results['b']} (Switched -)",
            f"{stats_results['d']} (Persistent -)",
            f"{c2}"
        ],
        "Row_Total": [
            f"{r1}",
            f"{r2}",
            f"{n_total}"
        ],
        f"{out_pos} Rate (%)": [
            f"{p1_pct:.1f}% (Pre-Event Rate)",
            f"{(c/r2*100.0) if r2 > 0 else 0.0:.1f}% (Conversion Rate)",
            f"{p2_pct:.1f}% (Post-Event Rate)"
        ]
    })
    render_table(crosstab_df, height=180)

    # Table 2: McNemar Statistical Association Tests Comparison
    st.markdown("**2. McNemar Statistical Association Tests Comparison**")
    st.markdown(f"""
    **What this table contains:**
    * Comprehensive comparison across standard testing methods for paired binary data:
      1. **Exact Binomial McNemar Test (Two-Sided)** — The gold standard for any sample size.
      2. **Exact Binomial McNemar Test (Right-Tail / Greater: c > b)** — Testing directional improvement towards {out_pos}.
      3. **Exact Binomial McNemar Test (Left-Tail / Less: c < b)** — Testing directional deterioration towards {out_neg}.
      4. **Edwards' Continuity-Corrected Chi-Square** — The classic large-sample approximation with a 0.5 discrete continuity adjustment.
      5. **Uncorrected McNemar Chi-Square** — Unadjusted asymptotic approximation.

    **How to interpret:**
    * **The Discordant Threshold Rule (b + c ≥ 25):** When the total number of discordant pairs is less than 25, the continuous $\\chi^2$ approximation produces distorted $p$-values. The **Exact Binomial Test** is strictly required. When $b + c \\ge 25$, Edwards' corrected $\\chi^2$ aligns closely with the exact test.
    """)
    tests_df = pd.DataFrame({
        "Test_Method": [
            "McNemar Exact Binomial Test (Two-Sided)",
            "McNemar Exact Binomial Test (Right-Tail / Greater)",
            "McNemar Exact Binomial Test (Left-Tail / Less)",
            "Edwards' Continuity-Corrected Chi-Square",
            "Uncorrected McNemar Chi-Square"
        ],
        "Mathematical_Basis": [
            "Exact Binomial Distribution (p = 0.5)",
            "Exact Binomial Distribution (p = 0.5)",
            "Exact Binomial Distribution (p = 0.5)",
            "Asymptotic χ² with Discrete 1.0 Adjustment",
            "Asymptotic Continuous χ² Approximation"
        ],
        "Test_Statistic": [
            f"b = {b}, c = {c} (n_disc = {n_disc})",
            f"c = {c} positive switchers",
            f"c = {c} positive switchers",
            f"χ² = {chi2_edwards:.3f} (df=1)",
            f"χ² = {chi2_uncorr:.3f} (df=1)"
        ],
        "p_value": [
            f"{p_exact_two_sided:.4f}",
            f"{p_exact_greater:.4f}",
            f"{p_exact_less:.4f}",
            f"{p_edwards:.4f}",
            f"{p_uncorr:.4f}"
        ],
        "Significance_Alpha_05": [
            "Significant (p < 0.05)" if p_exact_two_sided < 0.05 else "Not Significant",
            "Significant (p < 0.05)" if p_exact_greater < 0.05 else "Not Significant",
            "Significant (p < 0.05)" if p_exact_less < 0.05 else "Not Significant",
            "Significant (p < 0.05)" if p_edwards < 0.05 else "Not Significant",
            "Significant (p < 0.05)" if p_uncorr < 0.05 else "Not Significant"
        ],
        "Reliability_Assessment": [
            "Gold standard for any sample size, mandatory when b + c < 25.",
            "Use when testing directional hypothesis that Occasion 2 > Occasion 1.",
            "Use when testing directional hypothesis that Occasion 2 < Occasion 1.",
            "Standard large-sample test; well-calibrated when b + c ≥ 25.",
            "Uncorrected approximation; tends to overstate significance on small counts."
        ]
    })
    render_table(tests_df, height=240)

    # Table 3: Complete Binomial Distribution Table
    st.markdown(f"**3. Complete Binomial Distribution of All {len(dist)} Possible Splits of {n_disc} Discordant Pairs**")
    st.markdown(f"""
    **What this table contains:**
    * An exhaustive, transparent catalog of all **{len(dist)} mathematically possible** split configurations of the {n_disc} discordant switchers under the null hypothesis of symmetry ($p = 0.5$).
    * For each possible count of positive switchers $k$, displays the split ($b$ vs $c$), point probability $P(X = k)$, cumulative probability, and whether it contributes to the two-sided rejection tail.

    **How to interpret:**
    * **Point Probability ($P$):** The exact binomial likelihood of that specific split occurring by chance alone. All probabilities sum to exactly $1.00000$ ($100\\%$).
    * **⭐ OBSERVED SPLIT:** Marks the actual data collected in your study ($c = {c}$).
    * **Two-Sided Rejection:** Flags splits that are as extreme or more extreme than observed. Summing these probabilities produces the exact two-sided $p$-value.
    """)
    dist_df = pd.DataFrame({
        "Positive_Switchers_c": [f"k = {tbl['k']}" for tbl in dist],
        "Discordant_Split_b_vs_c": [tbl["split_repr"] for tbl in dist],
        "Point_Probability_P": [f"{tbl['prob']:.5f}" for tbl in dist],
        "Cumulative_P": [f"{tbl['cum_prob']:.5f}" for tbl in dist],
        "In_Two_Sided_Rejection": ["Yes (Tail)" if tbl["is_two_sided_rejection"] else "No" for tbl in dist],
        "In_Right_Tail": ["Yes (k ≥ c)" if tbl["is_right_tail"] else "No" for tbl in dist],
        "In_Left_Tail": ["Yes (k ≤ c)" if tbl["is_left_tail"] else "No" for tbl in dist],
        "Status_Notes": [
            "⭐ OBSERVED SPLIT IN STUDY" if tbl["is_observed"] else ("Contributes to Two-Sided p" if tbl["is_two_sided_rejection"] else "Center Distribution")
            for tbl in dist
        ]
    })
    render_table(dist_df, height=240)

    # Table 4: Epidemiological Effect Sizes
    st.markdown("**4. Paired Effect Sizes & Concordance Indices**")
    st.markdown(f"""
    **What this table contains:**
    * Comprehensive effect size and agreement metrics designed specifically for paired binary observations.
    * Includes **Marginal Risk Difference (Δp)**, **Conditional Odds Ratio**, **Marginal Relative Risk**, **Number Needed to Treat (NNT)**, and **Cohen's Kappa (κ)** with 95% Confidence Intervals.

    **How to interpret:**
    * **Marginal Risk Difference (Δp) [Baseline = 0.0%]:** The absolute shift in positive rate ($p_2 - p_1$). A 95% CI excluding $0.0\\%$ confirms a significant population shift.
    * **Conditional Odds Ratio [Baseline = 1.0]:** Compares the odds of switching to positive versus switching to negative ($c / b$). Values $> 1.0$ favor Occasion 2.
    * **Cohen's Kappa (κ):** Chance-corrected test-retest consistency. $\\kappa \\ge 0.60$ indicates substantial agreement; lower values indicate active transitions.
    """)
    effects_df = pd.DataFrame({
        "Effect_Metric": [
            "Marginal Risk Difference (Δp)",
            "Conditional Odds Ratio (OR_paired)",
            "Marginal Relative Risk (RR)",
            "Number Needed to Treat / Benefit (NNT)",
            "Cohen's Kappa (κ)",
            "Concordance Rate (a + d) / N",
            "Discordance Rate (b + c) / N"
        ],
        "Point_Estimate": [
            f"{risk_diff:+.3f} ({rd_pct:+.1f}%)",
            f"{odds_ratio:.3f}",
            f"{risk_ratio:.3f}",
            f"{nnt:.1f}" if nnt else "N/A (Δp = 0)",
            f"{kappa:.3f}",
            f"{conc_pct:.1f}% ({n_conc}/{n_total})",
            f"{disc_pct:.1f}% ({n_disc}/{n_total})"
        ],
        "CI_95_Lower": [
            f"{risk_diff_ci[0]:+.3f}",
            f"{odds_ratio_ci[0]:.3f}",
            f"{risk_ratio_ci[0]:.3f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ],
        "CI_95_Upper": [
            f"{risk_diff_ci[1]:+.3f}",
            f"{odds_ratio_ci[1]:.3f}",
            f"{risk_ratio_ci[1]:.3f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ],
        "Clinical_Interpretation_Guideline": [
            "Absolute change in proportion of positive outcomes between occasions (0.0% = no shift).",
            "Odds of positive switch vs negative switch (c / b; 1.0 = equal switching).",
            "Ratio of post-event rate to pre-event rate (p2 / p1; 1.0 = equal prevalence).",
            "Number of paired participants needed to see one net additional positive outcome.",
            "Inter-occasion agreement corrected for chance (0.0 = chance, 1.0 = perfect agreement).",
            "Proportion of subjects whose binary status remained identical across occasions.",
            "Proportion of subjects who changed binary status across occasions."
        ]
    })
    render_table(effects_df, height=240)

    # Table 5: Observational Dataset Preview
    st.markdown(f"**5. Subject-Level Paired Observational Data Preview (N = {n_total} Paired Participants)**")
    st.markdown("""
    **What this table contains:**
    * Raw participant-level microdata records (one row per individual subject) showing `Subject_ID`, `Time_1_Status`, `Time_2_Status`, `Transition_Type`, and whether the status changed.
    * Allows inspection of individual transition patterns and downloading the full simulated dataset as CSV.

    **How to interpret & utilize:**
    * In paired binary research, data is collected with paired columns for each participant.
    * Cross-tabulating `Time_1_Status` by `Time_2_Status` produces the $2 \\times 2$ table ($a, b, c, d$).
    """)
    df_cases = stats_results["df_cases"]
    render_table(df_cases.head(10), height=180)
    st.download_button(
        "📥 Download Simulated Paired Subject Dataset (CSV)",
        data=df_cases.to_csv(index=False),
        file_name="mcnemar_paired_subjects.csv",
        mime="text/csv",
        key="btn_download_mcnemar"
    )

    # 5. Core Assumption Evaluation Cards
    st.markdown("### 🛡️ Core Methodological Assumption Diagnostics")
    st.caption("McNemar's Test is designed for paired or matched binary observations where subjects are tracked across two occasions.")

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        if asm["status"] == "pass":
            status_color = "#10b981"
            bg_color = "#f0fdf4"
            badge_symbol = "✅"
        elif asm["status"] == "warning":
            status_color = "#f59e0b"
            bg_color = "#fffbeb"
            badge_symbol = "🟡"
        else:
            status_color = "#ef4444"
            bg_color = "#fef2f2"
            badge_symbol = "❌"

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 6. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.info(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # 7. Expandable Details for Alternative Tests
    with st.expander("🔎 View Alternative Test Details (Cochran's Q Test & Independent Chi-Square)", expanded=False):
        st.markdown(f"""
        * **Cochran's Q Test (3+ Paired Binary Occasions Alternative):**
          * If your observations were collected on the same individuals across **3 or more time points** or conditions (e.g., Baseline, Week 4, Week 8), McNemar's test cannot evaluate the omnibus trend.
          * Cochran's Q test extends McNemar's test to $k \\ge 3$ repeated binary measures, producing an omnibus $\\chi^2$ statistic with $k - 1$ degrees of freedom.
        * **Stuart-Maxwell / Bowker's Test of Symmetry (3+ Categorical States):**
          * If your paired categorical variable has 3 or more nominal categories (e.g. Mild, Moderate, Severe), use Bowker's Test of Symmetry rather than McNemar's test.
        * **Pearson's Chi-Square Test of Independence (Independent Samples):**
          * If observations at Time 1 and Time 2 came from **two separate, unrelated cohorts of individuals**, you must use Pearson's Chi-Square (or Fisher's Exact Test) rather than McNemar's test. McNemar's test strictly requires paired/matched data.
        """)


def render_pearson_correlation_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics
    for Pearson Correlation Coefficient (r and R-squared).
    """
    st.markdown("### 🎯 Pearson Correlation & Linear Association Analysis")
    st.caption("Quantifies bivariate linear association strength (r), shared variance explained (R²), regression trajectory, and assumption diagnostics.")

    # 1. Metrics Grid (4 columns)
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    r_val = stats_results["pearson_r"]
    r_ci = stats_results["r_ci"]
    r2_pct = stats_results["r_squared_pct"]
    r2_val = stats_results["r_squared"]
    adj_r2 = stats_results["adj_r_squared"]
    t_stat = stats_results["t_stat"]
    df = stats_results["df"]
    p_val = stats_results["pearson_p"]
    slope = stats_results["slope"]
    intercept = stats_results["intercept"]
    rmse = stats_results["rmse"]

    mcol1.metric(
        "Pearson's r (Correlation)",
        f"{r_val:+.3f}",
        help="Measures the direction and strength of linear association (-1.0 to +1.0). Cohen: 0.10 small, 0.30 medium, 0.50 large."
    )
    mcol1.caption(f"95% CI: [{r_ci[0]:+.3f}, {r_ci[1]:+.3f}] (Fisher's z)")

    mcol2.metric(
        "R² (Variance Explained)",
        f"{r2_pct:.1f}%",
        help="Coefficient of determination: the percentage of total variance in the outcome variable explained by linear dependence on the predictor."
    )
    mcol2.caption(f"Adj R² = {adj_r2:.3f} | Unexplained = {stats_results['unexplained_pct']:.1f}%")

    p_str = f"< 0.001 ({p_val:.1e})" if p_val < 0.001 else f"= {p_val:.4f}"
    mcol3.metric(
        "Hypothesis Test (H₀: ρ = 0)",
        f"t({df}) = {t_stat:.3f}",
        help="t-statistic testing whether the population correlation coefficient significantly differs from 0."
    )
    mcol3.caption(f"p-value {p_str} ({'Significant' if p_val < 0.05 else 'Not Significant'})")

    mcol4.metric(
        "OLS Linear Fit",
        f"b₁ = {slope:+.3f}",
        help="Slope of the ordinary least squares regression line: expected change in Y per 1-unit increase in X."
    )
    mcol4.caption(f"Intercept b₀ = {intercept:.2f} | RMSE = {rmse:.2f}")

    # 2. Plain-English Statistical Interpretation Callout Box
    var_x = stats_results["var_x_name"]
    var_y = stats_results["var_y_name"]
    eff_label = stats_results["effect_label"]
    dir_label = stats_results["direction_label"]

    st.info(f"""
    **Plain-English Statistical Interpretation & Effect Size Summary:**
    * **Null Hypothesis ($H_0: \\rho = 0$):** Assumes there is zero linear relationship between **{var_x}** and **{var_y}** in the target population.
    * **Statistical Decision:** {'Reject the null hypothesis' if p_val < 0.05 else 'Fail to reject the null hypothesis'} ($p {p_str}$). There is a **statistically significant {dir_label.lower()} linear correlation** ($r = {r_val:+.3f}$, $95\\%$ CI: $[{r_ci[0]:+.3f}, {r_ci[1]:+.3f}]$).
    * **Effect Size Magnitude (Cohen's Standard):** An observed $|r| = {abs(r_val):.3f}$ represents a **{eff_label}** effect size ($|r| \\ge 0.10$ small, $\\ge 0.30$ medium, $\\ge 0.50$ large).
    * **Shared Variance ($R^2$):** **{r2_pct:.1f}%** of the total variability in {var_y} is accounted for by linear variation in {var_x}. The remaining **{stats_results['unexplained_pct']:.1f}%** represents individual variation and unexplained factors.
    * **Prediction Formula:** $\\hat{{Y}} = {intercept:.2f} {slope:+.3f} \\times X$
    """)

    # 3. Visual Inspection Tabs
    st.markdown("### 📈 Visual Diagnostics & Model Exploration")
    tab1, tab2, tab3 = st.tabs([
        "📊 Scatterplot & Regression Fit (95% CI & Prediction Bands)",
        "📉 Residual Diagnostics & Q-Q Plot",
        "🥧 Variance Partitioning (R² Decomposition)"
    ])

    with tab1:
        st.markdown(f"""
        **What this chart shows:**
        * **Observed Data Points (Scatter):** Each point represents an individual subject with predictor **{var_x}** on the horizontal axis and outcome **{var_y}** on the vertical axis.
        * **OLS Regression Line (Solid Line):** The line of best fit minimizing the sum of squared vertical residuals ($\hat{{Y}} = {intercept:.2f} {slope:+.3f} X$).
        * **95% Confidence Interval Band (Inner Shaded Area):** The zone where we are 95% confident the *true population mean response* $\\mu_{{Y|X}}$ lies. It is narrowest at the bivariate mean $(\\bar{{X}}, \\bar{{Y}})$ and flares outward toward the extremes.
        * **95% Prediction Interval Band (Outer Dashed Lines):** The wider zone where 95% of *individual future observations* are predicted to fall, accounting for both parameter uncertainty and residual individual variation ($s_{{\\text{{res}}}}$).
        """)

        fig_scatter = go.Figure()

        # Prediction Interval Band
        x_grid = stats_results["x_grid"]
        pi_upper = stats_results["pi_upper_band"]
        pi_lower = stats_results["pi_lower_band"]
        ci_upper = stats_results["ci_upper_band"]
        ci_lower = stats_results["ci_lower_band"]
        y_fit = stats_results["y_grid_fit"]

        fig_scatter.add_trace(go.Scatter(
            x=list(x_grid) + list(x_grid[::-1]),
            y=list(pi_upper) + list(pi_lower[::-1]),
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.08)',
            line=dict(color='rgba(239, 68, 68, 0.35)', dash='dash', width=1),
            hoverinfo='skip',
            name='95% Prediction Band (Individual Obs)',
            showlegend=True
        ))

        # Confidence Interval Band
        fig_scatter.add_trace(go.Scatter(
            x=list(x_grid) + list(x_grid[::-1]),
            y=list(ci_upper) + list(ci_lower[::-1]),
            fill='toself',
            fillcolor='rgba(37, 99, 235, 0.20)',
            line=dict(color='rgba(37, 99, 235, 0.5)', width=1),
            hoverinfo='skip',
            name='95% Confidence Band (Mean Response)',
            showlegend=True
        ))

        # Fitted Regression Line
        fig_scatter.add_trace(go.Scatter(
            x=x_grid,
            y=y_fit,
            mode='lines',
            line=dict(color='#1e3a8a', width=3),
            name=f'Fitted OLS Line (Slope = {slope:+.3f})',
            hovertemplate='X = %{x:.2f}<br>Predicted Y = %{y:.2f}<extra></extra>'
        ))

        # Scatter Points
        df_cases = stats_results["df_cases"]
        fig_scatter.add_trace(go.Scatter(
            x=df_cases["X_Value"],
            y=df_cases["Y_Value"],
            mode='markers',
            marker=dict(
                color=np.where(df_cases["Is_Influential"], '#ef4444', '#2563eb'),
                size=np.where(df_cases["Is_Influential"], 11, 8),
                opacity=0.85,
                line=dict(color='white', width=1)
            ),
            name='Observed Data Points',
            customdata=np.stack((
                df_cases["Subject_ID"],
                df_cases["Fitted_Y"],
                df_cases["Residual"],
                df_cases["Cooks_Distance"]
            ), axis=-1),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>' +
                f'{var_x}: ' + '%{x:.2f}<br>' +
                f'{var_y}: ' + '%{y:.2f}<br>' +
                'Fitted Ŷ: %{customdata[1]:.2f}<br>' +
                'Residual eᵢ: %{customdata[2]:+.2f}<br>' +
                'Cook\'s D: %{customdata[3]:.4f}<extra></extra>'
            )
        ))

        fig_scatter.update_layout(
            title=f"Bivariate Scatterplot with OLS Regression Fit (r = {r_val:+.3f}, R² = {r2_pct:.1f}%)",
            xaxis_title=var_x,
            yaxis_title=var_y,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            height=520
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab2:
        st.markdown("""
        **What these diagnostic plots show:**
        * **Residuals vs. Fitted Values (Left Plot):** A residual $e_i = Y_i - \\hat{Y}_i$ is the vertical prediction error. In a well-specified linear model, points should be evenly and randomly distributed around the horizontal zero line ($e_i = 0$) with uniform spread (homoscedasticity) and no curved patterns.
        * **Normal Q-Q Plot of Residuals (Right Plot):** Compares the quantiles of standardized residuals against theoretical quantiles from a standard normal distribution. If residuals follow a normal distribution, points lie directly along the 45-degree diagonal reference line.
        """)

        col_diag_a, col_diag_b = st.columns(2)

        with col_diag_a:
            fig_res = go.Figure()
            # Reference zero line
            fig_res.add_hline(y=0, line_dash="solid", line_color="#475569", line_width=1.5)
            # 2 SD threshold lines
            fig_res.add_hline(y=2.0 * rmse, line_dash="dot", line_color="#f59e0b", line_width=1, annotation_text="+2s", annotation_position="top left")
            fig_res.add_hline(y=-2.0 * rmse, line_dash="dot", line_color="#f59e0b", line_width=1, annotation_text="-2s", annotation_position="bottom left")

            fig_res.add_trace(go.Scatter(
                x=stats_results["y_fitted"],
                y=stats_results["residuals"],
                mode='markers',
                marker=dict(
                    color=np.where(df_cases["Is_Influential"], '#ef4444', '#0284c7'),
                    size=8,
                    opacity=0.8
                ),
                hovertemplate='Fitted Ŷ: %{x:.2f}<br>Residual eᵢ: %{y:+.2f}<extra></extra>',
                name='Residuals'
            ))

            fig_res.update_layout(
                title="Residuals vs. Fitted Values (Homoscedasticity Check)",
                xaxis_title="Fitted Value (Ŷ)",
                yaxis_title="Residual (eᵢ = Y - Ŷ)",
                template="plotly_white",
                height=420,
                margin=dict(l=30, r=30, t=50, b=30)
            )
            st.plotly_chart(fig_res, use_container_width=True)

        with col_diag_b:
            # Scipy Q-Q plot calculation
            sorted_res = np.sort(stats_results["std_residuals"])
            osm, osr = stats.probplot(sorted_res, dist="norm", fit=False)
            slope_qq, intercept_qq, low_slope, high_slope = stats.mstats.theilslopes(osr, osm)

            fig_qq = go.Figure()
            # 45 degree line
            x_line = np.linspace(min(osm), max(osm), 50)
            fig_qq.add_trace(go.Scatter(
                x=x_line,
                y=intercept_qq + slope_qq * x_line,
                mode='lines',
                line=dict(color='#dc2626', width=2),
                name='Theoretical Normal'
            ))

            fig_qq.add_trace(go.Scatter(
                x=osm,
                y=osr,
                mode='markers',
                marker=dict(color='#2563eb', size=8, opacity=0.8),
                name='Standardized Residuals',
                hovertemplate='Theoretical Quantile: %{x:.2f}<br>Standardized Residual: %{y:.2f}<extra></extra>'
            ))

            fig_qq.update_layout(
                title=f"Normal Q-Q Plot of Residuals (Shapiro-Wilk p = {stats_results['p_norm_res']:.4f})",
                xaxis_title="Theoretical Standard Normal Quantiles",
                yaxis_title="Standardized Residuals",
                template="plotly_white",
                height=420,
                margin=dict(l=30, r=30, t=50, b=30)
            )
            st.plotly_chart(fig_qq, use_container_width=True)

    with tab3:
        st.markdown(f"""
        **What this variance decomposition chart shows:**
        * **Total Sum of Squares ($SS_{{\\text{{tot}}}}$ = {stats_results['ss_tot']:.2f}):** The total baseline variability of outcome **{var_y}** around its sample mean $\\bar{{Y}}$.
        * **Regression Sum of Squares ($SS_{{\\text{{reg}}}}$ = {stats_results['ss_reg']:.2f}):** The portion of variability directly explained by linear variation in predictor **{var_x}** ($R^2 = {r2_pct:.1f}\\%$).
        * **Residual Sum of Squares ($SS_{{\\text{{res}}}}$ = {stats_results['ss_res']:.2f}):** The unexplained variation left over ($1 - R^2 = {stats_results['unexplained_pct']:.1f}\\%$).
        """)

        fig_donut = go.Figure(data=[go.Pie(
            labels=[
                f"Explained by {var_x} (R² = {r2_pct:.1f}%)",
                f"Unexplained Residual Variance ({stats_results['unexplained_pct']:.1f}%)"
            ],
            values=[stats_results["ss_reg"], stats_results["ss_res"]],
            hole=0.55,
            marker=dict(colors=['#2563eb', '#94a3b8']),
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )])

        fig_donut.update_layout(
            title=f"Variance Decomposition: SS_tot = SS_reg + SS_res (R² = {r2_pct:.1f}%)",
            template="plotly_white",
            height=440,
            margin=dict(l=30, r=30, t=60, b=30),
            annotations=[dict(
                text=f"<b>R² = {r2_pct:.1f}%</b><br><span style='font-size:0.8rem;color:#64748b;'>Shared Variance</span>",
                x=0.5, y=0.5, font_size=16, showarrow=False
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # 4. Detailed Statistical Summary Tables (5 tables with introductory text)
    st.markdown("### 📋 Detailed Statistical Summary Tables")

    # Table 1: Bivariate Descriptive Statistics
    st.markdown("**1. Bivariate Descriptive Statistics & Summary Parameters**")
    st.markdown(f"""
    **What this table contains:**
    * Sample size ($N$), sample means ($\\bar{{X}}, \\bar{{Y}}$), sample standard deviations ($s_X, s_Y$), sample variances ($s^2$), min, max, and sample covariance ($\\text{{Cov}}(X, Y)$).

    **How to interpret:**
    * **Covariance:** Quantifies the joint variability of $X$ and $Y$ in raw data units ($s_{{XY}} = \\sum (X_i - \\bar{{X}})(Y_i - \\bar{{Y}}) / (N - 1)$). Pearson's $r$ is simply standardized covariance divided by the product of both standard deviations ($r = s_{{XY}} / (s_X s_Y)$).
    """)
    df_desc = pd.DataFrame({
        "Statistic_Parameter": [
            "Sample Size (N)",
            "Sample Mean (x̄ / ȳ)",
            "Standard Deviation (s)",
            "Sample Variance (s²)",
            "Minimum Observed",
            "Maximum Observed",
            "Sample Covariance Cov(X, Y)"
        ],
        f"Predictor X: {var_x}": [
            f"{stats_results['n_pts']}",
            f"{stats_results['mean_x']:.3f}",
            f"{stats_results['std_x']:.3f}",
            f"{stats_results['var_x']:.3f}",
            f"{stats_results['min_x']:.3f}",
            f"{stats_results['max_x']:.3f}",
            f"{stats_results['covariance']:+.3f}"
        ],
        f"Outcome Y: {var_y}": [
            f"{stats_results['n_pts']}",
            f"{stats_results['mean_y']:.3f}",
            f"{stats_results['std_y']:.3f}",
            f"{stats_results['var_y']:.3f}",
            f"{stats_results['min_y']:.3f}",
            f"{stats_results['max_y']:.3f}",
            f"{stats_results['covariance']:+.3f}"
        ]
    })
    render_table(df_desc, height=200)

    # Table 2: Correlation & Hypothesis Testing Summary
    st.markdown("**2. Correlation & Hypothesis Testing Summary**")
    st.markdown(f"""
    **What this table contains:**
    * Exact Pearson correlation coefficient ($r$), degrees of freedom ($df = N - 2$), standard error of $r$, $t$-statistic, two-tailed $p$-value, Fisher's $z$ transformation, 95% Confidence Interval for $\\rho$, and comparison with non-parametric Spearman rank correlation ($\\rho$).

    **How to determine statistical significance:**
    * **Check the $p$-value (Threshold $\\alpha = 0.05$):**
      * If **$p < 0.05$** $\\rightarrow$ **Statistically Significant**. There is less than a 5% probability of observing a correlation this strong by random luck if no real relationship existed in the population. We reject the null hypothesis of zero correlation ($H_0: \\rho = 0$).
      * If **$p \\ge 0.05$** $\\rightarrow$ **Not Statistically Significant**. The association could easily arise by chance alone; you cannot claim a real linear relationship.
    * **Check the 95% Confidence Interval for $r$:**
      * If the 95% interval **does NOT include zero** (e.g. $[+0.65, +0.89]$ or $[-0.72, -0.31]$), the correlation is statistically significant.
      * If the 95% interval **spans across zero** (e.g. $[-0.15, +0.28]$), the correlation is not significant, because zero (no correlation) is a plausible true value.
    * **Fisher's $z$-Transformation:** Because the sampling distribution of $r$ is skewed when $|\\rho| > 0$, Fisher's $z = \\frac{{1}}{{2}} \\ln\\frac{{1+r}}{{1-r}}$ stabilizes variance to construct exact, asymmetric 95% confidence intervals.
    * **Spearman $\\rho$ Comparison:** If Pearson's $r$ and Spearman's $\\rho$ are nearly identical, the relationship is strictly linear. A large difference indicates non-linear curvature or outlier distortion.
    """)
    df_corr = pd.DataFrame({
        "Metric_Index": [
            "Pearson Correlation Coefficient (r)",
            "Degrees of Freedom (df = N - 2)",
            "Standard Error of r (SE_r)",
            "Student's t-Statistic",
            "Two-Tailed p-value (H₀: ρ = 0)",
            "Fisher's z-Transformation Score",
            "Fisher's z Standard Error (SE_z)",
            "Pearson r 95% Confidence Interval",
            "Spearman Rank Correlation (ρ)",
            "Spearman Two-Tailed p-value"
        ],
        "Calculated_Value": [
            f"{r_val:+.4f}",
            f"{df}",
            f"{stats_results['se_r']:.4f}",
            f"t = {t_stat:.4f}",
            f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
            f"z = {stats_results['fisher_z']:.4f}",
            f"{stats_results['se_z']:.4f}",
            f"[{r_ci[0]:+.4f}, {r_ci[1]:+.4f}]",
            f"{stats_results['spearman_rho']:+.4f}",
            f"{stats_results['spearman_p']:.4e}" if stats_results['spearman_p'] < 0.0001 else f"{stats_results['spearman_p']:.4f}"
        ],
        "Methodological_Interpretation": [
            f"Bivariate linear correlation ({eff_label.lower()} effect size).",
            f"Effective degrees of freedom for bivariate sample of N = {stats_results['n_pts']}.",
            "Sampling uncertainty in r: SE_r = sqrt((1 - r²) / (N - 2)).",
            "Standardized distance from null parameter 0 in standard error units.",
            f"{'Statistically significant (p < 0.05)' if p_val < 0.05 else 'Not statistically significant (p ≥ 0.05)'}.",
            "Variance-stabilizing transformation: z = 0.5 * ln((1 + r) / (1 - r)).",
            f"Theoretical standard error of Fisher's z: 1 / sqrt(N - 3) = {stats_results['se_z']:.4f}.",
            "Exact 95% CI back-transformed from Fisher's z limits.",
            "Non-parametric rank correlation (evaluates monotonic relationship).",
            "Significance of monotonic rank ordering."
        ]
    })
    render_table(df_corr, height=240)

    # Table 3: OLS Simple Linear Regression Estimates
    st.markdown("**3. Ordinary Least Squares (OLS) Linear Regression Parameter Estimates**")
    st.markdown(f"""
    **What this table contains:**
    * Unstandardized regression coefficients (Slope $b_1$ and Intercept $b_0$), standard errors, 95% Confidence Intervals, individual $t$-tests, model $F$-statistic, and Root Mean Squared Error (RMSE).

    **How to interpret:**
    * **Slope ($b_1$):** Measures the rate of change. For every 1-unit increase in **{var_x}**, **{var_y}** is expected to change by **{slope:+.3f}** units.
      * *Hypothesis Test ($H_0: \\beta_1 = 0$):* Tests whether the regression line is tilted or flat. **Crucially, the $p$-value for the Slope is mathematically identical to the $p$-value for Pearson's correlation ($r$)!** If the slope is significant, the correlation is significant.
    * **Intercept ($b_0$):** The predicted baseline value of {var_y} when {var_x} is exactly zero ($\hat{{Y}}_{{X=0}} = {intercept:.2f}$).
      * *Hypothesis Test ($H_0: \\beta_0 = 0$):* Tests whether the outcome at $X = 0$ is significantly different from zero (i.e. whether the line crosses through the origin $(0, 0)$).
    * **Can only one of these be significant, or must both/neither be?**
      * **Yes, they are completely independent tests!** One can be significant while the other is not:
        * *Slope Significant, Intercept NOT Significant:* Common when an outcome starts at zero (e.g. commission earnings vs items sold: selling 0 items earns \\$0, so the intercept is 0 and not significant, but each sale adds \\$50, so the slope is highly significant).
        * *Intercept Significant, Slope NOT Significant:* Common when an outcome has a non-zero baseline but the predictor has no effect (e.g. body temperature vs shoe size: shoe size has no effect so the line is flat and the slope is not significant, but body temperature is still 37°C when shoe size is zero, so the intercept is highly significant).
        * *Both Significant:* Baseline starting value is non-zero AND the predictor causes significant changes (e.g. base salary + annual raise).
        * *Rule of thumb:* To determine if the two variables are genuinely related, **look strictly at the Slope's $p$-value**. The Intercept's $p$-value only describes the baseline position on the vertical axis when $X = 0$.
    * **RMSE (Standard Error of the Estimate):** Average prediction error magnitude in original units of {var_y} ({rmse:.2f}).
    """)
    slope_ci = stats_results["slope_ci"]
    int_ci = stats_results["intercept_ci"]
    df_ols = pd.DataFrame({
        "Model_Parameter": [
            f"Slope (b₁: {var_x})",
            "Intercept (b₀: Constant)",
            "Model Overall F(1, df)",
            "Root Mean Squared Error (RMSE)"
        ],
        "Estimate": [
            f"{slope:+.4f}",
            f"{intercept:+.4f}",
            f"F(1, {df}) = {stats_results['f_stat']:.3f}",
            f"{rmse:.4f}"
        ],
        "Std_Error": [
            f"{stats_results['std_err_slope']:.4f}",
            f"{stats_results['se_intercept']:.4f}",
            "N/A",
            "N/A"
        ],
        "CI_95_Lower": [
            f"{slope_ci[0]:+.4f}",
            f"{int_ci[0]:+.4f}",
            "N/A",
            "N/A"
        ],
        "CI_95_Upper": [
            f"{slope_ci[1]:+.4f}",
            f"{int_ci[1]:+.4f}",
            "N/A",
            "N/A"
        ],
        "t_Statistic": [
            f"t = {t_stat:.3f}",
            f"t = {stats_results['t_intercept']:.3f}",
            "N/A",
            "N/A"
        ],
        "p_value": [
            f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
            f"{stats_results['p_intercept']:.4e}" if stats_results['p_intercept'] < 0.0001 else f"{stats_results['p_intercept']:.4f}",
            f"{stats_results['p_f']:.4e}" if stats_results['p_f'] < 0.0001 else f"{stats_results['p_f']:.4f}",
            "N/A"
        ]
    })
    render_table(df_ols, height=200)

    # Table 4: Variance Partitioning & Effect Size Indices
    st.markdown("**4. Variance Partitioning & Effect Size Indices (ANOVA Table)**")
    st.markdown(f"""
    **What this table contains:**
    * Complete ANOVA decomposition of sum of squares ($SS$), degrees of freedom ($df$), mean squares ($MS = SS / df$), $F$-statistic, and shared variance proportions.

    **How to interpret:**
    * **$R^2$ (Coefficient of Determination):** $SS_{{\\text{{reg}}}} / SS_{{\\text{{tot}}}} = {r2_pct:.1f}\\%$. In simple linear regression, $R^2$ is mathematically identical to the squared Pearson correlation ($r^2$).
    * **Adjusted $R^2$:** Corrects for sample size degrees of freedom ($1 - (1 - R^2)(N - 1)/(N - 2)$).
    """)
    df_var = pd.DataFrame({
        "Source_of_Variation": [
            f"Regression / Model ({var_x})",
            "Residual / Error",
            "Total Unadjusted"
        ],
        "Sum_of_Squares_SS": [
            f"{stats_results['ss_reg']:.3f}",
            f"{stats_results['ss_res']:.3f}",
            f"{stats_results['ss_tot']:.3f}"
        ],
        "Degrees_of_Freedom_df": [
            "1",
            f"{df}",
            f"{stats_results['n_pts'] - 1}"
        ],
        "Mean_Square_MS": [
            f"{stats_results['ss_reg']:.3f}",
            f"{stats_results['ss_res'] / df:.3f}",
            "N/A"
        ],
        "F_Statistic": [
            f"{stats_results['f_stat']:.3f} (p = {stats_results['p_f']:.4e})",
            "N/A",
            "N/A"
        ],
        "Percent_of_Total_SS": [
            f"{r2_pct:.1f}% (R²)",
            f"{stats_results['unexplained_pct']:.1f}% (Residual)",
            "100.0%"
        ]
    })
    render_table(df_var, height=200)

    # Table 5: Subject-Level Observational Data Preview & Influence Diagnostics
    st.markdown(f"**5. Subject-Level Observational Data Preview & Influence Diagnostics (N = {stats_results['n_pts']})**")
    st.markdown(f"""
    **What this table contains:**
    * Raw participant-level microdata records showing observed predictor $X_i$, observed outcome $Y_i$, fitted predicted value $\\hat{{Y}}_i$, raw residual $e_i = Y_i - \\hat{{Y}}_i$, standardized residual $r_i$, Cook's distance $D_i$, and influential observation flag.

    **How to interpret & utilize:**
    * **Standardized Residual ($r_i$):** Observations with $|r_i| > 3.0$ are potential vertical outliers.
    * **Cook's Distance ($D_i$):** Measures the influence of each single observation on all regression parameter estimates. A value exceeding the standard threshold $4 / N = {4.0 / stats_results['n_pts']:.4f}$ flags an influential observation that pulls the regression line toward itself.
    """)
    render_table(df_cases.head(10), height=240)
    st.download_button(
        "📥 Download Simulated Bivariate Dataset (CSV)",
        data=df_cases.to_csv(index=False),
        file_name="pearson_correlation_bivariate_data.csv",
        mime="text/csv",
        key="btn_download_pearson"
    )

    # 5. Core Methodological Assumption Diagnostics Cards (4 cards)
    st.markdown("### 🛡️ Core Methodological Assumption Diagnostics")
    st.caption("Pearson correlation requires linearity, bivariate normality, homoscedasticity, and absence of distorting outliers.")

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        if asm["status"] == "pass":
            status_color = "#10b981"
            bg_color = "#f0fdf4"
            badge_symbol = "✅"
        elif asm["status"] == "warning":
            status_color = "#f59e0b"
            bg_color = "#fffbeb"
            badge_symbol = "🟡"
        else:
            status_color = "#ef4444"
            bg_color = "#fef2f2"
            badge_symbol = "❌"

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 6. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.info(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # 7. Expandable Details for Alternative Tests
    with st.expander("🔎 View Alternative Test Details (Spearman Rank Correlation, Kendall's Tau, & Multiple Regression)", expanded=False):
        st.markdown(f"""
        * **Spearman Rank Correlation ($\\rho$ — Non-Linear Monotonic / Non-Normal Alternative):**
          * If the relationship between variables is curved (monotonic, e.g. exponential growth or diminishing returns), or if variables are ordinal or non-normally distributed, **Spearman's $\\rho$** should be used instead of Pearson's $r$.
          * Spearman's $\\rho$ ranks data from $1$ to $N$ before computing correlation, making it robust against non-linear scaling and extreme outliers.
        * **Kendall's Tau ($\\tau$ — Small Sample / Tied Rank Alternative):**
          * When sample sizes are small ($N < 30$) or when there are frequent tied values in ranks, **Kendall's $\\tau$** provides more accurate standard error estimates and tighter $p$-value calibration than Spearman's $\\rho$.
        * **Multiple Linear Regression (OLS — Multiple Predictors / Confounding Adjustment):**
          * Pearson correlation evaluates only simple bivariate association ($X$ vs $Y$). If you need to adjust for third-variable confounding, evaluate non-linear polynomial terms ($X^2$), or combine multiple simultaneous predictors to forecast $Y$, use **Multiple Linear Regression**.
        """)


def render_spearman_correlation_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics
    for Spearman Rank Correlation Coefficient (rho).
    """
    st.markdown("### 🎯 Spearman Rank Correlation (ρ) & Monotonic Association Analysis")
    st.caption("Quantifies monotonic association strength (ρ), rank displacement, tie corrections, and non-parametric hypothesis testing without assuming linearity or bivariate normality.")

    # 1. Metrics Grid (4 columns)
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    rho_val = stats_results["spearman_rho"]
    rho_ci = stats_results["rho_ci"]
    p_val = stats_results["spearman_p"]
    t_stat = stats_results["t_stat"]
    df = stats_results["df"]
    sum_d2 = stats_results["sum_d_squared"]
    ties_x = stats_results["ties_x_pct"]
    ties_y = stats_results["ties_y_pct"]
    kendall_tau = stats_results["kendall_tau"]
    kendall_p = stats_results["kendall_p"]
    pearson_r = stats_results["pearson_r"]
    eff_label = stats_results["effect_label"]
    dir_label = stats_results["direction_label"]
    var_x = stats_results["var_x_name"]
    var_y = stats_results["var_y_name"]

    mcol1.metric(
        "Spearman's ρ (Rank Correlation)",
        f"{rho_val:+.3f}",
        help="Monotonic association on ranked observations (-1.0 to +1.0). Cohen: 0.10 small, 0.30 medium, 0.50 large."
    )
    mcol1.caption(f"95% CI: [{rho_ci[0]:+.3f}, {rho_ci[1]:+.3f}] (Fisher's z)")

    mcol2.metric(
        "Squared Rank Diff (Σd²)",
        f"{int(round(sum_d2)):,}",
        help="Sum of squared rank differences between paired observations: Σ(Rank_X - Rank_Y)²."
    )
    mcol2.caption(f"Ties: {ties_x:.0f}% in X, {ties_y:.0f}% in Y ({'Corrected' if stats_results['has_ties'] else 'No Ties'})")

    p_str = f"< 0.001 ({p_val:.1e})" if p_val < 0.001 else f"= {p_val:.4f}"
    mcol3.metric(
        "Hypothesis Test (H₀: ρ = 0)",
        f"t({df}) = {t_stat:.3f}",
        help="Student's t-approximation testing whether the population rank correlation differs significantly from zero."
    )
    mcol3.caption(f"p-value {p_str} ({'Significant' if p_val < 0.05 else 'Not Significant'})")

    mcol4.metric(
        "Kendall's Tau-b (τ)",
        f"{kendall_tau:+.3f}",
        help="Concordance-based rank correlation. Better calibrated for small sample sizes or heavy tied ranks."
    )
    mcol4.caption(f"Linear Pearson r = {pearson_r:+.3f} (p = {stats_results['pearson_p']:.4f})")

    # 2. Plain-English Statistical Interpretation Callout Box
    st.info(f"""
    **Plain-English Statistical Interpretation & Non-Parametric Concept Summary:**
    * **What 'Monotonic' Means (vs. Linear):** A relationship is *linear* only if it forms a straight line. It is **monotonic** if, as **{var_x}** increases, **{var_y}** *consistently moves in one direction* (either up or down), even if the path curves, flattens out, or accelerates (e.g. diminishing returns, exponential growth, or logarithmic saturation).
    * **How Ranking Works:** Raw numbers are replaced by their **leaderboard rank positions** (1st, 2nd, 3rd, ..., {stats_results['n_pts']}th). This straightens out curved monotonic paths and prevents extreme outliers from distorting the results.
    * **Statistical Decision:** {'Reject the null hypothesis' if p_val < 0.05 else 'Fail to reject the null hypothesis'} ($p {p_str}$). There is a **statistically significant {dir_label.lower()} monotonic relationship** ($\\rho = {rho_val:+.3f}$, $95\\%$ CI: $[{rho_ci[0]:+.3f}, {rho_ci[1]:+.3f}]$).
    * **Effect Size Magnitude (Cohen's Standard):** Observed $|\\rho| = {abs(rho_val):.3f}$ represents a **{eff_label}** monotonic effect size ($|\\rho| \\ge 0.10$ small, $\\ge 0.30$ medium, $\\ge 0.50$ large).
    * **Spearman vs. Pearson Comparison:** Spearman's $|\\rho| = {abs(rho_val):.3f}$ vs. Pearson's $|r| = {abs(pearson_r):.3f}$. {'Spearman correlation detects a noticeably stronger relationship because it is not penalized by curvature or outliers.' if abs(rho_val) > abs(pearson_r) + 0.08 else 'Both rank and linear coefficients are closely aligned.'}
    """)

    # 3. Visual Inspection Tabs (3 tabs with introductory explanations)
    st.markdown("### 📈 Visual Diagnostics & Monotonic Exploration")
    tab1, tab2, tab3 = st.tabs([
        "📊 Raw Data Scatterplot & Monotonic Trend",
        "🎯 Rank-Rank Scatterplot (Linearized on Ranks)",
        "📉 Rank Differences (dᵢ = Rank_X - Rank_Y) Distribution"
    ])

    with tab1:
        st.markdown(f"""
        **What this chart shows:**
        * **Raw Data Points:** Each point displays the actual observed value of predictor **{var_x}** against outcome **{var_y}** in original measurement units.
        * **Monotonic Trend Curve (Solid Line):** Traces the general directional progression. Notice that even if the trajectory curves (e.g. exponential increase or diminishing returns), Spearman rank correlation evaluates how faithfully $Y$ moves in one direction as $X$ increases.
        """)

        fig_raw = go.Figure()
        df_cases = stats_results["df_cases"]

        # Smoothed trend curve
        fig_raw.add_trace(go.Scatter(
            x=stats_results["x_sorted"],
            y=stats_results["y_smooth"],
            mode='lines',
            line=dict(color='#2563eb', width=2.5),
            name='Monotonic Trend Trajectory',
            hoverinfo='skip'
        ))

        # Scatter points
        fig_raw.add_trace(go.Scatter(
            x=df_cases["Raw_X"],
            y=df_cases["Raw_Y"],
            mode='markers',
            marker=dict(
                color='#1e40af',
                size=8,
                opacity=0.85,
                line=dict(color='white', width=1)
            ),
            name='Observed Data Pairs',
            customdata=np.stack((
                df_cases["Subject_ID"],
                df_cases["Rank_X"],
                df_cases["Rank_Y"],
                df_cases["Rank_Diff_d"]
            ), axis=-1),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>' +
                f'{var_x}: ' + '%{x:.2f} (Rank %{customdata[1]:.1f})<br>' +
                f'{var_y}: ' + '%{y:.2f} (Rank %{customdata[2]:.1f})<br>' +
                'Rank Difference dᵢ: %{customdata[3]:+.1f}<extra></extra>'
            )
        ))

        fig_raw.update_layout(
            title=f"Raw Data Scatterplot with Monotonic Trajectory (Spearman ρ = {rho_val:+.3f})",
            xaxis_title=var_x,
            yaxis_title=var_y,
            template="plotly_white",
            height=480,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    with tab2:
        st.markdown(f"""
        **What this rank transformation chart shows:**
        * **The 'Straightening' Effect of Ranking:** Here, raw measurements are replaced entirely with their rank order positions from $1$ to ${stats_results['n_pts']}$.
        * **Rank OLS Regression Line (Solid Blue):** The best-fit line through the ranked data ($\hat{{R}}_Y = {stats_results['rank_intercept']:.1f} + {stats_results['rank_slope']:+.3f} R_X$).
        * **Perfect Concordance Reference (Diagonal Dotted Line):** Points lying directly on the diagonal have identical ranks in both variables ($d_i = 0$). Ranking successfully converts complex non-linear monotonic curves into a linear pattern!
        """)

        fig_rank = go.Figure()
        n = stats_results["n_pts"]

        # 45-degree reference diagonal
        fig_rank.add_trace(go.Scatter(
            x=[1, n],
            y=[1, n] if rho_val >= 0 else [n, 1],
            mode='lines',
            line=dict(color='#94a3b8', dash='dot', width=1.5),
            name='Perfect Monotonic Concordance (d = 0)',
            hoverinfo='skip'
        ))

        # OLS fit on ranks
        x_rank_grid = np.linspace(1, n, 50)
        y_rank_fit = stats_results["rank_intercept"] + stats_results["rank_slope"] * x_rank_grid
        fig_rank.add_trace(go.Scatter(
            x=x_rank_grid,
            y=y_rank_fit,
            mode='lines',
            line=dict(color='#2563eb', width=2.5),
            name=f'Rank OLS Fit (Slope = {stats_results["rank_slope"]:+.3f})'
        ))

        # Rank Scatter Points
        fig_rank.add_trace(go.Scatter(
            x=df_cases["Rank_X"],
            y=df_cases["Rank_Y"],
            mode='markers',
            marker=dict(
                color=np.where(abs(df_cases["Rank_Diff_d"]) > n * 0.25, '#ef4444', '#0284c7'),
                size=8,
                opacity=0.85,
                line=dict(color='white', width=1)
            ),
            name='Subject Rank Positions',
            customdata=np.stack((
                df_cases["Subject_ID"],
                df_cases["Raw_X"],
                df_cases["Raw_Y"],
                df_cases["Rank_Diff_d"],
                df_cases["d_Squared"]
            ), axis=-1),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>' +
                'Rank X: %{x:.1f} (Raw: %{customdata[1]:.2f})<br>' +
                'Rank Y: %{y:.1f} (Raw: %{customdata[2]:.2f})<br>' +
                'Rank Diff dᵢ: %{customdata[3]:+.1f}<br>' +
                'dᵢ²: %{customdata[4]:.1f}<extra></extra>'
            )
        ))

        fig_rank.update_layout(
            title=f"Rank-Rank Scatterplot: Transformed into Linear Space (ρ = {rho_val:+.3f})",
            xaxis_title=f"Rank of {var_x} (1 to {n})",
            yaxis_title=f"Rank of {var_y} (1 to {n})",
            template="plotly_white",
            height=480,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    with tab3:
        st.markdown(f"""
        **What this rank displacement chart shows:**
        * **Rank Difference ($d_i = \\text{{Rank}}(X)_i - \\text{{Rank}}(Y)_i$):** Represents how many leaderboard places an individual shifted between the two variables.
        * **Zero Centerline ($d_i = 0$):** Subjects whose rank in {var_x} matched their rank in {var_y} exactly.
        * **Displacement Magnitude:** When correlation is strong, displacements cluster tightly around zero. As rank correlation weakens, bars spread widely across positive and negative values.
        """)

        fig_diff = go.Figure()
        fig_diff.add_hline(y=0, line_dash="solid", line_color="#475569", line_width=1.5)

        # Bar chart of d_i
        colors_diff = np.where(df_cases["Rank_Diff_d"] >= 0, '#2563eb', '#f59e0b')
        fig_diff.add_trace(go.Bar(
            x=df_cases["Subject_ID"],
            y=df_cases["Rank_Diff_d"],
            marker=dict(color=colors_diff),
            customdata=np.stack((
                df_cases["Rank_X"],
                df_cases["Rank_Y"],
                df_cases["d_Squared"]
            ), axis=-1),
            hovertemplate=(
                '<b>%{x}</b><br>' +
                'Displacement dᵢ: %{y:+.1f}<br>' +
                'Rank X: %{customdata[0]:.1f} | Rank Y: %{customdata[1]:.1f}<br>' +
                'dᵢ²: %{customdata[2]:.1f}<extra></extra>'
            ),
            name='Rank Displacement (dᵢ)'
        ))

        fig_diff.update_layout(
            title=f"Individual Rank Displacements dᵢ = Rank_X - Rank_Y (Total Σd² = {int(round(sum_d2)):,})",
            xaxis_title="Individual Subjects",
            yaxis_title="Rank Displacement (dᵢ)",
            template="plotly_white",
            height=440,
            margin=dict(l=30, r=30, t=50, b=40)
        )
        st.plotly_chart(fig_diff, use_container_width=True)

    # 4. Detailed Statistical Summary Tables (5 tables with introductory explanations)
    st.markdown("### 📋 Detailed Statistical Summary Tables")

    # Table 1: Raw vs Ranked Descriptive Statistics
    st.markdown("**1. Raw Measurements vs. Ranked Descriptive Parameters**")
    st.markdown(f"""
    **What this table contains:**
    * Comprehensive descriptive parameters for both raw physical measurements and fractional average ranks.
    * For raw data: sample size ($N$), mean, standard deviation ($s$), median, interquartile range ($IQR$), min, and max.
    * For ranks: rank mean ($= (N + 1) / 2 = {(stats_results['n_pts'] + 1) / 2.0:.1f}$) and rank standard deviation.

    **How to interpret:**
    * In non-parametric rank testing, the **median** and **IQR** provide a much more truthful summary of central tendency and spread than the mean and standard deviation, because ranks are unaffected by skewness or extreme values.
    """)
    df_desc = pd.DataFrame({
        "Descriptive_Parameter": [
            "Sample Size (N pairs)",
            "Raw Sample Mean (x̄ / ȳ)",
            "Raw Standard Deviation (s)",
            "Raw Median (50th percentile)",
            "Raw Interquartile Range (IQR)",
            "Minimum Observed",
            "Maximum Observed",
            "Mean Fractional Rank",
            "Rank Standard Deviation"
        ],
        f"Predictor X: {var_x}": [
            f"{stats_results['n_pts']}",
            f"{stats_results['mean_x']:.2f}",
            f"{stats_results['std_x']:.2f}",
            f"{stats_results['median_x']:.2f}",
            f"{stats_results['iqr_x']:.2f}",
            f"{stats_results['min_x']:.2f}",
            f"{stats_results['max_x']:.2f}",
            f"{stats_results['mean_rank_x']:.2f}",
            f"{stats_results['std_rank_x']:.2f}"
        ],
        f"Outcome Y: {var_y}": [
            f"{stats_results['n_pts']}",
            f"{stats_results['mean_y']:.2f}",
            f"{stats_results['std_y']:.2f}",
            f"{stats_results['median_y']:.2f}",
            f"{stats_results['iqr_y']:.2f}",
            f"{stats_results['min_y']:.2f}",
            f"{stats_results['max_y']:.2f}",
            f"{stats_results['mean_rank_y']:.2f}",
            f"{stats_results['std_rank_y']:.2f}"
        ]
    })
    render_table(df_desc, height=220)

    # Table 2: Correlation & Hypothesis Testing Summary
    st.markdown("**2. Non-Parametric Correlation & Hypothesis Testing Summary**")
    st.markdown(f"""
    **What this table contains:**
    * Exact tie-corrected Spearman rank correlation ($\\rho$), degrees of freedom ($df = N - 2$), Student's $t$-statistic, two-tailed $p$-value, Fisher's $z$ transformation, asymmetric 95% Confidence Interval for $\\rho$, untied Spearman estimate, and comparative Kendall's Tau-b ($\\tau_b$).

    **How to determine statistical significance:**
    * **Check the $p$-value (Threshold $\\alpha = 0.05$):**
      * If **$p < 0.05$** $\\rightarrow$ **Statistically Significant**. There is less than a 5% probability of observing a monotonic association this strong by chance alone if the variables were independent in the population. We reject $H_0: \\rho = 0$.
      * If **$p \\ge 0.05$** $\\rightarrow$ **Not Statistically Significant**. The rank ordering could easily have arisen by chance.
    * **Check the 95% Confidence Interval for $\\rho$:**
      * If the 95% interval **does NOT include zero** (e.g. $[+0.74, +0.92]$ or $[-0.88, -0.65]$), the rank correlation is statistically significant.
      * If the 95% interval **spans across zero** (e.g. $[-0.18, +0.22]$), zero is a plausible population correlation and the relationship is not significant.
    """)
    df_corr = pd.DataFrame({
        "Metric_Index": [
            "Spearman Rank Correlation (ρ)",
            "Degrees of Freedom (df = N - 2)",
            "Student's t-Statistic",
            "Two-Tailed p-value (H₀: ρ = 0)",
            "Fisher's z-Transformation Score",
            "Fisher's z Standard Error (SE_z)",
            "Spearman ρ 95% Confidence Interval",
            "Untied Formula Estimate (1 - 6Σd²/N(N²-1))",
            "Kendall's Tau-b (τ_b)",
            "Kendall's Tau Two-Tailed p-value"
        ],
        "Calculated_Value": [
            f"{rho_val:+.4f}",
            f"{df}",
            f"t = {t_stat:.4f}",
            f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
            f"z = {stats_results['fisher_z']:.4f}",
            f"{stats_results['se_z']:.4f}",
            f"[{rho_ci[0]:+.4f}, {rho_ci[1]:+.4f}]",
            f"{stats_results['spearman_rho_untied']:+.4f}",
            f"{kendall_tau:+.4f}",
            f"{kendall_p:.4e}" if kendall_p < 0.0001 else f"{kendall_p:.4f}"
        ],
        "Methodological_Interpretation": [
            f"Non-parametric monotonic correlation ({eff_label.lower()} effect size).",
            f"Effective degrees of freedom for sample size of N = {stats_results['n_pts']}.",
            "Standardized t-statistic: t = rho * sqrt(df) / sqrt(1 - rho²).",
            f"{'Statistically significant (p < 0.05)' if p_val < 0.05 else 'Not statistically significant (p ≥ 0.05)'}.",
            "Variance-stabilizing transformation: z = 0.5 * ln((1 + rho) / (1 - rho)).",
            f"Standard error of Fisher's z for ranks: 1.06 / sqrt(N - 3) = {stats_results['se_z']:.4f}.",
            "Exact 95% confidence interval back-transformed from Fisher's z limits.",
            "Classic formula estimate (matches tie-corrected rho when no ties exist).",
            "Concordance-based rank index; robust when ties are frequent.",
            "Two-tailed significance of Kendall concordance."
        ]
    })
    render_table(df_corr, height=240)

    # Table 3: Tied Ranks & Frequency Adjustment Parameters
    st.markdown("**3. Tied Ranks & Frequency Adjustment Parameters**")
    st.markdown("""
    **What this table contains:**
    * Number and percentage of tied observations in predictor $X$ and outcome $Y$, along with mathematical tie correction factors $T_X = \\sum (t_i^3 - t_i) / 12$ and $T_Y = \\sum (u_j^3 - u_j) / 12$.

    **How to interpret:**
    * When two or more subjects share the exact same measurement, they receive the **average fractional rank** (e.g. if two subjects tie for 3rd and 4th place, both receive rank $3.5$).
    * Ties reduce the variance of the ranks. The tie correction factors $T_X$ and $T_Y$ adjust the denominator in Pearson's rank correlation formula to prevent over-inflating $\\rho$.
    """)
    df_ties = pd.DataFrame({
        "Tie_Evaluation_Metric": [
            "Tied Observations Count",
            "Tied Observations Percentage",
            "Tie Correction Factor (T = Σ(t³ - t)/12)",
            "Impact on Formula",
            "Recommended Handling"
        ],
        f"Predictor X: {var_x}": [
            f"{stats_results['ties_x_count']} tied points",
            f"{stats_results['ties_x_pct']:.1f}%",
            f"Tx = {stats_results['tie_correction_x']:.2f}",
            "Adjusts rank variance of X denominator",
            "Use Pearson-on-ranks tie formula (applied)" if stats_results['ties_x_count'] > 0 else "No adjustment needed"
        ],
        f"Outcome Y: {var_y}": [
            f"{stats_results['ties_y_count']} tied points",
            f"{stats_results['ties_y_pct']:.1f}%",
            f"Ty = {stats_results['tie_correction_y']:.2f}",
            "Adjusts rank variance of Y denominator",
            "Use Pearson-on-ranks tie formula (applied)" if stats_results['ties_y_count'] > 0 else "No adjustment needed"
        ]
    })
    render_table(df_ties, height=200)

    # Table 4: Monotonic vs. Linear Association Contrast Matrix
    st.markdown("**4. Monotonic vs. Linear Association Contrast Matrix (Spearman ρ vs. Pearson r)**")
    gap_val = abs(rho_val) - abs(pearson_r)
    st.markdown(f"""
    **What this table contains:**
    * Side-by-side contrast comparing non-parametric **Spearman's $\\rho$** against parametric **Pearson's $r$**.

    **How to interpret:**
    * **When $\\rho \\approx r$:** The relationship is approximately linear and contains no distorting leverage outliers. Both tests lead to identical scientific conclusions.
    * **When $|\\rho| \\gg |r|$ (Observed Gap = {gap_val:+.3f}):** The relationship is non-linear monotonic (curved) or contains influential outliers. Ranking protects the analysis and reveals the true underlying association that Pearson's straight-line assumption misses.
    """)
    df_compare = pd.DataFrame({
        "Method_Comparison_Attribute": [
            "Correlation Coefficient",
            "Association Form Evaluated",
            "Sensitivity to Non-Linear Curvature",
            "Sensitivity to Outliers / Skew",
            "Assumed Data Distribution",
            "Two-Tailed p-value"
        ],
        "Spearman Rank Correlation (ρ)": [
            f"ρ = {rho_val:+.4f}",
            "Monotonic (any consistent curve)",
            "Robust: Evaluates rank order preservation",
            "Highly resistant (outliers bounded to ranks)",
            "Non-parametric (no distribution assumed)",
            f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
        ],
        "Pearson Linear Correlation (r)": [
            f"r = {pearson_r:+.4f}",
            "Strictly Linear (straight line only)",
            "Penalized: Drops when relationship curves",
            "Highly sensitive (single outlier can distort r)",
            "Parametric (assumes bivariate normality)",
            f"{stats_results['pearson_p']:.4e}" if stats_results['pearson_p'] < 0.0001 else f"{stats_results['pearson_p']:.4f}"
        ]
    })
    render_table(df_compare, height=220)

    # Table 5: Subject-Level Observational Data Preview
    st.markdown(f"**5. Subject-Level Rank Decomposition & Microdata Preview (N = {stats_results['n_pts']})**")
    st.markdown("""
    **What this table contains:**
    * Raw participant-level records displaying observed measurements, calculated fractional average ranks, rank difference ($d_i = \\text{Rank}(X)_i - \\text{Rank}(Y)_i$), and squared difference ($d_i^2$).
    * Allows full auditability of the Spearman calculation and direct download of the simulated dataset as CSV.
    """)
    render_table(df_cases.head(10), height=240)
    st.download_button(
        "📥 Download Simulated Spearman Rank Dataset (CSV)",
        data=df_cases.to_csv(index=False),
        file_name="spearman_rank_correlation_data.csv",
        mime="text/csv",
        key="btn_download_spearman"
    )

    # 5. Core Methodological Assumption Diagnostics Cards (4 cards)
    st.markdown("### 🛡️ Core Methodological Assumption Diagnostics")
    st.caption("Spearman Rank Correlation relaxes normality and linearity, requiring monotonicity, ordinal/continuous scale, and independent observations.")

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        if asm["status"] == "pass":
            status_color = "#10b981"
            bg_color = "#f0fdf4"
            badge_symbol = "✅"
        elif asm["status"] == "warning":
            status_color = "#f59e0b"
            bg_color = "#fffbeb"
            badge_symbol = "🟡"
        else:
            status_color = "#ef4444"
            bg_color = "#fef2f2"
            badge_symbol = "❌"

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 6. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.info(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # 7. Expandable Details for Alternative Tests
    with st.expander("🔎 View Alternative Test Details (Kendall's Tau, Pearson Correlation, & Ordinal Regression)", expanded=False):
        st.markdown(f"""
        * **Kendall's Tau-b ($\\tau_b$ — Heavy Ties / Small Sample Non-Parametric Alternative):**
          * When sample size is small ($N < 30$) or when tied ranks exceed $25\\%$ of observations, **Kendall's $\\tau_b$** provides more accurate standard errors and tighter $p$-value calibration than Spearman's $\\rho$.
          * Kendall's $\\tau$ measures concordant vs. discordant pairs, representing the net probability that pairs are ordered consistently.
        * **Pearson Correlation Coefficient ($r$ — Strict Linear & Normal Parametric Alternative):**
          * If your bivariate data satisfies bivariate normality, displays an approximately straight-line scatter, and contains no extreme outliers, **Pearson's $r$** provides greater statistical power and directly quantifies variance explained ($R^2$).
        * **Ordinal Logistic Regression (Multi-Variable Predictors of Ranked Outcomes):**
          * Spearman correlation evaluates only simple bivariate association ($X$ vs $Y$). If you need to evaluate multiple simultaneous predictors or adjust for confounding variables when predicting an ordinal outcome, use **Ordinal Logistic Regression (Proportional Odds Model)**.
        """)


def render_linear_regression_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, visual plots, and assumption diagnostics
    for Ordinary Least Squares (OLS) Linear Regression (Simple & Multiple).
    """
    st.markdown("### 🎯 Ordinary Least Squares (OLS) Linear Regression Analysis")
    st.caption("Quantifies relationships between predictor variables and continuous outcomes, model fit (R²), omnibus ANOVA test, standardized regression coefficients (β*), and residual diagnostics.")

    # Extract results
    n_pts = stats_results["n_pts"]
    k_preds = stats_results["k_preds"]
    outcome_name = stats_results["outcome_name"]
    pred_names = stats_results["pred_names"]
    param_names = stats_results["param_names"]
    beta = stats_results["beta"]
    se_beta = stats_results["se_beta"]
    t_beta = stats_results["t_beta"]
    p_beta = stats_results["p_beta"]
    ci_lower = stats_results["ci_lower"]
    ci_upper = stats_results["ci_upper"]
    beta_std = stats_results["beta_standardized"]
    r2_pct = stats_results["r_squared_pct"]
    adj_r2 = stats_results["adj_r_squared"]
    unexplained_pct = stats_results["unexplained_pct"]
    rmse_val = stats_results["rmse"]
    f_stat = stats_results["f_stat"]
    df_reg = stats_results["df_reg"]
    df_res = stats_results["df_res"]
    p_f = stats_results["p_f"]
    dw_stat = stats_results["dw_stat"]
    vifs = stats_results["vifs"]
    tolerances = stats_results["tolerances"]
    df_cases = stats_results["df_cases"]

    # 1. Metrics Grid (4 columns)
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    mcol1.metric(
        "R² Variance Explained",
        f"{r2_pct:.1f}%",
        help="Percentage of total outcome variance accounted for by the regression model (R² = SS_reg / SS_tot)."
    )
    mcol1.caption(f"Adjusted R² = {adj_r2:.3f} | Unexplained = {unexplained_pct:.1f}%")

    p_f_str = f"< 0.001 ({p_f:.1e})" if p_f < 0.001 else f"= {p_f:.4f}"
    mcol2.metric(
        "Omnibus Model Fit (F-test)",
        f"F({df_reg}, {df_res}) = {f_stat:.2f}",
        help="Tests the omnibus null hypothesis that all predictor regression slopes are simultaneously zero."
    )
    mcol2.caption(f"p-value {p_f_str} ({'Significant' if p_f < 0.05 else 'Not Significant'})")

    outcome_short = outcome_name.split("(")[0].strip()
    mcol3.metric(
        "Standard Error of Estimate (RMSE)",
        f"{rmse_val:.2f}",
        help="Root Mean Squared Error: typical distance that observed data points fall from the fitted regression line/plane."
    )
    mcol3.caption(f"Residual Std Error in {outcome_short} units")

    if k_preds == 1:
        p_b1 = p_beta[1]
        p_b1_str = f"< 0.001 ({p_b1:.1e})" if p_b1 < 0.001 else f"= {p_b1:.4f}"
        mcol4.metric(
            "Predictor Slope (B₁)",
            f"{beta[1]:+.2f}",
            help=f"Unit change in {outcome_short} for each 1-unit increase in {pred_names[0]}."
        )
        mcol4.caption(f"**{pred_names[0]}** (p {p_b1_str})")
    else:
        # Identify predictor with largest absolute standardized beta
        std_betas_no_int = [abs(b) if not np.isnan(b) else 0.0 for b in beta_std[1:]]
        max_idx = int(np.argmax(std_betas_no_int)) + 1
        top_name = pred_names[max_idx - 1]
        top_beta_std = beta_std[max_idx]
        top_p = p_beta[max_idx]
        top_p_str = f"< 0.001 ({top_p:.1e})" if top_p < 0.001 else f"= {top_p:.4f}"
        mcol4.metric(
            "Top Predictor Impact",
            f"β* = {top_beta_std:+.2f}",
            help=f"Strongest relative predictor: {top_name}. Standardized beta = {top_beta_std:+.3f} (p {top_p_str})."
        )
        mcol4.caption(f"**{top_name}** (p {top_p_str}, B = {beta[max_idx]:+.2f})")

    # 2. Plain-English Statistical Interpretation Callout Box
    # Construct regression prediction formula
    terms = [f"{beta[0]:.2f}"]
    for j in range(k_preds):
        sign = "+" if beta[j + 1] >= 0 else "-"
        terms.append(f"{sign} {abs(beta[j + 1]):.2f} × ({pred_names[j]})")
    formula_str = " ".join(terms)

    st.info(f"""
    **Plain-English Statistical Interpretation & Regression Concept Summary:**
    * **What Ordinary Least Squares (OLS) Does:** OLS regression finds the unique mathematical line (or multi-dimensional plane) that minimizes the sum of squared vertical prediction errors (called residuals: $e_i = Y_i - \hat{{Y}}_i$). It gives the single best mathematical formula to predict **{outcome_name}**.
    * **Estimated Model Prediction Equation:**
      $$\hat{{Y}} = {formula_str}$$
    * **Model Fit & Explained Variance ($R^2 = {r2_pct:.1f}\\%$):** The predictor set accounts for **{r2_pct:.1f}%** of the total variance in **{outcome_name}**. The remaining **{unexplained_pct:.1f}%** is unexplained residual variation caused by unmeasured factors or natural randomness.
    * **Omnibus Model Fit ($F$-test) vs. Individual Predictor Tests ($t$-tests):**
      * **Omnibus $F$-test:** Evaluates whether the overall model explains significantly more variance than a simple baseline average. Here, $F({df_reg}, {df_res}) = {f_stat:.2f}$ with $p {p_f_str}$ ({'statistically significant' if p_f < 0.05 else 'not statistically significant'}).
      * **Individual $t$-tests:** Each predictor has an individual slope $B$ with its own $p$-value, showing whether that specific variable makes a unique, statistically significant contribution after holding all other predictors constant.
    """)

    # 3. Visual Diagnostics Tabs (3 tabs with introductory explanations)
    st.markdown("### 📈 Visual Diagnostics & Regression Exploration")
    tab1, tab2, tab3 = st.tabs([
        "📊 Model Predictions vs. Actual Observations",
        "🔍 4-Panel Residual Diagnostics (Gauss-Markov Assumptions)",
        "⚖️ Variance Partitioning (ANOVA SS) & Predictor Importance (β*)"
    ])

    with tab1:
        if k_preds == 1:
            st.markdown(f"""
            **What this bivariate scatter and regression line chart shows:**
            * **Observed Data Points (Blue Circles):** Each dot represents one subject's observed measurement of **{pred_names[0]}** ($X$) and **{outcome_name}** ($Y$).
            * **OLS Regression Line (Solid Blue Line):** The best-fit linear prediction ($\hat{{Y}} = {beta[0]:.2f} {beta[1]:+.2f}X$).
            * **95% Confidence Band (Shaded Ribbon):** The 95% confidence interval around the mean response $\hat{{Y}}$. Notice how the ribbon narrows near the average of $X$ where estimation precision is highest.
            * **How to interpret:** If points cluster tightly around the regression line, prediction error (RMSE) is small and $R^2$ is high. Wide scattering indicates that other unmeasured factors drive $Y$.
            """)

            fig_pred = go.Figure()
            x_raw = stats_results["X_data"][:, 0]
            y_raw = stats_results["y"]
            sort_idx = np.argsort(x_raw)
            x_sort = x_raw[sort_idx]
            y_sort = y_raw[sort_idx]
            y_fit_sort = stats_results["y_fitted"][sort_idx]

            # 95% Confidence Band for mean response
            x_mean = np.mean(x_raw)
            ss_x = np.sum((x_raw - x_mean) ** 2)
            t_crit = stats.t.ppf(0.975, df_res)
            se_line = rmse_val * np.sqrt(1.0 / n_pts + ((x_sort - x_mean) ** 2) / max(1e-12, ss_x))
            ci_upper_line = y_fit_sort + t_crit * se_line
            ci_lower_line = y_fit_sort - t_crit * se_line

            # Shaded ribbon
            fig_pred.add_trace(go.Scatter(
                x=np.concatenate([x_sort, x_sort[::-1]]),
                y=np.concatenate([ci_upper_line, ci_lower_line[::-1]]),
                fill='toself',
                fillcolor='rgba(37, 99, 235, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=True,
                name='95% Mean Confidence Band'
            ))

            # Fitted line
            fig_pred.add_trace(go.Scatter(
                x=x_sort,
                y=y_fit_sort,
                mode='lines',
                line=dict(color='#2563eb', width=3),
                name=f'OLS Line (Slope B₁ = {beta[1]:+.2f})'
            ))

            # Scatter points
            fig_pred.add_trace(go.Scatter(
                x=x_raw,
                y=y_raw,
                mode='markers',
                marker=dict(color='#1e40af', size=8, opacity=0.8, line=dict(color='white', width=1)),
                name='Observed Data',
                customdata=np.stack((
                    df_cases["Subject_ID"],
                    stats_results["y_fitted"],
                    stats_results["residuals"]
                ), axis=-1),
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>' +
                    f'{pred_names[0]}: %{{x:.2f}}<br>' +
                    f'Actual {outcome_name}: %{{y:.2f}}<br>' +
                    'Fitted Ŷ: %{customdata[1]:.2f}<br>' +
                    'Residual eᵢ: %{customdata[2]:+.2f}<extra></extra>'
                )
            ))

            fig_pred.update_layout(
                title=f"Bivariate OLS Regression Fit: R² = {r2_pct:.1f}%, RMSE = {rmse_val:.2f}",
                xaxis_title=pred_names[0],
                yaxis_title=outcome_name,
                template="plotly_white",
                height=480,
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pred, use_container_width=True)

        else:
            st.markdown(f"""
            **What this Actual vs. Fitted calibration chart shows:**
            * **Observed vs. Model-Predicted Calibration:** Plots the model's fitted predictions ($\hat{{Y}}$, X-axis) against the actual observed outcomes ($Y$, Y-axis) across all {n_pts} observations.
            * **Perfect Prediction Line (Dotted Diagonal $Y = \hat{{Y}}$):** Represents 100% accurate prediction. Points lying directly on this line have zero prediction error ($e_i = 0$).
            * **Vertical Error Segments (Light Red / Blue Lines):** The vertical distance from each point to the diagonal line represents the residual ($e_i = Y_i - \hat{{Y}}_i$) — the vertical distance minimized by the OLS algorithm.
            * **How to interpret:** The closer the scatter cloud clusters along the 45-degree diagonal, the higher the model's predictive accuracy ($R^2$). A wide spread indicates large residual variance.
            """)

            fig_pred = go.Figure()
            y_raw = stats_results["y"]
            y_fit = stats_results["y_fitted"]
            min_val = min(float(np.min(y_raw)), float(np.min(y_fit))) * 0.95
            max_val = max(float(np.max(y_raw)), float(np.max(y_fit))) * 1.05

            # 45-degree reference line
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='#64748b', dash='dash', width=2),
                name='Perfect Prediction Reference (Y = Ŷ)',
                hoverinfo='skip'
            ))

            # Vertical residual line segments
            for i in range(len(y_raw)):
                fig_pred.add_trace(go.Scatter(
                    x=[y_fit[i], y_fit[i]],
                    y=[y_fit[i], y_raw[i]],
                    mode='lines',
                    line=dict(color='rgba(239, 68, 68, 0.35)' if stats_results['residuals'][i] < 0 else 'rgba(37, 99, 235, 0.35)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Scatter points
            fig_pred.add_trace(go.Scatter(
                x=y_fit,
                y=y_raw,
                mode='markers',
                marker=dict(
                    color=np.where(df_cases["Is_Influential"], '#ef4444', '#1e40af'),
                    size=8,
                    opacity=0.85,
                    line=dict(color='white', width=1)
                ),
                name='Cases (Red = High Influence Cook\'s D)',
                customdata=np.stack((
                    df_cases["Subject_ID"],
                    stats_results["residuals"],
                    df_cases["Cooks_Distance_D"]
                ), axis=-1),
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>' +
                    'Fitted Ŷ: %{x:.2f}<br>' +
                    f'Actual {outcome_name}: %{{y:.2f}}<br>' +
                    'Residual eᵢ: %{customdata[1]:+.2f}<br>' +
                    'Cook\'s D: %{customdata[2]:.3f}<extra></extra>'
                )
            ))

            fig_pred.update_layout(
                title=f"Actual vs. Fitted Values Calibration Plot (R² = {r2_pct:.1f}%, F({df_reg}, {df_res}) = {f_stat:.2f})",
                xaxis_title=f"Model Predicted Outcome Ŷ ({outcome_name})",
                yaxis_title=f"Actual Observed Outcome Y ({outcome_name})",
                template="plotly_white",
                height=480,
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pred, use_container_width=True)

    with tab2:
        st.markdown("""
        **What the 4-Panel Residual Diagnostic Suite shows:**
        * **1. Residuals vs. Fitted (Top Left):** Evaluates **Linearity** and **Homoscedasticity**. Points should form an unstructured random horizontal cloud centered at $0$. A curved pattern indicates non-linearity; a funnel/wedge shape indicates heteroscedasticity.
        * **2. Normal Q-Q Plot (Top Right):** Evaluates **Residual Normality**. Residuals are plotted against theoretical standard normal quantiles. Points falling closely along the 45-degree diagonal confirm normal errors.
        * **3. Scale-Location Plot (Bottom Left):** Evaluates **Constant Residual Spread**. Plots $\sqrt{|\text{Standardized Residuals}|}$ against fitted values. The red horizontal line should stay flat. An upward or downward tilt indicates non-constant variance.
        * **4. Residuals vs. Leverage (Bottom Right):** Identifies **Influential Outliers**. Leverage ($h_{ii}$) measures how far an observation's predictor values lie from the center. Points in the far right beyond Cook's distance threshold ($4/N$) disproportionately pull the regression line.
        """)

        fig_diag = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "1. Residuals vs Fitted (Linearity / Equal Spread)",
                "2. Normal Q-Q (Error Normality)",
                "3. Scale-Location (Spread-Location)",
                "4. Residuals vs Leverage (Influential Points)"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        y_fit = stats_results["y_fitted"]
        res = stats_results["residuals"]
        std_res = stats_results["std_residuals"]
        hat = stats_results["hat_diags"]
        cooks_d = stats_results["cooks_d"]

        # Panel 1: Residuals vs Fitted
        fig_diag.add_trace(
            go.Scatter(
                x=y_fit, y=res, mode='markers',
                marker=dict(color='#2563eb', size=6, opacity=0.7),
                name='Residuals', showlegend=False,
                hovertemplate='Fitted: %{x:.2f}<br>Residual: %{y:+.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        fig_diag.add_hline(y=0, line_dash="dash", line_color="#475569", row=1, col=1)

        # Panel 2: Normal Q-Q Plot
        osm, osr = stats.probplot(res, dist="norm", fit=False)
        fig_diag.add_trace(
            go.Scatter(
                x=osm, y=osr, mode='markers',
                marker=dict(color='#0284c7', size=6, opacity=0.7),
                name='Residual Quantiles', showlegend=False,
                hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        slope_qq, intercept_qq, _, _ = stats.mstats.theilslopes(osr, osm)
        q_line_x = np.array([float(np.min(osm)), float(np.max(osm))])
        fig_diag.add_trace(
            go.Scatter(
                x=q_line_x, y=intercept_qq + slope_qq * q_line_x, mode='lines',
                line=dict(color='#ef4444', width=2),
                name='Normal Reference', showlegend=False
            ),
            row=1, col=2
        )

        # Panel 3: Scale-Location Plot
        sqrt_abs_res = np.sqrt(np.abs(std_res))
        fig_diag.add_trace(
            go.Scatter(
                x=y_fit, y=sqrt_abs_res, mode='markers',
                marker=dict(color='#7c3aed', size=6, opacity=0.7),
                name='Scale-Location', showlegend=False,
                hovertemplate='Fitted: %{x:.2f}<br>√|Std Res|: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        mean_sqrt_res = float(np.mean(sqrt_abs_res))
        fig_diag.add_hline(y=mean_sqrt_res, line_dash="dash", line_color="#ef4444", row=2, col=1)

        # Panel 4: Residuals vs Leverage
        fig_diag.add_trace(
            go.Scatter(
                x=hat, y=std_res, mode='markers',
                marker=dict(
                    color=np.where(cooks_d > stats_results['cooks_thresh'], '#ef4444', '#059669'),
                    size=7, opacity=0.8
                ),
                name='Leverage vs Residuals', showlegend=False,
                customdata=np.stack((df_cases["Subject_ID"], cooks_d), axis=-1),
                hovertemplate='<b>%{customdata[0]}</b><br>Leverage h: %{x:.3f}<br>Std Res: %{y:+.2f}<br>Cook\'s D: %{customdata[1]:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        fig_diag.add_hline(y=0, line_dash="solid", line_color="#94a3b8", row=2, col=2)
        fig_diag.add_hline(y=2, line_dash="dot", line_color="#cbd5e1", row=2, col=2)
        fig_diag.add_hline(y=-2, line_dash="dot", line_color="#cbd5e1", row=2, col=2)
        # Vertical leverage threshold line: 2 * (k + 1) / N
        lev_thresh = 2.0 * (k_preds + 1) / n_pts
        fig_diag.add_vline(x=lev_thresh, line_dash="dash", line_color="#ef4444", row=2, col=2)

        fig_diag.update_xaxes(title_text="Fitted Values (Ŷ)", row=1, col=1)
        fig_diag.update_yaxes(title_text="Residuals (eᵢ)", row=1, col=1)
        fig_diag.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig_diag.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        fig_diag.update_xaxes(title_text="Fitted Values (Ŷ)", row=2, col=1)
        fig_diag.update_yaxes(title_text="√|Standardized Residuals|", row=2, col=1)
        fig_diag.update_xaxes(title_text=f"Leverage hᵢᵢ (Thresh = {lev_thresh:.3f})", row=2, col=2)
        fig_diag.update_yaxes(title_text="Standardized Residuals (rᵢ)", row=2, col=2)

        fig_diag.update_layout(
            template="plotly_white",
            height=620,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_diag, use_container_width=True)

    with tab3:
        st.markdown(f"""
        **What these variance decomposition and predictor importance charts show:**
        * **ANOVA Sum of Squares Partitioning (Left Donut):** Illustrates how the total variation in outcome **{outcome_name}** ($SS_{{\\text{{tot}}}} = {stats_results['ss_tot']:,.1f}$) is split into variance explained by the regression model ($SS_{{\\text{{reg}}}} = {stats_results['ss_reg']:,.1f}$, **{r2_pct:.1f}%**) vs unexplained residual error ($SS_{{\\text{{res}}}} = {stats_results['ss_res']:,.1f}$, **{unexplained_pct:.1f}%**).
        * **Standardized Beta Weights ($\\beta^*$, Right Bar Chart):** Converts all predictors into standard deviation units so their impacts can be compared fairly. A 1 standard deviation increase in a predictor changes the outcome by $\\beta^*$ standard deviations. The predictor with the largest bar exerts the strongest relative influence in the model.
        """)

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            fig_ss = go.Figure(data=[go.Pie(
                labels=["Model Explained (SS_reg)", "Unexplained Residual (SS_res)"],
                values=[stats_results["ss_reg"], stats_results["ss_res"]],
                hole=0.55,
                marker=dict(colors=["#2563eb", "#cbd5e1"]),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Sum of Squares: %{value:,.1f}<br>Share: %{percent}<extra></extra>"
            )])
            fig_ss.update_layout(
                title=f"Total Variance Decomposition (SS_tot = {stats_results['ss_tot']:,.1f})",
                template="plotly_white",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_ss, use_container_width=True)

        with col_v2:
            preds_names_list = pred_names
            b_std_vals = beta_std[1:]
            p_vals_list = p_beta[1:]
            b_raw_list = beta[1:]

            bar_colors = ["#2563eb" if val >= 0 else "#f59e0b" for val in b_std_vals]
            fig_beta = go.Figure(go.Bar(
                y=preds_names_list,
                x=b_std_vals,
                orientation='h',
                marker=dict(color=bar_colors),
                customdata=np.stack((b_raw_list, p_vals_list), axis=-1),
                hovertemplate=(
                    '<b>%{y}</b><br>' +
                    'Standardized β*: %{x:+.3f}<br>' +
                    'Unstandardized B: %{customdata[0]:+.3f}<br>' +
                    'p-value: %{customdata[1]:.4f}<extra></extra>'
                )
            ))
            fig_beta.add_vline(x=0, line_dash="solid", line_color="#475569", line_width=1.5)
            fig_beta.update_layout(
                title="Relative Predictor Importance (Standardized β*)",
                xaxis_title="Standardized Beta Weight (β*)",
                yaxis_title="Predictor Variable",
                template="plotly_white",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_beta, use_container_width=True)

    # 4. Detailed Statistical Summary Tables (5 tables with introductory explanations)
    st.markdown("### 📋 Detailed Statistical Summary Tables")

    # Table 1: Overall Model Fit
    st.markdown("**1. Overall Model Fit & Summary Statistics**")
    st.markdown(f"""
    **What this table contains:**
    * Key summary metrics evaluating the overall predictive performance of the regression equation: $R^2$, Adjusted $R^2$, Omnibus Model $F$-statistic with degrees of freedom ($df$), $p$-value, Root Mean Squared Error (RMSE), and Durbin-Watson autocorrelation test.

    **How to interpret:**
    * **$R^2$ vs. Adjusted $R^2$:** While $R^2$ always increases when new predictors are added, **Adjusted $R^2$** penalizes useless variables. If Adjusted $R^2$ drops noticeably below $R^2$, some predictors are not contributing useful information.
    * **Omnibus Model Fit ($p$-value):** If $p < 0.05$, we reject $H_0$ and conclude that the model explains significantly more variance than a simple horizontal line through the mean $\\bar{{Y}}$.
    * **Durbin-Watson:** Values near $2.0$ confirm independent errors. Values below $1.5$ indicate positive autocorrelation; values above $2.5$ indicate negative autocorrelation.
    """)
    df_fit = pd.DataFrame({
        "Model_Fit_Metric": [
            "Coefficient of Determination (R²)",
            "Adjusted R² (Adj R²)",
            "Unexplained Variance Share",
            "Omnibus F-Statistic",
            "Regression Degrees of Freedom (df_reg)",
            "Residual Degrees of Freedom (df_res)",
            "Omnibus Model p-value",
            "Root Mean Squared Error (RMSE / Residual SE)",
            "Durbin-Watson Autocorrelation Statistic",
            "Sample Size (N observations)"
        ],
        "Calculated_Value": [
            f"{r2_pct:.2f}%",
            f"{adj_r2:.4f}",
            f"{unexplained_pct:.2f}%",
            f"F = {f_stat:.3f}",
            f"{df_reg}",
            f"{df_res}",
            f"{p_f:.4e}" if p_f < 0.0001 else f"{p_f:.4f}",
            f"{rmse_val:.3f}",
            f"{dw_stat:.3f}",
            f"{n_pts}"
        ],
        "Methodological_Interpretation": [
            f"Proportion of total outcome variance explained by the {k_preds} predictor(s).",
            "Model fit adjusted for the number of predictors; prevents artificial overfitting inflation.",
            "Remaining outcome variation attributed to unmeasured factors or residual error.",
            f"Ratio of explained variance per df to unexplained variance per df: MS_reg / MS_res.",
            f"Number of predictor parameters estimated in model (k = {k_preds}).",
            f"Error degrees of freedom: N - k - 1 = {n_pts} - {k_preds} - 1 = {df_res}.",
            f"{'Statistically significant omnibus fit (p < 0.05)' if p_f < 0.05 else 'Not statistically significant (p ≥ 0.05)'}.",
            f"Typical prediction distance in {outcome_short} units.",
            f"{'Acceptable error independence (~2.0)' if 1.5 <= dw_stat <= 2.5 else 'Autocorrelation warning'}.",
            f"Total number of subject observations in regression dataset."
        ]
    })
    render_table(df_fit, height=240)

    # Table 2: OLS Regression Coefficients Table
    st.markdown("**2. OLS Regression Coefficients & Predictor Significance**")
    st.markdown(r"""
    **What this table contains:**
    * Parameter-by-parameter breakdown of the model: Unstandardized Slope ($B$), Standard Error ($SE$), 95% Confidence Interval, Standardized Beta ($\beta^*$), $t$-statistic, two-tailed $p$-value, Variance Inflation Factor (VIF), and Tolerance.

    **How to interpret:**
    * **Unstandardized Coefficient ($B$):** The expected change in outcome $Y$ for a 1-unit increase in that predictor, *holding all other predictors constant*.
    * **How to determine statistical significance:**
      * **$p$-value < 0.05:** The predictor has a statistically significant independent relationship with the outcome.
      * **95% Confidence Interval excludes 0:** If the interval does not cross zero (e.g. $[+0.42, +1.85]$), the slope is statistically significant at $\alpha = 0.05$.
    * **Standardized Beta ($\beta^*$):** Shows relative impact independent of measurement units. A larger $|\beta^*|$ means a stronger relative effect.
    """)
    coef_rows = []
    for i, p_name in enumerate(param_names):
        b_val = beta[i]
        se_val = se_beta[i]
        ci_l = ci_lower[i]
        ci_u = ci_upper[i]
        b_std_val = beta_std[i]
        t_val = t_beta[i]
        p_val = p_beta[i]
        vif_val = vifs[i]
        tol_val = tolerances[i]

        p_str_val = f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
        b_std_str = f"{b_std_val:+.3f}" if not np.isnan(b_std_val) else "— (Baseline)"
        vif_str = f"{vif_val:.2f}" if not np.isnan(vif_val) else "—"
        tol_str = f"{tol_val:.3f}" if not np.isnan(tol_val) else "—"

        coef_rows.append({
            "Parameter": p_name,
            "Unstandardized_B": f"{b_val:+.3f}",
            "Std_Error": f"{se_val:.3f}",
            "CI_95_Lower": f"{ci_l:+.3f}",
            "CI_95_Upper": f"{ci_u:+.3f}",
            "Standardized_Beta": b_std_str,
            "t_Statistic": f"t = {t_val:.3f}",
            "p_value": p_str_val,
            "VIF": vif_str,
            "Tolerance": tol_str
        })
    df_coef_table = pd.DataFrame(coef_rows)
    render_table(df_coef_table, height=220)

    # Table 3: ANOVA Sum of Squares Decomposition
    st.markdown("**3. ANOVA Sum of Squares Decomposition Table**")
    st.markdown("""
    **What this table contains:**
    * Partitioning of total variance into Regression (Model) and Residual (Error) sources, showing Sum of Squares ($SS$), degrees of freedom ($df$), Mean Squares ($MS = SS/df$), $F$-statistic, and $p$-value.

    **How to interpret:**
    * **$SS_{{\\text{{Total}}}} = SS_{{\\text{{Regression}}}} + SS_{{\\text{{Residual}}}}$:** Fundamental Pythagorean identity of regression.
    * **$F$-statistic ($MS_{{\\text{{reg}}}} / MS_{{\\text{{res}}}}$):** Ratio of variance explained by the model to unexplained background noise variance. If $F$ is substantially greater than $1.0$ and $p < 0.05$, the model explains more variation than expected by random chance.
    """)
    df_anova_tbl = pd.DataFrame({
        "Source_of_Variation": ["Regression (Model)", "Residual (Error)", "Total Variance"],
        "Sum_of_Squares (SS)": [
            f"{stats_results['ss_reg']:,.2f}",
            f"{stats_results['ss_res']:,.2f}",
            f"{stats_results['ss_tot']:,.2f}"
        ],
        "df": [f"{df_reg}", f"{df_res}", f"{stats_results['df_tot']}"],
        "Mean_Square (MS)": [
            f"{stats_results['ms_reg']:,.2f}",
            f"{stats_results['ms_res']:,.2f}",
            "—"
        ],
        "F_Statistic": [f"F = {f_stat:.3f}", "—", "—"],
        "p_value": [
            f"{p_f:.4e}" if p_f < 0.0001 else f"{p_f:.4f}",
            "—",
            "—"
        ]
    })
    render_table(df_anova_tbl, height=180)

    # Table 4: Multicollinearity Diagnostics
    st.markdown("**4. Multicollinearity Diagnostics (VIF & Tolerance)**")
    st.markdown("""
    **What this table contains:**
    * Variance Inflation Factor (VIF) and Tolerance ($1 / \text{VIF}$) for each predictor variable, along with collinearity risk ratings and recommended action.

    **How to interpret:**
    * **What Multicollinearity is:** When two or more predictors are highly correlated with each other, they provide overlapping information. This destabilizes slope estimates ($B$), balloons standard errors ($SE$), and can make truly important variables appear statistically non-significant ($p > 0.05$).
    * **Thresholds:**
      * **VIF < 5.0 (Tolerance > 0.20):** Low collinearity. Predictors are sufficiently independent; estimates are reliable.
      * **5.0 ≤ VIF < 10.0 (Tolerance 0.10 to 0.20):** Moderate collinearity. Interpret coefficients with caution.
      * **VIF ≥ 10.0 (Tolerance < 0.10):** Severe multicollinearity. Consider removing one of the correlated predictors or using Ridge/Lasso regularization.
    """)
    if k_preds == 1:
        df_collin_tbl = pd.DataFrame({
            "Predictor": [pred_names[0]],
            "VIF": ["1.00"],
            "Tolerance": ["1.000"],
            "Collinearity_Status": ["✅ Low / Safe (Single Predictor)"],
            "Diagnostic_Guidance": ["Multicollinearity requires at least 2 predictors; not applicable to simple regression."]
        })
    else:
        collin_rows = []
        for j in range(k_preds):
            v_val = vifs[j + 1]
            t_val = tolerances[j + 1]
            if v_val < 5.0:
                stat_badge = "✅ Safe (VIF < 5)"
                guide_str = "Independent predictive contribution; stable standard error."
            elif v_val < 10.0:
                stat_badge = "🟡 Moderate Collinearity (5 ≤ VIF < 10)"
                guide_str = "Moderate predictor overlap; standard error slightly inflated."
            else:
                stat_badge = "❌ Severe Multicollinearity (VIF ≥ 10)"
                guide_str = "High redundancy; unstable coefficient. Consider removing or regularizing."
            collin_rows.append({
                "Predictor": pred_names[j],
                "VIF": f"{v_val:.2f}",
                "Tolerance": f"{t_val:.3f}",
                "Collinearity_Status": stat_badge,
                "Diagnostic_Guidance": guide_str
            })
        df_collin_tbl = pd.DataFrame(collin_rows)
    render_table(df_collin_tbl, height=200)

    # Table 5: Subject-Level Observational Data Preview
    st.markdown(f"**5. Subject-Level Observational Data & Influence Microdata (N = {n_pts})**")
    st.markdown(f"""
    **What this table contains:**
    * Case-by-case microdata for the first 10 subjects, showing observed predictors, actual outcome $Y$, model-predicted $\hat{{Y}}$, raw residual ($e_i = Y_i - \hat{{Y}}_i$), standardized residual ($r_i$), leverage ($h_{{ii}}$), Cook's Distance ($D_i$), and influence status.
    * Allows complete auditing of calculations and download of the synthetic dataset for Excel / Python practice.
    """)
    render_table(df_cases.head(10), height=240)
    st.download_button(
        "📥 Download Simulated Regression Dataset (CSV)",
        data=df_cases.to_csv(index=False),
        file_name="linear_regression_simulation_data.csv",
        mime="text/csv",
        key="btn_download_linear_regression"
    )

    # 5. Core Methodological Assumption Diagnostics Cards (6 cards in 2 columns)
    st.markdown("### 🛡️ Core Methodological Assumption Diagnostics")
    st.caption("Ordinary Least Squares (OLS) guarantees Best Linear Unbiased Estimates (BLUE) under the Gauss-Markov theorem when these 6 assumptions hold.")

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        if asm["status"] == "pass":
            status_color = "#10b981"
            bg_color = "#f0fdf4"
            badge_symbol = "✅"
        elif asm["status"] == "warning":
            status_color = "#f59e0b"
            bg_color = "#fffbeb"
            badge_symbol = "🟡"
        else:
            status_color = "#ef4444"
            bg_color = "#fef2f2"
            badge_symbol = "❌"

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 6. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.info(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # 7. Expandable Details for Alternative Tests
    with st.expander("🔎 View Alternative Test Details (Polynomial Regression, Robust Regression, & Ridge/Lasso)", expanded=False):
        st.markdown(f"""
        * **Polynomial Regression & Generalized Additive Models (GAM — When Linearity Fails):**
          * If residuals display a curved U-shape or inverted-U pattern against fitted values, the relationship between $X$ and $Y$ is non-linear.
          * Adding quadratic terms ($X^2$) or using **Polynomial Regression / GAMs** allows the model to capture non-linear bends without violating model assumptions.
        * **Robust Regression & Huber M-Estimators (When Outliers or High Leverage Distort Fit):**
          * OLS squares all errors, giving massive leverage to distant outliers. If Cook's distance identifies high-influence observations, **Robust Regression (Huber or Tukey Bisquare loss)** down-weights extreme observations without deleting them from the sample.
        * **Heteroscedasticity-Consistent Standard Errors (Huber-White / HC3 — When Equal Variance Fails):**
          * If Breusch-Pagan detects heteroscedasticity (funnel-shaped variance), OLS slope estimates ($B$) remain unbiased, but ordinary standard errors and $p$-values become invalid.
          * Applying **HC3 Robust Standard Errors** or Weighted Least Squares (WLS) restores accurate hypothesis testing.
        * **Ridge, Lasso & Elastic Net Regularization (When Severe Multicollinearity Exists):**
          * When predictors are heavily intercorrelated (VIF $\ge 10$), ordinary OLS produces wild, unstable coefficient swings.
          * **Ridge regression** shrinks collinear coefficients toward zero, and **Lasso regression** automatically selects the most important predictors by shrinking redundant slopes to exactly zero.
        """)


def render_binary_logistic_regression_demo(demo_df: pd.DataFrame, stats_results: Dict[str, Any]):
    """Renders comprehensive demonstration, interactive Plotly visualizations,
    Odds Ratio tables, confusion matrix diagnostics, and assumption checks for Binary Logistic Regression.
    """
    st.markdown("### 🎯 Binary Logistic Regression: Model Parameters, Odds Ratios & Classification Diagnostics")

    # 1. Summary Metrics Bar (6 columns)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Sample Size (N)", f"{stats_results['n_pts']}", help="Total number of observations in sample.")
    col2.metric("Events & EPV", f"{stats_results['n_events']} (EPV: {stats_results['epv']:.1f})", help="Effective events in the minority class divided by number of predictors. Benchmark: EPV ≥ 10.")
    col3.metric("McFadden's Pseudo-R²", f"{stats_results['pseudo_r2_mcfadden']:.4f}", help="Log-likelihood improvement over intercept-only model. Values between 0.20 and 0.40 indicate excellent model fit.")
    col4.metric("ROC-AUC Score", f"{stats_results['auc']:.3f}", help="Area under the Receiver Operating Characteristic curve (0.5 = Chance, 1.0 = Perfect).")
    col5.metric("Likelihood Ratio χ²", f"{stats_results['lr_stat']:.2f}", help=f"Omnibus model fit statistic comparing full model to intercept-only null model (df = {stats_results['lr_df']}).")
    p_omni_str = "< 0.001" if stats_results['p_omnibus'] < 0.001 else f"{stats_results['p_omnibus']:.4f}"
    col6.metric("Omnibus p-value", p_omni_str, help="Omnibus test of model significance. p < 0.05 indicates model predictors collectively explain significant outcome variation.")

    st.markdown("---")

    # 2. Dual Plotly Visualizations (Sigmoid S-Curve & ROC Curve)
    vcol1, vcol2 = st.columns(2)

    with vcol1:
        st.markdown(f"**1. Sigmoid Probability S-Curve vs {stats_results['primary_x_name']}**")
        fig_curve = go.Figure()

        # Generate smooth logistic S-curve
        x_min = float(min(stats_results['primary_x_vals']))
        x_max = float(max(stats_results['primary_x_vals']))
        x_grid = np.linspace(x_min, x_max, 150)

        # Baseline linear predictor holding other predictors at sample mean
        z_grid = stats_results['beta'][0] + stats_results['beta'][1] * x_grid
        if stats_results['k_preds'] > 1:
            for j in range(2, stats_results['k_preds'] + 1):
                col_k = stats_results['col_keys'][j - 1]
                mean_xj = float(np.mean(demo_df[col_k]))
                z_grid += stats_results['beta'][j] * mean_xj

        p_grid = 1.0 / (1.0 + np.exp(-np.clip(z_grid, -25.0, 25.0)))

        fig_curve.add_trace(go.Scatter(
            x=x_grid,
            y=p_grid,
            mode="lines",
            name="Logistic Sigmoid p(Y=1)",
            line=dict(color="#2563eb", width=3)
        ))

        # Classification threshold 0.50
        fig_curve.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="Cutoff (p = 0.50)",
            annotation_position="bottom right"
        )

        # Observed individual points (jittered at top for 1 and bottom for 0)
        y_obs = np.array(stats_results['y_obs'])
        x_obs = np.array(stats_results['primary_x_vals'])
        p_obs = np.array(stats_results['prob_pred'])
        obs_ids = demo_df["Obs_ID"].values
        rng_jitter = np.random.default_rng(101)

        mask1 = (y_obs == 1)
        jitter1 = rng_jitter.uniform(0.96, 1.04, size=int(np.sum(mask1)))
        fig_curve.add_trace(go.Scatter(
            x=x_obs[mask1],
            y=jitter1,
            mode="markers",
            name=f"Observed Event (1, n={int(np.sum(mask1))})",
            marker=dict(color="#ef4444", size=8, opacity=0.75, symbol="circle"),
            customdata=np.column_stack([obs_ids[mask1], p_obs[mask1]]),
            hovertemplate="<b>%{customdata[0]}</b><br>Actual: Event (1)<br>" + stats_results['primary_x_name'] + ": %{x:.2f}<br>Model Predicted P: %{customdata[1]:.3f}<extra></extra>"
        ))

        mask0 = (y_obs == 0)
        jitter0 = rng_jitter.uniform(-0.04, 0.04, size=int(np.sum(mask0)))
        fig_curve.add_trace(go.Scatter(
            x=x_obs[mask0],
            y=jitter0,
            mode="markers",
            name=f"Observed Control (0, n={int(np.sum(mask0))})",
            marker=dict(color="#10b981", size=8, opacity=0.75, symbol="circle-open"),
            customdata=np.column_stack([obs_ids[mask0], p_obs[mask0]]),
            hovertemplate="<b>%{customdata[0]}</b><br>Actual: Control (0)<br>" + stats_results['primary_x_name'] + ": %{x:.2f}<br>Model Predicted P: %{customdata[1]:.3f}<extra></extra>"
        ))

        fig_curve.update_layout(
            height=380,
            margin=dict(l=40, r=40, t=30, b=40),
            xaxis_title=stats_results['primary_x_name'],
            yaxis_title="Predicted Probability P(Y = 1)",
            yaxis=dict(range=[-0.08, 1.08], tickvals=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    with vcol2:
        st.markdown(f"**2. ROC Curve & Model Discrimination (AUC = {stats_results['auc']:.3f})**")
        fig_roc = go.Figure()

        fig_roc.add_trace(go.Scatter(
            x=stats_results['roc_fpr'],
            y=stats_results['roc_tpr'],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.12)",
            name=f"Model ROC (AUC = {stats_results['auc']:.3f})",
            line=dict(color="#2563eb", width=3)
        ))

        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance Baseline (AUC = 0.50)",
            line=dict(color="#94a3b8", width=2, dash="dash")
        ))

        fpr_curr = float(1.0 - stats_results['specificity'])
        tpr_curr = float(stats_results['sensitivity'])
        fig_roc.add_trace(go.Scatter(
            x=[fpr_curr],
            y=[tpr_curr],
            mode="markers",
            name=f"Cutoff 0.50 (Sens: {tpr_curr:.2f}, Spec: {stats_results['specificity']:.2f})",
            marker=dict(color="#f59e0b", size=12, symbol="diamond", line=dict(color="#b45309", width=2)),
            hovertemplate="<b>Operating Point (Cutoff 0.50)</b><br>False Positive Rate: %{x:.3f}<br>Sensitivity (TPR): %{y:.3f}<extra></extra>"
        ))

        fig_roc.update_layout(
            height=380,
            margin=dict(l=40, r=40, t=30, b=40),
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            xaxis=dict(range=[-0.02, 1.02]),
            yaxis=dict(range=[-0.02, 1.02]),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # 3. Model Parameters, Odds Ratios & Wald Significance Table
    st.markdown("**3. Logistic Regression Coefficients, Standard Errors & Adjusted Odds Ratios (95% CI)**")
    st.caption("Slopes (β) represent change in log-odds per unit change in predictor. Adjusted Odds Ratio (OR = e^β) represents multiplicative factor on odds, holding all other covariates constant.")

    param_rows = []
    # Intercept row
    p_val_0 = stats_results['p_values'][0]
    p_str_0 = "< 0.001" if p_val_0 < 0.001 else f"{p_val_0:.4f}"
    sig_badge_0 = "✅ Significant (p < 0.05)" if p_val_0 < 0.05 else "⚪ Not Significant (p ≥ 0.05)"
    param_rows.append({
        "Parameter_or_Predictor": "Intercept (β₀)",
        "Coefficient_β": f"{stats_results['beta'][0]:.4f}",
        "Std_Error_SE": f"{stats_results['se'][0]:.4f}",
        "Wald_z": f"{stats_results['z_stats'][0]:.3f}",
        "p_value": p_str_0,
        "Adjusted_OR": f"{stats_results['odds_ratios'][0]:.3f}",
        "CI_95_Lower": f"{stats_results['or_ci_lower'][0]:.3f}",
        "CI_95_Upper": f"{stats_results['or_ci_upper'][0]:.3f}",
        "Significance_Status": sig_badge_0
    })

    for idx, p_name in enumerate(stats_results['pred_names']):
        j = idx + 1
        b_val = stats_results['beta'][j]
        se_val = stats_results['se'][j]
        z_val = stats_results['z_stats'][j]
        p_val = stats_results['p_values'][j]
        or_val = stats_results['odds_ratios'][j]
        cil_val = stats_results['or_ci_lower'][j]
        ciu_val = stats_results['or_ci_upper'][j]

        p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        sig_badge = "✅ Significant (p < 0.05)" if p_val < 0.05 else "⚪ Not Significant (p ≥ 0.05)"

        param_rows.append({
            "Parameter_or_Predictor": f"{p_name} (β_{idx+1})",
            "Coefficient_β": f"{b_val:.4f}",
            "Std_Error_SE": f"{se_val:.4f}",
            "Wald_z": f"{z_val:.3f}",
            "p_value": p_str,
            "Adjusted_OR": f"{or_val:.3f}",
            "CI_95_Lower": f"{cil_val:.3f}",
            "CI_95_Upper": f"{ciu_val:.3f}",
            "Significance_Status": sig_badge
        })

    df_params = pd.DataFrame(param_rows)
    render_table(df_params, height=180 + 35 * len(stats_results['pred_names']))

    # 4. Classification Diagnostics & Confusion Matrix
    st.markdown("**4. Classification Diagnostics & 2×2 Confusion Matrix (Decision Threshold = 0.50)**")
    cm_col1, cm_col2, cm_col3, cm_col4, cm_col5 = st.columns(5)
    cm_col1.metric("Overall Accuracy", f"{stats_results['accuracy']*100:.1f}%", help="Proportion of all cases correctly classified: (TP + TN) / N")
    cm_col2.metric("Sensitivity (Recall)", f"{stats_results['sensitivity']*100:.1f}%", help="True Positive Rate: TP / (TP + FN). Probability of predicting Event when actually Event.")
    cm_col3.metric("Specificity", f"{stats_results['specificity']*100:.1f}%", help="True Negative Rate: TN / (TN + FP). Probability of predicting Control when actually Control.")
    cm_col4.metric("Precision (PPV)", f"{stats_results['precision']*100:.1f}%", help="Positive Predictive Value: TP / (TP + FP). Probability that predicted Event is truly an Event.")
    cm_col5.metric("Negative Pred Value (NPV)", f"{stats_results['npv']*100:.1f}%", help="Negative Predictive Value: TN / (TN + FN). Probability that predicted Control is truly a Control.")

    # Render clean Confusion Matrix DataFrame
    tp = stats_results['tp']
    fn = stats_results['fn']
    fp = stats_results['fp']
    tn = stats_results['tn']
    df_cm = pd.DataFrame({
        "Actual_Outcome": ["Actual Event (Y = 1)", "Actual Control (Y = 0)", "Total Predicted"],
        "Predicted_Event (1)": [f"TP = {tp}", f"FP = {fp}", f"{tp + fp}"],
        "Predicted_Control (0)": [f"FN = {fn}", f"TN = {tn}", f"{fn + tn}"],
        "Total_Actual": [f"{tp + fn}", f"{fp + tn}", f"{tp + fn + fp + tn}"]
    })
    render_table(df_cm, height=140)

    # 5. Multicollinearity Diagnostics Table
    st.markdown("**5. Multicollinearity Diagnostics (Variance Inflation Factor / VIF & Tolerance)**")
    collin_rows = []
    for idx, p_name in enumerate(stats_results['pred_names']):
        v_val = stats_results['vifs'][idx] if idx < len(stats_results['vifs']) else 1.0
        t_val = 1.0 / v_val if v_val > 0 else 0.0
        if v_val < 5.0:
            stat_badge = "✅ Safe (VIF < 5.0)"
            guide_str = "Independent predictive contribution; stable coefficient variance."
        elif v_val < 10.0:
            stat_badge = "🟡 Moderate Collinearity (5.0 ≤ VIF < 10.0)"
            guide_str = "Moderate predictor correlation; standard error slightly inflated."
        else:
            stat_badge = "❌ Severe Multicollinearity (VIF ≥ 10.0)"
            guide_str = "High redundancy between predictors; unstable Odds Ratios. Consider L2 regularization."

        collin_rows.append({
            "Predictor": p_name,
            "VIF": f"{v_val:.2f}",
            "Tolerance": f"{t_val:.3f}",
            "Collinearity_Status": stat_badge,
            "Diagnostic_Guidance": guide_str
        })
    df_collin = pd.DataFrame(collin_rows)
    render_table(df_collin, height=140 + 35 * len(stats_results['pred_names']))

    # 6. Subject-Level Observational Data Preview
    st.markdown(f"**6. Subject-Level Observational Data & Model Predictions (N = {stats_results['n_pts']})**")
    st.markdown("""
    **What this table contains:**
    * Case-by-case microdata for the first 10 subjects, showing observation ID, actual binary outcome $Y$, observed predictors $X_1, X_2, \\dots$, model predicted probability $p_i$, and predicted class (0/1).
    * Enables complete calculation auditing and download of the synthetic dataset for Excel / Python practice.
    """)
    render_table(demo_df.head(10), height=240)
    st.download_button(
        "📥 Download Simulated Logistic Regression Dataset (CSV)",
        data=demo_df.to_csv(index=False),
        file_name="binary_logistic_regression_simulation_data.csv",
        mime="text/csv",
        key="btn_download_binary_logistic"
    )

    # 7. Core Methodological Assumption Diagnostics Cards (5 cards in 2 columns)
    st.markdown("### 🛡️ Core Methodological Assumption Diagnostics")
    st.caption("Maximum Likelihood Binary Logistic Regression provides reliable parameter estimates and valid Odds Ratios when these 5 assumptions hold.")

    assumptions = stats_results["assumptions_eval"]
    col_a, col_b = st.columns(2)

    for i, asm in enumerate(assumptions):
        target_col = col_a if i % 2 == 0 else col_b
        if asm["status"] == "pass":
            status_color = "#10b981"
            bg_color = "#f0fdf4"
            badge_symbol = "✅"
        elif asm["status"] == "warning":
            status_color = "#f59e0b"
            bg_color = "#fffbeb"
            badge_symbol = "🟡"
        else:
            status_color = "#ef4444"
            bg_color = "#fef2f2"
            badge_symbol = "❌"

        with target_col:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{asm['name']}</strong>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {status_color}; background: white; padding: 0.15rem 0.5rem; border-radius: 9999px; border: 1px solid {status_color};">
                        {badge_symbol} {asm['verdict']}
                    </span>
                </div>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #475569;"><em>{asm['description']}</em></p>
                <div style="font-size: 0.85rem; color: #334155; margin-bottom: 0.4rem;">
                    <strong>Diagnostic Metric:</strong> <code>{asm['diagnostic_value']}</code>
                </div>
                <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.4;">
                    {asm['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 8. Actionable Final Recommendation Banner
    st.markdown("### 📢 Methodological Recommendation & Verdict")
    if stats_results["overall_status"] == "pass":
        st.success(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)
    else:
        st.info(f"""
        ### {stats_results['overall_verdict_badge']}
        {stats_results['overall_recommendation']}
        """)

    # 9. Expandable Details for Alternative Tests
    with st.expander("🔎 View Alternative Model Details (Penalized Logistic Regression, Firth's Correction, & Regularization)", expanded=False):
        st.markdown(r"""
        * **L2 Ridge Regularization (When Multicollinearity or EPV < 10 Exists):**
          * When predictors are highly intercorrelated ($\text{VIF} \ge 10$) or effective sample size is modest ($\text{EPV} < 10$), standard maximum likelihood produces wild swings and inflated standard errors.
          * **L2 Ridge regularization** penalizes large squared coefficients ($\lambda \sum \beta_j^2$), shrinking estimation variance and preventing overfitting.
        * **Firth's Penalized Likelihood (When Complete / Quasi-Complete Separation Occurs):**
          * If a predictor perfectly classifies outcomes (e.g., all smokers had a CVD event and no non-smokers had one), standard MLE fails because the parameter estimate diverges to $\pm \infty$ with infinite standard errors.
          * **Firth's method** introduces a Jeffreys prior penalty to the likelihood function ($\frac{1}{2} \ln |I(\beta)|$), producing finite, consistent parameter estimates even with small samples or perfect separation.
        * **Multinomial & Ordinal Logistic Regression (When Outcome Has 3+ Levels):**
          * If the target variable has 3 or more unordered categories (e.g., Treatment A vs B vs C), use **Multinomial Logistic Regression**.
          * If the target variable has 3 or more natural ranked levels (e.g., Mild, Moderate, Severe), use **Ordinal Logistic Regression (Proportional Odds Model)**.
        * **Log-Binomial & Poisson Regression with Robust Variance (For Common Outcomes):**
          * When the outcome is frequent ($> 10\%$), the Odds Ratio ($OR$) substantially overstates the Relative Risk ($RR$).
          * In prospective cohort studies and randomized trials, **Log-Binomial Regression** or **Modified Poisson Regression** directly estimates adjusted Relative Risks ($RR$) instead of Odds Ratios.
        """)










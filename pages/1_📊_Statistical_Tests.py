"""Streamlit Page: Statistical Test Decision Wizard.

Interactive decision tree funnel for selecting hypothesis tests, with
rationale, assumptions, Excel guidelines, and live interactive simulations.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import streamlit as st
import sys
import os

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stats_calcs.decision_schema import (
    recommend_statistical_test,
    get_statistical_test_recommendation,
    get_all_statistical_tests,
    STATISTICAL_TESTS
)
from src.stats_calcs.synthetic_data import (
    generate_one_sample_demo,
    generate_wilcoxon_demo,
    generate_mann_whitney_demo,
    generate_welch_demo,
    generate_students_t_demo,
    generate_two_independent_demo,
    generate_paired_demo,
    generate_wilcoxon_paired_demo,
    generate_one_way_anova_demo,
    generate_kruskal_wallis_demo,
    generate_repeated_measures_anova_demo,
    generate_friedman_demo,
    generate_three_groups_demo,
    generate_contingency_demo,
    generate_fisher_exact_demo,
    generate_mcnemar_demo,
    generate_correlation_demo,
    generate_pearson_correlation_demo,
    generate_spearman_correlation_demo,
    generate_linear_regression_demo,
    generate_binary_logistic_regression_demo
)

from src.ui_components.stats_widgets import (
    render_recommendation_card,
    render_methodology_tabs,
    render_interactive_demo
)
from src.ui_components.sidebar_navigation import (
    render_custom_sidebar_navigation,
    render_roundtrip_navigation_banner
)
from src.ui_components.sample_size_badge import render_sample_size_deep_link_badge


st.set_page_config(
    page_title="Statistical Tests Finder",
    page_icon="📊",
    layout="wide"
)

# Render Custom Grouped Sidebar Navigation
render_custom_sidebar_navigation("pages/1_📊_Statistical_Tests.py")

st.title("📊 Statistical Test Decision Assistant")
render_roundtrip_navigation_banner()
st.markdown("""
Navigate the questions below to identify the appropriate statistical test for your research question, 
variable types, and study design.
""")

# Temporary progress & validation tracking expander requested by user
if st.session_state.get("debug_mode", False) or st.session_state.get("debug_model", False):
    with st.expander("📌 Statistical Methods Implementation & Validation Tracker (Temporary)", expanded=False):
        st.markdown("""
        **Project Status: Phase 1 — Statistical Tests Implementation**  
        This checklist tracks progress across all 17 statistical methods supported by the decision framework. 
        Each method includes interactive simulation, assumption diagnostic engines, and 3-step worked Excel & Python guides.
        """)

        st.markdown(r"""
        #### ⚖️ Option 1: Comparing Groups
        * **1 Group vs. Theoretical Benchmark:**
          * [x] **One-Sample t-test** (Continuous, Normal)
          * [x] **Wilcoxon Signed-Rank Test (1-Sample)** (Continuous/Ordinal, Non-Normal)
        * **2 Independent Groups:**
          * [x] **Mann-Whitney U Test** (Ordinal / Continuous Non-Normal)
          * [x] **Welch's Independent Two-Sample t-test** (Continuous, Normal, Unequal Variances)
          * [x] **Student's Independent Two-Sample t-test** (Continuous, Normal, Equal Variances)
        * **2 Paired / Repeated Groups:**
          * [x] **Paired Samples t-test** (Continuous, Normal Differences)
          * [x] **Wilcoxon Signed-Rank Test (Paired)** (Ordinal / Non-Normal Differences)
        * **3+ Groups (ANOVA & Non-Parametric):**
          * [x] **One-Way ANOVA** (3+ Independent, Normal, Homoscedastic)
          * [x] **Kruskal-Wallis H Test** (3+ Independent, Non-Normal / Ordinal)
          * [x] **Repeated Measures ANOVA** (3+ Repeated Measures, Normal, Sphericity)
          * [x] **Friedman Test** (3+ Repeated Measures, Non-Normal / Ordinal)

        #### 🗂️ Option 2: Categorical Proportions & Contingency Tables
        * [x] **Pearson's Chi-Square Test of Independence** (Adequate Expected Frequencies $\ge 5$)
        * [x] **Fisher's Exact Test** (Small Sample Frequencies $< 5$)
        * [x] **McNemar's Test** (Paired Binary Proportions)

        #### 📈 Option 3: Relationships, Associations & Regression
        * [x] **Pearson Correlation Coefficient ($r$ and $R^2$)** (Linear, Continuous, Normal)
        * [x] **Spearman Rank Correlation ($\rho$)** (Monotonic, Non-Normal / Ordinal)
        * [x] **Simple / Multiple Linear Regression (OLS)** (Continuous Outcome Prediction)
        * [x] **Binary Logistic Regression** (Binary Outcome Prediction / Adjusted Odds)
        """)

st.markdown("---")

# ----------------- Navigation Mode Selection -----------------
st.subheader("🧭 Choose Navigation Mode")

if "nav_mode" not in st.session_state:
    st.session_state["nav_mode"] = "guided"

MODE_GUIDED = "🎯 Guided Decision Finder (Steps 1 & 2: Route by research question & design)"
MODE_DIRECT = "⚡ Direct Method Explorer (Select any test directly from full catalog)"

mode_options = [MODE_GUIDED, MODE_DIRECT]
default_index = 0 if st.session_state["nav_mode"] == "guided" else 1

selected_mode_label = st.radio(
    "How would you like to explore statistical tests?",
    options=mode_options,
    index=default_index,
    horizontal=True,
    help="Switch between guided step-by-step decision routing and direct selection of any statistical method."
)

is_direct_mode = (selected_mode_label == MODE_DIRECT)
st.session_state["nav_mode"] = "direct" if is_direct_mode else "guided"

if not is_direct_mode:
    st.markdown("---")
    # ----------------- Step 1: Primary Objective -----------------
    st.subheader("Step 1: What is your primary research objective?")

    objective_options = {
        "compare_groups": "⚖️ Compare groups (e.g. Means or medians across treatments or populations)",
        "contingency_proportions": "🗂️ Analyze categorical frequencies / proportions (e.g. 2x2 contingency table)",
        "examine_relationship": "📈 Examine relationships or correlations (e.g. association between 2 continuous measures)",
        "time_to_event": "⏱️ Time-to-event / Survival data (e.g. time until remission or death with censoring)"
    }

    selected_objective = st.selectbox(
        "Select your objective:",
        options=list(objective_options.keys()),
        format_func=lambda x: objective_options[x],
        index=0,
    )

    answers = {"primary_objective": selected_objective}

    st.markdown("---")
    st.subheader("Step 2: Define your data characteristics & design")

    # ----------------- Step 2: Contextual Questions -----------------
    col_left, col_right = st.columns(2)

    if selected_objective == "compare_groups":
        with col_left:
            num_groups = st.selectbox(
                "How many groups / conditions are being compared?",
                options=["1", "2", "3+"],
                format_func=lambda x: {
                    "1": "1 group (compared against a known population benchmark)",
                    "2": "2 groups (e.g. Treatment vs Placebo)",
                    "3+": "3 or more groups (e.g. Drug A vs Drug B vs Control)"
                }[x],
                index=0
            )

            answers["num_groups"] = num_groups

            dep_type = st.selectbox(
                "What type of dependent (outcome) variable do you have?",
                options=["continuous", "ordinal"],
                format_func=lambda x: {
                    "continuous": "Continuous (e.g. blood pressure, cholesterol, reaction time)",
                    "ordinal": "Ordinal / Ranked (e.g. Likert scale 1-5, pain severity scores)"
                }[x],
                index=0
            )
            answers["dependent_type"] = dep_type

        with col_right:
            if num_groups != "1":
                group_design = st.selectbox(
                    "Are the samples independent or paired / repeated measures?",
                    options=["independent", "paired"],
                    format_func=lambda x: {
                        "independent": "Independent (different subjects in each group)",
                        "paired": "Paired / Repeated (same subjects tested repeatedly or matched pairs)"
                    }[x],
                    index=0
                )
                answers["group_design"] = group_design
            else:
                answers["group_design"] = "independent"

            if dep_type == "continuous":
                normality = st.selectbox(
                    "Are the data distributions approximately normal?",
                    options=["normal", "non_normal"],
                    format_func=lambda x: {
                        "normal": "Yes / Approximate Normal (or N > 30 per group via Central Limit Theorem)",
                        "non_normal": "No / Strongly Skewed (or small sample N < 30)"
                    }[x],
                    index=0
                )
                answers["normality"] = normality

                if num_groups == "2" and answers.get("group_design") == "independent" and normality == "normal":
                    equal_var = st.selectbox(
                        "Can you assume equal variances between groups?",
                        options=["unknown", "equal", "unequal"],
                        format_func=lambda x: {
                            "unknown": "Unknown / Not Tested (Welch's t-test is safest default)",
                            "equal": "Equal Variances Verified (Levene's test p > 0.05)",
                            "unequal": "Unequal Variances (Heteroscedastic)"
                        }[x],
                        index=0
                    )
                    answers["equal_variance"] = equal_var

    elif selected_objective == "contingency_proportions":
        with col_left:
            table_setup = st.selectbox(
                "How is your categorical comparison structured?",
                options=["two_by_two", "paired_binary"],
                format_func=lambda x: {
                    "two_by_two": "Independent 2x2 Contingency Table (e.g. Exposed vs Unexposed by Disease Yes/No)",
                    "paired_binary": "Paired Binary Observations (e.g. Before vs After response in same individuals)"
                }[x],
                index=0
            )
            answers["table_setup"] = table_setup

        with col_right:
            if table_setup == "two_by_two":
                cell_counts = st.selectbox(
                    "What are your expected cell counts?",
                    options=["adequate", "small"],
                    format_func=lambda x: {
                        "adequate": "All expected cell counts >= 5 (Standard sample size)",
                        "small": "At least one expected cell count < 5 (Small sample size)"
                    }[x],
                    index=0
                )
                answers["cell_counts"] = cell_counts

    elif selected_objective == "examine_relationship":
        with col_left:
            rel_goal = st.selectbox(
                "What is the analytical goal of the relationship?",
                options=["correlation", "predict_continuous", "predict_binary"],
                format_func=lambda x: {
                    "correlation": "Assess correlation / association between 2 variables",
                    "predict_continuous": "Predict a continuous outcome from 1 or more variables (Regression)",
                    "predict_binary": "Predict a binary outcome (Yes/No) from predictor variables (Classification)"
                }[x],
                index=0
            )
            answers["relationship_goal"] = rel_goal

        with col_right:
            if rel_goal == "correlation":
                linearity = st.selectbox(
                    "Is the relationship linear or monotonic / ranked?",
                    options=["linear", "monotonic_or_ranked"],
                    format_func=lambda x: {
                        "linear": "Linear & continuous normal distribution (Pearson r)",
                        "monotonic_or_ranked": "Monotonic, ranked, or skewed with outliers (Spearman rho)"
                    }[x],
                    index=0
                )
                answers["linearity"] = linearity

    elif selected_objective == "time_to_event":
        with col_left:
            surv_goal = st.selectbox(
                "What is your survival analysis aim?",
                options=["compare_curves", "model_covariates"],
                format_func=lambda x: {
                    "compare_curves": "Compare survival curves between groups (Kaplan-Meier & Log-rank test)",
                    "model_covariates": "Adjust for multiple covariates / predictors (Cox Regression)"
                }[x],
                index=0
            )
            answers["survival_goal"] = surv_goal

    recommendation = recommend_statistical_test(answers)

else:
    st.markdown("---")
    st.subheader("⚡ Direct Method Explorer")
    st.markdown(
        "Select any statistical test from our catalog to directly inspect its methodology rationale, "
        "key assumptions, 3-step decision context, worked Microsoft Excel guide, ready-to-run Python code, and interactive simulation."
    )

    CATEGORY_GROUPS = {
        "All Categories (Show All 19 Tests)": None,
        "⚖️ Comparing 1 Group": ["one_sample_t_test", "wilcoxon_signed_rank"],
        "⚖️ Comparing 2 Independent Groups": ["welch_t_test", "students_t_test", "mann_whitney_u"],
        "⚖️ Comparing 2 Paired / Repeated Groups": ["paired_t_test", "wilcoxon_paired"],
        "⚖️ Comparing 3+ Independent Groups": ["one_way_anova", "kruskal_wallis"],
        "⚖️ Comparing 3+ Paired / Repeated Groups": ["repeated_measures_anova", "friedman_test"],
        "🗂️ Categorical Proportions & Contingency": ["chi_square_independence", "fisher_exact", "mcnemar_test"],
        "📈 Relationships & Associations": ["pearson_correlation", "spearman_correlation"],
        "🎯 Outcome Prediction (Regression)": ["linear_regression", "binary_logistic_regression"],
        "⏱️ Survival & Time-to-Event Analysis": ["log_rank_test"],
    }

    col_cat, col_test = st.columns([1, 2])

    with col_cat:
        selected_category_group = st.selectbox(
            "Filter by Research Category:",
            options=list(CATEGORY_GROUPS.keys()),
            index=0,
            help="Filter the test list by clinical/research design family, or view all 19 tests."
        )

    allowed_keys = CATEGORY_GROUPS[selected_category_group]
    if allowed_keys is None:
        available_keys = list(STATISTICAL_TESTS.keys())
    else:
        available_keys = allowed_keys

    preselected_test = st.session_state.get("direct_test_key", "welch_t_test")
    if preselected_test not in available_keys:
        preselected_test = available_keys[0]

    test_index = available_keys.index(preselected_test)

    def format_direct_test_option(test_key: str) -> str:
        info = STATISTICAL_TESTS[test_key]
        param_tag = "Parametric" if info["parametric"] else "Non-parametric"
        return f"{info['name']} ({param_tag})"

    with col_test:
        chosen_test_key = st.selectbox(
            "Select Statistical Test:",
            options=available_keys,
            index=test_index,
            format_func=format_direct_test_option,
            help="Select the test to inspect methodology, Excel formulas, Python code, and live simulation."
        )
        st.session_state["direct_test_key"] = chosen_test_key

    chosen_info = STATISTICAL_TESTS[chosen_test_key]
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.info(f"📁 **Category:** {chosen_info['category']}")
    with col_m2:
        st.info(f"📊 **Type:** {'Parametric (Normal Distribution Assumed)' if chosen_info['parametric'] else 'Non-Parametric / Distribution-Free'}")
    with col_m3:
        st.info(f"🔄 **Alternative:** {chosen_info.get('alternative_test', 'N/A')}")

    recommendation = get_statistical_test_recommendation(chosen_test_key)

def configure_and_generate_demo_data(
    demo_type: Optional[str],
    recommendation: Dict[str, Any]
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Configures sidebar parameters and executes live demo generation for the active method."""
    if demo_type == "one_sample_continuous":
        st.sidebar.header("🧪 One-Sample t-test Settings")
        st.sidebar.markdown("Configure the sample data and theoretical benchmark:")
        
        benchmark = st.sidebar.number_input(
            "Theoretical Benchmark (μ₀)",
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            step=1.0,
            help="The population benchmark or historical target to compare the sample against."
        )
        n = st.sidebar.slider(
            "Sample Size (N)",
            min_value=10,
            max_value=150,
            value=35,
            step=5,
            help="Number of observations in the sample. N >= 30 triggers Central Limit Theorem robustness."
        )
        sample_mean = st.sidebar.slider(
            "Sample True Mean",
            min_value=60.0,
            max_value=140.0,
            value=106.0,
            step=1.0,
            help="True center for random data generation."
        )
        sample_std = st.sidebar.slider(
            "Sample Standard Deviation (s)",
            min_value=2.0,
            max_value=25.0,
            value=12.0,
            step=1.0,
            help="Variability of the sample."
        )
        dist_type = st.sidebar.selectbox(
            "Distribution Shape",
            options=["normal", "skewed_right", "skewed_left", "heavy_tailed"],
            format_func=lambda x: {
                "normal": "Normal (Symmetric, bell-shaped)",
                "skewed_right": "Right-Skewed (Positive skewness)",
                "skewed_left": "Left-Skewed (Negative skewness)",
                "heavy_tailed": "Heavy-Tailed (Student's t, df=3)"
            }[x],
            index=0,
            help="Select skewed or heavy-tailed distributions to test assumption checks."
        )
        outlier_count = st.sidebar.selectbox(
            "Inject Extreme Outliers",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 - None (Clean)",
                1: "1 Extreme Outlier (+4.5σ)",
                2: "2 Extreme Outliers (+4.5σ)",
                3: "3 Extreme Outliers (+4.5σ)"
            }[x],
            index=0,
            help="Test the Tukey IQR outlier diagnostic and observe how it impacts the test recommendation."
        )
        seed = st.sidebar.number_input(
            "Random Seed",
            min_value=1,
            max_value=99999,
            value=42,
            step=1,
            help="Seed for reproducible random generation."
        )
    
        df_demo, stats_res = generate_one_sample_demo(
            n=n,
            sample_mean=sample_mean,
            sample_std=sample_std,
            benchmark=benchmark,
            dist_type=dist_type,
            outlier_count=outlier_count,
            seed=seed
        )
        stats_res["preset_name"] = f"One-Sample (Benchmark μ₀ = {benchmark:.1f}, Shape = {dist_type})"
        return df_demo, stats_res
    
    elif demo_type == "wilcoxon_signed_rank":
        st.sidebar.header("🧪 Wilcoxon Signed-Rank Settings")
        st.sidebar.markdown("Configure the sample data and theoretical median benchmark:")
    
        benchmark = st.sidebar.number_input(
            "Theoretical Median Benchmark (μ₀)",
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            step=1.0,
            help="The theoretical population median or historical reference value to test against."
        )
        n = st.sidebar.slider(
            "Sample Size (N)",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Total number of observations in the sample."
        )
        sample_median = st.sidebar.slider(
            "Sample True Median",
            min_value=60.0,
            max_value=140.0,
            value=108.0,
            step=1.0,
            help="Target median around which the random sample is centered."
        )
        sample_scale = st.sidebar.slider(
            "Scale / Dispersion",
            min_value=2.0,
            max_value=30.0,
            value=15.0,
            step=1.0,
            help="Spread and variability of the distribution."
        )
        dist_shape = st.sidebar.selectbox(
            "Distribution Shape & Symmetry",
            options=["symmetric_non_normal", "moderately_skewed", "severely_skewed"],
            format_func=lambda x: {
                "symmetric_non_normal": "Symmetric Non-Normal (Laplace - Ideal for Wilcoxon)",
                "moderately_skewed": "Moderately Skewed (Gamma - Caution/Walsh Averages)",
                "severely_skewed": "Severely Skewed (Log-Normal - Triggers Sign Test Alternative)"
            }[x],
            index=0,
            help="Select skewed distributions to test how the symmetry diagnostic reacts."
        )
        zero_diff_count = st.sidebar.selectbox(
            "Observations Matching Benchmark (Ties = 0)",
            options=[0, 1, 2, 3, 5],
            format_func=lambda x: f"{x} ties (dropped from ranking)" if x > 0 else "0 (No ties)",
            index=0,
            help="Tests how the Wilcoxon procedure drops zero differences and calculates N_eff."
        )
        seed = st.sidebar.number_input(
            "Random Seed",
            min_value=1,
            max_value=99999,
            value=42,
            step=1,
            help="Seed for reproducible random generation."
        )
    
        df_demo, stats_res = generate_wilcoxon_demo(
            n=n,
            sample_median=sample_median,
            sample_scale=sample_scale,
            benchmark=benchmark,
            dist_shape=dist_shape,
            zero_diff_count=zero_diff_count,
            seed=seed
        )
        stats_res["preset_name"] = f"Wilcoxon 1-Sample (Median Benchmark μ₀ = {benchmark:.1f}, Shape = {dist_shape})"
        return df_demo, stats_res
    
    elif demo_type == "mann_whitney_u":
        st.sidebar.header("🧪 Mann-Whitney U Settings")
        st.sidebar.markdown("Configure the two independent groups and distribution properties:")
    
        st.sidebar.markdown("##### 👥 Group Sample Sizes")
        col_n1, col_n2 = st.sidebar.columns(2)
        n_a = col_n1.slider("Group A (n₁)", min_value=10, max_value=80, value=30, step=5, help="Number of observations in Group A (Control).")
        n_b = col_n2.slider("Group B (n₂)", min_value=10, max_value=80, value=30, step=5, help="Number of observations in Group B (Treatment).")
    
        st.sidebar.markdown("##### 🎯 Target Medians & Scales")
        col_m1, col_m2 = st.sidebar.columns(2)
        median_a = col_m1.slider("Median A", min_value=60.0, max_value=140.0, value=100.0, step=1.0, help="Target median for Group A.")
        median_b = col_m2.slider("Median B", min_value=60.0, max_value=140.0, value=112.0, step=1.0, help="Target median for Group B.")
    
        col_s1, col_s2 = st.sidebar.columns(2)
        scale_a = col_s1.slider("Scale A", min_value=5.0, max_value=30.0, value=15.0, step=1.0, help="Spread and dispersion for Group A.")
        scale_b = col_s2.slider("Scale B", min_value=5.0, max_value=30.0, value=15.0, step=1.0, help="Spread and dispersion for Group B.")
    
        dist_shape = st.sidebar.selectbox(
            "Distribution Shapes & Variance Homogeneity",
            options=["symmetric_non_normal", "moderately_skewed", "unequal_shapes_variances"],
            format_func=lambda x: {
                "symmetric_non_normal": "Symmetric Non-Normal (Laplace - Equal Shapes / Medians)",
                "moderately_skewed": "Moderately Skewed (Gamma - Equal Shapes in Both)",
                "unequal_shapes_variances": "Unequal Shapes & Variances (Heteroscedastic - Tests Alternatives)"
            }[x],
            index=0,
            help="Test the variance homogeneity and shape diagnostics by selecting unequal shapes."
        )
        seed = st.sidebar.number_input(
            "Random Seed",
            min_value=1,
            max_value=99999,
            value=42,
            step=1,
            help="Seed for reproducible random generation."
        )
    
        df_demo, stats_res = generate_mann_whitney_demo(
            n_a=n_a,
            n_b=n_b,
            median_a=median_a,
            median_b=median_b,
            scale_a=scale_a,
            scale_b=scale_b,
            dist_shape=dist_shape,
            seed=seed
        )
        stats_res["preset_name"] = f"Mann-Whitney U (Shape = {dist_shape}, N = {n_a + n_b})"
        return df_demo, stats_res
    
    elif demo_type == "welch_t_test":
        st.sidebar.header("🧪 Welch's t-Test Settings")
        st.sidebar.markdown("Configure sample sizes, means, and variances for two independent groups:")
    
        st.sidebar.markdown("##### 👥 Group Sample Sizes")
        col_n1, col_n2 = st.sidebar.columns(2)
        n_a = col_n1.slider("Group A (n₁)", min_value=10, max_value=100, value=35, step=5, help="Number of observations in Group A (Control).")
        n_b = col_n2.slider("Group B (n₂)", min_value=10, max_value=100, value=35, step=5, help="Number of observations in Group B (Treatment).")
    
        st.sidebar.markdown("##### 🎯 Target Population Means")
        col_m1, col_m2 = st.sidebar.columns(2)
        mean_a = col_m1.slider("Mean A (x̄₁)", min_value=60.0, max_value=140.0, value=100.0, step=1.0, help="Target population mean for Group A.")
        mean_b = col_m2.slider("Mean B (x̄₂)", min_value=60.0, max_value=140.0, value=108.0, step=1.0, help="Target population mean for Group B.")
    
        st.sidebar.markdown("##### 📏 Group Standard Deviations (Variances)")
        col_s1, col_s2 = st.sidebar.columns(2)
        std_a = col_s1.slider("Std Dev A (s₁)", min_value=2.0, max_value=30.0, value=10.0, step=1.0, help="Standard deviation for Group A.")
        std_b = col_s2.slider("Std Dev B (s₂)", min_value=2.0, max_value=30.0, value=18.0, step=1.0, help="Standard deviation for Group B (supports unequal variances / heteroscedasticity).")
    
        st.sidebar.markdown("##### 🛡️ Assumption Testing Scenarios")
        dist_shape = st.sidebar.selectbox(
            "Distribution Shape",
            options=["normal", "skewed_moderate", "heavy_tailed"],
            format_func=lambda x: {
                "normal": "Normal Distribution (Ideal for Welch t-test)",
                "skewed_moderate": "Moderately Skewed (Gamma - Tests CLT robustness)",
                "heavy_tailed": "Heavy-Tailed (Student t df=3 - Tests sensitivity)"
            }[x],
            index=0,
            help="Select to test how non-normality affects assumption evaluation and test recommendations."
        )
    
        outlier_count_b = st.sidebar.selectbox(
            "Extreme Outliers in Group B (>3×IQR)",
            options=[0, 1, 2, 3],
            index=0,
            help="Inject extreme outliers into Group B to test the Tukey IQR assumption check and see how the app recommends switching to the Mann-Whitney U test."
        )
    
        seed = st.sidebar.number_input(
            "Random Seed",
            min_value=1,
            max_value=99999,
            value=42,
            step=1,
            help="Seed for reproducible random generation."
        )
    
        df_demo, stats_res = generate_welch_demo(
            n_a=n_a,
            n_b=n_b,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            dist_shape=dist_shape,
            outlier_count_b=outlier_count_b,
            seed=seed
        )
        stats_res["preset_name"] = f"Welch t-test (Shape = {dist_shape}, N = {n_a + n_b})"
        return df_demo, stats_res
    
    elif demo_type in ["students_t_test", "two_independent_continuous"]:
        st.sidebar.header("🧪 Student's t-Test Demo Settings")
    
        n_a = st.sidebar.slider("Group A Sample Size (n₁)", 10, 150, 35)
        n_b = st.sidebar.slider("Group B Sample Size (n₂)", 10, 150, 35)
        mean_a = st.sidebar.slider("Group A Mean (μ₁)", 80.0, 160.0, 100.0, 1.0)
        mean_b = st.sidebar.slider("Group B Mean (μ₂)", 80.0, 160.0, 108.0, 1.0)
        std_a = st.sidebar.slider("Group A Standard Deviation (σ₁)", 2.0, 40.0, 12.0, 1.0)
        std_b = st.sidebar.slider(
            "Group B Standard Deviation (σ₂)",
            2.0,
            40.0,
            12.0,
            1.0,
            help="Adjust σ₂ relative to σ₁ to test the equal variance assumption (Homoscedasticity vs Heteroscedasticity)."
        )
        dist_shape = st.sidebar.selectbox(
            "Underlying Distribution Shape",
            options=["normal", "skewed_moderate", "heavy_tailed"],
            format_func=lambda x: {
                "normal": "Normal (Bell Curve) — Satisfies Assumption",
                "skewed_moderate": "Moderately Right-Skewed (Gamma)",
                "heavy_tailed": "Heavy-Tailed (Student's t, df=3)"
            }[x],
            index=0
        )
        outlier_count_b = st.sidebar.selectbox(
            "Extreme Outliers in Group B (>3×IQR)",
            options=[0, 1, 2, 3],
            index=0,
            help="Inject extreme outliers into Group B to test Tukey's IQR assumption check."
        )
        seed = st.sidebar.number_input(
            "Random Seed",
            min_value=1,
            max_value=99999,
            value=42,
            step=1
        )
    
        df_demo, stats_res = generate_students_t_demo(
            n_a=n_a,
            n_b=n_b,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            dist_shape=dist_shape,
            outlier_count_b=outlier_count_b,
            seed=seed
        )
        stats_res["preset_name"] = f"Student t-test (Shape = {dist_shape}, N = {n_a + n_b})"
        return df_demo, stats_res
    
    elif demo_type in ["paired_t_test", "paired_continuous"]:
        st.sidebar.header("🧪 Paired t-Test Demo Settings")
        n_pairs = st.sidebar.slider("Number of Matched Pairs (n)", 10, 100, 35)
        mean_pre = st.sidebar.slider("Baseline / Pre Mean", 50.0, 200.0, 140.0, 1.0)
        mean_diff = st.sidebar.slider("Mean Change / Effect (Post - Pre)", -30.0, 30.0, -8.5, 0.5)
        std_diff = st.sidebar.slider("Std Dev of Difference", 1.0, 25.0, 6.0, 0.5)
        dist_shape = st.sidebar.selectbox(
            "Difference Distribution Shape",
            options=["normal", "skewed_moderate", "heavy_tailed"],
            format_func=lambda x: {
                "normal": "Normal Differences — Satisfies Assumption",
                "skewed_moderate": "Moderately Skewed Differences (Gamma)",
                "heavy_tailed": "Heavy-Tailed Differences (Student's t, df=3)"
            }[x],
            index=0
        )
        outlier_count = st.sidebar.selectbox(
            "Extreme Outliers in Differences (>3×IQR)",
            options=[0, 1, 2, 3],
            index=0,
            help="Inject extreme outlier differences to test Tukey's IQR assumption check."
        )
        seed = st.sidebar.number_input("Random Seed", 1, 99999, 42, 1)
    
        df_demo, stats_res = generate_paired_demo(
            n_pairs=n_pairs,
            mean_pre=mean_pre,
            mean_diff=mean_diff,
            std_diff=std_diff,
            dist_shape=dist_shape,
            outlier_count=outlier_count,
            seed=seed
        )
        stats_res["preset_name"] = f"Paired t-test (Shape = {dist_shape}, N = {n_pairs} pairs)"
        return df_demo, stats_res
    
    elif demo_type == "wilcoxon_paired":
        st.sidebar.header("🧪 Paired Wilcoxon Signed-Rank Settings")
        data_type = st.sidebar.selectbox(
            "Measurement Data Type",
            options=["continuous_non_normal", "ordinal_likert"],
            format_func=lambda x: {
                "continuous_non_normal": "Continuous Non-Normal (e.g. Biomarkers, Reaction Times)",
                "ordinal_likert": "Ordinal Rating Scale (e.g. 7-point Likert Pre vs Post)"
            }[x],
            index=0,
            help="The Wilcoxon Signed-Rank Test applies to both continuous non-normal and ordinal paired measures."
        )
        n_pairs = st.sidebar.slider("Number of Matched Pairs (n)", 10, 100, 30)
        median_pre = st.sidebar.slider("Pre / Baseline Median", 50.0, 200.0, 140.0, 1.0)
        median_diff = st.sidebar.slider("Median Shift (Post - Pre)", -25.0, 25.0, -6.0, 0.5)
        scale_diff = st.sidebar.slider("Dispersion / Scale of Differences", 2.0, 25.0, 8.0, 0.5)
        dist_shape = st.sidebar.selectbox(
            "Distribution Shape & Symmetry of Differences",
            options=["symmetric_non_normal", "moderately_skewed", "severely_skewed"],
            format_func=lambda x: {
                "symmetric_non_normal": "Symmetric Non-Normal (Laplace - Ideal for Wilcoxon)",
                "moderately_skewed": "Moderately Skewed (Gamma - Hodges-Lehmann Pseudo-Median)",
                "severely_skewed": "Severely Skewed (Log-Normal - Triggers Exact Sign Test)"
            }[x],
            index=0,
            help="Test the symmetry diagnostic on pairwise differences."
        )
        zero_diff_count = st.sidebar.selectbox(
            "Pairs with Zero Difference (Post == Pre)",
            options=[0, 1, 2, 3, 5],
            format_func=lambda x: f"{x} ties (dropped from ranking)" if x > 0 else "0 (No ties)",
            index=0,
            help="Tests how zero differences (ties where subject did not change) are handled."
        )
        seed = st.sidebar.number_input("Random Seed", 1, 99999, 42, 1)
    
        df_demo, stats_res = generate_wilcoxon_paired_demo(
            n_pairs=n_pairs,
            median_pre=median_pre,
            median_diff=median_diff,
            scale_diff=scale_diff,
            data_type=data_type,
            dist_shape=dist_shape,
            zero_diff_count=zero_diff_count,
            seed=seed
        )
        stats_res["preset_name"] = f"Wilcoxon Paired (Shape = {dist_shape}, N = {n_pairs} pairs)"
        return df_demo, stats_res
    
    elif demo_type in ["one_way_anova", "three_independent_continuous"]:
        st.sidebar.header("🧪 One-Way ANOVA Demo Settings")
        n_grp = st.sidebar.slider("Sample Size per Group (n)", 10, 100, 30)
    
        st.sidebar.markdown("##### 🎯 Target Group Means")
        m1 = st.sidebar.slider("Group 1 Mean (x̄₁)", 30.0, 90.0, 50.0, 0.5)
        m2 = st.sidebar.slider("Group 2 Mean (x̄₂)", 30.0, 90.0, 58.0, 0.5)
        m3 = st.sidebar.slider("Group 3 Mean (x̄₃)", 30.0, 90.0, 65.0, 0.5)
    
        st.sidebar.markdown("##### 📏 Group Standard Deviations (Variances)")
        col_s1, col_s2, col_s3 = st.sidebar.columns(3)
        s1 = col_s1.slider("Std Dev 1", 2.0, 30.0, 8.0, 0.5)
        s2 = col_s2.slider("Std Dev 2", 2.0, 30.0, 8.0, 0.5)
        s3 = col_s3.slider("Std Dev 3", 2.0, 30.0, 8.0, 0.5)
    
        st.sidebar.markdown("##### 🛡️ Assumption Testing Scenarios")
        dist_shape = st.sidebar.selectbox(
            "Distribution Shape",
            options=["normal", "skewed_moderate", "heavy_tailed"],
            format_func=lambda x: {
                "normal": "Normal (Bell Curve) — Satisfies Assumption",
                "skewed_moderate": "Moderately Skewed (Gamma)",
                "heavy_tailed": "Heavy-Tailed (Student's t, df=3)"
            }[x],
            index=0,
            help="Select skewed or heavy-tailed distributions to test normality diagnostics."
        )
        outlier_count_g3 = st.sidebar.selectbox(
            "Extreme Outliers in Group 3 (>3×IQR)",
            options=[0, 1, 2, 3],
            index=0,
            help="Inject extreme outliers into Group 3 to test Tukey's IQR assumption check."
        )
        seed = st.sidebar.number_input("Random Seed", 1, 99999, 42, 1)
    
        df_demo, stats_res = generate_one_way_anova_demo(
            n_per_group=n_grp,
            means=(m1, m2, m3),
            stds=(s1, s2, s3),
            dist_shape=dist_shape,
            outlier_count_g3=outlier_count_g3,
            seed=seed
        )
        stats_res["preset_name"] = f"One-Way ANOVA (Shape = {dist_shape}, N = {n_g1 + n_g2 + n_g3})"
        return df_demo, stats_res
    
    elif demo_type in ["kruskal_wallis", "three_independent_ordinal"]:
        st.sidebar.header("🧪 Kruskal-Wallis Demo Settings")
        n_grp = st.sidebar.slider("Sample Size per Group (n)", 10, 100, 25)
    
        data_scale = st.sidebar.selectbox(
            "Measurement Scale",
            options=["continuous", "ordinal_likert"],
            format_func=lambda x: {
                "continuous": "Continuous Scores (e.g. Recovery Score)",
                "ordinal_likert": "Ordinal Likert Ratings (1 to 7 Scale)"
            }[x],
            index=0,
            help="Select Ordinal Likert to test mid-rank calculation with tied observations."
        )
    
        st.sidebar.markdown("##### 🎯 Group Location Shifts / Median Deltas")
        shift1 = st.sidebar.slider("Group 1 Shift (Δ₁)", -10.0, 20.0, 0.0, 0.5)
        shift2 = st.sidebar.slider("Group 2 Shift (Δ₂)", -10.0, 20.0, 4.0, 0.5)
        shift3 = st.sidebar.slider("Group 3 Shift (Δ₃)", -10.0, 20.0, 9.0, 0.5)
    
        st.sidebar.markdown("##### 🛡️ Distribution Shape Scenarios")
        dist_shape = st.sidebar.selectbox(
            "Underlying Distribution Shape",
            options=["skewed_moderate", "heavy_tailed", "normal", "unequal_shapes"],
            format_func=lambda x: {
                "skewed_moderate": "Moderately Right-Skewed (Gamma) — Standard Non-Normal",
                "heavy_tailed": "Heavy-Tailed (Student's t, df=3)",
                "normal": "Normal (Bell Curve) — Test Parametric ANOVA Contrast",
                "unequal_shapes": "Unequal Shapes Across Groups (Tests Stochastic Dominance)"
            }[x],
            index=0,
            help="Unequal shapes demonstrate when Kruskal-Wallis tests stochastic dominance rather than medians."
        )
        seed = st.sidebar.number_input("Random Seed", 1, 99999, 42, 1)
    
        df_demo, stats_res = generate_kruskal_wallis_demo(
            n_per_group=n_grp,
            shifts=(shift1, shift2, shift3),
            data_type=data_scale,
            dist_shape=dist_shape,
            seed=seed
        )
        stats_res["preset_name"] = f"Kruskal-Wallis (Scale = {data_scale}, Shape = {dist_shape})"
        return df_demo, stats_res
    
    elif demo_type in ["repeated_measures_anova", "three_repeated_continuous"]:
        st.sidebar.header("🧪 Repeated Measures ANOVA Settings")
        n_subj = st.sidebar.slider("Number of Matched Subjects (n)", 10, 80, 25, 5, help="Number of individuals tracked across all 3 repeated occasions.")
    
        st.sidebar.markdown("##### 🎯 Target Condition Means")
        m1 = st.sidebar.slider("Baseline Mean (x̄₁)", 30.0, 90.0, 55.0, 0.5)
        m2 = st.sidebar.slider("Month 1 Mean (x̄₂)", 30.0, 90.0, 62.0, 0.5)
        m3 = st.sidebar.slider("Month 3 Mean (x̄₃)", 30.0, 90.0, 71.0, 0.5)
    
        st.sidebar.markdown("##### 📏 Variance Components")
        col_v1, col_v2 = st.sidebar.columns(2)
        subj_sd = col_v1.slider("Subject SD (Between)", 2.0, 25.0, 10.0, 0.5, help="Random subject intercept variability.")
        within_sd = col_v2.slider("Within SD (Noise)", 1.0, 15.0, 4.0, 0.5, help="Measurement error across occasions.")
    
        st.sidebar.markdown("##### 🛡️ Assumption Testing Scenarios")
        dist_shape = st.sidebar.selectbox(
            "Distribution Shape",
            options=["normal", "skewed_moderate", "heavy_tailed"],
            format_func=lambda x: {
                "normal": "Normal (Bell Curve) — Satisfies Assumption",
                "skewed_moderate": "Moderately Skewed (Gamma)",
                "heavy_tailed": "Heavy-Tailed (Student's t, df=3)"
            }[x],
            index=0,
            help="Select skewed or heavy-tailed distributions to test normality diagnostics on differences."
        )
        sphericity_violation = st.sidebar.checkbox(
            "Violate Sphericity Assumption",
            value=False,
            help="Forces unequal variances across pairwise difference distributions to test Mauchly's test and Greenhouse-Geisser correction."
        )
        outlier_count = st.sidebar.selectbox(
            "Inject Extreme Within-Subject Outlier",
            options=[0, 1],
            format_func=lambda x: {
                0: "0 - None (Clean)",
                1: "1 Extreme Outlier (+4.5σ at Month 3)"
            }[x],
            index=0,
            help="Inject an extreme spike into the final condition to test Tukey's IQR outlier diagnostic."
        )
        seed = st.sidebar.number_input("Random Seed", 1, 99999, 42, 1)
    
        df_demo, stats_res = generate_repeated_measures_anova_demo(
            n_subjects=n_subj,
            means=(m1, m2, m3),
            subject_sd=subj_sd,
            within_sd=within_sd,
            sphericity_violation=sphericity_violation,
            dist_shape=dist_shape,
            outlier_count=outlier_count,
            seed=seed
        )
        stats_res["preset_name"] = f"Repeated Measures ANOVA (Shape = {dist_shape}, N = {n_subj} subjects)"
        return df_demo, stats_res
    
    elif demo_type in ["friedman_test", "repeated_measures_ordinal", "repeated_measures_continuous"]:
        st.sidebar.header("🧪 Friedman Test Settings")
        n_subj = st.sidebar.slider("Number of Matched Subjects (n)", 10, 80, 25, 5, help="Number of individuals tracked across all 3 repeated occasions.")
    
        data_scale = st.sidebar.selectbox(
            "Measurement Scale",
            options=["ordinal_likert", "continuous"],
            format_func=lambda x: {
                "ordinal_likert": "Ordinal Likert Ratings (1 to 7 Scale)",
                "continuous": "Continuous Scores (Skewed / Non-Normal)"
            }[x],
            index=0,
            help="Select Ordinal Likert to test within-subject mid-rank calculation with tied ratings."
        )
    
        st.sidebar.markdown("##### 🎯 Condition Location Shifts / Deltas")
        shift1 = st.sidebar.slider("Condition 1 Shift (Δ₁)", -10.0, 15.0, 0.0, 0.5)
        shift2 = st.sidebar.slider("Condition 2 Shift (Δ₂)", -10.0, 15.0, 3.0, 0.5)
        shift3 = st.sidebar.slider("Condition 3 Shift (Δ₃)", -10.0, 15.0, 6.5, 0.5)
    
        st.sidebar.markdown("##### 🛡️ Latent Distribution Shape")
        dist_shape = st.sidebar.selectbox(
            "Latent Noise Distribution",
            options=["skewed_moderate", "heavy_tailed", "normal"],
            format_func=lambda x: {
                "skewed_moderate": "Moderately Right-Skewed (Gamma) — Standard Non-Normal",
                "heavy_tailed": "Heavy-Tailed (Student's t, df=3)",
                "normal": "Normal (Bell Curve) — Test RM-ANOVA Contrast"
            }[x],
            index=0
        )
        seed = st.sidebar.number_input("Random Seed", 1, 99999, 42, 1)
    
        df_demo, stats_res = generate_friedman_demo(
            n_subjects=n_subj,
            shifts=(shift1, shift2, shift3),
            data_type=data_scale,
            dist_shape=dist_shape,
            seed=seed
        )
        stats_res["preset_name"] = f"Friedman Test (Scale = {data_scale}, Shape = {dist_shape}, N = {n_subj} subjects)"
        return df_demo, stats_res
    
    elif demo_type in ["contingency_table_2x2", "chi_square_independence"]:
        st.sidebar.header("🧪 2×2 Contingency Table Settings")
        st.sidebar.markdown("Configure cross-classification counts and clinical labels:")
    
        preset = st.sidebar.selectbox(
            "Clinical Scenario Presets",
            options=[
                "Standard Trial (Significant Association, E ≥ 5)",
                "Strong Exposure Effect (High OR, E ≥ 5)",
                "Null Association (Proportions Equal, E ≥ 5)",
                "Small Sample / Rare Event (E < 5, Triggers Fisher's Exact)",
                "Custom Cell Counts"
            ],
            index=0,
            help="Quickly load archetypal clinical and research scenarios."
        )
    
        if preset == "Standard Trial (Significant Association, E ≥ 5)":
            def_a, def_b, def_c, def_d = 45, 55, 20, 80
            def_g1, def_g2 = "Treatment Group", "Control Group"
            def_out_pos, def_out_neg = "Recovered (+)", "Not Recovered (-)"
        elif preset == "Strong Exposure Effect (High OR, E ≥ 5)":
            def_a, def_b, def_c, def_d = 70, 30, 15, 85
            def_g1, def_g2 = "High Exposure", "Unexposed"
            def_out_pos, def_out_neg = "Disease (+)", "Healthy (-)"
        elif preset == "Null Association (Proportions Equal, E ≥ 5)":
            def_a, def_b, def_c, def_d = 35, 65, 36, 64
            def_g1, def_g2 = "Cohort A", "Cohort B"
            def_out_pos, def_out_neg = "Response (+)", "No Response (-)"
        elif preset == "Small Sample / Rare Event (E < 5, Triggers Fisher's Exact)":
            def_a, def_b, def_c, def_d = 4, 16, 1, 19
            def_g1, def_g2 = "Drug Experimental", "Standard Care"
            def_out_pos, def_out_neg = "Adverse Event (+)", "No Adverse Event (-)"
        else:
            def_a, def_b, def_c, def_d = 45, 55, 20, 80
            def_g1, def_g2 = "Group 1", "Group 2"
            def_out_pos, def_out_neg = "Outcome (+)", "Outcome (-)"
    
        st.sidebar.markdown("##### 🏷️ Category Labels")
        g1_label = st.sidebar.text_input("Row 1 (Group 1 Label)", value=def_g1)
        g2_label = st.sidebar.text_input("Row 2 (Group 2 Label)", value=def_g2)
        pos_label = st.sidebar.text_input("Col 1 (Positive Outcome)", value=def_out_pos)
        neg_label = st.sidebar.text_input("Col 2 (Negative Outcome)", value=def_out_neg)
    
        st.sidebar.markdown(f"##### 🔢 Cell Frequencies for {g1_label}")
        exp_cases = st.sidebar.slider(f"{g1_label} × {pos_label} (Cell a)", 0, 300, def_a, 1)
        exp_noncases = st.sidebar.slider(f"{g1_label} × {neg_label} (Cell b)", 0, 300, def_b, 1)
    
        st.sidebar.markdown(f"##### 🔢 Cell Frequencies for {g2_label}")
        unexp_cases = st.sidebar.slider(f"{g2_label} × {pos_label} (Cell c)", 0, 300, def_c, 1)
        unexp_noncases = st.sidebar.slider(f"{g2_label} × {neg_label} (Cell d)", 0, 300, def_d, 1)
    
        df_demo, stats_res = generate_contingency_demo(
            exposed_cases=exp_cases,
            exposed_noncases=exp_noncases,
            unexposed_cases=unexp_cases,
            unexposed_noncases=unexp_noncases,
            group1_name=g1_label,
            group2_name=g2_label,
            outcome_pos_name=pos_label,
            outcome_neg_name=neg_label
        )
        stats_res["preset_name"] = preset
        return df_demo, stats_res
    
    elif demo_type == "fisher_exact":
        st.sidebar.header("🧪 Fisher's Exact Test Settings")
        st.sidebar.markdown("Configure 2×2 cross-classification counts for small-sample exact testing:")
    
        preset = st.sidebar.selectbox(
            "Clinical & Experimental Presets",
            options=[
                "Rare Adverse Event (Classic Safety Signal: 4/20 vs 1/20)",
                "Small Pilot RCT (Efficacy Signal with Small Cell: 7/12 vs 2/13)",
                "Zero-Cell Event (Complete Separation: 5/15 vs 0/15)",
                "Equal Margins Symmetrical Table (Null Effect: 3/10 vs 3/10)",
                "Custom Cell Counts"
            ],
            index=0,
            help="Pre-configured small-sample scenarios demonstrating exact hypergeometric probabilities and Cochran's rule."
        )
    
        if preset == "Rare Adverse Event (Classic Safety Signal: 4/20 vs 1/20)":
            def_a, def_b, def_c, def_d = 4, 16, 1, 19
            def_g1, def_g2 = "Experimental Drug", "Standard Control"
            def_out_pos, def_out_neg = "Adverse Event (+)", "No Event (-)"
        elif preset == "Small Pilot RCT (Efficacy Signal with Small Cell: 7/12 vs 2/13)":
            def_a, def_b, def_c, def_d = 7, 5, 2, 11
            def_g1, def_g2 = "Novel Therapy", "Placebo"
            def_out_pos, def_out_neg = "Clinical Response (+)", "No Response (-)"
        elif preset == "Zero-Cell Event (Complete Separation: 5/15 vs 0/15)":
            def_a, def_b, def_c, def_d = 5, 10, 0, 15
            def_g1, def_g2 = "High Exposure", "Unexposed"
            def_out_pos, def_out_neg = "Toxicity (+)", "Unaffected (-)"
        elif preset == "Equal Margins Symmetrical Table (Null Effect: 3/10 vs 3/10)":
            def_a, def_b, def_c, def_d = 3, 7, 3, 7
            def_g1, def_g2 = "Batch A", "Batch B"
            def_out_pos, def_out_neg = "Defect (+)", "Pass (-)"
        else:
            def_a, def_b, def_c, def_d = 4, 16, 1, 19
            def_g1, def_g2 = "Group 1", "Group 2"
            def_out_pos, def_out_neg = "Outcome (+)", "Outcome (-)"
    
        st.sidebar.markdown("##### 🏷️ Category Labels")
        g1_label = st.sidebar.text_input("Row 1 (Group 1 Label)", value=def_g1, key="fe_g1")
        g2_label = st.sidebar.text_input("Row 2 (Group 2 Label)", value=def_g2, key="fe_g2")
        pos_label = st.sidebar.text_input("Col 1 (Positive Outcome)", value=def_out_pos, key="fe_pos")
        neg_label = st.sidebar.text_input("Col 2 (Negative Outcome)", value=def_out_neg, key="fe_neg")
    
        st.sidebar.markdown(f"##### 🔢 Cell Frequencies for {g1_label}")
        cell_a = st.sidebar.slider(f"{g1_label} × {pos_label} (Cell a)", 0, 80, def_a, 1, key="fe_a")
        cell_b = st.sidebar.slider(f"{g1_label} × {neg_label} (Cell b)", 0, 80, def_b, 1, key="fe_b")
    
        st.sidebar.markdown(f"##### 🔢 Cell Frequencies for {g2_label}")
        cell_c = st.sidebar.slider(f"{g2_label} × {pos_label} (Cell c)", 0, 80, def_c, 1, key="fe_c")
        cell_d = st.sidebar.slider(f"{g2_label} × {neg_label} (Cell d)", 0, 80, def_d, 1, key="fe_d")
    
        alpha_val = st.sidebar.selectbox(
            "Significance Level (α)",
            options=[0.05, 0.01, 0.10],
            index=0,
            format_func=lambda x: f"α = {x:.2f} ({int((1-x)*100)}% Confidence Level)",
            key="fe_alpha"
        )
    
        df_demo, stats_res = generate_fisher_exact_demo(
            a=cell_a,
            b=cell_b,
            c=cell_c,
            d=cell_d,
            group1_name=g1_label,
            group2_name=g2_label,
            outcome_pos_name=pos_label,
            outcome_neg_name=neg_label,
            alpha=alpha_val
        )
        stats_res["preset_name"] = preset
        return df_demo, stats_res
    
    
    elif demo_type in ["mcnemar_test", "paired_binary"]:
        st.sidebar.header("🧪 McNemar's Test Demo Settings")
        st.sidebar.caption("Simulate paired binary repeated-measures data across two time points or matched conditions.")
    
        preset = st.sidebar.selectbox(
            "Demonstration Presets",
            options=[
                "Clinical Trial: Pre-Post Treatment Response (Significant Improvement: 15/5 vs 25/55)",
                "Diagnostic Test Comparison (Small Sample: Exact Binomial Required: 8/2 vs 9/11)",
                "Matched Case-Control Study (Smoking Concordance: 32/12 vs 28/28)",
                "Symmetrical Discordance (Null Effect / Equal Shifts: 20/12 vs 12/36)",
                "Custom Paired Frequencies"
            ],
            index=0,
            key="mc_preset"
        )
    
        if preset == "Clinical Trial: Pre-Post Treatment Response (Significant Improvement: 15/5 vs 25/55)":
            def_a, def_b, def_c, def_d = 15, 5, 25, 55
            def_t1, def_t2 = "Pre-Treatment (Baseline)", "Post-Treatment (Week 12)"
            def_pos, def_neg = "Responder (+)", "Non-Responder (-)"
        elif preset == "Diagnostic Test Comparison (Small Sample: Exact Binomial Required: 8/2 vs 9/11)":
            def_a, def_b, def_c, def_d = 8, 2, 9, 11
            def_t1, def_t2 = "Diagnostic Test A", "Diagnostic Test B"
            def_pos, def_neg = "Positive (+)", "Negative (-)"
        elif preset == "Matched Case-Control Study (Smoking Concordance: 32/12 vs 28/28)":
            def_a, def_b, def_c, def_d = 32, 12, 28, 28
            def_t1, def_t2 = "Case Subject (Lung Cancer)", "Matched Control Subject"
            def_pos, def_neg = "Smoker (+)", "Non-Smoker (-)"
        elif preset == "Symmetrical Discordance (Null Effect / Equal Shifts: 20/12 vs 12/36)":
            def_a, def_b, def_c, def_d = 20, 12, 12, 36
            def_t1, def_t2 = "Timepoint 1", "Timepoint 2"
            def_pos, def_neg = "Endorsed (+)", "Not Endorsed (-)"
        else:
            def_a, def_b, def_c, def_d = 15, 5, 25, 55
            def_t1, def_t2 = "Occasion 1", "Occasion 2"
            def_pos, def_neg = "State (+)", "State (-)"
    
        st.sidebar.markdown("##### 🏷️ Category & Condition Labels")
        t1_label = st.sidebar.text_input("Row Dimension (Occasion 1 / Pre)", value=def_t1, key="mc_t1")
        t2_label = st.sidebar.text_input("Column Dimension (Occasion 2 / Post)", value=def_t2, key="mc_t2")
        pos_label = st.sidebar.text_input("Positive / Favorable Outcome (+)", value=def_pos, key="mc_pos")
        neg_label = st.sidebar.text_input("Negative / Unfavorable Outcome (-)", value=def_neg, key="mc_neg")
    
        st.sidebar.markdown("##### 🔢 Concordant Pairs (Main Diagonal)")
        cell_a = st.sidebar.slider(f"Both {pos_label} (Cell a: +/+)", 0, 100, def_a, 1, key="mc_a", help="Concordant positive on both occasions.")
        cell_d = st.sidebar.slider(f"Both {neg_label} (Cell d: -/-)", 0, 100, def_d, 1, key="mc_d", help="Concordant negative on both occasions.")
    
        st.sidebar.markdown("##### ⚡ Discordant Pairs (Off-Diagonal Switchers)")
        cell_b = st.sidebar.slider(f"{t1_label} (+) → {t2_label} (-) [Cell b: +/-]", 0, 100, def_b, 1, key="mc_b", help="Switched from positive to negative.")
        cell_c = st.sidebar.slider(f"{t1_label} (-) → {t2_label} (+) [Cell c: -/+]", 0, 100, def_c, 1, key="mc_c", help="Switched from negative to positive.")
    
        alpha_val = st.sidebar.selectbox(
            "Significance Level (α)",
            options=[0.05, 0.01, 0.10],
            index=0,
            format_func=lambda x: f"α = {x:.2f} ({int((1-x)*100)}% Confidence Level)",
            key="mc_alpha"
        )
    
        df_demo, stats_res = generate_mcnemar_demo(
            a=cell_a,
            b=cell_b,
            c=cell_c,
            d=cell_d,
            time1_name=t1_label,
            time2_name=t2_label,
            outcome_pos_name=pos_label,
            outcome_neg_name=neg_label,
            alpha=alpha_val
        )
        stats_res["preset_name"] = preset
        return df_demo, stats_res
    
    
    elif demo_type in ["pearson_correlation", "correlation_continuous"]:
        st.sidebar.header("🧪 Pearson Correlation Demo Settings")
        st.sidebar.caption("Simulate bivariate continuous data, evaluate correlation strength (r), shared variance (R²), and test assumptions.")
    
        preset = st.sidebar.selectbox(
            "Demonstration Presets",
            options=[
                "Strong Positive Linear (Study Hours vs Exam Score: r ≈ 0.82, R² ≈ 67%)",
                "Moderate Negative Linear (Screen Time vs Sleep Quality: r ≈ -0.58, R² ≈ 34%)",
                "Zero / Null Correlation (Shoe Size vs IQ Score: r ≈ 0.05, R² ≈ 0%)",
                "Non-Linear Curvature / Diminishing Returns (Practice Hours vs Skill: r ≈ 0.65, ρ ≈ 0.90)",
                "Influential Leverage Outlier / Anscombe Effect (Outlier Distorting Fit)",
                "Custom Bivariate Parameters"
            ],
            index=0,
            key="pearson_preset"
        )
    
        if preset != "Custom Bivariate Parameters":
            st.sidebar.caption("💡 *Preset active:* The data follows this preset's design. Adjust **Residual Noise** or **Sample Size** to change significance, or select **'Custom'** above to freely set correlation r.")
        else:
            st.sidebar.caption("🛠️ *Custom mode active:* All sliders below directly control the data generation, correlation r, and residual noise.")
    
        if preset == "Strong Positive Linear (Study Hours vs Exam Score: r ≈ 0.82, R² ≈ 67%)":
            scen = "normal_linear"
            def_n = 60
            def_r = 0.82
            def_noise = 1.0
            def_x = "Study Time (Hours / Week)"
            def_y = "Exam Score (%)"
        elif preset == "Moderate Negative Linear (Screen Time vs Sleep Quality: r ≈ -0.58, R² ≈ 34%)":
            scen = "moderate_negative"
            def_n = 60
            def_r = -0.58
            def_noise = 1.0
            def_x = "Evening Screen Time (Hours)"
            def_y = "Sleep Quality Index (0-100)"
        elif preset == "Zero / Null Correlation (Shoe Size vs IQ Score: r ≈ 0.05, R² ≈ 0%)":
            scen = "null_correlation"
            def_n = 60
            def_r = 0.05
            def_noise = 1.0
            def_x = "Shoe Size (EU)"
            def_y = "IQ Score"
        elif preset == "Non-Linear Curvature / Diminishing Returns (Practice Hours vs Skill: r ≈ 0.65, ρ ≈ 0.90)":
            scen = "non_linear_curvature"
            def_n = 60
            def_r = 0.65
            def_noise = 0.6
            def_x = "Cumulative Practice (Hours)"
            def_y = "Skill Proficiency Index"
        elif preset == "Influential Leverage Outlier / Anscombe Effect (Outlier Distorting Fit)":
            scen = "outlier_leverage"
            def_n = 45
            def_r = 0.40
            def_noise = 1.0
            def_x = "Advertising Expenditure ($k)"
            def_y = "Quarterly Sales Volume ($k)"
        else:
            scen = "custom"
            def_n = 60
            def_r = 0.70
            def_noise = 1.0
            def_x = "Predictor Variable X"
            def_y = "Outcome Variable Y"
    
        st.sidebar.markdown("##### 🏷️ Variable Labels")
        var_x = st.sidebar.text_input("Predictor Variable (X)", value=def_x, key="pearson_x")
        var_y = st.sidebar.text_input("Outcome Variable (Y)", value=def_y, key="pearson_y")
    
        st.sidebar.markdown("##### 🔢 Bivariate Distribution Parameters")
        n_pts = st.sidebar.slider("Sample Size (N pairs)", min_value=20, max_value=300, value=def_n, step=5, key="pearson_n", help="Number of paired observations.")
        true_r = st.sidebar.slider("Population Correlation (r)", min_value=-0.99, max_value=0.99, value=def_r, step=0.01, key="pearson_r", help="True underlying linear correlation coefficient.")
        noise_lvl = st.sidebar.slider("Residual Noise Level", min_value=0.2, max_value=3.0, value=def_noise, step=0.1, key="pearson_noise", help="Error variance multiplier.")
    
        alpha_val = st.sidebar.selectbox(
            "Significance Level (α)",
            options=[0.05, 0.01, 0.10],
            index=0,
            format_func=lambda x: f"α = {x:.2f} ({int((1-x)*100)}% Confidence Level)",
            key="pearson_alpha"
        )
    
        seed_val = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1, key="pearson_seed")
    
        df_demo, stats_res = generate_pearson_correlation_demo(
            n_pts=n_pts,
            true_r=true_r,
            var_x_name=var_x,
            var_y_name=var_y,
            scenario=scen,
            noise_level=noise_lvl,
            alpha=alpha_val,
            seed=seed_val
        )
        stats_res["preset_name"] = preset
        return df_demo, stats_res
    
    
    elif demo_type in ["spearman_correlation", "spearman_rank"]:
        st.sidebar.header("🧪 Spearman Rank Correlation Demo Settings")
        st.sidebar.caption("Simulate monotonic data, evaluate rank transformations (ρ), rank displacement (Σd²), and test non-parametric assumptions.")
    
        preset = st.sidebar.selectbox(
            "Demonstration Presets",
            options=[
                "Non-Linear Diminishing Returns (Practice Hours vs Mastery: ρ ≈ 0.92, r ≈ 0.68)",
                "Ordinal Survey Likert Scales (Job Satisfaction vs Retention Intent: ρ ≈ 0.74, Ties Present)",
                "Strong Inverse Monotonic (Exponential Forgetting Curve: ρ ≈ -0.88)",
                "Distorting Outliers (Anscombe-Style Leverage Outliers: ρ ≈ 0.78, r ≈ 0.22)",
                "Zero / Null Association (Random Rank Ordering: ρ ≈ 0.04)",
                "Custom Non-Parametric Parameters"
            ],
            index=0,
            key="spearman_preset"
        )
    
        if preset != "Custom Non-Parametric Parameters":
            st.sidebar.caption("💡 *Preset active:* The data follows this preset's design. Adjust **Residual Noise** or **Sample Size** to change significance, or select **'Custom'** above to freely set target ρ.")
        else:
            st.sidebar.caption("🛠️ *Custom mode active:* All sliders below directly control the data generation, correlation strength, noise, and ties.")
    
        if preset == "Non-Linear Diminishing Returns (Practice Hours vs Mastery: ρ ≈ 0.92, r ≈ 0.68)":
            scen = "non_linear_monotonic"
            def_n = 50
            def_rho = 0.85
            def_noise = 1.0
            def_ties = 0.0
            def_x = "Cumulative Practice (Hours)"
            def_y = "Skill Proficiency Level"
        elif preset == "Ordinal Survey Likert Scales (Job Satisfaction vs Retention Intent: ρ ≈ 0.74, Ties Present)":
            scen = "ordinal_likert"
            def_n = 60
            def_rho = 0.75
            def_noise = 1.0
            def_ties = 0.35
            def_x = "Job Satisfaction (1-7 Likert)"
            def_y = "Retention Intent (1-7 Likert)"
        elif preset == "Strong Inverse Monotonic (Exponential Forgetting Curve: ρ ≈ -0.88)":
            scen = "inverse_monotonic"
            def_n = 45
            def_rho = -0.85
            def_noise = 1.0
            def_ties = 0.0
            def_x = "Time Since Learning (Days)"
            def_y = "Information Recall Rate (%)"
        elif preset == "Distorting Outliers (Anscombe-Style Leverage Outliers: ρ ≈ 0.78, r ≈ 0.22)":
            scen = "extreme_outliers"
            def_n = 40
            def_rho = 0.75
            def_noise = 1.0
            def_ties = 0.0
            def_x = "Marketing Investment ($k)"
            def_y = "Sales Conversion ($k)"
        elif preset == "Zero / Null Association (Random Rank Ordering: ρ ≈ 0.04)":
            scen = "null_rank"
            def_n = 50
            def_rho = 0.0
            def_noise = 1.0
            def_ties = 0.0
            def_x = "Participant Age"
            def_y = "Lottery Ticket Number"
        else:
            scen = "custom"
            def_n = 50
            def_rho = 0.70
            def_noise = 1.0
            def_ties = 0.0
            def_x = "Predictor Variable X"
            def_y = "Outcome Variable Y"
    
        st.sidebar.markdown("##### 🏷️ Variable Labels")
        var_x = st.sidebar.text_input("Predictor / Rank 1 Variable (X)", value=def_x, key="spearman_x")
        var_y = st.sidebar.text_input("Outcome / Rank 2 Variable (Y)", value=def_y, key="spearman_y")
    
        st.sidebar.markdown("##### 🔢 Non-Parametric Distribution Parameters")
        n_pts = st.sidebar.slider("Sample Size (N pairs)", min_value=15, max_value=250, value=def_n, step=5, key="spearman_n", help="Number of paired observations.")
        target_rho = st.sidebar.slider("Target Rank Correlation (ρ)", min_value=-0.99, max_value=0.99, value=def_rho, step=0.01, key="spearman_rho", help="Underlying monotonic rank association.")
        noise_lvl = st.sidebar.slider("Residual Noise Level", min_value=0.2, max_value=3.0, value=def_noise, step=0.1, key="spearman_noise", help="Error variance multiplier.")
        tie_freq = st.sidebar.slider("Tied Values Frequency", min_value=0.0, max_value=0.5, value=def_ties, step=0.05, format="%.2f", key="spearman_ties", help="Fraction of rounded values to induce ties.")
    
        alpha_val = st.sidebar.selectbox(
            "Significance Level (α)",
            options=[0.05, 0.01, 0.10],
            index=0,
            format_func=lambda x: f"α = {x:.2f} ({int((1-x)*100)}% Confidence Level)",
            key="spearman_alpha"
        )
    
        seed_val = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1, key="spearman_seed")
    
        df_demo, stats_res = generate_spearman_correlation_demo(
            n_pts=n_pts,
            target_rho=target_rho,
            var_x_name=var_x,
            var_y_name=var_y,
            scenario=scen,
            tie_frequency=tie_freq,
            noise_level=noise_lvl,
            alpha=alpha_val,
            seed=seed_val
        )
        stats_res["preset_name"] = preset
        return df_demo, stats_res
    
    
    elif demo_type in ["linear_regression", "multiple_linear_regression", "regression_continuous"]:
        st.sidebar.header("🧪 Linear Regression (OLS) Settings")
        st.sidebar.caption("Simulate continuous prediction models, evaluate model fit (R²), omnibus F-test, individual slope tests, multicollinearity (VIF), and residual diagnostics.")
    
        reg_preset = st.sidebar.selectbox(
            "Demonstration Presets",
            options=[
                "Academic Performance (Study Hours + Attendance + GPA → Exam Score: k=3)",
                "Real Estate Valuation (Square Footage + Bedrooms + Age → Home Price: k=3)",
                "Simple Bivariate Regression (Digital Ad Spend → Revenue: k=1)",
                "Multicollinearity Stress Test (High VIF & Correlated Predictors: k=3)",
                "Heteroscedasticity Violation (Funnel Residual Variance: k=2)",
                "Custom Regression Model (Specify Predictors & Parameters)"
            ],
            index=0,
            key="linear_reg_preset"
        )
    
        if reg_preset != "Custom Regression Model (Specify Predictors & Parameters)":
            st.sidebar.caption("💡 *Preset active:* Data follows this scenario's real-world structure. Adjust **Sample Size** or **Residual Noise** to observe shifts in p-values and R².")
        else:
            st.sidebar.caption("🛠️ *Custom mode active:* Select number of predictors and error variance freely.")
    
        if reg_preset == "Academic Performance (Study Hours + Attendance + GPA → Exam Score: k=3)":
            scen = "academic_performance"
            def_k = 3
            def_n = 80
            def_noise = 1.0
        elif reg_preset == "Real Estate Valuation (Square Footage + Bedrooms + Age → Home Price: k=3)":
            scen = "real_estate"
            def_k = 3
            def_n = 90
            def_noise = 1.0
        elif reg_preset == "Simple Bivariate Regression (Digital Ad Spend → Revenue: k=1)":
            scen = "simple_bivariate"
            def_k = 1
            def_n = 60
            def_noise = 1.0
        elif reg_preset == "Multicollinearity Stress Test (High VIF & Correlated Predictors: k=3)":
            scen = "multicollinear"
            def_k = 3
            def_n = 75
            def_noise = 1.0
        elif reg_preset == "Heteroscedasticity Violation (Funnel Residual Variance: k=2)":
            scen = "heteroscedastic"
            def_k = 2
            def_n = 80
            def_noise = 1.0
        else:
            scen = "custom"
            def_k = 2
            def_n = 70
            def_noise = 1.0
    
        if scen == "custom":
            k_val = st.sidebar.slider("Number of Predictors (k)", min_value=1, max_value=3, value=def_k, step=1, key="reg_k_preds", help="Number of explanatory independent variables.")
        else:
            k_val = def_k
    
        n_pts = st.sidebar.slider("Sample Size (N observations)", min_value=20, max_value=250, value=def_n, step=5, key="reg_n_pts", help="Total number of observations in sample.")
        noise_lvl = st.sidebar.slider("Residual Noise Level", min_value=0.2, max_value=3.0, value=def_noise, step=0.1, key="reg_noise_level", help="Multiplier scaling residual error variance (higher noise decreases R² and weakens p-values).")
    
        alpha_val = st.sidebar.selectbox(
            "Significance Level (α)",
            options=[0.01, 0.05, 0.10],
            index=1,
            format_func=lambda x: f"α = {x:.2f} ({int((1-x)*100)}% Confidence)",
            key="reg_alpha"
        )
    
        seed_val = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1, key="reg_seed")
    
        df_demo, stats_res = generate_linear_regression_demo(
            n_pts=n_pts,
            n_predictors=k_val,
            scenario=scen,
            noise_level=noise_lvl,
            alpha=alpha_val,
            seed=seed_val
        )
        stats_res["preset_name"] = reg_preset
        return df_demo, stats_res
    
    elif demo_type in ["binary_logistic_regression", "binary_classification"]:
        st.sidebar.header("🧪 Binary Logistic Regression Settings")
        st.sidebar.caption("Simulate dichotomous outcome prediction, evaluate model fit (McFadden Pseudo-R²), likelihood ratio tests, Wald z-tests, adjusted Odds Ratios, and ROC/AUC curves.")

        logit_preset = st.sidebar.selectbox(
            "Demonstration Presets",
            options=[
                "Clinical Cardiology: 10-Yr CVD Risk from Age, SBP & Smoking (k=3)",
                "Hospital Readmission: 30-Day Readmission from LOS & Comorbidities (k=2)",
                "Educational Testing: Exam Pass/Fail from Study Hours (k=1, Simple Logistic)",
                "High Multicollinearity Stress Test (Correlated Biomarkers: k=2)",
                "Sparse Events / Low EPV (Separation / Small Sample Diagnostic: k=2)",
                "Custom Logistic Model (Specify Predictors & Parameters)"
            ],
            index=0,
            key="binary_logit_preset"
        )

        if logit_preset != "Custom Logistic Model (Specify Predictors & Parameters)":
            st.sidebar.caption("💡 *Preset active:* Data follows this scenario's real-world structure. Adjust **Sample Size** or **Latent Noise** to observe shifts in Odds Ratios and p-values.")
        else:
            st.sidebar.caption("🛠️ *Custom mode active:* Select number of predictors and noise freely.")

        if logit_preset == "Clinical Cardiology: 10-Yr CVD Risk from Age, SBP & Smoking (k=3)":
            scen = "cardiology"
            def_k = 3
            def_n = 120
            def_noise = 1.0
        elif logit_preset == "Hospital Readmission: 30-Day Readmission from LOS & Comorbidities (k=2)":
            scen = "readmission"
            def_k = 2
            def_n = 100
            def_noise = 1.0
        elif logit_preset == "Educational Testing: Exam Pass/Fail from Study Hours (k=1, Simple Logistic)":
            scen = "exam_pass"
            def_k = 1
            def_n = 80
            def_noise = 1.0
        elif logit_preset == "High Multicollinearity Stress Test (Correlated Biomarkers: k=2)":
            scen = "multicollinear"
            def_k = 2
            def_n = 90
            def_noise = 1.0
        elif logit_preset == "Sparse Events / Low EPV (Separation / Small Sample Diagnostic: k=2)":
            scen = "sparse_events"
            def_k = 2
            def_n = 60
            def_noise = 1.0
        else:
            scen = "custom"
            def_k = 2
            def_n = 100
            def_noise = 1.0

        if scen == "custom":
            k_val = st.sidebar.slider("Number of Predictors (k)", min_value=1, max_value=3, value=def_k, step=1, key="logit_k_preds", help="Number of explanatory independent variables.")
        else:
            k_val = def_k

        n_pts = st.sidebar.slider("Sample Size (N observations)", min_value=30, max_value=300, value=def_n, step=5, key="logit_n_pts", help="Total number of observations in sample.")
        noise_lvl = st.sidebar.slider("Latent Noise Level", min_value=0.2, max_value=3.0, value=def_noise, step=0.1, key="logit_noise_level", help="Multiplier scaling latent logit error variance (higher noise attenuates Odds Ratios and weakens p-values).")

        alpha_val = st.sidebar.selectbox(
            "Significance Level (α)",
            options=[0.01, 0.05, 0.10],
            index=1,
            format_func=lambda x: f"α = {x:.2f} ({int((1-x)*100)}% Confidence)",
            key="logit_alpha"
        )

        seed_val = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1, key="logit_seed")

        df_demo, stats_res = generate_binary_logistic_regression_demo(
            n_pts=n_pts,
            n_predictors=k_val,
            scenario=scen,
            noise_level=noise_lvl,
            alpha=alpha_val,
            seed=seed_val
        )
        stats_res["preset_name"] = logit_preset
        return df_demo, stats_res

    else:
        st.info(
            f"ℹ️ **Interactive Simulation Status:** The live interactive simulation generator for **{recommendation['name']}** is scheduled in an upcoming implementation phase. "
        )
    

demo_type = recommendation.get("demo_generator")
df_demo, stats_res = configure_and_generate_demo_data(demo_type, recommendation)

# ----------------- Recommendation Output -----------------
st.markdown("---")
render_recommendation_card(recommendation)
render_sample_size_deep_link_badge(
    method_key=recommendation.get("test_key", ""),
    method_name=recommendation.get("name", "Statistical Test"),
    pillar="Classical Statistics",
    origin_page="pages/1_📊_Statistical_Tests.py",
    origin_title="Statistical Tests",
    origin_icon="📊"
)
render_methodology_tabs(recommendation, demo_df=df_demo, stats_results=stats_res)

# ----------------- Interactive Simulation -----------------
st.markdown("---")
if df_demo is not None and stats_res is not None:
    render_interactive_demo(demo_type, df_demo, stats_res)
else:
    st.info(
        f"ℹ️ **Interactive Simulation Status:** The live interactive simulation generator for **{recommendation['name']}** is scheduled in an upcoming implementation phase. "
        "Review the complete methodology rationale, distribution assumptions, worked Microsoft Excel guide, and ready-to-run Python code in the tabs above."
    )

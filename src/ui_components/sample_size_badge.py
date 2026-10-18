"""Unified Sample Size Deep-Link Badge Component.

Provides a standardized, 1-click biostatistical sample size and statistical power
planning launcher embedded across educational modules (Pages 1, 2, and 3)
linking directly into the matching calculation framework on Page 4.
"""

from typing import Dict, Any, Optional
import streamlit as st


# =====================================================================
# Cross-Pillar Method to Sample Size Framework Mapping
# =====================================================================

SAMPLE_SIZE_FRAMEWORK_MAP: Dict[str, Dict[str, str]] = {
    # ----------------- Pillar 1: Classical Statistical Tests -----------------
    "one_sample_t_test": {
        "custom_design": "Continuous: Two Independent Means (t-test)",
        "badge_title": "Continuous Mean Difference",
        "rationale": "Size your cohort to detect a target difference (delta) against the theoretical population benchmark at >= 80% power."
    },
    "wilcoxon_signed_rank": {
        "custom_design": "Continuous: Paired / Pre-Post Means",
        "badge_title": "Paired Non-Parametric Difference",
        "rationale": "Calculate paired sample size with a 15% Asymptotic Relative Efficiency (ARE) non-parametric sample cushion."
    },
    "welch_t_test": {
        "custom_design": "Continuous: Two Independent Means (t-test)",
        "badge_title": "Two Independent Means (t-test)",
        "rationale": "Size two independent experimental arms with Cohen's d effect size, alpha = 0.05, and unequal allocation k."
    },
    "students_t_test": {
        "custom_design": "Continuous: Two Independent Means (t-test)",
        "badge_title": "Two Independent Means (t-test)",
        "rationale": "Determine required sample size per arm assuming equal population variances under Student's t distribution."
    },
    "mann_whitney_u": {
        "custom_design": "Continuous: Two Independent Means (t-test)",
        "badge_title": "Mann-Whitney Non-Parametric Rank-Sum",
        "rationale": "Size two independent groups incorporating a Lehmann (1975) 15% non-parametric sample size cushion."
    },
    "paired_t_test": {
        "custom_design": "Continuous: Paired / Pre-Post Means",
        "badge_title": "Paired Means Difference",
        "rationale": "Calculate required matched pairs based on within-subject mean difference (delta) and correlation rho."
    },
    "wilcoxon_paired": {
        "custom_design": "Continuous: Paired / Pre-Post Means",
        "badge_title": "Paired Non-Parametric Difference",
        "rationale": "Size repeated pre-post measurements for skewed or ordinal data with a non-parametric inflation factor."
    },
    "one_way_anova": {
        "custom_design": "Continuous: One-Way ANOVA (Multiple Groups)",
        "badge_title": "One-Way ANOVA (F-test)",
        "rationale": "Size a multi-arm parallel trial (3+ groups) across Cohen's f effect sizes (0.10, 0.25, 0.40)."
    },
    "kruskal_wallis": {
        "custom_design": "Continuous: One-Way ANOVA (Multiple Groups)",
        "badge_title": "One-Way ANOVA Multi-Group",
        "rationale": "Size 3+ independent comparison groups with non-parametric rank-based adjustment."
    },
    "repeated_measures_anova": {
        "custom_design": "Continuous: Paired / Pre-Post Means",
        "badge_title": "Repeated Measures Mean Changes",
        "rationale": "Size repeated longitudinal observations across time points with Greenhouse-Geisser sphericity adjustment."
    },
    "friedman_test": {
        "custom_design": "Continuous: Paired / Pre-Post Means",
        "badge_title": "Repeated Measures Rank Differences",
        "rationale": "Size longitudinal rank-based repeated measures across multiple experimental conditions."
    },
    "chi_square_independence": {
        "custom_design": "Binary: Two Independent Proportions",
        "badge_title": "Two Independent Proportions",
        "rationale": "Determine sample size to detect a difference between two binomial event proportions (p1 vs p2) with Fleiss correction."
    },
    "fisher_exact": {
        "custom_design": "Binary: Two Independent Proportions",
        "badge_title": "Two Independent Proportions (Small Counts)",
        "rationale": "Calculate exact binomial sample requirements for low expected cell count contingency tables."
    },
    "mcnemar_test": {
        "custom_design": "Binary: Two Independent Proportions",
        "badge_title": "Matched Discordant Pairs (McNemar)",
        "rationale": "Size matched case-control or pre-post binary transitions based on the expected proportion of discordant pairs."
    },
    "pearson_correlation": {
        "custom_design": "Continuous: Pearson Correlation (r)",
        "badge_title": "Pearson Correlation (r)",
        "rationale": "Calculate required sample size via Fisher's z-transformation to detect correlation coefficient r at 80% power."
    },
    "spearman_correlation": {
        "custom_design": "Continuous: Pearson Correlation (r)",
        "badge_title": "Spearman Rank Correlation (rho)",
        "rationale": "Size monotonic continuous associations using Fisher's z normal approximation."
    },
    "linear_regression": {
        "custom_design": "Continuous: Two Independent Means (t-test)",
        "badge_title": "Linear Regression Slope & Effect",
        "rationale": "Determine sample size needed to detect significant slope coefficients or incremental R-squared increases."
    },
    "binary_logistic_regression": {
        "custom_design": "Binary: Retrospective Case-Control (Odds Ratio)",
        "badge_title": "Logistic Regression (Odds Ratio)",
        "rationale": "Size multivariable logistic regression using Hsieh (1998) equations and Events Per Variable (EPV >= 10) rules."
    },
    "log_rank_test": {
        "custom_design": "Survival: Cox Log-Rank Test (Hazard Ratio)",
        "badge_title": "Log-Rank Survival Sizing (Events E)",
        "rationale": "Determine total required failure events (E) and patient accrual N under Schoenfeld's (1981) log-rank formulation."
    },
    "cox_proportional_hazards": {
        "custom_design": "Survival: Cox Log-Rank Test (Hazard Ratio)",
        "badge_title": "Cox Proportional Hazards (Hazard Ratio)",
        "rationale": "Calculate required clinical endpoints to detect a target Hazard Ratio (HR) with multivariable covariate adjustment."
    },

    # ----------------- Pillar 2: Epidemiological Methods -----------------
    "direct_standardization": {
        "custom_design": "Continuous: Two Independent Means (t-test)",
        "badge_title": "Direct Standardized Rate (DSR) Precision",
        "rationale": "Determine minimum population denominators needed for stable age-specific rates and tight Poisson confidence intervals."
    },
    "indirect_standardization": {
        "custom_design": "Continuous: Two Independent Means (t-test)",
        "badge_title": "Standardized Mortality Ratio (SMR) Precision",
        "rationale": "Ensure sufficient expected events in a small district to prevent extreme Byar's CI volatility."
    },
    "relative_risk": {
        "custom_design": "Binary: Prospective Cohort (Relative Risk)",
        "badge_title": "Cohort Relative Risk (RR)",
        "rationale": "Size exposed and unexposed prospective cohorts to detect a clinically meaningful Relative Risk (RR) under Kelsey & Fleiss."
    },
    "odds_ratio": {
        "custom_design": "Binary: Retrospective Case-Control (Odds Ratio)",
        "badge_title": "Case-Control Odds Ratio (OR)",
        "rationale": "Determine required cases and matched controls (1:k ratio) based on control exposure prevalence p0 and target Odds Ratio."
    },
    "risk_difference": {
        "custom_design": "Binary: Two Independent Proportions",
        "badge_title": "Risk Difference & NNT",
        "rationale": "Calculate sample size to detect absolute risk reductions and quantify Number Needed to Treat (NNT = 1 / RD)."
    },
    "incidence_rate_ratio": {
        "custom_design": "Binary: Prospective Cohort (Relative Risk)",
        "badge_title": "Incidence Rate Ratio (Person-Time)",
        "rationale": "Size required person-years of follow-up to detect true differences in disease incidence rates."
    },
    "attributable_fraction": {
        "custom_design": "Binary: Prospective Cohort (Relative Risk)",
        "badge_title": "Attributable Fraction & Excess Risk",
        "rationale": "Determine cohort magnitude needed to bound the Population Attributable Fraction (PAF) within narrow confidence limits."
    },
    "diagnostic_accuracy": {
        "custom_design": "Diagnostic: Target Sensitivity Precision",
        "badge_title": "Target Sensitivity & Specificity",
        "rationale": "Calculate required diseased and non-diseased subjects to estimate sensitivity and specificity within a pre-specified margin of error W."
    },
    "predictive_values": {
        "custom_design": "Diagnostic: Target Sensitivity Precision",
        "badge_title": "Positive & Negative Predictive Values",
        "rationale": "Size screening cohorts accounting for true underlying disease prevalence to guarantee reliable PPV and NPV."
    },
    "likelihood_ratios": {
        "custom_design": "Diagnostic: Target Sensitivity Precision",
        "badge_title": "Diagnostic Likelihood Ratios (LR+ / LR-)",
        "rationale": "Determine sample size needed to evaluate clinically decisive post-test odds shifts."
    },
    "kaplan_meier": {
        "custom_design": "Survival: Cox Log-Rank Test (Hazard Ratio)",
        "badge_title": "Log-Rank Survival Sizing (Events E)",
        "rationale": "Determine total required failure events (E) and patient accrual N under Schoenfeld's (1981) log-rank formulation."
    },
    "cox_regression": {
        "custom_design": "Survival: Cox Log-Rank Test (Hazard Ratio)",
        "badge_title": "Cox Proportional Hazards (Hazard Ratio)",
        "rationale": "Calculate required clinical endpoints to detect a target Hazard Ratio (HR) with multivariable covariate adjustment."
    },
    "risk_stratification": {
        "custom_design": "Binary: Retrospective Case-Control (Odds Ratio)",
        "badge_title": "Risk Stratification Sizing",
        "rationale": "Ensure adequate sample representation across risk tiers with Events Per Variable (EPV >= 10) validation."
    },
    "mantel_haenszel": {
        "custom_design": "Binary: Retrospective Case-Control (Odds Ratio)",
        "badge_title": "Stratified Mantel-Haenszel OR",
        "rationale": "Size stratified contingency tables to maintain statistical power after controlling for confounders."
    },
    "outbreak_investigation": {
        "custom_design": "Binary: Retrospective Case-Control (Odds Ratio)",
        "badge_title": "Outbreak Investigation Odds Ratio",
        "rationale": "Determine sample size required in acute outbreak case-control investigations (1:1 to 1:4 control matching)."
    },
    "meta_analysis": {
        "custom_design": "Continuous: Two Independent Means (t-test)",
        "badge_title": "Meta-Analysis Statistical Power",
        "rationale": "Evaluate aggregate statistical power across pooled trials under DerSimonian-Laird random effects."
    },

    # ----------------- Pillar 3: Machine Learning Models -----------------
    "logistic_regression": {
        "custom_design": "Binary: Retrospective Case-Control (Odds Ratio)",
        "badge_title": "Binary Classification (EPV >= 10)",
        "rationale": "Ensure minimum 10 to 20 positive outcome events per candidate feature to prevent logistic overfitting."
    },
    "elastic_net_regression": {
        "custom_design": "Continuous: Two Independent Means (t-test)",
        "badge_title": "Regularized Regression Sizing",
        "rationale": "Size continuous outcome prediction models with sufficient observations to stabilize L1/L2 penalty tuning."
    },
    "cost_sensitive_rf": {
        "custom_design": "Binary: Retrospective Case-Control (Odds Ratio)",
        "badge_title": "Rare Event Classification (< 10%)",
        "rationale": "Size highly imbalanced cohorts to guarantee sufficient minority class instances for PR-AUC and Brier score convergence."
    },
    "fairness_audit": {
        "custom_design": "Binary: Two Independent Proportions",
        "badge_title": "Algorithmic Fairness (Four-Fifths Rule)",
        "rationale": "Size protected demographic sub-groups to detect statistically significant disparate impact differences."
    }
}


def render_sample_size_deep_link_badge(
    method_key: str,
    method_name: str,
    pillar: str = "Classical Statistics",
    origin_page: str = "pages/1_📊_Statistical_Tests.py",
    origin_title: str = "Statistical Tests",
    origin_icon: str = "📊"
) -> None:
    """Renders a unified, styled biostatistical sample size planning badge card

    with a direct 1-click launcher into the matching framework on Page 4.
    """
    is_clustering = (
        method_key in ("kmeans_clustering", "dbscan_clustering")
        or "cluster" in method_key.lower()
        or "unsupervised" in method_key.lower()
    )

    if is_clustering:
        st.markdown(r"""
        <div style="background-color: #fefce8; border: 1px solid #fef08a; border-left: 4px solid #ca8a04; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.65rem; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <span style="font-weight: 700; color: #713f12; font-size: 0.92rem;">
                        📐 Machine Learning Sample Scale & Cluster Stability
                    </span>
                    <span style="background-color: #fef9c3; color: #854d0e; font-size: 0.75rem; font-weight: 700; padding: 0.15rem 0.45rem; border-radius: 4px; border: 1px solid #fde047; margin-left: 0.4rem;">
                        Geometric Heuristic
                    </span>
                    <div style="font-size: 0.8rem; color: #854d0e; margin-top: 0.2rem; line-height: 1.35;">
                        Unsupervised discovery has no prior null hypothesis ($H_0$). Sizing requires $N \ge 20\text{--}50$ samples per expected cluster ($N \ge 20k$) and $N \ge 10 \times p$ observations per feature, evaluated via <strong>Silhouette Score</strong>, <strong>Davies-Bouldin Index</strong>, and <strong>Calinski-Harabasz Index</strong>.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    is_time_series = (
        method_key in ("holt_winters_exponential_smoothing", "sarima_forecasting", "lagged_ensemble_forecasting", "time_series_forecaster")
        or "forecasting" in method_key.lower()
        or "time_series" in method_key.lower()
        or "exponential_smoothing" in method_key.lower()
    )

    if is_time_series:
        st.markdown(r"""
        <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; border-left: 4px solid #2563eb; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.65rem; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <span style="font-weight: 700; color: #1e40af; font-size: 0.92rem;">
                        📈 Time Series Sequence Sizing & Temporal Validation
                    </span>
                    <span style="background-color: #dbeafe; color: #1d4ed8; font-size: 0.75rem; font-weight: 700; padding: 0.15rem 0.45rem; border-radius: 4px; border: 1px solid #93c5fd; margin-left: 0.4rem;">
                        Sequential Heuristic
                    </span>
                    <div style="font-size: 0.8rem; color: #1e3a8a; margin-top: 0.2rem; line-height: 1.35;">
                        Classical two-sample hypothesis testing ($H_0$) does not apply to sequential chronological series. Sizing requires a historical sequence of at least <strong>4 to 8 full seasonal cycles</strong> ($N \ge 50\text{--}100+$ observations) to stabilize level, trend, and seasonal index estimates, validated using <strong>Rolling-Origin Cross-Validation</strong> over forward forecast horizons.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    framework_info = SAMPLE_SIZE_FRAMEWORK_MAP.get(method_key)
    if not framework_info:
        # Generic fallback
        framework_info = {
            "custom_design": "Continuous: Two Independent Means (t-test)",
            "badge_title": "Biostatistical Power Sizing",
            "rationale": "Calculate evaluable sample size ($N$), target statistical power ($1-\\beta \\ge 80\\%$), and protocol attrition cushions."
        }

    custom_design = framework_info["custom_design"]
    badge_title = framework_info["badge_title"]
    rationale = framework_info["rationale"]

    badge_html = f"""
    <div style="background: linear-gradient(90deg, #fff7ed 0%, #ffedd5 100%); border: 1px solid #fed7aa; border-left: 4px solid #ea580c; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.65rem; margin-bottom: 0.75rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <span style="font-size: 1rem; margin-right: 0.3rem;">📐</span>
                <span style="font-weight: 700; color: #9a3412; font-size: 0.92rem;">
                    Matching Sample Size & Power Framework:
                </span>
                <span style="background-color: #ffedd5; color: #c2410c; font-size: 0.75rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 4px; border: 1px solid #fdba74; margin-left: 0.35rem;">
                    {badge_title}
                </span>
                <div style="font-size: 0.8rem; color: #7c2d12; margin-top: 0.25rem; line-height: 1.4;">
                    {rationale}
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)

    col_btn, _ = st.columns([1.5, 2.5])
    with col_btn:
        btn_key = f"btn_ss_badge_{method_key}_{origin_page.replace('/', '_').replace('.', '_')}"
        if st.button(
            f"📐 Calculate Sample Size for {method_name} 👉",
            key=btn_key,
            use_container_width=True,
            help=f"Launch Page 4 with the matching '{custom_design}' framework pre-selected."
        ):
            st.session_state["sample_size_scenario"] = "custom"
            st.session_state["sample_size_custom_design"] = custom_design
            st.session_state["selector_consultation"] = {
                "active": True,
                "problem_name": method_name,
                "method_name": f"Sample Size: {badge_title}",
                "method_key": method_key,
                "pillar_label": f"📐 Sample Size & Power (from {origin_title})",
                "domain_name": pillar,
                "source_title": origin_title,
                "source_icon": origin_icon,
                "return_page": origin_page,
                "return_label": f"← Return to {origin_title}",
                "sub_text": f"Pre-configured power sizing framework for {method_name}",
                "target_type": "sample_size"
            }
            st.switch_page("pages/4_📐_Sample_Size_and_Power.py")

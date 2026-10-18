"""Universal Study Design & Method Selector Recommendation Engine.

Evaluates user-specified context, analytical objectives, endpoint structures, and study design
constraints to route researchers to the optimal method, alternative fallback, sample size framework,
and deep-link payloads across Classical Statistics, Epidemiology, and Machine Learning.
"""

from typing import Dict, Any, List, Optional
from src.selector.study_design_schema import (
    DOMAIN_PROFILES,
    PROBLEM_DEFINITIONS,
    PROBLEM_BLUEPRINTS
)


def get_problem_blueprint(blueprint_key: str) -> Dict[str, Any]:
    """Retrieves a pre-configured problem blueprint by key, with default fallback."""
    bp = dict(PROBLEM_BLUEPRINTS.get(blueprint_key, PROBLEM_BLUEPRINTS["bp_cardiovascular_trial"]))
    if "domain_name" not in bp:
        domain_meta = DOMAIN_PROFILES.get(bp.get("domain", ""), {})
        bp["domain_name"] = domain_meta.get("name", bp.get("domain", ""))
    return bp


def route_universal_method_recommendation(
    domain: str = "healthcare_clinical",
    analytical_goal: Optional[str] = None,
    endpoint_type: str = "continuous",
    study_design: str = "parallel_rct",
    sample_scale: str = "medium",
    governance_priority: str = "trial_sizing",
    problem_key: Optional[str] = None
) -> Dict[str, Any]:
    """Routes the user's research problem to the optimal analytical framework across all pillars,
    dynamically adapting to study design architecture and sample magnitude constraints.
    """
    # 1. Base Problem Identification
    inferred_key = None

    if analytical_goal is None and problem_key is not None and problem_key in PROBLEM_DEFINITIONS:
        inferred_key = problem_key
    elif analytical_goal == "compare_groups" or (analytical_goal is None and (problem_key is None or problem_key not in PROBLEM_DEFINITIONS)):
        if study_design == "paired_pre_post":
            inferred_key = "paired_pre_post"
        elif study_design == "multi_arm":
            inferred_key = "multi_group_comparison"
        else:
            inferred_key = "two_group_comparison"

    elif analytical_goal == "paired_change":
        inferred_key = "paired_pre_post"

    elif analytical_goal == "contingency_rates":
        if study_design == "case_control":
            inferred_key = "retrospective_case_control"
        elif study_design in ("cohort", "observational_cohort", "prospective", "parallel_rct"):
            inferred_key = "prospective_cohort_study"
        else:
            inferred_key = "categorical_contingency"

    elif analytical_goal == "cohort_prospective":
        if study_design == "case_control":
            inferred_key = "retrospective_case_control"
        else:
            inferred_key = "prospective_cohort_study"

    elif analytical_goal == "case_control_retrospective":
        inferred_key = "retrospective_case_control"

    elif analytical_goal == "correlation_association":
        inferred_key = "linear_correlation"

    elif analytical_goal == "survival_time_to_event":
        inferred_key = "survival_time_to_event"

    elif analytical_goal == "diagnostic_accuracy":
        inferred_key = "diagnostic_screening_evaluation"

    elif analytical_goal == "demand_forecasting":
        inferred_key = "time_series_forecasting"

    elif analytical_goal == "predict_rare_event":
        inferred_key = "rare_event_classification"

    elif analytical_goal == "equity_fairness":
        inferred_key = "algorithmic_fairness_audit"

    elif analytical_goal == "clustering_phenotyping":
        inferred_key = "unsupervised_phenotyping"

    elif analytical_goal == "dimension_reduction":
        inferred_key = "dimension_reduction_pca"

    else:
        inferred_key = problem_key if (problem_key and problem_key in PROBLEM_DEFINITIONS) else "two_group_comparison"

    p_key = inferred_key
    selected_problem = dict(PROBLEM_DEFINITIONS.get(p_key, PROBLEM_DEFINITIONS["two_group_comparison"]))

    # 2. Extract domain metadata
    domain_meta = dict(DOMAIN_PROFILES.get(domain, DOMAIN_PROFILES["healthcare_clinical"]))

    # 3. Dynamic Specialization Based on Sample Scale & Study Design
    deep_link_session = dict(selected_problem["deep_link_session"])
    prim_name = selected_problem["primary_method_name"]
    prim_key = selected_problem["primary_method_key"]
    alt_name = selected_problem["alternative_method_name"]
    alt_key = selected_problem["alternative_method_key"]
    alt_cond = selected_problem["alternative_condition"]
    param_type = selected_problem["parametric_type"]
    rationale = selected_problem["rationale"]

    # Specialization when Sample Scale is SMALL (N < 30)
    if sample_scale == "small":
        if p_key == "two_group_comparison":
            prim_name = "Mann-Whitney U Test (Wilcoxon Rank-Sum)"
            prim_key = "mann_whitney_u"
            param_type = "Non-Parametric (Small Sample N < 30)"
            alt_name = "Welch's Independent Two-Sample t-test"
            alt_key = "welch_t_test"
            alt_cond = "Use only if normality and homoscedasticity can be defended despite small N."
            rationale = "Small Sample Notice (N < 30): The Central Limit Theorem cannot be relied upon to protect the Type I error rate against non-normality or outliers. The non-parametric Mann-Whitney U test is the mathematically robust, recommended primary default."
            deep_link_session["direct_test_key"] = "mann_whitney_u"

        elif p_key == "paired_pre_post":
            prim_name = "Wilcoxon Signed-Rank Test (Paired)"
            prim_key = "wilcoxon_paired"
            param_type = "Non-Parametric (Small Sample N < 30)"
            alt_name = "Paired Samples t-test"
            alt_key = "paired_t_test"
            alt_cond = "Use only if within-subject paired differences are proven to follow a strict Gaussian distribution."
            rationale = "Small Sample Notice (N < 30): Within-subject paired differences cannot be assumed to follow a normal distribution with small samples. The non-parametric Wilcoxon Signed-Rank test provides optimal statistical power and exact rank inference."
            deep_link_session["direct_test_key"] = "wilcoxon_paired"

        elif p_key == "multi_group_comparison":
            prim_name = "Kruskal-Wallis H Test"
            prim_key = "kruskal_wallis"
            param_type = "Non-Parametric (Small Sample N < 30)"
            alt_name = "One-Way Analysis of Variance (ANOVA)"
            alt_key = "one_way_anova"
            alt_cond = "Use only if homoscedasticity and residual normality hold across all arms."
            rationale = "Small Sample Notice (N < 30): Comparing 3+ small groups with ANOVA is highly vulnerable to variance heterogeneity and non-normality. The Kruskal-Wallis test assesses median differences without parametric assumptions."
            deep_link_session["direct_test_key"] = "kruskal_wallis"

        elif p_key == "categorical_contingency":
            prim_name = "Fisher's Exact Test"
            prim_key = "fisher_exact"
            param_type = "Exact Hypergeometric"
            alt_name = "Pearson Chi-Square Test of Independence"
            alt_key = "chi_square_independence"
            alt_cond = "Use only if all expected cell counts in the 2x2 table exceed 5 (E_ij >= 5)."
            rationale = "Small Sample Notice (N < 30): In small cohorts, expected contingency table cell counts frequently fall below 5 (E_ij < 5), invalidating the asymptotic Chi-Square distribution. Fisher's Exact Test calculates exact hypergeometric probabilities."
            deep_link_session["direct_test_key"] = "fisher_exact"

        elif p_key == "linear_correlation":
            prim_name = "Spearman Rank Correlation (rho)"
            prim_key = "spearman_correlation"
            param_type = "Non-Parametric Rank Association"
            alt_name = "Pearson Correlation Coefficient (r)"
            alt_key = "pearson_correlation"
            alt_cond = "Use only if bivariate normality is verified without leverage outliers."
            rationale = "Small Sample Notice (N < 30): Pearson correlation is acutely sensitive to bivariate outliers when N is small. Spearman's monotonic rank correlation provides a robust estimate of association."
            deep_link_session["direct_test_key"] = "spearman_correlation"

        elif p_key == "rare_event_classification":
            prim_name = "ElasticNet Penalized Logistic Regression"
            prim_key = "penalized_logistic_regression"
            param_type = "Regularized Linear (Low-Variance)"
            alt_name = "Random Forest Classifier (Balanced Sub-Sampling)"
            alt_key = "random_forest_classifier"
            alt_cond = "Use when non-linear feature interactions are hypothesized and cross-validation variance can be tolerated."
            rationale = "Small Sample Notice (N < 30): Training complex tree ensembles on small samples with rare events leads to severe overfitting and variance instability. Regularized Penalized Logistic Regression (ElasticNet) controls parameter variance and produces generalizable clinical Odds Ratios."
            deep_link_session["ml_direct_model_key"] = "penalized_logistic_regression"

    # Specialization when Sample Scale is LARGE (N >= 500)
    elif sample_scale == "large":
        if p_key == "two_group_comparison":
            rationale = "Large Sample Notice (N >= 500): With ample sample size, the Central Limit Theorem guarantees asymptotic normality of sample means. However, p-values become hyper-sensitive to trivial differences; prioritize Effect Size (Cohen's d) and 95% Confidence Intervals over raw significance."

        elif p_key == "rare_event_classification":
            prim_name = "Gradient Boosted Decision Trees (HistGradientBoosting / LightGBM)"
            prim_key = "gradient_boosting_classifier"
            deep_link_session["ml_direct_model_key"] = "gradient_boosting_classifier"
            rationale = "Large Sample Notice (N >= 500): With a large registry or streaming telemetry, non-linear gradient-boosted ensembles can learn rich feature interactions without overfitting. Evaluated strictly via PR-AUC and Brier calibration score."

        elif p_key == "categorical_contingency":
            rationale = "Large Sample Notice (N >= 500): Asymptotic Chi-Square approximation is highly accurate with large cell counts. Examine Cramer's V to quantify association magnitude beyond statistical significance."

    # Specific Study Design Causal Rationale Checks
    if study_design == "case_control" and p_key in ("retrospective_case_control", "prospective_cohort_study", "categorical_contingency"):
        rationale += " Study Design Alignment: In retrospective case-control sampling, subjects are recruited based on disease outcome. Because total population headcount denominators are unobservable, Relative Risk cannot be computed; the Odds Ratio is the mathematically valid causal estimator."

    elif study_design == "observational_cohort" and p_key == "prospective_cohort_study":
        rationale += " Study Design Alignment: Longitudinal forward tracking of exposed vs. unexposed cohorts enables direct calculation of cumulative incidence risk ratios (RR)."

    elif study_design == "paired_pre_post" and p_key == "paired_pre_post":
        rationale += " Study Design Alignment: Repeated measurements on the same observational units induce positive within-subject covariance; paired testing cancels inter-individual baseline variation."

    elif study_design == "multi_arm" and p_key == "multi_group_comparison":
        rationale += " Study Design Alignment: Multi-arm parallel trials evaluate 3+ groups simultaneously; omnibus testing controls the Family-Wise Error Rate (FWER) against alpha inflation."

    elif study_design == "cross_sectional" and p_key == "categorical_contingency":
        rationale += " Study Design Alignment: Cross-sectional surveys collect exposure and outcome concurrently; temporal precedence cannot be established, limiting causal inference to associations."

    # 4. Construct deep link payloads
    pillar = selected_problem["primary_pillar"]
    deep_link_info = {
        "page_route": selected_problem["page_route"],
        "session_state_updates": deep_link_session
    }

    # Matching Sample Size deep link payload
    if selected_problem.get("sample_size_framework") == "unsupervised_clustering":
        sample_size_deep_link = {
            "page_route": "pages/3_🤖_Machine_Learning.py",
            "session_state_updates": {
                "ml_nav_mode": "direct",
                "ml_direct_model_key": "kmeans_clustering",
                "clustering_inspect_metrics": True
            }
        }
    elif selected_problem.get("sample_size_framework") == "time_series_capacity":
        sample_size_deep_link = {
            "page_route": "pages/3_🤖_Machine_Learning.py",
            "session_state_updates": {
                "ml_nav_mode": "direct",
                "ml_direct_model_key": "holt_winters_exponential_smoothing",
                "time_series_inspect_metrics": True
            }
        }
    else:
        sample_size_deep_link = {
            "page_route": "pages/4_📐_Sample_Size_and_Power.py",
            "session_state_updates": {
                "sample_size_scenario": "custom",
                "sample_size_custom_design": selected_problem["sample_size_label"]
            }
        }

    # 5. Synthesize Methodological Checklist
    checklist = generate_study_design_checklist(
        selected_problem,
        sample_scale=sample_scale,
        governance_priority=governance_priority,
        study_design=study_design
    )

    # 6. Synthesize Pedagogical Pillar Comparison Matrix
    comparison = generate_pillar_comparison_matrix(selected_problem)

    return {
        "problem_key": p_key,
        "problem_name": selected_problem["name"],
        "summary": selected_problem["summary"],
        "domain": domain_meta,
        "primary_pillar": pillar,
        "primary_pillar_label": {
            "statistics": "📊 Classical Statistical Tests (Pillar 1)",
            "epidemiology": "🏥 Epidemiological & Public Health Methods (Pillar 2)",
            "machine_learning": "🤖 Machine Learning & Operational Analytics (Pillar 3)"
        }[pillar],
        "primary_method_name": prim_name,
        "primary_method_key": prim_key,
        "parametric_type": param_type,
        "rationale": rationale,
        "alternative_method_name": alt_name,
        "alternative_method_key": alt_key,
        "alternative_condition": alt_cond,
        "sample_size_framework": selected_problem["sample_size_framework"],
        "sample_size_label": selected_problem["sample_size_label"],
        "sample_size_params": selected_problem["sample_size_params"],
        "deep_link": deep_link_info,
        "sample_size_deep_link": sample_size_deep_link,
        "checklist": checklist,
        "comparison": comparison
    }


def generate_study_design_checklist(
    problem: Dict[str, Any],
    sample_scale: str = "medium",
    governance_priority: str = "trial_sizing",
    study_design: str = "parallel_rct"
) -> List[Dict[str, str]]:
    """Builds a dynamic 5-point methodological protocol checklist reflecting sample scale,
    study design architecture, and governance standards.
    """
    checks = []

    # Check 1: Sample Size & Sizing Adequacy
    if problem.get("sample_size_framework") == "unsupervised_clustering":
        if sample_scale == "small":
            checks.append({
                "badge": "1. Sizing: Small Cohort (N < 30)",
                "title": "High Cluster Instability & Idiosyncratic Partitions",
                "requirement": "Target at least 20–50 samples per expected cluster (N >= 20k) and N >= 10p features.",
                "guidance": "Unsupervised clustering does not evaluate a null hypothesis. With N < 30, k-means is prone to discovering spurious patient clusters. Validate stability via Silhouette Score (s > 0.50), Davies-Bouldin Index (DBI < 1.0), and Calinski-Harabasz Index."
            })
        elif sample_scale == "large":
            checks.append({
                "badge": "1. Sizing: Large Cohort (N >= 500)",
                "title": "Robust Cluster Convergence & Phenotypic Stability",
                "requirement": "Validate cluster reproducibility across bootstrap resamples.",
                "guidance": "Large sample scale provides high stability for centroid estimation. Use Elbow WCSS, Silhouette Score, and Calinski-Harabasz curves to determine optimal k, and audit cluster separation via Davies-Bouldin Index."
            })
        else:
            checks.append({
                "badge": "1. Sizing: Moderate Sample (30 <= N < 500)",
                "title": "Cluster Stability & Feature Dimensionality Ratio",
                "requirement": f"Heuristic sizing for {problem['sample_size_label']}.",
                "guidance": "Ensure sample scale satisfies N >= 20–50 per expected cluster and N >= 10 * features. Evaluate partition quality using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index rather than hypothesis testing power formulas."
            })
    elif problem.get("sample_size_framework") == "time_series_capacity" or problem.get("id") == "time_series_forecasting":
        if sample_scale == "small":
            checks.append({
                "badge": "1. Sizing: Short Series (N < 30 time points)",
                "title": "Severe State Space & Seasonal Parameter Instability",
                "requirement": "Target at least 4 to 8 full seasonal cycles (e.g. N >= 28–56 observations for weekly periodicity).",
                "guidance": "Time series models do not evaluate independent sample power. With fewer than 30 observations, triple exponential smoothing cannot reliably estimate initial seasonal indices and trend slope. Supplement with simple single exponential smoothing or aggregate temporal bins."
            })
        elif sample_scale == "large":
            checks.append({
                "badge": "1. Sizing: Long Historical Series (N >= 500 time points)",
                "title": "High Parameter Stability & Structural Break Auditing",
                "requirement": "Audit structural breaks, changepoints, and evolving variance across long histories.",
                "guidance": "Long series provide robust estimates for seasonal smoothing coefficients (alpha, beta, gamma). Ensure validation uses Rolling-Origin Cross-Validation (expanding window TimeSeriesSplit) and verify whether regime shifts require damped trend components."
            })
        else:
            checks.append({
                "badge": "1. Sizing: Moderate Historical Series (30 <= N < 500 time points)",
                "title": "Seasonal Cycle Adequacy & Rolling-Origin Validation",
                "requirement": f"Chronological sizing for {problem['sample_size_label']}.",
                "guidance": "Ensure historical sequence length covers at least 4 to 8 full seasonal cycles (N >= 50–100+ points). Evaluate forward forecast horizon accuracy using Rolling-Origin Cross-Validation (MAE, MAPE, RMSE) with shaded 95% prediction corridors."
            })
    elif sample_scale == "small":
        checks.append({
            "badge": "1. Sizing: Small Cohort (N < 30)",
            "title": "Constrained Power & Type II Error Risk",
            "requirement": f"Evaluate statistical power carefully for {problem['sample_size_label']}.",
            "guidance": "With N < 30, statistical power to detect moderate effect sizes is strictly limited. Pre-specify the Minimum Detectable Effect Size (MDES) and use exact non-parametric tests."
        })
    elif sample_scale == "large":
        checks.append({
            "badge": "1. Sizing: Large Cohort (N >= 500)",
            "title": "High Power & Clinical Relevance Focus",
            "requirement": f"Adequate statistical power (> 90%) for {problem['sample_size_label']}.",
            "guidance": "With large samples, trivial differences achieve p < 0.001. Pre-specify the Minimal Clinically Important Difference (MCID) and prioritize Effect Size and 95% CIs over raw p-values."
        })
    else:
        checks.append({
            "badge": "1. Sizing: Moderate Sample (30 <= N < 500)",
            "title": "Statistical Power & Protocol Sizing",
            "requirement": f"Formal sample size calculation required for {problem['sample_size_label']}.",
            "guidance": "ICH-E9 and CONSORT 2010 mandate pre-specifying alpha=0.05, power>=80%, MCID, and inflating for anticipated attrition."
        })

    # Check 2: Core Distributional Assumptions & Sample Scale
    if sample_scale == "small":
        checks.append({
            "badge": "2. Distribution: Non-Parametric Required",
            "title": "Normality Untenable (Small Sample)",
            "requirement": "Non-parametric rank test or exact permutation model required.",
            "guidance": f"Central Limit Theorem does not apply when N < 30. Use {problem['primary_method_name']} to protect Type I error against skewness or outliers."
        })
    else:
        checks.append({
            "badge": "2. Distribution: Parametric Assumed",
            "title": "Parametric Validity & Random Sampling",
            "requirement": "Independent observations without unmodeled clustering or autocorrelation.",
            "guidance": f"If distributional assumptions or variance homogeneity fail, seamlessly switch to {problem['alternative_method_name']}."
        })

    # Check 3: Confounding & Study Design Architecture
    if study_design == "case_control":
        checks.append({
            "badge": "3. Design: Retrospective Case-Control",
            "title": "Outcome-Dependent Sampling & Odds Ratios",
            "requirement": "Select controls independently of exposure; compute Odds Ratios.",
            "guidance": "Relative Risk cannot be measured directly because incidence denominators are unknown. Control confounding via stratified Mantel-Haenszel analysis or multivariable logistic regression."
        })
    elif study_design == "observational_cohort":
        checks.append({
            "badge": "3. Design: Prospective Longitudinal Cohort",
            "title": "Exposure Tracking & Confounding Control",
            "requirement": "Enumerate exposed and unexposed cohorts with person-time follow-up.",
            "guidance": "In prospective cohorts, evaluate confounding via the 10% change-in-estimate rule or multivariable Cox/Poisson regression under STROBE guidelines."
        })
    elif study_design == "parallel_rct":
        checks.append({
            "badge": "3. Design: Parallel Randomized Trial",
            "title": "Random Allocation & Intention-to-Treat",
            "requirement": "Randomized allocation to balance baseline covariates.",
            "guidance": "Follow CONSORT 2010 Item 7a; preserve random allocation integrity via Intention-to-Treat (ITT) principles without post-randomization attrition bias."
        })
    elif study_design == "paired_pre_post":
        checks.append({
            "badge": "3. Design: Paired / Repeated Measures",
            "title": "Within-Subject Variance Reduction",
            "requirement": "Analyze intra-individual difference scores (d = x_post - x_pre).",
            "guidance": "Do not use independent two-sample tests; intra-subject pairing cancels inter-individual heterogeneity and substantially improves statistical power."
        })
    elif study_design == "multi_arm":
        checks.append({
            "badge": "3. Design: Multi-Arm Parallel Trial",
            "title": "Family-Wise Error Rate & Multiplicity",
            "requirement": "Omnibus testing followed by post-hoc adjustment.",
            "guidance": "Protects against Type I false discovery inflation that occurs when conducting multiple unadjusted pairwise comparisons across 3+ treatment arms."
        })
    elif study_design == "cross_sectional":
        checks.append({
            "badge": "3. Design: Cross-Sectional Survey",
            "title": "Prevalence & Temporal Ambiguity",
            "requirement": "Simultaneous observation of exposure and outcome.",
            "guidance": "Cross-sectional data cannot determine temporal precedence or causal incidence rates. Report Prevalence Odds Ratios and treat findings as associative."
        })
    elif study_design in ("sensor_telemetry", "web_telemetry"):
        checks.append({
            "badge": "3. Design: Streaming Telemetry",
            "title": "High-Frequency Telemetry & Drift",
            "requirement": "Calibrate sensor baselines and normalize feature scales.",
            "guidance": "Inspect sensor degradation and stationarity before training machine learning models to avoid distribution shift."
        })
    else:
        checks.append({
            "badge": "3. Design: Chronological Sequence",
            "title": "Longitudinal Time Series Architecture",
            "requirement": "Strict chronological ordering without missing intervals.",
            "guidance": "Decompose trend, seasonality, and cyclicity. Enforce rolling-origin time series evaluation."
        })

    # Check 4: Temporal Integrity (Zero Leakage)
    if study_design == "longitudinal_series" or problem.get("id") == "time_series_forecasting":
        checks.append({
            "badge": "4. Temporal Integrity: Rolling Origin CV",
            "title": "Future-to-Past Leakage Prevention",
            "requirement": "Strict longitudinal chronological ordering in time-to-event or forecasting.",
            "guidance": "Never shuffle time series across cross-validation folds; enforce rolling-origin expanding window evaluation."
        })
    else:
        checks.append({
            "badge": "4. Unit of Analysis: Cluster Audit",
            "title": "Independence & Cluster Effect (ICC)",
            "requirement": "Independent observations without unmodeled cluster correlation.",
            "guidance": "If participants are clustered within clinics, hospital wards, or schools, inflate sample size by the Variance Inflation Factor (VIF)."
        })

    # Check 5: Governance & Ethical AI Equity
    if problem.get("id") in ("algorithmic_fairness_audit", "rare_event_classification"):
        checks.append({
            "badge": "5. Algorithmic Equity: Four-Fifths Parity",
            "title": "Protected Characteristics & Bias Audit",
            "requirement": "Auditing disparate impact across socio-economic deprivation or demographic tiers.",
            "guidance": "Ensure operational algorithms do not disproportionately penalize vulnerable cohorts (DIR >= 0.80) under NHS AI Lab ethics standards."
        })
    else:
        checks.append({
            "badge": "5. Governance: Protocol Registration",
            "title": "REC / IRB & Reporting Compliance",
            "requirement": "Pre-specify primary endpoints and analysis plan in a formal study protocol.",
            "guidance": "Follow international reporting guidelines (CONSORT 2010 for trials, STROBE for observational studies, TRIPOD for prediction models)."
        })

    return checks


def generate_pillar_comparison_matrix(problem: Dict[str, Any]) -> List[Dict[str, str]]:
    """Builds a pedagogical side-by-side comparison matrix across the 3 analytical pillars."""
    pillar = problem["primary_pillar"]

    return [
        {
            "pillar": "📊 Classical Statistics",
            "approach": "Hypothesis Testing & Unbiased Estimation",
            "optimizes_for": "p-values, 95% Confidence Intervals & Population Inferences",
            "when_to_choose": "Strict regulatory trials, confirmatory hypothesis testing, clean low-dimensional data.",
            "status": "⭐ Recommended Primary Approach" if pillar == "statistics" else "Benchmark Comparator"
        },
        {
            "pillar": "🏥 Epidemiological Methods",
            "approach": "Observational Risk & Causal Attribution",
            "optimizes_for": "Relative Risk, Odds Ratios, Hazard Ratios, Attributable Fractions & DSR",
            "when_to_choose": "Public health surveillance, retrospective disease etiology, screening diagnostic evaluation, survival.",
            "status": "⭐ Recommended Primary Approach" if pillar == "epidemiology" else "Alternative Causal Framework"
        },
        {
            "pillar": "🤖 Machine Learning & AI",
            "approach": "Out-of-Sample Predictive Generalization",
            "optimizes_for": "PR-AUC, ROC-AUC, Brier Calibration Score, Forecast MAE & Fairness Parity",
            "when_to_choose": "Complex non-linear interactions, rare event prediction (readmissions/failures), operational time series, high-dimensional features.",
            "status": "⭐ Recommended Primary Approach" if pillar == "machine_learning" else "Complementary Predictive Modeling"
        }
    ]


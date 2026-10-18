"""Unified Study Design & Method Selector Taxonomy & Schema.

Defines the decision graph, domain profiles, problem classifications, and cross-pillar
mappings connecting Classical Statistics, Epidemiology Methods, Machine Learning,
and Sample Size Estimation.
"""

from typing import Dict, Any, List

# =====================================================================
# 1. Real-World Domain Profiles
# =====================================================================

DOMAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "healthcare_clinical": {
        "id": "healthcare_clinical",
        "name": "🏥 Healthcare & Clinical Medicine",
        "organisation": "All Meadows Cardiovascular Institute / Inner City Teaching Hospital",
        "description": "Secondary care, randomized clinical trials, clinical prediction models, and inpatient hospital operations.",
        "typical_endpoints": ["Biomarker changes", "30-day mortality", "Disease remission", "Length of stay"]
    },
    "healthcare_public_health": {
        "id": "healthcare_public_health",
        "name": "🏛️ Public Health & Epidemiology",
        "organisation": "Riverside Community Health Trust / NHS Integrated Care Board",
        "description": "Population health surveillance, prospective cohorts, outbreak investigations, rate standardization, and health equity.",
        "typical_endpoints": ["Incidence rates", "Case-control odds", "Direct standardized rates (DSR)", "Did Not Attend (DNA)"]
    },
    "manufacturing": {
        "id": "manufacturing",
        "name": "🏭 Industry 4.0 & Advanced Manufacturing",
        "organisation": "Apex Precision Robotics & Advanced Manufacturing",
        "description": "High-frequency IoT sensor telemetry, condition monitoring, CNC spindle degradation, and predictive maintenance.",
        "typical_endpoints": ["Tool wear hours", "Bearing vibration RMS", "Equipment failure", "Assembly downtime"]
    },
    "marketing": {
        "id": "marketing",
        "name": "📢 Omnichannel Marketing & Retail",
        "organisation": "OmniChannel Direct Marketing & Retail Group",
        "description": "Customer purchase propensity, campaign attribution, dynamic discount allocation, and algorithmic redlining audits.",
        "typical_endpoints": ["Purchase conversion", "Cart abandonment", "Session dwell time", "Selection rate parity"]
    },
    "subscriptions": {
        "id": "subscriptions",
        "name": "📱 Digital SaaS & Subscription Platforms",
        "organisation": "PulseStream Digital Subscriptions & SaaS Platform",
        "description": "Subscriber lifecycle telemetry, monthly churn prevention, playback buffering diagnostics, and customer lifetime value.",
        "typical_endpoints": ["Monthly subscriber churn", "Streaming hours", "Support tickets", "Renewal payment failure"]
    },
    "supply_chain": {
        "id": "supply_chain",
        "name": "🚛 Supply Chain, Freight & Logistics",
        "organisation": "MetroLogix Distribution & Global Freight",
        "description": "Regional warehouse dispatch forecasting, freight truck fleet scheduling, holiday surge buffers, and inventory logistics.",
        "typical_endpoints": ["Daily pallets dispatched", "Freight truck trips", "Warehouse capacity utilization"]
    },
    "general_research": {
        "id": "general_research",
        "name": "🔬 Academic & General Quantitative Research",
        "organisation": "Academic Research & Biostatistics Collaborative",
        "description": "Hypothesis testing, observational surveys, experimental factorials, and general statistical inference.",
        "typical_endpoints": ["Survey scales", "Pre-post test scores", "Response rates", "Continuous measurements"]
    }
}


# =====================================================================
# 2. Universal Problem Taxonomy & Cross-Pillar Mapping
# =====================================================================

PROBLEM_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "two_group_comparison": {
        "id": "two_group_comparison",
        "name": "Comparing 2 Independent Groups (Treatment vs. Control)",
        "summary": "Assess whether a continuous biomarker or outcome differs significantly between two distinct arms.",
        "primary_pillar": "statistics",
        "primary_method_key": "welch_t_test",
        "primary_method_name": "Welch's Independent Two-Sample t-test",
        "parametric_type": "Parametric (Robust to Heteroscedasticity)",
        "alternative_method_key": "mann_whitney_u",
        "alternative_method_name": "Mann-Whitney U Test (Wilcoxon Rank-Sum)",
        "alternative_condition": "Use when sample size is small (N < 30) and data exhibit severe skewness or extreme outliers.",
        "sample_size_framework": "two_means",
        "sample_size_label": "Continuous: Two Independent Means (t-test)",
        "sample_size_params": {"mean1": 8.20, "mean2": 7.75, "sd": 1.10, "alpha": 0.05, "power": 0.85},
        "page_route": "pages/1_📊_Statistical_Tests.py",
        "deep_link_session": {"nav_mode": "direct", "direct_test_key": "welch_t_test"},
        "rationale": "Welch's t-test provides optimal Type I error control without assuming equal population variances, making it the universally recommended default for parallel two-arm trials."
    },

    "paired_pre_post": {
        "id": "paired_pre_post",
        "name": "Comparing Paired / Repeated Measures (Pre- vs. Post-Intervention)",
        "summary": "Measure intra-subject change over time or across matched pairs.",
        "primary_pillar": "statistics",
        "primary_method_key": "paired_t_test",
        "primary_method_name": "Paired Samples t-test",
        "parametric_type": "Parametric",
        "alternative_method_key": "wilcoxon_paired",
        "alternative_method_name": "Wilcoxon Signed-Rank Test (Paired)",
        "alternative_condition": "Use when paired differences (d = x_post - x_pre) violate normality and N < 30.",
        "sample_size_framework": "paired_means",
        "sample_size_label": "Continuous: Paired / Pre-Post Means",
        "sample_size_params": {"mean_diff": 2.0, "sd_diff": 4.0, "alpha": 0.05, "power": 0.85},
        "page_route": "pages/1_📊_Statistical_Tests.py",
        "deep_link_session": {"nav_mode": "direct", "direct_test_key": "paired_t_test"},
        "rationale": "Controls for inter-individual baseline variation by analyzing within-subject differences, significantly increasing statistical efficiency."
    },

    "multi_group_comparison": {
        "id": "multi_group_comparison",
        "name": "Comparing 3+ Independent Groups (Multi-Arm Trial or Cohorts)",
        "summary": "Test whether means differ across three or more treatment doses or demographic groups.",
        "primary_pillar": "statistics",
        "primary_method_key": "one_way_anova",
        "primary_method_name": "One-Way Analysis of Variance (ANOVA)",
        "parametric_type": "Parametric",
        "alternative_method_key": "kruskal_wallis",
        "alternative_method_name": "Kruskal-Wallis H Test",
        "alternative_condition": "Use when residuals are non-normal, variances are severely unequal, or data are ordinal.",
        "sample_size_framework": "anova",
        "sample_size_label": "Continuous: One-Way ANOVA (Multiple Groups)",
        "sample_size_params": {"num_groups": 3, "cohen_f": 0.25, "alpha": 0.05, "power": 0.85},
        "page_route": "pages/1_📊_Statistical_Tests.py",
        "deep_link_session": {"nav_mode": "direct", "direct_test_key": "one_way_anova"},
        "rationale": "Protects against Type I error inflation that would occur if multiple pairwise t-tests were conducted without family-wise alpha adjustment."
    },

    "linear_correlation": {
        "id": "linear_correlation",
        "name": "Evaluating Linear or Monotonic Correlation Between 2 Measures",
        "summary": "Quantify strength and direction of association between two continuous variables.",
        "primary_pillar": "statistics",
        "primary_method_key": "pearson_correlation",
        "primary_method_name": "Pearson Correlation Coefficient (r)",
        "parametric_type": "Parametric",
        "alternative_method_key": "spearman_correlation",
        "alternative_method_name": "Spearman Rank Correlation (rho)",
        "alternative_condition": "Use when the relationship is monotonic rather than strictly linear, or with skewed distributions.",
        "sample_size_framework": "correlation",
        "sample_size_label": "Continuous: Pearson Correlation (r)",
        "sample_size_params": {"r": 0.30, "alpha": 0.05, "power": 0.85},
        "page_route": "pages/1_📊_Statistical_Tests.py",
        "deep_link_session": {"nav_mode": "direct", "direct_test_key": "pearson_correlation"},
        "rationale": "Measures linear co-variation with exact Fisher z-transformation confidence intervals and p-values."
    },

    "categorical_contingency": {
        "id": "categorical_contingency",
        "name": "Comparing Event Frequencies Across Categories (2x2 Table)",
        "summary": "Test independence between categorical exposure and binary outcome.",
        "primary_pillar": "statistics",
        "primary_method_key": "chi_square_independence",
        "primary_method_name": "Pearson Chi-Square Test of Independence",
        "parametric_type": "Asymptotic",
        "alternative_method_key": "fisher_exact",
        "alternative_method_name": "Fisher's Exact Test",
        "alternative_condition": "Mandatory when any expected contingency table cell count is below 5 (E_ij < 5).",
        "sample_size_framework": "two_proportions",
        "sample_size_label": "Binary: Two Independent Proportions",
        "sample_size_params": {"p1": 0.12, "p2": 0.24, "alpha": 0.05, "power": 0.80},
        "page_route": "pages/1_📊_Statistical_Tests.py",
        "deep_link_session": {"nav_mode": "direct", "direct_test_key": "chi_square_independence"},
        "rationale": "Compares observed frequencies against expected frequencies under the null hypothesis of independence."
    },

    "prospective_cohort_study": {
        "id": "prospective_cohort_study",
        "name": "Prospective Cohort Study (Exposure to Longitudinal Disease Risk)",
        "summary": "Follow exposed and unexposed populations forward in time to compute cumulative incidence risk ratios.",
        "primary_pillar": "epidemiology",
        "primary_method_key": "relative_risk",
        "primary_method_name": "Relative Risk / Risk Ratio (RR)",
        "parametric_type": "Epidemiological Risk Ratio",
        "alternative_method_key": "incidence_rate_ratio",
        "alternative_method_name": "Incidence Rate Ratio (IRR with Person-Time)",
        "alternative_condition": "Use when cohort follow-up times vary widely across participants (dynamic cohort with person-years).",
        "sample_size_framework": "cohort_rr",
        "sample_size_label": "Binary: Prospective Cohort (Relative Risk)",
        "sample_size_params": {"p0": 0.10, "rr": 1.75, "alpha": 0.05, "power": 0.80},
        "page_route": "pages/2_🏥_Epidemiology_Methods.py",
        "deep_link_session": {"epi_nav_mode": "direct", "epi_direct_method_key": "relative_risk"},
        "rationale": "The gold-standard causal effect measure for prospective observational cohorts with known headcount denominators."
    },

    "retrospective_case_control": {
        "id": "retrospective_case_control",
        "name": "Retrospective Case-Control Study (Rare Disease Etiology)",
        "summary": "Sample diseased cases and disease-free controls to evaluate past exposure odds.",
        "primary_pillar": "epidemiology",
        "primary_method_key": "odds_ratio",
        "primary_method_name": "Odds Ratio (OR)",
        "parametric_type": "Epidemiological Odds Ratio",
        "alternative_method_key": "mantel_haenszel",
        "alternative_method_name": "Cochran-Mantel-Haenszel (CMH) Stratified OR",
        "alternative_condition": "Use when a potential confounder (e.g. age or sex) must be controlled across stratified 2x2 tables.",
        "sample_size_framework": "case_control_or",
        "sample_size_label": "Binary: Retrospective Case-Control (Odds Ratio)",
        "sample_size_params": {"p0": 0.20, "or": 2.50, "k": 2.0, "alpha": 0.05, "power": 0.80},
        "page_route": "pages/2_🏥_Epidemiology_Methods.py",
        "deep_link_session": {"epi_nav_mode": "direct", "epi_direct_method_key": "odds_ratio"},
        "rationale": "Because total population incidence is unknown in case-control sampling, the Odds Ratio is the mathematically valid estimator of relative risk under the rare disease assumption."
    },

    "diagnostic_screening_evaluation": {
        "id": "diagnostic_screening_evaluation",
        "name": "Diagnostic Device or Screening Accuracy Evaluation",
        "summary": "Evaluate index test sensitivity, specificity, and predictive values against a gold standard.",
        "primary_pillar": "epidemiology",
        "primary_method_key": "diagnostic_accuracy",
        "primary_method_name": "Diagnostic Sensitivity, Specificity & Youden's J",
        "parametric_type": "Diagnostic 2x2 Matrix",
        "alternative_method_key": "likelihood_ratios",
        "alternative_method_name": "Diagnostic Likelihood Ratios (LR+, LR-) & Fagan's Nomogram",
        "alternative_condition": "Use for patient-specific pre-to-post test probability updating across varied care settings.",
        "sample_size_framework": "diagnostic_accuracy",
        "sample_size_label": "Diagnostic: Target Sensitivity Precision",
        "sample_size_params": {"sensitivity": 0.90, "margin": 0.05, "prevalence": 0.15, "alpha": 0.05},
        "page_route": "pages/2_🏥_Epidemiology_Methods.py",
        "deep_link_session": {"epi_nav_mode": "direct", "epi_direct_method_key": "diagnostic_accuracy"},
        "rationale": "STARD-compliant framework evaluating true positive and negative discrimination with Wilson score confidence bounds."
    },

    "survival_time_to_event": {
        "id": "survival_time_to_event",
        "name": "Longitudinal Time-to-Event Survival Analysis",
        "summary": "Follow cohorts over time with right-censoring to compare survival distributions.",
        "primary_pillar": "epidemiology",
        "primary_method_key": "kaplan_meier",
        "primary_method_name": "Kaplan-Meier Survival Curves & Mantel-Cox Log-Rank Test",
        "parametric_type": "Non-Parametric Product-Limit",
        "alternative_method_key": "cox_regression",
        "alternative_method_name": "Cox Proportional Hazards Regression",
        "alternative_condition": "Use when multiple continuous or categorical prognostic covariates must be adjusted simultaneously.",
        "sample_size_framework": "survival",
        "sample_size_label": "Survival: Cox Log-Rank Test (Hazard Ratio)",
        "sample_size_params": {"hr": 0.70, "pevent": 0.55, "alpha": 0.05, "power": 0.85},
        "page_route": "pages/2_🏥_Epidemiology_Methods.py",
        "deep_link_session": {"epi_nav_mode": "direct", "epi_direct_method_key": "kaplan_meier"},
        "rationale": "Handles incomplete observation times via non-informative right-censoring, providing unbiased product-limit survival estimates."
    },

    "rare_event_classification": {
        "id": "rare_event_classification",
        "name": "Supervised Prediction of Rare Clinical or Operational Events",
        "summary": "Predict binary outcomes with severe class imbalance (< 15% event rate).",
        "primary_pillar": "machine_learning",
        "primary_method_key": "random_forest_classifier",
        "primary_method_name": "Random Forest Classifier (Balanced Sub-Sampling)",
        "parametric_type": "Tree Ensemble",
        "alternative_method_key": "penalized_logistic_regression",
        "alternative_method_name": "ElasticNet Penalized Logistic Regression",
        "alternative_condition": "Use when clinician governance mandates transparent linear Odds Ratios rather than tree ensembles.",
        "sample_size_framework": "two_proportions",
        "sample_size_label": "Binary: Two Independent Proportions",
        "sample_size_params": {"p1": 0.08, "p2": 0.16, "alpha": 0.05, "power": 0.80},
        "page_route": "pages/3_🤖_Machine_Learning.py",
        "deep_link_session": {"ml_nav_mode": "direct", "ml_direct_model_key": "random_forest_classifier"},
        "rationale": "Captures non-linear biomarker interactions while balanced sub-sampling overcomes the Accuracy Paradox in rare-event data (optimizing PR-AUC)."
    },

    "time_series_forecasting": {
        "id": "time_series_forecasting",
        "name": "Operational Demand & Capacity Surge Forecasting",
        "summary": "Project sequential daily or weekly demand over a forward 14-day rolling horizon.",
        "primary_pillar": "machine_learning",
        "primary_method_key": "holt_winters_exponential_smoothing",
        "primary_method_name": "Holt-Winters Triple Exponential Smoothing",
        "parametric_type": "State Space Smoothing",
        "alternative_method_key": "sarima_forecasting",
        "alternative_method_name": "Seasonal ARIMA (SARIMA)",
        "alternative_condition": "Use when multi-month autocorrelation and stochastic trends require formal statistical confidence bounds.",
        "sample_size_framework": "time_series_capacity",
        "sample_size_label": "Sequential Time Series Sequence Length (Seasonal Cycles & Rolling Horizon)",
        "sample_size_params": {"min_cycles": 4, "seasonal_period": 7, "min_time_points": 50, "test_horizon": 14},
        "page_route": "pages/3_🤖_Machine_Learning.py",
        "deep_link_session": {"ml_nav_mode": "direct", "ml_direct_model_key": "holt_winters_exponential_smoothing"},
        "rationale": "Decomposes level, trend, and rigid 7-day cyclical seasonality in real time with zero training latency and shaded 95% prediction corridors."
    },

    "algorithmic_fairness_audit": {
        "id": "algorithmic_fairness_audit",
        "name": "Algorithmic Equity & Disparate Impact Audit",
        "summary": "Evaluate operational or clinical risk algorithms across socio-economic or demographic tiers.",
        "primary_pillar": "machine_learning",
        "primary_method_key": "algorithmic_fairness_audit",
        "primary_method_name": "Algorithmic Fairness & Equity Audit (Four-Fifths Rule)",
        "parametric_type": "Fairness Diagnostic Framework",
        "alternative_method_key": "penalized_logistic_regression",
        "alternative_method_name": "Fairness-Constrained Penalized Logistic Regression",
        "alternative_condition": "Use to re-estimate group-specific decision thresholds to guarantee False Positive Rate parity.",
        "sample_size_framework": "two_proportions",
        "sample_size_label": "Binary: Two Independent Proportions",
        "sample_size_params": {"p1": 0.20, "p2": 0.28, "alpha": 0.05, "power": 0.80},
        "page_route": "pages/3_🤖_Machine_Learning.py",
        "deep_link_session": {"ml_nav_mode": "direct", "ml_direct_model_key": "algorithmic_fairness_audit"},
        "rationale": "Audits Disparate Impact Ratio (DIR >= 0.80) and False Positive Parity to ensure automated scheduling or marketing models do not entrench bias."
    },

    "unsupervised_phenotyping": {
        "id": "unsupervised_phenotyping",
        "name": "Discovering Unlabelled Patient Phenotypes or Customer Clusters",
        "summary": "Identify natural geometric clusters in multidimensional feature space without predefined outcome labels.",
        "primary_pillar": "machine_learning",
        "primary_method_key": "kmeans_clustering",
        "primary_method_name": "K-Means Clustering",
        "parametric_type": "Centroid-Based Unsupervised",
        "alternative_method_key": "dbscan_clustering",
        "alternative_method_name": "DBSCAN Density-Based Spatial Clustering",
        "alternative_condition": "Use when clusters have arbitrary non-linear shapes or when isolating anomalous noise points is essential.",
        "sample_size_framework": "unsupervised_clustering",
        "sample_size_label": "Unsupervised Cluster Stability (Silhouette, Davies-Bouldin & Calinski-Harabasz)",
        "sample_size_params": {"n_clusters": 3, "min_cluster_n": 30, "target_silhouette": 0.50},
        "page_route": "pages/3_🤖_Machine_Learning.py",
        "deep_link_session": {"ml_nav_mode": "direct", "ml_direct_model_key": "kmeans_clustering"},
        "rationale": "Iteratively partitions multidimensional observations by minimizing Within-Cluster Sum of Squares (WCSS). Partition quality and optimal cluster count (k) are evaluated via Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index."
    },

    "dimension_reduction_pca": {
        "id": "dimension_reduction_pca",
        "name": "Compressing High-Dimensional Biomarkers or Multicollinear Predictors",
        "summary": "Decompose high-dimensional continuous feature matrices into orthogonal variance components.",
        "primary_pillar": "machine_learning",
        "primary_method_key": "pca_reduction",
        "primary_method_name": "Principal Component Analysis (PCA)",
        "parametric_type": "Linear Eigen-Decomposition",
        "alternative_method_key": "elastic_net_regression",
        "alternative_method_name": "ElasticNet Regression (L1 Sparse Feature Selection)",
        "alternative_condition": "Use when individual original features must be preserved rather than projected into composite latent components.",
        "sample_size_framework": "correlation",
        "sample_size_label": "Continuous: Pearson Correlation (r)",
        "sample_size_params": {"r": 0.35, "alpha": 0.05, "power": 0.85},
        "page_route": "pages/3_🤖_Machine_Learning.py",
        "deep_link_session": {"ml_nav_mode": "direct", "ml_direct_model_key": "pca_reduction"},
        "rationale": "Orthogonalizes correlated predictors, eliminating severe multicollinearity while maximizing preserved patient variance."
    }
}


# =====================================================================
# 3. Pre-Configured End-to-End Problem Blueprints
# =====================================================================

PROBLEM_BLUEPRINTS: Dict[str, Dict[str, Any]] = {
    "bp_cardiovascular_trial": {
        "id": "bp_cardiovascular_trial",
        "title": "Clinical Trial of a Novel Cardiovascular Therapeutic",
        "domain": "healthcare_clinical",
        "domain_name": "🏥 Healthcare & Clinical Medicine",
        "organisation": "All Meadows Cardiovascular Institute",
        "problem_key": "two_group_comparison",
        "description": "Double-blind Phase III randomized controlled trial comparing a novel SGLT2 inhibitor against standard of care for HbA1c reduction in adult heart failure patients.",
        "icon": "🫀",
        "endpoint_type": "continuous",
        "study_design": "parallel_rct",
        "sample_scale": "medium",
        "governance_priority": "trial_sizing",
        "takeaway": "Requires Welch's t-test for the primary endpoint analysis, coupled with a formal sample size calculation with 15% attrition cushion to guarantee 85% statistical power."
    },

    "bp_hospital_outpatient_dna": {
        "id": "bp_hospital_outpatient_dna",
        "title": "Hospital Outpatient Appointment DNA Reduction & Equity Audit",
        "domain": "healthcare_public_health",
        "domain_name": "🏛️ Public Health & Epidemiology",
        "organisation": "Riverside Community Health Trust",
        "problem_key": "algorithmic_fairness_audit",
        "description": "Predicting patient outpatient missed appointments (DNA) to target travel support vouchers, audited across Index of Multiple Deprivation (IMD) deciles to eliminate double-booking penalties.",
        "icon": "⚖️",
        "endpoint_type": "binary",
        "study_design": "observational_cohort",
        "sample_scale": "large",
        "governance_priority": "algorithmic_equity",
        "takeaway": "Deploying predictive algorithms for clinic scheduling requires an Algorithmic Fairness Audit enforcing the Four-Fifths rule (DIR >= 0.80) to protect vulnerable patients from disproportionate false alarms."
    },

    "bp_emergency_bed_surge": {
        "id": "bp_emergency_bed_surge",
        "title": "Emergency Department Winter Demand & Bed Surge Forecasting",
        "domain": "healthcare_clinical",
        "domain_name": "🏥 Healthcare & Clinical Medicine",
        "organisation": "Inner City Teaching Hospital",
        "problem_key": "time_series_forecasting",
        "description": "Projecting daily Emergency Department patient arrivals and inpatient bed occupancy 14 days in advance to dynamically allocate nursing rotas and winter surge capacity.",
        "icon": "⏱️",
        "endpoint_type": "time_series",
        "study_design": "longitudinal_series",
        "sample_scale": "large",
        "governance_priority": "forecasting_accuracy",
        "takeaway": "Triple Exponential Smoothing (Holt-Winters) accurately captures 7-day cyclical rhythms (Monday peaks) and winter trend drift, preventing catastrophic bed management bottlenecks."
    },

    "bp_school_outbreak_investigation": {
        "id": "bp_school_outbreak_investigation",
        "title": "School Norovirus Outbreak & Cluster Trial Investigation",
        "domain": "healthcare_public_health",
        "domain_name": "🏛️ Public Health & Epidemiology",
        "organisation": "Riverside Health Protection Team",
        "problem_key": "retrospective_case_control",
        "description": "Rapid field epidemiology investigation identifying contaminated catering items via food-specific attack rates, accompanied by a cluster-randomized hand hygiene trial.",
        "icon": "🦠",
        "endpoint_type": "binary",
        "study_design": "case_control",
        "sample_scale": "medium",
        "governance_priority": "cluster_trial",
        "takeaway": "Combines retrospective Case-Control Odds Ratios for causal vehicle incrimination with cluster design effect inflation (VIF) for school classroom trial sizing."
    },

    "bp_manufacturing_spindle": {
        "id": "bp_manufacturing_spindle",
        "title": "CNC Spindle Bearing Breakdown & Predictive Maintenance",
        "domain": "manufacturing",
        "domain_name": "🏭 Industry 4.0 & Advanced Manufacturing",
        "organisation": "Apex Precision Robotics & Advanced Manufacturing",
        "problem_key": "rare_event_classification",
        "description": "Automated condition monitoring predicting rare industrial CNC spindle bearing failures (4.2% incidence) from high-frequency multi-axial vibration and thermal telemetry.",
        "icon": "🏭",
        "endpoint_type": "binary",
        "study_design": "sensor_telemetry",
        "sample_scale": "large",
        "governance_priority": "rare_event_auc",
        "takeaway": "Balanced Random Forest captures multi-sensor resonance patterns before catastrophic £45,000/hr assembly stoppage, evaluating model performance strictly via PR-AUC and Recall."
    },

    "bp_marketing_conversion": {
        "id": "bp_marketing_conversion",
        "title": "Customer Purchase Propensity & Promotional Redlining Audit",
        "domain": "marketing",
        "domain_name": "📢 Omnichannel Marketing & Retail",
        "organisation": "OmniChannel Direct Marketing & Retail Group",
        "problem_key": "algorithmic_fairness_audit",
        "description": "Scoring e-commerce customer purchase propensity to allocate promotional discounts, audited across demographic tiers to prevent discriminatory digital redlining.",
        "icon": "📢",
        "endpoint_type": "binary",
        "study_design": "web_telemetry",
        "sample_scale": "large",
        "governance_priority": "algorithmic_equity",
        "takeaway": "Propensity models targeting promotional discounts must be audited with the Four-Fifths Rule (DIR >= 0.80) to guarantee price-sensitive customer segments are not systematically excluded."
    },

    "bp_saas_churn_retention": {
        "id": "bp_saas_churn_retention",
        "title": "Monthly SaaS Subscriber Churn & Retention Optimization",
        "domain": "subscriptions",
        "domain_name": "📱 Digital SaaS & Subscription Platforms",
        "organisation": "PulseStream Digital Subscriptions & SaaS Platform",
        "problem_key": "rare_event_classification",
        "description": "Detecting early user disengagement from digital product telemetry 30 days prior to contract renewal, tuning decision thresholds to maximize net retention ROI.",
        "icon": "📱",
        "endpoint_type": "binary",
        "study_design": "product_telemetry",
        "sample_scale": "medium",
        "governance_priority": "unit_economics",
        "takeaway": "Tuning classification threshold to p* = 0.26 optimizes F1-score and balances concierge outreach cost against recurring customer lifetime value (£360/year)."
    },

    "bp_logistics_warehouse_dispatch": {
        "id": "bp_logistics_warehouse_dispatch",
        "title": "Regional Warehouse Pallet Dispatch & Freight Demand",
        "domain": "supply_chain",
        "domain_name": "🚛 Supply Chain, Freight & Logistics",
        "organisation": "MetroLogix Distribution & Global Freight",
        "problem_key": "time_series_forecasting",
        "description": "Forecasting daily regional warehouse pallet dispatch volumes across a 14-day rolling horizon to contract third-party freight truck fleets and prevent loading bay demurrage.",
        "icon": "🚛",
        "endpoint_type": "time_series",
        "study_design": "longitudinal_series",
        "sample_scale": "large",
        "governance_priority": "forecasting_accuracy",
        "takeaway": "Captures rigid 7-day cyclical logistics schedules (mid-week peaks) and holiday surges using rolling-origin temporal validation, keeping dispatch error under 7% MAPE."
    }
}


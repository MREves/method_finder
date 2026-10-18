# Statistical calculations and decision engine module
from src.stats_calcs.decision_schema import recommend_statistical_test, STATISTICAL_TESTS
from src.stats_calcs.sample_size_engine import (
    calculate_sample_size_two_means,
    calculate_sample_size_paired_means,
    calculate_sample_size_anova,
    calculate_sample_size_correlation,
    calculate_sample_size_two_proportions,
    calculate_sample_size_cohort_rr,
    calculate_sample_size_case_control_or,
    calculate_sample_size_paired_proportions_mcnemar,
    calculate_sample_size_survival_logrank,
    calculate_sample_size_diagnostic_accuracy,
    apply_attrition_inflation,
    calculate_unpadded_power_after_dropout,
    apply_cluster_design_effect,
    generate_power_curve_data,
    generate_protocol_justification_statement,
    evaluate_sample_size_prerequisites,
    generate_sample_size_demo_data,
    SampleSizeDemoData
)

__all__ = [
    "recommend_statistical_test",
    "STATISTICAL_TESTS",
    "calculate_sample_size_two_means",
    "calculate_sample_size_paired_means",
    "calculate_sample_size_anova",
    "calculate_sample_size_correlation",
    "calculate_sample_size_two_proportions",
    "calculate_sample_size_cohort_rr",
    "calculate_sample_size_case_control_or",
    "calculate_sample_size_paired_proportions_mcnemar",
    "calculate_sample_size_survival_logrank",
    "calculate_sample_size_diagnostic_accuracy",
    "apply_attrition_inflation",
    "calculate_unpadded_power_after_dropout",
    "apply_cluster_design_effect",
    "generate_power_curve_data",
    "generate_protocol_justification_statement",
    "evaluate_sample_size_prerequisites",
    "generate_sample_size_demo_data",
    "SampleSizeDemoData"
]


"""Epidemiological and public health analytical methods package."""

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

from src.epi_methods.relative_risk import (
    calculate_relative_risk,
    evaluate_relative_risk_prerequisites,
    generate_relative_risk_demo_data
)

from src.epi_methods.odds_ratio import (
    calculate_odds_ratio,
    evaluate_odds_ratio_prerequisites,
    generate_odds_ratio_demo_data
)

from src.epi_methods.risk_difference import (
    calculate_risk_difference,
    evaluate_risk_difference_prerequisites,
    generate_risk_difference_demo_data
)

from src.epi_methods.incidence_rate_ratio import (
    calculate_incidence_rate_ratio,
    evaluate_irr_prerequisites,
    generate_incidence_rate_ratio_demo_data
)

from src.epi_methods.attributable_fraction import (
    calculate_attributable_fraction,
    evaluate_attributable_fraction_prerequisites,
    generate_attributable_fraction_demo_data
)

from src.epi_methods.diagnostic_accuracy import (
    calculate_wilson_score_interval,
    calculate_diagnostic_accuracy,
    evaluate_diagnostic_prerequisites,
    generate_diagnostic_accuracy_demo_data
)

from src.epi_methods.predictive_values import (
    calculate_predictive_values,
    evaluate_predictive_value_prerequisites,
    generate_predictive_values_demo_data
)

from src.epi_methods.likelihood_ratios import (
    calculate_likelihood_ratios,
    evaluate_likelihood_ratio_prerequisites,
    generate_likelihood_ratio_demo_data
)

from src.epi_methods.kaplan_meier import (
    calculate_kaplan_meier_single_group,
    calculate_log_rank_test,
    calculate_landmark_survival_probability,
    calculate_kaplan_meier_two_groups,
    evaluate_kaplan_meier_prerequisites,
    generate_kaplan_meier_demo_data
)

from src.epi_methods.cox_regression import (
    fit_cox_proportional_hazards,
    calculate_harrell_c_index,
    calculate_schoenfeld_diagnostics,
    predict_cox_survival_curve,
    evaluate_cox_regression_prerequisites,
    generate_cox_regression_demo_data
)

from src.epi_methods.risk_stratification import (
    build_integer_scoring_system,
    calculate_risk_strata_performance,
    evaluate_model_calibration,
    calculate_decision_curve_analysis,
    evaluate_risk_stratification_prerequisites,
    generate_risk_stratification_demo_data
)

__all__ = [
    "generate_derbyshire_standardization_data",
    "get_derbyshire_geojson_path",
    "get_esp_2013_standard",
    "get_excel_aligned_dataset",
    "DISTRICT_PROFILES",
    "ESP_2013_BANDS",
    "SCENARIO_CONFIGS",
    "evaluate_standardization_prerequisites",
    "calculate_direct_standardized_rate",
    "calculate_indirect_standardization",
    "calculate_dobson_dsr_ci",
    "calculate_byar_poisson_ci",
    "calculate_relative_risk",
    "evaluate_relative_risk_prerequisites",
    "generate_relative_risk_demo_data",
    "calculate_odds_ratio",
    "evaluate_odds_ratio_prerequisites",
    "generate_odds_ratio_demo_data",
    "calculate_risk_difference",
    "evaluate_risk_difference_prerequisites",
    "generate_risk_difference_demo_data",
    "calculate_incidence_rate_ratio",
    "evaluate_irr_prerequisites",
    "generate_incidence_rate_ratio_demo_data",
    "calculate_attributable_fraction",
    "evaluate_attributable_fraction_prerequisites",
    "generate_attributable_fraction_demo_data",
    "calculate_wilson_score_interval",
    "calculate_diagnostic_accuracy",
    "evaluate_diagnostic_prerequisites",
    "generate_diagnostic_accuracy_demo_data",
    "calculate_predictive_values",
    "evaluate_predictive_value_prerequisites",
    "generate_predictive_values_demo_data",
    "calculate_likelihood_ratios",
    "evaluate_likelihood_ratio_prerequisites",
    "generate_likelihood_ratio_demo_data",
    "calculate_kaplan_meier_single_group",
    "calculate_log_rank_test",
    "calculate_landmark_survival_probability",
    "calculate_kaplan_meier_two_groups",
    "evaluate_kaplan_meier_prerequisites",
    "generate_kaplan_meier_demo_data",
    "fit_cox_proportional_hazards",
    "calculate_harrell_c_index",
    "calculate_schoenfeld_diagnostics",
    "predict_cox_survival_curve",
    "evaluate_cox_regression_prerequisites",
    "generate_cox_regression_demo_data",
    "build_integer_scoring_system",
    "calculate_risk_strata_performance",
    "evaluate_model_calibration",
    "calculate_decision_curve_analysis",
    "evaluate_risk_stratification_prerequisites",
    "generate_risk_stratification_demo_data"
]



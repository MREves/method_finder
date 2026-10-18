"""Machine Learning & Predictive Analytics module.

Exports decision schemas, taxonomies, scenarios, statistical/operational calculations,
fairness audits, time series models, and diagnostic evaluators.
"""

from src.ml_models.ml_schema import (
    ML_TASK_FAMILIES,
    ML_MODELS,
    ML_DECISION_NODES,
    ML_SCENARIOS,
)

from src.ml_models.ml_engine import (
    MLDemoData,
    generate_clinical_readmission_dataset,
    generate_dna_operational_dataset,
    generate_ed_demand_time_series,
    generate_workforce_sickness_time_series,
    generate_2d_synthetic_classification,
    generate_ml_demo_data,
    calculate_algorithmic_fairness_metrics,
    calculate_time_series_forecast,
    calculate_classification_threshold_metrics,
    generate_decision_boundary_surface,
    calculate_cluster_metrics,
    calculate_pca_decomposition,
    route_ml_recommendation,
    evaluate_ml_prerequisites,
    generate_ml_python_pipeline,
)

__all__ = [
    "ML_TASK_FAMILIES",
    "ML_MODELS",
    "ML_DECISION_NODES",
    "ML_SCENARIOS",
    "MLDemoData",
    "generate_clinical_readmission_dataset",
    "generate_dna_operational_dataset",
    "generate_ed_demand_time_series",
    "generate_workforce_sickness_time_series",
    "generate_2d_synthetic_classification",
    "generate_ml_demo_data",
    "calculate_algorithmic_fairness_metrics",
    "calculate_time_series_forecast",
    "calculate_classification_threshold_metrics",
    "generate_decision_boundary_surface",
    "calculate_cluster_metrics",
    "calculate_pca_decomposition",
    "route_ml_recommendation",
    "evaluate_ml_prerequisites",
    "generate_ml_python_pipeline",
]


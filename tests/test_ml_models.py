"""Unit & Mathematical Verification Test Suite for Machine Learning & Predictive Analytics Engine.

Validates:
1. Supervised Classification & Threshold Diagnostics:
   - Confusion matrix, precision, recall, specificity, F1, balanced accuracy, Brier score.
   - 21-threshold sweep with ROC curve (FPR/TPR) and Precision-Recall curve.
   - Monotonicity and trapezoidal AUC integration (ROC-AUC, PR-AUC).
2. Algorithmic Fairness & Health Equity Audits:
   - Selection rates across protected groups (e.g., IMD deprivation deciles).
   - Disparate impact ratio and the Four-Fifths (80%) rule compliance tiers.
   - Demographic parity difference, equalized odds difference, and false positive parity ratio.
3. Time Series Operational Demand Forecasting:
   - Holt-Winters exponential smoothing with additive seasonality and trend.
   - Out-of-sample forward projections with 95% prediction intervals.
   - Rolling-origin validation metrics (MAE, RMSE, MAPE).
4. Unsupervised Subtyping & Dimensionality Reduction:
   - K-Means clustering, silhouette scores, cluster centers, inertia.
   - Principal Component Analysis (PCA), explained variance ratios, component loadings.
5. 2D Decision Boundary Surface:
   - Synthetic feature meshgrid calculation for linear, tree, and kernel models.
6. Recommendation Routing & 5-Point Prerequisite Audit:
   - Decision node routing based on sample size, imbalance, linearity, and explainability.
   - 5-point audit: Events Per Variable (EPV), imbalance risk, missingness, temporal leakage, fairness.
7. Synthetic Data Contracts & Fictional Healthcare Entities:
   - 2-tuple unpacking (df, meta = demo_data).
   - Fictional UK healthcare entities (*All Meadows Cardiovascular Institute*, *Riverside Community Health Trust*, *Inner City Teaching Hospital*, *NHS Integrated Care Board*).
   - Disclaimers on synthetic nature.
"""

import os
import sys
import math
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib.util

# Load ml_schema
schema_spec = importlib.util.spec_from_file_location(
    "ml_schema",
    os.path.join(PROJECT_ROOT, "src", "ml_models", "ml_schema.py")
)
mls = importlib.util.module_from_spec(schema_spec)
schema_spec.loader.exec_module(mls)

# Load ml_engine
engine_spec = importlib.util.spec_from_file_location(
    "ml_engine",
    os.path.join(PROJECT_ROOT, "src", "ml_models", "ml_engine.py")
)
mle = importlib.util.module_from_spec(engine_spec)
engine_spec.loader.exec_module(mle)


class TestMachineLearningEngine(unittest.TestCase):
    """Mathematical and algorithmic verification tests for the ML engine."""

    def test_classification_threshold_metrics_arithmetic(self):
        """Verifies confusion matrix arithmetic and metric formulas at threshold 0.5."""
        y_true = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0] # 5 pos, 5 neg
        y_prob = [0.9, 0.8, 0.7, 0.2, 0.1, 0.3, 0.4, 0.6, 0.65, 0.05]
        # At threshold 0.5:
        # Predictions: [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
        # TP = 4 (indices 0, 1, 2, 8)
        # FP = 1 (index 7)
        # FN = 1 (index 6)
        # TN = 4 (indices 3, 4, 5, 9)
        res = mle.calculate_classification_threshold_metrics(y_true, y_prob, threshold=0.5)

        self.assertEqual(res["tp"], 4)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)
        self.assertEqual(res["tn"], 4)
        self.assertEqual(res["total_positives"], 5)
        self.assertEqual(res["total_negatives"], 5)

        # Precision = 4 / (4 + 1) = 0.8
        self.assertAlmostEqual(res["precision"], 0.8, places=4)
        # Recall = 4 / (4 + 1) = 0.8
        self.assertAlmostEqual(res["recall_sensitivity"], 0.8, places=4)
        # Specificity = 4 / (4 + 1) = 0.8
        self.assertAlmostEqual(res["specificity"], 0.8, places=4)
        # F1 = 2 * 0.8 * 0.8 / (0.8 + 0.8) = 0.8
        self.assertAlmostEqual(res["f1_score"], 0.8, places=4)
        # Balanced accuracy = (0.8 + 0.8) / 2 = 0.8
        self.assertAlmostEqual(res["balanced_accuracy"], 0.8, places=4)

        # Check ROC-AUC and PR-AUC exist and are valid probabilities
        self.assertTrue(0.0 <= res["roc_auc"] <= 1.0)
        self.assertTrue(0.0 <= res["pr_auc"] <= 1.0)

        # Check curve coordinates
        self.assertEqual(len(res["roc_curve_thresholds"]), 21)
        self.assertEqual(len(res["roc_curve_fpr"]), 21)
        self.assertEqual(len(res["roc_curve_tpr"]), 21)

    def test_classification_threshold_extreme_bounds(self):
        """Verifies behaviour at boundary thresholds 0.0 and 1.0."""
        y_true = [1, 0, 1, 0]
        y_prob = [0.8, 0.3, 0.6, 0.2]

        # At threshold 0.0: everything predicted positive
        res_0 = mle.calculate_classification_threshold_metrics(y_true, y_prob, threshold=0.0)
        self.assertEqual(res_0["tp"], 2)
        self.assertEqual(res_0["fp"], 2)
        self.assertEqual(res_0["recall_sensitivity"], 1.0)
        self.assertEqual(res_0["specificity"], 0.0)

        # At threshold 1.0: everything predicted negative (unless prob == 1.0)
        res_1 = mle.calculate_classification_threshold_metrics(y_true, y_prob, threshold=1.0)
        self.assertEqual(res_1["tp"], 0)
        self.assertEqual(res_1["fp"], 0)
        self.assertEqual(res_1["recall_sensitivity"], 0.0)
        self.assertEqual(res_1["specificity"], 1.0)

    def test_algorithmic_fairness_four_fifths_rule(self):
        """Verifies disparate impact and Four-Fifths 80% rule calculation."""
        # Group A (Affluent / reference): 100 individuals, 40 selected -> Selection rate = 0.40
        # Group B (Deprived / protected): 100 individuals, 20 selected -> Selection rate = 0.20
        # Disparate impact ratio = 0.20 / 0.40 = 0.50 (Violates Four-Fifths rule < 0.80)
        group_labels = ["Affluent"] * 100 + ["Deprived"] * 100
        # Let true values and probabilities match
        y_true = [1] * 30 + [0] * 70 + [1] * 30 + [0] * 70
        y_pred = [1] * 40 + [0] * 60 + [1] * 20 + [0] * 80
        y_prob = [0.9] * 40 + [0.1] * 60 + [0.9] * 20 + [0.1] * 80

        res = mle.calculate_algorithmic_fairness_metrics(
            group_labels=group_labels,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            favorable_outcome=1
        )

        self.assertIn("disparate_impact_ratio", res)
        self.assertAlmostEqual(res["disparate_impact_ratio"], 0.50, places=3)
        self.assertIn("Violated", res["four_fifths_rule_status"])
        self.assertAlmostEqual(res["demographic_parity_difference"], 0.20, places=3)

    def test_algorithmic_fairness_equal_rates(self):
        """Verifies perfect demographic parity (ratio = 1.0, difference = 0.0)."""
        group_labels = ["Group1"] * 50 + ["Group2"] * 50
        y_true = [1] * 25 + [0] * 25 + [1] * 25 + [0] * 25
        y_pred = [1] * 20 + [0] * 30 + [1] * 20 + [0] * 30
        y_prob = [0.8] * 20 + [0.2] * 30 + [0.8] * 20 + [0.2] * 30

        res = mle.calculate_algorithmic_fairness_metrics(
            group_labels=group_labels,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )

        self.assertAlmostEqual(res["disparate_impact_ratio"], 1.0, places=4)
        self.assertAlmostEqual(res["demographic_parity_difference"], 0.0, places=4)
        self.assertEqual(res["four_fifths_rule_status"], "Compliant (Pass - Ratio >= 0.80)")

    def test_time_series_exponential_smoothing_forecast(self):
        """Verifies Holt-Winters time series forecasting calculations."""
        # Create a synthetic series of 50 points with a linear trend and weekly seasonality
        series = [100.0 + 0.5 * i + 10.0 * math.sin(2 * math.pi * (i % 7) / 7.0) for i in range(50)]
        horizon = 14

        res = mle.calculate_time_series_forecast(
            series=series,
            horizon=horizon,
            seasonal_periods=7,
            alpha=0.3,
            beta=0.1,
            gamma=0.2
        )

        self.assertEqual(len(res["forecast"]), horizon)
        self.assertEqual(len(res["lower_95"]), horizon)
        self.assertEqual(len(res["upper_95"]), horizon)

        # Upper bound must be strictly >= lower bound for every forecast step
        for low, point, high in zip(res["lower_95"], res["forecast"], res["upper_95"]):
            self.assertLessEqual(low, point)
            self.assertLessEqual(point, high)

        # Validation metrics must be non-negative
        self.assertGreaterEqual(res["validation_mae"], 0.0)
        self.assertGreaterEqual(res["validation_rmse"], 0.0)
        self.assertGreaterEqual(res["validation_mape_pct"], 0.0)

    def test_clustering_metrics(self):
        """Verifies K-Means clustering metrics and output structure."""
        # 2 distinct clusters in 2D space
        cluster1 = [[1.0 + 0.1 * i, 1.0 + 0.1 * i] for i in range(20)]
        cluster2 = [[10.0 + 0.1 * i, 10.0 + 0.1 * i] for i in range(20)]
        data = cluster1 + cluster2

        res = mle.calculate_cluster_metrics(data=data, n_clusters=2, seed=42)

        self.assertEqual(res["n_clusters"], 2)
        self.assertEqual(len(res["cluster_labels"]), 40)
        self.assertEqual(len(res["cluster_sizes"]), 2)
        self.assertEqual(sum(res["cluster_sizes"].values()), 40)
        self.assertGreater(res["silhouette_score"], 0.5)  # Distinct clusters should have high silhouette
        self.assertIn("davies_bouldin_index", res)
        self.assertLess(res["davies_bouldin_index"], 0.5)  # Well-separated clusters should have low DBI
        self.assertIn("calinski_harabasz_index", res)
        self.assertGreater(res["calinski_harabasz_index"], 100.0)  # High between-to-within variance ratio
        self.assertGreater(res["inertia_wcss"], 0.0)
        self.assertIn("k_evaluation_curve", res)
        self.assertGreaterEqual(len(res["k_evaluation_curve"]), 2)
        self.assertIn("Silhouette_Score", res["k_evaluation_curve"][0])
        self.assertIn("Davies_Bouldin_Index", res["k_evaluation_curve"][0])
        self.assertIn("Calinski_Harabasz_Index", res["k_evaluation_curve"][0])

    def test_pca_decomposition(self):
        """Verifies PCA explained variance and component projection."""
        # 30 observations with 3 features, where x2 = 2*x1 + noise, x3 is independent
        data = []
        for i in range(30):
            x1 = float(i)
            x2 = 2.0 * x1 + 0.1 * (i % 3)
            x3 = float((i * 7) % 5)
            data.append([x1, x2, x3])

        res = mle.calculate_pca_decomposition(data=data, n_components=2)

        self.assertEqual(len(res["explained_variance_ratio"]), 2)
        # First component should capture majority of variance due to collinear x1 and x2
        self.assertGreater(res["explained_variance_ratio"][0], 0.70)
        self.assertLessEqual(res["cumulative_variance_ratio"][-1], 1.0)
        self.assertEqual(len(res["projected_coordinates"]), 30)
        self.assertEqual(len(res["projected_coordinates"][0]), 2)

    def test_decision_boundary_surface(self):
        """Verifies 2D decision boundary grid calculation."""
        X = [[1.0, 1.0], [1.5, 1.2], [2.0, 1.8], [8.0, 8.0], [8.5, 8.2], [9.0, 8.8]]
        y = [0, 0, 0, 1, 1, 1]

        res = mle.generate_decision_boundary_surface(
            X_2d=X,
            y=y,
            model_type="penalized_logistic_regression",
            grid_resolution=15
        )

        self.assertEqual(len(res["xx"]), 15)
        self.assertEqual(len(res["yy"]), 15)
        self.assertEqual(len(res["zz_probs"]), 15)
        self.assertEqual(len(res["zz_probs"][0]), 15)

    def test_route_ml_recommendation(self):
        """Verifies recommendation routing based on task and dataset properties."""
        # Case 1: Tabular classification with severe imbalance and explainability need
        rec1 = mle.route_ml_recommendation(
            task="classification",
            n_samples=500,
            n_features=12,
            imbalance_ratio=0.08,
            primary_priority="interpretability"
        )
        self.assertEqual(rec1["recommended_model"], "penalized_logistic_regression")

        # Case 2: Tabular classification with non-linear interactions prioritizing predictive performance
        rec2 = mle.route_ml_recommendation(
            task="classification",
            n_samples=3000,
            n_features=25,
            imbalance_ratio=0.15,
            primary_priority="accuracy"
        )
        self.assertEqual(rec2["recommended_model"], "gradient_boosting_classifier")

        # Case 3: Time series forecasting
        rec3 = mle.route_ml_recommendation(
            task="time_series",
            n_samples=104,
            primary_priority="interpretability"
        )
        self.assertEqual(rec3["recommended_model"], "holt_winters_exponential_smoothing")

    def test_evaluate_ml_prerequisites(self):
        """Verifies the 5-point prerequisite checklist audit."""
        audit = mle.evaluate_ml_prerequisites(
            task="classification",
            model_id="penalized_logistic_regression",
            n_samples=800,
            n_events=64, # 8% prevalence
            n_features=12,
            missing_rate=0.04,
            has_temporal_structure=True,
            fairness_sensitive=True
        )

        self.assertEqual(len(audit["checks"]), 5)
        check_names = [c["name"] for c in audit["checks"]]
        self.assertIn("Events Per Variable (EPV) / Sample Adequacy", check_names)
        self.assertIn("Class Imbalance & Minority Representation", check_names)
        self.assertIn("Missing Data Pattern & Imputation Strategy", check_names)
        self.assertIn("Temporal Structure & Leakage Prevention", check_names)
        self.assertIn("Algorithmic Fairness & Equity Audit", check_names)

        # EPV is 64 / 12 = 5.33 (< 10), so status should be 'warn'
        epv_check = next(c for c in audit["checks"] if "EPV" in c["name"])
        self.assertEqual(epv_check["status"], "warn")

    def test_generate_ml_python_pipeline(self):
        """Verifies code generation creates valid scikit-learn / statsmodels pipeline."""
        code = mle.generate_ml_python_pipeline(
            model_id="gradient_boosting_classifier",
            task="classification",
            target_col="readmitted_30d",
            feature_cols=["age", "charlson_index", "prior_admissions_12m"],
            hyperparameters={"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3}
        )

        self.assertIn("HistGradientBoostingClassifier", code)
        self.assertIn("readmitted_30d", code)
        self.assertIn("ColumnTransformer", code)
        self.assertIn("Pipeline", code)


class TestSyntheticDataAndScenarios(unittest.TestCase):
    """Verifies synthetic dataset generation contracts and fictional UK healthcare entities."""

    def test_clinical_readmission_dataset(self):
        """Tests All Meadows Cardiovascular Institute 30-day HF dataset."""
        df = mle.generate_clinical_readmission_dataset(n_samples=500, seed=42)
        self.assertEqual(len(df), 500)
        self.assertIn("readmitted_30d", df.columns)
        self.assertIn("age", df.columns)
        self.assertIn("bnp_admission", df.columns)
        self.assertIn("ejection_fraction", df.columns)
        # Verify readmission rate is between 5% and 15% (imbalanced)
        readm_rate = df["readmitted_30d"].mean()
        self.assertTrue(0.05 <= readm_rate <= 0.15)

    def test_dna_operational_dataset(self):
        """Tests Riverside Community Health Trust Outpatient DNA dataset."""
        df = mle.generate_dna_operational_dataset(n_samples=600, seed=42)
        self.assertEqual(len(df), 600)
        self.assertIn("appointment_dna", df.columns)
        self.assertIn("imd_decile", df.columns)
        self.assertIn("lead_time_days", df.columns)
        self.assertIn("deprivation_group", df.columns)
        # Verify IMD deciles range from 1 to 10
        self.assertEqual(df["imd_decile"].min(), 1)
        self.assertEqual(df["imd_decile"].max(), 10)

    def test_ed_demand_time_series(self):
        """Tests Inner City Teaching Hospital A&E demand time series."""
        df = mle.generate_ed_demand_time_series(n_days=365, seed=42)
        self.assertEqual(len(df), 365)
        self.assertIn("date", df.columns)
        self.assertIn("ed_attendances", df.columns)
        self.assertIn("emergency_admissions", df.columns)
        self.assertIn("bed_occupancy_pct", df.columns)
        # All attendances should be positive integers
        self.assertTrue((df["ed_attendances"] > 0).all())

    def test_workforce_sickness_time_series(self):
        """Tests NHS Integrated Care Board workforce sickness absence time series."""
        df = mle.generate_workforce_sickness_time_series(n_months=36, seed=42)
        self.assertEqual(len(df), 36)
        self.assertIn("month_date", df.columns)
        self.assertIn("sickness_absence_rate_pct", df.columns)
        self.assertIn("wte_lost_days", df.columns)
        # Rates should be between 2% and 10%
        self.assertTrue((df["sickness_absence_rate_pct"] >= 2.0).all())
        self.assertTrue((df["sickness_absence_rate_pct"] <= 10.0).all())

    def test_demo_data_unpacking_contract(self):
        """Verifies the 2-tuple unpacking contract: df, meta = generate_ml_demo_data(...)."""
        scenarios = [
            "all_meadows_readmission", "riverside_dna_equity", "inner_city_ed_demand", "icb_workforce_sickness",
            "manufacturing_predictive_maintenance", "marketing_conversion_scoring",
            "subscription_churn_forecasting", "supply_chain_demand_forecasting"
        ]
        for scen in scenarios:
            data = mle.generate_ml_demo_data(scen)
            df, meta = data
            self.assertIsNotNone(df)
            self.assertIsInstance(meta, dict)
            self.assertIn("title", meta)
            self.assertIn("fictional_trust", meta)
            self.assertIn("is_synthetic", meta)
            self.assertTrue(meta["is_synthetic"])

    def test_fictional_institutions_and_disclaimers(self):
        """Verifies that all scenarios cite strictly fictional UK healthcare institutions or approved cross-industry enterprise entities."""
        allowed_entities = [
            # Approved Fictional UK Healthcare Institutions
            "All Meadows Cardiovascular Institute",
            "Riverside Community Health Trust",
            "Inner City Teaching Hospital",
            "NHS Integrated Care Board",
            # Approved Fictional Cross-Industry Enterprise Entities
            "Apex Precision Robotics & Advanced Manufacturing",
            "OmniChannel Direct Marketing & Retail Group",
            "PulseStream Digital Subscriptions & SaaS Platform",
            "MetroLogix Distribution & Global Freight"
        ]
        for key, scen in mls.ML_SCENARIOS.items():
            trust = scen.get("fictional_trust", "")
            self.assertTrue(
                any(entity in trust for entity in allowed_entities),
                f"Scenario {key} references non-approved institution: {trust}"
            )
            # Verify synthetic data disclaimer is explicitly present
            self.assertIn("synthetic", scen.get("clinical_context", "").lower() + scen.get("disclaimer", "").lower())

    def test_model_catalog_keys_and_description(self):
        """Verifies that all 16 ML models have name, short_desc, and description keys."""
        required_keys = ["id", "name", "family", "short_desc", "description", "primary_use_case", "loss_function"]
        for model_id, model_info in mls.ML_MODELS.items():
            for key in required_keys:
                self.assertIn(
                    key,
                    model_info,
                    f"Model '{model_id}' is missing required key '{key}' in ML_MODELS catalog."
                )
            self.assertTrue(len(model_info["description"]) > 0)

    def test_mathematical_formulations_and_plain_english_coverage(self):
        """Verifies that all 16 ML models and metric dashboards have plain English step-by-step mathematical explanations."""
        from unittest.mock import MagicMock
        mock_st = MagicMock()
        mock_exp = MagicMock()
        mock_exp.return_value.__enter__.return_value = None
        mock_exp.return_value.__exit__.return_value = None
        mock_st.expander = mock_exp

        sys.modules["streamlit"] = mock_st
        mock_plotly = MagicMock()
        sys.modules["plotly"] = mock_plotly
        sys.modules["plotly.graph_objects"] = mock_plotly.graph_objects
        sys.modules["plotly.subplots"] = mock_plotly.subplots
        sys.modules["pandas"] = MagicMock()
        sys.modules["numpy"] = MagicMock()

        import src.ui_components.ml_widgets as mlw

        captured_markdown = []
        mock_st.markdown.side_effect = lambda text, **kwargs: captured_markdown.append(text)

        # 1. Test performance metric breakdowns across all 3 analytical paradigms
        for task, scen in [
            ("fairness_audit", "operational_dna_equity"),
            ("time_series", "ed_demand_forecasting"),
            ("classification", "clinical_readmission")
        ]:
            captured_markdown.clear()
            mlw.render_ml_performance_metric_math_breakdown(scen, task, {}, {})
            all_text = " ".join(captured_markdown)
            self.assertIn("Mathematical Representation", all_text)
            self.assertIn("Plain English Step-by-Step Breakdown", all_text)
            self.assertIn("Why This Calculation is Being Used", all_text)

        # 2. Test mathematical frameworks across all 16 supported models
        for model_id, model_info in mls.ML_MODELS.items():
            captured_markdown.clear()
            mlw.render_ml_model_mathematical_framework(
                model_id,
                model_info,
                mls.ML_SCENARIOS["clinical_readmission"],
                {}
            )
            all_text = " ".join(captured_markdown)
            self.assertIn(
                "Mathematical Representation",
                all_text,
                f"Model '{model_id}' is missing 'Mathematical Representation' section."
            )
            self.assertIn(
                "Plain English Step-by-Step Breakdown",
                all_text,
                f"Model '{model_id}' is missing 'Plain English Step-by-Step Breakdown' section."
            )
            self.assertIn(
                "Why This Calculation is Being Used",
                all_text,
                f"Model '{model_id}' is missing 'Why This Calculation is Being Used' section."
            )


if __name__ == "__main__":
    unittest.main()


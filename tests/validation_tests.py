"""Validation Test Suite for Statistical Tests Finder.

Verifies:
1. Decision Schema & Metadata integrity across all methods (including Binary Logistic Regression).
2. Decision routing logic and deterministic recommendation outcomes.
3. Sample data generation and download guidance for all 18 statistical methods.
4. UI component expander constraints (ensuring all expanders are collapsed by default).
5. Synthetic data generator & statistical calculation engine validation (when scientific stack is present).
"""

import os
import re
import unittest
from src.stats_calcs.decision_schema import (
    recommend_statistical_test,
    get_statistical_test_recommendation,
    get_all_statistical_tests,
    STATISTICAL_TESTS
)
from src.utils import format_p_value, format_ci

try:
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    from src.stats_calcs.synthetic_data import (
        generate_binary_logistic_regression_demo,
        generate_linear_regression_demo,
        generate_one_sample_demo,
        generate_contingency_demo,
        generate_fisher_exact_demo,
        generate_mcnemar_demo
    )
    from src.ui_components.stats_widgets import (
        _generate_fallback_dataset,
        _get_sample_data_guide_info,
        render_binary_logistic_regression_excel_guide,
        render_binary_logistic_regression_python_guide,
        render_binary_logistic_regression_demo
    )
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestDecisionSchemaValidation(unittest.TestCase):
    """Validates structural integrity and completeness of the decision schema."""

    def test_all_18_tests_exist(self):
        expected_tests = [
            "one_sample_t_test", "wilcoxon_signed_rank", "welch_t_test",
            "students_t_test", "mann_whitney_u", "paired_t_test",
            "wilcoxon_paired", "one_way_anova", "kruskal_wallis",
            "repeated_measures_anova", "friedman_test", "chi_square_independence",
            "fisher_exact", "mcnemar_test", "pearson_correlation",
            "spearman_correlation", "linear_regression", "binary_logistic_regression"
        ]
        for tk in expected_tests:
            self.assertIn(tk, STATISTICAL_TESTS, f"Missing test key in schema: {tk}")
            info = STATISTICAL_TESTS[tk]
            self.assertIn("name", info)
            self.assertIn("category", info)
            self.assertIn("parametric", info)
            self.assertIn("description", info)
            self.assertIn("when_to_use", info)
            self.assertIn("key_assumptions", info)
            self.assertIn("alternative_test", info)
            self.assertIn("excel_guide", info)
            self.assertIn("python_snippet", info)
            self.assertIn("demo_generator", info)

    def test_binary_logistic_regression_schema_details(self):
        rec = STATISTICAL_TESTS["binary_logistic_regression"]
        self.assertEqual(rec["name"], "Binary Logistic Regression")
        self.assertTrue(rec["parametric"])
        self.assertIn("Odds", rec["description"])
        
        # Verify 4-step pipeline snippet includes assumption checks and conditional statements
        snippet = rec["python_snippet"]
        self.assertIn("run_binary_logistic_pipeline", snippet)
        self.assertIn("[ASSURANCE - PASS]", snippet)
        self.assertIn("[WARNING", snippet)
        self.assertIn("[INAPPROPRIATE - FAIL]", snippet)
        self.assertIn("EPV", snippet)
        self.assertIn("variance_inflation_factor", snippet)
        self.assertIn("sm.Logit", snippet)
        self.assertIn("np.exp(model.params)", snippet)


class TestDecisionRoutingValidation(unittest.TestCase):
    """Validates deterministic recommendation routing."""

    def test_binary_logistic_routing_or_retrieval(self):
        rec = get_statistical_test_recommendation("binary_logistic_regression")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["name"], "Binary Logistic Regression")
        self.assertEqual(rec["test_key"], "binary_logistic_regression")

    def test_get_all_statistical_tests(self):
        all_tests = get_all_statistical_tests()
        self.assertGreaterEqual(len(all_tests), 18)
        keys = [t.get("key") or t.get("test_key") for t in all_tests]
        self.assertIn("binary_logistic_regression", keys)


class TestExpanderConstraints(unittest.TestCase):
    """Validates that all st.expander calls across the codebase are collapsed by default."""

    def test_no_expanded_true_in_codebase(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Search specifically within application source and pages
        target_dirs = [
            os.path.join(project_root, "pages"),
            os.path.join(project_root, "src"),
            project_root  # top-level scripts like app.py
        ]
        
        target_str = "expanded" + "=True"
        expanded_true_matches = []
        for tdir in target_dirs:
            for root, _, files in os.walk(tdir):
                if ".git" in root or "__pycache__" in root or "tests" in root:
                    continue
                for f in files:
                    if f.endswith(".py"):
                        pf = os.path.join(root, f)
                        with open(pf, "r", encoding="utf-8", errors="ignore") as file_obj:
                            content = file_obj.read()
                            if target_str in content:
                                expanded_true_matches.append(pf)

        self.assertEqual(
            expanded_true_matches,
            [],
            f"Found forbidden '{target_str}' in application files: {expanded_true_matches}. All expanders must have expanded=False."
        )


class TestSampleDataDownloadsValidation(unittest.TestCase):
    """Tests sample dataset generation and download guidance for all 18 methods."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack not installed in environment")

    def test_all_18_tests_have_sample_data_guidance(self):
        test_keys = [
            "one_sample_t_test", "wilcoxon_signed_rank", "welch_t_test",
            "students_t_test", "mann_whitney_u", "paired_t_test",
            "wilcoxon_paired", "one_way_anova", "kruskal_wallis",
            "repeated_measures_anova", "friedman_test", "chi_square_independence",
            "fisher_exact", "mcnemar_test", "pearson_correlation",
            "spearman_correlation", "linear_regression", "binary_logistic_regression"
        ]

        for tk in test_keys:
            rec = STATISTICAL_TESTS[tk]
            df, res = _generate_fallback_dataset(tk)
            self.assertIsNotNone(df, f"Fallback df should not be None for {tk}")
            self.assertGreater(len(df), 0, f"Fallback df should not be empty for {tk}")

            info = _get_sample_data_guide_info(tk, rec, df, res)
            self.assertIn("default_filename", info, f"Missing default_filename for {tk}")
            self.assertTrue(info["default_filename"].endswith(".csv"), f"Filename should end in .csv for {tk}")
            self.assertIn("excel_instructions", info, f"Missing excel_instructions for {tk}")
            self.assertIn("python_code", info, f"Missing python_code for {tk}")
            self.assertIn("pd.read_csv", info["python_code"], f"Python snippet should include pd.read_csv for {tk}")

    def test_binary_logistic_sample_data_guidance(self):
        tk = "binary_logistic_regression"
        rec = STATISTICAL_TESTS[tk]
        df, res = _generate_fallback_dataset(tk)
        self.assertIn("CVD_Event", df.columns)
        self.assertIn("Age_Years", df.columns)
        self.assertIn("Systolic_BP", df.columns)
        self.assertIn("Current_Smoker", df.columns)

        info = _get_sample_data_guide_info(tk, rec, df, res)
        self.assertEqual(info["default_filename"], "binary_logistic_regression_sample_data.csv")
        self.assertIn("GRG Nonlinear", info["excel_instructions"])
        self.assertIn("run_binary_logistic_pipeline", info["python_code"])


class TestBinaryLogisticRegressionEngine(unittest.TestCase):
    """Tests the Binary Logistic Regression synthetic data generator and statistical engine."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack not installed in environment")

    def test_cardiology_scenario(self):
        df, results = generate_binary_logistic_regression_demo(
            n_pts=120, n_predictors=3, scenario="cardiology", seed=42
        )
        self.assertEqual(len(df), 120)
        self.assertEqual(results["n_pts"], 120)
        self.assertEqual(results["k_preds"], 3)
        self.assertEqual(results["outcome_name"], "CVD_Event")
        self.assertEqual(len(results["pred_names"]), 3)
        self.assertEqual(len(results["beta"]), 4)
        self.assertEqual(len(results["se"]), 4)
        self.assertEqual(len(results["odds_ratios"]), 4)
        self.assertEqual(len(results["or_ci_lower"]), 4)
        self.assertEqual(len(results["or_ci_upper"]), 4)

        # Goodness of fit
        self.assertIn("ll_model", results)
        self.assertIn("ll_null", results)
        self.assertIn("lr_stat", results)
        self.assertIn("p_omnibus", results)
        self.assertGreaterEqual(results["pseudo_r2_mcfadden"], 0.0)
        self.assertLessEqual(results["pseudo_r2_mcfadden"], 1.0)

        # Discrimination & Classification
        self.assertGreaterEqual(results["auc"], 0.5)
        self.assertLessEqual(results["auc"], 1.0)
        self.assertGreaterEqual(results["accuracy"], 0.0)
        self.assertLessEqual(results["accuracy"], 1.0)
        self.assertEqual(results["tp"] + results["tn"] + results["fp"] + results["fn"], 120)

        # Assumptions
        self.assertEqual(len(results["assumptions_eval"]), 5)
        asm_names = [a["name"] for a in results["assumptions_eval"]]
        self.assertTrue(any("Binary Dependent Variable" in n for n in asm_names))
        self.assertTrue(any("Events Per Variable" in n for n in asm_names))
        self.assertTrue(any("Multicollinearity" in n for n in asm_names))
        self.assertTrue(any("Separation" in n for n in asm_names))
        self.assertTrue(any("Residual" in n for n in asm_names))

    def test_readmission_scenario(self):
        df, results = generate_binary_logistic_regression_demo(
            n_pts=100, n_predictors=2, scenario="readmission", seed=42
        )
        self.assertEqual(results["k_preds"], 2)
        self.assertEqual(results["outcome_name"], "Readmission_30d")
        self.assertGreaterEqual(results["auc"], 0.5)

    def test_exam_pass_scenario(self):
        df, results = generate_binary_logistic_regression_demo(
            n_pts=80, n_predictors=1, scenario="exam_pass", seed=42
        )
        self.assertEqual(results["k_preds"], 1)
        self.assertEqual(results["outcome_name"], "Exam_Passed")
        self.assertEqual(len(results["pred_names"]), 1)
        self.assertEqual(len(results["beta"]), 2)

    def test_multicollinear_scenario(self):
        df, results = generate_binary_logistic_regression_demo(
            n_pts=90, scenario="multicollinear", seed=42
        )
        self.assertGreaterEqual(results["max_vif"], 3.0)

    def test_sparse_events_scenario(self):
        df, results = generate_binary_logistic_regression_demo(
            n_pts=50, scenario="sparse_events", seed=42
        )
        epv_card = next(a for a in results["assumptions_eval"] if "Events Per Variable" in a["name"])
        self.assertIn("EPV", epv_card["verdict"])

    def test_custom_logistic_dimensions(self):
        for k in [1, 2, 3]:
            df, results = generate_binary_logistic_regression_demo(
                n_pts=60, n_predictors=k, scenario="custom", seed=42
            )
            self.assertEqual(results["k_preds"], k)
            self.assertEqual(len(results["pred_names"]), k)
            self.assertEqual(len(results["beta"]), k + 1)
            self.assertEqual(results["lr_df"], k)


if __name__ == "__main__":
    unittest.main()

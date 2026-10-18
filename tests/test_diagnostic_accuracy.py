"""Unit Test Suite for Diagnostic Test Accuracy & Screening Evaluation.

Validates:
1. Wilson Score 95% Confidence Intervals for Proportions (CLSI EP12-A2 standard).
2. Diagnostic Sensitivity (TPR, Recall) & False Negative Rate (FNR, Miss Rate).
3. Diagnostic Specificity (TNR) & False Positive Rate (FPR, Type I Error).
4. Overall Diagnostic Accuracy & Sample Prevalence.
5. Youden's Index J (Informedness = Sensitivity + Specificity - 1).
6. Diagnostic Odds Ratio (DOR) and Haldane-Anscombe continuity correction for zero cells.
7. Tri-color bias evaluations (Incorporation bias, diagnostic review bias, spectrum bias, cell sizes).
8. Synthetic clinical demo datasets (Derbyshire AI Retinopathy, Mobile Mammography, Pediatric Rapid RSV).
"""

import unittest

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    from src.epi_methods.diagnostic_accuracy import (
        calculate_wilson_score_interval,
        calculate_diagnostic_accuracy,
        evaluate_diagnostic_prerequisites,
        generate_diagnostic_accuracy_demo_data
    )
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestWilsonScoreInterval(unittest.TestCase):
    """Tests the Wilson score interval calculations for binomial proportions."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_wilson_interval_standard_case(self):
        # 188 successes out of 200 trials (p = 0.94)
        lower, upper = calculate_wilson_score_interval(188, 200, alpha=0.05)
        self.assertGreater(lower, 0.89)
        self.assertLess(lower, 0.94)
        self.assertGreater(upper, 0.94)
        self.assertLess(upper, 0.97)
        self.assertLessEqual(upper, 1.0)
        self.assertGreaterEqual(lower, 0.0)

    def test_wilson_interval_zero_successes(self):
        # 0 successes out of 50 trials
        lower, upper = calculate_wilson_score_interval(0, 50, alpha=0.05)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)
        self.assertLess(upper, 0.10)

    def test_wilson_interval_all_successes(self):
        # 50 successes out of 50 trials (p = 1.0)
        lower, upper = calculate_wilson_score_interval(50, 50, alpha=0.05)
        self.assertGreater(lower, 0.90)
        self.assertEqual(upper, 1.0)

    def test_wilson_interval_zero_denominator(self):
        lower, upper = calculate_wilson_score_interval(0, 0, alpha=0.05)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.0)


class TestDiagnosticAccuracyCalculations(unittest.TestCase):
    """Tests Sensitivity, Specificity, Accuracy, Youden's J, and DOR."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_diabetic_retinopathy_screening(self):
        # tp=188, fp=45, fn=12, tn=755
        # Total diseased = 188 + 12 = 200
        # Total healthy = 45 + 755 = 800
        # Grand total N = 1000
        res = calculate_diagnostic_accuracy(tp=188, fp=45, fn=12, tn=755)

        self.assertEqual(res["total_diseased"], 200)
        self.assertEqual(res["total_healthy"], 800)
        self.assertEqual(res["total_sample"], 1000)
        self.assertAlmostEqual(res["prevalence"], 0.20, places=4)

        # 1. Sensitivity & FNR
        expected_sens = 188 / 200  # 0.94
        self.assertAlmostEqual(res["sensitivity"], expected_sens, places=4)
        self.assertAlmostEqual(res["fnr"], 12 / 200, places=4)
        self.assertAlmostEqual(res["sensitivity"] + res["fnr"], 1.0, places=5)

        # Wilson CI for Sensitivity
        self.assertGreater(res["sensitivity_ci_lower"], 0.89)
        self.assertLess(res["sensitivity_ci_upper"], 0.97)

        # 2. Specificity & FPR
        expected_spec = 755 / 800  # 0.94375
        self.assertAlmostEqual(res["specificity"], expected_spec, places=4)
        self.assertAlmostEqual(res["fpr"], 45 / 800, places=4)
        self.assertAlmostEqual(res["specificity"] + res["fpr"], 1.0, places=5)

        # Wilson CI for Specificity
        self.assertGreater(res["specificity_ci_lower"], 0.92)
        self.assertLess(res["specificity_ci_upper"], 0.96)

        # 3. Overall Diagnostic Accuracy
        expected_acc = (188 + 755) / 1000  # 0.943
        self.assertAlmostEqual(res["accuracy"], expected_acc, places=4)

        # 4. Youden's Index J = Sensitivity + Specificity - 1
        expected_j = expected_sens + expected_spec - 1.0
        self.assertAlmostEqual(res["youdens_j"], expected_j, places=4)
        self.assertAlmostEqual(res["youdens_j"], 0.88375, places=4)

        # 5. Diagnostic Odds Ratio (DOR) = (TP * TN) / (FP * FN)
        expected_dor = (188 * 755) / (45 * 12)
        self.assertAlmostEqual(res["dor"], expected_dor, places=2)
        self.assertGreater(res["dor_ci_lower"], 100.0)
        self.assertGreater(res["dor_ci_upper"], res["dor"])

    def test_rapid_rsv_high_specificity_scenario(self):
        # High specificity test: tp=140, fp=4, fn=60, tn=396
        # Diseased = 200, Healthy = 400
        res = calculate_diagnostic_accuracy(tp=140, fp=4, fn=60, tn=396)

        # Sensitivity = 140 / 200 = 0.70
        self.assertAlmostEqual(res["sensitivity"], 0.70, places=4)
        # Specificity = 396 / 400 = 0.99
        self.assertAlmostEqual(res["specificity"], 0.99, places=4)
        # FPR = 4 / 400 = 0.01 (1.0%)
        self.assertAlmostEqual(res["fpr"], 0.01, places=4)
        # Youden's J = 0.70 + 0.99 - 1 = 0.69
        self.assertAlmostEqual(res["youdens_j"], 0.69, places=4)

    def test_zero_cells_continuity_correction(self):
        # Case where fp = 0 (100% specificity)
        res = calculate_diagnostic_accuracy(tp=50, fp=0, fn=5, tn=95)
        self.assertTrue(res["continuity_applied"])
        self.assertAlmostEqual(res["specificity"], 1.0, places=4)
        self.assertFalse(np.isnan(res["dor"]))
        self.assertGreater(res["dor"], 500.0)


class TestDiagnosticPrerequisites(unittest.TestCase):
    """Tests STARD & QUADAS-2 bias evaluation functions."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_all_passing_conditions(self):
        evals = evaluate_diagnostic_prerequisites(
            tp=188, fp=45, fn=12, tn=755,
            is_gold_standard_independent=True,
            is_blinded_interpretation=True,
            is_spectrum_representative=True
        )
        self.assertEqual(len(evals), 4)
        for ev in evals:
            self.assertEqual(ev["status"], "PASS")

    def test_incorporation_bias_failure(self):
        evals = evaluate_diagnostic_prerequisites(
            tp=100, fp=10, fn=10, tn=100,
            is_gold_standard_independent=False
        )
        indep_eval = next(e for e in evals if e["name"] == "Reference Standard Independence")
        self.assertEqual(indep_eval["status"], "FAIL")
        self.assertIn("Incorporation Bias", indep_eval["details"])

    def test_review_bias_warning(self):
        evals = evaluate_diagnostic_prerequisites(
            tp=100, fp=10, fn=10, tn=100,
            is_blinded_interpretation=False
        )
        blind_eval = next(e for e in evals if e["name"] == "Blinded Diagnostic Review")
        self.assertEqual(blind_eval["status"], "WARNING")

    def test_small_sample_size_failure(self):
        # Diseased = 5 + 2 = 7 < 15
        evals = evaluate_diagnostic_prerequisites(tp=5, fp=3, fn=2, tn=40)
        size_eval = next(e for e in evals if "Sample Size" in e["name"])
        self.assertEqual(size_eval["status"], "FAIL")


class TestDiagnosticDataGeneration(unittest.TestCase):
    """Tests demo data generators and synthetic schemas."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_generate_diabetic_retinopathy(self):
        df_cont, df_micro, meta = generate_diagnostic_accuracy_demo_data("diabetic_retinopathy")
        self.assertIsInstance(df_cont, pd.DataFrame)
        self.assertIsInstance(df_micro, pd.DataFrame)
        self.assertEqual(len(df_cont), 3)
        self.assertGreater(len(df_micro), 50)
        self.assertEqual(meta["tp"], 188)
        self.assertEqual(meta["fp"], 45)
        self.assertEqual(meta["fn"], 12)
        self.assertEqual(meta["tn"], 755)
        self.assertIn("disclaimer", meta)
        self.assertIn("synthetic data", meta["disclaimer"].lower())

    def test_generate_all_scenarios(self):
        for scen in ["diabetic_retinopathy", "breast_mammography", "rapid_rsv_antigen"]:
            df_cont, df_micro, meta = generate_diagnostic_accuracy_demo_data(scen)
            self.assertIsNotNone(df_cont)
            self.assertIsNotNone(df_micro)
            self.assertIn("calcs", meta)
            self.assertGreater(meta["calcs"]["sensitivity"], 0.0)
            self.assertGreater(meta["calcs"]["specificity"], 0.0)


if __name__ == "__main__":
    unittest.main()


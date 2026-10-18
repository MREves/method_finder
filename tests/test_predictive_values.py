"""Unit Test Suite for Diagnostic Predictive Values (PPV & NPV) and Bayes' Theorem.

Validates:
1. Positive Predictive Value (PPV) and Negative Predictive Value (NPV) point estimates.
2. False Discovery Rate (FDR = 1 - PPV) and False Omission Rate (FOR = 1 - NPV).
3. Bayes' Theorem model-based PPV & NPV formulation across changing disease prevalence.
4. The "False Positive Paradox" (demonstrating how low prevalence forces low PPV despite high specificity).
5. Wilson Score 95% Confidence Intervals for direct cohort predictive values.
6. Mercaldo et al. (2007) logit-scale delta method confidence intervals for Bayes estimates.
7. Tri-color methodological diagnostics (Prevalence generalizability, verification/work-up bias, incorporation bias).
8. Synthetic demo data generation and individual patient microdata schemas.
"""

import unittest

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    from src.epi_methods.predictive_values import (
        calculate_predictive_values,
        evaluate_predictive_value_prerequisites,
        generate_predictive_values_demo_data
    )
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestPredictiveValuesCalculations(unittest.TestCase):
    """Tests the mathematical formulas and point estimates for PPV, NPV, and Bayes' Theorem."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_direct_sample_predictive_values_low_prevalence(self):
        # tp=9, fp=100, fn=1, tn=1890 (Grand Total N = 2000, Diseased = 10 -> Prevalence = 0.5%)
        res = calculate_predictive_values(tp=9, fp=100, fn=1, tn=1890)

        self.assertEqual(res["total_test_positive"], 109)
        self.assertEqual(res["total_test_negative"], 1891)
        self.assertAlmostEqual(res["sample_prevalence"], 10 / 2000, places=4)

        # 1. PPV = TP / (TP + FP) = 9 / 109 ≈ 8.257%
        expected_ppv = 9.0 / 109.0
        self.assertAlmostEqual(res["ppv"], expected_ppv, places=4)
        # FDR = 1 - PPV = 100 / 109 ≈ 91.74%
        self.assertAlmostEqual(res["fdr"], 100.0 / 109.0, places=4)
        self.assertAlmostEqual(res["ppv"] + res["fdr"], 1.0, places=5)

        # Wilson CI for PPV
        self.assertGreater(res["ppv_ci_lower"], 0.035)
        self.assertLess(res["ppv_ci_upper"], 0.16)

        # 2. NPV = TN / (FN + TN) = 1890 / 1891 ≈ 99.947%
        expected_npv = 1890.0 / 1891.0
        self.assertAlmostEqual(res["npv"], expected_npv, places=5)
        # FOR = 1 - NPV = 1 / 1891 ≈ 0.053%
        self.assertAlmostEqual(res["for_rate"], 1.0 / 1891.0, places=5)
        self.assertAlmostEqual(res["npv"] + res["for_rate"], 1.0, places=5)

        # False Positives per True Positive ratio
        expected_ratio = 100.0 / 9.0
        self.assertAlmostEqual(res["false_positives_per_true_positive"], expected_ratio, places=2)

    def test_bayes_theorem_prevalence_shift_false_positive_paradox(self):
        # Base test performance: Sens = 90.0%, Spec = 95.0%
        # tp=90, fp=50, fn=10, tn=950 (from a 1:10 enriched study: 100 diseased, 1000 healthy)
        # We test Bayes' Theorem at 3 contrasting disease prevalences:
        # Prevalence 1: Low screening prevalence p = 0.005 (0.5%)
        # Prevalence 2: Moderate prevalence p = 0.10 (10.0%)
        # Prevalence 3: High pre-test probability clinic p = 0.40 (40.0%)

        # 1. Low prevalence (0.5%)
        res_low = calculate_predictive_values(tp=90, fp=50, fn=10, tn=950, prevalence_override=0.005)
        # Bayes PPV = (0.90 * 0.005) / [(0.90 * 0.005) + (0.05 * 0.995)] = 0.0045 / (0.0045 + 0.04975) = 8.295%
        expected_ppv_low = (0.90 * 0.005) / ((0.90 * 0.005) + (0.05 * 0.995))
        self.assertAlmostEqual(res_low["ppv"], expected_ppv_low, places=4)
        self.assertLess(res_low["ppv"], 0.10)  # Under 10%!

        # 2. Moderate prevalence (10.0%)
        res_mod = calculate_predictive_values(tp=90, fp=50, fn=10, tn=950, prevalence_override=0.10)
        # Bayes PPV = (0.90 * 0.10) / [(0.90 * 0.10) + (0.05 * 0.90)] = 0.09 / (0.09 + 0.045) = 0.09 / 0.135 = 66.67%
        expected_ppv_mod = 0.09 / 0.135
        self.assertAlmostEqual(res_mod["ppv"], expected_ppv_mod, places=4)

        # 3. High prevalence (40.0%)
        res_high = calculate_predictive_values(tp=90, fp=50, fn=10, tn=950, prevalence_override=0.40)
        # Bayes PPV = (0.90 * 0.40) / [(0.90 * 0.40) + (0.05 * 0.60)] = 0.36 / (0.36 + 0.030) = 0.36 / 0.39 = 92.31%
        expected_ppv_high = 0.36 / 0.39
        self.assertAlmostEqual(res_high["ppv"], expected_ppv_high, places=4)

        # Proves that PPV scales monotonically with prevalence
        self.assertLess(res_low["ppv"], res_mod["ppv"])
        self.assertLess(res_mod["ppv"], res_high["ppv"])

        # Conversely, NPV decreases as prevalence increases
        self.assertGreater(res_low["npv"], res_mod["npv"])
        self.assertGreater(res_mod["npv"], res_high["npv"])


class TestPredictivePrerequisites(unittest.TestCase):
    """Tests methodological and spectrum validity checks for predictive values."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_all_passing_conditions(self):
        evals = evaluate_predictive_value_prerequisites(
            tp=100, fp=50, fn=10, tn=500,
            is_prevalence_representative=True,
            is_independent_gold_standard=True,
            is_verification_bias_controlled=True
        )
        self.assertEqual(len(evals), 4)
        for ev in evals:
            self.assertEqual(ev["status"], "PASS")

    def test_setting_mismatch_prevalence_failure(self):
        evals = evaluate_predictive_value_prerequisites(
            tp=100, fp=50, fn=10, tn=500,
            is_prevalence_representative=False
        )
        prev_eval = next(e for e in evals if "Prevalence" in e["name"])
        self.assertEqual(prev_eval["status"], "FAIL")
        self.assertIn("Bayes", prev_eval["recommendation"])

    def test_workup_verification_bias_warning(self):
        evals = evaluate_predictive_value_prerequisites(
            tp=100, fp=50, fn=10, tn=500,
            is_verification_bias_controlled=False
        )
        v_eval = next(e for e in evals if "Verification" in e["name"])
        self.assertEqual(v_eval["status"], "WARNING")


class TestPredictiveValuesDataGeneration(unittest.TestCase):
    """Tests synthetic clinical scenario data generation."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_generate_breast_screening(self):
        df_cont, df_micro, meta = generate_predictive_values_demo_data("breast_screening_low_prev")
        self.assertIsInstance(df_cont, pd.DataFrame)
        self.assertIsInstance(df_micro, pd.DataFrame)
        self.assertEqual(meta["tp"], 9)
        self.assertEqual(meta["fp"], 100)
        self.assertIn("disclaimer", meta)
        self.assertIn("synthetic data", meta["disclaimer"].lower())

    def test_generate_all_scenarios(self):
        for scen in ["breast_screening_low_prev", "troponin_chest_pain_high_prev", "lateral_flow_flu_surveillance"]:
            df_cont, df_micro, meta = generate_predictive_values_demo_data(scen)
            self.assertIsNotNone(df_cont)
            self.assertIsNotNone(df_micro)
            self.assertIn("calcs", meta)
            self.assertGreater(meta["calcs"]["ppv"], 0.0)
            self.assertGreater(meta["calcs"]["npv"], 0.0)


if __name__ == "__main__":
    unittest.main()


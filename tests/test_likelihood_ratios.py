"""Unit Test Suite for Diagnostic Likelihood Ratios (LR+, LR-) and Fagan's Nomogram.

Validates:
1. Positive Likelihood Ratio (LR+) and Negative Likelihood Ratio (LR-) calculations.
2. Pre-to-post-test odds transformations via the Odds Form of Bayes' Theorem.
3. Simel, Samsa & Matchar (1991) log-scale 95% confidence intervals for LR+ and LR-.
4. Clinical strength classifications (conclusive rule-in, conclusive rule-out).
5. Continuity correction for zero cells in 2×2 contingency tables.
6. Diagnostic prerequisite cards (incorporation bias, spectrum bias, review bias).
7. Synthetic clinical demo datasets and microdata schemas.
"""

import unittest

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    from src.epi_methods.likelihood_ratios import (
        calculate_likelihood_ratios,
        evaluate_likelihood_ratio_prerequisites,
        generate_likelihood_ratio_demo_data
    )
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestLikelihoodRatioCalculations(unittest.TestCase):
    """Tests LR+, LR-, pre/post-test odds, and confidence interval math."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_pulmonary_embolism_rule_out(self):
        # tp=192, fp=360, fn=8, tn=440 (Diseased = 200, Healthy = 800, Total = 1000)
        # Pre-test probability = 20.0% (Wells Score intermediate risk)
        res = calculate_likelihood_ratios(tp=192, fp=360, fn=8, tn=440, pre_test_prob=0.20)

        # 1. Sensitivity & Specificity
        self.assertAlmostEqual(res["sensitivity"], 0.96, places=4)
        self.assertAlmostEqual(res["specificity"], 0.55, places=4)
        self.assertAlmostEqual(res["fpr"], 0.45, places=4)
        self.assertAlmostEqual(res["fnr"], 0.04, places=4)

        # 2. Positive Likelihood Ratio (LR+) = 0.96 / 0.45 ≈ 2.1333
        expected_lr_pos = 0.96 / 0.45
        self.assertAlmostEqual(res["lr_pos"], expected_lr_pos, places=3)
        self.assertGreater(res["lr_pos_ci_lower"], 1.90)
        self.assertLess(res["lr_pos_ci_upper"], 2.40)

        # 3. Negative Likelihood Ratio (LR-) = 0.04 / 0.55 ≈ 0.072727
        expected_lr_neg = 0.04 / 0.55
        self.assertAlmostEqual(res["lr_neg"], expected_lr_neg, places=4)
        self.assertLess(res["lr_neg"], 0.10)  # Conclusive rule-out threshold
        self.assertIn("Conclusive Rule-Out", res["clinical_strength_neg"])

        # 4. Pre-to-Post-Test Probability Updating
        # Pre-Test Odds = 0.20 / 0.80 = 0.25
        self.assertAlmostEqual(res["pre_test_odds"], 0.25, places=4)

        # Post-Test for Negative Result:
        # Post-Odds (-) = 0.25 * 0.072727 = 0.018182
        # Post-Prob (-) = 0.018182 / 1.018182 ≈ 0.017857 (1.79%)
        expected_post_prob_neg = (0.25 * expected_lr_neg) / (1.0 + 0.25 * expected_lr_neg)
        self.assertAlmostEqual(res["post_test_prob_neg"], expected_post_prob_neg, places=4)
        self.assertLess(res["post_test_prob_neg"], 0.02)  # Safely dropped below 2%!

        # Post-Test for Positive Result:
        # Post-Odds (+) = 0.25 * 2.1333 = 0.5333
        # Post-Prob (+) = 0.5333 / 1.5333 ≈ 0.3478 (34.8%)
        expected_post_prob_pos = (0.25 * expected_lr_pos) / (1.0 + 0.25 * expected_lr_pos)
        self.assertAlmostEqual(res["post_test_prob_pos"], expected_post_prob_pos, places=4)

        # 5. Diagnostic Odds Ratio (DOR) = (192 * 440) / (360 * 8) = 84480 / 2880 = 29.333
        expected_dor = (192 * 440) / (360 * 8)
        self.assertAlmostEqual(res["dor"], expected_dor, places=2)

    def test_strep_throat_rule_in(self):
        # tp=197, fp=22, fn=43, tn=538 (Diseased = 240, Healthy = 560)
        # Pre-test prob = 30.0%
        res = calculate_likelihood_ratios(tp=197, fp=22, fn=43, tn=538, pre_test_prob=0.30)

        # High specificity test -> High LR+
        self.assertGreater(res["lr_pos"], 15.0)
        self.assertIn("Conclusive Rule-In", res["clinical_strength_pos"])

        # Post-test prob (+) should jump from 30% to near 90%
        self.assertGreater(res["post_test_prob_pos"], 0.85)

    def test_zero_cells_continuity_correction(self):
        # fp = 0 (100% specificity)
        res = calculate_likelihood_ratios(tp=50, fp=0, fn=5, tn=95, pre_test_prob=0.10)
        self.assertTrue(res["continuity_applied"])
        self.assertFalse(np.isnan(res["lr_pos"]))
        self.assertFalse(np.isinf(res["lr_pos"]))
        self.assertGreater(res["lr_pos"], 50.0)


class TestLikelihoodRatioPrerequisites(unittest.TestCase):
    """Tests STARD & QUADAS-2 diagnostic cards for likelihood ratios."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_all_passing_conditions(self):
        evals = evaluate_likelihood_ratio_prerequisites(
            tp=192, fp=360, fn=8, tn=440,
            is_gold_standard_independent=True,
            is_spectrum_representative=True,
            is_blinded_interpretation=True
        )
        self.assertEqual(len(evals), 4)
        for ev in evals:
            self.assertEqual(ev["status"], "PASS")

    def test_incorporation_bias_failure(self):
        evals = evaluate_likelihood_ratio_prerequisites(
            tp=100, fp=50, fn=10, tn=200,
            is_gold_standard_independent=False
        )
        indep_eval = next(e for e in evals if "Reference Standard" in e["name"])
        self.assertEqual(indep_eval["status"], "FAIL")


class TestLikelihoodRatioDataGeneration(unittest.TestCase):
    """Tests clinical demo scenarios and microdata generation."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_generate_all_scenarios(self):
        for scen in ["pulmonary_embolism_d_dimer", "strep_throat_centor_rapid_antigen", "appendicitis_alvarado_ultrasound"]:
            df_cont, df_micro, meta = generate_likelihood_ratio_demo_data(scen)
            self.assertIsInstance(df_cont, pd.DataFrame)
            self.assertIsInstance(df_micro, pd.DataFrame)
            self.assertGreater(len(df_micro), 50)
            self.assertIn("disclaimer", meta)
            self.assertIn("synthetic data", meta["disclaimer"].lower())
            self.assertIn("calcs", meta)
            self.assertGreater(meta["calcs"]["lr_pos"], 0.0)
            self.assertGreater(meta["calcs"]["lr_neg"], 0.0)


    def test_visualization_shift_calculations(self):
        for scen in ["pulmonary_embolism_d_dimer", "strep_throat_centor_rapid_antigen", "appendicitis_alvarado_ultrasound"]:
            _, _, meta = generate_likelihood_ratio_demo_data(scen)
            calcs = meta["calcs"]
            pre_p = calcs["pre_test_prob"]
            post_p_pos = calcs["post_test_prob_pos"]
            post_p_neg = calcs["post_test_prob_neg"]
            shift_pos = (post_p_pos - pre_p) * 100.0
            shift_neg = (post_p_neg - pre_p) * 100.0
            self.assertGreater(shift_pos, 0.0)
            self.assertLess(shift_neg, 0.0)


if __name__ == "__main__":
    unittest.main()


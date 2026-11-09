"""Unit Tests for Causal Inference & Quasi-Experimental Engine.

Tests:
1. Difference-in-Differences (DiD) synthetic generation, 2x2 subtraction, and OLS interaction estimation.
2. Interrupted Time Series (ITS) segmented regression, level shift, slope change, and Durbin-Watson statistic.
3. Propensity Score Matching (PSM) logistic score derivation, 1:1 nearest-neighbor matching, and SMD reduction.
4. Regression Discontinuity Design (RDD) local linear regression jump, bandwidth filtering, and McCrary density check.
5. Instrumental Variables (IV) 2SLS estimation, first-stage F-statistic, and Wald estimator consistency.
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.causal_inference.causal_schema import CAUSAL_METHODS, CAUSAL_SCENARIOS
from src.causal_inference.causal_engine import (
    generate_did_dataset,
    calculate_did_model,
    generate_its_dataset,
    calculate_its_model,
    generate_psm_dataset,
    calculate_psm_model,
    generate_rdd_dataset,
    calculate_rdd_model,
    generate_iv_dataset,
    calculate_iv_model
)


class TestCausalInferenceEngine(unittest.TestCase):

    def test_schema_completeness(self):
        """Validates that all 5 causal inference methods are properly defined."""
        expected_keys = [
            "difference_in_differences",
            "interrupted_time_series",
            "propensity_score_matching",
            "regression_discontinuity",
            "instrumental_variables"
        ]
        for key in expected_keys:
            self.assertIn(key, CAUSAL_METHODS)
            info = CAUSAL_METHODS[key]
            self.assertIn("name", info)
            self.assertIn("category", info)
            self.assertIn("primary_estimand", info)
            self.assertIn("identification_assumptions", info)
            self.assertIn("plain_english_glossary", info)
            self.assertIn("mathematical_formulation", info)
            self.assertIn("when_to_use", info)
            self.assertIn("when_not_to_use", info)

    def test_did_estimation(self):
        """Tests Difference-in-Differences 2x2 calculation and OLS interaction."""
        df = generate_did_dataset(
            n_treated=50,
            n_control=50,
            n_periods=6,
            policy_period=3,
            true_att=-4.0,
            baseline_level=30.0,
            seed=42
        )
        self.assertEqual(len(df), 600)  # 100 units * 6 periods
        
        results = calculate_did_model(df)
        self.assertAlmostEqual(results["delta_att"], -4.0, delta=0.8)
        self.assertLess(results["p_value"], 0.001)
        self.assertIn("cell_means", results)
        self.assertIn("parallel_trends_test", results)
        self.assertEqual(results["parallel_trends_test"]["status"], "PASS")

    def test_its_estimation(self):
        """Tests Interrupted Time Series segmented regression parameters."""
        df = generate_its_dataset(
            n_periods=36,
            intervention_step=18,
            baseline_intercept=50.0,
            baseline_slope=0.20,
            level_shift=-10.0,
            slope_change=-0.60,
            seed=42
        )
        self.assertEqual(len(df), 36)
        
        results = calculate_its_model(df)
        self.assertAlmostEqual(results["beta_2_level_shift"], -10.0, delta=2.0)
        self.assertAlmostEqual(results["beta_3_slope_change"], -0.60, delta=0.35)
        self.assertLess(results["p_val_level_shift"], 0.01)
        self.assertIn("durbin_watson", results)

    def test_psm_estimation(self):
        """Tests Propensity Score derivation, 1:1 matching, and SMD reduction."""
        df = generate_psm_dataset(n=400, true_att=-3.5, seed=42)
        self.assertEqual(len(df), 400)
        
        results = calculate_psm_model(df, caliper_sd=0.25)
        self.assertGreater(results["matched_pairs_count"], 40)
        
        # Verify that post-match SMDs are substantially reduced compared to pre-match
        pre_smds = results["pre_smd"]
        post_smds = results["post_smd"]
        for cov in ["age", "charlson_comorbidity"]:
            self.assertLess(abs(post_smds[cov]), abs(pre_smds[cov]))
            self.assertLess(abs(post_smds[cov]), 0.15)
            
        # Matched ATT should be negative and close to true_att
        self.assertAlmostEqual(results["matched_att"], -3.5, delta=1.5)

    def test_rdd_estimation(self):
        """Tests Sharp Regression Discontinuity local linear regression at cutoff."""
        df = generate_rdd_dataset(n=600, cutoff=10.0, true_late=-6.0, seed=42)
        self.assertEqual(len(df), 600)
        
        results = calculate_rdd_model(df, cutoff=10.0, bandwidth=4.0)
        self.assertAlmostEqual(results["tau_late"], -6.0, delta=1.8)
        self.assertLess(results["p_value"], 0.05)
        self.assertIn("PASS", results["mccrary_check"]["status"])

    def test_iv_estimation(self):
        """Tests 2SLS Instrumental Variables vs naive confounded OLS."""
        df = generate_iv_dataset(
            n=800,
            causal_effect=-5.0,
            unmeasured_confounder_effect=7.0,
            instrument_strength=2.0,
            seed=42
        )
        self.assertEqual(len(df), 800)
        
        results = calculate_iv_model(df)
        # Instrument should be strong (F > 10)
        self.assertGreater(results["first_stage_f_statistic"], 10.0)
        self.assertIn("STRONG", results["instrument_strength"])
        
        # 2SLS estimate should recover negative causal effect
        self.assertAlmostEqual(results["beta_2sls_causal"], -5.0, delta=2.0)
        
        # Wald estimate should align closely with 2SLS
        self.assertAlmostEqual(results["wald_estimator"], results["beta_2sls_causal"], delta=0.8)


if __name__ == "__main__":
    unittest.main()

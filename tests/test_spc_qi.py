"""Unit Tests for Statistical Process Control (SPC) & Quality Improvement (QI) Engine.

Tests:
1. Schema completeness for all 5 SPC methods and clinical scenarios.
2. Healthcare Run Chart median calculation, runs count, shift, trend, and astronomical point rules.
3. Shewhart XmR / I-MR chart moving range derivation, 3-sigma limits, and special cause detection.
4. p-Chart and Laney p'-chart overdispersion factor sigma_z and dynamic limit adjustments.
5. u-Chart and Laney u'-chart Poisson count rates per 1,000 bed-day exposure.
6. NHS 'Making Data Count' Variation and Assurance icons and Cp / Cpk process capability metrics.
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spc_qi.spc_schema import SPC_METHODS, SPC_SCENARIOS
from src.spc_qi.spc_engine import (
    generate_run_chart_dataset,
    calculate_run_chart_model,
    generate_xmr_dataset,
    calculate_xmr_model,
    generate_p_chart_dataset,
    calculate_p_chart_model,
    generate_u_chart_dataset,
    calculate_u_chart_model,
    generate_mdc_dataset,
    calculate_mdc_capability_model
)


class TestSPCQIEngine(unittest.TestCase):
    """Verifies the mathematical and statistical integrity of the SPC & QI engine."""

    def test_schema_completeness(self):
        """Validates that all 5 SPC methods are properly defined in the schema."""
        expected_keys = [
            "healthcare_run_chart",
            "xmr_individual_chart",
            "p_chart_laney",
            "u_chart_laney",
            "making_data_count_capability"
        ]
        for key in expected_keys:
            self.assertIn(key, SPC_METHODS)
            info = SPC_METHODS[key]
            self.assertIn("name", info)
            self.assertIn("category", info)
            self.assertIn("primary_estimand", info)
            self.assertIn("assumptions", info)
            self.assertIn("plain_english_glossary", info)
            self.assertIn("mathematical_formulation", info)
            self.assertIn("when_to_use", info)
            self.assertIn("when_not_to_use", info)
            self.assertIn("pros", info)
            self.assertIn("cons", info)

    def test_run_chart_model(self):
        """Tests Healthcare Run Chart median calculation, useful points, and shift/astronomical rules."""
        df = generate_run_chart_dataset(n_points=24, baseline_median=14.0, intervention_point=13, shift_magnitude=-5.0, seed=42)
        self.assertEqual(len(df), 24)
        
        results = calculate_run_chart_model(df, baseline_periods=12)
        self.assertAlmostEqual(results["median"], 14.0, delta=1.5)
        self.assertGreaterEqual(results["n_useful"], 18)
        self.assertGreater(results["runs_count"], 0)
        self.assertIn("rule_1_shift", results)
        self.assertIn("rule_2_trend", results)
        self.assertIn("rule_3_runs", results)
        self.assertIn("rule_4_astronomical", results)
        self.assertTrue(results["has_special_cause_signal"])

    def test_xmr_chart_model(self):
        """Tests Shewhart XmR chart moving range derivation, 3-sigma limits, and special causes."""
        df = generate_xmr_dataset(n_points=30, baseline_mean=4.2, baseline_sd=0.65, intervention_step=20, improvement_effect=-1.1, seed=42)
        self.assertEqual(len(df), 30)
        
        results = calculate_xmr_model(df, baseline_periods=20)
        self.assertAlmostEqual(results["x_bar"], 4.2, delta=0.5)
        self.assertGreater(results["mr_bar"], 0.0)
        self.assertGreater(results["ucl_x"], results["x_bar"])
        self.assertLess(results["lcl_x"], results["x_bar"])
        self.assertAlmostEqual(results["ucl_x"], results["x_bar"] + 2.66 * results["mr_bar"], delta=0.01)
        self.assertTrue(results["special_cause_detected"])

    def test_p_chart_laney_model(self):
        """Tests p-chart and Laney p'-chart overdispersion factor sigma_z and dynamic limits."""
        df = generate_p_chart_dataset(n_periods=24, base_p=0.085, overdispersion_inflation=1.45, seed=42)
        self.assertEqual(len(df), 24)
        
        results = calculate_p_chart_model(df, baseline_periods=18)
        self.assertAlmostEqual(results["p_bar"], 0.085, delta=0.02)
        # Sigma_z should detect overdispersion
        self.assertGreater(results["sigma_z"], 1.1)
        self.assertEqual(len(results["std_ucl"]), 24)
        self.assertEqual(len(results["laney_ucl"]), 24)

    def test_u_chart_laney_model(self):
        """Tests u-chart and Laney u'-chart count rates per 1,000 bed-days."""
        df = generate_u_chart_dataset(n_periods=24, base_rate=4.8, overdispersion_inflation=1.38, seed=42)
        self.assertEqual(len(df), 24)
        
        results = calculate_u_chart_model(df, baseline_periods=18)
        self.assertAlmostEqual(results["u_bar"], 4.8, delta=1.0)
        self.assertGreater(results["sigma_z"], 1.0)
        self.assertEqual(len(results["std_ucl"]), 24)
        self.assertEqual(len(results["laney_ucl"]), 24)

    def test_mdc_capability_model(self):
        """Tests NHS 'Making Data Count' Variation and Assurance icons and Cp / Cpk metrics."""
        df = generate_mdc_dataset(n_periods=26, baseline_compliance=88.5, target_value=95.0, seed=42)
        self.assertEqual(len(df), 26)
        
        results = calculate_mdc_capability_model(df, target_value=95.0, is_higher_better=True, baseline_periods=20)
        self.assertAlmostEqual(results["x_bar"], 89.0, delta=2.5)
        self.assertLess(results["cpk"], 1.0)  # Incapable of reaching 95% standard reliably
        self.assertIn(results["assurance_code"], ["FAIL", "FLAPPING"])
        self.assertIn("Variation", results["variation_icon"])


if __name__ == "__main__":
    unittest.main()

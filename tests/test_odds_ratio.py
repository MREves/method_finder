"""Unit Test Suite for Epidemiological Odds Ratio (OR) & Case-Control Analysis.

Validates:
1. Cross-product Odds Ratio calculation accuracy: (a * d) / (b * c).
2. Woolf's log-scale Standard Error: sqrt(1/a + 1/b + 1/c + 1/d).
3. Wald 95% Confidence Intervals and hypothesis testing (Wald, Chi2, Fisher's exact).
4. Haldane-Anscombe continuity correction (+0.5) for zero/sparse cells.
5. Tri-color bias diagnostics (Rare disease assumption, Berkson's bias, Recall bias, Cell counts).
6. Case-Control demo data and participant microdata generation integrity.
"""

import unittest

try:
    import numpy as np
    import pandas as pd
    from src.epi_methods.odds_ratio import (
        calculate_odds_ratio,
        evaluate_odds_ratio_prerequisites,
        generate_odds_ratio_demo_data
    )
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestOddsRatioCalculations(unittest.TestCase):
    """Tests the mathematical accuracy of the Odds Ratio analytical engine."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_harmful_odds_ratio_mesothelioma_scenario(self):
        # Asbestos & Mesothelioma scenario: a=52, b=36, c=23, d=114
        res = calculate_odds_ratio(a=52, b=36, c=23, d=114, alpha=0.05)

        self.assertEqual(res["a"], 52)
        self.assertEqual(res["b"], 36)
        self.assertEqual(res["c"], 23)
        self.assertEqual(res["d"], 114)
        self.assertEqual(res["n_cases"], 75)
        self.assertEqual(res["n_controls"], 150)
        self.assertEqual(res["n_total"], 225)

        # Cross-product OR: (52 * 114) / (36 * 23) = 5928 / 828 ≈ 7.1594
        expected_or = (52 * 114) / (36 * 23)
        self.assertAlmostEqual(res["or_point"], expected_or, places=4)
        self.assertEqual(res["effect_direction"], "harmful")

        # Woolf's Standard Error: sqrt(1/52 + 1/36 + 1/23 + 1/114) ≈ 0.31505
        expected_se = np.sqrt(1.0/52 + 1.0/36 + 1.0/23 + 1.0/114)
        self.assertAlmostEqual(res["se_ln_or"], expected_se, places=4)

        # Wald 95% CI: exp(ln(OR) ± 1.95996 * SE)
        ln_or = np.log(expected_or)
        expected_ll = np.exp(ln_or - 1.95996 * expected_se)
        expected_ul = np.exp(ln_or + 1.95996 * expected_se)
        self.assertAlmostEqual(res["or_ci_lower"], expected_ll, places=2)
        self.assertAlmostEqual(res["or_ci_upper"], expected_ul, places=2)

        # Significant p-values
        self.assertLess(res["p_value_wald"], 0.001)
        self.assertLess(res["p_value_chi2"], 0.001)
        self.assertLess(res["p_value_fisher"], 0.001)

    def test_protective_odds_ratio_diet_scenario(self):
        # Mediterranean Diet & MI scenario: a=25, b=94, c=75, d=106
        res = calculate_odds_ratio(a=25, b=94, c=75, d=106)

        # Cross-product OR: (25 * 106) / (94 * 75) = 2650 / 7050 ≈ 0.37588
        expected_or = 2650.0 / 7050.0
        self.assertAlmostEqual(res["or_point"], expected_or, places=4)
        self.assertEqual(res["effect_direction"], "protective")

        # Percentage odds reduction: (1 - 0.37588) * 100 ≈ 62.4%
        self.assertAlmostEqual(res["odds_reduction_pct"], (1.0 - expected_or) * 100.0, places=2)
        self.assertLess(res["or_ci_upper"], 1.0)  # Entire 95% CI below 1.0

    def test_zero_cell_continuity_correction(self):
        # One zero cell (a=0)
        res = calculate_odds_ratio(a=0, b=10, c=15, d=20)
        self.assertTrue(res["continuity_applied"])
        self.assertFalse(np.isnan(res["or_point"]))
        self.assertFalse(np.isnan(res["se_ln_or"]))
        self.assertGreater(res["or_point"], 0.0)

    def test_null_association(self):
        # Equal odds in cases and controls: a=20, b=20, c=40, d=40
        res = calculate_odds_ratio(a=20, b=20, c=40, d=40)
        self.assertAlmostEqual(res["or_point"], 1.0, places=4)
        self.assertEqual(res["effect_direction"], "null")
        self.assertAlmostEqual(res["z_score"], 0.0, places=4)
        self.assertAlmostEqual(res["p_value_wald"], 1.0, places=4)


class TestOddsRatioPrerequisites(unittest.TestCase):
    """Validates the bias and prerequisite diagnostics engine."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_case_control_and_population_controls_pass(self):
        evals = evaluate_odds_ratio_prerequisites(
            study_design="case_control",
            a=50, b=30, c=25, d=120,
            control_sampling_type="population",
            disease_is_rare=True,
            is_recall_bias_risk=False
        )
        statuses = {e["name"]: e["status"] for e in evals}
        self.assertEqual(statuses["Study Design Suitability for Odds Ratio"], "PASS")
        self.assertEqual(statuses["Rare Disease Assumption (OR ≈ RR)"], "PASS")
        self.assertEqual(statuses["Control Selection Validity (Berkson's Bias)"], "PASS")
        self.assertEqual(statuses["Exposure Measurement Validity (Recall Bias)"], "PASS")
        self.assertEqual(statuses["2x2 Cell Count Robustness (Sample Size)"], "PASS")

    def test_common_disease_triggers_rare_disease_warning(self):
        evals = evaluate_odds_ratio_prerequisites(
            study_design="case_control",
            a=50, b=30, c=25, d=120,
            control_sampling_type="population",
            disease_is_rare=False
        )
        rare_ev = next(e for e in evals if "Rare Disease" in e["name"])
        self.assertEqual(rare_ev["status"], "WARNING")
        self.assertIn("exaggerates", rare_ev["details"].lower())

    def test_hospital_controls_trigger_berkson_warning(self):
        evals = evaluate_odds_ratio_prerequisites(
            study_design="case_control",
            a=50, b=30, c=25, d=120,
            control_sampling_type="hospital"
        )
        berk_ev = next(e for e in evals if "Berkson" in e["name"])
        self.assertEqual(berk_ev["status"], "WARNING")
        self.assertIn("berkson", berk_ev["details"].lower())

    def test_recall_bias_trigger(self):
        evals = evaluate_odds_ratio_prerequisites(
            study_design="case_control",
            a=50, b=30, c=25, d=120,
            is_recall_bias_risk=True
        )
        recall_ev = next(e for e in evals if "Recall" in e["name"])
        self.assertEqual(recall_ev["status"], "WARNING")
        self.assertIn("ruminate", recall_ev["details"].lower())

    def test_sparse_cells_trigger_warning(self):
        evals = evaluate_odds_ratio_prerequisites(
            study_design="case_control",
            a=3, b=20, c=25, d=30
        )
        sparse_ev = next(e for e in evals if "Cell Count" in e["name"])
        self.assertEqual(sparse_ev["status"], "WARNING")


class TestOddsRatioDemoDataGeneration(unittest.TestCase):
    """Validates the synthetic Case-Control scenario and microdata generator."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_all_scenarios_generate_validly(self):
        scenarios = ["mesothelioma_asbestos", "outbreak_investigation", "cardiovascular_diet", "patient_feedback_survey"]

        for sc in scenarios:
            df_cont, df_micro, meta = generate_odds_ratio_demo_data(scenario=sc, seed=42)

            self.assertIsInstance(df_cont, pd.DataFrame)
            self.assertIsInstance(df_micro, pd.DataFrame)
            self.assertIsInstance(meta, dict)


            # Check metadata fields
            self.assertIn("title", meta)
            self.assertIn("description", meta)
            self.assertIn("clinical_takeaway", meta)
            self.assertIn("disclaimer", meta)
            self.assertIn("made-up", meta["disclaimer"].lower())

            # Check 2x2 contingency summary
            self.assertEqual(len(df_cont), 3)  # Exposed, Unexposed, Marginals
            self.assertIn("Cases_Outcome_Pos", df_cont.columns)
            self.assertIn("Controls_Outcome_Neg", df_cont.columns)

            # Check microdata structure
            self.assertGreater(len(df_micro), 50)
            self.assertIn("Participant_ID", df_micro.columns)
            self.assertIn("Study_Group", df_micro.columns)
            self.assertIn("Exposure_Status", df_micro.columns)
            self.assertIn("Age", df_micro.columns)
            self.assertIn("Sex", df_micro.columns)

            # Check case and control counts match
            total_cases = (df_micro["Study_Group"] == "Case").sum()
            total_controls = (df_micro["Study_Group"] == "Control").sum()
            self.assertEqual(total_cases, meta["a"] + meta["c"])
            self.assertEqual(total_controls, meta["b"] + meta["d"])


if __name__ == "__main__":
    unittest.main()


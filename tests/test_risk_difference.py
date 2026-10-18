"""Unit Test Suite for Epidemiological Risk Difference (RD) & NNT/NNH Analysis.

Validates:
1. Risk Difference point estimates: I_e - I_u.
2. Linear-scale standard error and 95% Wald Confidence Intervals.
3. Number Needed to Treat (NNT = 1 / ARR) and Number Needed to Harm (NNH = 1 / RD).
4. Continuity correction handling on zero cells.
5. Tri-color bias diagnostics (Study design, baseline risk, cell sparsity, attrition bias).
6. Cohort and survey synthetic demo data generation and participant microdata schemas.
"""

import unittest

try:
    import numpy as np
    import pandas as pd
    from src.epi_methods.risk_difference import (
        calculate_risk_difference,
        evaluate_risk_difference_prerequisites,
        generate_risk_difference_demo_data
    )
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestRiskDifferenceCalculations(unittest.TestCase):
    """Tests the mathematical formulas and point estimates for Risk Difference and NNT/NNH."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_beneficial_statin_trial_nnt(self):
        # Statin trial: a=40, b=960, c=90, d=910 (n1=1000, n0=1000)
        res = calculate_risk_difference(a=40, b=960, c=90, d=910, alpha=0.05)

        self.assertEqual(res["n_exposed"], 1000)
        self.assertEqual(res["n_unexposed"], 1000)
        self.assertAlmostEqual(res["risk_exposed"], 0.04, places=4)
        self.assertAlmostEqual(res["risk_unexposed"], 0.09, places=4)

        # Risk Difference: 0.04 - 0.09 = -0.05 (-5.0 percentage points)
        self.assertAlmostEqual(res["risk_difference"], -0.05, places=4)
        self.assertAlmostEqual(res["arr"], 0.05, places=4)
        self.assertEqual(res["effect_direction"], "protective")

        # NNT = 1 / 0.05 = 20.0
        self.assertIsNotNone(res["nnt"])
        self.assertIsNone(res["nnh"])
        self.assertAlmostEqual(res["nnt"], 20.0, places=2)

        # Linear SE: sqrt(0.04*0.96/1000 + 0.09*0.91/1000) = sqrt(0.0000384 + 0.0000819) ≈ 0.010968
        expected_se = np.sqrt((0.04 * 0.96 / 1000) + (0.09 * 0.91 / 1000))
        self.assertAlmostEqual(res["se_rd"], expected_se, places=4)

        # 95% Wald CI: -0.05 ± 1.95996 * SE
        expected_ll = -0.05 - 1.95996 * expected_se
        expected_ul = -0.05 + 1.95996 * expected_se
        self.assertAlmostEqual(res["rd_ci_lower"], expected_ll, places=3)
        self.assertAlmostEqual(res["rd_ci_upper"], expected_ul, places=3)

        # Significant p-value
        self.assertLess(res["p_value_wald"], 0.001)

    def test_harmful_occupational_vibration_nnh(self):
        # Vibration cohort: a=88, b=312, c=24, d=576 (n1=400, n0=600)
        res = calculate_risk_difference(a=88, b=312, c=24, d=576)

        self.assertAlmostEqual(res["risk_exposed"], 0.22, places=4)
        self.assertAlmostEqual(res["risk_unexposed"], 0.04, places=4)

        # Risk Difference: 0.22 - 0.04 = +0.18 (+18.0%)
        self.assertAlmostEqual(res["risk_difference"], 0.18, places=4)
        self.assertEqual(res["effect_direction"], "harmful")

        # NNH = 1 / 0.18 ≈ 5.556
        self.assertIsNone(res["nnt"])
        self.assertIsNotNone(res["nnh"])
        self.assertAlmostEqual(res["nnh"], 1.0 / 0.18, places=2)

    def test_null_difference(self):
        # Parity: a=50, b=450, c=50, d=450
        res = calculate_risk_difference(a=50, b=450, c=50, d=450)
        self.assertAlmostEqual(res["risk_difference"], 0.0, places=4)
        self.assertEqual(res["effect_direction"], "null")
        self.assertIsNone(res["nnt"])
        self.assertIsNone(res["nnh"])
        self.assertAlmostEqual(res["z_score"], 0.0, places=4)
        self.assertAlmostEqual(res["p_value_wald"], 1.0, places=4)

    def test_zero_cell_handling(self):
        # Zero event cell: a=0
        res = calculate_risk_difference(a=0, b=100, c=10, d=90)
        self.assertTrue(res["continuity_applied"])
        self.assertFalse(np.isnan(res["risk_difference"]))
        self.assertFalse(np.isnan(res["se_rd"]))


class TestRiskDifferencePrerequisites(unittest.TestCase):
    """Validates the diagnostic evaluation engine for Risk Difference."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_cohort_and_rct_designs_pass(self):
        evals = evaluate_risk_difference_prerequisites(
            study_design="rct",
            a=40, b=960, c=90, d=910,
            loss_to_follow_up_pct=2.0
        )
        statuses = {e["name"]: e["status"] for e in evals}
        self.assertEqual(statuses["Study Design Suitability for Absolute Risk"], "PASS")
        self.assertEqual(statuses["Baseline Risk Generalizability (I_u Dependency)"], "PASS")
        self.assertEqual(statuses["Cell Count Robustness (Sample Size)"], "PASS")
        self.assertEqual(statuses["Follow-Up Completeness (Attrition Bias)"], "PASS")

    def test_case_control_fails_for_risk_difference(self):
        evals = evaluate_risk_difference_prerequisites(
            study_design="case_control",
            a=50, b=50, c=50, d=50
        )
        design_ev = next(e for e in evals if "Study Design" in e["name"])
        self.assertEqual(design_ev["status"], "FAIL")
        self.assertIn("Odds Ratio", design_ev["recommendation"])

    def test_high_loss_to_follow_up_fails(self):
        evals = evaluate_risk_difference_prerequisites(
            study_design="prospective_cohort",
            a=50, b=450, c=25, d=475,
            loss_to_follow_up_pct=25.0
        )
        loss_ev = next(e for e in evals if "Attrition" in e["name"])
        self.assertEqual(loss_ev["status"], "FAIL")


class TestRiskDifferenceDemoDataGeneration(unittest.TestCase):
    """Validates the synthetic scenario and participant microdata generation."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_all_scenarios_generate_validly(self):
        scenarios = ["statin_cardiovascular", "smoking_cessation", "occupational_vibration", "patient_feedback_survey"]

        for sc in scenarios:
            df_cont, df_micro, meta = generate_risk_difference_demo_data(scenario=sc, seed=42)

            self.assertIsInstance(df_cont, pd.DataFrame)
            self.assertIsInstance(df_micro, pd.DataFrame)
            self.assertIsInstance(meta, dict)

            self.assertIn("title", meta)
            self.assertIn("description", meta)
            self.assertIn("clinical_takeaway", meta)
            self.assertIn("disclaimer", meta)
            self.assertIn("made-up", meta["disclaimer"].lower())

            self.assertEqual(len(df_cont), 3)  # Exposed, Unexposed, Total
            self.assertIn("Stratum_Total_n", df_cont.columns)
            self.assertIn("Cumulative_Incidence_Risk", df_cont.columns)

            self.assertGreater(len(df_micro), 50)
            self.assertIn("Participant_ID", df_micro.columns)
            self.assertIn("Exposure_Group", df_micro.columns)
            self.assertIn("Outcome_Status", df_micro.columns)
            self.assertIn("Age", df_micro.columns)
            self.assertIn("Sex", df_micro.columns)

            total_cases = (df_micro["Outcome_Binary"] == 1).sum()
            self.assertEqual(total_cases, meta["a"] + meta["c"])


if __name__ == "__main__":
    unittest.main()


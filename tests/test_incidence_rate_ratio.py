"""Unit Test Suite for Epidemiological Incidence Rate Ratio (IRR) with Person-Time.

Validates:
1. Incidence Rates per unit of Person-Time: IR_e = a / PT_e and IR_u = c / PT_u.
2. Incidence Rate Ratio (IRR) point estimates: (a / PT_e) / (c / PT_u).
3. Incidence Rate Difference (IRD): (a / PT_e) - (c / PT_u).
4. Rothman-Greenland Log-scale Standard Error: sqrt(1/a + 1/c).
5. Wald 95% Confidence Intervals for IRR: exp(ln(IRR) ± 1.96 * SE).
6. Linear Standard Error & 95% CI for Rate Difference: sqrt(a/PT_e^2 + c/PT_u^2).
7. Exact Conditional Poisson test p-values (via stats.binomtest) and Wald hypothesis test.
8. Continuity correction handling on zero cells (+0.5).
9. Tri-color bias diagnostics (Study design, Poisson constant hazard, event volume, denominator completeness).
10. Synthetic dynamic cohort demo data generation and individual participant microdata schemas.
"""

import unittest

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    from src.epi_methods.incidence_rate_ratio import (
        calculate_incidence_rate_ratio,
        evaluate_irr_prerequisites,
        generate_incidence_rate_ratio_demo_data
    )
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestIncidenceRateRatioCalculations(unittest.TestCase):
    """Tests the mathematical formulas and point estimates for IRR with Person-Time."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_harmful_occupational_lead_cohort(self):
        # Derbyshire lead foundry scenario: a=78, pt_e=6240.0, c=45, pt_u=9000.0
        # IR_e = 78 / 6240 = 0.0125 (12.5 per 1,000 PY)
        # IR_u = 45 / 9000 = 0.0050 (5.0 per 1,000 PY)
        # IRR = 0.0125 / 0.0050 = 2.50
        res = calculate_incidence_rate_ratio(a=78, pt_e=6240.0, c=45, pt_u=9000.0, rate_multiplier=1000.0)

        self.assertEqual(res["a"], 78)
        self.assertEqual(res["c"], 45)
        self.assertAlmostEqual(res["pt_e"], 6240.0)
        self.assertAlmostEqual(res["pt_u"], 9000.0)
        self.assertAlmostEqual(res["pt_total"], 15240.0)

        self.assertAlmostEqual(res["ir_exposed"], 0.0125, places=5)
        self.assertAlmostEqual(res["ir_unexposed"], 0.0050, places=5)
        self.assertAlmostEqual(res["ir_exposed_scaled"], 12.5, places=2)
        self.assertAlmostEqual(res["ir_unexposed_scaled"], 5.0, places=2)

        # IRR point estimate
        self.assertAlmostEqual(res["irr"], 2.50, places=4)
        self.assertAlmostEqual(res["ln_irr"], np.log(2.50), places=4)
        self.assertEqual(res["effect_direction"], "harmful")

        # Rate Difference: 0.0125 - 0.0050 = 0.0075 (7.5 per 1,000 PY)
        self.assertAlmostEqual(res["ird"], 0.0075, places=5)
        self.assertAlmostEqual(res["ird_scaled"], 7.5, places=2)

        # Rothman-Greenland SE: sqrt(1/78 + 1/45) = sqrt(0.01282 + 0.02222) ≈ 0.18719
        expected_se_ln = np.sqrt(1.0 / 78 + 1.0 / 45)
        self.assertAlmostEqual(res["se_ln_irr"], expected_se_ln, places=4)

        # 95% Wald CI for IRR: exp(ln(2.50) ± 1.95996 * SE)
        expected_ll = np.exp(np.log(2.50) - 1.95996 * expected_se_ln)
        expected_ul = np.exp(np.log(2.50) + 1.95996 * expected_se_ln)
        self.assertAlmostEqual(res["irr_ci_lower"], expected_ll, places=3)
        self.assertAlmostEqual(res["irr_ci_upper"], expected_ul, places=3)

        # Linear SE for IRD: sqrt(78 / 6240^2 + 45 / 9000^2)
        expected_var_ird = (78 / (6240.0 ** 2)) + (45 / (9000.0 ** 2))
        expected_se_ird = np.sqrt(expected_var_ird)
        self.assertAlmostEqual(res["se_ird"], expected_se_ird, places=6)

        # Statistical significance: Wald p-value < 0.001 and Exact Poisson p-value < 0.001
        self.assertLess(res["p_value_wald"], 0.001)
        self.assertLess(res["p_value_exact_poisson"], 0.001)

    def test_protective_care_home_flooring(self):
        # Elderly care falls scenario: a=28, pt_e=56000.0, c=72, pt_u=60000.0
        # IR_e = 28 / 56000 = 0.0005 (0.50 per 1,000 resident-days)
        # IR_u = 72 / 60000 = 0.0012 (1.20 per 1,000 resident-days)
        # IRR = 0.0005 / 0.0012 = 0.41667
        res = calculate_incidence_rate_ratio(a=28, pt_e=56000.0, c=72, pt_u=60000.0, rate_multiplier=1000.0)

        self.assertAlmostEqual(res["ir_exposed_scaled"], 0.50, places=2)
        self.assertAlmostEqual(res["ir_unexposed_scaled"], 1.20, places=2)
        self.assertAlmostEqual(res["irr"], 0.41667, places=4)
        self.assertEqual(res["effect_direction"], "protective")

        # Rothman-Greenland SE: sqrt(1/28 + 1/72)
        expected_se = np.sqrt(1.0 / 28 + 1.0 / 72)
        self.assertAlmostEqual(res["se_ln_irr"], expected_se, places=4)
        self.assertLess(res["irr_ci_upper"], 1.0)  # Entire 95% CI below 1.0

    def test_null_rate_ratio(self):
        # Parity: identical rates
        res = calculate_incidence_rate_ratio(a=50, pt_e=5000.0, c=50, pt_u=5000.0)
        self.assertAlmostEqual(res["irr"], 1.0, places=4)
        self.assertAlmostEqual(res["ird"], 0.0, places=4)
        self.assertEqual(res["effect_direction"], "null")
        self.assertAlmostEqual(res["z_score"], 0.0, places=4)
        self.assertAlmostEqual(res["p_value_wald"], 1.0, places=4)
        self.assertAlmostEqual(res["p_value_exact_poisson"], 1.0, places=4)

    def test_zero_event_continuity_correction(self):
        # Zero event numerator: a=0
        res = calculate_incidence_rate_ratio(a=0, pt_e=1000.0, c=10, pt_u=1000.0)
        self.assertTrue(res["continuity_applied"])
        self.assertFalse(np.isnan(res["irr"]))
        self.assertFalse(np.isnan(res["se_ln_irr"]))
        self.assertFalse(np.isnan(res["irr_ci_lower"]))


class TestIncidenceRateRatioPrerequisites(unittest.TestCase):
    """Tests the bias diagnostics and prerequisite checks for IRR."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack not installed")

    def test_prerequisites_dynamic_cohort_pass(self):
        evals = evaluate_irr_prerequisites(
            study_design="dynamic_cohort",
            a=78, pt_e=6240.0, c=45, pt_u=9000.0,
            is_constant_hazard_verified=True
        )
        names = [e["name"] for e in evals]
        statuses = {e["name"]: e["status"] for e in evals}

        self.assertIn("Person-Time Metric Suitability", names)
        self.assertEqual(statuses["Person-Time Metric Suitability"], "PASS")
        self.assertEqual(statuses["Constant Hazard (Poisson Assumption)"], "PASS")
        self.assertEqual(statuses["Event Numerator Volume (Sample Size)"], "PASS")
        self.assertEqual(statuses["Person-Time Denominator Completeness"], "PASS")

    def test_prerequisites_case_control_fails(self):
        evals = evaluate_irr_prerequisites(
            study_design="case_control",
            a=20, pt_e=100.0, c=20, pt_u=100.0
        )
        pt_eval = next(e for e in evals if e["name"] == "Person-Time Metric Suitability")
        self.assertEqual(pt_eval["status"], "FAIL")
        self.assertIn("Odds Ratio", pt_eval["recommendation"])

    def test_prerequisites_sparse_events_warning(self):
        evals = evaluate_irr_prerequisites(
            study_design="prospective_cohort",
            a=3, pt_e=1000.0, c=25, pt_u=2000.0
        )
        event_eval = next(e for e in evals if e["name"] == "Event Numerator Volume (Sample Size)")
        self.assertEqual(event_eval["status"], "WARNING")

    def test_prerequisites_nonconstant_hazard_warning(self):
        evals = evaluate_irr_prerequisites(
            study_design="dynamic_cohort",
            a=50, pt_e=5000.0, c=50, pt_u=5000.0,
            is_constant_hazard_verified=False
        )
        hazard_eval = next(e for e in evals if e["name"] == "Constant Hazard (Poisson Assumption)")
        self.assertEqual(hazard_eval["status"], "WARNING")


class TestIncidenceRateRatioDataGeneration(unittest.TestCase):
    """Tests the synthetic dynamic cohort generation and microdata output."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack not installed")

    def test_lead_foundry_data_generation(self):
        df_cont, df_micro, meta = generate_incidence_rate_ratio_demo_data(
            scenario="derbyshire_lead_foundry", seed=42
        )

        self.assertIsInstance(df_cont, pd.DataFrame)
        self.assertIsInstance(df_micro, pd.DataFrame)
        self.assertIsInstance(meta, dict)

        # Verify disclaimer presence
        self.assertIn("disclaimer", meta)
        self.assertIn("educational illustration only", meta["disclaimer"].lower())

        # Verify contingency table structure (no None values, uniform columns)
        self.assertEqual(len(df_cont), 3)
        self.assertFalse(df_cont.isnull().values.any())

        # Verify microdata columns
        expected_cols = {
            "Participant_ID", "Cohort_Group", "Exposure_Binary",
            "Observation_Person_Time", "Event_Occurred", "Event_Status",
            "Age_At_Entry", "Sex"
        }
        self.assertTrue(expected_cols.issubset(set(df_micro.columns)))
        self.assertGreater(len(df_micro), 200)

    def test_care_falls_and_telehealth_scenarios(self):
        for scen in ["elderly_care_falls", "telehealth_copd_admissions"]:
            df_cont, df_micro, meta = generate_incidence_rate_ratio_demo_data(scenario=scen)
            self.assertEqual(len(df_cont), 3)
            self.assertFalse(df_cont.isnull().values.any())
            self.assertIn("educational illustration only", meta["disclaimer"].lower())


if __name__ == "__main__":
    unittest.main()


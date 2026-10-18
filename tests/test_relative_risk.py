"""Unit Tests for Epidemiological Relative Risk (Risk Ratio) Engine.

Tests:
1. calculate_relative_risk point estimates, log-scale SE, Wald 95% CIs, and NNH/NNT.
2. Continuity correction behavior on zero cells and sparse counts.
3. Protective effect vs. harmful effect metrics and attributable fractions.
4. evaluate_relative_risk_prerequisites across valid and invalid study designs.
5. generate_relative_risk_demo_data cohort microdata and 2x2 contingency table schemas.
"""

import unittest

try:
    import numpy as np
    import pandas as pd
    from src.epi_methods.relative_risk import (
        calculate_relative_risk,
        evaluate_relative_risk_prerequisites,
        generate_relative_risk_demo_data
    )
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestEpidemiologicalRelativeRisk(unittest.TestCase):
    """Test suite for Relative Risk mathematical formulas and assumption pipelines."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_harmful_exposure_calculation(self):
        """Verifies calculation for a harmful occupational exposure cohort (RR > 1)."""
        # Exposed: 81 cases / 450 total -> risk = 0.18 (18%)
        # Unexposed: 39 cases / 550 total -> risk = 0.0709 (7.09%)
        res = calculate_relative_risk(a=81, b=369, c=39, d=511)

        self.assertAlmostEqual(res["risk_exposed"], 81 / 450, places=4)
        self.assertAlmostEqual(res["risk_unexposed"], 39 / 550, places=4)
        expected_rr = (81 / 450) / (39 / 550)
        self.assertAlmostEqual(res["rr"], expected_rr, places=4)
        self.assertGreater(res["rr"], 1.0)
        self.assertEqual(res["effect_direction"], "harmful")

        # Confidence intervals must encompass the point estimate
        self.assertLess(res["rr_ci_lower"], res["rr"])
        self.assertGreater(res["rr_ci_upper"], res["rr"])
        self.assertGreater(res["rr_ci_lower"], 1.0)  # Statistically significant

        # Risk Difference and NNH
        self.assertGreater(res["risk_difference"], 0.0)
        self.assertIsNotNone(res["nnh"])
        self.assertIsNone(res["nnt"])

        # Attributable fraction among exposed (AF_e)
        expected_afe = ((expected_rr - 1.0) / expected_rr) * 100.0
        self.assertAlmostEqual(res["attributable_fraction_exposed"], expected_afe, places=2)

        # Hypothesis test p-values
        self.assertLess(res["p_value_wald"], 0.001)
        self.assertLess(res["p_value_chi2"], 0.001)
        self.assertLess(res["p_value_fisher"], 0.001)

    def test_protective_intervention_calculation(self):
        """Verifies calculation for a protective clinical trial / intervention (RR < 1)."""
        # Exposed: 24 cases / 750 total -> risk = 0.032
        # Unexposed: 72 cases / 750 total -> risk = 0.096
        res = calculate_relative_risk(a=24, b=726, c=72, d=678)

        expected_rr = 0.032 / 0.096  # 0.3333...
        self.assertAlmostEqual(res["rr"], expected_rr, places=4)
        self.assertLess(res["rr"], 1.0)
        self.assertEqual(res["effect_direction"], "protective")

        # Vaccine efficacy / Relative Risk Reduction
        expected_rrr = (1.0 - expected_rr) * 100.0
        self.assertAlmostEqual(res["attributable_fraction_exposed"], expected_rrr, places=2)

        # NNT should be defined, NNH should be None
        self.assertIsNotNone(res["nnt"])
        self.assertIsNone(res["nnh"])
        self.assertAlmostEqual(res["nnt"], 1.0 / (0.096 - 0.032), places=1)

    def test_zero_cell_continuity_correction(self):
        """Verifies automatic Haldane-Anscombe continuity correction when zero events occur."""
        res = calculate_relative_risk(a=0, b=50, c=10, d=40)
        self.assertTrue(res["continuity_applied"])
        self.assertFalse(np.isnan(res["rr"]))
        self.assertFalse(np.isnan(res["rr_ci_lower"]))
        self.assertFalse(np.isnan(res["rr_ci_upper"]))

    def test_prerequisite_evaluations(self):
        """Verifies diagnostic engine handles valid cohorts vs invalid case-control designs."""
        # Valid cohort
        evals_cohort = evaluate_relative_risk_prerequisites(
            study_design="prospective_cohort",
            a=50, b=450, c=25, d=475,
            loss_to_follow_up_pct=3.0,
            is_temporal_sequence_verified=True
        )
        statuses_cohort = [e["status"] for e in evals_cohort]
        self.assertIn("PASS", statuses_cohort)
        self.assertNotIn("FAIL", statuses_cohort)

        # Invalid case-control design
        evals_cc = evaluate_relative_risk_prerequisites(
            study_design="case_control",
            a=50, b=50, c=25, d=75
        )
        statuses_cc = [e["status"] for e in evals_cc]
        self.assertIn("FAIL", statuses_cc)
        fail_card = next(e for e in evals_cc if e["status"] == "FAIL")
        self.assertIn("Odds Ratio", fail_card["recommendation"])

    def test_synthetic_data_generation(self):
        """Verifies generation of demo scenarios, 2x2 contingency tables, and microdata."""
        df_cont, df_micro, meta = generate_relative_risk_demo_data(scenario="occupational_dust")
        self.assertEqual(len(df_cont), 2)
        self.assertIn("Cumulative_Incidence_Risk", df_cont.columns)
        self.assertEqual(len(df_micro), 1000)
        self.assertIn("Participant_ID", df_micro.columns)
        self.assertIn("Exposure_Status", df_micro.columns)
        self.assertIn("Outcome_Status", df_micro.columns)
        self.assertIn("calcs", meta)
        self.assertGreater(meta["calcs"]["rr"], 1.0)

        # Test Patient Feedback Survey scenario (Prevalence Ratio)
        df_cont_surv, df_micro_surv, meta_surv = generate_relative_risk_demo_data(scenario="patient_feedback_survey")
        self.assertEqual(len(df_micro_surv), 600)
        self.assertAlmostEqual(meta_surv["calcs"]["rr"], 1.3333, places=3)
        self.assertIn("Prevalence Ratio", meta_surv["title"] + meta_surv["description"])



if __name__ == "__main__":
    unittest.main()


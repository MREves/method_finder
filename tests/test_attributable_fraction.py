"""Unit Test Suite for Epidemiological Attributable Fractions (AF_e) and Population Attributable Fraction (PAF).

Validates:
1. Attributable Fraction in Exposed (AF_e = (RR - 1) / RR = 1 - 1/RR).
2. Population Attributable Fraction (PAF) via Direct Cohort, Levin's, and Miettinen's formulas.
3. Population Attributable Risk (PAR = I_total - I_u).
4. Walter / Greenland 95% Confidence Intervals for PAF.
5. Prevented Fractions for Protective Exposures (Vaccine Efficacy / PF_e = 1 - RR, PFP).
6. Geoffrey Rose's Prevention Paradox (high prevalence, modest RR yielding substantial PAF).
7. Continuity correction on zero cells.
8. Tri-color bias diagnostics (Causality, confounding, exposure prevalence, cell frequencies).
9. Synthetic public health demo data generation and participant microdata schemas.
"""

import unittest

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    from src.epi_methods.attributable_fraction import (
        calculate_attributable_fraction,
        evaluate_attributable_fraction_prerequisites,
        generate_attributable_fraction_demo_data
    )
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestAttributableFractionCalculations(unittest.TestCase):
    """Tests the mathematical formulas and point estimates for AF_e, PAF, and PAR."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas/scipy) not installed in this environment")

    def test_smoking_lung_cancer_harmful_scenario(self):
        # a=130, b=1270 (n1=1400), c=53, d=8547 (n0=8600)
        # I_e = 130 / 1400 = 0.092857 (9.29%)
        # I_u = 53 / 8600 = 0.0061628 (0.62%)
        # RR = 0.092857 / 0.0061628 ≈ 15.067
        res = calculate_attributable_fraction(a=130, b=1270, c=53, d=8547, pe_override=0.14)

        self.assertEqual(res["n1"], 1400)
        self.assertEqual(res["n0"], 8600)
        self.assertEqual(res["n_total"], 10000)
        self.assertAlmostEqual(res["risk_exposed"], 130 / 1400, places=5)
        self.assertAlmostEqual(res["risk_unexposed"], 53 / 8600, places=5)

        # Relative risk
        expected_rr = (130 / 1400) / (53 / 8600)
        self.assertAlmostEqual(res["rr"], expected_rr, places=3)
        self.assertEqual(res["effect_direction"], "harmful")

        # AF_e = (RR - 1) / RR = 1 - 1/RR
        expected_afe = (expected_rr - 1.0) / expected_rr
        self.assertAlmostEqual(res["afe"], expected_afe, places=4)
        self.assertAlmostEqual(res["afe"], 0.9336, places=3)  # ~93.4% of cancer in smokers is from smoking

        # AF_e CI bounds must be monotonic with RR CI bounds
        self.assertGreater(res["afe_ci_lower"], 0.85)
        self.assertLess(res["afe_ci_upper"], 1.0)

        # PAF: Levin's formula with Pe = 0.14
        expected_paf_levin = (0.14 * (expected_rr - 1.0)) / (1.0 + 0.14 * (expected_rr - 1.0))
        self.assertAlmostEqual(res["paf_levin"], expected_paf_levin, places=4)
        self.assertAlmostEqual(res["paf"], expected_paf_levin, places=3)  # ~66.3%

        # Miettinen's formula: P_{cases|exposed} * (RR - 1) / RR
        p_cases_exp = 130 / (130 + 53)
        expected_paf_miettinen = p_cases_exp * expected_afe
        self.assertAlmostEqual(res["paf_miettinen"], expected_paf_miettinen, places=4)

        # Walter / Greenland CI for PAF
        self.assertFalse(np.isnan(res["paf_ci_lower"]))
        self.assertFalse(np.isnan(res["paf_ci_upper"]))
        self.assertLess(res["paf_ci_lower"], res["paf"])
        self.assertGreater(res["paf_ci_upper"], res["paf"])

    def test_rose_prevention_paradox_physical_inactivity(self):
        # Physical inactivity: modest RR = 1.40, high prevalence Pe = 0.40
        # a=224, b=2976 (n1=3200, 40%), c=240, d=4560 (n0=4800, 60%)
        # I_e = 224 / 3200 = 0.070 (7.0%)
        # I_u = 240 / 4800 = 0.050 (5.0%)
        # RR = 0.070 / 0.050 = 1.40
        res = calculate_attributable_fraction(a=224, b=2976, c=240, d=4560, pe_override=0.40)

        self.assertAlmostEqual(res["rr"], 1.40, places=4)
        # AF_e = (1.4 - 1) / 1.4 = 0.4 / 1.4 = 28.57%
        self.assertAlmostEqual(res["afe"], 0.40 / 1.40, places=4)

        # PAF = [0.40 * 0.40] / [1 + 0.40 * 0.40] = 0.16 / 1.16 ≈ 0.13793 (13.8%)
        expected_paf = 0.16 / 1.16
        self.assertAlmostEqual(res["paf"], expected_paf, places=4)

        # Absolute PAR = I_total - I_u
        # I_total = (224 + 240) / 8000 = 464 / 8000 = 0.058 (5.8%)
        # PAR = 0.058 - 0.050 = 0.008 (0.8 percentage points excess in population)
        self.assertAlmostEqual(res["par"], 0.008, places=4)

    def test_protective_hpv_vaccine_scenario(self):
        # Protective: a=12, b=4238 (n1=4250), c=18, d=732 (n0=750)
        # I_e = 12 / 4250 ≈ 0.0028235
        # I_u = 18 / 750 = 0.0240
        # RR = 0.0028235 / 0.0240 ≈ 0.11765 (protective)
        res = calculate_attributable_fraction(a=12, b=4238, c=18, d=732, pe_override=0.85)

        self.assertEqual(res["effect_direction"], "protective")
        self.assertLess(res["rr"], 1.0)

        # Prevented Fraction in Exposed: PF_e = 1 - RR = Vaccine Efficacy
        expected_pfe = 1.0 - res["rr"]
        self.assertAlmostEqual(res["prevented_fraction_exposed"], expected_pfe, places=4)
        self.assertGreater(res["prevented_fraction_exposed"], 0.85)  # > 85% vaccine efficacy

        # Prevented Fraction in Population: PFP = Pe * (1 - RR) = 0.85 * (1 - 0.11765) ≈ 75.0%
        expected_pfp = 0.85 * expected_pfe
        self.assertAlmostEqual(res["prevented_fraction_population"], expected_pfp, places=4)

    def test_zero_cell_continuity_correction(self):
        # Zero cell: a=0
        res = calculate_attributable_fraction(a=0, b=100, c=10, d=90)
        self.assertTrue(res["continuity_applied"])
        self.assertFalse(np.isnan(res["rr"]))
        self.assertFalse(np.isnan(res["afe"]))


class TestAttributableFractionPrerequisites(unittest.TestCase):
    """Tests the bias diagnostics and prerequisite checks for AF_e and PAF."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack not installed")

    def test_prerequisites_all_pass(self):
        evals = evaluate_attributable_fraction_prerequisites(
            study_design="prospective_cohort",
            a=130, b=1270, c=53, d=8547,
            is_causality_established=True,
            is_confounding_controlled=True,
            is_exposure_prevalence_representative=True
        )
        statuses = {e["name"]: e["status"] for e in evals}
        self.assertEqual(statuses["Causal Relationship Assumption"], "PASS")
        self.assertEqual(statuses["Confounding & Multivariable Adjustment"], "PASS")
        self.assertEqual(statuses["Exposure Prevalence Representativeness"], "PASS")
        self.assertEqual(statuses["Cell Frequency & Variance Stability"], "PASS")

    def test_prerequisites_noncausal_fails(self):
        evals = evaluate_attributable_fraction_prerequisites(
            study_design="prospective_cohort",
            a=50, b=500, c=20, d=500,
            is_causality_established=False
        )
        causal_eval = next(e for e in evals if e["name"] == "Causal Relationship Assumption")
        self.assertEqual(causal_eval["status"], "FAIL")
        self.assertIn("NOT interpret", causal_eval["recommendation"])

    def test_prerequisites_unrepresentative_prevalence_warning(self):
        evals = evaluate_attributable_fraction_prerequisites(
            study_design="case_control",
            a=100, b=100, c=100, d=100,
            is_exposure_prevalence_representative=False
        )
        prev_eval = next(e for e in evals if e["name"] == "Exposure Prevalence Representativeness")
        self.assertEqual(prev_eval["status"], "WARNING")
        self.assertIn("Miettinen", prev_eval["recommendation"])


class TestAttributableFractionDataGeneration(unittest.TestCase):
    """Tests synthetic public health demo data generation."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack not installed")

    def test_smoking_and_inactivity_data_generation(self):
        for scen in ["smoking_lung_cancer", "physical_inactivity_cvd", "air_pollution_asthma", "hpv_vaccine_cervical"]:
            df_cont, df_micro, meta = generate_attributable_fraction_demo_data(scenario=scen, seed=42)
            self.assertIsInstance(df_cont, pd.DataFrame)
            self.assertIsInstance(df_micro, pd.DataFrame)
            self.assertIsInstance(meta, dict)

            # Disclaimer check
            self.assertIn("disclaimer", meta)
            self.assertIn("educational illustration only", meta["disclaimer"].lower())

            # Contingency table check (no None values, uniform 3 rows)
            self.assertEqual(len(df_cont), 3)
            self.assertFalse(df_cont.isnull().values.any())

            # Microdata columns check
            expected_cols = {"Participant_ID", "Exposure_Group", "Exposure_Binary", "Outcome_Status", "Outcome_Binary", "Age", "Sex"}
            self.assertTrue(expected_cols.issubset(set(df_micro.columns)))


if __name__ == "__main__":
    unittest.main()


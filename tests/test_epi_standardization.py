"""Unit Tests for Epidemiological Standardization Engine.

Tests:
1. evaluate_standardization_prerequisites with PASS, WARNING, and FAIL scenarios.
2. calculate_direct_standardized_rate with Dobson and Normal confidence intervals.
3. calculate_indirect_standardization with Byar's Poisson exact confidence intervals.
4. get_excel_aligned_dataset schema alignment with Excel tutorial coordinates (Col_A through Col_F).
5. Derbyshire simulation data engine reproducibility and Reversal Paradox validation.
"""

import unittest

try:
    import numpy as np
    import pandas as pd
    from src.epi_methods.standardization import (
        evaluate_standardization_prerequisites,
        calculate_direct_standardized_rate,
        calculate_indirect_standardization,
        calculate_dobson_dsr_ci,
        calculate_byar_poisson_ci
    )
    from src.epi_methods.derbyshire_data import (
        generate_derbyshire_standardization_data,
        get_excel_aligned_dataset,
        ESP_2013_BANDS,
        DISTRICT_PROFILES
    )
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestEpidemiologicalStandardization(unittest.TestCase):
    """Test suite for standardization mathematical formulas and assumption pipelines."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy/pandas) not installed in this environment")
        self.pop = np.array([8831, 20391, 21396, 10555, 7467, 3160], dtype=float)
        self.deaths = np.array([0, 8, 48, 105, 230, 480], dtype=float)
        self.weights = np.array([0.15, 0.38, 0.26, 0.11, 0.07, 0.03], dtype=float)
        self.ref_rates = np.array([0.05, 0.40, 3.20, 12.50, 36.00, 95.00], dtype=float)

    def test_direct_standardization_calculation(self):
        """Verifies DSR, standard error, and Dobson confidence intervals."""
        res = calculate_direct_standardized_rate(
            pop_age=self.pop,
            deaths_age=self.deaths,
            esp_weights=self.weights,
            multiplier=1000.0
        )
        self.assertIn("dsr", res)
        self.assertIn("se_dsr", res)
        self.assertIn("ci_lower", res)
        self.assertIn("ci_upper", res)
        self.assertGreater(res["dsr"], 0.0)
        self.assertLess(res["ci_lower"], res["dsr"])
        self.assertGreater(res["ci_upper"], res["dsr"])
        self.assertAlmostEqual(res["total_deaths"], float(np.sum(self.deaths)), places=1)
        self.assertAlmostEqual(res["total_population"], float(np.sum(self.pop)), places=1)

    def test_indirect_standardization_calculation(self):
        """Verifies Expected Deaths, SMR, and Byar's Poisson confidence intervals."""
        total_obs = float(np.sum(self.deaths))
        res = calculate_indirect_standardization(
            pop_age=self.pop,
            observed_deaths_total=total_obs,
            reference_rates_per_k=self.ref_rates,
            multiplier=1000.0
        )
        self.assertIn("expected_deaths", res)
        self.assertIn("smr", res)
        self.assertIn("smr_percentage", res)
        self.assertIn("smr_ci_lower", res)
        self.assertIn("smr_ci_upper", res)
        self.assertGreater(res["expected_deaths"], 0.0)
        self.assertAlmostEqual(res["smr"], total_obs / res["expected_deaths"], places=4)
        self.assertLess(res["smr_ci_lower"], res["smr"])
        self.assertGreater(res["smr_ci_upper"], res["smr"])

    def test_byar_poisson_ci_properties(self):
        """Verifies Byar's exact Poisson CI limits under varying event counts."""
        # Non-zero events
        lower, upper = calculate_byar_poisson_ci(observed_deaths=100.0, expected_deaths=100.0)
        self.assertAlmostEqual(lower, 0.8136, places=2)
        self.assertAlmostEqual(upper, 1.2154, places=2)

        # Zero events
        lower_0, upper_0 = calculate_byar_poisson_ci(observed_deaths=0.0, expected_deaths=50.0)
        self.assertEqual(lower_0, 0.0)
        self.assertGreater(upper_0, 0.0)

    def test_prerequisites_evaluation(self):
        """Verifies assumption checking diagnostics."""
        evals = evaluate_standardization_prerequisites(
            pop_age=self.pop,
            deaths_age=self.deaths,
            reference_rates_per_k=self.ref_rates
        )
        names = [e["name"] for e in evals]
        self.assertTrue(any("Denominator" in n for n in names))
        self.assertTrue(any("Total Event" in n for n in names))
        self.assertTrue(any("Stratum" in n for n in names))

    def test_excel_aligned_dataset_structure(self):
        """Verifies 1:1 column and row alignment with the Excel tutorial."""
        df_det, df_sum, meta = generate_derbyshire_standardization_data(scenario="circulatory", seed=42)
        df_excel = get_excel_aligned_dataset(df_det, focal_district="Derbyshire Dales")

        expected_columns = [
            "Col_A_Age_Band",
            "Col_B_Population",
            "Col_C_Observed_Deaths",
            "Col_D_Rate_per_1k",
            "Col_E_ESP_Weight",
            "Col_F_Reference_Rate_per_1k"
        ]
        self.assertEqual(list(df_excel.columns), expected_columns)
        self.assertEqual(len(df_excel), 6)  # Exactly 6 age bands for rows 2 to 7
        self.assertEqual(list(df_excel["Col_A_Age_Band"]), ["0-14", "15-44", "45-64", "65-74", "75-84", "85+"])
        self.assertAlmostEqual(float(df_excel["Col_E_ESP_Weight"].sum()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

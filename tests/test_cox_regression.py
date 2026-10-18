"""Unit tests for Cox Proportional Hazards Regression mathematical engine."""

import unittest

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False

if HAS_SCIENTIFIC_STACK:
    from src.epi_methods.cox_regression import (
        fit_cox_proportional_hazards,
        calculate_harrell_c_index,
        calculate_schoenfeld_diagnostics,
        predict_cox_survival_curve,
        evaluate_cox_regression_prerequisites,
        generate_cox_regression_demo_data
    )


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "NumPy/SciPy/Pandas required for Cox regression tests")
class TestCoxProportionalHazards(unittest.TestCase):
    """Test suite for Cox Proportional Hazards regression engine."""

    def setUp(self):
        """Set up small deterministic dataset for verification."""
        # 10 subjects, univariable treatment
        self.df_simple = pd.DataFrame({
            "Time": [3.0, 5.0, 6.0, 8.0, 12.0, 14.0, 15.0, 18.0, 22.0, 24.0],
            "Event": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
            "Treatment": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "Age": [55, 60, 48, 62, 58, 70, 65, 72, 68, 75]
        })

    def test_univariable_cox_fit(self):
        """Tests univariable Cox model convergence, hazard ratio, and Wald statistics."""
        model = fit_cox_proportional_hazards(
            df=self.df_simple,
            time_col="Time",
            event_col="Event",
            covariate_cols=["Treatment"],
            alpha=0.05
        )

        self.assertTrue(model["converged"])
        self.assertEqual(len(model["coefficients"]), 1)
        self.assertEqual(len(model["hazard_ratios"]), 1)

        hr = model["hazard_ratios"][0]
        self.assertGreater(hr, 0.0)
        self.assertLess(model["hr_ci_lower"][0], model["hr_ci_upper"][0])
        self.assertGreaterEqual(model["p_values"][0], 0.0)
        self.assertLessEqual(model["p_values"][0], 1.0)
        self.assertEqual(model["degrees_of_freedom"], 1)

    def test_multivariable_cox_fit(self):
        """Tests multivariable Cox model with two covariates (Treatment and Age)."""
        model = fit_cox_proportional_hazards(
            df=self.df_simple,
            time_col="Time",
            event_col="Event",
            covariate_cols=["Treatment", "Age"],
            alpha=0.05
        )

        self.assertTrue(model["converged"])
        self.assertEqual(len(model["coefficients"]), 2)
        self.assertEqual(model["covariance_matrix"].shape, (2, 2))
        self.assertGreater(model["lrt_statistic"], 0.0)
        self.assertEqual(model["degrees_of_freedom"], 2)

        # Baseline hazard should be monotonically non-decreasing
        cum_h0 = model["cum_baseline_hazard"]
        self.assertTrue(np.all(np.diff(cum_h0) >= 0.0))

    def test_harrell_c_index(self):
        """Tests Harrell's Concordance Index with perfect and random risk scores."""
        times = np.array([5.0, 10.0, 15.0, 20.0])
        events = np.array([1, 1, 1, 1])
        # Higher risk score -> fails earlier (shorter time)
        perfect_risk = np.array([4.0, 3.0, 2.0, 1.0])
        c_perf = calculate_harrell_c_index(times, events, perfect_risk)
        self.assertAlmostEqual(c_perf, 1.0, places=5)

        # Inverted risk score -> fails later
        inverted_risk = np.array([1.0, 2.0, 3.0, 4.0])
        c_inv = calculate_harrell_c_index(times, events, inverted_risk)
        self.assertAlmostEqual(c_inv, 0.0, places=5)

    def test_predict_cox_survival_curve(self):
        """Tests individual predicted survival curve starting at S(0) = 1.0."""
        model = fit_cox_proportional_hazards(
            df=self.df_simple,
            time_col="Time",
            event_col="Event",
            covariate_cols=["Treatment", "Age"]
        )

        pred = predict_cox_survival_curve(model, covariate_values={"Treatment": 1.0, "Age": 60.0})
        self.assertEqual(pred["times"][0], 0.0)
        self.assertEqual(pred["survival_probabilities"][0], 1.0)
        self.assertTrue(np.all(pred["survival_probabilities"] >= 0.0))
        self.assertTrue(np.all(pred["survival_probabilities"] <= 1.0))
        # Monotonically non-increasing
        self.assertTrue(np.all(np.diff(pred["survival_probabilities"]) <= 0.0))

    def test_schoenfeld_diagnostics(self):
        """Tests Schoenfeld residual calculation and format."""
        df_demo, meta = generate_cox_regression_demo_data("oncology_multivariable")
        model = fit_cox_proportional_hazards(
            df=df_demo,
            time_col=meta["time_col"],
            event_col=meta["event_col"],
            covariate_cols=meta["covariate_cols"]
        )

        ph = model["ph_diagnostics"]
        self.assertTrue(ph["tested"])
        self.assertEqual(len(ph["covariate_tests"]), len(meta["covariate_cols"]))
        for test in ph["covariate_tests"]:
            self.assertIn("Correlation_rho", test)
            self.assertIn("p_value", test)
            self.assertIn("Assumption_Holds", test)

    def test_prerequisites_evaluation(self):
        """Tests STROBE/CONSORT prerequisite diagnostic evaluation."""
        df_demo, meta = generate_cox_regression_demo_data("cardiology_heart_failure")
        model = fit_cox_proportional_hazards(
            df=df_demo,
            time_col=meta["time_col"],
            event_col=meta["event_col"],
            covariate_cols=meta["covariate_cols"]
        )

        evals = evaluate_cox_regression_prerequisites(model, df_demo, meta["covariate_cols"])
        self.assertIn("epv", evals)
        self.assertIn("proportionality", evals)
        self.assertIn("censoring", evals)
        self.assertIn("collinearity", evals)
        self.assertIn("discrimination", evals)

        for key, res in evals.items():
            self.assertIn(res["status"], ["pass", "warning", "fail"])
            self.assertTrue(len(res["badge"]) > 0)
            self.assertTrue(len(res["details"]) > 0)

    def test_demo_data_generation_scenarios(self):
        """Tests that all clinical & occupational scenarios generate valid DataFrames and metadata."""
        scenarios = ["oncology_multivariable", "cardiology_heart_failure", "nephrology_ckd", "occupational_return_to_work"]
        for sc in scenarios:
            df, meta = generate_cox_regression_demo_data(sc)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 100)
            self.assertIn(meta["time_col"], df.columns)
            self.assertIn(meta["event_col"], df.columns)
            for c in meta["covariate_cols"]:
                self.assertIn(c, df.columns)
            self.assertIn("organisation", meta)
            self.assertTrue(len(meta["title"]) > 0)


if __name__ == "__main__":
    unittest.main()


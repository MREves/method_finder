"""Unit and Contract Verification Test Suite for Time Series & Forecasting Module.

Validates:
1. Synthetic Data Generators (ED Attendances, Viral Positivity, Ambulance Surges, Intermittent Stockouts).
2. Time Series Decomposition (Conservation under Additive and Multiplicative modes).
3. Autocorrelation & Stationarity Diagnostics (ACF, PACF Durbin-Levinson, ADF OLS Test).
4. Forecasting Model Hierarchy (Naive, Seasonal Naive, Drift, SES, Holt's Damped, Holt-Winters, Lagged ML, Croston).
5. Comprehensive Evaluation Metrics (MAE, MSE, RMSE, MAPE zero-division safety, sMAPE, MASE, MDA, Outlier Dispersion).
6. Expanding-Window Rolling-Origin Cross-Validation Protocol (Zero Temporal Leakage).
7. AST Audit for 100% st.expander(expanded=False) compliance.
"""

import os
import sys
import ast
import math
import unittest
from unittest.mock import MagicMock

# Mock streamlit before importing UI widgets if not in environment
if "streamlit" not in sys.modules:
    mock_st = MagicMock()
    mock_st.session_state = {}
    sys.modules["streamlit"] = mock_st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.time_series.ts_engine import (
    generate_ed_demand_series,
    generate_seasonal_flu_series,
    generate_ambulance_callout_series,
    generate_intermittent_stockout_series,
    decompose_time_series,
    calculate_acf,
    calculate_pacf,
    calculate_adf_test,
    difference_series,
    forecast_naive,
    forecast_seasonal_naive,
    forecast_drift,
    forecast_simple_exponential_smoothing,
    forecast_holt_linear,
    forecast_holt_winters,
    forecast_lagged_regression,
    forecast_croston,
    calculate_forecast_metrics,
    rolling_origin_evaluation,
    TIME_SERIES_ALGORITHMS,
    get_algorithm_profile,
    generate_time_series_python_code,
    generate_time_series_excel_guide,
    generate_time_series_csv,
)


class TestTimeSeriesGenerators(unittest.TestCase):
    """Verifies synthetic time series generators for realistic healthcare scenarios."""

    def test_ed_demand_generator(self):
        data = generate_ed_demand_series(days=60, base_level=100.0, trend_slope=0.2, season_amplitude=20.0)
        self.assertEqual(len(data["series"]), 60)
        self.assertEqual(data["seasonal_period"], 7)
        self.assertTrue(all(v > 0 for v in data["series"]))

    def test_seasonal_flu_generator(self):
        data = generate_seasonal_flu_series(weeks=52)
        self.assertEqual(len(data["series"]), 52)
        self.assertEqual(data["seasonal_period"], 52)
        self.assertTrue(all(v >= 0.5 for v in data["series"]))

    def test_ambulance_callout_generator(self):
        data = generate_ambulance_callout_series(days=50, shock_magnitude=50.0)
        self.assertEqual(len(data["series"]), 50)
        self.assertEqual(data["seasonal_period"], 7)

    def test_intermittent_stockout_generator(self):
        data = generate_intermittent_stockout_series(days=60, zero_prob=0.8)
        self.assertEqual(len(data["series"]), 60)
        # Verify that zeroes are actually present
        zero_count = sum(1 for v in data["series"] if v == 0.0)
        self.assertGreater(zero_count, 10, "Intermittent series must contain substantial zeroes.")


class TestTimeSeriesDecomposition(unittest.TestCase):
    """Verifies classical additive and multiplicative decomposition conservation."""

    def test_additive_decomposition_conservation(self):
        data = generate_ed_demand_series(days=70)
        s = data["series"]
        decomp = decompose_time_series(s, period=7, mode="additive")

        self.assertEqual(len(decomp["trend"]), len(s))
        self.assertEqual(len(decomp["seasonal"]), len(s))
        self.assertEqual(len(decomp["residuals"]), len(s))

        # Check conservation: Y_t ~ Trend_t + Seasonal_t + Residual_t
        for i in range(len(s)):
            recon = decomp["trend"][i] + decomp["seasonal"][i] + decomp["residuals"][i]
            self.assertAlmostEqual(s[i], recon, delta=0.15)

        # Check that seasonal cycle sums approximately to 0
        cycle_sum = sum(decomp["seasonal_cycle"])
        self.assertAlmostEqual(cycle_sum, 0.0, delta=0.1)

    def test_multiplicative_decomposition(self):
        data = generate_ed_demand_series(days=70)
        s = data["series"]
        decomp = decompose_time_series(s, period=7, mode="multiplicative")

        # Seasonal cycle mean should be close to 1.0
        cycle_mean = sum(decomp["seasonal_cycle"]) / len(decomp["seasonal_cycle"])
        self.assertAlmostEqual(cycle_mean, 1.0, delta=0.05)


class TestAutocorrelationAndStationarity(unittest.TestCase):
    """Verifies sample ACF, PACF Durbin-Levinson, and ADF hypothesis testing."""

    def test_acf_calculation(self):
        data = generate_ed_demand_series(days=70)
        s = data["series"]
        acf_res = calculate_acf(s, max_lags=14)

        self.assertEqual(acf_res["lags"][0], 0)
        self.assertEqual(acf_res["acf"][0], 1.0)
        # All autocorrelations must be bounded in [-1, +1]
        for val in acf_res["acf"]:
            self.assertGreaterEqual(val, -1.0)
            self.assertLessEqual(val, 1.0)

    def test_pacf_calculation(self):
        data = generate_ed_demand_series(days=70)
        s = data["series"]
        pacf_res = calculate_pacf(s, max_lags=10)

        self.assertEqual(pacf_res["lags"][0], 0)
        self.assertEqual(pacf_res["pacf"][0], 1.0)
        for val in pacf_res["pacf"]:
            self.assertGreaterEqual(val, -1.0)
            self.assertLessEqual(val, 1.0)

    def test_adf_test_execution(self):
        # Trending series: 10, 20, 30, ... non-stationary
        trend_series = [10.0 + 2.0 * i for i in range(40)]
        adf_res = calculate_adf_test(trend_series)
        self.assertIn("test_statistic", adf_res)
        self.assertIn("p_value", adf_res)
        self.assertIn("is_stationary", adf_res)

        # Differenced series should be flat/stationary
        diff_s = difference_series(trend_series, 1)
        self.assertEqual(len(diff_s), len(trend_series) - 1)


class TestForecastingModelHierarchy(unittest.TestCase):
    """Verifies forecasting models, output horizons, and prediction intervals."""

    def setUp(self):
        self.series = generate_ed_demand_series(days=70)["series"]
        self.horizon = 14

    def test_naive_forecast(self):
        res = forecast_naive(self.series, horizon=self.horizon)
        self.assertEqual(len(res["forecast"]), self.horizon)
        self.assertEqual(res["forecast"][0], self.series[-1])
        self.assertTrue(all(l <= f <= u for l, f, u in zip(res["lower_95"], res["forecast"], res["upper_95"])))

    def test_seasonal_naive_forecast(self):
        res = forecast_seasonal_naive(self.series, period=7, horizon=self.horizon)
        self.assertEqual(len(res["forecast"]), self.horizon)
        self.assertEqual(res["forecast"][0], self.series[-7])

    def test_drift_forecast(self):
        res = forecast_drift(self.series, horizon=self.horizon)
        self.assertEqual(len(res["forecast"]), self.horizon)

    def test_ses_forecast(self):
        res = forecast_simple_exponential_smoothing(self.series, alpha=0.3, horizon=self.horizon)
        self.assertEqual(len(res["forecast"]), self.horizon)
        self.assertTrue(all(f == res["forecast"][0] for f in res["forecast"]))

    def test_holt_linear_damped(self):
        res = forecast_holt_linear(self.series, alpha=0.25, beta=0.08, phi=0.95, damped=True, horizon=self.horizon)
        self.assertEqual(len(res["forecast"]), self.horizon)

    def test_holt_winters(self):
        res_add = forecast_holt_winters(self.series, period=7, alpha=0.25, beta=0.05, gamma=0.3, mode="additive", horizon=self.horizon)
        self.assertEqual(len(res_add["forecast"]), self.horizon)
        self.assertTrue(all(l <= u for l, u in zip(res_add["lower_95"], res_add["upper_95"])))

        res_mul = forecast_holt_winters(self.series, period=7, alpha=0.25, beta=0.05, gamma=0.3, mode="multiplicative", horizon=self.horizon)
        self.assertEqual(len(res_mul["forecast"]), self.horizon)

    def test_lagged_regression(self):
        res = forecast_lagged_regression(self.series, lags=[1, 2, 7], seasonal_period=7, horizon=self.horizon)
        self.assertEqual(len(res["forecast"]), self.horizon)

    def test_croston_sba(self):
        sparse = generate_intermittent_stockout_series(days=60)["series"]
        res = forecast_croston(sparse, alpha=0.15, horizon=self.horizon)
        self.assertEqual(len(res["forecast"]), self.horizon)
        self.assertGreaterEqual(res["demand_rate_per_period"], 0.0)


class TestEvaluationMetrics(unittest.TestCase):
    """Verifies all forecast accuracy metrics, mathematical relationships, and edge cases."""

    def test_basic_metrics_mathematics(self):
        y_true = [100.0, 110.0, 120.0, 130.0]
        y_pred = [105.0, 108.0, 125.0, 126.0]

        metrics = calculate_forecast_metrics(y_true, y_pred, y_train=[95.0, 100.0, 105.0])

        # Expected errors: [-5, +2, -5, +4] -> abs: [5, 2, 5, 4] -> sum=16, mean=4.0
        self.assertAlmostEqual(metrics["mae"], 4.0, delta=0.01)
        # Squared errors: [25, 4, 25, 16] -> sum=70, mean=17.5
        self.assertAlmostEqual(metrics["mse"], 17.5, delta=0.01)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(17.5), delta=0.01)

        # Mathematical Axiom: RMSE >= MAE always!
        self.assertGreaterEqual(metrics["rmse"], metrics["mae"])

        # Nonzero data: MAPE should be computed cleanly
        self.assertIsNotNone(metrics["mape"])
        self.assertIsNone(metrics["mape_warning"])

    def test_mape_division_by_zero_safety(self):
        """Verifies that MAPE safely catches zero actual values and reports warning without crashing."""
        y_true_with_zero = [10.0, 0.0, 15.0, 20.0]
        y_pred = [11.0, 2.0, 14.0, 19.0]

        metrics = calculate_forecast_metrics(y_true_with_zero, y_pred)
        self.assertIsNone(metrics["mape"])
        self.assertTrue(metrics["has_zero"])
        self.assertIn("Division by Zero Error", metrics["mape_warning"])

        # sMAPE and MASE should still be valid!
        self.assertIsNotNone(metrics["smape"])
        self.assertIsNotNone(metrics["mase"])

    def test_mase_scale_benchmark(self):
        """Verifies MASE benchmark interpretation (< 1.0 beats naive, > 1.0 worse)."""
        y_train = [10.0, 12.0, 14.0, 16.0, 18.0]
        y_true = [20.0, 22.0]
        # Superior predictions
        y_pred_good = [20.1, 21.9]
        metrics_good = calculate_forecast_metrics(y_true, y_pred_good, y_train=y_train)
        self.assertLess(metrics_good["mase"], 1.0)
        self.assertIn("Superior to baseline", metrics_good["mase_interpretation"])

        # Terrible predictions
        y_pred_bad = [50.0, 60.0]
        metrics_bad = calculate_forecast_metrics(y_true, y_pred_bad, y_train=y_train)
        self.assertGreater(metrics_bad["mase"], 1.0)
        self.assertIn("Worse than baseline", metrics_bad["mase_interpretation"])

    def test_mda_directional_accuracy(self):
        """Verifies Mean Directional Accuracy sign logic."""
        y_true = [10.0, 15.0, 12.0, 18.0]  # Changes: +5, -3, +6
        y_pred = [10.0, 14.0, 11.0, 20.0]  # Pred changes: +4, -3, +8 (All correct direction!)
        metrics = calculate_forecast_metrics(y_true, y_pred)
        self.assertEqual(metrics["mda"], 100.0)


class TestTemporalValidation(unittest.TestCase):
    """Verifies expanding-window rolling-origin cross-validation."""

    def test_rolling_origin_cv_folds(self):
        series = generate_ed_demand_series(days=60)["series"]
        roll_res = rolling_origin_evaluation(
            series,
            model_type="holt_winters",
            horizon=7,
            min_train_size=28,
            step_size=7,
            seasonal_period=7
        )

        self.assertGreaterEqual(roll_res["num_folds"], 3)
        self.assertEqual(len(roll_res["folds"]), roll_res["num_folds"])

        # Verify expanding train size
        train_sizes = [f["train_size"] for f in roll_res["folds"]]
        self.assertEqual(train_sizes, sorted(train_sizes))

        # Verify overall metrics exist
        self.assertIn("mae", roll_res["overall_metrics"])
        self.assertIn("rmse", roll_res["overall_metrics"])


class TestAlgorithmProfiles(unittest.TestCase):
    """Verifies complete plain-English educational metadata for all 8 algorithms."""

    def test_algorithm_profiles_completeness(self):
        expected_keys = ["naive", "seasonal_naive", "drift", "ses", "holt_linear", "holt_winters", "lagged_ml", "croston"]
        for k in expected_keys:
            self.assertIn(k, TIME_SERIES_ALGORITHMS)
            prof = TIME_SERIES_ALGORITHMS[k]
            self.assertTrue(len(prof["name"]) > 0)
            self.assertTrue(len(prof["tagline"]) > 0)
            self.assertTrue(len(prof["what_it_is"]) > 20)
            self.assertGreaterEqual(len(prof["when_to_use"]), 2)
            self.assertGreaterEqual(len(prof["when_not_to_use"]), 2)
            self.assertGreaterEqual(len(prof["assumptions"]), 2)
            self.assertGreaterEqual(len(prof["pros"]), 2)
            self.assertGreaterEqual(len(prof["cons"]), 2)
            self.assertIn("$$", prof["formula"])
            self.assertGreaterEqual(len(prof["variables_explained"]), 2)

    def test_get_algorithm_profile_routing(self):
        self.assertEqual(get_algorithm_profile("1. Naive Baseline (Last Value)")["id"], "naive")
        self.assertEqual(get_algorithm_profile("2. Seasonal Naive Baseline")["id"], "seasonal_naive")
        self.assertEqual(get_algorithm_profile("3. Drift Baseline (Linear Extrapolation)")["id"], "drift")
        self.assertEqual(get_algorithm_profile("4. Simple Exponential Smoothing (SES)")["id"], "ses")
        self.assertEqual(get_algorithm_profile("5. Holt's Linear Trend (Damped)")["id"], "holt_linear")
        self.assertEqual(get_algorithm_profile("6. Holt-Winters Triple Exponential Smoothing")["id"], "holt_winters")
        self.assertEqual(get_algorithm_profile("7. Lagged Feature Ridge Regression (ML)")["id"], "lagged_ml")
        self.assertEqual(get_algorithm_profile("8. Croston SBA (Intermittent Demand)")["id"], "croston")


class TestTimeSeriesASTCompliance(unittest.TestCase):
    """Enforces strict AST code quality standards ensuring every st.expander has expanded=False."""

    def test_time_series_files_expander_compliance(self):
        files_to_check = [
            os.path.join(PROJECT_ROOT, "pages", "7_⏱️_Time_Series_and_Forecasting.py"),
            os.path.join(PROJECT_ROOT, "src", "ui_components", "time_series_widgets.py")
        ]

        for file_path in files_to_check:
            self.assertTrue(os.path.exists(file_path), f"File {file_path} must exist.")
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=file_path)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    is_expander = False
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "expander":
                        is_expander = True

                    if is_expander:
                        kw_dict = {kw.arg: kw.value for kw in node.keywords}
                        self.assertIn(
                            "expanded",
                            kw_dict,
                            f"st.expander call at line {node.lineno} in {file_path} missing explicit expanded=False!"
                        )
                        val_node = kw_dict["expanded"]
                        self.assertIsInstance(
                            val_node,
                            ast.Constant,
                            f"st.expander at line {node.lineno} in {file_path} must have a constant expanded argument."
                        )
                        self.assertIs(
                            val_node.value,
                            False,
                            f"st.expander at line {node.lineno} in {file_path} must be expanded=False!"
                        )


class TestCodeAndGuideGenerators(unittest.TestCase):
    """Verifies Python code generation, Excel guides, and CSV exports for offline training."""

    def setUp(self):
        self.scenario = generate_ed_demand_series(days=30, seed=42)

    def test_csv_generator_produces_valid_structure(self):
        csv_str = generate_time_series_csv(self.scenario)
        lines = csv_str.strip().split("\n")
        self.assertEqual(len(lines), 31)  # 1 header + 30 data rows
        self.assertIn("Period,Observation_Index,Observed_", lines[0])

    def test_python_code_generation_for_all_models(self):
        model_keys = ["naive", "seasonal_naive", "drift", "ses", "holt_linear", "holt_winters", "lagged_ml", "croston"]
        for key in model_keys:
            py_code = generate_time_series_python_code(key, self.scenario, {"horizon": 7, "alpha": 0.25, "beta": 0.05, "gamma": 0.3})
            self.assertIsInstance(py_code, str)
            self.assertIn("import numpy as np", py_code)
            self.assertIn("horizon = 7", py_code)

    def test_excel_guide_generation_for_all_models(self):
        model_keys = ["naive", "seasonal_naive", "drift", "ses", "holt_linear", "holt_winters", "lagged_ml", "croston"]
        for key in model_keys:
            guide = generate_time_series_excel_guide(key, self.scenario, {"horizon": 7, "alpha": 0.25, "beta": 0.05, "gamma": 0.3})
            self.assertIsInstance(guide, str)
            self.assertIn("### 📊 Microsoft Excel Implementation Guide:", guide)
            self.assertIn("####", guide)

    def test_greek_symbol_rendering_in_algorithm_profiles(self):
        """Verifies that Greek letters in explanatory text use clean Unicode characters or KaTeX math mode."""
        for key, profile in TIME_SERIES_ALGORITHMS.items():
            # Check what_it_is, assumptions, pros, cons
            full_text = profile["what_it_is"] + " " + " ".join(profile["assumptions"]) + " " + " ".join(profile["pros"]) + " " + " ".join(profile["cons"])
            # Ensure no literal raw ASCII control characters like \x07 (bell) or \x08 (backspace)
            self.assertNotIn("\x07", full_text, f"Profile {key} contains ASCII bell character from raw \\alpha escape!")
            self.assertNotIn("\x08", full_text, f"Profile {key} contains ASCII backspace character from raw \\beta escape!")

        # Specifically check Naive, SES and Holt profiles for LaTeX math symbol formatting
        naive_profile = TIME_SERIES_ALGORITHMS["naive"]
        self.assertIn(r"$F_{T+h}$ — **Forecast Projection**", naive_profile["variables_explained"])
        self.assertIn(r"$Y_T$ — **Final Historical Observation**", naive_profile["variables_explained"])

        ses_profile = TIME_SERIES_ALGORITHMS["ses"]
        self.assertIn("α (Alpha)", ses_profile["what_it_is"])
        self.assertIn(r"$\alpha$ — **Level Smoothing Factor (Alpha)**", ses_profile["variables_explained"])
        self.assertIn(r"$F_{T+h}$ — **Flat Forecast Projection**", ses_profile["variables_explained"])

        holt_profile = TIME_SERIES_ALGORITHMS["holt_linear"]
        self.assertIn("ϕ (Phi)", holt_profile["what_it_is"])
        self.assertIn("β (Beta)", holt_profile["assumptions"][0])
        self.assertIn(r"$\alpha$ — **Level Smoothing Factor (Alpha)**", holt_profile["variables_explained"])
        self.assertIn(r"$\beta$ — **Trend Smoothing Factor (Beta)**", holt_profile["variables_explained"])
        self.assertIn(r"$\phi$ — **Trend Damping Factor (Phi)**", holt_profile["variables_explained"])


if __name__ == "__main__":
    unittest.main()

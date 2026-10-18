"""Unit and Contract Verification Test Suite for Statistical Fundamentals.

Validates:
1. Exact mathematical formulations in fundamentals_engine.py (Central Tendency, Dispersion, SD vs SEM, CLT, CIs).
2. All 6 synthetic practice data generators produce valid DataFrames and CSV strings.
3. UI widgets in fundamentals_widgets.py render without exceptions.
4. Strict AST compliance ensuring all st.expander calls specify expanded=False.
5. Page 0 route exists and compiles cleanly.
"""

import os
import sys
import ast
import unittest
from unittest.mock import MagicMock

# Mock streamlit and plotly before importing UI widgets if not in environment
if "streamlit" not in sys.modules:
    mock_st = MagicMock()
    mock_st.session_state = {}
    sys.modules["streamlit"] = mock_st

if "plotly" not in sys.modules:
    mock_plotly = MagicMock()
    sys.modules["plotly"] = mock_plotly
    sys.modules["plotly.graph_objects"] = mock_plotly.graph_objects
    sys.modules["plotly.express"] = mock_plotly.express

if "plotly.graph_objects" not in sys.modules:
    sys.modules["plotly.graph_objects"] = MagicMock()

if "plotly.express" not in sys.modules:
    sys.modules["plotly.express"] = MagicMock()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.stats_calcs.fundamentals_engine import (
    calculate_central_tendency,
    calculate_dispersion_and_quartiles,
    calculate_sd_vs_sem,
    calculate_distribution_shape_and_clt,
    calculate_confidence_interval_mean,
    calculate_confidence_interval_proportion,
    generate_scales_practice_data,
    generate_central_tendency_practice_data,
    generate_dispersion_practice_data,
    generate_sd_sem_practice_data,
    generate_clt_practice_data,
    generate_confidence_intervals_practice_data
)
from src.ui_components.fundamentals_widgets import (
    render_scales_data_types_module,
    render_central_tendency_module,
    render_dispersion_iqr_module,
    render_sd_vs_sem_module,
    render_distributions_clt_module,
    render_confidence_intervals_module,
    render_html
)


class TestFundamentalsCalculations(unittest.TestCase):
    """Verifies numerical precision and statistical formulations."""

    def test_central_tendency_clean_and_outlier(self):
        """Tests mean, median, mode, and skew detection with an extreme outlier."""
        clean_data = [10.0, 11.0, 12.0, 12.0, 13.0, 14.0]
        res_clean = calculate_central_tendency(clean_data)
        self.assertAlmostEqual(res_clean["mean"], 12.0, places=2)
        self.assertAlmostEqual(res_clean["median"], 12.0, places=2)
        self.assertEqual(res_clean["mode"], 12.0)
        self.assertFalse(res_clean["is_noticeably_skewed"])

        # Inject extreme outlier
        outlier_data = [10.0, 11.0, 12.0, 12.0, 13.0, 14.0, 150.0]
        res_outlier = calculate_central_tendency(outlier_data)
        self.assertGreater(res_outlier["mean"], res_outlier["median"])
        self.assertTrue(res_outlier["is_noticeably_skewed"])
        self.assertLess(res_outlier["trimmed_mean"], res_outlier["mean"])

    def test_dispersion_and_tukey_fences(self):
        """Tests sample variance, SD, quartiles, IQR, and Tukey's 1.5x / 3x fences."""
        data = [2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 25.0]  # 25.0 is an outlier
        res = calculate_dispersion_and_quartiles(data)
        self.assertGreater(res["variance"], 0.0)
        self.assertAlmostEqual(res["std_dev"], res["variance"] ** 0.5, places=4)
        self.assertGreater(res["q3"], res["q1"])
        self.assertEqual(res["iqr"], res["q3"] - res["q1"])
        self.assertGreater(res["tukey_inner_upper"], res["q3"])
        self.assertIn(25.0, res["mild_outliers"] + res["extreme_outliers"])

    def test_sd_vs_sem_scaling(self):
        """Verifies that SEM decreases with sqrt(N) while SD reflects sample dispersion."""
        small_sample = [10.0, 12.0, 14.0, 16.0]
        res_small = calculate_sd_vs_sem(small_sample)
        self.assertAlmostEqual(res_small["sem"], res_small["std_dev"] / 2.0, places=4)

        large_sample = list(range(100))
        res_large = calculate_sd_vs_sem(large_sample)
        self.assertAlmostEqual(res_large["sem"], res_large["std_dev"] / 10.0, places=4)
        self.assertLess(res_large["sem"], res_large["std_dev"])

    def test_distribution_shape_and_clt_simulation(self):
        """Verifies skewness classification and CLT sample mean convergence."""
        # Exponential data (right skewed)
        exp_data = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 25.0, 50.0, 100.0]
        res = calculate_distribution_shape_and_clt(exp_data, n_sim_samples=50, sim_sample_size=15)
        self.assertGreater(res["skewness"], 0.5)
        self.assertIn("Right-Skewed", res["shape_label"])
        self.assertEqual(len(res["sim_means"]), 50)
        self.assertGreater(res["theoretical_sem"], 0.0)

    def test_confidence_interval_mean_z_vs_t(self):
        """Verifies that Student's t-interval is wider than normal z-interval for small N."""
        sample = [5.0, 6.0, 7.0, 8.0, 9.0]  # N = 5, df = 4
        res = calculate_confidence_interval_mean(sample, confidence=0.95)
        self.assertTrue(res["is_small_sample"])
        self.assertGreater(res["t_crit"], res["z_crit"])
        self.assertGreater(res["t_margin"], res["z_margin"])
        self.assertLess(res["t_lower"], res["z_lower"])
        self.assertGreater(res["t_upper"], res["z_upper"])
        self.assertGreater(res["t_penalty_pct"], 0.0)

    def test_confidence_interval_proportion_wald_vs_wilson(self):
        """Verifies that Wilson score interval avoids negative lower bound when x = 0."""
        # 0 events in 30 trials
        res_zero = calculate_confidence_interval_proportion(successes=0, trials=30, confidence=0.95)
        self.assertFalse(res_zero["wald_valid"])
        self.assertEqual(res_zero["wald_lower"], 0.0)
        self.assertEqual(res_zero["wald_upper"], 0.0)
        # Wilson score must be > 0 in upper bound!
        self.assertEqual(res_zero["wilson_lower"], 0.0)
        self.assertGreater(res_zero["wilson_upper"], 0.05)
        self.assertLess(res_zero["wilson_upper"], 0.20)

        # Standard adequate sample: 25 successes in 50 trials (p = 0.50)
        res_mid = calculate_confidence_interval_proportion(successes=25, trials=50, confidence=0.95)
        self.assertTrue(res_mid["wald_valid"])
        self.assertAlmostEqual(res_mid["p_hat"], 0.50, places=2)
        self.assertAlmostEqual(res_mid["wilson_lower"], res_mid["wald_lower"], places=1)


class TestPracticeDataGenerators(unittest.TestCase):
    """Verifies that all 6 practice dataset generators return valid DataFrames and CSV strings."""

    def test_all_generators(self):
        generators = [
            generate_scales_practice_data,
            generate_central_tendency_practice_data,
            generate_dispersion_practice_data,
            generate_sd_sem_practice_data,
            generate_clt_practice_data,
            generate_confidence_intervals_practice_data
        ]
        for gen in generators:
            df, csv_str = gen()
            self.assertIsNotNone(df)
            self.assertGreater(len(df), 10, f"Generator {gen.__name__} returned too few rows")
            self.assertIsInstance(csv_str, str)
            self.assertIn(",", csv_str)


class TestFundamentalsWidgets(unittest.TestCase):
    """Verifies that all 6 UI modules render cleanly with mocked Streamlit."""

    def setUp(self):
        import src.ui_components.fundamentals_widgets as widget_mod
        self.mock_st = MagicMock()
        self.mock_st.session_state = {}
        self.mock_st.markdown = MagicMock()
        self.mock_st.html = MagicMock()
        self.mock_st.success = MagicMock()
        self.mock_st.info = MagicMock()
        self.mock_st.warning = MagicMock()
        self.mock_st.error = MagicMock()
        self.mock_st.metric = MagicMock()
        self.mock_st.dataframe = MagicMock()
        self.mock_st.plotly_chart = MagicMock()
        self.mock_st.download_button = MagicMock()
        self.mock_st.columns = MagicMock(side_effect=lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))])
        self.mock_st.tabs = MagicMock(side_effect=lambda labels: [MagicMock() for _ in range(len(labels))])
        self.mock_st.selectbox = MagicMock(side_effect=lambda label, options, **kwargs: options[0] if options else None)
        self.mock_st.slider = MagicMock(side_effect=lambda label, min_value, max_value, value, **kwargs: value)
        self.mock_st.radio = MagicMock(side_effect=lambda label, options, **kwargs: options[0] if options else None)
        self.mock_st.number_input = MagicMock(side_effect=lambda label, value=0.0, **kwargs: value)
        self.mock_st.checkbox = MagicMock(side_effect=lambda label, value=False, **kwargs: value)
        widget_mod.st = self.mock_st

    def test_render_all_six_modules(self):
        """Verifies each of the 6 fundamental modules executes without exception."""
        modules = [
            render_scales_data_types_module,
            render_central_tendency_module,
            render_dispersion_iqr_module,
            render_sd_vs_sem_module,
            render_distributions_clt_module,
            render_confidence_intervals_module
        ]
        for mod in modules:
            try:
                mod()
            except Exception as e:
                self.fail(f"Module {mod.__name__} crashed: {e}")

    def test_render_html_safety(self):
        """Verifies render_html safely trims indentation."""
        render_html("<div class='test'><span>Hello</span></div>")
        self.assertTrue(self.mock_st.html.called or self.mock_st.markdown.called)


class TestFundamentalsASTCompliance(unittest.TestCase):
    """Enforces strict AST code quality standards and expander rules."""

    def test_fundamentals_files_expander_compliance(self):
        """Verifies every st.expander in fundamentals files has expanded=False."""
        files_to_check = [
            os.path.join(PROJECT_ROOT, "pages", "0_📚_Statistical_Fundamentals.py"),
            os.path.join(PROJECT_ROOT, "src", "ui_components", "fundamentals_widgets.py")
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


if __name__ == "__main__":
    unittest.main()

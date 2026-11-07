"""Unit and Contract Verification Test Suite for Health Inequalities (SII & RII).

Validates:
1. Cumulative relative rank (ridit) calculation and boundary constraints.
2. Slope Index of Inequality (SII) and Relative Index of Inequality (RII) point estimates and CIs.
3. Linear WLS and Log-linear GLM formulations.
4. Reverse direction ordering logic (e.g., life expectancy).
5. Diagnostic prerequisites evaluation (strata counts, populations, non-negative rates, monotonicity).
6. Synthetic scenario data structures.
7. Strict AST expander compliance (expanded=False).
8. Headless UI widget rendering without exceptions.
"""

import os
import sys
import ast
import unittest
from unittest.mock import MagicMock
import importlib.util

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import health_inequalities directly to avoid triggering src.epi_methods.__init__ if pandas not in local test env
hi_path = os.path.join(PROJECT_ROOT, "src", "epi_methods", "health_inequalities.py")
spec = importlib.util.spec_from_file_location("health_inequalities", hi_path)
hi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hi)

calculate_relative_ranks = hi.calculate_relative_ranks
calculate_sii_rii = hi.calculate_sii_rii
evaluate_health_inequalities_prerequisites = hi.evaluate_health_inequalities_prerequisites
generate_inequality_demo_data = hi.generate_inequality_demo_data

try:
    import numpy as np
    import pandas as pd
    import streamlit as st
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestHealthInequalitiesMathEngine(unittest.TestCase):
    """Verifies mathematical correctness of ridit ranks, WLS slopes, and inequality ratios."""

    def test_relative_ranks_calculation(self):
        """Verifies midpoint ridit calculations match cumulative share bounds."""
        populations = [1000, 2000, 3000, 4000]  # Total = 10,000
        ranks = calculate_relative_ranks(populations)
        self.assertEqual(ranks["total_population"], 10000)
        self.assertAlmostEqual(sum(ranks["proportions"]), 1.0, places=6)
        self.assertAlmostEqual(ranks["midpoint_ranks"][0], 0.05, places=5)
        self.assertAlmostEqual(ranks["midpoint_ranks"][1], 0.20, places=5)
        self.assertAlmostEqual(ranks["midpoint_ranks"][2], 0.45, places=5)
        self.assertAlmostEqual(ranks["midpoint_ranks"][3], 0.80, places=5)

    def test_relative_ranks_negative_population_error(self):
        """Verifies calculate_relative_ranks rejects negative or zero populations."""
        with self.assertRaises(ValueError):
            calculate_relative_ranks([100, 0, 200])
        with self.assertRaises(ValueError):
            calculate_relative_ranks([100, -50, 200])

    def test_sii_rii_derbyshire_cvd_scenario(self):
        """Verifies SII and RII estimates on the synthetic CVD educational dataset."""
        data = generate_inequality_demo_data("derbyshire_cvd")
        res = calculate_sii_rii(
            rates=data["rates"],
            populations=data["populations"],
            strata_names=data["strata_names"],
            indicator_name=data["indicator_name"]
        )

        # Check positive absolute gap and relative ratio > 1.0
        self.assertGreater(res["sii"], 80.0)
        self.assertLess(res["sii"], 110.0)
        self.assertGreater(res["rii"], 2.5)
        self.assertLess(res["rii"], 4.0)

        # Check CI consistency: lower < point < upper
        self.assertLess(res["sii_ci_lower"], res["sii"])
        self.assertGreater(res["sii_ci_upper"], res["sii"])
        self.assertLess(res["rii_ci_lower"], res["rii"])
        self.assertGreater(res["rii_ci_upper"], res["rii"])

        # Check statistical significance flags
        self.assertTrue(res["is_sii_significant"])
        self.assertLess(res["p_value_sii"], 0.05)
        self.assertIn("Statistically Significant", res["sii_significance_label"])

        # Check goodness of fit
        self.assertGreater(res["r_squared"], 0.90)
        self.assertTrue(res["is_gradient_pro_affluent"])
        self.assertEqual(len(res["table"]), 10)

    def test_statistical_significance_flag_non_significant(self):
        """Verifies is_sii_significant is False when slope CI spans zero."""
        # Flat rates with small noise
        rates = [50.0, 50.1, 49.9, 50.2, 50.0]
        pops = [10000, 10000, 10000, 10000, 10000]
        res = calculate_sii_rii(rates=rates, populations=pops)

        self.assertFalse(res["is_sii_significant"])
        self.assertGreaterEqual(res["p_value_sii"], 0.05)
        self.assertIn("Not Statistically Significant", res["sii_significance_label"])

    def test_synthetic_scenario_disclaimers(self):
        """Verifies all demo scenarios have clear educational disclaimers stating data is synthetic."""
        for key in ["derbyshire_cvd", "life_expectancy_gap", "childhood_obesity"]:
            data = generate_inequality_demo_data(key)
            self.assertIn("disclaimer", data, f"Scenario {key} must contain a disclaimer.")
            disclaimer_lower = data["disclaimer"].lower()
            self.assertTrue(
                "synthetic" in disclaimer_lower or "made-up" in disclaimer_lower,
                f"Scenario {key} disclaimer must state data is synthetic or made-up."
            )
            self.assertNotIn("authentic", data["description"].lower(), f"Scenario {key} description should not claim data is authentic.")

    def test_sii_rii_reversed_direction_life_expectancy(self):
        """Verifies life expectancy gradient where higher is better (positive SII under reverse_direction)."""
        data = generate_inequality_demo_data("life_expectancy_gap")
        res = calculate_sii_rii(
            rates=data["rates"],
            populations=data["populations"],
            strata_names=data["strata_names"],
            reverse_direction=True
        )

        # Life expectancy should show ~11.9 years gap between Q5 and Q1
        self.assertGreater(res["sii"], 8.0)
        self.assertLess(res["sii"], 13.0)
        self.assertGreater(res["rii"], 1.05)

        # Crucial check: Q1 is most deprived (lower LE ~72.4y), Q5 is least deprived (higher LE ~84.4y)
        # Prevents slope line inversion counter to deprivation trend
        self.assertLess(res["y_most_deprived"], res["y_least_deprived"])
        self.assertAlmostEqual(res["y_most_deprived"], 72.44, delta=0.5)
        self.assertAlmostEqual(res["y_least_deprived"], 84.37, delta=0.5)
        self.assertLess(res["fitted_rates"][0], res["fitted_rates"][-1])

    def test_log_linear_model_output(self):
        """Verifies log-linear GLM calculations and promotion to primary result keys."""
        data = generate_inequality_demo_data("childhood_obesity")
        res = calculate_sii_rii(
            rates=data["rates"],
            populations=data["populations"],
            strata_names=data["strata_names"],
            model_type="log_linear"
        )
        self.assertIn("log_model", res)
        log_res = res["log_model"]
        self.assertGreater(log_res["rii_log"], 1.5)
        self.assertLess(log_res["rii_log_ci_lower"], log_res["rii_log"])
        self.assertGreater(log_res["rii_log_ci_upper"], log_res["rii_log"])

        # Check primary key promotion
        self.assertEqual(res["model_type"], "log_linear")
        self.assertIn("Log-Linear GLM", res["model_name"])
        self.assertAlmostEqual(res["sii"], log_res["sii_log"], places=5)
        self.assertAlmostEqual(res["rii"], log_res["rii_log"], places=5)
        self.assertEqual(res["fitted_rates"], log_res["fitted_rates_log"])

    def test_model_type_toggle_differences(self):
        """Verifies switching from linear to log-linear changes point estimates and fitted rates."""
        data = generate_inequality_demo_data("derbyshire_cvd")
        res_lin = calculate_sii_rii(
            rates=data["rates"],
            populations=data["populations"],
            strata_names=data["strata_names"],
            model_type="linear"
        )
        res_log = calculate_sii_rii(
            rates=data["rates"],
            populations=data["populations"],
            strata_names=data["strata_names"],
            model_type="log_linear"
        )

        self.assertNotEqual(res_lin["sii"], res_log["sii"])
        self.assertNotEqual(res_lin["rii"], res_log["rii"])
        self.assertNotEqual(res_lin["fitted_rates"], res_log["fitted_rates"])
        self.assertNotEqual(res_lin["y_most_deprived"], res_log["y_most_deprived"])
        self.assertNotEqual(res_lin["y_least_deprived"], res_log["y_least_deprived"])

    def test_input_validation_errors(self):
        """Verifies error handling for mismatched lengths, <3 strata, and negative rates."""
        # <3 strata
        with self.assertRaises(ValueError):
            calculate_sii_rii([100, 50], [1000, 1000])
        # Mismatched length
        with self.assertRaises(ValueError):
            calculate_sii_rii([100, 80, 60], [1000, 1000])
        # Negative rate
        with self.assertRaises(ValueError):
            calculate_sii_rii([100, -10, 50], [1000, 1000, 1000])


class TestHealthInequalitiesDiagnostics(unittest.TestCase):
    """Verifies prerequisite checking and diagnostic evaluation."""

    def test_prerequisites_pass_on_valid_data(self):
        """Verifies valid 10-decile data passes all diagnostic checks."""
        data = generate_inequality_demo_data("derbyshire_cvd")
        diag = evaluate_health_inequalities_prerequisites(data["rates"], data["populations"])
        self.assertEqual(diag["overall_status"], "PASS")
        self.assertEqual(len(diag["checks"]), 4)

    def test_prerequisites_fail_on_negative_population(self):
        """Verifies negative populations trigger a FAIL check."""
        diag = evaluate_health_inequalities_prerequisites([100, 80, 60], [1000, -500, 1000])
        self.assertEqual(diag["overall_status"], "FAIL")

    def test_prerequisites_warning_on_small_strata(self):
        """Verifies K=3 strata yields a WARNING on statistical power/degrees of freedom."""
        diag = evaluate_health_inequalities_prerequisites([100, 80, 60], [1000, 1000, 1000])
        self.assertEqual(diag["overall_status"], "WARNING")


class TestHealthInequalitiesASTCompliance(unittest.TestCase):
    """Enforces strict AST quality standards and expander invariants."""

    def test_expander_expanded_false_compliance(self):
        """Verifies all st.expander calls in epi_health_inequalities_widgets.py have expanded=False."""
        widget_file = os.path.join(PROJECT_ROOT, "src", "ui_components", "epi_health_inequalities_widgets.py")
        self.assertTrue(os.path.exists(widget_file), f"File {widget_file} must exist.")

        with open(widget_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=widget_file)

        expander_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                is_expander = False
                if isinstance(node.func, ast.Attribute) and node.func.attr == "expander":
                    is_expander = True

                if is_expander:
                    expander_count += 1
                    kw_dict = {kw.arg: kw.value for kw in node.keywords}
                    self.assertIn(
                        "expanded",
                        kw_dict,
                        f"st.expander call at line {node.lineno} in {widget_file} missing explicit expanded=False!"
                    )
                    val_node = kw_dict["expanded"]
                    self.assertIsInstance(
                        val_node,
                        ast.Constant,
                        f"st.expander at line {node.lineno} in {widget_file} must have a constant expanded argument."
                    )
                    self.assertIs(
                        val_node.value,
                        False,
                        f"st.expander at line {node.lineno} in {widget_file} must be expanded=False!"
                    )

        self.assertGreaterEqual(expander_count, 1, "Must find at least one expander in widget file.")


class TestHealthInequalitiesUIHeadless(unittest.TestCase):
    """Verifies headless execution of UI rendering functions."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack not installed in this environment.")

    def test_render_health_inequalities_module_headless(self):
        """Verifies render_health_inequalities_module executes cleanly without crashing."""
        from src.ui_components.epi_health_inequalities_widgets import render_health_inequalities_module
        try:
            render_health_inequalities_module()
        except Exception as e:
            self.fail(f"render_health_inequalities_module crashed: {e}")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Missing Data Mechanisms, Imputation & Sensitivity Audits.
"""

import ast
import os
import sys
import unittest
import math
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.missing_data.missing_engine import (
    calculate_rubin_taxonomy_model,
    generate_rubin_dataset,
    calculate_complete_case_vs_single_model,
    generate_complete_case_dataset,
    calculate_mice_rubin_model,
    generate_mice_dataset,
    calculate_knn_imputation_model,
    generate_knn_dataset,
    calculate_mnar_tipping_point_model,
    generate_tipping_point_dataset
)
from src.missing_data.missing_schema import MISSING_DATA_METHODS, MISSING_DATA_SCENARIOS


class TestMissingDataEngine(unittest.TestCase):
    """Verifies mathematical validity of missing data calculation engines."""

    def test_rubin_taxonomy_mcar_vs_mar(self):
        """Tests Little's MCAR diagnostic audit: MAR should exhibit significant mean difference."""
        mar_model = calculate_rubin_taxonomy_model(n_records=800, pct_missing=0.25, mechanism="MAR_by_sofa", seed=42)
        self.assertGreater(mar_model["n_missing"], 0)
        self.assertGreater(mar_model["n_observed"], 0)
        self.assertGreater(mar_model["mean_sofa_missing"], mar_model["mean_sofa_observed"])
        self.assertLess(mar_model["p_value"], 0.05)
        self.assertIn("Evidence Against MCAR", mar_model["diagnosis"])

        mcar_model = calculate_rubin_taxonomy_model(n_records=800, pct_missing=0.25, mechanism="MCAR", seed=42)
        self.assertGreater(mcar_model["p_value"], 0.01)

    def test_complete_case_sample_loss_and_variance_collapse(self):
        """Tests listwise deletion attrition formula and mean imputation variance shrinkage."""
        res = calculate_complete_case_vs_single_model(n_records=500, n_covariates=6, pct_missing_per_var=0.12, seed=42)
        expected_retained = ((0.88) ** 6) * 100.0
        self.assertAlmostEqual(res["theoretical_retained_pct"], expected_retained, places=2)
        self.assertLess(res["empirical_retained_count"], 500)
        self.assertEqual(res["variance_shrinkage_pct"], 12.0)
        self.assertEqual(res["retained_variance_pct"], 88.0)
        self.assertAlmostEqual(res["se_deflation_factor"], math.sqrt(0.88), places=3)

    def test_mice_rubin_rules_pooling(self):
        """Verifies Rubin's Rules: Q-bar, U-bar, B, and Total Variance T = U-bar + (1 + 1/m)B."""
        res = calculate_mice_rubin_model(m_imputations=10, n_patients=600, pct_missing=0.20, true_treatment_effect=-0.65, seed=42)
        m = res["m_imputations"]
        u_bar = res["u_bar_within_variance"]
        b_var = res["b_between_variance"]
        expected_t = u_bar + (1.0 + 1.0 / m) * b_var

        self.assertAlmostEqual(res["total_variance_t"], expected_t, places=6)
        self.assertAlmostEqual(res["pooled_se"], math.sqrt(expected_t), places=6)
        self.assertGreater(res["fmi"], 0.0)
        self.assertLess(res["fmi"], 1.0)
        self.assertLess(res["p_value"], 0.001)

    def test_knn_imputation_engine(self):
        """Verifies k-NN donor imputation preserves sample size and computes finite RMSE."""
        res = calculate_knn_imputation_model(n_samples=200, k_neighbors=5, pct_missing=0.15, seed=42)
        self.assertGreater(res["n_imputed"], 0)
        self.assertGreater(res["n_donors"], 0)
        self.assertEqual(res["n_imputed"] + res["n_donors"], 200)
        self.assertGreater(res["imputation_rmse"], 0.0)
        self.assertLess(res["imputation_rmse"], 10.0)

    def test_mnar_tipping_point_sensitivity(self):
        """Verifies ICH E9 tipping point delta identification when p-value crosses 0.05."""
        res = calculate_mnar_tipping_point_model(
            baseline_mar_diff=-3.40,
            se_mar=1.15,
            dropout_rate_treated=0.22,
            delta_min=0.0,
            delta_max=6.0,
            delta_steps=25
        )
        self.assertIsNotNone(res["tipping_point_delta"])
        self.assertAlmostEqual(res["tipping_point_delta"], 5.25, delta=0.5)
        self.assertIn("Robust", res["robustness_tier"])


class TestMissingDataSchema(unittest.TestCase):
    """Verifies schema definitions, required keys, and mathematical formulations."""

    def test_all_five_methods_present(self):
        expected_methods = [
            "rubin_taxonomy",
            "complete_case_vs_single",
            "mice_rubin_rules",
            "knn_missforest",
            "mnar_tipping_point"
        ]
        self.assertEqual(len(MISSING_DATA_METHODS), 5)
        for m in expected_methods:
            self.assertIn(m, MISSING_DATA_METHODS)

    def test_schema_required_keys(self):
        required_keys = [
            "name", "short_name", "icon", "category", "tagline",
            "mathematical_formulation", "plain_english_glossary",
            "assumptions", "pros_cons"
        ]
        for key, method in MISSING_DATA_METHODS.items():
            for rk in required_keys:
                self.assertIn(rk, method, f"Method {key} missing required key: {rk}")
            self.assertGreater(len(method["mathematical_formulation"]), 100)
            self.assertGreater(len(method["plain_english_glossary"]), 0)
            self.assertGreater(len(method["assumptions"]), 0)

    def test_scenarios_reference_valid_methods(self):
        for s_key, scen in MISSING_DATA_SCENARIOS.items():
            self.assertIn(scen["method_key"], MISSING_DATA_METHODS)


class TestMissingDataASTCompliance(unittest.TestCase):
    """Enforces 100% strict adherence to st.expander(expanded=False)."""

    def test_all_missing_data_files_have_expanded_false(self):
        files_to_check = [
            os.path.join(PROJECT_ROOT, "pages", "11_🧩_Missing_Data_and_Imputation.py"),
            os.path.join(PROJECT_ROOT, "src", "ui_components", "missing_widgets.py")
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
                    elif isinstance(node.func, ast.Name) and node.func.id == "expander":
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

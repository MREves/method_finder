"""Unit tests for Longitudinal & Hierarchical Mixed-Effects Models Engine and Schema.
"""

import ast
import os
import sys
import unittest
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mixed_models.mixed_engine import (
    calculate_lmm_random_slopes_model,
    generate_lmm_dataset,
    calculate_glmm_binary_count_model,
    generate_glmm_dataset,
    calculate_rm_anova_vs_lmm_model,
    generate_rm_anova_dataset,
    calculate_hierarchical_3level_model,
    generate_3level_dataset,
    calculate_covariance_selection_model,
    generate_covariance_dataset
)
from src.mixed_models.mixed_schema import MIXED_MODELS_METHODS, MIXED_MODELS_SCENARIOS


class TestMixedModelsEngine(unittest.TestCase):
    """Verifies mathematical validity of longitudinal & hierarchical calculation engines."""

    def test_lmm_random_slopes_calculation(self):
        """Verifies LMM ICC, random variances, and fixed effect interaction test."""
        res = calculate_lmm_random_slopes_model(
            n_subjects=60,
            n_visits=4,
            beta_0=24.5,
            beta_time=-1.8,
            beta_tx=-0.5,
            beta_interaction=-1.2,
            sigma_u0=4.0,
            sigma_u1=1.0,
            rho_01=-0.2,
            sigma_eps=2.0,
            seed=42
        )
        self.assertEqual(res["n_subjects"], 60)
        self.assertEqual(res["total_observations"], 240)
        # ICC = 4.0^2 / (4.0^2 + 2.0^2) = 16 / 20 = 0.80
        self.assertAlmostEqual(res["icc"], 0.80, places=3)
        self.assertLess(res["p_interaction"], 0.05)
        self.assertGreater(res["shrinkage_factor"], 0.5)

        records = generate_lmm_dataset(n_subjects=10, n_visits=4)
        self.assertEqual(len(records), 40)
        self.assertIn("adas_cog_score", records[0])

    def test_glmm_binary_count_calculation(self):
        """Verifies GLMM subject-specific OR vs marginal population-averaged OR attenuation."""
        res = calculate_glmm_binary_count_model(
            n_clusters=40,
            cluster_size=15,
            beta_0=-0.8,
            beta_tx=-0.95,
            sigma_u=0.85,
            outcome_type="binary",
            seed=42
        )
        # OR_SS = exp(-0.95) ~ 0.3867
        self.assertAlmostEqual(res["or_subject_specific"], math.exp(-0.95), places=3)
        # Non-collapsibility: marginal effect is shrunk towards 1, so OR_marginal > OR_SS
        self.assertGreater(res["or_marginal"], res["or_subject_specific"])
        self.assertGreater(res["attenuation_pct"], 0.0)
        self.assertLess(res["attenuation_pct"], 30.0)

        records = generate_glmm_dataset(n_clusters=5, cluster_size=10)
        self.assertEqual(len(records), 50)
        self.assertIn("relapse_event", records[0])

    def test_rm_anova_vs_lmm_sphericity(self):
        """Verifies Mauchly's sphericity rejection, Greenhouse-Geisser epsilon, and LMM retention."""
        res = calculate_rm_anova_vs_lmm_model(
            n_patients=60,
            n_timepoints=4,
            missing_rate=0.25,
            sphericity_violation=True,
            seed=42
        )
        self.assertLess(res["gg_epsilon"], 1.0)
        self.assertGreater(res["gg_epsilon"], 0.33)  # lower bound is 1/(k-1) = 1/3
        self.assertGreater(res["pct_dropped_rm_anova"], 0.0)
        self.assertEqual(res["lmm_retained_pct"], 100.0)
        self.assertEqual(res["lmm_retained_patients"], 60)

        records = generate_rm_anova_dataset(n_patients=10, n_timepoints=4)
        self.assertEqual(len(records), 40)
        self.assertIn("pain_vas_score", records[0])

    def test_hierarchical_3level_model(self):
        """Verifies 3-level variance partitioning: hospital, clinician, and patient tiers."""
        res = calculate_hierarchical_3level_model(
            n_hospitals=20,
            clinicians_per_hosp=5,
            patients_per_clin=10,
            sigma_hosp=1.5,
            sigma_clin=1.0,
            sigma_error=2.0,
            seed=42
        )
        # Total var = 1.5^2 + 1.0^2 + 2.0^2 = 2.25 + 1.0 + 4.0 = 7.25
        expected_total = 7.25
        self.assertAlmostEqual(res["total_variance"], expected_total, places=3)
        # VPC sum must equal 100%
        sum_vpc = res["vpc_hosp"] + res["vpc_clin"] + res["vpc_error"]
        self.assertAlmostEqual(sum_vpc, 100.0, places=1)
        self.assertGreater(res["rho_same_clinician"], res["rho_diff_clinician_same_hosp"])

        records = generate_3level_dataset(n_hospitals=2, clinicians_per_hosp=2, patients_per_clin=5)
        self.assertEqual(len(records), 20)
        self.assertIn("recovery_days", records[0])

    def test_covariance_selection_model(self):
        """Verifies AR(1) vs CS AIC selection and Nakagawa marginal/conditional R2."""
        res = calculate_covariance_selection_model(n_subjects=50, n_timepoints=4, rho_ar1=0.65)
        self.assertEqual(len(res["models"]), 4)
        self.assertEqual(res["best_model_by_aic"], "Autoregressive AR(1)")
        # R2 marginal <= R2 conditional
        self.assertLess(res["r2_marginal"], res["r2_conditional"])
        self.assertGreater(res["r2_conditional"], 0.0)
        self.assertLess(res["r2_conditional"], 1.0)

        records = generate_covariance_dataset(n_subjects=10, n_timepoints=4)
        self.assertEqual(len(records), 40)
        self.assertIn("troponin_t_ng_l", records[0])


class TestMixedModelsSchema(unittest.TestCase):
    """Verifies schema structure, required keys, and scenario references."""

    def test_all_five_methods_present(self):
        expected_methods = [
            "lmm_random_slopes",
            "glmm_binary_count",
            "rm_anova_vs_lmm",
            "hierarchical_3level",
            "covariance_model_selection"
        ]
        self.assertEqual(len(MIXED_MODELS_METHODS), 5)
        for m in expected_methods:
            self.assertIn(m, MIXED_MODELS_METHODS)

    def test_schema_required_keys(self):
        required_keys = [
            "name", "short_name", "icon", "category", "tagline",
            "mathematical_formulation", "plain_english_glossary",
            "assumptions", "pros_cons"
        ]
        for key, method in MIXED_MODELS_METHODS.items():
            for rk in required_keys:
                self.assertIn(rk, method, f"Method {key} missing required key: {rk}")
            self.assertGreater(len(method["mathematical_formulation"]), 100)
            self.assertGreater(len(method["plain_english_glossary"]), 0)
            self.assertGreater(len(method["assumptions"]), 0)

    def test_scenarios_reference_valid_methods(self):
        for s_key, scen in MIXED_MODELS_SCENARIOS.items():
            self.assertIn(scen["method_key"], MIXED_MODELS_METHODS)


class TestMixedModelsASTCompliance(unittest.TestCase):
    """Enforces 100% strict adherence to st.expander(expanded=False)."""

    def test_all_mixed_models_files_have_expanded_false(self):
        files_to_check = [
            os.path.join(PROJECT_ROOT, "pages", "12_🪜_Longitudinal_and_Mixed_Models.py"),
            os.path.join(PROJECT_ROOT, "src", "ui_components", "mixed_widgets.py")
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

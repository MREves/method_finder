"""Unit Tests for Survey & Questionnaire Analysis Module.

Tests schema integrity, mathematical formulations (Likert metrics, Cronbach's Alpha,
McDonald's Omega, Cohen's Kappa, Factor Analysis, FPC Sizing, Survey Weighting),
synthetic dataset generators, and AST expander compliance.
"""

import os
import unittest
try:
    import numpy as np
    import pandas as pd
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False
    np = None
    pd = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from src.survey_analysis.survey_schema import (
    SURVEY_METHODS,
    SURVEY_CATEGORIES,
    SURVEY_DECISION_NODES
)
from src.survey_analysis.survey_engine import (
    calculate_likert_metrics,
    calculate_cronbachs_alpha,
    calculate_mcdonalds_omega,
    calculate_cohens_kappa,
    calculate_weighted_kappa,
    calculate_fleiss_kappa,
    calculate_factor_analysis,
    calculate_survey_sample_size_fpc,
    calculate_survey_weights_kish,
    generate_synthetic_survey_data
)


class TestSurveySchemaIntegrity(unittest.TestCase):
    """Verifies that all 7 survey methods have complete schemas."""

    def test_all_categories_defined(self):
        self.assertEqual(len(SURVEY_CATEGORIES), 5)
        for cat_id, info in SURVEY_CATEGORIES.items():
            self.assertIn("name", info)
            self.assertIn("icon", info)
            self.assertIn("description", info)

    def test_all_seven_methods_have_required_fields(self):
        self.assertEqual(len(SURVEY_METHODS), 7)
        expected_keys = [
            "likert_scale_analysis",
            "cronbachs_alpha",
            "mcdonalds_omega",
            "cohens_kappa",
            "exploratory_factor_analysis",
            "survey_sample_size",
            "survey_weighting"
        ]
        for key in expected_keys:
            self.assertIn(key, SURVEY_METHODS)
            meta = SURVEY_METHODS[key]
            self.assertEqual(meta["id"], key)
            self.assertIn("name", meta)
            self.assertIn("category", meta)
            self.assertIn("icon", meta)
            self.assertIn("summary", meta)
            self.assertIn("when_to_use", meta)
            self.assertIn("key_assumptions", meta)
            self.assertIn("common_pitfalls", meta)
            self.assertIn("plain_english_rules", meta)
            self.assertIn("formulas", meta)
            self.assertIn("excel_guide", meta)
            self.assertIn("python_code", meta)
            
            # Check Excel guide structure
            self.assertIn("cell_mapping", meta["excel_guide"])
            self.assertIn("steps", meta["excel_guide"])
            
            # Check formulas structure
            for _, f_data in meta["formulas"].items():
                self.assertIn("latex", f_data)
                self.assertIn("plain_english", f_data)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestLikertScaleCalculations(unittest.TestCase):
    """Verifies Likert scale descriptive metrics and Norman 2010 rules."""

    def setUp(self):
        self.df = generate_synthetic_survey_data("likert_scale_analysis", n_samples=200, seed=42)
        self.item_cols = [c for c in self.df.columns if c != "Respondent_ID"]

    def test_likert_metrics_structure(self):
        res = calculate_likert_metrics(self.df[self.item_cols])
        self.assertIn("item_summary", res)
        self.assertIn("composite", res)
        
        df_sum = res["item_summary"]
        self.assertEqual(len(df_sum), len(self.item_cols))
        for col in ["Top_2_Box_Pct", "Bottom_2_Box_Pct", "Net_Agreement_Pct", "Median", "IQR", "Mean", "SD"]:
            self.assertIn(col, df_sum.columns)
            
        # Top-2-Box + Bottom-2-Box + Neutral should equal approximately 100%
        for _, row in df_sum.iterrows():
            total_pct = row["Top_2_Box_Pct"] + row["Bottom_2_Box_Pct"] + row["Pct_Neutral"]
            self.assertAlmostEqual(total_pct, 100.0, delta=0.5)

    def test_composite_metrics(self):
        res = calculate_likert_metrics(self.df[self.item_cols])
        comp = res["composite"]
        self.assertGreater(comp["mean"], 5.0)
        self.assertGreater(comp["sd"], 0.0)
        self.assertIn("is_approximately_normal", comp)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestCronbachsAlphaCalculations(unittest.TestCase):
    """Verifies Cronbach's Alpha, Feldt CI, and Alpha-if-deleted."""

    def setUp(self):
        self.df = generate_synthetic_survey_data("cronbachs_alpha", n_samples=150, seed=42)
        self.item_cols = [c for c in self.df.columns if c != "Clinician_ID"]

    def test_cronbach_alpha_calculation(self):
        res = calculate_cronbachs_alpha(self.df[self.item_cols])
        alpha = res["alpha"]
        self.assertGreater(alpha, 0.70)
        self.assertLess(alpha, 1.0)
        
        # 95% CI
        ci_low, ci_high = res["ci_95"]
        self.assertLessEqual(ci_low, alpha)
        self.assertGreaterEqual(ci_high, alpha)
        
        # Item analysis table
        df_items = res["item_analysis"]
        self.assertEqual(len(df_items), len(self.item_cols))
        self.assertIn("Alpha_If_Item_Deleted", df_items.columns)
        self.assertIn("Corrected_Item_Total_r", df_items.columns)
        
        # Rating tier and detail checks
        self.assertIn("rating_tier", res)
        self.assertIn("rating_detail", res)
        self.assertIn(res["rating_tier"], ["Excellent", "Good", "Acceptable", "Marginal / Questionable", "Unacceptable"])


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestMcDonaldsOmegaCalculations(unittest.TestCase):
    """Verifies McDonald's Omega and tau-equivalence detection."""

    def setUp(self):
        self.df = generate_synthetic_survey_data("mcdonalds_omega", n_samples=150, seed=42)
        self.item_cols = [c for c in self.df.columns if c != "Clinician_ID"]

    def test_mcdonalds_omega_calculation(self):
        res = calculate_mcdonalds_omega(self.df[self.item_cols])
        omega = res["omega_total"]
        alpha = res["cronbach_alpha"]
        
        self.assertGreater(omega, 0.70)
        self.assertLessEqual(omega, 1.0)
        # In presence of varied loadings, omega is generally >= alpha
        self.assertGreaterEqual(omega, alpha - 0.05)
        self.assertIn("tau_equivalence_violated", res)
        self.assertEqual(len(res["loadings_table"]), len(self.item_cols))


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestCohensKappaCalculations(unittest.TestCase):
    """Verifies unweighted, linear, and quadratic weighted Cohen's Kappa."""

    def setUp(self):
        self.df = generate_synthetic_survey_data("cohens_kappa", n_samples=100, seed=42)
        self.r1 = self.df["Rater1_Triage_Nurse"].tolist()
        self.r2 = self.df["Rater2_Senior_Clinician"].tolist()

    def test_cohens_kappa_calculation(self):
        res = calculate_cohens_kappa(self.r1, self.r2)
        self.assertGreater(res["kappa_unweighted"], 0.40)
        self.assertGreater(res["kappa_quadratic_weighted"], 0.50)
        self.assertGreater(res["observed_agreement_pct"], res["expected_agreement_pct"])
        self.assertIn("Almost" in res["benchmark"] or "Substantial" in res["benchmark"] or "Moderate" in res["benchmark"], [True])
        self.assertEqual(res["confusion_matrix"].shape, (3, 3))

    def test_fleiss_kappa_calculation(self):
        # 10 subjects rated by 3 raters into 2 categories
        matrix = np.array([
            [3, 0],
            [3, 0],
            [2, 1],
            [0, 3],
            [0, 3],
            [1, 2],
            [3, 0],
            [0, 3],
            [2, 1],
            [3, 0]
        ])
        res = calculate_fleiss_kappa(matrix)
        self.assertGreater(res["fleiss_kappa"], 0.50)
        self.assertEqual(res["n_subjects"], 10)
        self.assertEqual(res["n_raters"], 3)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestFactorAnalysisCalculations(unittest.TestCase):
    """Verifies KMO, Bartlett's test, Scree eigenvalues, and Varimax loadings."""

    def setUp(self):
        self.df = generate_synthetic_survey_data("exploratory_factor_analysis", n_samples=200, seed=42)
        self.item_cols = [c for c in self.df.columns if c != "Patient_ID"]

    def test_factor_analysis_screening(self):
        res = calculate_factor_analysis(self.df[self.item_cols])
        self.assertGreaterEqual(res["kmo"], 0.60) # Adequate for EFA
        self.assertLess(res["bartlett_p_value"], 0.05) # Significant correlations
        self.assertGreaterEqual(res["n_factors_kaiser"], 2) # At least 2 factors with lambda >= 1.0
        
        df_scree = res["scree_table"]
        self.assertEqual(len(df_scree), len(self.item_cols))
        self.assertIn("Eigenvalue", df_scree.columns)
        
        df_loadings = res["loadings_table"]
        self.assertEqual(len(df_loadings), len(self.item_cols))
        self.assertIn("Communality (h²)", df_loadings.columns)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestSurveySampleSizeFPCCalculations(unittest.TestCase):
    """Verifies Cochran's survey sample size with Finite Population Correction."""

    def test_sample_size_fpc_exact(self):
        # Infinite population: 95% CI, 5% margin of error, p=0.5 -> n0 = 385
        res_inf = calculate_survey_sample_size_fpc(population_size_n=None, margin_of_error=0.05)
        self.assertEqual(res_inf["n_infinite_n0"], 385)
        self.assertEqual(res_inf["n_evaluable_fpc"], 385)
        
        # Finite population N = 1000 -> FPC reduces required sample to ~278
        res_fin = calculate_survey_sample_size_fpc(population_size_n=1000, margin_of_error=0.05, response_rate=0.35)
        self.assertLess(res_fin["n_evaluable_fpc"], 300)
        self.assertGreater(res_fin["n_evaluable_fpc"], 250)
        self.assertGreater(res_fin["fpc_savings_pct"], 20.0)
        
        # Response rate inflation: 278 / 0.35 -> ~795 invitations
        self.assertEqual(res_fin["n_invitations_required"], math.ceil(res_fin["n_evaluable_fpc"] / 0.35))
        
        # Sensitivity table
        self.assertEqual(len(res_fin["sensitivity_table"]), 10)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestSurveyWeightingKishCalculations(unittest.TestCase):
    """Verifies post-stratification cell weights, Kish design effect, and Neff."""

    def setUp(self):
        self.df = generate_synthetic_survey_data("survey_weighting", n_samples=300, seed=42)
        self.target_pop = {
            "18-34 (Young Adult)": 0.30,
            "35-64 (Working Age)": 0.50,
            "65+ (Older Adult)": 0.20
        }

    def test_weighting_and_kish_metrics(self):
        res = calculate_survey_weights_kish(
            df_responses=self.df,
            group_col="Age_Group",
            target_pop_props=self.target_pop,
            score_col="Satisfaction_Score_1_to_10"
        )
        
        # Kish Neff must be <= N original due to unequal weighting
        self.assertLess(res["n_effective_kish"], res["n_original"])
        self.assertGreater(res["design_effect_deff"], 1.0) # Variance inflation > 1.0
        self.assertLess(res["efficiency_pct"], 100.0)
        
        # Underrepresented young adults must receive weights > 1.0
        df_w = res["weight_table"]
        young_row = df_w[df_w["Demographic_Group"].str.contains("18-34")].iloc[0]
        self.assertGreater(young_row["Cell_Weight_w"], 1.5)
        
        # Overrepresented older adults must receive weights < 1.0
        older_row = df_w[df_w["Demographic_Group"].str.contains("65+")].iloc[0]
        self.assertLess(older_row["Cell_Weight_w"], 0.8)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestSyntheticDataGenerators(unittest.TestCase):
    """Verifies that synthetic data generation succeeds for all 7 survey methods."""

    def test_all_seven_method_generators(self):
        for method_key in SURVEY_METHODS.keys():
            df = generate_synthetic_survey_data(method_key, n_samples=50, seed=42)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)


class TestSurveyASTExpanderCompliance(unittest.TestCase):
    """Verifies strict AST expander compliance: all expanders must be collapsed (expanded=False)."""

    def test_survey_page_expander_compliance(self):
        file_path = os.path.join(PROJECT_ROOT, "pages", "5_📋_Survey_and_Questionnaire_Analysis.py")
        self.assertTrue(os.path.exists(file_path))
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("expanded" + "=True", content)

    def test_survey_widgets_expander_compliance(self):
        file_path = os.path.join(PROJECT_ROOT, "src", "ui_components", "survey_widgets.py")
        self.assertTrue(os.path.exists(file_path))
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("expanded" + "=True", content)


if __name__ == "__main__":
    import math
    unittest.main()

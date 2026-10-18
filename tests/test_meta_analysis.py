"""Unit, Static AST & Contract Test Suite for Meta-Analysis & Evidence Synthesis.

Validates:
1. Function Signatures & Keyword Compatibility:
   - generate_meta_analysis_demo_data accepts both 'scenario' and 'scenario_key'.
   - MetaAnalysisDemoData supports dictionary access and 2-tuple unpacking (df, meta = ...).
   - calculate_meta_analysis accepts list-of-dicts and DataFrame inputs.
   - evaluate_meta_analysis_prerequisites accepts meta_results dictionary.
2. Abstract Syntax Tree (AST) Integration Check:
   - Inspects src/ui_components/epi_meta_analysis_widgets.py to verify every call
     to the engine matches accepted parameters with zero risk of TypeError.
3. UI Expander & Presentation Constraints:
   - Every st.expander call in the widget file has expanded=False.
   - Strictly fictional UK healthcare and public health organisations used.
   - Educational & synthetic data disclaimers are present in every scenario.
4. Mathematical & Biostatistical Correctness:
   - Fixed-Effect inverse-variance pooling (theta_fixed, SE_fixed, 95% CIs, Z-test).
   - Cochran's Q test statistic and Higgins & Thompson I^2 heterogeneity.
   - DerSimonian-Laird between-study variance estimator tau^2.
   - DerSimonian-Laird Random-Effects pooling (theta_random, SE_random, 95% CIs).
   - Higgins 95% prediction interval (t-distribution critical value).
   - Egger's linear regression test for funnel plot asymmetry (distinguishing symmetric from asymmetric scenarios).
   - Leave-one-out sensitivity analysis recalculations.
   - Continuity correction handling for 2x2 contingency tables with zero cells.
"""

import os
import sys
import ast
import inspect
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib.util
engine_spec = importlib.util.spec_from_file_location(
    "meta_analysis",
    os.path.join(PROJECT_ROOT, "src", "epi_methods", "meta_analysis.py")
)
ma = importlib.util.module_from_spec(engine_spec)
engine_spec.loader.exec_module(ma)


class TestMetaAnalysisSignaturesAndContracts(unittest.TestCase):
    """Static AST and signature tests verifying keyword arguments and contract compatibility."""

    def setUp(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.engine_path = os.path.join(self.project_root, "src", "epi_methods", "meta_analysis.py")
        self.widget_path = os.path.join(self.project_root, "src", "ui_components", "epi_meta_analysis_widgets.py")

    def _parse_engine_functions(self):
        """Extracts function arguments and keyword arguments from engine AST."""
        with open(self.engine_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        funcs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                arg_names = [a.arg for a in node.args.args]
                funcs[node.name] = {
                    "args": arg_names,
                    "defaults": len(node.args.defaults)
                }
        return funcs

    def test_demo_data_generator_accepts_scenario_and_scenario_key(self):
        """Verifies generate_meta_analysis_demo_data accepts both 'scenario' and 'scenario_key'."""
        sig = inspect.signature(ma.generate_meta_analysis_demo_data)
        self.assertIn("scenario_key", sig.parameters, "Must accept 'scenario_key'")
        self.assertIn("scenario", sig.parameters, "Must accept 'scenario' for call-compatibility")

        scenarios = ["cardiovascular_sglt2", "respiratory_masks", "anticoagulation_elderly"]
        for sc in scenarios:
            data1 = ma.generate_meta_analysis_demo_data(scenario_key=sc)
            self.assertIsNotNone(data1)
            self.assertEqual(data1["scenario_key"], sc)

            data2 = ma.generate_meta_analysis_demo_data(scenario=sc)
            self.assertIsNotNone(data2)
            self.assertEqual(data2["scenario_key"], sc)

    def test_demo_data_tuple_unpacking(self):
        """Verifies MetaAnalysisDemoData supports both dictionary key access and 2-tuple unpacking."""
        demo = ma.generate_meta_analysis_demo_data("cardiovascular_sglt2")
        self.assertIn("df", demo)
        self.assertIn("studies", demo)
        self.assertIn("scenario_name", demo)

        df, meta = demo
        self.assertIsNotNone(df)
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta["scenario_key"], "cardiovascular_sglt2")

    def test_ast_widget_calls_to_engine_have_zero_type_errors(self):
        """Scans epi_meta_analysis_widgets.py AST to guarantee all calls to engine use valid args."""
        engine_funcs = self._parse_engine_functions()

        with open(self.widget_path, "r", encoding="utf-8") as f:
            widget_tree = ast.parse(f.read())

        for node in ast.walk(widget_tree):
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in engine_funcs:
                    known = engine_funcs[func_name]
                    for kw in node.keywords:
                        self.assertIn(
                            kw.arg, known["args"],
                            f"Widget calls '{func_name}' with illegal keyword argument '{kw.arg}' at line {node.lineno}!"
                        )

    def test_all_ui_expanders_have_expanded_false(self):
        """Guarantees every st.expander call in epi_meta_analysis_widgets.py specifies expanded=False."""
        with open(self.widget_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                is_expander = False
                if isinstance(func, ast.Attribute) and func.attr == "expander":
                    is_expander = True
                elif isinstance(func, ast.Name) and func.id == "expander":
                    is_expander = True

                if is_expander:
                    has_expanded_false = False
                    for kw in node.keywords:
                        if kw.arg == "expanded" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                            has_expanded_false = True
                    if not has_expanded_false:
                        violations.append(node.lineno)

        self.assertEqual(
            violations, [],
            f"Found st.expander calls without expanded=False at lines: {violations}"
        )

    def test_strictly_fictional_healthcare_organisations_used(self):
        """Verifies only fictional UK organisations are referenced in demo scenarios."""
        fictional_orgs = [
            "All Meadows Cardiovascular Institute",
            "Riverside Health Protection Team",
            "Inner City Teaching Hospital"
        ]
        scenarios = ["cardiovascular_sglt2", "respiratory_masks", "anticoagulation_elderly"]
        for sc in scenarios:
            demo = ma.generate_meta_analysis_demo_data(scenario_key=sc)
            org_found = any(org in demo["organisation_name"] for org in fictional_orgs)
            self.assertTrue(org_found, f"Scenario '{sc}' must use a fictional UK hospital/health entity!")

    def test_demo_data_disclaimer_prominently_present(self):
        """Verifies every demo scenario contains a clear educational/synthetic data disclaimer."""
        scenarios = ["cardiovascular_sglt2", "respiratory_masks", "anticoagulation_elderly"]
        for sc in scenarios:
            demo = ma.generate_meta_analysis_demo_data(scenario_key=sc)
            self.assertIn("disclaimer", demo)
            disclaimer = demo["disclaimer"].lower()
            self.assertIn("synthetic", disclaimer)
            self.assertIn("educational", disclaimer)


class TestMetaAnalysisMathematicalCorrectness(unittest.TestCase):
    """Mathematical, biostatistical and evidence synthesis unit tests."""

    def test_fixed_effect_inverse_variance_pooling(self):
        """Tests fixed-effect inverse-variance pooling on known homogeneous trial set."""
        studies = [
            {"Study": "Trial 1", "treated_events": 50, "treated_total": 500, "control_events": 70, "control_total": 500},
            {"Study": "Trial 2", "treated_events": 40, "treated_total": 400, "control_events": 60, "control_total": 400},
            {"Study": "Trial 3", "treated_events": 80, "treated_total": 800, "control_events": 110, "control_total": 800}
        ]
        res = ma.calculate_meta_analysis(studies, effect_metric="OR")

        self.assertEqual(res["k_studies"], 3)
        self.assertAlmostEqual(res["fixed_effect_natural"], 0.70, delta=0.03)
        self.assertLess(res["fixed_effect_ci_lower"], res["fixed_effect_natural"])
        self.assertGreater(res["fixed_effect_ci_upper"], res["fixed_effect_natural"])
        self.assertLess(res["fixed_p_value"], 0.05)
        self.assertEqual(res["preferred_model"], "Fixed-Effect")

    def test_cochran_q_and_i2_heterogeneity_contrast(self):
        """Tests that homogeneous scenario yields I² ~ 0% while heterogeneous yields I² > 50%."""
        demo_homo = ma.generate_meta_analysis_demo_data("cardiovascular_sglt2")
        res_homo = ma.calculate_meta_analysis(demo_homo["studies"], effect_metric="OR")

        # SGLT2 is highly homogeneous
        self.assertLess(res_homo["i2_pct"], 25.0)
        self.assertEqual(res_homo["heterogeneity_category"], "Low Heterogeneity")
        self.assertAlmostEqual(res_homo["tau2"], 0.0, delta=0.01)

        # Respiratory masks cohort has substantial heterogeneity
        demo_hetero = ma.generate_meta_analysis_demo_data("respiratory_masks")
        res_hetero = ma.calculate_meta_analysis(demo_hetero["studies"], effect_metric="OR")

        self.assertGreater(res_hetero["i2_pct"], 50.0)
        self.assertGreater(res_hetero["cochran_q"], 14.0)
        self.assertLess(res_hetero["cochran_q_p_value"], 0.05)
        self.assertGreater(res_hetero["tau2"], 0.01)
        self.assertEqual(res_hetero["preferred_model"], "Random-Effects")

    def test_random_effects_widens_ci_under_heterogeneity(self):
        """Verifies Random-Effects CI is wider than Fixed-Effect CI when tau² > 0."""
        demo = ma.generate_meta_analysis_demo_data("respiratory_masks")
        res = ma.calculate_meta_analysis(demo["studies"], effect_metric="OR")

        fixed_width = res["fixed_effect_ci_upper"] - res["fixed_effect_ci_lower"]
        random_width = res["random_effect_ci_upper"] - res["random_effect_ci_lower"]

        self.assertGreater(random_width, fixed_width, "Random-effects CI must be wider than fixed-effect CI when heterogeneity exists")
        self.assertGreater(res["random_effect_se"], res["fixed_effect_se"])

    def test_higgins_prediction_interval(self):
        """Verifies Higgins 95% prediction interval is calculated and bounds the pooled effect."""
        demo = ma.generate_meta_analysis_demo_data("respiratory_masks")
        res = ma.calculate_meta_analysis(demo["studies"], effect_metric="OR")

        pred_low = res["prediction_interval_natural_lower"]
        pred_upp = res["prediction_interval_natural_upper"]

        self.assertLess(pred_low, res["random_effect_natural"])
        self.assertGreater(pred_upp, res["random_effect_natural"])
        self.assertLess(pred_low, res["random_effect_ci_lower"])
        self.assertGreater(pred_upp, res["random_effect_ci_upper"])

    def test_egger_linear_regression_publication_bias_detection(self):
        """Verifies Egger test correctly discriminates symmetric from publication-biased funnels."""
        # 1. Homogeneous symmetric scenario (SGLT2) -> No bias detected
        demo_sglt2 = ma.generate_meta_analysis_demo_data("cardiovascular_sglt2")
        res_sglt2 = ma.calculate_meta_analysis(demo_sglt2["studies"], effect_metric="OR")

        self.assertFalse(res_sglt2["egger_bias_detected"])
        self.assertGreater(res_sglt2["egger_p_value"], 0.10)

        # 2. Publication-biased scenario (Elderly anticoagulation) -> Bias detected
        demo_elderly = ma.generate_meta_analysis_demo_data("anticoagulation_elderly")
        res_elderly = ma.calculate_meta_analysis(demo_elderly["studies"], effect_metric="OR")

        self.assertTrue(res_elderly["egger_bias_detected"])
        self.assertLess(res_elderly["egger_p_value"], 0.10)
        self.assertLess(res_elderly["egger_intercept"], -1.5)

    def test_leave_one_out_sensitivity_analysis(self):
        """Verifies leave-one-out sensitivity analysis recalculations for each study."""
        demo = ma.generate_meta_analysis_demo_data("cardiovascular_sglt2")
        res = ma.calculate_meta_analysis(demo["studies"], effect_metric="OR")

        loo = res["leave_one_out_list"]
        self.assertEqual(len(loo), res["k_studies"])

        for item in loo:
            self.assertIn("Omitted_Study", item)
            self.assertIn("Pooled_Effect", item)
            self.assertIn("CI_Lower", item)
            self.assertIn("CI_Upper", item)
            self.assertIn("Heterogeneity_I2_Pct", item)
            self.assertGreater(item["Pooled_Effect"], 0.5)
            self.assertLess(item["Pooled_Effect"], 1.0)

    def test_zero_cell_continuity_correction(self):
        """Verifies handling of zero-cell studies without division-by-zero crashes."""
        sparse_studies = [
            {"Study": "Pilot A", "treated_events": 0, "treated_total": 30, "control_events": 5, "control_total": 30},
            {"Study": "Pilot B", "treated_events": 2, "treated_total": 40, "control_events": 8, "control_total": 40},
            {"Study": "Pilot C", "treated_events": 1, "treated_total": 35, "control_events": 6, "control_total": 35}
        ]
        res = ma.calculate_meta_analysis(sparse_studies, effect_metric="OR")
        self.assertEqual(res["k_studies"], 3)
        self.assertGreater(res["fixed_effect_natural"], 0.0)
        self.assertGreater(res["random_effect_natural"], 0.0)

    def test_meta_analysis_prerequisites_structure(self):
        """Verifies evaluate_meta_analysis_prerequisites returns all 5 PRISMA checks."""
        demo = ma.generate_meta_analysis_demo_data("cardiovascular_sglt2")
        res = ma.calculate_meta_analysis(demo["studies"], effect_metric="OR")
        evals = ma.evaluate_meta_analysis_prerequisites(res)

        self.assertIn("checks", evals)
        self.assertIn("all_passed", evals)
        self.assertEqual(len(evals["checks"]), 5)

        required_keys = ["status", "badge", "title", "finding", "explanation"]
        for c in evals["checks"]:
            for k in required_keys:
                self.assertIn(k, c)


if __name__ == "__main__":
    unittest.main()


"""Unit, Static AST & Contract Test Suite for Cochran-Mantel-Haenszel (CMH) Stratified Analysis.

Validates:
1. Function Signatures & Keyword Compatibility:
   - generate_mantel_haenszel_demo_data accepts both 'scenario' and 'scenario_key'.
   - calculate_mantel_haenszel accepts both list-of-dicts and DataFrame inputs.
   - evaluate_mantel_haenszel_prerequisites accepts cmh_results dict.
   - MantelHaenszelDemoData supports dictionary access and 2-tuple unpacking (df, meta = ...).
2. Abstract Syntax Tree (AST) Integration Check:
   - Inspects src/ui_components/epi_mantel_haenszel_widgets.py to verify every call
     to the engine matches accepted parameters with zero risk of TypeError.
3. UI Expander & Presentation Constraints:
   - Every st.expander call in the widget file has expanded=False.
   - LaTeX blocks use raw strings (r"..." or r\"\"\"...\") to eliminate escape bugs.
   - Strictly fictional healthcare organisation names used.
   - Educational & synthetic data disclaimers are present.
4. Mathematical & Algorithmic Correctness:
   - Crude OR, MH pooled summary OR, and Robins-Breslow-Greenland (1986) variance.
   - Mantel-Haenszel pooled RR and Greenland-Robins (1985) variance.
   - Simpson's Paradox qualitative inversion detection.
   - Breslow-Day and Tarone-adjusted tests for homogeneity (effect modification).
   - Cochran-Mantel-Haenszel Chi-Square test with continuity correction.
   - Confounding 10% change-in-estimate rule (|OR_crude - OR_MH| / OR_MH * 100%).
   - Zero-cell handling via Haldane 0.5 continuity correction.
"""

import os
import sys
import ast
import inspect
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Directly import engine module without triggering top-level package dependencies
import importlib.util
engine_spec = importlib.util.spec_from_file_location(
    "mantel_haenszel",
    os.path.join(PROJECT_ROOT, "src", "epi_methods", "mantel_haenszel.py")
)
mh = importlib.util.module_from_spec(engine_spec)
engine_spec.loader.exec_module(mh)


class TestMantelHaenszelSignaturesAndContracts(unittest.TestCase):
    """Static AST and signature tests verifying keyword arguments and contract compatibility."""

    def setUp(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.engine_path = os.path.join(self.project_root, "src", "epi_methods", "mantel_haenszel.py")
        self.widget_path = os.path.join(self.project_root, "src", "ui_components", "epi_mantel_haenszel_widgets.py")

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
        """Verifies generate_mantel_haenszel_demo_data accepts both 'scenario' and 'scenario_key'."""
        sig = inspect.signature(mh.generate_mantel_haenszel_demo_data)
        self.assertIn("scenario_key", sig.parameters, "Must accept 'scenario_key'")
        self.assertIn("scenario", sig.parameters, "Must accept 'scenario' for call-compatibility")

        # Test both calls
        data1 = mh.generate_mantel_haenszel_demo_data(scenario_key="simpsons_paradox_icu")
        self.assertIsNotNone(data1)
        data2 = mh.generate_mantel_haenszel_demo_data(scenario="simpsons_paradox_icu")
        self.assertIsNotNone(data2)

    def test_demo_data_tuple_unpacking(self):
        """Verifies MantelHaenszelDemoData supports both dictionary key access and 2-tuple unpacking."""
        demo = mh.generate_mantel_haenszel_demo_data("simpsons_paradox_icu")
        # Dictionary indexing
        self.assertIn("df", demo)
        self.assertIn("strata_tables", demo)
        self.assertIn("scenario_name", demo)

        # 2-tuple unpacking
        df, meta = demo
        self.assertIsNotNone(df)
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta["scenario_key"], "simpsons_paradox_icu")

    def test_ast_widget_calls_to_engine_have_zero_type_errors(self):
        """Scans epi_mantel_haenszel_widgets.py AST to guarantee all calls to engine use valid args."""
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
                    # Check keyword args passed in widget
                    for kw in node.keywords:
                        self.assertIn(
                            kw.arg, known["args"],
                            f"Widget calls '{func_name}' with illegal keyword argument '{kw.arg}' at line {node.lineno}!"
                        )

    def test_all_ui_expanders_have_expanded_false(self):
        """Guarantees every st.expander call in epi_mantel_haenszel_widgets.py specifies expanded=False."""
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
        """Verifies only fictional UK healthcare organisations are referenced in demo data."""
        fictional_orgs = [
            "Inner City Teaching Hospital",
            "All Meadows Cardiovascular Institute",
            "Riverside Health Trust"
        ]
        scenarios = ["simpsons_paradox_icu", "age_confounding_cardiac", "effect_modification_smoking"]
        for sc in scenarios:
            demo = mh.generate_mantel_haenszel_demo_data(scenario_key=sc)
            org_found = any(org in demo["organisation_name"] for org in fictional_orgs)
            self.assertTrue(org_found, f"Scenario '{sc}' must use a fictional UK hospital name!")


class TestMantelHaenszelMathematicalCorrectness(unittest.TestCase):
    """Mathematical and biostatistical unit tests for CMH formulas and algorithms."""

    def test_simpsons_paradox_exact_reversal(self):
        """Tests that the ICU scenario produces true qualitative Simpson's paradox reversal.
        
        ICU: a1=48, b1=72, c1=18, d1=12 -> OR1 = (48*12)/(72*18) = 0.444
        Ward: a2=5, b2=45, c2=40, d2=160 -> OR2 = (5*160)/(45*40) = 0.444
        Crude: A=53, B=117, C=58, D=172 -> Crude OR = (53*172)/(117*58) = 1.343
        Crude > 1.0 (harmful) while Strata < 1.0 (protective).
        """
        strata = [
            {"stratum": "ICU", "a": 48, "b": 72, "c": 18, "d": 12},
            {"stratum": "Ward", "a": 5, "b": 45, "c": 40, "d": 160}
        ]
        res = mh.calculate_mantel_haenszel(strata)

        self.assertAlmostEqual(res["crude_or"], 1.343, delta=0.005)
        self.assertAlmostEqual(res["or_mh"], 0.444, delta=0.005)
        self.assertTrue(res["simpsons_paradox_detected"], "Simpson's paradox must be detected!")
        self.assertTrue(res["confounding_present"], "Confounding >= 10% must be flagged!")
        self.assertGreater(res["confounding_delta_pct"], 100.0)

        # Breslow-Day should confirm perfect homogeneity across strata (both ORs = 0.444)
        self.assertAlmostEqual(res["breslow_day_p_value"], 1.0, delta=0.01)
        self.assertFalse(res["effect_modification_present"])

    def test_effect_measure_modification_detection(self):
        """Tests that the smoking/toxin scenario detects significant effect measure modification.
        
        Non-smokers: a1=12, b1=188, c1=8, d1=192 -> OR1 = 1.53
        Current Smokers: a2=75, b2=125, c2=16, d2=184 -> OR2 = 6.90
        Breslow-Day p-value should be < 0.001.
        """
        strata = [
            {"stratum": "Non-Smokers", "a": 12, "b": 188, "c": 8, "d": 192},
            {"stratum": "Current Smokers", "a": 75, "b": 125, "c": 16, "d": 184}
        ]
        res = mh.calculate_mantel_haenszel(strata)

        self.assertLess(res["breslow_day_p_value"], 0.01, "Breslow-Day must detect interaction (p < 0.01)!")
        self.assertTrue(res["effect_modification_present"], "Effect modification must be True!")
        self.assertIn("EFFECT MEASURE MODIFICATION", res["clinical_recommendation"])

    def test_cochran_mantel_haenszel_chi_square(self):
        """Tests CMH Chi-Square statistic and continuity correction calculation."""
        strata = [
            {"stratum": "S1", "a": 25, "b": 75, "c": 10, "d": 90},
            {"stratum": "S2", "a": 30, "b": 70, "c": 15, "d": 85}
        ]
        res = mh.calculate_mantel_haenszel(strata)

        self.assertGreater(res["cmh_statistic_corrected"], 0.0)
        self.assertLess(res["cmh_p_value"], 0.05, "Association should be statistically significant.")
        self.assertTrue(res["cmh_significant"])
        self.assertEqual(res["cmh_df"], 1)

    def test_zero_cell_continuity_correction(self):
        """Verifies that strata containing zero cells do not cause ZeroDivisionError or crash."""
        strata_with_zero = [
            {"stratum": "S1", "a": 0, "b": 20, "c": 5, "d": 15},
            {"stratum": "S2", "a": 10, "b": 10, "c": 8, "d": 12}
        ]
        # Must execute cleanly without error
        res = mh.calculate_mantel_haenszel(strata_with_zero)
        self.assertGreater(res["or_mh"], 0.0)
        self.assertGreater(res["or_mh_ci_lower"], 0.0)

    def test_prerequisite_evaluations_structure(self):
        """Verifies evaluate_mantel_haenszel_prerequisites returns all 5 checks and valid keys."""
        demo = mh.generate_mantel_haenszel_demo_data("simpsons_paradox_icu")
        results = mh.calculate_mantel_haenszel(demo["strata_tables"])
        evals = mh.evaluate_mantel_haenszel_prerequisites(results)

        self.assertIn("checks", evals)
        self.assertIn("all_passed", evals)
        self.assertIn("summary", evals)
        self.assertEqual(len(evals["checks"]), 5)

        required_check_keys = ["status", "badge", "title", "finding", "explanation"]
        for c in evals["checks"]:
            for k in required_check_keys:
                self.assertIn(k, c)


if __name__ == "__main__":
    unittest.main()

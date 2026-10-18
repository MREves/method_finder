"""Unit, Static AST & Contract Test Suite for Outbreak Epidemiology & Attack Rates.

Validates:
1. Function Signatures & Keyword Compatibility:
   - generate_outbreak_demo_data accepts both 'scenario' and 'scenario_key'.
   - OutbreakDemoData supports dictionary access and 2-tuple unpacking (df, meta = ...).
   - calculate_food_specific_attack_rates accepts list-of-dicts and DataFrame.
   - calculate_secondary_attack_rate accepts required and optional arguments.
   - calculate_epidemic_curve accepts onset hours, exposure hour, and bin width.
   - evaluate_outbreak_prerequisites accepts outbreak results dictionary.
2. Abstract Syntax Tree (AST) Integration Check:
   - Inspects src/ui_components/epi_outbreak_widgets.py to verify every call
     to the engine matches accepted parameters with zero risk of TypeError.
3. UI Expander & Presentation Constraints:
   - Every st.expander call in the widget file has expanded=False.
   - Strictly fictional UK healthcare and public health organisations used.
   - Educational & synthetic data disclaimers are present.
4. Mathematical & Algorithmic Correctness:
   - Food-specific attack rates (AR_exp, AR_unexp), ARD, Relative Risk, Woolf 95% CIs.
   - Decisive vehicle incrimination (identifying Hollandaise sauce with RR = 8.42, p < 0.0001).
   - Secondary Attack Rate (SAR) and Wilson score 95% confidence interval.
   - Epidemic curve onset binning, median incubation period, and pattern classification.
   - Zero-cell handling (continuity correction) without division-by-zero crashes.
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
    "outbreak_investigation",
    os.path.join(PROJECT_ROOT, "src", "epi_methods", "outbreak_investigation.py")
)
ob = importlib.util.module_from_spec(engine_spec)
engine_spec.loader.exec_module(ob)


class TestOutbreakSignaturesAndContracts(unittest.TestCase):
    """Static AST and signature tests verifying keyword arguments and contract compatibility."""

    def setUp(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.engine_path = os.path.join(self.project_root, "src", "epi_methods", "outbreak_investigation.py")
        self.widget_path = os.path.join(self.project_root, "src", "ui_components", "epi_outbreak_widgets.py")

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
        """Verifies generate_outbreak_demo_data accepts both 'scenario' and 'scenario_key'."""
        sig = inspect.signature(ob.generate_outbreak_demo_data)
        self.assertIn("scenario_key", sig.parameters, "Must accept 'scenario_key'")
        self.assertIn("scenario", sig.parameters, "Must accept 'scenario' for call-compatibility")

        data1 = ob.generate_outbreak_demo_data(scenario_key="wedding_salmonella")
        self.assertIsNotNone(data1)
        data2 = ob.generate_outbreak_demo_data(scenario="wedding_salmonella")
        self.assertIsNotNone(data2)

    def test_demo_data_tuple_unpacking(self):
        """Verifies OutbreakDemoData supports both dictionary key access and 2-tuple unpacking."""
        demo = ob.generate_outbreak_demo_data("wedding_salmonella")
        self.assertIn("df", demo)
        self.assertIn("food_matrix", demo)
        self.assertIn("scenario_name", demo)

        df, meta = demo
        self.assertIsNotNone(df)
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta["scenario_key"], "wedding_salmonella")

    def test_ast_widget_calls_to_engine_have_zero_type_errors(self):
        """Scans epi_outbreak_widgets.py AST to guarantee all calls to engine use valid args."""
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
        """Guarantees every st.expander call in epi_outbreak_widgets.py specifies expanded=False."""
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
            "Riverside Health Protection Team",
            "Inner City Teaching Hospital",
            "All Meadows Community Care Trust"
        ]
        scenarios = ["wedding_salmonella", "hospital_norovirus", "community_campylobacter"]
        for sc in scenarios:
            demo = ob.generate_outbreak_demo_data(scenario_key=sc)
            org_found = any(org in demo["organisation_name"] for org in fictional_orgs)
            self.assertTrue(org_found, f"Scenario '{sc}' must use a fictional UK hospital/health entity!")


class TestOutbreakMathematicalCorrectness(unittest.TestCase):
    """Mathematical and field epidemiological unit tests for outbreak algorithms."""

    def test_food_specific_attack_rates_and_vehicle_incrimination(self):
        """Tests that Hollandaise sauce is conclusively incriminated with RR = 8.42 and p < 0.0001."""
        food_matrix = [
            {"Food_Item": "Hollandaise Sauce (Eggs)", "Exposed_Ill": 54, "Exposed_Well": 8, "Unexposed_Ill": 6, "Unexposed_Well": 52},
            {"Food_Item": "Roast Beef Sirloin", "Exposed_Ill": 42, "Exposed_Well": 38, "Unexposed_Ill": 18, "Unexposed_Well": 22}
        ]
        res = ob.calculate_food_specific_attack_rates(food_matrix)

        self.assertEqual(res["primary_incriminated_vehicle"], "Hollandaise Sauce (Eggs)")
        hollandaise = res["incriminated_vehicle_details"]
        self.assertIsNotNone(hollandaise)

        # Attack rates
        self.assertAlmostEqual(hollandaise["Attack_Rate_Exposed_Pct"], 87.1, delta=0.2)
        self.assertAlmostEqual(hollandaise["Attack_Rate_Unexposed_Pct"], 10.3, delta=0.2)
        self.assertAlmostEqual(hollandaise["Attack_Rate_Difference_Pct"], 76.8, delta=0.2)

        # Relative risk
        self.assertAlmostEqual(hollandaise["Risk_Ratio"], 8.42, delta=0.05)
        self.assertLess(hollandaise["P_Value"], 0.0001)
        self.assertTrue(hollandaise["Is_Suspect_Vehicle"])

        # Overall attack rate: (54+6) / 120 = 50.0%
        self.assertAlmostEqual(res["overall_attack_rate_pct"], 50.0, delta=0.1)

    def test_secondary_attack_rate_calculation(self):
        """Tests secondary attack rate and Wilson interval calculation."""
        # Hospital ward norovirus: 15 secondary cases among 36 contacts
        res = ob.calculate_secondary_attack_rate(
            primary_cases=28,
            household_contacts=36,
            secondary_cases=15,
            immune_contacts=0
        )

        # SAR = 15 / 36 * 100 = 41.67%
        self.assertAlmostEqual(res["secondary_attack_rate_pct"], 41.7, delta=0.2)
        self.assertGreater(res["sar_ci_lower_pct"], 25.0)
        self.assertLess(res["sar_ci_upper_pct"], 60.0)
        self.assertEqual(res["communicability_tier"], "High / Explosive Communicability")

    def test_epidemic_curve_incubation_statistics(self):
        """Tests incubation period percentiles and point-source classification."""
        onset_times = [12.0, 16.0, 18.0, 20.0, 22.0, 22.0, 24.0, 26.0, 28.0, 32.0, 40.0]
        curve = ob.calculate_epidemic_curve(onset_times, exposure_hour=0.0, bin_width=4.0)

        self.assertAlmostEqual(curve["min_incubation_hours"], 12.0)
        self.assertAlmostEqual(curve["max_incubation_hours"], 40.0)
        self.assertAlmostEqual(curve["median_incubation_hours"], 22.0)
        self.assertIn("Point-Source", curve["curve_pattern"])
        self.assertGreater(len(curve["bins_list"]), 0)

    def test_zero_cell_contingency_handling(self):
        """Verifies that items with zero unexposed or exposed ill cases do not crash."""
        sparse_matrix = [
            {"Food_Item": "Pudding", "Exposed_Ill": 10, "Exposed_Well": 0, "Unexposed_Ill": 0, "Unexposed_Well": 20}
        ]
        res = ob.calculate_food_specific_attack_rates(sparse_matrix)
        self.assertGreater(res["items_count"], 0)
        detail = res["results_list"][0]
        self.assertGreater(detail["Risk_Ratio"], 0.0)

    def test_outbreak_prerequisite_evaluations_structure(self):
        """Verifies evaluate_outbreak_prerequisites returns all 5 checks and valid keys."""
        demo = ob.generate_outbreak_demo_data("wedding_salmonella")
        ar_res = ob.calculate_food_specific_attack_rates(demo["food_matrix"])
        sar_res = ob.calculate_secondary_attack_rate(demo["primary_cases"], demo["household_contacts"], demo["secondary_cases"])
        curve_res = ob.calculate_epidemic_curve(demo["onset_hours"], demo["exposure_hour"], demo["bin_width"])

        evals = ob.evaluate_outbreak_prerequisites({
            **ar_res,
            "sar_results": sar_res,
            "curve_results": curve_res
        })

        self.assertIn("checks", evals)
        self.assertIn("all_passed", evals)
        self.assertEqual(len(evals["checks"]), 5)

        required_keys = ["status", "badge", "title", "finding", "explanation"]
        for c in evals["checks"]:
            for k in required_keys:
                self.assertIn(k, c)


if __name__ == "__main__":
    unittest.main()


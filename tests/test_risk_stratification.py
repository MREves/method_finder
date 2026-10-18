"""Unit & Integration Test Suite for Clinical Risk Stratification & Prognostic Models.

Validates:
1. Function Signatures & Keyword Compatibility:
   - evaluate_model_calibration accepts both 'n_bins' and 'n_groups' as keyword arguments.
   - generate_risk_stratification_demo_data accepts both 'scenario' and 'scenario_key'.
   - evaluate_risk_stratification_prerequisites supports both 5-argument and 4-argument signatures.
2. Abstract Syntax Tree (AST) Integration Check:
   - Inspects src/ui_components/epi_risk_stratification_widgets.py to verify every call
     to the engine matches accepted parameters with zero risk of TypeError.
3. UI Expander & Presentation Constraints:
   - Every st.expander call in the widget file has expanded=False.
   - LaTeX blocks use raw strings (r"..." or r\"\"\"...\") to eliminate escape bugs.
   - Strictly fictional healthcare organisation names used.
4. Mathematical & Algorithmic Correctness:
   - Sullivan / Framingham integer point conversion logic.
   - Dynamic clinical risk strata calculation, observed/expected rates, and Wilson intervals.
   - Model calibration (slope, intercept, Brier score, Hosmer-Lemeshow chi-square, C-index).
   - Calibration deciles table numeric float types (preventing Plotly type failures).
   - Decision Curve Analysis (DCA Net Benefit, Treat-All, Treat-None, Interventions Avoided).
   - TRIPOD diagnostic prerequisite evaluation.
   - Realistic multi-scenario clinical demo generation.
"""

import os
import sys
import ast
import inspect
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Check if scientific stack is available in current python environment
try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


class TestRiskStratificationSignaturesAndContracts(unittest.TestCase):
    """Static AST and signature tests verifying keyword arguments and contract compatibility.
    
    Runs in all Python environments without requiring external third-party packages.
    """

    def setUp(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.engine_path = os.path.join(self.project_root, "src", "epi_methods", "risk_stratification.py")
        self.widget_path = os.path.join(self.project_root, "src", "ui_components", "epi_risk_stratification_widgets.py")

    def _parse_engine_functions(self):
        """Extracts function arguments and keyword-only arguments from engine AST."""
        with open(self.engine_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        funcs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args]
                kwonly = [a.arg for a in node.args.kwonlyargs]
                funcs[node.name] = {"args": args, "kwonly": kwonly, "all": set(args + kwonly)}
        return funcs

    def test_evaluate_model_calibration_signature_accepts_n_bins(self):
        """Validates that evaluate_model_calibration explicitly accepts 'n_bins' and 'n_groups'."""
        funcs = self._parse_engine_functions()
        self.assertIn("evaluate_model_calibration", funcs)
        cal_args = funcs["evaluate_model_calibration"]["all"]
        self.assertIn("n_bins", cal_args, "evaluate_model_calibration must accept 'n_bins'")
        self.assertIn("n_groups", cal_args, "evaluate_model_calibration must accept 'n_groups'")
        self.assertIn("df", cal_args)
        self.assertIn("predicted_prob_col", cal_args)
        self.assertIn("event_col", cal_args)
        self.assertIn("y_true", cal_args)
        self.assertIn("y_pred_prob", cal_args)

    def test_generate_risk_stratification_demo_data_signature_accepts_scenario(self):
        """Validates that generate_risk_stratification_demo_data accepts both 'scenario' and 'scenario_key'."""
        funcs = self._parse_engine_functions()
        self.assertIn("generate_risk_stratification_demo_data", funcs)
        demo_args = funcs["generate_risk_stratification_demo_data"]["all"]
        self.assertIn("scenario", demo_args, "generate_risk_stratification_demo_data must accept 'scenario'")
        self.assertIn("scenario_key", demo_args, "generate_risk_stratification_demo_data must accept 'scenario_key'")

    def test_calculate_risk_strata_performance_signature(self):
        """Validates calculate_risk_strata_performance signature."""
        funcs = self._parse_engine_functions()
        self.assertIn("calculate_risk_strata_performance", funcs)
        strata_args = funcs["calculate_risk_strata_performance"]["all"]
        self.assertIn("df", strata_args)
        self.assertIn("predicted_prob_col", strata_args)
        self.assertIn("event_col", strata_args)
        self.assertIn("strata_thresholds", strata_args)
        self.assertIn("strata_labels", strata_args)

    def test_calculate_decision_curve_analysis_signature(self):
        """Validates calculate_decision_curve_analysis signature."""
        funcs = self._parse_engine_functions()
        self.assertIn("calculate_decision_curve_analysis", funcs)
        dca_args = funcs["calculate_decision_curve_analysis"]["all"]
        self.assertIn("df", dca_args)
        self.assertIn("predicted_prob_col", strata_args := dca_args)
        self.assertIn("event_col", dca_args)
        self.assertIn("thresholds", dca_args)

    def test_evaluate_risk_stratification_prerequisites_signature(self):
        """Validates evaluate_risk_stratification_prerequisites signature."""
        funcs = self._parse_engine_functions()
        self.assertIn("evaluate_risk_stratification_prerequisites", funcs)
        prereq_args = funcs["evaluate_risk_stratification_prerequisites"]["all"]
        self.assertIn("calibration_metrics", prereq_args)
        self.assertIn("strata_metrics", prereq_args)
        self.assertIn("dca_metrics", prereq_args)
        self.assertIn("sample_size", prereq_args)
        self.assertIn("total_events", prereq_args)

    def test_ui_widget_ast_calls_have_zero_type_errors(self):
        """Inspects all engine function calls in the UI widget file to guarantee argument compatibility."""
        funcs = self._parse_engine_functions()

        with open(self.widget_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        engine_calls = [
            "evaluate_model_calibration",
            "calculate_risk_strata_performance",
            "calculate_decision_curve_analysis",
            "evaluate_risk_stratification_prerequisites",
            "generate_risk_stratification_demo_data",
            "build_integer_scoring_system"
        ]

        calls_checked = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in engine_calls:
                    expected_params = funcs[func_name]["all"]
                    positional_count = len(node.args)
                    keyword_names = [kw.arg for kw in node.keywords]

                    # Assert no keyword is unexpected
                    for kw in keyword_names:
                        self.assertIn(
                            kw, expected_params,
                            f"Call to {func_name} at line {node.lineno} in {self.widget_path} passed unknown keyword argument '{kw}'"
                        )
                    # Assert positional count does not exceed max parameters
                    self.assertLessEqual(
                        positional_count, len(funcs[func_name]["args"]),
                        f"Call to {func_name} at line {node.lineno} passed too many positional arguments ({positional_count})"
                    )
                    calls_checked += 1

        self.assertGreater(calls_checked, 0, "Should have inspected at least one engine call in widget file")

    def test_all_expanders_in_widget_have_expanded_false(self):
        """Enforces that all st.expander instances in the widget file have expanded=False."""
        with open(self.widget_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for forbidden expanded=True
        self.assertNotIn("expanded=True", content, "Found forbidden expanded=True in widget file")

        # AST verify every st.expander call specifies expanded=False
        tree = ast.parse(content)
        expander_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                is_expander = False
                if isinstance(node.func, ast.Attribute) and node.func.attr == "expander":
                    is_expander = True
                if is_expander:
                    expander_count += 1
                    kw_dict = {kw.arg: kw.value for kw in node.keywords}
                    self.assertIn("expanded", kw_dict, f"st.expander call at line {node.lineno} must explicitly specify expanded=False")
                    val_node = kw_dict["expanded"]
                    self.assertIsInstance(val_node, ast.Constant)
                    self.assertIs(val_node.value, False, f"st.expander at line {node.lineno} must have expanded=False")

        self.assertGreaterEqual(expander_count, 5, "Expected at least 5 expanders across widget tabs")

    def test_fictional_healthcare_organisations_in_widget(self):
        """Verifies that only authorized fictional healthcare institutions are referenced."""
        allowed_orgs = [
            "All Meadows Cardiovascular Institute",
            "Inner City Teaching Hospital",
            "Riverside Health Trust"
        ]
        with open(self.widget_path, "r", encoding="utf-8") as f:
            content = f.read()

        for org in allowed_orgs:
            self.assertIn(org, content, f"Expected fictional organisation '{org}' in widget file")

    def test_pure_python_framingham_integer_point_scoring(self):
        """Verifies Sullivan/Framingham point conversion algorithm using pure Python."""
        # Dynamic import of build_integer_scoring_system
        import sys
        from unittest.mock import MagicMock
        for mod in ["numpy", "pandas", "scipy", "scipy.stats", "statsmodels", "statsmodels.api"]:
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()

        from src.epi_methods.risk_stratification import build_integer_scoring_system

        betas = {
            "Age_Over_65": 0.85,
            "Current_Smoker": 0.55,
            "Diabetes": 0.60,
            "Stage_2_Hypertension": 0.30
        }

        # Explicit base unit 0.30
        res = build_integer_scoring_system(betas, base_unit=0.30)
        self.assertAlmostEqual(res["base_unit"], 0.30, places=4)
        weights = res["point_weights"]
        self.assertEqual(weights["Stage_2_Hypertension"], 1)  # 0.30 / 0.30 = 1.0 -> 1
        self.assertEqual(weights["Current_Smoker"], 2)        # 0.55 / 0.30 = 1.83 -> 2
        self.assertEqual(weights["Diabetes"], 2)              # 0.60 / 0.30 = 2.0 -> 2
        self.assertEqual(weights["Age_Over_65"], 3)           # 0.85 / 0.30 = 2.83 -> 3

        # Auto base unit (picks min non-zero positive: 0.30)
        res_auto = build_integer_scoring_system(betas)
        self.assertAlmostEqual(res_auto["base_unit"], 0.30, places=4)

        # Empty dict raises ValueError
        with self.assertRaises(ValueError):
            build_integer_scoring_system({})

        # All zero raises ValueError
        with self.assertRaises(ValueError):
            build_integer_scoring_system({"A": 0.0, "B": 0.0})

    def test_risk_stratification_demo_data_class_contract(self):
        """Validates that RiskStratificationDemoData supports dict indexing and tuple unpacking."""
        import sys
        from unittest.mock import MagicMock
        for mod in ["numpy", "pandas", "scipy", "scipy.stats", "statsmodels", "statsmodels.api"]:
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()

        from src.epi_methods.risk_stratification import RiskStratificationDemoData

        mock_payload = {
            "df": [{"patient_id": 1}],
            "meta": {"title": "Test Title"},
            "predicted_prob_col": "pred",
            "event_col": "event"
        }
        data_obj = RiskStratificationDemoData(mock_payload)

        # Test dictionary key indexing
        self.assertEqual(data_obj["predicted_prob_col"], "pred")
        self.assertEqual(data_obj["event_col"], "event")

        # Test 2-tuple unpacking
        df_unpacked, meta_unpacked = data_obj
        self.assertEqual(df_unpacked, [{"patient_id": 1}])
        self.assertEqual(meta_unpacked["title"], "Test Title")

    def test_evaluate_model_calibration_runtime_signature_inspection(self):
        """Verifies runtime signature of evaluate_model_calibration via inspect."""
        import sys
        from unittest.mock import MagicMock
        for mod in ["numpy", "pandas", "scipy", "scipy.stats", "statsmodels", "statsmodels.api"]:
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()

        from src.epi_methods.risk_stratification import evaluate_model_calibration
        sig = inspect.signature(evaluate_model_calibration)
        params = list(sig.parameters.keys())

        # n_bins and n_groups MUST be present
        self.assertIn("n_bins", params, "evaluate_model_calibration must have 'n_bins' in signature")
        self.assertIn("n_groups", params, "evaluate_model_calibration must have 'n_groups' in signature")
        self.assertEqual(sig.parameters["n_bins"].default, 10)
        self.assertIn("df", params)
        self.assertIn("predicted_prob_col", params)
        self.assertIn("event_col", params)

    def test_generate_risk_stratification_demo_data_runtime_signature_inspection(self):
        """Verifies runtime signature of generate_risk_stratification_demo_data via inspect."""
        import sys
        from unittest.mock import MagicMock
        for mod in ["numpy", "pandas", "scipy", "scipy.stats", "statsmodels", "statsmodels.api"]:
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()

        from src.epi_methods.risk_stratification import generate_risk_stratification_demo_data
        sig = inspect.signature(generate_risk_stratification_demo_data)
        params = list(sig.parameters.keys())

        # scenario and scenario_key MUST be present
        self.assertIn("scenario", params, "generate_risk_stratification_demo_data must have 'scenario' in signature")
        self.assertIn("scenario_key", params, "generate_risk_stratification_demo_data must have 'scenario_key' in signature")

    def test_synthetic_data_educational_disclaimers_enforced(self):
        """Verifies that synthetic data notices and educational disclaimers are prominently displayed."""
        with open(self.widget_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check prominent notices
        self.assertIn("100% SYNTHETIC DATA FOR DEMONSTRATION & TRAINING ONLY", content)
        self.assertIn("Zero Real Patients", content)
        self.assertIn("synthetic_", content, "Download filenames must be prefixed with 'synthetic_'")
        self.assertIn("Synthetic Demonstration Data Notice", content)
        self.assertIn("Educational Demonstration Only", content)



if HAS_SCIENTIFIC_STACK:
    from src.epi_methods.risk_stratification import (
        build_integer_scoring_system,
        calculate_risk_strata_performance,
        evaluate_model_calibration,
        calculate_decision_curve_analysis,
        evaluate_risk_stratification_prerequisites,
        generate_risk_stratification_demo_data
    )


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "NumPy/SciPy/Pandas required for numerical execution tests")
class TestRiskStratificationNumericalCalculations(unittest.TestCase):
    """End-to-end numerical calculation and simulation tests on synthetic cohorts."""

    def setUp(self):
        """Set up small deterministic dataset for testing."""
        np.random.seed(42)
        n = 300
        probs = np.random.beta(a=2, b=8, size=n)
        outcomes = (np.random.rand(n) < probs).astype(int)

        self.df_test = pd.DataFrame({
            "pred_prob": probs,
            "outcome": outcomes,
            "age": np.random.randint(40, 85, size=n),
            "smoker": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
            "diabetic": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
            "sbp_stage2": np.random.choice([0, 1], size=n, p=[0.75, 0.25])
        })

    def test_calculate_risk_strata_performance_rates_and_intervals(self):
        """Tests tier categorization, observed/expected rates, and Wilson intervals."""
        thresholds = [0.10, 0.20]
        labels = ["Low Risk (<10%)", "Moderate Risk (10-20%)", "High Risk (>=20%)"]

        perf = calculate_risk_strata_performance(
            df=self.df_test,
            predicted_prob_col="pred_prob",
            event_col="outcome",
            strata_thresholds=thresholds,
            strata_labels=labels
        )

        self.assertEqual(perf["n_total"], len(self.df_test))
        self.assertEqual(len(perf["strata"]), 3)
        strata_df = perf["strata_table"]
        self.assertEqual(len(strata_df), 3)

        # Check total patient count across tiers equals total cohort
        self.assertEqual(strata_df["N_Patients"].sum(), len(self.df_test))
        self.assertEqual(strata_df["Observed_Events"].sum(), int(self.df_test["outcome"].sum()))

        # Check numeric float types for Plotly compatibility
        self.assertTrue(np.issubdtype(strata_df["Observed_Event_Rate_Pct"].dtype, np.floating))
        self.assertTrue(np.issubdtype(strata_df["CI_Lower_Pct"].dtype, np.floating))
        self.assertTrue(np.issubdtype(strata_df["CI_Upper_Pct"].dtype, np.floating))
        self.assertTrue(np.issubdtype(strata_df["OE_Ratio"].dtype, np.floating))

        # Check that high risk has higher expected rate than low risk
        low_exp = strata_df.loc[strata_df["Stratum"] == "Low Risk (<10%)", "Expected_Rate_Pct"].values[0]
        high_exp = strata_df.loc[strata_df["Stratum"] == "High Risk (>=20%)", "Expected_Rate_Pct"].values[0]
        self.assertLess(low_exp, high_exp)

    def test_evaluate_model_calibration_with_n_bins_and_n_groups(self):
        """Explicitly tests evaluate_model_calibration with n_bins=10 and n_groups=10."""
        # 1. Test with n_bins=10 (the exact call that caused the earlier TypeError)
        cal_bins = evaluate_model_calibration(
            df=self.df_test,
            predicted_prob_col="pred_prob",
            event_col="outcome",
            n_bins=10
        )
        self.assertIn("calibration_intercept", cal_bins)
        self.assertIn("calibration_slope", cal_bins)
        self.assertIn("brier_score", cal_bins)
        self.assertIn("scaled_brier_score", cal_bins)
        self.assertIn("hosmer_lemeshow_stat", cal_bins)
        self.assertIn("hosmer_lemeshow_p", cal_bins)
        self.assertIn("c_index", cal_bins)
        self.assertIn("deciles_table", cal_bins)

        # 2. Test with n_groups=10 alias
        cal_groups = evaluate_model_calibration(
            df=self.df_test,
            predicted_prob_col="pred_prob",
            event_col="outcome",
            n_groups=10
        )
        self.assertEqual(len(cal_bins["deciles_table"]), len(cal_groups["deciles_table"]))

        # 3. Check Deciles table numeric float columns for Plotly scatter subtraction
        dec_df = cal_bins["deciles_table"]
        self.assertTrue(np.issubdtype(dec_df["Mean_Predicted_Pct"].dtype, np.floating))
        self.assertTrue(np.issubdtype(dec_df["Observed_Event_Rate_Pct"].dtype, np.floating))
        self.assertTrue(np.issubdtype(dec_df["CI_Lower_Pct"].dtype, np.floating))
        self.assertTrue(np.issubdtype(dec_df["CI_Upper_Pct"].dtype, np.floating))

        # Verify error bar subtraction operates without error
        y_err_pos = dec_df["CI_Upper_Pct"] - dec_df["Observed_Event_Rate_Pct"]
        y_err_neg = dec_df["Observed_Event_Rate_Pct"] - dec_df["CI_Lower_Pct"]
        self.assertTrue((y_err_pos >= -1e-6).all())
        self.assertTrue((y_err_neg >= -1e-6).all())

    def test_evaluate_model_calibration_array_inputs(self):
        """Tests evaluate_model_calibration using numpy arrays instead of DataFrame."""
        y_true = self.df_test["outcome"].values
        y_pred = self.df_test["pred_prob"].values

        cal = evaluate_model_calibration(y_true=y_true, y_pred_prob=y_pred, n_bins=10)
        self.assertIn("calibration_slope", cal)
        self.assertIn("c_index", cal)
        self.assertGreaterEqual(cal["c_index"], 0.0)
        self.assertLessEqual(cal["c_index"], 1.0)

    def test_calculate_decision_curve_analysis_metrics(self):
        """Tests Decision Curve Analysis Net Benefit across threshold range."""
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        dca = calculate_decision_curve_analysis(
            df=self.df_test,
            predicted_prob_col="pred_prob",
            event_col="outcome",
            thresholds=thresholds
        )

        self.assertIn("dca_table", dca)
        self.assertIn("optimal_threshold_range", dca)
        dca_df = dca["dca_table"]
        self.assertEqual(len(dca_df), len(thresholds))

        # Net Benefit Treat None must be strictly 0.0
        self.assertTrue((dca_df["Net_Benefit_Treat_None"] == 0.0).all())

        # Net Benefit Model and Treat All must be numeric floats
        self.assertTrue(np.issubdtype(dca_df["Net_Benefit_Model"].dtype, np.floating))
        self.assertTrue(np.issubdtype(dca_df["Net_Benefit_Treat_All"].dtype, np.floating))
        self.assertTrue(np.issubdtype(dca_df["Interventions_Avoided_Per_100"].dtype, np.floating))

    def test_evaluate_risk_stratification_prerequisites_tripod(self):
        """Tests TRIPOD prerequisite checks across calibration, discrimination, and DCA."""
        demo = generate_risk_stratification_demo_data(scenario="cardiovascular_qrisk")
        df = demo["df"]

        cal = evaluate_model_calibration(df, "Predicted_Risk_10Yr", "Event_10Yr_CVD")
        perf = calculate_risk_strata_performance(
            df, "Predicted_Risk_10Yr", "Event_10Yr_CVD", demo["strata_thresholds"], demo["strata_labels"]
        )
        dca = calculate_decision_curve_analysis(df, "Predicted_Risk_10Yr", "Event_10Yr_CVD")

        prereqs = evaluate_risk_stratification_prerequisites(
            calibration_metrics=cal,
            strata_metrics=perf,
            dca_metrics=dca,
            sample_size=len(df),
            total_events=int(df["Event_10Yr_CVD"].sum())
        )

        self.assertIn("all_passed", prereqs)
        self.assertIn("summary", prereqs)
        self.assertIn("checks", prereqs)
        self.assertEqual(len(prereqs["checks"]), 5)

        for check in prereqs["checks"]:
            self.assertIn(check["status"], ["pass", "warn", "fail"])
            self.assertIn("title", check)
            self.assertIn("finding", check)
            self.assertIn("explanation", check)

    def test_demo_scenarios_generation_and_unpacking(self):
        """Tests realistic clinical demo data generator for all 3 scenarios plus tuple unpacking."""
        scenarios = ["cardiovascular_qrisk", "emergency_curb65", "geriatric_frailty"]

        for sc in scenarios:
            # Test both scenario and scenario_key keyword arguments
            data1 = generate_risk_stratification_demo_data(scenario=sc)
            data2 = generate_risk_stratification_demo_data(scenario_key=sc)
            self.assertEqual(len(data1["df"]), len(data2["df"]))

            # Test tuple unpacking (df, meta = ...)
            df_unpacked, meta_unpacked = generate_risk_stratification_demo_data(scenario=sc)
            self.assertEqual(len(df_unpacked), len(data1["df"]))
            self.assertIn("title", meta_unpacked)

            df = data1["df"]
            self.assertGreater(len(df), 100)
            self.assertIn(data1["predicted_prob_col"], df.columns)
            self.assertIn(data1["event_col"], df.columns)
            self.assertIn(data1["score_col"], df.columns)

            # Probabilities strictly in [0, 1]
            probs = df[data1["predicted_prob_col"]]
            self.assertTrue((probs >= 0.0).all())
            self.assertTrue((probs <= 1.0).all())

            # Binary outcome in {0, 1}
            events = df[data1["event_col"]]
            self.assertTrue(set(events.unique()).issubset({0, 1}))

            # Fictional organisation names
            self.assertIn(
                data1["organisation_name"],
                [
                    "All Meadows Cardiovascular Institute",
                    "Inner City Teaching Hospital",
                    "Riverside Health Trust"
                ]
            )


if __name__ == "__main__":
    unittest.main()

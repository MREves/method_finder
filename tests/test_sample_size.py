"""Unit & Mathematical Verification Test Suite for Sample Size & Statistical Power Engine.

Validates:
1. Continuous Outcome Calculations:
   - Two-sample independent t-test (equal and unequal allocation k = n2 / n1).
   - Paired t-test for repeated/pre-post measures.
   - One-Way ANOVA with Cohen's f effect sizes.
   - Pearson correlation via Fisher's z-transformation.
2. Binary Outcome & Risk Ratio Calculations:
   - Two independent proportions (Fleiss unpooled and Casagrande-Pike continuity correction).
   - Prospective cohort study for Relative Risk (RR).
   - Retrospective case-control study for Odds Ratio (OR) with matching ratios.
   - Matched pairs discordant McNemar test.
3. Longitudinal Survival Calculations:
   - Schoenfeld's (1981) required events formula E for target Hazard Ratios (HR).
   - Event-to-enrollment inflation based on cumulative follow-up event probability.
4. Diagnostic & Screening Accuracy Calculations:
   - Target sensitivity / specificity estimation within Wilson score margin of error W.
   - Prevalence-adjusted total screening cohort enrollment.
5. Trial Modifiers & Real-World Pragmatics:
   - Attrition & loss-to-follow-up inflation.
   - Cluster randomization design effect (ICC and VIF).
6. Power Curves & Sensitivity Surface:
   - Monotonicity checks across sample size sequences.
7. Protocol Statement & Prerequisite Evaluations:
   - Automatic CONSORT/STROBE/NIHR paragraph generation.
   - 5 ICH-E9 design assumption checks.
8. Demo Data Contracts & Fictional Healthcare Entities:
   - 2-tuple unpacking (df, meta = ...).
   - Strictly fictional UK healthcare entities.
   - Prominent synthetic data disclaimers.
"""

import os
import sys
import math
import inspect
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib.util
engine_spec = importlib.util.spec_from_file_location(
    "sample_size_engine",
    os.path.join(PROJECT_ROOT, "src", "stats_calcs", "sample_size_engine.py")
)
sse = importlib.util.module_from_spec(engine_spec)
engine_spec.loader.exec_module(sse)


class TestSampleSizeMathematicalEngine(unittest.TestCase):
    """Mathematical and biostatistical verification tests for sample size algorithms."""

    def test_two_means_equal_allocation(self):
        """Tests two-sample t-test with equal allocation (k = 1.0)."""
        # Difference = 2.0, SD = 4.0 -> Cohen's d = 0.50
        # For alpha = 0.05, power = 0.80: n per arm = 2 * (1.96 + 0.8416)^2 / 0.5^2 = 62.79 -> 63
        res = sse.calculate_sample_size_two_means(
            mean1=10.0, mean2=12.0, sd1=4.0, sd2=4.0, alpha=0.05, power=0.80, allocation_ratio=1.0
        )
        self.assertEqual(res["n1"], 63)
        self.assertEqual(res["n2"], 63)
        self.assertEqual(res["n_total"], 126)
        self.assertAlmostEqual(res["cohen_d"], 0.50, delta=0.01)
        self.assertGreaterEqual(res["achieved_power"], 0.80)
        self.assertEqual(res["cohen_tier"], "Medium (0.50 - 0.79)")

    def test_two_means_unequal_allocation(self):
        """Tests two-sample t-test with 1:2 unequal allocation (k = 2.0)."""
        res = sse.calculate_sample_size_two_means(
            mean1=10.0, mean2=12.0, sd1=4.0, sd2=4.0, alpha=0.05, power=0.80, allocation_ratio=2.0
        )
        self.assertEqual(res["allocation_ratio"], 2.0)
        self.assertEqual(res["n2"], res["n1"] * 2)
        # Unequal allocation requires fewer in group 1, but more in group 2
        self.assertLess(res["n1"], 63)
        self.assertGreater(res["n2"], 63)
        self.assertGreaterEqual(res["achieved_power"], 0.80)

    def test_paired_means_sample_size(self):
        """Tests paired t-test for repeated pre-post measures."""
        # Paired mean diff = 2.0, SD of diff = 4.0 -> Cohen's d_z = 0.50
        # n_pairs = (1.96 + 0.8416)^2 / 0.5^2 = 31.39 -> 32 pairs
        res = sse.calculate_sample_size_paired_means(
            mean_diff=2.0, sd_diff=4.0, alpha=0.05, power=0.80
        )
        self.assertEqual(res["n_pairs"], 32)
        self.assertEqual(res["n_total_observations"], 64)
        self.assertAlmostEqual(res["cohen_dz"], 0.50, delta=0.01)
        self.assertGreaterEqual(res["achieved_power"], 0.80)

    def test_anova_sample_size(self):
        """Tests One-Way ANOVA with Cohen's f medium effect size (f = 0.25)."""
        res = sse.calculate_sample_size_anova(num_groups=3, cohen_f=0.25, alpha=0.05, power=0.80)
        self.assertEqual(res["num_groups"], 3)
        self.assertGreaterEqual(res["n_per_group"], 40)
        self.assertEqual(res["n_total"], res["n_per_group"] * 3)
        self.assertEqual(res["df_numerator"], 2)
        self.assertEqual(res["df_denominator"], res["n_total"] - 3)

    def test_correlation_sample_size(self):
        """Tests Pearson linear correlation sample size via Fisher's z-transformation."""
        # For r = 0.30, alpha = 0.05, power = 0.80: N should be around 84 - 86
        res = sse.calculate_sample_size_correlation(r=0.30, alpha=0.05, power=0.80)
        self.assertIn(res["n_total"], [84, 85, 86])
        self.assertGreaterEqual(res["achieved_power"], 0.80)

    def test_two_proportions_sample_size(self):
        """Tests two independent proportions with uncorrected and continuity-corrected formulas."""
        # p1 = 0.20, p2 = 0.10, alpha = 0.05, power = 0.80
        res = sse.calculate_sample_size_two_proportions(
            p1=0.20, p2=0.10, alpha=0.05, power=0.80, continuity_correction=True
        )
        self.assertEqual(res["n1_uncorrected"], 199)
        self.assertEqual(res["n1"], 219)
        self.assertEqual(res["n2"], 219)
        self.assertEqual(res["n_total"], 438)
        self.assertAlmostEqual(res["relative_risk"], 0.50, delta=0.01)
        self.assertAlmostEqual(res["risk_difference"], -0.10, delta=0.01)

    def test_cohort_study_rr_sample_size(self):
        """Tests prospective cohort sample size for target Relative Risk."""
        # Baseline incidence p0 = 0.15, RR = 1.60 -> exposed incidence p1 = 0.24
        res = sse.calculate_sample_size_cohort_rr(
            p0=0.15, relative_risk=1.60, alpha=0.05, power=0.80, allocation_ratio=1.0
        )
        self.assertAlmostEqual(res["p1_exposed_incidence"], 0.24, delta=0.01)
        self.assertEqual(res["n_unexposed"], res["n_exposed"])
        self.assertGreater(res["n_total"], 500)
        self.assertGreater(res["total_expected_events"], 100)

    def test_case_control_study_or_sample_size(self):
        """Tests retrospective case-control sample size with 1:2 control-to-case matching."""
        # Control exposure = 0.20, target OR = 2.50, 2 controls per case
        res = sse.calculate_sample_size_case_control_or(
            p0_exposure=0.20, odds_ratio=2.50, alpha=0.05, power=0.80, controls_per_case=2.0
        )
        self.assertAlmostEqual(res["p1_case_exposure"], 0.3846, delta=0.005)
        self.assertEqual(res["n_controls"], res["n_cases"] * 2)
        self.assertEqual(res["n_total"], res["n_cases"] + res["n_controls"])
        self.assertGreater(res["n_cases"], 50)
        self.assertLess(res["n_cases"], 100)

    def test_survival_logrank_schoenfeld_events(self):
        """Tests Schoenfeld's formula for required events and total enrollment under proportional hazards."""
        # HR = 0.70, alpha = 0.05, power = 0.85
        # Schoenfeld E = (1.96 + 1.036)^2 / (0.25 * (ln 0.70)^2) ~ 283-284 events
        res = sse.calculate_sample_size_survival_logrank(
            hazard_ratio=0.70, alpha=0.05, power=0.85, cumulative_event_prob=0.55
        )
        self.assertIn(res["required_events"], [283, 284])
        self.assertEqual(res["events_arm1"] + res["events_arm2"], res["required_events"])
        # Total patients = ceil(283 / 0.55) = 515
        self.assertIn(res["n_total"], [515, 516, 517])

    def test_diagnostic_accuracy_wilson_precision(self):
        """Tests sample size for estimating sensitivity within target margin of error."""
        # Sensitivity = 0.92, margin W = 0.04 (+/-4%)
        # n_diseased = 1.96^2 * 0.92 * 0.08 / 0.04^2 = 176.7 -> 177
        res = sse.calculate_sample_size_diagnostic_accuracy(
            target_sensitivity=0.92, margin_of_error=0.04, disease_prevalence=0.15
        )
        self.assertEqual(res["required_diseased_cases"], 177)
        self.assertEqual(res["ci_lower_pct"], 88.0)
        self.assertEqual(res["ci_upper_pct"], 96.0)
        # Total screening cohort = ceil(177 / 0.15) = 1180
        self.assertEqual(res["required_total_screening_cohort"], 1180)

    def test_attrition_inflation(self):
        """Tests sample size inflation for anticipated participant loss to follow-up."""
        # N_eval = 200, dropout = 15% -> N_rec = ceil(200 / 0.85) = 236
        res = sse.apply_attrition_inflation(n_evaluable=200, dropout_rate=0.15)
        self.assertEqual(res["n_evaluable"], 200)
        self.assertEqual(res["n_recruited"], 236)
        self.assertEqual(res["additional_subjects_needed"], 36)
        self.assertAlmostEqual(res["inflation_factor"], 1.18, delta=0.01)

    def test_cluster_randomization_design_effect(self):
        """Tests Variance Inflation Factor (VIF) and cluster count."""
        # N = 200, cluster size m = 25, ICC = 0.05
        # VIF = 1 + (25 - 1) * 0.05 = 1 + 1.20 = 2.20
        # N_clustered = 200 * 2.2 = 440 -> ceil(440 / 25) = 18 clusters -> 18 * 25 = 450
        res = sse.apply_cluster_design_effect(n_individual=200, cluster_size=25, icc=0.05)
        self.assertAlmostEqual(res["design_effect_vif"], 2.20, delta=0.01)
        self.assertEqual(res["clusters_required"], 18)
        self.assertEqual(res["n_clustered_total"], 450)

    def test_power_curve_generation_monotonicity(self):
        """Verifies power curve generator returns valid, monotonically increasing points."""
        params = {"mean1": 10.0, "mean2": 12.0, "sd1": 4.0, "alpha": 0.05, "power": 0.80}
        curve = sse.generate_power_curve_data("two_means", params, sample_sizes=[20, 50, 100, 200])

        self.assertEqual(len(curve), 4)
        for pt in curve:
            self.assertIn("N", pt)
            self.assertIn("Power_Target", pt)
            self.assertIn("Power_Small", pt)
            self.assertIn("Power_Large", pt)
            self.assertGreaterEqual(pt["Power_Target"], 0.0)
            self.assertLessEqual(pt["Power_Target"], 100.0)

        # Monotonicity check
        powers = [pt["Power_Target"] for pt in curve]
        self.assertEqual(powers, sorted(powers))

    def test_protocol_justification_statement_generation(self):
        """Verifies automated protocol justification statement contains key details and attrition notes."""
        calc = sse.calculate_sample_size_two_means(mean1=8.2, mean2=7.75, sd1=1.1, alpha=0.05, power=0.85)
        stmt = sse.generate_protocol_justification_statement(
            calc, primary_endpoint_label="HbA1c reduction", dropout_rate=0.15
        )
        self.assertIn("HbA1c reduction", stmt)
        self.assertIn("85% statistical power", stmt)
        self.assertIn("alpha = 0.05", stmt)
        self.assertIn("15.0% loss-to-follow-up", stmt)
        self.assertIn("t-test", stmt)

    def test_calculate_unpadded_power_after_dropout(self):
        """Verifies degraded statistical power calculation when participants drop out without recruitment cushion."""
        # 1. Continuous two-means design (nominal power = 85.0%)
        calc_means = sse.calculate_sample_size_two_means(mean1=8.2, mean2=7.75, sd1=1.1, alpha=0.05, power=0.85)
        # 0% dropout -> no power loss
        p0 = sse.calculate_unpadded_power_after_dropout(calc_means, 0.0)
        self.assertEqual(p0["power_loss_pct"], 0.0)
        self.assertEqual(p0["degraded_power_pct"], p0["guaranteed_power_pct"])
        self.assertEqual(p0["effective_evaluable_n"], calc_means["n_total"])

        # 15% dropout -> degraded power drops below nominal power
        p15 = sse.calculate_unpadded_power_after_dropout(calc_means, 0.15)
        self.assertLess(p15["degraded_power_pct"], p15["guaranteed_power_pct"])
        self.assertGreater(p15["power_loss_pct"], 0.0)
        self.assertLess(p15["effective_evaluable_n"], calc_means["n_total"])
        self.assertAlmostEqual(p15["guaranteed_power_pct"] - p15["degraded_power_pct"], p15["power_loss_pct"], delta=0.2)

        # 2. Longitudinal survival log-rank design
        calc_surv = sse.calculate_sample_size_survival_logrank(hazard_ratio=0.70, alpha=0.05, power=0.80)
        p_surv = sse.calculate_unpadded_power_after_dropout(calc_surv, 0.20)
        self.assertLess(p_surv["degraded_power_pct"], 80.0)
        self.assertGreater(p_surv["power_loss_pct"], 5.0)

        # 3. Binary case-control design (Riverside)
        calc_cc = sse.calculate_sample_size_case_control_or(p0_exposure=0.20, odds_ratio=2.50, alpha=0.05, power=0.80)
        p_cc = sse.calculate_unpadded_power_after_dropout(calc_cc, 0.15)
        self.assertLess(p_cc["degraded_power_pct"], 80.0)

    def test_cluster_attrition_compounding_and_statement(self):
        """Verifies cluster trial attrition inflation compounds on clustered sample size rather than raw individual N."""
        # Unclustered N = 231 (77 cases, 154 controls)
        calc_cc = sse.calculate_sample_size_case_control_or(p0_exposure=0.20, odds_ratio=2.50, alpha=0.05, power=0.80, controls_per_case=2.0)
        clust = sse.apply_cluster_design_effect(n_individual=calc_cc["n_total"], cluster_size=30, icc=0.04)
        self.assertEqual(clust["n_clustered_total"], 510)
        self.assertEqual(clust["clusters_required"], 17)

        # 10% dropout on clustered total (510) -> ceil(510 / 0.90) = 567 (19 clusters)
        attr = sse.apply_attrition_inflation(clust["n_clustered_total"], 0.10)
        self.assertEqual(attr["n_recruited"], 567)
        self.assertEqual(attr["additional_subjects_needed"], 57)

        stmt = sse.generate_protocol_justification_statement(
            calc_cc, primary_endpoint_label="Campylobacter infection", dropout_rate=0.10, cluster_icc=0.04, cluster_size=30
        )
        self.assertIn("inflated to 510 participants across 17 distinct clusters", stmt)
        self.assertIn("final enrollment target of 567 individuals", stmt)
        self.assertIn("19 clusters", stmt)

    def test_prerequisites_evaluation_structure(self):
        """Verifies 5-point study planning prerequisite checks adhere to schema."""
        calc = sse.calculate_sample_size_two_means(mean1=8.2, mean2=7.75, sd1=1.1, alpha=0.05, power=0.85)
        evals = sse.evaluate_sample_size_prerequisites(calc)

        self.assertIn("checks", evals)
        self.assertIn("all_passed", evals)
        self.assertEqual(len(evals["checks"]), 5)

        required_keys = ["status", "badge", "title", "finding", "explanation"]
        for c in evals["checks"]:
            for k in required_keys:
                self.assertIn(k, c)

    def test_demo_data_scenarios_and_unpacking(self):
        """Verifies all 3 demo scenarios support 2-tuple unpacking and use fictional UK entities."""
        fictional_orgs = [
            "All Meadows Cardiovascular Institute",
            "Riverside Health Protection Team",
            "Inner City Teaching Hospital"
        ]
        scenarios = ["cardiovascular_rct", "public_health_case_control", "oncology_survival"]

        for sc in scenarios:
            demo1 = sse.generate_sample_size_demo_data(scenario_key=sc)
            demo2 = sse.generate_sample_size_demo_data(scenario=sc)
            self.assertEqual(demo1["scenario_key"], sc)
            self.assertEqual(demo2["scenario_key"], sc)

            # 2-tuple unpacking
            df, meta = demo1
            self.assertIsNotNone(df)
            self.assertIsInstance(meta, dict)

            # Organisation check
            org_found = any(org in demo1["organisation_name"] for org in fictional_orgs)
            self.assertTrue(org_found, f"Scenario '{sc}' must use a fictional UK hospital/health entity!")

            # Disclaimer check
            self.assertIn("disclaimer", demo1)
            disclaimer = demo1["disclaimer"].lower()
            self.assertIn("synthetic", disclaimer)
            self.assertIn("educational", disclaimer)

    def test_all_scenarios_power_curve_generation_succeeds(self):
        """Verifies generate_power_curve_data executes without KeyError for all demonstration scenarios."""
        scenarios = ["cardiovascular_rct", "public_health_case_control", "oncology_survival"]
        for sc in scenarios:
            demo = sse.generate_sample_size_demo_data(scenario_key=sc)
            calc_results = demo["calc_results_primary"]
            design_type = "two_means" if "cohen_d" in calc_results else ("survival" if "hazard_ratio" in calc_results else "two_proportions")

            # This must succeed without throwing KeyError (e.g. 'p1' in case-control / cohort)
            curve = sse.generate_power_curve_data(design_type, calc_results)
            self.assertGreater(len(curve), 5, f"Curve for {sc} should generate points")
            for pt in curve:
                self.assertIn("N", pt)
                self.assertIn("Power_Target", pt)
                self.assertIn("Power_Small", pt)
                self.assertIn("Power_Large", pt)

    def test_scenario_parameter_interactions_recalculate_accurately(self):
        """Verifies that adjusting statistical parameters in pre-set scenarios dynamically updates sizing."""
        # 1. Cardiovascular RCT: Higher power -> higher N; larger delta -> lower N
        res_85 = sse.calculate_sample_size_two_means(mean1=8.20, mean2=7.75, sd1=1.10, alpha=0.05, power=0.85)
        res_90 = sse.calculate_sample_size_two_means(mean1=8.20, mean2=7.75, sd1=1.10, alpha=0.05, power=0.90)
        res_large_diff = sse.calculate_sample_size_two_means(mean1=8.45, mean2=7.75, sd1=1.10, alpha=0.05, power=0.85)
        self.assertGreater(res_90["n_total"], res_85["n_total"])
        self.assertLess(res_large_diff["n_total"], res_85["n_total"])

        # 2. Riverside Case-Control: Stronger OR -> lower N; higher ICC -> higher clustered N
        res_cc_base = sse.calculate_sample_size_case_control_or(p0_exposure=0.20, odds_ratio=2.50, alpha=0.05, power=0.80, controls_per_case=2.0)
        res_cc_strong = sse.calculate_sample_size_case_control_or(p0_exposure=0.20, odds_ratio=3.50, alpha=0.05, power=0.80, controls_per_case=2.0)
        self.assertLess(res_cc_strong["n_total"], res_cc_base["n_total"])

        clust_low_icc = sse.apply_cluster_design_effect(res_cc_base["n_total"], cluster_size=30, icc=0.02)
        clust_high_icc = sse.apply_cluster_design_effect(res_cc_base["n_total"], cluster_size=30, icc=0.08)
        self.assertGreater(clust_high_icc["n_clustered_total"], clust_low_icc["n_clustered_total"])

        # 0% dropout preserves exact clustered total without artificial inflation
        attr_zero = sse.apply_attrition_inflation(clust_low_icc["n_clustered_total"], 0.0)
        self.assertEqual(attr_zero["n_recruited"], clust_low_icc["n_clustered_total"])
        self.assertEqual(attr_zero["additional_subjects_needed"], 0)

        # 3. Oncology Survival: Stronger HR (lower HR) -> fewer events and lower N
        res_surv_70 = sse.calculate_sample_size_survival_logrank(hazard_ratio=0.70, alpha=0.05, power=0.85, cumulative_event_prob=0.65)
        res_surv_55 = sse.calculate_sample_size_survival_logrank(hazard_ratio=0.55, alpha=0.05, power=0.85, cumulative_event_prob=0.65)
        self.assertLess(res_surv_55["n_total"], res_surv_70["n_total"])
        self.assertLess(res_surv_55["required_events"], res_surv_70["required_events"])


class TestSampleSizeUIWidgetsContract(unittest.TestCase):
    """AST verification, integration checks, and UI contract tests for sample_size_widgets.py."""

    def setUp(self):
        import ast
        self.ast = ast
        self.widgets_path = os.path.join(PROJECT_ROOT, "src", "ui_components", "sample_size_widgets.py")
        self.engine_path = os.path.join(PROJECT_ROOT, "src", "stats_calcs", "sample_size_engine.py")

        with open(self.widgets_path, "r", encoding="utf-8") as f:
            self.widget_tree = self.ast.parse(f.read())
        with open(self.engine_path, "r", encoding="utf-8") as f:
            self.engine_tree = self.ast.parse(f.read())

    def _parse_engine_functions(self):
        """Extracts function arguments and keyword arguments from engine AST."""
        funcs = {}
        for node in self.ast.walk(self.engine_tree):
            if isinstance(node, self.ast.FunctionDef):
                arg_names = [a.arg for a in node.args.args]
                funcs[node.name] = {
                    "args": arg_names,
                    "defaults": len(node.args.defaults)
                }
        return funcs

    def test_ui_widgets_ast_expander_compliance(self):
        """Strict AST check: Verifies EVERY st.expander call in sample_size_widgets.py has expanded=False."""
        expander_calls = 0
        violations = []
        for node in self.ast.walk(self.widget_tree):
            if isinstance(node, self.ast.Call):
                func = node.func
                is_expander = False
                if isinstance(func, self.ast.Attribute) and func.attr == "expander":
                    is_expander = True
                elif isinstance(func, self.ast.Name) and func.id == "expander":
                    is_expander = True

                if is_expander:
                    expander_calls += 1
                    has_expanded_false = False
                    for kw in node.keywords:
                        if kw.arg == "expanded" and isinstance(kw.value, self.ast.Constant) and kw.value.value is False:
                            has_expanded_false = True
                    if not has_expanded_false:
                        violations.append(node.lineno)

        self.assertGreater(expander_calls, 10, "Expected at least 10 expanders across the 4-tab architecture!")
        self.assertEqual(len(violations), 0, f"Found st.expander calls without expanded=False at lines: {violations}")

    def test_ast_widget_calls_to_engine_have_zero_type_errors(self):
        """Scans sample_size_widgets.py AST to guarantee all calls to engine use valid args."""
        engine_funcs = self._parse_engine_functions()

        for node in self.ast.walk(self.widget_tree):
            if isinstance(node, self.ast.Call):
                func_name = None
                if isinstance(node.func, self.ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, self.ast.Attribute):
                    func_name = node.func.attr

                if func_name in engine_funcs:
                    known = engine_funcs[func_name]
                    for kw in node.keywords:
                        if kw.arg is not None:
                            self.assertIn(
                                kw.arg, known["args"],
                                f"Widget calls '{func_name}' with illegal keyword argument '{kw.arg}' at line {node.lineno}!"
                            )

    def test_ui_widget_function_definitions_exist(self):
        """Verifies that all core UI rendering and visualization functions are defined in the AST."""
        required_functions = [
            "render_sample_size_plain_english_guide",
            "render_sample_size_prerequisite_cards",
            "render_sample_size_metric_dashboard",
            "build_power_curve_plot",
            "build_mdes_curve_plot",
            "render_sample_size_protocol_card",
            "render_sample_size_sensitivity_grid",
            "render_sample_size_educational_tabs",
            "render_sample_size_module"
        ]
        defined_functions = [
            node.name for node in self.ast.walk(self.widget_tree) if isinstance(node, self.ast.FunctionDef)
        ]
        for name in required_functions:
            self.assertIn(name, defined_functions, f"Missing function definition: '{name}' in sample_size_widgets.py")

    def test_ui_educational_tabs_architecture_intact(self):
        """Verifies that the 4-tab educational architecture is implemented in the widget file."""
        with open(self.widgets_path, "r", encoding="utf-8") as f:
            content = f.read()

        expected_tabs = [
            "📋 Rationale & Epidemiological Framework",
            "📗 Excel Implementation Guide",
            "🐍 Python Implementation Guide",
            "💾 Data Downloads"
        ]
        for tab in expected_tabs:
            self.assertIn(tab, content, f"Educational tab '{tab}' missing from sample_size_widgets.py!")

    def test_ui_widgets_file_contains_fictional_organisations(self):
        """Verifies that fictional UK healthcare entities are configured in sample_size_widgets.py."""
        with open(self.widgets_path, "r", encoding="utf-8") as f:
            content = f.read()

        fictional_orgs = [
            "All Meadows Cardiovascular Institute",
            "Riverside Health Protection Team",
            "Inner City Teaching Hospital"
        ]
        for org in fictional_orgs:
            self.assertIn(org, content, f"Fictional organisation '{org}' not found in sample_size_widgets.py!")

    def test_protocol_card_contains_user_responsibility_disclaimer(self):
        """Verifies that the protocol justification card incorporates the user responsibility disclaimer."""
        with open(self.widgets_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("user's responsibility", content)
        self.assertIn("no liability or responsibility is taken", content)


if __name__ == "__main__":
    unittest.main()


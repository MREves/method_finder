"""Unit and Contract Verification Test Suite for Universal Study Design & Method Selector.

Validates:
1. Domain Profiles Schema (7 domains across healthcare, industry, marketing, SaaS, logistics).
2. Universal Problem Taxonomy (14 problems spanning Statistics, Epidemiology, and Machine Learning).
3. Real-World Problem Blueprints (8 quick-start end-to-end presets).
4. Recommendation Routing Engine & Inferences.
5. Cross-Pillar 3-Way Comparison Matrix Generation.
6. 5-Point Study Protocol Audit Checklist Generation.
7. Deep-Link Target Contract Integrity (Page routes exist on filesystem, session keys match module contracts).
8. AST Audit for 100% st.expander(expanded=False) compliance.
"""

import os
import sys
import ast
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.selector.study_design_schema import (
    DOMAIN_PROFILES,
    PROBLEM_DEFINITIONS,
    PROBLEM_BLUEPRINTS
)
from src.selector.selector_engine import (
    get_problem_blueprint,
    route_universal_method_recommendation,
    generate_study_design_checklist,
    generate_pillar_comparison_matrix
)


class TestDomainProfiles(unittest.TestCase):
    """Verifies real-world domain profiles schema and metadata."""

    def test_domain_profiles_coverage(self):
        expected_domains = [
            "healthcare_clinical",
            "healthcare_public_health",
            "manufacturing",
            "marketing",
            "subscriptions",
            "supply_chain",
            "general_research"
        ]
        for d in expected_domains:
            self.assertIn(d, DOMAIN_PROFILES)
            profile = DOMAIN_PROFILES[d]
            self.assertEqual(profile["id"], d)
            self.assertTrue(len(profile["name"]) > 0)
            self.assertTrue(len(profile["organisation"]) > 0)
            self.assertTrue(len(profile["description"]) > 0)
            self.assertTrue(len(profile["typical_endpoints"]) >= 2)


class TestProblemDefinitions(unittest.TestCase):
    """Verifies universal problem taxonomy, cross-pillar mappings, and page routes."""

    def test_problem_definitions_coverage(self):
        self.assertGreaterEqual(len(PROBLEM_DEFINITIONS), 14)

    def test_problem_attributes_and_routes(self):
        for p_id, p in PROBLEM_DEFINITIONS.items():
            self.assertEqual(p["id"], p_id)
            self.assertTrue(len(p["name"]) > 0)
            self.assertTrue(len(p["summary"]) > 0)
            self.assertIn(p["primary_pillar"], ["statistics", "epidemiology", "machine_learning"])
            self.assertTrue(len(p["primary_method_key"]) > 0)
            self.assertTrue(len(p["primary_method_name"]) > 0)
            self.assertTrue(len(p["alternative_method_key"]) > 0)
            self.assertTrue(len(p["alternative_method_name"]) > 0)
            self.assertTrue(len(p["alternative_condition"]) > 0)
            self.assertTrue(len(p["sample_size_framework"]) > 0)
            self.assertTrue(len(p["sample_size_label"]) > 0)
            self.assertIsInstance(p["sample_size_params"], dict)

            # Deep Link Session State and Routes
            self.assertTrue(len(p["page_route"]) > 0)
            full_page_path = os.path.join(PROJECT_ROOT, p["page_route"])
            self.assertTrue(os.path.isfile(full_page_path), f"Page route does not exist: {p['page_route']}")
            self.assertIsInstance(p["deep_link_session"], dict)
            self.assertGreater(len(p["deep_link_session"]), 0)


class TestProblemBlueprints(unittest.TestCase):
    """Verifies pre-configured quick-start problem blueprints."""

    def test_blueprints_coverage(self):
        self.assertGreaterEqual(len(PROBLEM_BLUEPRINTS), 8)

    def test_blueprint_retrieval(self):
        for bp_id, bp in PROBLEM_BLUEPRINTS.items():
            retrieved = get_problem_blueprint(bp_id)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved["id"], bp_id)
            self.assertIn(retrieved["domain"], DOMAIN_PROFILES)
            self.assertIn(retrieved["problem_key"], PROBLEM_DEFINITIONS)
            self.assertTrue(len(retrieved["title"]) > 0)
            self.assertTrue(len(retrieved["organisation"]) > 0)
            self.assertIn("domain_name", retrieved)
            self.assertTrue(len(retrieved["domain_name"]) > 0)
            self.assertEqual(retrieved["domain_name"], DOMAIN_PROFILES[retrieved["domain"]]["name"])

        # Non-existent blueprint fallback
        fallback = get_problem_blueprint("non_existent_key")
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback["id"], "bp_cardiovascular_trial")


class TestSelectorEngine(unittest.TestCase):
    """Verifies recommendation routing, checklist building, and comparison matrix."""

    def test_direct_problem_routing(self):
        rec = route_universal_method_recommendation(
            domain="healthcare_clinical",
            problem_key="two_group_comparison"
        )
        self.assertEqual(rec["problem_key"], "two_group_comparison")
        self.assertEqual(rec["primary_pillar"], "statistics")
        self.assertEqual(rec["primary_method_key"], "welch_t_test")
        self.assertIn("deep_link", rec)
        self.assertEqual(rec["deep_link"]["page_route"], "pages/1_📊_Statistical_Tests.py")
        self.assertEqual(rec["deep_link"]["session_state_updates"]["direct_test_key"], "welch_t_test")
        self.assertIn("sample_size_deep_link", rec)
        self.assertEqual(rec["sample_size_deep_link"]["session_state_updates"]["sample_size_scenario"], "custom")

    def test_inferred_problem_routing(self):
        rec = route_universal_method_recommendation(
            domain="manufacturing",
            analytical_goal="demand_forecasting",
            study_design="longitudinal_series"
        )
        self.assertEqual(rec["problem_key"], "time_series_forecasting")
        self.assertEqual(rec["primary_pillar"], "machine_learning")
        self.assertEqual(rec["deep_link"]["page_route"], "pages/3_🤖_Machine_Learning.py")

    def test_checklist_generation(self):
        prob = PROBLEM_DEFINITIONS["prospective_cohort_study"]
        checklist = generate_study_design_checklist(prob, sample_scale="medium", governance_priority="high")
        self.assertEqual(len(checklist), 5)
        for check in checklist:
            self.assertIn("badge", check)
            self.assertIn("title", check)
            self.assertIn("requirement", check)
            self.assertIn("guidance", check)

    def test_comparison_matrix_generation(self):
        prob = PROBLEM_DEFINITIONS["rare_event_classification"]
        matrix = generate_pillar_comparison_matrix(prob)
        self.assertEqual(len(matrix), 3)
        pillars = [item["pillar"] for item in matrix]
        self.assertTrue(any("Statistics" in p for p in pillars))
        self.assertTrue(any("Epidemiolog" in p for p in pillars))
        self.assertTrue(any("Machine Learning" in p for p in pillars))
        for item in matrix:
            self.assertIn("status", item)
            self.assertIn("approach", item)
            self.assertIn("optimizes_for", item)
            self.assertIn("when_to_choose", item)

    def test_small_sample_scale_non_parametric_adaptation(self):
        """Verifies that sample_scale='small' (N < 30) dynamically promotes exact / non-parametric methods."""
        # 1. Two-group comparison promotes Mann-Whitney U
        rec_2g = route_universal_method_recommendation(
            analytical_goal="compare_groups",
            study_design="parallel_rct",
            sample_scale="small"
        )
        self.assertEqual(rec_2g["primary_method_key"], "mann_whitney_u")
        self.assertEqual(rec_2g["alternative_method_key"], "welch_t_test")
        self.assertIn("Small Sample Notice", rec_2g["rationale"])
        self.assertEqual(rec_2g["deep_link"]["session_state_updates"]["direct_test_key"], "mann_whitney_u")

        # 2. Paired change promotes Wilcoxon Signed-Rank
        rec_paired = route_universal_method_recommendation(
            analytical_goal="paired_change",
            study_design="paired_pre_post",
            sample_scale="small"
        )
        self.assertEqual(rec_paired["primary_method_key"], "wilcoxon_paired")
        self.assertEqual(rec_paired["alternative_method_key"], "paired_t_test")
        self.assertEqual(rec_paired["deep_link"]["session_state_updates"]["direct_test_key"], "wilcoxon_paired")

        # 3. Multi-group comparison promotes Kruskal-Wallis
        rec_multi = route_universal_method_recommendation(
            analytical_goal="compare_groups",
            study_design="multi_arm",
            sample_scale="small"
        )
        self.assertEqual(rec_multi["primary_method_key"], "kruskal_wallis")
        self.assertEqual(rec_multi["alternative_method_key"], "one_way_anova")
        self.assertEqual(rec_multi["deep_link"]["session_state_updates"]["direct_test_key"], "kruskal_wallis")

        # 4. Categorical contingency promotes Fisher's Exact
        rec_cat = route_universal_method_recommendation(
            analytical_goal="contingency_rates",
            study_design="cross_sectional",
            sample_scale="small"
        )
        self.assertEqual(rec_cat["primary_method_key"], "fisher_exact")
        self.assertEqual(rec_cat["alternative_method_key"], "chi_square_independence")
        self.assertEqual(rec_cat["deep_link"]["session_state_updates"]["direct_test_key"], "fisher_exact")

        # 5. Linear correlation promotes Spearman rank
        rec_corr = route_universal_method_recommendation(
            analytical_goal="correlation_association",
            sample_scale="small"
        )
        self.assertEqual(rec_corr["primary_method_key"], "spearman_correlation")
        self.assertEqual(rec_corr["alternative_method_key"], "pearson_correlation")
        self.assertEqual(rec_corr["deep_link"]["session_state_updates"]["direct_test_key"], "spearman_correlation")

        # 6. Rare event classification promotes ElasticNet Penalized Logistic Regression
        rec_rare = route_universal_method_recommendation(
            analytical_goal="predict_rare_event",
            sample_scale="small"
        )
        self.assertEqual(rec_rare["primary_method_key"], "penalized_logistic_regression")
        self.assertEqual(rec_rare["alternative_method_key"], "random_forest_classifier")
        self.assertEqual(rec_rare["deep_link"]["session_state_updates"]["ml_direct_model_key"], "penalized_logistic_regression")

    def test_large_sample_scale_adaptation(self):
        """Verifies that sample_scale='large' (N >= 500) provides asymptotic / scalable guidance."""
        # Two-group comparison highlights effect size over raw p-values
        rec_2g = route_universal_method_recommendation(
            analytical_goal="compare_groups",
            study_design="parallel_rct",
            sample_scale="large"
        )
        self.assertIn("Large Sample Notice", rec_2g["rationale"])
        self.assertIn("Cohen's d", rec_2g["rationale"])

        # Rare event classification promotes Gradient Boosted Decision Trees
        rec_rare = route_universal_method_recommendation(
            analytical_goal="predict_rare_event",
            sample_scale="large"
        )
        self.assertEqual(rec_rare["primary_method_key"], "gradient_boosting_classifier")
        self.assertEqual(rec_rare["deep_link"]["session_state_updates"]["ml_direct_model_key"], "gradient_boosting_classifier")
        self.assertIn("Large Sample Notice", rec_rare["rationale"])

    def test_study_design_routing_and_causal_adaptation(self):
        """Verifies that study_design dynamically alters problem classification and causal rationale."""
        # 1. Contingency rates + case_control -> Retrospective Case-Control (Odds Ratio)
        rec_cc = route_universal_method_recommendation(
            analytical_goal="contingency_rates",
            study_design="case_control",
            sample_scale="medium"
        )
        self.assertEqual(rec_cc["problem_key"], "retrospective_case_control")
        self.assertEqual(rec_cc["primary_pillar"], "epidemiology")
        self.assertEqual(rec_cc["primary_method_key"], "odds_ratio")
        self.assertIn("Odds Ratio is the mathematically valid causal estimator", rec_cc["rationale"])

        # 2. Contingency rates + observational_cohort -> Prospective Cohort (Relative Risk)
        rec_cohort = route_universal_method_recommendation(
            analytical_goal="contingency_rates",
            study_design="observational_cohort",
            sample_scale="medium"
        )
        self.assertEqual(rec_cohort["problem_key"], "prospective_cohort_study")
        self.assertEqual(rec_cohort["primary_pillar"], "epidemiology")
        self.assertEqual(rec_cohort["primary_method_key"], "relative_risk")

        # 3. Compare groups + paired_pre_post -> Paired Pre-Post
        rec_paired = route_universal_method_recommendation(
            analytical_goal="compare_groups",
            study_design="paired_pre_post",
            sample_scale="medium"
        )
        self.assertEqual(rec_paired["problem_key"], "paired_pre_post")
        self.assertEqual(rec_paired["primary_method_key"], "paired_t_test")
        self.assertIn("Repeated measurements on the same observational units", rec_paired["rationale"])

        # 4. Compare groups + multi_arm -> Multi-Group Comparison (ANOVA)
        rec_multi = route_universal_method_recommendation(
            analytical_goal="compare_groups",
            study_design="multi_arm",
            sample_scale="medium"
        )
        self.assertEqual(rec_multi["problem_key"], "multi_group_comparison")
        self.assertEqual(rec_multi["primary_method_key"], "one_way_anova")
        self.assertIn("Family-Wise Error Rate", rec_multi["rationale"])

    def test_dynamic_checklist_reflects_scale_and_design(self):
        """Verifies that the 5-point protocol checklist dynamically adapts to scale and design."""
        prob = PROBLEM_DEFINITIONS["two_group_comparison"]

        # Small sample checklist
        cl_small = generate_study_design_checklist(prob, sample_scale="small", study_design="paired_pre_post")
        self.assertIn("Small Cohort", cl_small[0]["badge"])
        self.assertIn("Non-Parametric Required", cl_small[1]["badge"])
        self.assertIn("Paired / Repeated Measures", cl_small[2]["badge"])

        # Large sample checklist with longitudinal series
        cl_large = generate_study_design_checklist(prob, sample_scale="large", study_design="longitudinal_series")
        self.assertIn("Large Cohort", cl_large[0]["badge"])
        self.assertIn("Parametric Assumed", cl_large[1]["badge"])
        self.assertIn("Chronological Sequence", cl_large[2]["badge"])
        self.assertIn("Rolling Origin CV", cl_large[3]["badge"])


class TestExpanderASTCompliance(unittest.TestCase):
    """Enforces 100% compliance with st.expander(expanded=False) across all selector code."""

    def test_selector_page_expanders_are_collapsed(self):
        selector_page_path = os.path.join(PROJECT_ROOT, "pages", "7_🎯_Study_Design_and_Method_Selector.py")
        with open(selector_page_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=selector_page_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Check for st.expander
                if (isinstance(func, ast.Attribute) and func.attr == "expander") or \
                   (isinstance(func, ast.Name) and func.id == "expander"):
                    # Check for expanded keyword argument
                    expanded_kw = [kw for kw in node.keywords if kw.arg == "expanded"]
                    self.assertTrue(
                        len(expanded_kw) > 0,
                        f"st.expander call at line {node.lineno} in {selector_page_path} missing explicit expanded=False!"
                    )
                    # Value must be False (ast.Constant with value False)
                    self.assertIsInstance(expanded_kw[0].value, ast.Constant)
                    self.assertFalse(
                        expanded_kw[0].value.value,
                        f"st.expander at line {node.lineno} has expanded={expanded_kw[0].value.value}, must be False!"
                    )

    def test_selector_widgets_expanders_are_collapsed(self):
        widgets_path = os.path.join(PROJECT_ROOT, "src", "ui_components", "selector_widgets.py")
        with open(widgets_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=widgets_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "expander") or \
                   (isinstance(func, ast.Name) and func.id == "expander"):
                    expanded_kw = [kw for kw in node.keywords if kw.arg == "expanded"]
                    self.assertTrue(
                        len(expanded_kw) > 0,
                        f"st.expander call at line {node.lineno} in {widgets_path} missing explicit expanded=False!"
                    )
                    self.assertIsInstance(expanded_kw[0].value, ast.Constant)
                    self.assertFalse(
                        expanded_kw[0].value.value,
                        f"st.expander at line {node.lineno} has expanded={expanded_kw[0].value.value}, must be False!"
                    )


class TestSidebarNavigation(unittest.TestCase):
    """Verifies custom grouped sidebar navigation component and page integrations."""

    def test_navigation_module_importable(self):
        if "streamlit" not in sys.modules:
            from unittest.mock import MagicMock
            sys.modules["streamlit"] = MagicMock()
        from src.ui_components.sidebar_navigation import render_custom_sidebar_navigation
        self.assertTrue(callable(render_custom_sidebar_navigation))

    def test_streamlit_config_disables_default_nav(self):
        config_path = os.path.join(PROJECT_ROOT, ".streamlit", "config.toml")
        self.assertTrue(os.path.isfile(config_path), ".streamlit/config.toml must exist!")
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("showSidebarNavigation = false", content)

    def test_all_pages_call_custom_sidebar_navigation(self):
        all_pages = [
            os.path.join(PROJECT_ROOT, "app.py"),
            os.path.join(PROJECT_ROOT, "pages", "1_📊_Statistical_Tests.py"),
            os.path.join(PROJECT_ROOT, "pages", "2_🏥_Epidemiology_Methods.py"),
            os.path.join(PROJECT_ROOT, "pages", "3_🤖_Machine_Learning.py"),
            os.path.join(PROJECT_ROOT, "pages", "4_📐_Sample_Size_and_Power.py"),
            os.path.join(PROJECT_ROOT, "pages", "5_📋_Survey_and_Questionnaire_Analysis.py"),
            os.path.join(PROJECT_ROOT, "pages", "7_🎯_Study_Design_and_Method_Selector.py"),
            os.path.join(PROJECT_ROOT, "pages", "8_🗺️_Decision_Flowcharts.py"),
        ]
        for p in all_pages:
            self.assertTrue(os.path.isfile(p), f"Page file does not exist: {p}")
            with open(p, "r", encoding="utf-8") as f:
                code = f.read()
            self.assertIn(
                "render_custom_sidebar_navigation",
                code,
                f"{p} is missing render_custom_sidebar_navigation call!"
            )

    def test_roundtrip_banner_importable_and_integrated(self):
        if "streamlit" not in sys.modules:
            from unittest.mock import MagicMock
            sys.modules["streamlit"] = MagicMock()
        from src.ui_components.sidebar_navigation import render_roundtrip_navigation_banner
        self.assertTrue(callable(render_roundtrip_navigation_banner))

        training_pages = [
            os.path.join(PROJECT_ROOT, "pages", "1_📊_Statistical_Tests.py"),
            os.path.join(PROJECT_ROOT, "pages", "2_🏥_Epidemiology_Methods.py"),
            os.path.join(PROJECT_ROOT, "pages", "3_🤖_Machine_Learning.py"),
            os.path.join(PROJECT_ROOT, "pages", "4_📐_Sample_Size_and_Power.py"),
            os.path.join(PROJECT_ROOT, "pages", "5_📋_Survey_and_Questionnaire_Analysis.py"),
        ]
        for p in training_pages:
            with open(p, "r", encoding="utf-8") as f:
                code = f.read()
            self.assertIn(
                "render_roundtrip_navigation_banner",
                code,
                f"{p} is missing render_roundtrip_navigation_banner call!"
            )

    def test_roundtrip_banner_inactive_safety(self):
        from src.ui_components.sidebar_navigation import render_roundtrip_navigation_banner
        import streamlit as st
        # Ensure session state has no active consultation
        st.session_state["selector_consultation"] = None
        # Must execute cleanly without exception
        try:
            render_roundtrip_navigation_banner()
        except Exception as e:
            self.fail(f"render_roundtrip_navigation_banner raised unexpected error: {e}")

    def test_roundtrip_banner_active_rendering(self):
        from src.ui_components.sidebar_navigation import render_roundtrip_navigation_banner
        from unittest.mock import MagicMock
        import streamlit as st

        col1, col2 = MagicMock(), MagicMock()
        st.columns = MagicMock(return_value=[col1, col2])
        st.button = MagicMock(return_value=False)
        st.session_state["selector_consultation"] = {
            "active": True,
            "problem_name": "Two-Group Comparison",
            "method_name": "Mann-Whitney U Test",
            "domain_name": "Healthcare & Clinical",
            "sample_scale": "Small Cohort (N < 30)"
        }

        try:
            render_roundtrip_navigation_banner()
        except Exception as e:
            self.fail(f"render_roundtrip_navigation_banner raised error when active: {e}")
        finally:
            st.session_state["selector_consultation"] = None

    def test_form_persistence_across_roundtrip(self):
        """Verifies that render_problem_definition_form preserves user selections across page switches."""
        from src.ui_components.selector_widgets import render_problem_definition_form
        from unittest.mock import MagicMock
        import streamlit as st

        # Mock selectbox to return the option at index
        def mock_selectbox(label, options, index=0, **kwargs):
            return options[index]

        st.selectbox = MagicMock(side_effect=mock_selectbox)
        col1, col2 = MagicMock(), MagicMock()
        st.columns = MagicMock(return_value=[col1, col2])

        # 1. User previously selected custom options in session state
        state_dict = {
            "saved_selector_domain": "manufacturing",
            "saved_selector_goal": "demand_forecasting",
            "saved_selector_design": "longitudinal_series",
            "saved_selector_scale": "large",
            "last_loaded_blueprint_key": None
        }
        st.session_state = state_dict

        # Call form with no blueprint (or same blueprint key)
        result = render_problem_definition_form(active_blueprint=None)

        self.assertEqual(result["domain"], "manufacturing")
        self.assertEqual(result["analytical_goal"], "demand_forecasting")
        self.assertEqual(result["study_design"], "longitudinal_series")
        self.assertEqual(result["sample_scale"], "large")
        self.assertEqual(result["problem_key"], "time_series_forecasting")

    def test_render_decision_path_flowchart_importable_and_callable(self):
        """Verifies render_decision_path_flowchart is properly exposed and callable."""
        from src.ui_components.selector_widgets import render_decision_path_flowchart
        self.assertTrue(callable(render_decision_path_flowchart))

    def test_decision_path_flowchart_rendering_all_blueprints(self):
        """Verifies render_decision_path_flowchart executes cleanly for all 8 real-world blueprints."""
        from src.ui_components.selector_widgets import render_decision_path_flowchart
        from src.selector.selector_engine import route_universal_method_recommendation, get_problem_blueprint
        from unittest.mock import MagicMock
        import streamlit as st

        col1, col2 = MagicMock(), MagicMock()
        st.columns = MagicMock(return_value=[col1, col2])
        st.markdown = MagicMock()

        for bp_key in PROBLEM_BLUEPRINTS.keys():
            bp = get_problem_blueprint(bp_key)
            form_inputs = {
                "domain": bp.get("domain", "healthcare_clinical"),
                "analytical_goal": "compare_groups",
                "study_design": bp.get("study_design", "parallel_rct"),
                "sample_scale": bp.get("sample_scale", "medium"),
                "problem_key": bp.get("problem_key")
            }
            rec = route_universal_method_recommendation(
                domain=form_inputs["domain"],
                study_design=form_inputs["study_design"],
                sample_scale=form_inputs["sample_scale"],
                problem_key=form_inputs["problem_key"]
            )
            try:
                render_decision_path_flowchart(rec, form_inputs)
            except Exception as e:
                self.fail(f"render_decision_path_flowchart failed for blueprint {bp_key}: {e}")

    def test_decision_path_flowchart_all_sample_scales(self):
        """Verifies render_decision_path_flowchart handles small, medium, and large sample scale gates."""
        from src.ui_components.selector_widgets import render_decision_path_flowchart
        from src.selector.selector_engine import route_universal_method_recommendation
        from unittest.mock import MagicMock
        import streamlit as st

        col1, col2 = MagicMock(), MagicMock()
        st.columns = MagicMock(return_value=[col1, col2])
        st.markdown = MagicMock()

        for scale in ["small", "medium", "large"]:
            form_inputs = {
                "domain": "healthcare_clinical",
                "analytical_goal": "compare_groups",
                "study_design": "parallel_rct",
                "sample_scale": scale,
                "problem_key": "two_group_comparison"
            }
            rec = route_universal_method_recommendation(
                domain="healthcare_clinical",
                analytical_goal="compare_groups",
                study_design="parallel_rct",
                sample_scale=scale,
                problem_key="two_group_comparison"
            )
            try:
                render_decision_path_flowchart(rec, form_inputs)
            except Exception as e:
                self.fail(f"render_decision_path_flowchart failed for scale {scale}: {e}")

    def test_decision_path_flowchart_missing_key_safety(self):
        """Verifies render_decision_path_flowchart safely handles empty or partial dictionaries without KeyError."""
        from src.ui_components.selector_widgets import render_decision_path_flowchart
        from unittest.mock import MagicMock
        import streamlit as st

        col1, col2 = MagicMock(), MagicMock()
        st.columns = MagicMock(return_value=[col1, col2])
        st.markdown = MagicMock()

        try:
            render_decision_path_flowchart({}, {})
        except Exception as e:
            self.fail(f"render_decision_path_flowchart crashed on empty inputs: {e}")

    def test_render_html_strips_indentation_and_blank_lines(self):
        """Verifies render_html completely strips leading indentation and blank lines to avoid CommonMark code blocks."""
        from src.ui_components.selector_widgets import render_html
        from unittest.mock import MagicMock
        import streamlit as st

        # Mock st.html and st.markdown
        st.html = MagicMock()
        st.markdown = MagicMock()

        sample_html = """
            <div style="color: red;">
                <span>Line 1</span>

                <span>Line 2</span>
            </div>
        """
        render_html(sample_html)

        # Must have called st.html
        self.assertTrue(st.html.called)
        called_arg = st.html.call_args[0][0]
        for line in called_arg.splitlines():
            self.assertFalse(line.startswith(" "), f"Line starts with whitespace: {line}")
        self.assertNotIn("\n\n", called_arg)

        # Now test fallback when st.html does not exist
        delattr(st, "html")
        st.markdown.reset_mock()
        render_html(sample_html)
        self.assertTrue(st.markdown.called)
        called_arg_md = st.markdown.call_args[0][0]
        for line in called_arg_md.splitlines():
            self.assertFalse(line.startswith(" "), f"Markdown fallback line starts with whitespace: {line}")
        self.assertTrue(st.markdown.call_args[1].get("unsafe_allow_html"))

    def test_unsupervised_clustering_sample_size_routing_and_guidance(self):
        """Verifies that selecting unsupervised clustering routes to cluster stability guidance rather than classical t-test."""
        from src.selector.selector_engine import route_universal_method_recommendation
        from src.ui_components.selector_widgets import render_recommendation_card
        from unittest.mock import MagicMock
        import streamlit as st

        # Route recommendation for unsupervised phenotyping
        rec = route_universal_method_recommendation(
            domain="healthcare_clinical",
            problem_key="unsupervised_phenotyping",
            sample_scale="medium",
            governance_priority="exploratory_discovery"
        )

        self.assertEqual(rec["primary_method_key"], "kmeans_clustering")
        self.assertEqual(rec["sample_size_framework"], "unsupervised_clustering")
        self.assertIn("Silhouette, Davies-Bouldin & Calinski-Harabasz", rec["sample_size_label"])

        # Deep link should route to Machine Learning page
        self.assertEqual(rec["sample_size_deep_link"]["page_route"], "pages/3_🤖_Machine_Learning.py")
        self.assertEqual(rec["sample_size_deep_link"]["session_state_updates"]["ml_direct_model_key"], "kmeans_clustering")

        # Checklist check 1 should focus on cluster stability rather than t-test
        self.assertIn("Cluster", rec["checklist"][0]["title"])
        self.assertIn("Silhouette", rec["checklist"][0]["guidance"])
        self.assertIn("Davies-Bouldin", rec["checklist"][0]["guidance"])
        self.assertIn("Calinski-Harabasz", rec["checklist"][0]["guidance"])

        # Render recommendation card and verify it renders stability guidance rather than hypothesis test
        st.markdown = MagicMock()
        st.html = MagicMock()
        st.button = MagicMock(return_value=False)
        st.caption = MagicMock()
        st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])

        render_recommendation_card(rec)
        self.assertTrue(st.html.called or st.markdown.called)

    def test_time_series_forecasting_sizing_guidance(self):
        """Verifies time series forecasting routes to temporal sequence length sizing and NOT two-sample t-test."""
        from src.selector.selector_engine import route_universal_method_recommendation
        from src.ui_components.selector_widgets import render_recommendation_card
        from unittest.mock import MagicMock
        import streamlit as st

        # Route recommendation for time series forecasting
        rec = route_universal_method_recommendation(
            domain="supply_chain",
            analytical_goal="demand_forecasting",
            study_design="longitudinal_series",
            sample_scale="medium"
        )

        self.assertEqual(rec["problem_key"], "time_series_forecasting")
        self.assertEqual(rec["primary_method_key"], "holt_winters_exponential_smoothing")
        self.assertEqual(rec["sample_size_framework"], "time_series_capacity")
        self.assertIn("Sequential Time Series Sequence Length", rec["sample_size_label"])
        self.assertNotIn("t-test", rec["sample_size_label"].lower())
        self.assertNotIn("two independent means", rec["sample_size_label"].lower())

        # Deep link should route to Machine Learning page
        self.assertEqual(rec["sample_size_deep_link"]["page_route"], "pages/3_🤖_Machine_Learning.py")
        self.assertEqual(rec["sample_size_deep_link"]["session_state_updates"]["ml_direct_model_key"], "holt_winters_exponential_smoothing")

        # Checklist check 1 should focus on seasonal cycles and rolling-origin rather than t-test
        self.assertIn("Seasonal", rec["checklist"][0]["title"])
        self.assertIn("seasonal cycles", rec["checklist"][0]["guidance"].lower())
        self.assertIn("rolling-origin", rec["checklist"][0]["guidance"].lower())

        # Render recommendation card and verify it renders time series sequence guidance rather than t-test
        st.markdown = MagicMock()
        st.html = MagicMock()
        st.button = MagicMock(return_value=False)
        st.caption = MagicMock()
        st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])

        render_recommendation_card(rec)
        self.assertTrue(st.html.called or st.markdown.called)
        # Check rendered HTML
        rendered_content = ""
        if st.html.called:
            rendered_content += "".join([str(arg[0]) for arg in st.html.call_args_list if arg[0]])
        if st.markdown.called:
            rendered_content += "".join([str(arg[0]) for arg in st.markdown.call_args_list if arg[0]])
        self.assertIn("Time Series Sequence Sizing", rendered_content)
        self.assertNotIn("Continuous: Two Independent Means", rendered_content)


if __name__ == "__main__":
    unittest.main()



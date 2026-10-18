"""Unit and Contract Verification Test Suite for Unified Sample Size Deep-Link Badges.

Validates:
1. Mapping completeness between analytical methods (Stats, Epi, ML) and Page 4 Sizing Frameworks.
2. Badge rendering without exceptions across all supported method families.
3. Session state contract execution (custom scenario, framework key, and return route).
4. Clustering stability guidance handling.
5. Strict AST expander compliance.
"""

import os
import sys
import ast
import unittest
from unittest.mock import MagicMock

# Mock streamlit before importing UI widgets if not in environment
if "streamlit" not in sys.modules:
    mock_st = MagicMock()
    mock_st.session_state = {}
    sys.modules["streamlit"] = mock_st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ui_components.sample_size_badge import (
    SAMPLE_SIZE_FRAMEWORK_MAP,
    render_sample_size_deep_link_badge
)
from src.stats_calcs.decision_schema import STATISTICAL_TESTS


class TestSampleSizeFrameworkMap(unittest.TestCase):
    """Verifies the integrity of the cross-pillar sample size mapping."""

    VALID_PAGE4_CUSTOM_OPTIONS = {
        "Continuous: Two Independent Means (t-test)",
        "Continuous: Paired / Pre-Post Means",
        "Continuous: One-Way ANOVA (Multiple Groups)",
        "Continuous: Pearson Correlation (r)",
        "Binary: Two Independent Proportions",
        "Binary: Prospective Cohort (Relative Risk)",
        "Binary: Retrospective Case-Control (Odds Ratio)",
        "Survival: Cox Log-Rank Test (Hazard Ratio)",
        "Diagnostic: Target Sensitivity Precision"
    }

    def test_framework_map_entries_structure(self):
        """Verifies every mapped framework has required fields and points to valid Page 4 options."""
        for m_key, info in SAMPLE_SIZE_FRAMEWORK_MAP.items():
            self.assertIn("custom_design", info, f"Missing custom_design in {m_key}")
            self.assertIn("badge_title", info, f"Missing badge_title in {m_key}")
            self.assertIn("rationale", info, f"Missing rationale in {m_key}")
            self.assertIn(
                info["custom_design"],
                self.VALID_PAGE4_CUSTOM_OPTIONS,
                f"Framework {info['custom_design']} for {m_key} is not in Page 4 custom_options!"
            )

    def test_all_18_statistical_tests_have_sample_size_mapping(self):
        """Verifies all 18 classical statistical tests have explicit sample size mappings."""
        for test_key in STATISTICAL_TESTS.keys():
            self.assertIn(
                test_key,
                SAMPLE_SIZE_FRAMEWORK_MAP,
                f"Statistical test '{test_key}' missing from SAMPLE_SIZE_FRAMEWORK_MAP!"
            )

    def test_core_epidemiological_methods_have_mapping(self):
        """Verifies primary epidemiological study designs have mappings."""
        core_epi = [
            "relative_risk", "odds_ratio", "risk_difference", "incidence_rate_ratio",
            "attributable_fraction", "diagnostic_accuracy", "predictive_values",
            "likelihood_ratios", "kaplan_meier", "cox_regression", "mantel_haenszel",
            "outbreak_investigation", "meta_analysis", "direct_standardization"
        ]
        for e_key in core_epi:
            self.assertIn(
                e_key,
                SAMPLE_SIZE_FRAMEWORK_MAP,
                f"Epidemiology method '{e_key}' missing from SAMPLE_SIZE_FRAMEWORK_MAP!"
            )


class TestSampleSizeBadgeRendering(unittest.TestCase):
    """Verifies that the badge component renders without exceptions across pillars."""

    def setUp(self):
        import src.ui_components.sample_size_badge as badge_mod
        self.mock_st = MagicMock()
        self.mock_st.session_state = {}
        self.mock_st.markdown = MagicMock()
        self.mock_st.button = MagicMock(return_value=False)
        self.mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
        self.mock_st.switch_page = MagicMock()
        badge_mod.st = self.mock_st

    def test_render_statistical_test_badge(self):
        """Verifies badge renders for Welch's t-test."""
        try:
            render_sample_size_deep_link_badge(
                method_key="welch_t_test",
                method_name="Welch's Independent t-test",
                pillar="Classical Statistics",
                origin_page="pages/1_📊_Statistical_Tests.py",
                origin_title="Statistical Tests",
                origin_icon="📊"
            )
        except Exception as e:
            self.fail(f"render_sample_size_deep_link_badge crashed for Welch's t-test: {e}")

    def test_render_epidemiology_badge(self):
        """Verifies badge renders for Cohort Relative Risk."""
        try:
            render_sample_size_deep_link_badge(
                method_key="relative_risk",
                method_name="Relative Risk (Risk Ratio)",
                pillar="Epidemiological Methods",
                origin_page="pages/2_🏥_Epidemiology_Methods.py",
                origin_title="Epidemiology Methods",
                origin_icon="🏥"
            )
        except Exception as e:
            self.fail(f"render_sample_size_deep_link_badge crashed for Relative Risk: {e}")

    def test_render_clustering_stability_guidance(self):
        """Verifies clustering renders geometric stability guidance rather than classical t-test."""
        self.mock_st.markdown.reset_mock()
        render_sample_size_deep_link_badge(
            method_key="kmeans_clustering",
            method_name="K-Means Clustering",
            pillar="Machine Learning & AI",
            origin_page="pages/3_🤖_Machine_Learning.py",
            origin_title="Machine Learning",
            origin_icon="🤖"
        )
        self.assertTrue(self.mock_st.markdown.called)
        called_html = str(self.mock_st.markdown.call_args[0][0])
        self.assertIn("Cluster Stability", called_html)
        self.assertIn("Silhouette", called_html)

    def test_render_time_series_sequence_guidance(self):
        """Verifies time series models render sequential sequence guidance rather than classical t-test."""
        self.mock_st.markdown.reset_mock()
        render_sample_size_deep_link_badge(
            method_key="holt_winters_exponential_smoothing",
            method_name="Holt-Winters Triple Exponential Smoothing",
            pillar="Machine Learning & AI",
            origin_page="pages/3_🤖_Machine_Learning.py",
            origin_title="Machine Learning",
            origin_icon="🤖"
        )
        self.assertTrue(self.mock_st.markdown.called)
        called_html = str(self.mock_st.markdown.call_args[0][0])
        self.assertIn("Time Series Sequence Sizing", called_html)
        self.assertIn("seasonal cycles", called_html)
        self.assertNotIn("Continuous: Two Independent Means", called_html)

    def test_button_click_sets_session_state_and_navigates(self):
        """Verifies clicking the badge button sets Page 4 session state and switches page."""
        self.mock_st.button = MagicMock(return_value=True)

        render_sample_size_deep_link_badge(
            method_key="odds_ratio",
            method_name="Case-Control Odds Ratio",
            pillar="Epidemiological Methods",
            origin_page="pages/2_🏥_Epidemiology_Methods.py",
            origin_title="Epidemiology Methods",
            origin_icon="🏥"
        )

        self.assertEqual(self.mock_st.session_state.get("sample_size_scenario"), "custom")
        self.assertEqual(
            self.mock_st.session_state.get("sample_size_custom_design"),
            "Binary: Retrospective Case-Control (Odds Ratio)"
        )
        consultation = self.mock_st.session_state.get("selector_consultation", {})
        self.assertTrue(consultation.get("active"))
        self.assertEqual(consultation.get("return_page"), "pages/2_🏥_Epidemiology_Methods.py")
        self.assertEqual(consultation.get("return_label"), "← Return to Epidemiology Methods")
        self.assertTrue(self.mock_st.switch_page.called)
        self.assertEqual(self.mock_st.switch_page.call_args[0][0], "pages/4_📐_Sample_Size_and_Power.py")


class TestBadgeASTCompliance(unittest.TestCase):
    """Enforces strict AST expander compliance."""

    def test_badge_file_expander_compliance(self):
        """Verifies no forbidden expanded=True in sample_size_badge.py."""
        file_path = os.path.join(PROJECT_ROOT, "src", "ui_components", "sample_size_badge.py")
        self.assertTrue(os.path.exists(file_path))
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("expanded" + "=True", content)


if __name__ == "__main__":
    unittest.main()

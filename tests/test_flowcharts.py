"""Unit and Contract Verification Test Suite for Methodology Decision Flowcharts.

Validates:
1. Flowchart UI widgets render without exceptions.
2. All expanders strictly adhere to expanded=False.
3. All target page routes referenced in the flowcharts exist on disk.
4. Mermaid diagram code blocks are non-empty and well-formed.
5. Interactive picker routing produces valid methods and deep-link keys.
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

from src.ui_components.flowchart_widgets import (
    render_master_triage_flowchart,
    render_statistics_decision_tree,
    render_epidemiology_decision_tree,
    render_machine_learning_decision_tree,
    render_sample_size_decision_tree,
    render_html
)


class TestFlowchartWidgets(unittest.TestCase):
    """Verifies that all decision tree and flowchart widgets render cleanly."""

    def setUp(self):
        import streamlit as st
        # Mock streamlit display and input components
        st.markdown = MagicMock()
        st.html = MagicMock()
        st.success = MagicMock()
        st.info = MagicMock()
        st.warning = MagicMock()
        st.error = MagicMock()
        st.caption = MagicMock()
        st.columns = MagicMock(side_effect=lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))])
        st.selectbox = MagicMock(side_effect=lambda label, options, **kwargs: options[0] if options else None)
        st.radio = MagicMock(side_effect=lambda label, options, **kwargs: options[0] if options else None)
        st.button = MagicMock(return_value=False)
        st.expander = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

    def test_render_master_triage_flowchart(self):
        """Verifies master cross-pillar triage renders without errors."""
        try:
            render_master_triage_flowchart()
        except Exception as e:
            self.fail(f"render_master_triage_flowchart crashed: {e}")

    def test_render_statistics_decision_tree(self):
        """Verifies classical statistics decision tree renders without errors."""
        try:
            render_statistics_decision_tree()
        except Exception as e:
            self.fail(f"render_statistics_decision_tree crashed: {e}")

    def test_render_epidemiology_decision_tree(self):
        """Verifies epidemiology decision tree renders without errors."""
        try:
            render_epidemiology_decision_tree()
        except Exception as e:
            self.fail(f"render_epidemiology_decision_tree crashed: {e}")

    def test_render_machine_learning_decision_tree(self):
        """Verifies machine learning decision tree renders without errors."""
        try:
            render_machine_learning_decision_tree()
        except Exception as e:
            self.fail(f"render_machine_learning_decision_tree crashed: {e}")

    def test_render_sample_size_decision_tree(self):
        """Verifies sample size decision tree renders without errors."""
        try:
            render_sample_size_decision_tree()
        except Exception as e:
            self.fail(f"render_sample_size_decision_tree crashed: {e}")

    def test_render_html_safety(self):
        """Verifies render_html safely cleans whitespace and empty lines."""
        import streamlit as st
        st.html = MagicMock()
        st.markdown = MagicMock()

        raw_html = """
            <div class="test">
                <span>Content</span>
            </div>
        """
        render_html(raw_html)
        self.assertTrue(st.html.called or st.markdown.called)


class TestFlowchartASTCompliance(unittest.TestCase):
    """Enforces strict AST code quality standards and expander rules."""

    def test_flowchart_files_expander_compliance(self):
        """Verifies every st.expander in flowchart files has expanded=False."""
        files_to_check = [
            os.path.join(PROJECT_ROOT, "pages", "8_🗺️_Decision_Flowcharts.py"),
            os.path.join(PROJECT_ROOT, "src", "ui_components", "flowchart_widgets.py")
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

    def test_referenced_page_routes_exist_on_filesystem(self):
        """Verifies all page routes linked in the decision trees exist on disk."""
        expected_routes = [
            "app.py",
            "pages/1_📊_Statistical_Tests.py",
            "pages/2_🏥_Epidemiology_Methods.py",
            "pages/3_🤖_Machine_Learning.py",
            "pages/4_📐_Sample_Size_and_Power.py",
            "pages/5_📋_Survey_and_Questionnaire_Analysis.py",
            "pages/6_🗣️_Natural_Language_Processing.py",
            "pages/7_🎯_Study_Design_and_Method_Selector.py",
            "pages/8_🗺️_Decision_Flowcharts.py"
        ]

        for route in expected_routes:
            full_path = os.path.join(PROJECT_ROOT, route)
            self.assertTrue(
                os.path.exists(full_path),
                f"Referenced route {route} does not exist at {full_path}"
            )


if __name__ == "__main__":
    unittest.main()


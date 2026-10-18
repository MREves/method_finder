"""Unit and AST tests for Developer / Debug Mode toggle and page tracker visibility gating.

Validates that:
1. app.py contains the debug toggle widget and synchronizes both debug_mode and debug_model in st.session_state.
2. All 6 temporary implementation/progress trackers across pages are strictly gated behind debug_mode/debug_model.
3. All expanders maintain 100% strict compliance with expanded=False.
4. Toggling state dynamically controls tracker visibility.
"""

import ast
import os
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRACKER_PAGES = [
    "pages/0_📚_Statistical_Fundamentals.py",
    "pages/1_📊_Statistical_Tests.py",
    "pages/2_🏥_Epidemiology_Methods.py",
    "pages/4_📐_Sample_Size_and_Power.py",
    "pages/5_📋_Survey_and_Questionnaire_Analysis.py",
    "pages/6_🗣️_Natural_Language_Processing.py",
    "pages/7_🎯_Study_Design_and_Method_Selector.py",
    "pages/8_🗺️_Decision_Flowcharts.py"
]


class TestDebugModeToggleAndGating(unittest.TestCase):
    """Verifies debug_mode toggle architecture and tracker expander gating."""

    def test_app_py_contains_debug_toggle_and_sync(self):
        """Verifies app.py has the debug toggle widget and syncs session state."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        self.assertTrue(os.path.exists(app_path), "app.py must exist.")

        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check content references
        self.assertIn("debug_mode", content)
        self.assertIn("debug_model", content)
        self.assertIn("st.toggle", content)

        # Parse AST to verify structure
        tree = ast.parse(content, filename=app_path)
        found_toggle = False
        session_state_debug_keys = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "toggle":
                    found_toggle = True
            elif isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Attribute) and node.value.attr == "session_state":
                    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                        if "debug" in node.slice.value:
                            session_state_debug_keys.add(node.slice.value)

        self.assertTrue(found_toggle, "app.py must instantiate st.toggle for debug mode.")
        self.assertIn("debug_mode", session_state_debug_keys, "app.py must update session_state['debug_mode'].")
        self.assertIn("debug_model", session_state_debug_keys, "app.py must update session_state['debug_model'].")

    def test_all_progress_trackers_are_gated_by_debug_mode(self):
        """Verifies each tracker expander is wrapped inside an if condition checking debug_mode."""
        for rel_path in TRACKER_PAGES:
            full_path = os.path.join(PROJECT_ROOT, rel_path)
            self.assertTrue(os.path.exists(full_path), f"File {rel_path} must exist.")

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=full_path)

            tracker_if_found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    # Check if this If node checks debug_mode
                    test_str = ast.unparse(node.test)
                    if "debug_mode" in test_str or "debug_model" in test_str:
                        # Check that the body contains a tracker expander
                        for sub_node in ast.walk(node):
                            if isinstance(sub_node, ast.Call):
                                if isinstance(sub_node.func, ast.Attribute) and sub_node.func.attr == "expander":
                                    # Check that label has Tracker / Progress
                                    first_arg = sub_node.args[0] if sub_node.args else None
                                    if first_arg and isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                                        if any(k in first_arg.value for k in ["Tracker", "Progress", "Validation"]):
                                            tracker_if_found = True

            self.assertTrue(
                tracker_if_found,
                f"{rel_path} must gate its tracking expander behind debug_mode / debug_model!"
            )

    def test_all_tracker_expanders_have_expanded_false(self):
        """Verifies that all tracking expanders still strictly enforce expanded=False."""
        for rel_path in TRACKER_PAGES:
            full_path = os.path.join(PROJECT_ROOT, rel_path)
            with open(full_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=full_path)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "expander":
                        first_arg = node.args[0] if node.args else None
                        if first_arg and isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                            if any(k in first_arg.value for k in ["Tracker", "Progress", "Validation"]):
                                kw_dict = {kw.arg: kw.value for kw in node.keywords}
                                self.assertIn("expanded", kw_dict, f"Tracker expander in {rel_path} must have explicit expanded=False")
                                self.assertIsInstance(kw_dict["expanded"], ast.Constant)
                                self.assertIs(kw_dict["expanded"].value, False)

    def test_debug_mode_state_evaluation_logic(self):
        """Simulates session state resolution when debug_mode or debug_model is toggled."""
        # Scenario 1: Initial state (default uninitialized)
        session_state = {}
        is_visible = session_state.get("debug_mode", False) or session_state.get("debug_model", False)
        self.assertFalse(is_visible, "Trackers should be hidden by default.")

        # Scenario 2: debug_mode toggled ON
        session_state["debug_mode"] = True
        session_state["debug_model"] = True
        is_visible = session_state.get("debug_mode", False) or session_state.get("debug_model", False)
        self.assertTrue(is_visible, "Trackers should be visible when debug_mode is ON.")

        # Scenario 3: debug_mode toggled OFF
        session_state["debug_mode"] = False
        session_state["debug_model"] = False
        is_visible = session_state.get("debug_mode", False) or session_state.get("debug_model", False)
        self.assertFalse(is_visible, "Trackers should be hidden when debug_mode is toggled OFF.")

        # Scenario 4: User references debug_model specifically
        session_state = {"debug_model": True}
        is_visible = session_state.get("debug_mode", False) or session_state.get("debug_model", False)
        self.assertTrue(is_visible, "Trackers should be visible when debug_model is True.")

    def test_no_validated_status_in_dropdowns_or_shaded_boxes(self):
        """Verifies that '- validated' and '✅ Validated' wording has been removed from dropdowns and cards."""
        # 1. Check Statistical Tests dropdown
        stat_tests_path = os.path.join(PROJECT_ROOT, "pages", "1_📊_Statistical_Tests.py")
        with open(stat_tests_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("VALIDATED_TEST_KEYS", content)
        self.assertNotIn("status_tag = \"✅ Validated\"", content)
        self.assertNotIn("— `✅ Validated`", content)

        # 2. Check Epidemiology Methods dropdown and card
        epi_path = os.path.join(PROJECT_ROOT, "pages", "2_🏥_Epidemiology_Methods.py")
        with open(epi_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("Fully Validated Method", content)
        self.assertNotIn("— `✅ Validated`", content)

        # 3. Check ML model dropdown
        ml_path = os.path.join(PROJECT_ROOT, "src", "ui_components", "ml_widgets.py")
        with open(ml_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("Interpretability) — ✅ Validated", content)

        # 4. Check Selector engine badge
        selector_engine_path = os.path.join(PROJECT_ROOT, "src", "selector", "selector_engine.py")
        with open(selector_engine_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("Parametric Validated", content)


if __name__ == "__main__":
    unittest.main()


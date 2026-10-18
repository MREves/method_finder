"""Unit tests for Kaplan-Meier Survival Analysis and Log-Rank Test."""

import unittest

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    from src.epi_methods.kaplan_meier import (
        calculate_kaplan_meier_single_group,
        calculate_log_rank_test,
        calculate_landmark_survival_probability,
        calculate_kaplan_meier_two_groups,
        evaluate_kaplan_meier_prerequisites,
        generate_kaplan_meier_demo_data
    )
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack (numpy, scipy, pandas) required for Kaplan-Meier tests")
class TestKaplanMeierCalculations(unittest.TestCase):
    """Test suite for mathematical and epidemiological correctness of Kaplan-Meier calculations."""

    def setUp(self):
        # Known textbook sample
        self.durations_1 = np.array([4.0, 5.0, 6.0, 8.0, 10.0, 12.0])
        self.events_1 = np.array([1, 0, 1, 1, 0, 1])

        # Second comparative group
        self.durations_2 = np.array([2.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        self.events_2 = np.array([1, 1, 0, 1, 1, 1])

    def test_single_group_survival_probabilities(self):
        """Tests that Kaplan-Meier product-limit formula matches exact hand calculations."""
        res = calculate_kaplan_meier_single_group(self.durations_1, self.events_1)
        times = res["table_times"]
        probs = res["survival_probabilities"]

        # Check initial state
        self.assertEqual(times[0], 0.0)
        self.assertAlmostEqual(probs[0], 1.0, places=5)

        # t = 4: d=1, n=6 -> S(4) = 5/6 = 0.833333
        idx_4 = times.index(4.0)
        self.assertAlmostEqual(probs[idx_4], 5.0 / 6.0, places=5)

        # t = 5: d=0, c=1, n=5 -> S(5) = 5/6
        idx_5 = times.index(5.0)
        self.assertAlmostEqual(probs[idx_5], 5.0 / 6.0, places=5)

        # t = 6: d=1, c=0, n=4 -> S(6) = (5/6)*(3/4) = 5/8 = 0.625
        idx_6 = times.index(6.0)
        self.assertAlmostEqual(probs[idx_6], 0.625, places=5)

        # t = 8: d=1, c=0, n=3 -> S(8) = (5/8)*(2/3) = 5/12 = 0.416667
        idx_8 = times.index(8.0)
        self.assertAlmostEqual(probs[idx_8], 5.0 / 12.0, places=5)

        # t = 10: d=0, c=1, n=2 -> S(10) = 5/12
        idx_10 = times.index(10.0)
        self.assertAlmostEqual(probs[idx_10], 5.0 / 12.0, places=5)

        # t = 12: d=1, c=0, n=1 -> S(12) = 0.0
        idx_12 = times.index(12.0)
        self.assertAlmostEqual(probs[idx_12], 0.0, places=5)

    def test_greenwood_standard_error(self):
        """Tests that Greenwood's formula yields the exact expected analytical standard error."""
        res = calculate_kaplan_meier_single_group(self.durations_1, self.events_1)
        times = res["table_times"]
        ses = res["se_linear"]

        # At t=4: Var = (5/6)^2 * [1/(6*5)] = (25/36)*(1/30) = 5/216
        expected_se_4 = np.sqrt(5.0 / 216.0)
        idx_4 = times.index(4.0)
        self.assertAlmostEqual(ses[idx_4], expected_se_4, places=5)

    def test_median_survival_time(self):
        """Tests that median survival time identifies the earliest time S(t) drops to <= 0.50."""
        res = calculate_kaplan_meier_single_group(self.durations_1, self.events_1)
        # S(6) = 0.625 > 0.5, S(8) = 0.4167 <= 0.5 -> Median = 8.0
        self.assertEqual(res["median_survival_time"], 8.0)
        self.assertTrue(res["median_reached"])

    def test_log_rank_test_two_groups(self):
        """Tests two-group Log-Rank test observed vs expected calculation and chi-square."""
        lr = calculate_log_rank_test(
            self.durations_1, self.events_1,
            self.durations_2, self.events_2
        )
        self.assertEqual(lr["degrees_of_freedom"], 1)
        self.assertGreater(lr["chisq"], 0.0)
        self.assertGreaterEqual(lr["p_value"], 0.0)
        self.assertLessEqual(lr["p_value"], 1.0)
        self.assertEqual(lr["n_group_1"], 6)
        self.assertEqual(lr["n_group_2"], 6)

        # Peto Hazard Ratio must be positive
        self.assertGreater(lr["peto_hazard_ratio"], 0.0)
        ci_low, ci_upp = lr["peto_hr_ci"]
        self.assertLess(ci_low, lr["peto_hazard_ratio"])
        self.assertGreater(ci_upp, lr["peto_hazard_ratio"])

    def test_landmark_survival_probability(self):
        """Tests landmark survival extraction at intermediate time."""
        res = calculate_kaplan_meier_single_group(self.durations_1, self.events_1)
        lm_7 = calculate_landmark_survival_probability(res, 7.0)
        # At t=7, survival is S(6) = 0.625
        self.assertAlmostEqual(lm_7["survival_probability"], 0.625, places=5)
        self.assertEqual(lm_7["landmark_time"], 7.0)

    def test_two_groups_full_pipeline(self):
        """Tests full two-group comparative survival pipeline."""
        df = pd.DataFrame({
            "Time": np.concatenate([self.durations_1, self.durations_2]),
            "Event": np.concatenate([self.events_1, self.events_2]),
            "Group": ["Group_A"] * 6 + ["Group_B"] * 6
        })
        res = calculate_kaplan_meier_two_groups(df, time_col="Time", event_col="Event", group_col="Group")
        self.assertIn("km_group_1", res)
        self.assertIn("km_group_2", res)
        self.assertIn("log_rank", res)
        self.assertIn("landmarks_df", res)
        self.assertFalse(res["landmarks_df"].empty)

    def test_prerequisite_evaluations(self):
        """Tests that prerequisite checks detect balanced censoring and adequate events."""
        df_micro, df_summary, meta = generate_kaplan_meier_demo_data("oncology_immunotherapy")
        res = calculate_kaplan_meier_two_groups(df_micro, time_col=meta["time_col"], event_col=meta["event_col"], group_col=meta["group_col"])
        evals = evaluate_kaplan_meier_prerequisites(res, df_micro, time_col=meta["time_col"], event_col=meta["event_col"], group_col=meta["group_col"])

        self.assertIn("independent_censoring", evals)
        self.assertIn("event_count", evals)
        self.assertIn("proportional_hazards", evals)
        self.assertIn("tail_stability", evals)
        self.assertIn("censoring_range", evals)

    def test_demo_scenarios_generation(self):
        """Tests synthetic data generators across all three clinical scenarios."""
        scenarios = ["oncology_immunotherapy", "cardiac_surgery_graft", "pediatric_leukemia_induction"]
        for sc in scenarios:
            df_micro, df_summary, meta = generate_kaplan_meier_demo_data(sc)
            self.assertFalse(df_micro.empty)
            self.assertFalse(df_summary.empty)
            self.assertEqual(meta["scenario_key"], sc)
            self.assertIn("All Meadows", meta["title"] if sc == "oncology_immunotherapy" else meta["description"])
            self.assertEqual(len(meta["landmark_times"]), 4)


if __name__ == "__main__":
    unittest.main()

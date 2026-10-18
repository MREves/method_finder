import unittest
from src.stats_calcs.decision_schema import (
    recommend_statistical_test,
    get_statistical_test_recommendation,
    get_all_statistical_tests,
    STATISTICAL_TESTS
)
from src.utils import format_p_value, format_ci

try:
    import numpy as np
    import scipy.stats as stats
    from src.stats_calcs.synthetic_data import (
        generate_one_sample_demo,
        generate_wilcoxon_demo,
        generate_mann_whitney_demo,
        generate_welch_demo,
        generate_students_t_demo,
        generate_two_independent_demo,
        generate_paired_demo,
        generate_wilcoxon_paired_demo,
        generate_one_way_anova_demo,
        generate_kruskal_wallis_demo,
        generate_repeated_measures_anova_demo,
        generate_friedman_demo,
        generate_three_groups_demo,
        generate_contingency_demo,
        generate_fisher_exact_demo,
        generate_mcnemar_demo,
        generate_correlation_demo,
        generate_pearson_correlation_demo,
        generate_spearman_correlation_demo,
        generate_linear_regression_demo,
        generate_binary_logistic_regression_demo
    )
    HAS_SCIENTIFIC_STACK = True

except ImportError:
    HAS_SCIENTIFIC_STACK = False



class TestStatisticalDecisionRouting(unittest.TestCase):
    """Tests the deterministic routing logic across all hypothesis test scenarios."""

    def test_welch_t_test_recommendation(self):
        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "2",
            "group_design": "independent",
            "dependent_type": "continuous",
            "normality": "normal",
            "equal_variance": "unknown"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "welch_t_test")
        self.assertTrue(rec["parametric"])
        self.assertTrue(rec["excel_guide"]["has_native_formula"])

    def test_student_t_test_recommendation(self):
        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "2",
            "group_design": "independent",
            "dependent_type": "continuous",
            "normality": "normal",
            "equal_variance": "equal"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "students_t_test")
        self.assertTrue(rec["parametric"])

    def test_mann_whitney_u_for_skewed_data(self):
        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "2",
            "group_design": "independent",
            "dependent_type": "continuous",
            "normality": "non_normal"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "mann_whitney_u")
        self.assertFalse(rec["parametric"])

    def test_mann_whitney_u_for_ordinal_data(self):
        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "2",
            "group_design": "independent",
            "dependent_type": "ordinal"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "mann_whitney_u")
        self.assertFalse(rec["parametric"])

    def test_paired_t_test(self):
        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "2",
            "group_design": "paired",
            "dependent_type": "continuous",
            "normality": "normal"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "paired_t_test")
        self.assertTrue(rec["parametric"])

    def test_wilcoxon_signed_rank_one_sample(self):
        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "1",
            "dependent_type": "continuous",
            "normality": "skewed"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "wilcoxon_signed_rank")
        self.assertFalse(rec["parametric"])

    def test_wilcoxon_signed_rank_paired(self):
        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "2",
            "group_design": "paired",
            "dependent_type": "ordinal"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "wilcoxon_paired")
        self.assertFalse(rec["parametric"])

    def test_one_way_anova(self):
        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "3+",
            "group_design": "independent",
            "dependent_type": "continuous",
            "normality": "normal"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "one_way_anova")
        self.assertTrue(rec["parametric"])

    def test_kruskal_wallis(self):
        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "3+",
            "group_design": "independent",
            "dependent_type": "continuous",
            "normality": "non_normal"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "kruskal_wallis")
        self.assertFalse(rec["parametric"])

    def test_repeated_measures_anova(self):
        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "3+",
            "group_design": "paired",
            "dependent_type": "continuous",
            "normality": "normal"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "repeated_measures_anova")
        self.assertTrue(rec["parametric"])

    def test_friedman_test(self):
        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "3+",
            "group_design": "paired",
            "dependent_type": "ordinal"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "friedman_test")
        self.assertFalse(rec["parametric"])

    def test_chi_square_independence(self):
        answers = {
            "primary_objective": "contingency_proportions",
            "table_setup": "two_by_two",
            "cell_counts": "adequate"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "chi_square_independence")
        self.assertFalse(rec["parametric"])

    def test_fisher_exact_small_counts(self):
        answers = {
            "primary_objective": "contingency_proportions",
            "table_setup": "two_by_two",
            "cell_counts": "small"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "fisher_exact")
        self.assertFalse(rec["parametric"])
        self.assertEqual(rec["demo_generator"], "fisher_exact")
        self.assertTrue(rec["excel_guide"]["has_native_formula"])
        self.assertIn("HYPGEOM.DIST", rec["excel_guide"]["formula_syntax"])
        self.assertIn("fisher_exact", rec["python_snippet"])

    def test_mcnemar_paired_binary(self):
        answers = {
            "primary_objective": "contingency_proportions",
            "table_setup": "paired_binary"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "mcnemar_test")

    def test_pearson_correlation(self):
        answers = {
            "primary_objective": "examine_relationship",
            "relationship_goal": "correlation",
            "linearity": "linear"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "pearson_correlation")
        self.assertTrue(rec["parametric"])

    def test_spearman_correlation(self):
        answers = {
            "primary_objective": "examine_relationship",
            "relationship_goal": "correlation",
            "linearity": "monotonic_or_ranked"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "spearman_correlation")
        self.assertFalse(rec["parametric"])

    def test_linear_regression(self):
        answers = {
            "primary_objective": "examine_relationship",
            "relationship_goal": "predict_continuous"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "linear_regression")

    def test_one_sample_t_test_recommendation(self):

        answers = {
            "primary_objective": "compare_groups",
            "num_groups": "1",
            "dependent_type": "continuous",
            "normality": "normal"
        }
        rec = recommend_statistical_test(answers)
        self.assertEqual(rec["test_key"], "one_sample_t_test")
        self.assertTrue(rec["parametric"])
        self.assertEqual(rec["demo_generator"], "one_sample_continuous")


class TestDirectStatisticalMethodAccess(unittest.TestCase):
    """Tests the direct selection and test catalog retrieval methods."""

    def test_get_all_statistical_tests(self):
        tests = get_all_statistical_tests()
        self.assertEqual(len(tests), 19)
        self.assertEqual(len(tests), len(STATISTICAL_TESTS))
        for t in tests:
            self.assertIn("key", t)
            self.assertIn("name", t)
            self.assertIn("category", t)
            self.assertIn("parametric", t)
            self.assertIn("description", t)

    def test_get_statistical_test_recommendation_welch(self):
        rec = get_statistical_test_recommendation("welch_t_test")
        self.assertEqual(rec["test_key"], "welch_t_test")
        self.assertEqual(rec["name"], "Welch's Independent Two-Sample t-test")
        self.assertTrue(rec["parametric"])
        self.assertEqual(rec["demo_generator"], "welch_t_test")
        self.assertIn("excel_guide", rec)
        self.assertIn("python_snippet", rec)
        self.assertTrue(any("Directly selected" in r for r in rec["reasoning"]))

    def test_get_statistical_test_recommendation_repeated_measures_anova(self):
        rec = get_statistical_test_recommendation("repeated_measures_anova")
        self.assertEqual(rec["test_key"], "repeated_measures_anova")
        self.assertTrue(rec["parametric"])
        self.assertEqual(rec["demo_generator"], "repeated_measures_anova")

    def test_get_statistical_test_recommendation_friedman(self):
        rec = get_statistical_test_recommendation("friedman_test")
        self.assertEqual(rec["test_key"], "friedman_test")
        self.assertFalse(rec["parametric"])
        self.assertEqual(rec["demo_generator"], "friedman_test")

    def test_get_statistical_test_recommendation_chi_square(self):
        rec = get_statistical_test_recommendation("chi_square_independence")
        self.assertEqual(rec["test_key"], "chi_square_independence")
        self.assertFalse(rec["parametric"])
        self.assertEqual(rec["demo_generator"], "contingency_table_2x2")

    def test_get_statistical_test_recommendation_fisher_exact(self):
        rec = get_statistical_test_recommendation("fisher_exact")
        self.assertEqual(rec["test_key"], "fisher_exact")
        self.assertFalse(rec["parametric"])
        self.assertEqual(rec["demo_generator"], "fisher_exact")
        self.assertIn("Fisher's Exact Test", rec["name"])

    def test_get_statistical_test_recommendation_fallback(self):
        rec = get_statistical_test_recommendation("non_existent_key_xyz")
        self.assertEqual(rec["test_key"], "welch_t_test")
        self.assertTrue(rec["parametric"])


class TestSyntheticDataGenerators(unittest.TestCase):
    """Tests that demonstration data generators execute reliably."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("numpy/scipy/pandas not installed in this environment")

    def test_one_sample_demo_normal_passes_assumptions(self):
        df, results = generate_one_sample_demo(
            n=40, sample_mean=105.0, sample_std=10.0, benchmark=100.0,
            dist_type="normal", outlier_count=0, seed=42
        )
        self.assertEqual(len(df), 40)
        self.assertEqual(results["df"], 39)
        self.assertEqual(results["overall_status"], "pass")
        self.assertIn("One-Sample t-test", results["overall_recommendation"])
        self.assertTrue(results["ci_lower"] < results["mean"] < results["ci_upper"])

    def test_one_sample_demo_skewed_violates_assumptions(self):
        df, results = generate_one_sample_demo(
            n=20, sample_mean=105.0, sample_std=10.0, benchmark=100.0,
            dist_type="skewed_right", outlier_count=0, seed=42
        )
        self.assertEqual(results["overall_status"], "fail")
        self.assertIn("Wilcoxon", results["overall_recommendation"])

    def test_one_sample_demo_extreme_outliers_violates_assumptions(self):
        df, results = generate_one_sample_demo(
            n=35, sample_mean=100.0, sample_std=10.0, benchmark=100.0,
            dist_type="normal", outlier_count=2, seed=42
        )
        self.assertEqual(results["overall_status"], "fail")
        self.assertGreaterEqual(results["extreme_outliers"], 1)
        self.assertIn("Wilcoxon", results["overall_recommendation"])

    def test_wilcoxon_demo_symmetric_passes_assumptions(self):
        df, results = generate_wilcoxon_demo(
            n=30, sample_median=108.0, dist_shape="symmetric_non_normal", seed=42
        )
        self.assertEqual(len(df), 30)
        self.assertEqual(results["overall_status"], "pass")
        self.assertGreaterEqual(results["w_stat"], 0.0)
        self.assertTrue(0.0 <= results["p_wilcoxon"] <= 1.0)
        self.assertIn("Wilcoxon Signed-Rank Test", results["overall_recommendation"])

    def test_wilcoxon_demo_severely_skewed_violates_symmetry(self):
        df, results = generate_wilcoxon_demo(
            n=30, dist_shape="severely_skewed", seed=42
        )
        self.assertEqual(results["overall_status"], "fail")
        self.assertTrue(0.0 <= results["p_sign_test"] <= 1.0)
        self.assertIn("Sign Test", results["overall_recommendation"])

    def test_wilcoxon_demo_zeros_handling(self):
        df, results = generate_wilcoxon_demo(
            n=30, zero_diff_count=4, seed=42
        )
        self.assertEqual(results["n_zeros"], 4)
        self.assertEqual(results["n_eff"], 26)

    def test_wilcoxon_paired_demo_continuous_symmetric(self):
        df, results = generate_wilcoxon_paired_demo(
            n_pairs=30, dist_shape="symmetric_non_normal", data_type="continuous_non_normal", seed=42
        )
        self.assertEqual(len(df), 30)
        self.assertEqual(results["overall_status"], "pass")
        self.assertTrue(0.0 <= results["p_wilcoxon"] <= 1.0)
        self.assertIn("Wilcoxon Signed-Rank Test (Paired)", results["overall_recommendation"])

    def test_wilcoxon_paired_demo_ordinal_likert(self):
        df, results = generate_wilcoxon_paired_demo(
            n_pairs=30, data_type="ordinal_likert", seed=42
        )
        self.assertEqual(len(df), 30)
        self.assertEqual(results["data_type"], "ordinal_likert")
        self.assertIn("Ordinal (Likert Scale)", results["assumptions_eval"][1]["diagnostic_value"])

    def test_wilcoxon_paired_demo_severely_skewed_violates_symmetry(self):
        df, results = generate_wilcoxon_paired_demo(
            n_pairs=30, dist_shape="severely_skewed", seed=42
        )
        self.assertEqual(results["overall_status"], "fail")
        self.assertTrue(0.0 <= results["p_sign_test"] <= 1.0)
        self.assertIn("Sign Test", results["overall_recommendation"])

    def test_wilcoxon_paired_demo_zeros_handling(self):
        df, results = generate_wilcoxon_paired_demo(
            n_pairs=30, zero_diff_count=4, seed=42
        )
        self.assertEqual(results["n_zeros"], 4)
        self.assertEqual(results["n_eff"], 26)

    def test_mann_whitney_demo_symmetric_passes_assumptions(self):
        df, results = generate_mann_whitney_demo(
            n_a=30, n_b=30, median_a=100.0, median_b=112.0, dist_shape="symmetric_non_normal", seed=42
        )
        self.assertEqual(len(df), 60)
        self.assertEqual(results["overall_status"], "pass")
        self.assertGreaterEqual(results["u_stat"], 0.0)
        self.assertTrue(0.0 <= results["p_mann"] <= 1.0)
        self.assertEqual(results["u_a"] + results["u_b"], 30 * 30)
        self.assertTrue(-1.0 <= results["r_rb"] <= 1.0)
        self.assertTrue(0.0 <= results["cles"] <= 1.0)
        self.assertIn("Mann-Whitney U Test", results["overall_recommendation"])

    def test_mann_whitney_demo_unequal_shapes_heteroscedastic(self):
        df, results = generate_mann_whitney_demo(
            n_a=30, n_b=30, dist_shape="unequal_shapes_variances", seed=42
        )
        self.assertIn(results["overall_status"], ["warning", "fail"])
        self.assertTrue(0.0 <= results["p_mann"] <= 1.0)
        # Recommendation suggests Brunner-Munzel or Welch's t-test
        rec_text = results["overall_recommendation"]
        self.assertTrue("Brunner-Munzel" in rec_text or "Welch" in rec_text or "stochastic superiority" in rec_text)

    def test_welch_demo_normal_passes_assumptions(self):
        df, results = generate_welch_demo(
            n_a=35, n_b=35, mean_a=100.0, mean_b=108.0, std_a=10.0, std_b=18.0, dist_shape="normal", seed=42
        )
        self.assertEqual(len(df), 70)
        self.assertEqual(results["overall_status"], "pass")
        self.assertTrue(0.0 <= results["p_val"] <= 1.0)
        self.assertGreater(results["df"], 0.0)
        self.assertIn("ci_lower", results)
        self.assertIn("ci_upper", results)
        self.assertTrue(results["ci_lower"] < results["ci_upper"])
        self.assertIn("cohens_d", results)
        self.assertIn("hedges_g", results)
        self.assertIn("Welch", results["overall_recommendation"])

    def test_welch_demo_extreme_outliers_violation(self):
        df, results = generate_welch_demo(
            n_a=20, n_b=20, dist_shape="normal", outlier_count_b=2, seed=42
        )
        self.assertEqual(results["overall_status"], "fail")
        self.assertGreaterEqual(results["total_ext"], 1)
        self.assertIn("Mann-Whitney", results["overall_recommendation"])

    def test_welch_demo_clt_robustness(self):
        df, results = generate_welch_demo(
            n_a=40, n_b=40, dist_shape="skewed_moderate", seed=42
        )
        # Moderate skew with N >= 30 yields either pass or caution/warning CLT protection
        self.assertIn(results["overall_status"], ["pass", "warning"])
        self.assertTrue(0.0 <= results["p_val"] <= 1.0)

    def test_students_t_demo_homoscedastic(self):
        df, results = generate_students_t_demo(
            n_a=35, n_b=35, mean_a=100.0, mean_b=108.0, std_a=12.0, std_b=12.0, seed=42
        )
        self.assertEqual(len(df), 70)
        self.assertEqual(results["df"], 68)  # Exact integer df = 35 + 35 - 2
        self.assertEqual(results["overall_status"], "pass")
        self.assertIn("Student's", results["overall_recommendation"])
        self.assertTrue(0.0 <= results["p_val"] <= 1.0)
        self.assertGreater(results["sp"], 0)
        self.assertGreater(results["se_diff"], 0)

    def test_students_t_demo_heteroscedastic(self):
        df, results = generate_students_t_demo(
            n_a=35, n_b=35, mean_a=100.0, mean_b=108.0, std_a=5.0, std_b=25.0, seed=42
        )
        # When variances are severely unequal, status should warn and recommend Welch
        self.assertIn(results["overall_status"], ["warning", "fail"])
        self.assertIn("Welch", results["overall_recommendation"])

    def test_students_t_demo_outliers(self):
        df, results = generate_students_t_demo(
            n_a=35, n_b=35, outlier_count_b=3, seed=42
        )
        self.assertEqual(results["overall_status"], "fail")
        self.assertGreaterEqual(results["total_ext"], 1)
        self.assertIn("Mann-Whitney", results["overall_recommendation"])

    def test_two_independent_demo(self):

        df, results = generate_two_independent_demo(n_a=25, n_b=25, seed=123)
        self.assertEqual(len(df), 50)
        self.assertIn("p_welch", results)
        self.assertTrue(0.0 <= results["p_welch"] <= 1.0)
        self.assertIn("mean_a", results)
        self.assertIn("mean_b", results)



    def test_paired_demo(self):
        df, results = generate_paired_demo(n_pairs=35, seed=42)
        self.assertEqual(len(df), 35)
        self.assertEqual(results["df"], 34)  # df = n - 1
        self.assertEqual(results["overall_status"], "pass")
        self.assertIn("Paired", results["overall_recommendation"])
        self.assertIn("p_paired_t", results)
        self.assertTrue(0.0 <= results["p_paired_t"] <= 1.0)
        self.assertGreater(results["se_diff"], 0)

    def test_paired_demo_outliers(self):
        df, results = generate_paired_demo(n_pairs=35, outlier_count=3, seed=42)
        self.assertEqual(results["overall_status"], "fail")
        self.assertGreaterEqual(results["total_ext"], 1)
        self.assertIn("Wilcoxon", results["overall_recommendation"])

    def test_contingency_demo(self):
        df_table, results = generate_contingency_demo(40, 60, 20, 80)
        self.assertEqual(df_table.shape, (2, 2))
        self.assertIn("p_chi2", results)
        self.assertIn("odds_ratio", results)
        self.assertGreater(results["odds_ratio"], 0)
        self.assertEqual(results["n_total"], 200)
        self.assertGreaterEqual(results["min_expected_count"], 5.0)
        self.assertEqual(results["overall_status"], "pass")
        self.assertIn("Chi-Square", results["overall_verdict_badge"])
        self.assertEqual(len(results["cells_diagnostics"]), 4)
        self.assertEqual(len(results["df_cases"]), 200)
        self.assertIn("risk_ratio", results)
        self.assertIn("risk_diff", results)
        self.assertIn("phi", results)
        self.assertIn("cramers_v", results)

    def test_contingency_demo_small_sample_triggers_fisher(self):
        df_table, results = generate_contingency_demo(4, 16, 1, 19)
        self.assertEqual(df_table.shape, (2, 2))
        self.assertLess(results["min_expected_count"], 5.0)
        self.assertEqual(results["overall_status"], "warning")
        self.assertIn("Fisher's Exact", results["overall_recommendation"])
        self.assertTrue(0.0 <= results["p_fisher"] <= 1.0)

    def test_contingency_demo_null_association(self):
        df_table, results = generate_contingency_demo(50, 50, 50, 50)
        self.assertEqual(df_table.shape, (2, 2))
        self.assertAlmostEqual(results["odds_ratio"], 1.0, places=2)
        self.assertAlmostEqual(results["risk_diff"], 0.0, places=2)
        self.assertGreaterEqual(results["p_chi2"], 0.90)
        self.assertEqual(results["overall_status"], "pass")
        self.assertIn("No Statistically Significant Association", results["overall_recommendation"])

    def test_fisher_exact_demo_standard(self):
        df_table, results = generate_fisher_exact_demo(a=4, b=16, c=1, d=19, alpha=0.05)
        self.assertEqual(df_table.shape, (2, 2))
        self.assertEqual(results["n_total"], 40)
        self.assertEqual(results["r1"], 20)
        self.assertEqual(results["r2"], 20)
        self.assertEqual(results["c1"], 5)
        self.assertEqual(results["c2"], 35)

        # Scipy validation
        scipy_or, scipy_p_two = stats.fisher_exact([[4, 16], [1, 19]], alternative="two-sided")
        _, scipy_p_greater = stats.fisher_exact([[4, 16], [1, 19]], alternative="greater")
        _, scipy_p_less = stats.fisher_exact([[4, 16], [1, 19]], alternative="less")

        self.assertAlmostEqual(results["p_fisher_two_sided"], scipy_p_two, places=6)
        self.assertAlmostEqual(results["p_fisher_greater"], scipy_p_greater, places=6)
        self.assertAlmostEqual(results["p_fisher_less"], scipy_p_less, places=6)

        # Hypergeometric distribution properties
        self.assertIn("df_dist", results)
        df_dist = results["df_dist"]
        self.assertEqual(len(df_dist), 6)  # k from 0 to 5
        self.assertAlmostEqual(df_dist["Probability P(k)"].sum(), 1.0, places=5)

        # Small sample Cochran violation
        self.assertLess(results["min_expected_count"], 5.0)
        self.assertGreater(results["expected_lt_5_pct"], 0.0)

        # Subject-level dataset
        self.assertEqual(len(results["df_cases"]), 40)
        self.assertIn("odds_ratio", results)
        self.assertIn("risk_ratio", results)

    def test_fisher_exact_demo_zero_cell(self):
        df_table, results = generate_fisher_exact_demo(a=5, b=10, c=0, d=15, alpha=0.05)
        self.assertEqual(df_table.shape, (2, 2))
        self.assertGreater(results["odds_ratio"], 0.0)
        self.assertGreater(results["or_ci_lower"], 0.0)
        self.assertGreater(results["or_ci_upper"], 0.0)
        self.assertTrue(0.0 <= results["p_fisher_two_sided"] <= 1.0)
        self.assertEqual(len(results["df_cases"]), 30)

    def test_fisher_exact_demo_null_case(self):
        df_table, results = generate_fisher_exact_demo(a=5, b=10, c=5, d=10, alpha=0.05)
        self.assertEqual(df_table.shape, (2, 2))
        self.assertAlmostEqual(results["odds_ratio"], 1.0, places=2)
        self.assertAlmostEqual(results["risk_diff"], 0.0, places=2)
        self.assertAlmostEqual(results["p_fisher_two_sided"], 1.0, places=2)
        self.assertIn("Fail to reject", results["overall_recommendation"])

    def test_mcnemar_demo_significant_improvement(self):
        df_table, results = generate_mcnemar_demo(a=15, b=5, c=25, d=55, alpha=0.05)
        self.assertEqual(df_table.shape, (2, 2))
        self.assertEqual(results["n_total"], 100)
        self.assertEqual(results["n_conc"], 70)
        self.assertEqual(results["n_disc"], 30)
        self.assertEqual(results["net_shift"], 20)
        self.assertAlmostEqual(results["risk_diff"], 0.20, places=2)
        self.assertAlmostEqual(results["odds_ratio"], 5.0, places=2)
        self.assertLess(results["p_exact_two_sided"], 0.05)
        self.assertLess(results["p_edwards"], 0.05)
        self.assertEqual(results["overall_status"], "pass")
        self.assertEqual(len(results["df_cases"]), 100)
        self.assertEqual(len(results["binom_distribution"]), 31)  # 0 to 30

    def test_mcnemar_demo_symmetrical_null(self):
        df_table, results = generate_mcnemar_demo(a=20, b=12, c=12, d=36, alpha=0.05)
        self.assertEqual(df_table.shape, (2, 2))
        self.assertEqual(results["net_shift"], 0)
        self.assertAlmostEqual(results["risk_diff"], 0.0, places=2)
        self.assertAlmostEqual(results["odds_ratio"], 1.0, places=2)
        self.assertAlmostEqual(results["p_exact_two_sided"], 1.0, places=2)
        self.assertEqual(results["overall_status"], "info")
        self.assertIn("Fail to reject", results["overall_recommendation"])

    def test_mcnemar_demo_small_discordant_recommends_exact(self):
        df_table, results = generate_mcnemar_demo(a=8, b=2, c=9, d=11, alpha=0.05)
        self.assertEqual(results["n_disc"], 11)
        self.assertEqual(results["primary_method"], "Exact Binomial Test")
        self.assertTrue(any(asm["status"] == "warning" and "b + c" in asm["name"] for asm in results["assumptions_eval"]))

    def test_mcnemar_demo_zero_discordant_cell(self):
        df_table, results = generate_mcnemar_demo(a=10, b=0, c=8, d=12, alpha=0.05)
        self.assertEqual(results["b"], 0)
        self.assertGreater(results["odds_ratio"], 0.0)
        self.assertLess(results["p_exact_two_sided"], 0.05)
        self.assertTrue(any(asm["status"] == "warning" and "Zero Discordant" in asm["verdict"] for asm in results["assumptions_eval"]))

    def test_one_way_anova_demo_normal_passes(self):
        df, results = generate_one_way_anova_demo(n_per_group=30, dist_shape="normal", seed=42)
        self.assertEqual(len(df), 90)
        self.assertEqual(results["overall_status"], "pass")
        self.assertIn("One-Way ANOVA", results["overall_recommendation"])
        self.assertEqual(results["df_between"], 2)
        self.assertEqual(results["df_within"], 87)
        self.assertTrue(0.0 <= results["p_anova"] <= 1.0)
        self.assertTrue(0.0 <= results["eta_squared"] <= 1.0)

    def test_one_way_anova_demo_heteroscedastic_recommends_welch(self):
        df, results = generate_one_way_anova_demo(n_per_group=30, stds=(4.0, 10.0, 24.0), seed=42)
        self.assertEqual(results["overall_status"], "fail")
        self.assertIn("Welch", results["overall_recommendation"])

    def test_one_way_anova_demo_non_normal_recommends_kruskal(self):
        df, results = generate_one_way_anova_demo(n_per_group=15, dist_shape="heavy_tailed", seed=42)
        self.assertEqual(results["overall_status"], "fail")
        self.assertIn("Kruskal-Wallis", results["overall_recommendation"])

    def test_one_way_anova_demo_outliers(self):
        df, results = generate_one_way_anova_demo(n_per_group=30, outlier_count_g3=2, seed=42)
        self.assertEqual(results["overall_status"], "fail")
        self.assertIn("Kruskal-Wallis", results["overall_recommendation"])

    def test_kruskal_wallis_demo_continuous_passes(self):
        df, results = generate_kruskal_wallis_demo(n_per_group=25, shifts=(0.0, 5.0, 10.0), data_type="continuous", seed=42)
        self.assertEqual(len(df), 75)
        self.assertEqual(results["k_groups"], 3)
        self.assertEqual(results["df_kruskal"], 2)
        self.assertLess(results["p_kruskal"], 0.05)
        self.assertGreater(results["h_stat"], 0.0)
        self.assertTrue(0.0 <= results["epsilon_squared"] <= 1.0)
        self.assertEqual(len(results["pairwise_comparisons"]), 3)
        self.assertIn("Reject the null hypothesis", results["overall_recommendation"])

    def test_kruskal_wallis_demo_ordinal_likert(self):
        df, results = generate_kruskal_wallis_demo(n_per_group=20, data_type="ordinal_likert", seed=42)
        self.assertEqual(len(df), 60)
        self.assertGreaterEqual(results["tie_count_total"], 1)
        self.assertLessEqual(results["tie_factor"], 1.0)
        self.assertTrue(all(1.0 <= val <= 7.0 for val in df["Score"]))

    def test_kruskal_wallis_demo_no_difference(self):
        df, results = generate_kruskal_wallis_demo(n_per_group=30, shifts=(0.0, 0.0, 0.0), dist_shape="normal", seed=42)
        self.assertGreater(results["p_kruskal"], 0.05)
        self.assertIn("Fail to reject", results["overall_recommendation"])

    def test_repeated_measures_anova_demo_normal_passes(self):
        df, results = generate_repeated_measures_anova_demo(n_subjects=25, seed=42)
        self.assertEqual(len(df), 75)
        self.assertEqual(results["n_subjects"], 25)
        self.assertEqual(results["df_time"], 2)
        self.assertEqual(results["df_error"], 48)
        self.assertEqual(results["overall_status"], "pass")
        self.assertIn("Repeated Measures ANOVA", results["overall_recommendation"])
        self.assertLess(results["p_anova"], 0.05)
        self.assertEqual(len(results["pairwise_comparisons"]), 3)
        self.assertGreater(results["w_mauchly"], 0.5)
        self.assertGreater(results["eps_gg"], 0.75)

    def test_repeated_measures_anova_demo_sphericity_violation(self):
        df, results = generate_repeated_measures_anova_demo(n_subjects=25, sphericity_violation=True, seed=42)
        self.assertEqual(results["overall_status"], "fail")
        self.assertIn("Greenhouse-Geisser", results["overall_recommendation"])
        self.assertLess(results["eps_gg"], 0.75)

    def test_repeated_measures_anova_demo_outliers(self):
        df, results = generate_repeated_measures_anova_demo(n_subjects=25, outlier_count=1, seed=42)
        self.assertEqual(results["overall_status"], "fail")
        self.assertIn("Friedman", results["overall_recommendation"])
        self.assertGreaterEqual(results["total_ext_outliers"], 1)

    def test_friedman_demo_ordinal_likert_passes(self):
        df, results = generate_friedman_demo(n_subjects=25, data_type="ordinal_likert", seed=42)
        self.assertEqual(len(df), 75)
        self.assertEqual(results["n_subjects"], 25)
        self.assertEqual(results["df_friedman"], 2)
        self.assertLess(results["p_friedman"], 0.05)
        self.assertGreater(results["q_stat"], 0.0)
        self.assertTrue(0.0 <= results["kendalls_w"] <= 1.0)
        self.assertEqual(len(results["pairwise_comparisons"]), 3)
        self.assertIn("Reject the null hypothesis", results["overall_recommendation"])
        self.assertTrue(all(1.0 <= val <= 7.0 for val in df["Score"]))

    def test_friedman_demo_continuous(self):
        df, results = generate_friedman_demo(n_subjects=20, data_type="continuous", shifts=(0.0, 4.0, 8.0), seed=42)
        self.assertEqual(len(df), 60)
        self.assertEqual(results["n_subjects"], 20)
        self.assertLess(results["p_friedman"], 0.05)

    def test_friedman_demo_no_difference(self):
        df, results = generate_friedman_demo(n_subjects=25, shifts=(0.0, 0.0, 0.0), dist_shape="normal", seed=42)
        self.assertGreater(results["p_friedman"], 0.05)
        self.assertIn("Fail to reject", results["overall_recommendation"])

    def test_pearson_correlation_strong_positive(self):
        df, results = generate_pearson_correlation_demo(
            n_pts=60, true_r=0.82, scenario="normal_linear", seed=42
        )
        self.assertEqual(len(df), 60)
        self.assertEqual(results["n_pts"], 60)
        self.assertEqual(results["df"], 58)
        self.assertGreater(results["pearson_r"], 0.70)
        self.assertLess(results["pearson_p"], 0.001)
        self.assertGreater(results["r_squared"], 0.50)
        self.assertEqual(results["overall_status"], "pass")
        self.assertIn("Reject the null hypothesis", results["overall_recommendation"])
        self.assertEqual(len(results["df_cases"]), 60)
        self.assertEqual(len(results["x_grid"]), 100)
        self.assertEqual(len(results["ci_lower_band"]), 100)
        self.assertEqual(len(results["pi_lower_band"]), 100)
        # CI within bounds
        self.assertLess(results["r_ci"][0], results["pearson_r"])
        self.assertGreater(results["r_ci"][1], results["pearson_r"])

    def test_pearson_correlation_moderate_negative(self):
        df, results = generate_pearson_correlation_demo(
            n_pts=50, true_r=-0.58, scenario="moderate_negative", seed=42
        )
        self.assertEqual(results["n_pts"], 50)
        self.assertLess(results["pearson_r"], -0.40)
        self.assertLess(results["slope"], 0.0)
        self.assertLess(results["pearson_p"], 0.05)
        self.assertEqual(results["direction_label"], "Negative")

    def test_pearson_correlation_null(self):
        df, results = generate_pearson_correlation_demo(
            n_pts=60, true_r=0.0, scenario="null_correlation", seed=42
        )
        self.assertLess(abs(results["pearson_r"]), 0.25)
        self.assertGreater(results["pearson_p"], 0.05)
        self.assertEqual(results["overall_status"], "info")
        self.assertIn("Fail to reject", results["overall_recommendation"])

    def test_pearson_correlation_non_linear_curvature(self):
        df, results = generate_pearson_correlation_demo(
            n_pts=60, true_r=0.65, scenario="non_linear_curvature", seed=42
        )
        # Non-linear scenario triggers curvature warning or lower linearity
        self.assertFalse(results["is_linear"])
        self.assertLess(results["p_curvature"], 0.05)
        self.assertTrue(any(asm["name"] == "Linear Relationship" and asm["status"] == "warning" for asm in results["assumptions_eval"]))

    def test_pearson_correlation_outlier_leverage(self):
        df, results = generate_pearson_correlation_demo(
            n_pts=45, scenario="outlier_leverage", seed=42
        )
        self.assertTrue(results["has_extreme_outliers"])
        self.assertGreaterEqual(results["high_cooks_count"], 1)
        self.assertTrue(any("Outlier" in asm["name"] and asm["status"] in ["warning", "fail"] for asm in results["assumptions_eval"]))

    def test_generate_correlation_demo_alias(self):
        df, results = generate_correlation_demo(n_pts=30, true_r=0.5, seed=123)
        self.assertEqual(len(df), 30)
        self.assertIn("pearson_r", results)
        self.assertIn("r_squared", results)

    def test_spearman_correlation_non_linear_monotonic(self):
        df, results = generate_spearman_correlation_demo(
            n_pts=50, scenario="non_linear_monotonic", seed=42
        )
        self.assertEqual(len(df), 50)
        self.assertEqual(results["n_pts"], 50)
        self.assertEqual(results["df"], 48)
        self.assertGreater(results["spearman_rho"], 0.80)
        self.assertLess(results["spearman_p"], 0.001)
        # Spearman outperforms Pearson on non-linear curve
        self.assertGreater(abs(results["spearman_rho"]), abs(results["pearson_r"]))
        self.assertEqual(results["overall_status"], "pass")
        self.assertIn("Reject the null hypothesis", results["overall_recommendation"])
        self.assertEqual(len(results["df_cases"]), 50)
        self.assertGreater(results["sum_d_squared"], 0)
        self.assertLess(results["rho_ci"][0], results["spearman_rho"])
        self.assertGreater(results["rho_ci"][1], results["spearman_rho"])

    def test_spearman_correlation_ordinal_likert_ties(self):
        df, results = generate_spearman_correlation_demo(
            n_pts=60, scenario="ordinal_likert", seed=42
        )
        self.assertEqual(results["n_pts"], 60)
        self.assertTrue(results["has_ties"])
        self.assertGreater(results["ties_x_count"], 0)
        self.assertGreater(results["ties_y_count"], 0)
        self.assertGreater(results["tie_correction_x"], 0)
        self.assertGreater(results["tie_correction_y"], 0)
        self.assertTrue(all(1.0 <= val <= 7.0 for val in results["x"]))
        self.assertTrue(all(1.0 <= val <= 7.0 for val in results["y"]))
        self.assertIn("kendall_tau", results)

    def test_spearman_correlation_inverse_monotonic(self):
        df, results = generate_spearman_correlation_demo(
            n_pts=45, scenario="inverse_monotonic", seed=42
        )
        self.assertEqual(results["n_pts"], 45)
        self.assertLess(results["spearman_rho"], -0.75)
        self.assertLess(results["spearman_p"], 0.001)
        self.assertEqual(results["direction_label"], "Negative")

    def test_spearman_correlation_extreme_outliers(self):
        df, results = generate_spearman_correlation_demo(
            n_pts=40, scenario="extreme_outliers", seed=42
        )
        # Spearman remains robust while Pearson is suppressed
        self.assertGreater(results["spearman_rho"], 0.65)
        self.assertLess(results["pearson_r"], 0.45)
        self.assertGreater(results["spearman_rho"] - results["pearson_r"], 0.20)

    def test_spearman_correlation_null(self):
        df, results = generate_spearman_correlation_demo(
            n_pts=50, scenario="null_rank", seed=42
        )
        self.assertLess(abs(results["spearman_rho"]), 0.30)
        self.assertGreater(results["spearman_p"], 0.05)
        self.assertEqual(results["overall_status"], "info")
        self.assertIn("Fail to reject", results["overall_recommendation"])


class TestLinearRegressionDemo(unittest.TestCase):
    """Tests the Linear Regression (OLS) calculation engine, ANOVA decomposition,
    multicollinearity diagnostics, and assumption evaluations.
    """

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack (numpy, scipy, pandas) not available.")

    def test_simple_bivariate_regression(self):
        df, results = generate_linear_regression_demo(
            n_pts=60, scenario="simple_bivariate", seed=42
        )
        self.assertEqual(results["n_pts"], 60)
        self.assertEqual(results["k_preds"], 1)
        self.assertGreater(results["r_squared"], 0.40)
        self.assertLess(results["p_f"], 0.001)
        self.assertEqual(results["vifs"][1], 1.0)
        self.assertEqual(results["tolerances"][1], 1.0)
        self.assertGreater(results["beta"][1], 3.0)  # Positive slope
        self.assertAlmostEqual(
            results["ss_tot"], results["ss_reg"] + results["ss_res"], delta=0.01
        )
        self.assertEqual(len(results["df_coefs"]), 2)  # Intercept + 1 predictor

    def test_multiple_regression_academic(self):
        df, results = generate_linear_regression_demo(
            n_pts=80, scenario="academic_performance", seed=42
        )
        self.assertEqual(results["n_pts"], 80)
        self.assertEqual(results["k_preds"], 3)
        self.assertGreater(results["r_squared"], 0.60)
        self.assertLess(results["p_f"], 0.001)
        self.assertEqual(len(results["df_coefs"]), 4)  # Intercept + 3 predictors
        self.assertEqual(len(results["pred_names"]), 3)
        self.assertEqual(len(results["beta_standardized"]), 4)
        self.assertTrue(np.isnan(results["beta_standardized"][0]))
        # Standardized betas for predictors should be real numbers
        for b_std in results["beta_standardized"][1:]:
            self.assertFalse(np.isnan(b_std))
        # Check Cook's D and leverage
        self.assertEqual(len(results["cooks_d"]), 80)
        self.assertEqual(len(results["hat_diags"]), 80)
        self.assertAlmostEqual(np.sum(results["hat_diags"]), 4.0, delta=0.01)  # sum(h_ii) = k + 1 = 4

    def test_multicollinear_stress(self):
        df, results = generate_linear_regression_demo(
            n_pts=75, scenario="multicollinear", seed=42
        )
        self.assertEqual(results["n_pts"], 75)
        # In multicollinear scenario, max VIF should be noticeably elevated
        self.assertGreater(results["max_vif"], 5.0)
        # Check that multicollinearity assumption is flagged
        multi_card = [a for a in results["assumptions_eval"] if "Multicollinearity" in a["name"]][0]
        self.assertIn(multi_card["status"], ["warning", "fail"])

    def test_heteroscedastic_residuals(self):
        df, results = generate_linear_regression_demo(
            n_pts=80, scenario="heteroscedastic", seed=42
        )
        self.assertEqual(results["n_pts"], 80)
        self.assertIn("p_bp", results)
        self.assertFalse(results["is_homoscedastic"])

    def test_custom_regression_dimensions(self):
        for k in [1, 2, 3]:
            df, results = generate_linear_regression_demo(
                n_pts=50, n_predictors=k, scenario="custom", seed=42
            )
            self.assertEqual(results["k_preds"], k)
            self.assertEqual(len(results["pred_names"]), k)
            self.assertEqual(len(results["beta"]), k + 1)
            self.assertEqual(results["df_reg"], k)
            self.assertEqual(results["df_res"], 50 - k - 1)


class TestBinaryLogisticRegressionDemo(unittest.TestCase):
    """Tests the Binary Logistic Regression synthetic data generator and statistical engine."""

    def setUp(self):
        if not HAS_SCIENTIFIC_STACK:
            self.skipTest("Scientific stack not installed in environment")

    def test_cardiology_scenario(self):
        df, results = generate_binary_logistic_regression_demo(
            n_pts=120, n_predictors=3, scenario="cardiology", seed=42
        )
        self.assertEqual(len(df), 120)
        self.assertEqual(results["n_pts"], 120)
        self.assertEqual(results["k_preds"], 3)
        self.assertEqual(results["outcome_name"], "CVD_Event")
        self.assertEqual(len(results["pred_names"]), 3)
        self.assertEqual(len(results["beta"]), 4)  # intercept + 3 predictors
        self.assertEqual(len(results["se"]), 4)
        self.assertEqual(len(results["odds_ratios"]), 4)
        self.assertEqual(len(results["or_ci_lower"]), 4)
        self.assertEqual(len(results["or_ci_upper"]), 4)

        # Check diagnostics
        self.assertIn("ll_model", results)
        self.assertIn("ll_null", results)
        self.assertIn("lr_stat", results)
        self.assertIn("p_omnibus", results)
        self.assertIn("pseudo_r2_mcfadden", results)
        self.assertGreaterEqual(results["pseudo_r2_mcfadden"], 0.0)
        self.assertLessEqual(results["pseudo_r2_mcfadden"], 1.0)

        # Classification metrics
        self.assertGreaterEqual(results["auc"], 0.5)
        self.assertLessEqual(results["auc"], 1.0)
        self.assertGreaterEqual(results["accuracy"], 0.0)
        self.assertLessEqual(results["accuracy"], 1.0)
        self.assertEqual(results["tp"] + results["tn"] + results["fp"] + results["fn"], 120)

        # Assumptions evaluation
        self.assertEqual(len(results["assumptions_eval"]), 5)
        asm_names = [a["name"] for a in results["assumptions_eval"]]
        self.assertTrue(any("Binary Dependent Variable" in n for n in asm_names))
        self.assertTrue(any("Events Per Variable" in n for n in asm_names))
        self.assertTrue(any("Multicollinearity" in n for n in asm_names))
        self.assertTrue(any("Separation" in n for n in asm_names))
        self.assertTrue(any("Residual" in n for n in asm_names))

    def test_readmission_scenario(self):
        df, results = generate_binary_logistic_regression_demo(
            n_pts=100, n_predictors=2, scenario="readmission", seed=42
        )
        self.assertEqual(results["k_preds"], 2)
        self.assertEqual(results["outcome_name"], "Readmission_30d")
        self.assertGreaterEqual(results["auc"], 0.5)
        self.assertIn("preset_name", results)

    def test_exam_pass_scenario(self):
        df, results = generate_binary_logistic_regression_demo(
            n_pts=80, n_predictors=1, scenario="exam_pass", seed=42
        )
        self.assertEqual(results["k_preds"], 1)
        self.assertEqual(results["outcome_name"], "Exam_Passed")
        self.assertEqual(len(results["pred_names"]), 1)
        self.assertEqual(len(results["beta"]), 2)

    def test_multicollinear_scenario(self):
        df, results = generate_binary_logistic_regression_demo(
            n_pts=90, scenario="multicollinear", seed=42
        )
        self.assertGreaterEqual(results["max_vif"], 3.0)
        vif_card = next(a for a in results["assumptions_eval"] if "Multicollinearity" in a["name"])
        self.assertIn(vif_card["status"], ["warning", "fail", "pass"])

    def test_sparse_events_scenario(self):
        df, results = generate_binary_logistic_regression_demo(
            n_pts=50, scenario="sparse_events", seed=42
        )
        self.assertIn(results["overall_status"], ["warning", "fail", "pass"])
        epv_card = next(a for a in results["assumptions_eval"] if "Events Per Variable" in a["name"])
        self.assertIn("EPV", epv_card["verdict"])

    def test_custom_logistic_dimensions(self):
        for k in [1, 2, 3]:
            df, results = generate_binary_logistic_regression_demo(
                n_pts=60, n_predictors=k, scenario="custom", seed=42
            )
            self.assertEqual(results["k_preds"], k)
            self.assertEqual(len(results["pred_names"]), k)
            self.assertEqual(len(results["beta"]), k + 1)
            self.assertEqual(results["lr_df"], k)


class TestUtils(unittest.TestCase):
    """Tests utility functions."""

    def test_format_p_value(self):
        self.assertEqual(format_p_value(0.0001), "< 0.001")
        self.assertEqual(format_p_value(0.042), "0.042")

    def test_format_ci(self):
        formatted = format_ci(1.5, 1.2, 1.8)
        self.assertEqual("1.50 (95% CI: 1.20 - 1.80)", formatted)


class TestSampleDataDownloads(unittest.TestCase):
    """Tests sample dataset generation and download guidance for all 18 methods."""

    def test_all_18_tests_have_sample_data_guidance(self):
        try:
            import pandas as pd
            import numpy as np
            from src.ui_components.stats_widgets import (
                _generate_fallback_dataset,
                _get_sample_data_guide_info
            )
            has_stack = True
        except ImportError:
            has_stack = False

        if not has_stack:
            self.skipTest("Scientific stack not installed in environment")

        test_keys = [
            "one_sample_t_test", "wilcoxon_signed_rank", "welch_t_test",
            "students_t_test", "mann_whitney_u", "paired_t_test",
            "wilcoxon_paired", "one_way_anova", "kruskal_wallis",
            "repeated_measures_anova", "friedman_test", "chi_square_independence",
            "fisher_exact", "mcnemar_test", "pearson_correlation",
            "spearman_correlation", "linear_regression", "binary_logistic_regression"
        ]

        for tk in test_keys:
            rec = STATISTICAL_TESTS[tk]
            df, res = _generate_fallback_dataset(tk)
            self.assertIsNotNone(df, f"Fallback df should not be None for {tk}")
            self.assertGreater(len(df), 0, f"Fallback df should not be empty for {tk}")

            info = _get_sample_data_guide_info(tk, rec, df, res)
            self.assertIn("default_filename", info, f"Missing default_filename for {tk}")
            self.assertTrue(info["default_filename"].endswith(".csv"), f"Filename should end in .csv for {tk}")
            self.assertIn("excel_instructions", info, f"Missing excel_instructions for {tk}")
            self.assertIn("python_code", info, f"Missing python_code for {tk}")
            self.assertIn("pd.read_csv", info["python_code"], f"Python snippet should include pd.read_csv for {tk}")


if __name__ == "__main__":
    unittest.main()


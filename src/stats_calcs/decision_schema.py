"""Statistical Tests Decision Schema & Logic Engine.

This module provides deterministic routing rules to identify the appropriate
statistical test based on research objectives, data types, sample properties,
and distribution assumptions.
"""

from typing import Dict, Any, List, Optional

STATISTICAL_TESTS: Dict[str, Dict[str, Any]] = {
    "welch_t_test": {
        "name": "Welch's Independent Two-Sample t-test",
        "category": "Comparing 2 Independent Groups",
        "parametric": True,
        "description": "Compares the means of two independent groups without assuming equal population variances (heteroscedasticity). Widely recommended as the optimal default over Student's t-test.",
        "when_to_use": [
            "Comparing 2 independent groups (e.g. Treatment vs Control, Cohort A vs Cohort B).",
            "The outcome variable is continuous (ratio or interval scale).",
            "Data in each group is approximately normally distributed (or N >= 30 per group via the Central Limit Theorem).",
            "Variances between the two groups may be unequal or unknown."
        ],
        "key_assumptions": [
            "Continuous measurement scale (interval or ratio).",
            "Independence of observations (different subjects in each group).",
            "Normality of the dependent variable within each group (robust for N >= 30).",
            "No extreme outliers that distort group means or standard deviations."
        ],
        "alternative_test": "Mann-Whitney U Test (if sample size is small N < 30 and distributions are skewed or have extreme outliers).",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": "=T.TEST(array1, array2, 2, 3)",
            "toolpak_available": True,
            "toolpak_steps": "Data Tab -> Data Analysis -> 't-Test: Two-Sample Assuming Unequal Variances' -> Select Variable 1 Range & Variable 2 Range -> OK.",
            "notes": "In =T.TEST(), tails=2 specifies two-tailed, and type=3 specifies Welch's unequal variance t-test."
        },
        "python_snippet": (
            "import scipy.stats as stats\n\n"
            "# Welch's independent two-sample t-test (equal_var=False)\n"
            "t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)\n"
            "print(f'Welch t-statistic: {t_stat:.4f}, p-value: {p_val:.4e}')"
        ),
        "demo_generator": "welch_t_test"
    },
    "students_t_test": {
        "name": "Student's Independent Two-Sample t-test (Equal Variance)",
        "category": "Comparing 2 Independent Groups",
        "parametric": True,
        "description": "Compares the means of two independent groups assuming equal population variances (homoscedasticity).",
        "when_to_use": [
            "You have 2 independent groups.",
            "The outcome variable is continuous and normally distributed.",
            "Variances of the two groups are proven equal (e.g., via Levene's test p > 0.05)."
        ],
        "key_assumptions": [
            "Independence of observations.",
            "Normality within each group.",
            "Homogeneity of variance (Levene's or Bartlett's test)."
        ],
        "alternative_test": "Welch's t-test (if variances are unequal) or Mann-Whitney U test (if non-normal).",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": "=T.TEST(array1, array2, 2, 2)",
            "toolpak_available": True,
            "toolpak_steps": "Data Tab -> Data Analysis -> 't-Test: Two-Sample Assuming Equal Variances' -> Select Ranges -> OK.",
            "notes": "In =T.TEST(), type=2 specifies pooled equal-variance Student's t-test."
        },
        "python_snippet": (
            "import scipy.stats as stats\n\n"
            "# Student's t-test (equal_var=True)\n"
            "t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=True)\n"
            "print(f't-statistic: {t_stat:.4f}, p-value: {p_val:.4e}')"
        ),
        "demo_generator": "students_t_test"
    },
    "mann_whitney_u": {
        "name": "Mann-Whitney U Test (Wilcoxon Rank-Sum Test)",
        "category": "Comparing 2 Independent Groups (Non-parametric)",
        "parametric": False,
        "description": "Non-parametric hypothesis test that compares ranks between two independent groups. Evaluates whether values in one group tend to be systematically higher or lower than the other without assuming normality.",
        "when_to_use": [
            "Comparing 2 independent groups (e.g. Treatment vs Control, Cohort A vs Cohort B).",
            "Outcome variable is ordinal (e.g. Likert scale, pain rating) OR continuous with skewed / non-normal distributions.",
            "Sample sizes are relatively small (N < 30 per group) where normality cannot be assumed.",
            "Extreme outliers or heavy tails exist that would distort parametric group means."
        ],
        "key_assumptions": [
            "Independent random samples (mutually exclusive subjects in each group).",
            "Measurement scale is at least ordinal (data values can be meaningfully ranked).",
            "Similar distribution shapes (variance and skewness) if interpreting as a test of median difference.",
            "Adequate sample size (n1, n2 >= 10 for normal Z-approximation, otherwise exact permutation)."
        ],
        "alternative_test": "Welch's t-test (if sample size is large enough for Central Limit Theorem) or Brunner-Munzel test (if group variances/shapes differ substantially).",
        "excel_guide": {
            "has_native_formula": False,
            "formula_syntax": (
                "Combined Ranks: =RANK.AVG(Score_cell, $Score_Range, 1)\n"
                "Rank Sum R1: =SUMIF($Group_Range, 'Group A', $Rank_Range)\n"
                "U1 Statistic: =n1*n2 + (n1*(n1+1))/2 - R1\n"
                "U Statistic: =MIN(U1, n1*n2 - U1)\n"
                "Large-Sample Z: =(ABS(U - (n1*n2)/2) - 0.5) / SQRT((n1*n2*(n1+n2+1))/12)\n"
                "Two-Tailed p-value: =2 * (1 - NORM.S.DIST(Z, TRUE))"
            ),
            "toolpak_available": False,
            "toolpak_steps": "Not built into standard Excel Toolpak without add-ins (XLSTAT/Real Statistics). Execute using column formulas for combined ranks and rank sums.",
            "notes": "Use =RANK.AVG() to handle ties properly with mid-ranks. For unequal shapes/variances, switch to Welch's t-test via =T.TEST(arr1, arr2, 2, 3)."
        },
        "python_snippet": (
            "import scipy.stats as stats\n\n"
            "# 1. Mann-Whitney U Test (two-sided)\n"
            "res = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')\n"
            "print(f'Mann-Whitney U: {res.statistic:.1f}, p-value: {res.pvalue:.4e}')\n\n"
            "# 2. Common Language Effect Size (CLES / Probability of Superiority)\n"
            "cles = res.statistic / (len(group_a) * len(group_b))\n"
            "print(f'CLES (P(A > B)): {cles:.3f}')"
        ),
        "demo_generator": "mann_whitney_u"
    },
    "paired_t_test": {
        "name": "Paired Samples t-test (Dependent t-test)",
        "category": "Comparing 2 Paired / Repeated Measures Groups",
        "parametric": True,
        "description": "Compares means between two related measurements taken from the exact same subjects (e.g. pre-treatment vs post-treatment) or matched pairs.",
        "when_to_use": [
            "You have paired observations (e.g., baseline vs 6-week follow-up in the same patients).",
            "Outcome is continuous.",
            "The pairwise differences (Post - Pre) are normally distributed."
        ],
        "key_assumptions": [
            "Dependent variable is continuous.",
            "The pairs are independent across subjects.",
            "The differences between pairs are approximately normally distributed."
        ],
        "alternative_test": "Wilcoxon Signed-Rank Test (if difference scores are skewed or ordinal).",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": "=T.TEST(pre_range, post_range, 2, 1)",
            "toolpak_available": True,
            "toolpak_steps": "Data Tab -> Data Analysis -> 't-Test: Paired Two Sample for Means' -> Select Variable 1 and 2 -> OK.",
            "notes": "In =T.TEST(), type=1 designates a paired t-test."
        },
        "python_snippet": (
            "import scipy.stats as stats\n\n"
            "# Paired t-test\n"
            "t_stat, p_val = stats.ttest_rel(pre_data, post_data)\n"
            "print(f't-statistic: {t_stat:.4f}, p-value: {p_val:.4e}')"
        ),
        "demo_generator": "paired_t_test"
    },
    "wilcoxon_paired": {
        "name": "Wilcoxon Signed-Rank Test (Paired / Dependent)",
        "category": "Comparing 2 Paired / Repeated Groups (Non-parametric)",
        "parametric": False,
        "description": "Non-parametric equivalent of the paired samples t-test. Evaluates whether the median difference between two dependent or matched measurements (e.g., Pre vs Post) significantly differs from zero by ranking the absolute pairwise difference scores and analyzing their signs.",
        "when_to_use": [
            "You have 2 paired, matched, or repeated measurements from the same subjects (e.g. Pre vs Post).",
            "The outcome variable is ordinal (e.g. Likert scale) OR continuous with non-normal / skewed difference scores.",
            "Sample size of pairs is small (N < 30) or extreme difference outliers preclude the paired t-test."
        ],
        "key_assumptions": [
            "Paired dependent design: Observations consist of matched pairs or repeated measurements from identical subjects.",
            "Measurement scale is at least ordinal: Data values and pairwise differences can be meaningfully ordered and ranked.",
            "Approximate symmetry of difference scores around the median difference.",
            "Adequate effective sample size (N_eff >= 10 non-zero pairs after dropping zero differences)."
        ],
        "alternative_test": "Exact Sign Test (if difference scores are severely asymmetric), or Paired Samples t-test (if difference scores are continuous and normally distributed).",
        "excel_guide": {
            "has_native_formula": False,
            "formula_syntax": (
                "Differences: =C2 - B2 (Post - Pre)\n"
                "Absolute Differences: =ABS(D2)\n"
                "Average Ranks: =IF(D2=0, \"\", RANK.AVG(E2, $E$2:$E$31, 1))\n"
                "Positive Ranks: =IF(D2>0, F2, 0)\n"
                "Negative Ranks: =IF(D2<0, F2, 0)\n"
                "Wilcoxon W: =MIN(SUM(G2:G31), SUM(H2:H31))\n"
                "Sign Test Alternative: =2 * BINOM.DIST(MIN(pos, n_eff - pos), n_eff, 0.5, TRUE)"
            ),
            "toolpak_available": False,
            "toolpak_steps": "Not built into standard Excel Toolpak; calculate directly via column formulas using =RANK.AVG() and =SUMIF(), or via the Exact Sign Test using =BINOM.DIST().",
            "notes": "Use =RANK.AVG() so tied absolute differences share average ranks properly."
        },
        "python_snippet": (
            "import scipy.stats as stats\n"
            "import numpy as np\n\n"
            "# 1. Compute paired differences\n"
            "diffs = post_data - pre_data\n"
            "nonzero_diffs = diffs[diffs != 0]\n\n"
            "# 2. Wilcoxon Signed-Rank Test for Paired Data\n"
            "res = stats.wilcoxon(post_data, pre_data, alternative='two-sided')\n"
            "print(f'Wilcoxon W = {res.statistic:.2f}, p-value = {res.pvalue:.4e}')\n\n"
            "# 3. Rank-Biserial Effect Size (r_rb)\n"
            "# r_rb = (W+ - W-) / (W+ + W-)  ranges from -1.0 to +1.0\n\n"
            "# 4. Alternative if Symmetry Fails: Exact Sign Test\n"
            "pos_count = int(np.sum(nonzero_diffs > 0))\n"
            "sign_res = stats.binomtest(pos_count, n=len(nonzero_diffs), p=0.5, alternative='two-sided')\n"
            "print(f'Exact Sign Test p-value = {sign_res.pvalue:.4e}')"
        ),
        "demo_generator": "wilcoxon_paired"
    },
    "wilcoxon_signed_rank": {
        "name": "Wilcoxon Signed-Rank Test (One-Sample)",
        "category": "Comparing 1 Group to a Median Benchmark (Non-parametric)",
        "parametric": False,
        "description": "Non-parametric hypothesis test that evaluates whether the median of a single sample differs from a theoretical benchmark by ranking the absolute deviations and analyzing their signs.",
        "when_to_use": [
            "Outcome variable is ordinal (e.g. Likert scale) OR continuous with a skewed, non-normal distribution.",
            "Comparing a single sample against a known median benchmark (or comparing paired/matched observations).",
            "Sample size is small (N < 30) where normality cannot be verified, or where outliers distort the mean."
        ],
        "key_assumptions": [
            "Measurement scale is at least ordinal (values can be ordered and ranked).",
            "Observations are mutually independent random draws.",
            "Distribution of differences around the median is approximately symmetric.",
            "Adequate effective sample size after dropping exact zero differences."
        ],
        "alternative_test": "Exact Sign Test (if distribution of differences is severely asymmetric) or One-Sample t-test (if data is normally distributed).",
        "excel_guide": {
            "has_native_formula": False,
            "formula_syntax": (
                "Differences: =A2 - benchmark\n"
                "Absolute Ranks: =IF(diff=0, '', RANK.AVG(ABS(diff), $range, 1))\n"
                "Signed Rank Sum (W): =SUMIF(diff_range, '>0', rank_range)\n"
                "Sign Test Alternative: =2 * BINOM.DIST(MIN(pos, n_eff - pos), n_eff, 0.5, TRUE)"
            ),
            "toolpak_available": False,
            "toolpak_steps": "Not built into Toolpak; execute via column formulas for ranks and signed sums, or via exact Sign Test using =BINOM.DIST().",
            "notes": "Use =RANK.AVG() to handle tied differences properly."
        },
        "python_snippet": (
            "import scipy.stats as stats\n"
            "import numpy as np\n\n"
            "# 1. Calculate differences from benchmark\n"
            "diffs = sample_data - benchmark\n"
            "nonzero_diffs = diffs[diffs != 0]\n\n"
            "# 2. Execute Wilcoxon Signed-Rank Test\n"
            "w_stat, p_val = stats.wilcoxon(nonzero_diffs, alternative='two-sided')\n"
            "print(f'Wilcoxon W = {w_stat:.2f}, p-value = {p_val:.4e}')\n\n"
            "# 3. If Symmetry Fails: Exact Sign Test\n"
            "pos_count = int(np.sum(nonzero_diffs > 0))\n"
            "sign_res = stats.binomtest(pos_count, n=len(nonzero_diffs), p=0.5, alternative='two-sided')\n"
            "print(f'Sign Test p-value = {sign_res.pvalue:.4e}')"
        ),
        "demo_generator": "wilcoxon_signed_rank"
    },

    "one_sample_t_test": {
        "name": "One-Sample t-test",
        "category": "Comparing 1 Group to a Standard Value",
        "parametric": True,
        "description": "Tests whether the mean of a single continuous sample differs significantly from a known benchmark or theoretical population mean.",
        "when_to_use": [
            "You have 1 sample group and want to compare its mean to a standard reference (e.g. national benchmark, historical target).",
            "Continuous outcome variable with normal distribution."
        ],
        "key_assumptions": [
            "Random sample, independent observations.",
            "Continuous scale.",
            "Normally distributed population."
        ],
        "alternative_test": "Wilcoxon Signed-Rank Test for 1 sample (against median benchmark).",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": (
                "t-stat: =(AVERAGE(A2:A36) - benchmark) / (STDEV.S(A2:A36) / SQRT(COUNT(A2:A36)))\n"
                "p-value: =T.DIST.2T(ABS(t_cell), COUNT(A2:A36) - 1)\n"
                "95% CI Margin: =CONFIDENCE.T(0.05, STDEV.S(A2:A36), COUNT(A2:A36))"
            ),
            "toolpak_available": True,
            "toolpak_steps": (
                "1. Data Tab -> Data Analysis -> 'Descriptive Statistics'.\n"
                "2. Select your data range, check 'Summary statistics' and 'Confidence Level for Mean (95%)'.\n"
                "3. Compute t-statistic by subtracting benchmark from reported Mean and dividing by reported Standard Error."
            ),
            "notes": "In Excel, =CONFIDENCE.T(0.05, s, n) returns the exact half-width margin of error for the 95% CI."
        },
        "python_snippet": (
            "import scipy.stats as stats\n"
            "import numpy as np\n\n"
            "# 1. One-sample t-test against benchmark\n"
            "t_stat, p_val = stats.ttest_1samp(sample_data, popmean=benchmark)\n"
            "print(f't({len(sample_data)-1}) = {t_stat:.4f}, p-value = {p_val:.4e}')\n\n"
            "# 2. Test Normality Assumption (Shapiro-Wilk)\n"
            "w_stat, w_p = stats.shapiro(sample_data)\n"
            "print(f'Shapiro-Wilk: W = {w_stat:.4f}, p = {w_p:.4e}')\n\n"
            "# 3. If Normality Fails: Wilcoxon Signed-Rank Test\n"
            "diffs = sample_data - benchmark\n"
            "w_alt, p_alt = stats.wilcoxon(diffs[diffs != 0])\n"
            "print(f'Wilcoxon alternative: W = {w_alt:.4f}, p = {p_alt:.4e}')"
        ),
        "demo_generator": "one_sample_continuous"
    },

    "one_way_anova": {
        "name": "One-Way Analysis of Variance (ANOVA)",
        "category": "Comparing 3+ Independent Groups",
        "parametric": True,
        "description": "Compares the means of three or more independent groups to test if at least one group mean significantly differs from the others. Uses variance partitioning (Between-Group vs Within-Group variance) to evaluate the omnibus null hypothesis.",
        "when_to_use": [
            "Comparing 3 or more independent treatment groups or cohorts (e.g. Placebo vs Low Dose vs High Dose).",
            "The dependent variable is continuous (ratio or interval scale).",
            "Data within each group is approximately normally distributed (or group sizes N_k >= 30 via CLT).",
            "Variances are approximately equal across groups (homoscedasticity)."
        ],
        "key_assumptions": [
            "Continuous measurement scale (interval or ratio).",
            "Independence of observations (different subjects in each group).",
            "Normality of the dependent variable within each group (robust for N_k >= 30).",
            "Homogeneity of variance (homoscedasticity; if violated, use Welch's ANOVA).",
            "No extreme outliers that distort group means or inflate Mean Square Error."
        ],
        "alternative_test": "Kruskal-Wallis H Test (if non-normal/ordinal or extreme outliers) or Welch's ANOVA (if variances are unequal).",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": (
                "Between Sum of Squares (SS_b): =SUMPRODUCT(counts, (means - grand_mean)^2)\n"
                "Within Sum of Squares (SS_w): =SUMPRODUCT(counts - 1, variances)\n"
                "F-Statistic: =(SS_b / (k - 1)) / (SS_w / (N - k))\n"
                "p-value: =F.DIST.RT(F_cell, k - 1, N - k)\n"
                "Toolpak: Data Tab -> Data Analysis -> 'Anova: Single Factor'"
            ),
            "toolpak_available": True,
            "toolpak_steps": "Data Tab -> Data Analysis -> 'Anova: Single Factor' -> Select Input Range across group columns -> Grouped By Columns -> Check 'Labels in first row' -> OK.",
            "notes": "Toolpak output provides the complete ANOVA table (SS, df, MS, F, p-value, and F-critical). Follow up with post-hoc Tukey HSD or Bonferroni adjustments if p < 0.05."
        },
        "python_snippet": (
            "import scipy.stats as stats\n"
            "import numpy as np\n\n"
            "# 1. Omnibus One-Way ANOVA\n"
            "f_stat, p_val = stats.f_oneway(group_1, group_2, group_3)\n"
            "print(f'One-Way ANOVA: F = {f_stat:.4f}, p-value = {p_val:.4e}')\n\n"
            "# 2. Post-Hoc Pairwise Comparisons (Tukey HSD)\n"
            "tukey_res = stats.tukey_hsd(group_1, group_2, group_3)\n"
            "print(tukey_res)"
        ),
        "demo_generator": "one_way_anova"
    },
    "kruskal_wallis": {
        "name": "Kruskal-Wallis H Test",
        "category": "Comparing 3+ Independent Groups (Non-parametric)",
        "parametric": False,
        "description": "Non-parametric companion to One-Way ANOVA. Evaluates whether 3 or more independent groups originate from identical population distributions by converting observations to global mid-ranks.",
        "when_to_use": [
            "Comparing 3 or more independent, mutually exclusive groups.",
            "Outcome variable is ordinal (e.g. Likert rating scales) or continuous with non-normal distributions.",
            "Extreme outliers or heteroscedasticity violate standard One-Way ANOVA assumptions."
        ],
        "key_assumptions": [
            "Independent random sampling across all groups.",
            "Measurement scale is at least ordinal (ordered categories or continuous ranks).",
            "Adequate group sample sizes (n_k >= 5 for valid Chi-Square approximation of H).",
            "Similar distribution shapes across groups if interpreting as a test of medians (if shapes differ, tests stochastic dominance / mean ranks)."
        ],
        "alternative_test": "One-Way ANOVA (if normality and equal variances hold) or Welch's ANOVA (if normal with unequal variances).",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": "=RANK.AVG(Score, Range, 1) + =SUMIF(...) + =CHISQ.DIST.RT(H, df)",
            "toolpak_available": False,
            "toolpak_steps": "Not built into standard Excel Toolpak. Implemented with global mid-ranks via =RANK.AVG() and =SUMIF() for group rank sums.",
            "notes": "Follow up significant omnibus H-tests with pairwise Dunn-Bonferroni rank tests."
        },
        "python_snippet": (
            "import scipy.stats as stats\n\n"
            "# 1. Kruskal-Wallis H Test across 3 independent groups\n"
            "h_stat, p_val = stats.kruskal(group_1, group_2, group_3)\n"
            "print(f'Kruskal-Wallis H = {h_stat:.4f}, p-value = {p_val:.4e}')\n\n"
            "# 2. Post-Hoc Pairwise Comparisons (Dunn Test / Mann-Whitney with Bonferroni)\n"
            "# For significant results (p < 0.05), perform pairwise comparisons adjusting alpha"
        ),
        "demo_generator": "kruskal_wallis"
    },
    "repeated_measures_anova": {
        "name": "Repeated Measures ANOVA",
        "category": "Comparing 3+ Paired / Repeated Measures Groups",
        "parametric": True,
        "description": "Within-subjects parametric test comparing means across three or more time points or conditions within the exact same subjects. Partitions individual subject differences out of the error term.",
        "when_to_use": [
            "Tracking the exact same subjects across 3 or more occasions, time points, or experimental conditions.",
            "Continuous dependent variable with approximately normal distribution across conditions.",
            "Within-subject design where individual differences between participants can be isolated."
        ],
        "key_assumptions": [
            "Continuous dependent variable measured on the same subjects (matched design).",
            "Normality of repeated measures / pairwise differences.",
            "Sphericity assumption (Mauchly's test: equal variances of pairwise differences). If violated, apply Greenhouse-Geisser or Huynh-Feldt degrees of freedom correction.",
            "Absence of extreme within-subject outliers."
        ],
        "alternative_test": "Greenhouse-Geisser correction (when sphericity fails), Friedman Test (non-parametric when non-normal/ordinal), or Linear Mixed-Effects Models.",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": "Data Analysis Toolpak 'Anova: Two-Factor Without Replication' or native SS decomposition formulas.",
            "toolpak_available": True,
            "toolpak_steps": "Data tab -> Data Analysis -> 'Anova: Two-Factor Without Replication' -> Input range (Rows = Subjects, Columns = Conditions) -> Labels in first row -> OK. The 'Columns' row outputs the omnibus within-subjects F-statistic and p-value.",
            "notes": "Follow up significant omnibus tests with Bonferroni-adjusted paired t-tests across time points: =MIN(1.0, 3 * T.TEST(range_1, range_2, 2, 1))."
        },
        "python_snippet": (
            "import scipy.stats as stats\n"
            "from statsmodels.stats.anova import AnovaRM\n\n"
            "# 1. Repeated Measures ANOVA (using statsmodels or manual SS decomposition)\n"
            "aov = AnovaRM(df_long, depvar='Score', subject='Subject_ID', within=['Time_Point'])\n"
            "res = aov.fit()\n"
            "print(res.summary())\n\n"
            "# 2. Post-Hoc Pairwise Paired t-tests with Bonferroni Correction\n"
            "# Follow up significant omnibus tests with paired comparisons across time points"
        ),
        "demo_generator": "repeated_measures_anova"
    },
    "friedman_test": {
        "name": "Friedman Test",
        "category": "Comparing 3+ Paired / Repeated Measures Groups (Non-parametric)",
        "parametric": False,
        "description": "Non-parametric counterpart to Repeated Measures ANOVA. Evaluates differences across 3 or more paired groups using within-subject ranks.",
        "when_to_use": [
            "Comparing 3 or more repeated measurements or conditions in the same subjects.",
            "Data is ordinal or skewed continuous violating normality or sphericity."
        ],
        "key_assumptions": [
            "One group measured on 3+ occasions or matched blocks.",
            "Dependent variable is at least ordinal.",
            "Blocks are mutually independent."
        ],
        "alternative_test": "Repeated Measures ANOVA (if parametric assumptions hold) or Wilcoxon signed-rank for post-hoc pairs.",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": "=CHISQ.DIST.RT((12/(N*k*(k+1)))*SUMSQ(R_1..R_k) - 3*N*(k+1), k-1)",
            "toolpak_available": False,
            "toolpak_steps": "Not built directly into Toolpak. Computed using within-row ranking (=RANK.AVG), column rank sums, and =CHISQ.DIST.RT.",
            "notes": "Follow up significant omnibus tests with Bonferroni-adjusted pairwise Wilcoxon signed-rank comparisons."
        },
        "python_snippet": (
            "import scipy.stats as stats\n\n"
            "# 1. Friedman Test across 3+ repeated conditions\n"
            "q_stat, p_val = stats.friedmanchisquare(time_1, time_2, time_3)\n"
            "print(f'Friedman Q: {q_stat:.4f}, p-value: {p_val:.4e}')\n\n"
            "# 2. Post-Hoc Pairwise Wilcoxon Signed-Rank Tests (Bonferroni Corrected)\n"
            "# For significant results (p < 0.05), perform pairwise comparisons adjusting alpha: p_adj = min(1.0, 3 * p_raw)"
        ),
        "demo_generator": "friedman_test"
    },
    "chi_square_independence": {
        "name": "Pearson's Chi-Square Test of Independence",
        "category": "Categorical Proportions & Contingency Tables",
        "parametric": False,
        "description": "Evaluates whether an association exists between two categorical variables in a 2x2 or RxC contingency table by comparing observed cell counts to expected counts under independence.",
        "when_to_use": [
            "Both independent and dependent variables are categorical (nominal or ordinal).",
            "Observations are independent (each subject cross-classified into exactly one cell).",
            "Adequate sample size satisfying Cochran's rule: all expected cell counts >= 5.0 for a 2x2 table (or >= 80% of cells >= 5.0 with no cell < 1.0 in RxC tables)."
        ],
        "key_assumptions": [
            "Mutually exclusive and exhaustive categories (each observation in one cell only).",
            "Independence of observations (no matched pairs, clusters, or repeated measurements).",
            "Expected cell frequency adequacy: Cochran's criterion requires E_ij >= 5 in each cell.",
            "Adequate total sample size (N >= 20, preferably N >= 40 for asymptotic power)."
        ],
        "alternative_test": "Fisher's Exact Test (if any expected cell count is < 5.0) or McNemar's Test (if data represents paired or matched binary responses).",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": "=CHISQ.TEST(actual_range, expected_range) or =CHISQ.DIST.RT(chi2_stat, df)",
            "toolpak_available": False,
            "toolpak_steps": "Not built into standard Excel Data Analysis Toolpak. Computed using native worksheet formulas:\n1. Create an Expected Frequencies table: For each cell, enter =(RowTotal * ColTotal) / GrandTotal.\n2. Verify that all expected cells are >= 5.0.\n3. Compute p-value directly: =CHISQ.TEST(actual_range, expected_range).\n4. Compute manual Chi-square statistic: For each cell calculate =(Actual - Expected)^2 / Expected, sum them up with =SUM(), then compute p-value via =CHISQ.DIST.RT(sum_cell, 1).\n5. Effect sizes: Odds Ratio =(a*d)/(b*c), Relative Risk =(a/(a+b))/(c/(c+d)), Phi =SQRT(chi2_stat / GrandTotal).",
            "notes": "=CHISQ.TEST directly returns the upper-tail p-value. If any expected frequency < 5, report Fisher's Exact test instead."
        },
        "python_snippet": (
            "import scipy.stats as stats\n"
            "import numpy as np\n"
            "from scipy.stats.contingency import expected_freq\n\n"
            "# 1. Define 2x2 contingency table: [[a, b], [c, d]]\n"
            "table = np.array([\n"
            "    [45, 55],   # Treatment: 45 events, 55 non-events (N=100)\n"
            "    [20, 80]    # Control:   20 events, 80 non-events (N=100)\n"
            "])\n\n"
            "# 2. Assumption Check: Cochran's Criterion (Expected Frequencies >= 5.0)\n"
            "expected = expected_freq(table)\n"
            "min_expected = np.min(expected)\n"
            "print(f'Expected Counts:\\n{expected.round(2)}')\n"
            "print(f'Minimum Expected Count: {min_expected:.2f}')\n\n"
            "if min_expected >= 5.0 and np.sum(table) >= 20:\n"
            "    print('[ASSURANCE - PASS] Cochran rule satisfied: Proceeding with Pearson Chi-Square.')\n"
            "    chi2_stat, p_val, dof, _ = stats.chi2_contingency(table, correction=False)\n"
            "    print(f'Pearson Chi2: {chi2_stat:.4f}, p-value: {p_val:.4e}, df: {dof}')\n"
            "else:\n"
            "    print('[INAPPROPRIATE - FAIL] Sparse expected frequencies (< 5.0): Switching to Fisher Exact Test.')\n"
            "    odds_ratio, p_val = stats.fisher_exact(table, alternative=\"two-sided\")\n"
            "    print(f'Fisher Exact Odds Ratio: {odds_ratio:.4f}, p-value: {p_val:.4e}')"
        ),
        "demo_generator": "contingency_table_2x2"
    },
    "fisher_exact": {
        "name": "Fisher's Exact Test",
        "category": "Categorical Proportions & Contingency Tables (Small Samples)",
        "parametric": False,
        "description": "Calculates the exact hypergeometric probability of observing the 2x2 contingency table under the null hypothesis of independence. Specifically recommended when sample sizes are small or cell frequencies are sparse (expected frequencies < 5.0).",
        "when_to_use": [
            "Examining association between 2 binary categorical variables arranged in a 2x2 contingency table.",
            "Small sample sizes where one or more expected cell counts is less than 5.0 (violating Cochran's criterion for Chi-Square).",
            "Rare events or sparse clinical outcome data where asymptotic continuous approximations fail."
        ],
        "key_assumptions": [
            "Mutually exclusive and exhaustive categories (each subject in exactly one cell).",
            "Independent observations (no repeated measures or matched pairs).",
            "2x2 contingency table design.",
            "Conditioning on observed marginal totals."
        ],
        "alternative_test": "Pearson's Chi-Square Test of Independence (if expected cell counts are all >= 5.0) or McNemar's Test (if binary data is paired/matched).",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": "=HYPGEOM.DIST(sample_s, number_sample, population_s, number_pop, cumulative)",
            "toolpak_available": False,
            "toolpak_steps": "Not built into standard Excel Data Analysis Toolpak. Excel computes Fisher's exact probabilities via the Hypergeometric distribution function:\n1. For cell a: Point probability =HYPGEOM.DIST(a, Row1_Total, Col1_Total, Grand_Total, FALSE).\n2. Left-tail probability P(X <= a): =HYPGEOM.DIST(a, Row1_Total, Col1_Total, Grand_Total, TRUE).\n3. Right-tail probability P(X >= a): =1 - HYPGEOM.DIST(a - 1, Row1_Total, Col1_Total, Grand_Total, TRUE).\n4. Two-sided p-value: Construct a column with all possible integer values of k from MAX(0, C1 - R2) to MIN(R1, C1). Compute =HYPGEOM.DIST(k, R1, C1, N, FALSE) for each. Then sum probabilities where P(k) <= observed_P using =SUMIF(prob_range, '<=' & observed_P_cell).\n5. Odds Ratio: =(a * d) / (b * c).",
            "notes": "=HYPGEOM.DIST with cumulative=FALSE gives the exact probability of obtaining that specific table configuration under H0."
        },
        "python_snippet": (
            "import scipy.stats as stats\n"
            "import numpy as np\n"
            "from scipy.stats.contingency import expected_freq\n\n"
            "# 1. Define 2x2 contingency table: [[a, b], [c, d]]\n"
            "table = np.array([\n"
            "    [4, 16],   # Treatment: 4 events, 16 non-events (N=20)\n"
            "    [1, 19]    # Control:   1 event,  19 non-events (N=20)\n"
            "])\n\n"
            "# 2. Assumption Check: Verify Expected Cell Frequencies & Sample Size\n"
            "expected = expected_freq(table)\n"
            "min_expected = np.min(expected)\n"
            "n_total = np.sum(table)\n"
            "print(f'Expected Table:\\n{expected.round(2)}')\n"
            "print(f'Minimum Expected Count: {min_expected:.2f}, Total N = {n_total}')\n\n"
            "if min_expected < 5.0 or n_total < 20:\n"
            "    print('[ASSURANCE - PASS] Sparse expected cell counts detected (Min E < 5.0): Fisher Exact Test is strictly required.')\n"
            "else:\n"
            "    print('[WARNING] Large sample size (Min E >= 5.0, N >= 20): Fisher Exact Test is valid; Pearson Chi-Square is standard.')\n\n"
            "# 3. Execute Fisher's Exact Test\n"
            "odds_ratio, p_two_sided = stats.fisher_exact(table, alternative='two-sided')\n"
            "print(f'Exact Odds Ratio: {odds_ratio:.4f}, Two-Sided p-value: {p_two_sided:.4e}')"
        ),
        "demo_generator": "fisher_exact"
    },
    "mcnemar_test": {
        "name": "McNemar's Test",
        "category": "Paired Categorical / Binary Data",
        "parametric": False,
        "description": "Tests for changes in proportions for paired binary outcomes (e.g., matched case-control studies or before-after intervention outcomes).",
        "when_to_use": [
            "Paired or matched observations.",
            "Binary outcome variable (e.g. Success vs Failure, Positive vs Negative).",
            "Tests whether discordant pairs (b vs c in 2x2 table) are symmetric."
        ],
        "key_assumptions": [
            "Paired dependent samples.",
            "Binary response.",
            "Discordant cells (b + c) should preferably be >= 25; otherwise use exact binomial version."
        ],
        "alternative_test": "Cochran's Q Test (if evaluating across 3 or more paired timepoints).",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": (
                "Step 1: In a 2x2 table, identify discordant cell b (row 1, col 2) and discordant cell c (row 2, col 1).\n"
                "Step 2 (Edwards' Continuity-Corrected Chi-Square): =((ABS(b - c) - 1)^2) / (b + c)\n"
                "Step 3 (p-value): =CHISQ.DIST.RT(chi_square_cell, 1)\n"
                "Alternative (Exact Binomial p-value when b + c < 25): =2 * BINOM.DIST(MIN(b, c), b + c, 0.5, TRUE)"
            ),
            "toolpak_available": False,
            "toolpak_steps": "Standard Excel arithmetic formulas directly compute both exact binomial and continuity-corrected McNemar tests.",
            "notes": "Concordant pairs (cells a and d) provide no information on marginal change and cancel out of McNemar's test."
        },
        "python_snippet": (
            "import numpy as np\n"
            "from statsmodels.stats.contingency_tables import mcnemar\n\n"
            "# 1. Define paired 2x2 table: [[both_pos, pos_neg], [neg_pos, both_neg]]\n"
            "table = np.array([\n"
            "    [15,  5],   # [a: Pre+, Post+ (Concordant); b: Pre+, Post- (Discordant)]\n"
            "    [25, 55]    # [c: Pre-, Post+ (Discordant); d: Pre-, Post- (Concordant)]\n"
            "])\n"
            "b, c = table[0, 1], table[1, 0]\n"
            "n_disc = b + c\n"
            "print(f'Total Discordant Pairs (b + c): {n_disc}')\n\n"
            "# 2. Assumption Check: Discordant Pair Count Adequacy\n"
            "if n_disc >= 25:\n"
            "    print(f'[ASSURANCE - PASS] Discordant count adequate (b + c = {n_disc} >= 25): Edwards Chi-Square is valid.')\n"
            "    res = mcnemar(table, exact=False, correction=True)\n"
            "    print(f'Edwards Chi2: {res.statistic:.4f}, Asymptotic p-value: {res.pvalue:.4e}')\n"
            "else:\n"
            "    print(f'[WARNING] Sparse discordant count (b + c = {n_disc} < 25): Switching to Exact Binomial Test.')\n"
            "    res = mcnemar(table, exact=True)\n"
            "    print(f'Exact Binomial p-value: {res.pvalue:.4e}')\n\n"
            "# 3. Marginal Shift & Conditional Odds Ratio\n"
            "n = np.sum(table)\n"
            "risk_diff = (c - b) / n\n"
            "odds_ratio = c / b if b > 0 else np.nan\n"
            "print(f'Marginal Shift: {risk_diff*100:+.1f}%, Conditional Odds Ratio: {odds_ratio:.3f}')"
        ),
        "demo_generator": "mcnemar_test"
    },
    "pearson_correlation": {
        "name": "Pearson Correlation Coefficient (r and R²)",
        "category": "Relationships & Associations (Linear)",
        "parametric": True,
        "description": "Measures the strength and direction of a linear relationship between two continuous variables.",
        "when_to_use": [
            "Both variables are continuous (ratio or interval scale).",
            "Scatterplot displays an approximately linear relationship.",
            "Both variables follow a bivariate normal distribution and contain no influential outliers."
        ],
        "key_assumptions": [
            "Linearity between variables.",
            "Bivariate normality.",
            "Homoscedasticity (equal variance of residuals across the range).",
            "No extreme outliers."
        ],
        "alternative_test": "Spearman Rank Correlation (if non-linear monotonic, non-normal, or contains outliers).",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": (
                "Step 1 (Correlation r & R²): =CORREL(array1, array2) or =PEARSON(array1, array2) [for R² use =RSQ(array1, array2)]\n"
                "Step 2 (Hypothesis Test t-stat): =r * SQRT(COUNT(array1) - 2) / SQRT(1 - r^2)\n"
                "Step 3 (Two-Tailed p-value): =T.DIST.2T(ABS(t_cell), COUNT(array1) - 2)\n"
                "Step 4 (Fisher's z 95% CI): =FISHER(r) gives z; SE = 1/SQRT(n - 3); CI limits = =FISHERINV(z ± 1.96 * SE)"
            ),
            "toolpak_available": True,
            "toolpak_steps": "Data Tab -> Data Analysis -> 'Correlation' -> Select columns -> OK. Or 'Regression' to obtain R², ANOVA table, and p-values.",
            "notes": "Both =CORREL and =PEARSON calculate sample Pearson r. Use =RSQ to get the coefficient of determination (explained variance)."
        },
        "python_snippet": (
            "import numpy as np\n"
            "from scipy import stats\n\n"
            "# 1. Assumption Check: Normality (Shapiro-Wilk) and Outliers\n"
            "shapiro_x = stats.shapiro(x)\n"
            "shapiro_y = stats.shapiro(y)\n"
            "outliers_x = np.sum(np.abs(stats.zscore(x)) > 3.0)\n"
            "outliers_y = np.sum(np.abs(stats.zscore(y)) > 3.0)\n"
            "has_outliers = (outliers_x + outliers_y) > 0\n"
            "is_normal = (shapiro_x.pvalue >= 0.05) and (shapiro_y.pvalue >= 0.05)\n\n"
            "if is_normal and not has_outliers:\n"
            "    print('[ASSURANCE - PASS] Bivariate normality satisfied and no severe outliers. Pearson correlation is valid.')\n"
            "    res = stats.pearsonr(x, y)\n"
            "    ci_low, ci_high = res.confidence_interval(confidence_level=0.95)\n"
            "    print(f'Pearson r: {res.statistic:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}]), p-value: {res.pvalue:.4e}')\n"
            "    print(f'R-squared: {res.statistic**2 * 100:.1f}% shared variance')\n"
            "else:\n"
            "    print('[INAPPROPRIATE - FAIL] Non-normality or extreme outliers detected. Switching to Spearman Rank Correlation.')\n"
            "    rho, p_val = stats.spearmanr(x, y)\n"
            "    print(f'Spearman rho: {rho:.4f}, p-value: {p_val:.4e}')"
        ),
        "demo_generator": "pearson_correlation"
    },
    "spearman_correlation": {
        "name": "Spearman Rank Correlation Coefficient (ρ)",
        "category": "Relationships & Associations (Monotonic / Non-parametric)",
        "parametric": False,
        "description": "Measures the monotonic relationship between two variables using ranked values. Robust against outliers and does not assume linear relationships or normality.",
        "when_to_use": [
            "Variables are ordinal OR continuous but violating bivariate normality.",
            "The relationship is monotonic (as X increases, Y consistently increases or decreases, even if not in a straight line).",
            "Outliers are present that would distort Pearson's r."
        ],
        "key_assumptions": [
            "Observations are independent.",
            "Variables are at least ordinal.",
            "Monotonic relationship."
        ],
        "alternative_test": "Kendall's Tau (often preferred for small datasets with many tied ranks) or Pearson correlation (if linear & normal).",
        "excel_guide": {
            "has_native_formula": False,
            "formula_syntax": (
                "Step 1 (Rank Transformations): In new columns, compute fractional ranks using =RANK.AVG(x1, $X$1:$X$N, 1) and =RANK.AVG(y1, $Y$1:$Y$N, 1) [ascending order]\n"
                "Step 2 (Spearman rho via Correlation of Ranks): =CORREL(rank_x_range, rank_y_range)\n"
                "Step 3 (Hypothesis Test t-stat): =rho_cell * SQRT(COUNT(rank_x_range) - 2) / SQRT(1 - rho_cell^2)\n"
                "Step 4 (Two-Tailed p-value): =T.DIST.2T(ABS(t_cell), COUNT(rank_x_range) - 2)\n"
                "Alternative (Untied Formula): =1 - (6 * SUMSQ(rank_diff_range)) / (N * (N^2 - 1))"
            ),
            "toolpak_available": False,
            "toolpak_steps": "Calculate rank columns for both variables using =RANK.AVG(), then run Data Analysis -> 'Correlation' on the ranked columns.",
            "notes": "Spearman's rho is mathematically identical to Pearson's r computed on the ranked values. Using =CORREL on =RANK.AVG automatically accounts for ties."
        },
        "python_snippet": (
            "import numpy as np\n"
            "from scipy import stats\n\n"
            "# 1. Assumption Check: Tied Ranks and Sample Size\n"
            "n = len(x)\n"
            "ties_x = (n - len(np.unique(x))) / n * 100\n"
            "ties_y = (n - len(np.unique(y))) / n * 100\n"
            "max_ties = max(ties_x, ties_y)\n"
            "print(f'Sample Size: N = {n}, Maximum Tied Ranks: {max_ties:.1f}%')\n\n"
            "if max_ties > 25.0 or n < 30:\n"
            "    print(f'[WARNING] High tied rank proportion ({max_ties:.1f}% > 25%) or small N: Kendall Tau-b is recommended.')\n"
            "    res_tau = stats.kendalltau(x, y)\n"
            "    print(f'Kendall Tau-b: {res_tau.statistic:.4f}, p-value = {res_tau.pvalue:.4e}')\n"
            "else:\n"
            "    print('[ASSURANCE - PASS] Low ties (< 25%) and adequate sample size: Spearman rho is optimal.')\n\n"
            "# 2. Execute Spearman Rank Correlation\n"
            "res = stats.spearmanr(x, y)\n"
            "rho, p_val = res.statistic, res.pvalue\n"
            "df = n - 2\n"
            "t_stat = rho * np.sqrt(df) / np.sqrt(1 - rho**2)\n"
            "z = np.arctanh(np.clip(rho, -0.99999, 0.99999))\n"
            "se_z = 1.06 / np.sqrt(n - 3)\n"
            "ci_lower, ci_upper = np.tanh(z - 1.96 * se_z), np.tanh(z + 1.96 * se_z)\n"
            "print(f'Spearman rho: {rho:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]), t({df}) = {t_stat:.4f}, p = {p_val:.4e}')\n"
            "print(f'Shared Rank Variance: {rho**2 * 100:.1f}%')"
        ),
        "demo_generator": "spearman_correlation"
    },
    "linear_regression": {
        "name": "Simple / Multiple Linear Regression (OLS)",
        "category": "Predicting Continuous Outcome (Regression)",
        "parametric": True,
        "description": "Models the relationship between a continuous outcome variable and one or more predictor variables (continuous or dummy-coded categorical).",
        "when_to_use": [
            "You want to quantify the effect of predictors or predict a continuous dependent variable.",
            "Evaluating independent effects while adjusting for potential confounders."
        ],
        "key_assumptions": [
            "Linear relationship between predictors and outcome.",
            "Independence of residuals (no autocorrelation).",
            "Homoscedasticity (constant variance of residuals).",
            "Normality of residuals.",
            "No severe multicollinearity among predictors."
        ],
        "alternative_test": "Robust Regression, Generalized Additive Models (GAM), or Quantile Regression.",
        "excel_guide": {
            "has_native_formula": True,
            "formula_syntax": (
                "Step 1 (Data Analysis Toolpak): Data -> Data Analysis -> 'Regression' -> Input Y Range (outcome) & Input X Range (adjacent columns for X1, X2, etc.) -> Check 'Labels' and 'Residuals' -> OK\n"
                "Step 2 (Full Model via Dynamic Array): =LINEST(y_range, x_range, TRUE, TRUE) [outputs slopes, intercept, SEs, R², F, df, SS_reg, SS_res]\n"
                "Step 3 (Simple Bivariate Formulas): Slope = =SLOPE(y_range, x_range) ; Intercept = =INTERCEPT(y_range, x_range) ; R² = =RSQ(y_range, x_range)\n"
                "Step 4 (Model Predictions): =intercept_cell + (b1_cell * x1_val) + (b2_cell * x2_val) or =FORECAST.LINEAR(x_val, y_range, x_range)"
            ),
            "toolpak_available": True,
            "toolpak_steps": "Data Tab -> Data Analysis -> 'Regression' -> Input Y Range & Input X Range -> Check 'Residuals' -> OK.",
            "notes": "Toolpak regression produces complete ANOVA table, R-squared, adjusted R-squared, coefficients, standard errors, and p-values for all predictors."
        },
        "python_snippet": (
            "import numpy as np\n"
            "import pandas as pd\n"
            "import scipy.stats as stats\n"
            "import statsmodels.api as sm\n"
            "from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow\n"
            "from statsmodels.stats.stattools import durbin_watson\n"
            "from statsmodels.stats.outliers_influence import variance_inflation_factor\n\n"
            "# 1. Fit initial OLS Model & Extract Residuals\n"
            "X_const = sm.add_constant(X)\n"
            "model = sm.OLS(y, X_const).fit()\n"
            "res = model.resid\n"
            "n, k = X.shape\n\n"
            "# 2. Evaluate Gauss-Markov Assumptions\n"
            "_, rb_p = linear_rainbow(model)\n"
            "dw = durbin_watson(res)\n"
            "_, bp_p, _, _ = het_breuschpagan(res, X_const)\n"
            "sw_stat, sw_p = stats.shapiro(res[:min(n, 5000)])\n"
            "vifs = [variance_inflation_factor(X_const.values, i) for i in range(1, X_const.shape[1])] if k > 1 else [1.0]\n"
            "max_vif = max(vifs)\n\n"
            "# 3. Conditional Print Statements (Assurance / Warning / Routing)\n"
            "print(f\"[Linearity] Rainbow test p = {rb_p:.4f} -> {'PASS' if rb_p >= 0.05 else 'WARN (Curvature detected)'}\")\n"
            "print(f\"[Independence] Durbin-Watson = {dw:.3f} -> {'PASS' if 1.5 <= dw <= 2.5 else 'WARN (Autocorrelation)'}\")\n"
            "print(f\"[Homoscedasticity] Breusch-Pagan p = {bp_p:.4f} -> {'PASS' if bp_p >= 0.05 else 'VIOLATED (Funnel variance)'}\")\n"
            "print(f\"[Normality] Shapiro-Wilk p = {sw_p:.4f} -> {'PASS' if sw_p >= 0.05 or n >= 30 else 'VIOLATED'}\")\n"
            "print(f\"[Multicollinearity] Max VIF = {max_vif:.2f} -> {'PASS' if max_vif < 5 else 'WARN/FAIL (Collinear)'}\")\n\n"
            "# 4. Final Routing\n"
            "if bp_p >= 0.05:\n"
            "    print('\\n--> [VERDICT] Assumptions satisfied: Reporting standard OLS model:')\n"
            "    print(model.summary())\n"
            "else:\n"
            "    print('\\n--> [VERDICT] Heteroscedasticity detected! Reporting HC3 Robust Standard Errors:')\n"
            "    print(model.get_robustcov_results(cov_type='HC3').summary())"
        ),
        "demo_generator": "linear_regression"
    },
    "binary_logistic_regression": {
        "name": "Binary Logistic Regression",
        "category": "Predicting Categorical Outcome (Classification / Odds)",
        "parametric": True,
        "description": "Models the probability of a binary outcome (e.g. Disease Present / Absent) as a function of one or more predictor variables, outputting adjusted Odds Ratios.",
        "when_to_use": [
            "Outcome is binary (0 or 1, Yes or No).",
            "You have continuous and/or categorical predictors.",
            "You want to control for confounding variables in observational or clinical data."
        ],
        "key_assumptions": [
            "Binary dependent variable.",
            "Independence of observations.",
            "Linearity of independent variables with the log-odds (logit transformation).",
            "No severe multicollinearity.",
            "Adequate sample size (rule of thumb: >= 10-20 events per variable)."
        ],
        "alternative_test": "Exact Logistic Regression (for very sparse events) or Penalized Logistic Regression (Ridge/Lasso).",
        "excel_guide": {
            "has_native_formula": False,
            "formula_syntax": "Not available as standard formulas; can be solved via Excel Solver maximizing log-likelihood.",
            "toolpak_available": False,
            "toolpak_steps": "Not available in standard Toolpak without Real Statistics or XLSTAT add-ins.",
            "notes": "Standardly run in Python (statsmodels.Logit) or R (glm(family=binomial))."
        },
        "python_snippet": (
            "import numpy as np\n"
            "import pandas as pd\n"
            "import scipy.stats as stats\n"
            "import statsmodels.api as sm\n"
            "from statsmodels.stats.outliers_influence import variance_inflation_factor\n\n"
            "def run_binary_logistic_pipeline(X, y, feature_names=None, alpha=0.05):\n"
            "    \"\"\"Automated Binary Logistic Regression Pipeline with comprehensive\n"
            "    assumption verification (Dichotomy, EPV, Multicollinearity, Separation,\n"
            "    Residual Outliers) and conditional assurance/warning printouts.\"\"\"\n"
            "    X = np.asarray(X)\n"
            "    y = np.asarray(y)\n"
            "    n, k = X.shape\n"
            "    if feature_names is None:\n"
            "        feature_names = [f'X{i+1}' for i in range(k)]\n"
            "    \n"
            "    # 1. Assumption Audits\n"
            "    unique_y = np.unique(y)\n"
            "    is_binary = len(unique_y) == 2 and set(unique_y).issubset({0, 1})\n"
            "    n_events = min(np.sum(y == 1), np.sum(y == 0))\n"
            "    epv = n_events / k\n"
            "    \n"
            "    X_const = sm.add_constant(X)\n"
            "    if k > 1:\n"
            "        vifs = [variance_inflation_factor(X_const, i) for i in range(1, k + 1)]\n"
            "        max_vif = max(vifs)\n"
            "    else:\n"
            "        max_vif = 1.0\n"
            "    \n"
            "    # 2. Fit Logistic Regression via Maximum Likelihood\n"
            "    model = sm.Logit(y, X_const).fit(disp=False)\n"
            "    p_pred = model.predict(X_const)\n"
            "    pearson_res = (y - p_pred) / np.sqrt(p_pred * (1.0 - p_pred))\n"
            "    extreme_outliers = np.sum(np.abs(pearson_res) > 3.0)\n"
            "    is_separated = np.any(model.bse > 25.0) or np.any(np.abs(model.params) > 15.0)\n"
            "    \n"
            "    # 3. Conditional Diagnostic Feedback\n"
            "    print('=' * 70)\n"
            "    print('=== BINARY LOGISTIC REGRESSION ASSUMPTION AUDIT ===')\n"
            "    print('=' * 70)\n"
            "    \n"
            "    if is_binary:\n"
            "        print(f'[ASSURANCE - PASS] Binary outcome verified (0: {np.sum(y==0)}, 1: {np.sum(y==1)}).')\n"
            "    else:\n"
            "        print('[INAPPROPRIATE - FAIL] Outcome is not strictly binary (0/1)!')\n"
            "        \n"
            "    if epv >= 10.0:\n"
            "        print(f'[ASSURANCE - PASS] Adequate sample size: EPV = {epv:.1f} >= 10 (Events = {n_events}, k = {k}).')\n"
            "    elif epv >= 5.0:\n"
            "        print(f'[WARNING] Borderline sample size: EPV = {epv:.1f} < 10. Consider penalized likelihood.')\n"
            "    else:\n"
            "        print(f'[INAPPROPRIATE - FAIL] Sparse events: EPV = {epv:.1f} < 5. Standard MLE is biased!')\n"
            "        \n"
            "    if max_vif < 5.0:\n"
            "        print(f'[ASSURANCE - PASS] Multicollinearity low: Max VIF = {max_vif:.2f} < 5.0.')\n"
            "    elif max_vif < 10.0:\n"
            "        print(f'[WARNING] Moderate multicollinearity: Max VIF = {max_vif:.2f} (5.0 - 10.0).')\n"
            "    else:\n"
            "        print(f'[INAPPROPRIATE - FAIL] Severe multicollinearity: Max VIF = {max_vif:.2f} >= 10.0!')\n"
            "        \n"
            "    if not is_separated:\n"
            "        print('[ASSURANCE - PASS] Complete separation absent: Coefficients converged cleanly.')\n"
            "    else:\n"
            "        print('[INAPPROPRIATE - FAIL] Quasi-complete separation detected! Standard errors exploded.')\n"
            "        \n"
            "    if extreme_outliers == 0:\n"
            "        print(f'[ASSURANCE - PASS] Pearson residuals well-behaved (Max |r| = {np.max(np.abs(pearson_res)):.2f} <= 3.0).')\n"
            "    else:\n"
            "        print(f'[WARNING] {extreme_outliers} observation(s) exhibit Pearson residuals |r| > 3.0.')\n"
            "        \n"
            "    # 4. Final Verdict and Model Output\n"
            "    print('\\n' + '=' * 70)\n"
            "    if epv >= 5.0 and max_vif < 10.0 and not is_separated:\n"
            "        print('--> [VERDICT] Assumptions satisfied: Reporting Binary Logistic Regression Model.')\n"
            "        print('=' * 70)\n"
            "        print(model.summary())\n"
            "        odds_ratios = pd.DataFrame({\n"
            "            'Odds_Ratio': np.exp(model.params),\n"
            "            '95%_CI_Lower': np.exp(model.conf_int()[0]),\n"
            "            '95%_CI_Upper': np.exp(model.conf_int()[1]),\n"
            "            'p-value': model.pvalues\n"
            "        })\n"
            "        print('\\n--- ADJUSTED ODDS RATIOS (95% CI) ---')\n"
            "        print(odds_ratios)\n"
            "    else:\n"
            "        print('--> [VERDICT] Switching to Penalized Logistic Regression (L2 Ridge / Firth).')\n"
            "        print('=' * 70)\n"
            "        from sklearn.linear_model import LogisticRegression\n"
            "        pen_model = LogisticRegression(penalty='l2', C=1.0).fit(X, y)\n"
            "        print(f'Penalized Intercept: {pen_model.intercept_[0]:.4f}')\n"
            "        for idx, coef in enumerate(pen_model.coef_[0]):\n"
            "            print(f'Penalized {feature_names[idx]}: Coef = {coef:.4f}, OR = {np.exp(coef):.4f}')\n"
            "\n"
            "# Example execution:\n"
            "# run_binary_logistic_pipeline(X, y)"
        ),
        "demo_generator": "binary_classification"
    },
    "log_rank_test": {
        "name": "Log-Rank Test (Kaplan-Meier Survival Comparison)",
        "category": "Survival & Time-to-Event Analysis",
        "parametric": False,
        "description": "Compares survival distributions of two or more groups with censored time-to-event data.",
        "when_to_use": [
            "Outcome is time-to-event (e.g., time to relapse, death, machine failure) with right-censored data.",
            "Comparing survival curves across 2 or more exposure groups."
        ],
        "key_assumptions": [
            "Non-informative censoring (censored subjects have the same survival chances as ongoing subjects).",
            "Proportional hazards / survival curves do not cross significantly.",
            "Event times are precisely measured."
        ],
        "alternative_test": "Cox Proportional Hazards Model (to adjust for multiple covariates) or Gehan-Breslow-Wilcoxon test (weights earlier events).",
        "excel_guide": {
            "has_native_formula": False,
            "formula_syntax": "Requires extensive manual life-table columns (risk sets, observed events, expected events).",
            "toolpak_available": False,
            "toolpak_steps": "Not built into Toolpak; dedicated survival analysis package recommended.",
            "notes": "Use Python 'lifelines' library or R 'survival' package."
        },
        "python_snippet": (
            "from lifelines.statistics import logrank_test\n\n"
            "# Log-rank test for survival data\n"
            "results = logrank_test(durations_A, durations_B, event_observed_A, event_observed_B)\n"
            "print(f'Test Statistic: {results.test_statistic:.4f}, p-value: {results.p_value:.4e}')"
        ),
        "demo_generator": "survival_analysis"
    }
}


def recommend_statistical_test(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates user answers and returns the recommended statistical test metadata.

    Args:
        answers: Dictionary containing user responses to the questionnaire.
                 Expected keys:
                 - primary_objective: 'compare_groups' | 'contingency_proportions' |
                                      'examine_relationship' | 'time_to_event'
                 - dependent_type: 'continuous' | 'ordinal' | 'binary' | 'nominal'
                 - num_groups: '1' | '2' | '3+'
                 - group_design: 'independent' | 'paired'
                 - normality: 'normal' | 'non_normal' | 'unknown'
                 - equal_variance: 'equal' | 'unequal' | 'unknown'
                 - cell_counts: 'adequate' | 'small'
                 - relationship_goal: 'correlation' | 'predict_continuous' | 'predict_binary'
                 - linearity: 'linear' | 'monotonic_or_ranked'
                 - survival_goal: 'compare_curves' | 'model_covariates'

    Returns:
        Dictionary containing recommendation details, reasoning, and test metadata.
    """
    objective = answers.get("primary_objective")
    test_key: Optional[str] = None
    reasoning: List[str] = []

    if objective == "compare_groups":
        num_groups = answers.get("num_groups", "2")
        dep_type = answers.get("dependent_type", "continuous")
        normality = answers.get("normality", "normal")
        design = answers.get("group_design", "independent")
        equal_var = answers.get("equal_variance", "unknown")

        if num_groups == "1":
            if dep_type == "continuous" and normality == "normal":
                test_key = "one_sample_t_test"
                reasoning.append("Comparing a single continuous sample against a benchmark population mean.")
                reasoning.append("Data satisfies normality assumption.")
            else:
                test_key = "wilcoxon_signed_rank"
                reasoning.append("Single group compared to a theoretical median benchmark.")
                reasoning.append("Data is ordinal or non-normally distributed; non-parametric rank test recommended.")

        elif num_groups == "2":
            if design == "independent":
                if dep_type == "continuous" and normality == "normal":
                    if equal_var == "equal":
                        test_key = "students_t_test"
                        reasoning.append("Comparing 2 independent groups with normally distributed continuous data.")
                        reasoning.append("Equal population variances verified (homoscedasticity).")
                    else:
                        test_key = "welch_t_test"
                        reasoning.append("Comparing 2 independent groups with continuous data.")
                        reasoning.append("Welch's t-test protects against Type I error when variances may differ.")
                else:
                    test_key = "mann_whitney_u"
                    reasoning.append("Comparing 2 independent groups.")
                    reasoning.append("Outcome is ordinal or continuous with non-normal distribution (ranks compared).")
            else:  # paired / repeated measures
                if dep_type == "continuous" and normality == "normal":
                    test_key = "paired_t_test"
                    reasoning.append("Comparing 2 paired or matched measurements from the same subjects.")
                    reasoning.append("Differences between pairs are normally distributed.")
                else:
                    test_key = "wilcoxon_paired"
                    reasoning.append("Comparing 2 paired measurements from the same subjects.")
                    reasoning.append("Differences are ordinal or non-normally distributed; signed-rank test is appropriate.")

        elif num_groups == "3+":
            if design == "independent":
                if dep_type == "continuous" and normality == "normal":
                    test_key = "one_way_anova"
                    reasoning.append("Comparing means across 3 or more independent groups.")
                    reasoning.append("Parametric assumptions of normality and homogeneity of variance are met.")
                else:
                    test_key = "kruskal_wallis"
                    reasoning.append("Comparing 3 or more independent groups.")
                    reasoning.append("Data is ordinal or continuous with non-normal distributions.")
            else:  # paired / repeated measures
                if dep_type == "continuous" and normality == "normal":
                    test_key = "repeated_measures_anova"
                    reasoning.append("Tracking 3 or more measurements within the same subjects over time or conditions.")
                    reasoning.append("Parametric assumptions (normality and sphericity) met.")
                else:
                    test_key = "friedman_test"
                    reasoning.append("Tracking 3 or more repeated conditions within the same subjects.")
                    reasoning.append("Data is ordinal or non-normal; rank-based repeated measures test selected.")

    elif objective == "contingency_proportions":
        setup = answers.get("table_setup", "two_by_two")
        cell_counts = answers.get("cell_counts", "adequate")

        if setup == "paired_binary":
            test_key = "mcnemar_test"
            reasoning.append("Assessing changes in proportions for paired/matched binary outcomes.")
            reasoning.append("Tests symmetry of discordant pairs across the 2 measurement occasions.")
        elif cell_counts == "small":
            test_key = "fisher_exact"
            reasoning.append("Evaluating association in a contingency table where sample size is small.")
            reasoning.append("Expected cell frequency < 5 in one or more cells; exact probability calculated.")
        else:
            test_key = "chi_square_independence"
            reasoning.append("Evaluating relationship between categorical variables in independent groups.")
            reasoning.append("All expected cell frequencies are >= 5, satisfying large-sample chi-square requirements.")

    elif objective == "examine_relationship":
        rel_goal = answers.get("relationship_goal", "correlation")
        linearity = answers.get("linearity", "linear")

        if rel_goal == "correlation":
            if linearity == "linear":
                test_key = "pearson_correlation"
                reasoning.append("Quantifying linear association between two continuous variables.")
                reasoning.append("Bivariate normality and linearity assumptions satisfied.")
            else:
                test_key = "spearman_correlation"
                reasoning.append("Quantifying monotonic association between ranked or non-normal variables.")
                reasoning.append("Non-parametric rank correlation robust to non-linearity and outliers.")
        elif rel_goal == "predict_continuous":
            test_key = "linear_regression"
            reasoning.append("Predicting a continuous target variable from one or more predictor variables.")
            reasoning.append("Quantifies effect size (coefficients) and statistical significance.")
        elif rel_goal == "predict_binary":
            test_key = "binary_logistic_regression"
            reasoning.append("Predicting a binary outcome (0/1, Success/Failure) from predictors.")
            reasoning.append("Models log-odds of the outcome and produces adjusted Odds Ratios.")

    elif objective == "time_to_event":
        surv_goal = answers.get("survival_goal", "compare_curves")
        if surv_goal == "compare_curves":
            test_key = "log_rank_test"
            reasoning.append("Comparing survival distributions over time between groups with censored events.")
        else:
            test_key = "linear_regression"  # Fallback indicator
            reasoning.append("For multivariable time-to-event regression, Cox Proportional Hazards is recommended.")

    # Default fallback if criteria somehow didn't match
    if not test_key or test_key not in STATISTICAL_TESTS:
        test_key = "welch_t_test"
        reasoning.append("Defaulted to standard comparative test based on input parameters.")

    test_info = STATISTICAL_TESTS[test_key]
    return {
        "test_key": test_key,
        "name": test_info["name"],
        "category": test_info["category"],
        "parametric": test_info["parametric"],
        "description": test_info["description"],
        "reasoning": reasoning,
        "when_to_use": test_info["when_to_use"],
        "key_assumptions": test_info["key_assumptions"],
        "alternative_test": test_info["alternative_test"],
        "excel_guide": test_info["excel_guide"],
        "python_snippet": test_info["python_snippet"],
        "demo_generator": test_info["demo_generator"],
    }


def get_statistical_test_recommendation(test_key: str) -> Dict[str, Any]:
    """Retrieves the full recommendation schema dictionary for a directly selected test key.

    Args:
        test_key: Identifier key in STATISTICAL_TESTS (e.g. 'repeated_measures_anova').

    Returns:
        Dictionary containing method details, reasoning, and implementation metadata.
    """
    if test_key not in STATISTICAL_TESTS:
        test_key = "welch_t_test"

    test_info = STATISTICAL_TESTS[test_key]
    reasoning = [
        f"Directly selected method: {test_info['name']}.",
        f"Belongs to the '{test_info['category']}' family.",
        f"Analytical classification: {'Parametric (assumes underlying distribution parameters)' if test_info['parametric'] else 'Non-parametric / Distribution-free'}."
    ]

    return {
        "test_key": test_key,
        "name": test_info["name"],
        "category": test_info["category"],
        "parametric": test_info["parametric"],
        "description": test_info["description"],
        "reasoning": reasoning,
        "when_to_use": test_info["when_to_use"],
        "key_assumptions": test_info["key_assumptions"],
        "alternative_test": test_info["alternative_test"],
        "excel_guide": test_info["excel_guide"],
        "python_snippet": test_info["python_snippet"],
        "demo_generator": test_info["demo_generator"],
    }


def get_all_statistical_tests() -> List[Dict[str, Any]]:
    """Returns a list of all supported statistical tests with metadata for direct selection UI."""
    tests = []
    for key, info in STATISTICAL_TESTS.items():
        tests.append({
            "key": key,
            "name": info["name"],
            "category": info["category"],
            "parametric": info["parametric"],
            "description": info["description"]
        })
    return tests


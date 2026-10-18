"""Synthetic Data Generators & Live Statistical Calculators.

Generates reproducible demonstration datasets and executes the corresponding
statistical calculations so users can visualize and interpret real outputs.
"""

from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
import scipy.stats as stats


def generate_two_independent_demo(
    n_a: int = 40,
    n_b: int = 40,
    mean_a: float = 120.0,
    mean_b: float = 128.0,
    std_a: float = 12.0,
    std_b: float = 14.0,
    skewed: bool = False,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic data for two independent groups and computes stats."""
    rng = np.random.default_rng(seed)
    
    if skewed:
        # Log-normal distribution to simulate positive skewness
        raw_a = rng.lognormal(mean=0.0, sigma=0.8, size=n_a)
        raw_b = rng.lognormal(mean=0.3, sigma=0.8, size=n_b)
        vals_a = (raw_a - np.mean(raw_a)) / np.std(raw_a) * std_a + mean_a
        vals_b = (raw_b - np.mean(raw_b)) / np.std(raw_b) * std_b + mean_b
    else:
        vals_a = rng.normal(loc=mean_a, scale=std_a, size=n_a)
        vals_b = rng.normal(loc=mean_b, scale=std_b, size=n_b)

    df_a = pd.DataFrame({"Group": "Group A (Control)", "Value": vals_a})
    df_b = pd.DataFrame({"Group": "Group B (Treatment)", "Value": vals_b})
    df = pd.concat([df_a, df_b], ignore_index=True)

    # Perform statistical tests
    t_welch, p_welch = stats.ttest_ind(vals_a, vals_b, equal_var=False)
    t_stud, p_stud = stats.ttest_ind(vals_a, vals_b, equal_var=True)
    u_stat, p_mann = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    lev_stat, p_levene = stats.levene(vals_a, vals_b)
    shap_a, p_shap_a = stats.shapiro(vals_a)
    shap_b, p_shap_b = stats.shapiro(vals_b)

    results = {
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "median_a": float(np.median(vals_a)),
        "median_b": float(np.median(vals_b)),
        "std_a": float(np.std(vals_a, ddof=1)),
        "std_b": float(np.std(vals_b, ddof=1)),
        "t_welch": float(t_welch),
        "p_welch": float(p_welch),
        "t_student": float(t_stud),
        "p_student": float(p_stud),
        "u_stat": float(u_stat),
        "p_mann_whitney": float(p_mann),
        "levene_stat": float(lev_stat),
        "p_levene": float(p_levene),
        "shapiro_p_a": float(p_shap_a),
        "shapiro_p_b": float(p_shap_b),
    }

    return df, results


def generate_paired_demo(
    n_pairs: int = 35,
    mean_pre: float = 140.0,
    mean_diff: float = -8.5,
    std_diff: float = 6.0,
    dist_shape: str = "normal",
    outlier_count: int = 0,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic data for paired/repeated measurements (Pre vs Post) with full diagnostics.

    Args:
        n_pairs: Number of matched pairs / subjects.
        mean_pre: Baseline/pre-intervention target mean.
        mean_diff: Target mean difference (Post - Pre).
        std_diff: Target standard deviation of the difference scores.
        dist_shape: Distribution shape of differences ('normal', 'skewed_moderate', 'heavy_tailed').
        outlier_count: Number of extreme outlier differences to inject (0, 1, 2, 3).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (combined_DataFrame, results_dict).
    """
    rng = np.random.default_rng(seed)

    # 1. Generate difference distribution shapes
    if dist_shape == "skewed_moderate":
        # Right-skewed Gamma distribution for differences
        raw_diff = rng.gamma(shape=2.5, scale=1.0, size=n_pairs)
        diff = (raw_diff - np.mean(raw_diff)) / (np.std(raw_diff) if np.std(raw_diff) > 0 else 1.0) * std_diff + mean_diff
    elif dist_shape == "heavy_tailed":
        # Heavy-tailed Student's t distribution (df=3) for differences
        raw_diff = rng.standard_t(df=3, size=n_pairs)
        diff = (raw_diff - np.mean(raw_diff)) / (np.std(raw_diff) if np.std(raw_diff) > 0 else 1.0) * std_diff + mean_diff
    else:  # normal
        diff = rng.normal(loc=mean_diff, scale=std_diff, size=n_pairs)

    # 2. Inject extreme outlier differences if requested
    if outlier_count > 0:
        for i in range(min(outlier_count, n_pairs)):
            diff[-(i + 1)] = mean_diff + 4.5 * std_diff

    # 3. Baseline Pre values and compute Post = Pre + Diff
    pre = rng.normal(loc=mean_pre, scale=15.0, size=n_pairs)
    pre = np.round(pre, 1)
    diff = np.round(diff, 1)
    post = np.round(pre + diff, 1)
    # Ensure exact arithmetic equality: diff == post - pre
    diff = np.round(post - pre, 1)

    df = pd.DataFrame({
        "Subject_ID": [f"Subj_{i+1:02d}" for i in range(n_pairs)],
        "Pre_Intervention": pre,
        "Post_Intervention": post,
        "Difference": diff
    })

    # 4. Summary Statistics
    calc_mean_pre = float(np.mean(pre))
    calc_mean_post = float(np.mean(post))
    calc_std_pre = float(np.std(pre, ddof=1))
    calc_std_post = float(np.std(post, ddof=1))
    calc_mean_diff = float(np.mean(diff))
    calc_std_diff = float(np.std(diff, ddof=1))
    calc_med_diff = float(np.median(diff))

    df_paired = int(n_pairs - 1)
    se_diff = float(calc_std_diff / np.sqrt(n_pairs)) if n_pairs > 0 else 0.0
    t_stat_val = float(calc_mean_diff / se_diff) if se_diff > 0 else 0.0
    p_val_paired = float(2.0 * stats.t.sf(abs(t_stat_val), df=df_paired))

    # 95% Confidence Interval for mean difference
    t_crit = float(stats.t.ppf(0.975, df=df_paired))
    ci_lower = calc_mean_diff - t_crit * se_diff
    ci_upper = calc_mean_diff + t_crit * se_diff

    # Paired effect sizes (Cohen's dz and Hedges' gav)
    cohens_dz = float(calc_mean_diff / calc_std_diff) if calc_std_diff > 0 else 0.0
    hedges_gav = float(cohens_dz * (1.0 - 3.0 / (4.0 * df_paired - 1.0))) if df_paired > 0 else cohens_dz

    # Pearson correlation between Pre and Post
    r_corr, _ = stats.pearsonr(pre, post)
    r_corr = float(r_corr)

    # 5. Diagnostic Computations on Difference Scores
    shap_stat, p_shap = stats.shapiro(diff)
    shapiro_p_diff = float(p_shap)
    skew_diff = float(stats.skew(diff))
    kurt_diff = float(stats.kurtosis(diff))

    # Tukey's Outlier Diagnostics on Differences
    q25_d, q75_d = float(np.percentile(diff, 25)), float(np.percentile(diff, 75))
    iqr_d = q75_d - q25_d
    mild_outliers = diff[(diff < q25_d - 1.5 * iqr_d) | (diff > q75_d + 1.5 * iqr_d)]
    ext_outliers = diff[(diff < q25_d - 3.0 * iqr_d) | (diff > q75_d + 3.0 * iqr_d)]
    total_mild = len(mild_outliers)
    total_ext = len(ext_outliers)

    # Non-parametric Alternative (Wilcoxon Signed-Rank Test)
    try:
        wilc_res = stats.wilcoxon(diff, alternative="two-sided")
        wilc_stat = float(wilc_res.statistic)
        p_wilc = float(wilc_res.pvalue)
    except ValueError:
        # Handle all-zero differences
        wilc_stat = 0.0
        p_wilc = 1.0

    # Q-Q Plot Coordinates on Differences for Plotly
    prob_diff = stats.probplot(diff, dist="norm")
    qq_osm = prob_diff[0][0].tolist()
    qq_osr = prob_diff[0][1].tolist()
    qq_slope = float(prob_diff[1][0])
    qq_intercept = float(prob_diff[1][1])

    # 6. Evaluate Core Assumptions
    scale_status = "pass"
    scale_verdict = "Continuous Scale Satisfied"
    scale_detail = f"Pre and Post values are continuous numeric measurements with mean difference d̄ = {calc_mean_diff:+.2f}."

    design_status = "pass"
    design_verdict = "Paired Dependent Design"
    design_detail = f"Observations represent {n_pairs} matched pairs / repeated measures from identical subjects (Pearson r = {r_corr:.3f})."

    # Normality of Differences
    if shapiro_p_diff >= 0.05:
        norm_status = "pass"
        norm_verdict = "Normal Differences (Shapiro p >= 0.05)"
        norm_detail = (
            f"Difference scores show no significant departure from normality "
            f"(Shapiro p = {shapiro_p_diff:.3f}, Skewness = {skew_diff:.2f}, Kurtosis = {kurt_diff:.2f})."
        )
    elif n_pairs >= 30 and abs(skew_diff) <= 1.0:
        norm_status = "warning"
        norm_verdict = "CLT Robust (n >= 30, Moderate Skew)"
        norm_detail = (
            f"Shapiro test indicated non-normality (p = {shapiro_p_diff:.3f}), but paired sample size "
            f"(n = {n_pairs} >= 30) and modest skewness (|skew| = {abs(skew_diff):.2f} <= 1.0) provide Central Limit Theorem robustness."
        )
    else:
        norm_status = "fail"
        norm_verdict = "Differences Non-Normal"
        norm_detail = (
            f"Differences significantly deviate from normal (Shapiro p = {shapiro_p_diff:.3f}, Skewness = {skew_diff:.2f}) "
            f"with small sample size (n = {n_pairs} < 30). Switch to Wilcoxon Signed-Rank Test."
        )

    # Outliers in Differences
    if total_ext == 0 and total_mild == 0:
        outlier_status = "pass"
        outlier_verdict = "No Outliers Detected"
        outlier_detail = "No difference scores fall outside Tukey's 1.5×IQR or 3.0×IQR fences."
    elif total_ext == 0 and total_mild > 0:
        outlier_status = "warning"
        outlier_verdict = f"{total_mild} Mild Outlier(s)"
        outlier_detail = (
            f"{total_mild} difference score(s) fall beyond the 1.5×IQR fence. Inspect whether these represent genuine response variation."
        )
    else:
        outlier_status = "fail"
        outlier_verdict = f"{total_ext} Extreme Outlier(s)"
        outlier_detail = (
            f"{total_ext} extreme difference score(s) exceed 3.0×IQR. Extreme outliers heavily distort mean difference and standard error. "
            f"Switch to Wilcoxon Signed-Rank Test."
        )

    assumptions_eval = [
        {
            "name": "1. Measurement Scale (Continuous)",
            "description": "Pre and Post variables must be continuous numeric measurements.",
            "test_applied": "Variable inspection",
            "diagnostic_value": "Continuous numeric",
            "status": scale_status,
            "verdict": scale_verdict,
            "detail": scale_detail
        },
        {
            "name": "2. Paired / Dependent Study Design",
            "description": "Observations must be matched pairs or repeated measures from the same subjects.",
            "test_applied": "Study design check",
            "diagnostic_value": f"{n_pairs} Matched Pairs (r = {r_corr:.3f})",
            "status": design_status,
            "verdict": design_verdict,
            "detail": design_detail
        },
        {
            "name": "3. Normality of Difference Scores",
            "description": "The pairwise differences (Post - Pre) must follow an approximately normal distribution.",
            "test_applied": "Shapiro-Wilk Test (p-value) & Skewness",
            "diagnostic_value": f"Shapiro p={shapiro_p_diff:.3f}, Skew={skew_diff:.2f}",
            "status": norm_status,
            "verdict": norm_verdict,
            "detail": norm_detail
        },
        {
            "name": "4. Outliers in Differences",
            "description": "No extreme difference scores (> 3×IQR) that artificially inflate mean difference or SE.",
            "test_applied": "Tukey's IQR Outlier Detection on Differences",
            "diagnostic_value": f"{total_ext} extreme, {total_mild} mild outliers",
            "status": outlier_status,
            "verdict": outlier_verdict,
            "detail": outlier_detail
        }
    ]

    # 7. Overall Recommendation Synthesis
    if norm_status == "fail" or outlier_status == "fail":
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption(s) Violated: Switch to Wilcoxon Signed-Rank Test"
        overall_recommendation = (
            f"The normality of difference scores was violated in a small sample size (p = {shapiro_p_diff:.4f}) "
            f"or extreme difference outliers were detected ({total_ext} extreme). Under these conditions, the mean difference is distorted. "
            f"\n\n**Actionable Recommendation:** Switch to the non-parametric **Wilcoxon Signed-Rank Test (Paired)**. "
            f"The Wilcoxon test analyzes the ranks and signs of paired differences, making it robust against non-normal skewness and extreme outliers. "
            f"\n\n**Alternative Test Result on this Data:** "
            f"Wilcoxon W = {wilc_stat:.1f}, **p-value = {p_wilc:.4f}**."
        )
    elif norm_status == "warning" or outlier_status == "warning":
        overall_status = "warning"
        overall_verdict_badge = "🟡 Assumptions Borderline: Proceed with Caution (CLT Robust)"
        overall_recommendation = (
            f"Mild non-normality or mild difference outliers were detected, but sample size (n = {n_pairs} >= 30) "
            f"provides Central Limit Theorem robustness for the mean difference. "
            f"\n\n**Actionable Recommendation:** You may proceed with the **Paired Samples t-Test** "
            f"(t({df_paired}) = {t_stat_val:.3f}, p = {p_val_paired:.4f}), but cross-reference "
            f"with the non-parametric **Wilcoxon Signed-Rank Test** (p = {p_wilc:.4f}) to confirm consistency."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ All Assumptions Met: Proceed with Paired Samples t-Test"
        overall_recommendation = (
            f"All core assumptions (continuous scale, matched pair design, normal difference scores, "
            f"and absence of extreme outliers) are satisfied. "
            f"\n\n**Actionable Recommendation:** Proceed with confidence using the **Paired Samples t-Test**. "
            f"t({df_paired}) = {t_stat_val:.3f}, p = {p_val_paired:.4f}, Mean Diff = {calc_mean_diff:+.2f}, "
            f"95% CI = [{ci_lower:.2f}, {ci_upper:.2f}], Cohen's dz = {cohens_dz:.2f}."
        )

    results = {
        "n_pairs": int(n_pairs),
        "mean_pre": calc_mean_pre,
        "mean_post": calc_mean_post,
        "std_pre": calc_std_pre,
        "std_post": calc_std_post,
        "mean_difference": calc_mean_diff,
        "std_difference": calc_std_diff,
        "median_difference": calc_med_diff,
        "se_diff": se_diff,
        "df": df_paired,
        "t_stat": t_stat_val,
        "p_val": p_val_paired,
        "p_paired_t": p_val_paired,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_dz": cohens_dz,
        "hedges_gav": hedges_gav,
        "r_pearson": r_corr,
        "wilcoxon_stat": wilc_stat,
        "p_wilcoxon": p_wilc,
        "shapiro_p_diff": shapiro_p_diff,
        "skew_diff": skew_diff,
        "kurt_diff": kurt_diff,
        "total_ext": total_ext,
        "total_mild": total_mild,
        "qq_osm": qq_osm,
        "qq_osr": qq_osr,
        "qq_slope": qq_slope,
        "qq_intercept": qq_intercept,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df, results


def generate_one_way_anova_demo(
    n_per_group: int = 30,
    means: Tuple[float, float, float] = (50.0, 56.0, 62.0),
    stds: Optional[Tuple[float, float, float]] = None,
    std: float = 8.0,
    group_names: Tuple[str, str, str] = ("Standard Care", "Treatment Low Dose", "Treatment High Dose"),
    dist_shape: str = "normal",
    outlier_count_g3: int = 0,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic data across three groups for One-Way ANOVA with comprehensive diagnostics.

    Args:
        n_per_group: Number of observations per group (balanced design).
        means: Target population means (mu_1, mu_2, mu_3).
        stds: Group-specific standard deviations (sigma_1, sigma_2, sigma_3). If None, uses std for all.
        std: Fallback standard deviation for all groups if stds is not provided.
        group_names: Custom names for the 3 groups.
        dist_shape: 'normal', 'skewed_moderate' (Gamma), or 'heavy_tailed' (Student's t, df=3).
        outlier_count_g3: Number of extreme outliers (+4.5*sigma) injected into Group 3.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (combined_DataFrame, results_dict).
    """
    rng = np.random.default_rng(seed)
    actual_stds = stds if stds is not None else (std, std, std)

    groups_data = []
    for idx, (m_val, s_val) in enumerate(zip(means, actual_stds)):
        if dist_shape == "skewed_moderate":
            raw = rng.gamma(shape=2.5, scale=1.0, size=n_per_group)
            g_vals = (raw - np.mean(raw)) / (np.std(raw) if np.std(raw) > 0 else 1.0) * s_val + m_val
        elif dist_shape == "heavy_tailed":
            raw = rng.standard_t(df=3, size=n_per_group)
            g_vals = (raw - np.mean(raw)) / (np.std(raw) if np.std(raw) > 0 else 1.0) * s_val + m_val
        else:  # normal
            g_vals = rng.normal(loc=m_val, scale=s_val, size=n_per_group)

        # Inject extreme outliers into group 3 if requested
        if idx == 2 and outlier_count_g3 > 0:
            for j in range(min(outlier_count_g3, n_per_group)):
                g_vals[-(j + 1)] = m_val + 4.5 * s_val

        g_vals = np.round(g_vals, 1)
        groups_data.append(g_vals)

    g1, g2, g3 = groups_data

    # Long-format DataFrame
    df = pd.DataFrame({
        "Subject_ID": [f"ID_{i+1:03d}" for i in range(3 * n_per_group)],
        "Group": ([group_names[0]] * n_per_group +
                  [group_names[1]] * n_per_group +
                  [group_names[2]] * n_per_group),
        "Score": np.concatenate([g1, g2, g3]).astype(float)
    })

    # Summary Statistics per Group
    k = 3
    n_total = 3 * n_per_group
    m1, m2, m3 = float(np.mean(g1)), float(np.mean(g2)), float(np.mean(g3))
    s1, s2, s3 = float(np.std(g1, ddof=1)), float(np.std(g2, ddof=1)), float(np.std(g3, ddof=1))
    v1, v2, v3 = s1 ** 2, s2 ** 2, s3 ** 2
    se1 = s1 / np.sqrt(n_per_group)
    se2 = s2 / np.sqrt(n_per_group)
    se3 = s3 / np.sqrt(n_per_group)

    # 95% Confidence Intervals for Means
    t_crit = float(stats.t.ppf(0.975, df=n_per_group - 1)) if n_per_group > 1 else 1.96
    ci1 = (m1 - t_crit * se1, m1 + t_crit * se1)
    ci2 = (m2 - t_crit * se2, m2 + t_crit * se2)
    ci3 = (m3 - t_crit * se3, m3 + t_crit * se3)

    # Grand Mean and ANOVA Sum of Squares Decomposition
    grand_mean = float((n_per_group * m1 + n_per_group * m2 + n_per_group * m3) / n_total)
    ss_between = float(n_per_group * ((m1 - grand_mean) ** 2 + (m2 - grand_mean) ** 2 + (m3 - grand_mean) ** 2))
    df_between = int(k - 1)

    ss_within = float((n_per_group - 1) * (v1 + v2 + v3))
    df_within = int(n_total - k)

    ss_total = ss_between + ss_within
    df_total = int(n_total - 1)

    ms_between = float(ss_between / df_between) if df_between > 0 else 0.0
    ms_within = float(ss_within / df_within) if df_within > 0 else 0.0

    f_stat = float(ms_between / ms_within) if ms_within > 0 else 0.0
    p_anova = float(stats.f.sf(f_stat, df_between, df_within)) if df_within > 0 else 1.0
    f_crit = float(stats.f.ppf(0.95, df_between, df_within)) if df_within > 0 else 3.0

    # Effect Sizes: Eta-squared and Omega-squared
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0
    omega_squared = float(max(0.0, (ss_between - df_between * ms_within) / (ss_total + ms_within))) if (ss_total + ms_within) > 0 else 0.0

    # Assumption 1: Normality per Group (Shapiro-Wilk, Skewness, Kurtosis)
    shap1_w, shap1_p = stats.shapiro(g1)
    shap2_w, shap2_p = stats.shapiro(g2)
    shap3_w, shap3_p = stats.shapiro(g3)
    skew1, kurt1 = float(stats.skew(g1)), float(stats.kurtosis(g1))
    skew2, kurt2 = float(stats.skew(g2)), float(stats.kurtosis(g2))
    skew3, kurt3 = float(stats.skew(g3)), float(stats.kurtosis(g3))

    all_normal = (shap1_p >= 0.05 and shap2_p >= 0.05 and shap3_p >= 0.05)
    clt_robust = (n_per_group >= 30 and max(abs(skew1), abs(skew2), abs(skew3)) <= 1.2)

    if all_normal:
        norm_status = "pass"
        norm_verdict = "PASS (ALL GROUPS NORMAL)"
        norm_detail = (
            f"Shapiro-Wilk test indicates normal distributions across all 3 groups "
            f"(p₁={shap1_p:.3f}, p₂={shap2_p:.3f}, p₃={shap3_p:.3f} >= 0.05)."
        )
    elif clt_robust:
        norm_status = "warning"
        norm_verdict = "CAUTION (CLT ROBUST)"
        norm_detail = (
            f"Mild departure from normality in one or more groups (min Shapiro p = {min(shap1_p, shap2_p, shap3_p):.4f}); "
            f"however, sample size n = {n_per_group} >= 30 per group activates the Central Limit Theorem protection."
        )
    else:
        norm_status = "fail"
        norm_verdict = "VIOLATED (NON-NORMAL)"
        norm_detail = (
            f"Significant non-normality detected (min Shapiro p = {min(shap1_p, shap2_p, shap3_p):.4f} < 0.05) "
            f"without sufficient sample size protection (n = {n_per_group} < 30). Non-parametric Kruskal-Wallis recommended."
        )

    # Assumption 2: Homogeneity of Variance (Levene's Test and Variance Ratio)
    lev_stat, p_levene = stats.levene(g1, g2, g3, center="median")
    lev_stat = float(lev_stat)
    p_levene = float(p_levene)
    var_min = min(v1, v2, v3)
    var_max = max(v1, v2, v3)
    var_ratio = float(var_max / var_min) if var_min > 0 else 1.0

    if p_levene >= 0.05 and var_ratio <= 2.5:
        homo_status = "pass"
        homo_verdict = "PASS (HOMOSCEDASTIC)"
        homo_detail = (
            f"Levene's test p = {p_levene:.3f} (>= 0.05) and variance ratio F_max = {var_ratio:.2f} (<= 2.5). "
            f"Group variances are equal; standard ANOVA F-test is valid."
        )
    elif var_ratio <= 4.0:
        homo_status = "warning"
        homo_verdict = "CAUTION (MILD VARIANCE RATIO)"
        homo_detail = (
            f"Levene's test p = {p_levene:.3f} (< 0.05), but variance ratio F_max = {var_ratio:.2f} <= 4.0. "
            f"With equal group sizes (n = {n_per_group}), standard ANOVA F-test is moderately robust."
        )
    else:
        homo_status = "fail"
        homo_verdict = "VIOLATED (HETEROSCEDASTIC)"
        homo_detail = (
            f"Severe variance inequality detected: F_max = {var_ratio:.2f} (> 4.0), Levene p = {p_levene:.4e}. "
            f"Type I error rates for standard ANOVA are distorted. Switch to Welch's ANOVA."
        )

    # Assumption 3: Absence of Extreme Outliers (Tukey IQR per group)
    outlier_counts = []
    for g_arr in [g1, g2, g3]:
        q25, q75 = np.percentile(g_arr, [25, 75])
        iqr = q75 - q25
        ext_cnt = int(np.sum((g_arr < q25 - 3.0 * iqr) | (g_arr > q75 + 3.0 * iqr)))
        outlier_counts.append(ext_cnt)
    total_ext_outliers = sum(outlier_counts)

    if total_ext_outliers == 0:
        outlier_status = "pass"
        outlier_verdict = "PASS"
        outlier_detail = "No extreme outliers (> 3.0×IQR) detected across any of the 3 groups."
    else:
        outlier_status = "fail"
        outlier_verdict = "VIOLATED (EXTREME OUTLIERS)"
        outlier_detail = (
            f"{total_ext_outliers} extreme outlier(s) detected across groups (Group 3: {outlier_counts[2]}). "
            f"Outliers inflate Mean Square Within and distort group means."
        )

    # Core Assumption Objects
    asm_scale = {
        "name": "1. Continuous Scale & Independent Groups",
        "description": "Dependent variable is continuous and sampled from 3 independent, mutually exclusive populations.",
        "test_applied": "Design Structure Inspection",
        "diagnostic_value": f"Continuous Metric ({k} Independent Groups)",
        "status": "pass",
        "verdict": "PASS",
        "detail": "Data satisfies continuous interval/ratio scale and independent random sampling requirements."
    }

    asm_norm = {
        "name": "2. Normality of Residuals within Groups",
        "description": "Data within each group should be approximately normal (robust for n >= 30).",
        "test_applied": "Shapiro-Wilk Test per Group & Skewness Check",
        "diagnostic_value": f"Min Shapiro p = {min(shap1_p, shap2_p, shap3_p):.4f}, Max |Skew| = {max(abs(skew1), abs(skew2), abs(skew3)):.2f}",
        "status": norm_status,
        "verdict": norm_verdict,
        "detail": norm_detail
    }

    asm_homo = {
        "name": "3. Homogeneity of Variance (Homoscedasticity)",
        "description": "Variances across all groups should be approximately equal (variance ratio <= 4.0).",
        "test_applied": "Levene's Test & Variance Ratio (F_max)",
        "diagnostic_value": f"Levene p = {p_levene:.4f}, Variance Ratio = {var_ratio:.2f}",
        "status": homo_status,
        "verdict": homo_verdict,
        "detail": homo_detail
    }

    asm_outliers = {
        "name": "4. Absence of Extreme Outliers",
        "description": "No observations lying beyond 3.0×IQR that disproportionately pull group means.",
        "test_applied": "Tukey's IQR Outlier Rule per Group",
        "diagnostic_value": f"Total Extreme Outliers = {total_ext_outliers}",
        "status": outlier_status,
        "verdict": outlier_verdict,
        "detail": outlier_detail
    }

    assumptions_eval = [asm_scale, asm_norm, asm_homo, asm_outliers]

    # Non-Parametric Alternative: Kruskal-Wallis H Test
    krusk_stat, p_kruskal = stats.kruskal(g1, g2, g3)
    krusk_stat = float(krusk_stat)
    p_kruskal = float(p_kruskal)

    # Post-Hoc Pairwise Comparisons (Tukey's HSD)
    try:
        tukey_res = stats.tukey_hsd(g1, g2, g3)
        p_12 = float(tukey_res.pvalue[0, 1])
        p_13 = float(tukey_res.pvalue[0, 2])
        p_23 = float(tukey_res.pvalue[1, 2])
        ci_12 = (float(tukey_res.confidence_interval.low[0, 1]), float(tukey_res.confidence_interval.high[0, 1]))
        ci_13 = (float(tukey_res.confidence_interval.low[0, 2]), float(tukey_res.confidence_interval.high[0, 2]))
        ci_23 = (float(tukey_res.confidence_interval.low[1, 2]), float(tukey_res.confidence_interval.high[1, 2]))
    except Exception:
        # Fallback pairwise t-tests with Bonferroni correction
        _, p_12_raw = stats.ttest_ind(g1, g2, equal_var=True)
        _, p_13_raw = stats.ttest_ind(g1, g3, equal_var=True)
        _, p_23_raw = stats.ttest_ind(g2, g3, equal_var=True)
        p_12 = min(1.0, float(p_12_raw * 3))
        p_13 = min(1.0, float(p_13_raw * 3))
        p_23 = min(1.0, float(p_23_raw * 3))
        diff_12 = m2 - m1
        diff_13 = m3 - m1
        diff_23 = m3 - m2
        se_pair = np.sqrt(2.0 * ms_within / n_per_group) if n_per_group > 0 else 1.0
        ci_12 = (diff_12 - t_crit * se_pair, diff_12 + t_crit * se_pair)
        ci_13 = (diff_13 - t_crit * se_pair, diff_13 + t_crit * se_pair)
        ci_23 = (diff_23 - t_crit * se_pair, diff_23 + t_crit * se_pair)

    raw_pairs = [
        (1, 0, m2 - m1, ci_12, p_12),
        (2, 0, m3 - m1, ci_13, p_13),
        (2, 1, m3 - m2, ci_23, p_23)
    ]
    pairwise_comparisons = []
    for g_b_idx, g_a_idx, diff, ci, p_adj in raw_pairs:
        is_sig = bool(p_adj < 0.05)
        name_a = group_names[g_a_idx]
        name_b = group_names[g_b_idx]
        p_str = f"{p_adj:.4f}" if p_adj >= 0.0001 else "< 0.0001"
        if is_sig:
            direction = "higher" if diff > 0 else "lower"
            plain = (
                f"{name_b} scored significantly {direction} than {name_a} "
                f"(mean difference: {diff:+.2f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}], adjusted p = {p_str})."
            )
        else:
            plain = (
                f"No statistically significant difference between {name_b} and {name_a} "
                f"(mean difference: {diff:+.2f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}], adjusted p = {p_str}). "
                f"Any observed gap is consistent with random sampling variability."
            )
        pairwise_comparisons.append({
            "comparison": f"{name_b} vs {name_a}",
            "group_a": name_a,
            "group_b": name_b,
            "diff": float(diff),
            "diff_str": f"{diff:+.2f}",
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "ci_str": f"[{ci[0]:.2f}, {ci[1]:.2f}]",
            "p_adj": float(p_adj),
            "p_adj_str": p_str,
            "significant": is_sig,
            "significance_verdict": "✅ Significant Difference" if is_sig else "➖ No Difference",
            "plain_english": plain
        })

    # Overall Recommendation Synthesis
    has_failure = any(a["status"] == "fail" for a in assumptions_eval)
    has_warning = any(a["status"] == "warning" for a in assumptions_eval)

    if homo_status == "fail":
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption Violated: Unequal Variances (Switch to Welch's ANOVA)"
        overall_recommendation = (
            f"The homogeneity of variance assumption is violated (Variance ratio F_max = {var_ratio:.2f} > 4.0, "
            f"Levene p = {p_levene:.4e}). Standard One-Way ANOVA suffers from inflated Type I error. "
            f"\n\n**Actionable Recommendation:** Switch to **Welch's ANOVA**, which relaxes the equal variance assumption "
            f"using Welch-Satterthwaite adjusted degrees of freedom."
        )
    elif norm_status == "fail" or outlier_status == "fail":
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption Violated: Non-Normality / Outliers (Switch to Kruskal-Wallis)"
        overall_recommendation = (
            f"Severe non-normality or extreme outliers violate parametric ANOVA assumptions. "
            f"\n\n**Actionable Recommendation:** Switch to the non-parametric **Kruskal-Wallis H Test** "
            f"(H = {krusk_stat:.2f}, **p = {p_kruskal:.4e}**), which evaluates median ranks across groups without normality assumptions."
        )
    elif has_warning:
        overall_status = "warning"
        overall_verdict_badge = "🟡 Assumptions Borderline: Proceed with Caution"
        overall_recommendation = (
            f"Minor departures from assumptions detected, but balanced sample sizes (n = {n_per_group} >= 30) "
            f"provide robust protection for the omnibus F-test. "
            f"\n\n**Actionable Recommendation:** You may proceed with **One-Way ANOVA** (F = {f_stat:.2f}, p = {p_anova:.4e}), "
            f"but compare results with the **Kruskal-Wallis Test** (p = {p_kruskal:.4e}) as a sensitivity check."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ All Assumptions Met: Proceed with One-Way ANOVA"
        overall_recommendation = (
            f"All core assumptions (continuous scale, independence, normality, homoscedasticity, and absence of extreme outliers) "
            f"are successfully satisfied. "
            f"\n\n**Actionable Recommendation:** Proceed with confidence using **One-Way ANOVA**. "
            f"Omnibus test: F({df_between}, {df_within}) = {f_stat:.2f}, **p = {p_anova:.4e}**, "
            f"Eta-squared η² = {eta_squared:.3f} ({'Large effect' if eta_squared >= 0.14 else ('Medium effect' if eta_squared >= 0.06 else 'Small effect')})."
        )

    results = {
        "n_per_group": int(n_per_group),
        "n_total": int(n_total),
        "k_groups": int(k),
        "group_names": list(group_names),
        "means": [m1, m2, m3],
        "stds": [s1, s2, s3],
        "variances": [v1, v2, v3],
        "grand_mean": grand_mean,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "df_between": df_between,
        "df_within": df_within,
        "df_total": df_total,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "f_stat": f_stat,
        "p_anova": p_anova,
        "f_crit": f_crit,
        "eta_squared": eta_squared,
        "omega_squared": omega_squared,
        "ci_groups": [ci1, ci2, ci3],
        "p_levene": p_levene,
        "var_ratio": var_ratio,
        "shapiro_p": [shap1_p, shap2_p, shap3_p],
        "skewness": [skew1, skew2, skew3],
        "kurtosis": [kurt1, kurt2, kurt3],
        "outlier_counts": outlier_counts,
        "kruskal_stat": krusk_stat,
        "p_kruskal": p_kruskal,
        "pairwise_comparisons": pairwise_comparisons,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df, results


# Alias for backwards compatibility
generate_three_groups_demo = generate_one_way_anova_demo


def generate_kruskal_wallis_demo(
    n_per_group: int = 25,
    group_names: Tuple[str, str, str] = ("Standard Care", "Treatment Low Dose", "Treatment High Dose"),
    shifts: Tuple[float, float, float] = (0.0, 4.0, 9.0),
    data_type: str = "continuous",
    dist_shape: str = "skewed_moderate",
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic multi-group data and calculates Kruskal-Wallis H Test with comprehensive diagnostics.

    Args:
        n_per_group: Sample size per group (balanced design).
        group_names: Names for the 3 comparison groups.
        shifts: Group location shifts / median differences relative to baseline.
        data_type: 'continuous' (e.g. recovery scores) or 'ordinal_likert' (1-7 scale).
        dist_shape: 'skewed_moderate' (Gamma), 'normal', 'heavy_tailed' (Student's t, df=3), or 'unequal_shapes'.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (combined_DataFrame, results_dict).
    """
    rng = np.random.default_rng(seed)
    k = len(group_names)

    # 1. Generate data for each group based on dist_shape and data_type
    raw_groups = []
    for i in range(k):
        shift = shifts[i]
        if dist_shape == "skewed_moderate":
            # Gamma shape=2.0, scale=3.0 (moderately right-skewed, skew ~ 1.4)
            vals = 20.0 + shift + rng.gamma(shape=2.0, scale=3.0, size=n_per_group)
        elif dist_shape == "normal":
            # Standard bell curve
            vals = rng.normal(loc=30.0 + shift, scale=6.0, size=n_per_group)
        elif dist_shape == "heavy_tailed":
            # Student's t with df=3
            vals = 25.0 + shift + rng.standard_t(df=3, size=n_per_group) * 4.5
        elif dist_shape == "unequal_shapes":
            # Group 0: Right-skewed, Group 1: Normal, Group 2: Left-skewed
            if i == 0:
                vals = 20.0 + shift + rng.gamma(shape=2.0, scale=3.0, size=n_per_group)
            elif i == 1:
                vals = 28.0 + shift + rng.normal(loc=0.0, scale=6.0, size=n_per_group)
            else:
                vals = 42.0 + shift - rng.gamma(shape=2.0, scale=3.0, size=n_per_group)
        else:
            vals = rng.normal(loc=30.0 + shift, scale=6.0, size=n_per_group)
        raw_groups.append(vals)

    # 2. Scale / discretize if ordinal
    if data_type == "ordinal_likert":
        pooled_min = min(float(np.min(g)) for g in raw_groups)
        pooled_max = max(float(np.max(g)) for g in raw_groups)
        span = pooled_max - pooled_min if pooled_max > pooled_min else 1.0
        final_groups = []
        for g in raw_groups:
            scaled = 1.0 + 6.0 * (g - pooled_min) / span
            likert = np.clip(np.round(scaled), 1.0, 7.0).astype(float)
            final_groups.append(likert)
    else:
        final_groups = [np.round(g, 1).astype(float) for g in raw_groups]

    g1, g2, g3 = final_groups[0], final_groups[1], final_groups[2]
    all_scores = np.concatenate(final_groups)
    n_total = len(all_scores)
    df_kruskal = k - 1

    # 3. Global Ranking (Mid-Ranks for Ties via stats.rankdata)
    all_ranks = stats.rankdata(all_scores, method="average")
    ranks_g1 = all_ranks[:len(g1)]
    ranks_g2 = all_ranks[len(g1):len(g1)+len(g2)]
    ranks_g3 = all_ranks[len(g1)+len(g2):]

    # 4. Group Descriptive & Rank Metrics
    medians = [float(np.median(g)) for g in final_groups]
    q25s = [float(np.percentile(g, 25)) for g in final_groups]
    q75s = [float(np.percentile(g, 75)) for g in final_groups]
    iqrs = [q75s[i] - q25s[i] for i in range(k)]
    means = [float(np.mean(g)) for g in final_groups]
    stds = [float(np.std(g, ddof=1)) for g in final_groups]
    skews = [float(stats.skew(g)) for g in final_groups]
    kurts = [float(stats.kurtosis(g)) for g in final_groups]

    rank_sums = [float(np.sum(ranks_g1)), float(np.sum(ranks_g2)), float(np.sum(ranks_g3))]
    mean_ranks = [float(np.mean(ranks_g1)), float(np.mean(ranks_g2)), float(np.mean(ranks_g3))]
    exp_mean_rank = float((n_total + 1.0) / 2.0)

    # 5. Kruskal-Wallis Omnibus Test
    h_stat, p_kruskal = stats.kruskal(g1, g2, g3)
    h_stat = float(h_stat)
    p_kruskal = float(p_kruskal)

    # Tie Correction Factor
    _, tie_counts = np.unique(all_scores, return_counts=True)
    tie_sum = float(np.sum(tie_counts**3 - tie_counts))
    tie_factor = float(1.0 - tie_sum / (n_total**3 - n_total)) if n_total > 1 else 1.0

    # Effect Size: Epsilon-Squared (epsilon_R^2 = (H - k + 1) / (N - k))
    epsilon_sq = max(0.0, min(1.0, float((h_stat - k + 1.0) / (n_total - k)))) if n_total > k else 0.0
    if epsilon_sq < 0.06:
        effect_magnitude = "Small effect"
    elif epsilon_sq < 0.14:
        effect_magnitude = "Medium effect"
    else:
        effect_magnitude = "Large effect"

    # 6. Post-Hoc Pairwise Comparisons: Dunn's Test with Bonferroni Adjustment
    sigma_0_sq = (n_total * (n_total + 1.0) / 12.0) - (tie_sum / (12.0 * (n_total - 1.0))) if n_total > 1 else 1.0
    sigma_0 = np.sqrt(max(0.0001, sigma_0_sq))

    pairwise_comparisons = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    num_comparisons = len(pairs)

    for i, j in pairs:
        ni, nj = len(final_groups[i]), len(final_groups[j])
        se_pair = sigma_0 * np.sqrt(1.0 / ni + 1.0 / nj)
        diff_rank = mean_ranks[j] - mean_ranks[i]
        z_stat = abs(diff_rank) / se_pair if se_pair > 0 else 0.0
        p_raw = 2.0 * (1.0 - stats.norm.cdf(z_stat))
        p_adj = min(1.0, float(p_raw * num_comparisons))

        # Also compute pairwise Mann-Whitney U for verification
        _, p_mw = stats.mannwhitneyu(final_groups[i], final_groups[j], alternative="two-sided")
        p_mw_adj = min(1.0, float(p_mw * num_comparisons))

        # 95% Bonferroni-adjusted confidence interval for difference in mean ranks
        z_crit_bonf = float(stats.norm.ppf(1.0 - 0.05 / (2.0 * num_comparisons)))
        ci_lower_rank = diff_rank - z_crit_bonf * se_pair
        ci_upper_rank = diff_rank + z_crit_bonf * se_pair

        is_sig = bool(p_adj < 0.05)
        diff_med = float(medians[j] - medians[i])
        p_str = f"{p_adj:.4f}" if p_adj >= 0.0001 else "< 0.0001"
        if is_sig:
            direction = "higher" if diff_rank > 0 else "lower"
            plain = (
                f"{group_names[j]} ranks significantly {direction} than {group_names[i]} "
                f"(rank diff: {diff_rank:+.1f}, median diff: {diff_med:+.1f}, 95% CI [{ci_lower_rank:.1f}, {ci_upper_rank:.1f}], adjusted p = {p_str})."
            )
        else:
            plain = (
                f"No statistically significant difference in rank distribution between {group_names[j]} and {group_names[i]} "
                f"(rank diff: {diff_rank:+.1f}, median diff: {diff_med:+.1f}, adjusted p = {p_str}). "
                f"Any observed rank gap is consistent with random chance."
            )

        pairwise_comparisons.append({
            "comparison": f"{group_names[j]} vs {group_names[i]}",
            "group_a": group_names[i],
            "group_b": group_names[j],
            "diff_mean_rank": float(diff_rank),
            "diff_rank_str": f"{diff_rank:+.1f}",
            "diff_median": float(diff_med),
            "diff_median_str": f"{diff_med:+.1f}",
            "se_diff": float(se_pair),
            "z_stat": float(z_stat),
            "p_raw": float(p_raw),
            "p_adj": float(p_adj),
            "p_adj_str": p_str,
            "p_mw_adj": float(p_mw_adj),
            "ci_lower_rank": float(ci_lower_rank),
            "ci_upper_rank": float(ci_upper_rank),
            "ci_rank_str": f"[{ci_lower_rank:.1f}, {ci_upper_rank:.1f}]",
            "significant": is_sig,
            "significance_verdict": "✅ Significant Difference" if is_sig else "➖ No Difference",
            "plain_english": plain
        })

    # 7. Parametric One-Way ANOVA Comparison
    f_stat, p_anova = stats.f_oneway(g1, g2, g3)
    f_stat = float(f_stat)
    p_anova = float(p_anova)

    # 8. Shape & Homoscedasticity Diagnostics (Levene's Test & Normality)
    try:
        lev_stat, p_levene = stats.levene(g1, g2, g3, center="median")
        lev_stat = float(lev_stat)
        p_levene = float(p_levene)
    except Exception:
        lev_stat = 0.0
        p_levene = 1.0

    shapiro_results = [stats.shapiro(g) for g in final_groups]
    min_shapiro_p = min(float(r.pvalue) for r in shapiro_results)
    max_skew_diff = max(skews) - min(skews)

    # 9. Core Assumption Evaluations
    asm_scale = {
        "name": "1. Measurement Scale & Independence",
        "description": "Dependent variable is at least ordinal, sampled from 3 independent, mutually exclusive groups.",
        "test_applied": "Design Structure & Scale Verification",
        "diagnostic_value": f"{'Ordinal Likert (1-7)' if data_type == 'ordinal_likert' else 'Continuous Scores'} (3 Independent Groups)",
        "status": "pass",
        "verdict": "PASS",
        "detail": "Data satisfies the ordinal/continuous measurement requirement with mutually exclusive random sampling."
    }

    min_n = min(len(g1), len(g2), len(g3))
    if min_n >= 15:
        size_status = "pass"
        size_verdict = "PASS"
        size_detail = f"All group sample sizes (n_k = {min_n}) satisfy the asymptotic Chi-Square threshold (n_k >= 5, ideally >= 15)."
    elif min_n >= 5:
        size_status = "warning"
        size_verdict = "BORDERLINE (5 <= n < 15)"
        size_detail = f"Sample sizes (n_k = {min_n}) meet the minimum asymptotic rule (n_k >= 5), but exact permutation tests are recommended if p is near 0.05."
    else:
        size_status = "fail"
        size_verdict = "VIOLATED (n < 5)"
        size_detail = f"Sample size (min n = {min_n} < 5) is too small for reliable Chi-Square approximation. Exact permutation tables required."

    asm_size = {
        "name": "2. Group Sample Sizes (n_k >= 5)",
        "description": "Each group must have at least 5 observations for the Chi-Square distribution to accurately model H.",
        "test_applied": "Asymptotic Sample Size Rule",
        "diagnostic_value": f"Minimum n_k = {min_n} (Total N = {n_total})",
        "status": size_status,
        "verdict": size_verdict,
        "detail": size_detail
    }

    # Shape similarity assumption
    if p_levene >= 0.05 and max_skew_diff <= 1.0:
        shape_status = "pass"
        shape_verdict = "PASS (Similar Shapes — Test of Medians)"
        shape_detail = (
            f"Group distribution shapes and variances are similar (Levene p = {p_levene:.4f}, max skew diff = {max_skew_diff:.2f}). "
            f"Kruskal-Wallis can be interpreted directly as a test of differences between population medians."
        )
    else:
        shape_status = "warning"
        shape_verdict = "DIFFERING SHAPES (Test of Stochastic Dominance)"
        shape_detail = (
            f"Distribution shapes or variances differ across groups (Levene p = {p_levene:.4f}, max skew diff = {max_skew_diff:.2f}). "
            f"Kruskal-Wallis remains mathematically valid, but tests stochastic dominance / mean rank differences rather than a pure difference in medians."
        )

    asm_shape = {
        "name": "3. Distribution Shape Similarity Across Groups",
        "description": "Similar group shapes allow interpretation as median differences; differing shapes test general stochastic dominance.",
        "test_applied": "Levene's Test & Skewness Range Comparison",
        "diagnostic_value": f"Levene p = {p_levene:.4f}, Max Skew Diff = {max_skew_diff:.2f}",
        "status": shape_status,
        "verdict": shape_verdict,
        "detail": shape_detail
    }

    # Non-parametric justification
    is_non_normal = (data_type == "ordinal_likert") or (min_shapiro_p < 0.05) or any(abs(s) > 0.8 for s in skews)
    if is_non_normal:
        nonpar_status = "pass"
        nonpar_verdict = "PASS (Non-Parametric Justified)"
        nonpar_detail = (
            f"Data is {'ordinal' if data_type == 'ordinal_likert' else 'non-normally distributed'} "
            f"(Min Shapiro p = {min_shapiro_p:.4f}, max |skew| = {max(abs(s) for s in skews):.2f}). "
            f"Kruskal-Wallis is the appropriate, robust methodological choice over standard ANOVA."
        )
    else:
        nonpar_status = "warning"
        nonpar_verdict = "INFO (Parametric ANOVA is Valid & More Powerful)"
        nonpar_detail = (
            f"Data satisfies normality (Min Shapiro p = {min_shapiro_p:.4f}) and homogeneity. "
            f"While Kruskal-Wallis is valid, One-Way ANOVA (F = {f_stat:.2f}, p = {p_anova:.4e}) "
            f"provides greater statistical power for normal continuous data."
        )

    asm_nonpar = {
        "name": "4. Non-Parametric Appropriateness vs ANOVA",
        "description": "Evaluates whether non-parametric ranking is justified over parametric One-Way ANOVA.",
        "test_applied": "Normality (Shapiro-Wilk) & Measurement Scale Audit",
        "diagnostic_value": f"{'Ordinal Likert' if data_type == 'ordinal_likert' else f'Min Shapiro p = {min_shapiro_p:.4f}'}",
        "status": nonpar_status,
        "verdict": nonpar_verdict,
        "detail": nonpar_detail
    }

    assumptions_eval = [asm_scale, asm_size, asm_shape, asm_nonpar]

    # 10. Overall Recommendation Synthesis
    if p_kruskal < 0.05:
        sig_count = sum(1 for p in pairwise_comparisons if p["significant"])
        overall_status = "pass" if shape_status == "pass" else "warning"
        overall_verdict_badge = "✅ Statistically Significant Difference Across Groups (p < 0.05)"
        overall_recommendation = (
            f"Reject the null hypothesis H₀. At least one group differs significantly in rank distribution "
            f"(Kruskal-Wallis H({df_kruskal}) = {h_stat:.3f}, **p = {p_kruskal:.4e}**, Epsilon-squared ε² = {epsilon_sq:.3f} [{effect_magnitude}]). "
            f"\n\n**Actionable Post-Hoc Guidance:** Inspect the pairwise Dunn's tests with Bonferroni correction below. "
            f"{sig_count} of {num_comparisons} pairwise comparison(s) are statistically significant at alpha = 0.05."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ No Statistically Significant Difference Across Groups (p ≥ 0.05)"
        overall_recommendation = (
            f"Fail to reject the null hypothesis H₀. Observed rank differences across groups are consistent "
            f"with random sampling variability (Kruskal-Wallis H({df_kruskal}) = {h_stat:.3f}, **p = {p_kruskal:.4f}**, ε² = {epsilon_sq:.3f}). "
            f"\n\n**Actionable Post-Hoc Guidance:** Pairwise post-hoc comparisons are not indicated because the omnibus H-test did not reach statistical significance."
        )

    # 11. Assemble DataFrame
    subject_ids = [f"Subj_{i+1:03d}" for i in range(n_total)]
    group_col = []
    for i, name in enumerate(group_names):
        group_col.extend([name] * len(final_groups[i]))

    combined_df = pd.DataFrame({
        "Subject_ID": subject_ids,
        "Group": group_col,
        "Score": all_scores,
        "Global_Rank": all_ranks
    })

    results = {
        "n_per_group": int(n_per_group),
        "n_total": int(n_total),
        "k_groups": int(k),
        "group_names": list(group_names),
        "data_type": data_type,
        "dist_shape": dist_shape,
        "medians": medians,
        "q25s": q25s,
        "q75s": q75s,
        "iqrs": iqrs,
        "means": means,
        "stds": stds,
        "skews": skews,
        "kurts": kurts,
        "rank_sums": rank_sums,
        "mean_ranks": mean_ranks,
        "expected_mean_rank": exp_mean_rank,
        "h_stat": h_stat,
        "p_kruskal": p_kruskal,
        "df_kruskal": df_kruskal,
        "tie_factor": tie_factor,
        "tie_count_total": int(np.sum(tie_counts[tie_counts > 1])),
        "epsilon_squared": epsilon_sq,
        "effect_magnitude": effect_magnitude,
        "pairwise_comparisons": pairwise_comparisons,
        "f_stat": f_stat,
        "p_anova": p_anova,
        "levene_stat": lev_stat,
        "p_levene": p_levene,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return combined_df, results


def generate_repeated_measures_anova_demo(
    n_subjects: int = 25,
    time_names: Tuple[str, str, str] = ("Baseline", "Month 1", "Month 3"),
    means: Tuple[float, float, float] = (55.0, 62.0, 71.0),
    subject_sd: float = 10.0,
    within_sd: float = 4.0,
    sphericity_violation: bool = False,
    dist_shape: str = "normal",
    outlier_count: int = 0,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic repeated measures (within-subjects) data and computes Repeated Measures ANOVA with full diagnostics.

    Args:
        n_subjects: Number of matched participants measured across all conditions.
        time_names: Names for the repeated occasions/conditions (k=3).
        means: Target population mean at each time point.
        subject_sd: Between-subject variability (random subject baseline offset).
        within_sd: Within-subject measurement noise across time points.
        sphericity_violation: If True, creates unequal pairwise difference variances (violating sphericity).
        dist_shape: 'normal', 'skewed_moderate' (Gamma), or 'heavy_tailed' (Student's t, df=3).
        outlier_count: Number of extreme within-subject spikes to inject.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (long_format_DataFrame, results_dict).
    """
    rng = np.random.default_rng(seed)
    k = len(time_names)

    # 1. Subject baseline random intercepts
    subj_effects = rng.normal(loc=0.0, scale=subject_sd, size=n_subjects)

    # 2. Within-subject variations across time points
    if dist_shape == "normal":
        if not sphericity_violation:
            err0 = rng.normal(0.0, within_sd, size=n_subjects)
            err1 = rng.normal(0.0, within_sd, size=n_subjects)
            err2 = rng.normal(0.0, within_sd, size=n_subjects)
        else:
            # Create unequal variance of differences (violates sphericity)
            # T1 and T2 are tightly coupled, while T3 has high independent variance
            err0 = rng.normal(0.0, 1.5, size=n_subjects)
            err1 = err0 * 0.9 + rng.normal(0.0, 1.0, size=n_subjects)
            err2 = rng.normal(0.0, max(within_sd * 2.8, 9.0), size=n_subjects)
    elif dist_shape == "skewed_moderate":
        shape_p = 2.0
        scale_p = within_sd / np.sqrt(shape_p)
        err0 = rng.gamma(shape=shape_p, scale=scale_p, size=n_subjects) - shape_p * scale_p
        err1 = rng.gamma(shape=shape_p, scale=scale_p, size=n_subjects) - shape_p * scale_p
        err2 = rng.gamma(shape=shape_p, scale=scale_p, size=n_subjects) - shape_p * scale_p
    elif dist_shape == "heavy_tailed":
        scale_t = within_sd * np.sqrt(1.0 / 3.0)
        err0 = rng.standard_t(df=3, size=n_subjects) * scale_t
        err1 = rng.standard_t(df=3, size=n_subjects) * scale_t
        err2 = rng.standard_t(df=3, size=n_subjects) * scale_t
    else:
        err0 = rng.normal(0.0, within_sd, size=n_subjects)
        err1 = rng.normal(0.0, within_sd, size=n_subjects)
        err2 = rng.normal(0.0, within_sd, size=n_subjects)

    # 3. Outlier injection on the final condition
    if outlier_count > 0:
        err2[-1] += 4.5 * max(subject_sd, within_sd, 6.0)

    # 4. Construct Wide Matrix Y (n_subjects x 3)
    y0 = np.round(means[0] + subj_effects + err0, 2)
    y1 = np.round(means[1] + subj_effects + err1, 2)
    y2 = np.round(means[2] + subj_effects + err2, 2)
    Y = np.column_stack([y0, y1, y2])

    # 5. Descriptive statistics per condition
    time_means = [float(np.mean(Y[:, j])) for j in range(k)]
    time_stds = [float(np.std(Y[:, j], ddof=1)) for j in range(k)]
    time_medians = [float(np.median(Y[:, j])) for j in range(k)]
    time_q25 = [float(np.percentile(Y[:, j], 25)) for j in range(k)]
    time_q75 = [float(np.percentile(Y[:, j], 75)) for j in range(k)]
    time_iqrs = [time_q75[j] - time_q25[j] for j in range(k)]
    grand_mean = float(np.mean(Y))

    # 6. Sum of Squares Partitioning
    subj_means = np.mean(Y, axis=1)  # shape (n,)
    col_means = np.mean(Y, axis=0)   # shape (k,)

    ss_total = float(np.sum((Y - grand_mean) ** 2))
    ss_subjects = float(k * np.sum((subj_means - grand_mean) ** 2))
    ss_time = float(n_subjects * np.sum((col_means - grand_mean) ** 2))
    ss_error = max(0.0001, ss_total - ss_subjects - ss_time)

    df_total = n_subjects * k - 1
    df_subjects = n_subjects - 1
    df_time = k - 1  # 2
    df_error = (n_subjects - 1) * (k - 1)  # 2 * (n - 1)

    ms_time = ss_time / df_time
    ms_error = ss_error / df_error
    ms_subjects = ss_subjects / df_subjects

    f_stat = float(ms_time / ms_error)
    p_anova = float(1.0 - stats.f.cdf(f_stat, df_time, df_error))
    f_crit = float(stats.f.ppf(0.95, df_time, df_error))

    eta_p2 = float(ss_time / (ss_time + ss_error))
    eta_g2 = float(ss_time / (ss_time + ss_subjects + ss_error))

    # 7. Sphericity Analysis (Mauchly's Test & Greenhouse-Geisser Epsilon)
    diff_10 = y1 - y0
    diff_20 = y2 - y0
    diff_21 = y2 - y1

    var_diff_10 = float(np.var(diff_10, ddof=1))
    var_diff_20 = float(np.var(diff_20, ddof=1))
    var_diff_21 = float(np.var(diff_21, ddof=1))
    diff_vars = [var_diff_10, var_diff_20, var_diff_21]
    max_diff_var = max(diff_vars)
    min_diff_var = min(diff_vars)
    var_ratio_diff = max_diff_var / max(0.0001, min_diff_var)

    # Orthonormal contrast matrix C (3 x 2)
    C = np.array([
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
        [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
        [0.0, -2.0 / np.sqrt(6.0)]
    ])
    S = np.cov(Y, rowvar=False)
    S_C = C.T @ S @ C
    eigvals = np.linalg.eigvalsh(S_C)
    eigvals = np.maximum(eigvals, 1e-10)
    tr_sc = float(np.sum(eigvals))
    det_sc = float(np.prod(eigvals))
    w_mauchly = max(0.0001, min(1.0, float(det_sc / ((tr_sc / 2.0) ** 2))))

    df_mauchly = 2
    if w_mauchly >= 0.999:
        p_mauchly = 1.0
        chi2_mauchly = 0.0
    else:
        d_factor = 1.0 - 1.0 / max(1, n_subjects - 1)
        chi2_mauchly = float(-(n_subjects - 1) * d_factor * np.log(w_mauchly))
        p_mauchly = float(1.0 - stats.chi2.cdf(chi2_mauchly, df=df_mauchly))

    eps_gg = float((tr_sc ** 2) / (2.0 * np.sum(eigvals ** 2)))
    eps_gg = max(0.50, min(1.0, eps_gg))  # Lower bound 1/(k-1) = 0.50

    df_time_gg = df_time * eps_gg
    df_error_gg = df_error * eps_gg
    p_gg = float(1.0 - stats.f.cdf(f_stat, df_time_gg, df_error_gg))

    # 8. Post-Hoc Pairwise Paired t-tests with Bonferroni Correction
    pairs_spec = [
        (1, 0, diff_10, f"{time_names[1]} vs {time_names[0]}"),
        (2, 0, diff_20, f"{time_names[2]} vs {time_names[0]}"),
        (2, 1, diff_21, f"{time_names[2]} vs {time_names[1]}")
    ]
    num_comparisons = len(pairs_spec)
    t_crit_bonf = float(stats.t.ppf(1.0 - 0.05 / (2.0 * num_comparisons), df=n_subjects - 1))

    pairwise_comparisons = []
    for g_b_idx, g_a_idx, diff_arr, comp_label in pairs_spec:
        name_b = time_names[g_b_idx]
        name_a = time_names[g_a_idx]
        diff_mean = float(np.mean(diff_arr))
        diff_std = float(np.std(diff_arr, ddof=1))
        diff_se = float(diff_std / np.sqrt(n_subjects)) if n_subjects > 0 else 1.0
        t_pair = float(diff_mean / diff_se) if diff_se > 0 else 0.0
        p_raw = float(2.0 * (1.0 - stats.t.cdf(abs(t_pair), df=n_subjects - 1)))
        p_adj = min(1.0, float(p_raw * num_comparisons))
        ci_low = float(diff_mean - t_crit_bonf * diff_se)
        ci_high = float(diff_mean + t_crit_bonf * diff_se)
        is_sig = bool(p_adj < 0.05)
        p_str = f"{p_adj:.4f}" if p_adj >= 0.0001 else "< 0.0001"

        if is_sig:
            direction = "higher" if diff_mean > 0 else "lower"
            plain = (
                f"{name_b} scores were significantly {direction} than {name_a} "
                f"(mean change: {diff_mean:+.2f}, 95% Bonferroni CI [{ci_low:.2f}, {ci_high:.2f}], adjusted p = {p_str})."
            )
        else:
            plain = (
                f"No statistically significant difference between {name_b} and {name_a} "
                f"(mean change: {diff_mean:+.2f}, 95% Bonferroni CI [{ci_low:.2f}, {ci_high:.2f}], adjusted p = {p_str}). "
                f"Any observed change is consistent with random chance."
            )

        pairwise_comparisons.append({
            "comparison": comp_label,
            "group_a": name_a,
            "group_b": name_b,
            "diff_mean": diff_mean,
            "diff_str": f"{diff_mean:+.2f}",
            "diff_std": diff_std,
            "diff_se": diff_se,
            "t_stat": t_pair,
            "p_raw": p_raw,
            "p_adj": p_adj,
            "p_adj_str": p_str,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "ci_str": f"[{ci_low:.2f}, {ci_high:.2f}]",
            "significant": is_sig,
            "significance_verdict": "✅ Significant Difference" if is_sig else "➖ No Difference",
            "plain_english": plain
        })

    # 9. Non-Parametric Alternative: Friedman Test
    friedman_stat, p_friedman = stats.friedmanchisquare(y0, y1, y2)
    friedman_stat = float(friedman_stat)
    p_friedman = float(p_friedman)

    # 10. Normality and Outlier Diagnostics on Differences
    diff_arrays = [diff_10, diff_20, diff_21]
    shap_res = [stats.shapiro(d) for d in diff_arrays]
    shap_p_list = [float(r.pvalue) for r in shap_res]
    min_shap_p = min(shap_p_list)
    skew_list = [float(stats.skew(d)) for d in diff_arrays]
    kurt_list = [float(stats.kurtosis(d)) for d in diff_arrays]
    max_abs_skew = max(abs(s) for s in skew_list)

    def count_diff_outliers(d):
        q25 = np.percentile(d, 25)
        q75 = np.percentile(d, 75)
        iqr = q75 - q25
        return int(np.sum((d < q25 - 3.0 * iqr) | (d > q75 + 3.0 * iqr)))

    outlier_counts = [count_diff_outliers(d) for d in diff_arrays]
    total_ext_outliers = sum(outlier_counts)

    # 11. Core Assumption Evaluations (4 Cards)
    asm_scale = {
        "name": "1. Repeated Measurement Scale & Subject Matching",
        "description": "Continuous dependent variable measured across identical matched participants across all occasions.",
        "test_applied": "Within-Subjects Matched Design Verification",
        "diagnostic_value": f"N = {n_subjects} matched subjects across {k} repeated conditions",
        "status": "pass",
        "verdict": "PASS",
        "detail": "Data satisfies continuous measurement scale with complete repeated observations on every participant."
    }

    if min_shap_p >= 0.05:
        norm_status = "pass"
        norm_verdict = "PASS (Normal Differences)"
        norm_detail = f"All pairwise differences conform to normality (Min Shapiro p = {min_shap_p:.4f}, max |skew| = {max_abs_skew:.2f})."
    elif n_subjects >= 30 and max_abs_skew <= 1.2:
        norm_status = "warning"
        norm_verdict = "BORDERLINE (CLT Robust, N >= 30)"
        norm_detail = f"Mild non-normality in differences (Min Shapiro p = {min_shap_p:.4f}), but N = {n_subjects} >= 30 provides Central Limit Theorem robustness."
    else:
        norm_status = "fail"
        norm_verdict = "VIOLATED (Non-Normal Differences)"
        norm_detail = f"Pairwise differences violate normality (Min Shapiro p = {min_shap_p:.4f}, max |skew| = {max_abs_skew:.2f}). Consider Friedman Test."

    asm_norm = {
        "name": "2. Normality of Pairwise Differences",
        "description": "Differences between repeated conditions should follow approximate normal distributions.",
        "test_applied": "Shapiro-Wilk Test & Skewness on Difference Scores",
        "diagnostic_value": f"Min Shapiro p = {min_shap_p:.4f}, Max |Skew| = {max_abs_skew:.2f}",
        "status": norm_status,
        "verdict": norm_verdict,
        "detail": norm_detail
    }

    if p_mauchly >= 0.05 and eps_gg >= 0.75:
        spher_status = "pass"
        spher_verdict = "PASS (Sphericity Satisfied)"
        spher_detail = f"Mauchly's test is non-significant (W = {w_mauchly:.3f}, p = {p_mauchly:.4f}, ε = {eps_gg:.3f} >= 0.75). Standard RM-ANOVA F-test is valid."
    elif eps_gg >= 0.75:
        spher_status = "warning"
        spher_verdict = "BORDERLINE (Minor Sphericity Departure)"
        spher_detail = f"Mauchly W = {w_mauchly:.3f} (p = {p_mauchly:.4f}), but Greenhouse-Geisser ε = {eps_gg:.3f} >= 0.75. RM-ANOVA is reasonably robust."
    else:
        spher_status = "fail"
        spher_verdict = "VIOLATED (Apply Greenhouse-Geisser Correction)"
        spher_detail = f"Sphericity is violated (Mauchly W = {w_mauchly:.3f}, p = {p_mauchly:.4f}, ε = {eps_gg:.3f} < 0.75). You MUST report Greenhouse-Geisser adjusted degrees of freedom."

    asm_spher = {
        "name": "3. Sphericity (Equal Variances of Differences)",
        "description": "Variances of differences between all possible condition pairs should be approximately equal.",
        "test_applied": "Mauchly's Sphericity Test & Greenhouse-Geisser Epsilon",
        "diagnostic_value": f"Mauchly W = {w_mauchly:.3f} (p = {p_mauchly:.4f}), GG ε = {eps_gg:.3f}",
        "status": spher_status,
        "verdict": spher_verdict,
        "detail": spher_detail
    }

    if total_ext_outliers == 0:
        outlier_status = "pass"
        outlier_verdict = "PASS (No Outliers)"
        outlier_detail = "Zero extreme within-subject outliers detected across all pairwise difference distributions."
    else:
        outlier_status = "fail"
        outlier_verdict = f"VIOLATED ({total_ext_outliers} Extreme Outliers)"
        outlier_detail = f"Found {total_ext_outliers} extreme difference score(s) (> 3.0×IQR). Distorts RM-ANOVA error variance."

    asm_outliers = {
        "name": "4. Absence of Extreme Within-Subject Outliers",
        "description": "No participant displaying extreme difference scores (> 3.0×IQR) that inflate within-subject error variance.",
        "test_applied": "Tukey's IQR Outlier Rule on Differences",
        "diagnostic_value": f"Total Extreme Outliers = {total_ext_outliers}",
        "status": outlier_status,
        "verdict": outlier_verdict,
        "detail": outlier_detail
    }

    assumptions_eval = [asm_scale, asm_norm, asm_spher, asm_outliers]

    # 12. Overall Recommendation Synthesis
    has_failure = any(a["status"] == "fail" for a in assumptions_eval)
    has_warning = any(a["status"] == "warning" for a in assumptions_eval)

    if spher_status == "fail":
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption Violated: Sphericity Failed (Use Greenhouse-Geisser Correction)"
        overall_recommendation = (
            f"The sphericity assumption is violated (Mauchly's W = {w_mauchly:.3f}, p = {p_mauchly:.4e}, "
            f"Greenhouse-Geisser ε = {eps_gg:.3f} < 0.75). Standard unadjusted degrees of freedom inflate Type I false alarms. "
            f"\n\n**Actionable Recommendation:** Apply the **Greenhouse-Geisser correction**, reducing degrees of freedom to "
            f"df₁ = {df_time_gg:.2f} and df₂ = {df_error_gg:.2f}. "
            f"Adjusted omnibus test: F({df_time_gg:.2f}, {df_error_gg:.2f}) = {f_stat:.2f}, **Greenhouse-Geisser p = {p_gg:.4e}**."
        )
    elif norm_status == "fail" or outlier_status == "fail":
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption Violated: Non-Normality / Outliers (Switch to Friedman Test)"
        overall_recommendation = (
            f"Severe non-normality or extreme within-subject outliers violate parametric assumptions. "
            f"\n\n**Actionable Recommendation:** Switch to the non-parametric **Friedman Test** "
            f"(Chi-Square({df_time}) = {friedman_stat:.2f}, **p = {p_friedman:.4e}**), which evaluates within-subject ranks "
            f"without requiring normality or sphericity."
        )
    elif has_warning:
        overall_status = "warning"
        overall_verdict_badge = "🟡 Assumptions Borderline: Proceed with Caution"
        overall_recommendation = (
            f"Minor departures from assumptions detected, but sample size (n = {n_subjects}) provides reasonable robustness. "
            f"\n\n**Actionable Recommendation:** You may report standard **Repeated Measures ANOVA** "
            f"(F({df_time}, {df_error}) = {f_stat:.2f}, p = {p_anova:.4e}), but cross-reference with the Greenhouse-Geisser adjusted "
            f"p-value (p = {p_gg:.4e}) and the non-parametric **Friedman Test** (p = {p_friedman:.4e})."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ All Assumptions Met: Proceed with Repeated Measures ANOVA"
        overall_recommendation = (
            f"All core assumptions (within-subject matching, normality of differences, sphericity, and lack of extreme outliers) "
            f"are successfully satisfied. "
            f"\n\n**Actionable Recommendation:** Proceed with confidence using **Repeated Measures ANOVA**. "
            f"Omnibus test: F({df_time}, {df_error}) = {f_stat:.2f}, **p = {p_anova:.4e}**, "
            f"Partial Eta-Squared η_p² = {eta_p2:.3f} ({'Large effect' if eta_p2 >= 0.14 else ('Medium effect' if eta_p2 >= 0.06 else 'Small effect')})."
        )

    # 13. Assemble Long-Format DataFrame
    subj_ids = [f"Subj_{i+1:03d}" for i in range(n_subjects)]
    long_rows = []
    for i in range(n_subjects):
        for j in range(k):
            long_rows.append({
                "Subject_ID": subj_ids[i],
                "Time_Point": time_names[j],
                "Time_Index": j,
                "Score": float(Y[i, j])
            })
    df_long = pd.DataFrame(long_rows)

    results = {
        "n_subjects": int(n_subjects),
        "k_times": int(k),
        "time_names": list(time_names),
        "means": time_means,
        "stds": time_stds,
        "medians": time_medians,
        "q25s": time_q25,
        "q75s": time_q75,
        "iqrs": time_iqrs,
        "grand_mean": grand_mean,
        "Y_wide": Y,
        "diff_10": diff_10,
        "diff_20": diff_20,
        "diff_21": diff_21,
        "diff_vars": diff_vars,
        "var_ratio_diff": var_ratio_diff,
        "ss_total": ss_total,
        "ss_subjects": ss_subjects,
        "ss_time": ss_time,
        "ss_error": ss_error,
        "df_total": df_total,
        "df_subjects": df_subjects,
        "df_time": df_time,
        "df_error": df_error,
        "ms_time": ms_time,
        "ms_error": ms_error,
        "ms_subjects": ms_subjects,
        "f_stat": f_stat,
        "p_anova": p_anova,
        "f_crit": f_crit,
        "eta_p2": eta_p2,
        "eta_g2": eta_g2,
        "w_mauchly": w_mauchly,
        "p_mauchly": p_mauchly,
        "chi2_mauchly": chi2_mauchly,
        "eps_gg": eps_gg,
        "df_time_gg": df_time_gg,
        "df_error_gg": df_error_gg,
        "p_gg": p_gg,
        "pairwise_comparisons": pairwise_comparisons,
        "friedman_stat": friedman_stat,
        "p_friedman": p_friedman,
        "shapiro_p": shap_p_list,
        "skewness": skew_list,
        "kurtosis": kurt_list,
        "outlier_counts": outlier_counts,
        "total_ext_outliers": total_ext_outliers,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df_long, results


def generate_friedman_demo(
    n_subjects: int = 25,
    condition_names: Tuple[str, str, str] = ("Condition 1 (Baseline)", "Condition 2 (Low)", "Condition 3 (High)"),
    shifts: Tuple[float, float, float] = (0.0, 3.0, 6.5),
    data_type: str = "ordinal_likert",
    dist_shape: str = "skewed_moderate",
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic repeated measures data and computes the Friedman Test with comprehensive rank diagnostics.

    Args:
        n_subjects: Number of matched participants (blocks) measured across all repeated occasions.
        condition_names: Names for the 3 repeated occasions/conditions.
        shifts: Group location shifts / median differences added to latent scores.
        data_type: 'ordinal_likert' (1-7 scale) or 'continuous' (scores).
        dist_shape: 'skewed_moderate' (Gamma), 'heavy_tailed' (Student's t, df=3), or 'normal'.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (long_format_DataFrame, results_dict).
    """
    rng = np.random.default_rng(seed)
    k = len(condition_names)

    # 1. Subject baseline random intercepts
    subj_intercepts = rng.normal(0.0, 6.0, size=n_subjects)

    # 2. Condition latent scores
    raw_conditions = []
    for j in range(k):
        shift = shifts[j]
        if dist_shape == "skewed_moderate":
            noise = rng.gamma(shape=2.0, scale=2.5, size=n_subjects)
        elif dist_shape == "heavy_tailed":
            noise = rng.standard_t(df=3, size=n_subjects) * 3.5
        elif dist_shape == "normal":
            noise = rng.normal(0.0, 4.0, size=n_subjects)
        else:
            noise = rng.normal(0.0, 4.0, size=n_subjects)
        raw_conditions.append(20.0 + subj_intercepts + shift + noise)

    # 3. Discretize/Scale based on data_type
    if data_type == "ordinal_likert":
        pooled_min = min(float(np.min(c)) for c in raw_conditions)
        pooled_max = max(float(np.max(c)) for c in raw_conditions)
        span = pooled_max - pooled_min if pooled_max > pooled_min else 1.0
        final_conditions = []
        for c in raw_conditions:
            scaled = 1.0 + 6.0 * (c - pooled_min) / span
            likert = np.clip(np.round(scaled), 1.0, 7.0).astype(float)
            final_conditions.append(likert)
    else:
        final_conditions = [np.round(c, 1).astype(float) for c in raw_conditions]

    c0, c1, c2 = final_conditions[0], final_conditions[1], final_conditions[2]
    Y = np.column_stack([c0, c1, c2])

    # 4. Within-Subject Ranks (Mid-ranks for ties across each subject row)
    R = np.zeros_like(Y, dtype=float)
    for i in range(n_subjects):
        R[i, :] = stats.rankdata(Y[i, :], method="average")

    # 5. Descriptive & Rank Metrics
    medians = [float(np.median(Y[:, j])) for j in range(k)]
    q25s = [float(np.percentile(Y[:, j], 25)) for j in range(k)]
    q75s = [float(np.percentile(Y[:, j], 75)) for j in range(k)]
    iqrs = [q75s[j] - q25s[j] for j in range(k)]
    means = [float(np.mean(Y[:, j])) for j in range(k)]
    stds = [float(np.std(Y[:, j], ddof=1)) for j in range(k)]
    skews = [float(stats.skew(Y[:, j])) for j in range(k)]

    rank_sums = [float(np.sum(R[:, j])) for j in range(k)]
    mean_ranks = [float(np.mean(R[:, j])) for j in range(k)]
    exp_mean_rank = float((k + 1.0) / 2.0)  # 2.0 for k=3

    # 6. Friedman Omnibus Test Statistic & p-value
    q_stat, p_friedman = stats.friedmanchisquare(c0, c1, c2)
    q_stat = float(q_stat)
    p_friedman = float(p_friedman)
    df_friedman = k - 1  # 2

    # Tie counts and tie correction factor
    total_tie_pairs = 0
    for i in range(n_subjects):
        _, row_counts = np.unique(Y[i, :], return_counts=True)
        total_tie_pairs += int(np.sum(row_counts[row_counts > 1]))

    # Effect Size: Kendall's W (Coefficient of Concordance)
    kendalls_w = float(q_stat / (n_subjects * (k - 1))) if n_subjects > 0 else 0.0
    kendalls_w = max(0.0, min(1.0, kendalls_w))
    if kendalls_w < 0.1:
        effect_magnitude = "Negligible effect"
    elif kendalls_w < 0.3:
        effect_magnitude = "Small effect"
    elif kendalls_w < 0.5:
        effect_magnitude = "Moderate effect"
    else:
        effect_magnitude = "Strong effect"

    # 7. Post-Hoc Pairwise Comparisons: Wilcoxon Signed-Rank with Bonferroni Correction
    pairs = [(1, 0), (2, 0), (2, 1)]
    num_comparisons = len(pairs)
    se_rank = np.sqrt((k * (k + 1.0)) / (6.0 * n_subjects)) if n_subjects > 0 else 1.0

    pairwise_comparisons = []
    for j, i in pairs:
        name_b = condition_names[j]
        name_a = condition_names[i]
        diff_arr = Y[:, j] - Y[:, i]
        diff_rank = mean_ranks[j] - mean_ranks[i]
        diff_med = medians[j] - medians[i]

        try:
            w_res = stats.wilcoxon(Y[:, j], Y[:, i], alternative="two-sided")
            w_stat = float(w_res.statistic)
            p_raw = float(w_res.pvalue)
        except Exception:
            w_stat = 0.0
            p_raw = 1.0

        p_adj = min(1.0, float(p_raw * num_comparisons))
        z_rank = abs(diff_rank) / se_rank if se_rank > 0 else 0.0

        is_sig = bool(p_adj < 0.05)
        p_str = f"{p_adj:.4f}" if p_adj >= 0.0001 else "< 0.0001"
        if is_sig:
            direction = "higher" if diff_rank > 0 else "lower"
            plain = (
                f"{name_b} ranked significantly {direction} than {name_a} "
                f"(mean rank diff: {diff_rank:+.2f}, median diff: {diff_med:+.1f}, "
                f"Wilcoxon W = {w_stat:.1f}, Bonferroni-adjusted p = {p_str})."
            )
        else:
            plain = (
                f"No statistically significant difference in ranks between {name_b} and {name_a} "
                f"(mean rank diff: {diff_rank:+.2f}, median diff: {diff_med:+.1f}, "
                f"Bonferroni-adjusted p = {p_str}). Any observed difference is consistent with random chance."
            )

        pairwise_comparisons.append({
            "comparison": f"{name_b} vs {name_a}",
            "group_a": name_a,
            "group_b": name_b,
            "diff_mean_rank": float(diff_rank),
            "diff_rank_str": f"{diff_rank:+.2f}",
            "diff_median": float(diff_med),
            "diff_median_str": f"{diff_med:+.1f}",
            "w_stat": float(w_stat),
            "z_rank": float(z_rank),
            "p_raw": float(p_raw),
            "p_adj": float(p_adj),
            "p_adj_str": p_str,
            "significant": is_sig,
            "significance_verdict": "✅ Significant Difference" if is_sig else "➖ No Difference",
            "plain_english": plain
        })

    # 8. Parametric Repeated Measures ANOVA Comparison
    grand_m = np.mean(Y)
    ss_tot = np.sum((Y - grand_m)**2)
    ss_sub = k * np.sum((np.mean(Y, axis=1) - grand_m)**2)
    ss_cond = n_subjects * np.sum((np.mean(Y, axis=0) - grand_m)**2)
    ss_res = max(0.0001, ss_tot - ss_sub - ss_cond)
    ms_cond = ss_cond / (k - 1)
    ms_res = ss_res / ((n_subjects - 1) * (k - 1))
    f_stat_param = float(ms_cond / ms_res)
    p_anova_param = float(1.0 - stats.f.cdf(f_stat_param, k - 1, (n_subjects - 1) * (k - 1)))

    # Normality check on difference scores
    diff_10 = c1 - c0
    diff_20 = c2 - c0
    diff_21 = c2 - c1
    shap_diff_p = min(float(stats.shapiro(d).pvalue) for d in [diff_10, diff_20, diff_21])

    # 9. Core Assumption Evaluations (4 Cards)
    asm_scale = {
        "name": "1. Measurement Scale & Matched Design",
        "description": "Dependent variable is at least ordinal, measured across identical matched participants across all occasions.",
        "test_applied": "Within-Subjects Matched Design Verification",
        "diagnostic_value": f"{'Ordinal Likert (1-7)' if data_type == 'ordinal_likert' else 'Continuous Scores'} (N = {n_subjects} matched subjects)",
        "status": "pass",
        "verdict": "PASS",
        "detail": "Data satisfies the ordinal or continuous measurement requirement with complete repeated observations on every subject."
    }

    asm_indep = {
        "name": "2. Mutual Independence of Subjects (Blocks)",
        "description": "Participants (blocks) must be mutually independent of one another (random sampling of participants).",
        "test_applied": "Subject Sampling & Block Independence Audit",
        "diagnostic_value": f"N = {n_subjects} independent subject blocks",
        "status": "pass",
        "verdict": "PASS",
        "detail": "Subjects were sampled independently; within-subject correlation is accounted for by the rank block design."
    }

    is_non_normal = (data_type == "ordinal_likert") or (shap_diff_p < 0.05) or any(abs(s) > 0.8 for s in skews)
    if is_non_normal:
        nonpar_status = "pass"
        nonpar_verdict = "PASS (Non-Parametric Justified)"
        nonpar_detail = (
            f"Data is {'ordinal' if data_type == 'ordinal_likert' else 'non-normally distributed'} "
            f"(Min difference Shapiro p = {shap_diff_p:.4f}, max |skew| = {max(abs(s) for s in skews):.2f}). "
            f"Friedman Test is the appropriate, robust methodological choice over standard RM-ANOVA."
        )
    else:
        nonpar_status = "warning"
        nonpar_verdict = "INFO (Parametric RM-ANOVA is Valid & More Powerful)"
        nonpar_detail = (
            f"Data satisfies normality (Min diff Shapiro p = {shap_diff_p:.4f}). "
            f"While the Friedman test is valid, Repeated Measures ANOVA (F = {f_stat_param:.2f}, p = {p_anova_param:.4e}) "
            f"provides greater statistical power for normal continuous data."
        )

    asm_nonpar = {
        "name": "3. Non-Parametric Appropriateness vs RM-ANOVA",
        "description": "Evaluates whether non-parametric within-subject ranking is justified over parametric Repeated Measures ANOVA.",
        "test_applied": "Normality (Shapiro-Wilk on Differences) & Measurement Scale Audit",
        "diagnostic_value": f"{'Ordinal Likert' if data_type == 'ordinal_likert' else f'Min Diff Shapiro p = {shap_diff_p:.4f}'}",
        "status": nonpar_status,
        "verdict": nonpar_verdict,
        "detail": nonpar_detail
    }

    if n_subjects >= 15:
        size_status = "pass"
        size_verdict = "PASS (N >= 15)"
        size_detail = f"Sample size (N = {n_subjects}) comfortably satisfies the asymptotic Chi-Square threshold (N >= 10 for k = 3)."
    elif n_subjects >= 10:
        size_status = "warning"
        size_verdict = "BORDERLINE (10 <= N < 15)"
        size_detail = f"Sample size (N = {n_subjects}) meets minimum asymptotic criteria (N >= 10), but exact permutation values are advisable if p is near 0.05."
    else:
        size_status = "fail"
        size_verdict = "VIOLATED (N < 10)"
        size_detail = f"Sample size (N = {n_subjects} < 10) is too small for asymptotic Chi-Square approximation. Exact permutation tables required."

    asm_size = {
        "name": "4. Asymptotic Sample Size Adequacy (N >= 10)",
        "description": "Number of matched subjects should be at least 10 (ideally >= 15) for Chi-Square distribution to accurately approximate Q.",
        "test_applied": "Asymptotic Sample Size Rule",
        "diagnostic_value": f"N = {n_subjects} matched subjects (k = {k})",
        "status": size_status,
        "verdict": size_verdict,
        "detail": size_detail
    }

    assumptions_eval = [asm_scale, asm_indep, asm_nonpar, asm_size]

    # 10. Overall Recommendation Synthesis
    if p_friedman < 0.05:
        sig_count = sum(1 for p in pairwise_comparisons if p["significant"])
        overall_status = "pass" if size_status == "pass" else "warning"
        overall_verdict_badge = "✅ Statistically Significant Difference Across Conditions (p < 0.05)"
        overall_recommendation = (
            f"Reject the null hypothesis H₀. Within-subject ranks differ significantly across repeated conditions "
            f"(Friedman Q({df_friedman}) = {q_stat:.3f}, **p = {p_friedman:.4e}**, Kendall's W = {kendalls_w:.3f} [{effect_magnitude}]). "
            f"\n\n**Actionable Post-Hoc Guidance:** Inspect the pairwise Wilcoxon signed-rank tests below. "
            f"{sig_count} of {num_comparisons} pairwise comparison(s) are statistically significant after Bonferroni correction at alpha = 0.05."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ No Statistically Significant Difference Across Conditions (p ≥ 0.05)"
        overall_recommendation = (
            f"Fail to reject the null hypothesis H₀. Observed rank differences across conditions are consistent "
            f"with random sampling variability (Friedman Q({df_friedman}) = {q_stat:.3f}, **p = {p_friedman:.4f}**, Kendall's W = {kendalls_w:.3f}). "
            f"\n\n**Actionable Post-Hoc Guidance:** Pairwise post-hoc comparisons are not indicated because the omnibus Friedman test did not reach statistical significance."
        )

    # 11. Assemble Long-Format DataFrame
    subj_ids = [f"Subj_{i+1:03d}" for i in range(n_subjects)]
    long_rows = []
    for i in range(n_subjects):
        for j in range(k):
            long_rows.append({
                "Subject_ID": subj_ids[i],
                "Condition": condition_names[j],
                "Condition_Index": j,
                "Score": float(Y[i, j]),
                "Within_Subject_Rank": float(R[i, j])
            })
    df_long = pd.DataFrame(long_rows)

    results = {
        "n_subjects": int(n_subjects),
        "k_conditions": int(k),
        "condition_names": list(condition_names),
        "data_type": data_type,
        "dist_shape": dist_shape,
        "medians": medians,
        "q25s": q25s,
        "q75s": q75s,
        "iqrs": iqrs,
        "means": means,
        "stds": stds,
        "skews": skews,
        "rank_sums": rank_sums,
        "mean_ranks": mean_ranks,
        "expected_mean_rank": exp_mean_rank,
        "q_stat": q_stat,
        "p_friedman": p_friedman,
        "df_friedman": df_friedman,
        "kendalls_w": kendalls_w,
        "effect_magnitude": effect_magnitude,
        "total_tie_pairs": total_tie_pairs,
        "pairwise_comparisons": pairwise_comparisons,
        "f_stat_param": f_stat_param,
        "p_anova_param": p_anova_param,
        "Y_wide": Y,
        "R_wide": R,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df_long, results


def generate_contingency_demo(
    exposed_cases: int = 45,
    exposed_noncases: int = 55,
    unexposed_cases: int = 20,
    unexposed_noncases: int = 80,
    group1_name: str = "Treatment / Exposed",
    group2_name: str = "Control / Unexposed",
    outcome_pos_name: str = "Event / Disease (+)",
    outcome_neg_name: str = "No Event / Healthy (-)",
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates 2x2 contingency table data, executes Pearson Chi-Square & Fisher's Exact tests,
    and calculates comprehensive cell residuals (Haberman), effect sizes, and assumption diagnostics.

    Args:
        exposed_cases: Cell (0, 0) - Group 1 with positive outcome.
        exposed_noncases: Cell (0, 1) - Group 1 with negative outcome.
        unexposed_cases: Cell (1, 0) - Group 2 with positive outcome.
        unexposed_noncases: Cell (1, 1) - Group 2 with negative outcome.
        group1_name: Label for row 1.
        group2_name: Label for row 2.
        outcome_pos_name: Label for column 1.
        outcome_neg_name: Label for column 2.
        alpha: Type I error significance threshold (default 0.05).

    Returns:
        Tuple of (df_table, results_dict).
    """
    a = max(0, int(exposed_cases))
    b = max(0, int(exposed_noncases))
    c = max(0, int(unexposed_cases))
    d = max(0, int(unexposed_noncases))

    table = np.array([[a, b], [c, d]], dtype=int)

    r1 = a + b
    r2 = c + d
    c1 = a + c
    c2 = b + d
    n_total = r1 + r2

    # Expected frequencies matrix: E_ij = (R_i * C_j) / N
    if n_total > 0:
        e11 = (r1 * c1) / n_total
        e12 = (r1 * c2) / n_total
        e21 = (r2 * c1) / n_total
        e22 = (r2 * c2) / n_total
    else:
        e11 = e12 = e21 = e22 = 0.0

    expected = np.array([[e11, e12], [e21, e22]], dtype=float)
    min_exp_count = float(np.min(expected)) if n_total > 0 else 0.0

    # 1. Uncorrected Pearson Chi-Square Test
    if n_total > 0 and (r1 * r2 * c1 * c2) > 0:
        chi2_uncorr = float(np.sum((table - expected) ** 2 / expected))
        p_uncorr = float(stats.chi2.sf(chi2_uncorr, df=1))
    else:
        chi2_uncorr = 0.0
        p_uncorr = 1.0

    # 2. Yates' Continuity-Corrected Chi-Square Test
    try:
        chi2_yates, p_yates, dof, _ = stats.chi2_contingency(table, correction=True)
        chi2_yates = float(chi2_yates)
        p_yates = float(p_yates)
    except Exception:
        chi2_yates = chi2_uncorr
        p_yates = p_uncorr
        dof = 1

    # 3. Likelihood Ratio Chi-Square (G-squared)
    try:
        lr_g2, p_lr, _, _ = stats.chi2_contingency(table, lambda_="log-likelihood")
        lr_g2 = float(lr_g2)
        p_lr = float(p_lr)
    except Exception:
        lr_g2 = chi2_uncorr
        p_lr = p_uncorr

    # 4. Fisher's Exact Test
    try:
        odds_ratio_fisher, p_fisher = stats.fisher_exact(table, alternative="two-sided")
        odds_ratio_fisher = float(odds_ratio_fisher)
        p_fisher = float(p_fisher)
    except Exception:
        odds_ratio_fisher = 1.0
        p_fisher = 1.0

    # 5. Cell Residuals and Post-Hoc Analysis (Haberman Adjusted Residuals)
    cell_coords = [
        (0, 0, group1_name, outcome_pos_name, a, e11, r1, c1),
        (0, 1, group1_name, outcome_neg_name, b, e12, r1, c2),
        (1, 0, group2_name, outcome_pos_name, c, e21, r2, c1),
        (1, 1, group2_name, outcome_neg_name, d, e22, r2, c2),
    ]

    cells_diagnostics = []
    for r_idx, c_idx, r_label, c_label, obs, exp, r_tot, c_tot in cell_coords:
        raw_diff = float(obs - exp)
        chi2_cell = float((raw_diff ** 2) / exp) if exp > 0 else 0.0
        pct_chi2 = float((chi2_cell / chi2_uncorr * 100.0)) if chi2_uncorr > 0 else 0.0
        pearson_res = float(raw_diff / np.sqrt(exp)) if exp > 0 else 0.0

        # Haberman adjusted standardized residual:
        # z = (O - E) / sqrt(E * (1 - R_i/N) * (1 - C_j/N))
        denom_var = exp * (1.0 - (r_tot / n_total)) * (1.0 - (c_tot / n_total)) if n_total > 0 else 0.0
        if denom_var > 0:
            haberman_z = float(raw_diff / np.sqrt(denom_var))
        else:
            haberman_z = 0.0

        abs_z = abs(haberman_z)
        p_z = float(2.0 * (1.0 - stats.norm.cdf(abs_z)))

        if abs_z >= 3.29:
            sig_badge = "Extremely Significant (p < 0.001)"
            sig_stars = "***"
        elif abs_z >= 2.58:
            sig_badge = "Very Significant (p < 0.01)"
            sig_stars = "**"
        elif abs_z >= 1.96:
            sig_badge = "Significant (p < 0.05)"
            sig_stars = "*"
        else:
            sig_badge = "Not Significant (p ≥ 0.05)"
            sig_stars = "ns"

        if raw_diff > 0.05:
            direction = "Excess (+)"
            plain_res = f"Observed count is higher than expected under independence ({obs} vs {exp:.1f})."
        elif raw_diff < -0.05:
            direction = "Deficit (-)"
            plain_res = f"Observed count is lower than expected under independence ({obs} vs {exp:.1f})."
        else:
            direction = "Equal (≈)"
            plain_res = f"Observed count aligns closely with expected frequency ({obs} vs {exp:.1f})."

        cells_diagnostics.append({
            "row_idx": r_idx,
            "col_idx": c_idx,
            "cell_name": f"{r_label} × {c_label}",
            "row_label": r_label,
            "col_label": c_label,
            "observed": int(obs),
            "expected": float(exp),
            "diff": raw_diff,
            "chi2_contrib": chi2_cell,
            "pct_chi2": pct_chi2,
            "pearson_residual": pearson_res,
            "haberman_residual": haberman_z,
            "p_value": p_z,
            "significance": sig_badge,
            "stars": sig_stars,
            "direction": direction,
            "plain_english": plain_res
        })

    # 6. Proportions, Risks, and Clinical Effect Sizes
    p1 = float(a / r1) if r1 > 0 else 0.0
    p2 = float(c / r2) if r2 > 0 else 0.0

    se_p1 = float(np.sqrt(p1 * (1.0 - p1) / r1)) if r1 > 0 else 0.0
    se_p2 = float(np.sqrt(p2 * (1.0 - p2) / r2)) if r2 > 0 else 0.0

    p1_ci = (float(max(0.0, p1 - 1.96 * se_p1)), float(min(1.0, p1 + 1.96 * se_p1)))
    p2_ci = (float(max(0.0, p2 - 1.96 * se_p2)), float(min(1.0, p2 + 1.96 * se_p2)))

    # Risk Difference (RD) / Absolute Risk Reduction (ARR)
    risk_diff = float(p1 - p2)
    se_rd = float(np.sqrt(se_p1 ** 2 + se_p2 ** 2))
    risk_diff_ci = (float(risk_diff - 1.96 * se_rd), float(risk_diff + 1.96 * se_rd))

    # Number Needed to Treat / Harm (NNT / NNH)
    nnt = float(1.0 / abs(risk_diff)) if abs(risk_diff) > 1e-6 else None

    # Relative Risk (RR) / Risk Ratio
    if p1 > 0 and p2 > 0 and a > 0 and c > 0:
        risk_ratio = float(p1 / p2)
        se_ln_rr = float(np.sqrt((b / (a * r1)) + (d / (c * r2))))
        rr_ci = (float(np.exp(np.log(risk_ratio) - 1.96 * se_ln_rr)),
                 float(np.exp(np.log(risk_ratio) + 1.96 * se_ln_rr)))
    else:
        # Haldane-Anscombe 0.5 correction for zero-cell stability
        a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        r1_adj, r2_adj = a_adj + b_adj, c_adj + d_adj
        risk_ratio = float((a_adj / r1_adj) / (c_adj / r2_adj))
        se_ln_rr = float(np.sqrt((b_adj / (a_adj * r1_adj)) + (d_adj / (c_adj * r2_adj))))
        rr_ci = (float(np.exp(np.log(risk_ratio) - 1.96 * se_ln_rr)),
                 float(np.exp(np.log(risk_ratio) + 1.96 * se_ln_rr)))

    # Odds Ratio (OR) with Woolf 95% CI
    if a > 0 and b > 0 and c > 0 and d > 0:
        odds_ratio = float((a * d) / (b * c))
        se_ln_or = float(np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d))
        or_ci = (float(np.exp(np.log(odds_ratio) - 1.96 * se_ln_or)),
                 float(np.exp(np.log(odds_ratio) + 1.96 * se_ln_or)))
    else:
        a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        odds_ratio = float((a_adj * d_adj) / (b_adj * c_adj))
        se_ln_or = float(np.sqrt(1.0 / a_adj + 1.0 / b_adj + 1.0 / c_adj + 1.0 / d_adj))
        or_ci = (float(np.exp(np.log(odds_ratio) - 1.96 * se_ln_or)),
                 float(np.exp(np.log(odds_ratio) + 1.96 * se_ln_or)))

    # Correlation / Strength of Association Coefficients
    phi = float(np.sqrt(chi2_uncorr / n_total)) if n_total > 0 else 0.0
    cramers_v = phi  # For 2x2 table, min(r-1, c-1) = 1, so Cramér's V == Phi
    cohens_w = phi

    # Effect Magnitude Rule of Thumb
    if cramers_v < 0.10:
        effect_mag = "Negligible / Trivial Association"
    elif cramers_v < 0.30:
        effect_mag = "Small Association"
    elif cramers_v < 0.50:
        effect_mag = "Moderate Association"
    else:
        effect_mag = "Large / Strong Association"

    # 7. Core Methodological Assumptions Diagnostics
    assumptions_eval = [
        {
            "name": "Measurement Scale & Exclusivity",
            "verdict": "Satisfied (2×2 Cross-Classification)",
            "status": "pass",
            "diagnostic_value": "Mutually Exclusive Categorical Factors",
            "description": "Requires two categorical variables where each subject belongs to exactly one row and one column.",
            "detail": f"Observed table contains {r1} subjects in '{group1_name}' and {r2} subjects in '{group2_name}' across '{outcome_pos_name}' and '{outcome_neg_name}'."
        },
        {
            "name": "Independence of Observations",
            "verdict": "Satisfied (Independent Sampling)",
            "status": "pass",
            "diagnostic_value": "Independent Factor Design",
            "description": "Each individual observation must be independent of all other observations.",
            "detail": "Data represents cross-sectional or independent cohort observations. No matched pairs or repeated measures (which would necessitate McNemar's test)."
        },
        {
            "name": "Sample Size Adequacy (Cochran's Criterion)",
            "verdict": "Satisfied (All Expected Counts ≥ 5)" if min_exp_count >= 5.0 else (
                "Violated (Expected Count < 5)" if min_exp_count >= 1.0 else "Severely Violated (Expected Count < 1)"
            ),
            "status": "pass" if min_exp_count >= 5.0 else ("warning" if min_exp_count >= 1.0 else "fail"),
            "diagnostic_value": f"Minimum Expected Cell = {min_exp_count:.2f} (Benchmark: ≥ 5.0)",
            "description": "Cochran's rule dictates that in a 2×2 table, 100% of expected cell frequencies must be at least 5.0.",
            "detail": (
                f"All 4 expected cell counts are ≥ 5.0 (range: {min_exp_count:.2f} to {float(np.max(expected)):.2f}). The asymptotic Chi-Square approximation is fully reliable."
                if min_exp_count >= 5.0 else
                f"The minimum expected cell frequency is {min_exp_count:.2f} (< 5.0). Chi-Square p-values may be distorted; report Fisher's Exact Test instead."
            )
        },
        {
            "name": "Grand Total Sample Size",
            "verdict": "Robust (N ≥ 40)" if n_total >= 40 else ("Adequate (N ≥ 20)" if n_total >= 20 else "Small Sample (N < 20)"),
            "status": "pass" if n_total >= 20 else "warning",
            "diagnostic_value": f"Grand Total N = {n_total} observations",
            "description": "Adequate sample size ensures sufficient precision and statistical power.",
            "detail": f"Total sample of N = {n_total} provides {'solid asymptotic sample size stability' if n_total >= 20 else 'limited statistical power; exact methods strongly recommended'}."
        }
    ]

    # Overall Verdict & Methodological Action
    if min_exp_count >= 5.0:
        overall_status = "pass"
        overall_verdict_badge = "✅ Statistically Valid: Pearson's Chi-Square Test of Independence"
        if p_uncorr < alpha:
            overall_recommendation = (
                f"**Statistically Significant Association Detected (p = {p_uncorr:.4f} < 0.05).** "
                f"We reject the null hypothesis of independence. There is a genuine statistical association between "
                f"**{group1_name} vs. {group2_name}** and **{outcome_pos_name}**. "
                f"Specifically, {group1_name} exhibited an event rate of {p1*100:.1f}% vs. {p2*100:.1f}% in {group2_name} "
                f"(Odds Ratio = {odds_ratio:.2f}, 95% CI: {or_ci[0]:.2f} – {or_ci[1]:.2f}; Risk Difference = {risk_diff*100:+.1f}%). "
                f"Because all expected cell counts are ≥ 5.0 (min: {min_exp_count:.2f}), reporting the standard Pearson Chi-Square test is fully appropriate."
            )
        else:
            overall_recommendation = (
                f"**No Statistically Significant Association Detected (p = {p_uncorr:.4f} ≥ 0.05).** "
                f"We fail to reject the null hypothesis of independence. The observed difference in event proportions "
                f"({p1*100:.1f}% in {group1_name} vs. {p2*100:.1f}% in {group2_name}) is consistent with random chance sampling variability. "
                f"Odds Ratio is {odds_ratio:.2f} (95% CI: {or_ci[0]:.2f} – {or_ci[1]:.2f}), encompassing 1.0 (no effect). "
                f"All expected cell frequencies satisfy Cochran's rule (min: {min_exp_count:.2f} ≥ 5.0)."
            )
    else:
        overall_status = "warning"
        overall_verdict_badge = "🟡 Cochran's Criterion Violated: Report Fisher's Exact Test"
        if p_fisher < alpha:
            overall_recommendation = (
                f"**Statistically Significant Association via Fisher's Exact Test (p = {p_fisher:.4f} < 0.05).** "
                f"Because the minimum expected cell count is {min_exp_count:.2f} (< 5.0), the asymptotic Chi-Square distribution is unreliable. "
                f"**Report Fisher's Exact Test p-value ({p_fisher:.4f}) and Exact Odds Ratio ({odds_ratio_fisher:.2f})** instead. "
                f"The data provides significant evidence of an association between groups and outcomes despite the small cell count."
            )
        else:
            overall_recommendation = (
                f"**No Statistically Significant Association via Fisher's Exact Test (p = {p_fisher:.4f} ≥ 0.05).** "
                f"Because the minimum expected cell count is {min_exp_count:.2f} (< 5.0), standard Chi-Square assumptions are violated. "
                f"**Report Fisher's Exact Test (p = {p_fisher:.4f})** to avoid false-positive inflation from small cell counts."
            )

    # 8. Tables & DataFrames
    df_table = pd.DataFrame(
        table,
        index=[group1_name, group2_name],
        columns=[outcome_pos_name, outcome_neg_name]
    )

    # Subject-level long observational dataframe for data inspection
    subj_ids = []
    subj_groups = []
    subj_outcomes = []
    subj_binary = []

    count = 1
    for _ in range(a):
        subj_ids.append(f"SUBJ_{count:03d}")
        subj_groups.append(group1_name)
        subj_outcomes.append(outcome_pos_name)
        subj_binary.append(1)
        count += 1
    for _ in range(b):
        subj_ids.append(f"SUBJ_{count:03d}")
        subj_groups.append(group1_name)
        subj_outcomes.append(outcome_neg_name)
        subj_binary.append(0)
        count += 1
    for _ in range(c):
        subj_ids.append(f"SUBJ_{count:03d}")
        subj_groups.append(group2_name)
        subj_outcomes.append(outcome_pos_name)
        subj_binary.append(1)
        count += 1
    for _ in range(d):
        subj_ids.append(f"SUBJ_{count:03d}")
        subj_groups.append(group2_name)
        subj_outcomes.append(outcome_neg_name)
        subj_binary.append(0)
        count += 1

    df_cases = pd.DataFrame({
        "Subject_ID": subj_ids,
        "Group": subj_groups,
        "Outcome": subj_outcomes,
        "Outcome_Binary": subj_binary
    })

    results = {
        "table": table,
        "df_cases": df_cases,
        "group1_name": group1_name,
        "group2_name": group2_name,
        "outcome_pos_name": outcome_pos_name,
        "outcome_neg_name": outcome_neg_name,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "r1": r1,
        "r2": r2,
        "c1": c1,
        "c2": c2,
        "n_total": n_total,
        "chi2_stat": chi2_uncorr,
        "p_chi2": p_uncorr,
        "chi2_uncorr": chi2_uncorr,
        "p_uncorr": p_uncorr,
        "chi2_yates": chi2_yates,
        "p_yates": p_yates,
        "lr_g2": lr_g2,
        "p_lr": p_lr,
        "degrees_of_freedom": int(dof),
        "expected_frequencies": expected.tolist(),
        "expected_matrix": expected,
        "min_expected_count": min_exp_count,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": or_ci,
        "odds_ratio_fisher": odds_ratio_fisher,
        "p_fisher": p_fisher,
        "risk_group1": p1,
        "risk_group1_ci": p1_ci,
        "risk_group2": p2,
        "risk_group2_ci": p2_ci,
        "risk_diff": risk_diff,
        "risk_diff_ci": risk_diff_ci,
        "risk_ratio": risk_ratio,
        "risk_ratio_ci": rr_ci,
        "nnt": nnt,
        "phi": phi,
        "cramers_v": cramers_v,
        "cohens_w": cohens_w,
        "effect_magnitude": effect_mag,
        "cells_diagnostics": cells_diagnostics,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df_table, results


def generate_fisher_exact_demo(
    a: int = 4,
    b: int = 16,
    c: int = 1,
    d: int = 19,
    group1_name: str = "Experimental Treatment",
    group2_name: str = "Standard Control",
    outcome_pos_name: str = "Adverse Event (+)",
    outcome_neg_name: str = "No Event (-)",
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates 2x2 contingency table data for small sample sizes and calculates exact hypergeometric
    probabilities, two-sided and one-sided Fisher's Exact tests, odds ratio with confidence intervals,
    and the complete discrete hypergeometric probability distribution across all possible tables.

    Args:
        a: Cell (0, 0) - Group 1 with positive outcome.
        b: Cell (0, 1) - Group 1 with negative outcome.
        c: Cell (1, 0) - Group 2 with positive outcome.
        d: Cell (1, 1) - Group 2 with negative outcome.
        group1_name: Label for row 1.
        group2_name: Label for row 2.
        outcome_pos_name: Label for column 1.
        outcome_neg_name: Label for column 2.
        alpha: Type I error significance threshold (default 0.05).

    Returns:
        Tuple of (df_table, results_dict).
    """
    a = max(0, int(a))
    b = max(0, int(b))
    c = max(0, int(c))
    d = max(0, int(d))

    table = np.array([[a, b], [c, d]], dtype=int)

    r1 = a + b
    r2 = c + d
    c1 = a + c
    c2 = b + d
    n_total = r1 + r2

    # Expected frequencies: E_ij = (R_i * C_j) / N
    if n_total > 0:
        e11 = (r1 * c1) / n_total
        e12 = (r1 * c2) / n_total
        e21 = (r2 * c1) / n_total
        e22 = (r2 * c2) / n_total
    else:
        e11 = e12 = e21 = e22 = 0.0

    expected = np.array([[e11, e12], [e21, e22]], dtype=float)
    min_exp_count = float(np.min(expected)) if n_total > 0 else 0.0

    # 1. Hypergeometric Probability Distribution across All Possible Tables
    # Range of valid cell a counts given fixed margins:
    # max(0, C1 - R2) <= k <= min(R1, C1)
    k_min = max(0, c1 - r2)
    k_max = min(r1, c1)

    all_k = list(range(k_min, k_max + 1))
    dist_records = []
    cum_prob = 0.0

    # Probability of observed table:
    if n_total > 0 and c1 <= n_total and r1 <= n_total:
        p_observed = float(stats.hypergeom.pmf(a, n_total, c1, r1))
    else:
        p_observed = 1.0

    for k in all_k:
        prob_k = float(stats.hypergeom.pmf(k, n_total, c1, r1)) if n_total > 0 else 0.0
        cum_prob += prob_k
        is_obs = (k == a)
        # In Fisher's two-sided test, tables with prob <= p_observed (with numerical tolerance) are included
        is_two_sided = (prob_k <= p_observed + 1e-10)
        is_rt = (k >= a)
        is_lt = (k <= a)

        # Full 2x2 table configuration for this k:
        cell_b_k = r1 - k
        cell_c_k = c1 - k
        cell_d_k = r2 - cell_c_k

        dist_records.append({
            "k": k,
            "prob": prob_k,
            "cum_prob": min(1.0, cum_prob),
            "is_observed": is_obs,
            "is_two_sided_rejection": is_two_sided,
            "is_right_tail": is_rt,
            "is_left_tail": is_lt,
            "table_repr": f"[[{k}, {cell_b_k}], [{cell_c_k}, {cell_d_k}]]",
            "b_k": cell_b_k,
            "c_k": cell_c_k,
            "d_k": cell_d_k
        })

    # 2. Scipy Exact Calculations (Two-Sided and One-Sided)
    try:
        odds_ratio_fisher, p_two_sided = stats.fisher_exact(table, alternative="two-sided")
        _, p_greater = stats.fisher_exact(table, alternative="greater")
        _, p_less = stats.fisher_exact(table, alternative="less")
        odds_ratio_fisher = float(odds_ratio_fisher)
        p_two_sided = float(p_two_sided)
        p_greater = float(p_greater)
        p_less = float(p_less)
    except Exception:
        odds_ratio_fisher = 1.0
        p_two_sided = float(sum(d["prob"] for d in dist_records if d["is_two_sided_rejection"]))
        p_greater = float(sum(d["prob"] for d in dist_records if d["is_right_tail"]))
        p_less = float(sum(d["prob"] for d in dist_records if d["is_left_tail"]))

    # 3. Sample Odds Ratio & Confidence Interval
    if a > 0 and b > 0 and c > 0 and d > 0:
        sample_or = float((a * d) / (b * c))
        se_ln_or = float(np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d))
        or_ci = (float(np.exp(np.log(sample_or) - 1.96 * se_ln_or)),
                 float(np.exp(np.log(sample_or) + 1.96 * se_ln_or)))
    else:
        a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        sample_or = float((a_adj * d_adj) / (b_adj * c_adj))
        se_ln_or = float(np.sqrt(1.0 / a_adj + 1.0 / b_adj + 1.0 / c_adj + 1.0 / d_adj))
        or_ci = (float(np.exp(np.log(sample_or) - 1.96 * se_ln_or)),
                 float(np.exp(np.log(sample_or) + 1.96 * se_ln_or)))

    # 4. Proportions, Risk Difference, and Relative Risk
    p1 = float(a / r1) if r1 > 0 else 0.0
    p2 = float(c / r2) if r2 > 0 else 0.0

    se_p1 = float(np.sqrt(p1 * (1.0 - p1) / r1)) if r1 > 0 else 0.0
    se_p2 = float(np.sqrt(p2 * (1.0 - p2) / r2)) if r2 > 0 else 0.0

    p1_ci = (float(max(0.0, p1 - 1.96 * se_p1)), float(min(1.0, p1 + 1.96 * se_p1)))
    p2_ci = (float(max(0.0, p2 - 1.96 * se_p2)), float(min(1.0, p2 + 1.96 * se_p2)))

    risk_diff = float(p1 - p2)
    se_rd = float(np.sqrt(se_p1 ** 2 + se_p2 ** 2))
    risk_diff_ci = (float(risk_diff - 1.96 * se_rd), float(risk_diff + 1.96 * se_rd))
    nnt = float(1.0 / abs(risk_diff)) if abs(risk_diff) > 1e-6 else None

    # Relative Risk (Risk Ratio)
    if p1 > 0 and p2 > 0 and a > 0 and c > 0:
        risk_ratio = float(p1 / p2)
        se_ln_rr = float(np.sqrt((b / (a * r1)) + (d / (c * r2))))
        rr_ci = (float(np.exp(np.log(risk_ratio) - 1.96 * se_ln_rr)),
                 float(np.exp(np.log(risk_ratio) + 1.96 * se_ln_rr)))
    else:
        a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        r1_adj, r2_adj = a_adj + b_adj, c_adj + d_adj
        risk_ratio = float((a_adj / r1_adj) / (c_adj / r2_adj))
        se_ln_rr = float(np.sqrt((b_adj / (a_adj * r1_adj)) + (d_adj / (c_adj * r2_adj))))
        rr_ci = (float(np.exp(np.log(risk_ratio) - 1.96 * se_ln_rr)),
                 float(np.exp(np.log(risk_ratio) + 1.96 * se_ln_rr)))

    # 5. Contrast with Asymptotic Chi-Square & Yates
    if n_total > 0 and (r1 * r2 * c1 * c2) > 0:
        chi2_uncorr = float(np.sum((table - expected) ** 2 / expected))
        p_chi2_uncorr = float(stats.chi2.sf(chi2_uncorr, df=1))
    else:
        chi2_uncorr = 0.0
        p_chi2_uncorr = 1.0

    try:
        chi2_yates, p_chi2_yates, _, _ = stats.chi2_contingency(table, correction=True)
        chi2_yates = float(chi2_yates)
        p_chi2_yates = float(p_chi2_yates)
    except Exception:
        chi2_yates = chi2_uncorr
        p_chi2_yates = p_chi2_uncorr

    p_discrepancy = abs(p_chi2_uncorr - p_two_sided)

    # 6. Correlation / Association Strength
    phi = float(np.sqrt(chi2_uncorr / n_total)) if n_total > 0 else 0.0
    cramers_v = phi

    if cramers_v < 0.10:
        effect_mag = "Negligible Association"
    elif cramers_v < 0.30:
        effect_mag = "Small Association"
    elif cramers_v < 0.50:
        effect_mag = "Moderate Association"
    else:
        effect_mag = "Large Association"

    # 7. Methodological Assumptions Diagnostics
    assumptions_eval = [
        {
            "name": "Measurement Scale & Exclusivity",
            "verdict": "Satisfied (2×2 Cross-Classification)",
            "status": "pass",
            "diagnostic_value": "Binary Factors Cross-Classification",
            "description": "Requires two categorical variables arranged in a 2×2 contingency table where each observation belongs to exactly one cell.",
            "detail": f"Observed sample has {r1} subjects in '{group1_name}' and {r2} in '{group2_name}' across '{outcome_pos_name}' and '{outcome_neg_name}'."
        },
        {
            "name": "Independence of Observations",
            "verdict": "Satisfied (Independent Sampling)",
            "status": "pass",
            "diagnostic_value": "Independent Factor Design",
            "description": "Each individual observation must be independent of all others (no repeated measures or matching).",
            "detail": "Data represents cross-sectional or independent cohort samples. Paired/matched binary data requires McNemar's test."
        },
        {
            "name": "Small Sample & Sparse Cell Suitability",
            "verdict": "Optimal Method (Cochran Criterion Bypassed Exactly)",
            "status": "pass",
            "diagnostic_value": f"Minimum Expected Cell = {min_exp_count:.2f} (Chi-Square Benchmark: ≥ 5.0)",
            "description": "Unlike Pearson Chi-Square, Fisher's Exact Test does not rely on continuous asymptotic approximations and is mathematically exact for any cell frequency.",
            "detail": (
                f"Minimum expected cell count is {min_exp_count:.2f} (< 5.0). Pearson Chi-Square would be invalid here; Fisher's Exact Test is strictly required."
                if min_exp_count < 5.0 else
                f"All expected cell counts are ≥ 5.0 (min: {min_exp_count:.2f}). Both Fisher's Exact and Pearson Chi-Square are valid; Fisher's provides the exact benchmark."
            )
        },
        {
            "name": "Fixed Marginals Conditioning",
            "verdict": "Conditioned on Observed Margins",
            "status": "pass",
            "diagnostic_value": f"Margins: Rows ({r1}, {r2}) | Cols ({c1}, {c2})",
            "description": "Fisher's test conditions on fixed row and column marginal totals to evaluate the exact hypergeometric probability.",
            "detail": f"Under the null hypothesis, there are exactly {len(all_k)} possible tables that preserve these marginal sums."
        }
    ]

    # Overall Verdict & Recommendation
    overall_status = "pass"
    overall_verdict_badge = "✅ Statistically Valid: Fisher's Exact Test (Hypergeometric Exact)"

    if p_two_sided < alpha:
        overall_recommendation = (
            f"**Statistically Significant Association via Fisher's Exact Test (p = {p_two_sided:.4f} < 0.05).** "
            f"We reject the null hypothesis of independence. There is a genuine statistical association between "
            f"**{group1_name} vs. {group2_name}** and **{outcome_pos_name}**. "
            f"Specifically, {group1_name} exhibited an event rate of {p1*100:.1f}% vs. {p2*100:.1f}% in {group2_name} "
            f"(Exact Odds Ratio = {odds_ratio_fisher:.2f}, 95% CI: {or_ci[0]:.2f} – {or_ci[1]:.2f}; Risk Difference = {risk_diff*100:+.1f}%). "
            f"Because expected cell counts are small (min: {min_exp_count:.2f}), reporting Fisher's Exact Test guarantees exact Type I error control."
        )
    else:
        overall_recommendation = (
            f"**No Statistically Significant Association via Fisher's Exact Test (p = {p_two_sided:.4f} ≥ 0.05).** "
            f"We fail to reject the null hypothesis of independence. The observed difference in event proportions "
            f"({p1*100:.1f}% in {group1_name} vs. {p2*100:.1f}% in {group2_name}) is consistent with random chance sampling variability. "
            f"Exact Odds Ratio is {odds_ratio_fisher:.2f} (95% CI: {or_ci[0]:.2f} – {or_ci[1]:.2f}), encompassing 1.0 (no effect). "
            f"Right-tail (p = {p_greater:.4f}) and left-tail (p = {p_less:.4f}) probabilities are also non-significant."
        )

    # 8. DataFrames
    df_table = pd.DataFrame(
        table,
        index=[group1_name, group2_name],
        columns=[outcome_pos_name, outcome_neg_name]
    )

    subj_ids = []
    subj_groups = []
    subj_outcomes = []
    subj_binary = []

    count = 1
    for _ in range(a):
        subj_ids.append(f"SUBJ_{count:03d}")
        subj_groups.append(group1_name)
        subj_outcomes.append(outcome_pos_name)
        subj_binary.append(1)
        count += 1
    for _ in range(b):
        subj_ids.append(f"SUBJ_{count:03d}")
        subj_groups.append(group1_name)
        subj_outcomes.append(outcome_neg_name)
        subj_binary.append(0)
        count += 1
    for _ in range(c):
        subj_ids.append(f"SUBJ_{count:03d}")
        subj_groups.append(group2_name)
        subj_outcomes.append(outcome_pos_name)
        subj_binary.append(1)
        count += 1
    for _ in range(d):
        subj_ids.append(f"SUBJ_{count:03d}")
        subj_groups.append(group2_name)
        subj_outcomes.append(outcome_neg_name)
        subj_binary.append(0)
        count += 1

    df_cases = pd.DataFrame({
        "Subject_ID": subj_ids,
        "Group": subj_groups,
        "Outcome": subj_outcomes,
        "Outcome_Binary": subj_binary
    })

    results = {
        "table": table,
        "df_cases": df_cases,
        "group1_name": group1_name,
        "group2_name": group2_name,
        "outcome_pos_name": outcome_pos_name,
        "outcome_neg_name": outcome_neg_name,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "r1": r1,
        "r2": r2,
        "c1": c1,
        "c2": c2,
        "n_total": n_total,
        "k_min": k_min,
        "k_max": k_max,
        "p_observed": p_observed,
        "p_two_sided": p_two_sided,
        "p_fisher": p_two_sided,
        "p_greater": p_greater,
        "p_less": p_less,
        "odds_ratio": sample_or,
        "odds_ratio_ci": or_ci,
        "odds_ratio_fisher": odds_ratio_fisher,
        "risk_group1": p1,
        "risk_group1_ci": p1_ci,
        "risk_group2": p2,
        "risk_group2_ci": p2_ci,
        "risk_diff": risk_diff,
        "risk_diff_ci": risk_diff_ci,
        "risk_ratio": risk_ratio,
        "risk_ratio_ci": rr_ci,
        "nnt": nnt,
        "phi": phi,
        "cramers_v": cramers_v,
        "effect_magnitude": effect_mag,
        "min_expected_count": min_exp_count,
        "expected_matrix": expected,
        "chi2_uncorr": chi2_uncorr,
        "p_chi2_uncorr": p_chi2_uncorr,
        "chi2_yates": chi2_yates,
        "p_chi2_yates": p_chi2_yates,
        "p_discrepancy": p_discrepancy,
        "hypergeom_distribution": dist_records,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df_table, results


def generate_mcnemar_demo(
    a: int = 15,
    b: int = 5,
    c: int = 25,
    d: int = 55,
    time1_name: str = "Pre-Intervention (Time 1)",
    time2_name: str = "Post-Intervention (Time 2)",
    outcome_pos_name: str = "Positive / Success (+)",
    outcome_neg_name: str = "Negative / Failure (-)",
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates paired binary contingency data and computes McNemar test statistics,
    including exact binomial test, Edwards' continuity-corrected chi-square, uncorrected chi-square,
    marginal risk difference, conditional odds ratio, Cohen's kappa, and complete discordant binomial distribution.

    Args:
        a: Cell (0, 0) - Concordant positive at both occasions (Time 1 +, Time 2 +).
        b: Cell (0, 1) - Discordant (Time 1 +, Time 2 -) [Relapsed / Switched Negative].
        c: Cell (1, 0) - Discordant (Time 1 -, Time 2 +) [Improved / Switched Positive].
        d: Cell (1, 1) - Concordant negative at both occasions (Time 1 -, Time 2 -).
        time1_name: Label for row dimension / baseline / condition 1.
        time2_name: Label for column dimension / followup / condition 2.
        outcome_pos_name: Label for positive status.
        outcome_neg_name: Label for negative status.
        alpha: Type I error significance threshold (default 0.05).

    Returns:
        Tuple of (df_table, results_dict).
    """
    a = max(0, int(a))
    b = max(0, int(b))
    c = max(0, int(c))
    d = max(0, int(d))

    table = np.array([[a, b], [c, d]], dtype=int)

    r1 = a + b
    r2 = c + d
    c1 = a + c
    c2 = b + d
    n_total = a + b + c + d

    n_conc = a + d
    n_disc = b + c
    net_shift = c - b

    # Marginal event proportions
    p1 = float(r1 / n_total) if n_total > 0 else 0.0
    p2 = float(c1 / n_total) if n_total > 0 else 0.0

    # Marginal Risk Difference (Delta p = p2 - p1 = (c - b) / N)
    risk_diff = float((c - b) / n_total) if n_total > 0 else 0.0
    if n_total > 0:
        se_rd = float(np.sqrt(max(0.0, (n_disc - ((c - b) ** 2) / n_total))) / n_total)
    else:
        se_rd = 0.0
    risk_diff_ci = (float(max(-1.0, risk_diff - 1.96 * se_rd)),
                    float(min(1.0, risk_diff + 1.96 * se_rd)))

    # Marginal Risk Ratio (Relative Risk: p2 / p1 = (a + c) / (a + b))
    if r1 > 0 and c1 > 0:
        risk_ratio = float(c1 / r1)
        se_ln_rr = float(np.sqrt(n_disc / (c1 * r1))) if c1 > 0 and r1 > 0 else 0.0
        risk_ratio_ci = (float(np.exp(np.log(risk_ratio) - 1.96 * se_ln_rr)),
                         float(np.exp(np.log(risk_ratio) + 1.96 * se_ln_rr)))
    else:
        risk_ratio = 1.0
        risk_ratio_ci = (1.0, 1.0)

    # Number Needed to Treat (NNT = 1 / |risk_diff|)
    if abs(c - b) > 0 and n_total > 0:
        nnt = float(n_total / abs(c - b))
    else:
        nnt = None

    # Conditional Odds Ratio (OR_paired = c / b)
    # Reflects the odds of switching from negative to positive vs switching from positive to negative
    if b > 0 and c > 0:
        odds_ratio = float(c / b)
        se_ln_or = float(np.sqrt(1.0 / b + 1.0 / c))
        odds_ratio_ci = (float(np.exp(np.log(odds_ratio) - 1.96 * se_ln_or)),
                         float(np.exp(np.log(odds_ratio) + 1.96 * se_ln_or)))
    else:
        b_adj = b + 0.5
        c_adj = c + 0.5
        odds_ratio = float(c_adj / b_adj)
        se_ln_or = float(np.sqrt(1.0 / b_adj + 1.0 / c_adj))
        odds_ratio_ci = (float(np.exp(np.log(odds_ratio) - 1.96 * se_ln_or)),
                         float(np.exp(np.log(odds_ratio) + 1.96 * se_ln_or)))

    # Cohen's Kappa (chance-adjusted agreement)
    if n_total > 0:
        p_obs_agr = float(n_conc / n_total)
        p_exp_agr = float((r1 * c1 + r2 * c2) / (n_total ** 2))
        if p_exp_agr < 1.0:
            kappa = float((p_obs_agr - p_exp_agr) / (1.0 - p_exp_agr))
        else:
            kappa = 1.0
    else:
        p_obs_agr = 0.0
        p_exp_agr = 0.0
        kappa = 0.0

    # 1. Exact Binomial McNemar Test (Gold Standard, especially when b + c < 25)
    # Under H0, b ~ Binomial(n = b + c, p = 0.5)
    if n_disc > 0:
        k_min_disc = min(b, c)
        p_exact_two_sided = float(min(1.0, 2.0 * stats.binom.cdf(k_min_disc, n_disc, 0.5)))
        # Directional tests:
        p_exact_greater = float(1.0 - stats.binom.cdf(c - 1, n_disc, 0.5)) if c > 0 else 1.0
        p_exact_less = float(stats.binom.cdf(c, n_disc, 0.5))
    else:
        p_exact_two_sided = 1.0
        p_exact_greater = 1.0
        p_exact_less = 1.0

    # 2. Edwards' Continuity-Corrected McNemar Chi-Square Statistic:
    # chi2_edwards = (|b - c| - 1)^2 / (b + c)
    if n_disc > 0:
        diff_val = abs(b - c)
        diff_corr = max(0.0, float(diff_val - 1.0))
        chi2_edwards = float((diff_corr ** 2) / n_disc)
        p_edwards = float(1.0 - stats.chi2.cdf(chi2_edwards, df=1))
    else:
        chi2_edwards = 0.0
        p_edwards = 1.0

    # 3. Uncorrected McNemar Chi-Square Statistic:
    # chi2_uncorr = (b - c)^2 / (b + c)
    if n_disc > 0:
        chi2_uncorr = float(((b - c) ** 2) / n_disc)
        p_uncorr = float(1.0 - stats.chi2.cdf(chi2_uncorr, df=1))
    else:
        chi2_uncorr = 0.0
        p_uncorr = 1.0

    # 4. Binomial Distribution of All Possible Discordant Splits
    binom_dist_records = []
    cum_prob_binom = 0.0
    for k in range(n_disc + 1):
        prob_k = float(stats.binom.pmf(k, n_disc, 0.5)) if n_disc > 0 else 1.0
        cum_prob_binom += prob_k
        is_obs = (k == c)
        is_two_sided = (k <= min(b, c) or k >= max(b, c)) if n_disc > 0 else True
        is_rt = (k >= c)
        is_lt = (k <= c)

        binom_dist_records.append({
            "k": k,
            "b_k": n_disc - k,
            "c_k": k,
            "prob": prob_k,
            "cum_prob": min(1.0, cum_prob_binom),
            "is_observed": is_obs,
            "is_two_sided_rejection": is_two_sided,
            "is_right_tail": is_rt,
            "is_left_tail": is_lt,
            "split_repr": f"b = {n_disc - k}, c = {k}"
        })

    # 5. Subject-Level Paired Cases Dataframe
    subj_ids = []
    time1_statuses = []
    time2_statuses = []
    transition_types = []
    status_changed = []
    count = 1

    # Cell a: + / + (Concordant Positive)
    for _ in range(a):
        subj_ids.append(f"SUBJ_{count:03d}")
        time1_statuses.append(outcome_pos_name)
        time2_statuses.append(outcome_pos_name)
        transition_types.append("Concordant (+/+): Persistent Positive")
        status_changed.append("No")
        count += 1

    # Cell b: + / - (Discordant: Relapsed / Switched Negative)
    for _ in range(b):
        subj_ids.append(f"SUBJ_{count:03d}")
        time1_statuses.append(outcome_pos_name)
        time2_statuses.append(outcome_neg_name)
        transition_types.append(f"Discordant (+/-): Switched {outcome_neg_name}")
        status_changed.append("Yes")
        count += 1

    # Cell c: - / + (Discordant: Improved / Switched Positive)
    for _ in range(c):
        subj_ids.append(f"SUBJ_{count:03d}")
        time1_statuses.append(outcome_neg_name)
        time2_statuses.append(outcome_pos_name)
        transition_types.append(f"Discordant (-/+): Switched {outcome_pos_name}")
        status_changed.append("Yes")
        count += 1

    # Cell d: - / - (Concordant Negative)
    for _ in range(d):
        subj_ids.append(f"SUBJ_{count:03d}")
        time1_statuses.append(outcome_neg_name)
        time2_statuses.append(outcome_neg_name)
        transition_types.append("Concordant (-/-): Persistent Negative")
        status_changed.append("No")
        count += 1

    df_cases = pd.DataFrame({
        "Subject_ID": subj_ids,
        "Time_1_Status": time1_statuses,
        "Time_2_Status": time2_statuses,
        "Transition_Type": transition_types,
        "Status_Changed": status_changed
    })

    # Summary table dataframe for UI display
    df_table = pd.DataFrame(
        table,
        index=[f"{time1_name}: {outcome_pos_name}", f"{time1_name}: {outcome_neg_name}"],
        columns=[f"{time2_name}: {outcome_pos_name}", f"{time2_name}: {outcome_neg_name}"]
    )

    primary_p = p_exact_two_sided if n_disc < 25 else p_edwards
    primary_method = "Exact Binomial Test" if n_disc < 25 else "Edwards' Continuity-Corrected Chi-Square"
    is_significant = (primary_p < alpha)

    # Assumptions Diagnostics
    assumptions_eval = [
        {
            "name": "Paired / Repeated Dependency Structure",
            "status": "pass",
            "verdict": "Paired Data Confirmed",
            "diagnostic_value": f"N = {n_total} paired observations ({time1_name} vs {time2_name})",
            "description": "McNemar's test requires paired data (same individuals or matched pairs).",
            "detail": f"All {n_total} records represent paired observations across {time1_name} and {time2_name}. Concordance and discordance are properly evaluated within each subject."
        },
        {
            "name": "Discordant Pair Sample Size (b + c ≥ 25)",
            "status": "pass" if n_disc >= 25 else "warning",
            "verdict": "Adequate Discordant Sample (≥ 25)" if n_disc >= 25 else "Small Discordant Sample (< 25)",
            "diagnostic_value": f"b + c = {n_disc} discordant pairs ({'≥ 25' if n_disc >= 25 else '< 25'})",
            "description": "Continuous Chi-Square requires at least 25 discordant pairs (b + c ≥ 25).",
            "detail": (
                f"With b + c = {n_disc} (≥ 25), the continuous chi-square approximation with Edwards' continuity correction is valid and well-calibrated."
                if n_disc >= 25 else
                f"With b + c = {n_disc} (< 25), the continuous approximation suffers from inflated error. The Exact Binomial McNemar Test (p = {p_exact_two_sided:.4f}) is strictly mandated."
            )
        },
        {
            "name": "Dichotomous Categorical Scale",
            "status": "pass",
            "verdict": "Binary 2-State Outcome",
            "diagnostic_value": f"2 mutually exclusive states: '{outcome_pos_name}' vs '{outcome_neg_name}'",
            "description": "Outcomes must have exactly two mutually exclusive categories at each measurement point.",
            "detail": "Data satisfies the binary classification requirement. If evaluating 3 or more outcome categories, Stuart-Maxwell or Bowker's test of symmetry would be required."
        },
        {
            "name": "Directional Transition Sparsity",
            "status": "pass" if (b > 0 and c > 0) else "warning",
            "verdict": "Bidirectional Transitions" if (b > 0 and c > 0) else "Zero Discordant Cell",
            "diagnostic_value": f"b = {b} (+ to -), c = {c} (- to +)",
            "description": "Checks whether transitions occurred in both directions without zero cells.",
            "detail": (
                f"Transitions occurred in both directions with a net difference of {net_shift:+d} pairs towards {outcome_pos_name}."
                if (b > 0 and c > 0) else
                f"One discordant cell contains 0 observations (b={b}, c={c}). The conditional odds ratio uses a Haldane-Anscombe (+0.5) adjustment. Exact binomial p-value remains exact and unaffected."
            )
        }
    ]

    p1_pct = p1 * 100.0
    p2_pct = p2 * 100.0
    rd_pct = risk_diff * 100.0

    if is_significant:
        overall_status = "pass"
        overall_verdict_badge = f"✅ Statistically Significant Proportion Change ({primary_method}: p = {primary_p:.4f} < {alpha})"
        overall_recommendation = (
            f"**Reject the null hypothesis of marginal homogeneity (p = {primary_p:.4f} < {alpha}).** "
            f"There is a statistically significant change in the proportion of participants experiencing **{outcome_pos_name}** "
            f"between **{time1_name}** ({p1_pct:.1f}%) and **{time2_name}** ({p2_pct:.1f}%), representing an absolute shift of **{rd_pct:+.1f}%** "
            f"(95% CI: {risk_diff_ci[0]*100:+.1f}% to {risk_diff_ci[1]*100:+.1f}%). "
            f"The conditional odds ratio of switching is **{odds_ratio:.2f}** (95% CI: {odds_ratio_ci[0]:.2f} – {odds_ratio_ci[1]:.2f}). "
            f"{'Because discordant pairs were < 25, report the Exact Binomial McNemar Test.' if n_disc < 25 else 'Report Edwards continuity-corrected chi-square (χ² = ' + f'{chi2_edwards:.3f}).'}"
        )
    else:
        overall_status = "info"
        overall_verdict_badge = f"🟡 No Statistically Significant Proportion Change ({primary_method}: p = {primary_p:.4f} ≥ {alpha})"
        overall_recommendation = (
            f"**Fail to reject the null hypothesis of marginal homogeneity (p = {primary_p:.4f} ≥ {alpha}).** "
            f"The proportion of participants experiencing **{outcome_pos_name}** was {p1_pct:.1f}% at **{time1_name}** and {p2_pct:.1f}% at **{time2_name}** "
            f"(absolute difference: {rd_pct:+.1f}%, 95% CI: {risk_diff_ci[0]*100:+.1f}% to {risk_diff_ci[1]*100:+.1f}%). "
            f"The observed switches (b = {b} vs c = {c}) are consistent with random chance directional symmetry."
        )

    results = {
        "table": table,
        "df_cases": df_cases,
        "time1_name": time1_name,
        "time2_name": time2_name,
        "outcome_pos_name": outcome_pos_name,
        "outcome_neg_name": outcome_neg_name,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "r1": r1,
        "r2": r2,
        "c1": c1,
        "c2": c2,
        "n_total": n_total,
        "n_conc": n_conc,
        "n_disc": n_disc,
        "net_shift": net_shift,
        "p1": p1,
        "p2": p2,
        "risk_diff": risk_diff,
        "risk_diff_ci": risk_diff_ci,
        "risk_ratio": risk_ratio,
        "risk_ratio_ci": risk_ratio_ci,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": odds_ratio_ci,
        "nnt": nnt,
        "kappa": kappa,
        "p_obs_agr": p_obs_agr,
        "p_exp_agr": p_exp_agr,
        "p_exact_two_sided": p_exact_two_sided,
        "p_exact_greater": p_exact_greater,
        "p_exact_less": p_exact_less,
        "chi2_edwards": chi2_edwards,
        "p_edwards": p_edwards,
        "chi2_uncorr": chi2_uncorr,
        "p_uncorr": p_uncorr,
        "primary_p": primary_p,
        "primary_method": primary_method,
        "binom_distribution": binom_dist_records,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df_table, results


def generate_pearson_correlation_demo(
    n_pts: int = 60,
    true_r: float = 0.82,
    var_x_name: str = "Study Time (Hours / Week)",
    var_y_name: str = "Exam Score (%)",
    scenario: str = "normal_linear",
    noise_level: float = 1.0,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates bivariate synthetic data and calculates comprehensive Pearson correlation metrics,
    coefficient of determination (R-squared), Fisher's z confidence intervals, OLS regression parameters,
    variance partitioning, and rigorous assumption diagnostics.

    Args:
        n_pts: Number of observation pairs (N).
        true_r: Underlying population correlation coefficient.
        var_x_name: Label for predictor / independent variable X.
        var_y_name: Label for outcome / dependent variable Y.
        scenario: Preset scenario ('normal_linear', 'moderate_negative', 'null_correlation',
                  'non_linear_curvature', 'outlier_leverage', 'custom').
        noise_level: Multiplier for residual error variance.
        alpha: Significance threshold (default 0.05).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (demo_df, results_dict).
    """
    n_pts = max(10, int(n_pts))
    rng = np.random.default_rng(seed)

    if scenario == "normal_linear":
        target_r = float(np.clip(true_r, -0.99, 0.99))
        cov = [[1.0, target_r], [target_r, 1.0]]
        pts = rng.multivariate_normal([0.0, 0.0], cov, size=n_pts)
        x = pts[:, 0] * 6.0 + 25.0
        extra_noise = rng.normal(0.0, 4.0 * max(0.0, noise_level - 0.8), size=n_pts) if noise_level > 0.8 else 0.0
        y = pts[:, 1] * 12.0 + 75.0 + extra_noise
        x = np.clip(x, 2.0, 50.0)
        y = np.clip(y, 10.0, 100.0)
    elif scenario == "moderate_negative":
        target_r = -0.58
        cov = [[1.0, target_r], [target_r, 1.0]]
        pts = rng.multivariate_normal([0.0, 0.0], cov, size=n_pts)
        x = pts[:, 0] * 2.0 + 6.5
        extra_noise = rng.normal(0.0, 0.8 * max(0.0, noise_level - 0.8), size=n_pts) if noise_level > 0.8 else 0.0
        y = pts[:, 1] * 1.5 + 7.0 + extra_noise
        x = np.clip(x, 1.0, 15.0)
        y = np.clip(y, 3.0, 11.0)
    elif scenario == "null_correlation":
        cov = [[1.0, 0.0], [0.0, 1.0]]
        pts = rng.multivariate_normal([0.0, 0.0], cov, size=n_pts)
        x = pts[:, 0] * 2.5 + 9.5
        y = pts[:, 1] * 15.0 + 100.0
    elif scenario == "non_linear_curvature":
        x = np.linspace(5.0, 60.0, n_pts)
        y = 20.0 + 22.0 * np.log(x) + rng.normal(0.0, 5.0 * noise_level, size=n_pts)
    elif scenario == "outlier_leverage":
        n_main = max(8, n_pts - 2)
        pts = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.05], [0.05, 1.0]], size=n_main)
        x_cluster = pts[:, 0] * 3.0 + 20.0
        y_cluster = pts[:, 1] * 4.0 + 50.0
        x = np.append(x_cluster, [55.0, 58.0])
        y = np.append(y_cluster, [92.0, 96.0])
    else:  # custom
        target_r = float(np.clip(true_r, -0.99, 0.99))
        cov = [[1.0, target_r], [target_r, 1.0]]
        pts = rng.multivariate_normal([0.0, 0.0], cov, size=n_pts)
        x = pts[:, 0] * 10.0 + 50.0
        noise_sd = 8.0 * max(0.0, noise_level - 0.5)
        y = pts[:, 1] * 15.0 + 100.0 + (rng.normal(0.0, noise_sd, size=n_pts) if noise_sd > 0 else 0.0)

    n = len(x)
    df = n - 2

    # Descriptive Statistics for X and Y
    mean_x = float(np.mean(x))
    std_x = float(np.std(x, ddof=1)) if n > 1 else 0.0
    var_x = float(np.var(x, ddof=1)) if n > 1 else 0.0
    min_x = float(np.min(x))
    max_x = float(np.max(x))

    mean_y = float(np.mean(y))
    std_y = float(np.std(y, ddof=1)) if n > 1 else 0.0
    var_y = float(np.var(y, ddof=1)) if n > 1 else 0.0
    min_y = float(np.min(y))
    max_y = float(np.max(y))

    # Covariance
    if n > 1:
        cov_matrix = np.cov(x, y)
        covariance = float(cov_matrix[0, 1])
    else:
        covariance = 0.0

    # Pearson Correlation Coefficient r and p-value
    r_val, p_val = stats.pearsonr(x, y)
    r_val = float(r_val)
    p_val = float(p_val)

    # Coefficient of Determination (R^2)
    r_squared = float(r_val ** 2)
    r_squared_pct = r_squared * 100.0
    unexplained_pct = (1.0 - r_squared) * 100.0

    if df > 0:
        adj_r_squared = float(1.0 - (1.0 - r_squared) * ((n - 1) / df))
    else:
        adj_r_squared = r_squared

    # Standard Error of r and t-statistic
    if df > 0 and abs(r_val) < 1.0:
        se_r = float(np.sqrt((1.0 - r_squared) / df))
        t_stat = float(r_val * np.sqrt(df) / np.sqrt(max(1e-12, 1.0 - r_squared)))
    else:
        se_r = 0.0
        t_stat = 0.0

    # Fisher's z-transformation 95% Confidence Interval for r
    if n > 3:
        r_clamped = float(np.clip(r_val, -0.99999, 0.99999))
        z_score = float(np.arctanh(r_clamped))
        se_z = float(1.0 / np.sqrt(n - 3))
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_low = z_score - z_crit * se_z
        z_high = z_score + z_crit * se_z
        r_ci = (float(np.tanh(z_low)), float(np.tanh(z_high)))
    else:
        z_score = 0.0
        se_z = 0.0
        r_ci = (-1.0, 1.0)

    # OLS Simple Linear Regression: Y = Intercept + Slope * X
    slope, intercept, r_lin, p_reg, std_err_slope = stats.linregress(x, y)
    slope = float(slope)
    intercept = float(intercept)
    std_err_slope = float(std_err_slope)

    y_fitted = intercept + slope * x
    residuals = y - y_fitted

    ss_tot = float(np.sum((y - mean_y) ** 2))
    ss_reg = float(np.sum((y_fitted - mean_y) ** 2))
    ss_res = float(np.sum(residuals ** 2))

    rmse = float(np.sqrt(ss_res / df)) if df > 0 else 0.0

    sum_x_dev2 = float(np.sum((x - mean_x) ** 2))
    if n > 0 and sum_x_dev2 > 0:
        se_intercept = float(rmse * np.sqrt(1.0 / n + (mean_x ** 2) / sum_x_dev2))
    else:
        se_intercept = 0.0

    t_intercept = float(intercept / se_intercept) if se_intercept > 0 else 0.0
    p_intercept = float(2.0 * (1.0 - stats.t.cdf(abs(t_intercept), df=df))) if df > 0 else 1.0

    t_crit_df = float(stats.t.ppf(1.0 - alpha / 2.0, df=df)) if df > 0 else 1.96
    slope_ci = (float(slope - t_crit_df * std_err_slope), float(slope + t_crit_df * std_err_slope))
    intercept_ci = (float(intercept - t_crit_df * se_intercept), float(intercept + t_crit_df * se_intercept))

    # Regression F-statistic
    ms_reg = ss_reg
    ms_res = ss_res / df if df > 0 else 1.0
    f_stat = float(ms_reg / ms_res) if ms_res > 0 else 0.0
    p_f = float(1.0 - stats.f.cdf(f_stat, 1, df)) if df > 0 else 1.0

    # Spearman Rank Correlation for Non-Parametric Comparison
    rho_spearman, p_spearman = stats.spearmanr(x, y)
    rho_spearman = float(rho_spearman)
    p_spearman = float(p_spearman)

    # Leverage, Standardized Residuals, and Cook's Distance
    if sum_x_dev2 > 0:
        leverage = 1.0 / n + ((x - mean_x) ** 2) / sum_x_dev2
    else:
        leverage = np.full(n, 1.0 / n)

    denom_std_res = np.sqrt(np.maximum(1e-10, 1.0 - leverage))
    std_residuals = residuals / (rmse * denom_std_res) if rmse > 0 else np.zeros(n)

    if rmse > 0:
        cooks_d = ((residuals ** 2) / (2.0 * (rmse ** 2))) * (leverage / (np.maximum(1e-10, 1.0 - leverage) ** 2))
    else:
        cooks_d = np.zeros(n)

    cooks_threshold = float(4.0 / n)
    high_cooks_count = int(np.sum(cooks_d > cooks_threshold))
    extreme_outliers_count = int(np.sum(np.abs(std_residuals) > 3.0))

    # Regression Bands for Plotting (95% Confidence & Prediction Intervals)
    x_grid = np.linspace(min_x, max_x, 100)
    y_grid_fit = intercept + slope * x_grid
    if sum_x_dev2 > 0:
        se_ci_band = rmse * np.sqrt(1.0 / n + ((x_grid - mean_x) ** 2) / sum_x_dev2)
        se_pi_band = rmse * np.sqrt(1.0 + 1.0 / n + ((x_grid - mean_x) ** 2) / sum_x_dev2)
    else:
        se_ci_band = np.zeros(100)
        se_pi_band = np.zeros(100)

    ci_upper_band = y_grid_fit + t_crit_df * se_ci_band
    ci_lower_band = y_grid_fit - t_crit_df * se_ci_band
    pi_upper_band = y_grid_fit + t_crit_df * se_pi_band
    pi_lower_band = y_grid_fit - t_crit_df * se_pi_band

    # Assumption Tests
    shapiro_x = stats.shapiro(x) if n >= 3 else (1.0, 1.0)
    shapiro_y = stats.shapiro(y) if n >= 3 else (1.0, 1.0)
    shapiro_res = stats.shapiro(residuals) if n >= 3 else (1.0, 1.0)
    p_norm_x = float(shapiro_x.pvalue)
    p_norm_y = float(shapiro_y.pvalue)
    p_norm_res = float(shapiro_res.pvalue)
    is_normal = (p_norm_x >= 0.05 and p_norm_y >= 0.05 and p_norm_res >= 0.05)

    poly_coeffs = np.polyfit(x, y, 2)
    y_poly_fit = np.polyval(poly_coeffs, x)
    ss_res_poly = float(np.sum((y - y_poly_fit) ** 2))
    df_poly = n - 3
    if df_poly > 0 and ss_res > ss_res_poly:
        f_poly = ((ss_res - ss_res_poly) / 1.0) / (ss_res_poly / df_poly)
        p_curvature = float(1.0 - stats.f.cdf(f_poly, 1, df_poly))
    else:
        f_poly = 0.0
        p_curvature = 1.0
    is_linear = (p_curvature >= 0.05)

    abs_res = np.abs(residuals)
    r_hetero, p_hetero = stats.spearmanr(y_fitted, abs_res)
    p_hetero = float(p_hetero)
    is_homoscedastic = (p_hetero >= 0.05)

    has_extreme_outliers = (extreme_outliers_count > 0 or float(np.max(cooks_d)) > 0.5)

    # Assumptions Diagnostics Cards
    assumptions_eval = [
        {
            "name": "Continuous Measurement Scale",
            "status": "pass",
            "verdict": "Interval / Ratio Metric Scale",
            "diagnostic_value": f"N = {n} continuous paired records",
            "description": "Both X and Y must be continuous numeric variables.",
            "detail": f"Both '{var_x_name}' and '{var_y_name}' are interval/ratio continuous variables with valid variance (s_X = {std_x:.2f}, s_Y = {std_y:.2f})."
        },
        {
            "name": "Bivariate & Residual Normality",
            "status": "pass" if is_normal else "warning",
            "verdict": "Normality Confirmed" if is_normal else "Distributional Departure",
            "diagnostic_value": f"Shapiro-Wilk: X (p={p_norm_x:.4f}), Y (p={p_norm_y:.4f}), Residuals (p={p_norm_res:.4f})",
            "description": "Bivariate normal distribution ensures valid inference for Pearson's r.",
            "detail": (
                "Both variables and regression residuals are normally distributed (p ≥ 0.05), confirming classical inferential test validity."
                if is_normal else
                "One or more variables depart from normality (p < 0.05). In large samples Pearson's r is robust via Central Limit Theorem; for severe departures compare with Spearman's rho."
            )
        },
        {
            "name": "Linearity of Relationship",
            "status": "pass" if is_linear else "warning",
            "verdict": "Linear Relationship Met" if is_linear else "Non-Linear Curvature Detected",
            "diagnostic_value": f"Polynomial Curvature Test: F = {f_poly:.2f} (p = {p_curvature:.4f})",
            "description": "Pearson's r strictly measures linear (straight-line) association.",
            "detail": (
                "No statistically significant quadratic curvature detected (p ≥ 0.05). Straight-line model represents the relationship well."
                if is_linear else
                f"Significant non-linear curvature detected (p = {p_curvature:.4f} < 0.05). Pearson's r ({r_val:.3f}) understates true association. Spearman's rho ({rho_spearman:.3f}) is more appropriate."
            )
        },
        {
            "name": "Homoscedasticity & Influential Points",
            "status": "pass" if (is_homoscedastic and not has_extreme_outliers) else "warning",
            "verdict": "Homoscedastic, No Extreme Outliers" if (is_homoscedastic and not has_extreme_outliers) else "Outlier / Heteroscedastic Warning",
            "diagnostic_value": f"Max Cook's D = {float(np.max(cooks_d)):.3f}, Residual-Fit p = {p_hetero:.4f}",
            "description": "Residuals must exhibit uniform spread without disproportionate leverage.",
            "detail": (
                "Residual variance is constant along the line (p ≥ 0.05) and no points exert disproportionate leverage (max Cook's D < 0.5)."
                if (is_homoscedastic and not has_extreme_outliers) else
                f"Detected potential heteroscedasticity or high-leverage outliers (Max Cook's D = {float(np.max(cooks_d)):.3f}, {extreme_outliers_count} points with |std_res| > 3.0). Check scatterplot."
            )
        }
    ]

    abs_r = abs(r_val)
    if abs_r < 0.10:
        effect_label = "Negligible / Trivial"
    elif abs_r < 0.30:
        effect_label = "Small / Weak"
    elif abs_r < 0.50:
        effect_label = "Medium / Moderate"
    elif abs_r < 0.70:
        effect_label = "Large / Strong"
    else:
        effect_label = "Very Strong / Substantial"

    direction_label = "Positive" if r_val > 0 else ("Negative" if r_val < 0 else "Zero")
    is_significant = (p_val < alpha)

    subj_ids = [f"SUBJ_{i+1:03d}" for i in range(n)]
    df_cases = pd.DataFrame({
        "Subject_ID": subj_ids,
        "X_Value": x,
        "Y_Value": y,
        "Fitted_Y": y_fitted,
        "Residual": residuals,
        "Standardized_Residual": std_residuals,
        "Cooks_Distance": cooks_d,
        "Is_Influential": (cooks_d > cooks_threshold)
    })

    demo_df = pd.DataFrame({
        "Predictor_X": x,
        "Outcome_Y": y
    })

    if is_significant:
        overall_status = "pass"
        overall_verdict_badge = f"✅ Statistically Significant Linear Correlation (r = {r_val:.3f}, R² = {r_squared_pct:.1f}%, p = {p_val:.4e} < {alpha})"
        overall_recommendation = (
            f"**Reject the null hypothesis of zero correlation (p = {p_val:.4e} < {alpha}).** "
            f"There is a statistically significant {direction_label.lower()} linear correlation between **{var_x_name}** and **{var_y_name}** "
            f"(Pearson's r = **{r_val:.3f}**, 95% CI: [{r_ci[0]:.3f}, {r_ci[1]:.3f}], t({df}) = {t_stat:.3f}). "
            f"The linear model accounts for **{r_squared_pct:.1f}%** of the total variance in {var_y_name} (R² = {r_squared:.3f}). "
            f"{'Because non-linear curvature was detected, consider reporting Spearman rho as well.' if not is_linear else 'Parametric linear assumptions are satisfied.'}"
        )
    else:
        overall_status = "info"
        overall_verdict_badge = f"🟡 No Statistically Significant Linear Correlation (r = {r_val:.3f}, R² = {r_squared_pct:.1f}%, p = {p_val:.4f} ≥ {alpha})"
        overall_recommendation = (
            f"**Fail to reject the null hypothesis of zero correlation (p = {p_val:.4f} ≥ {alpha}).** "
            f"There is no statistically significant linear association between **{var_x_name}** and **{var_y_name}** "
            f"(Pearson's r = {r_val:.3f}, 95% CI: [{r_ci[0]:.3f}, {r_ci[1]:.3f}], t({df}) = {t_stat:.3f}). "
            f"The linear relationship accounts for only **{r_squared_pct:.1f}%** of outcome variance (R² = {r_squared:.3f})."
        )

    results = {
        "x": x,
        "y": y,
        "df_cases": df_cases,
        "var_x_name": var_x_name,
        "var_y_name": var_y_name,
        "n_pts": n,
        "df": df,
        "mean_x": mean_x,
        "std_x": std_x,
        "var_x": var_x,
        "min_x": min_x,
        "max_x": max_x,
        "mean_y": mean_y,
        "std_y": std_y,
        "var_y": var_y,
        "min_y": min_y,
        "max_y": max_y,
        "covariance": covariance,
        "pearson_r": r_val,
        "pearson_p": p_val,
        "r_squared": r_squared,
        "r_squared_pct": r_squared_pct,
        "adj_r_squared": adj_r_squared,
        "unexplained_pct": unexplained_pct,
        "se_r": se_r,
        "t_stat": t_stat,
        "fisher_z": z_score,
        "se_z": se_z,
        "r_ci": r_ci,
        "slope": slope,
        "intercept": intercept,
        "std_err_slope": std_err_slope,
        "se_intercept": se_intercept,
        "t_intercept": t_intercept,
        "p_intercept": p_intercept,
        "slope_ci": slope_ci,
        "intercept_ci": intercept_ci,
        "rmse": rmse,
        "ss_tot": ss_tot,
        "ss_reg": ss_reg,
        "ss_res": ss_res,
        "f_stat": f_stat,
        "p_f": p_f,
        "spearman_rho": rho_spearman,
        "spearman_p": p_spearman,
        "x_grid": x_grid,
        "y_grid_fit": y_grid_fit,
        "ci_lower_band": ci_lower_band,
        "ci_upper_band": ci_upper_band,
        "pi_lower_band": pi_lower_band,
        "pi_upper_band": pi_upper_band,
        "y_fitted": y_fitted,
        "residuals": residuals,
        "std_residuals": std_residuals,
        "cooks_d": cooks_d,
        "max_cooks_d": float(np.max(cooks_d)),
        "high_cooks_count": high_cooks_count,
        "effect_label": effect_label,
        "direction_label": direction_label,
        "is_normal": is_normal,
        "p_norm_x": p_norm_x,
        "p_norm_y": p_norm_y,
        "p_norm_res": p_norm_res,
        "is_linear": is_linear,
        "p_curvature": p_curvature,
        "is_homoscedastic": is_homoscedastic,
        "p_hetero": p_hetero,
        "has_extreme_outliers": has_extreme_outliers,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return demo_df, results


def generate_correlation_demo(
    n_pts: int = 60,
    true_r: float = 0.65,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Backward-compatible alias for generate_pearson_correlation_demo."""
    return generate_pearson_correlation_demo(n_pts=n_pts, true_r=true_r, seed=seed)


def generate_spearman_correlation_demo(
    n_pts: int = 50,
    target_rho: float = 0.85,
    var_x_name: str = "Cumulative Practice (Hours)",
    var_y_name: str = "Skill Proficiency Level",
    scenario: str = "non_linear_monotonic",
    tie_frequency: float = 0.0,
    noise_level: float = 1.0,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates bivariate synthetic data and calculates comprehensive Spearman rank correlation metrics,
    rank transformations, squared difference summations, tie corrections, Fisher's z confidence intervals,
    comparisons with Pearson's r and Kendall's tau, and assumption diagnostics.

    Args:
        n_pts: Number of observation pairs (N).
        target_rho: Underlying target rank correlation coefficient (-0.99 to +0.99).
        var_x_name: Label for predictor / independent variable X.
        var_y_name: Label for outcome / dependent variable Y.
        scenario: Preset scenario ('non_linear_monotonic', 'ordinal_likert', 'inverse_monotonic',
                  'extreme_outliers', 'null_rank', 'custom').
        tie_frequency: Proportion of artificial ties to inject in custom scenarios (0.0 to 0.5).
        noise_level: Multiplier for residual error variance.
        alpha: Significance threshold (default 0.05).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (demo_df, results_dict).
    """
    n_pts = max(10, int(n_pts))
    rng = np.random.default_rng(seed)

    if scenario == "non_linear_monotonic":
        # Diminishing returns curve: monotonic but non-linear (logarithmic growth)
        # Spearman rho is very high (~0.90+), while Pearson r is noticeably lower (~0.65-0.75)
        u = np.linspace(1.5, 30.0, n_pts) + rng.uniform(-0.4, 0.4, size=n_pts)
        u = np.sort(np.maximum(1.0, u))
        x = u
        y = 18.0 + 26.0 * np.log(x) + rng.normal(0.0, 7.0 * noise_level, size=n_pts)
    elif scenario == "ordinal_likert":
        # Discrete Likert ratings (1 to 7) with frequent ties
        cov = [[1.0, 0.75], [0.75, 1.0]]
        pts = rng.multivariate_normal([0.0, 0.0], cov, size=n_pts)
        if noise_level > 1.0:
            pts[:, 1] += rng.normal(0.0, 0.5 * (noise_level - 1.0), size=n_pts)
        bins = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]
        x = (np.digitize(pts[:, 0], bins) + 1).astype(float)
        y = (np.digitize(pts[:, 1], bins) + 1).astype(float)
    elif scenario == "inverse_monotonic":
        # Non-linear exponential decay: inverse monotonic relationship
        u = np.linspace(1.0, 16.0, n_pts) + rng.uniform(-0.3, 0.3, size=n_pts)
        u = np.sort(np.maximum(0.5, u))
        x = u
        y = 95.0 * np.exp(-0.25 * x) + 8.0 + rng.normal(0.0, 7.0 * noise_level, size=n_pts)
    elif scenario == "extreme_outliers":
        # Near-linear correlation with 2 severe off-trend leverage outliers that wreck Pearson r
        n_main = max(8, n_pts - 2)
        cov = [[1.0, 0.78], [0.78, 1.0]]
        pts = rng.multivariate_normal([0.0, 0.0], cov, size=n_main)
        x_main = pts[:, 0] * 5.0 + 25.0
        extra_noise = rng.normal(0.0, 3.0 * max(0.0, noise_level - 1.0), size=n_main) if noise_level > 1.0 else 0.0
        y_main = pts[:, 1] * 8.0 + 50.0 + extra_noise
        # 2 severe outliers placed diametrically off the trend line
        x_outliers = np.array([45.0, 8.0])
        y_outliers = np.array([15.0, 85.0])
        x = np.concatenate([x_main, x_outliers])
        y = np.concatenate([y_main, y_outliers])
    elif scenario == "null_rank":
        x = rng.uniform(10.0, 50.0, size=n_pts)
        y = rng.uniform(20.0, 90.0, size=n_pts)
    else:
        # Custom scenario
        target_r = float(np.clip(target_rho, -0.99, 0.99))
        cov = [[1.0, target_r], [target_r, 1.0]]
        pts = rng.multivariate_normal([0.0, 0.0], cov, size=n_pts)
        x = pts[:, 0] * 5.0 + 25.0
        extra_noise = rng.normal(0.0, 5.0 * max(0.0, noise_level - 0.5), size=n_pts)
        y = pts[:, 1] * 8.0 + 50.0 + extra_noise
        if tie_frequency > 0.0:
            round_digits = 0 if tie_frequency >= 0.25 else 1
            x = np.round(x, decimals=round_digits)
            y = np.round(y, decimals=round_digits)

    # 1. Compute Fractional Average Ranks
    rank_x = stats.rankdata(x, method="average")
    rank_y = stats.rankdata(y, method="average")

    # 2. Differences in Ranks (d_i) and Squared Differences
    d_i = rank_x - rank_y
    d_squared = d_i ** 2
    sum_d_squared = float(np.sum(d_squared))

    # 3. Tied Ranks Calculations and Corrections
    _, counts_x = np.unique(x, return_counts=True)
    ties_x = counts_x[counts_x > 1]
    ties_x_count = int(np.sum(ties_x))
    ties_x_pct = float(ties_x_count / n_pts * 100.0)
    tie_corr_x = float(np.sum(ties_x**3 - ties_x) / 12.0)

    _, counts_y = np.unique(y, return_counts=True)
    ties_y = counts_y[counts_y > 1]
    ties_y_count = int(np.sum(ties_y))
    ties_y_pct = float(ties_y_count / n_pts * 100.0)
    tie_corr_y = float(np.sum(ties_y**3 - ties_y) / 12.0)

    has_ties = (ties_x_count > 0 or ties_y_count > 0)

    # Untied Spearman formula: 1 - 6*sum(d^2) / (n*(n^2 - 1))
    rho_untied = float(1.0 - (6.0 * sum_d_squared) / (n_pts * (n_pts**2 - 1)))

    # Tie-corrected Spearman (Scipy exact calculation via Pearson on ranks)
    res_spearman = stats.spearmanr(x, y)
    rho = float(res_spearman.statistic)
    p_val = float(res_spearman.pvalue)

    # 4. Hypothesis Testing (t-statistic for H0: rho = 0)
    df = n_pts - 2
    denom = np.sqrt(max(1e-12, 1.0 - rho**2))
    t_stat = float(rho * np.sqrt(df) / denom)

    # 5. Fisher's z 95% Confidence Interval for rho
    rho_clamped = float(np.clip(rho, -0.99999, 0.99999))
    z_score = float(np.arctanh(rho_clamped))
    se_z = float(1.06 / np.sqrt(max(1, n_pts - 3)))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_lower = float(np.tanh(z_score - z_crit * se_z))
    ci_upper = float(np.tanh(z_score + z_crit * se_z))
    rho_ci = (ci_lower, ci_upper)

    # 6. Comparative Indices: Pearson's r and Kendall's Tau-b
    res_pearson = stats.pearsonr(x, y)
    pearson_r = float(res_pearson.statistic)
    pearson_p = float(res_pearson.pvalue)

    res_kendall = stats.kendalltau(x, y)
    kendall_tau = float(res_kendall.statistic)
    kendall_p = float(res_kendall.pvalue)

    # 7. Linear Regression on Ranks
    rank_reg = stats.linregress(rank_x, rank_y)
    rank_slope = float(rank_reg.slope)
    rank_intercept = float(rank_reg.intercept)

    # 8. Descriptive Parameters
    mean_x, std_x = float(np.mean(x)), float(np.std(x, ddof=1))
    min_x, max_x = float(np.min(x)), float(np.max(x))
    median_x = float(np.median(x))
    iqr_x = float(np.percentile(x, 75) - np.percentile(x, 25))

    mean_y, std_y = float(np.mean(y)), float(np.std(y, ddof=1))
    min_y, max_y = float(np.min(y)), float(np.max(y))
    median_y = float(np.median(y))
    iqr_y = float(np.percentile(y, 75) - np.percentile(y, 25))

    mean_rank_x, std_rank_x = float(np.mean(rank_x)), float(np.std(rank_x, ddof=1))
    mean_rank_y, std_rank_y = float(np.mean(rank_y)), float(np.std(rank_y, ddof=1))

    # 9. Monotonic Sorted Points for Trend Line
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    window_sz = max(3, min(11, n_pts // 5))
    if window_sz % 2 == 0:
        window_sz += 1
    y_smooth = pd.Series(y_sorted).rolling(window=window_sz, center=True, min_periods=1).mean().to_numpy()

    # 10. Subject-Level Observational Microdata
    df_cases = pd.DataFrame({
        "Subject_ID": [f"SUBJ_{i+1:03d}" for i in range(n_pts)],
        "Raw_X": x,
        "Raw_Y": y,
        "Rank_X": rank_x,
        "Rank_Y": rank_y,
        "Rank_Diff_d": d_i,
        "d_Squared": d_squared
    })

    demo_df = pd.DataFrame({
        "Predictor_X": x,
        "Outcome_Y": y,
        "Rank_X": rank_x,
        "Rank_Y": rank_y
    })

    # 11. Effect Size & Direction Labels
    abs_rho = abs(rho)
    if abs_rho < 0.10:
        effect_label = "Negligible"
    elif abs_rho < 0.30:
        effect_label = "Small / Weak"
    elif abs_rho < 0.50:
        effect_label = "Moderate"
    else:
        effect_label = "Strong / Large"

    if rho >= 0.05:
        direction_label = "Positive"
    elif rho <= -0.05:
        direction_label = "Negative"
    else:
        direction_label = "Near Zero"

    is_significant = (p_val < alpha)

    # 12. Assumption Diagnostics
    assumptions_eval = []

    # Assumption 1: Monotonic Relationship
    if is_significant:
        gap = abs(rho) - abs(pearson_r)
        if gap > 0.08:
            mono_verdict = "Non-Linear Monotonic Confirmed (Spearman > Pearson)"
            mono_status = "pass"
            mono_detail = (
                f"Spearman's |ρ| ({abs_rho:.3f}) exceeds Pearson's |r| ({abs(pearson_r):.3f}) by {gap:+.3f}. "
                "This indicates a curved, non-linear monotonic relationship where ranking successfully linearizes the association."
            )
        else:
            mono_verdict = "Monotonic Relationship Present"
            mono_status = "pass"
            mono_detail = f"Statistically significant monotonic association detected (|ρ| = {abs_rho:.3f}, p = {p_val:.4e} < {alpha})."
    else:
        mono_verdict = "No Monotonic Association"
        mono_status = "info"
        mono_detail = f"No statistically significant monotonic trend found (p = {p_val:.4f} ≥ {alpha})."

    assumptions_eval.append({
        "name": "1. Monotonic Trend Direction",
        "description": "As X increases, Y should monotonically increase or decrease without reversing direction.",
        "diagnostic_value": f"Spearman ρ = {rho:+.3f} vs. Pearson r = {pearson_r:+.3f}",
        "verdict": mono_verdict,
        "status": mono_status,
        "detail": mono_detail
    })

    # Assumption 2: Scale of Measurement
    assumptions_eval.append({
        "name": "2. Scale of Measurement (Ordinal or Continuous)",
        "description": "Variables must be measured on at least an ordinal scale (ranked) or continuous scale.",
        "diagnostic_value": "Ordinal / Interval / Ratio Scale Satisfied",
        "verdict": "Assumption Satisfied",
        "status": "pass",
        "detail": f"Both variables ('{var_x_name}' and '{var_y_name}') can be uniquely ordered and ranked."
    })

    # Assumption 3: Independent Observations
    assumptions_eval.append({
        "name": "3. Paired Independent Observations",
        "description": "Each pair (X, Y) must represent an independent measurement from a distinct subject.",
        "diagnostic_value": f"N = {n_pts} Paired Subjects",
        "verdict": "Assumption Satisfied",
        "status": "pass",
        "detail": "Data pairs are independently sampled; no cross-contamination between observation pairs."
    })

    # Assumption 4: Tied Ranks Frequency
    max_ties_pct = max(ties_x_pct, ties_y_pct)
    if max_ties_pct == 0.0:
        tie_status = "pass"
        tie_verdict = "No Tied Values (Classic Formula Exact)"
        tie_detail = "All data points are distinct. Untied formula 1 - 6Σd²/(N(N²-1)) matches tie-corrected rho exactly."
    elif max_ties_pct <= 25.0:
        tie_status = "pass"
        tie_verdict = f"Moderate Ties ({max_ties_pct:.1f}%) — Corrected"
        tie_detail = (
            f"Ties present ({ties_x_count} in X [{ties_x_pct:.1f}%], {ties_y_count} in Y [{ties_y_pct:.1f}%]). "
            "Fractional average ranking and tie correction factors Tx and Ty applied."
        )
    else:
        tie_status = "warning"
        tie_verdict = f"Heavy Ties ({max_ties_pct:.1f}%) — Consider Kendall's Tau"
        tie_detail = (
            f"High frequency of tied ranks detected ({max_ties_pct:.1f}% tied). "
            f"When ties exceed 25%, Kendall's Tau-b (τ = {kendall_tau:+.3f}, p = {kendall_p:.4e}) is often preferred."
        )

    assumptions_eval.append({
        "name": "4. Tied Ranks Frequency & Adjustment",
        "description": "Tied values receive average fractional ranks; extensive ties reduce test power.",
        "diagnostic_value": f"Ties in X: {ties_x_pct:.1f}%, Ties in Y: {ties_y_pct:.1f}%",
        "verdict": tie_verdict,
        "status": tie_status,
        "detail": tie_detail
    })

    # 13. Overall Recommendation & Verdict
    if is_significant:
        overall_status = "pass"
        overall_verdict_badge = f"✅ Statistically Significant Monotonic Association (ρ = {rho:+.3f}, p = {p_val:.4e} < {alpha})"
        overall_recommendation = (
            f"**Reject the null hypothesis of zero rank correlation (p = {p_val:.4e} < {alpha}).** "
            f"There is a statistically significant {direction_label.lower()} monotonic relationship between **{var_x_name}** and **{var_y_name}** "
            f"(Spearman's ρ = **{rho:+.3f}**, 95% CI: [{rho_ci[0]:+.3f}, {rho_ci[1]:+.3f}], t({df}) = {t_stat:.3f}). "
            f"{'Because non-linear curvature was detected, Spearman rank correlation provides a much better summary of association than linear Pearson r.' if abs(rho) > abs(pearson_r) + 0.08 else 'Both rank and linear indices show consistent alignment.'}"
        )
    else:
        overall_status = "info"
        overall_verdict_badge = f"🟡 No Statistically Significant Monotonic Association (ρ = {rho:+.3f}, p = {p_val:.4f} ≥ {alpha})"
        overall_recommendation = (
            f"**Fail to reject the null hypothesis of zero rank correlation (p = {p_val:.4f} ≥ {alpha}).** "
            f"There is no statistically significant monotonic association between **{var_x_name}** and **{var_y_name}** "
            f"(Spearman's ρ = {rho:+.3f}, 95% CI: [{rho_ci[0]:+.3f}, {rho_ci[1]:+.3f}], t({df}) = {t_stat:.3f})."
        )

    results = {
        "x": x,
        "y": y,
        "rank_x": rank_x,
        "rank_y": rank_y,
        "d_i": d_i,
        "d_squared": d_squared,
        "sum_d_squared": sum_d_squared,
        "df_cases": df_cases,
        "var_x_name": var_x_name,
        "var_y_name": var_y_name,
        "n_pts": n_pts,
        "df": df,
        "spearman_rho": rho,
        "spearman_p": p_val,
        "spearman_rho_untied": rho_untied,
        "t_stat": t_stat,
        "fisher_z": z_score,
        "se_z": se_z,
        "rho_ci": rho_ci,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "kendall_tau": kendall_tau,
        "kendall_p": kendall_p,
        "rank_slope": rank_slope,
        "rank_intercept": rank_intercept,
        "ties_x_count": ties_x_count,
        "ties_y_count": ties_y_count,
        "ties_x_pct": ties_x_pct,
        "ties_y_pct": ties_y_pct,
        "tie_correction_x": tie_corr_x,
        "tie_correction_y": tie_corr_y,
        "has_ties": has_ties,
        "mean_x": mean_x,
        "std_x": std_x,
        "min_x": min_x,
        "max_x": max_x,
        "median_x": median_x,
        "iqr_x": iqr_x,
        "mean_y": mean_y,
        "std_y": std_y,
        "min_y": min_y,
        "max_y": max_y,
        "median_y": median_y,
        "iqr_y": iqr_y,
        "mean_rank_x": mean_rank_x,
        "std_rank_x": std_rank_x,
        "mean_rank_y": mean_rank_y,
        "std_rank_y": std_rank_y,
        "x_sorted": x_sorted,
        "y_sorted": y_sorted,
        "y_smooth": y_smooth,
        "effect_label": effect_label,
        "direction_label": direction_label,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return demo_df, results


def generate_linear_regression_demo(
    n_pts: int = 80,
    n_predictors: int = 3,
    scenario: str = "academic_performance",
    noise_level: float = 1.0,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic multivariate data and calculates complete Ordinary Least Squares (OLS)
    regression parameters, omnibus ANOVA decomposition, standardized beta weights, multicollinearity
    diagnostics (VIF/Tolerance), residual influence statistics (Cook's D and leverage), and regression assumptions.

    Args:
        n_pts: Number of observations (N).
        n_predictors: Number of predictor variables (1, 2, or 3).
        scenario: Preset scenario ('academic_performance', 'real_estate', 'simple_bivariate',
                  'multicollinear', 'heteroscedastic', 'custom').
        noise_level: Multiplier for residual error variance.
        alpha: Significance threshold (default 0.05).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (demo_df, results_dict).
    """
    n_pts = max(20, int(n_pts))
    rng = np.random.default_rng(seed)

    if scenario == "academic_performance":
        # Multiple regression: 3 predictors predicting Exam Score
        k = 3
        outcome_name = "Final Exam Score (%)"
        pred_names = ["Study Time (Hours / Week)", "Class Attendance Rate (%)", "Prior Baseline GPA"]
        x1 = np.clip(rng.normal(15.0, 4.5, size=n_pts), 2.0, 32.0)
        x2 = np.clip(rng.normal(84.0, 10.0, size=n_pts) + 0.4 * (x1 - 15.0), 50.0, 100.0)
        x3 = np.clip(rng.normal(3.1, 0.45, size=n_pts) + 0.03 * (x1 - 15.0), 2.0, 4.0)
        X_data = np.column_stack([x1, x2, x3])
        # True equation: Y = 14.0 + 1.45*X1 + 0.32*X2 + 8.2*X3 + noise
        err = rng.normal(0.0, 5.5 * noise_level, size=n_pts)
        y = 14.0 + 1.45 * x1 + 0.32 * x2 + 8.2 * x3 + err
        y = np.clip(y, 25.0, 100.0)

    elif scenario == "real_estate":
        # Multiple regression: 3 predictors predicting Home Price
        k = 3
        outcome_name = "Home Price ($k)"
        pred_names = ["Living Area (Hundred Sq Ft)", "Number of Bedrooms", "Property Age (Years)"]
        x1 = np.clip(rng.normal(22.0, 5.5, size=n_pts), 8.0, 42.0)
        x2 = np.clip(np.round(rng.normal(3.2, 0.8, size=n_pts) + 0.08 * (x1 - 22.0)), 1.0, 6.0)
        x3 = np.clip(rng.uniform(1.0, 45.0, size=n_pts), 1.0, 50.0)
        X_data = np.column_stack([x1, x2, x3])
        err = rng.normal(0.0, 22.0 * noise_level, size=n_pts)
        y = 85.0 + 14.5 * x1 + 18.0 * x2 - 2.8 * x3 + err
        y = np.maximum(y, 50.0)

    elif scenario == "simple_bivariate":
        # Simple OLS regression with 1 predictor
        k = 1
        outcome_name = "Quarterly Revenue ($k)"
        pred_names = ["Digital Marketing Budget ($k)"]
        x1 = np.clip(rng.normal(18.0, 5.0, size=n_pts), 4.0, 38.0)
        X_data = x1.reshape(-1, 1)
        err = rng.normal(0.0, 12.0 * noise_level, size=n_pts)
        y = 42.0 + 4.15 * x1 + err

    elif scenario == "multicollinear":
        # Severe multicollinearity between X1 and X2 (r > 0.90, VIF > 6.0)
        k = 2 if n_predictors < 3 else 3
        outcome_name = "Corporate Production Output"
        pred_names = ["Total Labor Hours", "Machine Operating Hours"]
        if k == 3:
            pred_names.append("Raw Material Inputs (Tons)")
        base = rng.normal(50.0, 10.0, size=n_pts)
        x1 = base + rng.normal(0.0, 2.0, size=n_pts)
        x2 = base * 0.95 + rng.normal(0.0, 2.2, size=n_pts)  # High correlation with x1
        cols = [x1, x2]
        if k == 3:
            x3 = rng.normal(30.0, 6.0, size=n_pts)
            cols.append(x3)
        X_data = np.column_stack(cols)
        err = rng.normal(0.0, 8.0 * noise_level, size=n_pts)
        y = 30.0 + 1.2 * x1 + 0.9 * x2 + (0.5 * x3 if k == 3 else 0.0) + err

    elif scenario == "heteroscedastic":
        # Heteroscedastic error variance increasing with predictor
        k = 2
        outcome_name = "Weekly Food Expenditure ($)"
        pred_names = ["Weekly Household Income ($100s)", "Household Family Size"]
        x1 = np.clip(rng.normal(20.0, 6.0, size=n_pts), 6.0, 45.0)
        x2 = np.clip(np.round(rng.normal(3.0, 1.0, size=n_pts)), 1.0, 7.0)
        X_data = np.column_stack([x1, x2])
        # Variance fans out as income increases (funnel shape)
        funnel_sd = (2.0 + 1.2 * x1) * noise_level
        err = rng.normal(0.0, funnel_sd, size=n_pts)
        y = 50.0 + 6.8 * x1 + 18.0 * x2 + err

    else:
        # Custom regression model
        k = max(1, min(3, int(n_predictors)))
        outcome_name = "Dependent Outcome Y"
        pred_names = [f"Predictor X{i+1}" for i in range(k)]
        cols = []
        for i in range(k):
            col = rng.normal(20.0 * (i + 1), 5.0, size=n_pts)
            cols.append(col)
        X_data = np.column_stack(cols)
        weights = [3.5, -2.0, 1.8][:k]
        signal = 25.0
        for i in range(k):
            signal += weights[i] * cols[i]
        err = rng.normal(0.0, 8.0 * noise_level, size=n_pts)
        y = signal + err

    n = len(y)
    k_preds = X_data.shape[1]

    # 1. Design Matrix & OLS Parameter Estimation
    X_mat = np.column_stack([np.ones(n), X_data])  # Shape (n, k + 1)
    # Solve beta = (X^T X)^(-1) X^T y
    xtx = X_mat.T @ X_mat
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X_mat.T @ y)

    # 2. Predicted Values & Residuals
    y_fitted = X_mat @ beta
    residuals = y - y_fitted
    mean_y = float(np.mean(y))

    # 3. Sum of Squares & ANOVA Decomposition
    ss_tot = float(np.sum((y - mean_y) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    ss_reg = float(max(0.0, ss_tot - ss_res))

    df_reg = k_preds
    df_res = n - k_preds - 1
    df_tot = n - 1

    ms_reg = ss_reg / max(1, df_reg)
    ms_res = ss_res / max(1, df_res)
    rmse = float(np.sqrt(ms_res))

    f_stat = float(ms_reg / ms_res) if ms_res > 0 else 0.0
    p_f = float(1.0 - stats.f.cdf(f_stat, df_reg, df_res))

    r_squared = float(ss_reg / ss_tot) if ss_tot > 0 else 0.0
    r_squared_pct = float(r_squared * 100.0)
    adj_r_squared = float(1.0 - (1.0 - r_squared) * (df_tot / max(1, df_res)))
    unexplained_pct = float(max(0.0, 100.0 - r_squared_pct))

    # 4. Standard Errors, t-values, p-values, 95% CIs
    cov_beta = ms_res * xtx_inv
    se_beta = np.sqrt(np.maximum(1e-12, np.diag(cov_beta)))
    t_beta = beta / se_beta
    p_beta = 2.0 * (1.0 - stats.t.cdf(np.abs(t_beta), df_res))

    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_res))
    ci_lower = beta - t_crit * se_beta
    ci_upper = beta + t_crit * se_beta

    # 5. Standardized Beta Weights (beta*)
    std_y = float(np.std(y, ddof=1))
    beta_standardized = [np.nan]  # Intercept has no standardized beta
    for j in range(k_preds):
        std_xj = float(np.std(X_data[:, j], ddof=1))
        b_std = float(beta[j + 1] * (std_xj / std_y)) if std_y > 0 else 0.0
        beta_standardized.append(b_std)

    # 6. Multicollinearity Diagnostics: VIF & Tolerance
    vifs = [np.nan]  # Intercept
    tolerances = [np.nan]
    if k_preds == 1:
        vifs.append(1.0)
        tolerances.append(1.0)
    else:
        for j in range(k_preds):
            y_aux = X_data[:, j]
            other_cols = [c for c in range(k_preds) if c != j]
            X_aux = np.column_stack([np.ones(n), X_data[:, other_cols]])
            aux_beta = np.linalg.pinv(X_aux.T @ X_aux) @ (X_aux.T @ y_aux)
            aux_pred = X_aux @ aux_beta
            ss_tot_aux = np.sum((y_aux - np.mean(y_aux)) ** 2)
            ss_res_aux = np.sum((y_aux - aux_pred) ** 2)
            r2_aux = max(0.0, min(0.9999, 1.0 - ss_res_aux / ss_tot_aux)) if ss_tot_aux > 0 else 0.0
            vif_val = float(1.0 / (1.0 - r2_aux))
            tol_val = float(1.0 / vif_val)
            vifs.append(vif_val)
            tolerances.append(tol_val)

    max_vif = float(np.nanmax(vifs[1:]))

    # 7. Residual Diagnostics, Leverage & Cook's Distance
    std_residuals = residuals / rmse if rmse > 0 else residuals
    # Hat matrix diagonals: h_ii = diag(X (X^T X)^(-1) X^T)
    hat_diags = np.sum((X_mat @ xtx_inv) * X_mat, axis=1)
    avg_leverage = float((k_preds + 1) / n)
    high_leverage_thresh = 2.0 * avg_leverage

    # Cook's distance: D_i = (r_i^2 / (k + 1)) * (h_ii / (1 - h_ii))
    cooks_d = (std_residuals ** 2 / (k_preds + 1)) * (hat_diags / np.maximum(1e-6, 1.0 - hat_diags))
    cooks_thresh = float(4.0 / n)
    high_cooks_count = int(np.sum(cooks_d > cooks_thresh))
    max_cooks_d = float(np.max(cooks_d))

    # 8. Durbin-Watson Autocorrelation Statistic
    diff_res = np.diff(residuals)
    dw_stat = float(np.sum(diff_res ** 2) / np.sum(residuals ** 2)) if np.sum(residuals ** 2) > 0 else 2.0

    # 9. Breusch-Pagan Homoscedasticity Test
    sq_res = residuals ** 2
    bp_beta = np.linalg.pinv(X_mat.T @ X_mat) @ (X_mat.T @ sq_res)
    bp_pred = X_mat @ bp_beta
    ss_tot_bp = np.sum((sq_res - np.mean(sq_res)) ** 2)
    ss_res_bp = np.sum((sq_res - bp_pred) ** 2)
    r2_bp = max(0.0, min(1.0, 1.0 - ss_res_bp / ss_tot_bp)) if ss_tot_bp > 0 else 0.0
    lm_stat = float(n * r2_bp)
    p_bp = float(1.0 - stats.chi2.cdf(lm_stat, k_preds))
    is_homoscedastic = (p_bp >= 0.05)

    # 10. Shapiro-Wilk Residual Normality Test
    res_sub = residuals[:min(n, 5000)]
    sw_stat, p_norm_res = stats.shapiro(res_sub)
    is_normal_res = (p_norm_res >= 0.05)

    # 11. Linearity Evaluation (Curvature check against fitted values)
    quad_term = (y_fitted - np.mean(y_fitted)) ** 2
    X_quad = np.column_stack([X_mat, quad_term])
    beta_quad = np.linalg.pinv(X_quad.T @ X_quad) @ (X_quad.T @ y)
    res_quad = y - (X_quad @ beta_quad)
    ss_res_quad = np.sum(res_quad ** 2)
    f_curv = float((ss_res - ss_res_quad) / (ss_res_quad / (n - k_preds - 2))) if ss_res_quad > 0 else 0.0
    p_linearity = float(1.0 - stats.f.cdf(f_curv, 1, n - k_preds - 2))
    is_linear = (p_linearity >= 0.05)

    # 12. Coefficients Table Construction
    param_names = ["Intercept (Constant)"] + pred_names
    df_coefs = pd.DataFrame({
        "Parameter": param_names,
        "Unstandardized_B": beta,
        "Std_Error": se_beta,
        "CI_95_Lower": ci_lower,
        "CI_95_Upper": ci_upper,
        "Standardized_Beta": beta_standardized,
        "t_Statistic": t_beta,
        "p_value": p_beta,
        "VIF": vifs,
        "Tolerance": tolerances
    })

    # 13. Subject-Level Observational Data Preview
    data_dict = {"Subject_ID": [f"SUBJ_{i+1:03d}" for i in range(n)]}
    for j in range(k_preds):
        data_dict[pred_names[j]] = X_data[:, j]
    data_dict[outcome_name] = y
    data_dict["Fitted_Ŷ"] = y_fitted
    data_dict["Residual_e"] = residuals
    data_dict["Std_Residual_r"] = std_residuals
    data_dict["Leverage_h"] = hat_diags
    data_dict["Cooks_Distance_D"] = cooks_d
    data_dict["Is_Influential"] = (cooks_d > cooks_thresh)

    df_cases = pd.DataFrame(data_dict)
    demo_df = df_cases.copy()

    # 14. Core Methodological Assumptions Evaluation
    assumptions_eval = []

    # 1. Linearity
    assumptions_eval.append({
        "name": "1. Linearity of Relationship",
        "description": "The mean response Y must be a linear combination of the predictor variables.",
        "diagnostic_value": f"Polynomial Curvature Test p = {p_linearity:.4f}",
        "verdict": "Assumption Satisfied" if is_linear else "Curvature Detected",
        "status": "pass" if is_linear else "warning",
        "detail": (
            "Linear model specification is appropriate. No statistically significant non-linear curvature detected."
            if is_linear else
            f"Curvature test p = {p_linearity:.4f} < 0.05 indicates non-linear relationship. Consider polynomial terms or non-linear transformation."
        )
    })

    # 2. Independence of Errors
    is_indep = (1.5 <= dw_stat <= 2.5)
    assumptions_eval.append({
        "name": "2. Independence of Residuals (Autocorrelation)",
        "description": "Observations and error terms must be statistically independent (no serial correlation).",
        "diagnostic_value": f"Durbin-Watson = {dw_stat:.3f} (Ideal: ~2.0)",
        "verdict": "No Autocorrelation" if is_indep else "Autocorrelation Warning",
        "status": "pass" if is_indep else "warning",
        "detail": (
            f"Durbin-Watson statistic of {dw_stat:.3f} is within acceptable bounds [1.5, 2.5], confirming independent errors."
            if is_indep else
            f"Durbin-Watson statistic of {dw_stat:.3f} indicates potential error correlation. Verify cross-sectional sampling or time-series structure."
        )
    })

    # 3. Homoscedasticity
    assumptions_eval.append({
        "name": "3. Homoscedasticity (Constant Residual Variance)",
        "description": "Residual variance must remain constant across all predicted values Ŷ.",
        "diagnostic_value": f"Breusch-Pagan Test p = {p_bp:.4f}",
        "verdict": "Homoscedastic" if is_homoscedastic else "Heteroscedasticity Present",
        "status": "pass" if is_homoscedastic else "warning",
        "detail": (
            f"Breusch-Pagan test p = {p_bp:.4f} ≥ 0.05 indicates equal variance across the range of fitted values."
            if is_homoscedastic else
            f"Breusch-Pagan test p = {p_bp:.4f} < 0.05 confirms non-constant variance (funnel shape). Consider robust standard errors (HC3) or WLS."
        )
    })

    # 4. Residual Normality
    assumptions_eval.append({
        "name": "4. Normality of Residuals",
        "description": "Regression error terms must be approximately normally distributed around zero.",
        "diagnostic_value": f"Shapiro-Wilk p = {p_norm_res:.4f} (W = {sw_stat:.3f})",
        "verdict": "Normally Distributed" if is_normal_res else "Non-Normal Residuals",
        "status": "pass" if is_normal_res else "warning",
        "detail": (
            f"Shapiro-Wilk p = {p_norm_res:.4f} ≥ 0.05 confirms normally distributed residuals."
            if is_normal_res else
            f"Shapiro-Wilk p = {p_norm_res:.4f} < 0.05 indicates skewness or kurtosis in errors. For N ≥ 30, OLS estimates remain robust by Central Limit Theorem."
        )
    })

    # 5. Multicollinearity
    if k_preds == 1:
        multi_status = "pass"
        multi_verdict = "N/A (Single Predictor)"
        multi_detail = "Multicollinearity only applies when 2 or more predictors are included."
    elif max_vif < 5.0:
        multi_status = "pass"
        multi_verdict = f"Low Collinearity (Max VIF = {max_vif:.2f})"
        multi_detail = f"All predictor VIF values are below standard threshold of 5.0 (Max VIF = {max_vif:.2f}), confirming independent explanatory power."
    elif max_vif < 10.0:
        multi_status = "warning"
        multi_verdict = f"Moderate Collinearity (Max VIF = {max_vif:.2f})"
        multi_detail = f"Max VIF of {max_vif:.2f} indicates moderate overlap between predictors. Standard errors may be somewhat inflated."
    else:
        multi_status = "fail"
        multi_verdict = f"Severe Collinearity (Max VIF = {max_vif:.2f})"
        multi_detail = f"Severe multicollinearity detected (Max VIF = {max_vif:.2f} ≥ 10). Predictors are redundant, destabilizing individual beta estimates."

    assumptions_eval.append({
        "name": "5. Absence of Multicollinearity",
        "description": "Predictors should not be excessively correlated with one another (VIF < 5.0).",
        "diagnostic_value": f"Maximum VIF = {max_vif:.2f}" if k_preds > 1 else "VIF = 1.0 (Bivariate)",
        "verdict": multi_verdict,
        "status": multi_status,
        "detail": multi_detail
    })

    # 6. Absence of Outliers & Leverage Points
    if high_cooks_count == 0:
        out_status = "pass"
        out_verdict = f"No Influential Points (Max D = {max_cooks_d:.3f})"
        out_detail = f"No observation exceeds Cook's threshold 4/N = {cooks_thresh:.3f}. Parameter estimates are stable."
    else:
        out_status = "warning"
        out_verdict = f"{high_cooks_count} Influential Point(s) Detected"
        out_detail = f"{high_cooks_count} observation(s) exceed Cook's distance threshold 4/N = {cooks_thresh:.3f} (Max D = {max_cooks_d:.3f})."

    assumptions_eval.append({
        "name": "6. Outliers & Influential Observations",
        "description": "No individual data point should exert excessive leverage or distortion on the regression line.",
        "diagnostic_value": f"Max Cook's D = {max_cooks_d:.3f} (Threshold 4/N = {cooks_thresh:.3f})",
        "verdict": out_verdict,
        "status": out_status,
        "detail": out_detail
    })

    # 15. Overall Verdict & Recommendation
    is_model_sig = (p_f < alpha)
    sig_preds_count = int(np.sum(p_beta[1:] < alpha))

    if is_model_sig:
        overall_status = "pass"
        overall_verdict_badge = f"✅ Statistically Significant Regression Model (R² = {r_squared_pct:.1f}%, F({df_reg}, {df_res}) = {f_stat:.2f}, p = {p_f:.4e} < {alpha})"
        overall_recommendation = (
            f"**Reject the null hypothesis of zero omnibus regression fit (p = {p_f:.4e} < {alpha}).** "
            f"The overall model significantly predicts **{outcome_name}**, accounting for **{r_squared_pct:.1f}%** of outcome variance "
            f"(Adjusted R² = {adj_r_squared:.3f}, RMSE = {rmse:.2f}). "
            f"{sig_preds_count} of {k_preds} individual predictor(s) demonstrate statistically significant independent contributions at α = {alpha}."
        )
    else:
        overall_status = "info"
        overall_verdict_badge = f"🟡 No Statistically Significant Model Fit (R² = {r_squared_pct:.1f}%, F({df_reg}, {df_res}) = {f_stat:.2f}, p = {p_f:.4f} ≥ {alpha})"
        overall_recommendation = (
            f"**Fail to reject the null hypothesis of zero omnibus regression fit (p = {p_f:.4f} ≥ {alpha}).** "
            f"The predictor set does not explain a statistically significant proportion of variance in **{outcome_name}** "
            f"(R² = {r_squared_pct:.1f}%, F({df_reg}, {df_res}) = {f_stat:.2f}, p = {p_f:.4f})."
        )

    results = {
        "n_pts": n,
        "k_preds": k_preds,
        "outcome_name": outcome_name,
        "pred_names": pred_names,
        "param_names": param_names,
        "X_data": X_data,
        "y": y,
        "y_fitted": y_fitted,
        "residuals": residuals,
        "std_residuals": std_residuals,
        "beta": beta,
        "se_beta": se_beta,
        "t_beta": t_beta,
        "p_beta": p_beta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "beta_standardized": beta_standardized,
        "vifs": vifs,
        "tolerances": tolerances,
        "max_vif": max_vif,
        "ss_tot": ss_tot,
        "ss_reg": ss_reg,
        "ss_res": ss_res,
        "df_reg": df_reg,
        "df_res": df_res,
        "df_tot": df_tot,
        "ms_reg": ms_reg,
        "ms_res": ms_res,
        "f_stat": f_stat,
        "p_f": p_f,
        "r_squared": r_squared,
        "r_squared_pct": r_squared_pct,
        "adj_r_squared": adj_r_squared,
        "unexplained_pct": unexplained_pct,
        "rmse": rmse,
        "dw_stat": dw_stat,
        "p_bp": p_bp,
        "is_homoscedastic": is_homoscedastic,
        "p_norm_res": p_norm_res,
        "is_normal_res": is_normal_res,
        "p_linearity": p_linearity,
        "is_linear": is_linear,
        "hat_diags": hat_diags,
        "avg_leverage": avg_leverage,
        "cooks_d": cooks_d,
        "cooks_thresh": cooks_thresh,
        "high_cooks_count": high_cooks_count,
        "max_cooks_d": max_cooks_d,
        "df_coefs": df_coefs,
        "df_cases": df_cases,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return demo_df, results


def generate_one_sample_demo(
    n: int = 35,
    sample_mean: float = 105.0,
    sample_std: float = 12.0,
    benchmark: float = 100.0,
    dist_type: str = "normal",
    outlier_count: int = 0,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic data for a One-Sample t-test and runs full assumption checks.

    Args:
        n: Number of observations.
        sample_mean: Target mean for the distribution.
        sample_std: Target standard deviation.
        benchmark: Theoretical benchmark or population mean (mu_0).
        dist_type: Distribution shape ('normal', 'skewed_right', 'skewed_left', 'heavy_tailed').
        outlier_count: Number of extreme outliers to inject (0, 1, 2, 3).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (pd.DataFrame, results_dict).
    """
    rng = np.random.default_rng(seed)

    # 1. Generate base random distribution
    if dist_type == "skewed_right":
        raw = rng.exponential(scale=1.0, size=n)
    elif dist_type == "skewed_left":
        raw = -rng.exponential(scale=1.0, size=n)
    elif dist_type == "heavy_tailed":
        raw = rng.standard_t(df=3, size=n)
    else:  # normal
        raw = rng.normal(loc=0.0, scale=1.0, size=n)

    # Standardize and rescale to match requested mean and std
    std_denom = float(np.std(raw, ddof=1)) if n > 1 else 1.0
    if std_denom == 0:
        std_denom = 1.0
    vals = (raw - np.mean(raw)) / std_denom * sample_std + sample_mean

    # Inject extreme outliers if requested
    if outlier_count > 0:
        outlier_indices = rng.choice(n, size=min(outlier_count, n), replace=False)
        for idx, offset_mult in enumerate(np.linspace(3.8, 5.0, num=len(outlier_indices))):
            vals[outlier_indices[idx]] = sample_mean + offset_mult * sample_std

    df = pd.DataFrame({
        "Observation_ID": [f"Obs_{i+1:03d}" for i in range(n)],
        "Value": vals
    })

    # 2. Compute One-Sample t-test metrics
    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))
    std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    se_val = float(std_val / np.sqrt(n)) if n > 0 else 0.0
    mean_diff = float(mean_val - benchmark)

    t_res = stats.ttest_1samp(vals, popmean=benchmark)
    t_stat = float(t_res.statistic)
    p_val = float(t_res.pvalue)
    df_deg = int(n - 1)

    t_crit = float(stats.t.ppf(0.975, df=df_deg)) if df_deg > 0 else 1.96
    ci_lower = float(mean_val - t_crit * se_val)
    ci_upper = float(mean_val + t_crit * se_val)
    cohens_d = float(mean_diff / std_val) if std_val > 0 else 0.0

    # 3. Non-parametric alternative: One-Sample Wilcoxon Signed-Rank Test
    diffs = vals - benchmark
    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) >= 5:
        try:
            wilc_res = stats.wilcoxon(nonzero_diffs)
            wilc_stat = float(wilc_res.statistic)
            wilc_p = float(wilc_res.pvalue)
        except Exception:
            wilc_stat, wilc_p = 0.0, 1.0
    else:
        wilc_stat, wilc_p = 0.0, 1.0

    # 4. Assumption Checks & Diagnostics
    # A. Normality Checks
    shapiro_w, shapiro_p = stats.shapiro(vals)
    skew_val = float(stats.skew(vals))
    kurt_val = float(stats.kurtosis(vals))

    if shapiro_p >= 0.05:
        normality_status = "pass"
        normality_verdict = "PASS"
        normality_detail = (
            f"Shapiro-Wilk test p = {shapiro_p:.4f} (≥ 0.05). No evidence against normality "
            f"(Skewness: {skew_val:.2f}, Kurtosis: {kurt_val:.2f})."
        )
    elif n >= 30 and abs(skew_val) <= 1.2:
        normality_status = "warning"
        normality_verdict = "CAUTION (CLT ROBUST)"
        normality_detail = (
            f"Shapiro-Wilk test p = {shapiro_p:.4f} (< 0.05). Non-normality detected; however, "
            f"sample size N = {n} ≥ 30 triggers the Central Limit Theorem (CLT) protection for "
            f"the sampling distribution of the mean. Moderate skewness = {skew_val:.2f}."
        )
    else:
        normality_status = "fail"
        normality_verdict = "VIOLATED"
        normality_detail = (
            f"Shapiro-Wilk test p = {shapiro_p:.4f} (< 0.05). Significant departure from normality "
            f"without sufficient sample size protection (N = {n}, Skewness: {skew_val:.2f})."
        )

    # B. Outlier Checks (Tukey IQR rule)
    q25, q75 = np.percentile(vals, [25, 75])
    iqr = q75 - q25
    mild_low, mild_high = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    ext_low, ext_high = q25 - 3.0 * iqr, q75 + 3.0 * iqr

    mild_outliers = int(np.sum((vals < mild_low) | (vals > mild_high)))
    extreme_outliers = int(np.sum((vals < ext_low) | (vals > ext_high)))

    if extreme_outliers == 0 and mild_outliers == 0:
        outlier_status = "pass"
        outlier_verdict = "PASS"
        outlier_detail = "No outliers detected via Tukey's IQR rule."
    elif extreme_outliers == 0 and mild_outliers <= 2:
        outlier_status = "warning"
        outlier_verdict = "CAUTION"
        outlier_detail = f"{mild_outliers} mild outlier(s) detected between 1.5×IQR and 3×IQR. Check influence on variance."
    else:
        outlier_status = "fail"
        outlier_verdict = "VIOLATED"
        outlier_detail = (
            f"{extreme_outliers} extreme outlier(s) detected beyond 3×IQR ({mild_outliers} total outliers). "
            f"Outliers strongly distort the mean and standard error."
        )

    # C. Continuous Scale & Independence Assumptions
    assumptions_eval = [
        {
            "name": "1. Continuous / Metric Measurement Scale",
            "description": "Dependent variable must be measured on a continuous interval or ratio scale.",
            "test_applied": "Variable type inspection",
            "diagnostic_value": "Continuous floating-point values",
            "status": "pass",
            "verdict": "PASS",
            "detail": "Data is continuous; calculation of arithmetic mean and standard deviation is appropriate."
        },
        {
            "name": "2. Independence of Observations",
            "description": "Observations must be independently sampled with no cluster or repeated-measures correlation.",
            "test_applied": "Sampling design evaluation",
            "diagnostic_value": "Independent random sampling",
            "status": "pass",
            "verdict": "PASS",
            "detail": "Observations are mutually independent without paired or repeated measurement structure."
        },
        {
            "name": "3. Normality of Distribution",
            "description": "Population distribution should be normal (or N ≥ 30 with CLT robustness).",
            "test_applied": "Shapiro-Wilk Test & Skewness / Kurtosis inspection",
            "diagnostic_value": f"W = {shapiro_w:.4f}, p = {shapiro_p:.4f}, Skew = {skew_val:.2f}",
            "status": normality_status,
            "verdict": normality_verdict,
            "detail": normality_detail
        },
        {
            "name": "4. Absence of Extreme Outliers",
            "description": "No extreme outliers that disproportionately pull the mean or inflate standard error.",
            "test_applied": "Tukey's IQR Rule (1.5×IQR mild, 3.0×IQR extreme)",
            "diagnostic_value": f"Mild Outliers: {mild_outliers}, Extreme Outliers: {extreme_outliers}",
            "status": outlier_status,
            "verdict": outlier_verdict,
            "detail": outlier_detail
        }
    ]

    # Overall Recommendation Synthesis
    has_failure = any(a["status"] == "fail" for a in assumptions_eval)
    has_warning = any(a["status"] == "warning" for a in assumptions_eval)

    if has_failure:
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption(s) Violated: Switch to Alternative Method"
        overall_recommendation = (
            f"Critical assumptions for the One-Sample t-test are violated. "
            f"The parametric t-test is vulnerable to biased p-values and inflated error rates in this scenario. "
            f"\n\n**Actionable Recommendation:** Switch to the non-parametric **Wilcoxon Signed-Rank Test (One-Sample)**, "
            f"which compares the sample median to the benchmark ({benchmark}) without normality assumptions. "
            f"\n\n**Alternative Test Result on this Data:** "
            f"Wilcoxon Statistic W = {wilc_stat:.1f}, **p-value = {wilc_p:.4f}**."
        )
    elif has_warning:
        overall_status = "warning"
        overall_verdict_badge = "🟡 Assumptions Borderline: Proceed with Caution"
        overall_recommendation = (
            f"Normality tests indicate minor departure, but sample size (N = {n} ≥ 30) allows the Central Limit Theorem "
            f"to protect the sampling distribution of the mean. "
            f"\n\n**Actionable Recommendation:** You may proceed with the **One-Sample t-test** (p = {p_val:.4f}), "
            f"but report the non-parametric **Wilcoxon Signed-Rank Test** (p = {wilc_p:.4f}) as a sensitivity check."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ All Assumptions Met: Proceed with One-Sample t-test"
        overall_recommendation = (
            f"All core assumptions (continuous scale, independence, normality, and absence of extreme outliers) "
            f"are successfully satisfied. "
            f"\n\n**Actionable Recommendation:** Proceed with confidence using the **One-Sample t-test**. "
            f"The parametric test provides optimal statistical power in this scenario."
        )

    # Q-Q plot theoretical quantiles and sorted sample values
    (osm, osr), (slope, intercept, _) = stats.probplot(vals, dist="norm")

    results = {
        "n": int(n),
        "benchmark": float(benchmark),
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "se": se_val,
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "p_val": p_val,
        "df": df_deg,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": cohens_d,
        "shapiro_w": float(shapiro_w),
        "shapiro_p": float(shapiro_p),
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "mild_outliers": mild_outliers,
        "extreme_outliers": extreme_outliers,
        "wilcoxon_stat": wilc_stat,
        "wilcoxon_p": wilc_p,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation,
        "qq_osm": osm.tolist(),
        "qq_osr": osr.tolist(),
        "qq_slope": float(slope),
        "qq_intercept": float(intercept)
    }

    return df, results


def generate_wilcoxon_demo(
    n: int = 30,
    sample_median: float = 108.0,
    sample_scale: float = 15.0,
    benchmark: float = 100.0,
    dist_shape: str = "symmetric_non_normal",
    zero_diff_count: int = 0,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic data for the Wilcoxon Signed-Rank Test and performs assumption diagnostics.

    Args:
        n: Total number of observations.
        sample_median: Target median for data generation.
        sample_scale: Scale / dispersion parameter.
        benchmark: Theoretical benchmark or reference median (mu_0).
        dist_shape: 'symmetric_non_normal' (Laplace), 'moderately_skewed' (Gamma), or 'severely_skewed' (Log-normal).
        zero_diff_count: Number of observations exactly equal to benchmark (tests tie/zero handling).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (pd.DataFrame, results_dict).
    """
    rng = np.random.default_rng(seed)

    # 1. Generate non-normal distribution based on shape selection
    if dist_shape == "symmetric_non_normal":
        # Laplace (double exponential) is non-normal and heavy-tailed, but perfectly symmetric
        raw = rng.laplace(loc=0.0, scale=sample_scale / 1.414, size=n)
        vals = raw + sample_median
    elif dist_shape == "moderately_skewed":
        # Gamma distribution with moderate skewness
        raw = rng.gamma(shape=3.0, scale=sample_scale / 3.0, size=n)
        raw_med = float(np.median(raw))
        vals = (raw - raw_med) + sample_median
    elif dist_shape == "severely_skewed":
        # Lognormal distribution with heavy positive skew
        raw = rng.lognormal(mean=0.0, sigma=1.0, size=n)
        raw_med = float(np.median(raw))
        vals = ((raw - raw_med) / (raw_med if raw_med > 0 else 1.0)) * sample_scale + sample_median
    else:
        # Uniform distribution (bounded, non-normal, symmetric)
        half_width = sample_scale * 1.732
        vals = rng.uniform(low=sample_median - half_width, high=sample_median + half_width, size=n)

    # Inject exact zeros (observations equal to benchmark) if requested
    if zero_diff_count > 0:
        zero_indices = rng.choice(n, size=min(zero_diff_count, n), replace=False)
        vals[zero_indices] = benchmark

    # 2. Compute Differences and Ranks
    diffs = vals - benchmark
    abs_diffs = np.abs(diffs)
    nonzero_mask = diffs != 0
    nonzero_diffs = diffs[nonzero_mask]
    nonzero_abs = abs_diffs[nonzero_mask]
    n_eff = len(nonzero_diffs)
    n_zeros = n - n_eff

    # Rank nonzero absolute differences (with average rank for ties)
    if n_eff > 0:
        ranks = stats.rankdata(nonzero_abs)
        pos_ranks = np.where(nonzero_diffs > 0, ranks, 0.0)
        neg_ranks = np.where(nonzero_diffs < 0, ranks, 0.0)
        sum_pos_ranks = float(np.sum(pos_ranks))
        sum_neg_ranks = float(np.sum(neg_ranks))
    else:
        ranks = np.array([])
        sum_pos_ranks = 0.0
        sum_neg_ranks = 0.0

    # Build DataFrame
    full_ranks = np.zeros(n)
    if n_eff > 0:
        full_ranks[nonzero_mask] = ranks

    df = pd.DataFrame({
        "Observation_ID": [f"Obs_{i+1:03d}" for i in range(n)],
        "Value": vals,
        "Difference": diffs,
        "Abs_Difference": abs_diffs,
        "Signed_Rank": np.where(diffs > 0, full_ranks, -full_ranks)
    })

    # 3. Statistical Calculations
    calc_median = float(np.median(vals))
    calc_mean = float(np.mean(vals))
    median_diff = float(calc_median - benchmark)

    # Wilcoxon Signed-Rank Test
    if n_eff >= 5:
        try:
            wilc_res = stats.wilcoxon(nonzero_diffs, alternative="two-sided")
            w_stat = float(wilc_res.statistic)
            p_wilcoxon = float(wilc_res.pvalue)
        except Exception:
            w_stat = float(min(sum_pos_ranks, sum_neg_ranks))
            p_wilcoxon = 1.0
    else:
        w_stat = float(min(sum_pos_ranks, sum_neg_ranks))
        p_wilcoxon = 1.0

    # Hodges-Lehmann pseudo-median estimate (median of Walsh averages)
    if n_eff > 0 and n_eff <= 200:
        walsh_diffs = [(nonzero_diffs[i] + nonzero_diffs[j]) / 2.0 for i in range(n_eff) for j in range(i, n_eff)]
        hl_estimate = float(np.median(walsh_diffs))
        hl_median = float(benchmark + hl_estimate)
    else:
        hl_estimate = median_diff
        hl_median = calc_median

    # Alternative Test: Exact Sign Test (Binomial)
    pos_count = int(np.sum(nonzero_diffs > 0))
    neg_count = int(np.sum(nonzero_diffs < 0))
    if n_eff > 0:
        sign_test_res = stats.binomtest(pos_count, n=n_eff, p=0.5, alternative="two-sided")
        p_sign_test = float(sign_test_res.pvalue)
    else:
        p_sign_test = 1.0

    # 4. Assumption Checks & Diagnostics
    # A. Scale of Measurement
    scale_status = "pass"
    scale_verdict = "PASS"
    scale_detail = "Data is continuous/ordinal, satisfying the requirement that values can be meaningfully ranked."

    # B. Independence of Observations
    indep_status = "pass"
    indep_verdict = "PASS"
    indep_detail = "Observations are mutually independent random draws without clustering or repeated-measures structure."

    # C. Symmetry of Differences around the Median
    if n_eff > 2:
        skew_diffs = float(stats.skew(nonzero_diffs))
    else:
        skew_diffs = 0.0

    if abs(skew_diffs) <= 0.8:
        symmetry_status = "pass"
        symmetry_verdict = "PASS"
        symmetry_detail = (
            f"Skewness of differences = {skew_diffs:.2f} (|Skew| ≤ 0.8). "
            f"The distribution of differences is approximately symmetric around the median, "
            f"fulfilling the core requirement for the Wilcoxon Signed-Rank Test."
        )
    elif abs(skew_diffs) <= 1.5:
        symmetry_status = "warning"
        symmetry_verdict = "CAUTION (MODERATE ASYMMETRY)"
        symmetry_detail = (
            f"Skewness of differences = {skew_diffs:.2f} (0.8 < |Skew| ≤ 1.5). "
            f"Moderate asymmetry detected. The Wilcoxon test evaluates the pseudo-median (Hodges-Lehmann location shift) "
            f"rather than the pure median. The test is generally robust, but check concordance with the Sign Test."
        )
    else:
        symmetry_status = "fail"
        symmetry_verdict = "VIOLATED (SEVERE ASYMMETRY)"
        symmetry_detail = (
            f"Skewness of differences = {skew_diffs:.2f} (|Skew| > 1.5). "
            f"Differences are strongly asymmetric. The Wilcoxon Signed-Rank Test can produce misleading results "
            f"because rank sums confound location differences with distribution shape. Switch to the Sign Test."
        )

    # D. Effective Sample Size & Zeros Check
    if n_eff >= 10:
        eff_n_status = "pass"
        eff_n_verdict = "PASS"
        eff_n_detail = f"Effective sample size N_eff = {n_eff} (excluding {n_zeros} zero differences). Adequate statistical power."
    elif n_eff >= 6:
        eff_n_status = "warning"
        eff_n_verdict = "CAUTION (SMALL SAMPLE)"
        eff_n_detail = f"Effective sample size N_eff = {n_eff} is small ({n_zeros} zero differences dropped). Exact distribution tables are required."
    else:
        eff_n_status = "fail"
        eff_n_verdict = "VIOLATED (INSUFFICIENT DATA)"
        eff_n_detail = f"Effective sample size N_eff = {n_eff} < 6 ({n_zeros} zeros). Insufficient data to reliably evaluate median differences."

    assumptions_eval = [
        {
            "name": "1. Measurement Scale (Ordinal or Continuous)",
            "description": "Data must be measured on at least an ordinal scale so differences can be ranked.",
            "test_applied": "Variable scale inspection",
            "diagnostic_value": "Rankable numerical/ordinal values",
            "status": scale_status,
            "verdict": scale_verdict,
            "detail": scale_detail
        },
        {
            "name": "2. Independence of Observations",
            "description": "Each observation must be independently sampled with no cluster correlation.",
            "test_applied": "Sampling design evaluation",
            "diagnostic_value": "Independent random sampling",
            "status": indep_status,
            "verdict": indep_verdict,
            "detail": indep_detail
        },
        {
            "name": "3. Symmetry of Differences around Median",
            "description": "Differences from the benchmark must be approximately symmetric for the test to strictly evaluate the median.",
            "test_applied": "Skewness of Differences (stats.skew)",
            "diagnostic_value": f"Skewness = {skew_diffs:.2f}",
            "status": symmetry_status,
            "verdict": symmetry_verdict,
            "detail": symmetry_detail
        },
        {
            "name": "4. Adequate Effective Sample Size (Non-Zero Differences)",
            "description": "Observations matching the benchmark exactly (diff = 0) are dropped; N_eff must be sufficient.",
            "test_applied": "Zero difference filtering count",
            "diagnostic_value": f"N_eff = {n_eff} (Zeros dropped = {n_zeros})",
            "status": eff_n_status,
            "verdict": eff_n_verdict,
            "detail": eff_n_detail
        }
    ]

    # Overall Recommendation Synthesis
    has_failure = any(a["status"] == "fail" for a in assumptions_eval)
    has_warning = any(a["status"] == "warning" for a in assumptions_eval)

    if has_failure:
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption(s) Violated: Switch to Alternative Method"
        overall_recommendation = (
            f"The assumption of symmetric differences is violated (|Skewness| = {skew_diffs:.2f} > 1.5) "
            f"or the effective sample size is insufficient. "
            f"\n\n**Actionable Recommendation:** Switch to the non-parametric **Exact Sign Test**. "
            f"The Sign Test makes zero assumptions about distribution symmetry, evaluating purely whether observations "
            f"are equally likely to fall above or below the benchmark ({benchmark:.1f}). "
            f"\n\n**Alternative Test Result on this Data:** "
            f"Positive Differences: {pos_count}/{n_eff}, **Sign Test p-value = {p_sign_test:.4f}**."
        )
    elif has_warning:
        overall_status = "warning"
        overall_verdict_badge = "🟡 Assumptions Borderline: Proceed with Caution"
        overall_recommendation = (
            f"Moderate asymmetry of differences is present (Skewness = {skew_diffs:.2f}). "
            f"The Wilcoxon Signed-Rank Test remains moderately robust, but technically tests the Hodges-Lehmann pseudo-median "
            f"(Walsh averages: {hl_median:.2f}) rather than the pure sample median ({calc_median:.2f}). "
            f"\n\n**Actionable Recommendation:** You may proceed with the **Wilcoxon Signed-Rank Test** (p = {p_wilcoxon:.4f}), "
            f"but compare with the **Exact Sign Test** (p = {p_sign_test:.4f}) for consistency."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ All Assumptions Met: Proceed with Wilcoxon Signed-Rank Test"
        overall_recommendation = (
            f"All core assumptions (ordinal/continuous scale, independent observations, symmetric differences, "
            f"and adequate non-zero sample size) are fully satisfied. "
            f"\n\n**Actionable Recommendation:** Proceed with confidence using the **Wilcoxon Signed-Rank Test**. "
            f"The test has optimal non-parametric power: W = {w_stat:.1f}, p = {p_wilcoxon:.4f}, "
            f"Hodges-Lehmann Median Shift = {hl_estimate:+.2f}."
        )

    results = {
        "n": int(n),
        "n_eff": int(n_eff),
        "n_zeros": int(n_zeros),
        "benchmark": float(benchmark),
        "median": calc_median,
        "mean": calc_mean,
        "median_diff": median_diff,
        "w_stat": w_stat,
        "sum_pos_ranks": sum_pos_ranks,
        "sum_neg_ranks": sum_neg_ranks,
        "p_wilcoxon": p_wilcoxon,
        "hl_estimate": hl_estimate,
        "hl_median": hl_median,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "p_sign_test": p_sign_test,
        "skew_diffs": skew_diffs,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df, results


def generate_wilcoxon_paired_demo(
    n_pairs: int = 30,
    median_pre: float = 140.0,
    median_diff: float = -6.0,
    scale_diff: float = 8.0,
    data_type: str = "continuous_non_normal",
    dist_shape: str = "symmetric_non_normal",
    zero_diff_count: int = 0,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic data for the Paired Wilcoxon Signed-Rank Test with full assumption diagnostics.

    Supports BOTH continuous non-normal paired data (e.g. biological/clinical measurements)
    and ordinal paired data (e.g. 5-point or 7-point Likert ratings Pre vs Post).

    Args:
        n_pairs: Number of matched pairs / subjects.
        median_pre: Baseline/pre-intervention target median.
        median_diff: Target median difference (Post - Pre).
        scale_diff: Scale / dispersion parameter for differences.
        data_type: 'continuous_non_normal' or 'ordinal_likert'.
        dist_shape: 'symmetric_non_normal' (Laplace), 'moderately_skewed' (Gamma), or 'severely_skewed' (Log-normal).
        zero_diff_count: Number of pairs with exact zero difference (Post == Pre).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (combined_DataFrame, results_dict).
    """
    rng = np.random.default_rng(seed)

    # 1. Generate data based on variable type and distribution shape
    if data_type == "ordinal_likert":
        # Simulate 7-point Likert scale (1 = Strongly Disagree to 7 = Strongly Agree)
        pre = rng.integers(1, 8, size=n_pairs).astype(float)
        # Shift centered around integer median_diff
        raw_shift = np.round(rng.normal(loc=median_diff, scale=max(1.0, scale_diff / 3.0), size=n_pairs))
        post = np.clip(pre + raw_shift, 1.0, 7.0)
        diff = post - pre
    else:
        # Continuous non-normal data
        pre = np.round(rng.normal(loc=median_pre, scale=15.0, size=n_pairs), 1)
        if dist_shape == "symmetric_non_normal":
            # Laplace distribution: symmetric, heavy-tailed non-normal (ideal for Wilcoxon)
            raw_diff = rng.laplace(loc=0.0, scale=scale_diff / 1.414, size=n_pairs)
            diff = np.round(raw_diff + median_diff, 1)
        elif dist_shape == "moderately_skewed":
            # Gamma distribution: moderately right-skewed
            raw_diff = rng.gamma(shape=3.0, scale=scale_diff / 3.0, size=n_pairs)
            raw_med = float(np.median(raw_diff))
            diff = np.round((raw_diff - raw_med) + median_diff, 1)
        elif dist_shape == "severely_skewed":
            # Log-normal distribution: severely right-skewed
            raw_diff = rng.lognormal(mean=0.0, sigma=1.0, size=n_pairs)
            raw_med = float(np.median(raw_diff))
            diff = np.round(((raw_diff - raw_med) / (raw_med if raw_med > 0 else 1.0)) * scale_diff + median_diff, 1)
        else:
            diff = np.round(rng.normal(loc=median_diff, scale=scale_diff, size=n_pairs), 1)

        post = np.round(pre + diff, 1)
        diff = np.round(post - pre, 1)

    # 2. Inject exact zero differences (ties where Post == Pre) if requested
    if zero_diff_count > 0:
        zero_indices = rng.choice(n_pairs, size=min(zero_diff_count, n_pairs), replace=False)
        for idx in zero_indices:
            post[idx] = pre[idx]
            diff[idx] = 0.0

    # 3. Compute Ranks on Non-zero Absolute Differences
    abs_diff = np.abs(diff)
    nonzero_mask = diff != 0.0
    nonzero_diffs = diff[nonzero_mask]
    nonzero_abs = abs_diff[nonzero_mask]
    n_eff = len(nonzero_diffs)
    n_zeros = n_pairs - n_eff

    if n_eff > 0:
        ranks_nonzero = stats.rankdata(nonzero_abs)
        pos_ranks = np.where(nonzero_diffs > 0, ranks_nonzero, 0.0)
        neg_ranks = np.where(nonzero_diffs < 0, ranks_nonzero, 0.0)
        sum_pos_ranks = float(np.sum(pos_ranks))
        sum_neg_ranks = float(np.sum(neg_ranks))
        w_stat = float(min(sum_pos_ranks, sum_neg_ranks))
    else:
        ranks_nonzero = np.array([])
        pos_ranks = np.array([])
        neg_ranks = np.array([])
        sum_pos_ranks = 0.0
        sum_neg_ranks = 0.0
        w_stat = 0.0

    # Full rank vector for DataFrame
    full_ranks = np.zeros(n_pairs, dtype=float)
    if n_eff > 0:
        full_ranks[nonzero_mask] = ranks_nonzero

    full_pos_ranks = np.zeros(n_pairs, dtype=float)
    full_neg_ranks = np.zeros(n_pairs, dtype=float)
    if n_eff > 0:
        full_pos_ranks[nonzero_mask] = pos_ranks
        full_neg_ranks[nonzero_mask] = neg_ranks

    df = pd.DataFrame({
        "Subject_ID": [f"Subj_{i+1:02d}" for i in range(n_pairs)],
        "Pre": pre.astype(float),
        "Post": post.astype(float),
        "Difference": diff.astype(float),
        "Abs_Difference": abs_diff.astype(float),
        "Wilcoxon_Rank": full_ranks,
        "Pos_Rank": full_pos_ranks,
        "Neg_Rank": full_neg_ranks
    })

    # 4. Statistical Metrics
    calc_med_pre = float(np.median(pre))
    calc_med_post = float(np.median(post))
    calc_med_diff = float(np.median(diff))
    calc_mean_diff = float(np.mean(diff))

    # Wilcoxon test via scipy
    if n_eff >= 5:
        try:
            res_wilc = stats.wilcoxon(nonzero_diffs, alternative="two-sided")
            p_wilcoxon = float(res_wilc.pvalue)
            scipy_stat = float(res_wilc.statistic)
        except Exception:
            scipy_stat = w_stat
            p_wilcoxon = 1.0
    else:
        scipy_stat = w_stat
        p_wilcoxon = 1.0

    # Large-sample Z approximation with continuity correction
    mu_w = (n_eff * (n_eff + 1)) / 4.0
    sigma_w = np.sqrt((n_eff * (n_eff + 1) * (2 * n_eff + 1)) / 24.0) if n_eff > 1 else 1.0
    z_stat = (abs(w_stat - mu_w) - 0.5) / sigma_w if sigma_w > 0 else 0.0

    # Effect sizes:
    # Rank-biserial correlation r_rb = (W+ - W-) / (W+ + W-)
    sum_ranks_tot = sum_pos_ranks + sum_neg_ranks
    r_rb = float((sum_pos_ranks - sum_neg_ranks) / sum_ranks_tot) if sum_ranks_tot > 0 else 0.0
    # Correlation effect size r = Z / sqrt(N_eff)
    r_effect = float(z_stat / np.sqrt(n_eff)) if n_eff > 0 else 0.0

    # Hodges-Lehmann median difference (median of all pairwise Walsh averages)
    if n_eff > 0:
        walsh_diffs = [(nonzero_diffs[i] + nonzero_diffs[j]) / 2.0 for i in range(n_eff) for j in range(i, n_eff)]
        hl_estimate = float(np.median(walsh_diffs))
    else:
        hl_estimate = 0.0

    # Exact Sign Test Alternative
    pos_count = int(np.sum(nonzero_diffs > 0))
    neg_count = int(np.sum(nonzero_diffs < 0))
    if n_eff > 0:
        sign_res = stats.binomtest(pos_count, n=n_eff, p=0.5, alternative="two-sided")
        p_sign_test = float(sign_res.pvalue)
    else:
        p_sign_test = 1.0

    # Paired t-test comparison
    try:
        t_res = stats.ttest_rel(post, pre)
        t_stat = float(t_res.statistic)
        p_ttest = float(t_res.pvalue)
    except Exception:
        t_stat, p_ttest = 0.0, 1.0

    # Difference Skewness (Symmetry Check)
    skew_diffs = float(stats.skew(nonzero_diffs)) if n_eff > 2 else 0.0

    # 5. Core Assumption Evaluations
    # A. Paired Dependent Design
    asm_design = {
        "name": "1. Paired / Dependent Study Design",
        "description": "Measurements must consist of matched pairs or repeated observations from identical subjects.",
        "test_applied": "Design Structure Verification",
        "diagnostic_value": f"{n_pairs} Matched Pairs (Pre & Post)",
        "status": "pass",
        "verdict": "PASS",
        "detail": "Data satisfies the dependent repeated-measures requirement. Each pair shares subject-specific variance."
    }

    # B. Measurement Scale
    scale_label = "Ordinal (Likert Scale)" if data_type == "ordinal_likert" else "Continuous Non-Normal"
    asm_scale = {
        "name": "2. Measurement Scale (At Least Ordinal)",
        "description": "Outcome variable must be at least ordinal so differences can be ordered and ranked.",
        "test_applied": "Scale Type Inspection",
        "diagnostic_value": f"{scale_label} Scale",
        "status": "pass",
        "verdict": "PASS",
        "detail": (
            f"Data type is {scale_label}. Ranking pairwise absolute differences preserves monotonic order "
            f"without requiring metric interval or normality assumptions."
        )
    }

    # C. Symmetry of Differences
    if abs(skew_diffs) <= 0.8:
        sym_status = "pass"
        sym_verdict = "PASS (SYMMETRIC)"
        sym_detail = (
            f"Skewness of difference scores = {skew_diffs:+.2f} (within [-0.8, +0.8]). "
            f"Differences are approximately symmetric around the median difference; the Wilcoxon test "
            f"validly tests the true population median difference."
        )
    elif abs(skew_diffs) <= 1.5:
        sym_status = "warning"
        sym_verdict = "CAUTION (MODERATE ASYMMETRY)"
        sym_detail = (
            f"Skewness of difference scores = {skew_diffs:+.2f} (between 0.8 and 1.5). "
            f"Distribution of differences is moderately skewed. The Wilcoxon test evaluates the "
            f"Hodges-Lehmann pseudo-median ({hl_estimate:+.2f}). Check concordance with the Exact Sign Test."
        )
    else:
        sym_status = "fail"
        sym_verdict = "VIOLATED (SEVERE ASYMMETRY)"
        sym_detail = (
            f"Skewness of difference scores = {skew_diffs:+.2f} (> 1.5). "
            f"Severe asymmetry invalidates the pure median shift interpretation because rank sums confound "
            f"location with skewness. Switch to the Exact Sign Test."
        )

    asm_sym = {
        "name": "3. Distribution Symmetry of Difference Scores",
        "description": "Pairwise differences should be approximately symmetric around their median.",
        "test_applied": "Sample Skewness of Differences",
        "diagnostic_value": f"Skewness = {skew_diffs:+.3f}",
        "status": sym_status,
        "verdict": sym_verdict,
        "detail": sym_detail
    }

    # D. Effective Sample Size & Zero Ties
    if n_eff >= 10:
        eff_status = "pass"
        eff_verdict = "PASS"
        eff_detail = f"Effective sample size N_eff = {n_eff} (after dropping {n_zeros} zero differences) provides adequate power and reliable ranking."
    elif n_eff >= 6:
        eff_status = "warning"
        eff_verdict = "CAUTION (LOW EFFECTIVE N)"
        eff_detail = f"Effective sample size N_eff = {n_eff} is low after dropping {n_zeros} ties. Prefer exact distribution over normal Z approximation."
    else:
        eff_status = "fail"
        eff_verdict = "VIOLATED (INSUFFICIENT DATA)"
        eff_detail = f"Effective sample size N_eff = {n_eff} is too low for reliable hypothesis testing."

    asm_eff = {
        "name": "4. Effective Sample Size & Zero Differences",
        "description": "Adequate non-zero paired differences (ties with diff = 0 are dropped from ranking).",
        "test_applied": "Count of Non-Zero Differences (N_eff)",
        "diagnostic_value": f"N_eff = {n_eff} (Ties = {n_zeros})",
        "status": eff_status,
        "verdict": eff_verdict,
        "detail": eff_detail
    }

    assumptions_eval = [asm_design, asm_scale, asm_sym, asm_eff]

    # Overall recommendation synthesis
    has_failure = any(a["status"] == "fail" for a in assumptions_eval)
    has_warning = any(a["status"] == "warning" for a in assumptions_eval)

    if has_failure:
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption(s) Violated: Switch to Exact Sign Test"
        overall_recommendation = (
            f"The symmetry assumption for difference scores is severely violated (Skewness = {skew_diffs:+.2f}). "
            f"Under severe asymmetry, Wilcoxon rank sums confound location with shape. "
            f"\n\n**Actionable Recommendation:** Switch to the non-parametric **Exact Sign Test**, "
            f"which makes zero symmetry assumptions and tests purely whether positive changes outnumber negative changes. "
            f"\n\n**Sign Test Result on this Data:** "
            f"Positive changes = {pos_count}, Negative changes = {neg_count}, **p-value = {p_sign_test:.4f}**."
        )
    elif has_warning:
        overall_status = "warning"
        overall_verdict_badge = "🟡 Assumptions Borderline: Proceed with Caution"
        overall_recommendation = (
            f"Moderate asymmetry detected in difference scores (Skewness = {skew_diffs:+.2f}). "
            f"The Wilcoxon Signed-Rank Test evaluates the Hodges-Lehmann median difference ({hl_estimate:+.2f}) rather than a pure median shift. "
            f"\n\n**Actionable Recommendation:** You may proceed with the **Wilcoxon Signed-Rank Test (Paired)** (p = {p_wilcoxon:.4f}), "
            f"but compare results with the **Exact Sign Test** (p = {p_sign_test:.4f}) to confirm robustness."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ All Assumptions Met: Proceed with Wilcoxon Signed-Rank Test (Paired)"
        overall_recommendation = (
            f"All core assumptions (paired design, ordinal/continuous scale, symmetric differences, "
            f"and adequate non-zero sample size) are fully satisfied. "
            f"\n\n**Actionable Recommendation:** Proceed with confidence using the **Wilcoxon Signed-Rank Test (Paired)**. "
            f"The test provides optimal non-parametric power: W = {w_stat:.1f}, p = {p_wilcoxon:.4f}, "
            f"Hodges-Lehmann Median Difference = {hl_estimate:+.2f}, Rank-Biserial Effect Size r_rb = {r_rb:+.2f}."
        )

    results = {
        "n_pairs": int(n_pairs),
        "n_eff": int(n_eff),
        "n_zeros": int(n_zeros),
        "data_type": data_type,
        "median_pre": calc_med_pre,
        "median_post": calc_med_post,
        "median_diff": calc_med_diff,
        "mean_diff": calc_mean_diff,
        "w_stat": w_stat,
        "scipy_stat": scipy_stat,
        "sum_pos_ranks": sum_pos_ranks,
        "sum_neg_ranks": sum_neg_ranks,
        "p_wilcoxon": p_wilcoxon,
        "z_stat": float(z_stat),
        "r_rb": r_rb,
        "r_effect": r_effect,
        "hl_estimate": hl_estimate,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "p_sign_test": p_sign_test,
        "t_stat": t_stat,
        "p_ttest": p_ttest,
        "skew_diffs": skew_diffs,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df, results


def generate_mann_whitney_demo(
    n_a: int = 30,
    n_b: int = 30,
    median_a: float = 100.0,
    median_b: float = 112.0,
    scale_a: float = 15.0,
    scale_b: float = 15.0,
    dist_shape: str = "symmetric_non_normal",
    group_a_name: str = "Group A (Control)",
    group_b_name: str = "Group B (Treatment)",
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic data for two independent groups and executes the Mann-Whitney U Test.

    Args:
        n_a: Sample size for Group A.
        n_b: Sample size for Group B.
        median_a: Target median for Group A.
        median_b: Target median for Group B.
        scale_a: Spread / dispersion parameter for Group A.
        scale_b: Spread / dispersion parameter for Group B.
        dist_shape: Distribution shape ('symmetric_non_normal', 'moderately_skewed', 'unequal_shapes_variances').
        group_a_name: Label for Group A.
        group_b_name: Label for Group B.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (combined_DataFrame, results_dict).
    """
    rng = np.random.default_rng(seed)

    # 1. Generate distributions based on selected shape
    if dist_shape == "symmetric_non_normal":
        # Laplace distributions (symmetric, heavier-tailed than normal)
        raw_a = rng.laplace(loc=0.0, scale=1.0, size=n_a)
        raw_b = rng.laplace(loc=0.0, scale=1.0, size=n_b)
        vals_a = (raw_a - np.median(raw_a)) * (scale_a / 1.414) + median_a
        vals_b = (raw_b - np.median(raw_b)) * (scale_b / 1.414) + median_b

    elif dist_shape == "moderately_skewed":
        # Right-skewed Gamma distributions with identical shape in both groups
        raw_a = rng.gamma(shape=2.5, scale=1.0, size=n_a)
        raw_b = rng.gamma(shape=2.5, scale=1.0, size=n_b)
        vals_a = (raw_a - np.median(raw_a)) * (scale_a / np.std(raw_a)) + median_a
        vals_b = (raw_b - np.median(raw_b)) * (scale_b / np.std(raw_b)) + median_b

    else:  # unequal_shapes_variances
        # Group A is symmetric/normal with small scale; Group B is heavily skewed log-normal with larger scale
        raw_a = rng.normal(loc=0.0, scale=1.0, size=n_a)
        raw_b = rng.lognormal(mean=0.0, sigma=1.1, size=n_b)
        vals_a = (raw_a - np.median(raw_a)) * (scale_a / np.std(raw_a)) + median_a
        vals_b = (raw_b - np.median(raw_b)) * (scale_b * 2.2 / np.std(raw_b)) + median_b

    # 2. Construct DataFrames
    df_a = pd.DataFrame({
        "Observation_ID": [f"A_{i+1:02d}" for i in range(n_a)],
        "Group": group_a_name,
        "Value": vals_a
    })
    df_b = pd.DataFrame({
        "Observation_ID": [f"B_{j+1:02d}" for j in range(n_b)],
        "Group": group_b_name,
        "Value": vals_b
    })
    df = pd.concat([df_a, df_b], ignore_index=True)

    # 3. Descriptive metrics
    med_a = float(np.median(vals_a))
    med_b = float(np.median(vals_b))
    mean_a = float(np.mean(vals_a))
    mean_b = float(np.mean(vals_b))
    std_a = float(np.std(vals_a, ddof=1))
    std_b = float(np.std(vals_b, ddof=1))
    q25_a, q75_a = float(np.percentile(vals_a, 25)), float(np.percentile(vals_a, 75))
    q25_b, q75_b = float(np.percentile(vals_b, 25)), float(np.percentile(vals_b, 75))
    iqr_a = q75_a - q25_a
    iqr_b = q75_b - q25_b
    median_diff = med_b - med_a

    # 4. Combined Ranks & Rank Sums
    combined_vals = np.concatenate([vals_a, vals_b])
    combined_ranks = stats.rankdata(combined_vals)
    ranks_a = combined_ranks[:n_a]
    ranks_b = combined_ranks[n_a:]
    r_a = float(np.sum(ranks_a))
    r_b = float(np.sum(ranks_b))

    df["Rank"] = combined_ranks

    # 5. Mann-Whitney U Calculations
    mw_res = stats.mannwhitneyu(vals_b, vals_a, alternative="two-sided")
    u_b = float(mw_res.statistic)
    p_mann = float(mw_res.pvalue)
    u_a = float(n_a * n_b - u_b)
    u_stat = float(min(u_a, u_b))

    # Effect Sizes
    # Rank-biserial correlation: r_rb = (U_B - U_A) / (n_A * n_B)
    r_rb = float((u_b - u_a) / (n_a * n_b))
    # Common Language Effect Size (Probability of Superiority: P(B > A))
    cles = float(u_b / (n_a * n_b))

    # Large-sample normal approximation Z-score
    mu_u = float(n_a * n_b / 2.0)
    sigma_u = float(np.sqrt((n_a * n_b * (n_a + n_b + 1)) / 12.0))
    z_stat = float((abs(u_stat - mu_u) - 0.5) / sigma_u) if sigma_u > 0 else 0.0
    r_effect = float(z_stat / np.sqrt(n_a + n_b))

    # Hodges-Lehmann difference (median of all pairwise differences B_j - A_i)
    pairwise_diffs = np.subtract.outer(vals_b, vals_a).flatten()
    hl_diff = float(np.median(pairwise_diffs))

    # 6. Diagnostics & Alternative Test Computations
    lev_stat, p_levene = stats.levene(vals_a, vals_b)
    lev_stat, p_levene = float(lev_stat), float(p_levene)

    skew_a = float(stats.skew(vals_a))
    skew_b = float(stats.skew(vals_b))
    delta_skew = float(abs(skew_b - skew_a))

    # Alternative Test 1: Welch's t-test
    t_welch, p_welch = stats.ttest_ind(vals_b, vals_a, equal_var=False)
    t_welch, p_welch = float(t_welch), float(p_welch)

    # Alternative Test 2: Brunner-Munzel test (robust to unequal variances & shapes)
    try:
        bm_res = stats.brunnermunzel(vals_b, vals_a, alternative="two-sided")
        bm_stat = float(bm_res.statistic)
        p_bm = float(bm_res.pvalue)
    except Exception:
        bm_stat = float("nan")
        p_bm = float("nan")

    # 7. Diagnostic Evaluation of 4 Core Assumptions
    # Assumption 1: Scale of Measurement
    scale_status = "pass"
    scale_verdict = "SATISFIED"
    scale_detail = (
        "The dependent variable is measured on a continuous scale (or ordinal ranks). "
        "Values can be ordered without loss of rank information."
    )

    # Assumption 2: Independence of Observations
    indep_status = "pass"
    indep_verdict = "SATISFIED"
    indep_detail = (
        f"Data consists of two independent cohorts ({group_a_name}: n={n_a}, {group_b_name}: n={n_b}) "
        "with no pairing, matching, or repeated measures across groups."
    )

    # Assumption 3: Distribution Shape Similarity & Dispersion
    if p_levene >= 0.05 and delta_skew <= 0.8:
        shape_status = "pass"
        shape_verdict = "PASSED (Similar Shapes)"
        shape_detail = (
            f"Both groups exhibit homogeneous variance (Levene's p = {p_levene:.3f} >= 0.05) and "
            f"comparable skewness (|ΔSkew| = {delta_skew:.2f} <= 0.8). "
            f"Because distribution shapes are similar, Mann-Whitney U validly evaluates a shift in group medians."
        )
    elif p_levene >= 0.01 and delta_skew <= 1.5:
        shape_status = "warning"
        shape_verdict = "CAUTION (Moderate Shape Asymmetry)"
        shape_detail = (
            f"Moderate dispersion difference (Levene's p = {p_levene:.3f}) or skewness discrepancy "
            f"(|ΔSkew| = {delta_skew:.2f}) observed. The test remains valid but evaluates general "
            f"stochastic dominance (P(B > A) ≠ 0.5) rather than a pure difference in population medians."
        )
    else:
        shape_status = "fail"
        shape_verdict = "VIOLATED (Heteroscedastic / Differing Shapes)"
        shape_detail = (
            f"Severe variance inequality (Levene's p = {p_levene:.4f} < 0.01) and shape disparity "
            f"(|ΔSkew| = {delta_skew:.2f} > 1.5) detected. Under heteroscedasticity, standard Mann-Whitney U "
            f"can experience inflated Type I error. The Brunner-Munzel test or Welch's t-test is strongly recommended."
        )

    # Assumption 4: Adequate Sample Size
    min_n = min(n_a, n_b)
    if min_n >= 20:
        size_status = "pass"
        size_verdict = "PASSED (Large Sample Size)"
        size_detail = (
            f"Both group sample sizes (n_A = {n_a}, n_B = {n_b}) are >= 20. "
            "The large-sample normal approximation with continuity correction provides robust and accurate p-values."
        )
    elif min_n >= 10:
        size_status = "warning"
        size_verdict = "CAUTION (Moderate Sample Size)"
        size_detail = (
            f"Smallest group size is {min_n} (>= 10). Normal approximation is acceptable with continuity correction, "
            "but exact permutation distribution or tables are preferred."
        )
    else:
        size_status = "fail"
        size_verdict = "VIOLATED (Small Sample Size)"
        size_detail = (
            f"Smallest group size is {min_n} (< 10). Asymptotic approximation is unreliable; "
            "exact permutation test must be used."
        )

    assumptions_eval = [
        {
            "name": "1. Measurement Scale (Ordinal or Continuous)",
            "description": "Data values must be capable of being ranked without ambiguous ordering.",
            "test_applied": "Variable type inspection",
            "diagnostic_value": "Continuous / Ordinal",
            "status": scale_status,
            "verdict": scale_verdict,
            "detail": scale_detail
        },
        {
            "name": "2. Independence of Observations",
            "description": "Subjects in Group A and Group B must be mutually independent random samples.",
            "test_applied": "Study design evaluation",
            "diagnostic_value": "2 Independent cohorts",
            "status": indep_status,
            "verdict": indep_verdict,
            "detail": indep_detail
        },
        {
            "name": "3. Distribution Shape & Variance Similarity",
            "description": "Shapes and dispersions must be similar to interpret the result as a difference in medians.",
            "test_applied": "Levene's Test (p-value) & Absolute Skewness Difference (|ΔSkew|)",
            "diagnostic_value": f"Levene p = {p_levene:.3f}, |ΔSkew| = {delta_skew:.2f}",
            "status": shape_status,
            "verdict": shape_verdict,
            "detail": shape_detail
        },
        {
            "name": "4. Sample Size & Approximation Adequacy",
            "description": "Minimum group size dictates whether asymptotic Z-approximation or exact test applies.",
            "test_applied": "Minimum group size min(n_A, n_B)",
            "diagnostic_value": f"n_A = {n_a}, n_B = {n_b} (min = {min_n})",
            "status": size_status,
            "verdict": size_verdict,
            "detail": size_detail
        }
    ]

    # 8. Overall Recommendation Synthesis
    has_failure = any(a["status"] == "fail" for a in assumptions_eval)
    has_warning = any(a["status"] == "warning" for a in assumptions_eval)

    if has_failure:
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption(s) Violated: Switch to Alternative Method"
        overall_recommendation = (
            f"Severe heteroscedasticity or shape divergence was detected between groups "
            f"(Levene's p = {p_levene:.4f}, |ΔSkew| = {delta_skew:.2f}). "
            f"Under unequal variances and shapes, the standard Mann-Whitney U test can produce false positives. "
            f"\n\n**Actionable Recommendation:** Switch to the **Brunner-Munzel Test** "
            f"(generalized Wilcoxon test robust against unequal variances) or **Welch's Two-Sample t-Test** "
            f"(if sample sizes are sufficient for Central Limit Theorem). "
            f"\n\n**Alternative Test Results on this Data:** "
            f"- Brunner-Munzel Test: Statistic = {bm_stat:.3f}, **p-value = {p_bm:.4f}** "
            f"- Welch's t-test: t = {t_welch:.3f}, **p-value = {p_welch:.4f}**"
        )
    elif has_warning:
        overall_status = "warning"
        overall_verdict_badge = "🟡 Assumptions Borderline: Proceed with Caution / Stochastic Dominance"
        overall_recommendation = (
            f"Moderate dispersion or skewness differences exist between groups (|ΔSkew| = {delta_skew:.2f}). "
            f"The Mann-Whitney U test is valid, but the result should be reported as **stochastic superiority** "
            f"(Common Language Effect Size CLES = {cles:.3f}, Probability of B > A) rather than a pure shift in medians. "
            f"\n\n**Actionable Recommendation:** Proceed with **Mann-Whitney U Test** (p = {p_mann:.4f}), "
            f"reporting both the Hodges-Lehmann difference ({hl_diff:+.2f}) and CLES ({cles:.3f})."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ All Assumptions Met: Proceed with Mann-Whitney U Test"
        overall_recommendation = (
            f"All core assumptions (independent groups, continuous/ordinal scale, similar distribution shapes, "
            f"and adequate sample sizes) are fully satisfied. "
            f"\n\n**Actionable Recommendation:** Proceed with confidence using the **Mann-Whitney U Test**. "
            f"The test has optimal non-parametric power: U = {u_stat:.1f}, p = {p_mann:.4f}, "
            f"Hodges-Lehmann Median Difference = {hl_diff:+.2f}, Rank-Biserial r = {r_rb:+.3f}."
        )

    results = {
        "n_a": int(n_a),
        "n_b": int(n_b),
        "n_total": int(n_a + n_b),
        "group_a_name": group_a_name,
        "group_b_name": group_b_name,
        "median_a": med_a,
        "median_b": med_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "std_a": std_a,
        "std_b": std_b,
        "iqr_a": iqr_a,
        "iqr_b": iqr_b,
        "median_diff": median_diff,
        "hl_diff": hl_diff,
        "rank_sum_a": r_a,
        "rank_sum_b": r_b,
        "u_a": u_a,
        "u_b": u_b,
        "u_stat": u_stat,
        "p_mann": p_mann,
        "r_rb": r_rb,
        "cles": cles,
        "z_stat": z_stat,
        "r_effect": r_effect,
        "levene_stat": lev_stat,
        "p_levene": p_levene,
        "skew_a": skew_a,
        "skew_b": skew_b,
        "delta_skew": delta_skew,
        "t_welch": t_welch,
        "p_welch": p_welch,
        "bm_stat": bm_stat,
        "p_bm": p_bm,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df, results


def generate_welch_demo(
    n_a: int = 35,
    n_b: int = 35,
    mean_a: float = 100.0,
    mean_b: float = 108.0,
    std_a: float = 10.0,
    std_b: float = 18.0,
    dist_shape: str = "normal",
    outlier_count_b: int = 0,
    group_a_name: str = "Group A (Control)",
    group_b_name: str = "Group B (Treatment)",
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic data for two independent groups and executes Welch's t-test with diagnostics.

    Args:
        n_a: Sample size for Group A.
        n_b: Sample size for Group B.
        mean_a: Target mean for Group A.
        mean_b: Target mean for Group B.
        std_a: Target standard deviation for Group A.
        std_b: Target standard deviation for Group B (supports heteroscedasticity).
        dist_shape: Distribution shape ('normal', 'skewed_moderate', 'heavy_tailed').
        outlier_count_b: Number of extreme outliers to inject into Group B (0, 1, 2, 3).
        group_a_name: Label for Group A.
        group_b_name: Label for Group B.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (combined_DataFrame, results_dict).
    """
    rng = np.random.default_rng(seed)

    # 1. Generate base distribution shapes
    if dist_shape == "skewed_moderate":
        # Right-skewed Gamma distributions
        raw_a = rng.gamma(shape=2.5, scale=1.0, size=n_a)
        raw_b = rng.gamma(shape=2.5, scale=1.0, size=n_b)
    elif dist_shape == "heavy_tailed":
        # Heavy-tailed Student's t distribution (df=3)
        raw_a = rng.standard_t(df=3, size=n_a)
        raw_b = rng.standard_t(df=3, size=n_b)
    else:  # normal
        raw_a = rng.normal(loc=0.0, scale=1.0, size=n_a)
        raw_b = rng.normal(loc=0.0, scale=1.0, size=n_b)

    # Standardize and scale to specified means & standard deviations
    vals_a = (raw_a - np.mean(raw_a)) / (np.std(raw_a) if np.std(raw_a) > 0 else 1.0) * std_a + mean_a
    vals_b = (raw_b - np.mean(raw_b)) / (np.std(raw_b) if np.std(raw_b) > 0 else 1.0) * std_b + mean_b

    # 2. Inject extreme outliers if requested
    if outlier_count_b > 0:
        for i in range(min(outlier_count_b, n_b)):
            vals_b[-(i + 1)] = mean_b + 4.5 * std_b

    # 3. Construct DataFrame
    df_a = pd.DataFrame({
        "Observation_ID": [f"A_{i+1:02d}" for i in range(n_a)],
        "Group": group_a_name,
        "Value": vals_a
    })
    df_b = pd.DataFrame({
        "Observation_ID": [f"B_{j+1:02d}" for j in range(n_b)],
        "Group": group_b_name,
        "Value": vals_b
    })
    df = pd.concat([df_a, df_b], ignore_index=True)

    # 4. Descriptive Summary Metrics
    calc_mean_a = float(np.mean(vals_a))
    calc_mean_b = float(np.mean(vals_b))
    calc_std_a = float(np.std(vals_a, ddof=1))
    calc_std_b = float(np.std(vals_b, ddof=1))
    var_a = calc_std_a ** 2
    var_b = calc_std_b ** 2
    calc_med_a = float(np.median(vals_a))
    calc_med_b = float(np.median(vals_b))
    q25_a, q75_a = float(np.percentile(vals_a, 25)), float(np.percentile(vals_a, 75))
    q25_b, q75_b = float(np.percentile(vals_b, 25)), float(np.percentile(vals_b, 75))
    iqr_a = q75_a - q25_a
    iqr_b = q75_b - q25_b
    mean_diff = calc_mean_b - calc_mean_a

    # 5. Welch's t-test Calculations
    se_diff = float(np.sqrt(var_a / n_a + var_b / n_b))
    t_stat_val = float(mean_diff / se_diff) if se_diff > 0 else 0.0

    # Welch-Satterthwaite degrees of freedom
    num_df = (var_a / n_a + var_b / n_b) ** 2
    denom_df = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    df_welch = float(num_df / denom_df) if denom_df > 0 else float(n_a + n_b - 2)

    # Two-tailed p-value
    p_val_welch = float(2.0 * stats.t.sf(abs(t_stat_val), df=df_welch))

    # 95% Confidence Interval for Difference (mu_B - mu_A)
    t_crit = float(stats.t.ppf(0.975, df=df_welch))
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Standardized Effect Sizes
    s_avg = float(np.sqrt((var_a + var_b) / 2.0))
    cohens_d = float(mean_diff / s_avg) if s_avg > 0 else 0.0
    hedges_g = float(cohens_d * (1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)))

    # 6. Diagnostic Computations
    shapiro_w_a, shapiro_p_a = stats.shapiro(vals_a)
    shapiro_w_b, shapiro_p_b = stats.shapiro(vals_b)
    shapiro_p_a, shapiro_p_b = float(shapiro_p_a), float(shapiro_p_b)
    skew_a, skew_b = float(stats.skew(vals_a)), float(stats.skew(vals_b))

    # Tukey's Outlier Diagnostics
    outliers_a = vals_a[(vals_a < q25_a - 1.5 * iqr_a) | (vals_a > q75_a + 1.5 * iqr_a)]
    ext_outliers_a = vals_a[(vals_a < q25_a - 3.0 * iqr_a) | (vals_a > q75_a + 3.0 * iqr_a)]
    outliers_b = vals_b[(vals_b < q25_b - 1.5 * iqr_b) | (vals_b > q75_b + 1.5 * iqr_b)]
    ext_outliers_b = vals_b[(vals_b < q25_b - 3.0 * iqr_b) | (vals_b > q75_b + 3.0 * iqr_b)]
    total_mild = len(outliers_a) + len(outliers_b)
    total_ext = len(ext_outliers_a) + len(ext_outliers_b)

    # Levene's Test (informational check for variance equality)
    lev_stat, p_levene = stats.levene(vals_a, vals_b)
    lev_stat, p_levene = float(lev_stat), float(p_levene)

    # Non-parametric Alternative Test (Mann-Whitney U)
    mw_res = stats.mannwhitneyu(vals_b, vals_a, alternative="two-sided")
    u_stat = float(min(mw_res.statistic, n_a * n_b - mw_res.statistic))
    p_mann = float(mw_res.pvalue)

    # Q-Q Plot Coordinates for Plotly
    prob_a = stats.probplot(vals_a, dist="norm")
    qq_osm_a = prob_a[0][0].tolist()
    qq_osr_a = prob_a[0][1].tolist()
    qq_slope_a = float(prob_a[1][0])
    qq_intercept_a = float(prob_a[1][1])

    prob_b = stats.probplot(vals_b, dist="norm")
    qq_osm_b = prob_b[0][0].tolist()
    qq_osr_b = prob_b[0][1].tolist()
    qq_slope_b = float(prob_b[1][0])
    qq_intercept_b = float(prob_b[1][1])

    # 7. Diagnostic Evaluations for 4 Core Assumptions
    # Assumption 1: Scale of Measurement
    scale_status = "pass"
    scale_verdict = "SATISFIED"
    scale_detail = (
        "The dependent variable is measured on a continuous scale (interval or ratio), "
        "allowing arithmetic operations such as means, standard deviations, and variances."
    )

    # Assumption 2: Independence of Observations
    indep_status = "pass"
    indep_verdict = "SATISFIED"
    indep_detail = (
        f"Data consists of two independent samples ({group_a_name}: n={n_a}, {group_b_name}: n={n_b}) "
        "with mutually exclusive subjects and no paired matching or repeated measurements."
    )

    # Assumption 3: Normality within each group
    both_normal = (shapiro_p_a >= 0.05 and shapiro_p_b >= 0.05)
    clt_applies = (n_a >= 30 and n_b >= 30 and abs(skew_a) <= 1.2 and abs(skew_b) <= 1.2)

    if both_normal:
        norm_status = "pass"
        norm_verdict = "PASSED (Normal Distributions)"
        norm_detail = (
            f"Both groups are approximately normally distributed (Shapiro-Wilk: Group A p = {shapiro_p_a:.3f}, "
            f"Group B p = {shapiro_p_b:.3f} >= 0.05; Skewness A = {skew_a:.2f}, B = {skew_b:.2f})."
        )
    elif clt_applies:
        norm_status = "warning"
        norm_verdict = "CAUTION (CLT Robust, N >= 30)"
        norm_detail = (
            f"Moderate deviation from normality detected (Shapiro-Wilk: Group A p = {shapiro_p_a:.3f}, "
            f"Group B p = {shapiro_p_b:.3f}), but group sample sizes (n_A = {n_a}, n_B = {n_b} >= 30) "
            "are sufficient for the Central Limit Theorem to guarantee normality of the sampling distribution of means."
        )
    else:
        norm_status = "fail"
        norm_verdict = "VIOLATED (Non-Normal & Small Sample)"
        norm_detail = (
            f"Normality is rejected (Shapiro-Wilk: Group A p = {shapiro_p_a:.3f}, Group B p = {shapiro_p_b:.3f}) "
            f"with insufficient sample size (n_A = {n_a}, n_B = {n_b} < 30) or severe skewness (|Skew| > 1.2). "
            "Parametric t-test results are unreliable. Switch to the non-parametric Mann-Whitney U test."
        )

    # Assumption 4: Outliers & Extreme Values
    if total_ext == 0 and total_mild == 0:
        outlier_status = "pass"
        outlier_verdict = "PASSED (No Outliers)"
        outlier_detail = "No data points exceed Tukey's 1.5×IQR threshold in either group."
    elif total_ext == 0:
        outlier_status = "pass" if min(n_a, n_b) >= 30 else "warning"
        outlier_verdict = "PASSED (Mild Outliers, CLT Robust)" if min(n_a, n_b) >= 30 else "CAUTION (Mild Outliers)"
        outlier_detail = (
            f"Mild outliers detected ({len(outliers_a)} in Group A, {len(outliers_b)} in Group B), "
            "but zero extreme outliers (> 3×IQR). The t-statistic remains stable under moderate sample sizes."
        )
    else:
        outlier_status = "fail"
        outlier_verdict = f"VIOLATED ({total_ext} Extreme Outliers)"
        outlier_detail = (
            f"{total_ext} extreme outlier(s) detected beyond 3×IQR fences. "
            "Extreme outliers heavily distort group means and inflate the standard error. "
            "Switch to the non-parametric Mann-Whitney U test."
        )

    assumptions_eval = [
        {
            "name": "1. Measurement Scale (Continuous)",
            "description": "The dependent variable must be continuous (ratio or interval scale).",
            "test_applied": "Variable inspection",
            "diagnostic_value": "Continuous numeric",
            "status": scale_status,
            "verdict": scale_verdict,
            "detail": scale_detail
        },
        {
            "name": "2. Independence of Observations",
            "description": "Observations in Group A and Group B must be mutually independent random samples.",
            "test_applied": "Study design check",
            "diagnostic_value": "2 Independent cohorts",
            "status": indep_status,
            "verdict": indep_verdict,
            "detail": indep_detail
        },
        {
            "name": "3. Normality within Groups",
            "description": "Values within each group should be normally distributed (or N >= 30 via CLT).",
            "test_applied": "Shapiro-Wilk Test (p-value) & Skewness",
            "diagnostic_value": f"Shapiro A: p={shapiro_p_a:.3f}, B: p={shapiro_p_b:.3f}",
            "status": norm_status,
            "verdict": norm_verdict,
            "detail": norm_detail
        },
        {
            "name": "4. Outliers & Extreme Values",
            "description": "No extreme outliers (> 3×IQR) that artificially inflate means or variance.",
            "test_applied": "Tukey's IQR Outlier Detection (1.5× and 3.0× fences)",
            "diagnostic_value": f"{total_ext} extreme, {total_mild} mild outliers",
            "status": outlier_status,
            "verdict": outlier_verdict,
            "detail": outlier_detail
        }
    ]

    # 8. Overall Recommendation Synthesis
    has_failure = any(a["status"] == "fail" for a in assumptions_eval)
    has_warning = any(a["status"] == "warning" for a in assumptions_eval)

    if has_failure:
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption(s) Violated: Switch to Alternative Method"
        overall_recommendation = (
            f"The normality assumption was violated in small sample sizes (p < 0.05) or extreme outliers were detected ({total_ext} extreme). "
            f"Under these conditions, Welch's t-test can suffer inflated Type I or Type II error. "
            f"\n\n**Actionable Recommendation:** Switch to the non-parametric **Mann-Whitney U Test** "
            f"(Wilcoxon Rank-Sum Test). The Mann-Whitney U test compares ranks rather than arithmetic means, "
            f"making it immune to extreme outliers and non-normal skewness. "
            f"\n\n**Alternative Test Result on this Data:** "
            f"Mann-Whitney U = {u_stat:.1f}, **p-value = {p_mann:.4f}**."
        )
    elif has_warning:
        overall_status = "warning"
        overall_verdict_badge = "🟡 Assumptions Borderline: Proceed with Caution (CLT Robust)"
        overall_recommendation = (
            f"Mild non-normality or mild outliers were detected, but group sample sizes (n_A={n_a}, n_B={n_b} >= 30) "
            f"provide Central Limit Theorem protection for the difference between means. "
            f"Additionally, Welch's t-test robustly accounts for unequal group variances (Levene's p = {p_levene:.3f}). "
            f"\n\n**Actionable Recommendation:** You may safely proceed with **Welch's Two-Sample t-Test** "
            f"(t = {t_stat_val:.3f}, df = {df_welch:.1f}, p = {p_val_welch:.4f}), but also cross-reference "
            f"with the non-parametric **Mann-Whitney U Test** (p = {p_mann:.4f}) to confirm consistency."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ All Assumptions Met: Proceed with Welch's Two-Sample t-Test"
        overall_recommendation = (
            f"All core assumptions (continuous scale, independent observations, approximate normality, "
            f"and absence of extreme outliers) are satisfied. Welch's t-test robustly handles unequal group variances "
            f"(Levene's p = {p_levene:.3f}). "
            f"\n\n**Actionable Recommendation:** Proceed with confidence using **Welch's Two-Sample t-Test**. "
            f"Welch t({df_welch:.1f}) = {t_stat_val:.3f}, p = {p_val_welch:.4f}, Mean Diff = {mean_diff:+.2f}, "
            f"95% CI = [{ci_lower:.2f}, {ci_upper:.2f}], Cohen's d = {cohens_d:.2f}."
        )

    results = {
        "n_a": int(n_a),
        "n_b": int(n_b),
        "n_total": int(n_a + n_b),
        "group_a_name": group_a_name,
        "group_b_name": group_b_name,
        "mean_a": calc_mean_a,
        "mean_b": calc_mean_b,
        "std_a": calc_std_a,
        "std_b": calc_std_b,
        "var_a": var_a,
        "var_b": var_b,
        "median_a": calc_med_a,
        "median_b": calc_med_b,
        "iqr_a": iqr_a,
        "iqr_b": iqr_b,
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "t_stat": t_stat_val,
        "df": df_welch,
        "p_val": p_val_welch,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": cohens_d,
        "hedges_g": hedges_g,
        "shapiro_p_a": shapiro_p_a,
        "shapiro_p_b": shapiro_p_b,
        "skew_a": skew_a,
        "skew_b": skew_b,
        "total_ext": total_ext,
        "total_mild": total_mild,
        "levene_stat": lev_stat,
        "p_levene": p_levene,
        "u_stat": u_stat,
        "p_mann": p_mann,
        "qq_osm_a": qq_osm_a,
        "qq_osr_a": qq_osr_a,
        "qq_slope_a": qq_slope_a,
        "qq_intercept_a": qq_intercept_a,
        "qq_osm_b": qq_osm_b,
        "qq_osr_b": qq_osr_b,
        "qq_slope_b": qq_slope_b,
        "qq_intercept_b": qq_intercept_b,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df, results


def generate_students_t_demo(
    n_a: int = 35,
    n_b: int = 35,
    mean_a: float = 100.0,
    mean_b: float = 108.0,
    std_a: float = 12.0,
    std_b: float = 12.0,
    dist_shape: str = "normal",
    outlier_count_b: int = 0,
    group_a_name: str = "Group A (Control)",
    group_b_name: str = "Group B (Treatment)",
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic data for two independent groups and executes Student's t-test with pooled variance and diagnostics.

    Args:
        n_a: Sample size for Group A.
        n_b: Sample size for Group B.
        mean_a: Target mean for Group A.
        mean_b: Target mean for Group B.
        std_a: Target standard deviation for Group A.
        std_b: Target standard deviation for Group B (supports homoscedasticity vs heteroscedasticity).
        dist_shape: Distribution shape ('normal', 'skewed_moderate', 'heavy_tailed').
        outlier_count_b: Number of extreme outliers to inject into Group B (0, 1, 2, 3).
        group_a_name: Label for Group A.
        group_b_name: Label for Group B.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (combined_DataFrame, results_dict).
    """
    rng = np.random.default_rng(seed)

    # 1. Generate base distribution shapes
    if dist_shape == "skewed_moderate":
        # Right-skewed Gamma distributions
        raw_a = rng.gamma(shape=2.5, scale=1.0, size=n_a)
        raw_b = rng.gamma(shape=2.5, scale=1.0, size=n_b)
    elif dist_shape == "heavy_tailed":
        # Heavy-tailed Student's t distribution (df=3)
        raw_a = rng.standard_t(df=3, size=n_a)
        raw_b = rng.standard_t(df=3, size=n_b)
    else:  # normal
        raw_a = rng.normal(loc=0.0, scale=1.0, size=n_a)
        raw_b = rng.normal(loc=0.0, scale=1.0, size=n_b)

    # Standardize and scale to specified means & standard deviations
    vals_a = (raw_a - np.mean(raw_a)) / (np.std(raw_a) if np.std(raw_a) > 0 else 1.0) * std_a + mean_a
    vals_b = (raw_b - np.mean(raw_b)) / (np.std(raw_b) if np.std(raw_b) > 0 else 1.0) * std_b + mean_b

    # 2. Inject extreme outliers if requested
    if outlier_count_b > 0:
        for i in range(min(outlier_count_b, n_b)):
            vals_b[-(i + 1)] = mean_b + 4.5 * std_b

    # 3. Construct DataFrame
    df_a = pd.DataFrame({
        "Observation_ID": [f"A_{i+1:02d}" for i in range(n_a)],
        "Group": group_a_name,
        "Value": vals_a
    })
    df_b = pd.DataFrame({
        "Observation_ID": [f"B_{j+1:02d}" for j in range(n_b)],
        "Group": group_b_name,
        "Value": vals_b
    })
    df = pd.concat([df_a, df_b], ignore_index=True)

    # 4. Descriptive Summary Metrics
    calc_mean_a = float(np.mean(vals_a))
    calc_mean_b = float(np.mean(vals_b))
    calc_std_a = float(np.std(vals_a, ddof=1))
    calc_std_b = float(np.std(vals_b, ddof=1))
    var_a = calc_std_a ** 2
    var_b = calc_std_b ** 2
    calc_med_a = float(np.median(vals_a))
    calc_med_b = float(np.median(vals_b))
    q25_a, q75_a = float(np.percentile(vals_a, 25)), float(np.percentile(vals_a, 75))
    q25_b, q75_b = float(np.percentile(vals_b, 25)), float(np.percentile(vals_b, 75))
    iqr_a = q75_a - q25_a
    iqr_b = q75_b - q25_b
    mean_diff = calc_mean_b - calc_mean_a

    # 5. Student's t-test Calculations (Equal Variance Pooled)
    df_student = int(n_a + n_b - 2)
    sp2 = float(((n_a - 1) * var_a + (n_b - 1) * var_b) / df_student) if df_student > 0 else 1.0
    sp = float(np.sqrt(sp2))
    se_diff = float(sp * np.sqrt(1.0 / n_a + 1.0 / n_b))
    t_stat_val = float(mean_diff / se_diff) if se_diff > 0 else 0.0

    # Two-tailed p-value
    p_val_student = float(2.0 * stats.t.sf(abs(t_stat_val), df=df_student))

    # 95% Confidence Interval for Difference (mu_B - mu_A)
    t_crit = float(stats.t.ppf(0.975, df=df_student))
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Standardized Effect Sizes (Pooled Cohen's d and Hedges' g)
    cohens_d = float(mean_diff / sp) if sp > 0 else 0.0
    hedges_g = float(cohens_d * (1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)))

    # 6. Diagnostic Computations
    shapiro_w_a, shapiro_p_a = stats.shapiro(vals_a)
    shapiro_w_b, shapiro_p_b = stats.shapiro(vals_b)
    shapiro_p_a, shapiro_p_b = float(shapiro_p_a), float(shapiro_p_b)
    skew_a, skew_b = float(stats.skew(vals_a)), float(stats.skew(vals_b))

    # Tukey's Outlier Diagnostics
    outliers_a = vals_a[(vals_a < q25_a - 1.5 * iqr_a) | (vals_a > q75_a + 1.5 * iqr_a)]
    ext_outliers_a = vals_a[(vals_a < q25_a - 3.0 * iqr_a) | (vals_a > q75_a + 3.0 * iqr_a)]
    outliers_b = vals_b[(vals_b < q25_b - 1.5 * iqr_b) | (vals_b > q75_b + 1.5 * iqr_b)]
    ext_outliers_b = vals_b[(vals_b < q25_b - 3.0 * iqr_b) | (vals_b > q75_b + 3.0 * iqr_b)]
    total_mild = len(outliers_a) + len(outliers_b)
    total_ext = len(ext_outliers_a) + len(ext_outliers_b)

    # Variance Equality Tests (Levene's test & Two-Sample F-test)
    lev_stat, p_levene = stats.levene(vals_a, vals_b, center="median")
    lev_stat, p_levene = float(lev_stat), float(p_levene)

    f_stat = float(var_a / var_b) if var_b > 0 else 1.0
    f_cdf = float(stats.f.cdf(f_stat, n_a - 1, n_b - 1))
    p_f = float(2.0 * min(f_cdf, 1.0 - f_cdf))

    # Non-parametric Alternative (Mann-Whitney U)
    mw_res = stats.mannwhitneyu(vals_b, vals_a, alternative="two-sided")
    u_stat = float(min(mw_res.statistic, n_a * n_b - mw_res.statistic))
    p_mann = float(mw_res.pvalue)

    # Unequal Variance Alternative (Welch's t-test)
    w_res = stats.ttest_ind(vals_b, vals_a, equal_var=False)
    t_welch = float(w_res.statistic)
    p_welch = float(w_res.pvalue)
    num_df = (var_a / n_a + var_b / n_b) ** 2
    denom_df = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    df_welch = float(num_df / denom_df) if denom_df > 0 else float(n_a + n_b - 2)

    # Q-Q Plot Coordinates for Plotly
    prob_a = stats.probplot(vals_a, dist="norm")
    qq_osm_a = prob_a[0][0].tolist()
    qq_osr_a = prob_a[0][1].tolist()
    qq_slope_a = float(prob_a[1][0])
    qq_intercept_a = float(prob_a[1][1])

    prob_b = stats.probplot(vals_b, dist="norm")
    qq_osm_b = prob_b[0][0].tolist()
    qq_osr_b = prob_b[0][1].tolist()
    qq_slope_b = float(prob_b[1][0])
    qq_intercept_b = float(prob_b[1][1])

    # 7. Evaluate Core Assumptions
    scale_status = "pass"
    scale_verdict = "Continuous Scale Satisfied"
    scale_detail = f"Values are continuous real numbers across both groups (Mean A={calc_mean_a:.1f}, Mean B={calc_mean_b:.1f})."

    indep_status = "pass"
    indep_verdict = "Independent Observations"
    indep_detail = "Observations represent distinct, unpaired participants sampled independently."

    # Normality
    if shapiro_p_a >= 0.05 and shapiro_p_b >= 0.05:
        norm_status = "pass"
        norm_verdict = "Normality Met (Shapiro p >= 0.05)"
        norm_detail = (
            f"Both groups show no significant departure from normality "
            f"(Group A Shapiro p = {shapiro_p_a:.3f}, Skew = {skew_a:.2f}; "
            f"Group B Shapiro p = {shapiro_p_b:.3f}, Skew = {skew_b:.2f})."
        )
    elif (n_a >= 30 and n_b >= 30) and (abs(skew_a) <= 1.0 and abs(skew_b) <= 1.0):
        norm_status = "warning"
        norm_verdict = "CLT Robust (N >= 30, Moderate Skew)"
        norm_detail = (
            f"Shapiro test indicated non-normality (A p={shapiro_p_a:.3f}, B p={shapiro_p_b:.3f}), "
            f"but sample sizes (n_A={n_a}, n_B={n_b} >= 30) and modest skewness (|skew| <= 1.0) "
            f"allow Central Limit Theorem (CLT) robustness for the difference in means."
        )
    else:
        norm_status = "fail"
        norm_verdict = "Normality Violated"
        norm_detail = (
            f"Data significantly deviates from normal (Group A Shapiro p = {shapiro_p_a:.3f}, Group B Shapiro p = {shapiro_p_b:.3f}) "
            f"without sufficient sample size or with substantial skewness. Switch to Mann-Whitney U test."
        )

    # Outliers
    if total_ext == 0 and total_mild == 0:
        outlier_status = "pass"
        outlier_verdict = "No Outliers Detected"
        outlier_detail = "No data points fall outside Tukey's 1.5×IQR or 3.0×IQR inner/outer fences."
    elif total_ext == 0 and total_mild > 0:
        outlier_status = "warning"
        outlier_verdict = f"{total_mild} Mild Outlier(s) Present"
        outlier_detail = (
            f"{total_mild} observation(s) fall beyond the 1.5×IQR inner fence. With sample sizes n_A={n_a}, n_B={n_b}, "
            f"pooled t-test is moderately robust, but inspect data points."
        )
    else:
        outlier_status = "fail"
        outlier_verdict = f"{total_ext} Extreme Outlier(s) Detected"
        outlier_detail = (
            f"{total_ext} extreme observation(s) exceed the 3.0×IQR fence. Extreme outliers distort means and pooled standard error. "
            f"Switch to the non-parametric Mann-Whitney U test."
        )

    # Homogeneity of Variance (Equal Variances)
    if p_levene >= 0.05 and p_f >= 0.05:
        var_status = "pass"
        var_verdict = "Homoscedastic (Equal Variances)"
        var_detail = (
            f"Homogeneity of variance confirmed by Levene's test (p = {p_levene:.3f}) and Two-Sample F-test (p = {p_f:.3f}). "
            f"The assumption of equal population variances is satisfied; pooled variance estimate is valid."
        )
    else:
        var_status = "fail"
        var_verdict = "Heteroscedastic (Unequal Variances)"
        var_detail = (
            f"Equal variance assumption is VIOLATED (Levene's test p = {p_levene:.4f}, F-test p = {p_f:.4f} < 0.05). "
            f"Student's t-test with pooled variance suffers inflated Type I error. Strongly recommend switching to Welch's t-test."
        )

    assumptions_eval = [
        {
            "name": "1. Measurement Scale (Continuous)",
            "description": "The dependent variable must be continuous (ratio or interval scale).",
            "test_applied": "Variable inspection",
            "diagnostic_value": "Continuous numeric",
            "status": scale_status,
            "verdict": scale_verdict,
            "detail": scale_detail
        },
        {
            "name": "2. Independence of Observations",
            "description": "Observations in Group A and Group B must be mutually independent random samples.",
            "test_applied": "Study design check",
            "diagnostic_value": "2 Independent cohorts",
            "status": indep_status,
            "verdict": indep_verdict,
            "detail": indep_detail
        },
        {
            "name": "3. Normality within Groups",
            "description": "Values within each group should be normally distributed (or N >= 30 via CLT).",
            "test_applied": "Shapiro-Wilk Test (p-value) & Skewness",
            "diagnostic_value": f"Shapiro A: p={shapiro_p_a:.3f}, B: p={shapiro_p_b:.3f}",
            "status": norm_status,
            "verdict": norm_verdict,
            "detail": norm_detail
        },
        {
            "name": "4. Outliers & Extreme Values",
            "description": "No extreme outliers (> 3×IQR) that artificially inflate means or pooled variance.",
            "test_applied": "Tukey's IQR Outlier Detection (1.5× and 3.0× fences)",
            "diagnostic_value": f"{total_ext} extreme, {total_mild} mild outliers",
            "status": outlier_status,
            "verdict": outlier_verdict,
            "detail": outlier_detail
        },
        {
            "name": "5. Homogeneity of Variance (Equal Variances)",
            "description": "Variances of both populations must be equal (homoscedasticity).",
            "test_applied": "Levene's Test (center=median) & Two-Sample F-Test",
            "diagnostic_value": f"Levene p={p_levene:.3f}, F-test p={p_f:.3f}, s1²/s2²={var_a/var_b:.2f}",
            "status": var_status,
            "verdict": var_verdict,
            "detail": var_detail
        }
    ]

    # 8. Overall Recommendation Synthesis
    if norm_status == "fail" or outlier_status == "fail":
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption(s) Violated: Switch to Mann-Whitney U Test"
        overall_recommendation = (
            f"The normality assumption was violated in small sample sizes or extreme outliers were detected ({total_ext} extreme). "
            f"Under these conditions, Student's t-test suffers distorted means and pooled variance. "
            f"\n\n**Actionable Recommendation:** Switch to the non-parametric **Mann-Whitney U Test** "
            f"(Wilcoxon Rank-Sum Test). The Mann-Whitney U test compares ranks rather than arithmetic means, "
            f"making it immune to extreme outliers and non-normal skewness. "
            f"\n\n**Alternative Test Result on this Data:** "
            f"Mann-Whitney U = {u_stat:.1f}, **p-value = {p_mann:.4f}**."
        )
    elif var_status == "fail":
        overall_status = "warning"
        overall_verdict_badge = "🟡 Equal Variance Violated: Switch to Welch's t-Test"
        overall_recommendation = (
            f"The assumption of equal population variances is **VIOLATED** (Levene p = {p_levene:.4f}, F-test p = {p_f:.4f} < 0.05). "
            f"Student's t-test relies on pooled variance; violating this assumption inflates Type I error rates. "
            f"\n\n**Actionable Recommendation:** Switch to **Welch's Two-Sample t-Test** (Alternative A). "
            f"Welch's test does not pool variances and adapts degrees of freedom using the Welch-Satterthwaite equation. "
            f"\n\n**Welch's Test Result on this Data:** "
            f"Welch t({df_welch:.1f}) = {t_welch:.3f}, **p-value = {p_welch:.4f}**."
        )
    elif norm_status == "warning" or outlier_status == "warning":
        overall_status = "warning"
        overall_verdict_badge = "🟡 Assumptions Borderline: Proceed with Caution (CLT Robust)"
        overall_recommendation = (
            f"Mild non-normality or mild outliers were detected, but group sample sizes (n_A={n_a}, n_B={n_b} >= 30) "
            f"provide Central Limit Theorem protection. Equal variances are verified (Levene p = {p_levene:.3f}). "
            f"\n\n**Actionable Recommendation:** You may proceed with **Student's Two-Sample t-Test** "
            f"(t = {t_stat_val:.3f}, df = {df_student}, p = {p_val_student:.4f}), but also cross-reference "
            f"with the non-parametric **Mann-Whitney U Test** (p = {p_mann:.4f}) to confirm consistency."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ All Assumptions Met: Proceed with Student's Two-Sample t-Test"
        overall_recommendation = (
            f"All core assumptions (continuous scale, independent observations, approximate normality, "
            f"absence of extreme outliers, and homogeneity of variance) are satisfied. "
            f"\n\n**Actionable Recommendation:** Proceed with confidence using **Student's Two-Sample t-Test** (Equal Variance). "
            f"Student's t({df_student}) = {t_stat_val:.3f}, p = {p_val_student:.4f}, Mean Diff = {mean_diff:+.2f}, "
            f"Pooled SD = {sp:.2f}, 95% CI = [{ci_lower:.2f}, {ci_upper:.2f}], Pooled Cohen's d = {cohens_d:.2f}."
        )

    results = {
        "n_a": int(n_a),
        "n_b": int(n_b),
        "n_total": int(n_a + n_b),
        "group_a_name": group_a_name,
        "group_b_name": group_b_name,
        "mean_a": calc_mean_a,
        "mean_b": calc_mean_b,
        "std_a": calc_std_a,
        "std_b": calc_std_b,
        "var_a": var_a,
        "var_b": var_b,
        "median_a": calc_med_a,
        "median_b": calc_med_b,
        "iqr_a": iqr_a,
        "iqr_b": iqr_b,
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "sp2": sp2,
        "sp": sp,
        "t_stat": t_stat_val,
        "df": df_student,
        "p_val": p_val_student,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": cohens_d,
        "hedges_g": hedges_g,
        "shapiro_p_a": shapiro_p_a,
        "shapiro_p_b": shapiro_p_b,
        "skew_a": skew_a,
        "skew_b": skew_b,
        "total_ext": total_ext,
        "total_mild": total_mild,
        "levene_stat": lev_stat,
        "p_levene": p_levene,
        "f_stat": f_stat,
        "p_f": p_f,
        "u_stat": u_stat,
        "p_mann": p_mann,
        "t_welch": t_welch,
        "p_welch": p_welch,
        "df_welch": df_welch,
        "qq_osm_a": qq_osm_a,
        "qq_osr_a": qq_osr_a,
        "qq_slope_a": qq_slope_a,
        "qq_intercept_a": qq_intercept_a,
        "qq_osm_b": qq_osm_b,
        "qq_osr_b": qq_osr_b,
        "qq_slope_b": qq_slope_b,
        "qq_intercept_b": qq_intercept_b,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation
    }

    return df, results


def generate_binary_logistic_regression_demo(
    n_pts: int = 120,
    n_predictors: int = 2,
    scenario: str = "cardiology",
    noise_level: float = 1.0,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic multivariate binary classification data and executes complete Binary Logistic
    Regression calculations, maximum likelihood parameter estimation (IRLS/Newton-Raphson), adjusted Odds Ratios,
    omnibus likelihood ratio tests, ROC/AUC curve coordinates, and foundational assumption checks.

    Args:
        n_pts: Number of observations (sample size).
        n_predictors: Number of predictor variables (1, 2, or 3).
        scenario: Archetypal scenario ('cardiology', 'readmission', 'exam_pass', 'multicollinear',
                  'sparse_events', 'custom').
        noise_level: Multiplier for latent residual variance.
        alpha: Significance threshold (default 0.05).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (demo_df, results_dict).
    """
    n_pts = max(30, int(n_pts))
    rng = np.random.default_rng(seed)

    # 1. Scenario-based Predictor & Latent Logit Setup
    if scenario == "cardiology":
        k = 3
        preset_name = "Clinical Cardiology: 10-Yr CVD Risk from Age, SBP & Smoking (k=3)"
        outcome_name = "CVD_Event"
        pred_names = ["Age (Years)", "Systolic BP (mmHg)", "Current Smoker (0/1)"]
        col_keys = ["Age_Years", "Systolic_BP_mmHg", "Current_Smoker"]
        x1 = np.clip(rng.normal(56.0, 8.5, size=n_pts), 35.0, 82.0)
        x2 = np.clip(rng.normal(132.0, 16.0, size=n_pts) + 0.35 * (x1 - 56.0), 95.0, 190.0)
        x3 = rng.binomial(1, 0.35, size=n_pts).astype(float)
        X_data = np.column_stack([x1, x2, x3])
        # True logit: z = -6.2 + 0.055*Age + 0.022*SBP + 0.85*Smoking
        true_z = -6.2 + 0.055 * x1 + 0.022 * x2 + 0.85 * x3 + rng.normal(0, 0.25 * noise_level, size=n_pts)

    elif scenario == "readmission":
        k = 2
        preset_name = "Hospital Readmission: 30-Day Readmission from LOS & Comorbidities (k=2)"
        outcome_name = "Readmission_30d"
        pred_names = ["Length of Stay (Days)", "Comorbidity Index (0-7)"]
        col_keys = ["LOS_Days", "Comorbidity_Score"]
        x1 = np.clip(rng.gamma(2.5, 2.5, size=n_pts), 1.0, 24.0)
        x2 = np.clip(np.round(rng.normal(2.8, 1.4, size=n_pts) + 0.08 * (x1 - 6.0)), 0.0, 7.0)
        X_data = np.column_stack([x1, x2])
        # True logit: z = -2.8 + 0.18*LOS + 0.42*Comorbidities
        true_z = -2.8 + 0.18 * x1 + 0.42 * x2 + rng.normal(0, 0.25 * noise_level, size=n_pts)

    elif scenario == "exam_pass":
        k = 1
        preset_name = "Educational Testing: Exam Pass/Fail from Study Hours (k=1, Simple Logistic)"
        outcome_name = "Exam_Passed"
        pred_names = ["Study Hours (Weekly)"]
        col_keys = ["Study_Hours"]
        x1 = np.clip(rng.normal(18.0, 5.5, size=n_pts), 4.0, 36.0)
        X_data = x1.reshape(-1, 1)
        # True logit: z = -3.5 + 0.22*Hours
        true_z = -3.5 + 0.22 * x1 + rng.normal(0, 0.2 * noise_level, size=n_pts)

    elif scenario == "multicollinear":
        k = 2 if n_predictors < 3 else 3
        preset_name = "High Multicollinearity Scenario (VIF Diagnostic, k=2)"
        outcome_name = "Treatment_Response"
        pred_names = ["Biomarker Alpha (ng/mL)", "Biomarker Beta (ng/mL)"]
        col_keys = ["Biomarker_Alpha", "Biomarker_Beta"]
        x1 = rng.normal(50.0, 10.0, size=n_pts)
        x2 = 0.94 * x1 + rng.normal(0.0, 2.8, size=n_pts)
        if k == 3:
            pred_names.append("Patient BMI")
            col_keys.append("Patient_BMI")
            x3 = rng.normal(26.0, 4.0, size=n_pts)
            X_data = np.column_stack([x1, x2, x3])
        else:
            X_data = np.column_stack([x1, x2])
        true_z = -3.0 + 0.04 * x1 + 0.03 * x2 + rng.normal(0, 0.25 * noise_level, size=n_pts)

    elif scenario == "sparse_events":
        k = 2
        preset_name = "Sparse Events / Low EPV (Separation / Small Sample Diagnostic)"
        outcome_name = "Rare_Adverse_Event"
        pred_names = ["Drug Dose (mg)", "Prior Sensitivity Score"]
        col_keys = ["Drug_Dose_mg", "Sensitivity_Score"]
        x1 = rng.uniform(10.0, 100.0, size=n_pts)
        x2 = rng.uniform(1.0, 10.0, size=n_pts)
        X_data = np.column_stack([x1, x2])
        # Very low probability -> small number of positive events
        true_z = -4.8 + 0.015 * x1 + 0.12 * x2

    else:  # custom
        k = max(1, min(3, int(n_predictors)))
        preset_name = f"Custom Logistic Model (k={k} Predictors, N={n_pts})"
        outcome_name = "Binary_Outcome"
        pred_names = [f"Predictor X{i+1}" for i in range(k)]
        col_keys = [f"Predictor_X{i+1}" for i in range(k)]
        cols = []
        true_z = -1.2
        for i in range(k):
            xi = rng.normal(25.0 * (i + 1), 6.0, size=n_pts)
            cols.append(xi)
            true_z = true_z + (0.12 / (i + 1)) * (xi - np.mean(xi))
        X_data = np.column_stack(cols)
        true_z = true_z + rng.normal(0, 0.3 * noise_level, size=n_pts)

    # 2. Simulate Binary Realization (Bernoulli)
    true_p = 1.0 / (1.0 + np.exp(-np.clip(true_z, -22.0, 22.0)))
    y = rng.binomial(1, true_p, size=n_pts)

    # Ensure non-trivial balance (at least 2 observations in each category)
    if np.sum(y == 1) < 2:
        top_idx = np.argsort(true_z)[-max(3, int(0.08 * n_pts)):]
        y[top_idx] = 1
    elif np.sum(y == 0) < 2:
        bot_idx = np.argsort(true_z)[:max(3, int(0.08 * n_pts))]
        y[bot_idx] = 0

    # 3. Maximum Likelihood Estimation via Newton-Raphson / IRLS
    X_mat = np.column_stack([np.ones(n_pts), X_data])
    beta = np.zeros(k + 1)

    for _ in range(40):
        z_est = np.clip(X_mat @ beta, -25.0, 25.0)
        p_est = 1.0 / (1.0 + np.exp(-z_est))
        p_est = np.clip(p_est, 1e-12, 1.0 - 1e-12)
        W = p_est * (1.0 - p_est)
        grad = X_mat.T @ (y - p_est)
        # Slight Tikhonov ridge (1e-6) for numerical stability against singular Hessians
        H = - (X_mat.T * W) @ X_mat - 1e-6 * np.eye(k + 1)
        step = np.linalg.solve(-H, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-6:
            break

    # Re-evaluate final predicted probabilities
    z_final = np.clip(X_mat @ beta, -25.0, 25.0)
    p_final = 1.0 / (1.0 + np.exp(-z_final))
    p_final = np.clip(p_final, 1e-12, 1.0 - 1e-12)
    W_final = p_final * (1.0 - p_final)

    # Covariance Matrix, Standard Errors & Wald Statistics
    info_mat = (X_mat.T * W_final) @ X_mat + 1e-6 * np.eye(k + 1)
    cov_matrix = np.linalg.pinv(info_mat)
    se = np.sqrt(np.maximum(1e-12, np.diag(cov_matrix)))
    z_stats = beta / se
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stats)))

    # Odds Ratios & 95% Confidence Intervals
    odds_ratios = np.exp(np.clip(beta, -15.0, 15.0))
    or_ci_lower = np.exp(np.clip(beta - 1.96 * se, -20.0, 20.0))
    or_ci_upper = np.exp(np.clip(beta + 1.96 * se, -20.0, 20.0))

    # 4. Omnibus Goodness of Fit & Likelihood Ratio Test
    p_null = np.clip(np.mean(y), 1e-12, 1.0 - 1e-12)
    ll_null = float(np.sum(y * np.log(p_null) + (1.0 - y) * np.log(1.0 - p_null)))
    ll_model = float(np.sum(y * np.log(p_final) + (1.0 - y) * np.log(1.0 - p_final)))
    lr_stat = float(max(0.0, 2.0 * (ll_model - ll_null)))
    p_omnibus = float(1.0 - stats.chi2.cdf(lr_stat, df=k))
    pseudo_r2 = float(max(0.0, min(1.0, 1.0 - (ll_model / ll_null)))) if ll_null < 0 else 0.0
    aic = float(2 * (k + 1) - 2 * ll_model)
    bic = float(np.log(n_pts) * (k + 1) - 2 * ll_model)

    # 5. Classification Performance & Confusion Matrix (0.5 Cutoff)
    y_pred = (p_final >= 0.5).astype(int)
    tp = int(np.sum((y == 1) & (y_pred == 1)))
    tn = int(np.sum((y == 0) & (y_pred == 0)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))
    accuracy = float((tp + tn) / n_pts)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # 6. ROC Curve & AUC Score
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    thresholds = np.linspace(0.0, 1.0, 101)
    roc_fpr, roc_tpr = [], []
    for th in thresholds:
        th_pred = (p_final >= th).astype(int)
        th_tp = np.sum((y == 1) & (th_pred == 1))
        th_fp = np.sum((y == 0) & (th_pred == 1))
        roc_tpr.append(float(th_tp / n_pos) if n_pos > 0 else 0.0)
        roc_fpr.append(float(th_fp / n_neg) if n_neg > 0 else 0.0)

    # AUC via Mann-Whitney U rank equivalence
    if n_pos > 0 and n_neg > 0:
        pos_p = p_final[y == 1]
        neg_p = p_final[y == 0]
        concordant = np.sum(pos_p[:, None] > neg_p[None, :])
        ties = 0.5 * np.sum(pos_p[:, None] == neg_p[None, :])
        auc_val = float((concordant + ties) / (n_pos * n_neg))
        auc_val = max(0.5, min(1.0, auc_val))
    else:
        auc_val = 0.5

    # 7. Assumption Checks & Diagnostic Auditing
    # A. Multicollinearity (VIF)
    vifs = []
    if k > 1:
        for i in range(k):
            Xi = np.delete(X_data, i, axis=1)
            Xi = np.column_stack([np.ones(n_pts), Xi])
            yi_col = X_data[:, i]
            b_aux = np.linalg.lstsq(Xi, yi_col, rcond=None)[0]
            ss_tot_aux = np.sum((yi_col - np.mean(yi_col))**2)
            ss_res_aux = np.sum((yi_col - Xi @ b_aux)**2)
            r2_aux = 1.0 - (ss_res_aux / ss_tot_aux) if ss_tot_aux > 0 else 0.0
            vif_i = 1.0 / (1.0 - r2_aux) if r2_aux < 0.999 else 100.0
            vifs.append(float(vif_i))
    else:
        vifs = [1.0]
    max_vif = float(np.max(vifs))

    # B. Events Per Variable (EPV)
    n_events = int(min(np.sum(y == 1), np.sum(y == 0)))
    epv = float(n_events / k)

    # C. Complete Separation Audit
    is_separated = bool(np.any(se[1:] > 25.0) or np.any(np.abs(beta[1:]) > 12.0))

    # D. Pearson Residuals & Extreme Outliers
    pearson_residuals = (y - p_final) / np.sqrt(p_final * (1.0 - p_final))
    max_abs_res = float(np.max(np.abs(pearson_residuals)))
    extreme_outliers_count = int(np.sum(np.abs(pearson_residuals) > 3.0))

    # Assemble Structured Assumptions Evaluation List
    asm_binary = {
        "name": "1. Binary Dependent Variable",
        "description": "Outcome variable must be dichotomous (0 or 1, Yes or No, Event or Non-Event).",
        "test_applied": "Categorical Scale Verification",
        "diagnostic_value": f"2 discrete categories: {int(np.sum(y == 1))} Events (1) vs {int(np.sum(y == 0))} Non-Events (0)",
        "status": "pass",
        "verdict": "PASS (Dichotomous Scale Confirmed)",
        "detail": f"Observed sample satisfies binary outcome requirements with {int(np.sum(y == 1))} positive cases ({np.mean(y)*100:.1f}%) and {int(np.sum(y == 0))} negative controls."
    }

    if epv >= 10.0:
        epv_status = "pass"
        epv_verdict = f"PASS (EPV = {epv:.1f} ≥ 10.0)"
        epv_detail = f"Adequate statistical sample size: {n_events} effective events across {k} predictors yields an EPV of {epv:.1f} (Benchmark: ≥ 10-20), minimizing overfitting risk."
    elif epv >= 5.0:
        epv_status = "warning"
        epv_verdict = f"BORDERLINE (EPV = {epv:.1f} < 10.0)"
        epv_detail = f"Sample size is marginal: {n_events} effective events yields EPV = {epv:.1f} (< 10). Standard errors may have slight upward bias. Cross-validate with penalized logistic regression."
    else:
        epv_status = "fail"
        epv_verdict = f"VIOLATED (EPV = {epv:.1f} < 5.0)"
        epv_detail = f"Severe sparse data: Only {n_events} effective events across {k} predictors (EPV = {epv:.1f} < 5.0). Standard maximum likelihood suffers from severe bias. Penalized logistic regression (Firth or Ridge) is strictly required."

    asm_epv = {
        "name": "2. Sample Size Adequacy (Events Per Variable / EPV)",
        "description": "Rule of thumb requires at least 10 to 20 events per predictor parameter to prevent overfitting.",
        "test_applied": "Peduzzi-Harrell EPV Criterion",
        "diagnostic_value": f"Effective Events = {n_events}, Predictors = {k} (EPV = {epv:.1f})",
        "status": epv_status,
        "verdict": epv_verdict,
        "detail": epv_detail
    }

    if max_vif <= 5.0:
        vif_status = "pass"
        vif_verdict = f"PASS (Max VIF = {max_vif:.2f} ≤ 5.0)"
        vif_detail = f"No severe multicollinearity detected among predictor variables (all VIFs ≤ {max_vif:.2f}). Standard errors of coefficients are stable and uninflated."
    elif max_vif <= 10.0:
        vif_status = "warning"
        vif_verdict = f"BORDERLINE (Max VIF = {max_vif:.2f} > 5.0)"
        vif_detail = f"Moderate collinearity between predictors (Max VIF = {max_vif:.2f}). Individual Odds Ratios may exhibit widened confidence intervals."
    else:
        vif_status = "fail"
        vif_verdict = f"VIOLATED (Max VIF = {max_vif:.2f} > 10.0)"
        vif_detail = f"Severe multicollinearity (Max VIF = {max_vif:.2f} > 10.0). Predictors share heavy redundant variance. Standard errors and p-values are untrustworthy. Consider L2 Ridge regularization or dropping collinear predictors."

    asm_vif = {
        "name": "3. Absence of Severe Multicollinearity",
        "description": "Predictor variables should not be excessively correlated with one another (VIF ≤ 5.0).",
        "test_applied": "Variance Inflation Factor (VIF)",
        "diagnostic_value": f"Max VIF = {max_vif:.2f} (Benchmark: ≤ 5.0)",
        "status": vif_status,
        "verdict": vif_verdict,
        "detail": vif_detail
    }

    if not is_separated:
        sep_status = "pass"
        sep_verdict = "PASS (No Complete Separation)"
        sep_detail = "Data exhibits adequate overlap between outcome categories. Maximum likelihood coefficients converged to finite, well-calibrated estimates."
    else:
        sep_status = "fail"
        sep_verdict = "VIOLATED (Quasi-Complete Separation Detected)"
        sep_detail = "A predictor or combination perfectly separates 0s and 1s, causing coefficient estimates or standard errors to explode (SE > 25.0). Use Firth's penalized logistic regression or exact logistic regression."

    asm_sep = {
        "name": "4. Complete / Quasi-Complete Separation",
        "description": "No single predictor or linear combination should perfectly classify the outcome, which inflates standard errors toward infinity.",
        "test_applied": "Hessian Stability & Parameter Variance Audit",
        "diagnostic_value": f"Max Beta = {float(np.max(np.abs(beta[1:]))):.2f}, Max SE = {float(np.max(se[1:])):.2f}",
        "status": sep_status,
        "verdict": sep_verdict,
        "detail": sep_detail
    }

    if extreme_outliers_count == 0:
        res_status = "pass"
        res_verdict = "PASS (No Extreme Residual Outliers)"
        res_detail = f"All standardized Pearson residuals remain within acceptable bounds (Max |r| = {max_abs_res:.2f} ≤ 3.0)."
    elif extreme_outliers_count <= 2:
        res_status = "warning"
        res_verdict = f"BORDERLINE ({extreme_outliers_count} Extreme Outlier(s))"
        res_detail = f"Detected {extreme_outliers_count} observation(s) with |residual| > 3.0 (Max |r| = {max_abs_res:.2f}). Check for data entry errors or unusual clinical presentations."
    else:
        res_status = "fail"
        res_verdict = f"VIOLATED ({extreme_outliers_count} Residual Outliers)"
        res_detail = f"Detected {extreme_outliers_count} extreme residual outliers (|r| > 3.0, Max |r| = {max_abs_res:.2f}). These observations strongly degrade model fit."

    asm_res = {
        "name": "5. Residual Influence & Extreme Outliers",
        "description": "No observations with extreme standardized Pearson residuals (|r| > 3.0) that exert disproportionate leverage.",
        "test_applied": "Standardized Pearson Residuals Audit",
        "diagnostic_value": f"Extreme Outliers (|r| > 3.0): {extreme_outliers_count}, Max |r| = {max_abs_res:.2f}",
        "status": res_status,
        "verdict": res_verdict,
        "detail": res_detail
    }

    assumptions_eval = [asm_binary, asm_epv, asm_vif, asm_sep, asm_res]

    # Overall Verdict Synthesis
    has_fail = any(a["status"] == "fail" for a in assumptions_eval)
    has_warn = any(a["status"] == "warning" for a in assumptions_eval)

    if has_fail:
        overall_status = "fail"
        overall_verdict_badge = "❌ Assumption(s) Violated: Switch to Penalized / Regularized Logistic Regression"
        overall_recommendation = (
            f"Critical assumptions for standard Maximum Likelihood Logistic Regression are violated "
            f"({'Severe Multicollinearity' if vif_status == 'fail' else ('Complete Separation' if sep_status == 'fail' else 'Sparse Sample / Low EPV')}). "
            f"\n\n**Actionable Recommendation:** Switch to **Penalized Logistic Regression (L2 Ridge / Firth's Penalized Likelihood)**. "
            f"Penalization shrinks coefficient variance, handles collinearity robustly, and prevents infinite standard errors."
        )
    elif has_warn:
        overall_status = "warning"
        overall_verdict_badge = "🟡 Assumptions Borderline: Proceed with Caution / Robust Verification"
        overall_recommendation = (
            f"Model assumptions exhibit minor to moderate tension (EPV = {epv:.1f}, Max VIF = {max_vif:.2f}). "
            f"Standard binary logistic regression remains interpretable, but cross-referencing with L2-regularized logistic regression is advised."
        )
    else:
        overall_status = "pass"
        overall_verdict_badge = "✅ All Assumptions Met: Proceed with Binary Logistic Regression"
        overall_recommendation = (
            f"All foundational assumptions (binary outcome, adequate EPV, low multicollinearity, lack of separation, and reasonable residuals) "
            f"are satisfied. The logistic regression model yields fully calibrated, statistically reliable Odds Ratios."
        )

    # 8. Assemble Clean Output DataFrame
    df_data = {"Obs_ID": [f"Obs_{i+1:03d}" for i in range(n_pts)], outcome_name: y}
    for idx, col_name in enumerate(col_keys):
        df_data[col_name] = np.round(X_data[:, idx], 2)
    df_data["Predicted_Probability"] = np.round(p_final, 4)
    df_data["Predicted_Class"] = y_pred

    demo_df = pd.DataFrame(df_data)

    results = {
        "n_pts": int(n_pts),
        "k_preds": int(k),
        "outcome_name": outcome_name,
        "pred_names": pred_names,
        "col_keys": col_keys,
        "beta": beta.tolist(),
        "se": se.tolist(),
        "z_stats": z_stats.tolist(),
        "p_values": p_values.tolist(),
        "odds_ratios": odds_ratios.tolist(),
        "or_ci_lower": or_ci_lower.tolist(),
        "or_ci_upper": or_ci_upper.tolist(),
        "ll_null": ll_null,
        "ll_model": ll_model,
        "lr_stat": lr_stat,
        "lr_df": k,
        "p_omnibus": p_omnibus,
        "pseudo_r2_mcfadden": pseudo_r2,
        "aic": aic,
        "bic": bic,
        "n_events": n_events,
        "epv": epv,
        "max_vif": max_vif,
        "vifs": vifs,
        "is_separated": is_separated,
        "auc": auc_val,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "prob_pred": p_final.tolist(),
        "y_obs": y.tolist(),
        "primary_x_name": pred_names[0],
        "primary_x_vals": X_data[:, 0].tolist(),
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "assumptions_eval": assumptions_eval,
        "overall_status": overall_status,
        "overall_verdict_badge": overall_verdict_badge,
        "overall_recommendation": overall_recommendation,
        "preset_name": preset_name
    }

    return demo_df, results


# Alias for backward compatibility with decision_schema
generate_binary_classification_demo = generate_binary_logistic_regression_demo







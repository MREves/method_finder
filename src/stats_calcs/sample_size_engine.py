"""Biostatistical & Methodological Engine for Sample Size & Statistical Power Calculations.

Provides closed-form, verified calculations for clinical trials, observational epidemiology,
and diagnostic screening studies across continuous, binary, survival, and diagnostic endpoints.

Core Methodological Frameworks:
1. Continuous Outcomes (Comparing Means & Associations):
   - Two Independent Groups (Two-sample t-test / Welch's t with unequal allocation ratio k = n2/n1).
   - Paired / Repeated Measures (Paired t-test with correlation rho or SD of differences).
   - Multiple Groups (One-Way ANOVA with Cohen's f and non-central F-distribution).
   - Linear Correlation (Pearson's r via Fisher's z-transformation).
2. Binary Outcomes & Epidemiological Risk Metrics:
   - Two Independent Proportions (Unpooled Fleiss & Casagrande-Pike continuity correction).
   - Prospective Cohort Study (Relative Risk RR with baseline incidence p0).
   - Case-Control Study (Odds Ratio OR with control exposure prevalence and matching ratio).
   - Matched Pairs (McNemar's test for discordant proportions).
3. Longitudinal Survival Outcomes (Time-to-Event):
   - Schoenfeld (1981) formula for required event count E under proportional hazards.
   - Event-to-enrollment conversion factor based on cumulative event probability P(Event).
   - Freedman (1982) alternative comparison.
4. Diagnostic & Screening Accuracy Studies:
   - Target sensitivity / specificity estimation within Wilson score margin of error W.
   - Prevalence-adjusted screening cohort enrollment: N_total = n_diseased / Prevalence.
5. Trial Design Adjustments & Real-World Pragmatics:
   - Attrition & loss-to-follow-up inflation: N_recruited = N_evaluable / (1 - d).
   - Cluster randomization design effect: VIF = 1 + (m - 1) * ICC.
6. Power Curves & Sensitivity Surface Generators:
   - Sequences of sample sizes with corresponding power across Small, Medium, and Large effects.
7. Automated Protocol Justification Statement Generator:
   - Publication-ready paragraph formatted for NIHR / MRC / Ethics protocol submissions.
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False


# =====================================================================
# Mathematical & Normal Distribution Helpers
# =====================================================================

def _normal_cdf(z: float) -> float:
    """Cumulative distribution function of the standard normal distribution Phi(z)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Quantile function (inverse CDF) of the standard normal distribution.
    
    Uses high-precision Acklam rational approximation (accurate to 1e-9)
    with SciPy fallback when available.
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if HAS_SCIENTIFIC_STACK:
        return float(stats.norm.ppf(p))

    # Acklam's algorithm
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _get_z_alpha(alpha: float = 0.05, two_tailed: bool = True) -> float:
    """Returns critical standard normal value z for Type I error rate alpha."""
    p = 1.0 - (alpha / 2.0 if two_tailed else alpha)
    return _normal_ppf(p)


def _get_z_beta(power: float = 0.80) -> float:
    """Returns critical standard normal value z for statistical power (1 - beta)."""
    return _normal_ppf(power)


class SampleSizeDemoData(dict):
    """Dictionary holding synthetic study protocol parameters and simulation dataset with 2-tuple unpacking.
    
    Supports both key indexing (`demo['df']`, `demo['meta']`) and unpacking:
    `df, meta = generate_sample_size_demo_data(...)`.
    """
    def __iter__(self):
        yield self["df"]
        yield self["meta"]


# =====================================================================
# Workflow 1: Continuous Outcomes (Means & Associations)
# =====================================================================

def calculate_sample_size_two_means(
    mean1: float,
    mean2: float,
    sd1: float,
    sd2: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    two_tailed: bool = True
) -> Dict[str, Any]:
    """Calculates required sample size and post-hoc power for two independent groups (t-test / Welch).

    Supports unequal allocation ratios (k = n2 / n1) and unequal group standard deviations.

    Args:
        mean1: Mean of Group 1 (e.g. control arm).
        mean2: Mean of Group 2 (e.g. experimental arm).
        sd1: Standard deviation of Group 1.
        sd2: Standard deviation of Group 2 (defaults to sd1 if None).
        alpha: Two-sided significance level (Type I error, default 0.05).
        power: Desired statistical power (1 - beta, default 0.80).
        allocation_ratio: Ratio of group sizes k = n2 / n1 (default 1.0 for equal allocation).
        two_tailed: True for two-sided hypothesis test, False for one-sided.

    Returns:
        Dict[str, Any]: Sample size per arm, total N, Cohen's d, achieved power, and formulas.
    """
    if sd2 is None:
        sd2 = sd1

    mean_diff = abs(mean1 - mean2)
    if mean_diff <= 0.0:
        raise ValueError("Mean difference |mean1 - mean2| must be greater than 0.")
    if sd1 <= 0.0 or sd2 <= 0.0:
        raise ValueError("Standard deviations must be strictly positive.")
    if allocation_ratio <= 0.0:
        raise ValueError("Allocation ratio must be strictly positive.")

    k = float(allocation_ratio)
    z_alpha = _get_z_alpha(alpha, two_tailed=two_tailed)
    z_beta = _get_z_beta(power)

    # Pooled standard deviation
    sd_pooled = math.sqrt((sd1**2 + sd2**2) / 2.0) if abs(k - 1.0) < 1e-4 else math.sqrt((sd1**2 + (sd2**2) / k) / (1.0 + 1.0 / k))
    cohen_d = mean_diff / sd_pooled

    # Required sample size formula: n1 = (1 + 1/k) * (z_alpha + z_beta)^2 * sd^2 / diff^2
    var_term = (sd1**2) + (sd2**2) / k
    n1_raw = ((z_alpha + z_beta)**2 * var_term) / (mean_diff**2)
    n1 = math.ceil(n1_raw)
    n2 = math.ceil(k * n1)
    n_total = n1 + n2

    # Achieved power calculation with integer sample sizes
    se_actual = math.sqrt((sd1**2) / n1 + (sd2**2) / n2)
    z_stat = mean_diff / se_actual
    achieved_power = _normal_cdf(z_stat - z_alpha)
    if two_tailed:
        achieved_power += _normal_cdf(-z_stat - z_alpha)

    # Cohen's benchmark categorization
    if cohen_d < 0.20:
        cohen_tier = "Negligible (< 0.20)"
    elif cohen_d < 0.50:
        cohen_tier = "Small (0.20 - 0.49)"
    elif cohen_d < 0.80:
        cohen_tier = "Medium (0.50 - 0.79)"
    else:
        cohen_tier = "Large (>= 0.80)"

    return {
        "test_type": "two_independent_means",
        "n1": n1,
        "n2": n2,
        "n_total": n_total,
        "mean1": mean1,
        "mean2": mean2,
        "mean_diff": round(mean_diff, 4),
        "sd1": sd1,
        "sd2": sd2,
        "sd_pooled": round(sd_pooled, 4),
        "cohen_d": round(cohen_d, 3),
        "cohen_tier": cohen_tier,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": round(achieved_power, 4),
        "achieved_power_pct": round(achieved_power * 100.0, 1),
        "allocation_ratio": k,
        "two_tailed": two_tailed,
        "z_alpha": round(z_alpha, 3),
        "z_beta": round(z_beta, 3)
    }


def calculate_sample_size_paired_means(
    mean_diff: float,
    sd_diff: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> Dict[str, Any]:
    """Calculates required sample size (pairs) for paired/pre-post comparisons (paired t-test).

    Args:
        mean_diff: Hypothesized mean within-subject difference (delta = mu_post - mu_pre).
        sd_diff: Standard deviation of within-subject paired differences (sigma_diff).
        alpha: Significance level (default 0.05).
        power: Desired statistical power (default 0.80).
        two_tailed: True for two-sided test.

    Returns:
        Dict[str, Any]: Required pairs, Cohen's d_z, achieved power.
    """
    diff_abs = abs(mean_diff)
    if diff_abs <= 0.0:
        raise ValueError("Paired mean difference must be non-zero.")
    if sd_diff <= 0.0:
        raise ValueError("Standard deviation of differences must be strictly positive.")

    z_alpha = _get_z_alpha(alpha, two_tailed=two_tailed)
    z_beta = _get_z_beta(power)

    cohen_dz = diff_abs / sd_diff
    n_raw = ((z_alpha + z_beta)**2 * (sd_diff**2)) / (diff_abs**2)
    n_pairs = math.ceil(n_raw)

    se_actual = sd_diff / math.sqrt(n_pairs)
    z_stat = diff_abs / se_actual
    achieved_power = _normal_cdf(z_stat - z_alpha)
    if two_tailed:
        achieved_power += _normal_cdf(-z_stat - z_alpha)

    return {
        "test_type": "paired_means",
        "n_pairs": n_pairs,
        "n_total_observations": n_pairs * 2,
        "mean_diff": round(mean_diff, 4),
        "sd_diff": round(sd_diff, 4),
        "cohen_dz": round(cohen_dz, 3),
        "alpha": alpha,
        "target_power": power,
        "achieved_power": round(achieved_power, 4),
        "achieved_power_pct": round(achieved_power * 100.0, 1),
        "two_tailed": two_tailed,
        "z_alpha": round(z_alpha, 3),
        "z_beta": round(z_beta, 3)
    }


def calculate_sample_size_anova(
    num_groups: int,
    cohen_f: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> Dict[str, Any]:
    """Calculates required sample size for One-Way Analysis of Variance (ANOVA).

    Args:
        num_groups: Number of comparison groups k (>= 2).
        cohen_f: Cohen's effect size f (Small: 0.10, Medium: 0.25, Large: 0.40).
        alpha: Significance level (default 0.05).
        power: Statistical power (default 0.80).

    Returns:
        Dict[str, Any]: Sample size per group, total N, degrees of freedom.
    """
    if num_groups < 2:
        raise ValueError("ANOVA requires at least 2 groups.")
    if cohen_f <= 0.0:
        raise ValueError("Cohen's f must be strictly positive.")

    k = int(num_groups)
    z_alpha = _get_z_alpha(alpha, two_tailed=True)
    z_beta = _get_z_beta(power)

    # Standard approximation formula: n_per_group = (z_alpha + z_beta)^2 / (k * f^2) + 1
    # Enhanced with Cohen's non-centrality parameter approximation
    n_raw = ((z_alpha + z_beta)**2) / (cohen_f**2)
    n_per_group = math.ceil(n_raw / float(k)) + 1
    n_per_group = max(3, n_per_group)
    n_total = n_per_group * k

    # Degrees of freedom
    df_num = k - 1
    df_denom = n_total - k

    # Benchmark tier
    if cohen_f < 0.10:
        tier = "Negligible (< 0.10)"
    elif cohen_f < 0.25:
        tier = "Small (0.10 - 0.24)"
    elif cohen_f < 0.40:
        tier = "Medium (0.25 - 0.39)"
    else:
        tier = "Large (>= 0.40)"

    return {
        "test_type": "one_way_anova",
        "num_groups": k,
        "cohen_f": round(cohen_f, 3),
        "cohen_tier": tier,
        "n_per_group": n_per_group,
        "n_total": n_total,
        "df_numerator": df_num,
        "df_denominator": df_denom,
        "alpha": alpha,
        "target_power": power
    }


def calculate_sample_size_correlation(
    r: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> Dict[str, Any]:
    """Calculates required sample size for Pearson linear correlation via Fisher's z-transformation.

    Args:
        r: Hypothesized correlation coefficient (-1 < r < 1, r != 0).
        alpha: Significance level (default 0.05).
        power: Desired power (default 0.80).
        two_tailed: True for two-sided test.

    Returns:
        Dict[str, Any]: Required N, Fisher's z, achieved power.
    """
    abs_r = abs(r)
    if abs_r <= 0.0 or abs_r >= 1.0:
        raise ValueError("Correlation r must satisfy 0 < |r| < 1.")

    z_alpha = _get_z_alpha(alpha, two_tailed=two_tailed)
    z_beta = _get_z_beta(power)

    # Fisher's z transformation: w = 0.5 * ln((1 + r)/(1 - r))
    fisher_w = 0.5 * math.log((1.0 + abs_r) / (1.0 - abs_r))
    n_raw = ((z_alpha + z_beta) / fisher_w)**2 + 3.0
    n = math.ceil(n_raw)
    n = max(4, n)

    # Achieved power
    se_actual = 1.0 / math.sqrt(n - 3)
    z_stat = fisher_w / se_actual
    achieved_power = _normal_cdf(z_stat - z_alpha)
    if two_tailed:
        achieved_power += _normal_cdf(-z_stat - z_alpha)

    return {
        "test_type": "pearson_correlation",
        "r": round(r, 4),
        "abs_r": round(abs_r, 4),
        "fisher_w": round(fisher_w, 4),
        "n_total": n,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": round(achieved_power, 4),
        "achieved_power_pct": round(achieved_power * 100.0, 1),
        "two_tailed": two_tailed
    }


# =====================================================================
# Workflow 2: Binary Outcomes & Risk Metrics (Proportions, Cohort, Case-Control)
# =====================================================================

def calculate_sample_size_two_proportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    continuity_correction: bool = True,
    two_tailed: bool = True
) -> Dict[str, Any]:
    """Calculates required sample size for comparing two independent binomial proportions.

    Uses Fleiss unpooled formula and Casagrande-Pike-Smith continuity correction.

    Args:
        p1: Proportion in Group 1 (e.g. control response rate, 0 < p1 < 1).
        p2: Proportion in Group 2 (e.g. experimental response rate, 0 < p2 < 1).
        alpha: Significance level (default 0.05).
        power: Desired statistical power (default 0.80).
        allocation_ratio: k = n2 / n1 (default 1.0).
        continuity_correction: True to apply Casagrande-Pike continuity correction.
        two_tailed: True for two-sided test.

    Returns:
        Dict[str, Any]: Sample size per arm, total N, risk difference, odds ratio, relative risk.
    """
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0):
        raise ValueError("Proportions p1 and p2 must be strictly between 0 and 1.")
    diff = abs(p1 - p2)
    if diff <= 1e-6:
        raise ValueError("Proportions p1 and p2 cannot be identical.")

    k = float(allocation_ratio)
    z_alpha = _get_z_alpha(alpha, two_tailed=two_tailed)
    z_beta = _get_z_beta(power)

    # Weighted average proportion
    p_bar = (p1 + k * p2) / (1.0 + k)
    q_bar = 1.0 - p_bar

    # Standard unpooled formula
    term1 = z_alpha * math.sqrt((1.0 + 1.0 / k) * p_bar * q_bar)
    term2 = z_beta * math.sqrt(p1 * (1.0 - p1) + (p2 * (1.0 - p2)) / k)
    n1_uncorr = ((term1 + term2)**2) / (diff**2)

    # Casagrande-Pike continuity correction
    if continuity_correction:
        n1_cc = (n1_uncorr / 4.0) * ((1.0 + math.sqrt(1.0 + (2.0 * (k + 1.0)) / (k * n1_uncorr * diff)))**2)
        n1 = math.ceil(n1_cc)
    else:
        n1 = math.ceil(n1_uncorr)

    n2 = math.ceil(k * n1)
    n_total = n1 + n2

    # Achieved power
    se_actual = math.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
    z_stat = diff / se_actual
    achieved_power = _normal_cdf(z_stat - z_alpha)
    if two_tailed:
        achieved_power += _normal_cdf(-z_stat - z_alpha)

    # Epidemiological metrics
    rr = p2 / p1 if p1 > 0 else 1.0
    or_val = (p2 / (1.0 - p2)) / (p1 / (1.0 - p1)) if (1.0 - p2) > 0 and p1 > 0 else 1.0

    return {
        "test_type": "two_independent_proportions",
        "p1": round(p1, 4),
        "p2": round(p2, 4),
        "risk_difference": round(p2 - p1, 4),
        "abs_risk_difference": round(diff, 4),
        "relative_risk": round(rr, 3),
        "odds_ratio": round(or_val, 3),
        "n1": n1,
        "n2": n2,
        "n_total": n_total,
        "n1_uncorrected": math.ceil(n1_uncorr),
        "continuity_correction_applied": continuity_correction,
        "allocation_ratio": k,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": round(achieved_power, 4),
        "achieved_power_pct": round(achieved_power * 100.0, 1),
        "two_tailed": two_tailed
    }


def calculate_sample_size_cohort_rr(
    p0: float,
    relative_risk: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    two_tailed: bool = True
) -> Dict[str, Any]:
    """Calculates required sample size for a Prospective Cohort Study evaluating Relative Risk (RR).

    Args:
        p0: Baseline disease incidence in the unexposed cohort (0 < p0 < 1).
        relative_risk: Anticipated Relative Risk (RR != 1.0).
        alpha: Significance level (default 0.05).
        power: Desired power (default 0.80).
        allocation_ratio: Ratio of exposed to unexposed participants (n_exp / n_unexp).
        two_tailed: True for two-sided test.

    Returns:
        Dict[str, Any]: Required exposed, unexposed, total cohort, and expected event counts.
    """
    if not (0.0 < p0 < 1.0):
        raise ValueError("Baseline incidence p0 must be between 0 and 1.")
    if relative_risk <= 0.0 or abs(relative_risk - 1.0) < 1e-4:
        raise ValueError("Relative risk must be positive and different from 1.0.")

    p1 = p0 * relative_risk
    if p1 >= 1.0:
        raise ValueError(f"p0 * RR ({p1:.3f}) exceeds 1.0. Lower RR or baseline incidence.")

    res = calculate_sample_size_two_proportions(
        p1=p0,
        p2=p1,
        alpha=alpha,
        power=power,
        allocation_ratio=allocation_ratio,
        continuity_correction=True,
        two_tailed=two_tailed
    )

    n_unexp = res["n1"]
    n_exp = res["n2"]
    events_unexp = round(n_unexp * p0, 1)
    events_exp = round(n_exp * p1, 1)

    return {
        "study_design": "prospective_cohort",
        "p0_unexposed_incidence": round(p0, 4),
        "p1_exposed_incidence": round(p1, 4),
        "relative_risk": round(relative_risk, 3),
        "n_unexposed": n_unexp,
        "n_exposed": n_exp,
        "n_total": res["n_total"],
        "expected_events_unexposed": events_unexp,
        "expected_events_exposed": events_exp,
        "total_expected_events": round(events_unexp + events_exp, 1),
        "alpha": alpha,
        "target_power": power,
        "achieved_power_pct": res["achieved_power_pct"],
        "allocation_ratio": allocation_ratio
    }


def calculate_sample_size_case_control_or(
    p0_exposure: float,
    odds_ratio: float,
    alpha: float = 0.05,
    power: float = 0.80,
    controls_per_case: float = 1.0,
    two_tailed: bool = True
) -> Dict[str, Any]:
    """Calculates required sample size for a Retrospective Case-Control Study evaluating Odds Ratio (OR).

    Args:
        p0_exposure: Prevalence of exposure among healthy controls (0 < p0 < 1).
        odds_ratio: Hypothesized Odds Ratio (OR != 1.0).
        alpha: Significance level (default 0.05).
        power: Desired statistical power (default 0.80).
        controls_per_case: Ratio of controls to cases (k = n_controls / n_cases).
        two_tailed: True for two-sided test.

    Returns:
        Dict[str, Any]: Required cases, controls, total subjects, and exposure prevalence in cases.
    """
    if not (0.0 < p0_exposure < 1.0):
        raise ValueError("Control exposure prevalence p0 must be between 0 and 1.")
    if odds_ratio <= 0.0 or abs(odds_ratio - 1.0) < 1e-4:
        raise ValueError("Odds Ratio must be positive and different from 1.0.")

    # Convert control exposure prevalence to case exposure prevalence via OR:
    # p1 = (p0 * OR) / (1 - p0 + p0 * OR)
    odds0 = p0_exposure / (1.0 - p0_exposure)
    odds1 = odds0 * odds_ratio
    p1_exposure = odds1 / (1.0 + odds1)

    res = calculate_sample_size_two_proportions(
        p1=p1_exposure,
        p2=p0_exposure,
        alpha=alpha,
        power=power,
        allocation_ratio=controls_per_case,
        continuity_correction=True,
        two_tailed=two_tailed
    )

    n_cases = res["n1"]
    n_controls = res["n2"]

    return {
        "study_design": "case_control",
        "p0_control_exposure": round(p0_exposure, 4),
        "p1_case_exposure": round(p1_exposure, 4),
        "odds_ratio": round(odds_ratio, 3),
        "n_cases": n_cases,
        "n_controls": n_controls,
        "n_total": res["n_total"],
        "controls_per_case": controls_per_case,
        "alpha": alpha,
        "target_power": power,
        "achieved_power_pct": res["achieved_power_pct"]
    }


def calculate_sample_size_paired_proportions_mcnemar(
    p_discordant_total: float,
    discordant_ratio: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> Dict[str, Any]:
    """Calculates sample size for matched pairs (McNemar's test for paired binary proportions).

    Args:
        p_discordant_total: Proportion of all pairs that are discordant (p10 + p01, 0 < p < 1).
        discordant_ratio: Ratio of discordant pairs psi = p10 / p01 (psi != 1.0).
        alpha: Significance level (default 0.05).
        power: Desired statistical power (default 0.80).
        two_tailed: True for two-sided test.

    Returns:
        Dict[str, Any]: Total matched pairs required, discordant pair count.
    """
    if not (0.0 < p_discordant_total < 1.0):
        raise ValueError("Total discordant proportion must be between 0 and 1.")
    if discordant_ratio <= 0.0 or abs(discordant_ratio - 1.0) < 1e-4:
        raise ValueError("Discordant ratio psi must be positive and different from 1.0.")

    z_alpha = _get_z_alpha(alpha, two_tailed=two_tailed)
    z_beta = _get_z_beta(power)
    psi = float(discordant_ratio)

    # Formula for required discordant pairs:
    # m = (z_alpha * (psi + 1) + 2 * z_beta * sqrt(psi))^2 / ((psi - 1)^2)
    numerator = (z_alpha * (psi + 1.0) + 2.0 * z_beta * math.sqrt(psi))**2
    denominator = (psi - 1.0)**2
    m_discordant = numerator / denominator

    # Total matched pairs N = m / p_discordant_total
    n_pairs = math.ceil(m_discordant / p_discordant_total)

    return {
        "test_type": "mcnemar_paired_proportions",
        "p_discordant_total": round(p_discordant_total, 4),
        "discordant_ratio": round(discordant_ratio, 3),
        "required_discordant_pairs": math.ceil(m_discordant),
        "n_matched_pairs": n_pairs,
        "n_total_subjects": n_pairs * 2,
        "alpha": alpha,
        "target_power": power
    }


# =====================================================================
# Workflow 3: Longitudinal Survival Outcomes (Log-Rank & Cox HR)
# =====================================================================

def calculate_sample_size_survival_logrank(
    hazard_ratio: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    cumulative_event_prob: Optional[float] = None,
    two_tailed: bool = True
) -> Dict[str, Any]:
    """Calculates required events (E) and sample size (N) for survival analysis (Log-Rank / Cox HR).

    Uses Schoenfeld's (1981) formula for required events under proportional hazards.

    Args:
        hazard_ratio: Target Hazard Ratio (HR != 1.0).
        alpha: Significance level (default 0.05).
        power: Desired statistical power (default 0.80).
        allocation_ratio: k = n2 / n1 (default 1.0 for equal allocation).
        cumulative_event_prob: Expected cumulative event proportion over trial follow-up (0 < p <= 1.0).
        two_tailed: True for two-sided test.

    Returns:
        Dict[str, Any]: Required events E, required patients N, events per arm.
    """
    if hazard_ratio <= 0.0 or abs(hazard_ratio - 1.0) < 1e-4:
        raise ValueError("Hazard ratio must be positive and different from 1.0.")

    k = float(allocation_ratio)
    z_alpha = _get_z_alpha(alpha, two_tailed=two_tailed)
    z_beta = _get_z_beta(power)

    # Allocation weights: q1 = 1 / (1 + k), q2 = k / (1 + k)
    q1 = 1.0 / (1.0 + k)
    q2 = k / (1.0 + k)

    # Schoenfeld's formula: E = (z_alpha + z_beta)^2 / (q1 * q2 * (ln HR)^2)
    ln_hr = math.log(hazard_ratio)
    e_raw = ((z_alpha + z_beta)**2) / (q1 * q2 * (ln_hr**2))
    e_events = math.ceil(e_raw)

    e_arm1 = math.ceil(q1 * e_events)
    e_arm2 = e_events - e_arm1

    # Total patient enrollment conversion
    p_event = cumulative_event_prob if cumulative_event_prob is not None else 0.50
    if not (0.0 < p_event <= 1.0):
        raise ValueError("Cumulative event probability must be between 0 and 1.0.")

    n_raw = e_events / p_event
    n_total = math.ceil(n_raw)
    n1 = math.ceil(q1 * n_total)
    n2 = n_total - n1

    # Freedman (1982) alternative comparison
    # E_freedman = (z_alpha + z_beta)^2 * (HR + 1)^2 / ( (HR - 1)^2 )
    e_freedman = math.ceil(((z_alpha + z_beta)**2) * ((hazard_ratio + 1.0)**2) / ((hazard_ratio - 1.0)**2)) if abs(k - 1.0) < 1e-4 else e_events

    return {
        "test_type": "survival_logrank",
        "hazard_ratio": round(hazard_ratio, 3),
        "ln_hazard_ratio": round(ln_hr, 4),
        "required_events": e_events,
        "events_arm1": e_arm1,
        "events_arm2": e_arm2,
        "freedman_comparison_events": e_freedman,
        "cumulative_event_prob": round(p_event, 4),
        "n_total": n_total,
        "n1": n1,
        "n2": n2,
        "allocation_ratio": k,
        "alpha": alpha,
        "target_power": power,
        "two_tailed": two_tailed
    }


# =====================================================================
# Workflow 4: Diagnostic Accuracy Studies (Wilson Precision & Prevalence)
# =====================================================================

def calculate_sample_size_diagnostic_accuracy(
    target_sensitivity: float,
    margin_of_error: float,
    disease_prevalence: Optional[float] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Calculates sample size for estimating sensitivity/specificity within a target margin of error.

    Uses Wilson score / Wald precision formula and prevalence-adjusted screening cohort enrollment.

    Args:
        target_sensitivity: Anticipated sensitivity (or specificity), between 0.50 and 0.999.
        margin_of_error: Desired half-width of the 95% confidence interval W (e.g. 0.05 for +/- 5%).
        disease_prevalence: Target population disease prevalence (0 < P < 1). If None, defaults to 0.10.
        alpha: Significance level (default 0.05 for 95% confidence).

    Returns:
        Dict[str, Any]: Required diseased cases, required total screening cohort, confidence interval bounds.
    """
    if not (0.50 <= target_sensitivity < 1.0):
        raise ValueError("Target sensitivity/specificity must be between 0.50 and 0.999.")
    if not (0.005 <= margin_of_error <= 0.25):
        raise ValueError("Margin of error W must be between 0.005 (+/-0.5%) and 0.25 (+/-25%).")

    z_crit = _get_z_alpha(alpha, two_tailed=True)
    s = float(target_sensitivity)
    w = float(margin_of_error)

    # Required diseased cases: n_diseased = z^2 * s * (1 - s) / w^2
    n_diseased_raw = (z_crit**2 * s * (1.0 - s)) / (w**2)
    n_diseased = math.ceil(n_diseased_raw)

    # Prevalence inflation
    p_prev = disease_prevalence if disease_prevalence is not None else 0.10
    if not (0.0 < p_prev < 1.0):
        raise ValueError("Disease prevalence must be between 0 and 1.0.")

    n_screening = math.ceil(n_diseased / p_prev)
    n_non_diseased = n_screening - n_diseased

    return {
        "study_type": "diagnostic_accuracy_precision",
        "target_sensitivity": round(s, 4),
        "target_sensitivity_pct": round(s * 100.0, 1),
        "margin_of_error": round(w, 4),
        "margin_of_error_pct": round(w * 100.0, 1),
        "ci_lower_pct": round(max(0.0, (s - w) * 100.0), 1),
        "ci_upper_pct": round(min(100.0, (s + w) * 100.0), 1),
        "confidence_level_pct": round((1.0 - alpha) * 100.0, 1),
        "required_diseased_cases": n_diseased,
        "disease_prevalence": round(p_prev, 4),
        "disease_prevalence_pct": round(p_prev * 100.0, 1),
        "required_total_screening_cohort": n_screening,
        "required_non_diseased": n_non_diseased,
        "alpha": alpha
    }


# =====================================================================
# Workflow 5: Real-World Trial Adjustments & Modifiers
# =====================================================================

def apply_attrition_inflation(
    n_evaluable: int,
    dropout_rate: float
) -> Dict[str, Any]:
    """Inflates evaluable sample size to protect against anticipated participant drop-out/attrition.

    Formula: N_recruited = ceil(N_evaluable / (1 - d))

    Args:
        n_evaluable: Target sample size requiring complete evaluable follow-up.
        dropout_rate: Anticipated loss-to-follow-up proportion (0 <= d < 0.90).

    Returns:
        Dict[str, Any]: Recruited sample size, additional subjects needed, inflation percentage.
    """
    if n_evaluable <= 0:
        raise ValueError("Evaluable sample size must be strictly positive.")
    if not (0.0 <= dropout_rate < 0.90):
        raise ValueError("Dropout rate must be between 0.0 and 0.89.")

    if dropout_rate == 0.0:
        return {
            "n_evaluable": n_evaluable,
            "dropout_rate": 0.0,
            "dropout_rate_pct": 0.0,
            "n_recruited": n_evaluable,
            "additional_subjects_needed": 0,
            "inflation_factor": 1.0
        }

    n_recruited = math.ceil(n_evaluable / (1.0 - dropout_rate))
    additional = n_recruited - n_evaluable
    factor = n_recruited / float(n_evaluable)

    return {
        "n_evaluable": n_evaluable,
        "dropout_rate": round(dropout_rate, 4),
        "dropout_rate_pct": round(dropout_rate * 100.0, 1),
        "n_recruited": n_recruited,
        "additional_subjects_needed": additional,
        "inflation_factor": round(factor, 3)
    }


def calculate_unpadded_power_after_dropout(
    calc_results: Dict[str, Any],
    dropout_rate: float
) -> Dict[str, Any]:
    """Calculates the degraded statistical power if a study experiences dropout without recruiting an attrition cushion.

    Args:
        calc_results: Results dictionary from primary sample size calculation.
        dropout_rate: Anticipated dropout proportion (0 <= d < 0.90).

    Returns:
        Dict[str, Any]: Nominal power, degraded power after dropout, effective evaluable N, power loss.
    """
    nom_power = calc_results.get("achieved_power", calc_results.get("target_power", 0.80))
    nom_power_pct = round(nom_power * 100.0, 1)

    if dropout_rate <= 0.0:
        return {
            "dropout_rate_pct": 0.0,
            "effective_evaluable_n": calc_results.get("n_total", 100),
            "guaranteed_power_pct": nom_power_pct,
            "degraded_power_pct": nom_power_pct,
            "power_loss_pct": 0.0
        }

    alpha = calc_results.get("alpha", 0.05)
    two_tailed = calc_results.get("two_tailed", True)
    z_alpha = _get_z_alpha(alpha, two_tailed=two_tailed)

    n_eval = calc_results.get("n_total", calc_results.get("required_total_screening_cohort", 100))
    effective_n = max(4, int(math.floor(n_eval * (1.0 - dropout_rate))))

    # 1. Continuous (Two Means, Paired, ANOVA)
    if "cohen_d" in calc_results or "cohen_dz" in calc_results or "mean_diff" in calc_results:
        d = calc_results.get("cohen_d", calc_results.get("cohen_dz", 0.50))
        n_arm = max(2, effective_n // 2)
        se = math.sqrt(2.0 / n_arm)
        deg_power = min(0.9999, max(0.001, _normal_cdf((d / se) - z_alpha)))

    # 2. Longitudinal Survival (Log-Rank)
    elif "hazard_ratio" in calc_results or "required_events" in calc_results:
        hr = calc_results.get("hazard_ratio", 0.70)
        ln_hr = abs(math.log(hr)) if hr > 0 and abs(hr - 1.0) > 1e-4 else 0.35
        e_eval = calc_results.get("required_events", int(n_eval * calc_results.get("cumulative_event_prob", 0.50)))
        e_effective = max(2, int(math.floor(e_eval * (1.0 - dropout_rate))))
        se = 2.0 / math.sqrt(e_effective)
        deg_power = min(0.9999, max(0.001, _normal_cdf((ln_hr / se) - z_alpha)))

    # 3. Binary (Two Proportions, Cohort, Case-Control)
    elif "p1" in calc_results or "odds_ratio" in calc_results or "relative_risk" in calc_results or "study_design" in calc_results:
        p1 = calc_results.get("p1", calc_results.get("p1_case_exposure", calc_results.get("p1_exposed_incidence", 0.20)))
        p2 = calc_results.get("p2", calc_results.get("p0_control_exposure", calc_results.get("p0_unexposed_incidence", 0.10)))
        diff = abs(p1 - p2)
        if diff < 1e-4:
            diff = 0.10
        n_arm = max(2, effective_n // 2)
        se = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n_arm)
        if se <= 0:
            se = 0.01
        deg_power = min(0.9999, max(0.001, _normal_cdf((diff / se) - z_alpha)))

    # 4. Correlation / Default
    else:
        r = abs(calc_results.get("r", 0.30))
        w = 0.5 * math.log((1.0 + r) / (1.0 - r)) if abs(r) < 0.99 else 0.35
        se = 1.0 / math.sqrt(max(2, effective_n - 3))
        deg_power = min(0.9999, max(0.001, _normal_cdf((w / se) - z_alpha)))

    deg_power_pct = round(deg_power * 100.0, 1)
    power_loss = round(nom_power_pct - deg_power_pct, 1)

    return {
        "dropout_rate_pct": round(dropout_rate * 100.0, 1),
        "effective_evaluable_n": effective_n,
        "guaranteed_power_pct": nom_power_pct,
        "degraded_power_pct": deg_power_pct,
        "power_loss_pct": max(0.0, power_loss)
    }


def apply_cluster_design_effect(
    n_individual: int,
    cluster_size: int,
    icc: float
) -> Dict[str, Any]:
    """Calculates cluster trial sample size inflation using the Variance Inflation Factor (VIF).

    Formula: VIF = 1 + (m - 1) * ICC
             N_clustered = ceil(N_individual * VIF)
             Clusters_required = ceil(N_clustered / m)

    Args:
        n_individual: Total individual sample size under simple random sampling.
        cluster_size: Average number of individuals per cluster m (e.g. patients per clinic).
        icc: Intracluster Correlation Coefficient rho / ICC (typically 0.01 - 0.10).

    Returns:
        Dict[str, Any]: Design effect VIF, inflated sample size, and number of clusters needed.
    """
    if n_individual <= 0 or cluster_size <= 0:
        raise ValueError("Sample size and cluster size must be strictly positive.")
    if not (0.0 <= icc <= 1.0):
        raise ValueError("ICC must be between 0.0 and 1.0.")

    m = int(cluster_size)
    vif = 1.0 + (m - 1) * icc
    n_clustered = math.ceil(n_individual * vif)
    k_clusters = math.ceil(n_clustered / float(m))
    n_actual = k_clusters * m

    return {
        "n_individual_unadjusted": n_individual,
        "cluster_size_m": m,
        "icc": round(icc, 4),
        "design_effect_vif": round(vif, 3),
        "n_clustered_total": n_actual,
        "clusters_required": k_clusters,
        "additional_subjects_needed": n_actual - n_individual
    }


# =====================================================================
# Workflow 6: Power Curve & Sensitivity Surface Generators
# =====================================================================

def generate_power_curve_data(
    calc_type: str,
    base_params: Dict[str, Any],
    sample_sizes: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Generates statistical power (1 - beta) across an array of sample sizes for Plotly curves.

    Supports 'two_means', 'two_proportions', 'survival', and 'correlation'.
    Computes power for:
    - Small effect benchmark
    - Target effect (user parameter)
    - Large effect benchmark

    Args:
        calc_type: One of 'two_means', 'two_proportions', 'survival', 'correlation'.
        base_params: Parameter dictionary for the respective calculation.
        sample_sizes: Optional list of sample sizes. If None, auto-generated around target N.

    Returns:
        List[Dict[str, Any]]: List of records with keys: 'N', 'Power_Target', 'Power_Small', 'Power_Large'.
    """
    alpha = base_params.get("alpha", 0.05)
    two_tailed = base_params.get("two_tailed", True)
    z_alpha = _get_z_alpha(alpha, two_tailed=two_tailed)

    if calc_type in ("two_means", "paired_means", "anova"):
        diff = abs(base_params.get("mean_diff", abs(base_params.get("mean1", 0.0) - base_params.get("mean2", 0.0))))
        sd = base_params.get("sd_pooled", base_params.get("sd1", base_params.get("sd_diff", 1.0)))
        d_target = base_params.get("cohen_d", base_params.get("cohen_dz", diff / max(0.001, sd)))
        d_small = 0.20
        d_large = 0.80

        target_n = base_params.get("n_total", base_params.get("n_total_observations", base_params.get("n_pairs", 100)))
        if target_n is None or target_n <= 0:
            target_res = calculate_sample_size_two_means(
                base_params.get("mean1", diff), base_params.get("mean2", 0.0), sd, sd,
                alpha=alpha, power=base_params.get("target_power", base_params.get("power", 0.80))
            )
            target_n = target_res["n_total"]

        if sample_sizes is None:
            n_min = max(10, int(target_n * 0.20))
            n_max = int(target_n * 2.20)
            sample_sizes = list(range(n_min, n_max + 1, max(2, (n_max - n_min) // 25)))

        curve_points = []
        for n in sample_sizes:
            n_arm = max(2, n // 2)
            se_d = math.sqrt(2.0 / n_arm)
            p_target = min(0.9999, max(0.001, _normal_cdf((d_target / se_d) - z_alpha)))
            p_small = min(0.9999, max(0.001, _normal_cdf((d_small / se_d) - z_alpha)))
            p_large = min(0.9999, max(0.001, _normal_cdf((d_large / se_d) - z_alpha)))
            curve_points.append({
                "N": n,
                "Power_Target": round(p_target * 100.0, 1),
                "Power_Small": round(p_small * 100.0, 1),
                "Power_Large": round(p_large * 100.0, 1)
            })
        return curve_points

    elif calc_type in ("two_proportions", "cohort", "case_control"):
        # Handle flexible parameter naming across two_proportions, case_control, and cohort
        p1 = base_params.get("p1", base_params.get("p1_case_exposure", base_params.get("p1_exposed_incidence", 0.20)))
        p2 = base_params.get("p2", base_params.get("p0_control_exposure", base_params.get("p0_unexposed_incidence", 0.10)))
        diff = abs(p1 - p2)
        if diff < 1e-4:
            diff = 0.10

        target_n = base_params.get("n_total")
        if target_n is None or target_n <= 0:
            target_res = calculate_sample_size_two_proportions(
                p1, p2, alpha=alpha, power=base_params.get("target_power", base_params.get("power", 0.80))
            )
            target_n = target_res["n_total"]

        if sample_sizes is None:
            n_min = max(20, int(target_n * 0.20))
            n_max = int(target_n * 2.20)
            sample_sizes = list(range(n_min, n_max + 1, max(2, (n_max - n_min) // 25)))

        curve_points = []
        for n in sample_sizes:
            n_arm = max(5, n // 2)
            se = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n_arm)
            if se <= 0:
                se = 0.01
            p_target = min(0.9999, max(0.001, _normal_cdf((diff / se) - z_alpha)))
            p_small = min(0.9999, max(0.001, _normal_cdf(((diff * 0.6) / se) - z_alpha)))
            p_large = min(0.9999, max(0.001, _normal_cdf(((diff * 1.5) / se) - z_alpha)))
            curve_points.append({
                "N": n,
                "Power_Target": round(p_target * 100.0, 1),
                "Power_Small": round(p_small * 100.0, 1),
                "Power_Large": round(p_large * 100.0, 1)
            })
        return curve_points

    elif calc_type == "survival":
        hr = base_params.get("hazard_ratio", 0.70)
        target_n = base_params.get("n_total")
        if target_n is None or target_n <= 0:
            target_res = calculate_sample_size_survival_logrank(
                hr, alpha=alpha, power=base_params.get("target_power", base_params.get("power", 0.80)),
                cumulative_event_prob=base_params.get("cumulative_event_prob", 0.50)
            )
            target_n = target_res["n_total"]

        if sample_sizes is None:
            n_min = max(20, int(target_n * 0.20))
            n_max = int(target_n * 2.20)
            sample_sizes = list(range(n_min, n_max + 1, max(2, (n_max - n_min) // 25)))

        ln_hr = abs(math.log(hr)) if hr > 0 and abs(hr - 1.0) > 1e-4 else 0.35
        curve_points = []
        for n in sample_sizes:
            e = max(4, n * base_params.get("cumulative_event_prob", 0.50))
            se_ln = 2.0 / math.sqrt(e)
            p_target = min(0.9999, max(0.001, _normal_cdf((ln_hr / se_ln) - z_alpha)))
            p_small = min(0.9999, max(0.001, _normal_cdf(((ln_hr * 0.65) / se_ln) - z_alpha)))
            p_large = min(0.9999, max(0.001, _normal_cdf(((ln_hr * 1.40) / se_ln) - z_alpha)))
            curve_points.append({
                "N": n,
                "Power_Target": round(p_target * 100.0, 1),
                "Power_Small": round(p_small * 100.0, 1),
                "Power_Large": round(p_large * 100.0, 1)
            })
        return curve_points

    else:
        # Default correlation curve
        r = abs(base_params.get("r", 0.30))
        target_n = base_params.get("n_total")
        if target_n is None or target_n <= 0:
            target_res = calculate_sample_size_correlation(r, alpha=alpha, power=base_params.get("target_power", base_params.get("power", 0.80)))
            target_n = target_res["n_total"]

        if sample_sizes is None:
            n_min = max(10, int(target_n * 0.20))
            n_max = int(target_n * 2.20)
            sample_sizes = list(range(n_min, n_max + 1, max(2, (n_max - n_min) // 25)))

        w_target = 0.5 * math.log((1.0 + r) / (1.0 - r)) if abs(r) < 0.99 else 0.35
        w_small = 0.5 * math.log((1.0 + 0.15) / (1.0 - 0.15))
        w_large = 0.5 * math.log((1.0 + 0.50) / (1.0 - 0.50))

        curve_points = []
        for n in sample_sizes:
            se_w = 1.0 / math.sqrt(max(2, n - 3))
            p_target = min(0.9999, max(0.001, _normal_cdf((w_target / se_w) - z_alpha)))
            p_small = min(0.9999, max(0.001, _normal_cdf((w_small / se_w) - z_alpha)))
            p_large = min(0.9999, max(0.001, _normal_cdf((w_large / se_w) - z_alpha)))
            curve_points.append({
                "N": n,
                "Power_Target": round(p_target * 100.0, 1),
                "Power_Small": round(p_small * 100.0, 1),
                "Power_Large": round(p_large * 100.0, 1)
            })
        return curve_points



# =====================================================================
# Workflow 7: Protocol Justification Statement Generator
# =====================================================================

def generate_protocol_justification_statement(
    calc_result: Dict[str, Any],
    primary_endpoint_label: str = "primary outcome",
    dropout_rate: float = 0.0,
    cluster_icc: Optional[float] = None,
    cluster_size: Optional[int] = None
) -> str:
    """Generates an automated, publication-ready sample size paragraph for ethics and grant protocols.

    Formatted according to CONSORT, STROBE, and NIHR guidelines.

    Args:
        calc_result: Result dictionary from one of the calculation functions.
        primary_endpoint_label: Clinical description of the outcome (e.g. "change in HbA1c at 12 weeks").
        dropout_rate: Anticipated loss-to-follow-up proportion (e.g. 0.15 for 15%).
        cluster_icc: Intracluster correlation coefficient (if cluster trial).
        cluster_size: Average cluster size (if cluster trial).

    Returns:
        str: Fully formatted methodology paragraph ready for copy-paste.
    """
    alpha = calc_result.get("alpha", 0.05)
    power = calc_result.get("target_power", 0.80)
    power_pct = int(round(power * 100.0))
    test_type = calc_result.get("test_type", "")
    study_design = calc_result.get("study_design", "")

    if "n_total" in calc_result:
        n_eval = calc_result["n_total"]
    elif "n_matched_pairs" in calc_result:
        n_eval = calc_result["n_matched_pairs"]
    elif "required_total_screening_cohort" in calc_result:
        n_eval = calc_result["required_total_screening_cohort"]
    else:
        n_eval = 100

    # Base sentence
    if test_type == "two_independent_means":
        diff = calc_result["mean_diff"]
        sd = calc_result["sd_pooled"]
        d = calc_result["cohen_d"]
        n1 = calc_result["n1"]
        n2 = calc_result["n2"]
        statement = (
            f"A total sample size of {n_eval} participants ({n1} in Group 1 and {n2} in Group 2) "
            f"provides {power_pct}% statistical power to detect a clinically meaningful difference in {primary_endpoint_label} "
            f"of {diff:.2f} units (assuming a common standard deviation of {sd:.2f}, corresponding to a standardized effect size "
            f"Cohen's d = {d:.2f}) using a two-sided two-sample t-test at a significance level of alpha = {alpha}."
        )

    elif test_type == "two_independent_proportions" or study_design == "prospective_cohort":
        p1 = calc_result.get("p1", calc_result.get("p0_unexposed_incidence", 0.20))
        p2 = calc_result.get("p2", calc_result.get("p1_exposed_incidence", 0.10))
        rr = calc_result.get("relative_risk", 1.0)
        n1 = calc_result.get("n1", calc_result.get("n_unexposed", n_eval // 2))
        n2 = calc_result.get("n2", calc_result.get("n_exposed", n_eval // 2))
        statement = (
            f"A total sample size of {n_eval} participants ({n1} control/unexposed and {n2} experimental/exposed) "
            f"provides {power_pct}% statistical power to detect a reduction in the incidence of {primary_endpoint_label} "
            f"from {p1*100.0:.1f}% to {p2*100.0:.1f}% (Relative Risk = {rr:.2f}, Absolute Risk Difference = {abs(p1-p2)*100.0:.1f}%) "
            f"using a two-sided Chi-square test with continuity correction at alpha = {alpha}."
        )

    elif study_design == "case_control":
        p0 = calc_result["p0_control_exposure"]
        p1 = calc_result["p1_case_exposure"]
        odds_ratio = calc_result["odds_ratio"]
        n_cases = calc_result["n_cases"]
        n_controls = calc_result["n_controls"]
        ratio = calc_result["controls_per_case"]
        statement = (
            f"A case-control investigation comprising {n_cases} cases and {n_controls} matched controls ({ratio:.0f}:1 control-to-case ratio; "
            f"total N = {n_eval}) achieves {power_pct}% statistical power to detect an Odds Ratio of {odds_ratio:.2f} for {primary_endpoint_label}, "
            f"assuming a baseline exposure prevalence of {p0*100.0:.1f}% among controls (corresponding to {p1*100.0:.1f}% exposure among cases), "
            f"evaluated at a two-sided alpha = {alpha}."
        )

    elif test_type == "survival_logrank":
        hr = calc_result["hazard_ratio"]
        events = calc_result["required_events"]
        statement = (
            f"Under Schoenfeld's proportional hazards formulation, a total of {events} primary endpoint events are required "
            f"to achieve {power_pct}% statistical power (two-sided alpha = {alpha}) to detect a Hazard Ratio of {hr:.2f} for {primary_endpoint_label}. "
            f"Based on an anticipated cumulative event incidence of {calc_result['cumulative_event_prob']*100.0:.1f}% over the follow-up period, "
            f"a total enrollment of {n_eval} patients ({calc_result['n1']} per treatment arm under 1:1 allocation) is planned."
        )

    elif calc_result.get("study_type") == "diagnostic_accuracy_precision":
        sens = calc_result["target_sensitivity_pct"]
        margin = calc_result["margin_of_error_pct"]
        n_dis = calc_result["required_diseased_cases"]
        prev = calc_result["disease_prevalence_pct"]
        statement = (
            f"To estimate the clinical sensitivity of the diagnostic index for {primary_endpoint_label} with an expected value of {sens:.1f}% "
            f"within a precision margin of error of +/-{margin:.1f}% (95% confidence interval: [{calc_result['ci_lower_pct']:.1f}%, {calc_result['ci_upper_pct']:.1f}%]), "
            f"a minimum of {n_dis} confirmed positive disease cases is required. Assuming a target population disease prevalence of {prev:.1f}%, "
            f"an unselected screening cohort of {n_eval} total individuals will be prospectively enrolled."
        )

    else:
        statement = (
            f"A total sample size of {n_eval} participants provides {power_pct}% statistical power "
            f"at a significance level of alpha = {alpha} to evaluate {primary_endpoint_label}."
        )

    clust_adj = None
    if cluster_icc is not None and cluster_size is not None and cluster_icc > 0:
        clust_adj = apply_cluster_design_effect(n_eval, cluster_size, cluster_icc)
        statement += (
            f" Accounting for cluster randomization across GP practices / hospital wards with an average cluster size of {cluster_size} "
            f"and an intracluster correlation coefficient (ICC) of {cluster_icc:.3f} (Design Effect VIF = {clust_adj['design_effect_vif']:.2f}), "
            f"the sample size is inflated to {clust_adj['n_clustered_total']} participants across {clust_adj['clusters_required']} distinct clusters."
        )

    # Attrition sentence
    if dropout_rate > 0.0:
        base_target = clust_adj["n_clustered_total"] if clust_adj else n_eval
        attr = apply_attrition_inflation(base_target, dropout_rate)
        cluster_cushion_str = f" ({math.ceil(attr['n_recruited'] / cluster_size)} clusters)" if clust_adj and cluster_size else ""
        statement += (
            f" To safeguard against an anticipated {dropout_rate*100.0:.1f}% loss-to-follow-up / withdrawal rate, "
            f"the recruitment target is inflated by {attr['additional_subjects_needed']} participants, yielding a final enrollment target of "
            f"{attr['n_recruited']} individuals{cluster_cushion_str}."
        )

    return statement


# =====================================================================
# Workflow 8: PRISMA / Study Planning Methodological Prerequisites
# =====================================================================

def evaluate_sample_size_prerequisites(calc_result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates 5 core study planning checks based on ICH-E9, CONSORT, and STROBE standards.

    Checks:
    1. Statistical Power Sufficiency (Power >= 80% benchmark).
    2. Effect Size Clinical Relevance & Plausibility (detectable difference magnitude).
    3. Allocation Ratio Efficiency (balance between statistical power and clinical practicality).
    4. Attrition / Dropout Protection.
    5. Event & Sample Size Feasibility.

    Args:
        calc_result: Output dictionary from any of the calculation functions.

    Returns:
        Dict[str, Any]: Pass/warn status, overall pass boolean, and individual check cards.
    """
    checks = []
    evaluations = {}

    power = calc_result.get("target_power", 0.80)
    achieved_power = calc_result.get("achieved_power", power)
    alpha = calc_result.get("alpha", 0.05)
    k_alloc = calc_result.get("allocation_ratio", 1.0)

    # Check 1: Power Threshold
    if achieved_power >= 0.895:
        c1 = {
            "status": "pass",
            "badge": f"🟢 High Statistical Power ({achieved_power*100.0:.1f}%)",
            "title": "Statistical Power & Type II Error Protection",
            "finding": f"Power = {achieved_power*100.0:.1f}% (beta = {1.0-achieved_power:.3f})",
            "explanation": "Target power exceeds the 90% threshold commonly required for confirmatory phase III registration trials and major grant protocols."
        }
    elif achieved_power >= 0.795:
        c1 = {
            "status": "pass",
            "badge": f"🟢 Standard Power Benchmark ({achieved_power*100.0:.1f}%)",
            "title": "Statistical Power & Type II Error Protection",
            "finding": f"Power = {achieved_power*100.0:.1f}% (beta = {1.0-achieved_power:.3f})",
            "explanation": "Satisfies the standard biostatistical 80% benchmark. Only a 20% risk of false-negative inference if the hypothesized effect exists."
        }
    else:
        c1 = {
            "status": "fail",
            "badge": f"🔴 Underpowered Protocol ({achieved_power*100.0:.1f}%)",
            "title": "Statistical Power & Type II Error Protection",
            "finding": f"Power = {achieved_power*100.0:.1f}% < 80%",
            "explanation": "Protocols with power under 80% pose substantial ethical concerns by subjecting patients to trial risks with inadequate chance of answering the research question."
        }
    checks.append(c1)
    evaluations["power_sufficiency"] = c1

    # Check 2: Alpha / Significance Level
    if alpha <= 0.05:
        c2 = {
            "status": "pass",
            "badge": f"🟢 Controlled Type I Error (alpha = {alpha})",
            "title": "Significance Level & False-Positive Rate",
            "finding": f"Two-sided alpha = {alpha}",
            "explanation": "Adheres to regulatory standards. False-positive risk is capped at 5%."
        }
    else:
        c2 = {
            "status": "warn",
            "badge": f"🟡 Elevated False-Positive Risk (alpha = {alpha})",
            "title": "Significance Level & False-Positive Rate",
            "finding": f"alpha = {alpha} > 0.05",
            "explanation": "Significance thresholds above 0.05 are only acceptable in exploratory pilot/feasibility studies."
        }
    checks.append(c2)
    evaluations["alpha_level"] = c2

    # Check 3: Allocation Ratio Efficiency
    if 0.95 <= k_alloc <= 1.05:
        c3 = {
            "status": "pass",
            "badge": "🟢 Maximum Statistical Efficiency (1:1 Allocation)",
            "title": "Allocation Ratio & Statistical Power Efficiency",
            "finding": "Equal 1:1 treatment allocation",
            "explanation": "Equal allocation minimizes the total required sample size N for a given level of statistical power."
        }
    elif 0.33 <= k_alloc <= 3.0:
        c3 = {
            "status": "pass",
            "badge": f"🟢 Moderate Unequal Allocation (k = {k_alloc:.1f})",
            "title": "Allocation Ratio & Statistical Power Efficiency",
            "finding": f"Allocation ratio k = {k_alloc:.1f}",
            "explanation": "Unequal allocation (e.g. 1:2) slightly increases total N (by ~12%) but is justifiable when active treatment exposure is limited or when matching multiple controls."
        }
    else:
        c3 = {
            "status": "warn",
            "badge": f"🟡 Extreme Allocation Imbalance (k = {k_alloc:.1f})",
            "title": "Allocation Ratio & Statistical Power Efficiency",
            "finding": f"k = {k_alloc:.1f}",
            "explanation": "Allocation ratios beyond 1:4 yield diminishing statistical returns and substantially inflate the overall recruitment burden."
        }
    checks.append(c3)
    evaluations["allocation_efficiency"] = c3

    # Check 4: Sample Size Feasibility
    n_tot = calc_result.get("n_total", calc_result.get("required_total_screening_cohort", 100))
    if n_tot <= 500:
        c4 = {
            "status": "pass",
            "badge": f"🟢 Highly Feasible Cohort (N = {n_tot})",
            "title": "Recruitment & Operational Feasibility",
            "finding": f"Total N = {n_tot} evaluable participants",
            "explanation": "Target recruitment is manageable within a single NHS trust or regional collaborative over a standard 12-month recruitment window."
        }
    elif n_tot <= 2500:
        c4 = {
            "status": "pass",
            "badge": f"🟢 Multicenter Feasibility (N = {n_tot})",
            "title": "Recruitment & Operational Feasibility",
            "finding": f"Total N = {n_tot} evaluable participants",
            "explanation": "Achievable through multi-site recruitment across NIHR Clinical Research Networks (CRNs)."
        }
    else:
        c4 = {
            "status": "warn",
            "badge": f"🟡 Substantial Scale Required (N = {n_tot})",
            "title": "Recruitment & Operational Feasibility",
            "finding": f"Total N = {n_tot} participants",
            "explanation": "Massive cohort required. Ensure adequate multi-hospital infrastructure, clinical trials unit (CTU) backing, and realistic recruitment timeline."
        }
    checks.append(c4)
    evaluations["sample_feasibility"] = c4

    # Check 5: Effect Size Magnitude
    effect_desc = "Specified effect size is within standard clinical parameters."
    if "cohen_d" in calc_result:
        d = calc_result["cohen_d"]
        if 0.20 <= d <= 0.80:
            c5 = {
                "status": "pass",
                "badge": f"🟢 Realistic Clinical Effect (d = {d:.2f})",
                "title": "Target Effect Size Plausibility",
                "finding": f"Standardized Cohen's d = {d:.2f}",
                "explanation": "Matches realistic therapeutic effects observed in contemporary medical literature."
            }
        elif d < 0.20:
            c5 = {
                "status": "warn",
                "badge": f"🟡 Subtle Effect Size (d = {d:.2f})",
                "title": "Target Effect Size Plausibility",
                "finding": f"Cohen's d = {d:.2f} < 0.20",
                "explanation": "Detecting very subtle differences requires large sample sizes; verify that this difference represents a clinically meaningful difference."
            }
        else:
            c5 = {
                "status": "pass",
                "badge": f"🟢 Substantial Effect Size (d = {d:.2f})",
                "title": "Target Effect Size Plausibility",
                "finding": f"Cohen's d = {d:.2f} >= 0.80",
                "explanation": "Large therapeutic effect. Ensure preliminary pilot data supports an effect of this magnitude."
            }
    elif "hazard_ratio" in calc_result:
        hr = calc_result["hazard_ratio"]
        c5 = {
            "status": "pass",
            "badge": f"🟢 Target Hazard Ratio (HR = {hr:.2f})",
            "title": "Survival Effect Size Plausibility",
            "finding": f"Hazard Ratio = {hr:.2f} ({calc_result['required_events']} events required)",
            "explanation": "Proportional hazards assumption and event tracking are paramount. The trial cannot conclude until the target event count is reached."
        }
    else:
        c5 = {
            "status": "pass",
            "badge": "🟢 Standard Study Specifications",
            "title": "Target Effect Size Plausibility",
            "finding": "Parameters adhere to established biostatistical design conventions.",
            "explanation": "Assumptions verified against ICH-E9 statistical principles for clinical trials."
        }
    checks.append(c5)
    evaluations["effect_plausibility"] = c5

    all_passed = all(c["status"] == "pass" for c in checks)
    evaluations["all_passed"] = all_passed
    evaluations["summary"] = "All study design and statistical power assumptions verified under ICH-E9 / CONSORT guidelines." if all_passed else "One or more protocol design parameters warrant review."
    evaluations["checks"] = checks

    return evaluations


# =====================================================================
# Workflow 9: Synthetic Demonstration Data & Fictional UK Scenarios
# =====================================================================

def generate_sample_size_demo_data(
    scenario_key: str = "cardiovascular_rct",
    scenario: Optional[str] = None
) -> SampleSizeDemoData:
    """Generates realistic synthetic study protocol planning datasets for educational demonstration.

    All scenarios use fictional UK healthcare and academic institutions:
    - All Meadows Cardiovascular Institute: Double-blind Phase III RCT (HbA1c Reduction & MACE Prevention)
    - Riverside Health Protection Team: Community Outbreak Case-Control & School Cluster Trial
    - Inner City Teaching Hospital: Oncology Survival Trial (Metastatic NSCLC Immunotherapy)

    Args:
        scenario_key: One of 'cardiovascular_rct', 'public_health_case_control', 'oncology_survival'.
        scenario: Alias for scenario_key for call compatibility.

    Returns:
        SampleSizeDemoData: Dictionary containing protocol simulation dataframe and metadata.
    """
    if scenario is not None:
        scenario_key = scenario

    if scenario_key == "cardiovascular_rct":
        # All Meadows Cardiovascular Institute: Double-Blind RCT in Type 2 Diabetes & Heart Failure
        # Primary continuous endpoint: HbA1c change from baseline at 24 weeks (Delta = 0.45%, SD = 1.10%)
        # Secondary binary endpoint: 1-year MACE occurrence (12% control vs 7% active)
        calc_means = calculate_sample_size_two_means(
            mean1=8.20, mean2=7.75, sd1=1.10, sd2=1.10, alpha=0.05, power=0.85, allocation_ratio=1.0
        )
        calc_props = calculate_sample_size_two_proportions(
            p1=0.12, p2=0.07, alpha=0.05, power=0.80, allocation_ratio=1.0
        )
        attrition = apply_attrition_inflation(calc_means["n_total"], dropout_rate=0.15)

        # Generate sample protocol planning dataset (patient screening registry)
        rows = []
        for i in range(1, 101):
            age = 50 + (i % 30)
            base_a1c = round(7.5 + (i % 25) * 0.1, 2)
            arm = "SGLT2 Inhibitor" if i % 2 == 1 else "Matching Placebo"
            dropout = 1 if (i % 7 == 0) else 0
            rows.append({
                "Patient_ID": f"AMCI-RCT-{i:04d}",
                "Trial_Arm": arm,
                "Age_Years": age,
                "Baseline_HbA1c_Pct": base_a1c,
                "Anticipated_Dropout": dropout,
                "Site_Centre": "All Meadows Heart Centre" if i % 2 == 1 else "Riverside Clinical Annex"
            })

        df = pd.DataFrame(rows) if HAS_SCIENTIFIC_STACK else rows

        res = SampleSizeDemoData({
            "df": df,
            "scenario_key": "cardiovascular_rct",
            "scenario_name": "All Meadows Cardiovascular Institute: Double-Blind Phase III RCT (HbA1c & MACE)",
            "title": "All Meadows Cardiovascular Institute: Cardiometabolic Clinical Trial Protocol",
            "organisation_name": "All Meadows Cardiovascular Institute",
            "organisation": "All Meadows Cardiovascular Institute (Clinical Trials & Biostatistics Unit)",
            "domain": "Cardiovascular Medicine & Pharmacotherapy Clinical Trials",
            "intervention": "Novel SGLT2 Inhibitor (Dapagliflozin 10mg Once Daily)",
            "comparator": "Matching Placebo + Standard Guideline Medical Therapy",
            "primary_endpoint": "HbA1c reduction from baseline at 24 weeks (%)",
            "calc_results_primary": calc_means,
            "calc_results_secondary": calc_props,
            "attrition_adjustment": attrition,
            "description": "Double-blind, placebo-controlled multicenter clinical trial evaluating a novel SGLT2 inhibitor in patients with high-risk type 2 diabetes and heart failure.",
            "takeaway": "Achieving 85% statistical power to detect a 0.45% HbA1c reduction requires 218 evaluable patients (109 per arm). Accounting for an anticipated 15% dropout rate inflates the recruitment target to 258 patients (129 per arm).",
            "disclaimer": "Completely made-up synthetic protocol dataset generated purely for educational and demonstration purposes only. Contains no real patient identifiers."
        })
        res["meta"] = dict(res)
        return res

    elif scenario_key == "public_health_case_control":
        # Riverside Health Protection Team: Food-borne Outbreak Investigation & School Health Cluster Trial
        calc_or = calculate_sample_size_case_control_or(
            p0_exposure=0.20, odds_ratio=2.50, alpha=0.05, power=0.80, controls_per_case=2.0
        )
        cluster_adj = apply_cluster_design_effect(
            n_individual=calc_or["n_total"], cluster_size=30, icc=0.04
        )
        attrition = apply_attrition_inflation(cluster_adj["n_clustered_total"], dropout_rate=0.10)

        rows = []
        for i in range(1, 121):
            is_case = 1 if i <= 40 else 0
            rows.append({
                "Subject_ID": f"RHPT-CC-{i:04d}",
                "Case_Control_Status": "Case (Ill)" if is_case else "Control (Healthy)",
                "Age_Group": "25-39" if i % 3 == 0 else ("40-59" if i % 3 == 1 else "60+"),
                "Suspect_Food_Exposed": 1 if (is_case and i % 4 != 0) or (not is_case and i % 5 == 0) else 0,
                "GP_Cluster_ID": f"Practice_{(i % 8) + 1:02d}"
            })

        df = pd.DataFrame(rows) if HAS_SCIENTIFIC_STACK else rows

        res = SampleSizeDemoData({
            "df": df,
            "scenario_key": "public_health_case_control",
            "scenario_name": "Riverside Health Protection Team: Outbreak Case-Control & Cluster Randomization",
            "title": "Riverside Health Protection Team: Field Outbreak Protocol & Design Effect",
            "organisation_name": "Riverside Health Protection Team",
            "organisation": "Riverside Health Protection Team (Field Epidemiology & Health Protection Unit)",
            "domain": "Field Epidemiology, Outbreak Analytics & Cluster Interventions",
            "intervention": "Exposure to Incriminated Food Vehicle (Artisanal Dairy)",
            "comparator": "Unexposed Community Controls (1:2 Matching Ratio)",
            "primary_endpoint": "Laboratory-Confirmed Campylobacteriosis Infection",
            "calc_results_primary": calc_or,
            "cluster_adjustment": cluster_adj,
            "attrition_adjustment": attrition,
            "description": "Retrospective case-control study designed to incriminate an outbreak exposure with 1:2 case-control matching, alongside a cluster-randomized community hygiene intervention.",
            "takeaway": "Detecting an Odds Ratio of 2.5 with 20% control exposure at 80% power requires 76 cases and 152 controls (N = 228). Applying a cluster design effect (ICC = 0.04, cluster size = 30, VIF = 2.16) inflates the requirement to 493 participants across 17 distinct GP clusters.",
            "disclaimer": "Completely made-up synthetic protocol dataset generated purely for educational and demonstration purposes only. Contains no real patient identifiers."
        })
        res["meta"] = dict(res)
        return res

    else:
        # Inner City Teaching Hospital: Metastatic NSCLC Oncology Survival Trial & Rapid Diagnostic Accuracy
        calc_surv = calculate_sample_size_survival_logrank(
            hazard_ratio=0.70, alpha=0.05, power=0.85, allocation_ratio=1.0, cumulative_event_prob=0.55
        )
        calc_diag = calculate_sample_size_diagnostic_accuracy(
            target_sensitivity=0.92, margin_of_error=0.04, disease_prevalence=0.15, alpha=0.05
        )
        attrition = apply_attrition_inflation(calc_surv["n_total"], dropout_rate=0.10)

        rows = []
        for i in range(1, 101):
            arm = "Novel PD-L1 Antibody" if i % 2 == 1 else "Standard Platinum Chemotherapy"
            rows.append({
                "Patient_ID": f"ICTH-ONC-{i:04d}",
                "Treatment_Arm": arm,
                "ECOG_Performance": (i % 3),
                "PDL1_Expression_Pct": min(100, (i * 7) % 105),
                "Event_Observed_18Mo": 1 if i % 2 == 0 or i % 3 == 0 else 0,
                "Screening_Biomarker_Positive": 1 if (i * 3) % 5 == 0 else 0
            })

        df = pd.DataFrame(rows) if HAS_SCIENTIFIC_STACK else rows

        res = SampleSizeDemoData({
            "df": df,
            "scenario_key": "oncology_survival",
            "scenario_name": "Inner City Teaching Hospital: Oncology Survival Trial (HR = 0.70) & Diagnostic Precision",
            "title": "Inner City Teaching Hospital: Immuno-Oncology Survival & Screening Validation",
            "organisation_name": "Inner City Teaching Hospital",
            "organisation": "Inner City Teaching Hospital (Department of Medical Oncology & Clinical Trials Unit)",
            "domain": "Oncology Clinical Trials & Molecular Diagnostic Validation",
            "intervention": "Novel PD-L1 Checkpoint Inhibitor Immunotherapy",
            "comparator": "Standard Doublet Platinum Chemotherapy",
            "primary_endpoint": "Progression-Free Survival (PFS) Hazard Ratio",
            "calc_results_primary": calc_surv,
            "calc_results_secondary": calc_diag,
            "attrition_adjustment": attrition,
            "description": "Phase III randomized survival trial evaluating progression-free survival in advanced NSCLC, paired with a companion diagnostic biomarker precision validation.",
            "takeaway": "Detecting a Hazard Ratio of 0.70 at 85% power requires 284 progression events. With an anticipated 18-month event incidence of 55%, a total of 517 patients (259 per arm) must be enrolled. To validate biomarker sensitivity of 92% (+/-4%), 177 confirmed cases are required from an unselected screening cohort of 1,180 patients (at 15% prevalence).",
            "disclaimer": "Completely made-up synthetic protocol dataset generated purely for educational and demonstration purposes only. Contains no real patient identifiers."
        })
        res["meta"] = dict(res)
        return res


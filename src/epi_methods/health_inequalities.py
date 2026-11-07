"""Mathematical & Algorithmic Engine for Health Inequalities: Slope Index of Inequality (SII)
and Relative Index of Inequality (RII).

Provides verified public health epidemiological calculations for:
1. Cumulative Relative Population Rank (Midpoint Ridit Scores):
   - Computes cumulative population share midpoints x_j in (0, 1) across ordered socioeconomic strata
     (e.g. IMD deciles / quintiles) accounting for unequal population sizes.
2. Slope Index of Inequality (SII - Absolute Inequality):
   - Linear Weighted Least Squares (WLS): y_j = alpha + beta * x_j, weighted by population share p_j.
   - SII = -beta (when ranked Most Deprived = 0 to Least Deprived = 1) representing the absolute
     gap in health indicator across the entire socioeconomic spectrum.
3. Relative Index of Inequality (RII - Relative Inequality):
   - Linear WLS: RII = y_hat(0) / y_hat(1) = alpha / (alpha + beta).
   - Log-Linear GLM: ln(y_j) = alpha + beta * x_j => RII = exp(-beta).
4. Variance & 95% Confidence Intervals:
   - WLS analytical standard error for SII = beta SE.
   - Delta Method variance approximation for RII ratio.
5. Prerequisite & diagnostic validation:
   - Monotonicity checks, population weight validity, non-negative rate bounds,
     minimum strata sufficiency (K >= 3).
6. Synthetic pedagogical example scenarios (made-up illustrative data):
   - Under-75 CVD Mortality by IMD Decile (synthetic illustrative gradient).
   - Male Life Expectancy at Birth across Deprivation Quintiles (synthetic Marmot gradient).
   - Childhood Obesity Prevalence by IMD Decile (synthetic educational gradient).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import math

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False
    np = None
    pd = None
    stats = None


def calculate_relative_ranks(populations: Union[List[float], Any]) -> Dict[str, Any]:
    """Calculates cumulative relative ranks (midpoint ridits) for ordered socioeconomic strata.

    Formula:
        p_j = n_j / N
        C_j = sum_{i=1}^j p_i (cumulative proportion, with C_0 = 0)
        x_j = C_{j-1} + 0.5 * p_j (stratum midpoint rank in [0, 1])

    Args:
        populations: Population counts or weights for ordered strata (e.g. Decile 1 to 10).

    Returns:
        Dict containing:
            - 'proportions': Group population shares p_j.
            - 'cumulative_proportions': Upper cumulative bounds C_j.
            - 'midpoint_ranks': Midpoint relative rank x_j in (0, 1).
            - 'total_population': Sum of all stratum populations.
    """
    pop = [float(p) for p in populations]
    if any(p <= 0 for p in pop):
        raise ValueError("Population counts must be strictly positive.")

    total_pop = sum(pop)
    proportions = [p / total_pop for p in pop]

    cum_proportions = []
    midpoint_ranks = []
    cum = 0.0
    for p in proportions:
        mid = cum + 0.5 * p
        cum += p
        midpoint_ranks.append(mid)
        cum_proportions.append(cum)

    return {
        "proportions": proportions,
        "cumulative_proportions": cum_proportions,
        "midpoint_ranks": midpoint_ranks,
        "total_population": total_pop
    }


def calculate_sii_rii(
    rates: Union[List[float], Any],
    populations: Union[List[float], Any],
    strata_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    indicator_name: str = "Rate per 100,000",
    reverse_direction: bool = False,
    model_type: str = "linear"
) -> Dict[str, Any]:
    """Calculates Slope Index of Inequality (SII) and Relative Index of Inequality (RII)

    via population-weighted regression over cumulative relative ranks (ridits).

    In UK Public Health (OHID / PHE Fingertips convention):
    - Stratum 1 is the MOST DEPRIVED (rank near 0).
    - Stratum K is the LEAST DEPRIVED (rank near 1).
    - If the health outcome is an adverse event (e.g. mortality):
        y_hat(0) = estimated rate for the most deprived individual.
        y_hat(1) = estimated rate for the least deprived individual.
        Absolute Gap (SII) = y_hat(0) - y_hat(1) = alpha - (alpha + beta) = -beta.
        A positive SII (+X) means the most deprived group suffers X excess events.
    - If reverse_direction=True, stratum 1 is least deprived and stratum K is most deprived.

    Args:
        rates: Observed or standardized health indicators across K strata.
        populations: Denominator populations for each stratum.
        strata_names: Human-readable labels (e.g. ['Decile 1 (Most)', ..., 'Decile 10 (Least)']).
        alpha: Significance level (default 0.05 for 95% CI).
        indicator_name: Unit/label for the health metric.
        reverse_direction: If True, treats group 1 as least deprived and group K as most deprived.
        model_type: 'linear' for Weighted Least Squares, 'log_linear' for log-transformed GLM.

    Returns:
        Dict containing full parameter estimates, CIs, diagnostics, and fitted values.
    """
    y = [float(r) for r in rates]
    pop = [float(p) for p in populations]
    k = len(y)

    if k < 3:
        raise ValueError(f"At least 3 socioeconomic strata are required for regression (got {k}).")
    if len(pop) != k:
        raise ValueError(f"Length mismatch: {k} rates vs {len(pop)} population counts.")
    if any(p <= 0 for p in pop):
        raise ValueError("Population counts must be strictly positive.")
    if any(r < 0 for r in y):
        raise ValueError("Health indicator rates cannot be negative.")

    if strata_names is None:
        strata_names = [f"Stratum {i+1}" for i in range(k)]

    # Compute relative ranks (midpoint ridits)
    rank_info = calculate_relative_ranks(pop)
    x = rank_info["midpoint_ranks"]
    weights = rank_info["proportions"]  # Sums to 1.0

    z_crit = 1.959963984540054
    if stats is not None and hasattr(stats, "norm"):
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    # -------------------------------------------------------------
    # Linear Weighted Least Squares (WLS)
    # -------------------------------------------------------------
    x_bar = sum(w * xi for w, xi in zip(weights, x))
    y_bar = sum(w * yi for w, yi in zip(weights, y))
    s_xx = sum(w * ((xi - x_bar) ** 2) for w, xi in zip(weights, x))
    s_xy = sum(w * (xi - x_bar) * (yi - y_bar) for w, xi, yi in zip(weights, x, y))

    beta = s_xy / s_xx if s_xx > 0 else 0.0
    alpha_hat = y_bar - beta * x_bar

    # Fitted values and residuals
    y_fitted = [alpha_hat + beta * xi for xi in x]
    residuals = [yi - yf for yi, yf in zip(y, y_fitted)]
    weighted_rss = sum(w * (res ** 2) for w, res in zip(weights, residuals))
    total_ss = sum(w * ((yi - y_bar) ** 2) for w, yi in zip(weights, y))
    r_squared = 1.0 - (weighted_rss / total_ss) if total_ss > 0 else 0.0

    # Residual variance and standard errors (df = k - 2)
    df_resid = max(1, k - 2)
    mse = weighted_rss / df_resid
    var_beta = mse / s_xx if s_xx > 0 else 0.0
    var_alpha = mse * (1.0 + (x_bar ** 2) / s_xx) if s_xx > 0 else 0.0
    cov_alpha_beta = -mse * x_bar / s_xx if s_xx > 0 else 0.0

    se_beta = math.sqrt(max(0.0, var_beta))
    se_alpha = math.sqrt(max(0.0, var_alpha))

    # Fitted values at the extreme hypothetical ranks:
    # Under UK / OHID convention, x=0.0 is the hypothetical MOST DEPRIVED extreme individual,
    # and x=1.0 is the hypothetical LEAST DEPRIVED extreme individual.
    y_most = alpha_hat
    y_least = alpha_hat + beta

    # Compute Linear SII and RII based on outcome indicator direction
    if not reverse_direction:
        # Adverse outcome (e.g. CVD mortality, obesity):
        # Most deprived has higher rate (beta < 0).
        # SII = y(0) - y(1) = -beta (excess burden in the most deprived).
        sii = y_most - y_least  # = -beta
        se_sii = se_beta
        sii_ci_lower = sii - z_crit * se_sii
        sii_ci_upper = sii + z_crit * se_sii

        # RII = y_most / y_least = alpha / (alpha + beta)
        if y_least > 0 and y_most >= 0:
            rii = y_most / y_least
            denom = (alpha_hat + beta) ** 2
            d_alpha = beta / denom
            d_beta = -alpha_hat / denom
            var_rii = (d_alpha ** 2) * var_alpha + (d_beta ** 2) * var_beta + 2.0 * d_alpha * d_beta * cov_alpha_beta
            se_rii = math.sqrt(max(0.0, var_rii))
            rii_ci_lower = max(0.0, rii - z_crit * se_rii)
            rii_ci_upper = rii + z_crit * se_rii
        else:
            rii = 1.0
            se_rii = 0.0
            rii_ci_lower, rii_ci_upper = 1.0, 1.0
    else:
        # Favourable outcome (e.g. Life Expectancy at Birth where higher is better):
        # Least deprived has higher rate (beta > 0).
        # SII = y_least - y_most = +beta (Marmot life expectancy gap in favour of affluent).
        sii = y_least - y_most  # = +beta
        se_sii = se_beta
        sii_ci_lower = sii - z_crit * se_sii
        sii_ci_upper = sii + z_crit * se_sii

        # RII = y_least / y_most = (alpha + beta) / alpha = 1 + beta / alpha
        if y_most > 0 and y_least >= 0:
            rii = y_least / y_most
            d_alpha = -beta / (alpha_hat ** 2)
            d_beta = 1.0 / alpha_hat
            var_rii = (d_alpha ** 2) * var_alpha + (d_beta ** 2) * var_beta + 2.0 * d_alpha * d_beta * cov_alpha_beta
            se_rii = math.sqrt(max(0.0, var_rii))
            rii_ci_lower = max(0.0, rii - z_crit * se_rii)
            rii_ci_upper = rii + z_crit * se_rii
        else:
            rii = 1.0
            se_rii = 0.0
            rii_ci_lower, rii_ci_upper = 1.0, 1.0

    linear_results = {
        "alpha_intercept": alpha_hat,
        "beta_slope": beta,
        "se_alpha": se_alpha,
        "se_beta": se_beta,
        "var_alpha": var_alpha,
        "var_beta": var_beta,
        "cov_alpha_beta": cov_alpha_beta,
        "y_most_deprived": y_most,
        "y_least_deprived": y_least,
        "sii": sii,
        "se_sii": se_sii,
        "sii_ci_lower": sii_ci_lower,
        "sii_ci_upper": sii_ci_upper,
        "rii": rii,
        "se_rii": se_rii,
        "rii_ci_lower": rii_ci_lower,
        "rii_ci_upper": rii_ci_upper,
        "r_squared": r_squared,
        "fitted_rates": y_fitted
    }

    # -------------------------------------------------------------
    # Log-Linear Fit (Log link for positive rates: ln(y) = gamma_0 + gamma_1 * x)
    # -------------------------------------------------------------
    log_model_available = all(yi > 0 for yi in y)
    log_results: Dict[str, Any] = {}
    if log_model_available:
        ln_y = [math.log(yi) for yi in y]
        ln_y_bar = sum(w * ly for w, ly in zip(weights, ln_y))
        s_x_lny = sum(w * (xi - x_bar) * (ly - ln_y_bar) for w, xi, ly in zip(weights, x, ln_y))
        gamma_1 = s_x_lny / s_xx if s_xx > 0 else 0.0
        gamma_0 = ln_y_bar - gamma_1 * x_bar

        ln_residuals = [ly - (gamma_0 + gamma_1 * xi) for ly, xi in zip(ln_y, x)]
        mse_log = sum(w * (res ** 2) for w, res in zip(weights, ln_residuals)) / df_resid
        var_gamma_1 = mse_log / s_xx if s_xx > 0 else 0.0
        var_gamma_0 = mse_log * (1.0 + (x_bar ** 2) / s_xx) if s_xx > 0 else 0.0
        cov_gamma_01 = -mse_log * x_bar / s_xx if s_xx > 0 else 0.0

        se_gamma_1 = math.sqrt(max(0.0, var_gamma_1))
        se_gamma_0 = math.sqrt(max(0.0, var_gamma_0))

        log_y_most = math.exp(gamma_0)
        log_y_least = math.exp(gamma_0 + gamma_1)
        fitted_rates_log = [math.exp(gamma_0 + gamma_1 * xi) for xi in x]

        # Log-model R-squared on original outcome scale
        weighted_rss_log = sum(w * ((yi - yf) ** 2) for w, yi, yf in zip(weights, y, fitted_rates_log))
        r_squared_log = 1.0 - (weighted_rss_log / total_ss) if total_ss > 0 else 0.0

        if not reverse_direction:
            # Adverse outcome:
            rii_log = math.exp(-gamma_1)
            rii_log_lower = math.exp(-gamma_1 - z_crit * se_gamma_1)
            rii_log_upper = math.exp(-gamma_1 + z_crit * se_gamma_1)
            sii_log = log_y_most - log_y_least

            # Delta method variance for SII_log = exp(gamma_0) - exp(gamma_0 + gamma_1):
            d0 = sii_log
            d1 = -log_y_least
            var_sii_log = (d0 ** 2) * var_gamma_0 + (d1 ** 2) * var_gamma_1 + 2.0 * d0 * d1 * cov_gamma_01
            se_sii_log = math.sqrt(max(0.0, var_sii_log))
            sii_log_lower = sii_log - z_crit * se_sii_log
            sii_log_upper = sii_log + z_crit * se_sii_log
        else:
            # Favourable outcome (Life Expectancy):
            rii_log = math.exp(gamma_1)
            rii_log_lower = math.exp(gamma_1 - z_crit * se_gamma_1)
            rii_log_upper = math.exp(gamma_1 + z_crit * se_gamma_1)
            sii_log = log_y_least - log_y_most

            # Delta method variance for SII_log = exp(gamma_0 + gamma_1) - exp(gamma_0):
            d0 = sii_log
            d1 = log_y_least
            var_sii_log = (d0 ** 2) * var_gamma_0 + (d1 ** 2) * var_gamma_1 + 2.0 * d0 * d1 * cov_gamma_01
            se_sii_log = math.sqrt(max(0.0, var_sii_log))
            sii_log_lower = sii_log - z_crit * se_sii_log
            sii_log_upper = sii_log + z_crit * se_sii_log

        log_results = {
            "gamma_0": gamma_0,
            "gamma_1": gamma_1,
            "se_gamma_0": se_gamma_0,
            "se_gamma_1": se_gamma_1,
            "var_gamma_0": var_gamma_0,
            "var_gamma_1": var_gamma_1,
            "cov_gamma_01": cov_gamma_01,
            "y_most_deprived": log_y_most,
            "y_least_deprived": log_y_least,
            "sii_log": sii_log,
            "se_sii_log": se_sii_log,
            "sii_log_ci_lower": sii_log_lower,
            "sii_log_ci_upper": sii_log_upper,
            "rii_log": rii_log,
            "se_gamma_1": se_gamma_1,
            "rii_log_ci_lower": rii_log_lower,
            "rii_log_ci_upper": rii_log_upper,
            "r_squared_log": r_squared_log,
            "fitted_rates_log": fitted_rates_log
        }

    # -------------------------------------------------------------
    # Active Model Selection: Promote selected model metrics to primary keys
    # -------------------------------------------------------------
    if model_type == "log_linear" and log_model_available:
        model_name = "Log-Linear GLM (Exponential link)"
        active_sii = log_results["sii_log"]
        active_se_sii = log_results["se_sii_log"]
        active_sii_lower = log_results["sii_log_ci_lower"]
        active_sii_upper = log_results["sii_log_ci_upper"]
        active_rii = log_results["rii_log"]
        active_se_rii = log_results["se_gamma_1"]
        active_rii_lower = log_results["rii_log_ci_lower"]
        active_rii_upper = log_results["rii_log_ci_upper"]
        active_alpha = log_results["gamma_0"]
        active_beta = log_results["gamma_1"]
        active_se_alpha = log_results["se_gamma_0"]
        active_se_beta = log_results["se_gamma_1"]
        active_var_alpha = log_results["var_gamma_0"]
        active_var_beta = log_results["var_gamma_1"]
        active_cov_ab = log_results["cov_gamma_01"]
        active_y_most = log_results["y_most_deprived"]
        active_y_least = log_results["y_least_deprived"]
        active_fitted = log_results["fitted_rates_log"]
        active_residuals = [yi - yf for yi, yf in zip(y, active_fitted)]
        active_r2 = log_results["r_squared_log"]
    else:
        model_name = "Linear WLS (Standard OHID / Fingertips)"
        active_sii = sii
        active_se_sii = se_sii
        active_sii_lower = sii_ci_lower
        active_sii_upper = sii_ci_upper
        active_rii = rii
        active_se_rii = se_rii
        active_rii_lower = rii_ci_lower
        active_rii_upper = rii_ci_upper
        active_alpha = alpha_hat
        active_beta = beta
        active_se_alpha = se_alpha
        active_se_beta = se_beta
        active_var_alpha = var_alpha
        active_var_beta = var_beta
        active_cov_ab = cov_alpha_beta
        active_y_most = y_most
        active_y_least = y_least
        active_fitted = y_fitted
        active_residuals = residuals
        active_r2 = r_squared

    # -------------------------------------------------------------
    # Statistical Significance Testing (H0: SII = 0, H0: RII = 1.0)
    # -------------------------------------------------------------
    z_sii = abs(active_sii) / active_se_sii if active_se_sii > 0 else 0.0
    p_value_sii = math.erfc(z_sii / math.sqrt(2.0))
    is_sii_significant = bool((active_sii_lower > 0 and active_sii_upper > 0) or (active_sii_lower < 0 and active_sii_upper < 0))
    p_value_sii_str = "< 0.001" if p_value_sii < 0.001 else f"= {p_value_sii:.4f}"
    sii_significance_label = "Statistically Significant (p < 0.05)" if is_sii_significant else "Not Statistically Significant (p ≥ 0.05)"

    if model_type == "log_linear" and log_model_available:
        z_rii = abs(active_beta) / active_se_beta if active_se_beta > 0 else 0.0
    else:
        z_rii = abs(active_rii - 1.0) / active_se_rii if active_se_rii > 0 else 0.0
    p_value_rii = math.erfc(z_rii / math.sqrt(2.0))
    is_rii_significant = bool(active_rii_lower > 1.0 or active_rii_upper < 1.0)
    p_value_rii_str = "< 0.001" if p_value_rii < 0.001 else f"= {p_value_rii:.4f}"
    rii_significance_label = "Statistically Significant (p < 0.05)" if is_rii_significant else "Not Statistically Significant (p ≥ 0.05)"

    # Summary table reflecting the active model's fitted rates and residuals
    table_rows = []
    for i in range(k):
        table_rows.append({
            "stratum": strata_names[i],
            "population": int(pop[i]),
            "population_share": round(weights[i], 4),
            "midpoint_rank": round(x[i], 4),
            "observed_rate": round(y[i], 2),
            "fitted_rate": round(active_fitted[i], 2),
            "residual": round(active_residuals[i], 2)
        })

    # Health gradient direction interpretation
    if not reverse_direction:
        if active_sii > 0:
            gradient_summary = f"Deprivation Gradient: The most deprived extreme suffers {active_sii:.1f} excess {indicator_name} compared to the least deprived (RII = {active_rii:.2f}x; {model_name}; {sii_significance_label})."
            is_gradient_pro_affluent = True
        elif active_sii < 0:
            gradient_summary = f"Reverse Gradient: The least deprived extreme experiences {-active_sii:.1f} higher {indicator_name} than the most deprived ({model_name}; {sii_significance_label})."
            is_gradient_pro_affluent = False
        else:
            gradient_summary = f"No Inequality: The health indicator is uniform across all socioeconomic strata ({model_name})."
            is_gradient_pro_affluent = False
    else:
        if active_sii > 0:
            gradient_summary = f"Social Gradient: The least deprived extreme enjoys {active_sii:.1f} additional {indicator_name} compared to the most deprived extreme (RII = {active_rii:.2f}x; {model_name}; {sii_significance_label})."
            is_gradient_pro_affluent = True
        else:
            gradient_summary = f"Reverse Gradient: Life expectancy favours the most deprived ({model_name}; {sii_significance_label})."
            is_gradient_pro_affluent = False

    return {
        "k_strata": k,
        "strata_names": strata_names,
        "indicator_name": indicator_name,
        "model_type": model_type,
        "model_name": model_name,
        "reverse_direction": reverse_direction,
        "x_ranks": x,
        "observed_rates": y,
        "fitted_rates": active_fitted,
        "populations": pop,
        "population_shares": weights,
        "alpha_intercept": active_alpha,
        "beta_slope": active_beta,
        "se_alpha": active_se_alpha,
        "se_beta": active_se_beta,
        "var_alpha": active_var_alpha,
        "var_beta": active_var_beta,
        "cov_alpha_beta": active_cov_ab,
        "r_squared": active_r2,
        "y_most_deprived": active_y_most,
        "y_least_deprived": active_y_least,
        "sii": active_sii,
        "se_sii": active_se_sii,
        "sii_ci_lower": active_sii_lower,
        "sii_ci_upper": active_sii_upper,
        "z_score_sii": z_sii,
        "p_value_sii": p_value_sii,
        "p_value_sii_str": p_value_sii_str,
        "is_sii_significant": is_sii_significant,
        "sii_significance_label": sii_significance_label,
        "rii": active_rii,
        "se_rii": active_se_rii,
        "rii_ci_lower": active_rii_lower,
        "rii_ci_upper": active_rii_upper,
        "z_score_rii": z_rii,
        "p_value_rii": p_value_rii,
        "p_value_rii_str": p_value_rii_str,
        "is_rii_significant": is_rii_significant,
        "rii_significance_label": rii_significance_label,
        "gradient_summary": gradient_summary,
        "is_gradient_pro_affluent": is_gradient_pro_affluent,
        "linear_model": linear_results,
        "log_model": log_results,
        "table": table_rows
    }


def evaluate_health_inequalities_prerequisites(
    rates: Union[List[float], Any],
    populations: Union[List[float], Any]
) -> Dict[str, Any]:
    """Validates epidemiological prerequisites for Slope and Relative Indices of Inequality.

    Assesses:
    1. Strata count sufficiency (K >= 3 recommended, K >= 5 ideal).
    2. Population denominators strictly positive.
    3. Non-negative rates / health indicators.
    4. Monotonicity inspection.
    5. Extremes gap vs model slope consistency.

    Returns:
        Dict of diagnostic checks with status ('PASS', 'WARNING', 'FAIL') and explanation.
    """
    y = [float(r) for r in rates]
    pop = [float(p) for p in populations]
    k = len(y)

    checks = []

    # 1. Strata Count
    if k >= 5:
        checks.append({
            "test": "Socioeconomic Strata Count (K)",
            "status": "PASS",
            "message": f"Found {k} ordered strata (e.g. quintiles or deciles). Sufficient statistical degrees of freedom for regression."
        })
    elif k >= 3:
        checks.append({
            "test": "Socioeconomic Strata Count (K)",
            "status": "WARNING",
            "message": f"Found {k} strata. Minimum viable for WLS slope estimation, but deciles (K=10) or quintiles (K=5) provide greater statistical reliability."
        })
    else:
        checks.append({
            "test": "Socioeconomic Strata Count (K)",
            "status": "FAIL",
            "message": f"Only {k} strata provided. Regression requires at least 3 ordered socioeconomic tiers."
        })

    # 2. Positive Denominators
    if all(p > 0 for p in pop):
        checks.append({
            "test": "Population Denominators",
            "status": "PASS",
            "message": f"All {k} strata have positive population sizes (total: {sum(pop):,.0f}). Valid population weights."
        })
    else:
        checks.append({
            "test": "Population Denominators",
            "status": "FAIL",
            "message": "One or more strata contain zero or negative population counts. Cannot compute relative ranks."
        })

    # 3. Non-Negative Rates
    if all(yi >= 0 for yi in y):
        checks.append({
            "test": "Health Indicator Non-Negativity",
            "status": "PASS",
            "message": f"All {k} rates are non-negative (range: {min(y):.1f} to {max(y):.1f})."
        })
    else:
        checks.append({
            "test": "Health Indicator Non-Negativity",
            "status": "FAIL",
            "message": "Health indicator contains negative values. Rates and prevalences must be non-negative."
        })

    # 4. Monotonicity Assessment
    if k >= 3:
        concordant = 0
        discordant = 0
        for i in range(k):
            for j in range(i + 1, k):
                if y[j] > y[i]:
                    concordant += 1
                elif y[j] < y[i]:
                    discordant += 1
        total_pairs = k * (k - 1) / 2
        tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0
        abs_tau = abs(tau)

        if abs_tau >= 0.70:
            checks.append({
                "test": "Social Gradient Monotonicity",
                "status": "PASS",
                "message": f"Strong monotonic gradient observed (Concordance / tau = {tau:.2f}). Linear regression slope is highly representative."
            })
        elif abs_tau >= 0.40:
            checks.append({
                "test": "Social Gradient Monotonicity",
                "status": "WARNING",
                "message": f"Moderate monotonicity (tau = {tau:.2f}). Some non-linear fluctuations across intermediate strata."
            })
        else:
            checks.append({
                "test": "Social Gradient Monotonicity",
                "status": "WARNING",
                "message": f"Weak or non-monotonic gradient (tau = {tau:.2f}). Inspect whether health outcome follows a U-shaped or non-linear pattern."
            })

    # Determine overall status
    statuses = [c["status"] for c in checks]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARNING" in statuses:
        overall = "WARNING"
    else:
        overall = "PASS"

    return {
        "overall_status": overall,
        "checks": checks
    }


def generate_inequality_demo_data(scenario_key: str = "derbyshire_cvd") -> Dict[str, Any]:
    """Generates synthetic illustrative example datasets across ranked strata for educational demonstration.

    (Pedagogical Notice: All rates, populations, and scenarios are fictional, made-up examples
    created solely for training in health inequality calculations and interpretations).

    Available Scenarios:
    1. 'derbyshire_cvd': Under-75 Cardiovascular Disease (CVD) Mortality across 10 IMD Deciles
       (made-up synthetic profile styled after local authority surveillance profiles, DSR per 100,000).
    2. 'life_expectancy_gap': Male Life Expectancy at Birth across 5 Income Deprivation Quintiles
       (made-up synthetic example illustrating Marmot Review social gradient concepts).
    3. 'childhood_obesity': Year 6 Childhood Obesity Prevalence (%) across 10 IMD Deciles
       (made-up synthetic example illustrating relative inequality).
    """
    scenarios: Dict[str, Dict[str, Any]] = {
        "derbyshire_cvd": {
            "title": "Synthetic Example: Under-75 CVD Mortality by IMD Decile",
            "indicator_name": "Direct Age-Standardized Rate (DSR per 100,000)",
            "description": "Fictional educational example demonstrating how to measure premature cardiovascular mortality gradients across 10 Index of Multiple Deprivation (IMD) deciles, from Decile 1 (Most Deprived) to Decile 10 (Least Deprived).",
            "disclaimer": "⚠️ Educational Disclaimer: All rates and populations in this scenario are entirely synthetic, made-up example data for teaching and statistical demonstration. They do not represent official published statistics for Derbyshire, Derby City, or any specific health authority.",
            "strata_names": [
                "Decile 1 (Most Deprived)",
                "Decile 2",
                "Decile 3",
                "Decile 4",
                "Decile 5",
                "Decile 6",
                "Decile 7",
                "Decile 8",
                "Decile 9",
                "Decile 10 (Least Deprived)"
            ],
            "rates": [142.4, 126.8, 114.2, 102.5, 91.8, 84.1, 75.6, 68.3, 61.2, 53.4],
            "populations": [78500, 81200, 79400, 82100, 80900, 83500, 81000, 79800, 80400, 83200],
            "reverse_direction": False
        },
        "life_expectancy_gap": {
            "title": "Synthetic Example: Male Life Expectancy at Birth across Deprivation Quintiles",
            "indicator_name": "Life Expectancy at Birth (Years)",
            "description": "Fictional educational example demonstrating the social gradient in male life expectancy across 5 income deprivation quintiles, modelled after Marmot Review gradient concepts.",
            "disclaimer": "⚠️ Educational Disclaimer: All life expectancy estimates and populations in this scenario are fictional, made-up example data created for pedagogical demonstration. They do not represent official published census or ONS figures.",
            "strata_names": [
                "Quintile 1 (Most Deprived 20%)",
                "Quintile 2",
                "Quintile 3",
                "Quintile 4",
                "Quintile 5 (Least Deprived 20%)"
            ],
            "rates": [73.5, 76.2, 78.4, 80.9, 83.1],
            "populations": [160000, 162000, 159000, 161000, 158000],
            "reverse_direction": True
        },
        "childhood_obesity": {
            "title": "Synthetic Example: Year 6 Childhood Obesity Prevalence across Deprivation Deciles",
            "indicator_name": "Obesity Prevalence (%)",
            "description": "Fictional educational example demonstrating how to evaluate relative and absolute inequality in Year 6 childhood obesity prevalence across 10 deprivation deciles, styled after NCMP concepts.",
            "disclaimer": "⚠️ Educational Disclaimer: All obesity prevalence percentages and cohort sizes in this scenario are entirely made-up example data created for training and illustration purposes.",
            "strata_names": [
                "Decile 1 (Most Deprived)",
                "Decile 2",
                "Decile 3",
                "Decile 4",
                "Decile 5",
                "Decile 6",
                "Decile 7",
                "Decile 8",
                "Decile 9",
                "Decile 10 (Least Deprived)"
            ],
            "rates": [29.2, 26.8, 24.5, 22.1, 19.8, 18.2, 16.5, 15.1, 13.9, 12.4],
            "populations": [11200, 11500, 11100, 11400, 11300, 11600, 11200, 11500, 11400, 11800],
            "reverse_direction": False
        }
    }

    return scenarios.get(scenario_key, scenarios["derbyshire_cvd"])

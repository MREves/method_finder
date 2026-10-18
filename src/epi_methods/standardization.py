"""Mathematical & Statistical Engine for Demographic Rate Standardization.

Provides verified epidemiological calculations for:
1. Direct Age-Standardization (DSR / ASR) using European Standard Population (ESP 2013).
2. Dobson & Normal 95% Confidence Intervals for DSR.
3. Indirect Standardization & Standardized Mortality Ratios (SMR).
4. Byar's Exact Poisson 95% Confidence Intervals for SMR.
5. Assumption & prerequisite diagnostic evaluation engine (sample stability, sparse cells, rate ratios).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def evaluate_standardization_prerequisites(
    pop_age: Union[List[float], np.ndarray, pd.Series],
    deaths_age: Union[List[float], np.ndarray, pd.Series],
    reference_rates_per_k: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    min_cell_deaths: int = 5,
    min_total_deaths: int = 25,
) -> List[Dict[str, Any]]:
    """Evaluates epidemiological prerequisites and assumptions for rate standardization.

    Assesses whether the data satisfies the criteria for Direct Standardization (DSR),
    requires Indirect Standardization (SMR), or warrants Poisson multivariable regression.

    Args:
        pop_age (Union[List[float], np.ndarray, pd.Series]): Local population counts 
            across age strata (denominators $n_i$). Must be non-negative integers or floats.
        deaths_age (Union[List[float], np.ndarray, pd.Series]): Observed disease events or 
            deaths across age strata (numerators $d_i$). Must be non-negative integers or floats.
        reference_rates_per_k (Optional[Union[List[float], np.ndarray, pd.Series]], optional): 
            Standard benchmark reference rates per 1,000 ($R_{std, i}$). Defaults to None.
        min_cell_deaths (int, optional): Minimum deaths per individual age stratum recommended 
            for stable stratum-specific rates. Defaults to 5.
        min_total_deaths (int, optional): Minimum aggregated deaths across all strata for 
            reliable DSR variance estimation. Defaults to 25.

    Returns:
        List[Dict[str, Any]]: Diagnostic evaluation cards for each assumption, containing:
            - 'name': Name of the assumption / prerequisite check.
            - 'status': Verdict status ('PASS', 'WARNING', or 'FAIL').
            - 'details': Plain-English interpretation of diagnostic values.
            - 'recommendation': Specific epidemiological guidance or alternative method.
    """
    pop = np.asarray(pop_age, dtype=float)
    deaths = np.asarray(deaths_age, dtype=float)

    evaluations: List[Dict[str, Any]] = []

    # 1. Denominator Non-Negativity & Positivity Check
    zero_or_neg_pop = np.sum(pop <= 0)
    if zero_or_neg_pop == 0:
        evaluations.append({
            "name": "Population Denominator Stability",
            "status": "PASS",
            "details": f"All {len(pop)} age strata have positive, non-zero populations (Total Population: {int(np.sum(pop)):,}).",
            "recommendation": "Denominators are valid for age-specific rate calculation."
        })
    else:
        evaluations.append({
            "name": "Population Denominator Stability",
            "status": "FAIL",
            "details": f"{zero_or_neg_pop} age stratum has zero or negative population. Rates cannot be computed for empty strata.",
            "recommendation": "Collapse sparse age strata or verify demographic data inputs."
        })

    # 2. Total Event Numerator Volume Check
    total_deaths = float(np.sum(deaths))
    if total_deaths >= 100:
        evaluations.append({
            "name": "Total Event Count Volume",
            "status": "PASS",
            "details": f"Observed total deaths = {int(total_deaths)}. Ample statistical power (>= 100 events) for Dobson/Normal DSR.",
            "recommendation": "Direct Standardization (DSR) is fully robust."
        })
    elif total_deaths >= min_total_deaths:
        evaluations.append({
            "name": "Total Event Count Volume",
            "status": "WARNING",
            "details": f"Observed total deaths = {int(total_deaths)} (between {min_total_deaths} and 99). Small event count.",
            "recommendation": "Use Dobson's exact method for DSR confidence intervals, or consider Indirect SMR for greater precision."
        })
    else:
        evaluations.append({
            "name": "Total Event Count Volume",
            "status": "FAIL",
            "details": f"Observed total deaths = {int(total_deaths)} (< {min_total_deaths}). Extreme Poisson random fluctuation.",
            "recommendation": "Direct DSR is unstable. Switch to Indirect Standardization (SMR) or aggregate time periods (e.g. 3-year rolling average)."
        })

    # 3. Stratum-Specific Sparsity (Zero Cells & Small Counts)
    zero_cells = int(np.sum(deaths == 0))
    sparse_cells = int(np.sum((deaths > 0) & (deaths < min_cell_deaths)))
    if zero_cells == 0 and sparse_cells == 0:
        evaluations.append({
            "name": "Age-Stratum Cell Sufficiency",
            "status": "PASS",
            "details": f"All {len(deaths)} age strata contain >= {min_cell_deaths} observed events. No zero or sparse cells detected.",
            "recommendation": "Age-specific rates are stable across all brackets."
        })
    elif zero_cells == 0:
        evaluations.append({
            "name": "Age-Stratum Cell Sufficiency",
            "status": "WARNING",
            "details": f"{sparse_cells} age stratum has fewer than {min_cell_deaths} deaths. Rates in these strata have wider variance.",
            "recommendation": "Direct DSR remains usable, but stratum-specific rates should be interpreted with caution."
        })
    else:
        evaluations.append({
            "name": "Age-Stratum Cell Sufficiency",
            "status": "WARNING",
            "details": f"{zero_cells} age stratum has 0 observed deaths, and {sparse_cells} has < {min_cell_deaths} deaths.",
            "recommendation": "Zero cells produce an empirical rate of 0.0 per 1,000. Indirect Standardization (SMR) is strongly recommended."
        })

    # 4. Homogeneity of Rate Ratios (Proportionality Check)
    if reference_rates_per_k is not None:
        ref = np.asarray(reference_rates_per_k, dtype=float)
        valid_mask = (pop > 0) & (ref > 0)
        if np.sum(valid_mask) > 1:
            local_rates = np.where(pop > 0, (deaths / pop) * 1000.0, 0.0)
            rate_ratios = local_rates[valid_mask] / ref[valid_mask]
            rr_non_zero = rate_ratios[rate_ratios > 0]
            if len(rr_non_zero) > 1:
                ratio_dispersion = float(np.max(rr_non_zero) / np.min(rr_non_zero))
                if ratio_dispersion <= 3.0:
                    evaluations.append({
                        "name": "Homogeneity of Stratum Rate Ratios",
                        "status": "PASS",
                        "details": f"Max-to-min stratum rate ratio dispersion = {ratio_dispersion:.2f}x (<= 3.0x). Rate differences are reasonably uniform.",
                        "recommendation": "Summary standardization metrics (DSR and SMR) are epidemiologically valid."
                    })
                else:
                    evaluations.append({
                        "name": "Homogeneity of Stratum Rate Ratios",
                        "status": "WARNING",
                        "details": f"Substantial heterogeneity across age strata (ratio dispersion = {ratio_dispersion:.2f}x > 3.0x). Relative risk varies by age bracket.",
                        "recommendation": "Inspect stratum-specific rate ratios individually. A single summary standardized rate may obscure age-specific effects."
                    })

    return evaluations


def calculate_byar_poisson_ci(
    observed_deaths: float,
    expected_deaths: float,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Calculates Byar's exact Poisson approximation 95% Confidence Interval for SMR.

    Byar's formula provides an extremely accurate approximation to the exact Poisson 
    interval (Rothman & Boice, 1979) and is the standard method prescribed by the UK 
    Office for Health Improvement & Disparities (OHID / PHE).

    Args:
        observed_deaths (float): Total observed deaths ($O$). Must be >= 0.
        expected_deaths (float): Total expected deaths ($E$). Must be > 0.
        confidence_level (float, optional): Two-sided confidence level. Defaults to 0.95.

    Returns:
        Tuple[float, float]: Tuple of (ci_lower, ci_upper) for the Standardized Mortality Ratio.
    """
    if expected_deaths <= 0:
        return 0.0, 0.0

    obs = float(observed_deaths)
    exp = float(expected_deaths)
    z = 1.959963984540054  # 95% two-sided normal quantile

    if obs == 0:
        # Exact Poisson upper limit for zero observed events: -ln(alpha/2) ~ 3.688879
        alpha = 1.0 - confidence_level
        upper_obs = -np.log(alpha / 2.0)
        return 0.0, float(upper_obs / exp)

    # Byar's formula for Poisson limits:
    obs_lower = obs * ((1.0 - (1.0 / (9.0 * obs)) - (z / (3.0 * np.sqrt(obs)))) ** 3)
    obs_upper = (obs + 1.0) * ((1.0 - (1.0 / (9.0 * (obs + 1.0))) + (z / (3.0 * np.sqrt(obs + 1.0)))) ** 3)

    ci_lower = max(0.0, float(obs_lower / exp))
    ci_upper = float(obs_upper / exp)

    return ci_lower, ci_upper


def calculate_dobson_dsr_ci(
    pop_age: Union[List[float], np.ndarray, pd.Series],
    deaths_age: Union[List[float], np.ndarray, pd.Series],
    esp_weights: Union[List[float], np.ndarray, pd.Series],
    multiplier: float = 1000.0,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Calculates Direct Age-Standardized Rate (DSR) and Dobson 95% Confidence Intervals.

    Implements Dobson's method (Dobson et al., 1991), which incorporates exact Poisson 
    confidence limits for total event counts when deaths are small (< 100), smoothly 
    converging to normal variance limits for large counts.

    Args:
        pop_age (Union[List[float], np.ndarray, pd.Series]): Local stratum population 
            counts ($n_i$).
        deaths_age (Union[List[float], np.ndarray, pd.Series]): Local stratum observed 
            deaths ($d_i$).
        esp_weights (Union[List[float], np.ndarray, pd.Series]): Standard population 
            reference weights ($w_i$). Automatically normalized so sum equals 1.0.
        multiplier (float, optional): Scaling factor for output rate (e.g., 1,000 for per 
            1,000 population, or 100,000 for per 100,000). Defaults to 1000.0.
        confidence_level (float, optional): Two-sided confidence level. Defaults to 0.95.

    Returns:
        Dict[str, float]: Dictionary containing:
            - 'dsr': Point estimate of Direct Standardized Rate.
            - 'var_dsr': Variance of the standardized rate.
            - 'se_dsr': Standard error of the standardized rate.
            - 'ci_lower': Lower bound of the 95% confidence interval.
            - 'ci_upper': Upper bound of the 95% confidence interval.
            - 'total_deaths': Total observed deaths across all strata.
            - 'total_population': Total local population denominator.
            - 'crude_rate': Unadjusted crude rate per multiplier.
    """
    pop = np.asarray(pop_age, dtype=float)
    deaths = np.asarray(deaths_age, dtype=float)
    raw_weights = np.asarray(esp_weights, dtype=float)
    weights = raw_weights / np.sum(raw_weights)

    # 1. Stratum-specific empirical rates per multiplier
    rates = np.where(pop > 0, (deaths / pop) * multiplier, 0.0)

    # 2. DSR weighted average
    dsr = float(np.sum(weights * rates))

    # 3. Variance under Poisson distribution: Var(DSR) = M^2 * sum( w_i^2 * d_i / n_i^2 )
    var_terms = np.where(pop > 0, (weights ** 2) * (deaths / (pop ** 2)) * (multiplier ** 2), 0.0)
    var_dsr = float(np.sum(var_terms))
    se_dsr = float(np.sqrt(max(0.0, var_dsr)))

    tot_deaths = float(np.sum(deaths))
    tot_pop = float(np.sum(pop))
    crude_rate = (tot_deaths / tot_pop) * multiplier if tot_pop > 0 else 0.0

    z = 1.959963984540054

    # 4. Confidence Interval: Dobson method for small events (< 100) or Normal approximation
    if tot_deaths >= 100:
        ci_lower = max(0.0, dsr - z * se_dsr)
        ci_upper = dsr + z * se_dsr
    elif tot_deaths > 0:
        # Dobson's method scales exact Poisson limits for total deaths by Var(DSR) / D
        d_lower, d_upper = calculate_byar_poisson_ci(tot_deaths, 1.0, confidence_level=confidence_level)
        scale = np.sqrt(var_dsr / tot_deaths) if tot_deaths > 0 else 0.0
        ci_lower = max(0.0, dsr + scale * (d_lower - tot_deaths))
        ci_upper = dsr + scale * (d_upper - tot_deaths)
    else:
        ci_lower = 0.0
        ci_upper = 0.0

    return {
        "dsr": dsr,
        "var_dsr": var_dsr,
        "se_dsr": se_dsr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "total_deaths": tot_deaths,
        "total_population": tot_pop,
        "crude_rate": crude_rate
    }


def calculate_direct_standardized_rate(
    pop_age: Union[List[float], np.ndarray, pd.Series],
    deaths_age: Union[List[float], np.ndarray, pd.Series],
    esp_weights: Union[List[float], np.ndarray, pd.Series],
    multiplier: float = 1000.0,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Calculates Direct Age-Standardized Rate (DSR) and complete stratum breakdown.

    Args:
        pop_age (Union[List[float], np.ndarray, pd.Series]): 1D array of local population 
            counts in each age bracket (stratum denominator $n_i$).
        deaths_age (Union[List[float], np.ndarray, pd.Series]): 1D array of observed health 
            events or deaths in each age bracket (stratum numerator $d_i$).
        esp_weights (Union[List[float], np.ndarray, pd.Series]): 1D array of reference 
            standard population weights (e.g. European Standard Population ESP 2013 weights $w_i$).
        multiplier (float, optional): Scaling factor for the rate (e.g., 1000.0 for rate per 
            1,000 population). Defaults to 1000.0.
        confidence_level (float, optional): Two-sided confidence level. Defaults to 0.95.

    Returns:
        Dict[str, Any]: Complete analysis results containing:
            - 'dsr': Point estimate of Direct Standardized Rate.
            - 'se_dsr': Standard error of the standardized rate.
            - 'ci_lower': Lower bound of the 95% confidence interval.
            - 'ci_upper': Upper bound of the 95% confidence interval.
            - 'crude_rate': Unadjusted crude rate for reference.
            - 'total_deaths': Aggregated deaths across all age strata.
            - 'total_population': Total population denominator.
            - 'age_specific_rates': Array of empirical age-specific rates per multiplier.
            - 'stratum_contributions': Weighted rate contribution from each age stratum.
    """
    pop = np.asarray(pop_age, dtype=float)
    deaths = np.asarray(deaths_age, dtype=float)
    weights = np.asarray(esp_weights, dtype=float) / np.sum(esp_weights)

    dobson_res = calculate_dobson_dsr_ci(
        pop_age=pop,
        deaths_age=deaths,
        esp_weights=weights,
        multiplier=multiplier,
        confidence_level=confidence_level
    )

    rates = np.where(pop > 0, (deaths / pop) * multiplier, 0.0)
    stratum_contrib = weights * rates

    return {
        "dsr": dobson_res["dsr"],
        "se_dsr": dobson_res["se_dsr"],
        "ci_lower": dobson_res["ci_lower"],
        "ci_upper": dobson_res["ci_upper"],
        "crude_rate": dobson_res["crude_rate"],
        "total_deaths": dobson_res["total_deaths"],
        "total_population": dobson_res["total_population"],
        "age_specific_rates": rates,
        "stratum_contributions": stratum_contrib
    }


def calculate_indirect_standardization(
    pop_age: Union[List[float], np.ndarray, pd.Series],
    observed_deaths_total: float,
    reference_rates_per_k: Union[List[float], np.ndarray, pd.Series],
    multiplier: float = 1000.0,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Calculates Indirect Standardization, Expected Deaths, SMR, and Byar's 95% Poisson CIs.

    Args:
        pop_age (Union[List[float], np.ndarray, pd.Series]): 1D array of local population 
            counts in each age bracket (stratum denominator $n_i$).
        observed_deaths_total (float): Total number of observed health events or deaths 
            aggregated across all age groups ($O = \sum d_i$).
        reference_rates_per_k (Union[List[float], np.ndarray, pd.Series]): 1D array of benchmark 
            reference age-specific rates per multiplier ($R_{std, i}$).
        multiplier (float, optional): Scaling denominator for reference rates (e.g., 1000.0). 
            Defaults to 1000.0.
        confidence_level (float, optional): Two-sided confidence level. Defaults to 0.95.

    Returns:
        Dict[str, Any]: Indirect standardization results containing:
            - 'expected_deaths': Total expected deaths ($E = \sum n_i \times R_{std, i} / \text{multiplier}$).
            - 'observed_deaths': Total observed deaths ($O$).
            - 'smr': Standardized Mortality Ratio ($O / E$, where 1.0 represents parity with benchmark).
            - 'smr_percentage': SMR expressed as a percentage ($100 \times O / E$).
            - 'smr_ci_lower': Lower bound of Byar's exact Poisson 95% CI for SMR.
            - 'smr_ci_upper': Upper bound of Byar's exact Poisson 95% CI for SMR.
            - 'smr_pct_ci_lower': Lower bound for SMR percentage ($100 \times \text{CI Lower}$).
            - 'smr_pct_ci_upper': Upper bound for SMR percentage ($100 \times \text{CI Upper}$).
            - 'stratum_expected_deaths': Expected deaths within each individual age stratum.
    """
    pop = np.asarray(pop_age, dtype=float)
    ref_rates = np.asarray(reference_rates_per_k, dtype=float)

    # 1. Stratum-specific expected deaths: E_i = n_i * R_std,i / multiplier
    stratum_exp = (pop * ref_rates) / multiplier
    tot_exp = float(np.sum(stratum_exp))
    obs = float(observed_deaths_total)

    # 2. SMR point estimate: O / E
    smr = float(obs / tot_exp) if tot_exp > 0 else 1.0
    smr_pct = smr * 100.0

    # 3. Byar's 95% Poisson confidence interval
    ci_lower, ci_upper = calculate_byar_poisson_ci(
        observed_deaths=obs,
        expected_deaths=tot_exp,
        confidence_level=confidence_level
    )

    return {
        "expected_deaths": tot_exp,
        "observed_deaths": obs,
        "smr": smr,
        "smr_percentage": smr_pct,
        "smr_ci_lower": ci_lower,
        "smr_ci_upper": ci_upper,
        "smr_pct_ci_lower": ci_lower * 100.0,
        "smr_pct_ci_upper": ci_upper * 100.0,
        "stratum_expected_deaths": stratum_exp
    }


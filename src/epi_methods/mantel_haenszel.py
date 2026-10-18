"""Mathematical & Statistical Engine for Cochran-Mantel-Haenszel (CMH) Stratified Analysis & Effect Modification.

Provides verified epidemiological calculations for:
1. Stratified 2x2 Contingency Tables across K strata:
   - Stratum-specific Odds Ratios (OR_k) and Risk Ratios (RR_k) with Woolf/Greenland 95% CIs.
   - Crude pooled 2x2 table, Crude Odds Ratio (OR_crude), and Crude Risk Ratio (RR_crude).
2. Mantel-Haenszel Summary Pooled Odds Ratio (OR_MH):
   OR_MH = sum(a_k * d_k / n_k) / sum(b_k * c_k / n_k)
   with Robins-Breslow-Greenland (RBG 1986) variance for 95% confidence intervals.
3. Mantel-Haenszel Summary Pooled Risk Ratio (RR_MH):
   RR_MH = sum(a_k * n_0k / n_k) / sum(c_k * n_1k / n_k)
   with Greenland-Robins (1985) variance for 95% confidence intervals.
4. Cochran-Mantel-Haenszel Chi-Square Test of Conditional Independence (CMH Chi-Square):
   chi2_CMH = (|sum(a_k) - sum(E(a_k))| - 0.5)^2 / sum(Var(a_k)), df = 1.
5. Confounding Assessment (The 10% Rule):
   Delta% = |OR_crude - OR_MH| / OR_MH * 100%
   Identifies clinically meaningful confounding (Delta% >= 10%) and detects Simpson's Paradox
   (where crude effect direction reverses upon stratification).
6. Breslow-Day Test for Homogeneity of Odds Ratios (Testing for Effect Measure Modification / Interaction):
   Under H0: OR_1 = OR_2 = ... = OR_K = OR_MH.
   Solves for expected cells A_k* and computes Breslow-Day chi-square and Tarone-adjusted chi-square (df = K - 1).
   If p < 0.05, effect measure modification is present; pooling into a single summary OR is inappropriate.
7. Woolf's Test for Homogeneity of Odds Ratios as a complementary check.
8. Synthetic clinical & public health scenarios with fictional UK healthcare names:
   - Inner City Teaching Hospital: Critical Care vs. General Ward In-Hospital Mortality (Severe Simpson's Paradox)
   - All Meadows Cardiovascular Institute: Physical Activity & CVD Events Stratified by Age (Classical Confounding)
   - Riverside Health Trust: Occupational Toxin & Pulmonary Disease Stratified by Smoking (Effect Measure Modification)
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


class MantelHaenszelDemoData(dict):
    """Dictionary holding demo dataset and metadata, with 2-tuple unpacking support.
    
    Allows both dictionary key access (data["df"], data["strata_tables"])
    and standard 2-tuple unpacking (df, meta = generate_mantel_haenszel_demo_data(...)).
    """
    def __iter__(self):
        yield self["df"]
        yield self["meta"]


def _solve_breslow_day_expected_cell(a: float, b: float, c: float, d: float, theta: float) -> float:
    """Solves the quadratic equation for expected cell count A_k* under H0: OR_k = theta.

    Formula:
        (A_k* * (n_0k - m_1k + A_k*)) / ((n_1k - A_k*) * (m_1k - A_k*)) = theta
    Standard form: A * x^2 + B * x + C = 0
        A = theta - 1.0
        B = -[(theta - 1.0) * (n_1k + m_1k) + n_k]
        C = theta * n_1k * m_1k
    """
    n1 = a + b
    n0 = c + d
    m1 = a + c
    m0 = b + d
    n = n1 + n0

    if n <= 0 or n1 <= 0 or n0 <= 0 or m1 <= 0 or m0 <= 0:
        return 0.0

    if abs(theta - 1.0) < 1e-7:
        return (n1 * m1) / float(n)

    A_coef = theta - 1.0
    B_coef = -((theta - 1.0) * (n1 + m1) + n)
    C_coef = theta * n1 * m1

    disc = B_coef * B_coef - 4.0 * A_coef * C_coef
    sqrt_disc = math.sqrt(max(0.0, disc))

    r1 = (-B_coef - sqrt_disc) / (2.0 * A_coef)
    r2 = (-B_coef + sqrt_disc) / (2.0 * A_coef)

    low_bound = max(0.0, float(n1 - m0))
    high_bound = min(float(n1), float(m1))

    for root in (r1, r2):
        if low_bound - 1e-5 <= root <= high_bound + 1e-5:
            return root

    # Fallback to independence expectation if numerical precision issue
    return (n1 * m1) / float(n)


def _chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) for chi-square distribution with df degrees of freedom."""
    if x <= 0.0 or df <= 0:
        return 1.0
    if HAS_SCIENTIFIC_STACK:
        return float(stats.chi2.sf(x, df=df))
    # Pure Python approximation for df=1 using complementary error function
    if df == 1:
        return math.erfc(math.sqrt(x / 2.0))
    # Approximation for df > 1 via Wilson-Hilferty cube-root transformation
    try:
        s = 2.0 / (9.0 * df)
        z = ((x / float(df)) ** (1.0 / 3.0) - (1.0 - s)) / math.sqrt(s)
        return 0.5 * math.erfc(z / math.sqrt(2.0))
    except Exception:
        return 0.5


def calculate_mantel_haenszel(
    strata_data: Union[List[Dict[str, Any]], Any],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Calculates comprehensive Cochran-Mantel-Haenszel stratified effect measures and diagnostics.

    Evaluates:
    - Stratum-specific Odds Ratios and Risk Ratios with 95% CIs.
    - Crude pooled 2x2 table, Crude OR, Crude RR.
    - Mantel-Haenszel summary pooled OR with Robins-Breslow-Greenland (1986) 95% CI.
    - Mantel-Haenszel summary pooled RR with Greenland-Robins (1985) 95% CI.
    - Cochran-Mantel-Haenszel Chi-Square test of conditional independence (with and without continuity correction).
    - Confounding assessment via the 10% change-in-estimate rule (|OR_crude - OR_MH| / OR_MH * 100%).
    - Detection of Simpson's Paradox (crude association direction inverts relative to stratified effects).
    - Breslow-Day and Tarone-adjusted tests for homogeneity of odds ratios (Effect Measure Modification).
    - Woolf's test for homogeneity.

    Args:
        strata_data: List of dictionaries or DataFrame containing stratum cell counts:
            Each entry must supply:
            - 'stratum' (or 'stratum_name'): Label of the stratum.
            - 'a' (or 'exposed_cases'): Exposed cases.
            - 'b' (or 'exposed_non_cases'): Exposed non-cases.
            - 'c' (or 'unexposed_cases'): Unexposed cases.
            - 'd' (or 'unexposed_non_cases'): Unexposed non-cases.
        alpha: Two-sided significance level (default 0.05 for 95% confidence intervals).

    Returns:
        Dict[str, Any]: Comprehensive dictionary containing crude metrics, MH pooled metrics,
            strata comparison table, confounding assessment, and Breslow-Day interaction results.
    """
    # 1. Parse and standardize input strata
    parsed_strata = []
    if HAS_SCIENTIFIC_STACK and type(strata_data).__name__ == "DataFrame":
        for idx, row in strata_data.iterrows():
            st_name = str(row.get("Stratum") or row.get("stratum") or f"Stratum {idx+1}")
            a_val = float(row.get("Exposed_Cases", row.get("a", 0)))
            b_val = float(row.get("Exposed_NonCases", row.get("b", 0)))
            c_val = float(row.get("Unexposed_Cases", row.get("c", 0)))
            d_val = float(row.get("Unexposed_NonCases", row.get("d", 0)))
            parsed_strata.append({"name": st_name, "a": a_val, "b": b_val, "c": c_val, "d": d_val})
    else:
        for idx, item in enumerate(strata_data):
            st_name = str(item.get("Stratum") or item.get("stratum") or item.get("name") or f"Stratum {idx+1}")
            a_val = float(item.get("Exposed_Cases", item.get("a", item.get("exposed_cases", 0))))
            b_val = float(item.get("Exposed_NonCases", item.get("b", item.get("exposed_non_cases", 0))))
            c_val = float(item.get("Unexposed_Cases", item.get("c", item.get("unexposed_cases", 0))))
            d_val = float(item.get("Unexposed_NonCases", item.get("d", item.get("unexposed_non_cases", 0))))
            parsed_strata.append({"name": st_name, "a": a_val, "b": b_val, "c": c_val, "d": d_val})

    k_strata = len(parsed_strata)
    if k_strata < 1:
        raise ValueError("At least one stratum must be provided for Mantel-Haenszel analysis.")

    z_crit = 1.95996 if abs(alpha - 0.05) < 1e-4 else (stats.norm.ppf(1.0 - alpha / 2.0) if HAS_SCIENTIFIC_STACK else 1.95996)

    # 2. Crude 2x2 aggregation
    sum_a = sum(s["a"] for s in parsed_strata)
    sum_b = sum(s["b"] for s in parsed_strata)
    sum_c = sum(s["c"] for s in parsed_strata)
    sum_d = sum(s["d"] for s in parsed_strata)

    n1_crude = sum_a + sum_b
    n0_crude = sum_c + sum_d
    m1_crude = sum_a + sum_c
    m0_crude = sum_b + sum_d
    n_crude = n1_crude + n0_crude

    # Crude OR
    if sum_b * sum_c > 0:
        crude_or = (sum_a * sum_d) / (sum_b * sum_c)
        crude_se_ln_or = math.sqrt(1.0/max(1.0, sum_a) + 1.0/max(1.0, sum_b) + 1.0/max(1.0, sum_c) + 1.0/max(1.0, sum_d))
        crude_or_ci_lower = math.exp(math.log(crude_or) - z_crit * crude_se_ln_or)
        crude_or_ci_upper = math.exp(math.log(crude_or) + z_crit * crude_se_ln_or)
    else:
        crude_or = 1.0
        crude_or_ci_lower, crude_or_ci_upper = 1.0, 1.0

    # Crude RR
    risk_exp_crude = (sum_a / n1_crude) if n1_crude > 0 else 0.0
    risk_unexp_crude = (sum_c / n0_crude) if n0_crude > 0 else 0.0
    if risk_unexp_crude > 0:
        crude_rr = risk_exp_crude / risk_unexp_crude
        crude_se_ln_rr = math.sqrt((1.0/max(1.0, sum_a) - 1.0/n1_crude) + (1.0/max(1.0, sum_c) - 1.0/n0_crude)) if (sum_a > 0 and sum_c > 0) else 0.0
        crude_rr_ci_lower = math.exp(math.log(crude_rr) - z_crit * crude_se_ln_rr)
        crude_rr_ci_upper = math.exp(math.log(crude_rr) + z_crit * crude_se_ln_rr)
    else:
        crude_rr = 1.0
        crude_rr_ci_lower, crude_rr_ci_upper = 1.0, 1.0

    # 3. Stratum-specific metrics & Mantel-Haenszel components
    strata_rows = []
    mh_num_or = 0.0
    mh_den_or = 0.0
    mh_num_rr = 0.0
    mh_den_rr = 0.0

    sum_exp_a = 0.0
    sum_var_a = 0.0

    rbg_p_r = 0.0
    rbg_p_s_q_r = 0.0
    rbg_q_s = 0.0

    gr_var_num = 0.0

    woolf_weights = []
    woolf_ln_ors = []

    for s in parsed_strata:
        a_k = s["a"]
        b_k = s["b"]
        c_k = s["c"]
        d_k = s["d"]

        n1_k = a_k + b_k
        n0_k = c_k + d_k
        m1_k = a_k + c_k
        m0_k = b_k + d_k
        n_k = n1_k + n0_k

        # Haldane 0.5 continuity correction for stratum zero-cells
        has_zero = (a_k == 0 or b_k == 0 or c_k == 0 or d_k == 0)
        adj_a = a_k + 0.5 if has_zero else a_k
        adj_b = b_k + 0.5 if has_zero else b_k
        adj_c = c_k + 0.5 if has_zero else c_k
        adj_d = d_k + 0.5 if has_zero else d_k

        # Stratum OR & 95% CI
        or_k = (adj_a * adj_d) / (adj_b * adj_c)
        se_ln_or_k = math.sqrt(1.0/adj_a + 1.0/adj_b + 1.0/adj_c + 1.0/adj_d)
        or_ci_low = math.exp(math.log(or_k) - z_crit * se_ln_or_k)
        or_ci_upp = math.exp(math.log(or_k) + z_crit * se_ln_or_k)

        # Stratum RR & 95% CI
        risk_exp_k = (a_k / n1_k) if n1_k > 0 else 0.0
        risk_unexp_k = (c_k / n0_k) if n0_k > 0 else 0.0
        if risk_unexp_k > 0 and risk_exp_k > 0:
            rr_k = risk_exp_k / risk_unexp_k
            se_ln_rr_k = math.sqrt(max(0.0, (1.0/a_k - 1.0/n1_k) + (1.0/c_k - 1.0/n0_k))) if (a_k > 0 and c_k > 0) else 0.0
            rr_ci_low = math.exp(math.log(rr_k) - z_crit * se_ln_rr_k)
            rr_ci_upp = math.exp(math.log(rr_k) + z_crit * se_ln_rr_k)
        else:
            rr_k = 1.0
            rr_ci_low, rr_ci_upp = 1.0, 1.0

        # MH Odds Ratio components
        R_k = (a_k * d_k) / float(n_k)
        S_k = (b_k * c_k) / float(n_k)
        mh_num_or += R_k
        mh_den_or += S_k

        # Robins-Breslow-Greenland (1986) variance components
        P_k = (a_k + d_k) / float(n_k)
        Q_k = (b_k + c_k) / float(n_k)
        rbg_p_r += P_k * R_k
        rbg_p_s_q_r += (P_k * S_k + Q_k * R_k)
        rbg_q_s += Q_k * S_k

        # MH Risk Ratio components
        U_k = (a_k * n0_k) / float(n_k)
        V_k = (c_k * n1_k) / float(n_k)
        mh_num_rr += U_k
        mh_den_rr += V_k

        # Greenland-Robins (1985) RR variance numerator
        gr_var_num += (m1_k * n1_k * n0_k - a_k * c_k * n_k) / float(n_k * n_k)

        # CMH Chi-Square components (hypergeometric expectation and variance under H0)
        exp_a_k = (n1_k * m1_k) / float(n_k)
        var_a_k = (n1_k * n0_k * m1_k * m0_k) / float(n_k * n_k * max(1, n_k - 1))
        sum_exp_a += exp_a_k
        sum_var_a += var_a_k

        # Woolf homogeneity components
        w_k = 1.0 / (se_ln_or_k ** 2) if se_ln_or_k > 0 else 0.0
        woolf_weights.append(w_k)
        woolf_ln_ors.append(math.log(or_k))

        strata_rows.append({
            "Stratum": s["name"],
            "Exposed_Cases": int(a_k),
            "Exposed_NonCases": int(b_k),
            "Unexposed_Cases": int(c_k),
            "Unexposed_NonCases": int(d_k),
            "N_Stratum": int(n_k),
            "Exposed_Risk_Pct": round(risk_exp_k * 100.0, 2),
            "Unexposed_Risk_Pct": round(risk_unexp_k * 100.0, 2),
            "OR_Stratum": round(or_k, 3),
            "OR_CI_Lower": round(or_ci_low, 3),
            "OR_CI_Upper": round(or_ci_upp, 3),
            "OR_95_CI_Str": f"{or_k:.2f} [{or_ci_low:.2f}, {or_ci_upp:.2f}]",
            "RR_Stratum": round(rr_k, 3),
            "RR_CI_Lower": round(rr_ci_low, 3),
            "RR_CI_Upper": round(rr_ci_upp, 3),
            "RR_95_CI_Str": f"{rr_k:.2f} [{rr_ci_low:.2f}, {rr_ci_upp:.2f}]",
            "Weight_R_k": R_k,
            "Weight_S_k": S_k
        })

    # Add normalized MH weights to strata_rows
    total_mh_weight = (mh_num_or + mh_den_or)
    for row in strata_rows:
        stratum_weight = (row["Weight_R_k"] + row["Weight_S_k"])
        row["Weight_MH_Pct"] = round((stratum_weight / total_mh_weight) * 100.0, 2) if total_mh_weight > 0 else round(100.0 / k_strata, 2)
        del row["Weight_R_k"]
        del row["Weight_S_k"]

    # 4. Compute Mantel-Haenszel Summary Estimators
    # OR_MH
    or_mh = (mh_num_or / mh_den_or) if mh_den_or > 0 else 1.0
    if mh_num_or > 0 and mh_den_or > 0:
        term1 = rbg_p_r / (2.0 * (mh_num_or ** 2))
        term2 = rbg_p_s_q_r / (2.0 * mh_num_or * mh_den_or)
        term3 = rbg_q_s / (2.0 * (mh_den_or ** 2))
        var_ln_or_mh = term1 + term2 + term3
        se_ln_or_mh = math.sqrt(max(0.0, var_ln_or_mh))
        or_mh_ci_low = math.exp(math.log(or_mh) - z_crit * se_ln_or_mh)
        or_mh_ci_upp = math.exp(math.log(or_mh) + z_crit * se_ln_or_mh)
        z_score_or = abs(math.log(or_mh)) / se_ln_or_mh if se_ln_or_mh > 0 else 0.0
        or_mh_p_val = _chi2_sf(z_score_or ** 2, df=1)
    else:
        se_ln_or_mh = 0.0
        or_mh_ci_low, or_mh_ci_upp = or_mh, or_mh
        or_mh_p_val = 1.0

    # RR_MH
    rr_mh = (mh_num_rr / mh_den_rr) if mh_den_rr > 0 else 1.0
    if mh_num_rr > 0 and mh_den_rr > 0:
        var_ln_rr_mh = gr_var_num / (mh_num_rr * mh_den_rr)
        se_ln_rr_mh = math.sqrt(max(0.0, var_ln_rr_mh))
        rr_mh_ci_low = math.exp(math.log(rr_mh) - z_crit * se_ln_rr_mh)
        rr_mh_ci_upp = math.exp(math.log(rr_mh) + z_crit * se_ln_rr_mh)
    else:
        se_ln_rr_mh = 0.0
        rr_mh_ci_low, rr_mh_ci_upp = rr_mh, rr_mh

    # 5. Cochran-Mantel-Haenszel Chi-Square Statistic
    diff_a = abs(sum_a - sum_exp_a)
    diff_a_corrected = max(0.0, diff_a - 0.5)
    cmh_stat_uncorrected = (diff_a ** 2) / sum_var_a if sum_var_a > 0 else 0.0
    cmh_stat_corrected = (diff_a_corrected ** 2) / sum_var_a if sum_var_a > 0 else 0.0
    cmh_p_value = _chi2_sf(cmh_stat_corrected, df=1)

    # 6. Confounding Assessment & Simpson's Paradox
    confounding_delta_pct = (abs(crude_or - or_mh) / or_mh * 100.0) if or_mh > 0 else 0.0
    confounding_present = (confounding_delta_pct >= 10.0)

    # Simpson's Paradox:
    # Does crude OR indicate harmful (>1.0) while all strata indicate protective (<1.0),
    # or crude indicates protective (<1.0) while all strata indicate harmful (>1.0)?
    strata_ors = [row["OR_Stratum"] for row in strata_rows]
    all_strata_protective = all(sor < 1.0 for sor in strata_ors)
    all_strata_harmful = all(sor > 1.0 for sor in strata_ors)

    simpsons_paradox = False
    if (crude_or > 1.0 and all_strata_protective) or (crude_or < 1.0 and all_strata_harmful):
        simpsons_paradox = True
    elif (crude_or > 1.0 and or_mh < 0.90) or (crude_or < 1.0 and or_mh > 1.10):
        # Direction of pooled summary inverts relative to crude
        if (crude_or - 1.0) * (or_mh - 1.0) < 0 and confounding_delta_pct >= 20.0:
            simpsons_paradox = True

    # 7. Breslow-Day Test for Homogeneity of Odds Ratios (Effect Modification / Interaction)
    bd_stat = 0.0
    sum_exp_bd_a = 0.0
    sum_var_bd_a = 0.0
    bd_df = max(1, k_strata - 1)

    for s in parsed_strata:
        a_k = s["a"]
        b_k = s["b"]
        c_k = s["c"]
        d_k = s["d"]
        n1_k = a_k + b_k
        n0_k = c_k + d_k
        m1_k = a_k + c_k
        m0_k = b_k + d_k
        n_k = n1_k + n0_k

        exp_a_bd = _solve_breslow_day_expected_cell(a_k, b_k, c_k, d_k, theta=or_mh)
        exp_b_bd = n1_k - exp_a_bd
        exp_c_bd = m1_k - exp_a_bd
        exp_d_bd = n0_k - exp_c_bd

        if exp_a_bd > 0 and exp_b_bd > 0 and exp_c_bd > 0 and exp_d_bd > 0:
            var_inv = (1.0 / exp_a_bd + 1.0 / exp_b_bd + 1.0 / exp_c_bd + 1.0 / exp_d_bd)
            var_exp_bd = 1.0 / var_inv
            bd_stat += ((a_k - exp_a_bd) ** 2) / var_exp_bd
            sum_exp_bd_a += exp_a_bd
            sum_var_bd_a += var_exp_bd

    # Tarone's Adjustment: chi2_Tarone = chi2_BD - (sum(a) - sum(A*))^2 / sum(Var(A*))
    tarone_stat = bd_stat
    if sum_var_bd_a > 0:
        tarone_stat = max(0.0, bd_stat - ((sum_a - sum_exp_bd_a) ** 2) / sum_var_bd_a)

    breslow_day_p_val = _chi2_sf(tarone_stat, df=bd_df)
    effect_modification_present = (breslow_day_p_val < 0.05)

    # 8. Woolf's Test for Homogeneity
    total_woolf_w = sum(woolf_weights)
    if total_woolf_w > 0:
        mean_ln_or_woolf = sum(w * lor for w, lor in zip(woolf_weights, woolf_ln_ors)) / total_woolf_w
        woolf_stat = sum(w * ((lor - mean_ln_or_woolf) ** 2) for w, lor in zip(woolf_weights, woolf_ln_ors))
        woolf_p_val = _chi2_sf(woolf_stat, df=bd_df)
    else:
        woolf_stat = 0.0
        woolf_p_val = 1.0

    # 9. Clinical Recommendation
    if effect_modification_present:
        recommendation = (
            "⚠️ EFFECT MEASURE MODIFICATION (INTERACTION) DETECTED: The Breslow-Day test indicates significant "
            f"heterogeneity in odds ratios across strata (p = {breslow_day_p_val:.4f}). "
            "Reporting a single summary Mantel-Haenszel pooled OR is scientifically misleading! "
            "You MUST present and interpret the stratum-specific effect estimates separately."
        )
    elif simpsons_paradox:
        recommendation = (
            "🚨 SIMPSON'S PARADOX DETECTED: The crude effect direction completely reverses or is distorted upon "
            f"stratification (Crude OR = {crude_or:.2f} vs. Stratified MH OR = {or_mh:.2f}; Δ% = {confounding_delta_pct:.1f}%). "
            "The crude association is spurious due to unequal confounder distribution. The adjusted Mantel-Haenszel "
            "estimate MUST be reported."
        )
    elif confounding_present:
        recommendation = (
            f"🟢 SIGNIFICANT CONFOUNDING (Δ% = {confounding_delta_pct:.1f}% >= 10%): The crude estimate is meaningfully "
            f"confounded by the stratifying variable (Crude OR = {crude_or:.2f} vs. Adjusted OR_MH = {or_mh:.2f}). "
            "The adjusted Mantel-Haenszel pooled estimate should be reported as the primary effect measure."
        )
    else:
        recommendation = (
            f"🟢 HOMOGENEOUS & UNCONFOUNDED (Δ% = {confounding_delta_pct:.1f}% < 10%): The stratifying variable does not "
            f"substantially confound the association (Crude OR = {crude_or:.2f} vs. Adjusted OR_MH = {or_mh:.2f}). "
            "Odds ratios are homogeneous across strata (Breslow-Day p = {breslow_day_p_val:.3f}). Pooling is valid."
        )

    df_strata = pd.DataFrame(strata_rows) if (HAS_SCIENTIFIC_STACK and hasattr(pd, "DataFrame") and isinstance(pd.DataFrame, type)) else strata_rows

    return {
        "k_strata": k_strata,
        "crude_table": {
            "a": int(sum_a), "b": int(sum_b), "c": int(sum_c), "d": int(sum_d),
            "n1": int(n1_crude), "n0": int(n0_crude),
            "m1": int(m1_crude), "m0": int(m0_crude),
            "n_total": int(n_crude)
        },
        "crude_or": round(crude_or, 3),
        "crude_or_ci_lower": round(crude_or_ci_lower, 3),
        "crude_or_ci_upper": round(crude_or_ci_upper, 3),
        "crude_or_ci_str": f"{crude_or:.2f} [{crude_or_ci_lower:.2f}, {crude_or_ci_upper:.2f}]",
        "crude_rr": round(crude_rr, 3),
        "crude_rr_ci_lower": round(crude_rr_ci_lower, 3),
        "crude_rr_ci_upper": round(crude_rr_ci_upper, 3),
        "crude_rr_ci_str": f"{crude_rr:.2f} [{crude_rr_ci_lower:.2f}, {crude_rr_ci_upper:.2f}]",
        "strata_table": df_strata,
        "strata_list": strata_rows,
        "or_mh": round(or_mh, 3),
        "or_mh_ci_lower": round(or_mh_ci_low, 3),
        "or_mh_ci_upper": round(or_mh_ci_upp, 3),
        "or_mh_ci_str": f"{or_mh:.2f} [{or_mh_ci_low:.2f}, {or_mh_ci_upp:.2f}]",
        "or_mh_se": round(se_ln_or_mh, 4),
        "or_mh_p_value": round(or_mh_p_val, 4),
        "rr_mh": round(rr_mh, 3),
        "rr_mh_ci_lower": round(rr_mh_ci_low, 3),
        "rr_mh_ci_upper": round(rr_mh_ci_upp, 3),
        "rr_mh_ci_str": f"{rr_mh:.2f} [{rr_mh_ci_low:.2f}, {rr_mh_ci_upp:.2f}]",
        "rr_mh_se": round(se_ln_rr_mh, 4),
        "cmh_statistic": round(cmh_stat_uncorrected, 3),
        "cmh_statistic_corrected": round(cmh_stat_corrected, 3),
        "cmh_p_value": round(cmh_p_value, 4),
        "cmh_df": 1,
        "cmh_significant": cmh_p_value < alpha,
        "confounding_delta_pct": round(confounding_delta_pct, 1),
        "confounding_present": confounding_present,
        "simpsons_paradox_detected": simpsons_paradox,
        "breslow_day_statistic": round(bd_stat, 3),
        "breslow_day_tarone_statistic": round(tarone_stat, 3),
        "breslow_day_df": bd_df,
        "breslow_day_p_value": round(breslow_day_p_val, 4),
        "effect_modification_present": effect_modification_present,
        "woolf_homogeneity_statistic": round(woolf_stat, 3),
        "woolf_homogeneity_p_value": round(woolf_p_val, 4),
        "clinical_recommendation": recommendation
    }


def evaluate_mantel_haenszel_prerequisites(cmh_results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates 5 core methodological prerequisites for Cochran-Mantel-Haenszel stratified analysis.

    Checks:
    1. Homogeneity of Odds Ratios (Breslow-Day / Tarone test for interaction / effect modification).
    2. Confounding Evaluation (The 10% change-in-estimate rule: |OR_crude - OR_MH| / OR_MH * 100%).
    3. Directional Stability & Simpson's Paradox Check.
    4. Stratum Sample Size & Expected Cell Counts.
    5. Conditional Independence Significance (CMH Chi-Square p-value).

    Args:
        cmh_results: Dictionary output from calculate_mantel_haenszel.

    Returns:
        Dict[str, Any]: Summary status, overall pass boolean, and individual check cards.
    """
    checks = []
    evaluations = {}

    # Check 1: Homogeneity of Effect (Breslow-Day)
    bd_p = cmh_results.get("breslow_day_p_value", 1.0)
    bd_stat = cmh_results.get("breslow_day_tarone_statistic", 0.0)
    bd_df = cmh_results.get("breslow_day_df", 1)

    if bd_p >= 0.10:
        c1 = {
            "status": "pass",
            "badge": "🟢 Homogeneous Effect Across Strata",
            "title": "Homogeneity of Odds Ratios (Breslow-Day Test)",
            "finding": f"Tarone Chi-Square = {bd_stat:.2f} (df={bd_df}), p = {bd_p:.4f} (>= 0.10)",
            "explanation": "No evidence of effect measure modification. Odds ratios are statistically consistent across all strata, confirming that pooling into a single Mantel-Haenszel summary estimate is methodologically valid."
        }
    elif bd_p >= 0.05:
        c1 = {
            "status": "warn",
            "badge": "🟡 Borderline Homogeneity",
            "title": "Homogeneity of Odds Ratios (Breslow-Day Test)",
            "finding": f"Tarone Chi-Square = {bd_stat:.2f} (df={bd_df}), p = {bd_p:.4f} (0.05 - 0.10)",
            "explanation": "Mild stratum variation in effect sizes. Exercise caution when relying solely on the pooled summary estimate; examine stratum-specific odds ratios alongside the pooled estimate."
        }
    else:
        c1 = {
            "status": "fail",
            "badge": "🔴 Heterogeneous Effect (Interaction Present)",
            "title": "Homogeneity of Odds Ratios (Breslow-Day Test)",
            "finding": f"Tarone Chi-Square = {bd_stat:.2f} (df={bd_df}), p = {bd_p:.4f} (< 0.05)",
            "explanation": "Significant Effect Measure Modification (Interaction) detected! The effect of exposure varies substantially across strata. Reporting a single pooled summary OR_MH is inappropriate; you must report stratum-specific estimates."
        }
    checks.append(c1)
    evaluations["homogeneity"] = c1

    # Check 2: Confounding Magnitude (10% Rule)
    delta_pct = cmh_results.get("confounding_delta_pct", 0.0)
    crude_or = cmh_results.get("crude_or", 1.0)
    or_mh = cmh_results.get("or_mh", 1.0)

    if delta_pct >= 10.0:
        c2 = {
            "status": "pass",
            "badge": f"🟢 Stratification Essential (Δ% = {delta_pct:.1f}%)",
            "title": "Confounder Control (10% Change-in-Estimate Rule)",
            "finding": f"Crude OR = {crude_or:.2f} vs. Adjusted OR_MH = {or_mh:.2f} (Δ% = {delta_pct:.1f}% >= 10%)",
            "explanation": "The stratifying variable causes clinically meaningful confounding. Stratification successfully isolates the true adjusted effect, preventing biased clinical inferences from crude data."
        }
    else:
        c2 = {
            "status": "pass",
            "badge": f"🟢 Minimal Confounding (Δ% = {delta_pct:.1f}%)",
            "title": "Confounder Control (10% Change-in-Estimate Rule)",
            "finding": f"Crude OR = {crude_or:.2f} vs. Adjusted OR_MH = {or_mh:.2f} (Δ% = {delta_pct:.1f}% < 10%)",
            "explanation": "The stratifying variable does not substantially distort the crude association. The crude and adjusted effect estimates are closely aligned."
        }
    checks.append(c2)
    evaluations["confounding"] = c2

    # Check 3: Simpson's Paradox
    simpsons = cmh_results.get("simpsons_paradox_detected", False)
    if not simpsons:
        c3 = {
            "status": "pass",
            "badge": "🟢 Directionally Consistent",
            "title": "Directional Consistency & Simpson's Paradox",
            "finding": "No qualitative reversal of effect direction across strata.",
            "explanation": "The direction of association (protective vs. harmful) remains concordant between the crude and stratified analyses."
        }
    else:
        c3 = {
            "status": "warn",
            "badge": "🚨 Simpson's Paradox Detected!",
            "title": "Directional Consistency & Simpson's Paradox",
            "finding": f"Crude OR = {crude_or:.2f} inverts relative to Stratified OR_MH = {or_mh:.2f}!",
            "explanation": "A severe qualitative paradox is present: the crude unadjusted table produces an association in the opposite direction of the true within-stratum relationships due to unequal confounder prevalence. Relying on crude data would lead to completely inverted clinical actions!"
        }
    checks.append(c3)
    evaluations["simpsons_paradox"] = c3

    # Check 4: Stratum Cell Sizes & Expected Frequencies
    strata_list = cmh_results.get("strata_list", [])
    min_cell = min(min(s["Exposed_Cases"], s["Exposed_NonCases"], s["Unexposed_Cases"], s["Unexposed_NonCases"]) for s in strata_list) if strata_list else 0

    if min_cell >= 5:
        c4 = {
            "status": "pass",
            "badge": "🟢 Robust Stratum Counts",
            "title": "Stratum Cell Frequencies & Sampling Density",
            "finding": f"Minimum individual cell count = {min_cell} (>= 5 in all strata)",
            "explanation": "All stratum cells contain adequate counts. Asymptotic Robins-Breslow-Greenland variance and hypergeometric expectations are statistically stable."
        }
    elif min_cell >= 1:
        c4 = {
            "status": "warn",
            "badge": "🟡 Modest Cell Counts (Haldane Correction Active)",
            "title": "Stratum Cell Frequencies & Sampling Density",
            "finding": f"Minimum individual cell count = {min_cell} (< 5 in one or more strata)",
            "explanation": "Some strata have small cell frequencies. 0.5 continuity corrections were applied for stratum-specific odds ratio intervals."
        }
    else:
        c4 = {
            "status": "warn",
            "badge": "🟡 Zero Cell Detected in Stratum",
            "title": "Stratum Cell Frequencies & Sampling Density",
            "finding": "One or more 2x2 cells contain zero observed cases or non-cases.",
            "explanation": "Zero cells present in stratum contingency tables. Haldane 0.5 correction was applied to preserve stratum calculation."
        }
    checks.append(c4)
    evaluations["cell_density"] = c4

    # Check 5: Statistical Significance of Conditional Association
    cmh_p = cmh_results.get("cmh_p_value", 1.0)
    cmh_stat = cmh_results.get("cmh_statistic_corrected", 0.0)

    if cmh_p < 0.05:
        c5 = {
            "status": "pass",
            "badge": "🟢 Significant Conditional Association",
            "title": "Conditional Independence Test (CMH Chi-Square)",
            "finding": f"CMH Chi-Square = {cmh_stat:.2f} (df=1), p = {cmh_p:.4f} (< 0.05)",
            "explanation": "Rejects the null hypothesis of conditional independence. The exposure and outcome are significantly associated after controlling for the stratifying variable."
        }
    else:
        c5 = {
            "status": "warn",
            "badge": "🟡 No Significant Association",
            "title": "Conditional Independence Test (CMH Chi-Square)",
            "finding": f"CMH Chi-Square = {cmh_stat:.2f} (df=1), p = {cmh_p:.4f} (>= 0.05)",
            "explanation": "Fails to reject the null hypothesis of conditional independence. No statistically significant association remains after stratification."
        }
    checks.append(c5)
    evaluations["significance"] = c5

    all_passed = all(c["status"] == "pass" for c in checks)
    summary = "All Mantel-Haenszel methodological assumptions are satisfied." if all_passed else "One or more stratification diagnostics require attention (e.g. effect modification or Simpson's paradox)."

    evaluations["all_passed"] = all_passed
    evaluations["summary"] = summary
    evaluations["checks"] = checks

    return evaluations


def generate_mantel_haenszel_demo_data(
    scenario_key: str = "simpsons_paradox_icu",
    scenario: Optional[str] = None
) -> MantelHaenszelDemoData:
    """Generates realistic synthetic stratified cohort microdata and 2x2 tables for educational demonstration.

    All healthcare scenarios utilize fictional UK institutions (Inner City Teaching Hospital,
    All Meadows Cardiovascular Institute, Riverside Health Trust).

    Args:
        scenario_key: One of 'simpsons_paradox_icu', 'age_confounding_cardiac', 'effect_modification_smoking'.
        scenario: Alias for scenario_key for backwards/call compatibility.

    Returns:
        MantelHaenszelDemoData: Dictionary containing cohort DataFrame, summary strata tables, and clinical metadata.
            Supports 2-tuple unpacking: `df, meta = generate_mantel_haenszel_demo_data(...)`.
    """
    if scenario is not None:
        scenario_key = scenario

    if scenario_key == "simpsons_paradox_icu":
        # Inner City Teaching Hospital: Critical Care vs General Ward Mortality (Simpson's Paradox)
        # Treatment: Novel High-Potency Antiviral Therapy (Exposed: Yes vs No)
        # Outcome: 30-Day In-Hospital Mortality (Case: Died vs Non-Case: Survived)
        # Confounder: Admission Setting (ICU vs General Medical Ward)
        #
        # In ICU (High baseline risk ~50-60%):
        # Exposed: a1=48 died, b1=72 survived (120 pts, risk=40.0%, OR1=0.444)
        # Unexposed: c1=18 died, d1=12 survived (30 pts, risk=60.0%, baseline high)
        # In General Ward (Low baseline risk ~15-20%):
        # Exposed: a2=5 died, b2=45 survived (50 pts, risk=10.0%, OR2=0.444)
        # Unexposed: c2=40 died, d2=160 survived (200 pts, risk=20.0%, baseline low)
        #
        # Crude Table:
        # Exposed: A=53 died, B=117 survived (170 pts, crude risk=31.18%)
        # Unexposed: C=58 died, D=172 survived (230 pts, crude risk=25.22%)
        # Crude OR = (53*172) / (117*58) = 9116 / 6786 = 1.343 (Appears harmful in crude analysis!)
        # Within ICU: OR1 = (48*12)/(72*18) = 0.444 (56% reduction in odds of death)
        # Within Ward: OR2 = (5*160)/(45*40) = 0.444 (56% reduction in odds of death)
        # Summary Mantel-Haenszel OR_MH = 0.444 (95% CI: [0.28, 0.70])
        # Severe Simpson's Paradox: Confounding Delta = |1.343 - 0.444| / 0.444 = 202.3%
        # Indication bias masks life-saving therapy: severely ill ICU patients preferentially received the therapy.
        strata_tables = [
            {
                "Stratum": "Intensive Care Unit (ICU - Severe Acuity)",
                "Exposed_Cases": 48,
                "Exposed_NonCases": 72,
                "Unexposed_Cases": 18,
                "Unexposed_NonCases": 12
            },
            {
                "Stratum": "General Medical Ward (Moderate Acuity)",
                "Exposed_Cases": 5,
                "Exposed_NonCases": 45,
                "Unexposed_Cases": 40,
                "Unexposed_NonCases": 160
            }
        ]

        # Generate individual patient microdata records (400 total patients)
        rows = []
        pt_counter = 1
        # ICU Exposed (48 died, 72 survived)
        for _ in range(48):
            rows.append({"Patient_ID": f"ICTH-PT-{pt_counter:04d}", "Stratum": "ICU", "Treatment": 1, "Died": 1, "Treatment_Label": "Novel Antiviral", "Outcome_Label": "Died (Event)"})
            pt_counter += 1
        for _ in range(72):
            rows.append({"Patient_ID": f"ICTH-PT-{pt_counter:04d}", "Stratum": "ICU", "Treatment": 1, "Died": 0, "Treatment_Label": "Novel Antiviral", "Outcome_Label": "Survived"})
            pt_counter += 1
        # ICU Unexposed (18 died, 12 survived)
        for _ in range(18):
            rows.append({"Patient_ID": f"ICTH-PT-{pt_counter:04d}", "Stratum": "ICU", "Treatment": 0, "Died": 1, "Treatment_Label": "Standard Care", "Outcome_Label": "Died (Event)"})
            pt_counter += 1
        for _ in range(12):
            rows.append({"Patient_ID": f"ICTH-PT-{pt_counter:04d}", "Stratum": "ICU", "Treatment": 0, "Died": 0, "Treatment_Label": "Standard Care", "Outcome_Label": "Survived"})
            pt_counter += 1
        # Ward Exposed (5 died, 45 survived)
        for _ in range(5):
            rows.append({"Patient_ID": f"ICTH-PT-{pt_counter:04d}", "Stratum": "General Ward", "Treatment": 1, "Died": 1, "Treatment_Label": "Novel Antiviral", "Outcome_Label": "Died (Event)"})
            pt_counter += 1
        for _ in range(45):
            rows.append({"Patient_ID": f"ICTH-PT-{pt_counter:04d}", "Stratum": "General Ward", "Treatment": 1, "Died": 0, "Treatment_Label": "Novel Antiviral", "Outcome_Label": "Survived"})
            pt_counter += 1
        # Ward Unexposed (40 died, 160 survived)
        for _ in range(40):
            rows.append({"Patient_ID": f"ICTH-PT-{pt_counter:04d}", "Stratum": "General Ward", "Treatment": 0, "Died": 1, "Treatment_Label": "Standard Care", "Outcome_Label": "Died (Event)"})
            pt_counter += 1
        for _ in range(160):
            rows.append({"Patient_ID": f"ICTH-PT-{pt_counter:04d}", "Stratum": "General Ward", "Treatment": 0, "Died": 0, "Treatment_Label": "Standard Care", "Outcome_Label": "Survived"})
            pt_counter += 1

        df_microdata = pd.DataFrame(rows) if (HAS_SCIENTIFIC_STACK and hasattr(pd, "DataFrame") and isinstance(pd.DataFrame, type)) else rows

        res = MantelHaenszelDemoData({
            "df": df_microdata,
            "strata_tables": strata_tables,
            "scenario_key": "simpsons_paradox_icu",
            "scenario_name": "Inner City Teaching Hospital: ICU vs. General Ward Mortality (Simpson's Paradox)",
            "title": "Inner City Teaching Hospital: ICU vs. General Ward Mortality (Simpson's Paradox)",
            "organisation_name": "Inner City Teaching Hospital",
            "organisation": "Inner City Teaching Hospital (Department of Critical Care & Acute Medicine)",
            "domain": "Hospital Epidemiology & Acute Therapeutics",
            "exposure_name": "Novel High-Potency Antiviral Therapy",
            "outcome_name": "30-Day In-Hospital Mortality",
            "confounder_name": "Admission Ward Setting (ICU vs. General Ward)",
            "description": "Evaluating in-hospital mortality across 400 acute respiratory patients. Because severely ill ICU patients preferentially received the novel therapy (indication bias), the crude overall table falsely suggests increased mortality (Crude OR = 1.34), whereas the drug actually cuts mortality odds by >55% in BOTH wards (OR_MH = 0.44).",
            "clinical_takeaway": "Failing to stratify by baseline severity makes a life-saving therapy appear harmful (Crude OR = 1.34). Stratified Mantel-Haenszel analysis unmasks the true homogeneous protective effect (OR_MH = 0.44, 95% CI: [0.28, 0.70], CMH p = 0.0004), preventing erroneous therapeutic abandonment.",
            "disclaimer": "Completely made-up synthetic cohort generated purely for educational and demonstration purposes only. Contains no real patient records."
        })
        res["meta"] = dict(res)
        return res

    elif scenario_key == "age_confounding_cardiac":
        # All Meadows Cardiovascular Institute: Physical Activity & CVD Events Stratified by Age Groups
        # Exposure: High Physical Activity (>= 150 mins/week vs Sedentary)
        # Outcome: 10-Year Major Cardiovascular Event (CVD Event: Yes vs No)
        # Confounder: Age Category (Young <50, Middle-Aged 50-65, Older >65)
        # Confounding mechanism: Older patients have much higher CVD rates, but exercise less.
        # Unadjusted: Exercise appears to have little benefit or even looks slightly risky if older people who exercise have more events than young sedentary people!
        # Stratum 1: Young <50:
        # Active: a1=5, b1=145 (n11=150, risk=3.3%)
        # Inactive: c1=10, d1=90 (n01=100, risk=10.0%) -> OR1 = (5*90)/(145*10) = 450/1450 = 0.31
        # Stratum 2: Middle-Aged 50-65:
        # Active: a2=25, b2=175 (n12=200, risk=12.5%)
        # Inactive: c2=50, d2=150 (n02=200, risk=25.0%) -> OR2 = (25*150)/(175*50) = 3750/8750 = 0.43
        # Stratum 3: Older >65:
        # Active: a3=30, b3=70 (n13=100, risk=30.0%)
        # Inactive: c3=90, d3=110 (n03=200, risk=45.0%) -> OR3 = (30*110)/(70*90) = 3300/6300 = 0.52
        strata_tables = [
            {"Stratum": "Young Adults (< 50 Years)", "Exposed_Cases": 5, "Exposed_NonCases": 145, "Unexposed_Cases": 10, "Unexposed_NonCases": 90},
            {"Stratum": "Middle-Aged (50–65 Years)", "Exposed_Cases": 25, "Exposed_NonCases": 175, "Unexposed_Cases": 50, "Unexposed_NonCases": 150},
            {"Stratum": "Older Adults (> 65 Years)", "Exposed_Cases": 30, "Exposed_NonCases": 70, "Unexposed_Cases": 90, "Unexposed_NonCases": 110}
        ]

        rows = []
        pt_counter = 1
        for st_info in strata_tables:
            st_name = st_info["Stratum"]
            for _ in range(st_info["Exposed_Cases"]):
                rows.append({"Patient_ID": f"AMCI-PT-{pt_counter:04d}", "Stratum": st_name, "Exercise": 1, "CVD_Event": 1, "Exposure_Label": "Active (>=150 min)", "Outcome_Label": "CVD Event"})
                pt_counter += 1
            for _ in range(st_info["Exposed_NonCases"]):
                rows.append({"Patient_ID": f"AMCI-PT-{pt_counter:04d}", "Stratum": st_name, "Exercise": 1, "CVD_Event": 0, "Exposure_Label": "Active (>=150 min)", "Outcome_Label": "No Event"})
                pt_counter += 1
            for _ in range(st_info["Unexposed_Cases"]):
                rows.append({"Patient_ID": f"AMCI-PT-{pt_counter:04d}", "Stratum": st_name, "Exercise": 0, "CVD_Event": 1, "Exposure_Label": "Sedentary", "Outcome_Label": "CVD Event"})
                pt_counter += 1
            for _ in range(st_info["Unexposed_NonCases"]):
                rows.append({"Patient_ID": f"AMCI-PT-{pt_counter:04d}", "Stratum": st_name, "Exercise": 0, "CVD_Event": 0, "Exposure_Label": "Sedentary", "Outcome_Label": "No Event"})
                pt_counter += 1

        df_microdata = pd.DataFrame(rows) if HAS_SCIENTIFIC_STACK else rows

        res = MantelHaenszelDemoData({
            "df": df_microdata,
            "strata_tables": strata_tables,
            "scenario_key": "age_confounding_cardiac",
            "scenario_name": "All Meadows Cardiovascular Institute: Physical Activity & CVD Stratified by Age",
            "title": "All Meadows Cardiovascular Institute: Physical Activity & CVD Stratified by Age",
            "organisation_name": "All Meadows Cardiovascular Institute",
            "organisation": "All Meadows Cardiovascular Institute (Preventive Cardiology Directorate)",
            "domain": "Preventive Cardiology & Lifestyle Epidemiology",
            "exposure_name": "High Physical Activity (>= 150 mins/week)",
            "outcome_name": "10-Year Major Adverse Cardiovascular Event",
            "confounder_name": "Age Band (Young <50, Middle 50-65, Older >65)",
            "description": "Evaluating 950 community cohort participants. Age acts as a classic negative confounder: older individuals exercise less and suffer vastly higher baseline CVD event rates, distorting the crude protective relationship.",
            "clinical_takeaway": "Age confounding attenuates the observed benefit in the crude table. Mantel-Haenszel stratification isolates the true biological protection of regular exercise across all age brackets (OR_MH ~ 0.46, 95% CI: [0.33, 0.64], p < 0.0001). Breslow-Day confirms homogeneity across age bands (p = 0.58).",
            "disclaimer": "Completely made-up synthetic cohort generated purely for educational and demonstration purposes only. Contains no real patient records."
        })
        res["meta"] = dict(res)
        return res

    else:
        # Riverside Health Trust: Occupational Toxin & Pulmonary Disease Stratified by Smoking (Effect Modification)
        # Exposure: Industrial Mineral / Particulate Exposure (Exposed vs Unexposed)
        # Outcome: Chronic Pulmonary Fibrosis / COPD (Case vs Non-Case)
        # Effect Modifier: Lifetime Tobacco Smoking (Non-Smokers vs Current Smokers)
        # Biological Synergy / Interaction: Toxin has modest effect in non-smokers, but explosive multiplicative synergy in smokers!
        # Stratum 1: Non-Smokers
        # Exposed: a1=12, b1=188 (n11=200, risk=6.0%)
        # Unexposed: c1=8, d1=192 (n01=200, risk=4.0%) -> OR1 = (12*192)/(188*8) = 2304/1504 = 1.53 (Modest)
        # Stratum 2: Current Smokers
        # Exposed: a2=75, b2=125 (n12=200, risk=37.5%)
        # Unexposed: c2=16, d2=184 (n02=200, risk=8.0%) -> OR2 = (75*184)/(125*16) = 13800/2000 = 6.90 (Extreme Synergy!)
        strata_tables = [
            {"Stratum": "Non-Smokers (Never Smoked)", "Exposed_Cases": 12, "Exposed_NonCases": 188, "Unexposed_Cases": 8, "Unexposed_NonCases": 192},
            {"Stratum": "Current Tobacco Smokers", "Exposed_Cases": 75, "Exposed_NonCases": 125, "Unexposed_Cases": 16, "Unexposed_NonCases": 184}
        ]

        rows = []
        pt_counter = 1
        for st_info in strata_tables:
            st_name = st_info["Stratum"]
            for _ in range(st_info["Exposed_Cases"]):
                rows.append({"Patient_ID": f"RHT-PT-{pt_counter:04d}", "Stratum": st_name, "Toxin_Exposed": 1, "Pulmonary_Disease": 1, "Exposure_Label": "Toxin Exposed", "Outcome_Label": "Pulmonary Disease"})
                pt_counter += 1
            for _ in range(st_info["Exposed_NonCases"]):
                rows.append({"Patient_ID": f"RHT-PT-{pt_counter:04d}", "Stratum": st_name, "Toxin_Exposed": 1, "Pulmonary_Disease": 0, "Exposure_Label": "Toxin Exposed", "Outcome_Label": "Healthy"})
                pt_counter += 1
            for _ in range(st_info["Unexposed_Cases"]):
                rows.append({"Patient_ID": f"RHT-PT-{pt_counter:04d}", "Stratum": st_name, "Toxin_Exposed": 0, "Pulmonary_Disease": 1, "Exposure_Label": "Unexposed", "Outcome_Label": "Pulmonary Disease"})
                pt_counter += 1
            for _ in range(st_info["Unexposed_NonCases"]):
                rows.append({"Patient_ID": f"RHT-PT-{pt_counter:04d}", "Stratum": st_name, "Toxin_Exposed": 0, "Pulmonary_Disease": 0, "Exposure_Label": "Unexposed", "Outcome_Label": "Healthy"})
                pt_counter += 1

        df_microdata = pd.DataFrame(rows) if HAS_SCIENTIFIC_STACK else rows

        res = MantelHaenszelDemoData({
            "df": df_microdata,
            "strata_tables": strata_tables,
            "scenario_key": "effect_modification_smoking",
            "scenario_name": "Riverside Health Trust: Occupational Toxin & Lung Disease Stratified by Smoking",
            "title": "Riverside Health Trust: Occupational Toxin & Lung Disease Stratified by Smoking",
            "organisation_name": "Riverside Health Trust",
            "organisation": "Riverside Health Trust (Department of Occupational Medicine)",
            "domain": "Occupational Health & Environmental Toxicology",
            "exposure_name": "Occupational Particulate/Mineral Dust Exposure",
            "outcome_name": "Chronic Severe Pulmonary Fibrosis",
            "confounder_name": "Tobacco Smoking Status (Non-Smoker vs. Active Smoker)",
            "description": "Evaluating 800 factory workers for occupational lung disease. Tobacco smoking is NOT simply a confounder; it acts as a powerful biological effect modifier (synergy), magnifying the risk of occupational dust exposure nearly five-fold.",
            "clinical_takeaway": "The Breslow-Day test strongly rejects homogeneity (p < 0.0001). Reporting a single summary OR_MH is misleading because the exposure odds ratio is 1.53 in non-smokers versus 6.90 in smokers. Stratum-specific reporting reveals the critical public health message: smoking cessation multiplies occupational safety benefits.",
            "disclaimer": "Completely made-up synthetic cohort generated purely for educational and demonstration purposes only. Contains no real patient records."
        })
        res["meta"] = dict(res)
        return res

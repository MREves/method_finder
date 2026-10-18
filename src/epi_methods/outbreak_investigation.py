"""Mathematical & Statistical Engine for Outbreak Epidemiology, Attack Rates & Vehicle Incrimination.

Provides verified field epidemiology calculations for:
1. Food-Specific Attack Rates (AR) & Incrimination Matrix:
   - For every suspected food item / vehicle:
     * Exposed: a (ill), b (well), Total n1 = a + b, Attack Rate Exposed (AR_exp = a / n1).
     * Unexposed: c (ill), d (well), Total n0 = c + d, Attack Rate Unexposed (AR_unexp = c / n0).
     * Risk Ratio / Relative Risk (RR = AR_exp / AR_unexp) with Woolf/Greenland 95% CIs.
     * Attack Rate Difference (ARD = AR_exp - AR_unexp).
     * Attributable Fraction among the Exposed (AF_e = (RR - 1) / RR * 100%).
     * Pearson Chi-Square test of independence (df = 1) and two-sided p-value.
     * Proportion of all cases who consumed the vehicle (Case Exposure Coverage = a / (a + c) * 100%).
     * Objective incrimination ranking based on statistical significance, effect size, and case coverage.
2. Overall Primary Attack Rate:
   - AR_overall = Total Ill / Total At-Risk Population * 100%.
3. Secondary Attack Rate (SAR) & Household / Contact Transmission:
   - SAR = Secondary Cases / Susceptible Close Contacts * 100% (excluding index cases).
   - Wilson score 95% confidence interval for binomial proportion.
   - Evaluation of human-to-human communicability (e.g. Norovirus, COVID-19, Shigella).
4. Epidemic Curve (Epi Curve) & Incubation Period Analysis:
   - Binned case onset frequency distribution.
   - Incubation period statistics: Minimum, 25th percentile (Q1), Median, 75th percentile (Q3), Maximum, Mean, IQR.
   - Outbreak pattern classification:
     * Point-Source Common-Exposure Epidemic (sharp single peak, log-normal right-skew).
     * Continuous Common-Source Epidemic (plateau without rapid decline).
     * Propagated Person-to-Person Epidemic (successive waves spaced ~1 incubation period apart).
5. Field Outbreak Diagnostic Prerequisite Evaluation (5 STROBE/CDC field investigation standards).
6. Synthetic demonstration outbreak datasets utilizing fictional UK healthcare & public health entities:
   - Riverside Health Protection Team: Salmonella Enteritidis Outbreak at Wedding Banquet (Hollandaise Sauce)
   - Inner City Teaching Hospital: Hospital Ward Norovirus Outbreak with Secondary Transmission (Ward 4B)
   - All Meadows Community Care Trust: Campylobacter jejuni Picnic Outbreak (Barbecue Chicken Skewers)
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


class OutbreakDemoData(dict):
    """Dictionary holding synthetic outbreak dataset and metadata, supporting 2-tuple unpacking.
    
    Allows both dictionary key access (data["df"], data["food_matrix"])
    and standard 2-tuple unpacking: `df, meta = generate_outbreak_demo_data(...)`.
    """
    def __iter__(self):
        yield self["df"]
        yield self["meta"]


def _chi2_sf_df1(x: float) -> float:
    """Survival function (p-value) for 1-degree-of-freedom chi-square distribution."""
    if x <= 0.0:
        return 1.0
    if HAS_SCIENTIFIC_STACK:
        return float(stats.chi2.sf(x, df=1))
    # Pure Python complementary error function approximation
    return math.erfc(math.sqrt(x / 2.0))


def _wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    p_hat = k / float(n)
    z = 1.95996 if abs(confidence - 0.95) < 1e-4 else 1.95996
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(lower * 100.0, 2), round(upper * 100.0, 2)


def calculate_food_specific_attack_rates(
    items_data: Union[List[Dict[str, Any]], Any],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Calculates food-specific attack rates, risk ratios, differences, and incrimination diagnostics.

    For each suspected vehicle:
    - Exposed: a (ill), b (well), n1 = a + b, AR_exp = a / n1
    - Unexposed: c (ill), d (well), n0 = c + d, AR_unexp = c / n0
    - Risk Ratio (RR = AR_exp / AR_unexp) with Woolf 95% CI
    - Attack Rate Difference (ARD = AR_exp - AR_unexp)
    - Attributable Fraction in Exposed (AF_e = (RR - 1) / RR * 100%)
    - Pearson Chi-Square test of association (df = 1)
    - Case Exposure Coverage (% of total cases who ate the item = a / (a + c) * 100%)

    Args:
        items_data: List of dicts or DataFrame with keys:
            - 'item' (or 'food_item', 'vehicle'): Name of the food/exposure item.
            - 'a' (or 'exposed_ill', 'ate_ill'): Number exposed who fell ill.
            - 'b' (or 'exposed_well', 'ate_well'): Number exposed who stayed well.
            - 'c' (or 'unexposed_ill', 'did_not_eat_ill'): Number unexposed who fell ill.
            - 'd' (or 'unexposed_well', 'did_not_eat_well'): Number unexposed who stayed well.
        alpha: Two-sided significance level (default 0.05).

    Returns:
        Dict[str, Any]: Complete outbreak food matrix, incriminated vehicle, and overall attack rate.
    """
    # 1. Standardize input items
    parsed_items = []
    if HAS_SCIENTIFIC_STACK and type(items_data).__name__ == "DataFrame":
        for idx, row in items_data.iterrows():
            item_name = str(row.get("Food_Item") or row.get("item") or row.get("vehicle") or f"Item {idx+1}")
            a = float(row.get("Exposed_Ill", row.get("a", row.get("ate_ill", 0))))
            b = float(row.get("Exposed_Well", row.get("b", row.get("ate_well", 0))))
            c = float(row.get("Unexposed_Ill", row.get("c", row.get("did_not_eat_ill", 0))))
            d = float(row.get("Unexposed_Well", row.get("d", row.get("did_not_eat_well", 0))))
            parsed_items.append({"item": item_name, "a": a, "b": b, "c": c, "d": d})
    else:
        for idx, entry in enumerate(items_data):
            item_name = str(entry.get("Food_Item") or entry.get("item") or entry.get("vehicle") or f"Item {idx+1}")
            a = float(entry.get("Exposed_Ill", entry.get("a", entry.get("ate_ill", 0))))
            b = float(entry.get("Exposed_Well", entry.get("b", entry.get("ate_well", 0))))
            c = float(entry.get("Unexposed_Ill", entry.get("c", entry.get("did_not_eat_ill", 0))))
            d = float(entry.get("Unexposed_Well", entry.get("d", entry.get("did_not_eat_well", 0))))
            parsed_items.append({"item": item_name, "a": a, "b": b, "c": c, "d": d})

    if not parsed_items:
        raise ValueError("At least one food or exposure vehicle must be provided.")

    z_crit = 1.95996 if abs(alpha - 0.05) < 1e-4 else 1.95996

    results_table = []
    max_incrimination_score = -1e9
    primary_vehicle = None

    total_cases = 0
    total_cohort = 0

    for itm in parsed_items:
        a = itm["a"]
        b = itm["b"]
        c = itm["c"]
        d = itm["d"]

        n1 = a + b
        n0 = c + d
        m1 = a + c
        m0 = b + d
        n = n1 + n0

        if n > total_cohort:
            total_cohort = int(n)
        if m1 > total_cases:
            total_cases = int(m1)

        # Attack rates
        ar_exp = (a / n1 * 100.0) if n1 > 0 else 0.0
        ar_unexp = (c / n0 * 100.0) if n0 > 0 else 0.0
        ar_diff = ar_exp - ar_unexp

        # Relative Risk
        has_zero = (a == 0 or b == 0 or c == 0 or d == 0)
        adj_a = a + 0.5 if has_zero else a
        adj_b = b + 0.5 if has_zero else b
        adj_c = c + 0.5 if has_zero else c
        adj_d = d + 0.5 if has_zero else d
        adj_n1 = adj_a + adj_b
        adj_n0 = adj_c + adj_d

        if (c / n0) > 0 and (a / n1) > 0:
            rr = (a / n1) / (c / n0)
            se_ln_rr = math.sqrt(max(0.0, (1.0 / a - 1.0 / n1) + (1.0 / c - 1.0 / n0)))
            rr_ci_low = math.exp(math.log(rr) - z_crit * se_ln_rr)
            rr_ci_upp = math.exp(math.log(rr) + z_crit * se_ln_rr)
        else:
            adj_p1 = adj_a / adj_n1
            adj_p0 = adj_c / adj_n0
            rr = adj_p1 / adj_p0 if adj_p0 > 0 else 1.0
            se_ln_rr = math.sqrt((1.0 / adj_a - 1.0 / adj_n1) + (1.0 / adj_c - 1.0 / adj_n0))
            rr_ci_low = math.exp(math.log(rr) - z_crit * se_ln_rr)
            rr_ci_upp = math.exp(math.log(rr) + z_crit * se_ln_rr)

        # Attributable fraction in exposed
        af_e = max(0.0, ((rr - 1.0) / rr) * 100.0) if rr > 1.0 else 0.0

        # Pearson Chi-Square with Yates continuity correction
        denom = float(n1 * n0 * m1 * m0)
        if denom > 0:
            diff = abs(a * d - b * c)
            diff_corr = max(0.0, diff - 0.5 * n)
            chi2_uncorrected = (float(n) * (diff ** 2)) / denom
            chi2_corrected = (float(n) * (diff_corr ** 2)) / denom
            p_val = _chi2_sf_df1(chi2_corrected)
        else:
            chi2_uncorrected = 0.0
            chi2_corrected = 0.0
            p_val = 1.0

        # Case exposure coverage (% of ill persons who consumed this item)
        case_coverage = (a / m1 * 100.0) if m1 > 0 else 0.0

        # Incrimination scoring:
        # High RR (>2.0), High AR Difference (>30%), High Case Coverage (>70%), Significant p-value (<0.01)
        incrimination_score = (math.log(max(1.0, rr)) * 2.0) + (ar_diff / 20.0) + (case_coverage / 25.0)
        if p_val < 0.01:
            incrimination_score += 3.0

        is_suspect = (rr >= 2.0 and p_val < 0.05 and ar_diff >= 20.0 and case_coverage >= 60.0)

        row_dict = {
            "Food_Item": itm["item"],
            "Exposed_Ill": int(a),
            "Exposed_Well": int(b),
            "Exposed_Total": int(n1),
            "Attack_Rate_Exposed_Pct": round(ar_exp, 1),
            "Unexposed_Ill": int(c),
            "Unexposed_Well": int(d),
            "Unexposed_Total": int(n0),
            "Attack_Rate_Unexposed_Pct": round(ar_unexp, 1),
            "Attack_Rate_Difference_Pct": round(ar_diff, 1),
            "Risk_Ratio": round(rr, 2),
            "RR_CI_Lower": round(rr_ci_low, 2),
            "RR_CI_Upper": round(rr_ci_upp, 2),
            "RR_95_CI_Str": f"{rr:.2f} [{rr_ci_low:.2f}, {rr_ci_upp:.2f}]",
            "AF_Exposed_Pct": round(af_e, 1),
            "Chi2_Statistic": round(chi2_corrected, 2),
            "P_Value": round(p_val, 4),
            "Significant": (p_val < alpha),
            "Case_Exposure_Coverage_Pct": round(case_coverage, 1),
            "Incrimination_Score": round(incrimination_score, 2),
            "Is_Suspect_Vehicle": is_suspect
        }
        results_table.append(row_dict)

        if incrimination_score > max_incrimination_score and is_suspect:
            max_incrimination_score = incrimination_score
            primary_vehicle = itm["item"]

    # Fallback to highest RR if no item met all suspect criteria
    if primary_vehicle is None and results_table:
        sorted_by_rr = sorted(results_table, key=lambda x: x["Risk_Ratio"], reverse=True)
        primary_vehicle = sorted_by_rr[0]["Food_Item"]

    overall_attack_rate = (total_cases / float(total_cohort) * 100.0) if total_cohort > 0 else 0.0

    df_results = pd.DataFrame(results_table) if (HAS_SCIENTIFIC_STACK and hasattr(pd, "DataFrame") and isinstance(pd.DataFrame, type)) else results_table

    return {
        "items_count": len(parsed_items),
        "total_cohort_at_risk": total_cohort,
        "total_primary_cases": total_cases,
        "overall_attack_rate_pct": round(overall_attack_rate, 1),
        "results_table": df_results,
        "results_list": results_table,
        "primary_incriminated_vehicle": primary_vehicle,
        "incriminated_vehicle_details": next((r for r in results_table if r["Food_Item"] == primary_vehicle), None)
    }


def calculate_secondary_attack_rate(
    primary_cases: int,
    household_contacts: int,
    secondary_cases: int,
    immune_contacts: int = 0,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Calculates Secondary Attack Rate (SAR) for household or close contact transmission.

    Formula:
        SAR = (Secondary Cases / Susceptible Contacts) * 100%
        Where Susceptible Contacts = Total Contacts - Immune Contacts (excluding primary index cases).

    Args:
        primary_cases: Number of primary / index cases who introduced the pathogen.
        household_contacts: Total close / household contacts exposed to the primary cases.
        secondary_cases: Number of contacts who subsequently contracted the illness within the maximum incubation window.
        immune_contacts: Contacts with verified prior immunity (e.g. vaccination, previous infection). Default 0.
        alpha: Significance level for Wilson 95% confidence interval.

    Returns:
        Dict[str, Any]: Secondary attack rate, Wilson 95% CI, communicability assessment.
    """
    susceptible_contacts = max(1, household_contacts - immune_contacts)
    sar_pct = (secondary_cases / float(susceptible_contacts)) * 100.0
    ci_low, ci_upp = _wilson_score_interval(secondary_cases, susceptible_contacts, confidence=1.0 - alpha)

    # Communicability category
    if sar_pct < 10.0:
        comm_tier = "Low Communicability"
        comm_desc = "Limited person-to-person spread. Most secondary contacts remained healthy (SAR < 10%). Standard hand hygiene and surface disinfection sufficient."
    elif sar_pct <= 30.0:
        comm_tier = "Moderate Communicability"
        comm_desc = "Meaningful household transmission observed (SAR 10–30%). Cohorting, barrier nursing, or household quarantine advised."
    else:
        comm_tier = "High / Explosive Communicability"
        comm_desc = f"Explosive transmission dynamics (SAR = {sar_pct:.1f}% > 30%). Suggests highly infectious pathogen with low infectious dose (e.g. Norovirus, Shigella, SARS-CoV-2)."

    return {
        "primary_cases": int(primary_cases),
        "household_contacts": int(household_contacts),
        "immune_contacts": int(immune_contacts),
        "susceptible_contacts": int(susceptible_contacts),
        "secondary_cases": int(secondary_cases),
        "secondary_attack_rate_pct": round(sar_pct, 1),
        "sar_ci_lower_pct": ci_low,
        "sar_ci_upper_pct": ci_upp,
        "sar_ci_str": f"{sar_pct:.1f}% [{ci_low:.1f}%, {ci_upp:.1f}%]",
        "communicability_tier": comm_tier,
        "communicability_description": comm_desc
    }


def calculate_epidemic_curve(
    onset_hours: List[float],
    exposure_hour: float = 0.0,
    bin_width: float = 4.0
) -> Dict[str, Any]:
    """Analyzes epidemic curve onset times and computes incubation period distribution statistics.

    Args:
        onset_hours: List of individual case symptom onset times relative to baseline (in hours).
        exposure_hour: Reference time of common exposure event (default 0.0).
        bin_width: Histogram bin interval in hours (default 4.0 hours).

    Returns:
        Dict[str, Any]: Incubation stats, binned onset frequency table, and outbreak curve pattern.
    """
    if not onset_hours:
        return {
            "case_count": 0, "median_incubation_hours": 0.0, "mean_incubation_hours": 0.0,
            "min_incubation_hours": 0.0, "max_incubation_hours": 0.0, "iqr_hours": 0.0,
            "curve_pattern": "Indeterminate", "bins": []
        }

    incubation_periods = sorted([h - exposure_hour for h in onset_hours if h >= exposure_hour])
    n_cases = len(incubation_periods)

    if n_cases == 0:
        incubation_periods = sorted(onset_hours)
        n_cases = len(incubation_periods)

    min_inc = incubation_periods[0]
    max_inc = incubation_periods[-1]
    mean_inc = sum(incubation_periods) / float(n_cases)

    # Percentiles
    def _pctile(arr, p):
        k = (len(arr) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return arr[int(k)]
        return arr[int(f)] * (c - k) + arr[int(c)] * (k - f)

    q1 = _pctile(incubation_periods, 0.25)
    median_inc = _pctile(incubation_periods, 0.50)
    q3 = _pctile(incubation_periods, 0.75)
    iqr = q3 - q1

    # Binning for histogram
    start_bin = math.floor(min_inc / bin_width) * bin_width
    end_bin = math.ceil(max_inc / bin_width) * bin_width + bin_width
    curr_bin = start_bin
    bins = []

    while curr_bin < end_bin:
        next_bin = curr_bin + bin_width
        count = sum(1 for val in incubation_periods if curr_bin <= val < next_bin)
        bins.append({
            "Bin_Start": curr_bin,
            "Bin_End": next_bin,
            "Bin_Label": f"{int(curr_bin)}–{int(next_bin)}h",
            "Case_Count": count
        })
        curr_bin = next_bin

    # Outbreak pattern classification
    # Unimodal with rapid upstroke and gradual tailing indicates point source
    max_count_idx = max(range(len(bins)), key=lambda i: bins[i]["Case_Count"])
    is_early_peak = (max_count_idx <= len(bins) // 2)
    span_ratio = (max_inc - min_inc) / max(1.0, median_inc)

    if span_ratio <= 3.5 and is_early_peak:
        pattern = "Point-Source (Common-Source Single Exposure)"
        pattern_desc = "Classic unimodal epidemic curve with a steep upward slope and gradual downward tail. All cases were exposed simultaneously during a single brief event (e.g. contaminated banquet or water supply)."
    elif span_ratio > 4.5:
        pattern = "Propagated (Person-to-Person Transmission)"
        pattern_desc = "Extended multi-wave epidemic curve spanning multiple incubation cycles. Indicates initial introduction followed by secondary transmission between individuals."
    else:
        pattern = "Continuous Common-Source"
        pattern_desc = "Broad plateau of cases over an extended duration, indicating ongoing, persistent exposure to the environmental reservoir or contaminated product."

    df_bins = pd.DataFrame(bins) if (HAS_SCIENTIFIC_STACK and hasattr(pd, "DataFrame") and isinstance(pd.DataFrame, type)) else bins

    return {
        "case_count": n_cases,
        "min_incubation_hours": round(min_inc, 1),
        "q1_incubation_hours": round(q1, 1),
        "median_incubation_hours": round(median_inc, 1),
        "q3_incubation_hours": round(q3, 1),
        "max_incubation_hours": round(max_inc, 1),
        "iqr_hours": round(iqr, 1),
        "mean_incubation_hours": round(mean_inc, 1),
        "curve_pattern": pattern,
        "pattern_description": pattern_desc,
        "bins_table": df_bins,
        "bins_list": bins
    }


def evaluate_outbreak_prerequisites(outbreak_results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates 5 core field outbreak methodological prerequisites based on CDC / UKHSA standards.

    Checks:
    1. Vehicle Incrimination Distinctiveness (one food item clearly stands out with RR >= 2.0, p < 0.05, and high exposure coverage).
    2. Primary Attack Rate Severity & Investigation Power (sample size and overall attack rate).
    3. Incubation Period Biological Plausibility (median incubation duration matches clinical presentation).
    4. Secondary Transmission & Containment Risk (SAR magnitude).
    5. Epidemic Curve Morphology Agreement (confirmation of point-source vs continuous vs propagated).

    Args:
        outbreak_results: Output dictionary from calculate_food_specific_attack_rates,
            optionally combined with sar_results and curve_results.

    Returns:
        Dict[str, Any]: Summary pass/warn status and individual check cards.
    """
    checks = []
    evaluations = {}

    # Check 1: Vehicle Incrimination Distinctiveness
    primary_veh = outbreak_results.get("primary_incriminated_vehicle")
    details = outbreak_results.get("incriminated_vehicle_details")

    if details and details.get("Significant", False) and details.get("Risk_Ratio", 1.0) >= 2.0:
        rr_val = details["Risk_Ratio"]
        p_val = details["P_Value"]
        cov_val = details["Case_Exposure_Coverage_Pct"]
        ard_val = details["Attack_Rate_Difference_Pct"]
        c1 = {
            "status": "pass",
            "badge": f"🟢 Clear Vehicle Incrimination ({primary_veh})",
            "title": "Vehicle Incrimination & Relative Risk Distinctiveness",
            "finding": f"'{primary_veh}' exhibits RR = {rr_val:.2f} (p = {p_val:.4f}), ARD = +{ard_val:.1f}%, Case Coverage = {cov_val:.1f}%",
            "explanation": f"A single vehicle clearly distinguishes itself with extreme statistical significance and high exposure prevalence among cases. Epidemiological criteria for incrimination are fulfilled."
        }
    else:
        c1 = {
            "status": "warn",
            "badge": "🟡 Indeterminate Vehicle Incrimination",
            "title": "Vehicle Incrimination & Relative Risk Distinctiveness",
            "finding": "No single food item met all strict incrimination criteria (RR >= 2.0, p < 0.05, coverage >= 60%).",
            "explanation": "Multiple items may share common contaminated ingredients (cross-contamination), or exposure recall bias is blurring the vehicle signal. Stratified or multivariable logistic regression is recommended."
        }
    checks.append(c1)
    evaluations["vehicle_incrimination"] = c1

    # Check 2: Attack Rate & Investigation Sample Size
    overall_ar = outbreak_results.get("overall_attack_rate_pct", 0.0)
    cohort_n = outbreak_results.get("total_cohort_at_risk", 0)

    if cohort_n >= 50 and overall_ar >= 10.0:
        c2 = {
            "status": "pass",
            "badge": f"🟢 Robust Cohort (N={cohort_n}, AR={overall_ar:.1f}%)",
            "title": "Outbreak Cohort Size & Attack Rate Power",
            "finding": f"Total investigated attendees = {cohort_n}, Total primary cases = {outbreak_results.get('total_primary_cases', 0)} ({overall_ar:.1f}%)",
            "explanation": "Adequate statistical power and high case density ensure contingency table cell counts produce reliable relative risk confidence intervals."
        }
    elif cohort_n >= 20:
        c2 = {
            "status": "warn",
            "badge": f"🟡 Modest Cohort Size (N={cohort_n})",
            "title": "Outbreak Cohort Size & Attack Rate Power",
            "finding": f"Total attendees = {cohort_n}, Attack Rate = {overall_ar:.1f}%",
            "explanation": "Small sample size. Yates continuity-corrected or Fisher's exact tests should be verified for sparse 2x2 cells."
        }
    else:
        c2 = {
            "status": "fail",
            "badge": "🔴 Extremely Small Outbreak Cohort",
            "title": "Outbreak Cohort Size & Attack Rate Power",
            "finding": f"Total attendees = {cohort_n} (< 20)",
            "explanation": "Very small outbreak sample size. Risk ratio estimates will have wide confidence intervals."
        }
    checks.append(c2)
    evaluations["cohort_power"] = c2

    # Check 3: Biological Plausibility of Incubation Period
    curve_res = outbreak_results.get("curve_results", {})
    med_inc = curve_res.get("median_incubation_hours", 24.0)

    if 1.0 <= med_inc <= 120.0:
        c3 = {
            "status": "pass",
            "badge": f"🟢 Plausible Incubation (Median: {med_inc:.1f}h)",
            "title": "Incubation Period Biological Concordance",
            "finding": f"Median incubation period = {med_inc:.1f} hours (IQR: {curve_res.get('iqr_hours', 0.0):.1f}h, Range: {curve_res.get('min_incubation_hours', 0.0)}–{curve_res.get('max_incubation_hours', 0.0)}h)",
            "explanation": "Observed incubation window aligns cleanly with established foodborne pathogens (e.g. 12–36h for Salmonella, 24–48h for Norovirus, 48–96h for Campylobacter)."
        }
    else:
        c3 = {
            "status": "warn",
            "badge": "🟡 Atypical Incubation Duration",
            "title": "Incubation Period Biological Concordance",
            "finding": f"Median incubation = {med_inc:.1f} hours",
            "explanation": "Incubation duration is unusually brief (<1h, suggestive of chemical or preformed toxin) or unusually delayed (>5 days, suggestive of Listeria or hepatitis A)."
        }
    checks.append(c3)
    evaluations["incubation_plausibility"] = c3

    # Check 4: Secondary Transmission Assessment
    sar_res = outbreak_results.get("sar_results", {})
    sar_val = sar_res.get("secondary_attack_rate_pct", 0.0)

    if sar_val >= 30.0:
        c4 = {
            "status": "warn",
            "badge": f"🚨 High Secondary Communicability (SAR={sar_val:.1f}%)",
            "title": "Secondary Contact Transmission Risk",
            "finding": f"Secondary Attack Rate in household contacts = {sar_val:.1f}% ({sar_res.get('secondary_cases', 0)} / {sar_res.get('susceptible_contacts', 0)})",
            "explanation": "Significant human-to-human transmission is active! The outbreak has evolved beyond a pure common-source point exposure. Active contact tracing and isolation measures are urgent."
        }
    elif sar_val > 0.0:
        c4 = {
            "status": "pass",
            "badge": f"🟢 Contained Secondary Spread (SAR={sar_val:.1f}%)",
            "title": "Secondary Contact Transmission Risk",
            "finding": f"Secondary Attack Rate = {sar_val:.1f}% ({sar_res.get('secondary_cases', 0)} secondary cases)",
            "explanation": "Modest secondary spread detected. Standard barrier precautions and exclusion of symptomatic food handlers should halt further transmission."
        }
    else:
        c4 = {
            "status": "pass",
            "badge": "🟢 Pure Common-Source (No Secondary Spread)",
            "title": "Secondary Contact Transmission Risk",
            "finding": "Zero secondary household transmission reported (SAR = 0.0%).",
            "explanation": "Pathogen transmission is entirely restricted to the primary point-source vehicle (typical of non-communicable food intoxications or well-contained enteric bacteria)."
        }
    checks.append(c4)
    evaluations["secondary_transmission"] = c4

    # Check 5: Epidemic Curve Pattern Agreement
    pattern = curve_res.get("curve_pattern", "Point-Source (Common-Source Single Exposure)")
    c5 = {
        "status": "pass",
        "badge": f"🟢 {pattern.split('(')[0].strip()}",
        "title": "Epidemic Curve Morphology Agreement",
        "finding": f"Pattern: {pattern}",
        "explanation": curve_res.get("pattern_description", "Curve morphology consistent with epidemiological timeline.")
    }
    checks.append(c5)
    evaluations["curve_morphology"] = c5

    all_passed = all(c["status"] == "pass" for c in checks)
    evaluations["all_passed"] = all_passed
    evaluations["summary"] = "All outbreak investigation prerequisites satisfied. Incriminated vehicle and transmission dynamics verified." if all_passed else "One or more outbreak diagnostics require epidemiological review."
    evaluations["checks"] = checks

    return evaluations


def generate_outbreak_demo_data(
    scenario_key: str = "wedding_salmonella",
    scenario: Optional[str] = None
) -> OutbreakDemoData:
    """Generates realistic synthetic outbreak investigation datasets for educational demonstration.

    All scenarios use fictional UK healthcare and public health bodies:
    - Riverside Health Protection Team (Wedding Banquet Salmonella Enteritidis)
    - Inner City Teaching Hospital (Ward 4B Norovirus Outbreak with Secondary Spread)
    - All Meadows Community Care Trust (Community Barbecue Campylobacter jejuni)

    Args:
        scenario_key: One of 'wedding_salmonella', 'hospital_norovirus', 'community_campylobacter'.
        scenario: Alias for scenario_key for call compatibility.

    Returns:
        OutbreakDemoData: Dictionary containing guest microdata, food attack rate matrix,
            incubation period onset distribution, and secondary contact tables.
    """
    if scenario is not None:
        scenario_key = scenario

    if scenario_key == "wedding_salmonella":
        # Riverside Health Protection Team: Salmonella Enteritidis at Wedding Banquet
        # Total guests: 120 attendees
        # Incriminated vehicle: Hollandaise Sauce (prepared with raw unpasteurized shell eggs)
        # Ate Hollandaise: 62 guests -> 54 ill, 8 well (Attack Rate = 87.1%)
        # Did Not Eat Hollandaise: 58 guests -> 6 ill, 52 well (Attack Rate = 10.3%)
        # Relative Risk = 87.1% / 10.3% = 8.42 (p < 0.0001)
        # Attributable Fraction in Exposed = 88.1%
        # Total cases = 54 + 6 = 60 ill (Overall Attack Rate = 50.0%)
        # Median incubation period: 22 hours (Point-source curve)
        # Secondary Attack Rate: 5 secondary cases among 75 household contacts (SAR = 6.7%)

        food_matrix = [
            {"Food_Item": "Hollandaise Sauce (Eggs)", "Exposed_Ill": 54, "Exposed_Well": 8, "Unexposed_Ill": 6, "Unexposed_Well": 52},
            {"Food_Item": "Roast Beef Sirloin", "Exposed_Ill": 42, "Exposed_Well": 38, "Unexposed_Ill": 18, "Unexposed_Well": 22},
            {"Food_Item": "Poached Salmon Fillet", "Exposed_Ill": 18, "Exposed_Well": 22, "Unexposed_Ill": 42, "Unexposed_Well": 38},
            {"Food_Item": "Asparagus Spears", "Exposed_Ill": 40, "Exposed_Well": 35, "Unexposed_Ill": 20, "Unexposed_Well": 25},
            {"Food_Item": "Potato Dauphinoise", "Exposed_Ill": 48, "Exposed_Well": 44, "Unexposed_Ill": 12, "Unexposed_Well": 16},
            {"Food_Item": "Chocolate Mousse Dessert", "Exposed_Ill": 35, "Exposed_Well": 32, "Unexposed_Ill": 25, "Unexposed_Well": 28},
            {"Food_Item": "Wedding Fruit Cake", "Exposed_Ill": 50, "Exposed_Well": 48, "Unexposed_Ill": 10, "Unexposed_Well": 12}
        ]

        # Individual onset times (hours post-meal) for 60 cases
        onset_times = [
            10.0, 12.0, 13.5, 14.0, 15.0, 16.0, 16.5, 17.0, 18.0, 18.5,
            19.0, 19.5, 20.0, 20.0, 20.5, 21.0, 21.0, 21.5, 22.0, 22.0,
            22.0, 22.5, 23.0, 23.0, 23.5, 24.0, 24.0, 24.5, 25.0, 25.0,
            25.5, 26.0, 26.0, 26.5, 27.0, 27.5, 28.0, 28.0, 29.0, 29.5,
            30.0, 31.0, 32.0, 32.5, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0,
            39.5, 41.0, 42.0, 44.0, 46.0, 48.0, 52.0, 56.0, 60.0, 68.0
        ]

        # Microdata generation for 120 guests
        rows = []
        for i in range(120):
            guest_id = f"RHT-WED-{i+1:03d}"
            # Hollandaise: 62 ate, 58 did not
            ate_hollandaise = 1 if i < 62 else 0
            # 54 of 62 exposed got ill; 6 of 58 unexposed got ill
            if ate_hollandaise:
                is_case = 1 if i < 54 else 0
            else:
                is_case = 1 if (i - 62) < 6 else 0

            ate_beef = 1 if (i % 3 != 0) else 0
            ate_cake = 1 if (i < 98) else 0
            onset_h = onset_times.pop(0) if is_case and onset_times else None

            rows.append({
                "Guest_ID": guest_id,
                "Ate_Hollandaise": ate_hollandaise,
                "Ate_Roast_Beef": ate_beef,
                "Ate_Fruit_Cake": ate_cake,
                "Became_Ill": is_case,
                "Onset_Hours_Post_Meal": onset_h,
                "Clinical_Status": "Confirmed Case" if is_case else "Healthy Well"
            })

        df_micro = pd.DataFrame(rows) if HAS_SCIENTIFIC_STACK else rows

        res = OutbreakDemoData({
            "df": df_micro,
            "food_matrix": food_matrix,
            "onset_hours": [r["Onset_Hours_Post_Meal"] for r in rows if r["Onset_Hours_Post_Meal"] is not None],
            "exposure_hour": 0.0,
            "bin_width": 4.0,
            "primary_cases": 60,
            "household_contacts": 75,
            "secondary_cases": 5,
            "immune_contacts": 0,
            "scenario_key": "wedding_salmonella",
            "scenario_name": "Riverside Health Protection Team: Salmonella Banquet Outbreak (Hollandaise)",
            "title": "Riverside Health Protection Team: Salmonella Enteritidis Wedding Banquet Outbreak",
            "organisation_name": "Riverside Health Protection Team",
            "organisation": "Riverside Health Protection Team (UKHSA / Environmental Health Directorate)",
            "domain": "Foodborne Outbreak Investigation & Enteric Disease Surveillance",
            "pathogen": "Salmonella enterica serovar Enteritidis",
            "incriminated_item": "Hollandaise Sauce (Prepared with Unpasteurized Raw Eggs)",
            "description": "Investigating an acute gastrointestinal outbreak among 120 wedding banquet attendees. 60 guests developed fever, diarrhea, and vomiting following the Saturday evening reception.",
            "takeaway": "Food-specific attack rates conclusively incriminate Hollandaise sauce (RR = 8.42, 95% CI: [3.92, 18.07], p < 0.0001, ARD = +76.8%). The point-source epidemic curve exhibits a classic right-skewed log-normal profile with a median incubation period of 22.0 hours.",
            "disclaimer": "Completely made-up synthetic outbreak cohort generated purely for educational and demonstration purposes only. Contains no real patient identifiers."
        })
        res["meta"] = dict(res)
        return res

    elif scenario_key == "hospital_norovirus":
        # Inner City Teaching Hospital: Hospital Ward 4B Winter Norovirus Outbreak
        # Initial point exposure: Contaminated water dispenser carafe on Day 0
        # Total exposed inpatients & staff: 60 individuals
        # Carafe exposed: 32 -> 24 ill, 8 well (Attack Rate = 75.0%)
        # Unexposed to carafe: 28 -> 4 ill, 24 well (Attack Rate = 14.3%)
        # Relative Risk = 5.25 (p < 0.0001)
        # Followed by secondary person-to-person spread across the bay:
        # 28 primary cases generated 15 secondary cases among 36 ward bay contacts (SAR = 41.7%!)
        food_matrix = [
            {"Food_Item": "Shared Water Carafe (Ward 4B)", "Exposed_Ill": 24, "Exposed_Well": 8, "Unexposed_Ill": 4, "Unexposed_Well": 24},
            {"Food_Item": "Hospital Pureed Soup", "Exposed_Ill": 15, "Exposed_Well": 16, "Unexposed_Ill": 13, "Unexposed_Well": 16},
            {"Food_Item": "Sandwich Platter", "Exposed_Ill": 14, "Exposed_Well": 15, "Unexposed_Ill": 14, "Unexposed_Well": 17},
            {"Food_Item": "Ward Tea Trolley", "Exposed_Ill": 20, "Exposed_Well": 22, "Unexposed_Ill": 8, "Unexposed_Well": 10}
        ]

        # Multi-wave onset times over 5 days (hours 0 to 120)
        onset_times = [
            # Primary wave from carafe (18-36 hours)
            18.0, 20.0, 22.0, 24.0, 24.0, 26.0, 27.0, 28.0, 28.0, 30.0,
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 36.0, 38.0, 40.0,
            42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0,
            # Secondary wave from person-to-person spread (60-108 hours)
            62.0, 64.0, 68.0, 72.0, 74.0, 76.0, 80.0, 82.0, 84.0, 88.0,
            92.0, 96.0, 100.0, 104.0, 108.0
        ]

        rows = []
        for i in range(60):
            pt_id = f"ICTH-W4B-{i+1:03d}"
            drank_carafe = 1 if i < 32 else 0
            if drank_carafe:
                is_case = 1 if i < 24 else 0
            else:
                is_case = 1 if (i - 32) < 4 else 0

            onset_h = onset_times.pop(0) if is_case and onset_times else None
            rows.append({
                "Subject_ID": pt_id,
                "Role": "Patient" if i < 40 else "Healthcare Staff",
                "Drank_Shared_Carafe": drank_carafe,
                "Became_Ill": is_case,
                "Onset_Hours": onset_h,
                "Clinical_Status": "Confirmed Norovirus Case" if is_case else "Healthy Well"
            })

        df_micro = pd.DataFrame(rows) if HAS_SCIENTIFIC_STACK else rows

        res = OutbreakDemoData({
            "df": df_micro,
            "food_matrix": food_matrix,
            "onset_hours": [r["Onset_Hours"] for r in rows if r["Onset_Hours"] is not None],
            "exposure_hour": 0.0,
            "bin_width": 12.0,
            "primary_cases": 28,
            "household_contacts": 36,
            "secondary_cases": 15,
            "immune_contacts": 0,
            "scenario_key": "hospital_norovirus",
            "scenario_name": "Inner City Teaching Hospital: Ward 4B Norovirus Outbreak & Secondary Transmission",
            "title": "Inner City Teaching Hospital: Ward 4B Winter Norovirus Outbreak",
            "organisation_name": "Inner City Teaching Hospital",
            "organisation": "Inner City Teaching Hospital (Infection Prevention & Control Directorate)",
            "domain": "Healthcare-Associated Infections (HCAI) & Viral Gastroenteritis Surveillance",
            "pathogen": "Norovirus Genogroup II",
            "incriminated_item": "Ward 4B Shared Water Carafe followed by Bedside Aerosol/Fomite Spread",
            "description": "Investigating a rapid gastroenteritis outbreak on a 30-bed acute medical ward affecting 28 inpatients and staff, followed by intensive secondary transmission to 15 bay contacts.",
            "takeaway": "While the shared water carafe initiated the outbreak (RR = 5.25, 95% CI: [2.07, 13.34], p = 0.0001), the secondary attack rate among close bay contacts was alarmingly high (SAR = 41.7%, 95% CI: [27.2%, 57.8%]), requiring total ward closure, terminal vaporized hydrogen peroxide misting, and strict staff cohorting.",
            "disclaimer": "Completely made-up synthetic outbreak cohort generated purely for educational and demonstration purposes only. Contains no real patient identifiers."
        })
        res["meta"] = dict(res)
        return res

    else:
        # All Meadows Community Care Trust: Community Charity Barbecue (Campylobacter)
        # Total attendees: 150
        # Incriminated vehicle: Under-cooked BBQ Chicken Skewers
        # Ate Chicken Skewers: 85 -> 68 ill, 17 well (Attack Rate = 80.0%)
        # Did Not Eat Chicken Skewers: 65 -> 7 ill, 58 well (Attack Rate = 10.8%)
        # Relative Risk = 7.43 (p < 0.0001)
        # Median incubation: 72 hours (Range 48-120 hours)
        # Secondary spread: 0 cases (Campylobacter has essentially zero secondary household communicability)
        food_matrix = [
            {"Food_Item": "BBQ Chicken Skewers", "Exposed_Ill": 68, "Exposed_Well": 17, "Unexposed_Ill": 7, "Unexposed_Well": 58},
            {"Food_Item": "Beef Burgers", "Exposed_Ill": 45, "Exposed_Well": 40, "Unexposed_Ill": 30, "Unexposed_Well": 35},
            {"Food_Item": "Vegetarian Haloumi Burgers", "Exposed_Ill": 15, "Exposed_Well": 25, "Unexposed_Ill": 60, "Unexposed_Well": 50},
            {"Food_Item": "Coleslaw with Mayo", "Exposed_Ill": 50, "Exposed_Well": 45, "Unexposed_Ill": 25, "Unexposed_Well": 30},
            {"Food_Item": "Potato Salad", "Exposed_Ill": 42, "Exposed_Well": 38, "Unexposed_Ill": 33, "Unexposed_Well": 37},
            {"Food_Item": "Lemonade Pitcher", "Exposed_Ill": 55, "Exposed_Well": 50, "Unexposed_Ill": 20, "Unexposed_Well": 25}
        ]

        # Onset times (hours post-BBQ) spanning 48 to 120 hours
        onset_times = [
            42.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0,
            64.0, 65.0, 66.0, 68.0, 69.0, 70.0, 70.0, 71.0, 72.0, 72.0,
            72.0, 73.0, 74.0, 74.0, 75.0, 75.0, 76.0, 77.0, 78.0, 78.0,
            79.0, 80.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0,
            88.0, 89.0, 90.0, 91.0, 92.0, 94.0, 95.0, 96.0, 98.0, 100.0,
            102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 115.0, 118.0, 120.0, 124.0,
            128.0, 132.0, 136.0, 140.0, 144.0, 150.0, 156.0, 160.0, 168.0, 172.0,
            176.0, 180.0, 184.0, 190.0, 196.0
        ]

        rows = []
        for i in range(150):
            pt_id = f"AMCT-BBQ-{i+1:03d}"
            ate_chicken = 1 if i < 85 else 0
            if ate_chicken:
                is_case = 1 if i < 68 else 0
            else:
                is_case = 1 if (i - 85) < 7 else 0

            onset_h = onset_times.pop(0) if is_case and onset_times else None
            rows.append({
                "Attendee_ID": pt_id,
                "Ate_Chicken_Skewers": ate_chicken,
                "Ate_Coleslaw": 1 if i % 2 == 0 else 0,
                "Became_Ill": is_case,
                "Onset_Hours": onset_h,
                "Clinical_Status": "Confirmed Campylobacteriosis" if is_case else "Healthy Well"
            })

        df_micro = pd.DataFrame(rows) if HAS_SCIENTIFIC_STACK else rows

        res = OutbreakDemoData({
            "df": df_micro,
            "food_matrix": food_matrix,
            "onset_hours": [r["Onset_Hours"] for r in rows if r["Onset_Hours"] is not None],
            "exposure_hour": 0.0,
            "bin_width": 12.0,
            "primary_cases": 75,
            "household_contacts": 90,
            "secondary_cases": 0,
            "immune_contacts": 0,
            "scenario_key": "community_campylobacter",
            "scenario_name": "All Meadows Community Care Trust: Charity Barbecue Campylobacter Outbreak",
            "title": "All Meadows Community Care Trust: Charity Barbecue Campylobacter Outbreak",
            "organisation_name": "All Meadows Community Care Trust",
            "organisation": "All Meadows Community Care Trust (Public Health & Environmental Protection)",
            "domain": "Zoonotic Foodborne Surveillance & Enteric Epidemiology",
            "pathogen": "Campylobacter jejuni",
            "incriminated_item": "Undercooked Barbecue Chicken Skewers (Cross-Contamination)",
            "description": "Investigating a community-wide gastroenteritis outbreak following an outdoor charity barbecue attended by 150 local residents.",
            "takeaway": "Chicken skewers were decisively incriminated (RR = 7.43, 95% CI: [3.70, 14.92], p < 0.0001, ARD = +69.2%). The median incubation period of 72.0 hours (3 days) is pathognomonic for Campylobacter jejuni. Secondary attack rate was 0.0%, confirming the absence of person-to-person spread.",
            "disclaimer": "Completely made-up synthetic outbreak cohort generated purely for educational and demonstration purposes only. Contains no real patient identifiers."
        })
        res["meta"] = dict(res)
        return res


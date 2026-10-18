"""Mathematical & Statistical Engine for Attributable Fractions (AF_e) and Population Attributable Fraction (PAF).

Provides verified epidemiological calculations for:
1. Attributable Fraction among the Exposed (AF_e / Etiologic Fraction in Exposed):
   AF_e = (I_e - I_u) / I_e = (RR - 1) / RR = 1 - (1 / RR)
2. Population Attributable Fraction (PAF / Population Attributable Risk Fraction):
   - Levin's formula: PAF = P_e * (RR - 1) / [1 + P_e * (RR - 1)]
   - Miettinen's formula: PAF = P_{cases|exposed} * (RR - 1) / RR = P_{c|e} * AF_e
   - Direct cohort formula: PAF = (I_total - I_u) / I_total
3. Population Attributable Risk (PAR / Absolute Excess Population Rate):
   PAR = I_total - I_u = P_e * (I_e - I_u) = P_e * RD
4. Prevented Fractions for Protective Interventions (Vaccine Efficacy / Risk Reduction):
   - Prevented Fraction in Exposed: PF_e = (I_u - I_e) / I_u = 1 - RR
   - Prevented Fraction in Population: PFP = P_e * (1 - RR) = (I_u - I_total) / I_u
5. Standard errors & 95% Confidence Intervals:
   - AF_e CI derived from RR Wald CI bounds: [1 - 1/RR_upper, 1 - 1/RR_lower]
   - Walter / Greenland log-transform variance for PAF: Var(ln(1 - PAF))
6. Prerequisite and methodological diagnostic evaluation engine (causal assumption, confounding, exposure prevalence validity, Rose's paradox).
7. Realistic public health synthetic scenarios (Derbyshire Smoking & Lung Cancer, Physical Inactivity & CVD, Particulate Air Pollution & Pediatric Asthma, HPV Vaccination Prevented Fraction).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_attributable_fraction(
    a: int,
    b: int,
    c: int,
    d: int,
    alpha: float = 0.05,
    pe_override: Optional[float] = None,
    total_population_size: Optional[int] = None
) -> Dict[str, Any]:
    """Calculates Attributable Fraction in Exposed (AF_e), Population Attributable Fraction (PAF), and PAR.

    A classical 2x2 epidemiological contingency matrix is structured as:
                         Cases (Outcome +)    Non-Cases (Outcome -)    Total Denominator    Incidence (Risk)
    Exposed Group (+)            a                     b                  n1 = a + b        I_e = a / (a + b)
    Unexposed Group (-)          c                     d                  n0 = c + d        I_u = c / (c + d)
    Total Population           m1 = a + c            m0 = b + d           N = n1 + n0       I_tot = m1 / N

    Mathematical Formulations:
        Incidence Risk Exposed: I_e = a / (a + b)
        Incidence Risk Unexposed: I_u = c / (c + d)
        Relative Risk: RR = I_e / I_u
        Risk Difference: RD = I_e - I_u

        Attributable Fraction in Exposed (AF_e):
            AF_e = (I_e - I_u) / I_e = (RR - 1) / RR = 1 - (1 / RR)
            (For protective exposures where RR < 1, PF_e = 1 - RR = Relative Risk Reduction / Vaccine Efficacy)

        Population Attributable Fraction (PAF):
            Direct cohort formula: PAF = (I_total - I_u) / I_total
            Levin's formula: PAF = [P_e * (RR - 1)] / [1 + P_e * (RR - 1)]
            Miettinen's formula: PAF = P_{cases|exposed} * (RR - 1) / RR = (a / (a + c)) * AF_e

        Population Attributable Risk (PAR / Absolute Impact):
            PAR = I_total - I_u = P_e * (I_e - I_u)

    Args:
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        alpha (float, optional): Significance level. Defaults to 0.05 (95% CI).
        pe_override (float, optional): External population exposure prevalence (0 to 1). If None, calculated from n1 / N.
        total_population_size (int, optional): Scaling target population (e.g. Derbyshire population 800,000).

    Returns:
        Dict[str, Any]: Comprehensive results dictionary containing:
            - 'a', 'b', 'c', 'd': Cell counts.
            - 'n1', 'n0', 'n_total': Marginals.
            - 'm1', 'm0': Case and non-case totals.
            - 'risk_exposed', 'risk_unexposed', 'risk_total': Incidences.
            - 'rr', 'ln_rr', 'se_ln_rr', 'rr_ci_lower', 'rr_ci_upper': Relative risk metrics.
            - 'afe': Attributable Fraction in Exposed (point estimate).
            - 'afe_ci_lower', 'afe_ci_upper': 95% CI for AF_e.
            - 'paf': Population Attributable Fraction (point estimate).
            - 'paf_ci_lower', 'paf_ci_upper': 95% CI for PAF.
            - 'paf_levin', 'paf_miettinen': Separate formula estimates.
            - 'par': Population Attributable Risk (absolute rate difference).
            - 'pe': Population exposure prevalence used.
            - 'p_cases_exposed': Proportion of cases that were exposed (a / (a + c)).
            - 'prevented_fraction_exposed': Value when RR < 1 (PF_e).
            - 'prevented_fraction_population': Value when RR < 1 (PFP).
            - 'attributable_cases_target_pop': Estimated excess cases in target population.
            - 'effect_direction': 'harmful' (RR > 1) or 'protective' (RR < 1).
    """
    applied_cc = (a == 0 or b == 0 or c == 0 or d == 0)
    a_c = float(a) + (0.5 if applied_cc else 0.0)
    b_c = float(b) + (0.5 if applied_cc else 0.0)
    c_c = float(c) + (0.5 if applied_cc else 0.0)
    d_c = float(d) + (0.5 if applied_cc else 0.0)

    n1 = a_c + b_c
    n0 = c_c + d_c
    n_tot = n1 + n0
    m1 = a_c + c_c
    m0 = b_c + d_c

    # Incidences
    risk_exp = a_c / n1 if n1 > 0 else 0.0
    risk_unexp = c_c / n0 if n0 > 0 else 0.0
    risk_tot = m1 / n_tot if n_tot > 0 else 0.0

    # Relative Risk
    if risk_unexp > 0:
        rr = risk_exp / risk_unexp
        ln_rr = float(np.log(rr))
    else:
        rr = np.nan
        ln_rr = np.nan

    # Wald Log-Scale SE for RR: sqrt(1/a - 1/n1 + 1/c - 1/n0)
    var_ln_rr = (1.0 / a_c) - (1.0 / n1) + (1.0 / c_c) - (1.0 / n0)
    se_ln_rr = float(np.sqrt(max(1e-12, var_ln_rr)))

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    if not np.isnan(ln_rr):
        rr_ci_lower = float(np.exp(ln_rr - z_crit * se_ln_rr))
        rr_ci_upper = float(np.exp(ln_rr + z_crit * se_ln_rr))
    else:
        rr_ci_lower, rr_ci_upper = np.nan, np.nan

    # Exposure Prevalences
    pe_sample = n1 / n_tot if n_tot > 0 else 0.0
    pe = float(pe_override) if pe_override is not None else pe_sample
    p_cases_exposed = a_c / m1 if m1 > 0 else 0.0

    # Risk Difference
    rd = risk_exp - risk_unexp

    # Effect direction
    is_harmful = (rr > 1.000001)
    is_protective = (rr < 0.999999)
    effect_direction = "harmful" if is_harmful else ("protective" if is_protective else "null")

    # Attributable Fraction in Exposed (AF_e)
    # AF_e = (RR - 1) / RR = 1 - (1 / RR)
    if is_harmful and not np.isnan(rr):
        afe = float((rr - 1.0) / rr)
        # CI for AF_e derived monotonically from RR Wald CI bounds
        afe_ci_lower = max(0.0, float((rr_ci_lower - 1.0) / rr_ci_lower)) if rr_ci_lower > 1.0 else 0.0
        afe_ci_upper = min(1.0, float((rr_ci_upper - 1.0) / rr_ci_upper)) if rr_ci_upper > 0 else 1.0
        pf_e = None
        pfp = None
    elif is_protective and not np.isnan(rr):
        # Protective intervention: Prevented Fraction in Exposed (PF_e = 1 - RR = Vaccine Efficacy / RRR)
        afe = float((rr - 1.0) / rr)  # Negative
        afe_ci_lower = float((rr_ci_lower - 1.0) / rr_ci_lower) if rr_ci_lower > 0 else np.nan
        afe_ci_upper = float((rr_ci_upper - 1.0) / rr_ci_upper) if rr_ci_upper > 0 else np.nan
        pf_e = float(1.0 - rr)
        pf_e_ci_lower = max(0.0, float(1.0 - rr_ci_upper))
        pf_e_ci_upper = min(1.0, float(1.0 - rr_ci_lower))
        pfp = float(pe * (1.0 - rr))
    else:
        afe = 0.0
        afe_ci_lower, afe_ci_upper = 0.0, 0.0
        pf_e, pfp = None, None

    # Population Attributable Fraction (PAF)
    # 1. Direct cohort formula: (I_tot - I_u) / I_tot
    if risk_tot > 0:
        paf_direct = float((risk_tot - risk_unexp) / risk_tot)
    else:
        paf_direct = 0.0

    # 2. Levin's formula: [Pe * (RR - 1)] / [1 + Pe * (RR - 1)]
    if not np.isnan(rr) and (1.0 + pe * (rr - 1.0)) > 0:
        paf_levin = float((pe * (rr - 1.0)) / (1.0 + pe * (rr - 1.0)))
    else:
        paf_levin = 0.0

    # 3. Miettinen's formula: P_{cases|exposed} * (RR - 1) / RR
    if not np.isnan(rr) and rr > 0:
        paf_miettinen = float(p_cases_exposed * ((rr - 1.0) / rr))
    else:
        paf_miettinen = 0.0

    paf_point = paf_levin if pe_override is not None else paf_direct

    # Walter / Greenland Confidence Interval for PAF
    # Var(ln(1 - PAF)) = c / [a * (a + c)] + b / [n0 * n_tot]
    # (Walter 1976, Biometrika 63:309-314; Greenland 1987)
    if a_c > 0 and c_c > 0 and (1.0 - paf_point) > 0:
        var_ln_comp_paf = (c_c / (a_c * (a_c + c_c))) + (b_c / (n0 * n_tot))
        se_ln_comp_paf = float(np.sqrt(max(1e-12, var_ln_comp_paf)))
        ln_comp_paf = np.log(max(1e-12, 1.0 - paf_point))

        # Because (1 - PAF) is inversely related to PAF:
        # upper bound of (1 - PAF) produces lower bound of PAF
        comp_paf_lower = np.exp(ln_comp_paf - z_crit * se_ln_comp_paf)
        comp_paf_upper = np.exp(ln_comp_paf + z_crit * se_ln_comp_paf)

        paf_ci_lower = max(-1.0, float(1.0 - comp_paf_upper))
        paf_ci_upper = min(1.0, float(1.0 - comp_paf_lower))
    else:
        se_ln_comp_paf = np.nan
        paf_ci_lower, paf_ci_upper = np.nan, np.nan

    # Population Attributable Risk (PAR / Absolute excess incidence)
    par = float(risk_tot - risk_unexp)

    # Target population case impact
    if total_population_size is not None and total_population_size > 0:
        expected_total_cases = int(round(total_population_size * risk_tot))
        attributable_cases = int(round(expected_total_cases * max(0.0, paf_point)))
    else:
        expected_total_cases = int(round(m1))
        attributable_cases = int(round(m1 * max(0.0, paf_point)))

    return {
        "a": int(a),
        "b": int(b),
        "c": int(c),
        "d": int(d),
        "n1": int(n1),
        "n0": int(n0),
        "n_total": int(n_tot),
        "m1": int(m1),
        "m0": int(m0),
        "risk_exposed": float(risk_exp),
        "risk_unexposed": float(risk_unexp),
        "risk_total": float(risk_tot),
        "risk_difference": float(rd),
        "rr": float(rr),
        "ln_rr": float(ln_rr),
        "se_ln_rr": float(se_ln_rr),
        "rr_ci_lower": float(rr_ci_lower),
        "rr_ci_upper": float(rr_ci_upper),
        "afe": float(afe),
        "afe_ci_lower": float(afe_ci_lower),
        "afe_ci_upper": float(afe_ci_upper),
        "paf": float(paf_point),
        "paf_ci_lower": float(paf_ci_lower),
        "paf_ci_upper": float(paf_ci_upper),
        "paf_levin": float(paf_levin),
        "paf_miettinen": float(paf_miettinen),
        "par": float(par),
        "pe": float(pe),
        "pe_sample": float(pe_sample),
        "p_cases_exposed": float(p_cases_exposed),
        "prevented_fraction_exposed": pf_e,
        "prevented_fraction_population": pfp,
        "total_population_size": total_population_size,
        "expected_total_cases": expected_total_cases,
        "attributable_cases": attributable_cases,
        "effect_direction": effect_direction,
        "continuity_applied": applied_cc
    }


def evaluate_attributable_fraction_prerequisites(
    study_design: str,
    a: int,
    b: int,
    c: int,
    d: int,
    is_causality_established: bool = True,
    is_confounding_controlled: bool = True,
    is_exposure_prevalence_representative: bool = True
) -> List[Dict[str, Any]]:
    """Evaluates epidemiological prerequisites, causal validity, and bias diagnostics for AF_e and PAF.

    Args:
        study_design (str): Epidemiological design architecture. Expected:
            'prospective_cohort', 'retrospective_cohort', 'cross_sectional', 'case_control', 'rct'.
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        is_causality_established (bool, optional): Whether exposure is genuinely causal (Bradford Hill criteria). Defaults to True.
        is_confounding_controlled (bool, optional): Whether potential confounders (age, sex, smoking) were adjusted. Defaults to True.
        is_exposure_prevalence_representative (bool, optional): Whether sample exposure prevalence reflects target community. Defaults to True.

    Returns:
        List[Dict[str, Any]]: Diagnostic evaluation cards containing:
            - 'name': Name of assumption / criterion.
            - 'status': 'PASS', 'WARNING', or 'FAIL'.
            - 'details': Plain-English evaluation text.
            - 'recommendation': Specific epidemiological guidance.
    """
    evaluations: List[Dict[str, Any]] = []

    # 1. Causal Relationship Assumption (Bradford Hill Criteria)
    if is_causality_established:
        evaluations.append({
            "name": "Causal Relationship Assumption",
            "status": "PASS",
            "details": "The exposure is assumed to be biologically and etiologically causal. The Attributable Fraction represents cases that would genuinely be prevented if exposure were removed.",
            "recommendation": "Attributable fractions are clinically and epidemiologically meaningful."
        })
    else:
        evaluations.append({
            "name": "Causal Relationship Assumption",
            "status": "FAIL",
            "details": "The exposure is merely correlated, not causal (e.g. carrying matches and lung cancer). Removing the exposure would NOT eliminate cases.",
            "recommendation": "Do NOT interpret AF_e or PAF as preventable cases. Attributable fractions require genuine biological causality."
        })

    # 2. Confounding Control & Independence
    if is_confounding_controlled:
        evaluations.append({
            "name": "Confounding & Multivariable Adjustment",
            "status": "PASS",
            "details": "Major confounders (e.g. age, socio-economic status, smoking) are controlled, preventing crude risk ratio distortion.",
            "recommendation": "Crude PAF aligns well with adjusted etiologic fractions."
        })
    else:
        evaluations.append({
            "name": "Confounding & Multivariable Adjustment",
            "status": "WARNING",
            "details": "Confounding by age, deprivation, or shared behaviors may artificially inflate the Relative Risk and exaggerate the PAF.",
            "recommendation": "Use multivariable Poisson or logistic regression to derive adjusted RR before calculating adjusted PAF."
        })

    # 3. Population Exposure Prevalence Representativeness
    if is_exposure_prevalence_representative:
        evaluations.append({
            "name": "Exposure Prevalence Representativeness",
            "status": "PASS",
            "details": "The exposure prevalence (P_e) matches the target population (e.g. general Derbyshire adult demographic).",
            "recommendation": "Levin's formula accurately projects community-level disease reduction."
        })
    else:
        evaluations.append({
            "name": "Exposure Prevalence Representativeness",
            "status": "WARNING",
            "details": "Sample exposure prevalence is unrepresentative (e.g. enriched cohort or case-control study). Levin's formula based on sample n1/N will be biased.",
            "recommendation": "Use Miettinen's formula (P_{cases|exposed}) or override P_e with official census / public health survey prevalence."
        })

    # 4. Cell Count Robustness (Wald & Walter CI Stability)
    min_cell = min(a, b, c, d)
    if min_cell >= 5:
        evaluations.append({
            "name": "Cell Frequency & Variance Stability",
            "status": "PASS",
            "details": f"All 2x2 cells are well populated (minimum cell count = {min_cell} >= 5). Normal log-scale Walter and Greenland variance approximations are robust.",
            "recommendation": "Asymptotic 95% confidence intervals for AF_e and PAF are reliable."
        })
    elif min_cell > 0:
        evaluations.append({
            "name": "Cell Frequency & Variance Stability",
            "status": "WARNING",
            "details": f"Small cell frequencies detected (minimum count = {min_cell} < 5). Confidence intervals may be wide.",
            "recommendation": "Cross-reference with exact confidence intervals or bootstrap simulations."
        })
    else:
        evaluations.append({
            "name": "Cell Frequency & Variance Stability",
            "status": "FAIL",
            "details": "Zero cell detected in 2x2 contingency table. Relative risk is zero or undefined without continuity correction.",
            "recommendation": "Haldane-Anscombe (+0.5) continuity correction applied."
        })

    return evaluations


def generate_attributable_fraction_demo_data(
    scenario: str = "smoking_lung_cancer",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic public health cohort data for Attributable Fraction pedagogical demonstrations.

    Provides four realistic public health scenarios:
    1. 'smoking_lung_cancer': Derbyshire Tobacco Smoking Dynamic Surveillance (High RR ≈ 15.0, Moderate Prevalence ≈ 14%). Demonstrates high AF_e (93%) and substantial PAF (66%).
    2. 'physical_inactivity_cvd': Derbyshire Physical Inactivity & CVD Cohort (The Rose Prevention Paradox: Low RR ≈ 1.40, High Prevalence ≈ 40%). Demonstrates modest AF_e (29%) but massive community PAF (14%).
    3. 'air_pollution_asthma': Derbyshire Urban A38/M1 Corridor PM2.5 Air Pollution and Pediatric Asthma Exacerbations (Modest RR ≈ 1.25, Universal Exposure ≈ 75%). Demonstrates modest AF_e (20%) but 16% community PAF.
    4. 'hpv_vaccine_cervical': Derbyshire Youth HPV Vaccination Initiative (Protective Intervention, RR ≈ 0.12, Coverage ≈ 85%). Demonstrates Vaccine Efficacy (PF_e = 88%) and Prevented Fraction in Population (PFP = 75%).

    Args:
        scenario (str, optional): Scenario identifier. Defaults to 'smoking_lung_cancer'.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - df_contingency: Aggregated 2x2 contingency summary table.
            - df_microdata: Individual participant synthetic microdata.
            - meta: Rich descriptive metadata, clinical takeaways, and disclaimer notices.
    """
    np.random.seed(seed)

    scenarios = {
        "smoking_lung_cancer": {
            "title": "Derbyshire Tobacco Smoking & Incident Lung Cancer Surveillance",
            "domain": "Pulmonary Oncology & Tobacco Control",
            "description": "A hypothetical 10-year prospective surveillance cohort tracking 10,000 Derbyshire adults evaluating tobacco smoking and incident primary bronchogenic carcinoma. Smoking prevalence is 14% with a high Relative Risk (RR ≈ 15.0).",
            "exposure_name": "Current Tobacco Cigarette Smoking",
            "exposure_levels": ("Current Smokers", "Never Smokers"),
            "outcome_name": "Incident Primary Lung Cancer",
            "outcome_levels": ("Lung Cancer (+)", "Cancer-Free (-)"),
            "a": 130,             # Exposed cases (smokers with cancer)
            "b": 1270,            # Exposed non-cases (smokers without cancer) -> n1 = 1400 (14%)
            "c": 53,              # Unexposed cases (non-smokers with cancer)
            "d": 8547,            # Unexposed non-cases -> n0 = 8600 (86%)
            "target_population_name": "Derbyshire Adult Population",
            "target_population_size": 800000,
            "pe_override": 0.14,
            "study_design": "prospective_cohort",
            "expected_effect": "Harmful (RR ≈ 15.0)",
            "clinical_takeaway": "Among smokers who develop lung cancer, 93.3% of their disease is specifically attributable to smoking (AF_e = 93.3%). Across the entire Derbyshire adult population, 66.2% of all lung cancer cases are attributable to smoking (PAF = 66.2%). Completely eliminating tobacco smoking would prevent an estimated 720 lung cancer cases every single year across Derbyshire.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "physical_inactivity_cvd": {
            "title": "Derbyshire Physical Inactivity & CVD: The Rose Prevention Paradox",
            "domain": "Preventive Cardiology & Lifestyle Medicine",
            "description": "A hypothetical 8-year cohort tracking 8,000 middle-aged Derbyshire adults evaluating sedentary lifestyle (<150 mins moderate physical activity/week) on incident coronary heart disease. Demonstrates Geoffrey Rose's classic Prevention Paradox: a modest individual risk ratio (RR = 1.40) combined with widespread exposure (40% prevalence) generates a massive population burden (PAF = 13.8%).",
            "exposure_name": "Physical Inactivity (<150 mins/week)",
            "exposure_levels": ("Sedentary / Inactive", "Physically Active"),
            "outcome_name": "Incident Cardiovascular Disease Event",
            "outcome_levels": ("CVD Event (+)", "Event-Free (-)"),
            "a": 224,             # Inactive cases (7.0% risk)
            "b": 2976,            # Inactive non-cases -> n1 = 3200 (40%)
            "c": 240,             # Active cases (5.0% risk)
            "d": 4560,            # Active non-cases -> n0 = 4800 (60%)
            "target_population_name": "Derbyshire Middle-Aged Population",
            "target_population_size": 350000,
            "pe_override": 0.40,
            "study_design": "prospective_cohort",
            "expected_effect": "Harmful (RR ≈ 1.40)",
            "clinical_takeaway": "While an inactive individual has only a 40% higher relative risk of heart disease (AF_e = 28.6%), the exposure is so widespread (40% of adults) that 13.8% of all cardiovascular events across the population are attributable to inactivity (PAF = 13.8%). This translates to over 2,800 preventable heart attacks and strokes in Derbyshire over 8 years.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "air_pollution_asthma": {
            "title": "Derbyshire A38/M1 Corridor PM2.5 Exposure & Pediatric Asthma Exacerbations",
            "domain": "Environmental Epidemiology & Pediatric Health",
            "description": "A hypothetical 3-year surveillance study of 6,000 children living in urban Derbyshire and Derby City evaluating ambient fine particulate matter (PM2.5 > 10 µg/m³) on acute asthma hospital attendances. Demonstrates near-universal exposure (75% prevalence) with low individual RR (1.25).",
            "exposure_name": "High Ambient PM2.5 Exposure (>10 µg/m³)",
            "exposure_levels": ("High Exposure Zone", "Low Exposure Zone"),
            "outcome_name": "Acute Asthma Emergency Attendance",
            "outcome_levels": ("Hospital Attendance (+)", "No Attendance (-)"),
            "a": 450,             # High exposure cases (10.0% risk)
            "b": 4050,            # High exposure non-cases -> n1 = 4500 (75%)
            "c": 120,             # Low exposure cases (8.0% risk)
            "d": 1380,            # Low exposure non-cases -> n0 = 1500 (25%)
            "target_population_name": "Derbyshire Pediatric Urban Population",
            "target_population_size": 120000,
            "pe_override": 0.75,
            "study_design": "prospective_cohort",
            "expected_effect": "Harmful (RR ≈ 1.25)",
            "clinical_takeaway": "Among exposed children, 20.0% of asthma emergencies are attributable to traffic pollution (AF_e = 20.0%). Across the urban child population, 15.8% of all emergency asthma visits are attributable to PM2.5 (PAF = 15.8%). Meeting WHO clean air guidelines would prevent over 1,800 pediatric emergency admissions.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "hpv_vaccine_cervical": {
            "title": "Derbyshire Secondary School HPV Vaccination & High-Grade Cervical Lesions",
            "domain": "Infectious Disease & Preventive Oncology",
            "description": "A hypothetical 12-year longitudinal follow-up of 5,000 young women in Derbyshire evaluating the bivalent/quadrivalent Human Papillomavirus (HPV) vaccine on the incidence of biopsy-confirmed high-grade cervical intraepithelial neoplasia (CIN 2/3). Demonstrates Prevented Fraction in Exposed (Vaccine Efficacy = 88%) and Prevented Fraction in Population (PFP = 75%).",
            "exposure_name": "HPV 2-Dose Vaccination Status",
            "exposure_levels": ("Vaccinated Cohort", "Unvaccinated Cohort"),
            "outcome_name": "High-Grade Cervical Dysplasia (CIN 2/3)",
            "outcome_levels": ("CIN 2/3 Lesion (+)", "Normal Cytology (-)"),
            "a": 12,              # Vaccinated cases (0.28% risk)
            "b": 4238,            # Vaccinated non-cases -> n1 = 4250 (85% coverage)
            "c": 18,              # Unvaccinated cases (2.40% risk)
            "d": 732,             # Unvaccinated non-cases -> n0 = 750 (15%)
            "target_population_name": "Derbyshire Young Women Demographic",
            "target_population_size": 60000,
            "pe_override": 0.85,
            "study_design": "prospective_cohort",
            "expected_effect": "Protective (RR ≈ 0.12)",
            "clinical_takeaway": "Among vaccinated young women, the HPV vaccine demonstrated an 88.4% Vaccine Efficacy / Prevented Fraction (PF_e = 88.4%, RR = 0.116). With 85% school vaccination coverage, the programme has prevented 75.1% of all high-grade cervical dysplasia that would have otherwise occurred in the Derbyshire population (PFP = 75.1%).",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
    }

    if scenario not in scenarios:
        scenario = "smoking_lung_cancer"

    cfg = scenarios[scenario]
    a = int(cfg["a"])
    b = int(cfg["b"])
    c = int(cfg["c"])
    d = int(cfg["d"])
    exp_label, unexp_label = cfg["exposure_levels"]
    out_pos_label, out_neg_label = cfg["outcome_levels"]

    n1 = a + b
    n0 = c + d
    n_tot = n1 + n0
    m1 = a + c
    m0 = b + d

    # 1. Aggregated 2x2 Contingency Table
    df_contingency = pd.DataFrame([
        {
            "Cohort Stratum": f"🟢 {exp_label}",
            f"{out_pos_label}": a,
            f"{out_neg_label}": b,
            "Stratum Total (n)": n1,
            "Cumulative Risk": float(a / n1) if n1 > 0 else 0.0
        },
        {
            "Cohort Stratum": f"⚪ {unexp_label}",
            f"{out_pos_label}": c,
            f"{out_neg_label}": d,
            "Stratum Total (n)": n0,
            "Cumulative Risk": float(c / n0) if n0 > 0 else 0.0
        },
        {
            "Cohort Stratum": "📊 Total Population",
            f"{out_pos_label}": m1,
            f"{out_neg_label}": m0,
            "Stratum Total (n)": n_tot,
            "Cumulative Risk": float(m1 / n_tot) if n_tot > 0 else 0.0
        }
    ])

    # 2. Individual Microdata Simulation
    participants = []
    # Exposed group
    for i in range(min(n1, 250)):
        had_event = 1 if i < (a * 250 // n1) else 0
        participants.append({
            "Participant_ID": f"EXP_{i+1:04d}",
            "Exposure_Group": exp_label,
            "Exposure_Binary": 1,
            "Outcome_Status": out_pos_label if had_event else out_neg_label,
            "Outcome_Binary": had_event,
            "Age": int(np.random.normal(55, 10)),
            "Sex": "Female" if np.random.rand() > 0.48 else "Male",
            "District": np.random.choice(["Derby City", "Chesterfield", "Amber Valley", "Derbyshire Dales", "High Peak"])
        })

    # Unexposed group
    for i in range(min(n0, 250)):
        had_event = 1 if i < (c * 250 // n0) else 0
        participants.append({
            "Participant_ID": f"UNEXP_{i+1:04d}",
            "Exposure_Group": unexp_label,
            "Exposure_Binary": 0,
            "Outcome_Status": out_pos_label if had_event else out_neg_label,
            "Outcome_Binary": had_event,
            "Age": int(np.random.normal(54, 10)),
            "Sex": "Female" if np.random.rand() > 0.48 else "Male",
            "District": np.random.choice(["Derby City", "Chesterfield", "Amber Valley", "Derbyshire Dales", "High Peak"])
        })

    df_microdata = pd.DataFrame(participants)

    calcs = calculate_attributable_fraction(
        a, b, c, d,
        pe_override=cfg.get("pe_override"),
        total_population_size=cfg.get("target_population_size")
    )

    meta = {
        **cfg,
        "scenario_key": scenario,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "calcs": calcs
    }

    return df_contingency, df_microdata, meta


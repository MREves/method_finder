"""Mathematical & Statistical Engine for Relative Risk (Risk Ratio) & Cohort Effect Measures.

Provides verified epidemiological calculations for:
1. Cumulative Incidence in Exposed (I_e) and Unexposed (I_u).
2. Relative Risk (Risk Ratio, RR) point estimate and log-scale Wald 95% Confidence Intervals.
3. Risk Difference (Attributable Risk, RD) and 95% Confidence Intervals.
4. Attributable Fraction in the Exposed (AF_e) and Relative Risk Reduction (RRR).
5. Number Needed to Treat (NNT) and Number Needed to Harm (NNH).
6. Log-scale Wald Z-test, Pearson Chi-Square, and Fisher's Exact Test p-values.
7. Haldane-Anscombe continuity correction (0.5 addition) for sparse/zero cell counts.
8. Prerequisite and study design diagnostic evaluation engine (cohort vs case-control vs cross-sectional).
9. Realistic cohort synthetic data generation (occupational dust, lifestyle intervention, vaccine trial).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_relative_risk(
    a: int,
    b: int,
    c: int,
    d: int,
    alpha: float = 0.05,
    continuity_correction: bool = False
) -> Dict[str, Any]:
    """Calculates Relative Risk (Risk Ratio), Risk Difference, and associated epidemiological metrics from a 2x2 table.

    A classical 2x2 contingency table is structured as follows:
                      Outcome (+)       Outcome (-)       Total
    Exposed (+)            a                 b            n1 = a + b
    Unexposed (-)          c                 d            n0 = c + d
    Total               m1 = a + c        m0 = b + d      N = a + b + c + d

    Args:
        a (int): Count of exposed individuals who developed the outcome (cases).
        b (int): Count of exposed individuals who remained healthy / disease-free.
        c (int): Count of unexposed individuals who developed the outcome (cases).
        d (int): Count of unexposed individuals who remained healthy / disease-free.
        alpha (float, optional): Significance level for two-sided confidence intervals. Defaults to 0.05 (95% CI).
        continuity_correction (bool, optional): Whether to apply the Haldane-Anscombe 0.5 
            continuity correction. Automatically applied if any cell is zero. Defaults to False.

    Returns:
        Dict[str, Any]: Comprehensive dictionary containing:
            - 'a', 'b', 'c', 'd': Raw cell counts.
            - 'n_exposed', 'n_unexposed', 'n_total': Marginals.
            - 'risk_exposed': Cumulative incidence in exposed (I_e = a / n1).
            - 'risk_unexposed': Cumulative incidence in unexposed (I_u = c / n0).
            - 'rr': Relative Risk point estimate (I_e / I_u).
            - 'ln_rr': Natural logarithm of the Relative Risk.
            - 'se_ln_rr': Standard error of ln(RR).
            - 'rr_ci_lower': Lower bound of (1 - alpha) confidence interval for RR.
            - 'rr_ci_upper': Upper bound of (1 - alpha) confidence interval for RR.
            - 'risk_difference': Absolute Risk Difference (RD = I_e - I_u).
            - 'se_rd': Standard error of the Risk Difference.
            - 'rd_ci_lower': Lower bound of CI for RD.
            - 'rd_ci_upper': Upper bound of CI for RD.
            - 'attributable_fraction_exposed': Attributable fraction among exposed (AF_e % or RRR %).
            - 'nnt': Number Needed to Treat (if protective, RD < 0). None if harmful.
            - 'nnh': Number Needed to Harm (if harmful, RD > 0). None if protective.
            - 'z_score': Wald Z-statistic on log-scale.
            - 'p_value_wald': Two-sided Wald p-value for H0: RR = 1.0.
            - 'p_value_chi2': Pearson Chi-Square test p-value (1 d.f.).
            - 'p_value_fisher': Fisher's exact test two-sided p-value.
            - 'continuity_applied': Boolean flag indicating if 0.5 correction was used.
            - 'effect_direction': 'harmful' (RR > 1), 'protective' (RR < 1), or 'null' (RR == 1).
    """
    applied_cc = continuity_correction or (a == 0 or b == 0 or c == 0 or d == 0)

    # Use float copies for calculation
    a_calc = float(a) + (0.5 if applied_cc else 0.0)
    b_calc = float(b) + (0.5 if applied_cc else 0.0)
    c_calc = float(c) + (0.5 if applied_cc else 0.0)
    d_calc = float(d) + (0.5 if applied_cc else 0.0)

    n1 = a_calc + b_calc
    n0 = c_calc + d_calc
    n_total = n1 + n0

    # Cumulative Incidences (Risk)
    risk_exposed = a_calc / n1 if n1 > 0 else 0.0
    risk_unexposed = c_calc / n0 if n0 > 0 else 0.0

    # Relative Risk point estimate
    if risk_unexposed > 0:
        rr = risk_exposed / risk_unexposed
    else:
        rr = np.nan

    # Log Relative Risk & Standard Error
    # SE(ln(RR)) = sqrt( (b / (a * n1)) + (d / (c * n0)) ) or sqrt(1/a - 1/n1 + 1/c - 1/n0)
    if not np.isnan(rr) and rr > 0 and a_calc > 0 and c_calc > 0:
        ln_rr = float(np.log(rr))
        se_ln_rr = float(np.sqrt((1.0 / a_calc) - (1.0 / n1) + (1.0 / c_calc) - (1.0 / n0)))
    else:
        ln_rr = np.nan
        se_ln_rr = np.nan

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    if not np.isnan(se_ln_rr):
        rr_ci_lower = float(np.exp(ln_rr - z_crit * se_ln_rr))
        rr_ci_upper = float(np.exp(ln_rr + z_crit * se_ln_rr))
        z_score = float(ln_rr / se_ln_rr) if se_ln_rr > 0 else 0.0
        p_value_wald = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
    else:
        rr_ci_lower = np.nan
        rr_ci_upper = np.nan
        z_score = np.nan
        p_value_wald = np.nan

    # Risk Difference (RD)
    risk_difference = float(risk_exposed - risk_unexposed)
    se_rd = float(np.sqrt((risk_exposed * (1.0 - risk_exposed) / n1) + (risk_unexposed * (1.0 - risk_unexposed) / n0)))
    rd_ci_lower = float(risk_difference - z_crit * se_rd)
    rd_ci_upper = float(risk_difference + z_crit * se_rd)

    # Attributable Fraction in Exposed (AF_e) & Relative Risk Reduction (RRR)
    if not np.isnan(rr):
        if rr >= 1.0:
            afe = float(((rr - 1.0) / rr) * 100.0) if rr > 0 else 0.0
            effect_direction = "harmful" if rr > 1.0 else "null"
        else:
            afe = float((1.0 - rr) * 100.0)  # Relative Risk Reduction / Vaccine Efficacy
            effect_direction = "protective"
    else:
        afe = np.nan
        effect_direction = "undefined"

    # Number Needed to Treat (NNT) / Harm (NNH)
    nnt = None
    nnh = None
    if abs(risk_difference) > 1e-6:
        if risk_difference > 0:
            nnh = float(1.0 / risk_difference)
        else:
            nnt = float(1.0 / abs(risk_difference))

    # Classical 2x2 Hypothesis tests (using raw counts)
    table_raw = np.array([[a, b], [c, d]])
    try:
        chi2_res = stats.chi2_contingency(table_raw, correction=False)
        p_value_chi2 = float(chi2_res.pvalue)
    except Exception:
        p_value_chi2 = np.nan

    try:
        _, p_value_fisher = stats.fisher_exact(table_raw)
        p_value_fisher = float(p_value_fisher)
    except Exception:
        p_value_fisher = np.nan

    return {
        "a": int(a),
        "b": int(b),
        "c": int(c),
        "d": int(d),
        "n_exposed": int(a + b),
        "n_unexposed": int(c + d),
        "n_total": int(a + b + c + d),
        "risk_exposed": float(risk_exposed),
        "risk_unexposed": float(risk_unexposed),
        "rr": float(rr),
        "ln_rr": float(ln_rr),
        "se_ln_rr": float(se_ln_rr),
        "rr_ci_lower": float(rr_ci_lower),
        "rr_ci_upper": float(rr_ci_upper),
        "risk_difference": float(risk_difference),
        "se_rd": float(se_rd),
        "rd_ci_lower": float(rd_ci_lower),
        "rd_ci_upper": float(rd_ci_upper),
        "attributable_fraction_exposed": float(afe),
        "nnt": nnt,
        "nnh": nnh,
        "z_score": float(z_score),
        "p_value_wald": float(p_value_wald),
        "p_value_chi2": float(p_value_chi2),
        "p_value_fisher": float(p_value_fisher),
        "continuity_applied": applied_cc,
        "effect_direction": effect_direction
    }


def evaluate_relative_risk_prerequisites(
    study_design: str,
    a: int,
    b: int,
    c: int,
    d: int,
    loss_to_follow_up_pct: float = 0.0,
    is_temporal_sequence_verified: bool = True
) -> List[Dict[str, Any]]:
    """Evaluates epidemiological prerequisites and assumptions for Relative Risk (Risk Ratio).

    Assesses study design validity, sample size robustness, loss to follow-up (attrition bias),
    and cell sparsity.

    Args:
        study_design (str): Study design architecture. Expected values:
            'prospective_cohort', 'retrospective_cohort', 'rct', 'case_control', 'cross_sectional'.
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        loss_to_follow_up_pct (float, optional): Percent of participants lost to follow-up. Defaults to 0.0.
        is_temporal_sequence_verified (bool, optional): Whether exposure unambiguously preceded outcome. Defaults to True.

    Returns:
        List[Dict[str, Any]]: Diagnostic evaluation cards containing:
            - 'name': Name of assumption / criterion.
            - 'status': 'PASS', 'WARNING', or 'FAIL'.
            - 'details': Plain-English evaluation text.
            - 'recommendation': Specific epidemiological guidance.
    """
    evaluations: List[Dict[str, Any]] = []

    # 1. Study Design Suitability
    valid_designs = {"prospective_cohort", "retrospective_cohort", "rct"}
    if study_design.lower() in valid_designs:
        evaluations.append({
            "name": "Study Design Suitability for Incidence / Risk",
            "status": "PASS",
            "details": f"Study design ({study_design.replace('_', ' ').title()}) tracks exposure forward to incident outcomes, permitting direct cumulative incidence calculation.",
            "recommendation": "Relative Risk (Risk Ratio) is mathematically and methodologically valid."
        })
    elif study_design.lower() == "case_control":
        evaluations.append({
            "name": "Study Design Suitability for Incidence / Risk",
            "status": "FAIL",
            "details": "Case-Control studies sample participants based on disease outcome (retrospective). Total population denominators are fixed by the researcher, so true cumulative incidence CANNOT be measured.",
            "recommendation": "Do NOT report Relative Risk. You MUST use Odds Ratio (OR) instead."
        })
    elif study_design.lower() == "cross_sectional":
        evaluations.append({
            "name": "Study Design Suitability for Incidence / Risk",
            "status": "WARNING",
            "details": "Cross-Sectional studies sample existing (prevalent) cases at a single slice in time, conflating incidence with disease duration.",
            "recommendation": "Report Prevalence Ratio (PR) or Prevalence Odds Ratio (POR). Interpret causal risk with severe caution."
        })
    else:
        evaluations.append({
            "name": "Study Design Suitability for Incidence / Risk",
            "status": "WARNING",
            "details": f"Unrecognized design ({study_design}). Ensure true incident new cases are tracked over a defined observation period.",
            "recommendation": "Verify whether baseline exposed and unexposed populations are known."
        })

    # 2. Temporal Precedence (Exposure precedes Outcome)
    if is_temporal_sequence_verified:
        evaluations.append({
            "name": "Temporal Precedence (Hill's Criterion)",
            "status": "PASS",
            "details": "Exposure measurement was verified prior to outcome occurrence, eliminating reverse causality.",
            "recommendation": "Causal directionality requirement satisfied."
        })
    else:
        evaluations.append({
            "name": "Temporal Precedence (Hill's Criterion)",
            "status": "FAIL",
            "details": "Temporal sequence cannot be confirmed. Outcome may have preceded or influenced exposure status (reverse causation).",
            "recommendation": "Prospective longitudinal tracking is required to establish incidence."
        })

    # 3. Minimum Cell Count Volume & Sparsity
    min_cell = min(a, b, c, d)
    if min_cell >= 5:
        evaluations.append({
            "name": "2x2 Cell Count Robustness (Sample Size)",
            "status": "PASS",
            "details": f"All four 2x2 contingency cells have at least 5 observations (minimum cell = {min_cell}).",
            "recommendation": "Large-sample Wald log-scale confidence intervals and Pearson Chi-Square tests are reliable."
        })
    elif min_cell > 0:
        evaluations.append({
            "name": "2x2 Cell Count Robustness (Sample Size)",
            "status": "WARNING",
            "details": f"One or more cells has fewer than 5 observations (minimum cell = {min_cell}). Normal log-scale approximation may suffer slight skew.",
            "recommendation": "Interpret Wald 95% CI with caution; cross-reference with Fisher's exact test p-value."
        })
    else:
        evaluations.append({
            "name": "2x2 Cell Count Robustness (Sample Size)",
            "status": "WARNING",
            "details": "One or more cells contains exactly zero events. Standard Relative Risk yields 0 or infinity.",
            "recommendation": "Automatic Haldane-Anscombe continuity correction (+0.5 added to all cells) has been applied."
        })

    # 4. Attrition / Loss to Follow-up
    if loss_to_follow_up_pct < 5.0:
        evaluations.append({
            "name": "Follow-Up Completeness (Attrition Bias)",
            "status": "PASS",
            "details": f"Loss to follow-up is minimal ({loss_to_follow_up_pct:.1f}% < 5.0%).",
            "recommendation": "Low risk of differential attrition bias."
        })
    elif loss_to_follow_up_pct <= 20.0:
        evaluations.append({
            "name": "Follow-Up Completeness (Attrition Bias)",
            "status": "WARNING",
            "details": f"Moderate loss to follow-up ({loss_to_follow_up_pct:.1f}%). If dropouts differed between exposed and unexposed, selection bias may distort the RR.",
            "recommendation": "Perform sensitivity analysis (best-case vs. worst-case imputation for lost participants)."
        })
    else:
        evaluations.append({
            "name": "Follow-Up Completeness (Attrition Bias)",
            "status": "FAIL",
            "details": f"Substantial loss to follow-up ({loss_to_follow_up_pct:.1f}% > 20.0%). The cohort represents a non-random survivor subset.",
            "recommendation": "High risk of attrition bias. Standard cumulative incidence may severely mislead."
        })

    return evaluations


def generate_relative_risk_demo_data(
    scenario: str = "occupational_dust",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic cohort data for Relative Risk pedagogical demonstrations.

    Provides three realistic public health scenarios:
    1. 'occupational_dust': Industrial Silica & Quarrying Dust Exposure vs. Incident COPD (10-yr follow-up).
    2. 'lifestyle_intervention': All Meadows Community Active Walking Programme vs. Coronary Heart Disease (5-yr follow-up).
    3. 'influenza_vaccine': High-Risk Elderly Winter Influenza Vaccination RCT vs. Confirmed Influenza Infection.

    Args:
        scenario (str, optional): Scenario identifier. Defaults to 'occupational_dust'.
        seed (int, optional): Random seed for reproducible individual microdata generation. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - df_contingency: Aggregated 2x2 contingency summary table.
            - df_microdata: Individual-level cohort participant microdata.
            - meta: Rich descriptive metadata and clinical interpretation narrative.
    """
    np.random.seed(seed)

    scenarios = {
        "occupational_dust": {
            "title": "Industrial Quarrying & Respirable Dust Cohort (Harmful Exposure)",
            "domain": "Occupational Lung Disease Epidemiology",
            "description": "A hypothetical 10-year prospective occupational cohort of quarry, limestone, and manufacturing workers across regional industrial sites. Simulates the impact of heavy respirable dust exposure on incident Chronic Obstructive Pulmonary Disease (COPD) for educational demonstration.",
            "exposure_name": "Heavy Respirable Dust Exposure",
            "exposure_levels": ("Exposed (Heavy Dust)", "Unexposed (Low/No Dust)"),
            "outcome_name": "Incident COPD (10-Year Follow-up)",
            "outcome_levels": ("Developed COPD", "No COPD"),
            "n_exposed": 450,
            "n_unexposed": 550,
            "p_exposed": 0.18,      # ~81 cases
            "p_unexposed": 0.07,    # ~39 cases
            "expected_effect": "Harmful (RR > 1.0)",
            "clinical_takeaway": "Heavy dust exposure increases COPD risk by approximately 2.5-fold. An Attributable Fraction of ~60% indicates that over half of COPD cases among exposed workers could be prevented by dust suppression controls.",
            "loss_to_follow_up_pct": 4.2,
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "lifestyle_intervention": {
            "title": "All Meadows Active Communities Physical Activity Initiative (Protective Intervention)",
            "domain": "Cardiovascular Prevention Epidemiology",
            "description": "A hypothetical 5-year community cohort simulating 1,200 sedentary adults aged 50-70 enrolled in All Meadows active walking and exercise initiatives versus sedentary routine care. Evaluates incident Coronary Heart Disease (CHD) events for educational demonstration.",
            "exposure_name": "Structured Walking / Exercise Programme",
            "exposure_levels": ("Active Programme", "Sedentary Control"),
            "outcome_name": "Incident Coronary Heart Disease (5-Year Follow-up)",
            "outcome_levels": ("CHD Event", "No Event"),
            "n_exposed": 600,
            "n_unexposed": 600,
            "p_exposed": 0.05,      # ~30 cases
            "p_unexposed": 0.11,    # ~66 cases
            "expected_effect": "Protective (RR < 1.0)",
            "clinical_takeaway": "Regular structured walking reduces 5-year CHD risk by ~55% (RR ~0.45). The Absolute Risk Reduction of ~6% corresponds to a Number Needed to Treat (NNT) of approximately 17 participants.",
            "loss_to_follow_up_pct": 3.5,
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "patient_feedback_survey": {
            "title": "Regional Healthcare NHS Patient Experience Survey (Youth vs. Adult Satisfaction)",
            "domain": "Health Services Research & Patient Experience Analytics",
            "description": "A cross-sectional patient experience survey (e.g. Friends and Family Test) across regional healthcare clinics and hospitals comparing satisfaction among young people (aged <= 18) versus adult patients (aged 19+). In cross-sectional surveys, Relative Risk is mathematically called the Prevalence Ratio (PR) — the mathematical calculations, standard errors, and confidence intervals are identical.",
            "exposure_name": "Demographic Cohort (Youth <= 18 vs. Adults 19+)",
            "exposure_levels": ("Young People (<= 18)", "Adults (19+)"),
            "outcome_name": "Overall Care Feedback Sentiment",
            "outcome_levels": ("Positive Feedback (+)", "Negative Feedback (-)"),
            "n_exposed": 200,
            "n_unexposed": 400,
            "p_exposed": 0.80,      # 160 positive out of 200 (80.0%)
            "p_unexposed": 0.60,    # 240 positive out of 400 (60.0%)
            "expected_effect": "Higher Positive Sentiment in Youth (Prevalence Ratio = 1.33)",
            "clinical_takeaway": "Young patients aged <= 18 were 1.33 times as likely (33% more likely) to rate their care positively compared to adult patients (Prevalence Ratio = 1.33, 95% CI: [1.20, 1.48]). In cross-sectional surveys, Relative Risk is termed the Prevalence Ratio (PR) because it measures the prevalence of existing sentiment rather than new incident health events over time.",
            "loss_to_follow_up_pct": 0.0,
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "influenza_vaccine": {
            "title": "Elderly Winter Influenza Immunization Trial (Vaccine Efficacy RCT)",
            "domain": "Infectious Disease Clinical Trial",
            "description": "A hypothetical double-blind randomized clinical trial simulating adjuvanted quadrivalent influenza vaccine versus standard control in adults aged 65+ across primary care practices during peak winter transmission season for educational demonstration.",
            "exposure_name": "Quadrivalent Influenza Vaccine",
            "exposure_levels": ("Vaccinated Group", "Placebo / Standard Control"),
            "outcome_name": "PCR-Confirmed Influenza Illness",
            "outcome_levels": ("Confirmed Flu", "No Infection"),
            "n_exposed": 750,
            "n_unexposed": 750,
            "p_exposed": 0.032,     # ~24 cases
            "p_unexposed": 0.096,   # ~72 cases
            "expected_effect": "Protective (Vaccine Efficacy = 1 - RR)",
            "clinical_takeaway": "Vaccine reduces influenza risk from 9.6% to 3.2% (RR = 0.33). Vaccine Efficacy (VE) is 67% (95% CI: 52% to 77%). NNT = 16 elderly adults vaccinated to prevent 1 laboratory-confirmed influenza infection.",
            "loss_to_follow_up_pct": 1.8,
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
    }


    if scenario not in scenarios:
        scenario = "occupational_dust"

    cfg = scenarios[scenario]

    # Generate exact or near-exact event counts
    a = int(round(cfg["n_exposed"] * cfg["p_exposed"]))
    b = int(cfg["n_exposed"] - a)
    c = int(round(cfg["n_unexposed"] * cfg["p_unexposed"]))
    d = int(cfg["n_unexposed"] - c)

    exp_label, unexp_label = cfg["exposure_levels"]
    out_pos_label, out_neg_label = cfg["outcome_levels"]

    # 1. Summary 2x2 Contingency Table
    df_contingency = pd.DataFrame([
        {
            "Exposure_Group": exp_label,
            "Cases_Outcome_Present": a,
            "Non_Cases_Outcome_Absent": b,
            "Total_Cohort_Size": a + b,
            "Cumulative_Incidence_Risk": float(a / (a + b))
        },
        {
            "Exposure_Group": unexp_label,
            "Cases_Outcome_Present": c,
            "Non_Cases_Outcome_Absent": d,
            "Total_Cohort_Size": c + d,
            "Cumulative_Incidence_Risk": float(c / (c + d))
        }
    ])

    # 2. Individual Participant Microdata
    participants = []
    is_survey = (scenario == "patient_feedback_survey")

    # Exposed group
    for i in range(a):
        age_val = int(np.random.randint(14, 19)) if is_survey else int(np.random.normal(56, 7))
        participants.append({
            "Participant_ID": f"EXP_{i+1:04d}",
            "Exposure_Status": exp_label,
            "Exposure_Binary": 1,
            "Outcome_Status": out_pos_label,
            "Outcome_Binary": 1,
            "Follow_Up_Years": 0.0 if is_survey else np.round(np.random.uniform(1.0, 10.0), 1),
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.45 else "Female"
        })
    for i in range(b):
        age_val = int(np.random.randint(14, 19)) if is_survey else int(np.random.normal(55, 7))
        participants.append({
            "Participant_ID": f"EXP_{a+i+1:04d}",
            "Exposure_Status": exp_label,
            "Exposure_Binary": 1,
            "Outcome_Status": out_neg_label,
            "Outcome_Binary": 0,
            "Follow_Up_Years": 0.0 if is_survey else (10.0 if scenario == "occupational_dust" else (5.0 if scenario == "lifestyle_intervention" else 1.0)),
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.45 else "Female"
        })
    # Unexposed group
    for i in range(c):
        age_val = int(np.random.randint(19, 76)) if is_survey else int(np.random.normal(55, 7))
        participants.append({
            "Participant_ID": f"UNEXP_{i+1:04d}",
            "Exposure_Status": unexp_label,
            "Exposure_Binary": 0,
            "Outcome_Status": out_pos_label,
            "Outcome_Binary": 1,
            "Follow_Up_Years": 0.0 if is_survey else np.round(np.random.uniform(1.0, 10.0), 1),
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.48 else "Female"
        })
    for i in range(d):
        age_val = int(np.random.randint(19, 76)) if is_survey else int(np.random.normal(54, 7))
        participants.append({
            "Participant_ID": f"UNEXP_{c+i+1:04d}",
            "Exposure_Status": unexp_label,
            "Exposure_Binary": 0,
            "Outcome_Status": out_neg_label,
            "Outcome_Binary": 0,
            "Follow_Up_Years": 0.0 if is_survey else (10.0 if scenario == "occupational_dust" else (5.0 if scenario == "lifestyle_intervention" else 1.0)),
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.48 else "Female"
        })

    df_microdata = pd.DataFrame(participants)


    # Attach calculated results to metadata
    calcs = calculate_relative_risk(a, b, c, d)
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


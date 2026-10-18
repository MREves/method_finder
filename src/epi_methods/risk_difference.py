"""Mathematical & Statistical Engine for Risk Difference (RD) & Number Needed to Treat (NNT/NNH).

Provides verified epidemiological calculations for:
1. Cumulative Incidence in Exposed (I_e) and Unexposed (I_u).
2. Risk Difference (RD) / Absolute Risk Reduction (ARR) point estimate: I_e - I_u.
3. Linear-Scale Standard Error: sqrt([I_e(1 - I_e)/n1] + [I_u(1 - I_u)/n0]).
4. Linear-Scale Wald 95% Confidence Intervals for RD: RD ± 1.96 * SE(RD).
5. Number Needed to Treat (NNT = 1 / |RD|) for beneficial interventions.
6. Number Needed to Harm (NNH = 1 / RD) for hazardous exposures.
7. Wald Z-test for H0: RD = 0.0, Pearson Chi-Square, and Fisher's Exact Test p-values.
8. Prerequisite and methodological diagnostic evaluation engine (Study Design, Baseline Risk Dependency, Cell Counts, Attrition Bias).
9. Realistic public health synthetic scenarios: Statin Cardiovascular Trial, Smoking Cessation, Occupational Vibration, and NHS Patient Feedback Survey.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_risk_difference(
    a: int,
    b: int,
    c: int,
    d: int,
    alpha: float = 0.05,
    continuity_correction: bool = False
) -> Dict[str, Any]:
    """Calculates Risk Difference (RD), Absolute Risk Reduction (ARR), NNT, NNH, SE, and Wald CIs from a 2x2 table.

    A classical 2x2 contingency cohort table is structured as:
                          Cases (Outcome +)       Non-Cases (Outcome -)       Total
    Exposed (+)                  a                         b                 n1 = a + b
    Unexposed (-)                c                         d                 n0 = c + d
    Total                     m1 = a + c                m0 = b + d           N = a + b + c + d

    Mathematical Definitions:
        Cumulative Incidence in Exposed (I_e) = a / n1 = a / (a + b)
        Cumulative Incidence in Unexposed (I_u) = c / n0 = c / (c + d)
        Risk Difference (RD) = I_e - I_u
        Linear Standard Error: SE(RD) = sqrt([I_e(1 - I_e)/n1] + [I_u(1 - I_u)/n0])
        95% Wald Confidence Interval = RD ± Z_(1 - alpha/2) * SE(RD)
        Number Needed to Treat (NNT) = 1 / |RD| (when RD < 0, i.e., protective treatment)
        Number Needed to Harm (NNH) = 1 / RD (when RD > 0, i.e., harmful hazard)

    Args:
        a (int): Count of exposed cases (with outcome).
        b (int): Count of exposed non-cases (without outcome).
        c (int): Count of unexposed cases (with outcome).
        d (int): Count of unexposed non-cases (without outcome).
        alpha (float, optional): Significance level for two-sided confidence intervals. Defaults to 0.05 (95% CI).
        continuity_correction (bool, optional): Whether to apply a 0.5 continuity correction if cells are sparse. Defaults to False.

    Returns:
        Dict[str, Any]: Comprehensive dictionary containing:
            - 'a', 'b', 'c', 'd': Raw cell frequencies.
            - 'n_exposed', 'n_unexposed', 'n_total': Cohort denominators.
            - 'risk_exposed': Cumulative incidence in exposed (I_e).
            - 'risk_unexposed': Cumulative incidence in unexposed (I_u).
            - 'risk_difference': Point estimate of Risk Difference (RD = I_e - I_u).
            - 'arr': Absolute Risk Reduction (|RD| if protective, else 0.0).
            - 'se_rd': Standard error of Risk Difference on the linear scale.
            - 'rd_ci_lower': Lower bound of (1 - alpha) CI for RD.
            - 'rd_ci_upper': Upper bound of (1 - alpha) CI for RD.
            - 'nnt': Number Needed to Treat (if protective / beneficial), else None.
            - 'nnh': Number Needed to Harm (if harmful / hazardous), else None.
            - 'attributable_fraction_exposed': AF_e % = (RD / I_e) * 100 if harmful.
            - 'z_score': Wald Z-statistic on linear scale (RD / SE).
            - 'p_value_wald': Two-sided Wald p-value for H0: RD = 0.0.
            - 'p_value_chi2': Pearson Chi-Square test p-value (1 d.f.).
            - 'p_value_fisher': Fisher's exact test two-sided p-value.
            - 'effect_direction': 'harmful' (RD > 0), 'protective' (RD < 0), or 'null' (RD == 0).
            - 'percentage_point_diff': RD expressed in percentage points (RD * 100).
    """
    applied_cc = continuity_correction or (a == 0 or b == 0 or c == 0 or d == 0)

    a_c = float(a) + (0.5 if applied_cc else 0.0)
    b_c = float(b) + (0.5 if applied_cc else 0.0)
    c_c = float(c) + (0.5 if applied_cc else 0.0)
    d_c = float(d) + (0.5 if applied_cc else 0.0)

    n1 = a + b
    n0 = c + d
    n1_calc = a_c + b_c
    n0_calc = c_c + d_c
    n_tot = n1 + n0

    risk_exp = float(a_c / n1_calc) if n1_calc > 0 else 0.0
    risk_unexp = float(c_c / n0_calc) if n0_calc > 0 else 0.0

    rd = float(risk_exp - risk_unexp)

    # Linear-scale Standard Error: sqrt(I_e*(1 - I_e)/n1 + I_u*(1 - I_u)/n0)
    var_exp = (risk_exp * (1.0 - risk_exp)) / n1_calc if n1_calc > 0 else 0.0
    var_unexp = (risk_unexp * (1.0 - risk_unexp)) / n0_calc if n0_calc > 0 else 0.0
    se_rd = float(np.sqrt(max(0.0, var_exp + var_unexp)))

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    if se_rd > 0:
        rd_ci_lower = max(-1.0, rd - z_crit * se_rd)
        rd_ci_upper = min(1.0, rd + z_crit * se_rd)
        z_score = float(rd / se_rd)
        p_value_wald = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
    else:
        rd_ci_lower = rd
        rd_ci_upper = rd
        z_score = 0.0
        p_value_wald = 1.0

    # Effect direction and NNT / NNH
    if rd > 1e-6:
        effect_direction = "harmful"
        arr = 0.0
        nnh = float(1.0 / rd)
        nnt = None
        afe = float((rd / risk_exp) * 100.0) if risk_exp > 0 else 0.0
    elif rd < -1e-6:
        effect_direction = "protective"
        arr = float(abs(rd))
        nnt = float(1.0 / arr)
        nnh = None
        afe = float((arr / risk_unexp) * 100.0) if risk_unexp > 0 else 0.0
    else:
        effect_direction = "null"
        arr = 0.0
        nnt = None
        nnh = None
        afe = 0.0

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
        "n_exposed": int(n1),
        "n_unexposed": int(n0),
        "n_total": int(n_tot),
        "risk_exposed": float(risk_exp),
        "risk_unexposed": float(risk_unexp),
        "risk_difference": float(rd),
        "arr": float(arr),
        "se_rd": float(se_rd),
        "rd_ci_lower": float(rd_ci_lower),
        "rd_ci_upper": float(rd_ci_upper),
        "nnt": nnt,
        "nnh": nnh,
        "attributable_fraction_exposed": float(afe),
        "z_score": float(z_score),
        "p_value_wald": float(p_value_wald),
        "p_value_chi2": float(p_value_chi2),
        "p_value_fisher": float(p_value_fisher),
        "continuity_applied": applied_cc,
        "effect_direction": effect_direction,
        "percentage_point_diff": float(rd * 100.0)
    }


def evaluate_risk_difference_prerequisites(
    study_design: str,
    a: int,
    b: int,
    c: int,
    d: int,
    loss_to_follow_up_pct: float = 0.0,
    baseline_risk_known: bool = True
) -> List[Dict[str, Any]]:
    """Evaluates epidemiological prerequisites, baseline dependency, and bias diagnostics for Risk Difference.

    Args:
        study_design (str): Epidemiological design architecture. Expected values:
            'prospective_cohort', 'retrospective_cohort', 'rct', 'cross_sectional', 'case_control'.
        a (int): Exposed cases.
        b (int): Exposed non-cases.
        c (int): Unexposed cases.
        d (int): Unexposed non-cases.
        loss_to_follow_up_pct (float, optional): Percent of participants lost to follow-up. Defaults to 0.0.
        baseline_risk_known (bool, optional): Whether true population baseline risk is representative. Defaults to True.

    Returns:
        List[Dict[str, Any]]: Diagnostic evaluation cards containing:
            - 'name': Name of assumption / criterion.
            - 'status': 'PASS', 'WARNING', or 'FAIL'.
            - 'details': Plain-English evaluation text.
            - 'recommendation': Specific epidemiological guidance.
    """
    evaluations: List[Dict[str, Any]] = []

    # 1. Study Design Suitability
    valid_forward = {"prospective_cohort", "retrospective_cohort", "rct"}
    if study_design.lower() in valid_forward:
        evaluations.append({
            "name": "Study Design Suitability for Absolute Risk",
            "status": "PASS",
            "details": f"Study design ({study_design.replace('_', ' ').title()}) follows participants across time with known denominators, permitting valid calculation of cumulative incidences and absolute risk differences.",
            "recommendation": "Risk Difference (RD) and NNT/NNH are methodologically valid and clinically meaningful."
        })
    elif study_design.lower() == "cross_sectional":
        evaluations.append({
            "name": "Study Design Suitability for Absolute Risk",
            "status": "PASS",
            "details": "Cross-Sectional survey design measures existing prevalence at a point in time. Risk Difference represents the 'Percentage Point Difference' in prevalence.",
            "recommendation": "Report as 'Percentage Point Difference' or 'Prevalence Difference'. Do NOT interpret NNT as longitudinal treatment efficacy."
        })
    elif study_design.lower() == "case_control":
        evaluations.append({
            "name": "Study Design Suitability for Absolute Risk",
            "status": "FAIL",
            "details": "Case-Control studies select participants based on disease outcome. Denominators are researcher-determined quotas, so absolute incidence and Risk Difference CANNOT be calculated.",
            "recommendation": "Do NOT report Risk Difference or NNT/NNH. You MUST use Odds Ratio (OR) instead."
        })
    else:
        evaluations.append({
            "name": "Study Design Suitability for Absolute Risk",
            "status": "WARNING",
            "details": f"Unrecognized design ({study_design}). Ensure true incident new cases and denominators are known.",
            "recommendation": "Verify whether participant recruitment was outcome-first or exposure-first."
        })

    # 2. Baseline Risk Dependency (The Core Epistemology of Absolute Risk)
    calcs = calculate_risk_difference(a, b, c, d)
    risk_unexp = calcs["risk_unexposed"]
    if baseline_risk_known and risk_unexp > 0.001:
        evaluations.append({
            "name": "Baseline Risk Generalizability (I_u Dependency)",
            "status": "PASS",
            "details": f"Baseline unexposed risk is measurable ({risk_unexp * 100.0:.1f}%). Unlike Relative Risk, Risk Difference and NNT/NNH depend directly on baseline risk.",
            "recommendation": "Always specify the target population's baseline risk when applying these NNT/NNH figures to other clinical settings."
        })
    else:
        evaluations.append({
            "name": "Baseline Risk Generalizability (I_u Dependency)",
            "status": "WARNING",
            "details": "Baseline risk is extremely low (< 0.1%) or uncertain. Even with a large relative reduction, absolute risk reduction will be tiny and NNT will be immense.",
            "recommendation": "Beware the 'Rare Disease Trap': verify whether treating very low-risk individuals is cost-effective."
        })

    # 3. Minimum Cell Count Volume & Sparsity
    min_cell = min(a, b, c, d)
    if min_cell >= 5:
        evaluations.append({
            "name": "Cell Count Robustness (Sample Size)",
            "status": "PASS",
            "details": f"All four cells have at least 5 observations (minimum cell = {min_cell}). Linear-scale Wald CI is robust.",
            "recommendation": "Standard asymptotic normal approximations for Risk Difference are reliable."
        })
    elif min_cell > 0:
        evaluations.append({
            "name": "Cell Count Robustness (Sample Size)",
            "status": "WARNING",
            "details": f"One or more cells has fewer than 5 observations (minimum cell = {min_cell}). Asymptotic normal CI may slightly under-cover.",
            "recommendation": "Cross-reference linear Wald 95% CI with Fisher's exact test p-value."
        })
    else:
        evaluations.append({
            "name": "Cell Count Robustness (Sample Size)",
            "status": "WARNING",
            "details": "One or more cells contains exactly zero observations. Continuity correction (+0.5) has been applied.",
            "recommendation": "Interpret confidence bounds with caution due to zero-event sparsity."
        })

    # 4. Follow-up Completeness (Attrition Bias)
    if study_design.lower() == "cross_sectional":
        evaluations.append({
            "name": "Follow-Up Completeness (Attrition Bias)",
            "status": "PASS",
            "details": "Cross-sectional survey snapshot (no longitudinal follow-up period). Attrition bias does not apply (though survey non-response bias should be evaluated).",
            "recommendation": "Examine survey response rate and demographic representativeness."
        })
    elif loss_to_follow_up_pct < 5.0:
        evaluations.append({
            "name": "Follow-Up Completeness (Attrition Bias)",
            "status": "PASS",
            "details": f"Minimal loss to follow-up ({loss_to_follow_up_pct:.1f}% < 5.0%). Very low risk of attrition bias.",
            "recommendation": "High methodological internal validity for cumulative incidence and absolute difference."
        })
    elif loss_to_follow_up_pct <= 20.0:
        evaluations.append({
            "name": "Follow-Up Completeness (Attrition Bias)",
            "status": "WARNING",
            "details": f"Moderate loss to follow-up ({loss_to_follow_up_pct:.1f}%). If dropouts differed between exposed and unexposed, selection bias may distort the Risk Difference.",
            "recommendation": "Perform sensitivity analysis (worst-case vs. best-case imputation for lost participants)."
        })
    else:
        evaluations.append({
            "name": "Follow-Up Completeness (Attrition Bias)",
            "status": "FAIL",
            "details": f"Substantial loss to follow-up ({loss_to_follow_up_pct:.1f}% > 20.0%). The cohort represents a non-random survivor subset.",
            "recommendation": "High risk of attrition bias. Standard cumulative incidence and Risk Difference may severely mislead."
        })

    return evaluations


def generate_risk_difference_demo_data(
    scenario: str = "statin_cardiovascular",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic cohort and survey data for Risk Difference and NNT/NNH pedagogical demonstrations.

    Provides four realistic public health scenarios:
    1. 'statin_cardiovascular': High-Risk Secondary Prevention Cardiovascular Statin RCT (Protective, NNT focus).
    2. 'smoking_cessation': All Meadows Community Smoking Cessation Pharmacotherapy Trial (Protective, NNT focus).
    3. 'occupational_vibration': Regional Quarrying & Industrial Pneumatic Drill Vibration Cohort (Harmful, NNH focus).
    4. 'patient_feedback_survey': Regional Healthcare NHS Patient Experience Survey: Youth vs Adult Satisfaction (Survey Percentage Point Difference).

    Args:
        scenario (str, optional): Scenario identifier. Defaults to 'statin_cardiovascular'.
        seed (int, optional): Random seed for reproducible individual microdata generation. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - df_contingency: Aggregated 2x2 contingency summary table.
            - df_microdata: Individual-level participant microdata.
            - meta: Rich descriptive metadata, clinical takeaways, and disclaimer notices.
    """
    np.random.seed(seed)

    scenarios = {
        "statin_cardiovascular": {
            "title": "Regional Secondary Prevention Statin Trial (Beneficial Intervention)",
            "domain": "Cardiovascular Preventive Medicine & Clinical Pharmacology",
            "description": "A hypothetical 5-year double-blind randomized clinical trial simulating high-intensity statin therapy versus standard care in 2,000 adult patients with pre-existing coronary artery disease recruited across regional cardiology clinics. Evaluates prevention of major adverse cardiovascular events (MACE).",
            "exposure_name": "High-Intensity Statin Therapy (80mg)",
            "exposure_levels": ("Statin Therapy", "Standard Care / Placebo"),
            "outcome_name": "Major Adverse Cardiac Event (5-Yr MACE)",
            "outcome_levels": ("MACE Event (+)", "Event-Free (-)"),
            "n_exposed": 1000,
            "n_unexposed": 1000,
            "p_exposed": 0.04,      # 40 cases (4.0%)
            "p_unexposed": 0.09,    # 90 cases (9.0%)
            "study_design": "rct",
            "expected_effect": "Protective (ARR = 5.0%, NNT = 20)",
            "clinical_takeaway": "Statin therapy reduces 5-year cardiac events from 9.0% down to 4.0%, achieving an Absolute Risk Reduction (ARR) of 5.0% (95% CI: [2.8%, 7.2%]). The Number Needed to Treat (NNT) is 20 (95% CI: [14, 36]) — treating 20 high-risk patients over 5 years prevents exactly 1 major cardiac event.",
            "loss_to_follow_up_pct": 2.1,
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "smoking_cessation": {
            "title": "All Meadows Community Smoking Cessation Pharmacotherapy Initiative",
            "domain": "Substance Misuse & Public Health Intervention",
            "description": "A hypothetical 1-year community randomized trial evaluating combined varenicline pharmacotherapy plus behavioral counseling versus behavioral counseling alone among 800 heavy smokers recruited from All Meadows, Riverside, and Inner City GP practices.",
            "exposure_name": "Varenicline + Behavioral Counseling",
            "exposure_levels": ("Combined Pharmacotherapy", "Counseling Alone"),
            "outcome_name": "Smoking Relapse at 12 Months",
            "outcome_levels": ("Relapsed to Smoking", "Sustained Abstinence"),
            "n_exposed": 400,
            "n_unexposed": 400,
            "p_exposed": 0.45,      # 180 relapsed (45.0%)
            "p_unexposed": 0.65,    # 260 relapsed (65.0%)
            "study_design": "rct",
            "expected_effect": "Protective against Relapse (ARR = 20.0%, NNT = 5)",
            "clinical_takeaway": "Adding pharmacotherapy cuts the 1-year relapse rate from 65.0% to 45.0% (Absolute Risk Reduction = 20.0%, 95% CI: [13.2%, 26.8%]). The Number Needed to Treat (NNT) is 5 — enrolling 5 smokers in combined therapy produces 1 additional sustained long-term quitter.",
            "loss_to_follow_up_pct": 3.8,
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "occupational_vibration": {
            "title": "Regional Quarrying & Industrial Hand-Arm Vibration Cohort (Harmful Hazard)",
            "domain": "Occupational Health & Musculoskeletal Epidemiology",
            "description": "A hypothetical 8-year prospective occupational cohort of 1,000 industrial quarrying, stonecutting, and roadworks employees. Evaluates the harmful impact of heavy daily pneumatic drill and jackhammer vibration on incident Hand-Arm Vibration Syndrome (HAVS / Vibration White Finger).",
            "exposure_name": "Heavy Pneumatic Tool Vibration (> 4 Hrs/Day)",
            "exposure_levels": ("Heavy Vibration Exposure", "Low/Unexposed Office/Yard"),
            "outcome_name": "Incident Hand-Arm Vibration Syndrome (HAVS)",
            "outcome_levels": ("Developed HAVS (+)", "No HAVS (-)"),
            "n_exposed": 400,
            "n_unexposed": 600,
            "p_exposed": 0.22,      # 88 cases (22.0%)
            "p_unexposed": 0.04,    # 24 cases (4.0%)
            "study_design": "prospective_cohort",
            "expected_effect": "Harmful (RD = +18.0%, NNH ≈ 5.6)",
            "clinical_takeaway": "Heavy tool vibration causes an absolute excess of 18.0 cases of HAVS per 100 workers (Risk Difference = +18.0%, 95% CI: [13.6%, 22.4%]). The Number Needed to Harm (NNH) is 5.6 (≈ 6 workers) — for every 6 workers exposed to heavy vibration over 8 years, 1 additional worker develops permanent Hand-Arm Vibration Syndrome.",
            "loss_to_follow_up_pct": 4.5,
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "patient_feedback_survey": {
            "title": "Regional Healthcare NHS Patient Experience Survey (Youth vs. Adult Satisfaction Difference)",
            "domain": "Health Services Research & Cross-Sectional Survey Analytics",
            "description": "A cross-sectional patient experience survey (e.g. Friends and Family Test) across regional healthcare clinics and hospitals comparing satisfaction between young patients (aged <= 18) and adult patients (aged 19+). In cross-sectional survey research, Risk Difference is termed the Percentage Point Difference (or Prevalence Difference).",
            "exposure_name": "Demographic Cohort (Youth <= 18 vs. Adults 19+)",
            "exposure_levels": ("Young People (<= 18)", "Adults (19+)"),
            "outcome_name": "Overall Care Feedback Sentiment",
            "outcome_levels": ("Positive Feedback (+)", "Negative Feedback (-)"),
            "n_exposed": 200,
            "n_unexposed": 400,
            "p_exposed": 0.80,      # 160 positive out of 200 (80.0%)
            "p_unexposed": 0.60,    # 240 positive out of 400 (60.0%)
            "study_design": "cross_sectional",
            "expected_effect": "Higher Positive Sentiment in Youth (+20.0 Percentage Points)",
            "clinical_takeaway": "Young patients aged <= 18 had an 80.0% positive satisfaction rate compared to 60.0% in adults. The Risk Difference is +20.0 percentage points (95% CI: [+12.9%, +27.1%]). In survey research, RD is called the Percentage Point Difference because it represents the absolute gap in prevalence between groups.",
            "loss_to_follow_up_pct": 0.0,
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
    }

    if scenario not in scenarios:
        scenario = "statin_cardiovascular"

    cfg = scenarios[scenario]

    a = int(round(cfg["n_exposed"] * cfg["p_exposed"]))
    b = int(cfg["n_exposed"] - a)
    c = int(round(cfg["n_unexposed"] * cfg["p_unexposed"]))
    d = int(cfg["n_unexposed"] - c)

    exp_label, unexp_label = cfg["exposure_levels"]
    out_pos_label, out_neg_label = cfg["outcome_levels"]

    # 1. Summary 2x2 Contingency Table
    df_contingency = pd.DataFrame([
        {
            "Cohort_Stratum": f"🟢 {exp_label}",
            "Cases_Outcome_Pos": a,
            "Non_Cases_Outcome_Neg": b,
            "Stratum_Total_n": a + b,
            "Cumulative_Incidence_Risk": float(a / (a + b)) if (a + b) > 0 else 0.0
        },
        {
            "Cohort_Stratum": f"⚪ {unexp_label}",
            "Cases_Outcome_Pos": c,
            "Non_Cases_Outcome_Neg": d,
            "Stratum_Total_n": c + d,
            "Cumulative_Incidence_Risk": float(c / (c + d)) if (c + d) > 0 else 0.0
        },
        {
            "Cohort_Stratum": "📊 Total Cohort",
            "Cases_Outcome_Pos": a + c,
            "Non_Cases_Outcome_Neg": b + d,
            "Stratum_Total_n": a + b + c + d,
            "Cumulative_Incidence_Risk": float((a + c) / (a + b + c + d)) if (a + b + c + d) > 0 else 0.0
        }
    ])

    # 2. Individual Participant Microdata
    participants = []
    is_survey = (scenario == "patient_feedback_survey")

    for i in range(a):
        age_val = int(np.random.randint(14, 19)) if is_survey else int(np.random.normal(58, 7))
        participants.append({
            "Participant_ID": f"EXP_{i+1:04d}",
            "Exposure_Group": exp_label,
            "Exposure_Binary": 1,
            "Outcome_Status": out_pos_label,
            "Outcome_Binary": 1,
            "Follow_Up_Years": 0.0 if is_survey else np.round(np.random.uniform(1.0, 5.0), 1),
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.45 else "Female"
        })
    for i in range(b):
        age_val = int(np.random.randint(14, 19)) if is_survey else int(np.random.normal(57, 7))
        participants.append({
            "Participant_ID": f"EXP_{a+i+1:04d}",
            "Exposure_Group": exp_label,
            "Exposure_Binary": 1,
            "Outcome_Status": out_neg_label,
            "Outcome_Binary": 0,
            "Follow_Up_Years": 0.0 if is_survey else 5.0,
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.45 else "Female"
        })
    for i in range(c):
        age_val = int(np.random.randint(19, 76)) if is_survey else int(np.random.normal(59, 7))
        participants.append({
            "Participant_ID": f"UNEXP_{i+1:04d}",
            "Exposure_Group": unexp_label,
            "Exposure_Binary": 0,
            "Outcome_Status": out_pos_label,
            "Outcome_Binary": 1,
            "Follow_Up_Years": 0.0 if is_survey else np.round(np.random.uniform(1.0, 5.0), 1),
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.48 else "Female"
        })
    for i in range(d):
        age_val = int(np.random.randint(19, 76)) if is_survey else int(np.random.normal(58, 7))
        participants.append({
            "Participant_ID": f"UNEXP_{c+i+1:04d}",
            "Exposure_Group": unexp_label,
            "Exposure_Binary": 0,
            "Outcome_Status": out_neg_label,
            "Outcome_Binary": 0,
            "Follow_Up_Years": 0.0 if is_survey else 5.0,
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.48 else "Female"
        })

    df_microdata = pd.DataFrame(participants)

    # Attach calculation results to metadata
    calcs = calculate_risk_difference(a, b, c, d)
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


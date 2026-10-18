"""Mathematical & Statistical Engine for Odds Ratio (OR) & Case-Control Effect Measures.

Provides verified epidemiological calculations for:
1. Exposure Odds in Cases and Controls.
2. Odds Ratio (OR) point estimate via cross-product ratio (a * d) / (b * c).
3. Woolf's log-scale Standard Error: sqrt(1/a + 1/b + 1/c + 1/d).
4. Wald 95% Confidence Intervals for OR on the natural log scale.
5. Haldane-Anscombe continuity correction (0.5 addition) for zero/sparse cell counts.
6. Rare Disease Assumption evaluation (when OR approximates Relative Risk).
7. Wald Z-test, Pearson Chi-Square, and Fisher's Exact Test p-values.
8. Prerequisite and methodological diagnostic evaluation engine (Case-Control vs Cohort, Recall Bias, Berkson's Bias).
9. Realistic Case-Control synthetic data generation (Mesothelioma/Asbestos, Outbreak Investigation, Cardiovascular Diet).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_odds_ratio(
    a: int,
    b: int,
    c: int,
    d: int,
    alpha: float = 0.05,
    continuity_correction: bool = False
) -> Dict[str, Any]:
    """Calculates Odds Ratio (OR), exposure odds, Woolf's standard error, and confidence intervals from a 2x2 table.

    A classical Case-Control 2x2 contingency table is structured as follows:
                          Cases (Outcome +)       Controls (Outcome -)       Total
    Exposed (+)                  a                         b                 m1 = a + b
    Unexposed (-)                c                         d                 m0 = c + d
    Total                     n1 = a + c                n0 = b + d           N = a + b + c + d

    Exposure Odds:
        Odds of exposure in Cases = a / c
        Odds of exposure in Controls = b / d
        Odds Ratio (OR) = (a / c) / (b / d) = (a * d) / (b * c)

    Args:
        a (int): Count of diseased cases who were exposed to the risk factor.
        b (int): Count of healthy controls who were exposed to the risk factor.
        c (int): Count of diseased cases who were unexposed to the risk factor.
        d (int): Count of healthy controls who were unexposed to the risk factor.
        alpha (float, optional): Significance level for two-sided confidence intervals. Defaults to 0.05 (95% CI).
        continuity_correction (bool, optional): Whether to apply the Haldane-Anscombe 0.5 
            continuity correction. Automatically applied if any cell is zero. Defaults to False.

    Returns:
        Dict[str, Any]: Comprehensive dictionary containing:
            - 'a', 'b', 'c', 'd': Raw cell counts.
            - 'n_cases', 'n_controls', 'n_total': Case and control marginals.
            - 'exposure_pct_cases': Percentage of cases exposed [a / (a + c) * 100].
            - 'exposure_pct_controls': Percentage of controls exposed [b / (b + d) * 100].
            - 'odds_cases': Exposure odds in cases (a / c).
            - 'odds_controls': Exposure odds in controls (b / d).
            - 'or_point': Odds Ratio point estimate (a * d) / (b * c).
            - 'ln_or': Natural logarithm of the Odds Ratio.
            - 'se_ln_or': Woolf's Standard Error of ln(OR).
            - 'or_ci_lower': Lower bound of (1 - alpha) confidence interval for OR.
            - 'or_ci_upper': Upper bound of (1 - alpha) confidence interval for OR.
            - 'z_score': Wald Z-statistic on log-scale [ln(OR) / SE].
            - 'p_value_wald': Two-sided Wald p-value for H0: OR = 1.0.
            - 'p_value_chi2': Pearson Chi-Square test p-value (1 d.f.).
            - 'p_value_fisher': Fisher's exact test two-sided p-value.
            - 'continuity_applied': Boolean flag indicating if 0.5 correction was used.
            - 'effect_direction': 'harmful' (OR > 1), 'protective' (OR < 1), or 'null' (OR == 1).
            - 'odds_reduction_pct': Percentage reduction in odds if protective [(1 - OR) * 100].
    """
    applied_cc = continuity_correction or (a == 0 or b == 0 or c == 0 or d == 0)

    # Use float copies for calculation
    a_calc = float(a) + (0.5 if applied_cc else 0.0)
    b_calc = float(b) + (0.5 if applied_cc else 0.0)
    c_calc = float(c) + (0.5 if applied_cc else 0.0)
    d_calc = float(d) + (0.5 if applied_cc else 0.0)

    n_cases = a + c
    n_controls = b + d
    n_total = n_cases + n_controls

    # Exposure proportions
    exposure_pct_cases = (float(a) / n_cases * 100.0) if n_cases > 0 else 0.0
    exposure_pct_controls = (float(b) / n_controls * 100.0) if n_controls > 0 else 0.0

    # Exposure Odds
    odds_cases = a_calc / c_calc if c_calc > 0 else np.nan
    odds_controls = b_calc / d_calc if d_calc > 0 else np.nan

    # Cross-product Odds Ratio
    if b_calc > 0 and c_calc > 0:
        or_point = (a_calc * d_calc) / (b_calc * c_calc)
    else:
        or_point = np.nan

    # Woolf's Log-Scale Variance & Standard Error: SE(ln(OR)) = sqrt(1/a + 1/b + 1/c + 1/d)
    if not np.isnan(or_point) and or_point > 0 and a_calc > 0 and b_calc > 0 and c_calc > 0 and d_calc > 0:
        ln_or = float(np.log(or_point))
        se_ln_or = float(np.sqrt((1.0 / a_calc) + (1.0 / b_calc) + (1.0 / c_calc) + (1.0 / d_calc)))
    else:
        ln_or = np.nan
        se_ln_or = np.nan

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    if not np.isnan(se_ln_or):
        or_ci_lower = float(np.exp(ln_or - z_crit * se_ln_or))
        or_ci_upper = float(np.exp(ln_or + z_crit * se_ln_or))
        z_score = float(ln_or / se_ln_or) if se_ln_or > 0 else 0.0
        p_value_wald = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
    else:
        or_ci_lower = np.nan
        or_ci_upper = np.nan
        z_score = np.nan
        p_value_wald = np.nan

    # Effect direction and interpretation
    if not np.isnan(or_point):
        if or_point > 1.0001:
            effect_direction = "harmful"
            odds_reduction_pct = 0.0
        elif or_point < 0.9999:
            effect_direction = "protective"
            odds_reduction_pct = float((1.0 - or_point) * 100.0)
        else:
            effect_direction = "null"
            odds_reduction_pct = 0.0
    else:
        effect_direction = "undefined"
        odds_reduction_pct = np.nan

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
        "n_cases": int(n_cases),
        "n_controls": int(n_controls),
        "n_total": int(n_total),
        "exposure_pct_cases": float(exposure_pct_cases),
        "exposure_pct_controls": float(exposure_pct_controls),
        "odds_cases": float(odds_cases),
        "odds_controls": float(odds_controls),
        "or_point": float(or_point),
        "ln_or": float(ln_or),
        "se_ln_or": float(se_ln_or),
        "or_ci_lower": float(or_ci_lower),
        "or_ci_upper": float(or_ci_upper),
        "z_score": float(z_score),
        "p_value_wald": float(p_value_wald),
        "p_value_chi2": float(p_value_chi2),
        "p_value_fisher": float(p_value_fisher),
        "continuity_applied": applied_cc,
        "effect_direction": effect_direction,
        "odds_reduction_pct": float(odds_reduction_pct)
    }


def evaluate_odds_ratio_prerequisites(
    study_design: str,
    a: int,
    b: int,
    c: int,
    d: int,
    control_sampling_type: str = "population",
    disease_is_rare: bool = True,
    is_recall_bias_risk: bool = False
) -> List[Dict[str, Any]]:
    """Evaluates methodological prerequisites and bias diagnostics for Odds Ratio.

    Assesses study design validity, rare disease assumption (for RR approximation),
    control group sampling (Berkson's bias), cell count robustness, and recall bias.

    Args:
        study_design (str): Epidemiological design architecture. Expected values:
            'case_control', 'cross_sectional', 'retrospective_cohort', 'prospective_cohort', 'rct'.
        a (int): Exposed cases.
        b (int): Exposed controls.
        c (int): Unexposed cases.
        d (int): Unexposed controls.
        control_sampling_type (str, optional): Control group source.
            'population' (community matched) or 'hospital' (clinical controls). Defaults to 'population'.
        disease_is_rare (bool, optional): Whether disease prevalence in population is < 5-10%. Defaults to True.
        is_recall_bias_risk (bool, optional): Whether past exposure was gathered by retrospective self-report. Defaults to False.

    Returns:
        List[Dict[str, Any]]: Diagnostic evaluation cards containing:
            - 'name': Name of assumption / criterion.
            - 'status': 'PASS', 'WARNING', or 'FAIL'.
            - 'details': Plain-English evaluation text.
            - 'recommendation': Specific epidemiological guidance.
    """
    evaluations: List[Dict[str, Any]] = []

    # 1. Study Design Suitability
    if study_design.lower() in {"case_control", "cross_sectional"}:
        evaluations.append({
            "name": "Study Design Suitability for Odds Ratio",
            "status": "PASS",
            "details": f"Study design ({study_design.replace('_', ' ').title()}) selects participants based on outcome status or cross-sectional prevalence. Odds Ratio is the standard, mathematically valid effect measure.",
            "recommendation": "Odds Ratio (OR) is the primary appropriate measure of association."
        })
    elif study_design.lower() in {"prospective_cohort", "retrospective_cohort", "rct"}:
        evaluations.append({
            "name": "Study Design Suitability for Odds Ratio",
            "status": "PASS",
            "details": f"Study design ({study_design.replace('_', ' ').title()}) tracks exposure forward to incident outcomes. OR is valid (especially in multivariable logistic regression), though Relative Risk (RR) can also be directly computed.",
            "recommendation": "You may report OR (e.g. from logistic regression) or Relative Risk (RR) directly."
        })
    else:
        evaluations.append({
            "name": "Study Design Suitability for Odds Ratio",
            "status": "WARNING",
            "details": f"Unrecognized design ({study_design}). Ensure cases and controls are selected independently of exposure status.",
            "recommendation": "Confirm whether participant recruitment was outcome-first or exposure-first."
        })

    # 2. Rare Disease Assumption (Approximating Relative Risk)
    if disease_is_rare:
        evaluations.append({
            "name": "Rare Disease Assumption (OR ≈ RR)",
            "status": "PASS",
            "details": "The disease is rare in the source population (< 5–10% cumulative incidence). The Odds Ratio closely approximates the true population Relative Risk (OR ≈ RR).",
            "recommendation": "You may cautiously interpret the Odds Ratio as a surrogate for relative risk."
        })
    else:
        evaluations.append({
            "name": "Rare Disease Assumption (OR ≈ RR)",
            "status": "WARNING",
            "details": "The disease is common (> 10% prevalence/incidence). The Odds Ratio exaggerates the effect compared to Relative Risk (if harmful: OR > RR; if protective: OR < RR).",
            "recommendation": "Do NOT interpret the OR as Relative Risk. Report strictly as 'Odds Ratio' or consider log-binomial / Poisson regression."
        })

    # 3. Control Group Sampling (Selection / Berkson's Bias)
    if control_sampling_type.lower() == "population":
        evaluations.append({
            "name": "Control Selection Validity (Berkson's Bias)",
            "status": "PASS",
            "details": "Controls are sampled from the same source community that generated the cases (population-based sampling).",
            "recommendation": "Low risk of control selection bias; exposure frequency in controls represents baseline population."
        })
    else:
        evaluations.append({
            "name": "Control Selection Validity (Berkson's Bias)",
            "status": "WARNING",
            "details": "Hospital or clinic-based controls were used. If the control hospital diagnosis is also linked to the exposure, Berkson's bias can artificially inflate or deflate the Odds Ratio.",
            "recommendation": "Ensure hospital control diagnoses are strictly independent of the exposure under study."
        })

    # 4. Recall / Information Bias
    if not is_recall_bias_risk:
        evaluations.append({
            "name": "Exposure Measurement Validity (Recall Bias)",
            "status": "PASS",
            "details": "Exposure data was obtained from objective historical records (employment archives, prescription registries, biomarkers).",
            "recommendation": "Minimal risk of differential recall bias."
        })
    else:
        evaluations.append({
            "name": "Exposure Measurement Validity (Recall Bias)",
            "status": "WARNING",
            "details": "Exposure was gathered by retrospective self-report/interviews. Ill cases frequently ruminate on and over-report past exposures compared to healthy controls.",
            "recommendation": "Corroborate self-reported exposures with medical records or blind the interviewers to case/control status."
        })

    # 5. Minimum Cell Count Volume & Sparsity
    min_cell = min(a, b, c, d)
    if min_cell >= 5:
        evaluations.append({
            "name": "2x2 Cell Count Robustness (Sample Size)",
            "status": "PASS",
            "details": f"All four cells have at least 5 observations (minimum cell = {min_cell}). Woolf's log-scale Wald CI is robust.",
            "recommendation": "Sample size is adequate for asymptotic Wald confidence intervals and Chi-Square tests."
        })
    elif min_cell > 0:
        evaluations.append({
            "name": "2x2 Cell Count Robustness (Sample Size)",
            "status": "WARNING",
            "details": f"One or more cells has fewer than 5 observations (minimum cell = {min_cell}). Woolf's normal approximation may be slightly skewed.",
            "recommendation": "Cross-reference Wald 95% CI with Fisher's exact test p-value."
        })
    else:
        evaluations.append({
            "name": "2x2 Cell Count Robustness (Sample Size)",
            "status": "WARNING",
            "details": "One or more cells contains exactly zero events. Standard cross-product yields 0 or infinity.",
            "recommendation": "Automatic Haldane-Anscombe continuity correction (+0.5 added to all cells) has been applied."
        })

    return evaluations


def generate_odds_ratio_demo_data(
    scenario: str = "mesothelioma_asbestos",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic Case-Control data for Odds Ratio pedagogical demonstrations.

    Provides three realistic public health scenarios:
    1. 'mesothelioma_asbestos': Regional Mesothelioma & Historical Asbestos Case-Control Investigation (Harmful Exposure, Rare Disease).
    2. 'outbreak_investigation': County Foodborne Gastroenteritis (Salmonella) Outbreak Investigation (Acute Public Health).
    3. 'cardiovascular_diet': Inner City Hospital Acute Myocardial Infarction & Mediterranean Diet Case-Control Study (Protective Exposure).

    Args:
        scenario (str, optional): Scenario identifier. Defaults to 'mesothelioma_asbestos'.
        seed (int, optional): Random seed for reproducible individual microdata generation. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - df_contingency: Aggregated 2x2 contingency summary table.
            - df_microdata: Individual-level Case-Control participant microdata.
            - meta: Rich descriptive metadata, clinical takeaways, and disclaimer notices.
    """
    np.random.seed(seed)

    scenarios = {
        "mesothelioma_asbestos": {
            "title": "Regional Mesothelioma & Historical Asbestos Case-Control Investigation",
            "domain": "Occupational Oncology & Environmental Epidemiology",
            "description": "A hypothetical hospital- and community-matched Case-Control study across regional general hospitals (Inner City Hospital and All Meadows Community Hospital). Investigates historical occupational asbestos exposure in 75 verified malignant mesothelioma cases matched with 150 disease-free community controls.",
            "exposure_name": "Historical Heavy Asbestos Exposure (> 1 Year)",
            "exposure_levels": ("Asbestos Exposed", "Unexposed"),
            "outcome_name": "Malignant Mesothelioma Status",
            "outcome_levels": ("Cases (Mesothelioma +)", "Controls (Disease Free)"),
            "a": 52,    # Exposed cases
            "c": 23,    # Unexposed cases (Total cases = 75; Exposure % in cases = 69.3%)
            "b": 36,    # Exposed controls
            "d": 114,   # Unexposed controls (Total controls = 150; Exposure % in controls = 24.0%)
            "control_sampling_type": "population",
            "disease_is_rare": True,
            "is_recall_bias_risk": False,  # Verified via historic occupational employer registers
            "expected_effect": "Strong Harmful Association (OR ≈ 7.1)",
            "clinical_takeaway": "Cases have over 7 times higher odds of historical asbestos exposure than controls (OR = 7.16, 95% CI: [3.85, 13.31]). Because mesothelioma is exceptionally rare (< 0.01% population incidence), this Odds Ratio accurately reflects the true relative risk (OR ≈ RR).",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "outbreak_investigation": {
            "title": "County Local Food Festival Salmonella Outbreak Investigation",
            "domain": "Acute Field Epidemiology & Health Protection",
            "description": "A hypothetical acute outbreak investigation conducted by local health protection teams following a summer artisan market. Evaluates food exposure histories among 60 festival attendees who developed acute gastroenteritis (cases) versus 120 healthy attendees (controls).",
            "exposure_name": "Consumed Artisan Unpasteurized Soft Cheese",
            "exposure_levels": ("Ate Artisan Cheese", "Did Not Eat Cheese"),
            "outcome_name": "Acute Salmonella Enteritis",
            "outcome_levels": ("Cases (Ill Attendees)", "Controls (Well Attendees)"),
            "a": 46,    # Exposed cases (76.7%)
            "c": 14,    # Unexposed cases
            "b": 38,    # Exposed controls (31.7%)
            "d": 82,    # Unexposed controls
            "control_sampling_type": "population",
            "disease_is_rare": False,  # Common in the festival attendee cohort
            "is_recall_bias_risk": True,  # Self-reported 48-hour dietary recall
            "expected_effect": "Harmful Outbreak Exposure (OR ≈ 7.1)",
            "clinical_takeaway": "Attendees who developed illness had 7.09 times higher odds of eating the unpasteurized cheese than well attendees. This massive odds ratio rapidly pinpointed the contaminated batch for immediate product recall within 48 hours.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "cardiovascular_diet": {
            "title": "Inner City & All Meadows Hospitals Acute Myocardial Infarction Study",
            "domain": "Cardiovascular Nutritional Epidemiology",
            "description": "A hypothetical Case-Control study conducted across coronary care units at Inner City Hospital and All Meadows Community Hospital. Evaluates high adherence to a Mediterranean diet among 100 first-time acute myocardial infarction (heart attack) patients (cases) versus 200 orthopedic outpatients free of coronary disease (controls).",
            "exposure_name": "High Mediterranean Diet Adherence (Score ≥ 8)",
            "exposure_levels": ("High Diet Adherence", "Low/Average Adherence"),
            "outcome_name": "Acute Myocardial Infarction Status",
            "outcome_levels": ("Cases (Heart Attack +)", "Controls (Cardiac Free)"),
            "a": 25,    # Exposed cases (25.0%)
            "c": 75,    # Unexposed cases
            "b": 94,    # Exposed controls (47.0%)
            "d": 106,   # Unexposed controls
            "control_sampling_type": "hospital",  # Orthopedic clinic controls
            "disease_is_rare": True,
            "is_recall_bias_risk": True,  # Food frequency questionnaire recall
            "expected_effect": "Protective Nutritional Factor (OR ≈ 0.38)",
            "clinical_takeaway": "Heart attack cases had 62% lower odds of following a Mediterranean diet compared to controls (OR = 0.38, 95% CI: [0.22, 0.63]). High dietary adherence is strongly protective against acute coronary events.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "patient_feedback_survey": {
            "title": "Regional Healthcare NHS Patient Experience Survey (Youth vs. Adult Satisfaction Odds)",
            "domain": "Health Services Research & Cross-Sectional Survey Analytics",
            "description": "A cross-sectional patient experience survey evaluating feedback sentiment between young patients (aged <= 18) and adult patients (aged 19+). In cross-sectional designs, the Odds Ratio is mathematically termed the Prevalence Odds Ratio (POR) — the cross-product math is identical. However, because positive reviews are common (60-80%), the POR (2.67) is much higher than the Prevalence Ratio (1.33).",
            "exposure_name": "Demographic Cohort (Youth <= 18 vs. Adults 19+)",
            "exposure_levels": ("Young People (<= 18)", "Adults (19+)"),
            "outcome_name": "Overall Care Feedback Sentiment",
            "outcome_levels": ("Positive Feedback (+)", "Negative Feedback (-)"),
            "a": 160,   # Young positive (160 out of 200 = 80.0%)
            "c": 240,   # Adult positive (240 out of 400 = 60.0%)
            "b": 40,    # Young negative (40 out of 200 = 20.0%)
            "d": 160,   # Adult negative (160 out of 400 = 40.0%)
            "control_sampling_type": "population",
            "disease_is_rare": False,  # Common outcome (80% vs 60%)
            "is_recall_bias_risk": False,
            "expected_effect": "Higher Satisfaction Odds in Youth (POR = 2.67 vs PR = 1.33)",
            "clinical_takeaway": "The odds of leaving positive feedback among young patients were 2.67 times higher than among adult patients (Prevalence Odds Ratio = 2.67, 95% CI: [1.79, 3.97]). Because positive feedback is very common (80% vs 60%), the Rare Disease Assumption does NOT hold, causing the Prevalence Odds Ratio (2.67) to appear more than double the Prevalence Ratio (1.33).",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
    }

    if scenario not in scenarios:
        scenario = "mesothelioma_asbestos"

    cfg = scenarios[scenario]
    a, b, c, d = cfg["a"], cfg["b"], cfg["c"], cfg["d"]

    exp_label, unexp_label = cfg["exposure_levels"]
    out_pos_label, out_neg_label = cfg["outcome_levels"]

    # 1. Summary 2x2 Contingency Table (Case-Control layout)
    df_contingency = pd.DataFrame([
        {
            "Exposure_Status": f"🟢 {exp_label}",
            "Cases_Outcome_Pos": a,
            "Controls_Outcome_Neg": b,
            "Total_Exposed": a + b,
            "Exposure_Percentage": float(a / (a + b) * 100.0) if (a + b) > 0 else 0.0
        },
        {
            "Exposure_Status": f"⚪ {unexp_label}",
            "Cases_Outcome_Pos": c,
            "Controls_Outcome_Neg": d,
            "Total_Unexposed": c + d,
            "Exposure_Percentage": float(c / (c + d) * 100.0) if (c + d) > 0 else 0.0
        },
        {
            "Exposure_Status": "📊 Total (Marginals)",
            "Cases_Outcome_Pos": a + c,
            "Controls_Outcome_Neg": b + d,
            "Total_Exposed": a + b + c + d,
            "Exposure_Percentage": 100.0
        }
    ])

    # 2. Individual Participant Microdata
    participants = []
    is_survey = (scenario == "patient_feedback_survey")

    # Exposed Cases (a: Young Positive)
    for i in range(a):
        age_val = int(np.random.randint(14, 19)) if is_survey else int(np.random.normal(67, 6))
        participants.append({
            "Participant_ID": f"CASE_{i+1:04d}",
            "Study_Group": "Positive Feedback" if is_survey else "Case",
            "Group_Binary": 1,
            "Exposure_Status": exp_label,
            "Exposure_Binary": 1,
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.45 else "Female"
        })
    # Unexposed Cases (c: Adult Positive)
    for i in range(c):
        age_val = int(np.random.randint(19, 76)) if is_survey else int(np.random.normal(68, 6))
        participants.append({
            "Participant_ID": f"CASE_{a+i+1:04d}",
            "Study_Group": "Positive Feedback" if is_survey else "Case",
            "Group_Binary": 1,
            "Exposure_Status": unexp_label,
            "Exposure_Binary": 0,
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.48 else "Female"
        })
    # Exposed Controls (b: Young Negative)
    for i in range(b):
        age_val = int(np.random.randint(14, 19)) if is_survey else int(np.random.normal(66, 6))
        participants.append({
            "Participant_ID": f"CTRL_{i+1:04d}",
            "Study_Group": "Negative Feedback" if is_survey else "Control",
            "Group_Binary": 0,
            "Exposure_Status": exp_label,
            "Exposure_Binary": 1,
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.45 else "Female"
        })
    # Unexposed Controls (d: Adult Negative)
    for i in range(d):
        age_val = int(np.random.randint(19, 76)) if is_survey else int(np.random.normal(66, 6))
        participants.append({
            "Participant_ID": f"CTRL_{b+i+1:04d}",
            "Study_Group": "Negative Feedback" if is_survey else "Control",
            "Group_Binary": 0,
            "Exposure_Status": unexp_label,
            "Exposure_Binary": 0,
            "Age": age_val,
            "Sex": "Male" if np.random.rand() > 0.48 else "Female"
        })

    df_microdata = pd.DataFrame(participants)


    # Attach calculation results to metadata
    calcs = calculate_odds_ratio(a, b, c, d)
    meta = {
        **cfg,
        "scenario_key": scenario,
        "calcs": calcs
    }

    return df_contingency, df_microdata, meta


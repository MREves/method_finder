"""Mathematical & Statistical Engine for Diagnostic Predictive Values (PPV & NPV) and Bayes' Theorem.

Provides verified epidemiological calculations for:
1. Positive Predictive Value (PPV, Post-Test Probability of Disease given Positive Test):
   PPV = P(Disease+ | Test+) = TP / (TP + FP)
2. Negative Predictive Value (NPV, Post-Test Probability of No Disease given Negative Test):
   NPV = P(Disease- | Test-) = TN / (FN + TN)
3. False Discovery Rate (FDR = 1 - PPV = FP / (TP + FP)):
   Proportion of positive results that are false alarms.
4. False Omission Rate (FOR = 1 - NPV = FN / (FN + TN)):
   Proportion of negative results that are missed cases.
5. Bayes' Theorem Formulations (Decoupling Test Characteristics from Population Prevalence):
   PPV = (Sensitivity * Prevalence) / [Sensitivity * Prevalence + (1 - Specificity) * (1 - Prevalence)]
   NPV = (Specificity * (1 - Prevalence)) / [Specificity * (1 - Prevalence) + (1 - Sensitivity) * Prevalence]
6. The "False Positive Paradox" (Needle-in-a-Haystack Effect):
   Demonstrating why highly accurate tests (e.g. 99% Sensitivity, 99% Specificity) yield low PPV (< 10%)
   in low-prevalence (< 0.1%) screening populations.
7. Wilson Score 95% Confidence Intervals for Direct Sample Cohorts.
8. Mercaldo, Lau & Zhou (2007) Logit-Scale Delta Confidence Intervals for Model-Based Bayes Estimates.
9. Prerequisite evaluation: Setting/prevalence generalizability, verification/work-up bias,
   incorporation bias, and predictive marginal stability.
10. Realistic synthetic scenarios (Asymptomatic Mobile Breast Screening, Emergency Department
    High-Sensitivity Cardiac Troponin Chest Pain Triage, Community Rapid Respiratory Lateral Flow).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.epi_methods.diagnostic_accuracy import calculate_wilson_score_interval


def calculate_predictive_values(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    prevalence_override: Optional[float] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Calculates comprehensive predictive values (PPV, NPV), error fractions, and Bayes' Theorem adjustments.

    In clinical epidemiology:
    - Sensitivity & Specificity are properties of the test instrument.
    - Predictive Values (PPV & NPV) describe patient post-test probabilities and DEPEND DIRECTLY
      ON DISEASE PREVALENCE in the population being screened.

    Args:
        tp (int): True Positives.
        fp (int): False Positives.
        fn (int): False Negatives.
        tn (int): True Negatives.
        prevalence_override (Optional[float], optional): External true population disease prevalence (0.0 to 1.0).
            If None, the natural sample prevalence (tp + fn) / N is used.
        alpha (float, optional): Significance level for confidence intervals. Defaults to 0.05 (95% CI).

    Returns:
        Dict[str, Any]: Complete predictive values dictionary containing:
            - 'tp', 'fp', 'fn', 'tn': Quadrant counts.
            - 'total_diseased', 'total_healthy', 'total_test_positive', 'total_test_negative', 'total_sample': Marginals.
            - 'sample_prevalence': Observed sample disease prevalence.
            - 'effective_prevalence': Disease prevalence used for Bayes' Theorem.
            - 'sensitivity', 'specificity': Intrinsic test characteristics.
            - 'ppv': Positive Predictive Value (point estimate).
            - 'ppv_ci_lower', 'ppv_ci_upper': 95% Confidence interval for PPV.
            - 'npv': Negative Predictive Value (point estimate).
            - 'npv_ci_lower', 'npv_ci_upper': 95% Confidence interval for NPV.
            - 'fdr': False Discovery Rate (1 - PPV).
            - 'for_rate': False Omission Rate (1 - NPV).
            - 'bayes_applied': Whether an external prevalence override was modeled.
            - 'false_positives_per_true_positive': Ratio of FP to TP (Quantifies false alarm burden).
    """
    total_diseased = tp + fn
    total_healthy = tn + fp
    total_test_pos = tp + fp
    total_test_neg = fn + tn
    total_sample = tp + fp + fn + tn

    # Calculate intrinsic sensitivity and specificity
    sens = float(tp) / float(total_diseased) if total_diseased > 0 else 0.0
    spec = float(tn) / float(total_healthy) if total_healthy > 0 else 0.0
    sample_prev = float(total_diseased) / float(total_sample) if total_sample > 0 else 0.0

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    if prevalence_override is None:
        # 1. Natural Sample Cohort (Direct Cross-Sectional / Prospective Cohort)
        effective_prev = sample_prev
        bayes_applied = False

        # Direct PPV = TP / (TP + FP)
        if total_test_pos > 0:
            ppv = float(tp) / float(total_test_pos)
            ppv_ll, ppv_ul = calculate_wilson_score_interval(tp, total_test_pos, alpha)
        else:
            ppv, ppv_ll, ppv_ul = 0.0, 0.0, 0.0

        # Direct NPV = TN / (FN + TN)
        if total_test_neg > 0:
            npv = float(tn) / float(total_test_neg)
            npv_ll, npv_ul = calculate_wilson_score_interval(tn, total_test_neg, alpha)
        else:
            npv, npv_ll, npv_ul = 0.0, 0.0, 0.0

    else:
        # 2. Bayes' Theorem Model-Based Predictive Values (External Population Prevalence)
        p = float(np.clip(prevalence_override, 1e-7, 1.0 - 1e-7))
        effective_prev = p
        bayes_applied = True

        # Numerator & Denominator for Bayes PPV
        # PPV = [Se * p] / [Se * p + (1 - Sp) * (1 - p)]
        p_t_pos = (sens * p) + ((1.0 - spec) * (1.0 - p))
        if p_t_pos > 0:
            ppv = float((sens * p) / p_t_pos)
        else:
            ppv = 0.0

        # Numerator & Denominator for Bayes NPV
        # NPV = [Sp * (1 - p)] / [Sp * (1 - p) + (1 - Se) * p]
        p_t_neg = (spec * (1.0 - p)) + ((1.0 - sens) * p)
        if p_t_neg > 0:
            npv = float((spec * (1.0 - p)) / p_t_neg)
        else:
            npv = 0.0

        # Mercaldo, Lau & Zhou (2007) Delta Method Confidence Intervals for Logit-PPV and Logit-NPV
        # logit(PPV) = ln(p / (1 - p)) + ln(Se / (1 - Sp))
        # Var(logit(PPV)) = Var(ln(Se)) + Var(ln(1 - Sp)) = (1 - Se)/(total_diseased * Se) + Sp / (total_healthy * (1 - Sp))
        if total_diseased > 0 and total_healthy > 0 and 0.0 < sens < 1.0 and 0.0 < spec < 1.0:
            var_logit_ppv = ((1.0 - sens) / (total_diseased * sens)) + (spec / (total_healthy * (1.0 - spec)))
            se_logit_ppv = float(np.sqrt(max(1e-9, var_logit_ppv)))
            logit_ppv = float(np.log(ppv / (1.0 - ppv))) if 0.0 < ppv < 1.0 else 0.0
            ppv_ll = float(1.0 / (1.0 + np.exp(-(logit_ppv - z_crit * se_logit_ppv))))
            ppv_ul = float(1.0 / (1.0 + np.exp(-(logit_ppv + z_crit * se_logit_ppv))))

            # logit(NPV) = ln((1 - p) / p) + ln(Sp / (1 - Se))
            # Var(logit(NPV)) = Var(ln(Sp)) + Var(ln(1 - Se)) = (1 - Sp)/(total_healthy * Sp) + Se / (total_diseased * (1 - Se))
            var_logit_npv = ((1.0 - spec) / (total_healthy * spec)) + (sens / (total_diseased * (1.0 - sens)))
            se_logit_npv = float(np.sqrt(max(1e-9, var_logit_npv)))
            logit_npv = float(np.log(npv / (1.0 - npv))) if 0.0 < npv < 1.0 else 0.0
            npv_ll = float(1.0 / (1.0 + np.exp(-(logit_npv - z_crit * se_logit_npv))))
            npv_ul = float(1.0 / (1.0 + np.exp(-(logit_npv + z_crit * se_logit_npv))))
        else:
            ppv_ll, ppv_ul = ppv, ppv
            npv_ll, npv_ul = npv, npv

    fdr = 1.0 - ppv
    for_rate = 1.0 - npv

    # Ratio of False Positives to True Positives
    # In low-prevalence screening, this illustrates how many healthy people are harmed per true case detected
    fp_to_tp_ratio = float(fdr / ppv) if ppv > 0 else np.nan

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_diseased": int(total_diseased),
        "total_healthy": int(total_healthy),
        "total_test_positive": int(total_test_pos),
        "total_test_negative": int(total_test_neg),
        "total_sample": int(total_sample),
        "sample_prevalence": float(sample_prev),
        "effective_prevalence": float(effective_prev),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "ppv_ci_lower": float(ppv_ll),
        "ppv_ci_upper": float(ppv_ul),
        "npv": float(npv),
        "npv_ci_lower": float(npv_ll),
        "npv_ci_upper": float(npv_ul),
        "fdr": float(fdr),
        "for_rate": float(for_rate),
        "bayes_applied": bayes_applied,
        "false_positives_per_true_positive": float(fp_to_tp_ratio)
    }


def evaluate_predictive_value_prerequisites(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    is_prevalence_representative: bool = True,
    is_independent_gold_standard: bool = True,
    is_verification_bias_controlled: bool = True
) -> List[Dict[str, Any]]:
    """Evaluates methodological validity, spectrum alignment, and verification bias for predictive values.

    Args:
        tp (int): True Positives.
        fp (int): False Positives.
        fn (int): False Negatives.
        tn (int): True Negatives.
        is_prevalence_representative (bool, optional): Whether disease prevalence matches intended clinical setting. Defaults to True.
        is_independent_gold_standard (bool, optional): Whether reference standard is independent of test. Defaults to True.
        is_verification_bias_controlled (bool, optional): Whether all participants (or random sample) received gold standard. Defaults to True.

    Returns:
        List[Dict[str, Any]]: Methodological diagnostic cards.
    """
    evaluations: List[Dict[str, Any]] = []

    # 1. Prevalence & Setting Generalizability (Spectrum / Referral Filter Bias)
    if is_prevalence_representative:
        evaluations.append({
            "name": "Prevalence & Setting Generalizability",
            "status": "PASS",
            "details": "The disease prevalence in this evaluation reflects the intended real-world practice setting (e.g. primary care, screening clinic, or emergency triage).",
            "recommendation": "Calculated PPV and NPV can be directly communicated to patients in this clinical setting."
        })
    else:
        evaluations.append({
            "name": "Prevalence & Setting Generalizability",
            "status": "FAIL",
            "details": "Setting mismatch: Study prevalence (e.g. 50% in a hospital specialist referral unit) differs drastically from general practice (e.g. 0.5%). Direct PPV will be severely misleading.",
            "recommendation": "Use Bayes' Theorem to adjust PPV and NPV to the true target population prevalence."
        })

    # 2. Verification / Work-Up Bias
    if is_verification_bias_controlled:
        evaluations.append({
            "name": "Verification / Work-Up Bias",
            "status": "PASS",
            "details": "Gold standard verification was conducted on both test-positive and test-negative individuals, or a representative random sample of test-negatives.",
            "recommendation": "Negative Predictive Value (NPV) and False Negative counts are unbiased."
        })
    else:
        evaluations.append({
            "name": "Verification / Work-Up Bias",
            "status": "WARNING",
            "details": "Only patients testing positive received the gold standard verification biopsy. Patients testing negative were assumed disease-free without verification.",
            "recommendation": "NPV will appear artificially near 100%. Follow up test-negatives over time or conduct partial verification modeling."
        })

    # 3. Reference Standard Independence (Incorporation Bias)
    if is_independent_gold_standard:
        evaluations.append({
            "name": "Reference Standard Independence",
            "status": "PASS",
            "details": "The index test is strictly independent of the reference standard diagnostic definition.",
            "recommendation": "Zero circular verification risk; predictive values are objective."
        })
    else:
        evaluations.append({
            "name": "Reference Standard Independence",
            "status": "FAIL",
            "details": "The index test is part of the gold standard criteria (Incorporation Bias), artificially inflating PPV and NPV.",
            "recommendation": "Re-evaluate the index test against an independent diagnostic standard."
        })

    # 4. Cell Size & Predictive Precision
    total_test_pos = tp + fp
    total_test_neg = fn + tn
    min_test_marginal = min(total_test_pos, total_test_neg)

    if min_test_marginal >= 40:
        evaluations.append({
            "name": "Test Marginal Count Stability",
            "status": "PASS",
            "details": f"Adequate volume of test-positive (n={total_test_pos}) and test-negative (n={total_test_neg}) patients. Confidence intervals for PPV and NPV are narrow and precise.",
            "recommendation": "Post-test probabilities have high clinical reliability."
        })
    elif min_test_marginal >= 10:
        evaluations.append({
            "name": "Test Marginal Count Stability",
            "status": "WARNING",
            "details": f"Moderate marginal volume (minimum test stratum = {min_test_marginal}). Confidence intervals will be wider.",
            "recommendation": "Interpret the full width of the Wilson / Mercaldo 95% confidence intervals."
        })
    else:
        evaluations.append({
            "name": "Test Marginal Count Stability",
            "status": "FAIL",
            "details": f"Very low test outcome volume (minimum stratum = {min_test_marginal} < 10). Point estimates may fluctuate substantially.",
            "recommendation": "Expand the sample size before utilizing predictive values for clinical protocol design."
        })

    return evaluations


def generate_predictive_values_demo_data(
    scenario: str = "breast_screening_low_prev",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic patient-level and 2x2 matrix data for Predictive Values and Bayes' Theorem demonstrations.

    Provides three realistic scenarios:
    1. 'breast_screening_low_prev': Asymptomatic Population Mammography Screening (The False Positive Paradox: Low Prevalence 0.5%, Sens = 85.0%, Spec = 95.0% -> PPV = 7.87%, ~11.7 False Positives per True Case!).
    2. 'troponin_chest_pain_high_prev': Inner City Hospital Emergency Department Acute Coronary Syndrome Triage (High Prevalence 25.0%, Sens = 92.0%, Spec = 90.0% -> PPV = 75.4%, NPV = 97.1%).
    3. 'lateral_flow_flu_surveillance': All Meadows Community Winter Respiratory Lateral Flow Surveillance (Moderate Prevalence 5.0%, Sens = 75.0%, Spec = 98.5% -> PPV = 72.5%, NPV = 98.7%).

    Args:
        scenario (str, optional): Scenario identifier. Defaults to 'breast_screening_low_prev'.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - df_contingency: 2x2 matrix with row and column totals.
            - df_microdata: Individual patient synthetic microdata.
            - meta: Scenario descriptive metadata, clinical takeaways, and disclaimer notices.
    """
    np.random.seed(seed)

    scenarios = {
        "breast_screening_low_prev": {
            "title": "County Mobile Breast Screening Service: The False Positive Paradox in Low-Prevalence Populations",
            "domain": "Preventive Oncology & Population Screening",
            "description": "A hypothetical screening evaluation of 2,000 asymptomatic women aged 50–70 attending County mobile breast screening units. In this unselected screening population, the true prevalence of invasive breast cancer is low (0.5% = 10 cases per 2,000 women). Demonstrates the classical 'False Positive Paradox': even with 95.0% Specificity, healthy women vastly outnumber cancer cases, causing 92% of all abnormal screening mammograms to be false alarms.",
            "test_name": "Digital Screening Mammography",
            "condition_name": "Biopsy-Confirmed Invasive Carcinoma",
            "test_levels": ("Mammogram Abnormal (Recall for Assessment)", "Mammogram Normal (Routine Recall)"),
            "condition_levels": ("Confirmed Carcinoma (Biopsy +)", "Healthy / Benign (Biopsy -)"),
            "tp": 9,               # True Positives (85% of 10 cases ≈ 9)
            "fp": 100,             # False Positives (5% of 1,990 healthy = 99.5 ≈ 100)
            "fn": 1,               # False Negatives (15% of 10 = 1)
            "tn": 1890,            # True Negatives (95% of 1,990 = 1890.5 ≈ 1890)
            "prevalence_override": None,
            "clinical_takeaway": "Although mammography achieved high Sensitivity (90.0%) and high Specificity (95.0%), the Positive Predictive Value (PPV) is only 8.3% (95% CI: 4.4% - 15.0%). For every 1 woman with true cancer caught, approximately 11 healthy women receive an alarming callback for benign changes. Conversely, the Negative Predictive Value (NPV) is 99.9%, meaning a normal mammogram offers near-certain reassurance.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "troponin_chest_pain_high_prev": {
            "title": "Inner City Hospital Emergency Department: High-Sensitivity Cardiac Troponin Triage",
            "domain": "Cardiology & Emergency Medicine",
            "description": "A hypothetical diagnostic investigation of 1,000 adult patients presenting with acute retrosternal chest pain to Inner City Hospital Emergency Department. Because this is an enriched, symptomatic cohort, the pre-test probability (prevalence) of Acute Coronary Syndrome (ACS) is high (25.0%). Demonstrates how high disease prevalence transforms high Specificity into high PPV.",
            "test_name": "High-Sensitivity Cardiac Troponin T Assay",
            "condition_name": "Coronary Angiography-Confirmed Myocardial Infarction",
            "test_levels": ("Troponin Elevated (> 14 ng/L)", "Troponin Normal (<= 14 ng/L)"),
            "condition_levels": ("Confirmed ACS / Infarction (+)", "Non-Cardiac Chest Pain (-)"),
            "tp": 230,             # True Positives (92% of 250 diseased = 230)
            "fp": 75,              # False Positives (10% of 750 healthy = 75)
            "fn": 20,              # False Negatives (8% of 250 diseased = 20)
            "tn": 675,             # True Negatives (90% of 750 healthy = 675)
            "prevalence_override": None,
            "clinical_takeaway": "In this symptomatic emergency setting (Prevalence = 25.0%), the high Specificity (90.0%) and Sensitivity (92.0%) yield a robust Positive Predictive Value of 75.4% (95% CI: 70.3% - 80.0%) and an exceptional Negative Predictive Value of 97.1% (95% CI: 95.6% - 98.2%). Elevated troponin strongly rules in acute infarction, while normal troponin safely identifies patients eligible for early discharge.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "lateral_flow_flu_surveillance": {
            "title": "All Meadows Community Winter Respiratory Lateral Flow Surveillance",
            "domain": "Public Health & Infectious Disease Surveillance",
            "description": "A hypothetical winter surveillance study of 1,500 symptomatic care home staff and residents across All Meadows region during an influenza surge. The community prevalence of influenza A/B is moderate (5.0%). Demonstrates the balance between rapid test specificity (98.5%) and moderate sensitivity (75.0%).",
            "test_name": "Rapid Influenza Lateral Flow Antigen Assay",
            "condition_name": "Laboratory Multiplex RT-PCR Confirmed Influenza",
            "test_levels": ("Lateral Flow Positive", "Lateral Flow Negative"),
            "condition_levels": ("Lab PCR Positive (+)", "Lab PCR Negative (-)"),
            "tp": 56,              # True Positives (75% of 75 diseased = 56)
            "fp": 21,              # False Positives (1.5% of 1,425 healthy = 21)
            "fn": 19,              # False Negatives (25% of 75 diseased = 19)
            "tn": 1404,            # True Negatives (98.5% of 1,425 healthy = 1404)
            "prevalence_override": None,
            "clinical_takeaway": "With an ultra-high Specificity of 98.5% and moderate community prevalence (5.0%), the Positive Predictive Value reaches 72.7% (95% CI: 61.8% - 81.5%). A positive lateral flow test justifies immediate infection control isolation. The Negative Predictive Value is 98.7%, though clinicians must remain alert to false negatives in high-risk frail individuals.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
    }

    if scenario not in scenarios:
        scenario = "breast_screening_low_prev"

    cfg = scenarios[scenario]
    tp = int(cfg["tp"])
    fp = int(cfg["fp"])
    fn = int(cfg["fn"])
    tn = int(cfg["tn"])

    test_pos_label, test_neg_label = cfg["test_levels"]
    cond_pos_label, cond_neg_label = cfg["condition_levels"]

    # 1. Aggregated 2x2 Contingency Matrix with Row Marginals (Predictive View)
    df_contingency = pd.DataFrame([
        {
            "Test Result": f"🔴 {test_pos_label}",
            f"{cond_pos_label}": tp,
            f"{cond_neg_label}": fp,
            "Total Test Positives": tp + fp,
            "Row Predictive Metric": f"PPV = {tp / (tp + fp) * 100.0:.1f}% (FDR: {fp / (tp + fp) * 100.0:.1f}%)"
        },
        {
            "Test Result": f"🟢 {test_neg_label}",
            f"{cond_pos_label}": fn,
            f"{cond_neg_label}": tn,
            "Total Test Negatives": fn + tn,
            "Row Predictive Metric": f"NPV = {tn / (fn + tn) * 100.0:.1f}% (FOR: {fn / (fn + tn) * 100.0:.1f}%)"
        },
        {
            "Test Result": "📊 Total Cohort",
            f"{cond_pos_label}": tp + fn,
            f"{cond_neg_label}": fp + tn,
            "Total Test Positives": tp + fp + fn + tn,
            "Row Predictive Metric": f"Sample Prevalence = {(tp + fn) / (tp + fp + fn + tn) * 100.0:.2f}%"
        }
    ])

    # 2. Individual Patient Synthetic Microdata
    patients = []
    # TP
    for i in range(min(tp, 120)):
        patients.append({
            "Patient_ID": f"PT_TP_{i+1:04d}",
            "Test_Result": test_pos_label,
            "Test_Binary": 1,
            "True_Condition": cond_pos_label,
            "Condition_Binary": 1,
            "Classification": "True Positive (TP)",
            "Predictive_Outcome": "Correct Positive Call",
            "Age": int(np.random.normal(61, 9)),
            "Sex": "Female" if "Breast" in cfg["title"] else ("Female" if np.random.rand() > 0.5 else "Male")
        })
    # FP
    for i in range(min(fp, 100)):
        patients.append({
            "Patient_ID": f"PT_FP_{i+1:04d}",
            "Test_Result": test_pos_label,
            "Test_Binary": 1,
            "True_Condition": cond_neg_label,
            "Condition_Binary": 0,
            "Classification": "False Positive (FP)",
            "Predictive_Outcome": "False Alarm (Type I Error)",
            "Age": int(np.random.normal(57, 10)),
            "Sex": "Female" if "Breast" in cfg["title"] else ("Female" if np.random.rand() > 0.5 else "Male")
        })
    # FN
    for i in range(min(fn, 80)):
        patients.append({
            "Patient_ID": f"PT_FN_{i+1:04d}",
            "Test_Result": test_neg_label,
            "Test_Binary": 0,
            "True_Condition": cond_pos_label,
            "Condition_Binary": 1,
            "Classification": "False Negative (FN)",
            "Predictive_Outcome": "Missed Case (Type II Error)",
            "Age": int(np.random.normal(63, 8)),
            "Sex": "Female" if "Breast" in cfg["title"] else ("Female" if np.random.rand() > 0.5 else "Male")
        })
    # TN
    for i in range(min(tn, 200)):
        patients.append({
            "Patient_ID": f"PT_TN_{i+1:04d}",
            "Test_Result": test_neg_label,
            "Test_Binary": 0,
            "True_Condition": cond_neg_label,
            "Condition_Binary": 0,
            "Classification": "True Negative (TN)",
            "Predictive_Outcome": "Correct Negative Clearance",
            "Age": int(np.random.normal(56, 11)),
            "Sex": "Female" if "Breast" in cfg["title"] else ("Female" if np.random.rand() > 0.5 else "Male")
        })

    df_microdata = pd.DataFrame(patients)
    calcs = calculate_predictive_values(tp, fp, fn, tn, prevalence_override=cfg["prevalence_override"])

    meta = {
        **cfg,
        "scenario_key": scenario,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "calcs": calcs
    }

    return df_contingency, df_microdata, meta


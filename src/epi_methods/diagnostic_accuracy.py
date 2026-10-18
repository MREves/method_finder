"""Mathematical & Statistical Engine for Diagnostic Test Accuracy & Screening Evaluation.

Provides verified epidemiological and clinical diagnostic calculations for:
1. Sensitivity (True Positive Rate, TPR, Recall):
   Sensitivity = TP / (TP + FN) = a / (a + c)
2. Specificity (True Negative Rate, TNR):
   Specificity = TN / (TN + FP) = d / (b + d)
3. False Positive Rate (FPR, Type I Error):
   FPR = FP / (TN + FP) = b / (b + d) = 1 - Specificity
4. False Negative Rate (FNR, Type II Error, Miss Rate):
   FNR = FN / (TP + FN) = c / (a + c) = 1 - Sensitivity
5. Overall Diagnostic Accuracy:
   Accuracy = (TP + TN) / (TP + FP + FN + TN) = (a + d) / N
6. Youden's J Statistic (Youden's Index / Informedness):
   J = Sensitivity + Specificity - 1 = (TPR - FPR)
7. Diagnostic Odds Ratio (DOR):
   DOR = (TP * TN) / (FP * FN) = (a * d) / (b * c)
8. Wilson Score 95% Confidence Intervals for Proportions:
   CLSI/CDC gold standard interval for binomial parameters, avoiding Wald overshoot near 0 and 1.
9. Prerequisite and methodological diagnostic evaluation engine (Gold standard verification,
   incorporation bias, diagnostic review blinding, spectrum bias, cell sample size).
10. Realistic clinical and public health synthetic scenarios (All Meadows AI Retinopathy Screening,
    County Mobile Mammography, Rapid RSV Lateral Flow Antigen in Pediatric A&E).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_wilson_score_interval(
    k: int,
    n: int,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """Calculates the Wilson score confidence interval for a binomial proportion.

    Recommended by CLSI EP12-A2 and CDC for diagnostic test sensitivity and specificity.
    Unlike the asymptotic Wald normal approximation (p ± 1.96 * sqrt(p(1-p)/n)), the
    Wilson score interval never overshoots the [0, 1] interval and maintains robust
    coverage probabilities even with small sample sizes or proportions near 100%.

    Formula:
        center = (p + z^2 / (2n)) / (1 + z^2 / n)
        half_width = (z * sqrt(p(1 - p)/n + z^2 / (4n^2))) / (1 + z^2 / n)
        CI = [center - half_width, center + half_width]

    Args:
        k (int): Number of successes (e.g. True Positives or True Negatives).
        n (int): Total trials (e.g. Total Diseased or Total Healthy).
        alpha (float, optional): Significance level. Defaults to 0.05 (95% CI).

    Returns:
        Tuple[float, float]: (lower_bound, upper_bound) between 0.0 and 1.0.
    """
    if n <= 0:
        return (0.0, 0.0)

    p = float(k) / float(n)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_sq = z * z

    denominator = 1.0 + (z_sq / float(n))
    center = (p + (z_sq / (2.0 * float(n)))) / denominator
    spread = (z * np.sqrt((p * (1.0 - p) / float(n)) + (z_sq / (4.0 * float(n) * float(n))))) / denominator

    lower = max(0.0, float(center - spread))
    upper = min(1.0, float(center + spread))
    return (lower, upper)


def calculate_diagnostic_accuracy(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Calculates comprehensive diagnostic accuracy metrics, error rates, and Wilson 95% CIs.

    A classical 2x2 diagnostic confusion matrix is structured as:
                         Disease Present (D+)    Disease Absent (D-)       Test Marginals
    Test Positive (T+)     True Positive (TP)    False Positive (FP)    Total Test Pos = TP + FP
    Test Negative (T-)     False Negative (FN)   True Negative (TN)     Total Test Neg = FN + TN
    Condition Marginals    Total Diseased = D+   Total Healthy = D-     Grand Total N = TP+FP+FN+TN

    Mathematical Formulations:
        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        False Positive Rate (FPR) = FP / (TN + FP) = 1 - Specificity
        False Negative Rate (FNR) = FN / (TP + FN) = 1 - Sensitivity
        Accuracy = (TP + TN) / N
        Youden's Index (J) = Sensitivity + Specificity - 1
        Diagnostic Odds Ratio (DOR) = (TP * TN) / (FP * FN)

    Args:
        tp (int): True Positives (Diseased patients with positive test).
        fp (int): False Positives (Healthy individuals with positive test; Type I error).
        fn (int): False Negatives (Diseased patients with negative test; Type II error / Miss rate).
        tn (int): True Negatives (Healthy individuals with negative test).
        alpha (float, optional): Significance level. Defaults to 0.05 (95% CI).

    Returns:
        Dict[str, Any]: Complete diagnostic performance dictionary containing:
            - 'tp', 'fp', 'fn', 'tn': Cell counts.
            - 'total_diseased': TP + FN.
            - 'total_healthy': TN + FP.
            - 'total_test_positive': TP + FP.
            - 'total_test_negative': FN + TN.
            - 'total_sample': Total N.
            - 'prevalence': Sample disease prevalence (total_diseased / total_sample).
            - 'sensitivity': Point estimate of Sensitivity.
            - 'sensitivity_ci_lower', 'sensitivity_ci_upper': Wilson 95% CI for Sensitivity.
            - 'specificity': Point estimate of Specificity.
            - 'specificity_ci_lower', 'specificity_ci_upper': Wilson 95% CI for Specificity.
            - 'fpr': False Positive Rate (1 - Specificity).
            - 'fnr': False Negative Rate (1 - Sensitivity).
            - 'accuracy': Overall diagnostic accuracy.
            - 'accuracy_ci_lower', 'accuracy_ci_upper': Wilson 95% CI for Accuracy.
            - 'youdens_j': Youden's Index J (Informedness).
            - 'dor': Diagnostic Odds Ratio (point estimate).
            - 'dor_ci_lower', 'dor_ci_upper': Log-scale 95% Wald CI for DOR.
    """
    applied_cc = (tp == 0 or fp == 0 or fn == 0 or tn == 0)
    tp_c = float(tp) + (0.5 if applied_cc else 0.0)
    fp_c = float(fp) + (0.5 if applied_cc else 0.0)
    fn_c = float(fn) + (0.5 if applied_cc else 0.0)
    tn_c = float(tn) + (0.5 if applied_cc else 0.0)

    total_diseased = tp + fn
    total_healthy = tn + fp
    total_test_pos = tp + fp
    total_test_neg = fn + tn
    total_sample = tp + fp + fn + tn

    # 1. Sensitivity & Wilson CI
    if total_diseased > 0:
        sens = float(tp) / float(total_diseased)
        sens_ll, sens_ul = calculate_wilson_score_interval(tp, total_diseased, alpha)
    else:
        sens, sens_ll, sens_ul = 0.0, 0.0, 0.0

    # 2. Specificity & Wilson CI
    if total_healthy > 0:
        spec = float(tn) / float(total_healthy)
        spec_ll, spec_ul = calculate_wilson_score_interval(tn, total_healthy, alpha)
    else:
        spec, spec_ll, spec_ul = 0.0, 0.0, 0.0

    # 3. Error Rates
    fpr = 1.0 - spec
    fnr = 1.0 - sens

    # 4. Overall Accuracy & Wilson CI
    if total_sample > 0:
        acc = float(tp + tn) / float(total_sample)
        acc_ll, acc_ul = calculate_wilson_score_interval(tp + tn, total_sample, alpha)
        sample_prev = float(total_diseased) / float(total_sample)
    else:
        acc, acc_ll, acc_ul = 0.0, 0.0, 0.0
        sample_prev = 0.0

    # 5. Youden's Index J
    youdens_j = float(sens + spec - 1.0)

    # 6. Diagnostic Odds Ratio (DOR) & Log-Scale CI
    if fp_c > 0 and fn_c > 0 and tp_c > 0 and tn_c > 0:
        dor = float((tp_c * tn_c) / (fp_c * fn_c))
        ln_dor = float(np.log(dor))
        se_ln_dor = float(np.sqrt((1.0 / tp_c) + (1.0 / fp_c) + (1.0 / fn_c) + (1.0 / tn_c)))
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        dor_ll = float(np.exp(ln_dor - z_crit * se_ln_dor))
        dor_ul = float(np.exp(ln_dor + z_crit * se_ln_dor))
    else:
        dor, se_ln_dor, dor_ll, dor_ul = np.nan, np.nan, np.nan, np.nan

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
        "prevalence": float(sample_prev),
        "sensitivity": float(sens),
        "sensitivity_ci_lower": float(sens_ll),
        "sensitivity_ci_upper": float(sens_ul),
        "specificity": float(spec),
        "specificity_ci_lower": float(spec_ll),
        "specificity_ci_upper": float(spec_ul),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "accuracy": float(acc),
        "accuracy_ci_lower": float(acc_ll),
        "accuracy_ci_upper": float(acc_ul),
        "youdens_j": float(youdens_j),
        "dor": float(dor),
        "dor_ci_lower": float(dor_ll),
        "dor_ci_upper": float(dor_ul),
        "continuity_applied": applied_cc
    }


def evaluate_diagnostic_prerequisites(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    is_gold_standard_independent: bool = True,
    is_blinded_interpretation: bool = True,
    is_spectrum_representative: bool = True
) -> List[Dict[str, Any]]:
    """Evaluates methodological validity, diagnostic bias risks, and sample adequacy.

    Evaluates key diagnostic accuracy principles from the STARD (Standards for Reporting
    Diagnostic Accuracy Studies) and QUADAS-2 (Quality Assessment of Diagnostic Accuracy Studies) frameworks:
    1. Gold Standard Independence (Verification & Incorporation Bias).
    2. Blinded Diagnostic Interpretation (Review Bias).
    3. Clinical Spectrum Representativeness (Spectrum Bias).
    4. Cell Frequency and Sample Size Stability.

    Args:
        tp (int): True Positives.
        fp (int): False Positives.
        fn (int): False Negatives.
        tn (int): True Negatives.
        is_gold_standard_independent (bool, optional): Whether reference standard is independent of index test. Defaults to True.
        is_blinded_interpretation (bool, optional): Whether interpreters were blinded to index/gold results. Defaults to True.
        is_spectrum_representative (bool, optional): Whether sample reflects target clinical spectrum. Defaults to True.

    Returns:
        List[Dict[str, Any]]: Diagnostic evaluation cards containing:
            - 'name': Name of criterion / bias risk.
            - 'status': 'PASS', 'WARNING', or 'FAIL'.
            - 'details': Plain-English evaluation text.
            - 'recommendation': Specific epidemiological guidance.
    """
    evaluations: List[Dict[str, Any]] = []

    # 1. Gold Standard Reference Independence (Incorporation Bias)
    if is_gold_standard_independent:
        evaluations.append({
            "name": "Reference Standard Independence",
            "status": "PASS",
            "details": "The index test is completely independent of the reference Gold Standard (e.g. imaging test evaluated against independent histopathology).",
            "recommendation": "Zero risk of incorporation bias; accuracy metrics are objective."
        })
    else:
        evaluations.append({
            "name": "Reference Standard Independence",
            "status": "FAIL",
            "details": "The index test is part of the reference standard definition (Incorporation Bias). This creates circular verification and artificially inflates Sensitivity and Specificity.",
            "recommendation": "Re-evaluate the index test against a strictly independent, external reference standard."
        })

    # 2. Blinded Diagnostic Review (Diagnostic Review Bias)
    if is_blinded_interpretation:
        evaluations.append({
            "name": "Blinded Diagnostic Review",
            "status": "PASS",
            "details": "Readers and laboratory scientists interpreted index test results with zero knowledge of the patient's reference standard status (and vice versa).",
            "recommendation": "Subjective interpretation bias is effectively eliminated."
        })
    else:
        evaluations.append({
            "name": "Blinded Diagnostic Review",
            "status": "WARNING",
            "details": "Readers knew the patient's true clinical condition prior to scoring the test (Diagnostic Review Bias). Borderline results may have been swayed.",
            "recommendation": "Implement blinded re-reading of borderline or indeterminate test runs."
        })

    # 3. Clinical Spectrum Representativeness (Spectrum Bias)
    if is_spectrum_representative:
        evaluations.append({
            "name": "Clinical Spectrum Representativeness",
            "status": "PASS",
            "details": "The study population reflects the intended clinical target demographic, including mild, early-stage, and severe cases, alongside healthy and symptomatic non-cases.",
            "recommendation": "Sensitivity and Specificity can be reliably generalized to real-world clinical screening."
        })
    else:
        evaluations.append({
            "name": "Clinical Spectrum Representativeness",
            "status": "WARNING",
            "details": "Study compares severe, late-stage hospital cases against young, ultra-healthy controls (Classic Spectrum Bias). This falsely exaggerates Sensitivity and Specificity.",
            "recommendation": "Test the assay on consecutive patients presenting with relevant symptoms in primary care or triage."
        })

    # 4. Cell Count Adequacy & Wilson CI Precision
    total_diseased = tp + fn
    total_healthy = tn + fp
    min_condition_size = min(total_diseased, total_healthy)

    if min_condition_size >= 50:
        evaluations.append({
            "name": "Sample Size & Wilson CI Precision",
            "status": "PASS",
            "details": f"Substantial sample size in both diseased (n={total_diseased}) and non-diseased (n={total_healthy}) cohorts. Wilson score 95% confidence intervals are narrow and clinically precise.",
            "recommendation": "Diagnostic estimates have high statistical power."
        })
    elif min_condition_size >= 15:
        evaluations.append({
            "name": "Sample Size & Wilson CI Precision",
            "status": "WARNING",
            "details": f"Moderate sample size (minimum cohort size = {min_condition_size}). Confidence intervals will be wider.",
            "recommendation": "Wilson score confidence intervals are robust, but consider expanding sample size before clinical guideline adoption."
        })
    else:
        evaluations.append({
            "name": "Sample Size & Wilson CI Precision",
            "status": "FAIL",
            "details": f"Very small sample size detected (minimum cohort size = {min_condition_size} < 15). Point estimates may be unstable.",
            "recommendation": "Do NOT rely on point estimates alone; interpret the full width of the Wilson 95% confidence intervals."
        })

    return evaluations


def generate_diagnostic_accuracy_demo_data(
    scenario: str = "diabetic_retinopathy",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic patient-level and 2x2 matrix data for Diagnostic Test Accuracy demonstrations.

    Provides three realistic public health & clinical scenarios:
    1. 'diabetic_retinopathy': All Meadows Community Diabetic Eye Screening (High Sensitivity Screening: Sens = 94.0%, Spec = 94.4%, SnNOut).
    2. 'breast_mammography': County Mobile Breast Screening Service (Pop-wide Screening with Low Disease Prevalence: Sens = 85.0%, Spec = 90.0%, 190 False Positives).
    3. 'rapid_rsv_antigen': Inner City Hospital Pediatric Emergency Department Rapid RSV Antigen Cassette (High Specificity Confirmatory Test: Sens = 70.0%, Spec = 99.0%, SpPIn).

    Args:
        scenario (str, optional): Scenario identifier. Defaults to 'diabetic_retinopathy'.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - df_contingency: Aggregated 2x2 diagnostic confusion matrix table.
            - df_microdata: Individual patient synthetic microdata.
            - meta: Rich descriptive metadata, clinical takeaways, and disclaimer notices.
    """
    np.random.seed(seed)

    scenarios = {
        "diabetic_retinopathy": {
            "title": "All Meadows Community Diabetic Eye Screening: Autonomous AI Fundus Photography",
            "domain": "Ophthalmology, Diabetes & Artificial Intelligence",
            "description": "A hypothetical diagnostic validation study of 1,000 diabetic patients attending primary care screening across All Meadows region. Evaluates an autonomous AI fundus camera screening algorithm against the independent Gold Standard of Expert Consultant Ophthalmologist Slit-Lamp Biomicroscopy.",
            "test_name": "Autonomous AI Fundus Screening Camera",
            "condition_name": "Referable Diabetic Retinopathy (R2/R3/M1)",
            "test_levels": ("AI Positive (Referral Recommended)", "AI Negative (Routine Recall)"),
            "condition_levels": ("Retinopathy Present (Gold Standard +)", "Retinopathy Absent (Gold Standard -)"),
            "tp": 188,             # True Positives
            "fp": 45,              # False Positives
            "fn": 12,              # False Negatives
            "tn": 755,             # True Negatives
            "study_design": "cross_sectional_diagnostic",
            "expected_strength": "High Sensitivity (SnNOut)",
            "clinical_takeaway": "The AI screening algorithm achieved a Sensitivity of 94.0% (95% CI: 89.9% - 96.5%) and a Specificity of 94.4% (95% CI: 92.6% - 95.8%), yielding a Youden's Index of J = 0.884. With a False Negative Rate of only 6.0% (12 missed out of 200 diseased patients), a negative AI screening result reliably rules out referable disease (SnNOut), safely sparing 755 healthy patients an unnecessary hospital consultation.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "breast_mammography": {
            "title": "County Mobile Breast Screening Service: Digital Screening Mammography",
            "domain": "Preventive Oncology & Women's Health",
            "description": "A hypothetical population screening evaluation of 2,000 asymptomatic women aged 50–70 attending mobile digital screening mammography units. Evaluates screening radiologist interpretation against the Gold Standard of Histopathological Excisional Core Biopsy and 2-year cancer registry follow-up.",
            "test_name": "Digital Screening Mammography",
            "condition_name": "Biopsy-Confirmed Invasive Breast Carcinoma",
            "test_levels": ("Mammogram Abnormal (Recall for Assessment)", "Mammogram Normal (Routine 3-Year Recall)"),
            "condition_levels": ("Confirmed Carcinoma (Biopsy +)", "Benign / Healthy (Biopsy -)"),
            "tp": 85,              # True Positives
            "fp": 190,             # False Positives (Recalled for benign changes/fibroadenomas)
            "fn": 15,              # False Negatives (Interval / occult cancers)
            "tn": 1710,            # True Negatives
            "study_design": "screening_cohort",
            "expected_strength": "Balanced Screening Assay",
            "clinical_takeaway": "Screening mammography demonstrated a Sensitivity of 85.0% (95% CI: 76.8% - 90.7%) and a Specificity of 90.0% (95% CI: 88.6% - 91.3%), with overall Accuracy of 89.8%. However, because cancer prevalence in screening populations is low (5.0%), the 10.0% False Positive Rate generates 190 healthy women who experience anxiety and callback diagnostic workup for benign changes.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "rapid_rsv_antigen": {
            "title": "Inner City Hospital Pediatric Emergency Department: Rapid Point-of-Care RSV Antigen Cassette",
            "domain": "Pediatric Emergency Medicine & Clinical Virology",
            "description": "A hypothetical triage study of 600 infants presenting with acute bronchiolitis during winter surge at Inner City Hospital. Evaluates a 15-minute point-of-care rapid antigen cassette test against the independent Gold Standard of Laboratory Multiplex RT-PCR.",
            "test_name": "Point-of-Care Rapid RSV Antigen Cassette",
            "condition_name": "Confirmed Respiratory Syncytial Virus Infection",
            "test_levels": ("Rapid Test Positive (RSV Detected)", "Rapid Test Negative (No RSV Detected)"),
            "condition_levels": ("Lab RT-PCR Confirmed (+)", "Lab RT-PCR Negative (-)"),
            "tp": 140,             # True Positives
            "fp": 4,               # False Positives (Very low)
            "fn": 60,              # False Negatives (Missed low-viral load cases)
            "tn": 396,             # True Negatives
            "study_design": "emergency_department_triage",
            "expected_strength": "High Specificity (SpPIn)",
            "clinical_takeaway": "The rapid bedside test showed exceptional Specificity of 99.0% (95% CI: 97.4% - 99.6%) with moderate Sensitivity of 70.0% (95% CI: 63.3% - 75.9%). With only 4 false positives out of 400 non-RSV infants (1.0% False Positive Rate), a positive rapid result rules in the disease with near certainty (SpPIn), allowing immediate isolation and cohorting in the pediatric ward.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
    }

    if scenario not in scenarios:
        scenario = "diabetic_retinopathy"

    cfg = scenarios[scenario]
    tp = int(cfg["tp"])
    fp = int(cfg["fp"])
    fn = int(cfg["fn"])
    tn = int(cfg["tn"])

    test_pos_label, test_neg_label = cfg["test_levels"]
    cond_pos_label, cond_neg_label = cfg["condition_levels"]

    # 1. Aggregated 2x2 Diagnostic Confusion Matrix Table
    df_contingency = pd.DataFrame([
        {
            "Test Outcome": f"🔴 {test_pos_label}",
            f"{cond_pos_label}": tp,
            f"{cond_neg_label}": fp,
            "Total Test Positives": tp + fp,
            "Quadrant Label": "TP = True Positives (a) | FP = False Positives (b)"
        },
        {
            "Test Outcome": f"🟢 {test_neg_label}",
            f"{cond_pos_label}": fn,
            f"{cond_neg_label}": tn,
            "Total Test Negatives": fn + tn,
            "Quadrant Label": "FN = False Negatives (c) | TN = True Negatives (d)"
        },
        {
            "Test Outcome": "📊 Total Population",
            f"{cond_pos_label}": tp + fn,
            f"{cond_neg_label}": fp + tn,
            "Total Test Positives": tp + fp + fn + tn,
            "Quadrant Label": "Grand Total N"
        }
    ])

    # 2. Individual Patient Synthetic Microdata
    patients = []
    # True Positives
    for i in range(min(tp, 150)):
        patients.append({
            "Patient_ID": f"PT_TP_{i+1:04d}",
            "Test_Result": test_pos_label,
            "Test_Binary": 1,
            "True_Condition": cond_pos_label,
            "Condition_Binary": 1,
            "Classification": "True Positive (TP)",
            "Age": int(np.random.normal(58, 11)),
            "Sex": "Female" if np.random.rand() > 0.48 else "Male"
        })
    # False Positives
    for i in range(min(fp, 100)):
        patients.append({
            "Patient_ID": f"PT_FP_{i+1:04d}",
            "Test_Result": test_pos_label,
            "Test_Binary": 1,
            "True_Condition": cond_neg_label,
            "Condition_Binary": 0,
            "Classification": "False Positive (FP)",
            "Age": int(np.random.normal(55, 12)),
            "Sex": "Female" if np.random.rand() > 0.48 else "Male"
        })
    # False Negatives
    for i in range(min(fn, 100)):
        patients.append({
            "Patient_ID": f"PT_FN_{i+1:04d}",
            "Test_Result": test_neg_label,
            "Test_Binary": 0,
            "True_Condition": cond_pos_label,
            "Condition_Binary": 1,
            "Classification": "False Negative (FN)",
            "Age": int(np.random.normal(60, 10)),
            "Sex": "Female" if np.random.rand() > 0.48 else "Male"
        })
    # True Negatives
    for i in range(min(tn, 200)):
        patients.append({
            "Patient_ID": f"PT_TN_{i+1:04d}",
            "Test_Result": test_neg_label,
            "Test_Binary": 0,
            "True_Condition": cond_neg_label,
            "Condition_Binary": 0,
            "Classification": "True Negative (TN)",
            "Age": int(np.random.normal(54, 12)),
            "Sex": "Female" if np.random.rand() > 0.48 else "Male"
        })

    df_microdata = pd.DataFrame(patients)
    calcs = calculate_diagnostic_accuracy(tp, fp, fn, tn)

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


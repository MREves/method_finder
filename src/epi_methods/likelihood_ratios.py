"""Mathematical & Statistical Engine for Diagnostic Likelihood Ratios (LR+, LR-) and Fagan's Nomogram.

Provides verified epidemiological and clinical formulations for:
1. Positive Likelihood Ratio (Positive LR, LR+):
   LR+ = Sensitivity / (1 - Specificity) = True Positive Rate / False Positive Rate
   LR+ = (a / (a + c)) / (b / (b + d))
2. Negative Likelihood Ratio (Negative LR, LR-):
   LR- = (1 - Sensitivity) / Specificity = False Negative Rate / True Negative Rate
   LR- = (c / (a + c)) / (d / (b + d))
3. Diagnostic Odds Ratio (DOR):
   DOR = LR+ / LR- = (TP * TN) / (FP * FN)
4. Odds Form of Bayes' Theorem (Pre-to-Post-Test Probability Updating):
   Pre-Test Odds = p / (1 - p)
   Post-Test Odds = Pre-Test Odds * LR
   Post-Test Probability = Post-Test Odds / (1 + Post-Test Odds)
5. Simel, Samsa & Matchar (1991) Log-Scale 95% Confidence Intervals for LR+ and LR-.
6. Fagan's Nomogram Coordinates & Two-Step Probability Transformations.
7. Tri-color diagnostic validity evaluation: Reference standard independence, spectrum bias,
   reader blinding, and cell count adequacy.
8. Clinical scenario generators (Emergency Department D-Dimer for Pulmonary Embolism,
   Primary Care Rapid Antigen for Strep Throat, Pediatric Appendicitis Ultrasound).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_likelihood_ratios(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    pre_test_prob: float = 0.20,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Calculates Positive and Negative Likelihood Ratios, 95% log-scale CIs, and post-test probabilities.

    Formulas:
        Sensitivity (TPR) = tp / (tp + fn)
        Specificity (TNR) = tn / (tn + fp)
        FPR = 1 - Specificity = fp / (tn + fp)
        FNR = 1 - Sensitivity = fn / (tp + fn)
        
        LR+ = Sensitivity / (1 - Specificity) = TPR / FPR
        LR- = (1 - Sensitivity) / Specificity = FNR / TNR
        
        Pre-Test Odds = p / (1 - p)
        Post-Test Odds (+) = Pre-Test Odds * LR+
        Post-Test Prob (+) = Post-Test Odds (+) / (1 + Post-Test Odds (+)) = PPV
        
        Post-Test Odds (-) = Pre-Test Odds * LR-
        Post-Test Prob (-) = Post-Test Odds (-) / (1 + Post-Test Odds (-)) = 1 - NPV

    Log-scale Confidence Intervals (Simel et al., 1991):
        SE(ln(LR+)) = sqrt((1 - Se)/(tp) + Sp/(fp)) = sqrt(fn / (tp * (tp + fn)) + tn / (fp * (tn + fp)))
        SE(ln(LR-)) = sqrt(Se/(fn) + (1 - Sp)/(tn)) = sqrt(tp / (fn * (tp + fn)) + fp / (tn * (tn + fp)))

    Args:
        tp (int): True Positives.
        fp (int): False Positives.
        fn (int): False Negatives.
        tn (int): True Negatives.
        pre_test_prob (float, optional): Pre-test probability of disease (0.001 to 0.999). Defaults to 0.20.
        alpha (float, optional): Significance level. Defaults to 0.05 (95% CI).

    Returns:
        Dict[str, Any]: Comprehensive metrics dictionary containing:
            - 'lr_pos', 'lr_pos_ci_lower', 'lr_pos_ci_upper'
            - 'lr_neg', 'lr_neg_ci_lower', 'lr_neg_ci_upper'
            - 'pre_test_prob', 'pre_test_odds'
            - 'post_test_prob_pos', 'post_test_odds_pos'
            - 'post_test_prob_neg', 'post_test_odds_neg'
            - 'sensitivity', 'specificity', 'fpr', 'fnr', 'dor'
            - 'clinical_strength_pos', 'clinical_strength_neg'
    """
    applied_cc = (tp == 0 or fp == 0 or fn == 0 or tn == 0)
    tp_c = float(tp) + (0.5 if applied_cc else 0.0)
    fp_c = float(fp) + (0.5 if applied_cc else 0.0)
    fn_c = float(fn) + (0.5 if applied_cc else 0.0)
    tn_c = float(tn) + (0.5 if applied_cc else 0.0)

    total_diseased = tp_c + fn_c
    total_healthy = tn_c + fp_c

    sens = tp_c / total_diseased
    spec = tn_c / total_healthy
    fpr = 1.0 - spec
    fnr = 1.0 - sens

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    # 1. Positive Likelihood Ratio (LR+)
    if fpr > 0:
        lr_pos = sens / fpr
    else:
        lr_pos = np.inf

    # Simel et al. SE for ln(LR+)
    if tp_c > 0 and fp_c > 0 and total_diseased > 0 and total_healthy > 0:
        se_ln_lr_pos = float(np.sqrt((fn_c / (tp_c * total_diseased)) + (tn_c / (fp_c * total_healthy))))
        ln_lr_pos = float(np.log(lr_pos))
        lr_pos_ll = float(np.exp(ln_lr_pos - z_crit * se_ln_lr_pos))
        lr_pos_ul = float(np.exp(ln_lr_pos + z_crit * se_ln_lr_pos))
    else:
        se_ln_lr_pos, lr_pos_ll, lr_pos_ul = np.nan, np.nan, np.nan

    # 2. Negative Likelihood Ratio (LR-)
    if spec > 0:
        lr_neg = fnr / spec
    else:
        lr_neg = np.inf

    # Simel et al. SE for ln(LR-)
    if fn_c > 0 and tn_c > 0 and total_diseased > 0 and total_healthy > 0:
        se_ln_lr_neg = float(np.sqrt((tp_c / (fn_c * total_diseased)) + (fp_c / (tn_c * total_healthy))))
        ln_lr_neg = float(np.log(lr_neg))
        lr_neg_ll = float(np.exp(ln_lr_neg - z_crit * se_ln_lr_neg))
        lr_neg_ul = float(np.exp(ln_lr_neg + z_crit * se_ln_lr_neg))
    else:
        se_ln_lr_neg, lr_neg_ll, lr_neg_ul = np.nan, np.nan, np.nan

    # 3. Diagnostic Odds Ratio (DOR)
    if lr_neg > 0 and not np.isinf(lr_pos):
        dor = lr_pos / lr_neg
    else:
        dor = np.nan

    # 4. Odds Form of Bayes' Theorem Updating
    p_pre = float(np.clip(pre_test_prob, 1e-6, 1.0 - 1e-6))
    pre_odds = p_pre / (1.0 - p_pre)

    # Post-Test for Positive Result
    post_odds_pos = pre_odds * lr_pos if not np.isinf(lr_pos) else np.inf
    post_prob_pos = post_odds_pos / (1.0 + post_odds_pos) if not np.isinf(post_odds_pos) else 1.0

    # Post-Test for Negative Result
    post_odds_neg = pre_odds * lr_neg if not np.isinf(lr_neg) else np.inf
    post_prob_neg = post_odds_neg / (1.0 + post_odds_neg) if not np.isinf(post_odds_neg) else 0.0

    # 5. Clinical Strength Categories (Evidence-Based Medicine guidelines)
    if lr_pos >= 10.0:
        strength_pos = "Conclusive Rule-In (Large increase in disease probability)"
    elif lr_pos >= 5.0:
        strength_pos = "Moderate Rule-In (Substantial increase in disease probability)"
    elif lr_pos >= 2.0:
        strength_pos = "Weak / Small Rule-In (Small increase in disease probability)"
    else:
        strength_pos = "Minimal / Unhelpful Diagnostic Utility"

    if lr_neg <= 0.10:
        strength_neg = "Conclusive Rule-Out (Large decrease in disease probability)"
    elif lr_neg <= 0.20:
        strength_neg = "Moderate Rule-Out (Substantial decrease in disease probability)"
    elif lr_neg <= 0.50:
        strength_neg = "Weak / Small Rule-Out (Small decrease in disease probability)"
    else:
        strength_neg = "Minimal / Unhelpful Diagnostic Utility"

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_diseased": int(tp + fn),
        "total_healthy": int(tn + fp),
        "total_sample": int(tp + fp + fn + tn),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "lr_pos": float(lr_pos),
        "lr_pos_ci_lower": float(lr_pos_ll),
        "lr_pos_ci_upper": float(lr_pos_ul),
        "lr_neg": float(lr_neg),
        "lr_neg_ci_lower": float(lr_neg_ll),
        "lr_neg_ci_upper": float(lr_neg_ul),
        "dor": float(dor),
        "pre_test_prob": float(p_pre),
        "pre_test_odds": float(pre_odds),
        "post_test_odds_pos": float(post_odds_pos),
        "post_test_prob_pos": float(post_prob_pos),
        "post_test_odds_neg": float(post_odds_neg),
        "post_test_prob_neg": float(post_prob_neg),
        "clinical_strength_pos": strength_pos,
        "clinical_strength_neg": strength_neg,
        "continuity_applied": applied_cc
    }


def evaluate_likelihood_ratio_prerequisites(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    is_gold_standard_independent: bool = True,
    is_spectrum_representative: bool = True,
    is_blinded_interpretation: bool = True
) -> List[Dict[str, Any]]:
    """Evaluates methodological validity and bias risks for Diagnostic Likelihood Ratios (STARD/QUADAS-2).

    Args:
        tp (int): True Positives.
        fp (int): False Positives.
        fn (int): False Negatives.
        tn (int): True Negatives.
        is_gold_standard_independent (bool, optional): Whether reference standard is independent. Defaults to True.
        is_spectrum_representative (bool, optional): Whether sample reflects clinical spectrum. Defaults to True.
        is_blinded_interpretation (bool, optional): Whether readers were blinded. Defaults to True.

    Returns:
        List[Dict[str, Any]]: Diagnostic prerequisite evaluation cards.
    """
    evaluations: List[Dict[str, Any]] = []

    # 1. Gold Standard Reference Independence (Incorporation Bias)
    if is_gold_standard_independent:
        evaluations.append({
            "name": "Reference Standard Independence",
            "status": "PASS",
            "details": "The index test is completely independent of the reference Gold Standard criteria.",
            "recommendation": "Zero incorporation bias; Positive LR and Negative LR reflect true discrimination."
        })
    else:
        evaluations.append({
            "name": "Reference Standard Independence",
            "status": "FAIL",
            "details": "The index test is part of the reference standard definition (Incorporation Bias), causing artificial inflation of LR+ and deflation of LR-.",
            "recommendation": "Re-evaluate the index test against an external, independent reference standard."
        })

    # 2. Clinical Spectrum Representativeness (Spectrum Bias)
    if is_spectrum_representative:
        evaluations.append({
            "name": "Clinical Spectrum Representativeness",
            "status": "PASS",
            "details": "The study population includes a representative mix of mild, moderate, and severe cases, alongside symptomatic non-cases.",
            "recommendation": "Likelihood ratios can be safely applied across the intended clinical setting."
        })
    else:
        evaluations.append({
            "name": "Clinical Spectrum Representativeness",
            "status": "WARNING",
            "details": "Extreme spectrum sampling: Comparing moribund intensive care cases against ultra-healthy young volunteers creates artificially extreme likelihood ratios.",
            "recommendation": "Validate likelihood ratios in prospective, consecutive patients presenting with diagnostic uncertainty."
        })

    # 3. Blinded Diagnostic Interpretation (Review Bias)
    if is_blinded_interpretation:
        evaluations.append({
            "name": "Blinded Diagnostic Review",
            "status": "PASS",
            "details": "Clinicians interpreted the index test without prior knowledge of gold standard findings, and vice versa.",
            "recommendation": "Observer expectation bias is effectively controlled."
        })
    else:
        evaluations.append({
            "name": "Blinded Diagnostic Review",
            "status": "WARNING",
            "details": "Interpreters were aware of patient clinical status or reference test outcomes before scoring the index test.",
            "recommendation": "Implement blinded re-reading to confirm sensitivity and specificity estimates."
        })

    # 4. Cell Count Adequacy for Likelihood Ratio Precision
    min_cell = min(tp, fp, fn, tn)
    total_sample = tp + fp + fn + tn

    if min_cell >= 10:
        evaluations.append({
            "name": "Cell Frequency & Confidence Interval Stability",
            "status": "PASS",
            "details": f"All four 2×2 contingency cells have robust counts (minimum cell = {min_cell}, Total N = {total_sample:,}). Confidence intervals for LR+ and LR- are tight.",
            "recommendation": "High statistical power for post-test probability calculations."
        })
    elif min_cell >= 1:
        evaluations.append({
            "name": "Cell Frequency & Confidence Interval Stability",
            "status": "WARNING",
            "details": f"Sparse cells detected (minimum cell count = {min_cell}). Log-scale confidence intervals will be wider.",
            "recommendation": "Examine the 95% confidence intervals on Fagan's Nomogram rather than relying on point estimates alone."
        })
    else:
        evaluations.append({
            "name": "Cell Frequency & Confidence Interval Stability",
            "status": "WARNING",
            "details": "Zero cell detected in contingency matrix. Haldane-Anscombe continuity correction (+0.5) was applied.",
            "recommendation": "Likelihood ratio point estimates are regularized, but larger sample validation is recommended."
        })

    return evaluations


def generate_likelihood_ratio_demo_data(
    scenario: str = "pulmonary_embolism_d_dimer",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic patient-level and 2x2 matrix data for Diagnostic Likelihood Ratios demonstrations.

    Provides three realistic scenarios:
    1. 'pulmonary_embolism_d_dimer': Inner City Hospital Emergency Department: Highly Sensitive D-Dimer for Pulmonary Embolism (Sens = 96.0%, Spec = 55.0% -> LR+ = 2.13, LR- = 0.073: Conclusive Rule-Out).
    2. 'strep_throat_centor_rapid_antigen': All Meadows Community Primary Care: Highly Specific Point-of-Care Strep A Rapid Antigen (Sens = 82.0%, Spec = 96.0% -> LR+ = 20.50, LR- = 0.188: Conclusive Rule-In).
    3. 'appendicitis_alvarado_ultrasound': Riverside Children's Hospital: Pediatric Acute Appendicitis Graded-Compression Ultrasound (Sens = 88.0%, Spec = 94.0% -> LR+ = 14.67, LR- = 0.128: Balanced Diagnostic Imaging).

    Args:
        scenario (str, optional): Scenario identifier. Defaults to 'pulmonary_embolism_d_dimer'.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - df_contingency: Aggregated 2x2 matrix with row and column marginals.
            - df_microdata: Individual synthetic patient records.
            - meta: Scenario descriptive metadata, clinical takeaways, and disclaimer notices.
    """
    np.random.seed(seed)

    scenarios = {
        "pulmonary_embolism_d_dimer": {
            "title": "Inner City Hospital Emergency Department: Wells Score & High-Sensitivity D-Dimer for Pulmonary Embolism",
            "domain": "Emergency Medicine & Acute Care",
            "description": "A hypothetical diagnostic validation study of 1,000 adult patients presenting with pleuritic chest pain and dyspnea to Inner City Hospital Emergency Department. Evaluates a quantitative latex agglutination D-Dimer assay against the Gold Standard of CT Pulmonary Angiography (CTPA). The cohort has an intermediate pre-test probability of Pulmonary Embolism (20.0%). Demonstrates how a very low Negative Likelihood Ratio (LR- = 0.073) conclusively rules out a life-threatening condition.",
            "test_name": "Quantitative Plasma D-Dimer Assay",
            "condition_name": "CTPA-Confirmed Acute Pulmonary Embolism",
            "test_levels": ("D-Dimer Elevated (> 500 ng/mL)", "D-Dimer Normal (<= 500 ng/mL)"),
            "condition_levels": ("Confirmed PE (CTPA +)", "No PE (CTPA -)"),
            "tp": 192,             # 96% of 200 diseased
            "fp": 360,             # 45% of 800 healthy (FPR = 45%, Specificity = 55%)
            "fn": 8,               # 4% of 200 diseased (Missed cases)
            "tn": 440,             # 55% of 800 healthy
            "pre_test_prob": 0.20,
            "expected_strength": "Conclusive Rule-Out (Ultra-Low LR-)",
            "clinical_takeaway": "D-Dimer demonstrated a Sensitivity of 96.0% and Specificity of 55.0%, producing a Positive Likelihood Ratio (Positive LR, LR+) of 2.13 and an exceptional Negative Likelihood Ratio (Negative LR, LR-) of 0.073 (95% CI: 0.038 - 0.141). Starting from an intermediate pre-test probability of 20.0%, a negative D-Dimer result drops post-test disease probability down to just 1.8% (Post-Test Odds: 0.018), safely ruling out pulmonary embolism and sparing 440 patients unnecessary intravenous contrast CT radiation.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "strep_throat_centor_rapid_antigen": {
            "title": "All Meadows Community Primary Care: Centor Criteria & Point-of-Care Strep A Rapid Antigen Cassette",
            "domain": "General Practice & Pediatric Infectious Diseases",
            "description": "A hypothetical primary care diagnostic study of 800 patients presenting with acute exudative pharyngitis across All Meadows Primary Care Network. Evaluates a 10-minute lateral flow optical immunoassay for Group A Beta-Hemolytic Streptococcus against the Gold Standard of 48-hour Laboratory Throat Swab Culture. Starting from a pre-test probability of 30.0% (Centor Score 3). Demonstrates how a high Positive Likelihood Ratio (LR+ = 20.50) conclusively confirms diagnosis and justifies targeted antibiotic therapy.",
            "test_name": "Rapid Point-of-Care Strep A Antigen Cassette",
            "condition_name": "Culture-Confirmed Group A Streptococcal Pharyngitis",
            "test_levels": ("Rapid Test Positive", "Rapid Test Negative"),
            "condition_levels": ("Culture Confirmed (+)", "Culture Negative (-)"),
            "tp": 197,             # 82% of 240 diseased
            "fp": 22,              # 4% of 560 healthy (Specificity = 96.0%)
            "fn": 43,              # 18% of 240 diseased
            "tn": 538,             # 96% of 560 healthy
            "pre_test_prob": 0.30,
            "expected_strength": "Conclusive Rule-In (High LR+)",
            "clinical_takeaway": "With a Sensitivity of 82.1% and a high Specificity of 96.1%, the rapid antigen test achieved an outstanding Positive Likelihood Ratio (Positive LR, LR+) of 20.89 (95% CI: 13.78 - 31.67) and a Negative Likelihood Ratio (Negative LR, LR-) of 0.186 (95% CI: 0.141 - 0.246). In a patient with a 30.0% pre-test suspicion, a positive rapid test elevates the post-test probability of Strep pharyngitis to 90.0%, providing indisputable clinical rationale for immediate antibiotic prescribing while curbing unnecessary antimicrobial overuse.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "appendicitis_alvarado_ultrasound": {
            "title": "Riverside Children's Hospital: Alvarado Score & Graded-Compression Abdominal Ultrasound",
            "domain": "Pediatric Surgery & Diagnostic Imaging",
            "description": "A hypothetical validation study of 600 children presenting with right lower quadrant abdominal pain to Riverside Children's Hospital. Evaluates expert graded-compression abdominal ultrasound against the Gold Standard of Surgical Histopathology and 30-day clinical resolution. Baseline pre-test probability of appendicitis is 35.0% based on Alvarado scoring.",
            "test_name": "Graded-Compression Abdominal Ultrasound",
            "condition_name": "Histopathology-Confirmed Acute Appendicitis",
            "test_levels": ("Ultrasound Positive (Non-Compressible Appendix > 6mm)", "Ultrasound Negative (Normal / Compressible)"),
            "condition_levels": ("Appendicitis Confirmed (Pathology +)", "No Appendicitis (Pathology -)"),
            "tp": 185,             # 88% of 210 diseased
            "fp": 23,              # 6% of 390 healthy (Specificity = 94.1%)
            "fn": 25,              # 12% of 210 diseased
            "tn": 367,             # 94.1% of 390 healthy
            "pre_test_prob": 0.35,
            "expected_strength": "Balanced Diagnostic Discrimination",
            "clinical_takeaway": "Graded-compression ultrasound demonstrated a Sensitivity of 88.1% and Specificity of 94.1%, yielding a Positive Likelihood Ratio (Positive LR, LR+) of 14.94 (95% CI: 10.02 - 22.28) and a Negative Likelihood Ratio (Negative LR, LR-) of 0.126 (95% CI: 0.086 - 0.186). For a child with an intermediate Alvarado pre-test risk of 35.0%, a positive ultrasound drives post-test probability up to 88.9% (justifying immediate surgical consultation), whereas a negative scan drops probability down to 6.4%, safely avoiding exploratory laparoscopy.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
    }

    if scenario not in scenarios:
        scenario = "pulmonary_embolism_d_dimer"

    cfg = scenarios[scenario]
    tp = int(cfg["tp"])
    fp = int(cfg["fp"])
    fn = int(cfg["fn"])
    tn = int(cfg["tn"])
    pre_test = float(cfg["pre_test_prob"])

    test_pos_label, test_neg_label = cfg["test_levels"]
    cond_pos_label, cond_neg_label = cfg["condition_levels"]

    # 1. Aggregated 2x2 Contingency Matrix with LR Marginals
    calcs = calculate_likelihood_ratios(tp, fp, fn, tn, pre_test_prob=pre_test)

    df_contingency = pd.DataFrame([
        {
            "Test Finding": f"🔴 {test_pos_label}",
            f"{cond_pos_label}": tp,
            f"{cond_neg_label}": fp,
            "Total Test Marginals": tp + fp,
            "Likelihood Ratio Dimension": f"Positive LR (LR+) = {calcs['lr_pos']:.2f} (95% CI: {calcs['lr_pos_ci_lower']:.2f} - {calcs['lr_pos_ci_upper']:.2f})"
        },
        {
            "Test Finding": f"🟢 {test_neg_label}",
            f"{cond_pos_label}": fn,
            f"{cond_neg_label}": tn,
            "Total Test Marginals": fn + tn,
            "Likelihood Ratio Dimension": f"Negative LR (LR-) = {calcs['lr_neg']:.3f} (95% CI: {calcs['lr_neg_ci_lower']:.3f} - {calcs['lr_neg_ci_upper']:.3f})"
        },
        {
            "Test Finding": "📊 Total Gold Standard Cohort",
            f"{cond_pos_label}": tp + fn,
            f"{cond_neg_label}": fp + tn,
            "Total Test Marginals": tp + fp + fn + tn,
            "Likelihood Ratio Dimension": f"Diagnostic Odds Ratio (DOR) = {calcs['dor']:.1f}"
        }
    ])

    # 2. Individual Patient Synthetic Microdata
    patients = []
    # TP
    for i in range(min(tp, 120)):
        patients.append({
            "Patient_ID": f"PT_TP_{i+1:04d}",
            "Test_Result": test_pos_label,
            "True_Condition": cond_pos_label,
            "Classification": "True Positive (TP)",
            "Pre_Test_Probability": pre_test,
            "Post_Test_Probability": calcs["post_test_prob_pos"],
            "Age": int(np.random.normal(52, 14)),
            "Sex": "Female" if np.random.rand() > 0.49 else "Male"
        })
    # FP
    for i in range(min(fp, 100)):
        patients.append({
            "Patient_ID": f"PT_FP_{i+1:04d}",
            "Test_Result": test_pos_label,
            "True_Condition": cond_neg_label,
            "Classification": "False Positive (FP)",
            "Pre_Test_Probability": pre_test,
            "Post_Test_Probability": calcs["post_test_prob_pos"],
            "Age": int(np.random.normal(50, 15)),
            "Sex": "Female" if np.random.rand() > 0.49 else "Male"
        })
    # FN
    for i in range(min(fn, 80)):
        patients.append({
            "Patient_ID": f"PT_FN_{i+1:04d}",
            "Test_Result": test_neg_label,
            "True_Condition": cond_pos_label,
            "Classification": "False Negative (FN)",
            "Pre_Test_Probability": pre_test,
            "Post_Test_Probability": calcs["post_test_prob_neg"],
            "Age": int(np.random.normal(55, 12)),
            "Sex": "Female" if np.random.rand() > 0.49 else "Male"
        })
    # TN
    for i in range(min(tn, 200)):
        patients.append({
            "Patient_ID": f"PT_TN_{i+1:04d}",
            "Test_Result": test_neg_label,
            "True_Condition": cond_neg_label,
            "Classification": "True Negative (TN)",
            "Pre_Test_Probability": pre_test,
            "Post_Test_Probability": calcs["post_test_prob_neg"],
            "Age": int(np.random.normal(48, 16)),
            "Sex": "Female" if np.random.rand() > 0.49 else "Male"
        })

    df_microdata = pd.DataFrame(patients)

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


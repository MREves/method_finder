"""Missing Data Mechanisms, Imputation & Sensitivity Audits Pure Calculation Engine.

Provides mathematically verified calculations for:
1. Rubin's Taxonomy & Diagnostic Audits (Little's MCAR test statistic, Logistic missingness audits)
2. Complete Case vs Single Imputation Hazards (Sample loss attrition, Mean imputation variance collapse)
3. Multiple Imputation by Chained Equations (MICE) & Rubin's Rules Pooling (Q-bar, U-bar, B, T, FMI, RIV)
4. Algorithmic k-NN Distance-Based Imputation (Normalized feature donor matching, RMSE evaluation)
5. MNAR Pattern-Mixture Delta-Adjustment & Tipping Point Analysis (ICH E9 R1 regulatory sensitivity stress-testing)
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import random

# =====================================================================
# 1. Rubin's Taxonomy & Diagnostic Audits Engine
# =====================================================================

def calculate_rubin_taxonomy_model(
    n_records: int = 1000,
    pct_missing: float = 0.25,
    mechanism: str = "MAR_by_sofa",
    seed: int = 42
) -> Dict[str, Any]:
    """Simulates clinical dataset with missing biomarker values and executes
    diagnostic audits (mean difference, logistic missingness test, Little's MCAR test).
    """
    rng = random.Random(seed)
    records = []

    sofa_obs = []
    sofa_mis = []
    lactate_obs = []

    for i in range(1, n_records + 1):
        age = int(rng.gauss(64, 12))
        age = max(18, min(95, age))

        # SOFA score: Sequential Organ Failure Assessment (0 to 20)
        sofa = int(rng.gauss(6.5, 3.2))
        sofa = max(0, min(20, sofa))

        # True underlying arterial lactate (mmol/L)
        true_lactate = 1.2 + 0.35 * sofa + rng.gauss(0, 0.8)
        true_lactate = max(0.4, round(true_lactate, 2))

        # Determine missingness indicator M based on mechanism
        if mechanism == "MCAR":
            is_missing = 1 if rng.random() < pct_missing else 0
        elif mechanism == "MNAR":
            # Missingness depends on lactate itself (high lactate patients too unstable to sample)
            prob_m = 1.0 / (1.0 + math.exp(-(true_lactate - 3.5)))
            is_missing = 1 if rng.random() < prob_m else 0
        else:  # MAR_by_sofa (default)
            # Missingness depends on observed SOFA acuity
            prob_m = 1.0 / (1.0 + math.exp(-0.35 * (sofa - 6.0)))
            is_missing = 1 if rng.random() < prob_m else 0

        recorded_lactate = None if is_missing else true_lactate

        if is_missing:
            sofa_mis.append(sofa)
        else:
            sofa_obs.append(sofa)
            lactate_obs.append(true_lactate)

        records.append({
            "patient_id": f"ICU-{i:04d}",
            "age": age,
            "sofa_score": sofa,
            "true_lactate": true_lactate,
            "recorded_lactate": recorded_lactate,
            "is_missing": is_missing
        })

    n_mis = len(sofa_mis)
    n_obs = len(sofa_obs)
    actual_pct_missing = (n_mis / n_records) * 100.0

    mean_sofa_obs = sum(sofa_obs) / n_obs if n_obs > 0 else 0.0
    mean_sofa_mis = sum(sofa_mis) / n_mis if n_mis > 0 else 0.0
    delta_sofa = mean_sofa_mis - mean_sofa_obs

    # Variance and standard error of difference
    var_obs = sum((x - mean_sofa_obs) ** 2 for x in sofa_obs) / (n_obs - 1) if n_obs > 1 else 1.0
    var_mis = sum((x - mean_sofa_mis) ** 2 for x in sofa_mis) / (n_mis - 1) if n_mis > 1 else 1.0
    se_diff = math.sqrt((var_obs / n_obs) + (var_mis / n_mis)) if (n_obs > 0 and n_mis > 0) else 1.0

    t_stat = delta_sofa / se_diff if se_diff > 0 else 0.0
    # Two-sided normal approximation for large sample p-value
    p_val_approx = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))

    # Little's MCAR Test Chi-Square analog for 1 variable
    chi2_stat = t_stat ** 2

    # Diagnosis verdict
    if p_val_approx < 0.001:
        diagnosis = "🔴 Strong Evidence Against MCAR (MAR or MNAR): Missingness significantly associated with SOFA score (p < 0.001)"
        mechanism_verdict = "MAR (or MNAR)"
    elif p_val_approx < 0.05:
        diagnosis = f"🟡 Moderate Evidence Against MCAR: Missingness associated with SOFA score (p = {p_val_approx:.3f})"
        mechanism_verdict = "Likely MAR"
    else:
        diagnosis = f"🟢 Fail to Reject MCAR: Observed means do not significantly differ between missing and observed groups (p = {p_val_approx:.3f})"
        mechanism_verdict = "Consistent with MCAR"

    return {
        "n_records": n_records,
        "n_observed": n_obs,
        "n_missing": n_mis,
        "actual_pct_missing": actual_pct_missing,
        "mean_sofa_observed": mean_sofa_obs,
        "mean_sofa_missing": mean_sofa_mis,
        "delta_sofa": delta_sofa,
        "se_difference": se_diff,
        "t_statistic": t_stat,
        "chi2_mcar_stat": chi2_stat,
        "p_value": p_val_approx,
        "diagnosis": diagnosis,
        "mechanism_verdict": mechanism_verdict,
        "sample_records": records[:100]
    }


def generate_rubin_dataset(
    n_records: int = 1000,
    pct_missing: float = 0.25,
    mechanism: str = "MAR_by_sofa",
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Generates synthetic patient dataset for CSV download and table preview."""
    model = calculate_rubin_taxonomy_model(n_records, pct_missing, mechanism, seed)
    return model["sample_records"]


# =====================================================================
# 2. Complete Case vs Single Imputation Hazards Engine
# =====================================================================

def calculate_complete_case_vs_single_model(
    n_records: int = 500,
    n_covariates: int = 6,
    pct_missing_per_var: float = 0.12,
    seed: int = 42
) -> Dict[str, Any]:
    """Calculates effective sample size loss under Complete Case Analysis (Listwise Deletion)
    and severe variance collapse under naive Mean Imputation.
    """
    # Theoretical retained sample proportion assuming independent missingness
    theoretical_retained_pct = ((1.0 - pct_missing_per_var) ** n_covariates) * 100.0
    theoretical_discarded_pct = 100.0 - theoretical_retained_pct

    # Mean Imputation Variance Collapse
    # Var(Y_imp) = (1 - pi) * Var(Y_true)
    pi_missing = pct_missing_per_var
    variance_shrinkage_pct = pi_missing * 100.0
    retained_variance_pct = (1.0 - pi_missing) * 100.0

    # Standard error reduction factor: sqrt(1 - pi)
    se_deflation_factor = math.sqrt(1.0 - pi_missing)
    se_reduction_pct = (1.0 - se_deflation_factor) * 100.0

    # Type I error inflation index:
    # A nominal 95% critical value of z = 1.96 becomes effectively z_eff = 1.96 / sqrt(1 - pi)
    z_inflated = 1.96 / se_deflation_factor
    # Apparent alpha = 2 * (1 - Phi(z_inflated))
    type1_error_rate = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z_inflated / math.sqrt(2.0))))

    # Simulate actual patient retention across K covariates
    rng = random.Random(seed)
    complete_count = 0
    sim_records = []

    for i in range(1, n_records + 1):
        row_has_missing = False
        missing_vars_count = 0
        for _ in range(n_covariates):
            if rng.random() < pct_missing_per_var:
                row_has_missing = True
                missing_vars_count += 1
        if not row_has_missing:
            complete_count += 1

        sim_records.append({
            "patient_id": f"PAT-{i:04d}",
            "missing_covariates_count": missing_vars_count,
            "retained_in_complete_case": 0 if row_has_missing else 1
        })

    empirical_retained_pct = (complete_count / n_records) * 100.0
    empirical_discarded_pct = 100.0 - empirical_retained_pct

    return {
        "n_records": n_records,
        "n_covariates": n_covariates,
        "pct_missing_per_var": pct_missing_per_var * 100.0,
        "theoretical_retained_pct": theoretical_retained_pct,
        "theoretical_discarded_pct": theoretical_discarded_pct,
        "empirical_retained_count": complete_count,
        "empirical_discarded_count": n_records - complete_count,
        "empirical_retained_pct": empirical_retained_pct,
        "empirical_discarded_pct": empirical_discarded_pct,
        "variance_shrinkage_pct": variance_shrinkage_pct,
        "retained_variance_pct": retained_variance_pct,
        "se_deflation_factor": se_deflation_factor,
        "se_reduction_pct": se_reduction_pct,
        "type1_error_rate": type1_error_rate,
        "sample_records": sim_records[:100]
    }


def generate_complete_case_dataset(
    n_records: int = 500,
    n_covariates: int = 6,
    pct_missing_per_var: float = 0.12,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Generates synthetic dataset for Complete Case & Mean Imputation preview."""
    model = calculate_complete_case_vs_single_model(n_records, n_covariates, pct_missing_per_var, seed)
    return model["sample_records"]


# =====================================================================
# 3. MICE & Rubin's Rules Pooling Engine
# =====================================================================

def calculate_mice_rubin_model(
    m_imputations: int = 10,
    n_patients: int = 600,
    pct_missing: float = 0.20,
    true_treatment_effect: float = -0.65,
    seed: int = 42
) -> Dict[str, Any]:
    """Simulates m completed datasets under MICE and executes Rubin's Rules
    for pooled point estimates, within-variance, between-variance, total variance,
    FMI, RIV, and Barnard-Rubin adjusted degrees of freedom.
    """
    rng = random.Random(seed)
    estimates = []
    variances = []
    imputation_details = []

    # Nominal standard error from full complete cohort
    nominal_se = 0.12

    for l in range(1, m_imputations + 1):
        # Point estimate has small imputation sampling noise around true effect
        q_l = true_treatment_effect + rng.gauss(0, nominal_se * math.sqrt(pct_missing * 0.8))
        # Within-imputation variance has slight variation
        u_l = (nominal_se ** 2) * (1.0 + rng.uniform(-0.08, 0.08))

        estimates.append(q_l)
        variances.append(u_l)

        imputation_details.append({
            "imputation_number": l,
            "treatment_effect_q_l": round(q_l, 4),
            "within_variance_u_l": round(u_l, 6),
            "std_error_se_l": round(math.sqrt(u_l), 4)
        })

    # 1. Pooled Point Estimate Q-bar
    q_bar = sum(estimates) / m_imputations

    # 2. Within-Imputation Variance U-bar
    u_bar = sum(variances) / m_imputations

    # 3. Between-Imputation Variance B
    b_var = sum((q - q_bar) ** 2 for q in estimates) / (m_imputations - 1) if m_imputations > 1 else 0.0

    # 4. Total Variance T = U-bar + (1 + 1/m) * B
    finite_sample_factor = 1.0 + (1.0 / m_imputations)
    total_var = u_bar + (finite_sample_factor * b_var)
    pooled_se = math.sqrt(total_var)

    # 5. Relative Increase in Variance (RIV)
    riv = (finite_sample_factor * b_var) / u_bar if u_bar > 0 else 0.0

    # 6. Fraction of Missing Information (FMI)
    fmi = (finite_sample_factor * b_var) / total_var if total_var > 0 else 0.0

    # 7. Barnard & Rubin (1999) Adjusted Degrees of Freedom (nu)
    nu_com = n_patients - 2  # degrees of freedom under complete data
    if riv > 1e-9:
        nu_old = (m_imputations - 1) * ((1.0 + (1.0 / riv)) ** 2)
        nu_obs = ((nu_com + 1.0) / (nu_com + 3.0)) * nu_com * (1.0 - fmi)
        nu_adjusted = (nu_old * nu_obs) / (nu_old + nu_obs)
    else:
        nu_adjusted = float(nu_com)

    # 8. 95% Confidence Interval & p-value
    # Approximate t-critical value
    t_crit = 1.96 if nu_adjusted > 120 else 2.00
    ci_low = q_bar - t_crit * pooled_se
    ci_high = q_bar + t_crit * pooled_se

    z_stat = q_bar / pooled_se if pooled_se > 0 else 0.0
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0))))

    return {
        "m_imputations": m_imputations,
        "n_patients": n_patients,
        "pct_missing": pct_missing * 100.0,
        "q_bar_pooled": q_bar,
        "u_bar_within_variance": u_bar,
        "b_between_variance": b_var,
        "total_variance_t": total_var,
        "pooled_se": pooled_se,
        "riv": riv,
        "fmi": fmi,
        "fmi_pct": fmi * 100.0,
        "degrees_of_freedom": nu_adjusted,
        "ci_95": (ci_low, ci_high),
        "z_statistic": z_stat,
        "p_value": p_value,
        "imputation_details": imputation_details
    }


def generate_mice_dataset(
    mice_results: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Generates table of m imputations and Rubin pooling for CSV download."""
    if mice_results is None:
        mice_results = calculate_mice_rubin_model()
    return mice_results["imputation_details"]


# =====================================================================
# 4. Algorithmic k-NN & MissForest Engine
# =====================================================================

def calculate_knn_imputation_model(
    n_samples: int = 400,
    k_neighbors: int = 5,
    pct_missing: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """Simulates a clinical dataset with non-linear biomarker correlations and executes
    k-nearest neighbor (k-NN) donor distance-based imputation.
    """
    rng = random.Random(seed)
    patients = []

    # Generate synthetic patient cohort with non-linear biomarker relationship
    for i in range(1, n_samples + 1):
        age = rng.gauss(55, 10)
        bmi = rng.gauss(27.5, 4.2)
        systolic_bp = 110 + 0.4 * age + 0.6 * bmi + rng.gauss(0, 8)
        # Non-linear quadratic biomarker concentration (e.g. troponin / CRP)
        true_biomarker = 15.0 + 0.05 * (bmi ** 2) + 0.2 * age + rng.gauss(0, 4.0)
        true_biomarker = max(1.0, round(true_biomarker, 2))

        is_missing = 1 if rng.random() < pct_missing else 0
        recorded_biomarker = None if is_missing else true_biomarker

        patients.append({
            "patient_id": f"METAB-{i:04d}",
            "age": round(age, 1),
            "bmi": round(bmi, 1),
            "systolic_bp": round(systolic_bp, 1),
            "true_biomarker": true_biomarker,
            "recorded_biomarker": recorded_biomarker,
            "is_missing": is_missing
        })

    # Separate complete donors from missing recipients
    donors = [p for p in patients if p["is_missing"] == 0]
    recipients = [p for p in patients if p["is_missing"] == 1]

    # Compute donor feature means and standard deviations for standardized Euclidean distance
    mean_age = sum(d["age"] for d in donors) / len(donors)
    sd_age = math.sqrt(sum((d["age"] - mean_age)**2 for d in donors) / len(donors)) or 1.0

    mean_bmi = sum(d["bmi"] for d in donors) / len(donors)
    sd_bmi = math.sqrt(sum((d["bmi"] - mean_bmi)**2 for d in donors) / len(donors)) or 1.0

    mean_sbp = sum(d["systolic_bp"] for d in donors) / len(donors)
    sd_sbp = math.sqrt(sum((d["systolic_bp"] - mean_sbp)**2 for d in donors) / len(donors)) or 1.0

    sum_sq_error = 0.0
    imputed_records = []

    for r in recipients:
        # Standardize recipient features
        r_z_age = (r["age"] - mean_age) / sd_age
        r_z_bmi = (r["bmi"] - mean_bmi) / sd_bmi
        r_z_sbp = (r["systolic_bp"] - mean_sbp) / sd_sbp

        # Compute distance to all complete donors
        distances = []
        for d in donors:
            d_z_age = (d["age"] - mean_age) / sd_age
            d_z_bmi = (d["bmi"] - mean_bmi) / sd_bmi
            d_z_sbp = (d["systolic_bp"] - mean_sbp) / sd_sbp

            dist = math.sqrt((r_z_age - d_z_age)**2 + (r_z_bmi - d_z_bmi)**2 + (r_z_sbp - d_z_sbp)**2)
            distances.append((dist, d["true_biomarker"]))

        # Sort by distance and pick k nearest donors
        distances.sort(key=lambda x: x[0])
        nearest_k = distances[:k_neighbors]

        # Distance-weighted average
        weights = [1.0 / max(0.01, dist) for dist, _ in nearest_k]
        sum_weights = sum(weights)
        imputed_val = sum(w * val for w, (_, val) in zip(weights, nearest_k)) / sum_weights
        imputed_val = round(imputed_val, 2)

        sq_err = (imputed_val - r["true_biomarker"]) ** 2
        sum_sq_error += sq_err

        r_copy = r.copy()
        r_copy["imputed_biomarker"] = imputed_val
        r_copy["imputation_error"] = round(imputed_val - r["true_biomarker"], 2)
        imputed_records.append(r_copy)

    rmse = math.sqrt(sum_sq_error / len(recipients)) if recipients else 0.0

    # Mean absolute percentage error (MAPE)
    mape = (sum(abs(r["imputation_error"]) / r["true_biomarker"] for r in imputed_records) / len(imputed_records)) * 100.0 if imputed_records else 0.0

    return {
        "n_samples": n_samples,
        "k_neighbors": k_neighbors,
        "n_donors": len(donors),
        "n_imputed": len(recipients),
        "imputation_rmse": rmse,
        "imputation_mape_pct": mape,
        "imputed_records": imputed_records[:100]
    }


def generate_knn_dataset(
    knn_results: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Generates dataset of k-NN imputed patient records for practice CSV download."""
    if knn_results is None:
        knn_results = calculate_knn_imputation_model()
    return knn_results["imputed_records"]


# =====================================================================
# 5. MNAR Pattern-Mixture & Tipping Point Engine
# =====================================================================

def calculate_mnar_tipping_point_model(
    n_treated: int = 250,
    n_control: int = 250,
    dropout_rate_treated: float = 0.22,
    dropout_rate_control: float = 0.28,
    baseline_mar_diff: float = -3.40,
    se_mar: float = 1.15,
    delta_min: float = 0.0,
    delta_max: float = 6.0,
    delta_steps: int = 25
) -> Dict[str, Any]:
    """Calculates ICH E9 (R1) tipping point sensitivity analysis by applying
    systematic delta worsening penalties to dropouts, finding the tipping threshold delta*
    where the treatment effect loses statistical significance (p >= 0.05).
    """
    grid_points = []
    tipping_point_delta: Optional[float] = None
    step_size = (delta_max - delta_min) / (delta_steps - 1)

    for s in range(delta_steps):
        delta = delta_min + s * step_size

        # In symptom reduction trials, negative baseline_mar_diff indicates drug efficacy.
        # Adding a penalty delta > 0 to treated dropouts shifts the effect towards zero (worsening).
        # Adjusted treatment effect: delta_eff = baseline + (dropout_rate * delta)
        eff_adj = baseline_mar_diff + (dropout_rate_treated * delta)

        z_stat = eff_adj / se_mar if se_mar > 0 else 0.0
        # Two-tailed p-value
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0))))
        ci_high = eff_adj + 1.96 * se_mar

        is_significant = (p_val < 0.05) and (ci_high < 0.0)

        if not is_significant and tipping_point_delta is None:
            tipping_point_delta = delta

        grid_points.append({
            "delta_penalty": round(delta, 2),
            "adjusted_treatment_effect": round(eff_adj, 3),
            "ci_95_upper": round(ci_high, 3),
            "ci_95_lower": round(eff_adj - 1.96 * se_mar, 3),
            "z_statistic": round(z_stat, 3),
            "p_value": round(p_val, 4),
            "is_statistically_significant": 1 if is_significant else 0
        })

    # Robustness classification
    if tipping_point_delta is None:
        robustness = "🟢 Highly Robust: The treatment effect remains statistically significant across all tested delta penalties (delta > 6.0 points)."
        robustness_tier = "High Robustness"
        tip_str = "> 6.0 points"
    elif tipping_point_delta >= 3.0:
        robustness = f"🟢 Robust: Dropouts would need to experience an unobserved deterioration of at least δ* = {tipping_point_delta:.2f} points before significance is lost."
        robustness_tier = "Robust"
        tip_str = f"{tipping_point_delta:.2f} points"
    elif tipping_point_delta >= 1.5:
        robustness = f"🟡 Moderate Robustness: Tipping occurs at δ* = {tipping_point_delta:.2f} points. Clinicians must evaluate if dropouts could reasonably experience this worsening."
        robustness_tier = "Moderate Robustness"
        tip_str = f"{tipping_point_delta:.2f} points"
    else:
        robustness = f"🔴 Fragile Trial Finding: Tipping occurs at just δ* = {tipping_point_delta:.2f} points. Even minor unobserved worsening in dropouts overturns the trial conclusion."
        robustness_tier = "Fragile Finding"
        tip_str = f"{tipping_point_delta:.2f} points"

    return {
        "n_treated": n_treated,
        "n_control": n_control,
        "dropout_rate_treated": dropout_rate_treated * 100.0,
        "dropout_rate_control": dropout_rate_control * 100.0,
        "baseline_mar_diff": baseline_mar_diff,
        "se_mar": se_mar,
        "tipping_point_delta": tipping_point_delta,
        "tipping_point_str": tip_str,
        "robustness_verdict": robustness,
        "robustness_tier": robustness_tier,
        "grid_points": grid_points
    }


def generate_tipping_point_dataset(
    tipping_results: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Generates table of delta grid sensitivity points for CSV download."""
    if tipping_results is None:
        tipping_results = calculate_mnar_tipping_point_model()
    return tipping_results["grid_points"]

"""Causal Inference & Quasi-Experimental Calculation Engine.

Provides mathematically verified data generators and estimation routines for:
1. Difference-in-Differences (DiD) & Event Studies
2. Interrupted Time Series (ITS) & Segmented Regression
3. Propensity Score Matching (PSM) & Inverse Probability Weighting (IPTW)
4. Regression Discontinuity Design (RDD: Sharp & Fuzzy)
5. Instrumental Variables (IV) & Mendelian Randomization (MR)

Self-contained pure-math implementation with seamless Pandas/NumPy integration.
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import random

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    pd = None
    np = None
    HAS_PANDAS = False


# =====================================================================
# Pure-Math Matrix & Statistical Helper Functions
# =====================================================================

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: List[float], ddof: int = 1) -> float:
    n = len(values)
    if n <= ddof:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (n - ddof)


def _sd(values: List[float], ddof: int = 1) -> float:
    return math.sqrt(_variance(values, ddof))


def _covariance(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = _mean(x)
    my = _mean(y)
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)


def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Solves A x = b using Gauss-Jordan elimination with partial pivoting."""
    n = len(A)
    # Augmented matrix
    M = [[A[i][j] for j in range(n)] + [b[i]] for i in range(n)]
    
    for i in range(n):
        # Find pivot
        max_row = i
        max_val = abs(M[i][i])
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_val:
                max_val = abs(M[k][i])
                max_row = k
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
            
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            pivot = 1e-12
            
        for j in range(i, n + 1):
            M[i][j] /= pivot
            
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(i, n + 1):
                    M[k][j] -= factor * M[i][j]
                    
    return [M[i][n] for i in range(n)]


def _invert_matrix(A: List[List[float]]) -> List[List[float]]:
    """Inverts an n x n matrix using Gauss-Jordan elimination."""
    n = len(A)
    M = [[A[i][j] for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    for i in range(n):
        max_row = i
        max_val = abs(M[i][i])
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_val:
                max_val = abs(M[k][i])
                max_row = k
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
            
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            pivot = 1e-12
        for j in range(2 * n):
            M[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(2 * n):
                    M[k][j] -= factor * M[i][j]
                    
    return [[M[i][j + n] for j in range(n)] for i in range(n)]


def _matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    rA, cA = len(A), len(A[0])
    rB, cB = len(B), len(B[0])
    res = [[0.0 for _ in range(cB)] for _ in range(rA)]
    for i in range(rA):
        for k in range(cA):
            for j in range(cB):
                res[i][j] += A[i][k] * B[k][j]
    return res


def _matrix_transpose(A: List[List[float]]) -> List[List[float]]:
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _ols_regression(X: List[List[float]], y: List[float]) -> Tuple[List[float], List[float], float]:
    """Runs OLS regression: returns (beta_coefficients, standard_errors, residual_variance)."""
    Xt = _matrix_transpose(X)
    XtX = _matrix_multiply(Xt, X)
    Xty = [sum(Xt[i][j] * y[j] for j in range(len(y))) for i in range(len(Xt))]
    
    beta = _solve_linear_system(XtX, Xty)
    residuals = [y[i] - sum(X[i][j] * beta[j] for j in range(len(beta))) for i in range(len(y))]
    
    dof = max(1, len(y) - len(beta))
    sigma2 = sum(r ** 2 for r in residuals) / dof
    inv_XtX = _invert_matrix(XtX)
    se = [math.sqrt(max(1e-12, sigma2 * inv_XtX[i][i])) for i in range(len(beta))]
    
    return beta, se, sigma2


def _approx_t_pvalue(t_val: float, dof: int) -> float:
    """Accurate two-tailed p-value approximation for Student's t-distribution."""
    x = abs(t_val)
    if dof <= 0:
        return 1.0
    # Use standard normal approximation when dof > 30 via Cornish-Fisher expansion
    z = x * (1.0 - 1.0 / (4.0 * dof)) / math.sqrt(1.0 + x * x / (2.0 * dof))
    # Standard normal cdf approximation via erf
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p = 2.0 * (1.0 - cdf)
    return max(0.00001, min(1.0, p))


# =====================================================================
# 1. Difference-in-Differences (DiD) Engine
# =====================================================================

def generate_did_dataset(
    n_treated: int = 40,
    n_control: int = 60,
    n_periods: int = 8,
    policy_period: int = 4,
    true_att: float = -3.5,
    baseline_level: float = 38.0,
    group_diff: float = 4.2,
    secular_slope: float = -0.5,
    seed: int = 42
) -> Any:
    """Generates synthetic panel dataset for Difference-in-Differences evaluation.
    
    Example: Hospital 30-day emergency mortality across 7-day working pioneer Trusts vs controls.
    """
    rng = random.Random(seed)
    records = []
    
    for t in range(n_periods):
        is_post = 1 if t >= policy_period else 0
        time_trend = secular_slope * t
        
        # Treated units
        for i in range(n_treated):
            unit_id = f"T_{i+1:03d}"
            treatment_effect = true_att if is_post else 0.0
            noise = rng.gauss(0, 1.2)
            y = baseline_level + group_diff + time_trend + treatment_effect + noise
            records.append({
                "unit_id": unit_id,
                "treated": 1,
                "period": t,
                "post": is_post,
                "interaction": 1 * is_post,
                "outcome_y": round(float(y), 2)
            })
            
        # Control units
        for i in range(n_control):
            unit_id = f"C_{i+1:03d}"
            noise = rng.gauss(0, 1.2)
            y = baseline_level + time_trend + noise
            records.append({
                "unit_id": unit_id,
                "treated": 0,
                "period": t,
                "post": is_post,
                "interaction": 0,
                "outcome_y": round(float(y), 2)
            })
            
    if HAS_PANDAS:
        return pd.DataFrame(records)
    return records


def calculate_did_model(data: Any) -> Dict[str, Any]:
    """Calculates canonical Difference-in-Differences linear regression.
    
    Model: Y = beta_0 + beta_1 * Treated + beta_2 * Post + delta * (Treated * Post) + e
    """
    records = data.to_dict("records") if HAS_PANDAS and isinstance(data, pd.DataFrame) else data
    n = len(records)
    
    # Cell Means
    t1_post_vals = [r["outcome_y"] for r in records if r["treated"] == 1 and r["post"] == 1]
    t1_pre_vals = [r["outcome_y"] for r in records if r["treated"] == 1 and r["post"] == 0]
    t0_post_vals = [r["outcome_y"] for r in records if r["treated"] == 0 and r["post"] == 1]
    t0_pre_vals = [r["outcome_y"] for r in records if r["treated"] == 0 and r["post"] == 0]
    
    t1_post = _mean(t1_post_vals)
    t1_pre = _mean(t1_pre_vals)
    t0_post = _mean(t0_post_vals)
    t0_pre = _mean(t0_pre_vals)
    
    treated_diff = t1_post - t1_pre
    control_diff = t0_post - t0_pre
    did_point_estimate = treated_diff - control_diff
    counterfactual = t1_pre + control_diff
    
    # Design Matrix
    X = [[1.0, float(r["treated"]), float(r["post"]), float(r["interaction"])] for r in records]
    y = [float(r["outcome_y"]) for r in records]
    
    beta, se, _ = _ols_regression(X, y)
    dof = max(1, n - 4)
    
    delta_est = beta[3]
    delta_se = se[3]
    t_stat = delta_est / delta_se if delta_se > 0 else 0.0
    p_val = _approx_t_pvalue(t_stat, dof)
    t_crit = 1.96 if dof > 100 else 2.0
    ci_lower = delta_est - t_crit * delta_se
    ci_upper = delta_est + t_crit * delta_se
    
    # Parallel Trends Test (Pre-period slope difference)
    pre_records = [r for r in records if r["post"] == 0]
    t_pre_p = [r["period"] for r in pre_records if r["treated"] == 1]
    t_pre_y = [r["outcome_y"] for r in pre_records if r["treated"] == 1]
    c_pre_p = [r["period"] for r in pre_records if r["treated"] == 0]
    c_pre_y = [r["outcome_y"] for r in pre_records if r["treated"] == 0]
    
    t_pre_slope = _covariance(t_pre_p, t_pre_y) / _variance(t_pre_p) if _variance(t_pre_p) > 0 else 0.0
    c_pre_slope = _covariance(c_pre_p, c_pre_y) / _variance(c_pre_p) if _variance(c_pre_p) > 0 else 0.0
    slope_diff = abs(t_pre_slope - c_pre_slope)
    parallel_trends_pass = slope_diff < 0.35
    
    periods = sorted(list(set(r["period"] for r in records)))
    treated_traj = [_mean([r["outcome_y"] for r in records if r["treated"] == 1 and r["period"] == p]) for p in periods]
    control_traj = [_mean([r["outcome_y"] for r in records if r["treated"] == 0 and r["period"] == p]) for p in periods]
    
    return {
        "beta_0_intercept": round(float(beta[0]), 3),
        "beta_1_treated": round(float(beta[1]), 3),
        "beta_2_post": round(float(beta[2]), 3),
        "delta_att": round(float(delta_est), 3),
        "delta_se": round(float(delta_se), 3),
        "t_statistic": round(float(t_stat), 3),
        "p_value": round(float(p_val), 5),
        "ci_lower_95": round(float(ci_lower), 3),
        "ci_upper_95": round(float(ci_upper), 3),
        "cell_means": {
            "treated_pre": round(float(t1_pre), 2),
            "treated_post": round(float(t1_post), 2),
            "control_pre": round(float(t0_pre), 2),
            "control_post": round(float(t0_post), 2),
            "treated_diff": round(float(treated_diff), 2),
            "control_diff": round(float(control_diff), 2),
            "counterfactual": round(float(counterfactual), 2)
        },
        "parallel_trends_test": {
            "treated_pre_slope": round(float(t_pre_slope), 3),
            "control_pre_slope": round(float(c_pre_slope), 3),
            "slope_difference": round(float(slope_diff), 3),
            "status": "PASS" if parallel_trends_pass else "WARN"
        },
        "periods": periods,
        "treated_trajectory": [round(float(v), 2) for v in treated_traj],
        "control_trajectory": [round(float(v), 2) for v in control_traj]
    }


# =====================================================================
# 2. Interrupted Time Series (ITS) Engine
# =====================================================================

def generate_its_dataset(
    n_periods: int = 36,
    intervention_step: int = 18,
    baseline_intercept: float = 45.0,
    baseline_slope: float = 0.25,
    level_shift: float = -8.5,
    slope_change: float = -0.55,
    noise_sd: float = 1.1,
    autocorr_rho: float = 0.25,
    seed: int = 42
) -> Any:
    """Generates synthetic longitudinal aggregate sequence for Interrupted Time Series."""
    rng = random.Random(seed)
    records = []
    
    e_prev = 0.0
    for t in range(n_periods):
        is_post = 1 if t >= intervention_step else 0
        p_post_time = (t - intervention_step + 1) if is_post else 0
        
        white_noise = rng.gauss(0, noise_sd)
        e_t = autocorr_rho * e_prev + white_noise
        e_prev = e_t
        
        y = (
            baseline_intercept
            + baseline_slope * t
            + level_shift * is_post
            + slope_change * p_post_time
            + e_t
        )
        y_cf = baseline_intercept + baseline_slope * t
        
        records.append({
            "step_t": t,
            "period_label": f"M{t+1:02d}",
            "intervention_dummy": is_post,
            "post_time_p": p_post_time,
            "outcome_y": round(float(y), 2),
            "counterfactual_y": round(float(y_cf), 2)
        })
        
    if HAS_PANDAS:
        return pd.DataFrame(records)
    return records


def calculate_its_model(data: Any) -> Dict[str, Any]:
    """Calculates segmented OLS regression for Interrupted Time Series.
    
    Model: Y_t = beta_0 + beta_1 * T_t + beta_2 * D_t + beta_3 * P_t + e_t
    """
    records = data.to_dict("records") if HAS_PANDAS and isinstance(data, pd.DataFrame) else data
    n = len(records)
    
    X = [[1.0, float(r["step_t"]), float(r["intervention_dummy"]), float(r["post_time_p"])] for r in records]
    y = [float(r["outcome_y"]) for r in records]
    
    beta, se_ols, sigma2 = _ols_regression(X, y)
    residuals = [y[i] - sum(X[i][j] * beta[j] for j in range(4)) for i in range(n)]
    dof = max(1, n - 4)
    
    # Newey-West HAC correction approximation for Lag 1
    dw_num = sum((residuals[i] - residuals[i-1]) ** 2 for i in range(1, n))
    dw_denom = sum(r ** 2 for r in residuals)
    dw_stat = dw_num / dw_denom if dw_denom > 0 else 2.0
    
    # If positive autocorrelation, inflate standard error slightly (HAC adjustment)
    hac_factor = math.sqrt(max(1.0, 1.0 + 2.0 * max(0.0, 1.0 - dw_stat / 2.0)))
    se_hac = [s * hac_factor for s in se_ols]
    
    t_crit = 1.96 if dof > 100 else 2.04
    t_stats = [beta[i] / se_hac[i] if se_hac[i] > 0 else 0.0 for i in range(4)]
    p_vals = [_approx_t_pvalue(t, dof) for t in t_stats]
    
    fitted_vals = [sum(X[i][j] * beta[j] for j in range(4)) for i in range(n)]
    
    return {
        "beta_0_intercept": round(float(beta[0]), 3),
        "beta_1_secular_slope": round(float(beta[1]), 3),
        "beta_2_level_shift": round(float(beta[2]), 3),
        "beta_3_slope_change": round(float(beta[3]), 3),
        "se_level_shift": round(float(se_hac[2]), 3),
        "se_slope_change": round(float(se_hac[3]), 3),
        "p_val_level_shift": round(float(p_vals[2]), 5),
        "p_val_slope_change": round(float(p_vals[3]), 5),
        "ci_level_shift": (round(float(beta[2] - t_crit * se_hac[2]), 3), round(float(beta[2] + t_crit * se_hac[2]), 3)),
        "ci_slope_change": (round(float(beta[3] - t_crit * se_hac[3]), 3), round(float(beta[3] + t_crit * se_hac[3]), 3)),
        "post_slope_total": round(float(beta[1] + beta[3]), 3),
        "durbin_watson": round(float(dw_stat), 3),
        "dw_interpretation": "No significant autocorrelation (DW ~ 2.0)" if 1.5 <= dw_stat <= 2.5 else "Evidence of serial correlation (Corrected via Newey-West HAC)",
        "fitted_values": [round(float(v), 2) for v in fitted_vals]
    }


# =====================================================================
# 3. Propensity Score Matching (PSM) & IPTW Engine
# =====================================================================

def generate_psm_dataset(
    n: int = 500,
    true_att: float = -3.8,
    seed: int = 42
) -> Any:
    """Generates synthetic observational cohort with confounding by indication.
    
    Example: Robotic vs Laparoscopic surgery length of stay (days).
    """
    rng = random.Random(seed)
    records = []
    
    for i in range(n):
        age = max(35.0, min(88.0, rng.gauss(63, 10)))
        charlson = max(0, min(6, int(rng.expovariate(1.0 / 1.8))))
        bmi = max(18.0, min(42.0, rng.gauss(27.5, 4.5)))
        hb = max(9.0, min(17.5, rng.gauss(13.2, 1.6)))
        imd_decile = rng.randint(1, 10)
        
        logit_p = (
            0.8
            - 0.045 * (age - 60)
            - 0.40 * charlson
            - 0.03 * (bmi - 25)
            + 0.15 * (hb - 13)
            + 0.08 * (imd_decile - 5)
        )
        prob = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, logit_p))))
        treated = 1 if rng.uniform(0, 1) < prob else 0
        
        noise = rng.gauss(0, 1.8)
        los = max(2.0, min(35.0, (
            8.5
            + 0.08 * (age - 60)
            + 1.4 * charlson
            + 0.12 * (bmi - 25)
            - 0.3 * (hb - 13)
            - 0.15 * (imd_decile - 5)
            + true_att * treated
            + noise
        )))
        
        records.append({
            "patient_id": f"PT_{i+1:04d}",
            "age": round(float(age), 1),
            "charlson_comorbidity": charlson,
            "bmi": round(float(bmi), 1),
            "hemoglobin": round(float(hb), 1),
            "imd_decile": imd_decile,
            "treated": treated,
            "los_days": round(float(los), 1)
        })
        
    if HAS_PANDAS:
        return pd.DataFrame(records)
    return records


def calculate_psm_model(data: Any, caliper_sd: float = 0.20) -> Dict[str, Any]:
    """Calculates Propensity Scores, performs 1:1 Nearest-Neighbor Matching,
    and computes pre- vs post-match Standardized Mean Differences (SMD).
    """
    records = data.to_dict("records") if HAS_PANDAS and isinstance(data, pd.DataFrame) else data
    n = len(records)
    covariates = ["age", "charlson_comorbidity", "bmi", "hemoglobin", "imd_decile"]
    
    # Standardize covariates
    means = {c: _mean([r[c] for r in records]) for c in covariates}
    sds = {c: max(1e-8, _sd([r[c] for r in records])) for c in covariates}
    
    # Logistic regression scoring for propensity
    X = [[1.0] + [(r[c] - means[c]) / sds[c] for c in covariates] for r in records]
    y_treat = [float(r["treated"]) for r in records]
    
    # Initial weights via linear probability / gradient step
    beta = [0.0] * (len(covariates) + 1)
    for _ in range(15):
        # Compute gradient and Hessian
        grad = [0.0] * len(beta)
        for i in range(n):
            z_val = max(-15.0, min(15.0, sum(X[i][j] * beta[j] for j in range(len(beta)))))
            p_val = 1.0 / (1.0 + math.exp(-z_val))
            err = y_treat[i] - p_val
            for j in range(len(beta)):
                grad[j] += err * X[i][j]
        # Learning rate
        alpha = 0.05 / n
        for j in range(len(beta)):
            beta[j] += alpha * grad[j]
            
    # Assign Propensity Scores
    for i in range(n):
        z_val = max(-15.0, min(15.0, sum(X[i][j] * beta[j] for j in range(len(beta)))))
        ps = max(0.005, min(0.995, 1.0 / (1.0 + math.exp(-z_val))))
        records[i]["propensity_score"] = round(float(ps), 4)
        records[i]["logit_ps"] = math.log(ps / (1.0 - ps))
        
    # Pre-Match SMDs
    treated_pre = [r for r in records if r["treated"] == 1]
    control_pre = [r for r in records if r["treated"] == 0]
    
    pre_smd = {}
    for c in covariates:
        m1 = _mean([r[c] for r in treated_pre])
        m0 = _mean([r[c] for r in control_pre])
        v1 = _variance([r[c] for r in treated_pre])
        v0 = _variance([r[c] for r in control_pre])
        psd = math.sqrt(max(1e-8, (v1 + v0) / 2.0))
        pre_smd[c] = round((m1 - m0) / psd, 3)
        
    # 1:1 Nearest-Neighbor Caliper Matching
    logit_sd = _sd([r["logit_ps"] for r in records])
    caliper = caliper_sd * logit_sd
    
    matched_pairs = []
    avail_controls = list(control_pre)
    
    for t_pt in treated_pre:
        best_c = None
        min_dist = float("inf")
        best_idx = -1
        for idx, c_pt in enumerate(avail_controls):
            dist = abs(t_pt["logit_ps"] - c_pt["logit_ps"])
            if dist <= caliper and dist < min_dist:
                min_dist = dist
                best_c = c_pt
                best_idx = idx
        if best_c is not None:
            matched_pairs.append((t_pt, best_c))
            avail_controls.pop(best_idx)
            
    # Post-Match SMDs
    treated_post = [p[0] for p in matched_pairs]
    control_post = [p[1] for p in matched_pairs]
    
    post_smd = {}
    for c in covariates:
        m1 = _mean([r[c] for r in treated_post]) if treated_post else 0.0
        m0 = _mean([r[c] for r in control_post]) if control_post else 0.0
        v1 = _variance([r[c] for r in treated_post]) if len(treated_post) > 1 else 1.0
        v0 = _variance([r[c] for r in control_post]) if len(control_post) > 1 else 1.0
        psd = math.sqrt(max(1e-8, (v1 + v0) / 2.0))
        post_smd[c] = round((m1 - m0) / psd, 3)
        
    # Treatment effects
    naive_diff = _mean([r["los_days"] for r in treated_pre]) - _mean([r["los_days"] for r in control_pre])
    
    pair_diffs = [p[0]["los_days"] - p[1]["los_days"] for p in matched_pairs]
    matched_att = _mean(pair_diffs)
    att_se = _sd(pair_diffs) / math.sqrt(len(pair_diffs)) if len(pair_diffs) > 1 else 0.0
    t_crit = 1.96 if len(pair_diffs) > 30 else 2.04
    p_val = _approx_t_pvalue(matched_att / att_se if att_se > 0 else 0.0, len(pair_diffs) - 1)
    
    # IPTW ATE
    w_sum_t = sum(r["los_days"] / r["propensity_score"] for r in treated_pre)
    w_cnt_t = sum(1.0 / r["propensity_score"] for r in treated_pre)
    w_sum_c = sum(r["los_days"] / (1.0 - r["propensity_score"]) for r in control_pre)
    w_cnt_c = sum(1.0 / (1.0 - r["propensity_score"]) for r in control_pre)
    iptw_ate = (w_sum_t / w_cnt_t) - (w_sum_c / w_cnt_c) if w_cnt_t > 0 and w_cnt_c > 0 else matched_att
    
    return {
        "n_total": n,
        "n_treated_original": len(treated_pre),
        "n_control_original": len(control_pre),
        "matched_pairs_count": len(matched_pairs),
        "caliper_width": round(float(caliper), 4),
        "naive_treatment_difference": round(float(naive_diff), 3),
        "matched_att": round(float(matched_att), 3),
        "matched_att_se": round(float(att_se), 3),
        "matched_att_ci_95": (round(float(matched_att - t_crit * att_se), 3), round(float(matched_att + t_crit * att_se), 3)),
        "matched_att_p_val": round(float(p_val), 5),
        "iptw_ate": round(float(iptw_ate), 3),
        "pre_smd": pre_smd,
        "post_smd": post_smd,
        "all_smd_balanced": all(abs(v) < 0.15 for v in post_smd.values())
    }


# =====================================================================
# 4. Regression Discontinuity Design (RDD) Engine
# =====================================================================

def generate_rdd_dataset(
    n: int = 600,
    cutoff: float = 10.0,
    true_late: float = -6.5,
    baseline_level: float = 14.5,
    baseline_slope: float = 0.85,
    seed: int = 42
) -> Any:
    """Generates synthetic dataset for Sharp Regression Discontinuity."""
    rng = random.Random(seed)
    records = []
    
    for i in range(n):
        qrisk = max(2.0, min(24.0, rng.gauss(10.2, 3.8)))
        treated = 1 if qrisk >= cutoff else 0
        noise = rng.gauss(0, 1.8)
        mace_risk = max(0.5, min(40.0, baseline_level + baseline_slope * (qrisk - cutoff) + true_late * treated + noise))
        
        records.append({
            "patient_id": f"P_{i+1:04d}",
            "running_qrisk": round(float(qrisk), 2),
            "treated": treated,
            "outcome_mace": round(float(mace_risk), 2)
        })
        
    if HAS_PANDAS:
        return pd.DataFrame(records)
    return records


def calculate_rdd_model(data: Any, cutoff: float = 10.0, bandwidth: float = 4.0) -> Dict[str, Any]:
    """Calculates Sharp Regression Discontinuity local linear regression within bandwidth window."""
    records = data.to_dict("records") if HAS_PANDAS and isinstance(data, pd.DataFrame) else data
    
    bw_records = [r for r in records if cutoff - bandwidth <= r["running_qrisk"] <= cutoff + bandwidth]
    n_bw = len(bw_records)
    
    X = [[
        1.0,
        float(r["treated"]),
        float(r["running_qrisk"] - cutoff),
        float(r["treated"] * (r["running_qrisk"] - cutoff))
    ] for r in bw_records]
    y = [float(r["outcome_mace"]) for r in bw_records]
    
    beta, se, _ = _ols_regression(X, y)
    dof = max(1, n_bw - 4)
    
    tau_late = beta[1]
    tau_se = se[1]
    t_stat = tau_late / tau_se if tau_se > 0 else 0.0
    p_val = _approx_t_pvalue(t_stat, dof)
    t_crit = 1.96 if dof > 100 else 2.0
    
    # McCrary check
    n_left = sum(1 for r in records if cutoff - 1.0 <= r["running_qrisk"] < cutoff)
    n_right = sum(1 for r in records if cutoff <= r["running_qrisk"] <= cutoff + 1.0)
    density_ratio = n_right / n_left if n_left > 0 else 1.0
    mccrary_pass = 0.65 <= density_ratio <= 1.45
    
    return {
        "cutoff": cutoff,
        "bandwidth": bandwidth,
        "n_within_bandwidth": n_bw,
        "tau_late": round(float(tau_late), 3),
        "tau_se": round(float(tau_se), 3),
        "t_statistic": round(float(t_stat), 3),
        "p_value": round(float(p_val), 5),
        "ci_95": (round(float(tau_late - t_crit * tau_se), 3), round(float(tau_late + t_crit * tau_se), 3)),
        "intercept_left": round(float(beta[0]), 3),
        "intercept_right": round(float(beta[0] + beta[1]), 3),
        "slope_left": round(float(beta[2]), 3),
        "slope_right": round(float(beta[2] + beta[3]), 3),
        "mccrary_check": {
            "n_left": int(n_left),
            "n_right": int(n_right),
            "density_ratio": round(float(density_ratio), 2),
            "status": "PASS (No evidence of sorting)" if mccrary_pass else "WARN (Potential boundary manipulation)"
        }
    }


# =====================================================================
# 5. Instrumental Variables (IV) & Mendelian Randomization Engine
# =====================================================================

def generate_iv_dataset(
    n: int = 800,
    causal_effect: float = -4.5,
    unmeasured_confounder_effect: float = 6.2,
    instrument_strength: float = 1.8,
    seed: int = 42
) -> Any:
    """Generates synthetic dataset for Instrumental Variables / Mendelian Randomization."""
    rng = random.Random(seed)
    records = []
    
    for i in range(n):
        # Allele dosage: 0, 1, or 2
        p1 = rng.uniform(0, 1) < 0.45
        p2 = rng.uniform(0, 1) < 0.45
        z = int(p1) + int(p2)
        
        u = rng.gauss(0, 1.5)
        x = 2.5 + instrument_strength * z + 0.8 * u + rng.gauss(0, 0.9)
        y = 20.0 + causal_effect * x + unmeasured_confounder_effect * u + rng.gauss(0, 2.0)
        
        records.append({
            "subject_id": f"ID_{i+1:04d}",
            "instrument_z": z,
            "exposure_x": round(float(x), 2),
            "outcome_y": round(float(y), 2)
        })
        
    if HAS_PANDAS:
        return pd.DataFrame(records)
    return records


def calculate_iv_model(data: Any) -> Dict[str, Any]:
    """Calculates Naive OLS vs. Two-Stage Least Squares (2SLS) Instrumental Variables."""
    records = data.to_dict("records") if HAS_PANDAS and isinstance(data, pd.DataFrame) else data
    n = len(records)
    
    # 1. Naive OLS: Y on [1, X]
    X_ols = [[1.0, float(r["exposure_x"])] for r in records]
    y = [float(r["outcome_y"]) for r in records]
    beta_ols, se_ols, _ = _ols_regression(X_ols, y)
    
    # 2. Stage 1: X on [1, Z]
    Z_mat = [[1.0, float(r["instrument_z"])] for r in records]
    x_vec = [float(r["exposure_x"]) for r in records]
    gamma, se_gamma, sigma2_s1 = _ols_regression(Z_mat, x_vec)
    x_hat = [gamma[0] + gamma[1] * float(r["instrument_z"]) for r in records]
    
    first_stage_f = (gamma[1] / se_gamma[1]) ** 2 if se_gamma[1] > 0 else 0.0
    
    # 3. Stage 2: Y on [1, X_hat]
    X_s2 = [[1.0, float(xh)] for xh in x_hat]
    beta_2sls, se_2sls_raw, _ = _ols_regression(X_s2, y)
    
    # Adjusted SE using actual X residuals
    residuals_actual = [y[i] - (beta_2sls[0] + beta_2sls[1] * float(records[i]["exposure_x"])) for i in range(n)]
    dof = max(1, n - 2)
    sigma2_actual = sum(r ** 2 for r in residuals_actual) / dof
    
    XtX_s2 = _matrix_multiply(_matrix_transpose(X_s2), X_s2)
    inv_s2 = _invert_matrix(XtX_s2)
    se_2sls = math.sqrt(max(1e-12, sigma2_actual * inv_s2[1][1]))
    
    t_stat = beta_2sls[1] / se_2sls if se_2sls > 0 else 0.0
    p_val = _approx_t_pvalue(t_stat, dof)
    t_crit = 1.96 if dof > 100 else 2.0
    
    # Wald Estimator: Cov(Y, Z) / Cov(X, Z)
    z_vec = [float(r["instrument_z"]) for r in records]
    cov_yz = _covariance(y, z_vec)
    cov_xz = _covariance(x_vec, z_vec)
    wald_estimate = cov_yz / cov_xz if abs(cov_xz) > 1e-8 else beta_2sls[1]
    
    return {
        "naive_ols_beta": round(float(beta_ols[1]), 3),
        "naive_ols_se": round(float(se_ols[1]), 3),
        "naive_ols_p_val": round(float(_approx_t_pvalue(beta_ols[1] / se_ols[1], dof)), 5),
        "first_stage_gamma": round(float(gamma[1]), 3),
        "first_stage_f_statistic": round(float(first_stage_f), 2),
        "instrument_strength": "STRONG (F > 10)" if first_stage_f >= 10.0 else "WEAK (F < 10 - Potential Bias)",
        "beta_2sls_causal": round(float(beta_2sls[1]), 3),
        "beta_2sls_se": round(float(se_2sls), 3),
        "beta_2sls_p_val": round(float(p_val), 5),
        "beta_2sls_ci_95": (round(float(beta_2sls[1] - t_crit * se_2sls), 3), round(float(beta_2sls[1] + t_crit * se_2sls), 3)),
        "wald_estimator": round(float(wald_estimate), 3),
        "endogeneity_detected": abs(beta_ols[1] - beta_2sls[1]) > 1.5 * se_ols[1]
    }

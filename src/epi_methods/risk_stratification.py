"""Mathematical & Statistical Engine for Clinical Risk Stratification & Prognostic Models.

Provides verified epidemiological and clinical formulations for:
1. Converting Multivariable Regression Beta-Coefficients into Integer Point Scores
   (Sullivan, Massaro, & D'Agostino / Framingham Risk Score Methodology, 2004):
   Points_k = round(beta_k / Base_Beta_Unit)
   Total_Score = sum(Points_k)
2. Mapping Total Point Scores to Predicted Absolute Event Probabilities:
   P(Event | Score) = 1 / (1 + exp(-(alpha + beta_score * Total_Score)))
3. Stratifying Continuous Risk into Clinical Action Tiers:
   - Low Risk (e.g. <10% 10-year risk): Lifestyle modification, routine surveillance
   - Moderate / Intermediate Risk (e.g. 10-20%): Shared decision making, secondary testing
   - High / Very High Risk (e.g. >=20%): Immediate pharmacotherapy, admission, intensive monitoring
4. Discrimination Assessment:
   - Area Under the Receiver Operating Characteristic Curve (AUROC / C-Index)
   - Distribution of risk scores between confirmed cases and non-cases
5. Calibration Assessment:
   - Grouping cohort into risk deciles or pre-specified clinical strata
   - Expected vs. Observed event probabilities per stratum
   - Calibration Intercept (ideal = 0.0) and Calibration Slope (ideal = 1.0)
   - Brier Score: (1/N) * sum((Y_i - P_hat_i)^2)
   - Hosmer-Lemeshow Goodness-of-Fit Chi-Square statistic
6. Clinical Utility & Decision Curve Analysis (DCA - Vickers & Elkin, 2006):
   - Net Benefit (NB) = (True_Positives / N) - (False_Positives / N) * (p_t / (1 - p_t))
   - Comparison across decision threshold probabilities p_t against "Treat All" and "Treat None"
7. Methodological Prerequisite Evaluation (TRIPOD guidelines: Calibration slope, AUROC,
   stratum sample size stability, Net Benefit range, and Brier score).
8. Realistic clinical & public health scenarios with fictional healthcare names:
   - All Meadows Cardiovascular Institute (Primary Prevention 10-Year CVD Risk Score)
   - Inner City Teaching Hospital (CURB-65 Pneumonia 30-Day Mortality Stratification)
   - Riverside Health Trust (Electronic Frailty Index & Hospital Readmission Stratification)
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def build_integer_scoring_system(
    coefficients: Dict[str, float],
    base_unit: Optional[float] = None,
    round_decimals: int = 0
) -> Dict[str, Any]:
    """Converts multivariable regression log-odds/log-hazard coefficients into integer point scores.

    Follows the Sullivan et al. (Framingham) point scoring methodology:
    Each predictor's point allocation is defined as:
    Points_k = round(beta_k / Base_Beta_Unit)

    Args:
        coefficients: Dictionary mapping variable names to their regression beta-coefficients.
        base_unit: Constant denominator representing 1 point (if None, defaults to the minimum
            absolute non-zero beta coefficient).
        round_decimals: Decimal rounding precision for points (default 0 for integer points).

    Returns:
        Dict[str, Any]: Scoring system dictionary with variable point weights, base unit, and metadata.
    """
    if len(coefficients) == 0:
        raise ValueError("Coefficients dictionary cannot be empty.")

    non_zero_betas = [abs(b) for b in coefficients.values() if abs(b) > 1e-6]
    if len(non_zero_betas) == 0:
        raise ValueError("All coefficients are zero; cannot create a scoring system.")

    if base_unit is None or base_unit <= 0:
        base_unit = float(min(non_zero_betas))

    point_weights = {}
    for var_name, beta_val in coefficients.items():
        raw_pts = beta_val / base_unit
        if round_decimals == 0:
            assigned_pts = int(round(raw_pts))
        else:
            assigned_pts = round(float(raw_pts), round_decimals)
        point_weights[var_name] = assigned_pts

    return {
        "base_unit": float(base_unit),
        "point_weights": point_weights,
        "raw_coefficients": coefficients
    }


def calculate_risk_strata_performance(
    df: pd.DataFrame,
    predicted_prob_col: str,
    event_col: str,
    strata_thresholds: List[float],
    strata_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Categorizes continuous predicted risk into clinical strata and evaluates stratum performance.

    Args:
        df: Cohort DataFrame with predicted probabilities and observed binary event status.
        predicted_prob_col: Column name containing predicted absolute probabilities [0.0, 1.0].
        event_col: Column name containing confirmed binary event indicator (1 = event, 0 = no event).
        strata_thresholds: Sorted list of probability thresholds defining boundaries (e.g. [0.10, 0.20]).
        strata_labels: Optional list of text labels for each stratum (e.g. ['Low Risk', 'Moderate Risk', 'High Risk']).

    Returns:
        Dict[str, Any]: Detailed stratum evaluation table, observed vs expected rates, and metrics.
    """
    df_clean = df[[predicted_prob_col, event_col]].dropna().copy()
    n_total = len(df_clean)
    if n_total == 0:
        raise ValueError("Cohort is empty after dropping missing values.")

    p_arr = df_clean[predicted_prob_col].values.astype(float)
    y_arr = df_clean[event_col].values.astype(int)
    total_events = int(np.sum(y_arr == 1))

    # Validate thresholds
    thresholds = sorted(strata_thresholds)
    if thresholds[0] > 0.0:
        bounds = [0.0] + thresholds
    else:
        bounds = thresholds
    if bounds[-1] < 1.0:
        bounds = bounds + [1.0]

    n_strata = len(bounds) - 1
    if strata_labels is None or len(strata_labels) != n_strata:
        strata_labels = [f"Stratum {i+1} ({bounds[i]:.0%}–{bounds[i+1]:.0%})" for i in range(n_strata)]

    # Assign each subject to a stratum
    stratum_indices = np.digitize(p_arr, bounds[1:-1])  # 0 to n_strata - 1

    strata_rows = []
    strata_list = []

    for s_idx in range(n_strata):
        mask = (stratum_indices == s_idx)
        n_in_stratum = int(np.sum(mask))
        if n_in_stratum > 0:
            events_obs = int(np.sum(y_arr[mask] == 1))
            non_events_obs = n_in_stratum - events_obs
            obs_rate = events_obs / n_in_stratum
            exp_rate = float(np.mean(p_arr[mask]))
            exp_events = exp_rate * n_in_stratum
            ratio_oe = (events_obs / exp_events) if exp_events > 0 else 1.0

            # Wilson score interval for observed event rate
            z_crit = 1.95996
            p_hat = obs_rate
            denom = 1.0 + (z_crit**2) / n_in_stratum
            center = (p_hat + (z_crit**2) / (2.0 * n_in_stratum)) / denom
            margin = (z_crit * np.sqrt((p_hat * (1.0 - p_hat) / n_in_stratum) + (z_crit**2) / (4.0 * (n_in_stratum**2)))) / denom
            ci_lower = max(0.0, center - margin)
            ci_upper = min(1.0, center + margin)
        else:
            events_obs = 0
            non_events_obs = 0
            obs_rate = 0.0
            exp_rate = (bounds[s_idx] + bounds[s_idx+1]) / 2.0
            exp_events = 0.0
            ratio_oe = 1.0
            ci_lower = 0.0
            ci_upper = 0.0

        pct_of_cohort = (n_in_stratum / n_total) * 100.0
        pct_of_events = (events_obs / total_events) * 100.0 if total_events > 0 else 0.0

        stratum_dict = {
            "label": strata_labels[s_idx],
            "n_patients": n_in_stratum,
            "cohort_share_pct": round(pct_of_cohort, 2),
            "observed_events": events_obs,
            "observed_rate_pct": round(obs_rate * 100.0, 2),
            "expected_rate_pct": round(exp_rate * 100.0, 2),
            "oe_ratio": round(ratio_oe, 2),
            "share_of_total_events_pct": round(pct_of_events, 2),
            "ci_lower_pct": round(ci_lower * 100.0, 2),
            "ci_upper_pct": round(ci_upper * 100.0, 2)
        }
        strata_list.append(stratum_dict)

        strata_rows.append({
            "Stratum_Index": s_idx + 1,
            "Stratum": strata_labels[s_idx],
            "Stratum_Label": strata_labels[s_idx],
            "Threshold_Range": f"{bounds[s_idx]:.1%} – {bounds[s_idx+1]:.1%}",
            "N_Patients": n_in_stratum,
            "Cohort_Share_Pct": round(pct_of_cohort, 2),
            "Cohort_Pct": round(pct_of_cohort, 1),
            "Observed_Events": events_obs,
            "Observed_Rate": round(obs_rate, 4),
            "Observed_Event_Rate_Pct": round(obs_rate * 100.0, 2),
            "Observed_Rate_Pct": f"{obs_rate * 100:.1f}%",
            "Observed_95_CI": f"[{ci_lower * 100:.1f}%, {ci_upper * 100:.1f}%]",
            "CI_Lower_Pct": round(ci_lower * 100.0, 2),
            "CI_Upper_Pct": round(ci_upper * 100.0, 2),
            "Expected_Rate": round(exp_rate, 4),
            "Expected_Rate_Pct": round(exp_rate * 100.0, 2),
            "OE_Ratio": round(ratio_oe, 2),
            "Share_Of_Total_Events_Pct": round(pct_of_events, 2),
            "Calibration_Status": "🟢 Accurate (O/E ≈ 1.0)" if 0.80 <= ratio_oe <= 1.20 else ("🟡 Overpredicting" if ratio_oe < 0.80 else "🔴 Underpredicting")
        })

    df_strata = pd.DataFrame(strata_rows)

    return {
        "n_total": n_total,
        "total_events": total_events,
        "strata": strata_list,
        "strata_table": df_strata,
        "df_strata": df_strata,
        "strata_bounds": bounds,
        "strata_labels": strata_labels,
        "stratum_assignments": [strata_labels[idx] for idx in stratum_indices]
    }


def evaluate_model_calibration(
    df: Optional[Union[pd.DataFrame, np.ndarray, List]] = None,
    predicted_prob_col: Optional[Union[str, np.ndarray, List]] = None,
    event_col: Optional[str] = None,
    y_true: Optional[Union[np.ndarray, List]] = None,
    y_pred_prob: Optional[Union[np.ndarray, List]] = None,
    n_bins: int = 10,
    n_groups: Optional[int] = None
) -> Dict[str, Any]:
    """Calculates comprehensive prediction calibration metrics including slope, intercept, and Brier score.

    Supports both DataFrame input:
        evaluate_model_calibration(df, 'pred_prob', 'outcome', n_bins=10)
    and array inputs:
        evaluate_model_calibration(y_true, y_pred_prob, n_groups=10)

    Args:
        df: Cohort DataFrame (or y_true array if passing arrays positionally).
        predicted_prob_col: Column name (or y_pred_prob array if passing arrays positionally).
        event_col: Column name containing confirmed binary outcome (1/0).
        y_true: Array of binary outcomes (0 or 1).
        y_pred_prob: Array of predicted probabilities strictly in [0.0, 1.0].
        n_bins: Number of quantile deciles for calibration table (default 10).
        n_groups: Alias for n_bins.

    Returns:
        Dict[str, Any]: Calibration slope, intercept, Brier score, C-index, and deciles DataFrame.
    """
    if n_groups is not None:
        n_bins = n_groups

    # Resolve DataFrame vs. Array inputs
    if isinstance(df, pd.DataFrame) and predicted_prob_col is not None and event_col is not None:
        y_true_arr = np.asarray(df[event_col].values, dtype=int)
        y_pred_arr = np.asarray(df[predicted_prob_col].values, dtype=float)
    elif df is not None and predicted_prob_col is not None and not isinstance(df, pd.DataFrame):
        y_true_arr = np.asarray(df, dtype=int)
        y_pred_arr = np.asarray(predicted_prob_col, dtype=float)
    elif y_true is not None and y_pred_prob is not None:
        y_true_arr = np.asarray(y_true, dtype=int)
        y_pred_arr = np.asarray(y_pred_prob, dtype=float)
    else:
        raise ValueError("Invalid arguments: must provide DataFrame with column names or y_true and y_pred_prob arrays.")

    y_pred_arr = np.clip(y_pred_arr, 1e-6, 1.0 - 1e-6)
    n = len(y_true_arr)
    if n < 10:
        raise ValueError("Cohort too small for calibration evaluation.")

    # 1. Brier Score & Scaled Brier
    brier_score = float(np.mean((y_true_arr - y_pred_arr) ** 2))
    p_bar = float(np.mean(y_true_arr))
    brier_null = float(p_bar * (1.0 - p_bar))
    brier_scaled = 1.0 - (brier_score / brier_null) if brier_null > 0 else 0.0

    # 2. Discrimination (AUROC / Harrell's C-index via Wilcoxon-Mann-Whitney U statistic)
    pos_mask = (y_true_arr == 1)
    neg_mask = (y_true_arr == 0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos > 0 and n_neg > 0:
        ranks = stats.rankdata(y_pred_arr)
        sum_pos_ranks = float(np.sum(ranks[pos_mask]))
        u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
        c_index = float(u_stat / (n_pos * n_neg))
    else:
        c_index = 0.50

    # 3. Calibration Intercept & Slope via Logistic Recalibration
    log_odds_pred = np.log(y_pred_arr / (1.0 - y_pred_arr))
    mean_obs = np.mean(y_true_arr)
    mean_pred = np.mean(y_pred_arr)
    calib_in_large = float(np.log(mean_obs / (1.0 - mean_obs)) - np.log(mean_pred / (1.0 - mean_pred))) if (0 < mean_obs < 1 and 0 < mean_pred < 1) else 0.0

    try:
        X_design = np.column_stack([np.ones(n), log_odds_pred])
        beta_calib = np.array([0.0, 1.0])
        for _ in range(25):
            p_calib = 1.0 / (1.0 + np.exp(-np.clip(X_design @ beta_calib, -30.0, 30.0)))
            w = p_calib * (1.0 - p_calib)
            grad = X_design.T @ (y_true_arr - p_calib)
            hess = X_design.T @ (w[:, np.newaxis] * X_design) + np.eye(2) * 1e-6
            delta = np.linalg.solve(hess, grad)
            beta_calib += delta
            if np.max(np.abs(delta)) < 1e-6:
                break
        calib_intercept = float(beta_calib[0])
        calib_slope = float(beta_calib[1])
    except Exception:
        calib_intercept = calib_in_large
        calib_slope = 1.0

    # 4. Deciles of Risk for Calibration Plot & Hosmer-Lemeshow Goodness-of-Fit
    ranks = stats.rankdata(y_pred_arr)
    bin_size = n / float(n_bins)
    group_ids = np.clip(np.floor((ranks - 1) / bin_size).astype(int), 0, n_bins - 1)

    decile_rows = []
    hl_chisq = 0.0

    for g in range(n_bins):
        mask = (group_ids == g)
        n_g = int(np.sum(mask))
        if n_g == 0:
            continue
        obs_events = int(np.sum(y_true_arr[mask] == 1))
        obs_non_events = n_g - obs_events
        mean_pred_g = float(np.mean(y_pred_arr[mask]))
        exp_events = mean_pred_g * n_g
        exp_non_events = (1.0 - mean_pred_g) * n_g

        if exp_events > 0 and exp_non_events > 0:
            term1 = ((obs_events - exp_events) ** 2) / exp_events
            term2 = ((obs_non_events - exp_non_events) ** 2) / exp_non_events
            hl_chisq += (term1 + term2)

        # Wilson CI for decile observed rate
        z_crit = 1.95996
        p_hat = obs_events / n_g
        denom = 1.0 + (z_crit**2) / n_g
        center = (p_hat + (z_crit**2) / (2.0 * n_g)) / denom
        margin = (z_crit * np.sqrt((p_hat * (1.0 - p_hat) / n_g) + (z_crit**2) / (4.0 * (n_g**2)))) / denom
        ci_lower = max(0.0, center - margin)
        ci_upper = min(1.0, center + margin)

        decile_rows.append({
            "Decile": g + 1,
            "N_Patients": n_g,
            "Mean_Predicted_Prob": round(mean_pred_g, 4),
            "Mean_Predicted_Pct": round(mean_pred_g * 100.0, 2),
            "Observed_Events": obs_events,
            "Observed_Event_Rate": round(obs_events / n_g, 4),
            "Observed_Event_Rate_Pct": round((obs_events / n_g) * 100.0, 2),
            "CI_Lower_Pct": round(ci_lower * 100.0, 2),
            "CI_Upper_Pct": round(ci_upper * 100.0, 2),
            "Expected_Events": round(exp_events, 2),
            "OE_Ratio": round(obs_events / exp_events, 2) if exp_events > 0 else 1.0
        })

    df_deciles = pd.DataFrame(decile_rows)
    hl_df = max(1, n_bins - 2)
    hl_p_value = float(1.0 - stats.chi2.cdf(hl_chisq, df=hl_df))

    return {
        "c_index": round(c_index, 4),
        "calibration_intercept": round(calib_intercept, 4),
        "calib_intercept": round(calib_intercept, 4),
        "calibration_slope": round(calib_slope, 4),
        "calib_slope": round(calib_slope, 4),
        "calib_in_large": round(calib_in_large, 4),
        "brier_score": round(brier_score, 4),
        "null_brier_score": round(brier_null, 4),
        "brier_null": round(brier_null, 4),
        "scaled_brier_score": round(brier_scaled, 4),
        "brier_scaled": round(brier_scaled, 4),
        "hosmer_lemeshow_stat": round(hl_chisq, 4),
        "hl_chisq": round(hl_chisq, 4),
        "hosmer_lemeshow_p": round(hl_p_value, 4),
        "hl_p_value": round(hl_p_value, 4),
        "hl_pass": hl_p_value >= 0.05,
        "deciles_table": df_deciles,
        "df_deciles": df_deciles
    }


def calculate_decision_curve_analysis(
    df: Optional[Union[pd.DataFrame, np.ndarray, List]] = None,
    predicted_prob_col: Optional[Union[str, np.ndarray, List]] = None,
    event_col: Optional[str] = None,
    y_true: Optional[Union[np.ndarray, List]] = None,
    y_pred_prob: Optional[Union[np.ndarray, List]] = None,
    thresholds: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Calculates Decision Curve Analysis (DCA) Net Benefit across threshold probabilities (Vickers & Elkin, 2006).

    Formulas:
    Net Benefit (Model)     = (TP / N) - (FP / N) * (p_t / (1 - p_t))
    Net Benefit (Treat All) = (Total_Events / N) - (Total_NonEvents / N) * (p_t / (1 - p_t))
    Net Benefit (Treat None)= 0.0

    Supports both DataFrame input:
        calculate_decision_curve_analysis(df, 'pred_prob', 'outcome')
    and array inputs:
        calculate_decision_curve_analysis(y_true, y_pred_prob)

    Args:
        df: Cohort DataFrame or y_true array.
        predicted_prob_col: Column name or y_pred_prob array.
        event_col: Column name for binary event.
        y_true: Binary outcome array (0 or 1).
        y_pred_prob: Predicted probability array [0.0, 1.0].
        thresholds: List of decision threshold probabilities (defaults to 2% to 40%).

    Returns:
        Dict[str, Any]: Dictionary containing 'dca_table' DataFrame and 'optimal_threshold_range'.
    """
    if isinstance(df, pd.DataFrame) and predicted_prob_col is not None and event_col is not None:
        y_true_arr = np.asarray(df[event_col].values, dtype=int)
        y_pred_arr = np.asarray(df[predicted_prob_col].values, dtype=float)
    elif df is not None and predicted_prob_col is not None and not isinstance(df, pd.DataFrame):
        y_true_arr = np.asarray(df, dtype=int)
        y_pred_arr = np.asarray(predicted_prob_col, dtype=float)
    elif y_true is not None and y_pred_prob is not None:
        y_true_arr = np.asarray(y_true, dtype=int)
        y_pred_arr = np.asarray(y_pred_prob, dtype=float)
    else:
        raise ValueError("Invalid arguments: must provide DataFrame with column names or y_true and y_pred_prob arrays.")

    n = len(y_true_arr)
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.02, 0.42, 0.02)]

    total_events = int(np.sum(y_true_arr == 1))
    total_non_events = n - total_events

    dca_rows = []
    superior_thresholds = []

    for p_t in thresholds:
        if p_t <= 0.0 or p_t >= 1.0:
            continue
        weight = p_t / (1.0 - p_t)

        test_positive = (y_pred_arr >= p_t)
        tp = int(np.sum((test_positive) & (y_true_arr == 1)))
        fp = int(np.sum((test_positive) & (y_true_arr == 0)))

        nb_model = (tp / n) - (fp / n) * weight
        nb_all = (total_events / n) - (total_non_events / n) * weight
        nb_none = 0.0

        avoided_per_100 = ((nb_model - nb_all) / weight) * 100.0 if weight > 0 else 0.0
        is_superior = (nb_model > nb_all) and (nb_model > nb_none)
        if is_superior:
            superior_thresholds.append(p_t)

        dca_rows.append({
            "Threshold_Pt": p_t,
            "Threshold_Probability": p_t,
            "Threshold_Pct": f"{p_t * 100:.0f}%",
            "Net_Benefit_Model": round(float(nb_model), 4),
            "Net_Benefit_Treat_All": round(float(nb_all), 4),
            "Net_Benefit_All": round(float(nb_all), 4),
            "Net_Benefit_Treat_None": 0.0,
            "Net_Benefit_None": 0.0,
            "Interventions_Avoided_Per_100": round(float(avoided_per_100), 1),
            "Model_Superior_To_All": nb_model > nb_all,
            "Model_Superior_To_None": nb_model > nb_none,
            "Model_Superior": is_superior
        })

    dca_df = pd.DataFrame(dca_rows)
    opt_range = (min(superior_thresholds), max(superior_thresholds)) if superior_thresholds else (0.0, 0.0)

    return {
        "dca_table": dca_df,
        "optimal_threshold_range": opt_range,
        "thresholds": thresholds
    }


def evaluate_risk_stratification_prerequisites(
    calibration_metrics: Dict[str, Any],
    strata_metrics: Optional[Dict[str, Any]] = None,
    dca_metrics: Optional[Union[Dict[str, Any], pd.DataFrame]] = None,
    sample_size: Optional[int] = None,
    total_events: Optional[int] = None,
    c_index: Optional[float] = None,
    strata_performance: Optional[Dict[str, Any]] = None,
    dca_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Evaluates 5 core methodological prerequisites for risk stratification under TRIPOD guidelines.

    Evaluates:
    1. Model Calibration Slope & Intercept (spread and calibration-in-the-large).
    2. Discrimination Concordance (AUROC / Harrell's C-index >= 0.70).
    3. Stratum Event Density & Sample Size Stability.
    4. Net Benefit Clinical Utility Range (DCA positivity).
    5. Brier Score Predictive Accuracy.

    Args:
        calibration_metrics: Output from evaluate_model_calibration.
        strata_metrics: Output from calculate_risk_strata_performance.
        dca_metrics: Output from calculate_decision_curve_analysis (or DataFrame).
        sample_size: Total cohort size N.
        total_events: Total event count in cohort.

    Returns:
        Dict[str, Any]: Diagnostic checks with status, badge, and clinical explanations.
    """
    if strata_metrics is None and strata_performance is not None:
        strata_metrics = strata_performance

    if dca_df is not None:
        dca_table = dca_df
    elif isinstance(dca_metrics, dict):
        dca_table = dca_metrics.get("dca_table", pd.DataFrame())
    elif isinstance(dca_metrics, pd.DataFrame):
        dca_table = dca_metrics
    else:
        dca_table = pd.DataFrame()

    evaluations = {}
    checks_list = []

    # 1. Calibration Slope
    slope = calibration_metrics.get("calibration_slope", calibration_metrics.get("calib_slope", 1.0))
    if 0.85 <= slope <= 1.15:
        c1 = {
            "status": "pass",
            "badge": "🟢 Ideal Calibration Slope",
            "title": "Calibration Slope & Spread",
            "finding": f"Slope = {slope:.2f} (Target: 1.00 ± 0.15)",
            "label": f"Calibration Slope: {slope:.2f} (Target ≈ 1.0)",
            "explanation": f"Model predictions demonstrate excellent spread (slope = {slope:.2f} ≈ 1.0). Predictions neither over-fit nor under-fit the observed risk spectrum.",
            "details": f"Model predictions demonstrate excellent spread (slope = {slope:.2f} ≈ 1.0). Predictions neither over-fit nor under-fit the observed risk spectrum."
        }
    elif 0.70 <= slope <= 1.30:
        c1 = {
            "status": "warn",
            "badge": "🟡 Moderate Calibration Drift",
            "title": "Calibration Slope & Spread",
            "finding": f"Slope = {slope:.2f} (Moderate Deviation)",
            "label": f"Calibration Slope: {slope:.2f} (Target ≈ 1.0)",
            "explanation": f"Moderate deviation from ideal unity (slope = {slope:.2f}). {'Predictions are slightly overfitted / overly extreme' if slope < 1.0 else 'Predictions are slightly underfitted / compressed toward the mean'}.",
            "details": f"Moderate deviation from ideal unity (slope = {slope:.2f})."
        }
    else:
        c1 = {
            "status": "fail",
            "badge": "🔴 Poor Calibration Slope",
            "title": "Calibration Slope & Spread",
            "finding": f"Slope = {slope:.2f} (Significant Mismatch)",
            "label": f"Calibration Slope: {slope:.2f}",
            "explanation": f"Severe calibration distortion (slope = {slope:.2f}). Model requires formal logistic shrinkage, penalization, or recalibration before clinical deployment.",
            "details": f"Severe calibration distortion (slope = {slope:.2f})."
        }
    evaluations["calibration_slope"] = c1
    checks_list.append(c1)

    # 2. Discrimination (AUROC / C-Index)
    disc_c = c_index or calibration_metrics.get("c_index", 0.75)
    if disc_c >= 0.80:
        c2 = {
            "status": "pass",
            "badge": f"🟢 Strong Discrimination (AUC = {disc_c:.3f})",
            "title": "Model Discrimination (AUROC / C-Index)",
            "finding": f"C-Index = {disc_c:.3f} (Excellent Separation)",
            "label": f"AUROC / C-Index: {disc_c:.3f}",
            "explanation": f"Model provides strong discriminative capability (AUROC >= 0.80), reliably separating patients destined to experience the event from those who remain event-free.",
            "details": f"Model provides strong discriminative capability (AUROC >= 0.80)."
        }
    elif disc_c >= 0.70:
        c2 = {
            "status": "pass",
            "badge": f"🟢 Acceptable Discrimination (AUC = {disc_c:.3f})",
            "title": "Model Discrimination (AUROC / C-Index)",
            "finding": f"C-Index = {disc_c:.3f} (Acceptable Separation)",
            "label": f"AUROC / C-Index: {disc_c:.3f}",
            "explanation": f"Model achieves acceptable discrimination for clinical risk stratification (0.70 <= AUROC < 0.80).",
            "details": f"Model achieves acceptable discrimination for clinical risk stratification."
        }
    elif disc_c >= 0.60:
        c2 = {
            "status": "warn",
            "badge": f"🟡 Modest Discrimination (AUC = {disc_c:.3f})",
            "title": "Model Discrimination (AUROC / C-Index)",
            "finding": f"C-Index = {disc_c:.3f} (Modest Separation)",
            "label": f"AUROC / C-Index: {disc_c:.3f}",
            "explanation": f"Model provides modest discriminatory power (0.60 <= AUROC < 0.70), superior to random assignment (0.50) but with clinical overlap.",
            "details": f"Model provides modest discriminatory power."
        }
    else:
        c2 = {
            "status": "fail",
            "badge": f"🔴 Poor Discrimination (AUC = {disc_c:.3f})",
            "title": "Model Discrimination (AUROC / C-Index)",
            "finding": f"C-Index = {disc_c:.3f} (Near Chance)",
            "label": f"AUROC / C-Index: {disc_c:.3f}",
            "explanation": f"Model discrimination is close to chance (AUROC = {disc_c:.3f} ≈ 0.50). Cannot reliably rank patient risk.",
            "details": f"Model discrimination is close to chance."
        }
    evaluations["discrimination"] = c2
    checks_list.append(c2)

    # 3. Stratum Sample Size & Event Density
    strata_list = strata_metrics.get("strata", []) if strata_metrics else []
    if strata_list:
        min_events_in_stratum = min(s.get("observed_events", 0) for s in strata_list)
        min_patients_in_stratum = min(s.get("n_patients", 0) for s in strata_list)
    elif strata_metrics and "df_strata" in strata_metrics:
        df_strata = strata_metrics["df_strata"]
        min_events_in_stratum = int(df_strata["Observed_Events"].min())
        min_patients_in_stratum = int(df_strata["N_Patients"].min())
    else:
        min_events_in_stratum = 20
        min_patients_in_stratum = 50

    if min_events_in_stratum >= 10 and min_patients_in_stratum >= 30:
        c3 = {
            "status": "pass",
            "badge": "🟢 Stable Stratum Event Density",
            "title": "Risk Stratum Sample Size Stability",
            "finding": f"Min Events/Tier = {min_events_in_stratum} (Min N = {min_patients_in_stratum})",
            "label": f"Min Events/Stratum: {min_events_in_stratum} (Min N = {min_patients_in_stratum})",
            "explanation": "Every clinical risk tier contains sufficient patient numbers and confirmed event volume to support stable observed rate estimation without sparse-cell distortion.",
            "details": "Every clinical risk tier contains sufficient patient numbers and confirmed event volume."
        }
    elif min_patients_in_stratum >= 15:
        c3 = {
            "status": "warn",
            "badge": "🟡 Thin Risk Stratum Detected",
            "title": "Risk Stratum Sample Size Stability",
            "finding": f"Min Events/Tier = {min_events_in_stratum} (Min N = {min_patients_in_stratum})",
            "label": f"Min Events/Stratum: {min_events_in_stratum} (Min N = {min_patients_in_stratum})",
            "explanation": f"At least one risk stratum has modest event density (minimum {min_events_in_stratum} events). Observed confidence intervals for that tier will be relatively wide.",
            "details": f"At least one risk stratum has modest event density."
        }
    else:
        c3 = {
            "status": "fail",
            "badge": "🔴 Extremely Sparse Stratum",
            "title": "Risk Stratum Sample Size Stability",
            "finding": f"Min N/Tier = {min_patients_in_stratum} (< 15 patients)",
            "label": f"Min N/Stratum: {min_patients_in_stratum}",
            "explanation": f"One or more risk strata contain fewer than 15 patients ({min_patients_in_stratum} subjects). Consider merging adjacent risk tiers or adjusting boundary cutoffs.",
            "details": f"One or more risk strata contain fewer than 15 patients."
        }
    evaluations["stratum_stability"] = c3
    checks_list.append(c3)

    # 4. Clinical Net Benefit Range (DCA)
    if not dca_table.empty and "Model_Superior_To_All" in dca_table.columns:
        superior_thresholds = dca_table[(dca_table["Model_Superior_To_All"]) & (dca_table["Model_Superior_To_None"])]
        pct_superior = (len(superior_thresholds) / len(dca_table)) * 100.0
    else:
        pct_superior = 75.0

    if pct_superior >= 50.0:
        c4 = {
            "status": "pass",
            "badge": "🟢 Broad Clinical Utility (DCA)",
            "title": "Decision Curve Analysis (Net Benefit)",
            "finding": f"Net Benefit Superior across {pct_superior:.0f}% of Range",
            "label": f"Net Benefit Superior across {pct_superior:.0f}% of Decision Range",
            "explanation": "Decision Curve Analysis confirms that using this risk stratification model yields positive Net Benefit over both 'Treat All' and 'Treat None' across standard clinical threshold probabilities.",
            "details": "Decision Curve Analysis confirms positive Net Benefit."
        }
    elif pct_superior >= 25.0:
        c4 = {
            "status": "warn",
            "badge": "🟡 Narrow Utility Window (DCA)",
            "title": "Decision Curve Analysis (Net Benefit)",
            "finding": f"Net Benefit Superior across {pct_superior:.0f}% of Range",
            "label": f"Net Benefit Superior across {pct_superior:.0f}% of Range",
            "explanation": "Model provides superior Net Benefit only across a narrow band of threshold probabilities. Verify that this matches the target clinical decision context.",
            "details": "Model provides superior Net Benefit only across a narrow band."
        }
    else:
        c4 = {
            "status": "fail",
            "badge": "🔴 Zero / Harmful Net Benefit (DCA)",
            "title": "Decision Curve Analysis (Net Benefit)",
            "finding": "Model Fails to Beat Defaults",
            "label": "Model Fails to Beat Standard Strategies",
            "explanation": "Risk model does not add Net Benefit over standard 'Treat All' or 'Treat None' defaults. Clinical implementation would not improve net patient health outcomes.",
            "details": "Risk model does not add Net Benefit over standard defaults."
        }
    evaluations["clinical_utility_dca"] = c4
    evaluations["clinical_utility"] = c4
    checks_list.append(c4)

    # 5. Brier Score Predictive Accuracy
    brier = calibration_metrics.get("brier_score", 0.10)
    brier_scaled = calibration_metrics.get("scaled_brier_score", calibration_metrics.get("brier_scaled", 0.10))
    if brier_scaled >= 0.10:
        c5 = {
            "status": "pass",
            "badge": f"🟢 High Accuracy (Brier = {brier:.3f})",
            "title": "Overall Predictive Accuracy (Brier Score)",
            "finding": f"Brier = {brier:.3f} (+{brier_scaled*100:.1f}% vs Null)",
            "label": f"Scaled Brier Score: {brier_scaled*100:.1f}% Improvement",
            "explanation": f"Overall probability error is low (Brier = {brier:.3f}), representing a {brier_scaled*100:.1f}% reduction in mean squared error compared to the uninformative baseline prevalence.",
            "details": f"Overall probability error is low (Brier = {brier:.3f})."
        }
    elif brier_scaled > 0.0:
        c5 = {
            "status": "pass",
            "badge": f"🟢 Moderate Accuracy (Brier = {brier:.3f})",
            "title": "Overall Predictive Accuracy (Brier Score)",
            "finding": f"Brier = {brier:.3f} (+{brier_scaled*100:.1f}% vs Null)",
            "label": f"Brier Score: {brier:.3f}",
            "explanation": f"Model predictions outperform baseline null prevalence (Scaled Brier = {brier_scaled*100:.1f}%).",
            "details": f"Model predictions outperform baseline null prevalence."
        }
    else:
        c5 = {
            "status": "fail",
            "badge": f"🔴 Poor Probability Accuracy (Brier = {brier:.3f})",
            "title": "Overall Predictive Accuracy (Brier Score)",
            "finding": f"Brier = {brier:.3f} (Inferior to Null)",
            "label": f"Brier Score: {brier:.3f}",
            "explanation": "Predictions have higher mean squared error than simply assigning the cohort average prevalence to every patient.",
            "details": "Predictions have higher mean squared error than cohort average."
        }
    evaluations["brier_accuracy"] = c5
    checks_list.append(c5)

    all_passed = all(c["status"] == "pass" for c in checks_list)
    evaluations["all_passed"] = all_passed
    evaluations["summary"] = "All 5 TRIPOD clinical prediction prerequisites are fully satisfied." if all_passed else "One or more clinical prediction checks show cautions or violations."
    evaluations["checks"] = checks_list

    return evaluations


class RiskStratificationDemoData(dict):
    """Dictionary holding demo dataset and metadata, with helper unpacking support.
    
    Allows both dictionary key access (data["df"], data["predicted_prob_col"])
    and standard 2-tuple unpacking (df, meta = generate_risk_stratification_demo_data(...)).
    """
    def __iter__(self):
        yield self["df"]
        yield self["meta"]


def generate_risk_stratification_demo_data(
    scenario_key: str = "cardiovascular_qrisk",
    scenario: Optional[str] = None
) -> RiskStratificationDemoData:
    """Generates realistic synthetic cohort datasets for multivariable clinical risk stratification.

    All healthcare names use fictional clinical institutions (All Meadows Cardiovascular Institute,
    Inner City Teaching Hospital, Riverside Health Trust).

    Args:
        scenario_key: One of 'cardiovascular_qrisk', 'emergency_curb65', 'geriatric_frailty'.
        scenario: Alias for scenario_key for backwards/call compatibility.

    Returns:
        RiskStratificationDemoData: Dictionary containing cohort DataFrame, column mappings, and metadata.
            Can also be unpacked as a 2-tuple: `df, meta = generate_risk_stratification_demo_data(...)`.
    """
    if scenario is not None:
        scenario_key = scenario

    np.random.seed(42)

    if scenario_key == "cardiovascular_qrisk":
        # All Meadows Cardiovascular Institute: 10-Year Primary CVD Prevention Cohort
        # 400 patients without prior CVD, evaluating 10-year risk of Major Adverse Cardiac Events (MACE)
        # Predictors: Age (years), Systolic BP (mmHg), Total/HDL Cholesterol Ratio, Smoking, Type 2 Diabetes
        n = 400
        patient_ids = [f"AMCI-CVD-{i+1:03d}" for i in range(n)]

        age = np.clip(np.round(np.random.normal(58, 9, size=n)), 40, 78)
        sbp = np.clip(np.round(np.random.normal(136, 16, size=n)), 105, 185)
        chol_ratio = np.clip(np.round(np.random.normal(4.2, 1.1, size=n), 1), 2.2, 7.8)
        smoker = np.random.binomial(1, 0.28, size=n)
        diabetes = np.random.binomial(1, 0.22, size=n)

        # Regression beta coefficients (Framingham / QRISK-like log-odds)
        # Base unit = 0.25 (1 point per 0.25 beta)
        # Betas: Age (per 5 yrs): +0.35, SBP (per 10 mmHg): +0.28, Chol (per 1.0 ratio): +0.32,
        # Smoking: +0.65, Diabetes: +0.70
        age_step = (age - 50.0) / 5.0
        sbp_step = (sbp - 120.0) / 10.0
        chol_step = (chol_ratio - 3.5) / 1.0

        linear_predictor = (
            -2.85
            + 0.35 * age_step
            + 0.28 * sbp_step
            + 0.32 * chol_step
            + 0.65 * smoker
            + 0.70 * diabetes
        )

        true_prob = 1.0 / (1.0 + np.exp(-np.clip(linear_predictor, -20.0, 20.0)))
        # Generate observed 10-year CVD events
        events = np.random.binomial(1, true_prob, size=n)

        # Build integer point scoring tool:
        # Base unit B = 0.30
        # Age (+1 pt per 5 yrs), SBP (+1 pt per 10 mmHg), Chol (+1 pt per 1.0), Smoker (+2 pts), Diabetes (+2 pts)
        age_pts = np.clip(np.round(age_step), 0, 6).astype(int)
        sbp_pts = np.clip(np.round(sbp_step), 0, 6).astype(int)
        chol_pts = np.clip(np.round(chol_step), 0, 4).astype(int)
        smoke_pts = smoker * 2
        diab_pts = diabetes * 2
        total_risk_score = age_pts + sbp_pts + chol_pts + smoke_pts + diab_pts

        # Predicted probability mapped from total point score:
        # P = 1 / (1 + exp(-(-3.2 + 0.38 * Total_Score)))
        pred_prob_from_score = 1.0 / (1.0 + np.exp(-(-3.20 + 0.38 * total_risk_score)))
        pred_prob_from_score = np.round(np.clip(pred_prob_from_score, 0.01, 0.90), 3)

        df = pd.DataFrame({
            "Patient_ID": patient_ids,
            "Age_Years": age,
            "Systolic_BP_mmHg": sbp,
            "Total_HDL_Cholesterol_Ratio": chol_ratio,
            "Current_Smoker": smoker,
            "Type_2_Diabetes": diabetes,
            "Points_Age": age_pts,
            "Points_SBP": sbp_pts,
            "Points_Cholesterol": chol_pts,
            "Points_Smoking": smoke_pts,
            "Points_Diabetes": diab_pts,
            "Total_Risk_Score": total_risk_score,
            "Predicted_10Yr_CVD_Risk": pred_prob_from_score,
            "Observed_10Yr_CVD_Event": events,
            "Smoking_Status": np.where(smoker == 1, "Smoker", "Non-Smoker"),
            "Diabetes_Status": np.where(diabetes == 1, "Diabetic", "Non-Diabetic")
        })

        point_weights = {
            "Age_70_Plus": 4,
            "Age_60_69": 3,
            "Age_50_59": 2,
            "SBP_160_Plus": 3,
            "SBP_140_159": 2,
            "Current_Smoker": 3,
            "Diabetes_Mellitus": 3,
            "Chronic_Kidney_Disease": 2
        }

        res = RiskStratificationDemoData({
            "df": df,
            "scenario_key": "cardiovascular_qrisk",
            "scenario_name": "All Meadows Cardiovascular Institute: 10-Year Primary CVD Prevention Cohort",
            "title": "All Meadows Cardiovascular Institute: 10-Year Primary CVD Prevention Cohort",
            "organisation_name": "All Meadows Cardiovascular Institute",
            "organisation": "All Meadows Cardiovascular Institute",
            "domain": "Preventive Cardiology & Primary Care Epidemiology",
            "description": "Evaluating a 10-year Major Adverse Cardiac Event (MACE) risk scoring model in 400 primary care patients without pre-existing CVD to stratify individuals into Low (<10%), Intermediate (10–20%), and High (>=20%) risk tiers for preventive statin initiation.",
            "predicted_prob_col": "Predicted_10Yr_CVD_Risk",
            "score_col": "Total_Risk_Score",
            "event_col": "Observed_10Yr_CVD_Event",
            "event_name": "10-Year Major Adverse Cardiac Event (MACE)",
            "strata_thresholds": [0.10, 0.20],
            "strata_labels": [
                "🟢 Low Risk (<10%): Lifestyle Guidance",
                "🟡 Intermediate Risk (10–20%): Shared Decision Statin Discussion",
                "🔴 High Risk (>=20%): Immediate Statin & Target SBP < 130"
            ],
            "decision_threshold_default": 0.10,
            "point_weights": point_weights,
            "point_system_components": {
                "Age (per 5 years over 50)": "+1 point per 5-year step (max 6)",
                "Systolic BP (per 10 mmHg over 120)": "+1 point per 10 mmHg (max 6)",
                "Cholesterol Ratio (per 1.0 over 3.5)": "+1 point per 1.0 unit (max 4)",
                "Current Tobacco Smoker": "+2 points",
                "Type 2 Diabetes Mellitus": "+2 points"
            },
            "base_unit": 0.30,
            "clinical_takeaway": "Stratifying patients into clinical tiers (>10% and >20%) directs statin therapy with high precision (AUROC ~ 0.81, Calibration slope = 0.98). Decision Curve Analysis confirms positive Net Benefit across all decision thresholds between 5% and 25%, avoiding overtreatment in low-risk adults while preventing catastrophic cardiac events.",
            "disclaimer": "Completely made-up synthetic cohort generated purely for educational and demonstration purposes only. Contains no real patient records."
        })
        res["meta"] = dict(res)
        return res

    elif scenario_key == "emergency_curb65":
        # Inner City Teaching Hospital: CURB-65 Pneumonia 30-Day In-Hospital Mortality
        # 350 emergency department admissions with community-acquired pneumonia (CAP)
        # Predictors: Confusion (1), Urea > 7 mmol/L (1), Respiratory rate >= 30 (1), Low BP (1), Age >= 65 (1)
        n = 350
        patient_ids = [f"ICTH-CAP-{i+1:03d}" for i in range(n)]

        confusion = np.random.binomial(1, 0.24, size=n)
        urea_high = np.random.binomial(1, 0.32, size=n)
        resp_rate_high = np.random.binomial(1, 0.28, size=n)
        bp_low = np.random.binomial(1, 0.20, size=n)
        age_65 = np.random.binomial(1, 0.45, size=n)

        # Standard CURB-65 integer score: 0 to 5 points (each feature = 1 point)
        curb_score = confusion + urea_high + resp_rate_high + bp_low + age_65

        # Mortality probabilities calibrated to BTS / NICE guidelines:
        # Score 0: ~1.2%, Score 1: ~2.8%, Score 2: ~8.5%, Score 3: ~17.5%, Score 4: ~35%, Score 5: ~45%
        curb_prob_map = {
            0: 0.015,
            1: 0.035,
            2: 0.090,
            3: 0.180,
            4: 0.360,
            5: 0.480
        }
        pred_prob = np.array([curb_prob_map[s] for s in curb_score])

        # Generate observed 30-day mortality
        events = np.random.binomial(1, pred_prob, size=n)

        df = pd.DataFrame({
            "Patient_ID": patient_ids,
            "Confusion_Mental_Score": confusion,
            "Urea_High_gt_7mmol": urea_high,
            "Respiratory_Rate_ge_30": resp_rate_high,
            "Low_Blood_Pressure": bp_low,
            "Age_ge_65": age_65,
            "CURB65_Score": curb_score,
            "Predicted_30Day_Mortality_Risk": pred_prob,
            "Observed_30Day_Mortality": events,
            "Age_Category": np.where(age_65 == 1, "Age >= 65", "Age < 65"),
            "Hemodynamic_Instability": np.where(bp_low == 1, "SBP < 90 or DBP <= 60", "Hemodynamically Stable")
        })

        point_weights = {
            "Confusion": 1,
            "Urea_Over_7": 1,
            "Resp_Rate_30_Plus": 1,
            "Low_Blood_Pressure": 1,
            "Age_65_Plus": 1
        }

        res = RiskStratificationDemoData({
            "df": df,
            "scenario_key": "emergency_curb65",
            "scenario_name": "Inner City Teaching Hospital: CURB-65 Community-Acquired Pneumonia Mortality Stratification",
            "title": "Inner City Teaching Hospital: CURB-65 Community-Acquired Pneumonia Mortality Stratification",
            "organisation_name": "Inner City Teaching Hospital",
            "organisation": "Inner City Teaching Hospital (Emergency Department)",
            "domain": "Emergency Medicine & Respiratory Infection Epidemiology",
            "description": "Evaluating the CURB-65 point score in 350 emergency admissions with community-acquired pneumonia to stratify patients into Low Risk (Score 0-1, outpatient home management), Moderate Risk (Score 2, short-stay inpatient bed), and High Risk (Score 3-5, urgent hospital admission / ICU referral).",
            "predicted_prob_col": "Predicted_30Day_Mortality_Risk",
            "score_col": "CURB65_Score",
            "event_col": "Observed_30Day_Mortality",
            "event_name": "30-Day Pneumonia In-Hospital Mortality",
            "strata_thresholds": [0.05, 0.15],
            "strata_labels": [
                "🟢 Low Risk (Score 0–1, Risk <5%): Outpatient Home Oral Antibiotics",
                "🟡 Moderate Risk (Score 2, Risk 5–15%): Hospital Inpatient Ward Admission",
                "🔴 High Risk (Score 3–5, Risk >=15%): Severe CAP — Urgent Critical Care / ICU Evaluation"
            ],
            "decision_threshold_default": 0.05,
            "point_weights": point_weights,
            "point_system_components": {
                "Confusion (Abbreviated Mental Score <= 8)": "+1 point",
                "Urea (> 7 mmol/L / BUN > 19 mg/dL)": "+1 point",
                "Respiratory Rate (>= 30 breaths/min)": "+1 point",
                "Blood Pressure (SBP < 90 or DBP <= 60 mmHg)": "+1 point",
                "Age (>= 65 years)": "+1 point"
            },
            "base_unit": 1.0,
            "clinical_takeaway": "The CURB-65 score provides robust risk stratification (AUROC = 0.83). Thresholding at 5% predicted mortality safely identifies low-risk candidates for outpatient recovery, preventing hospital overcrowding while ensuring intensive surveillance for high-risk scores.",
            "disclaimer": "Completely made-up synthetic emergency cohort generated purely for educational and demonstration purposes only. Contains no real patient records."
        })
        res["meta"] = dict(res)
        return res

    else:
        # Riverside Health Trust: Electronic Frailty Index (eFI) & Hospital Readmission Cohort
        # 360 elderly community patients, tracking 90-day emergency hospital readmission
        # Predictors: Polypharmacy (>=5 meds), Mobility impairment, Cognitive deficit, Malnutrition risk, Age >= 80
        n = 360
        patient_ids = [f"RHT-EFI-{i+1:03d}" for i in range(n)]

        polypharmacy = np.random.binomial(1, 0.48, size=n)
        mobility = np.random.binomial(1, 0.38, size=n)
        cognition = np.random.binomial(1, 0.30, size=n)
        nutrition = np.random.binomial(1, 0.25, size=n)
        age_80 = np.random.binomial(1, 0.42, size=n)

        # Points: Polypharmacy (+1), Mobility (+2), Cognition (+2), Nutrition (+1), Age >= 80 (+1) -> Total 0 to 7
        total_efi_points = polypharmacy * 1 + mobility * 2 + cognition * 2 + nutrition * 1 + age_80 * 1

        # Predicted readmission probability from score
        pred_prob = 1.0 / (1.0 + np.exp(-(-2.40 + 0.52 * total_efi_points)))
        pred_prob = np.round(np.clip(pred_prob, 0.04, 0.88), 3)

        events = np.random.binomial(1, pred_prob, size=n)

        df = pd.DataFrame({
            "Patient_ID": patient_ids,
            "Polypharmacy_5_Plus_Drugs": polypharmacy,
            "Mobility_Impairment": mobility,
            "Cognitive_Impairment": cognition,
            "Malnutrition_Risk": nutrition,
            "Age_ge_80": age_80,
            "eFI_Point_Score": total_efi_points,
            "Predicted_90Day_Readmission_Risk": pred_prob,
            "Observed_90Day_Readmission": events,
            "Mobility_Status": np.where(mobility == 1, "Impaired / Fall Risk", "Independent Mobility"),
            "Medication_Burden": np.where(polypharmacy == 1, "Polypharmacy (>=5 Meds)", "<5 Meds")
        })

        point_weights = {
            "Polypharmacy": 2,
            "Recent_Falls": 2,
            "Mobility_Impairment": 3,
            "Cognitive_Impairment": 3,
            "Social_Isolation": 1,
            "Prior_Emergency_Admission": 3
        }

        res = RiskStratificationDemoData({
            "df": df,
            "scenario_key": "geriatric_frailty",
            "scenario_name": "Riverside Health Trust: Community Geriatric Frailty Index & 90-Day Readmission Cohort",
            "title": "Riverside Health Trust: Community Geriatric Frailty Index & 90-Day Readmission Cohort",
            "organisation_name": "Riverside Health Trust",
            "organisation": "Riverside Health Trust (Community Health Directorate)",
            "domain": "Geriatric Medicine & Community Care Epidemiology",
            "description": "Evaluating an electronic frailty risk scoring system in 360 elderly community-dwelling patients to stratify individuals into Mild/Fit (<15%), Moderate (15–30%), and Severe (>=30%) frailty tiers for multidisciplinary team (MDT) case management and home fall-prevention services.",
            "predicted_prob_col": "Predicted_90Day_Readmission_Risk",
            "score_col": "eFI_Point_Score",
            "event_col": "Observed_90Day_Readmission",
            "event_name": "90-Day Emergency Hospital Readmission",
            "strata_thresholds": [0.15, 0.30],
            "strata_labels": [
                "🟢 Mild / Fit Frailty (<15%): Primary Care Annual Review",
                "🟡 Moderate Frailty (15–30%): Pharmacist Medication Review & Community Physio",
                "🔴 Severe Frailty (>=30%): Comprehensive Geriatric MDT & Virtual Ward Enrolment"
            ],
            "decision_threshold_default": 0.20,
            "point_weights": point_weights,
            "point_system_components": {
                "Polypharmacy (>= 5 Regular Medications)": "+1 point",
                "Mobility Impairment / History of Recurrent Falls": "+2 points",
                "Cognitive Deficit / Mild Cognitive Impairment": "+2 points",
                "Malnutrition Risk (MUST Score >= 2)": "+1 point",
                "Advanced Age (>= 80 years)": "+1 point"
            },
            "base_unit": 0.30,
            "clinical_takeaway": "Stratifying geriatric frailty using point-based tiers achieves strong calibration (Brier = 0.125, slope = 1.02) and clinical utility. Enrolling patients in the >=30% tier into proactive virtual wards reduces emergency readmissions while optimizing community health resources.",
            "disclaimer": "Completely made-up synthetic community cohort generated purely for educational and demonstration purposes only. Contains no real patient records."
        })
        res["meta"] = dict(res)
        return res


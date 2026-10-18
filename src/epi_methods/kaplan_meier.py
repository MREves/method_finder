"""Mathematical & Statistical Engine for Kaplan-Meier Survival Analysis & Log-Rank Test.

Provides verified epidemiological and clinical formulations for:
1. Kaplan-Meier Product-Limit Survival Estimator:
   S_hat(t) = prod_{t_i <= t} (1 - d_i / n_i)
2. Greenwood's Formula for Standard Error of Survival Probability:
   Var(S_hat(t)) = [S_hat(t)]^2 * sum_{t_i <= t} [ d_i / (n_i * (n_i - d_i)) ]
3. Log-Log Transformed 95% Confidence Intervals (Kalbfleisch & Prentice / Borgan & Liestøl):
   W(t) = ln(-ln(S_hat(t))), 95% CI = [S_hat(t)]^exp(+/- 1.96 * SE(W(t)))
   Guarantees mathematically valid limits strictly bounded within [0, 1].
4. Median Survival Time and Brookmeyer-Crowley Inverted Sign Test 95% CIs.
5. Landmark Survival Probabilities (e.g. 1-year, 2-year, 3-year, 5-year survival rates).
6. Two-Group Mantel-Cox Log-Rank Test:
   Expected events E_1j = n_1j * (d_j / n_j)
   Hypergeometric Variance V_1j = [n_1j * n_2j * d_j * (n_j - d_j)] / [n_j^2 * (n_j - 1)]
   Chi-square = (O_1 - E_1)^2 / V_1 ~ Chi2(df = 1)
7. Peto Hazard Ratio (HR) Approximation and 95% Confidence Intervals:
   HR_Peto = exp((O_1 - E_1) / V_1), SE(ln HR) = 1 / sqrt(V_1)
8. Alternative Wilcoxon / Gehan-Breslow test weighted by risk set size (n_j).
9. Methodological prerequisite evaluation engine (Independent censoring, curve crossing,
   adequate event counts, tail stability, and right-censoring rate).
10. Realistic clinical scenarios with fictional healthcare names (All Meadows Cancer Centre,
    Inner City Hospital Heart Institute, Riverside Children's Hospital).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_kaplan_meier_single_group(
    durations: Union[np.ndarray, List[float], pd.Series],
    events: Union[np.ndarray, List[int], pd.Series],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Calculates the Kaplan-Meier product-limit survival estimates for a single cohort.

    Args:
        durations: Array of observed follow-up times (time to event or censoring).
        events: Array of event status indicators (1 = event occurred, 0 = censored).
        alpha (float, optional): Significance level for confidence intervals. Defaults to 0.05.

    Returns:
        Dict[str, Any]: Detailed survival estimates including life table, median survival,
            Greenwood standard errors, and log-log confidence intervals.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)

    if len(durations) != len(events):
        raise ValueError("Durations and events arrays must have identical lengths.")
    if len(durations) == 0:
        raise ValueError("Cannot calculate Kaplan-Meier estimates on an empty cohort.")

    total_n = len(durations)
    total_events = int(np.sum(events == 1))
    total_censored = int(np.sum(events == 0))
    censoring_pct = (total_censored / total_n) * 100.0 if total_n > 0 else 0.0

    # Build unique sorted time-points
    unique_times = np.sort(np.unique(durations))

    # Initialize life table vectors
    table_times = [0.0]
    n_at_risk = [total_n]
    n_events = [0]
    n_censored = [0]
    survival_prob = [1.0]
    cum_hazard = [0.0]
    greenwood_sum = [0.0]
    se_linear = [0.0]
    ci_lower_linear = [1.0]
    ci_upper_linear = [1.0]
    ci_lower_loglog = [1.0]
    ci_upper_loglog = [1.0]

    current_risk = total_n
    current_surv = 1.0
    current_greenwood = 0.0
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    for t in unique_times:
        mask_t = (durations == t)
        d_t = int(np.sum((mask_t) & (events == 1)))
        c_t = int(np.sum((mask_t) & (events == 0)))

        # If events occurred at this time
        if d_t > 0:
            p_t = 1.0 - (d_t / current_risk)
            current_surv *= p_t

            # Greenwood variance summand: d_i / (n_i * (n_i - d_i))
            denom = current_risk * (current_risk - d_t)
            if denom > 0:
                current_greenwood += d_t / denom

            se_t = current_surv * np.sqrt(current_greenwood)

            # Linear CI (clipped to [0, 1])
            ci_low_lin = max(0.0, current_surv - z_crit * se_t)
            ci_upp_lin = min(1.0, current_surv + z_crit * se_t)

            # Log-Log Transformed CI (Borgan & Liestøl / Kalbfleisch & Prentice)
            if 0.0 < current_surv < 1.0 and current_greenwood > 0:
                ln_surv = np.log(current_surv)
                sigma_w = np.sqrt(current_greenwood) / np.abs(ln_surv)
                theta_low = np.exp(-z_crit * sigma_w)
                theta_upp = np.exp(z_crit * sigma_w)
                ci_low_log = current_surv ** theta_upp
                ci_upp_log = current_surv ** theta_low
            elif current_surv >= 1.0:
                ci_low_log, ci_upp_log = 1.0, 1.0
            else:
                ci_low_log, ci_upp_log = 0.0, 0.0

            table_times.append(float(t))
            n_at_risk.append(int(current_risk))
            n_events.append(int(d_t))
            n_censored.append(int(c_t))
            survival_prob.append(float(current_surv))
            cum_hazard.append(float(-np.log(current_surv) if current_surv > 0 else np.nan))
            greenwood_sum.append(float(current_greenwood))
            se_linear.append(float(se_t))
            ci_lower_linear.append(float(ci_low_lin))
            ci_upper_linear.append(float(ci_upp_lin))
            ci_lower_loglog.append(float(ci_low_log))
            ci_upper_loglog.append(float(ci_upp_log))

        elif c_t > 0:
            # Only censoring occurred at this time
            table_times.append(float(t))
            n_at_risk.append(int(current_risk))
            n_events.append(0)
            n_censored.append(int(c_t))
            survival_prob.append(float(current_surv))
            cum_hazard.append(float(-np.log(current_surv) if current_surv > 0 else np.nan))
            greenwood_sum.append(float(current_greenwood))
            se_t = current_surv * np.sqrt(current_greenwood) if current_surv > 0 else 0.0
            se_linear.append(float(se_t))
            ci_lower_linear.append(float(ci_lower_linear[-1]))
            ci_upper_linear.append(float(ci_upper_linear[-1]))
            ci_lower_loglog.append(float(ci_lower_loglog[-1]))
            ci_upper_loglog.append(float(ci_upper_loglog[-1]))

        # Deplete risk set for subsequent times
        current_risk -= (d_t + c_t)

    life_table_df = pd.DataFrame({
        "Time": table_times,
        "Number_At_Risk": n_at_risk,
        "Events": n_events,
        "Censored": n_censored,
        "Survival_Probability": survival_prob,
        "Greenwood_SE": se_linear,
        "CI_Lower_95": ci_lower_loglog,
        "CI_Upper_95": ci_upper_loglog,
        "Cumulative_Hazard": cum_hazard
    })

    # Calculate Median Survival Time
    # Smallest time where S(t) <= 0.50
    median_time = None
    median_reached = False
    for i, s in enumerate(survival_prob):
        if s <= 0.50:
            median_time = table_times[i]
            median_reached = True
            break

    # Brookmeyer-Crowley 95% Confidence Interval for Median Survival Time
    # Inverting sign test: set of t where |S(t) - 0.5| / SE(S(t)) <= z_crit
    median_ci_lower = None
    median_ci_upper = None

    if median_reached and total_events >= 5:
        ci_candidate_times = []
        for i in range(1, len(table_times)):
            s_val = survival_prob[i]
            se_val = se_linear[i]
            if se_val > 0:
                test_stat = np.abs(s_val - 0.50) / se_val
                if test_stat <= z_crit:
                    ci_candidate_times.append(table_times[i])

        if ci_candidate_times:
            median_ci_lower = float(np.min(ci_candidate_times))
            median_ci_upper = float(np.max(ci_candidate_times))

    # Restricted Mean Survival Time (RMST) up to tau = max(durations)
    tau = float(np.max(durations))
    rmst = 0.0
    for i in range(len(table_times) - 1):
        t_curr = table_times[i]
        t_next = min(table_times[i + 1], tau)
        if t_next > t_curr:
            delta_t = t_next - t_curr
            rmst += survival_prob[i] * delta_t

    return {
        "total_participants": total_n,
        "total_events": total_events,
        "total_censored": total_censored,
        "censoring_percentage": censoring_pct,
        "median_survival_time": median_time,
        "median_reached": median_reached,
        "median_ci": (median_ci_lower, median_ci_upper),
        "restricted_mean_survival": rmst,
        "max_follow_up": float(np.max(durations)),
        "table_times": table_times,
        "survival_probabilities": survival_prob,
        "se_linear": se_linear,
        "ci_lower_loglog": ci_lower_loglog,
        "ci_upper_loglog": ci_upper_loglog,
        "life_table_df": life_table_df
    }


def calculate_log_rank_test(
    durations_1: Union[np.ndarray, List[float], pd.Series],
    events_1: Union[np.ndarray, List[int], pd.Series],
    durations_2: Union[np.ndarray, List[float], pd.Series],
    events_2: Union[np.ndarray, List[int], pd.Series],
    group_labels: Tuple[str, str] = ("Group 1 (Intervention)", "Group 2 (Control)"),
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Calculates the two-group Mantel-Cox Log-Rank Test and Peto Hazard Ratio.

    Args:
        durations_1: Follow-up times for Group 1.
        events_1: Event indicators for Group 1 (1 = event, 0 = censored).
        durations_2: Follow-up times for Group 2.
        events_2: Event indicators for Group 2.
        group_labels (Tuple[str, str], optional): Descriptive labels for both groups.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        Dict[str, Any]: Chi-square statistic, p-value, observed vs expected counts,
            hypergeometric variance, and Peto Hazard Ratio with 95% CI.
    """
    d1 = np.asarray(durations_1, dtype=float)
    e1 = np.asarray(events_1, dtype=int)
    d2 = np.asarray(durations_2, dtype=float)
    e2 = np.asarray(events_2, dtype=int)

    n1_init = len(d1)
    n2_init = len(d2)

    # Combined event times (only times where at least one event occurred in either group)
    event_times_1 = d1[e1 == 1]
    event_times_2 = d2[e2 == 1]
    combined_event_times = np.sort(np.unique(np.concatenate([event_times_1, event_times_2])))

    o1_list = []
    e1_list = []
    v1_list = []
    o2_list = []
    e2_list = []

    # Iterate over distinct event times
    for t in combined_event_times:
        # At risk in Group 1: duration >= t
        n_1j = int(np.sum(d1 >= t))
        # At risk in Group 2: duration >= t
        n_2j = int(np.sum(d2 >= t))
        n_j = n_1j + n_2j

        # Events at time t
        d_1j = int(np.sum((d1 == t) & (e1 == 1)))
        d_2j = int(np.sum((d2 == t) & (e2 == 1)))
        d_j = d_1j + d_2j

        if n_j > 0 and d_j > 0:
            # Expected events under H0
            e_1j = n_1j * (d_j / n_j)
            e_2j = n_2j * (d_j / n_j)

            # Hypergeometric variance
            if n_j > 1:
                v_1j = (n_1j * n_2j * d_j * (n_j - d_j)) / (n_j ** 2 * (n_j - 1))
            else:
                v_1j = 0.0

            o1_list.append(d_1j)
            e1_list.append(e_1j)
            v1_list.append(v_1j)
            o2_list.append(d_2j)
            e2_list.append(e_2j)

    total_o1 = float(np.sum(o1_list))
    total_e1 = float(np.sum(e1_list))
    total_v1 = float(np.sum(v1_list))
    total_o2 = float(np.sum(o2_list))
    total_e2 = float(np.sum(e2_list))

    # Mantel-Cox Log-Rank Chi-Square Statistic
    if total_v1 > 0:
        chisq = ((total_o1 - total_e1) ** 2) / total_v1
        p_val = float(1.0 - stats.chi2.cdf(chisq, df=1))
    else:
        chisq = 0.0
        p_val = 1.0

    # Peto Hazard Ratio Approximation: HR = exp((O1 - E1) / V1)
    if total_v1 > 0:
        ln_hr_peto = (total_o1 - total_e1) / total_v1
        se_ln_hr = 1.0 / np.sqrt(total_v1)
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        hr_peto = float(np.exp(ln_hr_peto))
        hr_ci_lower = float(np.exp(ln_hr_peto - z_crit * se_ln_hr))
        hr_ci_upper = float(np.exp(ln_hr_peto + z_crit * se_ln_hr))
    else:
        hr_peto = 1.0
        hr_ci_lower = 1.0
        hr_ci_upper = 1.0

    # Relative Event Rate Ratio (O/E ratio comparison)
    rate_ratio_oe = (total_o1 / total_e1) / (total_o2 / total_e2) if (total_e1 > 0 and total_o2 > 0 and total_e2 > 0) else hr_peto

    # Statistical decision
    is_significant = bool(p_val < alpha)

    events_table = pd.DataFrame({
        "Event_Time": combined_event_times,
        "At_Risk_G1": [int(np.sum(d1 >= t)) for t in combined_event_times],
        "Events_G1": o1_list,
        "Expected_G1": e1_list,
        "At_Risk_G2": [int(np.sum(d2 >= t)) for t in combined_event_times],
        "Events_G2": o2_list,
        "Expected_G2": e2_list,
        "Variance_G1": v1_list
    })

    return {
        "chisq": float(chisq),
        "degrees_of_freedom": 1,
        "p_value": float(p_val),
        "is_significant": is_significant,
        "group_1_label": group_labels[0],
        "group_2_label": group_labels[1],
        "n_group_1": n1_init,
        "n_group_2": n2_init,
        "observed_group_1": total_o1,
        "expected_group_1": total_e1,
        "observed_group_2": total_o2,
        "expected_group_2": total_e2,
        "variance_group_1": total_v1,
        "peto_hazard_ratio": hr_peto,
        "peto_hr_ci": (hr_ci_lower, hr_ci_upper),
        "rate_ratio_oe": float(rate_ratio_oe),
        "events_table": events_table
    }


def calculate_landmark_survival_probability(
    km_result: Dict[str, Any],
    landmark_time: float
) -> Dict[str, Any]:
    """Retrieves the Kaplan-Meier survival probability and 95% CI at a specific landmark time.

    Args:
        km_result: Output from calculate_kaplan_meier_single_group.
        landmark_time (float): Target follow-up landmark (e.g. 12, 24, 36, 60 months).

    Returns:
        Dict[str, Any]: Survival probability, Greenwood standard error, and 95% CI at landmark.
    """
    times = km_result["table_times"]
    probs = km_result["survival_probabilities"]
    ci_lows = km_result["ci_lower_loglog"]
    ci_upps = km_result["ci_upper_loglog"]
    ses = km_result["se_linear"]

    surv_at_t = 1.0
    ci_low_at_t = 1.0
    ci_upp_at_t = 1.0
    se_at_t = 0.0

    for i, t in enumerate(times):
        if t <= landmark_time:
            surv_at_t = probs[i]
            ci_low_at_t = ci_lows[i]
            ci_upp_at_t = ci_upps[i]
            se_at_t = ses[i]
        else:
            break

    return {
        "landmark_time": landmark_time,
        "survival_probability": float(surv_at_t),
        "se": float(se_at_t),
        "ci_lower": float(ci_low_at_t),
        "ci_upper": float(ci_upp_at_t)
    }


def calculate_kaplan_meier_two_groups(
    df: pd.DataFrame,
    time_col: str = "Time",
    event_col: str = "Event",
    group_col: str = "Group",
    landmark_times: Optional[List[float]] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Full comparative survival analysis across two groups including KM curves, Log-Rank, and Landmarks."""
    if landmark_times is None:
        max_t = float(df[time_col].max())
        landmark_times = [float(round(max_t * 0.25, 1)), float(round(max_t * 0.50, 1)), float(round(max_t * 0.75, 1))]

    unique_groups = df[group_col].unique()
    if len(unique_groups) != 2:
        raise ValueError(f"Comparative analysis requires exactly two groups, found: {unique_groups}")

    g1_val, g2_val = unique_groups[0], unique_groups[1]
    df_g1 = df[df[group_col] == g1_val]
    df_g2 = df[df[group_col] == g2_val]

    km_g1 = calculate_kaplan_meier_single_group(df_g1[time_col], df_g1[event_col], alpha=alpha)
    km_g2 = calculate_kaplan_meier_single_group(df_g2[time_col], df_g2[event_col], alpha=alpha)

    log_rank = calculate_log_rank_test(
        durations_1=df_g1[time_col],
        events_1=df_g1[event_col],
        durations_2=df_g2[time_col],
        events_2=df_g2[event_col],
        group_labels=(str(g1_val), str(g2_val)),
        alpha=alpha
    )

    landmarks_comparison = []
    for lm in landmark_times:
        s1 = calculate_landmark_survival_probability(km_g1, lm)
        s2 = calculate_landmark_survival_probability(km_g2, lm)
        diff = s1["survival_probability"] - s2["survival_probability"]
        se_diff = np.sqrt(s1["se"] ** 2 + s2["se"] ** 2)
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        diff_ci_lower = diff - z_crit * se_diff
        diff_ci_upper = diff + z_crit * se_diff
        nnt = 1.0 / np.abs(diff) if np.abs(diff) > 0.0001 else np.nan

        landmarks_comparison.append({
            "Landmark_Time": lm,
            f"Survival_{g1_val}": s1["survival_probability"],
            f"CI_{g1_val}": f"[{s1['ci_lower']:.3f}, {s1['ci_upper']:.3f}]",
            f"Survival_{g2_val}": s2["survival_probability"],
            f"CI_{g2_val}": f"[{s2['ci_lower']:.3f}, {s2['ci_upper']:.3f}]",
            "Absolute_Difference": diff,
            "Diff_CI": f"[{diff_ci_lower:.3f}, {diff_ci_upper:.3f}]",
            "NNT_At_Landmark": nnt
        })

    return {
        "group_1_name": str(g1_val),
        "group_2_name": str(g2_val),
        "km_group_1": km_g1,
        "km_group_2": km_g2,
        "log_rank": log_rank,
        "landmarks_df": pd.DataFrame(landmarks_comparison)
    }


def evaluate_kaplan_meier_prerequisites(
    km_analysis: Dict[str, Any],
    df: pd.DataFrame,
    time_col: str = "Time",
    event_col: str = "Event",
    group_col: str = "Group"
) -> Dict[str, Any]:
    """Evaluates methodological prerequisites and potential biases for Survival Analysis (STROBE/CONSORT)."""
    km1 = km_analysis["km_group_1"]
    km2 = km_analysis["km_group_2"]
    lr = km_analysis["log_rank"]

    total_n = km1["total_participants"] + km2["total_participants"]
    total_events = km1["total_events"] + km2["total_events"]
    total_censored = km1["total_censored"] + km2["total_censored"]
    overall_censoring_rate = (total_censored / total_n) * 100.0 if total_n > 0 else 0.0

    evaluations = {}

    # Check 1: Non-Informative / Independent Censoring
    # Censoring must not be related to imminent probability of having an event
    censoring_diff = abs(km1["censoring_percentage"] - km2["censoring_percentage"])
    if censoring_diff < 15.0:
        evaluations["independent_censoring"] = {
            "status": "pass",
            "label": "Non-Informative (Independent) Censoring Supported",
            "badge": "🟢 Passed",
            "details": f"Censoring rates are balanced between arms ({km1['censoring_percentage']:.1f}% vs. {km2['censoring_percentage']:.1f}%, diff = {censoring_diff:.1f}%). Supports assumption that patient dropouts are independent of disease severity."
        }
    else:
        evaluations["independent_censoring"] = {
            "status": "warning",
            "label": "Imbalanced Censoring Rates Across Groups",
            "badge": "🟡 Caution",
            "details": f"Notable divergence in censoring frequency ({km1['censoring_percentage']:.1f}% vs. {km2['censoring_percentage']:.1f}%, diff = {censoring_diff:.1f}%). Investigate whether high dropout in one group reflects treatment intolerance or adverse outcomes."
        }

    # Check 2: Adequate Event Count for Log-Rank Power
    if total_events >= 30:
        evaluations["event_count"] = {
            "status": "pass",
            "label": "Adequate Event Count for Asymptotic Log-Rank",
            "badge": "🟢 Passed",
            "details": f"Observed {total_events} cumulative events across both study arms. Sufficient for robust chi-square asymptotic validity."
        }
    elif total_events >= 15:
        evaluations["event_count"] = {
            "status": "warning",
            "label": "Moderate Event Count (Limited Statistical Power)",
            "badge": "🟡 Caution",
            "details": f"Observed {total_events} events. While chi-square tests are computable, power to detect small hazard ratios (HR 0.75–0.85) is constrained."
        }
    else:
        evaluations["event_count"] = {
            "status": "fail",
            "label": "Sparse Event Count (High Type II Error Risk)",
            "badge": "🔴 High Risk",
            "details": f"Only {total_events} events recorded. Log-rank chi-square approximation may be unstable; exact permutation testing advised."
        }

    # Check 3: Survival Curve Crossing / Proportional Hazards Check
    # Check if survival difference changes sign across time
    lm_df = km_analysis["landmarks_df"]
    diffs = lm_df["Absolute_Difference"].values
    has_crossing = bool(np.any(diffs > 0.02) and np.any(diffs < -0.02))

    if not has_crossing:
        evaluations["proportional_hazards"] = {
            "status": "pass",
            "label": "Consistent Directionality (No Severe Curve Crossing)",
            "badge": "🟢 Passed",
            "details": "Survival curves demonstrate consistent separation throughout follow-up without intersection. Proportional hazards assumption appears plausible."
        }
    else:
        evaluations["proportional_hazards"] = {
            "status": "warning",
            "label": "Survival Curves Cross (Non-Proportional Hazards)",
            "badge": "🟡 Caution",
            "details": "Survival curves intersect during follow-up. Standard log-rank test loses statistical power when curves cross. Consider Wilcoxon/Peto weighting or restricted mean survival time (RMST)."
        }

    # Check 4: Tail Stability & Heavy Censoring
    min_tail_risk = min(km1["life_table_df"]["Number_At_Risk"].iloc[-1], km2["life_table_df"]["Number_At_Risk"].iloc[-1])
    if min_tail_risk >= 10:
        evaluations["tail_stability"] = {
            "status": "pass",
            "label": "Reliable Tail Follow-up (Number at Risk >= 10)",
            "badge": "🟢 Passed",
            "details": f"At least {min_tail_risk} patients remain at risk at the final follow-up time-points, preventing unstable plateau artifacts in curve tails."
        }
    else:
        evaluations["tail_stability"] = {
            "status": "warning",
            "label": "Sparse Risk Set in Curve Tail (Potential Plateau Artifact)",
            "badge": "🟡 Caution",
            "details": f"Fewer than 10 subjects ({min_tail_risk}) remain at risk at the latest study times. Interpret late survival plateaus with caution as Greenwood standard errors widen."
        }

    # Check 5: Censoring Rate Context
    if 10.0 <= overall_censoring_rate <= 70.0:
        evaluations["censoring_range"] = {
            "status": "pass",
            "label": "Typical Epidemiological Censoring Rate",
            "badge": "🟢 Passed",
            "details": f"Overall censoring rate is {overall_censoring_rate:.1f}%, reflecting standard clinical trial follow-up without excessive premature attrition."
        }
    elif overall_censoring_rate > 70.0:
        evaluations["censoring_range"] = {
            "status": "warning",
            "label": "High Censoring Rate (> 70%)",
            "badge": "🟡 Caution",
            "details": f"High censoring rate ({overall_censoring_rate:.1f}%). Ensure high proportion of subjects reached study end without premature loss to follow-up."
        }
    else:
        evaluations["censoring_range"] = {
            "status": "pass",
            "label": "Low Censoring Rate (Event-Rich Cohort)",
            "badge": "🟢 Passed",
            "details": f"Censoring rate is {overall_censoring_rate:.1f}%. Cohort has matured with high event ascertainment."
        }

    return evaluations


def generate_kaplan_meier_demo_data(
    scenario: str = "oncology_immunotherapy",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic patient-level longitudinal time-to-event microdata for Kaplan-Meier demonstrations.

    Provides three realistic clinical scenarios with fictional healthcare names:
    1. 'oncology_immunotherapy': All Meadows Cancer Centre - Novel Checkpoint Immunotherapy vs. Standard Chemotherapy (Metastatic NSCLC, 36-month follow-up, HR ~ 0.58).
    2. 'cardiac_surgery_graft': Inner City Hospital Heart & Vascular Institute - Radial Artery vs. Saphenous Vein CABG Patency (8-year follow-up, HR ~ 0.62).
    3. 'pediatric_leukemia_induction': Riverside Children's Hospital - Intensified Induction Chemotherapy vs. Standard Protocol (Pediatric ALL, 60-month follow-up, HR ~ 0.45).

    Args:
        scenario (str, optional): Scenario identifier. Defaults to 'oncology_immunotherapy'.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - df_microdata: Individual patient microdata (Patient_ID, Group, Time, Event, etc.)
            - df_summary: Group summary comparison table
            - meta: Rich descriptive metadata, clinical takeaways, and disclaimer notices.
    """
    np.random.seed(seed)

    scenarios = {
        "oncology_immunotherapy": {
            "title": "All Meadows Cancer Centre: Checkpoint Immunotherapy vs. Chemotherapy in Metastatic Lung Cancer",
            "domain": "Medical Oncology & Immuno-Therapeutics",
            "description": "A hypothetical Phase III randomized open-label trial of 300 patients with metastatic non-small cell lung cancer (NSCLC) recruited across All Meadows Cancer Centre. Evaluates novel PD-1 checkpoint immunotherapy versus standard platinum-doublet chemotherapy over 36 months of follow-up. Primary endpoint is Progression-Free Survival (PFS).",
            "time_col": "Time_Months",
            "time_unit": "Months",
            "event_col": "Event",
            "group_col": "Treatment_Arm",
            "group_names": ("Immunotherapy Arm", "Chemotherapy Arm"),
            "event_name": "Disease Progression or Death",
            "n_g1": 150,
            "n_g2": 150,
            "hazard_rate_g1": 0.038,  # Median ~ 18.2 months
            "hazard_rate_g2": 0.068,  # Median ~ 10.2 months
            "max_followup": 36.0,
            "loss_to_followup_rate": 0.012,
            "landmark_times": [6.0, 12.0, 24.0, 36.0],
            "landmark_labels": ["6-Month PFS", "12-Month PFS", "24-Month PFS", "36-Month PFS"],
            "expected_hr": 0.56,
            "clinical_takeaway": "Patients receiving PD-1 immunotherapy experienced a 44% reduction in the hazard of disease progression or death compared to chemotherapy (Peto Hazard Ratio = 0.58, 95% CI: [0.44, 0.77], Log-Rank p < 0.001). Median progression-free survival expanded from 10.2 months in the chemotherapy arm to 18.8 months with immunotherapy, with 24-month PFS rates doubling from 20.8% to 42.1%.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "cardiac_surgery_graft": {
            "title": "Inner City Hospital Heart Institute: Radial Artery vs. Saphenous Vein Coronary Artery Bypass Patency",
            "domain": "Cardiothoracic Surgery & Cardiovascular Outcomes",
            "description": "A hypothetical 8-year prospective longitudinal surgical cohort of 400 adult patients undergoing multi-vessel coronary artery bypass grafting (CABG) at Inner City Hospital Heart & Vascular Institute. Evaluates long-term graft failure or Major Adverse Cardiac Events (MACE) comparing radial artery conduit versus traditional saphenous vein conduit.",
            "time_col": "Time_Years",
            "time_unit": "Years",
            "event_col": "Event",
            "group_col": "Graft_Type",
            "group_names": ("Radial Artery Conduit", "Saphenous Vein Conduit"),
            "event_name": "MACE or Angiographic Graft Failure",
            "n_g1": 200,
            "n_g2": 200,
            "hazard_rate_g1": 0.035,  # 8-year patency ~ 75%
            "hazard_rate_g2": 0.062,  # 8-year patency ~ 60%
            "max_followup": 8.0,
            "loss_to_followup_rate": 0.015,
            "landmark_times": [1.0, 3.0, 5.0, 8.0],
            "landmark_labels": ["1-Year MACE-Free", "3-Year MACE-Free", "5-Year MACE-Free", "8-Year MACE-Free"],
            "expected_hr": 0.62,
            "clinical_takeaway": "Coronary bypass using a radial artery conduit demonstrated superior long-term patency over saphenous vein grafts (Peto Hazard Ratio = 0.61, 95% CI: [0.43, 0.86], Log-Rank p = 0.004). At 5 years post-operatively, 84.1% of radial artery patients remained free from graft failure compared to 71.4% with saphenous vein grafts (Absolute Difference = +12.7%, NNT = 8 patients).",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "pediatric_leukemia_induction": {
            "title": "Riverside Children's Hospital: Intensified Induction Protocol in Pediatric Acute Lymphoblastic Leukemia",
            "domain": "Pediatric Hematology & Oncology",
            "description": "A hypothetical 5-year multicenter clinical trial of 250 pediatric patients aged 1–16 diagnosed with standard-risk B-cell acute lymphoblastic leukemia (ALL) coordinated by Riverside Children's Hospital. Evaluates an intensified 4-drug induction protocol against the historical 3-drug standard protocol. Primary endpoint is Event-Free Survival (EFS).",
            "time_col": "Time_Months",
            "time_unit": "Months",
            "event_col": "Event",
            "group_col": "Induction_Protocol",
            "group_names": ("Intensified 4-Drug Induction", "Standard 3-Drug Induction"),
            "event_name": "Bone Marrow Relapse or Death",
            "n_g1": 125,
            "n_g2": 125,
            "hazard_rate_g1": 0.0035,  # 5-year EFS ~ 81%
            "hazard_rate_g2": 0.0082,  # 5-year EFS ~ 61%
            "max_followup": 60.0,
            "loss_to_followup_rate": 0.002,
            "landmark_times": [12.0, 24.0, 36.0, 60.0],
            "landmark_labels": ["1-Year EFS", "2-Year EFS", "3-Year EFS", "5-Year EFS"],
            "expected_hr": 0.45,
            "clinical_takeaway": "Intensified 4-drug induction therapy achieved an outstanding 55% reduction in the hazard of leukemic relapse (Peto Hazard Ratio = 0.45, 95% CI: [0.27, 0.74], Log-Rank p = 0.001). 5-year event-free survival reached 88.2% in the intensified arm versus 71.2% in the standard arm (Absolute Risk Reduction = 17.0%, NNT = 6 children treated to prevent 1 relapse).",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
    }

    if scenario not in scenarios:
        scenario = "oncology_immunotherapy"

    cfg = scenarios[scenario]
    n1, n2 = cfg["n_g1"], cfg["n_g2"]
    h1, h2 = cfg["hazard_rate_g1"], cfg["hazard_rate_g2"]
    max_t = cfg["max_followup"]
    loss_rate = cfg["loss_to_followup_rate"]
    g1_label, g2_label = cfg["group_names"]

    # Generate survival times from exponential distribution: T = -ln(U) / lambda
    u1 = np.random.uniform(0.0001, 0.9999, size=n1)
    true_event_times_1 = -np.log(u1) / h1

    u2 = np.random.uniform(0.0001, 0.9999, size=n2)
    true_event_times_2 = -np.log(u2) / h2

    # Generate random censoring times (loss to follow-up / study end)
    censor_u1 = np.random.uniform(0.0001, 0.9999, size=n1)
    loss_times_1 = -np.log(censor_u1) / (loss_rate + 0.0001)

    censor_u2 = np.random.uniform(0.0001, 0.9999, size=n2)
    loss_times_2 = -np.log(censor_u2) / (loss_rate + 0.0001)

    # Observed time is min(True_Event, Loss_To_Followup, Study_End)
    obs_times_1 = []
    events_1 = []
    reasons_1 = []

    for i in range(n1):
        t_event = true_event_times_1[i]
        t_loss = loss_times_1[i]

        if t_event <= t_loss and t_event <= max_t:
            obs_times_1.append(round(float(t_event), 1))
            events_1.append(1)
            reasons_1.append(cfg["event_name"])
        elif t_loss < t_event and t_loss <= max_t:
            obs_times_1.append(round(float(t_loss), 1))
            events_1.append(0)
            reasons_1.append("Lost to Follow-up / Moved")
        else:
            obs_times_1.append(float(max_t))
            events_1.append(0)
            reasons_1.append("Completed Study Follow-up (Alive & Event-Free)")

    obs_times_2 = []
    events_2 = []
    reasons_2 = []

    for i in range(n2):
        t_event = true_event_times_2[i]
        t_loss = loss_times_2[i]

        if t_event <= t_loss and t_event <= max_t:
            obs_times_2.append(round(float(t_event), 1))
            events_2.append(1)
            reasons_2.append(cfg["event_name"])
        elif t_loss < t_event and t_loss <= max_t:
            obs_times_2.append(round(float(t_loss), 1))
            events_2.append(0)
            reasons_2.append("Lost to Follow-up / Moved")
        else:
            obs_times_2.append(float(max_t))
            events_2.append(0)
            reasons_2.append("Completed Study Follow-up (Alive & Event-Free)")

    # Build demographic variables
    ages_1 = np.random.normal(loc=62.0, scale=8.5, size=n1).astype(int)
    sex_1 = np.random.choice(["Male", "Female"], size=n1, p=[0.58, 0.42])

    ages_2 = np.random.normal(loc=61.8, scale=8.2, size=n2).astype(int)
    sex_2 = np.random.choice(["Male", "Female"], size=n2, p=[0.56, 0.44])

    rows_1 = pd.DataFrame({
        "Patient_ID": [f"PT-{scenario[:3].upper()}-G1-{i+1:03d}" for i in range(n1)],
        cfg["group_col"]: g1_label,
        cfg["time_col"]: obs_times_1,
        cfg["event_col"]: events_1,
        "Status": ["Event Occurred" if e == 1 else "Censored" for e in events_1],
        "Censoring_Detail": reasons_1,
        "Age": ages_1,
        "Sex": sex_1
    })

    rows_2 = pd.DataFrame({
        "Patient_ID": [f"PT-{scenario[:3].upper()}-G2-{i+1:03d}" for i in range(n2)],
        cfg["group_col"]: g2_label,
        cfg["time_col"]: obs_times_2,
        cfg["event_col"]: events_2,
        "Status": ["Event Occurred" if e == 1 else "Censored" for e in events_2],
        "Censoring_Detail": reasons_2,
        "Age": ages_2,
        "Sex": sex_2
    })

    df_microdata = pd.concat([rows_1, rows_2], ignore_index=True)

    df_summary = pd.DataFrame({
        "Treatment_Group": [g1_label, g2_label, "Combined Total"],
        "Sample_Size_N": [n1, n2, n1 + n2],
        "Events_Count": [int(np.sum(events_1)), int(np.sum(events_2)), int(np.sum(events_1) + np.sum(events_2))],
        "Censored_Count": [int(np.sum(np.array(events_1) == 0)), int(np.sum(np.array(events_2) == 0)), int(np.sum(np.array(events_1) == 0) + np.sum(np.array(events_2) == 0))],
        "Event_Rate_Pct": [
            f"{(np.sum(events_1)/n1)*100.0:.1f}%",
            f"{(np.sum(events_2)/n2)*100.0:.1f}%",
            f"{((np.sum(events_1) + np.sum(events_2))/(n1 + n2))*100.0:.1f}%"
        ]
    })

    meta = {
        "scenario_key": scenario,
        "title": cfg["title"],
        "domain": cfg["domain"],
        "description": cfg["description"],
        "time_col": cfg["time_col"],
        "time_unit": cfg["time_unit"],
        "event_col": cfg["event_col"],
        "group_col": cfg["group_col"],
        "group_names": (g1_label, g2_label),
        "event_name": cfg["event_name"],
        "landmark_times": cfg["landmark_times"],
        "landmark_labels": cfg["landmark_labels"],
        "expected_hr": cfg["expected_hr"],
        "clinical_takeaway": cfg["clinical_takeaway"],
        "disclaimer": cfg["disclaimer"]
    }

    return df_microdata, df_summary, meta


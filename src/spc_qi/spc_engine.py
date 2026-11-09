"""Statistical Process Control (SPC) & Quality Improvement (QI) Calculation Engine.

Provides pure-math, zero-external-dependency algorithms and synthetic data generators for:
1. Healthcare Run Charts (Median, Runs Count, Shifts, Trends, Astronomical Points)
2. Shewhart XmR / I-MR Charts (Individuals, Moving Range, 3-Sigma Limits, Special Cause Rules)
3. Proportions: p-Chart & Laney p'-Chart (Binomial Rates & Overdispersion Correction)
4. Count Rates: u-Chart & Laney u'-Chart (Poisson Rates per Exposure Opportunity)
5. NHS "Making Data Count" System Icons & Capability Analysis (Cp / Cpk & Target Assurance)
"""

import math
import random
from typing import Dict, Any, List, Tuple, Optional

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    pd = None
    np = None
    HAS_PANDAS = False


# =====================================================================
# Math Helper Utilities
# =====================================================================

def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0)


def _stdev(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


# Perla, Provost, and Murray (2011) Table for Critical Runs at alpha=0.05
# Maps useful observations n_useful (10 to 60) to (R_min, R_max)
RUNS_TABLE_05: Dict[int, Tuple[int, int]] = {
    10: (3, 9), 11: (3, 10), 12: (3, 11), 13: (4, 11), 14: (4, 12),
    15: (4, 13), 16: (5, 13), 17: (5, 14), 18: (5, 15), 19: (6, 15),
    20: (6, 16), 21: (6, 17), 22: (7, 17), 23: (7, 18), 24: (7, 19),
    25: (8, 19), 26: (8, 20), 27: (8, 21), 28: (9, 21), 29: (9, 22),
    30: (9, 23), 31: (10, 23), 32: (10, 24), 33: (10, 25), 34: (11, 25),
    35: (11, 26), 36: (11, 27), 37: (12, 27), 38: (12, 28), 39: (12, 29),
    40: (13, 29), 41: (13, 30), 42: (13, 31), 43: (14, 31), 44: (14, 32),
    45: (14, 33), 46: (15, 33), 47: (15, 34), 48: (15, 35), 49: (16, 35),
    50: (16, 36), 55: (18, 39), 60: (20, 42)
}

def _lookup_runs_limits(n_useful: int) -> Tuple[int, int]:
    if n_useful < 10:
        return (1, max(2, n_useful))
    if n_useful in RUNS_TABLE_05:
        return RUNS_TABLE_05[n_useful]
    if n_useful > 60:
        # Normal approximation: E[R] = 1 + n/2, SE = sqrt((n-1)/4)
        exp_r = 1.0 + n_useful / 2.0
        se_r = math.sqrt((n_useful - 1) / 4.0)
        return (max(1, math.floor(exp_r - 1.96 * se_r)), math.ceil(exp_r + 1.96 * se_r))
    # Interpolate closest key below
    lower_key = max(k for k in RUNS_TABLE_05.keys() if k <= n_useful)
    return RUNS_TABLE_05[lower_key]


# =====================================================================
# 1. Healthcare Run Chart Engine
# =====================================================================

def generate_run_chart_dataset(
    n_points: int = 24,
    baseline_median: float = 14.0,
    intervention_point: int = 13,
    shift_magnitude: float = -5.0,
    seed: int = 42
) -> Any:
    """Generates synthetic longitudinal healthcare data for a Run Chart.
    
    Example: Hospital ward surgical site infections (SSI) before and after chlorhexidine prep.
    """
    rng = random.Random(seed)
    records = []
    
    for t in range(1, n_points + 1):
        is_post = 1 if t >= intervention_point else 0
        effect = shift_magnitude if is_post else 0.0
        # Poisson-like count around baseline
        val = max(1, int(round(rng.gauss(baseline_median + effect, 2.2))))
        
        # Inject one astronomical shock at period 7 for educational demonstration
        if t == 7:
            val += 9
            
        records.append({
            "period": t,
            "period_label": f"M{t:02d}",
            "value": val,
            "is_post_intervention": is_post
        })
        
    if HAS_PANDAS:
        return pd.DataFrame(records)
    return records


def calculate_run_chart_model(data: Any, baseline_periods: Optional[int] = None) -> Dict[str, Any]:
    """Evaluates healthcare run chart rules (Median, Useful Points, Runs, Shift, Trend, Astronomical)."""
    records = data.to_dict("records") if HAS_PANDAS and isinstance(data, pd.DataFrame) else data
    k = len(records)
    values = [float(r["value"]) for r in records]
    
    # Establish baseline median
    if baseline_periods and baseline_periods < k:
        baseline_vals = values[:baseline_periods]
    else:
        baseline_vals = values
        
    median_val = _median(baseline_vals)
    
    # Useful observations (excluding median ties)
    useful_indices = [i for i, v in enumerate(values) if v != median_val]
    n_useful = len(useful_indices)
    
    # Count Runs
    runs_count = 0
    if useful_indices:
        runs_count = 1
        last_sign = 1 if values[useful_indices[0]] > median_val else -1
        for idx in useful_indices[1:]:
            curr_sign = 1 if values[idx] > median_val else -1
            if curr_sign != last_sign:
                runs_count += 1
                last_sign = curr_sign
                
    r_min, r_max = _lookup_runs_limits(n_useful)
    runs_rule_violated = (runs_count < r_min) or (runs_count > r_max) if n_useful >= 10 else False
    
    # Rule 1: Shift (>= 6 consecutive points all above or all below median)
    shift_detected = False
    shift_indices = []
    curr_run = []
    last_dir = None
    
    for i, v in enumerate(values):
        if v == median_val:
            curr_run = []
            last_dir = None
            continue
        curr_dir = 1 if v > median_val else -1
        if curr_dir == last_dir:
            curr_run.append(i)
        else:
            if len(curr_run) >= 6:
                shift_detected = True
                shift_indices.extend(curr_run)
            curr_run = [i]
            last_dir = curr_dir
            
    if len(curr_run) >= 6:
        shift_detected = True
        shift_indices.extend(curr_run)
        
    # Rule 2: Trend (>= 5 consecutive points strictly increasing or decreasing)
    trend_detected = False
    trend_indices = []
    inc_run = [0] if k > 0 else []
    dec_run = [0] if k > 0 else []
    
    for i in range(1, k):
        # Increasing
        if values[i] > values[i - 1]:
            inc_run.append(i)
        else:
            if len(inc_run) >= 5:
                trend_detected = True
                trend_indices.extend(inc_run)
            inc_run = [i]
            
        # Decreasing
        if values[i] < values[i - 1]:
            dec_run.append(i)
        else:
            if len(dec_run) >= 5:
                trend_detected = True
                trend_indices.extend(dec_run)
            dec_run = [i]
            
    if len(inc_run) >= 5:
        trend_detected = True
        trend_indices.extend(inc_run)
    if len(dec_run) >= 5:
        trend_detected = True
        trend_indices.extend(dec_run)
        
    # Rule 4: Astronomical Point (Outlier obvious to anyone; > 3.5 SDs from median)
    sd_est = _stdev(values)
    astronomical_indices = []
    for i, v in enumerate(values):
        if abs(v - median_val) > 3.0 * sd_est and sd_est > 0:
            astronomical_indices.append(i)
            
    has_signal = shift_detected or trend_detected or runs_rule_violated or (len(astronomical_indices) > 0)
    
    return {
        "median": round(float(median_val), 2),
        "total_points": k,
        "n_useful": n_useful,
        "runs_count": runs_count,
        "expected_runs_min": r_min,
        "expected_runs_max": r_max,
        "rule_1_shift": {
            "status": "SIGNAL DETECTED (Shift >= 6 points)" if shift_detected else "NO SHIFT",
            "detected": shift_detected,
            "indices": list(set(shift_indices))
        },
        "rule_2_trend": {
            "status": "SIGNAL DETECTED (Trend >= 5 points)" if trend_detected else "NO TREND",
            "detected": trend_detected,
            "indices": list(set(trend_indices))
        },
        "rule_3_runs": {
            "status": f"NON-RANDOM ({runs_count} runs outside [{r_min}, {r_max}])" if runs_rule_violated else f"PASS ({runs_count} runs within [{r_min}, {r_max}])",
            "detected": runs_rule_violated
        },
        "rule_4_astronomical": {
            "status": f"ASTRONOMICAL POINT DETECTED ({len(astronomical_indices)} points)" if astronomical_indices else "NO ASTRONOMICAL POINTS",
            "detected": len(astronomical_indices) > 0,
            "indices": astronomical_indices
        },
        "has_special_cause_signal": has_signal
    }


# =====================================================================
# 2. Shewhart XmR / I-MR Chart Engine
# =====================================================================

def generate_xmr_dataset(
    n_points: int = 30,
    baseline_mean: float = 4.2,
    baseline_sd: float = 0.65,
    intervention_step: int = 20,
    improvement_effect: float = -1.1,
    seed: int = 42
) -> Any:
    """Generates synthetic continuous operational data for Shewhart XmR.
    
    Example: Average Emergency Department waiting hours from triage to physician consultation.
    """
    rng = random.Random(seed)
    records = []
    
    for t in range(1, n_points + 1):
        is_post = 1 if t >= intervention_step else 0
        effect = improvement_effect if is_post else 0.0
        val = max(1.2, rng.gauss(baseline_mean + effect, baseline_sd))
        
        # Inject one special cause shock in early period
        if t == 8:
            val += 2.4
            
        records.append({
            "week": t,
            "week_label": f"Wk {t:02d}",
            "wait_hours": round(float(val), 2),
            "is_post_improvement": is_post
        })
        
    if HAS_PANDAS:
        return pd.DataFrame(records)
    return records


def calculate_xmr_model(data: Any, baseline_periods: Optional[int] = None) -> Dict[str, Any]:
    """Calculates Shewhart XmR (Individuals and Moving Range) 3-sigma control limits and special causes."""
    records = data.to_dict("records") if HAS_PANDAS and isinstance(data, pd.DataFrame) else data
    k = len(records)
    
    # Extract values
    values = [float(r["wait_hours"]) if "wait_hours" in r else float(r["value"]) for r in records]
    
    # Baseline subset
    base_k = baseline_periods if baseline_periods and baseline_periods < k else k
    base_values = values[:base_k]
    
    # 1. Individuals Centerline
    x_bar = _mean(base_values)
    
    # 2. Moving Ranges (MR_i = |X_i - X_{i-1}|)
    moving_ranges = [0.0]
    for i in range(1, k):
        moving_ranges.append(abs(values[i] - values[i - 1]))
        
    base_mr = moving_ranges[1:base_k] if base_k > 1 else [1.0]
    mr_bar = _mean(base_mr)
    
    # 3. Unbiased Sigma Estimator via d2 = 1.128
    sigma_hat = mr_bar / 1.128 if mr_bar > 0 else 0.001
    
    # 4. Control Limits for Individuals (X)
    ucl_x = x_bar + 2.660 * mr_bar
    lcl_x = max(0.0, x_bar - 2.660 * mr_bar)
    
    # Control Limits for Moving Range (MR)
    ucl_mr = 3.267 * mr_bar
    lcl_mr = 0.0
    
    # 5. Special Cause Audit (Rules 1-4)
    # Rule 1: Beyond Limits
    rule1_indices = [i for i, v in enumerate(values) if v > ucl_x or v < lcl_x]
    
    # Rule 2: 8 consecutive points on one side of centerline
    rule2_indices = []
    run = []
    last_side = 0
    for i, v in enumerate(values):
        curr_side = 1 if v > x_bar else (-1 if v < x_bar else 0)
        if curr_side != 0 and curr_side == last_side:
            run.append(i)
        else:
            if len(run) >= 8:
                rule2_indices.extend(run)
            run = [i] if curr_side != 0 else []
            last_side = curr_side
    if len(run) >= 8:
        rule2_indices.extend(run)
        
    # Rule 3: 6 consecutive points strictly increasing or decreasing
    rule3_indices = []
    inc = [0]
    dec = [0]
    for i in range(1, k):
        if values[i] > values[i - 1]:
            inc.append(i)
        else:
            if len(inc) >= 6:
                rule3_indices.extend(inc)
            inc = [i]
            
        if values[i] < values[i - 1]:
            dec.append(i)
        else:
            if len(dec) >= 6:
                rule3_indices.extend(dec)
            dec = [i]
    if len(inc) >= 6:
        rule3_indices.extend(inc)
    if len(dec) >= 6:
        rule3_indices.extend(dec)
        
    all_special_cause_indices = sorted(list(set(rule1_indices + rule2_indices + rule3_indices)))
    
    return {
        "x_bar": round(float(x_bar), 3),
        "mr_bar": round(float(mr_bar), 3),
        "sigma_hat": round(float(sigma_hat), 3),
        "ucl_x": round(float(ucl_x), 3),
        "lcl_x": round(float(lcl_x), 3),
        "ucl_mr": round(float(ucl_mr), 3),
        "lcl_mr": round(float(lcl_mr), 3),
        "moving_ranges": [round(float(m), 2) for m in moving_ranges],
        "rule_1_beyond_limits": {
            "count": len(rule1_indices),
            "indices": rule1_indices,
            "status": "ALARM (Points outside 3-sigma limits)" if rule1_indices else "PASS"
        },
        "rule_2_shift_of_8": {
            "count": len(rule2_indices),
            "indices": list(set(rule2_indices)),
            "status": "ALARM (Shift >= 8 points on one side)" if rule2_indices else "PASS"
        },
        "rule_3_trend_of_6": {
            "count": len(rule3_indices),
            "indices": list(set(rule3_indices)),
            "status": "ALARM (Trend >= 6 points monotonic)" if rule3_indices else "PASS"
        },
        "special_cause_detected": len(all_special_cause_indices) > 0,
        "special_cause_indices": all_special_cause_indices
    }


# =====================================================================
# 3. Proportion Charts: p-Chart & Laney p'-Chart Engine
# =====================================================================

def generate_p_chart_dataset(
    n_periods: int = 24,
    base_p: float = 0.085,
    min_n: int = 1400,
    max_n: int = 2800,
    intervention_step: int = 15,
    effect_reduction: float = -0.025,
    overdispersion_inflation: float = 1.45,
    seed: int = 42
) -> Any:
    """Generates synthetic binomial proportion data with varying sample size and overdispersion."""
    rng = random.Random(seed)
    records = []
    
    for t in range(1, n_periods + 1):
        n_i = rng.randint(min_n, max_n)
        is_post = 1 if t >= intervention_step else 0
        true_p = max(0.015, base_p + (effect_reduction if is_post else 0.0))
        
        # Binomial sampling noise plus between-subgroup cluster shock
        se_binom = math.sqrt(true_p * (1 - true_p) / n_i)
        binom_noise = rng.gauss(0, se_binom)
        cluster_shock = rng.gauss(0, overdispersion_inflation * se_binom)
        sample_p = max(0.01, min(0.35, true_p + binom_noise + cluster_shock))
        defect_count = int(round(sample_p * n_i))
        rate_pct = round((defect_count / n_i) * 100.0, 2)
        
        records.append({
            "period": t,
            "period_label": f"M{t:02d}",
            "discharges_n": n_i,
            "readmissions_x": defect_count,
            "readmission_rate_pct": rate_pct,
            "is_post": is_post
        })
        
    if HAS_PANDAS:
        return pd.DataFrame(records)
    return records


def calculate_p_chart_model(data: Any, baseline_periods: Optional[int] = None) -> Dict[str, Any]:
    """Calculates standard p-chart and Laney p'-chart with overdispersion correction."""
    records = data.to_dict("records") if HAS_PANDAS and isinstance(data, pd.DataFrame) else data
    k = len(records)
    
    base_k = baseline_periods if baseline_periods and baseline_periods < k else k
    base_records = records[:base_k]
    
    tot_defects = sum(r["readmissions_x"] for r in base_records)
    tot_sample = sum(r["discharges_n"] for r in base_records)
    p_bar = tot_defects / tot_sample if tot_sample > 0 else 0.05
    
    # 1. Standard p-chart limits
    std_ucl = []
    std_lcl = []
    z_scores = []
    
    for r in records:
        n_i = r["discharges_n"]
        p_i = r["readmissions_x"] / n_i if n_i > 0 else p_bar
        se_i = math.sqrt(p_bar * (1 - p_bar) / n_i) if n_i > 0 else 0.01
        
        ucl_i = (p_bar + 3.0 * se_i) * 100.0
        lcl_i = max(0.0, (p_bar - 3.0 * se_i) * 100.0)
        std_ucl.append(round(float(ucl_i), 3))
        std_lcl.append(round(float(lcl_i), 3))
        
        # Standardized z-score: (p_i - p_bar) / SE_i
        z_scores.append((p_i - p_bar) / se_i if se_i > 0 else 0.0)
        
    # 2. Laney Sigma_z (Overdispersion factor via moving ranges of z-scores)
    base_z = z_scores[:base_k]
    mr_z = [abs(base_z[i] - base_z[i - 1]) for i in range(1, len(base_z))]
    avg_mr_z = _mean(mr_z) if mr_z else 1.0
    sigma_z = avg_mr_z / 1.128
    
    # 3. Laney p' control limits
    laney_ucl = []
    laney_lcl = []
    std_alarms = []
    laney_alarms = []
    
    for i, r in enumerate(records):
        n_i = r["discharges_n"]
        obs_rate = r["readmission_rate_pct"]
        se_i = math.sqrt(p_bar * (1 - p_bar) / n_i) if n_i > 0 else 0.01
        
        ucl_laney = (p_bar + 3.0 * sigma_z * se_i) * 100.0
        lcl_laney = max(0.0, (p_bar - 3.0 * sigma_z * se_i) * 100.0)
        laney_ucl.append(round(float(ucl_laney), 3))
        laney_lcl.append(round(float(lcl_laney), 3))
        
        if obs_rate > std_ucl[i] or obs_rate < std_lcl[i]:
            std_alarms.append(i)
        if obs_rate > ucl_laney or obs_rate < lcl_laney:
            laney_alarms.append(i)
            
    is_overdispersed = sigma_z > 1.30
    
    return {
        "p_bar": round(float(p_bar), 4),
        "p_bar_pct": round(float(p_bar * 100.0), 2),
        "sigma_z": round(float(sigma_z), 3),
        "is_overdispersed": is_overdispersed,
        "dispersion_status": "OVERDISPERSED (Laney p' Required)" if is_overdispersed else "NORMAL BINOMIAL (p-chart Valid)",
        "std_ucl": std_ucl,
        "std_lcl": std_lcl,
        "laney_ucl": laney_ucl,
        "laney_lcl": laney_lcl,
        "std_alarms_count": len(std_alarms),
        "laney_alarms_count": len(laney_alarms),
        "false_alarm_difference": len(std_alarms) - len(laney_alarms),
        "std_alarms_indices": std_alarms,
        "laney_alarms_indices": laney_alarms
    }


# =====================================================================
# 4. Count Rates: u-Chart & Laney u'-Chart Engine
# =====================================================================

def generate_u_chart_dataset(
    n_periods: int = 24,
    base_rate: float = 4.8,  # falls per 1,000 bed-days
    min_bed_days: int = 3200,
    max_bed_days: int = 5800,
    intervention_step: int = 15,
    effect_reduction: float = -1.4,
    overdispersion_inflation: float = 1.38,
    seed: int = 42
) -> Any:
    """Generates synthetic count rate data per occupied bed-day exposure."""
    rng = random.Random(seed)
    records = []
    
    for t in range(1, n_periods + 1):
        bed_days = rng.randint(min_bed_days, max_bed_days)
        denom_units = bed_days / 1000.0  # units of 1,000 bed-days
        is_post = 1 if t >= intervention_step else 0
        true_rate = max(1.0, base_rate + (effect_reduction if is_post else 0.0))
        
        # Poisson sampling noise plus between-subgroup cluster shock
        se_rate = math.sqrt(true_rate / denom_units)
        poisson_noise = rng.gauss(0, se_rate)
        cluster_shock = rng.gauss(0, overdispersion_inflation * se_rate)
        obs_rate = max(0.5, true_rate + poisson_noise + cluster_shock)
        incident_count = int(round(obs_rate * denom_units))
        calc_rate = round(incident_count / denom_units, 2)
        
        records.append({
            "period": t,
            "period_label": f"M{t:02d}",
            "bed_days": bed_days,
            "exposure_units_1k": round(float(denom_units), 3),
            "falls_count": incident_count,
            "falls_rate_per_1k": calc_rate,
            "is_post": is_post
        })
        
    if HAS_PANDAS:
        return pd.DataFrame(records)
    return records


def calculate_u_chart_model(data: Any, baseline_periods: Optional[int] = None) -> Dict[str, Any]:
    """Calculates standard u-chart and Laney u'-chart with Poisson overdispersion correction."""
    records = data.to_dict("records") if HAS_PANDAS and isinstance(data, pd.DataFrame) else data
    k = len(records)
    
    base_k = baseline_periods if baseline_periods and baseline_periods < k else k
    base_records = records[:base_k]
    
    tot_counts = sum(r["falls_count"] for r in base_records)
    tot_exposure = sum(r["exposure_units_1k"] for r in base_records)
    u_bar = tot_counts / tot_exposure if tot_exposure > 0 else 4.0
    
    # 1. Standard u-chart limits
    std_ucl = []
    std_lcl = []
    z_scores = []
    
    for r in records:
        n_i = r["exposure_units_1k"]
        u_i = r["falls_rate_per_1k"]
        se_i = math.sqrt(u_bar / n_i) if n_i > 0 else 0.5
        
        ucl_i = u_bar + 3.0 * se_i
        lcl_i = max(0.0, u_bar - 3.0 * se_i)
        std_ucl.append(round(float(ucl_i), 3))
        std_lcl.append(round(float(lcl_i), 3))
        
        z_scores.append((u_i - u_bar) / se_i if se_i > 0 else 0.0)
        
    # 2. Laney Sigma_z for counts
    base_z = z_scores[:base_k]
    mr_z = [abs(base_z[i] - base_z[i - 1]) for i in range(1, len(base_z))]
    avg_mr_z = _mean(mr_z) if mr_z else 1.0
    sigma_z = avg_mr_z / 1.128
    
    # 3. Laney u' limits
    laney_ucl = []
    laney_lcl = []
    std_alarms = []
    laney_alarms = []
    
    for i, r in enumerate(records):
        n_i = r["exposure_units_1k"]
        obs_rate = r["falls_rate_per_1k"]
        se_i = math.sqrt(u_bar / n_i) if n_i > 0 else 0.5
        
        ucl_laney = u_bar + 3.0 * sigma_z * se_i
        lcl_laney = max(0.0, u_bar - 3.0 * sigma_z * se_i)
        laney_ucl.append(round(float(ucl_laney), 3))
        laney_lcl.append(round(float(lcl_laney), 3))
        
        if obs_rate > std_ucl[i] or obs_rate < std_lcl[i]:
            std_alarms.append(i)
        if obs_rate > ucl_laney or obs_rate < lcl_laney:
            laney_alarms.append(i)
            
    is_overdispersed = sigma_z > 1.30
    
    return {
        "u_bar": round(float(u_bar), 3),
        "sigma_z": round(float(sigma_z), 3),
        "is_overdispersed": is_overdispersed,
        "dispersion_status": "OVERDISPERSED (Laney u' Required)" if is_overdispersed else "NORMAL POISSON (u-chart Valid)",
        "std_ucl": std_ucl,
        "std_lcl": std_lcl,
        "laney_ucl": laney_ucl,
        "laney_lcl": laney_lcl,
        "std_alarms_count": len(std_alarms),
        "laney_alarms_count": len(laney_alarms),
        "std_alarms_indices": std_alarms,
        "laney_alarms_indices": laney_alarms
    }


# =====================================================================
# 5. NHS "Making Data Count" System Icons & Capability (Cp / Cpk)
# =====================================================================

def generate_mdc_dataset(
    n_periods: int = 26,
    baseline_compliance: float = 88.5,
    sd_val: float = 2.4,
    target_value: float = 95.0,
    seed: int = 42
) -> Any:
    """Generates synthetic bi-weekly NHS 4-hour emergency department compliance data."""
    rng = random.Random(seed)
    records = []
    
    for t in range(1, n_periods + 1):
        # Gradual slight drift upward
        drift = 0.08 * t
        val = min(99.0, max(75.0, rng.gauss(baseline_compliance + drift, sd_val)))
        
        records.append({
            "period": t,
            "period_label": f"Period {t:02d}",
            "compliance_pct": round(float(val), 2),
            "target": target_value
        })
        
    if HAS_PANDAS:
        return pd.DataFrame(records)
    return records


def calculate_mdc_capability_model(
    data: Any,
    target_value: float = 95.0,
    is_higher_better: bool = True,
    baseline_periods: Optional[int] = None
) -> Dict[str, Any]:
    """Calculates NHS 'Making Data Count' Variation and Assurance icons alongside Cp / Cpk capability."""
    records = data.to_dict("records") if HAS_PANDAS and isinstance(data, pd.DataFrame) else data
    k = len(records)
    values = [float(r["compliance_pct"]) for r in records]
    
    base_k = baseline_periods if baseline_periods and baseline_periods < k else k
    base_vals = values[:base_k]
    
    x_bar = _mean(base_vals)
    mr = [abs(base_vals[i] - base_vals[i - 1]) for i in range(1, len(base_vals))]
    mr_bar = _mean(mr) if mr else 1.0
    sigma_hat = mr_bar / 1.128
    
    ucl = x_bar + 2.66 * mr_bar
    lcl = max(0.0, x_bar - 2.66 * mr_bar)
    
    # 1. Process Capability Calculations (Cp, Cpk)
    # Directional Cpk:
    if is_higher_better:
        # LSL = target_value
        cpk = (x_bar - target_value) / (3.0 * sigma_hat) if sigma_hat > 0 else 0.0
        cp = (ucl - lcl) / (6.0 * sigma_hat) if sigma_hat > 0 else 1.0
    else:
        # USL = target_value
        cpk = (target_value - x_bar) / (3.0 * sigma_hat) if sigma_hat > 0 else 0.0
        cp = (ucl - lcl) / (6.0 * sigma_hat) if sigma_hat > 0 else 1.0
        
    # 2. NHS Assurance Icon Determination
    if is_higher_better:
        if lcl >= target_value:
            assurance_icon = "🔵 Consistently Passing Target (Process Capable)"
            assurance_code = "PASS"
            assurance_desc = "The lower process limit is above the statutory standard. The system is designed to reliably pass without heroic effort."
        elif ucl < target_value:
            assurance_icon = "🟠 Consistently Failing Target (Process Incapable)"
            assurance_code = "FAIL"
            assurance_desc = "The upper process limit is below the statutory standard. The system is fundamentally incapable of meeting the target without structural redesign."
        else:
            assurance_icon = "🟡 Hit-and-Miss / Flapping (Target between Limits)"
            assurance_code = "FLAPPING"
            assurance_desc = "The target lies between the control limits. Passing or failing in any period is entirely driven by random common cause noise."
    else:
        if ucl <= target_value:
            assurance_icon = "🔵 Consistently Passing Target (Process Capable)"
            assurance_code = "PASS"
            assurance_desc = "The upper process limit is below the statutory maximum. The system reliably satisfies safety thresholds."
        elif lcl > target_value:
            assurance_icon = "🟠 Consistently Failing Target (Process Incapable)"
            assurance_code = "FAIL"
            assurance_desc = "The lower process limit is above the statutory threshold. Structural redesign is required."
        else:
            assurance_icon = "🟡 Hit-and-Miss / Flapping (Target between Limits)"
            assurance_code = "FLAPPING"
            assurance_desc = "Target lies between limits. Performance fluctuates across target purely due to process noise."
            
    # 3. NHS Variation Icon (evaluating last points)
    last_pts = values[-6:] if k >= 6 else values
    pts_above = sum(1 for v in last_pts if v > x_bar)
    pts_below = sum(1 for v in last_pts if v < x_bar)
    
    if is_higher_better:
        if pts_above >= 6:
            variation_icon = "🔷 Special Cause of Improvement (Favorable)"
            variation_code = "IMPROVEMENT"
        elif pts_below >= 6 or values[-1] < lcl:
            variation_icon = "🔶 Special Cause of Concern (Deterioration)"
            variation_code = "CONCERN"
        else:
            variation_icon = "🔘 Common Cause Variation (Process Stable)"
            variation_code = "COMMON_CAUSE"
    else:
        if pts_below >= 6 or values[-1] < lcl:
            variation_icon = "🔷 Special Cause of Improvement (Favorable)"
            variation_code = "IMPROVEMENT"
        elif pts_above >= 6 or values[-1] > ucl:
            variation_icon = "🔶 Special Cause of Concern (Deterioration)"
            variation_code = "CONCERN"
        else:
            variation_icon = "🔘 Common Cause Variation (Process Stable)"
            variation_code = "COMMON_CAUSE"
            
    return {
        "x_bar": round(float(x_bar), 2),
        "mr_bar": round(float(mr_bar), 2),
        "sigma_hat": round(float(sigma_hat), 2),
        "ucl": round(float(ucl), 2),
        "lcl": round(float(lcl), 2),
        "target_value": target_value,
        "is_higher_better": is_higher_better,
        "cp": round(float(cp), 3),
        "cpk": round(float(cpk), 3),
        "assurance_icon": assurance_icon,
        "assurance_code": assurance_code,
        "assurance_desc": assurance_desc,
        "variation_icon": variation_icon,
        "variation_code": variation_code
    }

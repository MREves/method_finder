"""Statistical Fundamentals Calculation & Practice Data Engine.

Provides exact biostatistical formulations, Tukey outlier detection, Central Limit
Theorem simulations, and robust confidence interval derivations across:
1. Measurement Scales & Data Types
2. Measures of Central Tendency (Mean, Median, Mode, Trimmed Mean)
3. Measures of Dispersion & Spread (Variance, SD, Quartiles, IQR, Tukey's Fences)
4. Standard Deviation (SD) vs. Standard Error of the Mean (SEM)
5. Distribution Shapes, Skewness & Central Limit Theorem (CLT)
6. Confidence Intervals (Continuous Mean: z vs t; Binary Proportion: Wald vs Wilson vs Agresti-Coull)
"""

import io
import csv
import math
import random
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False
    np = None
    pd = None
    stats = None


# =====================================================================
# Pure Python Statistical Helpers (Fallback when SciPy/NumPy missing)
# =====================================================================

def _normal_cdf(z: float) -> float:
    """Cumulative distribution function of standard normal distribution Phi(z)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Quantile function (inverse CDF) of standard normal distribution.
    
    Uses high-precision Acklam rational approximation (accurate to 1e-9).
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if HAS_SCIENTIFIC_STACK and stats is not None and hasattr(stats, "norm"):
        return float(stats.norm.ppf(p))

    # Acklam's algorithm
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


# Exact two-tailed 95% Student's t critical values (p = 0.975) for small sample sizes
_T_975_TABLE: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042
}


def _t_ppf(p: float, df: int) -> float:
    """Quantile function for Student's t distribution with degrees of freedom df."""
    if df < 1:
        return 1.95996
    if HAS_SCIENTIFIC_STACK and stats is not None and hasattr(stats, "t"):
        return float(stats.t.ppf(p, df=df))
    
    # Exact lookup for common two-sided 95% critical value
    if abs(p - 0.975) < 0.001 and df in _T_975_TABLE:
        return _T_975_TABLE[df]

    # Cornish-Fisher expansion for Student's t quantile
    z = _normal_ppf(p)
    term1 = (z**3 + z) / (4.0 * df)
    term2 = (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df * df)
    return z + term1 + term2


def _percentile(sorted_vals: List[float], p: float) -> float:
    """Linear interpolation percentile (equivalent to NumPy default method='linear' and Excel PERCENTILE.EXC/INC)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = (p / 100.0) * (n - 1)
    i = int(idx)
    frac = idx - i
    if i >= n - 1:
        return sorted_vals[-1]
    return sorted_vals[i] + frac * (sorted_vals[i + 1] - sorted_vals[i])


class FundamentalsDataFrame(list):
    """List of dict records mimicking a DataFrame when pandas is not installed."""
    def head(self, n: int = 5):
        return self[:n]

    def to_csv(self, index: bool = False) -> str:
        if not self:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(self[0].keys()))
        writer.writeheader()
        writer.writerows(self)
        return output.getvalue()

    def __getitem__(self, item):
        if isinstance(item, str):
            return [row.get(item) for row in self]
        return super().__getitem__(item)


# =====================================================================
# 1. Central Tendency Calculations
# =====================================================================

def calculate_central_tendency(values: List[float], trim_percent: float = 0.10) -> Dict[str, Any]:
    """Calculates mean, median, mode, and trimmed mean with outlier shift evaluation."""
    clean_vals = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    n = len(clean_vals)
    if n == 0:
        return {
            "n": 0, "mean": 0.0, "median": 0.0, "mode": 0.0,
            "trimmed_mean": 0.0, "trim_percent": trim_percent * 100,
            "std_dev": 0.0, "relative_shift": 0.0, "is_noticeably_skewed": False
        }

    mean_val = sum(clean_vals) / n
    sorted_vals = sorted(clean_vals)
    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    # Mode calculation
    counts: Dict[float, int] = {}
    for v in clean_vals:
        rounded_v = round(v, 2)
        counts[rounded_v] = counts.get(rounded_v, 0) + 1
    max_freq = max(counts.values()) if counts else 0
    modes = [k for k, count in counts.items() if count == max_freq]
    mode_val = float(modes[0]) if modes else median_val

    # Trimmed Mean
    k = int(round(n * trim_percent))
    if k == 0 and trim_percent > 0 and n >= 5:
        k = 1
    if 2 * k < n:
        trimmed_arr = sorted_vals[k:n - k]
        trimmed_val = sum(trimmed_arr) / len(trimmed_arr)
    else:
        trimmed_val = median_val

    # Standard deviation (ddof=1)
    if n > 1:
        var_val = sum((x - mean_val) ** 2 for x in clean_vals) / (n - 1)
        std_val = math.sqrt(var_val)
    else:
        std_val = 0.0

    # Mean vs Median divergence (skew pull metric)
    relative_shift = abs(mean_val - median_val) / std_val if std_val > 0 else 0.0

    return {
        "n": n,
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "trimmed_mean": trimmed_val,
        "trim_percent": trim_percent * 100,
        "std_dev": std_val,
        "relative_shift": relative_shift,
        "is_noticeably_skewed": relative_shift > 0.20
    }


# =====================================================================
# 2. Dispersion, Quartiles & Tukey's Fences
# =====================================================================

def calculate_dispersion_and_quartiles(values: List[float]) -> Dict[str, Any]:
    """Calculates variance, standard deviation, quartiles, IQR, and Tukey's outlier fences."""
    clean_vals = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    n = len(clean_vals)
    if n < 2:
        val = clean_vals[0] if n == 1 else 0.0
        return {
            "n": n, "variance": 0.0, "std_dev": 0.0, "min": val, "max": val,
            "range": 0.0, "q1": val, "median": val, "q3": val, "iqr": 0.0,
            "tukey_inner_lower": val, "tukey_inner_upper": val,
            "tukey_outer_lower": val, "tukey_outer_upper": val,
            "mild_outliers": [], "extreme_outliers": [],
            "total_outliers_count": 0
        }

    mean_val = sum(clean_vals) / n
    variance_val = sum((x - mean_val) ** 2 for x in clean_vals) / (n - 1)
    std_val = math.sqrt(variance_val)
    min_val = min(clean_vals)
    max_val = max(clean_vals)
    range_val = max_val - min_val

    sorted_vals = sorted(clean_vals)
    if HAS_SCIENTIFIC_STACK and np is not None:
        arr = np.array(clean_vals, dtype=float)
        q1 = float(np.percentile(arr, 25))
        median_val = float(np.median(arr))
        q3 = float(np.percentile(arr, 75))
    else:
        q1 = _percentile(sorted_vals, 25.0)
        median_val = _percentile(sorted_vals, 50.0)
        q3 = _percentile(sorted_vals, 75.0)

    iqr = q3 - q1

    # Tukey's Fences
    inner_lower = q1 - 1.5 * iqr
    inner_upper = q3 + 1.5 * iqr
    outer_lower = q1 - 3.0 * iqr
    outer_upper = q3 + 3.0 * iqr

    mild_outliers = [float(x) for x in clean_vals if (outer_lower <= x < inner_lower) or (inner_upper < x <= outer_upper)]
    extreme_outliers = [float(x) for x in clean_vals if (x < outer_lower) or (x > outer_upper)]

    return {
        "n": n,
        "variance": variance_val,
        "std_dev": std_val,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "q1": q1,
        "median": median_val,
        "q3": q3,
        "iqr": iqr,
        "tukey_inner_lower": inner_lower,
        "tukey_inner_upper": inner_upper,
        "tukey_outer_lower": outer_lower,
        "tukey_outer_upper": outer_upper,
        "mild_outliers": mild_outliers,
        "extreme_outliers": extreme_outliers,
        "total_outliers_count": len(mild_outliers) + len(extreme_outliers)
    }


# =====================================================================
# 3. Standard Deviation (SD) vs. Standard Error (SEM)
# =====================================================================

def calculate_sd_vs_sem(values: List[float]) -> Dict[str, Any]:
    """Calculates biological spread (SD) vs. estimation precision (SEM)."""
    clean_vals = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    n = len(clean_vals)
    if n < 2:
        return {
            "n": n, "mean": 0.0, "std_dev": 0.0, "sem": 0.0,
            "ratio": 1.0, "ci_95_margin": 0.0, "sd_to_sem_ratio": 1.0,
            "sem_as_pct_of_sd": 100.0
        }

    mean_val = sum(clean_vals) / n
    sd_val = math.sqrt(sum((x - mean_val) ** 2 for x in clean_vals) / (n - 1))
    sem_val = sd_val / math.sqrt(n)
    ci_95_margin = 1.96 * sem_val

    return {
        "n": n,
        "mean": mean_val,
        "std_dev": sd_val,
        "sem": sem_val,
        "ci_95_margin": ci_95_margin,
        "sd_to_sem_ratio": sd_val / sem_val if sem_val > 0 else 1.0,
        "sem_as_pct_of_sd": (sem_val / sd_val * 100.0) if sd_val > 0 else 100.0
    }


# =====================================================================
# 4. Distribution Shapes, Skewness & Central Limit Theorem (CLT)
# =====================================================================

def calculate_distribution_shape_and_clt(
    values: List[float],
    n_sim_samples: int = 400,
    sim_sample_size: int = 30,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Calculates skewness, kurtosis, and simulates the Central Limit Theorem
    sampling distribution of the sample mean.
    """
    clean_vals = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    n = len(clean_vals)
    if n < 3:
        return {
            "n": n, "skewness": 0.0, "kurtosis": 0.0, "shape_label": "Insufficient Data",
            "sim_means": [], "sim_mean_of_means": 0.0, "sim_std_of_means": 0.0, "theoretical_sem": 0.0
        }

    mean_val = sum(clean_vals) / n
    sd_val = math.sqrt(sum((x - mean_val) ** 2 for x in clean_vals) / (n - 1))

    # Fisher-Pearson standardized skewness and excess kurtosis
    if HAS_SCIENTIFIC_STACK and stats is not None and hasattr(stats, "skew"):
        skew_val = float(stats.skew(clean_vals, bias=False))
        kurt_val = float(stats.kurtosis(clean_vals, bias=False))
    else:
        if sd_val > 0:
            skew_val = (n / ((n - 1) * (n - 2))) * sum(((x - mean_val) / sd_val) ** 3 for x in clean_vals)
            if n >= 4:
                kurt_val = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(((x - mean_val) / sd_val) ** 4 for x in clean_vals) - (3.0 * (n - 1) ** 2) / ((n - 2) * (n - 3))
            else:
                kurt_val = 0.0
        else:
            skew_val = 0.0
            kurt_val = 0.0

    if abs(skew_val) <= 0.5:
        shape_label = "Symmetric / Approximately Normal"
    elif skew_val > 1.0:
        shape_label = "Highly Right-Skewed (Positive)"
    elif skew_val > 0.5:
        shape_label = "Moderately Right-Skewed"
    elif skew_val < -1.0:
        shape_label = "Highly Left-Skewed (Negative)"
    else:
        shape_label = "Moderately Left-Skewed"

    # Central Limit Theorem Simulation
    effective_k = min(sim_sample_size, n)
    sim_means: List[float] = []

    if HAS_SCIENTIFIC_STACK and np is not None:
        arr = np.array(clean_vals, dtype=float)
        rng = np.random.RandomState(random_seed)
        for _ in range(n_sim_samples):
            sample = rng.choice(arr, size=effective_k, replace=True)
            sim_means.append(float(np.mean(sample)))
    else:
        rng_py = random.Random(random_seed)
        for _ in range(n_sim_samples):
            sample = rng_py.choices(clean_vals, k=effective_k)
            sim_means.append(sum(sample) / float(effective_k))

    sim_mean_of_means = sum(sim_means) / len(sim_means) if sim_means else 0.0
    sim_std_of_means = math.sqrt(sum((x - sim_mean_of_means) ** 2 for x in sim_means) / (len(sim_means) - 1)) if len(sim_means) > 1 else 0.0
    theoretical_sem = sd_val / math.sqrt(effective_k) if effective_k > 0 else 0.0

    return {
        "n": n,
        "mean": mean_val,
        "std_dev": sd_val,
        "skewness": skew_val,
        "excess_kurtosis": kurt_val,
        "shape_label": shape_label,
        "sim_means": sim_means,
        "sim_sample_size": effective_k,
        "sim_mean_of_means": sim_mean_of_means,
        "sim_std_of_means": sim_std_of_means,
        "theoretical_sem": theoretical_sem
    }


# =====================================================================
# 5. Confidence Intervals: Mean (z vs. t)
# =====================================================================

def calculate_confidence_interval_mean(values: List[float], confidence: float = 0.95) -> Dict[str, Any]:
    """Calculates confidence intervals for a continuous mean using both
    normal z-approximation and Student's t-distribution.
    """
    clean_vals = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    n = len(clean_vals)
    if n < 2:
        val = clean_vals[0] if n == 1 else 0.0
        return {
            "n": n, "mean": val, "std_dev": 0.0, "sem": 0.0, "df": 0,
            "z_crit": 1.96, "t_crit": 1.96,
            "z_lower": val, "z_upper": val, "z_margin": 0.0,
            "t_lower": val, "t_upper": val, "t_margin": 0.0,
            "t_penalty_pct": 0.0, "is_small_sample": True
        }

    mean_val = sum(clean_vals) / n
    sd_val = math.sqrt(sum((x - mean_val) ** 2 for x in clean_vals) / (n - 1))
    sem_val = sd_val / math.sqrt(n)
    df = n - 1
    alpha = 1.0 - confidence

    # Normal z critical value
    z_crit = _normal_ppf(1.0 - alpha / 2.0)
    z_margin = z_crit * sem_val
    z_lower = mean_val - z_margin
    z_upper = mean_val + z_margin

    # Student's t critical value
    t_crit = _t_ppf(1.0 - alpha / 2.0, df=df)
    t_margin = t_crit * sem_val
    t_lower = mean_val - t_margin
    t_upper = mean_val + t_margin

    t_penalty_pct = ((t_margin - z_margin) / z_margin * 100.0) if z_margin > 0 else 0.0

    return {
        "n": n,
        "mean": mean_val,
        "std_dev": sd_val,
        "sem": sem_val,
        "df": df,
        "confidence": confidence * 100.0,
        "z_crit": z_crit,
        "t_crit": t_crit,
        "z_lower": z_lower,
        "z_upper": z_upper,
        "z_margin": z_margin,
        "t_lower": t_lower,
        "t_upper": t_upper,
        "t_margin": t_margin,
        "t_penalty_pct": t_penalty_pct,
        "is_small_sample": n < 30
    }


# =====================================================================
# 6. Confidence Intervals: Proportion (Wald vs. Wilson vs. Agresti-Coull)
# =====================================================================

def calculate_confidence_interval_proportion(successes: int, trials: int, confidence: float = 0.95) -> Dict[str, Any]:
    """Calculates binomial proportion confidence intervals using:
    1. Wald Normal Approximation (often fails near 0 or 1).
    2. Wilson Score Interval (inverts score test; guaranteed [0, 1]).
    3. Agresti-Coull 'Plus-Four' Interval (practitioner friendly).
    """
    n = max(trials, 1)
    x = max(0, min(successes, n))
    p_hat = x / n
    alpha = 1.0 - confidence
    z_crit = _normal_ppf(1.0 - alpha / 2.0)
    z2 = z_crit * z_crit

    # 1. Wald Interval
    se_wald = math.sqrt(p_hat * (1.0 - p_hat) / n) if (0.0 < p_hat < 1.0) else 0.0
    wald_margin = z_crit * se_wald
    wald_lower = max(0.0, p_hat - wald_margin)
    wald_upper = min(1.0, p_hat + wald_margin)
    wald_valid = (n * p_hat >= 5.0) and (n * (1.0 - p_hat) >= 5.0)

    # 2. Wilson Score Interval
    denom = 1.0 + z2 / n
    center_adj = (p_hat + z2 / (2.0 * n)) / denom
    inner_rad = (p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n))
    wilson_margin = (z_crit * math.sqrt(max(0.0, inner_rad))) / denom
    wilson_lower = max(0.0, center_adj - wilson_margin)
    wilson_upper = min(1.0, center_adj + wilson_margin)

    # 3. Agresti-Coull 'Plus-Four' Interval
    n_tilde = n + z2
    p_tilde = (x + z2 / 2.0) / n_tilde
    se_ac = math.sqrt(p_tilde * (1.0 - p_tilde) / n_tilde)
    ac_margin = z_crit * se_ac
    ac_lower = max(0.0, p_tilde - ac_margin)
    ac_upper = min(1.0, p_tilde + ac_margin)

    return {
        "n": n,
        "x": x,
        "p_hat": p_hat,
        "p_hat_pct": p_hat * 100.0,
        "confidence": confidence * 100.0,
        "z_crit": z_crit,
        "wald_lower": wald_lower,
        "wald_upper": wald_upper,
        "wald_margin": wald_margin,
        "wald_valid": wald_valid,
        "wilson_lower": wilson_lower,
        "wilson_upper": wilson_upper,
        "wilson_margin": wilson_margin,
        "agresti_coull_lower": ac_lower,
        "agresti_coull_upper": ac_upper,
        "agresti_coull_margin": ac_margin
    }


# =====================================================================
# 7. Practice Dataset Generators (CSV Ready)
# =====================================================================

def _wrap_dataset(records: List[Dict[str, Any]]) -> Tuple[Any, str]:
    """Wraps record dicts into DataFrame (or FundamentalsDataFrame) and CSV string."""
    if HAS_SCIENTIFIC_STACK and pd is not None:
        df = pd.DataFrame(records)
        csv_str = df.to_csv(index=False)
    else:
        df = FundamentalsDataFrame(records)
        csv_str = df.to_csv(index=False)
    return df, csv_str


def generate_scales_practice_data() -> Tuple[Any, str]:
    """Generates a clinical patient intake registry exhibiting Nominal, Ordinal,
    Interval, and Ratio measurement scales.
    """
    records = [
        {"Patient_ID": f"PT-{100+i:03d}",
         "Gender_Nominal": ["Female", "Male", "Non-Binary"][i % 3],
         "Blood_Group_Nominal": ["O+", "A+", "B+", "AB-"][i % 4],
         "NYHA_Class_Ordinal": ["Class I (Mild)", "Class II (Moderate)", "Class III (Severe)", "Class IV (End-Stage)"][i % 4],
         "Pain_Likert_Ordinal": (i % 5) + 1,
         "Body_Temp_Celsius_Interval": round(36.5 + (i % 7) * 0.3, 1),
         "Calendar_Year_Interval": 2024,
         "Systolic_BP_mmHg_Ratio": 115 + (i * 3) % 45,
         "Serum_Creatinine_umol_L_Ratio": round(65.0 + (i * 5.2) % 60, 1),
         "Hospital_Stay_Days_Ratio": (i % 12) + 1}
        for i in range(40)
    ]
    return _wrap_dataset(records)


def generate_central_tendency_practice_data(inject_extreme: bool = True) -> Tuple[Any, str]:
    """Generates hospital Length-of-Stay (LOS) data with severe right-skewness and extreme outliers."""
    if HAS_SCIENTIFIC_STACK and np is not None:
        rng = np.random.RandomState(42)
        base_stays = list(np.round(rng.lognormal(mean=1.2, sigma=0.55, size=48) + 1.0, 1))
    else:
        rng_py = random.Random(42)
        base_stays = [round(rng_py.lognormvariate(1.2, 0.55) + 1.0, 1) for _ in range(48)]

    if inject_extreme:
        base_stays.extend([62.0, 145.0])  # Prolonged ICU outliers

    records = [
        {
            "Admission_ID": f"ADM-{5000+i:04d}",
            "Department": "General Surgery" if i % 2 == 0 else "Acute Medicine",
            "Length_of_Stay_Days": float(base_stays[i]),
            "Is_Prolonged_Stay": bool(base_stays[i] > 14.0)
        }
        for i in range(len(base_stays))
    ]
    return _wrap_dataset(records)


def generate_dispersion_practice_data() -> Tuple[Any, str]:
    """Generates serum cholesterol assay values with clinical variance and Tukey outliers."""
    if HAS_SCIENTIFIC_STACK and np is not None:
        rng = np.random.RandomState(101)
        base_chol = list(np.round(rng.normal(loc=5.10, scale=0.85, size=46), 2))
    else:
        rng_py = random.Random(101)
        base_chol = [round(rng_py.gauss(5.10, 0.85), 2) for _ in range(46)]

    # Add mild and extreme outliers
    base_chol.extend([8.95, 9.40, 2.10, 1.85])

    records = [
        {
            "Sample_ID": f"LAB-{200+i:03d}",
            "Patient_Age": int(45 + (i * 2) % 35),
            "Total_Cholesterol_mmol_L": float(base_chol[i]),
            "Clinical_Risk_Tier": "Desirable" if base_chol[i] < 5.0 else ("Borderline" if base_chol[i] < 6.5 else "High")
        }
        for i in range(len(base_chol))
    ]
    return _wrap_dataset(records)


def generate_sd_sem_practice_data() -> Tuple[Any, str]:
    """Generates clinical trial baseline demographic data illustrating SD vs. SEM."""
    if HAS_SCIENTIFIC_STACK and np is not None:
        rng = np.random.RandomState(77)
        sbp_vals = list(np.round(rng.normal(loc=138.5, scale=14.2, size=60), 1))
    else:
        rng_py = random.Random(77)
        sbp_vals = [round(rng_py.gauss(138.5, 14.2), 1) for _ in range(60)]

    records = [
        {
            "Participant_ID": f"SUBJ-{1000+i:04d}",
            "Study_Arm": "Active Treatment" if i % 2 == 0 else "Placebo Comparator",
            "Baseline_SBP_mmHg": float(sbp_vals[i]),
            "Heart_Rate_bpm": int(65 + (i * 3) % 30)
        }
        for i in range(len(sbp_vals))
    ]
    return _wrap_dataset(records)


def generate_clt_practice_data() -> Tuple[Any, str]:
    """Generates Emergency Department triage waiting minutes exhibiting heavy exponential skewness."""
    if HAS_SCIENTIFIC_STACK and np is not None:
        rng = np.random.RandomState(88)
        wait_mins = list(np.round(rng.exponential(scale=45.0, size=80) + 12.0, 1))
    else:
        rng_py = random.Random(88)
        wait_mins = [round(rng_py.expovariate(1.0 / 45.0) + 12.0, 1) for _ in range(80)]

    records = [
        {
            "Triage_Ticket_ID": f"ED-{8000+i:04d}",
            "Triage_Category": ["Resuscitation (Cat 1)", "Emergency (Cat 2)", "Urgent (Cat 3)", "Standard (Cat 4)"][i % 4],
            "Wait_Time_Minutes": float(wait_mins[i]),
            "Breached_4_Hour_Target": bool(wait_mins[i] > 240.0)
        }
        for i in range(len(wait_mins))
    ]
    return _wrap_dataset(records)


def generate_confidence_intervals_practice_data() -> Tuple[Any, str]:
    """Generates surgical recovery days and binary post-op complications data."""
    if HAS_SCIENTIFIC_STACK and np is not None:
        rng = np.random.RandomState(99)
        recovery = list(np.round(rng.normal(loc=6.8, scale=1.4, size=50), 1))
    else:
        rng_py = random.Random(99)
        recovery = [round(rng_py.gauss(6.8, 1.4), 1) for _ in range(50)]

    # Rare complication: 3 out of 50
    complications = [1 if i in (7, 23, 41) else 0 for i in range(50)]

    records = [
        {
            "Case_ID": f"SURG-{300+i:03d}",
            "Procedure_Type": "Laparoscopic Cholecystectomy",
            "Recovery_Days": float(recovery[i]),
            "Major_Complication_30d": int(complications[i])
        }
        for i in range(50)
    ]
    return _wrap_dataset(records)


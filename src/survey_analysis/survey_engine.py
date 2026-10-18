from __future__ import annotations

"""Survey & Questionnaire Analysis Calculation Engines & Synthetic Generators.

Provides pure mathematical implementations for Likert scale metrics,
Cronbach's Alpha, McDonald's Omega, Cohen's Kappa, Factor Analysis,
Finite Population Sample Sizing, and Post-Stratification Weighting.
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.metrics import cohen_kappa_score
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False
    np = None
    pd = None
    stats = None
    PCA = None
    FactorAnalysis = None
    cohen_kappa_score = None


# =====================================================================
# 1. Likert Scale Analysis Engine
# =====================================================================

def calculate_likert_metrics(df_items: pd.DataFrame) -> Dict[str, Any]:
    """Calculates comprehensive Likert descriptive metrics for individual items

    and composite scale scores.
    """
    item_results = []
    
    for col in df_items.columns:
        series = df_items[col].dropna()
        n_valid = len(series)
        if n_valid == 0:
            continue
            
        counts = series.value_counts()
        
        # Proportions on 1-5 Likert scale
        pct_1 = (counts.get(1, 0) / n_valid) * 100.0
        pct_2 = (counts.get(2, 0) / n_valid) * 100.0
        pct_3 = (counts.get(3, 0) / n_valid) * 100.0
        pct_4 = (counts.get(4, 0) / n_valid) * 100.0
        pct_5 = (counts.get(5, 0) / n_valid) * 100.0
        
        top_box = pct_5
        top_2_box = pct_4 + pct_5
        bottom_box = pct_1
        bottom_2_box = pct_1 + pct_2
        neutral_box = pct_3
        net_agreement = top_2_box - bottom_2_box
        
        median_val = float(series.median())
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr_val = q75 - q25
        mean_val = float(series.mean())
        std_val = float(series.std(ddof=1)) if n_valid > 1 else 0.0
        
        item_results.append({
            "Item": col,
            "N": n_valid,
            "Pct_Strongly_Disagree": round(pct_1, 1),
            "Pct_Disagree": round(pct_2, 1),
            "Pct_Neutral": round(pct_3, 1),
            "Pct_Agree": round(pct_4, 1),
            "Pct_Strongly_Agree": round(pct_5, 1),
            "Top_Box_Pct": round(top_box, 1),
            "Top_2_Box_Pct": round(top_2_box, 1),
            "Bottom_2_Box_Pct": round(bottom_2_box, 1),
            "Net_Agreement_Pct": round(net_agreement, 1),
            "Median": round(median_val, 2),
            "Q1": round(q25, 2),
            "Q3": round(q75, 2),
            "IQR": round(iqr_val, 2),
            "Mean": round(mean_val, 2),
            "SD": round(std_val, 2)
        })
        
    df_summary = pd.DataFrame(item_results)
    
    # Composite Scale Analysis (Sum and Mean across items)
    composite_sums = df_items.sum(axis=1)
    composite_means = df_items.mean(axis=1)
    
    comp_mean = float(composite_sums.mean())
    comp_sd = float(composite_sums.std(ddof=1)) if len(composite_sums) > 1 else 0.0
    comp_median = float(composite_sums.median())
    comp_iqr = float(composite_sums.quantile(0.75) - composite_sums.quantile(0.25))
    comp_skew = float(stats.skew(composite_sums)) if len(composite_sums) > 2 else 0.0
    
    # Normality check of composite score (Shapiro-Wilk on sample if N <= 5000)
    shapiro_p = None
    if 3 <= len(composite_sums) <= 5000:
        _, shapiro_p = stats.shapiro(composite_sums)
        shapiro_p = round(float(shapiro_p), 4)
        
    return {
        "item_summary": df_summary,
        "composite": {
            "mean": round(comp_mean, 2),
            "sd": round(comp_sd, 2),
            "median": round(comp_median, 2),
            "iqr": round(comp_iqr, 2),
            "skewness": round(comp_skew, 3),
            "shapiro_p_value": shapiro_p,
            "is_approximately_normal": abs(comp_skew) < 1.0 and (shapiro_p is None or shapiro_p >= 0.05)
        }
    }


# =====================================================================
# 2. Cronbach's Alpha & Item-Total Reliability Engine
# =====================================================================

def calculate_cronbachs_alpha(df_items: pd.DataFrame) -> Dict[str, Any]:
    """Calculates Cronbach's Alpha, item variances, corrected item-total correlations,

    and the Alpha-If-Item-Deleted diagnostic table.
    """
    df_clean = df_items.dropna()
    n_samples, k_items = df_clean.shape
    
    if k_items < 2:
        raise ValueError("At least 2 questions are required to compute scale reliability.")
    if n_samples < 3:
        raise ValueError("At least 3 respondents are required for reliable variance calculation.")
        
    # Item variances (sample variance ddof=1)
    item_vars = df_clean.var(axis=0, ddof=1)
    sum_item_vars = float(item_vars.sum())
    
    # Total scale composite variance
    total_scores = df_clean.sum(axis=1)
    total_var = float(total_scores.var(ddof=1))
    
    if total_var == 0.0:
        alpha = 0.0
    else:
        alpha = (k_items / (k_items - 1.0)) * (1.0 - (sum_item_vars / total_var))
        
    # Standardized Cronbach's Alpha (based on correlation matrix)
    corr_matrix = df_clean.corr().values
    r_mean = float((corr_matrix.sum() - k_items) / (k_items * (k_items - 1)))
    alpha_standardized = (k_items * r_mean) / (1.0 + (k_items - 1.0) * r_mean)
    
    # 95% Confidence Interval for Alpha (Feldt, Woodruff, & Salih 1987)
    df1 = n_samples - 1
    df2 = (n_samples - 1) * (k_items - 1)
    f_lower = stats.f.ppf(0.025, df1, df2)
    f_upper = stats.f.ppf(0.975, df1, df2)
    ci_lower = 1.0 - (1.0 - alpha) * stats.f.ppf(0.975, df1, df2)
    ci_upper = 1.0 - (1.0 - alpha) * stats.f.ppf(0.025, df1, df2)
    ci_lower = max(0.0, float(ci_lower))
    ci_upper = min(1.0, float(ci_upper))
    
    # Item-Deleted Analysis
    item_deleted_records = []
    for col in df_clean.columns:
        sub_df = df_clean.drop(columns=[col])
        k_sub = sub_df.shape[1]
        sub_item_vars = float(sub_df.var(axis=0, ddof=1).sum())
        sub_total_scores = sub_df.sum(axis=1)
        sub_total_var = float(sub_total_scores.var(ddof=1))
        
        if sub_total_var == 0.0:
            alpha_del = 0.0
        else:
            alpha_del = (k_sub / (k_sub - 1.0)) * (1.0 - (sub_item_vars / sub_total_var))
            
        # Corrected Item-Total Correlation (Item vs Sum of remaining items)
        if sub_total_scores.std() > 0 and df_clean[col].std() > 0:
            r_item_rest = float(np.corrcoef(df_clean[col], sub_total_scores)[0, 1])
        else:
            r_item_rest = 0.0
            
        impact = "Retain"
        if alpha_del > alpha + 0.02:
            impact = "⚠️ Degrades Scale (Drop Candidate)"
        elif r_item_rest < 0.30:
            impact = "⚠️ Weak Correlation (< 0.30)"
            
        item_deleted_records.append({
            "Item": col,
            "Item_Mean": round(float(df_clean[col].mean()), 2),
            "Item_SD": round(float(df_clean[col].std(ddof=1)), 2),
            "Corrected_Item_Total_r": round(r_item_rest, 3),
            "Alpha_If_Item_Deleted": round(float(alpha_del), 3),
            "Recommendation": impact
        })
        
    df_item_analysis = pd.DataFrame(item_deleted_records)
    
    # Quality Benchmark Classification
    if alpha >= 0.90:
        rating_tier = "Excellent"
        rating_detail = "Clinical Standard (check redundancy if > 0.95)"
        rating = "Excellent (Clinical Standard; check for redundancy if > 0.95)"
        badge_color = "#15803d"
    elif alpha >= 0.80:
        rating_tier = "Good"
        rating_detail = "Recommended for research & policy decisions"
        rating = "Good (Recommended for research & policy decisions)"
        badge_color = "#16a34a"
    elif alpha >= 0.70:
        rating_tier = "Acceptable"
        rating_detail = "Standard academic research benchmark"
        rating = "Acceptable (Standard academic research benchmark)"
        badge_color = "#ca8a04"
    elif alpha >= 0.60:
        rating_tier = "Marginal / Questionable"
        rating_detail = "Exploratory surveys only"
        rating = "Marginal / Questionable (Exploratory surveys only)"
        badge_color = "#ea580c"
    else:
        rating_tier = "Unacceptable"
        rating_detail = "Scale cannot be reliably interpreted"
        rating = "Unacceptable (Scale cannot be reliably interpreted)"
        badge_color = "#dc2626"
        
    return {
        "alpha": round(float(alpha), 3),
        "alpha_standardized": round(float(alpha_standardized), 3),
        "ci_95": (round(ci_lower, 3), round(ci_upper, 3)),
        "rating": rating,
        "rating_tier": rating_tier,
        "rating_detail": rating_detail,
        "badge_color": badge_color,
        "k_items": k_items,
        "n_samples": n_samples,
        "total_variance": round(total_var, 3),
        "sum_item_variances": round(sum_item_vars, 3),
        "item_analysis": df_item_analysis,
        "correlation_matrix": df_clean.corr().round(3)
    }


# =====================================================================
# 3. McDonald's Omega Engine (Overcoming Tau-Equivalence)
# =====================================================================

def calculate_mcdonalds_omega(df_items: pd.DataFrame) -> Dict[str, Any]:
    """Calculates McDonald's Omega (total) using single-factor decomposition

    to account for unequal factor loadings (congeneric models).
    """
    df_clean = df_items.dropna()
    n_samples, k_items = df_clean.shape
    
    if k_items < 3:
        raise ValueError("At least 3 questions are required to compute McDonald's Omega.")
        
    # Standardize data
    std_df = (df_clean - df_clean.mean()) / df_clean.std(ddof=1)
    
    # Fit 1-factor model via Factor Analysis
    fa = FactorAnalysis(n_components=1, random_state=42, max_iter=1000)
    fa.fit(std_df)
    
    # Standardized loadings
    raw_loadings = fa.components_[0]
    loadings = np.abs(raw_loadings)
    
    # Bound loadings to avoid non-positive uniqueness
    loadings = np.clip(loadings, 0.05, 0.98)
    
    # Uniqueness (error variance) = 1 - lambda^2
    uniqueness = 1.0 - (loadings ** 2)
    
    sum_loadings = float(np.sum(loadings))
    sum_loadings_sq = sum_loadings ** 2
    sum_uniqueness = float(np.sum(uniqueness))
    
    omega_total = sum_loadings_sq / (sum_loadings_sq + sum_uniqueness)
    
    # Calculate Cronbach's Alpha for direct comparison
    alpha_res = calculate_cronbachs_alpha(df_clean)
    alpha_val = alpha_res["alpha"]
    
    discrepancy = round(omega_total - alpha_val, 3)
    
    # Table of loadings vs error variances
    loadings_records = []
    for col, l, u in zip(df_clean.columns, loadings, uniqueness):
        loadings_records.append({
            "Item": col,
            "Factor_Loading_λ": round(float(l), 3),
            "True_Variance_λ2": round(float(l**2), 3),
            "Uniqueness_Error_u": round(float(u), 3)
        })
        
    df_loadings = pd.DataFrame(loadings_records)
    
    # Variance of factor loadings (test of tau-equivalence)
    loadings_sd = float(np.std(loadings, ddof=1))
    tau_violation = loadings_sd > 0.12
    
    return {
        "omega_total": round(float(omega_total), 3),
        "cronbach_alpha": alpha_val,
        "discrepancy_omega_minus_alpha": discrepancy,
        "tau_equivalence_violated": tau_violation,
        "loadings_dispersion_sd": round(loadings_sd, 3),
        "sum_loadings_squared": round(sum_loadings_sq, 3),
        "sum_error_variances": round(sum_uniqueness, 3),
        "loadings_table": df_loadings
    }


# =====================================================================
# 4. Cohen's Kappa & Weighted Kappa Engine
# =====================================================================

def calculate_cohens_kappa(
    rater1: List[Any],
    rater2: List[Any],
    categories: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Calculates unweighted Cohen's Kappa, Linear Weighted Kappa, and

    Quadratic Weighted Kappa with detailed confusion matrices.
    """
    r1 = np.array(rater1)
    r2 = np.array(rater2)
    
    if len(r1) != len(r2):
        raise ValueError("Rater 1 and Rater 2 must evaluate the exact same number of subjects.")
    if len(r1) < 5:
        raise ValueError("At least 5 paired ratings are required.")
        
    # Unique categories
    if categories is None:
        cats = sorted(list(set(r1).union(set(r2))))
    else:
        cats = categories
        
    n_cats = len(cats)
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    
    # Build Confusion Matrix (Rows: Rater 1, Columns: Rater 2)
    conf_matrix = np.zeros((n_cats, n_cats), dtype=int)
    for v1, v2 in zip(r1, r2):
        if v1 in cat_to_idx and v2 in cat_to_idx:
            conf_matrix[cat_to_idx[v1], cat_to_idx[v2]] += 1
            
    n_total = int(conf_matrix.sum())
    p_matrix = conf_matrix / n_total
    
    # Observed agreement po
    po = float(np.trace(p_matrix))
    
    # Expected agreement pe
    row_marginals = p_matrix.sum(axis=1)
    col_marginals = p_matrix.sum(axis=0)
    pe = float(np.sum(row_marginals * col_marginals))
    
    # Unweighted Kappa
    if pe == 1.0:
        kappa_unweighted = 1.0
    else:
        kappa_unweighted = (po - pe) / (1.0 - pe)
        
    # Standard Error and 95% CI for unweighted Kappa
    se_kappa = math.sqrt((po * (1.0 - po)) / (n_total * (1.0 - pe)**2)) if n_total > 0 and pe < 1.0 else 0.0
    z_score = kappa_unweighted / se_kappa if se_kappa > 0 else 0.0
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))
    ci_lower = max(-1.0, kappa_unweighted - 1.96 * se_kappa)
    ci_upper = min(1.0, kappa_unweighted + 1.96 * se_kappa)
    
    # Quadratic Weighted Kappa
    w_matrix_quad = np.zeros((n_cats, n_cats))
    w_matrix_linear = np.zeros((n_cats, n_cats))
    
    for i in range(n_cats):
        for j in range(n_cats):
            if n_cats > 1:
                w_matrix_quad[i, j] = 1.0 - ((i - j) ** 2) / ((n_cats - 1) ** 2)
                w_matrix_linear[i, j] = 1.0 - abs(i - j) / (n_cats - 1)
            else:
                w_matrix_quad[i, j] = 1.0
                w_matrix_linear[i, j] = 1.0
                
    po_quad = float(np.sum(w_matrix_quad * p_matrix))
    pe_quad = float(np.sum(w_matrix_quad * np.outer(row_marginals, col_marginals)))
    kappa_quadratic = (po_quad - pe_quad) / (1.0 - pe_quad) if pe_quad < 1.0 else 1.0
    
    po_linear = float(np.sum(w_matrix_linear * p_matrix))
    pe_linear = float(np.sum(w_matrix_linear * np.outer(row_marginals, col_marginals)))
    kappa_linear = (po_linear - pe_linear) / (1.0 - pe_linear) if pe_linear < 1.0 else 1.0
    
    # Landis & Koch Interpretation
    if kappa_quadratic >= 0.81:
        benchmark = "Almost Perfect Agreement"
        badge_color = "#15803d"
    elif kappa_quadratic >= 0.61:
        benchmark = "Substantial Agreement (Standard for Clinical Coding)"
        badge_color = "#16a34a"
    elif kappa_quadratic >= 0.41:
        benchmark = "Moderate Agreement"
        badge_color = "#ca8a04"
    elif kappa_quadratic >= 0.21:
        benchmark = "Fair Agreement"
        badge_color = "#ea580c"
    elif kappa_quadratic >= 0.0:
        benchmark = "Slight Agreement"
        badge_color = "#dc2626"
    else:
        benchmark = "Poor Agreement (Worse than Chance)"
        badge_color = "#991b1b"
        
    df_conf = pd.DataFrame(
        conf_matrix,
        index=[f"Rater 1: {c}" for c in cats],
        columns=[f"Rater 2: {c}" for c in cats]
    )
    
    return {
        "n_subjects": n_total,
        "n_categories": n_cats,
        "categories": [str(c) for c in cats],
        "observed_agreement_pct": round(po * 100.0, 1),
        "expected_agreement_pct": round(pe * 100.0, 1),
        "kappa_unweighted": round(float(kappa_unweighted), 3),
        "se_kappa": round(float(se_kappa), 3),
        "ci_95": (round(float(ci_lower), 3), round(float(ci_upper), 3)),
        "p_value": round(float(p_val), 4),
        "kappa_linear_weighted": round(float(kappa_linear), 3),
        "kappa_quadratic_weighted": round(float(kappa_quadratic), 3),
        "benchmark": benchmark,
        "badge_color": badge_color,
        "confusion_matrix": df_conf
    }


def calculate_weighted_kappa(rater1: List[Any], rater2: List[Any], weight_type: str = "quadratic") -> float:
    """Convenience helper returning weighted kappa scalar."""
    res = calculate_cohens_kappa(rater1, rater2)
    return res["kappa_quadratic_weighted"] if weight_type == "quadratic" else res["kappa_linear_weighted"]


def calculate_fleiss_kappa(rating_matrix: np.ndarray) -> Dict[str, Any]:
    """Calculates Fleiss' Kappa for m >= 3 raters scoring N subjects into C categories.

    rating_matrix shape: (N subjects, C categories) where entry (i, j) is the
    count of raters who assigned subject i to category j.
    """
    n_subjects, n_categories = rating_matrix.shape
    n_raters = int(rating_matrix[0].sum())
    
    # Category marginal proportions (pj)
    total_ratings = n_subjects * n_raters
    pj = rating_matrix.sum(axis=0) / total_ratings
    
    # Expected agreement pe
    pe = float(np.sum(pj ** 2))
    
    # Observed agreement per subject (Pi)
    pi_subjects = (np.sum(rating_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    po = float(np.mean(pi_subjects))
    
    if pe == 1.0:
        fleiss_k = 1.0
    else:
        fleiss_k = (po - pe) / (1.0 - pe)
        
    return {
        "fleiss_kappa": round(float(fleiss_k), 3),
        "n_subjects": n_subjects,
        "n_categories": n_categories,
        "n_raters": n_raters,
        "po": round(po, 3),
        "pe": round(pe, 3)
    }


# =====================================================================
# 5. Exploratory Factor Analysis & Scree Engine
# =====================================================================

def calculate_factor_analysis(df_items: pd.DataFrame, max_factors: int = 5) -> Dict[str, Any]:
    """Performs correlation matrix evaluation, KMO sampling adequacy,

    Bartlett's test of sphericity, Scree eigenvalue extraction, and
    Varimax-rotated factor loadings.
    """
    df_clean = df_items.dropna()
    n_samples, n_items = df_clean.shape
    
    if n_items < 3:
        raise ValueError("At least 3 questions are required for factor analysis.")
    if n_samples < n_items:
        raise ValueError("Sample size N must exceed the number of items.")
        
    corr_matrix = df_clean.corr().values
    
    # 1. Kaiser-Meyer-Olkin (KMO) Sampling Adequacy
    try:
        inv_corr = np.linalg.inv(corr_matrix)
        diag_inv = np.diag(inv_corr)
        partial_corr = -inv_corr / np.sqrt(np.outer(diag_inv, diag_inv))
        
        # Zero out diagonals for sum
        np.fill_diagonal(partial_corr, 0.0)
        corr_off_diag = corr_matrix.copy()
        np.fill_diagonal(corr_off_diag, 0.0)
        
        sum_r2 = float(np.sum(corr_off_diag ** 2))
        sum_p2 = float(np.sum(partial_corr ** 2))
        
        kmo = sum_r2 / (sum_r2 + sum_p2) if (sum_r2 + sum_p2) > 0 else 0.70
    except np.linalg.LinAlgError:
        kmo = 0.70
        
    # KMO Interpretation
    if kmo >= 0.90:
        kmo_label = "Marvelous (Ideal for Factor Analysis)"
        kmo_color = "#15803d"
    elif kmo >= 0.80:
        kmo_label = "Meritorious / Good"
        kmo_color = "#16a34a"
    elif kmo >= 0.70:
        kmo_label = "Middling / Fair"
        kmo_color = "#ca8a04"
    elif kmo >= 0.60:
        kmo_label = "Mediocre (Bare Minimum for EFA)"
        kmo_color = "#ea580c"
    else:
        kmo_label = "Unacceptable (< 0.60; Data Unsuitable for EFA)"
        kmo_color = "#dc2626"
        
    # 2. Bartlett's Test of Sphericity (H0: Corr is Identity Matrix)
    det_corr = np.linalg.det(corr_matrix)
    det_corr = max(det_corr, 1e-15)
    df_bartlett = int(n_items * (n_items - 1) / 2)
    stat_bartlett = -(n_samples - 1.0 - (2.0 * n_items + 5.0) / 6.0) * math.log(det_corr)
    p_bartlett = float(stats.chi2.sf(stat_bartlett, df_bartlett))
    
    # 3. Eigenvalues & Scree Extraction via PCA
    pca = PCA()
    pca.fit(df_clean)
    eigenvalues = pca.explained_variance_
    var_ratio_pct = pca.explained_variance_ratio_ * 100.0
    cum_var_pct = np.cumsum(var_ratio_pct)
    
    n_kaiser = int(np.sum(eigenvalues >= 1.0))
    
    scree_records = []
    for idx, (ev, vp, cp) in enumerate(zip(eigenvalues, var_ratio_pct, cum_var_pct), 1):
        scree_records.append({
            "Factor": f"Factor {idx}",
            "Eigenvalue": round(float(ev), 3),
            "Variance_Explained_%": round(float(vp), 1),
            "Cumulative_Variance_%": round(float(cp), 1),
            "Kaiser_Criterion": "Retain (λ ≥ 1.0)" if ev >= 1.0 else "Drop"
        })
    df_scree = pd.DataFrame(scree_records)
    
    # 4. Factor Loadings (Varimax 2-factor or n_kaiser model)
    n_factors_to_fit = max(2, min(n_kaiser, max_factors, n_items - 1))
    fa = FactorAnalysis(n_components=n_factors_to_fit, rotation='varimax', random_state=42)
    fa.fit(df_clean)
    
    loadings = fa.components_.T
    df_loadings = pd.DataFrame(
        np.round(loadings, 3),
        index=df_clean.columns,
        columns=[f"Factor {i+1}" for i in range(n_factors_to_fit)]
    )
    
    # Add communalities (h2 = sum of squared loadings across retained factors)
    df_loadings["Communality (h²)"] = np.round(np.sum(loadings ** 2, axis=1), 3)
    
    return {
        "kmo": round(float(kmo), 3),
        "kmo_label": kmo_label,
        "kmo_color": kmo_color,
        "bartlett_chi2": round(float(stat_bartlett), 2),
        "bartlett_df": df_bartlett,
        "bartlett_p_value": round(float(p_bartlett), 4),
        "n_factors_kaiser": n_kaiser,
        "scree_table": df_scree,
        "loadings_table": df_loadings,
        "correlation_matrix": pd.DataFrame(np.round(corr_matrix, 3), index=df_clean.columns, columns=df_clean.columns)
    }


# =====================================================================
# 6. Survey Sample Size with Finite Population Correction (FPC)
# =====================================================================

def calculate_survey_sample_size_fpc(
    population_size_n: Optional[int] = 5000,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    proportion_p: float = 0.50,
    response_rate: float = 0.35
) -> Dict[str, Any]:
    """Calculates survey sample size with Finite Population Correction (FPC)

    and response-rate invitation buffer.
    """
    alpha = 1.0 - confidence_level
    z_score = stats.norm.ppf(1.0 - alpha / 2.0)
    
    # Cochran's infinite baseline
    n0 = (z_score ** 2 * proportion_p * (1.0 - proportion_p)) / (margin_of_error ** 2)
    n0_ceil = math.ceil(n0)
    
    # Finite Population Correction
    if population_size_n is not None and population_size_n > 0:
        n_evaluable = n0 / (1.0 + (n0 - 1.0) / population_size_n)
        savings_pct = ((n0 - n_evaluable) / n0) * 100.0
        sampling_fraction_pct = (n_evaluable / population_size_n) * 100.0
    else:
        n_evaluable = n0
        savings_pct = 0.0
        sampling_fraction_pct = 0.0
        
    n_evaluable_ceil = math.ceil(n_evaluable)
    n_invited_ceil = math.ceil(n_evaluable_ceil / response_rate) if response_rate > 0 else n_evaluable_ceil
    
    # Sensitivity Table across margins of error (1% to 10%)
    moe_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    sensitivity_records = []
    for moe in moe_range:
        n0_sens = (z_score ** 2 * proportion_p * (1.0 - proportion_p)) / (moe ** 2)
        if population_size_n and population_size_n > 0:
            n_fpc_sens = n0_sens / (1.0 + (n0_sens - 1.0) / population_size_n)
        else:
            n_fpc_sens = n0_sens
        sensitivity_records.append({
            "Margin_of_Error": f"±{int(moe*100)}%",
            "Infinite_N0": math.ceil(n0_sens),
            "Evaluable_N_FPC": math.ceil(n_fpc_sens),
            "Invitations_Needed": math.ceil(math.ceil(n_fpc_sens) / response_rate)
        })
        
    return {
        "confidence_level_pct": int(confidence_level * 100),
        "z_critical": round(float(z_score), 3),
        "margin_of_error_pct": round(margin_of_error * 100.0, 1),
        "population_n": population_size_n,
        "n_infinite_n0": n0_ceil,
        "n_evaluable_fpc": n_evaluable_ceil,
        "n_invitations_required": n_invited_ceil,
        "fpc_savings_pct": round(float(savings_pct), 1),
        "sampling_fraction_pct": round(float(sampling_fraction_pct), 1),
        "sensitivity_table": pd.DataFrame(sensitivity_records)
    }


# =====================================================================
# 7. Post-Stratification Survey Weighting & Kish Neff Engine
# =====================================================================

def calculate_survey_weights_kish(
    df_responses: pd.DataFrame,
    group_col: str,
    target_pop_props: Dict[str, float],
    score_col: Optional[str] = None
) -> Dict[str, Any]:
    """Calculates post-stratification cell weights, weighted mean scores,

    Kish's Design Effect (Deff), and Effective Sample Size (Neff).
    """
    n_total = len(df_responses)
    sample_counts = df_responses[group_col].value_counts()
    sample_props = sample_counts / n_total
    
    # Calculate group cell weights: w_g = P_population / P_sample
    weights_map = {}
    weight_table_records = []
    
    for group, p_pop in target_pop_props.items():
        n_sample_grp = int(sample_counts.get(group, 0))
        p_sample_grp = float(sample_props.get(group, 0.0))
        
        weight = (p_pop / p_sample_grp) if p_sample_grp > 0 else 1.0
        weights_map[group] = weight
        
        weight_table_records.append({
            "Demographic_Group": group,
            "Target_Population_%": round(p_pop * 100.0, 1),
            "Sample_Respondents_N": n_sample_grp,
            "Sample_%": round(p_sample_grp * 100.0, 1),
            "Cell_Weight_w": round(weight, 3),
            "Representation": "Overrepresented" if weight < 0.95 else ("Underrepresented" if weight > 1.05 else "Balanced")
        })
        
    df_weights_summary = pd.DataFrame(weight_table_records)
    
    # Map weights onto individual responses
    df_weighted = df_responses.copy()
    df_weighted["Survey_Weight"] = df_weighted[group_col].map(weights_map)
    
    weights = df_weighted["Survey_Weight"].values
    sum_w = float(np.sum(weights))
    sum_w2 = float(np.sum(weights ** 2))
    
    # Kish's Effective Sample Size: Neff = (sum(w))^2 / sum(w^2)
    neff = (sum_w ** 2) / sum_w2
    deff = n_total / neff if neff > 0 else 1.0
    efficiency_pct = (neff / n_total) * 100.0
    
    # Score metrics (if score_col provided)
    unweighted_mean = None
    weighted_mean = None
    if score_col and score_col in df_responses.columns:
        scores = df_responses[score_col].values
        unweighted_mean = round(float(np.mean(scores)), 2)
        weighted_mean = round(float(np.sum(scores * weights) / sum_w), 2)
        
    return {
        "n_original": n_total,
        "n_effective_kish": round(float(neff), 1),
        "design_effect_deff": round(float(deff), 3),
        "efficiency_pct": round(float(efficiency_pct), 1),
        "unweighted_mean": unweighted_mean,
        "weighted_mean": weighted_mean,
        "min_weight": round(float(np.min(weights)), 3),
        "max_weight": round(float(np.max(weights)), 3),
        "weight_table": df_weights_summary,
        "weighted_data": df_weighted
    }


# =====================================================================
# 8. Synthetic Survey Dataset Generator
# =====================================================================

def generate_synthetic_survey_data(method_key: str, n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic healthcare, NHS, and operational survey data."""
    rng = np.random.default_rng(seed)
    
    if method_key == "likert_scale_analysis":
        # 5-item Patient Reported Experience Measure (PREM)
        items = {
            "Q1_Clear_Communication": np.clip(rng.normal(3.8, 1.1, n_samples).round(), 1, 5).astype(int),
            "Q2_Involved_In_Decisions": np.clip(rng.normal(3.4, 1.2, n_samples).round(), 1, 5).astype(int),
            "Q3_Staff_Compassion": np.clip(rng.normal(4.1, 0.9, n_samples).round(), 1, 5).astype(int),
            "Q4_Cleanliness_Comfort": np.clip(rng.normal(3.7, 1.0, n_samples).round(), 1, 5).astype(int),
            "Q5_Overall_Care_Quality": np.clip(rng.normal(3.9, 1.0, n_samples).round(), 1, 5).astype(int)
        }
        df = pd.DataFrame(items)
        df.insert(0, "Respondent_ID", [f"RESP-{i+1:04d}" for i in range(n_samples)])
        return df
        
    elif method_key in ("cronbachs_alpha", "mcdonalds_omega"):
        # 6-item Clinician Wellbeing & Work Exhaustion Scale
        latent_burnout = rng.normal(0, 1, n_samples)
        
        # Loadings for omega vs alpha contrast
        loadings = [0.85, 0.78, 0.82, 0.65, 0.72, 0.45] # Q6 has weaker loading
        items = {}
        for idx, (col_name, l) in enumerate(zip(
            ["Q1_Emotional_Exhaustion", "Q2_Work_Overload", "Q3_Fatigue_End_Of_Shift", 
             "Q4_Cynicism_Frustration", "Q5_Diminished_Accomplishment", "Q6_Administrative_Burdens"],
            loadings
        )):
            noise = rng.normal(0, np.sqrt(1.0 - l**2), n_samples)
            z = l * latent_burnout + noise
            # Map continuous z to 1-5 Likert
            items[col_name] = np.clip(np.round(3.0 + 1.2 * z), 1, 5).astype(int)
            
        df = pd.DataFrame(items)
        df.insert(0, "Clinician_ID", [f"NHS-{i+1:04d}" for i in range(n_samples)])
        return df
        
    elif method_key == "cohens_kappa":
        # 2 Independent Clinicians assessing Patient Triage Urgency (3 categories: 1=Low, 2=Medium, 3=High)
        true_urgency = rng.choice([1, 2, 3], size=n_samples, p=[0.40, 0.35, 0.25])
        
        # Clinician 1 (85% concordance with true state)
        c1 = [u if rng.uniform() < 0.85 else rng.choice([1, 2, 3]) for u in true_urgency]
        # Clinician 2 (80% concordance with true state)
        c2 = [u if rng.uniform() < 0.80 else rng.choice([1, 2, 3]) for u in true_urgency]
        
        labels = {1: "1_Low_Urgency", 2: "2_Medium_Urgency", 3: "3_High_Urgency"}
        df = pd.DataFrame({
            "Patient_ID": [f"PT-{i+1:04d}" for i in range(n_samples)],
            "Rater1_Triage_Nurse": [labels[val] for val in c1],
            "Rater2_Senior_Clinician": [labels[val] for val in c2]
        })
        return df
        
    elif method_key == "exploratory_factor_analysis":
        # 9-item Quality of Life Scale (3 latent factors: Physical, Psychological, Social)
        f_phys = rng.normal(0, 1, n_samples)
        f_psych = 0.4 * f_phys + 0.9 * rng.normal(0, 1, n_samples)
        f_soc = 0.3 * f_phys + 0.3 * f_psych + 0.9 * rng.normal(0, 1, n_samples)
        
        items = {
            "Q1_Mobility": np.clip(np.round(3.2 + 1.1 * (0.85 * f_phys + 0.35 * rng.normal(0, 1, n_samples))), 1, 5).astype(int),
            "Q2_Pain_Discomfort": np.clip(np.round(3.0 + 1.1 * (0.80 * f_phys + 0.40 * rng.normal(0, 1, n_samples))), 1, 5).astype(int),
            "Q3_Physical_Vitality": np.clip(np.round(3.4 + 1.0 * (0.78 * f_phys + 0.42 * rng.normal(0, 1, n_samples))), 1, 5).astype(int),
            "Q4_Anxiety_Depression": np.clip(np.round(2.8 + 1.2 * (0.82 * f_psych + 0.38 * rng.normal(0, 1, n_samples))), 1, 5).astype(int),
            "Q5_Mental_Focus": np.clip(np.round(3.1 + 1.0 * (0.75 * f_psych + 0.45 * rng.normal(0, 1, n_samples))), 1, 5).astype(int),
            "Q6_Emotional_Resilience": np.clip(np.round(3.3 + 1.1 * (0.79 * f_psych + 0.41 * rng.normal(0, 1, n_samples))), 1, 5).astype(int),
            "Q7_Social_Engagement": np.clip(np.round(3.5 + 1.1 * (0.81 * f_soc + 0.40 * rng.normal(0, 1, n_samples))), 1, 5).astype(int),
            "Q8_Family_Support": np.clip(np.round(3.8 + 1.0 * (0.76 * f_soc + 0.45 * rng.normal(0, 1, n_samples))), 1, 5).astype(int),
            "Q9_Community_Belonging": np.clip(np.round(3.2 + 1.2 * (0.74 * f_soc + 0.47 * rng.normal(0, 1, n_samples))), 1, 5).astype(int)
        }
        df = pd.DataFrame(items)
        df.insert(0, "Patient_ID", [f"QOL-{i+1:04d}" for i in range(n_samples)])
        return df
        
    elif method_key == "survey_sample_size":
        # Sizing lookup table
        moe_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
        pop_sizes = [500, 1000, 2500, 5000, 10000, 50000]
        records = []
        for p_size in pop_sizes:
            for moe in moe_range:
                res = calculate_survey_sample_size_fpc(p_size, margin_of_error=moe)
                records.append({
                    "Population_Size_N": p_size,
                    "Margin_of_Error": f"±{int(moe*100)}%",
                    "Evaluable_N_Required": res["n_evaluable_fpc"],
                    "Invitations_at_35pct_Response": res["n_invitations_required"],
                    "Sampling_Fraction_%": res["sampling_fraction_pct"]
                })
        return pd.DataFrame(records)
        
    elif method_key == "survey_weighting":
        # 500 Patient Feedback Responses with Age-Group Underrepresentation
        # Target Population: 18-34: 30%, 35-64: 50%, 65+: 20%
        # Sample: 18-34: 10% (under), 35-64: 45%, 65+: 45% (over)
        age_groups = rng.choice(
            ["18-34 (Young Adult)", "35-64 (Working Age)", "65+ (Older Adult)"],
            size=n_samples,
            p=[0.10, 0.45, 0.45]
        )
        
        # Satisfaction scores (1 to 10 scale; younger patients slightly more critical)
        scores = []
        for ag in age_groups:
            if "18-34" in ag:
                s = rng.normal(6.5, 1.8)
            elif "35-64" in ag:
                s = rng.normal(7.4, 1.5)
            else:
                s = rng.normal(8.6, 1.2)
            scores.append(int(np.clip(round(s), 1, 10)))
            
        df = pd.DataFrame({
            "Respondent_ID": [f"RESP-{i+1:04d}" for i in range(n_samples)],
            "Age_Group": age_groups,
            "Satisfaction_Score_1_to_10": scores
        })
        return df
        
    else:
        # Generic fallback
        return pd.DataFrame({"Item_1": [1, 2, 3], "Item_2": [4, 5, 6]})

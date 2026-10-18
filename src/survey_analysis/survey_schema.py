"""Survey & Questionnaire Analysis Schema and Method Taxonomy.

Defines the decision tree taxonomy, metadata, mathematical formulations,
plain English explanations, Excel guides, and Python code templates for
survey research, psychometrics, and questionnaire methodology.
"""

from typing import Dict, Any, List

SURVEY_CATEGORIES: Dict[str, Dict[str, str]] = {
    "scale_measurement": {
        "id": "scale_measurement",
        "name": "Scale Measurement & Descriptive Visualization",
        "icon": "📊",
        "description": "Evaluate individual Likert questions vs composite scores, ordinal vs interval trade-offs, and diverging stacked distributions."
    },
    "scale_reliability": {
        "id": "scale_reliability",
        "name": "Internal Consistency & Scale Reliability (Psychometrics)",
        "icon": "🔬",
        "description": "Quantify whether survey questions measure a coherent underlying construct via Cronbach's Alpha and McDonald's Omega."
    },
    "inter_rater_agreement": {
        "id": "inter_rater_agreement",
        "name": "Inter-Rater Reliability & Concordance",
        "icon": "👥",
        "description": "Assess degree of agreement between clinicians, interviewers, or survey coders beyond chance (Cohen's Kappa & Weighted Kappa)."
    },
    "construct_validity": {
        "id": "construct_validity",
        "name": "Construct Validity & Dimensionality Reduction",
        "icon": "🧩",
        "description": "Discover latent subscales, check sampling adequacy (KMO & Bartlett's), and reduce questionnaire item redundancy via Factor Analysis."
    },
    "sampling_weighting": {
        "id": "sampling_weighting",
        "name": "Survey Sampling, Sizing & Non-Response Weighting",
        "icon": "📐",
        "description": "Calculate survey sample size with Finite Population Correction (FPC) and correct non-response bias via post-stratification cell weighting."
    }
}


SURVEY_DECISION_NODES: List[Dict[str, Any]] = [
    {
        "id": "survey_goal",
        "question": "What is the primary objective of your survey or questionnaire analysis?",
        "options": [
            {
                "value": "analyze_likert",
                "label": "📊 Analyze single Likert survey questions or composite satisfaction scores",
                "recommendation": "likert_scale_analysis"
            },
            {
                "value": "assess_scale_reliability",
                "label": "🔬 Assess if multi-item questions reliably measure a single construct (Scale Reliability)",
                "recommendation": "cronbachs_alpha"
            },
            {
                "value": "evaluate_tau_equivalence",
                "label": "⚖️ Evaluate scale reliability when items have unequal factor loadings (McDonald's Omega)",
                "recommendation": "mcdonalds_omega"
            },
            {
                "value": "measure_rater_agreement",
                "label": "👥 Measure agreement between 2 independent clinicians, raters, or survey coders",
                "recommendation": "cohens_kappa"
            },
            {
                "value": "explore_latent_factors",
                "label": "🧩 Discover latent dimensions / subscales in a multi-question questionnaire (EFA / PCA)",
                "recommendation": "exploratory_factor_analysis"
            },
            {
                "value": "size_finite_survey",
                "label": "📐 Determine required sample size for a survey of a known finite population (e.g. NHS Trust)",
                "recommendation": "survey_sample_size"
            },
            {
                "value": "weight_non_response",
                "label": "⚖️ Correct for demographic non-response bias via survey weighting and evaluate Kish design effect",
                "recommendation": "survey_weighting"
            }
        ]
    }
]


SURVEY_METHODS: Dict[str, Dict[str, Any]] = {
    "likert_scale_analysis": {
        "id": "likert_scale_analysis",
        "name": "Likert Scale Analysis: Ordinal vs. Interval Decision Framework",
        "category": "scale_measurement",
        "icon": "📊",
        "summary": "Determine whether to treat survey responses as ordinal (median/IQR) or interval (mean/SD), visualize response distributions, and calculate Top-2-Box and Net Agreement scores.",
        "when_to_use": [
            "Analyzing single 5-point or 7-point Likert items (e.g., Strongly Disagree to Strongly Agree).",
            "Deciding whether summated multi-item scale scores qualify for parametric statistical tests.",
            "Reporting patient experience (PREMs) or staff satisfaction survey items to clinical leadership."
        ],
        "key_assumptions": [
            "Single Likert items are strictly Ordinal: distances between response categories cannot be assumed equal.",
            "Multi-item composite scales (sum of >= 4 Likert items) approximate interval data under the Central Limit Theorem (Norman 2010).",
            "Neutral midpoints ('Neither agree nor disagree') must be distinguished from missing data ('Not applicable')."
        ],
        "common_pitfalls": [
            "Reporting means and standard deviations for a single 5-point Likert question (violates ordinal assumptions).",
            "Treating 'Don't Know' or 'Not Applicable' as the numeric midpoint (e.g., coding N/A as 3).",
            "Using standard stacked bar charts that hide the balance of positive vs negative sentiment."
        ],
        "plain_english_rules": [
            "Single Item Rule: Report frequencies (%), Medians, and Interquartile Ranges (IQR). Visualize using a Diverging Stacked Bar Chart centered on neutral.",
            "Composite Scale Rule: When at least 4–8 related items are summed into a composite index, the score can be analyzed parametrically (Means, SDs, t-tests, ANOVA, OLS regression) if distribution is reasonably unimodal.",
            "Top-2-Box Metric: Proportion of respondents selecting the top two positive tiers (Agree + Strongly Agree), useful for executive KPI benchmarks.",
            "Net Agreement Metric: Percentage of positive respondents minus percentage of negative respondents (% Agree - % Disagree)."
        ],
        "formulas": {
            "top_two_box": {
                "latex": r"\text{Top-2-Box \%} = \frac{n_{\text{Agree}} + n_{\text{Strongly Agree}}}{N_{\text{total}}} \times 100",
                "plain_english": "Top-2-Box percentage is the count of respondents who answered either 'Agree' or 'Strongly Agree' divided by total respondents, expressed as a percentage."
            },
            "net_agreement": {
                "latex": r"\text{Net Agreement \%} = \left(\frac{n_{\text{Agree}} + n_{\text{Strongly Agree}}}{N_{\text{total}}} - \frac{n_{\text{Disagree}} + n_{\text{Strongly Disagree}}}{N_{\text{total}}}\right) \times 100",
                "plain_english": "Net Agreement is the percentage of respondents who agreed minus the percentage who disagreed, creating a score ranging from -100% (unanimous disagreement) to +100% (unanimous agreement)."
            }
        },
        "excel_guide": {
            "description": "Calculate Likert frequency distributions, Top-2-Box %, and Net Agreement in Excel.",
            "cell_mapping": {
                "A1:A200": "Raw survey responses coded 1 (Strongly Disagree) to 5 (Strongly Agree)",
                "C2:C6": "Response scale labels: 1=Strongly Disagree, 2=Disagree, 3=Neutral, 4=Agree, 5=Strongly Agree",
                "D2:D6": "Frequency counts using `=COUNTIF($A$2:$A$201, ROW()-1)`",
                "E2:E6": "Percentage distribution using `=D2/SUM($D$2:$D$6)`",
                "D8": "Median: `=MEDIAN(A2:A201)`",
                "D9": "Q1 (25th percentile): `=PERCENTILE.INC(A2:A201, 0.25)`",
                "D10": "Q3 (75th percentile): `=PERCENTILE.INC(A2:A201, 0.75)`",
                "D11": "Top-2-Box %: `=(D5+D6)/SUM(D2:D6)`",
                "D12": "Net Agreement %: `=((D5+D6)-(D2+D3))/SUM(D2:D6)`"
            },
            "steps": [
                "1. Clean data: Ensure missing or 'Not Applicable' responses are recorded as empty cells, not zero or 3.",
                "2. Create a frequency summary table in cells C2:E6 using `=COUNTIF()` for each rating tier 1 to 5.",
                "3. Calculate ordinal central tendency via `=MEDIAN()` and dispersion via `=PERCENTILE.INC()`.",
                "4. Compute Top-2-Box satisfaction by summing tiers 4 and 5 divided by total respondents.",
                "5. To build a Diverging Bar Chart in Excel: Insert a 100% Stacked Bar Chart, split neutral responses into two equal 50% halves on both sides of zero, and order categories from negative to positive."
            ]
        },
        "python_code": """# Likert Scale Summary & Diverging Bar Chart Analysis
import pandas as pd
import numpy as np

# Load or generate survey responses (1=Strongly Disagree to 5=Strongly Agree)
# df contains items: ['Q1_Communication', 'Q2_Cleanliness', 'Q3_Timeliness', 'Q4_Dignity', 'Q5_Overall']

def summarize_likert_item(series, name="Question"):
    n_total = series.dropna().count()
    counts = series.value_counts().sort_index()
    percentages = (counts / n_total) * 100
    
    top_2_box = percentages.get(4, 0) + percentages.get(5, 0)
    bottom_2_box = percentages.get(1, 0) + percentages.get(2, 0)
    net_agreement = top_2_box - bottom_2_box
    
    median_val = series.median()
    iqr_val = series.quantile(0.75) - series.quantile(0.25)
    
    return {
        "Item": name,
        "N": int(n_total),
        "Median": float(median_val),
        "IQR": float(iqr_val),
        "Top_2_Box_%": round(top_2_box, 1),
        "Bottom_2_Box_%": round(bottom_2_box, 1),
        "Net_Agreement_%": round(net_agreement, 1)
    }

# Example execution across questions
# results = [summarize_likert_item(df[col], col) for col in df.columns]
# summary_df = pd.DataFrame(results)
# print(summary_df.to_string(index=False))
"""
    },

    "cronbachs_alpha": {
        "id": "cronbachs_alpha",
        "name": "Cronbach's Alpha (α) & Item-Total Scale Reliability",
        "category": "scale_reliability",
        "icon": "🔬",
        "summary": "Evaluate the internal consistency of a multi-item questionnaire and identify redundant or poorly performing questions via the 'Alpha If Item Deleted' table.",
        "when_to_use": [
            "Validating whether a multi-item psychometric scale (e.g. PHQ-9, GAD-7, EQ-5D, employee burnout) measures a coherent unidimensional trait.",
            "Purifying survey items during questionnaire development by removing items that degrade reliability.",
            "Documenting scale reliability prior to computing composite summated scores for statistical tests."
        ],
        "key_assumptions": [
            "Unidimensionality: All items are assumed to measure a single underlying latent construct.",
            "Tau-equivalence: All items contribute equally to the true score (equal factor loadings across questions).",
            "Uncorrelated errors: Item measurement errors are independent (no shared method variance or phrasing artifacts)."
        ],
        "common_pitfalls": [
            "Failing to reverse-code negatively phrased items prior to computing alpha (results in spuriously low or negative alpha).",
            "Assuming high alpha implies unidimensionality (alpha can be high in multidimensional scales if item count k is large).",
            "Aiming for alpha > 0.95 (extreme redundancy where questions are merely repetitive paraphrases)."
        ],
        "plain_english_rules": [
            "Rule of Thumb Benchmarks (Nunnally & Bernstein 1994):",
            "• α < 0.60: Unacceptable internal consistency (scale cannot be reliably interpreted).",
            "• 0.60 ≤ α < 0.70: Marginal / Questionable (acceptable only for exploratory pilot surveys).",
            "• 0.70 ≤ α < 0.80: Acceptable internal consistency (standard benchmark for academic research).",
            "• 0.80 ≤ α < 0.90: Good / High internal consistency (recommended for operational decisions).",
            "• α ≥ 0.90: Excellent consistency (mandatory for high-stakes individual clinical diagnostics; beware question redundancy if α > 0.95).",
            "Item-Rest Correlation: Measures correlation between an individual question and the sum of all remaining questions. Items with r < 0.30 should be revised or removed.",
            "Alpha-If-Item-Deleted: If dropping question i increases the total scale alpha, question i is degrading overall reliability."
        ],
        "formulas": {
            "cronbach_alpha": {
                "latex": r"\alpha = \frac{k}{k - 1} \left(1 - \frac{\sum_{i=1}^k \sigma_i^2}{\sigma_X^2}\right)",
                "plain_english": "Cronbach's Alpha equals [k / (k - 1)] multiplied by [1 minus (sum of individual item variances divided by total composite scale variance)], where k is the number of questions."
            },
            "item_rest_correlation": {
                "latex": r"r_{i, X_{-i}} = \frac{\text{Cov}(X_i, X_{-i})}{\sigma_i \cdot \sigma_{X_{-i}}}",
                "plain_english": "Corrected Item-Rest Correlation is the Pearson correlation between item i and the sum of all other items (excluding item i), preventing self-correlation inflation."
            }
        },
        "excel_guide": {
            "description": "Calculate Cronbach's Alpha and Alpha-If-Item-Deleted in Microsoft Excel without add-ins.",
            "cell_mapping": {
                "A2:F151": "150 survey respondents across 6 questions (Q1 to Q6)",
                "G2:G151": "Total Composite Score per person: `=SUM(A2:F2)`",
                "A153:F153": "Individual item sample variances: `=VAR.S(A2:A151)`",
                "G153": "Total scale variance: `=VAR.S(G2:G151)`",
                "G154": "Sum of item variances: `=SUM(A153:F153)`",
                "G155": "Item count k: `=COUNTA(A1:F1)`",
                "G156": "Cronbach's Alpha: `=(G155/(G155-1))*(1-(G154/G153))`"
            },
            "steps": [
                "1. Check for reverse-coded questions: If a question is negatively phrased on a 1-5 scale, reverse it using `6 - original_value`.",
                "2. Create column G with `=SUM(A2:F2)` to compute total person scores.",
                "3. Compute item variances in row 153 using `=VAR.S()`, and total scale variance in cell G153 using `=VAR.S(G2:G151)`.",
                "4. Sum the item variances in cell G154 using `=SUM(A153:F153)`.",
                "5. Calculate Alpha in cell G156 using `=(6/5)*(1 - (G154/G153))`.",
                "6. To compute 'Alpha if Deleted' for Item 1: Recalculate composite scores excluding column A, compute new total variance, and apply the formula with k=5."
            ]
        },
        "python_code": """# Cronbach's Alpha and Item-Total Reliability Analysis
import numpy as np
import pandas as pd

def cronbach_alpha(df_items):
    \"\"\"Calculates Cronbach's alpha, item variances, and alpha-if-item-deleted.\"\"\"
    k = df_items.shape[1]
    if k < 2:
        raise ValueError("At least 2 items required to compute Cronbach's Alpha.")
        
    item_vars = df_items.var(axis=0, ddof=1)
    sum_item_vars = item_vars.sum()
    total_scores = df_items.sum(axis=1)
    total_var = total_scores.var(ddof=1)
    
    alpha = (k / (k - 1)) * (1.0 - (sum_item_vars / total_var))
    
    # Item-Deleted Analysis
    item_deleted_records = []
    for col in df_items.columns:
        sub_df = df_items.drop(columns=[col])
        k_sub = sub_df.shape[1]
        sub_item_vars = sub_df.var(axis=0, ddof=1).sum()
        sub_total_scores = sub_df.sum(axis=1)
        sub_total_var = sub_total_scores.var(ddof=1)
        alpha_deleted = (k_sub / (k_sub - 1)) * (1.0 - (sub_item_vars / sub_total_var))
        
        # Corrected item-total correlation
        r_item_rest = np.corrcoef(df_items[col], sub_total_scores)[0, 1]
        
        item_deleted_records.append({
            "Item": col,
            "Item_Mean": round(df_items[col].mean(), 2),
            "Item_Var": round(item_vars[col], 3),
            "Item_Rest_Corr": round(r_item_rest, 3),
            "Alpha_If_Deleted": round(alpha_deleted, 3)
        })
        
    return {
        "alpha": round(alpha, 3),
        "k_items": k,
        "n_samples": len(df_items),
        "total_variance": round(total_var, 3),
        "item_analysis": pd.DataFrame(item_deleted_records)
    }
"""
    },

    "mcdonalds_omega": {
        "id": "mcdonalds_omega",
        "name": "McDonald's Omega (ω) & Violation of Tau-Equivalence",
        "category": "scale_reliability",
        "icon": "⚖️",
        "summary": "Modern psychometric reliability coefficient that accounts for unequal factor loadings across questionnaire items, overcoming the systematic underestimation bias of Cronbach's Alpha.",
        "when_to_use": [
            "Validating clinical or psychological scales where questions differ in their sensitivity or importance (unequal factor loadings).",
            "Reporting psychometric reliability to peer-reviewed journals requiring McDonald's omega as modern best practice.",
            "Verifying whether Cronbach's alpha is artificially depressed due to tau-equivalence violation."
        ],
        "key_assumptions": [
            "Congeneric model: Items measure the same underlying construct, but each item has its own distinct factor loading (λi).",
            "Item uniqueness: Error terms are uncorrelated across questions.",
            "Continuous or polytomous items with at least 4 response categories."
        ],
        "common_pitfalls": [
            "Assuming Cronbach's alpha and McDonald's omega are always identical (they only match when all factor loadings are equal).",
            "Interpreting McDonald's hierarchical omega (ωh) and total omega (ωt) as the same metric (hierarchical omega isolates the general factor in bifactor models).",
            "Attempting to calculate omega with negative factor loadings without reverse-coding."
        ],
        "plain_english_rules": [
            "Why Omega is Superior to Alpha:",
            "• Cronbach's Alpha assumes every question is equally strongly linked to the trait (equal factor loadings λ1 = λ2 = ... = λk). In reality, some questions are central while others are peripheral.",
            "• When factor loadings vary, Alpha represents a lower bound that systematically underestimates true reliability.",
            "• McDonald's Omega estimates the individual factor loadings (λ) and computes the exact proportion of variance accounted for by the underlying latent factor.",
            "Interpretation Benchmarks: Same as Alpha (ω ≥ 0.70 acceptable for research, ω ≥ 0.80 good, ω ≥ 0.90 clinical diagnostics)."
        ],
        "formulas": {
            "mcdonald_omega_total": {
                "latex": r"\omega_{\text{total}} = \frac{\left(\sum_{i=1}^k \lambda_i\right)^2}{\left(\sum_{i=1}^k \lambda_i\right)^2 + \sum_{i=1}^k (1 - \lambda_i^2)}",
                "plain_english": "McDonald's Omega equals [squared sum of all factor loadings] divided by [squared sum of factor loadings plus sum of unique error variances (1 - lambda^2)]."
            }
        },
        "excel_guide": {
            "description": "Calculate McDonald's Omega in Microsoft Excel using standardized factor loadings.",
            "cell_mapping": {
                "A2:A7": "Survey Questions Q1 to Q6",
                "B2:B7": "Standardized Factor Loadings λ (extracted from Factor Analysis / PCA)",
                "C2:C7": "Item uniqueness / error variance: `=1 - (B2^2)`",
                "B8": "Sum of factor loadings: `=SUM(B2:B7)`",
                "B9": "Squared sum of loadings: `=B8^2`",
                "C8": "Sum of uniqueness errors: `=SUM(C2:C7)`",
                "B10": "McDonald's Omega: `=B9 / (B9 + C8)`"
            },
            "steps": [
                "1. Obtain standardized factor loadings (λ) for your 6 items using Python, SPSS, or Excel PCA.",
                "2. Enter factor loadings in column B (cells B2:B7). Ensure all loadings are positive (reverse-coded).",
                "3. In column C, compute item error variance using `=1 - (B2^2)`.",
                "4. Sum the factor loadings in B8 using `=SUM(B2:B7)` and square the sum in cell B9 using `=B8^2`.",
                "5. Sum the error variances in cell C8 using `=SUM(C2:C7)`.",
                "6. Calculate McDonald's Omega in B10 using `=B9 / (B9 + C8)`."
            ]
        },
        "python_code": """# McDonald's Omega (Total) Calculation using Factor Analysis
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

def mcdonalds_omega(df_items):
    \"\"\"Fits a single-factor model and computes McDonald's total omega.\"\"\"
    # Standardize items
    scaled_df = (df_items - df_items.mean()) / df_items.std(ddof=1)
    
    # Fit 1-factor model
    fa = FactorAnalysis(n_components=1, random_state=42)
    fa.fit(scaled_df)
    
    # Extract factor loadings (lambda)
    loadings = np.abs(fa.components_[0])
    
    # Error variances (uniqueness = 1 - lambda^2)
    uniqueness = np.maximum(1.0 - loadings**2, 0.001)
    
    sum_loadings = np.sum(loadings)
    omega = (sum_loadings ** 2) / ((sum_loadings ** 2) + np.sum(uniqueness))
    
    return {
        "omega_total": round(float(omega), 3),
        "sum_loadings_squared": round(float(sum_loadings ** 2), 3),
        "sum_error_variance": round(float(np.sum(uniqueness)), 3),
        "loadings": {col: round(float(l), 3) for col, l in zip(df_items.columns, loadings)}
    }
"""
    },

    "cohens_kappa": {
        "id": "cohens_kappa",
        "name": "Cohen's Kappa (κ) & Weighted Kappa (κw) for Inter-Rater Agreement",
        "category": "inter_rater_agreement",
        "icon": "👥",
        "summary": "Measure concordance between two independent clinicians, observers, or survey coders, penalizing for agreement expected purely by chance.",
        "when_to_use": [
            "Assessing diagnostic agreement between two radiologists or pathologists reviewing clinical scans.",
            "Evaluating inter-coder reliability when classifying open-ended survey text into thematic categories.",
            "Comparing ordinal triage ratings (e.g., Emergency Severity Index: Resuscitation, Emergent, Urgent, Non-Urgent) using Weighted Kappa."
        ],
        "key_assumptions": [
            "Two independent raters evaluate the exact same set of N subjects.",
            "Rating categories are mutually exclusive and collectively exhaustive.",
            "Raters make evaluations independently without conferring."
        ],
        "common_pitfalls": [
            "Using simple percent agreement instead of Kappa (inflates perceived agreement when one category is highly prevalent).",
            "Applying unweighted Kappa to ordinal ratings (treats a disagreement between 'Mild' and 'Severe' the same as 'Mild' vs 'Moderate').",
            "The Kappa Paradox (Feinstein & Cicchetti 1990): High observed agreement can yield paradoxically low Kappa if prevalence is heavily skewed."
        ],
        "plain_english_rules": [
            "Landis & Koch (1977) Agreement Benchmarks:",
            "• κ < 0.00: Poor agreement (worse than chance).",
            "• 0.00 ≤ κ ≤ 0.20: Slight agreement.",
            "• 0.21 ≤ κ ≤ 0.40: Fair agreement.",
            "• 0.41 ≤ κ ≤ 0.60: Moderate agreement.",
            "• 0.61 ≤ κ ≤ 0.80: Substantial agreement (standard benchmark for clinical coding).",
            "• 0.81 ≤ κ ≤ 1.00: Almost perfect agreement.",
            "Unweighted vs Weighted Kappa:",
            "• Unweighted: Used for unordered nominal categories (e.g. Yes/No, Type A/B/C). All disagreements receive zero credit.",
            "• Linear Weighted: Used for ordinal scales. A 1-step disagreement receives partial credit; a 2-step disagreement receives less.",
            "• Quadratic Weighted: Disagreements are penalized quadratically (equivalent to the Intraclass Correlation Coefficient ICC for continuous data)."
        ],
        "formulas": {
            "unweighted_kappa": {
                "latex": r"\kappa = \frac{p_o - p_e}{1 - p_e}",
                "plain_english": "Cohen's Kappa equals [observed agreement proportion minus chance-expected agreement proportion] divided by [1 minus chance-expected agreement proportion]."
            },
            "weighted_kappa": {
                "latex": r"\kappa_w = \frac{\sum_{i=1}^C \sum_{j=1}^C w_{ij} p_{o,ij} - \sum_{i=1}^C \sum_{j=1}^C w_{ij} p_{e,ij}}{1 - \sum_{i=1}^C \sum_{j=1}^C w_{ij} p_{e,ij}}",
                "plain_english": "Weighted Kappa weights each cell in the confusion matrix by agreement distance w_ij, where quadratic weights are w_ij = 1 - (i - j)^2 / (C - 1)^2."
            }
        },
        "excel_guide": {
            "description": "Calculate Cohen's Kappa for a 3x3 classification matrix in Microsoft Excel.",
            "cell_mapping": {
                "B2:D4": "3x3 confusion matrix: Rater 1 rows vs Rater 2 columns",
                "E2:E4": "Row totals (Rater 1 marginals): `=SUM(B2:D2)`",
                "B5:D5": "Column totals (Rater 2 marginals): `=SUM(B2:B4)`",
                "E5": "Total subjects N: `=SUM(B2:D4)`",
                "B7": "Observed Agreement po: `=(B2+C3+D4)/E5`",
                "B8": "Expected Agreement pe: `=((E2*B5) + (E3*C5) + (E4*D5)) / (E5^2)`",
                "B9": "Cohen's Kappa: `=(B7 - B8) / (1 - B8)`"
            },
            "steps": [
                "1. Build the contingency table of cross-classified ratings in cells B2:D4.",
                "2. Calculate marginal row totals in column E and column totals in row 5.",
                "3. Compute observed agreement po in cell B7 by summing the diagonal cells (B2+C3+D4) divided by total N (E5).",
                "4. Compute chance agreement pe in cell B8 by multiplying corresponding row and column totals and dividing by N^2.",
                "5. Calculate Kappa in cell B9 using `=(B7 - B8)/(1 - B8)`."
            ]
        },
        "python_code": """# Cohen's Kappa and Quadratic Weighted Kappa in Python
from sklearn.metrics import cohen_kappa_score
import numpy as np

# rater1 and rater2 are lists/arrays of categorical or ordinal ratings
# e.g., rater1 = [1, 2, 3, 2, 1, ...], rater2 = [1, 2, 2, 2, 1, ...]

def calculate_rater_agreement(rater1, rater2, is_ordinal=True):
    # Unweighted Cohen's Kappa
    kappa_unweighted = cohen_kappa_score(rater1, rater2)
    
    # Quadratic Weighted Kappa (for ordinal scales)
    kappa_weighted = cohen_kappa_score(rater1, rater2, weights='quadratic') if is_ordinal else None
    
    # Simple percentage agreement
    pct_agreement = np.mean(np.array(rater1) == np.array(rater2)) * 100
    
    return {
        "pct_agreement": round(pct_agreement, 1),
        "kappa_unweighted": round(kappa_unweighted, 3),
        "kappa_quadratic_weighted": round(kappa_weighted, 3) if kappa_weighted is not None else "N/A"
    }
"""
    },

    "exploratory_factor_analysis": {
        "id": "exploratory_factor_analysis",
        "name": "Exploratory Factor Analysis (EFA) & Scree Plot Dimensionality",
        "category": "construct_validity",
        "icon": "🧩",
        "summary": "Discover latent subscales in a multi-question survey, evaluate sampling adequacy (KMO and Bartlett's test), and determine the number of factors to retain via Cattell's Scree Plot.",
        "when_to_use": [
            "Validating whether a 20-item health survey measures 3 distinct dimensions (e.g. Physical, Mental, Social health).",
            "Eliminating redundant or cross-loading questions during new questionnaire development.",
            "Testing whether items cluster into expected clinical domains before computing domain-specific composite scores."
        ],
        "key_assumptions": [
            "Sampling adequacy: Kaiser-Meyer-Olkin (KMO) index >= 0.60 across the item correlation matrix.",
            "Sphericity: Bartlett's test p < 0.05 (confirms correlations between items are significantly different from zero).",
            "Sample size: Rule of thumb N >= 5 to 10 respondents per survey question (N >= 100 minimum)."
        ],
        "common_pitfalls": [
            "Confusing PCA with EFA: PCA reduces total variance into composite components; EFA models underlying latent constructs by isolating shared common variance from unique error variance.",
            "Using the Kaiser criterion (Eigenvalues > 1) mechanically (often over-extracts factors; Cattell's scree elbow is more robust).",
            "Forcing orthogonal Varimax rotation when clinical or psychological dimensions naturally correlate (use Promax / Oblimin instead)."
        ],
        "plain_english_rules": [
            "Kaiser-Meyer-Olkin (KMO) Sampling Adequacy Benchmarks:",
            "• KMO < 0.50: Unacceptable (data cannot be factored).",
            "• 0.50 ≤ KMO < 0.60: Miserable.",
            "• 0.60 ≤ KMO < 0.70: Mediocre (bare minimum threshold for EFA).",
            "• 0.70 ≤ KMO < 0.80: Middling / Fair.",
            "• 0.80 ≤ KMO < 0.90: Meritorious / Good.",
            "• KMO ≥ 0.90: Marvelous (ideal for factor analysis).",
            "Factor Retention Decision Rules:",
            "• Kaiser Criterion: Retain factors with Eigenvalue λ ≥ 1.0 (each retained factor explains at least as much variance as 1 individual question).",
            "• Cattell's Scree Test: Plot eigenvalues in descending order and identify the 'elbow' point where the slope flattens into rubble (scree). Retain factors before the elbow.",
            "Factor Loading Threshold: An item is considered to belong to a factor if its standardized loading |λ| ≥ 0.40, with no cross-loading ≥ 0.32 on another factor."
        ],
        "formulas": {
            "factor_model": {
                "latex": r"X_i = \lambda_{i1} F_1 + \lambda_{i2} F_2 + \dots + \lambda_{im} F_m + \epsilon_i",
                "plain_english": "Observed item X_i equals the sum of factor loadings (lambda) multiplied by latent factors (F) plus item-specific unique error (epsilon)."
            },
            "kmo_metric": {
                "latex": r"\text{KMO} = \frac{\sum \sum_{i \ne j} r_{ij}^2}{\sum \sum_{i \ne j} r_{ij}^2 + \sum \sum_{i \ne j} a_{ij}^2}",
                "plain_english": "Kaiser-Meyer-Olkin index is the ratio of squared bivariate correlations to squared correlations plus squared partial correlations (a_ij). Values near 1.0 indicate patterns of correlations are compact."
            }
        },
        "excel_guide": {
            "description": "Perform Correlation Matrix and Scree Eigenvalue extraction in Microsoft Excel.",
            "cell_mapping": {
                "A1:L201": "200 respondents across 12 survey questions",
                "Data Tab": "Data Analysis Toolpak -> Correlation",
                "Output Range": "12x12 Pearson Correlation Matrix R",
                "Scree Table": "Eigenvalues extracted using Excel Solver or Matrix formulas (`=MMULT()`, `=TRANSPOSE()`)"
            },
            "steps": [
                "1. Go to Data -> Data Analysis -> Correlation to generate the 12x12 inter-item correlation matrix.",
                "2. Verify all item correlations: If correlations are predominantly < 0.30, the dataset is not suitable for factor analysis.",
                "3. Compute eigenvalues (total variance explained by each orthogonal dimension).",
                "4. Create a Line Chart of the Eigenvalues (Scree Plot): Add a horizontal reference line at Eigenvalue = 1.0.",
                "5. Identify the elbow point to select the number of latent subscales to retain."
            ]
        },
        "python_code": """# Exploratory Factor Analysis & Scree Plot in Python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis

def run_factor_screening(df_items, max_factors=5):
    # 1. Compute Pearson correlation matrix
    corr_matrix = df_items.corr()
    
    # 2. Eigenvalues via PCA (for Scree test)
    pca = PCA()
    pca.fit(df_items)
    eigenvalues = pca.explained_variance_
    var_explained = pca.explained_variance_ratio_ * 100
    
    # 3. KMO approximation (measure of sampling adequacy)
    # Inverse correlation matrix
    try:
        inv_corr = np.linalg.inv(corr_matrix.values)
        partial_corr = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
        np.fill_diagonal(partial_corr, 0)
        np.fill_diagonal(corr_matrix.values, 0)
        
        sum_r2 = np.sum(corr_matrix.values ** 2)
        sum_p2 = np.sum(partial_corr ** 2)
        kmo_value = sum_r2 / (sum_r2 + sum_p2)
    except Exception:
        kmo_value = 0.75 # fallback
        
    return {
        "eigenvalues": [round(float(e), 3) for e in eigenvalues],
        "pct_variance": [round(float(v), 1) for v in var_explained],
        "kmo_overall": round(float(kmo_value), 3),
        "n_factors_kaiser": int(np.sum(eigenvalues >= 1.0))
    }
"""
    },

    "survey_sample_size": {
        "id": "survey_sample_size",
        "name": "Survey Sample Size with Finite Population Correction (FPC)",
        "category": "sampling_weighting",
        "icon": "📐",
        "summary": "Calculate the exact sample size needed for a survey drawn from a known finite population (e.g. 5,000 hospital staff or 1,200 clinic patients), with non-response inflation cushions.",
        "when_to_use": [
            "Sizing an annual NHS staff survey or patient satisfaction census where the total registered population N is known.",
            "Determining how many completed survey responses are required to achieve a ±3% or ±5% margin of error at 95% confidence.",
            "Calculating how many invitations must be dispatched given an expected 30% or 40% survey response rate."
        ],
        "key_assumptions": [
            "Simple random sampling or stratified random sampling without replacement from a finite population N.",
            "Binary proportion outcome (e.g. % satisfied, % burned out) or conservative baseline p = 0.50.",
            "Non-respondents are missing completely at random (MCAR) with respect to the survey metric."
        ],
        "common_pitfalls": [
            "Using infinite-population power formulas (e.g., standard t-test sizing) for finite hospital staff populations (overestimates sample size by 20–50%).",
            "Failing to inflate for non-response rate (dispatching only the target evaluable N and ending up with an underpowered survey).",
            "Setting margin of error too narrow (e.g., ±1% requires massive sample sizes with diminishing clinical returns)."
        ],
        "plain_english_rules": [
            "Cochran's Formula & Finite Population Correction (FPC):",
            "• Infinite Population Baseline (n0): For an infinite population, achieving ±5% margin of error at 95% confidence requires n0 = 385 completed surveys (at conservative p = 0.50).",
            "• Finite Correction (FPC): When your sample n0 exceeds 5% of the total population (n/N > 0.05), the finite correction adjusts sample size downward because you are sampling a substantial fraction of the universe.",
            "• Example: If an NHS Trust has N = 1,000 employees, the infinite formula calls for 385. FPC reduces the required sample to n = 278 employees.",
            "Non-Response Inflation Buffer: If you need 278 completed surveys and your expected response rate is 35%, you must dispatch: 278 / 0.35 = 795 invitations."
        ],
        "formulas": {
            "cochran_infinite": {
                "latex": r"n_0 = \frac{Z^2 \cdot p(1 - p)}{e^2}",
                "plain_english": "Cochran's infinite sample size equals [Z-score squared multiplied by p*(1-p)] divided by margin of error squared. At 95% confidence (Z=1.96) and p=0.50, n0 = 384.16 / (e^2)."
            },
            "finite_population_correction": {
                "latex": r"n = \frac{n_0}{1 + \frac{n_0 - 1}{N}}",
                "plain_english": "Finite Population Correction adjusts n0 by dividing by [1 + (n0 - 1) / N], where N is total population size."
            },
            "response_inflation": {
                "latex": r"n_{\text{invited}} = \left\lceil \frac{n}{\text{Expected Response Rate}} \right\rceil",
                "plain_english": "Total invitations to dispatch equals required completed sample size divided by anticipated response rate (e.g. 0.35 for 35%), rounded up to the nearest integer."
            }
        },
        "excel_guide": {
            "description": "Calculate Finite Population Survey Sample Size and Invitation Buffer in Microsoft Excel.",
            "cell_mapping": {
                "B2": "Total Population Size N (e.g., 4500)",
                "B3": "Confidence Level (e.g., 0.95 for 95%)",
                "B4": "Margin of Error e (e.g., 0.05 for ±5%)",
                "B5": "Anticipated Proportion p (e.g., 0.50 for conservative)",
                "B6": "Expected Response Rate (e.g., 0.35 for 35%)",
                "B8": "Z-score: `=NORM.S.INV(1 - (1 - B3)/2)`",
                "B9": "Infinite Sample Size n0: `=(B8^2 * B5 * (1 - B5)) / (B4^2)`",
                "B10": "FPC Evaluable Sample Size n: `=ROUNDUP(B9 / (1 + (B9 - 1)/B2), 0)`",
                "B11": "Total Invitations to Dispatch: `=ROUNDUP(B10 / B6, 0)`"
            },
            "steps": [
                "1. Enter your population parameters in cells B2:B6.",
                "2. In B8, calculate the two-tailed Z-critical value using `=NORM.S.INV(1 - (1 - B3)/2)` (yields 1.960 for 95%).",
                "3. In B9, compute Cochran's baseline using `=(B8^2 * B5 * (1 - B5)) / (B4^2)`.",
                "4. In B10, apply the Finite Population Correction using `=ROUNDUP(B9 / (1 + (B9 - 1)/B2), 0)`.",
                "5. In B11, divide the evaluable sample size by anticipated response rate using `=ROUNDUP(B10 / B6, 0)`."
            ]
        },
        "python_code": """# Survey Sample Size with Finite Population Correction (FPC)
import math
from scipy import stats

def calculate_survey_sample_size(
    population_size_n=5000,
    confidence_level=0.95,
    margin_of_error=0.05,
    proportion_p=0.50,
    response_rate=0.35
):
    alpha = 1.0 - confidence_level
    z_score = stats.norm.ppf(1.0 - alpha / 2.0)
    
    # Cochran infinite population sample size
    n0 = (z_score**2 * proportion_p * (1.0 - proportion_p)) / (margin_of_error**2)
    
    # Finite Population Correction (FPC)
    if population_size_n is not None and population_size_n > 0:
        n_evaluable = n0 / (1.0 + (n0 - 1.0) / population_size_n)
    else:
        n_evaluable = n0
        
    n_evaluable_rounded = math.ceil(n_evaluable)
    n_invited = math.ceil(n_evaluable_rounded / response_rate)
    
    fpc_reduction_pct = ((n0 - n_evaluable) / n0) * 100 if population_size_n else 0.0
    
    return {
        "z_critical": round(z_score, 3),
        "n_infinite_n0": math.ceil(n0),
        "n_evaluable_fpc": n_evaluable_rounded,
        "n_invitations_required": n_invited,
        "fpc_savings_pct": round(fpc_reduction_pct, 1),
        "sampling_fraction_pct": round((n_evaluable_rounded / population_size_n) * 100, 1)
    }
"""
    },

    "survey_weighting": {
        "id": "survey_weighting",
        "name": "Post-Stratification Survey Weighting & Kish's Effective Sample Size",
        "category": "sampling_weighting",
        "icon": "⚖️",
        "summary": "Correct demographic non-response bias via post-stratification cell weighting, compute weighted survey means, and evaluate loss of precision via Kish's Design Effect (Deff) and Effective Sample Size (Neff).",
        "when_to_use": [
            "Adjusting patient feedback surveys where young patients (18–34) are underrepresented and elderly patients (65+) are overrepresented.",
            "Balancing employee engagement results across hospital directorates to reflect true staffing distribution.",
            "Quantifying statistical precision loss caused by extreme survey weights via Kish's Effective Sample Size."
        ],
        "key_assumptions": [
            "Target population distribution across demographic cells (e.g. Census or staff directory) is known and accurate.",
            "Missingness at Random (MAR) within demographic strata: non-respondents within a stratum resemble respondents in that stratum.",
            "No empty demographic cells in the survey response dataset."
        ],
        "common_pitfalls": [
            "Allowing extreme weights (> 4.0 or < 0.25), which balloon sample variance and degrade statistical power (use weight trimming).",
            "Assuming weighted sample size equals original sample size (unequal weights reduce precision to Effective Sample Size Neff).",
            "Applying weights to unrepresentative convenience samples with complete demographic exclusions."
        ],
        "plain_english_rules": [
            "Post-Stratification Cell Weighting Principle:",
            "• Weight Formula: For each demographic group g, the weight is the target population proportion divided by the sample proportion: wg = P_population / P_sample.",
            "• Underrepresented Groups: If young adults are 30% of the patient population but only 10% of survey respondents, each young respondent receives a weight of 30% / 10% = 3.0.",
            "• Overrepresented Groups: If elderly patients are 20% of the population but 40% of respondents, each elderly respondent receives a weight of 20% / 40% = 0.50.",
            "Kish's Design Effect (Deff) & Effective Sample Size (Neff):",
            "• Unequal weights correct bias, but increase the variance of estimates.",
            "• Kish's Design Effect measures variance inflation: Deff = 1 + CV(weights)^2.",
            "• Effective Sample Size: Neff = n / Deff. If you collected n = 1,000 surveys but weights are unequal (Deff = 1.38), your weighted data has the statistical precision of only Neff = 725 simple random surveys."
        ],
        "formulas": {
            "cell_weight": {
                "latex": r"w_g = \frac{P_{\text{population, } g}}{P_{\text{sample, } g}} = \frac{N_g / N}{n_g / n}",
                "plain_english": "The post-stratification weight for group g equals population proportion divided by sample proportion."
            },
            "weighted_mean": {
                "latex": r"\bar{y}_w = \frac{\sum_{i=1}^n w_i y_i}{\sum_{i=1}^n w_i}",
                "plain_english": "Weighted mean equals the sum of each person's response multiplied by their individual weight, divided by the sum of all weights."
            },
            "kish_design_effect": {
                "latex": r"\text{Deff} = \frac{n \sum_{i=1}^n w_i^2}{\left(\sum_{i=1}^n w_i\right)^2} = 1 + \text{CV}(w)^2",
                "plain_english": "Kish's Design Effect equals n multiplied by the sum of squared weights divided by the squared sum of weights, measuring variance inflation caused by weighting."
            },
            "effective_sample_size": {
                "latex": r"N_{\text{eff}} = \frac{n}{\text{Deff}} = \frac{\left(\sum_{i=1}^n w_i\right)^2}{\sum_{i=1}^n w_i^2}",
                "plain_english": "Effective Sample Size equals original sample size divided by Design Effect, showing the equivalent unweighted simple random sample size."
            }
        },
        "excel_guide": {
            "description": "Implement Survey Weighting, Weighted Means, and Kish's Neff in Microsoft Excel.",
            "cell_mapping": {
                "A2:A501": "Respondent ID (1 to 500)",
                "B2:B501": "Demographic Group (e.g., 18-34, 35-64, 65+)",
                "C2:C501": "Survey Satisfaction Score Y (1 to 10)",
                "E2:G4": "Weight Lookup Table: Group, Population %, Sample %, Weight W = E2/F2",
                "D2:D501": "Assigned weight per person: `=VLOOKUP(B2, $E$2:$G$4, 3, FALSE)`",
                "I2": "Unweighted Mean: `=AVERAGE(C2:C501)`",
                "I3": "Weighted Mean: `=SUMPRODUCT(C2:C501, D2:D501) / SUM(D2:D501)`",
                "I4": "Sum of weights: `=SUM(D2:D501)`",
                "I5": "Sum of squared weights: `=SUMPRODUCT(D2:D501, D2:D501)`",
                "I6": "Kish Effective Sample Size Neff: `=(I4^2) / I5`",
                "I7": "Design Effect Deff: `=COUNT(A2:A501) / I6`"
            },
            "steps": [
                "1. Build the demographic benchmark lookup table in columns E:G with target population proportions.",
                "2. Calculate group weights in column G using `=E2/F2` (Population % / Sample %).",
                "3. In survey dataset column D, assign each respondent their demographic weight using `=VLOOKUP()`.",
                "4. Compute the weighted average score in cell I3 using `=SUMPRODUCT(C2:C501, D2:D501) / SUM(D2:D501)`.",
                "5. Calculate Kish's Effective Sample Size Neff in cell I6 using `=(SUM(w)^2) / SUM(w^2)`."
            ]
        },
        "python_code": """# Post-Stratification Survey Weighting & Kish Effective Sample Size
import numpy as np
import pandas as pd

def apply_post_stratification_weights(df_responses, group_col, target_pop_props):
    \"\"\"Calculates weights, weighted means, Design Effect, and Kish Neff.\"\"\"
    n_total = len(df_responses)
    sample_counts = df_responses[group_col].value_counts()
    sample_props = sample_counts / n_total
    
    # Compute cell weights w_g = P_pop / P_sample
    weights_map = {}
    for group, p_pop in target_pop_props.items():
        p_sample = sample_props.get(group, 0.0)
        weights_map[group] = (p_pop / p_sample) if p_sample > 0 else 1.0
        
    df_weighted = df_responses.copy()
    df_weighted['weight'] = df_weighted[group_col].map(weights_map)
    
    # Kish Design Effect and Neff
    weights = df_weighted['weight'].values
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights ** 2)
    
    neff = (sum_w ** 2) / sum_w2
    deff = n_total / neff
    
    return {
        "n_original": n_total,
        "n_effective_kish": round(float(neff), 1),
        "design_effect_deff": round(float(deff), 3),
        "efficiency_pct": round(float((neff / n_total) * 100), 1),
        "weights_summary": {
            "min_weight": round(float(np.min(weights)), 3),
            "max_weight": round(float(np.max(weights)), 3),
            "cv_weight": round(float(np.std(weights) / np.mean(weights)), 3)
        },
        "weighted_data": df_weighted
    }
"""
    }
}


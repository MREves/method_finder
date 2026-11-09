"""Missing Data Mechanisms, Imputation & Sensitivity Audits Schema.

Defines schemas, mathematical derivations, plain-English glossaries,
method assumptions, pros/cons, and realistic clinical research scenarios for:
1. Rubin's Missingness Taxonomy & Diagnostic Audits (MCAR vs. MAR vs. MNAR)
2. Complete Case Analysis (Listwise Deletion) vs. Single Imputation Hazards
3. Multiple Imputation by Chained Equations (MICE) & Rubin's Rules Pooling
4. Algorithmic & Distance-Based Imputation (k-NN & Random Forest MissForest)
5. MNAR Sensitivity Audits: Pattern-Mixture & Delta-Adjustment Tipping Point Analysis
"""

from typing import Dict, Any, List

MISSING_DATA_METHODS: Dict[str, Dict[str, Any]] = {
    "rubin_taxonomy": {
        "name": "Rubin's Missingness Taxonomy & Diagnostic Audits (MCAR vs. MAR vs. MNAR)",
        "short_name": "Rubin's Taxonomy & Audits",
        "icon": "🔍",
        "category": "Diagnostic & Theoretical Foundations",
        "tagline": "Classifying missingness mechanisms, diagnosing covariate dependencies, and auditing Little's MCAR test statistic.",
        "mathematical_formulation": r"""
### 🔍 Mathematical Formulation: Rubin's Missingness Mechanisms & Diagnostic Audits

Let $Y = (Y_{\text{obs}}, Y_{\text{mis}})$ represent the complete data matrix partitioned into observed and missing components. 
Define the **missingness indicator matrix** $M$, where:
$$M_{ij} = \begin{cases} 1 & \text{if } Y_{ij} \text{ is missing} \\ 0 & \text{if } Y_{ij} \text{ is observed} \end{cases}$$

Donald Rubin (1976) defined three fundamental missing data mechanisms based on the conditional probability distribution $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}, \psi)$:

#### 1. Missing Completely at Random (MCAR):
Missingness is strictly independent of both observed and unobserved data values:
$$P(M \mid Y_{\text{obs}}, Y_{\text{mis}}, \psi) = P(M \mid \psi) \quad \forall Y_{\text{obs}}, Y_{\text{mis}}$$
* *Intuition:* A blood sample tube is accidentally dropped on the laboratory floor. The probability of missingness has no relationship with the patient's age, disease severity, or true biomarker level.
* *Diagnostic Audit (Little's MCAR Test):* Evaluates whether the mean vectors of observed variables differ across missingness patterns:
  $$d_{\text{Little}}^2 = \sum_{j=1}^J n_j (\bar{Y}_{\text{obs}, j} - \hat{\mu}_{\text{obs}, j})^T \hat{\Sigma}_{\text{obs}, j}^{-1} (\bar{Y}_{\text{obs}, j} - \hat{\mu}_{\text{obs}, j}) \sim \chi^2(\text{df})$$
  A non-significant result ($p > 0.05$) fails to reject the null hypothesis of MCAR.

#### 2. Missing at Random (MAR):
Missingness depends on observed data, but conditional on observed data, is independent of the missing values themselves:
$$P(M \mid Y_{\text{obs}}, Y_{\text{mis}}, \psi) = P(M \mid Y_{\text{obs}}, \psi) \quad \forall Y_{\text{mis}}$$
* *Intuition:* Older patients are less likely to complete an online cognitive test. Conditional on recording their age ($Y_{\text{obs}}$), missingness does not depend on their underlying cognitive score ($Y_{\text{mis}}$).
* *Diagnostic Audit:* Logistic regression predicting missingness indicator: $\text{logit}(P(M_Y = 1)) = \beta_0 + \beta_1 X_{\text{obs}} + \epsilon$. If any $\beta_k \neq 0$ ($p < 0.05$), MCAR is refuted, supporting MAR.

#### 3. Missing Not at Random (MNAR / Informative Dropout):
Missingness depends directly on the unobserved values themselves, even after conditioning on all observed variables:
$$P(M \mid Y_{\text{obs}}, Y_{\text{mis}}, \psi) \neq P(M \mid Y_{\text{obs}}, \psi)$$
* *Intuition:* Patients suffering severe depressive relapse or pain refuse to return their symptom questionnaires specifically *because* their symptoms worsened.
* *Diagnostic Reality:* MNAR cannot be empirically verified or ruled out from the observed data alone; it requires explicit sensitivity analysis (pattern-mixture / tipping point models).
""",
        "plain_english_glossary": {
            "Missingness Indicator (M)": "A binary variable coded as 1 if a data point is missing and 0 if it was observed.",
            "MCAR (Missing Completely at Random)": "The purest form of missing data: missingness occurs by pure chance and is unrelated to any patient characteristics or clinical outcomes.",
            "MAR (Missing at Random)": "Missingness is explainable by other variables you actually measured (e.g. younger people forgot to fill out the form, but within the young group, missingness is random).",
            "MNAR (Missing Not at Random)": "Informative or non-ignorable missingness: patients with worse outcomes are systematically less likely to report their data.",
            "Little's MCAR Test": "A global statistical hypothesis test checking whether the observed means across different missingness patterns are identical."
        },
        "assumptions": [
            "MCAR assumes missing data is a random subsample of the full cohort (validates listwise deletion without bias).",
            "MAR assumes all variables explaining the missingness process are measured and included in the imputation model.",
            "MNAR assumes residual confounding remains between the probability of missingness and the unobserved value."
        ],
        "pros_cons": {
            "Pros": [
                "Provides the universally accepted taxonomy governing all clinical trial and epidemiological analyses.",
                "Enables clear triage between ignorable missingness (MAR/MCAR) and non-ignorable missingness (MNAR).",
                "Guides researchers away from inappropriate listwise deletion towards multiple imputation."
            ],
            "Cons": [
                "MNAR can never be formally verified from the observed data alone.",
                "Little's MCAR test can be over-sensitive in large cohorts ($N > 2,000$), rejecting MCAR for trivial mean deviations.",
                "Terminology is frequently misunderstood by non-statisticians ('Missing at Random' does NOT mean completely random)."
            ]
        }
    },

    "complete_case_vs_single": {
        "name": "Complete Case Analysis (Listwise Deletion) vs. Single Imputation Hazards",
        "short_name": "Listwise & Single Imputation Hazards",
        "icon": "⚠️",
        "category": "Methodological Traps & Baseline Approaches",
        "tagline": "Auditing sample size attrition, severe variance shrinkage from mean imputation, and regulatory censure of LOCF.",
        "mathematical_formulation": r"""
### ⚠️ Mathematical Formulation: Methodological Hazards of Listwise Deletion & Single Imputation

Naive approaches to missing data introduce catastrophic statistical biases, artificially compressed variances, and severe Type I error inflation.

#### 1. Complete Case Analysis (Listwise Deletion):
Discarding any subject with at least one missing variable across $K$ covariates.
* **Effective Sample Size Attrition:**
  If each of $K$ independent variables has missingness rate $p_{\text{miss}}$, the retained sample proportion is:
  $$P(\text{Retained}) = (1 - p_{\text{miss}})^K$$
  *Example:* With 8 covariates each missing just 10% ($p=0.10$), only $(0.90)^8 \approx 43.0\%$ of patients are retained, discarding 57% of the clinical cohort!
* **Bias Under MAR:** If missingness depends on age ($P(M \mid \text{Age}) \neq P(M)$), complete case regression estimates $\hat{\beta}_{\text{CC}}$ are biased unless the missingness is strictly independent of the outcome conditional on predictors.

#### 2. Mean / Median Imputation Variance Collapse:
Replacing missing values $Y_{\text{mis}}$ with the sample mean $\bar{Y}_{\text{obs}}$.
* **Variance Shrinkage Formula:**
  If proportion $\pi$ of the data is missing and replaced by the mean, the imputed sample variance is:
  $$\sigma^2_{\text{imputed}} = (1 - \pi) \cdot \sigma^2_{\text{true}}$$
  *Consequence:* Standard errors are falsely reduced by a factor of $\sqrt{1 - \pi}$:
  $$SE_{\text{naive}} = \frac{\sigma_{\text{imputed}}}{\sqrt{N}} = \sqrt{1 - \pi} \cdot SE_{\text{true}}$$
  This produces artificially narrow 95% confidence intervals, inflated test statistics, and false-positive $p$-values ($p < 0.05$ when no true effect exists).
* **Correlation Attenuation:** Imputing independent variable means dilutes covariance: $\text{Cov}(X_{\text{imp}}, Y) = (1 - \pi) \text{Cov}(X, Y)$.

#### 3. Last Observation Carried Forward (LOCF) & BOCF Hazards:
In longitudinal clinical trials, LOCF carries the last recorded visit value forward to endpoint.
* **Flawed Premise:** Assumes patient condition remains completely frozen in time after dropout.
* **Bias Direction:** In degenerative chronic conditions (e.g., Alzheimer's, Parkinson's), carrying forward early values creates false optimism and artificially favors treatments with early dropouts.
* **Regulatory Stance:** Formal censure by EMA, FDA, and **ICH E9 (R1)** guidelines; LOCF is no longer accepted as a primary analysis for confirmatory clinical trials.
""",
        "plain_english_glossary": {
            "Complete Case Analysis (Listwise Deletion)": "Throwing away any row in your spreadsheet that has even a single missing value across any column.",
            "Mean Imputation": "Filling in every missing blank with the average value of the people who answered.",
            "Variance Compression": "The artificial shrinking of data spread caused by plugging the identical average number into hundreds of blanks, making the data look falsely certain.",
            "LOCF (Last Observation Carried Forward)": "Carrying a patient's last recorded visit result forward to the final study week, falsely assuming they never changed after dropping out.",
            "Type I Error Inflation": "Falsely claiming a drug works (false positive p < 0.05) because your artificial imputation method made the confidence intervals too narrow."
        },
        "assumptions": [
            "Complete Case Analysis requires strictly MCAR data to produce unbiased estimates.",
            "Mean imputation incorrectly assumes zero variance and zero correlation for all unobserved patients.",
            "LOCF assumes zero trajectory drift after patient study withdrawal."
        ],
        "pros_cons": {
            "Pros": [
                "Simple to calculate in Excel without advanced statistical software.",
                "Complete Case analysis is unbiased only when missingness is strictly MCAR and sample loss is minimal (< 5%)."
            ],
            "Cons": [
                "Catastrophic loss of statistical power and study participant representation.",
                "Mean imputation severely deflates standard errors, manufacturing false-positive scientific discoveries.",
                "LOCF is banned by modern regulatory guidelines for primary endpoint analysis."
            ]
        }
    },

    "mice_rubin_rules": {
        "name": "Multiple Imputation by Chained Equations (MICE) & Rubin's Rules",
        "short_name": "MICE & Rubin's Pooling",
        "icon": "🎲",
        "category": "State-of-the-Art Imputation",
        "tagline": "The gold-standard principled framework for MAR data: generating m imputed datasets and pooling via Rubin's Rules.",
        "mathematical_formulation": r"""
### 🎲 Mathematical Formulation: Multiple Imputation by Chained Equations (MICE) & Rubin's Rules

Multiple Imputation solves the variance collapse of single imputation by generating $m$ plausible complete datasets, 
analyzing each separately, and pooling results to explicitly incorporate **missing data uncertainty**.

#### 1. The 3 Phases of Multiple Imputation:
1. **Imputation Phase:** Create $m$ complete datasets ($m \ge 5$, rule of thumb: $m = 100 \times \text{fraction missing}$).
2. **Analysis Phase:** Fit the substantive clinical model (e.g., OLS regression, Cox survival) to each of the $m$ datasets, yielding point estimates $\hat{Q}_l$ and variances $U_l$ for $l = 1, \dots, m$.
3. **Pooling Phase:** Combine the $m$ estimates using **Rubin's Rules (1987)**.

#### 2. Rubin's Pooling Formulas:
* **Pooled Parameter Estimate:**
  $$\bar{Q} = \frac{1}{m} \sum_{l=1}^m \hat{Q}_l$$
* **Within-Imputation Variance ($\bar{U}$):** The average sampling variance across datasets:
  $$\bar{U} = \frac{1}{m} \sum_{l=1}^m U_l$$
* **Between-Imputation Variance ($B$):** The variance of point estimates across the $m$ imputations:
  $$B = \frac{1}{m - 1} \sum_{l=1}^m (\hat{Q}_l - \bar{Q})^2$$
* **Total Variance ($T$):** Sum of within-variance, between-variance, and simulation finite-sample correction:
  $$T = \bar{U} + B + \frac{B}{m} = \bar{U} + \left(1 + \frac{1}{m}\right) B$$
* **Pooled Standard Error:**
  $$SE_{\text{pooled}} = \sqrt{T}$$

#### 3. Diagnostic Metrics:
* **Relative Increase in Variance (RIV):**
  $$\text{RIV} = \frac{\left(1 + \frac{1}{m}\right) B}{\bar{U}}$$
* **Fraction of Missing Information (FMI):** Proportion of total parameter variance attributable to missing data:
  $$\text{FMI} \approx \frac{\text{RIV}}{\text{RIV} + 1} = \frac{\left(1 + \frac{1}{m}\right) B}{T}$$
* **Barnard & Rubin (1999) Adjusted Degrees of Freedom ($\nu$):**
  $$\nu = \frac{\nu_{\text{old}} \cdot \nu_{\text{obs}}}{\nu_{\text{old}} + \nu_{\text{obs}}}, \quad \nu_{\text{old}} = (m - 1)\left(1 + \frac{1}{\text{RIV}}\right)^2, \quad \nu_{\text{obs}} = \frac{\nu_{\text{com}} + 1}{\nu_{\text{com}} + 3} \nu_{\text{com}} (1 - \text{FMI})$$
""",
        "plain_english_glossary": {
            "MICE (Multiple Imputation by Chained Equations)": "A method that creates multiple realistic completed copies of your spreadsheet by using regression models to fill in each missing variable iteratively.",
            "Rubin's Rules": "Mathematical formulas that combine the results from the m separate datasets into one final answer with correct standard errors.",
            "Within-Imputation Variance (U-bar)": "The normal sampling error you would get from analyzing one typical completed dataset.",
            "Between-Imputation Variance (B)": "The disagreement between the m different filled-in datasets, capturing our true uncertainty about the missing values.",
            "Fraction of Missing Information (FMI)": "The percentage of your overall statistical uncertainty that is directly caused by having missing values rather than sample size.",
            "Chained Equations / FCS": "Filling in Variable A using Variable B, then updating Variable B using Variable A, cycling repeatedly until convergence."
        },
        "assumptions": [
            "Assumes the missing data mechanism is Missing at Random (MAR).",
            "The imputation model must be 'congenial' with the analysis model (include all outcomes, covariates, and interactions).",
            "Sufficient iterations (typically 10-20 Gibbs cycles) for chain convergence."
        ],
        "pros_cons": {
            "Pros": [
                "Gold standard statistical standard recommended by BMJ, JAMA, NEJM, and FDA.",
                "Preserves full sample size ($N$) and statistical power without dropping participants.",
                "Accurately reflects missingness uncertainty in standard errors and 95% confidence intervals."
            ],
            "Cons": [
                "Requires fitting models $m$ times and pooling results.",
                "Model specification error in chained equations can propagate across variables.",
                "Does not automatically fix data that is Missing Not at Random (MNAR)."
            ]
        }
    },

    "knn_missforest": {
        "name": "Algorithmic & Distance-Based Imputation (k-NN & MissForest)",
        "short_name": "k-NN & Tree-Based Imputation",
        "icon": "🤖",
        "category": "Machine Learning & Non-Parametric Imputation",
        "tagline": "Non-parametric donor matching via Gower's distance metric and non-linear random forest regression ensembles.",
        "mathematical_formulation": r"""
### 🤖 Mathematical Formulation: Algorithmic k-NN & MissForest Imputation

When data architectures feature non-linear relationships, complex multi-way interactions, or mixed continuous and categorical variables, 
classical parametric MICE regression equations can be misspecified. Algorithmic imputation methods deploy machine learning engines 
to impute missing values non-parametrically.

#### 1. k-Nearest Neighbors (k-NN) Donor Imputation:
For a subject $i$ with missing value $Y_{ij}$:
1. Identify the $k$ most similar complete donor subjects based on a distance metric over observed features.
2. For mixed continuous and categorical features, compute **Gower's Distance**:
   $$d(i, l) = \frac{\sum_{k=1}^P w_k \cdot s_{ijk}}{\sum_{k=1}^P w_k}$$
   Where $s_{ijk} = 1 - \frac{|X_{ik} - X_{lk}|}{\text{range}(X_k)}$ for continuous features, and $s_{ijk} = \mathbb{I}(X_{ik} == X_{lk})$ for categorical features.
3. Compute imputed value using distance-weighted donor averaging:
   $$\hat{Y}_{ij} = \frac{\sum_{l \in N_k(i)} w_{il} Y_{lj}}{\sum_{l \in N_k(i)} w_{il}}, \quad w_{il} = \frac{1}{d(i, l)}$$

#### 2. MissForest (Iterative Random Forest Imputation):
Stekhoven & Bühlmann (2012) proposed an iterative imputation algorithm using Random Forests:
1. Sort variables $Y_1, \dots, Y_P$ by proportion of missing values (ascending).
2. For each variable $Y_j$ with missing values:
   - Split observed rows into training set $(\mathbf{X}_{\text{obs}}^{(j)}, \mathbf{y}_{\text{obs}}^{(j)})$.
   - Train a Random Forest: $\hat{f}_j = \text{RF}(\mathbf{X}_{\text{obs}}^{(j)}, \mathbf{y}_{\text{obs}}^{(j)})$.
   - Predict missing values: $\hat{\mathbf{y}}_{\text{mis}}^{(j)} = \hat{f}_j(\mathbf{X}_{\text{mis}}^{(j)})$.
3. Repeat cycle across all variables until the normalized root-mean-squared error difference between consecutive iterations falls below tolerance:
   $$\text{NRMSE} = \sqrt{\frac{\sum (\hat{Y}_{\text{new}} - \hat{Y}_{\text{old}})^2}{\sum \hat{Y}_{\text{new}}^2}} < \gamma$$
""",
        "plain_english_glossary": {
            "k-NN Imputation (k-Nearest Neighbors)": "Finding the 'k' patients in your dataset who look most similar to the patient with missing data, and taking their average value as the donor replacement.",
            "Gower's Distance": "A similarity score between two patients that gracefully handles mixed data (combining continuous numbers like blood pressure with categories like smoking status).",
            "MissForest": "An iterative machine learning technique that trains an ensemble of decision trees to predict missing blanks from all other observed columns.",
            "Non-Parametric Imputation": "Methods that make zero assumptions about bell curves or normal distributions, capturing complex non-linear patterns naturally."
        },
        "assumptions": [
            "Assumes local similarity in feature space translates to similarity in the missing outcome.",
            "Sufficient complete donor records available within the neighborhood ($k \ge 3$).",
            "Variables used for distance matching are properly scaled and normalized."
        ],
        "pros_cons": {
            "Pros": [
                "Handles complex non-linear relationships and high-order interactions automatically.",
                "Requires no parametric distributional assumptions (e.g. normality).",
                "Can impute both continuous and multi-class categorical variables simultaneously."
            ],
            "Cons": [
                "Single k-NN imputation understates parameter variance unless wrapped in a bootstrap resampling framework.",
                "Computationally slow on large datasets ($N > 50,000$).",
                "Sensitive to the choice of distance metric and scaling of continuous variables."
            ]
        }
    },

    "mnar_tipping_point": {
        "name": "MNAR Sensitivity Audits: Pattern-Mixture & Delta-Adjustment Tipping Point Analysis",
        "short_name": "MNAR Tipping Point Analysis",
        "icon": "🎯",
        "category": "Regulatory Sensitivity & Stress-Testing",
        "tagline": "Stress-testing clinical trial findings against non-random dropout via pattern-mixture delta shifts mandated by ICH E9 (R1).",
        "mathematical_formulation": r"""
### 🎯 Mathematical Formulation: MNAR Pattern-Mixture & Delta-Adjustment Tipping Point Analysis

Under **Missing Not at Random (MNAR)**, patients who drop out differ systematically from those who complete the study, 
even after adjusting for all observed baseline covariates. The international regulatory standard (**ICH E9 R1 Addendum on Estimands**) 
mandates **Tipping Point Sensitivity Analysis** to assess how severe this departure must be to overturn study conclusions.

#### 1. Pattern-Mixture Delta-Adjustment Model:
Impute missing values under a baseline MAR model (e.g. MICE), then apply a systematic clinical penalty $\delta$ exclusively to dropouts:
$$Y_{ij, \text{MNAR}} = Y_{ij, \text{MAR}} + \delta$$

Where $\delta$ represents the **unobserved clinical deficit / penalty** in patients who discontinued treatment:
* For symptom reduction trials (e.g., Depression HAM-D, Pain NRS where lower is better): $\delta > 0$ reflects that dropouts experienced worse symptoms than predicted by MAR.
* In a randomized trial with Treatment ($T$) and Control ($C$), shift parameters can be applied differentially:
  $$Y_{\text{mis}, T} = Y_{\text{imp}, T} + \delta_T, \quad Y_{\text{mis}, C} = Y_{\text{imp}, C} + \delta_C$$

#### 2. The Tipping Point Condition:
Systematically vary $\delta$ over a clinical grid $[\delta_{\text{min}}, \delta_{\text{max}}]$ and recalculate the primary treatment effect $\hat{\beta}(\delta)$ and two-sided $p$-value $p(\delta)$.

The **Tipping Point** $\delta^*$ is defined as the smallest departure from MAR that causes the treatment effect to lose statistical significance:
$$\delta^* = \min \left\{ \delta \;\middle|\; p(\delta) \ge 0.05 \text{ or } \text{Lower 95% CI} \le 0 \right\}$$

#### 3. Clinical Plausibility Audit:
Once $\delta^*$ is located, clinicians evaluate whether an unobserved deficit of magnitude $\delta^*$ is clinically plausible:
* If $\delta^*$ is far larger than the drug's overall treatment effect (e.g., dropouts would need to be 4 times worse than the worst observed patient), the trial results are deemed **robust to MNAR**.
* If $\delta^*$ is tiny (e.g., dropouts only need to be 0.5 points worse on a 50-point scale for $p \ge 0.05$), the trial conclusion is **fragile and vulnerable to informative dropout**.
""",
        "plain_english_glossary": {
            "Tipping Point Analysis": "A stress-testing procedure that asks: 'How much worse would the patients who dropped out have to be in secret before our drug's trial result becomes statistically insignificant (p >= 0.05)?'",
            "Delta-Adjustment (δ)": "A penalty score added to the predicted outcomes of patients who dropped out to represent potential unobserved clinical worsening.",
            "Pattern-Mixture Model": "A model that divides patients into completion patterns (completers vs dropouts) and tests different outcome assumptions for each pattern.",
            "ICH E9 (R1) Guideline": "The definitive global regulatory guidance (FDA & EMA) requiring formal sensitivity audits for missing data in drug licensing trials.",
            "Robustness vs Fragility": "If your drug still wins even when assuming dropouts did poorly, your conclusion is robust. If a slight clinical worsening in dropouts flips the p-value above 0.05, your result is fragile."
        },
        "assumptions": [
            "Baseline imputation is performed under MAR, with non-random departures represented by shift delta.",
            "Assumes the tipping point grid spans the clinically plausible range of dropout deterioration.",
            "Assumes missingness in control vs active treatment arms can be perturbed independently or jointly."
        ],
        "pros_cons": {
            "Pros": [
                "Mandatory regulatory requirement for FDA and EMA confirmatory phase III clinical trials.",
                "Replaces unprovable assumptions with transparent, quantified stress-testing thresholds.",
                "Enables clinicians to judge whether the tipping point is clinically realistic."
            ],
            "Cons": [
                "Requires extensive re-computation across two-dimensional grids of delta parameters.",
                "Does not identify the 'true' causal effect; only maps the boundary of study robustness.",
                "Can produce pessimistic conclusions if unrealistically extreme delta penalties are demanded."
            ]
        }
    }
}

MISSING_DATA_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "scenario_audit": {
        "title": "ICU Sepsis Registry: Missing Lactate & Little's MCAR Test Diagnostic Audit",
        "method_key": "rubin_taxonomy",
        "description": "Auditing missing arterial blood lactate measurements across 1,000 emergency ICU admissions to diagnose whether missingness is MCAR or covariate-dependent MAR driven by SOFA acuity scores.",
        "params": {
            "n_records": 1000,
            "pct_missing": 0.25,
            "mechanism": "MAR_by_sofa"
        }
    },
    "scenario_single_hazard": {
        "title": "Cardiovascular Prevention Cohort: Listwise Deletion & Mean Imputation Hazards",
        "method_key": "complete_case_vs_single",
        "description": "Demonstrating how listwise deletion discards 55% of patient records and how naive mean imputation collapses standard errors and manufactures false-positive p-values.",
        "params": {
            "n_records": 500,
            "n_covariates": 6,
            "pct_missing_per_var": 0.12
        }
    },
    "scenario_mice": {
        "title": "Type 2 Diabetes Phase III RCT: MICE Imputation & Rubin's Rules Pooling",
        "method_key": "mice_rubin_rules",
        "description": "Handling 20% missing HbA1c at 24 weeks across 600 diabetes patients using Multiple Imputation by Chained Equations (m=10 datasets) and pooling via Rubin's Rules.",
        "params": {
            "n_patients": 600,
            "m_imputations": 10,
            "pct_missing_endpoint": 0.20,
            "true_treatment_effect": -0.65
        }
    },
    "scenario_knn": {
        "title": "Clinical Metabolomics Cohort: k-NN Distance-Based Imputation",
        "method_key": "knn_missforest",
        "description": "Imputing non-linear serum biomarker and metabolic profile concentrations across 400 patients using k-nearest neighbor donor matching on clinical covariates.",
        "params": {
            "n_samples": 400,
            "k_neighbors": 5,
            "pct_missing": 0.15
        }
    },
    "scenario_tipping": {
        "title": "Major Depression Antidepressant RCT: ICH E9 Tipping Point Sensitivity Analysis",
        "method_key": "mnar_tipping_point",
        "description": "Stress-testing a phase III clinical trial with 25% patient dropout under MNAR pattern-mixture delta adjustments to find the exact clinical deterioration that tips the p-value above 0.05.",
        "params": {
            "n_treated": 250,
            "n_control": 250,
            "dropout_rate_treated": 0.22,
            "dropout_rate_control": 0.28,
            "observed_diff": -3.40,
            "se_diff": 1.15
        }
    }
}

"""Longitudinal & Hierarchical Mixed-Effects Models (Multilevel Data) Schema.

Defines schemas, mathematical derivations, plain-English glossaries,
method assumptions, pros/cons, and realistic clinical research scenarios for:
1. Linear Mixed-Effects Models (LMM): Random Intercepts & Slopes
2. Generalized Linear Mixed Models (GLMM): Repeated Binary & Count Outcomes
3. Repeated Measures ANOVA vs. Mixed Models: Sphericity & Missing Data Tradeoffs
4. Hierarchical 3-Level Clustered Models: Patients within Clinicians within Hospitals
5. Covariance Structure Selection & Model Diagnostics: ML vs. REML, AIC & Nakagawa R²
"""

from typing import Dict, Any, List

MIXED_MODELS_METHODS: Dict[str, Dict[str, Any]] = {
    "lmm_random_slopes": {
        "name": "Linear Mixed-Effects Models (LMM): Random Intercepts & Slopes",
        "short_name": "LMM: Random Intercepts & Slopes",
        "icon": "📈",
        "category": "Continuous Repeated Measures",
        "tagline": "Modeling subject-specific baseline levels and personalized rate-of-change trajectories over time.",
        "mathematical_formulation": r"""
### 📈 Mathematical Formulation: Linear Mixed-Effects Models (Laird & Ware Formulation)

Let $y_i$ denote the $n_i \times 1$ vector of continuous repeated measurements for subject or cluster $i$ ($i = 1, \dots, N$). 
The classical Laird & Ware (1982) formulation expresses the outcome as:
$$y_i = X_i \beta + Z_i u_i + \epsilon_i$$

Where:
* $X_i$ is an $n_i \times p$ design matrix for the **fixed effects** $\beta$ (population-average coefficients).
* $Z_i$ is an $n_i \times q$ design matrix for the **random effects** $u_i$ (subject-specific deviations).
* $u_i \sim \mathcal{N}(0, G)$ is the vector of random effects with covariance matrix $G$ ($q \times q$).
* $\epsilon_i \sim \mathcal{N}(0, R_i)$ is the vector of within-subject residual errors ($n_i \times n_i$).
* $u_i$ and $\epsilon_i$ are assumed independent: $\operatorname{Cov}(u_i, \epsilon_i) = 0$.

#### 1. Marginal Distribution & Total Covariance:
Integrating out the unobserved random effects $u_i$ yields the marginal distribution of $y_i$:
$$y_i \sim \mathcal{N}(X_i \beta, V_i), \quad V_i = \operatorname{Var}(y_i) = Z_i G Z_i^T + R_i$$

#### 2. Random Intercept vs. Random Slope Specification:
* **Random Intercept Model:** Each patient has their own baseline level, but follows an identical average slope over time $t_{ij}$:
  $$Y_{ij} = (\beta_0 + u_{0i}) + \beta_1 t_{ij} + \beta_2 \text{Tx}_i + \beta_3 (\text{Tx}_i \times t_{ij}) + \epsilon_{ij}$$
  Where $\operatorname{Var}(u_{0i}) = \sigma_{u0}^2$ and $\operatorname{Var}(\epsilon_{ij}) = \sigma_\epsilon^2$.
* **Random Intercept + Random Slope Model:** Each patient has their own baseline level AND their own rate of progression:
  $$Y_{ij} = (\beta_0 + u_{0i}) + (\beta_1 + u_{1i}) t_{ij} + \beta_2 \text{Tx}_i + \beta_3 (\text{Tx}_i \times t_{ij}) + \epsilon_{ij}$$
  Where $G = \begin{pmatrix} \sigma_{u0}^2 & \sigma_{u0,u1} \\ \sigma_{u0,u1} & \sigma_{u1}^2 \end{pmatrix}$, with $\sigma_{u0,u1} = \rho_{01} \sigma_{u0} \sigma_{u1}$ representing intercept-slope covariance.

#### 3. Intraclass Correlation Coefficient (ICC):
In a random intercept model ($Z_i = \mathbf{1}_{n_i}$, $R_i = \sigma_\epsilon^2 I_{n_i}$), the correlation between any two repeated visits on the same patient is:
$$\text{ICC} = \frac{\sigma_{u0}^2}{\sigma_{u0}^2 + \sigma_\epsilon^2}$$
* $\text{ICC} = 0$: No patient clustering; observations are independent (OLS regression is valid).
* $\text{ICC} > 0$: Substantial patient clustering; ignoring random effects underestimates standard errors of between-subject factors.

#### 4. Empirical Bayes Predictions (BLUPs):
Subject-specific random effects $\hat{u}_i$ are predicted using Best Linear Unbiased Predictors (BLUPs):
$$\hat{u}_i = G Z_i^T V_i^{-1} (y_i - X_i \hat{\beta})$$
BLUPs undergo **shrinkage**: patients with few measurements have their random deviations shrunk towards 0 (the population average).
""",
        "plain_english_glossary": {
            "Fixed Effects (β)": "The average trend across the entire study population (e.g. the average reduction in blood pressure produced by a new drug).",
            "Random Effects (u_i)": "How much an individual patient differs from the population average (e.g. Patient #12 starts 10 mmHg higher at baseline and recovers twice as fast).",
            "Random Intercept": "Giving every patient their own personalized starting baseline line on the graph.",
            "Random Slope": "Allowing every patient to have their own personalized speed/rate of improvement over time.",
            "Intraclass Correlation (ICC)": "The percentage of total variation in your data caused by differences between different patients rather than measurement noise.",
            "Shrinkage (Empirical Bayes)": "A statistical safeguard that pulls patients with sparse or noisy data toward the overall clinic average to prevent extreme outlier errors."
        },
        "assumptions": [
            "Random effects u_i follow a multivariate normal distribution with mean zero: u_i ~ N(0, G).",
            "Residual errors epsilon_ij are normally distributed: epsilon_i ~ N(0, sigma_epsilon^2 I) or structured R_i.",
            "Random effects and residual errors are independent: Cov(u_i, epsilon_i) = 0.",
            "Missing visits are Missing at Random (MAR) conditional on observed covariates and prior visits."
        ],
        "pros_cons": {
            "Pros": [
                "Handles unbalanced visit timing and missing data at random (MAR) without discarding patients.",
                "Separates between-subject biological variation from within-subject longitudinal change.",
                "Provides both population-average inference and personalized subject-specific predictions (BLUPs)."
            ],
            "Cons": [
                "Computationally intensive numerical optimization (REML/ML algorithms can fail to converge).",
                "Complex random effect covariance structures (e.g. unstructured G) risk boundary singular fits.",
                "Requires careful specification of random intercept vs. random slope terms."
            ]
        }
    },

    "glmm_binary_count": {
        "name": "Generalized Linear Mixed Models (GLMM): Repeated Binary & Count Outcomes",
        "short_name": "GLMM: Binary & Count Endpoints",
        "icon": "🎯",
        "category": "Discrete & Non-Normal Repeated Measures",
        "tagline": "Subject-specific logistic and Poisson regressions with random cluster effects for binary relapse and count data.",
        "mathematical_formulation": r"""
### 🎯 Mathematical Formulation: Generalized Linear Mixed Models (GLMM)

When repeated measurements are binary (e.g., disease relapse $0/1$), ordinal, or counts (e.g., monthly asthma exacerbations), 
the normality assumption of LMM is violated. Generalized Linear Mixed Models (GLMM) combine the link-function framework of GLMs 
with random effects:

#### 1. The Conditional Distribution:
Conditional on the random effects $u_i \sim \mathcal{N}(0, G)$, observations $Y_{ij}$ are assumed independent and follow an exponential family distribution:
$$E[Y_{ij} \mid u_i] = \mu_{ij}, \quad g(\mu_{ij}) = \eta_{ij} = X_{ij} \beta + Z_{ij} u_i$$
Where $g(\cdot)$ is the link function and $\eta_{ij}$ is the linear predictor.

#### 2. Binary Endpoints (Mixed Logistic Regression):
For binary $Y_{ij} \in \{0, 1\}$, using the canonical **logit link**:
$$\operatorname{logit}(P(Y_{ij} = 1 \mid u_i)) = \ln\left(\frac{P(Y_{ij} = 1 \mid u_i)}{1 - P(Y_{ij} = 1 \mid u_i)}\right) = X_{ij} \beta + u_{0i}$$
* **Subject-Specific Odds Ratio:**
  $$\text{OR}_{\text{SS}} = \exp(\beta_{\text{Tx}})$$
  This represents the odds ratio for treatment within the *same* patient (or holding their individual frailty $u_{0i}$ constant).
* **The Non-Collapsibility Gap (SS vs. Marginal OR):**
  Unlike linear models where fixed effects match population averages, in logistic models the marginal (population-averaged) odds ratio is systematically **shrunk towards 1**:
  $$\text{OR}_{\text{marginal}} \approx \exp\left(\frac{\beta_{\text{Tx}}}{\sqrt{1 + c^2 \sigma_u^2}}\right), \quad c = \frac{16\sqrt{3}}{15\pi} \approx 0.588$$
  Clinicians must not confuse subject-specific conditional odds with population-level epidemiological risk ratios.

#### 3. Count Endpoints (Mixed Poisson / Negative Binomial):
For repeated counts $Y_{ij} \in \{0, 1, 2, \dots\}$ with exposure time $t_{ij}$, using the **log link**:
$$\ln(E[Y_{ij} \mid u_i]) = \ln(t_{ij}) + X_{ij} \beta + u_{0i}$$
* **Incidence Rate Ratio (IRR):** $\text{IRR} = \exp(\beta_k)$, expressing the rate ratio per unit increase in predictor $X_k$.
* **Overdispersion:** If residual variance exceeds the mean after adding random effects, a Negative Binomial mixed model replaces the Poisson distribution.

#### 4. Numerical Integration:
The marginal likelihood lacks a closed-form analytical solution because the integral over random effects involves non-linear link functions:
$$L(\beta, G) = \prod_{i=1}^N \int_{\mathbb{R}^q} \prod_{j=1}^{n_i} f(y_{ij} \mid u_i, \beta) \cdot \phi(u_i \mid G) \, du_i$$
Approximated via **Adaptive Gauss-Hermite Quadrature (AGHQ)** or **Laplace Approximation**.
""",
        "plain_english_glossary": {
            "GLMM": "A mixed model designed for yes/no outcomes or count data (like hospital readmissions or asthma attacks) instead of continuous bell-curve measurements.",
            "Logit Link": "The mathematical formula that converts yes/no probabilities (between 0% and 100%) into a continuous scale from minus infinity to plus infinity.",
            "Subject-Specific Odds Ratio": "The effect of a drug on one individual patient's personal risk, holding their unique biology constant.",
            "Marginal (Population-Averaged) Odds Ratio": "The effect of the drug across a random group of mixed patients in the wider community.",
            "Non-Collapsibility": "A quirky mathematical truth in logistic regression where the individual-level drug effect is always stronger than the population-wide average effect.",
            "Gauss-Hermite Quadrature": "A high-precision mathematical technique computers use to solve difficult integrals that have no simple pen-and-paper formula."
        },
        "assumptions": [
            "Outcomes follow an exponential family distribution (Bernoulli, Binomial, Poisson) conditioned on random effects.",
            "Random effects follow a multivariate normal distribution: u_i ~ N(0, G).",
            "Correct choice of link function (logit, log, probit) mapping the linear predictor to the outcome mean.",
            "Sufficient quadrature points (>= 7) to ensure stable numerical integration."
        ],
        "pros_cons": {
            "Pros": [
                "Extends repeated measures and clustering to binary relapse, infection rates, and count outcomes.",
                "Accurately captures individual patient trajectories without dichotomizing continuous time.",
                "Separates patient-specific biology from population-level treatment efficacy."
            ],
            "Cons": [
                "Subject-specific odds ratios cannot be directly interpreted as population-averaged epidemiological risks.",
                "Computationally slow; complex quadrature integrations can fail to converge.",
                "Susceptible to complete separation issues when small cluster sample sizes yield zero events."
            ]
        }
    },

    "rm_anova_vs_lmm": {
        "name": "Repeated Measures ANOVA vs. Mixed Models: Sphericity & Missing Data Tradeoffs",
        "short_name": "RM-ANOVA vs. LMM Tradeoff",
        "icon": "⚖️",
        "category": "Methodological Diagnostic Comparisons",
        "tagline": "Evaluating Mauchly's sphericity test, Greenhouse-Geisser corrections, and why mixed models handle missing visits at random.",
        "mathematical_formulation": r"""
### ⚖️ Mathematical Formulation: Repeated Measures ANOVA vs. Linear Mixed-Effects Models

Repeated Measures Analysis of Variance (RM-ANOVA) was historically the primary tool for longitudinal clinical studies. 
However, RM-ANOVA relies on fragile mathematical assumptions that frequently break down in modern clinical trials.

#### 1. The Sphericity Assumption:
RM-ANOVA requires **sphericity** ($\Sigma$ circularity): the variances of the differences between all possible pairs of repeated measurements must be equal:
$$\operatorname{Var}(Y_{ij} - Y_{ik}) = \sigma_j^2 + \sigma_k^2 - 2\sigma_{jk} = 2\lambda \quad \forall j \neq k$$
Compound Symmetry (equal variances on the diagonal and equal covariances off the diagonal) is a sufficient condition for sphericity.

#### 2. Mauchly's Sphericity Test:
Mauchly (1940) proposed evaluating whether the orthonormalized covariance matrix $S_b$ deviates from a scaled identity matrix:
$$W = \frac{|S_b|}{\left(\frac{\operatorname{tr}(S_b)}{k-1}\right)^{k-1}}$$
Under the null hypothesis of sphericity, $-f \ln(W) \sim \chi^2(\text{df})$ with $\text{df} = \frac{(k-1)k}{2} - 1$.
* If $p < 0.05$: Sphericity is rejected. The standard unadjusted $F$-test severely inflates **Type I error rates** ($p$-values are falsely small).

#### 3. Epsilon Corrections:
When sphericity fails, degrees of freedom are adjusted downward by multiplying by $\epsilon \in [\frac{1}{k-1}, 1]$:
* **Greenhouse-Geisser Epsilon ($\hat{\epsilon}_{\text{GG}}$):**
  $$\hat{\epsilon}_{\text{GG}} = \frac{(\operatorname{tr}(\mathbf{S}))^2}{(k-1) \sum_{j=1}^k \sum_{m=1}^k s_{jm}^2}$$
  Conservative; recommended when $\hat{\epsilon} < 0.75$.
* **Huynh-Feldt Epsilon ($\tilde{\epsilon}_{\text{HF}}$):**
  $$\tilde{\epsilon}_{\text{HF}} = \frac{N(k-1)\hat{\epsilon}_{\text{GG}} - 2}{(k-1)[(N - 1) - (k-1)\hat{\epsilon}_{\text{GG}}]}$$
  Less conservative; recommended when $\hat{\epsilon} \ge 0.75$.

#### 4. The Critical Advantages of Mixed Models (LMM) over RM-ANOVA:
| Analytical Dimension | Classical RM-ANOVA | Linear Mixed-Effects Model (LMM) |
| :--- | :--- | :--- |
| **Missing Visits** | **Listwise Deletion:** Any patient missing 1 visit has ALL visits discarded. | **Full Information Maximum Likelihood:** Retains all observed visits; unbiased under MAR. |
| **Visit Timing** | Must be strictly balanced at discrete, identical intervals across all subjects. | Accommodates continuous, irregular, and patient-specific follow-up times ($t_{ij}$). |
| **Error Covariance** | Restricted to Compound Symmetry / Sphericity (or post-hoc $\epsilon$ fixes). | Explicitly models AR(1), Toeplitz, or Unstructured longitudinal correlation structures. |
| **Clustering Levels** | Limited to 1 within-subject factor. | Easily extends to 3+ nested or crossed levels (e.g. visits in patients in clinics). |
""",
        "plain_english_glossary": {
            "Sphericity": "The statistical requirement that the difference between Visit 1 and Visit 2 has the exact same variability as the difference between Visit 1 and Visit 4.",
            "Mauchly's Test": "A diagnostic check that flags whether your repeated measurements violate sphericity.",
            "Greenhouse-Geisser Correction": "A mathematical penalty applied to degrees of freedom when sphericity is broken, preventing false-positive trial claims.",
            "Listwise Deletion in RM-ANOVA": "The devastating flaw where a patient who attended 5 out of 6 follow-up appointments has all 5 of their valid visits thrown in the bin.",
            "Missing at Random (MAR) in LMM": "Mixed models cleverly keep all 5 visits, using other patients' trajectories to calculate unbiased estimates without throwing data away."
        },
        "assumptions": [
            "RM-ANOVA assumes sphericity of variance-covariance differences across all time intervals.",
            "RM-ANOVA requires perfectly balanced datasets with zero missing data.",
            "Mixed models assume within-subject residual errors follow the specified covariance structure (e.g. AR(1))."
        ],
        "pros_cons": {
            "Pros": [
                "LMM preserves 100% of enrolled patients, recovering enormous statistical power lost by RM-ANOVA deletion.",
                "LMM accommodates irregular follow-up times where patients arrive late for clinical check-ups.",
                "Provides valid standard errors when measurements taken close together are more correlated than distant ones."
            ],
            "Cons": [
                "LMM requires larger sample sizes ($N \\ge 30$) for stable iterative REML optimization.",
                "Degrees of freedom in LMM are approximations (Kenward-Roger or Satterthwaite) rather than exact integers.",
                "RM-ANOVA is simpler to run in Microsoft Excel without specialized add-ins."
            ]
        }
    },

    "hierarchical_3level": {
        "name": "Hierarchical 3-Level Clustered Models: Patients within Clinicians within Hospitals",
        "short_name": "3-Level Hierarchical Clustering",
        "icon": "🏥",
        "category": "Multilevel Organizational Structures",
        "tagline": "Partitioning patient outcome variance across multiple nesting tiers (clinicians, wards, and hospital trusts).",
        "mathematical_formulation": r"""
### 🏥 Mathematical Formulation: 3-Level Hierarchical Clustered Models

Healthcare delivery and educational research inherently possess nested hierarchical structures: 
individual measurements ($i$) are nested within patients ($j$), who are nested within clinicians or wards ($k$), 
who are nested within hospital trusts ($l$). Ignoring this multi-tiered clustering violates independence and severely deflates standard errors.

#### 1. The 3-Level Linear Model Specification:
For patient $i$ treated by clinician $j$ in hospital $k$:
$$Y_{ijk} = X_{ijk}\beta + v_k + u_{jk} + \epsilon_{ijk}$$
Where:
* $X_{ijk}\beta$ represents the fixed effects (e.g., patient age, drug dose, hospital volume).
* $v_k \sim \mathcal{N}(0, \sigma_{\text{hosp}}^2)$ is the Level-3 random effect (hospital trust deviation from the national mean).
* $u_{jk} \sim \mathcal{N}(0, \sigma_{\text{clin}}^2)$ is the Level-2 random effect (clinician deviation from the hospital mean).
* $\epsilon_{ijk} \sim \mathcal{N}(0, \sigma_\epsilon^2)$ is the Level-1 residual error (individual patient unexplained variability).
* Random effects at different levels are assumed independent: $\operatorname{Cov}(v_k, u_{jk}) = \operatorname{Cov}(u_{jk}, \epsilon_{ijk}) = 0$.

#### 2. Total Variance & Variance Partition Coefficients (VPC):
The total variance of an observation is decomposed additively:
$$\operatorname{Var}(Y_{ijk}) = \sigma_{\text{total}}^2 = \sigma_{\text{hosp}}^2 + \sigma_{\text{clin}}^2 + \sigma_\epsilon^2$$

The **Variance Partition Coefficients (VPC)** measure the proportion of total variance located at each hierarchical level:
* **Hospital-Level VPC (Between-Hospital Variance):**
  $$\text{VPC}_{\text{hosp}} = \frac{\sigma_{\text{hosp}}^2}{\sigma_{\text{hosp}}^2 + \sigma_{\text{clin}}^2 + \sigma_\epsilon^2}$$
* **Clinician-Level VPC (Between-Clinician Variance within the same hospital):**
  $$\text{VPC}_{\text{clin}} = \frac{\sigma_{\text{clin}}^2}{\sigma_{\text{hosp}}^2 + \sigma_{\text{clin}}^2 + \sigma_\epsilon^2}$$
* **Total Cluster Correlation (Correlation between two patients treated by the SAME clinician in the SAME hospital):**
  $$\rho_{\text{same\_clin}} = \frac{\sigma_{\text{hosp}}^2 + \sigma_{\text{clin}}^2}{\sigma_{\text{total}}^2}$$
* **Correlation between two patients in the SAME hospital treated by DIFFERENT clinicians:**
  $$\rho_{\text{diff\_clin}} = \frac{\sigma_{\text{hosp}}^2}{\sigma_{\text{total}}^2} = \text{VPC}_{\text{hosp}}$$

#### 3. Contextual vs. Individual Effects (Within-Between Decomposition):
When a patient covariate $X_{ijk}$ (e.g., patient deprivation) is aggregated to the hospital level ($\bar{X}_{\cdot\cdot k}$), 
including both reveals whether the hospital environment exerts an independent contextual influence beyond patient biology:
$$Y_{ijk} = \beta_0 + \beta_{\text{within}} (X_{ijk} - \bar{X}_{\cdot\cdot k}) + \beta_{\text{between}} \bar{X}_{\cdot\cdot k} + v_k + u_{jk} + \epsilon_{ijk}$$
* If $\beta_{\text{between}} \neq \beta_{\text{within}}$: A genuine **contextual neighborhood effect** exists, disproving the ecological fallacy.
""",
        "plain_english_glossary": {
            "Hierarchical / Multilevel Data": "Data organized like Russian nesting dolls: patient checkups inside patients, patients inside GP surgeries, surgeries inside NHS trusts.",
            "Level 1, 2, and 3": "Level 1 is the individual patient visit; Level 2 is the doctor or clinic; Level 3 is the overarching hospital or regional healthcare system.",
            "Variance Partition Coefficient (VPC)": "The percentage of patient outcome variation that can be blamed on which hospital they went to vs. which doctor they saw vs. their own unique genetics.",
            "Contextual Effect": "When being treated in a high-volume teaching hospital gives you better survival outcomes, even after adjusting for your personal age and illness severity.",
            "Ecological Fallacy": "The logical error of assuming that an individual patient follows the exact same trend as the average of their hospital cluster."
        },
        "assumptions": [
            "Random effects at each hierarchical level (v_k, u_jk) follow mutually independent normal distributions.",
            "Sufficient clusters at Level 2 and Level 3 (typically >= 20-30 clusters) to obtain reliable variance estimates.",
            "Residual variance sigma_epsilon^2 is constant across all clusters (homoscedasticity across levels)."
        ],
        "pros_cons": {
            "Pros": [
                "Prevents spurious statistical significance caused by ignoring clustered healthcare delivery.",
                "Correctly separates policy/system performance (hospital trust) from practitioner skill (clinician).",
                "Enables formal exploration of institutional variation for NHS quality benchmarking."
            ],
            "Cons": [
                "Small numbers of clusters (< 15 hospitals) lead to severely biased variance components and anti-conservative CIs.",
                "Substantially more complex model tuning and parameter convergence hurdles.",
                "Requires careful centering of Level-1 predictors to avoid conflating within- and between-cluster effects."
            ]
        }
    },

    "covariance_model_selection": {
        "name": "Covariance Structure Selection & Model Diagnostics: ML vs. REML, AIC & Nakagawa R²",
        "short_name": "Covariance Structures & R² Metrics",
        "icon": "📐",
        "category": "Model Selection & Estimation Diagnostics",
        "tagline": "Contrasting ML vs REML estimation, testing Compound Symmetry vs AR(1) vs Unstructured errors, and Nakagawa conditional R².",
        "mathematical_formulation": r"""
### 📐 Mathematical Formulation: Estimation Methods, Covariance Selection & Goodness-of-Fit

Fitting and selecting longitudinal mixed-effects models requires navigating fundamental estimation choices 
(ML vs. REML), selecting the correct residual error correlation matrix, and reporting interpretable effect sizes ($R^2$).

#### 1. Maximum Likelihood (ML) vs. Restricted Maximum Likelihood (REML):
* **Maximum Likelihood (ML):** Estimates fixed effects $\beta$ and variance components $\theta$ simultaneously.
  * *Downside:* ML underestimates variance components $\sigma^2$ in small samples because it treats fixed effects as known without adjusting for lost degrees of freedom (analogous to dividing by $N$ instead of $N - p$).
  * *Use Case:* **Required** when comparing two nested models that differ in their **fixed effects** via Likelihood Ratio Tests (LRT).
* **Restricted Maximum Likelihood (REML):** Maximizes the likelihood of a set of error contrasts $K^T y$ orthogonal to $X$:
  $$L_{\text{REML}}(\theta) = -\frac{1}{2} \ln|V| - \frac{1}{2} \ln|X^T V^{-1} X| - \frac{1}{2} (y - X\hat{\beta})^T V^{-1} (y - X\hat{\beta})$$
  * *Upside:* Provides completely unbiased estimates of variance components $\sigma_u^2$ and $\sigma_\epsilon^2$.
  * *Use Case:* **Required** for final parameter reporting and when comparing models that differ in their **random effects covariance structure**.

#### 2. Within-Subject Residual Covariance Structures ($R_i$):
For repeated visits at times $t = (1, 2, \dots, T)$:
* **Independent / Variance Components (VC):** $R = \sigma^2 I_T$. Assumes zero correlation between repeated visits (rarely plausible).
* **Compound Symmetry (CS):** $R_{jk} = \sigma^2 \rho$ for $j \neq k$. Assumes constant correlation across all time lags.
* **First-Order Autoregressive (AR(1)):** $R_{jk} = \sigma^2 \rho^{|j - k|}$. Assumes correlation decays exponentially as time between visits increases.
* **Unstructured (UN):** $R_{jk} = \sigma_{jk}$. Estimates all $\frac{T(T+1)}{2}$ unique variances and covariances freely. Maximum flexibility, but costs many degrees of freedom.

#### 3. Likelihood Ratio Test on Boundary Variances (50:50 Mixture $\chi^2$):
When testing whether a random slope is necessary ($H_0: \sigma_{u1}^2 = 0$ vs. $H_1: \sigma_{u1}^2 > 0$), the null parameter lies on the boundary of the parameter space.
Classical Wilks' theorem fails. The asymptotic null distribution is a **50:50 mixture of $\chi^2(q)$ and $\chi^2(q+1)$ distributions**:
$$\text{LRT} = 2(\ln L_1 - \ln L_0) \sim 0.5 \chi^2(0) + 0.5 \chi^2(1)$$
Using standard $\chi^2(1)$ critical values is overly conservative, inflating Type II errors.

#### 4. Nakagawa & Schielzeth (2013) $R^2$ for Mixed Models:
Standard $R^2$ is undefined for mixed models. Nakagawa & Schielzeth formulated two distinct metrics:
* **Marginal $R^2$ ($R^2_m$):** Proportion of variance explained by the **fixed effects alone**:
  $$R^2_m = \frac{\sigma_f^2}{\sigma_f^2 + \sigma_u^2 + \sigma_\epsilon^2}$$
* **Conditional $R^2$ ($R^2_c$):** Proportion of variance explained by the **entire model (both fixed and random effects)**:
  $$R^2_c = \frac{\sigma_f^2 + \sigma_u^2}{\sigma_f^2 + \sigma_u^2 + \sigma_\epsilon^2}$$
Where $\sigma_f^2 = \operatorname{Var}(X\hat{\beta})$ is the variance of the fixed-effects linear predictor.
""",
        "plain_english_glossary": {
            "REML vs. ML": "ML is used when testing which clinical factors matter (comparing fixed effects); REML is used to get accurate estimates of individual patient variability without shrinking errors.",
            "Autoregressive AR(1) Error": "The realistic principle that your health today is very similar to your health yesterday, but less correlated with your health 6 months ago.",
            "Compound Symmetry": "The naive assumption that your health today has the exact same correlation with your health 10 years from now as it does with tomorrow.",
            "Marginal R² (R²_m)": "The percentage of outcome variation explained purely by your treatments and patient demographics (the fixed variables).",
            "Conditional R² (R²_c)": "The total percentage of outcome variation explained when combining treatments with individual patient personal baselines.",
            "Boundary Test (50:50 Mixture)": "A specialized statistical correction required when testing if a variance is zero, because variances cannot be negative."
        },
        "assumptions": [
            "ML must be used when comparing nested fixed effects using Likelihood Ratio Tests.",
            "REML must be used when comparing nested random effects or reporting final parameter estimates.",
            "Covariance structure must be positive definite and invertible."
        ],
        "pros_cons": {
            "Pros": [
                "Nakagawa R² provides intuitive, publication-standard effect size reporting for grant proposals and papers.",
                "AR(1) structures capture biological longitudinal decay accurately without consuming excessive parameters.",
                "REML prevents small-sample variance deflation."
            ],
            "Cons": [
                "Comparing ML models with different random effects or REML models with different fixed effects produces invalid tests.",
                "Unstructured covariance matrices quickly fail to converge when the number of time points exceeds 5-6.",
                "Boundary LRT p-values require manual mixture adjustments."
            ]
        }
    }
}

MIXED_MODELS_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "scenario_lmm": {
        "title": "Cognitive Decline Trial: Random Intercepts & Personalized Slopes across 4 Annual Visits",
        "method_key": "lmm_random_slopes",
        "description": "A 4-year Alzheimer's clinical trial tracking ADAS-Cog scores across 4 visits. Subjects start at different baseline severity (random intercept) and deteriorate at different speeds (random slope)."
    },
    "scenario_glmm": {
        "title": "ICU Sepsis Monitoring: Repeated Binary Organ Failure Relapse across Hospital Wards",
        "method_key": "glmm_binary_count",
        "description": "Evaluating a novel bundle intervention on repeated daily septic shock relapse (0/1) across ICU wards using mixed-effects logistic regression and subject-specific odds ratios."
    },
    "scenario_rm_anova": {
        "title": "Physiotherapy Pain Recovery: Evaluating Sphericity Failure and Missing Visit Recovery",
        "method_key": "rm_anova_vs_lmm",
        "description": "A 12-week chronic back pain trial where Mauchly's sphericity test fails (p < 0.001) and 20% of participants missed follow-up appointments, comparing RM-ANOVA deletion vs LMM likelihood."
    },
    "scenario_3level": {
        "title": "NHS Surgical Outcomes Audit: Patients Nested within Surgeons Nested within Trusts",
        "method_key": "hierarchical_3level",
        "description": "A 3-level hierarchical audit of 30-day post-operative complications across 4,500 patients nested within 120 surgeons across 25 NHS Hospital Trusts, partitioning variance across tiers."
    },
    "scenario_selection": {
        "title": "Longitudinal Cardiac Biomarker: Selecting AR(1) Errors & Nakagawa R² Reporting",
        "method_key": "covariance_model_selection",
        "description": "Selecting between Compound Symmetry and AR(1) error structures for hourly post-infarction Troponin decay, comparing ML vs REML fits, and calculating Nakagawa marginal and conditional R²."
    }
}

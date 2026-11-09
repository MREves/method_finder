"""Causal Inference & Quasi-Experimental Evaluation Schema.

Defines the taxonomy, estimands, identification assumptions, failure modes,
and worked clinical/policy scenarios for quasi-experimental evaluation methods:
1. Difference-in-Differences (DiD) & Event Studies
2. Interrupted Time Series (ITS) & Segmented Regression
3. Propensity Score Matching (PSM) & Weighting (IPTW)
4. Regression Discontinuity Design (RDD: Sharp & Fuzzy)
5. Instrumental Variables (IV) & Mendelian Randomization (MR)
"""

from typing import Dict, Any, List

CAUSAL_METHODS: Dict[str, Dict[str, Any]] = {
    "difference_in_differences": {
        "name": "Difference-in-Differences (DiD) & Event Studies",
        "category": "Quasi-Experimental Policy Evaluation",
        "icon": "⚖️",
        "tag": "Parallel Trends & Policy Shocks",
        "summary": "Estimates the Average Treatment Effect on the Treated (ATT) by comparing the pre-to-post change in an unexposed control group against the pre-to-post change in the exposed intervention group, eliminating time-invariant confounding.",
        "primary_estimand": "Average Treatment Effect on the Treated (ATT / delta)",
        "identification_assumptions": [
            "Parallel Trends Assumption: In the absence of treatment, the average outcome of the treated group would have followed the same trajectory as the control group.",
            "No Anticipation Effect: Units do not change their behavior prior to the formal policy implementation date.",
            "Stable Unit Treatment Value Assumption (SUTVA): No spillover effects from treated units to control units.",
            "Strict Exogeneity of Policy Timing: Treatment adoption is unrelated to idiosyncratic outcome shocks."
        ],
        "plain_english_glossary": {
            "$Y_{it}$": "**Observed Outcome** — The measurable clinical, health-economic, or operational metric (e.g. 30-day mortality rate, surgical waiting time in days, or emergency readmissions) for unit $i$ at time $t$.",
            "$\\text{Treated}_i$": "**Treated Group Indicator** — A binary dummy variable equal to $1$ if hospital or region $i$ receives the intervention, and $0$ if it serves as the unexposed comparator.",
            "$\\text{Post}_t$": "**Post-Policy Period Indicator** — A binary dummy variable equal to $1$ for all time periods after the policy enactment date, and $0$ for baseline pre-policy periods.",
            "$\\delta$ (Delta)": "**Difference-in-Differences Estimant (ATT)** — The causal parameter of interest, representing the net change attributable exclusively to the policy intervention after subtracting counterfactual secular time trends.",
            "$\\beta_0$": "**Baseline Intercept** — The baseline mean outcome of the control group during the pre-intervention period.",
            "$\\beta_1$": "**Baseline Group Difference** — The permanent pre-existing baseline difference between the treatment and control groups before any policy was introduced.",
            "$\\beta_2$": "**Secular Time Trend** — The natural background drift or macro-economic shift occurring over time across all units, irrespective of the intervention.",
            "$\\varepsilon_{it}$": "**Idiosyncratic Residual Error** — Random unmeasured variation and sampling noise."
        },
        "mathematical_formulation": r"""
        ### Difference-in-Differences Identification Equation
        
        The canonical two-way fixed-effects linear regression specification is formulated as:
        $$Y_{it} = \beta_0 + \beta_1 \cdot \text{Treated}_i + \beta_2 \cdot \text{Post}_t + \delta \cdot (\text{Treated}_i \times \text{Post}_t) + \varepsilon_{it}$$
        
        The sample average Difference-in-Differences estimand ($\hat{\delta}$) computes the subtraction of two differences:
        $$\hat{\delta}_{\text{DiD}} = \left( \bar{Y}_{\text{Treated, Post}} - \bar{Y}_{\text{Treated, Pre}} \right) - \left( \bar{Y}_{\text{Control, Post}} - \bar{Y}_{\text{Control, Pre}} \right)$$
        
        #### Counterfactual Deduction:
        $$\text{Counterfactual for Treated Group} = \bar{Y}_{\text{Treated, Pre}} + \left( \bar{Y}_{\text{Control, Post}} - \bar{Y}_{\text{Control, Pre}} \right)$$
        $$\hat{\delta}_{\text{ATT}} = \bar{Y}_{\text{Treated, Post}} - \text{Counterfactual}$$
        """,
        "when_to_use": [
            "Evaluating a national, regional, or hospital-level policy rolled out in some units but not others.",
            "Observational panel or repeated cross-sectional data observed across multiple pre- and post-intervention periods.",
            "When random assignment is unfeasible or unethical, but a comparable unexposed control cohort exists.",
            "When pre-treatment historical data is available to formally test the parallel trends assumption."
        ],
        "when_not_to_use": [
            "When pre-intervention outcome trajectories between treated and control units are diverging (Parallel trends violation).",
            "When treatment units selected into the policy based on expected future shocks or reverse causality.",
            "When cross-unit contamination or spillover is substantial (e.g. patients cross regional borders to receive treatment).",
            "When there is no valid unexposed comparison group (use Interrupted Time Series instead)."
        ],
        "falsification_tests": [
            "Event-Study Placebo Leads: Regressing outcome on pre-treatment policy leads. Coefficients on pre-treatment leads must be statistically indistinguishable from zero ($p > 0.05$).",
            "Placebo Outcome Test: Re-running the DiD model on an unrelated outcome that could not plausibly be affected by the intervention (e.g. elective dermatology waiting times when testing an acute cardiology pathway)."
        ]
    },

    "interrupted_time_series": {
        "name": "Interrupted Time Series (ITS) & Segmented Regression",
        "category": "Quasi-Experimental Temporal Evaluation",
        "icon": "📉",
        "tag": "Single Cohort Longitudinal Shocks",
        "summary": "Evaluates the longitudinal impact of an abrupt health intervention, clinical guideline release, or legislation on a continuous population sequence, separating immediate level shifts from sustained slope trajectory changes.",
        "primary_estimand": "Immediate Level Shift (beta_2) and Sustained Trend / Slope Change (beta_3)",
        "identification_assumptions": [
            "Secular Trend Stability: The historical pre-intervention trajectory would have continued linearly in the absence of the intervention.",
            "Accurate Intervention Timing: The exact calendar date of policy introduction is known and exogenous.",
            "Absence of Confounding Co-Interventions: No other major health system or economic reforms occurred simultaneously at the exact intervention point.",
            "Stationary Autocorrelation: Autoregressive error structures are properly modeled via Newey-West standard errors or Cochrane-Orcutt transformation."
        ],
        "plain_english_glossary": {
            "$Y_t$": "**Population Outcome at Time $t$** — The continuous aggregate rate or count (e.g. daily antibiotic prescribing rate per 1,000 patients, or monthly emergency hospitalizations).",
            "$T_t$": "**Continuous Time Elapsed** — A sequential counter ($1, 2, 3, \dots, n$) measuring time from the start of the entire observation window, capturing the baseline secular trajectory.",
            "$D_t$": "**Intervention Step Dummy** — A binary indicator set to $0$ for all time steps before the policy introduction, and $1$ for all time steps after policy enactment.",
            "$P_t$": "**Time Elapsed Post-Intervention** — A counter set to $0$ prior to intervention, and $(1, 2, 3, \dots)$ during the post-intervention window, measuring the sustained rate of change.",
            "$\\beta_0$": "**Baseline Starting Level** — The estimated outcome value at the very beginning of the observation sequence ($T=0$).",
            "$\\beta_1$": "**Pre-Intervention Secular Slope** — The background trajectory or drift in the outcome prior to any policy intervention.",
            "$\\beta_2$": "**Immediate Level Shift** — The sudden, immediate drop or jump in the outcome occurring at the moment the intervention is implemented.",
            "$\\beta_3$": "**Sustained Trajectory / Slope Change** — The difference between the post-intervention slope and the pre-intervention slope.",
            "$\\varepsilon_t$": "**Temporal Autocorrelated Error** — The residual disturbance, typically exhibiting first-order autoregressive serial correlation."
        },
        "mathematical_formulation": r"""
        ### Segmented Regression Model for Interrupted Time Series
        
        The standard single-group segmented ordinary least squares (OLS) regression model is specified as:
        $$Y_t = \beta_0 + \beta_1 \cdot T_t + \beta_2 \cdot D_t + \beta_3 \cdot P_t + \varepsilon_t$$
        
        #### Interpretation of Parameters:
        - **Baseline secular slope:** $\beta_1$
        - **Counterfactual slope in post-period:** $\beta_1$
        - **Actual post-intervention slope:** $\beta_1 + \beta_3$
        - **Immediate causal shock (level jump):** $\beta_2$
        - **Long-term causal divergence after $k$ steps:** $\beta_2 + (k \cdot \beta_3)$
        
        #### Autoregressive Error Correction (Newey-West):
        Standard OLS standard errors underestimate variance due to serial correlation ($e_t \sim \rho e_{t-1}$).
        Robust standard errors must be computed using Heteroscedasticity and Autocorrelation Consistent (HAC) covariance matrices.
        """,
        "when_to_use": [
            "Evaluating a population-level health reform enacted simultaneously everywhere (e.g. national smoking ban, COVID-19 lockdown, NICE guideline release).",
            "Longitudinal aggregate time series data with at least 12 (preferably 24+) data points before and after the event.",
            "When no concurrent unexposed control group is available, making Difference-in-Differences impossible.",
            "When interested in distinguishing immediate shocks from gradual cultural/behavioral adoption curves."
        ],
        "when_not_to_use": [
            "When the observation period has fewer than 8–10 points before the intervention (insufficient power to establish baseline slope).",
            "When other major policy interventions were launched at the exact same calendar month or quarter.",
            "When the outcome shows severe non-linear seasonality that cannot be accounted for via Fourier terms or harmonic regressors."
        ],
        "falsification_tests": [
            "Placebo Interruption Point: Testing for false step changes at artificial intervention dates (e.g. 6 months before the real policy).",
            "Durbin-Watson Autocorrelation Diagnostic: $DW \approx 2$ indicates no serial correlation; $DW < 1.5$ requires Newey-West HAC or ARIMA errors."
        ]
    },

    "propensity_score_matching": {
        "name": "Propensity Score Matching (PSM) & Weighting (IPTW)",
        "category": "Selection on Observables (Confounding Control)",
        "icon": "🎯",
        "tag": "Confounding by Indication",
        "summary": "Eliminates overt selection bias in observational cohorts by modeling the conditional probability of receiving treatment given baseline covariates, pairing treated and unexposed individuals with identical propensity scores.",
        "primary_estimand": "Average Treatment Effect (ATE) or Average Treatment Effect on the Treated (ATT)",
        "identification_assumptions": [
            "Conditional Independence Assumption (Unconfoundedness): Potential outcomes are independent of treatment assignment conditional on observed covariates $(Y(0), Y(1) \perp Z \mid X)$.",
            "Common Support (Positivity / Overlap): Every participant has a non-zero probability of receiving either treatment: $0 < P(Z=1 \mid X) < 1$.",
            "No Unmeasured Confounding: All clinical and socio-demographic factors influencing both treatment choice and outcome are measured without error.",
            "SUTVA: One individual's treatment status does not influence another individual's clinical outcome."
        ],
        "plain_english_glossary": {
            "$e(X)$": "**Propensity Score** — The single summary probability ($0.00$ to $1.00$) that a patient receives the active treatment given their baseline clinical characteristics $X$, typically modeled via multivariable logistic regression.",
            "$Z_i$": "**Treatment Indicator** — Binary indicator where $Z=1$ indicates receipt of the novel intervention, and $Z=0$ indicates standard care or control.",
            "$X_i$": "**Baseline Covariate Vector** — The full matrix of pre-treatment confounders (e.g. age, Charlson comorbidity index, baseline blood pressure, deprivation decile).",
            "$\\text{Caliper}$": "**Matching Tolerance Width** — The maximum allowable distance in propensity score (typically $0.20 \times \\text{SD}_{\\text{logit}(e)}$) between matched treated and control pairs.",
            "$\\text{SMD}$": "**Standardized Mean Difference** — The metric used to evaluate balance. An absolute value $|\text{SMD}| < 0.10$ signifies acceptable balance; $|\text{SMD}| < 0.05$ indicates excellent balance.",
            "$\\text{IPTW}$": "**Inverse Probability of Treatment Weighting** — Creates a synthetic pseudo-population where confounders are balanced by weighting treated patients by $1/e$ and controls by $1/(1-e)$."
        },
        "mathematical_formulation": r"""
        ### Propensity Score Formulation & Balancing Derivation
        
        The propensity score is defined by Rosenbaum & Rubin (1983) as:
        $$e(X_i) = P(Z_i = 1 \mid X_i) = \frac{1}{1 + \exp\left(-\left(\gamma_0 + \sum_{j=1}^p \gamma_j X_{ij}\right)\right)}$$
        
        #### Standardized Mean Difference (SMD) Balance Diagnostic:
        $$\text{SMD} = \frac{\bar{X}_{\text{Treated}} - \bar{X}_{\text{Control}}}{\sqrt{\frac{s_{\text{Treated}}^2 + s_{\text{Control}}^2}{2}}}$$
        - **Target:** $|\text{SMD}| < 0.10$ across all baseline covariates indicates adequate balance.
        
        #### Inverse Probability of Treatment Weighting (IPTW Estimator):
        $$w_i = \frac{Z_i}{e(X_i)} + \frac{1 - Z_i}{1 - e(X_i)}$$
        $$\hat{\tau}_{\text{ATE}} = \frac{\sum_{i=1}^n w_i Z_i Y_i}{\sum_{i=1}^n w_i Z_i} - \frac{\sum_{i=1}^n w_i (1 - Z_i) Y_i}{\sum_{i=1}^n w_i (1 - Z_i)}$$
        """,
        "when_to_use": [
            "Observational registry or electronic health record (EHR) studies comparing two treatment arms with confounding by indication.",
            "When rich, granular baseline clinical confounders are recorded prior to treatment initiation.",
            "When the number of baseline confounders is large relative to the number of outcome events (propensity scores avoid multi-collinearity).",
            "When estimating real-world effectiveness in broad, heterogeneous patient populations excluded from strict RCT eligibility criteria."
        ],
        "when_not_to_use": [
            "When major unmeasured confounders exist (e.g. clinician gut feeling, frailty not captured in ICD-10 codes).",
            "When there is poor common support (e.g. certain high-risk patient subgroups NEVER receive the conservative control treatment).",
            "When the sample size is very small, leading to severe sample attrition during caliper matching."
        ],
        "falsification_tests": [
            "Common Support Plot: Visual inspection of propensity score distribution overlap across treated and control arms.",
            "Negative Control Outcome: Testing whether the propensity-matched model produces a zero effect on a health outcome known to be biologically independent of the drug."
        ]
    },

    "regression_discontinuity": {
        "name": "Regression Discontinuity Design (RDD: Sharp & Fuzzy)",
        "category": "Quasi-Random Cutoff Identification",
        "icon": "✂️",
        "tag": "Clinical Triage Cutoffs & Guidelines",
        "summary": "Exploits strict, exogenous clinical thresholds or triage cutoff scores (e.g. QRISK >= 10% for statins, CD4 count, birth weight) where individuals just above and just below the threshold are as good as randomly assigned.",
        "primary_estimand": "Local Average Treatment Effect (LATE) at the Cutoff Threshold c",
        "identification_assumptions": [
            "Continuity of Potential Outcomes: In the absence of treatment, the relationship between the running variable and the outcome would be smooth and continuous across the cutoff.",
            "No Precise Manipulation / Sorting: Patients or clinicians cannot precisely manipulate their score to fall on the favored side of the threshold.",
            "Local Randomization: Patients within an infinitesimal bandwidth $[c - h, c + h]$ differ only by chance.",
            "Monotonicity (Fuzzy RDD): The threshold encourages treatment for some, but discourages treatment for none."
        ],
        "plain_english_glossary": {
            "$X_i$": "**Running / Forcing Variable** — The continuous diagnostic score or clinical measurement (e.g. QRISK2 10-year cardiovascular risk score, gestational birthweight, or HbA1c level) used to determine treatment eligibility.",
            "$c$": "**Threshold / Cutoff Point** — The exact statutory or clinical guideline boundary (e.g. $c = 10.0\\%$ for statin prescribing) where treatment protocol changes.",
            "$Z_i$": "**Treatment Assignment** — Equal to $1$ if $X_i \ge c$, and $0$ if $X_i < c$.",
            "$h$": "**Bandwidth Window** — The local analytical neighborhood around the cutoff $[c - h, c + h]$ within which local linear regressions are estimated.",
            "$\\tau_{\\text{SRDD}}$": "**Sharp RDD Treatment Effect (LATE)** — The vertical jump or discontinuity gap in the regression lines evaluated precisely at the cutoff boundary $c$.",
            "$\\text{Fuzzy RDD}$": "**Imperfect Compliance Setup** — Occurs when crossing the cutoff increases the probability of treatment, but does not guarantee it (estimated via Instrumental Variables / 2SLS)."
        },
        "mathematical_formulation": r"""
        ### Sharp Regression Discontinuity Model
        
        The local linear regression specification with differing slopes on either side of cutoff $c$ is:
        $$Y_i = \alpha + \tau \cdot Z_i + \beta_1 (X_i - c) + \beta_2 Z_i (X_i - c) + \varepsilon_i \quad \text{for } |X_i - c| \le h$$
        
        #### Identification of Causal Discontinuity:
        $$\tau_{\text{SRDD}} = \lim_{x \downarrow c} E[Y \mid X=x] - \lim_{x \uparrow c} E[Y \mid X=x]$$
        - **Sharp RDD:** Treatment assignment is a deterministic step function: $P(Z=1 \mid X) = \mathbf{1}(X \ge c)$.
        - **Fuzzy RDD (Instrumental Variables Ratio):**
          $$\tau_{\text{FRDD}} = \frac{\lim_{x \downarrow c} E[Y \mid X=x] - \lim_{x \uparrow c} E[Y \mid X=x]}{\lim_{x \downarrow c} E[D \mid X=x] - \lim_{x \uparrow c} E[D \mid X=x]}$$
        """,
        "when_to_use": [
            "Healthcare decisions, screening programs, or pharmaceutical subsidies assigned strictly by a numerical biomarker cutoff.",
            "Primary care clinical decision support rules (e.g. statin initiation at QRISK2 >= 10%).",
            "Emergency department triage scoring rules (e.g. NEWS2 score thresholds for intensive care admission).",
            "When internal validity close to an RCT is required from non-randomized administrative healthcare data."
        ],
        "when_not_to_use": [
            "When the running variable is heavily manipulated or rounded by clinicians to secure treatment eligibility.",
            "When multiple treatments or social benefits change at the exact same threshold value.",
            "When an estimate of the average treatment effect across the entire population (ATE) is required (RDD only estimates LATE at the cutoff)."
        ],
        "falsification_tests": [
            "McCrary Density Test: Testing for structural breaks or clustering of observations immediately adjacent to the threshold (detects manipulation).",
            "Covariate Continuity Test: Re-running RDD on baseline pre-determined covariates (e.g. age, sex). The estimated jump must be zero ($p > 0.05$)."
        ]
    },

    "instrumental_variables": {
        "name": "Instrumental Variables (IV) & Mendelian Randomization (MR)",
        "category": "Selection on Unobservables (Exogenous Shocks)",
        "icon": "🧬",
        "tag": "Mendelian Randomization & Natural Experiments",
        "summary": "Overcomes unmeasured confounding and reverse causality by leveraging an exogenous instrument (e.g. genetic variant, geographical distance, randomized lottery) that influences treatment uptake but has no direct link to the outcome.",
        "primary_estimand": "Local Average Treatment Effect (LATE) via Two-Stage Least Squares (2SLS)",
        "identification_assumptions": [
            "Relevance Condition: The instrument $Z$ is strongly associated with the exposure $X$ (First-Stage $F$-statistic $> 10$ to prevent weak instrument bias).",
            "Exclusion Restriction: The instrument $Z$ affects the outcome $Y$ solely and exclusively through its effect on exposure $X$.",
            "Independence (Exogeneity): The instrument $Z$ is independent of all unmeasured confounders $U$ influencing $X$ and $Y$.",
            "Monotonicity: There are no 'defiers' (individuals who would take the opposite treatment when instrumented)."
        ],
        "plain_english_glossary": {
            "$Z_i$": "**Instrumental Variable** — An exogenous factor (e.g. single nucleotide polymorphism SNP, hospital distance, differential regional prescribing preference) that acts as an unconfounded nudge.",
            "$X_i$": "**Endogenous Exposure / Treatment** — The clinical risk factor or drug intake (e.g. serum C-reactive protein level, or actual medication compliance) contaminated by unobserved confounding $U$.",
            "$Y_i$": "**Final Clinical Outcome** — The disease endpoint (e.g. coronary heart disease event or 10-year stroke incidence).",
            "$U_i$": "**Unmeasured Confounders** — Hidden factors (e.g. healthy user bias, unrecorded lifestyle habits, genetic frailty) biasing naive observational OLS regression.",
            "$\\text{First-Stage } F$": "**Weak Instrument Diagnostic** — Tests instrument strength. $F > 10$ (or Stock-Yogo $F > 16.38$) confirms the instrument is sufficiently strong to avoid severe asymptotic bias.",
            "$\\text{2SLS}$": "**Two-Stage Least Squares** — Regresses $X$ on $Z$ to extract unconfounded variation $\hat{X}$, then regresses $Y$ on $\hat{X}$ to estimate the unbiased causal coefficient."
        },
        "mathematical_formulation": r"""
        ### Two-Stage Least Squares (2SLS) Structural Model
        
        #### Stage 1: Predict Endogenous Exposure from Exogenous Instrument
        $$X_i = \gamma_0 + \gamma_1 Z_i + \Gamma W_i + v_i \quad \implies \quad \hat{X}_i = \hat{\gamma}_0 + \hat{\gamma}_1 Z_i + \hat{\Gamma} W_i$$
        - **Strength Criterion:** First-Stage $F = \frac{\hat{\gamma}_1^2}{\text{Var}(\hat{\gamma}_1)} > 10$
        
        #### Stage 2: Regress Outcome on Purged Predicted Exposure
        $$Y_i = \beta_0 + \beta_{\text{2SLS}} \hat{X}_i + \text{B} W_i + \varepsilon_i$$
        
        #### The Simple Wald Estimator:
        $$\hat{\beta}_{\text{IV}} = \frac{\text{Cov}(Y, Z)}{\text{Cov}(X, Z)} = \frac{\bar{Y}_{Z=1} - \bar{Y}_{Z=0}}{\bar{X}_{Z=1} - \bar{X}_{Z=0}} = \frac{\beta_{\text{Reduced Form}}}{\gamma_{\text{First Stage}}}$$
        """,
        "when_to_use": [
            "When confounding by indication or reverse causality is suspected and unmeasured confounders cannot be ruled out.",
            "Mendelian Randomization: Using lifelong randomly allocated genetic alleles (Meiosis) to establish whether a biomarker causally drives disease.",
            "Healthcare delivery evaluations where geographic distance to a specialized tertiary center acts as a natural randomized instrument.",
            "Randomized trials with non-compliance (estimating Complier Average Causal Effect [CACE])."
        ],
        "when_not_to_use": [
            "When the instrument is weak ($F < 10$), as 2SLS estimates will be more biased than confounded OLS.",
            "When horizontal pleiotropy is present (the genetic variant directly affects the outcome through biological pathways outside $X$).",
            "When the exclusion restriction is biologically or clinically implausible."
        ],
        "falsification_tests": [
            "First-Stage F-Statistic: Must exceed 10 (or 16.38) to reject weak instrument null hypothesis.",
            "Sargan-Hansen Overidentifying Restrictions Test: Validates that multiple instruments yield mutually consistent causal estimates ($p > 0.05$)."
        ]
    }
}

CAUSAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "hospital_7day_services": {
        "title": "Hospital 7-Day Consultant Working (NHS Weekend Mortality Policy)",
        "method": "difference_in_differences",
        "domain": "Healthcare Policy & Service Reconfiguration",
        "description": "Evaluation of whether introducing 7-day consultant-led ward rounds across 40 pioneer NHS acute hospital Trusts reduced 30-day weekend mortality compared to 60 standard-care control Trusts.",
        "outcome_metric": "30-Day Inpatient Emergency Mortality Rate (per 1,000 admissions)",
        "unit": "deaths / 1,000",
        "pre_policy_periods": 4,
        "post_policy_periods": 4
    },
    "nice_guideline_opioid": {
        "title": "NICE Guideline on Chronic Pain & Opioid De-prescribing",
        "method": "interrupted_time_series",
        "domain": "Pharmaceutical Policy & Clinical Guidelines",
        "description": "Longitudinal evaluation tracking monthly primary care high-dose opioid prescribing rates across 36 consecutive months before and after the release of updated NICE chronic primary pain guidelines.",
        "outcome_metric": "High-Dose Opioid Prescriptions (per 1,000 registered practice patients)",
        "unit": "scripts / 1,000",
        "total_periods": 36,
        "intervention_period": 18
    },
    "surgical_robotic_resection": {
        "title": "Robotic vs. Laparoscopic Colorectal Resection (Confounding by Indication)",
        "method": "propensity_score_matching",
        "domain": "Surgical Innovation & Real-World Evidence",
        "description": "Observational cohort of 500 surgical colorectal patients where robotic surgery was disproportionately allocated to younger, affluent, and less comorbid patients. Propensity matching pairs cases on 6 baseline confounders.",
        "outcome_metric": "Inpatient Hospital Length of Stay (Days)",
        "unit": "days",
        "sample_size": 500
    },
    "statin_qrisk_threshold": {
        "title": "Primary Prevention Statin Initiation at QRISK2 >= 10% Threshold",
        "method": "regression_discontinuity",
        "domain": "Preventive Cardiology & Clinical Guidelines",
        "description": "Clinical guidelines recommend initiating atorvastatin 20mg when 10-year cardiovascular risk reaches QRISK2 >= 10.0%. Evaluates 5-year major adverse cardiac events (MACE) immediately across the 10% threshold boundary.",
        "outcome_metric": "5-Year Major Adverse Cardiac Event (MACE) Incidence Rate (%)",
        "unit": "%",
        "cutoff": 10.0,
        "sample_size": 600
    },
    "crp_mendelian_randomization": {
        "title": "C-Reactive Protein (CRP) & Ischaemic Heart Disease (Genetic Instrument)",
        "method": "instrumental_variables",
        "domain": "Etiological Epidemiology & Mendelian Randomization",
        "description": "Observational studies show high circulating CRP correlates with coronary heart disease, but observational regression is contaminated by obesity and lifestyle confounding. A genetic variant (CRP gene rs1205) is used as an instrument.",
        "outcome_metric": "10-Year Coronary Heart Disease Hazard / Event Risk (Odds Ratio)",
        "unit": "log-odds",
        "sample_size": 800
    }
}

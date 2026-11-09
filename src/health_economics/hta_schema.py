"""Health Economics & Health Technology Assessment (HTA / NICE) Schema.

Defines schemas, mathematical derivations, plain-English glossaries,
method assumptions, pros/cons, and realistic NHS England clinical scenarios for:
1. Incremental Cost-Effectiveness Ratio (ICER) & Cost-Effectiveness Plane
2. Net Monetary Benefit (NMB) & Net Health Benefit (NHB)
3. Markov Decision Analytic Cohort Models (State Transitions, Half-Cycle Correction & 3.5% Discounting)
4. Probabilistic Sensitivity Analysis (PSA) & Cost-Effectiveness Acceptability Curves (CEAC)
5. Budget Impact Analysis (BIA) & NHS Payer Affordability
"""

from typing import Dict, Any, List

HTA_METHODS: Dict[str, Dict[str, Any]] = {
    "icer_plane": {
        "name": "Cost-Effectiveness & Incremental Cost-Effectiveness Ratio (ICER)",
        "short_name": "ICER & CE Plane",
        "icon": "⚖️",
        "category": "Cost-Effectiveness Analysis",
        "tagline": "Quantifying incremental cost per Quality-Adjusted Life Year (QALY) gained against NHS willingness-to-pay thresholds.",
        "mathematical_formulation": r"""
### ⚖️ Mathematical Formulation: Incremental Cost-Effectiveness Ratio (ICER)

In Health Technology Assessment (HTA) and health economics decision-making (e.g. UK National Institute for Health and Care Excellence - NICE), 
interventions are evaluated by comparing the incremental costs to the incremental health outcomes relative to the current standard of care.

#### 1. Incremental Cost and Health Gain:
$$\Delta C = C_T - C_C$$
$$\Delta E = E_T - E_C$$

Where:
* $C_T$ = Mean total lifetime cost of the novel intervention / technology.
* $C_C$ = Mean total lifetime cost of the comparator / current standard of care.
* $E_T$ = Mean total health gain (e.g. Quality-Adjusted Life Years, QALYs) for the intervention.
* $E_C$ = Mean total health gain (QALYs) for the comparator.

#### 2. The ICER Equation:
$$\text{ICER} = \frac{\Delta C}{\Delta E} = \frac{C_T - C_C}{E_T - E_C}$$

The ICER represents the **additional cost required to gain one additional QALY**.

#### 3. The 4-Quadrant Cost-Effectiveness Plane:
The cost-effectiveness plane plots incremental health outcomes ($\Delta E$, horizontal axis) against incremental costs ($\Delta C$, vertical axis):
* **North-East Quadrant ($\Delta C > 0, \Delta E > 0$):** More effective and more expensive. Cost-effective if $\text{ICER} \le \lambda$, where $\lambda$ is the payer's Willingness-To-Pay (WTP) threshold (standard UK NICE threshold is $£20,000 - £30,000$ per QALY).
* **South-East Quadrant ($\Delta C < 0, \Delta E > 0$):** **Dominant**. More effective and cost-saving. Intervention is unequivocally adopted.
* **North-West Quadrant ($\Delta C > 0, \Delta E < 0$):** **Dominated**. Less effective and more expensive. Intervention is unequivocally rejected.
* **South-West Quadrant ($\Delta C < 0, \Delta E < 0$):** Less effective but cheaper. Adopted only if the cost savings per QALY forgone exceed the threshold ($\text{ICER} \ge \lambda$).

#### 4. NICE Decision Rules:
* If $\text{ICER} < £20,000 / \text{QALY}$: High likelihood of recommendation based on cost-effectiveness.
* If $£20,000 \le \text{ICER} \le £30,000 / \text{QALY}$: Consideration of certainty of evidence, uncaptured benefits, and innovation.
* If $\text{ICER} > £30,000 / \text{QALY}$: Typically not cost-effective unless meeting the **Severity of Condition** weight modifier (up to $£50,000 / \text{QALY}$ for life-threatening diseases with high shortfall).
""",
        "plain_english_glossary": {
            "QALY (Quality-Adjusted Life Year)": "A universal metric of disease burden and health benefit combining quantity of life (years lived) and quality of life (health-related utility where 1.0 represents perfect health and 0.0 represents death). 1 QALY equals 1 year lived in perfect health.",
            "Incremental Cost (ΔC)": "The extra monetary expenditure incurred by switching an average patient from current standard care to the new treatment.",
            "Incremental Effect (ΔE)": "The extra clinical benefit (typically QALYs or life years) achieved by switching to the new treatment.",
            "Willingness-to-Pay Threshold (λ)": "The maximum price a healthcare system (such as the NHS) is willing to pay from public funds to purchase one additional QALY for its population.",
            "Dominance": "A state where one intervention is strictly superior because it produces more health benefits at lower total cost (dominant), or strictly inferior because it produces fewer health benefits at higher cost (dominated)."
        },
        "assumptions": [
            "Additivity and divisibility of health utilities and monetary expenditures across patient lifetimes.",
            "Perspective: Assumes an explicit perspective (e.g., NHS and Personal Social Services in the UK, excluding wider societal productivity losses).",
            "Linearity of opportunity costs across the healthcare budget (the marginal productivity of the healthcare system remains constant at λ).",
            "Stability of comparator pricing without dynamic patent expiration cliffs unless explicitly modelled."
        ],
        "pros_cons": {
            "Pros": [
                "Universal benchmark accepted by NICE, CADTH, PBAC, and health insurance regulators worldwide.",
                "Enables direct comparisons across completely different medical specialities (e.g. hip replacement vs cancer chemotherapy).",
                "Explicitly quantifies the opportunity cost of adopting expensive new treatments within a fixed healthcare budget."
            ],
            "Cons": [
                "Ratio instability when incremental effect is very close to zero (division by near-zero yields extreme ICERs).",
                "Does not naturally order multiple non-dominated comparators without incremental pairwise ranking.",
                "May overlook health inequalities and equity weights unless adjusted by formal severity multipliers."
            ]
        }
    },

    "nmb_nhb": {
        "name": "Net Monetary Benefit (NMB) & Net Health Benefit (NHB)",
        "short_name": "NMB & NHB",
        "icon": "💷",
        "category": "Cost-Effectiveness Analysis",
        "tagline": "Linear value transformation that overcomes ICER ratio instability and enables clean ranking of multiple mutually exclusive treatments.",
        "mathematical_formulation": r"""
### 💷 Mathematical Formulation: Net Monetary Benefit (NMB) & Net Health Benefit (NHB)

While the ICER is intuitive, ratio statistics suffer from mathematical instabilities: when $\Delta E \approx 0$, the ratio approaches 
$\pm \infty$; when $\Delta E < 0$, negative ICERs in the South-West quadrant look mathematically identical to negative ICERs in the 
dominant South-East quadrant. 

To overcome this, health economists rearrange the decision rule $\text{ICER} \le \lambda \iff \frac{\Delta C}{\Delta E} \le \lambda$ into 
a **linear net benefit framework**.

#### 1. Net Monetary Benefit (NMB):
$$\text{NMB}(\lambda) = \lambda \cdot \Delta E - \Delta C = (\lambda \cdot E_T - C_T) - (\lambda \cdot E_C - C_C)$$

For an individual treatment $k$:
$$\text{NMB}_k(\lambda) = \lambda \cdot E_k - C_k$$

#### 2. Net Health Benefit (NHB):
$$\text{NHB}(\lambda) = \Delta E - \frac{\Delta C}{\lambda}$$

For an individual treatment $k$:
$$\text{NHB}_k(\lambda) = E_k - \frac{C_k}{\lambda}$$

#### 3. Decision Rules:
* An intervention is **cost-effective** if:
  $$\text{NMB}(\lambda) > 0 \quad \text{or} \quad \text{NHB}(\lambda) > 0$$
* In a comparison of multiple mutually exclusive interventions, the optimal decision is to choose the option that **maximizes NMB**:
  $$k^* = \arg\max_k \text{NMB}_k(\lambda)$$

#### 4. Incremental NMB Confidence Intervals:
The variance of incremental NMB is linear:
$$\text{Var}(\text{NMB}) = \lambda^2 \text{Var}(\Delta E) + \text{Var}(\Delta C) - 2\lambda \text{Cov}(\Delta E, \Delta C)$$

This allows standard symmetrical parametric confidence intervals and standard regression models (Net Benefit Regression).
""",
        "plain_english_glossary": {
            "Net Monetary Benefit (NMB)": "The monetary value of the health gains produced by an intervention minus the financial cost of delivering it, evaluated at the payer's willingness-to-pay threshold λ.",
            "Net Health Benefit (NHB)": "The net health gain expressed in QALYs after subtracting the health displaced elsewhere in the NHS to fund the intervention.",
            "Opportunity Cost Displacement": "The reality that every £20,000-£30,000 spent on a new drug means canceling or withholding care elsewhere in the NHS that would have generated 1 full QALY for other patients.",
            "Linear Value Metric": "A measurement scale where differences can be added, subtracted, and averaged directly without dividing by small or negative denominators."
        },
        "assumptions": [
            "Willingness-to-pay threshold λ is known or evaluated over a sensible decision range (£10,000 to £50,000).",
            "Linear relationship between health utility and monetary value at the chosen threshold.",
            "Budget displacement occurs at the marginal threshold λ."
        ],
        "pros_cons": {
            "Pros": [
                "Completely eliminates ratio division by zero or negative incremental effectiveness.",
                "Enables seamless ranking and selection when comparing 3 or more competing clinical strategies.",
                "Directly compatible with ordinary linear regression (Net Benefit Regression) to adjust for patient baseline covariates."
            ],
            "Cons": [
                "Requires a specific willingness-to-pay value λ to be chosen upfront.",
                "Less historically familiar to clinicians than the conventional ICER £/QALY figure.",
                "Expresses health benefits in currency, which some non-economists find less intuitive than QALYs."
            ]
        }
    },

    "markov_model": {
        "name": "Markov Decision Analytic Cohort State-Transition Models",
        "short_name": "Markov Cohort Models",
        "icon": "🔄",
        "category": "Decision Analytic Modeling",
        "tagline": "Tracking chronic disease progression across mutually exclusive health states with matrix transitions, half-cycle correction, and 3.5% NICE discounting.",
        "mathematical_formulation": r"""
### 🔄 Mathematical Formulation: Markov Decision Analytic Cohort Models

Chronic diseases (e.g., cancer, cardiovascular disease, diabetes) involve ongoing risks and costs extending over years or decades. 
Markov state-transition models represent patient prognosis across a finite set of mutually exclusive health states over discrete time cycles.

#### 1. State Vector and Transition Probability Matrix:
Let $S_t = [s_{1,t}, s_{2,t}, \dots, s_{K,t}]$ be the proportion of the cohort in each of $K$ health states at cycle $t$, where $\sum_{k=1}^K s_{k,t} = 1$.

The transition matrix $P$ contains probabilities $p_{ij}$ of transitioning from state $i$ to state $j$ in a single cycle:
$$P = \begin{pmatrix} p_{11} & p_{12} & \dots & p_{1K} \\ p_{21} & p_{22} & \dots & p_{2K} \\ \vdots & \vdots & \ddots & \vdots \\ p_{K1} & p_{K2} & \dots & p_{KK} \end{pmatrix}, \quad \sum_{j=1}^K p_{ij} = 1 \quad \forall i$$

Cohort movement from cycle $t$ to $t+1$:
$$S_{t+1} = S_t \cdot P$$

#### 2. Annual Rate to Cycle Probability Conversion:
When transitioning clinical trial hazard rates $r$ into discrete cycle probabilities with cycle length $t_{\text{cycle}}$:
$$p = 1 - \exp(-r \cdot t_{\text{cycle}})$$

#### 3. Half-Cycle Correction:
Because transitions occur continuously throughout a cycle rather than instantaneously at the start or end, the **half-cycle correction** 
(trapezoidal rule) adjusts cumulative state membership:
$$\bar{S}_t = \frac{S_t + S_{t+1}}{2} \quad \text{for } t = 0, \dots, T-1$$

#### 4. NICE Future Discounting Formula (Costs & Health Outcomes):
NICE reference case requires an annual discount rate of $r_{\text{disc}} = 3.5\%$ (0.035) for both costs and health benefits:
$$PV(C) = \sum_{t=0}^T \frac{C_t}{(1 + r_{\text{disc}})^t}$$
$$PV(Q) = \sum_{t=0}^T \frac{Q_t}{(1 + r_{\text{disc}})^t}$$

Where $C_t = \sum_k \bar{s}_{k,t} \cdot c_k$ and $Q_t = \sum_k \bar{s}_{k,t} \cdot u_k \cdot t_{\text{cycle}}$, with state cost $c_k$ and state utility weight $u_k$.
""",
        "plain_english_glossary": {
            "Health State": "A mutually exclusive clinical category describing a patient's current disease status (e.g. 'Progression-Free', 'Progressed Disease', or 'Dead').",
            "Absorbing State": "A terminal health state from which patients cannot leave once entered (e.g., all-cause mortality / death).",
            "Transition Probability": "The statistical probability of moving from one health state to another over the duration of one cycle (e.g. 1 year or 1 month).",
            "Half-Cycle Correction": "A mathematical adjustment that accounts for events occurring throughout the year rather than entirely at midnight on New Year's Eve.",
            "Discounting (3.5% NICE Standard)": "Reflects time preference: people and society value benefits received today more than benefits deferred to the distant future, and cash spent today has an opportunity cost."
        },
        "assumptions": [
            "Markovian Memoryless Property: Future transition probabilities depend solely on the current state, not on previous history or time spent in earlier states (can be relaxed using tunnel states).",
            "Homogeneity: All cohort members within a state share identical average costs and utilities.",
            "Discrete cycles: Biological events occur smoothly, but are approximated across discrete time intervals."
        ],
        "pros_cons": {
            "Pros": [
                "Gold standard modeling framework for chronic progressive conditions submitted to NICE and CADTH.",
                "Naturally captures recursive processes, remissions, relapses, and cumulative lifetime survival.",
                "Can be built and validated transparently in both spreadsheet and programmatic environments."
            ],
            "Cons": [
                "Markovian property can oversimplify patient disease trajectories if history matters (e.g. number of prior strokes).",
                "Computational complexity increases exponentially if many tunnel states or patient covariates are tracked.",
                "Requires robust empirical transition probabilities and long-term extrapolation beyond trial horizons."
            ]
        }
    },

    "psa_ceac": {
        "name": "Probabilistic Sensitivity Analysis (PSA) & CEAC Curves",
        "short_name": "PSA & CEAC",
        "icon": "🎲",
        "category": "Uncertainty & Decision Analysis",
        "tagline": "Quantifying decision uncertainty via Monte Carlo sampling, parameter probability distributions, and Cost-Effectiveness Acceptability Curves.",
        "mathematical_formulation": r"""
### 🎲 Mathematical Formulation: Probabilistic Sensitivity Analysis (PSA) & CEAC

Economic evaluations contain parametric uncertainty across all inputs (clinical trial hazard ratios, health utilities, resource unit costs). 
NICE and ISPOR guidelines mandate **Probabilistic Sensitivity Analysis (PSA)** to establish whether a technology can be considered cost-effective 
given total parameter uncertainty.

#### 1. Statistical Parameter Distribution Mapping:
Parameters must be sampled from probability distributions matching their mathematical support:
* **Utilities ($U \in [0, 1]$) & Probabilities ($p \in [0, 1]$):** Sampled from **Beta distributions**:
  $$\alpha = \bar{p} \left( \frac{\bar{p}(1 - \bar{p})}{s^2} - 1 \right), \quad \beta = (1 - \bar{p}) \left( \frac{\bar{p}(1 - \bar{p})}{s^2} - 1 \right)$$
* **Costs ($C > 0$):** Positively skewed, strictly non-negative; sampled from **Gamma distributions**:
  $$\alpha = \frac{\bar{C}^2}{s^2}, \quad \beta = \frac{s^2}{\bar{C}} \quad (\text{shape } \alpha, \text{ scale } \beta)$$
* **Relative Risks / Hazard Ratios ($HR > 0$):** Sampled from **Log-Normal distributions** with $\mu_{\ln} = \ln(\hat{HR})$ and $\sigma_{\ln} = \frac{\ln(HR_{\text{upper}}) - \ln(HR_{\text{lower}})}{2 \times 1.96}$.

#### 2. Monte Carlo Iterations:
For $b = 1, \dots, B$ simulation draws (typically $B = 1,000$ to $5,000$ iterations):
1. Sample vector of all input parameters $\theta^{(b)} \sim f(\theta)$.
2. Calculate incremental costs $\Delta C^{(b)}$ and incremental QALYs $\Delta E^{(b)}$.
3. Calculate Net Monetary Benefit: $\text{NMB}^{(b)}(\lambda) = \lambda \cdot \Delta E^{(b)} - \Delta C^{(b)}$.

#### 3. Cost-Effectiveness Acceptability Curve (CEAC):
The CEAC plots the probability that the novel technology is cost-effective as a function of the threshold $\lambda$:
$$P(\text{Cost-Effective} \mid \lambda) = \frac{1}{B} \sum_{b=1}^B \mathbb{I}\left( \text{NMB}^{(b)}(\lambda) > 0 \right)$$

Where $\mathbb{I}(\cdot)$ is the indicator function ($1$ if true, $0$ if false).

#### 4. Expected Value of Perfect Information (EVPI):
EVPI estimates the maximum financial amount a healthcare system should pay to fund further clinical research to resolve parameter uncertainty:
$$\text{EVPI}(\lambda) = \mathbb{E}_\theta\left[ \max_k \text{NMB}_k(\theta, \lambda) \right] - \max_k \mathbb{E}_\theta\left[ \text{NMB}_k(\theta, \lambda) \right]$$
""",
        "plain_english_glossary": {
            "PSA (Probabilistic Sensitivity Analysis)": "A simulation technique that runs the economic model thousands of times, with all parameters randomly sampled from their statistical uncertainty distributions simultaneously.",
            "CEAC (Cost-Effectiveness Acceptability Curve)": "A graph showing the probability that a new treatment is the best choice across different values of willingness to pay per QALY (e.g. from £0 up to £100,000).",
            "Monte Carlo Simulation": "A computational algorithm that relies on repeated random sampling to compute numerical results and propagate uncertainty.",
            "Beta Distribution": "A continuous probability distribution constrained between 0 and 1, mathematically ideal for proportions and health utility scores.",
            "Gamma Distribution": "A continuous probability distribution constrained to positive numbers with a right-hand skew, ideal for healthcare costs where most patients incur modest expenses but a few incur catastrophic costs.",
            "EVPI (Expected Value of Perfect Information)": "The expected cost of healthcare uncertainty: the maximum justifiable spending on a new trial to eliminate decision error."
        },
        "assumptions": [
            "Independent parameter sampling (unless joint correlated parameter matrices or Cholesky decomposition is used for correlated inputs).",
            "Adequate simulation sample size (typically B >= 1,000 iterations to achieve stability in tail probabilities).",
            "Appropriate distributional shape selection (Beta for utilities, Gamma for costs)."
        ],
        "pros_cons": {
            "Pros": [
                "Mandatory standard for NICE technology appraisals and international HTA dossier submissions.",
                "Directly communicates decision risk and probability of making a sub-optimal reimbursement decision.",
                "Naturally informs research prioritization and value-of-information decisions (EVPI)."
            ],
            "Cons": [
                "Computationally intensive when combined with complex microsimulation or dynamic transmission models.",
                "Can give false precision if parameter covariance structures (e.g. cost-effectiveness correlation) are ignored.",
                "Non-technical stakeholders often find the CEAC probability curves challenging to interpret without guidance."
            ]
        }
    },

    "budget_impact": {
        "name": "Budget Impact Analysis (BIA) & Payer Affordability",
        "short_name": "Budget Impact Analysis",
        "icon": "📊",
        "category": "Financial & Health Service Planning",
        "tagline": "Evaluating 1-to-5 year total financial cash outlay, eligible patient epidemiology, market uptake, and disinvestment savings for NHS commissioners.",
        "mathematical_formulation": r"""
### 📊 Mathematical Formulation: Budget Impact Analysis (BIA)

While Cost-Effectiveness Analysis evaluates long-term economic efficiency (value for money per QALY across a lifetime), 
**Budget Impact Analysis (BIA)** evaluates **affordability and financial feasibility** for the healthcare payer over a pragmatic 
1-to-5 year operational budget horizon (ISPOR Principles of Good Practice for BIA).

An intervention can be highly cost-effective ($\text{ICER} < £20,000/\text{QALY}$) yet unaffordable if the target patient population is enormous.

#### 1. Eligible Target Population Sizing:
$$N_{\text{eligible}, t} = N_{\text{total}} \times p_{\text{prev}} \times p_{\text{diag}} \times p_{\text{eligible}}$$

Where:
* $N_{\text{total}}$ = Total regional or national catchment population (e.g., NHS England adult population $\approx 45,000,000$).
* $p_{\text{prev}}$ = Disease prevalence or annual incidence rate.
* $p_{\text{diag}}$ = Proportion of individuals diagnosed.
* $p_{\text{eligible}}$ = Proportion meeting the clinical guideline indication.

#### 2. Market Penetration & Uptake Trajectory:
In year $t \in \{1, 2, 3, 4, 5\}$, the treated cohort is split between the novel technology and existing comparators:
$$N_{\text{new}, t} = N_{\text{eligible}, t} \times m_{\text{new}, t}$$
$$N_{\text{comp}, t} = N_{\text{eligible}, t} \times (1 - m_{\text{new}, t})$$

Where $m_{\text{new}, t}$ is the market share of the novel therapy in year $t$.

#### 3. Total Annual Healthcare Expenditure Scenarios:
* **World Without New Intervention (Status Quo):**
  $$\text{Budget}_{\text{current}, t} = N_{\text{eligible}, t} \times C_{\text{current}}$$
* **World With New Intervention:**
  $$\text{Budget}_{\text{new}, t} = (N_{\text{new}, t} \times C_{\text{new}}) + (N_{\text{comp}, t} \times C_{\text{current}})$$

Where $C_{\text{new}}$ and $C_{\text{current}}$ include drug acquisition, administration, monitoring, and hospital admission offsets.

#### 4. Net Annual Budget Impact:
$$\Delta \text{Budget}_t = \text{Budget}_{\text{new}, t} - \text{Budget}_{\text{current}, t} = N_{\text{new}, t} \times (C_{\text{new}} - C_{\text{current}})$$

#### 5. Cumulative 5-Year Budget Impact:
$$\Delta \text{Budget}_{\text{total}} = \sum_{t=1}^5 \Delta \text{Budget}_t$$

*Note: In accordance with ISPOR and NICE guidelines, budget impact analysis figures are undiscounted to reflect actual cash-flow accounting in financial planning.*
""",
        "plain_english_glossary": {
            "Budget Impact Analysis (BIA)": "A financial forecasting model estimating the net cash change in a healthcare budget resulting from adopting and adopting a new medical technology over 1 to 5 years.",
            "Cost-Effectiveness vs Affordability": "Cost-effectiveness asks 'Is it worth the money over a lifetime?'; budget impact asks 'Can we afford the bill this financial year?'.",
            "Market Penetration / Uptake": "The percentage of eligible patients expected to receive the new drug each year as clinicians transition away from older therapies.",
            "Resource Offsets & Disinvestment": "Reductions in hospital stays, emergency admissions, or older drug purchases that partially reimburse the upfront cost of the new intervention.",
            "Undiscounted Cash Flow": "Financial budget amounts are not discounted into present value terms because financial directors must balance actual cash budgets year by year."
        },
        "assumptions": [
            "Fixed regional or national demographic catchment over the 5-year budget horizon.",
            "Stable unit acquisition costs over the horizon without mandatory renegotiated commercial framework rebates unless specified.",
            "Full adherence and completion rates unless explicit discontinuation curves are applied."
        ],
        "pros_cons": {
            "Pros": [
                "Essential for NHS England Integrated Care Boards (ICBs) and hospital trust finance directors to plan annual allocations.",
                "Directly translates epidemiological prevalence into tangible currency forecasts.",
                "Highlights commercial access agreements and patient access schemes (PAS) needed to make therapies affordable."
            ],
            "Cons": [
                "Ignores health benefits, quality of life, and QALY gains entirely.",
                "Limited to short-term horizons (typically 1-5 years), missing long-term disease prevention dividends.",
                "Highly sensitive to market uptake estimates, which can vary widely across clinical sites."
            ]
        }
    }
}

HTA_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "scenario_icer": {
        "title": "Severe Rheumatoid Arthritis: Novel Biologic DMARD vs. Standard Care",
        "method_key": "icer_plane",
        "description": "Evaluating a novel subcutaneous biologic disease-modifying anti-rheumatic drug (DMARD) against standard synthetic DMARDs for patients with active rheumatoid arthritis refractory to methotrexate.",
        "params": {
            "cost_intervention": 28400.0,
            "cost_comparator": 14200.0,
            "qaly_intervention": 6.85,
            "qaly_comparator": 6.15,
            "wtp_threshold": 25000.0
        }
    },
    "scenario_nmb": {
        "title": "Heart Failure Multi-Arm Comparison: SGLT2i vs ARNI vs Standard Care",
        "method_key": "nmb_nhb",
        "description": "Multi-arm health technology assessment ranking three competing clinical options for heart failure with reduced ejection fraction across NHS clinical commissioning groups.",
        "params": {
            "treatments": [
                {"name": "Standard Care (ACEi + Beta-Blocker)", "cost": 8500.0, "qalys": 4.20},
                {"name": "ARNI (Sacubitril/Valsartan)", "cost": 15800.0, "qalys": 4.62},
                {"name": "SGLT2i (Dapagliflozin Combo)", "cost": 19400.0, "qalys": 4.88}
            ],
            "wtp_threshold": 30000.0
        }
    },
    "scenario_markov": {
        "title": "Advanced Oncology 10-Year Markov Cohort: Immunotherapy vs Chemotherapy",
        "method_key": "markov_model",
        "description": "A 3-state Markov model (Progression-Free Survival, Progressed Disease, Death) evaluating 10-year survival, quality of life, and treatment costs with 3.5% NICE discounting and half-cycle correction.",
        "params": {
            "cycles": 10,
            "cycle_years": 1.0,
            "discount_rate": 0.035,
            "states": ["Progression-Free (PFS)", "Progressed Disease (PD)", "Dead"],
            "chemo_transitions": [
                [0.55, 0.35, 0.10],
                [0.00, 0.60, 0.40],
                [0.00, 0.00, 1.00]
            ],
            "immuno_transitions": [
                [0.72, 0.22, 0.06],
                [0.00, 0.65, 0.35],
                [0.00, 0.00, 1.00]
            ],
            "state_utilities": [0.82, 0.54, 0.00],
            "chemo_costs": [4500.0, 12000.0, 0.0],
            "immuno_costs": [18500.0, 12000.0, 0.0]
        }
    },
    "scenario_psa": {
        "title": "Cardiovascular Surgical Implant: 1,000-Iteration Monte Carlo PSA & CEAC",
        "method_key": "psa_ceac",
        "description": "Quantifying decision uncertainty and drawing Cost-Effectiveness Acceptability Curves for a novel minimally invasive heart valve replacement against open surgical replacement.",
        "params": {
            "iterations": 1000,
            "mean_cost_t": 32000.0,
            "sd_cost_t": 4500.0,
            "mean_cost_c": 24000.0,
            "sd_cost_c": 3800.0,
            "mean_qaly_t": 5.40,
            "sd_qaly_t": 0.45,
            "mean_qaly_c": 4.95,
            "sd_qaly_c": 0.50,
            "wtp_threshold": 30000.0
        }
    },
    "scenario_bia": {
        "title": "NHS England 5-Year Budget Impact: Novel Renal Protective Agent in CKD",
        "method_key": "budget_impact",
        "description": "5-year national budget impact analysis modeling uptake, pharmacy acquisition, and dialysis avoidance hospital admission offsets for 250,000 eligible chronic kidney disease patients.",
        "params": {
            "eligible_population": 250000,
            "uptake_rates": [0.05, 0.12, 0.22, 0.35, 0.48],
            "annual_cost_new": 1650.0,
            "annual_cost_current": 380.0,
            "annual_hospital_offset_per_treated": 620.0
        }
    }
}

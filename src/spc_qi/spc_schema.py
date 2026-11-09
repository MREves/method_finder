"""Statistical Process Control (SPC) & Quality Improvement (QI) Schema.

Defines the taxonomy, mathematical formulations, plain-English glossary,
identification rules, and healthcare worked scenarios for:
1. Healthcare Run Charts (Median Centerline & The 4 IHI / British Detection Rules)
2. Shewhart XmR / I-MR Charts (Individuals & Moving Range, 3-Sigma Limits)
3. Proportions: p-Chart & Laney p'-Chart (Binomial Rates & Overdispersion Correction)
4. Count Rates: u-Chart & Laney u'-Chart (Poisson Rates per 1,000 Bed-Days)
5. NHS "Making Data Count" System Icons & Process Capability (Cp / Cpk & Target Assurance)
"""

from typing import Dict, Any, List

SPC_METHODS: Dict[str, Dict[str, Any]] = {
    "healthcare_run_chart": {
        "name": "Healthcare Run Chart (Median & The 4 Detection Rules)",
        "category": "Statistical Process Control & Quality Improvement",
        "icon": "📈",
        "tag": "Sequential Median & Non-Parametric Detection",
        "summary": "Evaluates operational healthcare processes over time using a non-parametric median centerline and 4 validated probability rules (Shift, Trend, Runs Count, Astronomical Point) to identify non-random signals without assuming a Gaussian distribution.",
        "primary_estimand": "Non-Random Signal Detection & Median Trajectory",
        "assumptions": [
            "Data is ordered sequentially in chronological time (e.g. days, weeks, or months).",
            "Sufficient sample size is available (at least 10-12 data points; ideally 20-30 for establishing a reliable baseline).",
            "The baseline represents a stable observation period prior to intentional quality improvement changes.",
            "Observations are collected under consistent measurement definitions over time."
        ],
        "plain_english_glossary": {
            "$Y_t$": "**Observed Metric** — The sequential operational or clinical measurement at time point $t$ (e.g. monthly surgical site infections or weekly patient falls).",
            "$\\tilde{Y}$ (Median)": "**Centerline Median** — The exact middle value of the baseline data series, where precisely 50% of points fall above and 50% fall below. It is unaffected by extreme outliers.",
            "$R$ (Runs)": "**Number of Runs** — A run is a sequence of consecutive data points on one side of the median line. Every time the sequence crosses the median, a new run begins.",
            "$n_{\\text{useful}}$": "**Useful Observations** — Total sample size minus any data points that fall precisely on the median line ($n_{\\text{useful}} = N - \\text{median ties}$).",
            "**Rule 1 (Shift)**": "**Sustained Process Shift** — 6 or more consecutive data points falling entirely above or entirely below the median (probability $< 0.03$). Indicates a fundamental system change.",
            "**Rule 2 (Trend)**": "**Systematic Trend** — 5 or more consecutive data points continually increasing or continually decreasing without interruption.",
            "**Rule 3 (Runs Count)**": "**Too Few or Too Many Runs** — If total runs $R$ fall outside statistical tables $[R_{\\text{min}}, R_{\\text{max}}]$, the sequence exhibits non-random clustering or over-alternation.",
            "**Rule 4 (Astronomical Point)**": "**Blatant Outlier** — An extreme, anomalous data point that is immediately obvious to any clinical observer, reflecting an acute operational shock."
        },
        "mathematical_formulation": r"""
        ### Healthcare Run Chart Mathematical Formulation
        
        Given a chronologically ordered series of $N$ observations $Y_1, Y_2, \dots, Y_N$:
        
        #### 1. Centerline Calculation (Median):
        $$\tilde{Y} = \text{Median}(Y_1, \dots, Y_N)$$
        
        #### 2. Useful Observations:
        $$n_{\text{useful}} = \sum_{t=1}^N \mathbb{I}(Y_t \ne \tilde{Y})$$
        
        #### 3. Expected Number of Runs ($E[R]$) and Variance:
        Under the null hypothesis of pure random variation:
        $$E[R] = 1 + \frac{n_{\text{useful}}}{2}$$
        $$\text{Var}(R) = \frac{n_{\text{useful}} - 1}{4}$$
        
        Critical lower and upper limits $[R_{\text{min}}, R_{\text{max}}]$ for runs are evaluated against the 5% two-tailed thresholds published in the Perla, Provost, and Murray (2011) run chart reference tables.
        """,
        "when_to_use": [
            "Early-stage healthcare quality improvement projects with limited historical data ($N = 10 \text{ to } 20$).",
            "Non-normal, skewed operational metrics (e.g. length of stay or incident counts with zero inflation).",
            "When clinicians and frontline staff need an intuitive, transparent chart that does not rely on complex standard deviation formulas.",
            "Initial rapid Plan-Do-Study-Act (PDSA) cycle testing before establishing formal 3-sigma control limits."
        ],
        "when_not_to_use": [
            "When sufficient data ($N \ge 20 \text{ to } 30$) exists and a Shewhart control chart (XmR, p, u) can be constructed.",
            "When the denominator or population size fluctuates wildly from week to week (use p-chart or u-chart).",
            "When evaluating formal process capability ($C_p, C_{pk}$) against statutory specifications."
        ],
        "pros": [
            "Simple, highly intuitive, and transparent for frontline clinical staff.",
            "Completely distribution-free (non-parametric; requires no normality assumption).",
            "Robust against single extreme outliers because the median is used as the centerline."
        ],
        "cons": [
            "Less statistically sensitive to subtle shifts than 3-sigma Shewhart control charts.",
            "Does not account for changes in subgroup denominators or exposure opportunities.",
            "Cannot quantify formal statistical process capability or predict future defect rates."
        ]
    },

    "xmr_individual_chart": {
        "name": "Shewhart XmR / I-MR Chart (Individuals and Moving Range)",
        "category": "Statistical Process Control & Quality Improvement",
        "icon": "🎯",
        "tag": "Continuous Operational Metrics & 3-Sigma Limits",
        "summary": "Monitors continuous hospital operational metrics collected one at a time (subgroup size n=1), using the Average Moving Range to derive objective 3-sigma Upper and Lower Control Limits that separate common cause noise from special cause variation.",
        "primary_estimand": "Process Mean (X-bar), Process Dispersion (sigma-hat), Upper/Lower Control Limits (UCL / LCL)",
        "assumptions": [
            "Continuous measurement scale (e.g. hours, minutes, days, costs).",
            "Observations are gathered sequentially as individual data points ($n=1$).",
            "The baseline period is free from known major disruptions or prior interventions.",
            "Data distribution is approximately unimodal and symmetric (though Shewhart limits are robust to mild skewness)."
        ],
        "plain_english_glossary": {
            "$X_t$": "**Individual Measurement** — The sequential continuous value observed at time $t$ (e.g. average ED triage-to-admission hours for week $t$).",
            "$\\bar{X}$ (X-bar)": "**Process Mean Centerline** — The arithmetic average of all baseline continuous observations.",
            "$\\text{MR}_t$": "**Moving Range** — The absolute difference between two consecutive observations: $|X_t - X_{t-1}|$. It measures baseline short-term point-to-point variation.",
            "$\\overline{\\text{MR}}$": "**Average Moving Range** — The mean of all moving ranges across the baseline series.",
            "$d_2$": "**Unbiasing Constant ($d_2 = 1.128$)** — A mathematical factor derived from the normal distribution that converts the average moving range into an unbiased estimate of standard deviation: $\\hat{\\sigma} = \\overline{\\text{MR}} / 1.128$.",
            "$\\text{UCL}_X$": "**Upper Control Limit ($+3\\sigma$)** — The boundary calculated as $\\bar{X} + 2.66 \\times \\overline{\\text{MR}}$. A point above this indicates special cause variation.",
            "$\\text{LCL}_X$": "**Lower Control Limit ($-3\\sigma$)** — The boundary calculated as $\\bar{X} - 2.66 \\times \\overline{\\text{MR}}$ (bounded at zero if the metric cannot physically be negative).",
            "**Common Cause**": "**Routine Process Noise** — Natural variation inherent in the design of the current healthcare system. Tampering with common cause noise increases variability.",
            "**Special Cause**": "**External Signal** — A statistically significant change caused by an external shock, breakdown, or intentional quality improvement reform."
        },
        "mathematical_formulation": r"""
        ### Shewhart XmR / I-MR Chart Mathematical Formulation
        
        For a continuous series $X_1, X_2, \dots, X_k$:
        
        #### 1. Baseline Process Mean ($\bar{X}$):
        $$\bar{X} = \frac{1}{k} \sum_{t=1}^k X_t$$
        
        #### 2. Moving Range ($\text{MR}_t$) and Average Moving Range ($\overline{\text{MR}}$):
        $$\text{MR}_t = |X_t - X_{t-1}| \quad \text{for } t = 2, 3, \dots, k$$
        $$\overline{\text{MR}} = \frac{1}{k-1} \sum_{t=2}^k \text{MR}_t$$
        
        #### 3. Within-Subgroup Standard Deviation Estimator ($\hat{\sigma}$):
        $$\hat{\sigma} = \frac{\overline{\text{MR}}}{d_2} = \frac{\overline{\text{MR}}}{1.128}$$
        
        #### 4. Control Limits for Individuals ($X$-Chart):
        $$\text{UCL}_X = \bar{X} + 3 \hat{\sigma} = \bar{X} + \frac{3}{1.128} \overline{\text{MR}} = \bar{X} + 2.660 \cdot \overline{\text{MR}}$$
        $$\text{LCL}_X = \max\left(0, \bar{X} - 2.660 \cdot \overline{\text{MR}}\right)$$
        
        #### 5. Control Limits for Moving Range ($\text{MR}$-Chart):
        $$\text{Centerline}_{\text{MR}} = \overline{\text{MR}}$$
        $$\text{UCL}_{\text{MR}} = D_4 \cdot \overline{\text{MR}} = 3.267 \cdot \overline{\text{MR}}$$
        $$\text{LCL}_{\text{MR}} = D_3 \cdot \overline{\text{MR}} = 0$$
        """,
        "when_to_use": [
            "Tracking continuous healthcare operational metrics over time (e.g. average length of stay, turnaround times, theatre utilization %).",
            "Data arrives as single periodic summaries (e.g. one monthly or weekly aggregate average).",
            "Distinguishing routine operational fluctuations from genuine systemic breakthroughs or regressions.",
            "Providing objective 3-sigma boundaries for clinical governance and executive board oversight."
        ],
        "when_not_to_use": [
            "When the metric is a proportion or percentage derived from small, highly fluctuating denominators (use p-chart).",
            "When data is recorded as discrete counts per bed-day exposure (use u-chart).",
            "When data points are strongly autocorrelated time series requiring ARIMA or transfer functions."
        ],
        "pros": [
            "Gold standard for continuous operational healthcare quality monitoring.",
            "Separate X-chart (process level) and MR-chart (process consistency) gives deep diagnostic insight.",
            "Mathematically robust limits based on point-to-point moving ranges rather than naive sample variance."
        ],
        "cons": [
            "Assumes continuous scale; inappropriate for raw counts with low event volumes.",
            "Sensitive to extreme moving ranges if unedited; large one-off shocks inflate limits."
        ]
    },

    "p_chart_laney": {
        "name": "Proportion Charts: p-Chart & Laney p'-Chart (Overdispersion Correction)",
        "category": "Statistical Process Control & Quality Improvement",
        "icon": "📊",
        "tag": "Binomial Proportions & Overdispersion Correction",
        "summary": "Monitors binary clinical proportions (e.g. 30-day readmissions, surgical site infection rates) with variable sample sizes, featuring the Laney p'-chart transformation to prevent false alarms caused by large-sample overdispersion.",
        "primary_estimand": "Weighted Process Proportion (p-bar), Laney Overdispersion Factor (sigma_z), Dynamic Variable Limits",
        "assumptions": [
            "Binary classification for each unit (e.g. Readmitted vs. Not Readmitted, Infected vs. Clear).",
            "Sample size $n_i$ (the denominator) is known and recorded for every time period.",
            "Individual trials are approximately independent within each time period.",
            "When subgroup sizes are large ($n_i > 2,000$), the Laney $p'$-chart adjustment must be audited for overdispersion."
        ],
        "plain_english_glossary": {
            "$x_i$": "**Observed Defect Count** — Number of patients experiencing the event in period $i$ (e.g. 42 surgical site infections).",
            "$n_i$": "**Subgroup Sample Size** — Total population exposed in period $i$ (e.g. 1,250 surgical operations performed).",
            "$p_i$": "**Observed Sample Proportion** — Period rate: $p_i = x_i / n_i$.",
            "$\\bar{p}$ (p-bar)": "**Weighted Process Proportion Centerline** — Total defect count across all periods divided by total exposed population: $\\sum x_i / \\sum n_i$.",
            "$z_i$": "**Standardized Z-Score** — Unitless score quantifying how many standard deviations period proportion $p_i$ is from centerline $\\bar{p}$.",
            "$\\sigma_z$ (Laney Factor)": "**Laney Overdispersion Multiplier** — Ratio of observed variation between subgroups to expected binomial variation. If $\\sigma_z > 1.30$, standard binomial control limits are artificially narrow and will sound false alarms.",
            "$\\text{UCL}_i / \\text{LCL}_i$": "**Dynamic Control Limits** — Variable boundaries that widen during low-volume months and narrow during high-volume months."
        },
        "mathematical_formulation": r"""
        ### p-Chart & Laney p'-Chart Mathematical Formulation
        
        For $k$ subgroups with defect counts $x_i$ and sample sizes $n_i$:
        
        #### 1. Weighted Centerline Proportion ($\bar{p}$):
        $$\bar{p} = \frac{\sum_{i=1}^k x_i}{\sum_{i=1}^k n_i}$$
        
        #### 2. Standard Binomial Control Limits (Standard p-Chart):
        $$\text{UCL}_i = \bar{p} + 3 \sqrt{\frac{\bar{p}(1 - \bar{p})}{n_i}}$$
        $$\text{LCL}_i = \max\left(0, \bar{p} - 3 \sqrt{\frac{\bar{p}(1 - \bar{p})}{n_i}}\right)$$
        
        #### 3. Laney Standardized Transformation ($z_i$):
        $$z_i = \frac{p_i - \bar{p}}{\sqrt{\frac{\bar{p}(1 - \bar{p})}{n_i}}}$$
        
        #### 4. Laney Overdispersion Multiplier ($\sigma_z$):
        $$\text{MR}_i = |z_i - z_{i-1}|$$
        $$\sigma_z = \frac{\overline{\text{MR}}_z}{1.128} = \frac{1}{1.128 (k-1)} \sum_{i=2}^k |z_i - z_{i-1}|$$
        
        #### 5. Laney p'-Chart Control Limits:
        $$\text{UCL}_{i, \text{Laney}} = \bar{p} + 3 \cdot \sigma_z \sqrt{\frac{\bar{p}(1 - \bar{p})}{n_i}}$$
        $$\text{LCL}_{i, \text{Laney}} = \max\left(0, \bar{p} - 3 \cdot \sigma_z \sqrt{\frac{\bar{p}(1 - \bar{p})}{n_i}}\right)$$
        """,
        "when_to_use": [
            "Tracking binary compliance or complication rates (e.g. readmission %, patient fall %, hand hygiene compliance %).",
            "Hospital populations with fluctuating denominators from month to month.",
            "Large-volume hospitals where standard p-chart limits become unrealistically narrow, generating constant false alarms (use Laney p').",
            "NHS Trust performance reporting under 'Making Data Count' guidelines."
        ],
        "when_not_to_use": [
            "When events can occur multiple times per patient (e.g. 3 falls by the same patient; use u-chart).",
            "When sample size $n_i$ is extremely small ($n_i < 20$) or expected count $n_i \bar{p} < 5$.",
            "Continuous measurements like length of stay or waiting time in minutes (use XmR chart)."
        ],
        "pros": [
            "Correctly accounts for varying monthly sample sizes across hospital departments.",
            "Laney correction completely eliminates the 'Large N Trap' of false alarm over-signaling.",
            "Directly compatible with binomial healthcare quality indicators."
        ],
        "cons": [
            "Stepped/variable limits require careful explanation to non-technical stakeholders.",
            "Requires recording both numerator (events) and denominator (trials) for every period."
        ]
    },

    "u_chart_laney": {
        "name": "Count Rates: u-Chart & Laney u'-Chart (Poisson Rates per 1,000 Bed-Days)",
        "category": "Statistical Process Control & Quality Improvement",
        "icon": "🔢",
        "tag": "Poisson Rates & Area-of-Opportunity Exposure",
        "summary": "Tracks rates of non-conformities per unit of exposure (e.g. patient falls or catheter infections per 1,000 occupied bed-days) where patients can suffer multiple events, featuring Laney u'-chart overdispersion adjustments.",
        "primary_estimand": "Weighted Mean Rate (u-bar), Laney Dispersion Factor (sigma_z), Dynamic Rate Limits",
        "assumptions": [
            "Events follow an underlying Poisson count process (discrete integers: 0, 1, 2, 3...).",
            "The area of opportunity (exposure denominator $n_i$, such as 1,000 bed-days or machine-hours) is recorded.",
            "Multiple events can occur to a single unit or patient.",
            "Laney $u'$-chart adjustment is applied whenever between-subgroup variation exceeds Poisson variance ($\sigma_z > 1.30$)."
        ],
        "plain_english_glossary": {
            "$c_i$": "**Observed Event Count** — Total number of events occurring in period $i$ (e.g. 18 inpatient falls).",
            "$n_i$": "**Area of Opportunity (Exposure Units)** — Denominator expressed in standard units (e.g. $4.50 = 4,500$ occupied bed-days / 1,000).",
            "$u_i$": "**Observed Incidence Rate** — Rate per unit exposure: $u_i = c_i / n_i$ (e.g. 4.0 falls per 1,000 bed-days).",
            "$\\bar{u}$ (u-bar)": "**Weighted Mean Process Rate** — Total events divided by total exposure across all periods: $\\sum c_i / \\sum n_i$.",
            "$\\sigma_z$ (Laney Factor)": "**Poisson Overdispersion Factor** — Quantifies extra-Poisson variance due to unmeasured ward differences or seasonal illness waves.",
            "$\\text{UCL}_i / \\text{LCL}_i$": "**Dynamic 3-Sigma Rate Limits** — Upper and lower expected boundaries based on denominator size."
        },
        "mathematical_formulation": r"""
        ### u-Chart & Laney u'-Chart Mathematical Formulation
        
        For $k$ periods with event counts $c_i$ and exposure units $n_i$:
        
        #### 1. Weighted Mean Rate ($\bar{u}$):
        $$\bar{u} = \frac{\sum_{i=1}^k c_i}{\sum_{i=1}^k n_i}$$
        
        #### 2. Standard Poisson Control Limits (Standard u-Chart):
        $$\text{UCL}_i = \bar{u} + 3 \sqrt{\frac{\bar{u}}{n_i}}$$
        $$\text{LCL}_i = \max\left(0, \bar{u} - 3 \sqrt{\frac{\bar{u}}{n_i}}\right)$$
        
        #### 3. Standardized Z-Score Transformation:
        $$z_{u, i} = \frac{u_i - \bar{u}}{\sqrt{\bar{u} / n_i}}$$
        
        #### 4. Laney Dispersion Factor ($\sigma_z$):
        $$\sigma_z = \frac{1}{1.128 (k-1)} \sum_{i=2}^k |z_{u, i} - z_{u, i-1}|$$
        
        #### 5. Laney u'-Chart Control Limits:
        $$\text{UCL}_{i, \text{Laney}} = \bar{u} + 3 \cdot \sigma_z \sqrt{\frac{\bar{u}}{n_i}}$$
        $$\text{LCL}_{i, \text{Laney}} = \max\left(0, \bar{u} - 3 \cdot \sigma_z \sqrt{\frac{\bar{u}}{n_i}}\right)$$
        """,
        "when_to_use": [
            "Measuring incidence rates per unit of time or space (e.g. falls per 1,000 bed-days, pressure ulcers per 10,000 admissions).",
            "Healthcare environments where a single patient can suffer multiple incidents during their stay.",
            "Monitoring hospital-acquired infections (CAUTI, CLABSI, SSI) across hospital directorates."
        ],
        "when_not_to_use": [
            "Binary events where each unit can only experience the event once (e.g. mortality or 30-day readmission; use p-chart).",
            "When exposure denominator $n_i$ cannot be quantified.",
            "Continuous measurement values like patient age or blood pressure (use XmR chart)."
        ],
        "pros": [
            "Statistically standard for patient safety incident reporting (falls, infections, medication errors).",
            "Adjusts naturally for differing ward occupancy and lengths of stay.",
            "Laney correction prevents overdispersion false alarms in high-volume trusts."
        ],
        "cons": [
            "Requires reliable tracking of both incident counts and occupied bed-day exposure denominators."
        ]
    },

    "making_data_count_capability": {
        "name": "NHS 'Making Data Count' Icons & Process Capability (Cp / Cpk)",
        "category": "Statistical Process Control & Quality Improvement",
        "icon": "🧭",
        "tag": "NHS England Plot The Dots & Target Assurance",
        "summary": "Implements NHS England's standard 'Making Data Count' (MDC) operational governance framework, pairing statistical process limits with national targets through Variation Icons, Assurance Icons, and Process Capability metrics (Cp, Cpk).",
        "primary_estimand": "Process Capability Ratio (Cp), Min Capability Index (Cpk), NHS Variation & Assurance Icon Classification",
        "assumptions": [
            "The operational process is in statistical control (free of special cause variation) before capability is evaluated.",
            "National statutory target or specification limit (USL or LSL) is clearly defined (e.g. 4-hour ED standard >= 95%).",
            "Data follows approximately normal distribution for valid Gaussian capability formulas."
        ],
        "plain_english_glossary": {
            "**Variation Icon**": "**Process Stability Signal** — Classifies the current pattern into: Common Cause (Grey: No significant change), Special Cause of Concern (Orange: Deteriorating performance), or Special Cause of Improvement (Blue: Favorable change).",
            "**Assurance Icon**": "**Target Capability Signal** — Evaluates whether the system's control limits can achieve the standard: Consistently Passing (Blue: Process capable), Hit-and-Miss / Flapping (Yellow: Target lies between limits), or Consistently Failing (Orange: Process incapable).",
            "$\\text{USL} / \\text{LSL}$": "**Upper / Lower Specification Limit** — External statutory target set by NHS England or NICE (e.g. maximum acceptable waiting time $\\text{USL} = 4.0$ hours).",
            "$C_p$": "**Potential Process Capability** — Ratio of specification spread to natural process spread ($6\\hat{\\sigma}$). Measures whether the system could pass if centered perfectly.",
            "$C_{pk}$": "**Actual Process Capability** — Accounts for process centering relative to the specification limit: $\\min((\\text{USL} - \\mu)/3\\sigma, (\\mu - \\text{LSL})/3\\sigma)$. A value $\\ge 1.33$ indicates a capable system.",
            "**Tampering**": "**Over-Adjustment Trap** — Reacting to routine common cause variation as if it were a special cause. Tampering destabilizes clinical operations and degrades performance."
        },
        "mathematical_formulation": r"""
        ### NHS 'Making Data Count' & Process Capability Formulation
        
        #### 1. Potential Capability ($C_p$):
        $$C_p = \frac{\text{USL} - \text{LSL}}{6\hat{\sigma}}$$
        
        #### 2. Actual Process Capability ($C_{pk}$):
        $$C_{pu} = \frac{\text{USL} - \bar{X}}{3\hat{\sigma}}$$
        $$C_{pl} = \frac{\bar{X} - \text{LSL}}{3\hat{\sigma}}$$
        $$C_{pk} = \min\left(C_{pu}, C_{pl}\right)$$
        
        #### 3. NHS Assurance Categorization Rules:
        * **Consistently Passing (Blue):**
          - For desirable lower metrics (e.g. infection rate with $\text{USL}$): $\text{UCL} < \text{USL}$.
          - For desirable higher metrics (e.g. 4-hour standard with $\text{LSL}$): $\text{LCL} > \text{LSL}$.
        * **Hit-and-Miss / Flapping (Yellow):**
          - Target lies strictly between the control limits: $\text{LCL} \le \text{Target} \le \text{UCL}$.
        * **Consistently Failing (Orange):**
          - Entire process capability envelope fails the statutory target:
            - For desirable lower metrics: $\text{LCL} > \text{USL}$.
            - For desirable higher metrics: $\text{UCL} < \text{LSL}$.
        """,
        "when_to_use": [
            "Executive and board reporting under NHS England 'Making Data Count' guidelines.",
            "Assessing whether clinical services can realistically achieve national standards (ED 4-hour, RTT 18-week, Cancer 62-day).",
            "Educating management on the vital distinction between system capability and individual staff performance."
        ],
        "when_not_to_use": [
            "When a process is wildly unstable with uncontrolled special causes (stabilize the process before calculating $C_{pk}$).",
            "When no external specification target exists."
        ],
        "pros": [
            "Industry and NHS standard for board governance and operational performance review.",
            "Variation and Assurance icons synthesize statistical complexity into immediate decision guidance.",
            "Protects operational teams from being reprimanded for common cause fluctuations."
        ],
        "cons": [
            "Requires establishing external specification limits or contractual standards."
        ]
    }
}


SPC_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "run_chart_ssi": {
        "title": "Hospital Surgical Site Infections (SSI) Post-Cease Intervention",
        "description": "24 consecutive months of deep wound surgical site infections following elective orthopedic surgery. In month 13, a revised pre-op chlorhexidine skin prep protocol was introduced.",
        "method_key": "healthcare_run_chart",
        "unit_name": "Infections per month",
        "metric_label": "Surgical Site Infections",
        "baseline_periods": 12
    },
    "xmr_ed_wait": {
        "title": "Emergency Department Average Triage-to-Doctor Time",
        "description": "30 consecutive weeks of weekly average physician consultation waiting times in a major trauma centre emergency department.",
        "method_key": "xmr_individual_chart",
        "unit_name": "Hours to physician consultation",
        "metric_label": "Waiting Hours",
        "baseline_periods": 20
    },
    "p_chart_readmission": {
        "title": "30-Day Emergency Readmission Rate (Large Teaching Hospital)",
        "description": "24 consecutive months of emergency readmissions with fluctuating monthly discharges ranging from 1,200 to 2,800 patients. Evaluated with Laney p' correction.",
        "method_key": "p_chart_laney",
        "unit_name": "30-day readmissions (%)",
        "metric_label": "Readmission Rate",
        "baseline_periods": 18
    },
    "u_chart_falls": {
        "title": "Inpatient Falls per 1,000 Occupied Bed-Days (Elderly Care Wards)",
        "description": "24 consecutive months of inpatient falls across elderly care wards with occupied bed-day exposure tracked per 1,000 bed-days.",
        "method_key": "u_chart_laney",
        "unit_name": "Falls per 1,000 bed-days",
        "metric_label": "Falls Rate",
        "baseline_periods": 18
    },
    "mdc_ed_4hour": {
        "title": "Emergency Care 4-Hour Standard & Capability (NHS 95% Standard)",
        "description": "26 consecutive bi-weekly periods monitoring the percentage of emergency patients admitted, transferred, or discharged within 4 hours against the NHS 95% statutory target.",
        "method_key": "making_data_count_capability",
        "unit_name": "Patients treated within 4 hours (%)",
        "metric_label": "4-Hour Compliance %",
        "target_value": 95.0,
        "is_higher_better": True,
        "baseline_periods": 20
    }
}

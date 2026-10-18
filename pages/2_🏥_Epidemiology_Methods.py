"""Streamlit Page: Epidemiological & Public Health Methods.

Guides researchers, clinicians, and public health practitioners through study designs,
effect measures (RR, OR, IRR, RD), demographic rate standardization (DSR, SMR),
screening accuracy (Sensitivity, Specificity, PPV, NPV), and survival analysis.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is on Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.ui_components.epi_standardization_widgets import render_rate_standardization_module
from src.ui_components.epi_relative_risk_widgets import render_relative_risk_module
from src.ui_components.epi_odds_ratio_widgets import render_odds_ratio_module
from src.ui_components.epi_risk_difference_widgets import render_risk_difference_module
from src.ui_components.epi_incidence_rate_ratio_widgets import render_incidence_rate_ratio_module
from src.ui_components.epi_attributable_fraction_widgets import render_attributable_fraction_module
from src.ui_components.epi_diagnostic_accuracy_widgets import render_diagnostic_accuracy_module
from src.ui_components.epi_predictive_values_widgets import render_predictive_values_module
from src.ui_components.epi_likelihood_ratios_widgets import render_likelihood_ratios_module
from src.ui_components.epi_kaplan_meier_widgets import render_kaplan_meier_module
from src.ui_components.epi_cox_regression_widgets import render_cox_regression_module
from src.ui_components.epi_risk_stratification_widgets import render_risk_stratification_module
from src.ui_components.epi_mantel_haenszel_widgets import render_mantel_haenszel_module
from src.ui_components.epi_outbreak_widgets import render_outbreak_module
from src.ui_components.epi_meta_analysis_widgets import render_meta_analysis_module
from src.ui_components.sidebar_navigation import (
    render_custom_sidebar_navigation,
    render_roundtrip_navigation_banner
)
from src.ui_components.sample_size_badge import render_sample_size_deep_link_badge


st.set_page_config(
    page_title="Epidemiological Methods Finder",
    page_icon="🏥",
    layout="wide"
)

# Render Custom Grouped Sidebar Navigation
render_custom_sidebar_navigation("pages/2_🏥_Epidemiology_Methods.py")

st.title("🏥 Epidemiological & Public Health Methods")
render_roundtrip_navigation_banner()
st.markdown("""
Navigate the questions below to identify the appropriate study design, effect measure, 
demographic rate standardization method, or diagnostic screening evaluation for your public health investigation.
""")

# ----------------- Progress & Implementation Tracker -----------------
if st.session_state.get("debug_mode", False) or st.session_state.get("debug_model", False):
    with st.expander("📌 Epidemiological Methods Implementation Tracker", expanded=False):
        st.markdown("""
        **Project Status: Phase 3 — Epidemiological Methods (In Active Implementation)**  
        This tracker monitors the 17 core epidemiological methods across 9 thematic modules.
        """)
        st.markdown(r"""
        #### 🗺️ Module 1: Demographic Rate Standardization
        * [x] **Direct Age-Standardized Rate (DSR / ASR)** (European Standard Population 2013, Dobson 95% CIs)
        * [x] **Indirect Standardization & Standardized Mortality Ratio (SMR)** (Byar's Poisson 95% CIs)
        * [x] **Derbyshire & Derby City Geographical Spatial Case Study** (Interactive Folium / GeoPandas Choropleth & Reversal Paradox)

        #### ⚖️ Module 2: 2×2 Contingency Effect Measures & Study Designs
        * [x] **Relative Risk / Risk Ratio (RR)** (Prospective Cohort & Clinical Trials)
        * [x] **Odds Ratio (OR)** (Retrospective Case-Control & Cross-Sectional Studies)
        * [x] **Risk Difference (RD) / Attributable Risk (AR)** & **Number Needed to Treat (NNT/NNH)**
        * [x] **Incidence Rate Ratio (IRR with Person-Time)** (Dynamic Cohort Follow-up)

        #### 📊 Module 3: Population Burden & Impact Fractions
        * [x] **Attributable Fraction among the Exposed ($AF_e$)**
        * [x] **Population Attributable Fraction (PAF / PAR)** (Levin & Miettinen Formulas)

        #### 🎯 Module 4: Diagnostic Test Accuracy & Screening Evaluation
        * [x] **Sensitivity & Specificity (2×2 Matrix)**
        * [x] **Predictive Values (PPV & NPV) vs. Prevalence (Bayes' Theorem)**
        * [x] **Diagnostic Likelihood Ratios (LR+, LR-) & Fagan's Nomogram**

        #### ⏱️ Module 5: Longitudinal Survival Analysis in Epidemiology
        * [x] **Kaplan-Meier Survival Curves & Log-Rank Test**
        * [x] **Cox Proportional Hazards Regression (Adjusted Hazard Ratios)**

        #### 🗂️ Module 6: Clinical Risk Stratification & Prognostic Prediction Models
        * [x] **Clinical Risk Score Formulation, Stratification & Calibration** (Sullivan / Framingham Integer Points, TRIPOD Standards, Decision Curve Analysis)

        #### 📐 Module 7: Stratified Analysis, Confounding & Effect Modification
        * [x] **Cochran-Mantel-Haenszel (CMH) Stratified Analysis & Breslow-Day Homogeneity** (Robins-Breslow-Greenland 95% CIs, Simpson's Paradox Detection, 10% Confounding Rule, Tarone Adjustment)

        #### 🦠 Module 8: Outbreak Epidemiology, Attack Rates & Vehicle Incrimination
        * [x] **Food-Specific Attack Rates & Incrimination Matrix** (Relative Risk, Attack Rate Difference, Woolf 95% CIs)
        * [x] **Epidemic Curve & Incubation Period Distribution** (Point-source vs. Continuous vs. Propagated)
        * [x] **Secondary Attack Rate (SAR)** (Household / Close Contact Transmission & Wilson CIs)

        #### 📚 Module 9: Systematic Reviews & Meta-Analysis (Evidence Synthesis)
        * [x] **Fixed-Effect (Inverse-Variance) & DerSimonian-Laird Random-Effects Pooling** (Summary Odds/Risk Ratios, 95% CIs)
        * [x] **Between-Study Heterogeneity Metrics** (Cochran's Q, Higgins & Thompson's I², DerSimonian-Laird τ²)
        * [x] **Publication Bias & Funnel Plot Asymmetry** (Egger's Linear Regression Test & Pseudo-95% Limits)
        * [x] **Leave-One-Out Sensitivity Analysis & Prediction Intervals** (Higgins 95% Future Population Interval)
        """)

st.markdown("---")

# ----------------- Navigation Mode Selection -----------------
st.subheader("🧭 Choose Navigation Mode")

if "epi_nav_mode" not in st.session_state:
    st.session_state["epi_nav_mode"] = "direct"

MODE_GUIDED = "🎯 Guided Decision Finder (Route by study design or analytical task)"
MODE_DIRECT = "⚡ Direct Method Explorer (Select any epidemiological method from catalog)"

selected_mode_label = st.radio(
    "How would you like to explore epidemiological methods?",
    options=[MODE_GUIDED, MODE_DIRECT],
    index=1 if st.session_state["epi_nav_mode"] == "direct" else 0,
    horizontal=True,
    help="Switch between guided decision routing and direct selection of specific epidemiological methods."
)

is_direct_mode = (selected_mode_label == MODE_DIRECT)
st.session_state["epi_nav_mode"] = "direct" if is_direct_mode else "guided"

EPI_CATALOG = {
    "direct_standardization": {
        "name": "Direct Age-Standardized Rate (DSR / ASR)",
        "category": "Demographic Rate Standardization",
        "description": "Applies local age-specific rates to an external Standard Population (e.g. European Standard Population 2013) to eliminate demographic confounding when comparing regions.",
        "parametric": True,
        "when_to_use": [
            "Comparing disease or mortality rates across populations with differing age profiles.",
            "Evaluating geographical or temporal trends in chronic conditions (e.g., Derby City vs. Derbyshire Dales).",
            "Local age-specific counts are sufficiently large and reliable in every age band."
        ],
        "key_assumptions": [
            "Reliable local age-specific population denominators and event numerators.",
            "Uniform reference standard population weights applied consistently across all comparator regions.",
            "Age-specific rates should not cross wildly (homogeneity of rate ratios across strata)."
        ],
        "alternative_test": "Indirect Standardization & SMR (preferred when local age-specific counts are small, sparse, or unstable)."
    },
    "indirect_standardization": {
        "name": "Indirect Standardization & Standardized Mortality Ratio (SMR)",
        "category": "Demographic Rate Standardization",
        "description": "Applies standard reference benchmark rates to local population age structures to compute Expected Deaths (E), yielding SMR = Observed / Expected.",
        "parametric": True,
        "when_to_use": [
            "Local age-specific event counts are small, sparse, or subject to severe random fluctuation.",
            "Evaluating occupational cohorts, rare diseases, or small community health areas.",
            "Only total local deaths are available alongside age-specific population counts."
        ],
        "key_assumptions": [
            "Reference population age-specific rates accurately represent the expected benchmark.",
            "Proportionality: Relative mortality ratio is approximately constant across age groups.",
            "Cannot strictly compare SMRs between two study populations if their age structures differ markedly (must compare each to 1.0 or 100%)."
        ],
        "alternative_test": "Direct Standardization (DSR) or Poisson Multivariable Regression."
    },
    "relative_risk": {
        "name": "Relative Risk / Risk Ratio (RR)",
        "category": "Association & Risk in Cohorts & Trials",
        "description": "Measures the ratio of cumulative disease risk in the exposed group relative to the unexposed group in longitudinal cohort studies and randomized trials.",
        "parametric": True,
        "when_to_use": [
            "Study follows participants forward in time (Prospective Cohort or RCT).",
            "Outcome is cumulative incidence (risk) over a fixed observation window."
        ],
        "key_assumptions": ["Fixed cohort with complete follow-up.", "Independence of observations."],
        "alternative_test": "Odds Ratio (for case-control studies) or Incidence Rate Ratio (for person-time follow-up)."
    },
    "odds_ratio": {
        "name": "Odds Ratio (OR)",
        "category": "Association in Case-Control Studies",
        "description": "Quantifies the ratio of exposure odds in diseased cases versus healthy controls; approximates Relative Risk under the rare disease assumption.",
        "parametric": True,
        "when_to_use": [
            "Retrospective Case-Control studies where true population incidence cannot be calculated directly.",
            "Cross-sectional studies measuring prevalence odds.",
            "Binary multivariable logistic regression modeling."
        ],
        "key_assumptions": ["Appropriate control group sampling.", "Rare disease assumption if interpreting as RR."],
        "alternative_test": "Relative Risk (when study design is prospective cohort)."
    },
    "risk_difference": {
        "name": "Risk Difference (RD) & Number Needed to Treat (NNT/NNH)",
        "category": "Absolute Impact in Cohorts & Trials",
        "description": "Quantifies the absolute difference between exposed and unexposed cumulative incidence (I_e - I_u), directly measuring real-world clinical workload via Number Needed to Treat (NNT = 1 / ARR) or Harm (NNH = 1 / RD).",
        "parametric": True,
        "when_to_use": [
            "Quantifying absolute clinical workload, health economics, and real-world population impact.",
            "Translating relative reductions into actionable Number Needed to Treat (NNT) figures.",
            "Analyzing clinical trials or prospective cohorts where true population denominators are known."
        ],
        "key_assumptions": [
            "Baseline risk (I_u) must be known and representative of the target practice population.",
            "Fixed cohort or clinical trial with known group denominators and minimal attrition.",
            "Cannot be derived from case-control studies."
        ],
        "alternative_test": "Relative Risk (for relative biological strength) or Odds Ratio (for retrospective case-control studies)."
    },
    "incidence_rate_ratio": {
        "name": "Incidence Rate Ratio (IRR with Person-Time)",
        "category": "Dynamic Cohorts & Rate Ratios",
        "description": "Evaluates disease occurrence speed per unit of accumulated person-time (IR_e / IR_u) in dynamic or open cohorts with staggered entry, dropouts, or varying follow-up periods.",
        "parametric": True,
        "when_to_use": [
            "Dynamic or open cohorts where individuals enter at staggered dates or have varying follow-up durations.",
            "Occupational exposure surveillance, care home resident-days, or longitudinal registry data.",
            "Fitting multivariable Poisson regression models with ln(Person-Time) offset."
        ],
        "key_assumptions": [
            "Constant Hazard Assumption: Disease incidence rate is approximately uniform across observation time.",
            "Accurate person-time measurement from individual date of entry to date of exit/event.",
            "Independence of individual observation periods."
        ],
        "alternative_test": "Cox Proportional Hazards Regression (when hazard changes over time) or Relative Risk (when fixed cohort follow-up is uniform)."
    },
    "attributable_fraction": {
        "name": "Attributable Fractions (AF_e & PAF)",
        "category": "Population Burden & Impact Fractions",
        "description": "Quantifies the proportion of disease attributable to an exposure among exposed individuals (AF_e) and across the total community (PAF / Levin's & Miettinen's formulas), highlighting Geoffrey Rose's Prevention Paradox.",
        "parametric": True,
        "when_to_use": [
            "Translating biological risk ratios into real-world preventable disease burden.",
            "Public health policy, community-level resource allocation, and prevention prioritisation.",
            "Evaluating Rose's Prevention Paradox (modest risk with widespread prevalence vs. extreme risk with rare prevalence).",
            "Calculating Vaccine Efficacy (Prevented Fraction in Exposed, PF_e) or Population Prevented Fraction (PFP)."
        ],
        "key_assumptions": [
            "Causal Relationship: Biological and etiological causality (Bradford Hill criteria) is assumed.",
            "Confounding Control: Effect measure is free from major unadjusted confounding.",
            "Representative Exposure Prevalence: Sample P_e reflects target population demographic (or overridden with census data)."
        ],
        "alternative_test": "Risk Difference / NNT (for clinical trials) or Relative Risk (for etiological strength without causal attribution)."
    },
    "diagnostic_accuracy": {
        "name": "Diagnostic Accuracy: Sensitivity & Specificity (2×2 Matrix)",
        "category": "Diagnostic Test Accuracy & Screening",
        "description": "Evaluates screening and diagnostic test accuracy against gold standard clinical verification, quantifying True Positive Rate (TPR / Sensitivity), True Negative Rate (TNR / Specificity), False Positive Rate (FPR), False Negative Rate (FNR), Youden's J Index, and Diagnostic Odds Ratio (DOR) with Wilson score 95% confidence intervals.",
        "parametric": True,
        "when_to_use": [
            "Evaluating a new clinical diagnostic test, rapid antigen kit, or AI screening algorithm against an established Gold Standard.",
            "Assessing triage or population screening programs where minimizing false negatives (High Sensitivity Rules Out Disease: SnNOut) or false alarms (High Specificity Rules In Disease: SpPIn) is critical.",
            "Measuring overall diagnostic accuracy and composite informedness (Youden's Index J) independent of disease prevalence."
        ],
        "key_assumptions": [
            "Gold Standard Independence: The reference standard is strictly independent of the index test (no incorporation bias).",
            "Blinded Interpretation: Readers score the index test without prior knowledge of reference standard outcome (no review bias).",
            "Spectrum Representativeness: Study sample reflects the full real-world clinical target spectrum (no spectrum bias)."
        ],
        "alternative_test": "Positive Predictive Value (PPV) and Negative Predictive Value (NPV) when assessing individual patient post-test probability given a known disease prevalence."
    },
    "predictive_values": {
        "name": "Predictive Values (PPV & NPV) & Bayes' Theorem",
        "category": "Diagnostic Test Accuracy & Screening",
        "description": "Quantifies real-world clinical post-test probabilities of disease and health, measuring Positive Predictive Value (PPV), Negative Predictive Value (NPV), False Discovery Rate (FDR), and False Omission Rate (FOR) across changing population prevalences using Bayes' Theorem to expose the False Positive Paradox.",
        "parametric": True,
        "when_to_use": [
            "Communicating post-test disease probabilities directly to patients following positive or negative diagnostic screening results.",
            "Evaluating the impact of disease prevalence shifts between mass community screening (low prevalence) and hospital triage (high pre-test probability).",
            "Exposing the 'False Positive Paradox' when screening asymptomatic populations for rare conditions.",
            "Applying Bayes' Theorem to adjust predictive values for new target clinical settings without re-running laboratory trials."
        ],
        "key_assumptions": [
            "Prevalence Generalizability: The pre-test probability accurately reflects the target clinical demographic.",
            "Work-Up / Verification Control: Both test-positives and test-negatives undergo reference standard verification (no verification bias).",
            "Independent Reference Standard: Index test is completely separate from the gold standard definition (no incorporation bias)."
        ],
        "alternative_test": "Diagnostic Likelihood Ratios (LR+ & LR-) and Fagan's Nomogram for patient-specific pre-to-post test odds updating."
    },
    "likelihood_ratios": {
        "name": "Diagnostic Likelihood Ratios (LR+, LR-) & Fagan's Nomogram",
        "category": "Diagnostic Test Accuracy & Screening",
        "description": "Quantifies patient-specific pre-to-post-test diagnostic probability updating using the Odds Form of Bayes' Theorem and Fagan's Nomogram, measuring Positive Likelihood Ratio (Positive LR, LR+), Negative Likelihood Ratio (Negative LR, LR-), and Diagnostic Odds Ratio (DOR).",
        "parametric": True,
        "when_to_use": [
            "Updating disease suspicion for an individual patient starting from their unique pre-test probability.",
            "Evaluating a diagnostic test's rule-in power (Positive LR >= 10) or rule-out safety (Negative LR <= 0.10).",
            "Synthesizing Sensitivity and Specificity into portable multipliers that apply across varied clinical care settings.",
            "Visualizing Bayesian probability shifts on Fagan's Nomogram without complex manual calculations."
        ],
        "key_assumptions": [
            "Reference Standard Independence: The index test is evaluated against a strictly independent Gold Standard (no incorporation bias).",
            "Spectrum Representativeness: Study cohort reflects a realistic spectrum of mild, moderate, and severe clinical presentations (no spectrum bias).",
            "Blinded Diagnostic Review: Test interpreters score findings without prior knowledge of gold standard disease status."
        ],
        "alternative_test": "Predictive Values (PPV & NPV) when assessing population-wide post-test probabilities for a single fixed prevalence."
    },
    "kaplan_meier": {
        "name": "Kaplan-Meier Survival Curves & Log-Rank Test",
        "category": "Longitudinal Survival Analysis",
        "description": "Evaluates time-to-event outcomes, right-censoring, product-limit survival trajectories, Greenwood confidence bands, milestone survival benchmarks, and two-group Mantel-Cox log-rank comparisons.",
        "parametric": False,
        "when_to_use": [
            "Tracking prospective cohorts over longitudinal time until an event occurs (e.g. death, disease recurrence, graft failure).",
            "Handling subjects with incomplete follow-up or loss to follow-up through non-informative right-censoring.",
            "Estimating non-parametric survival probabilities S(t) and median survival times without assuming an underlying parametric distribution.",
            "Comparing survival distributions between two clinical arms using the Mantel-Cox Log-Rank Test."
        ],
        "key_assumptions": [
            "Independent / Non-Informative Censoring: The mechanism causing a patient to be censored is unrelated to their risk of experiencing the event.",
            "Secular Stability: Individuals recruited earlier in calendar time experience identical baseline risk and care to those recruited later.",
            "Exact Event Timing: Event and censoring times are accurately recorded on a continuous time scale.",
            "Proportional Hazards / Consistent Separation: For the Mantel-Cox Log-Rank test to have optimal statistical power, the hazard ratio should remain relatively constant without crossing survival curves."
        ],
        "alternative_test": "Cox Proportional Hazards Regression (when adjusting for multiple continuous or categorical baseline confounders simultaneously)."
    },
    "cox_regression": {
        "name": "Cox Proportional Hazards Regression (Adjusted Hazard Ratios)",
        "category": "Longitudinal Survival Analysis",
        "description": "Semi-parametric multivariable survival modeling for time-to-event outcomes, right-censoring, Adjusted Hazard Ratios (aHR), Breslow baseline hazard, and covariate-adjusted patient survival trajectories.",
        "parametric": False,
        "when_to_use": [
            "Estimating the independent effect of a treatment or exposure on time-to-event outcomes after adjusting for multiple baseline confounders.",
            "Analyzing longitudinal cohorts with variable follow-up duration and non-informative right-censoring.",
            "Computing Adjusted Hazard Ratios (aHR) and 95% Confidence Intervals for both continuous and categorical risk factors.",
            "Generating individualized predicted survival curves for specific patient risk profiles."
        ],
        "key_assumptions": [
            "Proportional Hazards Assumption: The hazard ratio between comparator groups remains constant over time (tested via Schoenfeld residuals).",
            "Independent / Non-Informative Censoring: Censoring mechanism is unrelated to the probability of experiencing the event.",
            "Linearity in the Log-Hazard: Continuous covariates have a log-linear relationship with hazard (or are modeled via splines/polynomials).",
            "Sufficient Events Per Variable (EPV): At least 10 to 15 confirmed events per model degree of freedom (Peduzzi et al. guideline)."
        ],
        "alternative_test": "Time-Stratified Cox Models or Accelerated Failure Time (AFT) models when the proportional hazards assumption is severely violated."
    },
    "risk_stratification": {
        "name": "Clinical Risk Stratification & Prognostic Prediction Models",
        "category": "Prognostic Modeling & Risk Stratification",
        "description": "Formulates multivariable integer point scores (Sullivan / Framingham methodology), segments populations into actionable risk strata (Low, Moderate, High), evaluates calibration (intercept, slope, Brier score) and discrimination (AUROC / C-Index), and measures clinical utility via Decision Curve Analysis (DCA Net Benefit).",
        "parametric": True,
        "when_to_use": [
            "Developing, validating, or implementing a clinical risk prediction score or prognostic rule (e.g. QRISK, CURB-65, eFI).",
            "Translating complex multivariable regression beta-coefficients into bedside integer point scores.",
            "Categorizing continuous disease risk into actionable clinical pathways (Low, Moderate, High tiers).",
            "Assessing model calibration (slope, intercept, Brier score) and clinical Net Benefit using Decision Curve Analysis (DCA)."
        ],
        "key_assumptions": [
            "Adequate Calibration: Calibration slope close to 1.00 (between 0.85 and 1.15) to prevent over- or under-treatment.",
            "Discrimination: AUROC / C-Index >= 0.70 for clinically meaningful separation of risk tiers.",
            "Sufficient Events per Tier: Minimum 10-20 confirmed events in each risk stratum to avoid extreme sampling instability.",
            "Positive Net Benefit: Decision Curve Analysis (DCA) demonstrates superior Net Benefit over 'Treat All' and 'Treat None' across relevant clinical threshold probabilities."
        ],
        "alternative_test": "Cox Proportional Hazards Regression (for longitudinal survival time-to-event) or Machine Learning Ensembles with post-hoc recalibration."
    },
    "mantel_haenszel": {
        "name": "Cochran-Mantel-Haenszel (CMH) Stratified Analysis & Effect Modification",
        "category": "Stratification, Confounding & Interaction",
        "description": "Precision-weighted pooling across K stratified 2x2 contingency tables (OR_MH with Robins-Breslow-Greenland 95% CIs, RR_MH), 10% change-in-estimate confounding control, Simpson's Paradox detection, and Breslow-Day / Tarone homogeneity tests for interaction / effect measure modification.",
        "parametric": True,
        "when_to_use": [
            "Controlling for a categorical confounding variable across stratified 2x2 contingency tables without multivariable regression.",
            "Testing whether an observed crude association is spurious or distorted by Simpson's Paradox.",
            "Evaluating whether the exposure-outcome effect is modified by a third factor (Effect Measure Modification / Interaction via Breslow-Day & Tarone tests).",
            "Computing Robins-Breslow-Greenland (1986) asymptotic variance for sparse or multi-center stratified data."
        ],
        "key_assumptions": [
            "Homogeneity of Odds Ratios: True stratum-specific odds ratios are approximately constant across strata (Breslow-Day p >= 0.05). If p < 0.05, reporting a single pooled summary is invalid.",
            "Stratum Independence: Individual 2x2 tables are statistically independent across strata.",
            "Adequate Cell Counts: At least 5 observations per stratum cell recommended, or Haldane 0.5 continuity correction applied for sparse tables."
        ],
        "alternative_test": "Multivariable Logistic Regression (for continuous confounders) or Stratum-Specific Reporting (when effect measure modification is present)."
    },
    "outbreak_investigation": {
        "name": "Outbreak Epidemiology, Attack Rates & Vehicle Incrimination",
        "category": "Field Outbreak & Infection Control",
        "description": "Calculates food-specific attack rates (AR_exp, AR_unexp), Relative Risk, Attack Rate Differences (ARD), and Chi-Square p-values across suspected exposure vehicles to incriminate contaminated items; evaluates epidemic curves (point-source vs. continuous vs. propagated) and median incubation periods; computes Secondary Attack Rates (SAR) with Wilson score 95% CIs.",
        "parametric": True,
        "when_to_use": [
            "Investigating an acute surge of gastrointestinal or respiratory cases among attendees of a common event, banquet, or healthcare ward.",
            "Comparing attack rates between exposed and unexposed individuals across multiple food or environmental items to incriminate the causal vehicle.",
            "Constructing an Epidemic Curve (Epi Curve) to classify outbreak transmission dynamics and determine median incubation duration.",
            "Measuring secondary household or close-contact communicability via the Secondary Attack Rate (SAR)."
        ],
        "key_assumptions": [
            "Complete Cohort Enumeration: Reliable attendee survey data on food/exposure consumption and symptom status.",
            "Uniform Case Definition: Clearly defined clinical criteria (symptoms, onset window, laboratory verification).",
            "Single Point Exposure: Common-source meal exposure time (T_0) must be verifiable for incubation calculation."
        ],
        "alternative_test": "Case-Control Odds Ratio (when the total attendee cohort cannot be fully enumerated) or Multivariable Logistic Regression (to control for multiple food co-consumptions)."
    },
    "meta_analysis": {
        "name": "Systematic Reviews & Meta-Analysis (Evidence Synthesis)",
        "category": "Evidence Synthesis & Meta-Analysis",
        "description": "Synthesizes quantitative evidence across K independent clinical trials or observational studies; computes Fixed-Effect (inverse-variance) and DerSimonian-Laird Random-Effects pooled estimates with 95% CIs and Higgins prediction intervals; evaluates between-study heterogeneity (Cochran's Q, tau^2, I^2); assesses publication bias via funnel plots and Egger's linear regression test; conducts leave-one-out sensitivity analyses.",
        "parametric": True,
        "when_to_use": [
            "Pooling effect sizes across multiple independent randomized trials or cohort studies addressing the same clinical question.",
            "Evaluating whether between-study heterogeneity is minimal (I^2 < 30%) or substantial (I^2 >= 30%) to guide model selection.",
            "Investigating publication bias and small-study effects via funnel plots and Egger's linear regression.",
            "Calculating Higgins 95% Prediction Intervals for the expected treatment effect in a future patient population."
        ],
        "key_assumptions": [
            "Clinical & Methodological Comparability: Included studies address sufficiently similar patient populations, interventions, and outcomes.",
            "Independent Study Samples: No overlapping or duplicate patient cohorts across included trials.",
            "Sufficient Study Volume: Minimum K >= 3 studies for pooling and between-study variance (tau^2) estimation."
        ],
        "alternative_test": "Individual Patient Data (IPD) Meta-Analysis (when raw participant records are available) or Network Meta-Analysis (for multiple competing treatments)."
    }
}

if not is_direct_mode:
    st.markdown("---")
    st.subheader("Step 1: What is your primary epidemiological objective?")
    selected_task = st.radio(
        "Select objective:",
        options=[
            "standardization",
            "cohort_risk",
            "population_impact",
            "case_control",
            "screening",
            "survival",
            "stratification",
            "cmh_confounding",
            "outbreak_investigation",
            "meta_analysis"
        ],
        format_func=lambda x: {
            "standardization": "🗺️ Compare disease/mortality rates across geographical regions or populations with different age profiles",
            "cohort_risk": "⏱️ Follow an exposed vs. unexposed cohort forward in time to assess disease risk (RR / IRR / RD)",
            "population_impact": "📊 Measure Population Burden & Preventable Cases: How much disease would disappear if we remove the hazard? (AF_e & PAF)",
            "case_control": "🔍 Start with diseased cases and look back at past exposures (Odds Ratio in Case-Control design)",
            "screening": "🎯 Evaluate a clinical diagnostic screening tool or rapid test kit (Sensitivity, Specificity, PPV, NPV, Likelihood Ratios)",
            "survival": "⏳ Longitudinal Survival Analysis: Follow subjects over time with right-censoring to evaluate time-to-event outcomes",
            "stratification": "🗂️ Clinical Risk Stratification & Prognostic Prediction Models (Integer Point Scores, Calibration, DCA Net Benefit)",
            "cmh_confounding": "⚖️ Stratified Analysis, Confounding Control & Interaction: Stratify 2x2 tables to control confounding or detect Simpson's Paradox / Effect Modification (Cochran-Mantel-Haenszel)",
            "outbreak_investigation": "🦠 Field Outbreak Investigation: Acute food poisoning or ward outbreak (Food-Specific Attack Rates, Epi Curve, Vehicle Incrimination, Secondary Attack Rate)",
            "meta_analysis": "📚 Systematic Review & Meta-Analysis: Synthesize evidence across multiple trials/studies (Fixed & Random Effects, I², Funnel Plot, Egger Test)"
        }[x]
    )

    if selected_task == "standardization":
        st.subheader("Step 2: Are your local age-specific event counts reliable and large?")
        counts_reliable = st.radio(
            "Local event counts sufficiency:",
            options=["reliable", "sparse"],
            format_func=lambda x: {
                "reliable": "✅ Yes: Local age-specific death/case counts are stable across all age bands -> Direct Standardization (DSR)",
                "sparse": "⚠️ No: Small area or rare event with zero/sparse counts in several age bands -> Indirect Standardization (SMR)"
            }[x]
        )
        active_method_key = "direct_standardization" if counts_reliable == "reliable" else "indirect_standardization"
    elif selected_task == "cohort_risk":
        st.subheader("Step 2: What is your cohort structure and effect measurement focus?")
        cohort_type = st.radio(
            "Cohort follow-up and metric type:",
            options=["relative_headcount", "rate_ratio_persontime", "absolute_risk"],
            format_func=lambda x: {
                "relative_headcount": "📈 Relative Risk (Risk Ratio, RR): Fixed closed cohort with headcount denominators",
                "rate_ratio_persontime": "⏱️ Incidence Rate Ratio (IRR with Person-Time): Dynamic open cohort with varying follow-up (person-years / resident-days)",
                "absolute_risk": "🏥 Absolute Risk Difference (RD & NNT/NNH): Absolute excess cases per 100 people and clinical workload"
            }[x]
        )
        if cohort_type == "relative_headcount":
            active_method_key = "relative_risk"
        elif cohort_type == "rate_ratio_persontime":
            active_method_key = "incidence_rate_ratio"
        else:
            active_method_key = "risk_difference"
    elif selected_task == "population_impact":
        active_method_key = "attributable_fraction"
    elif selected_task == "screening":
        st.subheader("Step 2: What is your primary diagnostic screening analytical focus?")
        screening_focus = st.radio(
            "Screening analytical focus:",
            options=["characteristics", "predictive_bayes", "likelihood_nomogram"],
            format_func=lambda x: {
                "characteristics": "🧪 Test Operating Characteristics (Fixed laboratory metrics: Sensitivity, Specificity, SnNOut, SpPIn, Youden's J)",
                "predictive_bayes": "🩺 Patient Post-Test Probabilities (Prevalence-dependent metrics: Positive & Negative Predictive Values [PPV, NPV], Bayes' Theorem)",
                "likelihood_nomogram": "⚖️ Pre-to-Post-Test Probability Updating (Patient-specific multipliers: Positive LR [LR+], Negative LR [LR-], Fagan's Nomogram)"
            }[x]
        )
        if screening_focus == "characteristics":
            active_method_key = "diagnostic_accuracy"
        elif screening_focus == "predictive_bayes":
            active_method_key = "predictive_values"
        else:
            active_method_key = "likelihood_ratios"
    elif selected_task == "survival":
        st.subheader("Step 2: What is your primary longitudinal survival analytical focus?")
        survival_focus = st.radio(
            "Survival analytical focus:",
            options=["kaplan_meier", "cox_regression"],
            format_func=lambda x: {
                "kaplan_meier": "📉 Non-Parametric Survival Curves & Two-Group Comparison (Kaplan-Meier Product-Limit & Mantel-Cox Log-Rank Test)",
                "cox_regression": "⚖️ Multivariable Risk Factor Adjustment (Cox Proportional Hazards Regression for Adjusted Hazard Ratios)"
            }[x]
        )
        active_method_key = "kaplan_meier" if survival_focus == "kaplan_meier" else "cox_regression"
    elif selected_task == "stratification":
        active_method_key = "risk_stratification"
    elif selected_task == "cmh_confounding":
        active_method_key = "mantel_haenszel"
    elif selected_task == "outbreak_investigation":
        active_method_key = "outbreak_investigation"
    elif selected_task == "meta_analysis":
        active_method_key = "meta_analysis"
    else:
        active_method_key = "odds_ratio"

else:
    st.markdown("---")
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        all_epi_keys = list(EPI_CATALOG.keys())
        default_epi_idx = 0
        preset_epi_key = st.session_state.get("epi_direct_method_key")
        if preset_epi_key in all_epi_keys:
            default_epi_idx = all_epi_keys.index(preset_epi_key)

        active_method_key = st.selectbox(
            "Select Epidemiological Method:",
            options=all_epi_keys,
            format_func=lambda k: f"{EPI_CATALOG[k]['name']} ({EPI_CATALOG[k]['category']})",
            index=default_epi_idx
        )
    with col_t2:
        st.info(f"**Domain:** {EPI_CATALOG[active_method_key]['category']}")

method_info = EPI_CATALOG[active_method_key]

# ----------------- Top Recommendation Card -----------------
st.markdown(f"""
<div style="background-color: #f0fdf4; border-left: 5px solid #10b981; padding: 1.25rem; border-radius: 6px; margin-bottom: 1.5rem; margin-top: 1rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #047857;">
            {method_info['category']}
        </span>
        <span style="font-size: 0.85rem; font-weight: 600; background: #d1fae5; color: #065f46; padding: 0.2rem 0.6rem; border-radius: 9999px;">
            🟢 Active Method
        </span>
    </div>
    <h2 style="margin: 0 0 0.5rem 0; color: #064e3b; font-size: 1.6rem;">{method_info['name']}</h2>
    <p style="margin: 0; font-size: 1.05rem; color: #1e293b; line-height: 1.5;">{method_info['description']}</p>
</div>
""", unsafe_allow_html=True)

render_sample_size_deep_link_badge(
    method_key=active_method_key,
    method_name=method_info.get("name", "Epidemiological Method"),
    pillar="Epidemiological Methods",
    origin_page="pages/2_🏥_Epidemiology_Methods.py",
    origin_title="Epidemiology Methods",
    origin_icon="🏥"
)


# ----------------- Method Rendering Controller -----------------
is_standardization = active_method_key in ["direct_standardization", "indirect_standardization"]
is_relative_risk = (active_method_key == "relative_risk")
is_odds_ratio = (active_method_key == "odds_ratio")
is_risk_difference = (active_method_key == "risk_difference")
is_incidence_rate_ratio = (active_method_key == "incidence_rate_ratio")
is_attributable_fraction = (active_method_key == "attributable_fraction")
is_diagnostic_accuracy = (active_method_key == "diagnostic_accuracy")
is_predictive_values = (active_method_key == "predictive_values")
is_likelihood_ratios = (active_method_key == "likelihood_ratios")
is_kaplan_meier = (active_method_key == "kaplan_meier")
is_cox_regression = (active_method_key == "cox_regression")
is_risk_stratification = (active_method_key == "risk_stratification")
is_mantel_haenszel = (active_method_key == "mantel_haenszel")
is_outbreak_investigation = (active_method_key == "outbreak_investigation")
is_meta_analysis = (active_method_key == "meta_analysis")

if is_standardization:
    render_rate_standardization_module(active_method_key, method_info)
elif is_relative_risk:
    render_relative_risk_module(active_method_key, method_info)
elif is_odds_ratio:
    render_odds_ratio_module(active_method_key, method_info)
elif is_risk_difference:
    render_risk_difference_module(active_method_key, method_info)
elif is_incidence_rate_ratio:
    render_incidence_rate_ratio_module(active_method_key, method_info)
elif is_attributable_fraction:
    render_attributable_fraction_module(active_method_key, method_info)
elif is_diagnostic_accuracy:
    render_diagnostic_accuracy_module(active_method_key, method_info)
elif is_predictive_values:
    render_predictive_values_module(active_method_key, method_info)
elif is_likelihood_ratios:
    render_likelihood_ratios_module(active_method_key, method_info)
elif is_kaplan_meier:
    render_kaplan_meier_module(active_method_key, method_info)
elif is_cox_regression:
    render_cox_regression_module(active_method_key, method_info)
elif is_risk_stratification:
    render_risk_stratification_module(active_method_key, method_info)
elif is_mantel_haenszel:
    render_mantel_haenszel_module(active_method_key, method_info)
elif is_outbreak_investigation:
    render_outbreak_module(active_method_key, method_info)
elif is_meta_analysis:
    render_meta_analysis_module(active_method_key, method_info)
else:
    st.info("""
    ### ⏳ Next Epidemiological Method
    All core epidemiological methods across 9 thematic modules are active and ready to explore!
    """)



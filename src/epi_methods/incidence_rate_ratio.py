"""Mathematical & Statistical Engine for Incidence Rate Ratio (IRR) with Person-Time.

Provides verified epidemiological calculations for:
1. Incidence Rates per unit of Person-Time: IR_e = a / PT_e and IR_u = c / PT_u.
2. Incidence Rate Ratio (IRR) point estimate: (a / PT_e) / (c / PT_u).
3. Incidence Rate Difference (IRD): (a / PT_e) - (c / PT_u).
4. Rothman-Greenland Log-Scale Standard Error: sqrt(1/a + 1/c).
5. Wald 95% Confidence Intervals for IRR: exp(ln(IRR) ± 1.96 * SE).
6. Linear Standard Error & 95% CI for Rate Difference: sqrt(a/PT_e^2 + c/PT_u^2).
7. Exact Poisson test and Wald hypothesis test p-values for H0: IRR = 1.0.
8. Prerequisite and methodological diagnostic evaluation engine (Poisson constant hazard assumption, person-time validity, event sparsity).
9. Realistic dynamic cohort synthetic data generation (Regional Lead Smelting Workers, Care Home Falls per Resident-Day, Telehealth COPD Emergency Admissions).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_incidence_rate_ratio(
    a: int,
    pt_e: float,
    c: int,
    pt_u: float,
    alpha: float = 0.05,
    rate_multiplier: float = 1000.0,
    continuity_correction: bool = False
) -> Dict[str, Any]:
    """Calculates Incidence Rates, IRR, Rate Difference (IRD), standard errors, and confidence intervals.

    A classical Person-Time epidemiological cohort table is structured as:
                          Incident Events (Cases)       Total Person-Time Follow-Up       Incidence Rate
    Exposed (+)                     a                              PT_e                     IR_e = a / PT_e
    Unexposed (-)                   c                              PT_u                     IR_u = c / PT_u
    Total                         a + c                         PT_e + PT_u                 IR_total = (a+c)/(PT_e+PT_u)

    Mathematical Formulations:
        Incidence Rate Ratio (IRR) = (a / PT_e) / (c / PT_u)
        Incidence Rate Difference (IRD) = (a / PT_e) - (c / PT_u)
        Log-scale Standard Error: SE(ln(IRR)) = sqrt(1/a + 1/c)
        95% Wald Confidence Interval for IRR: exp(ln(IRR) ± Z_(1 - alpha/2) * sqrt(1/a + 1/c))
        Linear Standard Error for IRD: SE(IRD) = sqrt(a / PT_e^2 + c / PT_u^2)
        95% Wald Confidence Interval for IRD: IRD ± Z_(1 - alpha/2) * SE(IRD)

    Args:
        a (int): Observed new incident events in the exposed group.
        pt_e (float): Total person-time accumulated by exposed cohort (e.g. person-years).
        c (int): Observed new incident events in the unexposed group.
        pt_u (float): Total person-time accumulated by unexposed cohort.
        alpha (float, optional): Significance level. Defaults to 0.05 (95% CI).
        rate_multiplier (float, optional): Multiplier for rates (e.g., 1,000 or 100,000). Defaults to 1000.0.
        continuity_correction (bool, optional): Add 0.5 to zero event numerators. Defaults to False.

    Returns:
        Dict[str, Any]: Comprehensive calculation dictionary containing:
            - 'a', 'c': Event counts.
            - 'pt_e', 'pt_u', 'pt_total': Person-time totals.
            - 'ir_exposed', 'ir_unexposed': Raw rates per person-time.
            - 'ir_exposed_scaled', 'ir_unexposed_scaled': Scaled rates (per multiplier).
            - 'irr': Incidence Rate Ratio point estimate.
            - 'ln_irr': Natural log of IRR.
            - 'se_ln_irr': Standard error of ln(IRR) = sqrt(1/a + 1/c).
            - 'irr_ci_lower', 'irr_ci_upper': 95% Wald CI for IRR.
            - 'ird': Rate Difference (unscaled).
            - 'ird_scaled': Rate Difference scaled per multiplier.
            - 'se_ird': Standard error of IRD.
            - 'ird_ci_lower', 'ird_ci_upper': 95% CI for scaled IRD.
            - 'z_score': Wald z-score on log scale: ln(IRR) / SE.
            - 'p_value_wald': Two-sided Wald p-value.
            - 'p_value_exact_poisson': Exact conditional binomial/Poisson test p-value.
            - 'effect_direction': 'harmful' (IRR > 1), 'protective' (IRR < 1), or 'null' (IRR == 1).
            - 'rate_multiplier': Multiplier used.
    """
    applied_cc = continuity_correction or (a == 0 or c == 0)

    a_calc = float(a) + (0.5 if applied_cc else 0.0)
    c_calc = float(c) + (0.5 if applied_cc else 0.0)
    pt_e_val = float(pt_e)
    pt_u_val = float(pt_u)
    pt_total = pt_e_val + pt_u_val

    # Unscaled raw rates per 1 unit of person-time
    ir_exp = (a_calc / pt_e_val) if pt_e_val > 0 else 0.0
    ir_unexp = (c_calc / pt_u_val) if pt_u_val > 0 else 0.0

    # Scaled rates per multiplier (e.g. per 1,000 person-years)
    ir_exp_scaled = ir_exp * rate_multiplier
    ir_unexp_scaled = ir_unexp * rate_multiplier

    # Rate Ratio (IRR) point estimate
    if ir_unexp > 0:
        irr = float(ir_exp / ir_unexp)
        ln_irr = float(np.log(irr))
    else:
        irr = np.nan
        ln_irr = np.nan

    # Rate Difference (IRD)
    ird = float(ir_exp - ir_unexp)
    ird_scaled = float(ird * rate_multiplier)

    # Standard Error of ln(IRR): sqrt(1/a + 1/c)
    if a_calc > 0 and c_calc > 0:
        se_ln_irr = float(np.sqrt((1.0 / a_calc) + (1.0 / c_calc)))
    else:
        se_ln_irr = np.nan

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    if not np.isnan(se_ln_irr) and not np.isnan(ln_irr):
        irr_ci_lower = float(np.exp(ln_irr - z_crit * se_ln_irr))
        irr_ci_upper = float(np.exp(ln_irr + z_crit * se_ln_irr))
        z_score = float(ln_irr / se_ln_irr) if se_ln_irr > 0 else 0.0
        p_value_wald = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
    else:
        irr_ci_lower = np.nan
        irr_ci_upper = np.nan
        z_score = np.nan
        p_value_wald = np.nan

    # Linear Standard Error of IRD: sqrt(a/PT_e^2 + c/PT_u^2)
    var_ird = (a_calc / (pt_e_val ** 2)) + (c_calc / (pt_u_val ** 2)) if (pt_e_val > 0 and pt_u_val > 0) else 0.0
    se_ird = float(np.sqrt(var_ird))
    se_ird_scaled = float(se_ird * rate_multiplier)
    ird_ci_lower = float((ird - z_crit * se_ird) * rate_multiplier)
    ird_ci_upper = float((ird + z_crit * se_ird) * rate_multiplier)

    # Exact Poisson test: Conditional on total cases (a + c), a follows Binomial(n = a + c, p = PT_e / PT_total)
    total_events = a + c
    if total_events > 0 and pt_total > 0:
        p_null = pt_e_val / pt_total
        # Two-sided exact binomial test
        try:
            binom_res = stats.binomtest(k=a, n=total_events, p=p_null, alternative="two-sided")
            p_value_exact_poisson = float(binom_res.pvalue)
        except Exception:
            p_value_exact_poisson = np.nan
    else:
        p_value_exact_poisson = np.nan

    # Effect direction
    if not np.isnan(irr):
        if irr > 1.0001:
            effect_direction = "harmful"
        elif irr < 0.9999:
            effect_direction = "protective"
        else:
            effect_direction = "null"
    else:
        effect_direction = "undefined"

    return {
        "a": int(a),
        "c": int(c),
        "pt_e": float(pt_e_val),
        "pt_u": float(pt_u_val),
        "pt_total": float(pt_total),
        "ir_exposed": float(ir_exp),
        "ir_unexposed": float(ir_unexp),
        "ir_exposed_scaled": float(ir_exp_scaled),
        "ir_unexposed_scaled": float(ir_unexp_scaled),
        "irr": float(irr),
        "ln_irr": float(ln_irr),
        "se_ln_irr": float(se_ln_irr),
        "irr_ci_lower": float(irr_ci_lower),
        "irr_ci_upper": float(irr_ci_upper),
        "ird": float(ird),
        "ird_scaled": float(ird_scaled),
        "se_ird": float(se_ird),
        "se_ird_scaled": float(se_ird_scaled),
        "ird_ci_lower": float(ird_ci_lower),
        "ird_ci_upper": float(ird_ci_upper),
        "z_score": float(z_score),
        "p_value_wald": float(p_value_wald),
        "p_value_exact_poisson": float(p_value_exact_poisson),
        "effect_direction": effect_direction,
        "rate_multiplier": float(rate_multiplier),
        "continuity_applied": applied_cc
    }


def evaluate_irr_prerequisites(
    study_design: str,
    a: int,
    pt_e: float,
    c: int,
    pt_u: float,
    is_constant_hazard_verified: bool = True
) -> List[Dict[str, Any]]:
    """Evaluates epidemiological prerequisites and bias diagnostics for Incidence Rate Ratio with Person-Time.

    Args:
        study_design (str): Epidemiological design architecture. Expected values:
            'dynamic_cohort', 'open_cohort', 'prospective_cohort', 'registry', 'case_control', 'cross_sectional'.
        a (int): Exposed incident events.
        pt_e (float): Exposed person-time.
        c (int): Unexposed incident events.
        pt_u (float): Unexposed person-time.
        is_constant_hazard_verified (bool, optional): Whether disease hazard is approximately constant across follow-up time. Defaults to True.

    Returns:
        List[Dict[str, Any]]: Diagnostic evaluation cards containing:
            - 'name': Name of assumption / criterion.
            - 'status': 'PASS', 'WARNING', or 'FAIL'.
            - 'details': Plain-English evaluation text.
            - 'recommendation': Specific epidemiological guidance.
    """
    evaluations: List[Dict[str, Any]] = []

    # 1. Study Design & Person-Time Suitability
    valid_pt_designs = {"dynamic_cohort", "open_cohort", "prospective_cohort", "registry", "retrospective_cohort"}
    if study_design.lower() in valid_pt_designs:
        evaluations.append({
            "name": "Person-Time Metric Suitability",
            "status": "PASS",
            "details": f"Study design ({study_design.replace('_', ' ').title()}) tracks individual participant entry, exit, and observation durations, providing valid Person-Time denominators.",
            "recommendation": "Incidence Rate Ratio (IRR) is mathematically valid and preferred over cumulative incidence."
        })
    elif study_design.lower() in {"case_control", "cross_sectional"}:
        evaluations.append({
            "name": "Person-Time Metric Suitability",
            "status": "FAIL",
            "details": f"Design is {study_design.replace('_', ' ').title()}. Participants are not followed longitudinally for person-time. Total person-time cannot be measured.",
            "recommendation": "Do NOT calculate Incidence Rate Ratio. Use Odds Ratio (for Case-Control) or Prevalence Ratio (for Cross-Sectional)."
        })
    else:
        evaluations.append({
            "name": "Person-Time Metric Suitability",
            "status": "WARNING",
            "details": f"Unrecognized design ({study_design}). Ensure exact follow-up durations were measured for every participant.",
            "recommendation": "Verify whether person-time was accurately calculated from date of entry to date of exit/event."
        })

    # 2. Constant Hazard / Rate Proportionality Assumption (Poisson Assumption)
    if is_constant_hazard_verified:
        evaluations.append({
            "name": "Constant Hazard (Poisson Assumption)",
            "status": "PASS",
            "details": "The disease incidence rate is assumed to remain relatively uniform across the follow-up window (e.g. constant hazard over calendar time and age).",
            "recommendation": "Single summary Incidence Rate Ratio is a faithful summary of effect."
        })
    else:
        evaluations.append({
            "name": "Constant Hazard (Poisson Assumption)",
            "status": "WARNING",
            "details": "The hazard rate changes substantially across age bands or follow-up duration (e.g. accelerating with age or wearing off over time).",
            "recommendation": "Stratify person-time into age/duration bands or use multivariable Poisson / Cox Proportional Hazards regression."
        })

    # 3. Event Numerator Sparsity (Poisson Approximation)
    min_events = min(a, c)
    if min_events >= 5:
        evaluations.append({
            "name": "Event Numerator Volume (Sample Size)",
            "status": "PASS",
            "details": f"Both exposed and unexposed groups have at least 5 incident events (minimum = {min_events}). Log-scale Wald CI sqrt(1/a + 1/c) is reliable.",
            "recommendation": "Asymptotic normal approximation for log(IRR) is robust."
        })
    elif min_events > 0:
        evaluations.append({
            "name": "Event Numerator Volume (Sample Size)",
            "status": "WARNING",
            "details": f"Sparse event numerators detected (minimum events = {min_events} < 5). Normal Wald approximation may exhibit slight asymmetry.",
            "recommendation": "Cross-reference Wald 95% CI with the Exact Conditional Poisson test p-value."
        })
    else:
        evaluations.append({
            "name": "Event Numerator Volume (Sample Size)",
            "status": "WARNING",
            "details": "One cohort has zero incident events. Standard Poisson rate ratio yields zero or infinity.",
            "recommendation": "Haldane-Anscombe (+0.5) continuity correction applied to numerators."
        })

    # 4. Denominator Person-Time Balance
    if pt_e > 0 and pt_u > 0:
        evaluations.append({
            "name": "Person-Time Denominator Completeness",
            "status": "PASS",
            "details": f"Total person-time is well-accounted ({pt_e + pt_u:,.1f} total units: {pt_e:,.1f} exposed, {pt_u:,.1f} unexposed).",
            "recommendation": "Denominator accurately accounts for staggered entry and varying follow-up lengths."
        })
    else:
        evaluations.append({
            "name": "Person-Time Denominator Completeness",
            "status": "FAIL",
            "details": "Zero person-time recorded for one or both groups. Rates cannot be computed.",
            "recommendation": "Ensure follow-up times are positive numbers."
        })

    return evaluations


def generate_incidence_rate_ratio_demo_data(
    scenario: str = "derbyshire_lead_foundry",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic dynamic cohort data for Incidence Rate Ratio pedagogical demonstrations.

    Provides three realistic public health scenarios:
    1. 'derbyshire_lead_foundry': Regional Historical Lead Smelting & Battery Recycling Dynamic Cohort vs. Unexposed Utility Workers (Incident Hypertension/CVD per 1,000 Person-Years).
    2. 'elderly_care_falls': All Meadows Residential Care Home Fall Prevention Flooring Initiative (Falls per 1,000 Resident-Days).
    3. 'telehealth_copd_admissions': All Meadows Community Remote Telemonitoring for Severe COPD vs. Routine GP Care (Emergency Hospital Admissions per 100 Person-Years).

    Args:
        scenario (str, optional): Scenario identifier. Defaults to 'derbyshire_lead_foundry'.
        seed (int, optional): Random seed for reproducible individual microdata generation. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - df_contingency: Aggregated Person-Time contingency summary table.
            - df_microdata: Individual participant dynamic cohort microdata.
            - meta: Rich descriptive metadata, clinical takeaways, and disclaimer notices.
    """
    np.random.seed(seed)

    scenarios = {
        "derbyshire_lead_foundry": {
            "title": "Regional Occupational Lead Smelting Dynamic Cohort (Harmful Exposure)",
            "domain": "Occupational Toxicology & Cardiovascular Epidemiology",
            "description": "A hypothetical 20-year dynamic occupational cohort (1995–2015) following 850 lead smelting and battery recycling workers across regional industrial sites versus 1,200 unexposed water utility workers. Workers entered at staggered hire dates and accumulated varying person-years until retirement, leaving, or diagnosis of severe hypertension/cardiovascular events.",
            "exposure_name": "Occupational Heavy Lead Exposure",
            "exposure_levels": ("Lead Exposed Workers", "Unexposed Utility Workers"),
            "outcome_name": "Incident Severe Cardiovascular Event",
            "outcome_levels": ("Incident CVD (+)", "Event-Free (-)"),
            "time_unit": "Person-Years",
            "rate_multiplier": 1000.0,
            "rate_multiplier_label": "per 1,000 Person-Years",
            "a": 78,              # Exposed CVD events
            "pt_e": 6240.0,       # Exposed person-years (~12.5 per 1,000 PY)
            "c": 45,              # Unexposed CVD events
            "pt_u": 9000.0,       # Unexposed person-years (~5.0 per 1,000 PY)
            "study_design": "dynamic_cohort",
            "expected_effect": "Harmful (IRR ≈ 2.50)",
            "clinical_takeaway": "Lead-exposed foundry workers had an incidence rate of 12.5 per 1,000 person-years compared to 5.0 per 1,000 person-years in unexposed workers (Incidence Rate Ratio = 2.50, 95% CI: [1.73, 3.61]). Heavy occupational lead exposure increases the rate of cardiovascular disease by 2.5-fold. The Rate Difference is +7.5 excess cases per 1,000 person-years.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "elderly_care_falls": {
            "title": "All Meadows Residential Care Home Fall Prevention Flooring Initiative",
            "domain": "Geriatric Healthcare & Patient Safety",
            "description": "A hypothetical quality-improvement open cohort tracking 18 residential care homes across All Meadows region over 18 months. Evaluates high-impact shock-absorbing safety flooring versus standard linoleum flooring on the incidence of fall-related fractures per 1,000 resident-days, accounting for varying lengths of resident stay.",
            "exposure_name": "Shock-Absorbing Safety Flooring",
            "exposure_levels": ("Safety Flooring Units", "Standard Flooring Units"),
            "outcome_name": "Fall with Severe Fracture",
            "outcome_levels": ("Fall Injury (+)", "No Injury (-)"),
            "time_unit": "Resident-Days",
            "rate_multiplier": 1000.0,
            "rate_multiplier_label": "per 1,000 Resident-Days",
            "a": 28,              # Events in safety units
            "pt_e": 56000.0,      # 56,000 resident-days (0.50 per 1,000 days)
            "c": 72,              # Events in standard units
            "pt_u": 60000.0,      # 60,000 resident-days (1.20 per 1,000 days)
            "study_design": "open_cohort",
            "expected_effect": "Protective (IRR ≈ 0.42)",
            "clinical_takeaway": "Safety flooring units had a fracture rate of 0.50 per 1,000 resident-days versus 1.20 per 1,000 in standard flooring units (Incidence Rate Ratio = 0.42, 95% CI: [0.27, 0.64]). The intervention produced a 58% relative rate reduction in fall-related fractures.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        },
        "telehealth_copd_admissions": {
            "title": "All Meadows Community Remote Tele-monitoring for Severe COPD",
            "domain": "Health Services Research & Chronic Disease Management",
            "description": "A hypothetical 3-year dynamic cohort evaluating daily digital pulse oximetry and symptom telemonitoring versus routine GP care in severe COPD patients across All Meadows Community Network. Evaluates the incidence rate of acute emergency hospital admissions per 100 person-years.",
            "exposure_name": "Digital Remote Telemonitoring",
            "exposure_levels": ("Telemonitoring Cohort", "Routine Primary Care"),
            "outcome_name": "Emergency Hospital Admission for COPD",
            "outcome_levels": ("Emergency Admission (+)", "No Admission (-)"),
            "time_unit": "Person-Years",
            "rate_multiplier": 100.0,
            "rate_multiplier_label": "per 100 Person-Years",
            "a": 54,              # Admissions in telemonitoring
            "pt_e": 360.0,        # 360 person-years (15.0 per 100 PY)
            "c": 108,             # Admissions in routine care
            "pt_u": 360.0,        # 360 person-years (30.0 per 100 PY)
            "study_design": "dynamic_cohort",
            "expected_effect": "Protective (IRR = 0.50)",
            "clinical_takeaway": "Remote telemonitoring cut emergency COPD admissions from 30.0 per 100 person-years down to 15.0 per 100 person-years (IRR = 0.50, 95% CI: [0.36, 0.69]). This achieves an absolute rate reduction of 15.0 avoided admissions per 100 patient-years of monitoring.",
            "disclaimer": "Made-up example scenario with synthetic data generated for educational illustration only."
        }
    }

    if scenario not in scenarios:
        scenario = "derbyshire_lead_foundry"

    cfg = scenarios[scenario]
    a = int(cfg["a"])
    pt_e = float(cfg["pt_e"])
    c = int(cfg["c"])
    pt_u = float(cfg["pt_u"])
    multiplier = float(cfg["rate_multiplier"])
    mult_label = cfg["rate_multiplier_label"]

    exp_label, unexp_label = cfg["exposure_levels"]
    out_pos_label, out_neg_label = cfg["outcome_levels"]

    # 1. Summary Person-Time Contingency Table
    df_contingency = pd.DataFrame([
        {
            "Cohort Stratum": f"🟢 {exp_label}",
            "Incident Events (a / c)": a,
            f"Person-Time ({cfg['time_unit']})": f"{pt_e:,.1f}",
            f"Incidence Rate ({mult_label})": float(a / pt_e * multiplier) if pt_e > 0 else 0.0
        },
        {
            "Cohort Stratum": f"⚪ {unexp_label}",
            "Incident Events (a / c)": c,
            f"Person-Time ({cfg['time_unit']})": f"{pt_u:,.1f}",
            f"Incidence Rate ({mult_label})": float(c / pt_u * multiplier) if pt_u > 0 else 0.0
        },
        {
            "Cohort Stratum": "📊 Total Cohort",
            "Incident Events (a / c)": a + c,
            f"Person-Time ({cfg['time_unit']})": f"{pt_e + pt_u:,.1f}",
            f"Incidence Rate ({mult_label})": float((a + c) / (pt_e + pt_u) * multiplier) if (pt_e + pt_u) > 0 else 0.0
        }
    ])

    # 2. Individual Participant Microdata Simulation (staggered dynamic cohort entries)
    participants = []
    # Exposed group
    n_exp_approx = max(a * 4, 150)
    pt_per_person_e = pt_e / n_exp_approx
    for i in range(n_exp_approx):
        had_event = 1 if i < a else 0
        py_val = np.round(np.random.exponential(pt_per_person_e) + 0.1, 2)
        participants.append({
            "Participant_ID": f"EXP_{i+1:04d}",
            "Cohort_Group": exp_label,
            "Exposure_Binary": 1,
            "Observation_Person_Time": py_val,
            "Event_Occurred": had_event,
            "Event_Status": out_pos_label if had_event else out_neg_label,
            "Age_At_Entry": int(np.random.normal(48, 8)),
            "Sex": "Male" if np.random.rand() > 0.3 else "Female"
        })

    # Unexposed group
    n_unexp_approx = max(c * 4, 200)
    pt_per_person_u = pt_u / n_unexp_approx
    for i in range(n_unexp_approx):
        had_event = 1 if i < c else 0
        py_val = np.round(np.random.exponential(pt_per_person_u) + 0.1, 2)
        participants.append({
            "Participant_ID": f"UNEXP_{i+1:04d}",
            "Cohort_Group": unexp_label,
            "Exposure_Binary": 0,
            "Observation_Person_Time": py_val,
            "Event_Occurred": had_event,
            "Event_Status": out_pos_label if had_event else out_neg_label,
            "Age_At_Entry": int(np.random.normal(49, 8)),
            "Sex": "Male" if np.random.rand() > 0.35 else "Female"
        })

    df_microdata = pd.DataFrame(participants)

    calcs = calculate_incidence_rate_ratio(a, pt_e, c, pt_u, rate_multiplier=multiplier)
    meta = {
        **cfg,
        "scenario_key": scenario,
        "a": a,
        "pt_e": pt_e,
        "c": c,
        "pt_u": pt_u,
        "calcs": calcs
    }

    return df_contingency, df_microdata, meta


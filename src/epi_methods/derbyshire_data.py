"""Derbyshire & Derby City Demographic & Epidemiological Data Architecture.

Provides:
1. Authentic population age profiles for the 9 Local Authorities in Derbyshire
   (Derby City Unitary Authority + 8 District Councils).
2. European Standard Population (ESP 2013) demographic reference weights.
3. Epidemiological simulation engine illustrating the Reversal Paradox
   (Simpson's Paradox in spatial epidemiology) across contrasting age pyramids.
4. GeoJSON boundary integration for interactive Folium choropleth mapping.
"""

import os
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# 1. European Standard Population (ESP 2013) Definition
# -----------------------------------------------------------------------------
# Aggregated into 6 standard analytical age bands from the 2013 ESP (total = 100,000)
ESP_2013_BANDS = [
    {"age_band": "0-14", "label": "0 to 14 years", "esp_weight": 0.150, "esp_pop": 15000, "order": 1},
    {"age_band": "15-44", "label": "15 to 44 years", "esp_weight": 0.380, "esp_pop": 38000, "order": 2},
    {"age_band": "45-64", "label": "45 to 64 years", "esp_weight": 0.260, "esp_pop": 26000, "order": 3},
    {"age_band": "65-74", "label": "65 to 74 years", "esp_weight": 0.110, "esp_pop": 11000, "order": 4},
    {"age_band": "75-84", "label": "75 to 84 years", "esp_weight": 0.070, "esp_pop": 7000, "order": 5},
    {"age_band": "85+", "label": "85 years and over", "esp_weight": 0.030, "esp_pop": 3000, "order": 6},
]


# -----------------------------------------------------------------------------
# 2. Local Authority Baseline Demographic Profiles
# -----------------------------------------------------------------------------
# Population sizes and age proportions calibrated against ONS Mid-Year Census Estimates.
# Captures the contrast between:
# - Derbyshire Dales / NE Derbyshire (high elderly %, rural retirement)
# - Derby City (younger urban, student & working-age demographic)
# - South Derbyshire (rapid growth family commuter belt)
DISTRICT_PROFILES = [
    {
        "name": "Derby",
        "ons_code": "E06000015",
        "authority_type": "Unitary Authority",
        "total_pop": 263500,
        "age_shares": {
            "0-14": 0.195,
            "15-44": 0.412,
            "45-64": 0.235,
            "65-74": 0.082,
            "75-84": 0.053,
            "85+": 0.023,
        },
        "character": "Urban unitary authority with younger demographic, higher university & working-age share, urban deprivation pockets."
    },
    {
        "name": "Derbyshire Dales",
        "ons_code": "E07000035",
        "authority_type": "District Council",
        "total_pop": 71800,
        "age_shares": {
            "0-14": 0.122,
            "15-44": 0.264,
            "45-64": 0.318,
            "65-74": 0.156,
            "75-84": 0.098,
            "85+": 0.042,
        },
        "character": "Rural Peak District market towns (Matlock, Bakewell, Ashbourne); highest proportion of retirees (29.6% aged 65+)."
    },
    {
        "name": "South Derbyshire",
        "ons_code": "E07000039",
        "authority_type": "District Council",
        "total_pop": 111400,
        "age_shares": {
            "0-14": 0.190,
            "15-44": 0.385,
            "45-64": 0.265,
            "65-74": 0.095,
            "75-84": 0.047,
            "85+": 0.018,
        },
        "character": "Fast-growing Trent Valley corridor (Swadlincote, Melbourne, Repton); substantial young family and child demographic."
    },
    {
        "name": "High Peak",
        "ons_code": "E07000037",
        "authority_type": "District Council",
        "total_pop": 92800,
        "age_shares": {
            "0-14": 0.148,
            "15-44": 0.332,
            "45-64": 0.298,
            "65-74": 0.124,
            "75-84": 0.071,
            "85+": 0.027,
        },
        "character": "North-western upland towns (Buxton, Glossop); mixed rural-commuter profile with aging demographic."
    },
    {
        "name": "Amber Valley",
        "ons_code": "E07000032",
        "authority_type": "District Council",
        "total_pop": 127200,
        "age_shares": {
            "0-14": 0.152,
            "15-44": 0.338,
            "45-64": 0.286,
            "65-74": 0.126,
            "75-84": 0.072,
            "85+": 0.026,
        },
        "character": "Central Derwent corridor (Belper, Ripley, Alfreton, Heanor); balanced industrial-rural profile."
    },
    {
        "name": "Chesterfield",
        "ons_code": "E07000034",
        "authority_type": "District Council",
        "total_pop": 104200,
        "age_shares": {
            "0-14": 0.155,
            "15-44": 0.350,
            "45-64": 0.275,
            "65-74": 0.118,
            "75-84": 0.074,
            "85+": 0.028,
        },
        "character": "Historic market town and former coalfield borough; higher baseline chronic disease burden."
    },
    {
        "name": "North East Derbyshire",
        "ons_code": "E07000038",
        "authority_type": "District Council",
        "total_pop": 102400,
        "age_shares": {
            "0-14": 0.140,
            "15-44": 0.315,
            "45-64": 0.295,
            "65-74": 0.138,
            "75-84": 0.082,
            "85+": 0.030,
        },
        "character": "Rural and commuter belt encircling Chesterfield (Dronfield, Clay Cross); older resident profile (25.0% aged 65+)."
    },
    {
        "name": "Bolsover",
        "ons_code": "E07000033",
        "authority_type": "District Council",
        "total_pop": 81500,
        "age_shares": {
            "0-14": 0.160,
            "15-44": 0.362,
            "45-64": 0.272,
            "65-74": 0.116,
            "75-84": 0.066,
            "85+": 0.024,
        },
        "character": "Eastern former coalfield district (Bolsover, Shirebrook); industrial heritage with socioeconomic health challenges."
    },
    {
        "name": "Erewash",
        "ons_code": "E07000036",
        "authority_type": "District Council",
        "total_pop": 113000,
        "age_shares": {
            "0-14": 0.162,
            "15-44": 0.365,
            "45-64": 0.268,
            "65-74": 0.112,
            "75-84": 0.067,
            "85+": 0.026,
        },
        "character": "South-eastern valley bordering Nottinghamshire (Ilkeston, Long Eaton); balanced urban-commuter demographic."
    }
]


# -----------------------------------------------------------------------------
# 3. Health Scenarios & Baseline Age-Specific Mortality Rates (per 1,000)
# -----------------------------------------------------------------------------
# Calibrated so that across equivalent age cohorts, urban/deprived areas have higher
# age-specific mortality, but rural aging areas have higher crude rates due to demographics.
SCENARIO_CONFIGS = {
    "circulatory": {
        "title": "Circulatory & Cardiovascular Mortality (CVD)",
        "unit": "per 1,000 population",
        "esp_reference_rates": {
            "0-14": 0.04,
            "15-44": 0.35,
            "45-64": 2.80,
            "65-74": 11.50,
            "75-84": 35.00,
            "85+": 98.00,
        },
        # Relative risk multiplier per district on age-specific rates:
        # Reflects underlying health status: Derby & Bolsover higher; Dales lower.
        "district_rate_multipliers": {
            "Derby": 1.18,
            "Bolsover": 1.15,
            "Chesterfield": 1.12,
            "Erewash": 1.05,
            "Amber Valley": 0.98,
            "High Peak": 0.94,
            "North East Derbyshire": 0.92,
            "South Derbyshire": 0.91,
            "Derbyshire Dales": 0.86,
        },
        "description": "Heart disease and stroke mortality, demonstrating the classic Reversal Paradox."
    },
    "cancer": {
        "title": "All-Malignant Neoplasms (Cancer) Mortality",
        "unit": "per 1,000 population",
        "esp_reference_rates": {
            "0-14": 0.05,
            "15-44": 0.28,
            "45-64": 3.20,
            "65-74": 12.80,
            "75-84": 26.50,
            "85+": 48.00,
        },
        "district_rate_multipliers": {
            "Derby": 1.16,
            "Bolsover": 1.14,
            "Chesterfield": 1.10,
            "Erewash": 1.04,
            "Amber Valley": 0.99,
            "High Peak": 0.95,
            "North East Derbyshire": 0.93,
            "South Derbyshire": 0.92,
            "Derbyshire Dales": 0.87,
        },
        "description": "Oncological mortality exhibiting steep age dependency and demographic confounding."
    },
    "all_cause": {
        "title": "All-Cause Mortality",
        "unit": "per 1,000 population",
        "esp_reference_rates": {
            "0-14": 0.80,
            "15-44": 1.40,
            "45-64": 7.50,
            "65-74": 24.00,
            "75-84": 68.00,
            "85+": 165.00,
        },
        "district_rate_multipliers": {
            "Derby": 1.15,
            "Bolsover": 1.13,
            "Chesterfield": 1.10,
            "Erewash": 1.05,
            "Amber Valley": 0.99,
            "High Peak": 0.95,
            "North East Derbyshire": 0.93,
            "South Derbyshire": 0.92,
            "Derbyshire Dales": 0.88,
        },
        "description": "Comprehensive all-cause mortality across all local authorities."
    }
}


def get_derbyshire_geojson_path() -> str:
    """Returns absolute path to local GeoJSON boundaries."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base_dir, "data", "boundaries", "derbyshire_districts.geojson")
    return path


def get_esp_2013_standard() -> pd.DataFrame:
    """Returns the European Standard Population (ESP 2013) weight table."""
    return pd.DataFrame(ESP_2013_BANDS)


def generate_derbyshire_standardization_data(
    scenario: str = "circulatory",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates synthetic epidemiological data for Derby City and the 8 Derbyshire districts.

    Args:
        scenario: 'circulatory', 'cancer', or 'all_cause'
        seed: Random seed for reproducible stochastic variation

    Returns:
        Tuple of:
        - df_detailed: 54 rows (9 districts x 6 age bands) with populations, deaths, rates, and weights.
        - df_summary: 9 rows (one per district) with crude vs standardized rates, 95% CIs, and rankings.
        - metadata: Contextual parameters, scenario notes, and Reversal Paradox highlights.
    """
    if scenario not in SCENARIO_CONFIGS:
        scenario = "circulatory"

    cfg = SCENARIO_CONFIGS[scenario]
    ref_rates = cfg["esp_reference_rates"]
    multipliers = cfg["district_rate_multipliers"]
    rng = np.random.default_rng(seed)

    detailed_rows: List[Dict[str, Any]] = []

    for dist in DISTRICT_PROFILES:
        d_name = dist["name"]
        d_code = dist["ons_code"]
        d_type = dist["authority_type"]
        d_tot_pop = dist["total_pop"]
        d_mult = multipliers.get(d_name, 1.0)

        for band in ESP_2013_BANDS:
            age_key = band["age_band"]
            esp_wt = band["esp_weight"]
            order = band["order"]

            # Calculate local population for age band
            pop_share = dist["age_shares"][age_key]
            n_age = int(round(d_tot_pop * pop_share))

            # Underlying true age-specific rate per 1,000
            base_r = ref_rates[age_key]
            r_age = base_r * d_mult

            # Expected deaths and simulated stochastic Poisson realization
            expected_lambda = (n_age * r_age) / 1000.0
            deaths = int(rng.poisson(lam=max(0.01, expected_lambda)))

            # Empirical observed rate per 1,000
            rate_per_1k = (deaths / n_age) * 1000.0 if n_age > 0 else 0.0

            # Direct standardization contribution: weight * rate
            dsr_component = esp_wt * rate_per_1k

            # Indirect standardization expected deaths using standard reference rate
            exp_indirect_deaths = (n_age * base_r) / 1000.0

            detailed_rows.append({
                "District": d_name,
                "ONS_Code": d_code,
                "Authority_Type": d_type,
                "Age_Band": age_key,
                "Age_Band_Label": band["label"],
                "Age_Order": order,
                "Population": n_age,
                "Observed_Deaths": deaths,
                "Age_Specific_Rate_per_1k": round(rate_per_1k, 2),
                "ESP_Weight": esp_wt,
                "DSR_Component_per_1k": round(dsr_component, 3),
                "Reference_Rate_per_1k": base_r,
                "Expected_Deaths_Indirect": round(exp_indirect_deaths, 2),
            })

    df_detailed = pd.DataFrame(detailed_rows)

    # -------------------------------------------------------------------------
    # Aggregate to Summary Table (9 Local Authorities)
    # -------------------------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []

    for dist in DISTRICT_PROFILES:
        d_name = dist["name"]
        d_sub = df_detailed[df_detailed["District"] == d_name]

        tot_pop = int(d_sub["Population"].sum())
        tot_deaths = int(d_sub["Observed_Deaths"].sum())

        # Population aged 65+
        pop_65_plus = int(d_sub[d_sub["Age_Band"].isin(["65-74", "75-84", "85+"])]["Population"].sum())
        pct_65_plus = (pop_65_plus / tot_pop) * 100.0 if tot_pop > 0 else 0.0

        # 1. Crude Death Rate per 1,000
        crude_rate = (tot_deaths / tot_pop) * 1000.0 if tot_pop > 0 else 0.0

        # 2. Direct Age-Standardized Rate (DSR) per 1,000
        dsr_val = float(d_sub["DSR_Component_per_1k"].sum())

        # Exact Dobson / Normal Variance for DSR:
        # Var(DSR) = sum( (w_i^2 * d_i) / n_i^2 ) * 1,000^2
        var_components = []
        for _, row in d_sub.iterrows():
            w_i = row["ESP_Weight"]
            d_i = row["Observed_Deaths"]
            n_i = row["Population"]
            if n_i > 0:
                var_i = (w_i ** 2) * (d_i / (n_i ** 2)) * (1000.0 ** 2)
                var_components.append(var_i)
            else:
                var_components.append(0.0)

        dsr_var = sum(var_components)
        dsr_se = float(np.sqrt(dsr_var))
        dsr_lower = max(0.0, dsr_val - 1.96 * dsr_se)
        dsr_upper = dsr_val + 1.96 * dsr_se

        # 3. Indirect Standardization: Standardized Mortality Ratio (SMR)
        exp_indirect_tot = float(d_sub["Expected_Deaths_Indirect"].sum())
        smr_ratio = (tot_deaths / exp_indirect_tot) if exp_indirect_tot > 0 else 1.0

        # Byar's approximation for Poisson 95% CI on SMR
        if tot_deaths > 0:
            smr_lower = (tot_deaths / exp_indirect_tot) * (1.0 - (1.0 / (9.0 * tot_deaths)) - (1.96 / (3.0 * np.sqrt(tot_deaths)))) ** 3
            smr_upper = ((tot_deaths + 1.0) / exp_indirect_tot) * (1.0 - (1.0 / (9.0 * (tot_deaths + 1.0))) + (1.96 / (3.0 * np.sqrt(tot_deaths + 1.0)))) ** 3
        else:
            smr_lower = 0.0
            smr_upper = 3.0 / exp_indirect_tot if exp_indirect_tot > 0 else 1.0

        summary_rows.append({
            "District": d_name,
            "ONS_Code": dist["ons_code"],
            "Authority_Type": dist["authority_type"],
            "Total_Population": tot_pop,
            "Pop_65_Plus": pop_65_plus,
            "Pct_65_Plus": round(pct_65_plus, 1),
            "Total_Deaths": tot_deaths,
            "Crude_Rate_per_1k": round(crude_rate, 2),
            "DSR_per_1k": round(dsr_val, 2),
            "DSR_95_Lower": round(dsr_lower, 2),
            "DSR_95_Upper": round(dsr_upper, 2),
            "DSR_SE": round(dsr_se, 3),
            "Expected_Deaths_Indirect": round(exp_indirect_tot, 1),
            "SMR": round(smr_ratio, 3),
            "SMR_Percentage": round(smr_ratio * 100.0, 1),
            "SMR_95_Lower": round(smr_lower, 3),
            "SMR_95_Upper": round(smr_upper, 3),
            "Character_Profile": dist["character"],
        })

    df_summary = pd.DataFrame(summary_rows)

    # Compute comparative rankings (1 = Lowest / Best health outcome, 9 = Highest mortality)
    df_summary["Crude_Rank"] = df_summary["Crude_Rate_per_1k"].rank(ascending=True).astype(int)
    df_summary["DSR_Rank"] = df_summary["DSR_per_1k"].rank(ascending=True).astype(int)
    df_summary["Rank_Shift"] = df_summary["Crude_Rank"] - df_summary["DSR_Rank"]

    metadata = {
        "scenario": scenario,
        "title": cfg["title"],
        "unit": cfg["unit"],
        "description": cfg["description"],
        "reversal_highlight": (
            "Notice how Derbyshire Dales exhibits a significantly elevated Crude Death Rate due to its 29.6% elderly "
            "population share, but drops substantially in rank under Direct Age-Standardization (DSR). Conversely, Derby City "
            "features a lower crude rate due to its youthful demographic, but reveals a higher standardized rate once demographic "
            "structure is controlled."
        ),
        "disclaimer": (
            "⚠️ Pedagogical Notice: Demographic populations reflect authentic local authority census distributions; "
            "disease event and mortality counts are synthetically generated for educational demonstration of the Reversal Paradox."
        )
    }

    return df_detailed, df_summary, metadata


def get_excel_aligned_dataset(
    df_detailed: pd.DataFrame,
    focal_district: str = "Derbyshire Dales"
) -> pd.DataFrame:
    """Extracts a 6-row dataset for a specific district matching the exact Excel guide columns.

    Matches the exact column and row coordinates described in the Microsoft Excel Implementation Guide:
    - Row 2: 0-14 years
    - Row 3: 15-44 years
    - Row 4: 45-64 years
    - Row 5: 65-74 years
    - Row 6: 75-84 years
    - Row 7: 85+ years

    Worksheet Column Layout:
    - Col_A_Age_Band: Column A in Excel (A2:A7) - Stratum age bracket
    - Col_B_Population: Column B in Excel (B2:B7) - Local stratum population count n_i
    - Col_C_Observed_Deaths: Column C in Excel (C2:C7) - Local observed deaths d_i
    - Col_D_Rate_per_1k: Column D in Excel (D2:D7) - Local age-specific rate per 1,000 (=C2/B2*1000)
    - Col_E_ESP_Weight: Column E in Excel (E2:E7) - European Standard Population weights w_i
    - Col_F_Reference_Rate_per_1k: Column F in Excel (F2:F7) - Standard reference benchmark rate R_std,i

    Args:
        df_detailed (pd.DataFrame): 54-row microdata generated by generate_derbyshire_standardization_data.
        focal_district (str, optional): Authority to extract. Defaults to "Derbyshire Dales".

    Returns:
        pd.DataFrame: 6-row DataFrame aligned with Excel cell coordinates.
    """
    sub = df_detailed[df_detailed["District"] == focal_district].copy()
    if sub.empty:
        sub = df_detailed[df_detailed["District"] == "Derbyshire Dales"].copy()

    sub = sub.sort_values(by="Age_Order")

    return pd.DataFrame({
        "Col_A_Age_Band": sub["Age_Band"].values,
        "Col_B_Population": sub["Population"].values,
        "Col_C_Observed_Deaths": sub["Observed_Deaths"].values,
        "Col_D_Rate_per_1k": sub["Age_Specific_Rate_per_1k"].values,
        "Col_E_ESP_Weight": sub["ESP_Weight"].values,
        "Col_F_Reference_Rate_per_1k": sub["Reference_Rate_per_1k"].values,
    })


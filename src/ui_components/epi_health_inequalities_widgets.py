"""Streamlit UI Components & Visualizations for Health Inequalities: Slope Index of Inequality (SII)
and Relative Index of Inequality (RII).

Provides:
1. render_health_inequalities_plain_english_guide: Zero-prior-knowledge walkthrough of social gradients,
   ridits (relative ranks), and the absolute vs. relative inequality paradox (Marmot Review / Core20PLUS5).
2. render_health_inequalities_diagnostics: Tri-color evaluation cards (PASS, WARNING, FAIL) for strata count,
   population denominators, and gradient monotonicity.
3. render_health_inequalities_dashboard: Metric cards (SII point estimate & 95% CI, RII point estimate & 95% CI,
   baseline affluence rate, most deprived rate, weighted R^2).
4. render_health_inequalities_visualizations: Plotly interactive population-weighted bubble chart with fitted
   WLS regression slope and 95% confidence ribbon, plus decile bar breakdown.
5. render_health_inequalities_excel_guide: Strict 3-step sequence with explicit cell mapping and native Excel formulas.
6. render_health_inequalities_python_guide: Strict 4-step sequence using statsmodels WLS with delta-method CIs.
7. render_health_inequalities_download_tab: Live CSV export of strata microdata.
8. render_health_inequalities_module: Top-level controller function.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.epi_methods.health_inequalities import (
    calculate_relative_ranks,
    calculate_sii_rii,
    evaluate_health_inequalities_prerequisites,
    generate_inequality_demo_data
)


def render_health_inequalities_plain_english_guide() -> None:
    """Renders the zero-prior-knowledge pedagogical guide for SII and RII."""
    with st.expander("💡 Plain English Guide: Measuring Health Inequalities (SII & RII)", expanded=False):
        st.markdown("""
        In public health, social epidemiology, and health equity (e.g. **OHID Fingertips**, **Marmot Reviews**, 
        and the **NHS Core20PLUS5** agenda), measuring health differences across society requires more than comparing 
        the richest 10% to the poorest 10%. We need to capture the **entire social gradient**.
        """)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(r"""
            #### 1. Why Not Just Compare Top vs. Bottom?
            * **The Range Flaw:** If you only compare the most deprived quintile (Q1) to the least deprived (Q5), you ignore the middle 60% of the population!
            * **Unequal Group Sizes:** Social groups are rarely of equal size. If the most deprived category expands or shrinks over time, simple ratios give a distorted picture.
            * **The Solution — Cumulative Relative Ranks (Ridits):**
              We line up the entire population from the **most deprived (rank 0.0)** to the **least deprived (rank 1.0)**.
              Each socioeconomic group (e.g., IMD decile) is assigned a midpoint rank on this line proportional to its cumulative population share.
            """)

        with col_g2:
            st.markdown(r"""
            #### 2. The Two Complementary Inequality Measures
            * **🟢 Slope Index of Inequality (SII — Absolute Gap):**
              - **The Question:** *"Across the entire society, how many excess deaths or cases occur at the most deprived extreme compared to the least deprived?"*
              - **Interpretation:** If $\text{SII} = 95.3$ per 100,000, it means the most disadvantaged hypothetical individual suffers **95.3 additional deaths per 100,000** compared to the most advantaged individual.
            * **🔵 Relative Index of Inequality (RII — Relative Ratio):**
              - **The Question:** *"What is the proportional ratio of disease risk between the bottom and the top of the social ladder?"*
              - **Interpretation:** If $\text{RII} = 3.16$, the rate of premature death at the most deprived extreme is **3.16 times (over 300%)** that of the least deprived extreme.
            """)

        st.markdown(r"""
        > [!IMPORTANT]
        > **The Absolute vs. Relative Inequality Paradox:**  
        > When a major public health intervention (e.g., nationwide smoking cessation or statin prescribing) successfully lowers overall cardiovascular mortality by 50% across all deprivation groups, the **absolute gap (SII) shrinks by half**, but the **relative ratio (RII) remains unchanged**! 
        > Commissioners must monitor **both** metrics to understand whether inequalities are genuinely narrowing in absolute and relative terms.
        """)


def render_health_inequalities_diagnostics(diagnostics: Dict[str, Any]) -> None:
    """Renders tri-color diagnostic cards for methodological prerequisites."""
    st.markdown("#### 🛡️ Epidemiological Prerequisites & Diagnostic Audit")

    cols = st.columns(len(diagnostics["checks"]))
    for i, check in enumerate(diagnostics["checks"]):
        with cols[i]:
            status = check["status"]
            if status == "PASS":
                st.success(f"**{check['test']}**\n\n🟢 {check['message']}")
            elif status == "WARNING":
                st.warning(f"**{check['test']}**\n\n🟡 {check['message']}")
            else:
                st.error(f"**{check['test']}**\n\n🔴 {check['message']}")


def render_health_inequalities_dashboard(res: Dict[str, Any]) -> None:
    """Renders top metric summary cards for SII, RII, baseline rates, and fit."""
    st.markdown("#### 📊 Inequality Indices & Gradient Summary")
    st.caption(
        f"**Active Model:** {res.get('model_name', 'Linear WLS')} &nbsp;|&nbsp; "
        f"**Gradient Direction:** {'Favourable / Protective Outcome (Higher is Better)' if res.get('reverse_direction') else 'Adverse Outcome / Risk (Lower is Better)'}"
    )

    is_sig = res.get("is_sii_significant", False)
    p_str = res.get("p_value_sii_str", "< 0.05" if is_sig else "≥ 0.05")

    # Comprehensive pedagogical tooltips explaining metric definition & interpretation
    help_sii = (
        "📌 What is the Slope Index of Inequality (SII)?\n\n"
        "• Definition: Measures the absolute difference in the health indicator across the entire continuous "
        "socioeconomic spectrum, from the hypothetical most deprived individual (rank 0.0) to the hypothetical "
        "least deprived individual (rank 1.0), derived from a population-weighted regression over all ordered strata.\n\n"
        "• How to interpret it:\n"
        "  - Adverse outcomes (e.g. CVD mortality, obesity): A positive SII of +X means the most disadvantaged "
        "extreme suffers X excess cases/deaths compared to the most advantaged extreme. If socioeconomic inequality "
        "were eliminated, X fewer events per unit population would occur.\n"
        "  - Favourable outcomes (e.g. Life Expectancy): A positive SII of +X years indicates the longevity advantage "
        "enjoyed by the most affluent extreme compared to the most deprived (e.g. +11.9 additional years of life).\n\n"
        "• Statistical Significance: If the 95% Confidence Interval excludes 0.0, the absolute gradient is "
        "statistically significant (p < 0.05)."
    )

    help_rii = (
        "📌 What is the Relative Index of Inequality (RII)?\n\n"
        "• Definition: Quantifies proportional inequality across the full socioeconomic hierarchy, calculated as the ratio "
        "of estimated health rates between the two hypothetical extremes of the population (Rank 0.0 vs Rank 1.0).\n\n"
        "• How to interpret it:\n"
        "  - RII = 1.00: Perfect socioeconomic equality (uniform health rates across society).\n"
        "  - RII > 1.00: For adverse outcomes, an RII of 3.16x means the disease rate at the most disadvantaged extreme is "
        "3.16 times (over 200% higher than) that of the least disadvantaged extreme.\n"
        "  - Unlike a crude decile 10 vs decile 1 ratio, RII accounts for population weights across all intermediate strata.\n\n"
        "• Statistical Significance: If the 95% Confidence Interval excludes 1.00, relative inequality is statistically "
        "significant (p < 0.05)."
    )

    help_y_most = (
        "📌 What is the Most Deprived (Rank 0.0) Rate?\n\n"
        "• Definition: The regression-modelled health indicator rate (ŷ) for a hypothetical individual at the absolute "
        "bottom of the socioeconomic hierarchy (cumulative relative rank 0.0), computed from the model intercept.\n\n"
        "• How to interpret it: Represents baseline health under maximum deprivation conditions. Comparing this to the "
        "observed rate in Decile 1 reveals how closely the continuous regression line captures the most disadvantaged group."
    )

    help_y_least = (
        "📌 What is the Least Deprived (Rank 1.0) Rate?\n\n"
        "• Definition: The regression-modelled health indicator rate (ŷ) for a hypothetical individual at the absolute "
        "top of the socioeconomic hierarchy (cumulative relative rank 1.0), computed as α + β (or exp(γ₀ + γ₁) in log-linear GLM).\n\n"
        "• How to interpret it: Represents baseline health under maximum affluence. In public health planning, this serves as "
        "the aspirational benchmark or unavoidable baseline risk if socioeconomic deprivation barriers were removed."
    )

    help_r2 = (
        "📌 What is the Gradient Fit (Weighted R²)?\n\n"
        "• Definition: The population-weighted coefficient of determination (R²), quantifying the proportion of total "
        "weighted variance in the health indicator across strata explained by the continuous socioeconomic rank continuum.\n\n"
        "• How to interpret it:\n"
        "  - R² ≥ 80%: Strong, systematic social gradient where deprivation is the primary driver of variation between strata.\n"
        "  - R² < 50%: Moderate or weak gradient; substantial non-linear fluctuations or unmeasured non-deprivation drivers exist."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="Slope Index (SII - Absolute)",
            value=f"{res['sii']:+.1f}" if res.get("reverse_direction") else f"{res['sii']:.1f}",
            delta=f"{'🟢 Significant' if is_sig else '⚪ Not Sig.'} (p {p_str})",
            delta_color="normal",
            help=help_sii
        )
    with col2:
        st.metric(
            label="Relative Index (RII - Ratio)",
            value=f"{res['rii']:.2f}x",
            delta=f"95% CI: [{res['rii_ci_lower']:.2f}, {res['rii_ci_upper']:.2f}]",
            delta_color="inverse" if not res.get("reverse_direction") and res["rii"] > 1.0 else "normal",
            help=help_rii
        )
    with col3:
        st.metric(
            label="Most Deprived (Rank 0.0)",
            value=f"{res['y_most_deprived']:.1f}",
            help=help_y_most
        )
    with col4:
        st.metric(
            label="Least Deprived (Rank 1.0)",
            value=f"{res['y_least_deprived']:.1f}",
            help=help_y_least
        )
    with col5:
        st.metric(
            label="Gradient Fit (Weighted R²)",
            value=f"{res['r_squared']:.1%}",
            delta=f"K={res['k_strata']} | {res.get('model_type', 'linear').title()}",
            help=help_r2
        )

    # Clear label and banner explicitly flagging statistical significance
    if is_sig:
        st.success(
            f"✅ **Statistical Significance Flag:** The Slope Index of Inequality (SII = {res['sii']:+.1f}) is "
            f"**STATISTICALLY SIGNIFICANT** ({res.get('sii_significance_label', 'p < 0.05')}, p {p_str} | 95% CI: [{res['sii_ci_lower']:.1f}, {res['sii_ci_upper']:.1f}]). "
            f"Because the 95% confidence interval does not cross zero, there is strong evidence of a genuine socioeconomic gradient across the population."
        )
    else:
        st.warning(
            f"⚠️ **Statistical Significance Flag:** The Slope Index of Inequality (SII = {res['sii']:+.1f}) is "
            f"**NOT STATISTICALLY SIGNIFICANT** ({res.get('sii_significance_label', 'p ≥ 0.05')}, p {p_str} | 95% CI: [{res['sii_ci_lower']:.1f}, {res['sii_ci_upper']:.1f}]). "
            f"Because the 95% confidence interval crosses zero, the observed rate difference across strata could plausibly be due to random variation."
        )

    if res.get("model_type") == "log_linear":
        st.info(
            f"📐 **Log-Linear Model Active:** Natural log link GLM $\\ln(y) = {res['alpha_intercept']:.3f} + ({res['beta_slope']:.3f}) \\cdot x$. "
            f"Guarantees non-negative fitted rates ($\\hat{{y}} > 0$) across the complete socioeconomic continuum."
        )

    st.info(f"🧭 **Public Health Assessment:** {res['gradient_summary']}")


def render_health_inequalities_visualizations(res: Dict[str, Any]) -> None:
    """Renders Plotly interactive regression gradient curve and decile bar chart."""
    st.markdown("#### 📈 Interactive Diagnostic Visualizer: Social Gradient Regression")

    x_vals = res["x_ranks"]
    y_obs = res["observed_rates"]
    y_fit = res["fitted_rates"]
    pop = res["populations"]
    shares = res["population_shares"]
    names = res["strata_names"]
    ind_name = res["indicator_name"]
    is_log = res.get("model_type") == "log_linear"

    # 1. Regression Plot with Population Bubble Sizes
    fig_reg = go.Figure()

    # Dense regression curve across rank continuum [0, 1]
    dense_x = np.linspace(0.0, 1.0, 100)
    var_a = res.get("var_alpha", 0.0)
    var_b = res.get("var_beta", 0.0)
    cov_ab = res.get("cov_alpha_beta", 0.0)
    alpha = res["alpha_intercept"]
    beta = res["beta_slope"]

    if is_log:
        dense_y = np.exp(alpha + beta * dense_x)
        line_label = "Log-Linear GLM Curve"
        # Delta method for log scale ribbon
        se_dense_log = np.sqrt(np.maximum(0.0, var_a + (dense_x ** 2) * var_b + 2.0 * dense_x * cov_ab))
        upper_band = np.exp(alpha + beta * dense_x + 1.96 * se_dense_log)
        lower_band = np.exp(alpha + beta * dense_x - 1.96 * se_dense_log)
    else:
        dense_y = alpha + beta * dense_x
        line_label = "Linear WLS Slope"
        se_dense = np.sqrt(np.maximum(0.0, var_a + (dense_x ** 2) * var_b + 2.0 * dense_x * cov_ab))
        upper_band = dense_y + 1.96 * se_dense
        lower_band = dense_y - 1.96 * se_dense

    # Fitted model line
    fig_reg.add_trace(go.Scatter(
        x=dense_x,
        y=dense_y,
        mode="lines",
        name=line_label,
        line=dict(color="#2563eb", width=3, dash="solid")
    ))

    # Confidence ribbon
    fig_reg.add_trace(go.Scatter(
        x=np.concatenate([dense_x, dense_x[::-1]]),
        y=np.concatenate([upper_band, lower_band[::-1]]),
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Band"
    ))

    # Observed points scaled by population
    max_pop = max(pop)
    norm_sizes = [max(12, min(36, 12 + 24 * (p / max_pop))) for p in pop]

    hover_texts = [
        f"<b>{names[i]}</b><br>"
        f"Population: {pop[i]:,}<br>"
        f"Pop Share: {shares[i]:.1%}<br>"
        f"Cumulative Relative Rank (Ridit): {x_vals[i]:.3f}<br>"
        f"Observed Rate: {y_obs[i]:.1f}<br>"
        f"Fitted Model Rate: {y_fit[i]:.1f}<br>"
        f"Residual: {y_obs[i] - y_fit[i]:+.1f}"
        for i in range(len(names))
    ]

    fig_reg.add_trace(go.Scatter(
        x=x_vals,
        y=y_obs,
        mode="markers+text",
        name="Observed Strata",
        marker=dict(
            size=norm_sizes,
            color=x_vals,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=dict(text="Relative<br>Rank", font=dict(color="#0f172a", size=11)),
                tickfont=dict(color="#0f172a", size=10)
            ),
            line=dict(width=1.5, color="#0f172a")
        ),
        text=[f"S{i+1}" for i in range(len(names))],
        textposition="top center",
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    fig_reg.update_layout(
        title=dict(
            text=f"<b>Social Gradient Regression ({res.get('model_name', 'Linear WLS')})</b><br><span style='font-size: 13px; color: #475569;'>Slope Index (SII) = {res['sii']:.1f} &nbsp;|&nbsp; Relative Index (RII) = {res['rii']:.2f}x &nbsp;|&nbsp; Indicator: {ind_name}</span>",
            font=dict(size=16, color="#0f172a", family="system-ui, -apple-system, sans-serif"),
            x=0.0,
            y=0.97,
            xanchor="left",
            yanchor="top"
        ),
        xaxis=dict(
            title=dict(
                text="<b>Cumulative Relative Population Rank</b> (0.0 = Most Deprived Extreme, 1.0 = Least Deprived Extreme)",
                font=dict(size=13, color="#0f172a")
            ),
            tickfont=dict(size=11, color="#0f172a"),
            range=[-0.05, 1.05],
            gridcolor="#e2e8f0",
            zeroline=False
        ),
        yaxis=dict(
            title=dict(
                text=f"<b>{ind_name}</b>",
                font=dict(size=13, color="#0f172a")
            ),
            tickfont=dict(size=11, color="#0f172a"),
            gridcolor="#e2e8f0",
            zeroline=False
        ),
        font=dict(color="#0f172a"),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        height=500,
        margin=dict(t=90, b=70, l=75, r=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#0f172a"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#cbd5e1",
            borderwidth=1
        )
    )

    st.plotly_chart(fig_reg, use_container_width=True)

    # 2. Bar Chart of Rates Across Strata
    fig_bar = px.bar(
        x=names,
        y=y_obs,
        color=x_vals,
        color_continuous_scale="Viridis",
        labels={"x": "Socioeconomic Stratum", "y": ind_name, "color": "Relative Rank"}
    )
    fig_bar.update_layout(
        title=dict(
            text=f"<b>Observed Health Indicator Across Strata ({len(names)} Strata)</b><br><span style='font-size: 13px; color: #475569;'>Ordered from Most Deprived (Left) to Least Deprived (Right) &nbsp;|&nbsp; {ind_name}</span>",
            font=dict(size=16, color="#0f172a", family="system-ui, -apple-system, sans-serif"),
            x=0.0,
            y=0.97,
            xanchor="left",
            yanchor="top"
        ),
        xaxis=dict(
            title=dict(
                text="<b>Socioeconomic Strata</b> (Ordered: Most Deprived ➔ Least Deprived)",
                font=dict(size=13, color="#0f172a")
            ),
            tickfont=dict(size=11, color="#0f172a"),
            gridcolor="#e2e8f0"
        ),
        yaxis=dict(
            title=dict(
                text=f"<b>{ind_name}</b>",
                font=dict(size=13, color="#0f172a")
            ),
            tickfont=dict(size=11, color="#0f172a"),
            gridcolor="#e2e8f0"
        ),
        font=dict(color="#0f172a"),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        height=390,
        margin=dict(t=90, b=75, l=75, r=40),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def render_health_inequalities_excel_guide(res: Dict[str, Any]) -> None:
    """Renders the step-by-step Excel guide with native formulas."""
    st.markdown("### 📋 Microsoft Excel Step-by-Step Implementation Guide")
    st.markdown("""
    Calculating the Slope Index of Inequality (SII) and Relative Index of Inequality (RII) in Microsoft Excel 
    requires a structured 3-stage sequence: **Relative Ridit Ranks** ➔ **Weighted Regression Slope** ➔ **Extremes Ratio**.
    """)

    with st.expander("📌 Stage 1: Excel Cell Mapping & Ridit Ranks", expanded=False):
        st.markdown(r"""
        Set up your spreadsheet with the following columns in **Sheet1**:
        * **Column A (Strata Labels):** `A2:A11` (`Decile 1` down to `Decile 10`).
        * **Column B (Population Counts $n_j$):** `B2:B11` (Strata populations).
        * **Column C (Observed Health Rates $y_j$):** `C2:C11` (DSR or prevalence).
        * **Column D (Population Share $p_j$):** Formula in `D2`: `=B2 / SUM($B$2:$B$11)`. Drag down to `D11`.
        * **Column E (Cumulative Proportion $C_j$):** Formula in `E2`: `=D2`. In `E3`: `=E2 + D3`. Drag down to `E11`.
        * **Column F (Midpoint Relative Rank $x_j$):** Formula in `F2`: `=E2 - 0.5 * D2`. Drag down to `F11`.
        """)

    with st.expander("📌 Stage 2: Regression Slope (SII) & Intercept in Excel", expanded=False):
        st.markdown(r"""
        Calculate the regression parameters using native Excel statistical functions:
        * **Regression Slope ($\beta$):**
          ```excel
          =SLOPE(C2:C11, F2:F11)
          ```
        * **Regression Intercept ($\alpha$):**
          ```excel
          =INTERCEPT(C2:C11, F2:F11)
          ```
        * **Slope Index of Inequality (SII — Absolute Gap):**
          Under the convention where Decile 1 is most deprived (rank near 0) and Decile 10 is least deprived (rank near 1), the slope is negative. The absolute inequality gap is:
          ```excel
          =-SLOPE(C2:C11, F2:F11)
          ```
        * **Estimated Rate for Most Deprived ($y_0$):**
          ```excel
          =INTERCEPT(C2:C11, F2:F11)
          ```
        * **Estimated Rate for Least Deprived ($y_1$):**
          ```excel
          =INTERCEPT(C2:C11, F2:F11) + SLOPE(C2:C11, F2:F11)
          ```
        """)

    with st.expander("📌 Stage 3: Relative Index of Inequality (RII) & Full LINEST Output", expanded=False):
        st.markdown(r"""
        * **Relative Index of Inequality (RII):**
          ```excel
          =INTERCEPT(C2:C11, F2:F11) / (INTERCEPT(C2:C11, F2:F11) + SLOPE(C2:C11, F2:F11))
          ```
        * **Standard Errors & $R^2$ using Array Formula `=LINEST()`:**
          Highlight a 2-column by 3-row cell range (e.g. `H2:I4`) and enter:
          ```excel
          =LINEST(C2:C11, F2:F11, TRUE, TRUE)
          ```
          *(Press Ctrl+Shift+Enter in older Excel versions).*
          - Cell `H2` returns Slope $\beta$.
          - Cell `I2` returns Intercept $\alpha$.
          - Cell `H3` returns $\text{SE}(\beta)$, which is the exact **Standard Error of the SII**!
          - Cell `H4` returns $R^2$.
        * **95% Confidence Interval for SII:**
          - Lower Limit: `=(-H2) - 1.96 * H3`
          - Upper Limit: `=(-H2) + 1.96 * H3`
        """)


def render_health_inequalities_python_guide(res: Dict[str, Any]) -> None:
    """Renders the production-grade Python script template."""
    st.markdown("### 🐍 Production Python Implementation Pipeline")
    st.markdown("""
    This self-contained, reproducible pipeline uses `pandas`, `statsmodels.api.WLS`, and `scipy.stats`
    to compute both **Linear WLS** and **Log-Linear GLM** models with exact Delta Method confidence bounds.
    """)

    code_template = f"""import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ----------------------------------------------------------------------
# 1. Define Public Health Strata Data ({res['strata_names'][0]} to {res['strata_names'][-1]})
# ----------------------------------------------------------------------
data = pd.DataFrame({{
    'stratum': {res['strata_names']},
    'population': {res['populations']},
    'rate': {res['observed_rates']}
}})

# ----------------------------------------------------------------------
# 2. Compute Cumulative Relative Ranks (Midpoint Ridit Scores)
# ----------------------------------------------------------------------
total_pop = data['population'].sum()
data['pop_share'] = data['population'] / total_pop
data['cum_pop'] = data['pop_share'].cumsum()
data['ridit_rank'] = data['cum_pop'] - 0.5 * data['pop_share']

# ----------------------------------------------------------------------
# 3. Fit Linear Weighted Least Squares (WLS) Model
# ----------------------------------------------------------------------
# y_j = alpha + beta * x_j, weighted by population share
X = sm.add_constant(data['ridit_rank'])
y = data['rate']
wls_model = sm.WLS(y, X, weights=data['pop_share']).fit()

alpha_hat = wls_model.params['const']
beta_hat = wls_model.params['ridit_rank']
se_beta = wls_model.bse['ridit_rank']
cov_matrix = wls_model.cov_params()

# Slope Index of Inequality (SII)
# Convention: rank 0 = most deprived, rank 1 = least deprived
sii = -beta_hat
sii_se = se_beta
z_crit = stats.norm.ppf(0.975)
sii_ci_lower = sii - z_crit * sii_se
sii_ci_upper = sii + z_crit * sii_se

# Relative Index of Inequality (RII via Delta Method)
y_most = alpha_hat
y_least = alpha_hat + beta_hat
rii = y_most / y_least

# Delta method variance for RII = alpha / (alpha + beta)
d_alpha = beta_hat / ((alpha_hat + beta_hat) ** 2)
d_beta = -alpha_hat / ((alpha_hat + beta_hat) ** 2)
var_alpha = cov_matrix.loc['const', 'const']
var_beta = cov_matrix.loc['ridit_rank', 'ridit_rank']
cov_ab = cov_matrix.loc['const', 'ridit_rank']
var_rii = (d_alpha**2)*var_alpha + (d_beta**2)*var_beta + 2*d_alpha*d_beta*cov_ab
se_rii = np.sqrt(max(0.0, var_rii))
rii_ci_lower = rii - z_crit * se_rii
rii_ci_upper = rii + z_crit * se_rii

# ----------------------------------------------------------------------
# 4. Display Formatted Epidemiological Results
# ----------------------------------------------------------------------
print("==================================================================")
print("  HEALTH INEQUALITY GRADIENT REPORT (OHID / WHO STANDARDS)       ")
print("==================================================================")
print(f"Outcome Indicator:           {res['indicator_name']}")
print(f"Number of Strata:            {{len(data)}}")
print(f"Total Population:            {{total_pop:,}}")
print(f"Weighted R-squared:          {{wls_model.rsquared:.1%}}")
print("------------------------------------------------------------------")
print(f"Slope Index of Inequality (SII): {{sii:.2f}} (95% CI: [{{sii_ci_lower:.2f}}, {{sii_ci_upper:.2f}}])")
print(f"Relative Index of Inequality (RII): {{rii:.2f}}x (95% CI: [{{rii_ci_lower:.2f}}, {{rii_ci_upper:.2f}}])")
print(f"Fitted Rate Most Deprived (x=0):  {{y_most:.2f}}")
print(f"Fitted Rate Least Deprived (x=1): {{y_least:.2f}}")
print("==================================================================")
"""
    st.code(code_template, language="python")


def render_health_inequalities_download_tab(res: Dict[str, Any]) -> None:
    """Renders the downloadable CSV export tab with strata metrics."""
    st.markdown("### 📥 Data Export & Microdata Breakdown")

    df = pd.DataFrame(res["table"])
    df.columns = [
        "Socioeconomic Stratum",
        "Population",
        "Population Share",
        "Midpoint Rank (Ridit)",
        f"Observed {res['indicator_name']}",
        f"Fitted Model Rate",
        "Residual Error"
    ]

    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Inequality Dataset (CSV)",
        data=csv_data,
        file_name="health_inequality_sii_rii_data.csv",
        mime="text/csv",
        key="btn_download_inequality_csv"
    )


def render_health_inequalities_module(active_method_key: str = "health_inequalities", method_info: Optional[Dict[str, Any]] = None) -> None:
    """Top-level Streamlit controller for the Health Inequalities (SII & RII) module."""
    # 1. Zero-Prior-Knowledge Pedagogical Guide
    render_health_inequalities_plain_english_guide()

    st.markdown("---")

    # 2. Scenario Selector & Parameter Inputs
    st.subheader("🎯 Configure Health Inequality Scenario")
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        scenario_key = st.selectbox(
            "Select Public Health Educational Scenario (Synthetic Illustrative Data):",
            options=["derbyshire_cvd", "life_expectancy_gap", "childhood_obesity"],
            format_func=lambda k: {
                "derbyshire_cvd": "🏥 Example 1: Under-75 CVD Mortality across 10 IMD Deciles (Synthetic Demo)",
                "life_expectancy_gap": "⏳ Example 2: Male Life Expectancy at Birth across 5 Deprivation Quintiles (Synthetic Demo)",
                "childhood_obesity": "🍎 Example 3: Year 6 Childhood Obesity Prevalence across 10 Deciles (Synthetic Demo)"
            }[k],
            key="hi_scenario_select"
        )
    with sc2:
        model_choice = st.selectbox(
            "Regression Model Type:",
            options=["linear", "log_linear"],
            format_func=lambda m: "Linear WLS (Standard OHID / Fingertips)" if m == "linear" else "Log-Linear GLM (Exponential link)",
            key="hi_model_type_select"
        )

    scenario_data = generate_inequality_demo_data(scenario_key)
    st.caption(f"**Scenario Context:** {scenario_data['description']}")
    if "disclaimer" in scenario_data:
        st.warning(scenario_data["disclaimer"])

    # 3. Compute Metrics
    res = calculate_sii_rii(
        rates=scenario_data["rates"],
        populations=scenario_data["populations"],
        strata_names=scenario_data["strata_names"],
        indicator_name=scenario_data["indicator_name"],
        reverse_direction=scenario_data.get("reverse_direction", False),
        model_type=model_choice
    )

    # 4. Prerequisite & Diagnostics Audit
    diag = evaluate_health_inequalities_prerequisites(scenario_data["rates"], scenario_data["populations"])
    render_health_inequalities_diagnostics(diag)

    st.markdown("---")

    # 5. Top Metrics Dashboard
    render_health_inequalities_dashboard(res)

    # 6. Interactive Visualizations
    render_health_inequalities_visualizations(res)

    st.markdown("---")

    # 7. Standard 4-Tab Pedagogical Container
    tab_theory, tab_excel, tab_python, tab_download = st.tabs([
        "📖 Mathematical Formulation & Plain-English Explanations",
        "📋 Microsoft Excel Step-by-Step Guide",
        "🐍 Production Python Pipeline",
        "📥 Practice Data & CSV Export"
    ])

    with tab_theory:
        st.markdown(r"""
        ### Mathematical Derivation: Slope Index of Inequality (SII) & Relative Index (RII)
        
        #### 1. The Cumulative Relative Rank (Ridit Score)
        In social epidemiology, strata (e.g. IMD deciles) are ordered from **most disadvantaged ($j=1$) to least disadvantaged ($j=K$)**.
        To ensure groups are weighted proportionally to their population size, each stratum is assigned a midpoint rank $x_j \in (0, 1)$:
        
        $$x_j = \left( \sum_{i=1}^{j-1} p_i \right) + \frac{1}{2} p_j, \quad \text{where } p_j = \frac{n_j}{\sum_{k=1}^K n_k}$$
        
        * $p_j$: The proportion of the total population residing in socioeconomic stratum $j$.
        * $\sum_{i=1}^{j-1} p_i$: The cumulative population share living in all more-deprived strata preceding group $j$.
        * $x_j$: The relative position of the median individual in stratum $j$ along the continuous deprivation continuum $[0, 1]$.
        
        #### 2. Population-Weighted Least Squares (WLS) Regression
        The health outcome $y_j$ is regressed onto the relative rank $x_j$ using population share weights $w_j = p_j$:
        
        $$\hat{y}_j = \alpha + \beta x_j$$
        
        $$\beta = \frac{\sum_{j=1}^K w_j (x_j - \bar{x})(y_j - \bar{y})}{\sum_{j=1}^K w_j (x_j - \bar{x})^2}, \quad \alpha = \bar{y} - \beta \bar{x}$$
        
        #### 3. Slope Index of Inequality (SII — Absolute Inequality)
        The SII measures the difference in the fitted health indicator between the two hypothetical extremes of the population:
        
        $$\text{SII} = \hat{y}(0) - \hat{y}(1) = \alpha - (\alpha + \beta) = -\beta$$
        
        * If $\beta < 0$ (mortality drops as affluence increases), $\text{SII} > 0$: represents the **absolute excess disease burden** suffered by the most deprived individual.
        * Standard error: $\text{SE}(\text{SII}) = \text{SE}(\beta) = \sqrt{\frac{\text{MSE}}{\sum w_j (x_j - \bar{x})^2}}$.
        * 95% Confidence Interval: $\text{SII} \pm 1.96 \times \text{SE}(\beta)$.
        
        #### 4. Relative Index of Inequality (RII — Relative Ratio)
        The RII measures the proportional ratio between the estimated health status at the extremes:
        
        $$\text{RII} = \frac{\hat{y}(0)}{\hat{y}(1)} = \frac{\alpha}{\alpha + \beta}$$
        
        * If $\text{RII} = 2.50$, the most disadvantaged extreme experiences **$2.5$ times the rate** of the most advantaged extreme.
        * Variance via the **Delta Method**:
        
        $$\text{Var}(\text{RII}) \approx \left(\frac{\beta}{(\alpha + \beta)^2}\right)^2 \text{Var}(\alpha) + \left(\frac{-\alpha}{(\alpha + \beta)^2}\right)^2 \text{Var}(\beta) + 2 \left(\frac{-\alpha \beta}{(\alpha + \beta)^4}\right) \text{Cov}(\alpha, \beta)$$
        """)

    with tab_excel:
        render_health_inequalities_excel_guide(res)

    with tab_python:
        render_health_inequalities_python_guide(res)

    with tab_download:
        render_health_inequalities_download_tab(res)

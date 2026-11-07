"""UI Components and Interactive Educational Visualizers for Time Series & Forecasting.

Provides interactive dashboards for:
1. Time series header & foundational overview.
2. Anatomy & Diagnostic Lab (Decomposition Additive/Multiplicative, ACF/PACF, ADF Test, Differencing).
3. Forecasting Models Sandbox (Naive, Seasonal Naive, Drift, SES, Holt's Damped, Holt-Winters, Lagged ML, Croston)
   with dynamic plain-English algorithm profiles, assumptions, pros/cons, and formula symbol dissections.
4. Evaluation Metrics Deep Dive & Interactive Outlier/Zero Laboratory (MAE, MSE, RMSE, MAPE, sMAPE, MASE, MDA)
   with exhaustive plain-English symbol dissections and tooltips.
5. Temporal Validation Protocol (Rolling-Origin Expanding-Window Cross-Validation).
6. Which Metric Should I Use? Interactive Decision Wizard.
"""

import math
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import streamlit as st
except ImportError:
    st = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.time_series.ts_engine import (
    generate_ed_demand_series,
    generate_seasonal_flu_series,
    generate_ambulance_callout_series,
    generate_intermittent_stockout_series,
    decompose_time_series,
    calculate_acf,
    calculate_pacf,
    calculate_adf_test,
    difference_series,
    forecast_naive,
    forecast_seasonal_naive,
    forecast_drift,
    forecast_simple_exponential_smoothing,
    forecast_holt_linear,
    forecast_holt_winters,
    forecast_lagged_regression,
    forecast_croston,
    calculate_forecast_metrics,
    rolling_origin_evaluation,
    TIME_SERIES_ALGORITHMS,
    get_algorithm_profile,
    generate_time_series_python_code,
    generate_time_series_excel_guide,
    generate_time_series_csv,
)


def render_html(html_str: str) -> None:
    """Safely renders HTML markup without markdown indent bugs."""
    clean_html = "\n".join(line.strip() for line in html_str.strip().split("\n") if line.strip())
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)


# =====================================================================
# 1. Header & Foundational Principles
# =====================================================================

def render_time_series_header() -> None:
    """Renders the top banner and introductory pedagogical summary."""
    st.title("⏱️ Time Series Analysis & Operational Forecasting")
    st.caption("Pillar 7: Sequential Temporal Modeling, Classical Decomposition, Model Hierarchies & Evaluation Metrics")

    st.markdown("""
    When data points are collected sequentially across chronological time, **classical independent-and-identically-distributed (i.i.d.) statistical assumptions completely break down**.
    Past values directly influence future values (*temporal autocorrelation*), ordering cannot be randomized, and conventional $K$-fold cross-validation causes catastrophic future-to-past data leakage.
    """)

    with st.expander("🛡️ The 3 Golden Rules of Time Series Analysis", expanded=False):
        rules_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #2563eb; padding: 0.9rem 1.1rem; border-radius: 4px; font-size: 0.88rem; line-height: 1.5; color: #1e293b;">
            <div style="font-weight: 700; color: #1e40af; margin-bottom: 0.35rem;">
                📌 Fundamental Axioms for Every Forecaster
            </div>
            • <strong>1. Never Shuffle Chronological Data:</strong> Shuffling data points across time destroys temporal dependencies and causes <em>lookahead bias</em>. Standard $K$-fold splits are mathematically invalid; you must strictly enforce <strong>Rolling-Origin (Expanding-Window) Evaluation</strong>.<br>
            • <strong>2. Always Beat the Naive Baseline First:</strong> Complex Machine Learning models and deep neural nets are useless if they cannot outperform a simple <em>Seasonal Naive baseline</em> ($y_{t+h} = y_{t+h-m}$) on held-out evaluation periods.<br>
            • <strong>3. Match the Evaluation Metric to Real-World Operational Cost:</strong> There is no single "best" metric. Use <strong>RMSE</strong> when large errors cause catastrophic shortages (e.g. ICU bed deficits), <strong>MAE</strong> for linear costs with noisy outliers, and <strong>MASE</strong> for cross-department comparisons or intermittent zero-heavy demand.
        </div>
        """
        render_html(rules_html)


# =====================================================================
# 2. Tab 1: Anatomy & Diagnostic Lab (Decomposition, Stationarity, ACF/PACF)
# =====================================================================

def render_anatomy_tab(scenario_data: Dict[str, Any]) -> None:
    """Renders time series components, interactive decomposition, ACF/PACF, and ADF test."""
    st.subheader("🔬 Module 1: Time Series Anatomy & Statistical Diagnostics")
    st.markdown("""
    Before applying any forecasting algorithm, a forecaster must inspect the series for its **underlying anatomy**:
    trend trajectory, seasonal frequency, variance stability, and **stationarity**.
    """)

    series = scenario_data["series"]
    m = scenario_data["seasonal_period"]
    dates = scenario_data["dates"]

    # 1. Raw Series Plot
    st.markdown(f"#### 1. Observed Chronological Sequence: {scenario_data['title']} (N = {len(series)} {scenario_data['frequency']} observations)")
    if HAS_PLOTLY:
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(
            x=dates,
            y=series,
            mode="lines+markers",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=4, color="#1e40af"),
            name="Observed Demand"
        ))
        fig_raw.update_layout(
            title=None,
            xaxis_title="Time Horizon",
            yaxis_title=scenario_data["unit"],
            template="plotly_white",
            height=340,
            margin=dict(l=40, r=40, t=25, b=40)
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    # 2. Classical Decomposition Lab
    st.markdown("---")
    st.markdown("#### 2. Classical Seasonal Decomposition Laboratory")
    st.caption("Deconstructs the observed series into: **Level/Trend ($T_t$)**, **Seasonality ($S_t$)**, and **Residual Noise ($R_t$)**.")

    col_dec1, col_dec2 = st.columns([1, 2])
    with col_dec1:
        decomp_mode = st.radio(
            "Select Decomposition Mathematical Mode:",
            options=["additive", "multiplicative"],
            format_func=lambda x: "➕ Additive: Y_t = Trend + Seasonal + Residual" if x == "additive" else "✖️ Multiplicative: Y_t = Trend × Seasonal × Residual",
            key="ts_decomp_mode_select",
            help="Choose 'Additive' if the seasonal swings stay constant over time, or 'Multiplicative' if the seasonal surges get bigger when overall demand grows."
        )
        st.info(
            "**Additive** assumes seasonal oscillations have constant magnitude across time. "
            "**Multiplicative** assumes seasonal swings scale proportionally with rising/falling demand volume."
        )

    try:
        decomp = decompose_time_series(series, period=m, mode=decomp_mode)
    except Exception as e:
        st.error(f"Decomposition error: {e}")
        return

    with col_dec2:
        m1, m2 = st.columns(2)
        m1.metric(
            "Trend Strength (Ft)",
            f"{decomp['trend_strength']:.3f}",
            help="Trend Strength (from 0.0 to 1.0): Indicates how much of the total variation is driven by a long-term upward or downward direction. Values > 0.60 mean the series has a strong trend."
        )
        m2.metric(
            "Seasonal Strength (Fs)",
            f"{decomp['seasonal_strength']:.3f}",
            help="Seasonal Strength (from 0.0 to 1.0): Indicates how dominant and predictable the recurring calendar cycles are. Values > 0.60 mean day-of-week or monthly patterns heavily influence demand."
        )

    if HAS_PLOTLY:
        fig_dec = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[
                f"1. Observed Series ({scenario_data['unit']})",
                f"2. Trend-Cycle Component (Centered Moving Average, m={m})",
                f"3. Seasonal Component (Phase Period = {m})",
                "4. Irregular Residuals (Noise / Shocks)"
            ]
        )
        fig_dec.add_trace(go.Scatter(x=dates, y=decomp["observed"], line=dict(color="#1e293b", width=1.8), name="Observed"), row=1, col=1)
        fig_dec.add_trace(go.Scatter(x=dates, y=decomp["trend"], line=dict(color="#dc2626", width=2), name="Trend"), row=2, col=1)
        fig_dec.add_trace(go.Scatter(x=dates, y=decomp["seasonal"], line=dict(color="#16a34a", width=1.8), name="Seasonal"), row=3, col=1)
        fig_dec.add_trace(go.Scatter(x=dates, y=decomp["residuals"], mode="markers+lines", line=dict(color="#9333ea", width=1), marker=dict(size=4), name="Residual"), row=4, col=1)

        fig_dec.update_layout(height=600, template="plotly_white", showlegend=False, margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig_dec, use_container_width=True)

    with st.expander("📐 Classical Decomposition Equations & Plain-English Symbol Dissection", expanded=False):
        st.markdown(r"""
        #### Additive Decomposition Model:
        $$Y_t = T_t + S_t + R_t$$

        * **$Y_t$ (Observed Demand):** The actual recorded measurement on day or week $t$ (e.g. 142 patient arrivals on Monday).
        * **$T_t$ (Trend-Cycle Component):** The underlying baseline trajectory calculated via a centered moving average of length $m$. It shows whether overall demand is rising, falling, or staying flat over months, completely ignoring day-of-week fluctuations.
        * **$S_t$ (Seasonal Component):** The recurring cyclical deviation associated with that specific calendar day or month. In an additive model, this is an absolute number added or subtracted from the trend (e.g. $+25$ patients every Monday, $-15$ patients every Wednesday). Across a full cycle of $m$ days, the seasonal values sum to approximately zero ($\sum_{i=1}^m S_i \approx 0$).
        * **$R_t$ (Residual / Irregular Component):** The leftover unexplained noise or random shocks calculated by subtraction ($R_t = Y_t - T_t - S_t$). This reveals unexpected anomalies, such as an IT outage or sudden epidemic spike.

        ---

        #### Multiplicative Decomposition Model:
        $$Y_t = T_t \times S_t \times R_t$$

        * **$S_t$ (Seasonal Multiplier):** In a multiplicative model, seasonality is expressed as a ratio or percentage of the baseline trend. A value of $1.25$ means demand on that day is typically $25\%$ higher than the trend level; a value of $0.80$ means demand is $20\%$ lower. Across a full cycle, the average seasonal multiplier is $1.00$.
        * **$R_t$ (Residual Multiplier):** The random shock expressed as a ratio ($R_t = Y_t / (T_t \times S_t)$). A residual of $1.00$ indicates perfect agreement between reality and the seasonal trend.
        """)

    # 3. Stationarity & Autocorrelation Lab (ACF, PACF, ADF Test)
    st.markdown("---")
    st.markdown("#### 3. Autocorrelation & Stationarity Diagnostics (ACF / PACF / ADF)")
    st.caption("Stationarity is required for ARIMA modeling and prevents spurious statistical regressions.")

    col_stat1, col_stat2 = st.columns([1.2, 1.8])
    with col_stat1:
        apply_diff = st.checkbox(
            "🔄 Apply First Differencing (Δ Y_t = Y_t - Y_{t-1})",
            value=False,
            key="ts_apply_diff_chk",
            help="Differencing subtracts yesterday's value from today's value. It transforms a trending series into a stationary series of daily changes."
        )
        st.markdown(r"""
        * **Stationary Definition:** Mean, variance, and autocovariance are constant over time.
        * **Differencing ($\Delta$):** Subtracting the previous observation removes linear trend and stabilizes the mean.
        """)

    active_series = difference_series(series, 1) if apply_diff else list(series)
    active_dates = dates[1:] if apply_diff else dates

    # Run ADF test
    adf_res = calculate_adf_test(active_series)
    with col_stat1:
        if adf_res["is_stationary"]:
            st.success(
                f"✅ **Augmented Dickey-Fuller Test:** `{adf_res['conclusion']}`\n\n"
                f"* **Test Statistic:** `{adf_res['test_statistic']:.3f}` (Critical 5%: `{adf_res['critical_values']['5%']}`)\n"
                f"* **p-value:** `{adf_res['p_value']:.3f}`"
            )
        else:
            st.warning(
                f"⚠️ **Augmented Dickey-Fuller Test:** `{adf_res['conclusion']}`\n\n"
                f"* **Test Statistic:** `{adf_res['test_statistic']:.3f}` (Critical 5%: `{adf_res['critical_values']['5%']}`)\n"
                f"* **p-value:** `{adf_res['p_value']:.3f}`"
            )

    # Compute ACF & PACF
    acf_data = calculate_acf(active_series, max_lags=min(21, len(active_series) // 2))
    pacf_data = calculate_pacf(active_series, max_lags=min(14, len(active_series) // 3))

    with col_stat2:
        if HAS_PLOTLY:
            fig_corr = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Autocorrelation Function (ACF)", "Partial Autocorrelation (PACF)"]
            )
            # ACF Lollipop
            lags = acf_data["lags"]
            for lag, val in zip(lags, acf_data["acf"]):
                fig_corr.add_trace(go.Scatter(x=[lag, lag], y=[0, val], mode="lines", line=dict(color="#2563eb", width=2), showlegend=False), row=1, col=1)
            fig_corr.add_trace(go.Scatter(x=lags, y=acf_data["acf"], mode="markers", marker=dict(size=6, color="#1e40af"), name="ACF"), row=1, col=1)
            # Bartlett confidence band
            ci_u = acf_data["ci_upper"]
            ci_l = acf_data["ci_lower"]
            fig_corr.add_trace(go.Scatter(x=[0, max(lags)], y=[ci_u, ci_u], mode="lines", line=dict(color="#ef4444", dash="dash", width=1.5), name="+95% CI"), row=1, col=1)
            fig_corr.add_trace(go.Scatter(x=[0, max(lags)], y=[ci_l, ci_l], mode="lines", line=dict(color="#ef4444", dash="dash", width=1.5), showlegend=False), row=1, col=1)

            # PACF Lollipop
            p_lags = pacf_data["lags"]
            for lag, val in zip(p_lags, pacf_data["pacf"]):
                fig_corr.add_trace(go.Scatter(x=[lag, lag], y=[0, val], mode="lines", line=dict(color="#9333ea", width=2), showlegend=False), row=1, col=2)
            fig_corr.add_trace(go.Scatter(x=p_lags, y=pacf_data["pacf"], mode="markers", marker=dict(size=6, color="#6b21a8"), name="PACF"), row=1, col=2)
            p_ci_u = pacf_data["ci_upper"]
            p_ci_l = pacf_data["ci_lower"]
            fig_corr.add_trace(go.Scatter(x=[0, max(p_lags)], y=[p_ci_u, p_ci_u], mode="lines", line=dict(color="#ef4444", dash="dash", width=1.5), showlegend=False), row=1, col=2)
            fig_corr.add_trace(go.Scatter(x=[0, max(p_lags)], y=[p_ci_l, p_ci_l], mode="lines", line=dict(color="#ef4444", dash="dash", width=1.5), showlegend=False), row=1, col=2)

            fig_corr.update_layout(height=340, template="plotly_white", margin=dict(l=20, r=20, t=35, b=30))
            st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("📖 Autocorrelation (ACF/PACF) & ADF Formulas in Plain English", expanded=False):
        st.markdown(r"""
        #### 1. Sample Autocorrelation Function ($r_k$):
        $$r_k = \frac{\sum_{t=k+1}^N (Y_t - \bar{Y})(Y_{t-k} - \bar{Y})}{\sum_{t=1}^N (Y_t - \bar{Y})^2}$$

        * **$r_k$ (Autocorrelation at Lag $k$):** A correlation coefficient between $-1.0$ and $+1.0$ measuring how strongly today's demand ($Y_t$) relates to the demand recorded exactly $k$ days ago ($Y_{t-k}$).
        * **Numerator:** Calculates the covariance between pairs separated by $k$ days. If Mondays are consistently followed by high attendances 7 days later, this numerator is strongly positive.
        * **Denominator:** The total sample variance across the entire series. Dividing by the variance scales the result so $r_0 = 1.0$ and all other lags fall strictly between $-1.0$ and $+1.0$.
        * **Bartlett 95% Confidence Band ($\pm 1.96 / \sqrt{N}$):** The red dashed threshold. If a blue spike crosses outside this band, there is less than a 5% probability that the correlation happened by random noise.

        ---

        #### 2. Augmented Dickey-Fuller (ADF) Regression:
        $$\Delta Y_t = \alpha + \gamma Y_{t-1} + \delta \Delta Y_{t-1} + \epsilon_t$$

        * **$\Delta Y_t$ (First Difference):** The day-to-day change in demand ($Y_t - Y_{t-1}$).
        * **$\gamma$ (Mean-Reverting Coefficient):** The core test parameter. If $\gamma = 0$, the series has a "unit root" (meaning past shocks persist forever, causing an uncontrollable wandering drift = Non-Stationary). If $\gamma < 0$, the series pulls itself back toward a stable long-term mean (= Stationary).
        * **$t$-statistic ($\hat{\gamma} / \text{SE}(\hat{\gamma})$):** If this value is more negative than the 5% critical cutoff (typically $-2.89$), we reject the null hypothesis of a unit root and conclude the series is stationary.
        """)


# =====================================================================
# 3. Tab 2: Forecasting Models & Sandbox
# =====================================================================

def render_models_tab(scenario_data: Dict[str, Any]) -> None:
    """Renders the interactive forecasting model sandbox comparing baselines, ETS, ARIMA, and ML."""
    st.subheader("🔮 Module 2: The Forecasting Model Hierarchy Sandbox")
    st.markdown("""
    Explore the progression from foundational benchmarks to exponential smoothing, state space models, and lagged machine learning.
    """)

    series = scenario_data["series"]
    dates = scenario_data["dates"]
    m = scenario_data["seasonal_period"]

    c_m1, c_m2, c_m3 = st.columns([1.3, 1.0, 1.0])
    with c_m1:
        model_choice = st.selectbox(
            "Select Forecasting Algorithm:",
            options=[
                "1. Naive Baseline (Last Value)",
                "2. Seasonal Naive Baseline",
                "3. Drift Baseline (Linear Extrapolation)",
                "4. Simple Exponential Smoothing (SES)",
                "5. Holt's Linear Trend (Damped)",
                "6. Holt-Winters Triple Exponential Smoothing",
                "7. Lagged Feature Ridge Regression (ML)",
                "8. Croston SBA (Intermittent Demand)"
            ],
            key="ts_model_sandbox_select",
            help="Choose any algorithm to inspect its forecast, parameter controls, mathematical formula, and operational pros/cons."
        )
    with c_m2:
        horizon = st.slider(
            "Forecast Horizon Steps ($h$):",
            min_value=1,
            max_value=28,
            value=14,
            key="ts_horizon_slider",
            help="How many time steps into the future you want to project (e.g. 14 days or 14 weeks)."
        )
    with c_m3:
        if "Exponential" in model_choice or "Holt" in model_choice or "Simple" in model_choice or "Croston" in model_choice:
            alpha_val = st.slider(
                "Level Smoothing α (Alpha):",
                min_value=0.01,
                max_value=0.99,
                value=0.28,
                step=0.02,
                key="ts_alpha_slider",
                help="Alpha (0.01 to 0.99): Determines memory speed. Higher values (e.g. 0.8) pay more attention to recent days; lower values (e.g. 0.1) produce a smoother, slower-moving baseline."
            )
        else:
            alpha_val = 0.28

    # Optional model-specific parameters
    beta_val, gamma_val = 0.05, 0.30
    if "Holt's" in model_choice or "Holt-Winters" in model_choice:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            beta_val = st.slider(
                "Trend Smoothing β (Beta):",
                min_value=0.01,
                max_value=0.50,
                value=0.06,
                step=0.01,
                key="ts_beta_slider",
                help="Beta (0.01 to 0.50): Controls how quickly the model updates its estimate of the upward or downward slope."
            )
        with col_p2:
            if "Holt-Winters" in model_choice:
                gamma_val = st.slider(
                    "Seasonal Smoothing γ (Gamma):",
                    min_value=0.01,
                    max_value=0.80,
                    value=0.30,
                    step=0.02,
                    key="ts_gamma_slider",
                    help="Gamma (0.01 to 0.80): Controls how quickly the model adapts to changing seasonal peaks (e.g. Monday surges changing shape over time)."
                )

    profile = get_algorithm_profile(model_choice)

    # ----------------- Fit Model & Generate Projections -----------------
    if "1. Naive" in model_choice:
        res = forecast_naive(series, horizon=horizon)
    elif "2. Seasonal" in model_choice:
        res = forecast_seasonal_naive(series, period=m, horizon=horizon)
    elif "3. Drift" in model_choice:
        res = forecast_drift(series, horizon=horizon)
    elif "4. Simple" in model_choice:
        res = forecast_simple_exponential_smoothing(series, alpha=alpha_val, horizon=horizon)
    elif "5. Holt's" in model_choice:
        res = forecast_holt_linear(series, alpha=alpha_val, beta=beta_val, damped=True, horizon=horizon)
    elif "6. Holt-Winters" in model_choice:
        res = forecast_holt_winters(series, period=m, alpha=alpha_val, beta=beta_val, gamma=gamma_val, horizon=horizon)
    elif "7. Lagged" in model_choice:
        res = forecast_lagged_regression(series, lags=[1, 2, m], seasonal_period=m, horizon=horizon)
    else:
        res = forecast_croston(series, alpha=alpha_val, horizon=horizon)

    # In-sample metrics on last test window
    test_len = min(horizon, len(series) // 4)
    y_test = series[-test_len:]
    y_pred_fitted = res["fitted"][-test_len:]
    fit_metrics = calculate_forecast_metrics(y_test, y_pred_fitted, y_train=series[:-test_len], seasonal_period=m)

    # KPI summary with rich Plain English tooltips
    st.markdown("#### Test Window Performance Summary")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "MAE (Scale Error)",
        f"{fit_metrics['mae']:.2f} {scenario_data['unit']}",
        help="Mean Absolute Error: The average positive distance between predicted demand and actual demand in raw units (e.g. patients). Lower is better. A value of 6.5 means your forecast is off by an average of 6.5 patients."
    )
    k2.metric(
        "RMSE (Quad Error)",
        f"{fit_metrics['rmse']:.2f}",
        help="Root Mean Squared Error: Measures forecast accuracy while heavily penalizing large blunders. Because errors are squared before averaging, a single massive miss will significantly inflate this number. Always greater than or equal to MAE."
    )
    if fit_metrics["mape"] is not None:
        k3.metric(
            "MAPE (Percentage)",
            f"{fit_metrics['mape']:.1f}%",
            help="Mean Absolute Percentage Error: The average percentage deviation of the forecast relative to the actual outcome. Expressed from 0% to 100%+. Only valid when actual values are strictly positive non-zeros."
        )
    else:
        k3.metric(
            "MAPE (Percentage)",
            "Undefined (0s)",
            help="Mean Absolute Percentage Error cannot be calculated because the data contains zero values (division by zero error)."
        )
    k4.metric(
        "MASE (vs Naive)",
        f"{fit_metrics['mase']:.3f}" if fit_metrics['mase'] else "N/A",
        help="Mean Absolute Scaled Error: Compares your model's MAE to a simple Naive benchmark. Values < 1.0 mean your model beats naive guessing (e.g. 0.85 = 15% better). Values > 1.0 mean your model is worse than an elementary baseline."
    )
    k5.metric(
        "Directional Acc (MDA)",
        f"{fit_metrics['mda']:.1f}%",
        help="Mean Directional Accuracy: The percentage of time periods where your model correctly anticipated whether demand would increase or decrease compared to the previous day. Crucial for shift staffing and capacity planning."
    )

    # Plot Forecast
    st.markdown(f"#### 📈 Multi-Step Forward Demand Projection: {res['model_name']}")
    if HAS_PLOTLY:
        forecast_dates = [f"Horizon +{h}" for h in range(1, horizon + 1)]
        fig_fc = go.Figure()

        # Historical Observed
        fig_fc.add_trace(go.Scatter(
            x=dates, y=series,
            mode="lines",
            line=dict(color="#1e293b", width=2),
            name="Historical Observed"
        ))
        # In-sample fitted
        fig_fc.add_trace(go.Scatter(
            x=dates, y=res["fitted"],
            mode="lines",
            line=dict(color="#0284c7", width=1.5, dash="dot"),
            name="In-Sample Fitted"
        ))
        # 95% Confidence Corridor
        fig_fc.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=res["upper_95"] + res["lower_95"][::-1],
            fill="toself",
            fillcolor="rgba(147, 51, 234, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Prediction Interval"
        ))
        # Forward Forecast
        fig_fc.add_trace(go.Scatter(
            x=forecast_dates, y=res["forecast"],
            mode="lines+markers",
            line=dict(color="#9333ea", width=2.5),
            marker=dict(size=6, color="#7e22ce"),
            name=f"Forward Forecast ({res['model_name']})"
        ))

        fig_fc.update_layout(
            title=None,
            xaxis_title="Time Steps",
            yaxis_title=scenario_data["unit"],
            template="plotly_white",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=30, b=40)
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    # ----------------- The 4 Standard Educational & Implementation Tabs -----------------
    st.markdown("---")
    st.markdown(f"### 📚 Methodological Mastery & Implementation: `{profile['name']}`")

    tab_method_exp, tab_method_py, tab_method_xls, tab_method_csv = st.tabs([
        "📖 Explanation of the Method",
        "💻 Implementation in Python",
        "📊 Excel Implementation",
        "📥 Download Practice Data (CSV)"
    ])

    with tab_method_exp:
        st.caption(f"**Category:** {profile['category']} • *{profile['tagline']}*")
        with st.container(border=True):
            col_exp1, col_exp2 = st.columns([1.1, 1.1])
            with col_exp1:
                st.markdown(f"**🔍 What is this Algorithm?**\n\n{profile['what_it_is']}")
                st.markdown("**🎯 When to Use:**")
                for item in profile["when_to_use"]:
                    st.markdown(f"• {item}")
                st.markdown("**🚫 When NOT to Use:**")
                for item in profile["when_not_to_use"]:
                    st.markdown(f"• {item}")

            with col_exp2:
                st.markdown("**🛡️ Assumptions That Must Be Met:**")
                for item in profile["assumptions"]:
                    st.markdown(f"• {item}")
                st.markdown("**✅ Relative Pros:**")
                for item in profile["pros"]:
                    st.markdown(f"• {item}")
                st.markdown("**⚠️ Relative Cons & Failure Modes:**")
                for item in profile["cons"]:
                    st.markdown(f"• {item}")

        with st.expander(f"📐 Plain-English Mathematical Formula & Symbol Dissection: {profile['name']}", expanded=False):
            st.markdown("#### Mathematical Formulation:")
            raw_formula = profile["formula"].strip("$").strip()
            st.latex(raw_formula)
            st.markdown("#### Variable-by-Variable Plain English Breakdown:")
            for sym, desc in profile["variables_explained"].items():
                st.markdown(f"• {sym}: {desc}")

    with tab_method_py:
        st.markdown(f"#### 🐍 Production-Grade Python Pipeline: `{profile['name']}`")
        st.markdown("""
        The executable script below demonstrates end-to-end data ingestion, parameter configuration,
        forward extrapolation, uncertainty corridor estimation (95% prediction intervals), and performance tracking.
        """)
        py_script = generate_time_series_python_code(
            profile["id"],
            scenario_data,
            params={
                "horizon": horizon,
                "alpha": alpha_val,
                "beta": beta_val,
                "gamma": gamma_val,
                "phi": 0.95
            }
        )
        st.code(py_script, language="python")
        st.download_button(
            label=f"📥 Download Python Script (forecast_{profile['id']}.py)",
            data=py_script,
            file_name=f"forecast_{profile['id']}.py",
            mime="text/x-python",
            key=f"dl_py_script_{profile['id']}"
        )

    with tab_method_xls:
        xls_guide = generate_time_series_excel_guide(
            profile["id"],
            scenario_data,
            params={
                "horizon": horizon,
                "alpha": alpha_val,
                "beta": beta_val,
                "gamma": gamma_val,
                "phi": 0.95
            }
        )
        st.markdown(xls_guide)

    with tab_method_csv:
        st.markdown(f"#### 📥 Practice Dataset for Offline Modeling: `{scenario_data['title']}`")
        st.markdown("""
        Download the active synthetic dataset to practice model fitting, seasonal decomposition,
        and spreadsheet formula verification offline in Microsoft Excel, Google Sheets, or Python.
        """)

        if pd is not None:
            df_preview = pd.DataFrame({
                "Period": scenario_data["dates"],
                "Observation_Index": list(range(1, len(scenario_data["series"]) + 1)),
                scenario_data["unit"]: scenario_data["series"]
            })
            if "true_trend" in scenario_data and len(scenario_data["true_trend"]) == len(scenario_data["series"]):
                df_preview["Underlying_Trend"] = scenario_data["true_trend"]
            if "true_seasonal" in scenario_data and len(scenario_data["true_seasonal"]) == len(scenario_data["series"]):
                df_preview["Seasonal_Component"] = scenario_data["true_seasonal"]
            st.dataframe(df_preview, use_container_width=True, height=280)

        csv_str = generate_time_series_csv(scenario_data)
        st.download_button(
            label=f"📥 Download Practice Dataset as CSV ({scenario_data['scenario_id']}.csv)",
            data=csv_str,
            file_name=f"{scenario_data['scenario_id']}_practice_data.csv",
            mime="text/csv",
            key=f"dl_csv_dataset_{scenario_data['scenario_id']}_{profile['id']}"
        )
        st.caption("💡 **Practice Exercise:** Import this CSV into Excel and follow the instructions in the **📊 Excel Implementation** tab above to replicate the forecast.")


# =====================================================================
# 4. Tab 3: Evaluation Metrics Deep Dive & Interactive Laboratory
# =====================================================================

def render_evaluation_metrics_tab(scenario_data: Dict[str, Any]) -> None:
    """Renders the comprehensive evaluation metrics deep dive with interactive Outlier & Zero Injectors."""
    st.subheader("🎯 Module 3: Forecast Evaluation Metrics Deep Dive & Decision Lab")
    st.markdown("""
    Choosing the wrong metric can lead you to deploy a model that systematically under-forecasts or blows up in production.
    Explore the **mathematical taxonomy of forecast metrics** and test them interactively below.
    """)

    # Taxonomy Card
    tax_html = r"""
    <div style="background-color: #f8fafc; border: 2px solid #cbd5e1; border-radius: 8px; padding: 1.1rem; margin-bottom: 1.2rem;">
        <div style="font-weight: 800; color: #0f172a; font-size: 1.05rem; margin-bottom: 0.5rem;">
            📐 The 3 Mathematical Metric Families
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; font-size: 0.85rem; line-height: 1.45;">
            <div style="background: #ffffff; padding: 0.75rem; border-radius: 6px; border-left: 3px solid #2563eb;">
                <strong style="color: #1e40af;">1. Scale-Dependent (MAE, MSE, RMSE)</strong><br>
                • In raw units of $Y$ (e.g. patients, beds).<br>
                • <strong>MAE</strong> targets median; robust to outliers.<br>
                • <strong>RMSE</strong> targets mean; quadratically penalizes large blunders.<br>
                • Cannot compare across series with different scales!
            </div>
            <div style="background: #ffffff; padding: 0.75rem; border-radius: 6px; border-left: 3px solid #ea580c;">
                <strong style="color: #c2410c;">2. Percentage Errors (MAPE, sMAPE)</strong><br>
                • Unitless & dimensionless (e.g. 5.2% error).<br>
                • Instantly understood by executives.<br>
                • <strong>Fatal Flaw:</strong> Divides by zero when $y_t=0$!<br>
                • Asymmetric: penalizes over-predictions more than under-predictions.
            </div>
            <div style="background: #ffffff; padding: 0.75rem; border-radius: 6px; border-left: 3px solid #16a34a;">
                <strong style="color: #15803d;">3. Scaled Benchmarks (MASE)</strong><br>
                • Compares model MAE against in-sample Naive MAE.<br>
                • <strong>MASE &lt; 1.0:</strong> Better than naive.<br>
                • <strong>MASE &gt; 1.0:</strong> Worse than naive.<br>
                • Scale-free & zero-immune (the modern gold standard).
            </div>
        </div>
    </div>
    """
    render_html(tax_html)

    # Interactive Outlier & Zero Injector Experiment
    st.markdown("#### 🧪 Interactive Metric Shock Laboratory: Outlier & Zero Injection")
    st.caption("Adjust the sliders below to inject an isolated outlier shock or zero-demand days, and observe in real-time how each metric responds.")

    clean_series = list(scenario_data["series"][-28:])
    hw_res = forecast_holt_winters(clean_series, period=7, horizon=14)
    y_test_base = clean_series[-14:]
    y_pred_base = hw_res["forecast"]

    c_inj1, c_inj2 = st.columns(2)
    with c_inj1:
        outlier_spike = st.slider(
            "⚡ Inject Extreme Outlier Shock on Day 10 (+ units):",
            min_value=0.0,
            max_value=150.0,
            value=0.0,
            step=10.0,
            key="ts_outlier_inj_slider",
            help="Injects an isolated sudden spike (e.g. +80 patients) on a single day. Watch how RMSE jumps far higher than MAE because squaring heavily penalizes big misses!"
        )
    with c_inj2:
        inject_zero = st.checkbox(
            "🚫 Inject Zero-Demand Day on Day 5 (Tests Division by Zero)",
            value=False,
            key="ts_zero_inj_chk",
            help="Injects a day with exactly 0 demand. Demonstrates why standard MAPE immediately fails (division by zero) while MASE remains completely stable."
        )

    # Mutate test data
    y_test_mod = list(y_test_base)
    if outlier_spike > 0:
        y_test_mod[9] += outlier_spike
    if inject_zero:
        y_test_mod[4] = 0.0

    # Calculate metrics on modified series
    m_res = calculate_forecast_metrics(y_test_mod, y_pred_base, y_train=clean_series[:-14], seasonal_period=7)

    # Display Side-by-Side Comparison with rich tooltips
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric(
            "MAE (Linear Penalty)",
            f"{m_res['mae']:.2f}",
            delta=f"+{m_res['mae'] - 6.5:.2f}" if outlier_spike > 0 else None,
            delta_color="inverse",
            help="Mean Absolute Error: Increases in direct linear proportion to error magnitude. If an outlier of +50 is added across 14 days, MAE increases by exactly 50/14 = +3.57. Robust against isolated noise."
        )
        st.caption("Scales linearly with error.")
    with col_kpi2:
        st.metric(
            "RMSE (Quadratic Penalty)",
            f"{m_res['rmse']:.2f}",
            delta=f"+{m_res['rmse'] - 7.5:.2f}" if outlier_spike > 0 else None,
            delta_color="inverse",
            help="Root Mean Squared Error: Squares errors before averaging. An error of 50 adds 50^2 = 2500 to the sum. RMSE jumps dramatically compared to MAE, highlighting that a major blunder occurred."
        )
        st.caption("Squaring heavily punishes the spike.")
    with col_kpi3:
        if m_res["mape"] is not None:
            st.metric(
                "MAPE (% Error)",
                f"{m_res['mape']:.1f}%",
                help="Mean Absolute Percentage Error: The average percentage discrepancy. Intuitively understood by executives, but vulnerable to zero values."
            )
            st.caption("Standard percentage error.")
        else:
            st.metric(
                "MAPE (% Error)",
                "💥 UNDEFINED",
                help="FATAL: Division by zero occurred! Because actual demand on Day 5 was 0, dividing by 0 explodes to infinity. Standard MAPE is mathematically invalid here."
            )
            st.caption("FATAL: Division by zero!")
    with col_kpi4:
        st.metric(
            "MASE (vs Naive)",
            f"{m_res['mase']:.3f}" if m_res['mase'] else "N/A",
            help="Mean Absolute Scaled Error: Compares your model error against the benchmark error of simply guessing yesterday's value. Scale-free and immune to zeroes."
        )
        st.caption("Scale-free & zero-immune.")

    if m_res["mape_warning"]:
        st.error(m_res["mape_warning"])

    # Dispersion Warning
    if m_res["dispersion_ratio"] >= 1.25:
        st.warning(f"🚨 **Outlier Distortion Alert:** The RMSE / MAE ratio is **{m_res['dispersion_ratio']:.2f}** (well above 1.25). This mathematically proves that a small number of massive outlier errors are dominating your model's RMSE score!")

    # Deep Formula Dissection Expander for ALL 7 Metrics
    with st.expander("📐 All 7 Evaluation Metric Formulas with Plain-English Symbol Dissections", expanded=False):
        st.markdown(r"""
        #### 1. Mean Absolute Error (MAE):
        $$\text{MAE} = \frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|$$
        * **$y_t$:** The actual observed value at time period $t$ (e.g. 150 real patients).
        * **$\hat{y}_t$:** The forecasted value produced by the model (e.g. 140 predicted patients).
        * **$|y_t - \hat{y}_t|$:** The absolute difference (ignoring whether we over-predicted or under-predicted; error is $+10$).
        * **$\sum_{t=1}^n$:** Adds up all absolute errors across all $n$ test periods.
        * **$\frac{1}{n}$:** Divides by the number of test days to get the average error in raw units (e.g. "off by an average of 6.2 patients"). Minimizing MAE mathematically targets the **conditional median**.

        ---

        #### 2. Mean Squared Error (MSE):
        $$\text{MSE} = \frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2$$
        * **$(y_t - \hat{y}_t)^2$:** Multiplies the error by itself. A miss of 2 becomes $2^2 = 4$, while a miss of 20 becomes $20^2 = 400$. This places a quadratic penalty on large blunders.
        * Expressed in **squared units** (e.g. $\text{patients}^2$), which is non-intuitive for clinical directors.

        ---

        #### 3. Root Mean Squared Error (RMSE):
        $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2}$$
        * **$\sqrt{\dots}$:** Takes the square root of the MSE, bringing the quadratic penalty back into original units of demand (e.g. patients).
        * Minimizing RMSE mathematically targets the **conditional mean**. Mathematically, $\text{RMSE} \ge \text{MAE}$ always. If $\text{RMSE} \gg \text{MAE}$, large occasional blunders exist.

        ---

        #### 4. Mean Absolute Percentage Error (MAPE):
        $$\text{MAPE} = \frac{100\%}{n} \sum_{t=1}^n \left| \frac{y_t - \hat{y}_t}{y_t} \right|$$
        * **$\frac{|y_t - \hat{y}_t|}{y_t}$:** Divides the error by the actual demand to get a fraction.
        * **Fatal Flaw 1 (Division by Zero):** If actual demand $y_t = 0$, the denominator is zero and the formula crashes.
        * **Fatal Flaw 2 (Asymmetry):** If actual $y_t = 100$, predicting 0 gives an error of $100\%$. But predicting 300 gives an error of $200\%$. Over-forecasting is penalized infinitely more than under-forecasting, which biases models to artificially predict too low!

        ---

        #### 5. Symmetric Mean Absolute Percentage Error (sMAPE):
        $$\text{sMAPE} = \frac{100\%}{n} \sum_{t=1}^n \frac{|y_t - \hat{y}_t|}{(|y_t| + |\hat{y}_t|) / 2}$$
        * **$(|y_t| + |\hat{y}_t|) / 2$:** Uses the midpoint between actual demand and predicted demand as the denominator.
        * Prevents infinite explosions and bounds percentage errors between $0\%$ and $200\%$.

        ---

        #### 6. Mean Absolute Scaled Error (MASE) — Rob Hyndman's Gold Standard:
        $$\text{MASE} = \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{in-sample Naive}}} = \frac{\frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|}{\frac{1}{T-1} \sum_{t=2}^T |y_t - y_{t-1}|}$$
        * **Numerator:** Your model's average error (MAE) on the unseen future test set.
        * **Denominator:** The average error you would have achieved on the historical training set if you had simply guessed "today will equal yesterday" ($y_t - y_{t-1}$).
        * **Interpretation:**
          * $\text{MASE} < 1.0$: Your model is **superior** to the naive baseline (e.g. 0.70 means 30% fewer errors).
          * $\text{MASE} = 1.0$: Your model is performing **identically to naive guessing**.
          * $\text{MASE} > 1.0$: Your model is **worse** than a naive benchmark (it adds negative value!).
        * **Why it is superior:** Never divides by zero unless the entire training history is a constant flat line; completely scale-free across different departments.

        ---

        #### 7. Mean Directional Accuracy (MDA):
        $$\text{MDA} = \frac{1}{n-1} \sum_{t=2}^n \mathbf{1}_{\text{sign}(y_t - y_{t-1}) == \text{sign}(\hat{y}_t - y_{t-1})} \times 100\%$$
        * **$\text{sign}(y_t - y_{t-1})$:** Checks whether real demand went UP ($+1$), DOWN ($-1$), or stayed flat ($0$).
        * **$\text{sign}(\hat{y}_t - y_{t-1})$:** Checks whether your model predicted an UP or a DOWN move.
        * **$\mathbf{1}_{\dots}$:** An indicator giving 1 if both agreed on the direction, and 0 if the model guessed the wrong direction.
        * Essential for operational shift rosters (knowing whether to roster extra staff up or down).
        """)

    # Deep Pedagogical Breakdown Table
    st.markdown("---")
    st.markdown("#### 📋 When to Choose Which Metric Over Another: Definitive Guide")

    guide_table = r"""
    <div style="overflow-x: auto; margin-top: 0.5rem; margin-bottom: 0.8rem;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.86rem; text-align: left;">
            <thead>
                <tr style="background-color: #f1f5f9; border-bottom: 2px solid #cbd5e1;">
                    <th style="padding: 0.65rem 0.85rem; color: #0f172a; font-weight: 700;">Operational / Research Context</th>
                    <th style="padding: 0.65rem 0.85rem; color: #166534; font-weight: 700;">Recommended Metric</th>
                    <th style="padding: 0.65rem 0.85rem; color: #1e40af; font-weight: 700;">Mathematical Rationale</th>
                    <th style="padding: 0.65rem 0.85rem; color: #991b1b; font-weight: 700;">Metrics to Avoid</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #e2e8f0; background-color: #ffffff;">
                    <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Single series; linear operational costs; noisy outliers present</td>
                    <td style="padding: 0.6rem 0.85rem; font-weight: 700; color: #166534;">MAE</td>
                    <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Linear penalty; robust to extreme isolated anomalies; targets conditional median.</td>
                    <td style="padding: 0.6rem 0.85rem; color: #991b1b;">MSE / RMSE (outliers excessively distort rankings).</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; background-color: #f8fafc;">
                    <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Catastrophic cost for large blunders (e.g. ICU bed shortages, flight seat capacity)</td>
                    <td style="padding: 0.6rem 0.85rem; font-weight: 700; color: #166534;">RMSE</td>
                    <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Quadratically squares misses; being off by 10 beds once is penalised 10x more than being off by 1 bed 10 times.</td>
                    <td style="padding: 0.6rem 0.85rem; color: #991b1b;">MAE (treats ten 1-bed errors identically to one 10-bed disaster).</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; background-color: #ffffff;">
                    <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Executive / Board-level presentations with non-zero continuous demand</td>
                    <td style="padding: 0.6rem 0.85rem; font-weight: 700; color: #166534;">MAPE (or sMAPE)</td>
                    <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Instantly communicated as a percentage error (e.g. "within 4.8% accuracy") without needing unit context.</td>
                    <td style="padding: 0.6rem 0.85rem; color: #991b1b;">Never use on intermittent data containing zero values (explodes to infinity).</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; background-color: #f8fafc;">
                    <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Intermittent / Zero-heavy demand (e.g. rare drug stockouts, spare parts)</td>
                    <td style="padding: 0.6rem 0.85rem; font-weight: 700; color: #166534;">MASE</td>
                    <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Division by zero is mathematically avoided; handles zeroes cleanly; benchmarked to naive.</td>
                    <td style="padding: 0.6rem 0.85rem; color: #991b1b;">MAPE (divides by 0) and RMSE (over-penalizes isolated Poisson spikes).</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; background-color: #ffffff;">
                    <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Comparing accuracy across multiple clinics or departments with different scales</td>
                    <td style="padding: 0.6rem 0.85rem; font-weight: 700; color: #166534;">MASE</td>
                    <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Completely scale-free; enables fair comparison of a small clinic (N=20/day) vs large hospital (N=2000/day).</td>
                    <td style="padding: 0.6rem 0.85rem; color: #991b1b;">MAE and RMSE (scale-dependent; large hospital always looks artificially worse).</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; background-color: #f8fafc;">
                    <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Operational staffing / shift scheduling (roster up vs roster down)</td>
                    <td style="padding: 0.6rem 0.85rem; font-weight: 700; color: #166534;">MDA (alongside RMSE)</td>
                    <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Measures whether the model correctly predicted direction of surge vs drop (drivers of staffing decisions).</td>
                    <td style="padding: 0.6rem 0.85rem; color: #991b1b;">Pure magnitude metrics (can have low MAE but miss every turning point).</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    render_html(guide_table)


# =====================================================================
# 5. Tab 4: Temporal Validation Protocol (Rolling-Origin Cross-Validation)
# =====================================================================

def render_validation_tab(scenario_data: Dict[str, Any]) -> None:
    """Renders the temporal validation protocol demonstrating rolling-origin evaluation."""
    st.subheader("🛡️ Module 4: Temporal Validation & Preventing Data Leakage")
    st.markdown("""
    In standard cross-sectional machine learning, randomized $K$-fold cross-validation is standard practice.
    In time series, **randomly shuffling observations is fatal**: it uses future data (e.g. October) to predict past data (e.g. August).
    """)

    series = scenario_data["series"]
    m = scenario_data["seasonal_period"]

    c_v1, c_v2 = st.columns([1.2, 1.8])
    with c_v1:
        val_model = st.selectbox(
            "Select Model for Rolling-Origin Evaluation:",
            options=["holt_winters", "seasonal_naive", "ses", "lagged_ml"],
            format_func=lambda x: {
                "holt_winters": "Holt-Winters Triple Exponential",
                "seasonal_naive": "Seasonal Naive Benchmark",
                "ses": "Simple Exponential Smoothing",
                "lagged_ml": "Lagged Ridge Regression (ML)"
            }[x],
            key="ts_val_model_select",
            help="Select which algorithm to test across expanding historical time slices."
        )
        val_horizon = st.slider(
            "Evaluation Window Steps (h):",
            min_value=3,
            max_value=14,
            value=7,
            key="ts_val_horizon_slider",
            help="The size of the future test window for each fold (e.g. 7 days)."
        )

    # Run rolling origin evaluation
    try:
        roll_res = rolling_origin_evaluation(
            series,
            model_type=val_model,
            horizon=val_horizon,
            min_train_size=max(28, 4 * m),
            step_size=val_horizon,
            seasonal_period=m
        )
    except Exception as e:
        st.error(f"Rolling validation error: {e}")
        return

    with c_v2:
        ov = roll_res["overall_metrics"]
        st.markdown(f"**Cross-Validation Summary ({roll_res['num_folds']} Temporal Folds across Timeline):**")
        v1, v2, v3 = st.columns(3)
        v1.metric(
            "Overall CV MAE",
            f"{ov['mae']:.2f}",
            help="The average cross-validated Mean Absolute Error aggregated across all expanding chronological folds without lookahead bias."
        )
        v2.metric(
            "Overall CV RMSE",
            f"{ov['rmse']:.2f}",
            help="The cross-validated Root Mean Squared Error across all expanding chronological test windows."
        )
        v3.metric(
            "Overall CV MASE",
            f"{ov['mase']:.3f}" if ov['mase'] else "N/A",
            help="The cross-validated Mean Absolute Scaled Error across all folds. Values below 1.0 verify consistent out-of-sample superiority over naive baselines."
        )

    # Folds Table
    st.markdown("#### Fold-by-Fold Expanding-Window Breakdown")
    fold_rows = []
    for f in roll_res["folds"]:
        fold_rows.append({
            "Fold #": f"Fold {f['fold']}",
            "Historical Training Days": f"Days 1 to {f['train_size']}",
            "Forecast Window (Horizon)": f"Days {f['train_size'] + 1} to {f['train_size'] + f['horizon']}",
            "MAE": f"{f['mae']:.2f}",
            "RMSE": f"{f['rmse']:.2f}",
            "MASE": f"{f['mase']:.3f}" if f['mase'] else "N/A",
            "Directional Accuracy": f"{f['mda']:.1f}%"
        })

    st.dataframe(fold_rows, use_container_width=True)

    with st.expander("📖 Why Expanding-Window Evaluation Mirrors Real Deployment", expanded=False):
        st.markdown(r"""
        * **Real-World Reality:** In hospital operational forecasting, on Sunday night you have all historical data up to Sunday. You forecast Monday through Sunday. Next Sunday, you add that week's new data to your training history and forecast the subsequent week.
        * **Expanding Window:** Strictly simulates this process by training on $[1 \dots T]$, testing on $[T+1 \dots T+h]$, then expanding training to $[1 \dots T+h]$ and testing on $[T+h+1 \dots T+2h]$.
        * **Zero Leakage:** At no point does the model observe future residuals or forward information.
        """)

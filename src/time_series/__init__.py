"""Time Series Analysis, Decomposition, Forecasting, and Metrics Module.

Provides comprehensive educational and analytical tools for:
- Time series anatomy, stationarity, autocorrelation (ACF/PACF), and ADF unit root testing.
- Classical additive and multiplicative seasonal decomposition (Trend, Seasonality, Residuals).
- Hierarchical forecasting models: Naive, Seasonal Naive, Drift, SES, Holt's Linear (Damped), Holt-Winters, Lagged Regression, and Croston's Intermittent Demand.
- Comprehensive forecast evaluation metrics: MAE, MSE, RMSE, MAPE, sMAPE, MASE, and MDA.
- Rolling-origin expanding-window cross-validation enforcing zero temporal leakage.
"""

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

__all__ = [
    "generate_ed_demand_series",
    "generate_seasonal_flu_series",
    "generate_ambulance_callout_series",
    "generate_intermittent_stockout_series",
    "decompose_time_series",
    "calculate_acf",
    "calculate_pacf",
    "calculate_adf_test",
    "difference_series",
    "forecast_naive",
    "forecast_seasonal_naive",
    "forecast_drift",
    "forecast_simple_exponential_smoothing",
    "forecast_holt_linear",
    "forecast_holt_winters",
    "forecast_lagged_regression",
    "forecast_croston",
    "calculate_forecast_metrics",
    "rolling_origin_evaluation",
    "TIME_SERIES_ALGORITHMS",
    "get_algorithm_profile",
    "generate_time_series_python_code",
    "generate_time_series_excel_guide",
    "generate_time_series_csv",
]

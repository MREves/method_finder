"""Shared helper functions and formatters for Method Finder."""

from typing import Union, Tuple, Optional


def format_p_value(p: float) -> str:
    """Formats a p-value for scientific and clinical display.

    Args:
        p: Float p-value between 0 and 1.

    Returns:
        Formatted string (e.g., '< 0.001' or '0.042').
    """
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


def format_ci(estimate: float, lower: float, upper: float, decimals: int = 2) -> str:
    """Formats an estimate and its 95% confidence interval.

    Args:
        estimate: Point estimate (e.g. Mean, Odds Ratio, Risk Ratio).
        lower: Lower bound of confidence interval.
        upper: Upper bound of confidence interval.
        decimals: Number of decimal places to format.

    Returns:
        Formatted string (e.g. '1.45 (95% CI: 1.10 - 1.92)').
    """
    fmt = f"{{:.{decimals}f}}"
    est_str = fmt.format(estimate)
    low_str = fmt.format(lower)
    upp_str = fmt.format(upper)
    return f"{est_str} (95% CI: {low_str} - {upp_str})"


def get_significance_badge(p: float, alpha: float = 0.05) -> str:
    """Returns a formatted badge indicating statistical significance at alpha."""
    if p < alpha:
        return "✅ Statistically Significant (p < 0.05)"
    return "⚪ Not Statistically Significant (p ≥ 0.05)"
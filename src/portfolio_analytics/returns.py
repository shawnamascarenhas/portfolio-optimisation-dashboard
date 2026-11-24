"""Return transformations and annualised portfolio-model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class ReturnCalculationError(ValueError):
    """Raised when prices or returns cannot support a valid calculation."""


@dataclass(frozen=True)
class ReturnAnalysis:
    """Complete set of return outputs used by later analytics."""

    simple_returns: pd.DataFrame
    log_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    annualised_expected_returns: pd.Series
    annualised_covariance: pd.DataFrame
    periods_per_year: int


def analyse_returns(prices: pd.DataFrame, periods_per_year: int) -> ReturnAnalysis:
    """Build all return series and annualised estimates from aligned prices."""
    clean_prices = validate_prices(prices)
    periods = validate_periods_per_year(periods_per_year)
    simple = calculate_simple_returns(clean_prices)
    log = calculate_log_returns(clean_prices)

    return ReturnAnalysis(
        simple_returns=simple,
        log_returns=log,
        cumulative_returns=calculate_cumulative_returns(simple),
        annualised_expected_returns=calculate_annualised_expected_returns(
            simple, periods
        ),
        annualised_covariance=calculate_annualised_covariance(simple, periods),
        periods_per_year=periods,
    )


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric price copy after enforcing the analytics contract."""
    if not isinstance(prices, pd.DataFrame):
        raise ReturnCalculationError("Prices must be provided as a DataFrame.")
    if prices.empty or prices.shape[1] == 0:
        raise ReturnCalculationError("Prices cannot be empty.")
    if len(prices) < 2:
        raise ReturnCalculationError("At least two price observations are required.")
    if prices.columns.has_duplicates:
        raise ReturnCalculationError("Price columns must be unique.")
    if prices.index.has_duplicates:
        raise ReturnCalculationError("Price dates must be unique.")
    if not prices.index.is_monotonic_increasing:
        raise ReturnCalculationError("Price dates must be sorted in ascending order.")

    numeric = prices.apply(pd.to_numeric, errors="coerce").astype(float)
    values = numeric.to_numpy()
    if not np.isfinite(values).all():
        raise ReturnCalculationError(
            "Prices must not contain missing or infinite values."
        )
    if (values <= 0).any():
        raise ReturnCalculationError("Prices must be strictly positive.")
    return numeric


def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate period-over-period percentage returns."""
    clean_prices = validate_prices(prices)
    returns = clean_prices.pct_change(fill_method=None).iloc[1:]
    returns.index.name = clean_prices.index.name
    return _validate_returns(returns)


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate continuously compounded period returns."""
    clean_prices = validate_prices(prices)
    returns = np.log(clean_prices / clean_prices.shift(1)).iloc[1:]
    returns.index.name = clean_prices.index.name
    return _validate_returns(returns)


def calculate_cumulative_returns(simple_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative growth above the initial value."""
    returns = _validate_returns(simple_returns)
    cumulative = (1.0 + returns).cumprod() - 1.0
    return cumulative.astype(float)


def calculate_annualised_expected_returns(
    simple_returns: pd.DataFrame, periods_per_year: int
) -> pd.Series:
    """Annualise arithmetic mean returns for mean-variance optimisation."""
    returns = _validate_returns(simple_returns)
    periods = validate_periods_per_year(periods_per_year)
    annualised = returns.mean(axis=0) * periods
    annualised.name = "annualised_expected_return"
    return annualised.astype(float)


def calculate_annualised_covariance(
    simple_returns: pd.DataFrame, periods_per_year: int
) -> pd.DataFrame:
    """Annualise the sample covariance matrix of simple returns."""
    returns = _validate_returns(simple_returns, minimum_rows=2)
    periods = validate_periods_per_year(periods_per_year)
    covariance = returns.cov() * periods
    if not np.isfinite(covariance.to_numpy()).all():
        raise ReturnCalculationError("The covariance matrix contains invalid values.")
    return covariance.astype(float)


def validate_periods_per_year(periods_per_year: int) -> int:
    """Validate the annualisation factor supplied by the data layer."""
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int):
        raise ReturnCalculationError("Periods per year must be a positive integer.")
    if periods_per_year <= 0:
        raise ReturnCalculationError("Periods per year must be a positive integer.")
    return periods_per_year


def _validate_returns(
    returns: pd.DataFrame, *, minimum_rows: int = 1
) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise ReturnCalculationError("Returns must be provided as a DataFrame.")
    if returns.shape[1] == 0 or len(returns) < minimum_rows:
        raise ReturnCalculationError(
            f"At least {minimum_rows} complete return observations are required."
        )
    if returns.columns.has_duplicates:
        raise ReturnCalculationError("Return columns must be unique.")
    numeric = returns.apply(pd.to_numeric, errors="coerce").astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ReturnCalculationError(
            "Returns must not contain missing or infinite values."
        )
    if (numeric.to_numpy() <= -1.0).any():
        raise ReturnCalculationError(
            "Simple returns cannot be less than or equal to -100%."
        )
    return numeric

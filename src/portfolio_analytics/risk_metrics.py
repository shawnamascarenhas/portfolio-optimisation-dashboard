"""Risk and risk-adjusted performance analytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_analytics.returns import (
    calculate_annualised_expected_returns,
    validate_periods_per_year,
)


class RiskCalculationError(ValueError):
    """Raised when risk metrics cannot be calculated from the supplied inputs."""


@dataclass(frozen=True)
class RiskAnalysis:
    """Asset risk metrics, correlations, and drawdown histories."""

    metrics: pd.DataFrame
    correlations: pd.DataFrame
    drawdowns: pd.DataFrame
    benchmark: str
    confidence_level: float
    periods_per_year: int


def analyse_risk(
    simple_returns: pd.DataFrame,
    benchmark: str,
    periods_per_year: int,
    *,
    risk_free_rate: float = 0.04,
    confidence_level: float = 0.95,
) -> RiskAnalysis:
    """Calculate the MVP asset-level risk analytics in one operation."""
    returns = _validate_return_table(simple_returns)
    periods = validate_periods_per_year(periods_per_year)
    risk_free = _validate_risk_free_rate(risk_free_rate)
    confidence = _validate_confidence_level(confidence_level)
    benchmark = benchmark.strip().upper() if isinstance(benchmark, str) else benchmark
    if benchmark not in returns.columns:
        raise RiskCalculationError(
            f"Benchmark {benchmark!r} is not present in the return table."
        )

    benchmark_variance = returns[benchmark].var(ddof=1)
    if not np.isfinite(benchmark_variance) or benchmark_variance <= 0:
        raise RiskCalculationError("Benchmark returns must have positive variance.")

    annualised_return = calculate_annualised_expected_returns(returns, periods)
    annualised_volatility = returns.std(ddof=1) * np.sqrt(periods)
    sharpe = _safe_ratio(annualised_return - risk_free, annualised_volatility)
    downside = calculate_downside_deviation(returns, periods, risk_free)
    sortino = _safe_ratio(annualised_return - risk_free, downside)
    drawdowns = calculate_drawdowns(returns)
    maximum_drawdown = drawdowns.min(axis=0)
    value_at_risk = calculate_historical_var(returns, confidence)
    expected_shortfall = calculate_historical_expected_shortfall(returns, confidence)
    beta = calculate_beta(returns, benchmark)

    metrics = pd.DataFrame(
        {
            "annualised_return": annualised_return,
            "annualised_volatility": annualised_volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "maximum_drawdown": maximum_drawdown,
            "historical_var": value_at_risk,
            "historical_expected_shortfall": expected_shortfall,
            "beta": beta,
        }
    )
    metrics.index.name = "asset"

    return RiskAnalysis(
        metrics=metrics,
        correlations=returns.corr(),
        drawdowns=drawdowns,
        benchmark=benchmark,
        confidence_level=confidence,
        periods_per_year=periods,
    )


def calculate_drawdowns(simple_returns: pd.DataFrame) -> pd.DataFrame:
    """Return each period's decline from its prior wealth peak."""
    returns = _validate_return_table(simple_returns)
    initial = pd.DataFrame(
        [np.ones(returns.shape[1])],
        columns=returns.columns,
        index=[_initial_index_label(returns.index)],
    )
    wealth = pd.concat([initial, (1.0 + returns).cumprod()])
    drawdowns = wealth / wealth.cummax() - 1.0
    return drawdowns.iloc[1:].astype(float)


def calculate_maximum_drawdown(simple_returns: pd.DataFrame) -> pd.Series:
    """Return the most negative peak-to-trough drawdown for each series."""
    maximum = calculate_drawdowns(simple_returns).min(axis=0)
    maximum.name = "maximum_drawdown"
    return maximum


def calculate_downside_deviation(
    simple_returns: pd.DataFrame,
    periods_per_year: int,
    annual_target_rate: float = 0.0,
) -> pd.Series:
    """Annualise lower partial deviation relative to a compounded target rate."""
    returns = _validate_return_table(simple_returns)
    periods = validate_periods_per_year(periods_per_year)
    target = _validate_risk_free_rate(annual_target_rate)
    period_target = (1.0 + target) ** (1.0 / periods) - 1.0
    downside = np.minimum(returns - period_target, 0.0)
    deviation = np.sqrt((downside**2).mean(axis=0)) * np.sqrt(periods)
    deviation.name = "annualised_downside_deviation"
    return deviation.astype(float)


def calculate_historical_var(
    simple_returns: pd.DataFrame, confidence_level: float = 0.95
) -> pd.Series:
    """Return positive historical one-period VaR as a fraction of capital."""
    returns = _validate_return_table(simple_returns)
    confidence = _validate_confidence_level(confidence_level)
    quantile = returns.quantile(1.0 - confidence, interpolation="linear")
    value_at_risk = (-quantile).clip(lower=0.0)
    value_at_risk.name = "historical_var"
    return value_at_risk.astype(float)


def calculate_historical_expected_shortfall(
    simple_returns: pd.DataFrame, confidence_level: float = 0.95
) -> pd.Series:
    """Return positive mean loss at or beyond each series' historical VaR cutoff."""
    returns = _validate_return_table(simple_returns)
    confidence = _validate_confidence_level(confidence_level)
    cutoff = returns.quantile(1.0 - confidence, interpolation="linear")
    expected_shortfall = pd.Series(index=returns.columns, dtype=float)
    for column in returns:
        tail = returns.loc[returns[column] <= cutoff[column], column]
        expected_shortfall[column] = max(0.0, -float(tail.mean()))
    expected_shortfall.name = "historical_expected_shortfall"
    return expected_shortfall


def calculate_beta(simple_returns: pd.DataFrame, benchmark: str) -> pd.Series:
    """Calculate each series' covariance beta relative to the benchmark."""
    returns = _validate_return_table(simple_returns)
    if benchmark not in returns.columns:
        raise RiskCalculationError(
            f"Benchmark {benchmark!r} is not present in the return table."
        )
    benchmark_variance = returns[benchmark].var(ddof=1)
    if not np.isfinite(benchmark_variance) or benchmark_variance <= 0:
        raise RiskCalculationError("Benchmark returns must have positive variance.")
    beta = returns.cov()[benchmark] / benchmark_variance
    beta.name = "beta"
    return beta.astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0.0, np.nan)
    return result.where(np.isfinite(result), np.nan).astype(float)


def _validate_return_table(simple_returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(simple_returns, pd.DataFrame):
        raise RiskCalculationError("Returns must be provided as a DataFrame.")
    if len(simple_returns) < 2 or simple_returns.shape[1] == 0:
        raise RiskCalculationError(
            "At least two complete observations and one return series are required."
        )
    if simple_returns.columns.has_duplicates:
        raise RiskCalculationError("Return columns must be unique.")
    numeric = simple_returns.apply(pd.to_numeric, errors="coerce").astype(float)
    values = numeric.to_numpy()
    if not np.isfinite(values).all():
        raise RiskCalculationError(
            "Returns must not contain missing or infinite values."
        )
    if (values <= -1.0).any():
        raise RiskCalculationError(
            "Simple returns cannot be less than or equal to -100%."
        )
    return numeric


def _validate_risk_free_rate(rate: float) -> float:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise RiskCalculationError("The annual rate must be numeric.")
    rate = float(rate)
    if not np.isfinite(rate) or rate <= -1.0:
        raise RiskCalculationError(
            "The annual rate must be finite and greater than -1."
        )
    return rate


def _validate_confidence_level(confidence_level: float) -> float:
    if isinstance(confidence_level, bool) or not isinstance(
        confidence_level, (int, float)
    ):
        raise RiskCalculationError("Confidence level must be numeric.")
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise RiskCalculationError("Confidence level must be between 0 and 1.")
    return confidence


def _initial_index_label(index: pd.Index):
    if isinstance(index, pd.DatetimeIndex):
        return index[0] - pd.Timedelta(nanoseconds=1)
    return "__initial_wealth__"

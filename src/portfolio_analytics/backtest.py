"""Chronological in-sample portfolio construction and out-of-sample evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_analytics.portfolio_comparison import (
    PortfolioComparisonError,
    compare_portfolios,
)
from portfolio_analytics.returns import (
    ReturnCalculationError,
    calculate_annualised_covariance,
    calculate_annualised_expected_returns,
    calculate_simple_returns,
    validate_periods_per_year,
    validate_prices,
)
from portfolio_analytics.risk_metrics import (
    RiskCalculationError,
    calculate_historical_expected_shortfall,
    calculate_historical_var,
    calculate_maximum_drawdown,
)


class BacktestError(ValueError):
    """Raised when a backtest request or calculation is invalid."""


@dataclass(frozen=True)
class BacktestResult:
    """Portfolio construction inputs and realised out-of-sample results."""

    in_sample_returns: pd.DataFrame
    out_of_sample_asset_returns: pd.DataFrame
    strategy_returns: pd.DataFrame
    portfolio_values: pd.DataFrame
    weights: pd.DataFrame
    in_sample_statistics: pd.DataFrame
    out_of_sample_statistics: pd.DataFrame
    split_date: pd.Timestamp
    benchmark_label: str
    initial_investment: float
    periods_per_year: int


def run_backtest(
    prices: pd.DataFrame,
    assets: list[str] | tuple[str, ...],
    benchmark: str,
    periods_per_year: int,
    *,
    split_ratio: float = 0.70,
    risk_free_rate: float = 0.04,
    maximum_weight: float = 1.0,
    confidence_level: float = 0.95,
    initial_investment: float = 100_000.0,
    custom_weights: pd.Series | Mapping[str, float] | None = None,
) -> BacktestResult:
    """Fit strategies in sample and evaluate them on later observations only."""
    clean_prices = _validate_backtest_prices(prices, assets, benchmark)
    periods = _translate_error(validate_periods_per_year, periods_per_year)
    ratio = _validate_split_ratio(split_ratio)
    initial_value = _validate_initial_investment(initial_investment)
    confidence = _validate_confidence_level(confidence_level)

    try:
        all_returns = calculate_simple_returns(clean_prices)
    except ReturnCalculationError as exc:
        raise BacktestError(str(exc)) from exc
    split_position = int(np.floor(len(all_returns) * ratio))
    minimum_in_sample = max(3, len(assets) + 1)
    if (
        split_position < minimum_in_sample
        or len(all_returns) - split_position < 2
    ):
        raise BacktestError(
            "The selected history must provide enough observations for covariance "
            "estimation in the in-sample period and at least two observations in "
            "the out-of-sample period."
        )

    in_sample = all_returns.iloc[:split_position].copy()
    out_of_sample = all_returns.iloc[split_position:].copy()
    asset_labels = list(assets)
    in_sample_assets = in_sample.loc[:, asset_labels]
    out_of_sample_assets = out_of_sample.loc[:, asset_labels]

    expected_returns = calculate_annualised_expected_returns(
        in_sample_assets, periods
    )
    covariance = calculate_annualised_covariance(in_sample_assets, periods)
    try:
        comparison = compare_portfolios(
            expected_returns,
            covariance,
            risk_free_rate=risk_free_rate,
            maximum_weight=maximum_weight,
            custom_weights=custom_weights,
        )
    except PortfolioComparisonError as exc:
        raise BacktestError(str(exc)) from exc

    # A constant-weight return series represents an idealised rebalance at each
    # observation. Transaction costs are intentionally excluded from the MVP.
    strategy_returns = pd.DataFrame(
        {
            strategy: out_of_sample_assets @ weights
            for strategy, weights in comparison.weights.iterrows()
        },
        index=out_of_sample.index,
    )
    benchmark_label = f"Benchmark: {benchmark}"
    strategy_returns[benchmark_label] = out_of_sample[benchmark]
    strategy_returns.index.name = out_of_sample.index.name

    portfolio_values = _calculate_portfolio_values(
        strategy_returns, initial_value
    )
    out_of_sample_statistics = _calculate_realised_statistics(
        strategy_returns,
        portfolio_values,
        periods,
        float(risk_free_rate),
        confidence,
    )

    return BacktestResult(
        in_sample_returns=in_sample,
        out_of_sample_asset_returns=out_of_sample,
        strategy_returns=strategy_returns,
        portfolio_values=portfolio_values,
        weights=comparison.weights,
        in_sample_statistics=comparison.statistics,
        out_of_sample_statistics=out_of_sample_statistics,
        split_date=pd.Timestamp(out_of_sample.index[0]),
        benchmark_label=benchmark_label,
        initial_investment=initial_value,
        periods_per_year=periods,
    )


def _calculate_portfolio_values(
    strategy_returns: pd.DataFrame, initial_investment: float
) -> pd.DataFrame:
    initial_date = _initial_index_label(strategy_returns.index)
    initial_row = pd.DataFrame(
        [np.full(strategy_returns.shape[1], initial_investment)],
        columns=strategy_returns.columns,
        index=[initial_date],
    )
    growth = (1.0 + strategy_returns).cumprod() * initial_investment
    values = pd.concat([initial_row, growth])
    values.index.name = strategy_returns.index.name
    return values.astype(float)


def _calculate_realised_statistics(
    strategy_returns: pd.DataFrame,
    portfolio_values: pd.DataFrame,
    periods_per_year: int,
    risk_free_rate: float,
    confidence_level: float,
) -> pd.DataFrame:
    annualised_return = calculate_annualised_expected_returns(
        strategy_returns, periods_per_year
    )
    volatility = strategy_returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (annualised_return - risk_free_rate) / volatility.replace(0.0, np.nan)
    maximum_drawdown = calculate_maximum_drawdown(strategy_returns)
    value_at_risk = calculate_historical_var(strategy_returns, confidence_level)
    expected_shortfall = calculate_historical_expected_shortfall(
        strategy_returns, confidence_level
    )
    final_value = portfolio_values.iloc[-1]
    cumulative_return = final_value / portfolio_values.iloc[0] - 1.0
    years = len(strategy_returns) / periods_per_year
    cagr = (final_value / portfolio_values.iloc[0]) ** (1.0 / years) - 1.0

    statistics = pd.DataFrame(
        {
            "initial_value": portfolio_values.iloc[0],
            "final_value": final_value,
            "cumulative_return": cumulative_return,
            "cagr": cagr,
            "annualised_return": annualised_return,
            "annualised_volatility": volatility,
            "sharpe_ratio": sharpe,
            "maximum_drawdown": maximum_drawdown,
            "historical_var": value_at_risk,
            "historical_expected_shortfall": expected_shortfall,
        }
    )
    statistics.index.name = "strategy"
    return statistics.astype(float)


def _validate_backtest_prices(
    prices: pd.DataFrame, assets: list[str] | tuple[str, ...], benchmark: str
) -> pd.DataFrame:
    if not isinstance(assets, (list, tuple)) or len(assets) < 2:
        raise BacktestError("At least two asset labels are required.")
    if len(set(assets)) != len(assets):
        raise BacktestError("Asset labels must be unique.")
    required = list(dict.fromkeys([*assets, benchmark]))
    missing = [label for label in required if label not in prices.columns]
    if missing:
        raise BacktestError(f"Price data is missing: {', '.join(missing)}.")
    try:
        return validate_prices(prices.loc[:, required])
    except (ReturnCalculationError, KeyError) as exc:
        raise BacktestError(str(exc)) from exc


def _validate_split_ratio(split_ratio: float) -> float:
    if isinstance(split_ratio, bool) or not isinstance(split_ratio, (int, float)):
        raise BacktestError("Split ratio must be numeric.")
    ratio = float(split_ratio)
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise BacktestError("Split ratio must be between zero and one.")
    return ratio


def _validate_initial_investment(initial_investment: float) -> float:
    if isinstance(initial_investment, bool) or not isinstance(
        initial_investment, (int, float)
    ):
        raise BacktestError("Initial investment must be numeric.")
    value = float(initial_investment)
    if not np.isfinite(value) or value <= 0:
        raise BacktestError("Initial investment must be positive and finite.")
    return value


def _validate_confidence_level(confidence_level: float) -> float:
    if isinstance(confidence_level, bool) or not isinstance(
        confidence_level, (int, float)
    ):
        raise BacktestError("Confidence level must be numeric.")
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise BacktestError("Confidence level must be between zero and one.")
    return confidence


def _translate_error(function, *args):
    try:
        return function(*args)
    except (ReturnCalculationError, RiskCalculationError) as exc:
        raise BacktestError(str(exc)) from exc


def _initial_index_label(index: pd.Index):
    if isinstance(index, pd.DatetimeIndex):
        return index[0] - pd.Timedelta(nanoseconds=1)
    return "__initial_value__"

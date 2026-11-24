"""Tests for chronological out-of-sample portfolio backtesting."""

import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.backtest import BacktestError, run_backtest
from portfolio_analytics.portfolio_comparison import (
    CUSTOM,
    EQUAL_WEIGHT,
    MAXIMUM_SHARPE,
    MINIMUM_VARIANCE,
)

ASSETS = ["A", "B", "C"]
BENCHMARK = "MARKET"


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    periods = 121
    asset_returns = rng.multivariate_normal(
        mean=[0.0003, 0.0005, 0.0007],
        cov=[
            [0.00010, 0.00002, 0.00001],
            [0.00002, 0.00016, 0.00003],
            [0.00001, 0.00003, 0.00025],
        ],
        size=periods - 1,
    )
    market_returns = 0.5 * asset_returns[:, 0] + 0.5 * asset_returns[:, 1]
    all_returns = np.column_stack([asset_returns, market_returns])
    values = np.vstack([np.ones(4), np.cumprod(1.0 + all_returns, axis=0)]) * 100
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2024-01-02", periods=periods),
        columns=[*ASSETS, BENCHMARK],
    )


def test_backtest_uses_chronological_70_30_split(prices: pd.DataFrame) -> None:
    result = run_backtest(
        prices,
        ASSETS,
        BENCHMARK,
        252,
        maximum_weight=0.60,
        initial_investment=100_000,
    )

    assert len(result.in_sample_returns) == 84
    assert len(result.out_of_sample_asset_returns) == 36
    assert result.in_sample_returns.index.max() < result.split_date
    assert result.out_of_sample_asset_returns.index.min() == result.split_date
    assert result.portfolio_values.iloc[0].eq(100_000).all()


def test_backtest_reports_all_required_strategies_and_metrics(
    prices: pd.DataFrame,
) -> None:
    result = run_backtest(
        prices,
        ASSETS,
        BENCHMARK,
        252,
        maximum_weight=0.60,
        custom_weights={"A": 0.40, "B": 0.35, "C": 0.25},
    )

    assert list(result.weights.index) == [
        EQUAL_WEIGHT,
        MINIMUM_VARIANCE,
        MAXIMUM_SHARPE,
        CUSTOM,
    ]
    assert list(result.strategy_returns.columns) == [
        EQUAL_WEIGHT,
        MINIMUM_VARIANCE,
        MAXIMUM_SHARPE,
        CUSTOM,
        "Benchmark: MARKET",
    ]
    assert list(result.out_of_sample_statistics.columns) == [
        "initial_value",
        "final_value",
        "cumulative_return",
        "cagr",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "historical_var",
        "historical_expected_shortfall",
    ]


def test_portfolio_values_compound_strategy_returns(prices: pd.DataFrame) -> None:
    result = run_backtest(prices, ASSETS, BENCHMARK, 252, maximum_weight=0.60)
    expected_final = 100_000 * (1.0 + result.strategy_returns).prod(axis=0)

    pd.testing.assert_series_equal(
        result.portfolio_values.iloc[-1], expected_final, check_names=False
    )
    expected_cumulative = expected_final / 100_000 - 1.0
    pd.testing.assert_series_equal(
        result.out_of_sample_statistics["cumulative_return"],
        expected_cumulative.rename_axis("strategy").rename("cumulative_return"),
    )


def test_future_data_cannot_change_fitted_weights(prices: pd.DataFrame) -> None:
    baseline = run_backtest(prices, ASSETS, BENCHMARK, 252, maximum_weight=0.60)
    changed = prices.copy()
    future_dates = baseline.out_of_sample_asset_returns.index
    changing_multiplier = np.linspace(1.0, 2.0, len(future_dates))
    changed.loc[future_dates, "C"] *= changing_multiplier
    alternative = run_backtest(changed, ASSETS, BENCHMARK, 252, maximum_weight=0.60)

    pd.testing.assert_frame_equal(baseline.weights, alternative.weights)
    assert not baseline.portfolio_values.equals(alternative.portfolio_values)


def test_realised_metrics_are_internally_consistent(prices: pd.DataFrame) -> None:
    result = run_backtest(
        prices, ASSETS, BENCHMARK, 252, maximum_weight=0.60, risk_free_rate=0.03
    )
    stats = result.out_of_sample_statistics
    years = len(result.strategy_returns) / 252
    expected_cagr = (
        stats["final_value"] / stats["initial_value"]
    ) ** (1 / years) - 1

    pd.testing.assert_series_equal(stats["cagr"], expected_cagr, check_names=False)
    assert (stats["historical_var"] >= 0).all()
    assert (stats["historical_expected_shortfall"] >= stats["historical_var"]).all()
    assert (stats["maximum_drawdown"] <= 0).all()


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.1, True])
def test_invalid_split_ratio_is_rejected(prices: pd.DataFrame, ratio) -> None:
    with pytest.raises(BacktestError, match="Split ratio"):
        run_backtest(prices, ASSETS, BENCHMARK, 252, split_ratio=ratio)


@pytest.mark.parametrize("value", [0, -1, np.inf, True])
def test_invalid_initial_investment_is_rejected(prices: pd.DataFrame, value) -> None:
    with pytest.raises(BacktestError, match="Initial investment"):
        run_backtest(prices, ASSETS, BENCHMARK, 252, initial_investment=value)


def test_missing_assets_and_short_history_are_rejected(prices: pd.DataFrame) -> None:
    with pytest.raises(BacktestError, match="missing"):
        run_backtest(prices, ["A", "B", "MISSING"], BENCHMARK, 252)
    with pytest.raises(BacktestError, match="enough observations"):
        run_backtest(prices.iloc[:5], ASSETS, BENCHMARK, 252)


def test_invalid_custom_weights_are_reported_as_backtest_error(
    prices: pd.DataFrame,
) -> None:
    with pytest.raises(BacktestError, match="sum to 100%"):
        run_backtest(
            prices,
            ASSETS,
            BENCHMARK,
            252,
            custom_weights={"A": 0.2, "B": 0.2, "C": 0.2},
        )

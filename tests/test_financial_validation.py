"""Independent financial-logic validation using known analytical results."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_analytics import (
    analyse_returns,
    analyse_risk,
    create_request,
    load_market_data,
    optimise_maximum_sharpe,
    optimise_minimum_variance,
    run_backtest,
)


def test_two_asset_minimum_variance_matches_closed_form_solution() -> None:
    """Validate SLSQP against the unconstrained two-asset analytical formula."""
    expected_returns = pd.Series({"A": 0.08, "B": 0.12})
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    result = optimise_minimum_variance(expected_returns, covariance)
    expected_weight_a = (0.09 - 0.01) / (0.04 + 0.09 - 2 * 0.01)

    assert result.weights["A"] == pytest.approx(expected_weight_a, abs=1e-7)
    assert result.weights["B"] == pytest.approx(1 - expected_weight_a, abs=1e-7)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-10)


def test_maximum_sharpe_matches_matrix_closed_form_solution() -> None:
    """Validate the long-only solver where the analytical tangency weights fit."""
    expected_returns = pd.Series({"A": 0.08, "B": 0.12, "C": 0.15})
    covariance = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]),
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    risk_free = 0.03
    raw = np.linalg.solve(
        covariance.to_numpy(), expected_returns.to_numpy() - risk_free
    )
    analytical = raw / raw.sum()
    result = optimise_maximum_sharpe(
        expected_returns, covariance, risk_free_rate=risk_free
    )

    np.testing.assert_allclose(result.weights.to_numpy(), analytical, atol=1e-6)


def test_risk_metrics_match_hand_calculated_series() -> None:
    returns = pd.DataFrame(
        {
            "ASSET": [0.10, -0.20, 0.05, -0.10],
            "MARKET": [0.05, -0.10, 0.02, -0.04],
        },
        index=pd.date_range("2025-01-31", periods=4, freq="ME"),
    )
    result = analyse_risk(
        returns,
        "MARKET",
        periods_per_year=12,
        risk_free_rate=0.0,
        confidence_level=0.75,
    )
    asset = returns["ASSET"]
    expected_volatility = asset.std(ddof=1) * np.sqrt(12)
    expected_beta = asset.cov(returns["MARKET"]) / returns["MARKET"].var(ddof=1)
    wealth = (1 + asset).cumprod()
    running_peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    expected_drawdown = np.min(wealth.to_numpy() / running_peak - 1.0)

    assert result.metrics.loc["ASSET", "annualised_return"] == pytest.approx(
        asset.mean() * 12
    )
    assert result.metrics.loc["ASSET", "annualised_volatility"] == pytest.approx(
        expected_volatility
    )
    assert result.metrics.loc["ASSET", "beta"] == pytest.approx(expected_beta)
    assert result.metrics.loc["ASSET", "maximum_drawdown"] == pytest.approx(
        expected_drawdown
    )
    expected_shortfall = result.metrics.loc[
        "ASSET", "historical_expected_shortfall"
    ]
    assert expected_shortfall == pytest.approx(0.20)


def test_end_to_end_pipeline_preserves_labels_dates_and_constraints(tmp_path) -> None:
    """Exercise data, returns, risk, optimisation, and backtest boundaries."""
    rng = np.random.default_rng(7)
    symbols = ["AAA", "BBB", "CCC", "SPY"]
    dates = pd.bdate_range("2023-01-02", periods=180)
    innovations = rng.normal(
        loc=[0.0003, 0.0004, 0.0005, 0.00035],
        scale=[0.008, 0.010, 0.012, 0.009],
        size=(len(dates) - 1, len(symbols)),
    )
    prices = np.vstack(
        [np.full(len(symbols), 100.0), 100 * np.cumprod(1 + innovations, axis=0)]
    )
    adjusted = pd.DataFrame(prices, index=dates, columns=symbols)

    request = create_request(
        symbols[:3], "SPY", date(2023, 1, 1), date(2023, 12, 31)
    )
    market = load_market_data(
        request,
        cache_dir=tmp_path,
        downloader=lambda *args, **kwargs: adjusted,
    )
    returns = analyse_returns(market.prices, market.periods_per_year)
    risk = analyse_risk(returns.simple_returns, "SPY", market.periods_per_year)
    backtest = run_backtest(
        market.prices,
        list(market.assets),
        market.benchmark,
        market.periods_per_year,
        maximum_weight=0.50,
    )

    assert list(market.prices.columns) == symbols
    assert returns.simple_returns.index.equals(market.prices.index[1:])
    assert risk.metrics.index.tolist() == symbols
    np.testing.assert_allclose(backtest.weights.sum(axis=1), 1.0, atol=1e-7)
    assert (backtest.weights >= -1e-8).all().all()
    assert (backtest.weights <= 0.50 + 1e-8).all().all()
    assert backtest.in_sample_returns.index.max() < backtest.split_date
    assert backtest.strategy_returns.index.min() == backtest.split_date
    assert np.isfinite(backtest.out_of_sample_statistics.to_numpy()).all()


def test_price_return_round_trip_identity() -> None:
    prices = pd.DataFrame(
        {"A": [100.0, 105.0, 102.9, 110.103]},
        index=pd.date_range("2025-01-01", periods=4),
    )
    result = analyse_returns(prices, periods_per_year=252)
    reconstructed = prices.iloc[0] * (1 + result.simple_returns).cumprod()

    np.testing.assert_allclose(reconstructed.to_numpy(), prices.iloc[1:].to_numpy())

"""Tests for asset-level risk and risk-adjusted performance metrics."""

import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.risk_metrics import (
    RiskCalculationError,
    analyse_risk,
    calculate_beta,
    calculate_downside_deviation,
    calculate_drawdowns,
    calculate_historical_expected_shortfall,
    calculate_historical_var,
    calculate_maximum_drawdown,
)


@pytest.fixture
def returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ALPHA": [0.10, -0.20, 0.05, -0.10, 0.15],
            "DEFENSIVE": [0.02, -0.01, 0.01, -0.02, 0.02],
            "MARKET": [0.05, -0.10, 0.02, -0.04, 0.08],
        },
        index=pd.date_range("2025-01-31", periods=5, freq="ME"),
    )


def test_drawdown_uses_prior_wealth_peak(returns: pd.DataFrame) -> None:
    result = calculate_drawdowns(returns[["ALPHA"]])

    expected_wealth = pd.Series([1.10, 0.88, 0.924, 0.8316, 0.95634])
    expected = expected_wealth / 1.10 - 1.0
    np.testing.assert_allclose(result["ALPHA"], expected)
    assert calculate_maximum_drawdown(returns[["ALPHA"]])["ALPHA"] == pytest.approx(
        expected.min()
    )


def test_historical_var_is_reported_as_positive_loss(returns: pd.DataFrame) -> None:
    result = calculate_historical_var(returns, confidence_level=0.80)
    expected_quantile = returns.quantile(0.20)

    pd.testing.assert_series_equal(
        result, (-expected_quantile).clip(lower=0.0).rename("historical_var")
    )


def test_expected_shortfall_is_at_least_var_for_loss_series(
    returns: pd.DataFrame,
) -> None:
    value_at_risk = calculate_historical_var(returns, 0.80)
    expected_shortfall = calculate_historical_expected_shortfall(returns, 0.80)

    assert (expected_shortfall >= value_at_risk).all()
    assert expected_shortfall["ALPHA"] == pytest.approx(0.20)


def test_all_positive_returns_have_zero_var_and_expected_shortfall() -> None:
    positive = pd.DataFrame({"A": [0.01, 0.02, 0.03]})

    assert calculate_historical_var(positive, 0.95)["A"] == 0.0
    assert calculate_historical_expected_shortfall(positive, 0.95)["A"] == 0.0


def test_downside_deviation_uses_compounded_period_target() -> None:
    monthly = pd.DataFrame({"A": [-0.02, 0.03, -0.01, 0.04]})
    result = calculate_downside_deviation(monthly, 12, annual_target_rate=0.0)
    expected = np.sqrt(np.mean(np.square([-0.02, 0.0, -0.01, 0.0]))) * np.sqrt(12)

    assert result["A"] == pytest.approx(expected)


def test_beta_matches_covariance_definition_and_market_is_one(
    returns: pd.DataFrame,
) -> None:
    result = calculate_beta(returns, "MARKET")
    expected_alpha = returns["ALPHA"].cov(returns["MARKET"]) / returns[
        "MARKET"
    ].var()

    assert result["ALPHA"] == pytest.approx(expected_alpha)
    assert result["MARKET"] == pytest.approx(1.0)


def test_analyse_risk_returns_complete_metric_bundle(returns: pd.DataFrame) -> None:
    result = analyse_risk(
        returns,
        benchmark="market",
        periods_per_year=12,
        risk_free_rate=0.03,
        confidence_level=0.80,
    )

    assert result.benchmark == "MARKET"
    assert result.confidence_level == 0.80
    assert result.periods_per_year == 12
    assert list(result.metrics.columns) == [
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "historical_var",
        "historical_expected_shortfall",
        "beta",
    ]
    assert result.metrics.loc["MARKET", "beta"] == pytest.approx(1.0)
    pd.testing.assert_frame_equal(result.correlations, returns.corr())
    assert result.drawdowns.shape == returns.shape


def test_zero_volatility_asset_has_undefined_ratios(returns: pd.DataFrame) -> None:
    with_constant = returns.assign(CASHLIKE=0.01)
    result = analyse_risk(with_constant, "MARKET", 12, risk_free_rate=0.03)

    assert np.isnan(result.metrics.loc["CASHLIKE", "sharpe_ratio"])
    assert np.isnan(result.metrics.loc["CASHLIKE", "sortino_ratio"])


def test_missing_or_constant_benchmark_is_rejected(returns: pd.DataFrame) -> None:
    with pytest.raises(RiskCalculationError, match="not present"):
        analyse_risk(returns, "MISSING", 12)
    constant = returns.assign(CONSTANT=0.01)
    with pytest.raises(RiskCalculationError, match="positive variance"):
        analyse_risk(constant, "CONSTANT", 12)
    with pytest.raises(RiskCalculationError, match="positive variance"):
        calculate_beta(constant, "CONSTANT")


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, True])
def test_invalid_confidence_level_is_rejected(
    returns: pd.DataFrame, confidence
) -> None:
    with pytest.raises(RiskCalculationError, match="Confidence"):
        calculate_historical_var(returns, confidence)


@pytest.mark.parametrize("rate", [-1.0, np.inf, "4%", True])
def test_invalid_annual_rate_is_rejected(returns: pd.DataFrame, rate) -> None:
    with pytest.raises(RiskCalculationError, match="annual rate"):
        calculate_downside_deviation(returns, 12, rate)


@pytest.mark.parametrize(
    "invalid",
    [
        pd.DataFrame(),
        pd.DataFrame({"A": [0.01]}),
        pd.DataFrame({"A": [0.01, np.nan]}),
        pd.DataFrame({"A": [0.01, -1.0]}),
    ],
)
def test_invalid_return_tables_are_rejected(invalid: pd.DataFrame) -> None:
    with pytest.raises(RiskCalculationError):
        calculate_drawdowns(invalid)

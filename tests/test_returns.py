"""Tests for return transformations and annualisation."""

import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.returns import (
    ReturnCalculationError,
    analyse_returns,
    calculate_annualised_covariance,
    calculate_annualised_expected_returns,
    calculate_cumulative_returns,
    calculate_log_returns,
    calculate_simple_returns,
    validate_periods_per_year,
    validate_prices,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ALPHA": [100.0, 110.0, 99.0, 118.8],
            "BETA": [50.0, 50.0, 55.0, 60.5],
        },
        index=pd.date_range("2025-01-01", periods=4, freq="D", name="Date"),
    )


def test_simple_returns_match_known_values(prices: pd.DataFrame) -> None:
    result = calculate_simple_returns(prices)
    expected = pd.DataFrame(
        {"ALPHA": [0.10, -0.10, 0.20], "BETA": [0.00, 0.10, 0.10]},
        index=prices.index[1:],
    )

    pd.testing.assert_frame_equal(result, expected)


def test_log_returns_reconstruct_price_relatives(prices: pd.DataFrame) -> None:
    result = calculate_log_returns(prices)
    reconstructed = np.exp(result)
    expected_relatives = prices.iloc[1:] / prices.shift(1).iloc[1:].to_numpy()

    pd.testing.assert_frame_equal(reconstructed, expected_relatives)


def test_cumulative_returns_compound_instead_of_sum(prices: pd.DataFrame) -> None:
    simple = calculate_simple_returns(prices)
    result = calculate_cumulative_returns(simple)

    assert result.iloc[-1]["ALPHA"] == pytest.approx(0.188)
    assert result.iloc[-1]["BETA"] == pytest.approx(0.21)


def test_expected_returns_use_arithmetic_annualisation(prices: pd.DataFrame) -> None:
    simple = calculate_simple_returns(prices)
    result = calculate_annualised_expected_returns(simple, periods_per_year=12)

    assert result["ALPHA"] == pytest.approx(simple["ALPHA"].mean() * 12)
    assert result["BETA"] == pytest.approx(simple["BETA"].mean() * 12)
    assert result.name == "annualised_expected_return"


def test_covariance_uses_sample_covariance_and_annualisation(
    prices: pd.DataFrame,
) -> None:
    simple = calculate_simple_returns(prices)
    result = calculate_annualised_covariance(simple, periods_per_year=12)

    pd.testing.assert_frame_equal(result, simple.cov() * 12)
    pd.testing.assert_frame_equal(result, result.T)


def test_analyse_returns_builds_consistent_bundle(prices: pd.DataFrame) -> None:
    result = analyse_returns(prices, periods_per_year=252)

    assert result.periods_per_year == 252
    assert len(result.simple_returns) == len(prices) - 1
    pd.testing.assert_frame_equal(
        result.cumulative_returns,
        (1.0 + result.simple_returns).cumprod() - 1.0,
    )
    pd.testing.assert_index_equal(
        result.annualised_covariance.columns,
        result.annualised_expected_returns.index,
    )


@pytest.mark.parametrize(
    "invalid_prices",
    [
        pd.DataFrame(),
        pd.DataFrame({"A": [100.0]}),
        pd.DataFrame({"A": [100.0, np.nan]}),
        pd.DataFrame({"A": [100.0, np.inf]}),
        pd.DataFrame({"A": [100.0, 0.0]}),
    ],
)
def test_invalid_prices_are_rejected(invalid_prices: pd.DataFrame) -> None:
    with pytest.raises(ReturnCalculationError):
        validate_prices(invalid_prices)


def test_unsorted_or_duplicate_price_axes_are_rejected() -> None:
    unsorted = pd.DataFrame(
        {"A": [100.0, 101.0]}, index=pd.to_datetime(["2025-01-02", "2025-01-01"])
    )
    duplicate_dates = pd.DataFrame(
        {"A": [100.0, 101.0]}, index=pd.to_datetime(["2025-01-01", "2025-01-01"])
    )
    duplicate_assets = pd.DataFrame(
        [[100.0, 101.0], [102.0, 103.0]], columns=["A", "A"]
    )

    for frame in (unsorted, duplicate_dates, duplicate_assets):
        with pytest.raises(ReturnCalculationError):
            validate_prices(frame)


@pytest.mark.parametrize("value", [0, -12, 12.5, True])
def test_invalid_annualisation_factor_is_rejected(value) -> None:
    with pytest.raises(ReturnCalculationError, match="positive integer"):
        validate_periods_per_year(value)


def test_covariance_requires_two_complete_returns() -> None:
    one_return = pd.DataFrame({"A": [0.01], "B": [0.02]})

    with pytest.raises(ReturnCalculationError, match="At least 2"):
        calculate_annualised_covariance(one_return, 252)


def test_invalid_simple_returns_are_rejected() -> None:
    invalid = pd.DataFrame({"A": [0.10, -1.0]})

    with pytest.raises(ReturnCalculationError, match="-100%"):
        calculate_cumulative_returns(invalid)

"""Tests for consistent portfolio-strategy comparisons."""

import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.portfolio_comparison import (
    CUSTOM,
    EQUAL_WEIGHT,
    MAXIMUM_SHARPE,
    MINIMUM_VARIANCE,
    PortfolioComparisonError,
    compare_portfolios,
    validate_custom_weights,
)


@pytest.fixture
def model_inputs() -> tuple[pd.Series, pd.DataFrame]:
    assets = ["A", "B", "C", "D"]
    expected_returns = pd.Series([0.06, 0.09, 0.12, 0.15], index=assets)
    covariance = pd.DataFrame(
        [
            [0.0100, 0.0010, 0.0010, 0.0005],
            [0.0010, 0.0225, 0.0020, 0.0010],
            [0.0010, 0.0020, 0.0400, 0.0030],
            [0.0005, 0.0010, 0.0030, 0.0625],
        ],
        index=assets,
        columns=assets,
    )
    return expected_returns, covariance


def test_comparison_builds_all_four_strategies(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    custom = {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10}
    result = compare_portfolios(
        expected_returns,
        covariance,
        risk_free_rate=0.03,
        maximum_weight=0.50,
        custom_weights=custom,
    )

    expected_strategies = [
        EQUAL_WEIGHT,
        MINIMUM_VARIANCE,
        MAXIMUM_SHARPE,
        CUSTOM,
    ]
    assert list(result.weights.index) == expected_strategies
    assert list(result.statistics.index) == expected_strategies
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0)
    assert (result.weights >= -1e-8).all().all()
    assert (result.weights <= 0.50 + 1e-8).all().all()
    assert result.risk_free_rate == 0.03
    assert result.maximum_weight == 0.50


def test_equal_weight_strategy_is_exact(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    result = compare_portfolios(expected_returns, covariance, maximum_weight=0.50)

    np.testing.assert_allclose(result.weights.loc[EQUAL_WEIGHT], 0.25)
    assert CUSTOM not in result.weights.index


def test_statistics_include_concentration_measures(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    result = compare_portfolios(expected_returns, covariance, maximum_weight=0.50)

    assert list(result.statistics.columns) == [
        "expected_return",
        "volatility",
        "sharpe_ratio",
        "largest_weight",
        "effective_holdings",
    ]
    assert result.statistics.loc[EQUAL_WEIGHT, "effective_holdings"] == pytest.approx(
        4.0
    )
    assert result.statistics.loc[EQUAL_WEIGHT, "largest_weight"] == pytest.approx(
        0.25
    )


def test_display_helpers_do_not_change_raw_weights(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    result = compare_portfolios(expected_returns, covariance, maximum_weight=0.50)
    original = result.weights.copy()

    percentages = result.percentage_weights(decimals=1)
    long_form = result.long_form_weights()

    pd.testing.assert_frame_equal(result.weights, original)
    np.testing.assert_allclose(percentages.loc[EQUAL_WEIGHT], 25.0)
    assert list(long_form.columns) == ["strategy", "asset", "weight"]
    assert len(long_form) == result.weights.size


def test_custom_weights_are_aligned_by_asset_label(model_inputs) -> None:
    expected_returns, _ = model_inputs
    custom = {"D": 0.10, "B": 0.20, "A": 0.40, "C": 0.30}
    result = validate_custom_weights(
        custom, expected_returns.index, maximum_weight=0.50
    )

    assert list(result.index) == list(expected_returns.index)
    assert result.tolist() == pytest.approx([0.40, 0.20, 0.30, 0.10])


@pytest.mark.parametrize(
    ("custom", "message"),
    [
        ({"A": 0.50, "B": 0.30, "C": 0.20}, "match the selected assets"),
        (
            {"A": 0.30, "B": 0.30, "C": 0.30, "D": 0.30},
            "sum to 100%",
        ),
        (
            {"A": 0.70, "B": 0.10, "C": 0.10, "D": 0.10},
            "exceed the maximum",
        ),
        (
            {"A": -0.10, "B": 0.40, "C": 0.40, "D": 0.30},
            "negative",
        ),
        (
            {"A": np.nan, "B": 0.30, "C": 0.30, "D": 0.40},
            "finite",
        ),
    ],
)
def test_invalid_custom_weights_are_rejected(model_inputs, custom, message) -> None:
    expected_returns, _ = model_inputs
    with pytest.raises(PortfolioComparisonError, match=message):
        validate_custom_weights(
            custom, expected_returns.index, maximum_weight=0.50
        )


def test_optimisation_errors_use_comparison_boundary(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    with pytest.raises(PortfolioComparisonError, match="infeasible"):
        compare_portfolios(expected_returns, covariance, maximum_weight=0.20)

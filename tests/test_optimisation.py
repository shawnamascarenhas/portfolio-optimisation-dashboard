"""Tests for constrained mean-variance portfolio optimisation."""

import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.optimisation import (
    OptimisationError,
    calculate_efficient_frontier,
    optimise_maximum_sharpe,
    optimise_minimum_variance,
    portfolio_statistics,
)


@pytest.fixture
def model_inputs() -> tuple[pd.Series, pd.DataFrame]:
    labels = ["LOW", "MID", "HIGH"]
    expected_returns = pd.Series([0.06, 0.10, 0.16], index=labels)
    covariance = pd.DataFrame(
        [
            [0.0100, 0.0020, 0.0010],
            [0.0020, 0.0225, 0.0030],
            [0.0010, 0.0030, 0.0625],
        ],
        index=labels,
        columns=labels,
    )
    return expected_returns, covariance


def _assert_valid_weights(weights: pd.Series, maximum: float) -> None:
    assert weights.sum() == pytest.approx(1.0, abs=1e-7)
    assert (weights >= -1e-8).all()
    assert (weights <= maximum + 1e-8).all()


def test_minimum_variance_solution_satisfies_constraints(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    result = optimise_minimum_variance(
        expected_returns, covariance, maximum_weight=0.60
    )

    _assert_valid_weights(result.weights, 0.60)
    equal = pd.Series(1 / 3, index=expected_returns.index)
    equal_variance = float(equal @ covariance @ equal)
    assert result.volatility**2 <= equal_variance + 1e-9


def test_maximum_sharpe_beats_equal_weight_portfolio(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    result = optimise_maximum_sharpe(
        expected_returns,
        covariance,
        risk_free_rate=0.03,
        maximum_weight=0.70,
    )
    equal = portfolio_statistics(
        pd.Series(1 / 3, index=expected_returns.index),
        expected_returns,
        covariance,
        risk_free_rate=0.03,
    )

    _assert_valid_weights(result.weights, 0.70)
    assert result.sharpe_ratio >= equal.sharpe_ratio


def test_frontier_is_ordered_and_respects_target_returns(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    frontier = calculate_efficient_frontier(
        expected_returns,
        covariance,
        maximum_weight=0.60,
        risk_free_rate=0.03,
        number_of_points=12,
    )

    assert frontier.points.shape == (12, 3)
    assert frontier.weights.shape == (12, 3)
    assert frontier.points["expected_return"].is_monotonic_increasing
    assert frontier.points["volatility"].is_monotonic_increasing
    for _, weights in frontier.weights.iterrows():
        _assert_valid_weights(weights, 0.60)


def test_frontier_starts_at_minimum_variance_portfolio(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    minimum = optimise_minimum_variance(expected_returns, covariance)
    frontier = calculate_efficient_frontier(
        expected_returns, covariance, number_of_points=5
    )

    assert frontier.points.iloc[0]["expected_return"] == pytest.approx(
        minimum.expected_return
    )
    assert frontier.points.iloc[0]["volatility"] == pytest.approx(
        minimum.volatility
    )


def test_portfolio_statistics_aligns_labels(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    weights = pd.Series({"HIGH": 0.2, "LOW": 0.5, "MID": 0.3})
    result = portfolio_statistics(
        weights, expected_returns, covariance, risk_free_rate=0.02
    )

    assert list(result.weights.index) == list(expected_returns.index)
    assert result.expected_return == pytest.approx(0.092)
    assert result.volatility > 0
    assert result.sharpe_ratio == pytest.approx(
        (result.expected_return - 0.02) / result.volatility
    )


def test_infeasible_maximum_weight_is_rejected(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    with pytest.raises(OptimisationError, match="infeasible"):
        optimise_minimum_variance(
            expected_returns, covariance, maximum_weight=0.30
        )


@pytest.mark.parametrize("maximum", [0.0, -0.1, 1.1, np.inf, True])
def test_invalid_maximum_weight_is_rejected(model_inputs, maximum) -> None:
    expected_returns, covariance = model_inputs
    with pytest.raises(OptimisationError, match="Maximum weight"):
        optimise_minimum_variance(
            expected_returns, covariance, maximum_weight=maximum
        )


def test_misaligned_or_invalid_covariance_is_rejected(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    misaligned = covariance.rename(index={"HIGH": "OTHER"})
    nonsymmetric = covariance.copy()
    nonsymmetric.iloc[0, 1] = 0.009
    non_psd = covariance.copy()
    non_psd.iloc[0, 1] = non_psd.iloc[1, 0] = 0.10

    with pytest.raises(OptimisationError, match="labels"):
        optimise_minimum_variance(expected_returns, misaligned)
    with pytest.raises(OptimisationError, match="symmetric"):
        optimise_minimum_variance(expected_returns, nonsymmetric)
    with pytest.raises(OptimisationError, match="positive semidefinite"):
        optimise_minimum_variance(expected_returns, non_psd)


def test_invalid_weights_are_rejected(model_inputs) -> None:
    expected_returns, covariance = model_inputs
    with pytest.raises(OptimisationError, match="labels"):
        portfolio_statistics(
            pd.Series({"LOW": 0.5, "MID": 0.5}), expected_returns, covariance
        )
    with pytest.raises(OptimisationError, match="sum to one"):
        portfolio_statistics(
            pd.Series([0.2, 0.2, 0.2], index=expected_returns.index),
            expected_returns,
            covariance,
        )


@pytest.mark.parametrize("points", [1, 2.5, True])
def test_invalid_frontier_point_count_is_rejected(model_inputs, points) -> None:
    expected_returns, covariance = model_inputs
    with pytest.raises(OptimisationError, match="frontier points|integer"):
        calculate_efficient_frontier(
            expected_returns, covariance, number_of_points=points
        )


@pytest.mark.parametrize("rate", [-1.0, np.inf, "4%", True])
def test_invalid_risk_free_rate_is_rejected(model_inputs, rate) -> None:
    expected_returns, covariance = model_inputs
    with pytest.raises(OptimisationError, match="Risk-free"):
        optimise_maximum_sharpe(
            expected_returns, covariance, risk_free_rate=rate
        )

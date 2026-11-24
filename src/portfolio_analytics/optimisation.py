"""Constrained mean-variance portfolio optimisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize


class OptimisationError(ValueError):
    """Raised when inputs or a numerical optimisation result are invalid."""


@dataclass(frozen=True)
class PortfolioSolution:
    """Weights and annualised statistics for one optimised portfolio."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float


@dataclass(frozen=True)
class EfficientFrontier:
    """Efficient portfolios ordered by increasing target return."""

    points: pd.DataFrame
    weights: pd.DataFrame


def optimise_minimum_variance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    maximum_weight: float = 1.0,
) -> PortfolioSolution:
    """Find the fully invested long-only portfolio with minimum variance."""
    returns, covariance = _validate_inputs(
        expected_returns, covariance, maximum_weight
    )
    bounds = _bounds(len(returns), maximum_weight)
    result = minimize(
        _portfolio_variance,
        _equal_weights(len(returns)),
        args=(covariance.to_numpy(),),
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    weights = _validate_solution(result, returns.index, maximum_weight)
    return portfolio_statistics(weights, returns, covariance)


def optimise_maximum_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_free_rate: float = 0.04,
    maximum_weight: float = 1.0,
) -> PortfolioSolution:
    """Find the fully invested long-only portfolio with maximum Sharpe ratio."""
    returns, covariance = _validate_inputs(
        expected_returns, covariance, maximum_weight
    )
    risk_free = _validate_risk_free_rate(risk_free_rate)
    covariance_array = covariance.to_numpy()
    return_array = returns.to_numpy()

    def negative_sharpe(weights: np.ndarray) -> float:
        volatility = np.sqrt(_portfolio_variance(weights, covariance_array))
        if volatility <= 1e-14:
            return 1e12
        return -float((weights @ return_array - risk_free) / volatility)

    result = minimize(
        negative_sharpe,
        _equal_weights(len(returns)),
        method="SLSQP",
        bounds=_bounds(len(returns), maximum_weight),
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    weights = _validate_solution(result, returns.index, maximum_weight)
    return portfolio_statistics(weights, returns, covariance, risk_free)


def calculate_efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    maximum_weight: float = 1.0,
    risk_free_rate: float = 0.04,
    number_of_points: int = 30,
) -> EfficientFrontier:
    """Calculate the upper, non-dominated branch of the efficient frontier."""
    returns, covariance = _validate_inputs(
        expected_returns, covariance, maximum_weight
    )
    risk_free = _validate_risk_free_rate(risk_free_rate)
    if isinstance(number_of_points, bool) or not isinstance(number_of_points, int):
        raise OptimisationError("Number of frontier points must be an integer.")
    if number_of_points < 2:
        raise OptimisationError("At least two frontier points are required.")

    minimum_variance = optimise_minimum_variance(
        returns, covariance, maximum_weight=maximum_weight
    )
    maximum_return_weights = _optimise_maximum_return(returns, maximum_weight)
    maximum_return = float(maximum_return_weights @ returns)
    targets = np.linspace(
        minimum_variance.expected_return, maximum_return, number_of_points
    )

    point_records: list[dict[str, float]] = []
    weight_records: list[pd.Series] = []
    covariance_array = covariance.to_numpy()
    return_array = returns.to_numpy()
    start_weights = minimum_variance.weights.to_numpy()

    for target in targets:
        result = minimize(
            _portfolio_variance,
            start_weights,
            args=(covariance_array,),
            method="SLSQP",
            bounds=_bounds(len(returns), maximum_weight),
            constraints=[
                {"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
                {
                    "type": "eq",
                    "fun": lambda weights, target=target: (
                        weights @ return_array - target
                    ),
                },
            ],
            options={"ftol": 1e-12, "maxiter": 2_000},
        )
        weights = _validate_solution(
            result,
            returns.index,
            maximum_weight,
            target_return=target,
            expected_returns=returns,
        )
        solution = portfolio_statistics(weights, returns, covariance, risk_free)
        point_records.append(
            {
                "expected_return": solution.expected_return,
                "volatility": solution.volatility,
                "sharpe_ratio": solution.sharpe_ratio,
            }
        )
        weight_records.append(solution.weights)
        start_weights = solution.weights.to_numpy()

    points = pd.DataFrame(point_records)
    points.index.name = "frontier_point"
    weights = pd.DataFrame(weight_records)
    weights.index = points.index
    return EfficientFrontier(points=points, weights=weights)


def portfolio_statistics(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> PortfolioSolution:
    """Calculate annualised return, volatility, and Sharpe for fixed weights."""
    returns, covariance = _validate_inputs(expected_returns, covariance, 1.0)
    risk_free = _validate_risk_free_rate(risk_free_rate)
    if not isinstance(weights, pd.Series):
        raise OptimisationError("Weights must be provided as a Series.")
    if set(weights.index) != set(returns.index):
        raise OptimisationError("Weight labels must match expected-return labels.")
    aligned_weights = weights.reindex(returns.index).astype(float)
    if not np.isfinite(aligned_weights.to_numpy()).all():
        raise OptimisationError("Weights must be finite.")
    if (aligned_weights < -1e-8).any() or not np.isclose(
        aligned_weights.sum(), 1.0, atol=1e-7
    ):
        raise OptimisationError("Weights must be non-negative and sum to one.")

    expected_return = float(aligned_weights @ returns)
    variance = _portfolio_variance(
        aligned_weights.to_numpy(), covariance.to_numpy()
    )
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = (
        float((expected_return - risk_free) / volatility)
        if volatility > 1e-14
        else np.nan
    )
    aligned_weights.name = "weight"
    return PortfolioSolution(
        weights=aligned_weights,
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
    )


def _validate_inputs(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    maximum_weight: float,
) -> tuple[pd.Series, pd.DataFrame]:
    if not isinstance(expected_returns, pd.Series) or expected_returns.empty:
        raise OptimisationError("Expected returns must be a non-empty Series.")
    if expected_returns.index.has_duplicates:
        raise OptimisationError("Expected-return labels must be unique.")
    returns = pd.to_numeric(expected_returns, errors="coerce").astype(float)
    if not np.isfinite(returns.to_numpy()).all():
        raise OptimisationError("Expected returns must be finite.")

    if not isinstance(covariance, pd.DataFrame) or covariance.empty:
        raise OptimisationError("Covariance must be a non-empty DataFrame.")
    if covariance.shape[0] != covariance.shape[1]:
        raise OptimisationError("Covariance matrix must be square.")
    if covariance.index.has_duplicates or covariance.columns.has_duplicates:
        raise OptimisationError("Covariance labels must be unique.")
    if set(covariance.index) != set(returns.index) or set(covariance.columns) != set(
        returns.index
    ):
        raise OptimisationError(
            "Covariance labels must match expected-return labels."
        )
    covariance = covariance.reindex(index=returns.index, columns=returns.index)
    covariance = covariance.apply(pd.to_numeric, errors="coerce").astype(float)
    covariance_array = covariance.to_numpy()
    if not np.isfinite(covariance_array).all():
        raise OptimisationError("Covariance matrix must be finite.")
    if not np.allclose(covariance_array, covariance_array.T, atol=1e-10):
        raise OptimisationError("Covariance matrix must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(covariance_array)
    if eigenvalues.min() < -1e-10:
        raise OptimisationError("Covariance matrix must be positive semidefinite.")
    if (np.diag(covariance_array) <= 0).any():
        raise OptimisationError("Every asset must have positive variance.")

    _validate_maximum_weight(maximum_weight, len(returns))
    return returns, covariance


def _validate_maximum_weight(maximum_weight: float, asset_count: int) -> float:
    if isinstance(maximum_weight, bool) or not isinstance(
        maximum_weight, (int, float)
    ):
        raise OptimisationError("Maximum weight must be numeric.")
    maximum = float(maximum_weight)
    if not np.isfinite(maximum) or not 0.0 < maximum <= 1.0:
        raise OptimisationError("Maximum weight must be between zero and one.")
    if asset_count * maximum < 1.0 - 1e-12:
        raise OptimisationError(
            "Maximum-weight constraint is infeasible for the number of assets."
        )
    return maximum


def _validate_risk_free_rate(risk_free_rate: float) -> float:
    if isinstance(risk_free_rate, bool) or not isinstance(
        risk_free_rate, (int, float)
    ):
        raise OptimisationError("Risk-free rate must be numeric.")
    rate = float(risk_free_rate)
    if not np.isfinite(rate) or rate <= -1.0:
        raise OptimisationError("Risk-free rate must be finite and greater than -1.")
    return rate


def _validate_solution(
    result: OptimizeResult,
    labels: pd.Index,
    maximum_weight: float,
    *,
    target_return: float | None = None,
    expected_returns: pd.Series | None = None,
) -> pd.Series:
    if not result.success or result.x is None:
        message = getattr(result, "message", "unknown solver error")
        raise OptimisationError(f"Portfolio optimisation failed: {message}")
    weights = np.asarray(result.x, dtype=float)
    if not np.isfinite(weights).all():
        raise OptimisationError("Optimiser returned non-finite weights.")
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise OptimisationError("Optimiser returned weights that do not sum to one.")
    if weights.min() < -1e-7 or weights.max() > maximum_weight + 1e-7:
        raise OptimisationError("Optimiser returned weights outside the constraints.")
    weights = np.clip(weights, 0.0, maximum_weight)
    weights = weights / weights.sum()
    series = pd.Series(weights, index=labels, name="weight")
    if target_return is not None and expected_returns is not None:
        realised = float(series @ expected_returns)
        if not np.isclose(realised, target_return, atol=1e-6):
            raise OptimisationError(
                "Optimiser did not satisfy the target-return constraint."
            )
    return series


def _optimise_maximum_return(
    expected_returns: pd.Series, maximum_weight: float
) -> pd.Series:
    result = minimize(
        lambda weights: -float(weights @ expected_returns.to_numpy()),
        _equal_weights(len(expected_returns)),
        method="SLSQP",
        bounds=_bounds(len(expected_returns), maximum_weight),
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    return _validate_solution(result, expected_returns.index, maximum_weight)


def _portfolio_variance(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(weights @ covariance @ weights)


def _equal_weights(asset_count: int) -> np.ndarray:
    return np.full(asset_count, 1.0 / asset_count)


def _bounds(asset_count: int, maximum_weight: float) -> tuple[tuple[float, float], ...]:
    return tuple((0.0, maximum_weight) for _ in range(asset_count))

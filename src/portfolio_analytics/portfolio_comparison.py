"""Comparable portfolio strategies and dashboard-ready allocation tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_analytics.optimisation import (
    OptimisationError,
    PortfolioSolution,
    optimise_maximum_sharpe,
    optimise_minimum_variance,
    portfolio_statistics,
)

EQUAL_WEIGHT = "Equal Weight"
MINIMUM_VARIANCE = "Minimum Variance"
MAXIMUM_SHARPE = "Maximum Sharpe"
CUSTOM = "Custom"


class PortfolioComparisonError(ValueError):
    """Raised when portfolio strategies cannot be compared consistently."""


@dataclass(frozen=True)
class PortfolioComparison:
    """Strategy allocations and modelled annual risk-return statistics."""

    weights: pd.DataFrame
    statistics: pd.DataFrame
    risk_free_rate: float
    maximum_weight: float

    def percentage_weights(self, decimals: int = 2) -> pd.DataFrame:
        """Return display weights in percentage points without changing raw data."""
        return (self.weights * 100.0).round(decimals)

    def long_form_weights(self) -> pd.DataFrame:
        """Return allocation data suitable for grouped bar charts."""
        return (
            self.weights.rename_axis(index="strategy", columns="asset")
            .stack()
            .rename("weight")
            .reset_index()
        )


def compare_portfolios(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_free_rate: float = 0.04,
    maximum_weight: float = 1.0,
    custom_weights: pd.Series | Mapping[str, float] | None = None,
) -> PortfolioComparison:
    """Construct portfolio strategies under a common set of model inputs."""
    try:
        minimum_variance = optimise_minimum_variance(
            expected_returns, covariance, maximum_weight=maximum_weight
        )
        maximum_sharpe = optimise_maximum_sharpe(
            expected_returns,
            covariance,
            risk_free_rate=risk_free_rate,
            maximum_weight=maximum_weight,
        )
        equal_weights = pd.Series(
            1.0 / len(expected_returns),
            index=expected_returns.index,
            name="weight",
        )
        equal_weight = portfolio_statistics(
            equal_weights,
            expected_returns,
            covariance,
            risk_free_rate=risk_free_rate,
        )
    except OptimisationError as exc:
        raise PortfolioComparisonError(str(exc)) from exc

    solutions: dict[str, PortfolioSolution] = {
        EQUAL_WEIGHT: equal_weight,
        MINIMUM_VARIANCE: minimum_variance,
        MAXIMUM_SHARPE: maximum_sharpe,
    }

    if custom_weights is not None:
        custom = validate_custom_weights(
            custom_weights,
            expected_returns.index,
            maximum_weight=maximum_weight,
        )
        try:
            solutions[CUSTOM] = portfolio_statistics(
                custom,
                expected_returns,
                covariance,
                risk_free_rate=risk_free_rate,
            )
        except OptimisationError as exc:
            raise PortfolioComparisonError(str(exc)) from exc

    weights = pd.DataFrame(
        {name: solution.weights for name, solution in solutions.items()}
    ).T
    weights.index.name = "strategy"
    weights.columns.name = "asset"

    statistics = pd.DataFrame(
        {
            name: {
                "expected_return": solution.expected_return,
                "volatility": solution.volatility,
                "sharpe_ratio": solution.sharpe_ratio,
                "largest_weight": float(solution.weights.max()),
                "effective_holdings": _effective_holdings(solution.weights),
            }
            for name, solution in solutions.items()
        }
    ).T
    statistics.index.name = "strategy"

    return PortfolioComparison(
        weights=weights,
        statistics=statistics,
        risk_free_rate=float(risk_free_rate),
        maximum_weight=float(maximum_weight),
    )


def validate_custom_weights(
    weights: pd.Series | Mapping[str, float],
    asset_labels: pd.Index,
    *,
    maximum_weight: float = 1.0,
) -> pd.Series:
    """Validate a user allocation expressed as decimal weights."""
    if isinstance(weights, Mapping):
        weights = pd.Series(dict(weights), dtype=float)
    if not isinstance(weights, pd.Series):
        raise PortfolioComparisonError(
            "Custom weights must be a labelled Series or mapping."
        )
    if weights.index.has_duplicates:
        raise PortfolioComparisonError("Custom weight labels must be unique.")
    if set(weights.index) != set(asset_labels):
        missing = sorted(set(asset_labels) - set(weights.index))
        extra = sorted(set(weights.index) - set(asset_labels))
        details = []
        if missing:
            details.append(f"missing: {', '.join(map(str, missing))}")
        if extra:
            details.append(f"unexpected: {', '.join(map(str, extra))}")
        detail_text = "; ".join(details)
        raise PortfolioComparisonError(
            f"Custom weights must match the selected assets ({detail_text})."
        )

    try:
        numeric = pd.to_numeric(weights.reindex(asset_labels), errors="raise").astype(
            float
        )
    except (TypeError, ValueError) as exc:
        raise PortfolioComparisonError("Custom weights must be numeric.") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise PortfolioComparisonError("Custom weights must be finite.")
    if (numeric < 0.0).any():
        raise PortfolioComparisonError("Custom weights cannot be negative.")
    if not np.isclose(numeric.sum(), 1.0, atol=1e-6):
        raise PortfolioComparisonError("Custom weights must sum to 100%.")
    if (numeric > float(maximum_weight) + 1e-8).any():
        offenders = numeric[numeric > float(maximum_weight) + 1e-8].index
        raise PortfolioComparisonError(
            "Custom weights exceed the maximum for: "
            + ", ".join(map(str, offenders))
            + "."
        )
    numeric.name = "weight"
    return numeric


def _effective_holdings(weights: pd.Series) -> float:
    """Return inverse concentration (1 / Herfindahl index)."""
    return float(1.0 / np.square(weights).sum())

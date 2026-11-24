"""Structured model-risk notices for the dashboard and reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Severity = Literal["info", "warning"]


@dataclass(frozen=True)
class SafeguardNotice:
    """One user-facing disclosure or model-risk warning."""

    code: str
    severity: Severity
    message: str


def assess_model_safeguards(
    prices: pd.DataFrame,
    strategy_weights: pd.DataFrame,
    *,
    periods_per_year: int,
    in_sample_observations: int,
    out_of_sample_observations: int,
    maximum_weight: float,
    cache_warning: str | None = None,
) -> tuple[SafeguardNotice, ...]:
    """Return deterministic disclosures based on the completed analysis."""
    _validate_inputs(
        prices,
        strategy_weights,
        periods_per_year,
        in_sample_observations,
        out_of_sample_observations,
        maximum_weight,
    )
    notices: list[SafeguardNotice] = []

    if len(prices) < periods_per_year * 2:
        notices.append(
            SafeguardNotice(
                "SHORT_TOTAL_HISTORY",
                "warning",
                "The aligned history contains less than two years of observations; "
                "estimated returns and covariance may be especially unstable.",
            )
        )
    if in_sample_observations < periods_per_year:
        notices.append(
            SafeguardNotice(
                "SHORT_ESTIMATION_WINDOW",
                "warning",
                "The in-sample estimation window contains less than one year of "
                "observations at the selected frequency.",
            )
        )
    minimum_evaluation_sample = max(12, int(np.ceil(periods_per_year * 0.10)))
    if out_of_sample_observations < minimum_evaluation_sample:
        notices.append(
            SafeguardNotice(
                "SHORT_EVALUATION_WINDOW",
                "warning",
                "The out-of-sample period is short; CAGR, drawdown, VaR, and "
                "Expected Shortfall may be unreliable.",
            )
        )

    binding = strategy_weights.ge(maximum_weight - 1e-4).any(axis=1)
    if binding.any():
        strategies = ", ".join(strategy_weights.index[binding].astype(str))
        notices.append(
            SafeguardNotice(
                "BINDING_CONCENTRATION_LIMIT",
                "warning",
                "The maximum-weight constraint is binding for: " + strategies + ".",
            )
        )
    if cache_warning:
        notices.append(SafeguardNotice("CACHE_UNAVAILABLE", "warning", cache_warning))

    notices.extend(
        [
            SafeguardNotice(
                "HISTORICAL_MODEL",
                "info",
                "Returns, covariance, VaR, and Expected Shortfall are historical "
                "estimates and are not forecasts.",
            ),
            SafeguardNotice(
                "FRICTIONLESS_BACKTEST",
                "info",
                "The backtest excludes transaction costs, taxes, bid-ask spreads, "
                "market impact, and currency conversion.",
            ),
            SafeguardNotice(
                "EDUCATIONAL_ONLY",
                "info",
                "This application is for education and analysis only and does not "
                "provide investment advice.",
            ),
        ]
    )
    return tuple(notices)


def _validate_inputs(
    prices: pd.DataFrame,
    strategy_weights: pd.DataFrame,
    periods_per_year: int,
    in_sample_observations: int,
    out_of_sample_observations: int,
    maximum_weight: float,
) -> None:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Prices must be a non-empty DataFrame.")
    if not isinstance(strategy_weights, pd.DataFrame) or strategy_weights.empty:
        raise ValueError("Strategy weights must be a non-empty DataFrame.")
    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")
    if in_sample_observations <= 0 or out_of_sample_observations <= 0:
        raise ValueError("Backtest sample sizes must be positive.")
    if not 0 < maximum_weight <= 1:
        raise ValueError("Maximum weight must be between zero and one.")

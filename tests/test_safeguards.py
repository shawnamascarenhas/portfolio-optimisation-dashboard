"""Tests for model-risk warnings and permanent disclosures."""

import pandas as pd
import pytest

from portfolio_analytics.safeguards import assess_model_safeguards


def _prices(observations: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"A": range(100, 100 + observations)},
        index=pd.date_range("2024-01-01", periods=observations),
    )


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": [0.60, 0.40], "B": [0.40, 0.60]},
        index=["Minimum Variance", "Maximum Sharpe"],
    )


def test_short_samples_binding_limit_and_cache_issue_generate_warnings() -> None:
    notices = assess_model_safeguards(
        _prices(100),
        _weights(),
        periods_per_year=252,
        in_sample_observations=70,
        out_of_sample_observations=20,
        maximum_weight=0.60,
        cache_warning="Cache unavailable.",
    )
    warning_codes = {
        notice.code for notice in notices if notice.severity == "warning"
    }

    assert warning_codes == {
        "SHORT_TOTAL_HISTORY",
        "SHORT_ESTIMATION_WINDOW",
        "SHORT_EVALUATION_WINDOW",
        "BINDING_CONCENTRATION_LIMIT",
        "CACHE_UNAVAILABLE",
    }


def test_permanent_disclosures_are_always_present() -> None:
    notices = assess_model_safeguards(
        _prices(600),
        pd.DataFrame(
            {"A": [0.50], "B": [0.50]}, index=["Equal Weight"]
        ),
        periods_per_year=252,
        in_sample_observations=400,
        out_of_sample_observations=200,
        maximum_weight=0.80,
    )
    codes = {notice.code for notice in notices}

    assert codes == {
        "HISTORICAL_MODEL",
        "FRICTIONLESS_BACKTEST",
        "EDUCATIONAL_ONLY",
    }
    assert all(notice.severity == "info" for notice in notices)


@pytest.mark.parametrize(
    "prices, weights, periods, in_sample, out_sample, maximum",
    [
        (pd.DataFrame(), _weights(), 252, 100, 50, 0.6),
        (_prices(20), pd.DataFrame(), 252, 100, 50, 0.6),
        (_prices(20), _weights(), 0, 100, 50, 0.6),
        (_prices(20), _weights(), 252, 0, 50, 0.6),
        (_prices(20), _weights(), 252, 100, 0, 0.6),
        (_prices(20), _weights(), 252, 100, 50, 1.1),
    ],
)
def test_invalid_safeguard_inputs_are_rejected(
    prices, weights, periods, in_sample, out_sample, maximum
) -> None:
    with pytest.raises(ValueError):
        assess_model_safeguards(
            prices,
            weights,
            periods_per_year=periods,
            in_sample_observations=in_sample,
            out_of_sample_observations=out_sample,
            maximum_weight=maximum,
        )

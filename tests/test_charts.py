"""Tests for dashboard Plotly figure construction."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from portfolio_analytics.charts import (
    allocation_chart,
    correlation_heatmap,
    cumulative_return_chart,
    drawdown_chart,
    efficient_frontier_chart,
    normalised_price_chart,
    portfolio_growth_chart,
)
from portfolio_analytics.optimisation import EfficientFrontier, PortfolioSolution


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": [100.0, 105.0, 103.0], "B": [50.0, 51.0, 55.0]},
        index=pd.date_range("2025-01-01", periods=3),
    )


@pytest.mark.parametrize(
    "builder",
    [
        normalised_price_chart,
        cumulative_return_chart,
        allocation_chart,
        portfolio_growth_chart,
        drawdown_chart,
    ],
)
def test_standard_charts_return_figures(builder, frame) -> None:
    figure = builder(frame)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) >= 1


def test_normalised_chart_rebases_each_asset_to_100(frame) -> None:
    figure = normalised_price_chart(frame)

    for trace in figure.data:
        assert trace.y[0] == pytest.approx(100.0)


def test_correlation_heatmap_has_fixed_scale() -> None:
    correlations = pd.DataFrame(
        [[1.0, -0.25], [-0.25, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    figure = correlation_heatmap(correlations)

    assert isinstance(figure, go.Figure)
    assert figure.data[0].zmin == -1
    assert figure.data[0].zmax == 1


def test_efficient_frontier_chart_contains_four_trace_groups() -> None:
    points = pd.DataFrame(
        {
            "expected_return": [0.08, 0.10, 0.12],
            "volatility": [0.10, 0.12, 0.16],
            "sharpe_ratio": [0.4, 0.5, 0.5],
        }
    )
    weights = pd.DataFrame(
        [[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], columns=["A", "B"]
    )
    frontier = EfficientFrontier(points=points, weights=weights)
    minimum = PortfolioSolution(
        pd.Series({"A": 0.6, "B": 0.4}), 0.08, 0.10, 0.4
    )
    maximum = PortfolioSolution(
        pd.Series({"A": 0.3, "B": 0.7}), 0.12, 0.16, 0.5
    )
    figure = efficient_frontier_chart(
        frontier,
        pd.Series({"A": 0.07, "B": 0.13}),
        pd.Series({"A": 0.11, "B": 0.20}),
        minimum,
        maximum,
    )

    assert [trace.name for trace in figure.data] == [
        "Efficient Frontier",
        "Assets",
        "Minimum Variance",
        "Maximum Sharpe",
    ]


def test_chart_builders_reject_empty_frames() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalised_price_chart(pd.DataFrame())

"""Interactive Plotly figures for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from portfolio_analytics.optimisation import EfficientFrontier, PortfolioSolution

_LAYOUT = {
    "template": "plotly_white",
    "hovermode": "x unified",
    "margin": {"l": 30, "r": 20, "t": 55, "b": 35},
    "legend": {"orientation": "h", "y": 1.08, "x": 0},
}


def normalised_price_chart(prices: pd.DataFrame) -> go.Figure:
    """Plot every price series rebased to 100 at its first observation."""
    _require_frame(prices, "prices")
    normalised = prices.divide(prices.iloc[0]).multiply(100.0)
    figure = px.line(
        normalised,
        x=normalised.index,
        y=normalised.columns,
        labels={"value": "Normalised value", "variable": "Asset"},
        title="Normalised Price Performance",
    )
    return _style(figure, y_tickformat=".1f")


def cumulative_return_chart(cumulative_returns: pd.DataFrame) -> go.Figure:
    """Plot compounded cumulative returns."""
    _require_frame(cumulative_returns, "cumulative returns")
    figure = px.line(
        cumulative_returns,
        x=cumulative_returns.index,
        y=cumulative_returns.columns,
        labels={"value": "Cumulative return", "variable": "Asset"},
        title="Cumulative Returns",
    )
    return _style(figure, y_tickformat=".1%")


def correlation_heatmap(correlations: pd.DataFrame) -> go.Figure:
    """Plot a labelled return-correlation heatmap."""
    _require_frame(correlations, "correlations")
    figure = go.Figure(
        go.Heatmap(
            z=correlations.to_numpy(),
            x=correlations.columns,
            y=correlations.index,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            reversescale=True,
            text=correlations.round(2).astype(str).to_numpy(),
            texttemplate="%{text}",
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(title="Return Correlations", **_LAYOUT)
    return figure


def efficient_frontier_chart(
    frontier: EfficientFrontier,
    asset_returns: pd.Series,
    asset_volatilities: pd.Series,
    minimum_variance: PortfolioSolution,
    maximum_sharpe: PortfolioSolution,
) -> go.Figure:
    """Plot the efficient frontier, assets, and selected optimal portfolios."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frontier.points["volatility"],
            y=frontier.points["expected_return"],
            mode="lines",
            name="Efficient Frontier",
            line={"width": 3, "color": "#2563EB"},
            hovertemplate="Volatility %{x:.2%}<br>Return %{y:.2%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=asset_volatilities,
            y=asset_returns,
            text=asset_returns.index,
            mode="markers+text",
            textposition="top center",
            name="Assets",
            marker={"size": 10, "color": "#64748B"},
            hovertemplate=(
                "%{text}<br>Volatility %{x:.2%}<br>Return %{y:.2%}<extra></extra>"
            ),
        )
    )
    for name, solution, colour, symbol in (
        ("Minimum Variance", minimum_variance, "#059669", "diamond"),
        ("Maximum Sharpe", maximum_sharpe, "#DC2626", "star"),
    ):
        figure.add_trace(
            go.Scatter(
                x=[solution.volatility],
                y=[solution.expected_return],
                mode="markers",
                name=name,
                marker={"size": 15, "color": colour, "symbol": symbol},
                hovertemplate=(
                    f"{name}<br>Volatility %{{x:.2%}}<br>Return %{{y:.2%}}"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualised volatility",
        yaxis_title="Annualised expected return",
        **_LAYOUT,
    )
    figure.update_xaxes(tickformat=".1%")
    figure.update_yaxes(tickformat=".1%")
    return figure


def allocation_chart(weights: pd.DataFrame) -> go.Figure:
    """Plot portfolio strategy weights as grouped bars."""
    _require_frame(weights, "weights")
    long_form = (
        weights.rename_axis(index="Strategy", columns="Asset")
        .stack()
        .rename("Weight")
        .reset_index()
    )
    figure = px.bar(
        long_form,
        x="Asset",
        y="Weight",
        color="Strategy",
        barmode="group",
        title="Portfolio Allocations",
    )
    return _style(figure, y_tickformat=".0%")


def portfolio_growth_chart(portfolio_values: pd.DataFrame) -> go.Figure:
    """Plot out-of-sample strategy values."""
    _require_frame(portfolio_values, "portfolio values")
    figure = px.line(
        portfolio_values,
        x=portfolio_values.index,
        y=portfolio_values.columns,
        labels={"value": "Portfolio value", "variable": "Strategy"},
        title="Out-of-Sample Portfolio Growth",
    )
    return _style(figure, y_tickformat=",")


def drawdown_chart(drawdowns: pd.DataFrame, title: str = "Drawdowns") -> go.Figure:
    """Plot peak-to-trough drawdown histories."""
    _require_frame(drawdowns, "drawdowns")
    figure = px.line(
        drawdowns,
        x=drawdowns.index,
        y=drawdowns.columns,
        labels={"value": "Drawdown", "variable": "Series"},
        title=title,
    )
    figure.add_hline(y=0, line_width=1, line_color="#94A3B8")
    return _style(figure, y_tickformat=".1%")


def _style(figure: go.Figure, *, y_tickformat: str) -> go.Figure:
    figure.update_layout(**_LAYOUT)
    figure.update_yaxes(tickformat=y_tickformat)
    return figure


def _require_frame(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label.capitalize()} must be a non-empty DataFrame.")

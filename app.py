"""Streamlit interface for the portfolio optimisation dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from portfolio_analytics import (
    MarketDataError,
    OptimisationError,
    PortfolioComparisonError,
    ReturnCalculationError,
    RiskCalculationError,
    SafeguardNotice,
    analyse_returns,
    analyse_risk,
    assess_model_safeguards,
    calculate_efficient_frontier,
    create_request,
    load_market_data,
    make_demo_downloader,
    optimise_maximum_sharpe,
    optimise_minimum_variance,
    run_backtest,
)
from portfolio_analytics.backtest import BacktestError, BacktestResult
from portfolio_analytics.charts import (
    allocation_chart,
    correlation_heatmap,
    cumulative_return_chart,
    drawdown_chart,
    efficient_frontier_chart,
    normalised_price_chart,
    portfolio_growth_chart,
)
from portfolio_analytics.data_loader import MarketDataResult
from portfolio_analytics.optimisation import EfficientFrontier, PortfolioSolution
from portfolio_analytics.returns import ReturnAnalysis
from portfolio_analytics.risk_metrics import RiskAnalysis, calculate_drawdowns


@dataclass(frozen=True)
class DashboardAnalysis:
    market_data: MarketDataResult
    returns: ReturnAnalysis
    risk: RiskAnalysis
    frontier: EfficientFrontier
    minimum_variance: PortfolioSolution
    maximum_sharpe: PortfolioSolution
    backtest: BacktestResult
    notices: tuple[SafeguardNotice, ...]
    data_source: str


st.set_page_config(
    page_title="Portfolio Optimisation Dashboard", page_icon="📈", layout="wide"
)
st.title("Portfolio Optimisation & Investment Analytics")
st.caption(
    "Explore historical diversification, constrained portfolio construction, "
    "and genuinely out-of-sample strategy performance."
)
st.warning(
    "Educational analysis only — historical results are not forecasts or "
    "investment advice."
)


def _parse_tickers(value: str) -> list[str]:
    return [ticker.strip().upper() for ticker in value.split(",") if ticker.strip()]


def _parse_custom_weights(
    value: str, assets: list[str]
) -> dict[str, float] | None:
    if not value.strip():
        return None
    try:
        percentages = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("Custom weights must be comma-separated numbers.") from exc
    if len(percentages) != len(assets):
        raise ValueError("Provide one custom weight for every selected asset.")
    return dict(zip(assets, (percentage / 100.0 for percentage in percentages)))


def _sidebar_inputs() -> dict[str, object] | None:
    st.sidebar.header("Analysis settings")
    with st.sidebar.form("analysis_form"):
        ticker_text = st.text_input(
            "Assets (comma separated)", value="AAPL, MSFT, JPM, XOM, GLD"
        )
        benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
        today = date.today()
        start = st.date_input("Start date", value=today - timedelta(days=365 * 5))
        end = st.date_input("End date", value=today)
        frequency = st.selectbox("Return frequency", ["daily", "weekly", "monthly"])
        risk_free_percent = st.number_input(
            "Annual risk-free rate (%)", value=4.0, step=0.25
        )
        maximum_weight_percent = st.slider(
            "Maximum asset weight (%)", min_value=10, max_value=100, value=40
        )
        confidence_percent = st.selectbox("VaR confidence (%)", [95, 99])
        initial_investment = st.number_input(
            "Initial investment", min_value=1.0, value=100_000.0, step=10_000.0
        )
        custom_weight_text = st.text_input(
            "Custom weights (%) — optional",
            placeholder="20, 20, 20, 20, 20",
            help="Enter one percentage for each asset, in the same order.",
        )
        demo_mode = st.checkbox(
            "Use reproducible demo data",
            value=True,
            help="Turn off to download adjusted prices from Yahoo Finance.",
        )
        submitted = st.form_submit_button("Run analysis", type="primary")
    if not submitted:
        return None
    assets = _parse_tickers(ticker_text)
    return {
        "assets": assets,
        "benchmark": benchmark,
        "start": start,
        "end": end,
        "frequency": frequency,
        "risk_free_rate": risk_free_percent / 100.0,
        "maximum_weight": maximum_weight_percent / 100.0,
        "confidence_level": confidence_percent / 100.0,
        "initial_investment": initial_investment,
        "custom_weight_text": custom_weight_text,
        "demo_mode": demo_mode,
    }


def _run_analysis(inputs: dict[str, object]) -> DashboardAnalysis:
    custom_weights = _parse_custom_weights(
        inputs["custom_weight_text"], inputs["assets"]
    )
    request = create_request(
        inputs["assets"],
        inputs["benchmark"],
        inputs["start"],
        inputs["end"],
        inputs["frequency"],
    )
    if inputs["demo_mode"]:
        market_data = load_market_data(
            request, downloader=make_demo_downloader(), use_cache=False
        )
        data_source = "Deterministic demonstration data"
    else:
        market_data = load_market_data(request)
        data_source = "Yahoo Finance via yfinance"
    returns = analyse_returns(market_data.prices, market_data.periods_per_year)
    risk = analyse_risk(
        returns.simple_returns,
        market_data.benchmark,
        market_data.periods_per_year,
        risk_free_rate=inputs["risk_free_rate"],
        confidence_level=inputs["confidence_level"],
    )
    labels = list(market_data.assets)
    asset_returns = returns.annualised_expected_returns.loc[labels]
    covariance = returns.annualised_covariance.loc[labels, labels]
    minimum_variance = optimise_minimum_variance(
        asset_returns, covariance, maximum_weight=inputs["maximum_weight"]
    )
    maximum_sharpe = optimise_maximum_sharpe(
        asset_returns,
        covariance,
        risk_free_rate=inputs["risk_free_rate"],
        maximum_weight=inputs["maximum_weight"],
    )
    frontier = calculate_efficient_frontier(
        asset_returns,
        covariance,
        maximum_weight=inputs["maximum_weight"],
        risk_free_rate=inputs["risk_free_rate"],
        number_of_points=30,
    )
    backtest = run_backtest(
        market_data.prices,
        labels,
        market_data.benchmark,
        market_data.periods_per_year,
        risk_free_rate=inputs["risk_free_rate"],
        maximum_weight=inputs["maximum_weight"],
        confidence_level=inputs["confidence_level"],
        initial_investment=inputs["initial_investment"],
        custom_weights=custom_weights,
    )
    notices = assess_model_safeguards(
        market_data.prices,
        backtest.weights,
        periods_per_year=market_data.periods_per_year,
        in_sample_observations=len(backtest.in_sample_returns),
        out_of_sample_observations=len(backtest.strategy_returns),
        maximum_weight=inputs["maximum_weight"],
        cache_warning=market_data.cache_warning,
    )
    return DashboardAnalysis(
        market_data,
        returns,
        risk,
        frontier,
        minimum_variance,
        maximum_sharpe,
        backtest,
        notices,
        data_source,
    )


def _format_metrics(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    percentages = {
        "annualised_return",
        "annualised_volatility",
        "maximum_drawdown",
        "historical_var",
        "historical_expected_shortfall",
        "cumulative_return",
        "cagr",
    }
    formats = {column: "{:.2%}" for column in frame.columns if column in percentages}
    formats.update(
        {
            column: "{:,.2f}"
            for column in frame.columns
            if column in {"initial_value", "final_value"}
        }
    )
    return frame.style.format(formats, na_rep="—")


def _render_dashboard(analysis: DashboardAnalysis) -> None:
    market = analysis.market_data
    backtest = analysis.backtest
    warnings = [notice for notice in analysis.notices if notice.severity == "warning"]
    if warnings:
        with st.expander("Model-risk warnings", expanded=True):
            for notice in warnings:
                st.warning(notice.message)
    st.caption(
        f"Source: {analysis.data_source} · Last aligned observation: "
        f"{market.prices.index[-1].date()} · "
        f"Data source: {'local cache' if market.from_cache else 'fresh download'}"
    )
    overview, assets, optimisation, backtesting, methodology = st.tabs(
        ["Overview", "Asset Analysis", "Optimisation", "Backtest", "Methodology"]
    )
    with overview:
        columns = st.columns(4)
        columns[0].metric("Assets", len(market.assets))
        columns[1].metric("Aligned observations", len(market.prices))
        columns[2].metric("Frequency", market.frequency.title())
        columns[3].metric("Backtest split", backtest.split_date.date().isoformat())
        st.plotly_chart(normalised_price_chart(market.prices), width="stretch")
        st.plotly_chart(
            cumulative_return_chart(analysis.returns.cumulative_returns),
            width="stretch",
        )
    with assets:
        st.subheader("Asset risk and return")
        st.dataframe(_format_metrics(analysis.risk.metrics), width="stretch")
        left, right = st.columns(2)
        left.plotly_chart(
            correlation_heatmap(analysis.risk.correlations), width="stretch"
        )
        right.plotly_chart(
            drawdown_chart(analysis.risk.drawdowns, "Asset Drawdowns"),
            width="stretch",
        )
    with optimisation:
        labels = list(market.assets)
        asset_returns = analysis.returns.annualised_expected_returns.loc[labels]
        asset_volatility = analysis.returns.simple_returns[labels].std() * (
            market.periods_per_year**0.5
        )
        st.plotly_chart(
            efficient_frontier_chart(
                analysis.frontier,
                asset_returns,
                asset_volatility,
                analysis.minimum_variance,
                analysis.maximum_sharpe,
            ),
            width="stretch",
        )
        st.plotly_chart(allocation_chart(backtest.weights), width="stretch")
        st.dataframe(
            (backtest.weights * 100).style.format("{:.2f}%"),
            width="stretch",
        )
        st.dataframe(
            _format_metrics(backtest.in_sample_statistics), width="stretch"
        )
    with backtesting:
        st.info(
            "Weights use only the first 70% of returns; the final 30% is reserved "
            "for evaluation."
        )
        st.plotly_chart(
            portfolio_growth_chart(backtest.portfolio_values), width="stretch"
        )
        st.plotly_chart(
            drawdown_chart(
                calculate_drawdowns(backtest.strategy_returns),
                "Out-of-Sample Drawdowns",
            ),
            width="stretch",
        )
        st.dataframe(
            _format_metrics(backtest.out_of_sample_statistics),
            width="stretch",
        )
    with methodology:
        st.markdown(
            """
            ### Methodology
            - Adjusted historical prices are aligned across all selected series.
            - Returns and sample covariance are annualised for the selected frequency.
            - Portfolios are long-only, fully invested, and concentration constrained.
            - VaR and Expected Shortfall are positive historical one-period losses.
            - The backtest is constant weight and excludes taxes, costs, spreads,
              currency conversion, and market impact.

            ### Limitations
            Historical estimates are sample-dependent and unstable. The optimiser
            does not forecast returns, and one 70/30 split cannot establish future
            performance.

            **Educational use only. This is not investment advice.**
            """
        )


inputs = _sidebar_inputs()
if inputs is not None:
    try:
        with st.spinner("Downloading data and running portfolio analysis..."):
            st.session_state["dashboard_analysis"] = _run_analysis(inputs)
    except (
        MarketDataError,
        ReturnCalculationError,
        RiskCalculationError,
        OptimisationError,
        PortfolioComparisonError,
        BacktestError,
        ValueError,
    ) as exc:
        st.error(str(exc))

if "dashboard_analysis" in st.session_state:
    _render_dashboard(st.session_state["dashboard_analysis"])
else:
    st.info("Choose settings in the sidebar and select **Run analysis**.")

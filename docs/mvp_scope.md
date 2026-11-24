# Portfolio Optimisation & Investment Analytics Dashboard

## MVP Scope (Version 1.0)

### 1. Product objective

Build an interactive dashboard that allows a user to select a small portfolio of publicly traded assets, evaluate their historical risk and return characteristics, construct optimised portfolios, and compare those portfolios with a benchmark using an out-of-sample backtest.

The MVP is intended to demonstrate practical knowledge of portfolio theory, quantitative investment analysis, Python, and clear financial communication. It is an educational analytical tool, not an investment recommendation engine.

### 2. Target user

The primary user is an investment analyst, finance student, or informed individual investor who wants to answer:

> Given a selected group of assets, what allocations would historically have produced the lowest volatility or highest Sharpe ratio, and how would those allocations have performed against a benchmark in a subsequent period?

### 3. Core user journey

1. The user selects between 3 and 10 valid ticker symbols.
2. The user selects a benchmark, analysis dates, return frequency, risk-free rate, and initial investment amount.
3. The application downloads and validates adjusted historical prices.
4. The application presents asset-level performance and risk analytics.
5. The application divides the available history into an in-sample period and an out-of-sample period.
6. The application constructs minimum-variance and maximum-Sharpe portfolios using only in-sample data.
7. The application compares those portfolios with an equal-weight portfolio and the benchmark during the out-of-sample period.
8. The user reviews allocation, performance, drawdown, and risk charts together with documented assumptions and limitations.

### 4. Required user inputs

| Input | MVP rule | Default |
|---|---|---|
| Assets | 3-10 unique valid tickers | AAPL, MSFT, JPM, XOM, GLD |
| Benchmark | One valid ticker, not required to be in the portfolio | SPY |
| Start date | Must leave enough data for estimation and testing | Five years before end date |
| End date | Must be after start date | Latest available date |
| Return frequency | Daily, weekly, or monthly | Daily |
| Risk-free rate | Annual percentage, entered by the user | 4.0% |
| Initial investment | Positive monetary value | 100,000 |
| Maximum asset weight | Between 10% and 100% and feasible for the number of assets | 40% |
| VaR confidence level | 95% or 99% | 95% |

The base currency is not converted in the MVP. All selected assets and the benchmark should therefore be economically comparable in a common currency.

### 5. Required analytics

#### Asset-level analytics

For every selected asset and the benchmark, calculate:

- Annualised historical return
- Annualised volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Historical Value at Risk (VaR)
- Historical Expected Shortfall (CVaR)
- Beta relative to the selected benchmark
- Pairwise return correlations

#### Portfolio construction

Construct and report:

- Equal-weight portfolio
- Long-only minimum-variance portfolio
- Long-only maximum-Sharpe portfolio

All optimised portfolios must satisfy:

- Weights sum to 100% within numerical tolerance.
- No short positions.
- No weight exceeds the user-selected maximum.
- Optimisation uses in-sample observations only.

#### Out-of-sample comparison

Use the first 70% of aligned observations as the in-sample period and the final 30% as the out-of-sample period. Hold the calculated weights constant through the MVP backtest.

For each portfolio and the benchmark, report:

- Initial and final value
- Cumulative return
- Compound annual growth rate (CAGR)
- Annualised volatility
- Sharpe ratio
- Maximum drawdown
- Historical VaR
- Historical Expected Shortfall

### 6. Required dashboard outputs

The MVP must provide:

1. A normalised asset-price chart.
2. A cumulative-return chart.
3. An asset-metrics table.
4. A correlation heatmap.
5. An efficient-frontier chart showing the selected assets, simulated portfolios for context, the minimum-variance portfolio, and the maximum-Sharpe portfolio.
6. A portfolio-weight table and allocation chart.
7. An out-of-sample portfolio-growth chart against the benchmark.
8. A drawdown chart.
9. A strategy comparison table.
10. A methodology and limitations section.

### 7. Error handling and validation

The application must display a clear user-facing message instead of failing when:

- A ticker is invalid or returns no data.
- A ticker has insufficient history.
- Fewer than three valid assets remain.
- Dates are invalid or provide insufficient observations.
- Asset histories do not sufficiently overlap.
- The maximum-weight constraint makes optimisation infeasible.
- Prices or calculated returns contain unusable missing or infinite values.
- The optimiser cannot find a valid solution.

The application must not silently replace, drop, or reinterpret a user-selected asset.

### 8. MVP assumptions

- Historical returns and covariances are estimates, not forecasts.
- Adjusted closing prices are used when available.
- Distributions are not assumed to be normal for historical VaR and Expected Shortfall.
- No short selling or leverage is permitted.
- Portfolio weights remain constant during the out-of-sample MVP backtest.
- Taxes, transaction costs, bid-ask spreads, market impact, and currency conversion are excluded.
- Fractional holdings are assumed, so portfolio weights can be applied exactly.
- Data quality depends on the selected market-data source.

### 9. Explicitly out of scope for Version 1.0

- Return forecasting or price prediction
- Machine-learning models
- Black-Litterman allocation
- Factor models
- Short selling and leveraged portfolios
- Live prices or intraday trading
- Brokerage integration or trade execution
- User accounts or saved portfolios
- Tax optimisation
- Currency hedging or foreign-exchange conversion
- Automated security recommendations
- Rolling or event-driven rebalancing
- Transaction-cost modelling
- Production-grade institutional market data

These may be considered only after the MVP is tested and documented.

### 10. Acceptance criteria

The MVP is complete when all of the following are true:

- A user can analyse 3-10 valid assets without editing source code.
- Inputs are validated before analytics or optimisation begins.
- Every required asset and portfolio metric is displayed.
- Minimum-variance and maximum-Sharpe weights satisfy all constraints.
- The efficient frontier renders and identifies both optimised portfolios.
- The backtest uses no out-of-sample observations to estimate portfolio weights.
- Equal-weight, minimum-variance, maximum-Sharpe, and benchmark results are comparable over identical dates.
- Invalid data and infeasible constraints produce understandable messages.
- Core calculations and optimisation constraints are covered by automated tests.
- The methodology, assumptions, limitations, and educational-use disclaimer are visible in the application.

### 11. MVP demonstration case

The initial demonstration will use:

- Assets: AAPL, MSFT, JPM, XOM, GLD
- Benchmark: SPY
- History: five years, subject to data availability
- Return frequency: daily
- Risk-free rate: 4.0% annually
- Maximum asset weight: 40%
- Starting portfolio value: 100,000
- Split: 70% in sample and 30% out of sample

The demonstration should answer whether the optimised portfolios improved historical diversification or risk-adjusted performance relative to an equal-weight portfolio and the S&P 500 benchmark, while explicitly noting that the evidence is historical and sample-dependent.

### 12. Success standard

The project succeeds when a reviewer can understand the investment question, reproduce the analysis, verify that the backtest avoids look-ahead bias, and explain the differences between equal-weight, minimum-variance, and maximum-Sharpe allocations.

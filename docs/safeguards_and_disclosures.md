# Safeguards and Disclosures

The application validates inputs at module boundaries and presents analytical
limitations alongside results rather than relying on a disclaimer hidden in the
README.

## Operational safeguards

- Invalid tickers, dates, frequencies, confidence levels, risk-free rates, and
  investment values are rejected with user-facing messages.
- Missing, non-finite, non-positive, insufficient, or non-overlapping price series
  cannot enter the analytical pipeline.
- A cache read failure triggers a fresh download. A cache write failure does not
  discard otherwise valid downloaded prices and is disclosed to the user.
- Infeasible maximum-weight constraints are rejected before optimisation.
- Solver outputs are independently checked for finite values, full investment,
  weight limits, and target-return accuracy.
- Custom portfolios require all selected assets, non-negative weights, a 100% total,
  and compliance with the selected concentration limit.
- Backtests require viable in-sample covariance estimation and a separate
  out-of-sample evaluation period.

## Dynamic model-risk warnings

The dashboard warns when:

- Total aligned history is shorter than two years at the selected frequency.
- The estimation window is shorter than one year.
- The evaluation sample is too short for stable tail and drawdown statistics.
- A portfolio reaches the maximum asset-weight constraint.
- Market data could not be cached.

## Persistent disclosures

Every completed analysis carries these disclosures:

- Historical returns, covariance, VaR, and Expected Shortfall are not forecasts.
- The backtest excludes costs, taxes, spreads, market impact, and currency effects.
- The tool is educational and does not provide investment advice.

The dashboard also displays the market-data source, last aligned observation date,
and whether the result used a fresh download or local cache.

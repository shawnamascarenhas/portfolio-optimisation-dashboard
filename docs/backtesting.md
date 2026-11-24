# Out-of-Sample Backtesting

The MVP backtest separates portfolio construction from performance evaluation to
avoid look-ahead bias.

## Chronological split

1. Prices are converted into aligned simple returns.
2. The first 70% of return observations form the in-sample period.
3. Annualised expected returns and covariance are estimated using only that period.
4. Equal-weight, minimum-variance, maximum-Sharpe, and optional custom weights are
   fixed before the out-of-sample period begins.
5. The final 30% of observations are used exclusively for evaluation.

The first out-of-sample return includes the price movement from the last in-sample
price date to the first out-of-sample price date. No out-of-sample return is used to
estimate an optimised weight.

## Portfolio-return convention

Each strategy return is the weighted sum of contemporaneous asset returns. This is
an idealised constant-weight portfolio, equivalent to restoring target weights at
each selected observation frequency. Transaction costs, spreads, taxes, and market
impact are excluded from the MVP and this limitation must be disclosed in the
dashboard.

## Realised outputs

Every strategy and the benchmark are evaluated over identical dates using:

- Initial and final portfolio value
- Cumulative compounded return
- CAGR based on the number of observations and annualisation factor
- Annualised arithmetic return
- Annualised volatility
- Sharpe ratio
- Maximum drawdown
- Historical one-period VaR
- Historical one-period Expected Shortfall

The displayed backtest is historical evidence, not a forecast. A single 70/30 split
can be sensitive to the selected start and end dates. Rolling or walk-forward tests,
transaction costs, and lower-frequency rebalancing are potential later extensions.

## Leakage control

The automated tests modify only the out-of-sample price path and verify that the
optimised weights remain exactly unchanged. This is a direct regression test for
the central no-look-ahead requirement.

# Model Validation

The automated suite combines unit, analytical benchmark, integration, and
regression tests. Network access is not required: external market-data responses
are replaced by deterministic provider fixtures.

## Validation matrix

| Area | Independent validation |
|---|---|
| Market data | Ticker/date validation, adjusted-close extraction, alignment, resampling, caching, missing histories, non-positive prices |
| Returns | Exact price relatives, log-return identity, compounded cumulative return, annualised arithmetic mean, sample covariance |
| Risk | Hand-calculated volatility, drawdown, beta, VaR, Expected Shortfall, downside deviation, zero-denominator behaviour |
| Minimum variance | Numerical weights compared with the closed-form two-asset solution |
| Maximum Sharpe | Numerical weights compared with the inverse-covariance tangency solution where all analytical weights are long only |
| Constraints | Weight sum, lower bound, maximum weight, target return, label alignment, infeasible constraint rejection |
| Backtest | Chronological 70/30 boundary, equal evaluation dates, compounding identity, and future-data leakage regression |
| Dashboard | Plotly figure construction, rebasing, correlation scale, and required frontier trace groups |
| Integration | Deterministic prices pass through data, returns, risk, optimisation, and backtesting without label or date drift |

## Numerical tolerances

- Closed-form portfolio weights: absolute tolerance of `1e-6` or tighter
- Full-investment checks: absolute tolerance of `1e-7`
- Optimiser bounds: tolerance of `1e-7`
- Target returns: absolute tolerance of `1e-6`
- Matrix symmetry: absolute tolerance of `1e-10`
- Positive-semidefinite covariance: minimum eigenvalue no lower than `-1e-10`

These tolerances accommodate floating-point and numerical-solver behaviour without
permitting financially material constraint violations.

## Run validation

```bash
pytest
ruff check .
pytest --cov=portfolio_analytics --cov-report=term-missing
```

The configured coverage threshold is 85%. Coverage is a supporting control rather
than proof of model correctness; known analytical results and invariant tests are
the primary financial validation evidence.

## Remaining validation limitations

- Yahoo Finance availability and vendor corrections cannot be validated offline.
- Historical results remain sensitive to the chosen assets and dates.
- Transaction costs, taxes, currency conversion, and execution effects are outside
  the MVP and therefore outside its validation scope.
- The optimiser is validated for the stated long-only constraints, not for short,
  leveraged, or mixed-integer portfolios.

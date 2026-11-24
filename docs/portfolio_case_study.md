# Portfolio Case Study: Diversified Multi-Asset Allocation

## Investment question

How do equal-weight, minimum-variance, and maximum-Sharpe allocations behave when
constructed from the same five-asset universe, and do their modelled in-sample
advantages persist in a later out-of-sample period?

## Demonstration setup

This case study uses the application's deterministic demonstration mode so every
reviewer receives exactly the same result without relying on a live data vendor.
The series are synthetic and labelled as familiar securities solely to make the
workflow intuitive; they are not actual security histories.

| Assumption | Value |
|---|---|
| Asset labels | AAPL, MSFT, JPM, XOM, GLD |
| Benchmark label | SPY |
| Analysis dates | 29 September 2021 to 28 September 2026 |
| Frequency | Daily |
| Risk-free rate | 4.0% annually |
| Maximum asset weight | 40% |
| Initial investment | 100,000 |
| In-sample share | 70% |
| Out-of-sample share | 30% |

The aligned dataset contains 1,304 price observations. The first 912 returns are
used for estimation and the final 391 returns, beginning 31 March 2025, are reserved
for evaluation.

## Modelled allocations

| Strategy | AAPL | MSFT | JPM | XOM | GLD |
|---|---:|---:|---:|---:|---:|
| Equal Weight | 20.00% | 20.00% | 20.00% | 20.00% | 20.00% |
| Minimum Variance | 40.00% | 40.00% | 19.74% | 0.26% | 0.00% |
| Maximum Sharpe | 40.00% | 0.00% | 20.00% | 0.00% | 40.00% |

Both optimised portfolios reach the 40% concentration limit. This is an important
model-risk signal: small changes in the estimated inputs or constraint could change
the solution materially.

## Out-of-sample result

| Strategy | Final value | CAGR | Volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| Equal Weight | 99,082 | -0.59% | 13.06% | -0.29 | -17.02% |
| Minimum Variance | 94,200 | -3.78% | 11.19% | -0.65 | -16.85% |
| Maximum Sharpe | 103,321 | 2.13% | 13.33% | -0.08 | -18.23% |
| Benchmark | 106,846 | 4.36% | 12.95% | 0.09 | -13.21% |

## Interpretation

The minimum-variance portfolio delivered the lowest realised volatility, consistent
with its objective, but its lower volatility did not produce the best return or
Sharpe ratio. The maximum-Sharpe portfolio preserved capital better than the other
constructed strategies but still underperformed the synthetic benchmark. Its
maximum drawdown was also the largest of the four series.

This distinction is central to the project: optimisation solves a problem using
estimated historical inputs; it does not guarantee that the same relationship will
hold in a later sample. The out-of-sample results therefore provide more useful
evidence than the in-sample objective value alone.

## Investment conclusion

For this synthetic case, none of the optimised portfolios dominated the benchmark
across return, volatility, Sharpe ratio, and drawdown. The minimum-variance strategy
achieved its narrow volatility objective, while the maximum-Sharpe strategy produced
the strongest result among constructed portfolios. A decision-maker should treat
the solutions as allocation diagnostics rather than recommended portfolios.

## Limitations

- Demonstration prices are synthetic and cannot support a real investment decision.
- Arithmetic historical means are unstable estimates of expected return.
- One chronological split is sensitive to the selected dates.
- Constant-weight portfolios imply idealised rebalancing at every observation.
- Transaction costs, taxes, spreads, market impact, and currency effects are absent.
- The 40% cap is a modelling choice rather than an investor-specific mandate.

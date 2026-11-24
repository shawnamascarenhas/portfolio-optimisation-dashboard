# Risk Analytics Methodology

The risk engine consumes aligned simple returns and produces an asset-level metrics
table, return-correlation matrix, and complete drawdown histories.

## Metric conventions

| Metric | MVP convention |
|---|---|
| Annualised return | Arithmetic mean period return multiplied by observations per year |
| Annualised volatility | Sample standard deviation multiplied by the square root of observations per year |
| Sharpe ratio | Annualised return less annual risk-free rate, divided by annualised volatility |
| Downside deviation | Root mean squared return shortfall below the equivalent compounded period target, annualised |
| Sortino ratio | Annualised return less annual risk-free rate, divided by annualised downside deviation |
| Maximum drawdown | Largest percentage fall in compounded wealth from its previous peak |
| Historical VaR | Positive one-period loss at the selected empirical lower-tail quantile |
| Expected Shortfall | Positive average one-period loss at or below the VaR return cutoff |
| Beta | Sample covariance with benchmark divided by benchmark sample variance |
| Correlation | Pearson correlation of aligned simple returns |

The risk-free rate is entered as an annual decimal. For downside deviation, it is
converted to an equivalent compounded period rate using
`(1 + annual_rate) ** (1 / periods_per_year) - 1`.

## VaR and Expected Shortfall interpretation

VaR and Expected Shortfall are returned as positive fractions of capital. A value
of `0.025` represents a 2.5% loss, not a return of -2.5%. The measures are
historical and non-parametric: the engine does not impose a normal distribution.

If the relevant historical lower tail contains only gains, the reported loss is
zero. These are one-period measures at the return frequency selected by the user.

## Undefined ratios

A Sharpe or Sortino ratio is represented by `NaN` when its denominator is zero.
This is preferable to displaying infinity for a constant or no-downside series.
The benchmark must have positive sample variance so beta is economically defined.

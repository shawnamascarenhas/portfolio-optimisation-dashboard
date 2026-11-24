# Portfolio Strategy Comparison

The comparison layer constructs all portfolio strategies from identical annualised
expected-return and covariance inputs.

## Strategies

| Strategy | Construction |
|---|---|
| Equal Weight | Allocates `1 / number_of_assets` to every selected asset |
| Minimum Variance | Uses the constrained minimum-variance solution |
| Maximum Sharpe | Uses the constrained maximum-Sharpe solution |
| Custom | Uses the allocation entered by the user |

Custom weights are entered as decimal fractions, must include every selected asset,
must be non-negative, must sum to 1.0 (100%), and must respect the same maximum
asset-weight setting used by the optimised portfolios.

## Comparison outputs

The raw allocation table retains decimal weights for further calculations. A
separate presentation helper converts them into percentage points for display.
Long-form allocation data is also available for grouped bar or faceted charts.

Each strategy reports:

- Annualised expected return
- Annualised volatility
- Sharpe ratio
- Largest individual asset weight
- Effective holdings

Effective holdings is the inverse Herfindahl concentration index:

`1 / sum(weight²)`

It equals the number of assets for an equal-weight portfolio and falls as an
allocation becomes more concentrated. These figures are modelled from the same
historical inputs; they are not out-of-sample results. Realised performance will be
added by the backtesting layer.

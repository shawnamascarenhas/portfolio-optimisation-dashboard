# Portfolio Optimisation Methodology

The optimisation engine consumes annualised arithmetic expected returns and an
annualised sample covariance matrix from the return engine.

## Constraints

Every MVP portfolio is:

- Fully invested: weights sum to 100%
- Long only: every weight is at least zero
- Concentration constrained: no weight exceeds the user-selected maximum

A maximum-weight selection is feasible only when `number_of_assets * maximum_weight
>= 1`. Infeasible selections are rejected before calling the numerical solver.

## Minimum-variance portfolio

The minimum-variance portfolio solves:

`minimise w'Σw`

subject to the three constraints above. Its volatility is the square root of the
optimised annual variance.

## Maximum-Sharpe portfolio

The maximum-Sharpe portfolio minimises the negative value of:

`(w'μ - risk_free_rate) / sqrt(w'Σw)`

The expected returns, covariance matrix, and risk-free rate must all use annual
units.

## Efficient frontier

The displayed frontier is the upper, non-dominated branch. It begins at the
minimum-variance portfolio and ends at the highest-return portfolio permitted by
the same weight constraints. For evenly spaced target returns in that range, the
engine minimises variance subject to an additional target-return equality.

The implementation uses SciPy's Sequential Least Squares Programming (SLSQP)
solver. Every result is independently checked for finite values, full investment,
weight bounds, and target-return accuracy before it is accepted.

## Input controls

The covariance matrix must:

- Be square, finite, symmetric, and positive semidefinite
- Have positive individual asset variances
- Use exactly the same asset labels as expected returns

Numerical optimisation can be sensitive to historical estimates. The resulting
weights describe the solution to the supplied model inputs, not a forecast or an
investment recommendation.

# Return Calculations

The return engine transforms the aligned price table from the data layer into the
inputs required for risk analytics and mean-variance optimisation.

## Definitions

For asset price \(P_t\):

- Simple return: \(r_t = P_t / P_{t-1} - 1\)
- Log return: \(g_t = \ln(P_t / P_{t-1})\)
- Cumulative return: \(\prod_{i=1}^{t}(1+r_i)-1\)
- Annualised expected return: \(E[r]_{annual} = \bar{r}N\)
- Annualised covariance: \(\Sigma_{annual} = \Sigma_{period}N\)

Here, \(N\) is 252 for daily, 52 for weekly, or 12 for monthly observations.

## Method choice

The optimiser will use the annualised arithmetic mean of simple returns and the
annualised sample covariance matrix. Arithmetic means are internally consistent
with single-period mean-variance optimisation. Cumulative performance compounds
actual period returns and is therefore not calculated by summing returns.

Log returns are retained for analysis and diagnostics, but they are not mixed with
the simple-return inputs used for portfolio wealth and optimisation.

## Validation contract

The engine requires:

- At least two chronological price observations
- Unique asset columns and dates
- Finite, strictly positive prices
- Complete aligned observations
- A positive integer annualisation factor

It returns new objects and does not mutate the price table supplied by the data
layer.

# Streamlit Dashboard

The dashboard connects every analytical module through one user workflow.

Users may optionally enter one custom percentage weight per selected asset, in the
same order as the ticker list. The allocation is validated against the 100% total
and maximum-weight constraint before analysis proceeds.

## Run locally

After installing the project and its dependencies:

```bash
streamlit run app.py
```

## Sections

- **Overview:** universe, aligned observations, frequency, split date, normalised
  prices, and cumulative returns.
- **Asset Analysis:** risk metrics, correlation heatmap, and asset drawdowns.
- **Optimisation:** efficient frontier, optimal portfolios, allocation chart,
  weights, and modelled in-sample statistics.
- **Backtest:** 70/30 methodology notice, out-of-sample growth, drawdowns, and
  realised performance comparison.
- **Methodology:** assumptions, exclusions, limitations, and disclaimer.

## Error handling

Known input, provider, return, risk, optimisation, comparison, and backtesting
errors are shown as user-facing messages. A failed run does not remove the most
recent successful analysis from the session.

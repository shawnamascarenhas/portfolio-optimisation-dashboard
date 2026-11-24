# Historical Market Data Layer

`portfolio_analytics.data_loader` converts a user request into a validated and
aligned adjusted-price table for the analytics engine.

## Public workflow

```python
from portfolio_analytics import create_request, load_market_data

request = create_request(
    assets=["AAPL", "MSFT", "JPM", "XOM", "GLD"],
    benchmark="SPY",
    start="2021-01-01",
    end="2025-12-31",
    frequency="daily",
)
result = load_market_data(request)
prices = result.prices
```

## Behaviour

- Tickers are trimmed, converted to uppercase, and validated.
- A request must contain 3-10 unique investable assets.
- End dates are inclusive at the product boundary.
- Yahoo Finance adjusted close is preferred; close is used only if adjusted close
  is unavailable.
- All series are aligned to common observations.
- Weekly observations use the last available price in a Friday-ending week.
- Monthly observations use the last available price in the calendar month.
- Non-numeric, infinite, missing, or non-positive observations cannot reach the
  final aligned table.
- Empty, missing, or insufficient histories raise explicit user-facing errors.
- Raw adjusted prices are cached under `data/cache/` using a request-specific key.

## Minimum aligned history

| Frequency | Minimum observations | Annualisation factor |
|---|---:|---:|
| Daily | 60 | 252 |
| Weekly | 26 | 52 |
| Monthly | 12 | 12 |

These are data-quality floors, not recommended investment-estimation windows. The
five-year demonstration case provides a materially larger estimation sample.

## Testing

Tests inject a deterministic provider response. This verifies validation,
adjusted-close selection, resampling, caching, missing-ticker handling, and history
requirements without relying on Yahoo Finance availability or an internet
connection.

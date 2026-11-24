"""Deterministic synthetic prices for demonstrations and offline review."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def make_demo_downloader(seed: int = 20260928) -> Callable[..., pd.DataFrame]:
    """Return a yfinance-compatible deterministic price downloader."""

    def download(
        tickers: Sequence[str],
        *,
        start: str,
        end: str,
        **kwargs,
    ) -> pd.DataFrame:
        symbols = list(dict.fromkeys(tickers))
        final_date = pd.Timestamp(end) - pd.Timedelta(days=1)
        dates = pd.bdate_range(start=start, end=final_date)
        if len(dates) < 2:
            return pd.DataFrame()

        rng = np.random.default_rng(seed)
        observations = len(dates) - 1
        market_factor = rng.normal(0.00030, 0.0090, size=observations)
        return_matrix = np.empty((observations, len(symbols)), dtype=float)
        for position, _symbol in enumerate(symbols):
            beta = 0.65 + 0.12 * (position % 5)
            daily_alpha = -0.00002 + 0.000035 * (position % 6)
            idiosyncratic_volatility = 0.0045 + 0.0012 * (position % 4)
            idiosyncratic = rng.normal(
                daily_alpha, idiosyncratic_volatility, size=observations
            )
            return_matrix[:, position] = beta * market_factor + idiosyncratic

        prices = np.vstack(
            [
                np.full(len(symbols), 100.0),
                100.0 * np.cumprod(1.0 + return_matrix, axis=0),
            ]
        )
        return pd.DataFrame(prices, index=dates, columns=symbols)

    return download

"""Historical market-price retrieval and validation.

The module keeps provider-specific behaviour at the boundary.  Downstream analytics
receive one clean, aligned price table in the requested observation frequency.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Frequency = Literal["daily", "weekly", "monthly"]
DownloadFunction = Callable[..., pd.DataFrame]

PERIODS_PER_YEAR: dict[Frequency, int] = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
}

MINIMUM_OBSERVATIONS: dict[Frequency, int] = {
    "daily": 60,
    "weekly": 26,
    "monthly": 12,
}

_TICKER_PATTERN = re.compile(r"^[A-Z0-9.^=-]+$")


class MarketDataError(RuntimeError):
    """Base exception for market-data failures."""


class RequestValidationError(MarketDataError):
    """Raised when a market-data request is invalid."""


class PriceDataError(MarketDataError):
    """Raised when downloaded prices are missing or unusable."""


@dataclass(frozen=True)
class MarketDataRequest:
    """Validated inputs for a historical market-data request."""

    assets: tuple[str, ...]
    benchmark: str
    start: date
    end: date
    frequency: Frequency = "daily"

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return assets plus the benchmark without duplicate downloads."""
        return tuple(dict.fromkeys((*self.assets, self.benchmark)))


@dataclass(frozen=True)
class MarketDataResult:
    """Clean output consumed by analytics and portfolio construction."""

    prices: pd.DataFrame
    assets: tuple[str, ...]
    benchmark: str
    frequency: Frequency
    periods_per_year: int
    from_cache: bool
    cache_warning: str | None = None

    @property
    def asset_prices(self) -> pd.DataFrame:
        """Return only investable asset prices in the user's original order."""
        return self.prices.loc[:, list(self.assets)].copy()

    @property
    def benchmark_prices(self) -> pd.Series:
        """Return the benchmark price series."""
        return self.prices.loc[:, self.benchmark].copy()


def create_request(
    assets: Sequence[str],
    benchmark: str,
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp,
    frequency: Frequency = "daily",
) -> MarketDataRequest:
    """Normalise and validate user inputs before contacting a data provider."""
    normalised_assets = tuple(_normalise_ticker(ticker) for ticker in assets)
    if len(normalised_assets) < 3 or len(normalised_assets) > 10:
        raise RequestValidationError("Select between 3 and 10 assets.")
    if len(set(normalised_assets)) != len(normalised_assets):
        raise RequestValidationError("Asset tickers must be unique.")

    normalised_benchmark = _normalise_ticker(benchmark)
    if frequency not in PERIODS_PER_YEAR:
        choices = ", ".join(PERIODS_PER_YEAR)
        raise RequestValidationError(f"Frequency must be one of: {choices}.")

    start_date = _normalise_date(start, "start")
    end_date = _normalise_date(end, "end")
    if start_date >= end_date:
        raise RequestValidationError("Start date must be earlier than end date.")
    if end_date > date.today():
        raise RequestValidationError("End date cannot be in the future.")

    return MarketDataRequest(
        assets=normalised_assets,
        benchmark=normalised_benchmark,
        start=start_date,
        end=end_date,
        frequency=frequency,
    )


def load_market_data(
    request: MarketDataRequest,
    *,
    cache_dir: str | Path = "data/cache",
    downloader: DownloadFunction | None = None,
    use_cache: bool = True,
    minimum_observations: int | None = None,
) -> MarketDataResult:
    """Download, cache, align, and validate adjusted historical prices.

    The provider is injectable to make tests deterministic.  Yahoo Finance is loaded
    lazily when no custom downloader is supplied.
    """
    cache_path = Path(cache_dir) / _cache_filename(request)
    raw_prices: pd.DataFrame
    from_cache = False
    cache_warning: str | None = None

    if use_cache and cache_path.exists():
        try:
            raw_prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            from_cache = True
        except (OSError, ValueError, pd.errors.ParserError):
            raw_prices = _download_prices(request, downloader)
            cache_warning = "The existing price cache was unreadable and was refreshed."
    else:
        raw_prices = _download_prices(request, downloader)

    if not from_cache and use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            raw_prices.to_csv(cache_path, index_label="Date")
        except OSError:
            cache_warning = (
                "Prices were loaded successfully but could not be written to cache."
            )

    prices = _prepare_prices(raw_prices, request)
    required = minimum_observations or MINIMUM_OBSERVATIONS[request.frequency]
    if required < 2:
        raise RequestValidationError("Minimum observations must be at least two.")
    if len(prices) < required:
        raise PriceDataError(
            f"Only {len(prices)} aligned {request.frequency} observations are "
            f"available; at least {required} are required."
        )

    return MarketDataResult(
        prices=prices,
        assets=request.assets,
        benchmark=request.benchmark,
        frequency=request.frequency,
        periods_per_year=PERIODS_PER_YEAR[request.frequency],
        from_cache=from_cache,
        cache_warning=cache_warning,
    )


def _download_prices(
    request: MarketDataRequest, downloader: DownloadFunction | None
) -> pd.DataFrame:
    if downloader is None:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise MarketDataError(
                "yfinance is not installed. Install the project dependencies first."
            ) from exc
        downloader = yf.download

    try:
        downloaded = downloader(
            list(request.symbols),
            start=request.start.isoformat(),
            # yfinance treats end as exclusive; the product input is inclusive.
            end=(request.end + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as exc:  # noqa: BLE001 - provider errors need a stable boundary
        raise MarketDataError(f"Market-data download failed: {exc}") from exc

    if not isinstance(downloaded, pd.DataFrame) or downloaded.empty:
        raise PriceDataError("The market-data provider returned no prices.")
    return _extract_adjusted_prices(downloaded, request.symbols)


def _extract_adjusted_prices(
    downloaded: pd.DataFrame, symbols: Sequence[str]
) -> pd.DataFrame:
    """Extract adjusted close (or close fallback) from yfinance-style output."""
    if not isinstance(downloaded.columns, pd.MultiIndex):
        price_field = "Adj Close" if "Adj Close" in downloaded.columns else "Close"
        if price_field in downloaded.columns and len(symbols) == 1:
            return downloaded[[price_field]].rename(columns={price_field: symbols[0]})
        # A direct symbol-column table is useful for alternate providers and tests.
        if set(symbols).issubset(downloaded.columns):
            return downloaded.loc[:, list(symbols)].copy()
        raise PriceDataError("Adjusted-close prices were not present in the response.")

    for field in ("Adj Close", "Close"):
        for level in range(downloaded.columns.nlevels):
            if field not in downloaded.columns.get_level_values(level):
                continue
            extracted = downloaded.xs(field, axis=1, level=level, drop_level=True)
            if isinstance(extracted, pd.Series):
                extracted = extracted.to_frame(name=symbols[0])
            extracted.columns = [str(column).upper() for column in extracted.columns]
            return extracted

    raise PriceDataError("Adjusted-close prices were not present in the response.")


def _prepare_prices(
    raw_prices: pd.DataFrame, request: MarketDataRequest
) -> pd.DataFrame:
    prices = raw_prices.copy()
    prices.columns = [str(column).strip().upper() for column in prices.columns]
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = None
    prices = prices.loc[~prices.index.isna()]
    prices = prices.loc[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices.loc[
        pd.Timestamp(request.start) : pd.Timestamp(request.end), list(request.symbols)
    ] if set(request.symbols).issubset(prices.columns) else prices

    missing_symbols = [symbol for symbol in request.symbols if symbol not in prices]
    if missing_symbols:
        joined = ", ".join(missing_symbols)
        raise PriceDataError(f"No price series was returned for: {joined}.")

    prices = prices.loc[:, list(request.symbols)].apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan)
    unusable = [symbol for symbol in prices if prices[symbol].dropna().empty]
    if unusable:
        joined = ", ".join(unusable)
        raise PriceDataError(f"No usable prices were returned for: {joined}.")
    non_positive = [symbol for symbol in prices if (prices[symbol].dropna() <= 0).any()]
    if non_positive:
        raise PriceDataError(
            f"Non-positive prices were returned for: {', '.join(non_positive)}."
        )

    if request.frequency == "weekly":
        prices = prices.resample("W-FRI").last()
    elif request.frequency == "monthly":
        prices = prices.resample("ME").last()

    prices = prices.dropna(how="any")
    if prices.empty:
        raise PriceDataError("The selected assets have no overlapping price history.")
    return prices.astype(float)


def _normalise_ticker(ticker: str) -> str:
    if not isinstance(ticker, str):
        raise RequestValidationError("Every ticker must be text.")
    normalised = ticker.strip().upper()
    if not normalised or not _TICKER_PATTERN.fullmatch(normalised):
        raise RequestValidationError(f"Invalid ticker symbol: {ticker!r}.")
    return normalised


def _normalise_date(value: str | date | pd.Timestamp, label: str) -> date:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"Invalid {label} date: {value!r}.") from exc
    if pd.isna(timestamp):
        raise RequestValidationError(f"Invalid {label} date: {value!r}.")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.date()


def _cache_filename(request: MarketDataRequest) -> str:
    identity = "|".join(
        (*request.symbols, request.start.isoformat(), request.end.isoformat())
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"adjusted_prices_{digest}.csv"

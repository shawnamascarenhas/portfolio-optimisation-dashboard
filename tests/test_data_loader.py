"""Tests for historical market-data retrieval and validation."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.data_loader import (
    MarketDataError,
    PriceDataError,
    RequestValidationError,
    create_request,
    load_market_data,
)

ASSETS = ["AAPL", "MSFT", "JPM"]
BENCHMARK = "SPY"


def _provider_frame(periods: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    symbols = [*ASSETS, BENCHMARK]
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close"], symbols], names=["Price", "Ticker"]
    )
    values: dict[tuple[str, str], np.ndarray] = {}
    for offset, symbol in enumerate(symbols):
        series = 100 + offset * 10 + np.arange(periods, dtype=float)
        values[("Adj Close", symbol)] = series
        values[("Close", symbol)] = series + 1
    return pd.DataFrame(values, index=dates, columns=columns)


def _request(frequency: str = "daily"):
    return create_request(
        ASSETS,
        BENCHMARK,
        date(2024, 1, 1),
        date(2024, 6, 30),
        frequency=frequency,
    )


def test_create_request_normalises_tickers() -> None:
    request = create_request(
        [" aapl ", "msft", "jpm"], " spy ", "2024-01-01", "2024-06-30"
    )

    assert request.assets == ("AAPL", "MSFT", "JPM")
    assert request.benchmark == "SPY"
    assert request.symbols == ("AAPL", "MSFT", "JPM", "SPY")


@pytest.mark.parametrize(
    ("assets", "message"),
    [
        (["AAPL", "MSFT"], "between 3 and 10"),
        (["AAPL", "MSFT", "AAPL"], "unique"),
        (["AAPL", "MSFT", "BAD TICKER"], "Invalid ticker"),
    ],
)
def test_create_request_rejects_invalid_assets(
    assets: list[str], message: str
) -> None:
    with pytest.raises(RequestValidationError, match=message):
        create_request(assets, BENCHMARK, "2024-01-01", "2024-06-30")


def test_create_request_rejects_invalid_dates_and_frequency() -> None:
    with pytest.raises(RequestValidationError, match="earlier"):
        create_request(ASSETS, BENCHMARK, "2024-06-30", "2024-01-01")
    with pytest.raises(RequestValidationError, match="Frequency"):
        create_request(ASSETS, BENCHMARK, "2024-01-01", "2024-06-30", "annual")


def test_create_request_rejects_future_end_date_and_non_text_ticker() -> None:
    with pytest.raises(RequestValidationError, match="future"):
        create_request(ASSETS, BENCHMARK, "2024-01-01", "2999-01-01")
    with pytest.raises(RequestValidationError, match="must be text"):
        create_request(["AAPL", "MSFT", 123], BENCHMARK, "2024-01-01", "2024-06-30")


def test_load_market_data_extracts_adjusted_close_and_metadata(tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def downloader(*args, **kwargs):
        calls.append({"args": args, **kwargs})
        return _provider_frame()

    result = load_market_data(
        _request(), cache_dir=tmp_path, downloader=downloader, minimum_observations=20
    )

    assert list(result.prices.columns) == [*ASSETS, BENCHMARK]
    assert result.prices.iloc[0]["AAPL"] == 100.0
    assert result.assets == tuple(ASSETS)
    assert result.periods_per_year == 252
    assert result.from_cache is False
    assert calls[0]["end"] == "2024-07-01"
    pd.testing.assert_frame_equal(result.asset_prices, result.prices[ASSETS])
    pd.testing.assert_series_equal(
        result.benchmark_prices, result.prices[BENCHMARK]
    )


def test_load_market_data_uses_cache_on_second_call(tmp_path) -> None:
    call_count = 0

    def downloader(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _provider_frame()

    first = load_market_data(
        _request(), cache_dir=tmp_path, downloader=downloader, minimum_observations=20
    )
    second = load_market_data(
        _request(), cache_dir=tmp_path, downloader=downloader, minimum_observations=20
    )

    assert first.from_cache is False
    assert second.from_cache is True
    assert call_count == 1
    pd.testing.assert_frame_equal(first.prices, second.prices, check_freq=False)


def test_cache_write_failure_does_not_discard_downloaded_prices(
    tmp_path, monkeypatch
) -> None:
    def fail_to_cache(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_cache)
    result = load_market_data(
        _request(),
        cache_dir=tmp_path,
        downloader=lambda *args, **kwargs: _provider_frame(),
        minimum_observations=20,
    )

    assert not result.prices.empty
    assert result.from_cache is False
    assert "could not be written" in result.cache_warning


def test_weekly_frequency_resamples_to_friday(tmp_path) -> None:
    result = load_market_data(
        _request("weekly"),
        cache_dir=tmp_path,
        downloader=lambda *args, **kwargs: _provider_frame(),
        minimum_observations=10,
    )

    assert result.periods_per_year == 52
    assert all(timestamp.dayofweek == 4 for timestamp in result.prices.index)


def test_monthly_frequency_resamples_to_month_end(tmp_path) -> None:
    result = load_market_data(
        _request("monthly"),
        cache_dir=tmp_path,
        downloader=lambda *args, **kwargs: _provider_frame(),
        minimum_observations=3,
    )

    assert result.periods_per_year == 12
    assert all(timestamp.is_month_end for timestamp in result.prices.index)


def test_direct_symbol_columns_are_supported(tmp_path) -> None:
    direct = _provider_frame().xs("Adj Close", axis=1, level="Price")
    result = load_market_data(
        _request(),
        cache_dir=tmp_path,
        downloader=lambda *args, **kwargs: direct,
        minimum_observations=20,
    )

    assert list(result.prices) == [*ASSETS, BENCHMARK]


def test_provider_failure_and_empty_response_are_translated(tmp_path) -> None:
    def broken_provider(*args, **kwargs):
        raise ConnectionError("offline")

    with pytest.raises(MarketDataError, match="offline"):
        load_market_data(
            _request(), cache_dir=tmp_path, downloader=broken_provider
        )
    with pytest.raises(PriceDataError, match="no prices"):
        load_market_data(
            _request(),
            cache_dir=tmp_path,
            downloader=lambda *args, **kwargs: pd.DataFrame(),
        )


def test_minimum_observations_must_be_at_least_two(tmp_path) -> None:
    with pytest.raises(RequestValidationError, match="at least two"):
        load_market_data(
            _request(),
            cache_dir=tmp_path,
            downloader=lambda *args, **kwargs: _provider_frame(),
            minimum_observations=1,
        )


def test_missing_symbol_is_reported_instead_of_silently_dropped(tmp_path) -> None:
    incomplete = _provider_frame().drop(columns=("Adj Close", "JPM")).drop(
        columns=("Close", "JPM")
    )

    with pytest.raises(PriceDataError, match="JPM"):
        load_market_data(
            _request(),
            cache_dir=tmp_path,
            downloader=lambda *args, **kwargs: incomplete,
            minimum_observations=20,
        )


def test_insufficient_aligned_history_is_rejected(tmp_path) -> None:
    with pytest.raises(PriceDataError, match="at least 60"):
        load_market_data(
            _request(),
            cache_dir=tmp_path,
            downloader=lambda *args, **kwargs: _provider_frame(periods=20),
        )


def test_non_positive_prices_are_rejected(tmp_path) -> None:
    invalid = _provider_frame()
    invalid.loc[invalid.index[0], ("Adj Close", "MSFT")] = 0.0

    with pytest.raises(PriceDataError, match="MSFT"):
        load_market_data(
            _request(),
            cache_dir=tmp_path,
            downloader=lambda *args, **kwargs: invalid,
            minimum_observations=20,
        )

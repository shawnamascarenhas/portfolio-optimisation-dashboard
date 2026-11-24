"""Tests for reproducible offline demonstration prices."""

import pandas as pd

from portfolio_analytics.demo_data import make_demo_downloader


def test_demo_downloader_is_reproducible_and_provider_compatible() -> None:
    downloader = make_demo_downloader(seed=123)
    first = downloader(
        ["A", "B", "SPY"], start="2024-01-01", end="2024-04-01"
    )
    second = downloader(
        ["A", "B", "SPY"], start="2024-01-01", end="2024-04-01"
    )

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["A", "B", "SPY"]
    assert first.index.is_monotonic_increasing
    assert (first > 0).all().all()


def test_demo_downloader_uses_exclusive_end_date() -> None:
    prices = make_demo_downloader()(
        ["A"], start="2024-01-01", end="2024-01-10"
    )

    assert prices.index.max() < pd.Timestamp("2024-01-10")

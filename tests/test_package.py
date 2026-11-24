"""Smoke tests for the project package."""

import portfolio_analytics


def test_package_version() -> None:
    assert portfolio_analytics.__version__ == "0.1.0"

# Deployment Guide

## Local installation

```bash
git clone <repository-url>
cd portfolio-optimisation-dashboard
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The reproducible demo is enabled by default and does not require network access.
Turn it off in the sidebar to request adjusted historical prices from Yahoo Finance.

## Pre-deployment checks

```bash
python -m pip install -r requirements-dev.txt
pytest
ruff check .
pytest --cov=portfolio_analytics --cov-report=term-missing
```

Do not deploy if tests, linting, or the configured 85% coverage threshold fails.

## Streamlit hosting

Use `app.py` as the application entry point and Python 3.11 as the runtime. The
repository's `requirements.txt` contains all runtime dependencies. No secrets are
required for deterministic demo mode or the current Yahoo Finance integration.

Recommended settings:

- Main file: `app.py`
- Python: 3.11
- Working directory: repository root
- Health check: confirm the page loads and the default demo completes

## Post-deployment verification

1. Confirm the sidebar renders all default inputs.
2. Run the default demonstration case.
3. Confirm five dashboard tabs appear.
4. Verify the efficient frontier and both optimal portfolios render.
5. Verify portfolio values and backtest statistics appear.
6. Confirm source, observation date, warnings, and disclaimer are visible.
7. Enter infeasible or invalid inputs and confirm the app shows a clear error.

## Operational considerations

- Yahoo Finance is suitable for a portfolio demonstration, not institutional
  production data.
- Cached market data is stored locally under `data/cache/` and excluded from Git.
- A failed cache write is non-fatal and is disclosed in the interface.
- The application stores the latest successful result only in the active Streamlit
  session; it has no user database or authentication layer.
- Review dependency versions and repeat model validation before material releases.

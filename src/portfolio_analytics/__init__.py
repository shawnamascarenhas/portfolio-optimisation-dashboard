"""Portfolio optimisation and investment analytics package."""

from portfolio_analytics.backtest import BacktestError, BacktestResult, run_backtest
from portfolio_analytics.data_loader import (
    MarketDataError,
    MarketDataRequest,
    MarketDataResult,
    PriceDataError,
    RequestValidationError,
    create_request,
    load_market_data,
)
from portfolio_analytics.demo_data import make_demo_downloader
from portfolio_analytics.optimisation import (
    EfficientFrontier,
    OptimisationError,
    PortfolioSolution,
    calculate_efficient_frontier,
    optimise_maximum_sharpe,
    optimise_minimum_variance,
    portfolio_statistics,
)
from portfolio_analytics.portfolio_comparison import (
    CUSTOM,
    EQUAL_WEIGHT,
    MAXIMUM_SHARPE,
    MINIMUM_VARIANCE,
    PortfolioComparison,
    PortfolioComparisonError,
    compare_portfolios,
    validate_custom_weights,
)
from portfolio_analytics.returns import (
    ReturnAnalysis,
    ReturnCalculationError,
    analyse_returns,
    calculate_annualised_covariance,
    calculate_annualised_expected_returns,
    calculate_cumulative_returns,
    calculate_log_returns,
    calculate_simple_returns,
)
from portfolio_analytics.risk_metrics import (
    RiskAnalysis,
    RiskCalculationError,
    analyse_risk,
    calculate_beta,
    calculate_downside_deviation,
    calculate_drawdowns,
    calculate_historical_expected_shortfall,
    calculate_historical_var,
    calculate_maximum_drawdown,
)
from portfolio_analytics.safeguards import SafeguardNotice, assess_model_safeguards

__version__ = "0.1.0"

__all__ = [
    "MarketDataError",
    "MarketDataRequest",
    "MarketDataResult",
    "EfficientFrontier",
    "BacktestError",
    "BacktestResult",
    "CUSTOM",
    "EQUAL_WEIGHT",
    "MAXIMUM_SHARPE",
    "MINIMUM_VARIANCE",
    "OptimisationError",
    "PortfolioSolution",
    "PortfolioComparison",
    "PortfolioComparisonError",
    "PriceDataError",
    "RequestValidationError",
    "ReturnAnalysis",
    "ReturnCalculationError",
    "RiskAnalysis",
    "RiskCalculationError",
    "SafeguardNotice",
    "analyse_returns",
    "analyse_risk",
    "assess_model_safeguards",
    "calculate_annualised_covariance",
    "calculate_annualised_expected_returns",
    "calculate_cumulative_returns",
    "calculate_beta",
    "calculate_downside_deviation",
    "calculate_drawdowns",
    "calculate_historical_expected_shortfall",
    "calculate_historical_var",
    "calculate_efficient_frontier",
    "calculate_log_returns",
    "calculate_maximum_drawdown",
    "calculate_simple_returns",
    "compare_portfolios",
    "create_request",
    "load_market_data",
    "make_demo_downloader",
    "optimise_maximum_sharpe",
    "optimise_minimum_variance",
    "portfolio_statistics",
    "run_backtest",
    "validate_custom_weights",
]

"""Gap Trading Backtesting Framework.

Provides comprehensive backtesting with minute-level data support
for accurate gap trading strategy validation and optimization.
"""

from .config import BacktestConfig
from .data_loader import GapBacktestDataLoader
from .signal_engine import SignalEngine, GapSignal
from .trade_simulator import TradeSimulator, SimulatedTrade
from .metrics_engine import MetricsEngine, BacktestMetrics
from .backtest_engine import BacktestEngine, BacktestResult

__all__ = [
    'BacktestConfig',
    'GapBacktestDataLoader',
    'SignalEngine',
    'GapSignal',
    'TradeSimulator',
    'SimulatedTrade',
    'MetricsEngine',
    'BacktestMetrics',
    'BacktestEngine',
    'BacktestResult',
]

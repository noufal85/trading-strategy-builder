"""
Sample strategies and usage examples for the Strategy Builder framework
"""

from .moving_average_crossover import MovingAverageCrossover
from .run_backtest import run_ma_crossover_backtest
from .run_alpaca_backtest import run_alpaca_backtest
from .run_marketstack_backtest import run_marketstack_backtest

__all__ = [
    'MovingAverageCrossover',
    'run_ma_crossover_backtest',
    'run_alpaca_backtest',
    'run_marketstack_backtest'
]

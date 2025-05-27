"""
Trend Following Strategy Package.

This package provides a complete trend following strategy implementation
for daily stock screening based on price trends and volume analysis.
"""

from .strategy import TrendFollowingStrategy
from .trend_analyzer import SimpleTrendAnalyzer, LinearRegressionTrendAnalyzer, VolumeAnalyzer
from .config import get_config, get_stock_list, validate_config, DEFAULT_CONFIG, POPULAR_STOCKS

__all__ = [
    'TrendFollowingStrategy',
    'SimpleTrendAnalyzer',
    'LinearRegressionTrendAnalyzer', 
    'VolumeAnalyzer',
    'get_config',
    'get_stock_list',
    'validate_config',
    'DEFAULT_CONFIG',
    'POPULAR_STOCKS'
]

__version__ = '1.0.0'

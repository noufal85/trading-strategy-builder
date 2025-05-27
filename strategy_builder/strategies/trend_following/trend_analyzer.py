"""
Trend analysis module for trend following strategies.

This module provides pluggable trend detection methods that can be used
by trend following strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TrendAnalyzer(ABC):
    """
    Abstract base class for trend analysis methods.
    
    This allows for pluggable trend detection strategies.
    """
    
    @abstractmethod
    def analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the trend in the given price data.
        
        Args:
            data: DataFrame with OHLCV data (indexed by date)
            
        Returns:
            Dict containing trend analysis results:
            - is_trending_up: bool
            - trend_strength: float (0.0 to 1.0)
            - trend_details: dict with specific metrics
        """
        pass


class SimpleTrendAnalyzer(TrendAnalyzer):
    """
    Simple trend analyzer using basic price momentum and moving averages.
    
    This analyzer considers a stock trending up if:
    1. The price slope over the analysis period is positive
    2. Recent prices are above the moving average
    3. Most recent days show positive price movement
    """
    
    def __init__(
        self,
        analysis_days: int = 5,
        min_slope: float = 0.01,
        min_positive_days: int = 3,
        ma_period: int = 5
    ):
        """
        Initialize the simple trend analyzer.
        
        Args:
            analysis_days: Number of recent days to analyze
            min_slope: Minimum price slope to consider trending up
            min_positive_days: Minimum number of positive days required
            ma_period: Moving average period for trend confirmation
        """
        self.analysis_days = analysis_days
        self.min_slope = min_slope
        self.min_positive_days = min_positive_days
        self.ma_period = ma_period
    
    def analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trend using simple price momentum methods.
        
        Args:
            data: DataFrame with OHLCV data (indexed by date)
            
        Returns:
            Dict containing trend analysis results
        """
        if len(data) < self.analysis_days:
            return {
                'is_trending_up': False,
                'trend_strength': 0.0,
                'trend_details': {'error': 'Insufficient data for analysis'}
            }
        
        # Get the most recent data
        recent_data = data.tail(self.analysis_days).copy()
        
        # Calculate price slope (linear regression)
        prices = recent_data['close'].values
        days = np.arange(len(prices))
        slope = np.polyfit(days, prices, 1)[0]
        
        # Calculate relative slope (as percentage of price)
        relative_slope = slope / prices[0] if prices[0] != 0 else 0
        
        # Count positive days (close > previous close)
        price_changes = recent_data['close'].pct_change().dropna()
        positive_days = (price_changes > 0).sum()
        
        # Calculate moving average
        if len(data) >= self.ma_period:
            ma_data = data.tail(self.ma_period)
            moving_average = ma_data['close'].mean()
            current_price = recent_data['close'].iloc[-1]
            above_ma = current_price > moving_average
        else:
            above_ma = True  # If not enough data for MA, don't penalize
            moving_average = recent_data['close'].iloc[-1]
        
        # Calculate trend strength (0.0 to 1.0)
        slope_strength = min(max(relative_slope / self.min_slope, 0), 1.0)
        positive_days_strength = positive_days / len(price_changes)
        ma_strength = 1.0 if above_ma else 0.5
        
        trend_strength = (slope_strength + positive_days_strength + ma_strength) / 3.0
        
        # Determine if trending up
        is_trending_up = (
            relative_slope >= self.min_slope and
            positive_days >= self.min_positive_days and
            above_ma
        )
        
        return {
            'is_trending_up': is_trending_up,
            'trend_strength': trend_strength,
            'trend_details': {
                'slope': slope,
                'relative_slope': relative_slope,
                'positive_days': positive_days,
                'total_days': len(price_changes),
                'above_moving_average': above_ma,
                'moving_average': moving_average,
                'current_price': recent_data['close'].iloc[-1],
                'price_change_pct': price_changes.sum() * 100,  # Total % change
                'analysis_period': self.analysis_days
            }
        }


class LinearRegressionTrendAnalyzer(TrendAnalyzer):
    """
    Trend analyzer using linear regression with R-squared for trend strength.
    """
    
    def __init__(
        self,
        analysis_days: int = 5,
        min_slope: float = 0.02,
        min_r_squared: float = 0.7
    ):
        """
        Initialize the linear regression trend analyzer.
        
        Args:
            analysis_days: Number of recent days to analyze
            min_slope: Minimum price slope (as % of price) to consider trending
            min_r_squared: Minimum R-squared value for trend reliability
        """
        self.analysis_days = analysis_days
        self.min_slope = min_slope
        self.min_r_squared = min_r_squared
    
    def analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trend using linear regression.
        
        Args:
            data: DataFrame with OHLCV data (indexed by date)
            
        Returns:
            Dict containing trend analysis results
        """
        if len(data) < self.analysis_days:
            return {
                'is_trending_up': False,
                'trend_strength': 0.0,
                'trend_details': {'error': 'Insufficient data for analysis'}
            }
        
        # Get the most recent data
        recent_data = data.tail(self.analysis_days).copy()
        prices = recent_data['close'].values
        days = np.arange(len(prices))
        
        # Perform linear regression
        coeffs = np.polyfit(days, prices, 1)
        slope, intercept = coeffs
        
        # Calculate R-squared
        predicted = slope * days + intercept
        ss_res = np.sum((prices - predicted) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate relative slope
        relative_slope = slope / prices[0] if prices[0] != 0 else 0
        
        # Calculate trend strength based on slope and R-squared
        slope_strength = min(max(relative_slope / self.min_slope, 0), 1.0)
        r_squared_strength = min(max(r_squared / self.min_r_squared, 0), 1.0)
        trend_strength = (slope_strength + r_squared_strength) / 2.0
        
        # Determine if trending up
        is_trending_up = (
            relative_slope >= self.min_slope and
            r_squared >= self.min_r_squared
        )
        
        return {
            'is_trending_up': is_trending_up,
            'trend_strength': trend_strength,
            'trend_details': {
                'slope': slope,
                'relative_slope': relative_slope,
                'r_squared': r_squared,
                'intercept': intercept,
                'current_price': prices[-1],
                'predicted_price': predicted[-1],
                'analysis_period': self.analysis_days
            }
        }


class VolumeAnalyzer:
    """
    Analyzer for volume-based screening criteria.
    """
    
    def __init__(
        self,
        volume_avg_days: int = 20,
        volume_threshold: float = 1.2,
        analysis_days: int = 5
    ):
        """
        Initialize the volume analyzer.
        
        Args:
            volume_avg_days: Days to calculate average volume
            volume_threshold: Multiplier for above-average volume (e.g., 1.2 = 120%)
            analysis_days: Recent days to check for volume spike
        """
        self.volume_avg_days = volume_avg_days
        self.volume_threshold = volume_threshold
        self.analysis_days = analysis_days
    
    def analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volume patterns in the data.
        
        Args:
            data: DataFrame with OHLCV data (indexed by date)
            
        Returns:
            Dict containing volume analysis results
        """
        if len(data) < self.volume_avg_days:
            return {
                'is_volume_above_average': False,
                'volume_strength': 0.0,
                'volume_details': {'error': 'Insufficient data for volume analysis'}
            }
        
        # Calculate average volume over the specified period
        avg_volume = data['volume'].tail(self.volume_avg_days).mean()
        
        # Get recent volume data
        recent_volume_data = data['volume'].tail(self.analysis_days)
        recent_avg_volume = recent_volume_data.mean()
        current_volume = data['volume'].iloc[-1]
        
        # Calculate volume ratio
        volume_ratio = recent_avg_volume / avg_volume if avg_volume > 0 else 0
        current_volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Calculate volume strength
        volume_strength = min(volume_ratio / self.volume_threshold, 1.0)
        
        # Determine if volume is above average
        is_volume_above_average = volume_ratio >= self.volume_threshold
        
        return {
            'is_volume_above_average': is_volume_above_average,
            'volume_strength': volume_strength,
            'volume_details': {
                'average_volume': avg_volume,
                'recent_average_volume': recent_avg_volume,
                'current_volume': current_volume,
                'volume_ratio': volume_ratio,
                'current_volume_ratio': current_volume_ratio,
                'volume_threshold': self.volume_threshold,
                'analysis_days': self.analysis_days,
                'volume_avg_days': self.volume_avg_days
            }
        }

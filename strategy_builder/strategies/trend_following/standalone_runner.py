"""
Standalone Trend Following Scanner

This is a simplified version that avoids package import issues.
Run this directly from the trend_following directory.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add paths
current_dir = os.path.dirname(__file__)
strategy_builder_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, strategy_builder_root)
sys.path.insert(0, current_dir)

# Import FMP package with yfinance fallback
try:
    from fmp import FMPClient, StockPrice
    USE_FMP = True
    print("Using FMP for data")
except ImportError:
    try:
        import yfinance as yf
        USE_FMP = False
        print("FMP not available, using yfinance as fallback")
    except ImportError:
        print("Error: Neither FMP nor yfinance package found.")
        print("Please install one of them:")
        print("pip install git+https://github.com/noufal85/fmp.git")
        print("pip install yfinance")
        sys.exit(1)

# Configuration constants
DEFAULT_CONFIG = {
    'analysis_days': 5,
    'volume_avg_days': 20,
    'volume_threshold': 1.2,
    'trend_method': 'simple',
    'min_slope': 0.01,
    'min_positive_days': 3,
    'ma_period': 5,
    'output_format': 'console',
    'output_file': 'trend_following_results.csv',
    'log_level': 'INFO',
    'detailed_output': True,
}

POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'CRM',
    'JPM', 'BAC', 'WFC', 'GS', 'MS',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
    'KO', 'PEP', 'WMT', 'HD', 'MCD',
    'BA', 'CAT', 'GE', 'MMM', 'UPS'
]


class SimpleTrendAnalyzer:
    """Simple trend analyzer for stock data."""
    
    def __init__(self, analysis_days=5, min_slope=0.01, min_positive_days=3, ma_period=5):
        self.analysis_days = analysis_days
        self.min_slope = min_slope
        self.min_positive_days = min_positive_days
        self.ma_period = ma_period
    
    def analyze_trend(self, data):
        """Analyze trend in the data."""
        if len(data) < self.analysis_days:
            return {
                'is_trending_up': False,
                'trend_strength': 0.0,
                'trend_details': {'error': 'Insufficient data'}
            }
        
        # Get recent data
        recent_data = data.tail(self.analysis_days).copy()
        prices = recent_data['close'].values
        days = np.arange(len(prices))
        slope = np.polyfit(days, prices, 1)[0]
        relative_slope = slope / prices[0] if prices[0] != 0 else 0
        
        # Count positive days
        price_changes = recent_data['close'].pct_change().dropna()
        positive_days = (price_changes > 0).sum()
        
        # Moving average check
        if len(data) >= self.ma_period:
            ma_data = data.tail(self.ma_period)
            moving_average = ma_data['close'].mean()
            current_price = recent_data['close'].iloc[-1]
            above_ma = current_price > moving_average
        else:
            above_ma = True
            moving_average = recent_data['close'].iloc[-1]
        
        # Calculate trend strength
        slope_strength = min(max(relative_slope / self.min_slope, 0), 1.0)
        positive_days_strength = positive_days / len(price_changes) if len(price_changes) > 0 else 0
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
                'price_change_pct': price_changes.sum() * 100,
                'analysis_period': self.analysis_days
            }
        }


class VolumeAnalyzer:
    """Volume analyzer for stock data."""
    
    def __init__(self, volume_avg_days=20, volume_threshold=1.2, analysis_days=5):
        self.volume_avg_days = volume_avg_days
        self.volume_threshold = volume_threshold
        self.analysis_days = analysis_days
    
    def analyze_volume(self, data):
        """Analyze volume patterns."""
        if len(data) < self.volume_avg_days:
            return {
                'is_volume_above_average': False,
                'volume_strength': 0.0,
                'volume_details': {'error': 'Insufficient data for volume analysis'}
            }
        
        # Calculate average volume
        avg_volume = data['volume'].tail(self.volume_avg_days).mean()
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


class FMPDataProvider:
    """Simple FMP data provider."""
    
    def __init__(self, api_key=None):
        self.client = FMPClient(api_key=api_key)
        self.stock_price = StockPrice(self.client)
    
    def get_historical_data(self, symbol, start_date, end_date=None):
        """Get historical data for a symbol."""
        if end_date is None:
            end_date = datetime.now()
        
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        try:
            data = self.stock_price.get_historical_price(
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
                as_dataframe=True
            )
            
            # Normalize data
            if not data.empty:
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data = data.set_index('date')
                data = data.sort_index()
            
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")


class YFinanceDataProvider:
    """Simple yfinance data provider."""
    
    def __init__(self, api_key=None):
        # yfinance doesn't need API key
        pass
    
    def get_historical_data(self, symbol, start_date, end_date=None):
        """Get historical data for a symbol using yfinance."""
        if end_date is None:
            end_date = datetime.now()
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            # Normalize data
            if not data.empty:
                # Convert column names to lowercase
                data.columns = [col.lower() for col in data.columns]
                data = data.sort_index()
            
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Standalone Trend Following Scanner')
    
    parser.add_argument('--stocks', type=str, help='Comma-separated list of stock symbols')
    parser.add_argument('--analysis-days', type=int, default=5, help='Days to analyze')
    parser.add_argument('--volume-threshold', type=float, default=1.2, help='Volume threshold')
    parser.add_argument('--api-key', type=str, help='FMP API key')
    parser.add_argument('--dry-run', action='store_true', help='Show configuration only')
    
    return parser.parse_args()


def analyze_stock(symbol, data, trend_analyzer, volume_analyzer):
    """Analyze a single stock."""
    # Perform trend analysis
    trend_result = trend_analyzer.analyze_trend(data)
    
    # Perform volume analysis
    volume_result = volume_analyzer.analyze_volume(data)
    
    # Determine if stock qualifies
    qualifies = (
        trend_result['is_trending_up'] and
        volume_result['is_volume_above_average']
    )
    
    # Get current price data
    latest_data = data.iloc[-1]
    
    return {
        'symbol': symbol,
        'qualifies': qualifies,
        'current_price': float(latest_data['close']),
        'current_volume': int(latest_data['volume']),
        'trend_strength': trend_result['trend_strength'],
        'volume_strength': volume_result['volume_strength'],
        'trend_details': trend_result['trend_details'],
        'volume_details': volume_result['volume_details']
    }


def print_results(results):
    """Print results to console."""
    if not results:
        print("\nNo stocks currently qualify for trend following criteria.")
        return
    
    print(f"\n{'='*80}")
    print(f"TREND FOLLOWING SCREENING RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"Found {len(results)} qualifying stocks:\n")
    
    for result in results:
        symbol = result['symbol']
        price = result['current_price']
        trend_strength = result['trend_strength']
        volume_strength = result['volume_strength']
        
        print(f"🔥 {symbol} - ${price:.2f}")
        print(f"   Trend Strength: {trend_strength:.2f} | Volume Strength: {volume_strength:.2f}")
        
        trend_details = result['trend_details']
        volume_details = result['volume_details']
        
        if 'relative_slope' in trend_details:
            print(f"   Price Slope: {trend_details['relative_slope']:.3f} ({trend_details['relative_slope']*100:.1f}%/day)")
        
        if 'positive_days' in trend_details:
            print(f"   Positive Days: {trend_details['positive_days']}/{trend_details.get('total_days', 'N/A')}")
        
        if 'volume_ratio' in volume_details:
            print(f"   Volume Ratio: {volume_details['volume_ratio']:.2f}x average")
        
        print(f"   Current Volume: {result['current_volume']:,}")
        print()
    
    print(f"{'='*80}\n")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Get stock list
    if args.stocks:
        stock_list = [stock.strip().upper() for stock in args.stocks.split(',')]
    else:
        stock_list = POPULAR_STOCKS[:10]  # Use first 10 for testing
    
    # Get API key (only required for FMP)
    api_key = args.api_key or os.getenv('FMP_API_KEY')
    if USE_FMP and not api_key:
        print("Error: FMP API key is required when using FMP. Set FMP_API_KEY environment variable or use --api-key option.")
        sys.exit(1)
    
    print("=" * 60)
    print("STANDALONE TREND FOLLOWING SCANNER")
    print("=" * 60)
    print(f"Analysis Days: {args.analysis_days}")
    print(f"Volume Threshold: {args.volume_threshold:.1f}x")
    print(f"Stocks to Analyze: {len(stock_list)} ({', '.join(stock_list[:5])}{'...' if len(stock_list) > 5 else ''})")
    print("=" * 60)
    print()
    
    if args.dry_run:
        print("Dry run mode - configuration shown above. Exiting without execution.")
        return
    
    try:
        # Initialize components
        if USE_FMP:
            data_provider = FMPDataProvider(api_key=api_key)
        else:
            data_provider = YFinanceDataProvider()
        
        trend_analyzer = SimpleTrendAnalyzer(
            analysis_days=args.analysis_days,
            min_slope=0.01,
            min_positive_days=3
        )
        volume_analyzer = VolumeAnalyzer(
            volume_threshold=args.volume_threshold
        )
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Get 30 days of data
        
        qualifying_stocks = []
        
        print("Starting screening process...")
        
        for symbol in stock_list:
            try:
                print(f"Analyzing {symbol}...")
                
                # Fetch data
                data = data_provider.get_historical_data(symbol, start_date, end_date)
                
                if data.empty:
                    print(f"  No data available for {symbol}")
                    continue
                
                # Analyze stock
                result = analyze_stock(symbol, data, trend_analyzer, volume_analyzer)
                
                if result['qualifies']:
                    qualifying_stocks.append(result)
                    print(f"  ✓ {symbol} qualifies")
                else:
                    print(f"  ✗ {symbol} does not qualify")
                
            except Exception as e:
                print(f"  Error analyzing {symbol}: {str(e)}")
                continue
        
        # Print results
        print_results(qualifying_stocks)
        
        print(f"Screening completed! Found {len(qualifying_stocks)} qualifying stocks.")
        
    except Exception as e:
        print(f"Error during screening: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()

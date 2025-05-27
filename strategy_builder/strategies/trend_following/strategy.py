"""
Trend Following Strategy Implementation.

This strategy screens stocks based on price trends and volume patterns,
identifying stocks that are trending upward with above-average volume.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add paths for imports when running directly
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Import from parent directories
from strategy_builder.core.strategy import Strategy
from strategy_builder.data.fmp_data_provider import FMPDataProvider
from strategy_builder.utils.types import Signal
from trend_analyzer import SimpleTrendAnalyzer, LinearRegressionTrendAnalyzer, VolumeAnalyzer
from config import get_config, validate_config, get_stock_list


class TrendFollowingStrategy(Strategy):
    """
    Trend following strategy that screens stocks for upward price trends
    combined with above-average volume.
    
    This strategy is designed to run daily before market open to identify
    stocks that meet the trending criteria.
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        stock_list: List[str] = None,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize the trend following strategy.
        
        Args:
            config: Strategy configuration dictionary
            stock_list: List of stocks to analyze
            api_key: FMP API key
            **kwargs: Additional arguments passed to base Strategy class
        """
        # Initialize base strategy
        super().__init__(
            name="Trend Following Strategy",
            log_level=config.get('log_level', 'INFO') if config else 'INFO',
            **kwargs
        )
        
        # Validate and set configuration
        self.config = validate_config(config or {})
        
        # Set up data provider
        self.data_provider = FMPDataProvider(api_key=api_key)
        
        # Set up stock list
        self.stock_list = stock_list or get_stock_list('popular')
        
        # Initialize analyzers based on configuration
        self._setup_analyzers()
        
        # Results storage
        self.screening_results = []
        
        self.logger.info(f"Initialized {self.name}")
        self.logger.info(f"Configuration: {self.config}")
        self.logger.info(f"Analyzing {len(self.stock_list)} stocks")
    
    def _setup_analyzers(self):
        """Set up trend and volume analyzers based on configuration."""
        # Set up trend analyzer
        if self.config['trend_method'] == 'linear_regression':
            self.trend_analyzer = LinearRegressionTrendAnalyzer(
                analysis_days=self.config['analysis_days'],
                min_slope=self.config['linear_min_slope'],
                min_r_squared=self.config['min_r_squared']
            )
        else:
            self.trend_analyzer = SimpleTrendAnalyzer(
                analysis_days=self.config['analysis_days'],
                min_slope=self.config['min_slope'],
                min_positive_days=self.config['min_positive_days'],
                ma_period=self.config['ma_period']
            )
        
        # Set up volume analyzer
        self.volume_analyzer = VolumeAnalyzer(
            volume_avg_days=self.config['volume_avg_days'],
            volume_threshold=self.config['volume_threshold'],
            analysis_days=self.config['analysis_days']
        )
    
    def run_daily_screening(self) -> List[Dict[str, Any]]:
        """
        Run the daily stock screening process.
        
        Returns:
            List of dictionaries containing screening results for qualifying stocks
        """
        self.logger.info("Starting daily trend following screening...")
        
        # Calculate date range for data fetching
        end_date = datetime.now()
        # Fetch extra days to ensure we have enough data for volume analysis
        start_date = end_date - timedelta(days=self.config['volume_avg_days'] + 10)
        
        qualifying_stocks = []
        total_analyzed = 0
        
        for symbol in self.stock_list:
            try:
                self.logger.info(f"Analyzing {symbol}...")
                
                # Fetch historical data
                data = self.data_provider.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data.empty:
                    self.logger.warning(f"No data available for {symbol}")
                    continue
                
                # Analyze the stock
                result = self.analyze_stock(symbol, data)
                
                if result['qualifies']:
                    qualifying_stocks.append(result)
                    self.logger.info(f"✓ {symbol} qualifies for trend following")
                else:
                    self.logger.debug(f"✗ {symbol} does not qualify")
                
                total_analyzed += 1
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        self.screening_results = qualifying_stocks
        
        self.logger.info(
            f"Screening complete. {len(qualifying_stocks)} out of {total_analyzed} "
            f"stocks qualify for trend following."
        )
        
        # Output results
        self._output_results(qualifying_stocks)
        
        return qualifying_stocks
    
    def analyze_stock(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a single stock for trend following criteria.
        
        Args:
            symbol: Stock symbol
            data: Historical OHLCV data
            
        Returns:
            Dict containing analysis results
        """
        # Perform trend analysis
        trend_result = self.trend_analyzer.analyze_trend(data)
        
        # Perform volume analysis
        volume_result = self.volume_analyzer.analyze_volume(data)
        
        # Determine if stock qualifies
        qualifies = (
            trend_result['is_trending_up'] and
            volume_result['is_volume_above_average']
        )
        
        # Get current price data
        latest_data = data.iloc[-1]
        
        # Compile results
        result = {
            'symbol': symbol,
            'qualifies': qualifies,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'current_price': float(latest_data['close']),
            'current_volume': int(latest_data['volume']),
            
            # Trend analysis results
            'is_trending_up': trend_result['is_trending_up'],
            'trend_strength': trend_result['trend_strength'],
            'trend_details': trend_result['trend_details'],
            
            # Volume analysis results
            'is_volume_above_average': volume_result['is_volume_above_average'],
            'volume_strength': volume_result['volume_strength'],
            'volume_details': volume_result['volume_details'],
            
            # Additional metrics
            'data_points': len(data),
            'date_range': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
        }
        
        return result
    
    def _output_results(self, results: List[Dict[str, Any]]):
        """Output screening results based on configuration."""
        if not results:
            print("\nNo stocks currently qualify for trend following criteria.")
            return
        
        # Console output
        if self.config['output_format'] in ['console', 'both']:
            self._print_console_results(results)
        
        # CSV output
        if self.config['output_format'] in ['csv', 'both']:
            self._save_csv_results(results)
    
    def _print_console_results(self, results: List[Dict[str, Any]]):
        """Print results to console."""
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
            
            if self.config['detailed_output']:
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
    
    def _save_csv_results(self, results: List[Dict[str, Any]]):
        """Save results to CSV file."""
        if not results:
            return
        
        # Flatten the results for CSV export
        flattened_results = []
        
        for result in results:
            flat_result = {
                'symbol': result['symbol'],
                'analysis_date': result['analysis_date'],
                'qualifies': result['qualifies'],
                'current_price': result['current_price'],
                'current_volume': result['current_volume'],
                'is_trending_up': result['is_trending_up'],
                'trend_strength': result['trend_strength'],
                'is_volume_above_average': result['is_volume_above_average'],
                'volume_strength': result['volume_strength'],
                'data_points': result['data_points'],
                'date_range': result['date_range']
            }
            
            # Add trend details
            for key, value in result['trend_details'].items():
                flat_result[f'trend_{key}'] = value
            
            # Add volume details
            for key, value in result['volume_details'].items():
                flat_result[f'volume_{key}'] = value
            
            flattened_results.append(flat_result)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(flattened_results)
        filename = self.config['output_file']
        df.to_csv(filename, index=False)
        
        self.logger.info(f"Results saved to {filename}")
        print(f"Results saved to {filename}")
    
    # Abstract method implementations (required by base Strategy class)
    def on_data(self, data: Dict[str, Any]) -> Optional[Signal]:
        """
        Process market data (not used in screening mode).
        
        This method is required by the base Strategy class but not used
        in the daily screening implementation.
        """
        return None
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """
        Calculate position size (not used in screening mode).
        
        This method is required by the base Strategy class but not used
        in the daily screening implementation.
        """
        return 0.0

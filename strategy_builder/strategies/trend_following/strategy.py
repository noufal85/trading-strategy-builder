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
from html_report_generator import HTMLReportGenerator


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
        self.all_results = []  # Store all results for enhanced reporting
        
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
    
    def run_daily_screening(
        self,
        show_all_analysis: bool = False,
        generate_html_report: bool = False,
        reports_folder: str = "reports"
    ) -> List[Dict[str, Any]]:
        """
        Run the daily stock screening process with enhanced reporting.
        
        Args:
            show_all_analysis: Show analysis for all stocks, not just qualifiers
            generate_html_report: Generate HTML report
            reports_folder: Folder for HTML reports
            
        Returns:
            List of dictionaries containing screening results for qualifying stocks
        """
        self.logger.info("Starting daily trend following screening...")
        
        # Calculate date range for data fetching
        end_date = datetime.now()
        # Fetch 1 year of data to calculate 52-week high/low
        start_date = end_date - timedelta(days=365)
        
        qualifying_stocks = []
        all_results = []
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
                
                # Store historical data for charts
                if generate_html_report:
                    result['historical_data'] = data
                
                # Add to all results
                all_results.append(result)
                
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
        self.all_results = all_results
        
        self.logger.info(
            f"Screening complete. {len(qualifying_stocks)} out of {total_analyzed} "
            f"stocks qualify for trend following."
        )
        
        # Enhanced output with detailed analysis
        if show_all_analysis:
            self._print_detailed_analysis(all_results)
        else:
            # Standard output for qualifying stocks only
            self._output_results(qualifying_stocks)
        
        # Generate HTML report if requested
        if generate_html_report:
            self._generate_html_report(all_results, reports_folder)
        
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
        current_price = float(latest_data['close'])
        
        # Calculate 52-week high and low
        # Get the last 252 trading days (approximately 1 year)
        one_year_data = data.tail(252) if len(data) >= 252 else data
        week_52_high = float(one_year_data['high'].max())
        week_52_low = float(one_year_data['low'].min())
        
        # Calculate percentage from 52-week high and low
        pct_from_high = ((current_price - week_52_high) / week_52_high * 100) if week_52_high > 0 else 0
        pct_from_low = ((current_price - week_52_low) / week_52_low * 100) if week_52_low > 0 else 0
        
        # Compile results
        result = {
            'symbol': symbol,
            'qualifies': qualifies,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'current_price': current_price,
            'current_volume': int(latest_data['volume']),
            
            # 52-week high/low data
            'week_52_high': week_52_high,
            'week_52_low': week_52_low,
            'pct_from_high': pct_from_high,
            'pct_from_low': pct_from_low,
            
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
    
    def _print_detailed_analysis(self, all_results: List[Dict[str, Any]]):
        """Print detailed analysis for all stocks."""
        print(f"\n{'='*80}")
        print(f"DETAILED TREND FOLLOWING ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Summary statistics
        total_stocks = len(all_results)
        qualified_stocks = len([r for r in all_results if r['qualifies']])
        
        print(f"📊 Summary: {qualified_stocks}/{total_stocks} stocks qualified ({(qualified_stocks/total_stocks*100) if total_stocks > 0 else 0:.1f}%)")
        print(f"{'='*80}")
        
        # Sort by trend strength for better readability
        sorted_results = sorted(all_results, key=lambda x: x['trend_strength'], reverse=True)
        
        for result in sorted_results:
            symbol = result['symbol']
            price = result['current_price']
            trend_details = result.get('trend_details', {})
            volume_details = result.get('volume_details', {})
            
            # Status indicator
            status_emoji = "✅" if result['qualifies'] else "❌"
            status_text = "QUALIFIED" if result['qualifies'] else "FAILED"
            
            print(f"\n{status_emoji} {symbol} - ${price:.2f} - {status_text}")
            
            # Trend analysis
            slope = trend_details.get('relative_slope', 0)
            slope_pass = slope >= self.config['min_slope']
            positive_days = trend_details.get('positive_days', 0)
            total_days = trend_details.get('total_days', 0)
            positive_days_pass = positive_days >= self.config['min_positive_days']
            ma_pass = trend_details.get('above_moving_average', False)
            
            print(f"   📈 Trend: slope={slope:.3f} ({'✅' if slope_pass else '❌'} need ≥{self.config['min_slope']:.3f}) | " +
                  f"positive_days={positive_days}/{total_days} ({'✅' if positive_days_pass else '❌'} need ≥{self.config['min_positive_days']})")
            
            # Volume analysis
            volume_ratio = volume_details.get('volume_ratio', 0)
            volume_pass = volume_ratio >= self.config['volume_threshold']
            
            print(f"   📊 Volume: {volume_ratio:.2f}x average ({'✅' if volume_pass else '❌'} need ≥{self.config['volume_threshold']:.2f}x)")
            
            # Moving average
            current_price = result['current_price']
            ma_price = trend_details.get('moving_average', 0)
            print(f"   📊 Above MA: ${current_price:.2f} > ${ma_price:.2f} ({'✅' if ma_pass else '❌'})")
            
            # Overall score
            criteria_passed = sum([slope_pass, positive_days_pass, ma_pass, volume_pass])
            print(f"   📋 Score: {criteria_passed}/4 criteria passed")
            
            # Failure reasons for non-qualifying stocks
            if not result['qualifies']:
                reasons = self._get_failure_reasons(result)
                if reasons:
                    print(f"   ❌ Failure reasons: {'; '.join(reasons)}")
        
        print(f"\n{'='*80}")
        
        # Failure pattern analysis
        self._print_failure_analysis(all_results)
    
    def _get_failure_reasons(self, result: Dict[str, Any]) -> List[str]:
        """Get specific failure reasons for a stock."""
        if result['qualifies']:
            return []
        
        reasons = []
        trend_details = result.get('trend_details', {})
        volume_details = result.get('volume_details', {})
        
        # Check trend criteria
        if trend_details.get('relative_slope', 0) < self.config['min_slope']:
            reasons.append(f"Slope too low ({trend_details.get('relative_slope', 0):.3f})")
        
        if trend_details.get('positive_days', 0) < self.config['min_positive_days']:
            reasons.append(f"Not enough positive days ({trend_details.get('positive_days', 0)})")
        
        if not trend_details.get('above_moving_average', False):
            reasons.append("Below moving average")
        
        # Check volume criteria
        if volume_details.get('volume_ratio', 0) < self.config['volume_threshold']:
            reasons.append(f"Volume too low ({volume_details.get('volume_ratio', 0):.2f}x)")
        
        return reasons
    
    def _print_failure_analysis(self, all_results: List[Dict[str, Any]]):
        """Print analysis of failure patterns."""
        failed_results = [r for r in all_results if not r['qualifies']]
        if not failed_results:
            return
        
        print(f"📊 FAILURE PATTERN ANALYSIS")
        print(f"{'='*40}")
        
        # Count failure reasons
        failure_counts = {
            'slope_too_low': 0,
            'insufficient_positive_days': 0,
            'below_moving_average': 0,
            'low_volume': 0
        }
        
        for result in failed_results:
            trend_details = result.get('trend_details', {})
            volume_details = result.get('volume_details', {})
            
            if trend_details.get('relative_slope', 0) < self.config['min_slope']:
                failure_counts['slope_too_low'] += 1
            
            if trend_details.get('positive_days', 0) < self.config['min_positive_days']:
                failure_counts['insufficient_positive_days'] += 1
            
            if not trend_details.get('above_moving_average', False):
                failure_counts['below_moving_average'] += 1
            
            if volume_details.get('volume_ratio', 0) < self.config['volume_threshold']:
                failure_counts['low_volume'] += 1
        
        total_failed = len(failed_results)
        
        print(f"🔻 Price Slope Too Low: {failure_counts['slope_too_low']} stocks ({failure_counts['slope_too_low']/total_failed*100:.1f}%)")
        print(f"📉 Not Enough Positive Days: {failure_counts['insufficient_positive_days']} stocks ({failure_counts['insufficient_positive_days']/total_failed*100:.1f}%)")
        print(f"📊 Below Moving Average: {failure_counts['below_moving_average']} stocks ({failure_counts['below_moving_average']/total_failed*100:.1f}%)")
        print(f"📊 Volume Too Low: {failure_counts['low_volume']} stocks ({failure_counts['low_volume']/total_failed*100:.1f}%)")
        print(f"{'='*40}")
    
    def _generate_html_report(self, all_results: List[Dict[str, Any]], reports_folder: str):
        """Generate HTML report with charts and detailed analysis."""
        try:
            html_generator = HTMLReportGenerator(reports_folder)
            
            # Create summary statistics
            summary = {
                'total_analyzed': len(all_results),
                'qualified': len([r for r in all_results if r['qualifies']]),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            html_generator.generate_report(all_results, self.config, summary)
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            print(f"Warning: Could not generate HTML report: {str(e)}")
    
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
            print(f"   52W High: ${result['week_52_high']:.2f} ({result['pct_from_high']:+.1f}%) | 52W Low: ${result['week_52_low']:.2f} ({result['pct_from_low']:+.1f}%)")
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

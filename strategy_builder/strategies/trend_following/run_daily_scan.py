"""
Daily Trend Following Stock Scanner.

This script runs the trend following strategy to screen stocks
for upward price trends with above-average volume. Designed to be
executed daily before market open.

Usage:
    python run_daily_scan.py [options]

Example:
    python run_daily_scan.py --config custom_config.json --stocks AAPL,MSFT,GOOGL
    python run_daily_scan.py --trend-method linear_regression --output both
    python run_daily_scan.py --show-all-analysis --generate-html-report
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add paths for direct imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..', '..')
root_dir = os.path.join(current_dir, '..', '..', '..')
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, root_dir)

# Try package imports first, then fall back to direct imports
try:
    from strategy_builder.strategies.trend_following.strategy import TrendFollowingStrategy
    from strategy_builder.strategies.trend_following.config import get_config, get_stock_list, DEFAULT_CONFIG
except ImportError:
    # Fall back to direct imports
    from strategy import TrendFollowingStrategy
    from config import get_config, get_stock_list, DEFAULT_CONFIG


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Daily Trend Following Stock Scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_daily_scan.py
  
  # Custom stock list
  python run_daily_scan.py --stocks AAPL,MSFT,GOOGL,AMZN,TSLA
  
  # Enhanced analysis with detailed output
  python run_daily_scan.py --show-all-analysis
  
  # Generate HTML report with charts
  python run_daily_scan.py --generate-html-report
  
  # Full analysis with custom parameters
  python run_daily_scan.py --show-all-analysis --generate-html-report --trend-method linear_regression
  
  # Custom reports folder
  python run_daily_scan.py --generate-html-report --reports-folder ./my_reports
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    
    # Stock list options
    parser.add_argument(
        '--stocks',
        type=str,
        help='Comma-separated list of stock symbols to analyze'
    )
    
    parser.add_argument(
        '--stock-list',
        choices=['popular', 'extended'],
        default='popular',
        help='Predefined stock list to use (default: popular)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--analysis-days',
        type=int,
        help=f'Number of recent days to analyze (default: {DEFAULT_CONFIG["analysis_days"]})'
    )
    
    parser.add_argument(
        '--volume-avg-days',
        type=int,
        help=f'Days for volume average calculation (default: {DEFAULT_CONFIG["volume_avg_days"]})'
    )
    
    parser.add_argument(
        '--volume-threshold',
        type=float,
        help=f'Volume threshold multiplier (default: {DEFAULT_CONFIG["volume_threshold"]})'
    )
    
    # Trend detection parameters
    parser.add_argument(
        '--trend-method',
        choices=['simple', 'linear_regression'],
        help=f'Trend detection method (default: {DEFAULT_CONFIG["trend_method"]})'
    )
    
    parser.add_argument(
        '--min-slope',
        type=float,
        help=f'Minimum price slope for trend (default: {DEFAULT_CONFIG["min_slope"]})'
    )
    
    parser.add_argument(
        '--min-positive-days',
        type=int,
        help=f'Minimum positive days required (default: {DEFAULT_CONFIG["min_positive_days"]})'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        choices=['console', 'csv', 'both'],
        help=f'Output format (default: {DEFAULT_CONFIG["output_format"]})'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help=f'CSV output filename (default: {DEFAULT_CONFIG["output_file"]})'
    )
    
    parser.add_argument(
        '--detailed-output',
        action='store_true',
        help='Include detailed metrics in output'
    )
    
    # Enhanced reporting options
    parser.add_argument(
        '--show-all-analysis',
        action='store_true',
        help='Show detailed analysis for ALL stocks (not just qualifiers) with failure reasons'
    )
    
    parser.add_argument(
        '--generate-html-report',
        action='store_true',
        default=True,
        help='Generate comprehensive HTML report with charts and individual stock pages (enabled by default)'
    )
    
    parser.add_argument(
        '--reports-folder',
        type=str,
        default=None,
        help='Folder for HTML reports (default: REPORTS_DIR env var or "reports")'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help=f'Logging level (default: {DEFAULT_CONFIG["log_level"]})'
    )
    
    # API key
    parser.add_argument(
        '--api-key',
        type=str,
        help='FMP API key (can also use FMP_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (show configuration without executing)'
    )
    
    return parser.parse_args()


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)


def build_config(args) -> Dict[str, Any]:
    """Build configuration from arguments and config file."""
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Load from config file if provided
    if args.config:
        file_config = load_config_file(args.config)
        config.update(file_config)
    
    # Override with command line arguments
    if args.analysis_days is not None:
        config['analysis_days'] = args.analysis_days
    if args.volume_avg_days is not None:
        config['volume_avg_days'] = args.volume_avg_days
    if args.volume_threshold is not None:
        config['volume_threshold'] = args.volume_threshold
    if args.trend_method is not None:
        config['trend_method'] = args.trend_method
    if args.min_slope is not None:
        config['min_slope'] = args.min_slope
    if args.min_positive_days is not None:
        config['min_positive_days'] = args.min_positive_days
    if args.output is not None:
        config['output_format'] = args.output
    if args.output_file is not None:
        config['output_file'] = args.output_file
    if args.detailed_output:
        config['detailed_output'] = True
    if args.log_level is not None:
        config['log_level'] = args.log_level
    
    return config


def build_stock_list(args) -> List[str]:
    """Build stock list from arguments."""
    if args.stocks:
        # Parse comma-separated stock list
        return [stock.strip().upper() for stock in args.stocks.split(',')]
    else:
        # Use predefined list
        return get_stock_list(args.stock_list)


def print_configuration(config: Dict[str, Any], stock_list: List[str], args):
    """Print the current configuration."""
    print("=" * 60)
    print("TREND FOLLOWING SCANNER CONFIGURATION")
    print("=" * 60)
    print(f"Analysis Days: {config['analysis_days']}")
    print(f"Volume Average Days: {config['volume_avg_days']}")
    print(f"Volume Threshold: {config['volume_threshold']:.1f}x")
    print(f"Trend Method: {config['trend_method']}")
    print(f"Minimum Slope: {config['min_slope']:.3f}")
    print(f"Minimum Positive Days: {config['min_positive_days']}")
    print(f"Output Format: {config['output_format']}")
    if config['output_format'] in ['csv', 'both']:
        print(f"Output File: {config['output_file']}")
    print(f"Log Level: {config['log_level']}")
    print(f"Stocks to Analyze: {len(stock_list)} ({', '.join(stock_list[:5])}{'...' if len(stock_list) > 5 else ''})")
    
    # Enhanced reporting options
    if args.show_all_analysis:
        print(f"Enhanced Analysis: ✅ Show detailed analysis for ALL stocks")
    if args.generate_html_report:
        reports_folder = args.reports_folder or os.getenv('REPORTS_DIR', 'reports')
        print(f"HTML Report: ✅ Generate in folder '{reports_folder}'")
    
    print("=" * 60)
    print()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Build configuration and stock list
    config = build_config(args)
    stock_list = build_stock_list(args)
    
    # Get API key
    api_key = args.api_key or os.getenv('FMP_API_KEY')
    if not api_key:
        print("Error: FMP API key is required. Set FMP_API_KEY environment variable or use --api-key option.")
        sys.exit(1)
    
    # Print configuration
    print_configuration(config, stock_list, args)
    
    # Dry run mode - just show configuration
    if args.dry_run:
        print("Dry run mode - configuration shown above. Exiting without execution.")
        return
    
    # Check for matplotlib if HTML report is requested
    if args.generate_html_report:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib is required for HTML report generation.")
            print("Install it with: pip install matplotlib")
            print("Continuing without HTML report generation...")
            args.generate_html_report = False
    
    try:
        # Initialize and run the strategy
        print("Initializing Trend Following Strategy...")
        strategy = TrendFollowingStrategy(
            config=config,
            stock_list=stock_list,
            api_key=api_key
        )
        
        print("Starting daily screening process...")
        
        # Get reports folder from environment if not specified
        reports_folder = args.reports_folder
        if reports_folder is None:
            reports_folder = os.getenv('REPORTS_DIR', 'reports')
        
        # Run with enhanced options
        results = strategy.run_daily_screening(
            show_all_analysis=args.show_all_analysis,
            generate_html_report=args.generate_html_report,
            reports_folder=reports_folder
        )
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 SCREENING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Analyzed: {len(stock_list)} stocks")
        print(f"🎯 Qualified: {len(results)} stocks meeting trend following criteria")
        
        if results:
            qualifying_symbols = [result['symbol'] for result in results]
            print(f"🔥 Qualifying stocks: {', '.join(qualifying_symbols)}")
        else:
            print("❌ No stocks currently meet the trend following criteria")
            if not args.show_all_analysis:
                print("💡 Use --show-all-analysis to see why stocks failed")
        
        # Report information
        if args.generate_html_report:
            print(f"📄 HTML report generated in: {reports_folder}/")
        
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\nScreening interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during screening: {str(e)}")
        if config.get('log_level') == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

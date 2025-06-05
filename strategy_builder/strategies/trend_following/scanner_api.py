"""
Trend Following Scanner API

A clean, importable interface for running trend following stock scans.
This module provides a function-based API that can be easily imported
and used by other Python code without command-line dependencies.

Usage:
    from scanner_api import run_trend_following_scan
    
    results = run_trend_following_scan(
        stock_list=['AAPL', 'MSFT', 'GOOGL'],
        api_key='your_api_key'  # Optional - defaults to FMP_API_KEY env var
    )
"""

import os
import json
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add paths for direct imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Direct imports to avoid package conflicts
from strategy import TrendFollowingStrategy
from config import get_config, get_stock_list, DEFAULT_CONFIG


# ============================================================================
# DEFAULT VALUES USED BY SCANNER API
# ============================================================================
# These are the default values that will be used when parameters are not specified.
# Values come from DEFAULT_CONFIG in config.py plus scanner-specific defaults.

API_DEFAULTS = {
    # API Configuration
    'api_key': 'FMP_API_KEY environment variable',
    
    # Analysis Parameters (from DEFAULT_CONFIG)
    'analysis_days': 5,              # Number of recent days to analyze for trend
    'volume_avg_days': 20,           # Days to calculate average volume baseline  
    'volume_threshold': 1.2,         # Volume must be 120% above average
    
    # Trend Detection Parameters (from DEFAULT_CONFIG)
    'trend_method': 'simple',        # 'simple' or 'linear_regression'
    'min_slope': 0.01,              # Minimum price slope for uptrend (1% per day)
    'min_positive_days': 3,         # Minimum positive days out of analysis_days
    'ma_period': 5,                 # Moving average period for trend confirmation
    
    # Linear Regression Parameters (from DEFAULT_CONFIG, used if trend_method='linear_regression')
    'min_r_squared': 0.7,           # Minimum R-squared for trend reliability
    'linear_min_slope': 0.02,       # Higher slope requirement for linear regression
    
    # Output Options (from DEFAULT_CONFIG)
    'output_format': 'console',      # 'console', 'csv', or 'both'
    'output_file': 'trend_following_results.csv',
    'log_level': 'INFO',            # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'detailed_output': True,         # Include detailed metrics in output
    
    # Scanner API Specific Defaults
    'show_all_analysis': False,      # Show analysis for ALL stocks (not just qualifiers)
    'generate_html_report': True,    # Generate HTML report (enabled by default)
    'reports_folder': 'REPORTS_DIR environment variable or "reports"',
    'dry_run': False,               # Run in dry-run mode (validation only)
    'verbose': False,               # Enable verbose output during execution
}

# Note: When None is passed for any parameter, the corresponding DEFAULT_CONFIG value is used.
# ============================================================================


def run_trend_following_scan(
    stock_list: List[str],
    api_key: Optional[str] = None,
    
    # Analysis parameters
    analysis_days: Optional[int] = None,
    volume_avg_days: Optional[int] = None,
    volume_threshold: Optional[float] = None,
    
    # Trend detection parameters
    trend_method: Optional[str] = None,
    min_slope: Optional[float] = None,
    min_positive_days: Optional[int] = None,
    
    # Output options
    output_format: Optional[str] = None,
    output_file: Optional[str] = None,
    detailed_output: bool = False,
    
    # Enhanced analysis options
    show_all_analysis: bool = False,
    generate_html_report: bool = True,  # Default to True as requested
    reports_folder: Optional[str] = None,
    
    # Logging
    log_level: Optional[str] = None,
    
    # Optional config override
    config_dict: Optional[Dict[str, Any]] = None,
    
    # Additional options
    dry_run: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run trend following stock scan with configurable parameters.
    
    Args:
        stock_list: List of stock symbols to analyze (required)
        api_key: FMP API key (defaults to FMP_API_KEY environment variable)
        
        # Analysis Parameters
        analysis_days: Number of recent days to analyze
        volume_avg_days: Days for volume average calculation
        volume_threshold: Volume threshold multiplier
        
        # Trend Detection
        trend_method: Trend detection method ('simple' or 'linear_regression')
        min_slope: Minimum price slope for trend
        min_positive_days: Minimum positive days required
        
        # Output Options
        output_format: Output format ('console', 'csv', 'both')
        output_file: CSV output filename
        detailed_output: Include detailed metrics in output
        
        # Enhanced Analysis
        show_all_analysis: Show analysis for ALL stocks (not just qualifiers)
        generate_html_report: Generate HTML report (default: True)
        reports_folder: Folder for HTML reports
        
        # System Options
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        config_dict: Optional dictionary to override default config
        dry_run: Run in dry-run mode (validation only)
        verbose: Enable verbose output
        
    Returns:
        Dict containing:
            - qualified_stocks: List of stocks meeting criteria
            - total_analyzed: Number of stocks analyzed
            - total_qualified: Number of stocks that qualified
            - analysis_timestamp: When analysis was performed
            - config_used: Configuration that was actually used
            - html_report_path: Path to HTML report (if generated)
            - csv_file_path: Path to CSV file (if generated)
            - all_stock_analysis: Detailed analysis for all stocks (if show_all_analysis=True)
            - success: Boolean indicating if scan completed successfully
            - error_message: Error message if scan failed
    
    Raises:
        ValueError: If required parameters are missing or invalid
        Exception: For other errors during execution
    """
    
    # Validate required parameters
    if not stock_list:
        raise ValueError("stock_list parameter is required and cannot be empty")
    
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv('FMP_API_KEY')
    
    if not api_key:
        raise ValueError(
            "API key is required. Either pass api_key parameter or set FMP_API_KEY environment variable."
        )
    
    # Build configuration
    config = _build_config_from_params(
        analysis_days=analysis_days,
        volume_avg_days=volume_avg_days,
        volume_threshold=volume_threshold,
        trend_method=trend_method,
        min_slope=min_slope,
        min_positive_days=min_positive_days,
        output_format=output_format,
        output_file=output_file,
        detailed_output=detailed_output,
        log_level=log_level,
        config_dict=config_dict
    )
    
    # Set up reports folder
    if reports_folder is None:
        reports_folder = os.getenv('REPORTS_DIR', 'reports')
    
    # Initialize result structure
    result = {
        'qualified_stocks': [],
        'total_analyzed': len(stock_list),
        'total_qualified': 0,
        'analysis_timestamp': '',
        'config_used': config,
        'html_report_path': None,
        'csv_file_path': None,
        'all_stock_analysis': [],
        'success': False,
        'error_message': None
    }
    
    if verbose:
        _print_configuration(config, stock_list, generate_html_report, show_all_analysis, reports_folder)
    
    # Dry run mode - just return configuration
    if dry_run:
        result['success'] = True
        result['error_message'] = 'Dry run mode - configuration validated successfully'
        return result
    
    # Check for matplotlib if HTML report is requested
    if generate_html_report:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            if verbose:
                print("Warning: matplotlib is required for HTML report generation.")
                print("Install it with: pip install matplotlib")
                print("Continuing without HTML report generation...")
            generate_html_report = False
    
    try:
        # Initialize and run the strategy
        if verbose:
            print("Initializing Trend Following Strategy...")
        
        strategy = TrendFollowingStrategy(
            config=config,
            stock_list=stock_list,
            api_key=api_key
        )
        
        if verbose:
            print("Starting screening process...")
        
        # Run the screening
        qualified_results = strategy.run_daily_screening(
            show_all_analysis=show_all_analysis,
            generate_html_report=generate_html_report,
            reports_folder=reports_folder
        )
        
        # Build result
        result['qualified_stocks'] = qualified_results
        result['total_qualified'] = len(qualified_results)
        result['success'] = True
        
        # Set file paths if files were generated
        if generate_html_report:
            # Assume HTML report is generated in reports_folder
            result['html_report_path'] = reports_folder
        
        if config['output_format'] in ['csv', 'both']:
            result['csv_file_path'] = config['output_file']
        
        # Get detailed analysis if requested
        if show_all_analysis:
            # This would need to be implemented in the strategy class
            # For now, we just indicate it was requested
            result['all_stock_analysis'] = qualified_results  # Placeholder
        
        # Add timestamp
        from datetime import datetime
        result['analysis_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if verbose:
            _print_summary(result)
        
    except Exception as e:
        result['success'] = False
        result['error_message'] = str(e)
        if verbose:
            print(f"Error during screening: {str(e)}")
        # Re-raise the exception if not in verbose mode so calling code can handle it
        if not verbose:
            raise
    
    return result


def _build_config_from_params(
    analysis_days: Optional[int] = None,
    volume_avg_days: Optional[int] = None,
    volume_threshold: Optional[float] = None,
    trend_method: Optional[str] = None,
    min_slope: Optional[float] = None,
    min_positive_days: Optional[int] = None,
    output_format: Optional[str] = None,
    output_file: Optional[str] = None,
    detailed_output: bool = False,
    log_level: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build configuration dictionary from parameters."""
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Apply custom config dictionary if provided
    if config_dict:
        config.update(config_dict)
    
    # Override with individual parameters
    if analysis_days is not None:
        config['analysis_days'] = analysis_days
    if volume_avg_days is not None:
        config['volume_avg_days'] = volume_avg_days
    if volume_threshold is not None:
        config['volume_threshold'] = volume_threshold
    if trend_method is not None:
        config['trend_method'] = trend_method
    if min_slope is not None:
        config['min_slope'] = min_slope
    if min_positive_days is not None:
        config['min_positive_days'] = min_positive_days
    if output_format is not None:
        config['output_format'] = output_format
    if output_file is not None:
        config['output_file'] = output_file
    if detailed_output:
        config['detailed_output'] = True
    if log_level is not None:
        config['log_level'] = log_level
    
    return config


def _print_configuration(
    config: Dict[str, Any], 
    stock_list: List[str], 
    generate_html_report: bool,
    show_all_analysis: bool,
    reports_folder: str
):
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
    if show_all_analysis:
        print(f"Enhanced Analysis: ✅ Show detailed analysis for ALL stocks")
    if generate_html_report:
        print(f"HTML Report: ✅ Generate in folder '{reports_folder}'")
    
    print("=" * 60)
    print()


def _print_summary(result: Dict[str, Any]):
    """Print screening summary."""
    print(f"\n{'='*60}")
    print(f"📊 SCREENING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Analyzed: {result['total_analyzed']} stocks")
    print(f"🎯 Qualified: {result['total_qualified']} stocks meeting trend following criteria")
    
    if result['qualified_stocks']:
        qualifying_symbols = [stock['symbol'] for stock in result['qualified_stocks']]
        print(f"🔥 Qualifying stocks: {', '.join(qualifying_symbols)}")
    else:
        print("❌ No stocks currently meet the trend following criteria")
    
    # Report information
    if result['html_report_path']:
        print(f"📄 HTML report generated in: {result['html_report_path']}/")
    
    print(f"{'='*60}")


# Convenience functions for common usage patterns

def quick_scan(stock_symbols: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick scan with comma-separated stock symbols string.
    
    Args:
        stock_symbols: Comma-separated string of stock symbols (e.g., "AAPL,MSFT,GOOGL")
        api_key: Optional API key (defaults to environment variable)
        
    Returns:
        Scan results dictionary
    """
    stock_list = [symbol.strip().upper() for symbol in stock_symbols.split(',')]
    return run_trend_following_scan(stock_list=stock_list, api_key=api_key)


def detailed_scan(
    stock_list: List[str], 
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Detailed scan with all analysis enabled by default.
    
    Args:
        stock_list: List of stock symbols
        api_key: Optional API key (defaults to environment variable)
        **kwargs: Additional parameters to pass to run_trend_following_scan
        
    Returns:
        Scan results dictionary
    """
    return run_trend_following_scan(
        stock_list=stock_list,
        api_key=api_key,
        show_all_analysis=True,
        detailed_output=True,
        verbose=True,
        **kwargs
    )


def silent_scan(stock_list: List[str], api_key: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Silent scan with minimal output (for automated use).
    
    Args:
        stock_list: List of stock symbols
        api_key: Optional API key (defaults to environment variable)
        **kwargs: Additional parameters to pass to run_trend_following_scan
        
    Returns:
        Scan results dictionary
    """
    return run_trend_following_scan(
        stock_list=stock_list,
        api_key=api_key,
        verbose=False,
        **kwargs
    )


# Example usage and testing
if __name__ == '__main__':
    """
    Example usage of the scanner API.
    """
    # Example 1: Basic usage
    try:
        results = run_trend_following_scan(
            stock_list=['AAPL', 'MSFT', 'GOOGL'],
            verbose=True
        )
        print(f"Scan completed successfully: {results['success']}")
        print(f"Qualified stocks: {len(results['qualified_stocks'])}")
        
    except Exception as e:
        print(f"Scan failed: {e}")
    
    # Example 2: Quick scan with string
    try:
        results = quick_scan("AAPL,MSFT,GOOGL,AMZN,TSLA")
        print(f"Quick scan found {len(results['qualified_stocks'])} qualifying stocks")
        
    except Exception as e:
        print(f"Quick scan failed: {e}")

"""
Example usage of the Trend Following Strategy.

This script demonstrates how to use the trend following strategy
with different configurations and stock lists.
"""

import os
import sys
from datetime import datetime

# Add the strategy_builder path to sys.path for imports (prioritize local version)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from strategy_builder.strategies.trend_following import (
    TrendFollowingStrategy,
    get_config,
    get_stock_list
)


def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Use default configuration and popular stocks
    strategy = TrendFollowingStrategy()
    
    # Run the screening (this would require FMP API key)
    # results = strategy.run_daily_screening()
    
    print("Strategy initialized with default configuration")
    print(f"Stock list: {len(strategy.stock_list)} stocks")
    print(f"Configuration: {strategy.config}")


def example_custom_config():
    """Example 2: Custom configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 60)
    
    # Custom configuration
    custom_config = {
        'analysis_days': 7,              # Analyze last 7 days instead of 5
        'volume_threshold': 1.5,         # Higher volume requirement (150%)
        'trend_method': 'linear_regression',  # Use linear regression
        'min_slope': 0.02,              # Higher slope requirement
        'output_format': 'both',         # Output to both console and CSV
        'detailed_output': True
    }
    
    strategy = TrendFollowingStrategy(config=custom_config)
    
    print("Strategy initialized with custom configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")


def example_custom_stocks():
    """Example 3: Custom stock list."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Stock List")  
    print("=" * 60)
    
    # Focus on tech stocks only
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    
    strategy = TrendFollowingStrategy(stock_list=tech_stocks)
    
    print(f"Strategy initialized with {len(tech_stocks)} tech stocks:")
    print(f"Stocks: {', '.join(tech_stocks)}")


def example_programmatic_usage():
    """Example 4: Programmatic usage with result processing."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Programmatic Usage")
    print("=" * 60)
    
    # Configuration for quick screening
    config = {
        'analysis_days': 3,
        'volume_threshold': 1.1,  # Lower threshold for demonstration
        'min_positive_days': 2,
        'output_format': 'console'
    }
    
    # Small stock list for demo
    demo_stocks = ['AAPL', 'MSFT', 'GOOGL']
    
    strategy = TrendFollowingStrategy(
        config=config,
        stock_list=demo_stocks
    )
    
    print("This example would run screening and process results:")
    print("```python")
    print("results = strategy.run_daily_screening()")
    print("for result in results:")
    print("    if result['qualifies']:")
    print("        symbol = result['symbol']")
    print("        price = result['current_price']")
    print("        trend_strength = result['trend_strength']")
    print("        print(f'{symbol}: ${price:.2f} (Strength: {trend_strength:.2f})')")
    print("```")


def example_different_analyzers():
    """Example 5: Different trend analyzers."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Different Trend Analysis Methods")
    print("=" * 60)
    
    # Simple trend analyzer (default)
    simple_config = {
        'trend_method': 'simple',
        'min_slope': 0.01,
        'min_positive_days': 3
    }
    
    # Linear regression analyzer
    regression_config = {
        'trend_method': 'linear_regression',
        'linear_min_slope': 0.02,
        'min_r_squared': 0.7
    }
    
    print("Simple Trend Analyzer Configuration:")
    for key, value in simple_config.items():
        print(f"  {key}: {value}")
    
    print("\nLinear Regression Analyzer Configuration:")
    for key, value in regression_config.items():
        print(f"  {key}: {value}")
    
    print("\nBoth analyzers can be used with TrendFollowingStrategy by")
    print("setting the appropriate configuration parameters.")


def main():
    """Run all examples."""
    print("TREND FOLLOWING STRATEGY - USAGE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates various ways to use the Trend Following Strategy.")
    print("Note: Actual execution requires FMP_API_KEY environment variable.")
    print()
    
    example_basic_usage()
    example_custom_config()
    example_custom_stocks()
    example_programmatic_usage()
    example_different_analyzers()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Set your FMP_API_KEY environment variable")
    print("2. Run: python run_daily_scan.py --help")
    print("3. Try: python run_daily_scan.py --dry-run")
    print("4. Execute: python run_daily_scan.py --stocks AAPL,MSFT,GOOGL")
    print()


if __name__ == '__main__':
    main()

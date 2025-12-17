#!/usr/bin/env python3
"""Run Gap Trading Backtest.

Usage:
    python -m strategy_builder.strategies.gap_trading.backtest_v2.scripts.run_backtest \
        --start 2025-11-01 --end 2025-12-15 \
        --capital 100000 \
        --symbols SPY QQQ AAPL MSFT

    # With parameter sweep
    python -m strategy_builder.strategies.gap_trading.backtest_v2.scripts.run_backtest \
        --start 2025-11-01 --end 2025-12-15 \
        --sweep
"""

import argparse
import logging
import os
import sys
from datetime import datetime, date
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from strategy_builder.strategies.gap_trading.backtest_v2 import (
    BacktestConfig,
    BacktestEngine,
    GapBacktestDataLoader,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Gap Trading Strategy Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python run_backtest.py --start 2025-11-01 --end 2025-12-15

  # With specific symbols
  python run_backtest.py --start 2025-11-01 --end 2025-12-15 --symbols SPY QQQ AAPL

  # Parameter sweep
  python run_backtest.py --start 2025-11-01 --end 2025-12-15 --sweep

  # Save results
  python run_backtest.py --start 2025-11-01 --end 2025-12-15 --output results.json
        """
    )

    # Required arguments
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )

    # Optional arguments
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital (default: 100000)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Symbols to backtest (default: diversified list)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.environ.get('FMP_API_KEY'),
        help='FMP API key (default: from FMP_API_KEY env var)'
    )

    # Strategy parameters
    parser.add_argument(
        '--min-gap',
        type=float,
        default=1.5,
        help='Minimum gap percentage (default: 1.5)'
    )
    parser.add_argument(
        '--max-gap',
        type=float,
        default=10.0,
        help='Maximum gap percentage (default: 10.0)'
    )
    parser.add_argument(
        '--confirmation-minutes',
        type=int,
        default=10,
        help='Minutes after open for confirmation (default: 10)'
    )
    parser.add_argument(
        '--stop-multiplier',
        type=float,
        default=1.5,
        help='ATR multiplier for stops (default: 1.5)'
    )
    parser.add_argument(
        '--no-risk-tiers',
        action='store_true',
        help='Disable risk tier position sizing'
    )
    parser.add_argument(
        '--no-minute-data',
        action='store_true',
        help='Use daily data only (faster, less accurate)'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (JSON)'
    )
    parser.add_argument(
        '--trades-csv',
        type=str,
        help='Output file for trades (CSV)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )

    # Analysis modes
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run parameter sweep'
    )
    parser.add_argument(
        '--analyze-gaps',
        action='store_true',
        help='Run gap continuation analysis'
    )

    return parser.parse_args()


def run_standard_backtest(args) -> None:
    """Run standard single backtest."""
    logger.info("=" * 60)
    logger.info("GAP TRADING BACKTEST")
    logger.info("=" * 60)

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: ${args.capital:,.2f}")
    logger.info(f"Gap Range: {args.min_gap}% - {args.max_gap}%")
    logger.info(f"Confirmation: {args.confirmation_minutes} minutes after open")

    # Create config
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        symbols=args.symbols,
        min_gap_pct=args.min_gap,
        max_gap_pct=args.max_gap,
        confirmation_minutes=args.confirmation_minutes,
        stop_atr_multiplier=args.stop_multiplier,
        use_risk_tiers=not args.no_risk_tiers,
        use_minute_data=not args.no_minute_data,
    )

    # Create and run backtest
    engine = BacktestEngine(config, api_key=args.api_key)

    def progress(trade_date, day_num, total):
        if args.verbose or day_num % 10 == 0:
            logger.info(f"Processing {trade_date} ({day_num}/{total})")

    result = engine.run(args.symbols, progress_callback=progress)

    # Print summary
    print(result.summary())

    # Print rejection analysis
    if result.rejection_analysis:
        print("\nSignal Rejection Analysis:")
        for reason, count in sorted(result.rejection_analysis.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Save results if requested
    if args.output:
        result.save_to_json(args.output)
        print(f"\nResults saved to {args.output}")

    if args.trades_csv:
        result.save_trades_csv(args.trades_csv)
        print(f"Trades saved to {args.trades_csv}")


def run_parameter_sweep(args) -> None:
    """Run parameter sweep optimization."""
    logger.info("=" * 60)
    logger.info("GAP TRADING PARAMETER SWEEP")
    logger.info("=" * 60)

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    # Base config
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        symbols=args.symbols,
    )

    # Parameter ranges to sweep
    param_ranges = {
        'min_gap_pct': [1.0, 1.5, 2.0, 2.5],
        'max_gap_pct': [7.5, 10.0, 15.0],
        'confirmation_minutes': [5, 10, 15, 20],
        'stop_atr_multiplier': [1.0, 1.5, 2.0],
    }

    logger.info(f"Sweeping parameters: {list(param_ranges.keys())}")

    # Run sweep
    engine = BacktestEngine(config, api_key=args.api_key)
    results_df = engine.run_parameter_sweep(param_ranges, args.symbols)

    # Print results
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 80)

    # Sort by total return
    results_df = results_df.sort_values('total_return_pct', ascending=False)
    print(results_df.to_string(index=False))

    # Best configuration
    print("\n" + "-" * 80)
    print("BEST CONFIGURATION:")
    best = results_df.iloc[0]
    print(f"  Min Gap: {best['min_gap_pct']}%")
    print(f"  Max Gap: {best['max_gap_pct']}%")
    print(f"  Confirmation: {best['confirmation_minutes']} minutes")
    print(f"  Stop Multiplier: {best['stop_atr_multiplier']}x ATR")
    print(f"  Return: {best['total_return_pct']:+.1f}%")
    print(f"  Sharpe: {best['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {best['win_rate']:.1f}%")

    # Save if requested
    if args.output:
        results_df.to_csv(args.output.replace('.json', '.csv'), index=False)
        print(f"\nSweep results saved to {args.output.replace('.json', '.csv')}")


def run_gap_analysis(args) -> None:
    """Run gap continuation analysis."""
    from strategy_builder.strategies.gap_trading.backtest_v2.signal_engine import (
        analyze_gap_continuation
    )

    logger.info("=" * 60)
    logger.info("GAP CONTINUATION ANALYSIS")
    logger.info("=" * 60)

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    # Get symbols
    symbols = args.symbols or [
        'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'
    ]

    # Create data loader
    data_loader = GapBacktestDataLoader(api_key=args.api_key)

    # Run analysis
    logger.info(f"Analyzing {len(symbols)} symbols from {start_date} to {end_date}")
    results = analyze_gap_continuation(
        data_loader,
        symbols,
        start_date,
        end_date,
        min_gap_pct=args.min_gap
    )

    # Print results
    print("\n" + "=" * 60)
    print("GAP CONTINUATION ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nTotal Gaps Analyzed: {results['total_gaps']}")

    print("\nContinuation Rate by Time After Open:")
    print("-" * 40)
    for minutes, data in sorted(results['by_time'].items()):
        total = data['continued'] + data['reversed']
        rate = data.get('rate', 0)
        print(f"  {minutes:2d} min: {rate:5.1f}% ({data['continued']}/{total})")

    print("\nContinuation Rate by Direction:")
    print("-" * 40)
    for direction, data in results['by_direction'].items():
        rate = data.get('rate', 0)
        print(f"  {direction}: {rate:.1f}% ({data['continued']}/{data['total']})")


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.sweep:
            run_parameter_sweep(args)
        elif args.analyze_gaps:
            run_gap_analysis(args)
        else:
            run_standard_backtest(args)

    except KeyboardInterrupt:
        print("\nBacktest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

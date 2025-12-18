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

    # With detailed file logging
    python -m strategy_builder.strategies.gap_trading.backtest_v2.scripts.run_backtest \
        --start 2025-11-01 --end 2025-12-15 \
        --log-file backtest.log \
        --verbose
"""

import argparse
import logging
from logging.handlers import RotatingFileHandler
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

logger = logging.getLogger(__name__)

# Default logs directory (relative to script location or current working dir)
DEFAULT_LOGS_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / 'logs'


class TradeLogFilter(logging.Filter):
    """Filter to only allow logs from main backtest script."""

    def filter(self, record):
        # Only allow logs from __main__ (this script)
        return record.name == '__main__'


def get_default_log_path() -> Path:
    """Generate default log file path with timestamp.

    Returns:
        Path like: logs/backtest_20251217_143052.log
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return DEFAULT_LOGS_DIR / f'backtest_{timestamp}.log'


def setup_logging(log_file: str = 'auto', verbose: bool = False) -> str:
    """Configure logging with optional file output.

    Args:
        log_file: Path to log file, 'auto' for default, 'none' to disable file logging.
        verbose: If True, sets DEBUG level logging.

    Returns:
        The actual log file path used (or None if disabled).

    The file logger only captures trade-related logs from this script,
    filtering out noisy data fetching and API call logs.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Determine actual log file path
    actual_log_file = None
    if log_file and log_file.lower() != 'none':
        if log_file == 'auto':
            actual_log_file = get_default_log_path()
        else:
            actual_log_file = Path(log_file)

    # Clean format for trade logging (no module names, just timestamp and message)
    trade_format = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console format (simpler)
    console_format = logging.Formatter('%(message)s')

    # Get our script's logger only
    script_logger = logging.getLogger(__name__)
    script_logger.setLevel(log_level)
    script_logger.handlers.clear()
    script_logger.propagate = False  # Don't propagate to root logger

    # Console handler for script logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_format)
    script_logger.addHandler(console_handler)

    # Suppress all other loggers (set to CRITICAL to hide everything)
    logging.getLogger().setLevel(logging.CRITICAL)  # Root logger

    # File handler (if specified) - only for trade logs
    if actual_log_file:
        actual_log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            str(actual_log_file),
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(trade_format)
        file_handler.addFilter(TradeLogFilter())  # Only __main__ logs
        script_logger.addHandler(file_handler)

        script_logger.info(f"Log file: {actual_log_file}")

    return str(actual_log_file) if actual_log_file else None


def log_trade_detail(trade, trade_num: int, total_trades: int) -> None:
    """Log detailed trade information with market times."""
    # Format: TRADE #N | DATE | SYMBOL | DIRECTION
    logger.info("")
    logger.info(f"TRADE #{trade_num}/{total_trades} | {trade.trade_date} | {trade.symbol} | {trade.direction}")
    logger.info("-" * 60)

    # Key decision parameters
    logger.info(f"  Gap: {trade.gap_pct:+.2f}%")
    if hasattr(trade, 'atr') and trade.atr:
        logger.info(f"  ATR: ${trade.atr:.2f}")
    if hasattr(trade, 'risk_tier') and trade.risk_tier:
        logger.info(f"  Risk Tier: {trade.risk_tier}")

    # Entry details (market time)
    logger.info(f"  ENTRY @ {trade.entry_time} | Price: ${trade.entry_price:.2f} | Shares: {trade.shares}")
    position_value = trade.entry_price * trade.shares
    logger.info(f"  Position Value: ${position_value:,.2f}")
    if hasattr(trade, 'stop_price') and trade.stop_price:
        logger.info(f"  Stop Price: ${trade.stop_price:.2f}")

    # Exit details (market time)
    exit_reason = str(trade.exit_reason).replace('ExitReason.', '')
    logger.info(f"  EXIT @ {trade.exit_time} | Price: ${trade.exit_price:.2f} | Reason: {exit_reason}")

    # Result
    result_emoji = "WIN" if trade.pnl > 0 else "LOSS"
    logger.info(f"  RESULT: {result_emoji} | P&L: ${trade.pnl:+,.2f} ({trade.pnl_pct:+.2f}%)")
    if hasattr(trade, 'hold_duration_minutes') and trade.hold_duration_minutes:
        logger.info(f"  Hold Duration: {trade.hold_duration_minutes} min")
    logger.info("")


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
    parser.add_argument(
        '--log-file',
        type=str,
        default='auto',
        help='Log file path (default: logs/backtest_TIMESTAMP.log, use "none" to disable)'
    )
    parser.add_argument(
        '--log-trades',
        action='store_true',
        help='Log detailed information for each trade'
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
    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    # Log configuration header
    logger.info("=" * 60)
    logger.info("GAP TRADING BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Symbols: {args.symbols or 'Default universe'}")
    logger.info(f"Capital: ${args.capital:,.0f} | Gap: {args.min_gap}-{args.max_gap}% | Confirm: {args.confirmation_minutes}min | Stop: {args.stop_multiplier}x ATR")
    logger.info("=" * 60)

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

    # Create and run backtest (no progress callback to reduce noise)
    engine = BacktestEngine(config, api_key=args.api_key)
    logger.info("Loading data and running backtest...")
    result = engine.run(args.symbols)

    # Log detailed trade information (main purpose of file logging)
    if args.log_trades and result.trades:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"TRADE LOG ({len(result.trades)} trades)")
        logger.info("=" * 60)
        for i, trade in enumerate(result.trades, 1):
            log_trade_detail(trade, i, len(result.trades))

    # Log daily summary
    if result.trades:
        logger.info("=" * 60)
        logger.info("DAILY SUMMARY")
        logger.info("=" * 60)
        trades_by_date = {}
        for trade in result.trades:
            trade_date = trade.trade_date
            if trade_date not in trades_by_date:
                trades_by_date[trade_date] = []
            trades_by_date[trade_date].append(trade)

        cumulative_pnl = 0
        for trade_date in sorted(trades_by_date.keys()):
            day_trades = trades_by_date[trade_date]
            day_pnl = sum(t.pnl for t in day_trades)
            cumulative_pnl += day_pnl
            winners = sum(1 for t in day_trades if t.pnl > 0)
            losers = sum(1 for t in day_trades if t.pnl < 0)
            logger.info(f"{trade_date} | {len(day_trades)} trades | W:{winners} L:{losers} | Day: ${day_pnl:+,.2f} | Total: ${cumulative_pnl:+,.2f}")

    # Log final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if result.metrics:
        m = result.metrics
        logger.info(f"Return: {m.total_return_pct:+.2f}% | Win Rate: {m.win_rate:.1f}% | Trades: {m.total_trades}")
        logger.info(f"Profit Factor: {m.profit_factor:.2f} | Sharpe: {m.sharpe_ratio:.2f} | Max DD: {m.max_drawdown_pct:.2f}%")
        logger.info(f"Avg Trade: ${m.avg_trade:,.2f} | Best: ${m.largest_win:,.2f} | Worst: ${m.largest_loss:,.2f}")

    # Print summary to console
    print(result.summary())

    # Print rejection analysis
    if result.rejection_analysis:
        print("\nSignal Rejection Analysis:")
        for reason, count in sorted(result.rejection_analysis.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Save results if requested
    if args.output:
        result.save_to_json(args.output)
        logger.info(f"Results saved to {args.output}")
        print(f"\nResults saved to {args.output}")

    if args.trades_csv:
        result.save_trades_csv(args.trades_csv)
        logger.info(f"Trades saved to {args.trades_csv}")
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

    # Setup logging with file output if requested
    setup_logging(log_file=args.log_file, verbose=args.verbose)

    try:
        if args.sweep:
            run_parameter_sweep(args)
        elif args.analyze_gaps:
            run_gap_analysis(args)
        else:
            run_standard_backtest(args)

        logger.info("")
        logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        logger.warning("Backtest interrupted by user")
        print("\nBacktest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

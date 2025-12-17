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


def setup_logging(log_file: str = None, verbose: bool = False) -> None:
    """Configure logging with optional file output.

    Args:
        log_file: Path to log file. If None, logs only to console.
        verbose: If True, sets DEBUG level logging.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Detailed format for file logging
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Simpler format for console
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler (10MB max, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Always capture DEBUG in file
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")


def log_trade_detail(trade, trade_num: int, total_trades: int) -> None:
    """Log detailed trade information."""
    logger.info("=" * 70)
    logger.info(f"TRADE {trade_num}/{total_trades}: {trade.symbol}")
    logger.info("=" * 70)
    logger.info(f"  Trade Date:    {trade.trade_date}")
    logger.info(f"  Direction:     {trade.direction}")
    logger.info(f"  Entry Time:    {trade.entry_time}")
    logger.info(f"  Entry Price:   ${trade.entry_price:.4f}")
    logger.info(f"  Shares:        {trade.shares}")
    logger.info(f"  Position Size: ${trade.entry_price * trade.shares:,.2f}")
    if hasattr(trade, 'stop_price') and trade.stop_price:
        logger.info(f"  Stop Price:    ${trade.stop_price:.4f}")
    logger.info(f"  Exit Time:     {trade.exit_time}")
    logger.info(f"  Exit Price:    ${trade.exit_price:.4f}")
    logger.info(f"  Exit Reason:   {trade.exit_reason}")
    logger.info(f"  P&L:           ${trade.pnl:+,.2f} ({trade.pnl_pct:+.2f}%)")
    logger.info(f"  Gap %:         {trade.gap_pct:+.2f}%")
    if hasattr(trade, 'atr') and trade.atr:
        logger.info(f"  ATR:           ${trade.atr:.4f}")
    if hasattr(trade, 'risk_tier') and trade.risk_tier:
        logger.info(f"  Risk Tier:     {trade.risk_tier}")
    if hasattr(trade, 'hold_duration_minutes') and trade.hold_duration_minutes:
        logger.info(f"  Hold Duration: {trade.hold_duration_minutes} minutes")
    logger.info("-" * 70)


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
        help='Log file path for detailed logging (e.g., backtest.log)'
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
    logger.info("=" * 70)
    logger.info("GAP TRADING BACKTEST")
    logger.info("=" * 70)

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    # Log configuration
    logger.info("CONFIGURATION:")
    logger.info(f"  Period:              {start_date} to {end_date}")
    logger.info(f"  Initial Capital:     ${args.capital:,.2f}")
    logger.info(f"  Symbols:             {args.symbols or 'Default universe'}")
    logger.info(f"  Min Gap %:           {args.min_gap}%")
    logger.info(f"  Max Gap %:           {args.max_gap}%")
    logger.info(f"  Confirmation:        {args.confirmation_minutes} minutes after open")
    logger.info(f"  Stop ATR Multiplier: {args.stop_multiplier}x")
    logger.info(f"  Risk Tiers:          {'Enabled' if not args.no_risk_tiers else 'Disabled'}")
    logger.info(f"  Minute Data:         {'Enabled' if not args.no_minute_data else 'Disabled'}")
    logger.info("-" * 70)

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

    # Track daily stats for logging
    daily_stats = {}

    def progress(trade_date, day_num, total):
        if args.verbose or day_num % 10 == 0:
            logger.info(f"Processing {trade_date} ({day_num}/{total})")

    logger.info("Starting backtest execution...")
    result = engine.run(args.symbols, progress_callback=progress)

    # Log detailed trade information
    if args.log_trades and result.trades:
        logger.info("")
        logger.info("=" * 70)
        logger.info("DETAILED TRADE LOG")
        logger.info("=" * 70)
        for i, trade in enumerate(result.trades, 1):
            log_trade_detail(trade, i, len(result.trades))

    # Log daily summary
    if result.trades:
        logger.info("")
        logger.info("=" * 70)
        logger.info("DAILY SUMMARY")
        logger.info("=" * 70)
        trades_by_date = {}
        for trade in result.trades:
            trade_date = trade.entry_time.date() if hasattr(trade.entry_time, 'date') else trade.entry_time
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
            logger.info(
                f"  {trade_date}: {len(day_trades)} trades, "
                f"W:{winners}/L:{losers}, "
                f"Day P&L: ${day_pnl:+,.2f}, "
                f"Cumulative: ${cumulative_pnl:+,.2f}"
            )

    # Log rejection analysis
    if result.rejection_analysis:
        logger.info("")
        logger.info("=" * 70)
        logger.info("SIGNAL REJECTION ANALYSIS")
        logger.info("=" * 70)
        total_rejected = sum(result.rejection_analysis.values())
        for reason, count in sorted(result.rejection_analysis.items(), key=lambda x: -x[1]):
            pct = (count / total_rejected * 100) if total_rejected > 0 else 0
            logger.info(f"  {reason}: {count} ({pct:.1f}%)")

    # Log final metrics
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    if result.metrics:
        m = result.metrics
        logger.info(f"  Total Return:    {m.total_return_pct:+.2f}%")
        logger.info(f"  Final Equity:    ${m.final_equity:,.2f}")
        logger.info(f"  Total Trades:    {m.total_trades}")
        logger.info(f"  Win Rate:        {m.win_rate:.1f}%")
        logger.info(f"  Profit Factor:   {m.profit_factor:.2f}")
        logger.info(f"  Sharpe Ratio:    {m.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown:    {m.max_drawdown_pct:.2f}%")
        logger.info(f"  Avg Trade:       ${m.avg_trade:,.2f}")
        logger.info(f"  Largest Win:     ${m.largest_win:,.2f}")
        logger.info(f"  Largest Loss:    ${m.largest_loss:,.2f}")

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

    # Log startup info
    logger.info("=" * 70)
    logger.info("GAP TRADING BACKTEST SYSTEM")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    try:
        if args.sweep:
            run_parameter_sweep(args)
        elif args.analyze_gaps:
            run_gap_analysis(args)
        else:
            run_standard_backtest(args)

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Backtest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.warning("Backtest interrupted by user")
        print("\nBacktest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""CLI Runner for Gap Trading Backtest.

Enhanced command-line interface for running backtests with various
output formats and configuration options.

Usage:
    # Run 30-day backtest with defaults
    python run_backtest.py --days 30

    # Run with specific date range
    python run_backtest.py --start 2025-10-01 --end 2025-12-01

    # Run with specific symbols and export results
    python run_backtest.py --days 30 --symbols SPY QQQ AAPL --export csv

    # Run with custom parameters
    python run_backtest.py --days 30 --capital 50000 --risk 0.5 --gap-min 2.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from strategy_builder.strategies.gap_trading.backtest import (
    BacktestConfig,
    GapTradingBacktester,
    BacktestResults,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Gap Trading Strategy Backtester',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --days 30
  %(prog)s --start 2025-10-01 --end 2025-12-01
  %(prog)s --days 30 --symbols SPY QQQ AAPL --export csv
  %(prog)s --days 30 --capital 50000 --risk 0.5
        """
    )

    # Date range options
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        '--days', type=int, default=30,
        help='Number of days to backtest (default: 30)'
    )
    date_group.add_argument(
        '--start', type=str,
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end', type=str,
        help='End date (YYYY-MM-DD, default: today)'
    )

    # Symbol options
    parser.add_argument(
        '--symbols', type=str, nargs='+',
        help='Symbols to backtest (default: universe or SPY,QQQ,etc)'
    )
    parser.add_argument(
        '--universe', action='store_true',
        help='Use database universe (requires DB connection)'
    )

    # Capital and risk options
    parser.add_argument(
        '--capital', type=float, default=100000,
        help='Initial capital (default: 100000)'
    )
    parser.add_argument(
        '--risk', type=float, default=1.0,
        help='Risk per trade %% (default: 1.0)'
    )
    parser.add_argument(
        '--max-positions', type=int, default=5,
        help='Maximum concurrent positions (default: 5)'
    )

    # Gap detection options
    parser.add_argument(
        '--gap-min', type=float, default=1.5,
        help='Minimum gap %% threshold (default: 1.5)'
    )
    parser.add_argument(
        '--gap-max', type=float, default=10.0,
        help='Maximum gap %% threshold (default: 10.0)'
    )
    parser.add_argument(
        '--stop-atr', type=float, default=1.5,
        help='ATR multiplier for stops (default: 1.5)'
    )

    # Cost options
    parser.add_argument(
        '--commission', type=float, default=0.0,
        help='Commission per trade (default: 0.0)'
    )
    parser.add_argument(
        '--slippage', type=float, default=0.05,
        help='Slippage %% (default: 0.05)'
    )

    # Output options
    parser.add_argument(
        '--export', type=str, choices=['json', 'csv', 'both'],
        help='Export results to file(s)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='.',
        help='Output directory for exports'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress detailed output'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose logging'
    )

    # Feature flags
    parser.add_argument(
        '--no-risk-tiers', action='store_true',
        help='Disable risk tier position sizing'
    )

    return parser.parse_args()


def run_backtest(args) -> BacktestResults:
    """Run the backtest with given arguments."""
    # Determine date range
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = (datetime.strptime(args.end, '%Y-%m-%d').date()
                    if args.end else datetime.now().date())
    else:
        end_date = (datetime.strptime(args.end, '%Y-%m-%d').date()
                    if args.end else datetime.now().date())
        start_date = end_date - timedelta(days=args.days)

    # Create config
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        risk_per_trade_pct=args.risk,
        min_gap_pct=args.gap_min,
        max_gap_pct=args.gap_max,
        stop_atr_multiplier=args.stop_atr,
        max_positions=args.max_positions,
        commission=args.commission,
        slippage_pct=args.slippage,
        use_risk_tiers=not args.no_risk_tiers,
    )

    # Get database connection if using universe
    db_conn = None
    if args.universe:
        try:
            import psycopg2
            db_conn = psycopg2.connect(
                host='localhost',
                port=5432,
                database='timescaledb',
                user='postgres',
                password='password'
            )
            logger.info("Connected to database for universe data")
        except Exception as e:
            logger.warning(f"Could not connect to database: {e}")
            logger.info("Using default symbols instead")

    # Create backtester
    backtester = GapTradingBacktester(config, db_conn=db_conn)

    # Run backtest
    results = backtester.run(symbols=args.symbols)

    # Close DB connection
    if db_conn:
        db_conn.close()

    return results


def export_results(results: BacktestResults, args):
    """Export results to file(s)."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'backtest_{timestamp}'

    if args.export in ['json', 'both']:
        # Export summary as JSON
        json_path = output_dir / f'{base_name}_summary.json'
        with open(json_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Exported summary to: {json_path}")

        # Export trades as JSON
        trades_json_path = output_dir / f'{base_name}_trades.json'
        with open(trades_json_path, 'w') as f:
            trades_data = [t.to_dict() for t in results.trades]
            json.dump(trades_data, f, indent=2)
        logger.info(f"Exported trades to: {trades_json_path}")

    if args.export in ['csv', 'both']:
        try:
            import pandas as pd

            # Export trades as CSV
            csv_path = output_dir / f'{base_name}_trades.csv'
            trades_data = [t.to_dict() for t in results.trades]
            if trades_data:
                df = pd.DataFrame(trades_data)
                df.to_csv(csv_path, index=False)
                logger.info(f"Exported trades to: {csv_path}")

            # Export equity curve as CSV
            equity_path = output_dir / f'{base_name}_equity.csv'
            if results.equity_curve:
                equity_df = pd.DataFrame(results.equity_curve)
                equity_df.to_csv(equity_path, index=False)
                logger.info(f"Exported equity curve to: {equity_path}")

        except ImportError:
            logger.warning("pandas not installed, skipping CSV export")


def print_detailed_results(results: BacktestResults):
    """Print detailed results with trade breakdown."""
    print(results.summary())

    if results.trades:
        print("\n" + "=" * 60)
        print("TRADE BREAKDOWN BY DIRECTION")
        print("=" * 60)

        # By direction
        long_trades = [t for t in results.trades if t.direction == 'LONG']
        short_trades = [t for t in results.trades if t.direction == 'SHORT']

        print(f"\nLONG Trades: {len(long_trades)}")
        if long_trades:
            long_pnl = sum(t.pnl for t in long_trades)
            long_winners = len([t for t in long_trades if t.pnl > 0])
            print(f"  P&L: ${long_pnl:,.2f}")
            print(f"  Win Rate: {long_winners/len(long_trades)*100:.1f}%")

        print(f"\nSHORT Trades: {len(short_trades)}")
        if short_trades:
            short_pnl = sum(t.pnl for t in short_trades)
            short_winners = len([t for t in short_trades if t.pnl > 0])
            print(f"  P&L: ${short_pnl:,.2f}")
            print(f"  Win Rate: {short_winners/len(short_trades)*100:.1f}%")

        # By risk tier
        print("\n" + "-" * 40)
        print("BY RISK TIER")
        print("-" * 40)

        for tier in ['LOW', 'MEDIUM', 'HIGH']:
            tier_trades = [t for t in results.trades if t.risk_tier == tier]
            if tier_trades:
                tier_pnl = sum(t.pnl for t in tier_trades)
                tier_winners = len([t for t in tier_trades if t.pnl > 0])
                print(f"\n{tier}: {len(tier_trades)} trades")
                print(f"  P&L: ${tier_pnl:,.2f}")
                print(f"  Win Rate: {tier_winners/len(tier_trades)*100:.1f}%")

        # By exit reason
        print("\n" + "-" * 40)
        print("BY EXIT REASON")
        print("-" * 40)

        stop_exits = [t for t in results.trades if t.exit_reason == 'stop_loss']
        eod_exits = [t for t in results.trades if t.exit_reason == 'eod']

        print(f"\nStop-Loss Exits: {len(stop_exits)}")
        if stop_exits:
            stop_pnl = sum(t.pnl for t in stop_exits)
            print(f"  P&L: ${stop_pnl:,.2f}")

        print(f"\nEOD Exits: {len(eod_exits)}")
        if eod_exits:
            eod_pnl = sum(t.pnl for t in eod_exits)
            eod_winners = len([t for t in eod_exits if t.pnl > 0])
            print(f"  P&L: ${eod_pnl:,.2f}")
            print(f"  Win Rate: {eod_winners/len(eod_exits)*100:.1f}%")

        # Top/bottom trades
        print("\n" + "-" * 40)
        print("TOP 5 TRADES")
        print("-" * 40)

        sorted_trades = sorted(results.trades, key=lambda t: t.pnl, reverse=True)
        for trade in sorted_trades[:5]:
            print(f"  {trade.trade_date} {trade.symbol} {trade.direction}: "
                  f"${trade.pnl:,.2f} ({trade.pnl_pct:+.1f}%)")

        print("\n" + "-" * 40)
        print("BOTTOM 5 TRADES")
        print("-" * 40)

        for trade in sorted_trades[-5:]:
            print(f"  {trade.trade_date} {trade.symbol} {trade.direction}: "
                  f"${trade.pnl:,.2f} ({trade.pnl_pct:+.1f}%)")


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    print("=" * 60)
    print("GAP TRADING BACKTEST")
    print("=" * 60)

    # Run backtest
    results = run_backtest(args)

    # Output results
    if not args.quiet:
        print_detailed_results(results)
    else:
        print(results.summary())

    # Export if requested
    if args.export:
        export_results(results, args)

    # Return exit code based on profitability
    if results.total_return > 0:
        print("\n[PASS] Strategy was profitable in backtest")
        return 0
    else:
        print("\n[WARN] Strategy was not profitable in backtest")
        return 1


if __name__ == '__main__':
    sys.exit(main())
